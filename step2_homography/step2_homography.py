"""
Step 2: Homography estimation via optimization
- Detects white pitch lines in the Veo frame
- Optimizes a homography to align a pitch model with detected lines
- Projects player positions from tracking data onto a 2D pitch view
"""

import cv2
import json
import numpy as np
from scipy.optimize import minimize

# --- Pitch model (meters) ---
PITCH_L = 105.0
PITCH_W = 68.0
HALF = 52.5
PEN_D = 16.5
PEN_W = 40.32
GOAL_W = 7.32
GA_D = 5.5
GA_W = 18.32
CR = 9.15  # center circle radius

def get_pitch_lines():
    """Return list of (start, end) pitch line segments in meters."""
    lines = []
    # Touchlines
    lines.append(((0, 0), (PITCH_L, 0)))          # far
    lines.append(((0, PITCH_W), (PITCH_L, PITCH_W)))  # near
    # Goal lines
    lines.append(((0, 0), (0, PITCH_W)))
    lines.append(((PITCH_L, 0), (PITCH_L, PITCH_W)))
    # Halfway
    lines.append(((HALF, 0), (HALF, PITCH_W)))
    # Left penalty area
    py1, py2 = (PITCH_W - PEN_W) / 2, (PITCH_W + PEN_W) / 2
    lines += [((0, py1), (PEN_D, py1)), ((0, py2), (PEN_D, py2)),
              ((PEN_D, py1), (PEN_D, py2))]
    # Right penalty area
    lines += [((PITCH_L, py1), (PITCH_L - PEN_D, py1)),
              ((PITCH_L, py2), (PITCH_L - PEN_D, py2)),
              ((PITCH_L - PEN_D, py1), (PITCH_L - PEN_D, py2))]
    # Left goal area
    gy1, gy2 = (PITCH_W - GA_W) / 2, (PITCH_W + GA_W) / 2
    lines += [((0, gy1), (GA_D, gy1)), ((0, gy2), (GA_D, gy2)),
              ((GA_D, gy1), (GA_D, gy2))]
    # Right goal area
    lines += [((PITCH_L, gy1), (PITCH_L - GA_D, gy1)),
              ((PITCH_L, gy2), (PITCH_L - GA_D, gy2)),
              ((PITCH_L - GA_D, gy1), (PITCH_L - GA_D, gy2))]
    # Center circle
    angles = np.linspace(0, 2 * np.pi, 64)
    for i in range(len(angles) - 1):
        p1 = (HALF + CR * np.cos(angles[i]), PITCH_W / 2 + CR * np.sin(angles[i]))
        p2 = (HALF + CR * np.cos(angles[i + 1]), PITCH_W / 2 + CR * np.sin(angles[i + 1]))
        lines.append((p1, p2))
    return lines


def params_to_H(params):
    """Convert 8 optimization params to 3x3 homography (h33=1)."""
    H = np.array([
        [params[0], params[1], params[2]],
        [params[3], params[4], params[5]],
        [params[6], params[7], 1.0]
    ])
    return H


def project_pitch_to_image(H_inv, px, py):
    """Project pitch (m) to image (px)."""
    pt = np.array([px, py, 1.0])
    r = H_inv @ pt
    if abs(r[2]) < 1e-10:
        return None
    return r[0] / r[2], r[1] / r[2]


def rasterize_pitch_lines(H_inv, w, h, pitch_lines):
    """Render pitch lines onto an image using H_inv (pitch->image)."""
    mask = np.zeros((h, w), dtype=np.uint8)
    for (x1, y1), (x2, y2) in pitch_lines:
        p1 = project_pitch_to_image(H_inv, x1, y1)
        p2 = project_pitch_to_image(H_inv, x2, y2)
        if p1 is None or p2 is None:
            continue
        ix1, iy1 = int(round(p1[0])), int(round(p1[1]))
        ix2, iy2 = int(round(p2[0])), int(round(p2[1]))
        # Clip to image bounds with margin
        if (ix1 < -500 or ix1 > w + 500 or iy1 < -500 or iy1 > h + 500 or
                ix2 < -500 or ix2 > w + 500 or iy2 < -500 or iy2 > h + 500):
            continue
        cv2.line(mask, (ix1, iy1), (ix2, iy2), 255, 3)
    return mask


def alignment_score(params, white_mask, pitch_lines, w, h):
    """Negative overlap between projected pitch lines and detected white pixels."""
    H = params_to_H(params)
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return 1e6

    projected = rasterize_pitch_lines(H_inv, w, h, pitch_lines)
    # Overlap: count white pixels that fall on projected lines (use int64 to avoid overflow)
    overlap = cv2.bitwise_and(projected, white_mask)
    overlap_count = np.count_nonzero(overlap)
    # Also penalize projected pixels that DON'T hit white (false positives)
    projected_count = np.count_nonzero(projected)
    if projected_count == 0:
        return 1e6
    # Score: maximize overlap, penalize false positives
    precision = overlap_count / max(projected_count, 1)
    recall = overlap_count / max(np.count_nonzero(white_mask), 1)
    if precision + recall == 0:
        return 1e6
    f1 = 2 * precision * recall / (precision + recall)
    return -f1


def detect_white_lines(frame):
    """Detect white pitch line pixels."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 160), (180, 55, 255))

    # Only keep pitch area (green region)
    green = cv2.inRange(hsv, (25, 15, 30), (95, 255, 255))
    # Dilate green to include line pixels surrounded by grass
    green_dilated = cv2.dilate(green, np.ones((25, 25), np.uint8))
    white_on_pitch = cv2.bitwise_and(white, green_dilated)

    # Remove large blobs (snow, advertising) - keep thin structures
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    white_on_pitch = cv2.morphologyEx(white_on_pitch, cv2.MORPH_OPEN, kernel)

    return white_on_pitch


def compute_initial_H():
    """Compute initial homography guess from the most confident correspondences."""
    # Use approximate correspondences based on detected features:
    # The camera is on the near side, slightly left of center, elevated.
    # Known: halfway line at x_pixel≈1245, far touchline at y_pixel≈240
    #
    # Initial guess: 4 corner correspondences of the visible pitch area
    # Adjusted initial guess based on detected features:
    # - Halfway line at x_pixel≈1245 = x_pitch 52.5m
    # - Far touchline at y_pixel≈240
    # - Near touchline extrapolated
    # - Left goal area at x_pixel≈280
    pts_pixel = np.array([
        [250, 248],      # far-left corner (goal line × far touchline)
        [1850, 235],     # far-right corner
        [1920, 1070],    # near-right corner
        [50, 1060],      # near-left corner
    ], dtype=np.float32)

    pts_pitch = np.array([
        [0, 0],
        [105, 0],
        [105, 68],
        [0, 68],
    ], dtype=np.float32)

    H, _ = cv2.findHomography(pts_pixel, pts_pitch)
    return H


def run():
    cap = cv2.VideoCapture("video2.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 270)
    ret, frame = cap.read()
    cap.release()
    h, w = frame.shape[:2]

    print("1. Detecting white pitch lines...")
    white_mask = detect_white_lines(frame)
    cv2.imwrite("white_mask_clean.jpg", white_mask)
    print(f"   White pixels: {(white_mask > 0).sum()}")

    print("2. Computing initial homography guess...")
    H_init = compute_initial_H()
    params_init = [
        H_init[0, 0], H_init[0, 1], H_init[0, 2],
        H_init[1, 0], H_init[1, 1], H_init[1, 2],
        H_init[2, 0], H_init[2, 1],
    ]

    pitch_lines = get_pitch_lines()

    # Verify initial guess
    H_inv_init = np.linalg.inv(H_init)
    score_init = alignment_score(params_init, white_mask, pitch_lines, w, h)
    print(f"   Initial score: {score_init:.0f}")

    # Draw initial guess
    vis_init = frame.copy()
    proj_init = rasterize_pitch_lines(H_inv_init, w, h, pitch_lines)
    vis_init[proj_init > 0] = [0, 255, 0]
    cv2.imwrite("homography_initial.jpg", vis_init)

    print("3. Optimizing homography...")
    # Use Powell method which handles this type of problem better
    result = minimize(
        alignment_score,
        params_init,
        args=(white_mask, pitch_lines, w, h),
        method="Powell",
        options={"maxiter": 10000, "ftol": 1e-6},
    )
    print(f"   Optimization: success={result.success}, score={result.fun:.4f}")
    print(f"   Iterations: {result.nit}")

    # Try Nelder-Mead from the Powell result for fine-tuning
    result2 = minimize(
        alignment_score,
        result.x,
        args=(white_mask, pitch_lines, w, h),
        method="Nelder-Mead",
        options={"maxiter": 10000, "xatol": 1e-10, "fatol": 1e-6, "adaptive": True},
    )
    if result2.fun < result.fun:
        result = result2
        print(f"   Nelder-Mead improved: score={result.fun:.4f}")

    H_opt = params_to_H(result.x)
    H_inv_opt = np.linalg.inv(H_opt)

    # Draw optimized result
    vis_opt = frame.copy()
    proj_opt = rasterize_pitch_lines(H_inv_opt, w, h, pitch_lines)
    vis_opt[proj_opt > 0] = [0, 255, 0]
    cv2.imwrite("homography_optimized.jpg", vis_opt)

    # Save homography
    np.save("homography.npy", H_opt)

    print("4. Projecting tracking data onto 2D pitch...")
    with open("tracking_data.json") as f:
        tracking = json.load(f)

    # Project player positions (center-bottom of bbox = feet position)
    for fd in tracking["frames"]:
        for p in fd["players"]:
            x1, y1, x2, y2 = p["bbox"]
            feet_x = (x1 + x2) / 2
            feet_y = y2  # bottom of bbox = feet
            pt = np.array([feet_x, feet_y, 1.0])
            result_pt = H_opt @ pt
            if abs(result_pt[2]) > 1e-10:
                px = result_pt[0] / result_pt[2]
                py = result_pt[1] / result_pt[2]
                p["pitch_x"] = round(float(px), 1)
                p["pitch_y"] = round(float(py), 1)
            else:
                p["pitch_x"] = None
                p["pitch_y"] = None

        if fd["ball"]:
            bx = (fd["ball"]["bbox"][0] + fd["ball"]["bbox"][2]) / 2
            by = (fd["ball"]["bbox"][1] + fd["ball"]["bbox"][3]) / 2
            pt = np.array([bx, by, 1.0])
            result_pt = H_opt @ pt
            if abs(result_pt[2]) > 1e-10:
                fd["ball"]["pitch_x"] = round(float(result_pt[0] / result_pt[2]), 1)
                fd["ball"]["pitch_y"] = round(float(result_pt[1] / result_pt[2]), 1)

    with open("tracking_data.json", "w") as f:
        json.dump(tracking, f)

    # 5. Draw one frame on 2D pitch
    print("5. Drawing 2D pitch view...")
    draw_2d_pitch(tracking["frames"][270], H_opt)

    print("\nDone! Files:")
    print("  homography_initial.jpg  - initial guess overlay")
    print("  homography_optimized.jpg - optimized overlay")
    print("  pitch_2d_frame270.jpg   - 2D tactical view")
    print("  homography.npy          - saved homography matrix")
    print("  tracking_data.json      - updated with pitch coordinates")


def draw_2d_pitch(frame_data, H):
    """Draw players on a 2D pitch diagram."""
    # Pitch image: 10px per meter
    scale = 10
    pw, ph = int(PITCH_L * scale), int(PITCH_W * scale)
    margin = 30
    img = np.ones((ph + 2 * margin, pw + 2 * margin, 3), dtype=np.uint8) * 40  # dark bg

    def m2px(mx, my):
        return int(mx * scale + margin), int(my * scale + margin)

    # Draw pitch surface
    cv2.rectangle(img, m2px(0, 0), m2px(PITCH_L, PITCH_W), (34, 120, 34), -1)

    # Draw lines
    WHITE = (255, 255, 255)
    for (x1, y1), (x2, y2) in get_pitch_lines():
        cv2.line(img, m2px(x1, y1), m2px(x2, y2), WHITE, 1)

    # Draw center circle
    cv2.circle(img, m2px(HALF, PITCH_W / 2), int(CR * scale), WHITE, 1)

    TEAM_COLORS = {0: (255, 100, 50), 1: (50, 100, 255), -1: (200, 200, 200)}

    # Draw players
    for p in frame_data["players"]:
        if p.get("pitch_x") is None:
            continue
        px, py = p["pitch_x"], p["pitch_y"]
        if 0 <= px <= PITCH_L and 0 <= py <= PITCH_W:
            color = TEAM_COLORS.get(p["team"], (200, 200, 200))
            sx, sy = m2px(px, py)
            cv2.circle(img, (sx, sy), 6, color, -1)
            cv2.circle(img, (sx, sy), 6, WHITE, 1)

            label = f"#{p['track_id']}"
            if p["class"] == "goalkeeper":
                label = "GK"
            elif p["class"] == "referee":
                label = "REF"
            cv2.putText(img, label, (sx + 8, sy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, WHITE, 1)

    # Draw ball
    if frame_data.get("ball") and frame_data["ball"].get("pitch_x"):
        bx, by = frame_data["ball"]["pitch_x"], frame_data["ball"]["pitch_y"]
        if 0 <= bx <= PITCH_L and 0 <= by <= PITCH_W:
            sx, sy = m2px(bx, by)
            cv2.circle(img, (sx, sy), 4, (0, 255, 255), -1)

    cv2.imwrite("pitch_2d_frame270.jpg", img)


if __name__ == "__main__":
    run()
