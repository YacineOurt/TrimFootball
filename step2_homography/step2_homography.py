"""
Step 2: Per-frame homography via pitch keypoint detection
- Uses a YOLOv8-pose model to detect 32 pitch keypoints per frame
- Computes homography per frame using cv2.findHomography (RANSAC)
- Projects player/ball positions from tracking data onto pitch coordinates
- Falls back to previous frame's homography when detection is insufficient
"""

import cv2
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# --- Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

KEYPOINT_MODEL = str(PROJECT_DIR / "models/pitch_keypoints/football-pitch-detection.pt")

# --- Pitch dimensions (meters) ---
PITCH_L, PITCH_W = 105.0, 68.0
PEN_D, PEN_W = 16.5, 40.32
GA_D, GA_W = 5.5, 18.32
CR = 9.15
PENALTY_SPOT = 11.0

pen_y1 = (PITCH_W - PEN_W) / 2   # 13.84
pen_y2 = (PITCH_W + PEN_W) / 2   # 54.16
ga_y1 = (PITCH_W - GA_W) / 2     # 24.84
ga_y2 = (PITCH_W + GA_W) / 2     # 43.16
mid_x = PITCH_L / 2              # 52.5
mid_y = PITCH_W / 2              # 34.0

# 32 keypoints: index → (x_meters, y_meters) on the pitch
# x = 0 → left goal line, x = 105 → right goal line
# y = 0 → far touchline, y = 68 → near touchline
KEYPOINT_PITCH_COORDS = [
    (0.0, 0.0),                          # 0:  corner far-left
    (0.0, pen_y1),                        # 1:  left penalty area far
    (0.0, ga_y1),                         # 2:  left goal post far
    (0.0, ga_y2),                         # 3:  left goal post near
    (0.0, pen_y2),                        # 4:  left penalty area near
    (0.0, PITCH_W),                       # 5:  corner near-left
    (GA_D, ga_y1),                        # 6:  left goal area far corner
    (GA_D, ga_y2),                        # 7:  left goal area near corner
    (PENALTY_SPOT, mid_y),                # 8:  left penalty spot
    (PEN_D, pen_y1),                      # 9:  left pen area far-outer
    (PEN_D, ga_y1),                       # 10: left pen area far-inner
    (PEN_D, ga_y2),                       # 11: left pen area near-inner
    (PEN_D, pen_y2),                      # 12: left pen area near-outer
    (mid_x, 0.0),                         # 13: halfway × far touchline
    (mid_x, mid_y - CR),                  # 14: center circle far
    (mid_x, mid_y + CR),                  # 15: center circle near
    (mid_x, PITCH_W),                     # 16: halfway × near touchline
    (PITCH_L - PEN_D, pen_y1),            # 17: right pen area far-outer
    (PITCH_L - PEN_D, ga_y1),             # 18: right pen area far-inner
    (PITCH_L - PEN_D, ga_y2),             # 19: right pen area near-inner
    (PITCH_L - PEN_D, pen_y2),            # 20: right pen area near-outer
    (PITCH_L - PENALTY_SPOT, mid_y),      # 21: right penalty spot
    (PITCH_L - GA_D, ga_y1),              # 22: right goal area far corner
    (PITCH_L - GA_D, ga_y2),              # 23: right goal area near corner
    (PITCH_L, 0.0),                       # 24: corner far-right
    (PITCH_L, pen_y1),                    # 25: right penalty area far
    (PITCH_L, ga_y1),                     # 26: right goal post far
    (PITCH_L, ga_y2),                     # 27: right goal post near
    (PITCH_L, pen_y2),                    # 28: right penalty area near
    (PITCH_L, PITCH_W),                   # 29: corner near-right
    (mid_x - CR, mid_y),                  # 30: center circle left
    (mid_x + CR, mid_y),                  # 31: center circle right
]

MIN_KEYPOINTS = 4
CONFIDENCE_THRESHOLD = 0.5


def detect_keypoints(model, frame):
    """Detect pitch keypoints in a frame. Returns (pixel_pts, pitch_pts) arrays."""
    results = model(frame, verbose=False)

    if results[0].keypoints is None:
        return None, None

    kps = results[0].keypoints
    xy = kps.xy[0].cpu().numpy()
    conf = kps.conf[0].cpu().numpy() if kps.conf is not None else np.ones(len(xy))

    pixel_pts = []
    pitch_pts = []

    for i, (pt, c) in enumerate(zip(xy, conf)):
        if c >= CONFIDENCE_THRESHOLD and i < len(KEYPOINT_PITCH_COORDS):
            px, py = pt[0], pt[1]
            if px > 0 and py > 0:  # valid detection
                pixel_pts.append([px, py])
                pitch_pts.append(KEYPOINT_PITCH_COORDS[i])

    if len(pixel_pts) < MIN_KEYPOINTS:
        return None, None

    return np.array(pixel_pts, dtype=np.float32), np.array(pitch_pts, dtype=np.float32)


def compute_homography(pixel_pts, pitch_pts):
    """Compute homography from pixel to pitch coordinates using RANSAC."""
    H, mask = cv2.findHomography(pixel_pts, pitch_pts, cv2.RANSAC, 5.0)
    if H is None:
        return None
    # Sanity check: project a few points and verify they're on the pitch
    projected = cv2.perspectiveTransform(pixel_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
    in_bounds = sum(1 for p in projected if -10 <= p[0] <= PITCH_L + 10 and -10 <= p[1] <= PITCH_W + 10)
    if in_bounds < len(projected) * 0.5:
        return None
    return H


def project_point(H, px, py):
    """Project a pixel point to pitch coordinates using homography H."""
    pt = np.array([px, py, 1.0])
    result = H @ pt
    if abs(result[2]) < 1e-10:
        return None, None
    return round(float(result[0] / result[2]), 1), round(float(result[1] / result[2]), 1)


def run(video_path=None, tracking_path=None, output_dir=None, progress_callback=None):
    video_path = video_path or str(PROJECT_DIR / "video2.mp4")
    tracking_path = tracking_path or str(PROJECT_DIR / "tracking_data.json")
    output_dir = Path(output_dir) if output_dir else SCRIPT_DIR

    print("Loading pitch keypoint model...")
    model = YOLO(KEYPOINT_MODEL)

    print("Loading tracking data...")
    with open(tracking_path) as f:
        tracking = json.load(f)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {total_frames} frames")

    last_good_H = None
    homography_stats = {"good": 0, "fallback": 0, "failed": 0}

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Detect keypoints and compute homography
        pixel_pts, pitch_pts = detect_keypoints(model, frame)

        H = None
        if pixel_pts is not None:
            H = compute_homography(pixel_pts, pitch_pts)

        if H is not None:
            last_good_H = H
            homography_stats["good"] += 1
        elif last_good_H is not None:
            H = last_good_H
            homography_stats["fallback"] += 1
        else:
            homography_stats["failed"] += 1

        # Project player positions
        fd = tracking["frames"][frame_idx]
        for p in fd["players"]:
            x1, y1, x2, y2 = p["bbox"]
            feet_x = (x1 + x2) / 2
            feet_y = y2  # bottom of bbox = feet

            if H is not None:
                px, py = project_point(H, feet_x, feet_y)
                p["pitch_x"] = px
                p["pitch_y"] = py
            else:
                p["pitch_x"] = None
                p["pitch_y"] = None

        # Project ball
        if fd.get("ball"):
            bx = (fd["ball"]["bbox"][0] + fd["ball"]["bbox"][2]) / 2
            by = (fd["ball"]["bbox"][1] + fd["ball"]["bbox"][3]) / 2
            if H is not None:
                px, py = project_point(H, bx, by)
                fd["ball"]["pitch_x"] = px
                fd["ball"]["pitch_y"] = py

        if frame_idx % 50 == 0:
            n_kp = len(pixel_pts) if pixel_pts is not None else 0
            status = "OK" if pixel_pts is not None and compute_homography(pixel_pts, pitch_pts) is not None else "fallback"
            print(f"  Frame {frame_idx}/{total_frames}: {n_kp} keypoints — {status}")

        if progress_callback and frame_idx % 20 == 0:
            progress_callback("homography", frame_idx / total_frames)

    cap.release()

    # Save tracking data
    with open(tracking_path, "w") as f:
        json.dump(tracking, f)

    # Save debug visualization for a few frames
    _save_debug_frames(video_path, tracking, output_dir)

    if progress_callback:
        progress_callback("homography", 1.0)

    total = sum(homography_stats.values())
    print(f"\nDone! Homography stats:")
    print(f"  Good:     {homography_stats['good']}/{total} ({100*homography_stats['good']/max(total,1):.0f}%)")
    print(f"  Fallback: {homography_stats['fallback']}/{total}")
    print(f"  Failed:   {homography_stats['failed']}/{total}")
    print(f"  Tracking: {tracking_path}")


def _save_debug_frames(video_path, tracking, output_dir):
    """Save a few debug images showing projected positions on 2D pitch."""
    output_dir = Path(output_dir)
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    debug_frames = [0, total // 4, total // 2, 3 * total // 4, total - 1]

    for fidx in debug_frames:
        if fidx >= len(tracking["frames"]):
            continue
        fd = tracking["frames"][fidx]
        img = _draw_2d_pitch(fd)
        cv2.imwrite(str(output_dir / f"pitch_2d_frame_{fidx}.jpg"), img)

    cap.release()
    print(f"  Debug images saved to {output_dir}/pitch_2d_frame_*.jpg")


def _draw_2d_pitch(frame_data):
    """Draw players on a 2D pitch diagram."""
    scale = 10
    pw, ph = int(PITCH_L * scale), int(PITCH_W * scale)
    margin = 30
    img = np.ones((ph + 2 * margin, pw + 2 * margin, 3), dtype=np.uint8) * 40

    def m2px(mx, my):
        return int(mx * scale + margin), int(my * scale + margin)

    # Pitch surface
    cv2.rectangle(img, m2px(0, 0), m2px(PITCH_L, PITCH_W), (34, 120, 34), -1)
    WHITE = (255, 255, 255)

    # Outline
    cv2.rectangle(img, m2px(0, 0), m2px(PITCH_L, PITCH_W), WHITE, 1)
    # Halfway
    cv2.line(img, m2px(mid_x, 0), m2px(mid_x, PITCH_W), WHITE, 1)
    # Center circle
    cv2.circle(img, m2px(mid_x, mid_y), int(CR * scale), WHITE, 1)
    # Penalty areas
    cv2.rectangle(img, m2px(0, pen_y1), m2px(PEN_D, pen_y2), WHITE, 1)
    cv2.rectangle(img, m2px(PITCH_L - PEN_D, pen_y1), m2px(PITCH_L, pen_y2), WHITE, 1)
    # Goal areas
    cv2.rectangle(img, m2px(0, ga_y1), m2px(GA_D, ga_y2), WHITE, 1)
    cv2.rectangle(img, m2px(PITCH_L - GA_D, ga_y1), m2px(PITCH_L, ga_y2), WHITE, 1)

    TEAM_COLORS = {0: (255, 100, 50), 1: (50, 100, 255), -1: (200, 200, 200)}

    for p in frame_data["players"]:
        if p.get("pitch_x") is None:
            continue
        px, py = p["pitch_x"], p["pitch_y"]
        if 0 <= px <= PITCH_L and 0 <= py <= PITCH_W:
            color = TEAM_COLORS.get(p.get("team", -1), (200, 200, 200))
            sx, sy = m2px(px, py)
            cv2.circle(img, (sx, sy), 6, color, -1)
            cv2.circle(img, (sx, sy), 6, WHITE, 1)

    if frame_data.get("ball") and frame_data["ball"].get("pitch_x"):
        bx, by = frame_data["ball"]["pitch_x"], frame_data["ball"]["pitch_y"]
        if 0 <= bx <= PITCH_L and 0 <= by <= PITCH_W:
            sx, sy = m2px(bx, by)
            cv2.circle(img, (sx, sy), 4, (0, 255, 255), -1)

    return img


if __name__ == "__main__":
    run()
