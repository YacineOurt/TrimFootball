"""
Generate a verification case: 3 images for a given frame.
  1. Original Veo frame
  2. Same frame with detection overlays (bboxes, teams, IDs)
  3. 2D pitch tactical view

Usage: python generate_case.py [frame_number] [case_name]
  Default: frame 270, case "cas1"
"""

import cv2
import json
import sys
import numpy as np
from pathlib import Path

# --- Config ---
VIDEO = "video2.mp4"
TRACKING = "tracking_data.json"
HOMOGRAPHY = "step2_homography/homography.npy"

PITCH_L, PITCH_W = 105.0, 68.0
PEN_D, PEN_W = 16.5, 40.32
GA_D, GA_W = 5.5, 18.32
CR = 9.15

TEAM_COLORS = {0: (255, 100, 50), 1: (50, 100, 255), -1: (200, 200, 200)}
BALL_COLOR = (0, 255, 255)


def remove_duplicate_detections(players, iou_thresh=0.5):
    """Remove overlapping detections (e.g. same person as 'player' and 'referee').
    Keep the one with highest confidence."""
    if not players:
        return players

    # Sort by confidence descending
    players = sorted(players, key=lambda p: p["conf"], reverse=True)
    keep = []

    for p in players:
        is_dup = False
        bx1, by1, bx2, by2 = p["bbox"]
        for k in keep:
            kx1, ky1, kx2, ky2 = k["bbox"]
            # Compute IoU
            ix1 = max(bx1, kx1)
            iy1 = max(by1, ky1)
            ix2 = min(bx2, kx2)
            iy2 = min(by2, ky2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            area_p = (bx2 - bx1) * (by2 - by1)
            area_k = (kx2 - kx1) * (ky2 - ky1)
            union = area_p + area_k - inter
            iou = inter / union if union > 0 else 0
            if iou > iou_thresh:
                is_dup = True
                break
        if not is_dup:
            keep.append(p)

    return keep


def draw_detection_frame(frame, players, ball):
    """Draw bboxes, team colors, IDs on frame."""
    vis = frame.copy()

    for p in players:
        x1, y1, x2, y2 = map(int, p["bbox"])
        team = p.get("team", -1)
        color = TEAM_COLORS.get(team, (200, 200, 200))

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        label = f"#{p['track_id']}"
        if p["class"] == "goalkeeper":
            label += " GK"
        elif p["class"] == "referee":
            label += " REF"
        team_label = f"T{team}" if team >= 0 else "?"
        label += f" {team_label}"

        # Background for text
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(vis, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    if ball:
        x1, y1, x2, y2 = map(int, ball["bbox"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), BALL_COLOR, 2)
        cv2.putText(vis, "BALL", (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, BALL_COLOR, 1)

    return vis


def draw_2d_pitch(players, ball):
    """Draw players on a 2D pitch diagram."""
    scale = 10
    pw, ph = int(PITCH_L * scale), int(PITCH_W * scale)
    margin = 30
    img = np.ones((ph + 2 * margin, pw + 2 * margin, 3), dtype=np.uint8) * 40

    def m2px(mx, my):
        return int(mx * scale + margin), int(my * scale + margin)

    # Pitch surface
    cv2.rectangle(img, m2px(0, 0), m2px(PITCH_L, PITCH_W), (34, 120, 34), -1)

    # Lines
    WHITE = (255, 255, 255)
    # Outline
    cv2.rectangle(img, m2px(0, 0), m2px(PITCH_L, PITCH_W), WHITE, 1)
    # Halfway
    cv2.line(img, m2px(PITCH_L / 2, 0), m2px(PITCH_L / 2, PITCH_W), WHITE, 1)
    # Center circle
    cv2.circle(img, m2px(PITCH_L / 2, PITCH_W / 2), int(CR * scale), WHITE, 1)
    cv2.circle(img, m2px(PITCH_L / 2, PITCH_W / 2), 3, WHITE, -1)
    # Penalty areas
    py1, py2 = (PITCH_W - PEN_W) / 2, (PITCH_W + PEN_W) / 2
    cv2.rectangle(img, m2px(0, py1), m2px(PEN_D, py2), WHITE, 1)
    cv2.rectangle(img, m2px(PITCH_L - PEN_D, py1), m2px(PITCH_L, py2), WHITE, 1)
    # Goal areas
    gy1, gy2 = (PITCH_W - GA_W) / 2, (PITCH_W + GA_W) / 2
    cv2.rectangle(img, m2px(0, gy1), m2px(GA_D, gy2), WHITE, 1)
    cv2.rectangle(img, m2px(PITCH_L - GA_D, gy1), m2px(PITCH_L, gy2), WHITE, 1)

    # Players
    for p in players:
        px, py = p.get("pitch_x"), p.get("pitch_y")
        if px is None or py is None:
            continue
        if not (0 <= px <= PITCH_L and 0 <= py <= PITCH_W):
            continue

        color = TEAM_COLORS.get(p.get("team", -1), (200, 200, 200))
        sx, sy = m2px(px, py)
        cv2.circle(img, (sx, sy), 6, color, -1)
        cv2.circle(img, (sx, sy), 6, WHITE, 1)

        label = f"#{p['track_id']}"
        if p["class"] == "goalkeeper":
            label = "GK"
        elif p["class"] == "referee":
            label = "REF"
        cv2.putText(img, label, (sx + 8, sy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, WHITE, 1)

    # Ball
    if ball and ball.get("pitch_x") is not None:
        bx, by = ball["pitch_x"], ball["pitch_y"]
        if 0 <= bx <= PITCH_L and 0 <= by <= PITCH_W:
            sx, sy = m2px(bx, by)
            cv2.circle(img, (sx, sy), 5, BALL_COLOR, -1)
            cv2.putText(img, "ball", (sx + 7, sy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, BALL_COLOR, 1)

    return img


def main():
    frame_num = int(sys.argv[1]) if len(sys.argv) > 1 else 270
    case_name = sys.argv[2] if len(sys.argv) > 2 else "cas1"
    out_dir = Path(case_name)
    out_dir.mkdir(exist_ok=True)

    # Load tracking data
    with open(TRACKING) as f:
        tracking = json.load(f)

    # Extract frame from video
    cap = cv2.VideoCapture(VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"ERROR: Could not read frame {frame_num}")
        return

    # Get frame data and deduplicate
    fd = tracking["frames"][frame_num]
    players = remove_duplicate_detections(fd["players"])
    ball = fd.get("ball")

    removed = len(fd["players"]) - len(players)
    print(f"Frame {frame_num}: {len(fd['players'])} detections -> {len(players)} after dedup ({removed} removed)")

    # Count per team
    for t in [0, 1, -1]:
        n = sum(1 for p in players if p.get("team") == t)
        label = {0: "Team A", 1: "Team B", -1: "Unknown"}[t]
        if n > 0:
            print(f"  {label}: {n} players")

    # 1. Original frame
    cv2.imwrite(str(out_dir / "1_original.jpg"), frame)

    # 2. Detection overlay
    det_frame = draw_detection_frame(frame, players, ball)
    cv2.imwrite(str(out_dir / "2_detection.jpg"), det_frame)

    # 3. 2D pitch
    pitch_img = draw_2d_pitch(players, ball)
    cv2.imwrite(str(out_dir / "3_pitch_2d.jpg"), pitch_img)

    print(f"\nSaved in {out_dir}/:")
    print(f"  1_original.jpg   - Veo frame")
    print(f"  2_detection.jpg  - Détections + équipes")
    print(f"  3_pitch_2d.jpg   - Vue tactique 2D")


if __name__ == "__main__":
    main()
