"""
Step 1: Player tracking + team separation by jersey color
- Runs YOLO + ByteTrack on the Veo video
- Extracts dominant jersey color per player crop
- Clusters players into 2 teams via K-means
- Outputs annotated video + tracking data as JSON
"""

import cv2
import json
import numpy as np
from collections import defaultdict
from pathlib import Path
from sklearn.cluster import KMeans
from ultralytics import YOLO

# --- Config ---
VIDEO_PATH = "video2.mp4"
PLAYERS_MODEL = "models/players_detection/best.pt"
BALL_MODEL = "models/ball_detection/yolo-football-ball-detection.pt"
CONF_THRESHOLD = 0.40
OUTPUT_VIDEO = "output_tracking.mp4"
OUTPUT_DATA = "tracking_data.json"


def get_jersey_color(frame, bbox):
    """Extract dominant color from the upper-body area of a player crop."""
    x1, y1, x2, y2 = map(int, bbox)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    # Take upper 60% of crop (jersey, not shorts/legs)
    h = crop.shape[0]
    jersey_crop = crop[: int(h * 0.6), :]
    if jersey_crop.size == 0:
        return None

    # Convert to HSV for better color clustering
    hsv = cv2.cvtColor(jersey_crop, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3).astype(np.float32)

    # Remove near-green pixels (grass bleed-through)
    green_mask = (pixels[:, 0] > 30) & (pixels[:, 0] < 90) & (pixels[:, 1] > 40)
    pixels = pixels[~green_mask]

    if len(pixels) < 10:
        return None

    # Return mean color of remaining pixels
    return pixels.mean(axis=0)


def classify_teams(track_colors):
    """Cluster tracked players into 2 teams based on jersey color."""
    track_ids = []
    features = []

    for track_id, colors in track_colors.items():
        if len(colors) >= 3:  # need enough samples
            mean_color = np.mean(colors, axis=0)
            track_ids.append(track_id)
            features.append(mean_color)

    if len(features) < 4:
        print(f"WARNING: Only {len(features)} players with enough color data")
        return {}

    features = np.array(features)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)

    team_map = {}
    for tid, label in zip(track_ids, labels):
        team_map[tid] = int(label)

    # Print team sizes
    t0 = sum(1 for v in team_map.values() if v == 0)
    t1 = sum(1 for v in team_map.values() if v == 1)
    print(f"Team clustering: Team A = {t0} players, Team B = {t1} players")

    return team_map


def run_tracking():
    model_players = YOLO(PLAYERS_MODEL)
    model_ball = YOLO(BALL_MODEL)

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {w}x{h} @ {fps:.1f}fps, {total_frames} frames")

    # --- Pass 1: Track + collect jersey colors ---
    print("\n--- Pass 1: Tracking + color extraction ---")
    all_frames_data = []
    track_colors = defaultdict(list)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Player tracking with ByteTrack (custom config for small Veo players)
        results = model_players.track(
            frame, conf=CONF_THRESHOLD, persist=True, tracker="custom_bytetrack.yaml", verbose=False
        )

        # Ball detection (no tracking needed - single object)
        ball_results = model_ball(frame, conf=0.15, verbose=False)

        frame_data = {"frame": frame_idx, "players": [], "ball": None}

        # Process player detections
        if results[0].boxes.id is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                track_id = int(boxes.id[i])
                cls_id = int(boxes.cls[i])
                cls_name = model_players.names[cls_id]
                conf = float(boxes.conf[i])
                bbox = boxes.xyxy[i].tolist()

                # Collect jersey color
                color = get_jersey_color(frame, bbox)
                if color is not None and cls_name == "player":
                    track_colors[track_id].append(color)

                frame_data["players"].append(
                    {
                        "track_id": track_id,
                        "class": cls_name,
                        "conf": round(conf, 2),
                        "bbox": [round(v, 1) for v in bbox],
                    }
                )

        # Process ball
        if len(ball_results[0].boxes) > 0:
            best = ball_results[0].boxes[0]
            frame_data["ball"] = {
                "conf": round(float(best.conf[0]), 2),
                "bbox": [round(v, 1) for v in best.xyxy[0].tolist()],
            }

        all_frames_data.append(frame_data)
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  Frame {frame_idx}/{total_frames}")

    print(f"  Tracked {len(track_colors)} unique player IDs")

    # --- Classify teams ---
    print("\n--- Team classification ---")
    team_map = classify_teams(track_colors)

    # Assign teams to frame data
    for fd in all_frames_data:
        for p in fd["players"]:
            tid = p["track_id"]
            p["team"] = team_map.get(tid, -1)

    # --- Pass 2: Write annotated video ---
    print("\n--- Pass 2: Writing annotated video ---")
    TEAM_COLORS = {0: (255, 100, 50), 1: (50, 100, 255), -1: (200, 200, 200)}
    BALL_COLOR = (0, 255, 255)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (w, h))

    for fd in all_frames_data:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw players
        for p in fd["players"]:
            x1, y1, x2, y2 = map(int, p["bbox"])
            team = p["team"]
            color = TEAM_COLORS.get(team, (200, 200, 200))
            thickness = 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            label = f"#{p['track_id']}"
            if p["class"] == "goalkeeper":
                label += " GK"
            elif p["class"] == "referee":
                label += " REF"
            team_label = f"T{team}" if team >= 0 else "?"
            label += f" {team_label}"

            cv2.putText(
                frame, label, (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
            )

        # Draw ball
        if fd["ball"]:
            x1, y1, x2, y2 = map(int, fd["ball"]["bbox"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), BALL_COLOR, 2)
            cv2.putText(
                frame, "BALL", (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, BALL_COLOR, 1,
            )

        out.write(frame)

    out.release()
    cap.release()

    # --- Save tracking data ---
    output = {
        "video": VIDEO_PATH,
        "fps": fps,
        "resolution": [w, h],
        "total_frames": total_frames,
        "team_map": {str(k): v for k, v in team_map.items()},
        "frames": all_frames_data,
    }

    with open(OUTPUT_DATA, "w") as f:
        json.dump(output, f)

    print(f"\nDone!")
    print(f"  Video: {OUTPUT_VIDEO}")
    print(f"  Data:  {OUTPUT_DATA}")

    # --- Stats ---
    all_track_ids = set()
    for fd in all_frames_data:
        for p in fd["players"]:
            all_track_ids.add(p["track_id"])

    ball_detected = sum(1 for fd in all_frames_data if fd["ball"] is not None)
    print(f"\n--- Stats ---")
    print(f"  Unique tracks: {len(all_track_ids)}")
    print(f"  Ball detected: {ball_detected}/{total_frames} frames ({100*ball_detected/total_frames:.0f}%)")
    for team_id in [0, 1]:
        players = [tid for tid, t in team_map.items() if t == team_id]
        print(f"  Team {chr(65+team_id)}: {len(players)} players (IDs: {sorted(players)})")


if __name__ == "__main__":
    run_tracking()
