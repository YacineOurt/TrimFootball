"""
Pipeline: run step1 (tracking) + step2 (homography) for a match.
Called as a background process by the Flask app after video upload.
"""

import json
import sys
import traceback
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent


def update_progress(match_dir, step, progress, message):
    """Write progress file for the frontend to poll."""
    progress_file = Path(match_dir) / "progress.json"
    with open(progress_file, "w") as f:
        json.dump({"step": step, "progress": round(progress, 2), "message": message}, f)


def run_pipeline(match_id, video_path, match_dir):
    """Run the full analysis pipeline for a match."""
    match_dir = Path(match_dir)
    video_path = str(video_path)
    tracking_path = str(match_dir / "tracking_data.json")

    try:
        # --- Step 1: Tracking ---
        update_progress(match_dir, 1, 0.0, "Détection des joueurs...")

        if str(PROJECT_DIR) not in sys.path:
            sys.path.insert(0, str(PROJECT_DIR))
        from step1_tracking.step1_tracking import run_tracking

        def step1_progress(stage, pct):
            update_progress(match_dir, 1, pct, f"Détection des joueurs... {int(pct*100)}%")

        run_tracking(
            video_path=video_path,
            output_data=tracking_path,
            output_video=str(match_dir / "output_tracking.mp4"),
            progress_callback=step1_progress,
        )

        # --- Step 2: Homography ---
        update_progress(match_dir, 2, 0.0, "Calibration du terrain...")

        from step2_homography.step2_homography import run as run_homography

        def step2_progress(stage, pct):
            update_progress(match_dir, 2, pct, f"Calibration du terrain... {int(pct*100)}%")

        run_homography(
            video_path=video_path,
            tracking_path=tracking_path,
            output_dir=str(match_dir),
            progress_callback=step2_progress,
        )

        # --- Done ---
        update_progress(match_dir, 3, 1.0, "Analyse terminée")

        # Read tracking to get metadata
        with open(tracking_path) as f:
            tdata = json.load(f)

        return {
            "success": True,
            "tracking_path": tracking_path,
            "fps": tdata.get("fps", 0),
            "total_frames": len(tdata.get("frames", [])),
        }

    except Exception as e:
        error_msg = f"Erreur: {e}"
        print(f"Pipeline error for match {match_id}: {traceback.format_exc()}")
        update_progress(match_dir, -1, 0, error_msg)
        return {"success": False, "error": error_msg}
