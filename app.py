import io
import json
import threading
import uuid as uuid_mod
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from flask import Flask, abort, jsonify, render_template, request, send_file

from analysis.events import detect_key_moments
from analysis.metrics import (
    PITCH_L,
    PITCH_W,
    aggregate_team_metrics,
    compute_all_metrics,
    get_team_positions,
    metrics_to_timeseries,
    remove_duplicates,
)
from analysis.summary import generate_match_summary
from database import (
    create_match,
    delete_match,
    get_all_matches,
    get_all_teams,
    get_match,
    get_matches_by_team,
    init_db,
    update_match,
)
from pipeline import run_pipeline

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB

MATCHES_DIR = Path(__file__).parent / "matches"

# ---------------------------------------------------------------------------
# Cache (single-user tool, in-memory is fine)
# ---------------------------------------------------------------------------
_tracking_cache = {}
_metrics_cache = {}
_events_cache = {}


def load_tracking(path):
    path = str(path)
    if path not in _tracking_cache:
        with open(path) as f:
            data = json.load(f)
        for fd in data["frames"]:
            fd["players"] = remove_duplicates(fd["players"])
        _tracking_cache[path] = data
    return _tracking_cache[path]


def get_match_analysis(match):
    match_id = match["id"]
    tracking = load_tracking(match["tracking_path"])

    if match_id not in _metrics_cache:
        _metrics_cache[match_id] = compute_all_metrics(tracking)
    all_metrics = _metrics_cache[match_id]

    if match_id not in _events_cache:
        _events_cache[match_id] = detect_key_moments(all_metrics, tracking["fps"])
    events = _events_cache[match_id]

    return tracking, all_metrics, events


def team_names_for_match(match):
    if match["team_home_id"] == 0:
        return {"team_a": match["team_home"], "team_b": match["team_away"]}
    return {"team_a": match["team_away"], "team_b": match["team_home"]}


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    matches = get_all_matches()
    teams = get_all_teams()
    return render_template("index.html", matches=matches, teams=teams)


@app.route("/match/<int:match_id>")
def dashboard(match_id):
    match = get_match(match_id)
    if not match:
        abort(404)
    return render_template("dashboard.html", match=match)


@app.route("/opponent/<team_name>")
def opponent_report(team_name):
    matches = get_matches_by_team(team_name)
    if not matches:
        abort(404)

    matches_metrics = []
    match_infos = []
    for m in matches:
        tracking = load_tracking(m["tracking_path"])
        all_metrics = compute_all_metrics(tracking)
        is_home = m["team_home"].lower() == team_name.lower()
        team_id = m["team_home_id"] if is_home else (1 - m["team_home_id"])
        team_key = f'team_{"a" if team_id == 0 else "b"}'
        matches_metrics.append(all_metrics[team_key])
        match_infos.append(m)

    aggregate = aggregate_team_metrics(matches_metrics)

    return render_template(
        "opponent.html",
        team_name=team_name,
        matches=match_infos,
        aggregate=aggregate,
    )


# ---------------------------------------------------------------------------
# API — match data
# ---------------------------------------------------------------------------

@app.route("/api/match/<int:match_id>/overview")
def api_overview(match_id):
    match = get_match(match_id)
    if not match:
        return jsonify({"error": "Match not found"}), 404

    tracking, all_metrics, events = get_match_analysis(match)
    names = team_names_for_match(match)
    half_frame = match.get("half_frame") or len(tracking["frames"]) // 2

    summary = generate_match_summary(all_metrics, tracking["fps"], half_frame, names)
    timeseries = metrics_to_timeseries(all_metrics, tracking["fps"])

    # Attach team_names to events
    for ev in events:
        ev["team_label"] = names.get(ev["team"], ev["team"])

    return jsonify({
        "match": match,
        "team_names": names,
        "summary": summary,
        "events": events,
        "metrics": timeseries,
        "fps": tracking["fps"],
        "total_frames": len(tracking["frames"]),
    })


@app.route("/api/match/<int:match_id>/frame/<int:frame_num>")
def api_frame(match_id, frame_num):
    match = get_match(match_id)
    if not match:
        return jsonify({"error": "Match not found"}), 404

    tracking = load_tracking(match["tracking_path"])
    if frame_num < 0 or frame_num >= len(tracking["frames"]):
        return jsonify({"error": "Frame out of range"}), 400

    fd = tracking["frames"][frame_num]

    players = []
    for p in fd["players"]:
        if p.get("pitch_x") is not None:
            px, py = p["pitch_x"], p["pitch_y"]
            if 0 <= px <= PITCH_L and 0 <= py <= PITCH_W:
                players.append({
                    "track_id": p["track_id"],
                    "team": p.get("team", -1),
                    "cls": p.get("class", "player"),
                    "pitch_x": px,
                    "pitch_y": py,
                })

    ball = None
    if fd.get("ball") and fd["ball"].get("pitch_x") is not None:
        bx, by = fd["ball"]["pitch_x"], fd["ball"]["pitch_y"]
        if 0 <= bx <= PITCH_L and 0 <= by <= PITCH_W:
            ball = {"pitch_x": bx, "pitch_y": by}

    return jsonify({
        "frame": frame_num,
        "time": round(frame_num / tracking["fps"], 2),
        "players": players,
        "ball": ball,
    })


@app.route("/api/match/<int:match_id>/video")
def api_video(match_id):
    match = get_match(match_id)
    if not match:
        abort(404)
    return send_file(match["video_path"], mimetype="video/mp4", conditional=True)


# ---------------------------------------------------------------------------
# API — import / delete
# ---------------------------------------------------------------------------

@app.route("/api/match/import", methods=["POST"])
def api_import():
    # Validate video file
    video_file = request.files.get("video")
    if not video_file or not video_file.filename:
        return jsonify({"error": "Vidéo requise"}), 400

    # Create match directory
    match_uuid = uuid_mod.uuid4().hex[:8]
    match_dir = MATCHES_DIR / match_uuid
    match_dir.mkdir(parents=True, exist_ok=True)

    # Save video
    video_ext = Path(video_file.filename).suffix or ".mp4"
    video_path = match_dir / f"video{video_ext}"
    video_file.save(video_path)

    score_home = request.form.get("score_home")
    score_away = request.form.get("score_away")

    match_id = create_match(
        date=request.form["date"],
        team_home=request.form["team_home"],
        team_away=request.form["team_away"],
        score_home=int(score_home) if score_home else None,
        score_away=int(score_away) if score_away else None,
        video_path=str(video_path.resolve()),
        tracking_path="",
        fps=0,
        total_frames=0,
        half_frame=0,
        team_home_id=int(request.form.get("team_home_id", 0)),
        status="processing",
        match_dir=str(match_dir.resolve()),
    )

    # Capture form values before leaving request context
    half_frame_input = request.form.get("half_frame")

    # Launch pipeline in background
    def _run_bg():
        result = run_pipeline(match_id, video_path, match_dir)
        if result["success"]:
            half = result["total_frames"] // 2
            if half_frame_input:
                half = int(half_frame_input)
            update_match(
                match_id,
                status="ready",
                tracking_path=result["tracking_path"],
                fps=result["fps"],
                total_frames=result["total_frames"],
                half_frame=half,
            )
        else:
            update_match(match_id, status="error")

    thread = threading.Thread(target=_run_bg, daemon=True)
    thread.start()

    return jsonify({"id": match_id})


@app.route("/api/match/<int:match_id>/status")
def api_status(match_id):
    match = get_match(match_id)
    if not match:
        return jsonify({"error": "Match not found"}), 404

    result = {"status": match["status"]}

    # Read progress file if processing
    if match["status"] == "processing" and match.get("match_dir"):
        progress_file = Path(match["match_dir"]) / "progress.json"
        if progress_file.exists():
            with open(progress_file) as f:
                result["progress"] = json.load(f)

    return jsonify(result)


@app.route("/api/match/<int:match_id>", methods=["DELETE"])
def api_delete(match_id):
    delete_match(match_id)
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# API — export
# ---------------------------------------------------------------------------

@app.route("/api/match/<int:match_id>/export/frame/<int:frame_num>")
def api_export_frame(match_id, frame_num):
    match = get_match(match_id)
    if not match:
        abort(404)

    tracking = load_tracking(match["tracking_path"])
    if frame_num < 0 or frame_num >= len(tracking["frames"]):
        abort(400)

    fd = tracking["frames"][frame_num]
    names = team_names_for_match(match)

    fig = _create_pitch_figure(fd, names)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)
    buf.seek(0)

    time_sec = frame_num / tracking["fps"]
    time_str = f"{int(time_sec // 60):02d}m{int(time_sec % 60):02d}s"
    return send_file(buf, mimetype="image/png", download_name=f"tactique_{time_str}.png")


@app.route("/api/opponent/<team_name>/data")
def api_opponent_data(team_name):
    matches = get_matches_by_team(team_name)
    if not matches:
        return jsonify({"error": "No matches found"}), 404

    matches_metrics = []
    match_infos = []
    for m in matches:
        tracking = load_tracking(m["tracking_path"])
        all_metrics = compute_all_metrics(tracking)
        is_home = m["team_home"].lower() == team_name.lower()
        team_id = m["team_home_id"] if is_home else (1 - m["team_home_id"])
        team_key = f'team_{"a" if team_id == 0 else "b"}'
        matches_metrics.append(all_metrics[team_key])
        match_infos.append({
            "id": m["id"],
            "date": m["date"],
            "opponent": m["team_away"] if is_home else m["team_home"],
            "score": f"{m['score_home']}-{m['score_away']}" if m["score_home"] is not None else "?",
        })

    aggregate = aggregate_team_metrics(matches_metrics)
    aggregate["matches"] = match_infos
    return jsonify(aggregate)


# ---------------------------------------------------------------------------
# Matplotlib pitch export
# ---------------------------------------------------------------------------

PEN_D, PEN_W = 16.5, 40.32
GA_D, GA_W = 5.5, 18.32
CR = 9.15
TEAM_COLORS = {0: "#ff6432", 1: "#3264ff", -1: "#aaaaaa"}


def _create_pitch_figure(frame_data, team_names):
    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    ax.set_xlim(-3, PITCH_L + 3)
    ax.set_ylim(-3, PITCH_W + 3)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("#1a1a2e")

    # Pitch surface
    pitch = plt.Rectangle((0, 0), PITCH_L, PITCH_W, fc="#228b22", ec="white", lw=1.5)
    ax.add_patch(pitch)

    # Halfway
    ax.plot([PITCH_L / 2, PITCH_L / 2], [0, PITCH_W], "w-", lw=1)
    # Center circle
    circle = plt.Circle((PITCH_L / 2, PITCH_W / 2), CR, fill=False, ec="white", lw=1)
    ax.add_patch(circle)
    # Penalty areas
    py1, py2 = (PITCH_W - PEN_W) / 2, (PITCH_W + PEN_W) / 2
    for x0, w in [(0, PEN_D), (PITCH_L - PEN_D, PEN_D)]:
        ax.add_patch(plt.Rectangle((x0, py1), w, PEN_W, fill=False, ec="white", lw=1))
    # Goal areas
    gy1, gy2 = (PITCH_W - GA_W) / 2, (PITCH_W + GA_W) / 2
    for x0, w in [(0, GA_D), (PITCH_L - GA_D, GA_D)]:
        ax.add_patch(plt.Rectangle((x0, gy1), w, GA_W, fill=False, ec="white", lw=1))

    # Convex hulls
    from scipy.spatial import ConvexHull

    for team_id in [0, 1]:
        positions = get_team_positions(frame_data, team_id)
        if len(positions) >= 3:
            pts = np.array(positions)
            try:
                hull = ConvexHull(pts)
                hull_pts = pts[hull.vertices]
                hull_pts = np.vstack([hull_pts, hull_pts[0]])
                ax.fill(hull_pts[:, 0], hull_pts[:, 1], alpha=0.15, color=TEAM_COLORS[team_id])
                ax.plot(hull_pts[:, 0], hull_pts[:, 1], "--", color=TEAM_COLORS[team_id], lw=1.5, alpha=0.6)
            except Exception:
                pass

    # Players
    for p in frame_data["players"]:
        if p.get("pitch_x") is None:
            continue
        px, py = p["pitch_x"], p["pitch_y"]
        if not (0 <= px <= PITCH_L and 0 <= py <= PITCH_W):
            continue
        team = p.get("team", -1)
        color = TEAM_COLORS.get(team, "#aaa")
        ax.plot(px, py, "o", color=color, markersize=10, markeredgecolor="white", markeredgewidth=1.5)

    # Ball
    ball = frame_data.get("ball")
    if ball and ball.get("pitch_x") is not None:
        bx, by = ball["pitch_x"], ball["pitch_y"]
        if 0 <= bx <= PITCH_L and 0 <= by <= PITCH_W:
            ax.plot(bx, by, "o", color="yellow", markersize=7, markeredgecolor="black", markeredgewidth=1)

    # Legend
    patches = []
    for team_id in [0, 1]:
        key = f'team_{"a" if team_id == 0 else "b"}'
        name = team_names.get(key, f"Team {team_id}")
        patches.append(mpatches.Patch(color=TEAM_COLORS[team_id], label=name))
    ax.legend(handles=patches, loc="upper right", fontsize=9, facecolor="#1a1a2e", edgecolor="#555", labelcolor="white")

    return fig


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    init_db()
    app.run(debug=True, port=5000)
