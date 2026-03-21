import numpy as np
from scipy.spatial import ConvexHull

PITCH_L, PITCH_W = 105.0, 68.0


def remove_duplicates(players, iou_thresh=0.5):
    players = sorted(players, key=lambda p: p["conf"], reverse=True)
    keep = []
    for p in players:
        bx1, by1, bx2, by2 = p["bbox"]
        dup = False
        for k in keep:
            kx1, ky1, kx2, ky2 = k["bbox"]
            ix1, iy1 = max(bx1, kx1), max(by1, ky1)
            ix2, iy2 = min(bx2, kx2), min(by2, ky2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = (bx2 - bx1) * (by2 - by1) + (kx2 - kx1) * (ky2 - ky1) - inter
            if union > 0 and inter / union > iou_thresh:
                dup = True
                break
        if not dup:
            keep.append(p)
    return keep


def get_team_positions(frame_data, team_id):
    positions = []
    for p in frame_data["players"]:
        if p.get("team") == team_id and p.get("pitch_x") is not None:
            px, py = p["pitch_x"], p["pitch_y"]
            if 0 <= px <= PITCH_L and 0 <= py <= PITCH_W:
                positions.append((px, py))
    return positions


def compute_compactness(positions):
    if len(positions) < 3:
        return None
    pts = np.array(positions)
    try:
        hull = ConvexHull(pts)
        return round(hull.volume, 1)
    except Exception:
        return None


def compute_block_height(positions, attacking_right=True):
    if len(positions) < 4:
        return None
    xs = sorted([p[0] for p in positions])
    if attacking_right:
        return round(np.mean(xs[:4]), 1)
    else:
        return round(np.mean(xs[-4:]), 1)


def compute_team_width(positions):
    if len(positions) < 2:
        return None
    ys = [p[1] for p in positions]
    return round(max(ys) - min(ys), 1)


def compute_team_length(positions):
    if len(positions) < 2:
        return None
    xs = [p[0] for p in positions]
    return round(max(xs) - min(xs), 1)


def detect_formation(positions, attacking_right=True):
    if len(positions) < 7:
        return "?"
    xs = sorted([p[0] for p in positions])
    if attacking_right:
        xs = xs[1:]
    else:
        xs = xs[:-1]
    if len(xs) < 3:
        return "?"

    gaps = [(xs[i + 1] - xs[i], i) for i in range(len(xs) - 1)]
    gaps.sort(reverse=True)
    split_indices = sorted([g[1] for g in gaps if g[0] > 3][:3])

    if not split_indices:
        return f"{len(xs)}"

    lines = []
    prev = 0
    for si in split_indices:
        lines.append(xs[prev : si + 1])
        prev = si + 1
    lines.append(xs[prev:])

    return "-".join(str(len(line)) for line in lines)


def compute_frame_metrics(frame_data):
    result = {}
    for team_id, key in [(0, "team_a"), (1, "team_b")]:
        pos = get_team_positions(frame_data, team_id)
        result[key] = {
            "compactness": compute_compactness(pos),
            "block_height": compute_block_height(pos, attacking_right=(team_id == 0)),
            "width": compute_team_width(pos),
            "length": compute_team_length(pos),
            "formation": detect_formation(pos, attacking_right=(team_id == 0)),
            "n_players": len(pos),
        }
    return result


def compute_all_metrics(tracking_data):
    metrics = {"team_a": [], "team_b": []}
    for fd in tracking_data["frames"]:
        fm = compute_frame_metrics(fd)
        metrics["team_a"].append(fm["team_a"])
        metrics["team_b"].append(fm["team_b"])
    return metrics


def metrics_to_timeseries(all_metrics, fps):
    result = {}
    for key in ["team_a", "team_b"]:
        n = len(all_metrics[key])
        result[key] = {
            "times": [round(i / fps, 2) for i in range(n)],
            "compactness": [m["compactness"] for m in all_metrics[key]],
            "block_height": [m["block_height"] for m in all_metrics[key]],
            "width": [m["width"] for m in all_metrics[key]],
            "length": [m["length"] for m in all_metrics[key]],
        }
    return result


def aggregate_team_metrics(matches_metrics):
    """Aggregate metrics across multiple matches for opponent report."""
    from collections import Counter

    all_compactness = []
    all_block_height = []
    all_width = []
    all_length = []
    all_formations = []

    per_match = []

    for metrics_list in matches_metrics:
        c_vals = [m["compactness"] for m in metrics_list if m["compactness"] is not None]
        h_vals = [m["block_height"] for m in metrics_list if m["block_height"] is not None]
        w_vals = [m["width"] for m in metrics_list if m["width"] is not None]
        l_vals = [m["length"] for m in metrics_list if m["length"] is not None]
        formations = [m["formation"] for m in metrics_list if m["formation"] != "?"]

        match_data = {}
        if c_vals:
            avg_c = np.mean(c_vals)
            all_compactness.append(avg_c)
            match_data["compactness"] = round(avg_c, 1)
        if h_vals:
            avg_h = np.mean(h_vals)
            all_block_height.append(avg_h)
            match_data["block_height"] = round(avg_h, 1)
        if w_vals:
            all_width.append(np.mean(w_vals))
            match_data["width"] = round(np.mean(w_vals), 1)
        if l_vals:
            all_length.append(np.mean(l_vals))
            match_data["length"] = round(np.mean(l_vals), 1)
        all_formations.extend(formations)
        if formations:
            match_data["formation"] = Counter(formations).most_common(1)[0][0]

        per_match.append(match_data)

    dominant = Counter(all_formations).most_common(1)[0][0] if all_formations else "?"

    return {
        "n_matches": len(matches_metrics),
        "dominant_formation": dominant,
        "avg_compactness": round(np.mean(all_compactness), 1) if all_compactness else None,
        "avg_block_height": round(np.mean(all_block_height), 1) if all_block_height else None,
        "avg_width": round(np.mean(all_width), 1) if all_width else None,
        "avg_length": round(np.mean(all_length), 1) if all_length else None,
        "per_match": per_match,
    }
