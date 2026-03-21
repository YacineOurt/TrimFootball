import numpy as np
from collections import Counter


def _classify_block(height):
    if height is None:
        return "N/A"
    if height < 35:
        return "bas"
    if height < 50:
        return "médian"
    return "haut"


def _classify_compactness(value):
    if value is None:
        return "N/A"
    if value < 400:
        return "serrée"
    if value < 700:
        return "correcte"
    return "étirée"


def _half_summary(metrics_slice, team_name):
    c_vals = [m["compactness"] for m in metrics_slice if m["compactness"] is not None]
    h_vals = [m["block_height"] for m in metrics_slice if m["block_height"] is not None]
    w_vals = [m["width"] for m in metrics_slice if m["width"] is not None]
    l_vals = [m["length"] for m in metrics_slice if m["length"] is not None]
    formations = [m["formation"] for m in metrics_slice if m["formation"] != "?"]

    avg_c = round(np.mean(c_vals), 0) if c_vals else None
    avg_h = round(np.mean(h_vals), 0) if h_vals else None
    avg_w = round(np.mean(w_vals), 0) if w_vals else None
    avg_l = round(np.mean(l_vals), 0) if l_vals else None
    dominant = Counter(formations).most_common(1)[0][0] if formations else "?"

    block_level = _classify_block(avg_h)
    compact_level = _classify_compactness(avg_c)

    parts = [f"{team_name} : {dominant}"]
    if avg_h is not None:
        parts.append(f"bloc {block_level} ({avg_h:.0f}m)")
    if avg_c is not None:
        parts.append(f"compacité {compact_level} ({avg_c:.0f}m²)")

    return {
        "text": ", ".join(parts),
        "formation": dominant,
        "avg_compactness": avg_c,
        "avg_block_height": avg_h,
        "avg_width": avg_w,
        "avg_length": avg_l,
        "block_level": block_level,
        "compact_level": compact_level,
    }


def generate_match_summary(all_metrics, fps, half_frame, team_names):
    total = len(all_metrics["team_a"])
    halves = [
        ("1ère MT", 0, half_frame),
        ("2ème MT", half_frame, total),
    ]

    summary = {}
    for half_label, start, end in halves:
        for team_key in ["team_a", "team_b"]:
            metrics_slice = all_metrics[team_key][start:end]
            name = team_names.get(team_key, team_key)
            s = _half_summary(metrics_slice, name)
            s["half_label"] = half_label
            key = f"{half_label}_{team_key}"
            summary[key] = s

    return summary
