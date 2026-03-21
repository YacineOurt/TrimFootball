import numpy as np
from scipy.ndimage import uniform_filter1d


def smooth_series(values, window):
    arr = np.array(values, dtype=float)
    mask = ~np.isnan(arr)
    if mask.sum() < window:
        return arr
    interp = np.interp(np.arange(len(arr)), np.where(mask)[0], arr[mask])
    return uniform_filter1d(interp, window)


def _detect_metric_events(values, fps, window, min_gap, team_key, metric_name, fmt_up, fmt_down, threshold_mult=2.0):
    events = []
    arr = np.array([v if v is not None else float("nan") for v in values], dtype=float)
    valid = ~np.isnan(arr)

    if valid.sum() < window * 2:
        return events

    interp = np.interp(np.arange(len(arr)), np.where(valid)[0], arr[valid])
    smoothed = smooth_series(interp, window)

    residual = interp - smoothed
    std = np.std(residual)
    if std < 1e-6:
        return events

    last_event = -min_gap
    for i in range(window, len(residual) - window):
        if i - last_event < min_gap:
            continue
        if abs(residual[i]) > threshold_mult * std:
            direction = "up" if residual[i] > 0 else "down"
            fmt = fmt_up if direction == "up" else fmt_down
            events.append({
                "frame": i,
                "time": round(i / fps, 1),
                "type": f"{metric_name}_{'spike' if direction == 'up' else 'drop'}",
                "team": team_key,
                "value": round(float(interp[i]), 1),
                "description": fmt.format(value=interp[i]),
            })
            last_event = i

    return events


def _detect_formation_changes(metrics, fps, team_key, min_gap):
    events = []
    formations = [m["formation"] for m in metrics]
    prev = None
    last_frame = -min_gap

    for i, f in enumerate(formations):
        if f != "?" and f != prev:
            if prev is not None and prev != "?" and i - last_frame > min_gap:
                events.append({
                    "frame": i,
                    "time": round(i / fps, 1),
                    "type": "formation_change",
                    "team": team_key,
                    "description": f"Formation : {prev} → {f}",
                })
                last_frame = i
            prev = f

    return events


def detect_key_moments(all_metrics, fps):
    events = []
    window = max(int(fps * 2), 5)
    min_gap = int(fps * 8)

    for team_key in ["team_a", "team_b"]:
        metrics = all_metrics[team_key]

        # Compactness events
        events.extend(_detect_metric_events(
            [m["compactness"] for m in metrics],
            fps, window, min_gap, team_key,
            metric_name="compactness",
            fmt_up="Compacité explose ({value:.0f}m²)",
            fmt_down="Compacité se resserre ({value:.0f}m²)",
            threshold_mult=2.0,
        ))

        # Block height events
        events.extend(_detect_metric_events(
            [m["block_height"] for m in metrics],
            fps, window, min_gap, team_key,
            metric_name="block_height",
            fmt_up="Bloc monte ({value:.0f}m)",
            fmt_down="Bloc descend ({value:.0f}m)",
            threshold_mult=2.0,
        ))

        # Formation changes
        events.extend(_detect_formation_changes(metrics, fps, team_key, min_gap))

    events.sort(key=lambda e: e["time"])
    return events
