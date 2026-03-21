"""
Step 3: Tactical metrics dashboard (Streamlit)
- Formation detection
- Heatmaps
- Team compactness
- Defensive block height
- Pressing intensity
"""

import json
import os
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
from scipy.spatial import ConvexHull

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

# --- Config ---
PITCH_L, PITCH_W = 105.0, 68.0
PEN_D, PEN_W = 16.5, 40.32
GA_D, GA_W = 5.5, 18.32
CR = 9.15
TEAM_NAMES = {0: "Team A", 1: "Team B"}
TEAM_COLORS_HEX = {0: "#FF6432", 1: "#3264FF"}


# ---- Data loading ----

@st.cache_data
def load_tracking():
    tracking_path = PROJECT_DIR / "tracking_data.json"
    with open(tracking_path) as f:
        data = json.load(f)
    # Deduplicate per frame
    for fd in data["frames"]:
        fd["players"] = remove_duplicates(fd["players"])
    return data


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
    """Get (x, y) positions for a team in a frame."""
    positions = []
    for p in frame_data["players"]:
        if p.get("team") == team_id and p.get("pitch_x") is not None:
            px, py = p["pitch_x"], p["pitch_y"]
            if 0 <= px <= PITCH_L and 0 <= py <= PITCH_W:
                positions.append((px, py))
    return positions


# ---- Metrics ----

def compute_compactness(positions):
    """Convex hull area of team positions (m²). Lower = more compact."""
    if len(positions) < 3:
        return None
    pts = np.array(positions)
    try:
        hull = ConvexHull(pts)
        return round(hull.volume, 1)  # 2D: volume = area
    except Exception:
        return None


def compute_block_height(positions, attacking_right=True):
    """Average x-position of the 4 deepest players (defensive line)."""
    if len(positions) < 4:
        return None
    xs = sorted([p[0] for p in positions])
    if attacking_right:
        return round(np.mean(xs[:4]), 1)  # 4 lowest x = defensive line
    else:
        return round(np.mean(xs[-4:]), 1)


def compute_team_width(positions):
    """Width spread of the team along y-axis (m)."""
    if len(positions) < 2:
        return None
    ys = [p[1] for p in positions]
    return round(max(ys) - min(ys), 1)


def compute_team_length(positions):
    """Length spread of the team along x-axis (m)."""
    if len(positions) < 2:
        return None
    xs = [p[0] for p in positions]
    return round(max(xs) - min(xs), 1)


def detect_formation(positions, attacking_right=True):
    """Detect formation (e.g. 4-3-3) by clustering x-positions into lines."""
    if len(positions) < 7:
        return "?"
    xs = sorted([p[0] for p in positions])
    # Remove GK (deepest player)
    if attacking_right:
        xs = xs[1:]  # remove lowest x (GK)
    else:
        xs = xs[:-1]

    if len(xs) < 3:
        return "?"

    # Find the N-1 gaps between sorted x-positions
    gaps = [(xs[i + 1] - xs[i], i) for i in range(len(xs) - 1)]
    gaps.sort(reverse=True)

    # Use the 2 or 3 largest gaps to split into 3 or 4 lines
    # Try 3 lines first (most common: DEF-MID-ATT)
    n_lines = min(3, len(gaps))
    split_indices = sorted([g[1] for g in gaps[:n_lines - 1]])

    # Only use splits where the gap is meaningful (>3m)
    split_indices = sorted([g[1] for g in gaps if g[0] > 3][:3])

    if not split_indices:
        return f"{len(xs)}"

    # Build lines from splits
    lines = []
    prev = 0
    for si in split_indices:
        lines.append(xs[prev:si + 1])
        prev = si + 1
    lines.append(xs[prev:])

    formation = "-".join(str(len(line)) for line in lines)
    return formation


def compute_all_metrics(tracking_data, frame_range=None):
    """Compute metrics for each frame."""
    frames = tracking_data["frames"]
    if frame_range:
        frames = frames[frame_range[0]:frame_range[1]]

    metrics = {"team_a": [], "team_b": []}
    for fd in frames:
        for team_id, key in [(0, "team_a"), (1, "team_b")]:
            pos = get_team_positions(fd, team_id)
            metrics[key].append({
                "compactness": compute_compactness(pos),
                "block_height": compute_block_height(pos, attacking_right=(team_id == 0)),
                "width": compute_team_width(pos),
                "length": compute_team_length(pos),
                "formation": detect_formation(pos, attacking_right=(team_id == 0)),
                "n_players": len(pos),
            })
    return metrics


# ---- Pitch drawing (Plotly) ----

def draw_pitch_plotly():
    """Create a plotly pitch figure."""
    fig = go.Figure()

    # Draw pitch as traces instead of shapes (more reliable rendering)
    # Pitch surface
    fig.add_trace(go.Scatter(
        x=[0, PITCH_L, PITCH_L, 0, 0], y=[0, 0, PITCH_W, PITCH_W, 0],
        fill="toself", fillcolor="rgb(34,120,34)",
        line=dict(color="white", width=2),
        showlegend=False, hoverinfo="skip",
    ))
    # Halfway line
    fig.add_trace(go.Scatter(
        x=[PITCH_L / 2, PITCH_L / 2], y=[0, PITCH_W],
        mode="lines", line=dict(color="white", width=1),
        showlegend=False, hoverinfo="skip",
    ))
    # Center circle
    theta = np.linspace(0, 2 * np.pi, 60)
    fig.add_trace(go.Scatter(
        x=PITCH_L / 2 + CR * np.cos(theta), y=PITCH_W / 2 + CR * np.sin(theta),
        mode="lines", line=dict(color="white", width=1),
        showlegend=False, hoverinfo="skip",
    ))
    # Penalty areas
    py1, py2 = (PITCH_W - PEN_W) / 2, (PITCH_W + PEN_W) / 2
    for x0, x1 in [(0, PEN_D), (PITCH_L - PEN_D, PITCH_L)]:
        fig.add_trace(go.Scatter(
            x=[x0, x1, x1, x0, x0], y=[py1, py1, py2, py2, py1],
            mode="lines", line=dict(color="white", width=1),
            showlegend=False, hoverinfo="skip",
        ))
    # Goal areas
    gy1, gy2 = (PITCH_W - GA_W) / 2, (PITCH_W + GA_W) / 2
    for x0, x1 in [(0, GA_D), (PITCH_L - GA_D, PITCH_L)]:
        fig.add_trace(go.Scatter(
            x=[x0, x1, x1, x0, x0], y=[gy1, gy1, gy2, gy2, gy1],
            mode="lines", line=dict(color="white", width=1),
            showlegend=False, hoverinfo="skip",
        ))

    fig.update_layout(
        width=800, height=530,
        xaxis=dict(range=[-3, PITCH_L + 3], showgrid=False, zeroline=False,
                   showticklabels=False, constrain="domain"),
        yaxis=dict(range=[-3, PITCH_W + 3], showgrid=False, zeroline=False,
                   showticklabels=False, scaleanchor="x"),
        plot_bgcolor="rgb(30,30,30)",
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


def add_players_to_pitch(fig, frame_data):
    """Add player and ball markers to a pitch figure."""
    # Draw convex hulls first (behind players)
    for team_id in [0, 1]:
        positions = get_team_positions(frame_data, team_id)
        if len(positions) >= 3:
            pts = np.array(positions)
            try:
                hull = ConvexHull(pts)
                hull_pts = pts[hull.vertices]
                hull_pts = np.vstack([hull_pts, hull_pts[0]])  # close polygon
                fig.add_trace(go.Scatter(
                    x=hull_pts[:, 0], y=hull_pts[:, 1],
                    fill="toself",
                    fillcolor=f"rgba({','.join(str(int(c)) for c in bytes.fromhex(TEAM_COLORS_HEX[team_id][1:]))},0.15)",
                    line=dict(color=TEAM_COLORS_HEX[team_id], width=2, dash="dot"),
                    showlegend=False, hoverinfo="skip",
                ))
            except Exception:
                pass

    # Then draw players on top
    for team_id in [0, 1]:
        positions = get_team_positions(frame_data, team_id)
        if not positions:
            continue
        xs, ys = zip(*positions)
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers+text",
            marker=dict(size=14, color=TEAM_COLORS_HEX[team_id],
                        line=dict(color="white", width=2)),
            name=TEAM_NAMES[team_id],
            hovertemplate="x=%{x:.1f}m, y=%{y:.1f}m<extra></extra>",
        ))

    # Ball
    ball = frame_data.get("ball")
    if ball and ball.get("pitch_x") is not None:
        bx, by = ball["pitch_x"], ball["pitch_y"]
        if 0 <= bx <= PITCH_L and 0 <= by <= PITCH_W:
            fig.add_trace(go.Scatter(
                x=[bx], y=[by], mode="markers",
                marker=dict(size=10, color="yellow",
                            line=dict(color="black", width=2),
                            symbol="circle"),
                name="Ball",
            ))

    return fig


def make_heatmap(tracking_data, team_id, frame_start=0, frame_end=None):
    """Generate a heatmap for a team over a range of frames."""
    frames = tracking_data["frames"][frame_start:frame_end]
    all_x, all_y = [], []
    for fd in frames:
        for p in fd["players"]:
            if p.get("team") == team_id and p.get("pitch_x") is not None:
                px, py = p["pitch_x"], p["pitch_y"]
                if 0 <= px <= PITCH_L and 0 <= py <= PITCH_W:
                    all_x.append(px)
                    all_y.append(py)

    if not all_x:
        return None

    # 2D histogram
    from scipy.ndimage import gaussian_filter
    heatmap, xedges, yedges = np.histogram2d(
        all_x, all_y, bins=[35, 23],
        range=[[0, PITCH_L], [0, PITCH_W]]
    )
    heatmap = gaussian_filter(heatmap, sigma=1.5)

    # Build figure: heatmap FIRST, then pitch lines on top
    fig = go.Figure()

    # Heatmap layer
    color = TEAM_COLORS_HEX[team_id]
    colorscale = [[0, "rgba(34,120,34,1)"], [0.5, color], [1, "white"]]
    fig.add_trace(go.Heatmap(
        z=heatmap.T,
        x0=0, dx=PITCH_L / 35,
        y0=0, dy=PITCH_W / 23,
        colorscale=colorscale,
        showscale=False,
        hoverinfo="skip",
    ))

    # Pitch lines on top (just outlines, no fill)
    lines_to_draw = [
        ([0, PITCH_L, PITCH_L, 0, 0], [0, 0, PITCH_W, PITCH_W, 0]),  # outline
        ([PITCH_L / 2, PITCH_L / 2], [0, PITCH_W]),  # halfway
    ]
    # Penalty areas
    py1, py2 = (PITCH_W - PEN_W) / 2, (PITCH_W + PEN_W) / 2
    lines_to_draw.append(([0, PEN_D, PEN_D, 0], [py1, py1, py2, py2]))
    lines_to_draw.append(([PITCH_L, PITCH_L - PEN_D, PITCH_L - PEN_D, PITCH_L], [py1, py1, py2, py2]))
    # Goal areas
    gy1, gy2 = (PITCH_W - GA_W) / 2, (PITCH_W + GA_W) / 2
    lines_to_draw.append(([0, GA_D, GA_D, 0], [gy1, gy1, gy2, gy2]))
    lines_to_draw.append(([PITCH_L, PITCH_L - GA_D, PITCH_L - GA_D, PITCH_L], [gy1, gy1, gy2, gy2]))

    for xs, ys in lines_to_draw:
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(color="rgba(255,255,255,0.6)", width=1),
            showlegend=False, hoverinfo="skip",
        ))
    # Center circle
    theta = np.linspace(0, 2 * np.pi, 60)
    fig.add_trace(go.Scatter(
        x=PITCH_L / 2 + CR * np.cos(theta), y=PITCH_W / 2 + CR * np.sin(theta),
        mode="lines", line=dict(color="rgba(255,255,255,0.6)", width=1),
        showlegend=False, hoverinfo="skip",
    ))

    fig.update_layout(
        width=800, height=530,
        xaxis=dict(range=[-1, PITCH_L + 1], showgrid=False, zeroline=False,
                   showticklabels=False, constrain="domain"),
        yaxis=dict(range=[-1, PITCH_W + 1], showgrid=False, zeroline=False,
                   showticklabels=False, scaleanchor="x"),
        plot_bgcolor="rgb(34,120,34)",
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


# ---- Streamlit App ----

def main():
    st.set_page_config(page_title="VAFC Tactical Analysis", layout="wide")
    st.title("VAFC - Analyse Tactique")

    tracking = load_tracking()
    n_frames = len(tracking["frames"])
    fps = tracking["fps"]

    # Sidebar
    st.sidebar.header("Navigation")
    frame_idx = st.sidebar.slider("Frame", 0, n_frames - 1, n_frames // 2)
    time_sec = frame_idx / fps
    st.sidebar.caption(f"Temps: {int(time_sec // 60)}:{int(time_sec % 60):02d}")

    view = st.sidebar.radio("Vue", ["Positions", "Heatmap Team A", "Heatmap Team B"])

    # --- Main content ---
    fd = tracking["frames"][frame_idx]

    # Debug info
    pos_a = get_team_positions(fd, 0)
    pos_b = get_team_positions(fd, 1)
    st.caption(f"Frame {frame_idx} — Team A: {len(pos_a)} joueurs, Team B: {len(pos_b)} joueurs")

    if view == "Positions":
        # Pitch with players
        fig = draw_pitch_plotly()
        fig = add_players_to_pitch(fig, fd)
        st.plotly_chart(fig, use_container_width=True)

    elif view.startswith("Heatmap"):
        team_id = 0 if "Team A" in view else 1
        fig = make_heatmap(tracking, team_id)
        if fig:
            st.plotly_chart(fig, use_container_width=False)
        else:
            st.warning("Pas assez de données pour la heatmap")

    # --- Metrics ---
    st.markdown("---")
    col1, col2 = st.columns(2)

    for team_id, col in [(0, col1), (1, col2)]:
        pos = get_team_positions(fd, team_id)
        name = TEAM_NAMES[team_id]
        color = TEAM_COLORS_HEX[team_id]

        with col:
            st.markdown(f"### <span style='color:{color}'>{name}</span>", unsafe_allow_html=True)

            if len(pos) < 3:
                st.warning(f"Seulement {len(pos)} joueurs détectés")
                continue

            formation = detect_formation(pos, attacking_right=(team_id == 0))
            compactness = compute_compactness(pos)
            block_h = compute_block_height(pos, attacking_right=(team_id == 0))
            width = compute_team_width(pos)
            length = compute_team_length(pos)

            m1, m2, m3 = st.columns(3)
            m1.metric("Formation", formation)
            m2.metric("Joueurs", len(pos))
            m3.metric("Compacité", f"{compactness} m²" if compactness else "N/A")

            m4, m5, m6 = st.columns(3)
            m4.metric("Bloc défensif", f"{block_h} m" if block_h else "N/A")
            m5.metric("Largeur", f"{width} m" if width else "N/A")
            m6.metric("Longueur", f"{length} m" if length else "N/A")

    # --- Timeline metrics ---
    st.markdown("---")
    st.subheader("Evolution temporelle")

    all_metrics = compute_all_metrics(tracking)

    # Compactness over time
    fig_compact = go.Figure()
    for team_id, key in [(0, "team_a"), (1, "team_b")]:
        values = [m["compactness"] for m in all_metrics[key]]
        frames_x = list(range(len(values)))
        times = [f / fps for f in frames_x]
        fig_compact.add_trace(go.Scatter(
            x=times, y=values, mode="lines",
            name=TEAM_NAMES[team_id],
            line=dict(color=TEAM_COLORS_HEX[team_id], width=2),
        ))
    # Current frame marker
    fig_compact.add_vline(x=frame_idx / fps, line_dash="dash", line_color="white", opacity=0.5)
    fig_compact.update_layout(
        title="Compacité (m²) — plus bas = plus compact",
        xaxis_title="Temps (s)", yaxis_title="Surface convex hull (m²)",
        height=300, template="plotly_dark",
    )
    st.plotly_chart(fig_compact, use_container_width=True)

    # Block height over time
    fig_block = go.Figure()
    for team_id, key in [(0, "team_a"), (1, "team_b")]:
        values = [m["block_height"] for m in all_metrics[key]]
        frames_x = list(range(len(values)))
        times = [f / fps for f in frames_x]
        fig_block.add_trace(go.Scatter(
            x=times, y=values, mode="lines",
            name=TEAM_NAMES[team_id],
            line=dict(color=TEAM_COLORS_HEX[team_id], width=2),
        ))
    fig_block.add_vline(x=frame_idx / fps, line_dash="dash", line_color="white", opacity=0.5)
    fig_block.update_layout(
        title="Hauteur du bloc défensif (m depuis la ligne de but)",
        xaxis_title="Temps (s)", yaxis_title="Position (m)",
        height=300, template="plotly_dark",
    )
    st.plotly_chart(fig_block, use_container_width=True)


if __name__ == "__main__":
    main()
