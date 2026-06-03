"""
BEV visualisation helpers for the CoSelector GUI.

Two panels are produced:
  1. Scene overview  — top-down map of vehicle positions in RSU frame
  2. BEV detection   — fused occupancy grid with predicted & GT boxes
"""

from __future__ import annotations

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from matplotlib.figure import Figure

# Agent colour palette (index = agent_id).  Index 0 = RSU.
AGENT_COLORS = [
    "#f1c40f",  # 0: RSU → gold
    "#e74c3c",  # 1: red
    "#2ecc71",  # 2: green
    "#3498db",  # 3: blue
    "#9b59b6",  # 4: purple
    "#e67e22",  # 5: orange
]

# ─────────────────────────────────────────────────────────────────────────────
# 1. Scene overview
# ─────────────────────────────────────────────────────────────────────────────

def draw_scene_overview(
    vehicle_positions: dict[int, tuple[float, float]],
    selected_indices: list[int],
    metas: list,
    fig_size: tuple[float, float] = (6, 6),
) -> Figure:
    """
    Top-down bird's-eye view of vehicle positions in the RSU reference frame.

    Parameters
    ----------
    vehicle_positions : {agent_id: (x_rsu, y_rsu)}
    selected_indices  : list of *dataset agent indices* (0=RSU, 1-5=vehicles)
                        that were selected for fusion.
    metas             : list of AgentMeta (in the same order as state features).
    """
    fig, ax = plt.subplots(figsize=fig_size, facecolor="#1a1a2e")
    ax.set_facecolor("#16213e")

    # Grid lines
    for v in np.arange(-40, 45, 10):
        ax.axhline(v, color="#0f3460", linewidth=0.4, zorder=0)
        ax.axvline(v, color="#0f3460", linewidth=0.4, zorder=0)

    selected_set = set(selected_indices)

    legend_handles = []

    for agent_id, (x, y) in vehicle_positions.items():
        color = AGENT_COLORS[agent_id % len(AGENT_COLORS)]
        is_selected = agent_id in selected_set

        if agent_id == 0:
            # RSU — draw a star
            ax.scatter(x, y, s=260, marker="*", color=color, zorder=10,
                       edgecolors="white", linewidths=0.8)
            legend_handles.append(
                mpatches.Patch(color=color, label="RSU (agent 0)")
            )
        else:
            alpha  = 1.0 if is_selected else 0.30
            size   = 150  if is_selected else 60
            edge   = "white" if is_selected else "#444444"
            lw     = 1.2   if is_selected else 0.5

            ax.scatter(x, y, s=size, color=color, zorder=8,
                       alpha=alpha, edgecolors=edge, linewidths=lw)

            # Agent label
            ax.text(x + 1.0, y + 1.0, f"{agent_id}",
                    fontsize=7, color=color, alpha=alpha, zorder=9,
                    fontweight="bold" if is_selected else "normal")

            # Speed arrow (from metas if available)
            meta = next((m for m in metas if m.agent_id == agent_id), None)
            if meta and meta.speed > 0.5 and is_selected:
                arrow_scale = 3.0
                ax.annotate(
                    "", xy=(x + meta.vx * arrow_scale, y + meta.vy * arrow_scale),
                    xytext=(x, y),
                    arrowprops=dict(arrowstyle="->", color=color,
                                    lw=1.0, alpha=0.8),
                    zorder=9,
                )

            legend_label = f"Agent {agent_id}" + (" ✓" if is_selected else "")
            legend_handles.append(
                mpatches.Patch(color=color, alpha=alpha, label=legend_label)
            )

    # RSU detection range circle (rough estimate)
    circle = plt.Circle((0, 0), 40, color="#f1c40f", fill=False,
                         linestyle="--", linewidth=0.7, alpha=0.4, zorder=1)
    ax.add_patch(circle)

    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m, RSU frame)", color="#aaaaaa", fontsize=8)
    ax.set_ylabel("Y (m, RSU frame)", color="#aaaaaa", fontsize=8)
    ax.tick_params(colors="#888888", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")

    ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=7,
        framealpha=0.4,
        facecolor="#0f3460",
        labelcolor="#cccccc",
        edgecolor="#333355",
    )
    ax.set_title("Scene overview (RSU frame)", color="#dddddd", fontsize=9, pad=6)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. BEV detection panel
# ─────────────────────────────────────────────────────────────────────────────

def draw_bev_detection(
    bev_img: np.ndarray,
    pred_boxes: list[np.ndarray],
    gt_boxes_phys: list[np.ndarray],
    config,
    pred_scores: list[float] | None = None,
    score_threshold: float = 0.3,
    fig_size: tuple[float, float] = (7, 7),
) -> Figure:
    """
    Draw the fused BEV occupancy image with detection overlays.

    Parameters
    ----------
    bev_img       : (H, W) float32 — max-projected occupancy (Z collapsed).
    pred_boxes    : list of (4, 2) corner arrays in physical metres.
    gt_boxes_phys : list of (4, 2) corner arrays in physical metres.
    config        : coperception Config (for area_extents and voxel_size).
    """
    area_extents = np.asarray(config.area_extents)   # [[xmin,xmax],[ymin,ymax],[zmin,zmax]]
    voxel_size   = np.asarray(config.voxel_size)     # [dx, dy, dz]

    H, W = bev_img.shape

    fig, ax = plt.subplots(figsize=fig_size, facecolor="#1a1a2e")
    ax.set_facecolor("#0d0d1a")

    # ── BEV occupancy as background ──────────────────────────────────────────
    # Display with correct physical extent.
    # In the stored BEV the axes convention after rot90 is:
    #   rows (axis 0, top→bottom) → Y axis  (y_max → y_min)
    #   cols (axis 1, left→right) → X axis  (x_min → x_max)
    extent = [
        float(area_extents[0, 0]),  # x left
        float(area_extents[0, 1]),  # x right
        float(area_extents[1, 0]),  # y bottom
        float(area_extents[1, 1]),  # y top
    ]
    ax.imshow(
        bev_img,
        cmap="Blues",
        origin="lower",   # row 0 = y_min at bottom; matches physical coords
        extent=extent,
        vmin=0, vmax=1,
        alpha=0.85,
        zorder=1,
    )

    # ── GT boxes (green dashed) ───────────────────────────────────────────────
    for corners in gt_boxes_phys:
        _draw_box(ax, corners, color="#2ecc71", linestyle="--",
                  linewidth=1.0, alpha=0.6, zorder=5, fill_alpha=0.05)

    # ── Predicted boxes (red solid) ───────────────────────────────────────────
    scores = pred_scores if pred_scores is not None else [1.0] * len(pred_boxes)
    for corners, score in zip(pred_boxes, scores):
        if score < score_threshold:
            continue
        _draw_box(ax, corners, color="#e74c3c", linestyle="-",
                  linewidth=1.0, alpha=min(1.0, 0.4 + 0.6 * score),
                  zorder=6, fill_alpha=0.10)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color="#e74c3c", label=f"Predicted ({len(pred_boxes)})"),
        mpatches.Patch(color="#2ecc71", label=f"GT ({len(gt_boxes_phys)})"),
    ]
    ax.legend(
        handles=legend_handles, loc="upper right", fontsize=7,
        framealpha=0.5, facecolor="#0f3460", labelcolor="#cccccc",
        edgecolor="#333355",
    )

    ax.set_xlim(area_extents[0, 0], area_extents[0, 1])
    ax.set_ylim(area_extents[1, 0], area_extents[1, 1])
    ax.set_xlabel("X (m, RSU frame)", color="#aaaaaa", fontsize=8)
    ax.set_ylabel("Y (m, RSU frame)", color="#aaaaaa", fontsize=8)
    ax.tick_params(colors="#888888", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    ax.set_title("Fused BEV — detection output", color="#dddddd", fontsize=9, pad=6)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _draw_box(
    ax,
    corners: np.ndarray,
    color: str,
    linestyle: str = "-",
    linewidth: float = 1.0,
    alpha: float = 1.0,
    zorder: int = 5,
    fill_alpha: float = 0.0,
):
    """
    Draw a 2-D oriented box given its 4 corner coordinates in metres.

    ``corners`` shape: (4, 2) — each row is (x, y) in physical metres.
    """
    if corners is None or len(corners) < 4:
        return
    corners_closed = np.vstack([corners, corners[0]])
    ax.plot(
        corners_closed[:, 0], corners_closed[:, 1],
        color=color, linestyle=linestyle, linewidth=linewidth,
        alpha=alpha, zorder=zorder,
    )
    if fill_alpha > 0:
        from matplotlib.patches import Polygon as MplPolygon
        poly = MplPolygon(corners, closed=True, color=color,
                          alpha=fill_alpha, zorder=zorder - 1)
        ax.add_patch(poly)
    # Front-face indicator (midpoint of first edge → heading direction)
    mid = (corners[0] + corners[1]) / 2.0
    center = corners.mean(axis=0)
    ax.plot(
        [center[0], mid[0]], [center[1], mid[1]],
        color=color, linewidth=0.7, alpha=alpha * 0.7, zorder=zorder,
    )
