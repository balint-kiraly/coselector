"""
CoSelector — V2X Cooperative Detection GUI

Run with:
    streamlit run gui/app.py
"""

from __future__ import annotations

import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import streamlit as st

# ── path setup ────────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from coperception.utils.detection_util import cal_local_mAP

from gui.engine import (
    DEFAULT_CKPT_PATH,
    DEFAULT_DATA_PATH,
    DEFAULT_STATE_PATH,
    AGENT_COLORS,
    compute_frame_ap,
    compute_running_map,
    get_frames_for_scene,
    load_dataset,
    load_model,
    load_state_index,
    run_frame_inference,
)
from gui.viz import draw_scene_overview, draw_bev_detection

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CoSelector — V2X Detection",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .main { background-color: #1a1a2e; color: #e0e0e0; }
  .stMetric label { color: #aaaaaa !important; font-size: 0.75rem; }
  div[data-testid="stSidebar"] { background-color: #0f3460; }
  h1,h2,h3 { color: #e0e0e0; }
  .section-label { color:#aaaaaa; font-size:0.78rem; font-weight:600;
                   text-transform:uppercase; letter-spacing:0.05em; margin-bottom:4px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Session-state initialisation
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULTS = {
    "last_result":       None,
    "last_key":          None,
    "det_results":       [],
    "annotations":       [],
    "history":           [],   # list of per-frame metric dicts
    "playing":           False,
    "play_frame_pos":    0,    # index into frames_in_scene list
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — controls
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📡 CoSelector")
    st.caption("V2X Cooperative Detection")
    st.divider()

    # ── Paths ─────────────────────────────────────────────────────────────────
    with st.expander("⚙️ Paths", expanded=False):
        data_path  = st.text_input("BEV data",    DEFAULT_DATA_PATH)
        state_path = st.text_input("State data",  DEFAULT_STATE_PATH)
        ckpt_path  = st.text_input("Checkpoint",  DEFAULT_CKPT_PATH)

    st.divider()

    # ── Dataset loading ───────────────────────────────────────────────────────
    try:
        _, frame_to_idx, all_frames, scenes = load_dataset(data_path)
    except Exception as e:
        st.error(f"Dataset error: {e}")
        st.stop()

    # ── Scene / Frame ─────────────────────────────────────────────────────────
    st.subheader("Scene")
    scene_id = st.selectbox("Scene ID", scenes,
                            format_func=lambda s: f"Scene {s}")

    frames_in_scene = sorted(f for s, f in all_frames if s == scene_id)

    # During playback the slider is locked to the current play position
    play_pos   = st.session_state.play_frame_pos
    play_pos   = min(play_pos, len(frames_in_scene) - 1)
    slider_val = frames_in_scene[play_pos] if st.session_state.playing else None

    frame_id = st.slider(
        "Frame",
        min_value=min(frames_in_scene),
        max_value=max(frames_in_scene),
        value=slider_val if slider_val is not None else min(frames_in_scene),
        step=1,
        disabled=st.session_state.playing,
    )
    # Sync play_frame_pos to slider when not playing
    if not st.session_state.playing and frame_id in frames_in_scene:
        st.session_state.play_frame_pos = frames_in_scene.index(frame_id)

    st.divider()

    # ── Selection strategy ────────────────────────────────────────────────────
    st.subheader("Selection")
    STRATEGY_LABELS = {
        "identity":  "All agents",
        "closest_k": "Closest K",
        "velocity":  "Fastest K",
        "heuristic": "Coverage",
        "bandwidth": "Bandwidth-aware",
        "ml_model":  "ML model",
    }
    strategy = st.selectbox("Strategy", list(STRATEGY_LABELS.keys()),
                             format_func=lambda s: STRATEGY_LABELS[s])
    k, budget_mb = 3, 2.0
    if strategy in ("closest_k", "velocity", "heuristic"):
        k = st.slider("K (agents)", 1, 5, 3)
    if strategy == "bandwidth":
        budget_mb = st.slider("Budget (MB)", 0.1, 20.0, 2.0, 0.1)

    st.divider()

    # ── Cost parameters ───────────────────────────────────────────────────────
    st.subheader("Cost parameters")
    alpha_bw     = st.slider("Bandwidth weight (α_bw)",   0.0, 2.0, 0.5, 0.05)
    alpha_lat    = st.slider("Latency weight (α_lat)",    0.0, 2.0, 0.5, 0.05)
    upload_lat   = st.slider("Upload latency/agent (ms)", 1,   200, 20)

    st.divider()

    # ── Display ───────────────────────────────────────────────────────────────
    st.subheader("Display")
    score_thresh = st.slider("Score threshold", 0.0, 1.0, 0.30, 0.05)

    st.divider()

    # ── Playback controls ─────────────────────────────────────────────────────
    st.subheader("▶ Playback")
    frame_delay = st.slider("Delay between frames (s)", 0.0, 2.0, 0.0, 0.1)

    # "Run frame" executes GPU inference for the currently selected
    # scene/frame/strategy.  Kept separate from Play so that scrubbing the
    # sliders never accidentally triggers a GPU call.
    run_frame_btn = st.button(
        "▶ Run frame", type="primary", use_container_width=True,
        disabled=st.session_state.playing,
    )

    pb_cols = st.columns(2)
    play_btn = pb_cols[0].button(
        "▶ Play", use_container_width=True,
        disabled=st.session_state.playing,
    )
    stop_btn = pb_cols[1].button(
        "⏹ Stop", use_container_width=True,
        disabled=not st.session_state.playing,
    )

    if play_btn:
        # Start playback from current slider position
        st.session_state.playing = True
        st.session_state.play_frame_pos = frames_in_scene.index(frame_id)
        # Reset accumulators for a clean playback run
        st.session_state.det_results = []
        st.session_state.annotations = []
        st.session_state.history     = []
    if stop_btn:
        st.session_state.playing = False

    if st.button("🗑 Clear history", use_container_width=True):
        st.session_state.det_results   = []
        st.session_state.annotations   = []
        st.session_state.history       = []
        st.session_state.last_result   = None
        st.session_state.last_key      = None
        st.experimental_rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Pre-warm shared resources
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Loading model & state index…"):
    try:
        load_model(ckpt_path)
        load_state_index(state_path)
    except Exception as e:
        st.error(f"Init error: {e}")
        st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Determine which frame to run
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.playing:
    play_pos = min(st.session_state.play_frame_pos, len(frames_in_scene) - 1)
    active_frame = frames_in_scene[play_pos]
else:
    active_frame = frame_id

current_key = (scene_id, active_frame, strategy, k, budget_mb,
               score_thresh, alpha_bw, alpha_lat, upload_lat)

# ─────────────────────────────────────────────────────────────────────────────
# Run inference (single frame or playback tick)
# ─────────────────────────────────────────────────────────────────────────────
run_needed = (
    st.session_state.playing   # playback auto-advances
    or run_frame_btn           # explicit single-frame trigger
    # NOTE: we intentionally do NOT auto-fire on key change so that
    # adjusting a slider/selectbox never launches a GPU call by itself.
)

if run_needed:
    with st.spinner(f"Inference scene {scene_id} frame {active_frame}…"):
        try:
            result = run_frame_inference(
                scene_id=scene_id,
                frame_id=active_frame,
                strategy=strategy,
                k=k,
                budget_mb=budget_mb,
                alpha_bw=alpha_bw,
                alpha_lat=alpha_lat,
                upload_lat_ms=float(upload_lat),
                data_path=data_path,
                state_path=state_path,
                ckpt_path=ckpt_path,
            )
            if result is not None:
                st.session_state.last_result = result
                st.session_state.last_key    = current_key

                # Accumulate mAP data
                try:
                    st.session_state.det_results, st.session_state.annotations = (
                        cal_local_mAP(
                            result["config"],
                            result["mAP_temp"],
                            st.session_state.det_results,
                            st.session_state.annotations,
                        )
                    )
                except Exception:
                    pass

                # Store per-frame metrics for graphs
                st.session_state.history.append({
                    "scene":        scene_id,
                    "frame":        active_frame,
                    "n_selected":   result["n_selected"],
                    "ap05":         result["frame_ap05"],
                    "ap07":         result["frame_ap07"],
                    "latency_ms":   result["total_latency_ms"],
                    "payload_mb":   result["payload_mb"],
                    "cost":         result["combined_cost"],
                })

                # Advance playback
                if st.session_state.playing:
                    next_pos = play_pos + 1
                    if next_pos >= len(frames_in_scene):
                        st.session_state.playing = False   # end of scene
                    else:
                        st.session_state.play_frame_pos = next_pos
                        if frame_delay > 0:
                            time.sleep(frame_delay)
                        st.experimental_rerun()

        except Exception as e:
            st.error(f"Inference error: {e}")
            import traceback
            st.code(traceback.format_exc())
            st.session_state.playing = False

result = st.session_state.last_result
if result is None:
    st.info("Use **▶ Run** (sidebar single frame) or **▶ Play** to start.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Title + playback status
# ─────────────────────────────────────────────────────────────────────────────
title_col, status_col = st.columns([3, 1])
title_col.title("📡 CoSelector — V2X Cooperative Detection")
title_col.caption(
    f"Scene **{result['scene_id']}** · Frame **{result['frame_id']}** · "
    f"Strategy: **{STRATEGY_LABELS[strategy]}**"
)

if st.session_state.playing:
    frac = play_pos / max(1, len(frames_in_scene) - 1)
    status_col.markdown(f"**▶ Playing** {play_pos+1}/{len(frames_in_scene)}")
    st.progress(frac)
elif len(st.session_state.history) > 0:
    status_col.markdown(f"**⏹ Stopped** — {len(st.session_state.history)} frames")
else:
    status_col.markdown("**⏸ Paused**")

# ─────────────────────────────────────────────────────────────────────────────
# Metrics dashboard
# ─────────────────────────────────────────────────────────────────────────────
n_sel   = result["n_selected"]
n_avail = result["n_available"]
n_hist  = len(st.session_state.history)

# Running mAP (only meaningful with >1 frame)
map05_run, map07_run = compute_running_map(
    st.session_state.det_results, st.session_state.annotations
)

st.markdown('<p class="section-label">Detection quality</p>', unsafe_allow_html=True)
qa, qb, qc, qd, qe = st.columns(5)
qa.metric("AP @ 0.5 (frame)",  f"{result['frame_ap05']:.3f}",
          help="Average Precision for this frame only")
qb.metric("AP @ 0.7 (frame)",  f"{result['frame_ap07']:.3f}")
qc.metric(f"mAP@0.5 ({n_hist} frames)", f"{map05_run:.3f}",
          help="Accumulated across all evaluated frames")
qd.metric("Pred boxes", len(result["pred_boxes"]))
qe.metric("GT boxes",   len(result["gt_boxes_phys"]))

st.markdown('<p class="section-label">Communication cost</p>', unsafe_allow_html=True)
ca, cb, cc, cd, ce = st.columns(5)
ca.metric("Selected / avail.",    f"{n_sel} / {n_avail}")
cb.metric("Payload (frame)",      f"{result['payload_mb']:.3f} MB")
cc.metric("Inference",            f"{result['inference_time_ms']:.1f} ms")
cd.metric("Total latency",        f"{result['total_latency_ms']:.1f} ms",
          help="Upload latency + inference time")
ce.metric("Combined cost",        f"{result['combined_cost']:.4f}",
          help=f"α_bw={alpha_bw}×BW + α_lat={alpha_lat}×(latency/1000)")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Selected-agent info cards
# ─────────────────────────────────────────────────────────────────────────────
selected_agent_ids = result["selected_indices"]
metas_by_id = {m.agent_id: m for m in result["metas"]}
sel_cols = st.columns(max(1, len(selected_agent_ids)))
for col, aid in zip(sel_cols, selected_agent_ids):
    color = AGENT_COLORS[aid % len(AGENT_COLORS)]
    pos   = result["vehicle_positions"].get(aid, (0, 0))
    meta  = metas_by_id.get(aid)
    speed_str = f"{meta.speed:.1f} m/s" if meta else "n/a"
    dist  = (pos[0]**2 + pos[1]**2) ** 0.5
    col.markdown(
        f"""<div style='background:{color}22;border:1px solid {color};
        border-radius:8px;padding:8px;text-align:center;'>
        <b style='color:{color}'>Agent {aid}</b><br/>
        <span style='font-size:0.75rem'>({pos[0]:.1f}, {pos[1]:.1f}) m</span><br/>
        <span style='font-size:0.75rem'>dist {dist:.1f} m · {speed_str}</span>
        </div>""",
        unsafe_allow_html=True,
    )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Visualisation panels
# ─────────────────────────────────────────────────────────────────────────────
col_scene, col_bev = st.columns(2, gap="medium")

with col_scene:
    fig_scene = draw_scene_overview(
        vehicle_positions=result["vehicle_positions"],
        selected_indices=result["selected_indices"],
        metas=result["metas"],
        fig_size=(5.5, 5.5),
    )
    st.pyplot(fig_scene, use_container_width=True)
    plt.close(fig_scene)

with col_bev:
    fig_bev = draw_bev_detection(
        bev_img=result["bev_img"],
        pred_boxes=result["pred_boxes"],
        gt_boxes_phys=result["gt_boxes_phys"],
        config=result["config"],
        pred_scores=result["pred_scores"],
        score_threshold=score_thresh,
        fig_size=(5.5, 5.5),
    )
    st.pyplot(fig_bev, use_container_width=True)
    plt.close(fig_bev)

# ─────────────────────────────────────────────────────────────────────────────
# Per-frame graphs (shown once there is playback history)
# ─────────────────────────────────────────────────────────────────────────────
history = st.session_state.history
if len(history) >= 2:
    st.divider()
    st.markdown("### 📈 Playback history")

    frames_x  = [h["frame"]      for h in history]
    ap05_y    = [h["ap05"]       for h in history]
    n_sel_y   = [h["n_selected"] for h in history]
    latency_y = [h["latency_ms"] for h in history]
    payload_y = [h["payload_mb"] for h in history]
    cost_y    = [h["cost"]       for h in history]

    fig_hist, axes = plt.subplots(
        2, 2, figsize=(12, 6),
        facecolor="#1a1a2e",
        constrained_layout=True,
    )
    fig_hist.patch.set_facecolor("#1a1a2e")

    _PANEL_STYLE = dict(
        facecolor="#16213e",
        tick_params=dict(colors="#888888", labelsize=7),
        spine_color="#333355",
        label_color="#aaaaaa",
    )

    def _style_ax(ax, title, ylabel, xlabel="Frame"):
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="#888888", labelsize=7)
        for s in ax.spines.values():
            s.set_edgecolor("#333355")
        ax.set_title(title, color="#dddddd", fontsize=9, pad=4)
        ax.set_xlabel(xlabel, color="#aaaaaa", fontsize=8)
        ax.set_ylabel(ylabel, color="#aaaaaa", fontsize=8)
        ax.grid(color="#0f3460", linewidth=0.4)

    # ── 1. Selected agents ─────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.bar(frames_x, n_sel_y, color="#3498db", alpha=0.8, width=0.6)
    ax.set_ylim(0, 6)
    _style_ax(ax, "Selected agents / frame", "# agents")

    # ── 2. AP@0.5 per frame ────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(frames_x, ap05_y, color="#2ecc71", linewidth=1.5, marker="o",
            markersize=3)
    ax.fill_between(frames_x, ap05_y, alpha=0.15, color="#2ecc71")
    ax.set_ylim(0, 1.05)
    # Draw accumulated mAP as horizontal reference
    if map05_run > 0:
        ax.axhline(map05_run, color="#f1c40f", linewidth=1, linestyle="--",
                   label=f"mAP@0.5 = {map05_run:.3f}")
        ax.legend(fontsize=7, facecolor="#0f3460", labelcolor="#cccccc",
                  edgecolor="#333355")
    _style_ax(ax, "AP @ 0.5 per frame", "AP")

    # ── 3. Total latency ───────────────────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(frames_x, latency_y, color="#e74c3c", linewidth=1.5, marker="o",
            markersize=3)
    ax.fill_between(frames_x, latency_y, alpha=0.15, color="#e74c3c")
    _style_ax(ax, "Total latency / frame", "ms")

    # ── 4. Payload & combined cost ─────────────────────────────────────────
    ax = axes[1, 1]
    ax2 = ax.twinx()
    ax.bar(frames_x, payload_y, color="#9b59b6", alpha=0.6, width=0.6,
           label="Payload (MB)")
    ax2.plot(frames_x, cost_y, color="#e67e22", linewidth=1.5, marker="s",
             markersize=3, label="Combined cost")
    ax.set_ylabel("Payload (MB)", color="#9b59b6", fontsize=8)
    ax2.set_ylabel("Combined cost", color="#e67e22", fontsize=8)
    ax2.tick_params(colors="#888888", labelsize=7)
    ax2.spines["right"].set_edgecolor("#333355")
    # legend
    lines = [
        matplotlib.patches.Patch(color="#9b59b6", label="Payload (MB)"),
        matplotlib.lines.Line2D([0], [0], color="#e67e22", label="Combined cost"),
    ]
    ax.legend(handles=lines, fontsize=7, facecolor="#0f3460",
              labelcolor="#cccccc", edgecolor="#333355")
    _style_ax(ax, "Bandwidth & cost / frame", "Payload (MB)")

    st.pyplot(fig_hist, use_container_width=True)
    plt.close(fig_hist)

    # ── Summary table ──────────────────────────────────────────────────────
    with st.expander("📋 Frame-by-frame table", expanded=False):
        import pandas as pd
        df = pd.DataFrame(history)
        df.columns = ["Scene", "Frame", "N selected", "AP@0.5", "AP@0.7",
                      "Latency (ms)", "Payload (MB)", "Combined cost"]
        st.dataframe(df.style.format({
            "AP@0.5": "{:.3f}", "AP@0.7": "{:.3f}",
            "Latency (ms)": "{:.1f}", "Payload (MB)": "{:.3f}",
            "Combined cost": "{:.4f}",
        }), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Strategy explanation
# ─────────────────────────────────────────────────────────────────────────────
STRATEGY_HELP = {
    "identity":  "**All agents** — all available vehicles upload their BEV. Highest bandwidth; mAP upper-bound for selection.",
    "closest_k": "**Closest K** — selects the K vehicles nearest to the RSU using T_{j→RSU} translation. Simple, low-latency baseline.",
    "velocity":  "**Fastest K** — selects K highest-speed vehicles. Fast movers observe the most scene change and provide fresh viewpoints.",
    "heuristic": "**Coverage** — greedy angular-diversity: each next pick maximises the min bearing angle from already-selected vehicles.",
    "bandwidth": "**Bandwidth-aware** — greedy budget-constrained: admits the smallest-payload vehicles first until the byte budget is exhausted.",
    "ml_model":  "**ML model** — MLP scores each vehicle; those above threshold are selected. Falls back to identity if no model is loaded.",
}
with st.expander("ℹ️ Strategy explanation", expanded=False):
    st.markdown(STRATEGY_HELP.get(strategy, ""))
