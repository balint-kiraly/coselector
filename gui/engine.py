"""
Inference engine for the CoSelector GUI.

Handles model loading (cached), dataset loading (cached), state-index loading
(cached), and per-frame inference triggered by Streamlit UI interactions.
"""

from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np
import torch
import torch.optim as optim
from torch import nn

# ── ensure coselector root is on sys.path ────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st

from coperception.configs import Config, ConfigGlobal
from coperception.datasets import V2XSimDet
from coperception.models.det import FaFNet
from coperception.utils.CoDetModule import FaFModule
from coperception.utils.loss import (
    SoftmaxFocalClassificationLoss,
    WeightedSmoothL1LocalizationLoss,
)

from coperception.utils.detection_util import cal_local_mAP
from coperception.utils.mean_ap import eval_map

from data_utils.build_state_features import build_state_features
from data_utils.cost_model import CostConfig, CostScenario, compute_cost
from data_utils.state_index import StateIndex
from selection.bev_builder import assemble_detection_inputs
from selection.policy import select_agents_from_metadata, SelectionMethod

# ── default paths ─────────────────────────────────────────────────────────────
DEFAULT_DATA_PATH  = "/mnt/10TB/balintkiraly/created_data/V2X-Sim-det/test"
DEFAULT_STATE_PATH = "/mnt/10TB/balintkiraly/created_data/V2X-Sim-States"
DEFAULT_CKPT_PATH  = (
    "/home/bkiraly/coperception/tools/det/checkpoints"
    "/upperbound/no_rsu/epoch_100.pth"
)

AGENT_IDX_RANGE = range(0, 6)  # RSU (0) + vehicles (1-5)

# ── Agent colour palette (index = agent_id) ───────────────────────────────────
AGENT_COLORS = ["gold", "#e74c3c", "#2ecc71", "#3498db", "#9b59b6", "#e67e22"]


# ─────────────────────────────────────────────────────────────────────────────
# Cached resource loaders
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading detector model…")
def load_model(ckpt_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config("train", binary=True, only_det=True)
    config_global = ConfigGlobal("train", binary=True, only_det=True)
    config.flag = "lowerbound"
    config.split = "test"

    model = FaFNet(config, layer=3, kd_flag=0, num_agent=1)
    model = nn.DataParallel(model)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = {
        "cls": SoftmaxFocalClassificationLoss(),
        "loc": WeightedSmoothL1LocalizationLoss(),
    }
    fafmodule = FaFModule(model, model, config, optimizer, criterion, 0)

    if not os.path.isfile(ckpt_path):
        st.error(f"Checkpoint not found: {ckpt_path}")
        st.stop()

    ckpt = torch.load(ckpt_path, map_location="cpu")
    fafmodule.model.load_state_dict(ckpt["model_state_dict"])
    fafmodule.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    fafmodule.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    fafmodule.model.eval()
    return fafmodule, config, device


@st.cache_resource(show_spinner="Loading dataset…")
def load_dataset(data_path: str):
    config = Config("train", binary=True, only_det=True)
    config_global = ConfigGlobal("train", binary=True, only_det=True)

    ds = V2XSimDet(
        dataset_roots=[f"{data_path}/agent{i}" for i in AGENT_IDX_RANGE],
        config=config,
        config_global=config_global,
        split="val",
        val=True,
        bound="lowerbound",
        kd_flag=0,
        rsu=True,
    )

    # Build (scene_id, frame_id) → dataset index.
    # We scan the filesystem to avoid loading all BEV data.
    agent0_dir = os.path.join(data_path, "agent0")
    frames: list[tuple[int, int]] = []
    for name in sorted(os.listdir(agent0_dir)):
        if not os.path.isdir(os.path.join(agent0_dir, name)):
            continue
        parts = name.split("_")
        if len(parts) == 2:
            try:
                frames.append((int(parts[0]), int(parts[1])))
            except ValueError:
                pass
    frames.sort()
    frame_to_idx = {sf: i for i, sf in enumerate(frames)}
    scenes = sorted({s for s, _ in frames})
    return ds, frame_to_idx, frames, scenes


@st.cache_resource(show_spinner="Loading state index…")
def load_state_index(state_path: str):
    return StateIndex.from_fs(state_path)


def get_frames_for_scene(frames: list, scene_id: int) -> list[int]:
    """Return sorted frame IDs available for a given scene."""
    return sorted(f for s, f in frames if s == scene_id)


# ─────────────────────────────────────────────────────────────────────────────
# Per-frame inference
# ─────────────────────────────────────────────────────────────────────────────

def compute_frame_ap(config, mAP_temp: dict) -> tuple[float, float]:
    """
    Per-frame AP@0.5 and AP@0.7.
    Runs eval_map on a single frame's detection results.
    Returns (ap05, ap07), both 0.0 if no GT boxes.
    """
    try:
        dr, ann = cal_local_mAP(config, mAP_temp, [], [])
        if not dr or not ann:
            return 0.0, 0.0
        ap05, _ = eval_map(dr, ann, iou_thr=0.5, scale_ranges=None,
                           dataset=None, logger=None)
        ap07, _ = eval_map(dr, ann, iou_thr=0.7, scale_ranges=None,
                           dataset=None, logger=None)
        return float(ap05), float(ap07)
    except Exception:
        return 0.0, 0.0


def run_frame_inference(
    scene_id: int,
    frame_id: int,
    strategy: str,
    k: int = 3,
    budget_mb: float = 2.0,
    alpha_bw: float = 0.5,
    alpha_lat: float = 0.5,
    upload_lat_ms: float = 20.0,
    data_path: str = DEFAULT_DATA_PATH,
    state_path: str = DEFAULT_STATE_PATH,
    ckpt_path: str = DEFAULT_CKPT_PATH,
) -> dict | None:
    """
    Run one frame through the full pipeline:
      1. Load frame BEV data from dataset.
      2. Build state features + run selection policy.
      3. Fuse selected BEVs into RSU frame.
      4. Run FaFNet forward pass.
      5. Return everything needed for visualisation.
    """
    fafmodule, config, device = load_model(ckpt_path)
    ds, frame_to_idx, frames, scenes = load_dataset(data_path)
    state_index = load_state_index(state_path)

    idx = frame_to_idx.get((scene_id, frame_id))
    if idx is None:
        return None

    # ── unpack dataset item ───────────────────────────────────────────────────
    # ds[idx] = list[6 agents] where each agent = tuple(12 fields)
    # Unlike the DataLoader path, ds[idx] returns numpy arrays — convert them
    # to torch.Tensor so bev_builder.py (which calls .dim()) works correctly.
    item = ds[idx]
    fields = list(zip(*item))         # transpose → [field][agent]

    def _to_tensor(x):
        if isinstance(x, (list, tuple)) and len(x) == 0:
            return x
        try:
            t = torch.as_tensor(np.asarray(x), dtype=torch.float32)
            # Dataset returns (T, H, W, C) = (1, 256, 256, 13).
            # DataLoader would add batch dim → (1, 1, 256, 256, 13) (5D).
            # bev_builder._ensure_bev_channels_first only activates on 5D tensors,
            # so we must add the missing batch dim to reproduce DataLoader behaviour.
            if t.dim() == 4:
                t = t.unsqueeze(0)   # (1, 1, 256, 256, 13)
            return t
        except Exception:
            return x

    padded_voxel_point_list    = [_to_tensor(f) for f in fields[0]]  # BEV (1,1, H, W, C)
    teacher_bev_list           = list(fields[1])   # [] for lowerbound
    label_one_hot_list         = list(fields[2])
    reg_target_list            = list(fields[3])
    reg_loss_mask_list         = list(fields[4])
    anchors_map_list           = list(fields[5])
    vis_maps_list              = list(fields[6])
    gt_max_iou_raw             = list(fields[7])   # [{'gt_box': (N,4)}]
    filenames_list             = list(fields[8])
    target_agent_id_list       = list(fields[9])
    num_agent_list             = list(fields[10])
    trans_matrices_list        = list(fields[11])  # (6, 4, 4) per agent

    # ── RSU reference (agent 0) ───────────────────────────────────────────────
    rsu_gt_max_iou    = gt_max_iou_raw[0]
    rsu_label_one_hot = label_one_hot_list[0]
    rsu_reg_target    = reg_target_list[0]
    rsu_reg_loss_mask = reg_loss_mask_list[0]
    rsu_anchors_map   = anchors_map_list[0]
    rsu_vis_maps      = vis_maps_list[0]

    # T_{j → RSU}, shape (6, 4, 4)
    _ref_table = torch.as_tensor(
        np.array(trans_matrices_list[0]), dtype=torch.float32
    )

    # ── agent selection ───────────────────────────────────────────────────────
    try:
        state_feats, metas = build_state_features(
            state_index, scene_id, frame_id, return_meta=True
        )
    except Exception:
        metas = []
        state_feats = torch.zeros((0, 14))

    state_feats = state_feats.to(device)
    agent_id_to_idx = {aid: i for i, aid in enumerate(AGENT_IDX_RANGE)}

    sel_kwargs: dict = {"rsu_trans": _ref_table, "metas": metas}
    if strategy in ("closest_k", "heuristic"):
        sel_kwargs["K"] = k
    elif strategy == "velocity":
        sel_kwargs["K"] = k
    elif strategy == "bandwidth":
        sel_kwargs["budget_bytes"] = int(budget_mb * 1024 * 1024)
        sel_kwargs["bev_list"] = padded_voxel_point_list

    selected_state_indices = select_agents_from_metadata(
        state_feats, method=SelectionMethod(strategy), **sel_kwargs
    )
    # Map state-feature indices → dataset agent indices
    selected_indices = [
        agent_id_to_idx[metas[i].agent_id]
        for i in selected_state_indices
        if i < len(metas) and metas[i].agent_id in agent_id_to_idx
    ]

    # Fallback: if state features are absent, use all non-RSU agents
    if not selected_indices:
        selected_indices = [
            i for i in range(1, 6)
            if i < len(padded_voxel_point_list)
        ]

    def _sel(lst):
        return [lst[i] for i in selected_indices]

    # Per-selected-agent T_{j → RSU}
    trans_sel = [_ref_table[j] for j in selected_indices]

    # ── assemble fused detection inputs ──────────────────────────────────────
    assemble_labels      = [rsu_label_one_hot]  + list(_sel(label_one_hot_list))
    assemble_reg_target  = [rsu_reg_target]     + list(_sel(reg_target_list))
    assemble_reg_mask    = [rsu_reg_loss_mask]  + list(_sel(reg_loss_mask_list))
    assemble_anchors_map = [rsu_anchors_map]    + list(_sel(anchors_map_list))
    assemble_vis_maps    = [rsu_vis_maps]       + list(_sel(vis_maps_list))

    # ── BEV payload size (bytes for one agent's BEV) ─────────────────────────
    bev_byte_size = 0
    try:
        sample_bev = padded_voxel_point_list[selected_indices[0]]
        bev_byte_size = int(
            torch.as_tensor(sample_bev).numel() * torch.as_tensor(sample_bev).element_size()
        )
    except Exception:
        pass

    with torch.no_grad():
        data = assemble_detection_inputs(
            config,
            _sel(padded_voxel_point_list),
            _sel(teacher_bev_list),
            assemble_labels,
            assemble_reg_target,
            assemble_reg_mask,
            assemble_anchors_map,
            assemble_vis_maps,
            _sel(target_agent_id_list),
            _sel(num_agent_list),
            trans_sel,
            device,
        )

        # Timed forward pass
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _, _, _, result = fafmodule.predict_all(data, 1, num_agent=1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        inference_time_ms = (time.perf_counter() - t0) * 1000.0

    # Release GPU tensors immediately — prevents CUDA memory accumulation
    # across playback frames and ensures a clean state if an error occurs.
    # BEV numpy copy is taken below before this point; result is CPU after
    # predict_all decoding, so nothing useful is lost.
    try:
        bev_np_raw = data["bev_seq"].squeeze().cpu().numpy()
        del data
        if device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception:
        bev_np_raw = None

    # ── communication cost (custom weights from UI) ───────────────────────────
    cost_cfg = CostConfig(
        alpha_bw=alpha_bw,
        alpha_lat=alpha_lat,
        lat_upload_per_agent_ms=upload_lat_ms,
    )
    comm_cost = compute_cost(
        n_selected=len(selected_indices),
        config=cost_cfg,
        bev_size_bytes=bev_byte_size,
        inference_time_ms=inference_time_ms,
    )

    # ── vehicle positions in RSU frame ────────────────────────────────────────
    # Translation component of T_{j → RSU} = position of agent j's origin in RSU frame
    ref_np = _ref_table.numpy()
    vehicle_positions: dict[int, tuple[float, float]] = {}
    for j in range(ref_np.shape[0]):
        vehicle_positions[j] = (float(ref_np[j, 0, 3]), float(ref_np[j, 1, 3]))

    # ── decode GT boxes ───────────────────────────────────────────────────────
    gt_boxes_phys = _decode_gt_boxes(
        rsu_gt_max_iou, rsu_anchors_map, rsu_reg_target, config
    )

    # ── decode predicted boxes ────────────────────────────────────────────────
    # result structure: [agent][tuple_idx][batch][pred_len][detection_dict]
    # result[0]       = tuple (cls/loc output)
    # result[0][0]    = list over batch (len=1)
    # result[0][0][0] = list over pred_len (len=1)
    # result[0][0][0][0] = {'pred': (N,1,4,2), 'score': (N,), 'selected_idx': ...}
    pred_boxes: list[np.ndarray] = []
    pred_scores: list[float] = []
    try:
        if (result and len(result) > 0
                and len(result[0]) > 0
                and len(result[0][0]) > 0
                and len(result[0][0][0]) > 0):
            pd = result[0][0][0][0]  # dict
            if isinstance(pd, dict) and "pred" in pd and len(pd["pred"]) > 0:
                preds  = pd["pred"]                   # (N, pred_len, 4, 2)
                scores = pd.get("score", [])
                if hasattr(preds,  "cpu"): preds  = preds.cpu().numpy()
                if hasattr(scores, "cpu"): scores = scores.cpu().numpy()
                for i, box in enumerate(preds):
                    # box shape: (pred_len, 4, 2) → take step 0
                    corners = box[0] if box.ndim == 3 else box   # (4, 2)
                    pred_boxes.append(corners)
                    pred_scores.append(float(scores[i]) if i < len(scores) else 1.0)
    except (IndexError, TypeError, KeyError):
        pass  # no predictions

    # ── fused BEV image for display ───────────────────────────────────────────
    # bev_np_raw was extracted and data deleted right after inference
    # to free GPU memory.  Fall back to a blank canvas if something went wrong.
    if bev_np_raw is not None:
        bev_img = bev_np_raw.max(axis=-1) if bev_np_raw.ndim == 3 else bev_np_raw
    else:
        bev_img = np.zeros((256, 256), dtype=np.float32)

    # ── per-frame mAP accumulation data (for cal_local_mAP / eval_map) ───────
    # Prepare `temp` dict in the format cal_local_mAP expects.
    # result_pred = the innermost list-of-dicts: [{'pred':..,'score':..}]
    result_pred = []
    try:
        if (result and len(result[0]) > 0
                and len(result[0][0]) > 0
                and len(result[0][0][0]) > 0):
            result_pred = result[0][0][0]   # list of prediction dicts
    except (IndexError, TypeError):
        pass

    rsu_gt_indices = rsu_gt_max_iou
    if isinstance(rsu_gt_max_iou, list) and len(rsu_gt_max_iou) > 0:
        entry = rsu_gt_max_iou[0]
        if isinstance(entry, dict) and "gt_box" in entry:
            rsu_gt_indices = entry["gt_box"]

    mAP_temp = {
        "result":      result_pred,
        "reg_targets": np.asarray(rsu_reg_target),    # (H, W, n_anch, 1, 6)
        "anchors_map": np.asarray(rsu_anchors_map),   # (H, W, n_anch, 6)
        "gt_max_iou":  rsu_gt_indices,                # (N_gt, 4) int64
    }

    # ── per-frame AP ──────────────────────────────────────────────────────────
    frame_ap05, frame_ap07 = compute_frame_ap(config, mAP_temp)

    return {
        "vehicle_positions":  vehicle_positions,   # {agent_id: (x_rsu, y_rsu)}
        "selected_indices":   selected_indices,    # list of dataset agent indices
        "metas":              metas,               # list of AgentMeta
        "gt_boxes_phys":      gt_boxes_phys,       # list of (4, 2) corner arrays [m]
        "pred_boxes":         pred_boxes,          # list of (4, 2) corner arrays [m]
        "pred_scores":        pred_scores,         # list of float
        "bev_img":            bev_img,             # (H, W) float32
        "n_available":        len(metas),
        "n_selected":         len(selected_indices),
        "config":             config,
        # ── metrics ──────────────────────────────────────────────────────────
        "inference_time_ms":  inference_time_ms,
        "payload_mb":         comm_cost.bandwidth_MB,
        "upload_latency_ms":  comm_cost.latency_upload_ms,
        "total_latency_ms":   comm_cost.latency_total_ms,
        "combined_cost":      comm_cost.combined_cost,
        "frame_ap05":         frame_ap05,          # AP@0.5 for this frame only
        "frame_ap07":         frame_ap07,
        "mAP_temp":           mAP_temp,            # for cal_local_mAP in app
        "scene_id":           scene_id,
        "frame_id":           frame_id,
    }


def compute_running_map(det_results: list, annotations: list) -> tuple[float, float]:
    """
    Compute mAP@0.5 and mAP@0.7 from accumulated detection results.
    Returns (mAP_05, mAP_07).  Returns (0, 0) if not enough data.
    """
    if not det_results or not annotations:
        return 0.0, 0.0
    try:
        map05, _ = eval_map(det_results, annotations, iou_thr=0.5,
                            scale_ranges=None, dataset=None, logger=None)
        map07, _ = eval_map(det_results, annotations, iou_thr=0.7,
                            scale_ranges=None, dataset=None, logger=None)
        return float(map05), float(map07)
    except Exception:
        return 0.0, 0.0


# ─────────────────────────────────────────────────────────────────────────────
# GT box decoding
# ─────────────────────────────────────────────────────────────────────────────

def _decode_gt_boxes(
    gt_max_iou_raw,
    anchors_map,
    reg_targets,
    config,
) -> list[np.ndarray]:
    """
    Decode GT box sparse indices into (4, 2) corner arrays in physical metres.

    ``gt_max_iou_raw`` is the raw entry from the dataset:
      [{'gt_box': ndarray (N, 4)}]
    Each row of gt_box = (row, col, anchor_idx, pred_step_idx) used to index
    into anchors_map and reg_targets.
    """
    from coperception.utils.detection_util import bev_box_decode_np
    from coperception.utils.detection_util import center_to_corner_box2d

    # Unwrap list wrapper
    gt_indices = gt_max_iou_raw
    if isinstance(gt_max_iou_raw, list) and len(gt_max_iou_raw) > 0:
        entry = gt_max_iou_raw[0]
        if isinstance(entry, dict) and "gt_box" in entry:
            gt_indices = entry["gt_box"]

    if not hasattr(gt_indices, "__len__") or len(gt_indices) == 0:
        return []

    anchors_np = np.asarray(anchors_map, dtype=np.float64)
    reg_np     = np.asarray(reg_targets, dtype=np.float64)

    corners_list: list[np.ndarray] = []
    for idx_row in gt_indices:
        try:
            anchor   = anchors_np[tuple(idx_row[:-1])]
            enc_box  = reg_np[tuple(idx_row[:-1]) + (0,)]
            dec_box  = bev_box_decode_np(enc_box, anchor)
            corner   = center_to_corner_box2d(
                np.asarray([dec_box[:2]]),
                np.asarray([dec_box[2:4]]),
                np.asarray([dec_box[4:]]),
            )[0]                                   # (4, 2)
            corners_list.append(corner)
        except Exception:
            pass
    return corners_list
