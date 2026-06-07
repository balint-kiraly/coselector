"""
REINFORCE training for the learned agent selector.

Usage example:
    python -m selection.train_selector \
        --ckpt /path/to/det.pth \
        --data_det /mnt/.../V2X-Sim-det/train \
        --data_state /mnt/.../V2X-Sim-States \
        --save_dir /tmp/selector_run \
        --train_scene_begin 0 --train_scene_end 100 \
        --val_scene_begin 0 --val_scene_end 100 \
        --rl_epochs 20 --curriculum_epochs 3 \
        --lambda_cost 0.4 --lr 1e-3
"""

import argparse
import csv
import gc
import os
import sys
import time

import torch
import torch.nn as nn
from torch import optim
from torch.distributions import Bernoulli
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from coperception.configs import Config, ConfigGlobal
from coperception.datasets import V2XSimDet
from coperception.models.det import FaFNet
from coperception.utils.CoDetModule import FaFModule, cal_local_mAP
from coperception.utils.loss import (
    SoftmaxFocalClassificationLoss,
    WeightedSmoothL1LocalizationLoss,
)
from coperception.utils.mean_ap import eval_map

from data_utils.build_state_features import build_state_features, ML_START, ML_DIM
from data_utils.state_index import StateIndex
from selection.bev_builder import assemble_detection_inputs
from selection.models import AgentSelectorMLP


# ── Constants ──────────────────────────────────────────────────────────────────
FEATURE_DIM = ML_DIM   # 12 — engineered features fed to the MLP
AGENT_START = 1
AGENT_END = 6          # exclusive; vehicles 1-5
AGENT_IDX_RANGE = range(AGENT_START, AGENT_END)
N_MAX = len(AGENT_IDX_RANGE)   # 5


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="REINFORCE selector training")

    # data
    p.add_argument("--ckpt", required=True, help="Frozen detection model checkpoint.")
    p.add_argument("--data_det", required=True,
                   help="Path to precomputed BEV data (train split directory).")
    p.add_argument("--data_state", required=True,
                   help="Path to state-feature JSON tree.")
    p.add_argument("--save_dir", required=True,
                   help="Directory for checkpoints and metrics CSV.")

    # scene ranges
    p.add_argument("--train_scene_begin", type=int, default=0)
    p.add_argument("--train_scene_end", type=int, default=100)
    p.add_argument("--val_scene_begin", type=int, default=0)
    p.add_argument("--val_scene_end", type=int, default=100)

    # training hypers
    p.add_argument("--rl_epochs", type=int, default=20)
    p.add_argument("--curriculum_epochs", type=int, default=3,
                   help="Epochs with lambda_cost=0 to bootstrap detection quality.")
    p.add_argument("--lambda_cost", type=float, default=0.4,
                   help="Cost penalty weight (activated after curriculum).")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--frames_per_epoch", type=int, default=0,
                   help="Subsample training frames per epoch (0 = use all).")
    p.add_argument("--num_workers", type=int, default=1)

    # validation
    p.add_argument("--val_every", type=int, default=3,
                   help="Validate every N epochs.")
    p.add_argument("--data_val", type=str, default="",
                   help="Val BEV data directory. Defaults to <data_det>/../val.")
    p.add_argument("--sel_threshold", type=float, default=0.5,
                   help="Greedy selection threshold used during validation.")

    # cache
    p.add_argument("--cache_dir", type=str, default="",
                   help="Directory for persisting pre-training frame caches "
                        "(q_lower/q_upper/features).  The file is named "
                        "selector_cache_s{begin}-{end}.pt automatically.  "
                        "Leave empty to disable persistence.")

    # logging
    p.add_argument("--tensorboard", action="store_true",
                   help="Log scalars to TensorBoard in save_dir.")

    return p.parse_args()


# ── Detector utilities ──────────────────────────────────────────────────────────

def _build_fafmodule(config, ckpt_path: str, device: torch.device) -> FaFModule:
    """Load the frozen FaFNet detector. Always uses num_agent=1 (fused BEV mode)."""
    model = FaFNet(config, kd_flag=False, num_agent=1)
    model = nn.DataParallel(model)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = {
        "cls": SoftmaxFocalClassificationLoss(),
        "loc": WeightedSmoothL1LocalizationLoss(),
    }
    fafmodule = FaFModule(model, model, config, optimizer, criterion, kd_flag=0)

    ckpt = torch.load(ckpt_path, map_location=device)
    fafmodule.model.load_state_dict(ckpt["model_state_dict"])
    fafmodule.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    fafmodule.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    print(f"Loaded frozen detector from {ckpt_path} (epoch {ckpt['epoch']})")
    return fafmodule


def _agent_id_to_dataset_idx() -> dict:
    return {aid: idx for idx, aid in enumerate(AGENT_IDX_RANGE)}


def _ref_table_to_trans(trans_matrices_list) -> torch.Tensor:
    """Extract the (N, 4, 4) transform table from a DataLoader batch item."""
    ref = torch.as_tensor(trans_matrices_list[0], dtype=torch.float32)
    # Shape: (1, N, 4, 4) or (N, 4, 4)
    while ref.dim() > 3:
        ref = ref.squeeze(0)
    return ref  # (N, 4, 4)


def _run_detector(fafmodule, config, selected_dataset_indices,
                  pvp_list, pvp_teacher_list, label_list, reg_target_list,
                  reg_loss_mask_list, anchors_list, vis_maps_list,
                  target_agent_id_list, num_agent_list,
                  trans_table: torch.Tensor, device: torch.device):
    """Fuse selected BEVs and run the frozen detector. Returns det_loss scalar."""
    def _sel(lst, indices):
        return [lst[i] for i in indices]

    data = assemble_detection_inputs(
        config,
        _sel(pvp_list, selected_dataset_indices),
        _sel(pvp_teacher_list, selected_dataset_indices),
        _sel(label_list, selected_dataset_indices),
        _sel(reg_target_list, selected_dataset_indices),
        _sel(reg_loss_mask_list, selected_dataset_indices),
        _sel(anchors_list, selected_dataset_indices),
        _sel(vis_maps_list, selected_dataset_indices),
        _sel(target_agent_id_list, selected_dataset_indices),
        _sel(num_agent_list, selected_dataset_indices),
        [trans_table[j] for j in selected_dataset_indices],
        device,
    )
    with torch.no_grad():
        det_loss, _, _, _ = fafmodule.predict_all(data, batch_size=1, num_agent=1)
    return det_loss  # predict_all already returns loss.item() — a Python float


# ── Pre-training cache (Task 3) ────────────────────────────────────────────────

def build_frame_cache(
    loader: DataLoader,
    state_index: StateIndex,
    fafmodule: FaFModule,
    config,
    device: torch.device,
) -> dict:
    """
    Iterate all training frames once and cache, for each frame:
      - q_lower  : detector quality (-loss) with only the nearest single agent
      - q_upper  : detector quality (-loss) with all agents
      - features : (N, 12) float32 tensor (pre-built state features)
      - N        : number of available agents
    Returns dict keyed by (scene_id, frame_id).
    """
    aid_to_didx = _agent_id_to_dataset_idx()
    cache = {}
    print("Building pre-training cache (q_lower / q_upper)…")
    fafmodule.model.eval()

    for batch in tqdm(loader, desc="Cache"):
        (
            pvp_list,
            pvp_teacher_list,
            label_list,
            reg_target_list,
            reg_loss_mask_list,
            anchors_list,
            vis_maps_list,
            _gt_max_iou,
            filenames,
            target_agent_id_list,
            num_agent_list,
            trans_matrices_list,
        ) = zip(*batch)

        pvp_list = list(pvp_list)
        pvp_teacher_list = list(pvp_teacher_list)
        label_list = list(label_list)
        reg_target_list = list(reg_target_list)
        reg_loss_mask_list = list(reg_loss_mask_list)
        anchors_list = list(anchors_list)
        vis_maps_list = list(vis_maps_list)
        target_agent_id_list = list(target_agent_id_list)
        num_agent_list = list(num_agent_list)

        filename0 = filenames[0][0][0]
        parts = filename0.split(os.sep)
        scene_id, frame_id = map(int, parts[-2].split("_"))

        feats26, metas = build_state_features(
            state_index, scene_id, frame_id, return_meta=True
        )                                            # (N, 26) raw + engineered
        if feats26.shape[0] == 0:
            continue
        ml_feats = feats26[:, ML_START:]             # (N, 12) engineered slice

        # Map feature rows → dataset indices
        feat_to_didx = [aid_to_didx.get(m.agent_id) for m in metas]
        valid_pairs = [(fi, di) for fi, di in enumerate(feat_to_didx) if di is not None]
        if not valid_pairs:
            continue

        trans_table = _ref_table_to_trans(trans_matrices_list)
        all_dataset_indices = list(range(len(AGENT_IDX_RANGE)))

        # q_upper: all agents
        try:
            loss_upper = _run_detector(
                fafmodule, config, all_dataset_indices,
                pvp_list, pvp_teacher_list, label_list, reg_target_list,
                reg_loss_mask_list, anchors_list, vis_maps_list,
                target_agent_id_list, num_agent_list, trans_table, device,
            )
        except Exception:
            continue

        # q_lower: nearest single agent (normalised range = ml_feats[:, 2])
        nearest_feat_idx = int(ml_feats[:, 2].argmin().item())
        nearest_dataset_idx = feat_to_didx[nearest_feat_idx]
        if nearest_dataset_idx is None:
            nearest_dataset_idx = valid_pairs[0][1]
        try:
            loss_lower = _run_detector(
                fafmodule, config, [nearest_dataset_idx],
                pvp_list, pvp_teacher_list, label_list, reg_target_list,
                reg_loss_mask_list, anchors_list, vis_maps_list,
                target_agent_id_list, num_agent_list, trans_table, device,
            )
        except Exception:
            loss_lower = loss_upper

        cache[(scene_id, frame_id)] = {
            "q_lower": -loss_lower,
            "q_upper": -loss_upper,
            "features": ml_feats,       # (N, 12) engineered ML features
            "feat_to_didx": feat_to_didx,
            "N": ml_feats.shape[0],
        }

    print(f"Cache built: {len(cache)} frames.")
    if cache:
        spans = [v["q_upper"] - v["q_lower"] for v in cache.values()]
        spans.sort()
        n = len(spans)
        print(f"  q_upper-q_lower span: min={spans[0]:.4f}  "
              f"p25={spans[n//4]:.4f}  median={spans[n//2]:.4f}  "
              f"p75={spans[3*n//4]:.4f}  max={spans[-1]:.4f}")
    return cache


# ── Checkpoint helpers (Task 5) ────────────────────────────────────────────────

def _save_checkpoint(path: str, state: dict) -> None:
    """Atomic checkpoint write: write to .tmp then rename."""
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def _load_checkpoint(path: str, selector, optimizer) -> tuple:
    """Load checkpoint. Returns (start_epoch, ema_mean, ema_var, best_val_map)."""
    ckpt = torch.load(path, map_location="cpu")
    selector.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return (
        ckpt["epoch"] + 1,
        ckpt["ema_mean"],
        ckpt["ema_var"],
        ckpt["best_val_map"],
    )


def _checkpoint_state(epoch, selector, optimizer, ema_mean, ema_var, best_val_map):
    return {
        "epoch": epoch,
        "model": selector.state_dict(),
        "optimizer": optimizer.state_dict(),
        "ema_mean": ema_mean,
        "ema_var": ema_var,
        "best_val_map": best_val_map,
    }


# ── REINFORCE epoch (Task 4) ───────────────────────────────────────────────────

def run_train_epoch(
    epoch: int,
    selector: AgentSelectorMLP,
    optimizer,
    loader: DataLoader,
    frame_cache: dict,
    state_index: StateIndex,
    fafmodule: FaFModule,
    config,
    device: torch.device,
    ema_mean: float,
    ema_var: float,
    lambda_cost: float,
    frames_per_epoch: int = 0,
    tb_writer=None,
    global_step: int = 0,
):
    selector.train()
    fafmodule.model.eval()

    aid_to_didx = _agent_id_to_dataset_idx()

    total_loss = 0.0
    total_entropy = 0.0
    total_reward = 0.0
    total_q_norm = 0.0
    total_cost_norm = 0.0
    total_n_sel = 0.0
    n_frames = 0

    tqdm_total = frames_per_epoch if frames_per_epoch > 0 else len(loader)
    bar = tqdm(total=tqdm_total, desc=f"Epoch {epoch}", leave=False)
    for batch in loader:
        if frames_per_epoch > 0 and n_frames >= frames_per_epoch:
            break

        (
            pvp_list,
            pvp_teacher_list,
            label_list,
            reg_target_list,
            reg_loss_mask_list,
            anchors_list,
            vis_maps_list,
            _gt_max_iou,
            filenames,
            target_agent_id_list,
            num_agent_list,
            trans_matrices_list,
        ) = zip(*batch)

        pvp_list = list(pvp_list)
        pvp_teacher_list = list(pvp_teacher_list)
        label_list = list(label_list)
        reg_target_list = list(reg_target_list)
        reg_loss_mask_list = list(reg_loss_mask_list)
        anchors_list = list(anchors_list)
        vis_maps_list = list(vis_maps_list)
        target_agent_id_list = list(target_agent_id_list)
        num_agent_list = list(num_agent_list)

        filename0 = filenames[0][0][0]
        parts = filename0.split(os.sep)
        scene_id, frame_id = map(int, parts[-2].split("_"))

        cached = frame_cache.get((scene_id, frame_id))
        if cached is None:
            continue

        features = cached["features"].to(device)        # (N, 12)
        q_lower = cached["q_lower"]
        q_upper = cached["q_upper"]
        feat_to_didx = cached["feat_to_didx"]
        N = cached["N"]

        # ── Policy: logits → probs → Bernoulli sample ──────────────────────────
        logits = selector(features)                     # (N,)
        probs = torch.sigmoid(logits)                   # (N,)
        a = Bernoulli(probs).sample()                   # (N,)

        # Guarantee at least one agent
        if a.sum() == 0:
            a[probs.argmax()] = 1.0

        selected_feat_indices = a.nonzero(as_tuple=True)[0].tolist()
        selected_dataset_indices = [
            feat_to_didx[i]
            for i in selected_feat_indices
            if feat_to_didx[i] is not None
        ]
        if not selected_dataset_indices:
            # No valid mapping — keep the closest agent
            selected_dataset_indices = [feat_to_didx[0] or 0]

        # ── Frozen detector (no grad for detection) ───────────────────────────
        try:
            trans_table = _ref_table_to_trans(trans_matrices_list)
            det_loss_val = _run_detector(
                fafmodule, config, selected_dataset_indices,
                pvp_list, pvp_teacher_list, label_list, reg_target_list,
                reg_loss_mask_list, anchors_list, vis_maps_list,
                target_agent_id_list, num_agent_list, trans_table, device,
            )
        except Exception as exc:
            print(f"  Detector error (scene {scene_id} frame {frame_id}): {exc}")
            continue

        # ── Reward computation ─────────────────────────────────────────────────
        q = -det_loss_val
        # Guard against near-zero span (q_upper ≈ q_lower).  When the fused-BEV
        # pipeline produces similar losses for all selections the raw span can be
        # tiny, turning the 1e-6 floor into a ×10^6 amplifier.  We enforce a
        # minimum span of MIN_SPAN (expressed as a fraction of |q_lower|) and
        # then hard-clip q_norm so a single bad frame can never derail the EMA.
        _MIN_SPAN = max(abs(q_lower) * 0.05, 1e-2)
        _span = max(q_upper - q_lower, _MIN_SPAN)
        q_norm = (q - q_lower) / _span
        q_norm = max(min(q_norm, 1.0), 0.0)    # clip to [0, 1]: 0=worst, 1=best
        cost_norm = len(selected_dataset_indices) / N_MAX
        reward = q_norm - lambda_cost * cost_norm

        # EMA variance-reduced baseline
        ema_mean = 0.95 * ema_mean + 0.05 * reward
        ema_var = 0.95 * ema_var + 0.05 * (reward - ema_mean) ** 2
        advantage = (reward - ema_mean) / (ema_var ** 0.5 + 1e-6)

        # ── Policy gradient + entropy bonus ───────────────────────────────────
        log_probs = (
            a * torch.log(probs + 1e-8) + (1 - a) * torch.log(1 - probs + 1e-8)
        )
        entropy = -(
            probs * torch.log(probs + 1e-8)
            + (1 - probs) * torch.log(1 - probs + 1e-8)
        ).sum()
        loss = -(log_probs.sum() * advantage) - 0.1 * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ── Accumulate metrics ────────────────────────────────────────────────
        total_loss += loss.item()
        total_entropy += entropy.item()
        total_reward += reward
        total_q_norm += q_norm
        total_cost_norm += cost_norm
        total_n_sel += len(selected_dataset_indices)
        n_frames += 1
        global_step += 1
        bar.update(1)

        # ── Per-step TensorBoard (negligible overhead vs detector forward) ────
        if tb_writer is not None:
            tb_writer.add_scalar("step/reward",      reward,                      global_step)
            tb_writer.add_scalar("step/q_norm",      q_norm,                      global_step)
            tb_writer.add_scalar("step/cost_norm",   cost_norm,                   global_step)
            tb_writer.add_scalar("step/loss",        loss.item(),                 global_step)
            tb_writer.add_scalar("step/entropy",     entropy.item(),              global_step)
            tb_writer.add_scalar("step/n_selected",  len(selected_dataset_indices), global_step)
            tb_writer.add_scalar("step/advantage",   advantage,                   global_step)
            tb_writer.add_scalar("step/ema_mean",    ema_mean,                    global_step)

        bar.set_postfix(
            R=f"{reward:.2f}",
            q=f"{q_norm:.2f}",
            c=f"{cost_norm:.2f}",
            loss=f"{loss.item():.3f}",
            H=f"{entropy.item():.2f}",
            n_bar=f"{len(selected_dataset_indices)}",
        )

    bar.close()

    if n_frames == 0:
        return ema_mean, ema_var, {}, global_step

    metrics = {
        "avg_loss": total_loss / n_frames,
        "avg_entropy": total_entropy / n_frames,
        "avg_reward": total_reward / n_frames,
        "avg_q_norm": total_q_norm / n_frames,
        "avg_cost_norm": total_cost_norm / n_frames,
        "avg_n_selected": total_n_sel / n_frames,
    }
    return ema_mean, ema_var, metrics, global_step


# ── Validation (Task 6) ───────────────────────────────────────────────────────

@torch.no_grad()
def run_validation(
    selector: AgentSelectorMLP,
    val_loader: DataLoader,
    state_index: StateIndex,
    fafmodule: FaFModule,
    config,
    device: torch.device,
    threshold: float = 0.5,
) -> float:
    """
    Greedy policy (probs > threshold) on val frames. Returns mAP@0.5.
    """
    from copy import deepcopy
    selector.eval()
    fafmodule.model.eval()

    aid_to_didx = _agent_id_to_dataset_idx()
    det_results_all = []
    annotations_all = []

    for batch in tqdm(val_loader, desc="Validation", leave=False):
        (
            pvp_list,
            pvp_teacher_list,
            label_list,
            reg_target_list,
            reg_loss_mask_list,
            anchors_list,
            vis_maps_list,
            gt_max_iou,
            filenames,
            target_agent_id_list,
            num_agent_list,
            trans_matrices_list,
        ) = zip(*batch)

        pvp_list = list(pvp_list)
        pvp_teacher_list = list(pvp_teacher_list)
        label_list = list(label_list)
        reg_target_list = list(reg_target_list)
        reg_loss_mask_list = list(reg_loss_mask_list)
        anchors_list = list(anchors_list)
        vis_maps_list = list(vis_maps_list)
        target_agent_id_list = list(target_agent_id_list)
        num_agent_list = list(num_agent_list)

        filename0 = filenames[0][0][0]
        parts = filename0.split(os.sep)
        scene_id, frame_id = map(int, parts[-2].split("_"))

        feats26, metas = build_state_features(
            state_index, scene_id, frame_id, return_meta=True
        )                                             # (N, 26) raw + engineered
        if feats26.shape[0] == 0:
            continue

        ml_feats = feats26[:, ML_START:].to(device)  # (N, 12) engineered slice
        feat_to_didx = [aid_to_didx.get(m.agent_id) for m in metas]

        # Greedy selection
        probs = torch.sigmoid(selector(ml_feats))
        sel_feat = (probs > threshold).nonzero(as_tuple=True)[0].tolist()
        if not sel_feat:
            sel_feat = [int(probs.argmax())]

        sel_dataset = [feat_to_didx[i] for i in sel_feat if feat_to_didx[i] is not None]
        if not sel_dataset:
            sel_dataset = [0]

        trans_table = _ref_table_to_trans(trans_matrices_list)

        try:
            def _sel(lst, idxs):
                return [lst[i] for i in idxs]

            data = assemble_detection_inputs(
                config,
                _sel(pvp_list, sel_dataset),
                _sel(pvp_teacher_list, sel_dataset),
                _sel(label_list, sel_dataset),
                _sel(reg_target_list, sel_dataset),
                _sel(reg_loss_mask_list, sel_dataset),
                _sel(anchors_list, sel_dataset),
                _sel(vis_maps_list, sel_dataset),
                _sel(target_agent_id_list, sel_dataset),
                _sel(num_agent_list, sel_dataset),
                [trans_table[j] for j in sel_dataset],
                device,
            )
            _, _, _, result = fafmodule.predict_all(data, batch_size=1, num_agent=1)
        except Exception:
            continue

        # Accumulate detection results for mAP
        padded_voxel_points = data["bev_seq"]
        reg_target = data["reg_targets"]
        anchors_map = data["anchors"]

        gt = gt_max_iou[0]
        if len(gt[0]["gt_box"]) == 0:
            gt_boxes = []
        else:
            gt_boxes = gt[0]["gt_box"][0, :, :]

        temp = {
            "bev_seq": padded_voxel_points[0, -1].cpu().numpy(),
            "result": [] if len(result[0]) == 0 else result[0][0][0],
            "reg_targets": reg_target.cpu().numpy()[0],
            "anchors_map": anchors_map.cpu().numpy()[0],
            "gt_max_iou": gt_boxes,
        }
        det_results_all, annotations_all = cal_local_mAP(
            config, temp, det_results_all, annotations_all
        )

    if not det_results_all:
        return 0.0

    try:
        mean_ap, _ = eval_map(
            det_results_all,
            annotations_all,
            scale_ranges=None,
            iou_thr=0.5,
            dataset=None,
            logger=None,
        )
        return float(mean_ap)
    except Exception:
        return 0.0


# ── Dataset helpers ────────────────────────────────────────────────────────────

def _build_dataset(data_dir: str, config, config_global, scene_begin: int, scene_end: int):
    dataset = V2XSimDet(
        dataset_roots=[
            os.path.join(data_dir, f"agent{i}") for i in AGENT_IDX_RANGE
        ],
        config=config,
        config_global=config_global,
        split="val",
        val=True,
        bound="lowerbound",
        kd_flag=False,
        rsu=False,
    )
    # Replace multiprocessing Manager cache with fast plain dicts
    dataset.cache = [{} for _ in range(dataset.num_agent)]

    # Filter to requested scene range
    scene_of_idx = dataset.seq_scenes[0]
    keep = [i for i, s in enumerate(scene_of_idx) if scene_begin <= s < scene_end]
    if not keep:
        raise RuntimeError(
            f"No frames in scene range [{scene_begin}, {scene_end}). "
            f"Scenes in dataset: {sorted(set(scene_of_idx))}"
        )
    return Subset(dataset, keep)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Scene range disjointness check ────────────────────────────────────────
    test_scenes = { 1, 3, 4, 63, 65, 68, 76, 78, 79, 84 }
    train_set = set(range(args.train_scene_begin, args.train_scene_end))
    val_set = set(range(args.val_scene_begin, args.val_scene_end))
    overlap_tv = train_set & val_set
    overlap_ttest = train_set & test_scenes
    overlap_vtest = val_set & test_scenes
    print(
        f"\nScene ranges:"
        f"\n  Train : [{args.train_scene_begin}, {args.train_scene_end})"
        f"\n  Val   : [{args.val_scene_begin}, {args.val_scene_end})"
        f"\n  Test  : {sorted(test_scenes)}"
    )
    if overlap_ttest:
        print(f"  WARNING: train ∩ test = {sorted(overlap_ttest)}")
    if overlap_vtest:
        print(f"  WARNING: val ∩ test = {sorted(overlap_vtest)}")
    if overlap_tv:
        print(f"  WARNING: train ∩ val = {sorted(overlap_tv)}")
    print()

    # ── Config ────────────────────────────────────────────────────────────────
    config = Config("train", binary=True, only_det=True)
    config.flag = "lowerbound"
    config_global = ConfigGlobal("train", binary=True, only_det=True)

    # ── Frozen detector ───────────────────────────────────────────────────────
    fafmodule = _build_fafmodule(config, args.ckpt, device)

    # Save a reference parameter to verify the detector is unchanged after training
    _det_ref_param = next(fafmodule.model.parameters()).detach().clone()

    # ── Datasets & loaders ───────────────────────────────────────────────────
    print("Loading train dataset…")
    train_dataset = _build_dataset(
        args.data_det, config, config_global,
        args.train_scene_begin, args.train_scene_end,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=args.num_workers
    )
    print(f"  Train frames: {len(train_dataset)}")

    val_data_dir = args.data_val or os.path.join(
        os.path.dirname(args.data_det.rstrip("/\\")) if "/train" not in args.data_det
        else args.data_det.replace("/train", "/val"),
        "",
    )
    # Simplest fallback: replace last component "train" → "val"
    if not val_data_dir or not os.path.isdir(val_data_dir):
        parent = os.path.dirname(args.data_det.rstrip("/\\"))
        val_data_dir = os.path.join(parent, "val")

    print("Loading val dataset…")
    try:
        val_dataset = _build_dataset(
            val_data_dir, config, config_global,
            args.val_scene_begin, args.val_scene_end,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers
        )
        print(f"  Val frames  : {len(val_dataset)}")
    except Exception as exc:
        print(f"  WARNING: could not load val dataset ({exc}). Validation disabled.")
        val_loader = None

    # ── State index ───────────────────────────────────────────────────────────
    all_scene_ids = set(range(args.train_scene_begin, args.train_scene_end))
    if val_loader is not None:
        all_scene_ids |= set(range(args.val_scene_begin, args.val_scene_end))
    state_index = StateIndex.from_fs(args.data_state, scene_ids=all_scene_ids)

    # ── Selector model ────────────────────────────────────────────────────────
    selector = AgentSelectorMLP(input_dim=FEATURE_DIM).to(device)
    optimizer = optim.Adam(selector.parameters(), lr=args.lr)

    # ── Pre-training cache (Task 3) ───────────────────────────────────────────
    # File is named selector_cache_s{begin}-{end}.pt inside the given directory.
    cache_file = (
        os.path.join(
            args.cache_dir,
            f"selector_cache_s{args.train_scene_begin}-{args.train_scene_end}.pt",
        )
        if args.cache_dir else ""
    )

    cache_loaded = False
    if cache_file and os.path.isfile(cache_file):
        print(f"Loading frame cache from {cache_file} …")
        frame_cache = torch.load(cache_file, map_location="cpu")
        print(f"Cache loaded: {len(frame_cache)} frames.")
        cache_loaded = True
    else:
        frame_cache = build_frame_cache(
            train_loader, state_index, fafmodule, config, device
        )
        if cache_file and len(frame_cache) > 0:
            os.makedirs(os.path.dirname(os.path.abspath(cache_file)), exist_ok=True)
            torch.save(frame_cache, cache_file)
            print(f"Frame cache saved to {cache_file}")

    # ── Resume from checkpoint if available ───────────────────────────────────
    last_ckpt = os.path.join(args.save_dir, "selector_last.pth")
    best_ckpt = os.path.join(args.save_dir, "selector_best.pth")
    start_epoch = 0
    ema_mean = 0.0
    ema_var = 1.0
    best_val_map = 0.0
    global_step = 0

    if os.path.isfile(last_ckpt):
        start_epoch, ema_mean, ema_var, best_val_map = _load_checkpoint(
            last_ckpt, selector, optimizer
        )
        global_step = start_epoch * args.frames_per_epoch  # approximate resume offset
        print(f"Resumed from epoch {start_epoch - 1} (best_val_map={best_val_map:.4f})")

    # ── CSV metrics file ──────────────────────────────────────────────────────
    csv_path = os.path.join(args.save_dir, "train_metrics.csv")
    csv_file = open(csv_path, "a", newline="")
    csv_writer = csv.writer(csv_file)
    if os.path.getsize(csv_path) == 0:
        csv_writer.writerow(
            ["epoch", "avg_reward", "avg_q_norm", "avg_cost_norm", "loss", "entropy", "avg_n_selected", "val_map"]
        )
        csv_file.flush()

    # ── Optional TensorBoard ──────────────────────────────────────────────────
    tb_writer = None
    if args.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_writer = SummaryWriter(log_dir=args.save_dir)
        except ImportError:
            print("WARNING: tensorboard not installed; --tensorboard ignored.")

    # ── Training loop ─────────────────────────────────────────────────────────
    try:
        for epoch in range(start_epoch, args.rl_epochs):
            # Free GPU memory fragmentation and DataLoader worker leaks from the
            # previous epoch's mid-loop break.  Recreating the loader each epoch
            # ensures workers are cleanly restarted rather than left dangling.
            gc.collect()
            torch.cuda.empty_cache()
            train_loader = DataLoader(
                train_dataset, batch_size=1, shuffle=True,
                num_workers=args.num_workers,
            )

            lambda_cost = 0.0 if epoch < args.curriculum_epochs else args.lambda_cost
            phase = "curriculum" if epoch < args.curriculum_epochs else "full"
            print(f"\nEpoch {epoch}/{args.rl_epochs - 1}  λ_cost={lambda_cost}  [{phase}]")

            ema_mean, ema_var, metrics, global_step = run_train_epoch(
                epoch=epoch,
                selector=selector,
                optimizer=optimizer,
                loader=train_loader,
                frame_cache=frame_cache,
                state_index=state_index,
                fafmodule=fafmodule,
                config=config,
                device=device,
                ema_mean=ema_mean,
                ema_var=ema_var,
                lambda_cost=lambda_cost,
                frames_per_epoch=args.frames_per_epoch,
                tb_writer=tb_writer,
                global_step=global_step,
            )

            # Periodic validation
            val_map = 0.0
            if val_loader is not None and (epoch + 1) % args.val_every == 0:
                print("  Running validation…")
                val_map = run_validation(
                    selector, val_loader, state_index, fafmodule,
                    config, device, threshold=args.sel_threshold,
                )
                print(f"  val mAP@0.5 = {val_map:.4f}")
                if val_map > best_val_map:
                    best_val_map = val_map
                    _save_checkpoint(
                        best_ckpt,
                        _checkpoint_state(epoch, selector, optimizer, ema_mean, ema_var, best_val_map),
                    )
                    print(f"  ✓ New best — saved to {best_ckpt}")

            # tqdm epoch summary
            avg_loss = metrics.get("avg_loss", float("nan"))
            avg_ent = metrics.get("avg_entropy", float("nan"))
            avg_n = metrics.get("avg_n_selected", float("nan"))
            avg_r = metrics.get("avg_reward", float("nan"))
            avg_q = metrics.get("avg_q_norm", float("nan"))
            avg_c = metrics.get("avg_cost_norm", float("nan"))
            print(
                f"  R={avg_r:.3f}  q={avg_q:.3f}  c={avg_c:.3f}  "
                f"loss={avg_loss:.3f}  H={avg_ent:.2f}  n̄={avg_n:.2f}  "
                f"val_mAP={val_map:.4f}"
            )

            # CSV row
            csv_writer.writerow([epoch, avg_r, avg_q, avg_c, avg_loss, avg_ent, avg_n, val_map])
            csv_file.flush()

            # TensorBoard
            if tb_writer is not None:
                tb_writer.add_scalar("train/reward", avg_r, epoch)
                tb_writer.add_scalar("train/q_norm", avg_q, epoch)
                tb_writer.add_scalar("train/cost_norm", avg_c, epoch)
                tb_writer.add_scalar("train/loss", avg_loss, epoch)
                tb_writer.add_scalar("train/entropy", avg_ent, epoch)
                tb_writer.add_scalar("train/avg_n_selected", avg_n, epoch)
                tb_writer.add_scalar("val/mAP_05", val_map, epoch)

            # Save last checkpoint every epoch
            _save_checkpoint(
                last_ckpt,
                _checkpoint_state(epoch, selector, optimizer, ema_mean, ema_var, best_val_map),
            )

    finally:
        # Crash-safe: always persist selector_last.pth on exit
        try:
            _save_checkpoint(
                last_ckpt,
                _checkpoint_state(
                    start_epoch, selector, optimizer, ema_mean, ema_var, best_val_map
                ),
            )
            print(f"\nProgress saved to {last_ckpt}")
        except Exception as exc:
            print(f"WARNING: could not save final checkpoint: {exc}")
        csv_file.close()
        if tb_writer is not None:
            tb_writer.close()

    # ── Verify detector unchanged ─────────────────────────────────────────────
    _det_ref_param_after = next(fafmodule.model.parameters()).detach()
    if torch.allclose(_det_ref_param.to(device), _det_ref_param_after):
        print("✓ Frozen detector weights are unchanged after training.")
    else:
        print("WARNING: Detector weights differ — this should not happen!")

    print(f"\nTraining complete. Best val mAP@0.5 = {best_val_map:.4f}")
    print(f"  selector_best.pth : {best_ckpt}")
    print(f"  selector_last.pth : {last_ckpt}")
    print(f"  train_metrics.csv : {csv_path}")


if __name__ == "__main__":
    main()
