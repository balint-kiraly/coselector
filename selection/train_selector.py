import argparse
import json
import os
from datetime import datetime
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from coperception import Config, ConfigGlobal
from coperception.models.det import FaFNet
from coperception.models.det.base import DetModelBase
from coperception.utils.CoDetModule import FaFModule
from coperception.utils.loss import SoftmaxFocalClassificationLoss, WeightedSmoothL1LocalizationLoss
from data_utils.build_state_features import build_state_features
from data_utils.state_index import StateIndex
from selection.bev_builder import assemble_detection_inputs
from selection.models import AgentSelectorMLP

from coperception.datasets import V2XSimDet


def filter_boxes_in_roi(
    boxes: torch.Tensor,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> torch.Tensor:
    """Return a boolean mask for boxes whose centers fall inside the ROI."""
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.bool, device=boxes.device)
    x_center = boxes[:, 0]
    y_center = boxes[:, 1]
    return (x_center >= x_min) & (x_center <= x_max) & (y_center >= y_min) & (y_center <= y_max)


def compute_f1_roi(
    result,
    gt_list,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    iou_thresh: float,
) -> float:
    """
    Compute an ROI-only detection F1 score.

    - Predictions and ground truths are filtered to the ROI using center coordinates.
    - Predictions are not confidence-thresholded (all scores are considered).
    - Greedy matching uses axis-aligned IoU on (x, y, w, l) with a fixed threshold.
    - Returns a scalar in [0, 1]; falls back to 0 when no GT exists in ROI.
    """

    def _prep_boxes(pred_entry):
        if len(pred_entry) == 0:
            return torch.empty((0, 7)), torch.empty((0,))
        pred_dict = pred_entry[0][0][0]  # predict_all output format
        preds = pred_dict.get("pred", torch.empty((0, 7)))
        scores = pred_dict.get("score", torch.empty((len(preds),)))
        if not isinstance(preds, torch.Tensor):
            preds = torch.as_tensor(preds)
        if not isinstance(scores, torch.Tensor):
            scores = torch.as_tensor(scores)
        return preds, scores

    def _prep_gt(gt_entry):
        if len(gt_entry[0]["gt_box"]) == 0:
            return torch.empty((0, 7))
        gt = gt_entry[0]["gt_box"][0]
        if not isinstance(gt, torch.Tensor):
            gt = torch.as_tensor(gt)
        return gt

    pred_boxes_all = []
    gt_boxes_all = []
    pred_scores_all = []

    for pred_entry, gt_entry in zip(result, gt_list):
        preds, scores = _prep_boxes(pred_entry)
        gts = _prep_gt(gt_entry)

        if preds.numel() > 0:
            roi_mask = filter_boxes_in_roi(preds, x_min, x_max, y_min, y_max)
            preds = preds[roi_mask]
            scores = scores[roi_mask]
        if gts.numel() > 0:
            gts = gts[filter_boxes_in_roi(gts, x_min, x_max, y_min, y_max)]

        if preds.numel() > 0:
            pred_boxes_all.append(preds)
            pred_scores_all.append(scores)
        if gts.numel() > 0:
            gt_boxes_all.append(gts)

    if not gt_boxes_all:
        return 0.0
    if not pred_boxes_all:
        return 0.0

    pred_boxes = torch.cat(pred_boxes_all, dim=0)
    pred_scores = torch.cat(pred_scores_all, dim=0)
    gt_boxes = torch.cat(gt_boxes_all, dim=0)

    # Sort predictions by confidence for greedy matching
    _, sort_idx = torch.sort(pred_scores, descending=True)
    pred_boxes = pred_boxes[sort_idx]

    # Helper: convert center-format box to axis-aligned corners
    def _box_to_corners(box):
        x_c, y_c, w, l = float(box[0]), float(box[1]), float(box[3]), float(box[4])
        half_w = w / 2.0
        half_l = l / 2.0
        return x_c - half_l, y_c - half_w, x_c + half_l, y_c + half_w

    gt_used = torch.zeros((gt_boxes.shape[0],), dtype=torch.bool)
    tp = 0

    for pred_box in pred_boxes:
        px1, py1, px2, py2 = _box_to_corners(pred_box)
        best_iou = 0.0
        best_gt_idx = -1
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_used[gt_idx]:
                continue
            gx1, gy1, gx2, gy2 = _box_to_corners(gt_box)
            inter_x1 = max(px1, gx1)
            inter_y1 = max(py1, gy1)
            inter_x2 = min(px2, gx2)
            inter_y2 = min(py2, gy2)
            inter_w = max(0.0, inter_x2 - inter_x1)
            inter_h = max(0.0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h
            if inter_area <= 0:
                continue
            pred_area = (px2 - px1) * (py2 - py1)
            gt_area = (gx2 - gx1) * (gy2 - gy1)
            union = pred_area + gt_area - inter_area
            iou = inter_area / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        if best_gt_idx >= 0 and best_iou >= iou_thresh:
            tp += 1
            gt_used[best_gt_idx] = True

    fp = pred_boxes.shape[0] - tp
    fn = gt_boxes.shape[0] - tp

    denom = (2 * tp + fp + fn)
    if denom == 0:
        return 0.0
    return float((2 * tp) / denom)


def compute_roi_bounds_from_rsu(metas, x_radius: float, y_radius: float):
    """Center the ROI on the RSU (agent 0) location for the current frame."""

    for meta in metas:
        if meta.agent_id == 0:
            x_c = float(meta.x)
            y_c = float(meta.y)
            return (
                x_c - x_radius,
                x_c + x_radius,
                y_c - y_radius,
                y_c + y_radius,
            )

    raise ValueError(
        "RSU (agent 0) metadata missing for this frame; cannot center ROI on RSU."
    )


def run_selection_strategy(state_feats, selector, strategy: str, topk: int):
    """Dispatch to the requested selector strategy before BEV construction."""

    num_agents = state_feats.shape[0]

    if strategy == "identity":
        # Keep everyone; no gradient contribution.
        actions = torch.ones((num_agents,), device=state_feats.device)
        log_probs = torch.zeros_like(actions)
        return list(range(num_agents)), actions, log_probs, torch.ones_like(actions)

    if strategy == "topk":
        k = min(topk, num_agents)
        # Assume the first two columns of the state features encode planar position.
        distances = torch.norm(state_feats[:, :2], dim=1)
        _, idx = torch.topk(distances, k=k, largest=False)
        actions = torch.zeros((num_agents,), device=state_feats.device)
        actions[idx] = 1.0
        log_probs = torch.zeros_like(actions)
        probs = torch.zeros_like(actions)
        probs[idx] = 1.0
        return idx.cpu().tolist(), actions, log_probs, probs

    # Learned RL policy
    logits = selector(state_feats)  # (N,)
    probs = torch.sigmoid(logits)
    distribution = torch.distributions.Bernoulli(probs)
    actions = distribution.sample()
    log_probs = distribution.log_prob(actions)
    selected_indices = actions.nonzero(as_tuple=True)[0].cpu().tolist()
    return selected_indices, actions, log_probs, probs


def parse_args():
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--data_det", type=str, required=True,
                   help="Path to precomputed detection data.")
    p.add_argument("--data_state", type=str, required=True,
                   help="Path to state index files.")

    p.add_argument("--scene_start", type=int, default=0)
    p.add_argument("--scene_end", type=int, default=100)
    p.add_argument("--agent_start", type=int, default=1)
    p.add_argument("--agent_end", type=int, default=6)

    # model
    p.add_argument("--ckpt", type=str, required=True,
                   help="Detection model checkpoint to load.")
    p.add_argument("--num_workers", type=int, default=1)

    # selector control / modes
    p.add_argument(
        "--mode",
        choices=["train", "eval"],
        default="train",
        help="Whether to update the selector (train) or just log selections (eval).",
    )
    p.add_argument(
        "--selector_strategy",
        choices=["learned", "identity", "topk"],
        default="learned",
        help=(
            "Selector to run before BEV extraction: RL policy (learned), "
            "identity (keep all), or topk (nearest agents by distance feature)."
        ),
    )
    p.add_argument(
        "--selector_topk",
        type=int,
        default=3,
        help="K for the top-k selector strategy (used only when selector_strategy=topk).",
    )


    # RL selector
    p.add_argument("--selector_hidden_dim", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lambda_cost", type=float, default=1000,
                   help="Bandwidth penalty weight.")
    p.add_argument("--roi_x_radius", type=float, default=10.0,
                   help="Half-width of ROI in meters (centered on RSU/agent 0).")
    p.add_argument("--roi_y_radius", type=float, default=10.0,
                   help="Half-height of ROI in meters (centered on RSU/agent 0).")
    p.add_argument("--roi_iou_thresh", type=float, default=0.5,
                   help="IoU threshold for ROI true positives.")
    p.add_argument(
        "--use_teacher_bev",
        action="store_true",
        help=(
            "If set, feed the upperbound/teacher BEV tensor to the detector. "
            "Leave unset to use each selected agent's own BEV so the selector "
            "actually gates which raw inputs reach the detector."
        ),
    )
    p.add_argument("--reward_baseline_momentum", type=float, default=0.9)
    p.add_argument(
        "--use_raw_lidar",
        action="store_true",
        help=(
            "If set, voxelize raw lidar for the selected agents on-the-fly so "
            "selection truly happens before BEV construction. If unset, "
            "precomputed BEVs from the dataset are used as a fallback."
        ),
    )

    p.add_argument("--save_path", type=str, required=True,)
    p.add_argument("--logs", type=str, default="logs",
                   help="Directory to save plots and frame-level selections.")

    return p.parse_args()

def build_model_from_config(config, num_agent) -> DetModelBase:
    return FaFNet(
        config, kd_flag=False, num_agent=num_agent
    )


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    ckpt_name = os.path.splitext(os.path.basename(args.ckpt))[0]
    exp_name = (
        f"selector_{ckpt_name}_sc{args.scene_start}-{args.scene_end}"
        f"_ag{args.agent_start}-{args.agent_end}_ep{args.epochs}_{timestamp}"
    )
    os.makedirs(args.logs, exist_ok=True)
    log_dir = os.path.join(args.logs, exp_name)
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logging to {log_dir}")

    # not include RSU
    args.agent_start = max(args.agent_start, 1)
    num_agent = args.agent_end - args.agent_start + 1

    # ------------------------------------------------------------------
    # Build detection dataset + dataloader (train split)
    # ------------------------------------------------------------------
    print("Loading detection dataset...")
    config = Config("train", binary=True, only_det=True)
    config_global = ConfigGlobal("train", binary=True, only_det=True)
    config.flag = "upperbound"
    agent_idx_range = range(args.agent_start, args.agent_end)

    train_dataset = V2XSimDet(
        dataset_roots=[os.path.join(args.data_det, f"agent{i}") for i in agent_idx_range],
        config=config,
        config_global=config_global,
        split="val",
        val=True,
        bound="upperbound",
        kd_flag=False,
        rsu=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers
    )
    print(f"Train dataset size: {len(train_dataset)} samples.")

    # ------------------------------------------------------------------
    # Build detection model (frozen)
    # ------------------------------------------------------------------
    print("Building detection model...")
    det_model = build_model_from_config(config, num_agent)
    det_model = nn.DataParallel(det_model)
    det_model.to(device)
    det_model.eval()
    for p in det_model.parameters():
        p.requires_grad = False
    det_optimizer = optim.Adam(det_model.parameters(), lr=0.001)
    criterion = {
        "cls": SoftmaxFocalClassificationLoss(),
        "loc": WeightedSmoothL1LocalizationLoss(),
    }

    fafmodule = FaFModule(det_model, det_model, config, det_optimizer, criterion, 0)

    checkpoint = torch.load(
        args.ckpt, map_location=device
    )
    start_epoch = checkpoint["epoch"] + 1
    fafmodule.model.load_state_dict(checkpoint["model_state_dict"])
    fafmodule.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    fafmodule.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    print(f"Loaded detection model checkpoint from {args.ckpt} (epoch {start_epoch})")

    # ------------------------------------------------------------------
    # Build state index (GNSS/IMU)
    # ------------------------------------------------------------------
    state_index = StateIndex.from_fs(args.data_state)

    # ------------------------------------------------------------------
    # Build selector policy
    # ------------------------------------------------------------------
    # Infer input_dim from first batch/features
    # Grab one sample from loader to get feature dim
    sample0 = next(iter(train_loader))
    (
        _pvp_list,
        _pvp_teacher_list,
        _label_list,
        _reg_target_list,
        _reg_loss_mask_list,
        _anchors_list,
        _vis_maps_list,
        _gt_max_iou,
        filenames0,
        _target_agent_id_list0,
        _num_agent_list0,
        _trans_mats_list0,
    ) = zip(*sample0)

    first_fname = filenames0[0][0][0]
    parts = first_fname.split(os.sep)
    scene_frame = parts[-2]  # e.g. "12_5"
    scene_id, frame_id = map(int, scene_frame.split("_"))

    state_feats0, _ = build_state_features(
        state_index=state_index,
        scene_id=scene_id,
        frame_id=frame_id,
        return_meta=True,
    )
    input_dim = state_feats0.shape[1]

    selector = AgentSelectorMLP(input_dim=input_dim, hidden_dim=args.selector_hidden_dim).to(device)
    selector_optimizer = torch.optim.Adam(selector.parameters(), lr=args.lr)

    # Baseline for REINFORCE (EMA of reward)
    baseline = 0.0
    beta = args.reward_baseline_momentum
    frame_logs: List[dict] = []
    loss_history: List[float] = []
    reward_history: List[float] = []
    frame_counter = 0
    rsu_missing_warned = False

    # ------------------------------------------------------------------
    # RL training loop (per-frame bandit)
    # ------------------------------------------------------------------
    print("Starting RL training of selector...")
    for epoch in range(args.epochs):
        selector.train()
        total_loss = 0.0
        total_frames = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            batch_fields = list(zip(*batch))
            if len(batch_fields) == 12:
                (
                    padded_voxel_point_list,
                    padded_voxel_points_teacher_list,
                    label_one_hot_list,
                    reg_target_list,
                    reg_loss_mask_list,
                    anchors_map_list,
                    vis_maps_list,
                    gt_max_iou,
                    filenames,
                    target_agent_id_list,
                    num_agent_list,
                    trans_matrices_list,
                ) = batch_fields
                raw_lidar_list = None
            elif len(batch_fields) == 13:
                (
                    padded_voxel_point_list,
                    padded_voxel_points_teacher_list,
                    label_one_hot_list,
                    reg_target_list,
                    reg_loss_mask_list,
                    anchors_map_list,
                    vis_maps_list,
                    gt_max_iou,
                    filenames,
                    target_agent_id_list,
                    num_agent_list,
                    trans_matrices_list,
                    raw_lidar_list,
                ) = batch_fields
            else:
                raise ValueError(
                    f"Unexpected batch size {len(batch_fields)}; expected 12 or 13 fields (got {len(batch_fields)})."
                )

            # --- build state features ---
            filename0 = filenames[0][0][0]
            parts = filename0.split(os.sep)
            scene_frame = parts[-2]  # e.g. "12_5"
            scene_id, frame_id = map(int, scene_frame.split("_"))

            state_feats, metas = build_state_features(
                state_index=state_index,
                scene_id=scene_id,
                frame_id=frame_id,
                return_meta=True,
            )
            state_feats = state_feats.to(device)  # (N, D)
            available_agent_ids = [m.agent_id for m in metas]

            # Center ROI on RSU (agent 0) for this frame.
            try:
                roi_x_min, roi_x_max, roi_y_min, roi_y_max = compute_roi_bounds_from_rsu(
                    metas, args.roi_x_radius, args.roi_y_radius
                )
            except ValueError as e:
                if not rsu_missing_warned:
                    print(f"Warning: {e}")
                    rsu_missing_warned = True
                # Fall back to origin-centered ROI to keep training running, but this
                # should be fixed in the metadata so the ROI tracks the RSU properly.
                roi_x_min, roi_x_max = -args.roi_x_radius, args.roi_x_radius
                roi_y_min, roi_y_max = -args.roi_y_radius, args.roi_y_radius

            padded_voxel_point_list = list(padded_voxel_point_list)
            padded_voxel_points_teacher_list = list(padded_voxel_points_teacher_list)
            label_one_hot_list = list(label_one_hot_list)
            reg_target_list = list(reg_target_list)
            reg_loss_mask_list = list(reg_loss_mask_list)
            anchors_map_list = list(anchors_map_list)
            vis_maps_list = list(vis_maps_list)
            target_agent_id_list = list(target_agent_id_list)
            num_agent_list = list(num_agent_list)
            trans_matrices_list = list(trans_matrices_list)
            raw_lidar_list = list(raw_lidar_list) if raw_lidar_list is not None else None

            # --- policy: logits -> probs -> Bernoulli sample ---
            # Selector runs *before* any BEV feature extraction so that only the
            # chosen agents incur voxelization/fusion cost.
            selected_indices, actions, log_probs, probs = run_selection_strategy(
                state_feats, selector, args.selector_strategy, args.selector_topk
            )
            num_selected = len(selected_indices)
            selected_agent_ids = [available_agent_ids[i] for i in selected_indices]
            det_loss_value = None
            metric_roi = 0.0
            bev_build_mode = "raw_lidar" if (args.use_raw_lidar and raw_lidar_list is not None) else "precomputed"

            if num_selected > 0:
                # --- filter detection inputs according to selected agents (pre-BEV if raw lidar is available) ---
                raw_pc_sel = None
                if args.use_raw_lidar and raw_lidar_list is not None:
                    raw_pc_sel = [raw_lidar_list[i] for i in selected_indices]

                data, num_agents = assemble_detection_inputs(
                    selected_indices=selected_indices,
                    raw_point_clouds=raw_pc_sel,
                    precomputed_bevs=padded_voxel_point_list,
                    teacher_bevs=padded_voxel_points_teacher_list,
                    labels=label_one_hot_list,
                    reg_targets=reg_target_list,
                    reg_loss_masks=reg_loss_mask_list,
                    anchors=anchors_map_list,
                    vis_maps=vis_maps_list,
                    trans_matrices_list=trans_matrices_list,
                    target_agent_id_list=target_agent_id_list,
                    device=device,
                    config=config,
                    use_teacher_bev=args.use_teacher_bev,
                )

                if num_agents > 0:
                    gt_list_sel = [gt_max_iou[i] for i in selected_indices]

                    # --- run detection model (frozen) ---
                    with torch.no_grad():
                        det_loss, cls_loss, loc_loss, result = fafmodule.predict_all(
                            data, batch_size=1, num_agent=num_agents
                        )

                    # --- ROI-filtered metric (optimize detection around RSU crossroad) ---
                    # Compute F1 inside the configured ROI using only selected agents' predictions/GTs.
                    metric_roi = compute_f1_roi(
                        result,
                        gt_list_sel,
                        roi_x_min,
                        roi_x_max,
                        roi_y_min,
                        roi_y_max,
                        args.roi_iou_thresh,
                    )

                    # reward: higher ROI metric is better; still penalize number of selected agents
                    det_loss_value = float(det_loss)
                    reward_value = metric_roi - args.lambda_cost * float(num_selected)
                else:
                    reward_value = -10000.0
            else:
                # no agents selected
                reward_value = -10000.0

            # update baseline only when learning
            if args.selector_strategy == "learned":
                baseline = beta * baseline + (1.0 - beta) * reward_value
                advantage = reward_value - baseline
            else:
                advantage = reward_value

            # --- RL loss (frame-level REINFORCE) ---
            # sum log_probs over all agents in this frame
            log_prob_sum = log_probs.sum()

            loss = -log_prob_sum * torch.tensor(advantage, device=device)

            loss_value = float(loss.item())

            if args.mode == "train" and args.selector_strategy == "learned":
                selector_optimizer.zero_grad()
                loss.backward()
                selector_optimizer.step()

            # frame-level logging
            loss_history.append(loss_value)
            reward_history.append(float(reward_value))
            frame_logs.append(
                {
                    "epoch": epoch + 1,
                    "frame_global_index": frame_counter,
                    "scene_id": scene_id,
                    "frame_id": frame_id,
                    "available_agent_ids": available_agent_ids,
                    "selected_indices": selected_indices,
                    "selected_agent_ids": selected_agent_ids,
                    "num_selected": num_selected,
                    "num_available": len(available_agent_ids),
                    "reward": float(reward_value),
                    "loss": loss_value,
                    "det_loss": det_loss_value,
                    "roi_metric_f1": metric_roi,
                    "actions": actions.detach().cpu().int().tolist(),
                    "probs": probs.detach().cpu().tolist(),
                    "selector_strategy": args.selector_strategy,
                    "mode": args.mode,
                    "bev_build_mode": bev_build_mode,
                }
            )
            frame_counter += 1

            total_loss += loss_value
            total_frames += 1


        avg_loss = total_loss / max(1, total_frames)
        print(f"[Epoch {epoch+1}/{args.epochs}] RL loss per frame: {avg_loss:.4f}")

    # ------------------------------------------------------------------
    # Persist logs and plots
    # ------------------------------------------------------------------
    metrics_path = os.path.join(log_dir, "metrics.json")
    frame_logs_path = os.path.join(log_dir, "frame_logs.json")
    plots = {}

    if frame_logs:
        frame_indices = [log["frame_global_index"] for log in frame_logs]

        # Agent selection scatter plot
        plt.figure(figsize=(10, 5))
        x_points = []
        y_points = []
        x_none = []
        for log in frame_logs:
            if log["selected_agent_ids"]:
                x_points.extend([log["frame_global_index"]] * len(log["selected_agent_ids"]))
                y_points.extend(log["selected_agent_ids"])
            else:
                x_none.append(log["frame_global_index"])
        if x_points:
            plt.scatter(x_points, y_points, s=14, alpha=0.7, label="Selected agents")
        if x_none:
            plt.scatter(x_none, [0] * len(x_none), marker="x", color="red", alpha=0.6, label="No agents selected")
        plt.xlabel("Frame")
        plt.ylabel("Agent ID (0 marks no selection)")
        plt.title("Agent selections per frame")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        selections_plot_path = os.path.join(log_dir, "agent_selections.png")
        plt.savefig(selections_plot_path, dpi=200)
        plt.close()
        plots["agent_selections"] = selections_plot_path

        # Loss and reward curves
        plt.figure(figsize=(10, 5))
        plt.plot(frame_indices, loss_history, label="RL loss")
        plt.plot(frame_indices, reward_history, label="Reward")
        plt.xlabel("Frame")
        plt.ylabel("Value")
        plt.title("RL loss and reward per frame")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        lr_plot_path = os.path.join(log_dir, "loss_reward.png")
        plt.savefig(lr_plot_path, dpi=200)
        plt.close()
        plots["loss_reward"] = lr_plot_path

    with open(frame_logs_path, "w") as f:
        json.dump(frame_logs, f, indent=2)

    run_metadata = {
        "experiment_name": exp_name,
        "log_dir": log_dir,
        "num_frames": len(frame_logs),
        "args": vars(args),
        "plots": plots,
        "loss_history": loss_history,
        "reward_history": reward_history,
    }
    with open(metrics_path, "w") as f:
        json.dump(run_metadata, f, indent=2)
    print(f"Saved frame logs and plots to {log_dir}")

    # ------------------------------------------------------------------
    # Save trained selector
    # ------------------------------------------------------------------
    torch.save(selector.state_dict(), args.save_path)
    print(f"Saved RL selector to {args.save_path}")


if __name__ == "__main__":
    main()
