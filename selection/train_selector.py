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

    # RL selector
    p.add_argument("--selector_hidden_dim", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lambda_cost", type=float, default=1000,
                   help="Bandwidth penalty weight.")
    p.add_argument("--reward_baseline_momentum", type=float, default=0.9)

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
        shuffle=True, # can be set to True later
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
    optimizer = optim.Adam(det_model.parameters(), lr=0.001)
    criterion = {
        "cls": SoftmaxFocalClassificationLoss(),
        "loc": WeightedSmoothL1LocalizationLoss(),
    }

    fafmodule = FaFModule(det_model, det_model, config, optimizer, criterion, 0)

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
    optimizer = torch.optim.Adam(selector.parameters(), lr=args.lr)

    # Baseline for REINFORCE (EMA of reward)
    baseline = 0.0
    beta = args.reward_baseline_momentum
    frame_logs: List[dict] = []
    loss_history: List[float] = []
    reward_history: List[float] = []
    frame_counter = 0

    # ------------------------------------------------------------------
    # RL training loop (per-frame bandit)
    # ------------------------------------------------------------------
    print("Starting RL training of selector...")
    for epoch in range(args.epochs):
        selector.train()
        total_loss = 0.0
        total_frames = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
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
            ) = zip(*batch)

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

            # --- policy: logits -> probs -> Bernoulli sample ---
            logits = selector(state_feats)           # (N,)
            probs = torch.sigmoid(logits)            # (N,)
            m = torch.distributions.Bernoulli(probs)
            actions = m.sample()                     # (N,)
            log_probs = m.log_prob(actions)          # (N,)

            #selected_indices = actions.nonzero(as_tuple=True)[0].cpu().tolist()
            selected_indices = [i for i in range(len(actions))] # temporary: select all agents
            num_selected = len(selected_indices)
            selected_agent_ids = [available_agent_ids[i] for i in selected_indices]
            det_loss_value = None

            if num_selected > 0:
                # --- filter detection inputs according to selected agents ---
                def _sel(lst):
                    return [lst[i] for i in selected_indices]

                pvp_list_sel = _sel(padded_voxel_point_list)
                pvp_teacher_list_sel = _sel(padded_voxel_points_teacher_list)
                label_list_sel = _sel(label_one_hot_list)
                reg_target_list_sel = _sel(reg_target_list)
                reg_loss_mask_list_sel = _sel(reg_loss_mask_list)
                anchors_list_sel = _sel(anchors_map_list)
                vis_maps_list_sel = _sel(vis_maps_list)
                target_agent_id_list_sel = _sel(target_agent_id_list)
                num_agent_list_sel = _sel(num_agent_list)
                trans_mats_list_sel = _sel(trans_matrices_list)

                # Fuse the post-selection agents using the same alignment and voxelization
                # used by create_data_det so detector inputs only contain chosen data.
                data = assemble_detection_inputs(
                    config,
                    pvp_list_sel,
                    pvp_teacher_list_sel,
                    label_list_sel,
                    reg_target_list_sel,
                    reg_loss_mask_list_sel,
                    anchors_list_sel,
                    vis_maps_list_sel,
                    target_agent_id_list_sel,
                    num_agent_list_sel,
                    trans_mats_list_sel,
                    device,
                )

                # --- run detection model (frozen) ---
                with torch.no_grad():
                    det_loss, cls_loss, loc_loss, result = fafmodule.predict_all(
                        data, batch_size=1, num_agent=num_selected
                    )

                # reward: lower loss is better; penalize number of selected agents
                det_loss_value = float(det_loss)
                reward_value = -det_loss_value - args.lambda_cost * float(num_selected)
            else:
                # no agents selected
                reward_value = -10000.0

            # update baseline
            baseline = beta * baseline + (1.0 - beta) * reward_value
            advantage = reward_value - baseline

            # --- RL loss (frame-level REINFORCE) ---
            # sum log_probs over all agents in this frame
            log_prob_sum = log_probs.sum()

            loss = -log_prob_sum * torch.tensor(advantage, device=device)

            loss_value = float(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
                    "actions": actions.detach().cpu().int().tolist(),
                    "probs": probs.detach().cpu().tolist(),
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
