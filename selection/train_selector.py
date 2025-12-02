import argparse
import os
from typing import List

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
    p.add_argument("--lambda_cost", type=float, default=0.01,
                   help="Bandwidth penalty weight.")
    p.add_argument("--reward_baseline_momentum", type=float, default=0.9)

    p.add_argument("--save_path", type=str, required=True,)

    return p.parse_args()

def build_model_from_config(config, num_agent) -> DetModelBase:
    return FaFNet(
        config, kd_flag=False, num_agent=num_agent
    )


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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

    state_feats0 = build_state_features(
        state_index=state_index,
        scene_id=scene_id,
        frame_id=frame_id,
    )
    input_dim = state_feats0.shape[1]

    selector = AgentSelectorMLP(input_dim=input_dim, hidden_dim=args.selector_hidden_dim).to(device)
    optimizer = torch.optim.Adam(selector.parameters(), lr=args.lr)

    # Baseline for REINFORCE (EMA of reward)
    baseline = 0.0
    beta = args.reward_baseline_momentum

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

            state_feats = build_state_features(
                state_index=state_index,
                scene_id=scene_id,
                frame_id=frame_id,
            )
            state_feats = state_feats.to(device)  # (N, D)

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
            log_probs = m.log_prob(actions)         # (N,)

            selected_indices = actions.nonzero(as_tuple=True)[0].cpu().tolist()
            num_selected = len(selected_indices)

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

                trans_matrices = torch.stack(tuple(trans_mats_list_sel), 1).to(device)
                target_agent_ids = torch.stack(tuple(target_agent_id_list_sel), 1).to(device)
                num_all_agents = torch.tensor([[num_selected]], dtype=torch.int64, device=device)

                # choose teacher or normal bev (here just use normal)
                padded_voxel_points = torch.cat(tuple(pvp_teacher_list_sel), 0).to(device)
                label_one_hot = torch.cat(tuple(label_list_sel), 0).to(device)
                reg_target = torch.cat(tuple(reg_target_list_sel), 0).to(device)
                reg_loss_mask = torch.cat(tuple(reg_loss_mask_list_sel), 0).to(device)
                anchors_map = torch.cat(tuple(anchors_list_sel), 0).to(device)
                vis_maps = torch.cat(tuple(vis_maps_list_sel), 0).to(device)

                data = {
                    "bev_seq": padded_voxel_points,
                    "labels": label_one_hot,
                    "reg_targets": reg_target,
                    "anchors": anchors_map,
                    "vis_maps": vis_maps,
                    "reg_loss_mask": reg_loss_mask.bool(),
                    "target_agent_ids": target_agent_ids,
                    "num_agent": num_all_agents,
                    "trans_matrices": trans_matrices,
                }

                # --- run detection model (frozen) ---
                with torch.no_grad():
                    det_loss, cls_loss, loc_loss, result = fafmodule.predict_all(
                        data, batch_size=1, num_agent=num_selected
                    )

                # reward: lower loss is better; penalize number of selected agents
                reward = -det_loss - args.lambda_cost * float(num_selected)
            else:
                # no agents selected
                reward = 0

            # update baseline
            baseline = beta * baseline + (1.0 - beta) * reward
            advantage = reward - baseline

            # --- RL loss (frame-level REINFORCE) ---
            # sum log_probs over all agents in this frame
            log_prob_sum = log_probs.sum()

            loss = -log_prob_sum * advantage

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_frames += 1

        avg_loss = total_loss / max(1, total_frames)
        print(f"[Epoch {epoch+1}/{args.epochs}] RL loss per frame: {avg_loss:.4f}")

    # ------------------------------------------------------------------
    # Save trained selector
    # ------------------------------------------------------------------
    torch.save(selector.state_dict(), args.save_path)
    print(f"Saved RL selector to {args.save_path}")


if __name__ == "__main__":
    main()
