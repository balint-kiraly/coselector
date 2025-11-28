# coperception/agent_selection/train_selector_rl.py

import argparse
import os
from typing import List

import torch
from torch.utils.data import DataLoader

from data_utils.build_state_features import build_state_features
from data_utils.state_index import StateIndex
from selection.models import AgentSelectorMLP

from coperception.datasets import V2XSimDet
from coperception.utils.config import load_config   # placeholder
from coperception.models.builder import build_model_from_config  # placeholder


def parse_args():
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--data_det", type=str, required=True,
                   help="Path to precomputed detection data (V2X-Sim-det root).")
    p.add_argument("--data_raw", type=str, required=True,
                   help="Path to raw V2X-Sim / nuScenes-style root (for GNSS/IMU).")

    p.add_argument("--scene_start", type=int, default=0)
    p.add_argument("--scene_end", type=int, default=79)
    p.add_argument("--agent_start", type=int, default=0)
    p.add_argument("--agent_end", type=int, default=4)

    # model / config
    p.add_argument("--config", type=str, required=True,
                   help="Path to detection model config.")
    p.add_argument("--ckpt", type=str, required=True,
                   help="Detection model checkpoint to load.")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)

    # RL selector
    p.add_argument("--selector_hidden_dim", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lambda_cost", type=float, default=0.01,
                   help="Bandwidth penalty weight.")
    p.add_argument("--gamma", type=float, default=1.0,
                   help="Discount factor (1.0 for bandit).")
    p.add_argument("--reward_baseline_momentum", type=float, default=0.9)

    p.add_argument("--save_path", type=str, default="selector_rl.pth")

    return p.parse_args()


def get_agent_ids_from_filenames(filenames) -> List[int]:
    # filenames is a tuple/list of len N_agents, each element is a tuple (because batch=1)
    import os
    ids = []
    for f in filenames:
        fpath = f[0]
        parts = fpath.split(os.sep)
        agent_part = [p for p in parts if p.startswith("agent")][0]
        aid = int(agent_part.replace("agent", ""))
        ids.append(aid)
    return ids


def get_scene_frame_from_filename(fname: str):
    import os
    parts = fname.split(os.sep)
    scene_frame = parts[-2]  # e.g. "12_5"
    scene_id, frame_id = map(int, scene_frame.split("_"))
    return scene_id, frame_id


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # 1) Build detection dataset + dataloader (train split)
    # ------------------------------------------------------------------
    config, config_global = load_config(args.config)   # adapt to your utils
    agent_idx_range = range(args.agent_start, args.agent_end + 1)

    train_dataset = V2XSimDet(
        dataset_roots=[os.path.join(args.data_det, f"agent{i}") for i in agent_idx_range],
        config=config,
        config_global=config_global,
        split="train",
        val=False,
        bound="lowerbound",  # or upperbound depending on your setup
        kd_flag=False,
        rsu=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )

    # ------------------------------------------------------------------
    # 2) Build detection model (frozen)
    # ------------------------------------------------------------------
    det_model = build_model_from_config(config, config_global)  # placeholder
    det_model.load_state_dict(torch.load(args.ckpt, map_location=device))
    det_model.to(device)
    det_model.eval()
    for p in det_model.parameters():
        p.requires_grad = False

    # ------------------------------------------------------------------
    # 3) Build state index (GNSS/IMU)
    # ------------------------------------------------------------------
    state_index = StateIndex(
        dataroot=args.data_raw,
        scene_start=args.scene_start,
        scene_end=args.scene_end,
        agent_start=args.agent_start,
        agent_end=args.agent_end,
    )

    # ------------------------------------------------------------------
    # 4) Build selector policy
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

    first_fname = filenames0[0][0]
    scene_id0, frame_id0 = get_scene_frame_from_filename(first_fname)
    agent_ids0 = get_agent_ids_from_filenames(filenames0)
    state_feats0, _ordered_ids0 = build_state_features(
        state_index=state_index,
        scene_id=scene_id0,
        frame_id=frame_id0,
        agent_ids=agent_ids0,
    )
    input_dim = state_feats0.shape[1]

    selector = AgentSelectorMLP(input_dim=input_dim, hidden_dim=args.selector_hidden_dim).to(device)
    optimizer = torch.optim.Adam(selector.parameters(), lr=args.lr)

    # Baseline for REINFORCE (EMA of reward)
    baseline = 0.0
    beta = args.reward_baseline_momentum

    # ------------------------------------------------------------------
    # 5) RL training loop (per-frame bandit)
    # ------------------------------------------------------------------
    for epoch in range(args.epochs):
        selector.train()
        total_loss = 0.0
        total_frames = 0

        for batch in train_loader:
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

            # batch_size is assumed 1 for now
            filenames = filenames[0]
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

            # --- build state features ---
            first_fname = filenames[0]
            scene_id, frame_id = get_scene_frame_from_filename(first_fname)
            agent_ids = get_agent_ids_from_filenames([(f,) for f in filenames])

            state_feats, ordered_agent_ids = build_state_features(
                state_index=state_index,
                scene_id=scene_id,
                frame_id=frame_id,
                agent_ids=agent_ids,
            )
            state_feats = state_feats.to(device)  # (N, D)

            # --- policy: logits -> probs -> Bernoulli sample ---
            logits = selector(state_feats)           # (N,)
            probs = torch.sigmoid(logits)            # (N,)
            m = torch.distributions.Bernoulli(probs)
            actions = m.sample()                     # (N,)
            log_probs = m.log_prob(actions)         # (N,)

            selected_indices = actions.nonzero(as_tuple=True)[0].cpu().tolist()
            num_selected = len(selected_indices)

            # if num_selected == 0:  # you explicitly said NO hardcoded zero-case
            #     ...

            # --- filter detection inputs according to selected agents ---
            def _sel(lst):
                return [lst[i] for i in selected_indices]

            if num_selected > 0:
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
            else:
                # no agents selected -> skip this frame
                continue

            # stack / cat exactly as in your test script
            trans_matrices = torch.stack(tuple(trans_mats_list_sel), 1).to(device)
            target_agent_ids = torch.stack(tuple(target_agent_id_list_sel), 1).to(device)
            num_all_agents = torch.tensor([[num_selected]], dtype=torch.int64, device=device)

            # if you use RSU logic, apply here
            # if not args.rsu:
            #     num_all_agents -= 1

            # choose teacher or normal bev (here just use normal)
            padded_voxel_points = torch.cat(tuple(pvp_list_sel), 0).to(device)
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
                det_loss, cls_loss, loc_loss, _ = det_model.predict_all(
                    data, batch_size=1, num_agent=int(num_all_agents.item())
                )

            det_loss_val = float(det_loss.item())
            # reward: lower loss is better; penalize number of selected agents
            reward = -det_loss_val - args.lambda_cost * float(num_selected)

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
