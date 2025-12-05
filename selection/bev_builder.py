"""Utilities to build BEV features only for the centrally selected agents.

This module mirrors the voxelization/fusion logic used in CoPerception's
preprocessing, but keeps the selector in the loop: once the RL (or heuristic)
selector decides which agents to keep, we voxelize only those agents' raw
point clouds and hand just those BEVs to the detection model.
"""

from typing import Iterable, List, Optional, Sequence, Tuple

import torch

from coperception.utils.data_util import voxelize_occupy


def voxelize_selected_points(
    raw_point_clouds: Sequence[torch.Tensor],
    selected_indices: Iterable[int],
    config,
    device: torch.device,
) -> torch.Tensor:
    """Voxelize only the selected agents' point clouds.

    Args:
        raw_point_clouds: list/tuple of point clouds (num_points, 4) in lidar frame.
        selected_indices: indices into ``raw_point_clouds`` to keep.
        config: CoPerception detection config with ``vdim``/``vdim_cross_road``.
        device: torch device for the resulting tensor.

    Returns:
        Tensor shaped (num_selected, bev_seq_len, H, W, C) matching the detector
        expectations. If no agent is selected, an empty tensor is returned.
    """

    bev_tensors: List[torch.Tensor] = []

    for idx in selected_indices:
        pc = raw_point_clouds[idx]
        if not torch.is_tensor(pc):
            pc = torch.as_tensor(pc)
        if pc.numel() == 0:
            continue

        # Match the preprocessing used by coperception: voxel occupancy with an
        # optional motion channel depending on the config.
        voxel = voxelize_occupy(pc, config)
        bev_tensors.append(torch.as_tensor(voxel, device=device))

    if not bev_tensors:
        return torch.zeros((0, *config.vdim), device=device)

    return torch.stack(bev_tensors, dim=0)


def assemble_detection_inputs(
    *,
    selected_indices: Sequence[int],
    raw_point_clouds: Optional[Sequence[torch.Tensor]],
    precomputed_bevs: Sequence[torch.Tensor],
    teacher_bevs: Optional[Sequence[torch.Tensor]],
    labels: Sequence[torch.Tensor],
    reg_targets: Sequence[torch.Tensor],
    reg_loss_masks: Sequence[torch.Tensor],
    anchors: Sequence[torch.Tensor],
    vis_maps: Sequence[torch.Tensor],
    trans_matrices_list: Sequence[torch.Tensor],
    target_agent_id_list: Sequence[torch.Tensor],
    device: torch.device,
    config,
    use_teacher_bev: bool = False,
) -> Tuple[dict, int]:
    """
    Build the detector input dict using only the selected agents.

    If ``raw_point_clouds`` is provided, BEVs are voxelized lazily for the
    selected subset; otherwise we fall back to the precomputed BEVs already
    stored on disk. This keeps the selector in front of feature extraction
    while maintaining compatibility with existing detection code.
    """

    if not selected_indices:
        return {}, 0

    def _select(seq: Sequence, use_indices: Sequence[int]):
        return [seq[i] for i in use_indices]

    if raw_point_clouds is not None:
        bev_seq = voxelize_selected_points(raw_point_clouds, selected_indices, config, device)
    else:
        if use_teacher_bev and teacher_bevs is not None:
            bev_seq = torch.cat(tuple(_select(teacher_bevs, selected_indices)), dim=0).to(device)
        else:
            bev_seq = torch.cat(tuple(_select(precomputed_bevs, selected_indices)), dim=0).to(device)

    label_one_hot = torch.cat(tuple(_select(labels, selected_indices)), dim=0).to(device)
    reg_target = torch.cat(tuple(_select(reg_targets, selected_indices)), dim=0).to(device)
    reg_loss_mask = torch.cat(tuple(_select(reg_loss_masks, selected_indices)), dim=0).to(device)
    anchors_map = torch.cat(tuple(_select(anchors, selected_indices)), dim=0).to(device)
    vis_maps = torch.cat(tuple(_select(vis_maps, selected_indices)), dim=0).to(device)

    trans_matrices = torch.stack(tuple(_select(trans_matrices_list, selected_indices)), dim=1).to(device)
    target_agent_ids = torch.stack(tuple(_select(target_agent_id_list, selected_indices)), dim=1).to(device)
    num_agents = len(selected_indices)

    data = {
        "bev_seq": bev_seq,
        "labels": label_one_hot,
        "reg_targets": reg_target,
        "anchors": anchors_map,
        "vis_maps": vis_maps,
        "reg_loss_mask": reg_loss_mask.bool(),
        "target_agent_ids": target_agent_ids,
        "num_agent": torch.tensor([[num_agents]], dtype=torch.int64, device=device),
        "trans_matrices": trans_matrices,
    }

    return data, num_agents
