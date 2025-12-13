import torch
import torch.nn.functional as F
from typing import List, Sequence


def _get_voxel_params(config):
    voxel_size = getattr(config, "voxel_size", None)
    area_extents = getattr(config, "area_extents", None)
    if voxel_size is None:
        voxel_size = [0.4, 0.4, 0.4]
    if area_extents is None:
        area_extents = [[-140.8, 140.8], [-40.0, 40.0], [-3.0, 1.0]]
    return torch.tensor(voxel_size, dtype=torch.float32), torch.tensor(
        area_extents, dtype=torch.float32
    )


def _match_template_shape(bev: torch.Tensor, template_shape: torch.Size) -> torch.Tensor:
    """Reshape or expand the BEV to match the expected detector template shape."""
    if bev.dim() == 4:
        bev = bev.unsqueeze(0)
    if bev.dim() == 5 and bev.shape[1] == 1 and template_shape[1] > 1:
        bev = bev.repeat(1, template_shape[1], 1, 1, 1)
    if bev.shape[2] < template_shape[2]:
        pad_channels = template_shape[2] - bev.shape[2]
        padding = torch.zeros(
            bev.shape[0], bev.shape[1], pad_channels, bev.shape[3], bev.shape[4], device=bev.device
        )
        bev = torch.cat([bev, padding], dim=2)
    elif bev.shape[2] > template_shape[2]:
        bev = bev[:, :, : template_shape[2]]
    if bev.shape[-2:] != template_shape[-2:]:
        h_pad = max(0, template_shape[-2] - bev.shape[-2])
        w_pad = max(0, template_shape[-1] - bev.shape[-1])
        if h_pad or w_pad:
            bev = F.pad(bev, (0, w_pad, 0, h_pad))
        bev = bev[:, :, :, : template_shape[-2], : template_shape[-1]]
    return bev


def _voxelize_points(point_clouds: List[torch.Tensor], trans_matrices: List[torch.Tensor], config, template: torch.Tensor) -> torch.Tensor:
    """Fuse raw point clouds by warping them into the ego frame then voxelizing."""
    voxel_size, area_extents = _get_voxel_params(config)
    device = template.device
    fused_points: List[torch.Tensor] = []
    for pts, trans in zip(point_clouds, trans_matrices):
        if pts is None:
            continue
        pts_t = torch.as_tensor(pts, device=device, dtype=torch.float32)
        if pts_t.numel() == 0:
            continue
        if pts_t.dim() > 2:
            pts_t = pts_t.view(-1, pts_t.shape[-1])
        if pts_t.shape[-1] < 3:
            continue
        trans_t = torch.as_tensor(trans, device=device, dtype=torch.float32)
        while trans_t.dim() > 2:
            trans_t = trans_t[0]
        homo = torch.cat([pts_t[:, :3], torch.ones((pts_t.shape[0], 1), device=device)], dim=1).t()
        pts_ego = (trans_t @ homo).t()
        intensity = pts_t[:, 3:4] if pts_t.shape[1] > 3 else torch.ones((pts_t.shape[0], 1), device=device)
        fused_points.append(torch.cat([pts_ego[:, :3], intensity], dim=1))

    if not fused_points:
        return template

    fused_points = torch.cat(fused_points, dim=0)

    try:
        from coperception.utils.data_util import voxelize_occupy

        bev = voxelize_occupy(
            fused_points.detach().cpu().numpy(),
            voxel_size.detach().cpu().numpy(),
            area_extents.detach().cpu().numpy(),
        )
        bev_tensor = torch.from_numpy(bev).to(device=device, dtype=torch.float32)
        if bev_tensor.dim() == 3:
            bev_tensor = bev_tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
        return _match_template_shape(bev_tensor, template.shape)
    except Exception:
        lower_bound = area_extents[:, 0].to(device)
        upper_bound = area_extents[:, 1].to(device)
        mask = (fused_points[:, :3] >= lower_bound) & (fused_points[:, :3] < upper_bound)
        mask = mask.all(dim=1)
        fused_points = fused_points[mask]
        if fused_points.numel() == 0:
            return template
        idx = ((fused_points[:, :3] - lower_bound) / voxel_size.to(device)).long()
        z_bins = int((upper_bound[2] - lower_bound[2]) / voxel_size[2])
        y_bins = int((upper_bound[1] - lower_bound[1]) / voxel_size[1])
        x_bins = int((upper_bound[0] - lower_bound[0]) / voxel_size[0])
        bev_tensor = torch.zeros((1, 1, z_bins, y_bins, x_bins), device=device)
        bev_tensor[0, 0, idx[:, 2], idx[:, 1], idx[:, 0]] = 1.0
        bev_tensor = bev_tensor.max(dim=2, keepdim=True).values
        return _match_template_shape(bev_tensor, template.shape)


def _default_template(config, device: torch.device, anchors_map_list: Sequence[torch.Tensor]):
    """Return a zero BEV matching config voxel extents when no template exists."""

    voxel_size, area_extents = _get_voxel_params(config)
    x_bins = int((area_extents[0, 1] - area_extents[0, 0]) / voxel_size[0])
    y_bins = int((area_extents[1, 1] - area_extents[1, 0]) / voxel_size[1])
    template = torch.zeros((1, 1, 1, y_bins, x_bins), device=device, dtype=torch.float32)

    # If we have anchors, try to align spatial dimensions to them while keeping
    # a single BEV channel, so downstream shapes stay consistent.
    for anchors in anchors_map_list:
        if anchors is None:
            continue
        anchors_t = torch.as_tensor(anchors, device=device)
        if anchors_t.numel() == 0 or anchors_t.dim() < 2:
            continue
        h, w = anchors_t.shape[-2], anchors_t.shape[-1]
        if h and w:
            template = template[..., :h, :w]
        break

    return template

def _ensure_bev_channels_first(bev: torch.Tensor) -> torch.Tensor:
    """Convert BEV tensor to (B, T, C, H, W) if it is channel-last."""

    if bev.dim() == 5 and bev.shape[2] > bev.shape[4] and bev.shape[3] > bev.shape[4]:
        bev = bev.permute(0, 1, 4, 2, 3)
    return bev

def _compute_affine(trans: torch.Tensor, voxel_size: torch.Tensor, area_extents: torch.Tensor, device) -> torch.Tensor:
    trans_t = torch.as_tensor(trans, device=device, dtype=torch.float32)
    while trans_t.dim() > 2:
        trans_t = trans_t[0]
    theta = torch.zeros((1, 2, 3), device=device)
    theta[:, :, :2] = trans_t[:2, :2]
    x_range = area_extents[0, 1] - area_extents[0, 0]
    y_range = area_extents[1, 1] - area_extents[1, 0]
    theta[:, 0, 2] = 2.0 * trans_t[0, 3] / x_range
    theta[:, 1, 2] = 2.0 * trans_t[1, 3] / y_range
    return theta


def _warp_and_merge_bevs(bev_list: Sequence[torch.Tensor], trans_matrices: List[torch.Tensor], config, template: torch.Tensor) -> torch.Tensor:
    voxel_size, area_extents = _get_voxel_params(config)
    device = template.device
    merged = None
    template_shape = template.shape
    for bev, trans in zip(bev_list, trans_matrices):
        if bev is None:
            continue
        bev_t = torch.as_tensor(bev, device=device, dtype=torch.float32)
        bev_t = _ensure_bev_channels_first(bev_t)
        if bev_t.dim() == 5:
            b, t, c, h, w = bev_t.shape
            bev_t = bev_t.view(b, t * c, h, w)
        elif bev_t.dim() == 4:
            pass
        else:
            continue
        theta = _compute_affine(trans, voxel_size, area_extents, device)
        grid = F.affine_grid(theta, bev_t.shape, align_corners=False)
        warped = F.grid_sample(bev_t, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        if merged is None:
            merged = warped
        else:
            merged = merged + warped
    if merged is None:
        return template
    merged = merged.unsqueeze(1) if merged.dim() == 4 else merged
    merged = merged.view(template_shape[0], template_shape[1], -1, template_shape[3], template_shape[4])
    return _match_template_shape(merged, template_shape)


def _merge_spatial_tensors_max(tensors: Sequence[torch.Tensor], device, dtype=torch.float32):
    """Merge a list of spatial tensors with a max-reduction, skipping empties."""
    merged = None
    for t in tensors:
        if t is None:
            continue
        t_t = torch.as_tensor(t, device=device, dtype=dtype)
        if t_t.numel() == 0:
            continue
        merged = t_t if merged is None else torch.max(merged, t_t)
    return merged


def _merge_regression_targets(reg_targets: Sequence[torch.Tensor], reg_masks: Sequence[torch.Tensor], device):
    """Merge regression targets using the first available target per location."""
    targets_t = []
    masks_t = []
    def _squeeze_leading_ones(t: torch.Tensor, max_dim: int):
        while t.dim() > max_dim and t.shape[0] == 1:
            t = t.squeeze(0)
        return t

    for tgt, msk in zip(reg_targets, reg_masks):
        if tgt is None or msk is None:
            continue
        tgt_t = _squeeze_leading_ones(torch.as_tensor(tgt, device=device, dtype=torch.float32), 5)
        msk_t = _squeeze_leading_ones(torch.as_tensor(msk, device=device), 5)
        if tgt_t.numel() == 0 or msk_t.numel() == 0:
            continue
        targets_t.append(tgt_t)
        masks_t.append(msk_t.bool())
    if not targets_t:
        return None, None

    merged_mask = torch.zeros_like(masks_t[0], dtype=torch.bool, device=device)
    merged_target = torch.zeros_like(targets_t[0], device=device, dtype=torch.float32)

    for tgt, msk in zip(targets_t, masks_t):
        # Ensure mask broadcasts over the regression code dimension.
        msk_b = msk
        while msk_b.dim() < tgt.dim():
            msk_b = msk_b.unsqueeze(-1)
        merged_target = torch.where(msk_b, tgt, merged_target)
        merged_mask = merged_mask | msk

    return merged_target, merged_mask


def _normalize_anchor_map(anchor, device):
    """Ensure anchor map is (H, W, num_anchors, box_code) without extra singleton dims."""
    if anchor is None:
        return None
    a = torch.as_tensor(anchor, device=device, dtype=torch.float32)
    # drop leading batch/time dims of size 1
    while a.dim() > 4 and a.shape[0] == 1:
        a = a.squeeze(0)
    if a.dim() > 4 and a.shape[0] != 1 and a.shape[1] == 1:
        a = a.squeeze(1)
    # drop pred_len dim if present
    if a.dim() > 4 and a.shape[-2] == 1:
        a = a.squeeze(-2)
    return a


def assemble_detection_inputs(
        config,
        padded_voxel_point_list: Sequence[torch.Tensor],
        padded_voxel_points_teacher_list: Sequence[torch.Tensor],
        label_one_hot_list: Sequence[torch.Tensor],
        reg_target_list: Sequence[torch.Tensor],
        reg_loss_mask_list: Sequence[torch.Tensor],
        anchors_map_list: Sequence[torch.Tensor],
        vis_maps_list: Sequence[torch.Tensor],
        target_agent_id_list: Sequence[torch.Tensor],
        num_agent_list: Sequence[torch.Tensor],
        trans_matrices_list: Sequence[torch.Tensor],
        device: torch.device,
):
    """
    Assemble detector inputs after agent selection by fusing selected agents' BEVs.

    The fusion follows the same alignment/voxelization strategy used by the
    detection data builder (``create_data_det``): selected agents' raw lidar
    points are first transformed into the reference (ego/RSU) frame via the
    provided transformation matrices, merged, and voxelized. When only
    precomputed BEVs are available, they are warped into the ego frame using the
    same transform parameters before being combined. The resulting fused BEV
    tensor replaces the simple per-agent stack so that detector inputs reflect
    only the chosen agents' data.
    """

    num_selected = len(trans_matrices_list)

    def _first_nonempty(tensors: Sequence[torch.Tensor]):
        for cand in tensors:
            if cand is None:
                continue
            cand_t = torch.as_tensor(cand, device=device, dtype=torch.float32)
            if cand_t.numel() == 0:
                continue
            return cand_t
        return None

    anchors_map_list_norm = [_normalize_anchor_map(a, device) for a in anchors_map_list]

    template_bev = _first_nonempty(padded_voxel_points_teacher_list)
    if template_bev is None:
        template_bev = _first_nonempty(padded_voxel_point_list)
    if template_bev is None:
        template_bev = _default_template(config, device, anchors_map_list_norm)
    else:
        template_bev = _ensure_bev_channels_first(template_bev)
        if template_bev.dim() == 4:
            template_bev = template_bev.unsqueeze(0)

    has_raw_point_clouds = bool(padded_voxel_point_list) and padded_voxel_point_list[0] != [] and padded_voxel_point_list[0].dim() <= 3
    has_teacher_bevs = bool(padded_voxel_points_teacher_list) and padded_voxel_points_teacher_list[0] != [] and torch.as_tensor(padded_voxel_points_teacher_list[0]).numel() > 0
    has_bev_points = bool(padded_voxel_point_list) and padded_voxel_point_list[0] != [] and padded_voxel_point_list[0].dim() >= 4

    if has_raw_point_clouds:
        fused_bev = _voxelize_points(
            list(padded_voxel_point_list), list(trans_matrices_list), config, template_bev
        ).to(device)
    elif has_teacher_bevs:
        fused_bev = _warp_and_merge_bevs(
            list(padded_voxel_points_teacher_list), list(trans_matrices_list), config, template_bev
        ).to(device)
    elif has_bev_points:
        bev_points_cf = [_ensure_bev_channels_first(torch.as_tensor(b, device=device, dtype=torch.float32)) if b is not None else None for b in padded_voxel_point_list]
        fused_bev = _warp_and_merge_bevs(
            bev_points_cf, list(trans_matrices_list), config, template_bev
        ).to(device)
    else:
        fused_bev = template_bev.to(device)

    if fused_bev.dim() == 4:
        fused_bev = fused_bev.unsqueeze(0)

    fused_bev = fused_bev.permute(0, 1, 3, 4, 2)  # (1, T, H, W, C)

    merged_labels = _merge_spatial_tensors_max(label_one_hot_list, device)
    merged_anchors = _merge_spatial_tensors_max(anchors_map_list_norm, device)
    merged_vis_maps = _merge_spatial_tensors_max(vis_maps_list, device)
    merged_reg_target, merged_reg_mask = _merge_regression_targets(reg_target_list, reg_loss_mask_list, device)

    label_shape_ref = _first_nonempty(label_one_hot_list)
    reg_target_ref = _first_nonempty(reg_target_list)
    reg_mask_ref = _first_nonempty(reg_loss_mask_list)
    anchor_ref = _first_nonempty(anchors_map_list_norm)
    vis_ref = _first_nonempty(vis_maps_list)

    if merged_labels is None:
        merged_labels = torch.zeros((1, *label_shape_ref.shape), device=device) if label_shape_ref is not None else torch.zeros((1,), device=device)
    else:
        merged_labels = merged_labels.unsqueeze(0)

    if merged_reg_target is None:
        if reg_target_ref is not None and reg_mask_ref is not None:
            merged_reg_target = torch.zeros((*reg_target_ref.shape,), device=device)
            merged_reg_mask = torch.zeros((*reg_mask_ref.shape,), device=device, dtype=torch.bool)
        else:
            merged_reg_target = torch.zeros((1,), device=device)
            merged_reg_mask = torch.zeros((1,), device=device, dtype=torch.bool)
    else:
        merged_reg_mask = merged_reg_mask.bool()

    # Ensure batch dimension only once: targets (B,H,W,A,pred_len,code), mask (B,H,W,A,pred_len)
    if merged_reg_target.dim() == 5:
        merged_reg_target = merged_reg_target.unsqueeze(0)
    if merged_reg_mask.dim() == 4:
        merged_reg_mask = merged_reg_mask.unsqueeze(0)

    if merged_anchors is None:
        merged_anchors = torch.zeros((1, *anchor_ref.shape), device=device) if anchor_ref is not None else torch.zeros((1,), device=device)
    else:
        merged_anchors = merged_anchors.unsqueeze(0)

    if merged_vis_maps is None:
        merged_vis_maps = torch.zeros((1, *vis_ref.shape), device=device) if vis_ref is not None else torch.zeros((1,), device=device)
    else:
        merged_vis_maps = merged_vis_maps.unsqueeze(0)

    target_agent_ids = torch.tensor([[torch.as_tensor(target_agent_id_list[0]).item()]], device=device)
    num_all_agents = torch.tensor([[num_selected]], device=device)
    trans_matrices = torch.eye(4, device=device, dtype=torch.float32).view(1, 1, 4, 4)

    data = {
        "bev_seq": fused_bev,
        "labels": merged_labels,
        "reg_targets": merged_reg_target,
        "anchors": merged_anchors,
        "vis_maps": merged_vis_maps,
        "reg_loss_mask": merged_reg_mask,
        "target_agent_ids": target_agent_ids,
        "num_agent": num_all_agents,
        "trans_matrices": trans_matrices,
    }

    return data
