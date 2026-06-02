"""
RSU-Centric Dataset for cooperative detection training.

For each training frame this dataset:
  1. Loads RSU-frame GT labels from agent0's preprocessed file.
  2. Loads each vehicle agent's own BEV (in that agent's frame) from agentJ's file.
  3. Transforms each vehicle BEV to RSU frame on-the-fly using a **planar
     (Z-preserving) transform**: applies T_{j→RSU} for X and Y but keeps
     vehicle-frame Z, so voxels remain inside the [−3, +2] m grid despite the
     RSU being mounted ~5.5 m above road level.  See `_transform_bev_to_rsu`.
  4. Randomly selects a SUBSET of k vehicles (k ~ uniform[min_agents, max_agents])
     and merges their RSU-frame BEVs, so the model is trained to handle any
     density from sparse (1 agent) to dense (all agents).

Why this dataset exists
-----------------------
The existing checkpoints are:
  - lowerbound/with_rsu : each agent detects in its OWN frame (wrong for RSU-centric eval)
  - upperbound/with_rsu : ALL agents merged in RSU frame (too dense; unseen sparsity at
                          test time when only k<5 agents are selected)

This dataset trains a model that detects robustly regardless of how many agents
the RSU selects, which is exactly the Phase 2 use case.

BEV format
----------
Sparse (H, W, C) = (256, 256, 13) boolean grid stored as occupied-voxel index arrays
(N, 3).  Physical extents: X ∈ [-32, 32] m, Y ∈ [-32, 32] m, Z ∈ [-3, 2] m.
Voxel sizes: 0.25 m (X, Y), 0.4 m (Z).  Array axes after rot90: rows→X, cols→Y.
"""

import os
import math
import random
from multiprocessing import Manager
from typing import List, Optional, Tuple

import numpy as np
from torch.utils.data import Dataset

# ── BEV / voxel constants (must match coperception Config) ─────────────────────
VOXEL_SIZE   = np.array([0.25, 0.25, 0.4], dtype=np.float32)
AREA_EXTENTS = np.array([[-32, 32], [-32, 32], [-3, 2]], dtype=np.float32)
BEV_DIMS     = (256, 256, 13)   # (H, W, C)  rows→X, cols→Y, channels→Z
EPS          = 0.01             # offset within cell for exact round-trip


# ── Sparse-BEV ↔ pseudo-point conversion ───────────────────────────────────────

def _voxel_indices_to_pseudopoints(indices: np.ndarray) -> np.ndarray:
    """
    Convert occupied-voxel indices (N, 3) → pseudo-points (N, 4) [x, y, z, 1].
    Uses EPS offset from the cell floor so the inverse mapping (re-voxelization)
    is exact even when extents / voxel_size is non-integer (Z axis: -3/0.4 = -7.5).
    """
    if indices.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float32)
    h, w, c = indices[:, 0], indices[:, 1], indices[:, 2]
    x = AREA_EXTENTS[0, 0] + (h + EPS) * VOXEL_SIZE[0]   # rows   → X
    y = AREA_EXTENTS[1, 0] + (w + EPS) * VOXEL_SIZE[1]   # cols   → Y
    z = AREA_EXTENTS[2, 0] + (c + EPS) * VOXEL_SIZE[2]   # chans  → Z
    ones = np.ones(len(h), dtype=np.float32)
    return np.stack([x, y, z, ones], axis=1).astype(np.float32)


def _pseudopoints_to_voxel_indices(pts: np.ndarray) -> np.ndarray:
    """
    Convert pseudo-points (N, 4) in physical coords → occupied-voxel indices (M, 3).
    Points outside the grid extent are silently dropped.  Duplicate voxels (two
    points landing in the same cell) are deduplicated.
    """
    if pts.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.int32)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    h = np.floor((x - AREA_EXTENTS[0, 0]) / VOXEL_SIZE[0]).astype(np.int32)
    w = np.floor((y - AREA_EXTENTS[1, 0]) / VOXEL_SIZE[1]).astype(np.int32)
    c = np.floor((z - AREA_EXTENTS[2, 0]) / VOXEL_SIZE[2]).astype(np.int32)

    # Filter out-of-bounds
    valid = (
        (h >= 0) & (h < BEV_DIMS[0]) &
        (w >= 0) & (w < BEV_DIMS[1]) &
        (c >= 0) & (c < BEV_DIMS[2])
    )
    h, w, c = h[valid], w[valid], c[valid]

    # Deduplicate via linear index
    lin = h * BEV_DIMS[1] * BEV_DIMS[2] + w * BEV_DIMS[2] + c
    lin = np.unique(lin)
    h = (lin // (BEV_DIMS[1] * BEV_DIMS[2])).astype(np.int32)
    w = ((lin // BEV_DIMS[2]) % BEV_DIMS[1]).astype(np.int32)
    c = (lin % BEV_DIMS[2]).astype(np.int32)
    return np.stack([h, w, c], axis=1)


def _transform_bev_to_rsu(indices: np.ndarray, T_j_to_rsu: np.ndarray) -> np.ndarray:
    """
    Transform a vehicle's BEV (given as occupied voxel indices) to RSU frame —
    using a **planar (Z-preserving) transform**.

    Why not a full 3-D transform?
    --------------------------------
    The RSU sensor is mounted ~5.5 m above the road (on a pole / gantry).
    A full T_{j→RSU} shifts voxel Z by −5.5 m, pushing every pre-voxelized
    vehicle cell from Z ∈ [−3, +2] m into Z ∈ [−8.5, −3.5] m — entirely below
    the RSU grid floor, leaving the merged BEV empty.

    The fix — planar transform:
    --------------------------------
    1. Convert indices → pseudo-points (x, y, z, 1) in vehicle frame.
    2. Save vehicle-frame Z values.
    3. Apply full T_{j→RSU}: X and Y land in RSU's horizontal coordinate system.
    4. **Restore Z** to the saved vehicle-frame values, discarding the vertical
       offset that would push points out of the grid.
    5. Re-voxelize: X and Y are now RSU-frame, Z stays in [−3, +2] → all voxels
       land inside the grid.

    This is valid because:
    * Vehicles and RSU are all roughly level (flat world, upright sensors), so
      T has a pure-yaw rotation — Z does not contaminate X or Y.
    * GT annotations are purely 2-D (row, col, anchor, class) — no Z label — so
      the absolute Z reference frame of the merged BEV does not affect supervision.
    * At inference the same transform is applied, so train/test distributions match.

    Args:
        indices    : (N, 3) int32   voxel indices in agent-j's own frame
        T_j_to_rsu : (4, 4) float64 rigid transform T_{j→RSU}

    Returns:
        (M, 3) int32  voxel indices in RSU (X, Y) frame, vehicle Z range
                      (M ≤ N after out-of-bounds drop in X / Y)
    """
    pts = _voxel_indices_to_pseudopoints(indices)   # (N, 4)
    z_vehicle = pts[:, 2].copy()                    # save vehicle-frame Z
    pts_rsu = (T_j_to_rsu @ pts.T).T               # (N, 4) full transform
    pts_rsu[:, 2] = z_vehicle                       # restore Z → stays in [-3, 2]
    return _pseudopoints_to_voxel_indices(pts_rsu)  # (M, 3)


def _merge_bev_indices(index_list: List[np.ndarray]) -> np.ndarray:
    """
    Merge multiple (Ni, 3) voxel index arrays into a single deduplicated set.
    """
    if not index_list:
        return np.zeros((0, 3), dtype=np.int32)
    all_idx = np.concatenate(index_list, axis=0)
    if all_idx.shape[0] == 0:
        return all_idx
    lin = (all_idx[:, 0] * BEV_DIMS[1] * BEV_DIMS[2] +
           all_idx[:, 1] * BEV_DIMS[2] +
           all_idx[:, 2])
    lin = np.unique(lin)
    h = (lin // (BEV_DIMS[1] * BEV_DIMS[2])).astype(np.int32)
    w = ((lin // BEV_DIMS[2]) % BEV_DIMS[1]).astype(np.int32)
    c = (lin % BEV_DIMS[2]).astype(np.int32)
    return np.stack([h, w, c], axis=1)


def _indices_to_padded_bev(indices: np.ndarray) -> np.ndarray:
    """
    (N, 3) indices → (1, H, W, C) float32 boolean BEV, rotated 90° CCW to match
    the convention used in the original dataset (np.rot90(arr, 3) == np.rot90(arr, -1)).
    """
    bev = np.zeros(BEV_DIMS, dtype=bool)
    if indices.shape[0] > 0:
        bev[indices[:, 0], indices[:, 1], indices[:, 2]] = True
    bev = np.rot90(bev, 3)                        # match V2XSimDet convention
    return bev[np.newaxis].astype(np.float32)     # (1, H, W, C)


# ── Helper: read a single .npy sample file ─────────────────────────────────────

def _load_sample(path: str):
    """Load a sparse BEV sample dict. Returns None if it's an empty placeholder."""
    data = np.load(path, allow_pickle=True)
    if data == 0:   # empty-agent placeholder
        return None
    return data.item()


# ── Dataset ────────────────────────────────────────────────────────────────────

class V2XSimDetRsuCentric(Dataset):
    """
    Dataset for training a FaFNet-lowerbound model on RSU-frame merged BEVs.

    Constructor args
    ----------------
    data_root : str
        Path to the split directory, e.g. `/path/to/V2X-Sim-det/train`.
        Must contain `agent0/`, `agent1/`, …, `agent5/` sub-directories.
    vehicle_agents : list[int]
        Which agent indices count as *vehicles* (typically [1,2,3,4,5]).
    min_agents : int
        Minimum number of vehicles to include per sample (≥ 1).
    max_agents : int
        Maximum number of vehicles to include per sample (≤ len(vehicle_agents)).
    include_rsu_bev : bool
        If True, always include agent0 (RSU) BEV in the merge.
        Set False to train a model that only sees vehicle BEVs.
    cache_size : int
        Number of RSU GT dicts to keep in memory.  Set 0 to disable.
    """

    def __init__(
        self,
        data_root: str,
        vehicle_agents: Optional[List[int]] = None,
        min_agents: int = 1,
        max_agents: int = 5,
        include_rsu_bev: bool = False,
        cache_size: int = 2000,
    ):
        if vehicle_agents is None:
            vehicle_agents = [1, 2, 3, 4, 5]

        self.data_root      = data_root
        self.vehicle_agents = vehicle_agents
        self.min_agents     = min_agents
        self.max_agents     = min(max_agents, len(vehicle_agents))
        self.include_rsu_bev = include_rsu_bev
        self.cache_size     = cache_size

        # ── collect file paths, sorted by (scene_idx, frame_idx) ──────────────
        def _sorted_files(agent_idx: int) -> List[str]:
            root = os.path.join(data_root, f"agent{agent_idx}")
            dirs = os.listdir(root)
            dirs.sort(key=lambda d: (int(d.split("_")[0]), int(d.split("_")[1])))
            files = []
            for d in dirs:
                full = os.path.join(root, d)
                if os.path.isdir(full):
                    for f in os.listdir(full):
                        if f.endswith(".npy"):
                            files.append(os.path.join(full, f))
            return files

        self.rsu_files      = _sorted_files(0)
        self.vehicle_files  = {j: _sorted_files(j) for j in vehicle_agents}

        # Sanity: all agents must have the same number of frames
        n = len(self.rsu_files)
        for j in vehicle_agents:
            assert len(self.vehicle_files[j]) == n, (
                f"Agent {j} has {len(self.vehicle_files[j])} samples but RSU has {n}"
            )
        self.num_samples = n
        print(f"V2XSimDetRsuCentric: {n} samples, vehicles={vehicle_agents}, "
              f"k~U[{min_agents},{self.max_agents}], include_rsu={include_rsu_bev}")

        # ── GT label shapes (for zero-padding empty frames) ───────────────────
        # Mirrors the shapes computed by V2XSimDet for binary, only_det mode.
        from coperception.configs import Config
        _cfg = Config("train", binary=True, only_det=True)
        _dims       = _cfg.map_dims          # [H, W, C] = [256, 256, 13]
        _anchor_sz  = _cfg.anchor_size
        _pred_len   = _cfg.pred_len
        _box_code   = _cfg.box_code_size
        _cat_num    = _cfg.category_num

        from coperception.utils.obj_util import init_anchors_no_check
        self.anchors_map = init_anchors_no_check(
            _cfg.area_extents, _cfg.voxel_size, _box_code, _anchor_sz
        ).astype(np.float32)

        _H, _W = _dims[0], _dims[1]
        _A      = len(_anchor_sz)
        self.reg_target_shape   = (_H, _W, _A, 1,       _box_code)
        self.label_shape        = (_H, _W, _A)
        self.label_one_hot_shape= (_H, _W, _A, _cat_num)

        # ── memory cache for RSU GT dicts ─────────────────────────────────────
        manager        = Manager()
        self._rsu_cache = manager.dict()

    # ── internal helpers ───────────────────────────────────────────────────────

    def _load_rsu_gt(self, idx: int):
        """Return agent0's sample dict (cached)."""
        if idx in self._rsu_cache:
            return self._rsu_cache[idx]
        d = _load_sample(self.rsu_files[idx])
        if d is not None and len(self._rsu_cache) < self.cache_size:
            self._rsu_cache[idx] = d
        return d

    def _build_label_one_hot(self, gt: dict) -> np.ndarray:
        from coperception.utils.obj_util import init_anchors_no_check
        allocation_mask = gt["allocation_mask"].astype(bool)
        label_sparse    = gt["label_sparse"]

        one_hot = np.zeros((label_sparse.shape[0], self.label_one_hot_shape[-1]))
        for i in range(label_sparse.shape[0]):
            one_hot[i, label_sparse[i]] = 1

        label_one_hot = np.zeros(self.label_one_hot_shape)
        label_one_hot[:, :, :, 0] = 1          # background by default
        label_one_hot[allocation_mask] = one_hot
        return label_one_hot.astype(np.float32)

    def _build_reg_target(self, gt: dict) -> Tuple[np.ndarray, np.ndarray]:
        allocation_mask  = gt["allocation_mask"].astype(bool)
        reg_loss_mask    = gt["reg_loss_mask"].astype(bool)
        reg_target_sparse = gt["reg_target_sparse"]

        reg_target = np.zeros(self.reg_target_shape).astype(reg_target_sparse.dtype)
        reg_target[allocation_mask] = reg_target_sparse
        reg_target[np.bitwise_not(reg_loss_mask)] = 0
        # only_det: already sliced to pred_len=1
        return reg_target[:, :, :, :1, :].astype(np.float32), \
               reg_loss_mask[:, :, :, :1].astype(bool)

    # ── __getitem__ ────────────────────────────────────────────────────────────

    def __getitem__(self, idx: int):
        # ── 1. Load RSU GT ─────────────────────────────────────────────────────
        rsu_gt = self._load_rsu_gt(idx)

        if rsu_gt is None:
            # Empty scene — return zeros
            bev  = np.zeros((1,) + BEV_DIMS, dtype=np.float32)
            loh  = np.zeros(self.label_one_hot_shape, dtype=np.float32)
            loh[:, :, :, 0] = 1
            rt   = np.zeros(self.reg_target_shape[:3] + (1, self.reg_target_shape[-1]),
                            dtype=np.float32)
            rlm  = np.zeros(self.reg_target_shape[:3] + (1,), dtype=bool)
            return (bev, loh, rt, rlm, self.anchors_map, np.zeros(0, dtype=np.float32))

        # ── 2. Randomly select k vehicle agents ────────────────────────────────
        k = random.randint(self.min_agents, self.max_agents)
        chosen = random.sample(self.vehicle_agents, k)

        # Trans-matrices from RSU's file: trans_matrices[j] = T_{j→RSU}
        T_all = rsu_gt["trans_matrices"]  # (6, 4, 4)

        # ── 3. Transform chosen vehicles' BEVs to RSU frame ───────────────────
        rsu_indices_list = []

        if self.include_rsu_bev:
            # RSU's own BEV is already in RSU frame
            rsu_idx = rsu_gt["voxel_indices_0"]
            if rsu_idx.shape[0] > 0:
                rsu_indices_list.append(rsu_idx)

        for j in chosen:
            veh_data = _load_sample(self.vehicle_files[j][idx])
            if veh_data is None:
                continue
            veh_indices = veh_data["voxel_indices_0"]   # agent j's own BEV
            if veh_indices.shape[0] == 0:
                continue
            T = T_all[j].astype(np.float64)             # T_{j→RSU}
            rsu_veh_indices = _transform_bev_to_rsu(veh_indices, T)
            if rsu_veh_indices.shape[0] > 0:
                rsu_indices_list.append(rsu_veh_indices)

        merged_indices = _merge_bev_indices(rsu_indices_list)

        # ── 4. Pack into padded BEV ────────────────────────────────────────────
        padded_bev = _indices_to_padded_bev(merged_indices)  # (1, H, W, C)

        # ── 5. Build GT tensors from RSU's labels ──────────────────────────────
        label_one_hot          = self._build_label_one_hot(rsu_gt)
        reg_target, reg_loss_mask = self._build_reg_target(rsu_gt)

        # vis_maps not used (use_vis=False by default)
        vis_maps = np.zeros(0, dtype=np.float32)

        return (
            padded_bev,
            label_one_hot,
            reg_target,
            reg_loss_mask,
            self.anchors_map,
            vis_maps,
        )

    def __len__(self) -> int:
        return self.num_samples
