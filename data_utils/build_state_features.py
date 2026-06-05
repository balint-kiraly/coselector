import math
import torch
from typing import List, Tuple, Union
from .state_index import StateIndex, AgentMeta

# ── Feature-layout constants (used by selectors) ──────────────────────────────
RAW_DIM   = 14   # raw GNSS/IMU fields  — indices  0-13
ML_DIM    = 12   # engineered features  — indices 14-25
TOTAL_DIM = RAW_DIM + ML_DIM   # 26

_R = 70.0   # lidar_max_range_m normalisation
_V = 30.0   # velocity normalisation (m/s)

# Raw-field indices (into columns 0-13)
RAW_X     = 0
RAW_Y     = 1
RAW_Z     = 2
RAW_YAW   = 3
RAW_VX    = 4
RAW_VY    = 5
RAW_SPEED = 6   # ← select_velocity_based

# Engineered-feature indices (into columns 14-25)
ML_START        = RAW_DIM  # 14
ML_X_NORM       = ML_START + 0    # x / R
ML_Y_NORM       = ML_START + 1    # y / R
ML_RANGE_NORM   = ML_START + 2    # sqrt(x²+y²) / R  ← closest_k / heuristic
ML_SIN_BEARING  = ML_START + 3
ML_COS_BEARING  = ML_START + 4
ML_SIN_YAW      = ML_START + 5
ML_COS_YAW      = ML_START + 6
ML_SIN_RELHDG   = ML_START + 7
ML_COS_RELHDG   = ML_START + 8
ML_VX_NORM      = ML_START + 9
ML_VY_NORM      = ML_START + 10
ML_SPEED_NORM   = ML_START + 11   # speed / V


def build_state_features(
    state_index: StateIndex,
    scene_id: int,
    frame_id: int,
    return_meta: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, List[AgentMeta]]]:
    """
    Get per-agent features for the given scene/frame.

    Returns an (N, 26) float32 tensor:
      columns  0-13 : raw GNSS/IMU fields  (x, y, z, yaw, vx, vy, speed,
                       yaw_rate, ax, ay, az, gx, gy, gz)
      columns 14-25 : 12 engineered features clipped to [-3, 3]
                       (x/R, y/R, range/R, sin/cos bearing, sin/cos yaw_rad,
                        sin/cos rel_heading, vx/V, vy/V, speed/V)

    Each selector strategy reads whichever columns it needs:
      - closest_k / heuristic : raw x,y (cols 0,1) or range_norm (col 14+2)
      - velocity_based        : raw speed (col 6)
      - ml_model              : engineered features (cols 14:26)
    """
    metas: List[AgentMeta] = state_index.get_agents_meta(scene_id, frame_id)

    if not metas:
        feats = torch.zeros((0, TOTAL_DIM), dtype=torch.float32)
        if return_meta:
            return feats, metas
        return feats

    feat_rows = []
    for meta in metas:
        x, y = float(meta.x), float(meta.y)
        yaw_rad = meta.yaw * math.pi / 180.0

        range_val = math.sqrt(x * x + y * y)
        bearing = math.atan2(y, x) if range_val >= 1e-6 else 0.0
        rel_heading = yaw_rad - bearing

        # ── 14 raw fields ──────────────────────────────────────────────────────
        raw = [
            meta.x,
            meta.y,
            meta.z,
            meta.yaw,
            meta.vx,
            meta.vy,
            meta.speed,
            meta.yaw_rate,
            meta.ax,
            meta.ay,
            meta.az,
            meta.gx,
            meta.gy,
            meta.gz,
        ]

        # ── 12 engineered features ─────────────────────────────────────────────
        eng = [
            x / _R,
            y / _R,
            range_val / _R,
            math.sin(bearing),
            math.cos(bearing),
            math.sin(yaw_rad),
            math.cos(yaw_rad),
            math.sin(rel_heading),
            math.cos(rel_heading),
            meta.vx / _V,
            meta.vy / _V,
            meta.speed / _V,
        ]

        feat_rows.append(torch.tensor(raw + eng, dtype=torch.float32))

    feats = torch.stack(feat_rows, dim=0)           # (N, 26)

    # Clip only the engineered columns; raw fields keep their physical values.
    feats[:, ML_START:] = feats[:, ML_START:].clamp(-3.0, 3.0)

    if return_meta:
        return feats, metas
    return feats


def build_selector_features(metas: List[AgentMeta]) -> torch.Tensor:
    """
    Compute only the 12 engineered ML features from a list of AgentMeta,
    without going through a StateIndex.  Returns (N, 12), clipped to [-3, 3].

    Useful when metas are already in hand and the full (N, 26) tensor is not
    needed (e.g. inside policy.py or for quick scoring).
    """
    if not metas:
        return torch.zeros((0, ML_DIM), dtype=torch.float32)

    feat_rows = []
    for meta in metas:
        x, y = float(meta.x), float(meta.y)
        yaw_rad = meta.yaw * math.pi / 180.0
        range_val = math.sqrt(x * x + y * y)
        bearing = math.atan2(y, x) if range_val >= 1e-6 else 0.0
        rel_heading = yaw_rad - bearing

        feat_rows.append(torch.tensor([
            x / _R, y / _R, range_val / _R,
            math.sin(bearing), math.cos(bearing),
            math.sin(yaw_rad), math.cos(yaw_rad),
            math.sin(rel_heading), math.cos(rel_heading),
            meta.vx / _V, meta.vy / _V, meta.speed / _V,
        ], dtype=torch.float32))

    return torch.stack(feat_rows, dim=0).clamp(-3.0, 3.0)
