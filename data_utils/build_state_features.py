import math
import torch
from typing import List, Tuple, Union
from .state_index import StateIndex, AgentMeta

_R = 70.0   # lidar_max_range_m (constant across all V2X-Sim agents)
_V = 30.0   # velocity normalisation ceiling (m/s)


def build_state_features(
    state_index: StateIndex,
    scene_id: int,
    frame_id: int,
    return_meta: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, List[AgentMeta]]]:
    """
    Get per-agent engineered state features for the given scene/frame.

    Returns an (N, 12) float32 tensor clipped to [-3, 3].
    Each selector strategy reads whichever indices it needs.

    Feature layout:
        0: x / R
        1: y / R
        2: sqrt(x² + y²) / R        normalised range to RSU  ← closest_k / heuristic
        3: sin(bearing)
        4: cos(bearing)
        5: sin(yaw_rad)
        6: cos(yaw_rad)
        7: sin(rel_heading)
        8: cos(rel_heading)
        9: vx / V
       10: vy / V
       11: speed / V                 ← velocity_based selector

    where yaw_rad = yaw * π / 180  (yaw stored in degrees in the state JSON),
          bearing = atan2(y, x)     (0 when range < 1e-6, RSU at origin),
          rel_heading = yaw_rad - bearing.
    """
    metas: List[AgentMeta] = state_index.get_agents_meta(scene_id, frame_id)

    if not metas:
        feats = torch.zeros((0, 12), dtype=torch.float32)
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

        vec = torch.tensor(
            [
                x / _R,                    # 0
                y / _R,                    # 1
                range_val / _R,            # 2  ← range
                math.sin(bearing),         # 3
                math.cos(bearing),         # 4
                math.sin(yaw_rad),         # 5
                math.cos(yaw_rad),         # 6
                math.sin(rel_heading),     # 7
                math.cos(rel_heading),     # 8
                meta.vx / _V,             # 9
                meta.vy / _V,             # 10
                meta.speed / _V,          # 11 ← speed
            ],
            dtype=torch.float32,
        )
        feat_rows.append(vec)

    feats = torch.stack(feat_rows, dim=0).clamp(-3.0, 3.0)  # (N, 12)

    if return_meta:
        return feats, metas
    return feats


def build_selector_features(metas: List[AgentMeta]) -> torch.Tensor:
    """
    Compute the 12-feature tensor from a list of AgentMeta objects directly,
    without going through a StateIndex.  Useful when metas are already in hand.

    Returns (N, 12) float32, clipped to [-3, 3].  Same layout as
    build_state_features.
    """
    if not metas:
        return torch.zeros((0, 12), dtype=torch.float32)

    feat_rows = []
    for meta in metas:
        x, y = float(meta.x), float(meta.y)
        yaw_rad = meta.yaw * math.pi / 180.0

        range_val = math.sqrt(x * x + y * y)
        bearing = math.atan2(y, x) if range_val >= 1e-6 else 0.0
        rel_heading = yaw_rad - bearing

        vec = torch.tensor(
            [
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
            ],
            dtype=torch.float32,
        )
        feat_rows.append(vec)

    return torch.stack(feat_rows, dim=0).clamp(-3.0, 3.0)
