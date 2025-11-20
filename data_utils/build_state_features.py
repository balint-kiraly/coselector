import torch
from typing import List, Tuple
from .state_index import StateIndex, AgentMeta


def build_state_features(
    state_index: StateIndex,
    scene_id: int,
    frame_id: int,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Get per-agent state features for the given scene/frame.
    Returns:
        feats: (N, D) tensor of state features for agents present in this frame.
               If an agent is missing, its features are zeroed.
    """
    metas: List[AgentMeta] = state_index.get_agents_meta(scene_id, frame_id)

    feat_rows = []

    for meta in metas:
        if not meta:
            # agent missing for this frame -> zero features
            vec = torch.zeros(16, dtype=torch.float32)
        else:
            vec = torch.tensor(
                [
                    1, # present
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
                    meta.timestamp,
                ],
                dtype=torch.float32,
            )
        feat_rows.append(vec)

    feats = torch.stack(feat_rows, dim=0)  # (N_agents, 13)
    return feats
