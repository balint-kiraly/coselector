import math
import numpy as np
import torch
from enum import Enum
from typing import List, Optional


class SelectionMethod(Enum):
    IDENTITY = "identity"          # keep all agents
    CLOSEST_K = "closest_k"        # pick K nearest to RSU
    VELOCITY_BASED = "velocity"    # pick K highest-speed agents (moving targets)
    HEURISTIC = "heuristic"        # maximize angular coverage around RSU
    ML_MODEL = "ml_model"          # learned model (MLP / RL policy)
    BANDWIDTH_AWARE = "bandwidth"  # greedy under a byte-budget constraint


def select_agents_from_metadata(
    state_features: torch.Tensor,
    method: SelectionMethod = SelectionMethod.IDENTITY,
    **kwargs,
) -> List[int]:
    """
    Dispatcher: select agents using the chosen strategy.

    Args:
        state_features: (N_agents, D) tensor of GNSS+IMU features.
        method: SelectionMethod enum value.
        kwargs: method-specific arguments (e.g. K=3 for closest_k).
            rsu_trans: (6, 4, 4) tensor — T_{j→RSU}; used by distance-aware methods.
            bev_list:  list of BEV tensors — used by bandwidth_aware for size estimation.

    Returns:
        selected_indices: list of state-feature row indices (NOT agent IDs).
    """

    if method == SelectionMethod.IDENTITY:
        return select_identity(state_features)
    elif method == SelectionMethod.CLOSEST_K:
        return select_closest_k(state_features, **kwargs)
    elif method == SelectionMethod.VELOCITY_BASED:
        return select_velocity_based(state_features, **kwargs)
    elif method == SelectionMethod.HEURISTIC:
        return select_heuristic(state_features, **kwargs)
    elif method == SelectionMethod.ML_MODEL:
        return select_ml_model(state_features, **kwargs)
    elif method == SelectionMethod.BANDWIDTH_AWARE:
        return select_bandwidth_aware(state_features, **kwargs)
    else:
        raise ValueError(f"Unknown selection method: {method}")


# ─────────────────────────────────────────────────────────────────────────────
# Helper: extract 2-D RSU-frame positions of each state-feature row.
# rsu_trans: (6, 4, 4) tensor where row j = T_{agent_j → RSU}.
# metas:     list of AgentMeta in the same order as state_features.
# Returns (N, 2) float array of [x_rsu, y_rsu] for each agent.
# ─────────────────────────────────────────────────────────────────────────────
def _rsu_positions(state_features: torch.Tensor, rsu_trans: Optional[torch.Tensor], metas) -> np.ndarray:
    N = state_features.shape[0]
    if rsu_trans is not None and len(metas) == N:
        trans_np = rsu_trans.detach().cpu().numpy() if isinstance(rsu_trans, torch.Tensor) else np.asarray(rsu_trans)
        pos = np.zeros((N, 2), dtype=np.float32)
        for i, meta in enumerate(metas):
            j = meta.agent_id  # column index in rsu_trans
            if j < trans_np.shape[0]:
                pos[i, 0] = trans_np[j, 0, 3]
                pos[i, 1] = trans_np[j, 1, 3]
        return pos
    # Fallback: use x, y from state features (indices 0, 1)
    feats = state_features.detach().cpu().numpy()
    return feats[:, :2]


# ─────────────────────────────────────────────────────────────────────────────
# Individual selectors
# ─────────────────────────────────────────────────────────────────────────────

def select_identity(state_features: torch.Tensor, **kwargs) -> List[int]:
    """Keep all available agents."""
    return list(range(state_features.shape[0]))


def select_closest_k(
    state_features: torch.Tensor,
    K: int = 3,
    rsu_trans: Optional[torch.Tensor] = None,
    metas=None,
    **kwargs,
) -> List[int]:
    """
    Pick the K agents closest (Euclidean) to the RSU in RSU frame.
    Uses the translation component of T_{j→RSU} for exact positions.
    """
    N = state_features.shape[0]
    K = min(K, N)
    if K <= 0:
        return []

    pos = _rsu_positions(state_features, rsu_trans, metas if metas else [])
    # RSU is at origin; distance = ||(x, y)||
    dists = np.linalg.norm(pos, axis=1)  # (N,)
    order = np.argsort(dists)            # ascending = closest first
    return order[:K].tolist()


def select_velocity_based(
    state_features: torch.Tensor,
    K: int = 3,
    **kwargs,
) -> List[int]:
    """
    Pick the K agents with the highest instantaneous speed.
    Rationale: fast-moving vehicles observe the most scene change per frame
    and provide complementary views not easily predicted from RSU data alone.
    Raw state feature index 6 = speed (m/s) in the (N, 26) build_state_features
    layout.  Raw fields occupy columns 0-13.
    """
    N = state_features.shape[0]
    K = min(K, N)
    if K <= 0:
        return []
    speeds = state_features[:, 6].detach().cpu().numpy()  # (N,) raw speed m/s
    order = np.argsort(-speeds)   # descending = fastest first
    return order[:K].tolist()


def select_heuristic(
    state_features: torch.Tensor,
    K: int = 3,
    rsu_trans: Optional[torch.Tensor] = None,
    metas=None,
    **kwargs,
) -> List[int]:
    """
    Greedy angular-coverage selector.

    Builds a set of K vehicles whose viewing angles (as seen from the RSU)
    are maximally spread: pick the first vehicle freely, then each subsequent
    vehicle is the one whose bearing from the RSU is farthest from the
    already-chosen bearings (max-min angular gap heuristic).

    This maximises the spatial diversity of the received point clouds and
    reduces the chance that all selected vehicles observe the same region.
    """
    N = state_features.shape[0]
    K = min(K, N)
    if K <= 0:
        return []
    if K == N:
        return list(range(N))

    pos = _rsu_positions(state_features, rsu_trans, metas if metas else [])
    bearings = np.arctan2(pos[:, 1], pos[:, 0])  # angle from RSU, in [-π, π]

    # Greedy selection
    selected = []
    remaining = list(range(N))

    # Start with the agent closest to the RSU (most reliable link)
    dists = np.linalg.norm(pos, axis=1)
    first = int(np.argmin(dists))
    selected.append(first)
    remaining.remove(first)

    while len(selected) < K and remaining:
        # For each remaining agent, compute min angular distance to any selected agent
        best_agent, best_score = None, -1.0
        for i in remaining:
            min_ang = min(
                _angular_dist(bearings[i], bearings[s]) for s in selected
            )
            if min_ang > best_score:
                best_score = min_ang
                best_agent = i
        selected.append(best_agent)
        remaining.remove(best_agent)

    return selected


def _angular_dist(a: float, b: float) -> float:
    """Shortest angular distance between two bearings in radians."""
    diff = abs(a - b) % (2 * math.pi)
    return min(diff, 2 * math.pi - diff)


@torch.no_grad()
def _select_ml(
    features: torch.Tensor,
    model,
    threshold: float = 0.5,
    min_agents: int = 1,
) -> List[int]:
    """Deterministic greedy selection: sigmoid(logits) > threshold."""
    probs = torch.sigmoid(model(features)).squeeze(-1)   # (N,)
    sel = (probs > threshold).nonzero(as_tuple=True)[0].tolist()
    if len(sel) < min_agents:
        sel = [int(probs.argmax())]
    return sel


def select_ml_model(
    state_features: torch.Tensor,
    model=None,
    threshold: float = 0.5,
    **kwargs,
) -> List[int]:
    """
    Use a trained MLP policy to select agents.

    ``state_features`` is the (N, 26) tensor from build_state_features.
    The engineered 12 features live in columns 14:26 and are fed to the model.

    Applies sigmoid to the logit outputs and thresholds at ``threshold``
    (default 0.5).  Sweeping 0.3 / 0.5 / 0.7 produces multiple Pareto points
    from a single trained model without re-running training.

    Falls back to identity (all agents) when ``model`` is None.
    """
    if model is None:
        return list(range(state_features.shape[0]))
    from data_utils.build_state_features import ML_START
    ml_feats = state_features[:, ML_START:]   # (N, 12) engineered columns
    return _select_ml(ml_feats, model, threshold=threshold)


def select_bandwidth_aware(
    state_features: torch.Tensor,
    budget_bytes: int = 2_000_000,
    bev_list=None,
    rsu_trans: Optional[torch.Tensor] = None,
    metas=None,
    **kwargs,
) -> List[int]:
    """
    Greedy bandwidth-constrained selection.

    Agents are ranked by (estimated BEV size in bytes).  We greedily include
    the smallest-payload agents first until the budget is exhausted.  If BEV
    sizes are unavailable we fall back to selecting the K closest agents.

    Args:
        budget_bytes:  maximum total payload in bytes.
        bev_list:      list of BEV tensors (one per agent) for size estimation.
    """
    N = state_features.shape[0]
    if N == 0:
        return []

    # Estimate per-agent BEV size
    sizes = np.zeros(N, dtype=np.float64)
    if bev_list is not None and len(bev_list) >= N:
        for i in range(N):
            bev = bev_list[i]
            if bev is None:
                sizes[i] = float("inf")
            else:
                try:
                    t = torch.as_tensor(bev)
                    sizes[i] = t.numel() * t.element_size()
                except Exception:
                    sizes[i] = float("inf")
    else:
        # No BEV list: assume equal size of 200 kB; sort by distance instead
        pos = _rsu_positions(state_features, rsu_trans, metas if metas else [])
        sizes = np.linalg.norm(pos, axis=1)  # proxy: closer = smaller payload

    # Greedy: cheapest first
    order = np.argsort(sizes)
    selected = []
    running_cost = 0.0
    for i in order:
        if sizes[i] == float("inf"):
            continue
        if running_cost + sizes[i] <= budget_bytes:
            selected.append(int(i))
            running_cost += sizes[i]
        # continue scanning cheaper agents even after one is too big
    # Always keep at least one agent
    if not selected:
        selected = [int(order[0])]
    return selected
