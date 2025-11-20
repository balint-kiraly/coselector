import torch
from enum import Enum
from typing import List


class SelectionMethod(Enum):
    IDENTITY = "identity"          # keep all agents
    CLOSEST_K = "closest_k"        # pick K nearest by distance or speed
    VELOCITY_BASED = "velocity"    # pick based on motion
    HEURISTIC = "heuristic"        # any rule-based logic
    ML_MODEL = "ml_model"          # learned model (MLP / RL policy)
    BANDWIDTH_AWARE = "bandwidth"  # account for per-agent data cost


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
        kwargs: method-specific arguments (e.g. K=3 for closest_k)

    Returns:
        selected_indices: list of agent indices (local indices in this frame)
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


# ----------------------------------------------------------------------
# Selection method stubs (implement later)
# ----------------------------------------------------------------------

def select_identity(state_features: torch.Tensor) -> List[int]:
    """Keep all agents."""
    N = state_features.shape[0]
    return list(range(N))


def select_closest_k(state_features: torch.Tensor, K: int = 3) -> List[int]:
    """Pick K agents closest to the ego (requires distance feature)."""
    # TODO: implement
    return list(range(state_features.shape[0]))


def select_velocity_based(state_features: torch.Tensor, K: int = 3) -> List[int]:
    """Pick K agents based on speed / motion."""
    # TODO: implement
    return list(range(state_features.shape[0]))


def select_heuristic(state_features: torch.Tensor, **kwargs) -> List[int]:
    """General rule-based selector."""
    # TODO: implement
    return list(range(state_features.shape[0]))


def select_ml_model(state_features: torch.Tensor, model=None, threshold=0.5, **kwargs) -> List[int]:
    """Use a trained neural model to pick agents."""
    # TODO: implement
    return list(range(state_features.shape[0]))


def select_bandwidth_aware(state_features: torch.Tensor, budget: float = None, **kwargs) -> List[int]:
    """Select agents under a bandwidth constraint."""
    # TODO: implement
    return list(range(state_features.shape[0]))
