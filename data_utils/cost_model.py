"""Communication cost model for cloud-based V2X cooperative perception.

In this architecture, each vehicle uploads a LiDAR bird's-eye-view (BEV) tensor
to a cloud server. The server selects a subset of agents, fuses their BEVs,
runs a 3D object detector, and returns bounding boxes to the fleet. This module
quantifies the communication cost of that pipeline as two physically distinct
components—bandwidth (total upload payload in MB) and latency (upload time plus
measured cloud inference time in ms)—and combines them into a single scalar for
rewards, plots, and scenario analysis. Scenario multipliers (``CostScenario``)
let the same experimental run be re-scored under different operating assumptions
without re-running the detector.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CostScenario(Enum):
    """Operating assumptions for weighting bandwidth vs. latency."""

    BANDWIDTH_CRITICAL = "bandwidth_critical"
    LATENCY_CRITICAL = "latency_critical"
    BALANCED = "balanced"


@dataclass
class CostConfig:
    """Parameters for ``compute_cost``"""

    alpha_bw: float = 1.0
    alpha_lat: float = 1.0
    lat_upload_per_agent_ms: float = 20.0

    @classmethod
    def from_scenario(cls, scenario: CostScenario) -> CostConfig:
        if scenario == CostScenario.BANDWIDTH_CRITICAL:
            return cls(alpha_bw=1.0, alpha_lat=0.0)
        if scenario == CostScenario.LATENCY_CRITICAL:
            return cls(alpha_bw=0.0, alpha_lat=1.0)
        if scenario == CostScenario.BALANCED:
            return cls(alpha_bw=0.5, alpha_lat=0.5)
        raise ValueError(f"Unknown scenario: {scenario}")


@dataclass
class CommunicationCost:
    """Result of ``compute_cost`` for one frame / selection decision."""

    n_selected: int
    bandwidth_MB: float
    latency_upload_ms: float
    latency_proc_ms: float
    latency_total_ms: float
    combined_cost: float

    def __str__(self) -> str:
        return (
            f"agents={self.n_selected} | bw={self.bandwidth_MB:.3f} MB | "
            f"lat={self.latency_total_ms:.1f} ms | cost={self.combined_cost:.5f}"
        )


def compute_cost(
    n_selected: int,
    config: CostConfig,
    bev_size_bytes: int,
    inference_time_ms: float,
) -> CommunicationCost:
    """Compute bandwidth, latency, and combined cost for ``n_selected`` agents."""
    if bev_size_bytes < 0:
        raise ValueError("bev_size_bytes must be non-negative")
    if inference_time_ms < 0:
        raise ValueError("inference_time_ms must be non-negative")

    bandwidth_MB = n_selected * bev_size_bytes / 1_000_000

    latency_upload_ms = config.lat_upload_per_agent_ms if n_selected > 0 else 0
    latency_proc_ms = inference_time_ms
    latency_total_ms = latency_upload_ms + latency_proc_ms

    combined_cost = config.alpha_bw * bandwidth_MB + config.alpha_lat * (
        latency_total_ms / 1000.0
    )

    return CommunicationCost(
        n_selected=n_selected,
        bandwidth_MB=bandwidth_MB,
        latency_upload_ms=latency_upload_ms,
        latency_proc_ms=latency_proc_ms,
        latency_total_ms=latency_total_ms,
        combined_cost=combined_cost,
    )


def bev_size_from_tensor(bev_tensor) -> int:
    """Byte size of one agent's BEV slice from a stacked multi-agent tensor.

    Args:
        bev_tensor: Tensor whose first dimension is the number of agents, e.g.
            shape ``(N_agents, T, H, W, C)`` or ``(N_agents, ...)``.

    Returns:
        ``element_size * (numel // shape[0])``. Set ``CostConfig.bev_size_bytes``
        to this value before calling ``compute_cost``.
    """
    import torch

    return bev_tensor.element_size() * (bev_tensor.numel() // bev_tensor.shape[0])


BANDWIDTH_CRITICAL_CFG = CostConfig.from_scenario(CostScenario.BANDWIDTH_CRITICAL)
LATENCY_CRITICAL_CFG = CostConfig.from_scenario(CostScenario.LATENCY_CRITICAL)
BALANCED_CFG = CostConfig.from_scenario(CostScenario.BALANCED)


if __name__ == "__main__":
    cfg = CostConfig.from_scenario(CostScenario.BALANCED)
    example_bev_size_bytes = 1_572_864

    for n in range(6):
        example_inference_ms = 42.5 * n
        result = compute_cost(n, cfg, bev_size_bytes=example_bev_size_bytes, inference_time_ms=example_inference_ms)
        print(result)
