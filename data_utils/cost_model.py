"""Three-axis communication cost model for cloud-based V2X cooperative perception.

Architecture assumed
--------------------
Vehicles transmit **raw LiDAR** (.pcd.bin, ~0.5 MB each) to a cloud server.
The server voxelizes each selected agent's upload into a BEV tensor (precomputation),
fuses them, runs detection model, and returns detection boxes.

Three cost axes
---------------
1. Bandwidth   bandwidth_MB = N_selected × lidar_size_MB
2. Latency     total_latency_ms = upload_ms + precomp_ms + inference_ms
               - upload_ms   = lidar_size_bytes × 8 / link_rate_bps × 1000
                              (one-time upload, N agents transmit in parallel)
               - precomp_ms  = N_selected × precomp_per_agent_ms
               - inference_ms is measured per frame and passed in at call time
3. Energy      total_energy_J = N_selected × precomp_energy_J + inference_energy_J

Combined cost (normalised)
--------------------------
    combined = α_bw     × (bandwidth_MB     / max_bandwidth_MB)
             + α_lat    × (total_latency_ms / max_latency_ms)
             + α_energy × (total_energy_J   / max_energy_J)

All three terms live in [0, 1] (capped) so the α weights are directly
interpretable as relative importance.  α_energy defaults to 0.

Measured constants
------------------
Run the two measurement tools to obtain the constants below:
    python tools/measure_lidar_size.py
    python tools/measure_precomp_cost.py
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Measured physical constants
# ---------------------------------------------------------------------------

#: Mean raw LiDAR file size across V2X-Sim scenes, agents 0-5.
#: Source: tools/measure_lidar_size.py
MEASURED_LIDAR_SIZE_MB: float = 0.4934

#: Mean per-agent voxelization time (CPU, server-side).
#: Source: tools/measure_precomp_cost.py
MEASURED_PRECOMP_PER_AGENT_MS: float = 1.982

#: Mean GPU energy consumed during one agent's voxelization window.
#: Source: tools/measure_precomp_cost.py
MEASURED_PRECOMP_ENERGY_J: float = 0.035252


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class CostConfig:
    """All parameters needed to evaluate the three-axis cost model.

    Measured constants
    ~~~~~~~~~~~~~~~~~~
    ``lidar_size_MB``           Raw LiDAR file size per agent (from measure_lidar_size.py).
    ``precomp_per_agent_ms``    Cloud voxelization time per agent (from measure_precomp_cost.py).
    ``precomp_energy_J``        GPU energy per agent voxelization window (from measure_precomp_cost.py).

    Configurable
    ~~~~~~~~~~~~
    ``link_rate_Mbps``          Uplink speed (Mbps).  Default 10 Mbps = urban V2X C-V2X Uu.
    ``alpha_bw/lat/energy``     Relative weights; must sum to > 0.
    ``alpha_energy``            Set > 0 to enable energy axis (defaults to 0 for 2-axis runs).

    Normalisation denominators
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    ``max_bandwidth_MB``        Bandwidth when all ``n_max_agents`` transmit.
    ``max_latency_ms``          Latency when all agents transmit + reference inference time.
    ``max_energy_J``            Energy when all agents transmit + reference inference energy.

    If any max value is 0.0 (not yet set), the corresponding normalised term is
    returned as 0 so a partially-configured model degrades gracefully.
    """

    # ── Measured constants (fill from measurement tools) ────────────────────
    lidar_size_MB: float = MEASURED_LIDAR_SIZE_MB
    precomp_per_agent_ms: float = MEASURED_PRECOMP_PER_AGENT_MS
    precomp_energy_J: float = MEASURED_PRECOMP_ENERGY_J

    # ── Configurable link parameters ─────────────────────────────────────────
    link_rate_Mbps: float = 10.0   # C-V2X Uu uplink, urban typical

    # ── α weights ────────────────────────────────────────────────────────────
    alpha_bw: float = 1.0
    alpha_lat: float = 1.0
    alpha_energy: float = 0.0      # disabled by default

    # ── Normalisation denominators ───────────────────────────────────────────
    # Bandwidth and energy max values are computed automatically from the
    # measured constants above.  Latency and energy depend on inference time,
    # which must come from an external source:
    #
    #   Option A (recommended): run the identity/all-agents baseline first,
    #     read avg_inference_ms from summary.csv, pass --max_inference_ms=X.
    #
    #   Option B: store the value in measurements/precomp_cost.json under
    #     "inference_norm_ms" and load via from_measurements_dir().
    #
    #   Option C: leave unset (0.0) → latency and energy terms are omitted
    #     from combined_cost; only bandwidth is normalised.
    #
    max_bandwidth_MB: float = 0.0
    max_latency_ms: float = 0.0
    max_energy_J: float = 0.0

    # ── Reference inference values for norm computation ───────────────────
    # Set via compute_normalization_constants() or from_measurements_dir().
    # Left at 0.0 means "not calibrated" — combined_cost will be bandwidth-only.
    inference_norm_ms: float = 0.0
    inference_norm_energy_J: float = 0.0

    # ── Upper-bound agent count ─────────────────
    n_max_agents: int = 5          # vehicles 1-5, RSU is passive

    @classmethod
    def from_measurements_dir(
        cls,
        measurements_dir: str = "measurements",
        **overrides,
    ) -> "CostConfig":
        """Load measured constants from the JSON files produced by the measurement tools.

        Reads:
          ``<measurements_dir>/lidar_size.json``   → lidar_size_MB
          ``<measurements_dir>/precomp_cost.json`` → precomp_per_agent_ms, precomp_energy_J

        Any extra keyword arguments are forwarded to the constructor and override
        the loaded values (e.g. ``link_rate_Mbps=50.0``).

        Raises FileNotFoundError if either JSON is missing — run the tools first.
        """
        lidar_path = os.path.join(measurements_dir, "lidar_size.json")
        precomp_path = os.path.join(measurements_dir, "precomp_cost.json")

        missing = [p for p in (lidar_path, precomp_path) if not os.path.isfile(p)]
        if missing:
            raise FileNotFoundError(
                f"Measurement file(s) not found: {missing}\n"
                "Run tools/measure_lidar_size.py and tools/measure_precomp_cost.py first."
            )

        with open(lidar_path) as f:
            lidar_data = json.load(f)
        with open(precomp_path) as f:
            precomp_data = json.load(f)

        kwargs: dict = dict(
            lidar_size_MB=lidar_data["lidar_size_MB"],
            precomp_per_agent_ms=precomp_data["precomp_per_agent_ms"],
            precomp_energy_J=precomp_data["precomp_energy_J"],
        )
        # Optional inference norm — set by the user after an identity baseline run.
        # Stored under "inference_norm_ms" / "inference_norm_energy_J" in precomp_cost.json.
        if precomp_data.get("inference_norm_ms") is not None:
            kwargs["inference_norm_ms"] = float(precomp_data["inference_norm_ms"])
        if precomp_data.get("inference_norm_energy_J") is not None:
            kwargs["inference_norm_energy_J"] = float(precomp_data["inference_norm_energy_J"])
        kwargs.update(overrides)
        cfg = cls(**kwargs)
        # Auto-compute normalization denominators if inference_norm_ms is available.
        if cfg.inference_norm_ms > 0:
            cfg._compute_norms()
        return cfg

    def compute_normalization_constants(
        self,
        ref_inference_ms: float | None = None,
        ref_inference_energy_J: float | None = None,
    ) -> None:
        """Fix the normalisation denominators for this run.

        The denominators represent the worst-case all-agents scenario so that
        combined_cost ∈ [0, 1] and comparisons across selection strategies are
        on the same scale.

        Recommended workflow
        -------------------
        1. Run the identity/all-agents baseline first::

               make test sel_method=identity

        2. Note ``avg_inference_ms`` from logs/summary.csv.
        3. Pass it as ``--max_inference_ms=<value>`` for all subsequent runs,
           or store it in ``measurements/precomp_cost.json`` under
           ``"inference_norm_ms"`` so it loads automatically.

        If both ``ref_inference_ms`` and ``self.inference_norm_ms`` are
        available, the explicit argument wins.

        Args:
            ref_inference_ms:       All-agents inference latency in ms.
                                    Falls back to ``self.inference_norm_ms``.
            ref_inference_energy_J: All-agents inference energy in J.
                                    Falls back to ``self.inference_norm_energy_J``.
        """
        ms  = ref_inference_ms       if ref_inference_ms       is not None else self.inference_norm_ms
        e_J = ref_inference_energy_J if ref_inference_energy_J is not None else self.inference_norm_energy_J
        if ms is not None and ms <= 0:
            raise ValueError(
                f"ref_inference_ms must be > 0 (got {ms}). "
                "Inference time is 0 when no agents are selected — use the "
                "all-agents baseline value from summary.csv instead."
            )
        if ms and ms > 0:
            self.inference_norm_ms = ms
        if e_J and e_J > 0:
            self.inference_norm_energy_J = e_J
        self._compute_norms()

    def _compute_norms(self) -> None:
        """Internal: recompute max_* from stored inference_norm values."""
        N = self.n_max_agents
        upload_ms = self._upload_ms_one_agent()
        self.max_bandwidth_MB = N * self.lidar_size_MB
        self.max_latency_ms = (
            upload_ms
            + N * self.precomp_per_agent_ms
            + self.inference_norm_ms
        )
        self.max_energy_J = N * self.precomp_energy_J + self.inference_norm_energy_J

    @property
    def norms_calibrated(self) -> bool:
        """True if normalization denominators have been set."""
        return self.inference_norm_ms > 0

    def _upload_ms_one_agent(self) -> float:
        """Upload time for a single agent's raw LiDAR over the configured link."""
        bits = self.lidar_size_MB * 1e6 * 8
        link_bps = self.link_rate_Mbps * 1e6
        return bits / link_bps * 1000.0  # ms


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CommunicationCost:
    """Per-frame cost breakdown returned by ``compute_cost``."""

    n_selected: int

    # Bandwidth axis
    bandwidth_MB: float

    # Latency axis (additive components)
    upload_ms: float
    precomp_ms: float
    inference_ms: float
    latency_total_ms: float

    # Energy axis
    precomp_energy_J: float
    inference_energy_J: float
    total_energy_J: float

    # Normalised combined cost ∈ [0, 1]
    combined_cost: float

    def __str__(self) -> str:
        return (
            f"agents={self.n_selected} | "
            f"bw={self.bandwidth_MB:.3f} MB | "
            f"lat={self.latency_total_ms:.1f} ms "
            f"(upload={self.upload_ms:.1f} + precomp={self.precomp_ms:.1f} + inf={self.inference_ms:.1f}) | "
            f"energy={self.total_energy_J:.4f} J | "
            f"cost={self.combined_cost:.5f}"
        )


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def compute_cost(
    n_selected: int,
    inference_ms: float,
    inference_energy_J: float,
    config: CostConfig,
) -> CommunicationCost:
    """Compute the three-axis cost for a single frame / selection decision.

    Args:
        n_selected:         Number of agents whose LiDAR was uploaded.
        inference_ms:       Measured model inference wall time for this frame.
        inference_energy_J: Measured consumed energy for this frame's inference.
        config:             CostConfig with measured constants and α weights.

    Returns:
        CommunicationCost with all axes populated.
    """
    if n_selected < 0:
        raise ValueError("n_selected must be >= 0")
    if inference_ms < 0:
        raise ValueError("inference_ms must be >= 0")

    # ── 1. Bandwidth ─────────────────────────────────────────────────────────
    bandwidth_MB = n_selected * config.lidar_size_MB

    # ── 2. Latency ───────────────────────────────────────────────────────────
    # Upload is parallel across agents; cost = one agent's upload time.
    upload_ms = config._upload_ms_one_agent() if n_selected > 0 else 0.0
    precomp_ms = n_selected * config.precomp_per_agent_ms
    latency_total_ms = upload_ms + precomp_ms + inference_ms

    # ── 3. Energy ────────────────────────────────────────────────────────────
    total_precomp_energy_J = n_selected * config.precomp_energy_J
    total_energy_J = total_precomp_energy_J + inference_energy_J

    # ── Combined (normalised) ────────────────────────────────────────────────
    def _norm(val: float, max_val: float) -> float:
        if max_val <= 0:
            return 0.0
        return min(val / max_val, 1.0)

    bw_term     = config.alpha_bw     * _norm(bandwidth_MB,    config.max_bandwidth_MB)
    lat_term    = config.alpha_lat    * _norm(latency_total_ms, config.max_latency_ms)
    energy_term = config.alpha_energy * _norm(total_energy_J,  config.max_energy_J)

    weight_sum = config.alpha_bw + config.alpha_lat + config.alpha_energy
    combined_cost = (bw_term + lat_term + energy_term) / max(weight_sum, 1e-9)

    return CommunicationCost(
        n_selected=n_selected,
        bandwidth_MB=bandwidth_MB,
        upload_ms=upload_ms,
        precomp_ms=precomp_ms,
        inference_ms=inference_ms,
        latency_total_ms=latency_total_ms,
        precomp_energy_J=total_precomp_energy_J,
        inference_energy_J=inference_energy_J,
        total_energy_J=total_energy_J,
        combined_cost=combined_cost,
    )


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def _base_config(
    alpha_bw: float,
    alpha_lat: float,
    alpha_energy: float,
    link_rate_Mbps: float,
    measurements_dir: str | None,
) -> CostConfig:
    if measurements_dir is not None:
        return CostConfig.from_measurements_dir(
            measurements_dir,
            alpha_bw=alpha_bw,
            alpha_lat=alpha_lat,
            alpha_energy=alpha_energy,
            link_rate_Mbps=link_rate_Mbps,
        )
    return CostConfig(
        alpha_bw=alpha_bw,
        alpha_lat=alpha_lat,
        alpha_energy=alpha_energy,
        link_rate_Mbps=link_rate_Mbps,
    )


def make_balanced_config(
    link_rate_Mbps: float = 10.0,
    measurements_dir: str | None = None,
) -> CostConfig:
    """Bandwidth and latency weighted equally; energy disabled."""
    return _base_config(1.0, 1.0, 0.0, link_rate_Mbps, measurements_dir)


def make_bandwidth_critical_config(
    link_rate_Mbps: float = 10.0,
    measurements_dir: str | None = None,
) -> CostConfig:
    return _base_config(1.0, 0.0, 0.0, link_rate_Mbps, measurements_dir)


def make_latency_critical_config(
    link_rate_Mbps: float = 10.0,
    measurements_dir: str | None = None,
) -> CostConfig:
    return _base_config(0.0, 1.0, 0.0, link_rate_Mbps, measurements_dir)


# ---------------------------------------------------------------------------
# Backwards-compat shim: bev_size_from_tensor (kept so old imports don't break)
# ---------------------------------------------------------------------------

def bev_size_from_tensor(bev_tensor) -> int:
    """Byte size of one BEV slice from a stacked multi-agent tensor (legacy)."""
    import torch
    return bev_tensor.element_size() * (bev_tensor.numel() // bev_tensor.shape[0])


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = make_balanced_config(link_rate_Mbps=10.0)
    # Simulate: call with a plausible reference inference time
    cfg.compute_normalization_constants(ref_inference_ms=45.0, ref_inference_energy_J=0.05)

    print(f"{'N':>3}  {'BW MB':>8}  {'lat ms':>10}  {'energy J':>10}  {'cost':>8}")
    print("-" * 48)
    for n in range(6):
        result = compute_cost(
            n_selected=n,
            inference_ms=45.0 / max(n, 1),
            inference_energy_J=0.05 / max(n, 1),
            config=cfg,
        )
        print(f"{n:>3}  {result.bandwidth_MB:>8.3f}  "
              f"{result.latency_total_ms:>10.2f}  "
              f"{result.total_energy_J:>10.6f}  "
              f"{result.combined_cost:>8.5f}")
