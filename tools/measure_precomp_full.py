"""Measure real-inference per-frame preprocessing cost on the cloud server.

At inference time the server receives raw LiDAR from each selected vehicle and
must perform two steps before FaFNet can run:

  1. from_file_multisweep_warp2com_sample_data
       Load the point cloud sweeps, apply ego-motion compensation, and produce
       the inter-agent transformation matrices needed for BEV fusion.

  2. voxelize_occupy
       Discretise the warped point cloud into a BEV occupancy grid.

Everything else that bev_precompute.py does (upperbound teacher clouds,
per-annotation box retrieval, visibility mapping, GT generation,
sanity-check assertions) is training-time only and does not run on the server
during inference.

Results are saved to JSON; precomp_per_agent_ms / precomp_energy_J can be
plugged directly into CostConfig.

Usage:
    conda run -n coselector python tools/measure_precomp_full.py \\
        --root /mnt/10TB/balintkiraly/data/data/V2X-Sim-2 \\
        --version v2.0 \\
        --n_frames 100 \\
        --agent_id 1 \\
        --output measurements/precomp_full_cost.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from typing import List

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from coperception.configs import Config
from coperception.utils.data_util import voxelize_occupy
from coperception.utils.nuscenes_pc_util import from_file_multisweep_warp2com_sample_data
from coperception.utils.v2x_sim_scene_split.parser import parse_scene_files


# ── codecarbon helper ────────────────────────────────────────────────────────

def _tracker_energy_J(tracker) -> float:
    if hasattr(tracker, "final_emissions_data") and tracker.final_emissions_data is not None:
        return tracker.final_emissions_data.energy_consumed * 3_600_000
    if hasattr(tracker, "_total_energy"):
        return tracker._total_energy.kwh * 3_600_000
    return 0.0


try:
    from codecarbon import EmissionsTracker as _ET
    _HAS_CODECARBON = True
except ImportError:
    _HAS_CODECARBON = False


def _make_tracker():
    if not _HAS_CODECARBON:
        return None
    return _ET(save_to_file=False, measure_power_secs=1)


# ── Inference-time pipeline for one frame ────────────────────────────────────

def _process_one_frame(nusc, curr_sample_data: dict, config: Config, agent_id: int) -> float:
    """Run the inference-time preprocessing pipeline for one sample_data entry.

    Steps:
      1. from_file_multisweep_warp2com_sample_data  (load + warp + transforms)
      2. voxelize_occupy per sweep                  (BEV discretisation)

    Returns wall time in ms.
    """
    t0 = time.perf_counter()

    # Step 1 – load warped point cloud and transformation matrices
    (
        all_pc,
        all_times,
        _trans_matrices,
        _trans_matrices_no_cross_road,
        _target_agent_id,
        _num_sensor,
    ) = from_file_multisweep_warp2com_sample_data(
        agent_id, nusc, curr_sample_data, return_trans_matrix=True
    )

    # Organise into per-sweep slices (mirrors bev_precompute create_data)
    pc = all_pc.points
    _, sort_idx = np.unique(all_times, return_index=True)
    unique_times = all_times[np.sort(sort_idx)]

    # Step 2 – voxelize each sweep
    for _time in unique_times:
        points_idx = np.where(all_times == _time)[0]
        sweep_pc = pc[:, points_idx].T  # (N, 4+)
        voxelize_occupy(
            sweep_pc[:, :4],
            voxel_size=config.voxel_size,
            extents=config.area_extents,
            return_indices=True,
        )

    return (time.perf_counter() - t0) * 1000.0


# ── Frame collection ─────────────────────────────────────────────────────────

def _collect_sample_data(nusc, scene_splits: dict, agent_id: int, n_frames: int) -> list:
    """Walk train scenes and collect up to n_frames sample_data records."""
    channel = f"LIDAR_TOP_id_{agent_id}"
    frames = []
    for scene_idx, scene in enumerate(nusc.scene):
        if len(frames) >= n_frames:
            break
        if scene_idx not in scene_splits.get("train", set()):
            continue
        sample_token = scene["first_sample_token"]
        while sample_token and len(frames) < n_frames:
            sample = nusc.get("sample", sample_token)
            if channel in sample["data"]:
                frames.append(nusc.get("sample_data", sample["data"][channel]))
            sample_token = sample["next"]
    return frames


# ── Main ─────────────────────────────────────────────────────────────────────

def measure(root: str, version: str, n_frames: int, agent_id: int, output: str) -> None:
    from nuscenes import NuScenes

    if not _HAS_CODECARBON:
        print("WARNING: codecarbon not found — energy will be recorded as 0")

    print(f"Loading NuScenes {version} from {root} ...")
    nusc = NuScenes(version=version, dataroot=root, verbose=False)

    scene_splits = parse_scene_files("./configs/scene_split")
    config = Config("train", binary=True, is_cross_road=(agent_id == 0))

    print(f"Collecting up to {n_frames} sample_data entries for agent {agent_id} ...")
    frames = _collect_sample_data(nusc, scene_splits, agent_id, n_frames)
    if not frames:
        print("ERROR: no frames found.")
        return
    print(f"Collected {len(frames)} frames.\n")

    # Warm-up (not counted)
    print("Warming up (3 frames) ...")
    for sd in frames[:3]:
        _process_one_frame(nusc, sd, config, agent_id)

    # Measurement
    print(f"Measuring inference preprocessing on {len(frames)} frames ...")
    tracker = _make_tracker()
    if tracker:
        tracker.start()

    times_ms: List[float] = []
    for i, sd in enumerate(frames):
        wall_ms = _process_one_frame(nusc, sd, config, agent_id)
        times_ms.append(wall_ms)
        if (i + 1) % 10 == 0 or (i + 1) == len(frames):
            print(f"  [{i+1}/{len(frames)}]  last={wall_ms:.1f} ms")

    total_energy_J = 0.0
    if tracker:
        tracker.stop()
        total_energy_J = _tracker_energy_J(tracker)

    n = len(times_ms)
    result = {
        "precomp_per_agent_ms": statistics.mean(times_ms),
        "precomp_energy_J": total_energy_J / n,
        "mean_ms": statistics.mean(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "stdev_ms": statistics.stdev(times_ms) if n > 1 else 0.0,
        "total_energy_J": total_energy_J,
        "per_frame_energy_J": total_energy_J / n,
        "n_frames": n,
        "agent_id": agent_id,
        "dataset_root": root,
        "dataset_version": version,
        "codecarbon_available": _HAS_CODECARBON,
        "note": (
            "Inference-only pipeline: from_file_multisweep_warp2com_sample_data "
            "+ voxelize_occupy. Upperbound teacher clouds, annotation box "
            "retrieval, visibility mapping, and GT generation are excluded "
            "(training-time only)."
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    print()
    print("=" * 70)
    print(f"  Agent {agent_id}   Frames: {n}")
    print("-" * 70)
    print(f"  mean  = {result['mean_ms']:8.2f} ms")
    print(f"  min   = {result['min_ms']:8.2f} ms")
    print(f"  max   = {result['max_ms']:8.2f} ms")
    print(f"  stdev = {result['stdev_ms']:8.2f} ms")
    print(f"  energy/frame = {result['per_frame_energy_J']:.6f} J")
    print("=" * 70)
    print(f"\n  >>> precomp_per_agent_ms = {result['mean_ms']:.2f} ms <<<")
    print(f"  >>> precomp_energy_J     = {result['per_frame_energy_J']:.6f} J <<<")

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Measure real-inference preprocessing cost (warp2com + voxelize)"
    )
    parser.add_argument("--root", default="/mnt/10TB/balintkiraly/data/data/V2X-Sim-2")
    parser.add_argument("--version", default="v2.0")
    parser.add_argument("--n_frames", type=int, default=100)
    parser.add_argument("--agent_id", type=int, default=1)
    parser.add_argument("--output", default="measurements/precomp_full_cost.json")
    args = parser.parse_args()
    measure(args.root, args.version, args.n_frames, args.agent_id, args.output)
