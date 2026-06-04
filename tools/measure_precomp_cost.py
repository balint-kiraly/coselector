"""Measure per-agent server-side precomputation cost.

Three timed steps that the cloud server performs per uploaded LiDAR frame:

  1. read_ms        — deserialise raw bytes into numpy array.
                      On the real server this corresponds to reading the
                      received TCP/UDP payload into memory.  Measured here
                      by np.fromfile() on the same .pcd.bin files.

  2. voxelize_ms    — pure CPU/numpy voxelization (voxelize_occupy).
                      Already measured separately, included again for the
                      full-pipeline comparison.

  3. full_precomp_ms (read + voxelize) — the realistic server scenario:
                      receive raw bytes, parse, voxelize, hand off to FaFNet.

All three batches run inside a single EmissionsTracker session each so the
energy counter accumulates enough signal before being divided by N.

Results are saved to JSON for CostConfig.from_measurements_dir() to load.

Usage:
    conda run -n coselector python tools/measure_precomp_cost.py \\
        --root /mnt/10TB/balintkiraly/data/data/V2X-Sim-2 \\
        --version v2.0 \\
        --n_frames 100 \\
        --agent_id 1 \\
        --output measurements/precomp_cost.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from typing import List, Tuple

import numpy as np

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from coperception.configs import Config
from coperception.utils.data_util import voxelize_occupy
from coperception.utils.nuscenes_pc_util import from_file_multisweep_warp2com_sample_data


# ── codecarbon energy helper ─────────────────────────────────────────────────
def _tracker_energy_J(tracker) -> float:
    """Extract total energy in joules (handles codecarbon 1.x and 2.x)."""
    if hasattr(tracker, "final_emissions_data") and tracker.final_emissions_data is not None:
        return tracker.final_emissions_data.energy_consumed * 3_600_000
    if hasattr(tracker, "_total_energy"):
        return tracker._total_energy.kwh * 3_600_000
    return 0.0


try:
    from codecarbon import EmissionsTracker as _EmissionsTracker
    _HAS_CODECARBON = True
except ImportError:
    _HAS_CODECARBON = False


def _make_tracker():
    if not _HAS_CODECARBON:
        return None
    return _EmissionsTracker(save_to_file=False, measure_power_secs=1)


# ── Individual step functions ────────────────────────────────────────────────

def _read_one_frame(filepath: str) -> Tuple[float, np.ndarray]:
    """Deserialise a .pcd.bin file into a numpy array.

    Simulates what the cloud server does after receiving the raw LiDAR payload:
    parse bytes → float32 array → reshape to (N, 5).

    Returns:
        (wall_ms, pts)  where pts has shape (N, 5) [x, y, z, intensity, ring]
    """
    t0 = time.perf_counter()
    raw = np.fromfile(filepath, dtype=np.float32).reshape(-1, 5)
    wall_ms = (time.perf_counter() - t0) * 1000.0
    return wall_ms, raw


def _voxelize_one_frame(pts: np.ndarray, config: Config) -> float:
    """Run voxelize_occupy on (N, 5) raw points; return wall time in ms."""
    t0 = time.perf_counter()
    voxelize_occupy(
        pts[:, :4],   # (N, 4): x, y, z, intensity
        voxel_size=config.voxel_size,
        extents=config.area_extents,
        return_indices=True,
    )
    return (time.perf_counter() - t0) * 1000.0


# ── Batch measurement helper ─────────────────────────────────────────────────

def _run_batch(fn, items, label: str, n_total: int) -> Tuple[List[float], float]:
    """Run fn(item) for each item inside one tracker session.

    Returns:
        (times_ms, total_energy_J)
    """
    tracker = _make_tracker()
    if tracker:
        tracker.start()

    times_ms: List[float] = []
    for i, item in enumerate(items):
        t = fn(item)
        times_ms.append(t)
        if (i + 1) % 10 == 0:
            print(f"  [{label}] {i + 1}/{n_total}  last: {t:.3f} ms")

    total_energy_J = 0.0
    if tracker:
        tracker.stop()
        total_energy_J = _tracker_energy_J(tracker)

    return times_ms, total_energy_J


# ── Stats helper ─────────────────────────────────────────────────────────────

def _stats(times_ms: List[float], total_energy_J: float) -> dict:
    n = len(times_ms)
    return {
        "mean_ms": statistics.mean(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "stdev_ms": statistics.stdev(times_ms) if n > 1 else 0.0,
        "total_energy_J": total_energy_J,
        "per_frame_energy_J": total_energy_J / n,
        "n_frames": n,
    }


# ── Main measurement ─────────────────────────────────────────────────────────

def measure(
    root: str,
    version: str,
    n_frames: int,
    agent_id: int,
    output: str,
) -> None:
    from nuscenes import NuScenes

    if not _HAS_CODECARBON:
        print("WARNING: codecarbon not found — energy will be recorded as 0")

    print(f"Loading NuScenes {version} from {root} ...")
    nusc = NuScenes(version=version, dataroot=root, verbose=False)

    config = Config("train", binary=True, is_cross_road=(agent_id == 0))
    channel = f"LIDAR_TOP_id_{agent_id}"

    # ── Collect file paths ────────────────────────────────────────────────
    print(f"Collecting {n_frames} .pcd.bin paths for {channel} ...")
    filepaths: List[str] = []
    for scene in nusc.scene:
        if len(filepaths) >= n_frames:
            break
        sample_token = scene["first_sample_token"]
        while sample_token and len(filepaths) < n_frames:
            sample = nusc.get("sample", sample_token)
            if channel in sample["data"]:
                sd = nusc.get("sample_data", sample["data"][channel])
                fp = os.path.join(root, sd["filename"])
                if os.path.isfile(fp):
                    filepaths.append(fp)
            sample_token = sample["next"]

    if not filepaths:
        print("ERROR: no files found.")
        return
    n = len(filepaths)
    print(f"Collected {n} file paths.\n")

    # ── Warm-up ────────────────────────────────────────────────────────────
    print("Warming up (3 frames) ...")
    for fp in filepaths[:3]:
        _, pts = _read_one_frame(fp)
        _voxelize_one_frame(pts, config)

    # ── Step 1: Read only ─────────────────────────────────────────────────
    print(f"\n[1/3] Measuring read time ({n} frames) ...")
    # Pre-read into memory so the voxelize batch doesn't re-read from disk
    pts_cache: List[np.ndarray] = []

    def _read_and_cache(fp: str) -> float:
        wall_ms, pts = _read_one_frame(fp)
        pts_cache.append(pts)
        return wall_ms

    read_times_ms, read_energy_J = _run_batch(_read_and_cache, filepaths, "read", n)

    # ── Step 2: Voxelize only (from cached arrays) ─────────────────────────
    print(f"\n[2/3] Measuring voxelization time ({n} frames, from cache) ...")
    vox_times_ms, vox_energy_J = _run_batch(
        lambda pts: _voxelize_one_frame(pts, config), pts_cache, "vox", n
    )

    # ── Step 3: Full pipeline (read-from-disk + voxelize, no cache) ───────
    print(f"\n[3/3] Measuring full pipeline: read + voxelize ({n} frames) ...")

    def _full_pipeline(fp: str) -> float:
        wall_ms_read, pts = _read_one_frame(fp)
        wall_ms_vox = _voxelize_one_frame(pts, config)
        return wall_ms_read + wall_ms_vox

    full_times_ms, full_energy_J = _run_batch(_full_pipeline, filepaths, "full", n)

    # ── Compute stats ──────────────────────────────────────────────────────
    read_stats  = _stats(read_times_ms,  read_energy_J)
    vox_stats   = _stats(vox_times_ms,   vox_energy_J)
    full_stats  = _stats(full_times_ms,  full_energy_J)

    # ── Print report ───────────────────────────────────────────────────────
    def _row(label, s):
        print(f"  {label:<25}  mean={s['mean_ms']:6.3f} ms  "
              f"min={s['min_ms']:6.3f}  max={s['max_ms']:6.3f}  "
              f"stdev={s['stdev_ms']:5.3f}  "
              f"energy/frm={s['per_frame_energy_J']:.6f} J")

    print()
    print("=" * 75)
    print(f"  Agent: {channel}   Frames: {n}")
    print("-" * 75)
    _row("read (deserialise)",  read_stats)
    _row("voxelize",            vox_stats)
    _row("full (read+voxelize)", full_stats)
    print("=" * 75)
    print(f"\n  >>> precomp_per_agent_ms = {full_stats['mean_ms']:.3f}  "
          f"(full pipeline: read + voxelize)  <<<")
    print(f"  >>> precomp_energy_J     = {full_stats['per_frame_energy_J']:.6f}  <<<")

    # ── Save JSON ─────────────────────────────────────────────────────────
    # CostConfig loads precomp_per_agent_ms and precomp_energy_J — use full pipeline.
    result = {
        "precomp_per_agent_ms": full_stats["mean_ms"],
        "precomp_energy_J": full_stats["per_frame_energy_J"],
        "read": read_stats,
        "voxelize": vox_stats,
        "full_pipeline": full_stats,
        "codecarbon_available": _HAS_CODECARBON,
        "note": (
            "precomp_per_agent_ms / precomp_energy_J use the full pipeline "
            "(read + voxelize). Voxelize is pure CPU/numpy; read simulates "
            "server-side buffer deserialisation after network receive."
        ),
        "agent_id": agent_id,
        "dataset_root": root,
        "dataset_version": version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Measure per-agent server-side precomputation cost (read + voxelize)"
    )
    parser.add_argument("--root", default="/mnt/10TB/balintkiraly/data/data/V2X-Sim-2")
    parser.add_argument("--version", default="v2.0")
    parser.add_argument("--n_frames", type=int, default=100)
    parser.add_argument("--agent_id", type=int, default=1,
                        help="Agent sensor index (1-5 are vehicles; 0 is RSU)")
    parser.add_argument("--output", default="measurements/precomp_cost.json")
    args = parser.parse_args()
    measure(args.root, args.version, args.n_frames, args.agent_id, args.output)
