"""Measure raw LiDAR file sizes from V2X-Sim.

Iterates over a sample of scenes and all agent sensors, reads each .pcd.bin
file, and reports point count, bytes-per-point, and per-file size statistics.
Results are saved to a JSON file that CostConfig.from_measurements_dir() can
load directly — no manual copy-paste needed.

Usage:
    conda run -n coselector python tools/measure_lidar_size.py \
        --root /mnt/10TB/balintkiraly/data/data/V2X-Sim-2 \
        --version v2.0 \
        --n_scenes 10 \
        --output measurements/lidar_size.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from datetime import datetime, timezone


# Number of float32 fields per LiDAR point in V2X-Sim: x, y, z, intensity, ring
FIELDS_PER_POINT = 5
BYTES_PER_FLOAT32 = 4
BYTES_PER_POINT = FIELDS_PER_POINT * BYTES_PER_FLOAT32   # 20


def measure(root: str, version: str, n_scenes: int, max_agents: int, output: str) -> None:
    from nuscenes import NuScenes

    print(f"Loading NuScenes {version} from {root} ...")
    nusc = NuScenes(version=version, dataroot=root, verbose=False)

    scenes = nusc.scene[:n_scenes]
    print(f"Sampling {len(scenes)} scenes, up to {max_agents} agent sensors each.\n")

    sizes_bytes: list[int] = []
    point_counts: list[int] = []
    agent_sensors = [f"LIDAR_TOP_id_{i}" for i in range(max_agents)]

    for scene in scenes:
        sample_token = scene["first_sample_token"]
        while sample_token:
            sample = nusc.get("sample", sample_token)
            for sensor in agent_sensors:
                if sensor not in sample["data"]:
                    continue
                sd = nusc.get("sample_data", sample["data"][sensor])
                filepath = os.path.join(root, sd["filename"])
                if not os.path.isfile(filepath):
                    continue
                file_bytes = os.path.getsize(filepath)
                n_points = file_bytes // BYTES_PER_POINT
                sizes_bytes.append(file_bytes)
                point_counts.append(n_points)
            sample_token = sample["next"]

    if not sizes_bytes:
        print("ERROR: no LiDAR files found. Check --root and --version.")
        return

    mb = [b / 1_000_000 for b in sizes_bytes]

    result = {
        "lidar_size_MB": statistics.mean(mb),
        "n_samples": len(mb),
        "min_MB": min(mb),
        "max_MB": max(mb),
        "stdev_MB": statistics.stdev(mb) if len(mb) > 1 else 0.0,
        "mean_points": statistics.mean(point_counts),
        "min_points": min(point_counts),
        "max_points": max(point_counts),
        "bytes_per_point": BYTES_PER_POINT,
        "fields_per_point": FIELDS_PER_POINT,
        "n_scenes_sampled": n_scenes,
        "dataset_root": root,
        "dataset_version": version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    print("=" * 55)
    print(f"  Samples measured  :  {result['n_samples']}")
    print(f"  Points per frame  :  mean={result['mean_points']:,.0f}  "
          f"min={result['min_points']:,}  max={result['max_points']:,}")
    print(f"  Bytes per point   :  {BYTES_PER_POINT}  "
          f"({FIELDS_PER_POINT} float32 fields × {BYTES_PER_FLOAT32} bytes)")
    print(f"  File size (MB)    :  mean={result['lidar_size_MB']:.4f}  "
          f"min={result['min_MB']:.4f}  max={result['max_MB']:.4f}  "
          f"stdev={result['stdev_MB']:.4f}")
    print("=" * 55)
    print(f"\n  >>> lidar_size_MB = {result['lidar_size_MB']:.4f}  <<<")

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure V2X-Sim raw LiDAR file sizes")
    parser.add_argument(
        "--root",
        default="/mnt/10TB/balintkiraly/data/data/V2X-Sim-2",
        help="Path to V2X-Sim-2 dataset root",
    )
    parser.add_argument(
        "--version",
        default="v2.0",
        help="NuScenes version string (v2.0 or v2.0-mini)",
    )
    parser.add_argument(
        "--n_scenes",
        type=int,
        default=10,
        help="Number of scenes to sample",
    )
    parser.add_argument(
        "--max_agents",
        type=int,
        default=6,
        help="Number of agent sensors to check (0-5)",
    )
    parser.add_argument(
        "--output",
        default="measurements/lidar_size.json",
        help="Path to save the JSON results (default: measurements/lidar_size.json)",
    )
    args = parser.parse_args()
    measure(args.root, args.version, args.n_scenes, args.max_agents, args.output)
