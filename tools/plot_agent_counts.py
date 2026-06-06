import os
import json
import argparse
import statistics
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--state_root",
        required=True,
        help="Root folder of precomputed state data (scene_XXX/frame_YYY/agent_ZZ.json).",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="If set, save the plot to this path instead of showing it.",
    )
    parser.add_argument(
        "--stats_path",
        type=str,
        default=None,
        help="If set, save statistics JSON to this path. "
             "Defaults to <save_path stem>_stats.json when --save_path is given.",
    )
    args = parser.parse_args()

    # ── Collect per-scene, per-frame agent counts ────────────────────────────
    # scene_id -> {"frames": {frame_id: n_agents}, ...}
    scene_data = {}

    for scene_name in sorted(os.listdir(args.state_root)):
        if not scene_name.startswith("scene_"):
            continue
        scene_dir = os.path.join(args.state_root, scene_name)
        if not os.path.isdir(scene_dir):
            continue
        try:
            scene_id = int(scene_name.split("_")[1])
        except (IndexError, ValueError):
            continue

        frame_counts = {}
        for frame_name in sorted(os.listdir(scene_dir)):
            if not frame_name.startswith("frame_"):
                continue
            frame_dir = os.path.join(scene_dir, frame_name)
            if not os.path.isdir(frame_dir):
                continue
            try:
                frame_id = int(frame_name.split("_")[1])
            except (IndexError, ValueError):
                continue

            agent_ids = []
            for f in os.listdir(frame_dir):
                if f.endswith(".json") and os.path.isfile(os.path.join(frame_dir, f)):
                    try:
                        aid = int(f.replace("agent_", "").replace(".json", ""))
                        agent_ids.append(aid)
                    except ValueError:
                        pass

            frame_counts[frame_id] = {
                "total": len(agent_ids),                          # RSU + vehicles
                "vehicles": sum(1 for a in agent_ids if a > 0),  # agents 1-5 only
                "has_rsu": 0 in agent_ids,
            }

        if frame_counts:
            scene_data[scene_id] = frame_counts

    # ── Per-scene summary ────────────────────────────────────────────────────
    def _stats(vals):
        return {
            "min": min(vals),
            "max": max(vals),
            "mean": round(statistics.mean(vals), 3),
            "median": statistics.median(vals),
            "stdev": round(statistics.stdev(vals), 3) if len(vals) > 1 else 0.0,
        }

    scene_summary = {}
    for scene_id, frame_counts in sorted(scene_data.items()):
        total_vals   = [v["total"]    for v in frame_counts.values()]
        vehicle_vals = [v["vehicles"] for v in frame_counts.values()]
        rsu_present  = any(v["has_rsu"] for v in frame_counts.values())
        scene_summary[scene_id] = {
            "n_frames": len(frame_counts),
            "rsu_present": rsu_present,
            "total_agents": _stats(total_vals),    # RSU + vehicles
            "vehicles": _stats(vehicle_vals),       # agents 1-5 only
            "per_frame": {
                str(fid): fc for fid, fc in sorted(frame_counts.items())
            },
        }

    # ── Overall statistics ───────────────────────────────────────────────────
    all_total    = [v["total"]    for fc in scene_data.values() for v in fc.values()]
    all_vehicles = [v["vehicles"] for fc in scene_data.values() for v in fc.values()]
    overall = {
        "n_scenes": len(scene_data),
        "n_frames_total": len(all_total),
        "total_agents": _stats(all_total),
        "vehicles": _stats(all_vehicles),
    }

    # ── Print summary ────────────────────────────────────────────────────────
    ov = overall["vehicles"]
    ot = overall["total_agents"]
    print(f"\nOverall across {overall['n_scenes']} scenes, "
          f"{overall['n_frames_total']} frames:")
    print(f"  vehicles (1-5): min={ov['min']}  max={ov['max']}  "
          f"mean={ov['mean']}  median={ov['median']}  stdev={ov['stdev']}")
    print(f"  total (incl RSU): min={ot['min']}  max={ot['max']}  "
          f"mean={ot['mean']}  median={ot['median']}  stdev={ot['stdev']}")
    print("\nPer-scene vehicle counts (agents 1-5):")
    for sid, s in sorted(scene_summary.items()):
        v = s["vehicles"]
        rsu = "RSU" if s["rsu_present"] else "no RSU"
        print(f"  scene {sid:03d}: frames={s['n_frames']}  {rsu}  "
              f"min={v['min']} max={v['max']} mean={v['mean']}")

    # ── Save JSON ────────────────────────────────────────────────────────────
    stats_path = args.stats_path
    if stats_path is None and args.save_path:
        base = os.path.splitext(args.save_path)[0]
        stats_path = base + "_stats.json"

    if stats_path:
        payload = {
            "overall": overall,
            "per_scene": scene_summary,
        }
        with open(stats_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nStatistics saved to {stats_path}")

    # ── Plot: vehicle agents per scene (1-5, RSU excluded) ──────────────────
    scene_ids = sorted(scene_summary.keys())
    mean_vehicles = [scene_summary[sid]["vehicles"]["mean"] for sid in scene_ids]
    ov = overall["vehicles"]

    plt.figure(figsize=(14, 4))
    plt.bar(scene_ids, mean_vehicles)
    plt.xlabel("Scene ID")
    plt.ylabel("Mean #vehicles per frame (agents 1–5)")
    plt.title(
        f"Vehicle count per scene (RSU excluded)  —  "
        f"overall mean={ov['mean']}, min={ov['min']}, max={ov['max']}"
    )
    plt.xticks(scene_ids[::5], rotation=45, fontsize=7)
    plt.tight_layout()

    if args.save_path:
        plt.savefig(args.save_path, dpi=150)
        print(f"Plot saved to {args.save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
