import os
import json
import argparse
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--state_root",
        required=True,
        help="Root folder of precomputed state data (scene_XXX/frame_YYY.json).",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="If set, save the plot to this path instead of showing.",
    )
    args = parser.parse_args()

    scene_counts = {}

    # iterate over scene folders
    for name in sorted(os.listdir(args.state_root)):
        if not name.startswith("scene_"):
            continue

        scene_dir = os.path.join(args.state_root, name)
        if not os.path.isdir(scene_dir):
            continue

        # scene index as int
        try:
            scene_id = int(name.split("_")[1])
        except (IndexError, ValueError):
            continue

        frame_name = f"frame_000"
        frame_dir = os.path.join(scene_dir, frame_name)

        if not os.path.exists(frame_dir):
            # no first frame file → count as 0
            scene_counts[scene_id] = 0
            continue

        files = [
            f for f in os.listdir(frame_dir)
            if os.path.isfile(os.path.join(frame_dir, f))
        ]
        num_agents = len(files)
        scene_counts[scene_id] = num_agents

    # sort by scene id
    scene_ids = sorted(scene_counts.keys())
    counts = [scene_counts[sid] for sid in scene_ids]

    print(counts)

    # plot
    plt.figure(figsize=(10, 4))
    plt.bar(scene_ids, counts)
    plt.xlabel("Scene ID")
    plt.ylabel(f"#agents")
    plt.title("Number of agents per scene")
    plt.tight_layout()

    if args.save_path:
        plt.savefig(args.save_path)
    else:
        plt.show()


if __name__ == "__main__":
    main()
