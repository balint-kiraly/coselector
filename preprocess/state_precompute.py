import argparse
import json
import os

from data_utils.state_index import StateIndex

def agent_meta_to_dict(am):
    return dict(
        scene_id=am.scene_id,
        frame_id=am.frame_id,
        agent_id=am.agent_id,
        timestamp=am.timestamp,
        x=am.x,
        y=am.y,
        z=am.z,
        yaw=am.yaw,
        vx=am.vx,
        vy=am.vy,
        speed=am.speed,
        yaw_rate=am.yaw_rate,
        ax=am.ax,
        ay=am.ay,
        az=am.az,
        gx=am.gx,
        gy=am.gy,
        gz=am.gz,
        lidar_channels=am.lidar_channels,
        lidar_max_range_m=am.lidar_max_range_m,
        lidar_points_per_second=am.lidar_points_per_second,
        lidar_rotation_hz=am.lidar_rotation_hz,
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--root",
        type=str,
        help="Root path to CoPerception dataset",
    )
    parser.add_argument("-b", "--scene_begin", default=0, type=int, help="scene_begin")
    parser.add_argument("-e", "--scene_end", default=100, type=int, help="scene_end")
    parser.add_argument(
        "-p",
        "--save_path",
        type=str,
        help="Directory for saving the generated data",
    )

    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    # Does not include RSU data
    state_index = StateIndex(
        dataroot=args.root,
        scene_start=args.scene_begin,
        scene_end=args.scene_end,
        agent_start=0,
        agent_end=6
    )

    state_index.build_index()

    for (scene_id, frame_id), metas in state_index.frame_index.items():
        scene_dir = os.path.join(args.save_path, f"scene_{scene_id:03d}")
        os.makedirs(scene_dir, exist_ok=True)

        frame_dir = os.path.join(scene_dir, f"frame_{frame_id:03d}")
        os.makedirs(frame_dir, exist_ok=True)

        for m in metas:
            agent_path = os.path.join(frame_dir, f"agent_{m.agent_id:02d}.json")
            data = agent_meta_to_dict(m)

            with open(agent_path, "w") as f:
                json.dump(data, f, indent=2)

    print(f"Saved state index to {args.save_path}")

if __name__ == "__main__":
    main()
