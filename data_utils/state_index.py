import json
import os
from dataclasses import dataclass
from typing import Dict, Tuple, List
import math

from nuscenes.nuscenes import NuScenes


@dataclass
class AgentMeta:
    scene_id: int
    frame_id: int
    agent_id: int
    timestamp: float

    # GNSS / orientation
    x: float
    y: float
    z: float
    yaw: float          # compass yaw (radians)

    # Derived motion
    vx: float
    vy: float
    speed: float
    yaw_rate: float

    # IMU
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float

    # LiDAR config (constant for V2X-Sim)
    lidar_channels: int = 32
    lidar_max_range_m: float = 70.0
    lidar_points_per_second: int = 250_000
    lidar_rotation_hz: float = 20.0


def _agent_meta_from_dict(d: dict) -> AgentMeta:
    """Helper to build AgentMeta from a dict"""
    return AgentMeta(
        scene_id=d["scene_id"],
        frame_id=d["frame_id"],
        agent_id=d["agent_id"],
        timestamp=d["timestamp"],
        x=d["x"],
        y=d["y"],
        z=d["z"],
        yaw=d["yaw"],
        vx=d["vx"],
        vy=d["vy"],
        speed=d["speed"],
        yaw_rate=d["yaw_rate"],
        ax=d["ax"],
        ay=d["ay"],
        az=d["az"],
        gx=d["gx"],
        gy=d["gy"],
        gz=d["gz"],
        lidar_channels=d.get("lidar_channels", 32),
        lidar_max_range_m=d.get("lidar_max_range_m", 70.0),
        lidar_points_per_second=d.get("lidar_points_per_second", 250_000),
        lidar_rotation_hz=d.get("lidar_rotation_hz", 20.0),
    )


class StateIndex:
    """
    Index over a V2X-Sim dataset (NuScenes format) that can answer:
      - For a given (scene_id, frame_id) -> list of AgentMeta for all agents present
    """

    def __init__(
            self,
            dataroot: str,
            scene_start: int,
            scene_end: int,
            agent_start: int,
            agent_end: int,
            version: str = "v2.0",
    ):
        """
        Only load scenes [scene_start, scene_end]
        and only agents [agent_start, agent_end]
        """
        self.dataroot = dataroot
        self.nusc_version = version
        self.scene_start = scene_start
        self.scene_end = scene_end
        self.agent_start = agent_start
        self.agent_end = agent_end

        # Main index: (scene_id, frame_id) -> [AgentMeta]
        self.frame_index: Dict[Tuple[int, int], List[AgentMeta]] = {}

        # Helper for motion: (scene_id, agent_id) -> last (frame_id, AgentMeta)
        self.prev_state: Dict[Tuple[int, int], AgentMeta] = {}

    @classmethod
    def from_fs(cls, root: str, scene_ids=None) -> "StateIndex":
        """
        Build a StateIndex from a folder tree:

            root/
              scene_000/
                frame_000/
                  agent_00.json
                  agent_01.json
                frame_001/
              scene_001/
                ...

        Args:
            root: path to the state-feature folder tree.
            scene_ids: optional iterable of scene ids to load. When given, scenes
                outside this set are skipped, which avoids reading tens of
                thousands of JSON files for scenes that will never be evaluated.

        Returns:
            StateIndex with frame_index[(scene_id, frame_id)] -> List[AgentMeta]
        """
        self = cls.__new__(cls)   # bypass __init__

        frame_index: Dict[Tuple[int, int], List[AgentMeta]] = {}

        scene_filter = set(scene_ids) if scene_ids is not None else None
        print("Loading index..." if scene_filter is None
              else f"Loading index for scenes {sorted(scene_filter)}...")

        for scene_name in sorted(os.listdir(root)):
            if not scene_name.startswith("scene_"):
                continue
            scene_dir = os.path.join(root, scene_name)
            if not os.path.isdir(scene_dir):
                continue

            try:
                scene_id = int(scene_name.split("_")[1])
            except (IndexError, ValueError):
                continue

            if scene_filter is not None and scene_id not in scene_filter:
                continue

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

                metas: List[AgentMeta] = []
                for fname in sorted(os.listdir(frame_dir)):
                    if not fname.endswith(".json"):
                        continue
                    fpath = os.path.join(frame_dir, fname)
                    with open(fpath, "r") as f:
                        d = json.load(f)
                    metas.append(_agent_meta_from_dict(d))

                frame_index[(scene_id, frame_id)] = metas

        self.frame_index = frame_index
        print("Index loaded.")
        return self

    # ----------------------------------------------------------
    # Building the index
    # ----------------------------------------------------------
    def build_index(self):
        nusc = NuScenes(version=self.nusc_version, dataroot=self.dataroot, verbose=True)

        print("Loading index...")

        for scene_id in range(self.scene_start, self.scene_end):
            scene = nusc.scene[scene_id]

            sample_token = scene["first_sample_token"]
            frame_id = 0

            while sample_token:
                sample = nusc.get("sample", sample_token)

                # For this frame, gather agents that appear (via GNSS_TOP_id_X)
                metas = []

                for channel, sd_token in sample["data"].items():

                    if not channel.startswith("GNSS_TOP_id_"):
                        continue

                    agent_id = int(channel.split("_id_")[1])
                    if not (self.agent_start <= agent_id < self.agent_end):
                        continue

                    # GNSS
                    sd_gnss = nusc.get("sample_data", sd_token)
                    sd_gnss_path = sd_gnss["filename"].replace("sweeps", "gnss")
                    gnss_arr = self._load_np(sd_gnss_path)

                    x, y, z = float(gnss_arr[0]), float(gnss_arr[1]), float(gnss_arr[2])

                    # IMU
                    imu_channel = f"IMU_TOP_id_{agent_id}"
                    imu_token = sample["data"][imu_channel]
                    sd_imu = nusc.get("sample_data", imu_token)
                    sd_imu_path = sd_imu["filename"].replace("sweeps", "imu")
                    imu_arr = self._load_np(sd_imu_path).astype(float)

                    ax, ay, az = imu_arr[0], imu_arr[1], imu_arr[2]
                    gx, gy, gz = imu_arr[3], imu_arr[4], imu_arr[5]
                    yaw_compass = imu_arr[6]    # rad, north=0

                    # Use compass yaw as main yaw
                    yaw = float(yaw_compass)

                    # Timestamp
                    timestamp = sd_gnss["timestamp"] / 1e6

                    # Motion estimate
                    vx, vy, yaw_rate = self._estimate_motion(
                        scene_id, agent_id, frame_id,
                        x, y, yaw, timestamp
                    )

                    speed = float(math.hypot(vx, vy))

                    meta = AgentMeta(
                        scene_id=scene_id,
                        frame_id=frame_id,
                        agent_id=agent_id,
                        timestamp=timestamp,
                        x=x, y=y, z=z,
                        yaw=yaw,
                        vx=vx, vy=vy,
                        speed=speed,
                        yaw_rate=yaw_rate,
                        ax=ax, ay=ay, az=az,
                        gx=gx, gy=gy, gz=gz,
                    )

                    metas.append(meta)


                self.frame_index[(scene_id, frame_id)] = metas

                # Advance
                sample_token = sample["next"]
                frame_id += 1

        print("Index loaded.")


    # ----------------------------------------------------------
    # Motion estimation (finite difference)
    # ----------------------------------------------------------
    def _estimate_motion(self, scene_id, agent_id, frame_id, x, y, yaw, t):
        key = (scene_id, agent_id)

        if key not in self.prev_state:
            self.prev_state[key] = AgentMeta(
                scene_id, frame_id, agent_id, t,
                x, y, 0, yaw, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0  # imu fields ignored
            )
            return 0.0, 0.0, 0.0

        prev = self.prev_state[key]
        dt = t - prev.timestamp
        if dt <= 0:
            return 0.0, 0.0, 0.0

        vx = (x - prev.x) / dt
        vy = (y - prev.y) / dt

        # yaw unwrap
        dyaw = (yaw - prev.yaw + math.pi) % (2 * math.pi) - math.pi
        yaw_rate = dyaw / dt

        # update for next time
        self.prev_state[key] = AgentMeta(
            scene_id, frame_id, agent_id, t,
            x, y, 0, yaw, vx, vy, 0, yaw_rate,
            0, 0, 0, 0, 0, 0
        )

        return float(vx), float(vy), float(yaw_rate)


    # ----------------------------------------------------------
    # Helper: load numpy array from sample_data
    # ----------------------------------------------------------
    def _load_np(self, sd_path):
        """
        Loads the .npy-like sensor data from file.
        sample_data['filename'] points to a .npy path relative to dataroot.
        """
        import os
        import numpy as np
        path = os.path.join(self.dataroot, sd_path)
        return np.load(path)


    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------
    def get_agents_meta(self, scene_id: int, frame_id: int) -> List[AgentMeta]:
        return self.frame_index.get((scene_id, frame_id), [])

if __name__ == "__main__":
    dataroot = "../../data/V2X-Sim-2"
    scene_start = 0
    scene_end = 1       # load only scene 0 for quick test
    agent_start = 0
    agent_end = 6       # load all agents

    print("Loading index...")
    index = StateIndex(
        dataroot=dataroot,
        scene_start=scene_start,
        scene_end=scene_end,
        agent_start=agent_start,
        agent_end=agent_end,
    )
    print("Index loaded.")

    # Try: scene 0, frame 0
    scene_id = 0
    frame_id = 6

    agents = index.get_agents_meta(scene_id, frame_id)
    print(f"Scene {scene_id}, Frame {frame_id}: {len(agents)} agents")

    for a in agents:
        print(
            f"Agent {a.agent_id} | "
            f"x={a.x:.2f}, y={a.y:.2f}, yaw={a.yaw:.2f}, "
            f"vx={a.vx:.2f}, vy={a.vy:.2f}, yaw_rate={a.yaw_rate:.3f}, "
            f"ax={a.ax:.2f}, ay={a.ay:.2f}, gz={a.gz:.2f}"
        )