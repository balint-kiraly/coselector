from functools import reduce

import numpy as np
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
import os.path as osp

from typing import Dict

NUM_CROSS_ROAD_SENSOR = 15
NUM_TYPES_OF_SENSORS = 23


def fuse_bev_lidar(
        nusc: "NuScenes",
        ref_sd_rec: Dict,
        return_trans_matrix: bool = False,
        min_distance: float = 1.0,
        no_cross_road=False,
):
    """
    Upperbound dataloader: transform the sweeps into the local coordinate of agent 0,
    :param ref_sd_rec: The current sample data record (lidar_top_id_0)
    :param return_trans_matrix: Whether need to return the transformation matrix
    :param min_distance: Distance below which points are discarded.
    :return: (all_pc, all_times). The aggregated point cloud and timestamps.
    """

    # Init
    points = np.zeros((LidarPointCloud.nbr_dims(), 0))
    all_pc = LidarPointCloud(points)
    all_times = np.zeros((1, 0))

    # Get reference pose and timestamp
    ref_pose_rec = nusc.get("ego_pose", ref_sd_rec["ego_pose_token"])
    ref_cs_rec = nusc.get("calibrated_sensor", ref_sd_rec["calibrated_sensor_token"])
    ref_time = 1e-6 * ref_sd_rec["timestamp"]

    # Homogeneous transform from ego car frame to reference frame
    ref_from_car = transform_matrix(
        ref_cs_rec["translation"], Quaternion(ref_cs_rec["rotation"]), inverse=True
    )

    # Homogeneous transformation matrix from global to _current_ ego car frame
    car_from_global = transform_matrix(
        ref_pose_rec["translation"], Quaternion(ref_pose_rec["rotation"]), inverse=True
    )

    # Aggregate current and previous sweeps.
    current_sd_rec = ref_sd_rec
    trans_matrix_list = list()
    skip_frame = 0
    sample_record = nusc.get("sample", ref_sd_rec["sample_token"])

    num_sensor = (
                         (len(sample_record["data"]) - NUM_CROSS_ROAD_SENSOR) // NUM_TYPES_OF_SENSORS
                 ) + 1

    for k in range(num_sensor):
        if no_cross_road and k == 0:
            continue

        # Load up the pointcloud.
        pointsensor_token = sample_record["data"]["LIDAR_TOP_id_" + str(k)]
        current_sd_rec = nusc.get("sample_data", pointsensor_token)
        current_pc = LidarPointCloud.from_file(
            osp.join(nusc.dataroot, current_sd_rec["filename"])
        )

        # Get past pose.
        current_pose_rec = nusc.get("ego_pose", current_sd_rec["ego_pose_token"])
        global_from_car = transform_matrix(
            current_pose_rec["translation"],
            Quaternion(current_pose_rec["rotation"]),
            inverse=False,
        )

        # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
        current_cs_rec = nusc.get(
            "calibrated_sensor", current_sd_rec["calibrated_sensor_token"]
        )
        car_from_current = transform_matrix(
            current_cs_rec["translation"],
            Quaternion(current_cs_rec["rotation"]),
            inverse=False,
        )

        # Fuse four transformation matrices into one and perform transform.
        trans_matrix = reduce(
            np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current]
        )

        current_pc.transform(trans_matrix)

        # Collect the transformation matrix
        trans_matrix_list.append(trans_matrix)

        # Remove close points and add timevector.
        current_pc.remove_close(min_distance)

        time_lag = ref_time - 1e-6 * (
                current_sd_rec["timestamp"] + k
        )  # positive difference

        if k % (skip_frame + 1) == 0:
            times = time_lag * np.ones((1, current_pc.nbr_points()))
        else:
            times = time_lag * np.ones((1, 1))  # dummy value

        all_times = np.hstack((all_times, times))
        all_pc.points = np.hstack((all_pc.points, current_pc.points))

    trans_matrix_list = np.stack(trans_matrix_list, axis=0)

    if return_trans_matrix:
        return all_pc, np.squeeze(all_times, 0), trans_matrix_list
    else:
        return all_pc, np.squeeze(all_times, 0)