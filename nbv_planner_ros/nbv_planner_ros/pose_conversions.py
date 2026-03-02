"""Utilities for converting 4x4 transform matrices to ROS geometry messages."""

import numpy as np
from geometry_msgs.msg import Pose, PoseArray, Point, Quaternion
from scipy.spatial.transform import Rotation


def t_cam_world_to_pose(T_cam_world: np.ndarray) -> Pose:
    """Convert a 4x4 T_cam_world extrinsic to a geometry_msgs/Pose.

    The planner produces T_cam_world (camera-from-world). We invert to get
    T_world_cam (camera pose expressed in the world/base_link frame), then
    extract position and quaternion.

    Camera convention: Z-forward, X-right, Y-down (matches ROS optical frame).
    """
    T_world_cam = np.linalg.inv(T_cam_world)

    position = T_world_cam[:3, 3]
    quat = Rotation.from_matrix(T_world_cam[:3, :3]).as_quat()  # [x, y, z, w]

    pose = Pose()
    pose.position = Point(
        x=float(position[0]),
        y=float(position[1]),
        z=float(position[2]),
    )
    pose.orientation = Quaternion(
        x=float(quat[0]),
        y=float(quat[1]),
        z=float(quat[2]),
        w=float(quat[3]),
    )
    return pose


def planning_result_to_pose_array(
    poses: list[np.ndarray],
    frame_id: str,
    stamp,
) -> PoseArray:
    """Convert a list of T_cam_world matrices to a stamped PoseArray."""
    msg = PoseArray()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    msg.poses = [t_cam_world_to_pose(T) for T in poses]
    return msg
