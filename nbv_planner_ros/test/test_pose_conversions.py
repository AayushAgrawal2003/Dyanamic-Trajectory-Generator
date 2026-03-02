"""Unit tests for pose conversion utilities."""

import numpy as np
import pytest
from geometry_msgs.msg import PoseArray
from builtin_interfaces.msg import Time
from scipy.spatial.transform import Rotation

from nbv_planner_ros.pose_conversions import (
    planning_result_to_pose_array,
    t_cam_world_to_pose,
)


def test_identity_matrix():
    """Identity T_cam_world means camera is at origin looking along Z."""
    T = np.eye(4)
    pose = t_cam_world_to_pose(T)

    assert abs(pose.position.x) < 1e-10
    assert abs(pose.position.y) < 1e-10
    assert abs(pose.position.z) < 1e-10
    assert abs(pose.orientation.x) < 1e-10
    assert abs(pose.orientation.y) < 1e-10
    assert abs(pose.orientation.z) < 1e-10
    assert abs(pose.orientation.w - 1.0) < 1e-10


def test_pure_translation():
    """Camera at [1, 2, 3] with no rotation."""
    T_world_cam = np.eye(4)
    T_world_cam[:3, 3] = [1.0, 2.0, 3.0]
    T_cam_world = np.linalg.inv(T_world_cam)

    pose = t_cam_world_to_pose(T_cam_world)

    assert abs(pose.position.x - 1.0) < 1e-10
    assert abs(pose.position.y - 2.0) < 1e-10
    assert abs(pose.position.z - 3.0) < 1e-10


def test_90deg_rotation_z():
    """90-degree rotation around Z axis."""
    R = Rotation.from_euler("z", 90, degrees=True)
    T_world_cam = np.eye(4)
    T_world_cam[:3, :3] = R.as_matrix()
    T_world_cam[:3, 3] = [0.5, 0.0, 0.3]
    T_cam_world = np.linalg.inv(T_world_cam)

    pose = t_cam_world_to_pose(T_cam_world)

    assert abs(pose.position.x - 0.5) < 1e-6
    assert abs(pose.position.z - 0.3) < 1e-6

    # Recover the quaternion and verify it matches
    q = [
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    ]
    recovered = Rotation.from_quat(q)
    angle = (R.inv() * recovered).magnitude()
    assert angle < 1e-6, f"Rotation mismatch: {np.degrees(angle)} deg"


def test_roundtrip_random():
    """Random T_cam_world -> Pose -> reconstruct T should match."""
    rng = np.random.default_rng(123)
    for _ in range(20):
        R = Rotation.random(random_state=rng)
        t = rng.uniform(-2, 2, size=3)
        T_world_cam = np.eye(4)
        T_world_cam[:3, :3] = R.as_matrix()
        T_world_cam[:3, 3] = t
        T_cam_world = np.linalg.inv(T_world_cam)

        pose = t_cam_world_to_pose(T_cam_world)

        pos_recovered = np.array([
            pose.position.x, pose.position.y, pose.position.z
        ])
        np.testing.assert_allclose(pos_recovered, t, atol=1e-8)

        q = [
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ]
        R_recovered = Rotation.from_quat(q)
        angle_err = (R.inv() * R_recovered).magnitude()
        assert angle_err < 1e-6


def test_planning_result_to_pose_array():
    """PoseArray has correct length, frame_id, and stamp."""
    poses = [np.eye(4) for _ in range(5)]
    stamp = Time(sec=10, nanosec=500)
    pa = planning_result_to_pose_array(poses, "base_link", stamp)

    assert isinstance(pa, PoseArray)
    assert len(pa.poses) == 5
    assert pa.header.frame_id == "base_link"
    assert pa.header.stamp.sec == 10
