"""Tests for the camera model and depth simulator."""

import numpy as np
import pytest

from nbv_planner.sensor.camera_model import CameraModel, look_at


class TestCameraModel:
    def setup_method(self):
        self.camera = CameraModel.default()

    def test_intrinsic_matrix_shape(self):
        K = self.camera.K
        assert K.shape == (3, 3)
        assert K[2, 2] == 1.0

    def test_focal_length_positive(self):
        assert self.camera.fx > 0
        assert self.camera.fy > 0

    def test_principal_point(self):
        assert self.camera.cx == self.camera.width / 2.0
        assert self.camera.cy == self.camera.height / 2.0

    def test_project_origin_at_identity(self):
        """A point on the z-axis should project to the image center."""
        T = np.eye(4)
        point = np.array([[0.0, 0.0, 0.5]])  # 0.5m in front
        pixels, depths = self.camera.project_points(point, T)
        assert np.abs(pixels[0, 0] - self.camera.cx) < 1.0
        assert np.abs(pixels[0, 1] - self.camera.cy) < 1.0
        assert np.abs(depths[0] - 0.5) < 1e-6

    def test_is_in_fov(self):
        T = np.eye(4)
        # Point directly in front, within depth range
        visible = np.array([[0.0, 0.0, 0.5]])
        assert self.camera.is_in_fov(visible, T)[0]

        # Point behind the camera
        behind = np.array([[0.0, 0.0, -0.5]])
        assert not self.camera.is_in_fov(behind, T)[0]

        # Point too far
        far = np.array([[0.0, 0.0, 5.0]])
        assert not self.camera.is_in_fov(far, T)[0]

    def test_depth_noise(self):
        assert self.camera.depth_noise_std(1.0) == pytest.approx(0.001)
        assert self.camera.depth_noise_std(2.0) == pytest.approx(0.004)

    def test_fast_visible_points(self):
        T = np.eye(4)
        points = np.array([
            [0.0, 0.0, 0.5],   # in front, facing camera
            [0.0, 0.0, -0.5],  # behind camera
        ])
        normals = np.array([
            [0.0, 0.0, -1.0],  # facing toward camera at origin
            [0.0, 0.0, 1.0],   # facing away
        ])
        mask = self.camera.fast_visible_points(points, normals, T)
        assert mask[0]  # Should be visible
        assert not mask[1]  # Behind camera


class TestLookAt:
    def test_look_at_identity(self):
        """Camera at origin looking down +Z should give identity-like transform."""
        cam_pos = np.array([0.0, 0.0, 0.0])
        target = np.array([0.0, 0.0, 1.0])
        T = look_at(cam_pos, target)
        assert T.shape == (4, 4)
        # Camera z-axis should point toward target
        T_world = np.linalg.inv(T)
        cam_z = T_world[:3, 2]
        assert np.dot(cam_z, np.array([0, 0, 1])) > 0.99

    def test_look_at_from_above(self):
        """Camera above target looking down."""
        cam_pos = np.array([0.0, 0.0, 0.5])
        target = np.array([0.0, 0.0, 0.0])
        T = look_at(cam_pos, target)
        assert T.shape == (4, 4)
        # Camera should be invertible
        T_world = np.linalg.inv(T)
        assert np.allclose(T_world[:3, 3], cam_pos, atol=1e-6)

    def test_look_at_rotation_valid(self):
        """The rotation part should be orthonormal with det=1."""
        cam_pos = np.array([0.3, 0.2, 0.4])
        target = np.array([0.0, 0.0, 0.0])
        T = look_at(cam_pos, target)
        R = T[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)
        assert np.abs(np.linalg.det(R) - 1.0) < 1e-6
