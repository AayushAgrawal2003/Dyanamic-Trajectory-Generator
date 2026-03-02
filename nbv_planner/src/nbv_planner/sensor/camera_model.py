"""Intel RealSense D435 camera model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import yaml


@dataclass
class CameraModel:
    """Pinhole camera model with depth range and noise characteristics."""

    width: int
    height: int
    horizontal_fov_deg: float
    vertical_fov_deg: float
    min_depth: float
    max_depth: float
    depth_noise_coeff: float
    planning_subsample: int

    @property
    def fx(self) -> float:
        return self.width / (2.0 * np.tan(np.radians(self.horizontal_fov_deg / 2.0)))

    @property
    def fy(self) -> float:
        return self.height / (2.0 * np.tan(np.radians(self.vertical_fov_deg / 2.0)))

    @property
    def cx(self) -> float:
        return self.width / 2.0

    @property
    def cy(self) -> float:
        return self.height / 2.0

    @property
    def K(self) -> np.ndarray:
        """3x3 intrinsic matrix."""
        return np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ])

    def depth_noise_std(self, depth: float | np.ndarray) -> float | np.ndarray:
        """Depth noise standard deviation: σ_z = coeff * z^2."""
        return self.depth_noise_coeff * depth**2

    def project_points(
        self,
        points_world: np.ndarray,
        T_cam_world: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project 3D world points into pixel coordinates.

        Args:
            points_world: (N, 3) points in world frame.
            T_cam_world: 4x4 camera-from-world transform (extrinsic).

        Returns:
            pixel_coords: (N, 2) pixel coordinates (u, v).
            depths: (N,) depth values in camera frame.
        """
        points_world = np.asarray(points_world)
        N = len(points_world)
        # Transform to camera frame
        ones = np.ones((N, 1))
        points_h = np.hstack([points_world, ones])  # (N, 4)
        points_cam = (T_cam_world @ points_h.T).T[:, :3]  # (N, 3)

        depths = points_cam[:, 2]
        # Project to pixel coordinates
        u = self.fx * points_cam[:, 0] / depths + self.cx
        v = self.fy * points_cam[:, 1] / depths + self.cy
        pixel_coords = np.column_stack([u, v])

        return pixel_coords, depths

    def is_in_fov(
        self,
        points_world: np.ndarray,
        T_cam_world: np.ndarray,
    ) -> np.ndarray:
        """Check which points fall within the camera FoV and depth range.

        Args:
            points_world: (N, 3) points in world frame.
            T_cam_world: 4x4 camera-from-world transform.

        Returns:
            Boolean mask (N,) — True for visible points.
        """
        pixel_coords, depths = self.project_points(points_world, T_cam_world)

        in_depth = (depths >= self.min_depth) & (depths <= self.max_depth)
        in_x = (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < self.width)
        in_y = (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < self.height)

        return in_depth & in_x & in_y

    def fast_visible_points(
        self,
        points_world: np.ndarray,
        normals_world: np.ndarray,
        T_cam_world: np.ndarray,
    ) -> np.ndarray:
        """Fast approximate visibility check (no raycasting).

        Checks: (a) within FoV, (b) within depth range, (c) facing camera.

        Args:
            points_world: (N, 3) points.
            normals_world: (N, 3) surface normals.
            T_cam_world: 4x4 extrinsic transform.

        Returns:
            Boolean mask (N,) of visible points.
        """
        # Camera position in world frame
        T_world_cam = np.linalg.inv(T_cam_world)
        cam_pos = T_world_cam[:3, 3]

        # Direction from point to camera
        to_cam = cam_pos - points_world  # (N, 3)
        to_cam_norm = np.linalg.norm(to_cam, axis=1, keepdims=True)
        to_cam_norm = np.maximum(to_cam_norm, 1e-10)
        to_cam_dir = to_cam / to_cam_norm

        # Normal facing check: dot(normal, to_camera) > 0
        facing = np.sum(normals_world * to_cam_dir, axis=1) > 0

        # FoV and depth check
        in_fov = self.is_in_fov(points_world, T_cam_world)

        return facing & in_fov

    @classmethod
    def from_yaml(cls, path: str) -> CameraModel:
        """Load camera parameters from YAML config."""
        with open(path) as f:
            cfg = yaml.safe_load(f)["camera"]
        return cls(
            width=cfg["width"],
            height=cfg["height"],
            horizontal_fov_deg=cfg["horizontal_fov_deg"],
            vertical_fov_deg=cfg["vertical_fov_deg"],
            min_depth=cfg["min_depth"],
            max_depth=cfg["max_depth"],
            depth_noise_coeff=cfg["depth_noise_coeff"],
            planning_subsample=cfg["planning_subsample"],
        )

    @classmethod
    def default(cls) -> CameraModel:
        """Create camera model with RealSense D435 defaults."""
        return cls(
            width=640,
            height=480,
            horizontal_fov_deg=87.0,
            vertical_fov_deg=58.0,
            min_depth=0.1,
            max_depth=1.0,
            depth_noise_coeff=0.001,
            planning_subsample=4,
        )


def look_at(
    camera_position: np.ndarray,
    target: np.ndarray,
    up: np.ndarray | None = None,
) -> np.ndarray:
    """Compute a 4x4 camera-from-world transform (look-at).

    Camera convention: Z forward, X right, Y down.

    Args:
        camera_position: (3,) camera position in world.
        target: (3,) point to look at.
        up: (3,) world up direction (default: [0, 0, 1]).

    Returns:
        T_cam_world: 4x4 extrinsic matrix.
    """
    if up is None:
        up = np.array([0.0, 0.0, 1.0])

    camera_position = np.asarray(camera_position, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    # Camera Z axis points toward the target
    z_axis = target - camera_position
    z_axis = z_axis / np.linalg.norm(z_axis)

    # Camera X axis
    x_axis = np.cross(z_axis, up)
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-6:
        # Camera looking straight up or down — use alternative up
        up = np.array([0.0, 1.0, 0.0])
        x_axis = np.cross(z_axis, up)
        x_norm = np.linalg.norm(x_axis)
    x_axis = x_axis / x_norm

    # Camera Y axis
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    # Rotation matrix: world-to-camera
    R = np.stack([x_axis, y_axis, z_axis], axis=0)  # (3, 3)

    # Translation
    t = -R @ camera_position

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T
