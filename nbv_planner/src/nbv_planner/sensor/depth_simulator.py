"""Raycasting-based depth image simulator."""

from __future__ import annotations

import numpy as np
import trimesh

from nbv_planner.sensor.camera_model import CameraModel


class DepthSimulator:
    """Simulates depth images by raycasting against a triangle mesh."""

    def __init__(
        self,
        mesh: trimesh.Trimesh,
        camera: CameraModel,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.mesh = mesh
        self.camera = camera
        self.rng = rng or np.random.default_rng(42)

        # Build ray intersector — prefer embree, fall back to built-in
        try:
            self._intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
        except Exception:
            self._intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)

    def simulate_depth_image(
        self,
        T_cam_world: np.ndarray,
        subsample: int = 1,
        add_noise: bool = True,
    ) -> np.ndarray:
        """Simulate a depth image from the given camera pose.

        Args:
            T_cam_world: 4x4 camera-from-world extrinsic.
            subsample: Pixel subsample factor (e.g., 4 means 1/4 resolution).
            add_noise: Whether to add depth-dependent Gaussian noise.

        Returns:
            depth_image: (H // subsample, W // subsample) depth array.
                         0.0 means no intersection.
        """
        W = self.camera.width // subsample
        H = self.camera.height // subsample

        # Generate pixel grid
        u = np.arange(W) * subsample + subsample / 2.0
        v = np.arange(H) * subsample + subsample / 2.0
        uu, vv = np.meshgrid(u, v, indexing="xy")
        pixels = np.column_stack([uu.ravel(), vv.ravel()])

        # Backproject pixels to rays in camera frame
        x_cam = (pixels[:, 0] - self.camera.cx) / self.camera.fx
        y_cam = (pixels[:, 1] - self.camera.cy) / self.camera.fy
        z_cam = np.ones(len(pixels))
        dirs_cam = np.column_stack([x_cam, y_cam, z_cam])
        dirs_cam /= np.linalg.norm(dirs_cam, axis=1, keepdims=True)

        # Transform rays to world frame
        T_world_cam = np.linalg.inv(T_cam_world)
        R = T_world_cam[:3, :3]
        cam_pos = T_world_cam[:3, 3]

        dirs_world = (R @ dirs_cam.T).T
        origins = np.tile(cam_pos, (len(dirs_world), 1))

        # Raycast
        locations, index_ray, _ = self._intersector.intersects_location(
            origins, dirs_world, multiple_hits=False
        )

        # Build depth image
        depth_flat = np.zeros(len(pixels))
        if len(locations) > 0:
            # Compute depth as distance along camera Z axis
            hit_cam = (T_cam_world[:3, :3] @ locations.T).T + T_cam_world[:3, 3]
            depths = hit_cam[:, 2]

            # Filter by depth range
            valid = (depths >= self.camera.min_depth) & (depths <= self.camera.max_depth)
            if add_noise:
                noise = self.rng.normal(
                    0, self.camera.depth_noise_std(depths), size=len(depths)
                )
                depths = depths + noise

            depth_flat[index_ray[valid]] = depths[valid]

        return depth_flat.reshape(H, W)

    def depth_to_point_cloud(
        self,
        depth_image: np.ndarray,
        T_cam_world: np.ndarray,
        subsample: int = 1,
    ) -> np.ndarray:
        """Convert a depth image to a 3D point cloud in world frame.

        Args:
            depth_image: (H, W) depth values (0 = no point).
            T_cam_world: 4x4 extrinsic.
            subsample: The subsample factor used when generating the depth image.

        Returns:
            points_world: (M, 3) point cloud.
        """
        H, W = depth_image.shape
        mask = depth_image > 0

        v_indices, u_indices = np.nonzero(mask)
        depths = depth_image[mask]

        # Pixel coordinates in original resolution
        u_orig = u_indices * subsample + subsample / 2.0
        v_orig = v_indices * subsample + subsample / 2.0

        # Backproject to camera frame
        x_cam = (u_orig - self.camera.cx) / self.camera.fx * depths
        y_cam = (v_orig - self.camera.cy) / self.camera.fy * depths
        z_cam = depths
        points_cam = np.column_stack([x_cam, y_cam, z_cam])

        # Transform to world frame
        T_world_cam = np.linalg.inv(T_cam_world)
        R = T_world_cam[:3, :3]
        t = T_world_cam[:3, 3]
        points_world = (R @ points_cam.T).T + t

        return points_world

    def simulate_observation(
        self,
        T_cam_world: np.ndarray,
        subsample: int = 1,
        add_noise: bool = True,
    ) -> np.ndarray:
        """Full pipeline: pose -> point cloud in world frame.

        Args:
            T_cam_world: 4x4 extrinsic.
            subsample: Pixel subsample factor.
            add_noise: Whether to add noise.

        Returns:
            points_world: (M, 3) observed point cloud.
        """
        depth = self.simulate_depth_image(T_cam_world, subsample, add_noise)
        return self.depth_to_point_cloud(depth, T_cam_world, subsample)
