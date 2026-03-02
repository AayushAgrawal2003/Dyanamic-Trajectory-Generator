"""Simplified KUKA LBR Med 7 workspace model."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import yaml


@dataclass
class ForbiddenZone:
    """Axis-aligned bounding box representing a forbidden zone."""

    min_corner: np.ndarray
    max_corner: np.ndarray

    def contains(self, point: np.ndarray) -> bool:
        return bool(np.all(point >= self.min_corner) and np.all(point <= self.max_corner))


@dataclass
class RobotWorkspace:
    """Simplified workspace model for KUKA LBR Med 7.

    Models the reachable workspace as a hollow sphere with orientation
    constraints and forbidden zones.
    """

    base_position: np.ndarray
    inner_radius: float
    outer_radius: float
    min_camera_angle_to_target: float  # cos(angle) threshold
    object_center: np.ndarray = field(default_factory=lambda: np.zeros(3))
    forbidden_zones: list[ForbiddenZone] = field(default_factory=list)

    def is_pose_feasible(self, T_cam_world: np.ndarray) -> bool:
        """Check if a camera pose is feasible.

        Checks:
        1. End-effector position within workspace sphere
        2. Camera roughly pointing toward object
        3. Not in a forbidden zone
        """
        # Extract camera position in world frame
        T_world_cam = np.linalg.inv(T_cam_world)
        cam_pos = T_world_cam[:3, 3]

        # 1. Distance from robot base
        dist = np.linalg.norm(cam_pos - self.base_position)
        if dist < self.inner_radius or dist > self.outer_radius:
            return False

        # 2. Camera orientation check
        # Camera z-axis in world frame (viewing direction)
        cam_z_world = T_world_cam[:3, 2]
        # Direction from camera to object
        to_object = self.object_center - cam_pos
        to_object_norm = np.linalg.norm(to_object)
        if to_object_norm < 1e-6:
            return False
        to_object_dir = to_object / to_object_norm
        cos_angle = np.dot(cam_z_world, to_object_dir)
        if cos_angle < self.min_camera_angle_to_target:
            return False

        # 3. Forbidden zones
        for zone in self.forbidden_zones:
            if zone.contains(cam_pos):
                return False

        return True

    def is_position_reachable(self, position: np.ndarray) -> bool:
        """Quick check if a position is within the workspace sphere."""
        dist = np.linalg.norm(position - self.base_position)
        return self.inner_radius <= dist <= self.outer_radius

    @classmethod
    def from_yaml(
        cls,
        path: str,
        object_center: np.ndarray | None = None,
    ) -> RobotWorkspace:
        """Load workspace parameters from YAML config."""
        with open(path) as f:
            cfg = yaml.safe_load(f)["robot"]

        forbidden = []
        for zone in cfg.get("forbidden_zones", []):
            forbidden.append(ForbiddenZone(
                min_corner=np.array(zone["min"]),
                max_corner=np.array(zone["max"]),
            ))

        return cls(
            base_position=np.array(cfg["base_position"]),
            inner_radius=cfg["workspace_inner_radius"],
            outer_radius=cfg["workspace_outer_radius"],
            min_camera_angle_to_target=cfg["min_camera_angle_to_target"],
            object_center=object_center if object_center is not None else np.zeros(3),
            forbidden_zones=forbidden,
        )

    @classmethod
    def default(cls, object_center: np.ndarray | None = None) -> RobotWorkspace:
        """Create workspace with default KUKA Med7 parameters."""
        return cls(
            base_position=np.array([0.0, -0.5, 0.3]),
            inner_radius=0.3,
            outer_radius=0.8,
            min_camera_angle_to_target=0.5,
            object_center=object_center if object_center is not None else np.zeros(3),
            forbidden_zones=[
                ForbiddenZone(
                    min_corner=np.array([-1.0, -1.0, -0.05]),
                    max_corner=np.array([1.0, 1.0, 0.0]),
                ),
            ],
        )
