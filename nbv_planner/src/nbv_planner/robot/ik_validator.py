"""IK validation wrapper.

Currently wraps the simplified workspace model.
Later to be replaced with actual MoveIt2 IK calls.
"""

from __future__ import annotations

import numpy as np

from nbv_planner.robot.workspace import RobotWorkspace


class IKValidator:
    """Validates camera poses for robot feasibility."""

    def __init__(self, workspace: RobotWorkspace) -> None:
        self.workspace = workspace

    def validate_pose(
        self,
        T_cam_world: np.ndarray,
        current_pose: np.ndarray | None = None,
    ) -> tuple[bool, float]:
        """Validate a camera pose for feasibility.

        Args:
            T_cam_world: 4x4 candidate camera pose.
            current_pose: 4x4 current camera pose (for cost computation).

        Returns:
            (feasible, cost) — feasibility flag and movement cost.
        """
        feasible = self.workspace.is_pose_feasible(T_cam_world)

        cost = 0.0
        if current_pose is not None and feasible:
            # Cost = Euclidean distance between camera positions
            T_world_cam_new = np.linalg.inv(T_cam_world)
            T_world_cam_cur = np.linalg.inv(current_pose)
            cost = float(np.linalg.norm(
                T_world_cam_new[:3, 3] - T_world_cam_cur[:3, 3]
            ))

        return feasible, cost
