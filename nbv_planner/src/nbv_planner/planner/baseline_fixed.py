"""Fixed trajectory baselines for comparison."""

from __future__ import annotations

import numpy as np

from nbv_planner.planner.greedy_nbv import PlanningResult
from nbv_planner.representation.coverage_tracker import CoverageTracker
from nbv_planner.scene.ground_truth_cloud import GroundTruthCloud
from nbv_planner.scene.synthetic_scene import SyntheticScene
from nbv_planner.sensor.camera_model import CameraModel, look_at
from nbv_planner.sensor.depth_simulator import DepthSimulator


class FixedArcBaseline:
    """Fixed circular arc trajectory around the object.

    N evenly spaced viewpoints over a configurable arc angle.
    Camera always points at object center.
    """

    def __init__(
        self,
        scene: SyntheticScene,
        camera: CameraModel,
        num_views: int = 20,
        arc_angle_deg: float = 180.0,
        radius: float = 0.35,
        height: float = 0.1,
        neighbor_threshold: float = 0.003,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.scene = scene
        self.camera = camera
        self.num_views = num_views
        self.arc_angle = np.radians(arc_angle_deg)
        self.radius = radius
        self.height = height
        self.neighbor_threshold = neighbor_threshold
        self.rng = rng or np.random.default_rng(42)

    def plan(self) -> PlanningResult:
        """Generate and execute the fixed arc trajectory."""
        gt_cloud = GroundTruthCloud(
            self.scene.ground_truth_points,
            self.scene.ground_truth_normals,
            neighbor_threshold=self.neighbor_threshold,
        )
        tracker = CoverageTracker(gt_cloud=gt_cloud)
        depth_sim = DepthSimulator(self.scene.mesh, self.camera, rng=self.rng)

        # Generate arc poses
        angles = np.linspace(
            -self.arc_angle / 2, self.arc_angle / 2, self.num_views
        )
        center = self.scene.center
        poses = []
        total_traj_length = 0.0

        for i, angle in enumerate(angles):
            x = center[0] + self.radius * np.cos(angle)
            y = center[1] + self.radius * np.sin(angle)
            z = center[2] + self.height
            cam_pos = np.array([x, y, z])

            T = look_at(cam_pos, center)
            poses.append(T)

            # Simulate observation
            points = depth_sim.simulate_observation(
                T, subsample=self.camera.planning_subsample, add_noise=True
            )
            tracker.record_observation(points, T)

            if i > 0:
                T_old = np.linalg.inv(poses[i - 1])
                T_new = np.linalg.inv(T)
                total_traj_length += np.linalg.norm(T_new[:3, 3] - T_old[:3, 3])

            print(f"[FixedArc] Step {i}: coverage {tracker.current_coverage:.1%}")

        return PlanningResult(
            method_name="fixed_arc",
            poses=poses,
            frame_metrics=list(tracker.frame_history),
            total_trajectory_length=total_traj_length,
        )


class RasterBaseline:
    """Raster (lawnmower) pattern scanning from above the object.

    Grid of viewpoints on a plane above the object, all looking down.
    """

    def __init__(
        self,
        scene: SyntheticScene,
        camera: CameraModel,
        num_views: int = 20,
        scan_height: float = 0.3,
        neighbor_threshold: float = 0.003,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.scene = scene
        self.camera = camera
        self.num_views = num_views
        self.scan_height = scan_height
        self.neighbor_threshold = neighbor_threshold
        self.rng = rng or np.random.default_rng(42)

    def plan(self) -> PlanningResult:
        """Generate and execute the raster scanning pattern."""
        gt_cloud = GroundTruthCloud(
            self.scene.ground_truth_points,
            self.scene.ground_truth_normals,
            neighbor_threshold=self.neighbor_threshold,
        )
        tracker = CoverageTracker(gt_cloud=gt_cloud)
        depth_sim = DepthSimulator(self.scene.mesh, self.camera, rng=self.rng)

        center = self.scene.center
        extent = self.scene.extent

        # Compute grid dimensions
        grid_side = int(np.ceil(np.sqrt(self.num_views)))
        x_range = np.linspace(
            center[0] - extent[0] * 0.6,
            center[0] + extent[0] * 0.6,
            grid_side,
        )
        y_range = np.linspace(
            center[1] - extent[1] * 0.6,
            center[1] + extent[1] * 0.6,
            grid_side,
        )

        # Lawnmower pattern
        poses = []
        total_traj_length = 0.0
        view_count = 0

        for i, x in enumerate(x_range):
            ys = y_range if i % 2 == 0 else y_range[::-1]
            for y in ys:
                if view_count >= self.num_views:
                    break
                cam_pos = np.array([x, y, center[2] + self.scan_height])
                T = look_at(cam_pos, center)
                poses.append(T)

                points = depth_sim.simulate_observation(
                    T, subsample=self.camera.planning_subsample, add_noise=True
                )
                tracker.record_observation(points, T)

                if view_count > 0:
                    T_old = np.linalg.inv(poses[-2])
                    T_new = np.linalg.inv(T)
                    total_traj_length += np.linalg.norm(
                        T_new[:3, 3] - T_old[:3, 3]
                    )

                print(f"[Raster] Step {view_count}: "
                      f"coverage {tracker.current_coverage:.1%}")
                view_count += 1
            if view_count >= self.num_views:
                break

        return PlanningResult(
            method_name="raster",
            poses=poses,
            frame_metrics=list(tracker.frame_history),
            total_trajectory_length=total_traj_length,
        )
