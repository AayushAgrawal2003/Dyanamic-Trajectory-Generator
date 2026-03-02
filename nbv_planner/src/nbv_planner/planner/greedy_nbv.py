"""Greedy Next-Best-View planner."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nbv_planner.planner.information_gain import InformationGainScorer
from nbv_planner.planner.viewpoint_sampler import ViewpointSampler
from nbv_planner.representation.coverage_tracker import CoverageTracker, FrameMetrics
from nbv_planner.robot.workspace import RobotWorkspace
from nbv_planner.scene.ground_truth_cloud import GroundTruthCloud
from nbv_planner.scene.synthetic_scene import SyntheticScene
from nbv_planner.sensor.camera_model import CameraModel, look_at
from nbv_planner.sensor.depth_simulator import DepthSimulator


@dataclass
class PlanningResult:
    """Result of a planning run."""

    method_name: str
    poses: list[np.ndarray]
    frame_metrics: list[FrameMetrics]
    total_trajectory_length: float = 0.0

    @property
    def final_coverage(self) -> float:
        if not self.frame_metrics:
            return 0.0
        return self.frame_metrics[-1].cumulative_coverage

    @property
    def num_views(self) -> int:
        return len(self.poses)


class GreedyNBVPlanner:
    """Greedy next-best-view planner.

    At each step:
    1. Sample candidate viewpoints.
    2. Score each candidate by expected information gain.
    3. Select the best candidate.
    4. Execute the view (simulate observation).
    5. Update coverage tracking.
    """

    def __init__(
        self,
        scene: SyntheticScene,
        camera: CameraModel,
        workspace: RobotWorkspace,
        max_views: int = 20,
        num_candidates: int = 300,
        convergence_threshold: float = 0.01,
        coverage_weight: float = 1.0,
        density_weight: float = 0.1,
        neighbor_threshold: float = 0.003,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.scene = scene
        self.camera = camera
        self.workspace = workspace
        self.max_views = max_views
        self.num_candidates = num_candidates
        self.convergence_threshold = convergence_threshold
        self.rng = rng or np.random.default_rng(42)

        # Initialize components
        self.gt_cloud = GroundTruthCloud(
            scene.ground_truth_points,
            scene.ground_truth_normals,
            neighbor_threshold=neighbor_threshold,
        )
        self.coverage_tracker = CoverageTracker(gt_cloud=self.gt_cloud)
        self.scorer = InformationGainScorer(
            self.gt_cloud, camera, coverage_weight, density_weight
        )
        self.sampler = ViewpointSampler(
            scene.center, workspace, rng=self.rng
        )
        self.depth_sim = DepthSimulator(scene.mesh, camera, rng=self.rng)

    def plan(
        self,
        initial_pose: np.ndarray | None = None,
    ) -> PlanningResult:
        """Run greedy NBV planning.

        Args:
            initial_pose: Optional initial camera pose. If None, uses a
                default pose looking at the object from above.

        Returns:
            PlanningResult with poses and metrics.
        """
        poses: list[np.ndarray] = []
        total_traj_length = 0.0

        # Default initial pose: slightly above and in front
        if initial_pose is None:
            initial_pos = self.scene.center + np.array([0.0, -0.3, 0.2])
            initial_pose = look_at(initial_pos, self.scene.center)

        # Execute initial observation
        current_pose = initial_pose
        self._execute_view(current_pose)
        poses.append(current_pose)

        print(f"[Greedy NBV] Initial coverage: "
              f"{self.coverage_tracker.current_coverage:.1%}")

        for step in range(1, self.max_views):
            # Get frontier information for directed sampling
            unobserved_idx = self.gt_cloud.unobserved_indices
            frontier_pts = self.scene.ground_truth_points[unobserved_idx]
            frontier_norms = self.scene.ground_truth_normals[unobserved_idx]

            # Sample candidates
            candidates = self.sampler.sample_combined(
                frontier_pts, frontier_norms, self.num_candidates
            )

            if not candidates:
                print(f"[Greedy NBV] Step {step}: No feasible candidates found.")
                break

            # Score candidates
            scores = self.scorer.score_viewpoints_batch(candidates)
            best_idx = np.argmax(scores)
            best_score = scores[best_idx]

            # Check convergence
            if best_score < self.convergence_threshold * self.gt_cloud.num_points:
                print(f"[Greedy NBV] Step {step}: Converged "
                      f"(best score {best_score:.0f}).")
                break

            # Execute best view
            best_pose = candidates[best_idx]

            # Track trajectory length
            T_world_cam_old = np.linalg.inv(current_pose)
            T_world_cam_new = np.linalg.inv(best_pose)
            step_dist = np.linalg.norm(
                T_world_cam_new[:3, 3] - T_world_cam_old[:3, 3]
            )
            total_traj_length += step_dist

            self._execute_view(best_pose)
            poses.append(best_pose)
            current_pose = best_pose

            print(f"[Greedy NBV] Step {step}: coverage "
                  f"{self.coverage_tracker.current_coverage:.1%}, "
                  f"new pts {self.coverage_tracker.frame_history[-1].new_unique_points}")

        return PlanningResult(
            method_name="greedy_nbv",
            poses=poses,
            frame_metrics=list(self.coverage_tracker.frame_history),
            total_trajectory_length=total_traj_length,
        )

    def _execute_view(self, T_cam_world: np.ndarray) -> None:
        """Simulate an observation from the given pose and update tracking."""
        points = self.depth_sim.simulate_observation(
            T_cam_world, subsample=self.camera.planning_subsample, add_noise=True
        )
        self.coverage_tracker.record_observation(points, T_cam_world)
