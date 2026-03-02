"""Sequence-optimized NBV planner with TSP-based trajectory optimization."""

from __future__ import annotations

import numpy as np

from nbv_planner.planner.greedy_nbv import PlanningResult
from nbv_planner.planner.information_gain import InformationGainScorer
from nbv_planner.planner.viewpoint_sampler import ViewpointSampler
from nbv_planner.representation.coverage_tracker import CoverageTracker
from nbv_planner.robot.workspace import RobotWorkspace
from nbv_planner.scene.ground_truth_cloud import GroundTruthCloud
from nbv_planner.scene.synthetic_scene import SyntheticScene
from nbv_planner.sensor.camera_model import CameraModel, look_at
from nbv_planner.sensor.depth_simulator import DepthSimulator


class SequenceOptimizer:
    """Two-phase NBV planner: select high-value viewpoints, then optimize order.

    Phase 1: Score all candidates against initial state, select top-M
             diverse viewpoints.
    Phase 2: Solve a TSP to find the shortest path through them.
    """

    def __init__(
        self,
        scene: SyntheticScene,
        camera: CameraModel,
        workspace: RobotWorkspace,
        max_views: int = 20,
        num_candidates: int = 500,
        min_angular_separation_deg: float = 15.0,
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
        self.min_angular_sep = np.radians(min_angular_separation_deg)
        self.rng = rng or np.random.default_rng(42)
        self.neighbor_threshold = neighbor_threshold
        self.coverage_weight = coverage_weight
        self.density_weight = density_weight

    def plan(
        self,
        initial_pose: np.ndarray | None = None,
    ) -> PlanningResult:
        """Run sequence-optimized NBV planning.

        Args:
            initial_pose: Optional initial camera pose.

        Returns:
            PlanningResult with TSP-ordered poses and metrics.
        """
        if initial_pose is None:
            initial_pos = self.scene.center + np.array([0.0, -0.3, 0.2])
            initial_pose = look_at(initial_pos, self.scene.center)

        # Phase 1: Select diverse high-value viewpoints
        print("[SeqOpt] Phase 1: Selecting high-value viewpoints...")
        selected_poses = self._select_viewpoints(initial_pose)
        print(f"[SeqOpt] Selected {len(selected_poses)} viewpoints.")

        # Phase 2: Optimize ordering with TSP
        print("[SeqOpt] Phase 2: Optimizing trajectory order...")
        ordered_poses = self._solve_tsp(initial_pose, selected_poses)

        # Execute the planned sequence and collect metrics
        print("[SeqOpt] Executing planned sequence...")
        result = self._execute_sequence(initial_pose, ordered_poses)
        return result

    def _select_viewpoints(
        self,
        initial_pose: np.ndarray,
    ) -> list[np.ndarray]:
        """Phase 1: Select diverse, high-scoring viewpoints."""
        gt_cloud = GroundTruthCloud(
            self.scene.ground_truth_points,
            self.scene.ground_truth_normals,
            neighbor_threshold=self.neighbor_threshold,
        )
        scorer = InformationGainScorer(
            gt_cloud, self.camera, self.coverage_weight, self.density_weight
        )
        sampler = ViewpointSampler(
            self.scene.center,
            self.workspace,
            rng=np.random.default_rng(self.rng.integers(2**31)),
        )

        # Generate a large pool of candidates
        candidates = sampler.sample_sphere(self.num_candidates)
        if not candidates:
            return []

        # Score all candidates against the initial (empty) state
        scores = scorer.score_viewpoints_batch(candidates)

        # Sort by score descending
        sorted_indices = np.argsort(-scores)

        # Greedily select diverse viewpoints
        selected: list[np.ndarray] = []
        selected_positions: list[np.ndarray] = []

        for idx in sorted_indices:
            if len(selected) >= self.max_views:
                break

            T = candidates[idx]
            T_world_cam = np.linalg.inv(T)
            pos = T_world_cam[:3, 3]

            # Check angular separation from all already-selected viewpoints
            too_close = False
            for sel_pos in selected_positions:
                # Angular separation as seen from object center
                v1 = pos - self.scene.center
                v2 = sel_pos - self.scene.center
                cos_angle = np.dot(v1, v2) / (
                    np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10
                )
                angle = np.arccos(np.clip(cos_angle, -1, 1))
                if angle < self.min_angular_sep:
                    too_close = True
                    break

            if not too_close:
                selected.append(T)
                selected_positions.append(pos)

        return selected

    def _solve_tsp(
        self,
        start_pose: np.ndarray,
        viewpoints: list[np.ndarray],
    ) -> list[np.ndarray]:
        """Solve TSP to order viewpoints for minimum travel distance.

        Uses nearest-neighbor heuristic + 2-opt improvement.
        """
        if len(viewpoints) <= 1:
            return viewpoints

        # Extract positions
        positions = []
        T_world_start = np.linalg.inv(start_pose)
        positions.append(T_world_start[:3, 3])
        for T in viewpoints:
            T_world = np.linalg.inv(T)
            positions.append(T_world[:3, 3])
        positions = np.array(positions)

        n = len(positions)
        # Distance matrix
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist_matrix[i, j] = np.linalg.norm(positions[i] - positions[j])

        # Nearest-neighbor heuristic starting from node 0 (start pose)
        order = self._nearest_neighbor_tsp(dist_matrix, start=0)

        # 2-opt improvement
        order = self._two_opt(order, dist_matrix)

        # Remove start node (index 0) and map back to viewpoints
        order_without_start = [i - 1 for i in order if i > 0]
        return [viewpoints[i] for i in order_without_start]

    def _nearest_neighbor_tsp(
        self,
        dist_matrix: np.ndarray,
        start: int = 0,
    ) -> list[int]:
        """Nearest-neighbor TSP heuristic."""
        n = len(dist_matrix)
        visited = {start}
        order = [start]

        current = start
        for _ in range(n - 1):
            # Find nearest unvisited
            best_dist = np.inf
            best_next = -1
            for j in range(n):
                if j not in visited and dist_matrix[current, j] < best_dist:
                    best_dist = dist_matrix[current, j]
                    best_next = j
            if best_next < 0:
                break
            visited.add(best_next)
            order.append(best_next)
            current = best_next

        return order

    def _two_opt(
        self,
        order: list[int],
        dist_matrix: np.ndarray,
        max_iterations: int = 100,
    ) -> list[int]:
        """2-opt improvement for TSP."""
        route = list(order)
        n = len(route)
        improved = True
        iteration = 0

        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    # Cost of current edges
                    d_old = (
                        dist_matrix[route[i - 1], route[i]]
                        + dist_matrix[route[j], route[(j + 1) % n]]
                    )
                    # Cost of reversed segment
                    d_new = (
                        dist_matrix[route[i - 1], route[j]]
                        + dist_matrix[route[i], route[(j + 1) % n]]
                    )
                    if d_new < d_old - 1e-10:
                        route[i:j + 1] = route[i:j + 1][::-1]
                        improved = True

        return route

    def _execute_sequence(
        self,
        initial_pose: np.ndarray,
        ordered_poses: list[np.ndarray],
    ) -> PlanningResult:
        """Execute the ordered sequence and collect metrics."""
        gt_cloud = GroundTruthCloud(
            self.scene.ground_truth_points,
            self.scene.ground_truth_normals,
            neighbor_threshold=self.neighbor_threshold,
        )
        tracker = CoverageTracker(gt_cloud=gt_cloud)
        depth_sim = DepthSimulator(
            self.scene.mesh, self.camera,
            rng=np.random.default_rng(self.rng.integers(2**31)),
        )

        all_poses = [initial_pose] + ordered_poses
        total_traj_length = 0.0

        for i, pose in enumerate(all_poses):
            points = depth_sim.simulate_observation(
                pose, subsample=self.camera.planning_subsample, add_noise=True
            )
            tracker.record_observation(points, pose)

            if i > 0:
                T_old = np.linalg.inv(all_poses[i - 1])
                T_new = np.linalg.inv(pose)
                total_traj_length += np.linalg.norm(T_new[:3, 3] - T_old[:3, 3])

            print(f"[SeqOpt] Step {i}: coverage {tracker.current_coverage:.1%}")

        return PlanningResult(
            method_name="sequence_optimized",
            poses=all_poses,
            frame_metrics=list(tracker.frame_history),
            total_trajectory_length=total_traj_length,
        )
