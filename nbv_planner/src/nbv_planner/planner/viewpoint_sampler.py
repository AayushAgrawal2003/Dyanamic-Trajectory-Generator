"""Sample candidate camera poses for NBV evaluation."""

from __future__ import annotations

import numpy as np

from nbv_planner.robot.workspace import RobotWorkspace
from nbv_planner.sensor.camera_model import look_at


class ViewpointSampler:
    """Generate candidate camera poses on spheres around the target."""

    def __init__(
        self,
        object_center: np.ndarray,
        workspace: RobotWorkspace,
        radii: list[float] | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.object_center = np.asarray(object_center)
        self.workspace = workspace
        self.radii = radii or [0.2, 0.3, 0.4, 0.5]
        self.rng = rng or np.random.default_rng(42)

    def sample_sphere(
        self,
        num_candidates: int = 300,
    ) -> list[np.ndarray]:
        """Strategy 1: Uniform sphere sampling at multiple radii.

        Returns:
            List of 4x4 camera-from-world transforms.
        """
        candidates = []
        per_radius = max(1, num_candidates // len(self.radii))

        for radius in self.radii:
            positions = self._sample_on_sphere(per_radius, radius)
            for pos in positions:
                T = look_at(pos, self.object_center)
                if self.workspace.is_pose_feasible(T):
                    candidates.append(T)

        return candidates

    def sample_frontier_directed(
        self,
        frontier_positions: np.ndarray,
        frontier_normals: np.ndarray | None = None,
        num_candidates: int = 300,
    ) -> list[np.ndarray]:
        """Strategy 2: Sample viewpoints biased toward uncovered regions.

        Args:
            frontier_positions: (M, 3) positions of uncovered/frontier points.
            frontier_normals: (M, 3) optional surface normals at frontier points.
            num_candidates: Number of candidates to generate.

        Returns:
            List of 4x4 camera-from-world transforms.
        """
        if len(frontier_positions) == 0:
            return self.sample_sphere(num_candidates)

        candidates = []
        attempts = 0
        max_attempts = num_candidates * 5

        while len(candidates) < num_candidates and attempts < max_attempts:
            attempts += 1

            # Pick a random frontier point
            idx = self.rng.integers(len(frontier_positions))
            target = frontier_positions[idx]

            # Choose a random radius
            radius = self.rng.choice(self.radii)

            # If we have normals, bias the viewing direction along the normal
            if frontier_normals is not None and len(frontier_normals) > 0:
                normal = frontier_normals[idx]
                # Camera should be on the normal side of the surface
                # Add some random perturbation
                direction = normal + 0.3 * self.rng.standard_normal(3)
                direction = direction / np.linalg.norm(direction)
            else:
                direction = self._random_direction()

            cam_pos = target + radius * direction

            T = look_at(cam_pos, self.object_center)
            if self.workspace.is_pose_feasible(T):
                candidates.append(T)

        return candidates

    def sample_combined(
        self,
        frontier_positions: np.ndarray | None = None,
        frontier_normals: np.ndarray | None = None,
        num_candidates: int = 300,
    ) -> list[np.ndarray]:
        """Combined sampling: sphere + frontier-directed.

        Uses frontier-directed if frontier info is available, otherwise sphere.
        Always includes some sphere samples for diversity.
        """
        if frontier_positions is None or len(frontier_positions) == 0:
            return self.sample_sphere(num_candidates)

        # 70% frontier-directed, 30% sphere
        n_frontier = int(0.7 * num_candidates)
        n_sphere = num_candidates - n_frontier

        frontier_cands = self.sample_frontier_directed(
            frontier_positions, frontier_normals, n_frontier
        )
        sphere_cands = self.sample_sphere(n_sphere)

        return frontier_cands + sphere_cands

    def _sample_on_sphere(self, n: int, radius: float) -> np.ndarray:
        """Sample n points uniformly on a sphere of given radius.

        Uses hemisphere (z > 0) since camera should be above the table.
        """
        positions = []
        while len(positions) < n:
            # Uniform sphere sampling via normal distribution
            batch_size = n * 3
            pts = self.rng.standard_normal((batch_size, 3))
            pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
            # Filter to upper hemisphere (camera above table)
            pts = pts[pts[:, 2] > 0.05]
            pts = pts * radius + self.object_center
            positions.extend(pts[:n - len(positions)])

        return np.array(positions[:n])

    def _random_direction(self) -> np.ndarray:
        """Sample a random unit direction (upper hemisphere)."""
        d = self.rng.standard_normal(3)
        if d[2] < 0:
            d[2] = -d[2]
        return d / np.linalg.norm(d)
