"""Ground truth point cloud management and coverage computation."""

from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree


class GroundTruthCloud:
    """Manages a ground truth point cloud and computes coverage."""

    def __init__(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        neighbor_threshold: float = 0.003,
    ) -> None:
        self.points = np.asarray(points, dtype=np.float64)  # (N, 3)
        self.normals = np.asarray(normals, dtype=np.float64)  # (N, 3)
        self.neighbor_threshold = neighbor_threshold
        self.num_points = len(self.points)

        # Build KD-tree for fast nearest-neighbor queries
        self._tree = KDTree(self.points)

        # Track which ground truth points have been observed
        self._observed_mask = np.zeros(self.num_points, dtype=bool)

    @property
    def observed_count(self) -> int:
        return int(self._observed_mask.sum())

    @property
    def coverage_fraction(self) -> float:
        if self.num_points == 0:
            return 0.0
        return self.observed_count / self.num_points

    @property
    def observed_indices(self) -> np.ndarray:
        return np.nonzero(self._observed_mask)[0]

    @property
    def unobserved_indices(self) -> np.ndarray:
        return np.nonzero(~self._observed_mask)[0]

    def update_coverage(self, observed_points: np.ndarray) -> int:
        """Update coverage with new observed points.

        Args:
            observed_points: (M, 3) array of observed 3D points.

        Returns:
            Number of newly covered ground truth points.
        """
        if len(observed_points) == 0:
            return 0

        observed_points = np.asarray(observed_points, dtype=np.float64)
        # Find ground truth points that have a neighbor in the observation
        # within the threshold distance.
        obs_tree = KDTree(observed_points)
        distances, _ = obs_tree.query(self.points)

        newly_covered = (distances <= self.neighbor_threshold) & (~self._observed_mask)
        self._observed_mask |= (distances <= self.neighbor_threshold)

        return int(newly_covered.sum())

    def compute_coverage_of(self, observed_points: np.ndarray) -> float:
        """Compute what fraction of GT points are covered by given points.

        Does NOT update internal state.
        """
        if len(observed_points) == 0:
            return 0.0
        obs_tree = KDTree(observed_points)
        distances, _ = obs_tree.query(self.points)
        covered = (distances <= self.neighbor_threshold).sum()
        return covered / self.num_points

    def count_visible_gt_points(
        self,
        visible_gt_indices: np.ndarray,
    ) -> tuple[int, int]:
        """Given indices of GT points visible from a viewpoint, return counts.

        Returns:
            (total_visible, new_unique) — total visible GT points and how many
            are not yet observed.
        """
        total_visible = len(visible_gt_indices)
        if total_visible == 0:
            return 0, 0
        new_unique = int((~self._observed_mask[visible_gt_indices]).sum())
        return total_visible, new_unique

    def reset(self) -> None:
        """Reset all coverage tracking."""
        self._observed_mask[:] = False
