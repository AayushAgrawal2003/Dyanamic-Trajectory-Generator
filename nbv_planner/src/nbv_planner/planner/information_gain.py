"""Score viewpoints by expected information gain."""

from __future__ import annotations

import numpy as np

from nbv_planner.scene.ground_truth_cloud import GroundTruthCloud
from nbv_planner.sensor.camera_model import CameraModel


class InformationGainScorer:
    """Score candidate viewpoints using the fast approximate sensor model.

    Uses vectorized operations to evaluate hundreds of candidates per second.
    """

    def __init__(
        self,
        gt_cloud: GroundTruthCloud,
        camera: CameraModel,
        coverage_weight: float = 1.0,
        density_weight: float = 0.1,
    ) -> None:
        self.gt_cloud = gt_cloud
        self.camera = camera
        self.alpha = coverage_weight
        self.beta = density_weight

        # Pre-compute for speed
        self._gt_points = gt_cloud.points
        self._gt_normals = gt_cloud.normals

    def score_viewpoint(self, T_cam_world: np.ndarray) -> float:
        """Score a single viewpoint.

        score = α * expected_new_unique_points + β * expected_total_points_in_frame

        Uses fast approximate visibility (no raycasting).
        """
        visible_mask = self.camera.fast_visible_points(
            self._gt_points, self._gt_normals, T_cam_world
        )
        visible_indices = np.nonzero(visible_mask)[0]

        total_visible, new_unique = self.gt_cloud.count_visible_gt_points(
            visible_indices
        )

        return self.alpha * new_unique + self.beta * total_visible

    def score_viewpoints_batch(
        self,
        candidates: list[np.ndarray],
    ) -> np.ndarray:
        """Score multiple viewpoints.

        Args:
            candidates: List of 4x4 camera-from-world transforms.

        Returns:
            (N,) array of scores.
        """
        scores = np.array([self.score_viewpoint(T) for T in candidates])
        return scores

    def get_visible_gt_indices(
        self,
        T_cam_world: np.ndarray,
    ) -> np.ndarray:
        """Get indices of ground truth points visible from a viewpoint."""
        visible_mask = self.camera.fast_visible_points(
            self._gt_points, self._gt_normals, T_cam_world
        )
        return np.nonzero(visible_mask)[0]

    def get_detailed_score(
        self,
        T_cam_world: np.ndarray,
    ) -> dict:
        """Get detailed breakdown of a viewpoint's score."""
        visible_mask = self.camera.fast_visible_points(
            self._gt_points, self._gt_normals, T_cam_world
        )
        visible_indices = np.nonzero(visible_mask)[0]
        total_visible, new_unique = self.gt_cloud.count_visible_gt_points(
            visible_indices
        )

        return {
            "total_visible": total_visible,
            "new_unique": new_unique,
            "score": self.alpha * new_unique + self.beta * total_visible,
            "visible_fraction": total_visible / max(1, self.gt_cloud.num_points),
            "new_fraction": new_unique / max(1, self.gt_cloud.num_points),
        }
