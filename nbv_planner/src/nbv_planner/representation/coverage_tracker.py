"""Simple point-based coverage tracking."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nbv_planner.scene.ground_truth_cloud import GroundTruthCloud


@dataclass
class FrameMetrics:
    """Metrics for a single observation frame."""

    frame_index: int
    total_points_in_frame: int
    new_unique_points: int
    cumulative_coverage: float
    camera_pose: np.ndarray  # 4x4


@dataclass
class CoverageTracker:
    """Track per-frame and cumulative coverage metrics."""

    gt_cloud: GroundTruthCloud
    frame_history: list[FrameMetrics] = field(default_factory=list)

    def record_observation(
        self,
        observed_points: np.ndarray,
        camera_pose: np.ndarray,
    ) -> FrameMetrics:
        """Record a new observation and update coverage.

        Args:
            observed_points: (M, 3) newly observed points.
            camera_pose: 4x4 camera pose used for this observation.

        Returns:
            FrameMetrics for this observation.
        """
        total_in_frame = len(observed_points)
        new_unique = self.gt_cloud.update_coverage(observed_points)
        cumulative = self.gt_cloud.coverage_fraction

        frame = FrameMetrics(
            frame_index=len(self.frame_history),
            total_points_in_frame=total_in_frame,
            new_unique_points=new_unique,
            cumulative_coverage=cumulative,
            camera_pose=camera_pose.copy(),
        )
        self.frame_history.append(frame)
        return frame

    @property
    def total_frames(self) -> int:
        return len(self.frame_history)

    @property
    def current_coverage(self) -> float:
        return self.gt_cloud.coverage_fraction

    @property
    def total_unique_observed(self) -> int:
        return self.gt_cloud.observed_count

    def get_coverage_curve(self) -> np.ndarray:
        """Return (N,) array of cumulative coverage after each frame."""
        return np.array([f.cumulative_coverage for f in self.frame_history])

    def get_new_points_per_frame(self) -> np.ndarray:
        """Return (N,) array of new unique points per frame."""
        return np.array([f.new_unique_points for f in self.frame_history])

    def get_points_per_frame(self) -> np.ndarray:
        """Return (N,) array of total points captured per frame."""
        return np.array([f.total_points_in_frame for f in self.frame_history])

    def reset(self) -> None:
        """Reset all tracking."""
        self.gt_cloud.reset()
        self.frame_history.clear()
