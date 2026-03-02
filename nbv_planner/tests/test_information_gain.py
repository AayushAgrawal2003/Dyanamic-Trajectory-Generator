"""Tests for information gain scoring."""

import numpy as np
import pytest

from nbv_planner.planner.information_gain import InformationGainScorer
from nbv_planner.scene.ground_truth_cloud import GroundTruthCloud
from nbv_planner.sensor.camera_model import CameraModel, look_at


class TestInformationGainScorer:
    def setup_method(self):
        # Create a simple set of ground truth points on a hemisphere
        rng = np.random.default_rng(42)
        n = 1000
        theta = rng.uniform(0, 2 * np.pi, n)
        phi = rng.uniform(0, np.pi / 2, n)
        r = 0.05
        self.points = np.column_stack([
            r * np.sin(phi) * np.cos(theta),
            r * np.sin(phi) * np.sin(theta),
            r * np.cos(phi),
        ])
        # Normals point outward
        self.normals = self.points / np.linalg.norm(self.points, axis=1, keepdims=True)

        self.gt_cloud = GroundTruthCloud(self.points, self.normals)
        self.camera = CameraModel.default()
        self.scorer = InformationGainScorer(self.gt_cloud, self.camera)

    def test_viewpoint_facing_object_has_positive_score(self):
        cam_pos = np.array([0.0, 0.0, 0.3])
        T = look_at(cam_pos, np.array([0.0, 0.0, 0.0]))
        score = self.scorer.score_viewpoint(T)
        assert score > 0

    def test_viewpoint_behind_object_has_low_score(self):
        # Camera below the hemisphere, looking up — normals face away
        cam_pos = np.array([0.0, 0.0, -0.3])
        T = look_at(cam_pos, np.array([0.0, 0.0, 0.0]))
        score_behind = self.scorer.score_viewpoint(T)

        # Compare with camera above, looking down
        cam_pos2 = np.array([0.0, 0.0, 0.3])
        T2 = look_at(cam_pos2, np.array([0.0, 0.0, 0.0]))
        score_front = self.scorer.score_viewpoint(T2)

        assert score_front > score_behind

    def test_batch_scoring(self):
        poses = [
            look_at(np.array([0.0, 0.0, 0.3]), np.zeros(3)),
            look_at(np.array([0.3, 0.0, 0.1]), np.zeros(3)),
            look_at(np.array([0.0, 0.3, 0.1]), np.zeros(3)),
        ]
        scores = self.scorer.score_viewpoints_batch(poses)
        assert len(scores) == 3
        assert all(s >= 0 for s in scores)

    def test_score_decreases_after_observation(self):
        cam_pos = np.array([0.0, 0.0, 0.3])
        T = look_at(cam_pos, np.array([0.0, 0.0, 0.0]))

        score_before = self.scorer.score_viewpoint(T)

        # Simulate observing these points
        visible = self.camera.fast_visible_points(self.points, self.normals, T)
        self.gt_cloud.update_coverage(self.points[visible])

        score_after = self.scorer.score_viewpoint(T)
        # After observing, re-viewing the same position should score lower
        assert score_after < score_before

    def test_detailed_score(self):
        cam_pos = np.array([0.0, 0.0, 0.3])
        T = look_at(cam_pos, np.array([0.0, 0.0, 0.0]))
        details = self.scorer.get_detailed_score(T)
        assert "total_visible" in details
        assert "new_unique" in details
        assert "score" in details
        assert details["total_visible"] >= details["new_unique"]
