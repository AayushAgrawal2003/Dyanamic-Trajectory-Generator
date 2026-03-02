"""Tests for the viewpoint sampler."""

import numpy as np
import pytest

from nbv_planner.planner.viewpoint_sampler import ViewpointSampler
from nbv_planner.robot.workspace import RobotWorkspace


class TestViewpointSampler:
    def setup_method(self):
        self.object_center = np.array([0.0, 0.0, 0.05])
        self.workspace = RobotWorkspace.default(object_center=self.object_center)
        self.sampler = ViewpointSampler(
            self.object_center,
            self.workspace,
            rng=np.random.default_rng(42),
        )

    def test_sphere_sampling_returns_poses(self):
        candidates = self.sampler.sample_sphere(num_candidates=50)
        assert len(candidates) > 0
        for T in candidates:
            assert T.shape == (4, 4)

    def test_sphere_samples_are_feasible(self):
        candidates = self.sampler.sample_sphere(num_candidates=50)
        for T in candidates:
            assert self.workspace.is_pose_feasible(T)

    def test_camera_looks_toward_object(self):
        candidates = self.sampler.sample_sphere(num_candidates=20)
        for T in candidates:
            T_world = np.linalg.inv(T)
            cam_pos = T_world[:3, 3]
            cam_z = T_world[:3, 2]  # Camera forward direction
            to_object = self.object_center - cam_pos
            to_object = to_object / np.linalg.norm(to_object)
            cos_angle = np.dot(cam_z, to_object)
            assert cos_angle > 0.3  # Roughly pointing toward object

    def test_frontier_directed_sampling(self):
        frontier_pts = np.array([
            [0.02, 0.02, 0.05],
            [-0.02, 0.02, 0.05],
            [0.0, -0.02, 0.08],
        ])
        frontier_norms = np.array([
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ])
        candidates = self.sampler.sample_frontier_directed(
            frontier_pts, frontier_norms, num_candidates=30
        )
        assert len(candidates) > 0

    def test_combined_sampling(self):
        candidates = self.sampler.sample_combined(num_candidates=50)
        assert len(candidates) > 0

    def test_deterministic_with_seed(self):
        s1 = ViewpointSampler(
            self.object_center, self.workspace, rng=np.random.default_rng(123)
        )
        s2 = ViewpointSampler(
            self.object_center, self.workspace, rng=np.random.default_rng(123)
        )
        c1 = s1.sample_sphere(20)
        c2 = s2.sample_sphere(20)
        assert len(c1) == len(c2)
        for t1, t2 in zip(c1, c2):
            assert np.allclose(t1, t2)
