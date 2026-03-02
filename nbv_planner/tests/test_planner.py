"""Integration tests for the planners."""

import numpy as np
import pytest

from nbv_planner.metrics.evaluation import evaluate_result
from nbv_planner.planner.baseline_fixed import FixedArcBaseline, RasterBaseline
from nbv_planner.planner.greedy_nbv import GreedyNBVPlanner
from nbv_planner.planner.sequence_optimizer import SequenceOptimizer
from nbv_planner.robot.workspace import RobotWorkspace
from nbv_planner.scene.synthetic_scene import create_sphere
from nbv_planner.sensor.camera_model import CameraModel


@pytest.fixture
def sphere_scene():
    return create_sphere(radius=0.05, num_gt_points=5000, seed=42)


@pytest.fixture
def camera():
    return CameraModel.default()


@pytest.fixture
def workspace(sphere_scene):
    return RobotWorkspace.default(object_center=sphere_scene.center)


class TestFixedArcBaseline:
    def test_runs_and_produces_result(self, sphere_scene, camera):
        baseline = FixedArcBaseline(
            sphere_scene, camera, num_views=5, rng=np.random.default_rng(42)
        )
        result = baseline.plan()
        assert result.num_views == 5
        assert result.final_coverage > 0

    def test_evaluation(self, sphere_scene, camera):
        baseline = FixedArcBaseline(
            sphere_scene, camera, num_views=5, rng=np.random.default_rng(42)
        )
        result = baseline.plan()
        ev = evaluate_result(result)
        assert 0 <= ev.total_coverage <= 1.0
        assert ev.trajectory_length >= 0


class TestRasterBaseline:
    def test_runs_and_produces_result(self, sphere_scene, camera):
        baseline = RasterBaseline(
            sphere_scene, camera, num_views=5, rng=np.random.default_rng(42)
        )
        result = baseline.plan()
        assert result.num_views == 5
        assert result.final_coverage > 0


class TestGreedyNBV:
    def test_runs_and_produces_result(self, sphere_scene, camera, workspace):
        planner = GreedyNBVPlanner(
            scene=sphere_scene,
            camera=camera,
            workspace=workspace,
            max_views=5,
            num_candidates=50,
            rng=np.random.default_rng(42),
        )
        result = planner.plan()
        assert result.num_views >= 1
        assert result.final_coverage > 0

    def test_coverage_increases(self, sphere_scene, camera, workspace):
        planner = GreedyNBVPlanner(
            scene=sphere_scene,
            camera=camera,
            workspace=workspace,
            max_views=5,
            num_candidates=50,
            rng=np.random.default_rng(42),
        )
        result = planner.plan()
        coverages = [f.cumulative_coverage for f in result.frame_metrics]
        # Coverage should be non-decreasing
        for i in range(1, len(coverages)):
            assert coverages[i] >= coverages[i - 1] - 1e-6


class TestSequenceOptimizer:
    def test_runs_and_produces_result(self, sphere_scene, camera, workspace):
        optimizer = SequenceOptimizer(
            scene=sphere_scene,
            camera=camera,
            workspace=workspace,
            max_views=5,
            num_candidates=50,
            rng=np.random.default_rng(42),
        )
        result = optimizer.plan()
        assert result.num_views >= 1
        assert result.final_coverage > 0
