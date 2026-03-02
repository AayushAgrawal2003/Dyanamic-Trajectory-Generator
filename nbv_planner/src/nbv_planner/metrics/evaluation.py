"""Evaluation metrics for NBV planning runs."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nbv_planner.planner.greedy_nbv import PlanningResult


@dataclass
class MethodEvaluation:
    """Comprehensive evaluation of a planning method."""

    method_name: str
    total_coverage: float
    mean_density_per_frame: float
    mean_new_points_per_frame: float
    coverage_at_5: float
    coverage_at_10: float
    coverage_at_15: float
    coverage_at_20: float
    trajectory_length: float
    efficiency: float  # coverage / trajectory_length
    num_views: int
    coverage_curve: np.ndarray
    new_points_curve: np.ndarray
    density_curve: np.ndarray


def evaluate_result(result: PlanningResult) -> MethodEvaluation:
    """Compute comprehensive metrics from a PlanningResult."""
    n = len(result.frame_metrics)

    coverage_curve = np.array([f.cumulative_coverage for f in result.frame_metrics])
    new_points_curve = np.array([f.new_unique_points for f in result.frame_metrics])
    density_curve = np.array([f.total_points_in_frame for f in result.frame_metrics])

    # Coverage at specific view counts
    def cov_at(k: int) -> float:
        if k <= n:
            return float(coverage_curve[k - 1])
        return float(coverage_curve[-1]) if n > 0 else 0.0

    total_coverage = float(coverage_curve[-1]) if n > 0 else 0.0
    traj_len = result.total_trajectory_length

    return MethodEvaluation(
        method_name=result.method_name,
        total_coverage=total_coverage,
        mean_density_per_frame=float(density_curve.mean()) if n > 0 else 0.0,
        mean_new_points_per_frame=float(new_points_curve.mean()) if n > 0 else 0.0,
        coverage_at_5=cov_at(5),
        coverage_at_10=cov_at(10),
        coverage_at_15=cov_at(15),
        coverage_at_20=cov_at(20),
        trajectory_length=traj_len,
        efficiency=total_coverage / max(traj_len, 1e-6),
        num_views=n,
        coverage_curve=coverage_curve,
        new_points_curve=new_points_curve,
        density_curve=density_curve,
    )


def compare_methods(
    results: list[PlanningResult],
) -> list[MethodEvaluation]:
    """Evaluate and compare multiple planning methods."""
    evaluations = [evaluate_result(r) for r in results]

    # Print comparison table
    print("\n" + "=" * 80)
    print("METHOD COMPARISON")
    print("=" * 80)
    header = (
        f"{'Method':<20} {'Coverage':>10} {'@5':>8} {'@10':>8} "
        f"{'@15':>8} {'@20':>8} {'TrajLen':>10} {'Eff':>8}"
    )
    print(header)
    print("-" * 80)

    for e in evaluations:
        row = (
            f"{e.method_name:<20} {e.total_coverage:>9.1%} "
            f"{e.coverage_at_5:>7.1%} {e.coverage_at_10:>7.1%} "
            f"{e.coverage_at_15:>7.1%} {e.coverage_at_20:>7.1%} "
            f"{e.trajectory_length:>9.3f}m {e.efficiency:>7.1f}"
        )
        print(row)

    print("=" * 80)

    return evaluations


def save_results(result: PlanningResult, path: str) -> None:
    """Save planning results to a .npz file."""
    poses_array = np.array([p for p in result.poses])
    coverage_curve = np.array(
        [f.cumulative_coverage for f in result.frame_metrics]
    )
    new_points_curve = np.array(
        [f.new_unique_points for f in result.frame_metrics]
    )
    density_curve = np.array(
        [f.total_points_in_frame for f in result.frame_metrics]
    )

    np.savez(
        path,
        method_name=result.method_name,
        poses=poses_array,
        coverage_curve=coverage_curve,
        new_points_curve=new_points_curve,
        density_curve=density_curve,
        total_trajectory_length=result.total_trajectory_length,
    )


def load_results_for_comparison(path: str) -> dict:
    """Load saved results for comparison."""
    data = np.load(path, allow_pickle=True)
    return {
        "method_name": str(data["method_name"]),
        "poses": data["poses"],
        "coverage_curve": data["coverage_curve"],
        "new_points_curve": data["new_points_curve"],
        "density_curve": data["density_curve"],
        "total_trajectory_length": float(data["total_trajectory_length"]),
    }
