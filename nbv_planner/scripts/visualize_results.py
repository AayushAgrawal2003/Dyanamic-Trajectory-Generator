#!/usr/bin/env python3
"""Generate all visualizations from saved results."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from nbv_planner.metrics.evaluation import MethodEvaluation
from nbv_planner.visualization.metrics_viz import generate_all_plots


def load_evaluation(path: str) -> MethodEvaluation:
    """Load a saved result and create a MethodEvaluation."""
    data = np.load(path, allow_pickle=True)

    coverage_curve = data["coverage_curve"]
    new_points_curve = data["new_points_curve"]
    density_curve = data["density_curve"]
    n = len(coverage_curve)

    def cov_at(k: int) -> float:
        if k <= n:
            return float(coverage_curve[k - 1])
        return float(coverage_curve[-1]) if n > 0 else 0.0

    total_cov = float(coverage_curve[-1]) if n > 0 else 0.0
    traj_len = float(data["total_trajectory_length"])

    return MethodEvaluation(
        method_name=str(data["method_name"]),
        total_coverage=total_cov,
        mean_density_per_frame=float(density_curve.mean()) if n > 0 else 0.0,
        mean_new_points_per_frame=float(new_points_curve.mean()) if n > 0 else 0.0,
        coverage_at_5=cov_at(5),
        coverage_at_10=cov_at(10),
        coverage_at_15=cov_at(15),
        coverage_at_20=cov_at(20),
        trajectory_length=traj_len,
        efficiency=total_cov / max(traj_len, 1e-6),
        num_views=n,
        coverage_curve=coverage_curve,
        new_points_curve=new_points_curve,
        density_curve=density_curve,
    )


def main() -> None:
    results_base = Path(__file__).parent.parent / "results"
    scenes = ["sphere", "bunny", "femoral_surface"]

    for scene_name in scenes:
        print(f"\n--- {scene_name} ---")
        evaluations = []

        # Try to load each method's results
        method_files = [
            ("baseline", f"{scene_name}_arc.npz"),
            ("baseline", f"{scene_name}_raster.npz"),
            ("greedy", f"{scene_name}_greedy.npz"),
            ("sequence_opt", f"{scene_name}_seqopt.npz"),
        ]

        for method_dir, filename in method_files:
            path = results_base / method_dir / filename
            if path.exists():
                ev = load_evaluation(str(path))
                evaluations.append(ev)
                print(f"  Loaded: {ev.method_name} ({ev.total_coverage:.1%} coverage)")
            else:
                print(f"  Missing: {path}")

        if evaluations:
            plot_dir = str(results_base / "plots" / scene_name)
            generate_all_plots(evaluations, output_dir=plot_dir)

    print("\nVisualization complete.")


if __name__ == "__main__":
    main()
