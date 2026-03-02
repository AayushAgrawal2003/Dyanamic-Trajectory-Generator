#!/usr/bin/env python3
"""Compare all planning methods side-by-side."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nbv_planner.metrics.evaluation import (
    compare_methods,
    evaluate_result,
    save_results,
)
from nbv_planner.planner.baseline_fixed import FixedArcBaseline, RasterBaseline
from nbv_planner.planner.greedy_nbv import GreedyNBVPlanner
from nbv_planner.planner.sequence_optimizer import SequenceOptimizer
from nbv_planner.robot.workspace import RobotWorkspace
from nbv_planner.scene.synthetic_scene import ALL_SCENES
from nbv_planner.sensor.camera_model import CameraModel
from nbv_planner.visualization.metrics_viz import generate_all_plots


def main() -> None:
    camera = CameraModel.default()
    results_base = Path(__file__).parent.parent / "results"
    results_base.mkdir(parents=True, exist_ok=True)

    for scene_name, factory in ALL_SCENES.items():
        print(f"\n{'#'*70}")
        print(f"# Scene: {scene_name}")
        print(f"{'#'*70}")

        scene = factory()
        workspace = RobotWorkspace.default(object_center=scene.center)

        results = []

        # 1. Fixed arc baseline
        print("\n--- Fixed Arc ---")
        arc = FixedArcBaseline(scene, camera)
        results.append(arc.plan())

        # 2. Raster baseline
        print("\n--- Raster ---")
        raster = RasterBaseline(scene, camera)
        results.append(raster.plan())

        # 3. Greedy NBV
        print("\n--- Greedy NBV ---")
        greedy = GreedyNBVPlanner(scene=scene, camera=camera, workspace=workspace)
        results.append(greedy.plan())

        # 4. Sequence-optimized
        print("\n--- Sequence Optimized ---")
        seq_opt = SequenceOptimizer(scene=scene, camera=camera, workspace=workspace)
        results.append(seq_opt.plan())

        # Compare
        evaluations = compare_methods(results)

        # Generate plots
        plot_dir = str(results_base / "plots" / scene_name)
        generate_all_plots(evaluations, output_dir=plot_dir)

        # Save all results
        for r in results:
            save_results(
                r, str(results_base / r.method_name / f"{scene_name}.npz")
            )

    print("\n\nComparison complete. Check results/ for outputs.")


if __name__ == "__main__":
    main()
