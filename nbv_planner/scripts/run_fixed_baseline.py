#!/usr/bin/env python3
"""Run the fixed trajectory baselines on each synthetic object."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nbv_planner.metrics.evaluation import evaluate_result, save_results
from nbv_planner.planner.baseline_fixed import FixedArcBaseline, RasterBaseline
from nbv_planner.scene.synthetic_scene import ALL_SCENES
from nbv_planner.sensor.camera_model import CameraModel


def main() -> None:
    camera = CameraModel.default()
    results_dir = Path(__file__).parent.parent / "results" / "baseline"
    results_dir.mkdir(parents=True, exist_ok=True)

    for scene_name, factory in ALL_SCENES.items():
        print(f"\n{'='*60}")
        print(f"Scene: {scene_name}")
        print(f"{'='*60}")

        scene = factory()

        # Fixed arc baseline
        print("\n--- Fixed Arc Baseline ---")
        arc = FixedArcBaseline(scene, camera)
        arc_result = arc.plan()
        arc_eval = evaluate_result(arc_result)
        print(f"Final coverage: {arc_eval.total_coverage:.1%}")
        print(f"Trajectory length: {arc_eval.trajectory_length:.3f}m")
        save_results(arc_result, str(results_dir / f"{scene_name}_arc.npz"))

        # Raster baseline
        print("\n--- Raster Baseline ---")
        raster = RasterBaseline(scene, camera)
        raster_result = raster.plan()
        raster_eval = evaluate_result(raster_result)
        print(f"Final coverage: {raster_eval.total_coverage:.1%}")
        print(f"Trajectory length: {raster_eval.trajectory_length:.3f}m")
        save_results(raster_result, str(results_dir / f"{scene_name}_raster.npz"))

    print("\nAll baseline results saved.")


if __name__ == "__main__":
    main()
