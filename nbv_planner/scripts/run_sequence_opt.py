#!/usr/bin/env python3
"""Run the sequence-optimized NBV planner on each synthetic object."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nbv_planner.metrics.evaluation import evaluate_result, save_results
from nbv_planner.planner.sequence_optimizer import SequenceOptimizer
from nbv_planner.robot.workspace import RobotWorkspace
from nbv_planner.scene.synthetic_scene import ALL_SCENES
from nbv_planner.sensor.camera_model import CameraModel


def main() -> None:
    camera = CameraModel.default()
    results_dir = Path(__file__).parent.parent / "results" / "sequence_opt"
    results_dir.mkdir(parents=True, exist_ok=True)

    for scene_name, factory in ALL_SCENES.items():
        print(f"\n{'='*60}")
        print(f"Scene: {scene_name}")
        print(f"{'='*60}")

        scene = factory()
        workspace = RobotWorkspace.default(object_center=scene.center)

        optimizer = SequenceOptimizer(
            scene=scene,
            camera=camera,
            workspace=workspace,
        )
        result = optimizer.plan()

        ev = evaluate_result(result)
        print(f"\nFinal coverage: {ev.total_coverage:.1%}")
        print(f"Trajectory length: {ev.trajectory_length:.3f}m")
        print(f"Coverage @5: {ev.coverage_at_5:.1%}")
        print(f"Coverage @10: {ev.coverage_at_10:.1%}")

        save_results(result, str(results_dir / f"{scene_name}_seqopt.npz"))

    print("\nAll sequence-optimized results saved.")


if __name__ == "__main__":
    main()
