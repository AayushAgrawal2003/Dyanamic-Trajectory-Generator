#!/usr/bin/env python3
"""Generate all synthetic scenes and save to data/."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nbv_planner.scene.synthetic_scene import ALL_SCENES, save_scene


def main() -> None:
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    for name, factory in ALL_SCENES.items():
        print(f"Generating scene: {name}...")
        scene = factory()
        save_scene(scene, data_dir)
        print(f"  Mesh bounds: {scene.bounding_box}")
        print(f"  Ground truth points: {scene.ground_truth_points.shape[0]}")
        print(f"  Saved to {data_dir}")
        print()

    print("All scenes generated successfully.")


if __name__ == "__main__":
    main()
