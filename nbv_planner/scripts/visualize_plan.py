#!/usr/bin/env python3
"""Run a quick NBV plan and visualize the result interactively.

Shows: ground truth point cloud (green/red), camera frustums, and trajectory.
Uses matplotlib 3D (no Open3D dependency needed).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from nbv_planner.metrics.evaluation import evaluate_result
from nbv_planner.planner.greedy_nbv import GreedyNBVPlanner
from nbv_planner.robot.workspace import RobotWorkspace
from nbv_planner.scene.synthetic_scene import (
    create_bunny,
    create_femoral_surface,
    create_sphere,
)
from nbv_planner.sensor.camera_model import CameraModel


def draw_frustum(ax, T_cam_world, scale=0.02, color="red", alpha=0.7):
    """Draw a wireframe camera frustum."""
    T_world_cam = np.linalg.inv(T_cam_world)
    origin = T_world_cam[:3, 3]
    R = T_world_cam[:3, :3]

    corners_cam = np.array([
        [0, 0, 0],
        [-1, -0.75, 1.5],
        [1, -0.75, 1.5],
        [1, 0.75, 1.5],
        [-1, 0.75, 1.5],
    ]) * scale

    corners = (R @ corners_cam.T).T + origin

    edges = [
        [corners[0], corners[1]],
        [corners[0], corners[2]],
        [corners[0], corners[3]],
        [corners[0], corners[4]],
        [corners[1], corners[2]],
        [corners[2], corners[3]],
        [corners[3], corners[4]],
        [corners[4], corners[1]],
    ]
    lc = Line3DCollection(edges, colors=color, linewidths=1.2, alpha=alpha)
    ax.add_collection3d(lc)


def visualize_plan(scene_name: str = "sphere", max_views: int = 10) -> None:
    factories = {
        "sphere": create_sphere,
        "bunny": create_bunny,
        "femoral_surface": create_femoral_surface,
    }
    if scene_name not in factories:
        print(f"Unknown scene '{scene_name}'. Choose from: {list(factories.keys())}")
        return

    print(f"Creating {scene_name} scene...")
    scene = factories[scene_name](num_gt_points=20000)
    camera = CameraModel.default()
    workspace = RobotWorkspace.default(object_center=scene.center)

    print(f"Running greedy NBV planner ({max_views} views)...")
    planner = GreedyNBVPlanner(
        scene=scene, camera=camera, workspace=workspace,
        max_views=max_views, num_candidates=200,
    )
    result = planner.plan()
    ev = evaluate_result(result)

    print(f"\nFinal coverage: {ev.total_coverage:.1%} in {ev.num_views} views")
    print(f"Trajectory length: {ev.trajectory_length:.3f}m")

    # --- Build figure ---
    fig = plt.figure(figsize=(14, 6))

    # Left: 3D scene
    ax = fig.add_subplot(121, projection="3d")

    # Subsample points for plotting speed
    pts = scene.ground_truth_points
    mask = planner.gt_cloud._observed_mask
    stride = max(1, len(pts) // 5000)

    obs_pts = pts[mask][::stride]
    unobs_pts = pts[~mask][::stride]

    if len(obs_pts) > 0:
        ax.scatter(
            obs_pts[:, 0], obs_pts[:, 1], obs_pts[:, 2],
            c="mediumseagreen", s=1, alpha=0.4, label="Observed",
        )
    if len(unobs_pts) > 0:
        ax.scatter(
            unobs_pts[:, 0], unobs_pts[:, 1], unobs_pts[:, 2],
            c="tomato", s=2, alpha=0.6, label="Unobserved",
        )

    # Camera frustums (color gradient red -> green)
    n = len(result.poses)
    cam_positions = []
    for i, pose in enumerate(result.poses):
        t = i / max(1, n - 1)
        color = (1.0 - t, t, 0.3)
        draw_frustum(ax, pose, scale=0.02, color=color)
        T_world = np.linalg.inv(pose)
        cam_positions.append(T_world[:3, 3])

    cam_positions = np.array(cam_positions)

    # Trajectory line
    ax.plot(
        cam_positions[:, 0], cam_positions[:, 1], cam_positions[:, 2],
        "b-", linewidth=1.5, alpha=0.7, label="Trajectory",
    )
    # Number each viewpoint
    for i, p in enumerate(cam_positions):
        ax.text(p[0], p[1], p[2], str(i), fontsize=7, color="navy")

    # Robot base
    bp = workspace.base_position
    ax.scatter(*bp, c="blue", s=60, marker="^", label="Robot base", zorder=5)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"{scene_name} — {ev.total_coverage:.0%} coverage in {n} views")
    ax.legend(loc="upper left", fontsize=8)

    # Equal aspect ratio
    all_pts = np.vstack([pts[::stride], cam_positions])
    mid = all_pts.mean(axis=0)
    span = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2 * 1.1
    ax.set_xlim(mid[0] - span, mid[0] + span)
    ax.set_ylim(mid[1] - span, mid[1] + span)
    ax.set_zlim(mid[2] - span, mid[2] + span)

    # Right: coverage curve
    ax2 = fig.add_subplot(122)
    coverage = ev.coverage_curve * 100
    views = np.arange(1, len(coverage) + 1)
    ax2.plot(views, coverage, "o-", color="teal", linewidth=2, markersize=5)
    ax2.fill_between(views, coverage, alpha=0.15, color="teal")
    ax2.set_xlabel("View #")
    ax2.set_ylabel("Coverage (%)")
    ax2.set_title("Cumulative Coverage")
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)

    # Annotate new-points per frame
    ax2_twin = ax2.twinx()
    ax2_twin.bar(
        views, ev.new_points_curve, alpha=0.25, color="coral", label="New pts",
    )
    ax2_twin.set_ylabel("New Unique Points", color="coral")
    ax2_twin.tick_params(axis="y", labelcolor="coral")

    plt.tight_layout()
    print("\nShowing interactive matplotlib window (close to exit)...")
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize NBV planning result")
    parser.add_argument(
        "--scene", default="sphere",
        choices=["sphere", "bunny", "femoral_surface"],
        help="Which synthetic scene to use",
    )
    parser.add_argument(
        "--views", type=int, default=10,
        help="Number of views to plan",
    )
    args = parser.parse_args()
    visualize_plan(args.scene, args.views)
