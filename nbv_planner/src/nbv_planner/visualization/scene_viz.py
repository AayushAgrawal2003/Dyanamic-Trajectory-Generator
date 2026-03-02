"""3D scene and viewpoint visualization using Open3D."""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

from nbv_planner.planner.greedy_nbv import PlanningResult
from nbv_planner.scene.synthetic_scene import SyntheticScene


def create_camera_frustum(
    T_cam_world: np.ndarray,
    scale: float = 0.03,
    color: tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> o3d.geometry.LineSet:
    """Create a wireframe camera frustum for visualization."""
    T_world_cam = np.linalg.inv(T_cam_world)
    origin = T_world_cam[:3, 3]
    R = T_world_cam[:3, :3]

    # Frustum corners in camera frame
    corners = np.array([
        [0, 0, 0],
        [-1, -0.75, 1.5],
        [1, -0.75, 1.5],
        [1, 0.75, 1.5],
        [-1, 0.75, 1.5],
    ], dtype=float) * scale

    # Transform to world frame
    corners_world = (R @ corners.T).T + origin

    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners_world)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return line_set


def visualize_scene(
    scene: SyntheticScene,
    result: PlanningResult | None = None,
    observed_points: np.ndarray | None = None,
    show: bool = True,
    save_path: str | None = None,
) -> None:
    """Visualize the scene with optional planned trajectory.

    Args:
        scene: The synthetic scene.
        result: Optional planning result with poses.
        observed_points: Optional observed point cloud.
        show: Whether to display the interactive viewer.
        save_path: Optional path to save a screenshot.
    """
    if not HAS_OPEN3D:
        print("Open3D not available. Skipping 3D visualization.")
        return

    geometries = []

    # Ground truth point cloud (gray)
    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = o3d.utility.Vector3dVector(scene.ground_truth_points)
    gt_pcd.paint_uniform_color([0.7, 0.7, 0.7])
    geometries.append(gt_pcd)

    if result is not None:
        # Camera frustums with color gradient
        n_poses = len(result.poses)
        for i, pose in enumerate(result.poses):
            t = i / max(1, n_poses - 1)
            color = (1.0 - t, t, 0.3)  # Red -> Green gradient
            frustum = create_camera_frustum(pose, color=color)
            geometries.append(frustum)

        # Trajectory line
        if n_poses > 1:
            positions = []
            for pose in result.poses:
                T_world = np.linalg.inv(pose)
                positions.append(T_world[:3, 3])
            positions = np.array(positions)

            lines = [[i, i + 1] for i in range(len(positions) - 1)]
            trajectory = o3d.geometry.LineSet()
            trajectory.points = o3d.utility.Vector3dVector(positions)
            trajectory.lines = o3d.utility.Vector2iVector(lines)
            trajectory.paint_uniform_color([0.0, 0.0, 1.0])
            geometries.append(trajectory)

    if observed_points is not None and len(observed_points) > 0:
        obs_pcd = o3d.geometry.PointCloud()
        obs_pcd.points = o3d.utility.Vector3dVector(observed_points)
        obs_pcd.paint_uniform_color([0.0, 0.8, 0.2])
        geometries.append(obs_pcd)

    # Coordinate frame at origin
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    geometries.append(coord_frame)

    if show:
        o3d.visualization.draw_geometries(
            geometries,
            window_name=f"NBV Scene: {scene.name}",
            width=1280,
            height=720,
        )

    if save_path:
        # Render offscreen
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=1280, height=720)
        for g in geometries:
            vis.add_geometry(g)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(save_path)
        vis.destroy_window()
        print(f"Saved screenshot to {save_path}")


def visualize_coverage_comparison(
    scene: SyntheticScene,
    gt_cloud_points: np.ndarray,
    observed_mask: np.ndarray,
    show: bool = True,
    save_path: str | None = None,
) -> None:
    """Visualize observed vs unobserved ground truth points.

    Observed points in green, unobserved in red.
    """
    if not HAS_OPEN3D:
        print("Open3D not available. Skipping coverage visualization.")
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(gt_cloud_points)

    colors = np.zeros((len(gt_cloud_points), 3))
    colors[observed_mask] = [0.0, 0.8, 0.2]  # Green = observed
    colors[~observed_mask] = [1.0, 0.0, 0.0]  # Red = unobserved
    pcd.colors = o3d.utility.Vector3dVector(colors)

    if show:
        o3d.visualization.draw_geometries(
            [pcd],
            window_name="Coverage Map",
            width=1280,
            height=720,
        )

    if save_path:
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=1280, height=720)
        vis.add_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(save_path)
        vis.destroy_window()
