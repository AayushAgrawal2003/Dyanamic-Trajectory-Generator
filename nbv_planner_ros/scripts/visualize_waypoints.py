#!/usr/bin/env python3
"""Visualize NBV waypoints from a .npy file alongside the target point cloud/mesh.

Usage:
    python3 visualize_waypoints.py waypoints.npy
    python3 visualize_waypoints.py waypoints.npy --mesh object.ply
    python3 visualize_waypoints.py waypoints.npy --mesh object.ply --frustum-scale 0.05
    python3 visualize_waypoints.py waypoints_4x4.npy --raw4x4
"""

import argparse
import sys

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation


def load_waypoints_7(path: str) -> np.ndarray:
    """Load (N,7) waypoints [x, y, z, qx, qy, qz, qw] -> list of 4x4."""
    data = np.load(path)
    assert data.ndim == 2 and data.shape[1] == 7, \
        f"Expected (N,7), got {data.shape}"
    poses = []
    for row in data:
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(row[3:7]).as_matrix()
        T[:3, 3] = row[:3]
        poses.append(T)
    return poses


def load_waypoints_4x4(path: str) -> list:
    """Load (N,4,4) raw T_cam_world matrices -> invert to T_world_cam."""
    data = np.load(path)
    assert data.ndim == 3 and data.shape[1:] == (4, 4), \
        f"Expected (N,4,4), got {data.shape}"
    return [np.linalg.inv(T) for T in data]


def create_camera_frustum(T_world_cam: np.ndarray, scale: float = 0.04,
                          color: list = None) -> o3d.geometry.LineSet:
    """Create a wireframe camera frustum at the given pose."""
    # Frustum corners in camera frame (Z-forward, X-right, Y-down)
    s = scale
    pts_cam = np.array([
        [0, 0, 0],           # camera center
        [-s, -s * 0.75, 2 * s],  # top-left
        [s, -s * 0.75, 2 * s],   # top-right
        [s, s * 0.75, 2 * s],    # bottom-right
        [-s, s * 0.75, 2 * s],   # bottom-left
    ])

    # Transform to world frame
    R = T_world_cam[:3, :3]
    t = T_world_cam[:3, 3]
    pts_world = (R @ pts_cam.T).T + t

    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],  # edges from center
        [1, 2], [2, 3], [3, 4], [4, 1],  # frame rectangle
    ]

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts_world)
    ls.lines = o3d.utility.Vector2iVector(lines)
    if color:
        ls.paint_uniform_color(color)
    return ls


def create_trajectory_line(poses: list, color: list = None) -> o3d.geometry.LineSet:
    """Create a line connecting waypoint positions in order."""
    positions = np.array([T[:3, 3] for T in poses])
    lines = [[i, i + 1] for i in range(len(positions) - 1)]

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(positions)
    ls.lines = o3d.utility.Vector2iVector(lines)
    if color:
        ls.paint_uniform_color(color)
    return ls


def main():
    parser = argparse.ArgumentParser(
        description="Visualize NBV waypoints with Open3D"
    )
    parser.add_argument("waypoints_npy", help="Path to waypoints .npy file")
    parser.add_argument("--mesh", "-m", default="",
                        help="Path to target mesh/point cloud .ply file")
    parser.add_argument("--raw4x4", action="store_true",
                        help="Waypoints are (N,4,4) T_cam_world matrices")
    parser.add_argument("--frustum-scale", type=float, default=0.04,
                        help="Camera frustum size (default: 0.04)")
    parser.add_argument("--no-numbers", action="store_true",
                        help="Don't print waypoint numbering to console")
    args = parser.parse_args()

    # Load waypoints
    if args.raw4x4:
        poses = load_waypoints_4x4(args.waypoints_npy)
        print(f"Loaded {len(poses)} waypoints (raw 4x4 T_cam_world, inverted)")
    else:
        poses = load_waypoints_7(args.waypoints_npy)
        print(f"Loaded {len(poses)} waypoints (x,y,z,qx,qy,qz,qw)")

    if not args.no_numbers:
        print("\nWaypoint positions (in base_link frame):")
        for i, T in enumerate(poses):
            pos = T[:3, 3]
            print(f"  [{i:2d}] x={pos[0]:+.4f}  y={pos[1]:+.4f}  z={pos[2]:+.4f}")

    # Build visualization
    geometries = []

    # Coordinate frame at origin
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    geometries.append(axes)

    # Load mesh / point cloud if provided
    if args.mesh:
        print(f"\nLoading mesh/cloud: {args.mesh}")
        # Try as mesh first, fall back to point cloud
        mesh = o3d.io.read_triangle_mesh(args.mesh)
        if len(mesh.triangles) > 0:
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color([0.7, 0.7, 0.7])
            geometries.append(mesh)
            print(f"  Mesh: {len(mesh.vertices)} vertices, "
                  f"{len(mesh.triangles)} triangles")
        else:
            pcd = o3d.io.read_point_cloud(args.mesh)
            if len(pcd.points) > 0:
                if not pcd.has_colors():
                    pcd.paint_uniform_color([0.5, 0.5, 0.5])
                geometries.append(pcd)
                print(f"  Point cloud: {len(pcd.points)} points")
            else:
                print(f"  WARNING: {args.mesh} has no geometry")

    # Camera frustums with color gradient (green -> red)
    n = len(poses)
    for i, T in enumerate(poses):
        t = i / max(n - 1, 1)
        color = [t, 1.0 - t, 0.2]  # green at start, red at end
        frustum = create_camera_frustum(T, scale=args.frustum_scale, color=color)
        geometries.append(frustum)

        # Small sphere at camera position
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)
        sphere.translate(T[:3, 3])
        sphere.paint_uniform_color(color)
        geometries.append(sphere)

    # Trajectory line
    traj = create_trajectory_line(poses, color=[1.0, 0.5, 0.0])
    geometries.append(traj)

    # Draw
    print(f"\nShowing {n} waypoints. Close the window to exit.")
    print("  Green frustum = first waypoint")
    print("  Red frustum = last waypoint")
    print("  Orange line = trajectory path")
    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"NBV Waypoints ({n} poses)",
        width=1280, height=720,
    )


if __name__ == "__main__":
    main()
