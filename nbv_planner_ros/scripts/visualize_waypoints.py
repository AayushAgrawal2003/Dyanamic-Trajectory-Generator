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
import matplotlib
matplotlib.use("TkAgg")  # force interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from scipy.spatial.transform import Rotation


def load_waypoints_7(path: str) -> list:
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


def load_ply_points(path: str) -> np.ndarray:
    """Load points from a PLY file using trimesh or open3d."""
    try:
        import open3d as o3d
        # Try as point cloud first
        pcd = o3d.io.read_point_cloud(path)
        pts = np.asarray(pcd.points)
        if len(pts) > 0:
            return pts
        # Try as mesh
        mesh = o3d.io.read_triangle_mesh(path)
        return np.asarray(mesh.vertices)
    except ImportError:
        pass

    try:
        import trimesh
        loaded = trimesh.load(path)
        if hasattr(loaded, 'vertices'):
            return np.asarray(loaded.vertices)
    except ImportError:
        pass

    print(f"WARNING: Cannot load {path} (install open3d or trimesh)")
    return np.zeros((0, 3))


def draw_frustum(ax, T_world_cam, scale=0.04, color='green', alpha=0.6):
    """Draw a camera frustum on a matplotlib 3D axis."""
    s = scale
    pts_cam = np.array([
        [0, 0, 0],              # camera center
        [-s, -s * 0.75, 2 * s], # top-left
        [ s, -s * 0.75, 2 * s], # top-right
        [ s,  s * 0.75, 2 * s], # bottom-right
        [-s,  s * 0.75, 2 * s], # bottom-left
    ])

    R = T_world_cam[:3, :3]
    t = T_world_cam[:3, 3]
    pts = (R @ pts_cam.T).T + t

    # Edges from center to corners
    for i in range(1, 5):
        ax.plot3D(*zip(pts[0], pts[i]), color=color, linewidth=1.0, alpha=alpha)

    # Rectangle
    rect = [pts[1], pts[2], pts[3], pts[4], pts[1]]
    for j in range(4):
        ax.plot3D(*zip(rect[j], rect[j + 1]), color=color, linewidth=1.0, alpha=alpha)

    # Draw Z-axis arrow (camera look direction)
    look_len = 2.5 * s
    look_end = t + R[:, 2] * look_len
    ax.plot3D(*zip(t, look_end), color=color, linewidth=0.5, alpha=0.4)


def set_equal_aspect(ax, points):
    """Set equal aspect ratio for 3D plot based on data bounds."""
    if len(points) == 0:
        return
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2
    max_range = (maxs - mins).max() / 2
    if max_range < 1e-6:
        max_range = 1.0
    # Add 20% padding
    max_range *= 1.2
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize NBV waypoints with matplotlib"
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
    parser.add_argument("--subsample", type=int, default=5000,
                        help="Max points to display from mesh (default: 5000)")
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

    # Extract positions
    positions = np.array([T[:3, 3] for T in poses])

    # ── Plot ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    all_plot_points = [positions]

    # Load and plot mesh points if provided
    if args.mesh:
        print(f"\nLoading mesh/cloud: {args.mesh}")
        mesh_pts = load_ply_points(args.mesh)
        if len(mesh_pts) > 0:
            print(f"  {len(mesh_pts)} points loaded")
            # Subsample for plotting speed
            if len(mesh_pts) > args.subsample:
                idx = np.random.default_rng(0).choice(
                    len(mesh_pts), args.subsample, replace=False
                )
                mesh_pts_plot = mesh_pts[idx]
            else:
                mesh_pts_plot = mesh_pts
            ax.scatter(
                mesh_pts_plot[:, 0], mesh_pts_plot[:, 1], mesh_pts_plot[:, 2],
                c='gray', s=0.3, alpha=0.3, label='Object'
            )
            all_plot_points.append(mesh_pts_plot)
        else:
            print(f"  WARNING: no geometry in {args.mesh}")

    # Color gradient: green -> red
    n = len(poses)
    colors = []
    for i in range(n):
        t = i / max(n - 1, 1)
        colors.append((t, 1.0 - t, 0.2))

    # Draw frustums
    for i, T in enumerate(poses):
        draw_frustum(ax, T, scale=args.frustum_scale, color=colors[i])

    # Camera position spheres
    ax.scatter(
        positions[:, 0], positions[:, 1], positions[:, 2],
        c=colors, s=40, edgecolors='black', linewidths=0.5,
        zorder=5, label='Waypoints'
    )

    # Waypoint index labels
    for i, pos in enumerate(positions):
        ax.text(pos[0], pos[1], pos[2], f' {i}', fontsize=7,
                color='black', ha='left', va='bottom')

    # Trajectory line
    ax.plot(
        positions[:, 0], positions[:, 1], positions[:, 2],
        color='orange', linewidth=2.0, alpha=0.8, label='Trajectory'
    )

    # Origin coordinate frame
    origin = np.zeros(3)
    axis_len = 0.08
    ax.quiver(*origin, axis_len, 0, 0, color='red', arrow_length_ratio=0.15, linewidth=2)
    ax.quiver(*origin, 0, axis_len, 0, color='green', arrow_length_ratio=0.15, linewidth=2)
    ax.quiver(*origin, 0, 0, axis_len, color='blue', arrow_length_ratio=0.15, linewidth=2)

    # Equal aspect ratio
    all_pts = np.vstack(all_plot_points)
    set_equal_aspect(ax, all_pts)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'NBV Waypoints ({n} poses)\n'
                 f'Green = first, Red = last, Orange = trajectory')
    ax.legend(loc='upper left', fontsize=8)

    print(f"\nShowing {n} waypoints. Close the window to exit.")
    print("  Green = first waypoint, Red = last waypoint")
    print("  Orange line = trajectory path")
    print("  Drag to rotate, scroll to zoom.")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
