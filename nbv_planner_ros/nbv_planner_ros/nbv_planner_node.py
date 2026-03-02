#!/usr/bin/env python3
"""ROS 2 node for Next-Best-View trajectory planning.

Loads a mesh file, runs NBV planning (greedy or sequence-optimized),
and publishes the resulting camera waypoints as geometry_msgs/PoseArray.
"""

import json
import time

import numpy as np
import open3d as o3d
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseArray
from std_msgs.msg import String
import trimesh

try:
    from nbv_planner.scene.synthetic_scene import SyntheticScene
    from nbv_planner.sensor.camera_model import CameraModel
    from nbv_planner.robot.workspace import ForbiddenZone, RobotWorkspace
    from nbv_planner.planner.greedy_nbv import GreedyNBVPlanner
    from nbv_planner.planner.sequence_optimizer import SequenceOptimizer
except ImportError as e:
    raise ImportError(
        "nbv_planner library not found. Install it with:\n"
        "  pip install -e /path/to/traj_optim/nbv_planner"
    ) from e

from nbv_planner_ros.pose_conversions import planning_result_to_pose_array


LATCHED_QOS = QoSProfile(
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    reliability=ReliabilityPolicy.RELIABLE,
)


class NbvPlannerNode(Node):
    def __init__(self):
        super().__init__("nbv_planner")

        self._declare_parameters()

        # Load scene from mesh file
        scene = self._load_scene()
        camera = self._create_camera_model()
        workspace = self._create_workspace(scene)

        # Set up publishers
        waypoints_topic = self.get_parameter("output.waypoints_topic").value
        coverage_topic = self.get_parameter("output.coverage_topic").value
        self._waypoints_pub = self.create_publisher(
            PoseArray, waypoints_topic, LATCHED_QOS
        )
        self._coverage_pub = self.create_publisher(
            String, coverage_topic, LATCHED_QOS
        )

        # Run planning
        self.get_logger().info("Starting NBV planning...")
        t0 = time.monotonic()
        result = self._run_planner(scene, camera, workspace)
        elapsed = time.monotonic() - t0

        # Convert poses and publish
        frame_id = self.get_parameter("output.frame_id").value
        stamp = self.get_clock().now().to_msg()
        pose_array = planning_result_to_pose_array(result.poses, frame_id, stamp)
        self._waypoints_pub.publish(pose_array)
        self._publish_coverage_info(result)

        # Save waypoints to .npy if path is provided
        save_path = self.get_parameter("output.save_npy_path").value
        if save_path:
            self._save_waypoints_npy(pose_array, result.poses, save_path)

        self.get_logger().info(
            f"Published {result.num_views} waypoints on '{waypoints_topic}' "
            f"(frame: {frame_id}). "
            f"Coverage: {result.final_coverage:.1%}, "
            f"Trajectory: {result.total_trajectory_length:.3f}m, "
            f"Planning time: {elapsed:.1f}s"
        )

    # ------------------------------------------------------------------
    # Parameter declaration
    # ------------------------------------------------------------------

    def _declare_parameters(self):
        # Scene
        self.declare_parameter("mesh_file_path", "")
        self.declare_parameter("points_file_path", "")
        self.declare_parameter("normals_file_path", "")
        self.declare_parameter("scene_name", "custom_scene")
        self.declare_parameter("num_sample_points", 50000)

        # Camera
        self.declare_parameter("camera.width", 640)
        self.declare_parameter("camera.height", 480)
        self.declare_parameter("camera.horizontal_fov_deg", 87.0)
        self.declare_parameter("camera.vertical_fov_deg", 58.0)
        self.declare_parameter("camera.min_depth", 0.1)
        self.declare_parameter("camera.max_depth", 1.0)
        self.declare_parameter("camera.depth_noise_coeff", 0.001)
        self.declare_parameter("camera.planning_subsample", 4)

        # Workspace
        self.declare_parameter("workspace.base_position", [0.0, -0.5, 0.3])
        self.declare_parameter("workspace.inner_radius", 0.3)
        self.declare_parameter("workspace.outer_radius", 0.8)
        self.declare_parameter("workspace.min_camera_angle_to_target", 0.5)
        self.declare_parameter("workspace.object_center", [0.0, 0.0, 0.0])
        self.declare_parameter("workspace.forbidden_zone_min", [-1.0, -1.0, -0.05])
        self.declare_parameter("workspace.forbidden_zone_max", [1.0, 1.0, 0.0])
        self.declare_parameter("workspace.use_forbidden_zone", True)

        # Planner
        self.declare_parameter("planner.method", "greedy")
        self.declare_parameter("planner.max_views", 20)
        self.declare_parameter("planner.num_candidates", 300)
        self.declare_parameter("planner.convergence_threshold", 0.01)
        self.declare_parameter("planner.coverage_weight", 1.0)
        self.declare_parameter("planner.density_weight", 0.1)
        self.declare_parameter("planner.neighbor_threshold", 0.003)
        self.declare_parameter("planner.min_angular_separation_deg", 15.0)
        self.declare_parameter("planner.random_seed", 42)

        # Output
        self.declare_parameter("output.waypoints_topic", "/nbv_waypoints")
        self.declare_parameter("output.coverage_topic", "/nbv_coverage")
        self.declare_parameter("output.frame_id", "base_link")
        self.declare_parameter("output.save_npy_path", "")

    # ------------------------------------------------------------------
    # Scene loading
    # ------------------------------------------------------------------

    def _load_scene(self) -> SyntheticScene:
        mesh_path = self.get_parameter("mesh_file_path").value
        points_path = self.get_parameter("points_file_path").value
        normals_path = self.get_parameter("normals_file_path").value
        scene_name = self.get_parameter("scene_name").value
        num_sample = self.get_parameter("num_sample_points").value

        if not mesh_path:
            self.get_logger().fatal(
                "Parameter 'mesh_file_path' is required. "
                "The planner needs a mesh for raycasting-based depth simulation."
            )
            raise ValueError("mesh_file_path is required")

        self.get_logger().info(f"Loading mesh from: {mesh_path}")
        mesh = self._load_mesh(mesh_path)

        self.get_logger().info(
            f"Mesh ready: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces"
        )

        if points_path:
            self.get_logger().info(f"Loading ground truth points from: {points_path}")
            points = np.load(points_path, allow_pickle=True)
            # Handle object arrays (e.g. saved with allow_pickle=True)
            if points.dtype == object:
                points = np.array(points.tolist(), dtype=np.float64)
            points = np.asarray(points, dtype=np.float64)
            if points.ndim == 1:
                points = points.reshape(-1, 3)
            if normals_path:
                normals = np.load(normals_path, allow_pickle=True)
                if normals.dtype == object:
                    normals = np.array(normals.tolist(), dtype=np.float64)
                normals = np.asarray(normals, dtype=np.float64)
            else:
                self.get_logger().info(
                    "No normals file provided, estimating from mesh faces."
                )
                closest_pts, distances, face_idx = \
                    trimesh.proximity.closest_point(mesh, points)

                # face_idx == -1 means the point had no valid closest face.
                bad_mask = face_idx < 0
                n_bad = int(bad_mask.sum())
                if n_bad > 0:
                    self.get_logger().warn(
                        f"{n_bad}/{len(points)} points had no matching mesh "
                        f"face (max dist: {distances[bad_mask].max():.4f}m). "
                        f"Using face 0 normals as fallback for those points."
                    )
                    face_idx[bad_mask] = 0

                normals = mesh.face_normals[face_idx]
            bbox = np.array([points.min(axis=0), points.max(axis=0)])
        else:
            self.get_logger().info(
                f"Sampling {num_sample} ground truth points from mesh surface."
            )
            points, face_indices = trimesh.sample.sample_surface(
                mesh, num_sample, seed=42
            )
            points = np.asarray(points)
            normals = mesh.face_normals[face_indices]
            bbox = np.array([mesh.bounds[0], mesh.bounds[1]])

        self.get_logger().info(
            f"Scene '{scene_name}': {len(points)} GT points, "
            f"bbox {bbox[0]} -> {bbox[1]}"
        )
        return SyntheticScene(
            name=scene_name,
            mesh=mesh,
            ground_truth_points=points,
            ground_truth_normals=normals,
            bounding_box=bbox,
        )

    def _load_mesh(self, mesh_path: str) -> trimesh.Trimesh:
        """Load a mesh from file, reconstructing from point cloud if needed.

        Handles three cases:
        1. Normal triangle mesh (.ply/.stl/.obj with faces) -> use directly
        2. Multi-body scene -> concatenate into single mesh
        3. Point cloud PLY (vertices only, no faces) -> reconstruct via
           Open3D Poisson surface reconstruction
        """
        loaded = trimesh.load(mesh_path, force="mesh")

        # Handle Scene with multiple geometries
        if isinstance(loaded, trimesh.Scene):
            meshes = [g for g in loaded.geometry.values()
                      if isinstance(g, trimesh.Trimesh) and len(g.faces) > 0]
            if meshes:
                mesh = trimesh.util.concatenate(meshes)
                self.get_logger().info(
                    f"Concatenated {len(meshes)} sub-meshes "
                    f"({len(mesh.faces)} faces)."
                )
                return mesh
            # Fall through to reconstruction if no faces found
            loaded = trimesh.Trimesh()

        # If we have a valid triangle mesh, return it
        if isinstance(loaded, trimesh.Trimesh) and len(loaded.faces) > 0:
            return loaded

        # ----------------------------------------------------------
        # The file is a point cloud (no faces). Reconstruct a mesh.
        # ----------------------------------------------------------
        self.get_logger().warn(
            "PLY file has no triangle faces — treating as point cloud. "
            "Reconstructing mesh via Poisson surface reconstruction..."
        )

        # Load as Open3D point cloud to get vertices (and colors if any)
        pcd = o3d.io.read_point_cloud(mesh_path)
        n_pts = len(pcd.points)
        if n_pts == 0:
            self.get_logger().fatal(f"File {mesh_path} contains no points.")
            raise ValueError("Empty point cloud")

        self.get_logger().info(f"Point cloud: {n_pts} points")

        # Estimate normals (required for Poisson reconstruction)
        if not pcd.has_normals():
            self.get_logger().info("Estimating normals...")
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.01, max_nn=30
                )
            )
            pcd.orient_normals_consistent_tangent_plane(k=15)

        # Poisson surface reconstruction
        self.get_logger().info("Running Poisson reconstruction (depth=8)...")
        o3d_mesh, densities = o3d.geometry.TriangleMesh \
            .create_from_point_cloud_poisson(pcd, depth=8)

        # Remove low-density vertices (reconstruction artifacts far from
        # the actual point cloud). Keep vertices above the 5th percentile.
        densities = np.asarray(densities)
        density_thresh = np.quantile(densities, 0.05)
        vertices_to_remove = densities < density_thresh
        o3d_mesh.remove_vertices_by_mask(vertices_to_remove)
        o3d_mesh.compute_vertex_normals()

        self.get_logger().info(
            f"Reconstructed mesh: {len(o3d_mesh.vertices)} vertices, "
            f"{len(o3d_mesh.triangles)} faces"
        )

        # Convert Open3D mesh -> trimesh
        mesh = trimesh.Trimesh(
            vertices=np.asarray(o3d_mesh.vertices),
            faces=np.asarray(o3d_mesh.triangles),
            vertex_normals=np.asarray(o3d_mesh.vertex_normals),
        )
        mesh.fix_normals()
        return mesh

    # ------------------------------------------------------------------
    # Component creation
    # ------------------------------------------------------------------

    def _create_camera_model(self) -> CameraModel:
        return CameraModel(
            width=self.get_parameter("camera.width").value,
            height=self.get_parameter("camera.height").value,
            horizontal_fov_deg=self.get_parameter("camera.horizontal_fov_deg").value,
            vertical_fov_deg=self.get_parameter("camera.vertical_fov_deg").value,
            min_depth=self.get_parameter("camera.min_depth").value,
            max_depth=self.get_parameter("camera.max_depth").value,
            depth_noise_coeff=self.get_parameter("camera.depth_noise_coeff").value,
            planning_subsample=self.get_parameter("camera.planning_subsample").value,
        )

    def _create_workspace(self, scene: SyntheticScene) -> RobotWorkspace:
        base_pos = np.array(self.get_parameter("workspace.base_position").value)
        obj_center = np.array(self.get_parameter("workspace.object_center").value)

        # If object_center is all zeros, use scene center
        if np.allclose(obj_center, 0.0):
            obj_center = scene.center
            self.get_logger().info(
                f"Using scene center as object_center: {obj_center}"
            )

        forbidden = []
        if self.get_parameter("workspace.use_forbidden_zone").value:
            fz_min = np.array(
                self.get_parameter("workspace.forbidden_zone_min").value
            )
            fz_max = np.array(
                self.get_parameter("workspace.forbidden_zone_max").value
            )
            forbidden.append(
                ForbiddenZone(min_corner=fz_min, max_corner=fz_max)
            )

        return RobotWorkspace(
            base_position=base_pos,
            inner_radius=self.get_parameter("workspace.inner_radius").value,
            outer_radius=self.get_parameter("workspace.outer_radius").value,
            min_camera_angle_to_target=self.get_parameter(
                "workspace.min_camera_angle_to_target"
            ).value,
            object_center=obj_center,
            forbidden_zones=forbidden,
        )

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def _run_planner(self, scene, camera, workspace):
        method = self.get_parameter("planner.method").value
        seed = self.get_parameter("planner.random_seed").value
        rng = np.random.default_rng(seed)

        common = dict(
            scene=scene,
            camera=camera,
            workspace=workspace,
            max_views=self.get_parameter("planner.max_views").value,
            num_candidates=self.get_parameter("planner.num_candidates").value,
            coverage_weight=self.get_parameter("planner.coverage_weight").value,
            density_weight=self.get_parameter("planner.density_weight").value,
            neighbor_threshold=self.get_parameter("planner.neighbor_threshold").value,
            rng=rng,
        )

        if method == "greedy":
            self.get_logger().info("Running greedy NBV planner...")
            planner = GreedyNBVPlanner(
                convergence_threshold=self.get_parameter(
                    "planner.convergence_threshold"
                ).value,
                **common,
            )
        elif method == "sequence_optimized":
            self.get_logger().info("Running sequence-optimized planner...")
            planner = SequenceOptimizer(
                min_angular_separation_deg=self.get_parameter(
                    "planner.min_angular_separation_deg"
                ).value,
                **common,
            )
        else:
            self.get_logger().fatal(
                f"Unknown planner.method: '{method}'. "
                f"Use 'greedy' or 'sequence_optimized'."
            )
            raise ValueError(f"Unknown planner method: {method}")

        return planner.plan()

    # ------------------------------------------------------------------
    # Coverage diagnostics
    # ------------------------------------------------------------------

    def _publish_coverage_info(self, result):
        data = {
            "method": result.method_name,
            "num_views": result.num_views,
            "total_trajectory_length": float(result.total_trajectory_length),
            "final_coverage": float(result.final_coverage),
            "per_frame": [
                {
                    "frame_index": fm.frame_index,
                    "total_points_in_frame": fm.total_points_in_frame,
                    "new_unique_points": fm.new_unique_points,
                    "cumulative_coverage": float(fm.cumulative_coverage),
                }
                for fm in result.frame_metrics
            ],
        }
        msg = String()
        msg.data = json.dumps(data, indent=2)
        self._coverage_pub.publish(msg)

    def _save_waypoints_npy(self, pose_array, t_cam_world_list, save_path):
        """Save waypoints to .npy file.

        Saves an (N, 7) array where each row is [x, y, z, qx, qy, qz, qw]
        representing the camera pose in the output frame (base_link).

        Also saves the raw 4x4 T_cam_world matrices as waypoints_4x4.npy
        alongside the main file for full-precision recovery.
        """
        from scipy.spatial.transform import Rotation

        n = len(pose_array.poses)
        waypoints = np.zeros((n, 7), dtype=np.float64)

        for i, pose in enumerate(pose_array.poses):
            waypoints[i, 0] = pose.position.x
            waypoints[i, 1] = pose.position.y
            waypoints[i, 2] = pose.position.z
            waypoints[i, 3] = pose.orientation.x
            waypoints[i, 4] = pose.orientation.y
            waypoints[i, 5] = pose.orientation.z
            waypoints[i, 6] = pose.orientation.w

        np.save(save_path, waypoints)
        self.get_logger().info(
            f"Saved {n} waypoints to {save_path}  "
            f"shape: ({n}, 7) = [x, y, z, qx, qy, qz, qw]"
        )

        # Also save raw 4x4 poses for full-precision round-trip
        raw_path = save_path.replace(".npy", "_4x4.npy")
        poses_4x4 = np.stack(t_cam_world_list, axis=0)  # (N, 4, 4)
        np.save(raw_path, poses_4x4)
        self.get_logger().info(
            f"Saved raw T_cam_world matrices to {raw_path}  "
            f"shape: {poses_4x4.shape}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = NbvPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
