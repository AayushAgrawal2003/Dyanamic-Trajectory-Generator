#!/usr/bin/env python3
"""ROS 2 node for Next-Best-View trajectory planning.

Loads a mesh file, runs NBV planning (greedy or sequence-optimized),
and publishes the resulting camera waypoints as geometry_msgs/PoseArray.
"""

import json
import time

import numpy as np
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
        loaded = trimesh.load(mesh_path, force="mesh")

        # trimesh.load with force="mesh" can still return a Scene if the
        # file contains multiple meshes. Concatenate into a single Trimesh.
        if isinstance(loaded, trimesh.Scene):
            meshes = [g for g in loaded.geometry.values()
                      if isinstance(g, trimesh.Trimesh)]
            if not meshes:
                self.get_logger().fatal(
                    f"No triangle geometry found in {mesh_path}"
                )
                raise ValueError("Mesh file contains no triangles")
            mesh = trimesh.util.concatenate(meshes)
            self.get_logger().info(
                f"Concatenated {len(meshes)} sub-meshes into one "
                f"({len(mesh.faces)} faces)."
            )
        else:
            mesh = loaded

        self.get_logger().info(
            f"Mesh loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces"
        )

        if points_path:
            self.get_logger().info(f"Loading ground truth points from: {points_path}")
            points = np.load(points_path)
            if normals_path:
                normals = np.load(normals_path)
            else:
                self.get_logger().info(
                    "No normals file provided, estimating from mesh faces."
                )
                closest_pts, distances, face_idx = \
                    trimesh.proximity.closest_point(mesh, points)

                # face_idx == -1 means the point had no valid closest face.
                # This happens when points are far from the mesh or the mesh
                # has degenerate faces. Replace bad indices with 0 and log.
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
