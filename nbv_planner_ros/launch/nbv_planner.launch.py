"""Launch file for nbv_planner_ros.

Usage:
  # Set mesh_file_path in your config YAML, then:
  ros2 launch nbv_planner_ros nbv_planner.launch.py \
      config_file:=/path/to/my_params.yaml

  # Or use the default config and override mesh path:
  ros2 launch nbv_planner_ros nbv_planner.launch.py \
      --ros-args -p mesh_file_path:=/path/to/object.ply

  # Override any parameter via CLI:
  ros2 launch nbv_planner_ros nbv_planner.launch.py \
      config_file:=/path/to/my_params.yaml \
      --ros-args -p planner.method:=sequence_optimized
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("nbv_planner_ros")
    default_config = os.path.join(pkg_share, "config", "default_params.yaml")

    return LaunchDescription([
        DeclareLaunchArgument(
            "config_file",
            default_value=default_config,
            description="Path to the YAML parameter file",
        ),
        Node(
            package="nbv_planner_ros",
            executable="nbv_planner_node",
            name="nbv_planner",
            output="screen",
            parameters=[LaunchConfiguration("config_file")],
        ),
    ])
