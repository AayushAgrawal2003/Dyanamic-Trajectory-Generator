# nbv_planner_ros

ROS 2 Humble package for Next-Best-View (NBV) trajectory optimization on the KUKA LBR Med 7. Takes a target object mesh (`.ply`) and/or point cloud (`.npy`), runs viewpoint planning, and publishes optimized camera waypoints as `geometry_msgs/PoseArray`.

---

## Prerequisites

- ROS 2 Humble
- Python 3.9+
- The `nbv_planner` library (the core planning engine)

### Python dependencies (installed with `nbv_planner`)

```
numpy >= 1.24
scipy >= 1.10
trimesh >= 4.0
open3d >= 0.17
pyyaml >= 6.0
```

---

## Setup on a Fresh Linux System (Ubuntu 22.04 + ROS 2 Humble)

### 0. Make sure ROS 2 Humble is installed

```bash
# Verify:
source /opt/ros/humble/setup.bash
ros2 --version
```

If not installed, follow the [official install guide](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html).

### 1. Clone the repo

```bash
cd ~
git clone <your-repo-url> traj_optim
```

Your repo should look like:
```
traj_optim/
├── nbv_planner/          # Python library
│   ├── pyproject.toml
│   ├── src/nbv_planner/
│   └── data/             # test meshes
├── nbv_planner_ros/      # ROS 2 package
│   ├── package.xml
│   └── ...
└── TECHNICAL_DETAILS.md
```

### 2. Install the Python planning library

```bash
cd ~/traj_optim/nbv_planner
pip install -e .
```

This installs all Python dependencies (numpy, scipy, trimesh, open3d, etc.) and makes `import nbv_planner` available to the ROS node.

### 3. Create a ROS 2 workspace and symlink the package

```bash
mkdir -p ~/ros2_ws/src
ln -s ~/traj_optim/nbv_planner_ros ~/ros2_ws/src/nbv_planner_ros
```

### 4. Build

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select nbv_planner_ros
source install/setup.bash
```

### 5. Test it works

```bash
# Run with the bundled test mesh:
ros2 run nbv_planner_ros nbv_planner_node \
    --ros-args -p mesh_file_path:=$HOME/traj_optim/nbv_planner/data/bunny_mesh.ply

# In another terminal:
source ~/ros2_ws/install/setup.bash
ros2 topic echo /nbv_waypoints --qos-durability transient_local
```

You should see a PoseArray with ~15-20 waypoints printed out.

> **Tip:** Add `source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash` to your `~/.bashrc` so you don't have to source every time.

---

## Quick Start

### Minimal run (node directly)

```bash
ros2 run nbv_planner_ros nbv_planner_node \
    --ros-args -p mesh_file_path:=/path/to/your_object.ply
```

### With launch file + config

```bash
# Copy and edit the default config
cp $(ros2 pkg prefix nbv_planner_ros)/share/nbv_planner_ros/config/default_params.yaml ~/my_params.yaml
# Edit ~/my_params.yaml: set mesh_file_path, camera params, workspace params, etc.

ros2 launch nbv_planner_ros nbv_planner.launch.py config_file:=$HOME/my_params.yaml
```

### Override individual parameters on the CLI

```bash
ros2 run nbv_planner_ros nbv_planner_node --ros-args \
    -p mesh_file_path:=/path/to/object.ply \
    -p planner.method:=sequence_optimized \
    -p planner.max_views:=15 \
    -p camera.horizontal_fov_deg:=69.0 \
    -p camera.vertical_fov_deg:=42.0 \
    -p workspace.base_position:="[0.0, -0.4, 0.2]" \
    -p output.frame_id:=base_link
```

---

## Reading the Output

The node publishes once with **transient_local** (latched) QoS, then spins to keep the message available for late subscribers.

### Waypoints

```bash
# In a separate terminal:
ros2 topic echo /nbv_waypoints --qos-durability transient_local
```

Returns a `geometry_msgs/PoseArray` in `base_link` frame. Each pose is the camera position and orientation (quaternion) in world coordinates. The poses are ordered — execute them sequentially for the planned trajectory.

### Coverage diagnostics

```bash
ros2 topic echo /nbv_coverage --qos-durability transient_local
```

Returns a `std_msgs/String` with JSON:

```json
{
  "method": "greedy_nbv",
  "num_views": 18,
  "total_trajectory_length": 1.234,
  "final_coverage": 0.923,
  "per_frame": [
    {"frame_index": 0, "total_points_in_frame": 4521, "new_unique_points": 4521, "cumulative_coverage": 0.09},
    {"frame_index": 1, "total_points_in_frame": 3892, "new_unique_points": 3102, "cumulative_coverage": 0.152},
    ...
  ]
}
```

---

## Feeding Waypoints to MoveIt

The PoseArray on `/nbv_waypoints` contains camera poses in the `base_link` frame. To use with MoveIt:

1. **Subscribe** to `/nbv_waypoints` (use `transient_local` durability QoS to receive the latched message).
2. **Iterate** through the poses in order.
3. **Plan + execute** to each pose using `moveit_commander` or the MoveIt C++ API.

Example subscriber snippet:

```python
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from geometry_msgs.msg import PoseArray

qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                  reliability=ReliabilityPolicy.RELIABLE)

self.create_subscription(PoseArray, '/nbv_waypoints', self.waypoints_cb, qos)
```

**Camera frame convention**: poses use the optical frame convention (Z-forward, X-right, Y-down). If your MoveIt end-effector frame differs, apply your camera-to-EE static transform (defined in your URDF/XACRO).

---

## Parameters Reference

All parameters live under the `nbv_planner` node namespace. Set them via YAML config or `--ros-args -p`.

### Scene (input data)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `mesh_file_path` | string | `""` | **Required.** Path to `.ply` mesh file. |
| `points_file_path` | string | `""` | Optional `.npy` with ground truth points `(N,3)`. If empty, sampled from mesh. |
| `normals_file_path` | string | `""` | Optional `.npy` with surface normals `(N,3)`. If empty, estimated from mesh. |
| `scene_name` | string | `"custom_scene"` | Label for logging. |
| `num_sample_points` | int | `50000` | Points to sample from mesh if no `points_file_path`. |

### Camera

| Parameter | Type | Default | Description |
|---|---|---|---|
| `camera.width` | int | `640` | Image width (px) |
| `camera.height` | int | `480` | Image height (px) |
| `camera.horizontal_fov_deg` | double | `87.0` | Horizontal FOV (degrees) |
| `camera.vertical_fov_deg` | double | `58.0` | Vertical FOV (degrees) |
| `camera.min_depth` | double | `0.1` | Min sensing range (m) |
| `camera.max_depth` | double | `1.0` | Max sensing range (m) |
| `camera.depth_noise_coeff` | double | `0.001` | Noise model: sigma_z = coeff * z^2 |
| `camera.planning_subsample` | int | `4` | Pixel downsample for fast scoring |

### Workspace

| Parameter | Type | Default | Description |
|---|---|---|---|
| `workspace.base_position` | double[] | `[0.0, -0.5, 0.3]` | Robot base in world frame |
| `workspace.inner_radius` | double | `0.3` | Min reach (m) |
| `workspace.outer_radius` | double | `0.8` | Max reach (m) |
| `workspace.min_camera_angle_to_target` | double | `0.5` | cos(angle) threshold |
| `workspace.object_center` | double[] | `[0,0,0]` | Object center. `[0,0,0]` = auto from mesh. |
| `workspace.forbidden_zone_min` | double[] | `[-1,-1,-0.05]` | Forbidden zone AABB min |
| `workspace.forbidden_zone_max` | double[] | `[1,1,0]` | Forbidden zone AABB max |
| `workspace.use_forbidden_zone` | bool | `true` | Enable table collision zone |

### Planner

| Parameter | Type | Default | Description |
|---|---|---|---|
| `planner.method` | string | `"greedy"` | `"greedy"` or `"sequence_optimized"` |
| `planner.max_views` | int | `20` | Maximum waypoints |
| `planner.num_candidates` | int | `300` | Candidate viewpoints per iteration |
| `planner.convergence_threshold` | double | `0.01` | Greedy only: stop when gain < threshold |
| `planner.coverage_weight` | double | `1.0` | Weight for new-point discovery (alpha) |
| `planner.density_weight` | double | `0.1` | Weight for total visible points (beta) |
| `planner.neighbor_threshold` | double | `0.003` | Coverage distance threshold (m) |
| `planner.min_angular_separation_deg` | double | `15.0` | Sequence optimizer: min viewpoint separation |
| `planner.random_seed` | int | `42` | RNG seed for reproducibility |

### Output

| Parameter | Type | Default | Description |
|---|---|---|---|
| `output.waypoints_topic` | string | `"/nbv_waypoints"` | PoseArray topic name |
| `output.coverage_topic` | string | `"/nbv_coverage"` | Coverage diagnostics topic |
| `output.frame_id` | string | `"base_link"` | TF frame for published poses |

---

## Topics Published

| Topic | Type | QoS | Description |
|---|---|---|---|
| `/nbv_waypoints` | `geometry_msgs/PoseArray` | transient_local, reliable | Ordered camera waypoints |
| `/nbv_coverage` | `std_msgs/String` | transient_local, reliable | JSON coverage diagnostics |

---

## Troubleshooting

**`ImportError: nbv_planner library not found`**
Run `pip install -e /path/to/traj_optim/nbv_planner` in the same Python environment ROS 2 uses.

**`mesh_file_path is required`**
The planner needs a mesh for raycasting. Provide a `.ply` file path.

**Late subscriber doesn't receive the PoseArray**
Your subscriber must use `transient_local` durability QoS to receive latched messages. See the example above.

**Poses seem rotated relative to my EE frame**
The published poses use optical frame convention (Z-forward). Apply your camera-to-EE transform from your URDF.
