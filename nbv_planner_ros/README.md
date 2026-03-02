# nbv_planner_ros

ROS 2 Humble package for Next-Best-View (NBV) trajectory optimization on the KUKA LBR Med 7. Takes a target object mesh (`.ply`) and/or point cloud (`.npy`), runs viewpoint planning, and publishes optimized camera waypoints as `geometry_msgs/PoseArray`.

---

## Prerequisites

- Ubuntu 22.04 + ROS 2 Humble
- Python 3.9+
- The `nbv_planner` library (included in this repo)

---

## Setup (Fresh Linux Machine)

### 1. Install ROS 2 Humble

```bash
source /opt/ros/humble/setup.bash
ros2 --version  # verify
```

If not installed: [official guide](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html).

### 2. Clone the repo

```bash
cd ~
git clone <your-repo-url> traj_optim
```

### 3. Install the Python planning library + dependencies

```bash
# Make sure pip/setuptools are up to date (needs setuptools >= 68)
pip install --upgrade pip setuptools wheel

# Install the planning engine
cd ~/traj_optim/nbv_planner
pip install -e .

# Install rtree (needed by trimesh for raycasting)
sudo apt install libspatialindex-dev
pip install rtree
```

### 4. Create ROS 2 workspace and build

```bash
mkdir -p ~/ros2_ws/src
ln -s ~/traj_optim/nbv_planner_ros ~/ros2_ws/src/nbv_planner_ros

cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select nbv_planner_ros
source install/setup.bash
```

> **Tip:** Add to `~/.bashrc`: `source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash`

---

## Configure for Your Setup

**There are three things to configure:** your camera sensor, your robot workspace, and your planner settings. All of these go in a single YAML config file — **not** in the XACRO.

### Why not the XACRO?

Your XACRO/URDF defines the robot's kinematic chain and link geometry (for MoveIt collision checking). The NBV planner needs different information: the camera's optical properties (FOV, resolution, depth range) and the robot's reachable workspace. These are set in the YAML config file.

Your XACRO handles one important thing separately: the **camera-to-end-effector transform**. If your camera is mounted at an offset from `link_7` (e.g., your custom EE link at 0.189m), that transform lives in the XACRO and is used by MoveIt when executing the waypoints — the planner doesn't need it.

### Step-by-step: Create your config

1. **Copy the default config:**

```bash
cp ~/traj_optim/nbv_planner_ros/config/default_params.yaml ~/my_scan_config.yaml
```

2. **Edit `~/my_scan_config.yaml`** — here's what to change:

```yaml
nbv_planner:
  ros__parameters:

    # ── YOUR OBJECT ───────────────────────────────────────────────
    mesh_file_path: "/path/to/your_object.ply"

    # ── YOUR CAMERA (from sensor datasheet) ───────────────────────
    # Example values shown for Intel RealSense D435.
    # Replace with YOUR camera's specs.
    camera:
      width: 640                    # image width in pixels
      height: 480                   # image height in pixels
      horizontal_fov_deg: 87.0      # horizontal FOV from spec sheet
      vertical_fov_deg: 58.0        # vertical FOV from spec sheet
      min_depth: 0.1                # minimum sensing range (meters)
      max_depth: 1.0                # maximum sensing range (meters)
      depth_noise_coeff: 0.001      # depth noise model (sigma = coeff * z^2)
      planning_subsample: 4         # pixel downsample for speed

    # ── YOUR ROBOT WORKSPACE ──────────────────────────────────────
    # These define where the planner can place viewpoints.
    workspace:
      base_position: [0.0, -0.5, 0.3]   # robot base position in world frame
      inner_radius: 0.3                   # min reach from base (meters)
      outer_radius: 0.8                   # max reach from base (meters)
      object_center: [0.0, 0.0, 0.0]     # [0,0,0] = auto-detect from mesh
      forbidden_zone_min: [-1.0, -1.0, -0.05]  # table surface AABB min
      forbidden_zone_max: [1.0, 1.0, 0.0]      # table surface AABB max
      use_forbidden_zone: true

    # ── PLANNER SETTINGS ──────────────────────────────────────────
    planner:
      method: "greedy"              # or "sequence_optimized"
      max_views: 20
      num_candidates: 300

    # ── OUTPUT ────────────────────────────────────────────────────
    output:
      frame_id: "base_link"
      save_npy_path: "/tmp/waypoints.npy"  # saves waypoints for later use
```

### Where to find your camera parameters

| Parameter | Where to get it |
|---|---|
| `width`, `height` | Camera resolution setting you use (e.g., 640x480, 1280x720) |
| `horizontal_fov_deg` | Sensor datasheet. RealSense D435: 87°, D455: 87°, Azure Kinect: 75° |
| `vertical_fov_deg` | Sensor datasheet. RealSense D435: 58°, D455: 58°, Azure Kinect: 65° |
| `min_depth`, `max_depth` | Operating range from datasheet. D435: 0.1–10m (but use practical range ~0.1–1.0m) |
| `depth_noise_coeff` | Noise model coefficient. 0.001 is reasonable for structured-light sensors |

### Where to find your workspace parameters

| Parameter | Where to get it |
|---|---|
| `base_position` | Measure your robot base position in the world frame (where `base_link` is) |
| `inner_radius` | Minimum useful reach — typically 0.2–0.4m for KUKA Med 7 |
| `outer_radius` | Maximum reach — ~0.8m for KUKA Med 7 (conservative to avoid joint limits) |
| `object_center` | Leave as `[0,0,0]` for auto-detection, or set manually if you know where the object is |
| `forbidden_zone_*` | The table/surface your object sits on. Set min/max to an AABB enclosing it |

---

## Run

### Option A: Launch file with config (recommended)

```bash
ros2 launch nbv_planner_ros nbv_planner.launch.py \
    config_file:=$HOME/my_scan_config.yaml
```

### Option B: Direct node with CLI overrides

```bash
ros2 run nbv_planner_ros nbv_planner_node --ros-args \
    -p mesh_file_path:=/path/to/object.ply \
    -p planner.method:=sequence_optimized \
    -p planner.max_views:=15 \
    -p camera.horizontal_fov_deg:=69.0 \
    -p camera.vertical_fov_deg:=42.0 \
    -p output.save_npy_path:=/tmp/waypoints.npy
```

### Option C: Quick test with bundled mesh

```bash
ros2 run nbv_planner_ros nbv_planner_node \
    --ros-args -p mesh_file_path:=$HOME/traj_optim/nbv_planner/data/bunny_mesh.ply
```

---

## Read the Output

The node publishes with **transient_local** (latched) QoS, then spins so late subscribers receive the data.

### Waypoints

```bash
ros2 topic echo /nbv_waypoints --qos-durability transient_local
```

Returns `geometry_msgs/PoseArray` in `base_link` frame. Each pose = camera position + orientation (quaternion). Execute them in order.

### Coverage diagnostics

```bash
ros2 topic echo /nbv_coverage --qos-durability transient_local
```

Returns JSON with method, num_views, final_coverage, per-frame metrics.

### Saved .npy files

If `output.save_npy_path` is set, the node saves:
- `waypoints.npy` — (N, 7) array: `[x, y, z, qx, qy, qz, qw]` in `base_link` frame
- `waypoints_4x4.npy` — (N, 4, 4) raw T_cam_world matrices (for full precision)

---

## Visualize Waypoints

Use the bundled visualization script to inspect generated waypoints:

```bash
# Basic — just waypoints
python3 ~/traj_optim/nbv_planner_ros/scripts/visualize_waypoints.py /tmp/waypoints.npy

# With the target mesh overlay
python3 ~/traj_optim/nbv_planner_ros/scripts/visualize_waypoints.py /tmp/waypoints.npy \
    --mesh /path/to/object.ply

# Using raw 4x4 matrices
python3 ~/traj_optim/nbv_planner_ros/scripts/visualize_waypoints.py /tmp/waypoints_4x4.npy \
    --raw4x4 --mesh /path/to/object.ply

# Adjust frustum size
python3 ~/traj_optim/nbv_planner_ros/scripts/visualize_waypoints.py /tmp/waypoints.npy \
    --mesh /path/to/object.ply --frustum-scale 0.06
```

Opens an Open3D window showing:
- **Green frustum** → first waypoint
- **Red frustum** → last waypoint
- **Orange line** → trajectory path
- **Gray mesh** → target object

---

## Feeding Waypoints to MoveIt

The PoseArray on `/nbv_waypoints` gives camera poses in `base_link`. To execute with MoveIt:

1. **Subscribe** with `transient_local` QoS
2. **Iterate** through poses in order
3. **Plan + execute** to each pose

```python
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from geometry_msgs.msg import PoseArray

qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                  reliability=ReliabilityPolicy.RELIABLE)

self.create_subscription(PoseArray, '/nbv_waypoints', self.waypoints_cb, qos)
```

**Camera frame convention:** Published poses use the optical frame convention (Z-forward, X-right, Y-down). If your MoveIt end-effector frame differs from the camera optical frame, apply the static transform defined in your URDF/XACRO. For example, if your camera is mounted on a custom EE link, MoveIt already knows the `camera_optical_frame → end_effector` transform from the XACRO — just plan to the poses using the camera optical frame as your planning frame, or manually transform using your known camera-to-EE offset.

---

## Full Parameter Reference

All parameters under the `nbv_planner` node. Set via YAML config or `--ros-args -p`.

### Scene

| Parameter | Type | Default | Description |
|---|---|---|---|
| `mesh_file_path` | string | `""` | **Required.** Path to `.ply` mesh file |
| `points_file_path` | string | `""` | Optional `.npy` with ground truth points `(N,3)` |
| `normals_file_path` | string | `""` | Optional `.npy` with normals `(N,3)` |
| `scene_name` | string | `"custom_scene"` | Label for logging |
| `num_sample_points` | int | `50000` | Points to sample if no `points_file_path` |

### Camera

| Parameter | Type | Default | Description |
|---|---|---|---|
| `camera.width` | int | `640` | Image width (px) |
| `camera.height` | int | `480` | Image height (px) |
| `camera.horizontal_fov_deg` | double | `87.0` | Horizontal FOV (degrees) |
| `camera.vertical_fov_deg` | double | `58.0` | Vertical FOV (degrees) |
| `camera.min_depth` | double | `0.1` | Min sensing range (m) |
| `camera.max_depth` | double | `1.0` | Max sensing range (m) |
| `camera.depth_noise_coeff` | double | `0.001` | Noise: sigma_z = coeff * z² |
| `camera.planning_subsample` | int | `4` | Pixel downsample for scoring |

### Workspace

| Parameter | Type | Default | Description |
|---|---|---|---|
| `workspace.base_position` | double[] | `[0.0, -0.5, 0.3]` | Robot base in world frame |
| `workspace.inner_radius` | double | `0.3` | Min reach (m) |
| `workspace.outer_radius` | double | `0.8` | Max reach (m) |
| `workspace.min_camera_angle_to_target` | double | `0.5` | cos(angle) threshold |
| `workspace.object_center` | double[] | `[0,0,0]` | Object center (`[0,0,0]` = auto) |
| `workspace.forbidden_zone_min` | double[] | `[-1,-1,-0.05]` | Forbidden AABB min |
| `workspace.forbidden_zone_max` | double[] | `[1,1,0]` | Forbidden AABB max |
| `workspace.use_forbidden_zone` | bool | `true` | Enable table collision zone |

### Planner

| Parameter | Type | Default | Description |
|---|---|---|---|
| `planner.method` | string | `"greedy"` | `"greedy"` or `"sequence_optimized"` |
| `planner.max_views` | int | `20` | Maximum waypoints |
| `planner.num_candidates` | int | `300` | Candidates per iteration |
| `planner.convergence_threshold` | double | `0.01` | Greedy: stop when gain < this |
| `planner.coverage_weight` | double | `1.0` | Weight for new-point discovery |
| `planner.density_weight` | double | `0.1` | Weight for total visible points |
| `planner.neighbor_threshold` | double | `0.003` | Coverage distance threshold (m) |
| `planner.min_angular_separation_deg` | double | `15.0` | Sequence optimizer only |
| `planner.random_seed` | int | `42` | RNG seed |

### Output

| Parameter | Type | Default | Description |
|---|---|---|---|
| `output.waypoints_topic` | string | `"/nbv_waypoints"` | PoseArray topic |
| `output.coverage_topic` | string | `"/nbv_coverage"` | Coverage diagnostics topic |
| `output.frame_id` | string | `"base_link"` | TF frame for published poses |
| `output.save_npy_path` | string | `""` | Save waypoints to .npy (empty = don't save) |

---

## Topics Published

| Topic | Type | QoS | Description |
|---|---|---|---|
| `/nbv_waypoints` | `geometry_msgs/PoseArray` | transient_local | Ordered camera waypoints |
| `/nbv_coverage` | `std_msgs/String` | transient_local | JSON coverage diagnostics |

---

## Troubleshooting

**`ImportError: nbv_planner library not found`**
→ `pip install -e ~/traj_optim/nbv_planner`

**`mesh_file_path is required`**
→ Provide a `.ply` file path in config or via `-p mesh_file_path:=...`

**`build backend is missing the build_editable hook`**
→ `pip install --upgrade pip setuptools wheel` (need setuptools >= 68)

**`ModuleNotFoundError: rtree`**
→ `sudo apt install libspatialindex-dev && pip install rtree`

**PLY has 0 faces (point cloud)**
→ The node auto-reconstructs a mesh via Poisson reconstruction. This is normal.

**Late subscriber doesn't receive PoseArray**
→ Your subscriber must use `transient_local` durability QoS.

**Poses seem rotated relative to my EE frame**
→ Published poses use optical frame convention (Z-forward). Apply your camera-to-EE transform from your URDF.
