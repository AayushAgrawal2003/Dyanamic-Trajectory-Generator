# Technical Details: NBV Trajectory Optimization for Robotic 3D Scanning

## Problem Statement

Given a target object represented as a triangle mesh, compute an ordered sequence of camera viewpoints for a KUKA LBR Med 7 manipulator that maximizes surface coverage while minimizing total trajectory length. The output is a set of SE(3) waypoints (position + orientation) that the robot can execute sequentially to scan the object.

---

## System Architecture

```
                    +-----------------+
                    |   Input Mesh    |
                    |   (.ply file)   |
                    +--------+--------+
                             |
                    +--------v--------+
                    | Ground Truth    |
                    | Point Sampling  |
                    | (50K surface    |
                    |  pts + normals) |
                    +--------+--------+
                             |
              +--------------+--------------+
              |                             |
     +--------v--------+          +--------v--------+
     |  Greedy NBV     |    OR   | Sequence Optim. |
     |  (iterative)    |         | (select + TSP)  |
     +--------+--------+         +--------+--------+
              |                             |
              +--------------+--------------+
                             |
                    +--------v--------+
                    | PlanningResult  |
                    | (ordered 4x4   |
                    |  camera poses)  |
                    +--------+--------+
                             |
                    +--------v--------+
                    |  ROS 2 Node     |
                    | T_cam_world ->  |
                    | PoseArray in    |
                    | base_link       |
                    +--------+--------+
                             |
                    +--------v--------+
                    |  /nbv_waypoints |
                    | geometry_msgs/  |
                    |   PoseArray     |
                    +-----------------+
```

---

## Core Algorithms

### 1. Greedy Next-Best-View Planner

A myopic, iterative algorithm that selects the single best viewpoint at each step given the current state of coverage.

**Algorithm:**

```
Input:  Mesh M, Camera model C, Workspace W, max_views K
Output: Ordered list of camera poses P = [p_0, p_1, ..., p_n]

1. Sample 50K ground truth points G from mesh surface M
2. Build KD-tree over G for coverage queries
3. Initialize: p_0 = default pose looking at object center
4. Execute p_0: raycast depth image, backproject to point cloud, update coverage

5. For step i = 1, ..., K-1:
   a. Identify frontier: unobserved GT points F = {g in G : not covered}
   b. Sample N candidate viewpoints:
      - 70% frontier-directed: pick random frontier point, offset along
        surface normal by random radius r in {0.2, 0.3, 0.4, 0.5}m
      - 30% uniform hemisphere: random directions at multiple radii
   c. Filter candidates by workspace feasibility:
      - Distance from robot base in [inner_radius, outer_radius]
      - Camera Z-axis pointing toward object (cos(angle) > threshold)
      - Not inside forbidden zones (table surface AABB)
   d. Score each candidate:
        score(p) = alpha * new_unique(p) + beta * total_visible(p)
      where:
        - total_visible(p) = count of GT points in camera FOV, depth range,
          and with surface normal facing camera
        - new_unique(p) = subset of total_visible not yet covered
   e. Select p_i = argmax(score)
   f. If score(p_i) < convergence_threshold * |G|: stop
   g. Execute p_i: simulate depth, update coverage

6. Return P = [p_0, ..., p_n], coverage metrics per frame
```

**Complexity per iteration:** O(N_candidates * |G|) for scoring, where |G| = 50K ground truth points. The fast visibility check avoids raycasting by using FoV projection + normal dot product test.

### 2. Sequence-Optimized Planner

A two-phase algorithm that decouples viewpoint selection from trajectory ordering.

**Phase 1 — Viewpoint Selection:**

```
1. Generate a large candidate pool (500 samples, uniform hemisphere)
2. Score ALL candidates against the initial empty coverage state
3. Sort by score descending
4. Greedily select top-M diverse viewpoints:
   For each candidate (in score order):
     - Compute angular separation from all already-selected viewpoints
       as seen from the object center
     - Accept only if min_separation > 15 degrees
     - Stop at max_views
```

This produces a set of high-information, spatially diverse viewpoints without the sequential bias of the greedy approach.

**Phase 2 — TSP Trajectory Optimization:**

```
1. Build pairwise Euclidean distance matrix D[i,j] between all
   selected viewpoint positions (in world frame)
2. Nearest-Neighbor heuristic:
   - Start from initial pose
   - Repeatedly visit nearest unvisited viewpoint
   - Produces initial tour
3. 2-opt local search (up to 100 iterations):
   - For each pair of edges (i, i+1) and (j, j+1):
     - If reversing the sub-tour [i+1 ... j] reduces total distance:
       reverse it
   - Repeat until no improvement found
4. Return reordered viewpoint sequence
```

The 2-opt improvement typically reduces trajectory length by 15-30% over the nearest-neighbor initial solution.

---

## Sensor Model

### Camera (Pinhole Model)

Intrinsics derived from field-of-view:

```
fx = width  / (2 * tan(hfov / 2))
fy = height / (2 * tan(vfov / 2))
cx = width  / 2
cy = height / 2
```

**Depth noise model:** Gaussian noise with depth-dependent standard deviation:

```
sigma_z = depth_noise_coeff * z^2
```

This models the quadratic depth uncertainty characteristic of structured-light depth sensors.

### Fast Visibility Approximation

For scoring hundreds of candidate viewpoints per iteration, full raycasting is too expensive. Instead, the scorer uses a three-check approximate visibility test:

```
visible(point, normal, T_cam_world) =
    in_fov(point, T_cam_world)             -- projects inside image bounds
  AND in_depth_range(point, T_cam_world)   -- depth in [min_depth, max_depth]
  AND facing_camera(normal, cam_position)  -- dot(normal, to_camera) > 0
```

This runs vectorized over all 50K GT points in ~1ms per candidate, enabling real-time scoring of 300+ candidates per planning step.

### Depth Simulation (for execution)

When a viewpoint is *selected* (not just scored), full raycasting is performed:

1. Generate camera rays for each pixel (or subsampled grid)
2. Raycast against the triangle mesh using trimesh (with pyembree acceleration if available)
3. Compute depth image from hit distances
4. Add Gaussian depth noise
5. Backproject valid depth pixels to 3D point cloud in world frame

---

## Coverage Model

### Ground Truth Representation

The target surface is represented as 50,000 points uniformly sampled from the mesh surface, each with an associated face normal. A KD-tree is built over these points for O(log N) proximity queries.

### Coverage Metric

A ground truth point g is *covered* if any observed point o satisfies:

```
||g - o||_2 <= neighbor_threshold    (default: 3mm)
```

Coverage is computed by querying the KD-tree of observed points against all ground truth points. The fraction of covered GT points gives the coverage metric:

```
coverage = |{g in G : min_o ||g - o|| <= threshold}| / |G|
```

### Information Gain Scoring

Each candidate viewpoint is scored by:

```
score(p) = alpha * new_unique_points(p) + beta * total_visible_points(p)
```

Where:
- `new_unique_points(p)`: GT points visible from p that are NOT yet covered (exploration incentive)
- `total_visible_points(p)`: all GT points visible from p (density incentive)
- `alpha = 1.0` (coverage weight): prioritizes discovering unseen surface
- `beta = 0.1` (density weight): secondary factor rewarding information-rich views

---

## Workspace Model

### KUKA LBR Med 7 Feasibility

The robot workspace is modeled as a hollow sphere centered on the robot base:

```
Feasible if ALL of:
  1. inner_radius <= ||cam_pos - base_pos|| <= outer_radius
  2. cos(angle(cam_z_axis, direction_to_object)) >= threshold
  3. cam_pos not inside any forbidden zone (AABB)
```

Default parameters:
- Base position: [0.0, -0.5, 0.3] m (relative to object)
- Inner radius: 0.3 m (minimum reach)
- Outer radius: 0.8 m (maximum reach)
- Angle threshold: cos(60 deg) = 0.5 (camera must face roughly toward object)
- Forbidden zone: table surface at z in [-0.05, 0.0] m

This is a simplified model — no joint-space IK is computed. The output poses are Cartesian camera poses in the `base_link` frame, intended to be fed to MoveIt for IK resolution and collision-free motion planning.

---

## Viewpoint Sampling Strategies

### Uniform Hemisphere Sampling

Points are sampled uniformly on the upper hemisphere (z > 0.05 to stay above the table) at multiple radii {0.2, 0.3, 0.4, 0.5} m from the object center. Each point generates a `look_at` pose directed at the object center.

### Frontier-Directed Sampling

Biased toward uncovered regions of the surface:

1. Pick a random uncovered GT point
2. Offset along its surface normal (+ 30% random perturbation) by a random radius
3. Generate `look_at` pose toward object center
4. Filter by workspace feasibility

The combined strategy uses 70% frontier-directed + 30% uniform hemisphere sampling to balance exploitation (cover frontiers) with exploration (discover new regions).

---

## Coordinate Conventions

### Transform Convention

All internal poses are `T_cam_world` (4x4 camera-from-world extrinsic matrices):

```
p_cam = T_cam_world @ p_world_homogeneous
```

### Camera Frame

- **Z-axis**: forward (into the scene, toward target)
- **X-axis**: right
- **Y-axis**: down

This matches the standard optical frame convention.

### ROS Output

The ROS node inverts each pose before publishing:

```
T_world_cam = inv(T_cam_world)
position    = T_world_cam[:3, 3]
quaternion  = rotation_matrix_to_quat(T_world_cam[:3, :3])
```

Published `geometry_msgs/Pose` values represent the camera's position and orientation in the `base_link` (world) frame.

---

## Voxel Grid (Internal)

A volumetric occupancy grid tracks explored vs. unexplored space:

- **Voxel size**: 2 mm
- **States**: UNKNOWN (0), FREE (1), OCCUPIED (2)
- **Integration**: observed points mark OCCUPIED voxels; ray-traced intermediate voxels mark FREE
- **Frontier detection**: UNKNOWN voxels adjacent (6-connected) to FREE voxels define the exploration boundary

The frontier is used by the greedy planner to direct viewpoint sampling toward unexplored regions.

---

## Performance Characteristics

| Metric | Greedy | Sequence Optimized |
|---|---|---|
| Planning style | Iterative, myopic | Batch select + TSP |
| Viewpoints scored per step | 300 | 500 (single batch) |
| Coverage bias | Exploits current frontier | Global diversity |
| Trajectory efficiency | Higher (shorter paths) | Variable (TSP helps) |
| Typical coverage (20 views) | 85-95% | 80-92% |
| Typical planning time | 30-120s | 15-60s |

The greedy planner tends to achieve higher coverage because it adapts to the actual observation at each step. The sequence optimizer is faster because scoring is done once against the initial state, but it cannot react to what was actually observed.

---

## Dependencies

| Library | Version | Purpose |
|---|---|---|
| numpy | >= 1.24 | Array operations, linear algebra |
| scipy | >= 1.10 | KD-tree spatial queries, rotation conversions |
| trimesh | >= 4.0 | Mesh loading, raycasting, surface sampling |
| open3d | >= 0.17 | 3D visualization (optional for ROS node) |
| rclpy | Humble | ROS 2 Python client |
| geometry_msgs | Humble | PoseArray message type |
