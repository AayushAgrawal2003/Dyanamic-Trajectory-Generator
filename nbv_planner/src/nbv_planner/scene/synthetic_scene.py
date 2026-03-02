"""Generate synthetic target objects for NBV planning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh


@dataclass
class SyntheticScene:
    """A synthetic scene with a target object mesh and ground truth cloud."""

    name: str
    mesh: trimesh.Trimesh
    ground_truth_points: np.ndarray  # (N, 3)
    ground_truth_normals: np.ndarray  # (N, 3)
    bounding_box: np.ndarray  # (2, 3) — min, max corners

    @property
    def center(self) -> np.ndarray:
        return (self.bounding_box[0] + self.bounding_box[1]) / 2.0

    @property
    def extent(self) -> np.ndarray:
        return self.bounding_box[1] - self.bounding_box[0]


def create_sphere(
    radius: float = 0.05,
    num_gt_points: int = 50_000,
    seed: int = 42,
) -> SyntheticScene:
    """Create a sphere centered at the origin."""
    mesh = trimesh.creation.icosphere(subdivisions=4, radius=radius)
    points, face_indices = trimesh.sample.sample_surface(mesh, num_gt_points, seed=seed)
    normals = mesh.face_normals[face_indices]

    bbox = np.array([mesh.bounds[0], mesh.bounds[1]])
    return SyntheticScene(
        name="sphere",
        mesh=mesh,
        ground_truth_points=np.asarray(points),
        ground_truth_normals=np.asarray(normals),
        bounding_box=bbox,
    )


def create_bunny(
    num_gt_points: int = 50_000,
    seed: int = 42,
) -> SyntheticScene:
    """Load a Stanford Bunny-style mesh from trimesh built-ins.

    The mesh is centered at the origin and scaled so that its longest
    axis is ~10 cm, matching a small anatomical object.
    """
    # trimesh doesn't ship the Stanford bunny directly — use a built-in primitive
    # and deform it to approximate an irregular shape. Alternatively, try loading
    # from trimesh's online model repository.
    try:
        # Try fetching the bunny from trimesh's model repository
        mesh = trimesh.load(
            "https://graphics.stanford.edu/~mdfisher/Data/Meshes/bunny.obj",
            force="mesh",
        )
    except Exception:
        # Fallback: create an irregular shape by deforming a sphere
        mesh = _create_deformed_sphere()

    # Center and scale
    mesh.vertices -= mesh.centroid
    extents = mesh.bounds[1] - mesh.bounds[0]
    scale = 0.10 / extents.max()  # longest axis = 10 cm
    mesh.vertices *= scale
    mesh.fix_normals()

    points, face_indices = trimesh.sample.sample_surface(mesh, num_gt_points, seed=seed)
    normals = mesh.face_normals[face_indices]

    bbox = np.array([mesh.bounds[0], mesh.bounds[1]])
    return SyntheticScene(
        name="bunny",
        mesh=mesh,
        ground_truth_points=np.asarray(points),
        ground_truth_normals=np.asarray(normals),
        bounding_box=bbox,
    )


def _create_deformed_sphere() -> trimesh.Trimesh:
    """Create an irregular mesh by deforming a sphere with harmonics."""
    mesh = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    rng = np.random.default_rng(seed=123)

    # Add low-frequency deformation using spherical coordinates
    r = np.linalg.norm(vertices, axis=1, keepdims=True)
    theta = np.arctan2(vertices[:, 1], vertices[:, 0])
    phi = np.arccos(np.clip(vertices[:, 2] / r.ravel(), -1, 1))

    deformation = (
        0.15 * np.sin(2 * theta) * np.cos(phi)
        + 0.10 * np.cos(3 * theta) * np.sin(2 * phi)
        + 0.08 * np.sin(theta + phi)
        + 0.05 * rng.standard_normal(len(vertices))
    )
    vertices *= (1.0 + deformation[:, np.newaxis])
    mesh.vertices = vertices
    mesh.fix_normals()
    return mesh


def create_femoral_surface(
    num_gt_points: int = 50_000,
    seed: int = 42,
) -> SyntheticScene:
    """Create a curved surface patch simulating femoral condyle geometry.

    Uses a parametric saddle/partial-ellipsoid surface, ~15cm x 10cm.
    """
    rng = np.random.default_rng(seed)

    # Parametric grid
    nu, nv = 100, 80
    u = np.linspace(-0.075, 0.075, nu)  # 15 cm range
    v = np.linspace(-0.05, 0.05, nv)    # 10 cm range
    U, V = np.meshgrid(u, v, indexing="ij")

    # Saddle-like surface: z = a*u^2 - b*v^2 + curvature
    a, b = 8.0, 12.0
    Z = a * U**2 - b * V**2
    # Add a dome component for condyle shape
    R = np.sqrt(U**2 + V**2)
    Z += 0.02 * np.cos(np.pi * R / 0.1)
    # Shift so surface is above z=0
    Z -= Z.min()

    vertices = np.column_stack([U.ravel(), V.ravel(), Z.ravel()])

    # Create triangulation
    faces = []
    for i in range(nu - 1):
        for j in range(nv - 1):
            idx = i * nv + j
            faces.append([idx, idx + 1, idx + nv])
            faces.append([idx + 1, idx + nv + 1, idx + nv])
    faces = np.array(faces)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.fix_normals()

    points, face_indices = trimesh.sample.sample_surface(mesh, num_gt_points, seed=seed)
    normals = mesh.face_normals[face_indices]

    bbox = np.array([mesh.bounds[0], mesh.bounds[1]])
    return SyntheticScene(
        name="femoral_surface",
        mesh=mesh,
        ground_truth_points=np.asarray(points),
        ground_truth_normals=np.asarray(normals),
        bounding_box=bbox,
    )


def save_scene(scene: SyntheticScene, output_dir: Path) -> None:
    """Save scene mesh and ground truth cloud to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scene.mesh.export(output_dir / f"{scene.name}_mesh.ply")
    np.savez(
        output_dir / f"{scene.name}_ground_truth.npz",
        points=scene.ground_truth_points,
        normals=scene.ground_truth_normals,
        bbox=scene.bounding_box,
    )


def load_scene(name: str, data_dir: Path) -> SyntheticScene:
    """Load a previously saved scene."""
    data_dir = Path(data_dir)
    mesh = trimesh.load(data_dir / f"{name}_mesh.ply", force="mesh")
    gt = np.load(data_dir / f"{name}_ground_truth.npz")

    return SyntheticScene(
        name=name,
        mesh=mesh,
        ground_truth_points=gt["points"],
        ground_truth_normals=gt["normals"],
        bounding_box=gt["bbox"],
    )


ALL_SCENES = {
    "sphere": create_sphere,
    "bunny": create_bunny,
    "femoral_surface": create_femoral_surface,
}
