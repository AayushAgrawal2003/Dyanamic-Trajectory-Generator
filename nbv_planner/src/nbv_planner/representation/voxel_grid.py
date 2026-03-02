"""Voxel grid representation for occupancy tracking."""

from __future__ import annotations

from enum import IntEnum

import numpy as np


class VoxelState(IntEnum):
    UNKNOWN = 0
    FREE = 1
    OCCUPIED = 2


class VoxelGrid:
    """3D voxel grid around the object bounding box.

    Tracks occupancy states: UNKNOWN, FREE, OCCUPIED.
    """

    def __init__(
        self,
        bbox_min: np.ndarray,
        bbox_max: np.ndarray,
        voxel_size: float = 0.002,
        padding: float = 0.05,
    ) -> None:
        self.voxel_size = voxel_size

        # Expand bounding box by padding
        self.origin = np.asarray(bbox_min) - padding
        self.upper = np.asarray(bbox_max) + padding

        # Grid dimensions
        extent = self.upper - self.origin
        self.shape = np.ceil(extent / voxel_size).astype(int)
        self.shape = np.maximum(self.shape, 1)

        # Voxel states
        self._grid = np.full(self.shape, VoxelState.UNKNOWN, dtype=np.int8)

    @property
    def total_voxels(self) -> int:
        return int(np.prod(self.shape))

    def world_to_voxel(self, points: np.ndarray) -> np.ndarray:
        """Convert world coordinates to voxel indices."""
        return np.floor((points - self.origin) / self.voxel_size).astype(int)

    def voxel_to_world(self, indices: np.ndarray) -> np.ndarray:
        """Convert voxel indices to world coordinates (voxel center)."""
        return indices.astype(float) * self.voxel_size + self.origin + self.voxel_size / 2

    def _valid_indices(self, voxel_indices: np.ndarray) -> np.ndarray:
        """Return boolean mask for indices within grid bounds."""
        return np.all(
            (voxel_indices >= 0) & (voxel_indices < self.shape), axis=1
        )

    def integrate_observation(
        self,
        point_cloud: np.ndarray,
        camera_position: np.ndarray,
    ) -> None:
        """Integrate an observation into the voxel grid.

        Marks voxels containing points as OCCUPIED.
        Marks voxels along rays from camera to points as FREE (simplified).
        """
        if len(point_cloud) == 0:
            return

        # Mark occupied voxels
        voxel_indices = self.world_to_voxel(point_cloud)
        valid = self._valid_indices(voxel_indices)
        vi = voxel_indices[valid]
        self._grid[vi[:, 0], vi[:, 1], vi[:, 2]] = VoxelState.OCCUPIED

        # Simplified free-space carving: mark a few voxels along each ray
        # Sample points between camera and hit points
        cam = np.asarray(camera_position)
        num_ray_samples = 5
        for alpha in np.linspace(0.1, 0.9, num_ray_samples):
            intermediate = cam + alpha * (point_cloud - cam)
            inter_vox = self.world_to_voxel(intermediate)
            valid_inter = self._valid_indices(inter_vox)
            iv = inter_vox[valid_inter]
            # Only mark as FREE if currently UNKNOWN
            mask = self._grid[iv[:, 0], iv[:, 1], iv[:, 2]] == VoxelState.UNKNOWN
            iv_free = iv[mask]
            if len(iv_free) > 0:
                self._grid[iv_free[:, 0], iv_free[:, 1], iv_free[:, 2]] = VoxelState.FREE

    def get_frontier_voxels(self) -> np.ndarray:
        """Return voxel indices of UNKNOWN voxels adjacent to FREE voxels.

        Returns:
            (M, 3) array of voxel indices.
        """
        # Find unknown voxels
        unknown = self._grid == VoxelState.UNKNOWN
        # Find free voxels — dilate by 1
        free = self._grid == VoxelState.FREE

        # Check 6-connectivity neighbors
        dilated = np.zeros_like(free)
        dilated[1:, :, :] |= free[:-1, :, :]
        dilated[:-1, :, :] |= free[1:, :, :]
        dilated[:, 1:, :] |= free[:, :-1, :]
        dilated[:, :-1, :] |= free[:, 1:, :]
        dilated[:, :, 1:] |= free[:, :, :-1]
        dilated[:, :, :-1] |= free[:, :, 1:]

        frontier = unknown & dilated
        return np.argwhere(frontier)

    def get_unknown_count(self) -> int:
        return int((self._grid == VoxelState.UNKNOWN).sum())

    def get_occupied_count(self) -> int:
        return int((self._grid == VoxelState.OCCUPIED).sum())

    def get_free_count(self) -> int:
        return int((self._grid == VoxelState.FREE).sum())

    def get_frontier_world_positions(self) -> np.ndarray:
        """Return world positions of frontier voxels."""
        indices = self.get_frontier_voxels()
        if len(indices) == 0:
            return np.empty((0, 3))
        return self.voxel_to_world(indices)
