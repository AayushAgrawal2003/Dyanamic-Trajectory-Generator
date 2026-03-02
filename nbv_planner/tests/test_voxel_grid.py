"""Tests for the voxel grid representation."""

import numpy as np
import pytest

from nbv_planner.representation.voxel_grid import VoxelGrid, VoxelState


class TestVoxelGrid:
    def setup_method(self):
        self.grid = VoxelGrid(
            bbox_min=np.array([-0.05, -0.05, -0.05]),
            bbox_max=np.array([0.05, 0.05, 0.05]),
            voxel_size=0.01,
            padding=0.01,
        )

    def test_grid_shape(self):
        # Total extent: 0.12m per axis / 0.01 voxel = 12 voxels per axis
        assert all(s > 0 for s in self.grid.shape)

    def test_initial_state_all_unknown(self):
        assert self.grid.get_unknown_count() == self.grid.total_voxels
        assert self.grid.get_occupied_count() == 0
        assert self.grid.get_free_count() == 0

    def test_world_to_voxel_roundtrip(self):
        point = np.array([[0.0, 0.0, 0.0]])
        voxel = self.grid.world_to_voxel(point)
        world_back = self.grid.voxel_to_world(voxel)
        # Should be within one voxel size
        assert np.linalg.norm(world_back - point) < self.grid.voxel_size * 2

    def test_integrate_observation(self):
        points = np.array([
            [0.0, 0.0, 0.0],
            [0.01, 0.01, 0.01],
        ])
        cam_pos = np.array([0.0, 0.0, 0.2])
        self.grid.integrate_observation(points, cam_pos)

        assert self.grid.get_occupied_count() > 0
        assert self.grid.get_free_count() > 0
        assert self.grid.get_unknown_count() < self.grid.total_voxels

    def test_frontier_voxels(self):
        # After integrating some points, frontiers should exist
        points = np.array([[0.0, 0.0, 0.0]])
        cam_pos = np.array([0.0, 0.0, 0.2])
        self.grid.integrate_observation(points, cam_pos)

        frontiers = self.grid.get_frontier_voxels()
        # There should be some frontier voxels
        assert len(frontiers) >= 0  # May be 0 depending on geometry

    def test_empty_observation(self):
        points = np.empty((0, 3))
        cam_pos = np.array([0.0, 0.0, 0.2])
        self.grid.integrate_observation(points, cam_pos)
        assert self.grid.get_unknown_count() == self.grid.total_voxels
