"""Tests for 3D Pencil Beam Kernel and Voxelized Patient Phantom."""

import numpy as np
import pytest

from hadron_engine.physics import PROTON
from hadron_engine.pencil_beam import (
    PatientCTVolume,
    PencilBeam,
    VoxelGrid,
    TISSUE_AIR,
    TISSUE_BONE,
    TISSUE_BRAIN,
    TISSUE_LUNG,
)


def test_tissue_properties():
    assert TISSUE_BONE.rsp > TISSUE_BRAIN.rsp > TISSUE_LUNG.rsp > TISSUE_AIR.rsp
    assert TISSUE_BONE.density_g_cm3 > 1.0
    assert TISSUE_AIR.density_g_cm3 < 0.01


def test_voxel_grid():
    grid = VoxelGrid(shape=(10, 12, 14), voxel_size_cm=(0.5, 0.5, 0.5), origin_cm=(-2.5, -3.0, -3.5))
    assert grid.total_voxels == 10 * 12 * 14
    xs, ys, zs = grid.coordinates_1d()
    assert len(xs) == 10
    assert len(ys) == 12
    assert len(zs) == 14
    assert xs[0] == -2.5
    assert xs[-1] == -2.5 + 9 * 0.5


def test_cranial_phantom_creation():
    vol = PatientCTVolume.create_cranial_phantom(grid_size=(30, 30, 20), voxel_size=(0.3, 0.3, 0.3))
    assert np.any(vol.ptv_mask)
    assert "brainstem" in vol.oar_masks
    assert "optic_chiasm" in vol.oar_masks
    assert np.any(vol.oar_masks["brainstem"])
    # PTV and OARs should not overlap
    assert not np.any(vol.ptv_mask & vol.oar_masks["brainstem"])
    assert not np.any(vol.ptv_mask & vol.oar_masks["optic_chiasm"])


def test_pencil_beam_dose_3d():
    vol = PatientCTVolume.create_cranial_phantom(grid_size=(24, 24, 16), voxel_size=(0.35, 0.35, 0.35))
    beam = PencilBeam(
        entry_point=np.array([0.0, -5.0, 0.0]),
        direction=np.array([0.0, 1.0, 0.0]),
        energy_mev=100.0,
        weight=1.0,
        particle=PROTON,
    )
    dose = beam.compute_dose_3d(vol)
    assert dose.shape == vol.grid.shape
    assert np.all(dose >= 0.0)
    assert np.max(dose) > 0.0
    # Dose behind propagation entry point must be 0
    y_coords = vol.grid.coordinates_1d()[1]
    behind_entry = y_coords < -5.0
    assert np.all(dose[:, behind_entry, :] == 0.0)
