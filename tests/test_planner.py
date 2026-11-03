"""Tests for IMPT Multi-Beam Inverse Treatment Planner."""

import numpy as np
import pytest

from hadron_engine.pencil_beam import PatientCTVolume
from hadron_engine.planner import InverseTreatmentPlanner


def test_inverse_treatment_planner():
    vol = PatientCTVolume.create_cranial_phantom(grid_size=(24, 24, 16), voxel_size=(0.4, 0.4, 0.4))
    planner = InverseTreatmentPlanner(vol, prescribed_dose_gy=50.0)

    spots = planner.generate_spot_lattice([90.0, 270.0], spot_spacing_cm=1.2, layer_spacing_cm=0.8)
    assert len(spots) > 0

    result = planner.optimize_plan(spots, max_iterations=20)
    assert result.dose_3d.shape == vol.grid.shape
    assert np.all(result.dose_3d >= 0.0)
    assert result.ptv_mean_dose > 0.0
    assert len(result.objective_history) == 20
    # Objective function should decrease
    assert result.objective_history[-1] <= result.objective_history[0]
