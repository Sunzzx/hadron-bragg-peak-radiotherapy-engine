import math
"""Tests for Dose-Volume Histogram and Clinical Metric Evaluation."""

import numpy as np
import pytest

from hadron_engine.physics import PROTON, CARBON_ION
from hadron_engine.dvh import (
    calculate_conformity_index,
    calculate_dvh,
    calculate_homogeneity_index,
    calculate_rbe_dose,
)


def test_calculate_dvh():
    # Synthetic spherical tumor dose
    dose_3d = np.zeros((20, 20, 20))
    mask_3d = np.zeros((20, 20, 20), dtype=bool)
    mask_3d[8:12, 8:12, 8:12] = True
    dose_3d[mask_3d] = np.random.uniform(55.0, 65.0, size=np.sum(mask_3d))

    dvh = calculate_dvh(dose_3d, mask_3d, structure_name="Tumor")
    assert dvh.structure_name == "Tumor"
    assert dvh.volume_fractions[0] == 100.0
    assert dvh.volume_fractions[-1] == 0.0
    # Monotonic decreasing volume fraction with increasing dose
    diffs = np.diff(dvh.volume_fractions)
    assert np.all(diffs <= 0.0)
    # D98 <= D50 <= D2
    assert dvh.d98 <= dvh.d50 <= dvh.d2


def test_homogeneity_and_conformity_indices():
    dose_3d = np.full((10, 10, 10), 60.0)
    ptv_mask = np.zeros((10, 10, 10), dtype=bool)
    ptv_mask[3:7, 3:7, 3:7] = True
    body_mask = np.ones((10, 10, 10), dtype=bool)

    dvh = calculate_dvh(dose_3d, ptv_mask, "PTV")
    hi = calculate_homogeneity_index(dvh)
    # Perfectly uniform dose -> HI == 0
    assert math.isclose(hi, 0.0, abs_tol=1e-4)

    ci = calculate_conformity_index(dose_3d, ptv_mask, body_mask, prescribed_dose_gy=60.0)
    assert ci > 0.0


def test_rbe_dose():
    dose = np.full((5, 5, 5), 10.0)
    rbe_p = calculate_rbe_dose(dose, PROTON)
    assert np.allclose(rbe_p, 11.0)  # Standard RBE = 1.1

    rbe_c = calculate_rbe_dose(dose, CARBON_ION)
    assert np.all(rbe_c > 10.0)
