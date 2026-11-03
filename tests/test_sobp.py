"""Tests for Spread-Out Bragg Peak (SOBP) Synthesis."""

import numpy as np
import pytest

from hadron_engine.physics import PROTON
from hadron_engine.sobp import optimize_sobp_weights


def test_optimize_sobp_weights():
    z_prox = 5.0
    z_dist = 8.5
    prescribed_dose = 60.0

    sobp = optimize_sobp_weights(
        target_proximal_cm=z_prox,
        target_distal_cm=z_dist,
        prescribed_dose_gy=prescribed_dose,
        particle=PROTON,
        num_layers=10,
    )

    assert len(sobp.energies) == 10
    assert np.all(sobp.weights >= 0.0)  # Non-negative fluence constraint
    assert sobp.composite_dose.shape == sobp.depths.shape

    # Target dose coverage
    target_mask = (sobp.depths >= z_prox) & (sobp.depths <= z_dist)
    target_doses = sobp.composite_dose[target_mask]
    assert np.mean(target_doses) > 0.0
    # Distal dose behind target must fall off sharply
    distal_mask = sobp.depths > (z_dist + 1.0)
    distal_max = np.max(sobp.composite_dose[distal_mask])
    assert distal_max < 0.2 * prescribed_dose


def test_sobp_invalid_inputs():
    with pytest.raises(ValueError):
        optimize_sobp_weights(target_proximal_cm=8.0, target_distal_cm=5.0)
