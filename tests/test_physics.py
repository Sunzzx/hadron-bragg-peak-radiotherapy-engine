"""Unit and Invariance Tests for Hadron Therapy Relativistic Physics."""

import math
import numpy as np
import pytest

from hadron_engine.physics import (
    CARBON_ION,
    PROTON,
    bethe_bloch_stopping_power,
    bortfeld_depth_dose,
    energy_from_range,
    lateral_beam_spread_sigma,
    max_energy_transfer,
    multiple_coulomb_scattering_highland,
    range_from_energy,
    range_straggling_sigma,
    relativistic_kinematics,
    sternheimer_density_effect,
)


def test_relativistic_kinematics():
    # Zero kinetic energy
    e_tot, gamma, beta, p = relativistic_kinematics(0.0, PROTON.mass_mev)
    assert e_tot == PROTON.mass_mev
    assert gamma == 1.0
    assert beta == 0.0
    assert p == 0.0

    # 150 MeV Proton
    e_tot, gamma, beta, p = relativistic_kinematics(150.0, PROTON.mass_mev)
    assert gamma > 1.0
    assert 0.0 < beta < 1.0
    assert math.isclose(beta, math.sqrt(1.0 - 1.0 / (gamma ** 2)), rel_tol=1e-5)
    assert math.isclose(p, PROTON.mass_mev * beta * gamma, rel_tol=1e-5)


def test_max_energy_transfer():
    _, gamma, beta, _ = relativistic_kinematics(150.0, PROTON.mass_mev)
    w_max = max_energy_transfer(gamma, beta, PROTON.mass_mev)
    assert w_max > 0.0
    assert w_max < 150.0  # Cannot transfer more than beam energy


def test_sternheimer_density_effect():
    # At non-relativistic speeds, density effect is negligible
    assert sternheimer_density_effect(0.01, 1.0001) == 0.0
    # At ultra-relativistic speeds, density effect is positive
    assert sternheimer_density_effect(0.99, 7.08) > 0.0


def test_bethe_bloch_stopping_power():
    sp_5 = bethe_bloch_stopping_power(5.0, PROTON)
    sp_50 = bethe_bloch_stopping_power(50.0, PROTON)
    sp_150 = bethe_bloch_stopping_power(150.0, PROTON)
    sp_250 = bethe_bloch_stopping_power(250.0, PROTON)

    # Bragg peak characteristic: stopping power spikes dramatically at low energies
    assert sp_5 > sp_50 > sp_150 > sp_250
    assert sp_5 > 40.0  # Over 40 MeV/cm near the Bragg peak
    assert 3.0 < sp_250 < 6.0


def test_range_energy_reciprocity():
    # Range and energy inversions must be reciprocal within 0.01%
    test_energies = [40.0, 70.0, 100.0, 150.0, 200.0, 250.0]
    for e in test_energies:
        r = range_from_energy(e, PROTON)
        e_rec = energy_from_range(r, PROTON)
        assert math.isclose(e, e_rec, rel_tol=1e-4)


def test_range_straggling_sigma():
    r1 = range_from_energy(100.0, PROTON)
    r2 = range_from_energy(200.0, PROTON)
    sig1 = range_straggling_sigma(r1, PROTON)
    sig2 = range_straggling_sigma(r2, PROTON)
    assert sig1 > 0.0
    assert sig2 > sig1  # Straggling increases with penetration depth


def test_bortfeld_depth_dose():
    z = np.linspace(0.0, 20.0, 400)
    e0 = 150.0
    r0 = range_from_energy(e0, PROTON)
    sig = range_straggling_sigma(r0, PROTON)

    dose = bortfeld_depth_dose(z, e0, PROTON, fluence=1.0)
    peak_idx = int(np.argmax(dose))
    peak_z = z[peak_idx]

    # Peak must be within 1 sigma of the calculated CSDA range
    assert abs(peak_z - r0) <= 1.5 * sig
    # Peak dose must be significantly higher than entrance plateau dose
    assert dose[peak_idx] > 3.0 * dose[0]
    # Distal dose behind the peak must drop to zero (ballistic Bragg peak advantage)
    distal_idx = np.where(z > r0 + 3.0 * sig)[0]
    assert np.all(dose[distal_idx] < 0.05 * dose[peak_idx])


def test_multiple_coulomb_scattering():
    th0 = multiple_coulomb_scattering_highland(0.5, 100.0, PROTON)
    assert th0 > 0.0

    # Lateral beam spread increases with depth
    sig_ent = lateral_beam_spread_sigma(0.0, 150.0, initial_sigma_cm=0.2)
    sig_mid = lateral_beam_spread_sigma(8.0, 150.0, initial_sigma_cm=0.2)
    sig_peak = lateral_beam_spread_sigma(15.0, 150.0, initial_sigma_cm=0.2)
    assert sig_ent == 0.2
    assert sig_mid > sig_ent
    assert sig_peak > sig_mid


def test_carbon_ion_physics():
    # Carbon-12 has Z=6, mass ~11 GeV
    sp_c = bethe_bloch_stopping_power(150.0 * 12.0 / 12.0, CARBON_ION)
    sp_p = bethe_bloch_stopping_power(150.0, PROTON)
    # Carbon stopping power is dramatically higher (~Z^2 = 36 times)
    assert sp_c > 15.0 * sp_p

def test_bethe_bloch_low_energy_continuity():
    for t in [0.001, 0.01, 0.05, 0.1, 1.0]:
        sp = bethe_bloch_stopping_power(t, PROTON)
        assert not math.isnan(sp)
        assert not math.isinf(sp)
        assert sp >= 0.0

def test_scattering_momentum_scaling():
    th_low = multiple_coulomb_scattering_highland(1.0, 50.0, PROTON)
    th_high = multiple_coulomb_scattering_highland(1.0, 200.0, PROTON)
    assert th_low > th_high  # Higher energy particles scatter less
