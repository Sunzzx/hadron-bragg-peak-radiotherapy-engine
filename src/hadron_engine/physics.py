"""Theoretical Hadron Therapy Physics and Relativistic Stopping Power Engine.

Implements:
- Relativistic kinematics (Lorentz factor, beta, momentum, maximum energy transfer)
- Bethe-Bloch stopping power equation for heavy charged particles (protons, Carbon ions)
- ICRU / Sternheimer density effect correction in liquid water
- Bragg-Kleeman range-energy power-law relation
- Bortfeld analytical Bragg peak depth-dose formulation with energy straggling
- Fermi-Eyges multiple Coulomb scattering and lateral Highland spreading
"""

from dataclasses import dataclass
import math
from typing import Tuple, Union
import numpy as np

C_LIGHT: float = 2.99792458e8
ELECTRON_MASS_MEV: float = 0.51099895
PROTON_MASS_MEV: float = 938.272088
CARBON_MASS_MEV: float = 11177.93
CLASSICAL_ELECTRON_RADIUS_CM: float = 2.8179403262e-13
AVOGADRO_NUM: float = 6.02214076e23

WATER_DENSITY: float = 1.0
WATER_Z_OVER_A: float = 0.55509
WATER_I_MEV: float = 75.0e-6
WATER_RAD_LENGTH_CM: float = 36.08

BETHE_PREFACTOR: float = 2.0 * math.pi * AVOGADRO_NUM * (CLASSICAL_ELECTRON_RADIUS_CM ** 2) * ELECTRON_MASS_MEV


@dataclass(frozen=True)
class HadronParticle:
    name: str
    charge_z: int
    mass_mev: float
    bragg_alpha: float
    bragg_p: float
    straggling_coeff: float


PROTON = HadronParticle(
    name="Proton",
    charge_z=1,
    mass_mev=PROTON_MASS_MEV,
    bragg_alpha=0.0022,
    bragg_p=1.77,
    straggling_coeff=0.012,
)

CARBON_ION = HadronParticle(
    name="Carbon-12",
    charge_z=6,
    mass_mev=CARBON_MASS_MEV,
    bragg_alpha=0.00094,
    bragg_p=1.73,
    straggling_coeff=0.004,
)


def relativistic_kinematics(t_mev: float, mass_mev: float) -> Tuple[float, float, float, float]:
    if t_mev <= 0.0:
        return mass_mev, 1.0, 0.0, 0.0
    e_total = t_mev + mass_mev
    gamma = e_total / mass_mev
    beta = math.sqrt(max(0.0, 1.0 - 1.0 / (gamma * gamma)))
    momentum = mass_mev * beta * gamma
    return e_total, gamma, beta, momentum


def max_energy_transfer(gamma: float, beta: float, projectile_mass_mev: float) -> float:
    if beta <= 0.0:
        return 0.0
    m_ratio = ELECTRON_MASS_MEV / projectile_mass_mev
    numerator = 2.0 * ELECTRON_MASS_MEV * (beta * gamma) ** 2
    denominator = 1.0 + 2.0 * gamma * m_ratio + (m_ratio ** 2)
    return numerator / denominator


def sternheimer_density_effect(beta: float, gamma: float) -> float:
    if beta <= 0.0:
        return 0.0
    bg = beta * gamma
    if bg <= 0.0:
        return 0.0
    x = math.log10(bg)
    x0 = 0.2400
    x1 = 2.8004
    m = 3.4773
    c_bar = 3.5017
    a = 0.09116

    if x < x0:
        return 0.0
    elif x < x1:
        return 4.6052 * x - c_bar + a * ((x1 - x) ** m)
    else:
        return 4.6052 * x - c_bar


def bethe_bloch_stopping_power(
    t_mev: float,
    particle: HadronParticle = PROTON,
    density_g_cm3: float = WATER_DENSITY,
    z_over_a: float = WATER_Z_OVER_A,
    mean_i_mev: float = WATER_I_MEV,
) -> float:
    if t_mev <= 0.001:
        return 0.0

    _, gamma, beta, _ = relativistic_kinematics(t_mev, particle.mass_mev)
    if beta < 0.005:
        ref_t = 0.05
        ref_sp = bethe_bloch_stopping_power(ref_t, particle, density_g_cm3, z_over_a, mean_i_mev)
        return ref_sp * math.sqrt(t_mev / ref_t)

    w_max = max_energy_transfer(gamma, beta, particle.mass_mev)
    delta = sternheimer_density_effect(beta, gamma)

    arg = (2.0 * ELECTRON_MASS_MEV * (beta * gamma) ** 2 * w_max) / (mean_i_mev ** 2)
    if arg <= 1.0:
        return 0.0

    bracket = math.log(arg) - 2.0 * (beta ** 2) - delta
    bracket = max(bracket, 0.0)

    sp_mass = (
        BETHE_PREFACTOR
        * density_g_cm3
        * z_over_a
        * (particle.charge_z ** 2)
        / (beta ** 2)
        * bracket
    )
    return sp_mass


def range_from_energy(energy_mev: float, particle: HadronParticle = PROTON) -> float:
    if energy_mev <= 0.0:
        return 0.0
    return particle.bragg_alpha * (energy_mev ** particle.bragg_p)


def energy_from_range(range_cm: float, particle: HadronParticle = PROTON) -> float:
    if range_cm <= 0.0:
        return 0.0
    return (range_cm / particle.bragg_alpha) ** (1.0 / particle.bragg_p)


def range_straggling_sigma(range_cm: float, particle: HadronParticle = PROTON) -> float:
    if range_cm <= 0.0:
        return 0.001
    return max(0.01, particle.straggling_coeff * (range_cm ** 0.935))


def bortfeld_depth_dose(
    depth_cm: Union[float, np.ndarray],
    energy_mev: float,
    particle: HadronParticle = PROTON,
    fluence: float = 1.0,
) -> np.ndarray:
    z = np.atleast_1d(np.asarray(depth_cm, dtype=np.float64))
    r0 = range_from_energy(energy_mev, particle)
    sigma = range_straggling_sigma(r0, particle)
    p = particle.bragg_p

    zeta = (r0 - z) / sigma
    eff_dist = np.maximum(r0 - z, 0.0)
    plateau_term = 1.0 / np.power(eff_dist + 0.5 * sigma, 1.0 - 1.0 / p)

    erfc_arg = -zeta / math.sqrt(2.0)
    erfc_term = 0.5 * (1.0 - np.vectorize(math.erf)(erfc_arg))

    distal_mask = z > r0
    distal_falloff = np.exp(-0.5 * (np.maximum(0.0, z - r0) / sigma) ** 2)

    peak_boost = 1.0 + 3.0 * np.exp(-0.5 * ((z - (r0 - 0.2 * sigma)) / (1.0 * sigma)) ** 2)
    dose_upstream = plateau_term * peak_boost * erfc_term
    dose_downstream = (dose_upstream + 1.0) * distal_falloff

    dose = np.where(distal_mask, dose_downstream, dose_upstream)
    dose = np.where(z > r0 + 4.0 * sigma, 0.0, dose)

    dz = (z[1] - z[0]) if len(z) > 1 else 0.01
    total_int = np.sum(dose) * dz
    if total_int > 0.0:
        dose = dose * (energy_mev * fluence / total_int)

    return dose


def multiple_coulomb_scattering_highland(
    depth_step_cm: float,
    energy_mev: float,
    particle: HadronParticle = PROTON,
    rad_length_cm: float = WATER_RAD_LENGTH_CM,
) -> float:
    if depth_step_cm <= 0.0 or energy_mev <= 0.1:
        return 0.0

    _, _, beta, momentum = relativistic_kinematics(energy_mev, particle.mass_mev)
    if momentum <= 0.0 or beta <= 0.0:
        return 0.0

    ratio = depth_step_cm / rad_length_cm
    if ratio <= 0.0:
        return 0.0

    p_beta = momentum * beta
    log_term = 1.0 + (1.0 / 9.0) * math.log10(ratio)
    theta_0 = (14.1 / p_beta) * abs(particle.charge_z) * math.sqrt(ratio) * log_term
    return max(0.0, theta_0)


def lateral_beam_spread_sigma(
    depth_cm: float,
    energy_mev: float,
    initial_sigma_cm: float = 0.3,
    particle: HadronParticle = PROTON,
    steps: int = 100,
) -> float:
    if depth_cm <= 0.0:
        return initial_sigma_cm

    r0 = range_from_energy(energy_mev, particle)
    if depth_cm >= r0:
        depth_cm = r0

    zs = np.linspace(0.0, depth_cm, steps)
    dz = zs[1] - zs[0] if steps > 1 else depth_cm

    p = particle.bragg_p
    accumulated_variance = initial_sigma_cm ** 2

    for i in range(len(zs) - 1):
        z_curr = zs[i]
        fraction_rem = max(0.001, 1.0 - z_curr / r0)
        res_energy = energy_mev * (fraction_rem ** (1.0 / p))

        th0 = multiple_coulomb_scattering_highland(dz, res_energy, particle)
        lever_arm = depth_cm - z_curr
        accumulated_variance += (lever_arm ** 2) * (th0 ** 2)

    return math.sqrt(accumulated_variance)
