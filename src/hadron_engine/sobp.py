"""Spread-Out Bragg Peak (SOBP) Synthesis and Non-Negative Fluence Optimization.

Modulates monoenergetic pristine hadron Bragg peaks across tumor thickness to deliver
a uniform, homogeneous target dose while eliminating downstream exit radiation.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from .physics import (
    PROTON,
    HadronParticle,
    bortfeld_depth_dose,
    energy_from_range,
    range_from_energy,
    range_straggling_sigma,
)


@dataclass
class SOBP:
    energies: np.ndarray
    weights: np.ndarray
    depths: np.ndarray
    composite_dose: np.ndarray
    pristine_doses: np.ndarray
    target_proximal_cm: float
    target_distal_cm: float
    prescribed_dose_gy: float
    plateau_homogeneity: float  # (D_max - D_min) / D_mean in target [%]

    @property
    def num_layers(self) -> int:
        return len(self.energies)


def optimize_sobp_weights(
    target_proximal_cm: float,
    target_distal_cm: float,
    prescribed_dose_gy: float = 60.0,
    particle: HadronParticle = PROTON,
    num_layers: int = 15,
    max_iter: int = 500,
    tolerance: float = 1e-5,
) -> SOBP:
    """Synthesize an optimal Spread-Out Bragg Peak covering [z_proximal, z_distal]."""
    if target_proximal_cm >= target_distal_cm:
        raise ValueError("target_proximal_cm must be strictly less than target_distal_cm")

    ranges = np.linspace(target_proximal_cm, target_distal_cm, num_layers)
    energies = np.array([energy_from_range(r, particle) for r in ranges])

    z_max = target_distal_cm + 2.5
    num_depths = 350
    depths = np.linspace(0.0, z_max, num_depths)

    A = np.zeros((num_depths, num_layers), dtype=np.float64)
    for j, e in enumerate(energies):
        A[:, j] = bortfeld_depth_dose(depths, e, particle, fluence=1.0)

    target_mask = (depths >= target_proximal_cm) & (depths <= target_distal_cm)
    distal_mask = depths > (target_distal_cm + 0.2)

    A_target = A[target_mask, :]
    A_distal = A[distal_mask, :]
    d_target = np.full(int(np.sum(target_mask)), prescribed_dose_gy, dtype=np.float64)

    # Objective: || A_target w - d_target ||^2 + 8.0 * || A_distal w ||^2 + 1e-4 * ||w||^2
    H = (A_target.T @ A_target) + 8.0 * (A_distal.T @ A_distal) + 1e-4 * np.eye(num_layers)
    b = A_target.T @ d_target

    L = float(np.linalg.norm(H, 2))
    step = 1.0 / (L + 1e-6)

    # Projected Nesterov Accelerated Gradient Descent
    w = np.zeros(num_layers, dtype=np.float64)
    y = np.zeros(num_layers, dtype=np.float64)
    t_prev = 1.0

    for it in range(max_iter):
        grad = H @ y - b
        w_next = np.maximum(0.0, y - step * grad)

        t_next = (1.0 + np.sqrt(1.0 + 4.0 * t_prev ** 2)) / 2.0
        y = w_next + ((t_prev - 1.0) / t_next) * (w_next - w)
        
        diff = np.linalg.norm(w_next - w) / (np.linalg.norm(w) + 1e-9)
        w = w_next
        t_prev = t_next
        if diff < tolerance:
            break

    composite_dose = A @ w
    target_doses = composite_dose[target_mask]
    current_mean = np.mean(target_doses) if len(target_doses) > 0 else 1.0
    if current_mean > 0:
        scale = prescribed_dose_gy / current_mean
        w = w * scale
        composite_dose = composite_dose * scale

    target_doses = composite_dose[target_mask]
    if len(target_doses) > 0:
        homogeneity = float((np.max(target_doses) - np.min(target_doses)) / np.mean(target_doses) * 100.0)
    else:
        homogeneity = 0.0

    pristine_weighted = A * w[np.newaxis, :]

    return SOBP(
        energies=energies,
        weights=w,
        depths=depths,
        composite_dose=composite_dose,
        pristine_doses=pristine_weighted,
        target_proximal_cm=target_proximal_cm,
        target_distal_cm=target_distal_cm,
        prescribed_dose_gy=prescribed_dose_gy,
        plateau_homogeneity=homogeneity,
    )
