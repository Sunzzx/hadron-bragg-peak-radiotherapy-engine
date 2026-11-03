"""Intensity-Modulated Proton Therapy (IMPT) Multi-Field Inverse Treatment Planner.

Optimizes non-negative spot fluences across multiple gantry angles using projected
gradient descent with quadratic penalties for target coverage and organ sparing.
"""

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Tuple
import numpy as np

from .physics import (
    PROTON,
    HadronParticle,
    energy_from_range,
    range_from_energy,
)
from .pencil_beam import PatientCTVolume, PencilBeam


@dataclass
class BeamField:
    gantry_angle_deg: float
    source_distance_cm: float
    isocenter_cm: np.ndarray
    spots: List[PencilBeam] = field(default_factory=list)


@dataclass
class PlanOptimizationResult:
    dose_3d: np.ndarray
    ptv_mean_dose: float
    ptv_min_dose: float
    ptv_max_dose: float
    oar_max_doses: Dict[str, float]
    oar_mean_doses: Dict[str, float]
    objective_history: List[float]
    num_spots: int
    total_monitor_units: float


class InverseTreatmentPlanner:
    def __init__(
        self,
        volume: PatientCTVolume,
        prescribed_dose_gy: float = 60.0,
        oar_limits: Optional[Dict[str, float]] = None,
        particle: HadronParticle = PROTON,
    ):
        self.volume = volume
        self.prescribed_dose_gy = prescribed_dose_gy
        self.oar_limits = oar_limits or {"brainstem": 15.0, "optic_chiasm": 20.0}
        self.particle = particle

    def generate_spot_lattice(
        self,
        gantry_angles_deg: List[float],
        spot_spacing_cm: float = 0.8,
        layer_spacing_cm: float = 0.5,
        sad_cm: float = 100.0,
    ) -> List[PencilBeam]:
        """Generate a 3D scanning spot grid covering the PTV from specified gantry angles."""
        X, Y, Z = self.volume.grid.mesh_coordinates()
        ptv_mask = self.volume.ptv_mask

        if not np.any(ptv_mask):
            raise ValueError("PatientCTVolume contains no PTV target voxels")

        ptv_x = X[ptv_mask]
        ptv_y = Y[ptv_mask]
        ptv_z = Z[ptv_mask]

        isocenter = np.array([np.mean(ptv_x), np.mean(ptv_y), np.mean(ptv_z)], dtype=np.float64)
        all_spots: List[PencilBeam] = []

        for angle_deg in gantry_angles_deg:
            rad = math.radians(angle_deg)
            dir_x = -math.sin(rad)
            dir_y = -math.cos(rad)
            dir_z = 0.0
            direction = np.array([dir_x, dir_y, dir_z], dtype=np.float64)
            direction = direction / np.linalg.norm(direction)

            source_pos = isocenter - direction * sad_cm
            lateral_u = np.array([math.cos(rad), -math.sin(rad), 0.0], dtype=np.float64)
            lateral_v = np.array([0.0, 0.0, 1.0], dtype=np.float64)

            rel_pts = np.stack([ptv_x - source_pos[0], ptv_y - source_pos[1], ptv_z - source_pos[2]], axis=1)
            t_pts = rel_pts @ direction
            u_pts = rel_pts @ lateral_u
            v_pts = rel_pts @ lateral_v

            u_min, u_max = np.min(u_pts), np.max(u_pts)
            v_min, v_max = np.min(v_pts), np.max(v_pts)
            t_min, t_max = np.min(t_pts), np.max(t_pts)

            u_grid = np.arange(u_min - 0.2, u_max + 0.3, spot_spacing_cm)
            v_grid = np.arange(v_min - 0.2, v_max + 0.3, spot_spacing_cm)
            t_grid = np.arange(t_min, t_max + 0.4, layer_spacing_cm)

            for u_val in u_grid:
                for v_val in v_grid:
                    ray_entry = source_pos + direction * (sad_cm - 8.0) + lateral_u * u_val + lateral_v * v_val
                    for t_val in t_grid:
                        depth_in_tissue = max(1.0, t_val - (sad_cm - 8.0))
                        energy = energy_from_range(depth_in_tissue, self.particle)

                        spot = PencilBeam(
                            entry_point=ray_entry,
                            direction=direction,
                            energy_mev=energy,
                            weight=1.0,
                            particle=self.particle,
                            initial_sigma_cm=0.25,
                        )
                        all_spots.append(spot)

        return all_spots

    def optimize_plan(
        self,
        spots: List[PencilBeam],
        max_iterations: int = 150,
        learning_rate: Optional[float] = None,
        ptv_weight: float = 2.0,
        oar_weight: float = 5.0,
        body_weight: float = 0.2,
        reg_weight: float = 1e-4,
    ) -> PlanOptimizationResult:
        """Run constrained inverse optimization to find optimal spot weights."""
        num_spots = len(spots)
        if num_spots == 0:
            raise ValueError("No spots provided for plan optimization")

        spot_doses: List[np.ndarray] = []
        for s in spots:
            s.weight = 1.0
            spot_doses.append(s.compute_dose_3d(self.volume))

        ptv_mask = self.volume.ptv_mask
        body_mask = self.volume.body_mask if self.volume.body_mask is not None else np.ones_like(ptv_mask)
        normal_tissue_mask = body_mask & (~ptv_mask)

        n_ptv = max(1, int(np.sum(ptv_mask)))
        n_normal = max(1, int(np.sum(normal_tissue_mask)))

        ptv_doses = np.array([d[ptv_mask] for d in spot_doses])  # [num_spots x n_ptv]
        normal_doses = np.array([d[normal_tissue_mask] for d in spot_doses])

        oar_doses_dict = {
            name: (np.array([d[mask] for d in spot_doses]), max(1, int(np.sum(mask))), self.oar_limits.get(name, 20.0))
            for name, mask in self.volume.oar_masks.items()
            if np.any(mask)
        }

        # Compute exact Lipschitz bound for PTV quadratic term
        H_ptv = (2.0 * ptv_weight / n_ptv) * (ptv_doses @ ptv_doses.T)
        norm_h = float(np.linalg.norm(H_ptv, 2))
        if learning_rate is None:
            alpha = 1.0 / (norm_h + 1e-4)
        else:
            alpha = min(learning_rate, 1.0 / (norm_h + 1e-4))

        weights = np.zeros(num_spots, dtype=np.float64)
        b_ptv = (2.0 * ptv_weight / n_ptv) * (ptv_doses @ np.full(n_ptv, self.prescribed_dose_gy))

        velocity = np.zeros_like(weights)
        momentum = 0.6
        obj_history: List[float] = []

        for iteration in range(max_iterations):
            d_ptv = ptv_doses.T @ weights
            d_normal = normal_doses.T @ weights

            ptv_err = d_ptv - self.prescribed_dose_gy
            loss_ptv = np.mean(ptv_err ** 2)
            grad_ptv = (2.0 / n_ptv) * (ptv_doses @ ptv_err)

            loss_normal = np.mean(d_normal ** 2)
            grad_normal = (2.0 / n_normal) * (normal_doses @ d_normal)

            loss_oar_total = 0.0
            grad_oar_total = np.zeros_like(weights)
            for name, (oar_mat, n_oar, limit) in oar_doses_dict.items():
                d_oar = oar_mat.T @ weights
                excess = np.maximum(0.0, d_oar - limit)
                loss_oar_total += np.mean(excess ** 2)
                grad_oar_total += (2.0 / n_oar) * (oar_mat @ excess)

            loss_reg = np.sum(weights ** 2)
            grad_reg = 2.0 * weights

            total_loss = (
                ptv_weight * loss_ptv
                + oar_weight * loss_oar_total
                + body_weight * loss_normal
                + reg_weight * loss_reg
            )
            obj_history.append(float(total_loss))

            total_grad = (
                ptv_weight * grad_ptv
                + oar_weight * grad_oar_total
                + body_weight * grad_normal
                + reg_weight * grad_reg
            )

            velocity = momentum * velocity + alpha * total_grad
            weights = np.maximum(0.0, weights - velocity)

        final_dose_3d = np.zeros(self.volume.grid.shape, dtype=np.float64)
        for w, d in zip(weights, spot_doses):
            if w > 0.0:
                final_dose_3d += w * d

        ptv_doses_final = final_dose_3d[ptv_mask]
        ptv_mean = float(np.mean(ptv_doses_final)) if len(ptv_doses_final) > 0 else 0.0
        ptv_min = float(np.min(ptv_doses_final)) if len(ptv_doses_final) > 0 else 0.0
        ptv_max = float(np.max(ptv_doses_final)) if len(ptv_doses_final) > 0 else 0.0

        oar_max = {
            name: float(np.max(final_dose_3d[mask])) if np.any(mask) else 0.0
            for name, mask in self.volume.oar_masks.items()
        }
        oar_mean = {
            name: float(np.mean(final_dose_3d[mask])) if np.any(mask) else 0.0
            for name, mask in self.volume.oar_masks.items()
        }

        return PlanOptimizationResult(
            dose_3d=final_dose_3d,
            ptv_mean_dose=ptv_mean,
            ptv_min_dose=ptv_min,
            ptv_max_dose=ptv_max,
            oar_max_doses=oar_max,
            oar_mean_doses=oar_mean,
            objective_history=obj_history,
            num_spots=num_spots,
            total_monitor_units=float(np.sum(weights)),
        )
