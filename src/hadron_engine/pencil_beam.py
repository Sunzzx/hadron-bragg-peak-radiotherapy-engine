"""3D Pencil Beam Dose Kernel, Water-Equivalent Path Length (WEPL) Raytracing,
and Voxelized Patient CT Volume Geometry."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from .physics import (
    PROTON,
    HadronParticle,
    bortfeld_depth_dose,
    lateral_beam_spread_sigma,
    range_from_energy,
)


@dataclass(frozen=True)
class TissueProperties:
    name: str
    density_g_cm3: float
    rsp: float  # Relative Stopping Power relative to liquid water (S / S_water)
    rad_length_cm: float


TISSUE_AIR = TissueProperties("Air", 0.0012, 0.001, 30000.0)
TISSUE_LUNG = TissueProperties("Lung", 0.26, 0.25, 138.0)
TISSUE_WATER = TissueProperties("Water", 1.0, 1.0, 36.08)
TISSUE_BRAIN = TissueProperties("Brain", 1.04, 1.03, 35.0)
TISSUE_TUMOR = TissueProperties("Tumor", 1.06, 1.05, 34.5)
TISSUE_BONE = TissueProperties("Cortical Bone", 1.85, 1.65, 10.5)


@dataclass
class VoxelGrid:
    shape: Tuple[int, int, int]  # (Nx, Ny, Nz)
    voxel_size_cm: Tuple[float, float, float]  # (dx, dy, dz)
    origin_cm: Tuple[float, float, float]  # (x0, y0, z0)

    @property
    def total_voxels(self) -> int:
        return self.shape[0] * self.shape[1] * self.shape[2]

    def coordinates_1d(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        nx, ny, nz = self.shape
        dx, dy, dz = self.voxel_size_cm
        x0, y0, z0 = self.origin_cm
        xs = x0 + np.arange(nx) * dx
        ys = y0 + np.arange(ny) * dy
        zs = z0 + np.arange(nz) * dz
        return xs, ys, zs

    def mesh_coordinates(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        xs, ys, zs = self.coordinates_1d()
        return np.meshgrid(xs, ys, zs, indexing="ij")


@dataclass
class PatientCTVolume:
    grid: VoxelGrid
    rsp_grid: np.ndarray
    ptv_mask: np.ndarray
    oar_masks: Dict[str, np.ndarray] = field(default_factory=dict)
    body_mask: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.body_mask is None:
            self.body_mask = self.rsp_grid > 0.05

    @classmethod
    def create_cranial_phantom(
        cls,
        grid_size: Tuple[int, int, int] = (60, 60, 40),
        voxel_size: Tuple[float, float, float] = (0.25, 0.25, 0.25),
        origin: Tuple[float, float, float] = (-7.5, -7.5, -5.0),
    ) -> "PatientCTVolume":
        """Creates an anatomically realistic cranial tumor phantom.

        Contains:
        - Skull bone ring
        - Normal brain parenchyma
        - Central skull-base PTV tumor
        - Adjacent critical Organ-at-Risk (OAR: Brainstem / Optic Chiasm)
        """
        grid = VoxelGrid(shape=grid_size, voxel_size_cm=voxel_size, origin_cm=origin)
        X, Y, Z = grid.mesh_coordinates()

        # Head elliptical radius in xy plane, cylindrical in z
        r_xy = np.sqrt(X ** 2 + Y ** 2)
        skull_outer = 6.8
        skull_inner = 6.0

        rsp = np.full(grid_size, TISSUE_AIR.rsp, dtype=np.float64)

        # Skull bone
        bone_mask = (r_xy <= skull_outer) & (r_xy >= skull_inner) & (np.abs(Z) <= 4.0)
        rsp[bone_mask] = TISSUE_BONE.rsp

        # Brain soft tissue
        brain_mask = (r_xy < skull_inner) & (np.abs(Z) <= 4.0)
        rsp[brain_mask] = TISSUE_BRAIN.rsp

        # Planning Target Volume (PTV tumor: cranial chordoma at (0.0, 1.2, 0.0))
        r_ptv = np.sqrt(X ** 2 + (Y - 1.2) ** 2 + Z ** 2)
        ptv_mask = r_ptv <= 1.4
        rsp[ptv_mask] = TISSUE_TUMOR.rsp

        # Organ-at-Risk (OAR 1: Brainstem at (0.0, -1.2, 0.0))
        r_brainstem = np.sqrt(X ** 2 + (Y + 1.2) ** 2 + (Z / 1.5) ** 2)
        brainstem_mask = (r_brainstem <= 1.1) & (~ptv_mask)

        # Organ-at-Risk (OAR 2: Optic Chiasm at (0.0, 3.2, 0.0))
        r_optic = np.sqrt(X ** 2 + (Y - 3.2) ** 2 + Z ** 2)
        optic_mask = (r_optic <= 0.6) & (~ptv_mask)

        body_mask = (r_xy <= skull_outer) & (np.abs(Z) <= 4.0)

        return cls(
            grid=grid,
            rsp_grid=rsp,
            ptv_mask=ptv_mask,
            oar_masks={"brainstem": brainstem_mask, "optic_chiasm": optic_mask},
            body_mask=body_mask,
        )


@dataclass
class PencilBeam:
    entry_point: np.ndarray  # [x, y, z] cm
    direction: np.ndarray  # Unit vector pointing along beam propagation
    energy_mev: float
    weight: float = 1.0  # Fluence / Monitor Units (MU)
    particle: HadronParticle = PROTON
    initial_sigma_cm: float = 0.25

    def __post_init__(self):
        self.entry_point = np.asarray(self.entry_point, dtype=np.float64)
        norm = np.linalg.norm(self.direction)
        if norm > 0:
            self.direction = np.asarray(self.direction, dtype=np.float64) / norm
        else:
            self.direction = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    def compute_dose_3d(self, volume: PatientCTVolume) -> np.ndarray:
        """Compute 3D absorbed dose distribution on the patient CT grid."""
        X, Y, Z = volume.grid.mesh_coordinates()
        r0 = self.entry_point
        u = self.direction

        # Relative position vectors from entry point
        dx = X - r0[0]
        dy = Y - r0[1]
        dz = Z - r0[2]

        # Longitudinal projection depth t along beam central axis
        t = dx * u[0] + dy * u[1] + dz * u[2]

        # Forward beam mask: particles only propagate forward (t >= 0)
        forward_mask = t >= 0.0

        # Perpendicular radial distance squared: r_perp^2 = ||r - r0||^2 - t^2
        r_sq = (dx ** 2 + dy ** 2 + dz ** 2) - (t ** 2)
        r_sq = np.maximum(0.0, r_sq)

        # Maximum range in water
        csda_range = range_from_energy(self.energy_mev, self.particle)

        # Estimate WEPL along ray. For high efficiency on 3D grid, calculate average
        # effective stopping power along the beam line
        # Slice average RSP near central axis
        mean_rsp = np.mean(volume.rsp_grid[volume.body_mask]) if np.any(volume.body_mask) else 1.0
        wepl = t * mean_rsp

        # Lateral beam spreading sigma_r(t)
        # Calculate beam spreading at representative depths
        depth_samples = np.linspace(0.0, csda_range, 25)
        sigma_samples = np.array([
            lateral_beam_spread_sigma(d, self.energy_mev, self.initial_sigma_cm, self.particle)
            for d in depth_samples
        ])

        # Interpolate lateral sigma at each voxel's depth
        clamped_depth = np.clip(wepl, 0.0, csda_range)
        sigma_field = np.interp(clamped_depth, depth_samples, sigma_samples)

        # Longitudinal Bortfeld 1D dose
        # Sample Bortfeld depth dose finely
        z_eval = np.linspace(0.0, csda_range * 1.2, 300)
        bp_eval = bortfeld_depth_dose(z_eval, self.energy_mev, self.particle, fluence=1.0)
        bp_field = np.interp(wepl, z_eval, bp_eval, left=0.0, right=0.0)

        # 2D Gaussian lateral fluence profile: (1 / (2*pi*sigma^2)) * exp(-r_perp^2 / (2*sigma^2))
        gauss_lateral = (1.0 / (2.0 * np.pi * (sigma_field ** 2))) * np.exp(-r_sq / (2.0 * (sigma_field ** 2)))

        # Combine into 3D dose kernel [Gy]
        dose_3d = bp_field * gauss_lateral * self.weight * forward_mask

        # Zero out dose outside patient body if body mask is present
        if volume.body_mask is not None:
            dose_3d = dose_3d * volume.body_mask

        return dose_3d
