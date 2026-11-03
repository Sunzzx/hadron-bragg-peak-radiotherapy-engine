"""Dose-Volume Histogram (DVH) Computation, ICRU Clinical Indices, and Radiobiological RBE."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np

from .physics import PROTON, CARBON_ION, HadronParticle


@dataclass
class DVHCurve:
    structure_name: str
    dose_bins: np.ndarray  # [Gy]
    volume_fractions: np.ndarray  # Cumulative volume percentage [0 to 100%]
    mean_dose: float
    min_dose: float
    max_dose: float
    d98: float
    d50: float
    d2: float

    def v_at_dose(self, dose_gy: float) -> float:
        """Percentage of structure volume receiving at least dose_gy (V_D metric)."""
        return float(np.interp(dose_gy, self.dose_bins, self.volume_fractions, right=0.0))


def calculate_dvh(
    dose_3d: np.ndarray,
    mask_3d: np.ndarray,
    structure_name: str = "Structure",
    num_bins: int = 250,
    max_dose_gy: Optional[float] = None,
) -> DVHCurve:
    """Calculate cumulative Dose-Volume Histogram for a segmented anatomical structure."""
    if not np.any(mask_3d):
        return DVHCurve(
            structure_name=structure_name,
            dose_bins=np.zeros(num_bins),
            volume_fractions=np.zeros(num_bins),
            mean_dose=0.0,
            min_dose=0.0,
            max_dose=0.0,
            d98=0.0,
            d50=0.0,
            d2=0.0,
        )

    voxel_doses = dose_3d[mask_3d]
    total_voxels = len(voxel_doses)

    min_d = float(np.min(voxel_doses))
    max_d = float(np.max(voxel_doses))
    mean_d = float(np.mean(voxel_doses))

    upper_limit = max_dose_gy if max_dose_gy is not None else max(max_d * 1.05, 1.0)
    bins = np.linspace(0.0, upper_limit, num_bins)

    # Cumulative volume calculation
    # Fraction of voxels receiving >= D
    sorted_doses = np.sort(voxel_doses)
    # Count voxels >= bin
    counts = total_voxels - np.searchsorted(sorted_doses, bins, side="left")
    vol_fracs = (counts / total_voxels) * 100.0

    # ICRU 83 metrics: D98 (near-min), D50 (median), D2 (near-max)
    # D98 is dose exceeded by 98% of volume -> 2nd percentile of dose
    d98 = float(np.percentile(voxel_doses, 2.0))
    d50 = float(np.percentile(voxel_doses, 50.0))
    d2 = float(np.percentile(voxel_doses, 98.0))

    return DVHCurve(
        structure_name=structure_name,
        dose_bins=bins,
        volume_fractions=vol_fracs,
        mean_dose=mean_d,
        min_dose=min_d,
        max_dose=max_d,
        d98=d98,
        d50=d50,
        d2=d2,
    )


def calculate_homogeneity_index(ptv_dvh: DVHCurve) -> float:
    """Compute ICRU 83 Homogeneity Index: HI = (D2 - D98) / D50.

    An ideal uniform dose distribution yields HI = 0.0.
    """
    if ptv_dvh.d50 <= 0.0:
        return 0.0
    return float((ptv_dvh.d2 - ptv_dvh.d98) / ptv_dvh.d50)


def calculate_conformity_index(
    dose_3d: np.ndarray,
    ptv_mask: np.ndarray,
    body_mask: np.ndarray,
    prescribed_dose_gy: float,
    isodose_fraction: float = 0.95,
) -> float:
    """Compute Paddick Conformity Index (CI).

    Formula:
        CI = (V_{PTV, ref})^2 / (V_{PTV} * V_{Total, ref})

    Where ref is typically the 95% isodose volume.
    Ideal value is 1.0 (perfect conformity).
    """
    ref_dose = prescribed_dose_gy * isodose_fraction
    dose_ref_mask = dose_3d >= ref_dose

    v_ptv_ref = np.sum(ptv_mask & dose_ref_mask)
    v_ptv_total = np.sum(ptv_mask)
    v_total_ref = np.sum(body_mask & dose_ref_mask)

    if v_ptv_total == 0 or v_total_ref == 0:
        return 0.0

    ci = (v_ptv_ref ** 2) / (v_ptv_total * v_total_ref)
    return float(ci)


def calculate_rbe_dose(
    physical_dose_3d: np.ndarray,
    particle: HadronParticle = PROTON,
    let_kev_um: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Convert physical absorbed dose [Gy] to biologically effective dose [Gy(RBE)].

    Protons: Constant RBE = 1.1 (standard clinical practice, ICRU 78).
    Carbon-12: LET-dependent biological weighting model.
    """
    if particle.name == PROTON.name:
        return physical_dose_3d * 1.1
    elif particle.name == CARBON_ION.name:
        # Carbon ions have variable RBE scaling with Linear Energy Transfer (LET)
        # Default LET estimated from Bragg peak stopping power if not provided
        if let_kev_um is None:
            # Rough proxy LET (keV/um) from physical dose relative peak
            max_p = np.max(physical_dose_3d) + 1e-6
            let_kev_um = 30.0 + 120.0 * (physical_dose_3d / max_p)

        # Wilkens-Oelfke / Carabe model
        alpha_let = 0.045
        beta_let = 0.0003
        rbe = 1.0 + (alpha_let * let_kev_um) / (1.0 + beta_let * (let_kev_um ** 2))
        return physical_dose_3d * rbe
    else:
        return physical_dose_3d * 1.0
