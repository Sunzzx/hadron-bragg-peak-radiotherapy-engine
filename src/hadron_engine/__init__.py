"""Hadron Bragg Peak Radiotherapy Engine."""

from .physics import (
    CARBON_ION,
    PROTON,
    HadronParticle,
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
from .pencil_beam import (
    PatientCTVolume,
    PencilBeam,
    TissueProperties,
    VoxelGrid,
    TISSUE_AIR,
    TISSUE_BONE,
    TISSUE_BRAIN,
    TISSUE_LUNG,
    TISSUE_TUMOR,
    TISSUE_WATER,
)
from .sobp import (
    SOBP,
    optimize_sobp_weights,
)
from .planner import (
    BeamField,
    InverseTreatmentPlanner,
    PlanOptimizationResult,
)
from .dvh import (
    DVHCurve,
    calculate_conformity_index,
    calculate_dvh,
    calculate_homogeneity_index,
    calculate_rbe_dose,
)
from .visualizer import (
    render_ct_dose_slice_ascii,
    render_depth_dose_ascii,
    render_dvh_ascii,
)

__version__ = "0.1.0"

__all__ = [
    "HadronParticle",
    "PROTON",
    "CARBON_ION",
    "relativistic_kinematics",
    "max_energy_transfer",
    "sternheimer_density_effect",
    "bethe_bloch_stopping_power",
    "range_from_energy",
    "energy_from_range",
    "range_straggling_sigma",
    "bortfeld_depth_dose",
    "multiple_coulomb_scattering_highland",
    "lateral_beam_spread_sigma",
    "TissueProperties",
    "TISSUE_AIR",
    "TISSUE_LUNG",
    "TISSUE_WATER",
    "TISSUE_BRAIN",
    "TISSUE_TUMOR",
    "TISSUE_BONE",
    "VoxelGrid",
    "PatientCTVolume",
    "PencilBeam",
    "SOBP",
    "optimize_sobp_weights",
    "BeamField",
    "PlanOptimizationResult",
    "InverseTreatmentPlanner",
    "DVHCurve",
    "calculate_dvh",
    "calculate_homogeneity_index",
    "calculate_conformity_index",
    "calculate_rbe_dose",
    "render_depth_dose_ascii",
    "render_ct_dose_slice_ascii",
    "render_dvh_ascii",
]
