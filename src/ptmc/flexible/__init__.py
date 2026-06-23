"""Semi-flexible extension: continuous side-chain chi angles.

Rigid backbone + N_chi continuous torsional DOFs per residue (§ design doc).
"""
from __future__ import annotations

from ptmc.flexible.chi_table import (
    CHI_TABLE,
    N_CHI,
    FLEXIBLE_RESIDUES,
    chi_atom_names,
    n_chi_for,
)
from ptmc.flexible.topology import ChiTopology, build_chi_topology
from ptmc.flexible.schedule import ChiSchedule, build_chi_schedule
from ptmc.flexible.kinematics import (
    apply_all_chi,
    apply_all_chi_serial,
    apply_chi_step,
    axis_angle_to_matrix,
    measure_dihedrals,
)
from ptmc.flexible.bonded import (
    BondedParams,
    E_bonded,
    E_bonded_numpy,
    build_bonded_params,
)
from ptmc.flexible.excl_table import ExclusionTable, build_exclusion_table
from ptmc.flexible.intra_nb import (
    IntraNBParams,
    E_intra_nb,
    E_intra_nb_numpy,
    build_intra_nb_params,
    R_FLOOR_NM,
)

__all__ = [
    "CHI_TABLE",
    "N_CHI",
    "FLEXIBLE_RESIDUES",
    "chi_atom_names",
    "n_chi_for",
    "ChiTopology",
    "build_chi_topology",
    "ChiSchedule",
    "build_chi_schedule",
    "apply_all_chi",
    "apply_all_chi_serial",
    "apply_chi_step",
    "axis_angle_to_matrix",
    "measure_dihedrals",
    "BondedParams",
    "build_bonded_params",
    "E_bonded",
    "E_bonded_numpy",
    "ExclusionTable",
    "build_exclusion_table",
    "IntraNBParams",
    "build_intra_nb_params",
    "E_intra_nb",
    "E_intra_nb_numpy",
    "R_FLOOR_NM",
]
