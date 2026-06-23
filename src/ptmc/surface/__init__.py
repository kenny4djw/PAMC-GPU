"""ptmc.surface — full-atomistic solid-surface construction for PTMC-GPU.

Typical usage::

    from ptmc.surface import build_surface, CLAYFF
    surface, grids = build_surface("rutile_tio2", hkl=(1, 1, 0), n_layers=4)
"""
from .lattice import (
    Lattice, cell_from_params,
    alpha_quartz, rutile_tio2, anatase_tio2, gold_fcc,
    get_crystal, BUILTIN_CRYSTALS,
)
from .slab import cut_slab, hydroxylate_quartz, hydroxylate_rutile
from .forcefield import FFParams, CLAYFF, assign_ff, lattice_to_surface
from .grid_pbc import build_grids_pbc, grid_energy_check_pbc, _replicate_xy
from .builder import build_surface

__all__ = [
    "Lattice", "cell_from_params",
    "alpha_quartz", "rutile_tio2", "anatase_tio2", "gold_fcc",
    "get_crystal", "BUILTIN_CRYSTALS",
    "cut_slab", "hydroxylate_quartz", "hydroxylate_rutile",
    "FFParams", "CLAYFF", "assign_ff", "lattice_to_surface",
    "build_grids_pbc", "grid_energy_check_pbc", "_replicate_xy",
    "build_surface",
]
