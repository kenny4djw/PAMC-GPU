"""High-level builder: crystal name + Miller indices -> DiscreteSurface + FieldGrids."""
from __future__ import annotations

import logging
from dataclasses import replace

from .lattice import get_crystal
from .slab import cut_slab, hydroxylate_quartz, hydroxylate_rutile
from .forcefield import lattice_to_surface, CLAYFF
from .grid_pbc import build_grids_pbc
from ptmc.model.structures import DiscreteSurface
from ptmc.energy.grid_build import FieldGrids

logger = logging.getLogger(__name__)


def build_surface(crystal: str, hkl=(0, 0, 1), n_layers: int = 4,
                  vacuum: float = 3.0, hydroxylate: bool = True,
                  lambda_D: float = 0.785, z_min: float = 0.2,
                  r_cut: float = 1.4, spacing: float = 0.05,
                  z_range: tuple | None = None,
                  name: str | None = None) -> tuple[DiscreteSurface, FieldGrids]:
    """Build a full-atomistic surface and its precomputed energy grids.

    Parameters
    ----------
    crystal : "alpha_quartz" | "rutile_tio2" | "anatase_tio2" | "gold_fcc"
    hkl : Miller indices of the surface plane
    n_layers : number of atomic layers in the slab
    vacuum : vacuum gap above the slab (nm)
    hydroxylate : add OH termination (silica surfaces only)
    lambda_D : Debye screening length (nm)
    z_min : hard-wall distance from surface (nm)
    r_cut : periodic-image cutoff for grid building (nm)
    spacing : isotropic grid spacing (nm)
    """
    if crystal == "gold_fcc":
        logger.warning(
            "gold_fcc uses CLAYFF Au parameters (σ=0.254 nm, ε=2.483 kJ/mol), "
            "which differ from the GolP-CHARMM Au ε/σ and from the pure-metal "
            "Lennard-Jones parameters used by INTERFACE-FF. The explicit-Au grid "
            "may therefore produce a different ΔG⁰_ads than the continuum Au(111) "
            "model even for identical proteins; cross-compare continuum and "
            "explicit results before interpreting the explicit-Au grid as a "
            "higher-accuracy reference."
        )
    bulk = get_crystal(crystal)
    slab = cut_slab(bulk, hkl=hkl, n_layers=n_layers, vacuum=vacuum)
    if hydroxylate:
        if "quartz" in crystal:
            slab = hydroxylate_quartz(slab, oh_z_cutoff=0.35)
        elif "rutile" in crystal:
            slab = hydroxylate_rutile(slab, oh_z_cutoff=0.25)
    surface = lattice_to_surface(slab, ff=CLAYFF, lambda_D=lambda_D, z_min=z_min)
    # Shift surface so topmost atom is at z=0.  This makes z_min a proper
    # distance from the top surface plane, rather than an absolute coordinate
    # inside a slab whose bottom starts at z=0.  Use dataclasses.replace so
    # the original ``surface`` object (and its .pos buffer) stays untouched.
    z_top = float(surface.pos[:, 2].max())
    new_pos = surface.pos.copy()
    new_pos[:, 2] -= z_top
    surface = replace(surface, pos=new_pos)
    Lx, Ly = float(slab.cell[0, 0]), float(slab.cell[1, 1])
    if z_range is None:
        z_range = (z_min, z_min + 0.5, 2.0)
    grids = build_grids_pbc(surface, cell=(Lx, Ly), r_cut=r_cut,
                            z_range=z_range, spacing=spacing, cap_g12=1e6)
    return surface, grids
