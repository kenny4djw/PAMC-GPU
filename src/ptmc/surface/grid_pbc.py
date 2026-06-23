"""Periodic-boundary-aware grid builder for DiscreteSurface + grid-energy path."""
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from ptmc.model.structures import DiscreteSurface
from ptmc.energy.grid_build import FieldGrids
from ptmc.energy.grid_build_jax import build_grids_cutoff
from ptmc.energy.grid_energy import grid_energy_positions


def _replicate_xy(surface: DiscreteSurface, cell: NDArray,
                  r_cut: float) -> DiscreteSurface:
    """Replicate surface atoms in xy to cover r_cut (3×3+ supercell)."""
    if r_cut <= 0:
        return surface
    Lx, Ly = float(cell[0]), float(cell[1])
    n_rep = max(3, int(np.ceil(r_cut / min(Lx, Ly))) + 1)
    k = (n_rep - 1) // 2
    offsets = np.array([[ix*Lx, iy*Ly, 0.0]
                        for ix in range(-k, k + 1)
                        for iy in range(-k, k + 1)], dtype=np.float64)
    pos_rep = (surface.pos[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    return DiscreteSurface(
        pos=pos_rep,
        q=np.tile(surface.q, len(offsets)),
        c6=np.tile(surface.c6, len(offsets)),
        c12=np.tile(surface.c12, len(offsets)),
        lambda_D=surface.lambda_D, z_min=surface.z_min,
        coulomb_factor=surface.coulomb_factor,
    )


def build_grids_pbc(surface: DiscreteSurface, cell: tuple,
                    r_cut: float, z_range: tuple,
                    spacing: float, cap_g12=None,
                    tail_correction: bool = True) -> FieldGrids:
    """Build energy grids with PBC minimum-image cutoff (no _replicate_xy).

    Parameters
    ----------
    surface : DiscreteSurface
        Single unit-cell surface (no PBC replication needed).
    cell : (Lx, Ly) in nm
    r_cut : cutoff radius (nm)
    z_range : (z_lo, z_hi) OR (z_lo, z_mid, z_pad)
        Two-tuple (preferred): grid covers z in [z_lo, z_hi] directly.
        Three-tuple (legacy): upper bound is computed as z_mid + z_pad.
    spacing : grid spacing (nm)
    cap_g12 : max G12 value (prevents overflow near surface)
    """
    Lx, Ly = cell
    if len(z_range) == 2:
        z_lo, z_hi = z_range
    elif len(z_range) == 3:
        z_lo, z_mid, z_pad = z_range
        z_hi = z_mid + z_pad
    else:
        raise ValueError(
            f"z_range must be a 2-tuple (z_lo, z_hi) or legacy 3-tuple "
            f"(z_lo, z_mid, z_pad); got length {len(z_range)}")
    return build_grids_cutoff(surface,
        x_range=(0.0, Lx), y_range=(0.0, Ly),
        z_range=(z_lo, z_hi),
        spacing=spacing, cell=(Lx, Ly), r_cut=r_cut, cap_g12=cap_g12)


def grid_energy_check_pbc(pose, atoms, grids: FieldGrids,
                           cell: tuple) -> float:
    """Grid energy with xy periodic wrapping of protein atom positions."""
    Lx, Ly = cell
    pos = pose.apply(atoms.pos0)
    pos[:, 0] = pos[:, 0] % Lx
    pos[:, 1] = pos[:, 1] % Ly
    return grid_energy_positions(pos, atoms, grids)
