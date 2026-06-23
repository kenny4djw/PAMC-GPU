"""JAX GPU-accelerated grid builder with PBC minimum-image cutoff.

Replaces _replicate_xy + all-pairs with direct minimum-image convention
and a cutoff radius, reducing surface atoms from ~23k back to ~2.5k.
All-pairs with masking on GPU — still dense compute but 9× fewer pairs.

Usage (drop-in for grid_pbc.build_grids_pbc):
    grids = build_grids_cutoff(surface, cell=(Lx, Ly), r_cut=1.2, ...)
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from ptmc.energy.grid_build import FieldGrids, _axis, _chunk_coordinates
from ptmc.config import GRID_BUILD_CHUNK_ELEMS


@jax.jit
def _grid_chunk_cutoff(grid_pts, surf_pos, sqrt_c12, sqrt_c6, surf_q,
                        coulomb_factor, lambda_D, cell, r_cut):
    """PBC minimum-image distance with cutoff mask (GPU kernel).

    grid_pts  (C, 3)   chunk of grid-point coordinates
    surf_pos  (M, 3)   surface atom positions (single unit cell)
    cell      (2,)     (Lx, Ly) in nm
    r_cut     float    cutoff radius (nm)
    """
    d = grid_pts[:, None, :] - surf_pos[None, :, :]       # (C, M, 3)
    # Minimum-image convention in xy
    d_xy = d[..., :2]
    d_xy = d_xy - cell[None, None, :] * jnp.round(d_xy / cell[None, None, :])
    d_z = d[..., 2:3]
    r2 = jnp.sum(d_xy * d_xy, axis=-1) + d_z[..., 0] * d_z[..., 0]  # (C, M)
    r = jnp.sqrt(jnp.maximum(r2, 1e-18))
    mask = r < r_cut
    inv_r6 = jnp.where(mask, r ** -6, 0.0)
    debye = jnp.where(mask, jnp.exp(-r / lambda_D) / r, 0.0)
    G12 = jnp.dot(inv_r6 * inv_r6, sqrt_c12)              # (C,)
    G6  = jnp.dot(inv_r6, sqrt_c6)
    phi = jnp.dot(debye, surf_q * coulomb_factor)
    return G12, G6, phi


def build_grids_cutoff(surface, x_range, y_range, z_range,
                        spacing, cell, r_cut, cap_g12=1e10):
    """Build G12/G6/phi grids with PBC minimum-image cutoff.

    Parameters
    ----------
    surface : DiscreteSurface
        Single unit-cell surface (NO _replicate_xy needed).
    x_range, y_range, z_range : (lo, hi)
    spacing : float
        Grid spacing (nm).
    cell : (Lx, Ly)
        Unit cell dimensions (nm).
    r_cut : float
        Cutoff radius (nm). 1.2 nm recommended for Debye-screened Coulomb.
    cap_g12 : float or None
        Cap G12 values to avoid float overflow near surface atoms.
        Default 1e10 (safe in FP32 after multiplication by sqrt(C12)~1e-3).
        Pass None to disable capping (NOT recommended on the FP32 JAX path).
    """
    if np.isscalar(spacing):
        dx = dy = dz = float(spacing)
    else:
        dx, dy, dz = (float(s) for s in spacing)

    z_lo = max(z_range[0], surface.z_min)
    xs = _axis(x_range[0], x_range[1], dx)
    ys = _axis(y_range[0], y_range[1], dy)
    zs = _axis(z_lo, z_range[1], dz)

    nx, ny, nz = xs.size, ys.size, zs.size
    P = nx * ny * nz
    M = surface.m

    pos_j = jnp.asarray(surface.pos.astype(np.float32))
    sqrt_c12_j = jnp.asarray(np.sqrt(surface.c12).astype(np.float32))
    sqrt_c6_j = jnp.asarray(np.sqrt(surface.c6).astype(np.float32))
    q_j = jnp.asarray(surface.q.astype(np.float32))
    cf = float(surface.coulomb_factor)
    ld = float(surface.lambda_D)
    cell_j = jnp.asarray([float(cell[0]), float(cell[1])], dtype=jnp.float32)
    rc = float(r_cut)

    G12 = np.empty(P, dtype=np.float32)
    G6 = np.empty(P, dtype=np.float32)
    phi = np.empty(P, dtype=np.float32)

    # Chunk by grid points: same formula as grid_build.py so both builders
    # respect PTMC_GRID_BUILD_CHUNK_ELEMS.  Coordinates are generated PER CHUNK
    # via grid_build._chunk_coordinates — peak host memory is bounded by
    # chunk * 3 * 4 bytes, not the full (P, 3) coordinate tensor.
    chunk = max(1, int(GRID_BUILD_CHUNK_ELEMS // max(surface.m, 1)))

    for s in range(0, P, chunk):
        e = min(s + chunk, P)
        pts_chunk = _chunk_coordinates(xs, ys, zs, s, e).astype(np.float32)
        pts_j = jnp.asarray(pts_chunk)
        g12, g6, ph = _grid_chunk_cutoff(pts_j, pos_j, sqrt_c12_j, sqrt_c6_j,
                                          q_j, cf, ld, cell_j, rc)
        G12[s:e] = np.asarray(g12)
        G6[s:e] = np.asarray(g6)
        phi[s:e] = np.asarray(ph)

    if cap_g12 is not None:
        G12 = np.minimum(G12, cap_g12)

    shape = (nx, ny, nz)
    return FieldGrids(
        G12=G12.astype(np.float64).reshape(shape),
        G6=G6.astype(np.float64).reshape(shape),
        phi=phi.astype(np.float64).reshape(shape),
        origin=np.array([xs[0], ys[0], zs[0]]),
        spacing=np.array([dx, dy, dz]),
        z_min=surface.z_min,
    )
