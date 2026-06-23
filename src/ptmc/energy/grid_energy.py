"""Grid-interpolated single-pose energy.

Trilinear interpolation of the three fields, then the factorized sum
    E = sum_i [ sqrt(C12_i) G12(r_i) - sqrt(C6_i) G6(r_i) + q_i phi(r_i) ].
Hard wall -> +inf if any atom has z < grids.z_min.
"""
from __future__ import annotations

import numpy as np

from ptmc.model.structures import Atoms, Pose
from ptmc.energy.grid_build import FieldGrids


def _trilinear(field, pos, origin, spacing):
    """Trilinear-interpolate field (nx,ny,nz) at points pos (N,3) -> (N,)."""
    nx, ny, nz = field.shape
    g = (pos - origin) / spacing
    lower = np.array([0.0, 0.0, 0.0])
    upper = np.array([nx - 1, ny - 1, nz - 1])
    if np.any(g < lower - 1e-12) or np.any(g > upper + 1e-12):
        raise ValueError(f"position outside grid: g in [{g.min(0)}, {g.max(0)}], "
                         f"grid range [0, {nx - 1}] x [0, {ny - 1}] x [0, {nz - 1}]")
    i0 = np.floor(g).astype(np.int64)
    i0[:, 0] = np.clip(i0[:, 0], 0, nx - 2)
    i0[:, 1] = np.clip(i0[:, 1], 0, ny - 2)
    i0[:, 2] = np.clip(i0[:, 2], 0, nz - 2)
    f = g - i0
    fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
    ix, iy, iz = i0[:, 0], i0[:, 1], i0[:, 2]
    c000 = field[ix, iy, iz];         c100 = field[ix + 1, iy, iz]
    c010 = field[ix, iy + 1, iz];     c110 = field[ix + 1, iy + 1, iz]
    c001 = field[ix, iy, iz + 1];     c101 = field[ix + 1, iy, iz + 1]
    c011 = field[ix, iy + 1, iz + 1]; c111 = field[ix + 1, iy + 1, iz + 1]
    c00 = c000 * (1 - fx) + c100 * fx
    c01 = c001 * (1 - fx) + c101 * fx
    c10 = c010 * (1 - fx) + c110 * fx
    c11 = c011 * (1 - fx) + c111 * fx
    c0 = c00 * (1 - fy) + c10 * fy
    c1 = c01 * (1 - fy) + c11 * fy
    return c0 * (1 - fz) + c1 * fz


def grid_energy_positions(pos, atoms: Atoms, grids: FieldGrids) -> float:
    """Energy (kJ/mol) of protein atoms at lab positions pos (N,3) via grids."""
    if np.any(pos[:, 2] < grids.z_min):
        return np.inf
    G12 = _trilinear(grids.G12, pos, grids.origin, grids.spacing)
    G6 = _trilinear(grids.G6, pos, grids.origin, grids.spacing)
    phi = _trilinear(grids.phi, pos, grids.origin, grids.spacing)
    e = atoms.sqrt_c12 * G12 - atoms.sqrt_c6 * G6 + atoms.q * phi
    return float(np.sum(e))


def grid_energy(pose: Pose, atoms: Atoms, grids: FieldGrids) -> float:
    """Single-pose grid energy: apply pose, interpolate, factorized sum."""
    return grid_energy_positions(pose.apply(atoms.pos0), atoms, grids)


# ---------------------------------------------------------------------------
# JAX-compatible trilinear interpolation for the agarose dual-potential model.
# Used in the MC hot loop (JIT-compiled, no numpy/scipy in the loop).
# ---------------------------------------------------------------------------

def _trilinear_indices_jax(field_shape, pos, origin, spacing):
    """Common index/weight computation for trilinear interpolation.

    Returns ``(ix, iy, iz, ix1, iy1, iz1, fx, fy, fz)`` so callers can share
    one index computation across multiple co-located fields (G6/G12/phi).
    """
    import jax.numpy as jnp

    nx, ny, nz = field_shape
    g = (pos - origin) / spacing
    g = jnp.clip(g, 0.0,
                 jnp.array([nx - 1.001, ny - 1.001, nz - 1.001],
                           dtype=g.dtype))
    i0 = jnp.floor(g).astype(jnp.int32)
    i1 = i0 + 1
    f = g - i0

    i0 = jnp.clip(i0, 0, jnp.array([nx - 2, ny - 2, nz - 2], dtype=jnp.int32))
    i1 = jnp.clip(i1, 1, jnp.array([nx - 1, ny - 1, nz - 1], dtype=jnp.int32))
    return (i0[:, 0], i0[:, 1], i0[:, 2],
            i1[:, 0], i1[:, 1], i1[:, 2],
            f[:, 0], f[:, 1], f[:, 2])


def _trilinear_apply(field, ix, iy, iz, ix1, iy1, iz1, fx, fy, fz):
    """Apply trilinear interpolation to a single field given pre-computed indices."""
    c000 = field[ix, iy, iz];     c100 = field[ix1, iy, iz]
    c010 = field[ix, iy1, iz];    c110 = field[ix1, iy1, iz]
    c001 = field[ix, iy, iz1];    c101 = field[ix1, iy, iz1]
    c011 = field[ix, iy1, iz1];   c111 = field[ix1, iy1, iz1]

    c00 = c000 * (1 - fx) + c100 * fx
    c01 = c001 * (1 - fx) + c101 * fx
    c10 = c010 * (1 - fx) + c110 * fx
    c11 = c011 * (1 - fx) + c111 * fx
    c0 = c00 * (1 - fy) + c10 * fy
    c1 = c01 * (1 - fy) + c11 * fy
    return c0 * (1 - fz) + c1 * fz


def _trilinear_jax(field, pos, origin, spacing):
    """Trilinear interpolation in JAX. SHAPE: field (nx,ny,nz), pos (N,3) -> (N,).

    Clips positions to the grid extents.  Origin and spacing are (3,) arrays.
    """
    ix, iy, iz, ix1, iy1, iz1, fx, fy, fz = _trilinear_indices_jax(
        field.shape, pos, origin, spacing)
    return _trilinear_apply(field, ix, iy, iz, ix1, iy1, iz1, fx, fy, fz)


def _trilinear_jax_stack(fields_stack, pos, origin, spacing):
    """Trilinear interpolation of K co-located fields with shared indices.

    SHAPE: fields_stack (K, nx, ny, nz), pos (N, 3) -> (K, N).

    Computes the floor/clip/weight arithmetic ONCE, then gathers the 8 corner
    values for all K fields in a single indexed load. Returns one row per
    input field — the caller picks them apart. Used by the full-atom surface
    energy where G12, G6, phi share the same grid geometry.
    """
    K, nx, ny, nz = fields_stack.shape
    ix, iy, iz, ix1, iy1, iz1, fx, fy, fz = _trilinear_indices_jax(
        (nx, ny, nz), pos, origin, spacing)
    # Gather 8 corners across all K fields at once: shape (K, N).
    c000 = fields_stack[:, ix, iy, iz];     c100 = fields_stack[:, ix1, iy, iz]
    c010 = fields_stack[:, ix, iy1, iz];    c110 = fields_stack[:, ix1, iy1, iz]
    c001 = fields_stack[:, ix, iy, iz1];    c101 = fields_stack[:, ix1, iy, iz1]
    c011 = fields_stack[:, ix, iy1, iz1];   c111 = fields_stack[:, ix1, iy1, iz1]

    c00 = c000 * (1 - fx) + c100 * fx
    c01 = c001 * (1 - fx) + c101 * fx
    c10 = c010 * (1 - fx) + c110 * fx
    c11 = c011 * (1 - fx) + c111 * fx
    c0 = c00 * (1 - fy) + c10 * fy
    c1 = c01 * (1 - fy) + c11 * fy
    return c0 * (1 - fz) + c1 * fz


def patterned_energy_jax(quat, trans, pos0, q, c6p, c12p, cA, cB,
                          lamD, z_min, phi_field, grid_origin, grid_spacing):
    """Steele 9-3 vdW (analytic) + grid-interpolated phi (patterned elec).

    Steele 9-3 uses the same analytic formula as the homogeneous surface.
    The electrostatic term uses trilinear interpolation of a precomputed
    phi(x,y,z) grid, enabling laterally heterogeneous surface charge patterns.

    SHAPE: pos0 (N,3), q (N,), c6p/c12p (N,), cA/cB scalars.
    phi_field (nx,ny,nz), grid_origin (3,), grid_spacing (3,).
    Returns scalar energy (kJ/mol).
    """
    import jax.numpy as jnp
    from ptmc.mc.moves import quat_rotate

    pos = quat_rotate(quat, pos0) + trans
    return patterned_energy_from_pos_jax(pos, q, c6p, c12p, cA, cB, lamD, z_min,
                                          phi_field, grid_origin, grid_spacing)


def patterned_energy_from_pos_jax(pos, q, c6p, c12p, cA, cB,
                                   lamD, z_min, phi_field, grid_origin, grid_spacing):
    """Patterned continuum energy evaluated at lab-frame positions ``pos``.

    Factorised from ``patterned_energy_jax`` for flexible (chi-aware) closures.
    SHAPE: pos (N,3), q/c6p/c12p (N,); phi_field (nx,ny,nz); scalars cA, cB, lamD, z_min.
    """
    import jax.numpy as jnp

    z = pos[:, 2]
    wall = jnp.any(z < z_min)
    zc = jnp.maximum(z, z_min + 1e-3)

    # Steele 9-3 vdW (identical to homogeneous case)
    e_vdw = jnp.sum(cA * c12p / zc ** 9 - cB * c6p / zc ** 3)

    # Grid-interpolated electrostatics
    phi_vals = _trilinear_jax(phi_field, pos, grid_origin, grid_spacing)
    e_el = jnp.sum(q * phi_vals)

    total = e_vdw + e_el
    return jnp.where(wall, jnp.array(jnp.inf, dtype=total.dtype), total)


def agarose_energy_jax(quat, trans, pos0, q,
                        U_steric_field, phi_elec_field,
                        grid_origin, grid_spacing):
    """3D agarose gel energy: Gaussian steric + Yukawa electrostatic."""
    import jax.numpy as jnp
    from ptmc.mc.moves import quat_rotate

    pos = quat_rotate(quat, pos0) + trans
    return agarose_energy_from_pos_jax(pos, q, U_steric_field, phi_elec_field,
                                        grid_origin, grid_spacing)


def agarose_energy_from_pos_jax(pos, q,
                                 U_steric_field, phi_elec_field,
                                 grid_origin, grid_spacing):
    """3D agarose gel energy evaluated at lab-frame positions ``pos``.

    Factorised from ``agarose_energy_jax`` for flexible (chi-aware) closures.
    SHAPE: pos (N,3); q (N,); U_steric/phi_elec (nx,ny,nz); origin/spacing (3,).
    """
    import jax.numpy as jnp

    fields_stack = jnp.stack([U_steric_field, phi_elec_field], axis=0)
    vals = _trilinear_jax_stack(fields_stack, pos, grid_origin, grid_spacing)
    e_steric = jnp.sum(vals[0])
    e_elec = jnp.sum(q * vals[1])

    return e_steric + e_elec


# ---------------------------------------------------------------------------
# Full-atom surface grid energy (JAX): G12/G6/phi trilinear interpolation
# with optional xy PBC wrapping.  Used by PT/PA closures and trajectory.
# ---------------------------------------------------------------------------

def grid_energy_from_pos_jax(pos, sqrt_c12, sqrt_c6, q,
                              G12_field, G6_field, phi_field,
                              grid_origin, grid_spacing, z_min,
                              cell_Lx: float = 0.0, cell_Ly: float = 0.0):
    """Full-atom surface energy evaluated at lab-frame positions ``pos``.

    Factorised from ``grid_energy_jax`` for flexible (chi-aware) closures.
    SHAPE: pos (N,3); sqrt_c12/sqrt_c6/q (N,); G12/G6/phi (nx,ny,nz);
    grid_origin/spacing (3,); scalars z_min, cell_Lx, cell_Ly.
    """
    import jax.numpy as jnp

    # Wrap xy into surface unit cell when lateral PBC is requested.
    if cell_Lx > 0.0 and cell_Ly > 0.0:
        pos = jnp.concatenate([
            (pos[:, :1] % cell_Lx),
            (pos[:, 1:2] % cell_Ly),
            pos[:, 2:3],
        ], axis=-1)

    z = pos[:, 2]
    wall = jnp.any(z < z_min)

    fields_stack = jnp.stack([G12_field, G6_field, phi_field], axis=0)
    vals = _trilinear_jax_stack(fields_stack, pos, grid_origin, grid_spacing)
    G12_v, G6_v, phi_v = vals[0], vals[1], vals[2]
    e = jnp.dot(sqrt_c12, G12_v) - jnp.dot(sqrt_c6, G6_v) + jnp.dot(q, phi_v)
    return jnp.where(wall, jnp.array(jnp.inf, dtype=e.dtype), e)


def grid_energy_jax(quat, trans, pos0, sqrt_c12, sqrt_c6, q,
                    G12_field, G6_field, phi_field,
                    grid_origin, grid_spacing, z_min,
                    cell_Lx: float = 0.0, cell_Ly: float = 0.0):
    """Full-atom surface energy from precomputed (G12, G6, phi) grids (JAX)."""
    import jax.numpy as jnp
    from ptmc.mc.moves import quat_rotate

    pos = quat_rotate(quat, pos0) + trans
    return grid_energy_from_pos_jax(pos, sqrt_c12, sqrt_c6, q,
                                     G12_field, G6_field, phi_field,
                                     grid_origin, grid_spacing, z_min,
                                     cell_Lx, cell_Ly)


def agarose_surface_energy_jax(quat, trans, pos0, q,
                               U_steric_field, phi_elec_field,
                               grid_origin, grid_spacing, z_min):
    """Flat gel-coated surface energy: agarose fields + hard wall at z_min."""
    import jax.numpy as jnp
    from ptmc.mc.moves import quat_rotate

    pos = quat_rotate(quat, pos0) + trans
    return agarose_surface_energy_from_pos_jax(pos, q, U_steric_field, phi_elec_field,
                                                grid_origin, grid_spacing, z_min)


def agarose_surface_energy_from_pos_jax(pos, q,
                                         U_steric_field, phi_elec_field,
                                         grid_origin, grid_spacing, z_min):
    """Flat gel-coated surface energy evaluated at lab-frame positions ``pos``.

    Factorised from ``agarose_surface_energy_jax`` for flexible (chi-aware) closures.
    SHAPE: pos (N,3); q (N,); U_steric/phi_elec (nx,ny,nz); origin/spacing (3,); z_min scalar.
    """
    import jax.numpy as jnp

    z = pos[:, 2]
    wall = jnp.any(z < z_min)

    fields_stack = jnp.stack([U_steric_field, phi_elec_field], axis=0)
    vals = _trilinear_jax_stack(fields_stack, pos, grid_origin, grid_spacing)
    e_steric = jnp.sum(vals[0])
    e_elec = jnp.sum(q * vals[1])

    total = e_steric + e_elec
    return jnp.where(wall, jnp.array(jnp.inf, dtype=total.dtype), total)
