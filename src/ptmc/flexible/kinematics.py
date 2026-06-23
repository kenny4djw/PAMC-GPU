"""Forward kinematics for chi rotations (JAX, jit-friendly).

Two entry points:
- ``apply_all_chi`` -- residue-parallel: 4 depth steps × R residues in one fused
  per-atom gather. O(D · N) work, no (R, N) intermediate. Production path.
- ``apply_all_chi_serial`` -- naive K-step serial loop, used as the ground-truth
  reference in F2 equivalence tests.

Both routines operate from the original ``pos0`` each call (no carry of
accumulated coordinates -- avoids FP32 chain drift, keeps state minimal).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from ptmc.flexible.schedule import ChiSchedule
from ptmc.flexible.topology import ChiTopology

# A small floor on the axis norm to keep gradients / division finite when
# pos[k] == pos[j] (degenerate, never happens in real geometry but XLA traces
# the path).
_AXIS_NORM_FLOOR = 1e-30


def axis_angle_to_matrix(axis: jnp.ndarray, angle: jnp.ndarray) -> jnp.ndarray:
    """Rodrigues rotation matrix.

    axis: (..., 3) unit vectors; angle: (...,) radians.
    Returns (..., 3, 3).
    """
    c = jnp.cos(angle)
    s = jnp.sin(angle)
    t = 1.0 - c
    x = axis[..., 0]
    y = axis[..., 1]
    z = axis[..., 2]
    row0 = jnp.stack([t * x * x + c,     t * x * y - s * z, t * x * z + s * y], axis=-1)
    row1 = jnp.stack([t * x * y + s * z, t * y * y + c,     t * y * z - s * x], axis=-1)
    row2 = jnp.stack([t * x * z - s * y, t * y * z + s * x, t * z * z + c    ], axis=-1)
    return jnp.stack([row0, row1, row2], axis=-2)


def _rotate_about(pos: jnp.ndarray, pivot: jnp.ndarray,
                  R: jnp.ndarray) -> jnp.ndarray:
    """Apply rotation R about pivot to each row of pos.

    pos: (N, 3), pivot: (3,), R: (3, 3).
    """
    return (pos - pivot) @ R.T + pivot


def apply_chi_step(pos: jnp.ndarray, i: int, j: int, k: int, l: int,
                   chi: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Rotate ``mask``-selected atoms about the j-k bond by angle ``chi``.

    pos: (N, 3); chi: scalar; mask: (N,) bool.
    Atoms outside the mask are returned unchanged. ``i`` and ``l`` are unused
    by the rotation itself (they only contribute to the dihedral *measurement*),
    but kept in the signature so callers can pass full chi rows.
    """
    del i, l  # not needed for application; only for measurement
    axis_vec = pos[k] - pos[j]
    axis_norm = jnp.linalg.norm(axis_vec)
    axis = axis_vec / jnp.maximum(axis_norm, _AXIS_NORM_FLOOR)
    R = axis_angle_to_matrix(axis, chi)
    pos_rot = _rotate_about(pos, pos[k], R)
    return jnp.where(mask[:, None], pos_rot, pos)


def apply_all_chi_serial(pos0: jnp.ndarray, chi: jnp.ndarray,
                         topo: ChiTopology) -> jnp.ndarray:
    """Naive K-step serial baseline.

    Uses Python iteration (K traced into the XLA graph as K rotations).
    Slow at runtime once K is large; used in F2 equivalence tests only.
    """
    chi_idx = jnp.asarray(topo.chi_idx)
    chi_mask = jnp.asarray(topo.chi_mask)
    K = topo.k

    pos = pos0
    for k in range(K):
        pos = apply_chi_step(
            pos,
            int(topo.chi_idx[k, 0]),
            int(topo.chi_idx[k, 1]),
            int(topo.chi_idx[k, 2]),
            int(topo.chi_idx[k, 3]),
            chi[k],
            chi_mask[k],
        )
    return pos


def _apply_depth_step(pos: jnp.ndarray, chi: jnp.ndarray,
                      j_idx: jnp.ndarray, k_idx: jnp.ndarray,
                      chi_global: jnp.ndarray, valid: jnp.ndarray,
                      owner: jnp.ndarray) -> jnp.ndarray:
    """One depth-step: apply all residues' chi rotations at this depth.

    Per-atom gather of (R, pivot) by owner index. Atoms with owner == -1 are
    untouched.

    j_idx, k_idx, chi_global, valid: (R,) -- one entry per flex residue.
    owner: (N,) int32 with -1 sentinel.
    """
    pos_j = pos[j_idx]              # (R, 3)
    pos_k = pos[k_idx]              # (R, 3)
    axis_vec = pos_k - pos_j
    axis_norm = jnp.linalg.norm(axis_vec, axis=-1, keepdims=True)
    axis = axis_vec / jnp.maximum(axis_norm, _AXIS_NORM_FLOOR)
    angle = jnp.where(valid, chi[chi_global], 0.0)  # (R,)
    R = axis_angle_to_matrix(axis, angle)           # (R, 3, 3)

    # Per-atom gather. Safe index 0 for sentinel; needs_rot then suppresses.
    needs_rot = owner >= 0                          # (N,)
    safe_owner = jnp.where(needs_rot, owner, 0)
    R_atom = R[safe_owner]                          # (N, 3, 3)
    pivot_atom = pos_k[safe_owner]                  # (N, 3)

    diff = pos - pivot_atom
    pos_rot = jnp.einsum('nij,nj->ni', R_atom, diff) + pivot_atom
    return jnp.where(needs_rot[:, None], pos_rot, pos)


def measure_dihedrals(pos: jnp.ndarray, chi_idx: jnp.ndarray) -> jnp.ndarray:
    """Measure dihedral angles (i,j,k,l) given current geometry.

    Praxeolitic atan2 formulation -- numerically stable, returns values in
    (-π, π]. Sign convention matches the IUPAC right-hand rule: if the i→j→k→l
    chain bends "right-handed" around the j-k axis, the angle is positive.

    pos: (N, 3); chi_idx: (K, 4) int32. Returns (K,) float (same dtype as pos).

    Identity used by F3 round-trip tests:
        measure_dihedrals(apply_all_chi(pos0, chi)) - measure_dihedrals(pos0)
        ≡ chi   (mod 2π)
    """
    i = chi_idx[:, 0]
    j = chi_idx[:, 1]
    k = chi_idx[:, 2]
    l = chi_idx[:, 3]
    b1 = pos[j] - pos[i]
    b2 = pos[k] - pos[j]
    b3 = pos[l] - pos[k]
    b2n = b2 / jnp.maximum(jnp.linalg.norm(b2, axis=-1, keepdims=True),
                           _AXIS_NORM_FLOOR)
    # Project b1 and b3 onto plane perpendicular to b2
    v = b1 - (b1 * b2n).sum(-1, keepdims=True) * b2n
    w = b3 - (b3 * b2n).sum(-1, keepdims=True) * b2n
    x = (v * w).sum(-1)
    y = (jnp.cross(b2n, v) * w).sum(-1)
    return jnp.arctan2(y, x)


def apply_all_chi(pos0: jnp.ndarray, chi: jnp.ndarray,
                  schedule: ChiSchedule) -> jnp.ndarray:
    """Residue-parallel forward kinematics.

    pos0: (N, 3); chi: (K,); schedule: precomputed ChiSchedule.
    Returns: (N, 3) after applying every chi.

    Implementation: for depth d in 0..D-1, gather per-residue axes from
    current pos and apply rotations to owned atoms in one fused pass.
    """
    chi_by_depth = jnp.asarray(schedule.chi_by_depth)
    valid_by_depth = jnp.asarray(schedule.valid_by_depth)
    chi_global_idx = jnp.asarray(schedule.chi_global_idx)
    owner_per_depth = jnp.asarray(schedule.owner_per_depth)

    pos = pos0
    for d in range(schedule.max_n_chi):
        pos = _apply_depth_step(
            pos,
            chi,
            j_idx=chi_by_depth[d, :, 1],
            k_idx=chi_by_depth[d, :, 2],
            chi_global=chi_global_idx[d, :],
            valid=valid_by_depth[d, :],
            owner=owner_per_depth[d, :],
        )
    return pos
