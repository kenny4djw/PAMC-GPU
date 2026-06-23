"""Intrinsic-frame orientation descriptors (azimuth-invariant).

The adsorption orientation must NOT be reported as absolute Euler angles. Under
a rotation of the whole system about the surface normal (lab z) by phi, the pose
rotation maps R -> Rz(phi) R and atom z-coordinates are unchanged. Two descriptors
are therefore invariant under that azimuthal rotation (and xy translation):

  * contact_normal(quat) = R^T (-z_lab): the BODY-frame direction that points
    into the surface. Rz(phi)^T z = z  =>  (Rz R)^T(-z) = R^T(-z): invariant.
  * contact_residues: residue ids whose nearest atom sits below a z cutoff;
    Rz and xy translation leave atom z unchanged, so the set is invariant.
"""
from __future__ import annotations

import numpy as np

from ptmc.model.structures import quat_to_matrix, Pose, Atoms

_ZLAB = np.array([0.0, 0.0, 1.0])


def quat_mul_np(a, b):
    """Hamilton product (w,x,y,z), numpy. SHAPE (4,)x(4,)->(4,)."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_about_z(phi):
    """Unit quaternion for a lab-frame rotation about z by phi. (4,)."""
    return np.array([np.cos(phi / 2), 0.0, 0.0, np.sin(phi / 2)])


def contact_normal(quat) -> np.ndarray:
    """Body-frame unit vector pointing into the surface (lab -z). (3,).
    Invariant under azimuthal (lab-z) rotation."""
    R = quat_to_matrix(quat)
    return R.T @ (-_ZLAB)


def lab_positions(quat, trans, atoms: Atoms) -> np.ndarray:
    """Atom lab coordinates (N,3) nm for a pose."""
    return Pose(quat=np.asarray(quat), trans=np.asarray(trans)).apply(atoms.pos0)


def contact_residues(quat, trans, atoms: Atoms, z_contact: float) -> frozenset:
    """Residue ids whose nearest atom has z < z_contact. Azimuth-invariant."""
    pos = lab_positions(quat, trans, atoms)
    z = pos[:, 2]
    resids = np.asarray(atoms.resids)
    out = {int(r) for r in np.unique(resids) if z[resids == r].min() < z_contact}
    return frozenset(out)


def tilt_angle(quat) -> float:
    """Polar tilt of the contact normal vs the surface inward normal (-z),
    in radians (intrinsic, azimuth-invariant)."""
    n = contact_normal(quat)
    return float(np.arccos(np.clip(-n[2], -1.0, 1.0)))
