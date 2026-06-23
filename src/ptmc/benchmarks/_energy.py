"""JAX continuum-surface energy for benchmark sampling + coarse protein builder.

Physics matches energy/reference.py ContinuumSurface (homogeneous half-space):
  vdW  : Steele 9-3   (pi rho/45) C12' z^-9 - (pi rho/6) C6' z^-3
  elec : linearized-PB surface potential  q_i psi0 exp(-z/lambda_D)
expressed in JAX so the MC core can sample orientations. Orientation dependence
enters through the atom z-distribution produced by the pose rotation.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from ptmc.model.structures import Atoms
from ptmc.mc.moves import quat_rotate


def make_continuum_energy(atoms: Atoms, rho_s, c6_surf, c12_surf,
                          lambda_D, z_min, psi0):
    """Return energy_fn(quat (4,), trans (3,)) -> scalar kJ/mol (JAX).

    Pre-computes combined LJ coefficients so steele_energy_jax is called
    directly without re-deriving c6p/c12p/cA/cB on every evaluation.
    """
    from ptmc.energy.reference import steele_energy_jax

    pos0 = jnp.asarray(atoms.pos0)
    q = jnp.asarray(atoms.q)
    c6p = jnp.sqrt(jnp.asarray(atoms.c6) * c6_surf)
    c12p = jnp.sqrt(jnp.asarray(atoms.c12) * c12_surf)
    cA = jnp.pi * rho_s / 45.0
    cB = jnp.pi * rho_s / 6.0

    def energy_fn(quat, trans):
        return steele_energy_jax(quat, trans, pos0, q, c6p, c12p, cA, cB,
                                 lambda_D, z_min, psi0)

    return energy_fn


def _fibonacci_sphere(n, r):
    """n approximately-even points on a sphere of radius r (nm). (n,3)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    gold = np.pi * (1 + 5 ** 0.5)
    theta = gold * i
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    return r * np.stack([x, y, z], axis=1)


def coarse_protein(beads_xyz, q, c6, c12) -> Atoms:
    """Build a coarse Atoms model (one bead per residue)."""
    n = beads_xyz.shape[0]
    return Atoms(pos0=beads_xyz, q=np.asarray(q, float),
                 c6=np.asarray(c6, float), c12=np.asarray(c12, float),
                 names=[f"B{i}" for i in range(n)],
                 resids=np.arange(n), resnames=["BEA"] * n,
                 elements=["C"] * n)
