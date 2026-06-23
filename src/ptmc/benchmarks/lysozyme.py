"""Benchmark: charged-protein orientation on a charged surface (lysozyme-like).

Reduced model: a coarse protein carrying a charge anisotropy (a positive patch
along body +x; net ~0 dipole) on a homogeneous charged surface (linearized-PB
surface potential psi0). vdW is isotropic so ELECTROSTATICS alone sets the
orientation. Literature (e.g. Kubiak-Ossowska & Mulheran; Romanowska et al.):
lysozyme's charge anisotropy makes its adsorption orientation switch with the
surface charge sign / ionic strength. Here the positive patch faces an
oppositely-charged surface, and the preferred orientation flips when psi0 flips.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from ptmc.benchmarks._energy import (
    make_continuum_energy, coarse_protein, _fibonacci_sphere,
)
from ptmc.mc.metropolis import run_chains, split_chain_keys
from ptmc.analysis.orientation import contact_normal
from ptmc.config import beta as beta_of

PATCH = np.array([1.0, 0.0, 0.0])     # body-frame positive-charge face (+x)


def build_model(r: float = 0.5, n: int = 12, q0: float = 0.8):
    xyz = _fibonacci_sphere(n, r)
    q = q0 * (xyz[:, 0] / r)           # linear charge dipole along +x
    return coarse_protein(xyz, q, np.full(n, 3e-3), np.full(n, 3e-6))


def sample_contact_normals(psi0: float, C: int = 192, steps: int = 4000,
                           seed: int = 0, T: float = 300.0) -> np.ndarray:
    atoms = build_model()
    E = make_continuum_energy(atoms, 50.0, 3e-3, 3e-6, 0.78, 0.30, psi0)
    keys = split_chain_keys(jax.random.PRNGKey(seed), C)
    qn = jax.random.normal(jax.random.PRNGKey(seed + 1), (C, 4))
    q0 = qn / jnp.linalg.norm(qn, axis=1, keepdims=True)
    t0 = jnp.tile(jnp.array([0., 0, 0.95]), (C, 1))
    out = run_chains(keys, q0, t0, E, float(beta_of(T)), 0.35, 0.04,
                     jnp.array([0., 0, 1.]), steps)
    qf = np.array(out["quat"][:, -1])
    return np.array([contact_normal(q) for q in qf])


def patch_alignment(psi0: float, **kw) -> float:
    """Mean alignment of the surface-facing body direction with the +charge
    patch. >0: positive patch faces surface; <0: faces away."""
    cn = sample_contact_normals(psi0, **kw)
    return float((cn @ PATCH).mean())
