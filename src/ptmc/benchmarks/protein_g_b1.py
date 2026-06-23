"""Benchmark: protein-G-B1-like protein on a hydrophobic surface.

Reduced model: a coarse protein with TWO hydrophobic patches (high C6) on
opposite faces (body +z and -z), near-neutral charges, on a hydrophobic
(strongly vdW-attractive) surface. Literature (e.g. Latour and co-workers):
protein G B1 on hydrophobic SAMs samples multiple adsorption orientations.
Here the two patches give two near-degenerate low-free-energy basins with
mutually exclusive contact faces (contact normal near +z vs -z).
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
from ptmc.analysis.clustering import cluster_orientations
from ptmc.analysis.free_energy import basin_free_energies
from ptmc.config import beta as beta_of


def build_model(r: float = 0.5, n: int = 12):
    xyz = _fibonacci_sphere(n, r)
    c6 = np.where(np.abs(xyz[:, 2]) > 0.30, 8e-3, 1e-3)   # two hydrophobic caps
    return coarse_protein(xyz, np.zeros(n), c6, c6 * 7e-4)


def sample_contact_normals(C: int = 256, steps: int = 5000, seed: int = 0,
                           T: float = 300.0) -> np.ndarray:
    atoms = build_model()
    E = make_continuum_energy(atoms, 50.0, 8e-3, 5.6e-6, 0.78, 0.30, 0.0)
    keys = split_chain_keys(jax.random.PRNGKey(seed), C)
    qn = jax.random.normal(jax.random.PRNGKey(seed + 1), (C, 4))
    q0 = qn / jnp.linalg.norm(qn, axis=1, keepdims=True)
    t0 = jnp.tile(jnp.array([0., 0, 0.95]), (C, 1))
    out = run_chains(keys, q0, t0, E, float(beta_of(T)), 0.35, 0.04,
                     jnp.array([0., 0, 1.]), steps)
    qf = np.array(out["quat"][:, -1])
    return np.array([contact_normal(q) for q in qf])


def find_basins(C: int = 256, steps: int = 5000, seed: int = 0):
    cn = sample_contact_normals(C, steps, seed)
    labels, cents = cluster_orientations(cn, k=2, seed=seed)
    p, dG = basin_free_energies(labels, float(beta_of(300.0)), k=2)
    return cn, labels, cents, p, dG
