"""Intra-protein non-bonded energy: LJ + Debye-Hueckel Coulomb.

EXPERIMENTAL — DO NOT QUOTE. This module is reached only from the
``MCConfig(flexible=True, flexible_ack_experimental=True)`` opt-in path.
The LJ combining is geometric (comb-rule 1/3), which is inconsistent with
the amber99sb-ildn topology (comb-rule 2, Lorentz-Berthelot). Quantitative
ΔG⁰_ads and orientation results obtained through this path are not
validated against literature and must not be reported as such until the
LB upgrade is performed. The validated rigid-body §S4–§S5 results in the
manuscript never invoke this path.

Implements § 5.2 of the design doc.

Functional form (kJ/mol, GROMACS units, comb-rule 1 geometric LJ):

    E_LJ(r)   = C12_ij / r^12 - C6_ij / r^6
    E_elec(r) = f * q_i q_j / r * exp(-r / lambda_D)
    E_pair    = scale_lj[i,j] * E_LJ + scale_qq[i,j] * E_elec
    E_total   = sum_{i < j} E_pair  (self / 1-2 / 1-3 zeroed via scale)

Implementation:
    - Chunked along the i axis. Chunk count from PTMC_INTRA_NB_CHUNK_ELEMS
      (max elements per chunk = chunk_rows * N).
    - r_floor = R_FLOOR_NM = 0.1 nm clamps the FP32-overflow regime at the
      hard-sphere wall -- physically equivalent to a hard repulsion shell.
    - The reduction sum is promoted to FP64 to keep the F5 gate (< 1e-3
      kJ/mol vs numpy reference) achievable with FP32 positions.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ptmc.config import COULOMB_FACTOR_KJ_NM_PER_E2
from ptmc.flexible.excl_table import ExclusionTable, build_exclusion_table
from ptmc.io.parse_topology import parse_topology

R_FLOOR_NM: float = 0.1

INTRA_NB_CHUNK_ELEMS: int = int(
    os.environ.get("PTMC_INTRA_NB_CHUNK_ELEMS", "200_000")
)
_INTRA_NB_FP64: bool = os.environ.get("PTMC_INTRA_NB_FP64", "").lower() in (
    "1", "true", "yes"
)


@dataclass(frozen=True)
class IntraNBParams:
    """Static per-system intra-NB parameters.

    SHAPES: q, sqrt_c6, sqrt_c12 (N,) float32; scale_lj/scale_qq (N,N) float32;
    fudges and scalars.
    """
    q: np.ndarray
    sqrt_c6: np.ndarray
    sqrt_c12: np.ndarray
    scale_lj: np.ndarray
    scale_qq: np.ndarray
    lambda_D: float
    r_floor: float = R_FLOOR_NM
    coulomb_factor: float = COULOMB_FACTOR_KJ_NM_PER_E2

    @property
    def n_atoms(self) -> int:
        return int(self.q.shape[0])

    def __post_init__(self) -> None:
        N = self.n_atoms
        assert self.sqrt_c6.shape == (N,)
        assert self.sqrt_c12.shape == (N,)
        assert self.scale_lj.shape == (N, N)
        assert self.scale_qq.shape == (N, N)


def build_intra_nb_params(top_path: str, lambda_D: float = 0.785,
                          excl: ExclusionTable | None = None,
                          ) -> IntraNBParams:
    """Build intra-NB params from a .top.

    Pulls per-atom q, C6, C12 via the existing comb-rule-aware parser, derives
    sqrt(C6) and sqrt(C12) for geometric pair combination, and attaches the
    exclusion / 1-4 scaling tables.
    """
    topo = parse_topology(top_path)
    if excl is None:
        excl = build_exclusion_table(top_path)
    return IntraNBParams(
        q=topo.q.astype(np.float32),
        sqrt_c6=np.sqrt(np.maximum(topo.c6, 0.0)).astype(np.float32),
        sqrt_c12=np.sqrt(np.maximum(topo.c12, 0.0)).astype(np.float32),
        scale_lj=excl.scale_lj,
        scale_qq=excl.scale_qq,
        lambda_D=float(lambda_D),
    )


# ---------------------------------------------------------------------------
# JAX kernel: chunked all-pairs
# ---------------------------------------------------------------------------

def _chunk_size_for(n: int, chunk_elems: int | None) -> int:
    """Per-chunk row count given the per-call element budget."""
    if chunk_elems is None:
        chunk_elems = INTRA_NB_CHUNK_ELEMS
    rows = max(1, chunk_elems // max(n, 1))
    return min(rows, n)


def E_intra_nb(pos: jnp.ndarray, params: IntraNBParams,
               chunk_elems: int | None = None) -> jnp.ndarray:
    """Total intra-protein non-bonded energy in kJ/mol.

    pos        : (N, 3)
    params     : pre-built IntraNBParams.
    chunk_elems: per-call cap on (chunk_rows * N) elements. Defaults to
                 INTRA_NB_CHUNK_ELEMS (env-tunable).

    Returns a scalar with dtype matching the sum accumulator (FP64 when
    jax_enable_x64 is on, else FP32).
    """
    pos = jnp.asarray(pos)
    n = params.n_atoms
    q = jnp.asarray(params.q)
    sqrt_c6 = jnp.asarray(params.sqrt_c6)
    sqrt_c12 = jnp.asarray(params.sqrt_c12)
    scale_lj = jnp.asarray(params.scale_lj)
    scale_qq = jnp.asarray(params.scale_qq)

    cf = params.coulomb_factor
    lam_inv = 1.0 / params.lambda_D
    r_floor = params.r_floor

    j_range = jnp.arange(n)
    acc_dtype = jnp.float64 if _INTRA_NB_FP64 else jnp.float32
    total = jnp.zeros((), dtype=acc_dtype)

    chunk = _chunk_size_for(n, chunk_elems)
    for i_start in range(0, n, chunk):
        i_end = min(i_start + chunk, n)
        i_idx = jnp.arange(i_start, i_end)            # (M,)
        pos_i = pos[i_idx]                            # (M, 3)
        d = pos_i[:, None, :] - pos[None, :, :]       # (M, N, 3)
        r2 = jnp.sum(d * d, axis=-1)                  # (M, N)
        r = jnp.sqrt(jnp.maximum(r2, 0.0))
        r_safe = jnp.maximum(r, r_floor)
        inv_r = 1.0 / r_safe
        inv_r6 = inv_r ** 6
        inv_r12 = inv_r6 * inv_r6
        C6 = sqrt_c6[i_idx, None] * sqrt_c6[None, :]    # (M, N)
        C12 = sqrt_c12[i_idx, None] * sqrt_c12[None, :]
        e_lj = C12 * inv_r12 - C6 * inv_r6
        e_qq = cf * q[i_idx, None] * q[None, :] * inv_r * jnp.exp(-r_safe * lam_inv)
        triu = (i_idx[:, None] < j_range[None, :])
        e_pair = (scale_lj[i_idx, :] * e_lj
                  + scale_qq[i_idx, :] * e_qq) * triu
        total = total + e_pair.astype(acc_dtype).sum()
    return total


# ---------------------------------------------------------------------------
# Numpy reference (full N x N, FP64) for F5 cross-validation
# ---------------------------------------------------------------------------

def E_intra_nb_numpy(pos: np.ndarray, params: IntraNBParams) -> float:
    """Reference: full (N, N) FP64 evaluation. O(N^2) memory."""
    pos = np.asarray(pos, dtype=np.float64)
    n = params.n_atoms
    q = params.q.astype(np.float64)
    sqrt_c6 = params.sqrt_c6.astype(np.float64)
    sqrt_c12 = params.sqrt_c12.astype(np.float64)
    scale_lj = params.scale_lj.astype(np.float64)
    scale_qq = params.scale_qq.astype(np.float64)

    d = pos[:, None, :] - pos[None, :, :]
    r2 = (d * d).sum(-1)
    r = np.sqrt(np.maximum(r2, 0.0))
    r_safe = np.maximum(r, params.r_floor)
    inv_r = 1.0 / r_safe
    inv_r6 = inv_r ** 6
    inv_r12 = inv_r6 * inv_r6
    C6 = sqrt_c6[:, None] * sqrt_c6[None, :]
    C12 = sqrt_c12[:, None] * sqrt_c12[None, :]
    e_lj = C12 * inv_r12 - C6 * inv_r6
    e_qq = (params.coulomb_factor * q[:, None] * q[None, :] * inv_r
            * np.exp(-r_safe / params.lambda_D))
    triu = np.triu(np.ones((n, n), dtype=bool), k=1)
    e = (scale_lj * e_lj + scale_qq * e_qq) * triu
    return float(e.sum())
