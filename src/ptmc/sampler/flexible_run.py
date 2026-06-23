"""Single-system × C-chains chi-aware MC driver.

This is the F8 entry point: vmap ``scan_flexible_metropolis`` over chains, with
positional per-chain / per-step keying so (a) chain results are independent
of how chains are batched and (b) [0, k) + [k, N) is identical to [0, N)
continuously (checkpoint-resume contract, locked down in G2).

Surface-specific runners (``run_systems``, ``run_systems_grid``, ...) in
``highthroughput.py`` are NOT touched here -- they keep the rigid contract
relied on by 200+ existing tests. This module supplies a fresh, energy-fn-
parametric driver that callers can wire to any chi-aware energy:

    flexible_run_chains(
        master_seed=42,
        n_chains=C,
        n_steps=10_000,
        init_chi=jnp.zeros((K,)),
        z0=0.95,
        energy_fn=lambda q, t, c: my_total_energy(q, t, c, ...),
        beta=1.0,
        sigma_rot=0.05, sigma_trans=0.05, sigma_chi=0.2,
        axis_mask=jnp.ones(3),
        move_weights=dof_move_weights(K),
    )
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from ptmc.config import INIT_QUAT_KEY_OFFSET
from ptmc.mc.flex_metropolis import (
    dof_move_weights,
    scan_flexible_metropolis,
)


def _chain_keys(master_seed: int, n_chains: int) -> jax.Array:
    """Per-chain PRNGKeys (C, 2) via fold_in(master, c)."""
    master = jax.random.PRNGKey(master_seed)
    cidx = jnp.arange(n_chains, dtype=jnp.uint32)
    return jax.vmap(jax.random.fold_in, in_axes=(None, 0))(master, cidx)


def _initial_pose(chain_keys: jax.Array, z0: float) -> tuple:
    """Random unit quat per chain + trans=[0,0,z0]."""
    n = chain_keys.shape[0]
    qn = jax.vmap(
        lambda k: jax.random.normal(
            jax.random.fold_in(k, INIT_QUAT_KEY_OFFSET), (4,))
    )(chain_keys)
    quat0 = qn / jnp.linalg.norm(qn, axis=1, keepdims=True)
    trans0 = jnp.stack([jnp.zeros((n,)), jnp.zeros((n,)),
                        jnp.full((n,), float(z0))], axis=1)
    return quat0, trans0


def _flex_chain_runner(chain_key, quat0, trans0, chi0,
                       energy_fn, beta,
                       sigma_rot, sigma_trans, sigma_chi,
                       axis_mask, move_weights,
                       start_step, n_steps):
    """Single chain: ``scan_flexible_metropolis`` wrapped for vmap."""
    out = scan_flexible_metropolis(
        chain_key, quat0, trans0, chi0,
        energy_fn, beta,
        sigma_rot, sigma_trans, sigma_chi,
        axis_mask, move_weights,
        n_steps, start_step=start_step,
    )
    return (out["quat_final"], out["trans_final"], out["chi_final"],
            out["energy_final"], out["acc_counts"], out["try_counts"])


# vmap over chains. Per-chain: key, quat0, trans0, chi0. Everything else None.
_flex_chain_vmapped = jax.vmap(
    _flex_chain_runner,
    in_axes=(0, 0, 0, 0,
             None, None,
             None, None, None,
             None, None,
             None, None))


def flexible_run_chains(master_seed: int, n_chains: int, n_steps: int,
                        init_chi, z0, energy_fn, beta,
                        sigma_rot, sigma_trans, sigma_chi,
                        axis_mask, move_weights=None,
                        init_quat=None, init_trans=None,
                        start_step: int = 0):
    """Run ``n_chains`` independent chi-aware MC chains in one batch.

    Parameters
    ----------
    master_seed : int
    n_chains : int
        C, number of parallel chains.
    n_steps : int
        Per-chain step budget.
    init_chi : (K,) array  OR  (C, K) array
        Initial chi state. If 1D, broadcast to all C chains. Use chi=0 when
        chi is interpreted as a delta from pos0 (the standard convention).
    z0 : float
        Initial z height for the random-unit-quat initial poses.
    energy_fn : callable (quat, trans, chi) -> scalar
        Must be jit-compatible and have a stable closure.
    beta : float
        Inverse temperature (mol/kJ).
    sigma_rot, sigma_trans, sigma_chi : float
        Proposal step sizes.
    axis_mask : (3,) array
        Translation axis mask (e.g. [0, 0, 1] for z-only).
    move_weights : (3,) array or None
        Log-weights for (rot, trans, chi) move-type categorical. Default
        ``dof_move_weights(K)`` (∝ (3, 3, K)).
    init_quat, init_trans : (C, 4), (C, 3) or None
        Override the default random-unit-quat + (0, 0, z0) initial pose.
    start_step : int
        Absolute step index for positional RNG.

    Returns
    -------
    dict with
        quats          (C, 4)
        transs         (C, 3)
        chis           (C, K)
        energies       (C,)
        acc_counts     (C, 3) int32
        try_counts     (C, 3) int32
        accept_rate_per_type (C, 3) float
    """
    K = int(init_chi.shape[-1])
    if move_weights is None:
        move_weights = dof_move_weights(K)

    keys = _chain_keys(master_seed, n_chains)  # (C, 2)
    if init_quat is None or init_trans is None:
        quat0, trans0 = _initial_pose(keys, z0)
    else:
        quat0, trans0 = init_quat, init_trans

    chi0 = jnp.asarray(init_chi)
    if chi0.ndim == 1:
        chi0 = jnp.broadcast_to(chi0[None, :], (n_chains, K))

    qf, tf, cf, ef, accc, tryc = _flex_chain_vmapped(
        keys, quat0, trans0, chi0,
        energy_fn, float(beta),
        float(sigma_rot), float(sigma_trans), float(sigma_chi),
        jnp.asarray(axis_mask), jnp.asarray(move_weights),
        int(start_step), int(n_steps))
    jax.block_until_ready((qf, tf, cf, ef, accc, tryc))

    safe_try = jnp.maximum(tryc, 1).astype(jnp.float32)
    return dict(
        quats=np.asarray(qf),
        transs=np.asarray(tf),
        chis=np.asarray(cf),
        energies=np.asarray(ef),
        acc_counts=np.asarray(accc),
        try_counts=np.asarray(tryc),
        accept_rate_per_type=np.asarray(accc.astype(jnp.float32) / safe_try),
    )
