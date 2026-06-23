"""Chi-aware Parallel Tempering (replica exchange) in JAX.

Extends ``parallel_tempering.py`` with a chi-aware variant where each replica
carries (quat, trans, chi) — shape (4,), (3,), (K,) — and the MC sweep uses
``scan_flexible_metropolis``.  Swap logic is unchanged (only temperatures
exchange), but swizzling must include the chi vector.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from ptmc.mc.flex_metropolis import (
    dof_move_weights,
    scan_flexible_metropolis,
)


def _pt_flex_chain(key, quat, trans, chi, energy_fn, beta,
                     sigma_rot, sigma_trans, sigma_chi,
                     axis_mask, move_weights, n_steps):
    """One flexible replica sweep (no trajectory)."""
    out = scan_flexible_metropolis(
        key, quat, trans, chi,
        energy_fn, beta,
        sigma_rot, sigma_trans, sigma_chi,
        axis_mask, move_weights,
        n_steps, start_step=None,
    )
    return (out["quat_final"], out["trans_final"], out["chi_final"],
            out["energy_final"])


_pt_flex_sweep_inner = jax.vmap(
    _pt_flex_chain, in_axes=(0, 0, 0, 0, None, 0, None, None, None, None, None, None))


@partial(jax.jit, static_argnums=(4, 11))
def pt_flex_sweep(keys, quats, transs, chis, energy_fn, betas,
                    sigma_rot, sigma_trans, sigma_chi,
                    axis_mask, move_weights, n_steps):
    """Run every flexible replica n_steps at its own beta.

    SHAPE: keys (B,2), quats (B,4), transs (B,3), chis (B,K), betas (B,).
    Returns (quat, trans, chi, energy) each (B, …).
    """
    return _pt_flex_sweep_inner(keys, quats, transs, chis,
                                  energy_fn, betas,
                                  sigma_rot, sigma_trans, sigma_chi,
                                  axis_mask, move_weights, n_steps)


@jax.jit
def swap_pass_flex(key, quats, transs, chis, energies, betas, parity):
    """One odd/even replica-exchange pass including chi swizzling.

    SHAPE: quats (R,4), transs (R,3), chis (R,K), energies (R,), betas (R,).
    ``parity`` is 0 (even) or 1 (odd).  Returns swapped (quats, transs, chis,
    energies) and accepted (max_pairs,).

    Forked from ``swap_pass`` in ``parallel_tempering.py``; S=1 (single-system)
    is the common case for the user-facing CLI pipeline.
    """
    R = energies.shape[0]

    # Pre-compute both parity cases.
    left_0 = jnp.arange(0, R - 1, 2)
    left_1 = jnp.arange(1, R - 1, 2)
    n_0, n_1 = left_0.shape[0], left_1.shape[0]
    max_pairs = max(n_0, n_1)
    pad_0, pad_1 = max_pairs - n_0, max_pairs - n_1
    L0 = jnp.concatenate([left_0, jnp.full(pad_0, R - 1, dtype=jnp.int32)])
    L1 = jnp.concatenate([left_1, jnp.full(pad_1, R - 1, dtype=jnp.int32)])
    R0 = jnp.concatenate([left_0 + 1, jnp.full(pad_0, R - 1, dtype=jnp.int32)])
    R1 = jnp.concatenate([left_1 + 1, jnp.full(pad_1, R - 1, dtype=jnp.int32)])
    valid_0 = jnp.arange(max_pairs) < n_0
    valid_1 = jnp.arange(max_pairs) < n_1

    is_even = jnp.asarray(parity) == 0
    L = jnp.where(is_even, L0, L1)
    R_idx = jnp.where(is_even, R0, R1)
    valid = jnp.where(is_even, valid_0, valid_1)

    Ei = energies[L]
    Ej = energies[R_idx]
    dbeta = betas[L] - betas[R_idx]
    arg = dbeta * (Ei - Ej)

    u = jax.random.uniform(key, (max_pairs,), minval=1e-38, maxval=1.0)
    finite = jnp.isfinite(Ei) & jnp.isfinite(Ej)
    acc = (jnp.log(u) < arg) & valid & finite

    # Build per-slot source index via vectorized scatter.
    src = jnp.arange(R)
    src = src.at[L].set(jnp.where(acc, R_idx, L))
    src = src.at[R_idx].set(jnp.where(acc, L, R_idx))

    q = quats[src]
    t = transs[src]
    c = chis[src]
    e = energies[src]
    return q, t, c, e, acc


def _scatter_pair_to_gap(acc_mean, parity, R):
    """Map per-pair stats (max_pairs,) → per-gap (R-1,) with parity-aware index."""
    max_pairs = acc_mean.shape[0]
    pair_idx = parity + 2 * jnp.arange(max_pairs)
    in_range = pair_idx < (R - 1)
    safe_idx = jnp.where(in_range, pair_idx, R - 2)
    contrib = acc_mean.astype(jnp.float32) * in_range.astype(jnp.float32)
    acc_per_gap = jnp.zeros(R - 1, dtype=jnp.float32).at[safe_idx].add(contrib)
    mask_per_gap = jnp.zeros(R - 1, dtype=jnp.float32).at[safe_idx].add(
        in_range.astype(jnp.float32))
    return acc_per_gap, mask_per_gap


def run_pt_flexible(key, init_quats, init_transs, init_chis, energy_fn,
                      betas, sigma_rot, sigma_trans, sigma_chi,
                      axis_mask, move_weights=None,
                      n_rounds=200, n_sweep=50):
    """Chi-aware Parallel Tempering over a temperature ladder.

    Parameters
    ----------
    key : PRNGKey
    init_quats : (R, 4)  initial quaternions for each replica
    init_transs : (R, 3)  initial translations (nm)
    init_chis : (R, K)  initial χ dihedral angles (rad)
    energy_fn : callable (quat, trans, chi) -> scalar
    betas : (R,)  inverse temperatures for the ladder
    sigma_rot, sigma_trans, sigma_chi : float
    axis_mask : (3,) array
    move_weights : (3,) array or None
    n_rounds : int
        Number of PT rounds (MC sweep + swap).
    n_sweep : int
        MC sweeps per round per replica.

    Returns
    -------
    dict with low-T trajectory, final states, MC/swap accept rates.
    """
    R = init_quats.shape[0]
    K = init_chis.shape[1]
    betas_j = jnp.asarray(betas)
    quats0 = jnp.asarray(init_quats)
    transs0 = jnp.asarray(init_transs)
    chis0 = jnp.asarray(init_chis)
    low_T_idx = int(jnp.argmax(betas_j))

    if move_weights is None:
        move_weights = dof_move_weights(K)

    def round_body(carry, rnd):
        quats, transs, chis = carry
        round_key = jax.random.fold_in(key, rnd)
        ks, kx = jax.random.split(round_key, 2)
        sweep_keys = jax.random.split(ks, R)

        q, t, c, e = pt_flex_sweep(
            sweep_keys, quats, transs, chis,
            energy_fn, betas_j,
            sigma_rot, sigma_trans, sigma_chi,
            axis_mask, move_weights, n_sweep)

        parity = rnd % 2
        q, t, c, e, acc = swap_pass_flex(kx, q, t, c, e, betas_j, parity)

        acc_mean = acc.astype(jnp.float32)
        swap_acc_gap, swap_mask_gap = _scatter_pair_to_gap(acc_mean, parity, R)

        ys = (
            q[low_T_idx],
            t[low_T_idx],
            c[low_T_idx],
            acc_mean,
            swap_acc_gap, swap_mask_gap,
        )
        return (q, t, c), ys

    (qf, tf, cf), (low_q, low_t, low_c, mc_acc_per_round,
                    swap_acc_per_round, swap_mask_per_round) = jax.lax.scan(
        round_body, (quats0, transs0, chis0), jnp.arange(n_rounds))

    mc_accept_rate = jnp.mean(mc_acc_per_round.astype(jnp.float32), axis=0)

    swap_mask_sum = jnp.sum(swap_mask_per_round, axis=0)
    swap_accept_rate = jnp.where(
        swap_mask_sum > 0,
        jnp.sum(swap_acc_per_round, axis=0) / jnp.maximum(swap_mask_sum, 1.0),
        0.0)

    return {
        "low_T_quat": low_q,                        # (n_rounds, 4)
        "low_T_trans": low_t,                        # (n_rounds, 3)
        "low_T_chi": low_c,                          # (n_rounds, K)
        "final_quats": qf,                           # (R, 4)
        "final_transs": tf,                          # (R, 3)
        "final_chis": cf,                            # (R, K)
        "mc_accept_rate": mc_accept_rate,
        "swap_accept_rate": swap_accept_rate,
        "low_T_idx": low_T_idx,
    }
