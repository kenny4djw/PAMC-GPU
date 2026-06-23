"""Parallel Tempering (replica exchange) over a temperature ladder, in JAX.

Replicas are organized as (n_systems, R): R temperatures per independent system.
Each round: every replica runs n_sweep Metropolis steps at its own beta
(per-replica beta via vmap), then adjacent-temperature replicas attempt an
exchange with odd/even pairing. Swaps happen ONLY within a system (the R axis),
so distinct system_ids never exchange (high-throughput isolation).

Swap acceptance (detailed balance): swapping replicas i,j (configs x_i,x_j) maps
weight exp(-b_i E_i - b_j E_j) -> exp(-b_i E_j - b_j E_i); the acceptance ratio
is exp((b_i - b_j)(E_i - E_j)), i.e. p_swap = min(1, exp((b_i-b_j)(E_i-E_j))).

Hot loop architecture
---------------------
The round loop is a single ``jax.lax.scan`` over n_rounds: zero Python-side
dispatch per round and a single fused XLA program. Per-round PRNG is positional
(``jax.random.fold_in(master, round)``), matching the high-throughput runner
so checkpoint-resume of PT is byte-identical to a single long run.

run_multi_pt uses a double vmap (outer=systems, inner=replicas) so per-system
arrays (pos0, q, c6p, c12p) are stored once per system rather than once per
(system × replica) chain — memory scales as O(S * N) instead of O(S * R * N).
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from ptmc.mc.metropolis import scan_one
from ptmc.energy.reference import steele_coefficients_for
from ptmc.config import INIT_QUAT_KEY_OFFSET


# ---------------------------------------------------------------------------
# Replica-exchange swap probability and one swap pass over (S, R) replicas.
# ---------------------------------------------------------------------------

def swap_probability(beta_i, beta_j, E_i, E_j):
    """Replica-exchange acceptance min(1, exp((b_i-b_j)(E_i-E_j))).

    NaN/inf safe: if either energy is non-finite the swap probability is 0
    (no exchange). Without the guard ``exp((b_i-b_j)*(inf-finite)) = inf``
    would bubble out as the acceptance, letting a +inf replica unconditionally
    cool down and poison the low-T trajectory.
    """
    finite = jnp.isfinite(E_i) & jnp.isfinite(E_j)
    arg = (beta_i - beta_j) * (E_i - E_j)
    arg = jnp.where(finite, arg, -jnp.inf)
    return jnp.where(finite, jnp.minimum(1.0, jnp.exp(arg)), 0.0)


def _pt_chain(key, quat, trans, energy_fn, beta, sigma_rot, sigma_trans,
              axis_mask, n_steps):
    """One replica sweep (no trajectory). HBM cost is O(state), not O(state·n_steps)."""
    return scan_one(key, quat, trans, energy_fn, beta,
                    sigma_rot, sigma_trans, axis_mask, n_steps, start_step=None)


# per-replica-beta batched sweep: beta (axis 4) is mapped per chain
_pt_sweep = jax.vmap(
    _pt_chain, in_axes=(0, 0, 0, None, 0, None, None, None, None))


@partial(jax.jit, static_argnums=(3, 8))
def pt_sweep(keys, quats, transs, energy_fn, betas, sigma_rot, sigma_trans,
             axis_mask, n_steps):
    """Run every replica n_steps at its own beta. SHAPE: keys (B,2),
    quats (B,4), transs (B,3), betas (B,). Returns final-state arrays
    quat (B,4), trans (B,3), energy (B,), accept_rate (B,).

    Internally uses ``scan_metropolis(collect_traj=False)`` so the (n_steps,
    B, …) trajectory tensor is NEVER materialised — only the final state and
    a scalar accept rate per chain are kept.
    """
    return _pt_sweep(keys, quats, transs, energy_fn, betas, sigma_rot,
                     sigma_trans, axis_mask, n_steps)


@jax.jit
def swap_pass(key, quats, transs, energies, betas, parity):
    """One odd/even replica-exchange pass within each system (R axis).

    SHAPE: quats (S,R,4), transs (S,R,3), energies (S,R), betas (R,).
    ``parity`` is a scalar (Python int or 0-d JAX int): 0 → pairs (0,1),(2,3),…;
    1 → (1,2),(3,4),…  Both parities are computed and masked, so ``parity`` can
    be a traced value (e.g. ``rnd % 2`` inside ``lax.scan``).

    Returns swapped (quats, transs, energies) and accepted (S, max_pairs).
    Vectorized over both systems and pairs (no Python loop on replicas).

    Implementation notes
    --------------------
    Pre-computed parity-0 and parity-1 index sets are padded to a common
    ``max_pairs`` length so XLA sees fixed shapes. Pad slots point at column
    R-1 and carry ``valid=False``, so the gather/scatter on the source-index
    table at those slots is idempotent (no real swap, the index is overwritten
    by a self-swap before the take_along_axis).

    RNG: per-system swap keys are derived via ``fold_in(key, system_id)`` so
    independent systems get independent random draws (the input ``key`` alone
    would correlate swap acceptances across systems in run_multi_pt).
    """
    S, R = energies.shape

    # Pre-compute both parity cases (static, JIT-friendly)
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

    Ei = energies[:, L]                          # (S,max_pairs)
    Ej = energies[:, R_idx]
    dbeta = betas[L] - betas[R_idx]              # (max_pairs,)
    arg = dbeta[None, :] * (Ei - Ej)             # (S,max_pairs)
    # Per-system independent keys so systems don't share the same u draws.
    sys_ids = jnp.arange(S, dtype=jnp.uint32)
    sys_keys = jax.vmap(jax.random.fold_in, in_axes=(None, 0))(key, sys_ids)
    u = jax.vmap(
        lambda k: jax.random.uniform(k, (max_pairs,), minval=1e-38, maxval=1.0)
    )(sys_keys)                                  # (S, max_pairs)
    # Guard against non-finite energies: a replica with E=+inf (hard wall)
    # would otherwise swap from hot to cold unconditionally, contaminating
    # the low-temperature trajectory.
    finite = jnp.isfinite(Ei) & jnp.isfinite(Ej)
    acc = (jnp.log(u) < arg) & valid[None, :] & finite

    # Build per-slot source index via vectorized scatter (no replica loop).
    src = jnp.broadcast_to(jnp.arange(R), (S, R))
    src = src.at[:, L].set(jnp.where(acc, R_idx, L))
    src = src.at[:, R_idx].set(jnp.where(acc, L, R_idx))

    q = jnp.take_along_axis(quats, src[:, :, None], axis=1)
    t = jnp.take_along_axis(transs, src[:, :, None], axis=1)
    e = jnp.take_along_axis(energies, src, axis=1)
    return q, t, e, acc


# ---------------------------------------------------------------------------
# Helper: scatter the per-pair acceptance (max_pairs,) into the (R-1,) gap
# accumulator for the current parity. Used inside the scan body.
# ---------------------------------------------------------------------------

def _scatter_pair_to_gap(acc_mean, parity, R):
    """Map per-pair stats (max_pairs,) → per-gap (R-1,) with parity-aware index.

    Slot j ∈ [0, max_pairs) of the parity-p pair list corresponds to gap index
    ``p + 2 j`` in the full inter-replica accumulator (R-1,). Returns
    (acc_per_gap, mask_per_gap), both shape (R-1,), with 0 in slots not
    attempted this round.
    """
    max_pairs = acc_mean.shape[0]
    pair_idx = parity + 2 * jnp.arange(max_pairs)         # (max_pairs,)
    in_range = pair_idx < (R - 1)
    safe_idx = jnp.where(in_range, pair_idx, R - 2)        # avoid OOB on R≤2
    contrib = acc_mean.astype(jnp.float32) * in_range.astype(jnp.float32)
    acc_per_gap = jnp.zeros(R - 1, dtype=jnp.float32).at[safe_idx].add(contrib)
    mask_per_gap = jnp.zeros(R - 1, dtype=jnp.float32).at[safe_idx].add(
        in_range.astype(jnp.float32))
    return acc_per_gap, mask_per_gap


# ===========================================================================
# Single-system PT driver (closure energy_fn)
# ===========================================================================

def run_pt(key, init_quats, init_transs, energy_fn, betas, sigma_rot,
           sigma_trans, axis_mask, n_rounds, n_sweep):
    """Run PT for n_rounds; each round = n_sweep MC steps + one swap pass.

    SHAPE: init_quats (S,R,4), init_transs (S,R,3), betas (R,). The round loop
    is a single ``lax.scan`` over n_rounds (no Python-side dispatch). Returns
    a dict with low-T trajectory (n_rounds, S, 3), final states, per-temperature
    MC accept rate (R,), and per-adjacent-gap swap accept rate (R-1,).
    """
    S, R = init_quats.shape[0], init_quats.shape[1]
    betas_j = jnp.asarray(betas)
    betas_flat = jnp.broadcast_to(betas_j, (S, R)).reshape(-1)
    quats0 = jnp.asarray(init_quats).reshape(S * R, 4)
    transs0 = jnp.asarray(init_transs).reshape(S * R, 3)
    low_T_idx = int(jnp.argmax(betas_j))

    def round_body(carry, rnd):
        quats, transs = carry
        round_key = jax.random.fold_in(key, rnd)
        ks, kx = jax.random.split(round_key, 2)
        sweep_keys = jax.random.split(ks, S * R)

        q, t, e, ar = pt_sweep(sweep_keys, quats, transs, energy_fn, betas_flat,
                                sigma_rot, sigma_trans, axis_mask, n_sweep)
        q3 = q.reshape(S, R, 4)
        t3 = t.reshape(S, R, 3)
        e2 = e.reshape(S, R)

        parity = rnd % 2
        q3, t3, e2, acc = swap_pass(kx, q3, t3, e2, betas_j, parity)

        acc_mean = acc.astype(jnp.float32).mean(axis=0)         # (max_pairs,)
        swap_acc_gap, swap_mask_gap = _scatter_pair_to_gap(acc_mean, parity, R)

        ys = (
            q3[:, low_T_idx, :],                                 # (S, 4)
            t3[:, low_T_idx, :],                                 # (S, 3)
            ar.reshape(S, R).mean(axis=0).astype(jnp.float32),   # (R,)
            swap_acc_gap, swap_mask_gap,                         # (R-1,)
        )
        return (q3.reshape(S * R, 4), t3.reshape(S * R, 3)), ys

    (qf, tf), (low_T_quat, low_T_trans, mc_acc_per_round,
               swap_acc_per_round, swap_mask_per_round) = jax.lax.scan(
        round_body, (quats0, transs0), jnp.arange(n_rounds))

    mc_accept_rate = jnp.mean(mc_acc_per_round, axis=0)               # (R,)
    swap_mask_sum = jnp.sum(swap_mask_per_round, axis=0)              # (R-1,)
    swap_accept_rate = jnp.where(
        swap_mask_sum > 0,
        jnp.sum(swap_acc_per_round, axis=0) / jnp.maximum(swap_mask_sum, 1.0),
        0.0)

    return {
        "low_T_quat": low_T_quat,                  # (n_rounds, S, 4)
        "low_T_trans": low_T_trans,                # (n_rounds, S, 3)
        "final_quats": qf.reshape(S, R, 4),
        "final_transs": tf.reshape(S, R, 3),
        "mc_accept_rate": mc_accept_rate,
        "swap_accept_rate": swap_accept_rate,
        "low_T_idx": low_T_idx,
        # Convenience per-system view (matches run_multi_pt's layout).
        "per_system": {
            "low_T_quat": low_T_quat.transpose(1, 0, 2),    # (S, n_rounds, 4)
            "low_T_trans": low_T_trans.transpose(1, 0, 2),  # (S, n_rounds, 3)
        },
    }


# ===========================================================================
# Multi-system PT: per-system atom params, shared temperature ladder.
# Swaps within each system (R axis); systems are fully independent.
# Double vmap (S outer, R inner) keeps per-system arrays stored once.
# ===========================================================================

def _multi_chain(key, quat, trans, beta, pos0, q, c6p, c12p, cA, cB, lamD,
                 z_min, psi0, sigma_rot, sigma_trans, axis_mask, n_steps):
    """One chain (one replica) with continuum energy from system params."""
    from ptmc.energy.reference import steele_energy_jax

    def efn(quat_, trans_):
        return steele_energy_jax(quat_, trans_, pos0, q, c6p, c12p, cA, cB,
                                  lamD, z_min, psi0)
    return scan_one(key, quat, trans, efn, beta,
                    sigma_rot, sigma_trans, axis_mask, n_steps, start_step=None)


# Inner vmap: over R replicas within ONE system. Per-replica: key, quat, trans, beta.
# Per-system: pos0, q, c6p, c12p, cA, cB, lamD, z_min, psi0.
_multi_inner = jax.vmap(
    _multi_chain,
    in_axes=(0, 0, 0, 0,
             None, None, None, None,
             None, None, None,
             None, None,
             None, None, None,
             None))

# Outer vmap: over S systems. All per-system arrays carry leading S axis.
_multi_outer = jax.vmap(
    _multi_inner,
    in_axes=(0, 0, 0, 0,
             0, 0, 0, 0,
             0, 0, 0,
             0, 0,
             None, None, None,
             None))


@partial(jax.jit, static_argnums=(16,))
def _multi_pt_sweep_ps(keys_sr, quat_sr, trans_sr, betas_sr,
                       pos0_s, q_s, c6p_s, c12p_s, cA_s, cB_s, lamD_s,
                       z_min_s, psi0_s,
                       sigma_rot, sigma_trans, axis_mask, n_steps):
    """Multi-system PT sweep (per-system atom params, double vmap)."""
    return _multi_outer(keys_sr, quat_sr, trans_sr, betas_sr,
                        pos0_s, q_s, c6p_s, c12p_s, cA_s, cB_s, lamD_s,
                        z_min_s, psi0_s,
                        sigma_rot, sigma_trans, axis_mask, n_steps)


def run_multi_pt(key, systems, betas, sigma_rot, sigma_trans, axis_mask,
                 n_rounds, n_sweep, seed=42):
    """Run PT on S independent systems sharing one temperature ladder (R temps).

    Parameters
    ----------
    systems : list[dict]
        Each dict: pos0 (N_s,3), q (N_s,), c6 (N_s,), c12 (N_s,),
                   c6_surf, c12_surf, rho_s, psi0, lambda_D, z_min, init_z.
        All systems MUST have same N (pad if needed).
    betas : (R,) array
        Inverse temperatures for the ladder.
    sigma_rot, sigma_trans : float
        MC proposal scales.
    axis_mask : (3,) bool/float
        Translation mask.
    n_rounds : int
        Number of PT rounds (swap attempts).
    n_sweep : int
        MC sweeps per round.
    seed : int
        Master PRNG seed.

    Returns
    -------
    dict with low_T_trans (n_rounds,S,3), final_quats (S,R,4),
         final_transs (S,R,3), mc_accept_rate (R,), swap_accept_rate (R-1,).
    """
    S = len(systems)
    R = len(betas)
    betas_j = jnp.asarray(betas)

    # ---- per-system params (S, ...) ---------------------------------------
    coeffs = [steele_coefficients_for(s["c6"], s["c12"],
                                      s.get("c6_surf", 1.0), s.get("c12_surf", 1.0),
                                      s.get("eps_surf"), s.get("sigma_surf"),
                                      s["rho_s"]) for s in systems]
    pos0_s  = jnp.asarray(jnp.stack([jnp.asarray(s["pos0"]) for s in systems]))   # (S,N,3)
    q_s     = jnp.asarray(jnp.stack([jnp.asarray(s["q"])    for s in systems]))   # (S,N)
    c6p_s   = jnp.asarray(jnp.stack([jnp.asarray(c[0])      for c in coeffs]))    # (S,N)
    c12p_s  = jnp.asarray(jnp.stack([jnp.asarray(c[1])      for c in coeffs]))    # (S,N)
    cA_s    = jnp.asarray(jnp.array([c[2]                   for c in coeffs]))    # (S,)
    cB_s    = jnp.asarray(jnp.array([c[3]                   for c in coeffs]))    # (S,)
    lamD_s  = jnp.asarray(jnp.array([s["lambda_D"]          for s in systems]))   # (S,)
    z_min_s = jnp.asarray(jnp.array([s["z_min"]             for s in systems]))   # (S,)
    psi0_s  = jnp.asarray(jnp.array([s["psi0"]              for s in systems]))   # (S,)
    z0_s    = jnp.asarray(jnp.array([s.get("init_z", s["z_min"] + 0.5)
                                      for s in systems]))                         # (S,)

    # ---- per-replica state (S, R, ...) ------------------------------------
    # Per-round keys via fold_in(master, round); per-replica/chain keys via
    # fold_in(round_key, sid*R + rid). Initial pose key fold_in offset matches
    # the high-throughput runner.
    master = jax.random.PRNGKey(seed)
    init_keys = jax.random.split(master, S * R).reshape(S, R, 2)
    qn = jax.vmap(jax.vmap(lambda k: jax.random.normal(
        jax.random.fold_in(k, INIT_QUAT_KEY_OFFSET), (4,))))(init_keys)
    quat_sr = qn / jnp.linalg.norm(qn, axis=-1, keepdims=True)
    trans_sr = jnp.stack([
        jnp.zeros((S, R)),
        jnp.zeros((S, R)),
        jnp.broadcast_to(z0_s[:, None], (S, R))], axis=-1)

    betas_sr = jnp.broadcast_to(betas_j[None, :], (S, R))      # (S, R) constant
    low_T_idx = int(jnp.argmax(betas_j))
    axis_mask_j = jnp.asarray(axis_mask)
    sigma_rot_f = float(sigma_rot)
    sigma_trans_f = float(sigma_trans)
    n_sweep_i = int(n_sweep)

    def round_body(carry, rnd):
        quat_sr, trans_sr = carry
        round_key = jax.random.fold_in(master, rnd)
        ks, kx = jax.random.split(round_key, 2)
        # (S, R) sweep keys for inner vmap
        sweep_keys = jax.random.split(ks, S * R).reshape(S, R, 2)

        qf, tf, ef, ar = _multi_pt_sweep_ps(
            sweep_keys, quat_sr, trans_sr, betas_sr,
            pos0_s, q_s, c6p_s, c12p_s, cA_s, cB_s, lamD_s,
            z_min_s, psi0_s,
            sigma_rot_f, sigma_trans_f, axis_mask_j, n_sweep_i)
        # qf,tf,ef,ar: (S, R, …)

        parity = rnd % 2
        q_post, t_post, e_post, acc = swap_pass(kx, qf, tf, ef, betas_j, parity)

        acc_mean = acc.astype(jnp.float32).mean(axis=0)
        swap_acc_gap, swap_mask_gap = _scatter_pair_to_gap(acc_mean, parity, R)
        ys = (
            q_post[:, low_T_idx, :],                                  # (S, 4)
            t_post[:, low_T_idx, :],                                  # (S, 3)
            ar.mean(axis=0).astype(jnp.float32),                      # (R,)
            swap_acc_gap, swap_mask_gap,                              # (R-1,)
        )
        return (q_post, t_post), ys

    (qf, tf), (low_T_quat, low_T_trans, mc_acc_per_round,
               swap_acc_per_round, swap_mask_per_round) = jax.lax.scan(
        round_body, (quat_sr, trans_sr), jnp.arange(n_rounds))

    mc_accept_rate = jnp.mean(mc_acc_per_round, axis=0)              # (R,)
    swap_mask_sum = jnp.sum(swap_mask_per_round, axis=0)             # (R-1,)
    swap_accept_rate = jnp.where(
        swap_mask_sum > 0,
        jnp.sum(swap_acc_per_round, axis=0) / jnp.maximum(swap_mask_sum, 1.0),
        0.0)

    return {
        "low_T_quat": low_T_quat,                    # (n_rounds, S, 4)
        "low_T_trans": low_T_trans,                  # (n_rounds, S, 3)
        "final_quats": qf,                           # (S, R, 4)
        "final_transs": tf,                          # (S, R, 3)
        "mc_accept_rate": mc_accept_rate,
        "swap_accept_rate": swap_accept_rate,
        "low_T_idx": low_T_idx,
        "per_system": {
            "low_T_quat": low_T_quat.transpose(1, 0, 2),   # (S, n_rounds, 4)
            "low_T_trans": low_T_trans.transpose(1, 0, 2),  # (S, n_rounds, 3)
        },
    }
