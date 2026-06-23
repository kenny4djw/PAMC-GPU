"""Metropolis MC over (quat, trans, chi) -- the semi-flexible extension.

The rigid-body kernel in ``metropolis.py`` is left untouched (it is the
ground-truth path used by 200+ existing tests). This module adds the
chi-aware variant that picks a move type per step and tracks a per-type
acceptance counter (§ 6 of the design doc).

State per step (single chain):
    quat   (4,)
    trans  (3,)
    chi    (K,)
    energy ()
    acc_counts    (3,) int32   [n_acc_rot, n_acc_trans, n_acc_chi]
    try_counts    (3,) int32   [n_try_rot, n_try_trans, n_try_chi]

Per-step proposal (§ 6.2 method 1): one move per step, chosen by categorical
over (rot, trans, chi) with weights (p_rot, p_trans, p_chi). The plan
suggests DOF-proportional weights (3, 3, K)/(6+K), but the actual values are
passed in as ``move_weights`` so callers can tune.

RNG positionalization (§ 6.6): the step key is split into sub-keys for each
named decision (type, rot omega, trans delta, chi index, chi delta, accept).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from ptmc.mc.metropolis import accept_logprob
from ptmc.mc.moves import (
    propose_chi,
    propose_rotation,
    propose_translation,
)


def flexible_metropolis_step(key, quat, trans, chi, energy,
                             energy_fn, beta,
                             sigma_rot, sigma_trans, sigma_chi,
                             axis_mask, move_weights):
    """One Metropolis step in the rot/trans/chi joint space.

    ``energy_fn`` must have signature ``(quat, trans, chi) -> scalar``.
    ``move_weights`` is a length-3 array of unnormalized log-weights for
    (rot, trans, chi) selection (passed through ``jax.random.categorical``).

    Returns
    -------
    (quat', trans', chi', energy', accepted, move_type)
        ``accepted`` is a bool scalar; ``move_type`` is int in {0, 1, 2}.
    """
    k_type, k_rot, k_trans, k_chi, k_acc = jax.random.split(key, 5)
    move_type = jax.random.categorical(k_type, move_weights)  # () int

    # Propose all three; pick one via where (cheap -- proposals are O(1)
    # compared to the energy evaluation).
    new_quat_rot = propose_rotation(k_rot, quat, sigma_rot)
    new_trans_t = propose_translation(k_trans, trans, sigma_trans, axis_mask)
    new_chi_c, _chi_idx = propose_chi(k_chi, chi, sigma_chi)

    new_quat = jnp.where(move_type == 0, new_quat_rot, quat)
    new_trans = jnp.where(move_type == 1, new_trans_t, trans)
    new_chi = jnp.where(move_type == 2, new_chi_c, chi)

    new_E = energy_fn(new_quat, new_trans, new_chi)
    dE = new_E - energy
    logu = jnp.log(jax.random.uniform(k_acc, minval=1e-38, maxval=1.0))
    accept = logu < accept_logprob(dE, beta)

    return (
        jnp.where(accept, new_quat, quat),
        jnp.where(accept, new_trans, trans),
        jnp.where(accept, new_chi, chi),
        jnp.where(accept, new_E, energy),
        accept,
        move_type,
    )


def dof_move_weights(K: int) -> jnp.ndarray:
    """Default DOF-proportional log-weights: (rot, trans, chi) ∝ (3, 3, K).

    Returns log-weights suitable for ``jax.random.categorical`` (unnormalized).
    When ``K == 0`` the chi weight is clamped to a tiny but finite floor so
    that the resulting log-probabilities remain finite. ``jax.random.categorical``
    on a vector containing ``-inf`` can produce non-deterministic move-type
    selection across XLA backends; a finite floor with relative weight ~1e-38
    means the chi branch is statistically never selected while still keeping
    the trace numerically clean.
    """
    chi_weight = max(float(K), 1e-38)
    w = jnp.asarray([3.0, 3.0, chi_weight], dtype=jnp.float32)
    return jnp.log(w)


def scan_flexible_metropolis(key, init_quat, init_trans, init_chi,
                             energy_fn, beta,
                             sigma_rot, sigma_trans, sigma_chi,
                             axis_mask, move_weights, n_steps,
                             start_step=None):
    """Reusable lax.scan core for the chi-aware Metropolis path.

    Two RNG modes (same convention as ``scan_metropolis`` in ``metropolis.py``):
        - ``start_step is None``  -> split(key, n_steps)
        - ``start_step is int``  -> fold_in(key, start_step + t)

    Carry: (quat, trans, chi, energy, acc_counts, try_counts). The acceptance
    counters live in the carry as int32 so they survive vmap / lax.scan.

    Returns
    -------
    dict with
        quat_final, trans_final, chi_final, energy_final  -- last-step state
        acc_counts (3,)  -- accepted moves per type (rot, trans, chi)
        try_counts (3,)  -- attempted moves per type
        accept_rate_per_type (3,) float -- acc / max(try, 1)
        overall_accept_rate float
    """
    E0 = energy_fn(init_quat, init_trans, init_chi)
    K = init_chi.shape[0]
    init_acc = jnp.zeros((3,), dtype=jnp.int32)
    init_try = jnp.zeros((3,), dtype=jnp.int32)

    use_fold_in = start_step is not None

    if use_fold_in:
        def body(carry, t):
            q, tr, ch, e, acc_c, try_c = carry
            k_step = jax.random.fold_in(key, t)
            q, tr, ch, e, accepted, mt = flexible_metropolis_step(
                k_step, q, tr, ch, e, energy_fn, beta,
                sigma_rot, sigma_trans, sigma_chi,
                axis_mask, move_weights)
            one_hot = jax.nn.one_hot(mt, 3, dtype=jnp.int32)
            try_c = try_c + one_hot
            acc_c = acc_c + one_hot * accepted.astype(jnp.int32)
            return (q, tr, ch, e, acc_c, try_c), None
        steps = start_step + jnp.arange(n_steps)
    else:
        def body(carry, k_step):
            q, tr, ch, e, acc_c, try_c = carry
            q, tr, ch, e, accepted, mt = flexible_metropolis_step(
                k_step, q, tr, ch, e, energy_fn, beta,
                sigma_rot, sigma_trans, sigma_chi,
                axis_mask, move_weights)
            one_hot = jax.nn.one_hot(mt, 3, dtype=jnp.int32)
            try_c = try_c + one_hot
            acc_c = acc_c + one_hot * accepted.astype(jnp.int32)
            return (q, tr, ch, e, acc_c, try_c), None
        steps = jax.random.split(key, n_steps)

    (qf, tf, cf, ef, acc_c, try_c), _ = jax.lax.scan(
        body, (init_quat, init_trans, init_chi, E0, init_acc, init_try),
        steps)

    safe_try = jnp.maximum(try_c, 1).astype(jnp.float32)
    return {
        "quat_final": qf,
        "trans_final": tf,
        "chi_final": cf,
        "energy_final": ef,
        "acc_counts": acc_c,
        "try_counts": try_c,
        "accept_rate_per_type": acc_c.astype(jnp.float32) / safe_try,
        "overall_accept_rate": (acc_c.sum().astype(jnp.float32)
                                / jnp.maximum(n_steps, 1)),
    }
