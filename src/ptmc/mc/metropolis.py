"""Metropolis MC over rigid-body pose (JAX): single chain and batched chains.

The module also exposes ``scan_metropolis``, a reusable lax.scan core that
wraps "compute E0 + scan n_steps of metropolis_step". All hot-loop drivers
(highthroughput, parallel_tempering, population_annealing) build on it, so
the trial-move + accept-reject logic lives in exactly one place.
"""
from __future__ import annotations
from functools import partial
import jax
import jax.numpy as jnp
from ptmc.mc.moves import propose_rotation, propose_translation


def accept_logprob(dE, beta):
    """log Metropolis acceptance min(1, exp(-beta dE)).

    Safe under non-finite dE (e.g. a proposal that lands in a hard wall and
    returns +inf): inf - finite = inf, -beta*inf = -inf, min(0,-inf) = -inf;
    callers compare ``log u < -inf`` which is always False → reject. The
    explicit ``jnp.where`` keeps the result well-defined when both energies
    are inf (dE = nan), still producing a sentinel reject value.
    """
    arg = -beta * dE
    safe = jnp.where(jnp.isnan(arg), -jnp.inf, arg)
    return jnp.minimum(0.0, safe)


def metropolis_step(key, quat, trans, energy, energy_fn, beta,
                    sigma_rot, sigma_trans, axis_mask):
    k_rot, k_trans, k_acc = jax.random.split(key, 3)
    new_quat = propose_rotation(k_rot, quat, sigma_rot)
    new_trans = propose_translation(k_trans, trans, sigma_trans, axis_mask)
    new_E = energy_fn(new_quat, new_trans)
    dE = new_E - energy
    logu = jnp.log(jax.random.uniform(k_acc, minval=1e-38, maxval=1.0))
    accept = logu < accept_logprob(dE, beta)
    return (jnp.where(accept, new_quat, quat),
            jnp.where(accept, new_trans, trans),
            jnp.where(accept, new_E, energy), accept)


def scan_metropolis(key, init_quat, init_trans, energy_fn, beta,
                    sigma_rot, sigma_trans, axis_mask, n_steps,
                    start_step=None, collect_traj=False):
    """Reusable lax.scan core: run n_steps Metropolis steps in one chain.

    Two RNG modes:
      * ``start_step is None``  → keys come from ``jax.random.split(key, n_steps)``
        (sequential mode, matches legacy ``_chain_impl``).
      * ``start_step is int/scalar`` → per-step key is ``fold_in(key, start_step+t)``
        (positional mode, matches high-throughput / checkpoint-resume contract).

    Returns
    -------
    dict with keys
        ``quat_final`` (4,), ``trans_final`` (3,), ``energy_final`` scalar,
        ``accept_rate`` scalar (mean across steps),
    and — when ``collect_traj=True`` — additionally:
        ``quat`` (n_steps,4), ``trans`` (n_steps,3), ``energy`` (n_steps,),
        ``accepted`` (n_steps,).

    The trajectory variant is used by analysis / unit tests; the streaming
    variant (default) is what the hot-loop drivers use to keep memory ~O(1)
    per chain regardless of step count.
    """
    E0 = energy_fn(init_quat, init_trans)
    use_fold_in = start_step is not None

    if use_fold_in:
        def body(carry, t):
            q, tr, e = carry
            k = jax.random.fold_in(key, t)
            q, tr, e, acc = metropolis_step(k, q, tr, e, energy_fn, beta,
                                            sigma_rot, sigma_trans, axis_mask)
            if collect_traj:
                return (q, tr, e), (q, tr, e, acc)
            return (q, tr, e), acc
        steps = start_step + jnp.arange(n_steps)
    else:
        def body(carry, k):
            q, tr, e = carry
            q, tr, e, acc = metropolis_step(k, q, tr, e, energy_fn, beta,
                                            sigma_rot, sigma_trans, axis_mask)
            if collect_traj:
                return (q, tr, e), (q, tr, e, acc)
            return (q, tr, e), acc
        steps = jax.random.split(key, n_steps)

    if collect_traj:
        (qf, tf, ef), (qs, ts, es, accs) = jax.lax.scan(
            body, (init_quat, init_trans, E0), steps)
        return {
            "quat": qs, "trans": ts, "energy": es, "accepted": accs,
            "quat_final": qf, "trans_final": tf, "energy_final": ef,
            "accept_rate": jnp.mean(accs.astype(jnp.float32)),
        }
    (qf, tf, ef), accs = jax.lax.scan(
        body, (init_quat, init_trans, E0), steps)
    return {
        "quat_final": qf, "trans_final": tf, "energy_final": ef,
        "accept_rate": jnp.mean(accs.astype(jnp.float32)),
    }


def scan_one(key, init_quat, init_trans, energy_fn, beta,
             sigma_rot, sigma_trans, axis_mask, n_steps, start_step=None):
    """Common no-trajectory chain runner used by HT/PT/PA hot loops.

    Thin tuple-returning wrapper around ``scan_metropolis(collect_traj=False)``
    so each path doesn't repeat the "scan + dict-unpack to tuple" boilerplate.
    Accepts both RNG modes via ``start_step`` (None → split, int → fold_in).

    Returns ``(quat_final, trans_final, energy_final, accept_rate)``.
    """
    out = scan_metropolis(key, init_quat, init_trans, energy_fn, beta,
                          sigma_rot, sigma_trans, axis_mask, n_steps,
                          start_step=start_step, collect_traj=False)
    return (out["quat_final"], out["trans_final"], out["energy_final"],
            out["accept_rate"])


def _chain_impl(key, init_quat, init_trans, energy_fn, beta,
                sigma_rot, sigma_trans, axis_mask, n_steps):
    """Single-chain scan core (legacy API, returns full trajectory). SHAPE:
    scalars per chain; returns trajectories."""
    out = scan_metropolis(key, init_quat, init_trans, energy_fn, beta,
                          sigma_rot, sigma_trans, axis_mask, n_steps,
                          start_step=None, collect_traj=True)
    return {"quat": out["quat"], "trans": out["trans"],
            "energy": out["energy"], "accepted": out["accepted"],
            "accept_rate": out["accept_rate"]}


run_chain = jax.jit(_chain_impl, static_argnums=(3, 8))
_chains_vmapped = jax.vmap(_chain_impl, in_axes=(0,0,0,None,None,None,None,None,None))

@partial(jax.jit, static_argnums=(3, 8))
def run_chains(keys, init_quats, init_transs, energy_fn, beta,
               sigma_rot, sigma_trans, axis_mask, n_steps):
    """n_chains independent chains (vmap+jit). Returns batched trajectories."""
    return _chains_vmapped(keys, init_quats, init_transs, energy_fn, beta,
                           sigma_rot, sigma_trans, axis_mask, n_steps)

def split_chain_keys(key, n_chains):
    return jax.random.split(key, n_chains)
