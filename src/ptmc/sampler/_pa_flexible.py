"""Chi-aware Population Annealing (PA) in JAX.

Extends ``population_annealing.py`` with a chi-aware variant where the state
carries (quats, transs, chis) — shape (M,4), (M,3), (M,K) — and the MC sweep
uses ``scan_flexible_metropolis`` instead of the rigid-body ``scan_one``.

The adaptive cooling / ESS / resample machinery is identical; only the
per-walker MC kernel and the state dimensionality change.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from ptmc.mc.flex_metropolis import (
    dof_move_weights,
    scan_flexible_metropolis,
)
from ptmc.log import logger

# Reuse the same stall threshold as the rigid PA driver.
_PA_STALL_DBETA: float = 1e-8


# ---------------------------------------------------------------------------
# ESS and adaptive Δβ — identical to the rigid PA driver.
# ---------------------------------------------------------------------------

def _ess_fraction_jax(log_w):
    """ESS / M from unnormalized log-weights (JAX). Shape (M,) -> scalar."""
    m = jnp.max(log_w)
    w = jnp.exp(log_w - m)
    s = jnp.sum(w)
    s2 = jnp.sum(w * w)
    M = log_w.shape[0]
    return jnp.where(s > 0, (s * s) / (s2 * M), 0.0)


@partial(jax.jit, static_argnums=(3,))
def _find_dbeta_jax(E, dbeta_max, target_ess_frac, n_iter=60):
    """Bisection on Δβ (largest value in [0, dbeta_max] with ESS frac ≥ target)."""
    def ess(d):
        log_w = -d * E
        return _ess_fraction_jax(log_w)

    full_ok = ess(dbeta_max) >= target_ess_frac

    def body(_, state):
        lo, hi = state
        mid = 0.5 * (lo + hi)
        ok = ess(mid) >= target_ess_frac
        new_lo = jnp.where(ok, mid, lo)
        new_hi = jnp.where(ok, hi, mid)
        return (new_lo, new_hi)

    lo, _ = jax.lax.fori_loop(0, n_iter, body, (jnp.asarray(0.0), jnp.asarray(dbeta_max)))
    return jnp.where(full_ok, dbeta_max, lo)


# ---------------------------------------------------------------------------
# Systematic resampling (low variance, fixed-size output)
# ---------------------------------------------------------------------------

@jax.jit
def _systematic_resample(key, log_w):
    """Systematic (low-variance) resampling. SHAPE log_w (M,) -> idx (M,)."""
    M = log_w.shape[0]
    w = jax.nn.softmax(log_w)
    u0 = jax.random.uniform(key) / M
    positions = u0 + jnp.arange(M) / M
    edges = jnp.cumsum(w)
    return jnp.searchsorted(edges, positions)


# ---------------------------------------------------------------------------
# Module-level JITs for the chi-aware PA sweep.
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnums=(0, 11))
def _pa_flex_sweep_step(energy_fn, key_s, quats, transs, chis, beta_j,
                          sigma_rot, sigma_trans, sigma_chi,
                          axis_mask, move_weights, n_sweep):
    """One PA sweep: every walker runs ``n_sweep`` flexible-Metropolis steps at ``beta_j``.

    SHAPE: returns (quats, transs, chis, E) -> (M,4), (M,3), (M,K), (M,).
    """
    M = quats.shape[0]
    keys = jax.random.split(key_s, M)

    def per_walker(k, q, t, c):
        out = scan_flexible_metropolis(
            k, q, t, c,
            energy_fn, beta_j,
            sigma_rot, sigma_trans, sigma_chi,
            axis_mask, move_weights,
            n_sweep, start_step=None,
        )
        return out["quat_final"], out["trans_final"], out["chi_final"], out["energy_final"]

    return jax.vmap(per_walker)(keys, quats, transs, chis)


@jax.jit
def _pa_step_scalars(E, dbeta_max, target_ess):
    """Fused: bisect dbeta, then derive log_w, logZ_inc, ess_frac in one trip."""
    M = E.shape[0]
    dbeta = _find_dbeta_jax(E, dbeta_max, target_ess)
    log_w = -dbeta * E
    logZ_inc = logsumexp(log_w) - jnp.log(M)
    ess_frac = _ess_fraction_jax(log_w)
    return dbeta, logZ_inc, ess_frac, log_w


# ---------------------------------------------------------------------------
# Chi-aware Population Annealing driver
# ---------------------------------------------------------------------------

def run_pa_flexible(key, init_quats, init_transs, init_chis, energy_fn,
                      beta0, beta_target, n_sweep,
                      sigma_rot, sigma_trans, sigma_chi,
                      axis_mask, move_weights=None,
                      target_ess=0.7, max_steps=400):
    """Chi-aware population annealing from beta0 to beta_target.

    Parameters
    ----------
    key : PRNGKey
    init_quats : (M, 4)  initial unit quaternions
    init_transs : (M, 3)  initial translations (nm)
    init_chis : (M, K)  initial χ dihedral angles (rad)
    energy_fn : callable (quat, trans, chi) -> scalar
        Must be JIT-compatible.
    beta0, beta_target : float
        Starting and target inverse temperature (mol/kJ).
    n_sweep : int
        MC sweeps per cooling step per walker.
    sigma_rot, sigma_trans, sigma_chi : float
        Proposal step sizes.
    axis_mask : (3,) array
        Translation axis mask.
    move_weights : (3,) array or None
        Log-weights for (rot, trans, chi) move-type categorical.
        Default: DOF-proportional weights.
    target_ess : float
        Minimum ESS fraction for adaptive cooling.
    max_steps : int
        Maximum cooling schedule steps.

    Returns
    -------
    dict with logZ_ratio, final state tensors, convergence flag, ESS history.
    """
    quats = jnp.asarray(init_quats)
    transs = jnp.asarray(init_transs)
    chis = jnp.asarray(init_chis)
    K = chis.shape[1]

    if move_weights is None:
        move_weights = dof_move_weights(K)

    init_E = jax.vmap(energy_fn)(quats, transs, chis)
    n_bad = int(jnp.sum(~jnp.isfinite(init_E)))
    if n_bad > 0:
        raise ValueError(
            f"{n_bad}/{quats.shape[0]} walkers have non-finite initial energy. "
            f"Try increasing init_z — the protein may be too close to the hard wall."
        )

    beta = float(beta0)
    logZ = 0.0
    ess_hist, betas_hist = [], [beta]
    E = None

    converged = False
    for step_idx in range(max_steps):
        if beta >= beta_target - 1e-12:
            converged = True
            break
        key, ks, kr = jax.random.split(key, 3)

        quats, transs, chis, E = _pa_flex_sweep_step(
            energy_fn, ks, quats, transs, chis, jnp.asarray(beta),
            sigma_rot, sigma_trans, sigma_chi,
            axis_mask, move_weights, n_sweep)

        # Guard against inf energies.
        n_bad_E = int(jnp.sum(~jnp.isfinite(E)))
        if n_bad_E > 0:
            raise RuntimeError(
                f"Population annealing aborted at beta={beta:.4f} mol/kJ: "
                f"{n_bad_E}/{E.shape[0]} walkers have non-finite energy after "
                f"a Metropolis sweep. Adaptive cooling cannot proceed because "
                f"ESS collapses to 0 at every trial Δβ. Likely cause: "
                f"a walker hit the hard wall and the proposal scale is too "
                f"small to recover. Try smaller sigma_trans or larger init_z.")

        dbeta, logZ_inc, ess_frac, log_w = _pa_step_scalars(
            E, beta_target - beta, target_ess)
        idx = _systematic_resample(kr, log_w)
        quats = quats[idx]
        transs = transs[idx]
        chis = chis[idx]
        E = E[idx]

        # One fused device->host transfer per cooling step.
        dbeta_h, logZ_inc_h, ess_frac_h = (
            float(x) for x in jax.device_get((dbeta, logZ_inc, ess_frac)))

        if dbeta_h < _PA_STALL_DBETA:
            raise RuntimeError(
                f"Population annealing stalled at beta={beta:.4f} mol/kJ "
                f"(remaining {beta_target - beta:.4f}): adaptive Δβ collapsed "
                f"to {dbeta_h:.2e}, below {_PA_STALL_DBETA:.0e}. ESS at every "
                f"trial Δβ is below target_ess={target_ess}. Try lowering "
                f"target_ess, increasing n_walkers, or running more equilibration "
                f"sweeps at beta0 before annealing.")
        beta += dbeta_h
        logZ += logZ_inc_h
        ess_hist.append(ess_frac_h)
        betas_hist.append(beta)

    if not converged:
        logger.warning(
            "PA flexible exhausted max_steps=%d before reaching "
            "beta_target=%.4f (final beta=%.4f, remaining=%.4f). "
            "Results may be biased; raise max_steps or lower target_ess.",
            max_steps, float(beta_target), beta, float(beta_target) - beta)

    return {
        "logZ_ratio": logZ,
        "final_quats": quats,
        "final_transs": transs,
        "final_chis": chis,
        "final_energies": (None if E is None else np.asarray(E)),
        "ess": np.asarray(ess_hist),
        "betas": np.asarray(betas_hist),
        "n_steps": len(ess_hist),
        "converged": converged,
    }
