"""Population Annealing (PA) + free energy (logZ), in JAX.

Maintain a population of M walkers. Starting hot (small beta) where sampling is
easy, cool along a temperature schedule. At each cooling step beta_k -> beta_{k+1}:
  1. equilibrate every walker with n_sweep Metropolis steps at beta_k,
  2. weight each walker w_i ∝ exp(-(beta_{k+1}-beta_k) E_i),
  3. accumulate log(Z_{k+1}/Z_k) = logmeanexp(log w)  (free-energy ratio),
  4. systematic-resample M walkers by normalized weight (population size fixed).
The cooling step size is chosen adaptively (bisection on Delta-beta) to keep the
effective sample size ESS = (sum w)^2 / sum w^2 above a target, preventing
weight collapse.

Analytic check (1-DOF harmonic E=1/2 k x^2): Z(beta) ∝ beta^{-1/2}, so
log(Z(beta_K)/Z(beta_0)) = -1/2 log(beta_K / beta_0).

GPU residency
-------------
Bisection (``_find_dbeta_jax``) runs entirely on device via ``lax.fori_loop``,
and the MC sweep keeps energies as JAX arrays from one cooling step to the next.
Only tiny scalars (Δβ, ESS, logZ increment) cross the host boundary each step,
keeping the steady-state cost dominated by the GPU MC kernels.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from ptmc.mc.metropolis import scan_one
from ptmc.log import logger

# Minimum cooling step magnitude (mol/kJ) below which we treat the schedule as
# stalled. Bisection naturally collapses to 0 when ESS at every Δβ tried is
# below target — usually a sign that some walker carries +inf energy.
_PA_STALL_DBETA: float = 1e-8


# ---------------------------------------------------------------------------
# ESS and adaptive Δβ — JAX-resident and numpy-wrapper variants.
# ---------------------------------------------------------------------------

def _ess_fraction_jax(log_w):
    """ESS / M from unnormalized log-weights (JAX). Shape (M,) -> scalar."""
    m = jnp.max(log_w)
    w = jnp.exp(log_w - m)
    s = jnp.sum(w)
    s2 = jnp.sum(w * w)
    M = log_w.shape[0]
    return jnp.where(s > 0, (s * s) / (s2 * M), 0.0)


def ess_fraction(log_w):
    """ESS / M from unnormalized log-weights. Accepts numpy or JAX arrays.

    Numpy path preserves the legacy public API (unit tests pass numpy in).
    """
    if isinstance(log_w, jnp.ndarray) or hasattr(log_w, "device"):
        return float(_ess_fraction_jax(jnp.asarray(log_w)))
    log_w = np.asarray(log_w)
    m = np.max(log_w)
    w = np.exp(log_w - m)
    s = np.sum(w)
    return float((s * s) / (np.sum(w * w) * w.size))


@partial(jax.jit, static_argnums=(3,))
def _find_dbeta_jax(E, dbeta_max, target_ess_frac, n_iter=60):
    """Bisection on Δβ (largest value in [0, dbeta_max] with ESS frac ≥ target).

    Pure JAX (``lax.fori_loop``): no Python-side ESS evaluations, no GPU→host
    transfers. ESS decreases monotonically with Δβ, so bisection is correct.
    """
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


def _find_dbeta(E, dbeta_max, target_ess_frac):
    """Bisection: largest Delta-beta in [0, dbeta_max] with ESS frac >= target.

    Numpy reference implementation; preserved as the public API used by tests.
    ESS decreases monotonically with Delta-beta; the offset in E cancels in ESS.
    """
    E = np.asarray(E)
    if dbeta_max <= 0.0:
        return 0.0
    if ess_fraction(-dbeta_max * E) >= target_ess_frac:
        return float(dbeta_max)
    lo, hi = 0.0, float(dbeta_max)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if ess_fraction(-mid * E) >= target_ess_frac:
            lo = mid
        else:
            hi = mid
    return lo


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
# Module-level JITs.  Hoisting them out of ``run_pa`` lets the JIT cache
# survive across multiple PA calls (parameter sweeps, restarts): the
# closure-hash now keys only on (energy_fn, n_sweep) instead of the per-call
# closure identity.
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnums=(0, 8))
def _pa_sweep_step(energy_fn, key_s, quats, transs, beta_j,
                   sigma_rot, sigma_trans, axis_mask, n_sweep):
    """One PA sweep: every walker runs ``n_sweep`` Metropolis steps at ``beta_j``.

    SHAPE: returns (quats, transs, E) -> (M,4), (M,3), (M,).
    """
    M = quats.shape[0]
    keys = jax.random.split(key_s, M)

    def per_walker(k, q, t):
        qf, tf, ef, _ar = scan_one(k, q, t, energy_fn, beta_j,
                                    sigma_rot, sigma_trans, axis_mask, n_sweep,
                                    start_step=None)
        return qf, tf, ef

    return jax.vmap(per_walker)(keys, quats, transs)


@jax.jit
def _pa_step_scalars(E, dbeta_max, target_ess):
    """Fused: bisect dbeta, then derive log_w, logZ_inc, ess_frac in one trip.

    A single device->host transfer pulls (dbeta, logZ_inc, ess_frac) per
    cooling step (vs. three separate transfers when the helpers were JITted
    apart).  log_w is also returned on-device for the subsequent resample.
    """
    M = E.shape[0]
    dbeta = _find_dbeta_jax(E, dbeta_max, target_ess)
    log_w = -dbeta * E
    logZ_inc = logsumexp(log_w) - jnp.log(M)
    ess_frac = _ess_fraction_jax(log_w)
    return dbeta, logZ_inc, ess_frac, log_w


# ---------------------------------------------------------------------------
# Population Annealing driver
# ---------------------------------------------------------------------------

def run_pa(key, init_quats, init_transs, energy_fn, beta0, beta_target, n_sweep,
           sigma_rot, sigma_trans, axis_mask, target_ess=0.7, max_steps=400):
    """Population annealing from beta0 to beta_target.

    SHAPE: init_quats (M,4), init_transs (M,3). Returns a dict with
    ``logZ_ratio`` = log(Z(beta_target)/Z(beta0)), ``ess`` (frac per step),
    ``betas`` (β_0 .. β_K), ``n_steps``, the final population
    (``final_quats``, ``final_transs``, ``final_energies``) and a
    ``converged`` flag (True iff β_target reached before ``max_steps``).
    """
    quats, transs = jnp.asarray(init_quats), jnp.asarray(init_transs)

    init_E = jax.vmap(energy_fn)(quats, transs)
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

    # Effective convergence tolerance: any remaining gap below this counts as
    # converged. Scaled to the total schedule so it represents a fixed RELATIVE
    # error (~1e-5 of β_target-β₀), then floored by the stall threshold so
    # boundary effects can never trigger a false-positive stall. The original
    # 1e-12 tolerance was orders of magnitude tighter than the 1e-8 stall
    # threshold, which let runs that had essentially reached β_target take
    # one more cooling step whose bisection upper bound was already < 1e-8 —
    # raising a spurious RuntimeError instead of accepting convergence.
    conv_tol = max(_PA_STALL_DBETA, 1e-5 * abs(float(beta_target) - float(beta0)))

    converged = False
    for _ in range(max_steps):
        if beta >= beta_target - conv_tol:
            converged = True
            break
        key, ks, kr = jax.random.split(key, 3)

        quats, transs, E = _pa_sweep_step(
            energy_fn, ks, quats, transs, jnp.asarray(beta),
            sigma_rot, sigma_trans, axis_mask, n_sweep)

        # Guard: a walker that drifted into +inf energy will force the
        # adaptive Δβ bisection to 0 (ESS at every trial Δβ is 0), and the
        # outer loop would silently spin until max_steps. Fail loudly with an
        # actionable message instead of returning a non-converged result.
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
        # Reindex energies by the SAME resample permutation so the returned
        # ``final_energies`` stay aligned with ``final_quats`` / ``final_transs``.
        # Within the loop E is recomputed by the next sweep, but on the final
        # cooling step (which breaks before another sweep) a non-identity
        # resample would otherwise leave E describing the PRE-resample walkers.
        E = E[idx]

        # One fused device->host transfer per cooling step: jax.device_get
        # pulls the (dbeta, logZ_inc, ess_frac) tuple in a single D2H roundtrip
        # instead of three.
        dbeta_h, logZ_inc_h, ess_frac_h = (
            float(x) for x in jax.device_get((dbeta, logZ_inc, ess_frac)))
        # Even without inf energies, bisection can collapse if ESS(target) is
        # below target at every Δβ (very heavy-tailed energy histogram). In
        # that case progress is impossible — bail out rather than silently
        # padding the schedule with zero-Δβ steps.
        #
        # Exception: when the remaining gap is already small relative to the
        # full schedule (< 100 × conv_tol ≈ 0.1 % of schedule), accept the
        # tiny step as convergence instead of raising. The residual logZ
        # contribution is bounded and the alternative (raising on a run that
        # was about to finish) is worse than the small bias of skipping it.
        # Heavy-tailed ESS collapse mid-schedule still raises as before.
        if dbeta_h < _PA_STALL_DBETA:
            remaining = float(beta_target) - beta
            if remaining < 100.0 * conv_tol:
                logger.warning(
                    "PA bisection collapsed near β_target (remaining=%.2e, "
                    "ESS at all trial Δβ < target_ess=%.2g). Treating as "
                    "converged — residual logZ contribution is bounded by "
                    "remaining·max|E|. Lower target_ess or add walkers for "
                    "a cleaner tail.", remaining, target_ess)
                # Add the bounded contribution and snap β to target.
                beta = float(beta_target)
                logZ += logZ_inc_h
                ess_hist.append(ess_frac_h)
                betas_hist.append(beta)
                converged = True
                break
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
            "Population annealing exhausted max_steps=%d before reaching "
            "beta_target=%.4f (final beta=%.4f, remaining=%.4f). "
            "Results may be biased; raise max_steps or lower target_ess.",
            max_steps, float(beta_target), beta, float(beta_target) - beta)

    return {
        "logZ_ratio": logZ,
        "final_quats": quats,
        "final_transs": transs,
        "final_energies": (None if E is None else np.asarray(E)),
        "ess": np.asarray(ess_hist),
        "betas": np.asarray(betas_hist),
        "n_steps": len(ess_hist),
        "converged": converged,
    }
