"""F7 — chi MC trial + 3-way acceptance counter + Boltzmann gate.

Acceptance gate (§ 11 / § 6): on a toy potential E(chi) = 0.5 k * sum(chi^2)
the chi-only sampler must converge to N(0, sigma^2) with sigma^2 = 1/(beta k).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ptmc.mc.flex_metropolis import (
    dof_move_weights,
    flexible_metropolis_step,
    scan_flexible_metropolis,
)
from ptmc.mc.moves import propose_chi


# ---------------------------------------------------------------------------
# propose_chi
# ---------------------------------------------------------------------------

def test_propose_chi_changes_only_one_component():
    chi = jnp.asarray(np.zeros(10, dtype=np.float32))
    key = jax.random.PRNGKey(0)
    new_chi, k = propose_chi(key, chi, sigma_chi=0.5)
    diff = np.asarray(new_chi) - np.asarray(chi)
    nonzero = np.flatnonzero(diff)
    assert len(nonzero) == 1
    assert nonzero[0] == int(k)


def test_propose_chi_distribution_is_normal():
    """Aggregate many proposals -> deltas are N(0, sigma^2)."""
    chi = jnp.zeros((4,), dtype=jnp.float32)
    sigma = 0.3
    deltas = []
    for s in range(5000):
        key = jax.random.PRNGKey(s)
        new_chi, k = propose_chi(key, chi, sigma_chi=sigma)
        delta = float(new_chi[int(k)] - chi[int(k)])
        deltas.append(delta)
    deltas = np.asarray(deltas)
    # Sample mean ≈ 0 (3-sigma bound at n=5000)
    assert abs(deltas.mean()) < 4.0 * sigma / np.sqrt(5000)
    # Sample std ≈ sigma (within ~1%)
    assert abs(deltas.std(ddof=1) - sigma) < 0.02


# ---------------------------------------------------------------------------
# 3-way acceptance counter integrity
# ---------------------------------------------------------------------------

def test_counter_total_matches_n_steps():
    """Counter sums (rot + trans + chi) should equal n_steps."""
    K = 5
    init_chi = jnp.zeros((K,), dtype=jnp.float32)
    init_quat = jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    init_trans = jnp.zeros((3,), dtype=jnp.float32)

    def E(q, t, c):
        return jnp.sum(c * c)  # cheap

    key = jax.random.PRNGKey(42)
    out = scan_flexible_metropolis(
        key, init_quat, init_trans, init_chi,
        E, beta=1.0,
        sigma_rot=0.1, sigma_trans=0.1, sigma_chi=0.1,
        axis_mask=jnp.ones(3, dtype=jnp.float32),
        move_weights=dof_move_weights(K),
        n_steps=2000,
    )
    n_tries = int(out["try_counts"].sum())
    assert n_tries == 2000


def test_counter_obeys_categorical_weights():
    """With weights (1, 0, 0) we should get 100% rot tries."""
    K = 5
    weights = jnp.log(jnp.array([1.0, 1e-30, 1e-30], dtype=jnp.float32))
    init_chi = jnp.zeros((K,), dtype=jnp.float32)
    init_quat = jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    init_trans = jnp.zeros((3,), dtype=jnp.float32)

    def E(q, t, c):
        return jnp.array(0.0, dtype=jnp.float32)

    out = scan_flexible_metropolis(
        jax.random.PRNGKey(7), init_quat, init_trans, init_chi,
        E, beta=1.0,
        sigma_rot=0.05, sigma_trans=0.05, sigma_chi=0.05,
        axis_mask=jnp.ones(3, dtype=jnp.float32),
        move_weights=weights, n_steps=200,
    )
    tries = np.asarray(out["try_counts"])
    assert tries[0] == 200
    assert tries[1] == 0
    assert tries[2] == 0


# ---------------------------------------------------------------------------
# Boltzmann distribution gate (the real F7 deliverable)
# ---------------------------------------------------------------------------

def test_chi_marginal_is_boltzmann_harmonic():
    """Toy potential E(chi) = 0.5 k * chi^2 (one chi DOF). The marginal
    distribution at temperature T should be N(0, sigma^2) with
    sigma^2 = 1 / (beta k).

    With chi-only proposals (weights (0, 0, 1)) the rigid-body DOF is
    irrelevant -- this is a single-particle harmonic-oscillator sampler.
    """
    K = 1
    k_spring = 50.0  # kJ/mol/rad^2
    beta = 1.0       # 1 / (kT)  -- kT == 1 kJ/mol
    sigma_true = np.sqrt(1.0 / (beta * k_spring))

    weights = jnp.log(jnp.array([1e-30, 1e-30, 1.0], dtype=jnp.float32))
    init_chi = jnp.zeros((K,), dtype=jnp.float32)
    init_quat = jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    init_trans = jnp.zeros((3,), dtype=jnp.float32)

    def E(q, t, c):
        return 0.5 * k_spring * jnp.sum(c * c)

    # Sample by running long chains and recording chi values via trajectory
    # collection. The scan_flexible_metropolis kernel does not currently
    # stream trajectory; do it manually here.
    chi_samples = []
    state = (init_quat, init_trans, init_chi, E(init_quat, init_trans, init_chi))
    sigma_chi = 0.2  # ~optimal step (≈ 1.5 sigma_true)
    n_steps = 50_000
    burn_in = 2000

    key = jax.random.PRNGKey(2026)

    @jax.jit
    def step(carry, t):
        q, tr, ch, e = carry
        k_step = jax.random.fold_in(key, t)
        q, tr, ch, e, acc, mt = flexible_metropolis_step(
            k_step, q, tr, ch, e, E, beta,
            0.05, 0.05, sigma_chi,
            jnp.ones(3, dtype=jnp.float32), weights)
        return (q, tr, ch, e), ch

    (q_f, tr_f, ch_f, e_f), traj = jax.lax.scan(
        step, state, jnp.arange(n_steps))
    chi_samples = np.asarray(traj[burn_in:, 0])

    sample_var = chi_samples.var()
    expected_var = 1.0 / (beta * k_spring)
    rel = abs(sample_var - expected_var) / expected_var
    assert rel < 0.05, (
        f"sample variance {sample_var:.5f} vs expected {expected_var:.5f}; "
        f"rel diff {rel:.3f} > 0.05 — Boltzmann gate failed")

    # Mean should be ≈ 0 (sample std / sqrt(N))
    mean_se = np.sqrt(expected_var / len(chi_samples))
    assert abs(chi_samples.mean()) < 4.0 * mean_se, (
        f"sample mean {chi_samples.mean():.5f} not within 4 SE = "
        f"{4 * mean_se:.5f} of 0")


def test_accept_rate_is_reasonable_for_tuned_sigma():
    """Confirm the harmonic-toy run yielded a non-degenerate acceptance rate
    (somewhere in 30 - 70 %). Catches a regression where every move is
    rejected (would still pass variance test if the sampler stayed at 0)."""
    K = 1
    weights = jnp.log(jnp.array([1e-30, 1e-30, 1.0], dtype=jnp.float32))
    init_chi = jnp.zeros((K,), dtype=jnp.float32)
    init_quat = jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    init_trans = jnp.zeros((3,), dtype=jnp.float32)

    k_spring = 50.0
    sigma_chi = 0.2

    def E(q, t, c):
        return 0.5 * k_spring * jnp.sum(c * c)

    out = scan_flexible_metropolis(
        jax.random.PRNGKey(99), init_quat, init_trans, init_chi,
        E, beta=1.0,
        sigma_rot=0.05, sigma_trans=0.05, sigma_chi=sigma_chi,
        axis_mask=jnp.ones(3, dtype=jnp.float32),
        move_weights=weights, n_steps=5000,
    )
    rate = float(out["accept_rate_per_type"][2])
    assert 0.3 < rate < 0.7, f"chi accept rate = {rate:.3f}"


# ---------------------------------------------------------------------------
# JIT smoke
# ---------------------------------------------------------------------------

def test_scan_flexible_metropolis_jits():
    K = 3
    init_chi = jnp.zeros((K,), dtype=jnp.float32)
    init_quat = jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    init_trans = jnp.zeros((3,), dtype=jnp.float32)

    def E(q, t, c):
        return jnp.sum(c * c)

    fn = jax.jit(scan_flexible_metropolis, static_argnums=(4, 11, 12))
    out = fn(jax.random.PRNGKey(0), init_quat, init_trans, init_chi,
             E, 1.0, 0.05, 0.05, 0.1,
             jnp.ones(3, dtype=jnp.float32),
             dof_move_weights(K), 100, None)
    assert np.isfinite(float(out["energy_final"]))
