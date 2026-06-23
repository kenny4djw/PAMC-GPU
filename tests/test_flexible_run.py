"""F8 — flexible HT driver tests.

* K=0 (no chi DOF) runs cleanly.
* sigma_chi=0 freezes chi forever.
* Batched flexible_run_chains result equals single-chain scan_flexible_metropolis
  for the same seed (batch == individual contract).
* Boltzmann distribution recovered on a quadratic-chi target.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ptmc.mc.flex_metropolis import (
    dof_move_weights,
    scan_flexible_metropolis,
)
from ptmc.sampler.flexible_run import flexible_run_chains


# ---------------------------------------------------------------------------
# K=0 (rigid mode) sanity
# ---------------------------------------------------------------------------

def test_K0_runs_without_chi():
    """No chi DOF: just rot + trans proposals. Weights set to (1, 1, 0)
    avoid the K=0 chi proposal entirely.
    """
    def E(q, t, c):
        # Energy independent of state (free particle): all moves accepted.
        return jnp.array(0.0, dtype=jnp.float32)

    out = flexible_run_chains(
        master_seed=7, n_chains=4, n_steps=200,
        init_chi=jnp.zeros((0,), dtype=jnp.float32), z0=0.5,
        energy_fn=E, beta=1.0,
        sigma_rot=0.05, sigma_trans=0.05, sigma_chi=0.0,
        axis_mask=jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32),
        move_weights=jnp.log(jnp.array([1.0, 1.0, 1e-30],
                                       dtype=jnp.float32)),
    )
    assert out["quats"].shape == (4, 4)
    assert out["transs"].shape == (4, 3)
    assert out["chis"].shape == (4, 0)
    assert np.all(np.isfinite(out["energies"]))
    # All moves accepted (constant energy)
    rates = out["accept_rate_per_type"]
    # only rot / trans were tried
    assert np.allclose(rates[:, :2], 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# sigma_chi=0 freezes chi
# ---------------------------------------------------------------------------

def test_sigma_chi_zero_freezes_chi():
    """With sigma_chi=0 the proposed chi delta is always zero, so chi must
    equal its initial value at every step (and at the end)."""
    K = 4
    init_chi = jnp.asarray([0.1, -0.2, 0.3, -0.4], dtype=jnp.float32)

    def E(q, t, c):
        return jnp.sum(c * c)  # depends on chi → if chi changed we'd see it

    out = flexible_run_chains(
        master_seed=11, n_chains=8, n_steps=500,
        init_chi=init_chi, z0=0.5,
        energy_fn=E, beta=1.0,
        sigma_rot=0.05, sigma_trans=0.05, sigma_chi=0.0,
        axis_mask=jnp.ones(3, dtype=jnp.float32),
        move_weights=dof_move_weights(K),
    )
    expected = np.broadcast_to(np.asarray(init_chi), (8, K))
    np.testing.assert_allclose(out["chis"], expected, atol=1e-7)


# ---------------------------------------------------------------------------
# Batch == individual
# ---------------------------------------------------------------------------

def test_batch_equals_individual():
    """Two-chain batched result must match running each chain individually
    with the same per-chain RNG seed (fold_in semantic)."""
    K = 2

    def E(q, t, c):
        return 5.0 * jnp.sum(c * c) + jnp.sum(t * t)

    weights = dof_move_weights(K)
    init_chi = jnp.zeros((K,), dtype=jnp.float32)

    # Batched
    out_batch = flexible_run_chains(
        master_seed=33, n_chains=3, n_steps=300,
        init_chi=init_chi, z0=0.5,
        energy_fn=E, beta=1.0,
        sigma_rot=0.05, sigma_trans=0.05, sigma_chi=0.1,
        axis_mask=jnp.ones(3, dtype=jnp.float32),
        move_weights=weights,
    )

    # Individual: replicate the per-chain key + initial pose, run one chain
    master = jax.random.PRNGKey(33)
    cidx = jnp.arange(3, dtype=jnp.uint32)
    chain_keys = jax.vmap(jax.random.fold_in,
                          in_axes=(None, 0))(master, cidx)
    from ptmc.sampler.flexible_run import _initial_pose
    quat0, trans0 = _initial_pose(chain_keys, 0.5)

    for c in range(3):
        out_one = scan_flexible_metropolis(
            chain_keys[c], quat0[c], trans0[c], init_chi,
            E, 1.0,
            0.05, 0.05, 0.1,
            jnp.ones(3, dtype=jnp.float32), weights,
            300, start_step=0,
        )
        np.testing.assert_allclose(
            np.asarray(out_batch["quats"][c]),
            np.asarray(out_one["quat_final"]), atol=1e-5,
            err_msg=f"chain {c} quat mismatch")
        np.testing.assert_allclose(
            np.asarray(out_batch["chis"][c]),
            np.asarray(out_one["chi_final"]), atol=1e-5,
            err_msg=f"chain {c} chi mismatch")
        np.testing.assert_allclose(
            float(out_batch["energies"][c]),
            float(out_one["energy_final"]), atol=1e-4,
            err_msg=f"chain {c} energy mismatch")


# ---------------------------------------------------------------------------
# Distribution check: harmonic chi → variance 1/(beta k)
# ---------------------------------------------------------------------------

def test_chi_variance_matches_boltzmann_under_full_move_weights():
    """With full DOF-proportional weights (rot, trans, chi), the chi
    marginal still relaxes to Boltzmann. Looser tolerance than F7's
    chi-only test because (a) rot/trans steals trial budget and (b)
    we run over multiple chains."""
    K = 1
    k_spring = 50.0
    beta = 1.0
    expected_var = 1.0 / (beta * k_spring)
    init_chi = jnp.zeros((K,), dtype=jnp.float32)

    def E(q, t, c):
        return 0.5 * k_spring * jnp.sum(c * c)

    out = flexible_run_chains(
        master_seed=42, n_chains=16, n_steps=20_000,
        init_chi=init_chi, z0=0.5,
        energy_fn=E, beta=beta,
        sigma_rot=0.05, sigma_trans=0.05, sigma_chi=0.2,
        axis_mask=jnp.ones(3, dtype=jnp.float32),
        move_weights=dof_move_weights(K),
    )

    # Use the per-chain final chi as samples (independent chains).
    chis = np.asarray(out["chis"][:, 0])
    sample_var = chis.var()
    rel = abs(sample_var - expected_var) / expected_var
    # Few independent samples (16 chains) -> high SE; lax tolerance.
    assert rel < 0.5, (
        f"chi sample var {sample_var:.4f} vs expected "
        f"{expected_var:.4f}; rel = {rel:.3f}")


# ---------------------------------------------------------------------------
# Acceptance counter coverage
# ---------------------------------------------------------------------------

def test_acceptance_counters_partition_n_steps():
    K = 3

    def E(q, t, c):
        return jnp.sum(c * c)

    out = flexible_run_chains(
        master_seed=99, n_chains=5, n_steps=400,
        init_chi=jnp.zeros((K,), dtype=jnp.float32), z0=0.5,
        energy_fn=E, beta=1.0,
        sigma_rot=0.05, sigma_trans=0.05, sigma_chi=0.1,
        axis_mask=jnp.ones(3, dtype=jnp.float32),
        move_weights=dof_move_weights(K),
    )
    # Each chain's try_counts sums to n_steps.
    assert np.all(out["try_counts"].sum(axis=-1) == 400)
    # accept <= try elementwise
    assert np.all(out["acc_counts"] <= out["try_counts"])


def test_jit_smoke_run():
    """Smoke: flexible_run_chains compiles and produces finite output."""
    K = 2

    def E(q, t, c):
        return jnp.sum(c * c) + jnp.sum(t * t)

    out = flexible_run_chains(
        master_seed=1, n_chains=2, n_steps=50,
        init_chi=jnp.zeros((K,), dtype=jnp.float32), z0=0.5,
        energy_fn=E, beta=1.0,
        sigma_rot=0.05, sigma_trans=0.05, sigma_chi=0.1,
        axis_mask=jnp.ones(3, dtype=jnp.float32),
    )
    assert np.all(np.isfinite(out["energies"]))
