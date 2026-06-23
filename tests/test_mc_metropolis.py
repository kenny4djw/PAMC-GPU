"""Tests for ptmc.mc.metropolis (JAX chain and batched Metropolis).

Tests run on CPU (forced in conftest.py).
Uses a simple harmonic energy to validate Boltzmann sampling.
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from ptmc.mc.metropolis import (
    accept_logprob, metropolis_step, run_chain, run_chains,
    split_chain_keys,
)
from ptmc.config import beta as beta_of


# ---------------------------------------------------------------------------
# Harmonic well energy for testing: E = 0.5 * k * |z|^2
# Boltzmann distribution: p(z) ~ exp(-beta * 0.5 * k * z^2)  ->  z ~ N(0, 1/(beta*k))
# ---------------------------------------------------------------------------
def harmonic_energy_z(quat, trans):
    """Energy depends only on z-translation: E = 0.5 * z^2."""
    return 0.5 * trans[2] ** 2


def harmonic_energy_3d(quat, trans):
    """Energy depends on all: E = 0.5 * |trans|^2."""
    return 0.5 * jnp.sum(trans ** 2)


class TestAcceptLogprob:
    def test_zero_dE(self):
        lp = accept_logprob(0.0, 1.0)
        assert lp == pytest.approx(0.0)

    def test_negative_dE(self):
        """Lower energy -> always accept."""
        lp = accept_logprob(-1.0, 1.0)
        assert lp == pytest.approx(0.0)

    def test_positive_dE(self):
        """Higher energy -> Metropolis factor."""
        lp = accept_logprob(1.0, 1.0)
        assert lp == pytest.approx(-1.0)

    def test_high_beta_less_accepting(self):
        """Higher beta -> more stringent for same dE."""
        lp_low = accept_logprob(1.0, 0.5)
        lp_high = accept_logprob(1.0, 2.0)
        assert lp_high < lp_low

    def test_clipped_at_zero(self):
        lp = accept_logprob(-10.0, 1.0)
        assert lp == pytest.approx(0.0)  # clamped to 0


class TestMetropolisStep:
    def test_deterministic_with_zero_sigma(self):
        """Zero proposal sigma -> no move -> acceptance is certain."""
        key = jax.random.PRNGKey(0)
        q = jnp.array([1.0, 0.0, 0.0, 0.0])
        t = jnp.array([0.0, 0.0, 0.3])
        E0 = harmonic_energy_z(q, t)
        q2, t2, E2, acc = metropolis_step(
            key, q, t, E0, harmonic_energy_z, 1.0,
            0.0, 0.0, jnp.array([0.0, 0.0, 1.0]))
        assert acc  # should accept (no move, same energy)
        np.testing.assert_allclose(q2, q, atol=1e-7)
        np.testing.assert_allclose(t2, t, atol=1e-7)
        assert E2 == pytest.approx(E0)

    def test_energy_decrease_accepted(self):
        """Proposal that reduces energy should always be accepted."""
        key = jax.random.PRNGKey(0)
        q = jnp.array([1.0, 0.0, 0.0, 0.0])
        # Starting far from origin (high energy), proposing towards origin
        t = jnp.array([0.0, 0.0, 3.0])
        E0 = harmonic_energy_z(q, t)
        # With big sigma, likely to find lower energy
        q2, t2, E2, acc = metropolis_step(
            key, q, t, E0, harmonic_energy_z, 1.0,
            0.0, 1.0, jnp.array([0.0, 0.0, 1.0]))
        if E2 < E0:
            assert acc  # definitely accept if energy decreased


class TestRunChain:
    def test_output_keys(self):
        key = jax.random.PRNGKey(42)
        q0 = jnp.array([1.0, 0.0, 0.0, 0.0])
        t0 = jnp.array([0.0, 0.0, 0.5])
        out = run_chain(key, q0, t0, harmonic_energy_z,
                        1.0, 0.1, 0.05, jnp.array([0.0, 0.0, 1.0]), 10)
        assert "quat" in out
        assert "trans" in out
        assert "energy" in out
        assert "accepted" in out
        assert "accept_rate" in out

    def test_output_shapes(self):
        key = jax.random.PRNGKey(42)
        q0 = jnp.array([1.0, 0.0, 0.0, 0.0])
        t0 = jnp.array([0.0, 0.0, 0.5])
        n_steps = 100
        out = run_chain(key, q0, t0, harmonic_energy_z,
                        1.0, 0.1, 0.05, jnp.array([0.0, 0.0, 1.0]), n_steps)
        assert out["quat"].shape == (n_steps, 4)
        assert out["trans"].shape == (n_steps, 3)
        assert out["energy"].shape == (n_steps,)
        assert out["accepted"].shape == (n_steps,)

    def test_accept_rate_in_range(self):
        key = jax.random.PRNGKey(42)
        q0 = jnp.array([1.0, 0.0, 0.0, 0.0])
        t0 = jnp.array([0.0, 0.0, 0.5])
        out = run_chain(key, q0, t0, harmonic_energy_z,
                        1.0, 0.1, 0.05, jnp.array([0.0, 0.0, 1.0]), 500)
        assert 0.0 <= out["accept_rate"] <= 1.0

    def test_trace_length(self):
        """Trajectory length matches n_steps."""
        for n in [1, 5, 100]:
            key = jax.random.PRNGKey(0)
            q0 = jnp.array([1.0, 0.0, 0.0, 0.0])
            t0 = jnp.array([0.0, 0.0, 0.5])
            out = run_chain(key, q0, t0, harmonic_energy_z,
                            1.0, 0.1, 0.05, jnp.array([0.0, 0.0, 1.0]), n)
            assert out["energy"].shape[0] == n


class TestRunChains:
    def test_batch_output_shapes(self):
        key = jax.random.PRNGKey(42)
        n_chains = 4
        n_steps = 50
        keys = split_chain_keys(key, n_chains)
        q0 = jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0]), (n_chains, 1))
        t0 = jnp.tile(jnp.array([0.0, 0.0, 0.5]), (n_chains, 1))
        out = run_chains(keys, q0, t0, harmonic_energy_z,
                         1.0, 0.1, 0.05, jnp.array([0.0, 0.0, 1.0]), n_steps)
        assert out["quat"].shape == (n_chains, n_steps, 4)
        assert out["trans"].shape == (n_chains, n_steps, 3)
        assert out["energy"].shape == (n_chains, n_steps)
        assert out["accepted"].shape == (n_chains, n_steps)
        assert out["accept_rate"].shape == (n_chains,)

    def test_chains_independent(self):
        """Different chains produce different trajectories."""
        key = jax.random.PRNGKey(42)
        n_chains = 4
        n_steps = 100
        keys = split_chain_keys(key, n_chains)
        q0 = jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0]), (n_chains, 1))
        t0 = jnp.tile(jnp.array([0.0, 0.0, 0.5]), (n_chains, 1))
        out = run_chains(keys, q0, t0, harmonic_energy_z,
                         1.0, 0.1, 0.05, jnp.array([0.0, 0.0, 1.0]), n_steps)
        # Check that not all final energies are identical
        energies = out["energy"][:, -1]
        assert not jnp.allclose(energies[0], energies[1:])

    def test_reproducible(self):
        """Same seed -> same results."""
        n_chains = 2
        n_steps = 20

        def run(seed):
            key = jax.random.PRNGKey(seed)
            keys = split_chain_keys(key, n_chains)
            q0 = jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0]), (n_chains, 1))
            t0 = jnp.tile(jnp.array([0.0, 0.0, 0.5]), (n_chains, 1))
            return run_chains(keys, q0, t0, harmonic_energy_z,
                              1.0, 0.1, 0.05, jnp.array([0.0, 0.0, 1.0]), n_steps)

        out1 = run(42)
        out2 = run(42)
        np.testing.assert_allclose(out1["energy"], out2["energy"])
        np.testing.assert_allclose(out1["accept_rate"], out2["accept_rate"])


class TestBoltzmannSampling:
    """Statistical test: chain on harmonic potential should produce
    correct Boltzmann distribution of z."""

    @pytest.mark.slow
    def test_z_variance(self):
        """For E=0.5*z^2 at beta=1, z ~ N(0,1). Var(z) ≈ 1."""
        key = jax.random.PRNGKey(42)
        q0 = jnp.array([1.0, 0.0, 0.0, 0.0])
        t0 = jnp.array([0.0, 0.0, 0.0])
        out = run_chain(key, q0, t0, harmonic_energy_z,
                        1.0, 0.3, 0.3, jnp.array([0.0, 0.0, 1.0]), 20_000)
        z_samples = out["trans"][500:, 2]  # burn-in 500
        var_z = jnp.var(z_samples)
        # Variance of z should be ~1/beta*k = 1/1*1 = 1
        assert var_z == pytest.approx(1.0, rel=0.2)

    @pytest.mark.slow
    def test_mean_energy(self):
        """Mean energy for harmonic oscillator at beta=1:
        <E> = 0.5*k*<z^2> = 0.5*k*Var(z) = 0.5.
        """
        key = jax.random.PRNGKey(42)
        q0 = jnp.array([1.0, 0.0, 0.0, 0.0])
        t0 = jnp.array([0.0, 0.0, 0.0])
        out = run_chain(key, q0, t0, harmonic_energy_z,
                        1.0, 0.3, 0.3, jnp.array([0.0, 0.0, 1.0]), 20_000)
        energies = out["energy"][500:]
        mean_E = jnp.mean(energies)
        # For harmonic: <E> = 0.5*k_B*T = 0.5/beta = 0.5
        assert mean_E == pytest.approx(0.5, rel=0.2)
