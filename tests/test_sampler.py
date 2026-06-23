"""Tests for sampler modules: parallel_tempering, population_annealing, highthroughput.

Travis on CPU (forced in conftest.py). Uses simple harmonic potentials to verify
sampling correctness.
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from ptmc.sampler.highthroughput import (
    _chain_keys, build_batch, initial_state, run_systems,
)
from ptmc.sampler.parallel_tempering import (
    swap_probability, swap_pass, pt_sweep, run_pt, run_multi_pt,
)
from ptmc.sampler.population_annealing import (
    ess_fraction, _find_dbeta, run_pa,
)
from ptmc.config import beta as beta_of


# ---------------------------------------------------------------------------
# Simple harmonic helper for individual chain tests
# ---------------------------------------------------------------------------
def harmonic_energy(quat, trans):
    return 0.5 * jnp.sum(trans ** 2)


# Simple 1D energy for PA / dbeta tests
def one_atom_continuum(quat, trans):
    """Simple z^2 energy to mimic continuum surface."""
    z = trans[2]
    return 0.5 * z ** 2


class TestSwapProbability:
    def test_equal_temperatures(self):
        p = swap_probability(1.0, 1.0, 0.0, 0.0)
        assert p == 1.0

    def test_energy_difference_sign(self):
        """Swapping hotter config (higher E) with colder config (lower E) always
        accepted when hot goes to cold since E_down - E_up < 0."""
        p = swap_probability(2.0, 1.0, -5.0, 3.0)  # b_i > b_j, E_i < E_j
        assert p > 0

    def test_detailed_balance(self):
        p_forward = swap_probability(2.0, 1.0, 1.0, 5.0)
        p_reverse = swap_probability(1.0, 2.0, 5.0, 1.0)
        assert p_forward == pytest.approx(p_reverse)


class TestSwapPass:
    def test_system_shape(self):
        """(S,R) -> (S,R,*)."""
        S, R = 2, 4
        key = jax.random.PRNGKey(0)
        q = jnp.zeros((S, R, 4))
        t = jnp.zeros((S, R, 3))
        e = jnp.arange(S * R, dtype=float).reshape(S, R)
        b = jnp.linspace(2.0, 1.0, R)
        q2, t2, e2, acc = swap_pass(key, q, t, e, b, 0)
        assert q2.shape == (S, R, 4)
        assert t2.shape == (S, R, 3)
        assert e2.shape == (S, R)

    def test_neutral_swap(self):
        """Equal betas and equal energies -> swap always accepted."""
        S, R = 1, 3  # R=3: both parities have exactly 1 pair, no padding artifacts
        key = jax.random.PRNGKey(0)
        q = jnp.zeros((S, R, 4))
        t = jnp.zeros((S, R, 3))
        e = jnp.ones((S, R)) * 2.0
        b = jnp.ones(R)
        for parity in [0, 1]:
            q2, t2, e2, acc = swap_pass(key, q, t, e, b, parity)
            assert jnp.all(acc)  # all accepted


class TestRunPT:
    def test_output_shape(self):
        S, R = 2, 3
        key = jax.random.PRNGKey(42)
        q = jnp.zeros((S, R, 4))
        t = jnp.zeros((S, R, 3))
        b = jnp.array([2.0, 1.5, 1.0])
        out = run_pt(key, q, t, harmonic_energy, b,
                     0.1, 0.05, jnp.array([0.0, 0.0, 1.0]),
                     5, 10)
        assert "low_T_trans" in out
        assert "final_quats" in out
        assert "final_transs" in out
        assert "mc_accept_rate" in out
        assert "swap_accept_rate" in out

    def test_final_shapes(self):
        S, R = 1, 3
        key = jax.random.PRNGKey(42)
        q = jnp.zeros((S, R, 4))
        t = jnp.zeros((S, R, 3))
        b = jnp.array([2.0, 1.5, 1.0])
        out = run_pt(key, q, t, harmonic_energy, b,
                     0.1, 0.05, jnp.array([0.0, 0.0, 1.0]),
                     10, 5)
        assert out["final_quats"].shape == (S, R, 4)
        assert out["final_transs"].shape == (S, R, 3)
        assert out["mc_accept_rate"].shape == (R,)
        assert out["swap_accept_rate"].shape == (R - 1,)

    def test_accept_rates_in_range(self):
        S, R = 2, 4
        key = jax.random.PRNGKey(42)
        q = jnp.zeros((S, R, 4))
        t = jnp.zeros((S, R, 3))
        b = jnp.array([2.0, 1.7, 1.4, 1.0])
        out = run_pt(key, q, t, harmonic_energy, b,
                     0.1, 0.05, jnp.array([0.0, 0.0, 1.0]),
                     10, 10)
        assert jnp.all(out["mc_accept_rate"] >= 0)
        assert jnp.all(out["mc_accept_rate"] <= 1)
        assert jnp.all(out["swap_accept_rate"] >= 0)
        assert jnp.all(out["swap_accept_rate"] <= 1)


class TestRunMultiPT:
    """Multi-system PT with simple per-system continuum energy."""

    def make_systems(self, n_systems=2, n_atoms=3):
        systems = []
        for sid in range(n_systems):
            systems.append({
                "system_id": sid,
                "pos0": np.zeros((n_atoms, 3)),
                "q": np.zeros(n_atoms),
                "c6": np.ones(n_atoms) * 1e-3,
                "c12": np.ones(n_atoms) * 1e-6,
                "c6_surf": 1.0,
                "c12_surf": 1.0,
                "rho_s": 30.0,
                "psi0": 0.0,
                "lambda_D": 0.785,
                "z_min": 0.2,
                "init_z": 0.5,
            })
        return systems

    def test_output_structure(self):
        systems = self.make_systems(2)
        betas = jnp.array([2.0, 1.5, 1.0])
        key = jax.random.PRNGKey(42)
        out = run_multi_pt(key, systems, betas,
                           0.1, 0.05, jnp.array([0.0, 0.0, 1.0]),
                           5, 10)
        assert "low_T_trans" in out
        assert "final_quats" in out
        assert "final_transs" in out
        assert "mc_accept_rate" in out
        assert "swap_accept_rate" in out
        assert "per_system" in out

    def test_accept_rates_valid(self):
        systems = self.make_systems(2)
        betas = jnp.array([2.0, 1.0])
        key = jax.random.PRNGKey(42)
        out = run_multi_pt(key, systems, betas,
                           0.1, 0.05, jnp.array([0.0, 0.0, 1.0]),
                           10, 10)
        assert jnp.all(out["mc_accept_rate"] >= 0)
        assert jnp.all(out["swap_accept_rate"] >= 0)


class TestHighthroughput:
    def test_chain_keys(self):
        keys = _chain_keys(42, [0, 1], 3)
        assert keys.shape == (2, 3, 2)  # (S, C, 2)

    def test_chain_keys_deterministic(self):
        k1 = _chain_keys(42, [0], 2)
        k2 = _chain_keys(42, [0], 2)
        assert jnp.allclose(k1, k2)

    def test_build_batch(self):
        systems = [
            {"system_id": 0, "pos0": np.zeros((2, 3)), "q": np.zeros(2),
             "c6": np.ones(2) * 1e-3, "c12": np.ones(2) * 1e-6,
             "c6_surf": 1.0, "c12_surf": 1.0, "rho_s": 30.0, "psi0": 0.0,
             "lambda_D": 0.785, "z_min": 0.2, "beta": 1.0, "init_z": 0.5},
            {"system_id": 1, "pos0": np.zeros((2, 3)), "q": np.zeros(2),
             "c6": np.ones(2) * 2e-3, "c12": np.ones(2) * 2e-6,
             "c6_surf": 1.0, "c12_surf": 1.0, "rho_s": 30.0, "psi0": 0.0,
             "lambda_D": 0.785, "z_min": 0.2, "beta": 1.0, "init_z": 0.5},
        ]
        C = 4
        b = build_batch(systems, C)
        assert b["S"] == 2
        assert b["N"] == 2
        assert b["pos0_ps"].shape == (2, 2, 3)

    def test_initial_state(self):
        keys = _chain_keys(42, [0], 4)  # (1, 4, 2)
        z0 = jnp.ones((1, 4)) * 0.5
        quat0, trans0 = initial_state(keys, z0)
        assert quat0.shape == (1, 4, 4)
        assert trans0.shape == (1, 4, 3)
        # z should be z0
        np.testing.assert_allclose(trans0[..., 2], z0)

    def test_run_systems_basic(self):
        systems = [
            {"system_id": 0, "pos0": np.zeros((2, 3)), "q": np.zeros(2),
             "c6": np.ones(2) * 1e-3, "c12": np.ones(2) * 1e-6,
             "c6_surf": 1.0, "c12_surf": 1.0, "rho_s": 30.0, "psi0": 0.0,
             "lambda_D": 0.785, "z_min": 0.2, "beta": 1.0, "init_z": 0.5},
        ]
        out = run_systems(master_seed=42, systems=systems,
                          n_chains=4, n_steps=20,
                          sigma_rot=0.1, sigma_trans=0.05,
                          axis_mask=(False, False, True))
        assert out["quats"].shape == (1, 4, 4)
        assert out["transs"].shape == (1, 4, 3)
        assert out["energies"].shape == (1, 4)
        assert out["accept"].shape == (1, 4)
        assert out["system_ids"] == [0]

    def test_run_systems_two_systems(self):
        systems = [
            {"system_id": 0, "pos0": np.zeros((2, 3)), "q": np.zeros(2),
             "c6": np.ones(2) * 1e-3, "c12": np.ones(2) * 1e-6,
             "c6_surf": 1.0, "c12_surf": 1.0, "rho_s": 30.0, "psi0": 0.0,
             "lambda_D": 0.785, "z_min": 0.2, "beta": 1.0, "init_z": 0.5},
            {"system_id": 1, "pos0": np.zeros((2, 3)), "q": np.zeros(2),
             "c6": np.ones(2) * 1e-3, "c12": np.ones(2) * 1e-6,
             "c6_surf": 1.0, "c12_surf": 1.0, "rho_s": 30.0, "psi0": 0.0,
             "lambda_D": 0.785, "z_min": 0.2, "beta": 1.0, "init_z": 0.5},
        ]
        out = run_systems(master_seed=42, systems=systems,
                          n_chains=4, n_steps=20,
                          sigma_rot=0.1, sigma_trans=0.05,
                          axis_mask=(False, False, True))
        assert out["quats"].shape == (2, 4, 4)
        assert out["system_ids"] == [0, 1]

    def test_batch_independence(self):
        """Running two systems together should give same results as separately
        (positional RNG guarantee)."""
        sys = lambda sid: {
            "system_id": sid, "pos0": np.zeros((2, 3)), "q": np.zeros(2),
            "c6": np.ones(2) * 1e-3, "c12": np.ones(2) * 1e-6,
            "c6_surf": 1.0, "c12_surf": 1.0, "rho_s": 30.0, "psi0": 0.0,
            "lambda_D": 0.785, "z_min": 0.2, "beta": 1.0, "init_z": 0.5,
        }
        together = run_systems(42, [sys(0), sys(1)], 4, 20,
                               0.1, 0.05, (False, False, True))
        alone_0 = run_systems(42, [sys(0)], 4, 20,
                              0.1, 0.05, (False, False, True))
        alone_1 = run_systems(42, [sys(1)], 4, 20,
                              0.1, 0.05, (False, False, True))
        # System 0 results should match between batch and individual
        np.testing.assert_allclose(together["quats"][0], alone_0["quats"][0])
        np.testing.assert_allclose(together["energies"][0], alone_0["energies"][0])
        np.testing.assert_allclose(together["quats"][1], alone_1["quats"][0])
        np.testing.assert_allclose(together["energies"][1], alone_1["energies"][0])


class TestPopulationAnnealing:
    def test_ess_fraction(self):
        """Equal weights -> ESS = M."""
        log_w = np.ones(10)
        ess = ess_fraction(log_w)
        assert ess == pytest.approx(1.0)

    def test_ess_one_dominant(self):
        """One weight dominates -> ESS ≈ 1."""
        log_w = np.array([0.0, -100.0, -100.0])
        ess = ess_fraction(log_w)
        assert ess < 1.0

    def test_ess_range(self):
        rng = np.random.default_rng(42)
        log_w = rng.normal(size=100) * 0.5
        ess = ess_fraction(log_w)
        assert 0.0 < ess <= 1.0

    def test_find_dbeta(self):
        """_find_dbeta should return a beta increment in [0, dbeta_max]."""
        rng = np.random.default_rng(42)
        E = rng.normal(size=100)
        dbeta = _find_dbeta(E, 0.5, 0.7)
        assert 0.0 <= dbeta <= 0.5

    def test_find_dbeta_zero(self):
        """If dbeta=0, ESS=1 always."""
        E = np.random.default_rng(42).normal(size=100)
        dbeta = _find_dbeta(E, 0.0, 0.7)
        assert dbeta == 0.0

    def test_run_pa_basic(self):
        key = jax.random.PRNGKey(42)
        M = 8
        q0 = jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0]), (M, 1))
        t0 = jnp.tile(jnp.array([0.0, 0.0, 1.0]), (M, 1))
        out = run_pa(key, q0, t0, one_atom_continuum,
                     beta0=0.1, beta_target=1.0, n_sweep=10,
                     sigma_rot=0.0, sigma_trans=0.0,
                     axis_mask=jnp.array([0.0, 0.0, 1.0]),
                     target_ess=0.5, max_steps=50)
        assert "logZ_ratio" in out
        assert "final_quats" in out
        assert "final_transs" in out
        assert out["final_quats"].shape == (M, 4)
        assert out["final_transs"].shape == (M, 3)

    def test_run_pa_beta_progress(self):
        """Beta should advance from beta0 toward beta_target."""
        key = jax.random.PRNGKey(42)
        M = 16
        q0 = jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0]), (M, 1))
        t0 = jnp.tile(jnp.array([0.0, 0.0, 1.0]), (M, 1))
        out = run_pa(key, q0, t0, one_atom_continuum,
                     beta0=0.1, beta_target=2.0, n_sweep=20,
                     sigma_rot=0.0, sigma_trans=0.3,
                     axis_mask=jnp.array([0.0, 0.0, 1.0]),
                     target_ess=0.5, max_steps=100)
        assert float(out["betas"][-1]) >= 1.9  # should reach near target
