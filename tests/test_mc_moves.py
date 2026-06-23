"""Tests for ptmc.mc.moves (JAX quaternion operations and MC proposals).

JAX is forced to CPU in conftest.py for deterministic testing.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ptmc.mc.moves import (
    quat_mul, normalize_quat, axisangle_to_quat,
    quat_rotate, propose_rotation, propose_translation,
)

_N_SAMPLES = 500  # samples for statistical tests


class TestQuatMul:
    def test_identity(self):
        q_ident = jnp.array([1.0, 0.0, 0.0, 0.0])
        q = jnp.array([0.1, 0.2, 0.3, 0.4])
        q = q / jnp.linalg.norm(q)
        result = quat_mul(q_ident, q)
        np.testing.assert_allclose(result, q, atol=1e-7)

    def test_inverse(self):
        """q * conj(q) = identity (for unit quat)."""
        q = jnp.array([0.6, 0.1, 0.2, 0.3])
        q = q / jnp.linalg.norm(q)
        conj = q.at[1:].multiply(-1)
        result = quat_mul(q, conj)
        np.testing.assert_allclose(result, [1.0, 0.0, 0.0, 0.0], atol=1e-7)

    def test_associative(self, rng):
        a = jnp.array(rng.normal(size=4))
        b = jnp.array(rng.normal(size=4))
        c = jnp.array(rng.normal(size=4))
        ab_c = quat_mul(quat_mul(a, b), c)
        a_bc = quat_mul(a, quat_mul(b, c))
        np.testing.assert_allclose(ab_c, a_bc, atol=1e-6)


class TestNormalizeQuat:
    def test_normalized_stays(self):
        q = jnp.array([1.0, 0.0, 0.0, 0.0])
        result = normalize_quat(q)
        np.testing.assert_allclose(result, q, atol=1e-7)

    def test_normalizes(self):
        q = jnp.array([2.0, 0.0, 0.0, 0.0])
        result = normalize_quat(q)
        np.testing.assert_allclose(result, [1.0, 0.0, 0.0, 0.0], atol=1e-7)

    def test_batch(self, rng):
        qs = jnp.array(rng.normal(size=(10, 4)))
        result = normalize_quat(qs)
        norms = jnp.linalg.norm(result, axis=-1)
        np.testing.assert_allclose(norms, jnp.ones(10), atol=1e-7)


class TestAxisAngleToQuat:
    def test_zero_rotation(self):
        q = axisangle_to_quat(jnp.zeros(3))
        np.testing.assert_allclose(q, [1.0, 0.0, 0.0, 0.0], atol=1e-7)

    def test_90_deg_z(self):
        """90 deg about z -> (cos45, 0, 0, sin45)."""
        omega = jnp.array([0.0, 0.0, np.pi / 2])
        q = axisangle_to_quat(omega)
        expected = jnp.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
        np.testing.assert_allclose(q, expected, atol=1e-7)

    def test_small_angle(self):
        """Very small angle should not produce NaN."""
        omega = jnp.array([1e-10, 0.0, 0.0])
        q = axisangle_to_quat(omega)
        assert jnp.all(jnp.isfinite(q))

    def test_unit_norm(self, rng):
        for _ in range(10):
            omega = jnp.array(rng.normal(size=3))
            q = axisangle_to_quat(omega)
            norm = jnp.linalg.norm(q)
            assert norm == pytest.approx(1.0, abs=2e-7)


class TestQuatRotate:
    def test_identity_rotation(self):
        q = jnp.array([1.0, 0.0, 0.0, 0.0])
        v = jnp.array([1.0, 2.0, 3.0])
        result = quat_rotate(q, v)
        np.testing.assert_allclose(result, v, atol=1e-7)

    def test_180_deg_z(self):
        """180 deg about z: (x,y,z) -> (-x,-y,z)."""
        q = jnp.array([0.0, 0.0, 0.0, 1.0])  # 180 deg about z
        v = jnp.array([1.0, 2.0, 3.0])
        result = quat_rotate(q, v)
        np.testing.assert_allclose(result, [-1.0, -2.0, 3.0], atol=1e-7)

    def test_preserves_norm(self, rng):
        q = jnp.array(rng.normal(size=4))
        q = q / jnp.linalg.norm(q)
        v = jnp.array(rng.normal(size=3))
        result = quat_rotate(q, v)
        assert jnp.linalg.norm(result) == pytest.approx(jnp.linalg.norm(v), rel=1e-4)

    def test_batch_rotation(self):
        """vmap over multiple vectors."""
        q = jnp.array([1.0, 0.0, 0.0, 0.0])
        vs = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        # Use vmap
        batch_rotate = jax.vmap(lambda v: quat_rotate(q, v))
        results = batch_rotate(vs)
        np.testing.assert_allclose(results, vs, atol=1e-7)


class TestProposeRotation:
    def test_output_shape(self):
        key = jax.random.PRNGKey(0)
        q = jnp.array([1.0, 0.0, 0.0, 0.0])
        q_new = propose_rotation(key, q, 0.1)
        assert q_new.shape == (4,)

    def test_output_is_unit(self):
        key = jax.random.PRNGKey(0)
        q = jnp.array([1.0, 0.0, 0.0, 0.0])
        for i in range(10):
            k = jax.random.fold_in(key, i)
            q_new = propose_rotation(k, q, 0.1)
            assert jnp.linalg.norm(q_new) == pytest.approx(1.0, abs=1e-7)

    def test_zero_sigma_returns_same(self):
        key = jax.random.PRNGKey(0)
        q = jnp.array([0.6, 0.1, 0.2, 0.3])
        q = q / jnp.linalg.norm(q)
        q_new = propose_rotation(key, q, 0.0)
        np.testing.assert_allclose(q_new, q, atol=1e-7)

    def test_statistically_symmetric(self, rng):
        """Mean angular displacement over many proposals is zero (symmetric)."""
        q = jnp.array([1.0, 0.0, 0.0, 0.0])
        key = jax.random.PRNGKey(42)
        vec_parts = []
        for i in range(_N_SAMPLES):
            k = jax.random.fold_in(key, i)
            q_new = propose_rotation(k, q, 0.3)
            # The implied delta-quaternion = q_new * conj(q) = q_new
            # Vector part (x,y,z) has zero mean under symmetric proposal
            vec_parts.append(q_new[1:4])
        mean_vec = np.mean(np.array(vec_parts), axis=0)
        # Mean of vector part should be near zero
        np.testing.assert_allclose(mean_vec, np.zeros(3), atol=0.05)


class TestProposeTranslation:
    def test_output_shape(self):
        key = jax.random.PRNGKey(0)
        t = jnp.array([0.0, 0.0, 0.5])
        axis_mask = jnp.array([0.0, 0.0, 1.0])
        t_new = propose_translation(key, t, 0.05, axis_mask)
        assert t_new.shape == (3,)

    def test_z_only_mask(self):
        """With mask (0,0,1), only z changes."""
        key = jax.random.PRNGKey(0)
        t = jnp.array([0.1, 0.2, 0.5])
        axis_mask = jnp.array([0.0, 0.0, 1.0])
        t_new = propose_translation(key, t, 0.05, axis_mask)
        assert t_new[0] == t[0]  # x unchanged
        assert t_new[1] == t[1]  # y unchanged
        # z may change

    def test_full_mask(self):
        """With mask (1,1,1), all axes can change."""
        key = jax.random.PRNGKey(0)
        t = jnp.array([0.0, 0.0, 0.5])
        axis_mask = jnp.array([1.0, 1.0, 1.0])
        t_new = propose_translation(key, t, 0.05, axis_mask)
        # Very unlikely all three are identical
        assert not jnp.allclose(t_new, t)

    def test_zero_sigma_returns_same(self):
        key = jax.random.PRNGKey(0)
        t = jnp.array([0.1, 0.2, 0.5])
        axis_mask = jnp.array([1.0, 1.0, 1.0])
        t_new = propose_translation(key, t, 0.0, axis_mask)
        np.testing.assert_allclose(t_new, t, atol=1e-7)

    def test_zero_mask_returns_same(self):
        key = jax.random.PRNGKey(0)
        t = jnp.array([0.1, 0.2, 0.5])
        axis_mask = jnp.array([0.0, 0.0, 0.0])
        t_new = propose_translation(key, t, 0.1, axis_mask)
        np.testing.assert_allclose(t_new, t, atol=1e-7)
