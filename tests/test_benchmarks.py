"""Tests for benchmark modules: _energy, protein_g_b1, lysozyme."""
import numpy as np
import pytest

from ptmc.benchmarks._energy import (
    make_continuum_energy, coarse_protein, _fibonacci_sphere,
)
from ptmc.benchmarks.protein_g_b1 import build_model as gb1_build, sample_contact_normals
from ptmc.benchmarks.lysozyme import build_model as lysozyme_build, patch_alignment


class TestFibonacciSphere:
    def test_shape(self):
        pts = _fibonacci_sphere(12, 0.5)
        assert pts.shape == (12, 3)

    def test_radius(self):
        pts = _fibonacci_sphere(100, 0.5)
        radii = np.linalg.norm(pts, axis=1)
        np.testing.assert_allclose(radii, np.full(100, 0.5), atol=1e-10)

    def test_n_positive(self):
        pts = _fibonacci_sphere(1, 1.0)
        assert pts.shape == (1, 3)


class TestCoarseProtein:
    def test_structure(self):
        xyz = _fibonacci_sphere(12, 0.5)
        atoms = coarse_protein(xyz, np.zeros(12), np.ones(12) * 1e-3,
                               np.ones(12) * 1e-6)
        assert atoms.n == 12
        assert atoms.q.shape == (12,)

    def test_default_names(self):
        xyz = _fibonacci_sphere(4, 0.3)
        atoms = coarse_protein(xyz, np.zeros(4), np.ones(4) * 1e-3,
                               np.ones(4) * 1e-6)
        assert atoms.names == ["B0", "B1", "B2", "B3"]
        assert atoms.resnames == ["BEA"] * 4


class TestMakeContinuumEnergy:
    def test_returns_callable(self):
        xyz = _fibonacci_sphere(4, 0.3)
        atoms = coarse_protein(xyz, np.zeros(4), np.ones(4) * 1e-3,
                               np.ones(4) * 1e-6)
        energy_fn = make_continuum_energy(atoms, 30.0, 1.0, 1.0,
                                          0.785, 0.2, 0.0)
        assert callable(energy_fn)

    def test_energy_finite_at_height(self):
        xyz = _fibonacci_sphere(4, 0.3)
        atoms = coarse_protein(xyz, np.zeros(4), np.ones(4) * 1e-3,
                               np.ones(4) * 1e-6)
        energy_fn = make_continuum_energy(atoms, 30.0, 1.0, 1.0,
                                          0.785, 0.2, 0.0)
        import jax.numpy as jnp
        q = jnp.array([1.0, 0.0, 0.0, 0.0])
        t = jnp.array([0.0, 0.0, 0.5])
        e = energy_fn(q, t)
        assert jnp.isfinite(e)

    def test_energy_inf_at_wall(self):
        xyz = _fibonacci_sphere(4, 0.3)
        atoms = coarse_protein(xyz, np.zeros(4), np.ones(4) * 1e-3,
                               np.ones(4) * 1e-6)
        energy_fn = make_continuum_energy(atoms, 30.0, 1.0, 1.0,
                                          0.785, 0.2, 0.0)
        import jax.numpy as jnp
        q = jnp.array([1.0, 0.0, 0.0, 0.0])
        t = jnp.array([0.0, 0.0, 0.0])  # z=0 < z_min=0.2
        e = energy_fn(q, t)
        assert e > 1e15  # large finite penalty, gradient-safe


class TestProteinGB1:
    def test_build_model(self):
        atoms = gb1_build(r=0.5, n=12)
        assert atoms.n == 12
        assert atoms.pos0.shape == (12, 3)

    def test_sample_runs(self):
        """Sample with minimal chains/steps; just check it doesn't crash."""
        cn = sample_contact_normals(C=8, steps=50, seed=0)
        assert cn.shape[1] == 3
        # Normals should be unit vectors
        norms = np.linalg.norm(cn, axis=1)
        np.testing.assert_allclose(norms, np.ones(8), atol=1e-6)


class TestLysozyme:
    def test_build_model(self):
        atoms = lysozyme_build(r=0.5, n=12, q0=0.8)
        assert atoms.n == 12
        # Should have charge anisotropy
        assert not np.allclose(atoms.q, 0.0)

    def test_patch_alignment_sign(self):
        """Positive patch should face oppositely-charged surface."""
        # psi0 > 0 attracts negative patch
        align_pos = patch_alignment(5.0, C=8, steps=50, seed=0)
        align_neg = patch_alignment(-5.0, C=8, steps=50, seed=1)
        assert align_pos != pytest.approx(align_neg, abs=0.5)
