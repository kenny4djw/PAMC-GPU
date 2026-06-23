"""Tests for ptmc.model.structures: Atoms, Pose, surfaces, quaternion helpers."""
import numpy as np
import pytest

from ptmc.model.structures import (
    Atoms, Pose, quat_to_matrix,
    DiscreteSurface, ContinuumSurface,
)
from ptmc.config import COULOMB_FACTOR_KJ_NM_PER_E2


class TestAtoms:
    def test_shape_validation(self, three_atom_atoms):
        assert three_atom_atoms.n == 3
        assert three_atom_atoms.pos0.shape == (3, 3)
        assert three_atom_atoms.q.shape == (3,)
        assert three_atom_atoms.c6.shape == (3,)
        assert three_atom_atoms.c12.shape == (3,)
        assert three_atom_atoms.resids.shape == (3,)

    def test_net_charge(self, three_atom_atoms):
        assert three_atom_atoms.net_charge == pytest.approx(0.0)

    def test_net_charge_diatomic(self, diatomic):
        assert diatomic.net_charge == pytest.approx(0.0)

    def test_sqrt_c6(self, three_atom_atoms):
        expected = np.sqrt(three_atom_atoms.c6)
        np.testing.assert_allclose(three_atom_atoms.sqrt_c6, expected)

    def test_sqrt_c12(self, three_atom_atoms):
        expected = np.sqrt(three_atom_atoms.c12)
        np.testing.assert_allclose(three_atom_atoms.sqrt_c12, expected)

    def test_atom_count_mismatch_raises(self):
        with pytest.raises(ValueError):
            Atoms(pos0=np.zeros((2, 3)),
                  q=np.zeros(3), c6=np.zeros(3), c12=np.zeros(3),
                  names=["A", "B", "C"], resids=np.zeros(3),
                  resnames=["X"]*3, elements=["C"]*3)

    def test_name_list_length(self, three_atom_atoms):
        assert len(three_atom_atoms.names) == 3
        assert len(three_atom_atoms.resnames) == 3
        assert len(three_atom_atoms.elements) == 3

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError):
            Atoms(pos0=np.zeros((3, 2)), q=np.zeros(3),
                  c6=np.zeros(3), c12=np.zeros(3),
                  names=["A"]*3, resids=np.zeros(3, int),
                  resnames=["X"]*3, elements=["C"]*3)


class TestQuatToMatrix:
    def test_identity(self):
        R = quat_to_matrix(np.array([1.0, 0.0, 0.0, 0.0]))
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)

    def test_rotation_180_z(self):
        """180 deg about z: x->-x, y->-y, z->z."""
        R = quat_to_matrix(np.array([0.0, 0.0, 0.0, 1.0]))
        expected = np.diag([-1.0, -1.0, 1.0])
        np.testing.assert_allclose(R, expected, atol=1e-12)

    def test_rotation_90_x(self):
        """90 deg about x: y->z, z->-y."""
        q = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])
        R = quat_to_matrix(q)
        v = R @ np.array([0.0, 1.0, 0.0])
        np.testing.assert_allclose(v, [0.0, 0.0, 1.0], atol=1e-12)

    def test_orthogonal(self, rng):
        for _ in range(10):
            q = rng.normal(size=4)
            q = q / np.linalg.norm(q)
            R = quat_to_matrix(q)
            np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
            assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-10)

    def test_zero_quat_returns_nan(self):
        """Zero quaternion produces NaN (known limitation, not protected)."""
        R = quat_to_matrix(np.zeros(4))
        assert np.any(np.isnan(R))


class TestPose:
    def test_identity_translates_nothing(self, three_atom_atoms):
        p = Pose.identity()
        out = p.apply(three_atom_atoms.pos0)
        np.testing.assert_allclose(out, three_atom_atoms.pos0)

    def test_translation(self, three_atom_atoms):
        t = np.array([0.1, 0.2, 0.3])
        p = Pose(quat=np.array([1.0, 0.0, 0.0, 0.0]), trans=t)
        out = p.apply(three_atom_atoms.pos0)
        expected = three_atom_atoms.pos0 + t
        np.testing.assert_allclose(out, expected)

    def test_rotation_preserves_distances(self, three_atom_atoms, rng):
        q = rng.normal(size=4)
        q = q / np.linalg.norm(q)
        p = Pose(quat=q, trans=np.zeros(3))
        out = p.apply(three_atom_atoms.pos0)
        # Pairwise distances preserved
        d0 = np.linalg.norm(three_atom_atoms.pos0[0] - three_atom_atoms.pos0[1])
        d1 = np.linalg.norm(out[0] - out[1])
        assert d1 == pytest.approx(d0)

    def test_rotation_then_translation(self, three_atom_atoms):
        """z-translation only: z coords increase by 0.5, xy unchanged."""
        p = Pose(quat=np.array([1.0, 0.0, 0.0, 0.0]),
                 trans=np.array([0.0, 0.0, 0.5]))
        out = p.apply(three_atom_atoms.pos0)
        np.testing.assert_allclose(out[:, :2], three_atom_atoms.pos0[:, :2])
        np.testing.assert_allclose(out[:, 2], three_atom_atoms.pos0[:, 2] + 0.5)

    def test_shape_validation(self):
        with pytest.raises(ValueError):
            Pose(quat=np.zeros(3), trans=np.zeros(3))
        with pytest.raises(ValueError):
            Pose(quat=np.zeros(4), trans=np.zeros(2))

    def test_identity_class_method(self):
        p = Pose.identity()
        np.testing.assert_allclose(p.quat, [1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(p.trans, [0.0, 0.0, 0.0])


class TestDiscreteSurface:
    def test_shape_validation(self, simple_discrete_surface):
        assert simple_discrete_surface.m == 2
        assert simple_discrete_surface.pos.shape == (2, 3)

    def test_default_coulomb_factor(self):
        ds = DiscreteSurface(
            pos=np.zeros((1, 3)), q=np.zeros(1),
            c6=np.zeros(1), c12=np.zeros(1),
            lambda_D=0.785, z_min=0.2,
        )
        assert ds.coulomb_factor == COULOMB_FACTOR_KJ_NM_PER_E2

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            DiscreteSurface(
                pos=np.zeros((2, 3)), q=np.zeros(3),
                c6=np.zeros(2), c12=np.zeros(2),
                lambda_D=0.785, z_min=0.2,
            )


class TestContinuumSurface:
    def test_defaults(self, simple_continuum_surface):
        s = simple_continuum_surface
        assert s.rho_s == 30.0
        assert s.c6_surf == 1.0
        assert s.c12_surf == 1.0
        assert s.lambda_D == 0.785
        assert s.z_min == 0.15
        assert s.psi0 == 0.0

    def test_charged(self, charged_continuum_surface):
        assert charged_continuum_surface.psi0 == 5.0
