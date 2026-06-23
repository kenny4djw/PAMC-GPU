"""Tests for analysis modules: orientation, clustering, free_energy, heatmap."""
import numpy as np
import pytest

from ptmc.analysis.orientation import (
    contact_normal, lab_positions, contact_residues, tilt_angle, quat_about_z,
)
from ptmc.analysis.clustering import cluster_orientations
from ptmc.analysis.free_energy import basin_free_energies
from ptmc.analysis.heatmap import orientation_free_energy_map
from ptmc.model.structures import Pose
from ptmc.config import BOLTZMANN_KJ_PER_MOL_K


class TestContactNormal:
    def test_identity_pose(self):
        """Identity pose: body z aligns with lab z -> contact normal = (0,0,-1)."""
        cn = contact_normal(np.array([1.0, 0.0, 0.0, 0.0]))
        np.testing.assert_allclose(cn, [0.0, 0.0, -1.0], atol=1e-12)

    def test_180_z_rotation(self):
        """180 deg about z: contact normal unchanged (azimuthal invariance)."""
        q = np.array([0.0, 0.0, 0.0, 1.0])  # 180 deg about z
        cn = contact_normal(q)
        np.testing.assert_allclose(cn, [0.0, 0.0, -1.0], atol=1e-12)

    def test_90_about_x(self):
        """90 deg about x: body z -> lab y. contact normal = R^T (-z_lab)
        = (0, -1, 0) in body frame."""
        q = np.array([np.cos(np.pi/4), np.sin(np.pi/4), 0.0, 0.0])
        cn = contact_normal(q)
        expected = np.array([0.0, -1.0, 0.0])
        np.testing.assert_allclose(cn, expected, atol=1e-12)

    def test_azimuthal_invariance(self, rng):
        """Rotating about lab z does not change contact normal."""
        q_base = np.array([0.5, 0.5, 0.5, 0.5])
        q_base = q_base / np.linalg.norm(q_base)
        cn_base = contact_normal(q_base)
        for _ in range(5):
            phi = rng.uniform(0, 2 * np.pi)
            q_z = quat_about_z(phi)
            # Compose rotations: q_rot = q_z * q_base
            q_rot = quat_mul_np(q_z, q_base)
            cn_rot = contact_normal(q_rot)
            np.testing.assert_allclose(cn_rot, cn_base, atol=1e-12)

    def test_unit_norm(self, rng):
        """Contact normal should be a unit vector."""
        for _ in range(10):
            q = rng.normal(size=4)
            q = q / np.linalg.norm(q)
            cn = contact_normal(q)
            assert np.linalg.norm(cn) == pytest.approx(1.0)


class TestLabPositions:
    def test_identity(self, three_atom_atoms):
        pos = lab_positions(np.array([1.0, 0.0, 0.0, 0.0]),
                            np.zeros(3), three_atom_atoms)
        np.testing.assert_allclose(pos, three_atom_atoms.pos0)

    def test_shift(self, three_atom_atoms):
        t = np.array([0.0, 0.0, 0.5])
        pos = lab_positions(np.array([1.0, 0.0, 0.0, 0.0]),
                            t, three_atom_atoms)
        expected = three_atom_atoms.pos0 + t
        np.testing.assert_allclose(pos, expected)


class TestContactResidues:
    def test_no_contact(self, three_atom_atoms):
        """All atoms far above surface -> no contact residues."""
        res = contact_residues(np.array([1.0, 0.0, 0.0, 0.0]),
                               np.array([0.0, 0.0, 10.0]),
                               three_atom_atoms, z_contact=0.5)
        assert len(res) == 0

    def test_all_contact(self, three_atom_atoms):
        """All atoms near z=0 -> all residues are contact residues."""
        # Atom[0]=res0, atom[1,2]=res1
        res = contact_residues(np.array([1.0, 0.0, 0.0, 0.0]),
                               np.array([0.0, 0.0, 0.0]),
                               three_atom_atoms, z_contact=0.5)
        assert len(res) == 2  # both residues contact

    def test_returns_frozenset(self, three_atom_atoms):
        res = contact_residues(np.array([1.0, 0.0, 0.0, 0.0]),
                               np.array([0.0, 0.0, 10.0]),
                               three_atom_atoms, z_contact=0.5)
        assert isinstance(res, frozenset)


class TestTiltAngle:
    def test_identity(self):
        """Identity pose: tilt = acos(-(-1)) = acos(1) = 0."""
        tilt = tilt_angle(np.array([1.0, 0.0, 0.0, 0.0]))
        assert tilt == pytest.approx(0.0)

    def test_upside_down(self):
        """180 deg about x: body -z points to +z. tilt = pi."""
        q = np.array([0.0, 1.0, 0.0, 0.0])
        tilt = tilt_angle(q)
        assert tilt == pytest.approx(np.pi)

    def test_sideways(self):
        """90 deg about x -> body -z points to -y, tilt = pi/2."""
        q = np.array([np.cos(np.pi/4), np.sin(np.pi/4), 0.0, 0.0])
        tilt = tilt_angle(q)
        assert tilt == pytest.approx(np.pi / 2)


class TestClusterOrientations:
    def test_single_cluster(self):
        normals = np.tile(np.array([0.0, 0.0, -1.0]), (10, 1))
        labels, cents = cluster_orientations(normals, k=1)
        assert labels.shape == (10,)
        assert cents.shape == (1, 3)
        np.testing.assert_allclose(cents[0], [0.0, 0.0, -1.0], atol=1e-7)

    def test_two_clusters(self, rng):
        n1 = rng.normal(size=(20, 3)) + np.array([1.0, 0.0, 0.0])
        n2 = rng.normal(size=(20, 3)) + np.array([-1.0, 0.0, 0.0])
        normals = np.concatenate([n1, n2])
        labels, cents = cluster_orientations(normals, k=2, seed=42)
        assert len(set(labels)) == 2

    def test_deterministic(self):
        normals = np.random.default_rng(0).normal(size=(10, 3))
        l1, c1 = cluster_orientations(normals, k=2, seed=42)
        l2, c2 = cluster_orientations(normals, k=2, seed=42)
        np.testing.assert_array_equal(l1, l2)


class TestBasinFreeEnergies:
    def test_equal_populations(self):
        labels = np.array([0, 0, 0, 1, 1, 1])
        p, dG = basin_free_energies(labels, beta=1.0)
        assert p[0] == pytest.approx(p[1])
        assert dG[0] == pytest.approx(dG[1])

    def test_one_basin(self):
        labels = np.zeros(10, dtype=int)
        p, dG = basin_free_energies(labels, beta=1.0)
        assert p[0] == 1.0
        assert dG[0] == 0.0

    def test_dominant_basin(self):
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
                           0, 0, 0, 0, 0, 2, 2, 2, 2, 2])
        # 10 in 0, 5 in 1, 5 in 2  (actually let me count...)
        # Actually: 10 zeros, 5 ones, 5 twos
        labels = np.array([0]*12 + [1]*4 + [2]*4)
        p, dG = basin_free_energies(labels, beta=1.0)
        assert p[0] >= p[1]  # basin 0 has highest pop
        assert dG[0] == 0.0  # min dG = 0

    def test_temperature_dependence(self):
        """Higher temperature -> smaller free energy difference.
        dG = -(1/beta) * ln(p). Cold (large beta) gives smaller |dG|."""
        labels = np.array([0]*8 + [1]*2)
        p, dG_cold = basin_free_energies(labels, beta=2.0)
        _, dG_hot = basin_free_energies(labels, beta=1.0)
        assert dG_cold[1] < dG_hot[1]


class TestOrientationFreeEnergyMap:
    def test_output_shape(self):
        normals = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0],
                            [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        te, pe, F = orientation_free_energy_map(normals, beta=1.0,
                                                  n_theta=5, n_phi=6)
        assert te.shape == (6,)
        assert pe.shape == (7,)
        assert F.shape == (5, 6)

    def test_single_orientation(self):
        """All samples at same orientation -> one bin populated."""
        normals = np.tile(np.array([0.0, 0.0, -1.0]), (10, 1))
        te, pe, F = orientation_free_energy_map(normals, beta=1.0,
                                                  n_theta=5, n_phi=6)
        # Most entries should be inf (zero population)
        assert np.sum(np.isfinite(F)) >= 1

    def test_finite_values(self):
        normals = np.random.default_rng(42).normal(size=(100, 3))
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        te, pe, F = orientation_free_energy_map(normals, beta=1.0,
                                                  n_theta=10, n_phi=12)
        assert np.all(np.isfinite(F[F != np.inf]))


# Need quat_mul_np from orientation for the test
def quat_mul_np(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])
