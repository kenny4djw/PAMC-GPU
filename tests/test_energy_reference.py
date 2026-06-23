"""Tests for ptmc.energy.reference: direct-sum energy functions."""
import numpy as np
import pytest

from ptmc.energy.reference import (
    lj_pair, screened_coulomb_pair, combine_geometric,
    energy_positions, energy_direct,
)
from ptmc.model.structures import Pose, Atoms


class TestLJPair:
    def test_lj_basic(self):
        """At very large r, LJ is near zero."""
        e = lj_pair(100.0, 1e-3, 1e-6)
        assert e == pytest.approx(0.0, abs=1e-14)

    def test_lj_attractive_region(self):
        """At moderate r, LJ is attractive (negative)."""
        e = lj_pair(0.4, 1e-3, 1e-6)
        assert e < 0

    def test_lj_repulsive_region(self):
        """At small r, LJ is repulsive (positive)."""
        e = lj_pair(0.2, 1e-3, 1e-6)
        assert e > 0

    def test_lj_broadcast(self, rng):
        c6 = rng.random(5) * 1e-2
        c12 = rng.random(5) * 1e-5
        r = rng.random(5) + 0.3
        e = lj_pair(r, c6, c12)
        assert e.shape == (5,)
        expected = c12 / r**12 - c6 / r**6
        np.testing.assert_allclose(e, expected)


class TestScreenedCoulomb:
    def test_repulsive(self):
        """Same-sign charges -> positive energy."""
        e = screened_coulomb_pair(0.5, 0.5, 0.5, 0.785, 138.935458)
        assert e > 0

    def test_attractive(self):
        """Opposite-sign charges -> negative energy."""
        e = screened_coulomb_pair(0.5, 0.5, -0.5, 0.785, 138.935458)
        assert e < 0

    def test_debye_screening(self):
        """Shorter lambda_D -> more screening -> smaller |e| at fixed r."""
        e_long = screened_coulomb_pair(0.5, 1.0, -1.0, 1.0, 138.935458)
        e_short = screened_coulomb_pair(0.5, 1.0, -1.0, 0.3, 138.935458)
        assert abs(e_short) < abs(e_long)

    def test_vacuum_limit(self):
        """Very large lambda_D approximates unscreened Coulomb."""
        e = screened_coulomb_pair(0.5, 1.0, -1.0, 1e6, 138.935458)
        expected = 138.935458 * (-1.0) / 0.5
        assert e == pytest.approx(expected, rel=1e-3)

    def test_zero_charge(self):
        assert screened_coulomb_pair(0.5, 0.0, 1.0, 0.785, 138.935458) == 0.0

    def test_broadcast(self, rng):
        r = rng.random(4) + 0.3
        qi = rng.random(4) * 2 - 1
        qj = rng.random(4) * 2 - 1
        e = screened_coulomb_pair(r, qi, qj, 0.785, 138.935458)
        assert e.shape == (4,)


class TestCombineGeometric:
    def test_basic(self):
        assert combine_geometric(4.0, 9.0) == pytest.approx(6.0)

    def test_commutative(self):
        a, b = 2.5, 3.7
        assert combine_geometric(a, b) == combine_geometric(b, a)

    def test_zero(self):
        assert combine_geometric(0.0, 5.0) == 0.0

    def test_broadcast(self):
        ci = np.array([1.0, 4.0, 9.0])
        cj = np.array([1.0, 4.0, 9.0])
        result = combine_geometric(ci[:, None], cj[None, :])
        assert result.shape == (3, 3)


class TestEnergyDiscreteSurface:
    def test_all_above_surface_returns_finite(self, three_atom_atoms,
                                              simple_discrete_surface):
        pos = three_atom_atoms.pos0.copy()
        pos[:, 2] += 0.5  # shift all above surface
        e = energy_positions(pos, three_atom_atoms, simple_discrete_surface)
        assert np.isfinite(e)

    def test_hard_wall(self, three_atom_atoms, simple_discrete_surface):
        pos = three_atom_atoms.pos0.copy()
        pos[:, 2] = 0.0  # below z_min
        e = energy_positions(pos, three_atom_atoms, simple_discrete_surface)
        assert e == np.inf

    def test_direct_via_pose(self, three_atom_atoms, simple_discrete_surface):
        """energy_direct should match energy_positions for same pose."""
        pose = Pose(quat=np.array([1.0, 0.0, 0.0, 0.0]),
                    trans=np.array([0.0, 0.0, 0.5]))
        e1 = energy_direct(pose, three_atom_atoms, simple_discrete_surface)
        pos = pose.apply(three_atom_atoms.pos0)
        e2 = energy_positions(pos, three_atom_atoms, simple_discrete_surface)
        assert e1 == pytest.approx(e2)

    def test_energy_reproducibility(self, three_atom_atoms,
                                    simple_discrete_surface, rng):
        """Same pose -> same energy."""
        pos = three_atom_atoms.pos0 + np.array([0.0, 0.0, 0.5])
        e1 = energy_positions(pos, three_atom_atoms, simple_discrete_surface)
        e2 = energy_positions(pos, three_atom_atoms, simple_discrete_surface)
        assert e1 == pytest.approx(e2)


class TestEnergyContinuumSurface:
    def test_all_above_surface_returns_finite(self, three_atom_atoms,
                                              simple_continuum_surface):
        pos = three_atom_atoms.pos0.copy()
        pos[:, 2] += 0.5
        e = energy_positions(pos, three_atom_atoms, simple_continuum_surface)
        assert np.isfinite(e)

    def test_hard_wall(self, three_atom_atoms, simple_continuum_surface):
        pos = three_atom_atoms.pos0.copy()
        pos[:, 2] = 0.0
        e = energy_positions(pos, three_atom_atoms, simple_continuum_surface)
        assert e == np.inf

    def test_charged_surface_energy(self, simple_continuum_surface,
                                    charged_continuum_surface):
        """A single charged atom should feel different energy on charged vs neutral surface."""
        atom = Atoms(pos0=np.array([[0.0, 0.0, 0.0]]),
                     q=np.array([0.5]),
                     c6=np.array([1e-3]),
                     c12=np.array([1e-6]),
                     names=["Q"], resids=np.array([0]),
                     resnames=["BEA"], elements=["C"])
        pos = np.array([[0.0, 0.0, 0.5]])
        e_neutral = energy_positions(pos, atom, simple_continuum_surface)
        e_charged = energy_positions(pos, atom, charged_continuum_surface)
        # Neutral surface: psi0=0 -> no electrostatic term
        # Charged surface: psi0=5.0 -> added electrostatic contribution
        assert e_charged != pytest.approx(e_neutral)
        # The electrostatic contribution = q * psi0 * exp(-z/lambda_D)
        expected_elec = 0.5 * 5.0 * np.exp(-0.5 / 0.785)
        assert (e_charged - e_neutral) == pytest.approx(expected_elec)

    def test_vdw_attractive_in_mid_range(self, single_atom,
                                         simple_continuum_surface):
        """Steele 9-3: z=0.6 should be more attractive than z=1.5."""
        pos_far = np.array([[0.0, 0.0, 1.5]])
        pos_near = np.array([[0.0, 0.0, 0.6]])
        e_far = energy_positions(pos_far, single_atom, simple_continuum_surface)
        e_near = energy_positions(pos_near, single_atom, simple_continuum_surface)
        assert e_near < e_far

    def test_unknown_surface_raises(self, three_atom_atoms):
        class FakeSurface:
            z_min = 0.0
        pos = three_atom_atoms.pos0 + np.array([0.0, 0.0, 0.5])
        with pytest.raises(TypeError, match="unknown surface"):
            energy_positions(pos, three_atom_atoms, FakeSurface())
