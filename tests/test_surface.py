"""Tests for the ptmc.surface package: lattice, slab, forcefield, grid_pbc, builder.

Tests use only CPU (forced in conftest.py). Validates:
  - Crystal lattice parameters and unit cell volumes
  - Slab cutting and termination (hydroxylation)
  - Force field parameter assignment against known values
  - PBC grid construction and periodic wrapping
  - End-to-end builder produces consistent grids
"""
import math
import numpy as np
import pytest

from ptmc.surface.lattice import (
    Lattice, cell_from_params, get_crystal,
    alpha_quartz, rutile_tio2, gold_fcc,
)
from ptmc.surface.slab import cut_slab, hydroxylate_quartz, _plane_distance
from ptmc.surface.forcefield import (
    FFParams, CLAYFF, assign_ff, lattice_to_surface,
)
from ptmc.surface.grid_pbc import (
    _replicate_xy, build_grids_pbc, grid_energy_check_pbc,
)
from ptmc.surface.builder import build_surface

from ptmc.model.structures import DiscreteSurface, Atoms, Pose
from ptmc.energy.grid_energy import grid_energy
from ptmc.energy.reference import energy_positions as direct_energy


# ===================================================================
# Lattice tests
# ===================================================================
class TestCellFromParams:
    def test_cubic(self):
        cell = cell_from_params(0.4, 0.4, 0.4, 90, 90, 90)
        assert cell.shape == (3, 3)
        # Diagonal should be a, b, c
        np.testing.assert_allclose(cell[0, 0], 0.4)
        np.testing.assert_allclose(cell[1, 1], 0.4)
        np.testing.assert_allclose(cell[2, 2], 0.4)
        # Volume
        vol = np.linalg.det(cell.T)
        assert vol == pytest.approx(0.4 ** 3)

    def test_triclinic(self):
        cell = cell_from_params(0.5, 0.6, 0.7, 70, 80, 90)
        vol = np.linalg.det(cell.T)
        # General formula: V = abc * sqrt(1 - cos²α - cos²β - cos²γ + 2cosα cosβ cosγ)
        import math
        cos = lambda d: math.cos(math.radians(d))
        expected = (0.5 * 0.6 * 0.7
                    * math.sqrt(1 - cos(70)**2 - cos(80)**2 - cos(90)**2
                                + 2 * cos(70) * cos(80) * cos(90)))
        assert vol == pytest.approx(expected, rel=1e-6)


class TestLattice:
    def test_basic_properties(self):
        cell = cell_from_params(0.4, 0.4, 0.4, 90, 90, 90)
        lat = Lattice(cell=cell.T, species=["A", "B"],
                      frac_pos=np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]))
        assert lat.m == 2
        assert lat.volume == pytest.approx(0.4 ** 3)

    def test_frac_to_cart_roundtrip(self):
        cell = cell_from_params(0.5, 0.6, 0.7, 90, 90, 120)
        lat = Lattice(cell=cell.T, species=["X"],
                      frac_pos=np.array([[0.25, 0.5, 0.75]]))
        cart = lat.frac_to_cart(lat.frac_pos)
        frac_back = lat.cart_to_frac(cart)
        np.testing.assert_allclose(frac_back, lat.frac_pos, atol=1e-12)

    def test_replicate(self):
        lat = alpha_quartz()
        rep = lat.replicate(2, 3, 1)
        assert rep.m == lat.m * 2 * 3 * 1
        assert rep.cell[0, 0] == pytest.approx(lat.cell[0, 0] * 2)


class TestBuiltinCrystals:
    def test_alpha_quartz_volume(self):
        q = alpha_quartz()
        assert q.m == 9
        # Expected volume for a=0.4913, c=0.5405 nm
        a, c = 0.4913, 0.5405
        expected_vol = a**2 * c * math.sin(math.radians(120))  # actually for trigonal
        expected_vol = a**2 * c * np.sqrt(3) / 2  # correct: A = a² sin(60°)
        assert q.volume == pytest.approx(expected_vol, rel=0.02)

    def test_rutile_atoms(self):
        r = rutile_tio2()
        assert r.m == 6

    def test_gold_fcc(self):
        au = gold_fcc()
        assert au.m == 4
        # FCC conventional cell: 4 atoms
        assert au.volume == pytest.approx(0.40782 ** 3, rel=1e-4)

    def test_get_crystal_unknown(self):
        with pytest.raises(ValueError, match="unknown crystal"):
            get_crystal("nonexistent")


# ===================================================================
# Slab tests
# ===================================================================
class TestSlabCutting:
    def test_cut_slab_basic(self):
        q = alpha_quartz()
        slab = cut_slab(q, hkl=(0, 0, 1), n_layers=4, vacuum=3.0)
        assert isinstance(slab, Lattice)
        assert slab.m > 0
        # z dimension should be > vacuum
        assert slab.cell[2, 2] > 3.0

    def test_cut_slab_gold_111(self):
        au = gold_fcc()
        slab = cut_slab(au, hkl=(1, 1, 1), n_layers=4, vacuum=2.0)
        assert slab.m > 0
        assert slab.cell[2, 2] > 2.0

    def test_plane_distance_quartz_001(self):
        q = alpha_quartz()
        d = _plane_distance((0, 0, 1), q.cell)
        # For hexagonal: d_001 = c = 0.5405 nm
        assert d == pytest.approx(0.5405, rel=0.02)


class TestHydroxylation:
    def test_hydroxylates_quartz(self):
        q = alpha_quartz()
        slab = cut_slab(q, hkl=(0, 0, 1), n_layers=4, vacuum=3.0)
        oh = hydroxylate_quartz(slab, oh_z_cutoff=0.35)
        assert "OH" in oh.species or "H_oh" in oh.species

    def test_hydroxyl_increases_atom_count(self):
        q = alpha_quartz()
        slab = cut_slab(q, hkl=(0, 0, 1), n_layers=4, vacuum=3.0)
        oh = hydroxylate_quartz(slab, oh_z_cutoff=0.35)
        assert oh.m >= slab.m


# ===================================================================
# Force field tests
# ===================================================================
class TestFFParams:
    def test_c6_c12_relationship(self):
        """For a given sigma/epsilon, verify C6/C12."""
        au = CLAYFF["Au"]
        c6_expected = 4.0 * au.epsilon * au.sigma ** 6
        c12_expected = 4.0 * au.epsilon * au.sigma ** 12
        assert au.c6 == pytest.approx(c6_expected)
        assert au.c12 == pytest.approx(c12_expected)

    def test_clayff_silica_has_params(self):
        assert "Si" in CLAYFF
        assert "Ob" in CLAYFF
        assert "OH" in CLAYFF
        assert "H_oh" in CLAYFF

    def test_gold_params(self):
        au = CLAYFF["Au"]
        assert au.q == pytest.approx(0.0)
        assert au.sigma == pytest.approx(0.254)
        assert au.epsilon == pytest.approx(2.483)


class TestAssignFF:
    def test_assign_single(self):
        q, c6, c12 = assign_ff(["Au"])
        assert q.shape == (1,)
        assert q[0] == pytest.approx(0.0)

    def test_assign_multi(self):
        q, c6, c12 = assign_ff(["Si", "Ob"])
        assert q.shape == (2,)
        assert q[0] > 0  # Si positive
        assert q[1] < 0  # O negative

    def test_missing_species_raises(self):
        with pytest.raises(ValueError, match="missing FF params"):
            assign_ff(["UnknownElement"])


class TestLatticeToSurface:
    def test_conversion(self):
        au = gold_fcc()
        surface = lattice_to_surface(au, lambda_D=0.785, z_min=0.2)
        assert isinstance(surface, DiscreteSurface)
        assert surface.m == 4
        assert surface.q[0] == pytest.approx(0.0)

    def test_positions_are_cartesian(self):
        q = alpha_quartz()
        surface = lattice_to_surface(q)
        # Fractional coords should have been converted to Cartesian, so
        # coordinates should be on the order of nm not ~0-1
        assert np.any(surface.pos > 0.1)


# ===================================================================
# PBC grid tests
# ===================================================================
class TestReplicateXY:
    def test_no_replication_for_negative_cutoff(self):
        """r_cut <= 0 returns original surface unchanged."""
        q = alpha_quartz()
        surface = lattice_to_surface(q)
        rep = _replicate_xy(surface, np.array([2.0, 2.0]), r_cut=-1)
        assert rep.m == surface.m

    def test_replication_increases_atoms(self):
        q = alpha_quartz()
        surface = lattice_to_surface(q, lambda_D=0.785, z_min=0.2)
        rep = _replicate_xy(surface, np.array([2.0, 2.0]), r_cut=1.0)
        assert rep.m > surface.m


class TestBuildGridsPBC:
    def test_output_type(self):
        au = gold_fcc()
        surface = lattice_to_surface(au, lambda_D=0.785, z_min=0.2)
        grids = build_grids_pbc(surface, cell=(2.0, 2.0), r_cut=1.0,
                                z_range=(0.2, 0.8, 1.0), spacing=0.1)
        from ptmc.energy.grid_build import FieldGrids
        assert isinstance(grids, FieldGrids)
        assert grids.G12.shape == grids.G6.shape == grids.phi.shape

    def test_hard_wall(self):
        au = gold_fcc()
        surface = lattice_to_surface(au, lambda_D=0.785, z_min=0.2)
        grids = build_grids_pbc(surface, cell=(2.0, 2.0), r_cut=1.0,
                                z_range=(0.2, 0.8, 1.0), spacing=0.1)
        atom = Atoms(pos0=np.array([[0.0, 0.0, 0.0]]), q=np.zeros(1),
                     c6=np.ones(1)*1e-3, c12=np.ones(1)*1e-6,
                     names=["X"], resids=np.array([0]),
                     resnames=["BEA"], elements=["C"])
        from ptmc.energy.grid_energy import grid_energy_positions
        # z=0 is below z_min=0.2 -> hard wall -> +inf
        pos = np.array([[0.0, 0.0, 0.0]])
        e = grid_energy_positions(pos, atom, grids)
        assert e == np.inf


class TestGridVsDirectPeriodic:
    """Validate PBC grid energy against direct sum with replicated images."""

    @pytest.fixture
    def small_gold_surface(self):
        """Small gold slab for quick tests."""
        au = gold_fcc()
        slab = cut_slab(au, hkl=(1, 1, 1), n_layers=2, vacuum=2.0)
        return lattice_to_surface(slab, lambda_D=0.785, z_min=0.2)

    def test_grid_matches_direct_far_field(self, small_gold_surface):
        """Grid energy approximates direct sum in the far field."""
        Lx = float(small_gold_surface.pos[:, 0].max()
                   - small_gold_surface.pos[:, 0].min()) + 0.5
        Ly = float(small_gold_surface.pos[:, 1].max()
                   - small_gold_surface.pos[:, 1].min()) + 0.5

        grids = build_grids_pbc(
            small_gold_surface, cell=(Lx, Ly), r_cut=0.8,
            z_range=(0.2, 0.5, 1.5), spacing=0.05, cap_g12=1e6)

        probe = Atoms(pos0=np.array([[0.0, 0.0, 0.0]]),
                      q=np.array([0.3]),
                      c6=np.array([1e-3]),
                      c12=np.array([1e-6]),
                      names=["P"], resids=np.array([0]),
                      resnames=["BEA"], elements=["C"])

        pose = Pose(quat=np.array([1.0, 0.0, 0.0, 0.0]),
                    trans=np.array([Lx/2, Ly/2, 0.5]))

        # Direct sum must use same replicated surface as the grid
        surface_rep = _replicate_xy(small_gold_surface,
                                    np.array([Lx, Ly]), r_cut=0.8)
        direct_e = direct_energy(pose.apply(probe.pos0), probe,
                                 surface_rep)
        grid_e = grid_energy(pose, probe, grids)

        assert grid_e == pytest.approx(direct_e, rel=0.2)


# ===================================================================
# End-to-end builder tests
# ===================================================================
class TestBuilder:
    def test_build_gold(self):
        surface, grids = build_surface(
            "gold_fcc", hkl=(1, 1, 1), n_layers=3, vacuum=3.0,
            hydroxylate=False, spacing=0.1)
        assert surface.m > 0
        assert grids.G12.shape == grids.G6.shape == grids.phi.shape
        # Gold is neutral
        assert surface.q.sum() == pytest.approx(0.0)

    def test_build_quartz(self):
        surface, grids = build_surface(
            "alpha_quartz", hkl=(0, 0, 1), n_layers=4, vacuum=3.0,
            hydroxylate=True, spacing=0.1)
        assert surface.m > 0
        # Quartz should have non-zero charges
        assert np.any(surface.q != 0.0)

    def test_build_titania(self):
        surface, grids = build_surface(
            "rutile_tio2", hkl=(1, 1, 0), n_layers=4, vacuum=3.0,
            hydroxylate=False, spacing=0.1)
        assert surface.m > 0

    def test_builder_invalid_crystal(self):
        with pytest.raises(ValueError, match="unknown crystal"):
            build_surface("invalid_crystal")
