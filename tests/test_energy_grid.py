"""Tests for ptmc.energy.grid_build and grid_energy."""
import numpy as np
import pytest

from ptmc.energy.grid_build import build_grids, FieldGrids
from ptmc.energy.grid_energy import _trilinear, grid_energy_positions, grid_energy
from ptmc.energy.reference import energy_positions as direct_energy
from ptmc.model.structures import Pose, Atoms


class TestFieldGrids:
    def test_shape(self, simple_discrete_surface):
        grids = build_grids(simple_discrete_surface,
                            x_range=(-0.5, 0.5), y_range=(-0.5, 0.5),
                            z_range=(0.2, 1.0), spacing=0.1)
        assert grids.G12.shape == grids.G6.shape == grids.phi.shape
        assert len(grids.shape) == 3

    def test_origin_and_spacing(self, simple_discrete_surface):
        grids = build_grids(simple_discrete_surface,
                            x_range=(0.0, 0.2), y_range=(0.0, 0.2),
                            z_range=(0.2, 0.5), spacing=0.1)
        assert grids.origin[0] == pytest.approx(0.0)
        assert grids.spacing[0] == pytest.approx(0.1)

    def test_z_min_clamping(self, simple_discrete_surface):
        """z_range[0] below z_min gets clamped to z_min."""
        grids = build_grids(simple_discrete_surface,
                            x_range=(0.0, 0.1), y_range=(0.0, 0.1),
                            z_range=(0.0, 0.5), spacing=0.1)
        expected_z_start = simple_discrete_surface.z_min  # 0.2
        assert grids.origin[2] == pytest.approx(expected_z_start)

    def test_upper_method(self, simple_discrete_surface):
        grids = build_grids(simple_discrete_surface,
                            x_range=(0.0, 0.3), y_range=(0.0, 0.3),
                            z_range=(0.2, 0.6), spacing=0.1)
        up = grids.upper()
        assert up[0] == pytest.approx(0.3)
        assert up[2] == pytest.approx(0.6)

    def test_cap_g12(self, simple_discrete_surface):
        """Capping G12 prevents overflow near surface atoms."""
        grids_capped = build_grids(simple_discrete_surface,
                                   x_range=(-0.5, 0.5), y_range=(-0.5, 0.5),
                                   z_range=(0.2, 1.0), spacing=0.1,
                                   cap_g12=100.0)
        assert grids_capped.G12.max() <= 100.0


class TestTrilinear:
    def test_on_grid_point(self, simple_discrete_surface):
        """Interpolating exactly at a grid point returns the field value."""
        grids = build_grids(simple_discrete_surface,
                            x_range=(0.0, 0.2), y_range=(0.0, 0.2),
                            z_range=(0.2, 0.5), spacing=0.1)
        # First grid point
        pos = grids.origin.copy()
        val = _trilinear(grids.G12, pos[None, :], grids.origin, grids.spacing)
        expected = grids.G12[0, 0, 0]
        assert val[0] == pytest.approx(expected)

    def test_midpoint_interpolation(self):
        """On a uniform field, linear interpolation should give midpoint value."""
        field = np.ones((3, 3, 3), dtype=np.float64)
        origin = np.array([0.0, 0.0, 0.0])
        spacing = np.array([0.1, 0.1, 0.1])
        pos = np.array([[0.05, 0.05, 0.05]])
        val = _trilinear(field, pos, origin, spacing)
        assert val[0] == pytest.approx(1.0)

    def test_multiple_points(self, rng):
        """Multiple positions should return correct number of values."""
        field = rng.random((5, 5, 5)).astype(np.float64)
        origin = np.zeros(3)
        spacing = np.array([0.1, 0.1, 0.1])
        pos = rng.random((10, 3)).astype(np.float64) * 0.4
        val = _trilinear(field, pos, origin, spacing)
        assert val.shape == (10,)


class TestGridVsDirect:
    """Grid energy should converge to direct-sum energy with fine spacing."""

    @pytest.fixture
    def probe_atom(self, simple_discrete_surface):
        """A single probe atom near the surface."""
        return Atoms(pos0=np.array([[0.05, 0.05, 0.0]]),
                     q=np.array([0.0]),
                     c6=np.array([1e-3]),
                     c12=np.array([1e-6]),
                     names=["P"], resids=np.array([0]),
                     resnames=["BEA"], elements=["C"])

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_grid_converges_with_spacing(self, simple_discrete_surface, probe_atom):
        """Grid energy at fine spacing approximates direct energy."""
        pose = Pose(quat=np.array([1.0, 0.0, 0.0, 0.0]),
                    trans=np.array([0.05, 0.05, 0.3]))

        direct_e = direct_energy(pose.apply(probe_atom.pos0), probe_atom,
                                 simple_discrete_surface)

        grids = build_grids(simple_discrete_surface,
                            x_range=(0.0, 0.2), y_range=(0.0, 0.2),
                            z_range=(0.2, 0.6), spacing=0.02)
        grid_e = grid_energy(pose, probe_atom, grids)

        assert grid_e == pytest.approx(direct_e, rel=0.1)

    def test_grid_hard_wall(self, simple_discrete_surface, probe_atom):
        """Grid energy should return +inf when atom below z_min."""
        grids = build_grids(simple_discrete_surface,
                            x_range=(0.0, 0.2), y_range=(0.0, 0.2),
                            z_range=(0.2, 0.6), spacing=0.05)
        pos = np.array([[0.05, 0.05, 0.1]])  # below z_min=0.2
        e = grid_energy_positions(pos, probe_atom, grids)
        assert e == np.inf

    def test_grid_matches_direct_atteps(self, simple_discrete_surface):
        """With enough steps (fine grid + mid-placement), G12/G6/phi sum matches
        direct in the far field (not too close to surface atoms)."""
        far_atom = Atoms(pos0=np.array([[0.1, 0.1, 0.0]]),
                         q=np.array([0.3]),
                         c6=np.array([1e-3]),
                         c12=np.array([1e-6]),
                         names=["F"], resids=np.array([0]),
                         resnames=["BEA"], elements=["C"])
        grids = build_grids(simple_discrete_surface,
                            x_range=(0.0, 0.3), y_range=(0.0, 0.3),
                            z_range=(0.2, 1.0), spacing=0.04)

        pose = Pose(quat=np.array([1.0, 0.0, 0.0, 0.0]),
                    trans=np.array([0.1, 0.1, 0.5]))
        direct_e = direct_energy(pose.apply(far_atom.pos0), far_atom,
                                 simple_discrete_surface)
        grid_e = grid_energy(pose, far_atom, grids)
        assert grid_e == pytest.approx(direct_e, rel=0.05)

    def test_grid_near_wall(self, simple_discrete_surface):
        """Check that grid correctly gives finite vs infinite near z_min."""
        atom = Atoms(pos0=np.array([[0.0, 0.0, 0.0]]),
                     q=np.zeros(1), c6=np.ones(1) * 1e-3,
                     c12=np.ones(1) * 1e-6,
                     names=["X"], resids=np.array([0]),
                     resnames=["BEA"], elements=["C"])
        grids = build_grids(simple_discrete_surface,
                            x_range=(0.0, 0.2), y_range=(0.0, 0.2),
                            z_range=(0.15, 0.5), spacing=0.05)
        # Just at z_min: should be finite (but possibly high)
        pos = np.array([[0.0, 0.0, 0.151]])
        e = grid_energy_positions(pos, atom, grids)
        assert np.isfinite(e)

        # Below z_min: should be inf
        pos2 = np.array([[0.0, 0.0, 0.14]])
        e2 = grid_energy_positions(pos2, atom, grids)
        assert e2 == np.inf
