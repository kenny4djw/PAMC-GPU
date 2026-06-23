"""Integration test: lysozyme (Amber99sb-ildn) on rutile TiO2(110) (CLAYFF).

Validates the complete workflow:
  1. Parse real lysozyme PDB + GROMACS topology into Atoms
  2. Build a rutile TiO2(110) periodic surface slab
  3. Assign CLAYFF parameters and build PBC energy grids
  4. Compute grid-interpolated energies at multiple poses
  5. Verify energy-vs-height and orientation dependence
  6. Short MC trajectory via continuum model (end-to-end integration)

Requires GROMACS Amber99sb-ildn FF (symlinked in data/).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from ptmc.io.parse_pdb import parse_pdb
from ptmc.io.parse_topology import parse_topology, build_atoms
from ptmc.model.structures import Atoms, Pose, DiscreteSurface
from ptmc.energy.grid_energy import grid_energy
from ptmc.energy.reference import energy_positions as direct_energy

from ptmc.surface.lattice import rutile_tio2, Lattice
from ptmc.surface.slab import cut_slab
from ptmc.surface.forcefield import CLAYFF, lattice_to_surface
from ptmc.surface.grid_pbc import _replicate_xy, build_grids_pbc

DATA = Path(__file__).resolve().parent.parent / "data"
PDB_PATH = str(DATA / "2LYZ_prot.pdb")
TOP_PATH = str(DATA / "2LYZ_prot.top")

# Rutile surface parameters
RUTILE_HKL = (1, 1, 0)
N_LAYERS = 4
VACUUM = 3.0
LAMBDA_D = 0.785     # Debye length (nm), ~100 mM monovalent salt
Z_MIN = 0.2          # hard-wall distance (nm)
GRID_SPACING = 0.2   # nm (coarse for test speed)
GRID_R_CUT = 1.4     # neighbour cutoff for PBC grid (nm)

# Target simulation cell size (nm) — large enough for lysozyme (~3-4 nm dia)
CELL_TARGET = 4.5


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture(scope="session")
def lysozyme_raw() -> Atoms:
    """Parse lysozyme PDB + topology into Atoms model (original PDB frame)."""
    pdb = parse_pdb(PDB_PATH)
    topo = parse_topology(TOP_PATH)
    return build_atoms(pdb, topo)


@pytest.fixture(scope="session")
def lysozyme(lysozyme_raw) -> Atoms:
    """Lysozyme with coordinates centered at origin."""
    center = lysozyme_raw.pos0.mean(axis=0)
    return Atoms(
        pos0=lysozyme_raw.pos0 - center,
        q=lysozyme_raw.q,
        c6=lysozyme_raw.c6,
        c12=lysozyme_raw.c12,
        names=lysozyme_raw.names,
        resids=lysozyme_raw.resids,
        resnames=lysozyme_raw.resnames,
        elements=lysozyme_raw.elements,
    )


@pytest.fixture(scope="session")
def rutile_slab() -> Lattice:
    """Large rutile TiO2(110) slab for protein adsorption simulations."""
    bulk = rutile_tio2()
    # Cut slab from unit cell, then replicate in xy for protein-sized cell
    slab = cut_slab(bulk, hkl=RUTILE_HKL, n_layers=N_LAYERS, vacuum=VACUUM)
    nx = max(1, int(np.ceil(CELL_TARGET / slab.cell[0, 0])))
    ny = max(1, int(np.ceil(CELL_TARGET / slab.cell[1, 1])))
    return slab.replicate(nx, ny, 1)


@pytest.fixture(scope="session")
def rutile_surface(rutile_slab) -> DiscreteSurface:
    """Rutile slab with CLAYFF parameters assigned."""
    return lattice_to_surface(rutile_slab, ff=CLAYFF,
                               lambda_D=LAMBDA_D, z_min=Z_MIN)


@pytest.fixture(scope="session")
def rutile_grids(rutile_slab, rutile_surface):
    """PBC energy grids for rutile surface."""
    Lx = rutile_slab.cell[0, 0]
    Ly = rutile_slab.cell[1, 1]
    # Grid covers z_min to z_min + 6.0 nm (enough for lysozyme)
    grids = build_grids_pbc(
        rutile_surface, cell=(Lx, Ly), r_cut=GRID_R_CUT,
        z_range=(Z_MIN, Z_MIN + 0.5, 5.5),
        spacing=GRID_SPACING, cap_g12=1e6)
    return grids, (Lx, Ly)


@pytest.fixture(scope="session")
def small_surface() -> DiscreteSurface:
    """Small 1-layer rutile surface for direct-sum validation (avoids OOM)."""
    bulk = rutile_tio2()
    slab = cut_slab(bulk, hkl=RUTILE_HKL, n_layers=1, vacuum=2.0)
    return lattice_to_surface(slab, ff=CLAYFF, lambda_D=LAMBDA_D, z_min=Z_MIN)


# ===================================================================
# Tests
# ===================================================================

class TestLysozymeParsing:
    """Lysozyme input parsing."""

    def test_parse_pdb(self):
        pdb = parse_pdb(PDB_PATH)
        assert pdb.n == 1960
        assert np.all(pdb.pos.max(0) - pdb.pos.min(0) < 5.0)

    def test_parse_topology(self):
        topo = parse_topology(TOP_PATH)
        assert len(topo.q) == 1960
        assert topo.comb_rule == 2  # Amber99sb-ildn: geometric combination

    def test_build_atoms(self, lysozyme):
        """Lysozyme has expected net charge at neutral pH."""
        assert lysozyme.n == 1960
        assert abs(lysozyme.net_charge - 8.0) < 0.5
        assert lysozyme.c6.min() >= 0.0
        assert lysozyme.c12.min() >= 0.0

    def test_centered(self, lysozyme):
        """Centered coordinates have near-zero mean."""
        assert abs(lysozyme.pos0.mean()) < 1e-10


class TestRutileSurface:
    """Rutile TiO2(110) surface building."""

    def test_slab_dimensions(self, rutile_slab):
        """Slab is large enough for lysozyme."""
        Lx, Ly = rutile_slab.cell[0, 0], rutile_slab.cell[1, 1]
        assert Lx > 4.0, f"Lx={Lx} too small for lysozyme"
        assert Ly > 4.0, f"Ly={Ly} too small for lysozyme"
        assert rutile_slab.cell[2, 2] > VACUUM

    def test_surface_has_charged_atoms(self, rutile_surface):
        """Rutile surface has both positive (Ti) and negative (O) charges."""
        assert np.any(rutile_surface.q > 0)
        assert np.any(rutile_surface.q < 0)

    def test_grid_coverage(self, rutile_grids):
        """Grid dimensions accommodate lysozyme."""
        grids, (Lx, Ly) = rutile_grids
        nx, ny, nz = grids.shape
        assert nx > 10 and ny > 10 and nz > 10
        # Grid xy covers the simulation cell
        upper = grids.origin + (np.array(grids.shape) - 1) * grids.spacing
        assert upper[0] >= Lx - GRID_SPACING
        assert upper[1] >= Ly - GRID_SPACING
        # Grid z covers enough height for the protein
        assert upper[2] > 4.0


class TestGridEnergy:
    """Grid energy computation for lysozyme on rutile."""

    def test_energy_finite(self, lysozyme, rutile_grids):
        """Grid energy is finite for a reasonable placement."""
        grids, (Lx, Ly) = rutile_grids
        pose = Pose(quat=np.array([1.0, 0.0, 0.0, 0.0]),
                    trans=np.array([Lx / 2, Ly / 2, 2.5]))
        e = grid_energy(pose, lysozyme, grids)
        assert np.isfinite(e), "energy should be finite at z=2.5"

    def test_hard_wall(self, lysozyme, rutile_grids):
        """Protein below z_min gets infinite energy."""
        grids, (Lx, Ly) = rutile_grids
        pose = Pose(quat=np.array([1.0, 0.0, 0.0, 0.0]),
                    trans=np.array([Lx / 2, Ly / 2, -1.0]))
        e = grid_energy(pose, lysozyme, grids)
        assert e == np.inf

    def test_energy_above_versus_below(self, lysozyme, rutile_grids):
        """Lower protein sees stronger interaction (repulsive or attractive)."""
        grids, (Lx, Ly) = rutile_grids
        p = Pose(quat=np.array([1.0, 0.0, 0.0, 0.0]),
                 trans=np.array([Lx / 2, Ly / 2, 2.5]))
        e_low = grid_energy(p, lysozyme, grids)

        p.trans = np.array([Lx / 2, Ly / 2, 3.5])
        e_high = grid_energy(p, lysozyme, grids)

        # Different heights → different energies (non-trivial interaction)
        assert e_low != pytest.approx(e_high, abs=0.1), \
            f"low z=2.5 e={e_low:.1f}, high z=3.5 e={e_high:.1f}"

    def test_grid_matches_direct_small(self, lysozyme, small_surface):
        """Grid energy on a small surface matches direct sum."""
        from ptmc.surface.grid_pbc import _replicate_xy

        # Build grid on small surface
        Lx = float(small_surface.pos[:, 0].max()
                   - small_surface.pos[:, 0].min()) + 0.5
        Ly = float(small_surface.pos[:, 1].max()
                   - small_surface.pos[:, 1].min()) + 0.5
        if Lx < 1.0:
            Lx = 1.0
        if Ly < 1.0:
            Ly = 1.0
        grids = build_grids_pbc(
            small_surface, cell=(Lx, Ly), r_cut=1.0,
            z_range=(Z_MIN, Z_MIN + 0.5, 1.5),
            spacing=GRID_SPACING, cap_g12=1e6)

        # Replicate surface for direct sum
        surface_rep = _replicate_xy(small_surface, np.array([Lx, Ly]),
                                     r_cut=1.0)

        # Use only first 10 lysozyme atoms for speed
        small = Atoms(
            pos0=lysozyme.pos0[:10],
            q=lysozyme.q[:10],
            c6=lysozyme.c6[:10],
            c12=lysozyme.c12[:10],
            names=lysozyme.names[:10],
            resids=lysozyme.resids[:10],
            resnames=lysozyme.resnames[:10],
            elements=lysozyme.elements[:10],
        )

        pose = Pose(quat=np.array([1.0, 0.0, 0.0, 0.0]),
                    trans=np.array([Lx / 2, Ly / 2, 0.6]))

        pos = pose.apply(small.pos0)
        grid_e = grid_energy(pose, small, grids)
        direct_e = direct_energy(pos, small, surface_rep)

        assert grid_e == pytest.approx(direct_e, rel=0.20), \
            f"grid={grid_e:.4f} direct={direct_e:.4f}"


class TestOrientationDependence:
    """Energy varies with protein orientation."""

    def _rotated_poses(self, Lx, Ly):
        """Poses with different azimuthal rotations."""
        from ptmc.mc.moves import axisangle_to_quat
        poses = []
        for angle_deg in [0, 90, 180]:
            angle_rad = math.radians(angle_deg)
            q = axisangle_to_quat(np.array([0.0, 0.0, angle_rad]))
            poses.append(
                Pose(quat=np.array(q), trans=np.array([Lx / 2, Ly / 2, 2.5]))
            )
        return poses

    def test_orientation_energies_differ(self, lysozyme, rutile_grids):
        """Different orientations yield measurably different energies."""
        grids, (Lx, Ly) = rutile_grids
        energies = []
        for pose in self._rotated_poses(Lx, Ly):
            e = grid_energy(pose, lysozyme, grids)
            assert np.isfinite(e), f"non-finite energy {e} at orientation"
            energies.append(e)

        spread = np.max(energies) - np.min(energies)
        assert spread > 1.0, \
            f"orientation energy spread {spread:.1f} kJ/mol too small"

    def test_inversion_symmetry(self, lysozyme, rutile_grids):
        """180° flip gives different energy (surface breaks z-symmetry)."""
        grids, (Lx, Ly) = rutile_grids
        from ptmc.mc.moves import axisangle_to_quat

        q_up = np.array([1.0, 0.0, 0.0, 0.0])
        q_down = axisangle_to_quat(np.array([math.pi, 0.0, 0.0]))
        base_trans = np.array([Lx / 2, Ly / 2, 2.5])

        e_up = grid_energy(
            Pose(quat=q_up, trans=base_trans), lysozyme, grids
        )
        e_down = grid_energy(
            Pose(quat=np.array(q_down), trans=base_trans.copy()), lysozyme, grids
        )

        # Protein dipole + surface field → different energies
        assert abs(e_up - e_down) > 0.5, \
            f"inversion energies too close: up={e_up:.1f} down={e_down:.1f}"


# ===================================================================
# Continuum-model MC (end-to-end integration using existing JAX path)
# ===================================================================
@pytest.mark.slow
class TestMonteCarlo:
    """Short MC trajectory with continuum-surface model.

    Uses the JAX-based continuum path (Steele 9-3 + PB) rather than the
    discrete grid path since the latter is numpy-based and not JIT-compatible.
    """

    @pytest.fixture(scope="class")
    def mc_result(self, lysozyme):
        import jax
        from ptmc.config import beta
        from ptmc.sampler.highthroughput import (
            build_batch, _chain_keys, initial_state, _ht_run_ps,
        )

        # Rutile continuum parameters from CLAYFF
        # TiO2 rutile(110): ~10.4 atoms/nm^2
        rho_s = 10.4
        c6_surf = np.sqrt(CLAYFF["Ti"].c6 * CLAYFF["Ob_ti"].c6)
        c12_surf = np.sqrt(CLAYFF["Ti"].c12 * CLAYFF["Ob_ti"].c12)

        system = {
            "system_id": 0,
            "pos0": lysozyme.pos0,
            "q": lysozyme.q,
            "c6": lysozyme.c6,
            "c12": lysozyme.c12,
            "c6_surf": c6_surf,
            "c12_surf": c12_surf,
            "rho_s": rho_s,
            "psi0": -0.1,          # modest negative surface potential (V)
            "lambda_D": LAMBDA_D,
            "z_min": Z_MIN,
            "beta": beta(300.0),
            "init_z": Z_MIN + 2.5,   # protein radius ~2.25 nm
        }

        n_chains = 4
        n_steps = 500
        sigma_rot = 0.5
        sigma_trans = 0.1
        axis_mask = np.array([True, True, True])

        S = 1
        b = build_batch([system], n_chains)
        keys_sc = _chain_keys(42, [0], n_chains)  # (S, C, 2)
        z0_sc = jax.numpy.broadcast_to(b["z0_ps"][:, None], (S, n_chains))
        quat0_sc, trans0_sc = initial_state(keys_sc, z0_sc)

        qf, tf, ef, ar = _ht_run_ps(
            keys_sc, quat0_sc, trans0_sc,
            b["beta_ps"], b["pos0_ps"], b["q_ps"],
            b["c6p_ps"], b["c12p_ps"], b["psi0_ps"],
            b["cA_ps"], b["cB_ps"], b["lamD_ps"], b["z_min_ps"],
            sigma_rot, sigma_trans, jax.numpy.asarray(axis_mask),
            0, n_steps)

        return dict(qf=np.asarray(qf), tf=np.asarray(tf),
                    ef=np.asarray(ef), ar=np.asarray(ar))

    def test_mc_acceptance_rate(self, mc_result):
        """MC acceptance rate is in a reasonable range."""
        ar = mc_result["ar"]
        assert 0.1 < ar.mean() < 0.9

    def test_mc_energy_finite(self, mc_result):
        """All MC energies are finite."""
        assert np.all(np.isfinite(mc_result["ef"]))

    def test_mc_energy_reasonable(self, mc_result):
        """Mean energy is in a physically plausible range."""
        mean_e = mc_result["ef"].mean()
        assert -10000 < mean_e < 10000
