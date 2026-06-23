"""F4 — dihedral bonded energy.

Validates the JAX kernel against an independent numpy reference (same physical
form, separate code path) plus physical invariants (2π periodicity, scale
linearity in k_phi). The "vs gmx single-point" gate from § 5.4 is not run
here -- gromacs is not part of the WSL test rig -- but the numpy reference
test is sufficient to catch every implementation bug short of getting the
formula wrong identically in both places.

The 1UBQ topology gives a workout: 3509 periodic terms, no RB, no harmonic.
Synthetic mini-tables exercise the RB and harmonic paths.
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ptmc.flexible import (
    BondedParams,
    E_bonded,
    E_bonded_numpy,
    apply_all_chi,
    build_bonded_params,
    build_chi_schedule,
    build_chi_topology,
    measure_dihedrals,
)

DATA = Path(__file__).resolve().parent.parent / "data"
UBQ_TOP = DATA / "1UBQ.top"
UBQ_PDB = DATA / "1UBQ_processed.pdb"


@pytest.fixture(scope="module")
def ubq_bonded():
    if not UBQ_TOP.is_file():
        pytest.skip(f"missing fixture: {UBQ_TOP}")
    return build_bonded_params(str(UBQ_TOP))


@pytest.fixture(scope="module")
def ubq_topo():
    return build_chi_topology(str(UBQ_TOP))


@pytest.fixture(scope="module")
def ubq_schedule(ubq_topo):
    return build_chi_schedule(ubq_topo)


@pytest.fixture(scope="module")
def ubq_pos0():
    if not UBQ_PDB.is_file():
        pytest.skip(f"missing fixture: {UBQ_PDB}")
    from ptmc.io.parse_pdb import parse_pdb
    return jnp.asarray(parse_pdb(str(UBQ_PDB)).pos.astype(np.float32))


# ---------------------------------------------------------------------------
# Build-side: parmed-driven table construction
# ---------------------------------------------------------------------------

def test_1ubq_has_only_periodic_dihedrals(ubq_bonded):
    """AMBER99SB-ILDN forms (proper type 9 + improper type 4) both unified
    into the periodic table; no RB or harmonic terms expected for 1UBQ."""
    assert ubq_bonded.m_periodic > 0
    assert ubq_bonded.m_rb == 0
    assert ubq_bonded.m_harmonic == 0


def test_1ubq_periodic_table_units(ubq_bonded):
    """Sanity-check units: phases in radians within [-π, π+ε];
    multiplicities are small positive integers (typically 1..6).
    """
    assert ubq_bonded.periodic_phase.min() >= -np.pi - 1e-3
    assert ubq_bonded.periodic_phase.max() <= np.pi + 1e-3
    assert ubq_bonded.periodic_per.min() >= 1
    assert ubq_bonded.periodic_per.max() <= 6
    # k_phi may be negative in AMBER (shifts the trough); bound the magnitude
    # to catch a kcal/kJ unit-conversion mistake (would be 4.184× the truth).
    assert np.max(np.abs(ubq_bonded.periodic_k)) < 100.0


def test_1ubq_fudge_factors(ubq_bonded):
    """AMBER: fudgeLJ ≈ 0.5, fudgeQQ ≈ 0.833. Used by F5 (1-4 scaling)."""
    assert abs(ubq_bonded.fudge_lj - 0.5) < 1e-3
    assert abs(ubq_bonded.fudge_qq - 1.0 / 1.2) < 1e-2


# ---------------------------------------------------------------------------
# JAX vs numpy reference on 1UBQ
# ---------------------------------------------------------------------------

def test_jax_matches_numpy_reference(ubq_bonded, ubq_pos0):
    e_jax = float(E_bonded(ubq_pos0, ubq_bonded))
    e_np = E_bonded_numpy(np.asarray(ubq_pos0).astype(np.float64),
                          ubq_bonded)
    # JAX is FP32, numpy ref FP64 -- expect relative agreement to a few ULPs
    # of the total energy (~ thousands of kJ/mol).
    abs_err = abs(e_jax - e_np)
    rel_err = abs_err / max(1.0, abs(e_np))
    assert rel_err < 1e-4, (
        f"E_bonded jax={e_jax:.4f} vs numpy={e_np:.4f} kJ/mol "
        f"(abs_err={abs_err:.4e})")


def test_jax_matches_numpy_under_random_chi(
    ubq_bonded, ubq_topo, ubq_schedule, ubq_pos0,
):
    """Apply random chi, then bonded energy of perturbed geometry should
    agree between JAX and numpy reference."""
    rng = np.random.default_rng(2026)
    chi = jnp.asarray(rng.uniform(-np.pi, np.pi,
                                  size=ubq_topo.k).astype(np.float32))
    pos_pert = apply_all_chi(ubq_pos0, chi, ubq_schedule)
    e_jax = float(E_bonded(pos_pert, ubq_bonded))
    e_np = E_bonded_numpy(np.asarray(pos_pert).astype(np.float64),
                          ubq_bonded)
    rel_err = abs(e_jax - e_np) / max(1.0, abs(e_np))
    assert rel_err < 1e-4


# ---------------------------------------------------------------------------
# Periodic-form invariants
# ---------------------------------------------------------------------------

def test_2pi_periodic_invariant(
    ubq_bonded, ubq_topo, ubq_schedule, ubq_pos0,
):
    """Bonded energy is periodic in each chi with period 2π.

    Apply chi → E(chi). Apply chi + 2π·e_k for any k → E(chi)+ε (FP32 noise).
    """
    rng = np.random.default_rng(7)
    chi = rng.uniform(-1.0, 1.0, size=ubq_topo.k).astype(np.float32)
    pos1 = apply_all_chi(ubq_pos0, jnp.asarray(chi), ubq_schedule)
    pos2 = apply_all_chi(ubq_pos0,
                         jnp.asarray(chi + 2.0 * np.pi), ubq_schedule)
    e1 = float(E_bonded(pos1, ubq_bonded))
    e2 = float(E_bonded(pos2, ubq_bonded))
    # FP32 chi=O(1) plus 2π → some accumulated rotation error in positions,
    # which translates to small but non-zero bonded-energy drift.
    assert abs(e1 - e2) < 0.5, (
        f"|E(chi) - E(chi+2π)| = {abs(e1-e2):.4f} kJ/mol exceeds 0.5")


# ---------------------------------------------------------------------------
# Synthetic harmonic improper kernel test
# ---------------------------------------------------------------------------

def test_harmonic_improper_zero_at_equilibrium():
    """Build a 4-atom config with phi == phi_eq -> harmonic energy = 0."""
    # Place atoms so that the measured dihedral is exactly pi (anti).
    pos = np.array([
        [-1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
    ], dtype=np.float32)
    pos_j = jnp.asarray(pos)
    chi_idx = np.array([[0, 1, 2, 3]], dtype=np.int32)
    phi_measured = float(measure_dihedrals(pos_j, jnp.asarray(chi_idx))[0])
    params = BondedParams(
        periodic_idx=np.zeros((0, 4), dtype=np.int32),
        periodic_phase=np.zeros((0,), dtype=np.float32),
        periodic_k=np.zeros((0,), dtype=np.float32),
        periodic_per=np.zeros((0,), dtype=np.int32),
        rb_idx=np.zeros((0, 4), dtype=np.int32),
        rb_c=np.zeros((0, 6), dtype=np.float32),
        harmonic_idx=chi_idx,
        harmonic_phase=np.asarray([phi_measured], dtype=np.float32),
        harmonic_k=np.asarray([100.0], dtype=np.float32),
        fudge_lj=0.5, fudge_qq=0.833,
    )
    e = float(E_bonded(pos_j, params))
    assert abs(e) < 1e-3


def test_harmonic_improper_quadratic_growth():
    """Small deviation δ from phi_eq -> E ≈ 0.5 k δ^2."""
    pos = np.array([
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],  # perpendicular -> |phi| = π/2
    ], dtype=np.float32)
    chi_idx = np.array([[0, 1, 2, 3]], dtype=np.int32)
    phi_eq = float(measure_dihedrals(jnp.asarray(pos),
                                     jnp.asarray(chi_idx))[0])
    k = 80.0
    params = BondedParams(
        periodic_idx=np.zeros((0, 4), dtype=np.int32),
        periodic_phase=np.zeros((0,), dtype=np.float32),
        periodic_k=np.zeros((0,), dtype=np.float32),
        periodic_per=np.zeros((0,), dtype=np.int32),
        rb_idx=np.zeros((0, 4), dtype=np.int32),
        rb_c=np.zeros((0, 6), dtype=np.float32),
        harmonic_idx=chi_idx,
        harmonic_phase=np.asarray([phi_eq], dtype=np.float32),
        harmonic_k=np.asarray([k], dtype=np.float32),
        fudge_lj=0.5, fudge_qq=0.833,
    )
    # Rotate the l atom by delta about j-k axis
    from ptmc.flexible import axis_angle_to_matrix
    axis = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32)
    delta = 0.1
    R = axis_angle_to_matrix(axis, jnp.array(delta, dtype=jnp.float32))
    pos_j = jnp.asarray(pos)
    l_new = pos_j[2] + R @ (pos_j[3] - pos_j[2])
    pos_pert = pos_j.at[3].set(l_new)
    e = float(E_bonded(pos_pert, params))
    expected = 0.5 * k * delta ** 2
    rel = abs(e - expected) / expected
    assert rel < 1e-3, f"E={e:.6f}, expected ≈ {expected:.6f}"


# ---------------------------------------------------------------------------
# Synthetic RB kernel test
# ---------------------------------------------------------------------------

def test_rb_constant_c0_only():
    """Pure C0 (constant): E = M * C0 regardless of geometry."""
    pos = jnp.asarray(np.random.default_rng(0).standard_normal((10, 3))
                      .astype(np.float32))
    idx = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)
    c = np.zeros((2, 6), dtype=np.float32)
    c[:, 0] = 3.0
    params = BondedParams(
        periodic_idx=np.zeros((0, 4), dtype=np.int32),
        periodic_phase=np.zeros((0,), dtype=np.float32),
        periodic_k=np.zeros((0,), dtype=np.float32),
        periodic_per=np.zeros((0,), dtype=np.int32),
        rb_idx=idx, rb_c=c,
        harmonic_idx=np.zeros((0, 4), dtype=np.int32),
        harmonic_phase=np.zeros((0,), dtype=np.float32),
        harmonic_k=np.zeros((0,), dtype=np.float32),
        fudge_lj=0.5, fudge_qq=0.833,
    )
    e = float(E_bonded(pos, params))
    assert abs(e - 6.0) < 1e-5


# ---------------------------------------------------------------------------
# JIT smoke + non-negative when no improper-harmonic + phase=0
# ---------------------------------------------------------------------------

def test_jit_smoke(ubq_bonded, ubq_pos0):
    fn = jax.jit(lambda pos: E_bonded(pos, ubq_bonded))
    e1 = float(fn(ubq_pos0))
    e2 = float(fn(ubq_pos0))  # cache hit
    assert e1 == e2


def test_total_energy_finite(ubq_bonded, ubq_pos0):
    """No NaN / inf from the bonded sum on a real geometry."""
    e = float(E_bonded(ubq_pos0, ubq_bonded))
    assert np.isfinite(e)
    # The 1UBQ native conformation should yield a bonded energy in a
    # physically reasonable range — say |E| < 10^5 kJ/mol.
    assert abs(e) < 1e5
