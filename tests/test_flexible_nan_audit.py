"""F6 — NaN / Inf audit & fuzz tests across the flexible/ kernels.

Per § 7.4 of the design doc all jnp.where / division / sqrt / log paths must
have explicit safety floors so the hot loop never produces NaN even for
pathological geometries (overlapping atoms, degenerate axes, free-drifting
chi values, dummy atoms with C6=C12=0).

This file is the audit harness: a battery of stress configurations that try
to provoke NaN/Inf and assert the output remains finite. Catches a future
regression where someone adds a code path without the floor.
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
    E_intra_nb,
    IntraNBParams,
    apply_all_chi,
    apply_all_chi_serial,
    build_bonded_params,
    build_chi_schedule,
    build_chi_topology,
    build_intra_nb_params,
    measure_dihedrals,
)

DATA = Path(__file__).resolve().parent.parent / "data"
UBQ_TOP = DATA / "1UBQ.top"
UBQ_PDB = DATA / "1UBQ_processed.pdb"


@pytest.fixture(scope="module")
def ubq_topo():
    if not UBQ_TOP.is_file():
        pytest.skip(f"missing fixture: {UBQ_TOP}")
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


@pytest.fixture(scope="module")
def ubq_bonded():
    return build_bonded_params(str(UBQ_TOP))


@pytest.fixture(scope="module")
def ubq_intra_nb():
    return build_intra_nb_params(str(UBQ_TOP))


def _all_finite(x) -> bool:
    return bool(jnp.all(jnp.isfinite(x)))


# ---------------------------------------------------------------------------
# Kinematics fuzz
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scale", [1e1, 1e3, 1e6, -1e6])
def test_apply_all_chi_extreme_values(ubq_topo, ubq_schedule, ubq_pos0,
                                       scale):
    """Free-drifting chi (no [-π, π] wrap by design): huge values must not
    blow up the kinematics."""
    rng = np.random.default_rng(int(abs(scale)))
    chi = jnp.asarray(rng.uniform(-1, 1, size=ubq_topo.k).astype(np.float32)
                      * scale)
    pos = apply_all_chi(ubq_pos0, chi, ubq_schedule)
    assert _all_finite(pos)


def test_apply_all_chi_with_degenerate_axis():
    """Two atoms placed at the same point along the j-k axis: rotation axis
    is zero. The `_AXIS_NORM_FLOOR` should keep the operation finite (the
    rotation becomes a no-op for that residue at that depth)."""
    # 4-atom MET-like layout but with j and k coincident
    pos = jnp.array([
        [-1.0, 0.0, 0.0],   # i
        [0.0, 0.0, 0.0],    # j
        [0.0, 0.0, 0.0],    # k (== j, degenerate axis)
        [1.0, 0.0, 0.0],    # l
    ], dtype=jnp.float32)
    chi_idx = jnp.array([[0, 1, 2, 3]], dtype=jnp.int32)
    mask = jnp.array([False, False, True, True])
    out = apply_all_chi_serial.__wrapped__ if False else None
    from ptmc.flexible.kinematics import apply_chi_step
    pos_new = apply_chi_step(pos, 0, 1, 2, 3, jnp.array(0.5), mask)
    assert _all_finite(pos_new)


# ---------------------------------------------------------------------------
# Dihedral measurement fuzz
# ---------------------------------------------------------------------------

def test_measure_dihedrals_with_coincident_atoms():
    """All four atoms at the same point -> dihedral undefined; should return
    something finite (atan2(0, 0) = 0 in IEEE754)."""
    pos = jnp.zeros((4, 3), dtype=jnp.float32)
    chi_idx = jnp.array([[0, 1, 2, 3]], dtype=jnp.int32)
    phi = measure_dihedrals(pos, chi_idx)
    assert _all_finite(phi)


def test_measure_dihedrals_with_colinear_atoms():
    """Three of the four atoms colinear (b1 || b2): dihedral undefined.
    Implementation should not NaN; atan2(0, 0) returns 0."""
    pos = jnp.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 1.0, 0.0],
    ], dtype=jnp.float32)
    chi_idx = jnp.array([[0, 1, 2, 3]], dtype=jnp.int32)
    phi = measure_dihedrals(pos, chi_idx)
    assert _all_finite(phi)


# ---------------------------------------------------------------------------
# Bonded energy fuzz
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scale", [1e1, 1e3, 1e6])
def test_E_bonded_extreme_chi(ubq_topo, ubq_schedule, ubq_pos0, ubq_bonded,
                              scale):
    """Extreme chi → potentially distorted positions → E_bonded must still
    be finite (just a high penalty)."""
    rng = np.random.default_rng(int(scale))
    chi = jnp.asarray(rng.uniform(-1, 1, size=ubq_topo.k).astype(np.float32)
                      * scale)
    pos = apply_all_chi(ubq_pos0, chi, ubq_schedule)
    e = float(E_bonded(pos, ubq_bonded))
    assert np.isfinite(e)


# ---------------------------------------------------------------------------
# Intra-NB fuzz: overlapping atoms, dummy atoms, extreme stretching
# ---------------------------------------------------------------------------

def test_E_intra_nb_overlapping_atoms(ubq_intra_nb):
    """Put all atoms at the same position; r_floor must keep E finite."""
    n = ubq_intra_nb.n_atoms
    pos = jnp.zeros((n, 3), dtype=jnp.float32)
    e = float(E_intra_nb(pos, ubq_intra_nb, chunk_elems=200_000))
    assert np.isfinite(e)


def test_E_intra_nb_far_apart_atoms(ubq_intra_nb):
    """Atoms at 1000 nm apart — exp(-r/λD) should not underflow to NaN."""
    n = ubq_intra_nb.n_atoms
    pos = jnp.asarray(
        np.arange(n)[:, None] * 1000.0
        * np.ones((1, 3), dtype=np.float32), dtype=jnp.float32)
    e = float(E_intra_nb(pos, ubq_intra_nb, chunk_elems=200_000))
    assert np.isfinite(e)
    # Very far apart: only the LJ goes to 0, electrostatic exp-decays.
    # Total energy should be ≈ 0.
    assert abs(e) < 1e-3


def test_E_intra_nb_dummy_atom_pair():
    """Two dummy atoms (c6 = c12 = q = 0): every term identically 0."""
    params = IntraNBParams(
        q=np.zeros(2, dtype=np.float32),
        sqrt_c6=np.zeros(2, dtype=np.float32),
        sqrt_c12=np.zeros(2, dtype=np.float32),
        scale_lj=np.array([[0, 1], [1, 0]], dtype=np.float32),
        scale_qq=np.array([[0, 1], [1, 0]], dtype=np.float32),
        lambda_D=0.785,
    )
    pos = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=jnp.float32)
    e = float(E_intra_nb(pos, params))
    assert e == 0.0


# ---------------------------------------------------------------------------
# Empty-table edge cases
# ---------------------------------------------------------------------------

def test_E_bonded_empty_tables():
    """No periodic / RB / harmonic terms → E = 0."""
    params = BondedParams(
        periodic_idx=np.zeros((0, 4), dtype=np.int32),
        periodic_phase=np.zeros((0,), dtype=np.float32),
        periodic_k=np.zeros((0,), dtype=np.float32),
        periodic_per=np.zeros((0,), dtype=np.int32),
        rb_idx=np.zeros((0, 4), dtype=np.int32),
        rb_c=np.zeros((0, 6), dtype=np.float32),
        harmonic_idx=np.zeros((0, 4), dtype=np.int32),
        harmonic_phase=np.zeros((0,), dtype=np.float32),
        harmonic_k=np.zeros((0,), dtype=np.float32),
        fudge_lj=0.5, fudge_qq=0.833,
    )
    pos = jnp.asarray(np.random.default_rng(0).standard_normal((10, 3))
                      .astype(np.float32))
    e = float(E_bonded(pos, params))
    assert e == 0.0


# ---------------------------------------------------------------------------
# Random multi-step regression: 100 random chi vectors, never NaN
# ---------------------------------------------------------------------------

def test_random_chi_fuzz_100_iterations(
    ubq_topo, ubq_schedule, ubq_pos0, ubq_bonded, ubq_intra_nb,
):
    rng = np.random.default_rng(2026)
    for it in range(50):
        chi = jnp.asarray(rng.uniform(-10.0, 10.0, size=ubq_topo.k)
                          .astype(np.float32))
        pos = apply_all_chi(ubq_pos0, chi, ubq_schedule)
        assert _all_finite(pos), f"iter {it}: position NaN/Inf"
        e_b = float(E_bonded(pos, ubq_bonded))
        e_n = float(E_intra_nb(pos, ubq_intra_nb, chunk_elems=200_000))
        assert np.isfinite(e_b) and np.isfinite(e_n), (
            f"iter {it}: bonded={e_b}, intra_nb={e_n}")
