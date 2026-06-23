"""F2 — single-chi rotation + residue-parallel apply_all_chi.

Validates:
  - chi=0 round-trips: pos == pos0
  - 2π rotation returns to pos0 (FP32 tolerance)
  - random chi preserves every bond length in the topology
  - random chi preserves every bond angle
  - residue-parallel apply_all_chi matches naive K-step serial baseline
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ptmc.flexible import (
    apply_all_chi,
    apply_all_chi_serial,
    apply_chi_step,
    axis_angle_to_matrix,
    build_chi_schedule,
    build_chi_topology,
)

DATA = Path(__file__).resolve().parent.parent / "data"
UBQ_TOP = DATA / "1UBQ.top"
UBQ_PDB = DATA / "1UBQ_processed.pdb"


@pytest.fixture(scope="module")
def ubq_pmd():
    if not UBQ_TOP.is_file():
        pytest.skip(f"missing fixture: {UBQ_TOP}")
    import parmed as pmd
    return pmd.load_file(str(UBQ_TOP), parametrize=True)


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
    # 1UBQ.top has no coordinates -- load them from the matching PDB.
    if not UBQ_PDB.is_file():
        pytest.skip(f"missing fixture: {UBQ_PDB}")
    from ptmc.io.parse_pdb import parse_pdb
    pdb = parse_pdb(str(UBQ_PDB))
    return jnp.asarray(pdb.pos.astype(np.float32))


# ---------------------------------------------------------------------------
# axis_angle_to_matrix sanity
# ---------------------------------------------------------------------------

def test_axis_angle_zero_is_identity():
    axis = jnp.array([1.0, 0.0, 0.0])
    R = axis_angle_to_matrix(axis, jnp.array(0.0))
    np.testing.assert_allclose(np.asarray(R), np.eye(3), atol=1e-7)


def test_axis_angle_2pi_is_identity():
    axis = jnp.array([0.0, 0.0, 1.0])
    R = axis_angle_to_matrix(axis, jnp.array(2.0 * np.pi))
    np.testing.assert_allclose(np.asarray(R), np.eye(3), atol=1e-6)


def test_axis_angle_half_turn():
    """π rotation about z: (x, y, z) -> (-x, -y, z)."""
    axis = jnp.array([0.0, 0.0, 1.0])
    R = axis_angle_to_matrix(axis, jnp.array(np.pi))
    v = jnp.array([0.3, 0.4, 0.5])
    np.testing.assert_allclose(
        np.asarray(R @ v), np.array([-0.3, -0.4, 0.5]), atol=1e-6)


# ---------------------------------------------------------------------------
# Schedule structure on 1UBQ
# ---------------------------------------------------------------------------

def test_schedule_shapes(ubq_topo, ubq_schedule):
    D = ubq_schedule.max_n_chi
    R = ubq_schedule.n_flex_res
    N = ubq_schedule.n_atoms
    assert N == ubq_topo.n_atoms
    assert D >= 1
    assert R >= 1
    assert ubq_schedule.chi_by_depth.shape == (D, R, 4)
    assert ubq_schedule.mask_by_depth.shape == (D, R, N)
    assert ubq_schedule.valid_by_depth.shape == (D, R)
    assert ubq_schedule.chi_global_idx.shape == (D, R)
    assert ubq_schedule.owner_per_depth.shape == (D, N)


def test_schedule_valid_count_matches_K(ubq_topo, ubq_schedule):
    assert int(ubq_schedule.valid_by_depth.sum()) == ubq_topo.k


def test_schedule_disjoint_owner_per_depth(ubq_schedule):
    """Owner must be -1 wherever no residue claims; never duplicated across r."""
    # Already validated at build time; here we double-check via masks.
    D, R, N = (ubq_schedule.max_n_chi, ubq_schedule.n_flex_res,
               ubq_schedule.n_atoms)
    for d in range(D):
        owner = ubq_schedule.owner_per_depth[d]
        any_owner = np.any(ubq_schedule.mask_by_depth[d], axis=0)
        assert np.all((owner >= 0) == any_owner)
        # For atoms with owner, mask of that residue at this depth must be True
        for a in np.where(owner >= 0)[0]:
            r = int(owner[a])
            assert ubq_schedule.mask_by_depth[d, r, a]


# ---------------------------------------------------------------------------
# Trivial chi: zero, 2π → identity
# ---------------------------------------------------------------------------

def test_apply_all_chi_zero_is_identity(ubq_topo, ubq_schedule, ubq_pos0):
    chi = jnp.zeros((ubq_topo.k,), dtype=jnp.float32)
    pos = apply_all_chi(ubq_pos0, chi, ubq_schedule)
    np.testing.assert_allclose(np.asarray(pos), np.asarray(ubq_pos0),
                               atol=1e-6)


def test_apply_all_chi_2pi_is_identity(ubq_topo, ubq_schedule, ubq_pos0):
    chi = jnp.full((ubq_topo.k,), 2.0 * np.pi, dtype=jnp.float32)
    pos = apply_all_chi(ubq_pos0, chi, ubq_schedule)
    # 2π through 4 depths can accumulate FP32 error; loose tol.
    np.testing.assert_allclose(np.asarray(pos), np.asarray(ubq_pos0),
                               atol=1e-4)


# ---------------------------------------------------------------------------
# Bond length & bond angle conservation under random chi
# ---------------------------------------------------------------------------

def _bond_pairs_from_pmd(top) -> np.ndarray:
    return np.array([(b.atom1.idx, b.atom2.idx) for b in top.bonds],
                    dtype=np.int64)


def _angle_triplets_from_pmd(top) -> np.ndarray:
    return np.array([(a.atom1.idx, a.atom2.idx, a.atom3.idx)
                     for a in top.angles], dtype=np.int64)


def test_bond_lengths_conserved_under_random_chi(
    ubq_topo, ubq_schedule, ubq_pos0, ubq_pmd,
):
    bonds = _bond_pairs_from_pmd(ubq_pmd)
    rng = np.random.default_rng(42)
    chi = jnp.asarray(rng.uniform(-np.pi, np.pi,
                                  size=ubq_topo.k).astype(np.float32))
    pos = np.asarray(apply_all_chi(ubq_pos0, chi, ubq_schedule))
    pos0 = np.asarray(ubq_pos0)

    d0 = np.linalg.norm(pos0[bonds[:, 0]] - pos0[bonds[:, 1]], axis=-1)
    d1 = np.linalg.norm(pos[bonds[:, 0]] - pos[bonds[:, 1]], axis=-1)
    err = np.abs(d0 - d1)
    assert err.max() < 1e-4, (
        f"max bond-length deviation {err.max():.2e} nm exceeds 1e-4 (mean "
        f"{err.mean():.2e}, n_bonds={len(bonds)})")


def test_bond_angles_conserved_under_random_chi(
    ubq_topo, ubq_schedule, ubq_pos0, ubq_pmd,
):
    angles = _angle_triplets_from_pmd(ubq_pmd)
    rng = np.random.default_rng(7)
    chi = jnp.asarray(rng.uniform(-np.pi, np.pi,
                                  size=ubq_topo.k).astype(np.float32))
    pos = np.asarray(apply_all_chi(ubq_pos0, chi, ubq_schedule))
    pos0 = np.asarray(ubq_pos0)

    def _angle(p, ijk):
        a = p[ijk[:, 0]] - p[ijk[:, 1]]
        b = p[ijk[:, 2]] - p[ijk[:, 1]]
        cos = (a * b).sum(-1) / (
            np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-30)
        return np.arccos(np.clip(cos, -1.0, 1.0))

    th0 = _angle(pos0, angles)
    th1 = _angle(pos, angles)
    err = np.abs(th0 - th1)
    assert err.max() < 1e-3, (
        f"max bond-angle deviation {err.max():.2e} rad exceeds 1e-3 "
        f"(mean {err.mean():.2e}, n_angles={len(angles)})")


# ---------------------------------------------------------------------------
# Naive serial vs residue-parallel equivalence
# ---------------------------------------------------------------------------

def test_parallel_equals_serial_for_random_chi(
    ubq_topo, ubq_schedule, ubq_pos0,
):
    rng = np.random.default_rng(123)
    chi = jnp.asarray(rng.uniform(-np.pi, np.pi,
                                  size=ubq_topo.k).astype(np.float32))
    pos_par = np.asarray(apply_all_chi(ubq_pos0, chi, ubq_schedule))
    pos_ser = np.asarray(apply_all_chi_serial(ubq_pos0, chi, ubq_topo))
    np.testing.assert_allclose(pos_par, pos_ser, atol=1e-5, rtol=0)


def test_parallel_equals_serial_with_zero_chi(
    ubq_topo, ubq_schedule, ubq_pos0,
):
    chi = jnp.zeros((ubq_topo.k,), dtype=jnp.float32)
    pos_par = np.asarray(apply_all_chi(ubq_pos0, chi, ubq_schedule))
    pos_ser = np.asarray(apply_all_chi_serial(ubq_pos0, chi, ubq_topo))
    np.testing.assert_allclose(pos_par, pos_ser, atol=1e-6, rtol=0)


def test_parallel_equals_serial_single_chi_only(
    ubq_topo, ubq_schedule, ubq_pos0,
):
    """Activate exactly one chi (middle of the array) and compare."""
    chi_np = np.zeros(ubq_topo.k, dtype=np.float32)
    chi_np[ubq_topo.k // 2] = 0.7
    chi = jnp.asarray(chi_np)
    pos_par = np.asarray(apply_all_chi(ubq_pos0, chi, ubq_schedule))
    pos_ser = np.asarray(apply_all_chi_serial(ubq_pos0, chi, ubq_topo))
    np.testing.assert_allclose(pos_par, pos_ser, atol=1e-5, rtol=0)


# ---------------------------------------------------------------------------
# JIT smoke test
# ---------------------------------------------------------------------------

def test_apply_all_chi_jits(ubq_topo, ubq_schedule, ubq_pos0):
    fn = jax.jit(lambda pos0, chi: apply_all_chi(pos0, chi, ubq_schedule))
    chi = jnp.zeros((ubq_topo.k,), dtype=jnp.float32)
    pos = fn(ubq_pos0, chi)
    np.testing.assert_allclose(np.asarray(pos), np.asarray(ubq_pos0),
                               atol=1e-6)
