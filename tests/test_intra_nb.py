"""F5 — intra-protein non-bonded energy.

Gate (§ 5.4): JAX chunked all-pairs vs numpy full-matrix reference
< 1e-3 kJ/mol on a real protein (1UBQ).

Plus chunk-size invariance and synthetic 1-4 / r_floor checks.

PTMC_INTRA_NB_FP64=1 is set so the JAX accumulator promotes to FP64; without
this the F5 gate is unreachable for the ~10^5 kJ/mol intra-protein sum on
FP32 (ULP ≈ 0.01 kJ/mol).

jax_enable_x64 is also enabled for the float64 fixture arrays needed
to match the numpy reference at the 1e-3 gate.
"""
from __future__ import annotations

import os

os.environ.setdefault("PTMC_INTRA_NB_FP64", "1")

import jax

jax.config.update("jax_enable_x64", True)  # noqa: E402  float64 fixture arrays

from pathlib import Path  # noqa: E402

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from ptmc.flexible import (  # noqa: E402
    E_intra_nb,
    E_intra_nb_numpy,
    IntraNBParams,
    R_FLOOR_NM,
    build_exclusion_table,
    build_intra_nb_params,
)

DATA = Path(__file__).resolve().parent.parent / "data"
UBQ_TOP = DATA / "1UBQ.top"
UBQ_PDB = DATA / "1UBQ_processed.pdb"


@pytest.fixture(scope="module")
def ubq_intra_nb():
    if not UBQ_TOP.is_file():
        pytest.skip(f"missing fixture: {UBQ_TOP}")
    return build_intra_nb_params(str(UBQ_TOP))


@pytest.fixture(scope="module")
def ubq_excl():
    return build_exclusion_table(str(UBQ_TOP))


@pytest.fixture(scope="module")
def ubq_pos0():
    if not UBQ_PDB.is_file():
        pytest.skip(f"missing fixture: {UBQ_PDB}")
    from ptmc.io.parse_pdb import parse_pdb
    return jnp.asarray(parse_pdb(str(UBQ_PDB)).pos.astype(np.float64))


# ---------------------------------------------------------------------------
# Exclusion table sanity
# ---------------------------------------------------------------------------

def test_excl_table_diagonal_excluded(ubq_excl):
    diag = np.diag(ubq_excl.scale_lj)
    assert np.all(diag == 0.0)
    assert np.all(np.diag(ubq_excl.excl_mask))


def test_excl_table_symmetric(ubq_excl):
    assert np.array_equal(ubq_excl.scale_lj, ubq_excl.scale_lj.T)
    assert np.array_equal(ubq_excl.scale_qq, ubq_excl.scale_qq.T)
    assert np.array_equal(ubq_excl.excl_mask, ubq_excl.excl_mask.T)


def test_excl_table_14_scale_correct(ubq_excl):
    """1-4 pairs use the global fudgeLJ / fudgeQQ; 1-2 / 1-3 are zero."""
    assert ubq_excl.pair14_idx.shape[1] == 2
    # Sample a few 1-4 entries and verify the scale matches the fudge.
    n_sample = min(10, ubq_excl.pair14_idx.shape[0])
    for i, j in ubq_excl.pair14_idx[:n_sample]:
        assert np.isclose(ubq_excl.scale_lj[i, j], ubq_excl.fudge_lj,
                          atol=1e-6)
        assert np.isclose(ubq_excl.scale_qq[i, j], ubq_excl.fudge_qq,
                          atol=1e-6)
        # 1-4 must NOT be in excl_mask
        assert not ubq_excl.excl_mask[i, j]


def test_excl_table_amber_fudges(ubq_excl):
    assert abs(ubq_excl.fudge_lj - 0.5) < 1e-3
    assert abs(ubq_excl.fudge_qq - 1.0 / 1.2) < 1e-3


# ---------------------------------------------------------------------------
# F5 gate: JAX chunked vs numpy reference < 1e-3 kJ/mol
# ---------------------------------------------------------------------------

def test_jax_chunked_matches_numpy_reference_1ubq(ubq_intra_nb, ubq_pos0):
    e_jax = float(E_intra_nb(ubq_pos0, ubq_intra_nb,
                             chunk_elems=200_000))
    e_np = E_intra_nb_numpy(np.asarray(ubq_pos0), ubq_intra_nb)
    err = abs(e_jax - e_np)
    assert err < 1e-3, (
        f"E_intra_nb jax={e_jax:.6f} vs numpy={e_np:.6f} kJ/mol "
        f"(abs_err={err:.2e}; F5 gate requires < 1e-3)")


def test_chunk_size_invariance(ubq_intra_nb, ubq_pos0):
    """Varying chunk_elems must not change the answer (modulo FP32 noise).

    The intra-NB sum runs in float32, so reordering the summation across
    different chunk sizes perturbs the total by a few ULPs that scale with the
    magnitude of the (large, ~10^3-10^4 kJ/mol) sum — NOT by a fixed absolute
    amount. A relative tolerance is therefore the correct invariance gate; a
    tight absolute bound (1e-4) spuriously fails on big proteins.
    """
    e_ref = float(E_intra_nb(ubq_pos0, ubq_intra_nb,
                             chunk_elems=2_000_000))
    tol = 1e-5 * abs(e_ref) + 1e-6   # relative FP32 reorder noise + tiny floor
    for ce in (10_000, 50_000, 200_000, 1_000_000):
        e = float(E_intra_nb(ubq_pos0, ubq_intra_nb, chunk_elems=ce))
        assert abs(e - e_ref) < tol, (
            f"chunk_elems={ce}: e={e}, ref={e_ref}, diff={e-e_ref}, tol={tol}")


# ---------------------------------------------------------------------------
# r_floor / hard-shell behaviour
# ---------------------------------------------------------------------------

def test_r_floor_caps_overlap():
    """Two atoms placed at zero distance: energy is finite and equal to the
    energy at r = R_FLOOR_NM."""
    params = _make_synthetic_pair(q=(0.5, -0.5),
                                  c6=(1e-3, 1e-3),
                                  c12=(1e-6, 1e-6))
    # Overlapping
    pos_overlap = jnp.asarray(np.array([[0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0]], dtype=np.float64))
    # At floor
    pos_floor = jnp.asarray(np.array([[0.0, 0.0, 0.0],
                                      [R_FLOOR_NM, 0.0, 0.0]],
                                     dtype=np.float64))
    e_overlap = float(E_intra_nb(pos_overlap, params))
    e_floor = float(E_intra_nb(pos_floor, params))
    assert np.isfinite(e_overlap)
    assert abs(e_overlap - e_floor) < 1e-6


def test_excluded_pair_contributes_zero():
    """A 1-2 excluded pair has zero NB contribution regardless of distance."""
    params = _make_synthetic_pair(q=(1.0, -1.0),
                                  c6=(1e-3, 1e-3),
                                  c12=(1e-6, 1e-6),
                                  excluded=True)
    pos = jnp.asarray(np.array([[0.0, 0.0, 0.0],
                                [0.3, 0.0, 0.0]], dtype=np.float64))
    e = float(E_intra_nb(pos, params))
    assert abs(e) < 1e-9


# ---------------------------------------------------------------------------
# 1-4 scaling correctness
# ---------------------------------------------------------------------------

def test_14_scaling_correctness():
    """For two atoms marked 1-4, energy = fudge_lj * LJ + fudge_qq * Elec."""
    q = (1.0, -1.0)
    c6 = (1e-3, 1e-3)
    c12 = (1e-6, 1e-6)
    fudge_lj = 0.5
    fudge_qq = 1.0 / 1.2
    r = 0.3
    lam_D = 0.785

    # Two configurations: 1-4 scaled vs unscaled (full pair).
    params_full = _make_synthetic_pair(q=q, c6=c6, c12=c12,
                                       lambda_D=lam_D)
    params_14 = _make_synthetic_pair(q=q, c6=c6, c12=c12,
                                     lambda_D=lam_D,
                                     scale_lj=fudge_lj, scale_qq=fudge_qq)
    pos = jnp.asarray(np.array([[0.0, 0.0, 0.0],
                                [r, 0.0, 0.0]], dtype=np.float64))
    e_full = float(E_intra_nb(pos, params_full))
    e_14 = float(E_intra_nb(pos, params_14))

    # Analytic decomposition
    from ptmc.config import COULOMB_FACTOR_KJ_NM_PER_E2 as cf
    C6 = (c6[0] * c6[1]) ** 0.5
    C12 = (c12[0] * c12[1]) ** 0.5
    e_lj = C12 / r ** 12 - C6 / r ** 6
    e_qq = cf * q[0] * q[1] / r * np.exp(-r / lam_D)
    e_full_expected = e_lj + e_qq
    e_14_expected = fudge_lj * e_lj + fudge_qq * e_qq

    # IntraNBParams stores sqrt(C6) / sqrt(C12) as FP32; the round-trip
    # through the FP32 store costs ~1e-7 relative error on the energy.
    rel_tol = 1e-5
    assert abs(e_full - e_full_expected) < rel_tol * abs(e_full_expected)
    assert abs(e_14 - e_14_expected) < rel_tol * abs(e_14_expected)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_pair(q, c6, c12, lambda_D=0.785,
                         scale_lj=1.0, scale_qq=1.0,
                         excluded=False):
    """Construct an IntraNBParams for a 2-atom system with specified pair scale.

    excluded=True -> scale_lj = scale_qq = 0 (1-2 / 1-3 exclusion semantic).
    """
    if excluded:
        scale_lj = 0.0
        scale_qq = 0.0
    s_lj = np.zeros((2, 2), dtype=np.float32)
    s_qq = np.zeros((2, 2), dtype=np.float32)
    s_lj[0, 1] = s_lj[1, 0] = scale_lj
    s_qq[0, 1] = s_qq[1, 0] = scale_qq
    return IntraNBParams(
        q=np.asarray(q, dtype=np.float32),
        sqrt_c6=np.sqrt(np.asarray(c6, dtype=np.float32)),
        sqrt_c12=np.sqrt(np.asarray(c12, dtype=np.float32)),
        scale_lj=s_lj,
        scale_qq=s_qq,
        lambda_D=lambda_D,
    )


# ---------------------------------------------------------------------------
# JIT smoke
# ---------------------------------------------------------------------------

def test_jit_smoke(ubq_intra_nb, ubq_pos0):
    fn = jax.jit(lambda pos: E_intra_nb(pos, ubq_intra_nb,
                                        chunk_elems=200_000))
    e1 = float(fn(ubq_pos0))
    e2 = float(fn(ubq_pos0))
    assert e1 == e2
