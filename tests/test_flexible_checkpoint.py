"""G2 — checkpoint / resume byte-equivalence with chi state.

Contract: running [0, N) directly must be byte-identical to running [0, k)
then resuming [k, N) from the saved carry. The positional RNG semantic
(fold_in(master, step_idx)) makes this trivially achievable; this test
locks down the contract so future refactors can't silently break it.

All four state components participate:
    quat (4,), trans (3,), chi (K,), energy ()
plus the accumulators
    acc_counts (3,), try_counts (3,)
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ptmc.mc.flex_metropolis import (
    dof_move_weights,
    scan_flexible_metropolis,
)


def _make_chain(K=3):
    init_chi = jnp.asarray(np.linspace(-0.1, 0.1, K), dtype=jnp.float32)
    init_quat = jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    init_trans = jnp.asarray([0.0, 0.0, 0.5], dtype=jnp.float32)
    return init_quat, init_trans, init_chi


def _E(q, t, c):
    return 5.0 * jnp.sum(c * c) + jnp.sum(t * t)


SIGMA_ROT = 0.05
SIGMA_TRANS = 0.05
SIGMA_CHI = 0.1
BETA = 1.0
AXIS_MASK = jnp.ones(3, dtype=jnp.float32)


# ---------------------------------------------------------------------------
# Direct vs split run: byte-identical (incl. chi)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("split_at", [50, 137, 250])
def test_resume_byte_identical_to_direct_run(split_at):
    """Run [0, N) vs [0, k) + [k, N). All carry components byte-equal."""
    init_quat, init_trans, init_chi = _make_chain()
    K = init_chi.shape[0]
    weights = dof_move_weights(K)
    key = jax.random.PRNGKey(2026)
    N = 500

    # Continuous
    full = scan_flexible_metropolis(
        key, init_quat, init_trans, init_chi,
        _E, BETA, SIGMA_ROT, SIGMA_TRANS, SIGMA_CHI,
        AXIS_MASK, weights, N, start_step=0,
    )

    # Two-segment
    seg1 = scan_flexible_metropolis(
        key, init_quat, init_trans, init_chi,
        _E, BETA, SIGMA_ROT, SIGMA_TRANS, SIGMA_CHI,
        AXIS_MASK, weights, split_at, start_step=0,
    )
    seg2 = scan_flexible_metropolis(
        key, seg1["quat_final"], seg1["trans_final"], seg1["chi_final"],
        _E, BETA, SIGMA_ROT, SIGMA_TRANS, SIGMA_CHI,
        AXIS_MASK, weights, N - split_at, start_step=split_at,
    )

    np.testing.assert_array_equal(
        np.asarray(seg2["quat_final"]), np.asarray(full["quat_final"]),
        err_msg="quat_final not byte-identical")
    np.testing.assert_array_equal(
        np.asarray(seg2["trans_final"]), np.asarray(full["trans_final"]),
        err_msg="trans_final not byte-identical")
    np.testing.assert_array_equal(
        np.asarray(seg2["chi_final"]), np.asarray(full["chi_final"]),
        err_msg="chi_final not byte-identical")
    np.testing.assert_array_equal(
        np.asarray(seg2["energy_final"]), np.asarray(full["energy_final"]),
        err_msg="energy_final not byte-identical")


def test_acc_counters_sum_across_segments():
    """Per-type counters [0, k) + [k, N) sum to the counters from [0, N)."""
    init_quat, init_trans, init_chi = _make_chain()
    K = init_chi.shape[0]
    weights = dof_move_weights(K)
    key = jax.random.PRNGKey(99)
    N = 400
    split = 150

    full = scan_flexible_metropolis(
        key, init_quat, init_trans, init_chi,
        _E, BETA, SIGMA_ROT, SIGMA_TRANS, SIGMA_CHI,
        AXIS_MASK, weights, N, start_step=0,
    )
    seg1 = scan_flexible_metropolis(
        key, init_quat, init_trans, init_chi,
        _E, BETA, SIGMA_ROT, SIGMA_TRANS, SIGMA_CHI,
        AXIS_MASK, weights, split, start_step=0,
    )
    seg2 = scan_flexible_metropolis(
        key, seg1["quat_final"], seg1["trans_final"], seg1["chi_final"],
        _E, BETA, SIGMA_ROT, SIGMA_TRANS, SIGMA_CHI,
        AXIS_MASK, weights, N - split, start_step=split,
    )
    np.testing.assert_array_equal(
        np.asarray(full["acc_counts"]),
        np.asarray(seg1["acc_counts"]) + np.asarray(seg2["acc_counts"]),
    )
    np.testing.assert_array_equal(
        np.asarray(full["try_counts"]),
        np.asarray(seg1["try_counts"]) + np.asarray(seg2["try_counts"]),
    )


def test_three_way_split_byte_identical():
    """Three resumption points -> same final state as the continuous run."""
    init_quat, init_trans, init_chi = _make_chain(K=4)
    K = init_chi.shape[0]
    weights = dof_move_weights(K)
    key = jax.random.PRNGKey(1234)
    N = 600

    full = scan_flexible_metropolis(
        key, init_quat, init_trans, init_chi,
        _E, BETA, SIGMA_ROT, SIGMA_TRANS, SIGMA_CHI,
        AXIS_MASK, weights, N, start_step=0,
    )

    # 0 -> 100 -> 350 -> 600
    s1 = scan_flexible_metropolis(
        key, init_quat, init_trans, init_chi, _E, BETA,
        SIGMA_ROT, SIGMA_TRANS, SIGMA_CHI, AXIS_MASK, weights,
        100, start_step=0)
    s2 = scan_flexible_metropolis(
        key, s1["quat_final"], s1["trans_final"], s1["chi_final"],
        _E, BETA, SIGMA_ROT, SIGMA_TRANS, SIGMA_CHI, AXIS_MASK, weights,
        250, start_step=100)
    s3 = scan_flexible_metropolis(
        key, s2["quat_final"], s2["trans_final"], s2["chi_final"],
        _E, BETA, SIGMA_ROT, SIGMA_TRANS, SIGMA_CHI, AXIS_MASK, weights,
        250, start_step=350)

    np.testing.assert_array_equal(
        np.asarray(s3["chi_final"]), np.asarray(full["chi_final"]))
    np.testing.assert_array_equal(
        np.asarray(s3["quat_final"]), np.asarray(full["quat_final"]))
    np.testing.assert_array_equal(
        np.asarray(s3["trans_final"]), np.asarray(full["trans_final"]))


def test_resume_with_zero_steps_is_identity():
    """Resuming with n_steps=0 returns the input state."""
    init_quat, init_trans, init_chi = _make_chain()
    K = init_chi.shape[0]
    weights = dof_move_weights(K)
    out = scan_flexible_metropolis(
        jax.random.PRNGKey(7), init_quat, init_trans, init_chi,
        _E, BETA, SIGMA_ROT, SIGMA_TRANS, SIGMA_CHI,
        AXIS_MASK, weights, 0, start_step=0,
    )
    np.testing.assert_array_equal(
        np.asarray(out["quat_final"]), np.asarray(init_quat))
    np.testing.assert_array_equal(
        np.asarray(out["chi_final"]), np.asarray(init_chi))
    assert int(out["try_counts"].sum()) == 0
