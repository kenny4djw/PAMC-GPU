"""Shared fixtures for PTMC-GPU tests.

JAX is forced to CPU for deterministic, GPU-free testing.
"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import pytest

from ptmc.model.structures import (
    Atoms, Pose, DiscreteSurface, ContinuumSurface,
)
from ptmc.config import SurfaceConfig, MCConfig, SimConfig


# ---------------------------------------------------------------------------
# Helper: assert arrays close with configurable tolerance
# ---------------------------------------------------------------------------
def assert_close(actual, expected, rtol=1e-5, atol=1e-8):
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# Minimal Atoms fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def three_atom_atoms():
    """Three atoms in a triangle (body frame), realistic-ish LJ/charge."""
    pos0 = np.array([[0.0, 0.0, 0.0],
                      [0.15, 0.0, 0.0],
                      [0.0, 0.15, 0.0]], dtype=np.float64)
    q = np.array([0.0, -0.5, 0.5], dtype=np.float64)
    c6 = np.array([1e-3, 2e-3, 2e-3], dtype=np.float64)
    c12 = np.array([1e-6, 2e-6, 2e-6], dtype=np.float64)
    return Atoms(pos0=pos0, q=q, c6=c6, c12=c12,
                 names=["C", "O", "N"],
                 resids=np.array([0, 0, 1]),
                 resnames=["ALA", "ALA", "GLY"],
                 elements=["C", "O", "N"])


@pytest.fixture
def single_atom():
    """Single atom at origin (simplest case)."""
    return Atoms(pos0=np.array([[0.0, 0.0, 0.0]]),
                 q=np.array([0.0]),
                 c6=np.array([1e-3]),
                 c12=np.array([1e-6]),
                 names=["X"], resids=np.array([0]),
                 resnames=["BEA"], elements=["C"])


@pytest.fixture
def diatomic():
    """Two atoms along z."""
    return Atoms(pos0=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.15]]),
                 q=np.array([0.3, -0.3]),
                 c6=np.array([1e-3, 2e-3]),
                 c12=np.array([1e-6, 2e-6]),
                 names=["A", "B"], resids=np.array([0, 0]),
                 resnames=["BEA", "BEA"], elements=["C", "C"])


# ---------------------------------------------------------------------------
# Surface fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def simple_discrete_surface():
    """A minimal 2-site discrete surface."""
    return DiscreteSurface(
        pos=np.array([[0.0, 0.0, 0.0], [0.2, 0.2, 0.0]]),
        q=np.array([0.0, 0.0]),
        c6=np.array([1e-3, 1e-3]),
        c12=np.array([1e-6, 1e-6]),
        lambda_D=0.785,
        z_min=0.15,
    )


@pytest.fixture
def simple_continuum_surface():
    """Default homogeneous continuum surface."""
    return ContinuumSurface(
        rho_s=30.0, c6_surf=1.0, c12_surf=1.0,
        lambda_D=0.785, z_min=0.15, psi0=0.0,
    )


@pytest.fixture
def charged_continuum_surface():
    """Continuum surface with non-zero potential."""
    return ContinuumSurface(
        rho_s=30.0, c6_surf=1.0, c12_surf=1.0,
        lambda_D=0.785, z_min=0.15, psi0=5.0,
    )


# ---------------------------------------------------------------------------
# Pose fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def identity_pose():
    return Pose.identity()


@pytest.fixture
def shifted_pose():
    return Pose(quat=np.array([1.0, 0.0, 0.0, 0.0]),
                trans=np.array([0.0, 0.0, 0.5]))


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def default_sim_config():
    return SimConfig(
        surface=SurfaceConfig(),
        mc=MCConfig(n_chains=4, n_steps=100, seed=42),
        pdb_path="", top_path="",
    )


@pytest.fixture
def rng():
    """Convenience: NumPy default RNG with fixed seed."""
    return np.random.default_rng(42)
