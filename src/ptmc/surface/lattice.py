"""Crystal lattice definitions and operations.

Unit convention: length nm (GROMACS), angle degrees (conventional).
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np


@dataclass
class Lattice:
    """Crystal unit cell with atoms in fractional coordinates."""
    cell: np.ndarray    # (3,3) column vectors in nm
    species: list[str]
    frac_pos: np.ndarray  # (M,3)

    def __post_init__(self):
        self.cell = np.asarray(self.cell, dtype=np.float64)
        assert self.cell.shape == (3, 3)
        self.frac_pos = np.asarray(self.frac_pos, dtype=np.float64)
        assert self.frac_pos.shape[1] == 3
        assert len(self.species) == self.frac_pos.shape[0]

    @property
    def m(self) -> int:
        return self.frac_pos.shape[0]

    @property
    def volume(self) -> float:
        return float(np.linalg.det(self.cell))

    def frac_to_cart(self, frac: np.ndarray) -> np.ndarray:
        return np.asarray(frac) @ self.cell.T

    def cart_to_frac(self, cart: np.ndarray) -> np.ndarray:
        return np.asarray(cart) @ np.linalg.inv(self.cell).T

    def replicate(self, nx: int, ny: int, nz: int) -> "Lattice":
        i = np.arange(nx); j = np.arange(ny); k = np.arange(nz)
        I, J, K = np.meshgrid(i, j, k, indexing="ij")
        offsets = np.stack([I.ravel(), J.ravel(), K.ravel()], axis=-1).astype(np.float64)
        new_frac = (self.frac_pos[None, :, :] + offsets[:, None, :]).reshape(-1, 3)
        cell = self.cell.copy()
        cell[:, 0] *= nx; cell[:, 1] *= ny; cell[:, 2] *= nz
        return Lattice(cell=cell, species=self.species * (nx * ny * nz), frac_pos=new_frac)

    def _as_array(self):
        return self.frac_pos


def cell_from_params(a, b, c, alpha, beta, gamma) -> np.ndarray:
    a_r = math.radians(alpha); b_r = math.radians(beta); g_r = math.radians(gamma)
    ax = a; ay = az = 0.0
    bx = b * math.cos(g_r); by = b * math.sin(g_r); bz = 0.0
    cx = c * math.cos(b_r)
    cy = c * (math.cos(a_r) - math.cos(b_r) * math.cos(g_r)) / math.sin(g_r)
    cz = math.sqrt(c**2 - cx**2 - cy**2)
    return np.array([[ax, bx, cx], [ay, by, cy], [az, bz, cz]], dtype=np.float64)


def alpha_quartz() -> Lattice:
    """α-quartz SiO₂ (trigonal, P3₂21). 9 atoms / cell."""
    a, c = 0.4913, 0.5405
    cell = cell_from_params(a, a, c, 90.0, 90.0, 120.0)
    species = ["Si", "Si", "Si", "Ob", "Ob", "Ob", "Ob", "Ob", "Ob"]
    frac = np.array([
        [0.4697, 0.0000, 0.6667], [1.0000, 0.4697, 0.1667],
        [0.5303, 0.5303, 0.0000], [0.4153, 0.2720, 0.2141],
        [0.2720, 0.4153, 0.5474], [0.7280, 0.9573, 0.5474],
        [0.9573, 0.7280, 0.2141], [0.5847, 0.0427, 0.8807],
        [0.0427, 0.5847, 0.8807],
    ], dtype=np.float64) % 1.0
    return Lattice(cell=cell.T, species=species, frac_pos=frac)


def rutile_tio2() -> Lattice:
    """Rutile TiO₂ (tetragonal, P4₂/mnm). 6 atoms / cell."""
    a, c = 0.4593, 0.2959
    cell = cell_from_params(a, a, c, 90.0, 90.0, 90.0)
    species = ["Ti", "Ti", "Ob_ti", "Ob_ti", "Ob_ti", "Ob_ti"]
    frac = np.array([
        [0.0, 0.0, 0.0], [0.5, 0.5, 0.5],
        [0.305, 0.305, 0.0], [0.695, 0.695, 0.0],
        [0.805, 0.195, 0.5], [0.195, 0.805, 0.5],
    ], dtype=np.float64)
    return Lattice(cell=cell.T, species=species, frac_pos=frac)


def anatase_tio2() -> Lattice:
    """Anatase TiO₂ (tetragonal, I4₁/amd). 12 atoms / cell."""
    a, c = 0.3784, 0.9515
    cell = cell_from_params(a, a, c, 90.0, 90.0, 90.0)
    frac = np.array([
        [0.0, 0.0, 0.0], [0.0, 0.5, 0.25], [0.5, 0.0, 0.75], [0.5, 0.5, 0.5],
        [0.0, 0.0, 0.207], [0.0, 0.0, 0.793], [0.0, 0.5, 0.457], [0.0, 0.5, 0.043],
        [0.5, 0.0, 0.543], [0.5, 0.0, 0.957], [0.5, 0.5, 0.707], [0.5, 0.5, 0.293],
    ], dtype=np.float64)
    return Lattice(cell=cell.T, species=["Ti"]*4 + ["Ob_ti"]*8, frac_pos=frac)


def gold_fcc() -> Lattice:
    """FCC gold (cubic, Fm-3m). 4 atoms / cell."""
    a = 0.40782
    cell = cell_from_params(a, a, a, 90.0, 90.0, 90.0)
    frac = np.array([[0.0,0.0,0.0],[0.5,0.5,0.0],[0.5,0.0,0.5],[0.0,0.5,0.5]], dtype=np.float64)
    return Lattice(cell=cell.T, species=["Au"]*4, frac_pos=frac)


BUILTIN_CRYSTALS: dict = {
    "alpha_quartz": alpha_quartz, "rutile_tio2": rutile_tio2,
    "anatase_tio2": anatase_tio2, "gold_fcc": gold_fcc,
}


def get_crystal(name: str) -> Lattice:
    if name not in BUILTIN_CRYSTALS:
        raise ValueError(f"unknown crystal '{name}'; available: {list(BUILTIN_CRYSTALS)}")
    return BUILTIN_CRYSTALS[name]()
