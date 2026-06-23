"""Data models: protein Atoms, rigid-body Pose, and two surface models.

Units (GROMACS): length nm, energy kJ/mol, charge e, temperature K.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ptmc.config import COULOMB_FACTOR_KJ_NM_PER_E2


@dataclass
class Atoms:
    """Protein atoms in the intrinsic (body) frame.

    SHAPE/UNITS: pos0 (N,3) nm; q (N,) e; c6 (N,) kJ/mol nm^6;
    c12 (N,) kJ/mol nm^12; resids (N,) int; names/resnames/elements len-N lists.
    Geometric pair combination (comb-rule 1/3): C6_ij=sqrt(C6_i C6_j).
    """
    pos0: np.ndarray
    q: np.ndarray
    c6: np.ndarray
    c12: np.ndarray
    names: list
    resids: np.ndarray
    resnames: list
    elements: list

    def __post_init__(self) -> None:
        self.pos0 = np.asarray(self.pos0, dtype=np.float64)
        self.q = np.asarray(self.q, dtype=np.float64)
        self.c6 = np.asarray(self.c6, dtype=np.float64)
        self.c12 = np.asarray(self.c12, dtype=np.float64)
        self.resids = np.asarray(self.resids, dtype=np.int64)
        n = self.pos0.shape[0]
        if self.pos0.shape != (n, 3):
            raise ValueError(f"pos0 shape must be (N, 3), got {self.pos0.shape}")
        for arr in (self.q, self.c6, self.c12, self.resids):
            if arr.shape != (n,):
                raise ValueError(f"expected ({n},), got {arr.shape}")
        if len(self.names) != n:
            raise ValueError(f"names length {len(self.names)} != N {n}")
        if len(self.resnames) != n:
            raise ValueError(f"resnames length {len(self.resnames)} != N {n}")
        if len(self.elements) != n:
            raise ValueError(f"elements length {len(self.elements)} != N {n}")

    @property
    def n(self) -> int:
        return self.pos0.shape[0]

    @property
    def net_charge(self) -> float:
        return float(self.q.sum())

    @property
    def sqrt_c6(self) -> np.ndarray:
        return np.sqrt(self.c6)

    @property
    def sqrt_c12(self) -> np.ndarray:
        return np.sqrt(self.c12)

    @property
    def eps(self) -> np.ndarray:
        """LJ well depth ε = C6²/(4·C12) in kJ/mol (from C6/C12 form).

        Returns 0.0 for dummy/virtual atoms (C6=C12=0) to avoid division by zero.
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            eps = np.where(self.c12 > 0, self.c6 ** 2 / (4.0 * self.c12), 0.0)
        return eps

    @property
    def sigma(self) -> np.ndarray:
        """LJ collision diameter σ = (C12/C6)^{1/6} in nm.

        Returns 0.0 for dummy/virtual atoms (C6=C12=0) to avoid division by zero.
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            sigma = np.where(self.c6 > 0, (self.c12 / self.c6) ** (1.0 / 6.0), 0.0)
        return sigma

    @property
    def sqrt_eps(self) -> np.ndarray:
        """sqrt(ε) for geometric ε combination."""
        return np.sqrt(self.eps)


def quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    """Unit quaternion (w,x,y,z) -> rotation matrix (3,3). Normalizes internally."""
    q = np.asarray(quat, dtype=np.float64)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


@dataclass
class Pose:
    """Rigid-body pose: quaternion (4,) (w,x,y,z) + translation (3,) nm."""
    quat: np.ndarray
    trans: np.ndarray

    def __post_init__(self) -> None:
        self.quat = np.asarray(self.quat, dtype=np.float64)
        self.trans = np.asarray(self.trans, dtype=np.float64)
        if self.quat.shape != (4,):
            raise ValueError(f"quat shape must be (4,), got {self.quat.shape}")
        if self.trans.shape != (3,):
            raise ValueError(f"trans shape must be (3,), got {self.trans.shape}")

    @classmethod
    def identity(cls) -> "Pose":
        return cls(quat=np.array([1.0, 0.0, 0.0, 0.0]), trans=np.zeros(3))

    def apply(self, pos0: np.ndarray) -> np.ndarray:
        """Map intrinsic coords (N,3) -> lab coords (N,3): R @ r0 + t."""
        R = quat_to_matrix(self.quat)
        return pos0 @ R.T + self.trans


@dataclass
class DiscreteSurface:
    """Explicit-atom surface: M sites with q, C6, C12.

    Pairwise LJ + screened Coulomb (Debye-Hueckel). Ground truth for the grid.
    SHAPE/UNITS: pos (M,3) nm; q (M,) e; c6/c12 (M,); lambda_D, z_min nm.
    """
    pos: np.ndarray
    q: np.ndarray
    c6: np.ndarray
    c12: np.ndarray
    lambda_D: float
    z_min: float
    coulomb_factor: float = COULOMB_FACTOR_KJ_NM_PER_E2

    def __post_init__(self) -> None:
        self.pos = np.asarray(self.pos, dtype=np.float64)
        self.q = np.asarray(self.q, dtype=np.float64)
        self.c6 = np.asarray(self.c6, dtype=np.float64)
        self.c12 = np.asarray(self.c12, dtype=np.float64)
        m = self.pos.shape[0]
        if self.pos.shape != (m, 3):
            raise ValueError(f"pos shape must be (M, 3), got {self.pos.shape}")
        for arr in (self.q, self.c6, self.c12):
            if arr.shape != (m,):
                raise ValueError(f"expected ({m},), got {arr.shape}")

    @property
    def m(self) -> int:
        return self.pos.shape[0]


@dataclass
class ContinuumSurface:
    """Homogeneous half-space surface (medium at z<0).

    Two vdW formalisms are supported:

    (Legacy) C6/C12 + geometric combination:
        V_vdw(z) = (πρ_s/45) C12' z⁻⁹ − (πρ_s/6) C6' z⁻³,
    with C6' = √(C6_i·C6_surf), C12' = √(C12_i·C12_surf).

    (Preferred) ε-σ + Lorentz-Berthelot:
        V_vdw(z) = (4πρ_s/45) εₚ σₚ¹² z⁻⁹ − (2πρ_s/3) εₚ σₚ⁶ z⁻³,
    with εₚ = √(ε_i·ε_surf), σₚ = (σ_i + σ_surf)/2.

    Electrostatics (both paths):
        V_elec(z) = q_i · ψ₀ · exp(−z/λ_D)

    Parameters
    ----------
    rho_s : float
        Surface atom number density (nm⁻³).
    c6_surf, c12_surf : float
        Surface LJ C6/C12 coefficients (kJ/mol nm⁶/nm¹²). Used by legacy path.
    eps_surf, sigma_surf : float | None
        Surface LJ ε (kJ/mol), σ (nm). Used by ε-σ LB path when provided.
    lambda_D : float
        Debye screening length (nm).
    z_min : float
        Hard-wall repulsion distance (nm). Default 0.15 (matches SurfaceConfig
        and the continuum/patterned CLI ``--z-min`` default).
    psi0 : float
        Surface electrostatic potential at z=0 (kJ/mol/e).
    """
    rho_s: float
    c6_surf: float = 1.0
    c12_surf: float = 1.0
    lambda_D: float = 0.785
    z_min: float = 0.15
    psi0: float = 0.0
    eps_surf: float | None = None
    sigma_surf: float | None = None

    def __post_init__(self) -> None:
        if (self.eps_surf is None) != (self.sigma_surf is None):
            raise ValueError(
                "eps_surf and sigma_surf must both be None (use C6/C12 path) "
                "or both set (use epsilon-sigma LB path).")


@dataclass
class PatternedContinuumSurface:
    """Half-space surface with chemically patterned terminal groups.

    Steele 9-3 vdW (substrate-mediated, unchanged from ContinuumSurface).
    Electrostatics from a laterally patterned surface charge density:

        sigma(x,y) = sigma_base - sum_i sigma_base * exp(-((x-xi)^2+(y-yi)^2)/(2 wi^2))

    where sigma_base (e/nm^2) is the baseline charge density from fully
    deprotonated -OH groups, and Gaussian holes at (xi,yi) with width wi
    represent neutral -CH3 patches where charge is removed.

    The potential phi(x,y,z) from sigma(x,y) is computed via FFT-based 2D
    convolution of sigma with the LPB Green's function kernel:
        phi(x,y,z) = f * ∫∫ sigma(x',y') * exp(-kappa R)/R dx' dy'
    with R = sqrt((x-x')^2 + (y-y')^2 + z^2), f = COULOMB_FACTOR.

    vdW parameters (rho_s, c6_surf, c12_surf) describe the underlying substrate
    (e.g., gold, silica) and are independent of the organic functionalization.
    """
    rho_s: float
    c6_surf: float
    c12_surf: float
    lambda_D: float
    z_min: float
    sigma_base: float          # baseline charge density (e/nm^2), 100% -OH
    patches: list              # [(x_i, y_i, w_i), ...] in nm, neutral -CH3 islands
    dielectric: float = 78.5  # relative permittivity (default 78.5)


# ---------------------------------------------------------------------------
# 3D Agarose hydrogel model (Voronoi fiber network + dual-potential field).
# ---------------------------------------------------------------------------

@dataclass
class AgaroseGel:
    """3D agarose hydrogel with Gaussian soft-core + Yukawa screened Coulomb.

    The gel is represented as a Poisson-Voronoi fiber network: N_seed seed
    points are Voronoi-tessellated in an extended box (L + margin); fiber
    nodes are densely sampled along the ridge edges that pass through the
    central L^3 region. This produces a true sponge-like topology.

    Two independent 3D fields are precomputed on a regular grid:

        U_steric(x,y,z) = sum_i A * exp(-|r - r_i|^2 / 2 sigma^2)
        phi_elec(x,y,z) = sum_i delta_i * (f/eps_r) * q_i * exp(-|r - r_i|/lamD) / |r - r_i|

    so that for a protein atom at position r with charge q_atom:

        E(r) = U_steric(r) + q_atom * phi_elec(r)

    Parameters
    ----------
    L : float
        Simulation box side length (nm). Should match pore diameter.
    n_seeds : int
        Number of Voronoi seed points (~polymer segment count).
    sigma : float
        Gaussian soft-core radius (nm). SAXS: 1.5-2.5 nm for 4-6% agarose.
    A : float
        Steric repulsion amplitude (kJ/mol). Must be >> k_BT (418-2092).
    doping_frac : float
        Fraction of fiber nodes carrying charge (dimensionless).
    q_ligand : float
        Charge per doped node (e). +1 for Q-Sepharose, -1 for SP-Sepharose.
    lambda_D : float
        Debye screening length (nm). 3.0 nm at 10 mM, 0.4 nm at 500 mM.
    dielectric : float
        Solvent relative permittivity (default 78.4 for water, 298 K).
    seed : int
        Master seed for Voronoi + doping (default 42).
    """
    L: float
    n_seeds: int
    sigma: float
    A: float
    doping_frac: float
    q_ligand: float
    lambda_D: float
    dielectric: float = 78.5
    seed: int = 42


@dataclass
class AgaroseSurface:
    """Flat agarose gel coating (fibers at z<0, protein at z>0).

    Same physics as AgaroseGel (Gaussian soft-core + Yukawa), but the fiber
    network is restricted to the half-space z<0.  The protein samples the
    z>0 region with a hard wall at z_min.  Translation is z-only by default.

    Parameters
    ----------
    L : float
        Lateral box side length (nm).  Fibers are periodic in xy.
    thickness : float
        Coating thickness (nm).  Fibers occupy z ∈ [-thickness, 0].
    n_seeds : int
        Number of Voronoi seeds in the coating slab.
    sigma : float
        Gaussian soft-core radius (nm).
    A : float
        Steric repulsion amplitude (kJ/mol).
    doping_frac : float
        Fraction of fiber nodes carrying charge.
    q_ligand : float
        Charge per doped node (e).
    lambda_D : float
        Debye screening length (nm).
    z_min : float
        Hard-wall distance from surface (nm).
    dielectric : float
        Solvent relative permittivity (default 78.5).
    seed : int
        Master seed for Voronoi + doping (default 42).
    """
    L: float
    thickness: float
    n_seeds: int
    sigma: float
    A: float
    doping_frac: float
    q_ligand: float
    lambda_D: float
    z_min: float = 0.2
    dielectric: float = 78.5
    seed: int = 42
