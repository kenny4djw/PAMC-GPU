"""Convert a DiscreteSurface into a continuous Fourier-mode charge-density field.

Physics
-------
For a 2D-periodic array of point charges {q_s, r_s = (x_s, y_s, z_s)} with
unit cell (Lx, Ly), the surface charge density projected onto the z=0 plane:

    sigma(x,y) = sum_s q_s * delta(x - x_s) * delta(y - y_s)   [periodic in xy]

The 2D Fourier coefficients are:

    sigma_G = (1/A) * sum_s q_s * exp(-i G·r_s) * exp(qG * (z_s - z_ref))

where A = Lx*Ly, G = 2pi*(n/Lx, m/Ly), qG = sqrt(|G|^2 + kappa^2).
Charges closer to the protein (higher z_s) contribute more strongly.

In the linearized Poisson-Boltzmann (Debye-Hueckel) approximation, each
Fourier mode produces a potential at height z:

    phi_G(x,y,z) = sigma_G / (2 * eps_r * eps0 * qG) * exp(i G·r) * exp(-qG * z)

with qG = sqrt(|G|^2 + kappa^2), kappa = 1/lambda_D.

The TOTAL electrostatic potential is the sum over all modes (including G=0):

    phi(x,y,z) = sum_G phi_G(x,y,z)

vdW interactions use the uniform Steele 9-3 potential (surface atom density
is assumed uniform — the Fourier decomposition is for electrostatics only).

Usage
-----
    from ptmc.surface.fourier import surface_to_fourier
    modes = surface_to_fourier(surface, cell=(Lx, Ly), n_max=10)
    # modes is a dict of JAX arrays ready for fourier_energy_jax()
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ptmc.model.structures import DiscreteSurface


@dataclass
class FourierModes:
    """Precomputed Fourier-mode coefficients for a periodic surface charge density.

    Attributes
    ----------
    Gx, Gy : (M,)   reciprocal lattice vectors (nm^-1)
    Gmag  : (M,)    |G| (nm^-1)
    qG    : (M,)    sqrt(|G|^2 + kappa^2) — effective decay constant (nm^-1)
    coeff_re, coeff_im : (M,)   sigma_G / (eps * qG) — potential coefficient
    lambda_D : float    Debye length (nm)
    z_ref  : float      reference plane height (nm) — atoms at z_s, coeff includes exp(-|G|*z_s)
    """
    Gx: np.ndarray
    Gy: np.ndarray
    Gmag: np.ndarray
    qG: np.ndarray
    coeff_re: np.ndarray
    coeff_im: np.ndarray
    lambda_D: float
    z_ref: float
    n_max: int
    n_modes: int

    @property
    def M(self) -> int:
        return self.n_modes


def surface_to_fourier(surface: DiscreteSurface, cell: tuple,
                        lambda_D: float, z_ref: float = 0.0,
                        n_max: int = 10, eps_r: float = 78.5,
                        ) -> FourierModes:
    """Convert a discrete surface to Fourier-mode charge-density coefficients.

    Parameters
    ----------
    surface : DiscreteSurface
        Surface with per-atom positions, charges.
    cell : (Lx, Ly)
        Unit cell dimensions (nm).
    lambda_D : float
        Debye screening length (nm).
    z_ref : float
        Reference z-plane for the 2D charge projection (nm). Atoms at height
        z_s contribute with factor exp(qG * (z_s - z_ref)). Set to the minimum atom z
        or the hard-wall position.
    n_max : int
        Maximum mode index in each direction. Total modes ≈ (2*n_max+1)^2.
    eps_r : float
        Relative permittivity of the solvent (78.5 for water at 25 C).

    Returns
    -------
    FourierModes with precomputed coefficients for energy evaluation.
    """
    Lx, Ly = float(cell[0]), float(cell[1])
    A = Lx * Ly
    kappa = 1.0 / lambda_D

    # Coulomb prefactor: 1 / (4*pi*eps0*eps_r) in kJ/mol * nm / e^2
    from ptmc.config import COULOMB_FACTOR_KJ_NM_PER_E2
    eps_factor = COULOMB_FACTOR_KJ_NM_PER_E2  # = 138.935 kJ nm / (mol e^2)

    pos = surface.pos        # (M, 3)
    q = surface.q            # (M,)

    modes_Gx, modes_Gy, modes_Gmag = [], [], []
    modes_coeff_re, modes_coeff_im = [], []

    for n in range(-n_max, n_max + 1):
        for m in range(-n_max, n_max + 1):
            Gx = 2.0 * np.pi * n / Lx
            Gy = 2.0 * np.pi * m / Ly
            Gmag = np.sqrt(Gx * Gx + Gy * Gy)
            qG = np.sqrt(Gmag * Gmag + kappa * kappa)

            # sigma_G = (1/A) * sum_s q_s * exp(-i G·r_s) * exp(qG * (z_s - z_ref))
            # Charges closer to the protein (higher z_s) contribute more.
            phase = Gx * pos[:, 0] + Gy * pos[:, 1]
            z_factor = np.exp(qG * (pos[:, 2] - z_ref))
            sigma_re = np.sum(q * np.cos(phase) * z_factor) / A
            sigma_im = -np.sum(q * np.sin(phase) * z_factor) / A

            # 2D Fourier transform of Debye-Hueckel Green's function:
            # FT[exp(-kappa*r)/r] = 2*pi / sqrt(G^2 + kappa^2) = 2*pi / qG
            # phi_G = sigma_G * (2*pi/qG) * exp(-qG*z)
            # NOTE: omits 1/eps_r to match grid convention (consistent with build_grids_cutoff).
            prefactor = eps_factor * (2 * np.pi / qG)

            modes_Gx.append(Gx)
            modes_Gy.append(Gy)
            modes_Gmag.append(Gmag)
            modes_coeff_re.append(sigma_re * prefactor)
            modes_coeff_im.append(sigma_im * prefactor)

    return FourierModes(
        Gx=np.array(modes_Gx, dtype=np.float32),
        Gy=np.array(modes_Gy, dtype=np.float32),
        Gmag=np.array(modes_Gmag, dtype=np.float32),
        qG=np.array([np.sqrt(g*g + kappa*kappa) for g in modes_Gmag], dtype=np.float32),
        coeff_re=np.array(modes_coeff_re, dtype=np.float32),
        coeff_im=np.array(modes_coeff_im, dtype=np.float32),
        lambda_D=lambda_D,
        z_ref=z_ref,
        n_max=n_max,
        n_modes=len(modes_Gx),
    )
