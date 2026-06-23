"""Build the surface field grids G12, G6, phi for a DiscreteSurface.

vdW is factorized so it needs only TWO grids:
    G12(r) = sum_s sqrt(C12_s) / |r - r_s|^12
    G6(r)  = sum_s sqrt(C6_s)  / |r - r_s|^6
    phi(r) = sum_s f q_s exp(-|r - r_s|/lambda_D) / |r - r_s|
so that for protein atom i:  E_i = sqrt(C12_i) G12 - sqrt(C6_i) G6 + q_i phi
reproduces the pairwise direct sum. Grids start at z = z_min (hard wall).

Chunked computation: grid points are generated per-chunk from (i,j,k) index
ranges to avoid materializing the full (nx,ny,nz,3) coordinate array. Peak
memory is bounded by chunk_size × surface_atoms × 3 floats.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ptmc.model.structures import DiscreteSurface
from ptmc.config import GRID_BUILD_CHUNK_ELEMS


@dataclass
class FieldGrids:
    """Three scalar fields on a regular 3D lattice.

    SHAPE: G12/G6/phi each (nx,ny,nz); origin (3,) nm; spacing (3,) nm.
    Lattice point (i,j,k) sits at origin + (i,j,k)*spacing.
    """
    G12: np.ndarray
    G6: np.ndarray
    phi: np.ndarray
    origin: np.ndarray
    spacing: np.ndarray
    z_min: float

    @property
    def shape(self):
        return self.G12.shape

    def upper(self):
        n = np.array(self.shape) - 1
        return self.origin + n * self.spacing


def _axis(lo, hi, step):
    n = int(np.floor((hi - lo) / step + 1e-9)) + 1
    return lo + step * np.arange(n)


def _chunk_coordinates(xs, ys, zs, s, e):
    """Generate grid coordinates for flat-index slice [s, e) without full
    meshgrid materialization. Returns (chunk_size, 3) array."""
    ny, nz = ys.size, zs.size
    idx = np.arange(s, e)
    i = idx // (ny * nz)
    j = (idx // nz) % ny
    k = idx % nz
    return np.column_stack([xs[i], ys[j], zs[k]])


def build_grids(surface: DiscreteSurface, x_range, y_range, z_range,
                spacing, cap_g12=None) -> FieldGrids:
    """Precompute G12, G6, phi on a regular lattice above the surface.

    z_range[0] is clamped to surface.z_min. cap_g12 optionally caps G12 to avoid
    float overflow near a surface atom (None disables).
    """
    if np.isscalar(spacing):
        dx = dy = dz = float(spacing)
    else:
        dx, dy, dz = (float(s) for s in spacing)

    z_lo = max(z_range[0], surface.z_min)
    xs = _axis(x_range[0], x_range[1], dx)
    ys = _axis(y_range[0], y_range[1], dy)
    zs = _axis(z_lo, z_range[1], dz)

    nx, ny, nz = xs.size, ys.size, zs.size
    P = nx * ny * nz

    sqrt_c12_s = np.sqrt(surface.c12)
    sqrt_c6_s = np.sqrt(surface.c6)
    surf_pos = surface.pos
    surf_q = surface.q
    coulomb_factor = surface.coulomb_factor
    lambda_D = surface.lambda_D

    G12 = np.empty(P, dtype=np.float64)
    G6 = np.empty(P, dtype=np.float64)
    phi = np.empty(P, dtype=np.float64)

    # Chunk by grid points: each chunk generates its own coords from indices.
    chunk = max(1, int(GRID_BUILD_CHUNK_ELEMS // max(surface.m, 1)))
    for s in range(0, P, chunk):
        e = min(s + chunk, P)
        pts = _chunk_coordinates(xs, ys, zs, s, e)
        d = pts[:, None, :] - surf_pos[None, :, :]
        r = np.sqrt(np.sum(d * d, axis=-1))
        r = np.maximum(r, 1e-9)
        inv_r6 = r ** -6
        G12[s:e] = (sqrt_c12_s[None, :] * inv_r6 * inv_r6).sum(axis=1)
        G6[s:e] = (sqrt_c6_s[None, :] * inv_r6).sum(axis=1)
        phi[s:e] = (coulomb_factor * surf_q[None, :]
                    * np.exp(-r / lambda_D) / r).sum(axis=1)

    if cap_g12 is not None:
        G12 = np.minimum(G12, cap_g12)

    shape = (nx, ny, nz)
    return FieldGrids(
        G12=G12.reshape(shape), G6=G6.reshape(shape), phi=phi.reshape(shape),
        origin=np.array([xs[0], ys[0], zs[0]]),
        spacing=np.array([dx, dy, dz]), z_min=surface.z_min,
    )


# ---------------------------------------------------------------------------
# Patterned continuum surface: phi(x,y,z) from 2D charge density sigma(x,y).
# ---------------------------------------------------------------------------

def build_sigma_grid(xs, ys, sigma_base, patches):
    """Build charge density sigma(x,y) on a regular grid.

    Parameters
    ----------
    xs, ys : (nx,), (ny,)  coordinate arrays (nm)
    sigma_base : float  baseline charge density (e/nm^2)
    patches : list of (xi, yi, wi)  Gaussian hole positions and widths (nm)

    Returns
    -------
    sigma : (nx, ny) ndarray  sigma(x,y) in e/nm^2
    """
    nx, ny = len(xs), len(ys)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    sigma = np.full((nx, ny), sigma_base)
    for xi, yi, wi in patches:
        sigma -= sigma_base * np.exp(
            -((XX - xi) ** 2 + (YY - yi) ** 2) / (2.0 * wi ** 2)
        )
    return sigma


def compute_phi_grid(xs, ys, zs, sigma_grid, lambda_D, dielectric=78.5,
                     half_space=True):
    """Compute phi(x,y,z) from sigma(x,y) via FFT-based 2D convolution.

    phi(x,y,z) = s · f · ∫∫ sigma(x',y') · exp(-kappa R)/R dx' dy'
    where R = sqrt((x-x')^2 + (y-y')^2 + z^2), f = COULOMB_FACTOR/ε_r and
    s is the boundary factor (see ``half_space`` below).

    The LPB Green's function kernel exp(-kappa*R)/R is convolved with sigma
    at each z-slice using zero-padded FFT (2x padding to avoid wraparound).

    Half-space vs. free-space convention (IMPORTANT)
    ------------------------------------------------
    Bare superposition of screened point sources over the charge plane radiates
    symmetrically into BOTH half-spaces, giving, for a uniform sheet,
        φ(z) = 2π f σ λ_D / ε_r · e^{-z/λ_D}    (free-space, s = 1).
    A real charged *surface* confines the electrolyte (and hence all the field)
    to z>0. The linearized-PB boundary condition −ε₀ε_r ∂φ/∂z|₀ = σ then yields
        φ(z) = 4π f σ λ_D / ε_r · e^{-z/λ_D}    (half-space, s = 2),
    which is exactly the ``ContinuumSurface`` convention encoded by
    :func:`ptmc.energy.reference.psi0_from_sigma_q`. ``half_space=True``
    (default, s = 2) makes a uniform ``sigma_grid`` reproduce that ψ₀, so the
    ``patterned`` surface is the lateral-heterogeneity generalisation of
    ``continuum`` with a CONSISTENT magnitude. Set ``half_space=False`` to keep
    the legacy free-space (s = 1) field, e.g. for charges embedded in bulk
    electrolyte with no boundary.

    Parameters
    ----------
    xs, ys, zs : (nx,), (ny,), (nz,) coordinate arrays (nm)
    sigma_grid : (nx, ny) ndarray  sigma(x,y) (e/nm^2)
    lambda_D : float  Debye screening length (nm)
    dielectric : float  water relative permittivity (default 78.5)
    half_space : bool  apply the s = 2 surface boundary factor (default True),
        making a uniform sheet match ``psi0_from_sigma_q``. False = legacy
        free-space superposition (s = 1, half the magnitude).

    Returns
    -------
    phi_grid : (nx, ny, nz) ndarray  phi(x,y,z) in kJ/(mol*e)
    origin : (3,)  grid origin = [xs[0], ys[0], zs[0]]
    spacing : (3,)  [dx, dy, dz]
    """
    from ptmc.config import COULOMB_FACTOR_KJ_NM_PER_E2

    nx, ny = len(xs), len(ys)
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    kappa = 1.0 / lambda_D
    coulomb = COULOMB_FACTOR_KJ_NM_PER_E2 / dielectric

    # Zero-pad sigma to 2x for wraparound-free linear convolution
    pad_x, pad_y = 2 * nx, 2 * ny
    sigma_pad = np.zeros((pad_x, pad_y))
    sigma_pad[:nx, :ny] = sigma_grid
    sigma_fft = np.fft.rfft2(sigma_pad)

    # Kernel coordinate grid (centered, then ifftshifted to FFT wrap order)
    kx_c = (np.arange(pad_x) - pad_x // 2) * dx
    ky_c = (np.arange(pad_y) - pad_y // 2) * dy
    KX, KY = np.meshgrid(kx_c, ky_c, indexing="ij")
    rho = np.sqrt(KX ** 2 + KY ** 2)

    # Batched FFT across z-slices: build the kernel as a (nz, pad_x, pad_y)
    # stack and let numpy fuse the per-slice rfft2 / multiply / irfft2 into
    # three calls instead of 3*nz. Memory cost ≤ nz * pad_x * pad_y * 8 B
    # (typical 60 nm x 60 nm x 30 z = 14 MB) — well within budget.
    zs_arr = np.asarray(zs)[:, None, None]                          # (nz, 1, 1)
    R = np.sqrt(rho[None, :, :] ** 2 + zs_arr ** 2)                 # (nz, pad_x, pad_y)
    kernel_centered = np.where(R > 1e-12, np.exp(-kappa * R) / R, 0.0)
    kernel = np.fft.ifftshift(kernel_centered, axes=(-2, -1))
    kernel_fft = np.fft.rfft2(kernel, axes=(-2, -1))                # (nz, pad_x, pad_y//2+1)
    phi_fft = sigma_fft[None, :, :] * kernel_fft
    phi_pad = np.fft.irfft2(phi_fft, s=(pad_x, pad_y), axes=(-2, -1))
    # Transpose (nz, nx, ny) → (nx, ny, nz) and apply dA, COULOMB prefactor.
    phi = (phi_pad[:, :nx, :ny] * (dx * dy)).transpose(1, 2, 0)

    dz = zs[1] - zs[0] if len(zs) > 1 else 0.1
    # s = 2 confines the field to the electrolyte half-space (z>0), matching the
    # ContinuumSurface / psi0_from_sigma_q convention; s = 1 = free-space sheet.
    surface_factor = 2.0 if half_space else 1.0
    phi *= coulomb * surface_factor
    return phi, np.array([xs[0], ys[0], zs[0]]), np.array([dx, dy, dz])
