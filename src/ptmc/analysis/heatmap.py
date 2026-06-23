"""Orientation free-energy heatmap over the contact-normal direction.

Uses intrinsic, azimuth-invariant coordinates of the body-frame contact normal:
theta = polar angle from body +z, phi = azimuth in the body frame. The 2D
occupancy histogram is converted to a free-energy surface F = -kT ln p (relative).
"""
from __future__ import annotations

import numpy as np


def orientation_free_energy_map(normals: np.ndarray, beta: float,
                                n_theta: int = 18, n_phi: int = 36):
    """Returns (theta_edges, phi_edges, F (n_theta,n_phi)) in kJ/mol, min-shifted.

    Theta bins use equal-area spacing (uniform in cos(theta)) so every bin
    subtends the same solid angle. Phi bins are uniform.
    """
    n = np.asarray(normals, dtype=np.float64)
    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    theta = np.arccos(np.clip(n[:, 2], -1, 1))         # [0,pi]
    phi = np.arctan2(n[:, 1], n[:, 0]) % (2 * np.pi)    # [0,2pi)
    cos_edges = np.linspace(1.0, -1.0, n_theta + 1)
    te = np.arccos(cos_edges)
    pe = np.linspace(0, 2 * np.pi, n_phi + 1)
    H, _, _ = np.histogram2d(theta, phi, bins=[te, pe])
    p = H / H.sum()
    with np.errstate(divide="ignore"):
        F = -(1.0 / beta) * np.log(p)
    finite = np.isfinite(F)
    if finite.any():
        F = F - F[finite].min()
    return te, pe, F
