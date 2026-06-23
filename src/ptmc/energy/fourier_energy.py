"""JAX energy for Fourier-mode continuous surface (Steele 9-3 vdW + Fourier electrostatics).

The electrostatic potential at protein atom position (x, y, z) is:

    phi(x,y,z) = sum_G [coeff_re(G)*cos(G·r) - coeff_im(G)*sin(G·r)] * exp(-qG * z)

where coeff_{re,im} are the precomputed Fourier potential coefficients and
qG = sqrt(|G|^2 + kappa^2) is the mode decay constant.

vdW: uniform Steele 9-3 (same as steele_energy_jax).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from ptmc.mc.moves import quat_rotate


@jax.jit
def fourier_energy_jax(quat, trans, pos0, q, c6p, c12p, cA, cB, z_min,
                        Gx, Gy, qG_arr, coeff_re, coeff_im, z_ref=0.0):
    """Compute energy for ONE protein pose on a Fourier-mode continuous surface.

    Parameters
    ----------
    quat : (4,)       rotation quaternion
    trans : (3,)      translation vector (nm)
    pos0 : (N,3)      protein atom positions at origin (nm)
    q : (N,)          protein atom charges (e)
    c6p, c12p : (N,)  protein LJ C6/C12 (kJ/mol nm^6, nm^12)
    cA, cB : scalar   Steele 9-3 prefactors cA = πρ/45, cB = πρ/6 (legacy
                      C6/C12 path) — see ``steele_coefficients``. Per-atom
                      arrays are NOT supported; broadcast against the
                      per-atom c6p/c12p arrays gives the same result anyway.
    z_min : float     hard-wall z position (nm)
    Gx, Gy : (M,)     reciprocal lattice vectors (nm^-1)
    qG_arr : (M,)     sqrt(|G|^2 + 1/lamD^2) (nm^-1)
    coeff_re, coeff_im : (M,)  Fourier potential coefficients (kJ/mol/e)
    z_ref : float     reference z-plane for Fourier expansion (nm)

    Returns
    -------
    energy : float    total energy (kJ/mol), inf if any atom below z_min
    """
    pos = quat_rotate(quat, pos0) + trans  # (N, 3)
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]

    # --- hard wall ------------------------------------------------
    below = jnp.any(z < z_min)

    # --- vdW: Steele 9-3 (uniform surface) ------------------------
    # The Steele 9-3 derivation assumes the surface lies at z=0; z is the
    # lab-frame distance to that plane.  Use the same convention as
    # reference.energy_positions and steele_energy_jax (NOT z-z_min) so this
    # path stays consistent with the discrete-surface ground truth.
    zc = jnp.maximum(z, z_min + 1e-3)  # avoid 1/0 inside hard wall
    vdw = jnp.sum(cA * c12p / zc ** 9 - cB * c6p / zc ** 3)

    # --- electrostatic: Fourier mode sum --------------------------
    # phi_G(x,y,z) = coeff_re*cos(G·r)*exp(-qG*(z-z_ref)) - coeff_im*sin(G·r)*exp(-qG*(z-z_ref))
    dz = z[:, None] - z_ref                                           # (N, 1)
    GdotR = Gx[None, :] * x[:, None] + Gy[None, :] * y[:, None]       # (N, M)
    cos_term = jnp.cos(GdotR)
    sin_term = jnp.sin(GdotR)
    decay = jnp.exp(-qG_arr[None, :] * dz)                            # (N, M)
    phi = jnp.sum((coeff_re[None, :] * cos_term - coeff_im[None, :] * sin_term) * decay, axis=1)
    e_el = jnp.dot(q, phi)

    return jnp.where(below, jnp.inf, vdw + e_el)


# ---------------------------------------------------------------------------
# Batched (vmapped) version for MC chains
# ---------------------------------------------------------------------------
_fourier_vmapped = jax.vmap(
    fourier_energy_jax,
    in_axes=(0, 0, None, None, None, None, None, None,
             None, None, None, None, None, None))
