"""Direct-sum single-pose reference energy (numpy / float64).

Physical ground truth; the stage-2 grid must converge to it.
  * DiscreteSurface: pairwise LJ + screened Coulomb (Debye-Hueckel).
  * ContinuumSurface: analytic Steele 9-3 vdW + linearized-PB surface potential.
Hard wall -> +inf if any atom has z < surface.z_min.

Two formalisms for the Steele 9-3 vdW:
  * (Legacy) C6/C12 + geometric combination:  V = (πρ/45) C12'/z⁹ − (πρ/6) C6'/z⁶
  * (Preferred) ε-σ + Lorentz-Berthelot:      V = (4πρ/45) εₚ σₚ¹²/z⁹ − (2πρ/3) εₚ σₚ⁶/z³
"""
from __future__ import annotations

import numpy as np

from ptmc.model.structures import (
    Atoms, Pose, DiscreteSurface, ContinuumSurface,
)
from ptmc.config import COULOMB_FACTOR_KJ_NM_PER_E2


def lj_pair(r, c6, c12):
    """LJ energy C12/r^12 - C6/r^6 (kJ/mol); broadcasts."""
    r6 = r ** 6
    return c12 / (r6 * r6) - c6 / r6


def screened_coulomb_pair(r, qi, qj, lambda_D, coulomb_factor):
    """Debye-Hueckel f q_i q_j exp(-r/lambda_D)/r (kJ/mol)."""
    return coulomb_factor * qi * qj * np.exp(-r / lambda_D) / r


def combine_geometric(ci, cj):
    """Geometric combination sqrt(ci cj) (comb-rule 1/3)."""
    return np.sqrt(ci * cj)


def energy_positions(pos: np.ndarray, atoms: Atoms, surface) -> float:
    """Energy (kJ/mol) of protein atoms at lab positions pos (N,3) nm."""
    if np.any(pos[:, 2] < surface.z_min):
        return np.inf

    if isinstance(surface, DiscreteSurface):
        d = pos[:, None, :] - surface.pos[None, :, :]
        r = np.maximum(np.sqrt(np.sum(d * d, axis=-1)), 1e-9)
        c6_ij = combine_geometric(atoms.c6[:, None], surface.c6[None, :])
        c12_ij = combine_geometric(atoms.c12[:, None], surface.c12[None, :])
        e_lj = lj_pair(r, c6_ij, c12_ij)
        e_el = screened_coulomb_pair(
            r, atoms.q[:, None], surface.q[None, :],
            surface.lambda_D, surface.coulomb_factor)
        return float(np.sum(e_lj + e_el))

    if isinstance(surface, ContinuumSurface):
        z = pos[:, 2]
        c6p = combine_geometric(atoms.c6, surface.c6_surf)
        c12p = combine_geometric(atoms.c12, surface.c12_surf)
        e_vdw = ((np.pi * surface.rho_s / 45.0) * c12p / z ** 9
                 - (np.pi * surface.rho_s / 6.0) * c6p / z ** 3)
        e_el = atoms.q * surface.psi0 * np.exp(-z / surface.lambda_D)
        return float(np.sum(e_vdw + e_el))

    raise TypeError(f"unknown surface type: {type(surface)!r}")


def energy_direct(pose: Pose, atoms: Atoms, surface) -> float:
    """Single-pose reference energy (kJ/mol): apply pose, then sum."""
    return energy_positions(pose.apply(atoms.pos0), atoms, surface)


# ---------------------------------------------------------------------------
# Conversion: C6/C12 ↔ ε/σ  (Lennard-Jones parameter sets)
# ---------------------------------------------------------------------------
def c6c12_to_eps_sigma(c6, c12):
    """Convert C6/C12 (kJ/mol·nm⁶, kJ/mol·nm¹²) → ε (kJ/mol), σ (nm).

    LJ 12-6 form equivalence:
        U(r) = C12/r¹² − C6/r⁶ = 4ε[(σ/r)¹² − (σ/r)⁶]
    ⇒  σ = (C12/C6)^{1/6},   ε = C6² / (4·C12).

    Atoms with C6=C12=0 (dummy/virtual sites) safely return (0.0, 0.0).
    """
    c6 = np.asarray(c6, dtype=np.float64)
    c12 = np.asarray(c12, dtype=np.float64)
    zero = (c6 == 0.0) | (c12 == 0.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        sigma = np.where(zero, 0.0, (c12 / c6) ** (1.0 / 6.0))
        eps = np.where(zero, 0.0, c6 ** 2 / (4.0 * c12))
    return eps, sigma


def eps_sigma_to_c6c12(eps, sigma):
    """Convert ε (kJ/mol), σ (nm) → C6 (kJ/mol·nm⁶), C12 (kJ/mol·nm¹²).

    C6 = 4εσ⁶,  C12 = 4εσ¹².
    """
    c6 = 4.0 * eps * sigma ** 6
    c12 = 4.0 * eps * sigma ** 12
    return c6, c12


# ---------------------------------------------------------------------------
# Lorentz-Berthelot combining rules (ε only: geometric; σ: arithmetic).
# ---------------------------------------------------------------------------
def lb_eps_sigma(eps_i, sigma_i, eps_j, sigma_j):
    """Lorentz-Berthelot combination: σ arithmetic mean, ε geometric mean.

    σ_{ij} = (σ_i + σ_j) / 2
    ε_{ij} = sqrt(ε_i · ε_j)
    """
    sigma_ij = (sigma_i + sigma_j) / 2.0
    eps_ij = np.sqrt(eps_i * eps_j)
    return eps_ij, sigma_ij


def lb_c6c12(c6_i, c12_i, c6_j, c12_j):
    """Lorentz-Berthelot combination in the C6/C12 representation.

    Converts → ε/σ, applies LB, converts back.
    """
    e_i, s_i = c6c12_to_eps_sigma(c6_i, c12_i)
    e_j, s_j = c6c12_to_eps_sigma(c6_j, c12_j)
    e_ij, s_ij = lb_eps_sigma(e_i, s_i, e_j, s_j)
    return eps_sigma_to_c6c12(e_ij, s_ij)


# ---------------------------------------------------------------------------
# ψ₀: surface electrostatic potential from surface charge density.
# ---------------------------------------------------------------------------
def psi0_from_sigma_q(sigma_q_e_nm2, lambda_D_nm, epsilon_r=78.5,
                       coulomb_factor=COULOMB_FACTOR_KJ_NM_PER_E2):
    """Surface potential ψ₀ [kJ/(mol·e)] from charge density σ_q [e/nm²].

    Solves the linearized Poisson-Boltzmann equation for a charged plane
    (electrolyte in z>0, surface at z=0):

        ∇²φ = κ²φ  (z>0),  BC:  -ε₀ε_r ∂φ/∂z|_{z=0⁺} = σ_q

    ⇒  φ(z) = ψ₀ · e^{-κz},   ψ₀ = σ_q / (ε₀ ε_r κ)

    Using the code-unit identity 1/ε₀ = 4πf (since f = 1/(4πε₀)):

        ψ₀  =  4π · f · σ_q / (ε_r κ)  =  4π · f · σ_q · λ_D / ε_r       [kJ/(mol·e)]

    Linear PB is only valid for |eψ₀| ≪ kT (~25 mV at 25 °C). For mineral
    oxide / silica surfaces with σ_q ≳ 0.05 C/m² (≈ 0.3 e/nm²) the linear
    result overshoots and unphysically pins large net-charged proteins flat
    against the surface. Use :func:`psi0_from_sigma_q_full_gc` for those
    regimes — it reduces to this function exactly when the linearization
    holds, so the GC form is a strictly safer default.

    Parameters
    ----------
    sigma_q_e_nm2 : float
        Surface charge density (e/nm²).
    lambda_D_nm : float
        Debye screening length (nm).
    epsilon_r : float
        Solvent relative permittivity (default 78.5 for water at 25°C).
    coulomb_factor : float
        Coulomb constant f = 1/(4πε₀) = 138.935458 kJ·nm·mol⁻¹·e⁻².

    Examples
    --------
    >>> psi0_from_sigma_q(0.6242, 0.785, epsilon_r=78.5)
    10.895  # kJ/(mol·e)
    """
    return (4.0 * np.pi * float(coulomb_factor) * float(sigma_q_e_nm2)
            * float(lambda_D_nm) / float(epsilon_r))


def psi0_from_sigma_q_full_gc(sigma_q_e_nm2, ionic_strength_M,
                                epsilon_r=78.5, temperature_K=298.15):
    """Surface potential ψ₀ [kJ/(mol·e)] from full nonlinear Gouy-Chapman.

    Solves the nonlinear Poisson-Boltzmann equation for a 1:1 electrolyte
    bounded by a uniformly charged plane (Grahame equation):

        σ_q = sqrt(8 ε₀ ε_r k_B T n₀) · sinh( e ψ₀ / 2 k_B T )

    ⇒   ψ₀ = (2 k_B T / e) · asinh( σ_q / sqrt(8 ε₀ ε_r k_B T n₀) )

    where n₀ = c · N_A is the bulk number density (per m³) of each ion type.

    Why we need this: the linearized PB result ``psi0_from_sigma_q`` is only
    correct for |ψ₀| < k_B T / e ≈ 25 mV. Common literature surfaces
    (rutile TiO₂ ~ −0.1 C/m², α-quartz ~ −0.2 C/m²) yield linear ψ₀ in the
    100–300 mV range, which overshoots the true GC value by 1.5–3× and pins
    large net-charged proteins (e.g. lysozyme +8e) excessively flat against
    the surface. ``asinh`` saturates at high |σ_q|, recovering the physical
    behaviour. At low |σ_q| the two functions agree to leading order, so
    switching code over to full GC introduces no error in the linear regime.

    Note on the decay form: the energy kernel multiplies ψ₀ by ``exp(-z/λ_D)``
    (linearized Debye-Hückel envelope). In a strict full-GC treatment the
    envelope is ``4 atanh[ tanh(eψ₀/4kT) · exp(-κz) ]``. Using GC-ψ₀ with the
    linear envelope is the standard "renormalised surface potential" approach
    (Hiemenz & Rajagopalan §11.5, Israelachvili Eq. 14.42): correct in
    magnitude at the surface and at the asymptotic far-field limit, slightly
    underestimates the field at z ≲ λ_D. This is the dominant correction —
    keeping the linear envelope avoids a JAX-side branch in the hot loop.

    Parameters
    ----------
    sigma_q_e_nm2 : float
        Surface charge density (e/nm²).
    ionic_strength_M : float
        1:1 electrolyte ionic strength (mol/L). Used to set n₀ = c·N_A.
    epsilon_r : float
        Solvent relative permittivity (default 78.5 for water at 25°C).
    temperature_K : float
        Temperature (K, default 298.15).

    Examples
    --------
    >>> # Rutile at pH 7: σ ≈ -0.1 C/m² = -0.624 e/nm², I = 0.1 M, ε_r = 78.5
    >>> import math
    >>> psi0_lin = psi0_from_sigma_q(-0.624, 0.961, epsilon_r=78.5)        # noqa
    >>> psi0_gc  = psi0_from_sigma_q_full_gc(-0.624, 0.1, epsilon_r=78.5)  # noqa
    >>> # |ψ₀_lin| ≈ 13.3 kJ/(mol·e) = 138 mV; |ψ₀_GC| ≈ 8.5 kJ/(mol·e) = 88 mV
    """
    import math
    # Physical constants (SI)
    eps0 = 8.8541878128e-12   # F/m
    kB   = 1.380649e-23       # J/K
    e_C  = 1.602176634e-19    # C
    NA   = 6.02214076e23      # /mol
    kBT = kB * float(temperature_K)
    # σ_q : e/nm² → C/m² ;  c : mol/L → mol/m³
    sigma_SI = float(sigma_q_e_nm2) * e_C / 1e-18
    n0 = float(ionic_strength_M) * 1000.0 * NA   # ions/m³
    denom = math.sqrt(8.0 * eps0 * float(epsilon_r) * kBT * n0)
    psi0_V = (2.0 * kBT / e_C) * math.asinh(sigma_SI / denom)
    # V → kJ/(mol·e): multiply by F = N_A · e (C/mol) then J → kJ
    return psi0_V * NA * e_C / 1000.0


def psi0_auto(sigma_q_e_nm2, ionic_strength_M, epsilon_r=78.5,
                temperature_K=298.15, mode="auto"):
    """Resolve ψ₀ [kJ/(mol·e)] selecting linear PB or full GC by σ_q magnitude.

    Convenience helper for runner scripts that want one call site regardless
    of surface strength.

    Parameters
    ----------
    sigma_q_e_nm2 : float
        Surface charge density (e/nm²). σ = 0 returns ψ₀ = 0.
    ionic_strength_M : float
        1:1 ionic strength (mol/L).
    epsilon_r : float
        Solvent permittivity, default 78.5.
    temperature_K : float
        Temperature (K), default 298.15.
    mode : {"auto", "linear", "gc"}
        - "auto" (default): always use full GC (linear regime is recovered
          automatically; safer default).
        - "linear": force linearized PB.
        - "gc": force full Gouy-Chapman.
    """
    if mode == "linear":
        from ptmc.config import debye_length_nm as _lam
        lamD = _lam(ionic_strength_M)
        return psi0_from_sigma_q(sigma_q_e_nm2, lamD, epsilon_r=epsilon_r)
    # auto / gc → full nonlinear (degrades to linear at low |σ|).
    return psi0_from_sigma_q_full_gc(
        sigma_q_e_nm2, ionic_strength_M,
        epsilon_r=epsilon_r, temperature_K=temperature_K)


# ---------------------------------------------------------------------------
# JAX continuum-surface energy (Steele 9-3 + screened Coulomb), ε-σ + LB form.
# Preferred over the legacy C6/C12 form: combines cleanly with force fields
# that store ε/σ natively (OPLS, CHARMM) and uses Lorentz-Berthelot rules.
# ---------------------------------------------------------------------------

def steele_energy_eps_sigma_jax(quat, trans, pos0, q,
                                 eps_p, sig6_p, sig12_p,
                                 cA, cB, lamD, z_min, psi0):
    """Steele 9-3 vdW (ε-σ form, LB combined) + screened Coulomb, in JAX.

    Per-atom energy:
        V(z_i) = ε_{ip}[cA·σ_{ip}¹²/z_i⁹ − cB·σ_{ip}⁶/z_i³]
                 + q_i·ψ₀·exp(−z_i/λ_D)

    where ε_{ip} = √(ε_i·ε_surf) and σ_{ip} = (σ_i + σ_surf)/2 (LB rules).
    cA = 4πρₛ/45,  cB = 2πρₛ/3.

    Pre-computed per-atom coefficients (from steele_eps_sigma_coefficients):
        eps_p   — LB combined ε_i [kJ/mol]
        sig6_p  — σ_{ip}⁶ [nm⁶]
        sig12_p — σ_{ip}¹² [nm¹²]

    SHAPE: pos0 (N,3), q (N,), eps_p (N,), sig6_p (N,), sig12_p (N,);
    scalars: cA (kJ·nm⁻³/mol), cB (kJ·nm⁻³/mol), lamD (nm), z_min (nm),
    psi0 (kJ·mol⁻¹·e⁻¹).
    """
    import jax.numpy as jnp
    from ptmc.mc.moves import quat_rotate

    pos = quat_rotate(quat, pos0) + trans
    z = pos[:, 2]
    wall = jnp.any(z < z_min)
    zc = jnp.maximum(z, z_min + 1e-3)
    e = jnp.sum(eps_p * (cA * sig12_p / zc ** 9 - cB * sig6_p / zc ** 3)
                + q * psi0 * jnp.exp(-zc / lamD))
    # Hard wall: +inf (consistent across all energy paths; Metropolis accept_logprob
    # short-circuits on non-finite dE so this never produces NaN downstream).
    return jnp.where(wall, jnp.array(jnp.inf, dtype=e.dtype), e)


def continuum_energy_eps_sigma_jax(quat, trans, pos0, q, eps, sigma,
                                    eps_surf, sigma_surf, rho_s,
                                    lambda_D, z_min, psi0):
    """Convenience wrapper: LB-combine ε,σ; call steele_energy_eps_sigma_jax.

    Usage in MC hot-loop:
        E_fn = partial(continuum_energy_eps_sigma_jax, pos0=..., q=..., ...)
        jax.jit(E_fn)(quat, trans)
    """
    import jax.numpy as jnp
    eps_p = jnp.sqrt(eps * eps_surf)
    sigma_p = (sigma + sigma_surf) / 2.0
    sig6_p = sigma_p ** 6
    sig12_p = sigma_p ** 12
    cA = 4.0 * jnp.pi * rho_s / 45.0
    cB = 2.0 * jnp.pi * rho_s / 3.0
    return steele_energy_eps_sigma_jax(quat, trans, pos0, q,
                                        eps_p, sig6_p, sig12_p,
                                        cA, cB, lambda_D, z_min, psi0)


def steele_eps_sigma_coefficients(eps, sigma, eps_surf, sigma_surf, rho_s):
    """Pre-compute per-atom ε-σ Steele 9-3 coefficients (numpy).

    Returns (eps_p, sig6_p, sig12_p, cA, cB) where:
        eps_p    = √(ε_i·ε_surf)          — LB combined ε
        sigma_p  = (σ_i + σ_surf) / 2      — LB combined σ
        sig6_p   = σ_p⁶,  sig12_p = σ_p¹²
        cA = 4πρₛ/45  [kJ·nm⁻³/mol]
        cB = 2πρₛ/3   [kJ·nm⁻³/mol]

    Callers pass these to steele_energy_eps_sigma_jax() to avoid recomputing
    the combined coefficients on every energy evaluation.
    """
    eps = np.asarray(eps); sigma = np.asarray(sigma)
    eps_p = np.sqrt(eps * eps_surf)
    sigma_p = (sigma + sigma_surf) / 2.0
    sig6_p = sigma_p ** 6
    sig12_p = sigma_p ** 12
    cA = 4.0 * np.pi * rho_s / 45.0
    cB = 2.0 * np.pi * rho_s / 3.0
    return eps_p, sig6_p, sig12_p, cA, cB


def steele_lb_effective_coefficients(c6, c12, eps_surf, sigma_surf, rho_s):
    """ε-σ Lorentz-Berthelot Steele 9-3 packed as *effective* (c6p, c12p, cA, cB)
    that feed the SAME ``steele_energy_jax`` / HT / PT / PA kernel unchanged.

    The geometric C6/C12 kernel evaluates ``cA·c12p/z⁹ − cB·c6p/z³``. The ε-σ LB
    Steele 9-3 is ``ε_p·(4πρ/45·σ_p¹²/z⁹ − 2πρ/3·σ_p⁶/z³)`` with
    ``ε_p = √(ε_i·ε_surf)`` and ``σ_p = (σ_i+σ_surf)/2``. Folding ε_p into the
    coefficients,
        c12p = ε_p·σ_p¹²,  c6p = ε_p·σ_p⁶,  cA = 4πρ/45,  cB = 2πρ/3,
    makes the two byte-identical, so the LB path needs NO separate hot-loop
    kernel — only different per-atom coefficients. Protein ε_i, σ_i are obtained
    from its C6/C12 via :func:`c6c12_to_eps_sigma`.

    Returns ``(c6p, c12p, cA, cB)`` with the same shapes/contract as
    :func:`steele_coefficients`.
    """
    eps_i, sigma_i = c6c12_to_eps_sigma(c6, c12)
    eps_p, sig6_p, sig12_p, cA, cB = steele_eps_sigma_coefficients(
        eps_i, sigma_i, eps_surf, sigma_surf, rho_s)
    c12p = eps_p * sig12_p
    c6p = eps_p * sig6_p
    return c6p, c12p, cA, cB


def steele_coefficients_for(c6, c12, c6_surf=1.0, c12_surf=1.0,
                            eps_surf=None, sigma_surf=None, rho_s=30.0):
    """Dispatch to the geometric-C6/C12 or ε-σ-LB Steele 9-3 coefficients.

    When BOTH ``eps_surf`` and ``sigma_surf`` are given (not None) the
    Lorentz-Berthelot ε-σ path is used (:func:`steele_lb_effective_coefficients`);
    otherwise the legacy geometric C6/C12 path (:func:`steele_coefficients`).
    Either way returns the common ``(c6p, c12p, cA, cB)`` tuple, so every
    downstream kernel (HT / PT / PA) is agnostic to which combining rule was
    selected. Centralising the branch here keeps the build_batch / run_multi_pt
    / pipeline call sites identical.
    """
    if eps_surf is not None and sigma_surf is not None:
        return steele_lb_effective_coefficients(c6, c12, eps_surf, sigma_surf, rho_s)
    return steele_coefficients(c6, c12, c6_surf, c12_surf, rho_s)


# ---------------------------------------------------------------------------
# Legacy JAX continuum-surface energy (C6/C12 + geometric combination).
# Kept for validation and grid-convergence testing (stage 2).
# ---------------------------------------------------------------------------
def steele_energy_jax(quat, trans, pos0, q, c6p, c12p, cA, cB, lamD, z_min, psi0):
    """Steele 9-3 + screened Coulomb energy for one pose, in JAX.

    Parameters use pre-computed per-chain coefficients c6p, c12p, cA, cB.
    See continuum_energy_jax() for the convenience wrapper that computes them.
    """
    import jax.numpy as jnp
    from ptmc.mc.moves import quat_rotate

    pos = quat_rotate(quat, pos0) + trans
    return steele_energy_from_pos_jax(pos, q, c6p, c12p, cA, cB, lamD, z_min, psi0)


def steele_energy_from_pos_jax(pos, q, c6p, c12p, cA, cB, lamD, z_min, psi0):
    """Steele 9-3 + screened Coulomb evaluated at lab-frame positions ``pos``.

    Factorised from ``steele_energy_jax`` so flexible (chi-aware) closures can
    apply chi kinematics *before* the rigid-body transform, then call this
    kernel on the final lab positions.  SHAPE: pos (N,3), q/c6p/c12p (N,);
    scalars cA, cB, lamD, z_min, psi0.
    """
    import jax.numpy as jnp
    z = pos[:, 2]
    wall = jnp.any(z < z_min)
    zc = jnp.maximum(z, z_min + 1e-3)
    e = jnp.sum(cA * c12p / zc ** 9 - cB * c6p / zc ** 3
                + q * psi0 * jnp.exp(-zc / lamD))
    return jnp.where(wall, jnp.array(jnp.inf, dtype=e.dtype), e)


def continuum_energy_jax(quat, trans, pos0, q, c6, c12,
                         c6_surf, c12_surf, rho_s, lambda_D, z_min, psi0):
    """Convenience wrapper: computes coefficients then calls steele_energy_jax."""
    import jax.numpy as jnp
    c6p = jnp.sqrt(c6 * c6_surf)
    c12p = jnp.sqrt(c12 * c12_surf)
    cA = jnp.pi * rho_s / 45.0
    cB = jnp.pi * rho_s / 6.0
    return steele_energy_jax(quat, trans, pos0, q, c6p, c12p, cA, cB,
                             lambda_D, z_min, psi0)


def steele_coefficients(c6, c12, c6_surf, c12_surf, rho_s):
    """Pre-compute per-atom Steele 9-3 pre-factors from LJ parameters.

    Returns (c6p, c12p, cA, cB) where:
      c6p_i  = sqrt(C6_i * C6_surf)  -- geometric combined C6
      c12p_i = sqrt(C12_i * C12_surf) -- geometric combined C12
      cA = pi * rho_s / 45            -- Steele z^-9  coefficient
      cB = pi * rho_s / 6              -- Steele z^-3  coefficient

    Callers pass these directly to steele_energy_jax() to avoid recomputing
    the combined coefficients on every energy evaluation.
    """
    import numpy as _np
    c6 = _np.asarray(c6);
    c12 = _np.asarray(c12)
    c6p = _np.sqrt(c6 * c6_surf)
    c12p = _np.sqrt(c12 * c12_surf)
    cA = _np.pi * rho_s / 45.0
    cB = _np.pi * rho_s / 6.0
    return c6p, c12p, cA, cB
