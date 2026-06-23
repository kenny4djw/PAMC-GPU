"""Adsorption free energy from PA's logZ_ratio.

Converts the box-conditional log[Z(β_target)/Z(β₀)] returned by
``ptmc.sampler.population_annealing.run_pa`` into physically meaningful
adsorption quantities.

Three numbers are produced from one logZ_ratio:

* ``dG_box``   – raw box-conditional ΔG = −kT · logZ_ratio. The free-energy
                 cost of going from a uniform distribution over the sampled
                 box at β₀ → 0 to the canonical distribution at β_target.
                 Sensible *only* if β₀ is hot enough that the initial
                 walker ensemble is actually close to the uniform prior;
                 ``beta0_adequacy()`` checks that.
* ``K_ads``    – surface-excess equilibrium constant (units of nm). Defined
                 as K_ads = ∫_{z_lo}^{z_hi} dz [⟨exp(−βE(z,Ω))⟩_Ω − 1].
                 Assumes z_hi is far enough that the protein is bulk-like
                 there (i.e. the rotation-averaged Boltzmann factor → 1).
* ``dG_std``   – standard-state-corrected ΔG⁰_ads (kJ/mol, 1 M default).
                 ΔG⁰_ads = −kT · ln(K_ads / λ_std) where
                 λ_std = (V_std)^(1/3) ≈ 1.18 nm at 1 M.

Conventions
-----------
Sign:    ΔG⁰_ads < 0  ⇔  binding favourable, K_ads > λ_std.
Units:   GROMACS — length nm, energy kJ/mol, c_std mol/L, T in K.
         k_B = 8.314·10⁻³ kJ mol⁻¹ K⁻¹.
Reference state: 1 M ideal solution, isotropic orientation. The rotational
8π² factor cancels exactly in Z_bound / Z_ref provided rotation is sampled
on *both* sides — this is the case for the project's PA driver.

Caveats
-------
* K_ads / ΔG⁰_ads assume ⟨exp(−βE)⟩_Ω → 1 at z_hi. If the slab is too thin
  the bulk baseline is not reached and K_ads is underestimated. Increase
  z_grid_max (crystal / full_atom) or pa_z_hi (continuum / patterned) until
  ΔG⁰_ads no longer shifts with the slab top.
* PA's estimator has Jensen-direction bias O(1/R) (Machta 2010) — log Z is
  systematically *underestimated*, so ΔG⁰_ads is systematically *more
  negative* (binding overestimated) at small population size. Use multiple
  independent runs for an honest error bar.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ptmc.config import BOLTZMANN_KJ_PER_MOL_K

# 1 mol/L in particles per nm³ = N_A · 1 mol/L · (1 L / 10²⁴ nm³).
NUMBER_DENSITY_AT_1M_PER_NM3: float = 0.6022140857


@dataclass(frozen=True)
class AdsorptionResult:
    """Adsorption free-energy summary derived from one PA run.

    All energies in kJ/mol, lengths in nm. ``dG_std`` is NaN when the sampled
    slab shows no net binding (K_ads ≤ 0) — that is the honest answer rather
    than a complex logarithm.
    """
    dG_box_kJ_per_mol: float
    K_ads_nm: float
    dG_std_kJ_per_mol: float
    logZ_ratio: float
    temperature_K: float
    z_lo: float
    z_hi: float
    lambda_std_nm: float
    c_std_M: float


def adsorption_free_energy(
    logZ_ratio: float,
    *,
    temperature: float,
    z_lo: float,
    z_hi: float,
    c_std: float = 1.0,
) -> AdsorptionResult:
    """Convert PA's logZ_ratio into ΔG_box, K_ads, ΔG⁰_ads.

    Assumes the PA run sampled z ∈ [z_lo, z_hi] with rotation over SO(3),
    started at β₀ small enough that ⟨exp(−β₀ E)⟩_Ω ≈ 1 over the slab, and
    reached β_target at ``temperature``.

    For lateral xy: either the surface is homogeneous and only z is sampled
    (xy integral cancels in the ratio), or xy is sampled with the surface
    periodic cell — both give the same K_ads.

    Parameters
    ----------
    logZ_ratio : float
        ``pa_out['logZ_ratio']`` = log[Z(β_target) / Z(β₀)].
    temperature : float
        Target temperature (K).
    z_lo, z_hi : float
        Slab bounds covered by the PA walkers (nm). Must satisfy z_hi > z_lo.
    c_std : float
        Standard reference concentration (mol/L). Default 1 M.

    Returns
    -------
    AdsorptionResult
    """
    if not (z_hi > z_lo):
        raise ValueError(f"z_hi ({z_hi}) must exceed z_lo ({z_lo})")
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0 K, got {temperature}")
    if c_std <= 0:
        raise ValueError(f"c_std must be > 0 mol/L, got {c_std}")

    kT = BOLTZMANN_KJ_PER_MOL_K * float(temperature)
    dz = float(z_hi) - float(z_lo)
    lz = float(logZ_ratio)

    # Box-conditional free energy. -kT log(Z_target / Z_ref).
    dG_box = -kT * lz

    # 1 M reference length λ_std = (V_std)^(1/3), V_std = 1/(N_A · c_std).
    V_std = 1.0 / (NUMBER_DENSITY_AT_1M_PER_NM3 * float(c_std))
    lambda_std = V_std ** (1.0 / 3.0)

    # K_ads (surface excess, nm). Conceptually:
    #     K_ads = dz · (exp(logZ_ratio) − 1)
    # but math.exp overflows for logZ_ratio ≳ 709 (float64 cap). Strong
    # continuum models (Au/graphite) routinely produce logZ_ratio ~ 100–200,
    # and the dG_std reduction below is log-space anyway, so do the whole
    # arithmetic in log space:
    #     log(K_ads / λ_std) = log(dz) + log(exp(lz) − 1) − log(λ_std)
    #     log(exp(lz) − 1)   = lz + log1p(−exp(−lz))     for lz > 0
    # The log1p form is exact at large lz (→ lz) and at small lz (→ log(lz)).
    # For lz ≤ 0 the protein has no net surface excess (K_ads ≤ 0) and dG_std
    # is undefined — that case still returns NaN, identical to the prior code.
    if lz > 0.0:
        # Stable log of (exp(lz) − 1); avoids the overflow on math.exp(lz).
        log_excess = lz + math.log1p(-math.exp(-lz))
        log_K_over_lstd = math.log(dz) + log_excess - math.log(lambda_std)
        dG_std = -kT * log_K_over_lstd
        # Materialise K_ads only if it actually fits in float64; otherwise
        # expose +inf so downstream consumers don't silently truncate.
        if lz < 700.0:
            K_ads = dz * (math.exp(lz) - 1.0)
        else:
            K_ads = float("inf")
    else:
        K_ads = dz * (math.exp(lz) - 1.0)   # safe: lz ≤ 0 → no overflow
        dG_std = float("nan")

    return AdsorptionResult(
        dG_box_kJ_per_mol=dG_box,
        K_ads_nm=K_ads,
        dG_std_kJ_per_mol=dG_std,
        logZ_ratio=lz,
        temperature_K=float(temperature),
        z_lo=float(z_lo),
        z_hi=float(z_hi),
        lambda_std_nm=lambda_std,
        c_std_M=float(c_std),
    )


def beta0_adequacy(
    energies: np.ndarray,
    beta0: float,
    *,
    threshold: float = 0.5,
) -> dict:
    """Diagnose whether β₀ is in the high-T limit for the given initial walkers.

    The PA estimator's interpretation as ΔG_ref→target only holds when the
    walker ensemble at β₀ samples the *uniform* prior. The acid test is
    ⟨|β₀·E|⟩ ≪ 1: if walkers feel the energy landscape at the starting
    temperature, Z(β₀) is *not* the geometric reference volume, and
    ``dG_box`` is biased by an unknown −kT·log[Z(β₀)/Z_uniform] offset.

    Parameters
    ----------
    energies : (M,) array
        Per-walker energies (kJ/mol) at β₀, before any cooling step.
    beta0 : float
        Starting inverse temperature (mol/kJ).
    threshold : float
        Acceptable max |β₀·E|. 0.5 ≈ corresponds to weights staying within
        a factor of e^{0.5} ≈ 1.6 of each other.

    Returns
    -------
    dict with keys::

        mean : float    ⟨|β₀ E|⟩
        max  : float    max |β₀ E|
        ok   : bool     True iff max ≤ threshold (β₀ is hot enough)
        msg  : str      human-readable recommendation
    """
    E = np.asarray(energies, dtype=np.float64)
    finite = np.isfinite(E)
    if not finite.any():
        return {"mean": float("inf"), "max": float("inf"),
                "ok": False,
                "msg": "all initial energies are non-finite"}
    abs_be = np.abs(float(beta0) * E[finite])
    m_mean = float(abs_be.mean())
    m_max = float(abs_be.max())
    ok = m_max <= float(threshold)
    if ok:
        msg = (f"β₀ adequate: max |β₀E| = {m_max:.2f} ≤ {threshold:.2f}, "
               f"initial walkers are approximately uniform.")
    else:
        # Rough suggestion: T_start scaled so max |β·E| ≈ threshold/2.
        from ptmc.config import BOLTZMANN_KJ_PER_MOL_K
        T_rec = (m_max / max(threshold, 1e-9)) * (1.0 / (beta0 * BOLTZMANN_KJ_PER_MOL_K))
        msg = (f"β₀ may be too cold: max |β₀E| = {m_max:.2f} > {threshold:.2f}. "
               f"Initial walkers are NOT uniformly sampling the slab — "
               f"dG_box is biased. Consider raising T_start to ≳ {T_rec:.0f} K.")
    return {"mean": m_mean, "max": m_max, "ok": ok, "msg": msg}
