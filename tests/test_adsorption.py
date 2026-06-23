"""Tests for ptmc.analysis.adsorption."""
import math

import numpy as np
import pytest

from ptmc.analysis.adsorption import (
    adsorption_free_energy,
    beta0_adequacy,
    NUMBER_DENSITY_AT_1M_PER_NM3,
)
from ptmc.config import BOLTZMANN_KJ_PER_MOL_K


class TestAdsorptionFreeEnergy:
    def test_zero_logZ_zero_excess(self):
        """logZ_ratio = 0 ⇒ Z(β_target) = Z(β₀), the slab average of
        ⟨exp(-βE)⟩ equals 1 ⇒ no surface excess, K_ads = 0, dG_box = 0,
        and dG_std = NaN (the dimensionless log argument vanishes)."""
        result = adsorption_free_energy(
            0.0, temperature=300.0, z_lo=0.5, z_hi=3.5,
        )
        assert result.dG_box_kJ_per_mol == pytest.approx(0.0, abs=1e-12)
        assert result.K_ads_nm == pytest.approx(0.0, abs=1e-12)
        assert math.isnan(result.dG_std_kJ_per_mol)

    def test_positive_logZ_attractive(self):
        """logZ_ratio > 0 ⇒ slab average > 1 ⇒ attractive surface ⇒
        K_ads > 0 and dG_std finite. Sign convention: dG_box < 0."""
        T = 300.0
        kT = BOLTZMANN_KJ_PER_MOL_K * T
        dz = 3.0
        lz = 2.0  # exp(2)≈7.39: avg Boltzmann factor 7.39 over the slab
        r = adsorption_free_energy(lz, temperature=T, z_lo=0.5, z_hi=0.5 + dz)
        assert r.dG_box_kJ_per_mol == pytest.approx(-kT * lz)
        assert r.K_ads_nm == pytest.approx(dz * (math.exp(lz) - 1.0))
        # dG_std finite, more negative than dG_box if K_ads > λ_std ≈ 1.18.
        assert math.isfinite(r.dG_std_kJ_per_mol)

    def test_negative_logZ_repulsive(self):
        """logZ_ratio < 0 ⇒ slab average < 1 ⇒ net-repulsive over slab ⇒
        K_ads < 0 and dG_std = NaN (no physical equilibrium adsorption)."""
        r = adsorption_free_energy(-0.5, temperature=300.0, z_lo=0.5, z_hi=3.5)
        assert r.K_ads_nm < 0.0
        assert math.isnan(r.dG_std_kJ_per_mol)
        # dG_box still defined and positive (free energy goes UP into slab).
        assert r.dG_box_kJ_per_mol > 0.0

    def test_standard_state_lambda(self):
        """λ_std = (1/(N_A · c_std))^(1/3); at 1 M ≈ 1.184 nm."""
        r = adsorption_free_energy(1.0, temperature=300.0,
                                    z_lo=0.0, z_hi=1.0, c_std=1.0)
        expected_lambda = (1.0 / NUMBER_DENSITY_AT_1M_PER_NM3) ** (1.0 / 3.0)
        assert r.lambda_std_nm == pytest.approx(expected_lambda, rel=1e-9)
        assert r.lambda_std_nm == pytest.approx(1.1839, abs=1e-3)

    def test_standard_state_scales_with_concentration(self):
        """Higher c_std ⇒ smaller V_std ⇒ smaller λ_std ⇒ more-negative dG_std
        for the same K_ads."""
        T = 300.0
        kT = BOLTZMANN_KJ_PER_MOL_K * T
        r1 = adsorption_free_energy(2.0, temperature=T, z_lo=0.5, z_hi=3.5,
                                     c_std=1.0)
        r10 = adsorption_free_energy(2.0, temperature=T, z_lo=0.5, z_hi=3.5,
                                      c_std=10.0)
        # K_ads identical, only λ_std differs by 10^{1/3}.
        assert r1.K_ads_nm == pytest.approx(r10.K_ads_nm)
        assert r10.lambda_std_nm == pytest.approx(
            r1.lambda_std_nm / (10.0 ** (1.0 / 3.0)), rel=1e-9)
        # dG_std at higher c_std MORE negative (binding looks tighter when
        # the reference state is more concentrated).
        assert r10.dG_std_kJ_per_mol < r1.dG_std_kJ_per_mol

    def test_dG_box_temperature_scaling(self):
        """dG_box = -kT · logZ_ratio scales linearly in T at fixed logZ."""
        r1 = adsorption_free_energy(1.5, temperature=150.0, z_lo=0.0, z_hi=1.0)
        r2 = adsorption_free_energy(1.5, temperature=300.0, z_lo=0.0, z_hi=1.0)
        assert r2.dG_box_kJ_per_mol == pytest.approx(2.0 * r1.dG_box_kJ_per_mol,
                                                      rel=1e-9)

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError):
            adsorption_free_energy(0.0, temperature=300.0,
                                    z_lo=1.0, z_hi=0.5)  # inverted
        with pytest.raises(ValueError):
            adsorption_free_energy(0.0, temperature=-1.0,
                                    z_lo=0.0, z_hi=1.0)  # bad T
        with pytest.raises(ValueError):
            adsorption_free_energy(0.0, temperature=300.0,
                                    z_lo=0.0, z_hi=1.0, c_std=0.0)  # bad c

    def test_slab_width_independence_dG_box(self):
        """dG_box = -kT·logZ depends only on logZ, NOT on the slab Δz.
        (The Δz dependence shows up in K_ads alone.)"""
        T = 300.0
        r_thin = adsorption_free_energy(1.0, temperature=T,
                                         z_lo=0.0, z_hi=0.5)
        r_wide = adsorption_free_energy(1.0, temperature=T,
                                         z_lo=0.0, z_hi=10.0)
        assert r_thin.dG_box_kJ_per_mol == pytest.approx(r_wide.dG_box_kJ_per_mol)
        # K_ads DOES scale with Δz at fixed logZ — captures that a wider slab
        # encompasses more bulk-like volume contributing 0 to the excess.
        assert r_wide.K_ads_nm == pytest.approx(20.0 * r_thin.K_ads_nm, rel=1e-6)


class TestBeta0Adequacy:
    def test_high_T_passes(self):
        """At very high T (β tiny), |β·E| is small ⇒ ok=True."""
        E = np.array([-50.0, -10.0, 5.0, 80.0])  # kJ/mol
        beta0 = 1.0 / (BOLTZMANN_KJ_PER_MOL_K * 1.0e5)  # 100000 K → β tiny
        check = beta0_adequacy(E, beta0, threshold=0.5)
        assert check["ok"] is True
        assert check["max"] == pytest.approx(beta0 * 80.0)

    def test_low_T_fails(self):
        """At room T with kJ/mol-scale energies, |β·E| ≫ 1 ⇒ ok=False."""
        E = np.array([-50.0, 30.0, 100.0])
        beta0 = 1.0 / (BOLTZMANN_KJ_PER_MOL_K * 300.0)  # ~0.4 mol/kJ
        check = beta0_adequacy(E, beta0, threshold=0.5)
        assert check["ok"] is False
        # Message should suggest a higher T_start.
        assert "T_start" in check["msg"]

    def test_all_inf_fails(self):
        """All non-finite ⇒ catastrophic, ok=False without crash."""
        E = np.array([np.inf, np.inf, np.inf])
        check = beta0_adequacy(E, beta0=1.0)
        assert check["ok"] is False
        assert math.isinf(check["mean"]) or math.isinf(check["max"])

    def test_mixed_finite(self):
        """Non-finite walkers are ignored; finite ones drive the verdict."""
        E = np.array([np.inf, 1.0, 2.0])
        beta0 = 0.1
        check = beta0_adequacy(E, beta0, threshold=0.5)
        # max over finite only: max(|0.1·1|, |0.1·2|) = 0.2 ≤ 0.5.
        assert check["max"] == pytest.approx(0.2)
        assert check["ok"] is True
