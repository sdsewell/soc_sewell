"""
Tests for m03_airglow_synthesis_2026_05_24.py
Spec:  specs/M03_airglow_synthesis_addendum_2026-05-24.md  (§4, T-M03-T01..T07)
Run:   pytest tests/test_m03_airglow_synthesis.py -v
"""

import numpy as np
import pytest

from src.fpi.airy_forward_model_2026_05_05 import InstrumentParams
from src.fpi.m03_airglow_synthesis_2026_05_24 import (
    delta_lambda_from_temperature,
    synthesise_airglow_image,
)


# ---------------------------------------------------------------------------
# Shared fixture: H05-calibrated instrument parameters
# ---------------------------------------------------------------------------

@pytest.fixture
def params():
    return InstrumentParams(
        t      = 20.1070e-3,
        R_refl = 0.239,
        alpha  = 1.6084e-4,
        I0     = 6479.9,
        I1     = -0.04684,
        I2     = -0.02599,
        sigma0 = 0.5528,
        sigma1 = 0.0,
        sigma2 = 0.0,
        B      = 2010.7,
        r_max  = 110.0,
    )


# ---------------------------------------------------------------------------
# T-M03-T01: use_temperature=False is byte-identical to pre-addendum code
# ---------------------------------------------------------------------------

def test_T01_temperature_false_is_identical(params):
    """use_temperature=False must produce identical output to the _05_13 module."""
    from src.fpi.m03_airglow_synthesis_2026_05_13 import (
        synthesise_airglow_image as synthesise_old,
    )
    rng_seed = 99

    new = synthesise_airglow_image(
        params, v_rel_ms=50.0, Y_line=1.0, Y_bg=0.0,
        add_noise=False, use_temperature=False,
        rng=np.random.default_rng(rng_seed),
    )
    old = synthesise_old(
        params, v_rel_ms=50.0, Y_line=1.0, Y_bg=0.0,
        add_noise=False,
        rng=np.random.default_rng(rng_seed),
    )

    diff = np.abs(new["profile_1d"] - old["profile_1d"]).max()
    assert diff < 1e-10, f"Max diff {diff:.3e} ADU (expected < 1e-10)"


# ---------------------------------------------------------------------------
# T-M03-T02: delta_lambda_from_temperature(800.0) in [1.30e-12, 1.40e-12] m
# ---------------------------------------------------------------------------

def test_T02_delta_lambda_800K():
    """Harding Eq. 12 at T=800 K should give ~1.355 pm."""
    dl = delta_lambda_from_temperature(800.0)
    assert 1.30e-12 <= dl <= 1.40e-12, f"delta_lambda={dl:.4e} m outside [1.30e-12, 1.40e-12]"


# ---------------------------------------------------------------------------
# T-M03-T03: delta_lambda_from_temperature(T_K=0) raises ValueError
# ---------------------------------------------------------------------------

def test_T03_delta_lambda_zero_raises():
    """Temperature <= 0 must raise ValueError."""
    with pytest.raises(ValueError):
        delta_lambda_from_temperature(0.0)

    with pytest.raises(ValueError):
        delta_lambda_from_temperature(-100.0)


# ---------------------------------------------------------------------------
# T-M03-T04: thermal path produces wider fringes than delta-function path
# ---------------------------------------------------------------------------

def test_T04_thermal_fringes_wider(params):
    """
    With T=800K the thermally broadened fringes should be wider.
    Measured as: number of radial bins above the half-amplitude level is
    larger for the thermal path than for the delta-function path.
    """
    kw = dict(v_rel_ms=0.0, Y_line=1.0, Y_bg=0.0, add_noise=False,
              rng=np.random.default_rng(0))

    p_delta = synthesise_airglow_image(params, use_temperature=False, **kw)["profile_1d"]
    p_therm = synthesise_airglow_image(params, use_temperature=True, T_true_K=800.0, **kw)["profile_1d"]

    def bins_above_half(profile):
        half = (profile.max() + profile.min()) / 2.0
        return int((profile > half).sum())

    w_delta = bins_above_half(p_delta)
    w_therm = bins_above_half(p_therm)
    assert w_therm > w_delta, (
        f"Thermal fringes not wider: thermal={w_therm} bins, delta={w_delta} bins"
    )


# ---------------------------------------------------------------------------
# T-M03-T05: T_true_K and delta_lambda_m keys present and finite
# ---------------------------------------------------------------------------

def test_T05_return_dict_keys(params):
    """use_temperature=True must add T_true_K and delta_lambda_m to return dict."""
    result = synthesise_airglow_image(
        params, v_rel_ms=0.0, add_noise=False,
        use_temperature=True, T_true_K=800.0,
    )
    assert "T_true_K" in result
    assert "delta_lambda_m" in result
    assert np.isfinite(result["T_true_K"]), "T_true_K is not finite"
    assert np.isfinite(result["delta_lambda_m"]), "delta_lambda_m is not finite"
    assert result["T_true_K"] == 800.0
    assert 1.30e-12 <= result["delta_lambda_m"] <= 1.40e-12

    # use_temperature=False must return None for both keys
    result_false = synthesise_airglow_image(
        params, v_rel_ms=0.0, add_noise=False, use_temperature=False,
    )
    assert result_false["T_true_K"] is None
    assert result_false["delta_lambda_m"] is None


# ---------------------------------------------------------------------------
# T-M03-T06: thermal integral ≈ delta-function integral (within 10%)
# ---------------------------------------------------------------------------

def test_T06_profile_integrals_within_10pct(params):
    """
    For Y_line=1, Y_bg=0, T=800K, v=0: the total radial integral (sum of
    profile_1d) should agree within 10% between the two paths.
    """
    kw = dict(v_rel_ms=0.0, Y_line=1.0, Y_bg=0.0, add_noise=False,
              rng=np.random.default_rng(0))

    p_delta = synthesise_airglow_image(params, use_temperature=False, **kw)["profile_1d"]
    p_therm = synthesise_airglow_image(
        params, use_temperature=True, T_true_K=800.0, **kw
    )["profile_1d"]

    ratio = p_therm.sum() / p_delta.sum()
    assert abs(ratio - 1.0) < 0.10, (
        f"Integral ratio {ratio:.4f} is not within 10% of 1.0"
    )


# ---------------------------------------------------------------------------
# T-M03-T07: convergence — L_synth=300 vs L_synth=200 (< 1% max difference)
# ---------------------------------------------------------------------------

def test_T07_lsynth_convergence(params):
    """
    Anti-inverse-crime check: L_synth=300 and L_synth=200 should agree to
    better than 1% across the entire radial profile.
    """
    kw = dict(v_rel_ms=0.0, Y_line=1.0, Y_bg=0.0, add_noise=False,
              use_temperature=True, T_true_K=800.0, rng=np.random.default_rng(0))

    p300 = synthesise_airglow_image(params, L_synth=300, **kw)["profile_1d"]
    p200 = synthesise_airglow_image(params, L_synth=200, **kw)["profile_1d"]

    max_diff_pct = float(np.abs(p300 - p200).max() / p300.max() * 100)
    assert max_diff_pct < 1.0, (
        f"L_synth convergence: max diff = {max_diff_pct:.4f}% (expected < 1%)"
    )
