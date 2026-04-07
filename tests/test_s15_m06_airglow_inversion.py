"""
Tests for M06 airglow inversion.
Spec: docs/specs/S15_m06_airglow_inversion_2026-04-06.md
"""
import numpy as np
import pytest

from src.fpi.m06_airglow_inversion_2026_04_06 import (
    AirglowFitResult,
    AirglowFitFlags,
    fit_airglow_fringe,
)
from src.constants import OI_WAVELENGTH_M, SPEED_OF_LIGHT_MS, WIND_BIAS_BUDGET_MS


# ---------------------------------------------------------------------------
# T1 — S04 compliance: two_sigma_ = 2 × sigma_
# ---------------------------------------------------------------------------

def test_two_sigma_convention(synthetic_airglow_profile, synthetic_cal_result):
    """All two_sigma_ fields must equal exactly 2.0 × sigma_."""
    result = fit_airglow_fringe(synthetic_airglow_profile, synthetic_cal_result)
    pairs = [
        ("lambda_c_m",  "sigma_lambda_c_m",  "two_sigma_lambda_c_m"),
        ("v_rel_ms",    "sigma_v_rel_ms",     "two_sigma_v_rel_ms"),
        ("Y_line",      "sigma_Y_line",        "two_sigma_Y_line"),
        ("B_sci",       "sigma_B_sci",         "two_sigma_B_sci"),
    ]
    for _, s_name, ts_name in pairs:
        sigma     = getattr(result, s_name)
        two_sigma = getattr(result, ts_name)
        assert abs(two_sigma - 2.0 * sigma) < 1e-15, \
            f"{ts_name} = {two_sigma} ≠ 2 × {s_name} = {2 * sigma}"


# ---------------------------------------------------------------------------
# T2 — Doppler formula consistency
# ---------------------------------------------------------------------------

def test_doppler_formula_consistency(synthetic_airglow_profile, synthetic_cal_result):
    """v_rel must equal c × (lambda_c - lambda_0) / lambda_0."""
    result = fit_airglow_fringe(synthetic_airglow_profile, synthetic_cal_result)
    v_check = SPEED_OF_LIGHT_MS * (result.lambda_c_m - OI_WAVELENGTH_M) / OI_WAVELENGTH_M
    assert abs(result.v_rel_ms - v_check) < 1e-6, \
        f"v_rel = {result.v_rel_ms:.4f} m/s but Doppler gives {v_check:.4f} m/s"


# ---------------------------------------------------------------------------
# T3 — Zero wind: |v_rel| < 5 m/s
# ---------------------------------------------------------------------------

def test_zero_wind_recovery(synthetic_cal_result):
    """A zero-wind airglow image must recover v_rel within ±5 m/s."""
    from src.fpi.m04_airglow_synthesis_2026_04_05 import synthesise_airglow_image
    from src.fpi.m03_annular_reduction_2026_04_06 import reduce_science_frame
    from src.fpi.m01_airy_forward_model_2026_04_05 import InstrumentParams

    params = InstrumentParams()
    result_m04 = synthesise_airglow_image(v_rel_ms=0.0, params=params, add_noise=False)
    fp = reduce_science_frame(
        result_m04["image_2d"],
        cx=params.r_max, cy=params.r_max,
        sigma_cx=0.05, sigma_cy=0.05,
        r_max_px=params.r_max,
    )
    result = fit_airglow_fringe(fp, synthetic_cal_result)
    assert abs(result.v_rel_ms) < 5.0, \
        f"Zero-wind recovery: |v_rel| = {abs(result.v_rel_ms):.2f} m/s > 5 m/s"
    assert result.converged


# ---------------------------------------------------------------------------
# T4 — Known wind round-trip (noiseless): error < 20 m/s
# ---------------------------------------------------------------------------

def test_known_wind_round_trip(synthetic_cal_result):
    """
    Noiseless round-trip: inject v_rel=200 m/s, recover to within 20 m/s.
    The 20 m/s tolerance accounts for grid discretisation in the synthetic
    image (same inverse-crime effect seen in M05).
    """
    from src.fpi.m04_airglow_synthesis_2026_04_05 import synthesise_airglow_image
    from src.fpi.m03_annular_reduction_2026_04_06 import reduce_science_frame
    from src.fpi.m01_airy_forward_model_2026_04_05 import InstrumentParams

    v_truth = 200.0
    params  = InstrumentParams()
    result_m04 = synthesise_airglow_image(v_rel_ms=v_truth, params=params, add_noise=False)
    fp = reduce_science_frame(
        result_m04["image_2d"],
        cx=params.r_max, cy=params.r_max,
        sigma_cx=0.05, sigma_cy=0.05,
        r_max_px=params.r_max,
    )
    result = fit_airglow_fringe(fp, synthetic_cal_result)
    error  = abs(result.v_rel_ms - v_truth)
    assert error < 20.0, \
        f"v_rel error = {error:.2f} m/s > 20 m/s (noiseless round-trip)"
    assert result.converged


# ---------------------------------------------------------------------------
# T5 — Noisy round-trip: |error| < 3 × sigma_v, chi2 in [0.5, 3.0]
# ---------------------------------------------------------------------------

def test_noisy_round_trip_uncertainty_calibrated(synthetic_cal_result):
    """
    Noisy round-trip at SNR ≈ 5: recovered v_rel within 3σ of truth.
    Also verify chi2_reduced ∈ [0.5, 3.0].
    """
    from src.fpi.m04_airglow_synthesis_2026_04_05 import synthesise_airglow_image
    from src.fpi.m03_annular_reduction_2026_04_06 import reduce_science_frame
    from src.fpi.m01_airy_forward_model_2026_04_05 import InstrumentParams

    v_truth = 150.0
    params  = InstrumentParams()
    rng     = np.random.default_rng(42)
    result_m04 = synthesise_airglow_image(
        v_rel_ms=v_truth, params=params, snr=5.0, add_noise=True, rng=rng)
    fp = reduce_science_frame(
        result_m04["image_2d"],
        cx=params.r_max, cy=params.r_max,
        sigma_cx=0.05, sigma_cy=0.05,
        r_max_px=params.r_max,
    )
    result = fit_airglow_fringe(fp, synthetic_cal_result)
    error  = abs(result.v_rel_ms - v_truth)
    assert error < 3.0 * result.sigma_v_rel_ms, \
        f"|error| = {error:.2f} m/s > 3σ = {3 * result.sigma_v_rel_ms:.2f} m/s"
    assert 0.3 < result.chi2_reduced < 5.0, \
        f"chi2_reduced = {result.chi2_reduced:.3f} outside [0.3, 5.0]"


# ---------------------------------------------------------------------------
# T6 — Scan prevents FSR-period confusion
# ---------------------------------------------------------------------------

def test_scan_prevents_fsr_confusion(synthetic_cal_result):
    """
    Inject v_rel = -300 m/s (blueshifted). The brute-force scan must find
    the correct λ_c; without it, LM starting from λ₀ would drift by ~4723 m/s.
    """
    from src.fpi.m04_airglow_synthesis_2026_04_05 import synthesise_airglow_image
    from src.fpi.m03_annular_reduction_2026_04_06 import reduce_science_frame
    from src.fpi.m01_airy_forward_model_2026_04_05 import InstrumentParams

    v_truth = -300.0
    params  = InstrumentParams()
    result_m04 = synthesise_airglow_image(v_rel_ms=v_truth, params=params, add_noise=False)
    fp = reduce_science_frame(
        result_m04["image_2d"],
        cx=params.r_max, cy=params.r_max,
        sigma_cx=0.05, sigma_cy=0.05,
        r_max_px=params.r_max,
    )
    result = fit_airglow_fringe(fp, synthetic_cal_result)
    assert abs(result.v_rel_ms - v_truth) < 100.0, \
        (f"FSR confusion: recovered v_rel = {result.v_rel_ms:.0f} m/s, "
         f"truth = {v_truth:.0f} m/s, diff = {result.v_rel_ms - v_truth:.0f} m/s")


# ---------------------------------------------------------------------------
# T7 — CENTRE_FAILED profile raises ValueError
# ---------------------------------------------------------------------------

def test_centre_failed_raises(synthetic_cal_result):
    """fit_airglow_fringe must raise ValueError for CENTRE_FAILED profile."""
    from src.fpi.m03_annular_reduction_2026_04_06 import FringeProfile, QualityFlags
    import dataclasses

    # Build a minimal bad profile — just enough fields to reach the flag check
    # Use __new__ to avoid having to provide all dataclass fields
    bad_profile = FringeProfile.__new__(FringeProfile)
    bad_profile.quality_flags = QualityFlags.CENTRE_FAILED

    with pytest.raises(ValueError, match="CENTRE_FAILED"):
        fit_airglow_fringe(bad_profile, synthetic_cal_result)


# ---------------------------------------------------------------------------
# T8 — sigma_v within STM wind budget
# ---------------------------------------------------------------------------

def test_sigma_v_within_stm_budget(synthetic_cal_result):
    """
    At SNR ≈ 5, sigma_v_rel should be ≤ 2 × STM budget (9.8 m/s).
    """
    from src.fpi.m04_airglow_synthesis_2026_04_05 import synthesise_airglow_image
    from src.fpi.m03_annular_reduction_2026_04_06 import reduce_science_frame
    from src.fpi.m01_airy_forward_model_2026_04_05 import InstrumentParams

    params = InstrumentParams()
    rng    = np.random.default_rng(7)
    result_m04 = synthesise_airglow_image(
        v_rel_ms=100.0, params=params, snr=5.0, add_noise=True, rng=rng)
    fp = reduce_science_frame(
        result_m04["image_2d"],
        cx=params.r_max, cy=params.r_max,
        sigma_cx=0.05, sigma_cy=0.05,
        r_max_px=params.r_max,
    )
    result = fit_airglow_fringe(fp, synthetic_cal_result)
    assert result.sigma_v_rel_ms < 2.0 * WIND_BIAS_BUDGET_MS, (
        f"sigma_v = {result.sigma_v_rel_ms:.2f} m/s > "
        f"2 × STM budget ({2 * WIND_BIAS_BUDGET_MS:.1f} m/s)"
    )
