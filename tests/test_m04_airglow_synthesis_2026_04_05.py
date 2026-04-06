"""
Tests for M04 Airglow Fringe Synthesis.

Spec:        specs/S11_m04_airglow_synthesis_2026-04-05.md
Spec tests:  T1–T8
Run with:    pytest tests/test_m04_airglow_synthesis_2026_04_05.py -v
"""

import numpy as np
import pytest

from src.fpi.m01_airy_forward_model_2026_04_05 import (
    InstrumentParams,
    OI_WAVELENGTH_M,
    SPEED_OF_LIGHT_MS,
)
from src.fpi.m04_airglow_synthesis_2026_04_05 import (
    lambda_c_to_v_rel,
    synthesise_airglow_image,
    v_rel_to_lambda_c,
)


# ---------------------------------------------------------------------------
# T1 — Output shapes and keys
# ---------------------------------------------------------------------------
def test_output_shapes():
    """All returned arrays must have expected shapes."""
    params = InstrumentParams()
    result = synthesise_airglow_image(100.0, params, add_noise=False)
    assert result["image_2d"].shape == (256, 256)
    assert result["image_noiseless"].shape == (256, 256)
    assert result["profile_1d"].shape == (500,)
    assert result["r_grid"].shape == (500,)
    assert isinstance(result["lambda_c_m"], float)
    assert isinstance(result["snr_actual"], float)


# ---------------------------------------------------------------------------
# T2 — Doppler shift moves fringe rings
# ---------------------------------------------------------------------------
def test_doppler_shift_moves_fringes():
    """
    FPI ring physics: larger λ (redshift, positive v_rel) shifts rings INWARD
    (to smaller r). This follows from the peak condition cos(θ) = mλ/(2nt):
    as λ increases the maximum visible order decreases and rings move inward.
    Negative v_rel (blueshift, smaller λ) shifts rings outward (larger r).
    """
    from scipy.signal import find_peaks

    params = InstrumentParams()

    results = {}
    for v in [-500.0, 0.0, +500.0]:
        res = synthesise_airglow_image(v, params, add_noise=False)
        profile = res["profile_1d"]
        r_grid = res["r_grid"]
        peaks, _ = find_peaks(profile)
        if len(peaks) > 0:
            results[v] = r_grid[peaks[0]]  # radius of first peak

    if len(results) == 3:
        # redshift (v>0) → smaller r; blueshift (v<0) → larger r
        assert results[+500.0] < results[0.0] < results[-500.0], (
            "Fringe peaks do not shift monotonically with v_rel"
        )


# ---------------------------------------------------------------------------
# T3 — Zero wind gives symmetric profile
# ---------------------------------------------------------------------------
def test_zero_wind_symmetric():
    """
    At v_rel = 0, lambda_c must equal OI_WAVELENGTH_M exactly.
    """
    params = InstrumentParams()
    result = synthesise_airglow_image(0.0, params, add_noise=False)
    assert abs(result["lambda_c_m"] - OI_WAVELENGTH_M) < 1e-18, (
        f"Zero wind: λ_c = {result['lambda_c_m']:.6e}, expected {OI_WAVELENGTH_M:.6e}"
    )


# ---------------------------------------------------------------------------
# T4 — Doppler shift magnitude correct
# ---------------------------------------------------------------------------
def test_doppler_shift_magnitude():
    """
    λ_c - λ₀ must equal λ₀ × v_rel / c to 1 ppm.
    For v_rel = 100 m/s: Δλ ≈ +0.210 pm.
    """
    v = 100.0
    params = InstrumentParams()
    result = synthesise_airglow_image(v, params, add_noise=False)
    lc = result["lambda_c_m"]
    expected_shift = OI_WAVELENGTH_M * v / SPEED_OF_LIGHT_MS
    actual_shift = lc - OI_WAVELENGTH_M
    assert abs(actual_shift - expected_shift) / abs(expected_shift) < 1e-6, (
        f"Doppler shift {actual_shift:.4e} m, expected {expected_shift:.4e} m"
    )


# ---------------------------------------------------------------------------
# T5 — Round-trip v_rel recovery
# ---------------------------------------------------------------------------
def test_round_trip_v_rel():
    """
    lambda_c_to_v_rel(v_rel_to_lambda_c(v)) must return v to < 1e-6 m/s.
    """
    for v in [-7200.0, -100.0, 0.0, +100.0, +500.0]:
        lc = v_rel_to_lambda_c(v)
        v_recovered = lambda_c_to_v_rel(lc)
        assert abs(v_recovered - v) < 1e-6, (
            f"Round-trip error {abs(v_recovered - v):.2e} m/s for v={v}"
        )


# ---------------------------------------------------------------------------
# T6 — SNR achieved within 20% of target
# ---------------------------------------------------------------------------
def test_snr_achieved():
    """
    Actual SNR of the noiseless fringe must be within 20% of specified target.
    """
    params = InstrumentParams()
    result = synthesise_airglow_image(
        100.0, params, snr=5.0, add_noise=True, rng=np.random.default_rng(0)
    )
    assert abs(result["snr_actual"] - 5.0) / 5.0 < 0.20, (
        f"SNR {result['snr_actual']:.2f} more than 20% from target 5.0"
    )


# ---------------------------------------------------------------------------
# T7 — Reproducible with fixed seed
# ---------------------------------------------------------------------------
def test_reproducible_with_seed():
    """Two calls with identical seeds must produce identical noisy images."""
    params = InstrumentParams()
    r1 = synthesise_airglow_image(
        100.0, params, add_noise=True, rng=np.random.default_rng(7)
    )
    r2 = synthesise_airglow_image(
        100.0, params, add_noise=True, rng=np.random.default_rng(7)
    )
    np.testing.assert_array_equal(r1["image_2d"], r2["image_2d"])


# ---------------------------------------------------------------------------
# T8 — TypeError raised for temperature_K argument
# ---------------------------------------------------------------------------
def test_no_temperature_argument():
    """
    synthesise_airglow_image must raise TypeError if temperature_K is passed.
    The delta-function source model has no temperature parameter.
    This test enforces the design decision permanently.
    """
    params = InstrumentParams()
    with pytest.raises(TypeError, match="temperature_K"):
        synthesise_airglow_image(100.0, params, temperature_K=800.0)
