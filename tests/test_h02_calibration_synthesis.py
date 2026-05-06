"""
Tests for H02 Calibration Fringe Synthesis.

Spec:        docs/specs/H02_calibration_synthesis_2026-05-05.md
Spec tests:  T1–T8
Run with:    pytest tests/test_h02_calibration_synthesis.py -v

Constants imported from windcube.constants (not from H01 module).
airy_modified() called with (r, lam, params) 3-arg form per H01 §6.
"""

import numpy as np
import pytest

from src.fpi.airy_forward_model_2026_05_05 import InstrumentParams, airy_modified
from src.fpi.m02_calibration_synthesis_2026_05_05 import synthesise_calibration_image
from windcube.constants import (
    NE_INTENSITY_2,
    NE_WAVELENGTH_1_AIR_M,
    NE_WAVELENGTH_2_AIR_M,
)


# ---------------------------------------------------------------------------
# T1 — Output shapes correct
# ---------------------------------------------------------------------------
def test_output_shapes():
    """All returned arrays must have the expected shapes."""
    params = InstrumentParams()
    result = synthesise_calibration_image(
        params, image_size=256, R_bins=500, add_noise=False
    )
    assert result["image_2d"].shape == (256, 256)
    assert result["image_noiseless"].shape == (256, 256)
    assert result["profile_1d"].shape == (500,)
    assert result["r_grid"].shape == (500,)
    assert result["cx"] is not None
    assert result["cy"] is not None


# ---------------------------------------------------------------------------
# T2 — Noiseless image everywhere positive
# ---------------------------------------------------------------------------
def test_image_positivity():
    """Noiseless calibration image must be everywhere positive."""
    params = InstrumentParams()
    result = synthesise_calibration_image(params, add_noise=False)
    assert np.all(result["image_noiseless"] > 0), (
        "Noiseless calibration image contains non-positive values"
    )


# ---------------------------------------------------------------------------
# T3 — Circular symmetry (dynamic trough finder)
# ---------------------------------------------------------------------------
def test_circular_symmetry():
    """
    T3 — Azimuthal symmetry of the Airy pattern.

    Evaluates the coefficient of variation (CV = std/mean) of the ideal
    Airy intensity across a thin annulus at a fringe trough radius.

    The trough is found dynamically: evaluate the 1D ideal Airy profile
    on a fine radial grid and locate the first local minimum beyond r=10 px.
    At a trough the intensity gradient is near zero, so pixel-to-pixel
    variation from integer rounding is negligible and a tight CV threshold
    is achievable and meaningful.

    Threshold: CV < 0.05 at the trough.
    """
    from scipy.signal import argrelmin
    from src.fpi.airy_forward_model_2026_05_05 import airy_ideal
    from windcube.constants import OI_WAVELENGTH_AIR_M

    params = InstrumentParams()

    # find the first trough beyond r=10 px
    r_fine = np.linspace(10.0, params.r_max, 2000)
    profile = airy_ideal(r_fine, OI_WAVELENGTH_AIR_M, params)

    minima_idx = argrelmin(profile, order=5)[0]
    assert len(minima_idx) > 0, (
        "No local minimum found in Airy profile between r=10 and r_max. "
        "Check params.alpha / params.t."
    )
    r_test = r_fine[minima_idx[0]]

    # build a thin annulus at r_test with ±0.5 px radial scatter
    n_az = 360
    angles = np.linspace(0, 2 * np.pi, n_az, endpoint=False)
    r_annulus = r_test + 0.5 * np.cos(angles)
    intensities = airy_ideal(r_annulus, OI_WAVELENGTH_AIR_M, params)

    mean_i = np.mean(intensities)
    std_i = np.std(intensities)
    assert mean_i > 0, "Mean intensity at trough annulus is zero"
    cv = std_i / mean_i

    assert cv < 0.05, (
        f"Circular symmetry CV={cv:.4f} exceeds 0.05 at trough r={r_test:.1f} px. "
        f"Expected near-zero CV at a fringe minimum."
    )


# ---------------------------------------------------------------------------
# T4 — Radial beat pattern present
# ---------------------------------------------------------------------------
def test_beat_pattern_present():
    """
    The 1D profile must show amplitude modulation from the two neon lines.
    Peak heights must vary by more than 10% (peak ratio > 1.10).
    """
    from scipy.signal import find_peaks

    params = InstrumentParams()
    result = synthesise_calibration_image(params, add_noise=False)
    profile = result["profile_1d"]
    peaks, _ = find_peaks(profile, height=0.3 * np.max(profile))
    assert len(peaks) >= 4, (
        f"Only {len(peaks)} peaks found — not enough to measure beat pattern"
    )
    peak_heights = profile[peaks]
    ratio = np.max(peak_heights) / np.min(peak_heights)
    assert ratio > 1.10, (
        f"No beat modulation detected: max/min peak ratio = {ratio:.3f} (expect > 1.10)"
    )


# ---------------------------------------------------------------------------
# T5 — Poisson noise statistics
# ---------------------------------------------------------------------------
def test_poisson_noise_statistics():
    """
    Variance of noise should equal mean signal (Poisson: Var = Mean).
    Allow 20% tolerance.
    """
    params = InstrumentParams()
    r1 = synthesise_calibration_image(
        params, add_noise=True, rng=np.random.default_rng(42)
    )
    r2 = synthesise_calibration_image(params, add_noise=False)
    noise = r1["image_2d"] - r2["image_noiseless"]
    signal = r2["image_noiseless"]
    mask = signal > 100
    assert mask.sum() >= 100, "Insufficient high-signal pixels for test"
    ratio = np.var(noise[mask]) / np.mean(signal[mask])
    assert 0.8 < ratio < 1.2, (
        f"Poisson noise check failed: Var/Mean = {ratio:.3f} (expect ≈ 1.0)"
    )


# ---------------------------------------------------------------------------
# T6 — Reproducible with fixed seed
# ---------------------------------------------------------------------------
def test_reproducible_with_seed():
    """Two calls with identical seeds must produce identical noisy images."""
    params = InstrumentParams()
    r1 = synthesise_calibration_image(
        params, add_noise=True, rng=np.random.default_rng(99)
    )
    r2 = synthesise_calibration_image(
        params, add_noise=True, rng=np.random.default_rng(99)
    )
    np.testing.assert_array_equal(
        r1["image_2d"],
        r2["image_2d"],
        err_msg="Same seed must produce identical images",
    )


# ---------------------------------------------------------------------------
# T7 — Custom fringe centre respected
# ---------------------------------------------------------------------------
def test_custom_centre():
    """
    Shifting the fringe centre by (10, 10) px must change the image.
    Verifies cx, cy are actually used in radial_profile_to_image.
    """
    params = InstrumentParams()
    r_default = synthesise_calibration_image(
        params, add_noise=False, cx=127.5, cy=127.5
    )
    r_shifted = synthesise_calibration_image(
        params, add_noise=False, cx=137.5, cy=137.5
    )
    assert not np.allclose(r_default["image_2d"], r_shifted["image_2d"]), (
        "Shifting fringe centre had no effect on image"
    )


# ---------------------------------------------------------------------------
# T8 — 1D profile matches direct H01 evaluation
# ---------------------------------------------------------------------------
def test_profile_matches_h01():
    """
    synthesise_calibration_image 1D profile must equal the direct superposition
    of two airy_modified() calls from H01. Tests that H02 is a correct wrapper.

    Uses InstrumentParams object interface (H01 §6 current signature).
    Constants imported from windcube/constants.py, not from H01 module.
    """
    params = InstrumentParams()
    R_bins = 500
    r = np.linspace(0, params.r_max, R_bins)

    A1 = airy_modified(r, NE_WAVELENGTH_1_AIR_M, params)
    A2 = airy_modified(r, NE_WAVELENGTH_2_AIR_M, params)
    expected = A1 + NE_INTENSITY_2 * A2 + params.B

    result = synthesise_calibration_image(params, R_bins=R_bins, add_noise=False)
    np.testing.assert_allclose(
        result["profile_1d"],
        expected,
        rtol=1e-10,
        err_msg="H02 profile does not match direct H01 superposition",
    )
