"""
Tests for Z03 synthetic calibration image generator.

Spec:   specs/z03_synthetic_calibration_image_generator_spec_2026-04-14.md
Module: src/fpi/z03_synthetic_calibration_image_generator.py

12 tests:
  T1   test_image_shape              — synthesise_image returns (260, 276)
  T2   test_image_dtype              — synthesise_image returns float64
  T3   test_image_values_positive    — noise-free image values > 0
  T4   test_derive_secondary         — alpha and I0 computed correctly
  T5   test_snr_to_ipeak             — quadratic formula gives correct I0
  T6   test_check_psf_positive       — PSF positivity check correct
  T7   test_check_vignetting_positive — vignetting positivity check correct
  T8   test_synthesise_profile_shape — profile has R_BINS elements
  T9   test_truth_json_complete      — truth JSON has all 11 user_param keys
  T10  test_output_files_exist       — cal and dark .bin files written
  T11  test_psf_broadening_effect    — non-zero sigma0 broadens fringes
  T12  test_vignetting_effect        — non-zero I1 creates radial gradient
"""

import json
import pathlib
import sys

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path
# ---------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.fpi.z03_synthetic_calibration_image_generator_2026_04_14 import (
    SynthParams,
    DerivedParams,
    derive_secondary,
    synthesise_image,
    synthesise_profile,
    build_instrument_params,
    check_psf_positive,
    check_vignetting_positive,
    snr_to_ipeak,
    write_truth_json,
    SIGMA_READ,
    R_MAX_PX,
    R_BINS,
    NROWS,
    NCOLS,
    LAM_640,
    PIX_M,
)


# ---------------------------------------------------------------------------
# Shared fixture: default SynthParams
# ---------------------------------------------------------------------------

@pytest.fixture
def default_params() -> SynthParams:
    """SynthParams with spec-default values."""
    return SynthParams(
        d_mm=20.106, f_mm=199.12, R=0.53,
        sigma0=0.5, sigma1=0.1, sigma2=-0.05,
        snr_peak=50.0, I1=-0.1, I2=0.005, B_dc=300.0, rel_638=0.8,
    )


@pytest.fixture
def default_derived(default_params) -> DerivedParams:
    """DerivedParams from default_params."""
    return derive_secondary(default_params)


# ---------------------------------------------------------------------------
# T1 — synthesise_image returns correct shape
# ---------------------------------------------------------------------------

def test_image_shape(default_params, default_derived):
    """synthesise_image must return shape (NROWS, NCOLS) = (260, 276)."""
    img = synthesise_image(default_params, default_derived)
    assert img.shape == (NROWS, NCOLS), (
        f"Expected shape ({NROWS}, {NCOLS}), got {img.shape}"
    )


# ---------------------------------------------------------------------------
# T2 — synthesise_image returns float64
# ---------------------------------------------------------------------------

def test_image_dtype(default_params, default_derived):
    """synthesise_image must return float64 (noise-free)."""
    img = synthesise_image(default_params, default_derived)
    assert img.dtype == np.float64, f"Expected float64, got {img.dtype}"


# ---------------------------------------------------------------------------
# T3 — noise-free image values are positive
# ---------------------------------------------------------------------------

def test_image_values_positive(default_params, default_derived):
    """All pixel values in the noise-free image must be > 0."""
    img = synthesise_image(default_params, default_derived)
    assert float(img.min()) > 0.0, f"Min pixel = {float(img.min()):.3f}, expected > 0"


# ---------------------------------------------------------------------------
# T4 — derive_secondary computes alpha and I0 correctly
# ---------------------------------------------------------------------------

def test_derive_secondary(default_params):
    """alpha = pix_m / f_m; I0 > B_dc (sanity check)."""
    derived = derive_secondary(default_params)
    expected_alpha = PIX_M / (default_params.f_mm * 1e-3)
    assert abs(derived.alpha_rad_per_px - expected_alpha) < 1e-12, (
        f"alpha = {derived.alpha_rad_per_px:.4e}, expected {expected_alpha:.4e}"
    )
    assert derived.I0 > default_params.B_dc, (
        f"I0 ({derived.I0:.1f}) should exceed B_dc ({default_params.B_dc})"
    )


# ---------------------------------------------------------------------------
# T5 — snr_to_ipeak gives correct quadratic root
# ---------------------------------------------------------------------------

def test_snr_to_ipeak():
    """Verify the SNR quadratic root: SNR^2 * I_peak + SNR^2 * noise = I_peak^2."""
    snr = 50.0
    B_dc = 300.0
    I0 = snr_to_ipeak(snr, B_dc, SIGMA_READ)
    # Check positive root: I_peak^2 - snr^2 * I_peak - snr^2 * (B + sigma^2) = 0
    noise_floor = B_dc + SIGMA_READ ** 2
    residual = I0 ** 2 - snr ** 2 * I0 - snr ** 2 * noise_floor
    assert abs(residual) < 1e-3, (
        f"Quadratic residual = {residual:.4e}, expected ~0"
    )
    assert I0 > 0, "I_peak must be positive"


# ---------------------------------------------------------------------------
# T6 — check_psf_positive
# ---------------------------------------------------------------------------

def test_check_psf_positive():
    """PSF positivity: sigma0 >= sqrt(sigma1^2 + sigma2^2)."""
    # Should pass: sigma0 = 0.5 >= sqrt(0.1^2 + 0.05^2) ≈ 0.112
    assert check_psf_positive(0.5, 0.1, -0.05) is True
    # Should pass: all zero
    assert check_psf_positive(0.0, 0.0, 0.0) is True
    # Should fail: sigma0 < magnitude of variation terms
    assert check_psf_positive(0.1, 0.5, 0.5) is False


# ---------------------------------------------------------------------------
# T7 — check_vignetting_positive
# ---------------------------------------------------------------------------

def test_check_vignetting_positive():
    """Vignetting envelope I(r) must be > 0 for all r in [0, r_max]."""
    I0 = 1000.0
    # Mild vignetting — should pass
    assert check_vignetting_positive(I0, -0.1, 0.005, R_MAX_PX) is True
    # Extreme negative vignetting — I(r_max) = I0*(1 - 1.0) = 0 — should fail
    assert check_vignetting_positive(I0, -1.0, 0.0, R_MAX_PX) is False


# ---------------------------------------------------------------------------
# T8 — synthesise_profile returns correct shape
# ---------------------------------------------------------------------------

def test_synthesise_profile_shape(default_params, default_derived):
    """synthesise_profile must return a profile of length R_BINS."""
    inst_params = build_instrument_params(default_params, default_derived)
    profile_1d, r_grid = synthesise_profile(inst_params, default_params.rel_638)
    assert len(profile_1d) == R_BINS, (
        f"Profile length = {len(profile_1d)}, expected {R_BINS}"
    )
    assert len(r_grid) == R_BINS
    # Profile should include the bias
    assert float(profile_1d.min()) >= default_params.B_dc * 0.9, (
        f"Min profile value {float(profile_1d.min()):.1f} seems too low"
    )


# ---------------------------------------------------------------------------
# T9 — truth JSON has all 11 user_param keys and required structure
# ---------------------------------------------------------------------------

def test_truth_json_complete(default_params, default_derived, tmp_path):
    """Truth JSON must contain all 11 user_params keys and required top-level keys."""
    path_cal  = tmp_path / "cal.bin"
    path_dark = tmp_path / "dark.bin"
    truth_path = tmp_path / "truth.json"

    write_truth_json(default_params, default_derived, 12345, path_cal, path_dark, truth_path)

    with open(truth_path) as f:
        truth = json.load(f)

    expected_user_keys = {
        "d_mm", "f_mm", "R", "sigma0", "sigma1", "sigma2",
        "snr_peak", "I1", "I2", "B_dc", "rel_638"
    }
    assert expected_user_keys == set(truth["user_params"].keys()), (
        f"user_params keys mismatch: {set(truth['user_params'].keys())}"
    )
    assert "output_cal_file"  in truth
    assert "output_dark_file" in truth
    assert "finesse_N"        in truth["derived_params"]
    assert truth["z03_version"] == "1.2"


# ---------------------------------------------------------------------------
# T10 — cal and dark .bin files are written with correct size
# ---------------------------------------------------------------------------

def test_output_files_exist(default_params, default_derived, tmp_path):
    """synthesise_image + noise model produces correct-size .bin output."""
    import numpy as np

    I_float = synthesise_image(default_params, default_derived)

    rng = np.random.default_rng(42)
    signal_counts = rng.poisson(np.clip(I_float, 0, None)).astype(np.float64)
    read_noise    = rng.standard_normal(size=signal_counts.shape) * SIGMA_READ
    image_cal     = np.clip(signal_counts + read_noise, 0, 16383).astype(np.uint16)

    dark_float  = np.full((NROWS, NCOLS), default_params.B_dc, dtype=np.float64)
    dark_counts = rng.poisson(dark_float).astype(np.float64)
    dark_read   = rng.standard_normal(size=dark_float.shape) * SIGMA_READ
    image_dark  = np.clip(dark_counts + dark_read, 0, 16383).astype(np.uint16)

    cal_path  = tmp_path / "cal.bin"
    dark_path = tmp_path / "dark.bin"
    image_cal.tofile(str(cal_path))
    image_dark.tofile(str(dark_path))

    expected_bytes = NROWS * NCOLS * 2  # uint16
    assert cal_path.stat().st_size  == expected_bytes
    assert dark_path.stat().st_size == expected_bytes


# ---------------------------------------------------------------------------
# T11 — PSF broadening effect
# ---------------------------------------------------------------------------

def test_psf_broadening_effect(default_params):
    """Non-zero sigma0 must produce broader fringes than sigma0=0."""
    params_sharp = SynthParams(
        d_mm=20.106, f_mm=199.12, R=0.53,
        sigma0=0.0, sigma1=0.0, sigma2=0.0,
        snr_peak=50.0, I1=0.0, I2=0.0, B_dc=300.0, rel_638=0.8,
    )
    params_broad = SynthParams(
        d_mm=20.106, f_mm=199.12, R=0.53,
        sigma0=2.0, sigma1=0.0, sigma2=0.0,
        snr_peak=50.0, I1=0.0, I2=0.0, B_dc=300.0, rel_638=0.8,
    )

    derived_sharp = derive_secondary(params_sharp)
    derived_broad = derive_secondary(params_broad)

    inst_sharp = build_instrument_params(params_sharp, derived_sharp)
    inst_broad = build_instrument_params(params_broad, derived_broad)

    prof_sharp, r_grid = synthesise_profile(inst_sharp, 0.8)
    prof_broad, _      = synthesise_profile(inst_broad, 0.8)

    # PSF broadening should smooth (reduce std of) the fringe profile
    assert np.std(prof_broad) < np.std(prof_sharp), (
        "PSF broadening should smooth (reduce std of) the fringe profile; "
        f"std_sharp={np.std(prof_sharp):.2f}, std_broad={np.std(prof_broad):.2f}"
    )


# ---------------------------------------------------------------------------
# T12 — Vignetting effect
# ---------------------------------------------------------------------------

def test_vignetting_effect(default_params):
    """Non-zero I1 must produce a measurable radial intensity gradient."""
    # Flat illumination (I1 = I2 = 0)
    params_flat = SynthParams(
        d_mm=20.106, f_mm=199.12, R=0.53,
        sigma0=0.5, sigma1=0.0, sigma2=0.0,
        snr_peak=50.0, I1=0.0, I2=0.0, B_dc=300.0, rel_638=0.8,
    )
    # Strong vignetting (I1 = -0.4)
    params_vig = SynthParams(
        d_mm=20.106, f_mm=199.12, R=0.53,
        sigma0=0.5, sigma1=0.0, sigma2=0.0,
        snr_peak=50.0, I1=-0.4, I2=0.0, B_dc=300.0, rel_638=0.8,
    )

    derived_flat = derive_secondary(params_flat)
    derived_vig  = derive_secondary(params_vig)

    img_flat = synthesise_image(params_flat, derived_flat)
    img_vig  = synthesise_image(params_vig,  derived_vig)

    cx, cy = 137, 129
    r_half = 55   # half of r_max

    def annulus_mean(img, r_inner, r_outer):
        rows, cols = np.ogrid[:img.shape[0], :img.shape[1]]
        r_map = np.sqrt((cols - cx) ** 2 + (rows - cy) ** 2)
        mask  = (r_map >= r_inner) & (r_map < r_outer)
        return float(img[mask].mean())

    inner_flat = annulus_mean(img_flat, 0, r_half)
    outer_flat = annulus_mean(img_flat, r_half, 110)
    inner_vig  = annulus_mean(img_vig,  0, r_half)
    outer_vig  = annulus_mean(img_vig,  r_half, 110)

    ratio_flat = inner_flat / outer_flat
    ratio_vig  = inner_vig  / outer_vig

    # Vignetting should make the inner/outer ratio more extreme
    assert ratio_vig > ratio_flat * 1.01, (
        f"Vignetting (I1=-0.4) should reduce outer vs inner; "
        f"ratio_flat={ratio_flat:.4f}, ratio_vig={ratio_vig:.4f}"
    )
