"""
Tests for Z03 synthetic calibration image generator.

Spec:   specs/z03_synthetic_calibration_image_generator_spec_2026-04-22.md
Module: validation/z03_synthetic_calibration_image_generator_2026_04_14.py

15 tests:
  T1   test_image_shape              — synthesise_image returns (260, 276)
  T2   test_image_dtype              — synthesise_image returns float64
  T3   test_image_values_positive    — noise-free image values > 0
  T4   test_derive_secondary         — alpha direct and I0 computed correctly
  T5   test_snr_to_ipeak             — quadratic formula gives correct I0
  T6   test_check_psf_positive       — PSF positivity check correct
  T7   test_check_vignetting_positive — vignetting positivity check correct
  T8   test_synthesise_profile_shape — profile has R_BINS elements
  T9   test_truth_json_complete      — truth JSON has all 11 user_param keys (v1.3)
  T10  test_output_files_exist       — cal and dark .bin files written
  T11  test_psf_broadening_effect    — non-zero sigma0 broadens fringes
  T12  test_vignetting_effect        — non-zero I1 creates radial gradient
  T13  test_I0_option_a              — I0_adu = I_peak / (1 + rel_638)
  T14  test_round_trip_I0            — Z03 truth I0 matches F01 recovered I0 within 5%
  T15  test_alpha_no_f_mm            — f_mm not present in SynthParams or derived_params
"""

import json
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path
# ---------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from validation.z03_synthetic_calibration_image_generator_2026_04_14 import (
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
# Shared fixture: default SynthParams (v1.3 defaults)
# ---------------------------------------------------------------------------

@pytest.fixture
def default_params() -> SynthParams:
    """SynthParams with spec v1.3 default values."""
    return SynthParams(
        d_mm=20.0006, alpha=1.6133e-4, R=0.53,
        sigma0=0.5, sigma1=0.1, sigma2=-0.05,
        snr_peak=50.0, I1=-0.1, I2=0.005, B_dc=300.0, rel_638=0.58,
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
# T4 — derive_secondary uses alpha directly and I0 is per-line
# ---------------------------------------------------------------------------

def test_derive_secondary(default_params):
    """alpha_rad_per_px == params.alpha (direct); I0 = I_peak/(1+rel_638)."""
    derived = derive_secondary(default_params)
    assert derived.alpha_rad_per_px == default_params.alpha, (
        f"alpha_rad_per_px={derived.alpha_rad_per_px:.6e}, "
        f"expected {default_params.alpha:.6e}"
    )
    assert derived.I0 > default_params.B_dc, (
        f"I0 ({derived.I0:.1f}) should exceed B_dc ({default_params.B_dc})"
    )
    I_peak_expected = snr_to_ipeak(
        default_params.snr_peak, default_params.B_dc, SIGMA_READ
    )
    I0_expected = I_peak_expected / (1.0 + default_params.rel_638)
    assert abs(derived.I0 - I0_expected) / I0_expected < 1e-10, (
        f"I0={derived.I0:.4f} vs expected={I0_expected:.4f}"
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
    assert check_psf_positive(0.5, 0.1, -0.05) is True
    assert check_psf_positive(0.0, 0.0, 0.0) is True
    assert check_psf_positive(0.1, 0.5, 0.5) is False


# ---------------------------------------------------------------------------
# T7 — check_vignetting_positive
# ---------------------------------------------------------------------------

def test_check_vignetting_positive():
    """Vignetting envelope I(r) must be > 0 for all r in [0, r_max]."""
    I0 = 1000.0
    assert check_vignetting_positive(I0, -0.1, 0.005, R_MAX_PX) is True
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
    assert float(profile_1d.min()) >= default_params.B_dc * 0.9, (
        f"Min profile value {float(profile_1d.min()):.1f} seems too low"
    )


# ---------------------------------------------------------------------------
# T9 — truth JSON has all user_param keys and required structure (v1.3)
# ---------------------------------------------------------------------------

def test_truth_json_complete(default_params, default_derived, tmp_path):
    """Truth JSON must contain alpha (not f_mm), I0_adu, Y_B, and version 1.3."""
    path_cal  = tmp_path / "cal.bin"
    path_dark = tmp_path / "dark.bin"
    truth_path = tmp_path / "truth.json"

    write_truth_json(default_params, default_derived, 12345, path_cal, path_dark, truth_path)

    with open(truth_path) as f:
        truth = json.load(f)

    expected_user_keys = {
        "d_mm", "alpha", "R", "sigma0", "sigma1", "sigma2",
        "snr_peak", "I1", "I2", "B_dc", "rel_638"
    }
    assert expected_user_keys == set(truth["user_params"].keys()), (
        f"user_params keys mismatch: {set(truth['user_params'].keys())}"
    )
    expected_derived_keys = {
        "alpha_rad_per_px", "I_peak_adu", "I0_adu", "Y_B",
        "FSR_m", "finesse_coefficient_F", "finesse_N"
    }
    assert expected_derived_keys == set(truth["derived_params"].keys()), (
        f"derived_params keys mismatch: {set(truth['derived_params'].keys())}"
    )
    assert "f_mm" not in truth["user_params"]
    assert "f_mm" not in truth["derived_params"]
    assert "output_cal_file"  in truth
    assert "output_dark_file" in truth
    assert truth["z03_version"] == "1.3"


# ---------------------------------------------------------------------------
# T10 — cal and dark .bin files are written with correct size
# ---------------------------------------------------------------------------

def test_output_files_exist(default_params, default_derived, tmp_path):
    """synthesise_image + noise model produces correct-size .bin output."""
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
        d_mm=20.0006, alpha=1.6133e-4, R=0.53,
        sigma0=0.0, sigma1=0.0, sigma2=0.0,
        snr_peak=50.0, I1=0.0, I2=0.0, B_dc=300.0, rel_638=0.58,
    )
    params_broad = SynthParams(
        d_mm=20.0006, alpha=1.6133e-4, R=0.53,
        sigma0=2.0, sigma1=0.0, sigma2=0.0,
        snr_peak=50.0, I1=0.0, I2=0.0, B_dc=300.0, rel_638=0.58,
    )

    derived_sharp = derive_secondary(params_sharp)
    derived_broad = derive_secondary(params_broad)

    inst_sharp = build_instrument_params(params_sharp, derived_sharp)
    inst_broad = build_instrument_params(params_broad, derived_broad)

    prof_sharp, r_grid = synthesise_profile(inst_sharp, 0.58)
    prof_broad, _      = synthesise_profile(inst_broad, 0.58)

    assert np.std(prof_broad) < np.std(prof_sharp), (
        "PSF broadening should smooth (reduce std of) the fringe profile; "
        f"std_sharp={np.std(prof_sharp):.2f}, std_broad={np.std(prof_broad):.2f}"
    )


# ---------------------------------------------------------------------------
# T12 — Vignetting effect
# ---------------------------------------------------------------------------

def test_vignetting_effect(default_params):
    """Non-zero I1 must produce a measurable radial intensity gradient."""
    params_flat = SynthParams(
        d_mm=20.0006, alpha=1.6133e-4, R=0.53,
        sigma0=0.5, sigma1=0.0, sigma2=0.0,
        snr_peak=50.0, I1=0.0, I2=0.0, B_dc=300.0, rel_638=0.58,
    )
    params_vig = SynthParams(
        d_mm=20.0006, alpha=1.6133e-4, R=0.53,
        sigma0=0.5, sigma1=0.0, sigma2=0.0,
        snr_peak=50.0, I1=-0.4, I2=0.0, B_dc=300.0, rel_638=0.58,
    )

    derived_flat = derive_secondary(params_flat)
    derived_vig  = derive_secondary(params_vig)

    img_flat = synthesise_image(params_flat, derived_flat)
    img_vig  = synthesise_image(params_vig,  derived_vig)

    cx, cy = 137, 129
    r_half = 55   # half of r_max (110 px)

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

    assert ratio_vig > ratio_flat * 1.01, (
        f"Vignetting (I1=-0.4) should reduce outer vs inner; "
        f"ratio_flat={ratio_flat:.4f}, ratio_vig={ratio_vig:.4f}"
    )


# ---------------------------------------------------------------------------
# T13 — I0 Option A: I0_adu = I_peak / (1 + rel_638)
# ---------------------------------------------------------------------------

def test_I0_option_a():
    """derive_secondary must set I0 = I_peak/(1+rel_638) to 6 significant figures."""
    params = SynthParams(
        d_mm=20.0006, alpha=1.6133e-4, R=0.53,
        sigma0=0.5, sigma1=0.0, sigma2=0.0,
        snr_peak=50.0, I1=-0.1, I2=0.005,
        B_dc=300.0, rel_638=0.58,
    )
    derived = derive_secondary(params)
    I_peak_expected = snr_to_ipeak(50.0, 300.0, SIGMA_READ)
    I0_expected     = I_peak_expected / (1.0 + 0.58)
    assert abs(derived.I0 - I0_expected) / I0_expected < 1e-6, (
        f"I0 mismatch: {derived.I0} vs {I0_expected}"
    )
    assert abs(derived.I_peak - I_peak_expected) / I_peak_expected < 1e-6, (
        f"I_peak mismatch: {derived.I_peak} vs {I_peak_expected}"
    )


# ---------------------------------------------------------------------------
# T14 — Round-trip I0: Z03 truth I0 matches F01 fitted I0 within 5%
# ---------------------------------------------------------------------------

def test_round_trip_I0():
    """F01 must recover I0 within 5% of Z03 truth I0 on a synthetic profile."""
    from src.fpi.f01_full_airy_fit_to_neon_image_2026_04_22 import (
        fit_neon_fringe, TolanskyResult,
    )

    params = SynthParams(
        d_mm=20.0006, alpha=1.6133e-4, R=0.53,
        sigma0=0.5, sigma1=0.0, sigma2=0.0,
        snr_peak=50.0, I1=-0.1, I2=0.005,
        B_dc=300.0, rel_638=0.58,
    )
    derived  = derive_secondary(params)
    inst     = build_instrument_params(params, derived)
    profile_1d, r_grid = synthesise_profile(inst, params.rel_638)

    rng   = np.random.default_rng(42)
    noisy = rng.poisson(np.maximum(profile_1d, 1)).astype(np.float32)
    sigma = np.maximum(np.sqrt(noisy) / 8.0, 1.0).astype(np.float32)

    fringe = SimpleNamespace(
        r_grid=r_grid.astype(np.float32),
        r2_grid=(r_grid ** 2).astype(np.float32),
        profile=noisy,
        sigma_profile=sigma,
        masked=np.zeros(len(r_grid), dtype=bool),
        r_max_px=float(R_MAX_PX),
        quality_flags=0,
    )
    tolansky = TolanskyResult(
        t_m=20.0006e-3, alpha_rpx=1.6133e-4,
        epsilon_640=0.7735, epsilon_638=0.2711, epsilon_cal=0.22,
    )
    result = fit_neon_fringe(fringe, tolansky)

    truth_I0 = derived.I0
    assert abs(result.I0 - truth_I0) / truth_I0 < 0.05, (
        f"F01 I0={result.I0:.1f} vs Z03 truth I0={truth_I0:.1f} (>{5}% error)"
    )


# ---------------------------------------------------------------------------
# T15 — alpha present, f_mm absent in SynthParams and derived_params
# ---------------------------------------------------------------------------

def test_alpha_no_f_mm():
    """SynthParams must have 'alpha' field and no 'f_mm' field."""
    params = SynthParams(
        d_mm=20.0006, alpha=1.6133e-4, R=0.53,
        sigma0=0.5, sigma1=0.0, sigma2=0.0,
        snr_peak=50.0, I1=0.0, I2=0.0,
        B_dc=300.0, rel_638=0.58,
    )
    derived = derive_secondary(params)

    user_keys = set(params.__dataclass_fields__.keys())
    assert "alpha" in user_keys, "SynthParams must have 'alpha' field"
    assert "f_mm" not in user_keys, "SynthParams must NOT have 'f_mm' field"
    assert derived.alpha_rad_per_px == params.alpha, (
        f"derived.alpha_rad_per_px={derived.alpha_rad_per_px} != params.alpha={params.alpha}"
    )
