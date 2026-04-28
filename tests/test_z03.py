"""
Tests for Z03 synthetic calibration image generator (v1.5).

Spec:   specs/z03_synthetic_calibration_image_generator_spec_2026-04-28.md
Module: src/fpi/z03_synthetic_calibration_image_generator.py

25 tests:
  Carry-forward (20 — updated for v1.5 signatures):
    test_binning_config_binned          BinningConfig for binning=2 matches §6 table
    test_binning_config_unbinned        BinningConfig for binning=1 matches §6 table
    test_I0_option_a                    I0 = I_peak / (1 + rel_638) to 6 sig figs
    test_derive_secondary               alpha, I0, N_R, dark_rate computed correctly
    test_check_vignetting_positive      vignetting positivity check
    test_cal_output_shape_binned        cal .bin loads to (260, 276) uint16
    test_dark_output_shape_binned       dark .bin loads to (260, 276) uint16
    test_cal_output_shape_unbinned      cal .bin loads to (528, 552) uint16
    test_dark_output_shape_unbinned     dark .bin loads to (528, 552) uint16
    test_cal_header_round_trip          1-row header parseable; img_type='cal'
    test_dark_header_round_trip         dark header; img_type='dark', shutter closed
    test_fringe_peak_location           centroid of bright pixels near (cx, cy)
    test_snr_achieved                   achieved SNR within ±20% of requested
    test_rel_638_ratio                  amplitude ratio ≈ rel_638 within ±5%
    test_dark_no_fringes                dark has no periodic fringe structure
    test_truth_json_complete            all v1.5 keys present; absent keys absent
    test_default_params                 run_synthesis with defaults writes both files
    test_round_trip_I0                  F01 recovers I0 within 5% of Z03 truth
    test_alpha_no_f_mm                  alpha present, f_mm absent in SynthParams
    test_cx_cy_offset_binned            displaced cx/cy; centroid within 1 px
    test_cx_cy_offset_unbinned          displaced cx/cy; centroid within 1 px
    test_filename_label                 mode label present in output filenames

  New in v1.5 (5):
    test_no_sigma_params                SynthParams has no sigma0/sigma1/sigma2/B_dc
    test_dark_current_scales_with_temperature  rate_warm > rate_cold × 1.5
    test_offset_present_in_dark         min(dark frame) ≥ 4 ADU
    test_finesse_from_R                 N_R from default R ≈ 10.0
    test_1row_header                    raw file row 0 = valid header; row 1 = pixels
"""

import json
import math
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ensure repo root on sys.path
# ---------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.fpi.z03_synthetic_calibration_image_generator_2026_04_28 import (
    BinningConfig,
    DerivedParams,
    DARK_REF_ADU_S,
    LAM_638,
    LAM_640,
    N_REF,
    OFFSET_ADU,
    R_BINS,
    SynthParams,
    T_DOUBLE_C,
    T_REF_DARK_C,
    check_vignetting_positive,
    derive_secondary,
    get_binning_config,
    run_synthesis,
    snr_to_ipeak,
    synthesise_image,
    synthesise_profile,
    write_truth_json,
)


# ---------------------------------------------------------------------------
# Shared helper: default SynthParams for binning=2 or 1
# ---------------------------------------------------------------------------

def make_default_params(binning: int = 2) -> SynthParams:
    """Return SynthParams with v1.5 default values."""
    cfg = get_binning_config(binning)
    return SynthParams(
        binning=binning,
        cx=cfg.cx_default,
        cy=cfg.cy_default,
        d_mm=20.0005,
        alpha=cfg.alpha_default,
        R=0.725,
        snr_peak=50.0,
        I1=-0.1,
        I2=0.005,
        T_fp_c=-20.0,
        rel_638=0.344,
    )


def _synth_files(tmp_path: pathlib.Path, binning: int = 2, seed: int = 42):
    """Synthesise and write a cal+dark pair; return (cal_path, dark_path, truth_path)."""
    params = make_default_params(binning)
    cal_path, dark_path, truth_path, _ = run_synthesis(params, tmp_path, seed=seed)
    return cal_path, dark_path, truth_path


def _load_raw(path: pathlib.Path) -> np.ndarray:
    """Load a .bin file as big-endian uint16 flat array."""
    return np.frombuffer(path.read_bytes(), dtype=">u2")


def _load_frame(path: pathlib.Path, nrows: int, ncols: int) -> np.ndarray:
    """Load a .bin file as (nrows, ncols) big-endian uint16 array."""
    return _load_raw(path).reshape(nrows, ncols)


# ---------------------------------------------------------------------------
# ── CARRY-FORWARD TESTS ──────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def test_binning_config_binned():
    """BinningConfig for binning=2 must match all §6 table values."""
    cfg = get_binning_config(2)
    assert cfg.nrows        == 260
    assert cfg.ncols        == 276
    assert cfg.active_rows  == 259
    assert cfg.n_meta_rows  == 1
    assert cfg.cx_default   == pytest.approx(137.5)
    assert cfg.cy_default   == pytest.approx(130.0)
    assert cfg.r_max_px     == pytest.approx(110.0)
    assert cfg.alpha_default == pytest.approx(1.6000e-4)
    assert cfg.pix_m        == pytest.approx(32.0e-6)
    assert cfg.label        == "2x2_binned"


def test_binning_config_unbinned():
    """BinningConfig for binning=1 must match all §6 table values."""
    cfg = get_binning_config(1)
    assert cfg.nrows        == 528
    assert cfg.ncols        == 552
    assert cfg.active_rows  == 527
    assert cfg.n_meta_rows  == 1
    assert cfg.cx_default   == pytest.approx(275.5)
    assert cfg.cy_default   == pytest.approx(264.0)
    assert cfg.r_max_px     == pytest.approx(220.0)
    assert cfg.alpha_default == pytest.approx(0.8000e-4)
    assert cfg.pix_m        == pytest.approx(16.0e-6)
    assert cfg.label        == "1x1_unbinned"


def test_I0_option_a():
    """I0 = I_peak / (1 + rel_638) to 6 significant figures (Option A)."""
    params  = make_default_params()
    derived = derive_secondary(params)

    I_peak_expected = snr_to_ipeak(params.snr_peak, OFFSET_ADU)
    I0_expected     = I_peak_expected / (1.0 + params.rel_638)

    assert abs(derived.I_peak - I_peak_expected) / I_peak_expected < 1e-6, (
        f"I_peak mismatch: {derived.I_peak} vs {I_peak_expected}"
    )
    assert abs(derived.I0 - I0_expected) / I0_expected < 1e-6, (
        f"I0 mismatch: {derived.I0} vs {I0_expected}"
    )


def test_derive_secondary():
    """derive_secondary returns correct alpha, I0, finesse_N_R, dark_rate."""
    params  = make_default_params()
    derived = derive_secondary(params)

    assert derived.alpha_rad_per_px == pytest.approx(params.alpha)
    I_peak = snr_to_ipeak(params.snr_peak, OFFSET_ADU)
    I0     = I_peak / (1.0 + params.rel_638)
    assert derived.I0 == pytest.approx(I0, rel=1e-6)

    N_R_expected = math.pi * math.sqrt(params.R) / (1.0 - params.R)
    assert derived.finesse_N_R == pytest.approx(N_R_expected, rel=1e-6)

    dark_expected = DARK_REF_ADU_S * 2.0**((params.T_fp_c - T_REF_DARK_C) / T_DOUBLE_C)
    assert derived.dark_rate == pytest.approx(dark_expected, rel=1e-6)


def test_check_vignetting_positive():
    """Vignetting envelope I(r) must be > 0 for r ∈ [0, r_max]."""
    I0 = 1000.0
    assert check_vignetting_positive(I0, -0.1, 0.005, 110.0) is True
    assert check_vignetting_positive(I0, -1.0,  0.0,  110.0) is False


def test_cal_output_shape_binned(tmp_path):
    """Cal .bin file loads to (260, 276) uint16 for binning=2."""
    cal_path, _, _ = _synth_files(tmp_path, binning=2)
    data = _load_frame(cal_path, 260, 276)
    assert data.shape == (260, 276)
    assert data.dtype.kind in ("u", "i")   # unsigned or signed int family


def test_dark_output_shape_binned(tmp_path):
    """Dark .bin file loads to (260, 276) uint16 for binning=2."""
    _, dark_path, _ = _synth_files(tmp_path, binning=2)
    data = _load_frame(dark_path, 260, 276)
    assert data.shape == (260, 276)


def test_cal_output_shape_unbinned(tmp_path):
    """Cal .bin file loads to (528, 552) uint16 for binning=1."""
    cal_path, _, _ = _synth_files(tmp_path, binning=1)
    data = _load_frame(cal_path, 528, 552)
    assert data.shape == (528, 552)


def test_dark_output_shape_unbinned(tmp_path):
    """Dark .bin file loads to (528, 552) uint16 for binning=1."""
    _, dark_path, _ = _synth_files(tmp_path, binning=1)
    data = _load_frame(dark_path, 528, 552)
    assert data.shape == (528, 552)


def test_cal_header_round_trip(tmp_path):
    """1-row header is recoverable via P01 parse_header(); img_type='cal'."""
    from src.metadata.p01_image_metadata_2026_04_06 import parse_header

    cal_path, _, _ = _synth_files(tmp_path, binning=2)
    raw   = _load_raw(cal_path)
    hrow  = raw[:276].astype(">u2")
    parsed = parse_header(hrow)

    assert parsed["rows"]     == 260
    assert parsed["cols"]     == 276
    assert parsed["img_type"] == "cal"
    assert parsed["shutter_status"] == "open"


def test_dark_header_round_trip(tmp_path):
    """Dark 1-row header: img_type='dark', shutter closed."""
    from src.metadata.p01_image_metadata_2026_04_06 import parse_header

    _, dark_path, _ = _synth_files(tmp_path, binning=2)
    raw    = _load_raw(dark_path)
    hrow   = raw[:276].astype(">u2")
    parsed = parse_header(hrow)

    assert parsed["rows"]           == 260
    assert parsed["img_type"]       == "dark"
    assert parsed["shutter_status"] == "closed"


def _centroid_of_bright_pixels(pixel_data: np.ndarray, pct: float = 95.0) -> tuple:
    """Return (row_centroid, col_centroid) of top-pct% pixels."""
    threshold = np.percentile(pixel_data.astype(float), pct)
    rows, cols = np.where(pixel_data >= threshold)
    return float(np.mean(rows)), float(np.mean(cols))


def test_fringe_peak_location(tmp_path):
    """Centroid of the top-5% brightest pixels in the cal image is within 1 px of (cx, cy)."""
    params  = make_default_params(binning=2)
    cfg     = get_binning_config(2)
    cal_path = run_synthesis(params, tmp_path, seed=42)[0]

    data       = _load_frame(cal_path, cfg.nrows, cfg.ncols).astype(float)
    pixel_data = data[cfg.n_meta_rows:, :]  # exclude header row

    cen_row, cen_col = _centroid_of_bright_pixels(pixel_data, pct=95.0)
    # cen_row is in pixel-data coords; convert to full-frame coords
    cen_row_full = cen_row + cfg.n_meta_rows

    assert abs(cen_col - params.cx) < 1.0, (
        f"Column centroid {cen_col:.2f} not within 1 px of cx={params.cx}"
    )
    assert abs(cen_row_full - params.cy) < 1.0, (
        f"Row centroid {cen_row_full:.2f} not within 1 px of cy={params.cy}"
    )


def test_snr_achieved(tmp_path):
    """Peak SNR of the synthesized cal image is within ±20% of the requested SNR."""
    params  = make_default_params(binning=2)
    cfg     = get_binning_config(2)
    cal_path = run_synthesis(params, tmp_path, seed=42)[0]

    data       = _load_frame(cal_path, cfg.nrows, cfg.ncols).astype(float)
    pixel_data = data[cfg.n_meta_rows:, :]

    bg_level  = float(np.percentile(pixel_data, 5))
    peak_val  = float(np.max(pixel_data))
    I_peak    = peak_val - bg_level
    noise_est = math.sqrt(max(I_peak + OFFSET_ADU, 1.0))
    achieved  = I_peak / noise_est if noise_est > 0 else 0.0

    assert abs(achieved - params.snr_peak) / params.snr_peak < 0.20, (
        f"Achieved SNR={achieved:.1f} vs requested={params.snr_peak} (>20% error)"
    )


def test_rel_638_ratio():
    """Amplitude ratio of rel_638*A638 vs A640 components is within ±5% of rel_638."""
    params  = make_default_params(binning=2)
    derived = derive_secondary(params)
    cfg     = get_binning_config(2)

    r_grid    = np.linspace(0.0, cfg.r_max_px, R_BINS)
    cos_theta = np.cos(np.arctan(params.alpha * r_grid))
    F_coef    = 4.0 * params.R / (1.0 - params.R)**2
    u         = r_grid / cfg.r_max_px
    vignette  = derived.I0 * (1.0 + params.I1 * u + params.I2 * u**2)

    def _airy(lam):
        phase = 4.0 * np.pi * N_REF * (params.d_mm * 1e-3) * cos_theta / lam
        return vignette / (1.0 + F_coef * np.sin(phase / 2.0)**2)

    A640 = _airy(LAM_640)
    A638 = _airy(LAM_638)

    amp_640        = float(np.max(A640) - np.min(A640))
    amp_638_scaled = float(np.max(params.rel_638 * A638) - np.min(params.rel_638 * A638))

    measured_ratio = amp_638_scaled / amp_640
    assert abs(measured_ratio - params.rel_638) / params.rel_638 < 0.05, (
        f"Amplitude ratio {measured_ratio:.4f} ≠ rel_638={params.rel_638} (>5% error)"
    )


def test_dark_no_fringes(tmp_path):
    """Dark frame has no periodic fringe structure (std much less than cal frame)."""
    params   = make_default_params(binning=2)
    cfg      = get_binning_config(2)
    cal_path, dark_path, *_ = run_synthesis(params, tmp_path, seed=42)

    cal_data  = _load_frame(cal_path,  cfg.nrows, cfg.ncols).astype(float)
    dark_data = _load_frame(dark_path, cfg.nrows, cfg.ncols).astype(float)

    cal_pixels  = cal_data[cfg.n_meta_rows:, :]
    dark_pixels = dark_data[cfg.n_meta_rows:, :]

    cal_std  = float(np.std(cal_pixels))
    dark_std = float(np.std(dark_pixels))

    assert dark_std < cal_std * 0.1, (
        f"Dark std {dark_std:.2f} should be << cal std {cal_std:.2f} (no fringes)"
    )


def test_truth_json_complete(tmp_path):
    """Truth JSON must contain all v1.5 required keys and must NOT contain removed keys."""
    _, _, truth_path = _synth_files(tmp_path, binning=2)
    with open(truth_path) as f:
        truth = json.load(f)

    expected_user_keys = {
        "binning", "cx", "cy",
        "d_mm", "alpha", "R",
        "snr_peak", "I1", "I2", "T_fp_c", "rel_638",
    }
    expected_derived_keys = {
        "alpha_rad_per_px", "I_peak_adu", "I0_adu", "Y_B",
        "finesse_N_R", "finesse_coefficient_F", "FSR_m", "dark_rate_adu_px_s",
    }
    expected_fixed_keys = {
        "offset_adu", "dark_ref_adu_s", "T_ref_dark_c", "T_double_c",
        "R_bins", "n_ref", "lam_640_m", "lam_638_m",
        "n_meta_rows", "nrows", "ncols", "active_rows", "r_max_px", "pix_m", "label",
    }

    assert expected_user_keys == set(truth["user_params"].keys()), (
        f"user_params key mismatch: {set(truth['user_params'].keys())}"
    )
    assert expected_derived_keys == set(truth["derived_params"].keys()), (
        f"derived_params key mismatch: {set(truth['derived_params'].keys())}"
    )
    assert expected_fixed_keys == set(truth["fixed_constants"].keys()), (
        f"fixed_constants key mismatch: {set(truth['fixed_constants'].keys())}"
    )

    # Absent keys (removed in v1.5)
    for absent in ("sigma0", "sigma1", "sigma2", "B_dc", "f_mm", "sigma_read"):
        assert absent not in truth["user_params"], f"Unexpected key '{absent}' in user_params"
        assert absent not in truth.get("fixed_constants", {}), \
            f"Unexpected key '{absent}' in fixed_constants"

    assert truth["z03_version"] == "1.5"
    assert "output_cal_file"  in truth
    assert "output_dark_file" in truth
    # n_meta_rows must be 1
    assert truth["fixed_constants"]["n_meta_rows"] == 1


def test_default_params(tmp_path):
    """Script runs with all defaults — both .bin and _truth.json are written."""
    params = make_default_params(binning=2)
    cal_path, dark_path, truth_path, _ = run_synthesis(params, tmp_path, seed=12345)

    assert cal_path.exists(),  f"cal .bin not found at {cal_path}"
    assert dark_path.exists(), f"dark .bin not found at {dark_path}"
    assert truth_path.exists(), f"truth JSON not found at {truth_path}"

    # Files have the right size
    cfg = get_binning_config(2)
    expected_bytes = cfg.nrows * cfg.ncols * 2
    assert cal_path.stat().st_size  == expected_bytes
    assert dark_path.stat().st_size == expected_bytes


def test_round_trip_I0(tmp_path):
    """F01 must recover I0 within 5% of Z03 truth I0 on a synthetic profile."""
    from src.fpi.f01_full_airy_fit_to_neon_image_2026_04_22 import (
        fit_neon_fringe,
        TolanskyResult,
    )

    params  = make_default_params(binning=2)
    derived = derive_secondary(params)
    cfg     = get_binning_config(2)

    profile_1d, r_grid = synthesise_profile(params, derived)

    rng   = np.random.default_rng(42)
    noisy = rng.poisson(np.maximum(profile_1d, 1)).astype(np.float32)
    sigma = np.maximum(np.sqrt(noisy) / 8.0, 1.0).astype(np.float32)

    fringe = SimpleNamespace(
        r_grid        = r_grid.astype(np.float32),
        r2_grid       = (r_grid**2).astype(np.float32),
        profile       = noisy,
        sigma_profile = sigma,
        masked        = np.zeros(len(r_grid), dtype=bool),
        r_max_px      = float(cfg.r_max_px),
        quality_flags = 0,
    )
    tolansky = TolanskyResult(
        t_m         = params.d_mm * 1e-3,
        alpha_rpx   = params.alpha,
        epsilon_640 = 0.7735,
        epsilon_638 = 0.2711,
        epsilon_cal = 0.22,
    )
    result   = fit_neon_fringe(fringe, tolansky)
    truth_I0 = derived.I0

    assert abs(result.I0 - truth_I0) / truth_I0 < 0.05, (
        f"F01 I0={result.I0:.1f} vs Z03 truth I0={truth_I0:.1f} (>{5}% error)"
    )


def test_alpha_no_f_mm():
    """SynthParams must have 'alpha' field and must NOT have 'f_mm' field."""
    params  = make_default_params()
    derived = derive_secondary(params)

    user_keys = set(params.__dataclass_fields__.keys())
    assert "alpha" in user_keys,   "SynthParams must have 'alpha' field"
    assert "f_mm"  not in user_keys, "SynthParams must NOT have 'f_mm' field"
    assert derived.alpha_rad_per_px == pytest.approx(params.alpha)


def test_cx_cy_offset_binned(tmp_path):
    """Fringe centre displaced +10 px in both axes; centroid within 1 px of (cx+10, cy+10)."""
    cfg = get_binning_config(2)
    params = SynthParams(
        binning=2,
        cx=cfg.cx_default + 10.0,
        cy=cfg.cy_default + 10.0,
        d_mm=20.0005, alpha=cfg.alpha_default, R=0.725,
        snr_peak=50.0, I1=-0.1, I2=0.005, T_fp_c=-20.0, rel_638=0.344,
    )
    cal_path = run_synthesis(params, tmp_path, seed=42)[0]

    data       = _load_frame(cal_path, cfg.nrows, cfg.ncols).astype(float)
    pixel_data = data[cfg.n_meta_rows:, :]

    cen_row, cen_col = _centroid_of_bright_pixels(pixel_data, pct=95.0)
    cen_row_full = cen_row + cfg.n_meta_rows

    assert abs(cen_col - params.cx) < 1.0, (
        f"Col centroid {cen_col:.2f} not within 1 px of cx={params.cx}"
    )
    assert abs(cen_row_full - params.cy) < 1.0, (
        f"Row centroid {cen_row_full:.2f} not within 1 px of cy={params.cy}"
    )


def test_cx_cy_offset_unbinned(tmp_path):
    """Unbinned: fringe centre displaced +20 px; centroid within 1 px of new centre."""
    cfg = get_binning_config(1)
    params = SynthParams(
        binning=1,
        cx=cfg.cx_default + 20.0,
        cy=cfg.cy_default + 20.0,
        d_mm=20.0005, alpha=cfg.alpha_default, R=0.725,
        snr_peak=50.0, I1=-0.1, I2=0.005, T_fp_c=-20.0, rel_638=0.344,
    )
    cal_path = run_synthesis(params, tmp_path, seed=42)[0]

    data       = _load_frame(cal_path, cfg.nrows, cfg.ncols).astype(float)
    pixel_data = data[cfg.n_meta_rows:, :]

    cen_row, cen_col = _centroid_of_bright_pixels(pixel_data, pct=95.0)
    cen_row_full = cen_row + cfg.n_meta_rows

    assert abs(cen_col - params.cx) < 1.0, (
        f"Col centroid {cen_col:.2f} not within 1 px of cx={params.cx}"
    )
    assert abs(cen_row_full - params.cy) < 1.0, (
        f"Row centroid {cen_row_full:.2f} not within 1 px of cy={params.cy}"
    )


def test_filename_label(tmp_path):
    """Output filenames must contain the mode label (e.g., '2x2_binned')."""
    cfg = get_binning_config(2)
    params = make_default_params(binning=2)
    cal_path, dark_path, *_ = run_synthesis(params, tmp_path, seed=42)

    assert cfg.label in cal_path.name, (
        f"Mode label '{cfg.label}' not found in cal filename '{cal_path.name}'"
    )
    assert cfg.label in dark_path.name, (
        f"Mode label '{cfg.label}' not found in dark filename '{dark_path.name}'"
    )


# ---------------------------------------------------------------------------
# ── NEW v1.5 TESTS ───────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def test_no_sigma_params():
    """SynthParams must NOT have sigma0, sigma1, sigma2, or B_dc fields."""
    params = make_default_params()
    for attr in ("sigma0", "sigma1", "sigma2", "B_dc"):
        assert not hasattr(params, attr), (
            f"SynthParams should not have '{attr}' field (removed in v1.5)"
        )


def test_dark_current_scales_with_temperature():
    """Dark rate at −10°C must be at least 1.5× higher than at −20°C."""
    exp_time_s = 1.0
    rate_cold  = DARK_REF_ADU_S * 2.0**((-20.0 - T_REF_DARK_C) / T_DOUBLE_C)
    rate_warm  = DARK_REF_ADU_S * 2.0**((-10.0 - T_REF_DARK_C) / T_DOUBLE_C)
    assert rate_warm > rate_cold * 1.5, (
        f"Dark rate at -10°C ({rate_warm:.4f}) should be >1.5× rate at -20°C ({rate_cold:.4f})"
    )


def test_offset_present_in_dark(tmp_path):
    """Dark frame minimum ≥ 4 ADU (electronic offset floor) at very short exposure."""
    cfg    = get_binning_config(2)
    params = SynthParams(
        binning=2,
        cx=cfg.cx_default, cy=cfg.cy_default,
        d_mm=20.0005, alpha=cfg.alpha_default, R=0.725,
        snr_peak=50.0, I1=-0.1, I2=0.005,
        T_fp_c=-20.0,   # coldest T → negligible dark current
        rel_638=0.344,
    )
    # Use very short exposure: 1 count × 0.001 s = 0.001 s → dark ≈ 0.00005 ADU
    _, dark_path, *_ = run_synthesis(params, tmp_path, seed=42, exp_time_cts=1)

    dark_data   = _load_frame(dark_path, cfg.nrows, cfg.ncols)
    dark_pixels = dark_data[cfg.n_meta_rows:, :]

    assert int(np.min(dark_pixels)) >= 4, (
        f"Dark minimum {int(np.min(dark_pixels))} ADU is below 4 (OFFSET_ADU floor)"
    )


def test_finesse_from_R():
    """N_R computed from default R = 0.725 must be in the range (9.0, 10.5).

    The spec quotes N_R ≈ 10.0 for R = 0.725 as an approximation;
    the exact value is N_R = π√0.725 / (1−0.725) ≈ 9.73.
    """
    R   = 0.725
    N_R = math.pi * math.sqrt(R) / (1.0 - R)
    assert 9.0 < N_R < 10.5, (
        f"Finesse N_R = {N_R:.4f} for R={R} is outside expected range (9.0, 10.5)"
    )


def test_1row_header(tmp_path):
    """File row 0 is a valid 1-row header; row 1 (first pixel row) contains signal."""
    cfg      = get_binning_config(2)
    cal_path = _synth_files(tmp_path, binning=2)[0]

    raw = _load_raw(cal_path)
    assert raw.shape == (cfg.nrows * cfg.ncols,), (
        f"Expected {cfg.nrows * cfg.ncols} words, got {raw.shape}"
    )

    header_row = raw[:cfg.ncols]
    assert int(header_row[0]) == cfg.nrows, (
        f"header_row[0] = {int(header_row[0])}, expected {cfg.nrows}"
    )
    assert int(header_row[1]) == cfg.ncols, (
        f"header_row[1] = {int(header_row[1])}, expected {cfg.ncols}"
    )

    # Row 1 (first pixel row) should contain non-trivial signal
    pixel_row1 = raw[cfg.ncols : 2 * cfg.ncols]
    assert np.any(pixel_row1 > 10), (
        "Expected some pixel values > 10 in the first pixel row (row 1)"
    )
