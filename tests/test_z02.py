"""
Tests for Z02 synthetic science image generator (v1.0).

Spec:   specs/z02_synthetic_science_image_generator_2026-04-29.md
Module: validation/z02_synthetic_science_image_generator.py

25 tests:
  Carry-forward (adapted from Z03 v1.5):
    test_binning_config_binned
    test_binning_config_unbinned
    test_I0_single_line               I0 = I_peak (no rel_638 denominator)
    test_derive_secondary
    test_check_vignetting_positive
    test_sci_output_shape_binned
    test_dark_output_shape_binned
    test_sci_output_shape_unbinned
    test_dark_output_shape_unbinned
    test_sci_header_round_trip        img_type='science'
    test_dark_header_round_trip
    test_fringe_peak_location
    test_snr_achieved
    test_dark_no_fringes
    test_truth_json_complete          v1.0 schema; v_los_ms present, rel_638 absent
    test_default_params
    test_round_trip_I0                self-consistency: profile peak ≈ I0
    test_alpha_no_f_mm
    test_binning_config_binned        (already counted above)
    test_cx_cy_offset_binned
    test_cx_cy_offset_unbinned
    test_filename_label

  Carry-forward (new in Z03 v1.5, retained):
    test_no_rel_638_param             SynthParams has no rel_638
    test_dark_current_scales_with_temperature
    test_offset_present_in_dark
    test_finesse_from_R
    test_1row_header

  New in Z02 (2):
    test_blueshift_moves_fringes_outward
    test_redshift_moves_fringes_inward
"""

import json
import math
import pathlib
import sys

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ensure repo root on sys.path
# ---------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from validation.z02_synthetic_science_image_generator import (
    BinningConfig,
    DerivedParams,
    C_LIGHT_MS,
    DARK_REF_ADU_S,
    LAM_OI,
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
# Shared helpers
# ---------------------------------------------------------------------------

def make_default_params(binning: int = 2) -> SynthParams:
    """Return SynthParams with Z02 v1.0 default values."""
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
        v_los_ms=-7500.0,
    )


def _synth_files(tmp_path: pathlib.Path, binning: int = 2, seed: int = 42):
    """Synthesise and write a sci+dark pair; return (sci_path, dark_path, truth_path)."""
    params = make_default_params(binning)
    sci_path, dark_path, truth_path, _ = run_synthesis(params, tmp_path, seed=seed)
    return sci_path, dark_path, truth_path


def _load_raw(path: pathlib.Path) -> np.ndarray:
    """Load a .bin file as big-endian uint16 flat array."""
    return np.frombuffer(path.read_bytes(), dtype=">u2")


def _load_frame(path: pathlib.Path, nrows: int, ncols: int) -> np.ndarray:
    """Load a .bin file as (nrows, ncols) big-endian uint16 array."""
    return _load_raw(path).reshape(nrows, ncols)


# ---------------------------------------------------------------------------
# ── TESTS ─────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def test_binning_config_binned():
    """BinningConfig for binning=2 must match all §6 table values."""
    cfg = get_binning_config(2)
    assert cfg.nrows         == 260
    assert cfg.ncols         == 276
    assert cfg.active_rows   == 259
    assert cfg.n_meta_rows   == 1
    assert cfg.cx_default    == pytest.approx(137.5)
    assert cfg.cy_default    == pytest.approx(130.0)
    assert cfg.r_max_px      == pytest.approx(110.0)
    assert cfg.alpha_default == pytest.approx(1.6000e-4)
    assert cfg.pix_m         == pytest.approx(32.0e-6)
    assert cfg.label         == "2x2_binned"


def test_binning_config_unbinned():
    """BinningConfig for binning=1 must match all §6 table values."""
    cfg = get_binning_config(1)
    assert cfg.nrows         == 528
    assert cfg.ncols         == 552
    assert cfg.active_rows   == 527
    assert cfg.n_meta_rows   == 1
    assert cfg.cx_default    == pytest.approx(275.5)
    assert cfg.cy_default    == pytest.approx(264.0)
    assert cfg.r_max_px      == pytest.approx(220.0)
    assert cfg.alpha_default == pytest.approx(0.8000e-4)
    assert cfg.pix_m         == pytest.approx(16.0e-6)
    assert cfg.label         == "1x1_unbinned"


def test_I0_single_line():
    """I0_adu = snr_to_ipeak(snr_peak, OFFSET_ADU); I0 = I_peak (no rel_638 division)."""
    params  = make_default_params()
    derived = derive_secondary(params)

    I_peak_expected = snr_to_ipeak(params.snr_peak, OFFSET_ADU)

    assert abs(derived.I_peak - I_peak_expected) / I_peak_expected < 1e-6, (
        f"I_peak mismatch: {derived.I_peak} vs {I_peak_expected}"
    )
    assert abs(derived.I0 - derived.I_peak) / derived.I_peak < 1e-6, (
        f"I0 should equal I_peak for single-line source: I0={derived.I0}, I_peak={derived.I_peak}"
    )
    assert abs(derived.I0 - I_peak_expected) / I_peak_expected < 1e-6, (
        f"I0 mismatch: {derived.I0} vs {I_peak_expected}"
    )


def test_derive_secondary():
    """derive_secondary returns correct alpha, I0, finesse_N_R, dark_rate, lambda_obs."""
    params  = make_default_params()
    derived = derive_secondary(params)

    assert derived.alpha_rad_per_px == pytest.approx(params.alpha)

    I_peak = snr_to_ipeak(params.snr_peak, OFFSET_ADU)
    assert derived.I0 == pytest.approx(I_peak, rel=1e-6)

    N_R_expected = math.pi * math.sqrt(params.R) / (1.0 - params.R)
    assert derived.finesse_N_R == pytest.approx(N_R_expected, rel=1e-6)

    dark_expected = DARK_REF_ADU_S * 2.0**((params.T_fp_c - T_REF_DARK_C) / T_DOUBLE_C)
    assert derived.dark_rate == pytest.approx(dark_expected, rel=1e-6)

    lam_obs_expected = LAM_OI * (1.0 + params.v_los_ms / C_LIGHT_MS)
    assert derived.lambda_obs == pytest.approx(lam_obs_expected, rel=1e-9)
    assert derived.delta_lam  == pytest.approx(lam_obs_expected - LAM_OI, rel=1e-6)


def test_check_vignetting_positive():
    """Vignetting envelope I(r) must be > 0 for r ∈ [0, r_max]."""
    I0 = 1000.0
    assert check_vignetting_positive(I0, -0.1, 0.005, 110.0) is True
    assert check_vignetting_positive(I0, -1.0,  0.0,  110.0) is False


def test_sci_output_shape_binned(tmp_path):
    """Sci .bin file loads to (260, 276) uint16 for binning=2."""
    sci_path, _, _ = _synth_files(tmp_path, binning=2)
    data = _load_frame(sci_path, 260, 276)
    assert data.shape == (260, 276)
    assert data.dtype.kind in ("u", "i")


def test_dark_output_shape_binned(tmp_path):
    """Dark .bin file loads to (260, 276) uint16 for binning=2."""
    _, dark_path, _ = _synth_files(tmp_path, binning=2)
    data = _load_frame(dark_path, 260, 276)
    assert data.shape == (260, 276)


def test_sci_output_shape_unbinned(tmp_path):
    """Sci .bin file loads to (528, 552) uint16 for binning=1."""
    sci_path, _, _ = _synth_files(tmp_path, binning=1)
    data = _load_frame(sci_path, 528, 552)
    assert data.shape == (528, 552)


def test_dark_output_shape_unbinned(tmp_path):
    """Dark .bin file loads to (528, 552) uint16 for binning=1."""
    _, dark_path, _ = _synth_files(tmp_path, binning=1)
    data = _load_frame(dark_path, 528, 552)
    assert data.shape == (528, 552)


def test_sci_header_round_trip(tmp_path):
    """1-row header is recoverable via P01 parse_header(); img_type='science'."""
    from src.metadata.p01_image_metadata_2026_04_06 import parse_header

    sci_path, _, _ = _synth_files(tmp_path, binning=2)
    raw    = _load_raw(sci_path)
    hrow   = raw[:276].astype(">u2")
    parsed = parse_header(hrow)

    assert parsed["rows"]           == 260
    assert parsed["cols"]           == 276
    assert parsed["img_type"]       == "science"
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
    """Centroid of top-5% brightest pixels in the sci image is within 1 px of (cx, cy)."""
    params   = make_default_params(binning=2)
    cfg      = get_binning_config(2)
    sci_path = run_synthesis(params, tmp_path, seed=42)[0]

    data       = _load_frame(sci_path, cfg.nrows, cfg.ncols).astype(float)
    pixel_data = data[cfg.n_meta_rows:, :]

    cen_row, cen_col = _centroid_of_bright_pixels(pixel_data, pct=95.0)
    cen_row_full = cen_row + cfg.n_meta_rows

    assert abs(cen_col - params.cx) < 1.0, (
        f"Column centroid {cen_col:.2f} not within 1 px of cx={params.cx}"
    )
    assert abs(cen_row_full - params.cy) < 1.0, (
        f"Row centroid {cen_row_full:.2f} not within 1 px of cy={params.cy}"
    )


def test_snr_achieved(tmp_path):
    """Peak SNR of the synthesized sci image is within ±20% of the requested SNR."""
    params   = make_default_params(binning=2)
    cfg      = get_binning_config(2)
    sci_path = run_synthesis(params, tmp_path, seed=42)[0]

    data       = _load_frame(sci_path, cfg.nrows, cfg.ncols).astype(float)
    pixel_data = data[cfg.n_meta_rows:, :]

    bg_level  = float(np.percentile(pixel_data, 5))
    peak_val  = float(np.max(pixel_data))
    I_peak    = peak_val - bg_level
    noise_est = math.sqrt(max(I_peak + OFFSET_ADU, 1.0))
    achieved  = I_peak / noise_est if noise_est > 0 else 0.0

    assert abs(achieved - params.snr_peak) / params.snr_peak < 0.20, (
        f"Achieved SNR={achieved:.1f} vs requested={params.snr_peak} (>20% error)"
    )


def test_dark_no_fringes(tmp_path):
    """Dark frame has no periodic fringe structure (std much less than sci frame)."""
    params              = make_default_params(binning=2)
    cfg                 = get_binning_config(2)
    sci_path, dark_path, *_ = run_synthesis(params, tmp_path, seed=42)

    sci_data  = _load_frame(sci_path,  cfg.nrows, cfg.ncols).astype(float)
    dark_data = _load_frame(dark_path, cfg.nrows, cfg.ncols).astype(float)

    sci_std  = float(np.std(sci_data[cfg.n_meta_rows:, :]))
    dark_std = float(np.std(dark_data[cfg.n_meta_rows:, :]))

    assert dark_std < sci_std * 0.1, (
        f"Dark std {dark_std:.2f} should be << sci std {sci_std:.2f} (no fringes)"
    )


def test_truth_json_complete(tmp_path):
    """Truth JSON must contain all v1.0 required keys and must NOT contain removed keys."""
    _, _, truth_path = _synth_files(tmp_path, binning=2)
    with open(truth_path) as f:
        truth = json.load(f)

    expected_user_keys = {
        "binning", "cx", "cy",
        "d_mm", "alpha", "R",
        "snr_peak", "I1", "I2", "T_fp_c",
        "v_los_ms",
    }
    expected_derived_keys = {
        "alpha_rad_per_px", "lambda_obs_m", "delta_lambda_m",
        "I_peak_adu", "I0_adu",
        "finesse_N_R", "finesse_coefficient_F", "FSR_m", "dark_rate_adu_px_s",
    }
    expected_fixed_keys = {
        "lam_oi_m", "c_light_ms",
        "offset_adu", "dark_ref_adu_s", "T_ref_dark_c", "T_double_c",
        "R_bins", "n_ref", "n_meta_rows", "nrows", "ncols",
        "active_rows", "r_max_px", "pix_m", "label",
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

    # Keys that must NOT appear anywhere in the truth JSON
    for absent in ("rel_638", "lam_640_m", "lam_638_m", "sigma0", "sigma1",
                   "sigma2", "B_dc", "f_mm", "sigma_read", "Y_B"):
        assert absent not in truth["user_params"], \
            f"Unexpected key '{absent}' in user_params"
        assert absent not in truth.get("fixed_constants", {}), \
            f"Unexpected key '{absent}' in fixed_constants"

    assert truth["z02_version"] == "1.0"
    assert "output_sci_file"  in truth
    assert "output_dark_file" in truth
    assert truth["fixed_constants"]["n_meta_rows"] == 1
    # I0_adu must equal I_peak_adu (single-line source)
    assert truth["derived_params"]["I0_adu"] == pytest.approx(
        truth["derived_params"]["I_peak_adu"], rel=1e-6
    )


def test_default_params(tmp_path):
    """Script runs with all defaults — both .bin and _truth.json are written."""
    params = make_default_params(binning=2)
    sci_path, dark_path, truth_path, _ = run_synthesis(params, tmp_path, seed=12345)

    assert sci_path.exists(),   f"sci .bin not found at {sci_path}"
    assert dark_path.exists(),  f"dark .bin not found at {dark_path}"
    assert truth_path.exists(), f"truth JSON not found at {truth_path}"

    cfg = get_binning_config(2)
    expected_bytes = cfg.nrows * cfg.ncols * 2
    assert sci_path.stat().st_size  == expected_bytes
    assert dark_path.stat().st_size == expected_bytes


def test_round_trip_I0():
    """Profile max above offset is within 15% of derived I0 (single-line self-consistency)."""
    params  = make_default_params(binning=2)
    derived = derive_secondary(params)

    profile_1d, _ = synthesise_profile(params, derived)
    peak_above_offset = float(profile_1d.max()) - OFFSET_ADU

    assert abs(peak_above_offset - derived.I0) / derived.I0 < 0.15, (
        f"Profile peak above offset {peak_above_offset:.1f} "
        f"not within 15% of I0={derived.I0:.1f}"
    )


def test_alpha_no_f_mm():
    """SynthParams must have 'alpha' field and must NOT have 'f_mm' field."""
    params = make_default_params()
    user_keys = set(params.__dataclass_fields__.keys())
    assert "alpha" in user_keys,    "SynthParams must have 'alpha' field"
    assert "f_mm"  not in user_keys, "SynthParams must NOT have 'f_mm' field"


def test_cx_cy_offset_binned(tmp_path):
    """Fringe centre displaced +10 px in both axes; centroid within 1 px of (cx+10, cy+10)."""
    cfg = get_binning_config(2)
    params = SynthParams(
        binning=2,
        cx=cfg.cx_default + 10.0,
        cy=cfg.cy_default + 10.0,
        d_mm=20.0005, alpha=cfg.alpha_default, R=0.725,
        snr_peak=50.0, I1=-0.1, I2=0.005, T_fp_c=-20.0, v_los_ms=-7500.0,
    )
    sci_path = run_synthesis(params, tmp_path, seed=42)[0]

    data       = _load_frame(sci_path, cfg.nrows, cfg.ncols).astype(float)
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
        snr_peak=50.0, I1=-0.1, I2=0.005, T_fp_c=-20.0, v_los_ms=-7500.0,
    )
    sci_path = run_synthesis(params, tmp_path, seed=42)[0]

    data       = _load_frame(sci_path, cfg.nrows, cfg.ncols).astype(float)
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
    """Output filenames must contain the mode label and '_sci_synth_z02_' / '_dark_synth_z02_'."""
    cfg = get_binning_config(2)
    params = make_default_params(binning=2)
    sci_path, dark_path, *_ = run_synthesis(params, tmp_path, seed=42)

    assert cfg.label in sci_path.name, (
        f"Mode label '{cfg.label}' not found in sci filename '{sci_path.name}'"
    )
    assert cfg.label in dark_path.name, (
        f"Mode label '{cfg.label}' not found in dark filename '{dark_path.name}'"
    )
    assert "_sci_synth_z02_"  in sci_path.name
    assert "_dark_synth_z02_" in dark_path.name


def test_no_rel_638_param():
    """SynthParams must NOT have a 'rel_638' field."""
    params = make_default_params()
    assert not hasattr(params, "rel_638"), (
        "SynthParams should not have 'rel_638' field (Z02 uses v_los_ms instead)"
    )
    with pytest.raises(AttributeError):
        _ = params.rel_638


def test_dark_current_scales_with_temperature():
    """Dark rate at −10°C must be at least 1.5× higher than at −20°C."""
    rate_cold = DARK_REF_ADU_S * 2.0**((-20.0 - T_REF_DARK_C) / T_DOUBLE_C)
    rate_warm = DARK_REF_ADU_S * 2.0**((-10.0 - T_REF_DARK_C) / T_DOUBLE_C)
    assert rate_warm > rate_cold * 1.5, (
        f"Dark rate at -10°C ({rate_warm:.6f}) should be >1.5× rate at -20°C ({rate_cold:.6f})"
    )


def test_offset_present_in_dark(tmp_path):
    """Dark frame minimum ≥ 4 ADU (electronic offset floor) at very short exposure."""
    cfg    = get_binning_config(2)
    params = SynthParams(
        binning=2,
        cx=cfg.cx_default, cy=cfg.cy_default,
        d_mm=20.0005, alpha=cfg.alpha_default, R=0.725,
        snr_peak=50.0, I1=-0.1, I2=0.005,
        T_fp_c=-20.0,   # coldest → negligible dark current
        v_los_ms=-7500.0,
    )
    _, dark_path, *_ = run_synthesis(params, tmp_path, seed=42, exp_time_cts=1)

    dark_data   = _load_frame(dark_path, cfg.nrows, cfg.ncols)
    dark_pixels = dark_data[cfg.n_meta_rows:, :]

    assert int(np.min(dark_pixels)) >= 4, (
        f"Dark minimum {int(np.min(dark_pixels))} ADU is below 4 (OFFSET_ADU floor)"
    )


def test_finesse_from_R():
    """N_R computed from default R = 0.725 must be in the range (9.0, 10.5)."""
    R   = 0.725
    N_R = math.pi * math.sqrt(R) / (1.0 - R)
    assert 9.0 < N_R < 10.5, (
        f"Finesse N_R = {N_R:.4f} for R={R} is outside expected range (9.0, 10.5)"
    )


def test_1row_header(tmp_path):
    """File row 0 is a valid 1-row header; row 1 (first pixel row) contains signal."""
    cfg      = get_binning_config(2)
    sci_path = _synth_files(tmp_path, binning=2)[0]

    raw = _load_raw(sci_path)
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

    pixel_row1 = raw[cfg.ncols : 2 * cfg.ncols]
    assert np.any(pixel_row1 > 10), (
        "Expected some pixel values > 10 in the first pixel row (row 1)"
    )


# ---------------------------------------------------------------------------
# ── NEW Z02 TESTS: Doppler fringe-shift direction ─────────────────────────
# ---------------------------------------------------------------------------

def _make_profile(v_los_ms: float) -> tuple[np.ndarray, np.ndarray]:
    """Synthesise a noise-free 1D profile for a given v_los_ms; return (profile_1d, r_grid)."""
    params  = make_default_params(binning=2)
    params  = SynthParams(
        binning=params.binning, cx=params.cx, cy=params.cy,
        d_mm=params.d_mm, alpha=params.alpha, R=params.R,
        snr_peak=params.snr_peak, I1=params.I1, I2=params.I2,
        T_fp_c=params.T_fp_c, v_los_ms=v_los_ms,
    )
    derived = derive_secondary(params)
    return synthesise_profile(params, derived)


def _first_bright_ring_radius(profile_1d: np.ndarray, r_grid: np.ndarray) -> float:
    """Return the radius of the outermost bright fringe peak (excluding r=0 region)."""
    from scipy.signal import find_peaks as _find_peaks
    bg    = float(np.percentile(profile_1d, 5))
    span  = profile_1d.max() - bg
    pks, _ = _find_peaks(profile_1d, height=bg + 0.3 * span, distance=10)
    assert len(pks) > 0, "No bright fringes found in profile"
    return float(r_grid[pks[-1]])


def test_blueshift_moves_fringes_outward():
    """Blueshift (negative v_los) → λ_obs decreases → phase increases → fringes shift outward."""
    profile_rest,  r_grid = _make_profile(v_los_ms=0.0)
    profile_blue,  _      = _make_profile(v_los_ms=-7500.0)

    r_peak_rest = _first_bright_ring_radius(profile_rest, r_grid)
    r_peak_blue = _first_bright_ring_radius(profile_blue, r_grid)

    assert r_peak_blue > r_peak_rest, (
        f"Blueshift should move fringes outward: "
        f"r_peak_blue={r_peak_blue:.3f} px, r_peak_rest={r_peak_rest:.3f} px"
    )


def test_redshift_moves_fringes_inward():
    """Redshift (positive v_los) → λ_obs increases → phase decreases → fringes shift inward."""
    profile_rest, r_grid = _make_profile(v_los_ms=0.0)
    profile_red,  _      = _make_profile(v_los_ms=500.0)

    r_peak_rest = _first_bright_ring_radius(profile_rest, r_grid)
    r_peak_red  = _first_bright_ring_radius(profile_red,  r_grid)

    assert r_peak_red < r_peak_rest, (
        f"Redshift should move fringes inward: "
        f"r_peak_red={r_peak_red:.3f} px, r_peak_rest={r_peak_rest:.3f} px"
    )
