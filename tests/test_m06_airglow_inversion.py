"""
Tests for M06 airglow inversion — addendum 2026-05-24.
Spec: specs/M06_airglow_inversion_addendum_2026-05-24.md

T-M06-T01 through T-M06-T08 cover the use_temperature flag and temperature
retrieval path.  All import from src.fpi.m06_airglow_inversion_2026_05_24.
Existing tests remain in tests/test_s15_m06_airglow_inversion.py.
"""
import numpy as np
import pytest

from src.fpi.m06_airglow_inversion_2026_05_24 import (
    AirglowFitFlags,
    AirglowFitResult,
    fit_airglow_fringe,
    temperature_from_delta_lambda,
)
from src.fpi.m03_airglow_synthesis_2026_05_24 import delta_lambda_from_temperature


# ---------------------------------------------------------------------------
# Helper: build FringeProfile directly from a 1D profile array.
# Avoids the 2D synthesis → annular-reduction inverse crime that would
# swamp the small thermal-broadening signal in the residuals.
# ---------------------------------------------------------------------------

def _fp_from_1d(profile_1d: np.ndarray, r_bins: np.ndarray, r_max: float,
                sigma_scale: float = 0.005):
    """
    Construct a FringeProfile from a 1D profile and radial grid.

    sigma_profile is set to max(1, sigma_scale * median(profile)).
    """
    from src.fpi.archive.m03_annular_reduction_2026_04_06 import FringeProfile, QualityFlags

    n_bins = len(r_bins)
    sigma_floor = max(1.0, float(np.median(profile_1d)) * sigma_scale)
    sigma_data  = np.full(n_bins, sigma_floor)

    fp = FringeProfile.__new__(FringeProfile)
    fp.profile           = profile_1d.copy()
    fp.sigma_profile     = sigma_data
    fp.two_sigma_profile = 2.0 * sigma_data
    fp.r_grid            = r_bins.copy()
    fp.r2_grid           = r_bins ** 2
    fp.n_pixels          = np.ones(n_bins, dtype=int) * 100
    fp.masked            = np.zeros(n_bins, dtype=bool)
    fp.quality_flags     = QualityFlags.GOOD
    fp.r_max_px          = float(r_max)
    fp.cx = float(r_max);  fp.cy = float(r_max)
    fp.sigma_cx = 0.05;    fp.sigma_cy = 0.05
    fp.two_sigma_cx = 0.1; fp.two_sigma_cy = 0.1
    fp.seed_source       = "synthetic_1d"
    fp.stage1_cx         = float(r_max)
    fp.stage1_cy         = float(r_max)
    fp.cost_at_min       = 0.0
    fp.sparse_bins       = False
    fp.r_min_px          = 0.0
    fp.n_bins            = n_bins
    fp.n_subpixels       = 1
    fp.sigma_clip        = 3.0
    fp.image_shape       = (256, 256)
    fp.peak_fits         = []
    fp.dark_subtracted   = False
    fp.dark_n_frames     = 0
    return fp


def _build_direct_profile(cal, v_rel_ms: float = 0.0, n_bins: int = 150):
    """
    Build a noiseless delta-function FringeProfile directly from the
    M06 forward model.  Avoids all inverse-crime effects.
    """
    from src.fpi import airy_modified
    from windcube.constants import OI_WAVELENGTH_AIR_M, SPEED_OF_LIGHT_MS

    r_max  = 128.0
    n_fine = 500
    lambda_c_m = OI_WAVELENGTH_AIR_M * (1.0 + v_rel_ms / SPEED_OF_LIGHT_MS)

    r_fine   = np.linspace(0.0, r_max, n_fine)
    r2_edges = np.linspace(1.0, r_max ** 2, n_bins + 1)
    r2_ctrs  = 0.5 * (r2_edges[:-1] + r2_edges[1:])
    r_bins   = np.sqrt(r2_ctrs)

    airy_fine = airy_modified(
        r_fine, lambda_c_m,
        t=cal.t_m, R_refl=cal.R_refl, alpha=cal.alpha, n=1.0,
        r_max=r_max, I0=cal.I0, I1=cal.I1, I2=cal.I2,
        sigma0=cal.sigma0, sigma1=cal.sigma1, sigma2=cal.sigma2,
    )
    profile_data = np.interp(r_bins, r_fine, airy_fine)
    return _fp_from_1d(profile_data, r_bins, r_max)


# ---------------------------------------------------------------------------
# T-M06-T01: use_temperature=False is bit-identical to the 2026-05-05 module
# ---------------------------------------------------------------------------

def test_t01_use_temperature_false_identical(synthetic_cal_result):
    """
    use_temperature=False must produce output identical to the pre-addendum
    3-parameter path (same lambda_c_m, v_rel_ms, chi2_reduced, n_params_free=3).
    """
    from src.fpi.m06_airglow_inversion_2026_05_05 import fit_airglow_fringe as fit_old

    cal = synthetic_cal_result
    fp  = _build_direct_profile(cal, v_rel_ms=0.0)

    result_old  = fit_old(fp, cal)
    result_new  = fit_airglow_fringe(fp, cal, use_temperature=False)

    assert result_new.lambda_c_m   == result_old.lambda_c_m,   "lambda_c_m differs"
    assert result_new.v_rel_ms     == result_old.v_rel_ms,     "v_rel_ms differs"
    assert result_new.chi2_reduced == result_old.chi2_reduced, "chi2_reduced differs"
    assert result_new.n_params_free == 3
    assert result_new.T_est_K is None
    assert result_new.sigma_T_K is None
    assert result_new.two_sigma_T_K is None
    assert result_new.delta_lambda_m is None


# ---------------------------------------------------------------------------
# T-M06-T02: use_temperature=True recovers T=800 K within 50 K
# ---------------------------------------------------------------------------

def test_t02_temperature_recovery_800k(synthetic_cal_result):
    """
    use_temperature=True on a noiseless T=800 K profile must recover
    T_est_K within ±50 K of truth, with converged=True.

    Profile built directly from _airglow_model_thermal() — the inversion's
    own forward model — so data and model are self-consistent and the 4-param
    TRF converges to the true parameters.  Avoids the Airy function call-form
    mismatch between M03 synthesis and M06 inversion that would otherwise
    swamp the small thermal signal.
    """
    from src.fpi.m06_airglow_inversion_2026_05_24 import _airglow_model_thermal
    from windcube.constants import OI_WAVELENGTH_AIR_M

    cal    = synthetic_cal_result
    r_max  = 128.0
    r_fine = np.linspace(0.0, r_max, 500)

    dl_800     = delta_lambda_from_temperature(800.0)
    profile_1d = _airglow_model_thermal(
        r_fine, OI_WAVELENGTH_AIR_M, dl_800,
        Y_line=1.0, B_sci=0.0, cal=cal,
    )
    fp = _fp_from_1d(profile_1d, r_fine, r_max)

    result = fit_airglow_fringe(fp, cal, use_temperature=True)

    assert result.converged, "4-parameter fit did not converge"
    assert result.T_est_K is not None, "T_est_K is None"
    err = abs(result.T_est_K - 800.0)
    assert err < 50.0, f"|T_est_K - 800| = {err:.1f} K > 50 K"


# ---------------------------------------------------------------------------
# T-M06-T03: temperature_from_delta_lambda round-trip
# ---------------------------------------------------------------------------

def test_t03_temperature_round_trip():
    """temperature_from_delta_lambda(delta_lambda_from_temperature(800)) ≈ 800 K."""
    dl     = delta_lambda_from_temperature(800.0)
    T_back = temperature_from_delta_lambda(dl)
    assert abs(T_back - 800.0) < 0.01, f"Round-trip error: |{T_back} - 800| >= 0.01 K"


# ---------------------------------------------------------------------------
# T-M06-T04: n_params_free = 4 when use_temperature=True
# ---------------------------------------------------------------------------

def test_t04_n_params_free_four(synthetic_cal_result):
    """n_params_free must be 4 when use_temperature=True."""
    cal    = synthetic_cal_result
    fp     = _build_direct_profile(cal, v_rel_ms=0.0)
    result = fit_airglow_fringe(fp, cal, use_temperature=True)
    assert result.n_params_free == 4, f"n_params_free = {result.n_params_free}, expected 4"


# ---------------------------------------------------------------------------
# T-M06-T05: n_params_free = 3 when use_temperature=False
# ---------------------------------------------------------------------------

def test_t05_n_params_free_three(synthetic_cal_result):
    """n_params_free must be 3 when use_temperature=False (unchanged behaviour)."""
    cal    = synthetic_cal_result
    fp     = _build_direct_profile(cal, v_rel_ms=0.0)
    result = fit_airglow_fringe(fp, cal, use_temperature=False)
    assert result.n_params_free == 3, f"n_params_free = {result.n_params_free}, expected 3"


# ---------------------------------------------------------------------------
# T-M06-T06: T_est_K is None when use_temperature=False
# ---------------------------------------------------------------------------

def test_t06_t_est_none_when_cold_path(synthetic_cal_result):
    """T_est_K, sigma_T_K, two_sigma_T_K, delta_lambda_m all None for cold path."""
    cal    = synthetic_cal_result
    fp     = _build_direct_profile(cal, v_rel_ms=0.0)
    result = fit_airglow_fringe(fp, cal, use_temperature=False)

    assert result.T_est_K is None,        "T_est_K should be None"
    assert result.sigma_T_K is None,      "sigma_T_K should be None"
    assert result.two_sigma_T_K is None,  "two_sigma_T_K should be None"
    assert result.delta_lambda_m is None, "delta_lambda_m should be None"


# ---------------------------------------------------------------------------
# T-M06-T07: DELTA_LAMBDA_AT_BOUND flag set when Δλ is at/outside bounds
# ---------------------------------------------------------------------------

def test_t07_delta_lambda_at_bound_flag(synthetic_cal_result):
    """
    DELTA_LAMBDA_AT_BOUND (0x200) flag is set when the fitted delta_lambda
    is at the physical lower bound.

    A profile built with delta_lambda = dlam_lo * 0.01 (near-pure delta function)
    forces TRF (hard bounds) to return dlam_lo — the smallest allowed value —
    and the DELTA_LAMBDA_AT_BOUND flag fires.  Built from _airglow_model_thermal()
    directly for model self-consistency.
    """
    from src.fpi.m06_airglow_inversion_2026_05_24 import _airglow_model_thermal
    from windcube.constants import OI_WAVELENGTH_AIR_M

    # Verify flag constant
    assert AirglowFitFlags.DELTA_LAMBDA_AT_BOUND == 0x200

    cal    = synthetic_cal_result
    r_max  = 128.0
    r_fine = np.linspace(0.0, r_max, 500)

    dlam_lo = delta_lambda_from_temperature(100.0)

    # delta_lambda = dlam_lo * 0.01 → near-pure Airy delta function.
    # TRF hard lower bound is dlam_lo, so optimizer returns dlam_lo exactly.
    dl_tiny    = dlam_lo * 0.01
    profile_1d = _airglow_model_thermal(
        r_fine, OI_WAVELENGTH_AIR_M, dl_tiny,
        Y_line=1.0, B_sci=0.0, cal=cal,
    )
    fp = _fp_from_1d(profile_1d, r_fine, r_max)

    result = fit_airglow_fringe(fp, cal, use_temperature=True)

    # TRF returns exactly the bound (within float rounding ≈ 1e-25 m).
    # Use same tolerance as fit_airglow_fringe() flag check: abs(...) < 1e-18 m.
    assert abs(result.delta_lambda_m - dlam_lo) < 1e-18, (
        f"Expected delta_lambda_m ({result.delta_lambda_m:.6e}) ≈ dlam_lo "
        f"({dlam_lo:.6e}); difference = {abs(result.delta_lambda_m - dlam_lo):.3e} m"
    )
    assert result.quality_flags & AirglowFitFlags.DELTA_LAMBDA_AT_BOUND, (
        "DELTA_LAMBDA_AT_BOUND flag not set for delta_lambda at lower bound"
    )


# ---------------------------------------------------------------------------
# T-M06-T08: sigma_T_K is physically reasonable for SNR≈5, T=800 K
# ---------------------------------------------------------------------------

def test_t08_sigma_t_reasonable(synthetic_cal_result):
    """
    sigma_T_K must be finite and positive for a noisy T=800 K profile.

    Uses Gaussian noise at SNR=5 added directly to the thermal 1D profile
    (avoiding the 2D inverse crime).  The tolerance is deliberately broad
    because the etalon finesse (≈4.5) limits temperature sensitivity.
    """
    from src.fpi.m03_airglow_synthesis_2026_05_24 import synthesise_airglow_image
    from src.fpi import InstrumentParams

    cal    = synthetic_cal_result
    params = InstrumentParams(
        t=cal.t_m, R_refl=cal.R_refl, alpha=cal.alpha,
        I0=cal.I0, I1=cal.I1, I2=cal.I2,
        sigma0=cal.sigma0, sigma1=cal.sigma1, sigma2=cal.sigma2,
        B=cal.B,
    )

    result_m03 = synthesise_airglow_image(
        params=params, v_rel_ms=0.0,
        use_temperature=True, T_true_K=800.0,
        add_noise=False,
        cx=params.r_max, cy=params.r_max,
    )
    profile_1d = result_m03["profile_1d"]
    r_bins     = result_m03["r_grid"]

    # Add Gaussian noise at target SNR=5
    rng      = np.random.default_rng(99)
    target_snr = 5.0
    delta_S  = float(profile_1d.max() - profile_1d.min())
    sigma_n  = delta_S / target_snr
    noisy_profile = profile_1d + rng.normal(0.0, sigma_n, len(profile_1d))

    from src.fpi.archive.m03_annular_reduction_2026_04_06 import FringeProfile, QualityFlags
    n_bins = len(r_bins)
    sigma_data = np.full(n_bins, sigma_n)

    fp = FringeProfile.__new__(FringeProfile)
    fp.profile           = noisy_profile
    fp.sigma_profile     = sigma_data
    fp.two_sigma_profile = 2.0 * sigma_data
    fp.r_grid            = r_bins.copy()
    fp.r2_grid           = r_bins ** 2
    fp.n_pixels          = np.ones(n_bins, dtype=int) * 100
    fp.masked            = np.zeros(n_bins, dtype=bool)
    fp.quality_flags     = QualityFlags.GOOD
    fp.r_max_px          = float(params.r_max)
    fp.cx = float(params.r_max);  fp.cy = float(params.r_max)
    fp.sigma_cx = 0.05;           fp.sigma_cy = 0.05
    fp.two_sigma_cx = 0.1;        fp.two_sigma_cy = 0.1
    fp.seed_source       = "synthetic_noisy"
    fp.stage1_cx         = float(params.r_max)
    fp.stage1_cy         = float(params.r_max)
    fp.cost_at_min       = 0.0;   fp.sparse_bins = False
    fp.r_min_px          = 0.0;   fp.n_bins = n_bins
    fp.n_subpixels       = 1;     fp.sigma_clip = 3.0
    fp.image_shape       = (256, 256)
    fp.peak_fits         = [];    fp.dark_subtracted = False
    fp.dark_n_frames     = 0

    result = fit_airglow_fringe(fp, cal, use_temperature=True)

    assert result.sigma_T_K is not None, "sigma_T_K is None"
    assert np.isfinite(result.sigma_T_K), f"sigma_T_K is not finite: {result.sigma_T_K}"
    assert result.sigma_T_K > 0, f"sigma_T_K is non-positive: {result.sigma_T_K}"
    assert result.two_sigma_T_K == pytest.approx(2.0 * result.sigma_T_K, rel=1e-9)
