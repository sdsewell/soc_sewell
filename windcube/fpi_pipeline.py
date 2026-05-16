"""
windcube/fpi_pipeline.py — FPI processing chain coordination library.

Spec:        specs/S_L01_fpi_pipeline_2026-05-14.md
Spec date:   2026-05-14
Generated:   2026-05-15
Tool:        Claude Code

Re-exports all core FPI pipeline functions under stable names and provides:
  - CalibrationResult, MasterCalibration dataclasses
  - average_calibrations()      : arithmetic mean of N CalibrationResults
  - process_cal_frame()         : full cal chain in one call
  - process_science_frame()     : full science chain in one call
  - to_h05_fringe_profile()     : FringeProfile → H05 internal type
  - master_cal_to_h06_cal()     : MasterCalibration → H06 internal type

Does NOT write any files to disk.
Does NOT contain interactive (tkinter/matplotlib) code.
Does NOT duplicate any algorithm.
"""

from __future__ import annotations

import importlib.util
import logging
import pathlib
import sys
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# sys.path setup — REPO_ROOT for windcube.* and src.fpi.*;
#                  src/processing for flat-name imports of center_finder,
#                  annular_reduction, H05, and H06 (no __init__.py there).
# ---------------------------------------------------------------------------
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_SRC_PROC = REPO_ROOT / "src" / "processing"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(_SRC_PROC) not in sys.path:
    sys.path.insert(0, str(_SRC_PROC))

# ---------------------------------------------------------------------------
# Tolansky — importlib required because the filename contains a hyphen.
# Registered in sys.modules under the key "tolansky_2026_05_13" so that
# repeated imports are idempotent.
# ---------------------------------------------------------------------------
_tol_path = REPO_ROOT / "src" / "fpi" / "tolansky_2026-05-13.py"
_tol_key  = "tolansky_2026_05_13"
if _tol_key not in sys.modules:
    _tol_spec = importlib.util.spec_from_file_location(_tol_key, str(_tol_path))
    _tol_mod  = importlib.util.module_from_spec(_tol_spec)
    sys.modules[_tol_key] = _tol_mod
    _tol_spec.loader.exec_module(_tol_mod)
else:
    _tol_mod = sys.modules[_tol_key]

run_tolansky_2line      = _tol_mod.run_tolansky_2line        # noqa: F401
to_m05_priors           = _tol_mod.to_m05_priors             # noqa: F401
TolanskyResult          = _tol_mod.TolanskyResult            # noqa: F401
InsufficientRingsError  = _tol_mod.InsufficientRingsError    # noqa: F401
print_rectangular_array = _tol_mod.print_rectangular_array   # noqa: F401
plot_tolansky_result    = _tol_mod.plot_tolansky_result      # noqa: F401

# ---------------------------------------------------------------------------
# Re-exports — centre finding
# center_finder.py lives in src/processing/ (not a package); flat import.
# ---------------------------------------------------------------------------
from center_finder import find_centre, CentreResult              # noqa: F401, E402

# ---------------------------------------------------------------------------
# Re-exports — annular reduction
# annular_reduction.py lives in src/processing/; flat import.
# ---------------------------------------------------------------------------
from annular_reduction import annular_reduce, FringeProfile      # noqa: F401, E402

# ---------------------------------------------------------------------------
# Re-exports — H05 calibration inversion (flat import via _SRC_PROC on path)
# ---------------------------------------------------------------------------
from H05_calibration_inversion_2026_05_12 import (              # noqa: F401, E402
    run_staged_inversion,
    save_cal_result,
    FitResult,
    _FringeProfile as _H05FringeProfile,
)

# ---------------------------------------------------------------------------
# Re-exports — H06 airglow inversion (flat import via _SRC_PROC on path)
# ---------------------------------------------------------------------------
from H06_airglow_inversion_2026_05_14 import (                  # noqa: F401, E402
    run_airglow_inversion,
    AirglowResult,
    load_cal_result,
    _CalResult as _H06CalResult,
)

# ---------------------------------------------------------------------------
# Re-exports — forward model (phase correction for gap seed)
# ---------------------------------------------------------------------------
from src.fpi.airy_forward_model_2026_05_05 import phase_correct_gap  # noqa: F401, E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
from windcube.constants import NE_WAVELENGTH_1_AIR_M            # noqa: F401, E402

log = logging.getLogger("fpi_pipeline")


# ---------------------------------------------------------------------------
# Helper: convert list[PeakFitR2] to the (N, 9) float64 array that
# run_tolansky_2line() / load_and_split_families() expects.
# ---------------------------------------------------------------------------

def _peak_fits_r2_to_array(peak_fits_r2: list) -> np.ndarray:
    """
    Convert list[PeakFitR2] from annular_reduce() to the (N, 9) float64
    array expected by run_tolansky_2line().

    Column mapping (matches _peak_fits_r2.npy format from annular_reduction.py):
        col 0: peak_num         (1-based sequential index)
        col 1: r2_raw_px2
        col 2: r2_fit_px2       (nan if fit_ok is False)
        col 3: sigma_r2_fit_px2 (nan if fit_ok is False)
        col 4: r_fit_derived    sqrt(r2_fit_px2)  (nan if fit_ok is False)
        col 5: sigma_r_derived  sigma_r2_fit_px2 / (2*sqrt(r2_fit_px2))
        col 6: amplitude_adu
        col 7: width_r2_px2     (nan if fit_ok is False)
        col 8: reduced_chi2     (nan if fit_ok is False)
    """
    rows = []
    for i, p in enumerate(peak_fits_r2):
        if p.fit_ok and p.r2_fit_px2 > 0:
            r2_fit = float(p.r2_fit_px2)
            sig_r2 = float(p.sigma_r2_fit_px2)
            r_fit  = float(np.sqrt(r2_fit))
            sig_r  = sig_r2 / (2.0 * r_fit)
            width  = float(p.width_r2_px2)
            chi2   = float(p.reduced_chi2)
        else:
            r2_fit = np.nan
            sig_r2 = np.nan
            r_fit  = np.nan
            sig_r  = np.nan
            width  = np.nan
            chi2   = np.nan
        rows.append([
            float(i + 1),
            float(p.r2_raw_px2),
            r2_fit,
            sig_r2,
            r_fit,
            sig_r,
            float(p.amplitude_adu),
            width,
            chi2,
        ])
    if not rows:
        raise ValueError("peak_fits_r2 is empty — no peaks detected in profile.")
    return np.array(rows, dtype=np.float64)


# ---------------------------------------------------------------------------
# §2.1 CalibrationResult — wraps H05 FitResult into a stable public type
# ---------------------------------------------------------------------------

@dataclass
class CalibrationResult:
    """
    Output of a single calibration frame inversion (H05).

    All parameters are the H05 10-parameter fit result for one
    dark-subtracted neon calibration frame. The 1σ uncertainty on each
    fitted parameter is stored alongside the value.

    This is the type averaged by average_calibrations() to produce
    a MasterCalibration.
    """
    # ── Etalon geometry ──────────────────────────────────────────────────
    t_m:          float   # etalon gap [m]
    sigma_t_m:    float   # 1σ uncertainty [m]
    alpha:        float   # plate scale [rad/px, 2×2 binned]
    sigma_alpha:  float   # 1σ uncertainty [rad/px]

    # ── Reflectivities ───────────────────────────────────────────────────
    R_refl:       float   # effective reflectivity at λ₁=640.2 nm → used by H06
    sigma_R_refl: float
    R2:           float   # reflectivity at λ₂=638.3 nm (reference only)
    sigma_R2:     float

    # ── Intensity model ──────────────────────────────────────────────────
    I0:           float   # mean intensity [ADU]
    sigma_I0:     float
    I1:           float   # linear vignetting coefficient
    sigma_I1:     float
    I2:           float   # quadratic vignetting coefficient
    sigma_I2:     float

    # ── PSF ─────────────────────────────────────────────────────────────
    sigma0:       float   # PSF base width [px]
    sigma_sigma0: float
    sigma1:       float   # PSF sin variation [px] — fixed=0 in H05
    sigma2:       float   # PSF cos variation [px] — fixed=0 in H05

    # ── Background ───────────────────────────────────────────────────────
    B:            float   # CCD bias pedestal [ADU]
    sigma_B:      float

    # ── Lamp ratio ───────────────────────────────────────────────────────
    ne_ratio:       float   # λ₂/λ₁ intensity ratio (fitted)
    sigma_ne_ratio: float

    # ── Zero-wind phase reference ─────────────────────────────────────────
    epsilon_cal:       float   # fractional interference order at centre
    sigma_epsilon_cal: float

    # ── Fit quality ──────────────────────────────────────────────────────
    chi2_red:    float   # reduced chi-squared
    converged:   bool    # True if LM converged
    n_bins_used: int     # number of profile bins used in fit

    # ── Provenance ───────────────────────────────────────────────────────
    source_file:       str   = ""    # path to the _profile_vs_r2.npy used
    t_tolansky_mm:     float = 0.0  # Tolansky gap seed [mm]
    eps_a_tolansky:    float = 0.0  # Tolansky ε_a seed
    alpha_tolansky:    float = 0.0  # Tolansky alpha seed [rad/px]


# ---------------------------------------------------------------------------
# §2.2 MasterCalibration — arithmetic mean of N CalibrationResult objects
# ---------------------------------------------------------------------------

@dataclass
class MasterCalibration:
    """
    Arithmetic mean of N CalibrationResult objects from one orbit.

    sigma_ fields are the standard error of the mean: sigma / sqrt(N).

    Exposes the same parameter interface as _CalResult (H06 internal) so
    it can be passed directly to run_airglow_inversion() and H06 helpers.
    """
    # Same fields as CalibrationResult (values are means, sigmas are SEMs)
    t_m:          float
    sigma_t_m:    float
    alpha:        float
    sigma_alpha:  float
    R_refl:       float   # = mean(R_refl) across N frames
    sigma_R_refl: float
    R2:           float
    sigma_R2:     float
    I0:           float
    sigma_I0:     float
    I1:           float
    sigma_I1:     float
    I2:           float
    sigma_I2:     float
    sigma0:       float
    sigma_sigma0: float
    sigma1:       float   # always 0.0 (fixed in H05)
    sigma2:       float   # always 0.0 (fixed in H05)
    B:            float
    sigma_B:      float
    ne_ratio:       float
    sigma_ne_ratio: float
    epsilon_cal:       float
    sigma_epsilon_cal: float

    # ── Averaging provenance ──────────────────────────────────────────────
    n_frames_averaged: int      # number of CalibrationResult objects averaged
    chi2_red_mean:     float    # mean chi2_red across frames
    n_converged:       int      # number of frames where converged=True
    orbit_number:      int = -1 # orbit this master cal belongs to (-1 = unknown)


# ---------------------------------------------------------------------------
# §3 FringeProfile adapter
# ---------------------------------------------------------------------------

def to_h05_fringe_profile(fp, r_max_px: float = 110.0):
    """
    Convert annular_reduction.FringeProfile to H05's internal _FringeProfile.

    Parameters
    ----------
    fp : annular_reduction.FringeProfile
        Output of annular_reduce().
    r_max_px : float
        Outer radius used in the inversion. Bins beyond this are excluded.

    Returns
    -------
    H05._FringeProfile-compatible object suitable for run_staged_inversion().
    """
    in_range = fp.r_grid <= r_max_px
    return _H05FringeProfile(
        profile       = fp.profile[in_range].copy(),
        r_grid        = fp.r_grid[in_range].copy(),
        sigma_profile = fp.sigma_profile[in_range].copy(),
        masked        = fp.masked[in_range].copy(),
        r_max_px      = float(r_max_px),
    )


# ---------------------------------------------------------------------------
# §4.1 average_calibrations
# ---------------------------------------------------------------------------

# Scalar parameter fields that have a corresponding sigma_ field.
_PARAM_FIELDS = (
    "t_m", "alpha", "R_refl", "R2",
    "I0", "I1", "I2",
    "sigma0", "B", "ne_ratio", "epsilon_cal",
)


def average_calibrations(
    results: list[CalibrationResult],
    orbit_number: int = -1,
) -> MasterCalibration:
    """
    Arithmetic mean of N CalibrationResult objects.

    Parameters
    ----------
    results : list[CalibrationResult]
        N results from independent inversion of N cal frames.
        Must have len >= 1. If len == 1, returns the single result
        wrapped as a MasterCalibration.
    orbit_number : int
        Orbit number for provenance. Default -1 (unknown).

    Returns
    -------
    MasterCalibration

    Notes
    -----
    Simple arithmetic mean is used (not inverse-variance weighting)
    because all N frames are acquired under identical conditions.
    The uncertainty on each mean parameter is the standard error
    of the mean: sigma_master = mean(sigma_i) / sqrt(N).

    Frames where converged=False are included in the average with a
    warning logged. Callers may wish to filter these before averaging.
    """
    if len(results) == 0:
        raise ValueError("results must contain at least one CalibrationResult")

    n = len(results)

    for i, r in enumerate(results):
        if not r.converged:
            log.warning(
                "CalibrationResult[%d] has converged=False; included in average", i
            )

    averaged: dict[str, float] = {}
    for f in _PARAM_FIELDS:
        sf = f"sigma_{f}"
        averaged[f]  = float(np.mean([getattr(r, f)  for r in results]))
        averaged[sf] = float(np.mean([getattr(r, sf) for r in results]) / np.sqrt(n))

    return MasterCalibration(
        t_m               = averaged["t_m"],
        sigma_t_m         = averaged["sigma_t_m"],
        alpha             = averaged["alpha"],
        sigma_alpha       = averaged["sigma_alpha"],
        R_refl            = averaged["R_refl"],
        sigma_R_refl      = averaged["sigma_R_refl"],
        R2                = averaged["R2"],
        sigma_R2          = averaged["sigma_R2"],
        I0                = averaged["I0"],
        sigma_I0          = averaged["sigma_I0"],
        I1                = averaged["I1"],
        sigma_I1          = averaged["sigma_I1"],
        I2                = averaged["I2"],
        sigma_I2          = averaged["sigma_I2"],
        sigma0            = averaged["sigma0"],
        sigma_sigma0      = averaged["sigma_sigma0"],
        sigma1            = 0.0,
        sigma2            = 0.0,
        B                 = averaged["B"],
        sigma_B           = averaged["sigma_B"],
        ne_ratio          = averaged["ne_ratio"],
        sigma_ne_ratio    = averaged["sigma_ne_ratio"],
        epsilon_cal       = averaged["epsilon_cal"],
        sigma_epsilon_cal = averaged["sigma_epsilon_cal"],
        n_frames_averaged = n,
        chi2_red_mean     = float(np.mean([r.chi2_red for r in results])),
        n_converged       = int(sum(r.converged for r in results)),
        orbit_number      = orbit_number,
    )


# ---------------------------------------------------------------------------
# §4.2 process_cal_frame
# ---------------------------------------------------------------------------

def process_cal_frame(
    pixels_ds: np.ndarray,
    r_max_px: float = 110.0,
    cx_seed: float | None = None,
    cy_seed: float | None = None,
    tolansky_n_pairs: int | None = None,
    h05_r_max_px: float | None = None,
) -> CalibrationResult:
    """
    Full calibration chain for one dark-subtracted cal frame.

    Runs in order:
      1. find_centre(pixels_ds)         → CentreResult
      2. annular_reduce(pixels_ds, ...) → FringeProfile
      3. run_tolansky_2line(peak_fits)  → TolanskyResult
      4. to_m05_priors(tol_result)      → dict
      5. run_staged_inversion(...)      → FitResult
      6. CalibrationResult from FitResult + Tolansky provenance

    Parameters
    ----------
    pixels_ds : np.ndarray, shape (H, W)
        Dark-subtracted calibration frame pixels. float32 or float64.
    r_max_px : float
        Outer radius for both annular reduction and H05 inversion.
        Default 110.0 (FlatSat/flight nominal).
    cx_seed, cy_seed : float or None
        Initial centre guess for find_centre(). Default: image centre.
    tolansky_n_pairs : int or None
        Number of ring pairs for Tolansky fit. None = use all.
    h05_r_max_px : float or None
        Override r_max for H05 inversion only. None = use r_max_px.

    Returns
    -------
    CalibrationResult

    Raises
    ------
    ValueError
        If run_staged_inversion fails to converge after 4 stages.
    InsufficientRingsError
        If Tolansky analysis finds fewer than 4 valid rings.
    """
    # 1. Centre finding
    centre = find_centre(pixels_ds, cx_seed=cx_seed, cy_seed=cy_seed)

    # 2. Annular reduction
    fp = annular_reduce(
        pixels_ds.astype(np.float32),
        centre.cx, centre.cy,
        centre.sigma_cx, centre.sigma_cy,
        r_max_px=r_max_px,
    )

    # 3. Tolansky analysis (uses peak_fits_r2 from FringeProfile)
    peak_array = _peak_fits_r2_to_array(fp.peak_fits_r2)
    tol = run_tolansky_2line(peak_array, n_pairs=tolansky_n_pairs)

    # 4. M05 priors
    priors = to_m05_priors(tol)
    t_eff  = phase_correct_gap(
        priors["t_init_mm"] * 1e-3,
        priors["epsilon_cal_a"],
        NE_WAVELENGTH_1_AIR_M,
    )

    # 5. H05 inversion
    h05_rmax = h05_r_max_px if h05_r_max_px is not None else r_max_px
    h05_fp   = to_h05_fringe_profile(fp, r_max_px=h05_rmax)
    fit      = run_staged_inversion(
        h05_fp, t_eff,
        alpha_init = priors["alpha_init"],
        eps_a      = priors["epsilon_cal_a"],
        R1_init    = 0.53,
        R2_init    = 0.53,
    )

    # 6. Wrap in CalibrationResult
    return CalibrationResult(
        t_m               = fit.t_m,
        sigma_t_m         = fit.sigma_t_m,
        alpha             = fit.alpha,
        sigma_alpha       = fit.sigma_alpha,
        R_refl            = fit.R1,          # R1 → H06 R_refl per H05 convention
        sigma_R_refl      = fit.sigma_R1,
        R2                = fit.R2,
        sigma_R2          = fit.sigma_R2,
        I0                = fit.I0,
        sigma_I0          = fit.sigma_I0,
        I1                = fit.I1,
        sigma_I1          = fit.sigma_I1,
        I2                = fit.I2,
        sigma_I2          = fit.sigma_I2,
        sigma0            = fit.sigma0,
        sigma_sigma0      = fit.sigma_sigma0,
        sigma1            = 0.0,
        sigma2            = 0.0,
        B                 = fit.B,
        sigma_B           = fit.sigma_B,
        ne_ratio          = fit.ne_ratio,
        sigma_ne_ratio    = fit.sigma_ne_ratio,
        epsilon_cal       = fit.epsilon_cal,
        sigma_epsilon_cal = fit.sigma_epsilon_cal,
        chi2_red          = fit.chi2_reduced,
        converged         = fit.converged,
        n_bins_used       = fit.n_bins_used,
        t_tolansky_mm     = priors["t_init_mm"],
        eps_a_tolansky    = priors["epsilon_cal_a"],
        alpha_tolansky    = priors["alpha_init"],
    )


# ---------------------------------------------------------------------------
# §4.3 master_cal_to_h06_cal
# ---------------------------------------------------------------------------

def master_cal_to_h06_cal(mc: MasterCalibration) -> _H06CalResult:
    """
    Convert MasterCalibration to H06's internal _CalResult shim.

    Allows MasterCalibration to be passed directly to
    run_airglow_inversion() and load_cal_result()-equivalent calls.
    """
    return _H06CalResult(
        t_m          = mc.t_m,
        alpha        = mc.alpha,
        R_refl       = mc.R_refl,
        I0           = mc.I0,
        I1           = mc.I1,
        I2           = mc.I2,
        sigma0       = mc.sigma0,
        sigma1       = mc.sigma1,
        sigma2       = mc.sigma2,
        B            = mc.B,
        epsilon_cal  = mc.epsilon_cal,
        quality_flags = 0,
    )


# ---------------------------------------------------------------------------
# §4.4 process_science_frame
# ---------------------------------------------------------------------------

def process_science_frame(
    pixels_ds: np.ndarray,
    master_cal: MasterCalibration,
    v_los_prior_ms: float,
    r_max_px: float = 110.0,
    cx_seed: float | None = None,
    cy_seed: float | None = None,
) -> AirglowResult:
    """
    Full science frame processing chain for one dark-subtracted science frame.

    Runs in order:
      1. find_centre(pixels_ds)
      2. annular_reduce(pixels_ds, ...)
      3. Apply sigma floor
      4. run_airglow_inversion(r_grid, profile, sigma, h06_cal,
                               r_max_px, v_los_prior_ms)

    Parameters
    ----------
    pixels_ds : np.ndarray, shape (H, W)
        Dark-subtracted science frame pixels. float32 or float64.
    master_cal : MasterCalibration
        Master calibration result for this orbit.
    v_los_prior_ms : float
        A-priori LOS velocity from H07 geometry engine [m/s].
        Must be computed BEFORE calling this function:
            v_los_prior_ms = geom.V_sc_LOS + geom.v_earth_LOS
    r_max_px : float
        Outer fringe radius [px]. Default 110.0.
    cx_seed, cy_seed : float or None
        Centre seed. If None, use image centre.

    Returns
    -------
    AirglowResult
    """
    # 1. Centre finding
    centre = find_centre(pixels_ds, cx_seed=cx_seed, cy_seed=cy_seed)

    # 2. Annular reduction
    fp = annular_reduce(
        pixels_ds.astype(np.float32),
        centre.cx, centre.cy,
        centre.sigma_cx, centre.sigma_cy,
        r_max_px=r_max_px,
    )

    # 3. Prepare profile arrays with sigma floor
    in_range    = fp.r_grid <= r_max_px
    r_grid      = fp.r_grid[in_range]
    profile_adu = fp.profile[in_range]
    sigma_adu   = fp.sigma_profile[in_range].copy()
    bad         = ~np.isfinite(sigma_adu)
    sigma_adu[bad] = (
        float(np.nanmedian(fp.sigma_profile))
        if np.any(np.isfinite(fp.sigma_profile)) else 1.0
    )
    s_floor   = max(1.0, float(np.median(profile_adu)) * 0.005)
    sigma_adu = np.maximum(sigma_adu, s_floor)

    # 4. H06 inversion
    h06_cal = master_cal_to_h06_cal(master_cal)
    return run_airglow_inversion(
        r_grid         = r_grid,
        profile_adu    = profile_adu,
        sigma_adu      = sigma_adu,
        cal            = h06_cal,
        r_max_px       = r_max_px,
        v_los_prior_ms = v_los_prior_ms,
    )
