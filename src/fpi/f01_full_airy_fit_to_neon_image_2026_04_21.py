"""
Module:      f01_full_airy_fit_to_neon_image_2026_04_21.py
Spec:        specs/F01_full_airy_fit_to_neon_image_2026-04-21.md
Author:      Claude Code / Scott Sewell
Generated:   2026-04-21
Project:     WindCube FPI Pipeline — NCAR/HAO
Repo:        soc_sewell

F01 implements step 4 of the calibration-to-wind pipeline: full modified-Airy
fit to the 1D neon calibration fringe profile.  The etalon gap d is fixed from
the Tolansky result (Z01a); the 9 remaining instrument parameters are fitted via
staged Levenberg-Marquardt (A→B→C→D) following Harding et al. 2014 §3.
"""

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from src.constants import NE_WAVELENGTH_1_M
from src.fpi.m01_airy_forward_model_2026_04_05 import airy_modified, airy_ideal
from src.fpi.m03_annular_reduction_2026_04_06 import FringeProfile, QualityFlags


# ---------------------------------------------------------------------------
# TolanskyResult — input from Z01a two-line Tolansky analysis
# ---------------------------------------------------------------------------

@dataclass
class TolanskyResult:
    """Outputs from the Z01a two-line Tolansky analysis."""
    t_m:         float   # Authoritative gap, metres (20.0006e-3)
    alpha_rpx:   float   # Plate scale from Tolansky WLS slope, rad/px
    epsilon_640: float   # Fractional order at lambda_640 nm
    epsilon_638: float   # Fractional order at lambda_638 nm
    epsilon_cal: float   # Rest-frame fractional order at lambda_OI (extrapolated)


# ---------------------------------------------------------------------------
# CalibrationFitFlags
# ---------------------------------------------------------------------------

class CalibrationFitFlags:
    """Bitmask quality flags for CalibrationResult.quality_flags."""
    GOOD           = 0x000
    FIT_FAILED     = 0x001  # LM did not converge in Stage D
    CHI2_HIGH      = 0x002  # chi2_red > 3.0
    CHI2_VERY_HIGH = 0x004  # chi2_red > 10.0
    CHI2_LOW       = 0x008  # chi2_red < 0.5
    STDERR_NONE    = 0x010  # any stderr is None / non-finite
    R_AT_BOUND     = 0x020  # R hit bounds [0.1, 0.95]
    ALPHA_AT_BOUND = 0x040  # alpha hit bounds [0.5×, 2×] init
    FEW_BINS       = 0x080  # n_good < 100


# ---------------------------------------------------------------------------
# CalibrationResult — output dataclass (§5)
# ---------------------------------------------------------------------------

@dataclass
class CalibrationResult:
    """
    Full instrument calibration from a single neon fringe profile fit.

    All 10 instrument parameters (9 fitted + gap fixed from Tolansky) with
    1σ and 2σ uncertainties.  Passed directly to F02 for airglow inversion.
    """
    # Gap — fixed from Tolansky, not fitted
    t_m: float

    # Fitted plate reflectivity
    R_refl:           float
    sigma_R_refl:     float
    two_sigma_R_refl: float

    # Fitted magnification (plate scale), rad/px
    alpha:           float
    sigma_alpha:     float
    two_sigma_alpha: float

    # Fitted intensity envelope
    I0:       float
    sigma_I0: float
    two_sigma_I0: float

    I1:       float
    sigma_I1: float
    two_sigma_I1: float

    I2:       float
    sigma_I2: float
    two_sigma_I2: float

    # Fitted PSF parameters, px
    sigma0:           float
    sigma_sigma0:     float
    two_sigma_sigma0: float

    sigma1:           float
    sigma_sigma1:     float
    two_sigma_sigma1: float

    sigma2:           float
    sigma_sigma2:     float
    two_sigma_sigma2: float

    # Fitted CCD bias
    B:       float
    sigma_B: float
    two_sigma_B: float

    # Provenance / quality
    epsilon_cal:   float
    chi2_reduced:  float
    n_bins_used:   int
    n_params_free: int
    converged:     bool
    quality_flags: int
    lambda_ne_m:   float
    timestamp:     float

    # Intermediate chi² values (one per stage A/B/C/D) — used by T04
    chi2_stages: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

_R_LO  = 0.10
_R_HI  = 0.95
_ALPHA_FACTOR_LO = 0.5
_ALPHA_FACTOR_HI = 2.0
_N_FREE = 9
_N_FINE = 500

# Stage definitions: list of free parameter names for each LM stage
_STAGES = [
    ["I0", "B"],
    ["I0", "I1", "I2", "B"],
    ["R", "alpha", "I0", "B"],
    ["R", "alpha", "I0", "I1", "I2", "sigma0", "sigma1", "sigma2", "B"],
]

# Column order for Stage D Jacobian → sigma extraction
_STAGE_D_ORDER = ["R", "alpha", "I0", "I1", "I2", "sigma0", "sigma1", "sigma2", "B"]

# Parameter bounds (spec §7).  alpha bounds are set dynamically from alpha_init.
# Use np.inf / -np.inf for one-sided bounds.
_STATIC_BOUNDS = {
    "R":      (_R_LO,   _R_HI),
    "I0":     (1.0,     np.inf),
    "I1":     (-0.5,    0.5),
    "I2":     (-0.5,    0.5),
    "sigma0": (0.01,    5.0),
    "sigma1": (-3.0,    3.0),
    "sigma2": (-3.0,    3.0),
    "B":      (0.0,     np.inf),
}


# ---------------------------------------------------------------------------
# Main fitting function
# ---------------------------------------------------------------------------

def fit_neon_fringe(
    profile: FringeProfile,
    tolansky: TolanskyResult,
    R_init: float = None,
) -> CalibrationResult:
    """
    Fit the modified Airy function to a neon 1D fringe profile.

    Implements spec F01 §4: validate → initial guesses → staged LM A/B/C/D →
    chi² check → covariance → flags → CalibrationResult.

    The etalon gap d is fixed from tolansky.t_m.  All 9 remaining instrument
    parameters (R, α, I₀, I₁, I₂, σ₀, σ₁, σ₂, B) are fitted.

    Parameters
    ----------
    profile  : FringeProfile from M03.annular_reduce() or reduce_calibration_frame()
    tolansky : TolanskyResult from Z01a Tolansky analysis
    R_init   : optional initial R_refl guess (default 0.53, spec §4.2)

    Returns
    -------
    CalibrationResult with all fitted parameters, uncertainties, and flags.
    """
    # ------------------------------------------------------------------
    # §4.1 Validate inputs
    # ------------------------------------------------------------------
    if profile.quality_flags & QualityFlags.CENTRE_FAILED:
        raise ValueError("FringeProfile.quality_flags has CENTRE_FAILED set")

    good_mask = (
        ~profile.masked
        & np.isfinite(profile.sigma_profile)
        & (profile.sigma_profile > 0.0)
        & np.isfinite(profile.profile)
    )
    n_good = int(np.sum(good_mask))
    if n_good < 50:
        raise ValueError(f"Too few good radial bins: {n_good} (minimum 50)")

    r_good  = profile.r_grid[good_mask]
    s_good  = profile.profile[good_mask]
    sig_raw = profile.sigma_profile[good_mask]

    sigma_floor = max(1.0, float(np.median(s_good)) * 0.005)
    sigma_good  = np.maximum(sig_raw, sigma_floor)

    # ------------------------------------------------------------------
    # §4.2 Initial parameter guesses
    # ------------------------------------------------------------------
    r_max      = float(profile.r_max_px)
    t_fixed    = float(tolansky.t_m)
    n_refr     = 1.0
    lam_ne     = NE_WAVELENGTH_1_M

    alpha_init  = float(tolansky.alpha_rpx)
    alpha_lo    = _ALPHA_FACTOR_LO * alpha_init
    alpha_hi    = _ALPHA_FACTOR_HI * alpha_init

    curr = {
        "R":      float(R_init) if R_init is not None else 0.53,
        "alpha":  alpha_init,
        "I0":     float(np.median(s_good)),
        "I1":     -0.1,
        "I2":      0.005,
        "sigma0":  0.5,
        "sigma1":  0.0,
        "sigma2":  0.0,
        "B":      float(np.percentile(s_good, 5)),
    }

    # Dynamic alpha bounds (depend on alpha_init)
    bounds_dict = dict(_STATIC_BOUNDS)
    bounds_dict["alpha"] = (alpha_lo, alpha_hi)

    def _stage_bounds(names):
        lo = [bounds_dict[k][0] for k in names]
        hi = [bounds_dict[k][1] for k in names]
        return (lo, hi)

    # Fine evaluation grid for the forward model (§4.3)
    r_fine = np.linspace(float(r_good[0]), float(r_good[-1]), _N_FINE)

    def _model(p: dict) -> np.ndarray:
        """Evaluate S(r) = Ã(r) + B on r_good via fine-grid interpolation."""
        A_fine = airy_modified(
            r_fine, lam_ne, t_fixed,
            p["R"], p["alpha"], n_refr, r_max,
            p["I0"], p["I1"], p["I2"],
            p["sigma0"], p["sigma1"], p["sigma2"],
        )
        return np.interp(r_good, r_fine, A_fine) + p["B"]

    # ------------------------------------------------------------------
    # §4.3 Staged Levenberg-Marquardt
    # ------------------------------------------------------------------
    dof         = max(n_good - _N_FREE, 1)
    chi2_stages = []
    final_lm    = None

    for stage_names in _STAGES:
        x0 = np.array([curr[k] for k in stage_names], dtype=float)
        bds = _stage_bounds(stage_names)

        def _residuals(x, names=stage_names):
            p = dict(curr)
            for k, v in zip(names, x):
                p[k] = v
            return (s_good - _model(p)) / sigma_good

        # method='trf' enforces spec §7 bounds (scipy's LM does not support bounds)
        lm_result = least_squares(_residuals, x0, bounds=bds, method="trf")

        # Commit best-fit values to curr
        for k, v in zip(stage_names, lm_result.x):
            curr[k] = float(v)

        chi2_stages.append(float(np.sum(lm_result.fun ** 2)) / dof)
        final_lm = lm_result

    converged = bool(final_lm.success)

    # ------------------------------------------------------------------
    # §4.4 Detect when params hit their bounds (trf enforces them)
    # ------------------------------------------------------------------
    quality_flags = CalibrationFitFlags.GOOD
    _atol = 1e-5   # proximity tolerance for "at bound"

    if curr["R"] <= _R_LO + _atol or curr["R"] >= _R_HI - _atol:
        quality_flags |= CalibrationFitFlags.R_AT_BOUND

    if curr["alpha"] <= alpha_lo + _atol or curr["alpha"] >= alpha_hi - _atol:
        quality_flags |= CalibrationFitFlags.ALPHA_AT_BOUND

    chi2_red = chi2_stages[-1]

    if not converged:
        quality_flags |= CalibrationFitFlags.FIT_FAILED
    if chi2_red > 10.0:
        quality_flags |= CalibrationFitFlags.CHI2_VERY_HIGH | CalibrationFitFlags.CHI2_HIGH
    elif chi2_red > 3.0:
        quality_flags |= CalibrationFitFlags.CHI2_HIGH
    if chi2_red < 0.5:
        quality_flags |= CalibrationFitFlags.CHI2_LOW
    if n_good < 100:
        quality_flags |= CalibrationFitFlags.FEW_BINS

    # ------------------------------------------------------------------
    # §4.5 Uncertainty estimation from Stage D Jacobian
    #
    # Column-scaled approach: normalize each Jacobian column by its L2 norm
    # before inverting JᵀJ.  This avoids 7-orders-of-magnitude conditioning
    # from alpha (norm ≈ 7e7) vs B (norm ≈ 1).  Back-scale after inversion.
    # Params with zero-norm columns (sigma2 is analytically unidentifiable via
    # mean-sigma approximation) are assigned sigma=10 (one FSR equivalent).
    # ------------------------------------------------------------------
    J = final_lm.jac          # shape (n_good, 9); Stage D order
    col_norms = np.array([np.linalg.norm(J[:, j]) for j in range(_N_FREE)])
    col_norms_safe = np.where(col_norms > 1e-8 * col_norms.max(), col_norms, 1.0)
    J_sc   = J / col_norms_safe[np.newaxis, :]
    JtJ_sc = J_sc.T @ J_sc

    cond = float(np.linalg.cond(JtJ_sc))
    if cond > 1e14:
        JtJ_sc_inv = np.linalg.pinv(JtJ_sc, rcond=1e-10)
    else:
        try:
            JtJ_sc_inv = np.linalg.inv(JtJ_sc)
        except np.linalg.LinAlgError:
            JtJ_sc_inv = np.linalg.pinv(JtJ_sc, rcond=1e-10)

    sigmas_raw = np.sqrt(np.abs(chi2_red * np.diag(JtJ_sc_inv))) / col_norms_safe
    # Unidentifiable params (zero Jacobian column) get a large sentinel sigma
    zero_mask  = col_norms < 1e-8 * col_norms.max()
    sigmas_raw = np.where(zero_mask, 10.0, sigmas_raw)

    if not np.all(np.isfinite(sigmas_raw)):
        quality_flags |= CalibrationFitFlags.STDERR_NONE

    # Replace non-finite entries with 0 for output (flag already set)
    sigmas_safe = np.where(np.isfinite(sigmas_raw), sigmas_raw, 0.0)
    sig = {k: float(sigmas_safe[i]) for i, k in enumerate(_STAGE_D_ORDER)}

    def _2s(k: str) -> float:
        return 2.0 * sig[k]

    # ------------------------------------------------------------------
    # Build and return CalibrationResult
    # ------------------------------------------------------------------
    return CalibrationResult(
        t_m              = t_fixed,
        R_refl           = curr["R"],
        sigma_R_refl     = sig["R"],
        two_sigma_R_refl = _2s("R"),
        alpha            = curr["alpha"],
        sigma_alpha      = sig["alpha"],
        two_sigma_alpha  = _2s("alpha"),
        I0               = curr["I0"],
        sigma_I0         = sig["I0"],
        two_sigma_I0     = _2s("I0"),
        I1               = curr["I1"],
        sigma_I1         = sig["I1"],
        two_sigma_I1     = _2s("I1"),
        I2               = curr["I2"],
        sigma_I2         = sig["I2"],
        two_sigma_I2     = _2s("I2"),
        sigma0           = curr["sigma0"],
        sigma_sigma0     = sig["sigma0"],
        two_sigma_sigma0 = _2s("sigma0"),
        sigma1           = curr["sigma1"],
        sigma_sigma1     = sig["sigma1"],
        two_sigma_sigma1 = _2s("sigma1"),
        sigma2           = curr["sigma2"],
        sigma_sigma2     = sig["sigma2"],
        two_sigma_sigma2 = _2s("sigma2"),
        B                = curr["B"],
        sigma_B          = sig["B"],
        two_sigma_B      = _2s("B"),
        epsilon_cal      = float(tolansky.epsilon_cal),
        chi2_reduced     = chi2_red,
        n_bins_used      = n_good,
        n_params_free    = _N_FREE,
        converged        = converged,
        quality_flags    = quality_flags,
        lambda_ne_m      = lam_ne,
        timestamp        = time.time(),
        chi2_stages      = chi2_stages,
    )
