"""
Module:      f01_full_airy_fit_to_neon_image_2026_04_22.py
Spec:        specs/F01_full_airy_fit_to_neon_image_2026-04-22.md
Author:      Claude Code / Scott Sewell
Generated:   2026-04-22
Project:     WindCube FPI Pipeline — NCAR/HAO
Repo:        soc_sewell

F01 v3 implements step 4b of the calibration-to-wind pipeline: two-line
modified-Airy fit to the 1D neon calibration fringe profile.  The etalon
gap d is fixed from the Tolansky result (Z01a); 11 free parameters
(R, α, I₀, I₁, I₂, σ₀, σ₁, σ₂, Y_A, Y_B, B) are fitted via staged
Levenberg-Marquardt (A→B→C→D→E) per spec §5.3.
"""

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks

from src.constants import NE_WAVELENGTH_1_M, NE_WAVELENGTH_2_M
from src.fpi.m01_airy_forward_model_2026_04_05 import airy_modified, airy_ideal  # noqa: F401
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
    FIT_FAILED     = 0x001  # LM did not converge in Stage E
    CHI2_HIGH      = 0x002  # chi2_red > 3.0
    CHI2_VERY_HIGH = 0x004  # chi2_red > 10.0
    CHI2_LOW       = 0x008  # chi2_red < 0.5
    STDERR_NONE    = 0x010  # any stderr is None / non-finite
    R_AT_BOUND     = 0x020  # R hit bounds [0.1, 0.95]
    ALPHA_AT_BOUND = 0x040  # alpha hit bounds [0.5×, 2×] init
    FEW_BINS       = 0x080  # n_good < 100
    YB_RATIO_LOW   = 0x100  # Y_B/Y_A < 0.3 (possible family misidentification)
    YB_RATIO_HIGH  = 0x200  # Y_B/Y_A > 1.0 (families may be swapped)


# ---------------------------------------------------------------------------
# CalibrationResult — output dataclass
# ---------------------------------------------------------------------------

@dataclass
class CalibrationResult:
    """
    Full instrument calibration from a two-line neon fringe profile fit.

    All 11 free parameters with 1σ and 2σ uncertainties.
    Passed directly to F02 for airglow Doppler inversion.
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
    I0:           float
    sigma_I0:     float
    two_sigma_I0: float

    I1:           float
    sigma_I1:     float
    two_sigma_I1: float

    I2:           float
    sigma_I2:     float
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

    # Primary line (640 nm) intensity scale
    Y_A:           float
    sigma_Y_A:     float
    two_sigma_Y_A: float

    # Secondary line (638 nm) intensity
    Y_B:           float
    sigma_Y_B:     float
    two_sigma_Y_B: float

    # Intensity ratio Y_B/Y_A (diagnostic; expected 0.55–0.65)
    intensity_ratio: float

    # Fitted CCD bias
    B:           float
    sigma_B:     float
    two_sigma_B: float

    # Provenance / quality
    epsilon_cal:   float
    chi2_reduced:  float
    n_bins_used:   int
    n_params_free: int
    converged:     bool
    quality_flags: int
    lambda_A_m:    float
    lambda_B_m:    float
    timestamp:     float

    # Intermediate chi² values — one per stage A–E
    chi2_stages: list = field(default_factory=list)

    def __post_init__(self):
        """Enforce two_sigma_ == 2 * sigma_ for every parameter."""
        pairs = [
            ("R_refl",   self.two_sigma_R_refl,   self.sigma_R_refl),
            ("alpha",    self.two_sigma_alpha,    self.sigma_alpha),
            ("I0",       self.two_sigma_I0,       self.sigma_I0),
            ("I1",       self.two_sigma_I1,       self.sigma_I1),
            ("I2",       self.two_sigma_I2,       self.sigma_I2),
            ("sigma0",   self.two_sigma_sigma0,   self.sigma_sigma0),
            ("sigma1",   self.two_sigma_sigma1,   self.sigma_sigma1),
            ("sigma2",   self.two_sigma_sigma2,   self.sigma_sigma2),
            ("Y_A",      self.two_sigma_Y_A,      self.sigma_Y_A),
            ("Y_B",      self.two_sigma_Y_B,      self.sigma_Y_B),
            ("B",        self.two_sigma_B,        self.sigma_B),
        ]
        for name, two_s, s in pairs:
            assert two_s == 2.0 * s, (
                f"two_sigma_{name} ({two_s}) != 2 * sigma_{name} ({s})"
            )


# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

_R_LO            = 0.10
_R_HI            = 0.95
_ALPHA_FACTOR_LO = 0.5
_ALPHA_FACTOR_HI = 2.0
_N_FREE          = 11   # reported in CalibrationResult (includes Y_A fixed at 1.0)
_N_FITTED        = 10   # actual free params in the optimisation (Y_A excluded)
_N_FINE          = 500

# Y_A is fixed at 1.0 by convention (spec §4.2); I₀ absorbs the absolute scale.
# This breaks the (Y_A, Y_B, I₀) → (k·Y_A, k·Y_B, I₀/k) degeneracy.
_Y_A_FIXED = 1.0

# Stage definitions: free parameter names for each LM stage A→E
# Y_A is NOT included — it is held at _Y_A_FIXED throughout.
_STAGES = [
    ["Y_B", "B"],                                                          # Stage A
    ["Y_B", "I0", "B"],                                                    # Stage B
    ["Y_B", "I0", "I1", "I2", "B"],                                       # Stage C
    ["R", "alpha", "Y_B", "I0", "B"],                                     # Stage D
    ["R", "alpha", "I0", "I1", "I2", "sigma0", "sigma1", "sigma2",
     "Y_B", "B"],                                                          # Stage E
]

# Column order for Stage E Jacobian → covariance extraction (must match Stage E)
_STAGE_E_ORDER = [
    "R", "alpha", "I0", "I1", "I2", "sigma0", "sigma1", "sigma2",
    "Y_B", "B",
]

# Static parameter bounds — alpha bounds are set dynamically from alpha_init
_STATIC_BOUNDS = {
    "R":      (_R_LO,   _R_HI),
    "I0":     (1.0,     np.inf),
    "I1":     (-0.5,    0.5),
    "I2":     (-0.5,    0.5),
    "sigma0": (0.01,    5.0),
    "sigma1": (-3.0,    3.0),
    "sigma2": (-3.0,    3.0),
    "Y_B":    (0.01,    np.inf),
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
    Fit the two-line modified Airy function to a neon 1D fringe profile.

    Implements F01 v3 §5: validate → initial guesses → staged LM A–E →
    chi² check → covariance → flags → CalibrationResult.

    The etalon gap d is fixed from tolansky.t_m.  All 11 free parameters
    (R, α, I₀, I₁, I₂, σ₀, σ₁, σ₂, Y_A, Y_B, B) are fitted.

    Parameters
    ----------
    profile  : FringeProfile from M03.annular_reduce()
    tolansky : TolanskyResult from Z01a Tolansky analysis
    R_init   : optional R_refl initial guess; if None, estimated from contrast

    Returns
    -------
    CalibrationResult with all fitted parameters, uncertainties, and flags.
    """
    # ------------------------------------------------------------------
    # §5.1 Validate inputs
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

    quality_flags = CalibrationFitFlags.GOOD
    if n_good < 100:
        quality_flags |= CalibrationFitFlags.FEW_BINS

    # ------------------------------------------------------------------
    # §5.2 Data-driven initial parameter estimates
    # ------------------------------------------------------------------
    r_max      = float(profile.r_max_px)
    t_fixed    = float(tolansky.t_m)
    n_refr     = 1.0

    alpha_init = float(tolansky.alpha_rpx)
    alpha_lo   = _ALPHA_FACTOR_LO * alpha_init
    alpha_hi   = _ALPHA_FACTOR_HI * alpha_init

    # B: 2nd percentile of profile
    B_init = float(np.percentile(s_good, 2))

    # Amplitude-based family separation (spec §5.2)
    peak_distance = max(5, int(len(s_good) / 25))
    peak_height   = float(np.percentile(s_good, 70))
    peak_idx, _   = find_peaks(s_good, distance=peak_distance, height=peak_height)

    if len(peak_idx) < 4:
        print(f"[F01 WARNING] Only {len(peak_idx)} peaks found; using fallback init.")
        I0_init      = max(float(np.percentile(s_good, 95)) - B_init, 10.0)
        Y_A_init     = 1.0
        Y_B_init     = 0.6
        R_from_data  = 0.53
    else:
        peak_amps  = s_good[peak_idx]
        sorted_idx = np.argsort(peak_amps)
        n_half     = len(sorted_idx) // 2

        family_B_idx = peak_idx[sorted_idx[:n_half]]   # dim  (638 nm)
        family_A_idx = peak_idx[sorted_idx[n_half:]]   # bright (640 nm)

        bright_amps = np.maximum(s_good[family_A_idx] - B_init, 1.0)
        dim_amps    = np.maximum(s_good[family_B_idx]  - B_init, 1.0)

        I0_init = max(float(np.percentile(bright_amps, 80)), 10.0)
        Y_A_init = 1.0

        if len(family_A_idx) >= 2 and len(family_B_idx) >= 2:
            Y_B_init = float(np.median(dim_amps) / np.median(bright_amps))
        else:
            Y_B_init = 0.6
            print("[F01 WARNING] Fewer than 2 peaks in one family; Y_B_init = 0.6")
        Y_B_init = float(np.clip(Y_B_init, 0.1, 1.5))

        # R from peak-to-trough contrast between consecutive A-family peaks.
        # When B >> fringe amplitude the denominator is dominated by background
        # and the contrast formula is unreliable; use a prior in that case.
        i_A_sorted = np.sort(family_A_idx)
        C_vals = []
        for k in range(len(i_A_sorted) - 1):
            lo, hi = i_A_sorted[k], i_A_sorted[k + 1]
            if hi > lo + 2:
                I_max = float(s_good[lo]) - B_init
                I_min = float(np.min(s_good[lo + 1:hi])) - B_init
                denom = I_max + I_min
                if denom > 1.0 and I_max > 0.0 and I_min >= 0.0:
                    C_vals.append(float(np.clip((I_max - I_min) / denom, 0.0, 1.0)))

        if len(C_vals) >= 2 and B_init <= float(np.median(bright_amps)):
            C = float(np.clip(np.median(C_vals), 0.05, 0.999))
            F_est = 2.0 * C / (1.0 - C)
            x = 2.0 * (np.sqrt(1.0 + F_est) - 1.0) / F_est
            R_from_data = float(np.clip(1.0 - x, _R_LO, _R_HI))
        else:
            R_from_data = 0.53  # background pedestal too large for contrast estimate

    print(
        f"[F01 init] B={B_init:.1f}  I0={I0_init:.1f}  "
        f"Y_A={Y_A_init:.3f}  Y_B={Y_B_init:.3f}  "
        f"R(data)={R_from_data:.3f}  alpha={alpha_init:.6e}"
    )

    curr = {
        "R":      float(R_init) if R_init is not None else R_from_data,
        "alpha":  alpha_init,
        "I0":     I0_init,
        "I1":     -0.1,
        "I2":      0.005,
        "sigma0":  0.5,
        "sigma1":  0.0,
        "sigma2":  0.0,
        "Y_A":    _Y_A_FIXED,   # held constant; not in stage lists
        "Y_B":    Y_B_init,
        "B":      B_init,
    }

    # Dynamic alpha bounds
    bounds_dict = dict(_STATIC_BOUNDS)
    bounds_dict["alpha"] = (alpha_lo, alpha_hi)

    def _stage_bounds(names):
        lo = [bounds_dict[k][0] for k in names]
        hi = [bounds_dict[k][1] for k in names]
        return (lo, hi)

    # Fine evaluation grid for forward model (spec §5.3)
    r_fine = np.linspace(float(r_good[0]), float(r_good[-1]), _N_FINE)

    def _model(p: dict) -> np.ndarray:
        """Two-line Airy: Ã(λ_A) + Y_B·Ã(λ_B) + B.  Y_A = 1.0 fixed."""
        A_fine_A = airy_modified(
            r_fine, NE_WAVELENGTH_1_M, t_fixed,
            p["R"], p["alpha"], n_refr, r_max,
            p["I0"], p["I1"], p["I2"],
            p["sigma0"], p["sigma1"], p["sigma2"],
        )
        A_fine_B = airy_modified(
            r_fine, NE_WAVELENGTH_2_M, t_fixed,
            p["R"], p["alpha"], n_refr, r_max,
            p["I0"], p["I1"], p["I2"],
            p["sigma0"], p["sigma1"], p["sigma2"],
        )
        combined = A_fine_A + p["Y_B"] * A_fine_B + p["B"]
        return np.interp(r_good, r_fine, combined)

    # ------------------------------------------------------------------
    # §5.3 Staged Levenberg-Marquardt (A→B→C→D→E)
    # ------------------------------------------------------------------
    dof         = max(n_good - _N_FITTED, 1)
    chi2_stages = []
    final_lm    = None

    for stage_names in _STAGES:
        x0  = np.array([curr[k] for k in stage_names], dtype=float)
        bds = _stage_bounds(stage_names)

        def _residuals(x, names=stage_names):
            p = dict(curr)
            for k, v in zip(names, x):
                p[k] = v
            return (s_good - _model(p)) / sigma_good

        lm_result = least_squares(_residuals, x0, bounds=bds, method="trf")

        # Clip to bounds before passing to the next stage
        for k, v in zip(stage_names, lm_result.x):
            lo, hi = bounds_dict[k]
            curr[k] = float(np.clip(v, lo, hi if np.isfinite(hi) else v))

        chi2_stages.append(float(np.sum(lm_result.fun ** 2)) / dof)
        final_lm = lm_result

    converged = bool(final_lm.success)

    # ------------------------------------------------------------------
    # §5.4 Bound-hit flags
    # ------------------------------------------------------------------
    _atol = 1e-5

    if curr["R"] <= _R_LO + _atol or curr["R"] >= _R_HI - _atol:
        quality_flags |= CalibrationFitFlags.R_AT_BOUND

    if curr["alpha"] <= alpha_lo + _atol or curr["alpha"] >= alpha_hi - _atol:
        quality_flags |= CalibrationFitFlags.ALPHA_AT_BOUND

    # ------------------------------------------------------------------
    # §5.5 Chi² quality flags
    # ------------------------------------------------------------------
    chi2_red = chi2_stages[-1]

    if not converged:
        quality_flags |= CalibrationFitFlags.FIT_FAILED
    if chi2_red > 10.0:
        quality_flags |= CalibrationFitFlags.CHI2_VERY_HIGH | CalibrationFitFlags.CHI2_HIGH
    elif chi2_red > 3.0:
        quality_flags |= CalibrationFitFlags.CHI2_HIGH
    if chi2_red < 0.5:
        quality_flags |= CalibrationFitFlags.CHI2_LOW

    # ------------------------------------------------------------------
    # §5.6 Uncertainty estimation — column-scaled Jacobian from Stage E
    # ------------------------------------------------------------------
    J = final_lm.jac   # shape (n_good, 10) in _STAGE_E_ORDER (Y_A excluded)
    col_norms      = np.array([np.linalg.norm(J[:, j]) for j in range(_N_FITTED)])
    # Only treat a column as zero if it is genuinely numerically zero; using a
    # relative threshold against the max norm incorrectly flags small-but-real
    # columns (e.g. I0 when alpha has an enormous column norm due to unit scale).
    col_norms_safe = np.where(col_norms > 0.0, col_norms, 1.0)
    J_sc   = J / col_norms_safe[np.newaxis, :]
    JtJ_sc = J_sc.T @ J_sc

    cond = float(np.linalg.cond(JtJ_sc))
    if cond > 1e14:
        JtJ_sc_inv = np.linalg.pinv(JtJ_sc, rcond=1e-10)
        quality_flags |= CalibrationFitFlags.STDERR_NONE
    else:
        try:
            JtJ_sc_inv = np.linalg.inv(JtJ_sc)
        except np.linalg.LinAlgError:
            JtJ_sc_inv = np.linalg.pinv(JtJ_sc, rcond=1e-10)
            quality_flags |= CalibrationFitFlags.STDERR_NONE

    sigmas_raw = np.sqrt(np.maximum(chi2_red * np.diag(JtJ_sc_inv), 0.0)) / col_norms_safe

    if not np.all(np.isfinite(sigmas_raw)):
        quality_flags |= CalibrationFitFlags.STDERR_NONE

    sigmas_safe = np.where(np.isfinite(sigmas_raw), sigmas_raw, 0.0)
    sig = {k: float(sigmas_safe[i]) for i, k in enumerate(_STAGE_E_ORDER)}
    sig["Y_A"] = 0.0   # Y_A is fixed at 1.0; no fitted uncertainty

    # YB ratio flags (after covariance)
    ratio = curr["Y_B"] / curr["Y_A"]
    if ratio < 0.3:
        quality_flags |= CalibrationFitFlags.YB_RATIO_LOW
    if ratio > 1.0:
        quality_flags |= CalibrationFitFlags.YB_RATIO_HIGH

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
        Y_A              = curr["Y_A"],
        sigma_Y_A        = sig["Y_A"],
        two_sigma_Y_A    = _2s("Y_A"),
        Y_B              = curr["Y_B"],
        sigma_Y_B        = sig["Y_B"],
        two_sigma_Y_B    = _2s("Y_B"),
        intensity_ratio  = float(ratio),
        B                = curr["B"],
        sigma_B          = sig["B"],
        two_sigma_B      = _2s("B"),
        epsilon_cal      = float(tolansky.epsilon_cal),
        chi2_reduced     = chi2_red,
        n_bins_used      = n_good,
        n_params_free    = _N_FREE,
        converged        = converged,
        quality_flags    = quality_flags,
        lambda_A_m       = NE_WAVELENGTH_1_M,
        lambda_B_m       = NE_WAVELENGTH_2_M,
        timestamp        = time.time(),
        chi2_stages      = chi2_stages,
    )
