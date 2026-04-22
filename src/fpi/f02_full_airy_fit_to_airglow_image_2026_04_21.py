"""
Module:      f02_full_airy_fit_to_airglow_image_2026_04_21.py
Spec:        specs/F02_full_airy_fit_to_airglow_image_2026-04-21.md
Author:      Claude Code / Scott Sewell
Generated:   2026-04-21
Project:     WindCube FPI Pipeline — NCAR/HAO
Repo:        soc_sewell

F02 implements steps 5–8 of the calibration-to-wind pipeline: 3-parameter
Levenberg-Marquardt inversion of an OI 630 nm airglow FringeProfile to recover
the LOS Doppler wind v_rel and its 2σ uncertainty.  All 10 instrument parameters
are fixed from the F01 CalibrationResult.

Scan and Doppler conversion use OI_WAVELENGTH_VACUUM_M = 630.0304 nm.
epsilon_sci diagnostic uses OI_WAVELENGTH_M = 629.95 nm (avoids N_int boundary).
n_scan = 211 (LD-1 locked decision from F01 spec §11).
"""

import logging
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from src.fpi.f01_full_airy_fit_to_neon_image_2026_04_21 import CalibrationResult
from src.fpi.m01_airy_forward_model_2026_04_05 import airy_modified
from src.fpi.m03_annular_reduction_2026_04_06 import FringeProfile, QualityFlags
from src.constants import (
    OI_WAVELENGTH_VACUUM_M,   # 630.0304e-9 m — scan centre and Doppler reference
    OI_WAVELENGTH_M,          # 629.95e-9 m  — epsilon_sci diagnostic only
    ETALON_FSR_OI_M,
    SPEED_OF_LIGHT_MS,
)

log = logging.getLogger(__name__)

# LD-1 locked: n_scan must be odd so v=0 falls exactly on a grid point.
# 211 is the smallest odd n ≥ 200 where v=0, ±200, ±150, ±300 m/s all avoid
# scan-step midpoints (see F01 spec §11 for derivation).
_N_SCAN = 211
_N_FINE = 500
_N_FREE = 3


# ---------------------------------------------------------------------------
# Quality-flag bitmask (spec §6)
# ---------------------------------------------------------------------------

class AirglowFitFlags:
    """Bitmask quality flags for AirglowFitResult (spec §6)."""
    GOOD                 = 0x000
    FIT_FAILED           = 0x001   # LM did not converge
    CHI2_HIGH            = 0x002   # chi2_red > 3.0
    CHI2_VERY_HIGH       = 0x004   # chi2_red > 10.0
    CHI2_LOW             = 0x008   # chi2_red < 0.5
    SCAN_AMBIGUOUS       = 0x010   # second-best scan chi2 within 10% of minimum
    LAMBDA_C_AT_BOUND    = 0x020   # lambda_c hit its bound
    STDERR_NONE          = 0x040   # any stderr non-finite
    LOW_SNR              = 0x080   # (max−min profile) / B_sci < 1.0
    CAL_QUALITY_DEGRADED = 0x100   # CalibrationResult had non-GOOD flags


# ---------------------------------------------------------------------------
# Output dataclass (spec §5)
# ---------------------------------------------------------------------------

@dataclass
class AirglowFitResult:
    """
    Output of F02 airglow inversion.

    All two_sigma_ fields are enforced to equal exactly 2 × sigma_ (S04).
    Passed to M07 for LOS-to-vector wind decomposition.
    """
    # Primary output — line centre and wind
    lambda_c_m:           float
    sigma_lambda_c_m:     float
    two_sigma_lambda_c_m: float

    v_rel_ms:           float
    sigma_v_rel_ms:     float
    two_sigma_v_rel_ms: float

    # Fitted parameters (diagnostics)
    Y_line:           float
    sigma_Y_line:     float
    two_sigma_Y_line: float

    B_sci:            float
    sigma_B_sci:      float
    two_sigma_B_sci:  float

    # Fit quality
    chi2_reduced:  float
    n_bins_used:   int
    n_params_free: int      # always 3 for F02
    converged:     bool
    quality_flags: int

    # Phase relationship to calibration (diagnostic; uses OI_WAVELENGTH_M 629.95 nm)
    epsilon_sci:   float    # (2 × lambda_c_m / OI_WAVELENGTH_M) mod 1
    delta_epsilon: float    # epsilon_sci − epsilon_cal

    # Input traceability
    calibration_t_m:         float
    calibration_epsilon_cal: float

    # Scan diagnostics
    lambda_c_scan_init_m: float   # scan centre (= OI_WAVELENGTH_VACUUM_M)
    lambda_c_lm_init_m:   float   # lambda_c passed to LM after scan

    def __post_init__(self) -> None:
        assert self.two_sigma_lambda_c_m == 2.0 * self.sigma_lambda_c_m, (
            f"two_sigma_lambda_c_m ({self.two_sigma_lambda_c_m}) ≠ "
            f"2 × sigma_lambda_c_m ({self.sigma_lambda_c_m})"
        )
        assert self.two_sigma_v_rel_ms == 2.0 * self.sigma_v_rel_ms, (
            f"two_sigma_v_rel_ms ({self.two_sigma_v_rel_ms}) ≠ "
            f"2 × sigma_v_rel_ms ({self.sigma_v_rel_ms})"
        )
        assert self.two_sigma_Y_line == 2.0 * self.sigma_Y_line, (
            f"two_sigma_Y_line ({self.two_sigma_Y_line}) ≠ "
            f"2 × sigma_Y_line ({self.sigma_Y_line})"
        )
        assert self.two_sigma_B_sci == 2.0 * self.sigma_B_sci, (
            f"two_sigma_B_sci ({self.two_sigma_B_sci}) ≠ "
            f"2 × sigma_B_sci ({self.sigma_B_sci})"
        )


# ---------------------------------------------------------------------------
# Forward model helpers
# ---------------------------------------------------------------------------

def _build_fine_grid(r_max: float, n_fine: int = _N_FINE) -> np.ndarray:
    return np.linspace(0.0, r_max, n_fine)


def _airglow_model_fine(
    r_fine: np.ndarray,
    lambda_c_m: float,
    Y_line: float,
    B_sci: float,
    cal: CalibrationResult,
) -> np.ndarray:
    r_max = float(r_fine[-1])
    airy = airy_modified(
        r_fine, lambda_c_m,
        t=cal.t_m, R_refl=cal.R_refl, alpha=cal.alpha, n=1.0,
        r_max=r_max,
        I0=cal.I0, I1=cal.I1, I2=cal.I2,
        sigma0=cal.sigma0, sigma1=cal.sigma1, sigma2=cal.sigma2,
    )
    return Y_line * airy + B_sci


# ---------------------------------------------------------------------------
# Brute-force scan for λ_c initialisation (spec §4.2)
# ---------------------------------------------------------------------------

def _lambda_c_scan(
    r_good: np.ndarray,
    profile_good: np.ndarray,
    sigma_good: np.ndarray,
    r_max: float,
    cal: CalibrationResult,
    n_scan: int = _N_SCAN,
    n_fine: int = _N_FINE,
) -> tuple:
    """
    Scan lambda_c over one FSR centred at OI_WAVELENGTH_VACUUM_M.

    At each of n_scan evenly-spaced lambda_c values, analytically solve for
    Y_line and B_sci (2×2 weighted linear least squares) and record chi2_red.
    Select lambda_c_best at minimum chi2.

    Returns (lambda_c_best, chi2_min, scan_flags).
    """
    lc_lo = OI_WAVELENGTH_VACUUM_M - 0.5 * ETALON_FSR_OI_M
    lc_hi = OI_WAVELENGTH_VACUUM_M + 0.5 * ETALON_FSR_OI_M
    scan_grid = np.linspace(lc_lo, lc_hi, n_scan)

    r_fine = _build_fine_grid(r_max, n_fine)
    n_good = len(r_good)
    chi2_arr = np.full(n_scan, np.inf)

    for k, lc in enumerate(scan_grid):
        airy_fine = airy_modified(
            r_fine, lc,
            t=cal.t_m, R_refl=cal.R_refl, alpha=cal.alpha, n=1.0,
            r_max=r_max,
            I0=cal.I0, I1=cal.I1, I2=cal.I2,
            sigma0=cal.sigma0, sigma1=cal.sigma1, sigma2=cal.sigma2,
        )
        airy_bins = np.interp(r_good, r_fine, airy_fine)

        w = 1.0 / sigma_good
        A = np.column_stack([airy_bins * w, w])
        b = profile_good * w
        AtA = A.T @ A
        Atb = A.T @ b
        try:
            p_lsq = np.linalg.solve(AtA, Atb)
        except np.linalg.LinAlgError:
            continue

        Y_line_k, B_sci_k = p_lsq
        model_k = Y_line_k * airy_bins + B_sci_k
        resid_k = (profile_good - model_k) / sigma_good
        chi2_arr[k] = float(np.sum(resid_k ** 2)) / max(n_good - 2, 1)

    best_idx = int(np.argmin(chi2_arr))
    chi2_min = float(chi2_arr[best_idx])
    lambda_c_best = float(scan_grid[best_idx])

    scan_flags = AirglowFitFlags.GOOD
    alt_mask = np.ones(n_scan, dtype=bool)
    alt_mask[best_idx] = False
    chi2_alt = chi2_arr[alt_mask]
    if len(chi2_alt) > 0 and np.nanmin(chi2_alt) < chi2_min * 1.10:
        scan_flags |= AirglowFitFlags.SCAN_AMBIGUOUS
        log.warning(
            f"SCAN_AMBIGUOUS: second-best chi2 {np.nanmin(chi2_alt):.4f} "
            f"within 10% of minimum {chi2_min:.4f}"
        )

    # Recover analytic Y_line, B_sci at the best scan point for LM warm-start
    airy_best = airy_modified(
        r_fine, lambda_c_best,
        t=cal.t_m, R_refl=cal.R_refl, alpha=cal.alpha, n=1.0,
        r_max=r_max,
        I0=cal.I0, I1=cal.I1, I2=cal.I2,
        sigma0=cal.sigma0, sigma1=cal.sigma1, sigma2=cal.sigma2,
    )
    airy_b = np.interp(r_good, r_fine, airy_best)
    w = 1.0 / sigma_good
    A = np.column_stack([airy_b * w, w])
    b_vec = profile_good * w
    try:
        p_best = np.linalg.solve(A.T @ A, A.T @ b_vec)
        Y_line_best = float(p_best[0])
        B_sci_best = float(p_best[1])
    except np.linalg.LinAlgError:
        Y_line_best = float(np.max(profile_good)) / float(np.max(airy_b)) if np.max(airy_b) > 0 else 1.0
        B_sci_best = float(np.min(profile_good)) * 0.8

    return lambda_c_best, chi2_min, Y_line_best, B_sci_best, scan_flags


# ---------------------------------------------------------------------------
# LM fit (spec §4.3)
# ---------------------------------------------------------------------------

def _run_airglow_lm(
    r_good: np.ndarray,
    profile_good: np.ndarray,
    sigma_good: np.ndarray,
    r_max: float,
    cal: CalibrationResult,
    lambda_c_init_m: float,
    Y_line_init: float,
    B_sci_init: float,
    n_fine: int = _N_FINE,
):
    """
    3-parameter LM fit: free [lambda_c, Y_line, B_sci]; all instrument params fixed.

    Uses soft-bound penalty residuals since scipy LM does not support bounds.
    """
    lambda_lo = OI_WAVELENGTH_VACUUM_M - 1.5 * ETALON_FSR_OI_M
    lambda_hi = OI_WAVELENGTH_VACUUM_M + 1.5 * ETALON_FSR_OI_M
    B_lo = 0.0
    B_hi = float(np.min(profile_good)) * 1.5 if np.min(profile_good) > 0 else 1e6
    if not np.isfinite(B_hi) or B_hi <= 0:
        B_hi = float(np.max(profile_good))

    lo_arr = np.array([lambda_lo, 0.0,  B_lo])
    hi_arr = np.array([lambda_hi, np.inf, B_hi])
    hi_finite = np.where(np.isfinite(hi_arr), hi_arr, lo_arr + 1e3)
    range_arr = hi_finite - lo_arr + 1e-30
    pen_sigma = range_arr * 0.01

    r_fine = _build_fine_grid(r_max, n_fine)
    p0 = np.array([lambda_c_init_m, Y_line_init, B_sci_init])

    def _residuals(p):
        lc, Y, B = p
        model_fine = _airglow_model_fine(r_fine, lc, Y, B, cal)
        model_bins = np.interp(r_good, r_fine, model_fine)
        data_r = (profile_good - model_bins) / sigma_good
        below = np.maximum(0.0, lo_arr - p) / pen_sigma
        above = np.maximum(0.0, p - hi_arr) / pen_sigma
        return np.append(data_r, below + above)

    return least_squares(
        _residuals, p0, method="lm",
        ftol=1e-12, xtol=1e-12, gtol=1e-12,
        max_nfev=50_000,
    )


# ---------------------------------------------------------------------------
# Covariance and uncertainties (spec §4.5)
# ---------------------------------------------------------------------------

def _compute_jacobian(
    r_good: np.ndarray,
    sigma_good: np.ndarray,
    r_fine: np.ndarray,
    lambda_c_m: float,
    Y_line: float,
    B_sci: float,
    cal: CalibrationResult,
) -> np.ndarray:
    """
    Analytical Jacobian of weighted residuals w.r.t. [lambda_c, Y_line, B_sci].

    Uses FSR/1000 finite-difference step for lambda_c (avoids the near-zero
    derivative from scipy's default relative step h ≈ sqrt(eps) × lambda_c ≈ 1 FSR).
    """
    step_lc = ETALON_FSR_OI_M / 1000.0

    model_hi = np.interp(
        r_good, r_fine,
        _airglow_model_fine(r_fine, lambda_c_m + step_lc, Y_line, B_sci, cal),
    )
    model_lo = np.interp(
        r_good, r_fine,
        _airglow_model_fine(r_fine, lambda_c_m - step_lc, Y_line, B_sci, cal),
    )
    d_dlambda = (model_hi - model_lo) / (2.0 * step_lc)

    airy_fine = airy_modified(
        r_fine, lambda_c_m,
        t=cal.t_m, R_refl=cal.R_refl, alpha=cal.alpha, n=1.0,
        r_max=float(r_fine[-1]),
        I0=cal.I0, I1=cal.I1, I2=cal.I2,
        sigma0=cal.sigma0, sigma1=cal.sigma1, sigma2=cal.sigma2,
    )
    airy_bins = np.interp(r_good, r_fine, airy_fine)

    return np.column_stack([
        -d_dlambda / sigma_good,
        -airy_bins / sigma_good,
        -np.ones(len(r_good)) / sigma_good,
    ])


def _compute_uncertainties(
    n_good: int,
    chi2_red: float,
    r_good: np.ndarray,
    sigma_good: np.ndarray,
    r_fine: np.ndarray,
    lambda_c_m: float,
    Y_line: float,
    B_sci: float,
    cal: CalibrationResult,
) -> tuple:
    """
    Returns (sigmas[3], cov[3,3], flags) via Jacobian covariance.

    cov = chi2_red × (JᵀJ)⁻¹; pseudoinverse fallback if cond > 1e14.
    """
    flags = AirglowFitFlags.GOOD
    J = _compute_jacobian(r_good, sigma_good, r_fine, lambda_c_m, Y_line, B_sci, cal)
    JTJ = J.T @ J

    try:
        cond = np.linalg.cond(JTJ)
        if cond < 1e14:
            JTJ_inv = np.linalg.inv(JTJ)
        else:
            JTJ_inv = np.linalg.pinv(JTJ, rcond=1e-10)
            flags |= AirglowFitFlags.STDERR_NONE
            log.warning(f"JTJ near-singular (cond={cond:.2e}): using pseudoinverse")
        cov = chi2_red * JTJ_inv
        sigmas = np.sqrt(np.maximum(np.diag(cov), 0.0))
        if not np.all(np.isfinite(sigmas)):
            raise np.linalg.LinAlgError("non-finite after pinv")
    except (np.linalg.LinAlgError, ValueError):
        sigmas = np.full(3, np.inf)
        cov = np.full((3, 3), np.inf)
        flags |= AirglowFitFlags.STDERR_NONE

    return sigmas, cov, flags


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_airglow_fringe(
    profile: FringeProfile,
    cal: CalibrationResult,
    n_fine: int = _N_FINE,
) -> AirglowFitResult:
    """
    Invert an OI 630 nm airglow FringeProfile to recover v_rel (spec §4).

    Parameters
    ----------
    profile : FringeProfile
        From M03 reduce_science_frame(). Dark subtraction must be applied.
    cal : CalibrationResult
        From F01. All 10 instrument params used as fixed constants.
    n_fine : int
        Points in the fine uniform-r forward-model grid (default 500).

    Returns
    -------
    AirglowFitResult

    Raises
    ------
    ValueError
        If profile.quality_flags & CENTRE_FAILED, or fewer than 10 good bins.
    RuntimeError
        If parameter stderrs remain non-finite after pseudoinverse fallback.
    """
    # ---- §4.1 Validate inputs ----
    if profile.quality_flags & QualityFlags.CENTRE_FAILED:
        raise ValueError("FringeProfile has CENTRE_FAILED flag — cannot invert")

    good_mask = (
        ~profile.masked
        & np.isfinite(profile.sigma_profile)
        & (profile.sigma_profile > 0)
        & np.isfinite(profile.profile)
    )
    n_good = int(np.sum(good_mask))
    if n_good < 10:
        raise ValueError(f"Only {n_good} unmasked bins — need ≥ 10")

    result_flags = AirglowFitFlags.GOOD
    if cal.quality_flags != 0:
        result_flags |= AirglowFitFlags.CAL_QUALITY_DEGRADED

    r_good = profile.r_grid[good_mask]
    profile_good = profile.profile[good_mask]
    sigma_raw = profile.sigma_profile[good_mask].copy()
    r_max = float(profile.r_max_px)

    median_signal = float(np.median(profile_good))
    sigma_floor = max(1.0, median_signal * 0.005)
    sigma_good = np.maximum(sigma_raw, sigma_floor)

    # ---- §4.2 Brute-force scan centred at OI_WAVELENGTH_VACUUM_M ----
    lc_start = float(OI_WAVELENGTH_VACUUM_M)
    r_fine = _build_fine_grid(r_max, n_fine)
    lambda_c_best, chi2_scan, Y_line_init, B_sci_init, scan_flags = _lambda_c_scan(
        r_good, profile_good, sigma_good, r_max, cal,
        n_scan=_N_SCAN, n_fine=n_fine,
    )
    result_flags |= scan_flags

    # Clamp B_sci_init to non-negative (soft bound at 0)
    if B_sci_init <= 0:
        B_sci_init = max(0.0, float(np.percentile(profile_good, 5)) * 0.1)
    if Y_line_init <= 0:
        Y_line_init = 1.0

    # ---- §4.3 LM fit ----
    lm_result = _run_airglow_lm(
        r_good, profile_good, sigma_good, r_max, cal,
        lambda_c_init_m=lambda_c_best,
        Y_line_init=Y_line_init,
        B_sci_init=B_sci_init,
        n_fine=n_fine,
    )

    lambda_c_m = float(lm_result.x[0])
    Y_line = float(lm_result.x[1])
    B_sci = float(lm_result.x[2])

    # ---- §4.4 Chi2 (data residuals only, exclude penalty rows) ----
    model_final = np.interp(
        r_good, r_fine,
        _airglow_model_fine(r_fine, lambda_c_m, Y_line, B_sci, cal),
    )
    data_resid = (profile_good - model_final) / sigma_good
    dof = max(n_good - _N_FREE, 1)
    chi2_red = float(np.sum(data_resid ** 2)) / dof

    # ---- §4.5 Uncertainties ----
    sigmas, cov, unc_flags = _compute_uncertainties(
        n_good, chi2_red,
        r_good, sigma_good, r_fine,
        lambda_c_m, Y_line, B_sci, cal,
    )
    result_flags |= unc_flags

    if not np.all(np.isfinite(sigmas)):
        result_flags |= AirglowFitFlags.STDERR_NONE
        raise RuntimeError(
            "F02 covariance failure: parameter stderrs non-finite after pseudoinverse."
        )

    sigma_lambda_c_m, sigma_Y_line, sigma_B_sci = sigmas

    # ---- §4.6 Doppler wind (uses OI_WAVELENGTH_VACUUM_M) ----
    v_rel_ms = SPEED_OF_LIGHT_MS * (lambda_c_m - OI_WAVELENGTH_VACUUM_M) / OI_WAVELENGTH_VACUUM_M
    sigma_v_rel_ms = SPEED_OF_LIGHT_MS * sigma_lambda_c_m / OI_WAVELENGTH_VACUUM_M

    # ---- §4.7 Phase diagnostic (uses OI_WAVELENGTH_M = 629.95e-9 nm) ----
    epsilon_sci = float((2.0 * lambda_c_m / OI_WAVELENGTH_M) % 1.0)
    delta_epsilon = epsilon_sci - float(cal.epsilon_cal)

    # ---- Quality flags ----
    if chi2_red > 10.0:
        result_flags |= AirglowFitFlags.CHI2_VERY_HIGH | AirglowFitFlags.CHI2_HIGH
        log.warning(f"chi2_reduced = {chi2_red:.2f} > 10")
    elif chi2_red > 3.0:
        result_flags |= AirglowFitFlags.CHI2_HIGH
        log.warning(f"chi2_reduced = {chi2_red:.2f} > 3.0")
    elif chi2_red < 0.5:
        result_flags |= AirglowFitFlags.CHI2_LOW

    lambda_lo = OI_WAVELENGTH_VACUUM_M - 1.5 * ETALON_FSR_OI_M
    lambda_hi = OI_WAVELENGTH_VACUUM_M + 1.5 * ETALON_FSR_OI_M
    if abs(lambda_c_m - lambda_lo) < 1e-15 or abs(lambda_c_m - lambda_hi) < 1e-15:
        result_flags |= AirglowFitFlags.LAMBDA_C_AT_BOUND

    snr_estimate = (
        (float(np.max(profile_good)) - float(np.min(profile_good))) / abs(B_sci)
        if abs(B_sci) > 0 else np.inf
    )
    if snr_estimate < 1.0:
        result_flags |= AirglowFitFlags.LOW_SNR

    converged = bool(lm_result.success or lm_result.cost < 1e-10)

    return AirglowFitResult(
        lambda_c_m           = lambda_c_m,
        sigma_lambda_c_m     = sigma_lambda_c_m,
        two_sigma_lambda_c_m = 2.0 * sigma_lambda_c_m,
        v_rel_ms             = v_rel_ms,
        sigma_v_rel_ms       = sigma_v_rel_ms,
        two_sigma_v_rel_ms   = 2.0 * sigma_v_rel_ms,
        Y_line               = Y_line,
        sigma_Y_line         = sigma_Y_line,
        two_sigma_Y_line     = 2.0 * sigma_Y_line,
        B_sci                = B_sci,
        sigma_B_sci          = sigma_B_sci,
        two_sigma_B_sci      = 2.0 * sigma_B_sci,
        chi2_reduced         = chi2_red,
        n_bins_used          = n_good,
        n_params_free        = _N_FREE,
        converged            = converged,
        quality_flags        = result_flags,
        epsilon_sci          = epsilon_sci,
        delta_epsilon        = delta_epsilon,
        calibration_t_m      = float(cal.t_m),
        calibration_epsilon_cal = float(cal.epsilon_cal),
        lambda_c_scan_init_m = lc_start,
        lambda_c_lm_init_m   = lambda_c_best,
    )
