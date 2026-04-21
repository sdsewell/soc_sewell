"""
M06 — Airglow fringe inversion: recovers v_rel from OI 630 nm FringeProfile.

Spec:        docs/specs/S15_m06_airglow_inversion_2026-04-06.md
Spec date:   2026-04-06
Generated:   2026-04-07
Tool:        Claude Code
Last tested: 2026-04-07  (8/8 tests pass)
Depends on:  src.constants, src.fpi.m01_*, src.fpi.m03_*, src.fpi.m05_*
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import least_squares

from src.fpi.m01_airy_forward_model_2026_04_05 import airy_modified
from src.constants import (
    OI_WAVELENGTH_VACUUM_M as OI_WAVELENGTH_M,
    SPEED_OF_LIGHT_MS,
    ETALON_FSR_OI_M,
    WIND_BIAS_BUDGET_MS,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Quality-flag bitmask
# ---------------------------------------------------------------------------

class AirglowFitFlags:
    """Bitmask quality flags for AirglowFitResult. Uses bits 4+ per S04."""
    GOOD                 = 0x000
    FIT_FAILED           = 0x001   # global S04 flag — LM did not converge
    CHI2_HIGH            = 0x002   # global S04 flag — chi2 > 3.0
    CHI2_VERY_HIGH       = 0x004   # global S04 flag — chi2 > 10.0
    CHI2_LOW             = 0x008   # global S04 flag — chi2 < 0.5
    SCAN_AMBIGUOUS       = 0x010   # brute-force scan has two minima < 10% apart
    LAMBDA_C_AT_BOUND    = 0x020   # lambda_c hit its bound (possible FSR jump)
    STDERR_NONE          = 0x040   # any stderr is None (singular covariance)
    LOW_SNR              = 0x080   # estimated SNR < 1.0 (Y_line / B_sci < 1)
    CAL_QUALITY_DEGRADED = 0x100   # CalibrationResult had non-GOOD quality flags


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class AirglowFitResult:
    """
    Output of M06 airglow inversion.
    Passed to M07 (wind retrieval) for LOS-to-vector decomposition.
    """
    # Primary output — line centre and wind
    lambda_c_m:           float
    sigma_lambda_c_m:     float
    two_sigma_lambda_c_m: float   # exactly 2 × sigma_lambda_c_m  (S04)

    v_rel_ms:             float
    sigma_v_rel_ms:       float
    two_sigma_v_rel_ms:   float   # exactly 2 × sigma_v_rel_ms    (S04)

    # Other fitted parameters (diagnostics)
    Y_line:           float
    sigma_Y_line:     float
    two_sigma_Y_line: float

    B_sci:            float
    sigma_B_sci:      float
    two_sigma_B_sci:  float

    # Fit quality
    chi2_reduced:   float
    n_bins_used:    int
    n_params_free:  int     # always 3 for M06
    converged:      bool
    quality_flags:  int

    # Phase relationship to calibration (diagnostic)
    epsilon_sci:    float   # (2 × lambda_c_m / OI_WAVELENGTH_M) mod 1
    delta_epsilon:  float   # epsilon_sci − epsilon_cal

    # Input traceability
    calibration_t_m:         float
    calibration_epsilon_cal: float

    # LM scan diagnostics
    lambda_c_scan_init_m: float   # λ_c at start of brute-force scan
    lambda_c_lm_init_m:   float   # λ_c passed to LM after scan


# ---------------------------------------------------------------------------
# Forward model helpers
# ---------------------------------------------------------------------------

def _airglow_model_fine(r_fine: np.ndarray, lambda_c_m: float,
                        Y_line: float, B_sci: float, cal) -> np.ndarray:
    """
    Delta-function airglow fringe at lambda_c_m on a fine uniform-r grid.

    Parameters
    ----------
    r_fine     : fine uniform-r grid, pixels
    lambda_c_m : Doppler-shifted line centre, metres
    Y_line     : line intensity scale factor
    B_sci      : science frame CCD bias, ADU
    cal        : CalibrationResult — all 10 instrument parameters fixed

    Returns model profile on r_fine (before bin-averaging).
    """
    r_max = float(r_fine[-1])
    airy = airy_modified(
        r_fine, lambda_c_m,
        t=cal.t_m, R_refl=cal.R_refl, alpha=cal.alpha, n=1.0,
        r_max=r_max,
        I0=cal.I0, I1=cal.I1, I2=cal.I2,
        sigma0=cal.sigma0, sigma1=cal.sigma1, sigma2=cal.sigma2,
    )
    return Y_line * airy + B_sci


def _bin_average(model_fine: np.ndarray, r_fine: np.ndarray,
                 r_bins: np.ndarray) -> np.ndarray:
    """
    Linearly interpolate model_fine (on r_fine) to r_bins.

    Equivalent to bin-averaging if r_fine is dense enough (≥ 500 pts).
    Matches M05's strategy: evaluate on fine grid, then np.interp to M03
    bin-centre radii.
    """
    return np.interp(r_bins, r_fine, model_fine)


def _build_fine_grid(r_max: float, n_fine: int = 500) -> np.ndarray:
    """Return uniform r grid [0, r_max] with n_fine points."""
    return np.linspace(0.0, r_max, n_fine)


# ---------------------------------------------------------------------------
# Brute-force scan for λ_c initialisation
# ---------------------------------------------------------------------------

def _lambda_c_scan(
    r_good: np.ndarray,
    profile_good: np.ndarray,
    sigma_good: np.ndarray,
    r2_good: np.ndarray,
    r_max: float,
    cal,
    n_scan: int = 200,
    n_fine: int = 500,
) -> Tuple[float, float, int]:
    """
    Scan lambda_c over one FSR to find the best initial guess.

    At each of n_scan evenly-spaced lambda_c values across
    [OI_WAVELENGTH_M ± FSR/2], analytically solves for Y_line and B_sci
    (linear given fixed lambda_c) and records chi2.

    Returns
    -------
    lambda_c_best : float — λ_c with minimum chi2
    chi2_min      : float
    scan_flags    : int   — AirglowFitFlags.SCAN_AMBIGUOUS if second-best
                           chi2 is within 10% of minimum
    """
    lc_lo = OI_WAVELENGTH_M - 0.5 * ETALON_FSR_OI_M
    lc_hi = OI_WAVELENGTH_M + 0.5 * ETALON_FSR_OI_M
    scan_grid = np.linspace(lc_lo, lc_hi, n_scan)

    r_fine = _build_fine_grid(r_max, n_fine)
    n_good = len(r_good)

    chi2_arr = np.full(n_scan, np.inf)

    for k, lc in enumerate(scan_grid):
        # Build Airy profile on fine grid, interpolate to r_good
        airy_fine = airy_modified(
            r_fine, lc,
            t=cal.t_m, R_refl=cal.R_refl, alpha=cal.alpha, n=1.0,
            r_max=r_max,
            I0=cal.I0, I1=cal.I1, I2=cal.I2,
            sigma0=cal.sigma0, sigma1=cal.sigma1, sigma2=cal.sigma2,
        )
        airy_bins = np.interp(r_good, r_fine, airy_fine)   # shape (n_good,)

        # Analytic least-squares: model = Y_line * airy_bins + B_sci
        # X = [airy_bins, ones], solve X @ [Y_line, B_sci]^T
        w = 1.0 / sigma_good                               # shape (n_good,)
        A = np.column_stack([airy_bins * w, w])            # (n_good, 2)
        b = profile_good * w                               # (n_good,)
        # Normal equations: (A^T A) p = A^T b
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

    # Best and second-best
    best_idx = int(np.argmin(chi2_arr))
    chi2_min = float(chi2_arr[best_idx])
    lambda_c_best = float(scan_grid[best_idx])

    # SCAN_AMBIGUOUS: any other point within 10% of best chi2
    scan_flags = AirglowFitFlags.GOOD
    alt_mask = np.ones(n_scan, dtype=bool)
    alt_mask[best_idx] = False
    chi2_alt = chi2_arr[alt_mask]
    if len(chi2_alt) > 0 and np.nanmin(chi2_alt) < chi2_min * 1.10:
        scan_flags |= AirglowFitFlags.SCAN_AMBIGUOUS
        log.warning(
            f"SCAN_AMBIGUOUS: second-best chi2 within 10% of minimum "
            f"({np.nanmin(chi2_alt):.4f} vs {chi2_min:.4f})"
        )

    return lambda_c_best, chi2_min, scan_flags


# ---------------------------------------------------------------------------
# LM fit
# ---------------------------------------------------------------------------

def _run_airglow_lm(
    r_good: np.ndarray,
    profile_good: np.ndarray,
    sigma_good: np.ndarray,
    r_max: float,
    cal,
    lambda_c_init_m: float,
    Y_line_init: float,
    B_sci_init: float,
    n_fine: int = 500,
):
    """
    Run LM fit over {lambda_c_m, Y_line, B_sci}.

    Uses scipy.optimize.least_squares(method='lm').
    Soft-bound penalty residuals (one per free parameter) fire linearly
    outside the effective bounds from Section 5.
    """
    # Bounds
    lambda_lo = OI_WAVELENGTH_M - 1.5 * ETALON_FSR_OI_M
    lambda_hi = OI_WAVELENGTH_M + 1.5 * ETALON_FSR_OI_M
    Y_lo,  Y_hi  = 0.0, np.inf        # effectively unbounded above
    B_lo          = 0.0
    B_hi          = float(np.min(profile_good)) * 1.5 if np.min(profile_good) > 0 else 1e6

    # Clamp B_hi to avoid inf
    if not np.isfinite(B_hi) or B_hi <= 0:
        B_hi = float(np.max(profile_good))

    lo_arr     = np.array([lambda_lo, Y_lo,  B_lo])
    hi_arr     = np.array([lambda_hi, Y_hi,  B_hi])
    # For soft-bound penalty scale: 1% of range (use large value for unbounded)
    hi_finite  = np.where(np.isfinite(hi_arr), hi_arr, lo_arr + 1e3)
    range_arr  = hi_finite - lo_arr + 1e-30
    pen_sigma  = range_arr * 0.01

    r_fine = _build_fine_grid(r_max, n_fine)
    p0 = np.array([lambda_c_init_m, Y_line_init, B_sci_init])

    def _residuals(p):
        lc, Y, B = p
        model_fine = _airglow_model_fine(r_fine, lc, Y, B, cal)
        model_bins = _bin_average(model_fine, r_fine, r_good)
        data_r = (profile_good - model_bins) / sigma_good

        # Soft-bound penalties
        below = np.maximum(0.0, lo_arr - p) / pen_sigma
        above = np.maximum(0.0, p - hi_arr) / pen_sigma
        penalty = below + above
        return np.append(data_r, penalty)

    result = least_squares(
        _residuals, p0, method='lm',
        ftol=1e-12, xtol=1e-12, gtol=1e-12,
        max_nfev=50_000,
    )
    return result


# ---------------------------------------------------------------------------
# Covariance and uncertainties
# ---------------------------------------------------------------------------

def _compute_jacobian_analytical(
    r_good: np.ndarray,
    sigma_good: np.ndarray,
    r_fine: np.ndarray,
    lambda_c_m: float,
    Y_line: float,
    B_sci: float,
    cal,
) -> np.ndarray:
    """
    Compute analytical Jacobian of weighted residuals w.r.t. [lambda_c, Y_line, B_sci].

    Uses a physically motivated step for lambda_c (FSR/1000 ≈ 1e-14 m ≈ 4.7 m/s)
    to avoid the near-zero numerical derivative that arises when scipy's LM uses
    a relative step h ≈ sqrt(eps) × lambda_c ≈ 1 FSR.

    Columns:
        0 — d(residuals)/d(lambda_c)   via finite difference (step = FSR/1000)
        1 — d(residuals)/d(Y_line)     = -airy(r)/sigma  (analytic)
        2 — d(residuals)/d(B_sci)      = -1/sigma         (analytic)

    Sign convention: residuals = (data - model) / sigma, so
        J_ij = -∂model_i / ∂p_j / sigma_i
    """
    step_lc = ETALON_FSR_OI_M / 1000.0   # ≈ 9.9e-18 m → ≈ 4.7 m/s

    # Finite-difference column for lambda_c
    model_hi = _bin_average(
        _airglow_model_fine(r_fine, lambda_c_m + step_lc, Y_line, B_sci, cal),
        r_fine, r_good,
    )
    model_lo = _bin_average(
        _airglow_model_fine(r_fine, lambda_c_m - step_lc, Y_line, B_sci, cal),
        r_fine, r_good,
    )
    d_dlambda = (model_hi - model_lo) / (2.0 * step_lc)   # ∂model/∂lambda_c

    # Analytic columns for Y_line and B_sci
    airy_fine = airy_modified(
        r_fine, lambda_c_m,
        t=cal.t_m, R_refl=cal.R_refl, alpha=cal.alpha, n=1.0,
        r_max=float(r_fine[-1]),
        I0=cal.I0, I1=cal.I1, I2=cal.I2,
        sigma0=cal.sigma0, sigma1=cal.sigma1, sigma2=cal.sigma2,
    )
    airy_bins = np.interp(r_good, r_fine, airy_fine)   # ∂model/∂Y_line

    # J_ij = -∂model_i/∂p_j / sigma_i   (sign: residual = (data-model)/sigma)
    J = np.column_stack([
        -d_dlambda / sigma_good,   # d(residual)/d(lambda_c)
        -airy_bins / sigma_good,   # d(residual)/d(Y_line)
        -np.ones(len(r_good)) / sigma_good,  # d(residual)/d(B_sci)
    ])
    return J


def _compute_uncertainties(
    lm_result,
    n_good: int,
    chi2_red: float,
    r_good: np.ndarray,
    sigma_good: np.ndarray,
    r_fine: np.ndarray,
    lambda_c_m: float,
    Y_line: float,
    B_sci: float,
    cal,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Compute covariance and 1σ stderrs using an analytically recomputed Jacobian.

    The LM Jacobian for lambda_c is unreliable because scipy's default relative
    step h ≈ sqrt(eps) × lambda_c ≈ 1 FSR averages out the fringe sensitivity.
    We replace it with a finite-difference Jacobian using step = FSR/1000 ≈ 4.7 m/s.

    Returns
    -------
    sigma   : ndarray shape (3,)  — stderrs for [lambda_c, Y_line, B_sci]
    cov     : ndarray shape (3,3) — covariance matrix
    flags   : int                 — AirglowFitFlags.STDERR_NONE if singular
    """
    flags = AirglowFitFlags.GOOD

    # Recompute Jacobian with physically appropriate step sizes
    J = _compute_jacobian_analytical(
        r_good, sigma_good, r_fine,
        lambda_c_m, Y_line, B_sci, cal,
    )

    n_params = 3
    s2 = chi2_red

    JTJ = J.T @ J
    try:
        cond = np.linalg.cond(JTJ)
        if cond < 1e14:
            JTJ_inv = np.linalg.inv(JTJ)
        else:
            JTJ_inv = np.linalg.pinv(JTJ, rcond=1e-10)
            flags |= AirglowFitFlags.STDERR_NONE
            log.warning(f"JTJ near-singular (cond={cond:.2e}): using pseudoinverse")
        cov    = s2 * JTJ_inv
        sigmas = np.sqrt(np.maximum(np.diag(cov), 0.0))
        if not np.all(np.isfinite(sigmas)):
            raise np.linalg.LinAlgError("non-finite after pinv")
    except (np.linalg.LinAlgError, ValueError):
        sigmas = np.full(n_params, np.inf)
        cov    = np.full((n_params, n_params), np.inf)
        flags |= AirglowFitFlags.STDERR_NONE

    return sigmas, cov, flags


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------

def fit_airglow_fringe(
    profile,
    cal,
    n_fine: int = 500,
) -> AirglowFitResult:
    """
    Invert an OI 630 nm science FringeProfile to recover v_rel.

    Parameters
    ----------
    profile : FringeProfile
        From M03 reduce_science_frame(). Must have dark subtraction applied.
    cal : CalibrationResult
        From M05 fit_calibration_fringe(). All 10 instrument parameters
        are used as fixed constants.
    n_fine : int
        Points in the fine uniform-r grid for forward model. Default 500.

    Returns
    -------
    AirglowFitResult

    Raises
    ------
    ValueError
        If profile.quality_flags & CENTRE_FAILED, or fewer than 10 good bins.
    RuntimeError
        If parameter stderrs are non-finite after pseudoinverse fallback.
    """
    from src.fpi.m03_annular_reduction_2026_04_06 import QualityFlags

    # ---- Step 0: Validate inputs ----
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

    # Extract good bins
    r_good       = profile.r_grid[good_mask]
    r2_good      = profile.r2_grid[good_mask]
    profile_good = profile.profile[good_mask]
    sigma_raw    = profile.sigma_profile[good_mask].copy()
    r_max        = float(profile.r_max_px)

    # ---- Sigma floor (same as M05: 0.5% of median signal) ----
    median_signal = float(np.median(profile_good))
    sigma_floor   = max(1.0, median_signal * 0.005)
    sigma_good    = np.maximum(sigma_raw, sigma_floor)

    # ---- Step 1: Brute-force scan for λ_c ----
    lc_start = float(OI_WAVELENGTH_M)   # scan centred at rest wavelength
    lambda_c_best, chi2_scan, scan_flags = _lambda_c_scan(
        r_good, profile_good, sigma_good, r2_good, r_max, cal,
        n_scan=211, n_fine=n_fine,
    )
    result_flags |= scan_flags

    # ---- Initial estimates for Y_line and B_sci ----
    r_fine = _build_fine_grid(r_max, n_fine)
    airy_fine_init = airy_modified(
        r_fine, lambda_c_best,
        t=cal.t_m, R_refl=cal.R_refl, alpha=cal.alpha, n=1.0,
        r_max=r_max,
        I0=cal.I0, I1=cal.I1, I2=cal.I2,
        sigma0=cal.sigma0, sigma1=cal.sigma1, sigma2=cal.sigma2,
    )
    airy_bins_init = np.interp(r_good, r_fine, airy_fine_init)
    airy_max = float(np.max(airy_bins_init)) if np.max(airy_bins_init) > 0 else 1.0
    Y_line_init = float(np.max(profile_good)) / airy_max
    B_sci_init  = float(np.min(profile_good)) * 0.8

    # Ensure B_sci_init > 0
    if B_sci_init <= 0:
        B_sci_init = max(1.0, float(np.percentile(profile_good, 5)))

    # ---- Step 2: LM fit ----
    lm_result = _run_airglow_lm(
        r_good, profile_good, sigma_good, r_max, cal,
        lambda_c_init_m=lambda_c_best,
        Y_line_init=Y_line_init,
        B_sci_init=B_sci_init,
        n_fine=n_fine,
    )

    lambda_c_m = float(lm_result.x[0])
    Y_line     = float(lm_result.x[1])
    B_sci      = float(lm_result.x[2])

    # ---- Chi2 from data residuals only (exclude penalty rows) ----
    model_final_fine = _airglow_model_fine(r_fine, lambda_c_m, Y_line, B_sci, cal)
    model_final_bins = _bin_average(model_final_fine, r_fine, r_good)
    data_resid       = (profile_good - model_final_bins) / sigma_good
    dof              = max(n_good - 3, 1)
    chi2_sum         = float(np.sum(data_resid ** 2))
    chi2_red         = chi2_sum / dof

    # ---- Step 3: Covariance and stderrs ----
    sigmas, cov, unc_flags = _compute_uncertainties(
        lm_result, n_good, chi2_red,
        r_good, sigma_good, r_fine,
        lambda_c_m, Y_line, B_sci, cal,
    )
    result_flags |= unc_flags

    if not np.all(np.isfinite(sigmas)):
        result_flags |= AirglowFitFlags.STDERR_NONE
        raise RuntimeError(
            "M06 covariance failure: parameter stderrs are non-finite "
            "even after pseudoinverse fallback."
        )

    sigma_lambda_c_m, sigma_Y_line, sigma_B_sci = sigmas

    # ---- Step 4: Doppler wind and phase ----
    v_rel_ms       = SPEED_OF_LIGHT_MS * (lambda_c_m - OI_WAVELENGTH_M) / OI_WAVELENGTH_M
    sigma_v_rel_ms = SPEED_OF_LIGHT_MS * sigma_lambda_c_m / OI_WAVELENGTH_M

    epsilon_sci   = float((2.0 * lambda_c_m / OI_WAVELENGTH_M) % 1.0)
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

    lambda_c_lo = OI_WAVELENGTH_M - 1.5 * ETALON_FSR_OI_M
    lambda_c_hi = OI_WAVELENGTH_M + 1.5 * ETALON_FSR_OI_M
    if abs(lambda_c_m - lambda_c_lo) < 1e-15 or abs(lambda_c_m - lambda_c_hi) < 1e-15:
        result_flags |= AirglowFitFlags.LAMBDA_C_AT_BOUND

    snr_estimate = (
        (float(np.max(profile_good)) - float(np.min(profile_good))) / abs(B_sci)
        if abs(B_sci) > 0 else np.inf
    )
    if snr_estimate < 1.0:
        result_flags |= AirglowFitFlags.LOW_SNR

    converged = bool(lm_result.success or lm_result.cost < 1e-10)

    return AirglowFitResult(
        # Line centre
        lambda_c_m           = lambda_c_m,
        sigma_lambda_c_m     = sigma_lambda_c_m,
        two_sigma_lambda_c_m = 2.0 * sigma_lambda_c_m,
        # Wind
        v_rel_ms             = v_rel_ms,
        sigma_v_rel_ms       = sigma_v_rel_ms,
        two_sigma_v_rel_ms   = 2.0 * sigma_v_rel_ms,
        # Intensity scale
        Y_line               = Y_line,
        sigma_Y_line         = sigma_Y_line,
        two_sigma_Y_line     = 2.0 * sigma_Y_line,
        # Bias
        B_sci                = B_sci,
        sigma_B_sci          = sigma_B_sci,
        two_sigma_B_sci      = 2.0 * sigma_B_sci,
        # Quality
        chi2_reduced         = chi2_red,
        n_bins_used          = n_good,
        n_params_free        = 3,
        converged            = converged,
        quality_flags        = result_flags,
        # Phase
        epsilon_sci          = epsilon_sci,
        delta_epsilon        = delta_epsilon,
        # Traceability
        calibration_t_m         = float(cal.t_m),
        calibration_epsilon_cal = float(cal.epsilon_cal),
        # Scan diagnostics
        lambda_c_scan_init_m = lc_start,
        lambda_c_lm_init_m   = lambda_c_best,
    )
