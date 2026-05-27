"""
fpi_cal_lib.py — WindCube FPI calibration library.

Version:  1.3
Date:     2026-05-17
Spec:     docs/specs/CAL01_fpi_cal_pipeline_2026-05-17.md

Single authoritative library containing all FPI calibration physics.
No tkinter. No plt.show(). Safe to import in headless environments.
All UX lives in run_cal_pipeline.py only.
"""

from __future__ import annotations

import datetime
import logging
import pathlib
import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit, least_squares, minimize
from scipy.signal import find_peaks

from windcube.constants import (
    ETALON_N,             # refractive index of gap (= 1.0, air)
    OI_WAVELENGTH_AIR_M,  # OI 630.0304 nm rest wavelength (NIST)
    NE_WAVELENGTH_1_AIR_M,  # Ne 640.2248 nm strong line (Burns 1950)
    NE_WAVELENGTH_2_AIR_M,  # Ne 638.2991 nm weak line  (Burns 1950)
    NE_INTENSITY_1,       # neon line intensity ratio (= 1.0)
    NE_INTENSITY_2,       # neon line intensity ratio (measured by H05)
    SPEED_OF_LIGHT_MS,    # exact SI value
)
# NOTE: ETALON_GAP_M, ETALON_R_INSTRUMENT, ALPHA_RAD_PX, R_MAX_PX are
# intentionally NOT imported here.  Instrument values (t, alpha, R) are
# always supplied via the Tolansky seed chain at runtime; importing them
# from constants would give a false impression they are used as defaults.


# ===========================================================================
# Section A — Centre finder
# (copied from src/processing/center_finder.py)
# ===========================================================================

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CentreResult:
    """Output of the two-pass azimuthal variance centre finder."""
    # Pass 2 — Nelder-Mead refined result (authoritative)
    cx:           float   # fringe centre x, pixels
    cy:           float   # fringe centre y, pixels
    sigma_cx:     float   # 1σ uncertainty on cx, pixels
    sigma_cy:     float   # 1σ uncertainty on cy, pixels
    two_sigma_cx: float   # 2 × sigma_cx
    two_sigma_cy: float   # 2 × sigma_cy
    cost_at_min:  float   # variance cost at the Nelder-Mead minimum
    # Pass 1 — coarse grid search result (seed for pass 2)
    grid_cx:      float   # best-grid cx, pixels
    grid_cy:      float   # best-grid cy, pixels
    grid_cost:    float   # variance cost at the grid minimum


# ---------------------------------------------------------------------------
# Azimuthal variance cost function
# ---------------------------------------------------------------------------

def _variance_cost(
    cx: float,
    cy: float,
    image: np.ndarray,
    r_min_sq: float,
    r_max_sq: float,
    n_var_bins: int,
) -> float:
    """
    Sum of per-bin intensity variance over the annular region.

    Minimum at the true optical axis.  Uses biased variance per bin and
    np.bincount for vectorised computation (no Python loop over bins).
    """
    H, W = image.shape
    row_c, col_c = np.mgrid[0:H, 0:W]
    r2 = (row_c.astype(np.float64) - cy)**2 + (col_c.astype(np.float64) - cx)**2

    mask = (r2 >= r_min_sq) & (r2 < r_max_sq)
    r2_valid  = r2[mask]
    adu_valid = image[mask].astype(np.float64)

    if r2_valid.size < n_var_bins:
        return 1e30

    dr2 = r_max_sq / n_var_bins
    bin_idx = np.floor(r2_valid / dr2).astype(np.int32)
    bin_idx = np.clip(bin_idx, 0, n_var_bins - 1)

    counts = np.bincount(bin_idx, minlength=n_var_bins).astype(np.float64)
    sum_a  = np.bincount(bin_idx, weights=adu_valid,    minlength=n_var_bins)
    sum_a2 = np.bincount(bin_idx, weights=adu_valid**2, minlength=n_var_bins)

    good    = counts >= 2
    mean_sq = np.where(good, sum_a2 / np.where(good, counts, 1.0), 0.0)
    sq_mean = np.where(good, (sum_a / np.where(good, counts, 1.0))**2, 0.0)
    var_bins = np.where(good, mean_sq - sq_mean, 0.0)
    return float(np.sum(var_bins))


# ---------------------------------------------------------------------------
# Two-pass azimuthal variance minimisation
# ---------------------------------------------------------------------------

def azimuthal_variance_centre(
    image: np.ndarray,
    cx_seed: float,
    cy_seed: float,
    var_r_min_px: float = 5.0,
    var_r_max_px: Optional[float] = None,
    var_n_bins: int = 250,
    var_search_px: float = 15.0,
) -> tuple[float, float, float]:
    """
    Two-pass azimuthal variance minimisation.

    Pass 1 — coarse grid search over ±var_search_px with step
             max(2.0, var_search_px / 8) px.
    Pass 2 — Nelder-Mead from best grid point with small initial simplex.

    Returns (cx_fine, cy_fine, cost_min, grid_cx, grid_cy, grid_cost).

    Notes
    -----
    Single-pass Nelder-Mead fails on this cost function (narrow basin ~1 px
    wide); the two-pass approach is mandatory per spec Section 6.2.
    """
    H, W = image.shape
    if var_r_max_px is None:
        var_r_max_px = min(H, W) // 2 - 10

    r_min_sq = var_r_min_px ** 2
    r_max_sq = var_r_max_px ** 2

    def cost(xy):
        return _variance_cost(xy[0], xy[1], image, r_min_sq, r_max_sq, var_n_bins)

    # --- Pass 1: coarse grid search ---
    offsets   = np.linspace(-var_search_px, var_search_px, 20)
    grid_step = float(offsets[1] - offsets[0])   # used for simplex_r below

    best_cost = np.inf
    best_cx   = cx_seed
    best_cy   = cy_seed

    for dy in offsets:
        for dx in offsets:
            c = cost([cx_seed + dx, cy_seed + dy])
            if c < best_cost:
                best_cost = c
                best_cx   = cx_seed + dx
                best_cy   = cy_seed + dy

    # --- Pass 2: Nelder-Mead from best grid point ---
    simplex_r = grid_step + 0.5
    x0 = np.array([best_cx, best_cy])
    initial_simplex = np.array([
        x0,
        x0 + [simplex_r, 0.0],
        x0 + [0.0, simplex_r],
    ])

    result = minimize(
        cost, x0,
        method="Nelder-Mead",
        options={
            "initial_simplex": initial_simplex,
            "xatol": 0.02,
            "fatol": 1.0,
            "maxiter": 500,
        },
    )

    cx_fine  = float(result.x[0])
    cy_fine  = float(result.x[1])
    cost_min = float(result.fun)
    return cx_fine, cy_fine, cost_min, best_cx, best_cy, best_cost


# ---------------------------------------------------------------------------
# Centre uncertainty from cost-surface curvature
# ---------------------------------------------------------------------------

def estimate_centre_uncertainty(
    cx: float,
    cy: float,
    cost_fn: Callable,
    delta_px: float = 0.1,
) -> tuple[float, float]:
    """
    Estimate 1σ uncertainty on cx and cy from curvature of cost surface.

    σ_cx = sqrt(2 / (d²C/dcx²))  at the minimum.
    Finite-difference second derivative with step delta_px.
    Clamped to [0.02, 5.0] px.
    """
    c0  = cost_fn([cx, cy])
    cxp = cost_fn([cx + delta_px, cy])
    cxm = cost_fn([cx - delta_px, cy])
    cyp = cost_fn([cx, cy + delta_px])
    cym = cost_fn([cx, cy - delta_px])

    d2cx = (cxp - 2.0 * c0 + cxm) / (delta_px ** 2)
    d2cy = (cyp - 2.0 * c0 + cym) / (delta_px ** 2)

    sigma_cx = float(np.sqrt(2.0 / d2cx)) if d2cx > 0 else 5.0
    sigma_cy = float(np.sqrt(2.0 / d2cy)) if d2cy > 0 else 5.0

    sigma_cx = float(np.clip(sigma_cx, 0.02, 5.0))
    sigma_cy = float(np.clip(sigma_cy, 0.02, 5.0))
    return sigma_cx, sigma_cy


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------

def find_centre(
    image: np.ndarray,
    cx_seed: Optional[float] = None,
    cy_seed: Optional[float] = None,
    var_r_min_px: float = 5.0,
    var_r_max_px: Optional[float] = None,
    var_n_bins: int = 250,
    var_search_px: float = 15.0,
) -> CentreResult:
    """
    Run the two-pass azimuthal variance centre finder and return a CentreResult.

    Parameters
    ----------
    image        : 2-D uint16 or float ndarray — ROI containing the fringe pattern
    cx_seed      : initial x guess (pixels); defaults to image centre
    cy_seed      : initial y guess (pixels); defaults to image centre
    var_r_min_px : inner exclusion radius for variance computation (pixels)
    var_r_max_px : outer radius; defaults to min(H,W)//2 - 10
    var_n_bins   : number of r² bins used in variance cost
    var_search_px: half-width of coarse grid search (pixels)
    """
    H, W = image.shape
    if cx_seed is None:
        cx_seed = (W - 1) / 2.0
    if cy_seed is None:
        cy_seed = (H - 1) / 2.0

    # Clip extreme outliers so hot pixels don't bias the variance minimum
    p99_5 = float(np.percentile(image, 99.5))
    image_for_cost = np.clip(image, None, p99_5)

    if var_r_max_px is None:
        var_r_max_px = float(min(H, W) // 2 - 10)

    cx_fine, cy_fine, cost_min, grid_cx, grid_cy, grid_cost = azimuthal_variance_centre(
        image_for_cost, cx_seed, cy_seed,
        var_r_min_px=var_r_min_px,
        var_r_max_px=var_r_max_px,
        var_n_bins=var_n_bins,
        var_search_px=var_search_px,
    )

    r_min_sq = var_r_min_px ** 2
    r_max_sq = var_r_max_px ** 2

    def _cost_fn(xy):
        return _variance_cost(xy[0], xy[1], image_for_cost, r_min_sq, r_max_sq, var_n_bins)

    sigma_cx, sigma_cy = estimate_centre_uncertainty(cx_fine, cy_fine, _cost_fn)

    return CentreResult(
        cx           = cx_fine,
        cy           = cy_fine,
        sigma_cx     = sigma_cx,
        sigma_cy     = sigma_cy,
        two_sigma_cx = 2.0 * sigma_cx,
        two_sigma_cy = 2.0 * sigma_cy,
        cost_at_min  = cost_min,
        grid_cx      = grid_cx,
        grid_cy      = grid_cy,
        grid_cost    = grid_cost,
    )


# ===========================================================================
# Section B — Airy forward model
# (copied from src/fpi/airy_forward_model_2026_05_05.py)
# ===========================================================================

# ---------------------------------------------------------------------------
# InstrumentParams dataclass
# ---------------------------------------------------------------------------

@dataclass
class InstrumentParams:
    """
    Container for all WindCube FPI instrument parameters.

    Passed between M01, M02, M04, M05, M06 to avoid long argument lists.
    Defaults reflect the actual WindCube instrument as characterised by
    the Tolansky analysis and FlatSat calibration.

    IMPORTANT: alpha = 1.6071e-4 rad/px is the 2×2 binned value from the
    Tolansky two-line analysis. The Harding paper value (8.5e-5) is for
    a different instrument configuration and must NOT be used here.
    """

    # Etalon
    t:       float = 20.1076267e-3  # gap, metres; H05 TBAL inversion (2026-05-24)
    R_refl:  float = 0.241          # effective reflectivity R1 @ 640.2 nm; H05 TBAL (2026-05-24)
    n:       float = 1.0            # refractive index (air gap)
    alpha:   float = 1.60854e-4     # rad/pixel, 2×2 binned; H05 TBAL (2026-05-24)

    # Intensity envelope
    I0:  float =  1000.0   # average intensity, counts
    I1:  float =    -0.1   # linear vignetting coefficient
    I2:  float =   0.005   # quadratic vignetting coefficient

    # PSF
    sigma0: float =  0.5    # average PSF width, pixels
    sigma1: float =  0.1    # sine variation, pixels
    sigma2: float = -0.05   # cosine variation, pixels

    # CCD
    B:     float = 300.0    # bias pedestal, counts
    r_max: float = 128.0    # max usable radius, pixels (256px image / 2)

    def finesse_coefficient(self) -> float:
        """F = 4R / (1-R)²"""
        return 4.0 * self.R_refl / (1.0 - self.R_refl) ** 2

    def finesse(self) -> float:
        """Instrument finesse = π√R / (1-R)"""
        return np.pi * np.sqrt(self.R_refl) / (1.0 - self.R_refl)

    @property
    def t_m(self) -> float:
        """Alias for t (etalon gap, metres). Matches CalibrationResult.t_m naming."""
        return self.t

    def free_spectral_range(self, wavelength: float) -> float:
        """FSR = λ² / (2nt),  metres"""
        return wavelength ** 2 / (2.0 * self.n * self.t)


# ---------------------------------------------------------------------------
# Core functions — implement in dependency order
# ---------------------------------------------------------------------------

def theta_from_r(
    r: np.ndarray,   # radial positions, pixels, shape (R,)
    alpha: float,    # magnification constant, rad/pixel
) -> np.ndarray:
    """
    Map pixel radius to angle with optical axis.

    θ(r) = arctan(α · r)    (Harding Eq. 3)

    Parameters
    ----------
    r     : radial positions in pixels, shape (R,) or scalar
    alpha : magnification constant, rad/pixel.
            WindCube 2×2 binned: 1.6071e-4 rad/px (from S03 / Tolansky)

    Returns
    -------
    theta : angle in radians, same shape as r
    """
    return np.arctan(alpha * r)


def intensity_envelope(
    r: np.ndarray,   # radial positions, pixels, shape (R,)
    r_max: float,    # maximum radius, pixels
    I0: float,       # average intensity, counts
    I1: float,       # linear falloff coefficient
    I2: float,       # quadratic falloff coefficient
    I3: float = 0.0, # cubic falloff coefficient
) -> np.ndarray:
    """
    Cubic intensity envelope accounting for optical vignetting.

    I(r) = I₀ · (1 + I₁·(r/r_max) + I₂·(r/r_max)² + I₃·(r/r_max)³)

    Extended from Harding (2014) Eq. 4 by adding the cubic term I₃.
    I₃ defaults to 0.0 for backward compatibility with callers that
    pass only the quadratic form.

    Returns
    -------
    I : intensity in counts, same shape as r
    """
    rn = r / r_max
    return I0 * (1.0 + I1 * rn + I2 * rn ** 2 + I3 * rn ** 3)


def airy_ideal(
    r: np.ndarray,
    wavelength: float,
    t,          # float (etalon gap, m) OR InstrumentParams
    R_refl: float = None,
    alpha: float = None,
    n: float = None,
    r_max: float = None,
    I0: float = None,
    I1: float = None,
    I2: float = None,
    I3: float = 0.0,
) -> np.ndarray:
    """
    Ideal (unbroadened) Airy transmission function at a single wavelength.

    A(r; λ) = I(r) / [1 + F · sin²(π · 2nt·cos(θ(r)) / λ)]

    Accepts two calling conventions:
      airy_ideal(r, lam, params)                    — 3-arg (H01 §6 spec form)
      airy_ideal(r, lam, t, R_refl, ..., I2)        — 10-arg legacy form

    Uses exact cosine (not small-angle approximation).
    Finesse coefficient F = 4R/(1-R)² computed internally.

    Returns
    -------
    A : CCD counts, shape (R,), values in [I(r)/(1+F), I(r)]
    """
    if hasattr(t, "R_refl"):  # duck-type: any InstrumentParams-like object
        p = t
        t, R_refl, alpha, n = p.t, p.R_refl, p.alpha, p.n
        r_max, I0, I1, I2 = p.r_max, p.I0, p.I1, p.I2
        I3 = getattr(p, 'I3', 0.0)
    theta = theta_from_r(r, alpha)
    I_env = intensity_envelope(r, r_max, I0, I1, I2, I3)
    F = 4.0 * R_refl / (1.0 - R_refl) ** 2
    OPD = 2.0 * n * t * np.cos(theta)
    phase = np.pi * OPD / wavelength
    return I_env / (1.0 + F * np.sin(phase) ** 2)


def psf_sigma(
    r: np.ndarray,    # radial positions, pixels, shape (R,)
    r_max: float,     # maximum radius, pixels
    sigma0: float,    # average PSF width, pixels
    sigma1: float,    # sine variation amplitude, pixels
    sigma2: float,    # cosine variation amplitude, pixels
) -> np.ndarray:
    """
    Shift-variant Gaussian PSF width as a function of radius.

    σ(r) = σ₀ + σ₁·sin(π·r/r_max) + σ₂·cos(π·r/r_max)    (Harding Eq. 5)

    The PSF captures optical defects in the etalon and imaging lens that
    cause fringe broadening beyond the ideal Airy function. σ(r) must
    be positive everywhere for physically valid parameters.

    Returns
    -------
    sigma : PSF width in pixels, same shape as r. Always > 0 for valid inputs.
    """
    return sigma0 + sigma1 * np.sin(np.pi * r / r_max) \
                  + sigma2 * np.cos(np.pi * r / r_max)


def airy_modified(
    r: np.ndarray,
    wavelength: float,
    t,          # float (etalon gap, m) OR InstrumentParams
    R_refl: float = None,
    alpha: float = None,
    n: float = None,
    r_max: float = None,
    I0: float = None,
    I1: float = None,
    I2: float = None,
    sigma0: float = None,
    sigma1: float = None,
    sigma2: float = None,
    I3: float = 0.0,
) -> np.ndarray:
    """
    PSF-broadened Airy function at a single wavelength.

    Accepts two calling conventions:
      airy_modified(r, lam, params)               — 3-arg (H01 §6 spec form)
      airy_modified(r, lam, t, R_refl, ..., sigma2) — 13-arg legacy form

    Applies a shift-variant Gaussian convolution to the ideal Airy
    function. Uses the mean sigma across the profile as the filter width
    (mean-sigma approximation, accurate to < 1% for smooth sigma profiles).

    When sigma0 = sigma1 = sigma2 = 0, returns exactly airy_ideal().
    This is enforced by T3.

    Returns
    -------
    A_mod : PSF-broadened CCD counts, shape (R,)
    """
    if hasattr(t, "R_refl"):  # duck-type: any InstrumentParams-like object
        p = t
        t, R_refl, alpha, n = p.t, p.R_refl, p.alpha, p.n
        r_max, I0, I1, I2 = p.r_max, p.I0, p.I1, p.I2
        sigma0, sigma1, sigma2 = p.sigma0, p.sigma1, p.sigma2
        I3 = getattr(p, 'I3', 0.0)
    A_ideal = airy_ideal(r, wavelength, t, R_refl, alpha, n, r_max, I0, I1, I2, I3)
    sigma = psf_sigma(r, r_max, sigma0, sigma1, sigma2)
    sigma_mean = float(np.mean(sigma))
    if sigma_mean < 1e-6:
        return A_ideal
    # gaussian_filter1d sigma is in array elements; convert from pixel units
    r = np.asarray(r)
    dr = float((r[-1] - r[0]) / (len(r) - 1)) if len(r) > 1 else 1.0
    sigma_samples = sigma_mean / dr if dr > 0 else sigma_mean
    return gaussian_filter1d(A_ideal, sigma=sigma_samples)


def phase_correct_gap(
    t_tolansky: float,   # Tolansky-recovered gap, metres
    eps_a: float,        # Tolansky excess fraction for line a (0 <= eps_a < 1)
    lam_a: float,        # wavelength of line a, metres
) -> float:
    """
    Return a phase-corrected effective gap for fringe synthesis.

    The Tolansky analysis returns (t, eps_a) as a self-consistent pair, but
    floating-point evaluation of 2*t/lam_a does not in general recover eps_a
    because t has uncertainty ~200 nm ~ 0.6 FSR.  This function nudges t by
    at most lam_a/4 (~160 nm) so that (2*t_eff/lam_a) % 1 == eps_a exactly,
    anchoring the absolute fringe position for synthesis.

    The correction is purely a synthesis convenience — t_tolansky remains the
    authoritative physical gap for all other purposes.

    Parameters
    ----------
    t_tolansky : Tolansky-recovered etalon gap, metres.
    eps_a      : Tolansky excess fraction for the anchor wavelength (line a).
                 Must satisfy 0 <= eps_a < 1.
    lam_a      : Anchor wavelength in metres. Use NE_WAVELENGTH_1_AIR_M for
                 neon calibration synthesis.

    Returns
    -------
    t_eff : float, phase-corrected gap in metres.
            Satisfies: abs(t_eff - t_tolansky) < lam_a / 4

    Example
    -------
    >>> t_eff = phase_correct_gap(20.1070707e-3, 0.23286, 640.2248e-9)
    >>> print(f"{(2*t_eff/640.2248e-9) % 1:.5f}")   # should print 0.23286
    """
    if not (0.0 <= eps_a < 1.0):
        raise ValueError(f"eps_a={eps_a} must be in [0, 1)")

    eps_current = (2.0 * t_tolansky / lam_a) % 1.0
    delta_eps = eps_a - eps_current

    # Wrap into (-0.5, +0.5] — take nearest FSR, not always the positive one
    if delta_eps > 0.5:
        delta_eps -= 1.0
    elif delta_eps <= -0.5:
        delta_eps += 1.0

    delta_t = delta_eps * lam_a / 2.0
    t_eff = t_tolansky + delta_t

    return t_eff


def build_instrument_matrix(
    r: np.ndarray,           # radial bin centres, pixels, shape (R,)
    wavelengths: np.ndarray, # wavelength bin centres, metres, shape (L,)
    t,                       # float gap (m) OR InstrumentParams (duck-typing)
    R_refl: float = None,
    alpha: float = None,
    n: float = None,
    r_max: float = None,
    I0: float = None,
    I1: float = None,
    I2: float = None,
    sigma0: float = None,
    sigma1: float = None,
    sigma2: float = None,
    n_subpixels: int = 1,    # accepted for API compatibility; currently unused
) -> np.ndarray:
    """
    Build the instrument matrix A of shape (R, L).

    Column j of A is airy_modified(r; λⱼ) for wavelength wavelengths[j].
    The forward model is:
        s = A @ y + B    (Harding Eq. 16)
    where y is the source spectrum (counts/m), B is the CCD bias vector.

    Supports two calling conventions:
      build_instrument_matrix(r, wavelengths, t, R_refl, alpha, ...)   # 13-arg
      build_instrument_matrix(r, wavelengths, params, n_subpixels=N)   # InstrumentParams

    Parameters
    ----------
    r           : radial bin centres in pixels, shape (R,)
    wavelengths : wavelength bin centres in metres, shape (L,)
                  Use L=101 for inversion, L=300 for synthesis.
    n_subpixels : accepted for API compatibility; not used in computation.

    Returns
    -------
    A : np.ndarray, shape (R, L)
        All values >= 0. No NaN or Inf for valid inputs.
    """
    if hasattr(t, "R_refl"):  # duck-type: InstrumentParams-like object
        p = t
        t, R_refl, alpha, n = p.t, p.R_refl, p.alpha, p.n
        r_max, I0, I1, I2 = p.r_max, p.I0, p.I1, p.I2
        sigma0, sigma1, sigma2 = p.sigma0, p.sigma1, p.sigma2

    R_bins = len(r)
    L_bins = len(wavelengths)
    A = np.zeros((R_bins, L_bins))
    # Scale each column by Δλ so that A @ y integrates: s = A @ y + B
    # where y is a spectral density (counts/m) and A has units counts·m.
    dlam = float(wavelengths[1] - wavelengths[0]) if L_bins > 1 else 1.0
    for j, lam in enumerate(wavelengths):
        A[:, j] = airy_modified(r, lam, t, R_refl, alpha, n,
                                r_max, I0, I1, I2, sigma0, sigma1, sigma2) * dlam
    return A


def make_wavelength_grid(
    center_wavelength: float,           # metres
    params: "InstrumentParams",
    n_fsr: float = 3.0,                 # FSR spans to cover
    L: int = 101,                       # number of wavelength bins
) -> np.ndarray:
    """
    Construct the wavelength grid for the instrument matrix.

    Spans ±(n_fsr/2) free spectral ranges about center_wavelength.

    Parameters
    ----------
    center_wavelength : metres. Use OI_WAVELENGTH_AIR_M for airglow (M04/M06),
                        NE_WAVELENGTH_1_AIR_M for calibration (M02/M05).
    n_fsr : number of FSRs to span. Default 3.0 covers the full beat
            pattern between the two neon lines.
    L     : number of bins. Use 101 for inversion, 300 for synthesis
            (anti-inverse-crime rule).

    Returns
    -------
    wavelengths : np.ndarray, shape (L,), units metres, monotonically increasing
    """
    fsr = params.free_spectral_range(center_wavelength)
    lam_min = center_wavelength - n_fsr * fsr / 2.0
    lam_max = center_wavelength + n_fsr * fsr / 2.0
    return np.linspace(lam_min, lam_max, L)


# ---------------------------------------------------------------------------
# Source spectrum constructors (Harding Eqs 10–11)
# ---------------------------------------------------------------------------

def make_ne_spectrum(lam_grid: np.ndarray, I_line: float = 1.0) -> np.ndarray:
    """
    Construct the two-line neon source spectrum vector.

    Places the two neon lines (NE_WAVELENGTH_1_AIR_M, NE_WAVELENGTH_2_AIR_M)
    at the nearest wavelength bins in lam_grid, weighted by their
    intensity ratios (NE_INTENSITY_1, NE_INTENSITY_2) from src.constants.

    This implements the delta-function neon source model (Sec 4.2):
      Y_Ne(λ) = I_line · [NE_INTENSITY_1·δ(λ−λ₁) + NE_INTENSITY_2·δ(λ−λ₂)]

    Both wavelengths must fall within lam_grid; raises ValueError otherwise.

    Parameters
    ----------
    lam_grid : (L,) wavelength grid (metres) from make_wavelength_grid()
    I_line   : overall brightness scale (ADU); default 1.0

    Returns
    -------
    y_ne : (L,) source spectrum vector; units match I_line

    Notes
    -----
    NE_WAVELENGTH_1_AIR_M = 640.2248e-9 m  (strong line)
    NE_WAVELENGTH_2_AIR_M = 638.2991e-9 m  (weak line)
    NE_INTENSITY_1 = 1.0
    NE_INTENSITY_2 = 0.8
    """
    lam_min, lam_max = lam_grid[0], lam_grid[-1]
    if not (lam_min <= NE_WAVELENGTH_1_AIR_M <= lam_max):
        raise ValueError(
            f"NE_WAVELENGTH_1_AIR_M={NE_WAVELENGTH_1_AIR_M*1e9:.4f} nm "
            f"outside lam_grid [{lam_min*1e9:.4f}, {lam_max*1e9:.4f}] nm"
        )
    if not (lam_min <= NE_WAVELENGTH_2_AIR_M <= lam_max):
        raise ValueError(
            f"NE_WAVELENGTH_2_AIR_M={NE_WAVELENGTH_2_AIR_M*1e9:.4f} nm "
            f"outside lam_grid [{lam_min*1e9:.4f}, {lam_max*1e9:.4f}] nm"
        )
    y = np.zeros(len(lam_grid))
    idx1 = int(np.argmin(np.abs(lam_grid - NE_WAVELENGTH_1_AIR_M)))
    idx2 = int(np.argmin(np.abs(lam_grid - NE_WAVELENGTH_2_AIR_M)))
    y[idx1] += I_line * NE_INTENSITY_1
    y[idx2] += I_line * NE_INTENSITY_2
    return y


def make_airglow_spectrum(
    lam_grid: np.ndarray,
    v_rel: float,
    Y_line: float = 1.0,
    Y_bg: float = 0.0,
) -> np.ndarray:
    """
    Construct the OI 630.0 nm airglow source spectrum vector.

    Implements the delta-function Doppler-shifted source model (Sec 4.3):
      Y_OI(λ) = Y_bg + Y_line · δ(λ − λ_c)
      λ_c = OI_WAVELENGTH_AIR_M · (1 + v_rel / SPEED_OF_LIGHT_MS)

    Temperature broadening (Harding Eq. 12) is explicitly NOT applied.
    WindCube uses the delta-function approximation throughout; temperature
    is not a science product.

    Parameters
    ----------
    lam_grid : (L,) wavelength grid (metres) from make_wavelength_grid()
    v_rel    : line-of-sight velocity (m/s); positive = recession
               Valid range: −7700 m/s to +1000 m/s
    Y_line   : line intensity (ADU); default 1.0
    Y_bg     : spectrally flat background per wavelength bin (ADU); default 0.0

    Returns
    -------
    y_oi : (L,) source spectrum vector

    Raises
    ------
    ValueError : if v_rel is outside [−7700, +1000] m/s
    ValueError : if λ_c falls outside lam_grid

    Notes
    -----
    Velocity sign convention:
      Positive v_rel → recession → λ_c > λ₀ → fringes shift inward (smaller r)
      Negative v_rel → approach  → λ_c < λ₀ → fringes shift outward (larger r)

    OI_WAVELENGTH_AIR_M = 630.0304e-9 m (imported from src.constants)
    SPEED_OF_LIGHT_MS   = 299_792_458 m/s (imported from src.constants)
    """
    if not (-7700 <= v_rel <= 1000):
        raise ValueError(
            f"v_rel={v_rel} m/s outside valid range [-7700, +1000] m/s"
        )
    lam_c = OI_WAVELENGTH_AIR_M * (1.0 + v_rel / SPEED_OF_LIGHT_MS)
    lam_min, lam_max = lam_grid[0], lam_grid[-1]
    if not (lam_min <= lam_c <= lam_max):
        raise ValueError(
            f"Doppler-shifted λ_c={lam_c*1e9:.6f} nm outside "
            f"lam_grid [{lam_min*1e9:.6f}, {lam_max*1e9:.6f}] nm"
        )
    y = np.full(len(lam_grid), Y_bg)
    idx_c = int(np.argmin(np.abs(lam_grid - lam_c)))
    y[idx_c] += Y_line
    return y


# ===========================================================================
# Section C — Annular reduction
# (copied from src/processing/annular_reduction.py)
# ===========================================================================

# ---------------------------------------------------------------------------
# Peak dataclass and Gaussian helper  (defined first; used inside annular_reduce)
# ---------------------------------------------------------------------------

@dataclass
class PeakFit:
    """Result of a single-peak Gaussian fit to the radial profile."""
    peak_idx:       int    # bin index of the find_peaks detection
    r_raw_px:       float  # r_grid value at the detected bin (px)
    profile_raw:    float  # profile value at the detected bin (ADU)
    r_fit_px:       float  # Gaussian centroid from curve_fit (px)
    sigma_r_fit_px: float  # 1-sigma uncertainty on centroid (px); nan if fit failed
    amplitude_adu:  float  # Gaussian amplitude above background (ADU)
    width_px:       float  # Gaussian sigma width (px); nan if fit failed
    fit_ok:         bool   # False if curve_fit failed or window too small
    reduced_chi2:   float  # chi² / (n_points - 4); nan if fit failed


@dataclass
class PeakFitR2:
    """Result of a single-peak Gaussian fit to the radial profile in r² domain."""
    peak_idx:          int    # bin index of the find_peaks detection
    r2_raw_px2:        float  # r2_grid value at the detected bin (px²)
    r_raw_px:          float  # r_grid value at the detected bin (px)
    profile_raw:       float  # profile value at the detected bin (ADU)
    r2_fit_px2:        float  # Gaussian centroid in r² from curve_fit (px²)
    sigma_r2_fit_px2:  float  # 1-sigma uncertainty on centroid (px²); nan if fit failed
    amplitude_adu:     float  # Gaussian amplitude above background (ADU)
    width_r2_px2:      float  # Gaussian sigma width in r² (px²); nan if fit failed
    fit_ok:            bool   # False if curve_fit failed or window too small
    reduced_chi2:      float  # chi² / (n_points - 4); nan if fit failed
    para_ok:           bool   # True if parabolic centroid succeeded
    para_rms:          float  # RMS of parabolic fit residuals (px²); nan if failed
    gauss_ok:          bool   # True if Gaussian fallback succeeded


def _gaussian(r: np.ndarray, A: float, mu: float, sig: float, B: float) -> np.ndarray:
    """Gaussian with flat background: A*exp(-0.5*((r-mu)/sig)^2) + B."""
    return A * np.exp(-0.5 * ((r - mu) / sig) ** 2) + B


_R2_WINDOWS: dict[int, tuple] = {  # 2-tuple (lo,hi) or 3-tuple (lo,centre,hi)
     0: (    187,    387),   # P0  640.2 nm  ctr≈287
     1: (   700,   1100),   # P1  638.3 nm  ctr≈900
     2: (  1250,   1775),   # P2  640.2 nm  ctr≈1518
     3: (  1925,   2325),   # P3  638.3 nm  ctr≈2122
     4: (  2475,   3000),   # P4  640.2 nm  ctr≈2750
     5: (  3150,   3550),   # P5  638.3 nm  ctr≈3345
     6: (  3700,   4225),   # P6  640.2 nm  ctr≈3980
     7: (  4375,   4800),   # P7  638.3 nm  ctr≈4579
     8: (  4950,   5450),   # P8  640.2 nm  ctr≈5209
     9: (  5600,   6025),   # P9  638.3 nm  ctr≈5802
    10: (  6175,   6675),   # P10 640.2 nm  ctr≈6441
    11: (  6825,   7250),   # P11 638.3 nm  ctr≈7021
    12: (  7400,   7900),   # P12 640.2 nm  ctr≈7672
    13: (  8147,   8347),   # P13 638.3 nm  ctr≈8247
    14: (  8625,   9125),   # P14 640.2 nm  ctr≈8902
    15: (  9372,   9572),   # P15 638.3 nm  ctr≈9472
    16: (  9850,  10375),   # P16 640.2 nm  ctr≈10132
    17: ( 10600,  10800),   # P17 638.3 nm  ctr≈10698
    18: ( 11075,  11600),   # P18 640.2 nm  ctr≈11363
    19: ( 11850,  12000),   # P19 638.3 nm  ctr≈11919
}


def _find_valley_bounds(
    profile: np.ndarray,
    peak_idx: int,
    search_hw: int = 120,
    smooth_sigma: float = 3.0,
) -> tuple[int, int]:
    """
    Find the valley (inter-peak minimum) on each side of peak_idx.

    Applies a light Gaussian smooth (sigma=smooth_sigma bins) to suppress
    noise before searching for the minimum. The smooth prevents premature
    stopping at noise dips while preserving the true inter-peak valley.
    Uses np.argmin on the smoothed profile within [peak_idx-search_hw,
    peak_idx] (left) and [peak_idx, peak_idx+search_hw] (right).

    Returns (valley_L_idx, valley_R_idx) as absolute bin indices into
    the original (unsmoothed) profile array.
    """
    n        = len(profile)
    smoothed = gaussian_filter1d(profile.astype(float), sigma=smooth_sigma)

    lo_L = max(0, peak_idx - search_hw)
    hi_R = min(n - 1, peak_idx + search_hw)

    if peak_idx > lo_L:
        valley_L = lo_L + int(np.argmin(smoothed[lo_L:peak_idx]))
    else:
        valley_L = lo_L

    if hi_R > peak_idx:
        valley_R = peak_idx + 1 + int(np.argmin(smoothed[peak_idx + 1:hi_R + 1]))
    else:
        valley_R = hi_R

    return valley_L, valley_R


def _measure_hwhm(
    profile:      np.ndarray,
    peak_idx:     int,
    valley_L:     int,
    valley_R:     int,
    smooth_sigma: float = 3.0,
) -> tuple[int, int]:
    """
    Measure the half-width at half-maximum on each flank of a peak.

    Uses the smoothed profile. Half-maximum is defined relative to the
    local valley floor on each side independently:
      half_level_L = profile[valley_L] + 0.5*(profile[peak_idx] - profile[valley_L])
      half_level_R = profile[valley_R] + 0.5*(profile[peak_idx] - profile[valley_R])

    Walks inward from each valley toward the peak and finds the first
    bin that crosses the half-level. Returns (hwhm_L_bins, hwhm_R_bins)
    as bin counts (not absolute indices).
    Falls back to (peak_idx - valley_L, valley_R - peak_idx) if no
    crossing is found.
    """
    smoothed = gaussian_filter1d(profile.astype(float), sigma=smooth_sigma)
    peak_val = smoothed[peak_idx]

    # Left HWHM
    floor_L = smoothed[valley_L]
    half_L  = floor_L + 0.5 * (peak_val - floor_L)
    hwhm_L  = peak_idx - valley_L   # fallback
    for i in range(valley_L, peak_idx):
        if smoothed[i] >= half_L:
            hwhm_L = peak_idx - i
            break

    # Right HWHM
    floor_R = smoothed[valley_R]
    half_R  = floor_R + 0.5 * (peak_val - floor_R)
    hwhm_R  = valley_R - peak_idx   # fallback
    for i in range(valley_R, peak_idx, -1):
        if smoothed[i] >= half_R:
            hwhm_R = i - peak_idx
            break

    return max(4, hwhm_L), max(4, hwhm_R)


def _parabolic_centroid(
    x: np.ndarray,
    y: np.ndarray,
    top_fraction: float = 0.5,
) -> tuple[float, float]:
    """
    Fit a parabola to the top fraction of a peak and return the centroid.

    Selects bins where y >= y_min + top_fraction*(y_max - y_min), fits
    a quadratic y = a*x^2 + b*x + c, and returns centroid = -b/(2a).

    Returns (centroid, residual_rms). Returns (x[argmax], nan) if fewer
    than 3 points are available or if parabola opens upward (a >= 0).
    """
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    threshold = y_min + top_fraction * (y_max - y_min)
    mask = y >= threshold
    xm, ym = x[mask], y[mask]
    if len(xm) < 3:
        return float(x[int(np.argmax(y))]), float("nan")
    coeffs = np.polyfit(xm, ym, 2)
    a, b, _ = coeffs
    if a >= 0:
        return float(x[int(np.argmax(y))]), float("nan")
    centroid = -b / (2.0 * a)
    residuals = ym - np.polyval(coeffs, xm)
    rms = float(np.sqrt(np.mean(residuals ** 2)))
    return centroid, rms


def _find_and_fit_peaks(
    r_grid:          np.ndarray,
    profile:         np.ndarray,
    sigma_profile:   np.ndarray,
    masked:          np.ndarray,
    distance:        int   = 5,
    prominence:      float = 100.0,
    fit_half_window: int   = 6,
    min_sep_px:      float = 3.0,
) -> list[PeakFit]:
    """
    Locate peaks in the radial profile and fit a Gaussian to each one.

    Called at the end of annular_reduce so that the profile and SEM arrays
    are used directly without copying.  Only unmasked bins enter find_peaks.
    SEM values are passed as absolute_sigma weights to curve_fit; bins with
    infinite SEM are excluded from each fit window.

    Parameters
    ----------
    r_grid           : bin-centre radii (px), shape (n_bins,)
    profile          : mean intensity per bin (ADU), shape (n_bins,)
    sigma_profile    : SEM per bin (ADU); np.inf for masked/sparse bins
    masked           : bool mask, True = bin excluded
    distance         : minimum peak separation in *good* bins.
                       WARNING: find_peaks counts only good (unmasked) bins,
                       not original bin indices.  If masked bins lie between
                       two true peaks the good-bin separation is smaller than
                       the original separation, and a peak can be suppressed.
                       The safe_distance computed from min_sep_px (below)
                       overrides this value whenever it would be tighter.
    prominence       : minimum prominence (ADU)
    fit_half_window  : maximum half-width of the Gaussian fitting window (bins).
                       The actual window used for each peak is clamped to
                       floor((nearest_neighbour_separation - 1) / 2) so the
                       window never reaches an adjacent peak.  This parameter
                       acts as an upper bound; the adaptive clamp controls the
                       effective window in densely-packed profiles.
    min_sep_px       : minimum physical peak separation (px) used to derive a
                       safe lower bound on the good-bin distance parameter.
                       Protects against the good-bin compression effect.

    Returns
    -------
    List of PeakFit sorted by r_raw_px.
    """
    good         = ~masked
    good_indices = np.where(good)[0]
    profile_good = profile[good]

    if profile_good.size == 0:
        return []

    # Derive a physics-grounded distance floor from the actual good-bin spacing.
    # Good-bin spacing can vary across the profile (r^2 binning gives denser bins
    # at small r), so use the median spacing as a robust representative value.
    # We take the floor so we never accidentally merge two real peaks that are
    # physically closer than min_sep_px.
    if good_indices.size > 1:
        median_dr_px = float(np.median(np.diff(r_grid[good])))
        if median_dr_px > 0.0:
            safe_distance = max(1, int(np.floor(min_sep_px / median_dr_px)))
        else:
            safe_distance = distance
            median_dr_px  = 1.0
    else:
        safe_distance = distance
        median_dr_px  = 1.0
    # Use the tighter (smaller) of the caller-supplied distance and the
    # physics-derived floor so that neither can suppress real peaks.
    effective_distance = min(distance, safe_distance)

    peaks_sub, _ = find_peaks(profile_good, distance=effective_distance, prominence=prominence)

    # Build the full list of detected bin indices now so each peak can
    # look up its nearest neighbours when sizing its fitting window.
    all_bin_indices = [int(good_indices[s]) for s in peaks_sub]

    results: list[PeakFit] = []
    for peak_pos, (sub_idx, bin_idx) in enumerate(zip(peaks_sub, all_bin_indices)):

        # Adaptive fitting window: clamp half-width so the window never reaches
        # an adjacent detected peak.  With peaks ~7-8 bins apart, the unclamped
        # default (fit_half_window=8) would always engulf the neighbour, causing
        # curve_fit to lock onto the larger flanking peak instead of the target.
        #
        # Rule: leave at least 1 bin gap to the nearest neighbour.
        #   max_hw = floor((nearest_neighbour_separation - 1) / 2)
        # Minimum of 2 bins on each side ensures at least 5 points in the window.
        # HWHM-based asymmetric window
        valley_L, valley_R = _find_valley_bounds(profile, bin_idx, search_hw=fit_half_window)
        hwhm_L, hwhm_R    = _measure_hwhm(profile, bin_idx, valley_L, valley_R)
        hw_L = min(max(6, int(np.round(0.8 * hwhm_L))), fit_half_window)
        hw_R = min(max(6, int(np.round(0.8 * hwhm_R))), fit_half_window)

        lo      = max(0, bin_idx - hw_L)
        hi      = min(len(r_grid) - 1, bin_idx + hw_R)
        win     = np.arange(lo, hi + 1)
        usable  = ~masked[win] & np.isfinite(sigma_profile[win])
        win_use = win[usable]

        r_fit_px       = float(r_grid[bin_idx])   # fallback if fit fails
        sigma_r_fit_px = np.nan
        amplitude_adu  = float(profile[bin_idx])
        width_px       = np.nan
        reduced_chi2   = np.nan
        fit_ok         = False

        if win_use.size >= 4:
            r_w   = r_grid[win_use]
            p_w   = profile[win_use]
            sem_w = sigma_profile[win_use]

            # Robust background: 20th-percentile of the window rather than the
            # minimum.  The minimum is always the trough between two peaks, so
            # it under-estimates the local background and sets A0 too high,
            # giving curve_fit a poor starting point for narrow peaks.
            B0   = float(np.percentile(p_w, 20))
            A0   = max(float(profile[bin_idx]) - B0, 1.0)
            mu0  = float(r_grid[bin_idx])
            # sig0: use 1/6 of the window span as a starting estimate.
            # span/4 (old formula) is 8x the true width for narrow small peaks
            # and causes the fitter to search in completely the wrong region.
            sig0 = max((float(r_w[-1]) - float(r_w[0])) / 6.0, median_dr_px * 0.5)
            p0   = [A0, mu0, sig0, B0]
            bounds = (
                [0.0,    float(r_w[0]),  0.3 * median_dr_px,   0.0   ],
                [np.inf, float(r_w[-1]), float(r_w[-1]) - float(r_w[0]), np.inf],
            )
            try:
                popt, pcov = curve_fit(
                    _gaussian, r_w, p_w,
                    p0=p0, sigma=sem_w, absolute_sigma=True,
                    bounds=bounds, maxfev=5000,
                )
                perr           = np.sqrt(np.diag(pcov))
                r_fit_px       = float(popt[1])
                sigma_r_fit_px = float(perr[1])
                amplitude_adu  = float(popt[0])
                width_px       = float(abs(popt[2]))
                n_dof          = len(r_w) - 4
                if n_dof > 0:
                    chi2         = float(np.sum(((p_w - _gaussian(r_w, *popt)) / sem_w) ** 2))
                    reduced_chi2 = chi2 / n_dof
                fit_ok         = True
                # Parabolic centroid refinement
                para_mu, _ = _parabolic_centroid(r_w, p_w, top_fraction=0.4)
                if (r_w[0] <= para_mu <= r_w[-1] and
                        abs(para_mu - r_fit_px) < 2.0 * width_px):
                    r_fit_px = para_mu
            except (RuntimeError, ValueError):
                pass

        results.append(PeakFit(
            peak_idx       = bin_idx,
            r_raw_px       = float(r_grid[bin_idx]),
            profile_raw    = float(profile[bin_idx]),
            r_fit_px       = r_fit_px,
            sigma_r_fit_px = sigma_r_fit_px,
            amplitude_adu  = amplitude_adu,
            width_px       = width_px,
            fit_ok         = fit_ok,
            reduced_chi2   = reduced_chi2,
        ))

    results.sort(key=lambda p: p.r_raw_px)
    return results


def _find_and_fit_peaks_r2(
    profile:        np.ndarray,
    r_grid:         np.ndarray,
    r2_grid:        np.ndarray,
    sigma_profile:  np.ndarray,
    masked:         np.ndarray,
    distance:       int   = 5,
    prominence:     float = 50.0,
    fit_half_window: int  = 80,
    min_sep_px:     float = 5.0,
) -> list["PeakFitR2"]:
    """
    Fit every window defined in _R2_WINDOWS directly, using the 3-tuple
    centre hint (or window midpoint) as the starting position.

    The find_peaks auto-detection step has been removed entirely.  Every
    entry in _R2_WINDOWS is attempted regardless of local prominence,
    giving deterministic, user-controlled peak positions.
    """
    if profile.size == 0:
        return []

    # r² bin spacing — used for Gaussian fit bounds
    good = ~masked
    if good.sum() > 1:
        median_dr2_px2 = float(np.median(np.diff(r2_grid[good])))
        if median_dr2_px2 <= 0.0:
            median_dr2_px2 = 1.0
    else:
        median_dr2_px2 = 1.0

    results: list[PeakFitR2] = []

    for win_key in sorted(_R2_WINDOWS.keys()):
        _wk     = _R2_WINDOWS[win_key]
        r2_lo   = _wk[0];  r2_hi = _wk[-1]
        # Centre hint: middle of 3-tuple, or midpoint of 2-tuple
        r2_ctr  = _wk[1] if len(_wk) == 3 else 0.5 * (r2_lo + r2_hi)

        # Map r² bounds and centre to bin indices
        lo      = int(np.searchsorted(r2_grid, r2_lo, side='left'))
        hi      = int(np.searchsorted(r2_grid, r2_hi, side='right')) - 1
        lo      = max(0, lo)
        hi      = min(len(r2_grid) - 1, hi)
        bin_idx = int(np.argmin(np.abs(r2_grid - r2_ctr)))
        bin_idx = max(lo, min(hi, bin_idx))

        win     = np.arange(lo, hi + 1)
        usable  = ~masked[win] & np.isfinite(sigma_profile[win])
        win_use = win[usable]

        # ── Parabolic primary fit ─────────────────────────────────────────────
        para_ok    = False
        para_mu    = float(r2_grid[bin_idx])
        para_rms   = np.nan
        para_resid = None

        if win_use.size >= 4:
            r2_w  = r2_grid[win_use]
            p_w   = profile[win_use]
            sem_w = sigma_profile[win_use]

            para_mu_raw, para_rms = _parabolic_centroid(r2_w, p_w, top_fraction=0.5)
            if r2_w[0] <= para_mu_raw <= r2_w[-1]:
                para_mu = para_mu_raw
                para_ok = True
                threshold = np.min(p_w) + 0.5 * (np.max(p_w) - np.min(p_w))
                mask_top  = p_w >= threshold
                if mask_top.sum() >= 3:
                    coeffs     = np.polyfit(r2_w[mask_top], p_w[mask_top], 2)
                    para_resid = p_w[mask_top] - np.polyval(coeffs, r2_w[mask_top])

        # ── Gaussian fallback ─────────────────────────────────────────────────
        gauss_ok         = False
        r2_fit_px2       = para_mu
        sigma_r2_fit_px2 = np.nan
        amplitude_adu    = float(profile[bin_idx])
        width_r2_px2     = np.nan
        reduced_chi2     = np.nan

        if win_use.size >= 4:
            r2_w  = r2_grid[win_use]
            p_w   = profile[win_use]
            sem_w = sigma_profile[win_use]

            B0   = float(np.percentile(p_w, 20))
            A0   = max(float(profile[bin_idx]) - B0, 1.0)
            mu0  = para_mu if para_ok else float(r2_grid[bin_idx])
            sig0 = max((float(r2_w[-1]) - float(r2_w[0])) / 6.0,
                       median_dr2_px2 * 0.5)
            p0     = [A0, mu0, sig0, B0]
            bounds = (
                [0.0,    float(r2_w[0]),  0.3 * median_dr2_px2,    -np.inf],
                [np.inf, float(r2_w[-1]), float(r2_w[-1]) - float(r2_w[0]), np.inf],
            )
            try:
                popt, pcov       = curve_fit(
                    _gaussian, r2_w, p_w,
                    p0=p0, sigma=sem_w, absolute_sigma=True,
                    bounds=bounds, maxfev=5000,
                )
                perr             = np.sqrt(np.diag(pcov))
                sigma_r2_fit_px2 = float(perr[1])
                amplitude_adu    = float(popt[0])
                width_r2_px2     = float(abs(popt[2]))
                n_dof            = len(r2_w) - 4
                if n_dof > 0:
                    chi2         = float(np.sum(
                        ((p_w - _gaussian(r2_w, *popt)) / sem_w) ** 2))
                    reduced_chi2 = chi2 / n_dof
                gauss_ok         = True
                if not para_ok:
                    r2_fit_px2   = float(popt[1])
            except (RuntimeError, ValueError):
                pass

        fit_ok = para_ok or gauss_ok

        results.append(PeakFitR2(
            peak_idx         = bin_idx,
            r2_raw_px2       = float(r2_grid[bin_idx]),
            r_raw_px         = float(r_grid[bin_idx]),
            profile_raw      = float(profile[bin_idx]),
            r2_fit_px2       = r2_fit_px2,
            sigma_r2_fit_px2 = sigma_r2_fit_px2,
            amplitude_adu    = amplitude_adu,
            width_r2_px2     = width_r2_px2,
            fit_ok           = fit_ok,
            reduced_chi2     = reduced_chi2,
            para_ok          = para_ok,
            para_rms         = para_rms,
            gauss_ok         = gauss_ok,
        ))

    results.sort(key=lambda p: p.r2_raw_px2)
    return results


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class FringeProfile:
    """
    1-D radial fringe profile produced by annular reduction.
    Input to M05 (calibration inversion) and M06 (airglow inversion).
    """
    # Profile arrays — shape (n_bins,)
    profile:           np.ndarray   # mean intensity per r^2 bin, ADU
    sigma_profile:     np.ndarray   # SEM per bin, ADU (np.inf for masked bins)
    two_sigma_profile: np.ndarray   # exactly 2 x sigma_profile
    r_grid:            np.ndarray   # bin centre radii, pixels
    r2_grid:           np.ndarray   # bin centre r^2 values, pixels^2
    n_pixels:          np.ndarray   # actual CCD pixel count per bin (int)
    masked:            np.ndarray   # bool, True = bin excluded from fitting

    # Centre (passed in from center_finder)
    cx:        float
    cy:        float
    sigma_cx:  float
    sigma_cy:  float

    # Reduction parameters
    r_min_px:    float
    r_max_px:    float
    n_bins:      int
    n_subpixels: int
    sigma_clip:  float
    image_shape: tuple

    # Quality flag
    sparse_bins: bool   # True if > 10 % of bins have fewer than min_pixels_per_bin

    # Peaks detected in the radial profile (populated by annular_reduce)
    peak_fits:    list[PeakFit]   = field(default_factory=list)
    peak_fits_r2: list[PeakFitR2] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Annular reduction (Mulligan 1986 sub-pixel technique, r^2 bins)
# ---------------------------------------------------------------------------

def annular_reduce(
    image: np.ndarray,
    cx: float,
    cy: float,
    sigma_cx: float,
    sigma_cy: float,
    r_min_px: float = 0.0,
    r_max_px: float = 110.0,
    n_bins: int = 1500,
    n_subpixels: int = 1,
    sigma_clip_threshold: float = 3.0,
    min_pixels_per_bin: int = 3,
    bad_pixel_mask: Optional[np.ndarray] = None,
    peak_distance: int = 50,
    peak_prominence: float = 50.0,
    peak_fit_half_window: int = 40,  # search radius for valley-bounded asymmetric window (bins)
    min_peak_sep_px: float = 3.0,
) -> FringeProfile:
    """
    Reduce a 2-D CCD image to a 1-D r^2-binned radial intensity profile.

    Peak finding and Gaussian fitting are performed on the profile and SEM
    immediately after binning and stored in FringeProfile.peak_fits.

    Parameters
    ----------
    image               : 2-D ndarray (uint16 or float)
    cx, cy              : fringe centre in pixel coordinates (from center_finder)
    sigma_cx, sigma_cy  : 1-sigma centre uncertainties in pixels (from center_finder)
    r_min_px            : inner exclusion radius (pixels)
    r_max_px            : outer radius (pixels)
    n_bins              : number of r^2 bins
    n_subpixels         : sub-pixel grid size per axis (must be 1 to match M05/M06)
    sigma_clip_threshold: outlier rejection threshold (sigma)
    min_pixels_per_bin  : bins with fewer pixels are masked
    bad_pixel_mask      : optional bool array, True = bad pixel to exclude
    peak_distance       : minimum peak separation passed to find_peaks, measured
                          in *good* (unmasked) bins — NOT in original bin indices
                          and NOT in pixels.  Masked bins between two true peaks
                          compress the good-bin separation below this value and
                          can cause real peaks to be suppressed.  The safe lower
                          bound derived from min_peak_sep_px (below) prevents
                          this.  Default 5 is safe for 150 bins over 110 px
                          with ~0.7 px/bin spacing and peaks ~7-8 bins apart.
    peak_prominence     : minimum prominence passed to find_peaks (ADU).
                          Prominence is measured peak-to-trough (local), NOT
                          relative to zero.  A high background pedestal does
                          not affect this value.
    peak_fit_half_window: search radius for valley-bounded asymmetric window (bins)
    min_peak_sep_px     : minimum physical separation between peaks in pixels,
                          used to derive a safe lower bound on peak_distance
                          from the actual good-bin spacing.  Prevents the
                          good-bin compression effect from suppressing real peaks
                          when sparse/masked bins lie between two adjacent peaks.
    """
    H, W   = image.shape
    r2_max = r_max_px ** 2
    r2_min = r_min_px ** 2
    dr2    = r2_max / n_bins

    valid = np.ones((H, W), dtype=bool)
    if bad_pixel_mask is not None:
        valid &= ~bad_pixel_mask

    row_c, col_c = np.mgrid[0:H, 0:W]
    rows_v = row_c[valid].astype(np.float64)
    cols_v = col_c[valid].astype(np.float64)
    adus_v = image[valid].astype(np.float64)
    N_v    = len(rows_v)

    r2_edges = np.linspace(0.0, r2_max, n_bins + 1)
    r2_grid  = 0.5 * (r2_edges[:-1] + r2_edges[1:])
    r_grid   = np.sqrt(r2_grid)

    if N_v == 0:
        return FringeProfile(
            profile=np.zeros(n_bins), sigma_profile=np.full(n_bins, np.inf),
            two_sigma_profile=np.full(n_bins, np.inf),
            r_grid=r_grid, r2_grid=r2_grid,
            n_pixels=np.zeros(n_bins, dtype=int),
            masked=np.ones(n_bins, dtype=bool),
            cx=cx, cy=cy, sigma_cx=sigma_cx, sigma_cy=sigma_cy,
            r_min_px=r_min_px, r_max_px=r_max_px,
            n_bins=n_bins, n_subpixels=n_subpixels,
            sigma_clip=sigma_clip_threshold, image_shape=(H, W),
            sparse_bins=True, peak_fits=[], peak_fits_r2=[],
        )

    # Sub-pixel offsets — shape (N_sub^2,)
    k  = np.arange(n_subpixels)
    o  = (k + 0.5) / n_subpixels - 0.5
    dc_2d, dr_2d = np.meshgrid(o, o)
    dr_flat = dr_2d.ravel().astype(np.float64)
    dc_flat = dc_2d.ravel().astype(np.float64)
    N_sub2  = n_subpixels ** 2

    # r^2 for every (pixel, sub-pixel) pair — shape (N_v, N_sub^2)
    r2_all = (
        (rows_v[:, None] + dr_flat[None, :] - cy) ** 2 +
        (cols_v[:, None] + dc_flat[None, :] - cx) ** 2
    )

    in_ann               = (r2_all >= r2_min) & (r2_all < r2_max)
    bin_idx_all          = np.floor(r2_all / dr2).astype(np.int32)
    bin_idx_all          = np.clip(bin_idx_all, 0, n_bins - 1)
    bin_idx_all[~in_ann] = n_bins  # sentinel for out-of-annulus

    pix_idx_2d = (np.arange(N_v, dtype=np.int64)[:, None]
                  * np.ones(N_sub2, dtype=np.int64)[None, :])

    in_ann_flat = in_ann.ravel()
    pix_flat    = pix_idx_2d.ravel()[in_ann_flat]
    bin_flat    = bin_idx_all.ravel()[in_ann_flat].astype(np.int64)

    # Deduplicate (pixel, bin) pairs so each pixel contributes once per bin
    pair_ids        = pix_flat * n_bins + bin_flat
    unique_pair_ids = np.unique(pair_ids)
    unique_pix_idx  = (unique_pair_ids // n_bins).astype(np.int64)
    unique_bin_idx  = (unique_pair_ids %  n_bins).astype(np.int64)
    unique_adus     = adus_v[unique_pix_idx]

    sort_order  = np.argsort(unique_bin_idx, kind="stable")
    sorted_bins = unique_bin_idx[sort_order]
    sorted_adus = unique_adus[sort_order]

    bin_starts = np.searchsorted(sorted_bins, np.arange(n_bins, dtype=np.int64))
    bin_ends   = np.searchsorted(sorted_bins, np.arange(n_bins, dtype=np.int64),
                                 side="right")

    out_profile = np.zeros(n_bins)
    out_sigma   = np.full(n_bins, np.inf)
    out_npix    = np.zeros(n_bins, dtype=int)
    out_masked  = np.zeros(n_bins, dtype=bool)

    for b in range(n_bins):
        s, e = int(bin_starts[b]), int(bin_ends[b])
        if e <= s:
            out_masked[b] = True
            continue

        bin_adus = sorted_adus[s:e].copy()

        if len(bin_adus) >= 2:
            mean_v = np.mean(bin_adus)
            std_v  = np.std(bin_adus, ddof=1)
            if std_v > 0.0:
                keep = np.abs(bin_adus - mean_v) <= sigma_clip_threshold * std_v
                if keep.sum() >= min_pixels_per_bin:
                    bin_adus = bin_adus[keep]

        N_pix = len(bin_adus)
        out_npix[b] = N_pix

        if N_pix < min_pixels_per_bin:
            out_masked[b] = True
            out_profile[b] = np.mean(bin_adus) if N_pix > 0 else 0.0
            continue

        mean_v         = np.mean(bin_adus)
        std_v          = np.std(bin_adus, ddof=1) if N_pix > 1 else 0.0
        out_profile[b] = mean_v
        out_sigma[b]   = std_v / np.sqrt(N_pix)   # SEM uses actual pixel count

    sparse_bins = bool(out_masked.sum() > 0.1 * n_bins)

    # -- Peak finding on the freshly computed profile and SEM -----------------
    peaks = _find_and_fit_peaks(
        r_grid           = r_grid,
        profile          = out_profile,
        sigma_profile    = out_sigma,
        masked           = out_masked,
        distance         = peak_distance,
        prominence       = peak_prominence,
        fit_half_window  = peak_fit_half_window,
        min_sep_px       = min_peak_sep_px,
    )
    peaks_r2 = _find_and_fit_peaks_r2(
        r_grid           = r_grid,
        r2_grid          = r2_grid,
        profile          = out_profile,
        sigma_profile    = out_sigma,
        masked           = out_masked,
        distance         = peak_distance,
        prominence       = peak_prominence,
        fit_half_window  = peak_fit_half_window,
        min_sep_px       = min_peak_sep_px,
    )

    return FringeProfile(
        profile           = out_profile,
        sigma_profile     = out_sigma,
        two_sigma_profile = 2.0 * out_sigma,
        r_grid            = r_grid,
        r2_grid           = r2_grid,
        n_pixels          = out_npix,
        masked            = out_masked,
        cx                = cx,
        cy                = cy,
        sigma_cx          = sigma_cx,
        sigma_cy          = sigma_cy,
        r_min_px          = r_min_px,
        r_max_px          = r_max_px,
        n_bins            = n_bins,
        n_subpixels       = n_subpixels,
        sigma_clip        = sigma_clip_threshold,
        image_shape       = (H, W),
        sparse_bins       = sparse_bins,
        peak_fits         = peaks,
        peak_fits_r2      = peaks_r2,
    )


# ---------------------------------------------------------------------------
# Private plotting helpers
# ---------------------------------------------------------------------------

def _plot_first_fringe_diagnostic_r2(
    fp: FringeProfile,
    fit_half_window: int = 40,
) -> None:
    """
    Diagnostic figure for the Gaussian fit to the first detected fringe peak
    in the r² domain.  Mirrors _plot_first_fringe_diagnostic but with r² on
    the x-axis, matching the fitting performed by _find_and_fit_peaks_r2.

    Fringes are expected to be evenly spaced in r², so a Gaussian fit in r²
    gives a more physically motivated centroid for calibration comparison.
    """
    if not fp.peak_fits_r2:
        print("No r²-domain peaks detected — skipping first-fringe r² diagnostic.")
        return

    pf = fp.peak_fits_r2[0]   # first (innermost) peak

    # --- Reconstruct median_dr2_px2 (same logic as _find_and_fit_peaks_r2) ---
    good         = ~fp.masked
    good_indices = np.where(good)[0]
    if good_indices.size > 1:
        median_dr_px = float(np.median(np.diff(fp.r_grid[good])))
        if median_dr_px <= 0.0:
            median_dr_px = 1.0
        median_dr2_px2 = float(np.median(np.diff(fp.r2_grid[good])))
        if median_dr2_px2 <= 0.0:
            median_dr2_px2 = 1.0
    else:
        median_dr_px   = 1.0
        median_dr2_px2 = 1.0

    # --- Reconstruct adaptive effective_hw for the first peak -----------------
    right_sep   = (fp.peak_fits_r2[1].peak_idx - pf.peak_idx) if len(fp.peak_fits_r2) > 1 else 9999
    nearest     = right_sep
    adaptive_hw = max(2, (nearest - 1) // 2)
    effective_hw = min(fit_half_window, adaptive_hw)

    bin_idx = pf.peak_idx
    lo      = max(0, bin_idx - effective_hw)
    hi      = min(len(fp.r2_grid) - 1, bin_idx + effective_hw)
    win     = np.arange(lo, hi + 1)
    usable  = ~fp.masked[win] & np.isfinite(fp.sigma_profile[win])
    win_use = win[usable]

    r2_w  = fp.r2_grid[win_use]
    p_w   = fp.profile[win_use]
    sem_w = fp.sigma_profile[win_use]

    # --- Reconstruct p0 and bounds (identical to _find_and_fit_peaks_r2) -----
    B0   = float(np.percentile(p_w, 20)) if len(p_w) > 0 else 0.0
    A0   = max(float(fp.profile[bin_idx]) - B0, 1.0)
    mu0  = float(fp.r2_grid[bin_idx])
    sig0 = max((float(r2_w[-1]) - float(r2_w[0])) / 6.0, median_dr2_px2 * 0.5) if len(r2_w) > 1 else median_dr2_px2
    p0   = [A0, mu0, sig0, B0]

    bounds_lo = [0.0,    float(r2_w[0]),  0.3 * median_dr2_px2,                     0.0   ]
    bounds_hi = [np.inf, float(r2_w[-1]), float(r2_w[-1]) - float(r2_w[0]), np.inf]

    # --- Re-run curve_fit to get full diagnostics ----------------------------
    fit_ok   = False
    popt     = list(p0)
    perr     = [np.nan] * 4
    pcov     = np.full((4, 4), np.nan)
    mesg     = "fit not attempted (too few usable points)"

    if win_use.size >= 4:
        try:
            popt, pcov = curve_fit(
                _gaussian, r2_w, p_w,
                p0=p0, sigma=sem_w, absolute_sigma=True,
                bounds=(bounds_lo, bounds_hi), maxfev=5000,
            )
            perr   = list(np.sqrt(np.diag(pcov)))
            fit_ok = True
            mesg   = "converged"
        except RuntimeError as exc:
            mesg = f"RuntimeError: {exc}"
        except ValueError as exc:
            mesg = f"ValueError: {exc}"

    if fit_ok and len(r2_w) - 4 > 0:
        _chi2             = float(np.sum(((p_w - _gaussian(r2_w, *popt)) / sem_w) ** 2))
        reduced_chi2_diag = _chi2 / (len(r2_w) - 4)
    else:
        reduced_chi2_diag = np.nan

    # --- Fine grid for plotting curves ---------------------------------------
    if len(r2_w) > 0:
        r2_fine = np.linspace(r2_w[0], r2_w[-1], 500)
    else:
        r2_fine = np.linspace(mu0 - 5, mu0 + 5, 500)
    y_init = _gaussian(r2_fine, *p0)
    y_fit  = _gaussian(r2_fine, *popt) if fit_ok else None

    # --- Derived r from r²_fit -----------------------------------------------
    r2_fit_val    = popt[1] if fit_ok else mu0
    r_fit_derived = float(np.sqrt(r2_fit_val)) if r2_fit_val > 0 else np.nan
    sig_r_derived = float(perr[1] / (2.0 * r_fit_derived)) if (fit_ok and r_fit_derived > 0) else np.nan

    # --- Build annotation text -----------------------------------------------
    ann = "\n".join([
        "ALGORITHM",
        "curve_fit + bounds  →  TRF (Trust Region Reflective)",
        "  fitting in r² domain — x-axis is r² (px²)",
        "",
        "MODEL (r² domain)",
        "  f(r²) = A·exp(-½·((r²-μ)/σ)²) + B",
        "",
        "FITTING WINDOW",
        f"  bins {lo}–{hi}  ({hi - lo + 1} total, {win_use.size} usable)",
        f"  r² = {r2_w[0]:.2f} – {r2_w[-1]:.2f} px²",
        f"  median r² bin width = {median_dr2_px2:.3f} px²",
        f"  adaptive_hw = min({fit_half_window}, ({nearest}-1)//2={adaptive_hw}) = {effective_hw}",
        "",
        "INITIAL GUESS  p0",
        f"  A₀ = {A0:.2f}  (profile[peak] − 20th-pct bkg)",
        f"  μ₀ = {mu0:.2f} px²  (detected bin centre in r²)",
        f"  σ₀ = {sig0:.3f} px²  (window_span_r2/6, ≥0.5·dr2)",
        f"  B₀ = {B0:.2f}  (20th-pct of window)",
        "",
        "BOUNDS  (lower, upper)",
        f"  A  : (0,  ∞)",
        f"  μ  : ({bounds_lo[1]:.2f},  {bounds_hi[1]:.2f}) px²",
        f"  σ  : ({bounds_lo[2]:.3f},  {bounds_hi[2]:.3f}) px²",
        f"  B  : (0,  ∞)",
        "",
        "FIT RESULT",
        f"  status      : {mesg}",
        f"  A           = {popt[0]:.2f}  ±  {perr[0]:.2f}",
        f"  μ (r²)      = {popt[1]:.2f}  ±  {perr[1]:.2f} px²",
        f"  σ (r²)      = {popt[2]:.3f}  ±  {perr[2]:.3f} px²",
        f"  B           = {popt[3]:.2f}  ±  {perr[3]:.2f}",
        f"  χ²_red      = {reduced_chi2_diag:.3f}  (n_dof = {len(r2_w) - 4})",
        f"  r_derived   = √μ = {r_fit_derived:.4f} ± {sig_r_derived:.4f} px",
        "",
        "STORED IN PeakFitR2",
        f"  r2_raw  = {pf.r2_raw_px2:.2f} px²  (detected bin)",
        f"  r2_fit  = {pf.r2_fit_px2:.2f} px²  (TRF centroid μ)",
        f"  σ_r²    = {pf.sigma_r2_fit_px2:.2f} px²",
    ])

    # --- Residuals at data points --------------------------------------------
    if fit_ok:
        y_fit_at_data  = _gaussian(r2_w, *popt)
        residuals      = p_w - y_fit_at_data
    else:
        residuals = None

    # --- Figure (context | fit+residuals | annotation) -----------------------
    fig = plt.figure(figsize=(16, 8))
    gs  = fig.add_gridspec(2, 3,
                           width_ratios=[2, 2.2, 1.8],
                           height_ratios=[3, 1.2],
                           wspace=0.35, hspace=0.12)
    ax_ctx  = fig.add_subplot(gs[:, 0])
    ax_zoom = fig.add_subplot(gs[0, 1])
    ax_res  = fig.add_subplot(gs[1, 1], sharex=ax_zoom)
    ax_ann  = fig.add_subplot(gs[:, 2])
    ax_ann.axis("off")

    # Left — full profile in r² domain, first ~1600 px² (~40 px)
    r2_max_ctx = min(40.0 ** 2, float(fp.r2_grid[good].max()))
    ax_ctx.plot(fp.r2_grid[good], fp.profile[good],
                color="steelblue", linewidth=0.8,
                marker=".", markersize=4, markeredgewidth=0)
    ax_ctx.axvspan(fp.r2_grid[lo], fp.r2_grid[hi],
                   alpha=0.20, color="gold", label="Fitting window")
    ax_ctx.axvline(pf.r2_raw_px2, color="darkorange", linewidth=1.2,
                   linestyle="--", label=f"Detected  {pf.r2_raw_px2:.1f} px²")
    if pf.fit_ok:
        ax_ctx.axvline(pf.r2_fit_px2, color="crimson", linewidth=1.4,
                       label=f"TRF centroid  {pf.r2_fit_px2:.1f} px²")
    ax_ctx.set_xlim(0, r2_max_ctx)
    ax_ctx.set_xlabel("r² (px²)", fontsize=9)
    ax_ctx.set_ylabel("Mean intensity (ADU)", fontsize=9)
    ax_ctx.set_title(f"Full profile  (0 – {r2_max_ctx:.0f} px²)", fontsize=9)
    ax_ctx.legend(fontsize=7)
    ax_ctx.tick_params(labelsize=7)

    # Middle — zoomed fitting window in r²
    ax_zoom.errorbar(r2_w, p_w, yerr=sem_w,
                     fmt="o", color="steelblue", markersize=5,
                     ecolor="cornflowerblue", elinewidth=1.2, capsize=3,
                     zorder=3, label="Data ± 1σ SEM")
    ax_zoom.plot(r2_fine, y_init, color="goldenrod", linewidth=1.5,
                 linestyle="--", zorder=2, label="Initial guess p0")
    if fit_ok and y_fit is not None:
        ax_zoom.plot(r2_fine, y_fit, color="crimson", linewidth=2.0,
                     zorder=4, label="TRF fit")
    ax_zoom.axvline(pf.r2_raw_px2, color="darkorange", linewidth=1.2,
                    linestyle="--", alpha=0.8,
                    label=f"Detected  {pf.r2_raw_px2:.1f} px²")
    if pf.fit_ok:
        ax_zoom.axvline(pf.r2_fit_px2, color="crimson", linewidth=1.4,
                        linestyle="-", alpha=0.9,
                        label=f"TRF centroid  {pf.r2_fit_px2:.1f} px²")
    ax_zoom.set_ylabel("Mean intensity (ADU)", fontsize=9)
    ax_zoom.set_title("Fitting window — zoomed (r² domain)", fontsize=9)
    ax_zoom.legend(fontsize=7)
    ax_zoom.tick_params(labelsize=7)
    plt.setp(ax_zoom.get_xticklabels(), visible=False)

    # Residuals panel
    if fit_ok and residuals is not None:
        ax_res.errorbar(r2_w, residuals, yerr=sem_w,
                        fmt="o", color="steelblue", markersize=5,
                        ecolor="cornflowerblue", elinewidth=1.2, capsize=3,
                        zorder=3, label="Data − fit  ± 1σ SEM")
        ax_res.axhline(0, color="crimson", linewidth=1.2, linestyle="-")
        ax_res.axhspan(-sem_w.mean(), sem_w.mean(),
                       alpha=0.12, color="crimson", label="Mean ±1σ SEM band")
        ax_res.set_ylabel("Residual (ADU)", fontsize=9)
        ax_res.legend(fontsize=7)
    else:
        ax_res.text(0.5, 0.5, "fit failed — no residuals",
                    transform=ax_res.transAxes, ha="center", va="center",
                    fontsize=8, color="gray")
    ax_res.axvline(pf.r2_raw_px2, color="darkorange", linewidth=1.0,
                   linestyle="--", alpha=0.7)
    if pf.fit_ok:
        ax_res.axvline(pf.r2_fit_px2, color="crimson", linewidth=1.0,
                       linestyle="-", alpha=0.7)
    ax_res.set_xlabel("r² (px²)", fontsize=9)
    ax_res.tick_params(labelsize=7)

    # Annotation panel
    ax_ann.text(0.03, 0.97, ann,
                transform=ax_ann.transAxes,
                fontsize=7.5, fontfamily="monospace",
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.6",
                          facecolor="#F8F9FA", edgecolor="#AAAAAA"))

    fig.suptitle(
        f"First Fringe Diagnostic (r² domain)  —  Peak 1  |  "
        f"r²_raw = {pf.r2_raw_px2:.2f} px²   r²_fit = {pf.r2_fit_px2:.2f} px²   "
        f"σ_r² = {pf.sigma_r2_fit_px2:.2f} px²   fit_ok = {pf.fit_ok}",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout()


def _plot_all_fringe_diagnostics_r2(
    fp: FringeProfile,
    fit_half_window: int = 40,
    n_cols: int = 4,
    save_path: "Path | str | None" = None,
    peak_offset: int = 0,
) -> None:
    """
    Grid figure showing the r²-domain Gaussian fitting window for every
    detected peak.  Mirrors _plot_all_fringe_diagnostics but with r² on the
    x-axis, matching the fitting performed by _find_and_fit_peaks_r2.
    """
    peaks = fp.peak_fits_r2
    if not peaks:
        print("No r²-domain peaks detected — skipping all-fringe r² diagnostic.")
        return

    n_peaks = len(peaks)
    n_cols  = min(n_cols, n_peaks)
    n_rows  = (n_peaks + n_cols - 1) // n_cols

    good         = ~fp.masked
    good_indices = np.where(good)[0]
    if good_indices.size > 1:
        median_dr_px = float(np.median(np.diff(fp.r_grid[good])))
        if median_dr_px <= 0.0:
            median_dr_px = 1.0
        median_dr2_px2 = float(np.median(np.diff(fp.r2_grid[good])))
        if median_dr2_px2 <= 0.0:
            median_dr2_px2 = 1.0
    else:
        median_dr_px   = 1.0
        median_dr2_px2 = 1.0

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.2 * n_cols, 3.6 * n_rows),
        squeeze=False,
    )
    fig.suptitle(
        f"All-Fringe Fit Diagnostics (r² domain)  —  {n_peaks} peaks  |  "
        f"median r² bin width = {median_dr2_px2:.3f} px²  |  "
        f"windows from _R2_WINDOWS",
        fontsize=10, fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5, 0.982,
        r"Parabola fit:  $f(r^2) = A \cdot (r^2)^2 + B \cdot r^2 + C$"
        r"  where $A < 0$ (opens downward),  centroid $= -B\,/\,(2A)$,"
        r"  $r$ = radius (px),  $r^2$ = squared radius (px²)",
        ha="center", va="top", fontsize=8.5, color="dimgray",
        fontstyle="italic",
    )

    for k, pf in enumerate(peaks):
        row, col = divmod(k, n_cols)
        ax = axes[row, col]

        bin_idx = pf.peak_idx

        pk_num = k + peak_offset
        if pk_num not in _R2_WINDOWS:
            continue
        _wk2 = _R2_WINDOWS[pk_num]
        r2_lo_w, r2_hi_w = _wk2[0], _wk2[-1]
        lo = int(np.searchsorted(fp.r2_grid, r2_lo_w, side="left"))
        hi = int(np.searchsorted(fp.r2_grid, r2_hi_w, side="right")) - 1
        lo = max(0, lo)
        hi = min(len(fp.r2_grid) - 1, hi)

        win     = np.arange(lo, hi + 1)
        usable  = ~fp.masked[win] & np.isfinite(fp.sigma_profile[win])
        win_use = win[usable]

        r2_w  = fp.r2_grid[win_use]
        p_w   = fp.profile[win_use]
        sem_w = fp.sigma_profile[win_use]

        # Stored fit results from _find_and_fit_peaks_r2
        fit_ok       = pf.fit_ok
        para_ok      = pf.para_ok
        gauss_ok     = pf.gauss_ok
        r2_fit       = pf.r2_fit_px2
        sigma_r2_fit = pf.sigma_r2_fit_px2
        reduced_chi2 = pf.reduced_chi2
        para_rms     = pf.para_rms

        mu0        = float(fp.r2_grid[bin_idx])
        r2_lo_plot = r2_w[0]  if len(r2_w) > 0 else mu0 - 2 * median_dr2_px2
        r2_hi_plot = r2_w[-1] if len(r2_w) > 0 else mu0 + 2 * median_dr2_px2
        r2_fine    = np.linspace(r2_lo_plot, r2_hi_plot, 300)

        # ── Plot ─────────────────────────────────────────────────────────────
        ax.set_facecolor("#F0FFF4" if fit_ok else "#FFF0F0")

        if win_use.size > 0:
            ax.errorbar(r2_w, p_w, yerr=sem_w,
                        fmt="o", color="steelblue", markersize=4,
                        ecolor="cornflowerblue", elinewidth=1.0, capsize=2,
                        zorder=3, label="Data")

        # Parabolic fit curve — recompute polynomial for display (orange solid)
        coeffs = errs = None
        if para_ok and win_use.size >= 3:
            y_min_w  = float(np.min(p_w))
            y_max_w  = float(np.max(p_w))
            thresh   = y_min_w + 0.5 * (y_max_w - y_min_w)
            mask_top = p_w >= thresh
            xm, ym   = r2_w[mask_top], p_w[mask_top]
            if len(xm) >= 3:
                try:
                    coeffs, cov = np.polyfit(xm, ym, 2, cov=True)
                    errs = np.sqrt(np.diag(cov))
                except np.linalg.LinAlgError:
                    coeffs = np.polyfit(xm, ym, 2)
                    errs = np.full(3, float("nan"))
                if coeffs[0] < 0:
                    y_floor   = coeffs[2]
                    poly_vals = np.polyval(coeffs, r2_fine)
                    mask_above = poly_vals >= y_floor
                    if mask_above.any():
                        ax.plot(r2_fine[mask_above], poly_vals[mask_above],
                                color="darkorange", lw=1.8, zorder=4,
                                label="Parabola + offset")

        # Raw detection (gray dashed) and fitted centroid (red solid + band)
        ax.axvline(pf.r2_raw_px2, color="gray", lw=0.9, ls="--", alpha=0.6)
        if fit_ok:
            ax.axvline(r2_fit, color="red", lw=1.2, ls="-", zorder=6,
                       label=f"centroid r²={r2_fit:.1f}")
            if np.isfinite(sigma_r2_fit):
                ax.axvspan(r2_fit - sigma_r2_fit, r2_fit + sigma_r2_fit,
                           alpha=0.15, color="darkorange", zorder=1)

        ax.set_xlim(max(0.0, r2_lo_plot), r2_hi_plot)

        # ── Infobox with 1-sigma parabola uncertainties ───────────────────────
        if coeffs is not None and errs is not None:
            info_txt = (
                f"A={coeffs[0]:.4f}±{errs[0]:.4f}\n"
                f"B={coeffs[1]:.2f}±{errs[1]:.2f}\n"
                f"C={coeffs[2]:.0f}±{errs[2]:.0f} ADU\n"
                f"r²={r2_fit:.2f}±{sigma_r2_fit:.2f} px²\n"
                f"χ²={reduced_chi2:.2f}"
            )
            ax.text(0.97, 0.04, info_txt,
                    transform=ax.transAxes, fontsize=5.5,
                    verticalalignment="bottom", horizontalalignment="right",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              alpha=0.75, edgecolor="gray"))

        # ── Title ────────────────────────────────────────────────────────────
        # pk_num set above — title shows peak number and wavelength only
        lam_str = "640.2" if pk_num % 2 == 0 else "638.3"
        if fit_ok:
            title       = f"Peak {pk_num}  λ={lam_str} nm"
            title_color = "#1a6e2e"
        else:
            title       = f"Peak {pk_num}  λ={lam_str} nm  FAILED"
            title_color = "#b22222"

        ax.set_title(title, fontsize=9.0, color=title_color)
        ax.tick_params(labelsize=6.5)
        ax.set_xlabel("r² [px²]", fontsize=7)
        ax.set_ylabel("ADU", fontsize=7)
        ax.legend(fontsize=5.5, loc="upper left")

    for idx in range(n_peaks, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r, c].axis("off")

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    # ── Peak summary table ───────────────────────────────────────────────────
    col_labels = [
        "Peak", "λ (nm)", "r²_raw (px²)", "r²_fit (px²)", "±σ r² (px²)",
        "r_derived (px)", "±σ_r (px)", "Amp (ADU)", "Width σ r² (px²)",
        "para_ok", "χ²_red",
    ]
    table_data = []
    for k, pf in enumerate(peaks):
        lam_str_k = "640.2" if k % 2 == 0 else "638.3"
        if pf.fit_ok and pf.r2_fit_px2 > 0:
            r_der      = float(np.sqrt(pf.r2_fit_px2))
            sigma_r_dr = pf.sigma_r2_fit_px2 / (2.0 * r_der)
        else:
            r_der      = float("nan")
            sigma_r_dr = float("nan")
        chi2_s = f"{pf.reduced_chi2:.2f}" if np.isfinite(pf.reduced_chi2) else "—"
        row = [
            str(k),
            lam_str_k,
            f"{pf.r2_raw_px2:.1f}",
            f"{pf.r2_fit_px2:.3f}" if pf.fit_ok else "—",
            f"{pf.sigma_r2_fit_px2:.3f}" if (pf.fit_ok and np.isfinite(pf.sigma_r2_fit_px2)) else "—",
            f"{r_der:.4f}" if np.isfinite(r_der) else "—",
            f"{sigma_r_dr:.4f}" if np.isfinite(sigma_r_dr) else "—",
            f"{pf.amplitude_adu:.1f}",
            f"{pf.width_r2_px2:.2f}" if np.isfinite(pf.width_r2_px2) else "—",
            "yes" if pf.para_ok else "no",
            chi2_s,
        ]
        table_data.append(row)

    n_tbl_rows = len(table_data)
    n_tbl_cols = len(col_labels)
    fig_tbl_h  = max(4.0, 0.35 * n_tbl_rows + 1.5)
    fig_tbl, ax_tbl = plt.subplots(figsize=(22, fig_tbl_h))
    ax_tbl.axis("off")
    tbl = ax_tbl.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.3)
    for col in range(n_tbl_cols):
        tbl[(0, col)].set_facecolor("#c8d8f0")
    for row in range(1, n_tbl_rows + 1):
        bg = "#f0f4ff" if row % 2 == 0 else "white"
        for col in range(n_tbl_cols):
            tbl[(row, col)].set_facecolor(bg)
    ax_tbl.set_title(
        "Peak Fit Results (r² domain) — parabolic centroids",
        fontsize=11, fontweight="bold", pad=12,
    )
    fig_tbl.tight_layout()
    plt.close(fig_tbl)  # standalone table suppressed; results in figure2_peak_table()


def _print_peak_table(peak_fits: list[PeakFit]) -> None:
    """Print a formatted summary table of detected peaks to stdout."""
    sep = "-" * 92
    print(f"\n{sep}")
    print(f"  Detected peaks in radial profile  ({len(peak_fits)} found)")
    print(sep)
    if not peak_fits:
        print("  (none)")
        print(sep)
        return

    print(
        f"  {'Peak':>4}  {'r_raw (px)':>10}  {'r_fit (px)':>10}  "
        f"{'+/-sig_r (px)':>13}  {'r_fit (px²)':>12}  {'+/-sig_r (px²)':>14}  "
        f"{'Amp (ADU)':>9}  {'Width sig (px)':>14}"
    )
    print(sep)
    for i, pf in enumerate(peak_fits):
        if pf.fit_ok:
            r_fit_sq = pf.r_fit_px ** 2
            sig_r_sq = 2.0 * pf.r_fit_px * pf.sigma_r_fit_px
            print(
                f"  {i + 1:>4}  {pf.r_raw_px:>10.2f}  {pf.r_fit_px:>10.3f}  "
                f"{pf.sigma_r_fit_px:>13.3f}  {r_fit_sq:>12.2f}  {sig_r_sq:>14.3f}  "
                f"{pf.amplitude_adu:>9.1f}  {pf.width_px:>14.2f}"
            )
        else:
            print(
                f"  {i + 1:>4}  {pf.r_raw_px:>10.2f}  {'---':>10}  "
                f"{'---':>13}  {'---':>12}  {'---':>14}  "
                f"{pf.profile_raw:>9.1f}  {'---':>14}"
            )
    print(sep)


def _print_peak_table_r2(peak_fits_r2: list[PeakFitR2]) -> None:
    """Print a formatted summary table of r²-domain peak fits to stdout."""
    sep = "-" * 100
    print(f"\n{sep}")
    print(f"  r²-domain peak fits  ({len(peak_fits_r2)} found)")
    print(sep)
    if not peak_fits_r2:
        print("  (none)")
        print(sep)
        return

    print(
        f"  {'Peak':>4}  {'r2_raw (px²)':>13}  {'r2_fit (px²)':>13}  "
        f"{'+/-sig_r2 (px²)':>16}  {'r_derived (px)':>14}  {'+/-sig_r (px)':>13}  "
        f"{'Amp (ADU)':>9}  {'Width σ r2 (px²)':>16}"
    )
    print(sep)
    for i, pf in enumerate(peak_fits_r2):
        if pf.fit_ok and pf.r2_fit_px2 > 0:
            r_fit_derived   = float(np.sqrt(pf.r2_fit_px2))
            sigma_r_derived = pf.sigma_r2_fit_px2 / (2.0 * r_fit_derived)
            print(
                f"  {i + 1:>4}  {pf.r2_raw_px2:>13.2f}  {pf.r2_fit_px2:>13.3f}  "
                f"{pf.sigma_r2_fit_px2:>16.3f}  {r_fit_derived:>14.3f}  {sigma_r_derived:>13.3f}  "
                f"{pf.amplitude_adu:>9.1f}  {pf.width_r2_px2:>16.2f}"
            )
        else:
            print(
                f"  {i + 1:>4}  {pf.r2_raw_px2:>13.2f}  {'---':>13}  "
                f"{'---':>16}  {'---':>14}  {'---':>13}  "
                f"{pf.profile_raw:>9.1f}  {'---':>16}"
            )
    print(sep)



# =========================================================================
# Section D — Tolansky two-line analysis
# (copied from src/fpi/tolansky.py)
# =========================================================================


class InsufficientRingsError(ValueError):
    """Raised when the peak table has too few valid rings for analysis."""


# ---------------------------------------------------------------------------
# Task 1 -- result dataclass  (spec §5)
# ---------------------------------------------------------------------------

@dataclass
class TolanskyResult:
    """
    Output of the Tolansky two-line analysis (S13a).
    All two_sigma_ fields are exactly 2 x sigma_ (S04 convention).
    Focal length f is not reported; alpha is the sole plate-scale parameter.
    """
    # --- Family assignment provenance ---
    n_peaks_total:     int
    n_nan_dropped:     int
    n_rings_a:         int
    n_rings_b:         int
    amp_threshold:     float
    Y_B_obs:           float

    # --- Line a  (lam_a = 640.2248 nm) ---
    Delta_a:           float        # [Eq. 3.85]
    sigma_Delta_a:     float
    two_sigma_Delta_a: float
    eps_a:             float        # [Eq. 3.86]
    sigma_eps_a:       float
    two_sigma_eps_a:   float
    chi2_dof_a:        float
    delta_a:           np.ndarray   # successive r^2-differences (px^2)
    r2_a:              np.ndarray   # r^2 values used in fit (for plotting)
    sigma_r2_a:        np.ndarray   # 1sigma uncertainty on r^2, line a

    # --- Line b  (lam_b = 638.2991 nm) ---
    Delta_b:           float        # [Eq. 3.87]
    sigma_Delta_b:     float
    two_sigma_Delta_b: float
    eps_b:             float        # [Eq. 3.88]
    sigma_eps_b:       float
    two_sigma_eps_b:   float
    chi2_dof_b:        float
    delta_b:           np.ndarray
    r2_b:              np.ndarray
    sigma_r2_b:        np.ndarray   # 1sigma uncertainty on r^2, line b

    # --- Consistency check ---
    Delta_ratio_obs:      float
    Delta_ratio_expected: float
    Delta_ratio_residual: float     # |obs - expected| / expected

    # --- Integer disambiguation ---
    N_Delta: int                    # N_Delta = na - nb  [Eq. 3.96 / Benoit]

    # --- Plate spacing recovery  [Eq. 3.97] ---
    d_m:           float
    sigma_d_m:     float
    two_sigma_d_m: float

    # --- Plate scale alpha (primary geometric output) ---
    alpha_a:           float        # alpha from line a  (rad/px)  [from 3.85]
    alpha_b:           float        # alpha from line b  (rad/px)  cross-check
    alpha_mean:        float        # mean of alpha_a and alpha_b  (rad/px)
    sigma_alpha:       float
    two_sigma_alpha:   float
    alpha_consistency: float        # |alpha_a - alpha_b| / alpha_a

    # --- Wavelengths (provenance) ---
    lam_a_nm: float                 # 640.2248
    lam_b_nm: float                 # 638.2991

    # --- Source file (provenance) ---
    source_path: str = ""           # path to the _peak_fits_r2.npy used


# ---------------------------------------------------------------------------
# Task 2 -- input loading and family assignment
# ---------------------------------------------------------------------------

def load_and_split_families(path_or_array, n_pairs: int | None = None):
    """
    Load _peak_fits_r2.npy and split into the two neon-line families.

    Accepts 9-column (legacy amplitude-threshold) and 10-column (guided,
    line_id in col 9) layouts from annular_reduction_2026-05-13.py.

    Parameters
    ----------
    path_or_array : str, Path, or (N, 9|10) ndarray
    n_pairs : int or None
        Keep only the innermost n_pairs rings from each family.  None = all.

    Returns
    -------
    p_a, r2_a, sigma_r2_a  -- line a (640.2248 nm)
    p_b, r2_b, sigma_r2_b  -- line b (638.2991 nm)
    amp_threshold, Y_B_obs, n_nan_dropped, n_peaks_total
    """
    if isinstance(path_or_array, (str, pathlib.Path)):
        peaks = np.load(str(path_or_array))
    else:
        peaks = np.asarray(path_or_array, dtype=float)

    if peaks.ndim != 2 or peaks.shape[1] not in (9, 10):
        raise ValueError(
            f"Expected shape (N, 9) or (N, 10), got {peaks.shape}.  "
            "File must be _peak_fits_r2.npy from annular_reduction.py."
        )

    has_line_id   = (peaks.shape[1] == 10)
    n_peaks_total = peaks.shape[0]

    valid         = np.isfinite(peaks[:, 2])
    n_nan_dropped = int((~valid).sum())
    peaks_ok      = peaks[valid]

    if peaks_ok.shape[0] < 4:
        raise InsufficientRingsError(
            f"Only {peaks_ok.shape[0]} valid fitted peaks; "
            "need >= 4 (>=2 per family)."
        )

    amps          = peaks_ok[:, 6]
    amp_threshold = float(np.median(amps))

    if has_line_id:
        line_id = peaks_ok[:, 9]
        unknown = ~np.isin(line_id, [0.0, 1.0])
        if unknown.any():
            warnings.warn(
                f"{unknown.sum()} peak(s) with line_id not in {{0.0, 1.0}} "
                "excluded from family split.",
                RuntimeWarning, stacklevel=2,
            )
        mask_a = (line_id == 0.0)
        mask_b = (line_id == 1.0)
    else:
        mask_a = amps > amp_threshold
        mask_b = amps <= amp_threshold

    peaks_a = peaks_ok[mask_a]
    peaks_b = peaks_ok[mask_b]

    if n_pairs is not None:
        n_pairs = int(n_pairs)
        peaks_a = peaks_a[:n_pairs]
        peaks_b = peaks_b[:n_pairs]

    if peaks_a.shape[0] < 2 or peaks_b.shape[0] < 2:
        raise InsufficientRingsError(
            "Family split produced < 2 rings in one family "
            f"({'line_id' if has_line_id else 'amplitude threshold'}).  "
            "Check peak detection parameters."
        )

    Y_B_obs = float(np.median(amps[mask_b]) / np.median(amps[mask_a]))
    if Y_B_obs < 0.15 or Y_B_obs > 0.60:
        warnings.warn(
            f"Y_B_obs = {Y_B_obs:.3f} outside expected range [0.15, 0.60].  "
            "Check exposure, dark subtraction, or family assignment.",
            RuntimeWarning, stacklevel=3,
        )

    p_a        = np.arange(1, peaks_a.shape[0] + 1, dtype=float)
    p_b        = np.arange(1, peaks_b.shape[0] + 1, dtype=float)
    r2_a       = peaks_a[:, 2]
    sigma_r2_a = peaks_a[:, 3]
    r2_b       = peaks_b[:, 2]
    sigma_r2_b = peaks_b[:, 3]

    return (p_a, r2_a, sigma_r2_a,
            p_b, r2_b, sigma_r2_b,
            amp_threshold, Y_B_obs, n_nan_dropped, n_peaks_total)


# ---------------------------------------------------------------------------
# Task 3 -- WLS helper  (spec §4 Step 4, equations given verbatim)
# ---------------------------------------------------------------------------

def _wls(p, r2, sigma_r2):
    """Weighted least-squares fit r^2 = S*p + b (spec §4 Step 4)."""
    w        = 1.0 / sigma_r2 ** 2
    sum_w    = np.sum(w)
    sum_wp   = np.sum(w * p)
    sum_wp2  = np.sum(w * p ** 2)
    sum_wr2  = np.sum(w * r2)
    sum_wpr2 = np.sum(w * p * r2)
    Lambda   = sum_w * sum_wp2 - sum_wp ** 2
    S        = (sum_w * sum_wpr2 - sum_wp * sum_wr2) / Lambda
    b        = (sum_wp2 * sum_wr2 - sum_wp * sum_wpr2) / Lambda
    var_S    = sum_w  / Lambda
    var_b    = sum_wp2 / Lambda
    eps      = 1.0 + b / S
    sig_eps  = np.sqrt((np.sqrt(var_b) / S) ** 2
                       + (b * np.sqrt(var_S) / S ** 2) ** 2)
    chi2_dof = np.sum(w * (r2 - S * p - b) ** 2) / max(len(p) - 2, 1)
    return dict(
        Delta=S,           sigma_Delta=np.sqrt(var_S),
        eps=eps % 1.0,     sigma_eps=sig_eps,
        chi2_dof=chi2_dof,
        intercept=b,       sigma_intercept=np.sqrt(var_b),
    )


# ---------------------------------------------------------------------------
# Task 4 -- Benoit d recovery  (Vaughan Eq. 3.97)
# ---------------------------------------------------------------------------

def benoit_d(eps_a, sigma_eps_a, eps_b, sigma_eps_b,
             lam_a_m, lam_b_m, d_prior_m, n_air=1.0):
    """
    Recover etalon plate spacing d from the two fractional orders.

    d = (N_Delta + eps_a - eps_b) * lam_a * lam_b / (2*n_air*(lam_b - lam_a))

    N_Delta is resolved from d_prior_m and fixes only the FSR-period ambiguity.

    Returns (d_m, sigma_d_m, N_Delta).
    """
    N_Delta = int(round(2.0 * d_prior_m * (1.0 / lam_a_m - 1.0 / lam_b_m)))

    d = ((N_Delta + eps_a - eps_b)
         * lam_a_m * lam_b_m
         / (2.0 * n_air * (lam_b_m - lam_a_m)))
    d = abs(d)

    factor  = lam_a_m * lam_b_m / (2.0 * n_air * abs(lam_b_m - lam_a_m))
    sigma_d = factor * np.sqrt(sigma_eps_a ** 2 + sigma_eps_b ** 2)

    return d, sigma_d, N_Delta


# ---------------------------------------------------------------------------
# Task 5 -- alpha recovery  (from Vaughan Eq. 3.85)
# ---------------------------------------------------------------------------

def recover_alpha(Delta_a, sigma_Delta_a, Delta_b, sigma_Delta_b,
                  d_m, sigma_d_m, lam_a_m, lam_b_m, n_air=1.0):
    """
    Recover angular pixel scale alpha (rad/px).

    alpha = sqrt(lam * n_air / (d * Delta))   [from Vaughan Eq. 3.85]

    Returns (alpha_a, alpha_b, alpha_mean, sigma_alpha, alpha_consistency).
    No focal length is computed or returned.
    """
    alpha_a = np.sqrt(lam_a_m * n_air / (d_m * Delta_a))
    alpha_b = np.sqrt(lam_b_m * n_air / (d_m * Delta_b))
    alpha_mean = 0.5 * (alpha_a + alpha_b)

    sig_alpha = 0.5 * alpha_a * np.sqrt(
        (sigma_Delta_a / Delta_a) ** 2 + (sigma_d_m / d_m) ** 2
    )
    alpha_consistency = abs(alpha_a - alpha_b) / alpha_a

    return alpha_a, alpha_b, alpha_mean, sig_alpha, alpha_consistency


# ---------------------------------------------------------------------------
# Task 6 -- top-level run_tolansky_2line()
# ---------------------------------------------------------------------------

def run_tolansky_2line(
    peaks_input,
    lam_a_m:   float = 640.2248e-9,
    lam_b_m:   float = 638.2991e-9,
    d_prior_m: float = 20.008e-3,
    n_air:     float = 1.0,
    n_pairs:   int | None = None,
) -> TolanskyResult:
    """
    Run the full S13a two-line Tolansky analysis on a neon calibration image.

    Parameters
    ----------
    peaks_input : ndarray or path
        (N, 9) or (N, 10) float64 array from annular_reduction.py, or path to
        the ``{stem}_peak_fits_r2.npy`` file it writes.
    lam_a_m : float
        Brighter-line wavelength [m].  Default: 640.2248 nm.
    lam_b_m : float
        Dimmer-line wavelength [m].    Default: 638.2991 nm.
    d_prior_m : float
        Prior plate spacing [m]; used only to resolve N_Delta.
        Default: 20.008 mm (ICOS build report).
    n_air : float
        Refractive index of etalon gap.  Default: 1.0.
    n_pairs : int or None
        Number of ring pairs (one per neon line) to use in the WLS fit.
        Keeps the innermost n_pairs rings from each family.  None uses all.

    Returns
    -------
    TolanskyResult
    """
    source_path = (str(peaks_input)
                   if isinstance(peaks_input, (str, pathlib.Path)) else "")

    (p_a, r2_a, sigma_r2_a,
     p_b, r2_b, sigma_r2_b,
     amp_threshold, Y_B_obs,
     n_nan_dropped, n_peaks_total) = load_and_split_families(peaks_input,
                                                              n_pairs=n_pairs)

    fit_a = _wls(p_a, r2_a, sigma_r2_a)
    fit_b = _wls(p_b, r2_b, sigma_r2_b)

    Delta_a      = fit_a["Delta"]
    sigma_Delta_a = fit_a["sigma_Delta"]
    eps_a        = fit_a["eps"]
    sigma_eps_a  = fit_a["sigma_eps"]
    chi2_dof_a   = fit_a["chi2_dof"]

    Delta_b      = fit_b["Delta"]
    sigma_Delta_b = fit_b["sigma_Delta"]
    eps_b        = fit_b["eps"]
    sigma_eps_b  = fit_b["sigma_eps"]
    chi2_dof_b   = fit_b["chi2_dof"]

    delta_a = np.diff(r2_a)
    delta_b = np.diff(r2_b)


    Delta_ratio_obs      = Delta_a / Delta_b
    Delta_ratio_expected = lam_a_m / lam_b_m
    Delta_ratio_residual = (abs(Delta_ratio_obs - Delta_ratio_expected)
                            / Delta_ratio_expected)
    if Delta_ratio_residual > 0.002:
        warnings.warn(
            f"Delta_a/Delta_b = {Delta_ratio_obs:.6f} deviates from "
            f"lam_a/lam_b = {Delta_ratio_expected:.6f} by "
            f"{Delta_ratio_residual*1e6:.0f} ppm (>200 ppm).  "
            "Check family assignment.",
            RuntimeWarning,
            stacklevel=2,
        )

    d_m, sigma_d_m, N_Delta = benoit_d(
        eps_a, sigma_eps_a, eps_b, sigma_eps_b,
        lam_a_m, lam_b_m, d_prior_m, n_air,
    )

    (alpha_a, alpha_b, alpha_mean,
     sigma_alpha, alpha_consistency) = recover_alpha(
        Delta_a, sigma_Delta_a, Delta_b, sigma_Delta_b,
        d_m, sigma_d_m, lam_a_m, lam_b_m, n_air,
    )

    if alpha_consistency > 0.001:
        warnings.warn(
            f"|alpha_a - alpha_b| / alpha_a = "
            f"{alpha_consistency*1e6:.0f} ppm (>1000 ppm).  "
            "Check family assignment or ring quality.",
            RuntimeWarning,
            stacklevel=2,
        )

    return TolanskyResult(
        n_peaks_total=n_peaks_total,
        n_nan_dropped=n_nan_dropped,
        n_rings_a=len(p_a),
        n_rings_b=len(p_b),
        amp_threshold=amp_threshold,
        Y_B_obs=Y_B_obs,

        Delta_a=Delta_a,
        sigma_Delta_a=sigma_Delta_a,
        two_sigma_Delta_a=2.0 * sigma_Delta_a,
        eps_a=eps_a,
        sigma_eps_a=sigma_eps_a,
        two_sigma_eps_a=2.0 * sigma_eps_a,
        chi2_dof_a=chi2_dof_a,
        delta_a=delta_a,
        r2_a=r2_a.copy(),
        sigma_r2_a=sigma_r2_a.copy(),

        Delta_b=Delta_b,
        sigma_Delta_b=sigma_Delta_b,
        two_sigma_Delta_b=2.0 * sigma_Delta_b,
        eps_b=eps_b,
        sigma_eps_b=sigma_eps_b,
        two_sigma_eps_b=2.0 * sigma_eps_b,
        chi2_dof_b=chi2_dof_b,
        delta_b=delta_b,
        r2_b=r2_b.copy(),
        sigma_r2_b=sigma_r2_b.copy(),

        Delta_ratio_obs=Delta_ratio_obs,
        Delta_ratio_expected=Delta_ratio_expected,
        Delta_ratio_residual=Delta_ratio_residual,

        N_Delta=N_Delta,

        d_m=d_m,
        sigma_d_m=sigma_d_m,
        two_sigma_d_m=2.0 * sigma_d_m,

        alpha_a=alpha_a,
        alpha_b=alpha_b,
        alpha_mean=alpha_mean,
        sigma_alpha=sigma_alpha,
        two_sigma_alpha=2.0 * sigma_alpha,
        alpha_consistency=alpha_consistency,

        lam_a_nm=lam_a_m * 1e9,
        lam_b_nm=lam_b_m * 1e9,
        source_path=source_path,
    )


# ---------------------------------------------------------------------------
# Task 7 -- rectangular array table  (spec §6)
# ---------------------------------------------------------------------------

def print_rectangular_array(result: TolanskyResult) -> None:
    """Print the Vaughan (1989) Table 3.1 analog and Benoit summary."""
    r = result
    sep = "=" * 68

    print(f"\n{sep}")
    print("=== TOLANSKY RECTANGULAR ARRAY  (Vaughan 1989, Table 3.1 analog) ===")
    print(sep)

    yb_status = "PASS" if 0.15 <= r.Y_B_obs <= 0.60 else "WARN"
    print(f"\nFamily assignment:  amp_threshold = {r.amp_threshold:.1f} ADU"
          f"   Y_B_obs = {r.Y_B_obs:.3f}  [{yb_status}]")
    print(f"  640.2248 nm (line a):  N rings = {r.n_rings_a:2d}")
    print(f"  638.2991 nm (line b):  N rings = {r.n_rings_b:2d}")

    for lbl, r2_arr, delta_arr, Delta, sDelta, eps, seps, chi2 in [
        ("a", r.r2_a, r.delta_a,
         r.Delta_a, r.sigma_Delta_a, r.eps_a, r.sigma_eps_a, r.chi2_dof_a),
        ("b", r.r2_b, r.delta_b,
         r.Delta_b, r.sigma_Delta_b, r.eps_b, r.sigma_eps_b, r.chi2_dof_b),
    ]:
        lam_str = "640.2248" if lbl == "a" else "638.2991"
        sub     = "a" if lbl == "a" else "b"
        print(f"\nComponent {lbl}  (lam_{sub} = {lam_str} nm)")

        n = len(r2_arr)
        p_strs  = "".join(f"  {i+1:>10}" for i in range(n))
        r2_strs = "".join(f"  {v:>10.2f}" for v in r2_arr)
        print(f"  p   :{p_strs}")
        print(f"  r^2 :{r2_strs}  (px^2)")

        if len(delta_arr) > 0:
            d_strs = "".join(
                f"  d{i+1}{i+2}={delta_arr[i]:.1f}"
                for i in range(len(delta_arr))
            )
            print(f"        {d_strs}")

        print(f"  Delta_{sub} (WLS slope) = {Delta:.2f} +/- {sDelta:.2f} px^2"
              f"   chi2_dof = {chi2:.2f}")
        print(f"  eps_{sub}               = {eps:.4f}   +/- {seps:.4f}")

    ratio_ppm = r.Delta_ratio_residual * 1e6
    ratio_ok  = "PASS" if ratio_ppm < 200 else "WARN"
    print(f"\nRatio  Delta_a/Delta_b observed = {r.Delta_ratio_obs:.6f}"
          f"   expected (lam_a/lam_b) = {r.Delta_ratio_expected:.6f}")
    print(f"        residual = {ratio_ppm:.1f} ppm"
          f"   [{ratio_ok} if < 200 ppm]")

    print(f"\n{sep}")
    print("=== BENOIT RECOVERY  (Vaughan Eqs. 3.94-3.97) ===")
    print(f"  N_Delta = na - nb = {r.N_Delta}"
          f"   [from d_prior = 20.008 mm, Eq. 3.96]")
    d_mm   = r.d_m * 1e3
    sd_mm  = r.sigma_d_m * 1e3
    sd2_mm = r.two_sigma_d_m * 1e3
    print(f"  d   = {d_mm:.6f} +/- {sd_mm:.4f} mm   (2sigma = {sd2_mm:.4f} mm)")

    print(f"\n{sep}")
    print("=== PLATE SCALE ===")
    alpha_ok = "PASS" if r.alpha_consistency < 0.001 else "WARN"
    print(f"  alpha_a  = {r.alpha_a:.4e} +/- {r.sigma_alpha:.4e} rad/px"
          f"   [from Delta_a, d, lam_a]")
    print(f"  alpha_b  = {r.alpha_b:.4e} rad/px"
          f"   [from Delta_b, d, lam_b; cross-check]")
    print(f"  alpha_mean = {r.alpha_mean:.4e} +/- {r.sigma_alpha:.4e} rad/px"
          f"   (2sigma = {r.two_sigma_alpha:.4e})")
    print(f"  |alpha_a - alpha_b| / alpha_a = {r.alpha_consistency*1e6:.1f} ppm"
          f"   [{alpha_ok} if < 1000 ppm]")
    print(sep)


# ---------------------------------------------------------------------------
# Task 8 -- M05 priors handoff  (spec §7)
# ---------------------------------------------------------------------------

def to_m05_priors(result: TolanskyResult) -> dict:
    """Convert TolanskyResult to the prior dict expected by M05 FitConfig."""
    d_mm     = result.d_m * 1e3
    sig_d_mm = result.sigma_d_m * 1e3
    return {
        "t_init_mm":    d_mm,
        "t_bounds_mm":  (d_mm - 3 * sig_d_mm, d_mm + 3 * sig_d_mm),
        "alpha_init":   result.alpha_mean,
        "alpha_bounds": (result.alpha_mean * 0.875,
                         result.alpha_mean * 1.125),
        "epsilon_cal_a": result.eps_a,
        "epsilon_cal_b": result.eps_b,
    }


# ---------------------------------------------------------------------------
# Task 9 -- diagnostic figure
# ---------------------------------------------------------------------------

def plot_tolansky_result(
    result: TolanskyResult,
    save_path=None,
    subtitle: str = "",
):
    """
    Four-panel diagnostic figure for the S13a two-line Tolansky analysis.

    A) Joint Tolansky plot -- r^2 vs p for both neon families with individual
       WLS fit lines.  Confirms linearity and correct slope ratio.

    B) WLS fit residuals for both families.  Random scatter with no trend
       confirms good ring detection and correct family assignment.

    C) Successive Delta(r^2) for both families with dashed mean-slope references.
       CV < 2 % confirms etalon parallelism.

    D) Summary text -- recovered d, alpha, eps_a, eps_b, N_Delta, chi2/nu.

    Parameters
    ----------
    result   : TolanskyResult from run_tolansky_2line()
    save_path: optional file path; if given the figure is saved at dpi=150.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    r = result
    u2 = "px²"

    BLUE   = "tab:blue"
    ORANGE = "tab:orange"
    GREEN  = "tab:green"
    RED    = "tab:red"
    GRAY   = "gray"
    BLACK  = "black"

    fig = plt.figure(figsize=(14, 10), facecolor="white")
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            hspace=0.44, wspace=0.37,
                            left=0.09, right=0.97,
                            top=0.91, bottom=0.08)
    ax_tol = fig.add_subplot(gs[0, 0])
    ax_res = fig.add_subplot(gs[1, 0])
    ax_dr2 = fig.add_subplot(gs[0, 1])
    ax_txt = fig.add_subplot(gs[1, 1])

    for ax in [ax_tol, ax_res, ax_dr2, ax_txt]:
        ax.set_facecolor("white")
        ax.tick_params(colors=BLACK, which="both", direction="in")
        for sp in ax.spines.values():
            sp.set_edgecolor(BLACK)
        ax.xaxis.label.set_color(BLACK)
        ax.yaxis.label.set_color(BLACK)
        ax.title.set_color(BLACK)

    p_a = np.arange(1, r.n_rings_a + 1, dtype=float)
    p_b = np.arange(1, r.n_rings_b + 1, dtype=float)
    int_a = r.Delta_a * (r.eps_a - 1.0)
    int_b = r.Delta_b * (r.eps_b - 1.0)
    p_all  = np.concatenate([p_a, p_b])
    p_fine = np.linspace(0.0, p_all.max() + 0.3, 300)
    fit_a  = r.Delta_a * p_fine + int_a
    fit_b  = r.Delta_b * p_fine + int_b

    data_max = max(r.r2_a.max(), r.r2_b.max()) * 1.05
    y_min_tol = -0.05 * data_max   # small negative margin
    ratio_ppm = r.Delta_ratio_residual * 1e6

    _x_max_tol = min(float(p_all.max()), 16.0)

    # -- A: Joint Tolansky plot ------------------------------------------------
    ax_tol.axhline(0, color=GRAY, lw=0.7, ls=":", zorder=0)
    ax_tol.errorbar(p_a, r.r2_a, yerr=r.sigma_r2_a,
                    fmt="o", color=BLUE, ecolor=GRAY, capsize=4, ms=6,
                    lw=1.4, zorder=3,
                    label=f"λ_a = {r.lam_a_nm:.4f} nm (air)  (n={r.n_rings_a})")
    ax_tol.errorbar(p_b, r.r2_b, yerr=r.sigma_r2_b,
                    fmt="s", color=ORANGE, ecolor=GRAY, capsize=4, ms=6,
                    lw=1.4, zorder=3,
                    label=f"λ_b = {r.lam_b_nm:.4f} nm (air)  (n={r.n_rings_b})")
    ax_tol.plot(p_fine, fit_a, color=BLUE,   lw=1.8, ls="-",  zorder=2,
                label=f"Fit a:  Δ_a = {r.Delta_a:.4g}")
    ax_tol.plot(p_fine, fit_b, color=ORANGE, lw=1.8, ls="--", zorder=2,
                label=f"Fit b:  Δ_b = {r.Delta_b:.4g}")
    ax_tol.set_xlim(-0.2, _x_max_tol + 0.5)
    ax_tol.set_ylim(y_min_tol, data_max)
    ax_tol.set_xticks(range(0, int(_x_max_tol) + 1))
    ax_tol.set_xlabel("Fringe index  $p$", fontsize=11)
    ax_tol.set_ylabel(f"$r^2$  [{u2}]", fontsize=11)
    ax_tol.set_title("A — Tolansky Plot  (both neon lines)",
                      fontsize=11, fontweight="bold", pad=7)
    _tol_leg = ax_tol.legend(fontsize=8, facecolor="white", labelcolor=BLACK,
                             edgecolor=BLACK, framealpha=0.9, ncol=2, loc="upper left")
    _wls_ann = (
        f"WLS fit results\n"
        f"  Δ_a={r.Delta_a:.5g}±{r.sigma_Delta_a:.3g} px²/fr"
        f"  ε_a={r.eps_a:.5f}±{r.sigma_eps_a:.2g}"
        f"  χ²/ν={r.chi2_dof_a:.3f}\n"
        f"  Δ_b={r.Delta_b:.5g}±{r.sigma_Delta_b:.3g} px²/fr"
        f"  ε_b={r.eps_b:.5f}±{r.sigma_eps_b:.2g}"
        f"  χ²/ν={r.chi2_dof_b:.3f}\n"
        f"  Δ_a/Δ_b={r.Delta_ratio_obs:.6f}"
        f"  λ_a/λ_b={r.Delta_ratio_expected:.6f}"
        f"  Δ={ratio_ppm:.0f} ppm"
        f"  {'[PASS]' if ratio_ppm < 200 else '[WARN]'}"
    )

    # -- B: Residuals ----------------------------------------------------------
    resid_a = r.r2_a - (r.Delta_a * p_a + int_a)
    resid_b = r.r2_b - (r.Delta_b * p_b + int_b)
    ax_res.axhline(0, color=GRAY, lw=1.0, ls="--", zorder=1)
    ax_res.errorbar(p_a, resid_a, yerr=r.sigma_r2_a,
                    fmt="o", color=BLUE, ecolor=GRAY, capsize=4, ms=6,
                    lw=1.4, zorder=3, label="Line a")
    ax_res.errorbar(p_b, resid_b, yerr=r.sigma_r2_b,
                    fmt="s", color=ORANGE, ecolor=GRAY, capsize=4, ms=6,
                    lw=1.4, zorder=3, label="Line b")
    ax_res.set_xlabel("Fringe index  $p$", fontsize=11)
    ax_res.set_ylabel(f"Residual  [{u2}]", fontsize=11)
    ax_res.set_title("B — WLS Residuals",
                      fontsize=11, fontweight="bold", pad=7)
    ax_res.legend(fontsize=9, facecolor="white", labelcolor=BLACK,
                  edgecolor=BLACK, framealpha=0.9)
    ax_res.set_xlim(-0.2, _x_max_tol + 0.5)

    # -- C: Successive Delta(r^2) ----------------------------------------------
    p_mid_a = 0.5 * (p_a[:-1] + p_a[1:])
    p_mid_b = 0.5 * (p_b[:-1] + p_b[1:])
    sdelta_a = np.sqrt(r.sigma_r2_a[1:] ** 2 + r.sigma_r2_a[:-1] ** 2)
    sdelta_b = np.sqrt(r.sigma_r2_b[1:] ** 2 + r.sigma_r2_b[:-1] ** 2)

    cv_a = (r.delta_a.std() / abs(r.delta_a.mean()) * 100
            if r.delta_a.mean() != 0 and len(r.delta_a) > 1 else np.nan)
    cv_b = (r.delta_b.std() / abs(r.delta_b.mean()) * 100
            if r.delta_b.mean() != 0 and len(r.delta_b) > 1 else np.nan)

    ax_dr2.axhline(r.Delta_a, color=BLUE,   lw=1.2, ls="--",
                   label=f"Δ_a = {r.Delta_a:.4g} (WLS slope)")
    ax_dr2.axhline(r.Delta_b, color=ORANGE, lw=1.2, ls=":",
                   label=f"Δ_b = {r.Delta_b:.4g} (WLS slope)")
    if len(r.delta_a) > 0:
        ax_dr2.errorbar(p_mid_a, r.delta_a, yerr=sdelta_a,
                        fmt="o", color=BLUE, ecolor=GRAY,
                        capsize=4, ms=6, lw=1.4, zorder=3)
    if len(r.delta_b) > 0:
        ax_dr2.errorbar(p_mid_b, r.delta_b, yerr=sdelta_b,
                        fmt="s", color=ORANGE, ecolor=GRAY,
                        capsize=4, ms=6, lw=1.4, zorder=3)
    ax_dr2.set_xlabel("Fringe index  $p$  (midpoint)", fontsize=11)
    ax_dr2.set_ylabel(f"$\\Delta(r^2)$  [{u2}]", fontsize=11)
    ax_dr2.set_title("C — Successive  $\\Delta(r^2)$",
                      fontsize=11, fontweight="bold", pad=7)
    ax_dr2.legend(fontsize=8.5, facecolor="white", labelcolor=BLACK,
                  edgecolor=BLACK, framealpha=0.9)
    cv_col = GREEN if max(cv_a, cv_b) < 2 else ("goldenrod" if max(cv_a, cv_b) < 5 else RED)
    ax_dr2.text(0.97, 0.07,
                f"CV_a = {cv_a:.1f}%   CV_b = {cv_b:.1f}%",
                transform=ax_dr2.transAxes,
                ha="right", va="bottom", fontsize=8.5, color=cv_col)
    ax_dr2.set_xlim(-0.2, _x_max_tol + 0.5)

    # -- D: Summary ------------------------------------------------------------
    ax_txt.axis("off")

    d_mm      = r.d_m * 1e3
    sig_d_mm  = r.sigma_d_m * 1e3
    sig2_d_mm = r.two_sigma_d_m * 1e3
    ratio_col = GREEN if ratio_ppm < 200 else RED
    yb_col    = GREEN if 0.15 <= r.Y_B_obs <= 0.60 else RED
    chi_col_a = GREEN if r.chi2_dof_a < 2 else "goldenrod"
    chi_col_b = GREEN if r.chi2_dof_b < 2 else "goldenrod"

    lines_txt = [
        (f"  N rings: {r.n_peaks_total} total"
         f"  ({r.n_rings_a} line a + {r.n_rings_b} line b)"
         f"   NaN dropped: {r.n_nan_dropped}",                         GRAY,    8.5,  "normal", "normal", 0.00),
        (f"  Y_B_obs = {r.Y_B_obs:.3f}"
         f"   {'[PASS]' if 0.15<=r.Y_B_obs<=0.60 else '[WARN]'}"
         f"   amp_threshold = {r.amp_threshold:.0f} ADU",              yb_col,  8.5,  "normal", "normal", 0.00),
        ("── Benoit recovery ───────────────────────────────────────",  GRAY,    8.0,  "normal", "normal", 0.01),
        ("  d = (N_Δ + ε_a−ε_b)·λ_a·λ_b / [2·n·(λ_b−λ_a)]",          GRAY,    8.0,  "normal", "italic", 0.00),
        (f"  N_Δ = {r.N_Delta}"
         f"   ε_a − ε_b = {r.eps_a - r.eps_b:+.6f}",                 BLACK,   8.5,  "normal", "normal", 0.00),
        (f"  d  = {d_mm:.5f} ± {sig_d_mm:.4f} mm"
         f"   (2σ: ±{sig2_d_mm:.4f} mm)",                             GREEN,   9.5,  "bold",   "normal", 0.00),
        ("── Plate scale ───────────────────────────────────────────",  GRAY,    8.0,  "normal", "normal", 0.01),
        ("  α_a = √(λ_a·n_air / (d·Δ_a))",                             GRAY,    8.0,  "normal", "italic", 0.00),
        (f"  α_a = {r.alpha_a:.6e} rad/px"
         f"   α_b = {r.alpha_b:.6e} rad/px",                          BLACK,   8.5,  "normal", "normal", 0.00),
        (f"  α   = {r.alpha_mean:.6e} ± {r.sigma_alpha:.2e} rad/px"
         f"   (2σ: ±{r.two_sigma_alpha:.2e})",                        "purple", 9.5, "bold",   "normal", 0.00),
        (f"  consistency: {r.alpha_consistency*1e6:.1f} ppm"
         f"   {'[PASS]' if r.alpha_consistency<0.001 else '[WARN]'}",
         GREEN if r.alpha_consistency < 0.001 else RED, 8.5, "normal", "normal", 0.00),
    ]

    y = 0.98
    for text, color, size, weight, style, gap in lines_txt:
        y -= gap
        ax_txt.text(0.02, y, text, transform=ax_txt.transAxes,
                    ha="left", va="top", fontsize=size,
                    color=color, fontweight=weight, fontstyle=style,
                    fontfamily="monospace")
        y -= size * 0.010 + 0.005

    # Place WLS annotation flush below the legend bottom edge
    fig.canvas.draw()
    _renderer = fig.canvas.get_renderer()
    _leg_bb   = _tol_leg.get_window_extent(_renderer)
    _leg_y0   = ax_tol.transAxes.inverted().transform((0, _leg_bb.y0))[1]
    ax_tol.text(0.02, _leg_y0 - 0.04, _wls_ann,
                transform=ax_tol.transAxes,
                ha="left", va="top", fontsize=7.5,
                color=BLACK, fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=GRAY, alpha=0.85))

    src_label = (subtitle or
                 (str(pathlib.Path(r.source_path)) if r.source_path else ""))
    fig.suptitle(
        "Tolansky Two-Line Analysis  "
        f"(λ_a = {r.lam_a_nm:.4f} nm,  λ_b = {r.lam_b_nm:.4f} nm)",
        color=BLACK, fontsize=13, fontweight="bold", y=0.99,
    )
    if src_label:
        fig.text(0.5, 0.965, src_label,
                 ha="center", va="top", fontsize=9,
                 color=GRAY, fontfamily="monospace")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  Figure saved -> {save_path}")
    return fig


# ---------------------------------------------------------------------------
# Backward-compatibility alias
# run_tolansky() was the public name in tolansky_2line_2026-05-05.py.
# Existing callers (tests, tolansky-2line.py wrapper) continue to work.
# ---------------------------------------------------------------------------
run_tolansky = run_tolansky_2line  # noqa: F811


# =========================================================================
# Section E — H05 staged calibration inversion
# (copied from src/processing/H05_calibration_inversion_2026_05_12.py)
# =========================================================================


log = logging.getLogger("H05")


# ---------------------------------------------------------------------------
# Parameter ordering (12 positions in p_all vector)
# ---------------------------------------------------------------------------
_NAMES = ['t_m', 'alpha', 'R1', 'R2', 'I0', 'I1', 'I2', 'I3',
          'sigma0', 'sigma1', 'sigma2', 'B', 'ne_ratio']
_IDX   = {n: i for i, n in enumerate(_NAMES)}

# sigma1, sigma2 fixed at 0 throughout — see §2 (PSF investigation)
_FIXED_PSF = {'sigma1', 'sigma2'}

_STAGE_FREE = {
    1: ['I0', 'I1', 'I2', 'I3', 'B'],
    2: ['t_m', 'alpha', 'R1', 'R2', 'I0', 'I1', 'I2', 'B'],
    3: ['t_m', 'alpha', 'R1', 'R2', 'I0', 'I1', 'I2', 'I3', 'sigma0', 'B'],
    4: [n for n in _NAMES if n not in _FIXED_PSF],   # 11 free
}

# ---------------------------------------------------------------------------
# Forward model
# ---------------------------------------------------------------------------

def _neon_model(r_arr, r_max, t, alpha, R1, R2, I0, I1, I2, I3,
                sigma0, sigma1, sigma2, B, ne_ratio, _N_fine=500):
    """Two-line neon: S(r) = Ã(r;λ₁,R1) + ne_ratio·Ã(r;λ₂,R2) + B"""
    r_fine = np.linspace(0.0, r_max, _N_fine)
    A1 = airy_modified(r_fine, NE_WAVELENGTH_1_AIR_M,
                       t, R1, alpha, 1.0, r_max,
                       I0, I1, I2, sigma0, sigma1, sigma2, I3=I3)
    A2 = airy_modified(r_fine, NE_WAVELENGTH_2_AIR_M,
                       t, R2, alpha, 1.0, r_max,
                       I0, I1, I2, sigma0, sigma1, sigma2, I3=I3)
    return np.interp(r_arr, r_fine, A1 + ne_ratio * A2 + B)


def _model_components(r_arr, r_max, t, alpha, R1, R2, I0, I1, I2, I3,
                      sigma0, sigma1, sigma2, B, ne_ratio, n_fine=2000):
    """Return (composite, lam1+B, ne_ratio*lam2+B) for plotting."""
    r_fine = np.linspace(0.0, r_max, n_fine)
    A1 = airy_modified(r_fine, NE_WAVELENGTH_1_AIR_M,
                       t, R1, alpha, 1.0, r_max,
                       I0, I1, I2, sigma0, sigma1, sigma2, I3=I3)
    A2 = airy_modified(r_fine, NE_WAVELENGTH_2_AIR_M,
                       t, R2, alpha, 1.0, r_max,
                       I0, I1, I2, sigma0, sigma1, sigma2, I3=I3)
    comp = np.interp(r_arr, r_fine, A1 + ne_ratio * A2 + B)
    lam1 = np.interp(r_arr, r_fine, A1 + B)
    lam2 = np.interp(r_arr, r_fine, ne_ratio * A2 + B)
    return comp, lam1, lam2


# ---------------------------------------------------------------------------
# Finite-difference Jacobian for covariance
# ---------------------------------------------------------------------------

def _fd_jacobian(residual_fn, p0, rel_step=1e-5):
    """
    Forward finite-difference Jacobian at p0.  No bounds required.
    step = max(rel_step * |p0[j]|, 1e-10) per parameter.
    Returns J shape (n_residuals, n_params).
    """
    r0  = residual_fn(p0)
    n_r, n_p = len(r0), len(p0)
    J   = np.zeros((n_r, n_p))
    for j in range(n_p):
        h      = max(rel_step * abs(p0[j]), 1e-10)
        p_fwd  = p0.copy(); p_fwd[j] += h
        J[:, j] = (residual_fn(p_fwd) - r0) / h
    return J


# ---------------------------------------------------------------------------
# LM stage runner
# ---------------------------------------------------------------------------

def _run_stage(r_good, prof_good, sig_good, r_max,
               p_all, free_names, bounds_dict, config):
    """
    Run one LM stage (soft-bound penalties) then compute FD covariance.
    Returns (p_updated, cov, stderrs, chi2_red, lm_result).
    stderrs indexed by position in free_names.
    """
    free_idx  = np.array([_IDX[n] for n in free_names])
    p_fixed   = p_all.copy()
    p0        = p_fixed[free_idx]
    n_good    = len(r_good)
    n_free    = len(free_names)

    lo_arr    = np.array([bounds_dict[n][1] for n in free_names])
    hi_arr    = np.array([bounds_dict[n][2] for n in free_names])
    range_arr = hi_arr - lo_arr + 1e-30
    pen_sigma = range_arr * 0.01

    def _residuals_pen(p_free):
        p = p_fixed.copy(); p[free_idx] = p_free
        t, alpha, R1, R2, I0, I1, I2, I3, s0, s1, s2, B, ne = p
        model  = _neon_model(r_good, r_max, t, alpha, R1, R2,
                             I0, I1, I2, I3, s0, s1, s2, B, ne)
        data_r = (prof_good - model) / sig_good
        below  = np.maximum(0.0, lo_arr - p_free) / pen_sigma
        above  = np.maximum(0.0, p_free - hi_arr) / pen_sigma
        return np.append(data_r, below + above)

    lm = least_squares(_residuals_pen, p0, method='lm',
                       ftol=config['ftol'], xtol=config['xtol'],
                       gtol=config['gtol'], max_nfev=config['max_nfev'])

    p_updated = p_fixed.copy(); p_updated[free_idx] = lm.x

    t, alpha, R1, R2, I0, I1, I2, I3, s0, s1, s2, B, ne = p_updated
    model_f = _neon_model(r_good, r_max, t, alpha, R1, R2,
                          I0, I1, I2, I3, s0, s1, s2, B, ne)
    data_r  = (prof_good - model_f) / sig_good
    dof     = max(n_good - n_free, 1)
    chi2    = float(np.sum(data_r ** 2)) / dof

    def _residuals_data(p_free):
        p = p_fixed.copy(); p[free_idx] = p_free
        t, alpha, R1, R2, I0, I1, I2, I3, s0, s1, s2, B, ne = p
        model = _neon_model(r_good, r_max, t, alpha, R1, R2,
                            I0, I1, I2, I3, s0, s1, s2, B, ne)
        return (prof_good - model) / sig_good

    try:
        J        = _fd_jacobian(_residuals_data, lm.x)
        _, s, VT = np.linalg.svd(J, full_matrices=False)
        threshold = np.finfo(float).eps * max(J.shape) * s[0]
        s_inv    = np.where(s > threshold, 1.0 / s, 0.0)
        JTJ_inv  = (VT.T * s_inv**2) @ VT
        cov      = chi2 * JTJ_inv
        stderrs  = np.sqrt(np.maximum(np.diag(cov), 0.0))
        if not np.all(np.isfinite(stderrs)):
            raise np.linalg.LinAlgError("non-finite stderrs")
    except (np.linalg.LinAlgError, ValueError) as exc:
        log.warning(f"  Covariance failed: {exc}")
        stderrs = np.full(n_free, np.inf)
        cov     = np.full((n_free, n_free), np.inf)

    return p_updated, cov, stderrs, chi2, lm


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class _FringeProfile:
    profile: np.ndarray; r_grid: np.ndarray
    sigma_profile: np.ndarray; masked: np.ndarray; r_max_px: float


@dataclass
class FitResult:
    """11-free-parameter calibration fit result (sigma1=sigma2=0 fixed)."""
    t_m: float; alpha: float
    R1: float;  R2: float
    I0: float;  I1: float;  I2: float;  I3: float
    sigma0: float; sigma1: float; sigma2: float
    B: float;  ne_ratio: float

    sigma_t_m: float;    sigma_alpha: float
    sigma_R1: float;     sigma_R2: float
    sigma_I0: float;     sigma_I1: float;  sigma_I2: float;  sigma_I3: float
    sigma_sigma0: float; sigma_sigma1: float; sigma_sigma2: float
    sigma_B: float;      sigma_ne_ratio: float

    epsilon_cal: float;  sigma_epsilon_cal: float

    chi2_reduced: float; chi2_by_stage: list
    n_bins_used: int;    converged: bool


# ---------------------------------------------------------------------------
# Staged inversion
# ---------------------------------------------------------------------------

def run_staged_inversion(fp: _FringeProfile,
                         t_eff: float, alpha_init: float, eps_a: float,
                         R1_init: float = 0.53, R2_init: float = 0.53,
                         sigma0_init: float = 0.5,
                         ne_ratio_init: float = None,
                         max_nfev: int = 100_000,
                         ftol: float = 1e-14,
                         xtol: float = 1e-14,
                         gtol: float = 1e-14) -> FitResult:
    """4-stage LM: sigma1=sigma2=0 fixed. FD covariance at Stage 4."""
    good   = (~fp.masked & np.isfinite(fp.sigma_profile)
              & (fp.sigma_profile > 0) & np.isfinite(fp.profile))
    r_good = fp.r_grid[good]
    p_good = fp.profile[good]
    s_good = np.maximum(fp.sigma_profile[good].copy(), 1.0)
    r_max  = float(fp.r_max_px)
    n_good = int(good.sum())

    if n_good < 30:
        raise ValueError(f"Only {n_good} usable bins.")

    I0_init = float(np.percentile(p_good, 75))
    B_init  = float(np.percentile(p_good, 5)) * 0.8
    if ne_ratio_init is None:
        ne_ratio_init = float(NE_INTENSITY_2)

    bounds = {
        't_m':      (t_eff,         t_eff - 20e-6,     t_eff + 20e-6),
        'alpha':    (alpha_init,    alpha_init*0.95,   alpha_init*1.05),
        'R1':       (R1_init,       0.05,              0.95),
        'R2':       (R2_init,       0.05,              0.95),
        'I0':       (I0_init,       100.0,             15000.0),
        'I1':       (0.0,           -0.5,              0.5),
        'I2':       (0.0,           -0.5,              0.5),
        'I3':       (0.0,           -0.5,              0.5),
        'sigma0':   (sigma0_init,   0.01,              5.0),
        'sigma1':   (0.0,           -2.0,              2.0),
        'sigma2':   (0.0,           -2.0,              2.0),
        'B':        (B_init,        10.0,              2000.0),
        'ne_ratio': (ne_ratio_init, 0.01,              2.0),
    }

    cfg   = dict(max_nfev=max_nfev, ftol=ftol, xtol=xtol, gtol=gtol)
    p_all = np.array([bounds[n][0] for n in _NAMES], dtype=float)
    chi2_stages = []

    log.info("Stage 1 — photometric baseline")
    p_all, _, _, c1, _ = _run_stage(r_good, p_good, s_good, r_max, p_all,
                                     _STAGE_FREE[1], bounds, cfg)
    chi2_stages.append(c1); log.info(f"  χ²/ν = {c1:.3f}")

    log.info("Stage 2 — geometry + reflectivities")
    p_all, _, _, c2, _ = _run_stage(r_good, p_good, s_good, r_max, p_all,
                                     _STAGE_FREE[2], bounds, cfg)
    chi2_stages.append(c2)
    log.info(f"  χ²/ν = {c2:.3f}  R1={p_all[_IDX['R1']]:.4f}  R2={p_all[_IDX['R2']]:.4f}")

    log.info("Stage 3 — + sigma0  (sigma1=sigma2=0 fixed)")
    p_all, _, _, c3, _ = _run_stage(r_good, p_good, s_good, r_max, p_all,
                                     _STAGE_FREE[3], bounds, cfg)
    chi2_stages.append(c3)
    log.info(f"  χ²/ν = {c3:.3f}  sigma0={p_all[_IDX['sigma0']]:.4f} px")

    log.info("Stage 4 — 11 free params; FD covariance")
    p_all, cov4, se4, c4, res4 = _run_stage(r_good, p_good, s_good, r_max,
                                              p_all, _STAGE_FREE[4], bounds, cfg)
    chi2_stages.append(c4)
    log.info(f"  χ²/ν = {c4:.3f}  R1={p_all[_IDX['R1']]:.4f}  "
             f"R2={p_all[_IDX['R2']]:.4f}  ne={p_all[_IDX['ne_ratio']]:.4f}  "
             f"sigma0={p_all[_IDX['sigma0']]:.4f}")

    s4_names = _STAGE_FREE[4]
    s4_lkp   = {n: i for i, n in enumerate(s4_names)}

    def _se(name):
        return float(se4[s4_lkp[name]]) if name in s4_lkp else float('nan')

    log.info("  Stderrs:")
    for name in s4_names:
        log.info(f"    sigma_{name:12s} = {_se(name):.3e}")

    converged = bool(res4.success or res4.cost < 1e-10)
    t_f, alpha_f, R1_f, R2_f, I0_f, I1_f, I2_f, I3_f, s0_f, s1_f, s2_f, B_f, ne_f = p_all
    eps_cal       = (2.0 * t_f / NE_WAVELENGTH_1_AIR_M) % 1.0
    sigma_eps_cal = (2.0 / NE_WAVELENGTH_1_AIR_M) * _se('t_m')

    return FitResult(
        t_m=float(t_f), alpha=float(alpha_f),
        R1=float(R1_f), R2=float(R2_f),
        I0=float(I0_f), I1=float(I1_f), I2=float(I2_f), I3=float(I3_f),
        sigma0=float(s0_f), sigma1=0.0, sigma2=0.0,
        B=float(B_f), ne_ratio=float(ne_f),

        sigma_t_m=_se('t_m'),       sigma_alpha=_se('alpha'),
        sigma_R1=_se('R1'),         sigma_R2=_se('R2'),
        sigma_I0=_se('I0'),         sigma_I1=_se('I1'),    sigma_I2=_se('I2'),
        sigma_I3=_se('I3'),
        sigma_sigma0=_se('sigma0'), sigma_sigma1=float('nan'),
        sigma_sigma2=float('nan'),
        sigma_B=_se('B'),           sigma_ne_ratio=_se('ne_ratio'),

        epsilon_cal=float(eps_cal), sigma_epsilon_cal=float(sigma_eps_cal),

        chi2_reduced=float(c4), chi2_by_stage=chi2_stages,
        n_bins_used=n_good,     converged=converged,
    )


# ---------------------------------------------------------------------------
# Save calibration result
# ---------------------------------------------------------------------------

def save_cal_result(fit: FitResult, npy_path: pathlib.Path,
                    t_tolansky_mm: float, eps_a: float,
                    alpha_tolansky_init: float, r_max_px: float) -> pathlib.Path:
    """
    Save FitResult as a .npy dict alongside the input profile file.

    The output file is named:  <input_stem>_cal_result.npy
    Load it in H06 with:  np.load(path, allow_pickle=True).item()

    Note on R_refl:
      H06 uses a single R_refl for the OI 630.0 nm airglow line.
      R1 (fitted at λ₁=640.2 nm) is used because 630 nm is closer to λ₁.
      R2 and ΔR are saved for reference.

    Returns the path of the saved file.
    """
    out_path = npy_path.parent / (npy_path.stem.replace('_profile_vs_r2', '')
                                   .replace('_profile_vs_r', '') + '_cal_result.npy')

    cal_dict = {
        # ---- Fitted instrument parameters ----
        # R_refl: use R1 for H06 (OI 630 nm closer to λ₁=640.2 nm than λ₂=638.3 nm)
        't_m':         fit.t_m,
        'alpha':       fit.alpha,
        'R_refl':      fit.R1,      # ← R1 used as H06 reflectivity
        'R1':          fit.R1,      # saved for reference
        'R2':          fit.R2,      # saved for reference
        'delta_R':     fit.R2 - fit.R1,
        'I0':          fit.I0,
        'I1':          fit.I1,
        'I2':          fit.I2,
        'sigma0':      fit.sigma0,
        'sigma1':      0.0,         # fixed
        'sigma2':      0.0,         # fixed
        'B':           fit.B,
        'ne_ratio':    fit.ne_ratio,
        'epsilon_cal': fit.epsilon_cal,

        # ---- 1σ uncertainties ----
        'sigma_t_m':         fit.sigma_t_m,
        'sigma_alpha':       fit.sigma_alpha,
        'sigma_R_refl':      fit.sigma_R1,
        'sigma_R1':          fit.sigma_R1,
        'sigma_R2':          fit.sigma_R2,
        'sigma_I0':          fit.sigma_I0,
        'sigma_I1':          fit.sigma_I1,
        'sigma_I2':          fit.sigma_I2,
        'sigma_sigma0':      fit.sigma_sigma0,
        'sigma_B':           fit.sigma_B,
        'sigma_ne_ratio':    fit.sigma_ne_ratio,
        'sigma_epsilon_cal': fit.sigma_epsilon_cal,

        # ---- Fit quality ----
        'chi2_reduced':  fit.chi2_reduced,
        'chi2_by_stage': list(fit.chi2_by_stage),
        'n_bins_used':   fit.n_bins_used,
        'converged':     fit.converged,
        'quality_flags': 0,   # 0 = GOOD; non-zero reserved for H06 degraded-cal flag

        # ---- Provenance ----
        'source_file':          str(npy_path),
        't_tolansky_mm':        t_tolansky_mm,
        'eps_a':                eps_a,
        'alpha_tolansky_init':  alpha_tolansky_init,
        'r_max_px':             r_max_px,
        'date_utc':             datetime.datetime.utcnow().isoformat(),
        'script':               'H05_calibration_inversion_2026_05_12.py',
    }

    np.save(out_path, cal_dict)
    return out_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_profile(path, r_max_px):
    arr = np.load(path)
    sigma = None
    if arr.ndim == 1:
        log.warning("1D array — inferring r²")
        r2, prof = np.linspace(0.0, r_max_px**2, len(arr)), arr.astype(float)
    elif arr.ndim == 2 and arr.shape[0] == 3:
        r2, prof = arr[0].astype(float), arr[1].astype(float)
        sigma    = arr[2].astype(float)
    elif arr.ndim == 2 and arr.shape[0] == 2:
        r2, prof = arr[0].astype(float), arr[1].astype(float)
    elif arr.ndim == 2 and arr.shape[1] == 2:
        r2, prof = arr[:, 0].astype(float), arr[:, 1].astype(float)
    else:
        raise ValueError(f"Unexpected shape {arr.shape}")
    return np.sqrt(np.maximum(r2, 0.0)), prof, sigma


def _estimate_sigma(profile):
    return np.maximum(np.sqrt(np.maximum(profile, 1.0)), 1.0)


def _fmt_unc(value):
    if np.isnan(value): return "fixed"
    return f"±{value:.2e}"


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(r2_data, profile, sigma,
                r_fine, model_fine, lam1_fine, lam2_fine,
                fit: FitResult, source_name: str = "",
                source_path: str = "",
                seeds_dict: dict | None = None) -> plt.Figure:

    r2_fine       = r_fine ** 2
    model_at_data = np.interp(r2_data, r2_fine, model_fine)
    residual      = profile - model_at_data

    fig = plt.figure(figsize=(14, 11.5))
    gs  = gridspec.GridSpec(3, 1, height_ratios=[3, 1.5, 2.6],
                            hspace=0.08, top=0.90, bottom=0.03,
                            left=0.09, right=0.97)
    ax_fit = fig.add_subplot(gs[0])
    ax_res = fig.add_subplot(gs[1], sharex=ax_fit)
    ax_tbl = fig.add_subplot(gs[2])
    ax_tbl.axis("off")

    ax_fit.errorbar(r2_data, profile, yerr=sigma, fmt="none",
                    ecolor="darkorange", elinewidth=0.5, alpha=0.35, zorder=2)
    ax_fit.plot(r2_data, profile, color="darkorange", lw=0.9, alpha=0.85,
                zorder=3, label=f"Data  ({source_name})")
    ax_fit.plot(r2_fine, model_fine, color="black", lw=1.5,
                zorder=4, label="Best-fit composite  (Stage 4)")
    ax_fit.plot(r2_fine, lam1_fine, color="steelblue", lw=0.8,
                ls="--", alpha=0.65, label=f"λ₁ 640.2 nm  (R1={fit.R1:.3f})")
    ax_fit.plot(r2_fine, lam2_fine, color="firebrick", lw=0.8,
                ls="--", alpha=0.65,
                label=f"λ₂ 638.3 nm ×{fit.ne_ratio:.3f}  (R2={fit.R2:.3f})")
    ax_fit.set_ylabel("CCD signal  (ADU)", fontsize=11)
    ax_fit.legend(fontsize=8.5, loc="upper right")
    ax_fit.grid(True, alpha=0.2)
    ax_fit.tick_params(labelbottom=False)

    ax_res.axhline(0, color="black", lw=0.8, ls="--")
    ax_res.fill_between(r2_data, -sigma, sigma,
                        color="steelblue", alpha=0.22, label="±1σ")
    ax_res.plot(r2_data, residual, color="steelblue", lw=0.8,
                alpha=0.9, label="Residual (data − model)")
    ax_res.set_xlabel(r"$r^2$  (pixels², 2×2 binned)", fontsize=11)
    ax_res.set_ylabel("Residual  (ADU)", fontsize=10)
    ax_res.legend(fontsize=8.5, loc="upper right")
    ax_res.grid(True, alpha=0.2)
    ax_res.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, symmetric=True))

    _alpha_exp     = int(np.floor(np.log10(abs(fit.alpha))))
    _alpha_unc_str = f"±{fit.sigma_alpha / 10**_alpha_exp:.2g}e{_alpha_exp:+03d}"

    # Extract seed values — fall back to "—" if seeds_dict not provided
    _s = seeds_dict or {}
    _seed_t     = (f"{_s['t_eff_mm']:.7f} mm"
                   if 't_eff_mm' in _s else "—")
    _seed_alpha = (f"{_s['alpha_init']:.5e}"
                   if 'alpha_init' in _s else "—")
    _seed_R1    = (f"{_s['R1_init']:.3f}"
                   if 'R1_init' in _s else "—")
    _seed_R2    = (f"{_s['R2_init']:.3f}"
                   if 'R2_init' in _s else "—")
    _seed_ne    = (f"{_s['ne_ratio_init']:.4f}"
                   if 'ne_ratio_init' in _s else "—")
    _n_pairs = (str(int(_s['n_pairs_tolansky']))
                if 'n_pairs_tolansky' in _s else "?")
    _seed_s0    = (f"{_s['sigma0_init']:.3f} px"
                   if 'sigma0_init' in _s else "—")

    rows = [
        ("t",        _seed_t,
                     f"{fit.t_m*1e3:.7f} mm",
                     f"±{fit.sigma_t_m*1e9:.2g} nm" if not np.isnan(fit.sigma_t_m) else "fixed",
                     f"Etalon gap  [Tolansky-seeded ({_n_pairs} fringe pairs) · Group A = tight-bounded from Tolansky]"),
        ("α",        _seed_alpha,
                     f"{fit.alpha:.5e} rad/px",
                     _alpha_unc_str,
                     f"Plate scale  [Tolansky-seeded ({_n_pairs} fringe pairs) · Group A = tight-bounded from Tolansky]"),
        ("R1",       _seed_R1,
                     f"{fit.R1:.5f}",
                     f"±{fit.sigma_R1:.2g}",
                     "Reflectivity λ₁=640.2 nm  [Group B] → used as R_refl in H06"),
        ("R2",       _seed_R2,
                     f"{fit.R2:.5f}",
                     f"±{fit.sigma_R2:.2g}",
                     "Reflectivity λ₂=638.3 nm  [Group B, reference only]"),
        ("ΔR=R2−R1", "—",
                     f"{fit.R2-fit.R1:+.5f}",
                     "—",
                     "Wavelength-dependent finesse difference  (4.7σ significant)"),
        ("I₀",       "~P75 of profile",
                     f"{fit.I0:.1f} ADU",
                     f"±{fit.sigma_I0:.2g}",
                     "Mean intensity  [Group B, shared]"),
        ("I₁",       "0.0",
                     f"{fit.I1:.5f}",
                     f"±{fit.sigma_I1:.2g}",
                     "Linear vignetting  [Group B, shared]"),
        ("I₂",       "0.0",
                     f"{fit.I2:.5f}",
                     f"±{fit.sigma_I2:.2g}",
                     "Quadratic vignetting  [Group B, shared]"),
        ("I₃",       "0.0",
                     f"{fit.I3:.5f}",
                     f"±{fit.sigma_I3:.2g}",
                     "Cubic vignetting  [Group B, shared]"),
        ("σ₀",       _seed_s0,
                     f"{fit.sigma0:.4f} px",
                     f"±{fit.sigma_sigma0:.2g}",
                     "PSF base width  [Group B]  σ(r)=σ₀ (constant)"),
        ("σ₁",       "0.0 px",
                     "0.0000 px",
                     "fixed",
                     "PSF sin variation  [fixed=0; F-test p=0.998]"),
        ("σ₂",       "0.0 px",
                     "0.0000 px",
                     "fixed",
                     "PSF cos variation  [fixed=0; singular Hessian]"),
        ("B",        "~P5 × 0.8",
                     f"{fit.B:.1f} ADU",
                     f"±{fit.sigma_B:.2g}",
                     "CCD bias pedestal  [Group B]"),
        ("ne_ratio", _seed_ne,
                     f"{fit.ne_ratio:.4f}",
                     f"±{fit.sigma_ne_ratio:.2g}",
                     f"λ₂/λ₁ intensity ratio  [nominal={NE_INTENSITY_2:.2f}]"),
        ("ε_cal",    "—",
                     f"{fit.epsilon_cal:.6f}",
                     f"±{fit.sigma_epsilon_cal:.2g}",
                     "Fractional order at centre  ε_cal = (2·t / λ₁) mod 1"
                     "  where λ₁ = 640.2248 nm  (zero-wind phase reference)"),
    ]

    tbl = ax_tbl.table(cellText=rows,
        colLabels=["Param", "Initial seed", "Fitted value", "1σ", "Description"],
        cellLoc="left", loc="upper center",
        colWidths=[0.08, 0.18, 0.16, 0.11, 0.47])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.0)
    tbl.scale(1, 1.18)

    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#c8d8f0"); cell.set_text_props(fontweight="bold")
        elif row % 2 == 0: cell.set_facecolor("#f0f4ff")
        if row in (3, 4): cell.set_facecolor("#fff4c2")
        if row in (11, 12): cell.set_facecolor("#eeeeee")
        if row == 14: cell.set_facecolor("#f4fff4")

    stage_str = "  ".join(f"S{i+1}: {v:.2f}" for i, v in enumerate(fit.chi2_by_stage))
    ax_tbl.text(0.01, 0.055,
        f"χ²/ν by stage:  {stage_str}    bins used: {fit.n_bins_used}   "
        f"free params: 11  (σ₁=σ₂=0 fixed; F-test p=0.998)",
        transform=ax_tbl.transAxes, va="bottom", ha="left",
        fontsize=8.5, fontfamily="monospace", color="dimgrey")

    ax_tbl.text(0.01, 0.01,
        "Group A = t, α  seeded from Tolansky with tight bounds (±20 µm, ±5%);"
        "  fitted value refines fringe shape within those bounds.   "
        "Group B = R, I₀, I₁, I₂, σ₀, B, ne_ratio  fitted freely from fringe contrast and envelope.",
        transform=ax_tbl.transAxes, va="bottom", ha="left",
        fontsize=8.0, fontfamily="monospace", color="#444466",
        style="italic")

    fig.suptitle("WindCube FPI — Neon Calibration Fringe Inversion  "
                 "(10-param: independent R1, R2; constant PSF / Harding 2014)",
                 fontsize=12, fontweight="bold", y=0.980)
    if source_path:
        fig.text(0.5, 0.963, source_path, ha="center", va="top", fontsize=8,
                 fontfamily="monospace", color="dimgrey")
    fig.text(0.5, 0.948,
        r"$S(r)=\tilde{A}(r;\lambda_1,t,\alpha,R_1)\circledast\mathcal{G}(\sigma_0)"
        r"+n_\mathrm{ratio}\cdot\tilde{A}(r;\lambda_2,t,\alpha,R_2)"
        r"\circledast\mathcal{G}(\sigma_0)+B$"
        "\n"
        r"$I(r)=I_0[1+I_1(r/r_\mathrm{max})+I_2(r/r_\mathrm{max})^2]$"
        r"$\quad\sigma(r)=\sigma_0$  (constant blur)"
        r"$\quad[\tilde{A}=\mathrm{Airy}\times I(r)]$",
        ha="center", va="top", fontsize=8.5, color="#1a1a2e", linespacing=1.6)
    return fig


# =========================================================================
# Section F — Pipeline-level functions
# (copied from src/fpi/fpi_cal_lib_v1.2_2026-05-17.py)
# =========================================================================


@dataclass
class TolanskySeedMean:
    """Per-orbit mean of Tolansky seed quantities, averaged over N cal frames."""
    n_frames_total:          int
    n_frames_used:           int
    n_frames_rejected:       int

    d_m_mean:                float   # mean etalon gap (m)
    sigma_d_m_mean:          float   # std / sqrt(N_used), m
    two_sigma_d_m_mean:      float   # = 2 × sigma_d_m_mean

    alpha_mean:              float   # mean plate scale (rad/px)
    sigma_alpha_mean:        float
    two_sigma_alpha_mean:    float

    eps_a_mean:              float   # mean fractional order, 640.2 nm line
    sigma_eps_a_mean:        float
    two_sigma_eps_a_mean:    float

    Delta_a_mean:            float   # mean r²-spacing (px²/fringe)
    sigma_Delta_a_mean:      float
    two_sigma_Delta_a_mean:  float

    Y_B_obs_mean:            float   # mean amplitude ratio (638 / 640)


@dataclass
class OrbitCalResult:
    """
    Complete orbit instrument calibration — handoff object between layers.

    Produced by run_cal_pipeline.py (calibration layer).
    Consumed by run_airglow_inversion.py (science layer, SPEC-CAL02).
    Saved as orbit_cal_result.npy (numpy dict, allow_pickle=True).
    """
    seeds:             TolanskySeedMean
    fit:               FitResult          # H05 result — 10 free params
    source_cal_files:  list[str]
    source_dark_files: list[str]
    date_utc:          str                # ISO-8601
    pipeline_version:  str = "SPEC-CAL01 v1.1"


# ── Thin wrapper functions ────────────────────────────────────────────────────

def load_bin_frame(
    path: pathlib.Path,
    shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Load a WindCube FPI binary frame → float64 image array (H, W).

    Big-endian uint16 ('>u2'). Dimensions read from header words 0 and 1.
    Header row 0 is stripped — only image pixel rows are returned.
    Handles both 2×2 (259×276) and 1×1 (527×552) automatically.
    The `shape` argument is ignored; kept for backward compatibility only.
    """
    with open(path, "rb") as f:
        first_words = np.frombuffer(f.read(4), dtype=">u2")
    n_rows_frame = int(first_words[0])
    n_cols_frame = int(first_words[1])
    expected = n_rows_frame * n_cols_frame * 2
    actual   = pathlib.Path(path).stat().st_size
    if actual != expected:
        raise ValueError(
            f"{pathlib.Path(path).name}: file size {actual} B, "
            f"expected {expected} for {n_rows_frame}×{n_cols_frame} uint16."
        )
    raw = np.frombuffer(open(path, "rb").read(), dtype=">u2")
    return raw[n_cols_frame:].reshape(n_rows_frame - 1, n_cols_frame).astype(np.float64)


def make_master_dark(
    paths: list[pathlib.Path],
    shape: tuple[int, int] = (260, 276),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Median-stack N dark frames.

    Returns
    -------
    master_dark : float64 (H, W) — median per pixel
    dark_sigma  : float64 (H, W) — std per pixel
    """
    frames = np.stack([load_bin_frame(p, shape) for p in paths])
    return np.median(frames, axis=0), frames.std(axis=0)


def dark_subtract(
    frame: np.ndarray,
    master_dark: np.ndarray,
) -> np.ndarray:
    """Return frame.astype(float64) - master_dark (float64)."""
    return frame.astype(np.float64) - master_dark


def build_radial_profile(
    image: np.ndarray,
    cx: float,
    cy: float,
    n_bins: int = 1500,  # 8 px²/bin — required for clean Gaussian peak fits
    r_max_px: float = 110.0,
) -> FringeProfile:
    """
    Mulligan r²-binned annular reduction → 1-D radial profile.

    Wraps annular_reduce(). Returns FringeProfile with fields:
    r_grid, r2_grid, profile, sigma_profile, masked, cx, cy, r_max_px.
    sigma_cx / sigma_cy default to 0.5 px (typical centre uncertainty).
    """
    return annular_reduce(
        image, cx, cy,
        sigma_cx=0.5,
        sigma_cy=0.5,
        n_bins=n_bins,
        r_max_px=r_max_px,
    )


def fit_peaks(
    fp: FringeProfile,
    distance: int = 5,
    prominence: float = 100.0,
    fit_half_window: int = 56,
    chi2_refit_threshold: float = 15.0,
) -> list[PeakFitR2]:
    """
    Detect peaks in radial profile and Gaussian-fit each in r² domain.

    Two-pass fitting strategy:
      Pass 1 — standard _find_and_fit_peaks_r2() with n_bins=1500 sampling.
      Pass 2 — re-fit any peak with chi2_red > chi2_refit_threshold using
                improved initial guesses derived from the median sigma of the
                well-fitted inner peaks and valley-minimum background estimation.
                This robustly handles outer weak 638.3 nm peaks whose windows
                contain bleed-through tails from the adjacent 640.2 nm peak.

    Parameters
    ----------
    fit_half_window      : search radius for valley-bounded asymmetric window (bins).
    chi2_refit_threshold : peaks with chi2_red above this trigger pass 2.
    """
    from scipy.optimize import curve_fit as _curve_fit

    def _gauss(x, A, mu, sig, B):
        return A * np.exp(-0.5 * ((x - mu) / sig) ** 2) + B

    # ── Pass 1: standard fit ──────────────────────────────────────────
    peaks = _find_and_fit_peaks_r2(
        fp.r_grid,
        fp.r2_grid,
        fp.profile,
        fp.sigma_profile,
        fp.masked,
        distance=distance,
        prominence=prominence,
        fit_half_window=fit_half_window,
    )

    if len(peaks) < 2:
        return peaks

    # ── Derive robust sigma prior from well-fitted inner peaks ────────
    good_sigma = [p.width_r2_px2 for p in peaks
                  if p.fit_ok
                  and np.isfinite(p.width_r2_px2)
                  and np.isfinite(p.reduced_chi2)
                  and p.reduced_chi2 < chi2_refit_threshold]

    if len(good_sigma) < 3:
        return peaks   # insufficient reference; return pass-1 as-is

    median_sigma = float(np.median(good_sigma))
    sigma_lo     = 0.4 * median_sigma
    sigma_hi     = 2.5 * median_sigma

    # ── Pass 2: re-fit poor peaks with improved guesses ───────────────
    r2_grid  = fp.r2_grid
    profile  = fp.profile
    sig_prof = fp.sigma_profile
    good_all = ~fp.masked
    all_bin_indices = [p.peak_idx for p in peaks]
    improved = list(peaks)

    for i_pk, p in enumerate(peaks):
        needs_refit = (
            not p.fit_ok
            or not np.isfinite(p.reduced_chi2)
            or p.reduced_chi2 > chi2_refit_threshold
        )
        if not needs_refit:
            continue

        bin_idx = p.peak_idx

        if i_pk not in _R2_WINDOWS:
            continue   # no user window defined — skip refit
        _wi = _R2_WINDOWS[i_pk]; r2_lo_w, r2_hi_w = _wi[0], _wi[-1]
        lo = int(np.searchsorted(r2_grid, r2_lo_w, side="left"))
        hi = int(np.searchsorted(r2_grid, r2_hi_w, side="right")) - 1
        lo = max(0, lo)
        hi = min(len(r2_grid) - 1, hi)
        win = np.arange(lo, hi + 1)
        use = good_all[win] & np.isfinite(sig_prof[win]) & (sig_prof[win] > 0)
        win_use = win[use]

        if win_use.size < 5:
            continue

        r2_w  = r2_grid[win_use]
        p_w   = profile[win_use]
        sem_w = sig_prof[win_use]
        n     = len(p_w)
        q     = max(1, n // 4)

        # Background: minimum of the two flanks (avoids neighbour tail bias)
        B0 = float(min(np.min(p_w[:q]), np.min(p_w[-q:])))

        # Amplitude: peak value minus flank minimum
        peak_val = float(profile[bin_idx])
        A0 = max(peak_val - B0, 50.0)

        # Centre: raw detected peak position
        mu0  = float(r2_grid[bin_idx])
        sig0 = median_sigma

        p0 = [A0, mu0, sig0, B0]
        bounds = (
            [0.0,    float(r2_w[0]),  sigma_lo,  B0 - 0.3 * abs(B0)],
            [np.inf, float(r2_w[-1]), sigma_hi,  B0 + 0.3 * abs(B0) + A0],
        )

        try:
            popt, pcov = _curve_fit(
                _gauss, r2_w, p_w,
                p0=p0, sigma=sem_w, absolute_sigma=True,
                bounds=bounds, maxfev=10000,
            )
            perr  = np.sqrt(np.diag(pcov))
            n_dof = max(1, len(r2_w) - 4)
            chi2  = float(np.sum(((p_w - _gauss(r2_w, *popt)) / sem_w) ** 2))
            new_chi2 = chi2 / n_dof

            # Accept only if it improves on pass-1 or pass-1 failed outright
            if not p.fit_ok or (np.isfinite(new_chi2) and
                                 new_chi2 < p.reduced_chi2 * 0.95):
                improved[i_pk] = PeakFitR2(
                    peak_idx         = bin_idx,
                    r2_raw_px2       = p.r2_raw_px2,
                    r_raw_px         = p.r_raw_px,
                    profile_raw      = p.profile_raw,
                    r2_fit_px2       = float(popt[1]),
                    sigma_r2_fit_px2 = float(perr[1]),
                    amplitude_adu    = float(popt[0]),
                    width_r2_px2     = float(abs(popt[2])),
                    fit_ok           = True,
                    reduced_chi2     = new_chi2,
                    para_ok          = p.para_ok,
                    para_rms         = p.para_rms,
                    gauss_ok         = True,
                )
        except (RuntimeError, ValueError):
            pass  # keep pass-1 result

    return improved


def peaks_to_array(peaks: list[PeakFitR2]) -> np.ndarray:
    """
    Convert list[PeakFitR2] → float64 array (N, 10) for Tolansky input.

    Columns:
      0  peak_idx          (int, cast to float)
      1  r2_raw_px2
      2  r2_fit_px2        (NaN if fit failed)
      3  sigma_r2_fit_px2  (NaN if fit failed)
      4  r_raw_px
      5  0.0               (reserved)
      6  amplitude_adu
      7  width_r2_px2      (NaN if fit failed)
      8  reduced_chi2      (NaN if fit failed)
      9  line_id           (0.0 = 640.2 nm, 1.0 = 638.3 nm; median-amp split)
    """
    # Family assignment: strict interleaving by list position (i=0,1,2,...).
    # Even list positions are 640.2 nm (strong), odd are 638.3 nm (weak).
    # This is the physical reality — the two neon lines produce strictly
    # alternating rings in r².  A median-amplitude split fails at large r²
    # where the lines partially overlap.
    rows = []
    for i, p in enumerate(peaks):
        line_id = 0.0 if (i % 2 == 0) else 1.0
        rows.append([
            float(p.peak_idx),
            p.r2_raw_px2,
            p.r2_fit_px2,
            p.sigma_r2_fit_px2,
            p.r_raw_px,
            0.0,
            p.amplitude_adu,
            p.width_r2_px2,
            p.reduced_chi2,
            line_id,
        ])
    return np.array(rows, dtype=float)


def run_tolansky(
    peak_array: np.ndarray,
    n_pairs: int | None = None,
) -> "TolanskyResult":
    """
    Two-line Tolansky WLS analysis.

    peak_array : (N, 10) float64 from peaks_to_array().
    Wraps run_tolansky_2line().
    Key outputs: Delta_a, eps_a, alpha_mean, d_m, Y_B_obs, chi2_dof_a, chi2_dof_b.
    """
    return run_tolansky_2line(peak_array, n_pairs=n_pairs)


def average_tolansky_seeds(
    results: list,
    chi2_threshold: float = 15.0,
) -> TolanskySeedMean:
    """
    Average Tolansky seeds over N cal frames.

    Frames where chi2_dof_a > threshold OR chi2_dof_b > threshold are
    excluded with a warning printed to stdout.
    Raises ValueError if no frames pass the filter.

    chi2_threshold = 15.0 (raised from 5.0 on 2026-05-25).
    With ideal peak-fit uncertainties chi2/nu ~ 1; values up to ~15
    indicate noisy but physically consistent ring positions.  All five
    FlatSat frames agreed on t and alpha to sub-micron level despite
    chi2/nu up to ~11, confirming that 5.0 was over-aggressive and
    left N=1, making the SEM undefined.

    Computes mean and std/sqrt(N) for: d_m, alpha_mean, eps_a, Delta_a, Y_B_obs.
    All two_sigma_ fields = exactly 2 × sigma_.
    """
    kept = []
    for i, r in enumerate(results):
        if r.chi2_dof_a > chi2_threshold or r.chi2_dof_b > chi2_threshold:
            print(
                f"  WARNING: frame {i} rejected "
                f"(chi2_a={r.chi2_dof_a:.2f}, chi2_b={r.chi2_dof_b:.2f} "
                f"> {chi2_threshold})"
            )
        else:
            kept.append(r)

    if not kept:
        raise ValueError("No Tolansky frames passed chi2 filter.")

    N = len(kept)

    def _mean_sem(vals: list[float]) -> tuple[float, float]:
        a = np.array(vals, dtype=float)
        return float(a.mean()), float(a.std() / np.sqrt(N))

    d_m,  sig_d  = _mean_sem([r.d_m       for r in kept])
    al,   sig_al = _mean_sem([r.alpha_mean for r in kept])
    ep,   sig_ep = _mean_sem([r.eps_a      for r in kept])
    da,   sig_da = _mean_sem([r.Delta_a    for r in kept])
    yb = float(np.mean([r.Y_B_obs for r in kept]))

    return TolanskySeedMean(
        n_frames_total=len(results),
        n_frames_used=N,
        n_frames_rejected=len(results) - N,
        d_m_mean=d_m,
        sigma_d_m_mean=sig_d,
        two_sigma_d_m_mean=2.0 * sig_d,
        alpha_mean=al,
        sigma_alpha_mean=sig_al,
        two_sigma_alpha_mean=2.0 * sig_al,
        eps_a_mean=ep,
        sigma_eps_a_mean=sig_ep,
        two_sigma_eps_a_mean=2.0 * sig_ep,
        Delta_a_mean=da,
        sigma_Delta_a_mean=sig_da,
        two_sigma_Delta_a_mean=2.0 * sig_da,
        Y_B_obs_mean=yb,
    )


def run_h05(
    fp: FringeProfile,
    seeds: TolanskySeedMean,
    R1_init: float = 0.53,
    R2_init: float = 0.53,
    sigma0_init: float = 0.55,
) -> FitResult:
    """
    Full H05 staged Levenberg-Marquardt calibration fit.

    Builds the init vector from TolanskySeedMean (see §4 Stage 6 mapping),
    calls run_staged_inversion(), and returns FitResult.

    This is the culmination of the instrument characterisation pipeline.
    The returned FitResult is packed into OrbitCalResult by the wrapper
    and constitutes the sole input to the science inversion layer.

    Phase correction: t_eff is phase-corrected via phase_correct_gap()
    so that (2*t_eff/λ₁) % 1 == seeds.eps_a_mean exactly. This anchors
    the absolute fringe phase and eliminates the sinusoidal residuals
    that arise when the raw Tolansky gap is used as the LM seed.
    """
    t_eff = phase_correct_gap(
        seeds.d_m_mean,
        seeds.eps_a_mean,
        NE_WAVELENGTH_1_AIR_M,
    )
    return run_staged_inversion(
        fp,
        t_eff=t_eff,
        alpha_init=seeds.alpha_mean,
        eps_a=seeds.eps_a_mean,
        ne_ratio_init=seeds.Y_B_obs_mean,
        R1_init=R1_init,
        R2_init=R2_init,
        sigma0_init=sigma0_init,
    )
