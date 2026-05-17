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
import pathlib
import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit, minimize
from scipy.signal import find_peaks

from windcube.constants import (
    ETALON_GAP_M,
    ETALON_N,
    ETALON_R_INSTRUMENT,
    ALPHA_RAD_PX,
    R_MAX_PX,
)
from src.constants import (
    OI_WAVELENGTH_AIR_M,
    NE_WAVELENGTH_1_AIR_M,
    NE_WAVELENGTH_2_AIR_M,
    NE_INTENSITY_1,
    NE_INTENSITY_2,
    SPEED_OF_LIGHT_MS,
)


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
    t:       float = 20.1070707e-3  # gap, metres; authoritative Tolansky result (Z01a 2026-04-21)
    R_refl:  float = 0.53        # effective reflectivity (FlatSat cal)
    n:       float = 1.0         # refractive index (air gap)
    alpha:   float = 1.6084e-4   # rad/pixel, 2×2 binned (Tolansky 2026)

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
) -> np.ndarray:
    """
    Quadratic intensity envelope accounting for optical vignetting.

    I(r) = I₀ · (1 + I₁·(r/r_max) + I₂·(r/r_max)²)    (Harding Eq. 4)

    The envelope must be positive everywhere for physically valid inputs.
    Caller is responsible for choosing I1, I2 such that I(r) > 0 for
    all r in [0, r_max].

    Returns
    -------
    I : intensity in counts, same shape as r
    """
    rn = r / r_max
    return I0 * (1.0 + I1 * rn + I2 * rn ** 2)


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
    theta = theta_from_r(r, alpha)
    I_env = intensity_envelope(r, r_max, I0, I1, I2)
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
    A_ideal = airy_ideal(r, wavelength, t, R_refl, alpha, n, r_max, I0, I1, I2)
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


def _gaussian(r: np.ndarray, A: float, mu: float, sig: float, B: float) -> np.ndarray:
    """Gaussian with flat background: A*exp(-0.5*((r-mu)/sig)^2) + B."""
    return A * np.exp(-0.5 * ((r - mu) / sig) ** 2) + B


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
        left_sep  = (bin_idx - all_bin_indices[peak_pos - 1]) if peak_pos > 0                          else 9999
        right_sep = (all_bin_indices[peak_pos + 1] - bin_idx) if peak_pos < len(all_bin_indices) - 1  else 9999
        nearest   = min(left_sep, right_sep)
        adaptive_hw = max(2, (nearest - 1) // 2)
        effective_hw = min(fit_half_window, adaptive_hw)

        lo      = max(0, bin_idx - effective_hw)
        hi      = min(len(r_grid) - 1, bin_idx + effective_hw)
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
    r_grid:          np.ndarray,
    r2_grid:         np.ndarray,
    profile:         np.ndarray,
    sigma_profile:   np.ndarray,
    masked:          np.ndarray,
    distance:        int   = 5,
    prominence:      float = 100.0,
    fit_half_window: int   = 6,
    min_sep_px:      float = 3.0,
) -> list[PeakFitR2]:
    """
    Locate peaks in the radial profile and fit a Gaussian to each in r² space.

    Identical peak detection to _find_and_fit_peaks; differs only in the
    fitting step, which uses r2_grid as the x-axis.  Fabry-Pérot fringes are
    expected to be evenly spaced in r², so a Gaussian centroid in r² gives a
    more physically motivated peak position for calibration.

    Parameters mirror _find_and_fit_peaks.  min_sep_px is in pixels and is
    converted to safe_distance bins via r_grid spacing.
    """
    good         = ~masked
    good_indices = np.where(good)[0]
    profile_good = profile[good]

    if profile_good.size == 0:
        return []

    # Safe distance from min_sep_px using r_grid (same as r-domain fitting)
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

    # r² bin spacing for initial guess and bounds scaling
    if good_indices.size > 1:
        median_dr2_px2 = float(np.median(np.diff(r2_grid[good])))
        if median_dr2_px2 <= 0.0:
            median_dr2_px2 = 1.0
    else:
        median_dr2_px2 = 1.0

    effective_distance = min(distance, safe_distance)
    peaks_sub, _ = find_peaks(profile_good, distance=effective_distance, prominence=prominence)
    all_bin_indices = [int(good_indices[s]) for s in peaks_sub]

    results: list[PeakFitR2] = []
    for peak_pos, (sub_idx, bin_idx) in enumerate(zip(peaks_sub, all_bin_indices)):
        left_sep  = (bin_idx - all_bin_indices[peak_pos - 1]) if peak_pos > 0                          else 9999
        right_sep = (all_bin_indices[peak_pos + 1] - bin_idx) if peak_pos < len(all_bin_indices) - 1  else 9999
        nearest      = min(left_sep, right_sep)
        adaptive_hw  = max(2, (nearest - 1) // 2)
        effective_hw = min(fit_half_window, adaptive_hw)

        lo      = max(0, bin_idx - effective_hw)
        hi      = min(len(r2_grid) - 1, bin_idx + effective_hw)
        win     = np.arange(lo, hi + 1)
        usable  = ~masked[win] & np.isfinite(sigma_profile[win])
        win_use = win[usable]

        r2_fit_px2       = float(r2_grid[bin_idx])   # fallback if fit fails
        sigma_r2_fit_px2 = np.nan
        amplitude_adu    = float(profile[bin_idx])
        width_r2_px2     = np.nan
        reduced_chi2     = np.nan
        fit_ok           = False

        if win_use.size >= 4:
            r2_w  = r2_grid[win_use]
            p_w   = profile[win_use]
            sem_w = sigma_profile[win_use]

            B0   = float(np.percentile(p_w, 20))
            A0   = max(float(profile[bin_idx]) - B0, 1.0)
            mu0  = float(r2_grid[bin_idx])
            sig0 = max((float(r2_w[-1]) - float(r2_w[0])) / 6.0, median_dr2_px2 * 0.5)
            p0   = [A0, mu0, sig0, B0]
            bounds = (
                [0.0,    float(r2_w[0]),  0.3 * median_dr2_px2,    -np.inf],
                [np.inf, float(r2_w[-1]), float(r2_w[-1]) - float(r2_w[0]), np.inf],
            )
            try:
                popt, pcov = curve_fit(
                    _gaussian, r2_w, p_w,
                    p0=p0, sigma=sem_w, absolute_sigma=True,
                    bounds=bounds, maxfev=5000,
                )
                perr             = np.sqrt(np.diag(pcov))
                r2_fit_px2       = float(popt[1])
                sigma_r2_fit_px2 = float(perr[1])
                amplitude_adu    = float(popt[0])
                width_r2_px2     = float(abs(popt[2]))
                n_dof            = len(r2_w) - 4
                if n_dof > 0:
                    chi2         = float(np.sum(((p_w - _gaussian(r2_w, *popt)) / sem_w) ** 2))
                    reduced_chi2 = chi2 / n_dof
                fit_ok           = True
            except (RuntimeError, ValueError):
                pass

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
    peak_fit_half_window: int = 40,  # upper bound; adaptive clamp controls effective value
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
    peak_fit_half_window: half-width of Gaussian fitting window per peak (bins)
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
    n_cols: int = 5,
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

    all_bin_indices = [pf.peak_idx for pf in peaks]

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.2 * n_cols, 3.6 * n_rows),
        squeeze=False,
    )
    fig.suptitle(
        f"All-Fringe Gaussian Fit Diagnostics (r² domain)  —  {n_peaks} peaks  |  "
        f"median r² bin width = {median_dr2_px2:.3f} px²  |  "
        f"fit_half_window = {fit_half_window} (adaptive clamp applied per peak)",
        fontsize=10, fontweight="bold",
    )

    for k, pf in enumerate(peaks):
        row, col = divmod(k, n_cols)
        ax = axes[row, col]

        bin_idx   = all_bin_indices[k]
        left_sep  = (bin_idx - all_bin_indices[k - 1]) if k > 0           else 9999
        right_sep = (all_bin_indices[k + 1] - bin_idx) if k < n_peaks - 1 else 9999
        nearest      = min(left_sep, right_sep)
        adaptive_hw  = max(2, (nearest - 1) // 2)
        effective_hw = min(fit_half_window, adaptive_hw)

        lo      = max(0, bin_idx - effective_hw)
        hi      = min(len(fp.r2_grid) - 1, bin_idx + effective_hw)
        win     = np.arange(lo, hi + 1)
        usable  = ~fp.masked[win] & np.isfinite(fp.sigma_profile[win])
        win_use = win[usable]

        r2_w  = fp.r2_grid[win_use]
        p_w   = fp.profile[win_use]
        sem_w = fp.sigma_profile[win_use]

        # Initial guess — same formulas as _find_and_fit_peaks_r2
        B0   = float(np.percentile(p_w, 20)) if len(p_w) > 0 else 0.0
        A0   = max(float(fp.profile[bin_idx]) - B0, 1.0)
        mu0  = float(fp.r2_grid[bin_idx])
        sig0 = max((float(r2_w[-1]) - float(r2_w[0])) / 6.0,
                   median_dr2_px2 * 0.5) if len(r2_w) > 1 else median_dr2_px2
        p0   = [A0, mu0, sig0, B0]

        bounds_lo = [0.0,    float(r2_w[0])  if len(r2_w) else mu0 - 1,
                     0.3 * median_dr2_px2,                                    0.0   ]
        bounds_hi = [np.inf, float(r2_w[-1]) if len(r2_w) else mu0 + 1,
                     float(r2_w[-1]) - float(r2_w[0]) if len(r2_w) > 1
                     else median_dr2_px2 * 4,                                 np.inf]

        fit_ok = False
        popt   = p0[:]
        perr   = [np.nan] * 4
        mesg   = f"n_usable = {win_use.size} < 4"
        reduced_chi2 = np.nan
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
                n_dof  = len(r2_w) - 4
                if n_dof > 0:
                    chi2         = float(np.sum(
                        ((p_w - _gaussian(r2_w, *popt)) / sem_w) ** 2
                    ))
                    reduced_chi2 = chi2 / n_dof
            except RuntimeError as exc:
                mesg = f"RuntimeError: {str(exc)[:55]}"
            except ValueError as exc:
                mesg = f"ValueError: {str(exc)[:55]}"

        r2_lo_plot = r2_w[0]  if len(r2_w) else mu0 - 2 * median_dr2_px2
        r2_hi_plot = r2_w[-1] if len(r2_w) else mu0 + 2 * median_dr2_px2
        r2_fine = np.linspace(r2_lo_plot, r2_hi_plot, 300)
        y_init  = _gaussian(r2_fine, *p0)

        # ── Plot ──────────────────────────────────────────────────────────────
        ax.set_facecolor("#F0FFF4" if fit_ok else "#FFF0F0")

        if win_use.size > 0:
            ax.errorbar(r2_w, p_w, yerr=sem_w,
                        fmt="o", color="steelblue", markersize=4,
                        ecolor="cornflowerblue", elinewidth=1.0, capsize=2,
                        zorder=3)
        ax.plot(r2_fine, y_init, color="goldenrod", lw=1.2, ls="--", zorder=2)
        if fit_ok:
            y_fit = _gaussian(r2_fine, *popt)
            ax.plot(r2_fine, y_fit, color="crimson", lw=1.8, zorder=4)
            # Red band = ±1σ uncertainty on the r²-domain Gaussian centroid μ
            ax.axvspan(popt[1] - perr[1], popt[1] + perr[1],
                       alpha=0.15, color="crimson")

        ax.axvline(pf.r2_raw_px2, color="darkorange", lw=0.9, ls="--", alpha=0.8)
        if fit_ok:
            ax.axvline(popt[1], color="crimson", lw=1.0, ls="-", alpha=0.9)

        # ── Title ─────────────────────────────────────────────────────────────
        lam = "640.2" if (k + 1) % 2 == 1 else "638.3"
        if fit_ok:
            r_derived = float(np.sqrt(popt[1])) if popt[1] > 0 else float("nan")
            title = (
                f"P{k+1} · {lam} nm  ·  hw={effective_hw}\n"
                f"r²={popt[1]:.1f} ± {perr[1]:.1f} px²   r={r_derived:.3f} px   χ²={reduced_chi2:.2f}"
            )
            title_color = "#1a6e2e"
        else:
            title = (
                f"P{k+1} · {lam} nm  ·  hw={effective_hw}  FAILED\n"
                f"{mesg[:48]}"
            )
            title_color = "#b22222"

        ax.set_title(title, fontsize=7.5, color=title_color)
        ax.tick_params(labelsize=6.5)
        ax.set_xlabel("r² [px²]", fontsize=7)
        ax.set_ylabel("ADU", fontsize=7)

    for idx in range(n_peaks, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r, c].axis("off")

    fig.tight_layout()


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
