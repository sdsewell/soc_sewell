"""
Module:      m03_annular_reduction_2026_04_06.py
Spec:        specs/S12_m03_annular_reduction_2026-04-06.md
Author:      Claude Code
Generated:   2026-04-06
Last tested: 2026-04-06  (10/10 tests pass)
Project:     WindCube FPI Pipeline — NCAR/HAO
Repo:        soc_sewell
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import minimize, curve_fit
from scipy.signal import find_peaks


# ---------------------------------------------------------------------------
# Dark frame construction and subtraction (Section 3)
# ---------------------------------------------------------------------------

def make_master_dark(dark_frames: list) -> np.ndarray:
    """
    Construct a master dark frame by median-combining a list of dark images.

    Parameters
    ----------
    dark_frames : list of np.ndarray
        Each array is a single dark exposure, same shape and dtype as the
        science/calibration image. Minimum 1 frame; median of 1 is itself.
        dtype is typically uint16 (raw CCD counts, Level 0).

    Returns
    -------
    np.ndarray, float64
        Master dark frame, same spatial shape as input frames.
        Returned as float64 to allow unbiased subtraction from float64 images.

    Notes
    -----
    Median combination is preferred over mean because it is robust against
    cosmic ray hits in individual dark frames. With 1–5 frames the gain
    in cosmic-ray rejection is modest, but median is correct by construction
    and costs nothing.

    If dark_frames contains exactly 1 frame, the median is that frame
    converted to float64 (no change in values).

    The master dark is NOT normalised by exposure time here. The dark
    frames must already be taken at the same exposure time as the
    science/calibration frame they will be subtracted from. This is
    guaranteed by the on-orbit operations sequence (WC-SE-0003 §5, steps
    3 and 8 acquire darks at the same cadence as science exposures).
    """
    if len(dark_frames) == 0:
        raise ValueError("dark_frames must contain at least one frame")
    stack = np.stack([f.astype(np.float64) for f in dark_frames], axis=0)
    return np.median(stack, axis=0)


def subtract_dark(
    image: np.ndarray,
    master_dark: np.ndarray,
    clip_negative: bool = True,
) -> np.ndarray:
    """
    Subtract master dark from a raw image and return a float64 result.

    Parameters
    ----------
    image : np.ndarray
        Raw CCD image (uint16 or float). Same spatial shape as master_dark.
    master_dark : np.ndarray, float64
        Master dark from make_master_dark(). Must match image shape exactly.
    clip_negative : bool
        If True (default), clip negative values to 0.0 after subtraction.
        Negative values can arise from readout noise in dark pixels; they
        are unphysical as photon counts and would bias the annular reduction
        mean toward negative values in low-signal bins. Default True.

    Returns
    -------
    np.ndarray, float64
        Dark-subtracted image. Shape identical to input.

    Raises
    ------
    ValueError
        If image.shape != master_dark.shape.

    Notes
    -----
    Conversion to float64 occurs before subtraction. The raw uint16 image
    is never modified in place.
    """
    if image.shape != master_dark.shape:
        raise ValueError(
            f"Image shape {image.shape} does not match "
            f"master dark shape {master_dark.shape}"
        )
    result = image.astype(np.float64) - master_dark
    if clip_negative:
        result = np.clip(result, 0.0, None)
    return result


# ---------------------------------------------------------------------------
# Quality flags
# ---------------------------------------------------------------------------

class QualityFlags:
    GOOD                    = 0x00
    STAGE1_FAILED           = 0x01
    SEED_DISAGREEMENT       = 0x02
    SEED_FALLBACK_GEOMETRIC = 0x04
    LOW_CONFIDENCE          = 0x08
    CENTRE_FAILED           = 0x10
    SPARSE_BINS             = 0x20
    CENTRE_JUMP             = 0x40


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CentreResult:
    cx:           float
    cy:           float
    sigma_cx:     float
    sigma_cy:     float
    two_sigma_cx: float
    two_sigma_cy: float
    cost_at_min:  float
    grid_cx:      float
    grid_cy:      float
    grid_cost:    float


@dataclass
class PeakFit:
    peak_idx:       int    # bin index in the full r_grid
    r_raw_px:       float  # r_grid at detected bin (px)
    profile_raw:    float  # profile value at detected bin (ADU)
    r_fit_px:       float  # Gaussian centroid (px); falls back to r_raw if failed
    sigma_r_fit_px: float  # 1σ uncertainty on centroid (px); nan if failed
    amplitude_adu:  float  # Gaussian amplitude above background (ADU)
    width_px:       float  # Gaussian sigma (px); nan if failed
    fit_ok:         bool


@dataclass
class FringeProfile:
    # Profile arrays — shape (n_bins,)
    profile:           np.ndarray   # mean ADU per r² bin
    sigma_profile:     np.ndarray   # 1σ SEM (np.inf for masked)
    two_sigma_profile: np.ndarray   # exactly 2 × sigma_profile (S04)
    r_grid:            np.ndarray   # bin centre radii, px
    r2_grid:           np.ndarray   # bin centre r², px²
    n_pixels:          np.ndarray   # CCD pixel count per bin
    masked:            np.ndarray   # bool, True = excluded

    # Centre (S04 uncertainty fields)
    cx:             float
    cy:             float
    sigma_cx:       float
    sigma_cy:       float
    two_sigma_cx:   float   # 2 × sigma_cx
    two_sigma_cy:   float   # 2 × sigma_cy

    # Provenance
    seed_source:   str    # 'human'|'history'|'com'|'geometric'|'provided'
    stage1_cx:     float
    stage1_cy:     float
    cost_at_min:   float
    quality_flags: int
    sparse_bins:   bool   # True if > 10% bins below min_pixels_per_bin

    # Reduction parameters
    r_min_px:    float
    r_max_px:    float
    n_bins:      int
    n_subpixels: int
    sigma_clip:  float
    image_shape: tuple

    # Peak fits — populated for calibration frames, empty for science frames
    peak_fits: list   # list[PeakFit], sorted by r_raw_px

    # Dark subtraction provenance (Section 3.6)
    dark_subtracted: bool   # True if master_dark was provided and applied
    dark_n_frames:   int    # number of dark frames combined into master (0 if none)


# ---------------------------------------------------------------------------
# Helper: Gaussian function
# ---------------------------------------------------------------------------

def _gaussian(x, A, mu, sigma, B):
    """Gaussian + constant background: A*exp(-(x-mu)²/(2σ²)) + B."""
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + B


# ---------------------------------------------------------------------------
# Variance cost function
# ---------------------------------------------------------------------------

def _variance_cost(
    cx: float,
    cy: float,
    image: np.ndarray,    # pre-clipped to 99.5th percentile
    r_min_sq: float,
    r_max_sq: float,
    n_var_bins: int,
) -> float:
    """
    Sum of per-bin intensity variance over the annular region.
    Uses np.bincount — no Python loop over bins.
    Minimum at the true optical axis regardless of fringe amplitude.
    """
    rows, cols = image.shape
    col_idx = np.arange(cols, dtype=float)
    row_idx = np.arange(rows, dtype=float)
    col_grid, row_grid = np.meshgrid(col_idx, row_idx)

    r2 = (col_grid - cx) ** 2 + (row_grid - cy) ** 2
    r2_flat  = r2.ravel()
    img_flat = image.ravel()

    mask = (r2_flat >= r_min_sq) & (r2_flat < r_max_sq)
    r2_sel  = r2_flat[mask]
    img_sel = img_flat[mask]

    if len(r2_sel) == 0:
        return 1e18

    bin_idx = np.floor(
        (r2_sel - r_min_sq) / (r_max_sq - r_min_sq) * n_var_bins
    ).astype(int)
    bin_idx = np.clip(bin_idx, 0, n_var_bins - 1)

    count   = np.bincount(bin_idx, minlength=n_var_bins)
    s1      = np.bincount(bin_idx, weights=img_sel,      minlength=n_var_bins)
    s2      = np.bincount(bin_idx, weights=img_sel ** 2, minlength=n_var_bins)

    good = count >= 2
    if not np.any(good):
        return 1e18

    n_g    = count[good].astype(float)
    mean_g = s1[good] / n_g
    var_g  = s2[good] / n_g - mean_g ** 2
    var_g  = np.maximum(var_g, 0.0)

    return float(np.sum(var_g))


# ---------------------------------------------------------------------------
# Stage 1 — Coarse centre: intensity-weighted CoM
# ---------------------------------------------------------------------------

def coarse_centre_com(
    image: np.ndarray,
    overscan_left: int    = 8,
    overscan_bottom: int  = 4,
    coarse_top_pct: float = 0.5,
    coarse_inner_frac: float = 0.45,
) -> tuple:
    """
    Intensity-weighted CoM of the brightest pixels in the inner 45% of frame.
    Returns (cx_coarse, cy_coarse) or (None, None) on failure (STAGE1_FAILED).
    """
    rows, cols = image.shape

    # Remove overscan: left columns and bottom rows
    img_crop = image[:-overscan_bottom, overscan_left:] if overscan_bottom > 0 \
               else image[:, overscan_left:]
    if img_crop.size == 0:
        return (None, None)

    crop_rows, crop_cols = img_crop.shape
    cx_mid = (crop_cols - 1) / 2.0
    cy_mid = (crop_rows - 1) / 2.0
    inner_r = coarse_inner_frac * min(crop_rows, crop_cols) / 2.0

    col_idx = np.arange(crop_cols, dtype=float)
    row_idx = np.arange(crop_rows, dtype=float)
    col_grid, row_grid = np.meshgrid(col_idx, row_idx)
    dist = np.sqrt((col_grid - cx_mid) ** 2 + (row_grid - cy_mid) ** 2)
    inner_mask = dist <= inner_r

    if not np.any(inner_mask):
        return (None, None)

    vals_inner = img_crop[inner_mask]
    threshold  = np.percentile(vals_inner, (1.0 - coarse_top_pct) * 100.0)
    bright_mask = inner_mask & (img_crop >= threshold)

    if not np.any(bright_mask):
        return (None, None)

    bright_vals = img_crop[bright_mask].astype(float)
    bright_cols = col_grid[bright_mask]
    bright_rows = row_grid[bright_mask]
    total = float(np.sum(bright_vals))

    if total <= 0:
        return (None, None)

    cx_crop = float(np.sum(bright_cols * bright_vals) / total)
    cy_crop = float(np.sum(bright_rows * bright_vals) / total)

    # Convert back to full image coordinates
    # overscan_left columns were removed from the left → add back
    # overscan_bottom rows were removed from the bottom → row index unchanged
    cx_full = cx_crop + overscan_left
    cy_full = cy_crop

    return (cx_full, cy_full)


# ---------------------------------------------------------------------------
# Seed resolution
# ---------------------------------------------------------------------------

def resolve_seed(
    cx_human: Optional[float],
    cy_human: Optional[float],
    cx_com: Optional[float],
    cy_com: Optional[float],
    cx_history: Optional[float],
    cy_history: Optional[float],
    image_size: int,
    disagreement_threshold_px: float = 5.0,
) -> tuple:
    """
    Priority: human → history → CoM → geometric centre (last resort).
    Returns (cx_seed, cy_seed, seed_source, quality_flags_partial).
    """
    flags = QualityFlags.GOOD

    # Build list of available seeds
    available = []
    if cx_human  is not None and cy_human  is not None:
        available.append(('human',   cx_human,   cy_human))
    if cx_history is not None and cy_history is not None:
        available.append(('history', cx_history, cy_history))
    if cx_com    is not None and cy_com    is not None:
        available.append(('com',     cx_com,     cy_com))

    # Check pairwise disagreement
    if len(available) >= 2:
        for i in range(len(available)):
            for j in range(i + 1, len(available)):
                dist = np.hypot(available[i][1] - available[j][1],
                                available[i][2] - available[j][2])
                if dist > disagreement_threshold_px:
                    flags |= QualityFlags.SEED_DISAGREEMENT
                    break

    # Priority order
    if cx_human  is not None and cy_human  is not None:
        return (cx_human, cy_human, 'human', flags)
    if cx_history is not None and cy_history is not None:
        return (cx_history, cy_history, 'history', flags)
    if cx_com    is not None and cy_com    is not None:
        return (cx_com, cy_com, 'com', flags)

    # Last resort: geometric centre
    ctr = (image_size - 1) / 2.0
    flags |= QualityFlags.SEED_FALLBACK_GEOMETRIC
    return (ctr, ctr, 'geometric', flags)


# ---------------------------------------------------------------------------
# Stage 2 — Two-pass azimuthal variance minimisation
# ---------------------------------------------------------------------------

def azimuthal_variance_centre(
    image: np.ndarray,    # pre-clipped to 99.5th percentile
    cx_seed: float,
    cy_seed: float,
    var_r_min_px: float  = 5.0,
    var_r_max_px: float  = None,
    var_n_bins: int      = 250,
    var_search_px: float = 15.0,
    return_grid: bool    = False,
) -> tuple:
    """
    Pass 1: coarse grid, spacing max(2.0, var_search_px/8), ±var_search_px.
    Pass 2: Nelder-Mead from Pass 1 minimum.
            xatol=0.02, fatol=1.0, maxiter=500.
    Returns (cx_fine, cy_fine, cost_min, grid_cx, grid_cy, grid_cost).
    If return_grid=True, returns an additional tuple element:
        (cx_fine, cy_fine, cost_min, grid_cx, grid_cy, grid_cost,
         all_cx, all_cy, all_cost)
    where all_cx, all_cy, all_cost are 1D arrays of every grid-search point.
    Single-pass Nelder-Mead MUST NOT be used — it fails T2.
    """
    rows, cols = image.shape
    if var_r_max_px is None:
        var_r_max_px = min(rows, cols) / 2.0

    r_min_sq = var_r_min_px  ** 2
    r_max_sq = var_r_max_px  ** 2

    # Precompute pixel coordinates for efficiency
    col_idx  = np.arange(cols, dtype=float)
    row_idx  = np.arange(rows, dtype=float)
    col_grid, row_grid = np.meshgrid(col_idx, row_idx)
    img_flat = image.ravel()

    def _fast_cost(cx, cy):
        r2_flat = ((col_grid - cx) ** 2 + (row_grid - cy) ** 2).ravel()
        mask    = (r2_flat >= r_min_sq) & (r2_flat < r_max_sq)
        r2_sel  = r2_flat[mask]
        img_sel = img_flat[mask]
        if len(r2_sel) == 0:
            return 1e18
        bin_idx = np.floor(
            (r2_sel - r_min_sq) / (r_max_sq - r_min_sq) * var_n_bins
        ).astype(int)
        bin_idx = np.clip(bin_idx, 0, var_n_bins - 1)
        count = np.bincount(bin_idx, minlength=var_n_bins)
        s1    = np.bincount(bin_idx, weights=img_sel,      minlength=var_n_bins)
        s2    = np.bincount(bin_idx, weights=img_sel ** 2, minlength=var_n_bins)
        good  = count >= 2
        if not np.any(good):
            return 1e18
        n_g   = count[good].astype(float)
        mean_g = s1[good] / n_g
        var_g  = np.maximum(s2[good] / n_g - mean_g ** 2, 0.0)
        return float(np.sum(var_g))

    # Pass 1: coarse grid
    spacing = max(2.0, var_search_px / 8.0)
    offsets = np.arange(-var_search_px, var_search_px + spacing * 0.5, spacing)

    best_cost = np.inf
    best_cx   = cx_seed
    best_cy   = cy_seed

    # Collect all grid evaluations for optional return_grid output
    all_grid_cx   = []
    all_grid_cy   = []
    all_grid_cost = []

    for dcx in offsets:
        for dcy in offsets:
            trial_cx = cx_seed + dcx
            trial_cy = cy_seed + dcy
            cost = _fast_cost(trial_cx, trial_cy)
            all_grid_cx.append(trial_cx)
            all_grid_cy.append(trial_cy)
            all_grid_cost.append(cost)
            if cost < best_cost:
                best_cost = cost
                best_cx   = trial_cx
                best_cy   = trial_cy

    grid_cx   = best_cx
    grid_cy   = best_cy
    grid_cost = best_cost

    # Pass 2: Nelder-Mead refinement from grid minimum
    result = minimize(
        lambda p: _fast_cost(p[0], p[1]),
        x0=[best_cx, best_cy],
        method="Nelder-Mead",
        options={"xatol": 0.02, "fatol": 1.0, "maxiter": 500},
    )

    cx_fine  = float(result.x[0])
    cy_fine  = float(result.x[1])
    cost_min = float(result.fun)

    if return_grid:
        return (cx_fine, cy_fine, cost_min, grid_cx, grid_cy, grid_cost,
                np.array(all_grid_cx), np.array(all_grid_cy),
                np.array(all_grid_cost))
    return (cx_fine, cy_fine, cost_min, grid_cx, grid_cy, grid_cost)


# ---------------------------------------------------------------------------
# Centre uncertainty estimation
# ---------------------------------------------------------------------------

def estimate_centre_uncertainty(
    cx: float,
    cy: float,
    cost_fn: callable,    # closure capturing image_for_cost, r bounds, n_bins
    delta_px: float = 0.5,
) -> tuple:
    """
    σ_cx = sqrt(2 / |d²C/dcx²|) from finite-difference Hessian.
    Clamped to [0.02, 5.0] px. Returns (sigma_cx, sigma_cy).
    """
    c0   = cost_fn(cx, cy)
    cpp  = cost_fn(cx + delta_px, cy)
    cmm  = cost_fn(cx - delta_px, cy)
    d2cx = (cpp - 2.0 * c0 + cmm) / (delta_px ** 2)

    cpp_y = cost_fn(cx, cy + delta_px)
    cmm_y = cost_fn(cx, cy - delta_px)
    d2cy  = (cpp_y - 2.0 * c0 + cmm_y) / (delta_px ** 2)

    def _sigma(d2):
        if d2 > 0.0:
            return float(np.clip(np.sqrt(2.0 / d2), 0.02, 5.0))
        return 5.0

    return (_sigma(d2cx), _sigma(d2cy))


# ---------------------------------------------------------------------------
# Top-level centre API
# ---------------------------------------------------------------------------

def find_centre(
    image: np.ndarray,
    cx_seed: float       = None,
    cy_seed: float       = None,
    var_r_min_px: float  = 5.0,
    var_r_max_px: float  = None,
    var_n_bins: int      = 250,
    var_search_px: float = 15.0,
) -> CentreResult:
    """
    Two-pass azimuthal variance centre finder with 99.5th-percentile
    hot-pixel clip applied to the cost function only.
    Seeds default to image geometric centre if not provided.
    """
    rows, cols = image.shape
    if cx_seed is None:
        cx_seed = (cols - 1) / 2.0
    if cy_seed is None:
        cy_seed = (rows - 1) / 2.0
    if var_r_max_px is None:
        var_r_max_px = min(rows, cols) / 2.0

    # Hot-pixel clip — ONLY for centre finding, never for annular reduce
    p99_5          = float(np.percentile(image, 99.5))
    image_for_cost = np.clip(image, None, p99_5)

    cx_fine, cy_fine, cost_min, grid_cx, grid_cy, grid_cost = azimuthal_variance_centre(
        image_for_cost, cx_seed, cy_seed,
        var_r_min_px  = var_r_min_px,
        var_r_max_px  = var_r_max_px,
        var_n_bins    = var_n_bins,
        var_search_px = var_search_px,
    )

    r_min_sq = var_r_min_px  ** 2
    r_max_sq = var_r_max_px  ** 2

    def cost_closure(cx, cy):
        return _variance_cost(cx, cy, image_for_cost, r_min_sq, r_max_sq, var_n_bins)

    sigma_cx, sigma_cy = estimate_centre_uncertainty(cx_fine, cy_fine, cost_closure)

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


# ---------------------------------------------------------------------------
# Peak finding
# ---------------------------------------------------------------------------

def _find_and_fit_peaks(
    r_grid: np.ndarray,
    profile: np.ndarray,
    sigma_profile: np.ndarray,
    masked: np.ndarray,
    distance: int       = 5,
    prominence: float   = 100.0,
    fit_half_window: int = 8,
    min_sep_px: float   = 3.0,
) -> list:
    """
    Locate peaks and fit a Gaussian to each.

    Both adaptive fixes are required to achieve 20/20 peaks on real FlatSat data:
    (1) Physics-grounded distance floor — prevents good-bin compression from
        suppressing real peaks by converting min_sep_px to a safe bin distance.
    (2) Adaptive half-window clamp — prevents window engulfment of adjacent peaks.
    """
    good_mask = ~masked & np.isfinite(sigma_profile) & np.isfinite(profile)
    good_idx  = np.where(good_mask)[0]

    if len(good_idx) < 3:
        return []

    good_profile = profile[good_idx]
    good_r       = r_grid[good_idx]

    # ---- Fix 1: physics-grounded distance floor ----
    if len(good_r) >= 2:
        median_dr_px = float(np.median(np.diff(good_r)))
    else:
        median_dr_px = 1.0

    if median_dr_px > 0:
        safe_distance = max(1, int(np.floor(min_sep_px / median_dr_px)))
    else:
        safe_distance = 1
    effective_distance = min(distance, safe_distance)

    # ---- Run find_peaks on good-bin subset ----
    peaks_in_good, _ = find_peaks(
        good_profile,
        distance=effective_distance,
        prominence=prominence,
    )

    if len(peaks_in_good) == 0:
        return []

    peak_orig_idx = good_idx[peaks_in_good]   # indices into full arrays

    results = []
    for k, (peak_orig, peak_good) in enumerate(zip(peak_orig_idx, peaks_in_good)):
        r_raw    = float(r_grid[peak_orig])
        prof_raw = float(profile[peak_orig])

        # ---- Fix 2: adaptive half-window ----
        left_sep  = int(peak_good)      if k == 0                        \
                    else int(peak_good - peaks_in_good[k - 1])
        right_sep = int(len(good_idx) - 1 - peak_good) \
                    if k == len(peaks_in_good) - 1                        \
                    else int(peaks_in_good[k + 1] - peak_good)

        adaptive_hw  = max(2, (min(left_sep, right_sep) - 1) // 2)
        effective_hw = min(fit_half_window, adaptive_hw)

        win_start = max(0,                  peak_orig - effective_hw)
        win_end   = min(len(r_grid) - 1,   peak_orig + effective_hw)

        win_r     = r_grid[win_start:win_end + 1]
        win_prof  = profile[win_start:win_end + 1]
        win_sigma = sigma_profile[win_start:win_end + 1]
        win_good  = good_mask[win_start:win_end + 1]

        if np.sum(win_good) < 4:
            results.append(PeakFit(
                peak_idx       = int(peak_orig),
                r_raw_px       = r_raw,
                profile_raw    = prof_raw,
                r_fit_px       = r_raw,
                sigma_r_fit_px = float("nan"),
                amplitude_adu  = prof_raw,
                width_px       = float("nan"),
                fit_ok         = False,
            ))
            continue

        wr = win_r[win_good]
        wp = win_prof[win_good]
        ws = win_sigma[win_good]

        # Background = 20th percentile of window (not minimum)
        B0  = float(np.percentile(wp, 20))
        A0  = float(prof_raw) - B0
        sig0 = max(median_dr_px, 0.3 * median_dr_px) if median_dr_px > 0 else 1.0
        sig_low = 0.3 * median_dr_px if median_dr_px > 0 else 0.1

        # Safe sigma weights (avoid division by zero)
        ws_safe = np.where(ws > 0, ws, 1.0)

        try:
            popt, pcov = curve_fit(
                _gaussian,
                wr, wp,
                p0=[max(A0, 1.0), r_raw, sig0, B0],
                sigma=ws_safe,
                absolute_sigma=True,
                bounds=(
                    [0.0,        float(wr[0]),  sig_low, -np.inf],
                    [np.inf,     float(wr[-1]), np.inf,   np.inf],
                ),
                maxfev=2000,
            )
            perr           = np.sqrt(np.diag(pcov))
            r_fit          = float(popt[1])
            sigma_r_fit    = float(perr[1])
            amplitude      = float(popt[0])
            width          = float(popt[2])
            fit_ok         = True
        except Exception:
            r_fit          = r_raw
            sigma_r_fit    = float("nan")
            amplitude      = max(A0, 0.0)
            width          = float("nan")
            fit_ok         = False

        results.append(PeakFit(
            peak_idx       = int(peak_orig),
            r_raw_px       = r_raw,
            profile_raw    = prof_raw,
            r_fit_px       = r_fit,
            sigma_r_fit_px = sigma_r_fit,
            amplitude_adu  = amplitude,
            width_px       = width,
            fit_ok         = fit_ok,
        ))

    results.sort(key=lambda p: p.r_raw_px)
    return results


# ---------------------------------------------------------------------------
# Annular reduction
# ---------------------------------------------------------------------------

def annular_reduce(
    image: np.ndarray,
    cx: float,
    cy: float,
    sigma_cx: float,
    sigma_cy: float,
    r_min_px: float              = 0.0,
    r_max_px: float              = 128.0,
    n_bins: int                  = 150,
    n_subpixels: int             = 1,
    sigma_clip_threshold: float  = 3.0,
    min_pixels_per_bin: int      = 3,
    bad_pixel_mask: np.ndarray   = None,
    peak_distance: int           = 5,
    peak_prominence: float       = 100.0,
    peak_fit_half_window: int    = 8,
    min_peak_sep_px: float       = 3.0,
) -> FringeProfile:
    """
    Reduce a 2D image to a 1D r²-binned profile, then find and fit peaks.

    r_max_px defaults:
      128.0 px — synthetic images (M02/M04, 256×256, centred fringe)
      110.0 px — FlatSat / flight frames (clipped outer ring at ~113 px)

    SEM denominator: N_pixels (CCD pixel count), NOT N_subpixels.
    Masked bins: sigma_profile = two_sigma_profile = np.inf.
    peak_fits: populated via _find_and_fit_peaks().
    """
    rows, cols = image.shape

    # r² bin edges and centres (r² spacing, not r spacing)
    r2_min   = r_min_px ** 2
    r2_max   = r_max_px ** 2
    r2_edges = np.linspace(r2_min, r2_max, n_bins + 1)
    r2_ctrs  = 0.5 * (r2_edges[:-1] + r2_edges[1:])
    r_grid   = np.sqrt(r2_ctrs)

    # Sub-pixel offsets for more precise r² assignment per CCD pixel
    if n_subpixels > 1:
        sub_off = (np.arange(n_subpixels) + 0.5) / n_subpixels - 0.5
    else:
        sub_off = np.array([0.0])

    col_base = np.arange(cols, dtype=float)
    row_base = np.arange(rows, dtype=float)
    col_grid_px, row_grid_px = np.meshgrid(col_base, row_base)

    # Compute mean r² per CCD pixel over sub-pixel grid
    pixel_r2 = np.zeros((rows, cols), dtype=float)
    for drow in sub_off:
        for dcol in sub_off:
            pixel_r2 += (row_grid_px + drow - cy) ** 2 \
                      + (col_grid_px + dcol - cx) ** 2
    pixel_r2 /= n_subpixels ** 2

    # Flatten and apply bad-pixel mask
    r2_flat  = pixel_r2.ravel()
    img_flat = image.ravel().astype(float)
    if bad_pixel_mask is not None:
        bad_flat = bad_pixel_mask.ravel()
    else:
        bad_flat = np.zeros(r2_flat.size, dtype=bool)

    in_range = (r2_flat >= r2_min) & (r2_flat < r2_max) & (~bad_flat)
    r2_sel   = r2_flat[in_range]
    img_sel  = img_flat[in_range]

    # Digitise into r² bins (0-based)
    bin_idx  = np.digitize(r2_sel, r2_edges) - 1
    bin_idx  = np.clip(bin_idx, 0, n_bins - 1)

    # Per-bin sigma-clip → profile and SEM
    profile_out   = np.zeros(n_bins, dtype=float)
    sigma_out     = np.full(n_bins, np.inf)
    n_pixels_out  = np.zeros(n_bins, dtype=int)
    masked_out    = np.zeros(n_bins, dtype=bool)

    for b in range(n_bins):
        sel   = (bin_idx == b)
        vals  = img_sel[sel]

        if len(vals) == 0:
            masked_out[b] = True
            continue

        # Iterative sigma clip (up to 3 passes)
        for _ in range(3):
            mn = np.mean(vals)
            sd = np.std(vals)
            if sd <= 0.0:
                break
            keep = np.abs(vals - mn) <= sigma_clip_threshold * sd
            if np.sum(keep) < min_pixels_per_bin:
                break      # keep vals from previous iteration
            vals = vals[keep]

        n_pixels_out[b] = len(vals)

        if len(vals) < min_pixels_per_bin:
            masked_out[b] = True
            continue

        mn_final = float(np.mean(vals))
        sd_final = float(np.std(vals))

        profile_out[b] = mn_final
        # SEM denominator = N_pixels (CCD count), NOT N_subpixels
        sigma_out[b]   = sd_final / np.sqrt(float(len(vals)))

    # Ensure masked bins carry inf (already initialised that way)
    two_sigma_out = 2.0 * sigma_out
    # For masked bins 2×inf = inf; explicit for clarity
    two_sigma_out[masked_out] = np.inf

    # Sparse-bin flag: > 10% of bins below min_pixels_per_bin
    n_sparse    = int(np.sum(n_pixels_out < min_pixels_per_bin))
    sparse_bins = n_sparse > 0.1 * n_bins

    # Peak finding on completed profile
    peak_fits = _find_and_fit_peaks(
        r_grid, profile_out, sigma_out, masked_out,
        distance        = peak_distance,
        prominence      = peak_prominence,
        fit_half_window = peak_fit_half_window,
        min_sep_px      = min_peak_sep_px,
    )

    return FringeProfile(
        profile           = profile_out,
        sigma_profile     = sigma_out,
        two_sigma_profile = two_sigma_out,
        r_grid            = r_grid,
        r2_grid           = r2_ctrs,
        n_pixels          = n_pixels_out,
        masked            = masked_out,
        cx                = cx,
        cy                = cy,
        sigma_cx          = sigma_cx,
        sigma_cy          = sigma_cy,
        two_sigma_cx      = 2.0 * sigma_cx,
        two_sigma_cy      = 2.0 * sigma_cy,
        seed_source       = "provided",
        stage1_cx         = float("nan"),
        stage1_cy         = float("nan"),
        cost_at_min       = float("nan"),
        quality_flags     = QualityFlags.GOOD,
        sparse_bins       = sparse_bins,
        r_min_px          = r_min_px,
        r_max_px          = r_max_px,
        n_bins            = n_bins,
        n_subpixels       = n_subpixels,
        sigma_clip        = sigma_clip_threshold,
        image_shape       = image.shape,
        peak_fits         = peak_fits,
        dark_subtracted   = False,
        dark_n_frames     = 0,
    )


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------

def reduce_calibration_frame(
    image: np.ndarray,
    master_dark: np.ndarray     = None,
    cx_human: float             = None,
    cy_human: float             = None,
    cx_history: float           = None,
    cy_history: float           = None,
    r_min_px: float             = 0.0,
    r_max_px: float             = 128.0,
    n_bins: int                 = 150,
    n_subpixels: int            = 1,
    sigma_clip_threshold: float = 3.0,
    min_pixels_per_bin: int     = 3,
    bad_pixel_mask: np.ndarray  = None,
    var_r_min_px: float         = 5.0,
    var_r_max_px: float         = None,
    var_n_bins: int             = 250,
    var_search_px: float        = 15.0,
    peak_distance: int          = 5,
    peak_prominence: float      = 100.0,
    peak_fit_half_window: int   = 8,
    min_peak_sep_px: float      = 3.0,
) -> FringeProfile:
    """
    Find centre AND reduce AND find peaks. Full calibration pipeline step.
    Dark subtraction is the first operation when master_dark is provided.
    """
    # Dark subtraction — first operation, before any other processing
    if master_dark is not None:
        image = subtract_dark(image, master_dark, clip_negative=True)
        dark_subtracted = True
        dark_n_frames   = 1   # caller sets n_frames; single master passed here
    else:
        image = image.astype(np.float64)
        dark_subtracted = False
        dark_n_frames   = 0

    rows, cols = image.shape

    # Stage 1: coarse CoM
    cx_com, cy_com = coarse_centre_com(image)
    stage1_cx = cx_com if cx_com is not None else float("nan")
    stage1_cy = cy_com if cy_com is not None else float("nan")

    quality_flags = QualityFlags.GOOD
    if cx_com is None:
        quality_flags |= QualityFlags.STAGE1_FAILED

    # Resolve seed
    cx_seed, cy_seed, seed_source, flags_partial = resolve_seed(
        cx_human  = cx_human,
        cy_human  = cy_human,
        cx_com    = cx_com,
        cy_com    = cy_com,
        cx_history = cx_history,
        cy_history = cy_history,
        image_size = min(rows, cols),
    )
    quality_flags |= flags_partial

    # Stage 2: two-pass variance minimisation (hot-pixel clip inside find_centre)
    if var_r_max_px is None:
        var_r_max_px = r_max_px

    centre = find_centre(
        image,
        cx_seed      = cx_seed,
        cy_seed      = cy_seed,
        var_r_min_px = var_r_min_px,
        var_r_max_px = var_r_max_px,
        var_n_bins   = var_n_bins,
        var_search_px = var_search_px,
    )

    # Annular reduction + peak finding (uses full unclipped image)
    fp = annular_reduce(
        image,
        cx                   = centre.cx,
        cy                   = centre.cy,
        sigma_cx             = centre.sigma_cx,
        sigma_cy             = centre.sigma_cy,
        r_min_px             = r_min_px,
        r_max_px             = r_max_px,
        n_bins               = n_bins,
        n_subpixels          = n_subpixels,
        sigma_clip_threshold = sigma_clip_threshold,
        min_pixels_per_bin   = min_pixels_per_bin,
        bad_pixel_mask       = bad_pixel_mask,
        peak_distance        = peak_distance,
        peak_prominence      = peak_prominence,
        peak_fit_half_window = peak_fit_half_window,
        min_peak_sep_px      = min_peak_sep_px,
    )

    # Fill provenance fields
    fp.seed_source      = seed_source
    fp.stage1_cx        = stage1_cx
    fp.stage1_cy        = stage1_cy
    fp.cost_at_min      = centre.cost_at_min
    fp.quality_flags    = quality_flags
    fp.dark_subtracted  = dark_subtracted
    fp.dark_n_frames    = dark_n_frames
    if fp.sparse_bins:
        fp.quality_flags |= QualityFlags.SPARSE_BINS

    return fp


def reduce_science_frame(
    image: np.ndarray,
    master_dark: np.ndarray     = None,
    cx: float                   = None,
    cy: float                   = None,
    sigma_cx: float             = 0.1,
    sigma_cy: float             = 0.1,
    r_min_px: float             = 0.0,
    r_max_px: float             = 128.0,
    n_bins: int                 = 150,
    n_subpixels: int            = 1,
    sigma_clip_threshold: float = 3.0,
    min_pixels_per_bin: int     = 3,
    bad_pixel_mask: np.ndarray  = None,
) -> FringeProfile:
    """
    Uses provided centre, reduces only. No peak finding. peak_fits = [].
    seed_source is set to 'provided'.
    Dark subtraction is the first operation when master_dark is provided.
    """
    # Dark subtraction — first operation, before any other processing
    if master_dark is not None:
        image = subtract_dark(image, master_dark, clip_negative=True)
        dark_subtracted = True
        dark_n_frames   = 1
    else:
        image = image.astype(np.float64)
        dark_subtracted = False
        dark_n_frames   = 0

    fp = annular_reduce(
        image,
        cx                   = cx,
        cy                   = cy,
        sigma_cx             = sigma_cx,
        sigma_cy             = sigma_cy,
        r_min_px             = r_min_px,
        r_max_px             = r_max_px,
        n_bins               = n_bins,
        n_subpixels          = n_subpixels,
        sigma_clip_threshold = sigma_clip_threshold,
        min_pixels_per_bin   = min_pixels_per_bin,
        bad_pixel_mask       = bad_pixel_mask,
        peak_distance        = 5,
        peak_prominence      = 1e18,   # effectively disables peak finding
        peak_fit_half_window = 8,
        min_peak_sep_px      = 3.0,
    )

    # Science frame: no peaks, seed_source = 'provided'
    fp.seed_source     = "provided"
    fp.peak_fits       = []
    fp.dark_subtracted = dark_subtracted
    fp.dark_n_frames   = dark_n_frames

    return fp
