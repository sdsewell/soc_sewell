"""
annular_reduction.py — Mulligan (1986) r²-binned annular reduction.

Requires a pre-determined fringe centre produced by center_finder.py
(saved as 2027-01-01T00-00-00Z_science_ROI_L1.1_center.npz).  No centre finding is performed here.

Pipeline:
  1. Select image ROI (e.g. 2027-01-01T00-00-00Z_science_ROI_L1.1.npy) produced by load_image.py
  2. Select corresponding _center.npz (cx, cy, sigma_cx, sigma_cy).
  3. Annular reduction → 1-D radial intensity profile (FringeProfile).
     Peak finding and Gaussian fits are performed here, on the profile
     and SEM arrays as soon as they are available.
  4. Plot: image with centre | radial profile with labelled peaks | peak fit table.

Outputs (all saved alongside the input .npy, sharing its stem)
--------------------------------------------------------------
  <stem>_L1.2.npz
      Merged NumPy archive containing all radial profile fields and the
      fringe centre.  Required by M05 (calibration inversion).
      Keys:
        r_grid        (n_bins,)  — bin-centre radii, px
        r2_grid       (n_bins,)  — bin-centre radii squared, px²
        profile       (n_bins,)  — mean intensity per bin, ADU
        sigma_profile (n_bins,)  — SEM per bin, ADU (np.inf for masked bins)
        masked        (n_bins,)  — bool, True = bin excluded from fitting
        cx, cy                   — fringe centre, px
        sigma_cx, sigma_cy       — 1-sigma centre uncertainties, px

  <stem>_profile_vs_r.npy
      2-D float64 array, shape (2, n_bins).
        row 0 : r_grid  (px)  — bin-centre radii
        row 1 : profile (ADU) — mean intensity per bin
      Designed for direct loading by p02_synthetic_cal_spectrum; compatible
      with the (2, N) branch of that script's array-shape handling.

  <stem>_profile_vs_r2.npy
      2-D float64 array, shape (3, n_bins).
        row 0 : r2_grid       (px²) — bin-centre radii squared
        row 1 : profile       (ADU) — mean intensity per bin
        row 2 : sigma_profile (ADU) — SEM per bin (np.inf for masked/sparse bins)
      x-axis is r² for Tolansky/H05 calibration inversion analysis.

  <stem>_peak_fits.npy
      2-D float64 array, one row per detected fringe peak.  9 columns:
        col 0 : peak_num
        col 1 : r_raw (px)          — detected bin centre (find_peaks)
        col 2 : r_fit (px)          — TRF Gaussian centroid μ
        col 3 : sigma_r_fit (px)    — 1-sigma uncertainty on μ
        col 4 : r_fit (px²)         — μ², for use in r²-domain calibration
        col 5 : sigma_r_fit (px²)   — 2·μ·σ_μ  (propagated uncertainty)
        col 6 : amplitude (ADU)     — Gaussian amplitude A above background
        col 7 : width_sigma (px)    — Gaussian width σ
        col 8 : reduced_chi2        — χ²/(n_points − 4); see note below
      Cols 2–8 are NaN when the Gaussian fit failed for that peak.

  Reduced chi-squared interpretation
      A value near 1 indicates that the fit residuals are consistent with
      the per-bin SEM weights — the Gaussian model is a good description of
      the fringe shape.  A value significantly greater than 1 indicates
      either a poor fit (wrong model, window too wide/narrow, or a biased
      initial guess) or underestimated SEM.  A value much less than 1
      suggests the SEM is overestimated or the window contains too few
      degrees of freedom (n_points − 4 ≤ 0 gives NaN).

References:
  Harding et al. (2014) Section 3
  Niciejewski et al. (1992) SPIE 1745
  Mulligan (1986) J. Phys. E 19, 545

n_bins parameter note
---------------------
n_bins controls the r²-bin resolution and can be increased freely without
affecting the equal-area Mulligan technique — it is purely a sampling
parameter.  With n_bins=1500 and r_max=110 px each bin spans
dr² = 110²/1500 ≈ 8.1 px², giving a radial bin width of dr ≈ 0.16 px at
the first fringe (~25 px), which is sufficient to resolve narrow fringes
accurately.  Two bin-unit parameters must be scaled in proportion whenever
n_bins is changed:

  peak_distance     (bins) — minimum peak separation passed to find_peaks.
                    Scale with n_bins so it continues to represent the same
                    physical separation.  At n_bins=1500 use peak_distance=50
                    (equivalent to the original peak_distance=5 at n_bins=150).

  peak_fit_half_window (bins) — half-width of the Gaussian fitting window.
                    Must be chosen so the window covers the fringe without
                    extending into the zero-count centre region, which biases
                    the Gaussian centroid inward.  At n_bins=1500 use
                    peak_fit_half_window=40 (≈ ±6 px at the first fringe,
                    ±0.5 px at r=100 px — the adaptive clamp keeps the window
                    away from adjacent peaks at all radii).

All pixel-unit parameters (peak_prominence, min_peak_sep_px, r_min_px,
r_max_px, sigma_clip_threshold, etc.) are unaffected by changes to n_bins.

Usage
-----
    python ingest/annular_reduction.py
"""

from __future__ import annotations

import os
import pathlib
import tkinter as tk
from dataclasses import dataclass, field
from tkinter import filedialog
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


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
                [0.0,    float(r2_w[0]),  0.3 * median_dr2_px2,                     0.0   ],
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
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Load a L1.1 .npy image and its _centre.npz from center_finder.py,
    run annular reduction (includes peak finding), save outputs, and plot.
    """
    npy_dir = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     r"..\raw_images_with_metadata")
    )

    root = tk.Tk()
    root.withdraw()
    npy_file = filedialog.askopenfilename(
        title="Select L1.1 numpy array (.npy)",
        initialdir=npy_dir,
        filetypes=[("NumPy array", "*.npy"), ("All files", "*.*")],
    )
    if not npy_file:
        root.destroy()
        print("No image file selected — exiting.")
        return

    centre_file = filedialog.askopenfilename(
        title="Select centre file (cal_image_centre.npz) from center_finder",
        initialdir=os.path.dirname(npy_file),
        filetypes=[("NumPy archive", "*.npz"), ("All files", "*.*")],
    )
    root.destroy()
    if not centre_file:
        print("No centre file selected — exiting.")
        return

    src    = pathlib.Path(npy_file)
    image  = np.load(src)
    print(f"Image  : {src.name}")
    print(f"Shape  : {image.shape}  dtype: {image.dtype}")
    print(f"Range  : {image.min()} - {image.max()}  ADU")

    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D array, got shape {image.shape}")

    cdata = np.load(centre_file)
    available_keys = list(cdata.keys())
    print(f"Centre file keys: {available_keys}")
    required = {"cx", "cy", "sigma_cx", "sigma_cy"}
    missing  = required - set(available_keys)
    if missing:
        raise KeyError(
            f"Centre file '{pathlib.Path(centre_file).name}' is missing keys: {missing}\n"
            f"  Available keys: {available_keys}\n"
            f"  Make sure you selected the file saved by center_finder.py "
            f"(cal_image_centre.npz), not a different .npz."
        )
    cx       = float(cdata["cx"])
    cy       = float(cdata["cy"])
    sigma_cx = float(cdata["sigma_cx"])
    sigma_cy = float(cdata["sigma_cy"])
    print(f"\nCentre : cx = {cx:.3f} +/- {sigma_cx:.3f} px,  "
          f"cy = {cy:.3f} +/- {sigma_cy:.3f} px  "
          f"(from {pathlib.Path(centre_file).name})")

    print("\nRunning annular reduction ...")
    fp = annular_reduce(image, cx, cy, sigma_cx, sigma_cy)

    good_bins = int((~fp.masked).sum())
    print(f"Bins   : {fp.n_bins} total,  {good_bins} good,  "
          f"{fp.n_bins - good_bins} masked")
    if fp.sparse_bins:
        print("  WARNING: > 10 % of bins are sparse or masked")

    # -- Save peak fits — one row per detected peak ----------------------------
    peaks_path = src.with_name(src.stem + "_peak_fits.npy")
    if fp.peak_fits:
        peaks_array = np.array([
            [i + 1,
             pf.r_raw_px,
             pf.r_fit_px           if pf.fit_ok else np.nan,
             pf.sigma_r_fit_px     if pf.fit_ok else np.nan,
             pf.r_fit_px ** 2      if pf.fit_ok else np.nan,
             2.0 * pf.r_fit_px * pf.sigma_r_fit_px if pf.fit_ok else np.nan,
             pf.amplitude_adu,
             pf.width_px           if pf.fit_ok else np.nan,
             pf.reduced_chi2       if pf.fit_ok else np.nan]
            for i, pf in enumerate(fp.peak_fits)
        ], dtype=np.float64)
    else:
        peaks_array = np.empty((0, 9), dtype=np.float64)
    np.save(peaks_path, peaks_array)
    print(f"Peaks saved: {peaks_path}")
    print(f"  columns  : peak_num | r_raw_px | r_fit_px | sigma_r_fit_px "
          f"| r_fit_sq | sigma_r_fit_sq | amplitude_adu | width_px | reduced_chi2")
    print(f"  rows     : {peaks_array.shape[0]} peak(s)")

    # -- Save r²-domain peak fits -----------------------------------------------
    peaks_r2_path = src.with_name(src.stem + "_peak_fits_r2.npy")
    if fp.peak_fits_r2:
        peaks_r2_array = np.array([
            [i + 1,
             pf.r2_raw_px2,
             pf.r2_fit_px2            if pf.fit_ok else np.nan,
             pf.sigma_r2_fit_px2      if pf.fit_ok else np.nan,
             float(np.sqrt(pf.r2_fit_px2))
                 if pf.fit_ok and pf.r2_fit_px2 > 0 else np.nan,
             pf.sigma_r2_fit_px2 / (2.0 * float(np.sqrt(pf.r2_fit_px2)))
                 if pf.fit_ok and pf.r2_fit_px2 > 0 else np.nan,
             pf.amplitude_adu,
             pf.width_r2_px2          if pf.fit_ok else np.nan,
             pf.reduced_chi2          if pf.fit_ok else np.nan]
            for i, pf in enumerate(fp.peak_fits_r2)
        ], dtype=np.float64)
    else:
        peaks_r2_array = np.empty((0, 9), dtype=np.float64)
    np.save(peaks_r2_path, peaks_r2_array)
    print(f"r² peaks saved: {peaks_r2_path}")
    print(f"  columns  : peak_num | r2_raw_px2 | r2_fit_px2 | sigma_r2_fit_px2 "
          f"| r_fit_derived | sigma_r_derived | amplitude_adu | width_r2_px2 | reduced_chi2")
    print(f"  rows     : {peaks_r2_array.shape[0]} peak(s)")

    # -- Save merged L1.2 — all radial profile fields + centre in one archive --
    l12_path = src.with_name(src.stem + "_L1.2.npz")
    np.savez(
        l12_path,
        r_grid        = fp.r_grid,
        r2_grid       = fp.r2_grid,
        profile       = fp.profile,
        sigma_profile = fp.sigma_profile,
        masked        = fp.masked,
        cx            = np.array(cx),
        cy            = np.array(cy),
        sigma_cx      = np.array(sigma_cx),
        sigma_cy      = np.array(sigma_cy),
    )
    print(f"L1.2 saved : {l12_path}")

    # -- Save standalone profile arrays for easy loading by p02 ---------------
    profile_r_path = src.with_name(src.stem + "_profile_vs_r.npy")
    np.save(profile_r_path,
            np.array([fp.r_grid, fp.profile], dtype=np.float64))
    print(f"Profile vs r   saved : {profile_r_path}  shape (2, {fp.n_bins})")

    profile_r2_path = src.with_name(src.stem + "_profile_vs_r2.npy")
    np.save(profile_r2_path,
            np.array([fp.r2_grid, fp.profile, fp.sigma_profile], dtype=np.float64))
    print(f"Profile vs r²  saved : {profile_r2_path}  shape (3, {fp.n_bins})")

    # -- Plotting --------------------------------------------------------------
    good   = ~fp.masked
    finite = good & np.isfinite(fp.sigma_profile)
    vlo    = float(np.percentile(image,  1))
    vhi    = float(np.percentile(image, 99))
    hdr_bg = "#2C3E50"
    alt_bg = "#EBF5FB"

    # ── Figure: Annular Reduction (r² domain) ────────────────────────────────
    n_peaks_r2 = len(fp.peak_fits_r2)
    table_h_r2 = max(1.5, 0.32 * (n_peaks_r2 + 2))
    fig2 = plt.figure(figsize=(14, 10 + table_h_r2))
    gs2  = fig2.add_gridspec(3, 1, hspace=0.40, height_ratios=[4, 4, table_h_r2])

    # Top panel — same image with centre overlaid
    ax0b = fig2.add_subplot(gs2[0])
    ax0b.imshow(image, cmap="gray", origin="lower", vmin=vlo, vmax=vhi, aspect="equal")
    ax0b.axhline(cy, color="cyan", linewidth=0.8, linestyle="--", alpha=0.9)
    ax0b.axvline(cx, color="cyan", linewidth=0.8, linestyle="--", alpha=0.9)
    ax0b.plot(cx, cy, "+", color="yellow", markersize=14, markeredgewidth=1.5)
    ax0b.set_title(
        f"Fine Center Determination from Nelder-Mead:  "
        f"cx = {cx:.3f} +/- {sigma_cx:.3f} px,  cy = {cy:.3f} +/- {sigma_cy:.3f} px",
        fontsize=9,
    )
    ax0b.set_xlabel("Column (pixel)", fontsize=8)
    ax0b.set_ylabel("Row (pixel)",    fontsize=8)
    ax0b.tick_params(labelsize=7)

    # Middle panel — mean intensity vs r²
    ax1b = fig2.add_subplot(gs2[1])
    ax1b.errorbar(
        fp.r2_grid[finite], fp.profile[finite],
        yerr=fp.two_sigma_profile[finite],
        fmt="none", ecolor="navy", alpha=0.45, linewidth=0.9,
        label="+/-2 sigma SEM",
    )
    ax1b.errorbar(
        fp.r2_grid[finite], fp.profile[finite],
        yerr=fp.sigma_profile[finite],
        fmt="none", ecolor="darkblue", alpha=0.85, linewidth=1.8,
        label="+/-1 sigma SEM  (fit weight)",
    )
    ax1b.plot(fp.r2_grid[good], fp.profile[good],
              color="steelblue", linewidth=1.0,
              marker=".", markersize=10, markerfacecolor="steelblue",
              markeredgewidth=0, label="Mean ADU")
    if fp.masked.any():
        ax1b.plot(fp.r2_grid[fp.masked], fp.profile[fp.masked],
                  "rx", markersize=4, label="Masked bins")

    for i, pf in enumerate(fp.peak_fits_r2):
        ax1b.axvline(pf.r2_raw_px2, color="darkorange", linewidth=0.9,
                     linestyle="--", alpha=0.7,
                     label="Detected peak" if i == 0 else None)
        if pf.fit_ok:
            ax1b.axvline(pf.r2_fit_px2, color="crimson", linewidth=1.4,
                         linestyle="-", alpha=0.9,
                         label="Gaussian centroid" if i == 0 else None)
            # Red band = ±1σ uncertainty on the r²-domain Gaussian centroid μ
            ax1b.axvspan(pf.r2_fit_px2 - pf.sigma_r2_fit_px2,
                         pf.r2_fit_px2 + pf.sigma_r2_fit_px2,
                         alpha=0.10, color="crimson")

    ax1b.set_title(
        f"Radial profile vs r²  ({good_bins}/{fp.n_bins} bins)  |  "
        f"r_max = {fp.r_max_px:.0f} px  |  "
        f"{n_peaks_r2} peak(s) found (r² fit)  |  "
        f"{'SPARSE' if fp.sparse_bins else 'OK'}",
        fontsize=9,
    )
    ax1b.set_xlabel("r²  (pixel²)", fontsize=8)
    ax1b.set_ylabel("Mean intensity  (ADU)", fontsize=8)
    ax1b.tick_params(labelsize=7)
    ax1b.legend(fontsize=7)

    # Peak fit results table — r² domain
    ax2b = fig2.add_subplot(gs2[2])
    ax2b.axis("off")
    col_labels_r2 = [
        "Peak", "r²_raw (px²)", "r²_fit (px²)", "+/-sig_r² (px²)",
        "r_derived (px)", "+/-sig_r (px)", "Amp (ADU)", "Width σ r² (px²)", "χ²_red",
    ]
    cell_text_r2 = []
    for i, pf in enumerate(fp.peak_fits_r2):
        if pf.fit_ok and pf.r2_fit_px2 > 0:
            r_fit_derived   = float(np.sqrt(pf.r2_fit_px2))
            sigma_r_derived = pf.sigma_r2_fit_px2 / (2.0 * r_fit_derived)
            cell_text_r2.append([
                str(i + 1),
                f"{pf.r2_raw_px2:.2f}",
                f"{pf.r2_fit_px2:.3f}",
                f"{pf.sigma_r2_fit_px2:.3f}",
                f"{r_fit_derived:.3f}",
                f"{sigma_r_derived:.3f}",
                f"{pf.amplitude_adu:.1f}",
                f"{pf.width_r2_px2:.2f}",
                f"{pf.reduced_chi2:.3f}",
            ])
        else:
            cell_text_r2.append([
                str(i + 1),
                f"{pf.r2_raw_px2:.2f}",
                "---", "---", "---", "---",
                f"{pf.profile_raw:.1f}",
                "---", "---",
            ])
    if not cell_text_r2:
        cell_text_r2 = [["—"] * len(col_labels_r2)]

    tbl2 = ax2b.table(
        cellText=cell_text_r2,
        colLabels=col_labels_r2,
        loc="upper center",
        cellLoc="center",
    )
    tbl2.auto_set_font_size(False)
    tbl2.set_fontsize(8.5)
    n_cols_r2 = len(col_labels_r2)
    n_rows_r2 = len(cell_text_r2)
    for c in range(n_cols_r2):
        tbl2[0, c].set_facecolor(hdr_bg)
        tbl2[0, c].set_text_props(color="white", fontweight="bold")
        tbl2[0, c].set_edgecolor("#CCCCCC")
    for r_idx in range(n_rows_r2):
        for c in range(n_cols_r2):
            tbl2[r_idx + 1, c].set_edgecolor("#CCCCCC")
            if r_idx % 2 == 1:
                tbl2[r_idx + 1, c].set_facecolor(alt_bg)
    ax2b.set_title(
        f"Peak Fit Results (r² domain) — for comparison with r-domain fits\n{peaks_r2_path.name}",
        fontsize=10, fontweight="bold", pad=8,
    )

    fig2.suptitle(
        f"Annular Reduction (r² domain) -- {src.name}",
        fontsize=11, fontweight="bold",
    )
    fig2.tight_layout()

    # -- Peak tables to terminal -----------------------------------------------
    _print_peak_table(fp.peak_fits)
    _print_peak_table_r2(fp.peak_fits_r2)
    if fp.peak_fits_r2:
        _plot_first_fringe_diagnostic_r2(fp, fit_half_window=40)
    if fp.peak_fits_r2:
        _plot_all_fringe_diagnostics_r2(fp)

    plt.show()



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
        f"  σ_r2    = {pf.sigma_r2_fit_px2:.2f} px²",
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
    plt.show()



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
    plt.show()


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


if __name__ == "__main__":
    main()
