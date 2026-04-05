"""
annular_reduction.py — Mulligan (1986) r²-binned annular reduction.

Requires a pre-determined fringe centre produced by center_finder.py
(saved as cal_image_centre.npz).  No centre finding is performed here.

Pipeline:
  1. Select L1.1 .npy image ROI.
  2. Select corresponding _centre.npz (cx, cy, sigma_cx, sigma_cy).
  3. Annular reduction → 1-D radial intensity profile (FringeProfile).
     Peak finding and Gaussian fits are performed here, on the profile
     and SEM arrays as soon as they are available.
  4. Save _L1.2.npz with all fields required by M05.
  5. Save cal_image_L1.3.npy  (r_grid | profile | sigma_profile).
  6. Plot: image with centre | radial profile with labelled peaks.

References:
  Harding et al. (2014) Section 3
  Niciejewski et al. (1992) SPIE 1745
  Mulligan (1986) J. Phys. E 19, 545

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
    fit_half_window: int   = 8,
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
        ))

    results.sort(key=lambda p: p.r_raw_px)
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
    peak_fits: list[PeakFit] = field(default_factory=list)


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
    n_bins: int = 150,
    n_subpixels: int = 1,
    sigma_clip_threshold: float = 3.0,
    min_pixels_per_bin: int = 3,
    bad_pixel_mask: Optional[np.ndarray] = None,
    peak_distance: int = 5,
    peak_prominence: float = 50.0,
    peak_fit_half_window: int = 8,   # upper bound; adaptive clamp controls effective value
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
            sparse_bins=True, peak_fits=[],
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

    # -- Save L1.3 — radial profile array (r_grid | profile | sigma_profile) --
    l13_path  = src.with_name("cal_image_L1.3.npy")
    l13_array = np.column_stack([fp.r_grid, fp.profile, fp.sigma_profile])
    np.save(l13_path, l13_array)
    print(f"L1.3 saved : {l13_path}")
    print(f"  shape    : {l13_array.shape}  (n_bins x 3: r_grid | profile | sigma_profile)")

    # -- Save radial_profile_peaks.npy — one row per detected peak -------------
    peaks_path = src.with_name("radial_profile_peaks.npy")
    if fp.peak_fits:
        peaks_array = np.array([
            [i + 1,
             pf.r_raw_px,
             pf.r_fit_px       if pf.fit_ok else np.nan,
             pf.sigma_r_fit_px if pf.fit_ok else np.nan,
             pf.amplitude_adu,
             pf.width_px       if pf.fit_ok else np.nan]
            for i, pf in enumerate(fp.peak_fits)
        ], dtype=np.float64)
    else:
        peaks_array = np.empty((0, 6), dtype=np.float64)
    np.save(peaks_path, peaks_array)
    print(f"Peaks saved: {peaks_path}")
    print(f"  columns  : peak_num | r_raw_px | r_fit_px | sigma_r_fit_px "
          f"| amplitude_adu | width_px")
    print(f"  rows     : {peaks_array.shape[0]} peak(s)")

    # -- Save L1.2 — all fields required by M05 --------------------------------
    l12_path = src.with_name(src.stem.replace("_L1.1", "") + "_L1.2.npz")
    np.savez(
        l12_path,
        profile       = fp.profile,
        sigma_profile = fp.sigma_profile,
        r2_grid       = fp.r2_grid,
        masked        = fp.masked,
        cx            = np.array(cx),
        cy            = np.array(cy),
        sigma_cx      = np.array(sigma_cx),
        sigma_cy      = np.array(sigma_cy),
    )
    print(f"L1.2 saved : {l12_path}")

    # -- Plotting --------------------------------------------------------------
    fig = plt.figure(figsize=(14, 9))
    gs  = fig.add_gridspec(2, 1, hspace=0.35)

    # Top panel — image with centre overlaid
    ax0 = fig.add_subplot(gs[0])
    vlo = float(np.percentile(image,  1))
    vhi = float(np.percentile(image, 99))
    ax0.imshow(image, cmap="gray", origin="lower", vmin=vlo, vmax=vhi,
               aspect="equal")
    ax0.axhline(cy, color="cyan", linewidth=0.8, linestyle="--", alpha=0.9)
    ax0.axvline(cx, color="cyan", linewidth=0.8, linestyle="--", alpha=0.9)
    ax0.plot(cx, cy, "+", color="yellow", markersize=14, markeredgewidth=1.5)
    ax0.set_title(
        f"{src.name}  |  "
        f"cx = {cx:.2f} +/- {sigma_cx:.3f} px,  cy = {cy:.2f} +/- {sigma_cy:.3f} px",
        fontsize=9,
    )
    ax0.set_xlabel("Column (pixel)", fontsize=8)
    ax0.set_ylabel("Row (pixel)",    fontsize=8)
    ax0.tick_params(labelsize=7)

    # Bottom panel — radial profile with peak labels
    ax1    = fig.add_subplot(gs[1])
    good   = ~fp.masked
    finite = good & np.isfinite(fp.sigma_profile)

    # +/-2-sigma outer band (context)
    ax1.errorbar(
        fp.r_grid[finite], fp.profile[finite],
        yerr=fp.two_sigma_profile[finite],
        fmt="none", ecolor="navy", alpha=0.45, linewidth=0.9,
        label="+/-2 sigma SEM",
    )
    # +/-1-sigma inner band — these are the actual fit weights passed to cal_inversion
    ax1.errorbar(
        fp.r_grid[finite], fp.profile[finite],
        yerr=fp.sigma_profile[finite],
        fmt="none", ecolor="darkblue", alpha=0.85, linewidth=1.8,
        label="+/-1 sigma SEM  (fit weight)",
    )
    ax1.plot(fp.r_grid[good], fp.profile[good],
             color="steelblue", linewidth=1.0,
             marker=".", markersize=10, markerfacecolor="steelblue",
             markeredgewidth=0, label="Mean ADU")
    if fp.masked.any():
        ax1.plot(fp.r_grid[fp.masked], fp.profile[fp.masked],
                 "rx", markersize=4, label="Masked bins")

    # Peak labels — offset computed from the profile data range
    finite_profile = fp.profile[finite]
    ax_span = max(float(finite_profile.max() - finite_profile.min()), 1.0) \
              if finite_profile.size > 0 else 1.0

    for i, pf in enumerate(fp.peak_fits):
        # Dashed orange line at raw detection position
        ax1.axvline(pf.r_raw_px, color="darkorange", linewidth=0.9,
                    linestyle="--", alpha=0.7,
                    label="Detected peak" if i == 0 else None)

        if pf.fit_ok:
            # Solid crimson line + shaded +/-1-sigma band at Gaussian centroid
            ax1.axvline(pf.r_fit_px, color="crimson", linewidth=1.4,
                        linestyle="-", alpha=0.9,
                        label="Gaussian centroid" if i == 0 else None)
            ax1.axvspan(pf.r_fit_px - pf.sigma_r_fit_px,
                        pf.r_fit_px + pf.sigma_r_fit_px,
                        alpha=0.10, color="crimson")
            label_str = (
                f"P{i + 1}\n"
                f"r = {pf.r_fit_px:.1f} +/- {pf.sigma_r_fit_px:.2f} px\n"
                f"A = {pf.amplitude_adu:.0f} ADU"
            )
            r_label = pf.r_fit_px
        else:
            label_str = f"P{i + 1}\nr = {pf.r_raw_px:.1f} px\n(fit failed)"
            r_label   = pf.r_raw_px

        # Text annotation just above the peak, nudged right
        ax1.annotate(
            label_str,
            xy       = (r_label, pf.profile_raw),
            xytext   = (r_label + 0.5, pf.profile_raw + 0.06 * ax_span),
            fontsize = 6.5,
            color    = "crimson" if pf.fit_ok else "darkorange",
            va="bottom", ha="left",
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.6),
        )

    ax1.set_title(
        f"Radial profile  ({good_bins}/{fp.n_bins} bins)  |  "
        f"r_max = {fp.r_max_px:.0f} px  |  "
        f"{len(fp.peak_fits)} peak(s) found  |  "
        f"{'SPARSE' if fp.sparse_bins else 'OK'}",
        fontsize=9,
    )
    ax1.set_xlabel("Radius  (pixel)", fontsize=8)
    ax1.set_ylabel("Mean intensity  (ADU)", fontsize=8)
    ax1.tick_params(labelsize=7)
    ax1.legend(fontsize=7)

    fig.suptitle(
        f"Annular Reduction -- {src.name}",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()

    # -- Peak table to terminal (printed before plt.show so it is visible
    #    while the figure is open) -------------------------------------------
    _print_peak_table(fp.peak_fits)

    plt.show()


def _print_peak_table(peak_fits: list[PeakFit]) -> None:
    """Print a formatted summary table of detected peaks to stdout."""
    sep = "-" * 74
    print(f"\n{sep}")
    print(f"  Detected peaks in radial profile  ({len(peak_fits)} found)")
    print(sep)
    if not peak_fits:
        print("  (none)")
        print(sep)
        return

    print(
        f"  {'Peak':>4}  {'r_raw (px)':>10}  {'r_fit (px)':>10}  "
        f"{'+/-sig_r (px)':>13}  {'Amp (ADU)':>9}  {'Width sig (px)':>14}  {'Status':>10}"
    )
    print(sep)
    for i, pf in enumerate(peak_fits):
        if pf.fit_ok:
            print(
                f"  {i + 1:>4}  {pf.r_raw_px:>10.2f}  {pf.r_fit_px:>10.3f}  "
                f"{pf.sigma_r_fit_px:>13.3f}  {pf.amplitude_adu:>9.1f}  "
                f"{pf.width_px:>14.2f}  {'OK':>10}"
            )
        else:
            print(
                f"  {i + 1:>4}  {pf.r_raw_px:>10.2f}  {'---':>10}  "
                f"{'---':>13}  {pf.profile_raw:>9.1f}  {'---':>14}  {'fit failed':>10}"
            )
    print(sep)


if __name__ == "__main__":
    main()
