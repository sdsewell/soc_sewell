"""
fit_cal_image.py — End-to-end calibration image pipeline.

Chains four steps in sequence, passing results in memory.
Only user interaction is a single file dialog to select the raw .bin image.

Pipeline
--------
  1. load_real_image  — decode binary, extract ROI  → L1.1 .npy
  2. center_finder    — two-pass variance minimisation → cx, cy  → _centre.npz
  3. annular_reduction — Mulligan r² reduction → radial profile → L1.2 .npz
  4. cal_inversion    — staged Airy fit → instrument parameters

Each step saves its validation plots as PNG files alongside the source binary.

Usage
-----
    python ingest/fit_cal_image.py
"""

from __future__ import annotations

import datetime
import os
import pathlib
import sys
import tkinter as tk
from tkinter import filedialog


# ---------------------------------------------------------------------------
# Tee — mirror stdout to a log file
# ---------------------------------------------------------------------------

class _Tee:
    """Writes every print() to both the terminal and an open file."""
    def __init__(self, file_obj):
        self._file    = file_obj
        self._stdout  = sys.stdout

    def write(self, data: str) -> None:
        self._stdout.write(data)
        self._file.write(data)

    def flush(self) -> None:
        self._stdout.flush()
        self._file.flush()

import matplotlib.pyplot as plt
import numpy as np

# ── Add pipeline root to path so fpi package is importable ───────────────────
_HERE         = pathlib.Path(__file__).resolve().parent
_PIPELINE_ROOT = _here = _HERE.parent
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

# ── Imports from sub-modules ──────────────────────────────────────────────────
from load_real_image import (
    load_raw, parse_header, extract_roi,
    build_metadata_figure, _plot_image, _plot_hist,
    FRINGE_CENTER, ROI_HALF,
)
from center_finder import find_centre, _variance_cost
from annular_reduction import annular_reduce

from fpi.m03_annular_reduction import FringeProfile as M03FringeProfile
from cal_inversion_new import fit_calibration_fringe, FitConfig, _ne_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_fig(fig: plt.Figure, path: pathlib.Path) -> None:
    """Save figure as PNG and print confirmation."""
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  PNG saved : {path.name}")


def _build_m03_fringe_profile(fp, cr) -> M03FringeProfile:
    """
    Construct the fpi.m03_annular_reduction.FringeProfile that cal_inversion
    expects from the ingest annular_reduction FringeProfile and CentreResult.
    """
    return M03FringeProfile(
        profile           = fp.profile,
        sigma_profile     = fp.sigma_profile,
        two_sigma_profile = fp.two_sigma_profile,
        r_grid            = fp.r_grid,
        r2_grid           = fp.r2_grid,
        n_pixels          = fp.n_pixels,
        masked            = fp.masked,
        cx                = fp.cx,
        cy                = fp.cy,
        sigma_cx          = fp.sigma_cx,
        sigma_cy          = fp.sigma_cy,
        two_sigma_cx      = 2.0 * fp.sigma_cx,
        two_sigma_cy      = 2.0 * fp.sigma_cy,
        seed_source       = "provided",
        stage1_cx         = np.nan,
        stage1_cy         = np.nan,
        cost_at_min       = cr.cost_at_min,
        quality_flags     = 0,
        r_min_px          = fp.r_min_px,
        r_max_px          = fp.r_max_px,
        n_bins            = fp.n_bins,
        n_subpixels       = fp.n_subpixels,
        sigma_clip        = fp.sigma_clip,
        image_shape       = fp.image_shape,
    )


# ---------------------------------------------------------------------------
# Step 1 — load_real_image
# ---------------------------------------------------------------------------

def step1_load(bin_file: str) -> tuple:
    """
    Load binary, extract ROI, save L1.1 .npy, save validation PNGs.

    Returns (image_roi, metadata, src_path, out_dir)
    """
    import matplotlib.patches as mpatches

    src      = pathlib.Path(bin_file)
    out_dir  = src.parent
    filename = src.name

    print(f"\n{'='*60}")
    print(f"STEP 1 — Load image: {filename}")
    print(f"{'='*60}")

    header_be, image = load_raw(bin_file)
    metadata = parse_header(header_be)

    print(f"  Shape      : {image.shape[0]} × {image.shape[1]}  ADU")
    print(f"  UTC        : {metadata['utc_timestamp']}")
    print(f"  Image type : {metadata['img_type']}")

    roi = extract_roi(image, FRINGE_CENTER, ROI_HALF)
    roi_cx = roi.shape[1] / 2.0 - 0.5
    roi_cy = roi.shape[0] / 2.0 - 0.5
    print(f"  ROI shape  : {roi.shape[0]} × {roi.shape[1]} px  "
          f"[user-determined centre: row {FRINGE_CENTER[0]}, col {FRINGE_CENTER[1]}]")
    print(f"  ROI half   : {ROI_HALF} px  [user-determined]")
    print(f"  ROI centre : cx = {roi_cx:.1f},  cy = {roi_cy:.1f}  px  [derived from ROI shape]")

    # Save L1.1
    l11_path = src.with_name(src.stem.replace("_L0", "") + "_L1.1.npy")
    np.save(l11_path, roi)
    print(f"  L1.1 saved : {l11_path.name}")

    # Save metadata as CSV
    import csv
    csv_path = src.with_name(src.stem + "_metadata.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["field", "value"])
        for key, val in metadata.items():
            writer.writerow([key, val])
    print(f"  Metadata CSV : {csv_path.name}")

    # ── Validation figure 1: image + ROI + histograms ────────────────────────
    cr, cc = FRINGE_CENTER
    roi_title = (f"ROI  {roi.shape[0]}×{roi.shape[1]} px  "
                 f"centred at ({cr}, {cc})")

    fig1, axes = plt.subplots(2, 2, figsize=(14, 10))
    _plot_image(axes[0, 0], fig1, image, "Unmasked (full frame)")
    _plot_image(axes[0, 1], fig1, roi,   roi_title)

    ax0 = axes[0, 0]
    ax0.axhline(cr, color="cyan",   linewidth=0.8, linestyle="--", alpha=0.9)
    ax0.axvline(cc, color="cyan",   linewidth=0.8, linestyle="--", alpha=0.9)
    _ARM = 15
    ax0.plot([cc - _ARM, cc + _ARM], [cr, cr],
             color="yellow", linewidth=1.5, alpha=1.0)
    ax0.plot([cc, cc], [cr - _ARM, cr + _ARM],
             color="yellow", linewidth=1.5, alpha=1.0)
    r_lo = max(0, cr - ROI_HALF);  r_hi = min(image.shape[0], cr + ROI_HALF)
    c_lo = max(0, cc - ROI_HALF);  c_hi = min(image.shape[1], cc + ROI_HALF)
    ax0.add_patch(mpatches.Rectangle(
        (c_lo - 0.5, r_lo - 0.5), c_hi - c_lo, r_hi - r_lo,
        linewidth=1.2, edgecolor="red", facecolor="none",
    ))

    _plot_hist(axes[1, 0], image, "Pixel Distribution — Full frame")
    _plot_hist(axes[1, 1], roi,   "Pixel Distribution — ROI")
    fig1.suptitle(f"Step 1: Load — {filename}", fontsize=12, fontweight="bold")
    fig1.tight_layout()
    _save_fig(fig1, out_dir / (src.stem + "_step1_image.png"))

    # ── Validation figure 2: metadata table ──────────────────────────────────
    fig2 = build_metadata_figure(metadata, filename)
    _save_fig(fig2, out_dir / (src.stem + "_step1_metadata.png"))

    plt.show()
    plt.close("all")
    return roi, metadata, src, out_dir


# ---------------------------------------------------------------------------
# Step 2 — center_finder
# ---------------------------------------------------------------------------

def step2_centre(roi: np.ndarray, src: pathlib.Path,
                 out_dir: pathlib.Path) -> object:
    """
    Two-pass azimuthal variance centre finding.
    Saves _centre.npz and validation PNG.

    Returns CentreResult.
    """
    print(f"\n{'='*60}")
    print("STEP 2 — Centre finding")
    print(f"{'='*60}")

    cr = find_centre(roi)
    print(f"  Pass 1 — coarse grid search:")
    print(f"    cx = {cr.grid_cx:.3f} px,  cy = {cr.grid_cy:.3f} px  "
          f"(cost = {cr.grid_cost:.4g})")
    print(f"  Pass 2 — Nelder-Mead refinement (authoritative):")
    print(f"    cx = {cr.cx:.3f} ± {cr.sigma_cx:.3f} px")
    print(f"    cy = {cr.cy:.3f} ± {cr.sigma_cy:.3f} px")
    print(f"    cost = {cr.cost_at_min:.4g}  "
          f"(improvement: {cr.grid_cost - cr.cost_at_min:.4g})")

    # Save centre
    centre_path = src.with_name(src.stem.replace("_L1.1", "").replace("_L0", "")
                                + "_centre.npz")
    np.savez(
        centre_path,
        cx       = np.array(cr.cx),
        cy       = np.array(cr.cy),
        sigma_cx = np.array(cr.sigma_cx),
        sigma_cy = np.array(cr.sigma_cy),
    )
    print(f"  Centre saved : {centre_path.name}")

    # ── Shared variance parameters ────────────────────────────────────────────
    H, W          = roi.shape
    var_r_max_px  = float(min(H, W) // 2 - 10)
    r_min_sq      = 5.0 ** 2
    r_max_sq      = var_r_max_px ** 2
    n_var_bins    = 250
    var_search_px = 15.0
    grid_step     = max(2.0, var_search_px / 8.0)
    img_c         = np.clip(roi, None, float(np.percentile(roi, 99.5)))

    # Row 0 — coarse grid: evaluated points along each axis
    cx_seed      = (W - 1) / 2.0
    cy_seed      = (H - 1) / 2.0
    grid_offsets = np.arange(-var_search_px, var_search_px + grid_step * 0.5, grid_step)
    grid_cx_pts  = cx_seed + grid_offsets
    grid_cy_pts  = cy_seed + grid_offsets
    grid_cost_cx = np.array([_variance_cost(cx, cr.grid_cy, img_c, r_min_sq, r_max_sq, n_var_bins)
                             for cx in grid_cx_pts])
    grid_cost_cy = np.array([_variance_cost(cr.grid_cx, cy, img_c, r_min_sq, r_max_sq, n_var_bins)
                             for cy in grid_cy_pts])
    coarse_cx_line = np.linspace(grid_cx_pts[0], grid_cx_pts[-1], 300)
    coarse_cy_line = np.linspace(grid_cy_pts[0], grid_cy_pts[-1], 300)
    coarse_cost_cx_line = np.array([_variance_cost(cx, cr.grid_cy, img_c, r_min_sq, r_max_sq, n_var_bins)
                                    for cx in coarse_cx_line])
    coarse_cost_cy_line = np.array([_variance_cost(cr.grid_cx, cy, img_c, r_min_sq, r_max_sq, n_var_bins)
                                    for cy in coarse_cy_line])

    # Row 1 — fine scan around Nelder-Mead result
    fine_half    = 5.0
    fine_pts     = 201
    fine_cx_scan = np.linspace(cr.cx - fine_half, cr.cx + fine_half, fine_pts)
    fine_cy_scan = np.linspace(cr.cy - fine_half, cr.cy + fine_half, fine_pts)
    fine_cost_cx = np.array([_variance_cost(cx, cr.cy, img_c, r_min_sq, r_max_sq, n_var_bins)
                             for cx in fine_cx_scan])
    fine_cost_cy = np.array([_variance_cost(cr.cx, cy, img_c, r_min_sq, r_max_sq, n_var_bins)
                             for cy in fine_cy_scan])

    # ── Validation figure — 2 rows × 3 cols ──────────────────────────────────
    vlo = float(np.percentile(roi, 1))
    vhi = float(np.percentile(roi, 99))

    fig = plt.figure(figsize=(17, 10))
    gs  = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.32)

    # [0,0] Image — grid best
    ax00 = fig.add_subplot(gs[0, 0])
    ax00.imshow(roi, cmap="gray", origin="lower", vmin=vlo, vmax=vhi, aspect="equal")
    ax00.axhline(cr.grid_cy, color="lime", linewidth=0.8, linestyle="--", alpha=0.9)
    ax00.axvline(cr.grid_cx, color="lime", linewidth=0.8, linestyle="--", alpha=0.9)
    ax00.plot(cr.grid_cx, cr.grid_cy, "s", color="lime", markersize=10,
              markeredgewidth=1.5, markerfacecolor="none")
    ax00.set_title(f"Pass 1 — coarse grid  (step = {grid_step:.0f} px)\n"
                   f"cx = {cr.grid_cx:.2f} px,  cy = {cr.grid_cy:.2f} px", fontsize=9)
    ax00.set_xlabel("Column (pixel)", fontsize=8);  ax00.set_ylabel("Row (pixel)", fontsize=8)
    ax00.tick_params(labelsize=7)

    # [0,1] Variance vs cx — coarse with grid points
    ax01 = fig.add_subplot(gs[0, 1])
    ax01.plot(coarse_cx_line, coarse_cost_cx_line,
              color="steelblue", linewidth=1.0, alpha=0.5, zorder=1)
    ax01.plot(grid_cx_pts, grid_cost_cx, "o", color="steelblue",
              markersize=10, markeredgewidth=0.8, markeredgecolor="navy",
              zorder=2, label=f"{len(grid_cx_pts)} grid points")
    ax01.axvline(cr.grid_cx, color="lime", linewidth=1.2,
                 label=f"grid best cx = {cr.grid_cx:.2f} px")
    ax01.set_title("Variance vs cx  [grid search, cy fixed at seed]", fontsize=9)
    ax01.set_xlabel("cx  (pixel)", fontsize=8);  ax01.set_ylabel("Azimuthal variance cost", fontsize=8)
    ax01.tick_params(labelsize=7);  ax01.legend(fontsize=7, loc="upper right")

    # [0,2] Variance vs cy — coarse with grid points
    ax02 = fig.add_subplot(gs[0, 2])
    ax02.plot(coarse_cy_line, coarse_cost_cy_line,
              color="darkorange", linewidth=1.0, alpha=0.5, zorder=1)
    ax02.plot(grid_cy_pts, grid_cost_cy, "o", color="darkorange",
              markersize=10, markeredgewidth=0.8, markeredgecolor="saddlebrown",
              zorder=2, label=f"{len(grid_cy_pts)} grid points")
    ax02.axvline(cr.grid_cy, color="lime", linewidth=1.2,
                 label=f"grid best cy = {cr.grid_cy:.2f} px")
    ax02.set_title("Variance vs cy  [grid search, cx fixed at seed]", fontsize=9)
    ax02.set_xlabel("cy  (pixel)", fontsize=8);  ax02.set_ylabel("Azimuthal variance cost", fontsize=8)
    ax02.tick_params(labelsize=7);  ax02.legend(fontsize=7, loc="upper right")

    # [1,0] Image — Nelder-Mead result
    ax10 = fig.add_subplot(gs[1, 0])
    ax10.imshow(roi, cmap="gray", origin="lower", vmin=vlo, vmax=vhi, aspect="equal")
    ax10.axhline(cr.cy, color="cyan", linewidth=0.8, linestyle="--", alpha=0.9)
    ax10.axvline(cr.cx, color="cyan", linewidth=0.8, linestyle="--", alpha=0.9)
    ax10.plot(cr.cx, cr.cy, "+", color="yellow", markersize=14, markeredgewidth=1.5)
    ax10.set_title(f"Pass 2 — Nelder-Mead refinement\n"
                   f"cx = {cr.cx:.3f} ± {cr.sigma_cx:.3f} px\n"
                   f"cy = {cr.cy:.3f} ± {cr.sigma_cy:.3f} px", fontsize=9)
    ax10.set_xlabel("Column (pixel)", fontsize=8);  ax10.set_ylabel("Row (pixel)", fontsize=8)
    ax10.tick_params(labelsize=7)

    # [1,1] Variance vs cx — fine scan
    ax11 = fig.add_subplot(gs[1, 1])
    ax11.plot(fine_cx_scan, fine_cost_cx, color="steelblue", linewidth=1.2,
              marker="o", markersize=4, markerfacecolor="steelblue", markeredgewidth=0)
    ax11.axvline(cr.cx, color="red", linewidth=1.2, label=f"NM cx = {cr.cx:.3f} px")
    ax11.axvspan(cr.cx - cr.sigma_cx,     cr.cx + cr.sigma_cx,
                 alpha=0.22, color="red",    label=f"±1σ = {cr.sigma_cx:.3f} px")
    ax11.axvspan(cr.cx - cr.two_sigma_cx, cr.cx + cr.two_sigma_cx,
                 alpha=0.10, color="orange", label=f"±2σ = {cr.two_sigma_cx:.3f} px")
    ax11.set_title(f"Variance vs cx  [Nelder-Mead, ±{fine_half:.0f} px window]", fontsize=9)
    ax11.set_xlabel("cx  (pixel)", fontsize=8);  ax11.set_ylabel("Azimuthal variance cost", fontsize=8)
    ax11.tick_params(labelsize=7);  ax11.legend(fontsize=7, loc="upper right")

    # [1,2] Variance vs cy — fine scan
    ax12 = fig.add_subplot(gs[1, 2])
    ax12.plot(fine_cy_scan, fine_cost_cy, color="darkorange", linewidth=1.2,
              marker="o", markersize=4, markerfacecolor="darkorange", markeredgewidth=0)
    ax12.axvline(cr.cy, color="red", linewidth=1.2, label=f"NM cy = {cr.cy:.3f} px")
    ax12.axvspan(cr.cy - cr.sigma_cy,     cr.cy + cr.sigma_cy,
                 alpha=0.22, color="red",    label=f"±1σ = {cr.sigma_cy:.3f} px")
    ax12.axvspan(cr.cy - cr.two_sigma_cy, cr.cy + cr.two_sigma_cy,
                 alpha=0.10, color="orange", label=f"±2σ = {cr.two_sigma_cy:.3f} px")
    ax12.set_title(f"Variance vs cy  [Nelder-Mead, ±{fine_half:.0f} px window]", fontsize=9)
    ax12.set_xlabel("cy  (pixel)", fontsize=8);  ax12.set_ylabel("Azimuthal variance cost", fontsize=8)
    ax12.tick_params(labelsize=7);  ax12.legend(fontsize=7, loc="upper right")

    fig.suptitle(
        f"Step 2: Centre finding — {src.stem}  |  "
        f"grid cost = {cr.grid_cost:.4g}  →  NM cost = {cr.cost_at_min:.4g}",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    _save_fig(fig, out_dir / (src.stem.replace("_L1.1", "").replace("_L0", "")
                              + "_step2_centre.png"))
    plt.show()
    plt.close("all")
    return cr


# ---------------------------------------------------------------------------
# Step 3 — annular_reduction
# ---------------------------------------------------------------------------

def step3_reduce(roi: np.ndarray, cr, src: pathlib.Path,
                 out_dir: pathlib.Path):
    """
    Mulligan r² annular reduction using the located centre.
    Saves _L1.2.npz and validation PNG.

    Returns (ingest FringeProfile, M03FringeProfile for cal_inversion).
    """
    print(f"\n{'='*60}")
    print("STEP 3 — Annular reduction")
    print(f"{'='*60}")

    fp = annular_reduce(roi, cr.cx, cr.cy, cr.sigma_cx, cr.sigma_cy)
    good_bins = int((~fp.masked).sum())
    print(f"  Bins : {fp.n_bins} total,  {good_bins} good,  "
          f"{fp.n_bins - good_bins} masked")
    if fp.sparse_bins:
        print("  WARNING: > 10 % of bins are sparse or masked")

    # Save L1.2
    base     = src.stem.replace("_L1.1", "").replace("_L0", "")
    l12_path = out_dir / (base + "_L1.2.npz")
    np.savez(
        l12_path,
        profile       = fp.profile,
        sigma_profile = fp.sigma_profile,
        r2_grid       = fp.r2_grid,
        masked        = fp.masked,
        cx            = np.array(cr.cx),
        cy            = np.array(cr.cy),
        sigma_cx      = np.array(cr.sigma_cx),
        sigma_cy      = np.array(cr.sigma_cy),
    )
    print(f"  L1.2 saved : {l12_path.name}")
    print(f"  Profile range : {fp.profile.min():.1f} – {fp.profile.max():.1f}  ADU")

    # ── Validation figure ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 9))
    gs  = fig.add_gridspec(2, 1, hspace=0.35)

    ax0 = fig.add_subplot(gs[0])
    vlo = float(np.percentile(roi, 1));  vhi = float(np.percentile(roi, 99))
    ax0.imshow(roi, cmap="gray", origin="lower", vmin=vlo, vmax=vhi, aspect="equal")
    ax0.axhline(cr.cy, color="cyan", linewidth=0.8, linestyle="--", alpha=0.9)
    ax0.axvline(cr.cx, color="cyan", linewidth=0.8, linestyle="--", alpha=0.9)
    ax0.plot(cr.cx, cr.cy, "+", color="yellow", markersize=14, markeredgewidth=1.5)
    ax0.set_title(f"cx = {cr.cx:.2f} ± {cr.sigma_cx:.3f} px,  "
                  f"cy = {cr.cy:.2f} ± {cr.sigma_cy:.3f} px", fontsize=9)
    ax0.set_xlabel("Column (pixel)", fontsize=8);  ax0.set_ylabel("Row (pixel)", fontsize=8)
    ax0.tick_params(labelsize=7)

    ax1    = fig.add_subplot(gs[1])
    good   = ~fp.masked
    finite = good & np.isfinite(fp.sigma_profile)
    # ±2σ outer band (context)
    ax1.errorbar(fp.r_grid[finite], fp.profile[finite],
                 yerr=fp.two_sigma_profile[finite],
                 fmt="none", ecolor="navy", alpha=0.45, linewidth=0.9,
                 label="±2σ SEM")
    # ±1σ inner band — actual fit weights passed to cal_inversion
    ax1.errorbar(fp.r_grid[finite], fp.profile[finite],
                 yerr=fp.sigma_profile[finite],
                 fmt="none", ecolor="darkblue", alpha=0.85, linewidth=1.8,
                 label="±1σ SEM  (fit weight)")
    ax1.plot(fp.r_grid[good], fp.profile[good],
             color="steelblue", linewidth=1.0,
             marker=".", markersize=10, markerfacecolor="steelblue",
             markeredgewidth=0, label="Mean ADU")
    if fp.masked.any():
        ax1.plot(fp.r_grid[fp.masked], fp.profile[fp.masked],
                 "rx", markersize=4, label="Masked bins")
    ax1.set_title(f"Radial profile  ({good_bins}/{fp.n_bins} bins)  |  "
                  f"{'SPARSE' if fp.sparse_bins else 'OK'}", fontsize=9)
    ax1.set_xlabel("Radius  (pixel)", fontsize=8)
    ax1.set_ylabel("Mean intensity  (ADU)", fontsize=8)
    ax1.tick_params(labelsize=7);  ax1.legend(fontsize=7)

    fig.suptitle("Step 3: Annular reduction", fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, out_dir / (base + "_step3_profile.png"))
    plt.show()
    plt.close("all")

    m03_fp = _build_m03_fringe_profile(fp, cr)
    return fp, m03_fp


# ---------------------------------------------------------------------------
# Step 4 — cal_inversion
# ---------------------------------------------------------------------------

def step4_invert(m03_fp: M03FringeProfile, src: pathlib.Path,
                 out_dir: pathlib.Path) -> object:
    """
    Staged Airy calibration fit.
    Saves fit results as .npz and validation PNG.

    Returns CalibrationResult.
    """
    print(f"\n{'='*60}")
    print("STEP 4 — Calibration inversion")
    print(f"{'='*60}")

    result = fit_calibration_fringe(m03_fp)

    print(f"  {'Parameter':<12}  {'Value':>18}  {'±1σ':>16}  {'±2σ':>16}")
    print(f"  {'-'*66}")
    print(f"  {'t_m':<12}  {result.t_m*1e3:>15.6f} mm  "
          f"  {result.sigma_t_m*1e6:>11.3f} µm  {result.two_sigma_t_m*1e6:>11.3f} µm")
    print(f"  {'R':<12}  {result.R_refl:>18.6f}  "
          f"{result.sigma_R_refl:>16.6f}  {result.two_sigma_R_refl:>16.6f}")
    print(f"  {'alpha':<12}  {result.alpha:>18.6g}  "
          f"{result.sigma_alpha:>16.4g}  {result.two_sigma_alpha:>16.4g}")
    print(f"  {'I0':<12}  {result.I0:>18.3f}  "
          f"{result.sigma_I0:>16.3f}  {result.two_sigma_I0:>16.3f}")
    print(f"  {'I1':<12}  {result.I1:>18.3f}  "
          f"{result.sigma_I1:>16.3f}  {result.two_sigma_I1:>16.3f}")
    print(f"  {'I2':<12}  {result.I2:>18.3f}  "
          f"{result.sigma_I2:>16.3f}  {result.two_sigma_I2:>16.3f}")
    print(f"  {'B':<12}  {result.B:>18.3f}  "
          f"{result.sigma_B:>16.3f}  {result.two_sigma_B:>16.3f}")
    print(f"  {'sigma0':<12}  {result.sigma0:>18.6f}  "
          f"{result.sigma_sigma0:>16.6f}  {result.two_sigma_sigma0:>16.6f}")
    print(f"  {'sigma1':<12}  {result.sigma1:>18.6f}  "
          f"{result.sigma_sigma1:>16.6f}  {result.two_sigma_sigma1:>16.6f}")
    print(f"  {'sigma2':<12}  {result.sigma2:>18.6f}  "
          f"{result.sigma_sigma2:>16.6f}  {result.two_sigma_sigma2:>16.6f}")
    print(f"  {'-'*66}")
    print(f"  chi²_red = {result.chi2_reduced:.4f}  ({result.n_bins_used} bins,  "
          f"{result.n_params_free} free params)")
    print(f"  ε_cal    = {result.epsilon_cal:.4f}   ε_sci = {result.epsilon_sci:.4f}")
    print(f"  m0_cal   = {result.m0_cal:.4f}         m0_sci = {result.m0_sci:.4f}")
    print(f"  converged: {result.converged}")
    print(f"  quality flags : 0x{result.quality_flags:02X}  "
          f"({'GOOD' if result.quality_flags == 0 else 'flagged'})")

    # Save result
    base      = src.stem.replace("_L1.1", "").replace("_L0", "")
    cal_path  = out_dir / (base + "_cal_result.npz")
    np.savez(
        cal_path,
        t_m          = np.array(result.t_m),
        R_refl       = np.array(result.R_refl),
        alpha        = np.array(result.alpha),
        I0           = np.array(result.I0),
        I1           = np.array(result.I1),
        I2           = np.array(result.I2),
        B            = np.array(result.B),
        sigma0       = np.array(result.sigma0),
        sigma1       = np.array(result.sigma1),
        sigma2       = np.array(result.sigma2),
        epsilon_cal  = np.array(result.epsilon_cal),
        epsilon_sci  = np.array(result.epsilon_sci),
        chi2_reduced = np.array(result.chi2_reduced),
        quality_flags= np.array(result.quality_flags),
    )
    print(f"  Cal result saved : {cal_path.name}")

    # ── Validation figure — parameter table (top) + radial profile (bottom) ──
    good   = ~m03_fp.masked
    finite = good & np.isfinite(m03_fp.sigma_profile)
    cfg    = result.fit_config

    # Build initial-guess values — FitConfig holds explicit inits where
    # available; intensity/PSF params are auto-derived and shown as "auto".
    t_init_str = f"{cfg.t_init_mm:.6f} mm"
    B_init_str = f"{cfg.B_init:.1f}" if cfg.B_init is not None else "auto (5th pct)"

    # chi² by stage as a short string for the table footer
    chi2_stage_str = "  ".join(
        f"S{i}: {v:.3f}" for i, v in enumerate(result.chi2_by_stage)
    )

    col_labels = ["Parameter", "Initial guess", "Fitted value", "±1σ", "±2σ"]
    rows = [
        ["t_m",    t_init_str,
         f"{result.t_m*1e3:.6f} mm",
         f"{result.sigma_t_m*1e6:.2f} µm",
         f"{result.two_sigma_t_m*1e6:.2f} µm"],
        ["R",      f"{cfg.R_init:.4f}",
         f"{result.R_refl:.6f}",
         f"{result.sigma_R_refl:.6f}",
         f"{result.two_sigma_R_refl:.6f}"],
        ["α",      f"{cfg.alpha_init:.4g}",
         f"{result.alpha:.5g}",
         f"{result.sigma_alpha:.4g}",
         f"{result.two_sigma_alpha:.4g}"],
        ["I0",     "auto",
         f"{result.I0:.3f}",
         f"{result.sigma_I0:.3f}",
         f"{result.two_sigma_I0:.3f}"],
        ["I1",     "auto",
         f"{result.I1:.3f}",
         f"{result.sigma_I1:.3f}",
         f"{result.two_sigma_I1:.3f}"],
        ["I2",     "auto",
         f"{result.I2:.3f}",
         f"{result.sigma_I2:.3f}",
         f"{result.two_sigma_I2:.3f}"],
        ["B",      B_init_str,
         f"{result.B:.3f}",
         f"{result.sigma_B:.3f}",
         f"{result.two_sigma_B:.3f}"],
        ["σ0",     f"{cfg.sigma0_init:.4f}",
         f"{result.sigma0:.6f}",
         f"{result.sigma_sigma0:.6f}",
         f"{result.two_sigma_sigma0:.6f}"],
        ["σ1",     "auto",
         f"{result.sigma1:.6f}",
         f"{result.sigma_sigma1:.6f}",
         f"{result.two_sigma_sigma1:.6f}"],
        ["σ2",     "auto",
         f"{result.sigma2:.6f}",
         f"{result.sigma_sigma2:.6f}",
         f"{result.two_sigma_sigma2:.6f}"],
        ["ε_cal",  "—",  f"{result.epsilon_cal:.4f}", "—", "—"],
        ["ε_sci",  "—",  f"{result.epsilon_sci:.4f}", "—", "—"],
    ]

    fig = plt.figure(figsize=(14, 11))
    gs  = fig.add_gridspec(2, 1, height_ratios=[1.1, 1.0], hspace=0.38)

    # ── Top: parameter table ──────────────────────────────────────────────────
    ax_tbl = fig.add_subplot(gs[0])
    ax_tbl.axis("off")

    tbl = ax_tbl.table(
        cellText   = rows,
        colLabels  = col_labels,
        colWidths  = [0.10, 0.22, 0.22, 0.20, 0.20],
        loc        = "center",
        cellLoc    = "center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)

    hdr_bg  = "#2C3E50"
    alt_bg  = "#EBF5FB"
    n_cols  = len(col_labels)
    for c in range(n_cols):
        tbl[0, c].set_facecolor(hdr_bg)
        tbl[0, c].set_text_props(color="white", fontweight="bold")
        tbl[0, c].set_edgecolor("#AAAAAA")
    for r in range(1, len(rows) + 1):
        for c in range(n_cols):
            tbl[r, c].set_edgecolor("#CCCCCC")
            if r % 2 == 0:
                tbl[r, c].set_facecolor(alt_bg)

    ax_tbl.set_title(
        f"Fitted parameters  |  χ²_red = {result.chi2_reduced:.4f}  "
        f"({result.n_bins_used} bins,  {result.n_params_free} free)  |  "
        f"flags 0x{result.quality_flags:02X}  |  converged: {result.converged}\n"
        f"χ² by stage — {chi2_stage_str}",
        fontsize=8.5, pad=8,
    )

    # ── Bottom: radial profile + best-fit model overlay ──────────────────────
    ax1 = fig.add_subplot(gs[1])
    ax1.errorbar(m03_fp.r_grid[finite], m03_fp.profile[finite],
                 yerr=2.0 * m03_fp.sigma_profile[finite],
                 fmt="none", ecolor="steelblue", alpha=0.35, linewidth=0.8)
    ax1.plot(m03_fp.r_grid[good], m03_fp.profile[good],
             color="steelblue", linewidth=0.0,
             marker="o", markersize=4, markerfacecolor="steelblue",
             markeredgewidth=0, label="Observed profile")
    if m03_fp.masked.any():
        ax1.plot(m03_fp.r_grid[m03_fp.masked], m03_fp.profile[m03_fp.masked],
                 "rx", markersize=4, label="Masked bins")

    # Best-fit Airy model — smooth line on fine r-grid
    r_fine_model = np.linspace(0.0, m03_fp.r_max_px, 1000)
    model_fine = _ne_model(
        r_fine_model, m03_fp.r_max_px,
        t=result.t_m, R_refl=result.R_refl, alpha=result.alpha, n=1.0,
        I0=result.I0, I1=result.I1, I2=result.I2,
        sigma0=result.sigma0, sigma1=result.sigma1, sigma2=result.sigma2,
        B=result.B, n_bins=None,
    )
    ax1.plot(r_fine_model, model_fine,
             color="firebrick", linewidth=1.5, label="Best-fit model")

    ax1.set_title(f"Radial profile  ({result.n_bins_used} bins used)  |  "
                  f"χ²_red = {result.chi2_reduced:.4f}", fontsize=9)
    ax1.set_xlabel("Radius  (pixel)", fontsize=8)
    ax1.set_ylabel("Mean intensity  (ADU)", fontsize=8)
    ax1.tick_params(labelsize=7)
    ax1.legend(fontsize=7)

    fig.suptitle("Step 4: Calibration inversion", fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, out_dir / (base + "_step4_calinversion.png"))
    plt.show()
    plt.close("all")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    root = tk.Tk()
    root.withdraw()
    raw_dir = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     r"..\raw_images_with_metadata")
    )
    bin_file = filedialog.askopenfilename(
        title="Select FPI calibration binary image (.bin)",
        initialdir=raw_dir,
        filetypes=[("Binary image", "*.bin"), ("All files", "*.*")],
    )
    root.destroy()
    if not bin_file:
        print("No file selected — exiting.")
        return

    log_path = pathlib.Path(bin_file).parent / "fit_cal_image_results.txt"
    with open(log_path, "w", encoding="utf-8") as _log:
        sys.stdout = _Tee(_log)
        try:
            _log.write(
                f"fit_cal_image.py  —  {datetime.datetime.now().isoformat(timespec='seconds')}\n"
                f"Source : {bin_file}\n"
                f"{'='*60}\n"
            )

            roi, _, src, out_dir = step1_load(bin_file)
            cr                   = step2_centre(roi, src, out_dir)
            _, m03_fp            = step3_reduce(roi, cr, src, out_dir)
            _                    = step4_invert(m03_fp, src, out_dir)

            print(f"\n{'='*60}")
            print("Pipeline complete.")
            print(f"  Output folder : {out_dir}")
            print(f"  Log saved     : {log_path}")
            print(f"{'='*60}")
        finally:
            sys.stdout = sys.stdout._stdout   # restore terminal


if __name__ == "__main__":
    main()
