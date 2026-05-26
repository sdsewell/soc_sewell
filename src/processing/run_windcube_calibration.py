"""
run_windcube_calibration.py
===========================
WindCube FPI neon calibration pipeline.

Workflow
--------
1.  File selection  — Windows dialog boxes to select one cal .bin and one
                      dark .bin file.
2.  Image loading   — load_raw() decodes the big-endian binary header and
                      pixel data; dark-subtracted image is cast to float64.
3.  Figure 1        — 2×2 subplot grid:
      [0,0] Dark-subtracted cal image (grey, with centre cross)
      [0,1] ADU histogram, x-axis fixed 0 – 16383
      [1,0] Coarse grid-search cost map with best-grid point marked
      [1,1] Fine Nelder-Mead result: fringe image with recovered centre
4.  Annular reduction — r²-binned 1-D profile; parabola-primary peak fits
                        (Gaussian fallback) using _R2_WINDOWS below.
5.  Peak table      — matplotlib table figure listing line ID, r²_fit,
                      ±σ, χ²_red for every identified peak.
6.  Figure 3        — 4-column × ⌈N/4⌉-row subplot showing fit window and
                      parabola fit for every peak (from _plot_all_fringe_
                      diagnostics_r2).
7.  Tolansky        — two-line analysis (640.2248 nm + 638.2991 nm):
                      recovers d, α, ε_a, ε_b, Δ_a, Δ_b; prints the
                      Vaughan rectangular-array table; shows four-panel
                      diagnostic figure.

Configuration
-------------
All quantities the user is likely to tune are collected in the
"USER-TUNABLE PARAMETERS" block immediately below the imports.  Edit
_R2_WINDOWS to shift fitting windows; adjust CENTRE_SEED, R_MAX_PX, etc.
as needed.

Libraries used (all from fpi_cal_lib or standard scientific Python)
---------------------------------------------------------------------
  fpi_cal_lib :  find_centre, annular_reduce, peaks_to_array,
                 run_tolansky_2line, plot_tolansky_result,
                 print_rectangular_array, _plot_all_fringe_diagnostics_r2,
                 FringeProfile, PeakFitR2, _R2_WINDOWS, _variance_cost
  numpy, matplotlib, scipy (via fpi_cal_lib), tkinter (dialog only)

Note: No single-line Tolansky analysis is performed; only the two-line
analysis (640.2248 nm + 638.2991 nm) is executed as per WindCube S13a spec.
"""

from __future__ import annotations

import os
import sys
import struct
import pathlib
import warnings
import tkinter as tk
from tkinter import filedialog

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# fpi_cal_lib imports
# All physics lives in the library; this script is UX + orchestration only.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from fpi_cal_lib import (
    # Section A — Centre finder
    find_centre,
    _variance_cost,
    CentreResult,
    # Section C — Annular reduction
    annular_reduce,
    FringeProfile,
    PeakFitR2,
    _R2_WINDOWS,
    _plot_all_fringe_diagnostics_r2,
    _print_peak_table_r2,
    peaks_to_array,
    # Section D — Tolansky
    run_tolansky_2line,
    plot_tolansky_result,
    print_rectangular_array,
)


# ===========================================================================
# USER-TUNABLE PARAMETERS
# ===========================================================================
# Edit these values to adjust centre seed, fitting geometry, and peak windows.
# All _R2_WINDOWS entries below override the auto-detected windows inside
# _find_and_fit_peaks_r2 on a per-peak basis (peak index 0, 1, 2, ...).
# Peaks are ordered by r² ascending; even indices → 640.2 nm (strong),
# odd indices → 638.3 nm (weak).

# ── Centre finding ──────────────────────────────────────────────────────────
# Seed in (row, col) image coordinates.  find_centre() expects (cx, cy)
# where cx = col, cy = row — the conversion is handled automatically below.
CENTRE_SEED_ROW = 145      # row coordinate of fringe centre seed
CENTRE_SEED_COL = 143      # col coordinate of fringe centre seed

VAR_R_MIN_PX    =  5.0     # inner exclusion radius for cost computation (px)
VAR_R_MAX_PX    = None     # None → min(H,W)//2 - 10; or set e.g. 110.0
VAR_N_BINS      = 250      # r² bins used in azimuthal variance cost
VAR_SEARCH_PX   = 15.0     # ±half-width of coarse grid search (px)

# ── Annular reduction ────────────────────────────────────────────────────────
R_MAX_PX        = 122.0    # outer radius for annular integration (px)
N_BINS          = 1500     # number of r² bins  (→ ~43 bins/FWHM)
PEAK_PROMINENCE = 50.0     # minimum peak prominence in ADU

# ── Peak fitting windows in r²  (px²) ───────────────────────────────────────
# These override the auto-detected windows peak-by-peak.
# Format:  { peak_index: (r2_lo, r2_hi), ... }
# Copy _R2_WINDOWS from fpi_cal_lib as the starting point; edit freely.
# Peak index 0 = innermost; even = 640.2 nm, odd = 638.3 nm.
_R2_WINDOWS.update({
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
})

# ── Tolansky two-line analysis ───────────────────────────────────────────────
LAM_A_M     = 640.2248e-9  # Ne strong line wavelength (Burns 1950, IAU 'S')
LAM_B_M     = 638.2991e-9  # Ne weak   line wavelength (Burns 1950, IAU 'S')
D_PRIOR_M   = 20.008e-3    # ICOS nominal gap — used ONLY to resolve N_Delta


# ===========================================================================
# END USER-TUNABLE PARAMETERS
# ===========================================================================


# ---------------------------------------------------------------------------
# Binary loader  (adapted from load_image_swapped.py)
# ---------------------------------------------------------------------------

_KNOWN_FRAME_SIZES = [(260, 276), (528, 552)]


def _load_raw(path: str):
    """
    Load a big-endian WindCube FPI binary.

    Returns
    -------
    header : ndarray (n_cols,) uint16 — decoded header row
    image  : ndarray (n_rows-1, n_cols) uint16 — pixel data
    """
    with open(path, "rb") as f:
        first_words = np.frombuffer(f.read(4), dtype=">u2")
    n_rows_frame = int(first_words[0])
    n_cols_frame = int(first_words[1])
    actual   = os.path.getsize(path)
    expected = n_rows_frame * n_cols_frame * 2
    if actual != expected:
        for rows, cols in _KNOWN_FRAME_SIZES:
            if rows * cols * 2 == actual:
                print(
                    f"  WARNING: header says {n_rows_frame}×{n_cols_frame} "
                    f"but file is {actual} B — using known size {rows}×{cols}."
                )
                n_rows_frame, n_cols_frame = rows, cols
                break
        else:
            raise ValueError(
                f"File size mismatch: got {actual} B, "
                f"expected {expected} for {n_rows_frame}×{n_cols_frame} uint16."
            )
    raw = np.frombuffer(open(path, "rb").read(), dtype=">u2")
    header = raw[:n_cols_frame].copy()
    image  = raw[n_cols_frame:].reshape(n_rows_frame - 1, n_cols_frame).copy()
    return header, image


def _f64(h: np.ndarray, w: int) -> float:
    """Mixed-endian float64: 4 BE uint16 words in LE word order."""
    b = struct.pack(">4H", *reversed([h[w + i] for i in range(4)]))
    return struct.unpack(">d", b)[0]


def _u64(h: np.ndarray, w: int) -> int:
    """Mixed-endian uint64: 4 BE uint16 words in LE word order."""
    return sum(int(h[w + i]) << (16 * i) for i in range(4))


def _exp_time_s(h: np.ndarray) -> float:
    """Exposure time in seconds (word 2 = ms)."""
    return float(int(h[2])) * 1e-3


def _image_type_str(h: np.ndarray) -> str:
    """Quick classification from lamp and shutter bits."""
    gpio  = [int(h[100 + i]) & 0xFF for i in range(4)]
    lamps = [int(h[104 + i]) & 0xFF for i in range(6)]
    shutter = "closed" if (gpio[0] == 1 and gpio[3] == 1) else "open"
    if any(lamps):
        return "cal"
    if shutter == "closed":
        return "dark"
    return "science"


# ---------------------------------------------------------------------------
# Windows file-dialog helpers
# ---------------------------------------------------------------------------

def _pick_file(title: str, initialdir: str | None = None) -> str:
    """Open a Windows dialog box and return the selected path (or '')."""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title=title,
        initialdir=initialdir,
        filetypes=[("WindCube binary", "*.bin"), ("All files", "*.*")],
        parent=root,
    )
    root.destroy()
    return path


# ---------------------------------------------------------------------------
# Figure 1 — image overview + centre finding
# ---------------------------------------------------------------------------

def _build_cost_map(
    image: np.ndarray,
    cx_seed: float,
    cy_seed: float,
    search_px: float,
    r_min_px: float,
    r_max_px: float,
    n_var_bins: int,
    n_grid: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate the azimuthal variance cost on a coarse grid and return
    (cost_map, cx_offsets, cy_offsets) for plotting.
    """
    offsets = np.linspace(-search_px, search_px, n_grid)
    r_min_sq = r_min_px ** 2
    r_max_sq = r_max_px ** 2

    # Clip hot pixels (same pre-processing as find_centre)
    p99_5 = float(np.percentile(image, 99.5))
    img_c = np.clip(image, None, p99_5)

    cost_map = np.zeros((n_grid, n_grid))
    for i, dy in enumerate(offsets):
        for j, dx in enumerate(offsets):
            cost_map[i, j] = _variance_cost(
                cx_seed + dx, cy_seed + dy,
                img_c, r_min_sq, r_max_sq, n_var_bins,
            )
    return cost_map, offsets, offsets


def figure1_overview(
    cal_ds: np.ndarray,
    centre: CentreResult,
    cal_path: str,
    dark_path: str,
    var_r_max_px: float,
) -> plt.Figure:
    """
    2×2 subplot figure:
      [0,0] Dark-subtracted calibration image with centre crosshair
      [0,1] ADU histogram, x-axis 0–16383
      [1,0] Coarse grid search cost map (20×20 samples)
      [1,1] Dark-subtracted image with fine Nelder-Mead centre marked
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(
        f"WindCube FPI — Overview & Centre Finding\n"
        f"Cal: {os.path.basename(cal_path)}   "
        f"Dark: {os.path.basename(dark_path)}",
        fontsize=11, fontweight="bold",
    )

    # ── [0,0] Dark-subtracted image ─────────────────────────────────────────
    ax = axes[0, 0]
    vlo = float(np.percentile(cal_ds, 1))
    vhi = float(np.percentile(cal_ds, 99))
    im  = ax.imshow(cal_ds, cmap="gray", origin="lower",
                    vmin=vlo, vmax=vhi, aspect="equal",
                    interpolation="none")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Counts (ADU)", fontsize=8)
    # Centre crosshair — fine Nelder-Mead result
    cx, cy = centre.cx, centre.cy
    arm = 18
    ax.plot([cx - arm, cx + arm], [cy, cy], color="cyan",
            lw=1.2, ls="--", alpha=0.9)
    ax.plot([cx, cx], [cy - arm, cy + arm], color="cyan",
            lw=1.2, ls="--", alpha=0.9)
    ax.plot(cx, cy, "+", color="yellow", ms=10, mew=1.8)
    ax.set_title(
        f"Dark-subtracted cal  |  "
        f"ADU [{cal_ds.min():.0f}, {cal_ds.max():.0f}]  "
        f"mean {cal_ds.mean():.0f}  std {cal_ds.std():.1f}",
        fontsize=8.5,
    )
    ax.set_xlabel("Column (pixel)", fontsize=8)
    ax.set_ylabel("Row (pixel)", fontsize=8)
    ax.tick_params(labelsize=7)

    # ── [0,1] ADU histogram fixed 0–16383 ───────────────────────────────────
    ax = axes[0, 1]
    ax.hist(cal_ds.ravel(), bins=256, range=(0, 16383),
            color="steelblue", edgecolor="none")
    ax.set_xlim(0, 16383)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2048))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(512))
    ax.set_xlabel("ADU (14-bit, 0–16383)", fontsize=8)
    ax.set_ylabel("Number of pixels", fontsize=8)
    ax.set_title("Pixel ADU distribution — dark-subtracted cal", fontsize=8.5)
    ax.tick_params(labelsize=7)
    # Mark mean and median
    mn  = cal_ds.mean()
    med = float(np.median(cal_ds))
    ax.axvline(mn,  color="orange", lw=1.2, ls="--",
               label=f"mean   {mn:.0f}")
    ax.axvline(med, color="red",    lw=1.2, ls=":",
               label=f"median {med:.0f}")
    ax.legend(fontsize=7)

    # ── [1,0] Coarse grid cost map ────────────────────────────────────────────
    ax = axes[1, 0]
    cx_seed = float(CENTRE_SEED_COL)
    cy_seed = float(CENTRE_SEED_ROW)
    cost_map, dx_offsets, dy_offsets = _build_cost_map(
        cal_ds, cx_seed, cy_seed,
        search_px=VAR_SEARCH_PX,
        r_min_px=VAR_R_MIN_PX,
        r_max_px=var_r_max_px,
        n_var_bins=VAR_N_BINS,
        n_grid=20,
    )
    # Plot cost as 2-D heatmap in absolute pixel coordinates
    # x-axis = cx offset + seed, y-axis = cy offset + seed
    cx_axis = cx_seed + dx_offsets
    cy_axis = cy_seed + dy_offsets
    img_cost = ax.pcolormesh(
        cx_axis, cy_axis, cost_map,
        cmap="viridis_r", shading="auto",
    )
    cb2 = fig.colorbar(img_cost, ax=ax, fraction=0.046, pad=0.04)
    cb2.set_label("Variance cost (ADU²)", fontsize=8)
    # Mark best grid point
    ax.plot(centre.grid_cx, centre.grid_cy, "r+",
            ms=14, mew=2.0, label=f"Grid best ({centre.grid_cx:.2f}, {centre.grid_cy:.2f})")
    ax.set_xlabel("cx  (column, px)", fontsize=8)
    ax.set_ylabel("cy  (row, px)", fontsize=8)
    ax.set_title(
        f"Pass 1 — coarse grid cost map  (20×20, ±{VAR_SEARCH_PX:.0f} px)\n"
        f"Best grid point: cx={centre.grid_cx:.2f}, cy={centre.grid_cy:.2f}  "
        f"cost={centre.grid_cost:.1f}",
        fontsize=8.5,
    )
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)

    # ── [1,1] Fine Nelder-Mead result ─────────────────────────────────────────
    ax = axes[1, 1]
    ax.imshow(cal_ds, cmap="gray", origin="lower",
              vmin=vlo, vmax=vhi, aspect="equal",
              interpolation="none")
    # Fine centre crosshair (cyan dashed) + marker (yellow +)
    ax.axhline(centre.cy, color="cyan", lw=0.9, ls="--", alpha=0.85)
    ax.axvline(centre.cx, color="cyan", lw=0.9, ls="--", alpha=0.85)
    ax.plot(centre.cx, centre.cy, "+", color="yellow", ms=12, mew=2.0)
    ax.set_title(
        f"Pass 2 — Nelder-Mead fine centre\n"
        f"cx = {centre.cx:.3f} ± {centre.sigma_cx:.3f} px  "
        f"cy = {centre.cy:.3f} ± {centre.sigma_cy:.3f} px",
        fontsize=8.5,
    )
    ax.set_xlabel("Column (pixel)", fontsize=8)
    ax.set_ylabel("Row (pixel)", fontsize=8)
    ax.tick_params(labelsize=7)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


# ---------------------------------------------------------------------------
# Figure 2 — peak identification table
# ---------------------------------------------------------------------------

def figure2_peak_table(peaks: list[PeakFitR2]) -> plt.Figure:
    """
    Standalone matplotlib figure showing the peak identification table.

    Columns: Peak #, λ (nm), r²_raw (px²), r²_fit (px²), ±σ_r² (px²),
             r_derived (px), ±σ_r (px), Amplitude (ADU), χ²_red, para_ok
    """
    col_labels = [
        "Peak", "λ (nm)",
        "r²_raw\n(px²)", "r²_fit\n(px²)", "±σ_r²\n(px²)",
        "r_derived\n(px)", "±σ_r\n(px)",
        "Amp\n(ADU)", "χ²_red", "para\nok",
    ]

    rows = []
    row_colours = []
    for k, pf in enumerate(peaks):
        lam_str = "640.2" if k % 2 == 0 else "638.3"
        if pf.fit_ok and np.isfinite(pf.r2_fit_px2) and pf.r2_fit_px2 > 0:
            r_der   = float(np.sqrt(pf.r2_fit_px2))
            sig_r   = (pf.sigma_r2_fit_px2 / (2.0 * r_der)
                       if np.isfinite(pf.sigma_r2_fit_px2) else float("nan"))
            r2_fit  = f"{pf.r2_fit_px2:.2f}"
            sig_r2  = (f"{pf.sigma_r2_fit_px2:.3f}"
                       if np.isfinite(pf.sigma_r2_fit_px2) else "—")
            r_d_str = f"{r_der:.3f}"
            s_r_str = f"{sig_r:.4f}" if np.isfinite(sig_r) else "—"
            chi2_s  = (f"{pf.reduced_chi2:.2f}"
                       if np.isfinite(pf.reduced_chi2) else "—")
        else:
            r2_fit  = "—"
            sig_r2  = "—"
            r_d_str = "—"
            s_r_str = "—"
            chi2_s  = "—"

        para_s = "✓" if pf.para_ok else "✗"
        rows.append([
            str(k),
            lam_str,
            f"{pf.r2_raw_px2:.1f}",
            r2_fit, sig_r2,
            r_d_str, s_r_str,
            f"{pf.amplitude_adu:.0f}",
            chi2_s, para_s,
        ])
        # Colour by line family
        row_colours.append(
            "#EAF4FF" if k % 2 == 0 else "#FFF6EA"   # blue tint / amber tint
        )

    n_rows    = len(rows)
    fig_h     = max(4.0, 0.38 * n_rows + 2.0)
    fig, ax   = plt.subplots(figsize=(15, fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)

    # Header row
    for c in range(len(col_labels)):
        tbl[0, c].set_facecolor("#2C3E50")
        tbl[0, c].set_text_props(color="white", fontweight="bold")

    # Data rows — alternate by line family
    for r_idx, bg in enumerate(row_colours):
        fail = not peaks[r_idx].fit_ok
        for c in range(len(col_labels)):
            tbl[r_idx + 1, c].set_facecolor("#FFE0E0" if fail else bg)

    ax.set_title(
        f"Peak Identification Table — {n_rows} peaks detected\n"
        "Blue: 640.2 nm (strong)  |  Amber: 638.3 nm (weak)  |  "
        "Red: fit failed",
        fontsize=10, fontweight="bold", pad=10,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    print("=" * 68)
    print("  WindCube FPI Neon Calibration Pipeline")
    print("=" * 68)

    # ── Step 1: file selection ───────────────────────────────────────────────
    print("\n[1/7]  Select files via dialog...")
    cal_path  = _pick_file("Select CALIBRATION (.bin) image")
    if not cal_path:
        print("  No calibration file selected — exiting.")
        return
    dark_path = _pick_file(
        "Select DARK (.bin) image",
        initialdir=str(pathlib.Path(cal_path).parent),
    )
    if not dark_path:
        print("  No dark file selected — exiting.")
        return

    print(f"  Cal  : {cal_path}")
    print(f"  Dark : {dark_path}")

    # ── Step 2: load and dark-subtract ──────────────────────────────────────
    print("\n[2/7]  Loading images...")
    hdr_cal,  img_cal  = _load_raw(cal_path)
    hdr_dark, img_dark = _load_raw(dark_path)

    print(f"  Cal  : {img_cal.shape[0]}×{img_cal.shape[1]}  "
          f"ADU [{img_cal.min()}, {img_cal.max()}]  "
          f"mean {img_cal.mean():.1f}  "
          f"exp {_exp_time_s(hdr_cal)*1000:.0f} ms  "
          f"type '{_image_type_str(hdr_cal)}'")
    print(f"  Dark : {img_dark.shape[0]}×{img_dark.shape[1]}  "
          f"ADU [{img_dark.min()}, {img_dark.max()}]  "
          f"mean {img_dark.mean():.1f}  "
          f"exp {_exp_time_s(hdr_dark)*1000:.0f} ms  "
          f"type '{_image_type_str(hdr_dark)}'")

    if img_cal.shape != img_dark.shape:
        print("  ERROR: cal and dark images have different shapes — exiting.")
        return

    # Dark subtraction — result is float64; floor at 0 to avoid negative ADU
    cal_ds = np.maximum(
        img_cal.astype(np.float64) - img_dark.astype(np.float64), 0.0
    )
    print(f"  Dark-subtracted: ADU [{cal_ds.min():.1f}, {cal_ds.max():.1f}]  "
          f"mean {cal_ds.mean():.1f}")

    # ── Step 3: centre finding ───────────────────────────────────────────────
    print("\n[3/7]  Two-pass azimuthal variance centre finding...")
    print(f"  Seed : cx={CENTRE_SEED_COL}, cy={CENTRE_SEED_ROW}  "
          f"(col, row)")

    H, W = cal_ds.shape
    var_r_max_px = VAR_R_MAX_PX if VAR_R_MAX_PX is not None else float(min(H, W) // 2 - 10)

    centre = find_centre(
        cal_ds,
        cx_seed=float(CENTRE_SEED_COL),
        cy_seed=float(CENTRE_SEED_ROW),
        var_r_min_px=VAR_R_MIN_PX,
        var_r_max_px=var_r_max_px,
        var_n_bins=VAR_N_BINS,
        var_search_px=VAR_SEARCH_PX,
    )
    print(f"  Pass 1 grid best: cx={centre.grid_cx:.3f} cy={centre.grid_cy:.3f}  "
          f"cost={centre.grid_cost:.1f}")
    print(f"  Pass 2 NM fine : cx={centre.cx:.4f} ± {centre.sigma_cx:.4f}  "
          f"cy={centre.cy:.4f} ± {centre.sigma_cy:.4f}  "
          f"cost={centre.cost_at_min:.1f}")

    # ── Figure 1 ─────────────────────────────────────────────────────────────
    print("\n  Building Figure 1 (overview + centre finding)...")
    fig1 = figure1_overview(cal_ds, centre, cal_path, dark_path, var_r_max_px)
    fig1.show()

    # ── Step 4: annular reduction + peak fitting ─────────────────────────────
    print("\n[4/7]  Annular reduction and parabolic peak fitting...")
    print(f"  r_max={R_MAX_PX} px  n_bins={N_BINS}  "
          f"prominence={PEAK_PROMINENCE} ADU")

    fp = annular_reduce(
        cal_ds,
        cx=centre.cx,
        cy=centre.cy,
        sigma_cx=centre.sigma_cx,
        sigma_cy=centre.sigma_cy,
        r_max_px=R_MAX_PX,
        n_bins=N_BINS,
        peak_prominence=PEAK_PROMINENCE,
    )

    peaks = fp.peak_fits_r2
    n_peaks = len(peaks)
    n_ok    = sum(1 for p in peaks if p.fit_ok)
    n_para  = sum(1 for p in peaks if p.para_ok)
    print(f"  Peaks detected : {n_peaks}")
    print(f"  Fit ok         : {n_ok}/{n_peaks}")
    print(f"  Parabola ok    : {n_para}/{n_peaks}")

    # Print text table to console
    _print_peak_table_r2(peaks)

    # ── Step 5: peak table figure ─────────────────────────────────────────────
    print("\n[5/7]  Building Figure 2 (peak identification table)...")
    fig2 = figure2_peak_table(peaks)
    fig2.show()

    # ── Step 6: per-peak fit diagnostic grid ──────────────────────────────────
    print("\n[6/7]  Building Figure 3 (4-col per-peak fit diagnostics)...")
    _plot_all_fringe_diagnostics_r2(fp, fit_half_window=40, n_cols=4)

    # ── Step 7: Tolansky two-line analysis ────────────────────────────────────
    print("\n[7/7]  Tolansky two-line analysis (640.2248 + 638.2991 nm)...")
    peak_array = peaks_to_array(peaks)
    n_valid = np.sum(np.isfinite(peak_array[:, 2]))
    print(f"  Peaks in array : {len(peak_array)}  ({n_valid} with valid r²_fit)")

    if n_valid < 4:
        print("  ERROR: fewer than 4 valid fitted peaks — "
              "Tolansky analysis skipped.")
        print("  → Check _R2_WINDOWS and PEAK_PROMINENCE settings.")
    else:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = run_tolansky_2line(
                peak_array,
                lam_a_m=LAM_A_M,
                lam_b_m=LAM_B_M,
                d_prior_m=D_PRIOR_M,
            )

        # Surface any warnings
        for w in caught:
            print(f"  ⚠  {w.message}")

        # Print Vaughan rectangular-array table
        print_rectangular_array(result)

        # Summary
        print(f"\n  ── Tolansky results ──────────────────────────────────")
        print(f"  d     = {result.d_m*1e3:.6f} ± {result.sigma_d_m*1e3:.4f} mm  "
              f"(2σ = ±{result.two_sigma_d_m*1e3:.4f} mm)")
        print(f"  alpha = {result.alpha_mean:.4e} ± {result.sigma_alpha:.4e} rad/px  "
              f"(2σ = ±{result.two_sigma_alpha:.4e})")
        print(f"  eps_a = {result.eps_a:.5f} ± {result.sigma_eps_a:.5f}  "
              f"(640.2 nm fractional order)")
        print(f"  eps_b = {result.eps_b:.5f} ± {result.sigma_eps_b:.5f}  "
              f"(638.3 nm fractional order)")
        print(f"  Δ_a   = {result.Delta_a:.3f} ± {result.sigma_Delta_a:.3f} px²/fr  "
              f"χ²/ν = {result.chi2_dof_a:.3f}")
        print(f"  Δ_b   = {result.Delta_b:.3f} ± {result.sigma_Delta_b:.3f} px²/fr  "
              f"χ²/ν = {result.chi2_dof_b:.3f}")
        ratio_ppm = result.Delta_ratio_residual * 1e6
        ratio_flag = "PASS" if ratio_ppm < 200 else "WARN"
        print(f"  Δ_a/Δ_b consistency : {ratio_ppm:.1f} ppm  [{ratio_flag}]")
        alpha_ppm = result.alpha_consistency * 1e6
        alpha_flag = "PASS" if alpha_ppm < 1000 else "WARN"
        print(f"  α consistency       : {alpha_ppm:.1f} ppm  [{alpha_flag}]")
        print(f"  N_Δ                 : {result.N_Delta}")
        print(f"  ──────────────────────────────────────────────────────")

        # Four-panel Tolansky figure
        print("\n  Building Figure 4 (Tolansky diagnostic)...")
        fig4 = plot_tolansky_result(
            result,
            subtitle=f"{os.path.basename(cal_path)}",
        )
        fig4.show()

    print("\n[done]  All figures displayed.  "
          "Close figure windows to exit.")
    plt.show(block=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
