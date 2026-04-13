"""
Script:  z00_validate_annular_reduction_2026-04-13.py
Spec:    specs/Z00_validate_annular_reduction_peak_finding_2026-04-13.md
Author:  Claude Code
Created: 2026-04-13
Project: WindCube FPI Pipeline — NCAR/HAO
Repo:    soc_sewell

End-to-end interactive validation of M03 annular reduction and peak finding.
Produces seven diagnostic figures (a)–(g) and a structured peak table .npy.

Usage:
    python validation/z00_validate_annular_reduction_2026-04-13.py
"""

import math
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")        # headless-safe; switch to TkAgg if interactive display wanted
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

# Add both repo root and src/ to path so imports work consistently
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from src.fpi.m03_annular_reduction_2026_04_06 import (
    make_master_dark,
    subtract_dark,
    reduce_calibration_frame,
    FringeProfile,
    PeakFit,
    QualityFlags,
    azimuthal_variance_centre,
    _variance_cost,
)

# ---------------------------------------------------------------------------
# Section 4 — Instrument / reduction parameters
# ---------------------------------------------------------------------------

R_MAX_PX                 = 110.0     # FlatSat confirmed (S12 §14)
N_BINS                   = 150       # S12 §14
N_SUBPIXELS              = 1         # S12 §14 (real data)
PEAK_DISTANCE            = 5         # S12 §10.3
PEAK_PROMINENCE          = 100.0     # ADU; S12 §10.3
PEAK_FIT_HALF_WINDOW     = 8         # S12 §10.3
MIN_PEAK_SEP_PX          = 3.0       # S12 §10.3
VAR_SEARCH_PX            = 15.0      # S12 §8
NEON_A_WAVELENGTH_NM     = 640.2248  # Burns et al. 1950
NEON_B_WAVELENGTH_NM     = 638.2991  # Burns et al. 1950
OI_WAVELENGTH_NM         = 630.0304  # rest wavelength (air)
NEON_AMPLITUDE_SPLIT_ADU = 1000.0    # FlatSat threshold (S13)

plt.rcParams["font.size"] = 11
plt.rcParams["figure.dpi"] = 150


# ---------------------------------------------------------------------------
# Section 5 — File selection
# ---------------------------------------------------------------------------

def select_npy_file(prompt: str) -> str:
    """Open a file-selection dialog and return the chosen path."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        print(f"\n{prompt}")
        path = filedialog.askopenfilename(
            title=prompt,
            filetypes=[("NumPy arrays", "*.npy"), ("All files", "*.*")],
        )
        root.destroy()
        if not path:
            raise FileNotFoundError("No file selected — aborting.")
        return path
    except Exception as exc:
        # Headless fallback
        print(f"\n{prompt}")
        print(f"  (tkinter unavailable: {exc})")
        path = input("  Enter path to .npy file: ").strip()
        if not path:
            raise FileNotFoundError("No path entered — aborting.")
        return path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gaussian(x, A, mu, sigma, B):
    """Gaussian + constant background."""
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + B


def _decode_quality_flags(flags: int) -> list:
    """Return list of flag name strings for non-zero quality flags."""
    names = {
        QualityFlags.STAGE1_FAILED:            "STAGE1_FAILED",
        QualityFlags.SEED_DISAGREEMENT:        "SEED_DISAGREEMENT",
        QualityFlags.SEED_FALLBACK_GEOMETRIC:  "SEED_FALLBACK_GEOMETRIC",
        QualityFlags.LOW_CONFIDENCE:           "LOW_CONFIDENCE",
        QualityFlags.CENTRE_FAILED:            "CENTRE_FAILED",
        QualityFlags.SPARSE_BINS:              "SPARSE_BINS",
        QualityFlags.CENTRE_JUMP:              "CENTRE_JUMP",
    }
    return [name for bit, name in names.items() if flags & bit]


def _baseline_adu(peak: PeakFit, fp: FringeProfile) -> float:
    """
    Estimate baseline (B0) for a peak from the 20th-percentile of its fit
    window, consistent with S12 §10.3.  Falls back to profile_raw - amplitude.
    """
    dr = float(np.median(np.diff(fp.r_grid)))
    hw = PEAK_FIT_HALF_WINDOW * dr
    mask = (~fp.masked) & np.isfinite(fp.sigma_profile)
    r_lo = peak.r_raw_px - hw
    r_hi = peak.r_raw_px + hw
    win = fp.profile[mask & (fp.r_grid >= r_lo) & (fp.r_grid <= r_hi)]
    if len(win) >= 2:
        return float(np.percentile(win, 20))
    return float(peak.profile_raw - peak.amplitude_adu)


# ---------------------------------------------------------------------------
# Figure (a) — Dark-subtracted image and histogram
# ---------------------------------------------------------------------------

def make_fig_a(
    img_ds: np.ndarray,
    fp: FringeProfile,
    output_dir: pathlib.Path,
    stem: str,
) -> pathlib.Path:
    """Dark-subtracted image (left) and pixel histogram (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Left — image
    ax = axes[0]
    vmin = float(np.percentile(img_ds, 1))
    vmax = float(np.percentile(img_ds, 99))
    im = ax.imshow(img_ds, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    ax.plot(fp.cx, fp.cy, "r+", markersize=12, markeredgewidth=2, label="Centre")
    ax.set_title("Dark-subtracted ROI")
    plt.colorbar(im, ax=ax, label="ADU")
    ax.legend(loc="upper right", fontsize=9)

    # Right — histogram
    ax2 = axes[1]
    counts, edges = np.histogram(img_ds.ravel(), bins=256)
    centres = 0.5 * (edges[:-1] + edges[1:])
    ax2.step(centres, counts, where="mid", color="steelblue", linewidth=0.8)
    mean_val   = float(np.mean(img_ds))
    median_val = float(np.median(img_ds))
    ax2.axvline(mean_val,   color="red",   linestyle="--", label=f"Mean {mean_val:.1f}")
    ax2.axvline(median_val, color="green", linestyle="--", label=f"Median {median_val:.1f}")
    ax2.set_xlabel("ADU")
    ax2.set_ylabel("Pixel count")
    ax2.set_title("Pixel histogram")
    ax2.legend(fontsize=9)

    fig.suptitle(
        f"Fig (a): Dark-subtracted calibration ROI\n"
        f"(dark_n_frames={fp.dark_n_frames},  "
        f"cx={fp.cx:.3f} px,  cy={fp.cy:.3f} px)"
    )
    fig.tight_layout()

    out_path = output_dir / f"{stem}_z00_figa.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Figure (b) — Coarse grid search
# ---------------------------------------------------------------------------

def make_fig_b(
    all_grid_cx: np.ndarray,
    all_grid_cy: np.ndarray,
    all_grid_cost: np.ndarray,
    cx_seed: float,
    cy_seed: float,
    grid_cx_best: float,
    grid_cy_best: float,
    fp: FringeProfile,
    output_dir: pathlib.Path,
    stem: str,
) -> pathlib.Path:
    """Coarse grid variance vs cx and cy."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    unique_cx = np.unique(all_grid_cx)
    unique_cy = np.unique(all_grid_cy)

    # Rebuild 2D cost grid [cy_idx, cx_idx]
    n_cy = len(unique_cy)
    n_cx = len(unique_cx)
    cost_2d = np.full((n_cy, n_cx), np.nan)
    for k in range(len(all_grid_cx)):
        ix = np.searchsorted(unique_cx, all_grid_cx[k])
        iy = np.searchsorted(unique_cy, all_grid_cy[k])
        cost_2d[iy, ix] = all_grid_cost[k]

    # Find row / column nearest to seed
    iy_seed = int(np.argmin(np.abs(unique_cy - cy_seed)))
    ix_seed = int(np.argmin(np.abs(unique_cx - cx_seed)))

    # Left — cost vs cx (row nearest cy_seed)
    ax = axes[0]
    ax.plot(unique_cx, cost_2d[iy_seed, :], "b-o", markersize=4)
    ax.axvline(cx_seed,      color="grey",  linestyle="--", label="Initial seed")
    ax.axvline(grid_cx_best, color="red",   linestyle="--", label="Grid minimum")
    ax.axvline(fp.cx,        color="blue",  linestyle="-",  label="Final (Nelder-Mead)")
    ax.set_xlabel("cx (px)")
    ax.set_ylabel("Azimuthal variance cost")
    ax.set_title("Coarse search: cost vs. cx")
    ax.legend(fontsize=9)

    # Right — cost vs cy (column nearest cx_seed)
    ax2 = axes[1]
    ax2.plot(unique_cy, cost_2d[:, ix_seed], "b-o", markersize=4)
    ax2.axvline(cy_seed,      color="grey",  linestyle="--", label="Initial seed")
    ax2.axvline(grid_cy_best, color="red",   linestyle="--", label="Grid minimum")
    ax2.axvline(fp.cy,        color="blue",  linestyle="-",  label="Final (Nelder-Mead)")
    ax2.set_xlabel("cy (px)")
    ax2.set_ylabel("Azimuthal variance cost")
    ax2.set_title("Coarse search: cost vs. cy")
    ax2.legend(fontsize=9)

    fig.suptitle(
        f"Fig (b): Coarse centre-finding grid search  "
        f"(search ±{VAR_SEARCH_PX:.1f} px around seed)"
    )
    fig.tight_layout()

    out_path = output_dir / f"{stem}_z00_figb.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Figure (c) — Nelder-Mead convergence
# ---------------------------------------------------------------------------

def make_fig_c(
    image_for_cost: np.ndarray,
    grid_cx_best: float,
    grid_cy_best: float,
    fp: FringeProfile,
    r_min_sq: float,
    r_max_sq: float,
    var_n_bins: int,
    output_dir: pathlib.Path,
    stem: str,
) -> pathlib.Path:
    """Nelder-Mead fine minimisation convergence."""
    nm_cx   = []
    nm_cy   = []
    nm_cost = []

    initial_cx = grid_cx_best
    initial_cy = grid_cy_best

    def cb(x):
        cost = _variance_cost(x[0], x[1], image_for_cost, r_min_sq, r_max_sq, var_n_bins)
        # Keep only evaluations within ±5 px of starting point
        if abs(x[0] - initial_cx) <= 5.0 and abs(x[1] - initial_cy) <= 5.0:
            nm_cx.append(float(x[0]))
            nm_cy.append(float(x[1]))
            nm_cost.append(float(cost))

    optimize.minimize(
        lambda p: _variance_cost(p[0], p[1], image_for_cost, r_min_sq, r_max_sq, var_n_bins),
        x0=[initial_cx, initial_cy],
        method="Nelder-Mead",
        options={"xatol": 0.02, "fatol": 1.0, "maxiter": 500},
        callback=cb,
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Left — cost vs iteration
    ax = axes[0]
    if nm_cost:
        ax.plot(range(len(nm_cost)), nm_cost, "b-o", markersize=4)
        ax.axhline(fp.cost_at_min, color="red", linestyle="--", label=f"Final cost {fp.cost_at_min:.2f}")
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "No callback data captured", transform=ax.transAxes, ha="center")
    ax.set_xlabel("Nelder-Mead iteration")
    ax.set_ylabel("Azimuthal variance cost")
    ax.set_title("Fine minimisation convergence")

    # Right — trajectory in (cx, cy)
    ax2 = axes[1]
    if nm_cx:
        ax2.plot(nm_cx, nm_cy, "b-", linewidth=1, zorder=1)
        ax2.scatter(nm_cx, nm_cy, s=15, c=range(len(nm_cx)),
                    cmap="viridis", zorder=2)
        ax2.plot(nm_cx[0],  nm_cy[0],  "go", markersize=10, label="Start (grid min)")
        ax2.plot(nm_cx[-1], nm_cy[-1], "r*", markersize=14, label="End")
    else:
        ax2.text(0.5, 0.5, "No trajectory data", transform=ax2.transAxes, ha="center")
    ax2.set_xlabel("cx (px)")
    ax2.set_ylabel("cy (px)")
    ax2.set_title("Centre trajectory (Nelder-Mead)")
    ax2.legend(fontsize=9)

    fig.suptitle(
        f"Fig (c): Nelder-Mead fine minimisation  "
        f"(cx={fp.cx:.4f},  cy={fp.cy:.4f},  "
        f"σ_cx={fp.sigma_cx:.4f},  σ_cy={fp.sigma_cy:.4f} px)"
    )
    fig.tight_layout()

    out_path = output_dir / f"{stem}_z00_figc.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Figure (d) — Annular reduction profile
# ---------------------------------------------------------------------------

def make_fig_d(
    fp: FringeProfile,
    output_dir: pathlib.Path,
    stem: str,
) -> pathlib.Path:
    """r²-binned annular reduction profile with SEM envelope and peak markers."""
    fig, ax = plt.subplots(figsize=(10, 5))

    good = ~fp.masked & np.isfinite(fp.sigma_profile)
    ax.plot(fp.r_grid[good], fp.profile[good], "b-", linewidth=1.2)
    ax.fill_between(
        fp.r_grid[good],
        (fp.profile - fp.sigma_profile)[good],
        (fp.profile + fp.sigma_profile)[good],
        alpha=0.25,
        color="steelblue",
        label="±1σ SEM",
    )

    for p in fp.peak_fits:
        ax.axvline(p.r_fit_px, color="red", linestyle="--", linewidth=0.8, alpha=0.8)

    ax.set_xlabel("Radius (px)")
    ax.set_ylabel("Mean intensity (ADU)")
    ax.set_title("Annular reduction profile")
    ax.text(
        0.98, 0.95,
        f"{len(fp.peak_fits)} peaks found",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=10,
    )
    ax.legend(fontsize=9)

    fig.suptitle(
        f"Fig (d): r²-binned annular reduction profile\n"
        f"(N_bins={fp.n_bins},  r_max={fp.r_max_px:.1f} px,  "
        f"n_subpixels={fp.n_subpixels})"
    )
    fig.tight_layout()

    out_path = output_dir / f"{stem}_z00_figd.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Figure (e) — Gaussian fits to individual peaks
# ---------------------------------------------------------------------------

def make_fig_e(
    fp: FringeProfile,
    peak_labels: list,
    output_dir: pathlib.Path,
    stem: str,
) -> pathlib.Path:
    """Per-peak Gaussian fit sub-panels."""
    n_peaks = len(fp.peak_fits)
    if n_peaks == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No peaks found", transform=ax.transAxes, ha="center")
        out_path = output_dir / f"{stem}_z00_fige.png"
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    ncols = math.ceil(math.sqrt(n_peaks))
    nrows = math.ceil(n_peaks / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.0 * nrows))
    axes_flat = np.array(axes).ravel()

    dr = float(np.median(np.diff(fp.r_grid)))
    hw = PEAK_FIT_HALF_WINDOW * dr

    for k, (p, label) in enumerate(zip(fp.peak_fits, peak_labels)):
        ax = axes_flat[k]
        good = ~fp.masked & np.isfinite(fp.sigma_profile)
        r_lo = p.r_raw_px - hw
        r_hi = p.r_raw_px + hw
        win_mask = good & (fp.r_grid >= r_lo) & (fp.r_grid <= r_hi)

        wr = fp.r_grid[win_mask]
        wp = fp.profile[win_mask]
        ws = fp.sigma_profile[win_mask]
        ws_safe = np.where(ws > 0, ws, 1.0)

        ax.errorbar(wr, wp, yerr=ws_safe, fmt="ko", markersize=3, capsize=2,
                    linewidth=0.8, elinewidth=0.6)

        # Overlay fit if fit_ok
        if not math.isnan(p.r_fit_px) and len(wr) > 0:
            r_fine = np.linspace(wr[0] if len(wr) else r_lo, wr[-1] if len(wr) else r_hi, 200)
            B0 = _baseline_adu(p, fp)
            try:
                y_fit = _gaussian(r_fine, p.amplitude_adu, p.r_fit_px, p.width_px, B0)
                color  = "red" if p.fit_ok else "orange"
                style  = "-"   if p.fit_ok else "--"
                label_fit = None if p.fit_ok else "FIT FAILED"
                ax.plot(r_fine, y_fit, color=color, linestyle=style,
                        linewidth=1.5, label=label_fit)
                if not p.fit_ok:
                    ax.legend(fontsize=7)
            except Exception:
                pass

        ax.axvline(p.r_fit_px, color="blue", linestyle=":", linewidth=1.0)
        ax.set_title(f"Ring {label}", fontsize=9)
        ax.set_xlabel("r (px)", fontsize=8)
        ax.set_ylabel("ADU", fontsize=8)
        ax.tick_params(labelsize=7)

    # Hide unused axes
    for k in range(n_peaks, len(axes_flat)):
        axes_flat[k].set_visible(False)

    fig.suptitle("Fig (e): Gaussian peak fits")
    fig.tight_layout()

    out_path = output_dir / f"{stem}_z00_fige.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Figure (f) — Peak fit results table
# ---------------------------------------------------------------------------

def make_fig_f(
    fp: FringeProfile,
    peaks_A: list,
    peaks_B: list,
    fringe_type: int,
    output_dir: pathlib.Path,
    stem: str,
) -> pathlib.Path:
    """Colour-coded peak fit results table."""
    # Build rows sorted by r_fit_px
    rows_data   = []
    row_colours = []

    if fringe_type == 1:
        col_headers = [
            "Neon-A (640.2248 nm)",
            "Neon-B (638.2991 nm)",
            "r_fit (px) ± 2σ",
            "r² (px²) ± 2σ",
            "Amplitude (ADU)",
            "Baseline (ADU)",
            "fit_ok",
        ]
        all_peaks = [(p, "A", i + 1) for i, p in enumerate(peaks_A)] + \
                    [(p, "B", i + 1) for i, p in enumerate(peaks_B)]
        all_peaks.sort(key=lambda x: x[0].r_fit_px)

        for p, fam, idx in all_peaks:
            r2  = p.r_fit_px ** 2
            two_sigma_r  = 2.0 * p.sigma_r_fit_px if not math.isnan(p.sigma_r_fit_px) else float("nan")
            sigma_r2     = 2.0 * p.r_fit_px * p.sigma_r_fit_px if not math.isnan(p.sigma_r_fit_px) else float("nan")
            two_sigma_r2 = 2.0 * sigma_r2 if not math.isnan(sigma_r2) else float("nan")
            B0           = _baseline_adu(p, fp)

            col_a = str(idx) if fam == "A" else ""
            col_b = str(idx) if fam == "B" else ""
            rows_data.append([
                col_a,
                col_b,
                f"{p.r_fit_px:.3f} ± {two_sigma_r:.3f}" if not math.isnan(two_sigma_r) else f"{p.r_fit_px:.3f} ± nan",
                f"{r2:.2f} ± {two_sigma_r2:.2f}" if not math.isnan(two_sigma_r2) else f"{r2:.2f} ± nan",
                f"{p.amplitude_adu:.1f}",
                f"{B0:.1f}",
                "✓" if p.fit_ok else "✗",
            ])
            row_colours.append("#ddeeff" if fam == "A" else "#fffacc")

    else:
        col_headers = [
            "OI (630.0304 nm air-rest)",
            "r_fit (px) ± 2σ",
            "r² (px²) ± 2σ",
            "Amplitude (ADU)",
            "Baseline (ADU)",
            "fit_ok",
        ]
        for i, p in enumerate(fp.peak_fits):
            r2  = p.r_fit_px ** 2
            two_sigma_r  = 2.0 * p.sigma_r_fit_px if not math.isnan(p.sigma_r_fit_px) else float("nan")
            sigma_r2     = 2.0 * p.r_fit_px * p.sigma_r_fit_px if not math.isnan(p.sigma_r_fit_px) else float("nan")
            two_sigma_r2 = 2.0 * sigma_r2 if not math.isnan(sigma_r2) else float("nan")
            B0           = _baseline_adu(p, fp)

            rows_data.append([
                str(i + 1),
                f"{p.r_fit_px:.3f} ± {two_sigma_r:.3f}" if not math.isnan(two_sigma_r) else f"{p.r_fit_px:.3f} ± nan",
                f"{r2:.2f} ± {two_sigma_r2:.2f}" if not math.isnan(two_sigma_r2) else f"{r2:.2f} ± nan",
                f"{p.amplitude_adu:.1f}",
                f"{B0:.1f}",
                "✓" if p.fit_ok else "✗",
            ])
            row_colours.append("white" if i % 2 == 0 else "#eeeeee")

    n_rows = len(rows_data)
    fig_f, ax_f = plt.subplots(figsize=(12, max(4, 0.4 * n_rows + 1.5)))
    ax_f.axis("off")

    if rows_data:
        tbl = ax_f.table(
            cellText  = rows_data,
            colLabels = col_headers,
            cellLoc   = "center",
            loc       = "center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.auto_set_column_width(col=list(range(len(col_headers))))

        for j in range(len(col_headers)):
            tbl[0, j].set_facecolor("#cccccc")

        for i, colour in enumerate(row_colours):
            for j in range(len(col_headers)):
                tbl[i + 1, j].set_facecolor(colour)

    fig_f.suptitle(
        f"Fig (f): Peak fit results\n"
        f"({len(fp.peak_fits)} peaks, {sum(p.fit_ok for p in fp.peak_fits)} fit_ok)"
    )
    fig_f.tight_layout()

    out_path = output_dir / f"{stem}_z00_figf.png"
    fig_f.savefig(out_path)
    plt.close(fig_f)
    return out_path


# ---------------------------------------------------------------------------
# Figure (g) — Melissinos Table 7.5 + Fig. 7.28 analogs
# ---------------------------------------------------------------------------

def make_fig_g(
    fp: FringeProfile,
    peaks_A: list,
    peaks_B: list,
    fringe_type: int,
    output_dir: pathlib.Path,
    stem: str,
) -> tuple:
    """
    Melissinos Table 7.5 analog (left) + Fig. 7.28 p vs r² scatter + OLS (right).
    Returns (out_path, r2_fit_A, r2_fit_B) where r2_fit is R² of OLS for each family.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ----------------------------------------------------------------
    # Left panel — table per family
    # ----------------------------------------------------------------
    ax_tbl = axes[0]
    ax_tbl.axis("off")

    def _make_table_data(peaks_list, label):
        rows = []
        n = len(peaks_list)
        for i, p in enumerate(peaks_list):
            r2_val   = p.r_fit_px ** 2
            delta_r2 = f"{peaks_list[i + 1].r_fit_px ** 2 - r2_val:.2f}" if i < n - 1 else ""
            rows.append([
                str(i + 1),
                f"{p.r_fit_px:.3f}",
                f"{r2_val:.2f}",
                delta_r2,
            ])
        return rows

    if fringe_type == 1:
        families = []
        if peaks_A:
            families.append((peaks_A, f"Neon-A  ({NEON_A_WAVELENGTH_NM} nm)"))
        if peaks_B:
            families.append((peaks_B, f"Neon-B  ({NEON_B_WAVELENGTH_NM} nm)"))
    else:
        families = [(fp.peak_fits, f"OI {OI_WAVELENGTH_NM} nm (air-rest)")]

    col_labels = ["p", "r_p (px)", "r²_p (px²)", "Δr²_{p,p+1} (px²)"]
    n_families = len(families)
    y_step    = 1.0 / (n_families + 1)

    for fi, (pk_list, fam_label) in enumerate(families):
        tbl_data = _make_table_data(pk_list, fam_label)
        if not tbl_data:
            continue
        y_pos = 1.0 - (fi + 0.5) * y_step - 0.05

        sub_tbl = ax_tbl.table(
            cellText  = tbl_data,
            colLabels = col_labels,
            cellLoc   = "center",
            bbox      = [0.0, y_pos - y_step * 0.45, 1.0, y_step * 0.85],
        )
        sub_tbl.auto_set_font_size(False)
        sub_tbl.set_fontsize(8)
        ax_tbl.text(
            0.5, y_pos + y_step * 0.42,
            fam_label,
            ha="center", va="bottom",
            fontsize=9, fontweight="bold",
            transform=ax_tbl.transAxes,
        )

    # ----------------------------------------------------------------
    # Right panel — p vs r² scatter with OLS fit
    # ----------------------------------------------------------------
    ax_scat = axes[1]
    r2_fits = {}

    if fringe_type == 1:
        plot_families = [
            (peaks_A, "Neon-A", "bo", "blue",  f"{NEON_A_WAVELENGTH_NM} nm"),
            (peaks_B, "Neon-B", "rs", "red",   f"{NEON_B_WAVELENGTH_NM} nm"),
        ]
    else:
        plot_families = [
            (fp.peak_fits, "OI", "g^", "green", f"{OI_WAVELENGTH_NM} nm"),
        ]

    legend_lines = []
    for pk_list, fam_name, marker, colour, wl_label in plot_families:
        if not pk_list:
            continue
        r2_vals = np.array([p.r_fit_px ** 2 for p in pk_list])
        p_vals  = np.arange(1, len(pk_list) + 1, dtype=float)

        ax_scat.scatter(r2_vals, p_vals, marker=marker[1], color=colour,
                        zorder=3, s=40, label=f"{fam_name} ({wl_label})")

        if len(r2_vals) >= 2:
            m, b   = np.polyfit(r2_vals, p_vals, 1)
            p_pred = m * r2_vals + b
            ss_res = float(np.sum((p_vals - p_pred) ** 2))
            ss_tot = float(np.sum((p_vals - np.mean(p_vals)) ** 2))
            r2_fit = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            r2_fits[fam_name] = r2_fit

            r2_line = np.linspace(r2_vals.min(), r2_vals.max(), 200)
            ax_scat.plot(r2_line, m * r2_line + b, color=colour, linestyle="--",
                         linewidth=1.5,
                         label=f"{fam_name}: m={m:.6f}, b={b:.4f}  (ε={1 - b % 1:.4f})")

    ax_scat.set_xlabel("r² (px²)")
    ax_scat.set_ylabel("Ring order p")
    ax_scat.set_title("p vs. r²  (Melissinos Fig. 7.28 analog)")
    ax_scat.legend(fontsize=8, loc="upper left")

    # R² annotation box
    r2_text = "\n".join(
        f"R²_fit ({fam}) = {val:.6f}" for fam, val in r2_fits.items()
    )
    ax_scat.text(
        0.98, 0.05, r2_text,
        transform=ax_scat.transAxes,
        ha="right", va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    fig.suptitle(
        "Fig (g): Melissinos Table 7.5 + Fig. 7.28 analogs\n"
        "(ring order p vs. r² demonstrates equal spacing in r²)"
    )
    fig.tight_layout()

    out_path = output_dir / f"{stem}_z00_figg.png"
    fig.savefig(out_path)
    plt.close(fig)

    return out_path, r2_fits


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """
    Processing chain per Z00 spec Section 6.
    Returns 0 on PASS, 1 on FAIL.
    """
    # ----------------------------------------------------------------
    # File selection
    # ----------------------------------------------------------------
    # Check if synthetic test mode requested (for smoke testing)
    synthetic_mode = "--synthetic" in sys.argv

    if synthetic_mode:
        print("\n[Synthetic mode: generating test arrays]")
        N = 256
        rng = np.random.default_rng(42)
        # Try to import synthesise_calibration_image from M02
        try:
            from src.fpi.m02_calibration_synthesis_2026_04_05 import (
                synthesise_calibration_image,
                InstrumentParams,
            )
            params = InstrumentParams()
            result = synthesise_calibration_image(params, add_noise=True,
                                                   rng=np.random.default_rng(0))
            cal_raw = result["image_2d"]
        except Exception:
            # Fallback: plain random arrays
            cal_raw = (rng.random((N, N)) * 500 + 200).astype(np.uint16)

        dark_raw = (rng.random((N, N)) * 50 + 100).astype(np.uint16)

        # Save to a temp dir
        import tempfile
        _tmpdir = tempfile.mkdtemp(prefix="z00_test_")
        cal_roi_path  = pathlib.Path(_tmpdir) / "synthetic_cal_roi.npy"
        dark_roi_path = pathlib.Path(_tmpdir) / "synthetic_dark_roi.npy"
        np.save(cal_roi_path,  cal_raw)
        np.save(dark_roi_path, dark_raw)
        print(f"  cal_roi  saved to: {cal_roi_path}")
        print(f"  dark_roi saved to: {dark_roi_path}")
    else:
        cal_roi_path  = pathlib.Path(select_npy_file("Select the CALIBRATION ROI .npy file"))
        dark_roi_path = pathlib.Path(select_npy_file("Select the DARK ROI .npy file"))

    cal_roi_stem = cal_roi_path.stem
    output_dir   = cal_roi_path.parent

    # ----------------------------------------------------------------
    # Load arrays
    # ----------------------------------------------------------------
    cal_roi  = np.load(str(cal_roi_path))
    dark_roi = np.load(str(dark_roi_path))
    print(f"\nLoaded cal_roi  shape={cal_roi.shape}  dtype={cal_roi.dtype}")
    print(f"Loaded dark_roi shape={dark_roi.shape}  dtype={dark_roi.dtype}")

    N         = cal_roi.shape[0]
    cx_seed   = (N - 1) / 2.0
    cy_seed   = (N - 1) / 2.0

    if N < 2 * R_MAX_PX:
        print(f"WARNING: ROI size {N}×{N} is smaller than 2×r_max "
              f"({2 * R_MAX_PX:.0f} px). Reducing r_max to {N/2 - 2:.0f} px.")
        r_max = N / 2.0 - 2.0
    else:
        r_max = R_MAX_PX

    # ----------------------------------------------------------------
    # Fringe-type prompt
    # ----------------------------------------------------------------
    if synthetic_mode:
        fringe_type = 1
        print("  [Synthetic mode: fringe_type=1 (neon)]")
    else:
        print("\nFringe type: [1] Neon calibration (2-line, 640.2 + 638.3 nm)")
        print("             [2] Airglow science  (1-line, 630.0 nm)")
        choice = input("Enter 1 or 2: ").strip()
        fringe_type = int(choice) if choice in ("1", "2") else 1

    # ----------------------------------------------------------------
    # Master dark
    # ----------------------------------------------------------------
    master_dark = make_master_dark([dark_roi])

    # ----------------------------------------------------------------
    # Pre-compute dark-subtracted image (for figure a)
    # ----------------------------------------------------------------
    img_ds = subtract_dark(cal_roi, master_dark, clip_negative=True)

    # ----------------------------------------------------------------
    # Reduction via production API
    # ----------------------------------------------------------------
    print("\nRunning reduce_calibration_frame() …")
    fp = reduce_calibration_frame(
        image                = cal_roi,
        master_dark          = master_dark,
        cx_human             = cx_seed,
        cy_human             = cy_seed,
        r_max_px             = r_max,
        n_bins               = N_BINS,
        n_subpixels          = N_SUBPIXELS,
        peak_distance        = PEAK_DISTANCE,
        peak_prominence      = PEAK_PROMINENCE,
        peak_fit_half_window = PEAK_FIT_HALF_WINDOW,
        min_peak_sep_px      = MIN_PEAK_SEP_PX,
        var_search_px        = VAR_SEARCH_PX,
    )

    # ----------------------------------------------------------------
    # Section 6.5 — Terminal summary
    # ----------------------------------------------------------------
    print("\n=== Z00 ANNULAR REDUCTION SUMMARY ===")
    print(f"  Input ROI     : {cal_roi_path}  shape={cal_roi.shape}")
    print(f"  Dark ROI      : {dark_roi_path}")
    print(f"  Dark subtracted: {fp.dark_subtracted}  ({fp.dark_n_frames} frame(s))")
    print(f"  Centre (fine) : cx={fp.cx:.4f} px,  cy={fp.cy:.4f} px")
    print(f"  Centre sigma  : σ_cx={fp.sigma_cx:.4f} px,  σ_cy={fp.sigma_cy:.4f} px")
    print(f"  Seed source   : {fp.seed_source}")
    print(f"  Quality flags : 0x{fp.quality_flags:02X}")
    print(f"  Cost at min   : {fp.cost_at_min:.4f}")
    # grid_cx_best / grid_cy_best captured below from the separate return_grid call
    print(f"  Peaks found   : {len(fp.peak_fits)}  ({sum(p.fit_ok for p in fp.peak_fits)} fit_ok)")
    print(f"  Profile bins  : {fp.n_bins}  (r_max={fp.r_max_px:.1f} px)")
    print(f"  Sparse bins   : {fp.sparse_bins}")
    print("=====================================")
    if fp.quality_flags != 0:
        flags_list = _decode_quality_flags(fp.quality_flags)
        print(f"  Quality flags decoded: {', '.join(flags_list)}")

    # ----------------------------------------------------------------
    # Section 7 — Peak-family separation
    # ----------------------------------------------------------------
    if fringe_type == 1:
        peaks_A = sorted(
            [p for p in fp.peak_fits if p.amplitude_adu >= NEON_AMPLITUDE_SPLIT_ADU],
            key=lambda p: p.r_fit_px,
        )
        peaks_B = sorted(
            [p for p in fp.peak_fits if p.amplitude_adu < NEON_AMPLITUDE_SPLIT_ADU],
            key=lambda p: p.r_fit_px,
        )
        print(f"\n  Neon-A peaks: {len(peaks_A)}  (≥ {NEON_AMPLITUDE_SPLIT_ADU:.0f} ADU)")
        print(f"  Neon-B peaks: {len(peaks_B)}  (< {NEON_AMPLITUDE_SPLIT_ADU:.0f} ADU)")
    else:
        peaks_A = []
        peaks_B = []
        peaks_OI = sorted(fp.peak_fits, key=lambda p: p.r_fit_px)

    # ----------------------------------------------------------------
    # Build peak labels for figure (e)
    # ----------------------------------------------------------------
    if fringe_type == 1:
        a_labels = {id(p): f"A{i+1}" for i, p in enumerate(peaks_A)}
        b_labels = {id(p): f"B{i+1}" for i, p in enumerate(peaks_B)}
        all_sorted = sorted(fp.peak_fits, key=lambda p: p.r_fit_px)
        peak_labels = [
            a_labels.get(id(p), b_labels.get(id(p), f"?{i+1}"))
            for i, p in enumerate(all_sorted)
        ]
    else:
        all_sorted  = sorted(fp.peak_fits, key=lambda p: p.r_fit_px)
        peak_labels = [f"OI{i+1}" for i in range(len(all_sorted))]

    # ----------------------------------------------------------------
    # Capture coarse grid data (for figure b)
    # ----------------------------------------------------------------
    print("\nCapturing coarse grid data for Fig (b) …")
    p99_5          = float(np.percentile(img_ds, 99.5))
    image_for_cost = np.clip(img_ds, None, p99_5)
    var_r_max_px   = r_max

    grid_result = azimuthal_variance_centre(
        image_for_cost,
        cx_seed,
        cy_seed,
        var_r_max_px  = var_r_max_px,
        var_search_px = VAR_SEARCH_PX,
        return_grid   = True,
    )
    _, _, _, grid_cx_best, grid_cy_best, _, all_grid_cx, all_grid_cy, all_grid_cost = grid_result
    print(f"  Grid seed cx  : {grid_cx_best:.4f},  grid seed cy={grid_cy_best:.4f}")

    r_min_sq   = 5.0 ** 2
    r_max_sq   = var_r_max_px ** 2
    var_n_bins = 250

    # ----------------------------------------------------------------
    # Generate figures
    # ----------------------------------------------------------------
    print("\nGenerating figures …")
    saved_paths = {}

    saved_paths["a"] = make_fig_a(img_ds, fp, output_dir, cal_roi_stem)
    print(f"  (a) saved: {saved_paths['a'].name}  "
          f"({saved_paths['a'].stat().st_size // 1024} KB)")

    saved_paths["b"] = make_fig_b(
        all_grid_cx, all_grid_cy, all_grid_cost,
        cx_seed, cy_seed, grid_cx_best, grid_cy_best,
        fp, output_dir, cal_roi_stem,
    )
    print(f"  (b) saved: {saved_paths['b'].name}  "
          f"({saved_paths['b'].stat().st_size // 1024} KB)")

    saved_paths["c"] = make_fig_c(
        image_for_cost, grid_cx_best, grid_cy_best, fp,
        r_min_sq, r_max_sq, var_n_bins,
        output_dir, cal_roi_stem,
    )
    print(f"  (c) saved: {saved_paths['c'].name}  "
          f"({saved_paths['c'].stat().st_size // 1024} KB)")

    saved_paths["d"] = make_fig_d(fp, output_dir, cal_roi_stem)
    print(f"  (d) saved: {saved_paths['d'].name}  "
          f"({saved_paths['d'].stat().st_size // 1024} KB)")

    saved_paths["e"] = make_fig_e(fp, peak_labels, output_dir, cal_roi_stem)
    print(f"  (e) saved: {saved_paths['e'].name}  "
          f"({saved_paths['e'].stat().st_size // 1024} KB)")

    saved_paths["f"] = make_fig_f(
        fp, peaks_A, peaks_B, fringe_type, output_dir, cal_roi_stem,
    )
    print(f"  (f) saved: {saved_paths['f'].name}  "
          f"({saved_paths['f'].stat().st_size // 1024} KB)")

    g_path, r2_fits = make_fig_g(
        fp, peaks_A, peaks_B, fringe_type, output_dir, cal_roi_stem,
    )
    saved_paths["g"] = g_path
    print(f"  (g) saved: {saved_paths['g'].name}  "
          f"({saved_paths['g'].stat().st_size // 1024} KB)")

    # ----------------------------------------------------------------
    # Section 10 — Save structured peak table
    # ----------------------------------------------------------------
    dtype_cal = np.dtype([
        ("family",            "U8"),
        ("ring_index",        "i4"),
        ("r_fit_px",          "f8"),
        ("sigma_r_fit_px",    "f8"),
        ("two_sigma_r_px",    "f8"),
        ("r2_fit_px2",        "f8"),
        ("sigma_r2_px2",      "f8"),
        ("two_sigma_r2_px2",  "f8"),
        ("amplitude_adu",     "f8"),
        ("baseline_adu",      "f8"),
        ("width_px",          "f8"),
        ("fit_ok",            "bool"),
        ("peak_idx",          "i4"),
        ("r_raw_px",          "f8"),
    ])

    def _build_rows(pk_list, family_str):
        rows = []
        for i, p in enumerate(pk_list):
            sig_r  = p.sigma_r_fit_px if not math.isnan(p.sigma_r_fit_px) else 0.0
            sig_r2 = 2.0 * p.r_fit_px * sig_r
            B0     = _baseline_adu(p, fp)
            rows.append((
                family_str,
                i + 1,
                p.r_fit_px,
                p.sigma_r_fit_px,
                2.0 * p.sigma_r_fit_px if not math.isnan(p.sigma_r_fit_px) else float("nan"),
                p.r_fit_px ** 2,
                sig_r2,
                2.0 * sig_r2,
                p.amplitude_adu,
                B0,
                p.width_px,
                p.fit_ok,
                p.peak_idx,
                p.r_raw_px,
            ))
        return rows

    if fringe_type == 1:
        all_rows = _build_rows(peaks_A, "neon_A") + _build_rows(peaks_B, "neon_B")
    else:
        all_rows = _build_rows(fp.peak_fits, "oi")

    # Sort by r_fit_px, A before B on tie
    all_rows.sort(key=lambda r: r[2])

    peak_table = np.array(all_rows, dtype=dtype_cal)
    npy_out    = output_dir / f"{cal_roi_stem}_fringe_peaks.npy"
    np.save(str(npy_out), peak_table)
    print(f"\nSaved peak table: {npy_out}")
    print(f"  {len(peak_table)} rows × {len(peak_table.dtype.names)} columns")

    # ----------------------------------------------------------------
    # Section 11 — Acceptance criteria
    # ----------------------------------------------------------------
    print("\n=== ACCEPTANCE CHECK ===")
    failures = []

    if not fp.dark_subtracted:
        failures.append("dark_subtracted is False")
    if fp.dark_n_frames < 1:
        failures.append(f"dark_n_frames={fp.dark_n_frames} < 1")
    if fp.quality_flags != 0x00:
        failures.append(f"quality_flags=0x{fp.quality_flags:02X} (not GOOD)")
    if fp.sigma_cx >= 0.5:
        failures.append(f"sigma_cx={fp.sigma_cx:.4f} px >= 0.5 px")
    if fp.sigma_cy >= 0.5:
        failures.append(f"sigma_cy={fp.sigma_cy:.4f} px >= 0.5 px")

    n_peaks  = len(fp.peak_fits)
    n_fit_ok = sum(p.fit_ok for p in fp.peak_fits)

    if fringe_type == 1:
        if n_peaks != 20:
            failures.append(f"peaks found={n_peaks} (expected 20)")
        if n_fit_ok < 18:
            failures.append(f"fit_ok={n_fit_ok} < 18 required")
        if len(peaks_A) != 10:
            failures.append(f"Neon-A peaks={len(peaks_A)} (expected 10)")
        if len(peaks_B) != 10:
            failures.append(f"Neon-B peaks={len(peaks_B)} (expected 10)")
        for fam, r2_fit in r2_fits.items():
            if math.isnan(r2_fit) or r2_fit < 0.9999:
                failures.append(f"R²_fit ({fam}) = {r2_fit:.6f} < 0.9999")
    else:
        if n_peaks < 4:
            failures.append(f"peaks found={n_peaks} < 4 required")
        if n_fit_ok < n_peaks:
            failures.append(f"fit_ok={n_fit_ok} < {n_peaks} (all required for airglow)")

    if not npy_out.exists() or npy_out.stat().st_size == 0:
        failures.append(f"Peak table not saved or empty: {npy_out}")
    else:
        expected_rows = len(fp.peak_fits)
        if len(peak_table) != expected_rows:
            failures.append(f"Peak table rows={len(peak_table)}, expected {expected_rows}")

    for letter, path in saved_paths.items():
        if not path.exists() or path.stat().st_size == 0:
            failures.append(f"Figure ({letter}) not saved or empty: {path}")

    if failures:
        print("FAIL")
        for f in failures:
            print(f"  - {f}")
        exit_code = 1
    else:
        print("PASS")
        exit_code = 0

    print("========================")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
