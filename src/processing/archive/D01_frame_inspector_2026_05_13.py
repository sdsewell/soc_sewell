"""
D01_frame_inspector_2026_05_13.py  —  WindCube FPI frame inspector (v2).

Loads one science and one calibration .bin frame from a GEN01 output
bin_frames/ directory, subtracts a master dark (averaged from all *dark*.bin
files in the same directory), locates the fringe centre with Nelder-Mead
azimuthal-variance minimisation, and produces two diagnostic figures.

Figure 1 (4 panels):  sci image | cal image | master dark | param table
Figure 2 (3 panels):  NM convergence | dark-subtracted sci profile |
                       dark-subtracted cal profile + Airy overlay

Usage
-----
    python src/processing/D01_frame_inspector_2026_05_13.py [bin_dir]

If bin_dir is omitted, a folder picker dialog opens.
"""

from __future__ import annotations

import pathlib
import sys
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

# ── repo root on sys.path ─────────────────────────────────────────────────────
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Binary layout — 2×2 binned (143 520 bytes) ──────────────────────────────
_N_ROWS   = 260
_N_COLS   = 276
_SCI_R0   = 1       # science region: row offset (0-based)
_SCI_C0   = 10      # science region: col offset (0-based)
_SCI_ROWS = 256
_SCI_COLS = 256
_N_HDR    = 276     # header words


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_bin_frame(path: pathlib.Path) -> dict:
    """
    Read a WindCube FPI 2×2-binned binary frame.

    Returns dict:
      image      : float32 (256, 256) — science pixel region
      header     : uint16 (276,)      — raw header words
      filename   : str
      path       : pathlib.Path
      img_type   : str ('science' | 'cal' | 'dark')
      exp_time_s : float
      ccd_temp_c : float
      lua_ts_ms  : int
    """
    import struct

    path = pathlib.Path(path)
    raw = np.frombuffer(path.read_bytes(), dtype=">u2")

    expected = _N_ROWS * _N_COLS
    if raw.size != expected:
        raise ValueError(
            f"read_bin_frame: expected {expected} uint16 words, "
            f"got {raw.size} from {path.name}"
        )

    header = raw[:_N_HDR].copy()
    pixels = raw[_N_COLS:].reshape(_N_ROWS - 1, _N_COLS)
    sci    = pixels[_SCI_R0 : _SCI_R0 + _SCI_ROWS,
                    _SCI_C0 : _SCI_C0 + _SCI_COLS].astype(np.float32)

    # ── decode header fields ──────────────────────────────────────────────────
    def _u64(w):
        return sum(int(header[w + i]) << (16 * i) for i in range(4))

    def _f64(w):
        b = struct.pack(">4H", *reversed([header[w + i] for i in range(4)]))
        return struct.unpack(">d", b)[0]

    lua_ts_ms  = _u64(8)
    exp_time_s = float(header[2]) / 100.0
    ccd_temp_c = _f64(4)
    gpio       = [int(header[100 + i]) & 0xFF for i in range(4)]
    lamps      = [int(header[104 + i]) & 0xFF for i in range(6)]
    shutter    = "closed" if (gpio[0] == 1 and gpio[3] == 1) else "open"
    if any(lamps):
        img_type = "cal"
    elif shutter == "closed":
        img_type = "dark"
    else:
        img_type = "science"

    return {
        "image":      sci,
        "header":     header,
        "filename":   path.name,
        "path":       path,
        "img_type":   img_type,
        "exp_time_s": exp_time_s,
        "ccd_temp_c": ccd_temp_c,
        "lua_ts_ms":  lua_ts_ms,
    }


def load_master_dark(bin_dir: pathlib.Path) -> dict:
    """
    Find all *dark*.bin files in bin_dir, read each with read_bin_frame(),
    stack images, and return their mean as the master dark.

    Returns dict:
      image      : float32 (256, 256) — mean dark frame
      n_frames   : int
      dark_paths : list[pathlib.Path]
      mean_adu   : float
      std_adu    : float
      max_adu    : float
      min_adu    : float
    """
    dark_files = sorted(pathlib.Path(bin_dir).glob("*dark*.bin"))
    if not dark_files:
        print("WARNING: No dark frames found in bin_dir. Dark subtraction skipped.")
        return {
            "image":      np.zeros((_SCI_ROWS, _SCI_COLS), dtype=np.float32),
            "n_frames":   0,
            "dark_paths": [],
            "mean_adu":   0.0,
            "std_adu":    0.0,
            "max_adu":    0.0,
            "min_adu":    0.0,
        }
    stack  = np.stack([read_bin_frame(f)["image"] for f in dark_files])
    master = stack.mean(axis=0).astype(np.float32)
    return {
        "image":      master,
        "n_frames":   len(dark_files),
        "dark_paths": dark_files,
        "mean_adu":   float(master.mean()),
        "std_adu":    float(master.std()),
        "max_adu":    float(master.max()),
        "min_adu":    float(master.min()),
    }


def dark_subtract(frame_image: np.ndarray, dark_image: np.ndarray) -> np.ndarray:
    """
    Subtract master dark from frame.  Clip to 0 (no negative pixels).
    Returns float32 same shape as input.

    Per S12: dark subtraction is the single authoritative step performed
    before any annular reduction or fitting.
    """
    return np.clip(
        frame_image.astype(np.float32) - dark_image.astype(np.float32),
        0.0, None,
    )


# ---------------------------------------------------------------------------
# Centre finding
# ---------------------------------------------------------------------------

def _az_variance_cost(
    cx: float,
    cy: float,
    image: np.ndarray,
    r_max_sq: float,
    n_bins: int,
) -> float:
    """Sum of per-bin azimuthal intensity variance over the annular region."""
    H, W   = image.shape
    rc, cc = np.mgrid[0:H, 0:W]
    r2     = (rc.astype(np.float64) - cy) ** 2 + (cc.astype(np.float64) - cx) ** 2
    mask   = r2 < r_max_sq
    if mask.sum() < n_bins:
        return 1e30
    r2v    = r2[mask]
    aduv   = image[mask].astype(np.float64)
    dr2    = r_max_sq / n_bins
    bidx   = np.clip(np.floor(r2v / dr2).astype(np.int32), 0, n_bins - 1)
    cnt    = np.bincount(bidx, minlength=n_bins).astype(np.float64)
    sumA   = np.bincount(bidx, weights=aduv,    minlength=n_bins)
    sumA2  = np.bincount(bidx, weights=aduv**2, minlength=n_bins)
    good   = cnt >= 2
    var    = np.where(good, sumA2 / np.where(good, cnt, 1.0)
                            - (sumA / np.where(good, cnt, 1.0)) ** 2, 0.0)
    return float(var.sum())


def find_center_nelder_mead(
    image: np.ndarray,
    r_max_px: float = 110.0,
    n_bins: int = 500,
) -> tuple:
    """
    Locate the FPI fringe centre using two-pass azimuthal variance minimisation.

    Pass 1: 20×20 coarse grid search over ±15 px from image centre.
    Pass 2: Nelder-Mead from the best grid point.

    Returns
    -------
    (cx_opt, cy_opt, nm_result, cost_history)
      cx_opt, cy_opt : float  — sub-pixel centre coordinates (column, row)
      nm_result      : scipy OptimizeResult
      cost_history   : list[float] — cost at each NM function evaluation
    """
    H, W       = image.shape
    cx_seed    = (W - 1) / 2.0
    cy_seed    = (H - 1) / 2.0
    r_max_sq   = r_max_px ** 2

    # Clip hot pixels so they don't bias the variance minimum
    p99 = float(np.percentile(image, 99.5))
    img_c = np.clip(image, None, p99)

    def cost(xy: np.ndarray) -> float:
        return _az_variance_cost(xy[0], xy[1], img_c, r_max_sq, n_bins)

    # ── Pass 1: coarse grid ──────────────────────────────────────────────────
    search_half = 15.0
    offsets = np.linspace(-search_half, search_half, 20)
    best_c, best_cx, best_cy = np.inf, cx_seed, cy_seed
    for dy in offsets:
        for dx in offsets:
            c = cost([cx_seed + dx, cy_seed + dy])
            if c < best_c:
                best_c, best_cx, best_cy = c, cx_seed + dx, cy_seed + dy

    # ── Pass 2: Nelder-Mead from grid best ──────────────────────────────────
    cost_history: list[float] = []

    def cost_tracked(xy: np.ndarray) -> float:
        c = cost(xy)
        cost_history.append(c)
        return c

    step = float(offsets[1] - offsets[0]) + 0.5
    x0   = np.array([best_cx, best_cy])
    nm_result = minimize(
        cost_tracked, x0,
        method="Nelder-Mead",
        options={
            "initial_simplex": np.array([x0, x0 + [step, 0.0], x0 + [0.0, step]]),
            "xatol": 0.02,
            "fatol": 1.0,
            "maxiter": 500,
        },
    )

    cx_opt = float(nm_result.x[0])
    cy_opt = float(nm_result.x[1])
    return cx_opt, cy_opt, nm_result, cost_history


# ---------------------------------------------------------------------------
# Radial profile
# ---------------------------------------------------------------------------

def annular_mean(
    image: np.ndarray,
    cx: float,
    cy: float,
    r_max_px: float = 110.0,
    n_bins: int = 500,
) -> tuple:
    """
    Compute a simple r²-binned radial intensity profile.

    Returns (r_grid_px, mean_adu) both shape (n_bins,).
    Bins with no pixels have mean_adu = NaN.
    """
    H, W    = image.shape
    r2_max  = r_max_px ** 2
    r2_edges = np.linspace(0.0, r2_max, n_bins + 1)
    r2_grid  = 0.5 * (r2_edges[:-1] + r2_edges[1:])
    r_grid   = np.sqrt(r2_grid)
    dr2      = r2_max / n_bins

    rc, cc  = np.mgrid[0:H, 0:W]
    r2      = (rc.astype(np.float64) - cy) ** 2 + (cc.astype(np.float64) - cx) ** 2
    mask    = r2 < r2_max
    r2v     = r2[mask]
    aduv    = image[mask].astype(np.float64)
    bidx    = np.clip(np.floor(r2v / dr2).astype(np.int32), 0, n_bins - 1)

    cnt     = np.bincount(bidx, minlength=n_bins).astype(np.float64)
    sumA    = np.bincount(bidx, weights=aduv, minlength=n_bins)
    mean_adu = np.where(cnt > 0, sumA / cnt, np.nan)
    return r_grid.astype(np.float32), mean_adu.astype(np.float32)


# ---------------------------------------------------------------------------
# Figure 1 — frame overview
# ---------------------------------------------------------------------------

def make_figure1(
    sci: dict,
    cal: dict,
    dark: dict,
    sci_ds: np.ndarray,
    cal_ds: np.ndarray,
    cx_opt: float,
    cy_opt: float,
    folder: pathlib.Path,
    timestamp: str,
) -> None:
    """
    4-panel overview: sci image | cal image | master dark | param table.

    Saves figure to folder as D01_fig1_{timestamp}.png and shows it.
    """
    fig, axes = plt.subplots(
        1, 4,
        figsize=(19, 7),
        gridspec_kw={"width_ratios": [3, 3, 3, 4]},
    )
    fig.suptitle(
        f"D01 Frame Inspector  —  {timestamp}",
        fontsize=12, fontweight="bold",
    )

    def _show_image(ax, img, title, cmap="gray", cx=None, cy=None, marker_color="r"):
        vlo = float(np.percentile(img, 1))
        vhi = float(np.percentile(img, 99))
        im  = ax.imshow(img, cmap=cmap, origin="upper", vmin=vlo, vmax=vhi,
                        aspect="equal", interpolation="nearest")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("Column (px)", fontsize=7)
        ax.set_ylabel("Row (px)",    fontsize=7)
        ax.tick_params(labelsize=6)
        if cx is not None and cy is not None:
            ax.plot(cx, cy, "+", color=marker_color, markersize=14,
                    markeredgewidth=1.5, label=f"NM cx={cx:.1f}\ncy={cy:.1f}")

    # ── Col 0: science image (NM centre = red '+') ───────────────────────────
    _show_image(
        axes[0], sci["image"],
        f"Science  ({sci['img_type']})\n"
        f"exp={sci['exp_time_s']:.1f}s  T={sci['ccd_temp_c']:.1f}°C",
        cx=cx_opt, cy=cy_opt, marker_color="r",
    )

    # ── Col 1: cal image (NM centre from dark-sub cal = green '+') ───────────
    _show_image(
        axes[1], cal["image"],
        f"Calibration  ({cal['img_type']})\n"
        f"exp={cal['exp_time_s']:.1f}s  T={cal['ccd_temp_c']:.1f}°C",
        cx=cx_opt, cy=cy_opt, marker_color="g",
    )

    # ── Col 2: master dark (inferno to show thermal structure) ───────────────
    if dark["n_frames"] > 0:
        vlo_d = float(np.percentile(dark["image"], 1))
        vhi_d = float(np.percentile(dark["image"], 99))
    else:
        vlo_d, vhi_d = 0.0, 1.0
    im_d = axes[2].imshow(
        dark["image"], cmap="inferno", origin="upper",
        vmin=vlo_d, vmax=vhi_d, aspect="equal", interpolation="nearest",
    )
    plt.colorbar(im_d, ax=axes[2], fraction=0.046, pad=0.04)
    axes[2].set_title(
        f"Master dark\n({dark['n_frames']} frames averaged)",
        fontsize=8,
    )
    axes[2].set_xlabel("Column (px)", fontsize=7)
    axes[2].set_ylabel("Row (px)",    fontsize=7)
    axes[2].tick_params(labelsize=6)
    axes[2].text(
        0.02, 0.98,
        f"mean={dark['mean_adu']:.2f}\nstd={dark['std_adu']:.2f}\n"
        f"max={dark['max_adu']:.2f}\nmin={dark['min_adu']:.2f}",
        transform=axes[2].transAxes,
        fontsize=7, color="white", va="top", ha="left",
        bbox=dict(facecolor="black", alpha=0.5, pad=2),
    )

    # ── Col 3: parameter table ────────────────────────────────────────────────
    axes[3].axis("off")

    rows = [
        # ── Science frame ──
        ["── Science ──",    ""],
        ["Filename",         sci["filename"]],
        ["Image type",       sci["img_type"]],
        ["Exp time",         f"{sci['exp_time_s']:.2f} s"],
        ["CCD temp",         f"{sci['ccd_temp_c']:.2f} °C"],
        ["Raw peak",         f"{sci['image'].max():.0f} ADU"],
        ["Raw mean",         f"{sci['image'].mean():.2f} ADU"],
        # ── Calibration frame ──
        ["── Calibration ──", ""],
        ["Filename",         cal["filename"]],
        ["Image type",       cal["img_type"]],
        ["Exp time",         f"{cal['exp_time_s']:.2f} s"],
        ["CCD temp",         f"{cal['ccd_temp_c']:.2f} °C"],
        ["Raw peak",         f"{cal['image'].max():.0f} ADU"],
        ["Raw mean",         f"{cal['image'].mean():.2f} ADU"],
        # ── NM centre ──
        ["── NM Centre (cal DS) ──", ""],
        ["cx_opt",           f"{cx_opt:.3f} px"],
        ["cy_opt",           f"{cy_opt:.3f} px"],
        # ── Dark subtraction ──
        ["── Dark subtraction ──", ""],
        ["Dark frames averaged",  str(dark["n_frames"])],
        ["Master dark mean",      f"{dark['mean_adu']:.2f} ADU"],
        ["Master dark std",       f"{dark['std_adu']:.2f} ADU"],
        ["Master dark max",       f"{dark['max_adu']:.2f} ADU"],
        ["Sci DS peak",           f"{sci_ds.max():.0f} ADU"],
        ["Sci DS mean",           f"{sci_ds.mean():.2f} ADU"],
        ["Cal DS peak",           f"{cal_ds.max():.0f} ADU"],
        ["Cal DS mean",           f"{cal_ds.mean():.2f} ADU"],
        ["Expected floor (DS)",   "0 ADU  (clipped; dark subtracted)"],
    ]

    tbl = axes[3].table(
        cellText=[[r[0], r[1]] for r in rows],
        colLabels=["Parameter", "Value"],
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1.0, 1.35)

    _HDR_BG  = "#2C3E50"
    _SEC_BG  = "#D5E8D4"
    _ALT_BG  = "#EBF5FB"
    for (r_idx, c_idx), cell in tbl.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        cell.set_linewidth(0.5)
        if r_idx == 0:
            cell.set_facecolor(_HDR_BG)
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            label = rows[r_idx - 1][0]
            if label.startswith("──"):
                cell.set_facecolor(_SEC_BG)
                cell.get_text().set_fontweight("bold")
            elif r_idx % 2 == 0:
                cell.set_facecolor(_ALT_BG)

    axes[3].set_title("Frame Parameters", fontsize=9, fontweight="bold", pad=8)

    fig.tight_layout()
    out = pathlib.Path(folder) / f"D01_fig1_{timestamp}.png"
    try:
        fig.savefig(str(out), dpi=150, bbox_inches="tight")
        print(f"  Figure 1 saved: {out}")
    except Exception as exc:
        print(f"  WARNING: Could not save Figure 1: {exc}")
    plt.show()


# ---------------------------------------------------------------------------
# Figure 2 — radial profiles
# ---------------------------------------------------------------------------

def make_figure2(
    sci: dict,
    cal: dict,
    sci_ds: np.ndarray,
    cal_ds: np.ndarray,
    dark: dict,
    cx_opt: float,
    cy_opt: float,
    cost_history: list,
    folder: pathlib.Path,
    timestamp: str,
) -> None:
    """
    3-panel profile figure:
      Col 1: Nelder-Mead convergence (dark-subtracted cal)
      Col 2: dark-subtracted science radial profile
      Col 3: dark-subtracted cal radial profile + Airy model overlay

    Saves to folder/D01_fig2_{timestamp}.png.
    """
    r_sci, p_sci = annular_mean(sci_ds, cx_opt, cy_opt)
    r_cal, p_cal = annular_mean(cal_ds, cx_opt, cy_opt)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"D01 Radial Profiles (dark-subtracted)  —  {timestamp}",
        fontsize=12, fontweight="bold",
    )

    # ── Col 0: NM convergence ────────────────────────────────────────────────
    n = len(cost_history)
    if n > 0:
        axes[0].plot(
            np.arange(1, n + 1), cost_history,
            color="steelblue", linewidth=0.9, alpha=0.8,
        )
        axes[0].axhline(
            cost_history[-1], color="red", linewidth=1.2, linestyle="--",
            label=f"final cost = {cost_history[-1]:.4g}",
        )
    axes[0].set_title(
        f"Nelder-Mead center search (dark-subtracted cal)\n({n} evaluations)",
        fontsize=9,
    )
    axes[0].set_xlabel("Evaluation #", fontsize=8)
    axes[0].set_ylabel("Azimuthal variance cost", fontsize=8)
    axes[0].tick_params(labelsize=7)
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.2)
    axes[0].text(
        0.02, 0.98,
        f"cx={cx_opt:.3f} px\ncy={cy_opt:.3f} px",
        transform=axes[0].transAxes, fontsize=7, va="top", ha="left",
        bbox=dict(facecolor="white", alpha=0.7, pad=2),
    )

    # ── Col 1: dark-subtracted science profile ───────────────────────────────
    good_sci = np.isfinite(p_sci)
    axes[1].plot(
        r_sci[good_sci], p_sci[good_sci],
        color="steelblue", linewidth=0.9, alpha=0.85,
        label="DS science",
    )
    axes[1].set_title(
        "Dark-subtracted science profile\n(NM centre)",
        fontsize=9,
    )
    axes[1].set_xlabel("Radius (px)", fontsize=8)
    axes[1].set_ylabel("Mean intensity (ADU)", fontsize=8)
    axes[1].tick_params(labelsize=7)
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.2)
    axes[1].text(
        0.98, 0.97,
        f"Raw peak: {sci['image'].max():.0f} ADU\n"
        f"DS peak:  {sci_ds.max():.0f} ADU\n"
        f"Dark mean: {dark['mean_adu']:.2f} ADU",
        transform=axes[1].transAxes, fontsize=7, va="top", ha="right",
        bbox=dict(facecolor="white", alpha=0.7, pad=2),
    )

    # ── Col 2: dark-subtracted cal profile + Airy model overlay ─────────────
    good_cal = np.isfinite(p_cal)
    axes[2].plot(
        r_cal[good_cal], p_cal[good_cal],
        color="steelblue", linewidth=0.9, alpha=0.85,
        label="DS cal",
    )

    # Airy model overlay: scale to data via linear least squares
    try:
        from src.fpi.airy_forward_model_2026_05_05 import (
            airy_modified,
            InstrumentParams,
            NE_WAVELENGTH_1_M,
        )
        params      = InstrumentParams()
        r_good      = r_cal[good_cal].astype(np.float64)
        airy_vals   = airy_modified(r_good, NE_WAVELENGTH_1_M, params)
        # Linear least squares: p_cal ≈ A * airy + B
        airy_max    = float(airy_vals.max()) if airy_vals.max() > 0 else 1.0
        airy_n      = airy_vals / airy_max
        data_n      = p_cal[good_cal].astype(np.float64)
        A_mat       = np.column_stack([airy_n, np.ones_like(airy_n)])
        coeff, _, _, _ = np.linalg.lstsq(A_mat, data_n, rcond=None)
        model_fit   = coeff[0] * airy_n + coeff[1]
        axes[2].plot(
            r_good, model_fit,
            color="darkorange", linewidth=1.4, linestyle="--",
            label="Airy model (Ne 640.2 nm)",
            alpha=0.9,
        )
    except Exception as exc:
        print(f"  NOTE: Airy overlay skipped: {exc}")

    axes[2].set_title(
        "Dark-subtracted cal profile\n(NM centre)",
        fontsize=9,
    )
    axes[2].set_xlabel("Radius (px)", fontsize=8)
    axes[2].set_ylabel("Mean intensity (ADU)", fontsize=8)
    axes[2].tick_params(labelsize=7)
    axes[2].legend(fontsize=7)
    axes[2].grid(True, alpha=0.2)
    axes[2].text(
        0.98, 0.97,
        f"Raw peak: {cal['image'].max():.0f} ADU\n"
        f"DS peak:  {cal_ds.max():.0f} ADU",
        transform=axes[2].transAxes, fontsize=7, va="top", ha="right",
        bbox=dict(facecolor="white", alpha=0.7, pad=2),
    )

    fig.tight_layout()
    out = pathlib.Path(folder) / f"D01_fig2_{timestamp}.png"
    try:
        fig.savefig(str(out), dpi=150, bbox_inches="tight")
        print(f"  Figure 2 saved: {out}")
    except Exception as exc:
        print(f"  WARNING: Could not save Figure 2: {exc}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Inspect one science + one cal frame from a GEN01 bin_frames directory."""
    # ── Pick bin_dir ─────────────────────────────────────────────────────────
    if len(sys.argv) > 1:
        bin_dir = pathlib.Path(sys.argv[1])
    else:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        chosen = filedialog.askdirectory(
            title="Select GEN01 bin_frames directory",
            initialdir=r"C:\Users\sewell\WindCube\G01_outputs",
        )
        root.destroy()
        if not chosen:
            print("No directory selected — exiting.")
            return
        bin_dir = pathlib.Path(chosen)

    if not bin_dir.is_dir():
        print(f"ERROR: Not a directory: {bin_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"\nD01 Frame Inspector  —  {bin_dir}")
    print("=" * 65)

    # ── Auto-detect science and cal frames ────────────────────────────────────
    all_bins = sorted(bin_dir.glob("*.bin"))
    sci_paths = [f for f in all_bins if "_science" in f.name]
    cal_paths = [f for f in all_bins if "_cal" in f.name]

    if not sci_paths:
        print("ERROR: No *_science.bin frames found in bin_dir.", file=sys.stderr)
        sys.exit(1)
    if not cal_paths:
        print("ERROR: No *_cal.bin frames found in bin_dir.", file=sys.stderr)
        sys.exit(1)

    # Use first science and first cal frame
    sci_path = sci_paths[0]
    cal_path = cal_paths[0]
    print(f"  Science frame : {sci_path.name}")
    print(f"  Cal frame     : {cal_path.name}")

    # ── Load frames ───────────────────────────────────────────────────────────
    sci = read_bin_frame(sci_path)
    cal = read_bin_frame(cal_path)
    print(f"\n  Science: type={sci['img_type']}  "
          f"exp={sci['exp_time_s']:.1f}s  T={sci['ccd_temp_c']:.1f}°C  "
          f"peak={sci['image'].max():.0f} ADU")
    print(f"  Cal:     type={cal['img_type']}  "
          f"exp={cal['exp_time_s']:.1f}s  T={cal['ccd_temp_c']:.1f}°C  "
          f"peak={cal['image'].max():.0f} ADU")

    # ── Load master dark and subtract ─────────────────────────────────────────
    print("\nLoading master dark frames...")
    dark = load_master_dark(bin_dir)
    print(f"  {dark['n_frames']} dark frames averaged.")
    if dark["n_frames"] > 0:
        print(f"  Master dark: mean={dark['mean_adu']:.2f}  "
              f"std={dark['std_adu']:.2f}  "
              f"max={dark['max_adu']:.2f} ADU")

    sci_ds = dark_subtract(sci["image"], dark["image"])
    cal_ds = dark_subtract(cal["image"], dark["image"])
    print(f"\nDark-subtracted science: min={sci_ds.min():.1f}  "
          f"max={sci_ds.max():.1f}  mean={sci_ds.mean():.2f} ADU")
    print(f"Dark-subtracted cal:     min={cal_ds.min():.1f}  "
          f"max={cal_ds.max():.1f}  mean={cal_ds.mean():.2f} ADU")

    # ── Find fringe centre (dark-subtracted cal) ──────────────────────────────
    print("\nRunning Nelder-Mead center finder on dark-subtracted cal frame...")
    cx_opt, cy_opt, nm_result, cost_history = find_center_nelder_mead(
        cal_ds, r_max_px=110, n_bins=500
    )
    print(f"  NM centre: cx={cx_opt:.3f} px  cy={cy_opt:.3f} px")
    print(f"  NM converged: {nm_result.success}  "
          f"evaluations: {len(cost_history)}  "
          f"final cost: {nm_result.fun:.4g}")

    # ── Timestamp for filenames ───────────────────────────────────────────────
    from datetime import datetime, timezone
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # ── Figures ───────────────────────────────────────────────────────────────
    print("\nGenerating Figure 1 (frame overview)...")
    make_figure1(sci, cal, dark, sci_ds, cal_ds,
                 cx_opt, cy_opt, bin_dir, timestamp)

    print("Generating Figure 2 (radial profiles)...")
    make_figure2(sci, cal, sci_ds, cal_ds, dark,
                 cx_opt, cy_opt, cost_history, bin_dir, timestamp)

    print("\nD01 complete.")


if __name__ == "__main__":
    main()
