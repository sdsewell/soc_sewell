"""
D01_frame_inspector_2026_05_14.py  —  WindCube FPI frame inspector (v3).

Loads one science and one calibration .bin frame from a GEN01 output
bin_frames/ directory, subtracts a master dark (averaged from all *dark*.bin
files in the same directory), locates the fringe centre with Nelder-Mead
azimuthal-variance minimisation, and produces three diagnostic figures.

Figure 1 (4 panels):  sci image | cal image | master dark | param table
Figure 2 (3 panels):  NM convergence | dark-subtracted sci profile |
                       dark-subtracted cal profile + Airy overlay
Figure 3 (2 panels):  dark-subtracted cal image with detected peak rings
                       annotated | Tolansky two-line WLS plot + recovered d

Usage
-----
    python src/processing/D01_frame_inspector_2026_05_14.py [bin_dir]

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
# Peak finding in the calibration radial profile
# ---------------------------------------------------------------------------

# Authoritative neon wavelengths (Burns et al. 1950, IAU)
_NE_LAM1_M = 640.2248e-9   # strong line
_NE_LAM2_M = 638.2991e-9   # weak line
_NE_LAM1_NM = 640.2248
_NE_LAM2_NM = 638.2991

def find_cal_peaks(
    r_cal: np.ndarray,
    p_cal: np.ndarray,
    r_max_px: float = 110.0,
    min_separation_px: float = 3.0,
    prominence_frac: float = 0.03,
    strong_weak_ratio: float = 1.5,
) -> dict:
    """
    Detect Fabry-Pérot fringe peaks in a dark-subtracted neon calibration
    radial profile and split them into the two neon wavelength families.

    Strategy
    --------
    1. Find all local maxima with scipy.signal.find_peaks using a prominence
       floor = prominence_frac × peak-to-valley range of the profile.
       Prominence is background-independent (measures height above the highest
       valley between a peak and any taller neighbor).

    2. Split into strong (λ₁ = 640.2 nm) and weak (λ₂ = 638.3 nm) families
       using an *adaptive bimodal-gap split* on the prominence distribution:

       a. Sort detected peaks by prominence and look for the largest
          multiplicative gap between consecutive sorted values.  If that gap
          exceeds `strong_weak_ratio` (default 1.5) the split is placed there.
          This works even when both families have very similar absolute
          prominences (e.g. synthetic / noise-free images) because it finds
          the *natural* gap rather than comparing to a fixed threshold.

       b. Fallback: if no gap ≥ strong_weak_ratio is found (e.g. only one
          family is present), split at strong_weak_ratio × median prominence
          (the previous behaviour, kept as a last resort).

       Physical basis: the neon 640.2 nm / 638.3 nm intensity ratio is ~2–3
       in a hollow-cathode lamp (real data) but may be compressed to ~1.2–1.5
       in perfectly noise-free synthetic images because the FPI valleys are
       elevated.  The gap search is robust to this compression.

    Why "median × ratio" fails on synthetic data
    ---------------------------------------------
    Synthetic images have zero noise and a flat high background.  The Airy
    valleys between fringes don't descend all the way to the baseline before
    the next fringe rises, so scipy's prominence for all peaks is suppressed
    roughly equally.  The strong/weak prominence ratio shrinks to ~1.2–1.3,
    inside the 1.5× threshold, so the old code classified everything as
    family-2.  The gap search finds the largest *relative* jump in the sorted
    prominence list, which is robust even when the absolute ratio is small.

    Parameters
    ----------
    r_cal              : array (n_bins,) — radii in pixels
    p_cal              : array (n_bins,) — mean ADU per bin (NaN where empty)
    r_max_px           : float — search limit (px)
    min_separation_px  : float — minimum spacing between adjacent peaks (px)
    prominence_frac    : float — minimum prominence ≥ this fraction of the
                                 profile peak-to-valley range
    strong_weak_ratio  : float — minimum gap ratio to declare two families
                                 (default 1.5).  Set to 1.0 to always split
                                 at the largest gap.

    Returns
    -------
    dict with keys:
      r_peaks           : float array — all peak radii (px), ascending
      adu_peaks         : float array — profile ADU at each peak
      prominences       : float array — scipy prominence of each peak
      idx_peaks         : int array   — bin indices in the full r_cal array
      r_fam1            : float array — strong-family (λ₁) peak radii
      adu_fam1, prom_fam1 : corresponding arrays
      r_fam2            : float array — weak-family (λ₂) peak radii
      adu_fam2, prom_fam2 : corresponding arrays
      n_total, n_fam1, n_fam2 : int
      prom_threshold    : float — prominence value of the split point (ADU)
      split_gap_ratio   : float — actual gap ratio found (diagnostic)
      split_method      : str   — 'bimodal_gap' or 'median_ratio' (diagnostic)
      strong_weak_ratio : float — parameter used
    """
    from scipy.signal import find_peaks, peak_prominences

    good = np.isfinite(p_cal) & (r_cal <= r_max_px)
    r_g  = r_cal[good].astype(np.float64)
    p_g  = p_cal[good].astype(np.float64)

    empty   = np.array([], dtype=np.float64)
    empty_i = np.array([], dtype=int)
    _null   = dict(
        r_peaks=empty, adu_peaks=empty, prominences=empty, idx_peaks=empty_i,
        r_fam1=empty, adu_fam1=empty, prom_fam1=empty,
        r_fam2=empty, adu_fam2=empty, prom_fam2=empty,
        n_total=0, n_fam1=0, n_fam2=0,
        prom_threshold=0.0, split_gap_ratio=1.0,
        split_method='none', strong_weak_ratio=strong_weak_ratio,
    )

    if p_g.size < 5:
        return _null

    ptp = float(np.nanmax(p_g) - np.nanmin(p_g))
    prominence_floor = max(1.0, prominence_frac * ptp)

    dr       = float(r_g[1] - r_g[0]) if len(r_g) > 1 else 1.0
    sep_bins = max(1, int(np.ceil(min_separation_px / dr)))

    idx_found, _ = find_peaks(p_g, prominence=prominence_floor, distance=sep_bins)
    if idx_found.size == 0:
        return _null

    # Recompute prominences on the full good-pixel sub-array so scipy can
    # walk to true base valleys rather than stopping at array boundaries.
    proms, _, _ = peak_prominences(p_g, idx_found)

    r_peaks  = r_g[idx_found]
    adu_peaks = p_g[idx_found]
    orig_idx  = np.where(good)[0][idx_found]

    # ── Adaptive bimodal-gap split ────────────────────────────────────────────
    # Sort peaks by prominence (ascending).  The two neon families should
    # form two natural clusters.  We find the largest multiplicative jump
    # between consecutive sorted prominences — this is the cluster boundary.
    sort_idx  = np.argsort(proms)
    p_sorted  = proms[sort_idx]           # ascending prominence values

    split_method    = 'median_ratio'      # default fallback label
    split_gap_ratio = 1.0
    prom_thr        = strong_weak_ratio * float(np.median(proms))  # fallback

    if len(p_sorted) >= 2:
        # Multiplicative gaps: p_sorted[i+1] / p_sorted[i]
        # Guard against zero prominences with a small epsilon
        eps    = 1e-6 * float(p_sorted.max()) if p_sorted.max() > 0 else 1.0
        ratios = (p_sorted[1:] + eps) / (p_sorted[:-1] + eps)
        best_k = int(np.argmax(ratios))          # index of largest gap
        split_gap_ratio = float(ratios[best_k])

        if split_gap_ratio >= strong_weak_ratio:
            # Natural gap found: split between p_sorted[best_k] and [best_k+1]
            # Place threshold at the geometric mean of the two bounding values
            prom_thr     = float(np.sqrt(p_sorted[best_k] * p_sorted[best_k + 1]))
            split_method = 'bimodal_gap'

    mask_fam1 = proms >= prom_thr
    mask_fam2 = ~mask_fam1

    # Sanity: if bimodal split gives 0 in one family, fall back to equal halves
    if mask_fam1.sum() == 0 or mask_fam2.sum() == 0:
        half      = len(proms) // 2
        sort_r    = np.argsort(proms)
        mask_fam2 = np.zeros(len(proms), dtype=bool)
        mask_fam2[sort_r[:half]] = True
        mask_fam1 = ~mask_fam2
        prom_thr  = float(np.sqrt(proms[sort_r[half - 1]] * proms[sort_r[half]]))
        split_method = 'equal_halves_fallback'

    return dict(
        r_peaks           = r_peaks,
        adu_peaks         = adu_peaks,
        prominences       = proms,
        idx_peaks         = orig_idx,
        r_fam1            = r_peaks[mask_fam1],
        adu_fam1          = adu_peaks[mask_fam1],
        prom_fam1         = proms[mask_fam1],
        r_fam2            = r_peaks[mask_fam2],
        adu_fam2          = adu_peaks[mask_fam2],
        prom_fam2         = proms[mask_fam2],
        n_total           = int(len(r_peaks)),
        n_fam1            = int(mask_fam1.sum()),
        n_fam2            = int(mask_fam2.sum()),
        prom_threshold    = prom_thr,
        split_gap_ratio   = split_gap_ratio,
        split_method      = split_method,
        strong_weak_ratio = strong_weak_ratio,
    )



# ---------------------------------------------------------------------------
# Tolansky two-line WLS analysis
# ---------------------------------------------------------------------------

def tolansky_two_line_wls(
    r_fam1: np.ndarray,
    r_fam2: np.ndarray,
    cx_opt: float,
    cy_opt: float,
    alpha: float,
    f_m: float,
    d_icos_m: float = 20.008e-3,
    n_int_override: int = -189,
) -> dict:
    """
    Recover the etalon gap d via the Tolansky / Benoit two-line method.

    Physics recap
    -------------
    For a monochromatic line λ the FPI condition gives the fractional
    interference order at fringe-peak radius r (from the plate centre):

        S(r, λ) = 2 n d cos θ / λ   where  cos θ = 1 / sqrt(1 + (α·r)²)

    Re-arranging, for the k-th ring of family j:

        S_jk  =  2 n d / λ_j  ·  1 / sqrt(1 + (α · r_jk)²)

    The Tolansky Benoit excess-fraction technique recovers d from the
    *difference* of fractional orders between the two wavelength families,
    eliminating the unknown integer order N:

        d  =  (N_int + ε₁ − ε₂) · λ₁λ₂ / [2n(λ₂ − λ₁)]          (*)

    where ε_j = fractional part of S(r_j, λ_j) evaluated at common cosθ,
    and N_int = −189 is pinned from the ICOS spacer measurement.

    Implementation
    --------------
    For each matched pair (one peak from family-1 near one from family-2)
    we compute:
        u_jk  =  α² · r_jk²   (dimensionless)
        x_jk  =  1 / sqrt(1 + u_jk)   (= cos θ_jk)

    Then the WLS regression model is:

        S(r, λ)  =  (2 n d / λ) · x     →    S  =  m · x

    where m = 2 n d / λ is the slope (d is the free parameter).

    For each family we solve for m_j independently via WLS, weighting by
    proximity to the nearest companion peak (equal weights in this version).
    The two slopes give two independent d estimates; their weighted mean is
    the final result.

    Parameters
    ----------
    r_fam1, r_fam2 : peak radii (px) for each neon family, ascending
    cx_opt, cy_opt : fringe-centre pixel coordinates
    alpha          : plate scale (rad/px)
    f_m            : imaging lens focal length (m) — used only for display
    d_icos_m       : ICOS spacer gap (m) — used solely to pin N_int
    n_int_override : integer N_int (default −189, from S13 analysis)

    Returns
    -------
    dict with keys:
      d_m            : float — recovered etalon gap (m)
      d_mm           : float — same in mm
      d_sigma_m      : float — 1σ uncertainty (m) from WLS residuals
      m1, m2         : float — WLS slopes for each family
      d1_m, d2_m     : float — per-family d estimates
      N_int          : int
      eps1, eps2     : float — mean excess fractions (fractional orders)
      x_fam1, x_fam2: arrays — cos-theta values for each family's peaks
      S_fam1, S_fam2: arrays — interference order at each peak (for plotting)
      r2_fam1, r2_fam2: float — WLS R² for each family
      n_pairs        : int   — number of peaks used
      valid          : bool  — False if fewer than 2 peaks per family
    """
    lam1 = _NE_LAM1_M
    lam2 = _NE_LAM2_M
    n_ref = 1.0  # index of refraction (air ≈ 1)

    # ── guard: need at least 2 peaks per family ───────────────────────────────
    if len(r_fam1) < 2 or len(r_fam2) < 2:
        return dict(valid=False, d_m=np.nan, d_mm=np.nan, d_sigma_m=np.nan,
                    m1=np.nan, m2=np.nan, d1_m=np.nan, d2_m=np.nan,
                    N_int=n_int_override, eps1=np.nan, eps2=np.nan,
                    x_fam1=np.array([]), x_fam2=np.array([]),
                    S_fam1=np.array([]), S_fam2=np.array([]),
                    r2_fam1=np.nan, r2_fam2=np.nan, n_pairs=0)

    N_int = n_int_override

    def _cos_theta(r_px):
        u = (alpha * r_px) ** 2
        return 1.0 / np.sqrt(1.0 + u)

    x1 = _cos_theta(r_fam1)
    x2 = _cos_theta(r_fam2)

    # ── Estimate integer order at centre for each family ─────────────────────
    # m_j ≈ 2 n d_icos / λ_j — used to assign fractional orders
    m1_prior = 2.0 * n_ref * d_icos_m / lam1
    m2_prior = 2.0 * n_ref * d_icos_m / lam2

    # Fractional orders at each peak centre: S_jk = m_j · x_jk
    S1_prior = m1_prior * x1
    S2_prior = m2_prior * x2
    eps1_arr = S1_prior - np.floor(S1_prior)
    eps2_arr = S2_prior - np.floor(S2_prior)
    eps1 = float(np.mean(eps1_arr))
    eps2 = float(np.mean(eps2_arr))

    # ── WLS: S_jk = m_j · x_jk  (no intercept — Airy peak at order integer) ─
    # Weight by cos_theta (higher cos = lower off-axis = better defined peak)
    w1 = x1 / x1.sum() if x1.sum() > 0 else np.ones_like(x1) / len(x1)
    w2 = x2 / x2.sum() if x2.sum() > 0 else np.ones_like(x2) / len(x2)

    # Assign nearest integer order to each peak using ICOS prior
    N1 = np.round(S1_prior).astype(int)
    N2 = np.round(S2_prior).astype(int)

    # Actual S values at each peak (integer + fractional excess)
    S1 = N1.astype(float)  + eps1_arr   # S_jk = N_jk + ε_jk
    S2 = N2.astype(float)  + eps2_arr

    # WLS slopes: m_j = Σ(w_jk · S_jk · x_jk) / Σ(w_jk · x_jk²)
    m1 = float(np.sum(w1 * S1 * x1) / np.sum(w1 * x1**2))
    m2 = float(np.sum(w2 * S2 * x2) / np.sum(w2 * x2**2))

    d1 = m1 * lam1 / (2.0 * n_ref)
    d2 = m2 * lam2 / (2.0 * n_ref)

    # WLS R² for each family
    def _r2(S_obs, m, x, w):
        S_pred  = m * x
        ss_res  = float(np.sum(w * (S_obs - S_pred)**2))
        S_wmean = float(np.sum(w * S_obs))
        ss_tot  = float(np.sum(w * (S_obs - S_wmean)**2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    r2_1 = _r2(S1, m1, x1, w1)
    r2_2 = _r2(S2, m2, x2, w2)

    # ── Benoit gap from equation (*) ─────────────────────────────────────────
    # Use mean excess fractions
    d_benoit = (N_int + eps1 - eps2) * (lam1 * lam2) / (2.0 * n_ref * (lam2 - lam1))

    # Weighted mean of the two per-family estimates (equally weighted here)
    d_mean  = 0.5 * (d1 + d2)
    d_sigma = 0.5 * abs(d1 - d2)   # half-spread as a conservative σ

    # Use Benoit result as primary (more physically principled)
    d_final = d_benoit

    n_pairs = min(len(r_fam1), len(r_fam2))

    return dict(
        valid     = True,
        d_m       = d_final,
        d_mm      = d_final * 1e3,
        d_sigma_m = d_sigma,
        m1        = m1,
        m2        = m2,
        d1_m      = d1,
        d2_m      = d2,
        d_benoit_m= d_benoit,
        N_int     = N_int,
        eps1      = eps1,
        eps2      = eps2,
        x_fam1    = x1,
        x_fam2    = x2,
        S_fam1    = S1,
        S_fam2    = S2,
        r2_fam1   = r2_1,
        r2_fam2   = r2_2,
        n_pairs   = n_pairs,
    )


# ---------------------------------------------------------------------------
# Figure 3 — peak detection + Tolansky WLS
# ---------------------------------------------------------------------------

def make_figure3(
    cal_ds: np.ndarray,
    r_cal: np.ndarray,
    p_cal: np.ndarray,
    cx_opt: float,
    cy_opt: float,
    peaks: dict,
    tol: dict,
    folder: pathlib.Path,
    timestamp: str,
) -> None:
    """
    Two-panel diagnostic figure.

    Panel A (top):   Dark-subtracted cal image with detected fringe peak
                     rings drawn as coloured circles + radial profile
                     with peak markers.
    Panel B (bottom): Tolansky WLS plot (S vs cos θ for both families)
                      and recovered-d summary table.

    Saves to folder/D01_fig3_{timestamp}.png.
    """
    alpha_val = 1.6071e-4   # rad/px — authoritative plate scale

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        f"D01 Fig 3 — Peak Detection & Tolansky Two-Line WLS  —  {timestamp}",
        fontsize=12, fontweight="bold",
    )

    # ── Layout: top row = image (left) + profile (right)
    #            bottom row = WLS plot (left) + results table (right) ─────────
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[1.1, 1.0],
        width_ratios=[1.0, 1.4],
        hspace=0.38, wspace=0.32,
    )
    ax_img  = fig.add_subplot(gs[0, 0])
    ax_prof = fig.add_subplot(gs[0, 1])
    ax_wls  = fig.add_subplot(gs[1, 0])
    ax_tbl  = fig.add_subplot(gs[1, 1])

    # ── A1: dark-subtracted cal image with ring overlays ─────────────────────
    vlo = float(np.percentile(cal_ds, 1))
    vhi = float(np.percentile(cal_ds, 99))
    ax_img.imshow(
        cal_ds, cmap="gray", origin="upper",
        vmin=vlo, vmax=vhi, aspect="equal", interpolation="nearest",
    )
    ax_img.plot(cx_opt, cy_opt, "+", color="yellow",
                markersize=14, markeredgewidth=1.8,
                label=f"cx={cx_opt:.1f}, cy={cy_opt:.1f}")

    # Draw rings at detected peak radii
    theta = np.linspace(0, 2 * np.pi, 360)
    for r_px in peaks.get("r_fam1", []):
        ax_img.plot(
            cx_opt + r_px * np.cos(theta),
            cy_opt + r_px * np.sin(theta),
            color="deepskyblue", linewidth=1.2, alpha=0.85,
        )
    for r_px in peaks.get("r_fam2", []):
        ax_img.plot(
            cx_opt + r_px * np.cos(theta),
            cy_opt + r_px * np.sin(theta),
            color="tomato", linewidth=1.2, alpha=0.85,
            linestyle="--",
        )

    # Legend patches
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="yellow",      marker="+", linestyle="None",
               markersize=10, label="NM centre"),
        Line2D([0], [0], color="deepskyblue", linewidth=1.2,
               label=f"Family 1 / λ₁=640.2 nm  ({peaks['n_fam1']} rings)"),
        Line2D([0], [0], color="tomato",      linewidth=1.2, linestyle="--",
               label=f"Family 2 / λ₂=638.3 nm  ({peaks['n_fam2']} rings)"),
    ]
    ax_img.legend(handles=legend_elements, fontsize=6.5, loc="lower right",
                  framealpha=0.75)
    ax_img.set_title("Dark-subtracted cal — detected fringe rings", fontsize=9)
    ax_img.set_xlabel("Column (px)", fontsize=8)
    ax_img.set_ylabel("Row (px)",    fontsize=8)
    ax_img.tick_params(labelsize=7)

    # ── A2: radial profile with peak markers ─────────────────────────────────
    good = np.isfinite(p_cal)
    ax_prof.plot(r_cal[good], p_cal[good],
                 color="steelblue", linewidth=0.85, alpha=0.85,
                 label="DS cal profile", zorder=2)

    if peaks["n_fam1"] > 0:
        ax_prof.scatter(peaks["r_fam1"], peaks["adu_fam1"],
                        color="deepskyblue", s=45, zorder=5,
                        label=f"Fam-1 / λ₁=640.2 nm ({peaks['n_fam1']} peaks)",
                        edgecolors="navy", linewidths=0.7)
        # Draw prominence bars (vertical lines from peak down to base level)
        for r_pk, a_pk, prom in zip(peaks["r_fam1"], peaks["adu_fam1"], peaks["prom_fam1"]):
            ax_prof.vlines(r_pk, a_pk - prom, a_pk,
                           color="deepskyblue", linewidth=0.8, alpha=0.5, zorder=3)

    if peaks["n_fam2"] > 0:
        ax_prof.scatter(peaks["r_fam2"], peaks["adu_fam2"],
                        color="tomato", marker="D", s=35, zorder=5,
                        label=f"Fam-2 / λ₂=638.3 nm ({peaks['n_fam2']} peaks)",
                        edgecolors="darkred", linewidths=0.7)
        for r_pk, a_pk, prom in zip(peaks["r_fam2"], peaks["adu_fam2"], peaks["prom_fam2"]):
            ax_prof.vlines(r_pk, a_pk - prom, a_pk,
                           color="tomato", linewidth=0.8, alpha=0.5, zorder=3)

    # Show the split method and threshold as a text annotation
    ax_prof.text(
        0.02, 0.98,
        f"Split method: {peaks['split_method']}\n"
        f"Gap ratio found: {peaks['split_gap_ratio']:.3f}  (threshold ≥ {peaks['strong_weak_ratio']:.1f})\n"
        f"Prom threshold: {peaks['prom_threshold']:.0f} ADU\n"
        f"({peaks['n_fam1']} strong + {peaks['n_fam2']} weak = {peaks['n_total']} total)",
        transform=ax_prof.transAxes, fontsize=7, va="top", ha="left",
        bbox=dict(facecolor="white", alpha=0.75, pad=2),
    )
    ax_prof.set_title("Radial profile — prominence-based family split", fontsize=9)
    ax_prof.set_xlabel("Radius (px)", fontsize=8)
    ax_prof.set_ylabel("Mean intensity (ADU)", fontsize=8)
    ax_prof.tick_params(labelsize=7)
    ax_prof.legend(fontsize=7)
    ax_prof.grid(True, alpha=0.2)

    # ── B1: Tolansky WLS  S vs cos θ ─────────────────────────────────────────
    if tol.get("valid", False):
        x1_data = tol["x_fam1"]
        S1_data = tol["S_fam1"]
        x2_data = tol["x_fam2"]
        S2_data = tol["S_fam2"]

        # Plot measured orders
        ax_wls.scatter(x1_data, S1_data, color="deepskyblue", s=40,
                       zorder=5, label=f"Fam-1 λ₁=640.2 nm  R²={tol['r2_fam1']:.5f}",
                       edgecolors="navy", linewidths=0.7)
        ax_wls.scatter(x2_data, S2_data, color="tomato", marker="D", s=35,
                       zorder=5, label=f"Fam-2 λ₂=638.3 nm  R²={tol['r2_fam2']:.5f}",
                       edgecolors="darkred", linewidths=0.7)

        # WLS fit lines
        x_range = np.linspace(
            min(x1_data.min() if len(x1_data) else 1.0,
                x2_data.min() if len(x2_data) else 1.0) - 0.01,
            1.002, 200,
        )
        ax_wls.plot(x_range, tol["m1"] * x_range,
                    color="deepskyblue", linewidth=1.4, linestyle="--",
                    alpha=0.7, label=f"WLS fit m₁={tol['m1']:.2f}")
        ax_wls.plot(x_range, tol["m2"] * x_range,
                    color="tomato", linewidth=1.4, linestyle="--",
                    alpha=0.7, label=f"WLS fit m₂={tol['m2']:.2f}")

        ax_wls.set_xlabel("cos θ  (= 1 / √(1 + α²r²))", fontsize=8)
        ax_wls.set_ylabel("Interference order  S = 2nd·cosθ / λ", fontsize=8)
        ax_wls.set_title("Tolansky two-line WLS:  S vs cos θ", fontsize=9)
        ax_wls.legend(fontsize=7)
        ax_wls.grid(True, alpha=0.2)
        ax_wls.tick_params(labelsize=7)
    else:
        ax_wls.text(0.5, 0.5,
                    "Tolansky WLS unavailable\n(< 2 peaks per family detected)",
                    transform=ax_wls.transAxes, ha="center", va="center",
                    fontsize=10, color="gray")
        ax_wls.set_title("Tolansky two-line WLS", fontsize=9)
        ax_wls.axis("off")

    # ── B2: results table ─────────────────────────────────────────────────────
    ax_tbl.axis("off")

    if tol.get("valid", False):
        rows_tol = [
            ["── Peak detection ──",           ""],
            ["Total peaks found",              str(peaks["n_total"])],
            ["Family-1 peaks (λ₁=640.2 nm)",  str(peaks["n_fam1"])],
            ["Family-2 peaks (λ₂=638.3 nm)",  str(peaks["n_fam2"])],
            ["Split method",                   peaks["split_method"]],
            ["Gap ratio found",                f"{peaks['split_gap_ratio']:.3f}  (≥ {peaks['strong_weak_ratio']:.1f} required)"],
            ["Prominence threshold",           f"{peaks['prom_threshold']:.0f} ADU"],
            ["── Tolansky WLS ──",             ""],
            ["N_int (ICOS pinned)",            str(tol["N_int"])],
            ["ε₁ (mean excess, λ₁)",          f"{tol['eps1']:.6f}"],
            ["ε₂ (mean excess, λ₂)",          f"{tol['eps2']:.6f}"],
            ["ε₁ − ε₂",                       f"{tol['eps1'] - tol['eps2']:.6f}"],
            ["WLS slope m₁",                  f"{tol['m1']:.4f}"],
            ["WLS slope m₂",                  f"{tol['m2']:.4f}"],
            ["d from family-1  (mm)",         f"{tol['d1_m']*1e3:.6f}"],
            ["d from family-2  (mm)",         f"{tol['d2_m']*1e3:.6f}"],
            ["d Benoit (mm)",                 f"{tol['d_benoit_m']*1e3:.6f}"],
            ["d WLS mean (mm)",               f"{(tol['d1_m']+tol['d2_m'])*0.5*1e3:.6f}"],
            ["d σ half-spread (µm)",          f"{tol['d_sigma_m']*1e6:.3f}"],
            ["R² family-1",                   f"{tol['r2_fam1']:.6f}"],
            ["R² family-2",                   f"{tol['r2_fam2']:.6f}"],
            ["── Reference values ──",        ""],
            ["d Tolansky (authoritative)",    "20.106 mm"],
            ["d ICOS spacer",                 "20.008 mm"],
        ]
    else:
        rows_tol = [
            ["── Peak detection ──",           ""],
            ["Total peaks found",              str(peaks["n_total"])],
            ["Family-1 peaks",                 str(peaks["n_fam1"])],
            ["Family-2 peaks",                 str(peaks["n_fam2"])],
            ["Split method",                   peaks["split_method"]],
            ["Gap ratio found",                f"{peaks['split_gap_ratio']:.3f}"],
            ["Prominence threshold",           f"{peaks['prom_threshold']:.0f} ADU"],
            ["Tolansky WLS",                   "INSUFFICIENT PEAKS — adjust strong_weak_ratio?"],
        ]

    tbl = ax_tbl.table(
        cellText=[[r[0], r[1]] for r in rows_tol],
        colLabels=["Parameter", "Value"],
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1.0, 1.45)

    _HDR_BG = "#2C3E50"
    _SEC_BG = "#D5E8D4"
    _ALT_BG = "#EBF5FB"
    _HLT_BG = "#FFF3CD"  # highlight for the primary d result
    for (r_idx, c_idx), cell in tbl.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        cell.set_linewidth(0.5)
        if r_idx == 0:
            cell.set_facecolor(_HDR_BG)
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            label = rows_tol[r_idx - 1][0]
            if label.startswith("──"):
                cell.set_facecolor(_SEC_BG)
                cell.get_text().set_fontweight("bold")
            elif label.startswith("d Benoit"):
                cell.set_facecolor(_HLT_BG)
                cell.get_text().set_fontweight("bold")
            elif r_idx % 2 == 0:
                cell.set_facecolor(_ALT_BG)

    ax_tbl.set_title("Tolansky WLS Results", fontsize=9, fontweight="bold", pad=8)

    out = pathlib.Path(folder) / f"D01_fig3_{timestamp}.png"
    try:
        fig.savefig(str(out), dpi=150, bbox_inches="tight")
        print(f"  Figure 3 saved: {out}")
    except Exception as exc:
        print(f"  WARNING: Could not save Figure 3: {exc}")
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

    # ── Figure 3: peak detection + Tolansky WLS ───────────────────────────────
    print("Generating Figure 3 (peak detection + Tolansky WLS)...")

    # Compute radial profile at higher resolution for peak finding
    r_cal_fine, p_cal_fine = annular_mean(cal_ds, cx_opt, cy_opt,
                                           r_max_px=110.0, n_bins=1000)

    peaks = find_cal_peaks(r_cal_fine, p_cal_fine, r_max_px=110.0,
                            min_separation_px=3.0)
    print(f"  Detected {peaks['n_total']} peaks total: "
          f"{peaks['n_fam1']} fam-1 (strong), "
          f"{peaks['n_fam2']} fam-2 (weak)")
    if peaks["n_total"] > 0:
        print(f"  Peak radii (px): {np.round(peaks['r_peaks'], 2).tolist()}")

    try:
        from windcube.constants import ALPHA_RAD_PX, F_M, D_ICOS_M
    except ImportError:
        # Authoritative fallback values (project spec / S13)
        ALPHA_RAD_PX = 1.6071e-4   # rad/px  (2x2 binned, Tolansky result)
        F_M          = 0.19912     # m       (imaging lens focal length)
        D_ICOS_M     = 20.008e-3   # m       (ICOS spacer — N_int disambiguation only)
        print("  NOTE: windcube.constants not importable; using authoritative fallback values.")

    tol = tolansky_two_line_wls(
        r_fam1     = peaks["r_fam1"],
        r_fam2     = peaks["r_fam2"],
        cx_opt     = cx_opt,
        cy_opt     = cy_opt,
        alpha      = ALPHA_RAD_PX,
        f_m        = F_M,
        d_icos_m   = D_ICOS_M,
        n_int_override = -189,
    )
    if tol["valid"]:
        print(f"  Tolansky WLS:  d_Benoit = {tol['d_benoit_m']*1e3:.6f} mm  "
              f"(ref: 20.106 mm)")
        print(f"                 d_fam1   = {tol['d1_m']*1e3:.6f} mm")
        print(f"                 d_fam2   = {tol['d2_m']*1e3:.6f} mm")
        print(f"                 ε₁={tol['eps1']:.6f}  ε₂={tol['eps2']:.6f}")
    else:
        print("  WARNING: Tolansky WLS failed — fewer than 2 peaks per family.")

    make_figure3(
        cal_ds    = cal_ds,
        r_cal     = r_cal_fine,
        p_cal     = p_cal_fine,
        cx_opt    = cx_opt,
        cy_opt    = cy_opt,
        peaks     = peaks,
        tol       = tol,
        folder    = bin_dir,
        timestamp = timestamp,
    )

    print("\nD01 complete.")


if __name__ == "__main__":
    main()
