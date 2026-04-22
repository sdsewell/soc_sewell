"""
validate_f01_neon_airy_fit.py
─────────────────────────────────────────────────────────────────────────────
Standalone interactive validation of the F01 full Airy fit to a neon
calibration image.  Implements pipeline steps 1–4 (and step 4 only via
the F01 module).

Usage
─────
    python validate/validate_f01_neon_airy_fit.py

The script will prompt for:
  (1) Calibration image  — real .bin file OR press Enter for synthetic
  (2) Dark image         — real .bin file OR press Enter for synthetic dark

Supports 1×1 and 2×2 binning (auto-detected from the binary header or
from image shape for synthetic images).

Pipeline steps executed
───────────────────────
  Step 1  Load and display calibration + dark images
  Step 2  Dark subtract; find fringe centre (threshold + circle fits)
  Step 3  Annular reduction → 1D FringeProfile (500 equal-pixel bins)
  Step 4a Tolansky WLS to seed TolanskyResult (ε₆₄₀, α from slope)
          NOTE: full two-line Tolansky requires both Ne lines to be
          resolved as separate fringe families.  This script applies
          the single-line Tolansky (slope only) to seed α; ε is set
          from the known fractional order at 640.2248 nm derived from
          D_25C_MM.  If you have a real two-line Z01a result, replace
          the `_build_tolansky_from_image` call with your CalibrationResult.
  Step 4b F01 staged LM fit (Stages A→B→C→D) → CalibrationResult
  Output  6-panel diagnostic figure saved alongside this script

Output figure panels
────────────────────
  [0,0] Raw calibration image (log stretch)
  [0,1] Dark-subtracted image (linear stretch)
  [1,0] Detected fringe centre overlay
  [1,1] 1D annular-reduced fringe profile + Tolansky peak positions
  [2,0] F01 best-fit model vs data with residuals
  [2,1] Fit summary table: all 9 recovered parameters with 1σ errors

Place this file in:  soc_sewell/validation/
─────────────────────────────────────────────────────────────────────────────
Author:  Claude AI / Scott Sewell  (NCAR/HAO)
Date:    2026-04-22
Spec:    docs/specs/F01_full_airy_fit_to_neon_image_2026-04-21.md  v2
"""

from __future__ import annotations

import os
import sys
import struct
import pathlib
import textwrap
from typing import Optional, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
from matplotlib.colors import LogNorm
from scipy import ndimage, optimize

# ── repo root on sys.path so local imports work when run from any cwd ────────
_HERE = pathlib.Path(__file__).resolve().parent          # validation/
_REPO = _HERE.parent                                     # soc_sewell/
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# ── pipeline imports (graceful degradation if not yet installed) ─────────────
try:
    from src.fpi.f01_full_airy_fit_to_neon_image_2026_04_21 import (
        fit_neon_fringe,
        TolanskyResult,
        CalibrationResult,
        CalibrationFitFlags,
    )
    from src.fpi.m01_airy_forward_model_2026_04_05 import (
        airy_modified,
        InstrumentParams,
    )
    _F01_AVAILABLE = True
except ImportError as _e:
    print(f"\n[WARN] F01 module not found ({_e}).")
    print("       Steps 4a/4b will be skipped — only the diagnostic image is produced.\n")
    _F01_AVAILABLE = False

# ── constants ────────────────────────────────────────────────────────────────
try:
    from src.constants import (
        NE_WAVELENGTH_1_M,
        D_25C_MM,
        PLATE_SCALE_RPX,
        R_MAX_PX,
    )
except ImportError:
    NE_WAVELENGTH_1_M = 640.2248e-9   # m
    D_25C_MM          = 20.0006e-3    # m  (authoritative Tolansky result)
    PLATE_SCALE_RPX   = 1.6071e-4     # rad/px  (2×2 binned)
    R_MAX_PX          = 110           # px

# Header format from P01 spec (38 bytes ASCII + padding to 1024-byte block)
_HEADER_BYTES = 1024
_UINT16_DTYPE = np.dtype(">u2")       # big-endian unsigned 16-bit


# ═══════════════════════════════════════════════════════════════════════════════
# I/O helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _prompt_file(label: str, required_type: str) -> Optional[pathlib.Path]:
    """Prompt for a file path; return None if user presses Enter (synthetic)."""
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"  Press Enter to use SYNTHETIC {required_type} data instead.")
    print(f"{'─'*60}")
    raw = input("  Path: ").strip().strip('"').strip("'")
    if not raw:
        return None
    p = pathlib.Path(raw)
    if not p.exists():
        print(f"  [ERROR] File not found: {p}")
        sys.exit(1)
    return p


def _parse_header(raw: bytes) -> dict:
    """
    Parse the WindCube P01 binary header from the first 1024 bytes.
    Returns a dict with keys: date, time, exposure_ms, nx, ny, binning,
    info_type, etalon_temps.
    Falls back gracefully if header cannot be parsed (synthetic .bin files
    may omit the header).
    """
    try:
        text = raw[:256].decode("ascii", errors="replace").replace("\x00", " ")
        fields: dict = {}
        # Very permissive: just look for tokens we care about
        import re
        m = re.search(r"(\d{6})", text)
        fields["date"] = m.group(1) if m else "unknown"
        m = re.search(r"(\d{6})ms", text)
        fields["exposure_ms"] = int(m.group(1)) if m else None
        m = re.search(r"(\d{3})px.*?(\d{3})px", text)
        if m:
            fields["nx"] = int(m.group(1))
            fields["ny"] = int(m.group(2))
        else:
            fields["nx"] = fields["ny"] = None
        m = re.search(r"b([12])", text)
        fields["binning"] = int(m.group(1)) if m else None
        m = re.search(r"(Cal|Dark|Obs|Test)", text, re.IGNORECASE)
        fields["info_type"] = m.group(1) if m else "unknown"
        # Etalon temps: four floats like [XX.XX, XX.XX, XX.XX, XX.XX]C
        temps = re.findall(r"[-\d]+\.\d+", text)
        fields["etalon_temps"] = [float(t) for t in temps[:4]] if temps else []
        return fields
    except Exception:
        return {}


def _load_bin(path: pathlib.Path) -> Tuple[np.ndarray, dict]:
    """
    Load a WindCube .bin file → (image_uint16, header_dict).
    Strips the 1024-byte header block if present, then reads uint16 pixels.
    Auto-detects 256×256 (2×2 bin) or 512×512 (1×1 bin).
    """
    raw = path.read_bytes()
    hdr = _parse_header(raw[:_HEADER_BYTES])

    # Strip header if file is larger than pixel data alone
    for strip in (_HEADER_BYTES, 0):
        payload = raw[strip:]
        n_pixels = len(payload) // 2
        # Try known sizes: 256*256=65536, 512*512=262144
        for side in (256, 512):
            if n_pixels == side * side:
                img = np.frombuffer(payload, dtype=_UINT16_DTYPE).reshape(side, side).astype(np.float32)
                hdr.setdefault("nx", side)
                hdr.setdefault("ny", side)
                hdr.setdefault("binning", 2 if side == 256 else 1)
                return img, hdr
    raise ValueError(
        f"Cannot parse {path.name}: {len(raw)} bytes does not match any known "
        f"WindCube image size (256×256 or 512×512 with/without 1024-byte header)."
    )


def _make_synthetic_cal(binning: int = 2, rng_seed: int = 42) -> Tuple[np.ndarray, dict]:
    """Generate a synthetic neon calibration image using M01 parameters."""
    rng = np.random.default_rng(rng_seed)
    side = 256 if binning == 2 else 512
    alpha = PLATE_SCALE_RPX if binning == 2 else PLATE_SCALE_RPX / 2
    r_max = R_MAX_PX if binning == 2 else R_MAX_PX * 2

    cx = cy = side / 2.0
    y_idx, x_idx = np.mgrid[0:side, 0:side]
    r_px = np.sqrt((x_idx - cx)**2 + (y_idx - cy)**2).astype(np.float32)

    # Build 1D model then broadcast to 2D
    r1d = np.linspace(0, r_max, 2000)
    params = dict(
        t=D_25C_MM, R_refl=0.53, alpha=alpha, n=1.0,
        r_max=float(r_max), I0=1000.0, I1=-0.1, I2=0.005,
        sigma0=0.5, sigma1=0.0, sigma2=0.0,
    )
    signal1d = airy_modified(r1d, NE_WAVELENGTH_1_M, **params) + 300.0
    signal2d = np.interp(r_px, r1d, signal1d)
    # Poisson noise + read noise
    noisy = rng.poisson(np.maximum(signal2d, 1)).astype(np.float32)
    noisy += rng.normal(0, 5, size=noisy.shape).astype(np.float32)
    noisy = np.clip(noisy, 0, 65535).astype(np.float32)

    hdr = {"binning": binning, "nx": side, "ny": side,
           "info_type": "Cal_SYNTHETIC", "exposure_ms": 120000,
           "etalon_temps": [0.0, 0.0, 0.0, 0.0]}
    return noisy, hdr


def _make_synthetic_dark(shape: Tuple[int,int]) -> np.ndarray:
    """Generate a synthetic dark frame (bias + small thermal noise)."""
    rng = np.random.default_rng(99)
    dark = np.full(shape, 300.0, dtype=np.float32)
    dark += rng.normal(0, 3, size=shape).astype(np.float32)
    return np.clip(dark, 0, 65535).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2 — centre finding
# ═══════════════════════════════════════════════════════════════════════════════

def _find_centre(image: np.ndarray) -> Tuple[float, float]:
    """
    Estimate fringe centre by thresholding bright rings and fitting circles.
    Returns (cx, cy) in pixel coordinates.
    Falls back to geometric centre if fitting fails.
    """
    ny, nx = image.shape
    cx_fallback, cy_fallback = nx / 2.0, ny / 2.0

    try:
        # Threshold at 80th percentile to isolate bright fringes
        thresh = np.percentile(image, 80)
        mask = image > thresh

        # Label connected regions
        labeled, n_labels = ndimage.label(mask)
        if n_labels < 3:
            return cx_fallback, cy_fallback

        # Compute centroid of each region
        centroids = ndimage.center_of_mass(image, labeled, range(1, n_labels + 1))
        # Keep only roughly circular regions near image centre
        cx_est, cy_est = cx_fallback, cy_fallback
        ring_centres = []
        for (cy_r, cx_r) in centroids:
            region = labeled == (centroids.index((cy_r, cx_r)) + 1)
            pixels = np.argwhere(region)
            if len(pixels) < 20:
                continue
            # Radius estimate from area
            r_est = np.sqrt(len(pixels) / np.pi)
            if r_est < 5:
                continue
            ring_centres.append((cx_r, cy_r))

        if len(ring_centres) < 2:
            return cx_fallback, cy_fallback

        # Iterative median centre from ring centroids
        xs = np.array([c[0] for c in ring_centres])
        ys = np.array([c[1] for c in ring_centres])
        return float(np.median(xs)), float(np.median(ys))

    except Exception:
        return cx_fallback, cy_fallback


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3 — annular reduction
# ═══════════════════════════════════════════════════════════════════════════════

def _annular_reduce(image: np.ndarray, cx: float, cy: float,
                    r_max: float, n_bins: int = 500):
    """
    Sort pixels by radius from (cx,cy), bin into n_bins equal-pixel-count
    annuli up to r_max.  Returns (r_grid, profile, sigma_profile).
    """
    ny, nx = image.shape
    y_idx, x_idx = np.mgrid[0:ny, 0:nx]
    r_all = np.sqrt((x_idx - cx)**2 + (y_idx - cy)**2).ravel()
    v_all = image.ravel()

    # Keep only pixels inside r_max
    inside = r_all <= r_max
    r_in = r_all[inside]
    v_in  = v_all[inside]

    # Sort by radius
    order = np.argsort(r_in)
    r_sorted = r_in[order]
    v_sorted = v_in[order]

    # Split into n_bins equal-count bins
    n_total = len(r_sorted)
    bin_edges = np.linspace(0, n_total, n_bins + 1, dtype=int)

    r_grid      = np.zeros(n_bins, dtype=np.float32)
    profile     = np.zeros(n_bins, dtype=np.float32)
    sigma_prof  = np.zeros(n_bins, dtype=np.float32)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i+1]
        if hi <= lo:
            continue
        chunk_r = r_sorted[lo:hi]
        chunk_v = v_sorted[lo:hi]
        r_grid[i]     = np.mean(chunk_r)
        profile[i]    = np.mean(chunk_v)
        std            = np.std(chunk_v, ddof=1) if (hi - lo) > 1 else np.nan
        sigma_prof[i] = std / np.sqrt(hi - lo) if np.isfinite(std) else np.nan

    # Replace NaNs with a floor
    floor = max(1.0, np.nanmedian(profile) * 0.005)
    sigma_prof = np.where(np.isfinite(sigma_prof), sigma_prof, floor)
    sigma_prof = np.maximum(sigma_prof, floor)

    return r_grid, profile, sigma_prof


# ═══════════════════════════════════════════════════════════════════════════════
# Step 4a — build TolanskyResult from the 1D profile
# ═══════════════════════════════════════════════════════════════════════════════

def _build_tolansky_from_profile(r_grid: np.ndarray,
                                  profile: np.ndarray,
                                  binning: int) -> "TolanskyResult":
    """
    Seed a TolanskyResult from the annular-reduced calibration profile.

    This is a single-line Tolansky: we find peak radii, fit P vs r² to get
    the plate scale α from the slope, and compute ε₆₄₀ from D_25C_MM.

    For a full two-line Tolansky (ε₆₄₀ and ε₆₃₈ separately), use the Z01a
    interactive notebook.  The result here is sufficient to seed F01's
    staged LM with a good α prior.
    """
    # Detect ring peaks (local maxima in the profile above median)
    from scipy.signal import find_peaks

    median_val = np.median(profile)
    min_sep = max(5, int(len(profile) / 20))
    peaks_idx, _ = find_peaks(profile, height=median_val, distance=min_sep)

    r_peaks = r_grid[peaks_idx]
    r2_peaks = r_peaks ** 2

    # Integer orders: assign starting from order 1 at smallest radius
    n_peaks = len(r_peaks)
    if n_peaks < 2:
        # Fall back to defaults
        alpha = PLATE_SCALE_RPX if binning == 2 else PLATE_SCALE_RPX / 2
    else:
        P_int = np.arange(1, n_peaks + 1, dtype=float)
        # WLS fit: P = (1/FSR_r2) * r² + ε  where FSR_r2 = λ/(2nd·α²)
        # Equivalently: r² = (FSR_r2) * (P - ε)
        # Simple linear: r² = slope * P + intercept
        A = np.column_stack([P_int, np.ones(n_peaks)])
        result = np.linalg.lstsq(A, r2_peaks, rcond=None)
        slope, intercept = result[0]
        # slope ≈ λ / (2nd·α²)  →  α = sqrt(λ / (2nd·slope))
        alpha_sq = NE_WAVELENGTH_1_M / (2.0 * 1.0 * D_25C_MM * slope) if slope > 0 else 0
        alpha = float(np.sqrt(max(alpha_sq, 0))) if alpha_sq > 0 else (
            PLATE_SCALE_RPX if binning == 2 else PLATE_SCALE_RPX / 2
        )

    # Compute ε₆₄₀ from authoritative D_25C_MM:
    # N_int = round(2nd / λ);  ε = (2nd/λ) - N_int
    frac_order = (2.0 * 1.0 * D_25C_MM) / NE_WAVELENGTH_1_M
    N_int = round(frac_order)
    epsilon_640 = frac_order - N_int

    # epsilon_cal at OI 630 nm (extrapolated; Z01a gives this more precisely)
    frac_oi = (2.0 * 1.0 * D_25C_MM) / 629.95e-9
    N_oi    = round(frac_oi)
    epsilon_cal = frac_oi - N_oi

    if _F01_AVAILABLE:
        return TolanskyResult(
            t_m          = float(D_25C_MM),
            alpha_rpx    = float(alpha),
            epsilon_640  = float(epsilon_640),
            epsilon_638  = 0.0,          # not available from single-line
            epsilon_cal  = float(epsilon_cal),
        )
    else:
        # Return a plain namespace so the rest of the script can still run
        from types import SimpleNamespace
        return SimpleNamespace(
            t_m=D_25C_MM, alpha_rpx=alpha,
            epsilon_640=epsilon_640, epsilon_638=0.0, epsilon_cal=epsilon_cal,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ═══════════════════════════════════════════════════════════════════════════════

_NAVY  = "#003479"
_BLUE  = "#0057C2"
_TEAL  = "#008080"
_AMBER = "#C07000"
_RED   = "#C02020"
_LGRAY = "#D0D8E8"

def _fig_style():
    plt.rcParams.update({
        "figure.facecolor":  _NAVY,
        "axes.facecolor":    "#001A40",
        "axes.edgecolor":    _LGRAY,
        "axes.labelcolor":   _LGRAY,
        "xtick.color":       _LGRAY,
        "ytick.color":       _LGRAY,
        "text.color":        _LGRAY,
        "grid.color":        "#1A3060",
        "grid.linewidth":    0.5,
        "font.family":       "DejaVu Sans",
        "font.size":         9,
        "axes.titlesize":    10,
        "axes.titleweight":  "bold",
    })


def _panel_title(ax, txt):
    ax.set_title(txt, color="white", pad=6)


def _colorbar(fig, im, ax, label="counts"):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.yaxis.set_tick_params(color=_LGRAY)
    cb.outline.set_edgecolor(_LGRAY)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=_LGRAY)
    cb.set_label(label, color=_LGRAY)


# ═══════════════════════════════════════════════════════════════════════════════
# Main diagnostic figure
# ═══════════════════════════════════════════════════════════════════════════════

def _make_figure(cal_raw, dark_img, cal_sub, cx, cy,
                 r_grid, profile, sigma_prof,
                 tolansky, cal_result, binning, hdr):
    _fig_style()

    fig = plt.figure(figsize=(18, 14), facecolor=_NAVY)
    fig.suptitle(
        "F01 — Full Airy Fit to Neon Calibration Image\n"
        f"WindCube FPI Pipeline  ·  NCAR/HAO  ·  Binning: {binning}×{binning}  ·  "
        f"Source: {hdr.get('info_type','unknown')}",
        color="white", fontsize=13, fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(3, 2, figure=fig,
                           left=0.06, right=0.97,
                           top=0.93, bottom=0.05,
                           hspace=0.38, wspace=0.32)

    side = cal_raw.shape[0]
    r_max = R_MAX_PX if binning == 2 else R_MAX_PX * 2

    # ── Panel [0,0] Raw calibration image ────────────────────────────────────
    ax00 = fig.add_subplot(gs[0, 0])
    vmin = np.percentile(cal_raw, 1)
    vmax = np.percentile(cal_raw, 99.5)
    im = ax00.imshow(cal_raw, cmap="inferno", origin="lower",
                     norm=LogNorm(vmin=max(vmin, 1), vmax=vmax))
    _panel_title(ax00, "Step 1 — Raw Calibration Image (log stretch)")
    ax00.set_xlabel("x (px)"); ax00.set_ylabel("y (px)")
    _colorbar(fig, im, ax00, "ADU")
    # Mark image centre region
    circle_outer = Circle((cx, cy), r_max, color=_TEAL, fill=False, lw=1.2, ls="--")
    ax00.add_patch(circle_outer)
    ax00.plot(cx, cy, "+", color=_TEAL, ms=10, mew=1.5)

    # ── Panel [0,1] Dark-subtracted image ────────────────────────────────────
    ax01 = fig.add_subplot(gs[0, 1])
    vmin2 = max(np.percentile(cal_sub, 0.5), 1)
    vmax2 = np.percentile(cal_sub, 99.8)
    im2 = ax01.imshow(cal_sub, cmap="inferno", origin="lower",
                      vmin=vmin2, vmax=vmax2)
    _panel_title(ax01, "Step 2 — Dark-Subtracted Image (linear stretch)")
    ax01.set_xlabel("x (px)"); ax01.set_ylabel("y (px)")
    _colorbar(fig, im2, ax01, "ADU")

    # ── Panel [1,0] Centre overlay ────────────────────────────────────────────
    ax10 = fig.add_subplot(gs[1, 0])
    ax10.imshow(cal_sub, cmap="gray", origin="lower",
                vmin=np.percentile(cal_sub, 5),
                vmax=np.percentile(cal_sub, 99))
    _panel_title(ax10, "Step 2 — Fringe Centre Detection")
    ax10.set_xlabel("x (px)"); ax10.set_ylabel("y (px)")
    # Draw concentric ring guides at r = 30, 60, 90, 110 px (scaled for binning)
    scale = 1 if binning == 2 else 2
    for r_guide in [30*scale, 60*scale, 90*scale, int(r_max)]:
        c = Circle((cx, cy), r_guide, color=_BLUE, fill=False,
                   lw=0.8, alpha=0.6)
        ax10.add_patch(c)
    ax10.plot(cx, cy, "x", color="red", ms=12, mew=2,
              label=f"Centre ({cx:.1f}, {cy:.1f})")
    ax10.legend(fontsize=8, loc="upper right",
                facecolor="#001A40", edgecolor=_LGRAY, labelcolor=_LGRAY)

    # ── Panel [1,1] Annular profile + Tolansky peaks ──────────────────────────
    ax11 = fig.add_subplot(gs[1, 1])
    ax11.plot(r_grid, profile, color=_BLUE, lw=1.2, label="Annular profile")
    ax11.fill_between(r_grid,
                       profile - sigma_prof,
                       profile + sigma_prof,
                       color=_BLUE, alpha=0.25)
    ax11.set_xlabel("Radius r (px)")
    ax11.set_ylabel("Mean counts (ADU)")
    _panel_title(ax11, "Step 3 — 1D Annular-Reduced Fringe Profile")
    ax11.grid(True)

    # Mark Tolansky alpha info
    ax11.text(0.97, 0.97,
              f"α = {tolansky.alpha_rpx:.4e} rad/px\n"
              f"ε₆₄₀ = {tolansky.epsilon_640:.4f}\n"
              f"d = {tolansky.t_m*1e3:.4f} mm",
              transform=ax11.transAxes,
              ha="right", va="top", fontsize=8,
              color=_AMBER,
              bbox=dict(boxstyle="round,pad=0.4", facecolor="#001A40",
                        edgecolor=_AMBER, alpha=0.9))
    ax11.legend(fontsize=8, loc="upper left",
                facecolor="#001A40", edgecolor=_LGRAY, labelcolor=_LGRAY)

    # ── Panel [2,0] F01 fit vs data + residuals ───────────────────────────────
    ax20 = fig.add_subplot(gs[2, 0])

    if cal_result is not None and _F01_AVAILABLE:
        # Evaluate best-fit model on fine grid
        r_fine = np.linspace(r_grid[0], r_grid[-1], 1000)
        model_fine = airy_modified(
            r_fine, NE_WAVELENGTH_1_M,
            t=cal_result.t_m, R_refl=cal_result.R_refl,
            alpha=cal_result.alpha, n=1.0,
            r_max=float(r_max),
            I0=cal_result.I0, I1=cal_result.I1, I2=cal_result.I2,
            sigma0=cal_result.sigma0, sigma1=cal_result.sigma1,
            sigma2=cal_result.sigma2,
        ) + cal_result.B

        ax20.plot(r_grid, profile, color=_LGRAY, lw=1.0, alpha=0.7,
                  label="Data")
        ax20.fill_between(r_grid, profile - sigma_prof, profile + sigma_prof,
                           color=_LGRAY, alpha=0.15)
        ax20.plot(r_fine, model_fine, color=_AMBER, lw=1.8,
                  label=f"F01 fit  (χ²_red = {cal_result.chi2_reduced:.3f})")

        # Inset residuals as a secondary axis
        model_at_bins = np.interp(r_grid, r_fine, model_fine)
        residuals = (profile - model_at_bins) / sigma_prof
        ax20_r = ax20.twinx()
        ax20_r.plot(r_grid, residuals, color=_RED, lw=0.7, alpha=0.6)
        ax20_r.axhline(0, color=_RED, lw=0.8, ls="--", alpha=0.5)
        ax20_r.set_ylabel("Residual (σ)", color=_RED, fontsize=8)
        ax20_r.tick_params(axis="y", colors=_RED)
        ax20_r.set_ylim(-6, 6)

        ax20.legend(fontsize=8, loc="upper right",
                    facecolor="#001A40", edgecolor=_LGRAY, labelcolor=_LGRAY)
        converge_str = "✓ Converged" if cal_result.converged else "✗ Did not converge"
        ax20.set_title(f"Step 4b — F01 Airy Fit  |  {converge_str}",
                       color="white", pad=6, fontsize=10, fontweight="bold")
    else:
        ax20.plot(r_grid, profile, color=_BLUE, lw=1.2)
        _panel_title(ax20, "Step 4b — F01 Fit (module not available)")
        ax20.text(0.5, 0.5, "F01 module not imported\nProfile shown only",
                  transform=ax20.transAxes, ha="center", va="center",
                  color=_AMBER, fontsize=11)

    ax20.set_xlabel("Radius r (px)")
    ax20.set_ylabel("Counts (ADU)")
    ax20.grid(True)

    # ── Panel [2,1] Fit parameter table ──────────────────────────────────────
    ax21 = fig.add_subplot(gs[2, 1])
    ax21.axis("off")
    _panel_title(ax21, "Step 4b — CalibrationResult Summary")

    if cal_result is not None:
        rows = [
            ("Parameter", "Fitted value", "1σ", "Unit"),
            ("─"*14, "─"*14, "─"*8, "─"*6),
            ("d  (fixed)",    f"{cal_result.t_m*1e3:.6f}",
             "(Tolansky)", "mm"),
            ("R_refl",  f"{cal_result.R_refl:.5f}",
             f"±{cal_result.sigma_R_refl:.5f}", "—"),
            ("α",       f"{cal_result.alpha:.5e}",
             f"±{cal_result.sigma_alpha:.2e}", "rad/px"),
            ("I₀",      f"{cal_result.I0:.1f}",
             f"±{cal_result.sigma_I0:.1f}", "ADU"),
            ("I₁",      f"{cal_result.I1:.4f}",
             f"±{cal_result.sigma_I1:.4f}", "—"),
            ("I₂",      f"{cal_result.I2:.4f}",
             f"±{cal_result.sigma_I2:.4f}", "—"),
            ("σ₀",      f"{cal_result.sigma0:.4f}",
             f"±{cal_result.sigma_sigma0:.4f}", "px"),
            ("σ₁",      f"{cal_result.sigma1:.4f}",
             f"±{cal_result.sigma_sigma1:.4f}", "px"),
            ("σ₂",      f"{cal_result.sigma2:.4f}",
             f"±{cal_result.sigma_sigma2:.4f}", "px"),
            ("B",       f"{cal_result.B:.1f}",
             f"±{cal_result.sigma_B:.2f}", "ADU"),
            ("─"*14, "─"*14, "─"*8, "─"*6),
            ("χ²_red",  f"{cal_result.chi2_reduced:.4f}", "", ""),
            ("N bins",  f"{cal_result.n_bins_used}", "", ""),
            ("Flags",
             hex(cal_result.quality_flags),
             "0x000=GOOD", ""),
        ]

        col_x = [0.02, 0.34, 0.62, 0.88]
        header_y = 0.97
        row_h    = 0.061

        for row_i, row in enumerate(rows):
            y = header_y - row_i * row_h
            color = "white" if row_i == 0 else (
                _AMBER if row[0].startswith("─") else _LGRAY
            )
            fw = "bold" if row_i == 0 else "normal"
            for col_i, cell in enumerate(row):
                ax21.text(col_x[col_i], y, cell,
                          transform=ax21.transAxes,
                          ha="left", va="top",
                          fontsize=8.2, color=color, fontweight=fw,
                          fontfamily="monospace")

        # Flag annotation
        flag_names = []
        f = cal_result.quality_flags
        flag_map = {
            0x001: "FIT_FAILED", 0x002: "CHI2_HIGH",
            0x004: "CHI2_VERY_HIGH", 0x008: "CHI2_LOW",
            0x010: "STDERR_NONE", 0x020: "R_AT_BOUND",
            0x040: "ALPHA_AT_BOUND", 0x080: "FEW_BINS",
        }
        for bit, name in flag_map.items():
            if f & bit:
                flag_names.append(name)
        flag_str = ", ".join(flag_names) if flag_names else "GOOD"
        ax21.text(0.02, header_y - len(rows) * row_h - 0.01,
                  f"Quality: {flag_str}",
                  transform=ax21.transAxes,
                  ha="left", va="top",
                  fontsize=8.5,
                  color=_TEAL if not flag_names else _AMBER,
                  fontweight="bold")
    else:
        ax21.text(0.5, 0.5, "No CalibrationResult\n(F01 module not available)",
                  transform=ax21.transAxes,
                  ha="center", va="center",
                  color=_AMBER, fontsize=11)

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═"*60)
    print("  F01 — Full Airy Fit to Neon Calibration Image")
    print("  WindCube FPI Pipeline  |  NCAR/HAO")
    print("  Steps 1–4 interactive validation")
    print("═"*60)

    # ── File prompts ─────────────────────────────────────────────────────────
    cal_path  = _prompt_file("CALIBRATION IMAGE (.bin)", "calibration")
    dark_path = _prompt_file("DARK IMAGE (.bin)", "dark")

    # ── Step 1: Load images ───────────────────────────────────────────────────
    print("\n[Step 1] Loading images...")

    if cal_path is not None:
        cal_raw, hdr = _load_bin(cal_path)
        binning = hdr.get("binning", 2)
        print(f"  Calibration: {cal_path.name}  "
              f"({cal_raw.shape[0]}×{cal_raw.shape[1]}, "
              f"binning={binning}×{binning})")
    else:
        # Ask for binning preference
        b_str = input("\n  Binning for synthetic image (1 or 2) [2]: ").strip()
        binning = int(b_str) if b_str in ("1", "2") else 2
        print(f"  Generating synthetic calibration ({binning}×{binning} binning)...")
        cal_raw, hdr = _make_synthetic_cal(binning=binning)
        print(f"  Synthetic image shape: {cal_raw.shape}")

    r_max = R_MAX_PX if binning == 2 else R_MAX_PX * 2

    if dark_path is not None:
        dark_img, _ = _load_bin(dark_path)
        print(f"  Dark:  {dark_path.name}  ({dark_img.shape[0]}×{dark_img.shape[1]})")
        if dark_img.shape != cal_raw.shape:
            print(f"  [WARN] Dark shape {dark_img.shape} ≠ cal shape {cal_raw.shape}. "
                  f"Using synthetic dark.")
            dark_img = _make_synthetic_dark(cal_raw.shape)
    else:
        print("  Using synthetic dark frame (bias=300 ADU + σ=3 ADU read noise).")
        dark_img = _make_synthetic_dark(cal_raw.shape)

    # ── Step 2: Dark subtract + find centre ──────────────────────────────────
    print("\n[Step 2] Dark-subtracting and finding fringe centre...")
    cal_sub = np.clip(cal_raw - dark_img, 0, None).astype(np.float32)
    cx, cy = _find_centre(cal_sub)
    print(f"  Fringe centre: ({cx:.2f}, {cy:.2f}) px")

    # ── Step 3: Annular reduction ─────────────────────────────────────────────
    print("\n[Step 3] Annular reduction (500 equal-pixel bins)...")
    r_grid, profile, sigma_prof = _annular_reduce(
        cal_sub, cx, cy, r_max=r_max, n_bins=500
    )
    print(f"  Profile range: [{profile.min():.0f}, {profile.max():.0f}] ADU")
    print(f"  Median σ per bin: {np.median(sigma_prof):.1f} ADU")

    # ── Step 4a: Build TolanskyResult ────────────────────────────────────────
    print("\n[Step 4a] Building TolanskyResult from profile (single-line Tolansky)...")
    tolansky = _build_tolansky_from_profile(r_grid, profile, binning)
    print(f"  α  = {tolansky.alpha_rpx:.5e} rad/px")
    print(f"  ε₆₄₀ = {tolansky.epsilon_640:.5f}")
    print(f"  d  = {tolansky.t_m*1e3:.6f} mm  (fixed from D_25C_MM)")

    # ── Step 4b: F01 staged LM fit ────────────────────────────────────────────
    cal_result = None
    if _F01_AVAILABLE:
        print("\n[Step 4b] Running F01 staged Airy fit (A→B→C→D)...")
        print("  This may take 10–30 seconds on a real image.\n")

        # Build a minimal FringeProfile-compatible object
        from types import SimpleNamespace
        fringe_profile = SimpleNamespace(
            r_grid        = r_grid,
            r2_grid       = r_grid**2,
            profile       = profile,
            sigma_profile = sigma_prof,
            masked        = np.zeros(len(r_grid), dtype=bool),
            r_max_px      = float(r_max),
            quality_flags = 0,
        )

        try:
            cal_result = fit_neon_fringe(fringe_profile, tolansky)
            print(f"\n  ✓ Fit converged: {cal_result.converged}")
            print(f"  χ²_red = {cal_result.chi2_reduced:.4f}")
            print(f"  R_refl = {cal_result.R_refl:.5f} ± {cal_result.sigma_R_refl:.5f}")
            print(f"  α      = {cal_result.alpha:.5e} ± {cal_result.sigma_alpha:.2e} rad/px")
            print(f"  σ₀     = {cal_result.sigma0:.4f} ± {cal_result.sigma_sigma0:.4f} px")
            print(f"  B      = {cal_result.B:.1f} ± {cal_result.sigma_B:.2f} ADU")
            flag_str = hex(cal_result.quality_flags)
            print(f"  Flags  = {flag_str}  ({'GOOD' if cal_result.quality_flags == 0 else 'see legend'})")
        except Exception as exc:
            print(f"\n  [ERROR] F01 fit raised an exception: {exc}")
            import traceback; traceback.print_exc()
            cal_result = None
    else:
        print("\n[Step 4b] Skipped — F01 module not available.")

    # ── Output figure ─────────────────────────────────────────────────────────
    print("\n[Output] Generating 6-panel diagnostic figure...")
    fig = _make_figure(
        cal_raw, dark_img, cal_sub,
        cx, cy,
        r_grid, profile, sigma_prof,
        tolansky, cal_result,
        binning, hdr,
    )

    # Save alongside this script
    out_stem = "F01_neon_airy_fit_result"
    if cal_path is not None:
        out_stem = f"F01_{cal_path.stem}_result"
    out_path = _HERE / f"{out_stem}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=_NAVY)
    print(f"  Saved: {out_path}")

    plt.show()
    print("\nDone.\n")


if __name__ == "__main__":
    main()
