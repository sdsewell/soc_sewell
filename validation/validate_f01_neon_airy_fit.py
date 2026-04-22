"""
validate_f01_neon_airy_fit.py
─────────────────────────────────────────────────────────────────────────────
Standalone interactive validation of the F01 full Airy fit to a neon
calibration image.  Implements the complete pipeline steps 1–4, including
a full two-line Tolansky analysis to seed the TolanskyResult.

Usage
─────
    python validation/validate_f01_neon_airy_fit.py

Prompts
───────
  (1) Calibration image  — real WindCube .bin  OR  Enter → synthetic
  (2) Dark image         — real WindCube .bin  OR  Enter → synthetic dark
  (3) Binning (1 or 2)   — asked only when using synthetic images

Supports 1×1 and 2×2 binning; auto-detected from binary header for real files.

Pipeline steps executed
───────────────────────
  Step 1   Load calibration + dark images; display header metadata
  Step 2   Dark subtract; find fringe centre via thresholding + circle fits
  Step 3   Annular reduction → 1D FringeProfile (500 equal-pixel-count bins)
  Step 4a  TWO-LINE TOLANSKY (Benoit/Vaughan gap recovery)
             • Find all peaks in annular profile (adaptive distance floor)
             • Separate the two neon families by amplitude
               (640.2248 nm family = brighter peaks; 638.2991 nm = dimmer)
             • WLS fit of ring-order P vs r² for each family independently
             • Recover ε₆₄₀, ε₆₃₈ from WLS intercept/slope ratio
             • Compute N_Δ from ICOS prior gap d₀
             • Apply Benoit eq. (Vaughan 3.97) to recover d
             • α from slope of the 640 nm family
  Step 4b  F01 staged LM fit (Stages A→B→C→D) → CalibrationResult

Output
──────
  8-panel diagnostic figure saved as PNG alongside this script:
    [0,0]  Raw calibration image (log stretch) with r_max circle
    [0,1]  Dark-subtracted image (linear stretch)
    [1,0]  Fringe centre detection overlay
    [1,1]  Annular profile with TWO-LINE TOLANSKY peak family annotations
    [2,0]  Tolansky P vs r² WLS fits — both families on same axes
    [2,1]  Benoit gap recovery annotation box + α, ε values
    [3,0]  F01 best-fit model vs data with residuals in σ
    [3,1]  CalibrationResult parameter table (all 9 params + 1σ)

Place this file in:  soc_sewell/validation/
─────────────────────────────────────────────────────────────────────────────
Author:  Claude AI / Scott Sewell  (NCAR/HAO)
Date:    2026-04-22
Spec:    docs/specs/F01_full_airy_fit_to_neon_image_2026-04-21.md  v2
         docs/specs/F02_full_airy_fit_to_airglow_image_2026-04-21.md  v2
"""

from __future__ import annotations

import os
import sys
import pathlib
from types import SimpleNamespace
from typing import Optional, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
from matplotlib.colors import LogNorm
from scipy import ndimage
from scipy.signal import find_peaks
from scipy.optimize import least_squares

# ── repo root on sys.path ────────────────────────────────────────────────────
_HERE = pathlib.Path(__file__).resolve().parent   # validation/
_REPO = _HERE.parent                              # soc_sewell/
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# ── pipeline imports ─────────────────────────────────────────────────────────
try:
    from src.fpi.f01_full_airy_fit_to_neon_image_2026_04_21 import (
        fit_neon_fringe,
        TolanskyResult,
        CalibrationResult,
        CalibrationFitFlags,
    )
    from src.fpi.m01_airy_forward_model_2026_04_05 import airy_modified
    _F01_AVAILABLE = True
except ImportError as _e:
    print(f"\n[WARN] F01 module not found ({_e}).")
    print("       Steps 4b will be skipped — Tolansky + diagnostic plot still produced.\n")
    _F01_AVAILABLE = False
    TolanskyResult = SimpleNamespace      # fallback type alias

# ── constants ────────────────────────────────────────────────────────────────
try:
    from src.constants import (
        NE_WAVELENGTH_1_M,   # 640.2248e-9 m  primary neon line
        D_25C_MM,            # 20.0006e-3 m   authoritative gap
        PLATE_SCALE_RPX,     # 1.6071e-4 rad/px  2×2 binned
        R_MAX_PX,            # 110 px
    )
    try:
        from src.constants import NE_WAVELENGTH_2_M   # 638.2991e-9 m
    except ImportError:
        NE_WAVELENGTH_2_M = 638.2991e-9
    try:
        from src.constants import NE_INTENSITY_2       # 0.8 relative intensity
    except ImportError:
        NE_INTENSITY_2 = 0.8
except ImportError:
    NE_WAVELENGTH_1_M  = 640.2248e-9
    NE_WAVELENGTH_2_M  = 638.2991e-9
    NE_INTENSITY_2     = 0.8
    D_25C_MM           = 20.0006e-3
    PLATE_SCALE_RPX    = 1.6071e-4
    R_MAX_PX           = 110

_HEADER_BYTES = 1024
_UINT16_DTYPE = np.dtype(">u2")   # big-endian uint16

# ── colour palette (NCAR brand) ───────────────────────────────────────────────
_NAVY  = "#003479"
_BLUE  = "#0057C2"
_TEAL  = "#009999"
_AMBER = "#C07000"
_RED   = "#CC2222"
_GREEN = "#22AA44"
_LGRAY = "#C8D4E8"


# ═══════════════════════════════════════════════════════════════════════════════
# I/O
# ═══════════════════════════════════════════════════════════════════════════════

def _prompt_file(label: str, kind: str) -> Optional[pathlib.Path]:
    print(f"\n{'─'*62}")
    print(f"  {label}")
    print(f"  Press Enter to use SYNTHETIC {kind} data.")
    print(f"{'─'*62}")
    raw = input("  Path: ").strip().strip('"').strip("'")
    if not raw:
        return None
    p = pathlib.Path(raw)
    if not p.exists():
        print(f"  [ERROR] File not found: {p}")
        sys.exit(1)
    return p


def _parse_header(raw: bytes) -> dict:
    import re
    try:
        text = raw[:256].decode("ascii", errors="replace").replace("\x00", " ")
        h: dict = {}
        m = re.search(r"(\d{6})", text);         h["date"] = m.group(1) if m else "unknown"
        m = re.search(r"(\d{6})ms", text);       h["exposure_ms"] = int(m.group(1)) if m else None
        m = re.search(r"(\d{3})px.*?(\d{3})px", text)
        if m: h["nx"], h["ny"] = int(m.group(1)), int(m.group(2))
        else: h["nx"] = h["ny"] = None
        m = re.search(r"b([12])", text);         h["binning"] = int(m.group(1)) if m else None
        m = re.search(r"(Cal|Dark|Obs|Test)", text, re.I); h["info_type"] = m.group(1) if m else "unknown"
        temps = re.findall(r"[-\d]+\.\d+", text); h["etalon_temps"] = [float(t) for t in temps[:4]]
        return h
    except Exception:
        return {}


def _load_bin(path: pathlib.Path) -> Tuple[np.ndarray, dict]:
    raw = path.read_bytes()
    hdr = _parse_header(raw[:_HEADER_BYTES])
    for strip in (_HEADER_BYTES, 0):
        payload = raw[strip:]
        n_pixels = len(payload) // 2
        for side in (256, 512):
            if n_pixels == side * side:
                img = np.frombuffer(payload, dtype=_UINT16_DTYPE).reshape(side, side).astype(np.float32)
                hdr.setdefault("nx", side); hdr.setdefault("ny", side)
                hdr.setdefault("binning", 2 if side == 256 else 1)
                return img, hdr
    raise ValueError(
        f"Cannot parse {path.name}: {len(raw)} bytes does not match "
        "256×256 or 512×512 (with/without 1024-byte header)."
    )


def _make_synthetic_cal(binning: int = 2, rng_seed: int = 42) -> Tuple[np.ndarray, dict]:
    """Two-line synthetic neon calibration image."""
    rng  = np.random.default_rng(rng_seed)
    side = 256 if binning == 2 else 512
    alpha = PLATE_SCALE_RPX if binning == 2 else PLATE_SCALE_RPX / 2
    r_max = float(R_MAX_PX if binning == 2 else R_MAX_PX * 2)

    cx = cy = side / 2.0
    y_idx, x_idx = np.mgrid[0:side, 0:side]
    r_px = np.sqrt((x_idx - cx)**2 + (y_idx - cy)**2).astype(np.float32)

    r1d = np.linspace(0, r_max, 3000)
    kw = dict(t=D_25C_MM, R_refl=0.53, alpha=alpha, n=1.0, r_max=r_max,
              I0=1000.0, I1=-0.1, I2=0.005, sigma0=0.5, sigma1=0.0, sigma2=0.0)
    sig_a = airy_modified(r1d, NE_WAVELENGTH_1_M, **kw)            # bright line
    sig_b = airy_modified(r1d, NE_WAVELENGTH_2_M, **kw) * NE_INTENSITY_2  # dim line
    signal1d = sig_a + sig_b + 300.0
    signal2d = np.interp(r_px, r1d, signal1d)

    noisy = rng.poisson(np.maximum(signal2d, 1)).astype(np.float32)
    noisy += rng.normal(0, 5, size=noisy.shape).astype(np.float32)
    noisy = np.clip(noisy, 0, 65535).astype(np.float32)
    hdr = {"binning": binning, "nx": side, "ny": side,
           "info_type": "Cal_SYNTHETIC", "exposure_ms": 120000,
           "etalon_temps": [0.0, 0.0, 0.0, 0.0]}
    return noisy, hdr


def _make_synthetic_dark(shape: Tuple[int,int]) -> np.ndarray:
    rng = np.random.default_rng(99)
    d = np.full(shape, 300.0, dtype=np.float32)
    d += rng.normal(0, 3, size=shape).astype(np.float32)
    return np.clip(d, 0, 65535).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2 — centre finding
# ═══════════════════════════════════════════════════════════════════════════════

def _find_centre(image: np.ndarray) -> Tuple[float, float]:
    ny, nx = image.shape
    cx0, cy0 = nx / 2.0, ny / 2.0
    try:
        thresh   = np.percentile(image, 80)
        mask     = image > thresh
        labeled, n = ndimage.label(mask)
        if n < 3:
            return cx0, cy0
        props = ndimage.center_of_mass(image, labeled, range(1, n + 1))
        xs = [p[1] for p in props if p is not None]
        ys = [p[0] for p in props if p is not None]
        # Filter to regions within 30% of image half-diagonal from centre
        half_diag = 0.3 * np.sqrt(nx**2 + ny**2) / 2
        keep_x, keep_y = [], []
        for x, y in zip(xs, ys):
            if np.sqrt((x-cx0)**2 + (y-cy0)**2) < half_diag:
                keep_x.append(x); keep_y.append(y)
        if len(keep_x) < 2:
            return cx0, cy0
        return float(np.median(keep_x)), float(np.median(keep_y))
    except Exception:
        return cx0, cy0


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3 — annular reduction
# ═══════════════════════════════════════════════════════════════════════════════

def _annular_reduce(image: np.ndarray, cx: float, cy: float,
                    r_max: float, n_bins: int = 500):
    ny, nx = image.shape
    y_idx, x_idx = np.mgrid[0:ny, 0:nx]
    r_all = np.sqrt((x_idx - cx)**2 + (y_idx - cy)**2).ravel()
    v_all = image.ravel()
    inside = r_all <= r_max
    r_s = r_all[inside][np.argsort(r_all[inside])]
    v_s = v_all[inside][np.argsort(r_all[inside])]
    edges = np.linspace(0, len(r_s), n_bins + 1, dtype=int)

    r_grid     = np.zeros(n_bins, dtype=np.float32)
    profile    = np.zeros(n_bins, dtype=np.float32)
    sigma_prof = np.zeros(n_bins, dtype=np.float32)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i+1]
        if hi <= lo:
            continue
        cr, cv = r_s[lo:hi], v_s[lo:hi]
        r_grid[i]    = np.mean(cr)
        profile[i]   = np.mean(cv)
        std = np.std(cv, ddof=1) if (hi-lo) > 1 else np.nan
        sigma_prof[i] = std / np.sqrt(hi-lo) if np.isfinite(std) else np.nan

    floor = max(1.0, float(np.nanmedian(profile)) * 0.005)
    sigma_prof = np.where(np.isfinite(sigma_prof), sigma_prof, floor)
    sigma_prof = np.maximum(sigma_prof, floor)
    return r_grid, profile, sigma_prof


# ═══════════════════════════════════════════════════════════════════════════════
# Step 4a — FULL TWO-LINE TOLANSKY
# ═══════════════════════════════════════════════════════════════════════════════

def _two_line_tolansky(
    r_grid: np.ndarray,
    profile: np.ndarray,
    binning: int,
    d_prior: float = D_25C_MM,
    lam_a: float   = NE_WAVELENGTH_1_M,   # 640.2248 nm  bright
    lam_b: float   = NE_WAVELENGTH_2_M,   # 638.2991 nm  dim
    intensity_ratio_b: float = NE_INTENSITY_2,
) -> dict:
    """
    Full two-line Tolansky analysis following the Z01a algorithm.

    Returns a dict with keys:
        epsilon_a, epsilon_b  — fractional fringe orders at lam_a, lam_b
        alpha                 — plate scale, rad/px (from lam_a WLS slope)
        d_recovered           — Benoit gap recovery, metres
        N_delta               — integer order difference
        r2_a, P_a             — ring data for lam_a family
        r2_b, P_b             — ring data for lam_b family
        slope_a, intercept_a  — WLS fit coefficients for lam_a
        slope_b, intercept_b  — WLS fit coefficients for lam_b
        n_peaks_a, n_peaks_b  — number of peaks found per family
        warnings              — list of warning strings
    """
    warnings_list: list[str] = []

    # ── 1. Adaptive peak finding ──────────────────────────────────────────────
    # Minimum separation: ~half the expected ring spacing in r-space.
    # For d ≈ 20 mm, α ≈ 1.6e-4 rad/px, FSR ≈ λ²/2d ≈ 10 pm.
    # Ring spacing Δr ≈ sqrt(FSR / (α²·d)) ≈ 10–20 bins for 500-bin profile.
    # Use adaptive floor: largest gap between consecutive profile maxima / 3.
    min_sep = max(5, int(len(profile) / 25))

    # Baseline-subtracted profile for peak finding
    baseline = np.percentile(profile, 20)
    prof_bs  = profile - baseline

    # Find peaks with adaptive distance; require height > 30% of peak range
    pk_height = float(np.percentile(prof_bs, 85))
    peak_idx, peak_props = find_peaks(
        prof_bs,
        height   = pk_height * 0.3,
        distance = min_sep,
    )

    if len(peak_idx) < 4:
        # Fall back to a lower threshold
        peak_idx, peak_props = find_peaks(
            prof_bs,
            height   = float(np.percentile(prof_bs, 60)) * 0.3,
            distance = max(3, min_sep // 2),
        )
        warnings_list.append(
            f"Low peak count with primary threshold — relaxed to {len(peak_idx)} peaks"
        )

    if len(peak_idx) < 2:
        warnings_list.append("FATAL: fewer than 2 peaks found; Tolansky cannot proceed")
        alpha_fallback = PLATE_SCALE_RPX if binning == 2 else PLATE_SCALE_RPX / 2
        return {
            "epsilon_a": 0.0, "epsilon_b": 0.0, "alpha": alpha_fallback,
            "d_recovered": d_prior, "N_delta": 0,
            "r2_a": np.array([]), "P_a": np.array([]),
            "r2_b": np.array([]), "P_b": np.array([]),
            "slope_a": 0.0, "intercept_a": 0.0,
            "slope_b": 0.0, "intercept_b": 0.0,
            "n_peaks_a": 0, "n_peaks_b": 0,
            "warnings": warnings_list,
        }

    peak_r   = r_grid[peak_idx]
    peak_val = profile[peak_idx]   # raw profile amplitudes at peaks

    # ── 2. Separate families by amplitude ────────────────────────────────────
    # The 640.2248 nm line is stronger; 638.2991 nm has relative intensity ~0.8.
    # Sort all peaks by amplitude and split at the median amplitude.
    # This is the Z01a locked approach: bright half → family A (640 nm),
    # dim half → family B (638 nm).
    # When an odd number of peaks, the extra goes to family A.
    n_peaks = len(peak_idx)
    n_a = (n_peaks + 1) // 2   # larger half goes to bright family
    n_b = n_peaks - n_a

    amp_order = np.argsort(peak_val)[::-1]   # descending amplitude
    idx_a = np.sort(amp_order[:n_a])         # bright peaks → 640 nm family
    idx_b = np.sort(amp_order[n_a:])         # dim  peaks → 638 nm family

    r_a  = peak_r[idx_a];   r2_a = r_a ** 2
    r_b  = peak_r[idx_b];   r2_b = r_b ** 2

    # ── 3. Assign integer orders P starting at 1 from innermost ring ─────────
    P_a = np.arange(1, len(r_a) + 1, dtype=float)
    P_b = np.arange(1, len(r_b) + 1, dtype=float)

    # ── 4. WLS fit: r² = slope·P + intercept  for each family ────────────────
    # Weights: uniform (each ring equally trusted)
    def _wls_fit(P: np.ndarray, r2: np.ndarray):
        if len(P) < 2:
            return np.nan, np.nan, np.nan, np.nan
        A  = np.column_stack([P, np.ones_like(P)])
        ATA = A.T @ A
        ATb = A.T @ r2
        try:
            coeffs = np.linalg.solve(ATA, ATb)
        except np.linalg.LinAlgError:
            coeffs = np.linalg.lstsq(A, r2, rcond=None)[0]
        slope, intercept = float(coeffs[0]), float(coeffs[1])
        # Residuals for quality estimate
        resid = r2 - (slope * P + intercept)
        rms = float(np.sqrt(np.mean(resid**2)))
        return slope, intercept, rms, resid

    slope_a, intercept_a, rms_a, resid_a = _wls_fit(P_a, r2_a)
    slope_b, intercept_b, rms_b, resid_b = _wls_fit(P_b, r2_b)

    # ── 5. Recover fractional orders ε ────────────────────────────────────────
    # From the Tolansky equation:  r² = (λ / (2nd·α²)) · (P − ε)
    # Therefore:  ε = −intercept / slope
    epsilon_a = float(-intercept_a / slope_a) if (slope_a and np.isfinite(slope_a) and abs(slope_a) > 1e-6) else 0.0
    epsilon_b = float(-intercept_b / slope_b) if (slope_b and np.isfinite(slope_b) and abs(slope_b) > 1e-6) else 0.0

    # Keep ε in (0, 1) — Tolansky fractional orders are defined mod 1
    epsilon_a = epsilon_a % 1.0
    epsilon_b = epsilon_b % 1.0

    # ── 6. Plate scale α from lam_a slope ─────────────────────────────────────
    # slope = λ_a / (2·n·d·α²)  →  α = sqrt(λ_a / (2·n·d·slope))
    if np.isfinite(slope_a) and slope_a > 0:
        alpha_sq = lam_a / (2.0 * 1.0 * d_prior * slope_a)
        alpha    = float(np.sqrt(alpha_sq)) if alpha_sq > 0 else (
            PLATE_SCALE_RPX if binning == 2 else PLATE_SCALE_RPX / 2
        )
    else:
        alpha = PLATE_SCALE_RPX if binning == 2 else PLATE_SCALE_RPX / 2
        warnings_list.append("α fell back to default: slope_a not positive")

    # ── 7. Benoit gap recovery (Vaughan Eq. 3.97) ────────────────────────────
    # N_Δ = round(2·d₀ · (1/λ_a − 1/λ_b))
    N_delta = int(round(2.0 * d_prior * (1.0 / lam_a - 1.0 / lam_b)))

    # d = (N_Δ + ε_a − ε_b) · λ_a·λ_b / [2·(λ_b − λ_a)]
    numerator   = (N_delta + epsilon_a - epsilon_b) * lam_a * lam_b
    denominator = 2.0 * (lam_b - lam_a)
    if abs(denominator) > 0:
        d_recovered = float(numerator / denominator)
    else:
        d_recovered = d_prior
        warnings_list.append("Benoit denominator near zero — using prior gap")

    # Sanity check: recovered d should be within ±5 µm of prior
    if abs(d_recovered - d_prior) > 5e-6:
        warnings_list.append(
            f"Recovered d={d_recovered*1e3:.6f} mm deviates >5 µm from prior "
            f"{d_prior*1e3:.6f} mm — check family separation"
        )

    return {
        "epsilon_a":   epsilon_a,
        "epsilon_b":   epsilon_b,
        "alpha":       alpha,
        "d_recovered": d_recovered,
        "N_delta":     N_delta,
        "r2_a": r2_a, "P_a": P_a,
        "r2_b": r2_b, "P_b": P_b,
        "slope_a":     slope_a,    "intercept_a": intercept_a,
        "slope_b":     slope_b,    "intercept_b": intercept_b,
        "n_peaks_a":   len(r_a),   "n_peaks_b":   len(r_b),
        "rms_a":       rms_a,      "rms_b":       rms_b,
        "peak_r":      peak_r,     "peak_val":    peak_val,
        "idx_a":       idx_a,      "idx_b":       idx_b,
        "warnings":    warnings_list,
    }


def _build_tolansky_result(tol: dict, binning: int) -> object:
    """Wrap the Tolansky dict into a TolanskyResult (or SimpleNamespace fallback)."""
    # epsilon_cal at OI 629.95 nm (inversion rest wavelength)
    frac_oi = (2.0 * 1.0 * tol["d_recovered"]) / 629.95e-9
    epsilon_cal = frac_oi % 1.0

    kwargs = dict(
        t_m         = float(tol["d_recovered"]),
        alpha_rpx   = float(tol["alpha"]),
        epsilon_640 = float(tol["epsilon_a"]),
        epsilon_638 = float(tol["epsilon_b"]),
        epsilon_cal = float(epsilon_cal),
    )
    if _F01_AVAILABLE:
        return TolanskyResult(**kwargs)
    else:
        return SimpleNamespace(**kwargs)


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════════

def _fig_style():
    plt.rcParams.update({
        "figure.facecolor": _NAVY, "axes.facecolor": "#001A40",
        "axes.edgecolor": _LGRAY, "axes.labelcolor": _LGRAY,
        "xtick.color": _LGRAY, "ytick.color": _LGRAY,
        "text.color": _LGRAY, "grid.color": "#1A3060",
        "grid.linewidth": 0.5, "font.family": "DejaVu Sans",
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    })


def _cb(fig, im, ax, label="ADU"):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.yaxis.set_tick_params(color=_LGRAY)
    cb.outline.set_edgecolor(_LGRAY)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=_LGRAY)
    cb.set_label(label, color=_LGRAY)


def _make_figure(cal_raw, dark_img, cal_sub, cx, cy,
                 r_grid, profile, sigma_prof,
                 tol_dict, tolansky, cal_result,
                 binning, hdr):
    _fig_style()
    fig = plt.figure(figsize=(20, 22), facecolor=_NAVY)
    fig.suptitle(
        "F01 — Full Airy Fit to Neon Calibration Image  |  WindCube FPI  ·  NCAR/HAO\n"
        f"Source: {hdr.get('info_type','?')}   "
        f"Binning: {binning}×{binning}   "
        f"Exposure: {hdr.get('exposure_ms','?')} ms   "
        f"Etalon temps: {hdr.get('etalon_temps',[])} °C",
        color="white", fontsize=12, fontweight="bold", y=0.995,
    )

    gs = gridspec.GridSpec(4, 2, figure=fig,
                           left=0.06, right=0.97,
                           top=0.965, bottom=0.04,
                           hspace=0.42, wspace=0.30)

    side  = cal_raw.shape[0]
    r_max = float(R_MAX_PX if binning == 2 else R_MAX_PX * 2)

    # ── [0,0]  Raw calibration image ─────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    v1 = max(np.percentile(cal_raw, 1), 1)
    v2 = np.percentile(cal_raw, 99.5)
    im = ax.imshow(cal_raw, cmap="inferno", origin="lower",
                   norm=LogNorm(vmin=v1, vmax=v2))
    ax.set_title("Step 1 — Raw Calibration Image (log)", color="white", pad=5)
    ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
    _cb(fig, im, ax)
    ax.add_patch(Circle((cx, cy), r_max, color=_TEAL, fill=False, lw=1.2, ls="--"))
    ax.plot(cx, cy, "+", color=_TEAL, ms=10, mew=1.5)

    # ── [0,1]  Dark-subtracted image ─────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    v1 = max(np.percentile(cal_sub, 0.5), 1)
    v2 = np.percentile(cal_sub, 99.8)
    im2 = ax.imshow(cal_sub, cmap="inferno", origin="lower", vmin=v1, vmax=v2)
    ax.set_title("Step 2 — Dark-Subtracted Image (linear)", color="white", pad=5)
    ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
    _cb(fig, im2, ax)

    # ── [1,0]  Centre detection ───────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(cal_sub, cmap="gray", origin="lower",
              vmin=np.percentile(cal_sub, 5),
              vmax=np.percentile(cal_sub, 99))
    ax.set_title("Step 2 — Fringe Centre Detection", color="white", pad=5)
    ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
    scale = 1 if binning == 2 else 2
    for r_g in [int(30*scale), int(60*scale), int(90*scale), int(r_max)]:
        ax.add_patch(Circle((cx, cy), r_g, color=_BLUE, fill=False, lw=0.8, alpha=0.6))
    ax.plot(cx, cy, "x", color="red", ms=12, mew=2,
            label=f"Centre ({cx:.1f}, {cy:.1f})")
    ax.legend(fontsize=8, loc="upper right",
              facecolor="#001A40", edgecolor=_LGRAY, labelcolor=_LGRAY)

    # ── [1,1]  Annular profile with peak family annotations ──────────────────
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(r_grid, profile, color=_LGRAY, lw=1.0, alpha=0.7, label="Profile")
    ax.fill_between(r_grid, profile - sigma_prof, profile + sigma_prof,
                    color=_LGRAY, alpha=0.12)

    if tol_dict and "peak_r" in tol_dict and len(tol_dict["peak_r"]) > 0:
        pr   = tol_dict["peak_r"]
        pv   = tol_dict["peak_val"]
        ia   = tol_dict["idx_a"]
        ib   = tol_dict["idx_b"]
        # Family A — bright (640 nm)
        ax.scatter(pr[ia], pv[ia], s=55, marker="o", color=_AMBER, zorder=5,
                   label="640.2248 nm family (bright)")
        # Family B — dim (638 nm)
        ax.scatter(pr[ib], pv[ib], s=40, marker="s", color=_BLUE, zorder=5,
                   label="638.2991 nm family (dim)")
        # Label P orders for family A
        for k, (r_k, v_k, p_k) in enumerate(zip(pr[ia], pv[ia], tol_dict["P_a"])):
            ax.annotate(f"P={int(p_k)}", (r_k, v_k),
                        textcoords="offset points", xytext=(0, 6),
                        fontsize=6.5, color=_AMBER, ha="center")

    ax.set_xlabel("Radius r (px)"); ax.set_ylabel("Mean counts (ADU)")
    ax.set_title("Step 3 — Annular Profile + Tolansky Peak Families", color="white", pad=5)
    ax.grid(True)
    ax.legend(fontsize=8, loc="upper right",
              facecolor="#001A40", edgecolor=_LGRAY, labelcolor=_LGRAY)

    # ── [2,0]  Tolansky P vs r² WLS fits ─────────────────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    ax.set_title("Step 4a — Tolansky P vs r²  (two-line WLS)", color="white", pad=5)
    ax.set_xlabel("r²_fit  (px²)"); ax.set_ylabel("Ring order  P")
    ax.grid(True)

    if tol_dict and len(tol_dict["r2_a"]) >= 2:
        r2_a = tol_dict["r2_a"]; P_a = tol_dict["P_a"]
        r2_b = tol_dict["r2_b"]; P_b = tol_dict["P_b"]
        sa   = tol_dict["slope_a"]; ia_ = tol_dict["intercept_a"]
        sb   = tol_dict["slope_b"]; ib_ = tol_dict["intercept_b"]
        eps_a = tol_dict["epsilon_a"]; eps_b = tol_dict["epsilon_b"]

        # Data points
        ax.scatter(r2_a, P_a, s=60, marker="o", color=_AMBER, zorder=5,
                   label=f"640.2248 nm  ε = {eps_a:.4f}")
        ax.scatter(r2_b, P_b, s=50, marker="s", color=_BLUE, zorder=5,
                   label=f"638.2991 nm  ε = {eps_b:.4f}")

        # Fit lines
        all_r2 = np.concatenate([r2_a, r2_b])
        r2_line = np.linspace(0, all_r2.max() * 1.05, 200)
        ax.plot(r2_line, sa * r2_line + ia_, color=_AMBER, lw=1.5, ls="--", alpha=0.8)
        ax.plot(r2_line, sb * r2_line + ib_, color=_BLUE,  lw=1.5, ls="--", alpha=0.8)

        ax.legend(fontsize=8.5, loc="upper left",
                  facecolor="#001A40", edgecolor=_LGRAY, labelcolor=_LGRAY)
    else:
        ax.text(0.5, 0.5, "Insufficient peaks for WLS fit",
                transform=ax.transAxes, ha="center", va="center",
                color=_AMBER, fontsize=11)

    # ── [2,1]  Benoit recovery + Tolansky summary ─────────────────────────────
    ax = fig.add_subplot(gs[2, 1])
    ax.axis("off")
    ax.set_title("Step 4a — Tolansky / Benoit Summary", color="white", pad=5)

    if tol_dict:
        d_rec   = tol_dict["d_recovered"]
        N_delta = tol_dict["N_delta"]
        eps_a   = tol_dict["epsilon_a"]
        eps_b   = tol_dict["epsilon_b"]
        alpha   = tol_dict["alpha"]
        na      = tol_dict["n_peaks_a"]
        nb      = tol_dict["n_peaks_b"]

        lines = [
            ("Two-Line Tolansky Results", "", "header"),
            ("─"*28, "", "rule"),
            ("λ_a  (640.2248 nm)",      f"ε_a  = {eps_a:.5f}", "data"),
            ("λ_b  (638.2991 nm)",      f"ε_b  = {eps_b:.5f}", "data"),
            ("N_Δ  (integer diff)",     f"{N_delta}", "data"),
            ("─"*28, "", "rule"),
            ("Benoit  d  =", "", "subhead"),
            ("  (N_Δ + ε_a − ε_b)·λ_a·λ_b", "", "formula"),
            ("  ─────────────────────────", "", "formula"),
            ("      2·(λ_b − λ_a)", "", "formula"),
            ("─"*28, "", "rule"),
            ("d  recovered",            f"{d_rec*1e3:.6f} mm", "result"),
            ("d  prior (ICOS)",         f"{D_25C_MM*1e3:.6f} mm", "data"),
            ("Δd  (rec − prior)",       f"{(d_rec-D_25C_MM)*1e6:+.1f} nm", "data"),
            ("─"*28, "", "rule"),
            ("α  (from λ_a slope)",     f"{alpha:.5e} rad/px", "result"),
            ("α  prior (Tolansky)",     f"{PLATE_SCALE_RPX:.5e} rad/px", "data"),
            ("─"*28, "", "rule"),
            (f"Peaks: λ_a={na}  λ_b={nb}", "", "data"),
        ]
        if tol_dict["warnings"]:
            lines.append(("─"*28, "", "rule"))
            for w in tol_dict["warnings"]:
                lines.append((f"⚠ {w[:45]}", "", "warn"))

        y0 = 0.97; dy = 0.047
        for i, (left, right, style) in enumerate(lines):
            y = y0 - i * dy
            col = {"header": "white", "rule": "#3050A0", "subhead": _AMBER,
                   "formula": "#90AACC", "result": _GREEN,
                   "data": _LGRAY, "warn": _RED}.get(style, _LGRAY)
            fw  = "bold" if style in ("header", "subhead", "result") else "normal"
            ax.text(0.02, y, left,  transform=ax.transAxes,
                    ha="left", va="top", fontsize=8.3,
                    color=col, fontweight=fw, fontfamily="monospace")
            if right:
                ax.text(0.60, y, right, transform=ax.transAxes,
                        ha="left", va="top", fontsize=8.3,
                        color=col, fontweight=fw, fontfamily="monospace")

    # ── [3,0]  F01 fit vs data + residuals ───────────────────────────────────
    ax = fig.add_subplot(gs[3, 0])
    ax.set_xlabel("Radius r (px)"); ax.set_ylabel("Counts (ADU)")
    ax.grid(True)

    if cal_result is not None and _F01_AVAILABLE:
        r_fine = np.linspace(r_grid[0], r_grid[-1], 1000)
        model_fine = airy_modified(
            r_fine, NE_WAVELENGTH_1_M,
            t=cal_result.t_m, R_refl=cal_result.R_refl,
            alpha=cal_result.alpha, n=1.0, r_max=r_max,
            I0=cal_result.I0, I1=cal_result.I1, I2=cal_result.I2,
            sigma0=cal_result.sigma0, sigma1=cal_result.sigma1,
            sigma2=cal_result.sigma2,
        ) + cal_result.B

        ax.plot(r_grid, profile, color=_LGRAY, lw=1.0, alpha=0.7, label="Data")
        ax.fill_between(r_grid, profile-sigma_prof, profile+sigma_prof,
                        color=_LGRAY, alpha=0.12)
        ax.plot(r_fine, model_fine, color=_AMBER, lw=2.0,
                label=f"F01 fit  χ²_red = {cal_result.chi2_reduced:.3f}")

        ax_r = ax.twinx()
        model_bins = np.interp(r_grid, r_fine, model_fine)
        resid = (profile - model_bins) / sigma_prof
        ax_r.plot(r_grid, resid, color=_RED, lw=0.7, alpha=0.5)
        ax_r.axhline(0, color=_RED, lw=0.8, ls="--", alpha=0.4)
        ax_r.set_ylabel("Residual (σ)", color=_RED, fontsize=8)
        ax_r.tick_params(axis="y", colors=_RED); ax_r.set_ylim(-6, 6)

        ax.legend(fontsize=8, loc="upper right",
                  facecolor="#001A40", edgecolor=_LGRAY, labelcolor=_LGRAY)
        converge_str = "✓ Converged" if cal_result.converged else "✗ Did not converge"
        ax.set_title(f"Step 4b — F01 Staged Airy Fit  |  {converge_str}",
                     color="white", pad=5)
    else:
        ax.plot(r_grid, profile, color=_BLUE, lw=1.2)
        ax.set_title("Step 4b — F01 Fit (module not available)", color="white", pad=5)
        ax.text(0.5, 0.5, "Install F01 module to enable full fit",
                transform=ax.transAxes, ha="center", va="center",
                color=_AMBER, fontsize=11)

    # ── [3,1]  CalibrationResult table ───────────────────────────────────────
    ax = fig.add_subplot(gs[3, 1])
    ax.axis("off")
    ax.set_title("Step 4b — CalibrationResult Summary", color="white", pad=5)

    if cal_result is not None:
        rows = [
            ("Parameter",        "Fitted",                        "1σ",                           "Unit"),
            ("─"*13,             "─"*13,                          "─"*9,                           "─"*5),
            ("d  (fixed)",       f"{cal_result.t_m*1e3:.6f}",     "(Tolansky)",                    "mm"),
            ("R_refl",           f"{cal_result.R_refl:.5f}",      f"±{cal_result.sigma_R_refl:.5f}", "—"),
            ("α",                f"{cal_result.alpha:.4e}",        f"±{cal_result.sigma_alpha:.2e}", "rad/px"),
            ("I₀",               f"{cal_result.I0:.1f}",           f"±{cal_result.sigma_I0:.1f}",   "ADU"),
            ("I₁",               f"{cal_result.I1:.4f}",           f"±{cal_result.sigma_I1:.4f}",   "—"),
            ("I₂",               f"{cal_result.I2:.4f}",           f"±{cal_result.sigma_I2:.4f}",   "—"),
            ("σ₀",               f"{cal_result.sigma0:.4f}",       f"±{cal_result.sigma_sigma0:.4f}", "px"),
            ("σ₁",               f"{cal_result.sigma1:.4f}",       f"±{cal_result.sigma_sigma1:.4f}", "px"),
            ("σ₂",               f"{cal_result.sigma2:.4f}",       f"±{cal_result.sigma_sigma2:.4f}", "px"),
            ("B",                f"{cal_result.B:.1f}",            f"±{cal_result.sigma_B:.2f}",    "ADU"),
            ("─"*13,             "─"*13,                          "─"*9,                           "─"*5),
            ("χ²_red",           f"{cal_result.chi2_reduced:.4f}", "",                             ""),
            ("N bins",           f"{cal_result.n_bins_used}",      "",                             ""),
            ("Flags",            hex(cal_result.quality_flags),    "0x000=GOOD",                    ""),
        ]
        cx_ = [0.02, 0.36, 0.63, 0.88]
        y0_ = 0.97; dy_ = 0.058
        for ri, row in enumerate(rows):
            y = y0_ - ri * dy_
            is_header = (ri == 0); is_rule = row[0].startswith("─")
            col = "white" if is_header else (_AMBER if is_rule else _LGRAY)
            fw  = "bold" if is_header else "normal"
            for ci, cell in enumerate(row):
                ax.text(cx_[ci], y, cell, transform=ax.transAxes,
                        ha="left", va="top", fontsize=8.2,
                        color=col, fontweight=fw, fontfamily="monospace")

        # Flag decode
        flag_names = []
        f = cal_result.quality_flags
        for bit, name in [(0x001,"FIT_FAILED"),(0x002,"CHI2_HIGH"),
                          (0x004,"CHI2_VERY_HIGH"),(0x008,"CHI2_LOW"),
                          (0x010,"STDERR_NONE"),(0x020,"R_AT_BOUND"),
                          (0x040,"ALPHA_AT_BOUND"),(0x080,"FEW_BINS")]:
            if f & bit:
                flag_names.append(name)
        qstr = ", ".join(flag_names) if flag_names else "GOOD"
        ax.text(0.02, y0_ - len(rows)*dy_ - 0.02,
                f"Quality: {qstr}",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=8.5, fontweight="bold",
                color=_GREEN if not flag_names else _RED)
    else:
        ax.text(0.5, 0.5, "F01 not run — no CalibrationResult",
                transform=ax.transAxes, ha="center", va="center",
                color=_AMBER, fontsize=11)

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═"*64)
    print("  F01  Full Airy Fit to Neon Calibration Image")
    print("  WindCube FPI Pipeline  ·  NCAR/HAO  ·  Steps 1–4")
    print("═"*64)

    # ── Prompts ───────────────────────────────────────────────────────────────
    cal_path  = _prompt_file("CALIBRATION IMAGE (.bin)", "calibration")
    dark_path = _prompt_file("DARK IMAGE (.bin)", "dark")

    # ── Step 1: Load ──────────────────────────────────────────────────────────
    print("\n[Step 1] Loading images...")
    if cal_path is not None:
        cal_raw, hdr = _load_bin(cal_path)
        binning = hdr.get("binning", 2)
        print(f"  Calibration: {cal_path.name}  "
              f"({cal_raw.shape[0]}×{cal_raw.shape[1]}, {binning}×{binning} bin)")
    else:
        b_str = input("\n  Binning for synthetic images (1 or 2) [2]: ").strip()
        binning = int(b_str) if b_str in ("1", "2") else 2
        print(f"  Generating synthetic two-line neon image ({binning}×{binning} binning)...")
        cal_raw, hdr = _make_synthetic_cal(binning=binning)

    r_max = float(R_MAX_PX if binning == 2 else R_MAX_PX * 2)

    if dark_path is not None:
        dark_img, _ = _load_bin(dark_path)
        if dark_img.shape != cal_raw.shape:
            print(f"  [WARN] Dark/cal shape mismatch — using synthetic dark.")
            dark_img = _make_synthetic_dark(cal_raw.shape)
        else:
            print(f"  Dark: {dark_path.name}  ({dark_img.shape[0]}×{dark_img.shape[1]})")
    else:
        print("  Using synthetic dark (bias=300 ADU, σ_read=3 ADU).")
        dark_img = _make_synthetic_dark(cal_raw.shape)

    # ── Step 2: Dark subtract + centre ───────────────────────────────────────
    print("\n[Step 2] Dark subtract + fringe centre...")
    cal_sub = np.clip(cal_raw - dark_img, 0, None).astype(np.float32)
    cx, cy  = _find_centre(cal_sub)
    print(f"  Centre: ({cx:.2f}, {cy:.2f}) px")

    # ── Step 3: Annular reduction ─────────────────────────────────────────────
    print("\n[Step 3] Annular reduction (500 bins)...")
    r_grid, profile, sigma_prof = _annular_reduce(
        cal_sub, cx, cy, r_max=r_max, n_bins=500)
    print(f"  Profile: [{profile.min():.0f}, {profile.max():.0f}] ADU  "
          f"(median σ = {np.median(sigma_prof):.1f} ADU)")

    # ── Step 4a: Two-line Tolansky ────────────────────────────────────────────
    print("\n[Step 4a] Two-line Tolansky analysis (Benoit gap recovery)...")
    tol_dict = _two_line_tolansky(r_grid, profile, binning)

    print(f"  Peaks found: λ_a={tol_dict['n_peaks_a']}  λ_b={tol_dict['n_peaks_b']}")
    print(f"  ε_640 = {tol_dict['epsilon_a']:.5f}   ε_638 = {tol_dict['epsilon_b']:.5f}")
    print(f"  N_Δ   = {tol_dict['N_delta']}")
    print(f"  d (Benoit) = {tol_dict['d_recovered']*1e3:.6f} mm   "
          f"(prior = {D_25C_MM*1e3:.6f} mm, "
          f"Δ = {(tol_dict['d_recovered']-D_25C_MM)*1e6:+.1f} nm)")
    print(f"  α     = {tol_dict['alpha']:.5e} rad/px  "
          f"(prior = {PLATE_SCALE_RPX:.5e})")
    for w in tol_dict["warnings"]:
        print(f"  [WARN] {w}")

    tolansky = _build_tolansky_result(tol_dict, binning)

    # ── Step 4b: F01 staged LM fit ────────────────────────────────────────────
    cal_result = None
    if _F01_AVAILABLE:
        print("\n[Step 4b] F01 staged Airy fit (A→B→C→D)...")

        fringe_profile = SimpleNamespace(
            r_grid        = r_grid,
            r2_grid       = r_grid**2,
            profile       = profile,
            sigma_profile = sigma_prof,
            masked        = np.zeros(len(r_grid), dtype=bool),
            r_max_px      = r_max,
            quality_flags = 0,
        )
        try:
            cal_result = fit_neon_fringe(fringe_profile, tolansky)
            print(f"  Converged: {cal_result.converged}   χ²_red = {cal_result.chi2_reduced:.4f}")
            print(f"  R_refl = {cal_result.R_refl:.5f} ± {cal_result.sigma_R_refl:.5f}")
            print(f"  α      = {cal_result.alpha:.5e} ± {cal_result.sigma_alpha:.2e} rad/px")
            print(f"  σ₀     = {cal_result.sigma0:.4f} ± {cal_result.sigma_sigma0:.4f} px")
            print(f"  B      = {cal_result.B:.1f} ± {cal_result.sigma_B:.2f} ADU")
            print(f"  Flags  = {hex(cal_result.quality_flags)}")
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            import traceback; traceback.print_exc()
    else:
        print("\n[Step 4b] Skipped — F01 module not available.")

    # ── Figure ────────────────────────────────────────────────────────────────
    print("\n[Output] Generating 8-panel diagnostic figure...")
    fig = _make_figure(
        cal_raw, dark_img, cal_sub, cx, cy,
        r_grid, profile, sigma_prof,
        tol_dict, tolansky, cal_result,
        binning, hdr,
    )

    stem = f"F01_{cal_path.stem}_result" if cal_path else "F01_synthetic_result"
    out  = _HERE / f"{stem}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=_NAVY)
    print(f"  Saved: {out}")
    plt.show()
    print("\nDone.\n")


if __name__ == "__main__":
    main()
