"""
validate_f01_step4b_airy_fit.py  (v3)
─────────────────────────────────────────────────────────────────────────────
Standalone, diagnostically-rich wrapper around F01 v3 (two-line Airy fit).

Upgrades from v2:
  • Y_B (638 nm intensity ratio) is now a free parameter — not hardcoded 0.8
  • 11 reported free parameters (10 fitted; Y_A = 1.0 fixed to break degeneracy)
  • 5-stage LM (A→E); correct column-scaled Jacobian covariance
  • Figures updated: chi² stages bar chart, two-line decomposition panel,
    11-parameter result table, new quality flags (YB_RATIO_LOW/HIGH)

INPUT FORMAT — 3-column CSV
─────────────────────────────
    r_grid_px,  profile_adu,  sigma_profile_adu

Lines beginning with '#' are comments.  Example header:
    # r_grid_px,  profile_adu,  sigma_profile_adu
    # binning: 2
    # r_max_px: 110.0

PROMPTS
───────
  (1) Windows file-open dialog — navigate to your 3-column profile CSV
      and click Open.  Cancel the dialog to use SYNTHETIC data instead.
      Falls back to a console path prompt if tkinter is unavailable.
  (2) Benoit gap d [mm]         — console prompt, default 20.0006
  (3) Plate-scale alpha [rad/px]— console prompt, default 1.6071e-4
  (4) r_max [px]                — console prompt, default 110
  (5) Binning (1 or 2)          — console prompt, default 2

OUTPUTS — three separate figures
──────────────────────────────────
  Figure 1:  Pre-fit diagnostic
    [0,0]  Data profile with annotated data-driven parameter estimates
    [0,1]  Initial parameter guesses vs bounds table
    [1,0]  Initial model vs data  (BEFORE any fitting; uses Y_B_init)
    [1,1]  Finesse/reflectivity estimate from fringe contrast scan

  Figure 2:  LM convergence and two-line model diagnostics
    [0,0]  Stage chi²_red bar chart (A → E, 5 stages)
    [0,1]  Two-line decomposition: A-family, B-family, total vs data
    [1,0]  Before (initial model) vs After (final model) vs data
    [1,1]  Quality summary: all flags with pass/fail annotation

  Figure 3:  Final result
    [0,0]  Final model vs data with residuals in σ
    [0,1]  Stage chi²_red progression bar chart
    [1,0]  Residual distribution histogram vs N(0,1)
    [1,1]  CalibrationResult: all 11 parameters with 1σ

Place this file in:  soc_sewell/validation/
─────────────────────────────────────────────────────────────────────────────
Author:  Claude AI / Scott Sewell  (NCAR/HAO)
Date:    2026-04-22
Spec:    specs/F01_full_airy_fit_to_neon_image_2026-04-22.md  v3
"""

from __future__ import annotations
import sys, pathlib
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── repo root ────────────────────────────────────────────────────────────────
_HERE = pathlib.Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# ── pipeline imports ─────────────────────────────────────────────────────────
try:
    from src.fpi.m01_airy_forward_model_2026_04_05 import airy_modified
    _M01_AVAILABLE = True
except ImportError:
    _M01_AVAILABLE = False

try:
    from src.fpi.f01_full_airy_fit_to_neon_image_2026_04_22 import (
        fit_neon_fringe, TolanskyResult, CalibrationResult, CalibrationFitFlags,
    )
    _F01_AVAILABLE = True
except ImportError:
    _F01_AVAILABLE = False
    print("[WARN] F01 v3 module not found — fit will not be available.")

# ── constants ────────────────────────────────────────────────────────────────
try:
    from src.constants import NE_WAVELENGTH_1_M, NE_WAVELENGTH_2_M, D_25C_MM, \
                              PLATE_SCALE_RPX, R_MAX_PX
except ImportError:
    NE_WAVELENGTH_1_M = 640.2248e-9
    NE_WAVELENGTH_2_M = 638.2991e-9
    D_25C_MM          = 20.0006e-3
    PLATE_SCALE_RPX   = 1.6071e-4
    R_MAX_PX          = 110

# ── palette ───────────────────────────────────────────────────────────────────
_NAVY  = "#003479"; _BLUE  = "#0057C2"; _TEAL  = "#009999"
_AMBER = "#C07000"; _RED   = "#CC2222"; _GREEN = "#22AA44"
_LGRAY = "#C8D4E8"; _DGRAY = "#1A2840"

def _fig_style():
    plt.rcParams.update({
        "figure.facecolor": _NAVY, "axes.facecolor": _DGRAY,
        "axes.edgecolor": _LGRAY, "axes.labelcolor": _LGRAY,
        "xtick.color": _LGRAY, "ytick.color": _LGRAY,
        "text.color": _LGRAY, "grid.color": "#253860",
        "grid.linewidth": 0.5, "font.family": "DejaVu Sans",
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    })


# ═══════════════════════════════════════════════════════════════════════════════
# CSV I/O
# ═══════════════════════════════════════════════════════════════════════════════

def load_profile_csv(path: pathlib.Path):
    meta = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith("#"): break
            if ":" in line:
                k, _, v = line.lstrip("#").partition(":")
                meta[k.strip()] = v.strip()
    df = pd.read_csv(path, comment="#", header="infer")
    if df.shape[1] < 3:
        raise ValueError(f"{path.name}: expected 3 columns, got {df.shape[1]}")
    try:
        return (df.iloc[:, 0].to_numpy(dtype=np.float32),
                df.iloc[:, 1].to_numpy(dtype=np.float32),
                df.iloc[:, 2].to_numpy(dtype=np.float32),
                meta)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"{path.name}: columns are not numeric — did you accidentally select "
            f"the _f01_result.csv instead of _annular_profile.csv?\n  ({exc})"
        ) from exc

def save_profile_csv(path, r, p, s, binning=2, r_max=110.0, source="synthetic"):
    with open(path, "w") as fh:
        fh.write("# r_grid_px,  profile_adu,  sigma_profile_adu\n")
        fh.write(f"# source: {source}\n# binning: {binning}\n# r_max_px: {r_max:.1f}\n")
        for ri, pi, si in zip(r, p, s):
            fh.write(f"{ri:.4f}, {pi:.4f}, {si:.4f}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Inline Airy  (zero-dependency fallback for pre-fit visualization)
# ═══════════════════════════════════════════════════════════════════════════════

def _airy(r, lam, d, R, alpha, n, r_max, I0, I1, I2, s0, s1, s2):
    theta = np.arctan(alpha * np.asarray(r, float))
    I_env = I0 * (1 + I1*(r/r_max) + I2*(r/r_max)**2)
    F     = 4*R / (1-R)**2
    phase = np.pi * 2*n*d*np.cos(theta) / lam
    A     = I_env / (1 + F * np.sin(phase)**2)
    sigma_r = s0 + s1*np.sin(np.pi*r/r_max) + s2*np.cos(np.pi*r/r_max)
    sm = float(np.mean(sigma_r))
    if sm < 1e-6: return A
    dr = float((r[-1]-r[0])/(len(r)-1)) if len(r)>1 else 1.0
    return gaussian_filter1d(A, sigma=sm/max(dr, 1e-9))


def _eval_model(r_eval, r_fine, d, R, alpha, n, r_max,
                I0, I1, I2, s0, s1, s2, Y_B, B):
    """
    Evaluate the two-line Airy model at r_eval via fine-grid interpolation.

    Y_B is the 638 nm intensity ratio relative to the 640 nm line.
    Y_A = 1.0 is implicit (fixed by convention, spec §4.2).
    """
    if _M01_AVAILABLE:
        kw = dict(t=d, R_refl=R, alpha=alpha, n=n, r_max=r_max,
                  I0=I0, I1=I1, I2=I2, sigma0=s0, sigma1=s1, sigma2=s2)
        fine_a = airy_modified(r_fine, NE_WAVELENGTH_1_M, **kw)
        fine_b = airy_modified(r_fine, NE_WAVELENGTH_2_M, **kw)
    else:
        fine_a = _airy(r_fine, NE_WAVELENGTH_1_M, d, R, alpha, n, r_max,
                       I0, I1, I2, s0, s1, s2)
        fine_b = _airy(r_fine, NE_WAVELENGTH_2_M, d, R, alpha, n, r_max,
                       I0, I1, I2, s0, s1, s2)
    model_fine = fine_a + Y_B * fine_b + B
    return np.interp(r_eval, r_fine, model_fine)


def _eval_model_components(r_eval, r_fine, d, R, alpha, n, r_max,
                            I0, I1, I2, s0, s1, s2, Y_B, B):
    """Return (total, A_only, B_only) evaluated at r_eval — for decomposition plots."""
    if _M01_AVAILABLE:
        kw = dict(t=d, R_refl=R, alpha=alpha, n=n, r_max=r_max,
                  I0=I0, I1=I1, I2=I2, sigma0=s0, sigma1=s1, sigma2=s2)
        fine_a = airy_modified(r_fine, NE_WAVELENGTH_1_M, **kw)
        fine_b = airy_modified(r_fine, NE_WAVELENGTH_2_M, **kw)
    else:
        fine_a = _airy(r_fine, NE_WAVELENGTH_1_M, d, R, alpha, n, r_max,
                       I0, I1, I2, s0, s1, s2)
        fine_b = _airy(r_fine, NE_WAVELENGTH_2_M, d, R, alpha, n, r_max,
                       I0, I1, I2, s0, s1, s2)
    a_at_r = np.interp(r_eval, r_fine, fine_a)
    b_at_r = np.interp(r_eval, r_fine, fine_b)
    return a_at_r + Y_B * b_at_r + B, a_at_r + B, Y_B * b_at_r + B


# ═══════════════════════════════════════════════════════════════════════════════
# DATA-DRIVEN INITIAL GUESS ESTIMATOR
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_initial_params(r_grid, profile, sigma_prof, d_m, alpha_init, r_max,
                             R_prior=0.53):
    """
    Derive physically-motivated initial guesses entirely from the 1D profile.
    Prints a detailed explanation of the reasoning for each parameter.
    Returns a dict of initial guesses (including Y_B_init) and a dict of bounds.
    """
    print("\n" + "─"*64)
    print("  DATA-DRIVEN INITIAL PARAMETER ESTIMATION")
    print("─"*64)

    # ── B: bias = Airy minimum ────────────────────────────────────────────────
    B_est = float(np.percentile(profile, 2))
    print(f"\n  B  (CCD bias)")
    print(f"     Method:  2nd percentile of profile  (Airy minimum ≈ B)")
    print(f"     Value:   {B_est:.1f} ADU")

    # ── Amplitude-based family separation → Y_B initial estimate ─────────────
    # The two neon lines (640 nm and 638 nm) create interleaved fringe families.
    # Sorting detected peaks by amplitude: top half → 640 nm (A-family, Y_A=1),
    # bottom half → 638 nm (B-family, Y_B to be fitted).
    win = max(5, int(len(profile)/30))
    peaks_idx, _ = find_peaks(profile, distance=win//2,
                               height=np.percentile(profile, 70))

    Y_B_init = 0.6   # reasonable prior
    I0_est = max(float(np.percentile(profile - B_est, 90)), 10.0)

    if len(peaks_idx) >= 4:
        peak_amps = profile[peaks_idx] - B_est
        sorted_pk = np.argsort(peak_amps)
        n_half    = len(sorted_pk) // 2
        dim_amps  = peak_amps[sorted_pk[:n_half]]
        bright_amps = peak_amps[sorted_pk[n_half:]]
        I0_est = max(float(np.percentile(bright_amps, 80)), 10.0)
        if len(dim_amps) >= 2 and np.median(bright_amps) > 0:
            Y_B_init = float(np.clip(
                np.median(dim_amps) / np.median(bright_amps), 0.1, 1.5))
        print(f"\n  Y_B  (638 nm / 640 nm intensity ratio)")
        print(f"     Method:  amplitude-ratio of dim vs bright peak families")
        print(f"              bright (640 nm) median amp = {np.median(bright_amps):.1f} ADU")
        print(f"              dim   (638 nm) median amp  = {np.median(dim_amps):.1f} ADU")
        print(f"     Value:   {Y_B_init:.3f}  (fitted freely in Stage A→E)")
    else:
        print(f"\n  Y_B  — fewer than 4 peaks found; using prior {Y_B_init:.2f}")

    # ── Fringe contrast → R estimate ─────────────────────────────────────────
    troughs_idx, _ = find_peaks(-profile, distance=win//2,
                                 height=-np.percentile(profile, 30))
    if len(peaks_idx) >= 2 and len(troughs_idx) >= 2:
        I_max = float(np.median(profile[peaks_idx]))
        I_min = float(np.median(profile[troughs_idx]))
        denom = I_max + I_min - 2*B_est
        C = (I_max - I_min) / denom if abs(denom) > 1 else 0.5
        C = float(np.clip(C, 0.05, 0.999))
        F_est  = 2*C / (1 - C)
        x = 2*(np.sqrt(1+F_est)-1)/F_est
        R_est = float(np.clip(1 - x, 0.1, 0.95))
    else:
        I_max = float(np.percentile(profile, 98))
        I_min = float(np.percentile(profile, 2))
        R_est = R_prior
        F_est  = 4*R_prior/(1-R_prior)**2
        C = F_est/(2+F_est)
        print(f"     [WARN] Too few peaks/troughs; using R_prior={R_prior}")

    # Use 0.53 prior when background pedestal dominates (contrast unreliable)
    bright_amps_ref = float(np.percentile(profile - B_est, 80))
    if B_est > bright_amps_ref:
        R_est = R_prior
        print(f"     [INFO] B ({B_est:.0f}) > fringe amplitude ({bright_amps_ref:.0f}); "
              f"using R_prior={R_prior}")

    print(f"\n  R  (plate reflectivity)")
    print(f"     Method:  fringe contrast C = (I_max−I_min)/(I_max+I_min−2B)")
    print(f"              I_max = {I_max:.1f} ADU,  I_min = {I_min:.1f} ADU")
    print(f"              C = {C:.4f}  →  F = {F_est:.3f}  →  R = {R_est:.4f}")

    # ── I₀: mean fringe peak intensity ───────────────────────────────────────
    print(f"\n  I₀  (mean fringe peak intensity)")
    print(f"     Method:  80th percentile of (peak_profile − B) for bright family")
    print(f"     Value:   {I0_est:.1f} ADU")
    print(f"     NOTE:    Y_A = 1.0 (fixed); I₀ absorbs the absolute 640 nm scale.")

    # ── σ₀: PSF width estimate ────────────────────────────────────────────────
    FSR_m      = NE_WAVELENGTH_1_M**2 / (2*d_m)
    finesse_airy = np.pi*np.sqrt(R_est)/(1-R_est)
    FSR_r_sq   = FSR_m / NE_WAVELENGTH_1_M / (alpha_init**2)
    FWHM_ideal_r = np.sqrt(max(FSR_r_sq, 0.0)) / finesse_airy if finesse_airy > 0 else 5.0
    sigma0_est = 0.5
    if len(peaks_idx) >= 3:
        mid = peaks_idx[len(peaks_idx)//2]
        half_max = B_est + (profile[mid] - B_est) / 2.0
        lo, hi = mid, mid
        while lo > 0 and profile[lo] > half_max: lo -= 1
        while hi < len(profile)-1 and profile[hi] > half_max: hi += 1
        FWHM_meas_bins = max(1, hi - lo)
        FWHM_meas_r = FWHM_meas_bins * (r_grid[-1]-r_grid[0]) / len(r_grid)
        excess_sq = max(0, FWHM_meas_r**2 - FWHM_ideal_r**2)
        sigma0_est = float(np.clip(
            np.sqrt(excess_sq) / (2*np.sqrt(2*np.log(2))) + 0.2, 0.1, 4.0))

    print(f"\n  σ₀  (PSF width)")
    print(f"     Finesse_Airy ≈ {finesse_airy:.2f}  →  σ₀_init = {sigma0_est:.3f} px")

    # ── α: cross-check ────────────────────────────────────────────────────────
    alpha_check = alpha_init
    if len(peaks_idx) >= 3:
        r2_peaks = r_grid[peaks_idx]**2
        dr2 = np.diff(r2_peaks)
        dr2_med = float(np.median(dr2[dr2 > 0])) if len(dr2) > 0 else None
        if dr2_med and dr2_med > 0:
            alpha_sq_check = NE_WAVELENGTH_1_M / (2*d_m * dr2_med)
            if alpha_sq_check > 0:
                alpha_check = float(np.sqrt(alpha_sq_check))

    print(f"\n  α  (plate scale)")
    print(f"     Tolansky WLS:    {alpha_init:.5e} rad/px")
    print(f"     r² peak spacing: {alpha_check:.5e} rad/px", end="")
    if abs(alpha_check - alpha_init)/alpha_init > 0.05:
        print(f"  [WARN >5% discrepancy — Tolansky value used]")
    else:
        print(f"  (Δ={abs(alpha_check-alpha_init)/alpha_init*100:.1f}%)  ✓")

    # ── Assemble ─────────────────────────────────────────────────────────────
    p0 = dict(R=R_est, alpha=alpha_init, I0=I0_est, I1=-0.1, I2=0.005,
              sigma0=sigma0_est, sigma1=0.0, sigma2=0.0, Y_B=Y_B_init, B=B_est)

    bounds = dict(
        R      = (0.10,          0.95),
        alpha  = (0.5*alpha_init, 2.0*alpha_init),
        I0     = (1.0,           2**14 - 1),
        I1     = (-0.5,          0.5),
        I2     = (-0.5,          0.5),
        sigma0 = (0.01,          5.0),
        sigma1 = (-3.0,          3.0),
        sigma2 = (-3.0,          3.0),
        Y_B    = (0.01,          float("inf")),
        B      = (0.0,           float("inf")),
    )

    print(f"\n{'─'*64}")
    print(f"  {'Parameter':<12}  {'Initial':>12}  {'Lower':>12}  {'Upper':>12}  Source")
    print(f"  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*18}")
    src = {"R": "contrast scan", "alpha": "Tolansky WLS",
           "I0": "80th pctile peaks", "I1": "hardcoded",
           "I2": "hardcoded", "sigma0": "FWHM analysis",
           "sigma1": "hardcoded", "sigma2": "hardcoded",
           "Y_B": "amplitude ratio", "B": "2nd pctile"}
    fmt = {"R": ":.5f", "alpha": ":.4e", "I0": ":.1f", "I1": ":.4f", "I2": ":.4f",
           "sigma0": ":.4f", "sigma1": ":.4f", "sigma2": ":.4f",
           "Y_B": ":.4f", "B": ":.1f"}
    for name in ["R", "alpha", "I0", "I1", "I2", "sigma0", "sigma1", "sigma2",
                 "Y_B", "B"]:
        f = fmt[name]
        v  = p0[name]
        lo, hi = bounds[name]
        lo_s = f"{lo:.4g}" if np.isfinite(lo) else "−∞"
        hi_s = f"{hi:.4g}" if np.isfinite(hi) else "+∞"
        print(f"  {name:<12}  {format(v, f[1:]):>12}  {lo_s:>12}  {hi_s:>12}  {src[name]}")
    print(f"  {'d (FIXED)':<12}  {d_m*1e3:>12.6f}  {'—':>12}  {'—':>12}  Benoit/Tolansky")
    print(f"  {'Y_A (FIXED)':<12}  {'1.000000':>12}  {'—':>12}  {'—':>12}  degeneracy break")
    print(f"{'─'*64}")

    return p0, bounds


# ═══════════════════════════════════════════════════════════════════════════════
# F01 v3 fit wrapper
# ═══════════════════════════════════════════════════════════════════════════════

def _run_fit_v3(r_grid, profile, sigma_prof, d_m, r_max, alpha_init,
                R_init=None):
    """
    Wrap fit_neon_fringe (F01 v3) to accept raw arrays.

    Creates FringeProfile-compatible SimpleNamespace and TolanskyResult
    from the numeric inputs, calls the staged LM fitter, and returns
    the CalibrationResult directly.
    """
    if not _F01_AVAILABLE:
        raise RuntimeError(
            "F01 v3 module is not importable. "
            "Ensure src/fpi/f01_full_airy_fit_to_neon_image_2026_04_22.py exists."
        )
    profile_ns = SimpleNamespace(
        r_grid        = r_grid,
        r2_grid       = r_grid ** 2,
        profile       = profile,
        sigma_profile = sigma_prof,
        masked        = np.zeros(len(r_grid), dtype=bool),
        r_max_px      = float(r_max),
        quality_flags = 0,
    )
    tolansky = TolanskyResult(
        t_m         = float(d_m),
        alpha_rpx   = float(alpha_init),
        epsilon_640 = 0.0,
        epsilon_638 = 0.0,
        epsilon_cal = 0.0,
    )
    return fit_neon_fringe(profile_ns, tolansky, R_init=R_init)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: Pre-fit diagnostic
# ═══════════════════════════════════════════════════════════════════════════════

def _fig1_prefit(r_grid, profile, sigma_prof, p0, bounds,
                 d_m, alpha_init, r_max, source_label):
    _fig_style()
    fig = plt.figure(figsize=(18, 12), facecolor=_NAVY)
    fig.suptitle(
        f"F01 v3 Step 4b — Pre-Fit Diagnostic  |  source: {source_label}",
        color="white", fontsize=12, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, left=0.07, right=0.97,
                           top=0.93, bottom=0.06, hspace=0.40, wspace=0.32)

    param_names = ["R", "alpha", "I0", "I1", "I2", "sigma0", "sigma1",
                   "sigma2", "Y_B", "B"]
    r_fine = np.linspace(r_grid[0], r_grid[-1], 1000)

    # ── [0,0]  Annotated data profile ────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(r_grid, profile, color=_LGRAY, lw=1.0, alpha=0.8, label="Data")
    ax.fill_between(r_grid, profile-sigma_prof, profile+sigma_prof,
                    color=_LGRAY, alpha=0.12)
    B_est  = p0["B"]; I0_est = p0["I0"]
    ax.axhline(B_est, color=_AMBER, lw=1.2, ls="--",
               label=f"B init = {B_est:.0f} ADU  (2nd pctile)")
    ax.axhline(B_est + I0_est, color=_GREEN, lw=1.2, ls="--",
               label=f"B+I₀ = {B_est+I0_est:.0f} ADU  (bright peaks)")
    ax.set_xlabel("Radius  r  (px)"); ax.set_ylabel("Counts  (ADU)")
    ax.set_title("Data Profile with Initial Estimate Annotations", color="white", pad=5)
    ax.grid(True)
    ax.legend(fontsize=8, facecolor=_DGRAY, edgecolor=_LGRAY, labelcolor=_LGRAY)

    # ── [0,1]  Parameter table with bounds ───────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    ax.axis("off")
    ax.set_title("Initial Guesses and Fit Bounds (v3 — 10 free params + 2 fixed)",
                 color="white", pad=5)
    fmt_map = {"R": ":.5f", "alpha": ":.4e", "I0": ":.1f", "I1": ":.4f",
               "I2": ":.4f", "sigma0": ":.4f", "sigma1": ":.4f",
               "sigma2": ":.4f", "Y_B": ":.4f", "B": ":.1f"}
    src_map = {"R": "fringe contrast", "alpha": "Tolansky WLS",
               "I0": "80th pctile peaks", "I1": "hardcoded",
               "I2": "hardcoded", "sigma0": "FWHM analysis",
               "sigma1": "hardcoded", "sigma2": "hardcoded",
               "Y_B": "amplitude ratio", "B": "2nd pctile"}
    headers = ["Param", "Init. guess", "Lower", "Upper", "Source"]
    col_x = [0.02, 0.18, 0.38, 0.55, 0.70]
    y0 = 0.96; dy = 0.064
    for ci, h in enumerate(headers):
        ax.text(col_x[ci], y0, h, transform=ax.transAxes,
                ha="left", va="top", fontsize=9, color="white",
                fontweight="bold", fontfamily="monospace")

    for ri, name in enumerate(param_names):
        y = y0 - (ri+1)*dy
        v = p0[name]; lo, hi = bounds[name]
        lo_s = f"{lo:.3g}" if np.isfinite(lo) else "−∞"
        hi_s = f"{hi:.3g}" if np.isfinite(hi) else "+∞"
        f = fmt_map[name]
        pct = (v - lo)/(hi - lo) if (np.isfinite(lo) and np.isfinite(hi)) else 0.5
        col = _RED if (pct < 0.05 or pct > 0.95) else _LGRAY
        for ci, cell in enumerate([name, format(v, f[1:]), lo_s, hi_s, src_map[name]]):
            ax.text(col_x[ci], y, cell, transform=ax.transAxes,
                    ha="left", va="top", fontsize=8, color=col,
                    fontfamily="monospace")
    # Fixed-param rows
    for extra_row, label, val, note in [
        (len(param_names)+1, "d  (FIXED)",  f"{d_m*1e3:.6f} mm", "Benoit/Tolansky"),
        (len(param_names)+2, "Y_A (FIXED)", "1.000000",           "degeneracy break"),
    ]:
        y = y0 - extra_row*dy
        ax.text(col_x[0], y, label, transform=ax.transAxes,
                ha="left", va="top", fontsize=8, color=_AMBER, fontfamily="monospace")
        ax.text(col_x[1], y, val, transform=ax.transAxes,
                ha="left", va="top", fontsize=8, color=_AMBER, fontfamily="monospace")
        ax.text(col_x[4], y, note, transform=ax.transAxes,
                ha="left", va="top", fontsize=8, color=_AMBER, fontfamily="monospace")

    # ── [1,0]  Initial model vs data ─────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    model_init = _eval_model(
        r_grid, r_fine, d_m,
        p0["R"], p0["alpha"], 1.0, r_max,
        p0["I0"], p0["I1"], p0["I2"],
        p0["sigma0"], p0["sigma1"], p0["sigma2"],
        p0["Y_B"], p0["B"])
    model_no_b = _eval_model(
        r_grid, r_fine, d_m,
        p0["R"], p0["alpha"], 1.0, r_max,
        p0["I0"], p0["I1"], p0["I2"],
        p0["sigma0"], p0["sigma1"], p0["sigma2"],
        0.0, p0["B"])

    ax.plot(r_grid, profile, color=_LGRAY, lw=0.9, alpha=0.7, label="Data")
    ax.plot(r_grid, model_init, color=_AMBER, lw=1.8,
            label=f"Init model (640+638 nm, Y_B={p0['Y_B']:.2f})")
    ax.plot(r_grid, model_no_b, color=_TEAL,  lw=1.2, ls="--",
            label="Init model (640 nm only)")
    ax.set_xlabel("Radius  r  (px)"); ax.set_ylabel("Counts  (ADU)")
    ax.set_title("Initial Model vs Data  (BEFORE fitting — d, α, Y_A fixed)",
                 color="white", pad=5)
    ax.grid(True)
    ax.legend(fontsize=8, facecolor=_DGRAY, edgecolor=_LGRAY, labelcolor=_LGRAY)
    sigma_safe = np.maximum(sigma_prof, 1.0)
    chi2_init = float(np.mean(((profile - model_init)/sigma_safe)**2))
    ax.text(0.97, 0.97, f"χ²_red (init) = {chi2_init:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            color=_AMBER,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=_DGRAY,
                      edgecolor=_AMBER, alpha=0.9))

    # ── [1,1]  Local R estimate from contrast ─────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    win = max(8, int(len(profile)/20))
    contrast_r, contrast_c = [], []
    for i in range(win, len(profile)-win, win//2):
        chunk = profile[i-win:i+win]
        lo_v  = np.percentile(chunk, 5)
        hi_v  = np.percentile(chunk, 95)
        denom = hi_v + lo_v - 2*p0["B"]
        if abs(denom) > 5:
            C_loc  = float(np.clip((hi_v - lo_v) / denom, 0.05, 0.99))
            F_loc  = 2*C_loc / (1-C_loc)
            x_loc  = 2*(np.sqrt(1+F_loc)-1)/F_loc
            contrast_r.append(r_grid[i])
            contrast_c.append(float(np.clip(1-x_loc, 0.05, 0.98)))

    if contrast_r:
        ax.plot(contrast_r, contrast_c, color=_BLUE, lw=1.5,
                label="Local R estimate (from contrast)")
        ax.axhline(p0["R"], color=_AMBER, lw=1.5, ls="--",
                   label=f"Global R init = {p0['R']:.4f}")
        ax.axhline(0.53, color=_TEAL, lw=1.0, ls=":",
                   label="FlatSat prior R = 0.53")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Radius  r  (px)"); ax.set_ylabel("Estimated  R")
        ax.set_title("Local Reflectivity Estimate from Fringe Contrast",
                     color="white", pad=5)
        ax.legend(fontsize=8, facecolor=_DGRAY, edgecolor=_LGRAY, labelcolor=_LGRAY)
        ax.grid(True)
    else:
        ax.text(0.5, 0.5, "Insufficient contrast data", transform=ax.transAxes,
                ha="center", va="center", color=_AMBER)

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: LM convergence and two-line model diagnostics
# ═══════════════════════════════════════════════════════════════════════════════

def _fig2_convergence(r_grid, profile, sigma_prof, cal, p0,
                      d_m, r_max, source_label):
    """
    Figure 2 (v3): four-panel convergence and two-line decomposition view.

    [0,0]  Stage chi²_red bar chart (A → E, 5 stages)
    [0,1]  Two-line decomposition: A-family, B-family, total vs data
    [1,0]  Before (initial guess) vs After (final model) vs data
    [1,1]  Quality summary: all flags with pass/fail annotation
    """
    _fig_style()
    fig = plt.figure(figsize=(18, 12), facecolor=_NAVY)
    fig.suptitle(
        f"F01 v3 Step 4b — LM Convergence & Two-Line Diagnostics  |  {source_label}",
        color="white", fontsize=12, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, left=0.07, right=0.97,
                           top=0.93, bottom=0.06, hspace=0.40, wspace=0.32)

    r_fine = np.linspace(r_grid[0], r_grid[-1], 1000)
    sigma_safe = np.maximum(sigma_prof, 1.0)

    # ── [0,0]  Stage chi² bar chart ──────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    stage_labels = ["A\n(Y_B,B)", "B\n(Y_B,I₀,B)", "C\n(+I₁,I₂)",
                    "D\n(+R,α)", "E\n(all 10)"]
    chi2s  = list(cal.chi2_stages)
    colors = [_TEAL, _BLUE, _AMBER, _GREEN, "#FF88AA"]
    bars = ax.bar(stage_labels, chi2s, color=colors, edgecolor=_LGRAY,
                  linewidth=0.7, width=0.6)
    for bar, v in zip(bars, chi2s):
        ax.text(bar.get_x()+bar.get_width()/2,
                v + max(chi2s)*0.02,
                f"{v:.3f}", ha="center", va="bottom",
                fontsize=8, color="white", fontweight="bold")
    ax.axhline(1.0, color=_GREEN, lw=1.2, ls="--", label="χ²=1 (ideal)")
    ax.axhline(3.0, color=_RED,   lw=1.0, ls=":",  label="χ²=3 (caution)")
    ax.set_ylabel("χ²_red"); ax.set_ylim(bottom=0)
    ax.set_title("Stage-by-Stage χ² Reduction  (5 stages, F01 v3)", color="white", pad=5)
    ax.legend(fontsize=8, facecolor=_DGRAY, edgecolor=_LGRAY, labelcolor=_LGRAY)
    ax.grid(axis="y")

    # ── [0,1]  Two-line decomposition ────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    total, a_only, b_only = _eval_model_components(
        r_grid, r_fine, d_m,
        cal.R_refl, cal.alpha, 1.0, r_max,
        cal.I0, cal.I1, cal.I2,
        cal.sigma0, cal.sigma1, cal.sigma2,
        cal.Y_B, cal.B)
    ax.plot(r_grid, profile, color=_LGRAY, lw=0.9, alpha=0.65, label="Data")
    ax.plot(r_grid, total,  color=_AMBER, lw=2.0,
            label=f"Total model (χ²={cal.chi2_reduced:.3f})")
    ax.plot(r_grid, a_only, color=_TEAL,  lw=1.2, ls="--",
            label=f"640 nm only  (Y_A=1.0)")
    ax.plot(r_grid, b_only, color=_GREEN, lw=1.2, ls=":",
            label=f"638 nm only  (Y_B={cal.Y_B:.3f})")
    ax.set_xlabel("Radius  r  (px)"); ax.set_ylabel("Counts  (ADU)")
    ax.set_title(
        f"Two-Line Decomposition  |  Y_B/Y_A = {cal.intensity_ratio:.4f} "
        f"(±{cal.sigma_Y_B:.4f})",
        color="white", pad=5)
    ax.grid(True)
    ax.legend(fontsize=8, facecolor=_DGRAY, edgecolor=_LGRAY, labelcolor=_LGRAY)

    # ── [1,0]  Before vs After ────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    model_init = _eval_model(
        r_grid, r_fine, d_m,
        p0["R"], p0["alpha"], 1.0, r_max,
        p0["I0"], p0["I1"], p0["I2"],
        p0["sigma0"], p0["sigma1"], p0["sigma2"],
        p0["Y_B"], p0["B"])
    ax.plot(r_grid, profile,    color=_LGRAY, lw=0.9, alpha=0.6, label="Data")
    ax.plot(r_grid, model_init, color=_BLUE,  lw=1.4, ls="--",
            label=f"Initial model  (R={p0['R']:.3f}, Y_B={p0['Y_B']:.2f})")
    ax.plot(r_grid, total,      color=_AMBER, lw=1.8,
            label=f"Final model    (R={cal.R_refl:.4f}, Y_B={cal.Y_B:.3f})")
    ax.set_xlabel("Radius  r  (px)"); ax.set_ylabel("Counts  (ADU)")
    ax.set_title("Before vs After: Initial Guess → Final Fit", color="white", pad=5)
    ax.grid(True)
    ax.legend(fontsize=8, facecolor=_DGRAY, edgecolor=_LGRAY, labelcolor=_LGRAY)

    # ── [1,1]  Quality flag summary ───────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    ax.set_title("Quality Flag Summary  (F01 v3 flags)", color="white", pad=5)

    all_flags = [
        (CalibrationFitFlags.FIT_FAILED,     "FIT_FAILED",     "LM did not converge"),
        (CalibrationFitFlags.CHI2_HIGH,       "CHI2_HIGH",      "χ²_red > 3.0"),
        (CalibrationFitFlags.CHI2_VERY_HIGH,  "CHI2_VERY_HIGH", "χ²_red > 10.0"),
        (CalibrationFitFlags.CHI2_LOW,        "CHI2_LOW",       "χ²_red < 0.5"),
        (CalibrationFitFlags.STDERR_NONE,     "STDERR_NONE",    "singular Jacobian"),
        (CalibrationFitFlags.R_AT_BOUND,      "R_AT_BOUND",     "R hit bound [0.10, 0.95]"),
        (CalibrationFitFlags.ALPHA_AT_BOUND,  "ALPHA_AT_BOUND", "α hit [0.5×, 2×] init"),
        (CalibrationFitFlags.FEW_BINS,        "FEW_BINS",       "n_good < 100"),
        (CalibrationFitFlags.YB_RATIO_LOW,    "YB_RATIO_LOW",   "Y_B/Y_A < 0.30"),
        (CalibrationFitFlags.YB_RATIO_HIGH,   "YB_RATIO_HIGH",  "Y_B/Y_A > 1.00"),
    ]

    col_x = [0.02, 0.40, 0.65]
    y0_ = 0.95; dy_ = 0.075
    for ci, hdr in enumerate(["Flag", "Status", "Meaning"]):
        ax.text(col_x[ci], y0_, hdr, transform=ax.transAxes,
                ha="left", va="top", fontsize=9, color="white",
                fontweight="bold", fontfamily="monospace")

    for ri, (bit, name, meaning) in enumerate(all_flags):
        y = y0_ - (ri+1)*dy_
        fired = bool(cal.quality_flags & bit)
        col   = _RED if fired else _GREEN
        ax.text(col_x[0], y, name,   transform=ax.transAxes,
                ha="left", va="top", fontsize=8, color=col, fontfamily="monospace")
        ax.text(col_x[1], y, "RAISED" if fired else "ok",
                transform=ax.transAxes,
                ha="left", va="top", fontsize=8, color=col,
                fontweight="bold" if fired else "normal", fontfamily="monospace")
        ax.text(col_x[2], y, meaning, transform=ax.transAxes,
                ha="left", va="top", fontsize=7.5, color=_LGRAY, fontfamily="monospace")

    status = "GOOD" if cal.quality_flags == 0 else "FLAGS SET"
    col    = _GREEN if cal.quality_flags == 0 else _RED
    ax.text(0.02, 0.02,
            f"Overall: {status}  |  χ²_red = {cal.chi2_reduced:.4f}  "
            f"|  n_bins = {cal.n_bins_used}  |  n_params = {cal.n_params_free}",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9, fontweight="bold", color=col)

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3: Final result
# ═══════════════════════════════════════════════════════════════════════════════

def _fig3_final(r_grid, profile, sigma_prof, cal, d_m, alpha_init, r_max,
                source_label):
    _fig_style()
    fig = plt.figure(figsize=(18, 12), facecolor=_NAVY)
    fig.suptitle(
        f"F01 v3 Step 4b — Final Result  |  source: {source_label}",
        color="white", fontsize=12, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, left=0.07, right=0.97,
                           top=0.93, bottom=0.06, hspace=0.40, wspace=0.32)

    r_fine = np.linspace(r_grid[0], r_grid[-1], 1000)
    sigma_safe = np.maximum(sigma_prof, 1.0)

    model_final = _eval_model(
        r_grid, r_fine, d_m,
        cal.R_refl, cal.alpha, 1.0, r_max,
        cal.I0, cal.I1, cal.I2,
        cal.sigma0, cal.sigma1, cal.sigma2,
        cal.Y_B, cal.B)

    # ── [0,0]  Final model vs data + residuals ────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(r_grid, profile, color=_LGRAY, lw=0.9, alpha=0.7, label="Data")
    ax.fill_between(r_grid, profile-sigma_prof, profile+sigma_prof,
                    color=_LGRAY, alpha=0.12)
    ax.plot(r_grid, model_final, color=_AMBER, lw=2.0,
            label=f"Final fit  (χ²_red = {cal.chi2_reduced:.4f})")
    ax_r = ax.twinx()
    resid = (profile - model_final) / sigma_safe
    ax_r.plot(r_grid, resid, color=_RED, lw=0.7, alpha=0.55)
    ax_r.axhline(0, color=_RED, lw=0.8, ls="--", alpha=0.4)
    ax_r.set_ylabel("Residual (σ)", color=_RED, fontsize=8)
    ax_r.tick_params(axis="y", colors=_RED); ax_r.set_ylim(-6, 6)
    conv_str = "✓ Converged" if cal.converged else "✗ Did not converge"
    ax.set_title(f"Final Model vs Data  |  {conv_str}", color="white", pad=5)
    ax.set_xlabel("Radius  r  (px)"); ax.set_ylabel("Counts  (ADU)")
    ax.grid(True)
    ax.legend(fontsize=8, facecolor=_DGRAY, edgecolor=_LGRAY, labelcolor=_LGRAY)

    # ── [0,1]  Stage chi² progression bar chart ───────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    stage_labels = ["A\n(Y_B,B)", "B\n(Y_B,I₀,B)", "C\n(+I₁,I₂)",
                    "D\n(+R,α)", "E\n(all 10)"]
    chi2s  = list(cal.chi2_stages)
    colors = [_TEAL, _BLUE, _AMBER, _GREEN, "#FF88AA"]
    bars = ax.bar(stage_labels, chi2s, color=colors, edgecolor=_LGRAY,
                  linewidth=0.7, width=0.6)
    for bar, v in zip(bars, chi2s):
        ax.text(bar.get_x()+bar.get_width()/2,
                v + max(chi2s)*0.02,
                f"{v:.3f}", ha="center", va="bottom",
                fontsize=8, color="white", fontweight="bold")
    ax.axhline(1.0, color=_GREEN, lw=1.2, ls="--", label="χ²=1 (ideal)")
    ax.axhline(3.0, color=_RED,   lw=1.0, ls=":",  label="χ²=3 (caution)")
    ax.set_ylabel("χ²_red"); ax.set_ylim(bottom=0)
    ax.set_title("Stage-by-Stage χ² Reduction", color="white", pad=5)
    ax.legend(fontsize=8, facecolor=_DGRAY, edgecolor=_LGRAY, labelcolor=_LGRAY)
    ax.grid(axis="y")

    # ── [1,0]  Residual histogram ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    good = np.isfinite(resid)
    ax.hist(resid[good], bins=40, color=_BLUE, edgecolor=_NAVY,
            alpha=0.8, density=True, label="Normalised residuals")
    xs = np.linspace(-5, 5, 300)
    ax.plot(xs, np.exp(-xs**2/2)/np.sqrt(2*np.pi), color=_GREEN,
            lw=2.0, label="N(0,1) ideal")
    rms_r = float(np.std(resid[good]))
    ax.text(0.97, 0.97,
            f"RMS residual = {rms_r:.3f} σ\nideal = 1.000 σ",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            color=_AMBER,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=_DGRAY,
                      edgecolor=_AMBER, alpha=0.9))
    ax.set_xlabel("Residual (σ)"); ax.set_ylabel("Density")
    ax.set_title("Residual Distribution vs Ideal N(0,1)", color="white", pad=5)
    ax.legend(fontsize=8, facecolor=_DGRAY, edgecolor=_LGRAY, labelcolor=_LGRAY)
    ax.grid(True)

    # ── [1,1]  Parameter table (all 11 free params) ───────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    ax.set_title("CalibrationResult — 11 free params (Y_A=1.0 fixed) + d fixed",
                 color="white", pad=5)
    rows = [
        ("Param",    "Fitted",                        "1σ",                     "Unit"),
        ("─"*10,     "─"*14,                          "─"*11,                   "─"*5),
        ("d FIXED",  f"{cal.t_m*1e3:.6f}",             "(Tolansky)",             "mm"),
        ("R_refl",   f"{cal.R_refl:.5f}",               f"±{cal.sigma_R_refl:.5f}", "—"),
        ("α",        f"{cal.alpha:.4e}",                 f"±{cal.sigma_alpha:.2e}",  "rad/px"),
        ("I₀",       f"{cal.I0:.1f}",                    f"±{cal.sigma_I0:.1f}",     "ADU"),
        ("I₁",       f"{cal.I1:.5f}",                    f"±{cal.sigma_I1:.5f}",     "—"),
        ("I₂",       f"{cal.I2:.5f}",                    f"±{cal.sigma_I2:.5f}",     "—"),
        ("σ₀",       f"{cal.sigma0:.5f}",                f"±{cal.sigma_sigma0:.5f}", "px"),
        ("σ₁",       f"{cal.sigma1:.5f}",                f"±{cal.sigma_sigma1:.5f}", "px"),
        ("σ₂",       f"{cal.sigma2:.5f}",                f"±{cal.sigma_sigma2:.5f}", "px"),
        ("Y_A FIXED","1.00000",                          "(fixed)",                  "—"),
        ("Y_B",      f"{cal.Y_B:.5f}",                   f"±{cal.sigma_Y_B:.5f}",    "—"),
        ("Y_B/Y_A",  f"{cal.intensity_ratio:.5f}",       "(ratio)",                  "—"),
        ("B",        f"{cal.B:.2f}",                     f"±{cal.sigma_B:.3f}",      "ADU"),
        ("─"*10,     "─"*14,                          "─"*11,                   "─"*5),
        ("χ²_red",   f"{cal.chi2_reduced:.5f}",          "",                         ""),
        ("α Δ%",     f"{(cal.alpha-alpha_init)/alpha_init*100:+.2f}%",
                     "(vs Tolansky init)",                                           ""),
        ("n_bins",   str(cal.n_bins_used),                "",                         ""),
    ]
    cx_ = [0.02, 0.30, 0.60, 0.87]
    y0_ = 0.97; dy_ = 0.048
    for ri, row in enumerate(rows):
        y = y0_ - ri*dy_
        is_hdr   = (ri == 0)
        is_rule  = row[0].startswith("─")
        is_fixed = row[0].endswith("FIXED") or row[0].startswith("d ")
        col = ("white" if is_hdr else _AMBER if is_rule else
               "#D09000" if is_fixed else _LGRAY)
        fw  = "bold" if is_hdr else "normal"
        for ci, cell in enumerate(row):
            ax.text(cx_[ci], y, cell, transform=ax.transAxes,
                    ha="left", va="top", fontsize=8.0,
                    color=col, fontweight=fw, fontfamily="monospace")

    # Quality flags summary
    flag_bits = [
        (CalibrationFitFlags.FIT_FAILED,    "FIT_FAILED"),
        (CalibrationFitFlags.CHI2_HIGH,     "CHI2_HIGH"),
        (CalibrationFitFlags.CHI2_VERY_HIGH,"CHI2_VERY_HIGH"),
        (CalibrationFitFlags.CHI2_LOW,      "CHI2_LOW"),
        (CalibrationFitFlags.STDERR_NONE,   "STDERR_NONE"),
        (CalibrationFitFlags.R_AT_BOUND,    "R_AT_BOUND"),
        (CalibrationFitFlags.ALPHA_AT_BOUND,"ALPHA_AT_BOUND"),
        (CalibrationFitFlags.FEW_BINS,      "FEW_BINS"),
        (CalibrationFitFlags.YB_RATIO_LOW,  "YB_RATIO_LOW"),
        (CalibrationFitFlags.YB_RATIO_HIGH, "YB_RATIO_HIGH"),
    ]
    fired = [name for bit, name in flag_bits if cal.quality_flags & bit]
    y_flag = y0_ - len(rows)*dy_ - 0.02
    ax.text(0.02, y_flag,
            f"Quality: {'GOOD' if not fired else ', '.join(fired)}",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            fontweight="bold",
            color=_GREEN if not fired else _RED)

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Prompts + main
# ═══════════════════════════════════════════════════════════════════════════════

def _ask(label, default, unit=""):
    u = f" [{unit}]" if unit else ""
    ans = input(f"  {label}{u}  [default: {default}]: ").strip()
    return ans if ans else default


def _prompt_csv_dialog() -> Optional[pathlib.Path]:
    """
    Open a Windows file-open dialog to locate the 3-column profile CSV.
    Falls back to a console prompt if tkinter is unavailable.
    Returns the chosen Path, or None to use synthetic data.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        print("\n  Opening file dialog — select your 3-column profile CSV.")
        print("  (Cancel the dialog to use SYNTHETIC data instead.)\n")

        path_str = filedialog.askopenfilename(
            title     = "Select annular-profile CSV  "
                        "(r_grid_px, profile_adu, sigma_profile_adu)",
            filetypes = [
                ("Annular profile CSV", "*_annular_profile.csv"),
                ("All files",           "*.*"),
            ],
            initialdir = str(_HERE),
        )
        root.destroy()

        if not path_str:
            print("  Dialog cancelled — using SYNTHETIC two-line neon profile.")
            return None

        p = pathlib.Path(path_str)
        if not p.exists():
            print(f"  [ERROR] Selected file not found: {p}")
            sys.exit(1)
        print(f"  Selected: {p}")
        return p

    except Exception as _tk_err:
        print(f"  [INFO] tkinter unavailable ({_tk_err}).  Falling back to console.")
        print(f"\n{'─'*64}")
        print("  PROFILE CSV  (3-column: r_px, profile_adu, sigma_adu)")
        print("  Press Enter for SYNTHETIC two-line neon profile.")
        print(f"{'─'*64}")
        raw = input("  Path: ").strip().strip('"').strip("'")
        if not raw:
            return None
        p = pathlib.Path(raw)
        if not p.exists():
            print(f"  [ERROR] Not found: {p}")
            sys.exit(1)
        return p


def main():
    print("\n" + "═"*64)
    print("  F01 v3 Step 4b — Two-Line Airy Fit  (Diagnostic Edition)")
    print("  WindCube FPI Pipeline  ·  NCAR/HAO")
    print("═"*64)

    csv_path = _prompt_csv_dialog()

    # Quick metadata peek (α default from CSV header if present)
    _meta_peek: dict = {}
    if csv_path:
        with open(csv_path) as _fh:
            for _line in _fh:
                _line = _line.strip()
                if not _line.startswith("#"):
                    break
                if ":" in _line:
                    _k, _, _v = _line.lstrip("#").partition(":")
                    _meta_peek[_k.strip()] = _v.strip()
    _alpha_default = (
        f"{float(_meta_peek['alpha_rad_px']):.4e}"
        if "alpha_rad_px" in _meta_peek else f"{PLATE_SCALE_RPX:.4e}"
    )

    print("\n  Fit parameters (press Enter to accept default):\n")
    d_mm       = float(_ask("Benoit gap  d",  f"{D_25C_MM*1e3:.6f}", "mm"))
    alpha_init = float(_ask("Plate scale α",  _alpha_default,         "rad/px"))
    r_max_val  = float(_ask("r_max",           str(R_MAX_PX),            "px"))
    binning    = int(_ask("Binning",           "2",                      "1 or 2"))

    if binning == 1 and alpha_init > 1e-4:
        print(f"  [NOTE] 1×1 binning: α scaled {alpha_init:.4e} → {alpha_init/2:.4e}")
        alpha_init /= 2.0

    d_m = d_mm * 1e-3

    # ── Load / generate profile ───────────────────────────────────────────────
    if csv_path:
        print(f"\n  Loading {csv_path.name} ...")
        r_grid, profile, sigma_prof, meta = load_profile_csv(csv_path)
        if "r_max_px" in meta and r_max_val == R_MAX_PX:
            r_max_val = float(meta["r_max_px"])
            print(f"  r_max from CSV header: {r_max_val:.1f} px")
        source_label = csv_path.name
        stem = csv_path.stem
    else:
        print(f"\n  Generating synthetic two-line neon profile ...")
        rng  = np.random.default_rng(42)
        r1d  = np.linspace(0.5, r_max_val, 500).astype(np.float32)
        if _M01_AVAILABLE:
            kw = dict(t=d_m, R_refl=0.53, alpha=alpha_init, n=1.0,
                      r_max=r_max_val, I0=1000.0, I1=-0.1, I2=0.005,
                      sigma0=0.5, sigma1=0.0, sigma2=0.0)
            sig = (airy_modified(r1d, NE_WAVELENGTH_1_M, **kw) +
                   airy_modified(r1d, NE_WAVELENGTH_2_M, **kw)*0.6 + 300.0)
        else:
            sig = (_airy(r1d, NE_WAVELENGTH_1_M, d_m, 0.53, alpha_init, 1.0, r_max_val,
                         1000., -0.1, 0.005, 0.5, 0., 0.) +
                   _airy(r1d, NE_WAVELENGTH_2_M, d_m, 0.53, alpha_init, 1.0, r_max_val,
                         1000., -0.1, 0.005, 0.5, 0., 0.)*0.6 + 300.0)
        noisy = rng.poisson(np.maximum(sig, 1)).astype(np.float32)
        sigma = np.maximum(np.sqrt(noisy), 1.0).astype(np.float32)
        r_grid, profile, sigma_prof = r1d, noisy, sigma
        source_label = "synthetic"; stem = "synthetic"
        out_csv = _HERE / f"synthetic_profile_{binning}x{binning}.csv"
        save_profile_csv(out_csv, r_grid, profile, sigma_prof,
                         binning=binning, r_max=r_max_val)
        print(f"  Saved: {out_csv.name}")

    print(f"  Bins: {len(r_grid)},  r=[{r_grid[0]:.1f},{r_grid[-1]:.1f}] px,  "
          f"counts=[{profile.min():.0f},{profile.max():.0f}]")

    # ── Step 1: estimate initial params ───────────────────────────────────────
    p0, bounds = estimate_initial_params(
        r_grid, profile, sigma_prof, d_m, alpha_init, r_max_val)

    # ── Figure 1: pre-fit ─────────────────────────────────────────────────────
    _fig_style()
    fig1 = _fig1_prefit(r_grid, profile, sigma_prof, p0, bounds,
                        d_m, alpha_init, r_max_val, source_label)
    out1 = _HERE / f"{stem}_f01_step4b_prefit.png"
    fig1.savefig(out1, dpi=150, bbox_inches="tight", facecolor=_NAVY)
    print(f"\n  Pre-fit figure saved: {out1.name}")
    plt.show(block=False); plt.pause(0.5)

    # ── Step 2: F01 v3 fit ────────────────────────────────────────────────────
    print("\n  Running F01 v3 fit (5-stage LM, A→E, 10 fitted + Y_A fixed)...\n")
    sigma_safe = np.maximum(sigma_prof, 1.0)
    good = np.isfinite(profile) & np.isfinite(sigma_safe) & (sigma_safe > 0)
    cal = _run_fit_v3(r_grid[good], profile[good], sigma_safe[good],
                      d_m, r_max_val, alpha_init)

    # ── Figure 2: convergence + two-line diagnostics ──────────────────────────
    fig2 = _fig2_convergence(r_grid, profile, sigma_prof, cal, p0,
                             d_m, r_max_val, source_label)
    out2 = _HERE / f"{stem}_f01_step4b_stages.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight", facecolor=_NAVY)
    print(f"\n  Convergence figure saved: {out2.name}")
    plt.show(block=False); plt.pause(0.5)

    # ── Print final summary ───────────────────────────────────────────────────
    print(f"\n{'═'*56}")
    print(f"  FINAL RESULT  (F01 v3 — two-line fit)")
    print(f"{'─'*56}")
    print(f"  Converged : {cal.converged}")
    print(f"  χ²_red    : {cal.chi2_reduced:.5f}")
    print(f"  R_refl    : {cal.R_refl:.5f}  ±  {cal.sigma_R_refl:.5f}")
    print(f"  α         : {cal.alpha:.5e}  ±  {cal.sigma_alpha:.2e}  rad/px")
    print(f"             (Δ = {(cal.alpha-alpha_init)/alpha_init*100:+.2f}% vs Tolansky)")
    print(f"  I₀        : {cal.I0:.1f}  ±  {cal.sigma_I0:.1f}  ADU")
    print(f"  σ₀        : {cal.sigma0:.4f}  ±  {cal.sigma_sigma0:.4f}  px")
    print(f"  Y_A       : 1.000000  (fixed)")
    print(f"  Y_B       : {cal.Y_B:.5f}  ±  {cal.sigma_Y_B:.5f}")
    print(f"  Y_B/Y_A   : {cal.intensity_ratio:.5f}")
    print(f"  B         : {cal.B:.2f}  ±  {cal.sigma_B:.3f}  ADU")
    print(f"  d (fixed) : {cal.t_m*1e3:.6f}  mm")
    print(f"  chi2_stages: {[f'{v:.3f}' for v in cal.chi2_stages]}")
    if cal.quality_flags:
        fired = [nm for bit, nm in [
            (CalibrationFitFlags.FIT_FAILED,    "FIT_FAILED"),
            (CalibrationFitFlags.CHI2_HIGH,     "CHI2_HIGH"),
            (CalibrationFitFlags.CHI2_VERY_HIGH,"CHI2_VERY_HIGH"),
            (CalibrationFitFlags.CHI2_LOW,      "CHI2_LOW"),
            (CalibrationFitFlags.STDERR_NONE,   "STDERR_NONE"),
            (CalibrationFitFlags.R_AT_BOUND,    "R_AT_BOUND"),
            (CalibrationFitFlags.ALPHA_AT_BOUND,"ALPHA_AT_BOUND"),
            (CalibrationFitFlags.FEW_BINS,      "FEW_BINS"),
            (CalibrationFitFlags.YB_RATIO_LOW,  "YB_RATIO_LOW"),
            (CalibrationFitFlags.YB_RATIO_HIGH, "YB_RATIO_HIGH"),
        ] if cal.quality_flags & bit]
        print(f"  Flags set : {', '.join(fired)}")
    else:
        print(f"  Flags     : GOOD (none set)")
    print(f"{'═'*56}")

    # ── Save result CSV ───────────────────────────────────────────────────────
    out_csv = _HERE / f"{stem}_f01_result.csv"
    with open(out_csv, "w") as fh:
        fh.write("# F01 v3 Step 4b CalibrationResult\n"
                 "# parameter,value,sigma_1,note\n")
        for nm, v, s, note in [
            ("d_mm",           cal.t_m*1e3,          0.0,               "fixed"),
            ("R_refl",         cal.R_refl,            cal.sigma_R_refl,  "fitted"),
            ("alpha_radpx",    cal.alpha,             cal.sigma_alpha,   "fitted"),
            ("I0_adu",         cal.I0,                cal.sigma_I0,      "fitted"),
            ("I1",             cal.I1,                cal.sigma_I1,      "fitted"),
            ("I2",             cal.I2,                cal.sigma_I2,      "fitted"),
            ("sigma0_px",      cal.sigma0,            cal.sigma_sigma0,  "fitted"),
            ("sigma1_px",      cal.sigma1,            cal.sigma_sigma1,  "fitted"),
            ("sigma2_px",      cal.sigma2,            cal.sigma_sigma2,  "fitted"),
            ("Y_A",            cal.Y_A,               0.0,               "fixed"),
            ("Y_B",            cal.Y_B,               cal.sigma_Y_B,     "fitted"),
            ("intensity_ratio",cal.intensity_ratio,   float("nan"),      "Y_B/Y_A"),
            ("B_adu",          cal.B,                 cal.sigma_B,       "fitted"),
            ("chi2_red",       cal.chi2_reduced,      float("nan"),      "diagnostic"),
            ("quality_flags",  float(cal.quality_flags), float("nan"),   "bitmask"),
        ]:
            fh.write(f"{nm},{v:.8g},{s:.8g},{note}\n")
    print(f"  Result CSV saved: {out_csv.name}")

    # ── Figure 3: final result ────────────────────────────────────────────────
    fig3 = _fig3_final(r_grid, profile, sigma_prof, cal,
                       d_m, alpha_init, r_max_val, source_label)
    out3 = _HERE / f"{stem}_f01_step4b_result.png"
    fig3.savefig(out3, dpi=150, bbox_inches="tight", facecolor=_NAVY)
    print(f"  Final result figure saved: {out3.name}")
    plt.show()
    print("\nDone.\n")


if __name__ == "__main__":
    main()
