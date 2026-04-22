"""
validate_f01_step4b_airy_fit.py
─────────────────────────────────────────────────────────────────────────────
Standalone, diagnostically-rich implementation of F01 Step 4b:
  Staged Levenberg–Marquardt Airy fit to a pre-reduced 1D neon fringe profile.

This version is designed for interactive investigation of real calibration data.
It shows every intermediate state of the fit so you can see exactly where
convergence is helping and where it is struggling.

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
    [1,0]  Initial model vs data  (BEFORE any fitting)
    [1,1]  Finesse/reflectivity estimate from fringe contrast scan

  Figure 2:  Per-stage convergence
    [0,0..1,1]  One panel per stage (A,B,C,D) showing model vs data overlay
                and Δχ²_red annotation

  Figure 3:  Final result
    [0,0]  Final model vs data with residuals in σ
    [0,1]  Stage χ²_red bar chart
    [1,0]  Residual distribution histogram vs N(0,1)
    [1,1]  CalibrationResult parameter table

Place this file in:  soc_sewell/validation/
─────────────────────────────────────────────────────────────────────────────
Author:  Claude AI / Scott Sewell  (NCAR/HAO)
Date:    2026-04-22
Spec:    docs/specs/F01_full_airy_fit_to_neon_image_2026-04-21.md  v2
"""

from __future__ import annotations
import sys, pathlib
from types import SimpleNamespace
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.optimize import least_squares
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
    from src.fpi.f01_full_airy_fit_to_neon_image_2026_04_21 import (
        fit_neon_fringe, TolanskyResult, CalibrationResult,
    )
    _F01_AVAILABLE = True
except ImportError:
    _F01_AVAILABLE = False

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
try:
    NE_WAVELENGTH_2_M  # may not be in older constants.py
except NameError:
    NE_WAVELENGTH_2_M = 638.2991e-9

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
# Inline Airy  (zero-dependency fallback)
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

def _eval_model(r_eval, r_fine, lam, d, R, alpha, n, r_max,
                I0, I1, I2, s0, s1, s2, B, two_line=True):
    """Evaluate (optionally two-line) Airy model at r_eval by fine-grid interp."""
    if _M01_AVAILABLE:
        kw = dict(t=d, R_refl=R, alpha=alpha, n=n, r_max=r_max,
                  I0=I0, I1=I1, I2=I2, sigma0=s0, sigma1=s1, sigma2=s2)
        fine_a = airy_modified(r_fine, NE_WAVELENGTH_1_M, **kw)
        fine_b = airy_modified(r_fine, NE_WAVELENGTH_2_M, **kw) * 0.8 if two_line else 0.0
    else:
        fine_a = _airy(r_fine, NE_WAVELENGTH_1_M, d, R, alpha, n, r_max,
                       I0, I1, I2, s0, s1, s2)
        fine_b = (_airy(r_fine, NE_WAVELENGTH_2_M, d, R, alpha, n, r_max,
                        I0, I1, I2, s0, s1, s2) * 0.8) if two_line else 0.0
    model_fine = fine_a + fine_b + B
    return np.interp(r_eval, r_fine, model_fine)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA-DRIVEN INITIAL GUESS ESTIMATOR
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_initial_params(r_grid, profile, sigma_prof, d_m, alpha_init, r_max,
                             R_prior=0.53):
    """
    Derive physically-motivated initial guesses entirely from the 1D profile.
    Prints a detailed explanation of the reasoning for each parameter.
    Returns a dict of initial guesses and a dict of bounds.
    """
    print("\n" + "─"*64)
    print("  DATA-DRIVEN INITIAL PARAMETER ESTIMATION")
    print("─"*64)

    # ── B: bias = Airy minimum ────────────────────────────────────────────────
    # The true minimum of the Airy function is  I₀/(1+F) + B.
    # On a real image the profile minimum closely approximates B
    # because the fringes dip toward B between peaks.
    # Use the 2nd percentile to avoid isolated noisy low bins.
    B_est = float(np.percentile(profile, 2))
    print(f"\n  B  (CCD bias)")
    print(f"     Method:  2nd percentile of profile  (Airy minimum ≈ B)")
    print(f"     Value:   {B_est:.1f} ADU")

    # ── Fringe contrast → R estimate ─────────────────────────────────────────
    # Contrast C = (I_max - I_min) / (I_max + I_min - 2B)
    #            = F / (2 + F)    for ideal Airy, where F = 4R/(1-R)²
    # Solving:  F = 2C / (1-C),  R = (√(1+F) - 1) / √(1+F)  (one root)
    # Use a rolling window to estimate local maxima and minima and average.
    win = max(5, int(len(profile)/30))
    from scipy.signal import find_peaks as fp_
    peaks_idx,  _ = fp_(profile,  distance=win//2,
                        height=np.percentile(profile, 70))
    troughs_idx, _ = fp_(-profile, distance=win//2,
                         height=-np.percentile(profile, 30))

    if len(peaks_idx) >= 2 and len(troughs_idx) >= 2:
        I_max_arr = profile[peaks_idx]
        I_min_arr = profile[troughs_idx]
        I_max = float(np.median(I_max_arr))
        I_min = float(np.median(I_min_arr))
        denom = I_max + I_min - 2*B_est
        C = (I_max - I_min) / denom if abs(denom) > 1 else 0.5
        C = np.clip(C, 0.05, 0.999)
        F_est  = 2*C / (1 - C)
        # R from F = 4R/(1-R)²: R² + (2 - F/1)·R + 1 = 0  → quadratic in (1-R)
        # simpler: R = (√(F/4 + 1) - 1) / (√(F/4 + 1) + 1)  ... or just:
        # 4R = F(1-R)²  →  FR² - (F+2)R·2 + F = 0 ... use numeric solve
        from numpy.polynomial import polynomial as P_
        # F·R² - (2+F)·R + F = ... wait: 4R = F(1-2R+R²) → FR² -(F+2)R +(F-... no
        # Correct: F = 4R/(1-R)²  is already solved for R by quadratic in (1-R)
        # Let x = 1-R:  F = 4(1-x)/x²  → Fx² + 4x - 4 = 0
        # x = (-4 + √(16+16F))/(2F) = (-4 + 4√(1+F))/(2F) = 2(√(1+F)-1)/F
        x = 2*(np.sqrt(1+F_est)-1)/F_est
        R_est = float(np.clip(1 - x, 0.1, 0.95))
    else:
        I_max = float(np.percentile(profile, 98))
        I_min = float(np.percentile(profile, 2))
        R_est = R_prior
        F_est  = 4*R_prior/(1-R_prior)**2
        C = F_est/(2+F_est)
        print(f"     [WARN] Too few peaks/troughs for contrast estimate; "
              f"using R_prior={R_prior}")

    print(f"\n  R  (plate reflectivity)")
    print(f"     Method:  fringe contrast C = (I_max−I_min)/(I_max+I_min−2B)")
    print(f"              I_max = {I_max:.1f} ADU,  I_min = {I_min:.1f} ADU,  B = {B_est:.1f} ADU")
    print(f"              C = {C:.4f}  →  F = {F_est:.3f}  →  R = {R_est:.4f}")
    print(f"     Value:   {R_est:.4f}  (R_prior = {R_prior})")

    # ── I₀: mean fringe peak intensity ───────────────────────────────────────
    # The Airy envelope peak equals I₀ + B.
    # Fit a quadratic to the upper envelope formed by the peak values.
    # I₀ = median of (peak_values - B) / peak_of_Airy_at_r=0
    # At r=0 the Airy is at maximum transmission = 1.0, so peak_signal = I₀ + B.
    # Use median of (peak_profile - B) after removing the outermost 10% of r
    # where vignetting starts to dominate.
    r_inner_mask = r_grid <= 0.9 * r_max
    if len(peaks_idx) >= 2:
        pk_r = r_grid[peaks_idx]
        pk_v = profile[peaks_idx]
        inner = pk_r <= 0.9*r_max
        if inner.sum() >= 1:
            I0_est = float(np.percentile(pk_v[inner] - B_est, 80))
        else:
            I0_est = float(np.percentile(profile - B_est, 90))
    else:
        I0_est = float(np.percentile(profile - B_est, 90))
    I0_est = max(I0_est, 10.0)

    print(f"\n  I₀  (mean fringe peak intensity)")
    print(f"     Method:  80th percentile of (peak_profile − B) for r < 0.9·r_max")
    print(f"              This approximates the Airy envelope at peak transmission.")
    print(f"     Value:   {I0_est:.1f} ADU")
    print(f"     NOTE:    I₁, I₂ describe the *shape* of the envelope falloff.")
    print(f"              Init: I₁=−0.1 (slight linear rolloff), I₂=0.005 (small quad).")

    # ── σ₀: PSF width estimate from fringe FWHM ──────────────────────────────
    # The ideal Airy FWHM (in r) at a given fringe order ≈ (1/π·F)·FSR_r
    # where FSR_r = √(λ/(2nd))/α  (ring spacing in pixels).
    # Measured FWHM > ideal → σ₀ = √(FWHM_meas² - FWHM_ideal²) / (2√(2ln2))
    FSR_m   = NE_WAVELENGTH_1_M**2 / (2*d_m)            # FSR in metres
    FSR_r_sq = FSR_m / (NE_WAVELENGTH_1_M) * (1/alpha_init**2)  # approx
    finesse_airy = np.pi*np.sqrt(R_est)/(1-R_est)
    FWHM_ideal_r = np.sqrt(FSR_r_sq) / finesse_airy if finesse_airy>0 else 5.0

    # Measure actual FWHM from a representative peak
    sigma0_est = 0.5   # default
    if len(peaks_idx) >= 3:
        # Use the median peak
        mid = peaks_idx[len(peaks_idx)//2]
        half_max = B_est + (profile[mid] - B_est) / 2.0
        # walk left and right from peak to find half-max crossing
        lo, hi = mid, mid
        while lo > 0 and profile[lo] > half_max: lo -= 1
        while hi < len(profile)-1 and profile[hi] > half_max: hi += 1
        FWHM_meas_bins = max(1, hi - lo)
        FWHM_meas_r = FWHM_meas_bins * (r_grid[-1]-r_grid[0]) / len(r_grid)
        excess_sq = max(0, FWHM_meas_r**2 - FWHM_ideal_r**2)
        sigma0_est = float(np.sqrt(excess_sq) / (2*np.sqrt(2*np.log(2))) + 0.2)
        sigma0_est = np.clip(sigma0_est, 0.1, 4.0)

    print(f"\n  σ₀  (PSF width)")
    print(f"     Method:  measured FWHM of a representative fringe − ideal Airy FWHM")
    print(f"              Finesse_Airy ≈ {finesse_airy:.2f}")
    print(f"              FWHM_ideal ≈ {FWHM_ideal_r:.2f} px-equivalent (profile bins)")
    print(f"     Value:   {sigma0_est:.3f} px")

    # ── α: already provided by Tolansky ──────────────────────────────────────
    # But we can cross-check: from r² spacing of consecutive peaks
    # r²(P+1) - r²(P) = FSR_r² = λ/(2nd·α²)  →  α = √(λ/(2nd·FSR_r²_meas))
    alpha_check = alpha_init
    if len(peaks_idx) >= 3:
        r2_peaks = r_grid[peaks_idx]**2
        dr2 = np.diff(r2_peaks)
        dr2_median = float(np.median(dr2[dr2 > 0])) if len(dr2) > 0 else None
        if dr2_median and dr2_median > 0:
            alpha_sq_check = NE_WAVELENGTH_1_M / (2*d_m * dr2_median)
            if alpha_sq_check > 0:
                alpha_check = float(np.sqrt(alpha_sq_check))

    print(f"\n  α  (plate scale)")
    print(f"     Source:  Tolansky two-line WLS slope  →  {alpha_init:.5e} rad/px")
    print(f"     Cross-check from r² peak spacing:     →  {alpha_check:.5e} rad/px")
    if abs(alpha_check - alpha_init)/alpha_init > 0.05:
        print(f"     [WARN] >5% discrepancy — Tolansky value used; "
              f"check centre-finding or peak assignment.")
    else:
        print(f"     Agreement within {abs(alpha_check-alpha_init)/alpha_init*100:.1f}%  ✓")

    # ── Assemble ─────────────────────────────────────────────────────────────
    p0 = dict(R=R_est, alpha=alpha_init, I0=I0_est, I1=-0.1, I2=0.005,
              sigma0=sigma0_est, sigma1=0.0, sigma2=0.0, B=B_est)

    # Parameter bounds  (LM with method='lm' doesn't support bounds natively,
    # so we clip inside the residual function; we report them here for display)
    bounds = dict(
        R       = (0.10,               0.95),
        alpha   = (0.5*alpha_init,     2.0*alpha_init),
        I0      = (1.0,                2**14 - 1),
        I1      = (-0.5,               0.5),
        I2      = (-0.5,               0.5),
        sigma0  = (0.01,               5.0),
        sigma1  = (-3.0,               3.0),
        sigma2  = (-3.0,               3.0),
        B       = (0.0,                float("inf")),
    )

    print(f"\n{'─'*64}")
    print(f"  {'Parameter':<12}  {'Initial':>12}  {'Lower':>12}  {'Upper':>12}  Source")
    print(f"  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*18}")
    src = {"R":"contrast scan", "alpha":"Tolansky WLS",
           "I0":"80th pctile peaks", "I1":"hardcoded",
           "I2":"hardcoded", "sigma0":"FWHM analysis",
           "sigma1":"hardcoded", "sigma2":"hardcoded", "B":"2nd pctile"}
    fmt = {"R":":.5f","alpha":":.4e","I0":":.1f","I1":":.4f","I2":":.4f",
           "sigma0":":.4f","sigma1":":.4f","sigma2":":.4f","B":":.1f"}
    for name in ["R","alpha","I0","I1","I2","sigma0","sigma1","sigma2","B"]:
        f = fmt[name]
        v  = p0[name]
        lo, hi = bounds[name]
        lo_s = f"{lo:.4g}" if np.isfinite(lo) else "−∞"
        hi_s = f"{hi:.4g}" if np.isfinite(hi) else "+∞"
        print(f"  {name:<12}  {format(v, f[1:]):>12}  {lo_s:>12}  {hi_s:>12}  {src[name]}")
    print(f"  {'d (FIXED)':<12}  {d_m*1e3:>12.6f}  {'—':>12}  {'—':>12}  Benoit/Tolansky")
    print(f"{'─'*64}")

    return p0, bounds


# ═══════════════════════════════════════════════════════════════════════════════
# Staged LM
# ═══════════════════════════════════════════════════════════════════════════════

def _run_staged_lm(r_good, profile_good, sigma_good, d, r_max, p0, alpha_init,
                   two_line=True):
    """
    Four-stage LM fit.  Returns list of (stage_label, params_dict, chi2_red).
    """
    r_fine = np.linspace(r_good[0], r_good[-1], 500)
    param_names = ["R","alpha","I0","I1","I2","sigma0","sigma1","sigma2","B"]
    bounds_lo = np.array([0.10, 0.5*alpha_init, 1.0, -0.5, -0.5, 0.01, -3., -3., 0.])
    bounds_hi = np.array([0.95, 2.0*alpha_init, 2**14-1, 0.5, 0.5, 5.0, 3., 3., np.inf])

    def _pack(p_dict): return np.array([p_dict[k] for k in param_names])
    def _unpack(arr):  return dict(zip(param_names, arr))

    def _residuals_for_stage(free_names, p_cur_dict):
        def _resid(x_free):
            p = dict(p_cur_dict)
            for k, v in zip(free_names, x_free):
                p[k] = v
            # Apply soft bounds via clipping (LM method='lm' ignores bounds param)
            p_arr = _pack(p)
            p_arr = np.clip(p_arr, bounds_lo, np.where(np.isfinite(bounds_hi), bounds_hi, 1e15))
            p = _unpack(p_arr)
            model = _eval_model(r_good, r_fine,
                                NE_WAVELENGTH_1_M, d, p["R"], p["alpha"], 1.0, r_max,
                                p["I0"], p["I1"], p["I2"],
                                p["sigma0"], p["sigma1"], p["sigma2"], p["B"],
                                two_line=two_line)
            return (profile_good - model) / sigma_good
        return _resid

    stages_def = [
        ("A  (I₀, B)",         ["I0","B"]),
        ("B  (I₀,I₁,I₂,B)",   ["I0","I1","I2","B"]),
        ("C  (R,α,I₀,B)",      ["R","alpha","I0","B"]),
        ("D  (all 9)",          param_names),
    ]

    history = []
    p_cur = dict(p0)

    for stage_label, free_names in stages_def:
        print(f"  Stage {stage_label} ...", end="", flush=True)
        x0 = [p_cur[k] for k in free_names]
        res = least_squares(_residuals_for_stage(free_names, p_cur),
                            x0, method='lm', max_nfev=5000)
        # Update current params
        p_arr = _pack(p_cur)
        for i, k in enumerate(free_names):
            idx = param_names.index(k)
            p_arr[idx] = res.x[i]
        p_arr = np.clip(p_arr, bounds_lo,
                        np.where(np.isfinite(bounds_hi), bounds_hi, 1e15))
        p_cur = _unpack(p_arr)

        n_dof = max(len(r_good) - len(free_names), 1)
        chi2  = float(np.sum(res.fun**2) / n_dof)
        history.append((stage_label, dict(p_cur), chi2, res))
        print(f"  χ²_red = {chi2:.4f}  (nfev={res.nfev})")

    # Covariance from final Jacobian
    J = history[-1][3].jac
    chi2_final = history[-1][2]
    try:
        JTJ  = J.T @ J
        cond = np.linalg.cond(JTJ)
        cov  = (np.linalg.pinv(JTJ, rcond=1e-10) if cond > 1e14
                else np.linalg.inv(JTJ)) * chi2_final
        stderrs = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except Exception:
        stderrs = np.full(9, np.nan)

    p_final = history[-1][1]
    result = SimpleNamespace(
        t_m=d,
        R_refl=p_final["R"],         sigma_R_refl=stderrs[0],
        alpha=p_final["alpha"],       sigma_alpha=stderrs[1],
        I0=p_final["I0"],             sigma_I0=stderrs[2],
        I1=p_final["I1"],             sigma_I1=stderrs[3],
        I2=p_final["I2"],             sigma_I2=stderrs[4],
        sigma0=p_final["sigma0"],     sigma_sigma0=stderrs[5],
        sigma1=p_final["sigma1"],     sigma_sigma1=stderrs[6],
        sigma2=p_final["sigma2"],     sigma_sigma2=stderrs[7],
        B=p_final["B"],               sigma_B=stderrs[8],
        chi2_reduced=chi2_final,
        n_bins_used=int(len(r_good)), n_params_free=9,
        converged=history[-1][3].success,
        quality_flags=0,
        stage_history=history,
        epsilon_cal=float((2.0*d/629.95e-9) % 1.0),
    )
    # Add two_sigma_ fields
    for attr in ["R_refl","alpha","I0","I1","I2","sigma0","sigma1","sigma2","B"]:
        setattr(result, f"two_sigma_{attr}", 2*getattr(result, f"sigma_{attr}"))
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: Pre-fit diagnostic
# ═══════════════════════════════════════════════════════════════════════════════

def _fig1_prefit(r_grid, profile, sigma_prof, p0, bounds,
                 d_m, alpha_init, r_max, source_label):
    _fig_style()
    fig = plt.figure(figsize=(18, 12), facecolor=_NAVY)
    fig.suptitle(
        f"F01 Step 4b — Pre-Fit Diagnostic  |  source: {source_label}",
        color="white", fontsize=12, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, left=0.07, right=0.97,
                           top=0.93, bottom=0.06, hspace=0.40, wspace=0.32)

    param_names = ["R","alpha","I0","I1","I2","sigma0","sigma1","sigma2","B"]
    r_fine = np.linspace(r_grid[0], r_grid[-1], 1000)

    # ── [0,0]  Annotated data profile ────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(r_grid, profile, color=_LGRAY, lw=1.0, alpha=0.8, label="Data")
    ax.fill_between(r_grid, profile-sigma_prof, profile+sigma_prof,
                    color=_LGRAY, alpha=0.12)
    # Annotate key data statistics used for guesses
    B_est = p0["B"]; I0_est = p0["I0"]
    ax.axhline(B_est, color=_AMBER, lw=1.2, ls="--",
               label=f"B est = {B_est:.0f} ADU  (2nd pctile)")
    ax.axhline(B_est + I0_est, color=_GREEN, lw=1.2, ls="--",
               label=f"B+I₀ = {B_est+I0_est:.0f} ADU  (80th pctile peaks)")
    ax.set_xlabel("Radius  r  (px)"); ax.set_ylabel("Counts  (ADU)")
    ax.set_title("Data Profile with Initial Estimate Annotations", color="white", pad=5)
    ax.grid(True)
    ax.legend(fontsize=8, facecolor=_DGRAY, edgecolor=_LGRAY, labelcolor=_LGRAY)

    # ── [0,1]  Parameter table with bounds ───────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    ax.axis("off")
    ax.set_title("Initial Guesses and Fit Bounds (all 9 free parameters)",
                 color="white", pad=5)
    fmt_map = {"R":":.5f","alpha":":.4e","I0":":.1f","I1":":.4f","I2":":.4f",
               "sigma0":":.4f","sigma1":":.4f","sigma2":":.4f","B":":.1f"}
    src_map = {"R":"fringe contrast","alpha":"Tolansky WLS",
               "I0":"80th pctile peaks","I1":"hardcoded (Harding)",
               "I2":"hardcoded (Harding)","sigma0":"FWHM analysis",
               "sigma1":"hardcoded","sigma2":"hardcoded","B":"2nd pctile"}
    headers = ["Param","Init. guess","Lower","Upper","Source"]
    col_x = [0.02, 0.18, 0.38, 0.55, 0.70]
    y0 = 0.96; dy = 0.073
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
        # Highlight if initial guess is near a bound
        pct = (v - lo)/(hi - lo) if (np.isfinite(lo) and np.isfinite(hi)) else 0.5
        col = _RED if (pct < 0.05 or pct > 0.95) else _LGRAY
        for ci, cell in enumerate([name,
                                    format(v, f[1:]),
                                    lo_s, hi_s, src_map[name]]):
            ax.text(col_x[ci], y, cell, transform=ax.transAxes,
                    ha="left", va="top", fontsize=8, color=col,
                    fontfamily="monospace")
    # d row
    y = y0 - (len(param_names)+1)*dy
    ax.text(col_x[0], y, "d  (FIXED)", transform=ax.transAxes,
            ha="left", va="top", fontsize=8, color=_AMBER, fontfamily="monospace")
    ax.text(col_x[1], y, f"{d_m*1e3:.6f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=8, color=_AMBER, fontfamily="monospace")
    ax.text(col_x[2], y, "—", transform=ax.transAxes,
            ha="left", va="top", fontsize=8, color=_AMBER, fontfamily="monospace")
    ax.text(col_x[3], y, "—", transform=ax.transAxes,
            ha="left", va="top", fontsize=8, color=_AMBER, fontfamily="monospace")
    ax.text(col_x[4], y, "Benoit/Tolansky", transform=ax.transAxes,
            ha="left", va="top", fontsize=8, color=_AMBER, fontfamily="monospace")

    # ── [1,0]  Initial model vs data ─────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    model_init_twolines = _eval_model(
        r_grid, r_fine, NE_WAVELENGTH_1_M, d_m,
        p0["R"], p0["alpha"], 1.0, r_max,
        p0["I0"], p0["I1"], p0["I2"],
        p0["sigma0"], p0["sigma1"], p0["sigma2"], p0["B"],
        two_line=True)
    model_init_oneline = _eval_model(
        r_grid, r_fine, NE_WAVELENGTH_1_M, d_m,
        p0["R"], p0["alpha"], 1.0, r_max,
        p0["I0"], p0["I1"], p0["I2"],
        p0["sigma0"], p0["sigma1"], p0["sigma2"], p0["B"],
        two_line=False)

    ax.plot(r_grid, profile, color=_LGRAY, lw=0.9, alpha=0.7, label="Data")
    ax.plot(r_grid, model_init_twolines, color=_AMBER, lw=1.8,
            label="Initial model (640+638 nm, 0.8× intensity ratio)")
    ax.plot(r_grid, model_init_oneline,  color=_TEAL,  lw=1.2, ls="--",
            label="Initial model (640 nm only)")
    ax.set_xlabel("Radius  r  (px)"); ax.set_ylabel("Counts  (ADU)")
    ax.set_title("Initial Model vs Data  (BEFORE fitting — d and α fixed)",
                 color="white", pad=5)
    ax.grid(True)
    ax.legend(fontsize=8, facecolor=_DGRAY, edgecolor=_LGRAY, labelcolor=_LGRAY)
    # χ² of initial guess
    sigma_safe = np.maximum(sigma_prof, 1.0)
    chi2_init = float(np.mean(((profile - model_init_twolines)/sigma_safe)**2))
    ax.text(0.97, 0.97, f"χ²_red (init, 2-line) = {chi2_init:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            color=_AMBER,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=_DGRAY,
                      edgecolor=_AMBER, alpha=0.9))

    # ── [1,1]  Fringe contrast profile (R estimate scan) ─────────────────────
    ax = fig.add_subplot(gs[1, 1])
    # Rolling contrast estimate across r
    win = max(8, int(len(profile)/20))
    contrast_r, contrast_c = [], []
    for i in range(win, len(profile)-win, win//2):
        chunk = profile[i-win:i+win]
        lo_v  = np.percentile(chunk, 5)
        hi_v  = np.percentile(chunk, 95)
        B_loc = p0["B"]
        denom = hi_v + lo_v - 2*B_loc
        if abs(denom) > 5:
            C_loc  = (hi_v - lo_v) / denom
            F_loc  = 2*np.clip(C_loc,0.05,0.99) / (1-np.clip(C_loc,0.05,0.99))
            x_loc  = 2*(np.sqrt(1+F_loc)-1)/F_loc
            R_loc  = np.clip(1-x_loc, 0.05, 0.98)
            contrast_r.append(r_grid[i])
            contrast_c.append(R_loc)

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
        ax.text(0.97, 0.05,
                "R should be roughly constant with r\n"
                "if the annular reduction is clean.",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7.5, color=_LGRAY, style="italic")
    else:
        ax.text(0.5, 0.5, "Insufficient contrast data", transform=ax.transAxes,
                ha="center", va="center", color=_AMBER)

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: Per-stage overlays
# ═══════════════════════════════════════════════════════════════════════════════

def _fig2_stages(r_grid, profile, sigma_prof, stage_history,
                 d_m, r_max, p0_chi2, source_label):
    _fig_style()
    fig = plt.figure(figsize=(18, 12), facecolor=_NAVY)
    fig.suptitle(
        f"F01 Step 4b — Per-Stage LM Convergence  |  source: {source_label}",
        color="white", fontsize=12, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, left=0.07, right=0.97,
                           top=0.93, bottom=0.06, hspace=0.40, wspace=0.32)
    axes = [fig.add_subplot(gs[i//2, i%2]) for i in range(4)]

    r_fine = np.linspace(r_grid[0], r_grid[-1], 1000)
    sigma_safe = np.maximum(sigma_prof, 1.0)
    stage_colors = [_TEAL, _BLUE, _AMBER, _GREEN]
    prev_chi2 = p0_chi2

    for i, (stage_label, p, chi2, _) in enumerate(stage_history):
        ax = axes[i]
        ax.plot(r_grid, profile, color=_LGRAY, lw=0.9, alpha=0.65, label="Data")
        model = _eval_model(r_grid, r_fine, NE_WAVELENGTH_1_M, d_m,
                            p["R"], p["alpha"], 1.0, r_max,
                            p["I0"], p["I1"], p["I2"],
                            p["sigma0"], p["sigma1"], p["sigma2"], p["B"],
                            two_line=True)
        ax.plot(r_grid, model, color=stage_colors[i], lw=1.8,
                label=f"Model after Stage {stage_label.split()[0]}")
        # Shaded residual
        resid = (profile - model)
        ax.fill_between(r_grid, model, model+resid, color=_RED, alpha=0.25,
                        label="Residual ×1")
        delta = chi2 - prev_chi2
        ax.set_title(
            f"Stage {stage_label}  |  χ²_red = {chi2:.4f}  "
            f"(Δ = {delta:+.4f} from previous)",
            color="white", pad=5)
        ax.set_xlabel("Radius  r  (px)"); ax.set_ylabel("Counts  (ADU)")
        ax.grid(True)
        ax.legend(fontsize=8, facecolor=_DGRAY, edgecolor=_LGRAY, labelcolor=_LGRAY)

        # Parameter change table inset
        changed_params = stage_label.split("(")[-1].rstrip(")").split(",") \
                         if "(" in stage_label else []
        prev_chi2 = chi2

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3: Final result
# ═══════════════════════════════════════════════════════════════════════════════

def _fig3_final(r_grid, profile, sigma_prof, cal, d_m, alpha_init, r_max,
                source_label):
    _fig_style()
    fig = plt.figure(figsize=(18, 12), facecolor=_NAVY)
    fig.suptitle(
        f"F01 Step 4b — Final Result  |  source: {source_label}",
        color="white", fontsize=12, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, left=0.07, right=0.97,
                           top=0.93, bottom=0.06, hspace=0.40, wspace=0.32)

    r_fine = np.linspace(r_grid[0], r_grid[-1], 1000)
    sigma_safe = np.maximum(sigma_prof, 1.0)

    model_final = _eval_model(
        r_grid, r_fine, NE_WAVELENGTH_1_M, d_m,
        cal.R_refl, cal.alpha, 1.0, r_max,
        cal.I0, cal.I1, cal.I2,
        cal.sigma0, cal.sigma1, cal.sigma2, cal.B,
        two_line=True)

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

    # ── [0,1]  Stage χ²_red bar chart ────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    labels = ["Init"] + [h[0].split()[0] for h in cal.stage_history]
    chi2s  = [cal.stage_history[0][2]*3] + [h[2] for h in cal.stage_history]
    # approximate "init" χ² from stage A before the first improvement
    # (we stored it separately — fall back to 3× final if not available)
    colors = [_LGRAY, _TEAL, _BLUE, _AMBER, _GREEN]
    bars = ax.bar(labels, chi2s, color=colors,
                  edgecolor=_LGRAY, linewidth=0.7)
    for bar, v in zip(bars[1:], chi2s[1:]):
        ax.text(bar.get_x()+bar.get_width()/2, v + 0.01*max(chi2s),
                f"{v:.4f}", ha="center", va="bottom",
                fontsize=8, color="white", fontweight="bold")
    ax.axhline(1.0, color=_GREEN, lw=1.2, ls="--", label="χ²_red=1 (ideal)")
    ax.axhline(3.0, color=_RED,   lw=1.0, ls=":",  label="χ²_red=3 (caution)")
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
    ax.text(0.97, 0.97, f"RMS residual = {rms_r:.3f} σ\n"
                        f"ideal = 1.000 σ",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            color=_AMBER,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=_DGRAY,
                      edgecolor=_AMBER, alpha=0.9))
    ax.set_xlabel("Residual (σ)"); ax.set_ylabel("Density")
    ax.set_title("Residual Distribution vs Ideal N(0,1)", color="white", pad=5)
    ax.legend(fontsize=8, facecolor=_DGRAY, edgecolor=_LGRAY, labelcolor=_LGRAY)
    ax.grid(True)

    # ── [1,1]  Parameter table ────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    ax.set_title("CalibrationResult — All 9 Fitted Parameters + d",
                 color="white", pad=5)
    rows = [
        ("Param",    "Fitted",                      "1σ",                      "Unit"),
        ("─"*10,     "─"*13,                        "─"*11,                    "─"*5),
        ("d FIXED",  f"{cal.t_m*1e3:.6f}",           "(Tolansky)",              "mm"),
        ("R_refl",   f"{cal.R_refl:.5f}",             f"±{cal.sigma_R_refl:.5f}", "—"),
        ("α",        f"{cal.alpha:.4e}",               f"±{cal.sigma_alpha:.2e}", "rad/px"),
        ("I₀",       f"{cal.I0:.1f}",                  f"±{cal.sigma_I0:.1f}",   "ADU"),
        ("I₁",       f"{cal.I1:.5f}",                  f"±{cal.sigma_I1:.5f}",   "—"),
        ("I₂",       f"{cal.I2:.5f}",                  f"±{cal.sigma_I2:.5f}",   "—"),
        ("σ₀",       f"{cal.sigma0:.5f}",               f"±{cal.sigma_sigma0:.5f}","px"),
        ("σ₁",       f"{cal.sigma1:.5f}",               f"±{cal.sigma_sigma1:.5f}","px"),
        ("σ₂",       f"{cal.sigma2:.5f}",               f"±{cal.sigma_sigma2:.5f}","px"),
        ("B",        f"{cal.B:.2f}",                   f"±{cal.sigma_B:.3f}",    "ADU"),
        ("─"*10,     "─"*13,                        "─"*11,                    "─"*5),
        ("χ²_red",   f"{cal.chi2_reduced:.5f}",       "",                       ""),
        ("α Δ%",     f"{(cal.alpha-alpha_init)/alpha_init*100:+.2f}%",
                     "(vs Tolansky init)",             ""),
    ]
    cx_ = [0.02, 0.30, 0.60, 0.87]
    y0_ = 0.97; dy_ = 0.057
    for ri, row in enumerate(rows):
        y = y0_ - ri*dy_
        is_hdr  = (ri == 0)
        is_rule = row[0].startswith("─")
        is_fixed = row[0].startswith("d ")
        col = ("white" if is_hdr else _AMBER if is_rule else
               "#D09000" if is_fixed else _LGRAY)
        fw  = "bold" if is_hdr else "normal"
        for ci, cell in enumerate(row):
            ax.text(cx_[ci], y, cell, transform=ax.transAxes,
                    ha="left", va="top", fontsize=8.2,
                    color=col, fontweight=fw, fontfamily="monospace")
    flag_names = [n for bit, n in
                  [(0x001,"FIT_FAILED"),(0x002,"CHI2_HIGH"),(0x004,"CHI2_VERY_HIGH"),
                   (0x008,"CHI2_LOW"),(0x010,"STDERR_NONE"),(0x020,"R_AT_BOUND"),
                   (0x040,"ALPHA_AT_BOUND"),(0x080,"FEW_BINS")]
                  if cal.quality_flags & bit]
    ax.text(0.02, y0_-len(rows)*dy_-0.02,
            f"Quality: {'GOOD' if not flag_names else ', '.join(flag_names)}",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            fontweight="bold",
            color=_GREEN if not flag_names else _RED)
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

    Uses tkinter (standard library on all Windows Python installs).
    If tkinter is unavailable (headless server), falls back to a console prompt.

    Returns the chosen Path, or None if the user cancelled / pressed Enter
    (which triggers synthetic data generation).
    """
    # ── try tkinter first ────────────────────────────────────────────────────
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox

        # Hide the root Tk window — we only want the dialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)   # bring dialog to front on Windows

        print("\n  Opening file dialog — select your 3-column profile CSV.")
        print("  (Cancel the dialog to use SYNTHETIC data instead.)\n")

        path_str = filedialog.askopenfilename(
            title       = "Select annular-profile CSV  "
                          "(r_grid_px, profile_adu, sigma_profile_adu)",
            filetypes   = [
                ("Annular profile CSV",    "*_annular_profile.csv"),
                ("All files",              "*.*"),
            ],
            initialdir  = str(_HERE),       # start in validation/ folder
        )
        root.destroy()

        if not path_str:
            # User pressed Cancel → use synthetic
            print("  Dialog cancelled — using SYNTHETIC two-line neon profile.")
            return None

        p = pathlib.Path(path_str)
        if not p.exists():
            print(f"  [ERROR] Selected file not found: {p}")
            sys.exit(1)

        print(f"  Selected: {p}")
        return p

    except Exception as _tk_err:
        # ── tkinter unavailable — fall back to console prompt ────────────────
        print(f"  [INFO] tkinter dialog unavailable ({_tk_err}).")
        print(f"  Falling back to console prompt.")
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
    print("  F01 Step 4b — Staged Airy Fit  (Diagnostic Edition)")
    print("  WindCube FPI Pipeline  ·  NCAR/HAO")
    print("═"*64)

    # ── File dialog ───────────────────────────────────────────────────────────
    csv_path = _prompt_csv_dialog()

    # ── Quick metadata peek (α default comes from CSV header if present) ────────
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

    # ── Numerical parameters ──────────────────────────────────────────────────
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
        from types import SimpleNamespace as _SN
        rng  = np.random.default_rng(42)
        r1d  = np.linspace(0.5, r_max_val, 500).astype(np.float32)
        kw   = dict(t=d_m, R_refl=0.53, alpha=alpha_init, n=1.0,
                    r_max=r_max_val, I0=1000.0, I1=-0.1, I2=0.005,
                    sigma0=0.5, sigma1=0.0, sigma2=0.0)
        if _M01_AVAILABLE:
            sig = (airy_modified(r1d, NE_WAVELENGTH_1_M, **kw) +
                   airy_modified(r1d, NE_WAVELENGTH_2_M, **kw)*0.8 + 300.0)
        else:
            sig = (_airy(r1d,NE_WAVELENGTH_1_M,d_m,0.53,alpha_init,1.0,r_max_val,
                         1000.,-0.1,0.005,0.5,0.,0.) +
                   _airy(r1d,NE_WAVELENGTH_2_M,d_m,0.53,alpha_init,1.0,r_max_val,
                         1000.,-0.1,0.005,0.5,0.,0.)*0.8 + 300.0)
        noisy = rng.poisson(np.maximum(sig,1)).astype(np.float32)
        sigma = np.maximum(np.sqrt(noisy)/8, 1.0).astype(np.float32)
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

    # ── Step 2: staged LM ─────────────────────────────────────────────────────
    print("\n  Running staged LM fit (A→B→C→D)...\n")
    sigma_safe = np.maximum(sigma_prof, 1.0)
    good = np.isfinite(profile) & np.isfinite(sigma_safe) & (sigma_safe > 0)
    cal = _run_staged_lm(r_grid[good], profile[good], sigma_safe[good],
                         d_m, r_max_val, p0, alpha_init, two_line=True)

    # approximate init chi2 for the bar chart
    r_fine0 = np.linspace(r_grid[0], r_grid[-1], 500)
    m_init = _eval_model(r_grid[good], r_fine0, NE_WAVELENGTH_1_M, d_m,
                         p0["R"],p0["alpha"],1.0,r_max_val,
                         p0["I0"],p0["I1"],p0["I2"],
                         p0["sigma0"],p0["sigma1"],p0["sigma2"],p0["B"],
                         two_line=True)
    p0_chi2 = float(np.mean(((profile[good]-m_init)/sigma_safe[good])**2))

    # ── Figure 2: per-stage ───────────────────────────────────────────────────
    fig2 = _fig2_stages(r_grid, profile, sigma_prof, cal.stage_history,
                        d_m, r_max_val, p0_chi2, source_label)
    out2 = _HERE / f"{stem}_f01_step4b_stages.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight", facecolor=_NAVY)
    print(f"\n  Stage convergence figure saved: {out2.name}")
    plt.show(block=False); plt.pause(0.5)

    # ── Print final summary ───────────────────────────────────────────────────
    print(f"\n{'═'*50}")
    print(f"  FINAL RESULT")
    print(f"{'─'*50}")
    print(f"  Converged : {cal.converged}")
    print(f"  χ²_red    : {cal.chi2_reduced:.5f}")
    print(f"  R_refl    : {cal.R_refl:.5f}  ±  {cal.sigma_R_refl:.5f}")
    print(f"  α         : {cal.alpha:.5e}  ±  {cal.sigma_alpha:.2e}  rad/px")
    print(f"             (Δ = {(cal.alpha-alpha_init)/alpha_init*100:+.2f}% vs Tolansky)")
    print(f"  I₀        : {cal.I0:.1f}  ±  {cal.sigma_I0:.1f}  ADU")
    print(f"  σ₀        : {cal.sigma0:.4f}  ±  {cal.sigma_sigma0:.4f}  px")
    print(f"  B         : {cal.B:.2f}  ±  {cal.sigma_B:.3f}  ADU")
    print(f"  d (fixed) : {cal.t_m*1e3:.6f}  mm")
    print(f"{'═'*50}")

    # ── Save result CSV ───────────────────────────────────────────────────────
    out_csv = _HERE / f"{stem}_f01_result.csv"
    with open(out_csv, "w") as fh:
        fh.write("# F01 Step 4b CalibrationResult\n# parameter,value,sigma_1,note\n")
        for nm, v, s, note in [
            ("d_mm",       cal.t_m*1e3,   0.0,              "fixed"),
            ("R_refl",     cal.R_refl,    cal.sigma_R_refl, "fitted"),
            ("alpha_radpx",cal.alpha,     cal.sigma_alpha,  "fitted"),
            ("I0_adu",     cal.I0,        cal.sigma_I0,     "fitted"),
            ("I1",         cal.I1,        cal.sigma_I1,     "fitted"),
            ("I2",         cal.I2,        cal.sigma_I2,     "fitted"),
            ("sigma0_px",  cal.sigma0,    cal.sigma_sigma0, "fitted"),
            ("sigma1_px",  cal.sigma1,    cal.sigma_sigma1, "fitted"),
            ("sigma2_px",  cal.sigma2,    cal.sigma_sigma2, "fitted"),
            ("B_adu",      cal.B,         cal.sigma_B,      "fitted"),
            ("chi2_red",   cal.chi2_reduced, float("nan"),  "diagnostic"),
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
