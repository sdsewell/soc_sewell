"""
Script:  p05_cal_diagnostics_2026_05_06.py
Purpose: Four-panel diagnostic figure to understand WHY the two-line neon
         calibration fit is not capturing the λ₂ (638.3 nm) peaks correctly.

         Uses the ALREADY-FITTED parameters from the last p05 run — you
         just type them into the FITTED PARAMETERS block below (they are
         printed to the terminal when p05 runs).  No new fitting is done.

         Four panels:
           A — Data vs model, plotted vs r  (not r²) — equal spacing shows
               whether misfit is concentrated at inner or outer fringes
           B — Residual vs r, with the TWO NEON COMPONENTS overlaid
               separately so you can see which line family drives the misfit
           C — Residual vs r², same as p05 but zoomed to show structure
           D — Scatter: residual vs model value — should be flat if the
               model is correct; a slope means multiplicative (gain) error

         Run from repo root:
             python src/processing/p05_cal_diagnostics_2026_05_06.py

         INSTRUCTIONS:
           1. Run p05_calibration_inversion_2026_05_06.py first.
           2. Copy the printed fit parameters into the block below.
           3. Run this script and select the same .npy file.
"""

import pathlib
import sys
import tkinter as tk
from tkinter import filedialog

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
# Make repo root importable
# ---------------------------------------------------------------------------
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.fpi.airy_forward_model_2026_05_05 import (   # noqa: E402
    airy_modified,
    phase_correct_gap,
)
from windcube.constants import (                       # noqa: E402
    NE_WAVELENGTH_1_AIR_M,
    NE_WAVELENGTH_2_AIR_M,
    NE_INTENSITY_2 as NE_INTENSITY_2_NOMINAL,
)

# ===========================================================================
# *** PASTE YOUR FITTED PARAMETERS HERE ***
# Copy these from the terminal output of p05_calibration_inversion_2026_05_06.py
# ===========================================================================
FIT = dict(
    t_m      = 20.1069745e-3,   # metres  (etalon gap, phase-corrected)
    alpha    = 1.60821e-4,      # rad/px
    R        = 0.2147,          # effective reflectivity
    I0       = 5570.3,          # ADU
    I1       = 0.0806,
    I2       = -0.14173,
    sigma0   = 0.4416,          # px
    sigma1   = -0.0001,         # px
    sigma2   = 1.9925,          # px
    B        = 2599.1,          # ADU
    ne_ratio = 0.3608,          # λ₂/λ₁ fitted ratio
)

# Tolansky values (for phase_correct_gap)
T_TOLANSKY_MM  = 20.1070707    # mm
EPS_A          = 0.23286
ALPHA_TOLANSKY = 1.6084e-4     # rad/px
R_MAX_PX       = 110.0         # px
# ===========================================================================


def _load_profile(path, r_max_px):
    arr = np.load(path)
    sigma = None
    if arr.ndim == 1:
        r2, prof = np.linspace(0.0, r_max_px**2, len(arr)), arr.astype(float)
    elif arr.ndim == 2 and arr.shape[0] == 3:
        r2, prof, sigma = arr[0].astype(float), arr[1].astype(float), arr[2].astype(float)
    elif arr.ndim == 2 and arr.shape[0] == 2:
        r2, prof = arr[0].astype(float), arr[1].astype(float)
    elif arr.ndim == 2 and arr.shape[1] == 2:
        r2, prof = arr[:, 0].astype(float), arr[:, 1].astype(float)
    else:
        raise ValueError(f"Unexpected array shape {arr.shape}")
    r_grid = np.sqrt(np.maximum(r2, 0.0))
    return r_grid, prof, sigma


def _estimate_sigma(profile):
    n, half = len(profile), 2
    sigma = np.empty(n)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        sigma[i] = float(np.std(profile[lo:hi]))
    return np.maximum(sigma, max(1.0, float(np.median(profile)) * 0.005))


def _build_models(r_arr, r_max, fit, n_fine=2000):
    """
    Return (model_composite, model_lam1_only, model_lam2_only) on r_arr.
    Each includes the bias B so they are in ADU and directly comparable to data.
    """
    r_fine = np.linspace(0.0, r_max, n_fine)
    t, alpha, R = fit['t_m'], fit['alpha'], fit['R']
    I0, I1, I2  = fit['I0'], fit['I1'], fit['I2']
    s0, s1, s2  = fit['sigma0'], fit['sigma1'], fit['sigma2']
    B           = fit['B']
    ne_ratio    = fit['ne_ratio']

    A1_fine = airy_modified(r_fine, NE_WAVELENGTH_1_AIR_M,
                            t, R, alpha, 1.0, r_max, I0, I1, I2, s0, s1, s2)
    A2_fine = airy_modified(r_fine, NE_WAVELENGTH_2_AIR_M,
                            t, R, alpha, 1.0, r_max, I0, I1, I2, s0, s1, s2)

    comp_fine  = A1_fine + ne_ratio * A2_fine + B
    lam1_fine  = A1_fine + B
    lam2_fine  = ne_ratio * A2_fine + B

    comp  = np.interp(r_arr, r_fine, comp_fine)
    lam1  = np.interp(r_arr, r_fine, lam1_fine)
    lam2  = np.interp(r_arr, r_fine, lam2_fine)
    return comp, lam1, lam2


def main():

    # 1. Load file
    _tk = tk.Tk(); _tk.withdraw()
    npy_path = filedialog.askopenfilename(
        title="Select neon calibration profile (.npy, same file as p05)",
        filetypes=[("NumPy arrays", "*.npy"), ("All files", "*.*")])
    _tk.destroy()
    if not npy_path:
        print("No file selected — exiting."); return
    npy_path = pathlib.Path(npy_path)
    print(f"Loaded: {npy_path.name}")

    # 2. Load and clip profile
    r_grid, profile, sigma_loaded = _load_profile(npy_path, R_MAX_PX)
    in_range = r_grid <= R_MAX_PX
    r_grid, profile = r_grid[in_range], profile[in_range]
    if sigma_loaded is not None:
        sigma = sigma_loaded[in_range]
        bad = ~np.isfinite(sigma)
        if bad.any():
            sigma[bad] = _estimate_sigma(profile)[bad]
    else:
        sigma = _estimate_sigma(profile)

    r2 = r_grid ** 2

    # 3. Build models
    comp, lam1, lam2 = _build_models(r_grid, R_MAX_PX, FIT)
    residual = profile - comp

    # Normalised residual (units of σ)
    resid_norm = residual / np.maximum(sigma, 1.0)

    # Scatter arrays for panel D
    sort_idx   = np.argsort(comp)
    comp_sort  = comp[sort_idx]
    resid_sort = residual[sort_idx]
    # Smooth the scatter with a wide running median for the trend line
    win = max(1, len(comp_sort) // 40)
    trend = np.array([
        float(np.median(resid_sort[max(0,i-win):min(len(resid_sort),i+win+1)]))
        for i in range(len(resid_sort))
    ])

    # 4. Figure layout — 2×2
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        f"Neon Calibration Fit Diagnostics — {npy_path.name}\n"
        f"t = {FIT['t_m']*1e3:.7f} mm   R = {FIT['R']:.4f}   "
        f"σ₀ = {FIT['sigma0']:.3f} px   σ₂ = {FIT['sigma2']:.3f} px   "
        f"ne_ratio = {FIT['ne_ratio']:.4f}",
        fontsize=10, y=0.99)

    gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.30,
                           top=0.93, bottom=0.07, left=0.07, right=0.97)
    ax_r   = fig.add_subplot(gs[0, 0])   # A: data+model vs r
    ax_res = fig.add_subplot(gs[0, 1])   # B: residual vs r with components
    ax_r2  = fig.add_subplot(gs[1, 0])   # C: normalised residual vs r²
    ax_sct = fig.add_subplot(gs[1, 1])   # D: residual vs model (gain check)

    # ---- Panel A: data + model vs r (equal radial spacing) ---------------
    ax_r.plot(r_grid, profile, color="darkorange", lw=0.7, alpha=0.8,
              label="Data")
    ax_r.plot(r_grid, comp, color="black", lw=1.4, label="Model (composite)")
    ax_r.plot(r_grid, lam1, color="steelblue", lw=0.8, ls="--", alpha=0.7,
              label=f"λ₁ 640.2 nm only")
    ax_r.plot(r_grid, lam2, color="firebrick", lw=0.8, ls="--", alpha=0.7,
              label=f"λ₂ 638.3 nm (×{FIT['ne_ratio']:.3f})")
    ax_r.set_xlabel("Radius  r  (pixels)", fontsize=10)
    ax_r.set_ylabel("CCD signal  (ADU)", fontsize=10)
    ax_r.set_title("A — Data vs model  (vs r,  equal spacing)", fontsize=10)
    ax_r.legend(fontsize=7.5, loc="upper right")
    ax_r.grid(True, alpha=0.2)

    # ---- Panel B: residual vs r with component curves scaled down --------
    # Scale the two component curves to residual amplitude for overlay
    comp_range = comp.max() - comp.min()
    resid_range = np.percentile(np.abs(residual), 95) * 2
    scale = resid_range / comp_range if comp_range > 0 else 1.0

    ax_res.axhline(0, color="black", lw=0.8, ls="--")
    ax_res.fill_between(r_grid, -sigma, sigma, color="grey", alpha=0.2,
                        label="±1σ")
    ax_res.plot(r_grid, residual, color="steelblue", lw=0.7, alpha=0.9,
                label="Residual (data − model)")

    # Overlay the λ₁ and λ₂ Airy curves (mean-subtracted, rescaled) so
    # you can see if residual spikes align with one line family
    A1_overlay = (lam1 - lam1.mean()) * scale
    A2_overlay = (lam2 - lam2.mean()) * scale
    ax_res.plot(r_grid, A1_overlay, color="steelblue", lw=0.6, ls=":",
                alpha=0.6, label="λ₁ shape (rescaled)")
    ax_res.plot(r_grid, A2_overlay, color="firebrick", lw=0.6, ls=":",
                alpha=0.6, label="λ₂ shape (rescaled)")

    ax_res.set_xlabel("Radius  r  (pixels)", fontsize=10)
    ax_res.set_ylabel("Residual  (ADU)", fontsize=10)
    ax_res.set_title("B — Residual vs r with line-family overlays", fontsize=10)
    ax_res.legend(fontsize=7.5, loc="upper right")
    ax_res.grid(True, alpha=0.2)
    ax_res.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, symmetric=True))

    # ---- Panel C: normalised residual vs r² — same axis as p05 ----------
    ax_r2.axhline(0, color="black", lw=0.8, ls="--")
    ax_r2.axhline(+3, color="red", lw=0.5, ls=":", alpha=0.6, label="±3σ")
    ax_r2.axhline(-3, color="red", lw=0.5, ls=":", alpha=0.6)
    ax_r2.fill_between(r2, -1, 1, color="grey", alpha=0.15, label="±1σ")
    ax_r2.plot(r2, resid_norm, color="steelblue", lw=0.7, alpha=0.85,
               label="(data − model) / σ")
    # Running RMS to show if misfit grows with radius
    win2 = max(1, len(r2) // 30)
    rms_running = np.array([
        float(np.sqrt(np.mean(resid_norm[max(0,i-win2):min(len(resid_norm),i+win2+1)]**2)))
        for i in range(len(resid_norm))
    ])
    ax_r2.plot(r2, rms_running, color="darkorange", lw=1.2, alpha=0.9,
               label="Running RMS (σ units)")
    ax_r2.set_xlabel(r"$r^2$  (pixels²)", fontsize=10)
    ax_r2.set_ylabel("Normalised residual  (σ)", fontsize=10)
    ax_r2.set_title("C — Normalised residual vs r²  (should be ~flat at ≤3σ)", fontsize=10)
    ax_r2.legend(fontsize=7.5, loc="upper right")
    ax_r2.grid(True, alpha=0.2)
    ax_r2.set_ylim(-8, 8)

    # ---- Panel D: residual vs model value — gain/linearity check ---------
    ax_sct.scatter(comp_sort, resid_sort, s=1.2, color="steelblue",
                   alpha=0.3, label="Residual vs model")
    ax_sct.plot(comp_sort, trend, color="darkorange", lw=1.5,
                label="Running median trend")
    ax_sct.axhline(0, color="black", lw=0.8, ls="--")
    ax_sct.set_xlabel("Model value  (ADU)", fontsize=10)
    ax_sct.set_ylabel("Residual  (ADU)", fontsize=10)
    ax_sct.set_title(
        "D — Residual vs model  (flat = good;\n"
        "slope = gain error; curve = saturation/nonlinearity)", fontsize=9)
    ax_sct.legend(fontsize=7.5, loc="upper right")
    ax_sct.grid(True, alpha=0.2)

    # Annotate key fit quality numbers
    chi2 = float(np.mean(resid_norm**2))
    rms_adu = float(np.std(residual))
    fig.text(0.5, 0.005,
             f"χ²/ν ≈ {chi2:.2f}   RMS residual = {rms_adu:.1f} ADU   "
             f"σ₂ = {FIT['sigma2']:.4f} px  (bound = 2.0; hitting limit = "
             f"{'YES ← investigate' if FIT['sigma2'] > 1.8 else 'no'})",
             ha="center", va="bottom", fontsize=8.5,
             fontfamily="monospace",
             color="darkred" if FIT['sigma2'] > 1.8 else "dimgrey")

    plt.show()


if __name__ == "__main__":
    main()
