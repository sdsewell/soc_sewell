"""
Script:  p05_calibration_inversion_2026_05_06.py
Purpose: Load a real two-line neon calibration radial profile (tabulated vs r²),
         run an 11-parameter staged Levenberg-Marquardt calibration inversion,
         and produce a diagnostic figure modelled on Harding et al. (2014) Fig. 4:
           • Top panel:  data vs. best-fit modified Airy model (vs r²)
           • Middle panel: residual (data − model), with ±1σ band
           • Bottom panel: fitted parameter table

         Parameters fitted (11 total):
           Group A — Tolansky-seeded, tight bounds:
             t_eff   — phase-corrected etalon gap (H01 §8)
             alpha   — plate scale (rad/px)
           Group B — freely fitted from fringe shape:
             R       — effective reflectivity
             I0, I1, I2  — intensity envelope
             sigma0, sigma1, sigma2  — PSF width
             B       — CCD bias pedestal
             ne_ratio — λ₂/λ₁ intensity ratio  ← 11th parameter, NEW
                        NE_INTENSITY_2 = 0.36 is only a starting point;
                        the real lamp ratio is recovered from the fit.

         NOTE: This script does NOT call M05's fit_calibration_fringe().
         It implements its own 4-stage LM fit so that ne_ratio can be a
         free parameter.  M05's _neon_model() hardcodes NE_INTENSITY_2 = 0.36
         and cannot fit the ratio.  The staged inversion strategy and bounds
         architecture follow H05 exactly.

Input .npy file formats accepted:
   (N,)      — profile values only; r² inferred (not recommended)
   (2, N)    — row 0 = r² (px²), row 1 = profile (ADU)
   (3, N)    — row 0 = r² (px²), row 1 = profile (ADU), row 2 = SEM (ADU)
   (N, 2)    — col 0 = r² (px²), col 1 = profile (ADU)

Run from repo root:
    python src/processing/p05_calibration_inversion_2026_05_06.py
"""

import pathlib
import sys
import tkinter as tk
from tkinter import filedialog, simpledialog
import logging
from dataclasses import dataclass
from typing import Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.optimize import least_squares

# ---------------------------------------------------------------------------
# Make repo root importable regardless of working directory
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

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s  %(name)s  %(message)s")
log = logging.getLogger("p05")

# ---------------------------------------------------------------------------
# Parameter ordering (11 parameters)
# ---------------------------------------------------------------------------
# Index:  0      1        2    3    4    5    6        7        8        9    10
_NAMES = ['t_m', 'alpha', 'R', 'I0', 'I1', 'I2', 'sigma0', 'sigma1', 'sigma2', 'B', 'ne_ratio']
_IDX   = {n: i for i, n in enumerate(_NAMES)}

# Stage free-parameter sets — mirrors H05 staged strategy plus ne_ratio
_STAGE_FREE = {
    1: ['I0', 'I1', 'I2', 'B'],                                    # photometric baseline
    2: ['t_m', 'alpha', 'R', 'I0', 'I1', 'I2', 'B'],              # geometry + reflectivity
    3: ['t_m', 'alpha', 'R', 'I0', 'I1', 'I2', 'sigma0', 'B'],   # + PSF base width
    4: _NAMES,                                                       # all 11, including ne_ratio
}

# ---------------------------------------------------------------------------
# Forward model — two-line neon with free ne_ratio
# ---------------------------------------------------------------------------

def _neon_model_11(r_arr, r_max, t, alpha, R, I0, I1, I2,
                   sigma0, sigma1, sigma2, B, ne_ratio,
                   _N_fine=500):
    """
    Two-line neon forward model with ne_ratio as a free parameter.

    Evaluates on a fine uniform grid (_N_fine points), then interpolates
    to r_arr to match the sampling behaviour of M05's _neon_model().

    S(r) = A(r; λ₁) + ne_ratio × A(r; λ₂) + B
    """
    r_fine = np.linspace(0.0, r_max, _N_fine)
    # airy_modified accepts (r, lam, t, R, alpha, n, r_max, I0, I1, I2,
    #                        sigma0, sigma1, sigma2)
    A1 = airy_modified(r_fine, NE_WAVELENGTH_1_AIR_M,
                       t, R, alpha, 1.0, r_max, I0, I1, I2, sigma0, sigma1, sigma2)
    A2 = airy_modified(r_fine, NE_WAVELENGTH_2_AIR_M,
                       t, R, alpha, 1.0, r_max, I0, I1, I2, sigma0, sigma1, sigma2)
    model_fine = A1 + ne_ratio * A2 + B
    return np.interp(r_arr, r_fine, model_fine)


# ---------------------------------------------------------------------------
# LM stage runner
# ---------------------------------------------------------------------------

def _run_stage(r_good, prof_good, sig_good, r_max,
               p_all, free_names, bounds_dict, config):
    """
    Run one Levenberg-Marquardt stage with soft-bound penalties.

    Returns (p_updated, cov, stderrs, chi2_red, lm_result).
    """
    free_idx  = np.array([_IDX[n] for n in free_names])
    p_fixed   = p_all.copy()
    p0        = p_fixed[free_idx]
    n_good    = len(r_good)
    n_free    = len(free_names)

    lo_arr    = np.array([bounds_dict[n][1] for n in free_names])
    hi_arr    = np.array([bounds_dict[n][2] for n in free_names])
    range_arr = hi_arr - lo_arr + 1e-30
    pen_sigma = range_arr * 0.01   # soft-bound penalty: 1% of range per unit

    def _residuals(p_free):
        p = p_fixed.copy()
        p[free_idx] = p_free
        t, alpha, R, I0, I1, I2, sigma0, sigma1, sigma2, B, ne_ratio = p
        model  = _neon_model_11(r_good, r_max, t, alpha, R, I0, I1, I2,
                                sigma0, sigma1, sigma2, B, ne_ratio)
        data_r = (prof_good - model) / sig_good
        below  = np.maximum(0.0, lo_arr - p_free) / pen_sigma
        above  = np.maximum(0.0, p_free - hi_arr) / pen_sigma
        return np.append(data_r, below + above)

    lm = least_squares(_residuals, p0, method='lm',
                       ftol=config['ftol'], xtol=config['xtol'],
                       gtol=config['gtol'], max_nfev=config['max_nfev'])

    p_updated = p_fixed.copy()
    p_updated[free_idx] = lm.x

    # chi² from data residuals only (not penalty rows)
    t, alpha, R, I0, I1, I2, sigma0, sigma1, sigma2, B, ne_ratio = p_updated
    model_f = _neon_model_11(r_good, r_max, t, alpha, R, I0, I1, I2,
                             sigma0, sigma1, sigma2, B, ne_ratio)
    data_r  = (prof_good - model_f) / sig_good
    dof     = max(n_good - n_free, 1)
    chi2    = float(np.sum(data_r ** 2)) / dof

    # Covariance from data-only Jacobian rows
    J = lm.jac[:n_good, :]
    try:
        JTJ = J.T @ J
        if np.linalg.cond(JTJ) < 1e14:
            JTJ_inv = np.linalg.inv(JTJ)
        else:
            JTJ_inv = np.linalg.pinv(JTJ, rcond=1e-10)
        cov     = chi2 * JTJ_inv
        stderrs = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except (np.linalg.LinAlgError, ValueError):
        stderrs = np.full(n_free, np.inf)
        cov     = np.full((n_free, n_free), np.inf)

    return p_updated, cov, stderrs, chi2, lm


# ---------------------------------------------------------------------------
# Minimal FringeProfile shim
# ---------------------------------------------------------------------------

@dataclass
class _FringeProfile:
    profile:       np.ndarray
    r_grid:        np.ndarray
    sigma_profile: np.ndarray
    masked:        np.ndarray
    r_max_px:      float


# ---------------------------------------------------------------------------
# Fit result dataclass
# ---------------------------------------------------------------------------

@dataclass
class FitResult11:
    """Fit result for 11-parameter neon calibration inversion."""
    t_m:      float;  alpha:    float;  R_refl:   float
    I0:       float;  I1:       float;  I2:       float
    sigma0:   float;  sigma1:   float;  sigma2:   float
    B:        float;  ne_ratio: float

    sigma_t_m:      float;  sigma_alpha:    float;  sigma_R_refl:   float
    sigma_I0:       float;  sigma_I1:       float;  sigma_I2:       float
    sigma_sigma0:   float;  sigma_sigma1:   float;  sigma_sigma2:   float
    sigma_B:        float;  sigma_ne_ratio: float

    epsilon_cal:       float
    sigma_epsilon_cal: float

    chi2_reduced:  float
    chi2_by_stage: list
    n_bins_used:   int
    converged:     bool


# ---------------------------------------------------------------------------
# Staged inversion
# ---------------------------------------------------------------------------

def run_staged_inversion(fp: _FringeProfile,
                         t_eff: float,
                         alpha_init: float,
                         eps_a: float,
                         R_init: float = 0.53,
                         sigma0_init: float = 0.5,
                         ne_ratio_init: float = None,
                         max_nfev: int = 100_000,
                         ftol: float = 1e-14,
                         xtol: float = 1e-14,
                         gtol: float = 1e-14) -> FitResult11:
    """
    Run the 4-stage LM inversion on fp with 11 free parameters.

    Stage 1: I0, I1, I2, B                          (photometric baseline)
    Stage 2: t, alpha, R, I0, I1, I2, B             (geometry + reflectivity)
    Stage 3: t, alpha, R, I0, I1, I2, sigma0, B     (+ PSF base width)
    Stage 4: all 11                                  (+ sigma1, sigma2, ne_ratio)
    """
    good     = (~fp.masked & np.isfinite(fp.sigma_profile)
                & (fp.sigma_profile > 0) & np.isfinite(fp.profile))
    r_good   = fp.r_grid[good]
    p_good   = fp.profile[good]
    s_good   = fp.sigma_profile[good].copy()
    r_max    = float(fp.r_max_px)
    n_good   = int(good.sum())

    # Sigma floor: prevent near-zero weights inflating chi² on noiseless data
    s_floor = max(1.0, float(np.median(p_good)) * 0.005)
    s_good  = np.maximum(s_good, s_floor)

    if n_good < 30:
        raise ValueError(f"Only {n_good} usable bins — need ≥ 30.")

    # Initial estimates from profile statistics
    I0_init = float(np.percentile(p_good, 75))
    B_init  = float(np.percentile(p_good, 5)) * 0.8
    if ne_ratio_init is None:
        ne_ratio_init = float(NE_INTENSITY_2_NOMINAL)

    # Bounds: (init, lo, hi)
    bounds = {
        't_m':      (t_eff,          t_eff - 20e-6,     t_eff + 20e-6),
        'alpha':    (alpha_init,     alpha_init * 0.95, alpha_init * 1.05),
        'R':        (R_init,         0.05,              0.95),
        'I0':       (I0_init,        100.0,             15000.0),
        'I1':       (0.0,            -0.5,              0.5),
        'I2':       (0.0,            -0.5,              0.5),
        'sigma0':   (sigma0_init,    0.01,              5.0),
        'sigma1':   (0.0,            -2.0,              2.0),
        'sigma2':   (0.0,            -2.0,              2.0),
        'B':        (B_init,         10.0,              2000.0),
        'ne_ratio': (ne_ratio_init,  0.01,              2.0),
    }

    cfg = dict(max_nfev=max_nfev, ftol=ftol, xtol=xtol, gtol=gtol)
    p_all = np.array([bounds[n][0] for n in _NAMES], dtype=float)

    chi2_by_stage = []

    log.info("Stage 1 — photometric baseline {I0, I1, I2, B}")
    p_all, _, _, chi2_1, _ = _run_stage(
        r_good, p_good, s_good, r_max, p_all, _STAGE_FREE[1], bounds, cfg)
    chi2_by_stage.append(chi2_1)
    log.info(f"  χ²/ν = {chi2_1:.3f}")

    log.info("Stage 2 — geometry + reflectivity {t, alpha, R, I0, I1, I2, B}")
    p_all, _, _, chi2_2, _ = _run_stage(
        r_good, p_good, s_good, r_max, p_all, _STAGE_FREE[2], bounds, cfg)
    chi2_by_stage.append(chi2_2)
    log.info(f"  χ²/ν = {chi2_2:.3f}")

    log.info("Stage 3 — + PSF base width sigma0")
    p_all, _, _, chi2_3, _ = _run_stage(
        r_good, p_good, s_good, r_max, p_all, _STAGE_FREE[3], bounds, cfg)
    chi2_by_stage.append(chi2_3)
    log.info(f"  χ²/ν = {chi2_3:.3f}")

    log.info("Stage 4 — full free optimisation (all 11 params incl. ne_ratio)")
    p_all, cov4, se4, chi2_4, res4 = _run_stage(
        r_good, p_good, s_good, r_max, p_all, _STAGE_FREE[4], bounds, cfg)
    chi2_by_stage.append(chi2_4)
    log.info(f"  χ²/ν = {chi2_4:.3f}")

    converged = bool(res4.success or res4.cost < 1e-10)
    t_fit, alpha_fit, R_fit, I0_fit, I1_fit, I2_fit, \
        s0_fit, s1_fit, s2_fit, B_fit, ne_fit = p_all

    eps_cal       = (2.0 * t_fit / NE_WAVELENGTH_1_AIR_M) % 1.0
    sigma_eps_cal = (2.0 / NE_WAVELENGTH_1_AIR_M) * float(se4[_IDX['t_m']])

    return FitResult11(
        t_m=float(t_fit),   alpha=float(alpha_fit),   R_refl=float(R_fit),
        I0=float(I0_fit),   I1=float(I1_fit),         I2=float(I2_fit),
        sigma0=float(s0_fit), sigma1=float(s1_fit),   sigma2=float(s2_fit),
        B=float(B_fit),     ne_ratio=float(ne_fit),

        sigma_t_m=float(se4[_IDX['t_m']]),
        sigma_alpha=float(se4[_IDX['alpha']]),
        sigma_R_refl=float(se4[_IDX['R']]),
        sigma_I0=float(se4[_IDX['I0']]),
        sigma_I1=float(se4[_IDX['I1']]),
        sigma_I2=float(se4[_IDX['I2']]),
        sigma_sigma0=float(se4[_IDX['sigma0']]),
        sigma_sigma1=float(se4[_IDX['sigma1']]),
        sigma_sigma2=float(se4[_IDX['sigma2']]),
        sigma_B=float(se4[_IDX['B']]),
        sigma_ne_ratio=float(se4[_IDX['ne_ratio']]),

        epsilon_cal=float(eps_cal),
        sigma_epsilon_cal=float(sigma_eps_cal),

        chi2_reduced=float(chi2_4),
        chi2_by_stage=chi2_by_stage,
        n_bins_used=n_good,
        converged=converged,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_profile(path, r_max_px):
    arr = np.load(path)
    sigma = None
    if arr.ndim == 1:
        log.warning("1D profile — inferring r² as linspace(0, r_max², N)")
        r2, prof = np.linspace(0.0, r_max_px**2, len(arr)), arr.astype(float)
    elif arr.ndim == 2 and arr.shape[0] == 3:
        r2, prof, sigma = arr[0].astype(float), arr[1].astype(float), arr[2].astype(float)
    elif arr.ndim == 2 and arr.shape[0] == 2:
        r2, prof = arr[0].astype(float), arr[1].astype(float)
    elif arr.ndim == 2 and arr.shape[1] == 2:
        r2, prof = arr[:, 0].astype(float), arr[:, 1].astype(float)
    else:
        raise ValueError(f"Unexpected array shape {arr.shape}")
    return np.sqrt(np.maximum(r2, 0.0)), prof, sigma


def _estimate_sigma(profile):
    n, half = len(profile), 2
    sigma = np.empty(n)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        sigma[i] = float(np.std(profile[lo:hi]))
    return np.maximum(sigma, max(1.0, float(np.median(profile)) * 0.005))


# ---------------------------------------------------------------------------
# Figure — Harding Fig. 4 style
# ---------------------------------------------------------------------------

def make_figure(r2_data, profile, sigma, r2_model, model_best,
                fit: FitResult11, source_name: str = "") -> plt.Figure:

    model_at_data = np.interp(r2_data, r2_model, model_best)
    residual      = profile - model_at_data

    fig = plt.figure(figsize=(14, 10.5))
    gs  = gridspec.GridSpec(3, 1, height_ratios=[3, 1.5, 2.2],
                            hspace=0.08, top=0.93, bottom=0.03,
                            left=0.09, right=0.97)
    ax_fit = fig.add_subplot(gs[0])
    ax_res = fig.add_subplot(gs[1], sharex=ax_fit)
    ax_tbl = fig.add_subplot(gs[2])
    ax_tbl.axis("off")

    # Top panel — data + model
    ax_fit.errorbar(r2_data, profile, yerr=sigma, fmt="none",
                    ecolor="darkorange", elinewidth=0.5, alpha=0.4, zorder=2)
    ax_fit.plot(r2_data, profile, color="darkorange", lw=0.9, alpha=0.85,
                zorder=3, label=f"Data  ({source_name})")
    ax_fit.plot(r2_model, model_best, color="black", lw=1.5,
                zorder=4, label="Best-fit modified Airy  (11-param Stage 4)")
    ax_fit.set_ylabel("CCD signal  (ADU)", fontsize=11)
    ax_fit.legend(fontsize=9, loc="upper right")
    ax_fit.grid(True, alpha=0.2)
    ax_fit.tick_params(labelbottom=False)

    conv_str = "converged" if fit.converged else "NOT converged"
    ax_fit.text(
        0.02, 0.97,
        f"χ²/ν = {fit.chi2_reduced:.3f}   {conv_str}\n"
        f"ne_ratio (fitted) = {fit.ne_ratio:.4f} ± {fit.sigma_ne_ratio:.4f}   "
        f"[nominal = {NE_INTENSITY_2_NOMINAL:.2f}]",
        transform=ax_fit.transAxes, va="top", ha="left", fontsize=8.5,
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="grey", alpha=0.85))

    # Middle panel — residual
    ax_res.axhline(0, color="black", lw=0.8, ls="--")
    ax_res.fill_between(r2_data, -sigma, sigma,
                        color="steelblue", alpha=0.25, label="±1σ")
    ax_res.plot(r2_data, residual, color="steelblue", lw=0.9,
                alpha=0.9, label="Residual")
    ax_res.set_xlabel(r"$r^2$  (pixels², 2×2 binned)", fontsize=11)
    ax_res.set_ylabel("Residual  (ADU)", fontsize=10)
    ax_res.legend(fontsize=8.5, loc="upper right")
    ax_res.grid(True, alpha=0.2)
    ax_res.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, symmetric=True))

    # Bottom panel — parameter table
    rows = [
        ("t",        f"{fit.t_m*1e3:.7f} mm",    f"±{fit.sigma_t_m*1e6:.3f} µm",    "Etalon gap  [Tolansky-seeded, Group A]"),
        ("α",        f"{fit.alpha:.5e} rad/px",   f"±{fit.sigma_alpha:.2e}",          "Plate scale  [Tolansky-seeded, Group A]"),
        ("R",        f"{fit.R_refl:.4f}",          f"±{fit.sigma_R_refl:.4f}",         "Effective reflectivity  [Group B]"),
        ("I₀",       f"{fit.I0:.1f} ADU",          f"±{fit.sigma_I0:.1f}",             "Mean intensity  [Group B]"),
        ("I₁",       f"{fit.I1:.4f}",              f"±{fit.sigma_I1:.4f}",             "Linear vignetting  [Group B]"),
        ("I₂",       f"{fit.I2:.5f}",              f"±{fit.sigma_I2:.5f}",             "Quadratic vignetting  [Group B]"),
        ("σ₀",       f"{fit.sigma0:.4f} px",       f"±{fit.sigma_sigma0:.4f}",         "PSF base width  [Group B]"),
        ("σ₁",       f"{fit.sigma1:.4f} px",       f"±{fit.sigma_sigma1:.4f}",         "PSF sin variation  [Group B]"),
        ("σ₂",       f"{fit.sigma2:.4f} px",       f"±{fit.sigma_sigma2:.4f}",         "PSF cos variation  [Group B]"),
        ("B",        f"{fit.B:.1f} ADU",            f"±{fit.sigma_B:.2f}",              "CCD bias pedestal  [Group B]"),
        ("ne_ratio", f"{fit.ne_ratio:.4f}",         f"±{fit.sigma_ne_ratio:.4f}",       f"λ₂/λ₁ intensity ratio  [Group B, nominal={NE_INTENSITY_2_NOMINAL:.2f}]"),
        ("ε_cal",    f"{fit.epsilon_cal:.6f}",      f"±{fit.sigma_epsilon_cal:.6f}",    "Fractional order at centre  (zero-wind reference)"),
    ]

    tbl = ax_tbl.table(
        cellText=rows,
        colLabels=["Param", "Fitted value", "1σ", "Description"],
        cellLoc="left", loc="upper center",
        colWidths=[0.08, 0.21, 0.15, 0.56])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.28)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#c8d8f0")
            cell.set_text_props(fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f0f4ff")
        if row == 11:   # ne_ratio row — highlight in yellow
            cell.set_facecolor("#fff4c2")

    stage_str = "  ".join(f"S{i+1}: {v:.2f}" for i, v in enumerate(fit.chi2_by_stage))
    ax_tbl.text(0.01, 0.02,
                f"χ²/ν by stage:  {stage_str}    bins used: {fit.n_bins_used}   "
                f"free params: 11",
                transform=ax_tbl.transAxes, va="bottom", ha="left",
                fontsize=8.5, fontfamily="monospace", color="dimgrey")

    fig.suptitle(
        "WindCube FPI — Neon Calibration Fringe Inversion  (11-param / Harding 2014)",
        fontsize=12, fontweight="bold", y=0.97)
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    # 1. File picker
    _tk = tk.Tk(); _tk.withdraw()
    npy_path = filedialog.askopenfilename(
        title="Select neon calibration profile (.npy, tabulated vs r²)",
        filetypes=[("NumPy arrays", "*.npy"), ("All files", "*.*")])
    _tk.destroy()
    if not npy_path:
        print("No file selected — exiting."); return
    npy_path = pathlib.Path(npy_path)
    print(f"\nLoaded: {npy_path.name}")

    # 2. Parameter dialogs
    _tk2 = tk.Tk(); _tk2.withdraw()

    r_max_px = simpledialog.askfloat("r_max",
        "Maximum usable fringe radius (px, 2×2 binned).\nFlatSat/flight: 110",
        initialvalue=110.0, minvalue=50.0, maxvalue=200.0) or 110.0

    t_tolansky_mm = simpledialog.askfloat("Tolansky gap",
        "Tolansky-recovered etalon gap (mm).\nExample: 20.1070707",
        initialvalue=20.1070707, minvalue=19.5, maxvalue=20.5) or 20.1070707

    eps_a = simpledialog.askfloat("Tolansky ε_a",
        "Tolansky excess fraction ε_a for λ₁ = 640.2248 nm.\nExample: 0.23286",
        initialvalue=0.23286, minvalue=0.0, maxvalue=0.9999) or 0.23286

    alpha_tolansky = simpledialog.askfloat("Tolansky alpha",
        "Tolansky plate scale α (rad/px, 2×2 binned).\nExample: 1.6084e-4",
        initialvalue=1.6084e-4, minvalue=1e-5, maxvalue=1e-3) or 1.6084e-4

    ne_ratio_init = simpledialog.askfloat("ne_ratio initial guess",
        "Starting guess for λ₂/λ₁ intensity ratio.\n"
        f"Nominal = {NE_INTENSITY_2_NOMINAL:.2f} — the fit will find the true value.",
        initialvalue=float(NE_INTENSITY_2_NOMINAL),
        minvalue=0.01, maxvalue=2.0) or float(NE_INTENSITY_2_NOMINAL)

    _tk2.destroy()

    t_tolansky_m = t_tolansky_mm * 1e-3

    # 3. Phase-correct the gap
    t_eff = phase_correct_gap(t_tolansky_m, eps_a, NE_WAVELENGTH_1_AIR_M)
    print(f"\nphase_correct_gap:  t_tolansky = {t_tolansky_m*1e3:.7f} mm  "
          f"→  t_eff = {t_eff*1e3:.7f} mm  "
          f"(correction = {(t_eff - t_tolansky_m)*1e9:.1f} nm)")

    # 4. Load profile
    r_grid, profile_adu, sigma_loaded = _load_profile(npy_path, r_max_px)
    in_range    = r_grid <= r_max_px
    r_grid      = r_grid[in_range]
    profile_adu = profile_adu[in_range]
    if sigma_loaded is not None:
        sigma_loaded = sigma_loaded[in_range]
        bad = ~np.isfinite(sigma_loaded)
        if bad.any():
            sigma_loaded[bad] = _estimate_sigma(profile_adu)[bad]
        sigma_adu = sigma_loaded
        print(f"  SEM from file  ({bad.sum()} bins estimated by fallback)")
    else:
        sigma_adu = _estimate_sigma(profile_adu)
        print("  SEM estimated from local scatter")

    fp = _FringeProfile(
        profile=profile_adu, r_grid=r_grid,
        sigma_profile=sigma_adu,
        masked=np.zeros(len(r_grid), dtype=bool),
        r_max_px=r_max_px)

    print(f"  {len(r_grid)} bins, r ∈ [{r_grid.min():.1f}, {r_grid.max():.1f}] px, "
          f"signal ∈ [{profile_adu.min():.0f}, {profile_adu.max():.0f}] ADU")

    # 5. Run inversion
    print("\nRunning 11-parameter staged LM inversion…")
    fit = run_staged_inversion(
        fp, t_eff, alpha_tolansky, eps_a,
        ne_ratio_init=ne_ratio_init)

    # 6. Print results
    print(f"\n{'='*65}")
    print("11-PARAMETER CALIBRATION INVERSION RESULT")
    print(f"{'='*65}")
    print(f"  Converged:      {fit.converged}")
    print(f"  χ²/ν:           {fit.chi2_reduced:.4f}")
    print(f"  χ²/ν by stage:  {[f'{v:.3f}' for v in fit.chi2_by_stage]}")
    print(f"  Bins used:      {fit.n_bins_used}")
    print()
    print(f"  --- Group A (Tolansky-seeded) ---")
    print(f"  t      = {fit.t_m*1e3:.7f} mm  ±{fit.sigma_t_m*1e6:.3f} µm")
    print(f"  alpha  = {fit.alpha:.5e}  ±{fit.sigma_alpha:.2e} rad/px")
    print()
    print(f"  --- Group B (fringe shape + new ne_ratio) ---")
    print(f"  R        = {fit.R_refl:.4f}  ±{fit.sigma_R_refl:.4f}")
    print(f"  I0       = {fit.I0:.1f}  ±{fit.sigma_I0:.1f} ADU")
    print(f"  I1       = {fit.I1:.4f}  ±{fit.sigma_I1:.4f}")
    print(f"  I2       = {fit.I2:.5f}  ±{fit.sigma_I2:.5f}")
    print(f"  sigma0   = {fit.sigma0:.4f}  ±{fit.sigma_sigma0:.4f} px")
    print(f"  sigma1   = {fit.sigma1:.4f}  ±{fit.sigma_sigma1:.4f} px")
    print(f"  sigma2   = {fit.sigma2:.4f}  ±{fit.sigma_sigma2:.4f} px")
    print(f"  B        = {fit.B:.2f}  ±{fit.sigma_B:.2f} ADU")
    print(f"  ne_ratio = {fit.ne_ratio:.4f}  ±{fit.sigma_ne_ratio:.4f}  "
          f"[nominal {NE_INTENSITY_2_NOMINAL:.2f}]")
    print()
    print(f"  --- Phase reference ---")
    print(f"  ε_cal  = {fit.epsilon_cal:.6f}  ±{fit.sigma_epsilon_cal:.6f}")
    print(f"{'='*65}")

    # 7. Reconstruct model on fine r grid for smooth plot
    r_fine = np.linspace(0.0, r_max_px, 2000)
    model_fine = _neon_model_11(
        r_fine, r_max_px,
        fit.t_m, fit.alpha, fit.R_refl,
        fit.I0, fit.I1, fit.I2,
        fit.sigma0, fit.sigma1, fit.sigma2,
        fit.B, fit.ne_ratio)

    # 8. Plot
    fig = make_figure(
        r2_data=r_grid**2, profile=profile_adu, sigma=sigma_adu,
        r2_model=r_fine**2, model_best=model_fine,
        fit=fit, source_name=npy_path.name)
    plt.show()


if __name__ == "__main__":
    main()
