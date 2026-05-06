"""
Script:  H05_calibration_inversion_2026_05_06.py
Purpose: Load a real two-line neon calibration radial profile (tabulated vs r²),
         run a 12-parameter staged Levenberg-Marquardt calibration inversion,
         and produce a diagnostic figure modelled on Harding et al. (2014) Fig. 4.

         KEY CHANGE FROM PREVIOUS VERSION (11-param):
         The two neon lines now have INDEPENDENT effective reflectivities:
           R1 — for λ₁ = 640.2248 nm  (strong line)
           R2 — for λ₂ = 638.2991 nm  (weak line)

         Physical motivation: the etalon coating reflectivity is wavelength-
         dependent.  The ~2 nm separation between the two neon lines is
         sufficient to produce a measurable difference in effective finesse,
         causing λ₂ peaks to be sharper/taller than the λ₁ peaks.  Forcing
         a single shared R causes the fit to push σ₂ to its bound trying to
         compensate.  Allowing R1 ≠ R2 resolves this.

         Parameters fitted (12 total):
           Group A — Tolansky-seeded, tight bounds:
             t_eff   — phase-corrected etalon gap (H01 §8)
             alpha   — plate scale (rad/px)
           Group B — freely fitted from fringe shape:
             R1      — effective reflectivity for λ₁ = 640.2 nm
             R2      — effective reflectivity for λ₂ = 638.3 nm  ← NEW
             I0, I1, I2   — shared intensity envelope
             sigma0, sigma1, sigma2  — shared PSF width
             B       — CCD bias pedestal
             ne_ratio — λ₂/λ₁ intensity scale ratio

         Forward model:
           S(r) = Ã(r; λ₁, R1) + ne_ratio × Ã(r; λ₂, R2) + B

         Staged inversion (follows H05 architecture):
           Stage 1: I0, I1, I2, B              (photometric baseline)
           Stage 2: t, alpha, R1, R2, I0, I1, I2, B   (geometry + reflectivities)
           Stage 3: + sigma0                   (PSF base width)
           Stage 4: all 12                     (+ sigma1, sigma2, ne_ratio)

Input .npy file formats accepted:
   (N,)      — profile values only; r² inferred (not recommended)
   (2, N)    — row 0 = r² (px²), row 1 = profile (ADU)
   (3, N)    — row 0 = r² (px²), row 1 = profile (ADU), row 2 = SEM (ADU)
   (N, 2)    — col 0 = r² (px²), col 1 = profile (ADU)

Run from repo root:
    python src/processing/H05_calibration_inversion_2026_05_06.py
"""

import pathlib
import sys
import tkinter as tk
from tkinter import filedialog, simpledialog
import logging
from dataclasses import dataclass
from typing import Optional, List

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
log = logging.getLogger("H05")

# ---------------------------------------------------------------------------
# Parameter ordering (12 parameters)
# ---------------------------------------------------------------------------
# Idx:  0      1        2     3     4     5     6     7        8        9        10   11
_NAMES = ['t_m', 'alpha', 'R1', 'R2', 'I0', 'I1', 'I2', 'sigma0', 'sigma1', 'sigma2', 'B', 'ne_ratio']
_IDX   = {n: i for i, n in enumerate(_NAMES)}

# Stage free-parameter sets
_STAGE_FREE = {
    1: ['I0', 'I1', 'I2', 'B'],
    2: ['t_m', 'alpha', 'R1', 'R2', 'I0', 'I1', 'I2', 'B'],
    3: ['t_m', 'alpha', 'R1', 'R2', 'I0', 'I1', 'I2', 'sigma0', 'B'],
    4: _NAMES,   # all 12
}

# ---------------------------------------------------------------------------
# Forward model — two-line neon with independent R1, R2 and free ne_ratio
# ---------------------------------------------------------------------------

def _neon_model_12(r_arr, r_max, t, alpha, R1, R2, I0, I1, I2,
                   sigma0, sigma1, sigma2, B, ne_ratio, _N_fine=500):
    """
    Two-line neon forward model with separate reflectivities for each line.

        S(r) = Ã(r; λ₁, R1) + ne_ratio × Ã(r; λ₂, R2) + B

    Both lines share the same intensity envelope (I0, I1, I2) and PSF
    (sigma0, sigma1, sigma2), but have independent effective reflectivities
    R1 and R2.  This allows the model to capture wavelength-dependent finesse.

    Evaluates on a fine uniform grid then interpolates to r_arr.
    """
    r_fine = np.linspace(0.0, r_max, _N_fine)

    # airy_modified signature: (r, lam, t, R, alpha, n, r_max,
    #                            I0, I1, I2, sigma0, sigma1, sigma2)
    A1 = airy_modified(r_fine, NE_WAVELENGTH_1_AIR_M,
                       t, R1, alpha, 1.0, r_max,
                       I0, I1, I2, sigma0, sigma1, sigma2)
    A2 = airy_modified(r_fine, NE_WAVELENGTH_2_AIR_M,
                       t, R2, alpha, 1.0, r_max,
                       I0, I1, I2, sigma0, sigma1, sigma2)

    model_fine = A1 + ne_ratio * A2 + B
    return np.interp(r_arr, r_fine, model_fine)


def _components_12(r_arr, r_max, t, alpha, R1, R2, I0, I1, I2,
                   sigma0, sigma1, sigma2, B, ne_ratio, n_fine=2000):
    """Return (composite, lam1_plus_B, ne_ratio*lam2_plus_B) for plotting."""
    r_fine = np.linspace(0.0, r_max, n_fine)
    A1 = airy_modified(r_fine, NE_WAVELENGTH_1_AIR_M,
                       t, R1, alpha, 1.0, r_max,
                       I0, I1, I2, sigma0, sigma1, sigma2)
    A2 = airy_modified(r_fine, NE_WAVELENGTH_2_AIR_M,
                       t, R2, alpha, 1.0, r_max,
                       I0, I1, I2, sigma0, sigma1, sigma2)
    comp  = np.interp(r_arr, r_fine, A1 + ne_ratio * A2 + B)
    lam1  = np.interp(r_arr, r_fine, A1 + B)
    lam2  = np.interp(r_arr, r_fine, ne_ratio * A2 + B)
    return comp, lam1, lam2


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
    pen_sigma = range_arr * 0.01

    def _residuals(p_free):
        p = p_fixed.copy()
        p[free_idx] = p_free
        t, alpha, R1, R2, I0, I1, I2, sigma0, sigma1, sigma2, B, ne_ratio = p
        model  = _neon_model_12(r_good, r_max, t, alpha, R1, R2,
                                I0, I1, I2, sigma0, sigma1, sigma2, B, ne_ratio)
        data_r = (prof_good - model) / sig_good
        below  = np.maximum(0.0, lo_arr - p_free) / pen_sigma
        above  = np.maximum(0.0, p_free - hi_arr) / pen_sigma
        return np.append(data_r, below + above)

    lm = least_squares(_residuals, p0, method='lm',
                       ftol=config['ftol'], xtol=config['xtol'],
                       gtol=config['gtol'], max_nfev=config['max_nfev'])

    p_updated = p_fixed.copy()
    p_updated[free_idx] = lm.x

    # chi² from data residuals only
    t, alpha, R1, R2, I0, I1, I2, sigma0, sigma1, sigma2, B, ne_ratio = p_updated
    model_f = _neon_model_12(r_good, r_max, t, alpha, R1, R2,
                             I0, I1, I2, sigma0, sigma1, sigma2, B, ne_ratio)
    data_r  = (prof_good - model_f) / sig_good
    dof     = max(n_good - n_free, 1)
    chi2    = float(np.sum(data_r ** 2)) / dof

    # Covariance from data-only Jacobian rows
    J = lm.jac[:n_good, :]
    try:
        JTJ = J.T @ J
        JTJ_inv = (np.linalg.inv(JTJ)
                   if np.linalg.cond(JTJ) < 1e14
                   else np.linalg.pinv(JTJ, rcond=1e-10))
        cov     = chi2 * JTJ_inv
        stderrs = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except (np.linalg.LinAlgError, ValueError):
        stderrs = np.full(n_free, np.inf)
        cov     = np.full((n_free, n_free), np.inf)

    return p_updated, cov, stderrs, chi2, lm


# ---------------------------------------------------------------------------
# Shims and dataclasses
# ---------------------------------------------------------------------------

@dataclass
class _FringeProfile:
    profile:       np.ndarray
    r_grid:        np.ndarray
    sigma_profile: np.ndarray
    masked:        np.ndarray
    r_max_px:      float


@dataclass
class FitResult12:
    """Fit result for 12-parameter neon calibration inversion."""
    t_m:      float;  alpha:    float
    R1:       float;  R2:       float
    I0:       float;  I1:       float;  I2:      float
    sigma0:   float;  sigma1:   float;  sigma2:  float
    B:        float;  ne_ratio: float

    sigma_t_m:      float;  sigma_alpha:  float
    sigma_R1:       float;  sigma_R2:     float
    sigma_I0:       float;  sigma_I1:     float;  sigma_I2:      float
    sigma_sigma0:   float;  sigma_sigma1: float;  sigma_sigma2:  float
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
                         R1_init: float = 0.53,
                         R2_init: float = 0.53,
                         sigma0_init: float = 0.5,
                         ne_ratio_init: float = None,
                         max_nfev: int = 100_000,
                         ftol: float = 1e-14,
                         xtol: float = 1e-14,
                         gtol: float = 1e-14) -> FitResult12:
    """
    Run the 4-stage LM inversion with 12 free parameters.

    R1 and R2 are both free from Stage 2 onward with independent bounds,
    allowing the fit to capture wavelength-dependent finesse differences
    between the two neon lines.
    """
    good     = (~fp.masked & np.isfinite(fp.sigma_profile)
                & (fp.sigma_profile > 0) & np.isfinite(fp.profile))
    r_good   = fp.r_grid[good]
    p_good   = fp.profile[good]
    s_good   = fp.sigma_profile[good].copy()
    r_max    = float(fp.r_max_px)
    n_good   = int(good.sum())

    s_floor = max(1.0, float(np.median(p_good)) * 0.005)
    s_good  = np.maximum(s_good, s_floor)

    if n_good < 30:
        raise ValueError(f"Only {n_good} usable bins — need ≥ 30.")

    I0_init = float(np.percentile(p_good, 75))
    B_init  = float(np.percentile(p_good, 5)) * 0.8
    if ne_ratio_init is None:
        ne_ratio_init = float(NE_INTENSITY_2_NOMINAL)

    # bounds: (init, lo, hi)
    bounds = {
        't_m':      (t_eff,          t_eff - 20e-6,      t_eff + 20e-6),
        'alpha':    (alpha_init,     alpha_init * 0.95,  alpha_init * 1.05),
        'R1':       (R1_init,        0.05,               0.95),
        'R2':       (R2_init,        0.05,               0.95),
        'I0':       (I0_init,        100.0,              15000.0),
        'I1':       (0.0,            -0.5,               0.5),
        'I2':       (0.0,            -0.5,               0.5),
        'sigma0':   (sigma0_init,    0.01,               5.0),
        'sigma1':   (0.0,            -2.0,               2.0),
        'sigma2':   (0.0,            -2.0,               2.0),
        'B':        (B_init,         10.0,               2000.0),
        'ne_ratio': (ne_ratio_init,  0.01,               2.0),
    }

    cfg   = dict(max_nfev=max_nfev, ftol=ftol, xtol=xtol, gtol=gtol)
    p_all = np.array([bounds[n][0] for n in _NAMES], dtype=float)

    chi2_by_stage = []

    log.info("Stage 1 — photometric baseline {I0, I1, I2, B}")
    p_all, _, _, chi2_1, _ = _run_stage(
        r_good, p_good, s_good, r_max, p_all, _STAGE_FREE[1], bounds, cfg)
    chi2_by_stage.append(chi2_1)
    log.info(f"  χ²/ν = {chi2_1:.3f}")

    log.info("Stage 2 — geometry + R1 + R2 {t, alpha, R1, R2, I0, I1, I2, B}")
    p_all, _, _, chi2_2, _ = _run_stage(
        r_good, p_good, s_good, r_max, p_all, _STAGE_FREE[2], bounds, cfg)
    chi2_by_stage.append(chi2_2)
    log.info(f"  χ²/ν = {chi2_2:.3f}   R1 = {p_all[_IDX['R1']]:.4f}   "
             f"R2 = {p_all[_IDX['R2']]:.4f}")

    log.info("Stage 3 — + PSF base width sigma0")
    p_all, _, _, chi2_3, _ = _run_stage(
        r_good, p_good, s_good, r_max, p_all, _STAGE_FREE[3], bounds, cfg)
    chi2_by_stage.append(chi2_3)
    log.info(f"  χ²/ν = {chi2_3:.3f}   sigma0 = {p_all[_IDX['sigma0']]:.4f} px")

    log.info("Stage 4 — full free optimisation (all 12 params)")
    p_all, cov4, se4, chi2_4, res4 = _run_stage(
        r_good, p_good, s_good, r_max, p_all, _STAGE_FREE[4], bounds, cfg)
    chi2_by_stage.append(chi2_4)
    log.info(f"  χ²/ν = {chi2_4:.3f}   R1 = {p_all[_IDX['R1']]:.4f}   "
             f"R2 = {p_all[_IDX['R2']]:.4f}   ne_ratio = {p_all[_IDX['ne_ratio']]:.4f}")

    converged = bool(res4.success or res4.cost < 1e-10)
    t_f, alpha_f, R1_f, R2_f, I0_f, I1_f, I2_f, \
        s0_f, s1_f, s2_f, B_f, ne_f = p_all

    eps_cal       = (2.0 * t_f / NE_WAVELENGTH_1_AIR_M) % 1.0
    sigma_eps_cal = (2.0 / NE_WAVELENGTH_1_AIR_M) * float(se4[_IDX['t_m']])

    return FitResult12(
        t_m=float(t_f),     alpha=float(alpha_f),
        R1=float(R1_f),     R2=float(R2_f),
        I0=float(I0_f),     I1=float(I1_f),     I2=float(I2_f),
        sigma0=float(s0_f), sigma1=float(s1_f), sigma2=float(s2_f),
        B=float(B_f),       ne_ratio=float(ne_f),

        sigma_t_m=float(se4[_IDX['t_m']]),
        sigma_alpha=float(se4[_IDX['alpha']]),
        sigma_R1=float(se4[_IDX['R1']]),
        sigma_R2=float(se4[_IDX['R2']]),
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
        r2, prof = arr[0].astype(float), arr[1].astype(float)
        sigma    = arr[2].astype(float)
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
# Figure — Harding Fig. 4 style, updated for 12 parameters
# ---------------------------------------------------------------------------

def make_figure(r2_data, profile, sigma,
                r_fine, model_fine, lam1_fine, lam2_fine,
                fit: FitResult12, source_name: str = "") -> plt.Figure:

    r2_fine       = r_fine ** 2
    model_at_data = np.interp(r2_data, r2_fine, model_fine)
    residual      = profile - model_at_data

    fig = plt.figure(figsize=(14, 11))
    gs  = gridspec.GridSpec(3, 1, height_ratios=[3, 1.5, 2.5],
                            hspace=0.08, top=0.93, bottom=0.03,
                            left=0.09, right=0.97)
    ax_fit = fig.add_subplot(gs[0])
    ax_res = fig.add_subplot(gs[1], sharex=ax_fit)
    ax_tbl = fig.add_subplot(gs[2])
    ax_tbl.axis("off")

    # ---- Top panel -------------------------------------------------------
    ax_fit.errorbar(r2_data, profile, yerr=sigma, fmt="none",
                    ecolor="darkorange", elinewidth=0.5, alpha=0.35, zorder=2)
    ax_fit.plot(r2_data, profile, color="darkorange", lw=0.9, alpha=0.85,
                zorder=3, label=f"Data  ({source_name})")
    ax_fit.plot(r2_fine, model_fine, color="black", lw=1.5,
                zorder=4, label="Best-fit composite  (12-param Stage 4)")
    ax_fit.plot(r2_fine, lam1_fine, color="steelblue", lw=0.8,
                ls="--", alpha=0.65, label=f"λ₁ 640.2 nm  (R1={fit.R1:.3f})")
    ax_fit.plot(r2_fine, lam2_fine, color="firebrick", lw=0.8,
                ls="--", alpha=0.65,
                label=f"λ₂ 638.3 nm ×{fit.ne_ratio:.3f}  (R2={fit.R2:.3f})")
    ax_fit.set_ylabel("CCD signal  (ADU)", fontsize=11)
    ax_fit.legend(fontsize=8.5, loc="upper right")
    ax_fit.grid(True, alpha=0.2)
    ax_fit.tick_params(labelbottom=False)

    conv_str = "converged" if fit.converged else "NOT converged"
    R_diff   = fit.R2 - fit.R1
    ax_fit.text(
        0.02, 0.97,
        f"χ²/ν = {fit.chi2_reduced:.3f}   {conv_str}\n"
        f"R1 = {fit.R1:.4f} ± {fit.sigma_R1:.4f}   "
        f"R2 = {fit.R2:.4f} ± {fit.sigma_R2:.4f}   "
        f"ΔR = R2−R1 = {R_diff:+.4f}\n"
        f"ne_ratio = {fit.ne_ratio:.4f} ± {fit.sigma_ne_ratio:.4f}   "
        f"[nominal = {NE_INTENSITY_2_NOMINAL:.2f}]",
        transform=ax_fit.transAxes, va="top", ha="left", fontsize=8.5,
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="grey", alpha=0.88))

    # ---- Residual panel --------------------------------------------------
    ax_res.axhline(0, color="black", lw=0.8, ls="--")
    ax_res.fill_between(r2_data, -sigma, sigma,
                        color="steelblue", alpha=0.22, label="±1σ")
    ax_res.plot(r2_data, residual, color="steelblue", lw=0.8,
                alpha=0.9, label="Residual (data − model)")
    ax_res.set_xlabel(r"$r^2$  (pixels², 2×2 binned)", fontsize=11)
    ax_res.set_ylabel("Residual  (ADU)", fontsize=10)
    ax_res.legend(fontsize=8.5, loc="upper right")
    ax_res.grid(True, alpha=0.2)
    ax_res.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, symmetric=True))

    # ---- Parameter table -------------------------------------------------
    def _fmt_sig(v, sv):
        """Format value ± stderr; use µm for t_m."""
        return v, sv

    rows = [
        ("t",         f"{fit.t_m*1e3:.7f} mm",   f"±{fit.sigma_t_m*1e6:.3f} µm",    "Etalon gap  [Tolansky-seeded, Group A]"),
        ("α",         f"{fit.alpha:.5e} rad/px",  f"±{fit.sigma_alpha:.2e}",          "Plate scale  [Tolansky-seeded, Group A]"),
        ("R1",        f"{fit.R1:.5f}",             f"±{fit.sigma_R1:.5f}",             "Reflectivity λ₁ = 640.2 nm  [Group B]"),
        ("R2",        f"{fit.R2:.5f}",             f"±{fit.sigma_R2:.5f}",             "Reflectivity λ₂ = 638.3 nm  [Group B, NEW]"),
        ("ΔR=R2−R1",  f"{fit.R2-fit.R1:+.5f}",   "—",                                "Wavelength-dependent finesse difference"),
        ("I₀",        f"{fit.I0:.1f} ADU",         f"±{fit.sigma_I0:.1f}",             "Mean intensity  [Group B, shared]"),
        ("I₁",        f"{fit.I1:.4f}",             f"±{fit.sigma_I1:.4f}",             "Linear vignetting  [Group B, shared]"),
        ("I₂",        f"{fit.I2:.5f}",             f"±{fit.sigma_I2:.5f}",             "Quadratic vignetting  [Group B, shared]"),
        ("σ₀",        f"{fit.sigma0:.4f} px",      f"±{fit.sigma_sigma0:.4f}",         "PSF base width  [Group B, shared]"),
        ("σ₁",        f"{fit.sigma1:.4f} px",      f"±{fit.sigma_sigma1:.4f}",         "PSF sin variation  [Group B, shared]"),
        ("σ₂",        f"{fit.sigma2:.4f} px",      f"±{fit.sigma_sigma2:.4f}",         "PSF cos variation  [Group B, shared]"),
        ("B",         f"{fit.B:.1f} ADU",           f"±{fit.sigma_B:.2f}",              "CCD bias pedestal  [Group B]"),
        ("ne_ratio",  f"{fit.ne_ratio:.4f}",        f"±{fit.sigma_ne_ratio:.4f}",       f"λ₂/λ₁ intensity ratio  [Group B, nominal={NE_INTENSITY_2_NOMINAL:.2f}]"),
        ("ε_cal",     f"{fit.epsilon_cal:.6f}",     f"±{fit.sigma_epsilon_cal:.6f}",    "Fractional order at centre  (zero-wind reference)"),
    ]

    tbl = ax_tbl.table(
        cellText=rows,
        colLabels=["Param", "Fitted value", "1σ", "Description"],
        cellLoc="left", loc="upper center",
        colWidths=[0.10, 0.20, 0.14, 0.56])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.2)
    tbl.scale(1, 1.24)

    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#c8d8f0")
            cell.set_text_props(fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f0f4ff")
        if row in (3, 4):   # R2 and ΔR rows — highlight
            cell.set_facecolor("#fff4c2")
        if row == 13:       # ne_ratio row
            cell.set_facecolor("#f4fff4")

    stage_str = "  ".join(
        f"S{i+1}: {v:.2f}" for i, v in enumerate(fit.chi2_by_stage))
    ax_tbl.text(0.01, 0.01,
                f"χ²/ν by stage:  {stage_str}    "
                f"bins used: {fit.n_bins_used}   free params: 12",
                transform=ax_tbl.transAxes, va="bottom", ha="left",
                fontsize=8.5, fontfamily="monospace", color="dimgrey")

    fig.suptitle(
        "WindCube FPI — Neon Calibration Fringe Inversion  "
        "(12-param: independent R1, R2 / Harding 2014)",
        fontsize=12, fontweight="bold", y=0.975)

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

    R1_init = simpledialog.askfloat("R1 initial guess",
        "Starting guess for λ₁ (640.2 nm) effective reflectivity.\n"
        "FlatSat measured: 0.53",
        initialvalue=0.53, minvalue=0.05, maxvalue=0.95) or 0.53

    R2_init = simpledialog.askfloat("R2 initial guess",
        "Starting guess for λ₂ (638.3 nm) effective reflectivity.\n"
        "Try 0.53 first — fit will find the true value.\n"
        "Expect R2 > R1 if λ₂ peaks are sharper than λ₁.",
        initialvalue=0.53, minvalue=0.05, maxvalue=0.95) or 0.53

    ne_ratio_init = simpledialog.askfloat("ne_ratio initial guess",
        f"Starting guess for λ₂/λ₁ intensity ratio.\n"
        f"Nominal = {NE_INTENSITY_2_NOMINAL:.2f}",
        initialvalue=float(NE_INTENSITY_2_NOMINAL),
        minvalue=0.01, maxvalue=2.0) or float(NE_INTENSITY_2_NOMINAL)

    _tk2.destroy()

    t_tolansky_m = t_tolansky_mm * 1e-3

    # 3. Phase-correct the gap
    t_eff = phase_correct_gap(t_tolansky_m, eps_a, NE_WAVELENGTH_1_AIR_M)
    print(f"\nphase_correct_gap:  t_tolansky = {t_tolansky_m*1e3:.7f} mm  "
          f"→  t_eff = {t_eff*1e3:.7f} mm  "
          f"(correction = {(t_eff - t_tolansky_m)*1e9:.1f} nm)")

    # 4. Load and prepare profile
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
    print(f"\nRunning 12-parameter staged LM inversion  "
          f"(R1 init={R1_init:.3f}, R2 init={R2_init:.3f})…")
    fit = run_staged_inversion(
        fp, t_eff, alpha_tolansky, eps_a,
        R1_init=R1_init, R2_init=R2_init,
        ne_ratio_init=ne_ratio_init)

    # 6. Print results
    print(f"\n{'='*68}")
    print("12-PARAMETER CALIBRATION INVERSION RESULT")
    print(f"{'='*68}")
    print(f"  Converged:      {fit.converged}")
    print(f"  χ²/ν:           {fit.chi2_reduced:.4f}")
    print(f"  χ²/ν by stage:  {[f'{v:.3f}' for v in fit.chi2_by_stage]}")
    print(f"  Bins used:      {fit.n_bins_used}")
    print()
    print(f"  --- Group A (Tolansky-seeded) ---")
    print(f"  t      = {fit.t_m*1e3:.7f} mm  ±{fit.sigma_t_m*1e6:.3f} µm")
    print(f"  alpha  = {fit.alpha:.5e}  ±{fit.sigma_alpha:.2e} rad/px")
    print()
    print(f"  --- Group B (fringe shape) ---")
    print(f"  R1       = {fit.R1:.5f}  ±{fit.sigma_R1:.5f}   [λ₁ = 640.2 nm]")
    print(f"  R2       = {fit.R2:.5f}  ±{fit.sigma_R2:.5f}   [λ₂ = 638.3 nm]")
    print(f"  ΔR=R2−R1 = {fit.R2-fit.R1:+.5f}   "
          f"({'R2>R1: λ₂ sharper as expected' if fit.R2>fit.R1 else 'R2<R1: λ₂ broader'})")
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
    print(f"{'='*68}")

    # 7. Build fine-grid model components for plotting
    r_fine = np.linspace(0.0, r_max_px, 2000)
    model_fine, lam1_fine, lam2_fine = _components_12(
        r_fine, r_max_px,
        fit.t_m, fit.alpha, fit.R1, fit.R2,
        fit.I0, fit.I1, fit.I2,
        fit.sigma0, fit.sigma1, fit.sigma2,
        fit.B, fit.ne_ratio)

    # 8. Plot
    fig = make_figure(
        r2_data=r_grid**2, profile=profile_adu, sigma=sigma_adu,
        r_fine=r_fine, model_fine=model_fine,
        lam1_fine=lam1_fine, lam2_fine=lam2_fine,
        fit=fit, source_name=npy_path.name)
    plt.show()


if __name__ == "__main__":
    main()
