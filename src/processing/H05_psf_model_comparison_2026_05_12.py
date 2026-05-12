"""
Script:  H05_psf_model_comparison_2026_05_12.py
Purpose: Investigate PSF parameter degeneracy in the neon calibration fringe
         inversion by fitting three nested PSF models and comparing results.

         The three models differ only in which PSF width parameters are free:

           Model A: σ(r) = σ₀ + σ₁·sin(πr/r_max)    [2 PSF params — current]
                    (σ₂ was already fixed at 0 in H05; Model A = current model)

           Model B: σ(r) = σ₀                          [1 PSF param — constant blur]
                    (σ₁ also fixed at 0)

         All other parameters (t, alpha, R1, R2, I0, I1, I2, B, ne_ratio)
         are free in both models, using the same Stage 1-4 sequence as H05.

         Outputs:
           1. Side-by-side terminal table: fitted values, 1σ, χ²/ν for A and B
           2. Three-panel comparison figure:
                Top:    Model A residual vs r²
                Middle: Model B residual vs r²
                Bottom: Overlay of both model curves vs data (vs r, equal spacing)
           3. F-test and Δχ²/ν to guide the model choice decision

         Decision rule used here:
           If Δχ²/ν = χ²_B − χ²_A < 0.05 AND residuals look similar,
           → Model B (σ₀ only) is preferred: simpler, more honest.
           Otherwise keep Model A.

Run from repo root:
    python src/processing/H05_psf_model_comparison_2026_05_12.py
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
from scipy.stats import f as f_dist

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
log = logging.getLogger("H05_psf")

# ---------------------------------------------------------------------------
# Parameter ordering (12 total — same as H05)
# ---------------------------------------------------------------------------
_NAMES = ['t_m', 'alpha', 'R1', 'R2', 'I0', 'I1', 'I2',
          'sigma0', 'sigma1', 'sigma2', 'B', 'ne_ratio']
_IDX   = {n: i for i, n in enumerate(_NAMES)}

# Model definitions: which PSF parameters are fixed at 0
# sigma2 is always fixed (unidentifiable — established in H05)
# Model A: sigma1 free,  sigma2 fixed  ← current H05 model
# Model B: sigma1 fixed, sigma2 fixed  ← simpler constant-blur model
_MODELS = {
    'A': dict(label='Model A: σ₀ + σ₁·sin  (current)',
              fix_sigma1=False, fix_sigma2=True),
    'B': dict(label='Model B: σ₀ only  (constant blur)',
              fix_sigma1=True,  fix_sigma2=True),
}


def _stage_free(fix_sigma1, fix_sigma2):
    """Build _STAGE_FREE dict for this PSF model."""
    always_fixed = []
    if fix_sigma1:
        always_fixed.append('sigma1')
    if fix_sigma2:
        always_fixed.append('sigma2')

    def _free(base):
        return [n for n in base if n not in always_fixed]

    return {
        1: _free(['I0', 'I1', 'I2', 'B']),
        2: _free(['t_m', 'alpha', 'R1', 'R2', 'I0', 'I1', 'I2', 'B']),
        3: _free(['t_m', 'alpha', 'R1', 'R2', 'I0', 'I1', 'I2', 'sigma0', 'B']),
        4: _free([n for n in _NAMES if n != 'sigma2']),
    }


# ---------------------------------------------------------------------------
# Forward model (same as H05)
# ---------------------------------------------------------------------------

def _neon_model(r_arr, r_max, t, alpha, R1, R2, I0, I1, I2,
                sigma0, sigma1, sigma2, B, ne_ratio, _N_fine=500):
    r_fine = np.linspace(0.0, r_max, _N_fine)
    A1 = airy_modified(r_fine, NE_WAVELENGTH_1_AIR_M,
                       t, R1, alpha, 1.0, r_max,
                       I0, I1, I2, sigma0, sigma1, sigma2)
    A2 = airy_modified(r_fine, NE_WAVELENGTH_2_AIR_M,
                       t, R2, alpha, 1.0, r_max,
                       I0, I1, I2, sigma0, sigma1, sigma2)
    return np.interp(r_arr, r_fine, A1 + ne_ratio * A2 + B)


def _model_components(r_arr, r_max, t, alpha, R1, R2, I0, I1, I2,
                      sigma0, sigma1, sigma2, B, ne_ratio, n_fine=2000):
    r_fine = np.linspace(0.0, r_max, n_fine)
    A1 = airy_modified(r_fine, NE_WAVELENGTH_1_AIR_M,
                       t, R1, alpha, 1.0, r_max,
                       I0, I1, I2, sigma0, sigma1, sigma2)
    A2 = airy_modified(r_fine, NE_WAVELENGTH_2_AIR_M,
                       t, R2, alpha, 1.0, r_max,
                       I0, I1, I2, sigma0, sigma1, sigma2)
    comp = np.interp(r_arr, r_fine, A1 + ne_ratio * A2 + B)
    lam1 = np.interp(r_arr, r_fine, A1 + B)
    lam2 = np.interp(r_arr, r_fine, ne_ratio * A2 + B)
    return comp, lam1, lam2


# ---------------------------------------------------------------------------
# Finite-difference Jacobian (same as H05)
# ---------------------------------------------------------------------------

def _fd_jacobian(residual_fn, p0, rel_step=1e-5):
    r0  = residual_fn(p0)
    n_r = len(r0)
    n_p = len(p0)
    J   = np.zeros((n_r, n_p))
    for j in range(n_p):
        h      = max(rel_step * abs(p0[j]), 1e-10)
        p_fwd  = p0.copy(); p_fwd[j] += h
        J[:, j] = (residual_fn(p_fwd) - r0) / h
    return J


# ---------------------------------------------------------------------------
# Stage runner (same as H05, parameterised by stage_free dict)
# ---------------------------------------------------------------------------

def _run_stage(r_good, prof_good, sig_good, r_max,
               p_all, free_names, bounds_dict, config):
    free_idx  = np.array([_IDX[n] for n in free_names])
    p_fixed   = p_all.copy()
    p0        = p_fixed[free_idx]
    n_good    = len(r_good)
    n_free    = len(free_names)

    lo_arr    = np.array([bounds_dict[n][1] for n in free_names])
    hi_arr    = np.array([bounds_dict[n][2] for n in free_names])
    range_arr = hi_arr - lo_arr + 1e-30
    pen_sigma = range_arr * 0.01

    def _residuals_penalised(p_free):
        p = p_fixed.copy()
        p[free_idx] = p_free
        t, alpha, R1, R2, I0, I1, I2, sigma0, sigma1, sigma2, B, ne_ratio = p
        model  = _neon_model(r_good, r_max, t, alpha, R1, R2,
                             I0, I1, I2, sigma0, sigma1, sigma2, B, ne_ratio)
        data_r = (prof_good - model) / sig_good
        below  = np.maximum(0.0, lo_arr - p_free) / pen_sigma
        above  = np.maximum(0.0, p_free - hi_arr) / pen_sigma
        return np.append(data_r, below + above)

    lm = least_squares(_residuals_penalised, p0, method='lm',
                       ftol=config['ftol'], xtol=config['xtol'],
                       gtol=config['gtol'], max_nfev=config['max_nfev'])

    p_updated = p_fixed.copy()
    p_updated[free_idx] = lm.x

    t, alpha, R1, R2, I0, I1, I2, sigma0, sigma1, sigma2, B, ne_ratio = p_updated
    model_f = _neon_model(r_good, r_max, t, alpha, R1, R2,
                          I0, I1, I2, sigma0, sigma1, sigma2, B, ne_ratio)
    data_r  = (prof_good - model_f) / sig_good
    dof     = max(n_good - n_free, 1)
    chi2    = float(np.sum(data_r ** 2)) / dof

    def _res_data_only(p_free):
        p = p_fixed.copy()
        p[free_idx] = p_free
        t, alpha, R1, R2, I0, I1, I2, sigma0, sigma1, sigma2, B, ne_ratio = p
        model = _neon_model(r_good, r_max, t, alpha, R1, R2,
                            I0, I1, I2, sigma0, sigma1, sigma2, B, ne_ratio)
        return (prof_good - model) / sig_good

    try:
        J       = _fd_jacobian(_res_data_only, lm.x)
        _, s, VT = np.linalg.svd(J, full_matrices=False)
        threshold = np.finfo(float).eps * max(J.shape) * s[0]
        s_inv    = np.where(s > threshold, 1.0 / s, 0.0)
        JTJ_inv  = (VT.T * s_inv**2) @ VT
        cov      = chi2 * JTJ_inv
        stderrs  = np.sqrt(np.maximum(np.diag(cov), 0.0))
        if not np.all(np.isfinite(stderrs)):
            raise np.linalg.LinAlgError("non-finite stderrs")
    except (np.linalg.LinAlgError, ValueError) as exc:
        log.warning(f"  Covariance failed: {exc}")
        stderrs = np.full(n_free, np.inf)

    return p_updated, stderrs, chi2, lm


# ---------------------------------------------------------------------------
# Full staged inversion for one PSF model
# ---------------------------------------------------------------------------

@dataclass
class ModelResult:
    label:       str
    p_all:       np.ndarray     # full 12-element parameter vector
    stderrs:     dict           # name -> stderr (NaN if fixed)
    chi2:        float
    chi2_stages: list
    n_free:      int            # number of free params in Stage 4
    n_bins:      int
    converged:   bool
    fix_sigma1:  bool
    fix_sigma2:  bool


def run_model(r_good, prof_good, s_good, r_max, bounds, cfg,
              fix_sigma1, fix_sigma2, label):

    stage_free = _stage_free(fix_sigma1, fix_sigma2)
    p_all      = np.array([bounds[n][0] for n in _NAMES], dtype=float)
    chi2_stages = []

    for stage_num in [1, 2, 3, 4]:
        free = stage_free[stage_num]
        if not free:
            log.info(f"  Stage {stage_num} — no free params, skipping")
            chi2_stages.append(chi2_stages[-1] if chi2_stages else 999.0)
            continue
        p_all, se, chi2, lm = _run_stage(
            r_good, prof_good, s_good, r_max, p_all, free, bounds, cfg)
        chi2_stages.append(chi2)
        log.info(f"  Stage {stage_num}  χ²/ν = {chi2:.4f}   "
                 f"free: {free}")

    # Build stderr dict indexed by name
    free4  = stage_free[4]
    se4, _ = _run_stage_stderrs_only(
        r_good, prof_good, s_good, r_max, p_all, free4, bounds, cfg)
    stderrs_dict = {}
    for i, name in enumerate(free4):
        stderrs_dict[name] = float(se4[i])
    for name in _NAMES:
        if name not in stderrs_dict:
            stderrs_dict[name] = float('nan')

    converged = True   # if we got here without exception
    n_free    = len(free4)

    return ModelResult(
        label=label, p_all=p_all, stderrs=stderrs_dict,
        chi2=chi2_stages[-1], chi2_stages=chi2_stages,
        n_free=n_free, n_bins=len(r_good),
        converged=converged,
        fix_sigma1=fix_sigma1, fix_sigma2=fix_sigma2,
    )


def _run_stage_stderrs_only(r_good, prof_good, s_good, r_max,
                             p_all, free_names, bounds_dict, config):
    """Re-run Stage 4 from current p_all to get covariance at final solution."""
    free_idx = np.array([_IDX[n] for n in free_names])
    p_fixed  = p_all.copy()
    n_good   = len(r_good)
    n_free   = len(free_names)

    lo_arr   = np.array([bounds_dict[n][1] for n in free_names])
    hi_arr   = np.array([bounds_dict[n][2] for n in free_names])
    range_arr = hi_arr - lo_arr + 1e-30
    pen_sigma = range_arr * 0.01

    def _residuals_pen(p_free):
        p = p_fixed.copy(); p[free_idx] = p_free
        t, alpha, R1, R2, I0, I1, I2, s0, s1, s2, B, ne = p
        model  = _neon_model(r_good, r_max, t, alpha, R1, R2,
                             I0, I1, I2, s0, s1, s2, B, ne)
        data_r = (prof_good - model) / s_good
        below  = np.maximum(0.0, lo_arr - p_free) / pen_sigma
        above  = np.maximum(0.0, p_free - hi_arr) / pen_sigma
        return np.append(data_r, below + above)

    lm = least_squares(_residuals_pen, p_fixed[free_idx], method='lm',
                       ftol=config['ftol'], xtol=config['xtol'],
                       gtol=config['gtol'], max_nfev=config['max_nfev'])

    p_updated = p_fixed.copy(); p_updated[free_idx] = lm.x

    t, alpha, R1, R2, I0, I1, I2, s0, s1, s2, B, ne = p_updated
    model_f = _neon_model(r_good, r_max, t, alpha, R1, R2,
                          I0, I1, I2, s0, s1, s2, B, ne)
    data_r  = (prof_good - model_f) / s_good
    dof     = max(n_good - n_free, 1)
    chi2    = float(np.sum(data_r ** 2)) / dof

    def _res_data_only(p_free):
        p = p_fixed.copy(); p[free_idx] = p_free
        t, alpha, R1, R2, I0, I1, I2, s0, s1, s2, B, ne = p
        model = _neon_model(r_good, r_max, t, alpha, R1, R2,
                            I0, I1, I2, s0, s1, s2, B, ne)
        return (prof_good - model) / s_good

    try:
        J = _fd_jacobian(_res_data_only, lm.x)
        _, s, VT = np.linalg.svd(J, full_matrices=False)
        threshold = np.finfo(float).eps * max(J.shape) * s[0]
        s_inv  = np.where(s > threshold, 1.0 / s, 0.0)
        JTJ_inv = (VT.T * s_inv**2) @ VT
        cov     = chi2 * JTJ_inv
        se      = np.sqrt(np.maximum(np.diag(cov), 0.0))
        if not np.all(np.isfinite(se)):
            raise np.linalg.LinAlgError("non-finite")
    except (np.linalg.LinAlgError, ValueError):
        se = np.full(n_free, np.inf)

    return se, chi2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
        raise ValueError(f"Unexpected shape {arr.shape}")
    return np.sqrt(np.maximum(r2, 0.0)), prof, sigma


def _estimate_sigma(profile):
    return np.maximum(np.sqrt(np.maximum(profile, 1.0)), 1.0)


def _pval(name, result):
    """Return (value_str, stderr_str) for a parameter."""
    v  = result.p_all[_IDX[name]]
    se = result.stderrs.get(name, float('nan'))
    if np.isnan(se):
        se_str = "  fixed "
    else:
        se_str = f"±{se:.2e}"
    return v, se_str


# ---------------------------------------------------------------------------
# Figure: three-panel comparison
# ---------------------------------------------------------------------------

def make_comparison_figure(r_grid, profile, sigma,
                           results: list,
                           r_max_px: float,
                           source_name: str = "") -> plt.Figure:

    r2_data = r_grid ** 2
    r_fine  = np.linspace(0.0, r_max_px, 2000)
    r2_fine = r_fine ** 2

    fig = plt.figure(figsize=(15, 12))
    gs  = gridspec.GridSpec(3, 2,
                            height_ratios=[2.5, 2.5, 3],
                            hspace=0.35, wspace=0.30,
                            top=0.93, bottom=0.04,
                            left=0.08, right=0.97)

    colors = ['steelblue', 'firebrick']

    # ---- Top two panels: residuals for each model ----------------------
    for col, res in enumerate(results):
        ax = fig.add_subplot(gs[col // 2 * 0 + col, col % 2])
        # This layout puts A top-left, B top-right, overlay bottom spanning

        comp, _, _ = _model_components(
            r_grid, r_max_px, *res.p_all, n_fine=2000)
        # unpack p_all
        t, alpha, R1, R2, I0, I1, I2, s0, s1, s2, B, ne = res.p_all
        comp_fine, lam1, lam2 = _model_components(
            r_fine, r_max_px, t, alpha, R1, R2, I0, I1, I2, s0, s1, s2, B, ne)

        model_at_data = np.interp(r2_data, r2_fine, comp_fine)
        residual      = profile - model_at_data

        ax.axhline(0, color='black', lw=0.8, ls='--')
        ax.fill_between(r2_data, -sigma, sigma,
                        color='grey', alpha=0.2, label='±1σ')
        ax.plot(r2_data, residual, color=colors[col], lw=0.8,
                alpha=0.85, label='Residual')
        ax.set_xlabel(r'$r^2$ (px²)', fontsize=9)
        ax.set_ylabel('Residual (ADU)', fontsize=9)
        ax.set_title(
            f'{res.label}\n'
            f'χ²/ν = {res.chi2:.4f}   n_free = {res.n_free}   '
            f'σ₀={res.p_all[_IDX["sigma0"]]:.3f}  '
            f'σ₁={res.p_all[_IDX["sigma1"]]:.3f} '
            f'({"free" if not res.fix_sigma1 else "fixed=0"})',
            fontsize=8.5)
        ax.legend(fontsize=7.5)
        ax.grid(True, alpha=0.2)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, symmetric=True))

    # ---- Bottom panel: overlay both model curves vs data (vs r) --------
    ax_ov = fig.add_subplot(gs[2, :])

    ax_ov.plot(r_grid, profile, color='darkorange', lw=0.8, alpha=0.7,
               label=f'Data  ({source_name})', zorder=2)

    for col, res in enumerate(results):
        t, alpha, R1, R2, I0, I1, I2, s0, s1, s2, B, ne = res.p_all
        comp_fine, _, _ = _model_components(
            r_fine, r_max_px, t, alpha, R1, R2, I0, I1, I2, s0, s1, s2, B, ne)
        lw   = 1.8 if col == 0 else 1.2
        ls   = '-' if col == 0 else '--'
        ax_ov.plot(r_fine, comp_fine, color=colors[col], lw=lw, ls=ls,
                   alpha=0.85, zorder=3 + col,
                   label=f'{res.label}  χ²/ν={res.chi2:.4f}')

    ax_ov.set_xlabel('Radius r  (pixels)', fontsize=10)
    ax_ov.set_ylabel('CCD signal  (ADU)', fontsize=10)
    ax_ov.set_title('Model overlay vs data  (vs r, equal spacing)', fontsize=10)
    ax_ov.legend(fontsize=8.5, loc='upper right')
    ax_ov.grid(True, alpha=0.2)

    fig.suptitle(
        'WindCube FPI — PSF Model Comparison  '
        '(nested F-test: is σ₁ identifiable?)',
        fontsize=12, fontweight='bold', y=0.975)

    return fig


# ---------------------------------------------------------------------------
# Print comparison table
# ---------------------------------------------------------------------------

def print_comparison(results: list):
    A, B = results[0], results[1]

    # F-test: does removing σ₁ significantly worsen the fit?
    # RSS = chi2_red * dof * sigma²  — but since we're using normalised
    # residuals (divided by sigma), RSS_normalised = chi2_red * dof
    dof_A  = A.n_bins - A.n_free
    dof_B  = B.n_bins - B.n_free
    rss_A  = A.chi2 * dof_A
    rss_B  = B.chi2 * dof_B
    # Number of extra constraints going from A to B
    delta_p = A.n_free - B.n_free   # should be 1 (sigma1)
    if delta_p > 0 and rss_A > 0:
        F_stat = ((rss_B - rss_A) / delta_p) / (rss_A / dof_A)
        p_val  = 1.0 - f_dist.cdf(F_stat, delta_p, dof_A)
    else:
        F_stat, p_val = float('nan'), float('nan')

    delta_chi2 = B.chi2 - A.chi2

    print()
    print("=" * 72)
    print("PSF MODEL COMPARISON — NESTED F-TEST")
    print("=" * 72)
    print(f"  Model A  ({A.label}):  χ²/ν = {A.chi2:.4f}   n_free = {A.n_free}")
    print(f"  Model B  ({B.label}):  χ²/ν = {B.chi2:.4f}   n_free = {B.n_free}")
    print(f"  Δχ²/ν = χ²_B − χ²_A  = {delta_chi2:+.4f}")
    print(f"  F statistic           = {F_stat:.3f}  (df1={delta_p}, df2={dof_A})")
    print(f"  p-value               = {p_val:.4f}")
    print()
    if p_val < 0.05:
        print("  DECISION: σ₁ is statistically significant (p < 0.05).")
        print("            Keep Model A (σ₀ + σ₁·sin).")
    else:
        print("  DECISION: σ₁ is NOT statistically significant (p ≥ 0.05).")
        print("            Prefer Model B (σ₀ only) — simpler and equally good.")
    print()

    # Side-by-side parameter table
    print("-" * 72)
    print(f"{'Param':<12} {'Model A value':>18} {'±1σ':>12}   "
          f"{'Model B value':>18} {'±1σ':>12}   {'Δ (B−A)':>12}")
    print("-" * 72)

    param_display = [
        ('t_m',      lambda v: f"{v*1e3:.7f} mm",  "etalon gap"),
        ('alpha',    lambda v: f"{v:.5e}",           "plate scale"),
        ('R1',       lambda v: f"{v:.5f}",           "reflect. λ₁"),
        ('R2',       lambda v: f"{v:.5f}",           "reflect. λ₂"),
        ('I0',       lambda v: f"{v:.1f}",           "intensity"),
        ('I1',       lambda v: f"{v:.5f}",           "vignette lin"),
        ('I2',       lambda v: f"{v:.5f}",           "vignette quad"),
        ('sigma0',   lambda v: f"{v:.4f} px",        "PSF σ₀"),
        ('sigma1',   lambda v: f"{v:.4f} px",        "PSF σ₁"),
        ('sigma2',   lambda v: f"{v:.4f} px",        "PSF σ₂"),
        ('B',        lambda v: f"{v:.1f}",           "bias"),
        ('ne_ratio', lambda v: f"{v:.4f}",           "ne_ratio"),
    ]

    for name, fmt, desc in param_display:
        v_A  = A.p_all[_IDX[name]]
        v_B  = B.p_all[_IDX[name]]
        se_A = A.stderrs.get(name, float('nan'))
        se_B = B.stderrs.get(name, float('nan'))
        se_A_str = f"±{se_A:.2e}" if not np.isnan(se_A) else "  fixed "
        se_B_str = f"±{se_B:.2e}" if not np.isnan(se_B) else "  fixed "
        delta    = v_B - v_A
        # Express delta in units of sigma_A if available
        if not np.isnan(se_A) and se_A > 0:
            delta_sig = delta / se_A
            delta_str = f"{delta_sig:+.1f}σ"
        else:
            delta_str = "  n/a  "
        print(f"  {desc:<14} {fmt(v_A):>18} {se_A_str:>12}   "
              f"{fmt(v_B):>18} {se_B_str:>12}   {delta_str:>10}")
    print("=" * 72)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    # 1. File picker
    _tk = tk.Tk(); _tk.withdraw()
    npy_path = filedialog.askopenfilename(
        title="Select neon calibration profile (.npy, same file as H05)",
        filetypes=[("NumPy arrays", "*.npy"), ("All files", "*.*")])
    _tk.destroy()
    if not npy_path:
        print("No file selected — exiting."); return
    npy_path = pathlib.Path(npy_path)
    print(f"\nLoaded: {npy_path.name}")

    # 2. Parameter dialogs (same defaults as H05)
    _tk2 = tk.Tk(); _tk2.withdraw()

    r_max_px = simpledialog.askfloat("r_max",
        "Maximum usable fringe radius (px).\nFlatSat/flight: 110",
        initialvalue=110.0, minvalue=50.0, maxvalue=200.0) or 110.0

    t_tolansky_mm = simpledialog.askfloat("Tolansky gap",
        "Tolansky-recovered etalon gap (mm).\nExample: 20.1070707",
        initialvalue=20.1070707, minvalue=19.5, maxvalue=20.5) or 20.1070707

    eps_a = simpledialog.askfloat("Tolansky ε_a",
        "Tolansky excess fraction ε_a.\nExample: 0.23286",
        initialvalue=0.23286, minvalue=0.0, maxvalue=0.9999) or 0.23286

    alpha_tolansky = simpledialog.askfloat("Tolansky alpha",
        "Tolansky plate scale α (rad/px).\nExample: 1.6084e-4",
        initialvalue=1.6084e-4, minvalue=1e-5, maxvalue=1e-3) or 1.6084e-4

    R1_init = simpledialog.askfloat("R1 init",
        "Starting guess for R1 (λ₁ reflectivity).",
        initialvalue=0.53, minvalue=0.05, maxvalue=0.95) or 0.53

    R2_init = simpledialog.askfloat("R2 init",
        "Starting guess for R2 (λ₂ reflectivity).",
        initialvalue=0.53, minvalue=0.05, maxvalue=0.95) or 0.53

    ne_ratio_init = simpledialog.askfloat("ne_ratio init",
        f"Starting guess for λ₂/λ₁ ratio.\nNominal={NE_INTENSITY_2_NOMINAL:.2f}",
        initialvalue=float(NE_INTENSITY_2_NOMINAL),
        minvalue=0.01, maxvalue=2.0) or float(NE_INTENSITY_2_NOMINAL)

    _tk2.destroy()

    # 3. Phase-correct the gap
    t_eff = phase_correct_gap(t_tolansky_mm * 1e-3, eps_a, NE_WAVELENGTH_1_AIR_M)
    print(f"t_eff = {t_eff*1e3:.7f} mm  "
          f"(correction = {(t_eff - t_tolansky_mm*1e-3)*1e9:.1f} nm)")

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
    else:
        sigma_adu = _estimate_sigma(profile_adu)

    s_good = np.maximum(sigma_adu, 1.0)
    print(f"  {len(r_grid)} bins, signal ∈ "
          f"[{profile_adu.min():.0f}, {profile_adu.max():.0f}] ADU")

    # 5. Build shared bounds
    I0_init = float(np.percentile(profile_adu, 75))
    B_init  = float(np.percentile(profile_adu, 5)) * 0.8

    bounds = {
        't_m':      (t_eff,          t_eff - 20e-6,      t_eff + 20e-6),
        'alpha':    (alpha_tolansky, alpha_tolansky*0.95, alpha_tolansky*1.05),
        'R1':       (R1_init,        0.05,               0.95),
        'R2':       (R2_init,        0.05,               0.95),
        'I0':       (I0_init,        100.0,              15000.0),
        'I1':       (0.0,            -0.5,               0.5),
        'I2':       (0.0,            -0.5,               0.5),
        'sigma0':   (0.5,            0.01,               5.0),
        'sigma1':   (0.0,            -2.0,               2.0),
        'sigma2':   (0.0,            -2.0,               2.0),
        'B':        (B_init,         10.0,               2000.0),
        'ne_ratio': (ne_ratio_init,  0.01,               2.0),
    }
    cfg = dict(max_nfev=100_000, ftol=1e-14, xtol=1e-14, gtol=1e-14)

    # 6. Fit both models
    results = []
    for mkey, mdef in _MODELS.items():
        print(f"\n{'─'*60}")
        print(f"Fitting {mdef['label']} ...")
        print(f"{'─'*60}")
        res = run_model(
            r_grid, profile_adu, s_good, r_max_px,
            bounds, cfg,
            fix_sigma1=mdef['fix_sigma1'],
            fix_sigma2=mdef['fix_sigma2'],
            label=mdef['label'])
        results.append(res)
        print(f"  → χ²/ν = {res.chi2:.4f}   n_free = {res.n_free}")

    # 7. Print comparison table + F-test
    print_comparison(results)

    # 8. Figure
    fig = make_comparison_figure(
        r_grid, profile_adu, sigma_adu,
        results, r_max_px, source_name=npy_path.name)
    plt.show()


if __name__ == "__main__":
    main()
