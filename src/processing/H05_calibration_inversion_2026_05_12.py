"""
Script:  H05_calibration_inversion_2026_05_12.py
Purpose: Load a real two-line neon calibration radial profile (tabulated vs r²),
         run a 10-free-parameter staged Levenberg-Marquardt calibration inversion,
         and produce a diagnostic figure modelled on Harding et al. (2014) Fig. 4.

═══════════════════════════════════════════════════════════════════════════════
 PHYSICAL MODEL SUMMARY
═══════════════════════════════════════════════════════════════════════════════

 Forward model:
   S(r) = Ã(r; λ₁, t, α, R1) ⊛ G(σ(r))
        + ne_ratio · Ã(r; λ₂, t, α, R2) ⊛ G(σ(r))
        + B

 Intensity envelope (Harding Eq. 4):
   I(r) = I₀ [1 + I₁(r/r_max) + I₂(r/r_max)²]

 PSF width — constant blur (σ₁ = σ₂ = 0, see investigation below):
   σ(r) = σ₀

 Free parameters (10 total):
   Group A — Tolansky-seeded, tight ±20 µm / ±5% bounds:
     t       — phase-corrected etalon gap (H01 §8)
     alpha   — plate scale (rad/px)
   Group B — freely fitted from fringe shape:
     R1      — effective reflectivity for λ₁ = 640.2248 nm
     R2      — effective reflectivity for λ₂ = 638.2991 nm
     I0, I1, I2   — shared intensity envelope
     sigma0  — PSF base width (constant blur)
     B       — CCD bias pedestal
     ne_ratio — λ₂/λ₁ intensity scale ratio

 Fixed parameters (2):
   sigma1 = 0  — sinusoidal PSF variation (see PSF investigation below)
   sigma2 = 0  — cosinusoidal PSF variation (see PSF investigation below)

═══════════════════════════════════════════════════════════════════════════════
 KEY ARCHITECTURAL DECISIONS AND THEIR PHYSICAL MOTIVATION
═══════════════════════════════════════════════════════════════════════════════

 1. INDEPENDENT REFLECTIVITIES R1 AND R2
    The etalon coating reflectivity is wavelength-dependent. The ~2 nm
    separation between the two neon lines (640.2 nm and 638.3 nm) is
    sufficient to produce a measurable finesse difference. On the
    1_cal_120sexp_swapped_ROI_L1.1 calibration image, the λ₂ peaks are
    visibly sharper and taller than the λ₁ peaks at the same radius.

    Diagnosis (2026-05-12): The four-panel diagnostic plot
    (H05_cal_diagnostics_2026_05_06.py) showed a strong S-curve in the
    residual-vs-model scatter plot — positive residual at low model values
    (troughs, where λ₂ peaks sit) and near-zero at high model values
    (λ₁ peaks). This is the fingerprint of a wavelength-dependent PSF
    width, specifically a higher effective reflectivity at λ₂.

    Result (2026-05-12, 1_cal_120sexp_swapped_ROI_L1.1):
      R1 = 0.239 ± 0.010   [λ₁ = 640.2 nm]
      R2 = 0.334 ± 0.019   [λ₂ = 638.3 nm]
      ΔR = R2 − R1 = +0.095  (4.7σ significant)
    χ²/ν dropped from 7.0 (single shared R) to 1.58 (independent R1, R2).
    The S-curve in the scatter plot disappeared entirely.

    Note: The FlatSat-measured R = 0.53 was measured at a single wavelength
    and cannot resolve this wavelength dependence. R1 and R2 must be
    recovered from fringe data for each calibration image.

 2. PSF MODEL — CONSTANT BLUR (σ₀ ONLY)
    Harding (2014) uses a three-term Fourier PSF:
      σ(r) = σ₀ + σ₁·sin(πr/r_max) + σ₂·cos(πr/r_max)

    Investigation (2026-05-12, H05_psf_model_comparison_2026_05_12.py):
    A nested F-test was performed comparing:
      Model A: σ(r) = σ₀ + σ₁·sin  (σ₂ fixed at 0 — already unidentifiable)
      Model B: σ(r) = σ₀             (σ₁ and σ₂ both fixed at 0)

    Results on 1_cal_120sexp_swapped_ROI_L1.1:
      Model A:  χ²/ν = 1.5776   n_free = 11
      Model B:  χ²/ν = 1.5765   n_free = 10   (slightly BETTER)
      Δχ²/ν = −0.0011   F = 0.000   p-value = 0.9984

    DECISION: σ₁ is not statistically significant (p = 0.998 >> 0.05).
    Model B is preferred — simpler, more honest, and every science-relevant
    parameter (R1, R2, ne_ratio, α, t) shifted by < 0.1σ between models.

    Physical interpretation: the WindCube FPI finesse is low (F ≈ 5,
    R ≈ 0.24–0.33), so the fringes are broad and the PSF contributes
    relatively little additional broadening. A constant-width Gaussian
    PSF fully captures the instrumental blur for this instrument.

    Authoritative PSF width (from Model B):
      σ₀ = 0.553 ± 0.010 px   (2×2 binned pixels)
    Note: the Model A value σ₀ = 0.354 px was an artefact of the σ₀/σ₁
    degeneracy — when σ₁ is freed, it absorbs part of the blur, making
    σ₀ appear smaller than it really is.

 3. ne_ratio — LAMP INTENSITY RATIO FITTED FROM DATA
    The nominal λ₂/λ₁ intensity ratio from Burns et al. (1950) spectroscopic
    standards is 0.36. The fitted value from the calibration image is
      ne_ratio = 0.509 ± 0.012
    This is significantly above the spectroscopic standard, likely due to
    the specific discharge conditions of the WindCube neon lamp. The ratio
    must be treated as a fit parameter, not a fixed constant.

 4. COVARIANCE — FINITE-DIFFERENCE JACOBIAN
    Three covariance approaches were tried before finding one that works:

    Attempt 1: lm.jac[:n_good, :] — the data rows of the LM Jacobian.
    FAILED: The soft-bound penalty rows appended to the residual vector
    contaminate lm.jac via finite-difference coupling, producing
    artificially tiny stderrs (e.g., σ_R ≈ 7e-14).

    Attempt 2: trf re-evaluation with explicit bounds.
    FAILED: When lm.x sits exactly on a soft-bound edge (common with the
    LM penalty approach), scipy.optimize.least_squares raises
    "Initial guess is outside of provided bounds".

    Attempt 3 (current): _fd_jacobian() — direct forward finite-difference
    Jacobian computed at the LM solution, using a data-only residual
    function with no bounds. SVD-based covariance inversion handles any
    near-singular directions gracefully.
    SUCCESS: Produces physically meaningful stderrs.

 5. STAGED INVERSION RATIONALE
    Stage 1: Photometric baseline (I0, I1, I2, B) — establish the intensity
             envelope before allowing geometry parameters to move.
    Stage 2: Add geometry and reflectivities (t, alpha, R1, R2) — the
             fringe positions and peak heights.
    Stage 3: Add PSF width (sigma0) — fringe peak widths. sigma1 and sigma2
             are NOT added here (they are permanently fixed at 0).
    Stage 4: Full free optimisation of all 10 free parameters, with
             finite-difference covariance computed at the solution.

═══════════════════════════════════════════════════════════════════════════════
 CHANGELOG
═══════════════════════════════════════════════════════════════════════════════

 2026-05-12 (this version):
   - sigma1 fixed at 0 (in addition to sigma2). Justified by nested F-test:
     p = 0.998, Δχ²/ν = −0.0011. Authoritative σ₀ = 0.553 ± 0.010 px.
   - _STAGE_FREE[4] updated: sigma1 and sigma2 both excluded.
   - FitResult.sigma_sigma1 = NaN (fixed). Table row shows "fixed".
   - Stage 3 log message updated: sigma1 no longer mentioned as free.
   - Stage 4 free param count updated from 11 to 10.
   - Model equation in figure footer updated: σ(r) = σ₀ (constant).
   - Full investigative notes added to this docstring.

 2026-05-12 (prior commit same day):
   - _fd_jacobian() replaces trf re-evaluation for covariance estimation.
   - Import confirmed as airy_forward_model_2026_05_05 (contains
     phase_correct_gap; no _05_06 file exists in the repo).

 2026-05-06 (H05_calibration_inversion_2026_05_06.py):
   - Independent R1, R2 added (previously single shared R).
   - ne_ratio added as free parameter (previously fixed at NE_INTENSITY_2).
   - sigma2 fixed at 0 (unidentifiable — singular Hessian column).
   - Poisson sigma floor replaces rolling-std estimator.

═══════════════════════════════════════════════════════════════════════════════
 INPUT FORMAT AND USAGE
═══════════════════════════════════════════════════════════════════════════════

 Input .npy file formats accepted:
   (2, N)    — row 0 = r² (px²), row 1 = profile (ADU)   ← preferred
   (3, N)    — row 0 = r², row 1 = profile, row 2 = SEM
   (N, 2)    — col 0 = r², col 1 = profile
   (N,)      — profile only; r² inferred (not recommended)

 Run from repo root:
    python src/processing/H05_calibration_inversion_2026_05_12.py
"""

import pathlib
import sys
import tkinter as tk
from tkinter import filedialog, simpledialog
import logging
from dataclasses import dataclass

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
# Parameter ordering (12 positions in p_all vector)
# ---------------------------------------------------------------------------
_NAMES = ['t_m', 'alpha', 'R1', 'R2', 'I0', 'I1', 'I2',
          'sigma0', 'sigma1', 'sigma2', 'B', 'ne_ratio']
_IDX   = {n: i for i, n in enumerate(_NAMES)}

# sigma1 and sigma2 are both fixed at 0 throughout all stages.
# sigma1: nested F-test (p = 0.998) shows it is not identifiable from data.
# sigma2: always had a near-zero Jacobian column (singular Hessian direction).
# See docstring §2 for full investigation notes.
_FIXED_PSF = {'sigma1', 'sigma2'}

_STAGE_FREE = {
    1: ['I0', 'I1', 'I2', 'B'],
    2: ['t_m', 'alpha', 'R1', 'R2', 'I0', 'I1', 'I2', 'B'],
    3: ['t_m', 'alpha', 'R1', 'R2', 'I0', 'I1', 'I2', 'sigma0', 'B'],
    4: [n for n in _NAMES if n not in _FIXED_PSF],   # 10 free params
}

# ---------------------------------------------------------------------------
# Forward model
# ---------------------------------------------------------------------------

def _neon_model(r_arr, r_max, t, alpha, R1, R2, I0, I1, I2,
                sigma0, sigma1, sigma2, B, ne_ratio, _N_fine=500):
    """
    Two-line neon forward model with independent reflectivities R1, R2.
        S(r) = Ã(r; λ₁, R1) + ne_ratio × Ã(r; λ₂, R2) + B
    sigma1 = sigma2 = 0 always (constant-blur PSF, see docstring §2).
    """
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
    """Return (composite, lam1+B, ne_ratio*lam2+B) for plotting."""
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
# Finite-difference Jacobian for covariance estimation
# ---------------------------------------------------------------------------

def _fd_jacobian(residual_fn, p0, rel_step=1e-5):
    """
    Forward finite-difference Jacobian of residual_fn at p0.

    step = max(rel_step * |p0[j]|, 1e-10) for each parameter j.
    Returns J with shape (n_residuals, n_params).

    Called only after the LM solve for covariance estimation — the extra
    n_params forward evaluations are negligible in runtime.
    No bounds needed: we evaluate derivatives at a fixed point.

    This approach was adopted after two failed alternatives:
      - lm.jac[:n_good,:] → penalty-row contamination → stderrs ≈ 1e-14
      - trf re-evaluation → "Initial guess outside bounds" when lm.x on edge
    See docstring §4 for full history.
    """
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
# LM stage runner
# ---------------------------------------------------------------------------

def _run_stage(r_good, prof_good, sig_good, r_max,
               p_all, free_names, bounds_dict, config):
    """
    Run one LM stage with soft-bound penalty residuals.
    Covariance via _fd_jacobian at the solution (data residuals only).

    Returns (p_updated, cov, stderrs, chi2_red, lm_result).
    stderrs and cov are indexed by position in free_names.
    """
    free_idx  = np.array([_IDX[n] for n in free_names])
    p_fixed   = p_all.copy()
    p0        = p_fixed[free_idx]
    n_good    = len(r_good)
    n_free    = len(free_names)

    lo_arr    = np.array([bounds_dict[n][1] for n in free_names])
    hi_arr    = np.array([bounds_dict[n][2] for n in free_names])
    range_arr = hi_arr - lo_arr + 1e-30
    pen_sigma = range_arr * 0.01   # soft-bound: 1% of range per sigma unit

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

    # chi² from data residuals only (no penalty rows)
    t, alpha, R1, R2, I0, I1, I2, sigma0, sigma1, sigma2, B, ne_ratio = p_updated
    model_f = _neon_model(r_good, r_max, t, alpha, R1, R2,
                          I0, I1, I2, sigma0, sigma1, sigma2, B, ne_ratio)
    data_r  = (prof_good - model_f) / sig_good
    dof     = max(n_good - n_free, 1)
    chi2    = float(np.sum(data_r ** 2)) / dof

    # Covariance via FD Jacobian — bounds-free, penalty-free
    def _residuals_data_only(p_free):
        p = p_fixed.copy()
        p[free_idx] = p_free
        t, alpha, R1, R2, I0, I1, I2, sigma0, sigma1, sigma2, B, ne_ratio = p
        model = _neon_model(r_good, r_max, t, alpha, R1, R2,
                            I0, I1, I2, sigma0, sigma1, sigma2, B, ne_ratio)
        return (prof_good - model) / sig_good

    try:
        J        = _fd_jacobian(_residuals_data_only, lm.x)
        _, s, VT = np.linalg.svd(J, full_matrices=False)
        threshold = np.finfo(float).eps * max(J.shape) * s[0]
        s_inv    = np.where(s > threshold, 1.0 / s, 0.0)
        JTJ_inv  = (VT.T * s_inv**2) @ VT
        cov      = chi2 * JTJ_inv
        stderrs  = np.sqrt(np.maximum(np.diag(cov), 0.0))
        if not np.all(np.isfinite(stderrs)):
            raise np.linalg.LinAlgError("non-finite stderrs after SVD")
    except (np.linalg.LinAlgError, ValueError) as exc:
        log.warning(f"  Covariance estimation failed: {exc}")
        stderrs = np.full(n_free, np.inf)
        cov     = np.full((n_free, n_free), np.inf)

    return p_updated, cov, stderrs, chi2, lm


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class _FringeProfile:
    profile:       np.ndarray
    r_grid:        np.ndarray
    sigma_profile: np.ndarray
    masked:        np.ndarray
    r_max_px:      float


@dataclass
class FitResult:
    """
    Fit result for the 10-free-parameter neon calibration inversion.

    sigma1 = sigma2 = 0 (fixed).  sigma_sigma1 = sigma_sigma2 = NaN.
    See docstring §2 for the PSF investigation that justified fixing them.
    """
    t_m:      float;  alpha:    float
    R1:       float;  R2:       float
    I0:       float;  I1:       float;  I2:      float
    sigma0:   float
    sigma1:   float   # always 0.0 — fixed (see docstring §2)
    sigma2:   float   # always 0.0 — fixed (see docstring §2)
    B:        float;  ne_ratio: float

    sigma_t_m:      float;  sigma_alpha:    float
    sigma_R1:       float;  sigma_R2:       float
    sigma_I0:       float;  sigma_I1:       float;  sigma_I2:    float
    sigma_sigma0:   float
    sigma_sigma1:   float   # NaN — fixed
    sigma_sigma2:   float   # NaN — fixed
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
                         gtol: float = 1e-14) -> FitResult:
    """
    4-stage LM inversion with 10 free parameters.
    sigma1 = sigma2 = 0 fixed throughout (constant-blur PSF).
    Covariance from FD Jacobian at Stage 4 solution.
    """
    good   = (~fp.masked & np.isfinite(fp.sigma_profile)
              & (fp.sigma_profile > 0) & np.isfinite(fp.profile))
    r_good = fp.r_grid[good]
    p_good = fp.profile[good]
    s_good = np.maximum(fp.sigma_profile[good].copy(), 1.0)
    r_max  = float(fp.r_max_px)
    n_good = int(good.sum())

    if n_good < 30:
        raise ValueError(f"Only {n_good} usable bins — need ≥ 30.")

    I0_init = float(np.percentile(p_good, 75))
    B_init  = float(np.percentile(p_good, 5)) * 0.8
    if ne_ratio_init is None:
        ne_ratio_init = float(NE_INTENSITY_2_NOMINAL)

    bounds = {
        't_m':      (t_eff,          t_eff - 20e-6,      t_eff + 20e-6),
        'alpha':    (alpha_init,     alpha_init * 0.95,  alpha_init * 1.05),
        'R1':       (R1_init,        0.05,               0.95),
        'R2':       (R2_init,        0.05,               0.95),
        'I0':       (I0_init,        100.0,              15000.0),
        'I1':       (0.0,            -0.5,               0.5),
        'I2':       (0.0,            -0.5,               0.5),
        'sigma0':   (sigma0_init,    0.01,               5.0),
        'sigma1':   (0.0,            -2.0,               2.0),   # never freed
        'sigma2':   (0.0,            -2.0,               2.0),   # never freed
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

    log.info("Stage 2 — geometry + reflectivities {t, alpha, R1, R2, I0, I1, I2, B}")
    p_all, _, _, chi2_2, _ = _run_stage(
        r_good, p_good, s_good, r_max, p_all, _STAGE_FREE[2], bounds, cfg)
    chi2_by_stage.append(chi2_2)
    log.info(f"  χ²/ν = {chi2_2:.3f}   R1={p_all[_IDX['R1']]:.4f}  "
             f"R2={p_all[_IDX['R2']]:.4f}")

    log.info("Stage 3 — + PSF base width sigma0  (sigma1, sigma2 fixed at 0)")
    p_all, _, _, chi2_3, _ = _run_stage(
        r_good, p_good, s_good, r_max, p_all, _STAGE_FREE[3], bounds, cfg)
    chi2_by_stage.append(chi2_3)
    log.info(f"  χ²/ν = {chi2_3:.3f}   sigma0={p_all[_IDX['sigma0']]:.4f} px")

    log.info("Stage 4 — 10 free params (sigma1=sigma2=0 fixed); FD covariance")
    p_all, cov4, se4, chi2_4, res4 = _run_stage(
        r_good, p_good, s_good, r_max, p_all, _STAGE_FREE[4], bounds, cfg)
    chi2_by_stage.append(chi2_4)
    log.info(f"  χ²/ν = {chi2_4:.3f}   R1={p_all[_IDX['R1']]:.4f}  "
             f"R2={p_all[_IDX['R2']]:.4f}  "
             f"sigma0={p_all[_IDX['sigma0']]:.4f} px  "
             f"ne_ratio={p_all[_IDX['ne_ratio']]:.4f}")

    # se4 indexed by position in _STAGE_FREE[4] (10 elements)
    s4_names = _STAGE_FREE[4]
    s4_lkp   = {n: i for i, n in enumerate(s4_names)}

    def _se(name):
        return float(se4[s4_lkp[name]]) if name in s4_lkp else float('nan')

    log.info("  Stage 4 stderrs:")
    for name in s4_names:
        log.info(f"    sigma_{name:12s} = {_se(name):.3e}")
    log.info("    sigma_sigma1    = NaN  (fixed at 0)")
    log.info("    sigma_sigma2    = NaN  (fixed at 0)")

    converged = bool(res4.success or res4.cost < 1e-10)
    t_f, alpha_f, R1_f, R2_f, I0_f, I1_f, I2_f, \
        s0_f, s1_f, s2_f, B_f, ne_f = p_all

    eps_cal       = (2.0 * t_f / NE_WAVELENGTH_1_AIR_M) % 1.0
    sigma_eps_cal = (2.0 / NE_WAVELENGTH_1_AIR_M) * _se('t_m')

    return FitResult(
        t_m=float(t_f),     alpha=float(alpha_f),
        R1=float(R1_f),     R2=float(R2_f),
        I0=float(I0_f),     I1=float(I1_f),     I2=float(I2_f),
        sigma0=float(s0_f),
        sigma1=0.0,          # fixed
        sigma2=0.0,          # fixed
        B=float(B_f),       ne_ratio=float(ne_f),

        sigma_t_m=_se('t_m'),        sigma_alpha=_se('alpha'),
        sigma_R1=_se('R1'),          sigma_R2=_se('R2'),
        sigma_I0=_se('I0'),          sigma_I1=_se('I1'),    sigma_I2=_se('I2'),
        sigma_sigma0=_se('sigma0'),
        sigma_sigma1=float('nan'),   # fixed
        sigma_sigma2=float('nan'),   # fixed
        sigma_B=_se('B'),            sigma_ne_ratio=_se('ne_ratio'),

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
    """Poisson 1σ: sqrt(signal), floor 1 ADU."""
    return np.maximum(np.sqrt(np.maximum(profile, 1.0)), 1.0)


def _fmt_unc(value):
    """Format uncertainty in scientific notation, or 'fixed' if NaN."""
    if np.isnan(value):
        return "fixed"
    return f"±{value:.2e}"


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(r2_data, profile, sigma,
                r_fine, model_fine, lam1_fine, lam2_fine,
                fit: FitResult, source_name: str = "",
                source_path: str = "") -> plt.Figure:

    r2_fine       = r_fine ** 2
    model_at_data = np.interp(r2_data, r2_fine, model_fine)
    residual      = profile - model_at_data

    fig = plt.figure(figsize=(14, 11.5))
    gs  = gridspec.GridSpec(3, 1, height_ratios=[3, 1.5, 2.6],
                            hspace=0.08, top=0.90, bottom=0.03,
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
                zorder=4, label="Best-fit composite  (Stage 4)")
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
    ax_fit.text(
        0.02, 0.97,
        f"χ²/ν = {fit.chi2_reduced:.3f}   {conv_str}\n"
        f"R1 = {fit.R1:.4f} {_fmt_unc(fit.sigma_R1)}   "
        f"R2 = {fit.R2:.4f} {_fmt_unc(fit.sigma_R2)}   "
        f"ΔR = R2−R1 = {fit.R2-fit.R1:+.4f}\n"
        f"ne_ratio = {fit.ne_ratio:.4f} {_fmt_unc(fit.sigma_ne_ratio)}   "
        f"[nominal = {NE_INTENSITY_2_NOMINAL:.2f}]   "
        f"σ₀ = {fit.sigma0:.3f} {_fmt_unc(fit.sigma_sigma0)} px",
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
    rows = [
        ("t",        f"{fit.t_m*1e3:.7f} mm",
                     f"±{fit.sigma_t_m*1e6:.2e} µm" if not np.isnan(fit.sigma_t_m) else "fixed",
                     "Etalon gap  [Tolansky-seeded, Group A]"),
        ("α",        f"{fit.alpha:.5e} rad/px",
                     _fmt_unc(fit.sigma_alpha),
                     "Plate scale  [Tolansky-seeded, Group A]"),
        ("R1",       f"{fit.R1:.5f}",
                     _fmt_unc(fit.sigma_R1),
                     "Reflectivity λ₁ = 640.2 nm  [Group B]"),
        ("R2",       f"{fit.R2:.5f}",
                     _fmt_unc(fit.sigma_R2),
                     "Reflectivity λ₂ = 638.3 nm  [Group B]"),
        ("ΔR=R2−R1", f"{fit.R2-fit.R1:+.5f}",
                     "—",
                     "Wavelength-dependent finesse difference  (4.7σ significant)"),
        ("I₀",       f"{fit.I0:.1f} ADU",
                     _fmt_unc(fit.sigma_I0),
                     "Mean intensity  [Group B, shared]"),
        ("I₁",       f"{fit.I1:.5f}",
                     _fmt_unc(fit.sigma_I1),
                     "Linear vignetting  [Group B, shared]"),
        ("I₂",       f"{fit.I2:.5f}",
                     _fmt_unc(fit.sigma_I2),
                     "Quadratic vignetting  [Group B, shared]"),
        ("σ₀",       f"{fit.sigma0:.4f} px",
                     _fmt_unc(fit.sigma_sigma0),
                     "PSF base width  [Group B]  σ(r) = σ₀  (constant blur)"),
        ("σ₁",       "0.0000 px",
                     "fixed",
                     "PSF sin variation  [fixed=0; F-test p=0.998, see docstring §2]"),
        ("σ₂",       "0.0000 px",
                     "fixed",
                     "PSF cos variation  [fixed=0; singular Hessian column]"),
        ("B",        f"{fit.B:.1f} ADU",
                     _fmt_unc(fit.sigma_B),
                     "CCD bias pedestal  [Group B]"),
        ("ne_ratio", f"{fit.ne_ratio:.4f}",
                     _fmt_unc(fit.sigma_ne_ratio),
                     f"λ₂/λ₁ intensity ratio  [Group B, nominal={NE_INTENSITY_2_NOMINAL:.2f}]"),
        ("ε_cal",    f"{fit.epsilon_cal:.6f}",
                     _fmt_unc(fit.sigma_epsilon_cal),
                     "Fractional order at centre  (zero-wind phase reference)"),
    ]

    tbl = ax_tbl.table(
        cellText=rows,
        colLabels=["Param", "Fitted value", "1σ", "Description"],
        cellLoc="left", loc="upper center",
        colWidths=[0.10, 0.20, 0.13, 0.57])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.0)
    tbl.scale(1, 1.18)

    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#c8d8f0")
            cell.set_text_props(fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f0f4ff")
        if row in (3, 4):           # R2 and ΔR — yellow highlight
            cell.set_facecolor("#fff4c2")
        if row in (10, 11):         # σ₁, σ₂ fixed — grey
            cell.set_facecolor("#eeeeee")
        if row == 13:               # ne_ratio — green tint
            cell.set_facecolor("#f4fff4")

    stage_str = "  ".join(
        f"S{i+1}: {v:.2f}" for i, v in enumerate(fit.chi2_by_stage))
    ax_tbl.text(0.01, 0.01,
                f"χ²/ν by stage:  {stage_str}    "
                f"bins used: {fit.n_bins_used}   "
                f"free params: 10  (σ₁=σ₂=0 fixed; F-test p=0.998)",
                transform=ax_tbl.transAxes, va="bottom", ha="left",
                fontsize=8.5, fontfamily="monospace", color="dimgrey")

    fig.suptitle(
        "WindCube FPI — Neon Calibration Fringe Inversion  "
        "(10-param: independent R1, R2; constant PSF / Harding 2014)",
        fontsize=12, fontweight="bold", y=0.980)
    if source_path:
        fig.text(0.5, 0.963, source_path,
                 ha="center", va="top", fontsize=8,
                 fontfamily="monospace", color="dimgrey")
    # Model equation — updated to reflect constant-blur PSF
    _model_eq = (
        r"$S(r) = \tilde{A}(r;\,\lambda_1, t, \alpha, R_1)"
        r"\;\circledast\;\mathcal{G}(\sigma_0)"
        r"\;+\;n_\mathrm{ratio}\cdot\tilde{A}(r;\,\lambda_2, t, \alpha, R_2)"
        r"\;\circledast\;\mathcal{G}(\sigma_0)\;+\;B$"
        "\n"
        r"$I(r)=I_0[1+I_1(r/r_\mathrm{max})+I_2(r/r_\mathrm{max})^2]$"
        r"$\quad\sigma(r)=\sigma_0\;(\mathrm{constant\;blur})$"
        r"$\quad[\tilde{A}=\mathrm{Airy}\times I(r),\;\mathcal{G}=\mathrm{Gaussian\;PSF}]$"
    )
    fig.text(0.5, 0.948, _model_eq,
             ha="center", va="top", fontsize=8.5, color="#1a1a2e",
             linespacing=1.6)
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    _tk = tk.Tk(); _tk.withdraw()
    npy_path = filedialog.askopenfilename(
        title="Select neon calibration profile (.npy, tabulated vs r²)",
        filetypes=[("NumPy arrays", "*.npy"), ("All files", "*.*")])
    _tk.destroy()
    if not npy_path:
        print("No file selected — exiting."); return
    npy_path = pathlib.Path(npy_path)
    print(f"\nLoaded: {npy_path.name}")

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
        "Starting guess for λ₁ (640.2 nm) effective reflectivity.\nFlatSat: 0.53",
        initialvalue=0.53, minvalue=0.05, maxvalue=0.95) or 0.53

    R2_init = simpledialog.askfloat("R2 initial guess",
        "Starting guess for λ₂ (638.3 nm) effective reflectivity.\n"
        "Expect R2 > R1 if λ₂ peaks are sharper.",
        initialvalue=0.53, minvalue=0.05, maxvalue=0.95) or 0.53

    ne_ratio_init = simpledialog.askfloat("ne_ratio initial guess",
        f"Starting guess for λ₂/λ₁ intensity ratio.\n"
        f"Nominal = {NE_INTENSITY_2_NOMINAL:.2f}",
        initialvalue=float(NE_INTENSITY_2_NOMINAL),
        minvalue=0.01, maxvalue=2.0) or float(NE_INTENSITY_2_NOMINAL)

    _tk2.destroy()

    t_tolansky_m = t_tolansky_mm * 1e-3
    t_eff = phase_correct_gap(t_tolansky_m, eps_a, NE_WAVELENGTH_1_AIR_M)
    print(f"\nphase_correct_gap:  t_tolansky = {t_tolansky_m*1e3:.7f} mm  "
          f"→  t_eff = {t_eff*1e3:.7f} mm  "
          f"(correction = {(t_eff - t_tolansky_m)*1e9:.1f} nm)")

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
        print("  SEM: Poisson sqrt(signal)")

    fp = _FringeProfile(
        profile=profile_adu, r_grid=r_grid,
        sigma_profile=sigma_adu,
        masked=np.zeros(len(r_grid), dtype=bool),
        r_max_px=r_max_px)

    print(f"  {len(r_grid)} bins, r ∈ [{r_grid.min():.1f}, {r_grid.max():.1f}] px, "
          f"signal ∈ [{profile_adu.min():.0f}, {profile_adu.max():.0f}] ADU  "
          f"σ ∈ [{sigma_adu.min():.1f}, {sigma_adu.max():.1f}] ADU")

    print(f"\nRunning 10-parameter staged LM inversion  "
          f"(R1 init={R1_init:.3f}, R2 init={R2_init:.3f}  "
          f"σ₁=σ₂=0 fixed)…")
    fit = run_staged_inversion(
        fp, t_eff, alpha_tolansky, eps_a,
        R1_init=R1_init, R2_init=R2_init,
        ne_ratio_init=ne_ratio_init)

    print(f"\n{'='*68}")
    print("CALIBRATION INVERSION RESULT  (10 free params; σ₁=σ₂=0 fixed)")
    print(f"{'='*68}")
    print(f"  Converged:      {fit.converged}")
    print(f"  χ²/ν:           {fit.chi2_reduced:.4f}")
    print(f"  χ²/ν by stage:  {[f'{v:.3f}' for v in fit.chi2_by_stage]}")
    print(f"  Bins used:      {fit.n_bins_used}")
    print()
    print(f"  --- Group A (Tolansky-seeded) ---")
    print(f"  t      = {fit.t_m*1e3:.7f} mm   {_fmt_unc(fit.sigma_t_m*1e6)} µm")
    print(f"  alpha  = {fit.alpha:.5e}   {_fmt_unc(fit.sigma_alpha)} rad/px")
    print()
    print(f"  --- Group B ---")
    print(f"  R1       = {fit.R1:.5f}   {_fmt_unc(fit.sigma_R1)}   [λ₁=640.2 nm]")
    print(f"  R2       = {fit.R2:.5f}   {_fmt_unc(fit.sigma_R2)}   [λ₂=638.3 nm]")
    print(f"  ΔR=R2−R1 = {fit.R2-fit.R1:+.5f}   "
          f"({'R2>R1: λ₂ sharper' if fit.R2>fit.R1 else 'R2<R1: λ₂ broader'})")
    print(f"  I0       = {fit.I0:.1f}   {_fmt_unc(fit.sigma_I0)} ADU")
    print(f"  I1       = {fit.I1:.5f}   {_fmt_unc(fit.sigma_I1)}")
    print(f"  I2       = {fit.I2:.5f}   {_fmt_unc(fit.sigma_I2)}")
    print(f"  sigma0   = {fit.sigma0:.4f}   {_fmt_unc(fit.sigma_sigma0)} px  "
          f"[σ₁=σ₂=0 fixed; F-test p=0.998]")
    print(f"  B        = {fit.B:.2f}   {_fmt_unc(fit.sigma_B)} ADU")
    print(f"  ne_ratio = {fit.ne_ratio:.4f}   {_fmt_unc(fit.sigma_ne_ratio)}   "
          f"[nominal {NE_INTENSITY_2_NOMINAL:.2f}]")
    print()
    print(f"  ε_cal  = {fit.epsilon_cal:.6f}   {_fmt_unc(fit.sigma_epsilon_cal)}")
    print(f"{'='*68}")

    r_fine = np.linspace(0.0, r_max_px, 2000)
    model_fine, lam1_fine, lam2_fine = _model_components(
        r_fine, r_max_px,
        fit.t_m, fit.alpha, fit.R1, fit.R2,
        fit.I0, fit.I1, fit.I2,
        fit.sigma0, fit.sigma1, fit.sigma2,
        fit.B, fit.ne_ratio)

    fig = make_figure(
        r2_data=r_grid**2, profile=profile_adu, sigma=sigma_adu,
        r_fine=r_fine, model_fine=model_fine,
        lam1_fine=lam1_fine, lam2_fine=lam2_fine,
        fit=fit, source_name=npy_path.name, source_path=str(npy_path))
    plt.show()


if __name__ == "__main__":
    main()
