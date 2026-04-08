"""
M05 — Calibration Inversion  (redesigned stage architecture)
fpi/m05_calibration_inversion.py

Eight-stage staged inversion of a WindCube neon lamp FringeProfile
into the 10 instrument parameters {t, R, α, I₀, I₁, I₂, B, σ₀, σ₁, σ₂}.

Redesigned stage sequence compared to the original six-stage version:

  ORIGINAL problem:  alpha was never independently measured before
  optimisation began.  Stages 2 and 3 freed alpha and R simultaneously
  (while the other was held fixed), creating a correlation trap: alpha
  drifted to a wrong value while sigma0 inflated to absorb the residual.

  NEW sequence exploits the unique observability structure of the Airy
  fringe pattern:

    Stage P  — Peak-finding pre-solve: measure alpha analytically from
               the r²-spacing of bright (λ₁) peaks.  No optimizer.
               Robust to R, sigma0, I0, B (none affect r²-spacing).

    Stage 1  — t-only brute-force scan: WLS-solve I0, B at each t.
               alpha is now measured, not guessed.

    Stage 2  — WLS amplitude solve: I0, B analytically given geometry.

    Stage 3  — R only, small-r bins (r < r_R_cutoff_px):
               PSF contribution is negligible at small r, so R is
               independently constrained.  sigma0 stays fixed.

    Stage 4  — sigma0 only, large-r bins (r > r_s0_cutoff_px):
               Fringe spacing compressed, PSF dominates width trend.
               R is now fixed from Stage 3.

    Stage 5  — Envelope shape: I1, I2 (linear WLS given everything else).

    Stage 6  — PSF shape: sigma1, sigma2 (fine-tune radial PSF variation).

    Stage 7  — Full free: all 10 parameters, seeded from Stages 1–6.

Reference: Harding et al. (2014), Section 3–5
Spec: docs/specs/m05_calibration_inversion_spec.md
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.linalg import svd as _svd
from scipy.optimize import least_squares
from scipy.signal import find_peaks

from fpi.m01_airy_forward_model import (
    InstrumentParams,
    NE_INTENSITY_2,
    NE_WAVELENGTH_1_M,
    NE_WAVELENGTH_2_M,
    OI_WAVELENGTH_M,
    airy_ideal,
    psf_sigma,
)
from fpi.m03_annular_reduction import FringeProfile

from fpi.fpi_airy_fit import (
    CalibrationFitConfig as _CalibrationFitConfig,
    InstrumentConfig as _InstrumentConfig,
    fit_calibration_profile as _fit_calibration_profile,
)

# ---------------------------------------------------------------------------
# Warnings / errors
# ---------------------------------------------------------------------------

class FitWarning(UserWarning):
    """Issued for non-fatal fit quality issues."""

class FitError(RuntimeError):
    """Raised when convergence guard fails with require_convergence_guard=True."""


# ---------------------------------------------------------------------------
# Fit flags
# ---------------------------------------------------------------------------

class FitFlags:
    GOOD                = 0x00
    STAGE1_WRONG_PERIOD = 0x01  # t_best deviated > FSR/4 from t_stage0
    T_ALPHA_DEGENERATE  = 0x02  # |corr(t,α)| > 0.98
    R_SIGMA_DEGENERATE  = 0x04  # |corr(R,σ₀)| > 0.95
    PARAM_AT_BOUND      = 0x08  # any parameter hit its bound
    CHI2_HIGH           = 0x10  # chi2_red > 2.0
    CHI2_LOW            = 0x20  # chi2_red < 0.5
    MULTIPLE_MINIMA     = 0x40  # convergence guard found different minimum
    PSF_UNPHYSICAL      = 0x80  # σ(r) < 0.01 px at any radius
    ALPHA_PRESOLVE_FAIL = 0x100 # Stage P peak-finding returned < 4 peaks


# ---------------------------------------------------------------------------
# Three-tier bounds architecture
# ---------------------------------------------------------------------------

# Tier 1 — physics hard limits
PHYSICS_BOUNDS = {
    't_m':    (19e-3,   21e-3),
    'R':      (0.30,    0.90),
    'alpha':  (1e-5,    1e-3),
    'I0':     (1.0,     130000.0),
    'I1':     (-0.9,    0.9),
    'I2':     (-0.9,    0.9),
    'sigma0': (0.01,    5.0),
    'sigma1': (-3.0,    3.0),
    'sigma2': (-3.0,    3.0),
    'B':      (0.0,     10000.0),
}

# Tier 2 — WindCube instrument defaults
# alpha upper limit raised to accommodate measured values near 1.77e-4.
# sigma0 ceiling lowered: physically > 1.2 px is a sign of absorbing alpha error.
# R floor raised: etalon is specified for high-finesse operation.
INSTRUMENT_DEFAULTS = {
    't_m':    (20.05e-3, 20.20e-3),
    'R':      (0.45,     0.85),
    'alpha':  (1.50e-4,  2.20e-4),
    'I0':     (100.0,    15000.0),
    'I1':     (-0.5,     0.5),
    'I2':     (-0.5,     0.5),
    'sigma0': (0.05,     1.20),
    'sigma1': (-1.0,     1.0),
    'sigma2': (-1.0,     1.0),
    'B':      (50.0,     10000.0),
}


@dataclass
class FitConfig:
    """
    User-facing configuration for M05 staged calibration inversion.
    All fields optional — defaults from INSTRUMENT_DEFAULTS.

    Key changes from original:
      alpha_init       — now ignored if Stage P pre-solve succeeds; used
                         only as fallback if fewer than 4 bright peaks found.
      sigma0_bounds    — ceiling lowered to 1.2 px by default to prevent
                         PSF inflation absorbing alpha error.
      r_R_cutoff_px    — Stage 3 fits R using only bins with r < this value,
                         where PSF contribution is negligible.
      r_s0_cutoff_px   — Stage 4 fits sigma0 using only bins with r > this
                         value, where fringe spacing has compressed enough
                         that PSF dominates the observed peak width.
    """
    t_init_mm:    float = 20.117          # Gap from ring positions
    t_bounds_mm:  Optional[tuple] = None  # None → INSTRUMENT_DEFAULTS

    R_init:       float = 0.65            # raised from 0.53 — better prior for high-finesse etalon
    R_bounds:     Optional[tuple] = None

    alpha_init:   float = 1.77e-4         # fallback only; Stage P measures alpha from data
    alpha_bounds: Optional[tuple] = None

    sigma0_init:  float = 0.5             # lowered from 0.8; high sigma0 is a red flag
    sigma0_bounds: Optional[tuple] = None # None → INSTRUMENT_DEFAULTS ceiling of 1.2 px

    B_init:       Optional[float] = None  # None → auto (5th percentile of profile)
    B_bounds:     Optional[tuple] = None

    dark_level:   Optional[float] = None

    # Radial zone boundaries for decoupled R / sigma0 fitting (pixels)
    # Stage 3 (R):      use bins with r < r_R_cutoff_px
    # Stage 4 (sigma0): use bins with r > r_s0_cutoff_px
    r_R_cutoff_px:  float = 30.0   # PSF negligible below this radius
    r_s0_cutoff_px: float = 40.0   # fringe spacing compressed above this radius

    # Stage P peak-finding parameters
    # peak_height_percentile: percentile threshold to select bright (λ₁) peaks only,
    #   filtering out the dimmer interleaved λ₂ peaks (set > 50, typically 75–80).
    # peak_distance_fraction: minimum peak separation = this fraction × estimated FSR.
    peak_height_percentile: float = 78.0
    peak_distance_fraction: float = 0.55

    max_nfev:     int   = 10_000
    ftol:         float = 1e-14
    xtol:         float = 1e-14
    gtol:         float = 1e-14

    n_convergence_perturbations: int  = 3
    require_convergence_guard:   bool = False

    n_subpixels: int = 1

    def effective_bounds(self, param: str) -> tuple:
        """Return (min, max) in SI units after merging all three tiers."""
        phys_lo, phys_hi = PHYSICS_BOUNDS[param]
        inst_lo, inst_hi = INSTRUMENT_DEFAULTS[param]

        _attr_map = {
            't_m':    't_bounds_mm',
            'R':      'R_bounds',
            'alpha':  'alpha_bounds',
            'I0':     None,
            'I1':     None,
            'I2':     None,
            'sigma0': 'sigma0_bounds',
            'sigma1': None,
            'sigma2': None,
            'B':      'B_bounds',
        }
        attr = _attr_map.get(param)

        if param == 't_m':
            raw = self.t_bounds_mm
            if raw is not None:
                user_lo, user_hi = raw[0] * 1e-3, raw[1] * 1e-3
            else:
                user_lo, user_hi = inst_lo, inst_hi
        elif attr is not None:
            raw = getattr(self, attr, None)
            if raw is not None:
                user_lo, user_hi = raw
            else:
                user_lo, user_hi = inst_lo, inst_hi
        else:
            user_lo, user_hi = inst_lo, inst_hi

        eff_lo = max(phys_lo, user_lo)
        eff_hi = min(phys_hi, user_hi)

        if eff_lo >= eff_hi:
            raise ValueError(
                f"FitConfig: effective bounds for '{param}' are empty "
                f"[{eff_lo}, {eff_hi}]. User override contradicts physics bounds."
            )
        return (eff_lo, eff_hi)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class CalibrationResult:
    """
    Output of M05 calibration inversion.
    All fitted parameters have 1σ and 2σ uncertainty fields.
    Parameter order in covariance: [t_m, R_refl, alpha, I0, I1, I2, B, sigma0, sigma1, sigma2]

    New field: alpha_presolve — alpha measured analytically in Stage P,
    before any optimisation.  Useful for diagnosing whether the TRF
    fit moved alpha significantly away from the data-implied value.
    """
    # Best-fit parameters
    t_m:      float
    R_refl:   float
    alpha:    float
    I0:       float
    I1:       float
    I2:       float
    B:        float
    sigma0:   float
    sigma1:   float
    sigma2:   float

    # 1σ standard errors
    sigma_t_m:      float
    sigma_R_refl:   float
    sigma_alpha:    float
    sigma_I0:       float
    sigma_I1:       float
    sigma_I2:       float
    sigma_B:        float
    sigma_sigma0:   float
    sigma_sigma1:   float
    sigma_sigma2:   float

    # 2σ values
    two_sigma_t_m:      float
    two_sigma_R_refl:   float
    two_sigma_alpha:    float
    two_sigma_I0:       float
    two_sigma_I1:       float
    two_sigma_I2:       float
    two_sigma_B:        float
    two_sigma_sigma0:   float
    two_sigma_sigma1:   float
    two_sigma_sigma2:   float

    # Phase reference
    epsilon_cal:  float   # fractional order at λ₁=640.2248nm
    epsilon_sci:  float   # projected to science wavelength 630.0nm
    m0_cal:       float
    m0_sci:       float

    # Diagnostic: alpha from Stage P peak-finding (pre-optimisation)
    alpha_presolve:       float   # rad/px, or nan if Stage P failed
    n_peaks_presolve:     int     # number of bright peaks found in Stage P

    # Fit quality
    chi2_reduced:    float
    n_bins_used:     int
    n_params_free:   int
    covariance:      np.ndarray
    correlation:     np.ndarray
    converged:       bool
    quality_flags:   int

    # Stage progression (one entry per stage: P,1,2,3,4,5,6,7)
    chi2_by_stage:   list

    # Config used
    fit_config:      FitConfig


# ---------------------------------------------------------------------------
# validate_result_uncertainties
# ---------------------------------------------------------------------------

def validate_result_uncertainties(result: CalibrationResult) -> None:
    """
    Validate all sigma_ and two_sigma_ fields.
    Raises AssertionError if any field is None, zero, non-finite,
    or if two_sigma_ ≠ 2 × sigma_ (to relative precision 1e-8).
    """
    _fields = [
        't_m', 'R_refl', 'alpha', 'I0', 'I1', 'I2', 'B', 'sigma0', 'sigma1', 'sigma2',
    ]
    for f in _fields:
        sigma = getattr(result, f'sigma_{f}')
        two_sigma = getattr(result, f'two_sigma_{f}')
        assert sigma is not None,       f"sigma_{f} is None"
        assert np.isfinite(sigma),      f"sigma_{f} = {sigma} is not finite"
        assert sigma > 0,               f"sigma_{f} = {sigma} is not positive"
        expected = 2.0 * sigma
        assert abs(two_sigma - expected) < 1e-8 * expected + 1e-30, (
            f"two_sigma_{f} = {two_sigma} ≠ 2 × sigma_{f} = {expected}"
        )


# ---------------------------------------------------------------------------
# Internal physics helpers (unchanged from original)
# ---------------------------------------------------------------------------

_PARAM_NAMES = ['t_m', 'R_refl', 'alpha', 'I0', 'I1', 'I2', 'B', 'sigma0', 'sigma1', 'sigma2']


def _apply_psf_vectorized(A_ideal: np.ndarray, r_fine: np.ndarray,
                           sigma_arr: np.ndarray) -> np.ndarray:
    """Shift-variant Gaussian PSF convolution, fully vectorized."""
    n = len(A_ideal)
    dr = r_fine[1] - r_fine[0]
    sig_max = float(np.max(sigma_arr))
    if sig_max <= 0.01:
        return A_ideal.copy()

    half_win = int(np.ceil(4.0 * sig_max / dr))
    offsets = np.arange(-half_win, half_win + 1)

    idx = np.arange(n)[:, np.newaxis] + offsets[np.newaxis, :]
    valid = (idx >= 0) & (idx < n)
    idx_safe = np.clip(idx, 0, n - 1)

    r_diff = offsets[np.newaxis, :] * dr
    sig_col = sigma_arr[:, np.newaxis]
    w = np.exp(-0.5 * (r_diff / sig_col) ** 2) * valid
    w /= w.sum(axis=1, keepdims=True)

    return np.sum(w * A_ideal[idx_safe], axis=1)


def _ne_model(r_data: np.ndarray, r_max: float,
              t: float, R_refl: float, alpha: float, n: float,
              I0: float, I1: float, I2: float,
              sigma0: float, sigma1: float, sigma2: float,
              B: float, n_fine: int = 2000,
              n_bins: int = None, n_subpixels: int = 8) -> np.ndarray:
    """
    Two-line Ne lamp model with PSF convolution and r²-bin averaging.
    S(r) = Ã(r; λ₁) + NE_INTENSITY_2 × Ã(r; λ₂) + B
    """
    if n_bins is not None:
        r_fine_psf = np.linspace(0.0, r_max, n_fine)
        kw = dict(t=t, R_refl=R_refl, alpha=alpha, n=n, r_max=r_max, I0=I0, I1=I1, I2=I2)
        A1_ideal = airy_ideal(r_fine_psf, NE_WAVELENGTH_1_M, **kw)
        A2_ideal = airy_ideal(r_fine_psf, NE_WAVELENGTH_2_M, **kw)
        sigma_arr = psf_sigma(r_fine_psf, r_max, sigma0, sigma1, sigma2)
        A1 = _apply_psf_vectorized(A1_ideal, r_fine_psf, sigma_arr)
        A2 = _apply_psf_vectorized(A2_ideal, r_fine_psf, sigma_arr)
        profile_1d = A1 + NE_INTENSITY_2 * A2 + B
        r2_max = r_max ** 2
        r2_fine = np.linspace(0.0, r2_max, n_fine)
        profile_r2 = np.interp(np.sqrt(r2_fine), r_fine_psf, profile_1d)
        dr2 = r2_max / n_bins
        bin_idx = np.clip(np.floor(r2_fine / dr2).astype(int), 0, n_bins - 1)
        counts = np.bincount(bin_idx, minlength=n_bins).astype(float)
        sums   = np.bincount(bin_idx, weights=profile_r2, minlength=n_bins)
        return np.where(counts > 0, sums / np.maximum(counts, 1.0), B)

    r_fine = np.linspace(0.0, r_max, n_fine)
    kw = dict(t=t, R_refl=R_refl, alpha=alpha, n=n, r_max=r_max, I0=I0, I1=I1, I2=I2)
    A1_ideal = airy_ideal(r_fine, NE_WAVELENGTH_1_M, **kw)
    A2_ideal = airy_ideal(r_fine, NE_WAVELENGTH_2_M, **kw)
    sigma_arr = psf_sigma(r_fine, r_max, sigma0, sigma1, sigma2)
    A1 = _apply_psf_vectorized(A1_ideal, r_fine, sigma_arr)
    A2 = _apply_psf_vectorized(A2_ideal, r_fine, sigma_arr)
    return np.interp(r_data, r_fine, A1 + NE_INTENSITY_2 * A2 + B)


def _ne_model_nopsf(r_data: np.ndarray, r_max: float,
                    t: float, R_refl: float, alpha: float, n: float,
                    I0: float, I1: float, I2: float,
                    B: float, n_fine: int = 2000,
                    n_bins: int = None, n_subpixels: int = 8) -> np.ndarray:
    """Two-line Ne model without PSF, with optional r²-bin averaging."""
    if n_bins is not None:
        r_fine_nopsf = np.linspace(0.0, r_max, n_fine)
        kw = dict(t=t, R_refl=R_refl, alpha=alpha, n=n, r_max=r_max, I0=I0, I1=I1, I2=I2)
        A1 = airy_ideal(r_fine_nopsf, NE_WAVELENGTH_1_M, **kw)
        A2 = airy_ideal(r_fine_nopsf, NE_WAVELENGTH_2_M, **kw)
        profile_1d = A1 + NE_INTENSITY_2 * A2 + B
        r2_max = r_max ** 2
        r2_fine = np.linspace(0.0, r2_max, n_fine)
        profile_r2 = np.interp(np.sqrt(r2_fine), r_fine_nopsf, profile_1d)
        dr2 = r2_max / n_bins
        bin_idx = np.clip(np.floor(r2_fine / dr2).astype(int), 0, n_bins - 1)
        counts = np.bincount(bin_idx, minlength=n_bins).astype(float)
        sums   = np.bincount(bin_idx, weights=profile_r2, minlength=n_bins)
        return np.where(counts > 0, sums / np.maximum(counts, 1.0), B)

    r_fine = np.linspace(0.0, r_max, n_fine)
    kw = dict(t=t, R_refl=R_refl, alpha=alpha, n=n, r_max=r_max, I0=I0, I1=I1, I2=I2)
    A1 = airy_ideal(r_fine, NE_WAVELENGTH_1_M, **kw)
    A2 = airy_ideal(r_fine, NE_WAVELENGTH_2_M, **kw)
    return np.interp(r_data, r_fine, A1 + NE_INTENSITY_2 * A2 + B)


def _compute_chi2(model: np.ndarray, data: np.ndarray,
                  sem: np.ndarray, n_params: int) -> float:
    """Compute chi²_reduced = Σ[(data-model)²/sem²] / (n_valid - n_params)."""
    valid = np.isfinite(data) & np.isfinite(sem) & (sem > 0)
    dof = max(int(valid.sum()) - n_params, 1)
    return float(np.sum(((data[valid] - model[valid]) / sem[valid]) ** 2) / dof)


def _compute_covariance(jac: np.ndarray, chi2_red: float) -> np.ndarray:
    """Compute covariance from least-squares Jacobian using SVD."""
    try:
        _, s, VT = _svd(jac, full_matrices=False)
        s = np.where(s > 1e-15 * s[0], s, 1e-15 * s[0])
        return (VT.T @ np.diag(1.0 / s ** 2) @ VT) * chi2_red
    except Exception:
        n = jac.shape[1]
        return np.full((n, n), np.nan)


def _run_trf_fit(r_data, profile_vals, sem_vals, valid,
                 free_params: dict, fixed_params: dict, bounds: dict,
                 max_nfev: int, ftol: float, xtol: float, gtol: float,
                 r_max: float, n_idx: float = 1.0, n_fine: int = 2000,
                 n_bins: int = None, n_subpixels: int = 8):
    """
    General TRF fit for any subset of the 10 Ne model parameters.
    Returns (result_params, covariance, chi2_red, stderr, param_names).
    """
    param_names = list(free_params.keys())
    x0 = np.array([free_params[nm] for nm in param_names])
    lb = np.array([bounds[nm][0] for nm in param_names])
    ub = np.array([bounds[nm][1] for nm in param_names])
    x0 = np.clip(x0, lb, ub)

    d_v = profile_vals[valid]
    s_v = sem_vals[valid]

    def get_all_p(x):
        p = dict(fixed_params)
        for i, nm in enumerate(param_names):
            p[nm] = float(x[i])
        return p

    if n_bins is not None:
        def residual(x):
            p = get_all_p(x)
            model = _ne_model(
                r_data, r_max,
                t=p['t_m'], R_refl=p['R_refl'], alpha=p['alpha'], n=n_idx,
                I0=p['I0'], I1=p['I1'], I2=p['I2'],
                sigma0=p['sigma0'], sigma1=p['sigma1'], sigma2=p['sigma2'],
                B=p['B'], n_fine=n_fine, n_bins=n_bins,
            )
            return (d_v - model[valid]) / s_v
    else:
        r_d = r_data[valid]
        def residual(x):
            p = get_all_p(x)
            model = _ne_model(
                r_d, r_max,
                t=p['t_m'], R_refl=p['R_refl'], alpha=p['alpha'], n=n_idx,
                I0=p['I0'], I1=p['I1'], I2=p['I2'],
                sigma0=p['sigma0'], sigma1=p['sigma1'], sigma2=p['sigma2'],
                B=p['B'], n_fine=n_fine,
            )
            return (d_v - model) / s_v

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        opt = least_squares(residual, x0, bounds=(lb, ub), method='trf',
                            ftol=ftol, xtol=xtol, gtol=gtol, max_nfev=max_nfev)

    result_params = get_all_p(opt.x)
    n_params = len(param_names)
    n_valid  = int(valid.sum())
    dof      = max(n_valid - n_params, 1)
    chi2_red = float(np.sum(opt.fun ** 2) / dof)

    cov    = _compute_covariance(opt.jac, chi2_red)
    stderr = np.sqrt(np.maximum(np.diag(cov), 0))
    return result_params, cov, chi2_red, stderr, param_names


def _get_valid_mask(profile: FringeProfile) -> np.ndarray:
    good = (~profile.masked) & np.isfinite(profile.profile) & \
           np.isfinite(profile.sigma_profile) & (profile.sigma_profile > 0)
    good &= profile.r_grid <= 0.97 * profile.r_max_px
    return good


def _get_all_bounds(config: FitConfig) -> dict:
    return {
        't_m':    config.effective_bounds('t_m'),
        'R_refl': config.effective_bounds('R'),
        'alpha':  config.effective_bounds('alpha'),
        'I0':     config.effective_bounds('I0'),
        'I1':     config.effective_bounds('I1'),
        'I2':     config.effective_bounds('I2'),
        'B':      config.effective_bounds('B'),
        'sigma0': config.effective_bounds('sigma0'),
        'sigma1': config.effective_bounds('sigma1'),
        'sigma2': config.effective_bounds('sigma2'),
    }


# ---------------------------------------------------------------------------
# Public shared model builder (unchanged — used by M06)
# ---------------------------------------------------------------------------

def build_model_binned(
    r2_edges: np.ndarray,
    wavelengths: np.ndarray,
    source_spectrum: np.ndarray,
    cal_result: "CalibrationResult",
    n_subpixels: int = 8,
    n_fine: int = 2000,
) -> np.ndarray:
    """
    Build Mulligan-matched model fringe profile matching M03 r²-uniform binning.
    Shared infrastructure for M05 and M06.
    """
    n_bins = len(r2_edges) - 1
    r_max  = float(np.sqrt(r2_edges[-1]))
    r_fine = np.linspace(0.0, r_max, n_fine)

    kw = dict(
        t=cal_result.t_m, R_refl=cal_result.R_refl, alpha=cal_result.alpha,
        n=1.0, r_max=r_max,
        I0=cal_result.I0, I1=cal_result.I1, I2=cal_result.I2,
    )
    sigma_arr = psf_sigma(
        r_fine, r_max, cal_result.sigma0, cal_result.sigma1, cal_result.sigma2
    )

    profile_fine = np.zeros(n_fine)
    for j, lam in enumerate(wavelengths):
        A_ideal = airy_ideal(r_fine, float(lam), **kw)
        A_psf   = _apply_psf_vectorized(A_ideal, r_fine, sigma_arr)
        profile_fine += A_psf * float(source_spectrum[j])

    r2_max    = float(r2_edges[-1])
    r2_uniform = np.linspace(0.0, r2_max, n_fine)
    profile_r2 = np.interp(np.sqrt(r2_uniform), r_fine, profile_fine)

    dr2     = r2_max / n_bins
    bin_idx = np.clip(np.floor(r2_uniform / dr2).astype(int), 0, n_bins - 1)
    counts  = np.bincount(bin_idx, minlength=n_bins).astype(float)
    sums    = np.bincount(bin_idx, weights=profile_r2, minlength=n_bins)
    return np.where(counts > 0, sums / np.maximum(counts, 1.0), 0.0)


# ===========================================================================
# NEW Stage P — Analytic alpha pre-solve from r²-peak spacing
# ===========================================================================

def _stageP_measure_alpha(
    profile: FringeProfile,
    config: FitConfig,
) -> tuple[float, int, int]:
    """
    Measure alpha analytically from the r²-spacing of bright (λ₁) fringe peaks.

    Physics basis
    -------------
    The Airy maximum condition gives peak radii:
        r_m² ≈ (1/alpha²) · [m·λ₁/(2t)]⁻² - 1) · [m·λ₁/(2t)]²  (exact)
    For adjacent orders the r²-spacing is:
        Δ(r²) ≈ λ₁ / (alpha² · t)          [independent of R, sigma0, I0, B]

    So: alpha = sqrt( λ₁ / (mean_Δr² · t) )

    Peak isolation
    --------------
    The radial profile contains BOTH λ₁ (bright) and λ₂ (dim) peaks, with λ₂
    peaks interleaved between each adjacent pair of λ₁ peaks.  Two measures
    select only the bright (λ₁) peaks:
      1. Height threshold at config.peak_height_percentile (default 78th pct) —
         λ₂ peaks are at most 0.8× the height of λ₁ peaks.
      2. Minimum separation = config.peak_distance_fraction × estimated FSR —
         set to ~55% of FSR so adjacent λ₁ peaks are found but not λ₂ peaks
         (which are half an FSR away).

    Returns
    -------
    alpha_measured : float  — alpha in rad/px, or config.alpha_init if failed
    n_peaks        : int    — number of bright peaks found
    quality_flag   : int    — 0 if OK, FitFlags.ALPHA_PRESOLVE_FAIL if < 4 peaks
    """
    vals = profile.profile.copy()
    good = _get_valid_mask(profile)
    vals[~good] = np.nan

    # 5-point smoothing to reduce noise before peak-finding
    kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    smooth = np.convolve(vals, kernel, mode='same')
    # Restore NaN at masked positions so they don't become spurious peaks
    smooth[~good] = np.nan

    # Height threshold — selects λ₁ bright peaks, rejects dimmer λ₂ peaks
    finite_vals = vals[good]
    if len(finite_vals) < 10:
        warnings.warn("Stage P: fewer than 10 valid bins — cannot measure alpha.",
                      FitWarning)
        return config.alpha_init, 0, FitFlags.ALPHA_PRESOLVE_FAIL

    height_threshold = float(np.nanpercentile(smooth, config.peak_height_percentile))

    # Estimate FSR in pixels at mid-range radius to set minimum separation.
    # FSR_r ≈ λ₁ / (2 · alpha_guess² · t · r_mid)
    r_mid = 0.5 * profile.r_max_px
    alpha_guess = config.alpha_init
    fsr_at_rmid = NE_WAVELENGTH_1_M / (2.0 * alpha_guess**2 * config.t_init_mm * 1e-3 * r_mid)
    min_separation = max(3, int(fsr_at_rmid * config.peak_distance_fraction
                                * len(profile.r_grid) / profile.r_max_px))

    peaks, _ = find_peaks(
        np.nan_to_num(smooth, nan=0.0),
        height=height_threshold,
        distance=min_separation,
    )

    # Filter to good bins only
    peaks = peaks[good[peaks]]

    if len(peaks) < 4:
        warnings.warn(
            f"Stage P: only {len(peaks)} bright peaks found (need ≥ 4). "
            f"Falling back to alpha_init = {config.alpha_init:.4e} rad/px. "
            "Consider adjusting peak_height_percentile or peak_distance_fraction.",
            FitWarning,
        )
        return config.alpha_init, len(peaks), FitFlags.ALPHA_PRESOLVE_FAIL

    r_peaks = profile.r_grid[peaks]
    r2_peaks = r_peaks ** 2
    n_arr = np.arange(len(r2_peaks), dtype=float)

    # Linear fit: r²[n] = r²[0] + n · Δr²  (Δr² = λ₁/(alpha²·t))
    slope, _ = np.polyfit(n_arr, r2_peaks, 1)

    if slope <= 0:
        warnings.warn(
            f"Stage P: r²-peak spacing slope = {slope:.2f} ≤ 0 — unexpected ordering. "
            f"Falling back to alpha_init.",
            FitWarning,
        )
        return config.alpha_init, len(peaks), FitFlags.ALPHA_PRESOLVE_FAIL

    t_icos = config.t_init_mm * 1e-3
    alpha_measured = float(np.sqrt(NE_WAVELENGTH_1_M / (slope * t_icos)))

    # Sanity-check: measured alpha must be within physics bounds
    alpha_lo, alpha_hi = PHYSICS_BOUNDS['alpha']
    if not (alpha_lo < alpha_measured < alpha_hi):
        warnings.warn(
            f"Stage P: measured alpha = {alpha_measured:.4e} is outside physics bounds "
            f"[{alpha_lo:.4e}, {alpha_hi:.4e}]. Falling back to alpha_init.",
            FitWarning,
        )
        return config.alpha_init, len(peaks), FitFlags.ALPHA_PRESOLVE_FAIL

    return alpha_measured, len(peaks), 0


# ===========================================================================
# Stage 1 — brute-force t scan (seeded with measured alpha from Stage P)
# ===========================================================================

def _stage1_brute_force_t(
    profile: FringeProfile,
    params_s0: dict,
    config: FitConfig,
    n_bins: int = None,
) -> tuple[float, float, float]:
    """
    Fine-step scan over t near t_init to find the correct FSR-period minimum.
    alpha is fixed at params_s0['alpha'] — which is now the Stage P measured
    value rather than a raw guess.

    Returns (t_best, I0_best, B_best).
    """
    eff_lo, eff_hi = config.effective_bounds('t_m')

    fsr_val    = NE_WAVELENGTH_1_M / 2.0
    scan_step  = fsr_val / 40.0
    window_half = fsr_val / 2.0

    t_center = params_s0['t_m']
    scan_lo = max(eff_lo, t_center - window_half)
    scan_hi = min(eff_hi, t_center + window_half)

    if scan_lo >= scan_hi:
        scan_lo, scan_hi = eff_lo, eff_hi
        n_scan = 2000
    else:
        n_scan = max(200, int((scan_hi - scan_lo) / scan_step) + 1)

    t_scan = np.linspace(scan_lo, scan_hi, n_scan)

    valid = _get_valid_mask(profile)
    d_v = profile.profile[valid]
    s_v = profile.sigma_profile[valid]

    _n_bins = n_bins if n_bins is not None else len(profile.r2_grid)
    p0 = params_s0

    I0_lo, I0_hi = config.effective_bounds('I0')
    B_lo,  B_hi  = config.effective_bounds('B')

    w     = 1.0 / (s_v ** 2)
    sum_w = float(np.sum(w))
    wd    = float(np.dot(w, d_v))

    chi2_arr = np.empty(n_scan)
    I0_arr   = np.empty(n_scan)
    B_arr    = np.empty(n_scan)

    for k, t_val in enumerate(t_scan):
        model_full = _ne_model(
            profile.r_grid, profile.r_max_px,
            t=t_val, R_refl=p0['R_refl'], alpha=p0['alpha'], n=1.0,
            I0=1.0, I1=0.0, I2=0.0,
            sigma0=p0['sigma0'], sigma1=0.0, sigma2=0.0,
            B=0.0, n_fine=2000, n_bins=_n_bins,
        )
        m = model_full[valid]

        wmm = float(np.dot(w, m * m))
        wm1 = float(np.dot(w, m))
        wmd = float(np.dot(w * m, d_v))

        det = wmm * sum_w - wm1 * wm1
        if det <= 0.0:
            chi2_arr[k] = np.inf
            I0_arr[k]   = p0['I0']
            B_arr[k]    = p0['B']
            continue

        I0_opt = float(np.clip((wmd * sum_w - wd  * wm1) / det, I0_lo, I0_hi))
        B_opt  = float(np.clip((wmm * wd   - wm1 * wmd) / det, B_lo,  B_hi))

        resid = (d_v - I0_opt * m - B_opt) / s_v
        chi2_arr[k] = float(np.dot(resid, resid))
        I0_arr[k]   = I0_opt
        B_arr[k]    = B_opt

    best_k  = int(np.argmin(chi2_arr))
    t_best  = float(t_scan[best_k])
    I0_best = float(I0_arr[best_k])
    B_best  = float(B_arr[best_k])

    fsr_half = NE_WAVELENGTH_1_M / 4.0
    if abs(t_best - params_s0['t_m']) > fsr_half:
        warnings.warn(
            f"Stage 1: t_best={t_best*1e6:.3f} µm deviates from t_init="
            f"{params_s0['t_m']*1e6:.3f} µm by "
            f"{abs(t_best-params_s0['t_m'])*1e9:.1f} nm > FSR/4="
            f"{fsr_half*1e9:.1f} nm. May be in wrong FSR period.",
            FitWarning,
        )

    return t_best, I0_best, B_best


# ===========================================================================
# Stage 2 — WLS amplitude solve: I0 and B analytically (no TRF needed)
# ===========================================================================

def _stage2_wls_amplitude(
    profile: FringeProfile,
    params_s1: dict,
    config: FitConfig,
) -> tuple[float, float]:
    """
    Solve I0 and B analytically via WLS given fixed geometry and PSF.

    The model is linear in I0 and B:
        data ≈ I0 · shape(r) + B
    where shape(r) is the normalised two-line Airy profile.

    This is exact — no iteration, no local-minimum risk.
    Returns (I0_opt, B_opt).
    """
    valid = _get_valid_mask(profile)
    d_v = profile.profile[valid]
    s_v = profile.sigma_profile[valid]

    p = params_s1
    _n_bins = len(profile.r2_grid)

    shape_full = _ne_model(
        profile.r_grid, profile.r_max_px,
        t=p['t_m'], R_refl=p['R_refl'], alpha=p['alpha'], n=1.0,
        I0=1.0, I1=0.0, I2=0.0,
        sigma0=p['sigma0'], sigma1=0.0, sigma2=0.0,
        B=0.0, n_fine=2000, n_bins=_n_bins,
    )
    m = shape_full[valid]

    w     = 1.0 / (s_v ** 2)
    sum_w = float(np.sum(w))
    wd    = float(np.dot(w, d_v))
    wmm   = float(np.dot(w, m * m))
    wm1   = float(np.dot(w, m))
    wmd   = float(np.dot(w * m, d_v))

    det = wmm * sum_w - wm1 * wm1
    if det <= 0.0:
        warnings.warn("Stage 2: WLS det ≤ 0, using initial I0/B.", FitWarning)
        return p['I0'], p['B']

    I0_lo, I0_hi = config.effective_bounds('I0')
    B_lo,  B_hi  = config.effective_bounds('B')

    I0_opt = float(np.clip((wmd * sum_w - wd  * wm1) / det, I0_lo, I0_hi))
    B_opt  = float(np.clip((wmm * wd   - wm1 * wmd) / det, B_lo,  B_hi))

    return I0_opt, B_opt


# ===========================================================================
# Stage 3 — R only, using small-r bins (PSF-free zone)
# ===========================================================================

def _stage3_R_small_r(
    profile: FringeProfile,
    params_s2: dict,
    config: FitConfig,
    chi2_prev: float,
) -> tuple[dict, np.ndarray, float, np.ndarray, list]:
    """
    Fit R using only bins with r < config.r_R_cutoff_px.

    At small r, the fringe FWHM_Airy is much larger than FWHM_PSF
    (because fringe spacing is wide at small radius), so R is
    independently constrained without confusion from sigma0.

    Free:  {R_refl}
    Fixed: {t_m, alpha, I0, B, I1=0, I2=0, sigma0, sigma1=0, sigma2=0}
    """
    # Build a validity mask restricted to small-r bins
    base_valid = _get_valid_mask(profile)
    small_r    = profile.r_grid < config.r_R_cutoff_px
    valid      = base_valid & small_r

    if valid.sum() < 3:
        warnings.warn(
            f"Stage 3: only {valid.sum()} valid bins with r < {config.r_R_cutoff_px:.0f} px. "
            "Skipping Stage 3 R-fit; R remains at initial value.",
            FitWarning,
        )
        result = dict(params_s2)
        n_bins = len(profile.r2_grid)
        model = _ne_model(
            profile.r_grid, profile.r_max_px,
            t=result['t_m'], R_refl=result['R_refl'], alpha=result['alpha'], n=1.0,
            I0=result['I0'], I1=0.0, I2=0.0,
            sigma0=result['sigma0'], sigma1=0.0, sigma2=0.0,
            B=result['B'], n_fine=2000, n_bins=n_bins,
        )
        chi2 = _compute_chi2(model[base_valid], profile.profile[base_valid],
                             profile.sigma_profile[base_valid], n_params=1)
        return result, np.full((1,1), np.nan), chi2, np.array([1e-6]), ['R_refl']

    bounds = _get_all_bounds(config)
    p = params_s2

    free  = {'R_refl': p['R_refl']}
    fixed = {
        't_m':    p['t_m'],
        'alpha':  p['alpha'],
        'I0':     p['I0'],
        'I1':     0.0, 'I2': 0.0,
        'sigma0': p['sigma0'], 'sigma1': 0.0, 'sigma2': 0.0,
        'B':      p['B'],
    }

    n_bins = len(profile.r2_grid)
    res, cov, chi2, stderr, pnames = _run_trf_fit(
        profile.r_grid, profile.profile, profile.sigma_profile, valid,
        free_params=free, fixed_params=fixed, bounds=bounds,
        max_nfev=config.max_nfev, ftol=config.ftol, xtol=config.xtol,
        gtol=config.gtol, r_max=profile.r_max_px, n_fine=2000, n_bins=n_bins,
    )

    if stderr is None or not np.isfinite(stderr[0]):
        warnings.warn("Stage 3: stderr not finite for R.", FitWarning)

    return res, cov, chi2, stderr, pnames


# ===========================================================================
# Stage 4 — sigma0 only, using large-r bins (PSF-dominant zone)
# ===========================================================================

def _stage4_sigma0_large_r(
    profile: FringeProfile,
    params_s3: dict,
    config: FitConfig,
    chi2_prev: float,
) -> tuple[dict, np.ndarray, float, np.ndarray, list]:
    """
    Fit sigma0 using only bins with r > config.r_s0_cutoff_px.

    At large r, the fringe spacing has compressed so that FWHM_PSF is
    a significant fraction of the fringe spacing.  The radial trend of
    peak broadening is almost entirely sigma0.  R is fixed from Stage 3.

    Free:  {sigma0}
    Fixed: {t_m, R_refl, alpha, I0, B, I1=0, I2=0, sigma1=0, sigma2=0}
    """
    base_valid = _get_valid_mask(profile)
    large_r    = profile.r_grid > config.r_s0_cutoff_px
    valid      = base_valid & large_r

    if valid.sum() < 5:
        warnings.warn(
            f"Stage 4: only {valid.sum()} valid bins with r > {config.r_s0_cutoff_px:.0f} px. "
            "Skipping Stage 4 sigma0-fit; sigma0 remains at initial value.",
            FitWarning,
        )
        result = dict(params_s3)
        n_bins = len(profile.r2_grid)
        model = _ne_model(
            profile.r_grid, profile.r_max_px,
            t=result['t_m'], R_refl=result['R_refl'], alpha=result['alpha'], n=1.0,
            I0=result['I0'], I1=0.0, I2=0.0,
            sigma0=result['sigma0'], sigma1=0.0, sigma2=0.0,
            B=result['B'], n_fine=2000, n_bins=n_bins,
        )
        chi2 = _compute_chi2(model[base_valid], profile.profile[base_valid],
                             profile.sigma_profile[base_valid], n_params=1)
        return result, np.full((1,1), np.nan), chi2, np.array([1e-6]), ['sigma0']

    bounds = _get_all_bounds(config)
    p = params_s3

    free  = {'sigma0': p['sigma0']}
    fixed = {
        't_m':    p['t_m'],
        'R_refl': p['R_refl'],
        'alpha':  p['alpha'],
        'I0':     p['I0'],
        'I1':     p.get('I1', 0.0), 'I2': p.get('I2', 0.0),
        'sigma1': 0.0, 'sigma2': 0.0,
        'B':      p['B'],
    }

    n_bins = len(profile.r2_grid)
    res, cov, chi2, stderr, pnames = _run_trf_fit(
        profile.r_grid, profile.profile, profile.sigma_profile, valid,
        free_params=free, fixed_params=fixed, bounds=bounds,
        max_nfev=config.max_nfev, ftol=config.ftol, xtol=config.xtol,
        gtol=config.gtol, r_max=profile.r_max_px, n_fine=2000, n_bins=n_bins,
    )

    if res.get('sigma0', 0) <= 0.01:
        warnings.warn(f"Stage 4: sigma0={res['sigma0']:.4f} px ≤ 0.01 px.", FitWarning)

    return res, cov, chi2, stderr, pnames


# ===========================================================================
# Stage 5 — WLS envelope shape: I1, I2 (linear given everything else)
# ===========================================================================

def _stage5_wls_envelope(
    profile: FringeProfile,
    params_s4: dict,
    config: FitConfig,
) -> tuple[float, float, float, float]:
    """
    Solve I1, I2 analytically via WLS.

    The intensity envelope I(r) = I0·(1 + I1·(r/rmax) + I2·(r/rmax)²) is
    linear in I1 and I2 once the fringe shape is known.  Build the design
    matrix [shape·(r/rmax), shape·(r/rmax)²] and solve WLS.

    Returns (I0_new, I1_opt, I2_opt, B_new) — I0 and B are re-solved
    simultaneously so amplitude is consistent with the new envelope shape.
    """
    valid = _get_valid_mask(profile)
    d_v   = profile.profile[valid]
    s_v   = profile.sigma_profile[valid]
    r_v   = profile.r_grid[valid]
    rmax  = profile.r_max_px

    p = params_s4
    _n_bins = len(profile.r2_grid)

    # Build shape columns: Airy evaluated with I0=1, I1=0, I2=0 (baseline shape)
    # and with radial weighting factors for I1 and I2 terms.
    # Column 0: I0 contribution (shape with flat envelope)
    # Column 1: I1 contribution (shape × r/rmax)
    # Column 2: I2 contribution (shape × (r/rmax)²)
    # Column 3: B (constant 1)

    # Base shape — normalised Airy (I0=1, envelope flat)
    def _shape(I0_in, I1_in, I2_in):
        full = _ne_model(
            profile.r_grid, rmax,
            t=p['t_m'], R_refl=p['R_refl'], alpha=p['alpha'], n=1.0,
            I0=I0_in, I1=I1_in, I2=I2_in,
            sigma0=p['sigma0'], sigma1=0.0, sigma2=0.0,
            B=0.0, n_fine=2000, n_bins=_n_bins,
        )
        return full[valid]

    # Pure Airy shape with unit amplitude and flat envelope
    s_base = _shape(1.0, 0.0, 0.0)
    # r/rmax weighting at each bin
    rn = r_v / rmax
    # I1 column: Airy weighted by r/rmax  (I0=0, I1=1, I2=0 effectively)
    # We approximate by multiplying s_base by rn — exact for the linear term
    s_I1 = s_base * rn
    s_I2 = s_base * rn**2

    # Design matrix A: columns [s_base, s_I1, s_I2, ones]
    # data ≈ A @ [I0, I1·I0, I2·I0, B]  (I1,I2 appear multiplied by I0 in the model)
    # Re-parameterise: x = [I0, c1=I0·I1, c2=I0·I2, B]
    A = np.column_stack([s_base, s_I1, s_I2, np.ones(len(d_v))])
    W = np.diag(1.0 / s_v**2)

    try:
        AtWA = A.T @ W @ A
        AtWd = A.T @ W @ d_v
        x, _, _, _ = np.linalg.lstsq(AtWA, AtWd, rcond=None)
        I0_new, c1, c2, B_new = x
        I0_new = float(np.clip(I0_new, *config.effective_bounds('I0')))
        B_new  = float(np.clip(B_new,  *config.effective_bounds('B')))
        # Recover I1, I2 from c1=I0·I1 and c2=I0·I2
        I1_opt = float(np.clip(c1 / I0_new if I0_new > 0 else 0.0,
                               *config.effective_bounds('I1')))
        I2_opt = float(np.clip(c2 / I0_new if I0_new > 0 else 0.0,
                               *config.effective_bounds('I2')))
    except np.linalg.LinAlgError:
        warnings.warn("Stage 5: WLS envelope solve failed. Keeping I1=I2=0.", FitWarning)
        I0_new = p['I0']
        I1_opt = 0.0
        I2_opt = 0.0
        B_new  = p['B']

    return I0_new, I1_opt, I2_opt, B_new


# ===========================================================================
# Stage 6 — PSF shape: sigma1, sigma2
# ===========================================================================

def _stage6_psf_shape(
    profile: FringeProfile,
    params_s5: dict,
    config: FitConfig,
    chi2_prev: float,
) -> tuple[dict, np.ndarray, float, np.ndarray, list]:
    """
    Fit sigma1, sigma2 (radial PSF variation) with all other parameters fixed.

    Free:  {sigma1, sigma2}
    Fixed: {t_m, R_refl, alpha, I0, I1, I2, B, sigma0}
    """
    valid  = _get_valid_mask(profile)
    bounds = _get_all_bounds(config)
    p = params_s5

    free  = {'sigma1': 0.0, 'sigma2': 0.0}
    fixed = {
        't_m':    p['t_m'],
        'R_refl': p['R_refl'],
        'alpha':  p['alpha'],
        'I0':     p['I0'],
        'I1':     p.get('I1', 0.0), 'I2': p.get('I2', 0.0),
        'sigma0': p['sigma0'],
        'B':      p['B'],
    }

    n_bins = len(profile.r2_grid)
    res, cov, chi2, stderr, pnames = _run_trf_fit(
        profile.r_grid, profile.profile, profile.sigma_profile, valid,
        free_params=free, fixed_params=fixed, bounds=bounds,
        max_nfev=config.max_nfev, ftol=config.ftol, xtol=config.xtol,
        gtol=config.gtol, r_max=profile.r_max_px, n_fine=2000, n_bins=n_bins,
    )

    # Check PSF remains positive
    r = profile.r_grid
    sigma_r = psf_sigma(r, profile.r_max_px,
                        p['sigma0'], res.get('sigma1', 0.0), res.get('sigma2', 0.0))
    if np.any(sigma_r < 0.01):
        warnings.warn("Stage 6: σ(r) < 0.01 px — PSF shape unphysical. "
                      "Reverting sigma1=sigma2=0.", FitWarning)
        res['sigma1'] = 0.0
        res['sigma2'] = 0.0

    return res, cov, chi2, stderr, pnames


# ===========================================================================
# Stage 7 — Full free optimisation (all 10 parameters), with epsilon
# ===========================================================================

def _stage7_full_free(
    profile: FringeProfile,
    params_s6: dict,
    config: FitConfig,
    chi2_s6: float,
) -> tuple:
    """
    Final 10-parameter free optimisation.  Seeded from Stages 1–6.
    Then computes epsilon_cal and epsilon_sci via fpi_airy_fit.

    Returns (result_params, cov, chi2_red, stderr, pnames,
             epsilon_cal, epsilon_sci, m0_cal, m0_sci).
    """
    valid  = _get_valid_mask(profile)
    bounds = _get_all_bounds(config)
    p = params_s6

    free = {
        't_m':    p['t_m'],
        'R_refl': p['R_refl'],
        'alpha':  p['alpha'],
        'I0':     p['I0'],
        'I1':     p.get('I1', 0.0),
        'I2':     p.get('I2', 0.0),
        'B':      p['B'],
        'sigma0': p.get('sigma0', 0.5),
        'sigma1': p.get('sigma1', 0.0),
        'sigma2': p.get('sigma2', 0.0),
    }
    fixed = {}

    n_bins = len(profile.r2_grid)
    res, cov, chi2, stderr, pnames = _run_trf_fit(
        profile.r_grid, profile.profile, profile.sigma_profile, valid,
        free_params=free, fixed_params=fixed, bounds=bounds,
        max_nfev=config.max_nfev, ftol=config.ftol, xtol=config.xtol,
        gtol=config.gtol, r_max=profile.r_max_px, n_fine=2000, n_bins=n_bins,
        n_subpixels=config.n_subpixels,
    )

    if stderr is None:
        raise RuntimeError("Stage 7: stderr is None — TRF did not converge.")

    for i, nm in enumerate(pnames):
        if not np.isfinite(stderr[i]):
            warnings.warn(f"Stage 7: stderr[{nm}]={stderr[i]:.3g} not finite.", FitWarning)

    if chi2 > chi2_s6 * 1.05:
        warnings.warn(f"Stage 7: chi2_red={chi2:.3f} > stage6 {chi2_s6:.3f}.", FitWarning)

    # Compute epsilon via fpi_airy_fit
    t_m7  = res['t_m']
    eff_t = config.effective_bounds('t_m')
    cal_cfg = _CalibrationFitConfig(
        instrument=_InstrumentConfig(focal_length_mm=200.0, pixel_pitch_um=32.0),
        d_mm_init=t_m7 * 1e3,
        R_init=float(res['R_refl']),
        d_mm_bounds=(eff_t[0] * 1e3, eff_t[1] * 1e3),
        R_bounds=config.effective_bounds('R'),
    )

    class _RadialProfile:
        def __init__(self, fp: FringeProfile):
            gd = ~fp.masked & np.isfinite(fp.profile) & \
                 np.isfinite(fp.sigma_profile) & (fp.sigma_profile > 0)
            self.bin_centers_r2 = fp.r2_grid
            self.mean_I = fp.profile.copy()
            self.mean_I[~gd] = np.nan
            self.sem_I = fp.sigma_profile.copy()
            self.sem_I[~gd] = np.nan
            self.bin_centers_r = fp.r_grid

    try:
        airy_result = _fit_calibration_profile(_RadialProfile(profile), cal_cfg)
        epsilon_cal = float(airy_result.epsilon_ref)
        epsilon_sci = float(airy_result.epsilon_sci)
        m0_cal      = float(airy_result.m0)
        m0_sci      = float(airy_result.m0_sci)
    except Exception as e:
        warnings.warn(
            f"Stage 7: fpi_airy_fit.fit_calibration_profile failed ({e}). "
            "Computing epsilon directly from t_m.",
            FitWarning,
        )
        m0_cal      = 2.0 * t_m7 / NE_WAVELENGTH_1_M
        epsilon_cal = float(m0_cal % 1.0)
        m0_sci      = 2.0 * t_m7 / OI_WAVELENGTH_M
        epsilon_sci = float(m0_sci % 1.0)

    return res, cov, chi2, stderr, pnames, epsilon_cal, epsilon_sci, m0_cal, m0_sci


# ---------------------------------------------------------------------------
# Convergence guard (unchanged logic, updated stage reference)
# ---------------------------------------------------------------------------

def _convergence_guard(
    profile: FringeProfile,
    stage7_result: dict,
    stage7_stderr: np.ndarray,
    stage7_pnames: list,
    config: FitConfig,
    perturbation_scale: float = 0.05,
) -> bool:
    """
    Re-run Stage 7 from N perturbed starting points.
    Returns True if all runs converge to the same minimum (within 3σ).
    """
    valid  = _get_valid_mask(profile)
    bounds = _get_all_bounds(config)
    n_pert = config.n_convergence_perturbations
    rng    = np.random.default_rng(7777)

    all_agree = True
    for _ in range(n_pert):
        free_pert = {}
        for i, nm in enumerate(stage7_pnames):
            val     = stage7_result[nm]
            perturb = rng.normal(0.0, perturbation_scale * abs(val) + 1e-10)
            lo, hi  = bounds[nm]
            free_pert[nm] = float(np.clip(val + perturb, lo, hi))

        try:
            n_bins = len(profile.r2_grid)
            res_pert, _, _, se_pert, pn_pert = _run_trf_fit(
                profile.r_grid, profile.profile, profile.sigma_profile, valid,
                free_params=free_pert, fixed_params={}, bounds=bounds,
                max_nfev=config.max_nfev, ftol=config.ftol, xtol=config.xtol,
                gtol=config.gtol, r_max=profile.r_max_px, n_fine=2000, n_bins=n_bins,
                n_subpixels=config.n_subpixels,
            )
        except Exception:
            all_agree = False
            continue

        for i, nm in enumerate(stage7_pnames):
            sigma_i = float(stage7_stderr[i]) if np.isfinite(stage7_stderr[i]) else 0.0
            diff    = abs(res_pert.get(nm, stage7_result[nm]) - stage7_result[nm])
            if sigma_i > 0 and diff > 3.0 * sigma_i:
                all_agree = False
                break

    return all_agree


# ---------------------------------------------------------------------------
# Post-fit verification (unchanged logic)
# ---------------------------------------------------------------------------

def _post_fit_verification(
    result: CalibrationResult,
    profile: FringeProfile,
    t_init_m: float,
    config: FitConfig,
) -> int:
    """Run all post-fit checks. Returns quality_flags bitmask."""
    flags = FitFlags.GOOD

    # V1 — Correct FSR period
    fsr_half = NE_WAVELENGTH_1_M / 4.0
    if abs(result.t_m - t_init_m) > fsr_half:
        flags |= FitFlags.STAGE1_WRONG_PERIOD
        warnings.warn("V1 FAILED: t may be in wrong FSR period.", FitWarning)

    # V2 — Correlation matrix degeneracy
    if result.covariance is not None and np.all(np.isfinite(result.covariance)):
        corr = result.correlation
        if abs(corr[0, 2]) > 0.98:
            flags |= FitFlags.T_ALPHA_DEGENERATE
        if abs(corr[1, 7]) > 0.95:
            flags |= FitFlags.R_SIGMA_DEGENERATE

    # V2 physical — PSF positive
    r = profile.r_grid
    sigma_r = psf_sigma(r, profile.r_max_px, result.sigma0, result.sigma1, result.sigma2)
    if np.any(sigma_r < 0.01):
        flags |= FitFlags.PSF_UNPHYSICAL
        warnings.warn("PSF σ(r) < 0.01 px at some radii.", FitWarning)

    # V3 — Parameters at bounds
    for pname, pkey in [('t_m', 't_m'), ('R_refl', 'R'),
                         ('alpha', 'alpha'), ('sigma0', 'sigma0')]:
        val    = getattr(result, pname)
        lo, hi = config.effective_bounds(pkey)
        if abs(val - lo) < 1e-10 * (hi - lo) or abs(val - hi) < 1e-10 * (hi - lo):
            flags |= FitFlags.PARAM_AT_BOUND
            warnings.warn(f"Parameter '{pname}' = {val:.6g} is at its bound.", FitWarning)

    # V4 — Chi-squared range
    if result.chi2_reduced > 2.0:
        flags |= FitFlags.CHI2_HIGH
        warnings.warn(f"chi2_red = {result.chi2_reduced:.2f} > 2.0", FitWarning)
    if result.chi2_reduced < 0.5:
        flags |= FitFlags.CHI2_LOW
        warnings.warn(f"chi2_red = {result.chi2_reduced:.2f} < 0.5", FitWarning)

    # V5 — Stage P presolve agreement
    if np.isfinite(result.alpha_presolve):
        alpha_discrepancy = abs(result.alpha - result.alpha_presolve) / result.alpha_presolve
        if alpha_discrepancy > 0.05:
            warnings.warn(
                f"Final alpha = {result.alpha:.4e} differs from Stage P measurement "
                f"{result.alpha_presolve:.4e} by {100*alpha_discrepancy:.1f}% > 5%. "
                "This suggests sigma0 or R may still be absorbing alpha error.",
                FitWarning,
            )

    return flags


# ===========================================================================
# Top-level entry point
# ===========================================================================

def fit_calibration_fringe(
    profile: FringeProfile,
    config: FitConfig = None,
) -> CalibrationResult:
    """
    Full staged calibration inversion of a WindCube neon lamp FringeProfile.

    Redesigned 8-stage sequence (P, 1, 2, 3, 4, 5, 6, 7) compared to the
    original 6-stage version.  The key improvement is Stage P: an analytic
    pre-solve of alpha from the r²-peak spacing, which breaks the alpha-R-sigma0
    correlation trap in the original architecture.

    Parameters
    ----------
    profile : FringeProfile from M03 annular_reduce()
    config  : FitConfig.  If None, uses all defaults.

    Returns
    -------
    CalibrationResult — all 10 parameters with uncertainties, phase
    reference ε_cal and ε_sci, alpha_presolve diagnostic, and quality flags.

    Raises
    ------
    ValueError   : profile has < 30 valid bins
    RuntimeError : Stage 7 TRF did not converge (stderr = None)
    FitError     : require_convergence_guard=True and guard failed
    """
    if config is None:
        config = FitConfig()

    valid = _get_valid_mask(profile)
    if valid.sum() < 30:
        raise ValueError(
            f"Profile has only {valid.sum()} valid bins; need at least 30."
        )

    chi2_by_stage = []

    # ------------------------------------------------------------------
    # Stage P — Analytic alpha pre-solve (no optimiser)
    # ------------------------------------------------------------------
    alpha_measured, n_peaks, presolve_flag = _stageP_measure_alpha(profile, config)

    # Initial parameter dictionary — alpha seeded from Stage P measurement
    vals = profile.profile.copy()
    good = ~profile.masked & np.isfinite(vals)
    I0_est = float(np.percentile(vals[good], 75)) if good.sum() > 5 else 1000.0
    B_est  = float(np.percentile(vals[good], 5))  if good.sum() > 5 else 300.0
    if config.dark_level is not None:
        B_est = config.dark_level
    if config.B_init is not None:
        B_est = config.B_init

    params = {
        't_m':    config.t_init_mm * 1e-3,
        'R_refl': config.R_init,
        'alpha':  alpha_measured,          # from Stage P — not a guess
        'I0':     max(I0_est - B_est, 100.0),
        'I1':     0.0,
        'I2':     0.0,
        'sigma0': config.sigma0_init,
        'sigma1': 0.0,
        'sigma2': 0.0,
        'B':      B_est,
    }

    # Stage P chi² (no-PSF model at initial estimates)
    n_bins = len(profile.r2_grid)
    model_p = _ne_model_nopsf(
        profile.r_grid, profile.r_max_px,
        t=params['t_m'], R_refl=params['R_refl'], alpha=params['alpha'],
        n=1.0, I0=params['I0'], I1=0.0, I2=0.0,
        B=params['B'], n_fine=2000, n_bins=n_bins,
    )
    chi2_p = _compute_chi2(model_p[valid], profile.profile[valid],
                           profile.sigma_profile[valid], n_params=3)
    chi2_by_stage.append(float(chi2_p))

    # ------------------------------------------------------------------
    # Stage 1 — brute-force t scan (alpha fixed at Stage P value)
    # ------------------------------------------------------------------
    t_best, I0_best, B_best = _stage1_brute_force_t(profile, params, config, n_bins=n_bins)
    params = dict(params, t_m=t_best, I0=I0_best, B=B_best)

    model_1 = _ne_model(
        profile.r_grid, profile.r_max_px,
        t=t_best, R_refl=params['R_refl'], alpha=params['alpha'], n=1.0,
        I0=params['I0'], I1=0.0, I2=0.0,
        sigma0=params['sigma0'], sigma1=0.0, sigma2=0.0,
        B=params['B'], n_fine=2000, n_bins=n_bins,
    )
    chi2_1 = _compute_chi2(model_1[valid], profile.profile[valid],
                           profile.sigma_profile[valid], n_params=4)
    chi2_by_stage.append(float(chi2_1))

    # ------------------------------------------------------------------
    # Stage 2 — WLS amplitude: I0, B analytically
    # ------------------------------------------------------------------
    I0_wls, B_wls = _stage2_wls_amplitude(profile, params, config)
    params = dict(params, I0=I0_wls, B=B_wls)

    model_2 = _ne_model(
        profile.r_grid, profile.r_max_px,
        t=params['t_m'], R_refl=params['R_refl'], alpha=params['alpha'], n=1.0,
        I0=params['I0'], I1=0.0, I2=0.0,
        sigma0=params['sigma0'], sigma1=0.0, sigma2=0.0,
        B=params['B'], n_fine=2000, n_bins=n_bins,
    )
    chi2_2 = _compute_chi2(model_2[valid], profile.profile[valid],
                           profile.sigma_profile[valid], n_params=4)
    chi2_by_stage.append(float(chi2_2))

    # ------------------------------------------------------------------
    # Stage 3 — R only, small-r bins
    # ------------------------------------------------------------------
    res3, _, chi2_3, _, _ = _stage3_R_small_r(profile, params, config, chi2_2)
    params = dict(params, R_refl=res3['R_refl'])
    chi2_by_stage.append(float(chi2_3))

    # ------------------------------------------------------------------
    # Stage 4 — sigma0 only, large-r bins
    # ------------------------------------------------------------------
    res4, _, chi2_4, _, _ = _stage4_sigma0_large_r(profile, params, config, chi2_3)
    params = dict(params, sigma0=res4['sigma0'])
    chi2_by_stage.append(float(chi2_4))

    # ------------------------------------------------------------------
    # Stage 5 — WLS envelope shape: I0, I1, I2, B
    # ------------------------------------------------------------------
    I0_env, I1_env, I2_env, B_env = _stage5_wls_envelope(profile, params, config)
    params = dict(params, I0=I0_env, I1=I1_env, I2=I2_env, B=B_env)

    model_5 = _ne_model(
        profile.r_grid, profile.r_max_px,
        t=params['t_m'], R_refl=params['R_refl'], alpha=params['alpha'], n=1.0,
        I0=params['I0'], I1=params['I1'], I2=params['I2'],
        sigma0=params['sigma0'], sigma1=0.0, sigma2=0.0,
        B=params['B'], n_fine=2000, n_bins=n_bins,
    )
    chi2_5 = _compute_chi2(model_5[valid], profile.profile[valid],
                           profile.sigma_profile[valid], n_params=7)
    chi2_by_stage.append(float(chi2_5))

    # ------------------------------------------------------------------
    # Stage 6 — PSF shape: sigma1, sigma2
    # ------------------------------------------------------------------
    res6, _, chi2_6, _, _ = _stage6_psf_shape(profile, params, config, chi2_5)
    params = dict(params, sigma1=res6.get('sigma1', 0.0), sigma2=res6.get('sigma2', 0.0))
    chi2_by_stage.append(float(chi2_6))

    # ------------------------------------------------------------------
    # Stage 7 — Full free (all 10 parameters)
    # ------------------------------------------------------------------
    (res7, cov7, chi2_7, se7, pn7,
     epsilon_cal, epsilon_sci, m0_cal, m0_sci) = _stage7_full_free(
        profile, params, config, chi2_6)
    chi2_by_stage.append(float(chi2_7))

    if se7 is None:
        raise RuntimeError("Stage 7 returned None stderr.")

    # ------------------------------------------------------------------
    # Build 10×10 covariance in canonical parameter order
    # ------------------------------------------------------------------
    cov10 = np.full((10, 10), np.nan)
    for i, ni in enumerate(pn7):
        ki = _PARAM_NAMES.index(ni)
        for j, nj in enumerate(pn7):
            kj = _PARAM_NAMES.index(nj)
            cov10[ki, kj] = cov7[i, j]

    se10 = np.sqrt(np.maximum(np.diag(cov10), 0))
    se10 = np.where(np.isfinite(se10) & (se10 > 0), se10, 1e-30)

    diag_std = np.sqrt(np.maximum(np.diag(cov10), 0))
    outer    = np.outer(diag_std, diag_std)
    with np.errstate(invalid='ignore', divide='ignore'):
        corr10 = np.where(outer > 0, cov10 / outer, 0.0)

    p7 = res7

    def _se(name):
        idx = _PARAM_NAMES.index(name)
        v   = float(se10[idx])
        return v if v > 0 else 1e-30

    n_valid_bins = int(valid.sum())
    n_params_free = 10

    result = CalibrationResult(
        t_m=float(p7['t_m']),
        R_refl=float(p7['R_refl']),
        alpha=float(p7['alpha']),
        I0=float(p7['I0']),
        I1=float(p7['I1']),
        I2=float(p7['I2']),
        B=float(p7['B']),
        sigma0=float(p7['sigma0']),
        sigma1=float(p7['sigma1']),
        sigma2=float(p7['sigma2']),

        sigma_t_m=_se('t_m'),
        sigma_R_refl=_se('R_refl'),
        sigma_alpha=_se('alpha'),
        sigma_I0=_se('I0'),
        sigma_I1=_se('I1'),
        sigma_I2=_se('I2'),
        sigma_B=_se('B'),
        sigma_sigma0=_se('sigma0'),
        sigma_sigma1=_se('sigma1'),
        sigma_sigma2=_se('sigma2'),

        two_sigma_t_m=2.0 * _se('t_m'),
        two_sigma_R_refl=2.0 * _se('R_refl'),
        two_sigma_alpha=2.0 * _se('alpha'),
        two_sigma_I0=2.0 * _se('I0'),
        two_sigma_I1=2.0 * _se('I1'),
        two_sigma_I2=2.0 * _se('I2'),
        two_sigma_B=2.0 * _se('B'),
        two_sigma_sigma0=2.0 * _se('sigma0'),
        two_sigma_sigma1=2.0 * _se('sigma1'),
        two_sigma_sigma2=2.0 * _se('sigma2'),

        epsilon_cal=float(epsilon_cal),
        epsilon_sci=float(epsilon_sci),
        m0_cal=float(m0_cal),
        m0_sci=float(m0_sci),

        alpha_presolve=float(alpha_measured),
        n_peaks_presolve=int(n_peaks),

        chi2_reduced=float(chi2_7),
        n_bins_used=n_valid_bins,
        n_params_free=n_params_free,
        covariance=cov10,
        correlation=corr10,
        converged=True,
        quality_flags=FitFlags.GOOD | presolve_flag,

        chi2_by_stage=chi2_by_stage,
        fit_config=config,
    )

    # ------------------------------------------------------------------
    # Convergence guard
    # ------------------------------------------------------------------
    guard_ok = _convergence_guard(profile, p7, se7, pn7, config)
    if not guard_ok:
        result.quality_flags |= FitFlags.MULTIPLE_MINIMA
        warnings.warn(
            "Convergence guard: different minimum found from perturbed start. "
            "Inspect this frame carefully.",
            FitWarning,
        )
        if config.require_convergence_guard:
            validate_result_uncertainties(result)
            raise FitError("Convergence guard failed and require_convergence_guard=True.")

    # ------------------------------------------------------------------
    # Post-fit verification
    # ------------------------------------------------------------------
    result.quality_flags = _post_fit_verification(
        result, profile, config.t_init_mm * 1e-3, config)
    if not guard_ok:
        result.quality_flags |= FitFlags.MULTIPLE_MINIMA

    # ------------------------------------------------------------------
    # Final uncertainty validation
    # ------------------------------------------------------------------
    validate_result_uncertainties(result)
    return result
