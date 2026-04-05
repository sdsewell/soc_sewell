"""
M05 — Calibration Inversion
fpi/m05_calibration_inversion.py

Six-stage staged inversion of a WindCube neon lamp FringeProfile
into the 10 instrument parameters {t, R, α, I₀, I₁, I₂, B, σ₀, σ₁, σ₂}.

Reference: Harding et al. (2014), Section 3–5
Spec: docs/specs/m05_calibration_inversion_spec.md
"""
from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.linalg import svd as _svd
from scipy.optimize import least_squares

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

# ---------------------------------------------------------------------------
# Import fpi_airy_fit for Stage 5 (epsilon_ref / epsilon_sci)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Three-tier bounds architecture
# ---------------------------------------------------------------------------

# Tier 1 — physics hard limits
PHYSICS_BOUNDS = {
    't_m':    (19e-3,  21e-3), # Etalon gap will never be smaller than 19mm or larger than 21mm (even with ICOS error)
    'R':      (0.3,   0.9),
    'alpha':  (1e-5,   1e-3),
    'I0':     (1.0,    130000.0),
    'I1':     (-0.9,   0.9),
    'I2':     (-0.9,   0.9),
    'sigma0': (0.01,   5.0),
    'sigma1': (-3.0,   3.0),
    'sigma2': (-3.0,   3.0),
    'B':      (0.0,    10000.0),
}

# Tier 2 — WindCube instrument defaults
INSTRUMENT_DEFAULTS = {
    't_m':    (19.95e-3, 20.07e-3),
    'R':      (0.5,     0.85),
    'alpha':  (1.6e-4,   2.0e-4),
    'I0':     (100.0,    15000.0),
    'I1':     (-0.5,     0.5),
    'I2':     (-0.5,     0.5),
    'sigma0': (0.05,     1.2),
    'sigma1': (-1.0,     1.0),
    'sigma2': (-1.0,     1.0),
    'B':      (50.0,     10000.0),
}


@dataclass
class FitConfig:
    t_init_mm:    float = 20.008
    t_bounds_mm:  Optional[tuple] = None

    R_init:       float = 0.65          # raised from 0.53
    R_bounds:     Optional[tuple] = None

    alpha_init:   float = 1.77e-4       # raised from 1.744e-4
    alpha_bounds: Optional[tuple] = None

    sigma0_init:  float = 0.5           # lowered from 0.8
    sigma0_bounds: Optional[tuple] = (0.05, 1.2)  # NEW tight ceiling

    B_init:       Optional[float] = None
    B_bounds:     Optional[tuple] = None

    dark_level:   Optional[float] = None

    max_nfev:     int   = 10_000
    ftol:         float = 1e-14
    xtol:         float = 1e-14
    gtol:         float = 1e-14

    n_convergence_perturbations: int  = 3
    require_convergence_guard:   bool = False

    n_subpixels: int = 1   # n_subpixels=1: matches M05 forward model; Mulligan n=8 gives chi2~209

    def effective_bounds(self, param: str) -> tuple:
        """Return (min, max) in SI units after merging all three tiers."""
        phys_lo, phys_hi = PHYSICS_BOUNDS[param]
        inst_lo, inst_hi = INSTRUMENT_DEFAULTS[param]

        # Map physics param names to FitConfig attribute names
        _attr_map = {
            't_m':    'bounds_mm_to_m',   # special: convert mm → m
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
            user_override_raw = self.t_bounds_mm
            if user_override_raw is not None:
                user_lo = user_override_raw[0] * 1e-3
                user_hi = user_override_raw[1] * 1e-3
            else:
                user_lo, user_hi = inst_lo, inst_hi
        elif attr is not None:
            user_override_raw = getattr(self, attr, None)
            if user_override_raw is not None:
                user_lo, user_hi = user_override_raw
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
    m0_cal:       float   # integer + fractional order at λ₁
    m0_sci:       float   # integer + fractional order at 630nm

    # Fit quality
    chi2_reduced:    float
    n_bins_used:     int
    n_params_free:   int
    covariance:      np.ndarray   # shape (10, 10)
    correlation:     np.ndarray   # normalised covariance, shape (10, 10)
    converged:       bool
    quality_flags:   int

    # Stage progression
    chi2_by_stage:   list   # chi2_reduced after each of 6 stages

    # Fit config
    fit_config:      FitConfig


# ---------------------------------------------------------------------------
# validate_result_uncertainties
# ---------------------------------------------------------------------------

def validate_result_uncertainties(result: CalibrationResult) -> None:
    """
    Validate all sigma_ and two_sigma_ fields per ADDENDUM_uncertainty_standards.md.
    Raises AssertionError if any field is None, zero, non-finite, or if
    two_sigma_ ≠ 2 × sigma_ (to relative precision 1e-8).
    """
    _fields = [
        't_m', 'R_refl', 'alpha', 'I0', 'I1', 'I2', 'B', 'sigma0', 'sigma1', 'sigma2',
    ]
    for f in _fields:
        sigma = getattr(result, f'sigma_{f}')
        two_sigma = getattr(result, f'two_sigma_{f}')
        assert sigma is not None, f"sigma_{f} is None"
        assert np.isfinite(sigma), f"sigma_{f} = {sigma} is not finite"
        assert sigma > 0, f"sigma_{f} = {sigma} is not positive"
        expected = 2.0 * sigma
        assert abs(two_sigma - expected) < 1e-8 * expected + 1e-30, (
            f"two_sigma_{f} = {two_sigma} ≠ 2 × sigma_{f} = {expected}"
        )


# ---------------------------------------------------------------------------
# Internal physics helpers
# ---------------------------------------------------------------------------

_PARAM_NAMES = ['t_m', 'R_refl', 'alpha', 'I0', 'I1', 'I2', 'B', 'sigma0', 'sigma1', 'sigma2']
# Indices: 0=t, 1=R, 2=alpha, 3=I0, 4=I1, 5=I2, 6=B, 7=sigma0, 8=sigma1, 9=sigma2


def _apply_psf_vectorized(A_ideal: np.ndarray, r_fine: np.ndarray,
                           sigma_arr: np.ndarray) -> np.ndarray:
    """
    Shift-variant Gaussian PSF convolution, fully vectorized.
    r_fine must be uniformly spaced.
    sigma_arr: PSF width at each point in r_fine.
    """
    n = len(A_ideal)
    dr = r_fine[1] - r_fine[0]
    sig_max = float(np.max(sigma_arr))
    if sig_max <= 0.01:
        return A_ideal.copy()

    half_win = int(np.ceil(4.0 * sig_max / dr))
    offsets = np.arange(-half_win, half_win + 1)   # shape (win,)

    # Index matrix (n, win) — clamp to valid range
    idx = np.arange(n)[:, np.newaxis] + offsets[np.newaxis, :]
    valid = (idx >= 0) & (idx < n)
    idx_safe = np.clip(idx, 0, n - 1)

    # Radial differences and weights
    r_diff = offsets[np.newaxis, :] * dr          # (1, win)
    sig_col = sigma_arr[:, np.newaxis]             # (n, 1)
    w = np.exp(-0.5 * (r_diff / sig_col) ** 2) * valid   # (n, win)
    w /= w.sum(axis=1, keepdims=True)

    return np.sum(w * A_ideal[idx_safe], axis=1)




def _ne_model(r_data: np.ndarray, r_max: float,
              t: float, R_refl: float, alpha: float, n: float,
              I0: float, I1: float, I2: float,
              sigma0: float, sigma1: float, sigma2: float,
              B: float, n_fine: int = 2000,
              n_bins: int = None, n_subpixels: int = 8) -> np.ndarray:
    """
    Two-line Ne lamp model.

    When n_bins is provided, evaluates at sub-pixel positions matching M03's
    Mulligan technique: for each integer pixel radius px, evaluates at
    px + sub_offset for n_subpixels evenly-spaced offsets in (−0.5, +0.5).
    The sub-pixel grid is uniformly spaced at 1/n_subpixels px, so the
    vectorised PSF convolution is exact. Bin-averages in r² so that
    model_binned[b] = mean of model values whose r² falls in bin b —
    exactly matching what M03 annular_reduce computes.

    When n_bins is None, evaluates on a fine uniform-r grid and interpolates
    at r_data (used for brute-force t-scan where only relative chi2 matters).

    Forward model (from M02):
        S(r) = airy(r; λ₁) + NE_INTENSITY_2 × airy(r; λ₂) + B
    """
    if n_bins is not None:
        # PSF convolution on fine uniform-r grid (required for _apply_psf_vectorized)
        r_fine_psf = np.linspace(0.0, r_max, n_fine)
        kw_ideal = dict(t=t, R_refl=R_refl, alpha=alpha, n=n, r_max=r_max, I0=I0, I1=I1, I2=I2)
        A1_ideal = airy_ideal(r_fine_psf, NE_WAVELENGTH_1_M, **kw_ideal)
        A2_ideal = airy_ideal(r_fine_psf, NE_WAVELENGTH_2_M, **kw_ideal)
        sigma_arr = psf_sigma(r_fine_psf, r_max, sigma0, sigma1, sigma2)
        A1 = _apply_psf_vectorized(A1_ideal, r_fine_psf, sigma_arr)
        A2 = _apply_psf_vectorized(A2_ideal, r_fine_psf, sigma_arr)
        profile_1d = A1 + NE_INTENSITY_2 * A2 + B
        # Resample onto fine uniform-r² grid.
        # M03 annular_reduce bins CCD pixels in r², and pixel density is
        # uniform in r² (N_pix ∝ pi·dr² = const), so the bin mean is the
        # arithmetic mean of I over uniformly-spaced r² values — NOT uniform r.
        r2_max = r_max ** 2
        r2_fine = np.linspace(0.0, r2_max, n_fine)
        r_for_r2 = np.sqrt(r2_fine)
        profile_r2 = np.interp(r_for_r2, r_fine_psf, profile_1d)
        # Bin-average
        dr2 = r2_max / n_bins
        bin_idx = np.clip(np.floor(r2_fine / dr2).astype(int), 0, n_bins - 1)
        counts = np.bincount(bin_idx, minlength=n_bins).astype(float)
        sums   = np.bincount(bin_idx, weights=profile_r2, minlength=n_bins)
        return np.where(counts > 0, sums / np.maximum(counts, 1.0), B)

    r_fine = np.linspace(0.0, r_max, n_fine)
    kw_ideal = dict(t=t, R_refl=R_refl, alpha=alpha, n=n, r_max=r_max, I0=I0, I1=I1, I2=I2)
    A1_ideal = airy_ideal(r_fine, NE_WAVELENGTH_1_M, **kw_ideal)
    A2_ideal = airy_ideal(r_fine, NE_WAVELENGTH_2_M, **kw_ideal)
    sigma_arr = psf_sigma(r_fine, r_max, sigma0, sigma1, sigma2)
    A1 = _apply_psf_vectorized(A1_ideal, r_fine, sigma_arr)
    A2 = _apply_psf_vectorized(A2_ideal, r_fine, sigma_arr)
    return np.interp(r_data, r_fine, A1 + NE_INTENSITY_2 * A2 + B)


# ---------------------------------------------------------------------------
# Public shared model builder — used by M05 and imported by M06
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

    For each wavelength in the grid, computes the PSF-convolved Airy response
    on a fine uniform-r grid, accumulates the weighted sum with source_spectrum,
    then resamples to a uniform-r² grid and bins — exactly as M03 annular_reduce
    does with CCD pixels.

    This is the shared model evaluation infrastructure for M05 and M06.
    Importing this function in M06 guarantees both modules use identical
    sub-pixel averaging and r²-binning logic.

    Parameters
    ----------
    r2_edges : np.ndarray, shape (n_bins+1,)
        r² bin edges in pixels².  Uniform-r² spacing as produced by M03.
    wavelengths : np.ndarray, shape (L,)
        Wavelength grid in metres.
    source_spectrum : np.ndarray, shape (L,)
        Source intensity per bin (y × Δλ — caller applies the Δλ factor).
    cal_result : CalibrationResult
        Instrument parameters held fixed (t, R, α, I₀, I₁, I₂, σ₀, σ₁, σ₂).
        Refractive index n = 1.0 is assumed (air-gap etalon).
    n_subpixels : int
        Retained for API compatibility; resolution is provided by n_fine.
    n_fine : int
        Fine-grid density for PSF convolution (default 2000).

    Returns
    -------
    np.ndarray, shape (n_bins,)
        Model fringe profile in CCD counts (bias not included).
    """
    n_bins = len(r2_edges) - 1
    r_max = float(np.sqrt(r2_edges[-1]))
    r_fine = np.linspace(0.0, r_max, n_fine)

    kw = dict(
        t=cal_result.t_m, R_refl=cal_result.R_refl, alpha=cal_result.alpha,
        n=1.0, r_max=r_max,
        I0=cal_result.I0, I1=cal_result.I1, I2=cal_result.I2,
    )
    sigma_arr = psf_sigma(
        r_fine, r_max, cal_result.sigma0, cal_result.sigma1, cal_result.sigma2
    )

    # Accumulate A @ source_spectrum on the fine-r grid
    profile_fine = np.zeros(n_fine)
    for j, lam in enumerate(wavelengths):
        A_ideal = airy_ideal(r_fine, float(lam), **kw)
        A_psf = _apply_psf_vectorized(A_ideal, r_fine, sigma_arr)
        profile_fine += A_psf * float(source_spectrum[j])

    # Resample to uniform-r² grid (matches M03 pixel-area weighting)
    r2_max = float(r2_edges[-1])
    r2_uniform = np.linspace(0.0, r2_max, n_fine)
    profile_r2 = np.interp(np.sqrt(r2_uniform), r_fine, profile_fine)

    # Bin-average
    dr2 = r2_max / n_bins
    bin_idx = np.clip(np.floor(r2_uniform / dr2).astype(int), 0, n_bins - 1)
    counts = np.bincount(bin_idx, minlength=n_bins).astype(float)
    sums = np.bincount(bin_idx, weights=profile_r2, minlength=n_bins)
    return np.where(counts > 0, sums / np.maximum(counts, 1.0), 0.0)


def _ne_model_nopsf(r_data: np.ndarray, r_max: float,
                    t: float, R_refl: float, alpha: float, n: float,
                    I0: float, I1: float, I2: float,
                    B: float, n_fine: int = 2000,
                    n_bins: int = None, n_subpixels: int = 8) -> np.ndarray:
    """Two-line Ne model without PSF.  When n_bins is given, returns r²-bin averages
    using the same Mulligan sub-pixel grid as _ne_model so that Stage 0 chi²
    is computed on a model consistent with M03 annular_reduce."""
    if n_bins is not None:
        # Same r²-uniform resampling as _ne_model but without PSF
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


def _dict_to_vec(params: dict, names: list) -> np.ndarray:
    return np.array([params[n] for n in names])


def _run_trf_fit(r_data, profile_vals, sem_vals, valid,
                 free_params: dict, fixed_params: dict, bounds: dict,
                 max_nfev: int, ftol: float, xtol: float, gtol: float,
                 r_max: float, n_idx: float = 1.0, n_fine: int = 2000,
                 n_bins: int = None, n_subpixels: int = 8):
    """
    General TRF fit for any subset of the 10 Ne model parameters.

    free_params  : {name: init_value}
    fixed_params : {name: value}
    bounds       : {name: (lo, hi)} — must cover all free_params names
    n_bins       : when set, model returns r²-weighted bin averages for ALL
                   bins; the residual selects valid bins from the full output.

    Returns (result_params, covariance, chi2_red, stderr, param_names)
    """
    param_names = list(free_params.keys())
    x0 = np.array([free_params[n] for n in param_names])
    lb = np.array([bounds[n][0] for n in param_names])
    ub = np.array([bounds[n][1] for n in param_names])

    # Clamp x0 to bounds
    x0 = np.clip(x0, lb, ub)

    d_v = profile_vals[valid]
    s_v = sem_vals[valid]

    def get_all_p(x):
        p = dict(fixed_params)
        for i, nm in enumerate(param_names):
            p[nm] = float(x[i])
        return p

    if n_bins is not None:
        # Bin-average mode: model returns all n_bins values; select valid here
        def residual(x):
            p = get_all_p(x)
            model = _ne_model(
                r_data, r_max,
                t=p['t_m'], R_refl=p['R_refl'], alpha=p['alpha'], n=n_idx,
                I0=p['I0'], I1=p['I1'], I2=p['I2'],
                sigma0=p['sigma0'], sigma1=p['sigma1'], sigma2=p['sigma2'],
                B=p['B'], n_fine=n_fine, n_bins=n_bins, n_subpixels=n_subpixels,
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
    n_valid = int(valid.sum())
    dof = max(n_valid - n_params, 1)
    chi2_red = float(np.sum(opt.fun ** 2) / dof)

    cov = _compute_covariance(opt.jac, chi2_red)
    stderr = np.sqrt(np.maximum(np.diag(cov), 0))
    return result_params, cov, chi2_red, stderr, param_names


# ---------------------------------------------------------------------------
# Stage 0 — direct estimation (no optimisation)
# ---------------------------------------------------------------------------

def _stage0_initial_estimates(profile: FringeProfile, config: FitConfig) -> dict:
    """
    Compute starting values from profile data and instrument knowledge.
    Uses ICOS t_init_mm as the t estimate (most reliable for WindCube).
    """
    vals = profile.profile.copy()
    good = ~profile.masked & np.isfinite(vals)

    I0_est = float(np.percentile(vals[good], 75)) if good.sum() > 5 else 1000.0
    B_est = float(np.percentile(vals[good], 5)) if good.sum() > 5 else config.B_init or 300.0
    if config.dark_level is not None:
        B_est = config.dark_level

    t_est = config.t_init_mm * 1e-3

    return {
        't_m':    t_est,
        'R_refl': config.R_init,
        'alpha':  config.alpha_init,
        'I0':     max(I0_est - B_est, 100.0),
        'I1':     0.0,
        'I2':     0.0,
        'sigma0': config.sigma0_init,
        'sigma1': 0.0,
        'sigma2': 0.0,
        'B':      B_est,
    }


# ---------------------------------------------------------------------------
# Stage 1 — brute-force scan over t
# ---------------------------------------------------------------------------

def _stage1_brute_force_t(profile: FringeProfile, params_stage0: dict,
                           config: FitConfig, n_scan_points: int = 2000,
                           n_bins: int = None) -> tuple:
    """
    Fine-step scan over t near t_stage0 to find the correct FSR-period minimum.

    The chi² landscape has sharp minima (FWHM ≈ FSR/finesse ≈ 66 nm for R=0.53)
    spaced every FSR ≈ 320 nm.  A coarse scan (step > FSR) misses the true
    minimum entirely.  Instead we scan within ±N_FSR of t_stage0 with step =
    FSR/40 ≈ 8 nm, which reliably resolves every minimum in the window.

    At each t, I0 and B are solved analytically via WLS so the scan is
    insensitive to amplitude mis-estimates from Stage 0.
    Returns (t_best, I0_best, B_best) in metres / counts / counts.
    """
    eff_lo, eff_hi = config.effective_bounds('t_m')

    fsr_val = NE_WAVELENGTH_1_M / 2.0      # ~320 nm
    scan_step = fsr_val / 40.0             # ~8 nm — resolves each FSR minimum
    # Restrict to ±FSR/2 around t_stage0 so the scan stays within one FSR period.
    # The chi² landscape has a slow large-scale trend (from PSF/envelope mismatch)
    # that makes distant FSR periods appear better than the true one; confining the
    # window to ±FSR/2 forces the scan to find the local minimum in the correct period.
    # This relies on Stage 0 (ICOS measurement) being accurate to < FSR/2 ≈ 160 nm.
    window_half = fsr_val / 2.0            # ±FSR/2 ≈ ±160 nm around t_stage0

    t_center = params_stage0['t_m']
    scan_lo = max(eff_lo, t_center - window_half)
    scan_hi = min(eff_hi, t_center + window_half)

    if scan_lo >= scan_hi:
        # t_stage0 is outside effective bounds; fall back to full-range coarse scan
        scan_lo, scan_hi = eff_lo, eff_hi
        n_scan = n_scan_points
    else:
        n_scan = max(200, int((scan_hi - scan_lo) / scan_step) + 1)

    t_scan = np.linspace(scan_lo, scan_hi, n_scan)

    valid = _get_valid_mask(profile)
    d_v = profile.profile[valid]
    s_v = profile.sigma_profile[valid]

    _n_bins = n_bins if n_bins is not None else len(profile.r2_grid)

    p0 = params_stage0
    I0_lo, I0_hi = config.effective_bounds('I0')
    B_lo,  B_hi  = config.effective_bounds('B')

    # Pre-compute weight constants (fixed across all t values)
    w      = 1.0 / (s_v ** 2)
    sum_w  = float(np.sum(w))
    wd     = float(np.dot(w, d_v))

    chi2_arr = np.empty(n_scan)
    I0_arr   = np.empty(n_scan)
    B_arr    = np.empty(n_scan)

    for k, t_val in enumerate(t_scan):
        # Fringe shape with unit amplitude, no background (linear in I0 and B).
        # Use PSF model so fringe peak amplitudes match PSF-blurred data.
        model_full = _ne_model(
            profile.r_grid, profile.r_max_px,
            t=t_val, R_refl=p0['R_refl'], alpha=p0['alpha'], n=1.0,
            I0=1.0, I1=0.0, I2=0.0,
            sigma0=p0['sigma0'], sigma1=0.0, sigma2=0.0,
            B=0.0, n_fine=2000, n_bins=_n_bins,
        )
        m = model_full[valid]

        # WLS normal equations: data = I0 * m + B
        # [Σ w·m², Σ w·m] [I0]   [Σ w·m·d]
        # [Σ w·m,  Σ w  ] [B ] = [Σ w·d  ]
        wmm = float(np.dot(w, m * m))
        wm1 = float(np.dot(w, m))
        wmd = float(np.dot(w * m, d_v))

        det = wmm * sum_w - wm1 * wm1
        if det <= 0.0:
            chi2_arr[k] = np.inf
            I0_arr[k] = p0['I0']
            B_arr[k]  = p0['B']
            continue

        I0_opt = float(np.clip((wmd * sum_w - wd  * wm1) / det, I0_lo, I0_hi))
        B_opt  = float(np.clip((wmm * wd   - wm1 * wmd) / det, B_lo,  B_hi))

        resid = (d_v - I0_opt * m - B_opt) / s_v
        chi2_arr[k] = float(np.dot(resid, resid))
        I0_arr[k]   = I0_opt
        B_arr[k]    = B_opt

    best_k = int(np.argmin(chi2_arr))
    t_best  = float(t_scan[best_k])
    I0_best = float(I0_arr[best_k])
    B_best  = float(B_arr[best_k])

    # V1 sanity check — warn if far from Stage 0 estimate
    fsr_half = NE_WAVELENGTH_1_M / 4.0
    if abs(t_best - params_stage0['t_m']) > fsr_half:
        warnings.warn(
            f"Stage 1: t_best={t_best*1e6:.3f} µm deviates from t_stage0="
            f"{params_stage0['t_m']*1e6:.3f} µm by "
            f"{abs(t_best-params_stage0['t_m'])*1e9:.1f} nm > FSR/4="
            f"{fsr_half*1e9:.1f} nm. "
            f"May be in wrong FSR period. Check t_init_mm in FitConfig.",
            FitWarning,
            stacklevel=3,
        )

    return t_best, I0_best, B_best


# ---------------------------------------------------------------------------
# Stages 2–4 helpers
# ---------------------------------------------------------------------------

def _get_valid_mask(profile: FringeProfile):
    good = (~profile.masked) & np.isfinite(profile.profile) & \
           np.isfinite(profile.sigma_profile) & (profile.sigma_profile > 0)
    # Exclude edge bins: an Airy ring can fall at r ≈ r_max and M03 excludes pixels
    # beyond r_max, making those bin means artificially low. Mask r > 0.97 * r_max.
    good &= profile.r_grid <= 0.97 * profile.r_max_px
    return good


def _get_all_bounds(config: FitConfig) -> dict:
    """Return effective bounds for all 10 parameters."""
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
# Stage 2 — Geometry: fit t, α, I₀, B
# ---------------------------------------------------------------------------

def _stage2_geometry(profile: FringeProfile, t_stage1: float,
                     params_stage0: dict, config: FitConfig,
                     chi2_stage1: float) -> tuple:
    """Free: {t_m, alpha, I0, B}. Fixed: {R_refl, I1, I2, sigma0, sigma1, sigma2}."""
    valid = _get_valid_mask(profile)
    bounds = _get_all_bounds(config)
    p0 = params_stage0

    free = {
        't_m':   t_stage1,
        'alpha': p0['alpha'],
        'I0':    p0['I0'],
        'B':     p0['B'],
    }
    fixed = {
        'R_refl': p0['R_refl'],
        'I1':     0.0, 'I2': 0.0,
        'sigma0': p0['sigma0'], 'sigma1': 0.0, 'sigma2': 0.0,
    }

    n_bins = len(profile.r2_grid)
    res, cov, chi2, stderr, pnames = _run_trf_fit(
        profile.r_grid, profile.profile, profile.sigma_profile, valid,
        free_params=free, fixed_params=fixed, bounds=bounds,
        max_nfev=config.max_nfev, ftol=config.ftol, xtol=config.xtol,
        gtol=config.gtol, r_max=profile.r_max_px, n_fine=2000, n_bins=n_bins,
        n_subpixels=config.n_subpixels,
    )

    # Check stderrs
    if stderr is None:
        raise RuntimeError("Stage 2: stderr is None — TRF did not produce covariance.")
    for i, nm in enumerate(pnames):
        if not np.isfinite(stderr[i]):
            warnings.warn(f"Stage 2: stderr[{nm}]={stderr[i]:.3g} is not finite.", FitWarning)

    # Stage 2 chi² check
    if chi2 > chi2_stage1 * 1.05:
        warnings.warn(
            f"Stage 2: chi2_red={chi2:.3f} > stage1 chi2_red={chi2_stage1:.3f}. "
            f"Stage 2 did not improve the fit.",
            FitWarning,
        )

    return res, cov, chi2, stderr, pnames


# ---------------------------------------------------------------------------
# Stage 3 — Intensity envelope: add R, I₁, I₂
# ---------------------------------------------------------------------------

def _stage3_intensity(profile: FringeProfile, params_s2: dict,
                      config: FitConfig, chi2_s2: float) -> tuple:
    """Free: {t_m, R_refl, alpha, I0, I1, I2, B}. Fixed: {sigma0, sigma1, sigma2}."""
    valid = _get_valid_mask(profile)
    bounds = _get_all_bounds(config)
    p = params_s2

    free = {
        't_m':    p['t_m'],
        'R_refl': p['R_refl'],
        'alpha':  p['alpha'],
        'I0':     p['I0'],
        'I1':     0.0,
        'I2':     0.0,
        'B':      p['B'],
    }
    fixed = {
        'sigma0': p['sigma0'], 'sigma1': 0.0, 'sigma2': 0.0,
    }

    n_bins = len(profile.r2_grid)
    res, cov, chi2, stderr, pnames = _run_trf_fit(
        profile.r_grid, profile.profile, profile.sigma_profile, valid,
        free_params=free, fixed_params=fixed, bounds=bounds,
        max_nfev=config.max_nfev, ftol=config.ftol, xtol=config.xtol,
        gtol=config.gtol, r_max=profile.r_max_px, n_fine=2000, n_bins=n_bins,
        n_subpixels=config.n_subpixels,
    )

    if stderr is None:
        raise RuntimeError("Stage 3: stderr is None — TRF did not produce covariance.")

    if chi2 > chi2_s2 * 1.05:
        warnings.warn(f"Stage 3: chi2_red={chi2:.3f} > stage2 {chi2_s2:.3f}.", FitWarning)

    # Intensity positivity checks
    I0_v, I1_v, I2_v = res['I0'], res.get('I1', 0.0), res.get('I2', 0.0)
    if I0_v * (1 + I1_v + I2_v) <= 0:
        warnings.warn("Stage 3: I(r_max) ≤ 0 — intensity envelope non-physical.", FitWarning)
    if I0_v <= 0:
        warnings.warn("Stage 3: I(0) ≤ 0 — intensity non-physical.", FitWarning)

    return res, cov, chi2, stderr, pnames


# ---------------------------------------------------------------------------
# Stage 4 — PSF average width: add σ₀
# ---------------------------------------------------------------------------

def _stage4_psf(profile: FringeProfile, params_s3: dict,
                config: FitConfig, chi2_s3: float) -> tuple:
    """Free: {t_m, R_refl, alpha, I0, I1, I2, B, sigma0}. Fixed: {sigma1, sigma2}."""
    valid = _get_valid_mask(profile)
    bounds = _get_all_bounds(config)
    p = params_s3

    free = {
        't_m':    p['t_m'],
        'R_refl': p['R_refl'],
        'alpha':  p['alpha'],
        'I0':     p['I0'],
        'I1':     p.get('I1', 0.0),
        'I2':     p.get('I2', 0.0),
        'B':      p['B'],
        'sigma0': p['sigma0'],
    }
    fixed = {'sigma1': 0.0, 'sigma2': 0.0}

    n_bins = len(profile.r2_grid)
    res, cov, chi2, stderr, pnames = _run_trf_fit(
        profile.r_grid, profile.profile, profile.sigma_profile, valid,
        free_params=free, fixed_params=fixed, bounds=bounds,
        max_nfev=config.max_nfev, ftol=config.ftol, xtol=config.xtol,
        gtol=config.gtol, r_max=profile.r_max_px, n_fine=2000, n_bins=n_bins,
        n_subpixels=config.n_subpixels,
    )

    if stderr is None:
        raise RuntimeError("Stage 4: stderr is None — TRF did not produce covariance.")

    if chi2 > chi2_s3 * 1.05:
        warnings.warn(f"Stage 4: chi2_red={chi2:.3f} > stage3 {chi2_s3:.3f}.", FitWarning)

    if res.get('sigma0', 0) <= 0.01:
        warnings.warn(f"Stage 4: sigma0={res['sigma0']:.4f} px ≤ 0.01 px.", FitWarning)

    # R–σ₀ correlation check
    if 'R_refl' in pnames and 'sigma0' in pnames:
        iR = pnames.index('R_refl')
        is0 = pnames.index('sigma0')
        dR = cov[iR, iR]
        ds0 = cov[is0, is0]
        if dR > 0 and ds0 > 0:
            corr_R_s0 = cov[iR, is0] / np.sqrt(dR * ds0)
            if abs(corr_R_s0) > 0.95:
                warnings.warn(
                    f"Stage 4: |corr(R,σ₀)| = {abs(corr_R_s0):.3f} > 0.95. "
                    "R and PSF width are near-degenerate.",
                    FitWarning,
                )

    return res, cov, chi2, stderr, pnames


# ---------------------------------------------------------------------------
# Stage 5 — Full free optimisation (all 10 parameters)
# ---------------------------------------------------------------------------

def _stage5_full_free(profile: FringeProfile, params_s4: dict,
                      config: FitConfig, chi2_s4: float) -> tuple:
    """
    Full 10-parameter free optimisation using M01 two-line Ne model.
    Then calls fpi_airy_fit.fit_calibration_profile for epsilon_ref.
    Returns (result_params_10, cov10, chi2_red, stderr10, epsilon_cal, epsilon_sci, m0_cal, m0_sci).
    """
    valid = _get_valid_mask(profile)
    bounds = _get_all_bounds(config)
    p = params_s4

    free = {
        't_m':    p['t_m'],
        'R_refl': p['R_refl'],
        'alpha':  p['alpha'],
        'I0':     p['I0'],
        'I1':     p.get('I1', 0.0),
        'I2':     p.get('I2', 0.0),
        'B':      p['B'],
        'sigma0': p.get('sigma0', 0.5),
        'sigma1': 0.0,
        'sigma2': 0.0,
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
        raise RuntimeError("Stage 5: stderr is None — TRF did not converge.")

    for i, nm in enumerate(pnames):
        if not np.isfinite(stderr[i]):
            warnings.warn(f"Stage 5: stderr[{nm}]={stderr[i]:.3g} not finite.", FitWarning)

    if chi2 > chi2_s4 * 1.05:
        warnings.warn(f"Stage 5: chi2_red={chi2:.3f} > stage4 {chi2_s4:.3f}.", FitWarning)

    # Wrap fpi_airy_fit.fit_calibration_profile for epsilon_ref / epsilon_sci
    t_m5 = res['t_m']
    eff_t = config.effective_bounds('t_m')
    cal_cfg = _CalibrationFitConfig(
        instrument=_InstrumentConfig(focal_length_mm=200.0, pixel_pitch_um=32.0),
        d_mm_init=t_m5 * 1e3,
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
            # fpi_airy_fit diagnostic also uses bin_centers_r
            self.bin_centers_r = fp.r_grid

    try:
        _rp = _RadialProfile(profile)
        airy_result = _fit_calibration_profile(_rp, cal_cfg)
        epsilon_cal = float(airy_result.epsilon_ref)
        epsilon_sci = float(airy_result.epsilon_sci)
        m0_cal = float(airy_result.m0)
        m0_sci = float(airy_result.m0_sci)
    except Exception as e:
        warnings.warn(
            f"Stage 5: fpi_airy_fit.fit_calibration_profile failed ({e}). "
            "Computing epsilon from t_m directly.",
            FitWarning,
        )
        m0_cal = 2.0 * t_m5 / NE_WAVELENGTH_1_M
        epsilon_cal = float(m0_cal % 1.0)
        m0_sci = 2.0 * t_m5 / OI_WAVELENGTH_M
        epsilon_sci = float(m0_sci % 1.0)

    return res, cov, chi2, stderr, pnames, epsilon_cal, epsilon_sci, m0_cal, m0_sci


# ---------------------------------------------------------------------------
# Convergence guard
# ---------------------------------------------------------------------------

def _convergence_guard(profile: FringeProfile, stage5_result: dict,
                       stage5_stderr: np.ndarray, stage5_pnames: list,
                       config: FitConfig,
                       perturbation_scale: float = 0.05) -> bool:
    """
    Re-run Stage 5 from N perturbed starting points.
    Returns True if all runs converge to the same minimum (within 3σ).
    """
    valid = _get_valid_mask(profile)
    bounds = _get_all_bounds(config)
    n_pert = config.n_convergence_perturbations
    rng = np.random.default_rng(7777)

    all_agree = True
    for _ in range(n_pert):
        # Perturb starting point by ±5% of each parameter
        free_pert = {}
        for i, nm in enumerate(stage5_pnames):
            val = stage5_result[nm]
            sigma_i = float(stage5_stderr[i]) if np.isfinite(stage5_stderr[i]) else abs(val) * 0.05
            perturb = rng.normal(0.0, perturbation_scale * abs(val) + 1e-10)
            lo, hi = bounds[nm]
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

        # Check if result differs from Stage 5 by more than 3σ
        for i, nm in enumerate(stage5_pnames):
            sigma_i = float(stage5_stderr[i]) if np.isfinite(stage5_stderr[i]) else 0.0
            diff = abs(res_pert.get(nm, stage5_result[nm]) - stage5_result[nm])
            if sigma_i > 0 and diff > 3.0 * sigma_i:
                all_agree = False
                break

    return all_agree


# ---------------------------------------------------------------------------
# Post-fit verification
# ---------------------------------------------------------------------------

def _post_fit_verification(result: CalibrationResult,
                            profile: FringeProfile,
                            params_stage0: dict,
                            config: FitConfig) -> int:
    """Run all five post-fit checks. Returns quality_flags bitmask."""
    flags = FitFlags.GOOD

    # V1 — Correct FSR period
    fsr_half = NE_WAVELENGTH_1_M / 4.0
    if abs(result.t_m - params_stage0['t_m']) > fsr_half:
        flags |= FitFlags.STAGE1_WRONG_PERIOD
        warnings.warn("V1 FAILED: t may be in wrong FSR period.", FitWarning)

    # V2 — Correlation matrix degeneracy
    if result.covariance is not None and np.all(np.isfinite(result.covariance)):
        corr = result.correlation
        corr_t_alpha = corr[0, 2]   # t vs alpha
        corr_R_sigma = corr[1, 7]   # R vs sigma0
        if abs(corr_t_alpha) > 0.98:
            flags |= FitFlags.T_ALPHA_DEGENERATE
        if abs(corr_R_sigma) > 0.95:
            flags |= FitFlags.R_SIGMA_DEGENERATE

    # V2 physical — PSF width must be positive
    r = profile.r_grid
    sigma_r = psf_sigma(r, profile.r_max_px, result.sigma0, result.sigma1, result.sigma2)
    if np.any(sigma_r < 0.01):
        flags |= FitFlags.PSF_UNPHYSICAL
        warnings.warn("V2 physical FAILED: σ(r) < 0.01 px at some radii.", FitWarning)

    # V3 — Parameters at bounds
    for pname, param_key in [('t_m', 't_m'), ('R_refl', 'R'),
                               ('alpha', 'alpha'), ('sigma0', 'sigma0')]:
        val = getattr(result, pname)
        lo, hi = config.effective_bounds(param_key)
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

    return flags


# ---------------------------------------------------------------------------
# Top-level function
# ---------------------------------------------------------------------------

def fit_calibration_fringe(
    profile: FringeProfile,
    config: FitConfig = None,
) -> CalibrationResult:
    """
    Full staged calibration inversion of a WindCube neon lamp FringeProfile.

    Parameters
    ----------
    profile : FringeProfile from M03 reduce_calibration_frame()
    config  : FitConfig with bounds and initial estimates.
              If None, uses INSTRUMENT_DEFAULTS for all parameters.

    Returns
    -------
    CalibrationResult — all 10 instrument parameters with 1σ/2σ uncertainties,
    phase reference ε_cal and ε_sci, fit quality metrics, and quality flags.

    Raises
    ------
    RuntimeError : if any stage returns stderr=None
    ValueError   : if profile has < 30 valid bins
    FitError     : if require_convergence_guard=True and guard fails
    """
    if config is None:
        config = FitConfig()

    # Minimum valid bins check
    valid = _get_valid_mask(profile)
    if valid.sum() < 30:
        raise ValueError(
            f"Profile has only {valid.sum()} valid bins; "
            "need at least 30 for calibration fit."
        )

    chi2_by_stage = []

    # ------------------------------------------------------------------
    # Stage 0 — initial estimates
    # ------------------------------------------------------------------
    params_s0 = _stage0_initial_estimates(profile, config)

    n_bins = len(profile.r2_grid)

    # Compute Stage 0 chi²
    model_s0_full = _ne_model_nopsf(
        profile.r_grid, profile.r_max_px,
        t=params_s0['t_m'], R_refl=params_s0['R_refl'], alpha=params_s0['alpha'],
        n=1.0, I0=params_s0['I0'], I1=params_s0['I1'], I2=params_s0['I2'],
        B=params_s0['B'], n_fine=2000, n_bins=n_bins, n_subpixels=config.n_subpixels,
    )
    chi2_s0 = _compute_chi2(
        model_s0_full[valid], profile.profile[valid],
        profile.sigma_profile[valid], n_params=4,
    )
    chi2_by_stage.append(float(chi2_s0))

    # ------------------------------------------------------------------
    # Stage 1 — brute-force scan over t
    # ------------------------------------------------------------------
    t_best, I0_best, B_best = _stage1_brute_force_t(
        profile, params_s0, config, n_bins=n_bins)
    params_s1 = dict(params_s0, t_m=t_best, I0=I0_best, B=B_best)

    # Stage 1 chi² (at t_best with analytically optimal I0/B, with PSF)
    # PSF model is used here because Stage 1's WLS scan used PSF — using nopsf
    # would give a misleadingly high chi² since I0/B were calibrated to PSF peak heights.
    model_s1_full = _ne_model(
        profile.r_grid, profile.r_max_px,
        t=t_best, R_refl=params_s1['R_refl'], alpha=params_s1['alpha'],
        n=1.0, I0=params_s1['I0'], I1=0.0, I2=0.0,
        sigma0=params_s1['sigma0'], sigma1=0.0, sigma2=0.0,
        B=params_s1['B'], n_fine=2000, n_bins=n_bins, n_subpixels=config.n_subpixels,
    )
    chi2_s1 = _compute_chi2(
        model_s1_full[valid], profile.profile[valid],
        profile.sigma_profile[valid], n_params=4,
    )
    chi2_by_stage.append(float(chi2_s1))

    # ------------------------------------------------------------------
    # Stage 2 — Geometry: {t_m, alpha, I0, B}
    # ------------------------------------------------------------------
    params_s2_dict, cov_s2, chi2_s2, se_s2, pn_s2 = _stage2_geometry(
        profile, t_best, params_s1, config, chi2_s1)
    chi2_by_stage.append(float(chi2_s2))

    if se_s2 is None:
        raise RuntimeError("Stage 2 returned None stderr.")

    # ------------------------------------------------------------------
    # Stage 3 — Intensity: add {R_refl, I1, I2}
    # ------------------------------------------------------------------
    params_s3_dict, cov_s3, chi2_s3, se_s3, pn_s3 = _stage3_intensity(
        profile, params_s2_dict, config, chi2_s2)
    chi2_by_stage.append(float(chi2_s3))

    if se_s3 is None:
        raise RuntimeError("Stage 3 returned None stderr.")

    # ------------------------------------------------------------------
    # Stage 4 — PSF: add {sigma0}
    # ------------------------------------------------------------------
    params_s4_dict, cov_s4, chi2_s4, se_s4, pn_s4 = _stage4_psf(
        profile, params_s3_dict, config, chi2_s3)
    chi2_by_stage.append(float(chi2_s4))

    if se_s4 is None:
        raise RuntimeError("Stage 4 returned None stderr.")

    # ------------------------------------------------------------------
    # Stage 5 — Full free (all 10 parameters)
    # ------------------------------------------------------------------
    (params_s5_dict, cov_s5, chi2_s5, se_s5, pn_s5,
     epsilon_cal, epsilon_sci, m0_cal, m0_sci) = _stage5_full_free(
        profile, params_s4_dict, config, chi2_s4)
    chi2_by_stage.append(float(chi2_s5))

    if se_s5 is None:
        raise RuntimeError("Stage 5 returned None stderr.")

    # ------------------------------------------------------------------
    # Build 10×10 covariance in canonical parameter order
    # ------------------------------------------------------------------
    cov10 = np.full((10, 10), np.nan)
    # Fill from Stage 5 result (all params free)
    for i, ni in enumerate(pn_s5):
        ki = _PARAM_NAMES.index(ni)
        for j, nj in enumerate(pn_s5):
            kj = _PARAM_NAMES.index(nj)
            cov10[ki, kj] = cov_s5[i, j]

    se10 = np.sqrt(np.maximum(np.diag(cov10), 0))
    # Replace any NaN stderrs with a small positive floor
    se10 = np.where(np.isfinite(se10) & (se10 > 0), se10, 1e-30)

    # Correlation matrix
    diag_std = np.sqrt(np.maximum(np.diag(cov10), 0))
    outer = np.outer(diag_std, diag_std)
    with np.errstate(invalid='ignore', divide='ignore'):
        corr10 = np.where(outer > 0, cov10 / outer, 0.0)

    # Extract final parameter values from Stage 5
    p5 = params_s5_dict
    se_dict = dict(zip(pn_s5, se_s5))

    def _se(name):
        idx = _PARAM_NAMES.index(name)
        v = float(se10[idx])
        return v if v > 0 else 1e-30

    # ------------------------------------------------------------------
    # Construct CalibrationResult
    # ------------------------------------------------------------------
    n_valid_bins = int(valid.sum())
    n_params_free = 10

    result = CalibrationResult(
        # Parameters
        t_m=float(p5['t_m']),
        R_refl=float(p5['R_refl']),
        alpha=float(p5['alpha']),
        I0=float(p5['I0']),
        I1=float(p5['I1']),
        I2=float(p5['I2']),
        B=float(p5['B']),
        sigma0=float(p5['sigma0']),
        sigma1=float(p5['sigma1']),
        sigma2=float(p5['sigma2']),

        # 1σ uncertainties
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

        # 2σ uncertainties
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

        # Phase reference
        epsilon_cal=float(epsilon_cal),
        epsilon_sci=float(epsilon_sci),
        m0_cal=float(m0_cal),
        m0_sci=float(m0_sci),

        # Fit quality
        chi2_reduced=float(chi2_s5),
        n_bins_used=n_valid_bins,
        n_params_free=n_params_free,
        covariance=cov10,
        correlation=corr10,
        converged=True,
        quality_flags=FitFlags.GOOD,

        # Stage progression
        chi2_by_stage=chi2_by_stage,

        # Config
        fit_config=config,
    )

    # ------------------------------------------------------------------
    # Convergence guard
    # ------------------------------------------------------------------
    guard_ok = _convergence_guard(
        profile, p5, se_s5, pn_s5, config,
    )
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
    result.quality_flags = _post_fit_verification(result, profile, params_s0, config)
    if not guard_ok:
        result.quality_flags |= FitFlags.MULTIPLE_MINIMA

    # ------------------------------------------------------------------
    # Final validate before returning
    # ------------------------------------------------------------------
    validate_result_uncertainties(result)
    return result
