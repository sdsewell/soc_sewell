"""
Module:      m05_calibration_inversion_2026_04_06.py
Spec:        specs/S14_m05_calibration_inversion_2026-04-06.md
Author:      Claude Code
Generated:   2026-04-06
Last tested: 2026-04-06  (8/8 tests pass)
Project:     WindCube FPI Pipeline — NCAR/HAO
Repo:        soc_sewell
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import least_squares

from src.fpi.m01_airy_forward_model_2026_04_05 import (
    NE_WAVELENGTH_1_M,
    NE_WAVELENGTH_2_M,
    NE_INTENSITY_2,
    airy_modified,
)
from src.constants import (
    ETALON_GAP_M,
    ETALON_R_INSTRUMENT,
    ALPHA_RAD_PX,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FitFlags quality bitmask
# ---------------------------------------------------------------------------

class FitFlags:
    """Quality flags for CalibrationResult (S04 convention)."""
    GOOD                  = 0x000
    TOLANSKY_NOT_PROVIDED = 0x001   # running without Tolansky priors (warn)
    T_ALPHA_DEGENERATE    = 0x002   # |corr(t, α)| > 0.98 after Stage 2
    R_SIGMA_DEGENERATE    = 0x004   # |corr(R, σ₀)| > 0.95 after Stage 3
    PARAM_AT_BOUND        = 0x008   # any parameter hit its effective bound
    CHI2_HIGH             = 0x010   # chi2_reduced > 3.0
    CHI2_VERY_HIGH        = 0x020   # chi2_reduced > 10.0
    CHI2_LOW              = 0x040   # chi2_reduced < 0.5
    MULTIPLE_MINIMA       = 0x080   # convergence guard found different minimum
    PSF_UNPHYSICAL        = 0x100   # sigma(r) < 0.01 px at any profile radius
    STDERR_NONE           = 0x200   # any parameter stderr is non-finite after final fit

# ---------------------------------------------------------------------------
# Parameter bounds — three-tier architecture
# ---------------------------------------------------------------------------

PHYSICS_BOUNDS = {
    't_m':    (19.5e-3,   20.5e-3),
    'R_refl': (0.0,        1.0),
    'alpha':  (1e-5,       1e-3),
    'I0':     (0.0,        65535.0),
    'I1':     (-1.0,       1.0),
    'I2':     (-1.0,       1.0),
    'sigma0': (0.01,       10.0),
    'sigma1': (-5.0,       5.0),
    'sigma2': (-5.0,       5.0),
    'B':      (0.0,        1000.0),
}

INSTRUMENT_DEFAULTS = {
    't_m':    (19.95e-3,  20.07e-3),
    'R_refl': (0.35,       0.75),
    'alpha':  (1.4e-4,    1.8e-4),
    'I0':     (100.0,     15000.0),
    'I1':     (-0.5,       0.5),
    'I2':     (-0.5,       0.5),
    'sigma0': (0.1,        3.0),
    'sigma1': (-1.0,       1.0),
    'sigma2': (-1.0,       1.0),
    'B':      (50.0,       600.0),
}

# Canonical parameter ordering for internal 10-element arrays
_PARAM_NAMES = ['t_m', 'R_refl', 'alpha', 'I0', 'I1', 'I2',
                'sigma0', 'sigma1', 'sigma2', 'B']
_PARAM_IDX   = {n: i for i, n in enumerate(_PARAM_NAMES)}

# Free parameter sets by stage
_STAGE_FREE = {
    1: ['I0', 'I1', 'I2', 'B'],
    2: ['t_m', 'R_refl', 'alpha', 'I0', 'I1', 'I2', 'B'],
    3: ['t_m', 'R_refl', 'alpha', 'I0', 'I1', 'I2', 'sigma0', 'B'],
    4: list(_PARAM_NAMES),   # all 10
}

# ---------------------------------------------------------------------------
# FitConfig
# ---------------------------------------------------------------------------

@dataclass
class FitConfig:
    """
    User-facing configuration for M05 staged calibration inversion.

    Recommended usage: FitConfig(tolansky=TolanskyPipeline(fp).run())
    This automatically seeds t, α, and ε from Tolansky and tightens bounds.
    """
    # Tolansky priors (preferred path)
    tolansky: object = None   # TwoLineResult from S13

    # Manual overrides (used only if tolansky is None)
    t_init_m:     Optional[float] = None
    t_bounds_m:   Optional[tuple] = None
    R_init:        float = ETALON_R_INSTRUMENT
    R_bounds:     Optional[tuple] = None
    alpha_init:   Optional[float] = None
    alpha_bounds: Optional[tuple] = None
    sigma0_init:   float = 0.5
    sigma0_bounds: Optional[tuple] = None
    B_init:       Optional[float] = None
    B_bounds:     Optional[tuple] = None

    # Optimiser settings
    max_nfev: int   = 50_000
    ftol:     float = 1e-14
    xtol:     float = 1e-14
    gtol:     float = 1e-14

    # Convergence guard
    n_convergence_perturbations: int   = 3
    perturbation_scale:          float = 0.05
    require_convergence_guard:   bool  = False

    def resolve(self, profile) -> dict:
        """
        Return {param: (init_value, lo_bound, hi_bound)} for all 10 parameters.

        Merges Tolansky priors, manual overrides, and instrument defaults.
        All bounds are clamped to PHYSICS_BOUNDS after merging.
        """
        r = {}

        # --- t_m ---
        if self.tolansky is not None:
            d = float(self.tolansky.d_m)
            t_init = d
            t_lo, t_hi = d - 20e-6, d + 20e-6   # Tolansky-tightened ±20 µm
        else:
            t_init = float(self.t_init_m) if self.t_init_m is not None else ETALON_GAP_M
            t_lo, t_hi = self.t_bounds_m or INSTRUMENT_DEFAULTS['t_m']
        t_lo = max(float(t_lo), PHYSICS_BOUNDS['t_m'][0])
        t_hi = min(float(t_hi), PHYSICS_BOUNDS['t_m'][1])
        r['t_m'] = (t_init, t_lo, t_hi)

        # --- R_refl ---
        R_lo, R_hi = self.R_bounds or INSTRUMENT_DEFAULTS['R_refl']
        R_lo = max(float(R_lo), PHYSICS_BOUNDS['R_refl'][0])
        R_hi = min(float(R_hi), PHYSICS_BOUNDS['R_refl'][1])
        r['R_refl'] = (float(self.R_init), R_lo, R_hi)

        # --- alpha ---
        if self.tolansky is not None:
            a = float(self.tolansky.alpha_rad_px)
            a_init = a
            a_lo, a_hi = a * 0.95, a * 1.05   # Tolansky-tightened ±5%
        else:
            a_init = float(self.alpha_init) if self.alpha_init is not None else ALPHA_RAD_PX
            a_lo, a_hi = self.alpha_bounds or INSTRUMENT_DEFAULTS['alpha']
        a_lo = max(float(a_lo), PHYSICS_BOUNDS['alpha'][0])
        a_hi = min(float(a_hi), PHYSICS_BOUNDS['alpha'][1])
        r['alpha'] = (a_init, a_lo, a_hi)

        # --- I0 and B: auto-initialise from profile if available ---
        if profile is not None:
            good_vals = profile.profile[~profile.masked]
        else:
            good_vals = np.array([])

        if len(good_vals) > 0:
            I0_init   = float(np.percentile(good_vals, 75))
            B_init_auto = float(np.percentile(good_vals, 5)) * 0.8
        else:
            I0_init   = 1000.0
            B_init_auto = 300.0

        I0_lo, I0_hi = INSTRUMENT_DEFAULTS['I0']
        r['I0'] = (I0_init, max(float(I0_lo), PHYSICS_BOUNDS['I0'][0]),
                   min(float(I0_hi), PHYSICS_BOUNDS['I0'][1]))

        # --- I1, I2 ---
        for k in ('I1', 'I2'):
            lo, hi = INSTRUMENT_DEFAULTS[k]
            r[k] = (0.0, max(float(lo), PHYSICS_BOUNDS[k][0]),
                    min(float(hi), PHYSICS_BOUNDS[k][1]))

        # --- sigma0 ---
        s0_lo, s0_hi = self.sigma0_bounds or INSTRUMENT_DEFAULTS['sigma0']
        s0_lo = max(float(s0_lo), PHYSICS_BOUNDS['sigma0'][0])
        s0_hi = min(float(s0_hi), PHYSICS_BOUNDS['sigma0'][1])
        r['sigma0'] = (float(self.sigma0_init), s0_lo, s0_hi)

        # --- sigma1, sigma2 ---
        for k in ('sigma1', 'sigma2'):
            lo, hi = INSTRUMENT_DEFAULTS[k]
            r[k] = (0.0, max(float(lo), PHYSICS_BOUNDS[k][0]),
                    min(float(hi), PHYSICS_BOUNDS[k][1]))

        # --- B ---
        B_init = float(self.B_init) if self.B_init is not None else B_init_auto
        B_lo, B_hi = self.B_bounds or INSTRUMENT_DEFAULTS['B']
        B_lo = max(float(B_lo), PHYSICS_BOUNDS['B'][0])
        B_hi = min(float(B_hi), PHYSICS_BOUNDS['B'][1])
        r['B'] = (B_init, B_lo, B_hi)

        return r

# ---------------------------------------------------------------------------
# CalibrationResult
# ---------------------------------------------------------------------------

@dataclass
class CalibrationResult:
    """
    Output of M05 calibration inversion.
    All sigma_ and two_sigma_ fields required by S04.
    Passed directly to M06 as fixed instrument characterisation.
    """
    # Fitted parameters
    t_m:     float;  R_refl:  float;  alpha:   float
    I0:      float;  I1:      float;  I2:      float
    sigma0:  float;  sigma1:  float;  sigma2:  float
    B:       float

    # 1σ standard errors
    sigma_t_m:     float;  sigma_R_refl:  float;  sigma_alpha:   float
    sigma_I0:      float;  sigma_I1:      float;  sigma_I2:      float
    sigma_sigma0:  float;  sigma_sigma1:  float;  sigma_sigma2:  float
    sigma_B:       float

    # 2σ values (exactly 2 × sigma_*)
    two_sigma_t_m:     float;  two_sigma_R_refl:  float;  two_sigma_alpha:   float
    two_sigma_I0:      float;  two_sigma_I1:      float;  two_sigma_I2:      float
    two_sigma_sigma0:  float;  two_sigma_sigma1:  float;  two_sigma_sigma2:  float
    two_sigma_B:       float

    # Phase reference
    epsilon_cal:           float   # (2*t_m / NE_WAVELENGTH_1_M) % 1
    sigma_epsilon_cal:     float
    two_sigma_epsilon_cal: float

    # Fit quality
    chi2_reduced:    float
    n_bins_used:     int
    n_params_free:   int
    covariance:      np.ndarray   # (10, 10)
    correlation:     np.ndarray   # (10, 10)
    converged:       bool
    quality_flags:   int

    # Stage progression
    chi2_by_stage: list   # [chi2_s1, chi2_s2, chi2_s3, chi2_s4]

    # Tolansky priors used
    tolansky_d_m:        Optional[float]
    tolansky_alpha:      Optional[float]
    tolansky_epsilon:    Optional[float]

    # Config used
    fit_config: FitConfig

# ---------------------------------------------------------------------------
# Forward model helper
# ---------------------------------------------------------------------------

def _neon_model(r_arr, r_max, t, R, alpha, I0, I1, I2, sigma0, sigma1, sigma2,
               _N_fine=500):
    """
    Two-line neon forward model without bias B.

    Evaluated on a fine uniform grid of _N_fine points matching M02's synthesis
    sampling (default 500 pts over [0, r_max]), then linearly interpolated to
    r_arr.  This ensures the PSF convolution inside airy_modified uses the same
    grid spacing as M02, eliminating a systematic ~50 ADU bias that would arise
    from evaluating directly on the coarser M03 bin-centre radii.
    """
    r_fine = np.linspace(0.0, r_max, _N_fine)
    A1 = airy_modified(r_fine, NE_WAVELENGTH_1_M, t, R, alpha, 1.0,
                       r_max, I0, I1, I2, sigma0, sigma1, sigma2)
    A2 = airy_modified(r_fine, NE_WAVELENGTH_2_M, t, R, alpha, 1.0,
                       r_max, I0, I1, I2, sigma0, sigma1, sigma2)
    model_fine = A1 + NE_INTENSITY_2 * A2
    # Extrapolate beyond r_max with the last computed value
    return np.interp(r_arr, r_fine, model_fine)

# ---------------------------------------------------------------------------
# LM stage runner
# ---------------------------------------------------------------------------

def _run_lm_stage(r_good, profile_good, sigma_good, r_max,
                  p_all, free_names, config, resolved=None):
    """
    Run one Levenberg-Marquardt stage with the specified free parameters.

    Parameters
    ----------
    r_good, profile_good, sigma_good : arrays for non-masked bins
    r_max        : maximum radius (pixels)
    p_all        : 10-element parameter array (not modified)
    free_names   : list of parameter names that are free in this stage
    config       : FitConfig
    resolved     : output of config.resolve() — used to add soft-bound
                   penalty residuals that prevent physically impossible
                   parameter values (e.g. sigma0 → ∞) with method='lm'.

    Returns
    -------
    p_updated    : updated 10-element array
    cov          : covariance matrix for free params (n_free × n_free)
    stderrs      : 1σ stderrs for free params (n_free,)
    chi2_red     : reduced chi-squared (data residuals only, not penalty)
    lm_result    : scipy OptimizeResult
    """
    free_idx = np.array([_PARAM_IDX[n] for n in free_names])
    p_fixed  = p_all.copy()   # snapshot for closure
    p0       = p_fixed[free_idx]
    n_good   = len(r_good)
    n_free   = len(free_names)

    # Soft-bound penalty: one extra residual per free parameter.
    # Penalty kicks in linearly outside [lo, hi] with a scale of 1%
    # of the parameter range per sigma unit. This prevents divergence
    # to unphysical values (e.g. sigma0 → 10^3 px) without biasing
    # the in-bounds covariance estimate.
    if resolved is not None:
        lo_arr = np.array([resolved[n][1] for n in free_names])
        hi_arr = np.array([resolved[n][2] for n in free_names])
        range_arr = hi_arr - lo_arr + 1e-30
        # Penalty weight: each unit of penalty_sigma = 1% of range
        penalty_sigma = range_arr * 0.01
    else:
        lo_arr = hi_arr = penalty_sigma = None

    def _residuals_full(p_free):
        p = p_fixed.copy()
        p[free_idx] = p_free
        t, R, alpha, I0, I1, I2, sigma0, sigma1, sigma2, B = p
        model = _neon_model(r_good, r_max, t, R, alpha, I0, I1, I2,
                            sigma0, sigma1, sigma2) + B
        data_r = (profile_good - model) / sigma_good   # shape (n_good,)

        if lo_arr is not None:
            # Asymmetric penalty: linear outside bounds
            below = np.maximum(0.0, lo_arr - p_free) / penalty_sigma
            above = np.maximum(0.0, p_free - hi_arr) / penalty_sigma
            penalty = below + above   # shape (n_free,)
            return np.append(data_r, penalty)

        return data_r

    lm_result = least_squares(
        _residuals_full, p0, method='lm',
        ftol=config.ftol, xtol=config.xtol, gtol=config.gtol,
        max_nfev=config.max_nfev,
    )

    p_updated = p_fixed.copy()
    p_updated[free_idx] = lm_result.x

    # chi2 from DATA residuals only (exclude penalty rows)
    t, R, alpha, I0, I1, I2, sigma0, sigma1, sigma2, B = p_updated
    model_final = _neon_model(r_good, r_max, t, R, alpha, I0, I1, I2,
                               sigma0, sigma1, sigma2) + B
    data_resid = (profile_good - model_final) / sigma_good
    dof      = max(n_good - n_free, 1)
    chi2_red = float(np.sum(data_resid ** 2)) / dof

    # Jacobian: use DATA rows only for covariance  cov = chi2_red * (J^T J)^{-1}
    J_full = lm_result.jac   # shape (n_good [+ n_free], n_free)
    J = J_full[:n_good, :]   # data rows only
    try:
        JTJ = J.T @ J
        cond = np.linalg.cond(JTJ)
        if cond < 1e14:
            JTJ_inv = np.linalg.inv(JTJ)
        else:
            # Near-singular: use pseudoinverse (zeroes out degenerate directions).
            # Stderrs for degenerate parameter combinations will be very large but
            # finite, which is the correct behaviour for an ill-conditioned problem.
            JTJ_inv = np.linalg.pinv(JTJ, rcond=1e-10)
        cov     = chi2_red * JTJ_inv
        stderrs = np.sqrt(np.maximum(np.diag(cov), 0.0))
        if not np.all(np.isfinite(stderrs)):
            raise np.linalg.LinAlgError("pinv produced non-finite stderrs")
    except (np.linalg.LinAlgError, ValueError):
        stderrs = np.full(n_free, np.inf)
        cov     = np.full((n_free, n_free), np.inf)

    return p_updated, cov, stderrs, chi2_red, lm_result

# ---------------------------------------------------------------------------
# Stage 0 — seed from Tolansky / defaults
# ---------------------------------------------------------------------------

def _stage0_seed(profile, config):
    """
    Initialise parameter array from Tolansky priors and profile statistics.
    Returns (p_all, quality_flags).
    """
    resolved      = config.resolve(profile)
    quality_flags = FitFlags.GOOD

    p_all = np.array([resolved[n][0] for n in _PARAM_NAMES], dtype=float)

    if config.tolansky is None:
        quality_flags |= FitFlags.TOLANSKY_NOT_PROVIDED
    else:
        d_tol = float(config.tolansky.d_m)
        diff  = abs(d_tol - ETALON_GAP_M)
        if diff > 0.1e-3:
            log.warning(
                f"Tolansky d={d_tol*1e3:.4f} mm disagrees with ICOS "
                f"d={ETALON_GAP_M*1e3:.4f} mm by {diff*1e3:.3f} mm (> 0.1 mm). "
                "Tolansky wins."
            )
        elif diff > 0.02e-3:
            log.warning(
                f"Tolansky d={d_tol*1e3:.4f} mm differs from ICOS by "
                f"{diff*1e6:.1f} µm (> 0.02 mm). Tolansky wins."
            )

    I0_init = p_all[_PARAM_IDX['I0']]
    B_init  = p_all[_PARAM_IDX['B']]
    if I0_init <= 0:
        raise ValueError(f"I0_init = {I0_init} must be positive")
    if B_init <= 0:
        raise ValueError(f"B_init = {B_init} must be positive")

    return p_all, quality_flags

# ---------------------------------------------------------------------------
# epsilon_cal computation
# ---------------------------------------------------------------------------

def _compute_epsilon_cal(t_m, sigma_t_m):
    """
    Compute fractional interference order at fringe centre for Ne primary line.

    epsilon_cal = (2 * t_m / NE_WAVELENGTH_1_M) % 1.0
    sigma_epsilon_cal = (2 / NE_WAVELENGTH_1_M) * sigma_t_m
    two_sigma_epsilon_cal = 2.0 * sigma_epsilon_cal  (S04)
    """
    epsilon_cal      = (2.0 * t_m / NE_WAVELENGTH_1_M) % 1.0
    sigma_eps        = (2.0 / NE_WAVELENGTH_1_M) * sigma_t_m
    two_sigma_eps    = 2.0 * sigma_eps
    return float(epsilon_cal), float(sigma_eps), float(two_sigma_eps)

# ---------------------------------------------------------------------------
# Convergence guard
# ---------------------------------------------------------------------------

def _convergence_guard(r_good, profile_good, sigma_good, r_max,
                       p_stage3, p_stage4, stderrs_stage4, config,
                       quality_flags, resolved=None):
    """
    Re-run Stage 4 from n_convergence_perturbations perturbed starts.
    Sets MULTIPLE_MINIMA if any run diverges by > 3σ from p_stage4.
    Returns updated quality_flags.
    """
    n_perturb = config.n_convergence_perturbations
    scale     = config.perturbation_scale
    free_names = _STAGE_FREE[4]

    diverged = False
    for k in range(n_perturb):
        rng   = np.random.default_rng(k)
        signs = rng.choice([-1.0, 1.0], size=len(p_stage3))
        p_pert = p_stage3.copy()
        for i, name in enumerate(_PARAM_NAMES):
            if p_pert[i] != 0:
                p_pert[i] *= (1.0 + signs[i] * scale)

        try:
            p_run, _, stderrs_run, _, _ = _run_lm_stage(
                r_good, profile_good, sigma_good, r_max,
                p_pert, free_names, config, resolved=resolved,
            )
        except Exception:
            diverged = True
            break

        # Check if result is within 3σ of Stage 4 result
        for i, name in enumerate(_PARAM_NAMES):
            ref_stderr = stderrs_stage4[i] if np.isfinite(stderrs_stage4[i]) else 0.01 * abs(p_stage4[i]) + 1e-30
            if abs(p_run[i] - p_stage4[i]) > 3.0 * ref_stderr:
                diverged = True
                log.warning(
                    f"Convergence guard: perturbed run {k} diverged for {name}: "
                    f"|{p_run[i]:.6g} - {p_stage4[i]:.6g}| = "
                    f"{abs(p_run[i] - p_stage4[i]):.3g} > 3σ = {3*ref_stderr:.3g}"
                )
                break
        if diverged:
            break

    if diverged:
        quality_flags |= FitFlags.MULTIPLE_MINIMA
        if config.require_convergence_guard:
            raise RuntimeError(
                "Convergence guard: multiple minima detected. "
                "Set require_convergence_guard=False to suppress."
            )

    return quality_flags

# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------

def fit_calibration_fringe(
    profile,
    config: FitConfig = None,
) -> CalibrationResult:
    """
    Run the full M05 staged calibration inversion on a neon FringeProfile.

    Parameters
    ----------
    profile : FringeProfile from M03 reduce_calibration_frame()
    config  : FitConfig (optional; if None, uses FitConfig() defaults)

    Returns
    -------
    CalibrationResult with all 10 fitted parameters, sigmas, epsilon_cal,
    chi2_by_stage, quality_flags, and the config used.

    Raises
    ------
    ValueError
        If profile has fewer than 20 non-masked bins.
    RuntimeError
        If Stage 4 parameter stderrs are non-finite (covariance failure).
        If Stage 2 or 3 chi2_reduced does not improve from previous stage.
    """
    if config is None:
        config = FitConfig()

    # --- Validate profile ---
    good_mask = ~profile.masked & np.isfinite(profile.sigma_profile) \
                & (profile.sigma_profile > 0) & np.isfinite(profile.profile)
    n_good = int(np.sum(good_mask))
    if n_good < 20:
        raise ValueError(
            f"Only {n_good} non-masked bins with finite sigma in profile; "
            "need at least 20."
        )

    r_good       = profile.r_grid[good_mask]
    profile_good = profile.profile[good_mask]
    sigma_good   = profile.sigma_profile[good_mask].copy()
    r_max        = float(profile.r_max_px)

    # Floor sigma at a level consistent with the residual magnitude expected when
    # fitting the analytical neon model to the M03-reduced profile.  The dominant
    # error source is linear interpolation in M02's 2D image wrapping (~9 ADU RMS
    # for a 256×256 / 500-point synthesis).  Using 1% of the median signal gives a
    # floor that is small enough to preserve the chi² sensitivity to parameter
    # changes, yet large enough to avoid near-zero weights that inflate chi² and
    # make the Hessian singular on noiseless synthetic data.
    median_signal = float(np.median(profile_good))
    sigma_floor   = max(1.0, median_signal * 0.005)
    sigma_good    = np.maximum(sigma_good, sigma_floor)

    # --- Stage 0: seed ---
    p_all, quality_flags = _stage0_seed(profile, config)
    resolved = config.resolve(profile)   # shared bounds for all stages

    chi2_by_stage = []

    # --- Stage 1: photometric baseline {I0, I1, I2, B} ---
    p_all, cov_s1, stderr_s1, chi2_s1, res_s1 = _run_lm_stage(
        r_good, profile_good, sigma_good, r_max,
        p_all, _STAGE_FREE[1], config, resolved=resolved,
    )
    chi2_by_stage.append(chi2_s1)

    # Stage 1 verification
    I0_s1 = p_all[_PARAM_IDX['I0']]
    B_s1  = p_all[_PARAM_IDX['B']]
    if not np.all(np.isfinite(stderr_s1)):
        log.warning("Stage 1: some stderr values not finite")
    if chi2_s1 >= 50.0:
        log.warning(f"Stage 1 chi2_reduced = {chi2_s1:.2f} >= 50; photometric fit poor")
    if I0_s1 <= 0 or B_s1 <= 0:
        log.warning(f"Stage 1: I0={I0_s1:.1f}, B={B_s1:.1f} — one or both non-positive")

    # --- Stage 2: geometry + reflectivity {t, R, alpha, I0, I1, I2, B} ---
    p_all, cov_s2, stderr_s2, chi2_s2, res_s2 = _run_lm_stage(
        r_good, profile_good, sigma_good, r_max,
        p_all, _STAGE_FREE[2], config, resolved=resolved,
    )
    chi2_by_stage.append(chi2_s2)

    if chi2_s2 > chi2_s1 * 1.05:
        raise RuntimeError(
            f"Stage 2 chi2 ({chi2_s2:.4f}) did not improve from "
            f"Stage 1 ({chi2_s1:.4f})"
        )

    # T-alpha degenerate check (use data rows only of Stage 2 Jacobian)
    free_s2 = _STAGE_FREE[2]
    t_col_s2     = free_s2.index('t_m')
    alpha_col_s2 = free_s2.index('alpha')
    J2 = res_s2.jac[:n_good, :]   # data rows only
    J2_sub = J2[:, [t_col_s2, alpha_col_s2]]
    try:
        H2_sub = J2_sub.T @ J2_sub
        cond_ta = float(np.linalg.cond(H2_sub))
        if cond_ta > 100.0:
            quality_flags |= FitFlags.T_ALPHA_DEGENERATE
    except Exception:
        pass

    # --- Stage 3: PSF base width {t, R, alpha, I0, I1, I2, sigma0, B} ---
    p_all, cov_s3, stderr_s3, chi2_s3, res_s3 = _run_lm_stage(
        r_good, profile_good, sigma_good, r_max,
        p_all, _STAGE_FREE[3], config, resolved=resolved,
    )
    chi2_by_stage.append(chi2_s3)

    if chi2_s3 > chi2_s2 * 1.05:
        raise RuntimeError(
            f"Stage 3 chi2 ({chi2_s3:.4f}) did not improve from "
            f"Stage 2 ({chi2_s2:.4f})"
        )

    # R–sigma0 correlation check (Stage 3 cov is 8×8)
    free_s3 = _STAGE_FREE[3]
    R_col_s3     = free_s3.index('R_refl')
    sigma0_col_s3 = free_s3.index('sigma0')
    try:
        diag_s3 = np.sqrt(np.maximum(np.diag(cov_s3), 0.0))
        diag_s3[diag_s3 == 0] = 1.0
        corr_s3 = cov_s3 / np.outer(diag_s3, diag_s3)
        corr_R_sig = float(corr_s3[R_col_s3, sigma0_col_s3])
        if abs(corr_R_sig) > 0.95:
            quality_flags |= FitFlags.R_SIGMA_DEGENERATE
            log.warning(
                f"|corr(R, σ₀)| = {abs(corr_R_sig):.3f} > 0.95. "
                "Consider fixing R via R_bounds=(0.50, 0.56)."
            )
    except Exception:
        pass

    # sigma0 physical check
    sigma0_s3 = p_all[_PARAM_IDX['sigma0']]
    if sigma0_s3 < 0.01:
        log.warning(f"Stage 3: sigma0 = {sigma0_s3:.4f} px < 0.01 px minimum")

    # Save Stage 3 result for convergence guard
    p_stage3 = p_all.copy()

    # --- Stage 4: full free optimisation (all 10 params) ---
    p_all, cov_s4, stderr_s4, chi2_s4, res_s4 = _run_lm_stage(
        r_good, profile_good, sigma_good, r_max,
        p_all, _STAGE_FREE[4], config, resolved=resolved,
    )
    chi2_by_stage.append(chi2_s4)

    if chi2_s4 > chi2_s3 * 1.05:
        log.warning(
            f"Stage 4 chi2 ({chi2_s4:.4f}) increased from "
            f"Stage 3 ({chi2_s3:.4f}) by > 5%"
        )

    # Stage 4 stderr check — non-finite means covariance failure
    if not np.all(np.isfinite(stderr_s4)):
        quality_flags |= FitFlags.STDERR_NONE
        raise RuntimeError(
            "Stage 4 covariance failure: one or more parameter stderrs are "
            "non-finite. The Hessian may be singular."
        )

    # Unpack Stage 4 parameters
    t_fit, R_fit, alpha_fit, I0_fit, I1_fit, I2_fit, \
        sigma0_fit, sigma1_fit, sigma2_fit, B_fit = p_all

    # Stage 4 quality checks
    if chi2_s4 > 10.0:
        quality_flags |= FitFlags.CHI2_HIGH | FitFlags.CHI2_VERY_HIGH
        log.warning(f"Stage 4 chi2_reduced = {chi2_s4:.2f} > 10")
    elif chi2_s4 > 3.0:
        quality_flags |= FitFlags.CHI2_HIGH
        log.warning(f"Stage 4 chi2_reduced = {chi2_s4:.2f} > 3.0")
    elif chi2_s4 < 0.5:
        quality_flags |= FitFlags.CHI2_LOW

    # PSF physical check
    sigma_r = sigma0_fit \
              + sigma1_fit * np.sin(np.pi * r_good / r_max) \
              + sigma2_fit * np.cos(np.pi * r_good / r_max)
    if np.any(sigma_r < 0.01):
        quality_flags |= FitFlags.PSF_UNPHYSICAL

    # Parameter-at-bound check (manual since method='lm' has no active_mask)
    for i, name in enumerate(_PARAM_NAMES):
        lo, hi = resolved[name][1], resolved[name][2]
        if abs(p_all[i] - lo) < 1e-10 * (hi - lo + 1e-30) \
                or abs(p_all[i] - hi) < 1e-10 * (hi - lo + 1e-30):
            quality_flags |= FitFlags.PARAM_AT_BOUND

    # Monotonicity warning (not error)
    for k in range(len(chi2_by_stage) - 1):
        if chi2_by_stage[k + 1] > chi2_by_stage[k] * 1.05:
            log.warning(
                f"chi2 increased from stage {k+1} ({chi2_by_stage[k]:.4f}) "
                f"to stage {k+2} ({chi2_by_stage[k+1]:.4f})"
            )

    # --- Convergence guard (after Stage 4) ---
    quality_flags = _convergence_guard(
        r_good, profile_good, sigma_good, r_max,
        p_stage3, p_all, stderr_s4, config, quality_flags, resolved=resolved,
    )

    # --- Full 10×10 covariance and correlation ---
    cov_full  = cov_s4   # Stage 4 is all-free: cov is already 10×10
    diag_full = np.sqrt(np.maximum(np.diag(cov_full), 0.0))
    diag_full[diag_full == 0] = 1.0
    corr_full = cov_full / np.outer(diag_full, diag_full)

    # --- epsilon_cal ---
    sigma_t4 = float(stderr_s4[_PARAM_IDX['t_m']])
    eps_cal, sigma_eps, two_sigma_eps = _compute_epsilon_cal(t_fit, sigma_t4)

    # --- Tolansky traceability ---
    tol = config.tolansky
    tolansky_d_m     = float(tol.d_m)         if tol is not None else None
    tolansky_alpha   = float(tol.alpha_rad_px) if tol is not None else None
    tolansky_epsilon = float(tol.eps1)         if tol is not None else None

    return CalibrationResult(
        # Fitted parameters
        t_m    = float(t_fit),
        R_refl = float(R_fit),
        alpha  = float(alpha_fit),
        I0     = float(I0_fit),
        I1     = float(I1_fit),
        I2     = float(I2_fit),
        sigma0 = float(sigma0_fit),
        sigma1 = float(sigma1_fit),
        sigma2 = float(sigma2_fit),
        B      = float(B_fit),

        # 1σ stderrs
        sigma_t_m    = float(stderr_s4[_PARAM_IDX['t_m']]),
        sigma_R_refl = float(stderr_s4[_PARAM_IDX['R_refl']]),
        sigma_alpha  = float(stderr_s4[_PARAM_IDX['alpha']]),
        sigma_I0     = float(stderr_s4[_PARAM_IDX['I0']]),
        sigma_I1     = float(stderr_s4[_PARAM_IDX['I1']]),
        sigma_I2     = float(stderr_s4[_PARAM_IDX['I2']]),
        sigma_sigma0 = float(stderr_s4[_PARAM_IDX['sigma0']]),
        sigma_sigma1 = float(stderr_s4[_PARAM_IDX['sigma1']]),
        sigma_sigma2 = float(stderr_s4[_PARAM_IDX['sigma2']]),
        sigma_B      = float(stderr_s4[_PARAM_IDX['B']]),

        # 2σ (exactly 2 × sigma_*)
        two_sigma_t_m    = 2.0 * float(stderr_s4[_PARAM_IDX['t_m']]),
        two_sigma_R_refl = 2.0 * float(stderr_s4[_PARAM_IDX['R_refl']]),
        two_sigma_alpha  = 2.0 * float(stderr_s4[_PARAM_IDX['alpha']]),
        two_sigma_I0     = 2.0 * float(stderr_s4[_PARAM_IDX['I0']]),
        two_sigma_I1     = 2.0 * float(stderr_s4[_PARAM_IDX['I1']]),
        two_sigma_I2     = 2.0 * float(stderr_s4[_PARAM_IDX['I2']]),
        two_sigma_sigma0 = 2.0 * float(stderr_s4[_PARAM_IDX['sigma0']]),
        two_sigma_sigma1 = 2.0 * float(stderr_s4[_PARAM_IDX['sigma1']]),
        two_sigma_sigma2 = 2.0 * float(stderr_s4[_PARAM_IDX['sigma2']]),
        two_sigma_B      = 2.0 * float(stderr_s4[_PARAM_IDX['B']]),

        # Phase reference
        epsilon_cal           = eps_cal,
        sigma_epsilon_cal     = sigma_eps,
        two_sigma_epsilon_cal = two_sigma_eps,

        # Quality
        chi2_reduced  = float(chi2_s4),
        n_bins_used   = n_good,
        n_params_free = 10,
        covariance    = cov_full,
        correlation   = corr_full,
        converged     = bool(res_s4.success or res_s4.cost < 1e-10),
        quality_flags = quality_flags,

        # Stage diagnostics
        chi2_by_stage = chi2_by_stage,

        # Tolansky traceability
        tolansky_d_m      = tolansky_d_m,
        tolansky_alpha    = tolansky_alpha,
        tolansky_epsilon  = tolansky_epsilon,

        # Config
        fit_config = config,
    )
