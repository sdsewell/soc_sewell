# S14 — M05 Staged Calibration Inversion

**Spec ID:** S14
**Spec file:** `docs/specs/S14_m05_calibration_inversion_2026-04-06.md`
**Project:** WindCube FPI Science Operations Center Pipeline
**Institution:** NCAR / High Altitude Observatory (HAO)
**Status:** Authoritative
**Depends on:** S01, S02, S03, S04, S09 (M01 — Airy forward model),
S12 (M03 — FringeProfile), S13 (Tolansky — TwoLineResult priors)
**Used by:** S15 (M06 — airglow inversion receives CalibrationResult),
S17 (INT02), S18 (INT03)
**Last updated:** 2026-04-06
**Created/Modified by:** Claude AI

---

## 1. Purpose

M05 recovers the ten WindCube FPI instrument parameters from the neon
calibration `FringeProfile` produced by M03, using a staged
Levenberg-Marquardt optimisation strategy seeded by Tolansky priors from
S13.

The recovered parameters are:

| Symbol | Description | Units |
|--------|-------------|-------|
| `t` | Etalon plate spacing | m |
| `R` | Effective plate reflectivity | — |
| `alpha` | Magnification constant (pixel → angle) | rad/px |
| `I0` | Intensity envelope peak | ADU |
| `I1` | Envelope linear coefficient | — |
| `I2` | Envelope quadratic coefficient | — |
| `sigma0` | PSF base width | px |
| `sigma1` | PSF sine variation | px |
| `sigma2` | PSF cosine variation | px |
| `B` | CCD bias | ADU |

The single most important output is `epsilon_cal` — the fractional
interference order at the fringe centre for the neon primary line
(640.2248 nm). This is the zero-wind phase reference against which all
science Doppler shifts are subsequently measured. All other parameters
define the instrument function that M06 uses to invert the airglow fringe.


---

## 2. The role of Tolansky priors

### 2.2 What Tolansky provides

S13 (`TolanskyPipeline`) runs on the same `FringeProfile` and recovers, from
peak positions alone:

| Tolansky output | M05 use | Accuracy |
|----------------|---------|----------|
| `d_m` | Seeds `t_init_m`; eliminates FSR ambiguity | < 1 µm |
| `alpha_rad_px` | Seeds `alpha_init`; tightens α bounds | ~1–3% |
| `eps1` | Seeds fractional order ε₁; used to compute `epsilon_cal` | ~0.01 fringe |
| `eps2` | Cross-check against eps1; not directly used in fitting | |
| `f_px` | Diagnostic only; not used as a fit parameter | |

**Critical consequence:** Because Tolansky places `t` within the correct FSR
period with < 1 µm accuracy, the brute-force scan over `t` (Stage 1 of the
old M05 spec) is **entirely eliminated**. The full 4-stage sequence begins
with `t` already correct.

### 2.3 What Tolansky cannot provide

Tolansky uses peak positions only — not the full fringe shape. It therefore
cannot recover:
- `R` — fringe contrast requires the Airy function amplitude
- `sigma0/1/2` — PSF width requires the fringe peak shape
- `I0/1/2` — photometric envelope requires absolute ADU calibration
- `B` — CCD bias requires the inter-fringe baseline

These four groups of parameters are the exclusive province of M05.

---

## 3. Degeneracy analysis

Understanding these degeneracies determines the stage ordering.

### 3.1 t–α phase degeneracy (breaks with Tolansky)

The fringe spacing (distance between successive peaks in r²) depends on the
ratio `f²λ/(nd)`. A small increase in `t` can be exactly compensated by a
small increase in `α` because both shift the fringe positions. Without
Tolansky, this creates a correlated (t, α) valley in χ². With Tolansky both
are seeded accurately, and the valley is narrow enough that the LM fit finds
the bottom reliably.

**Residual risk:** Even with good seeds, the condition number of the (t, α)
sub-block of the Hessian should be checked after Stage 2. If it exceeds 100,
flag `T_ALPHA_DEGENERATE`.

### 3.2 R–σ₀ fringe-width degeneracy

Both `R` (reflectivity) and `σ₀` (PSF base width) broaden fringe peaks.
A high-R, large-σ₀ solution can fit the same data as a low-R, small-σ₀
solution. This is a **genuine physical degeneracy** in the full Airy model.
It is broken by:
1. Introducing `R` before `σ₀` (Stage 2 before Stage 3)
2. Tight bounds on `R` from the ICOS coating specification (Section 5)
3. Checking `|corr(R, σ₀)|` after Stage 3 and flagging if > 0.95

### 3.3 I0–B photometric degeneracy

Both `I₀` (peak envelope intensity) and `B` (bias) are additive offsets in
the fringe profile. They are distinguished only by their spatial structure:
`I₀` varies across the image (through `I₁`, `I₂`), while `B` is constant.
For this reason, Stage 1 fits them together before R or σ are introduced.

---

## 4. Physical constants from S03

```python
from src.constants import (
    NE_WAVELENGTH_1_M,    # 640.2248e-9 m — primary neon line
    NE_WAVELENGTH_2_M,    # 638.2991e-9 m — secondary neon line
    NE_INTENSITY_2,       # 0.3 — secondary line relative intensity
    ETALON_GAP_M,         # 20.008e-3 m — ICOS measured gap (d prior)
    ETALON_R_INSTRUMENT,  # 0.53 — FlatSat effective reflectivity
    ALPHA_RAD_PX,         # 1.6e-4 rad/px — nominal magnification
    CCD_PIXEL_2X2_UM,     # 32.0 µm — effective pixel pitch (2×2 binned)
    FOCAL_LENGTH_M,       # 0.200 m — FPI imaging lens
)
```

---

## 5. Parameter bounds — three-tier architecture

Three tiers, applied in order. User overrides can only tighten bounds, never
relax beyond physics.

```python
PHYSICS_BOUNDS = {
    't_m':    (19.5e-3,  20.5e-3),   # physically realisable gap range
    'R':      (0.0,       1.0),       # reflectivity must be in (0, 1)
    'alpha':  (1e-5,      1e-3),      # physically realisable magnification
    'I0':     (0.0,       65535.0),   # CCD ADU range (16-bit)
    'I1':     (-1.0,      1.0),       # dimensionless vignetting
    'I2':     (-1.0,      1.0),
    'sigma0': (0.01,      10.0),      # PSF width in pixels
    'sigma1': (-5.0,      5.0),
    'sigma2': (-5.0,      5.0),
    'B':      (0.0,       1000.0),    # bias in ADU
}

INSTRUMENT_DEFAULTS = {
    # Bounds derived from ICOS build report, FlatSat measurements, CCD97 spec
    # These are the widest bounds Claude Code should use without user override
    't_m':    (19.95e-3,  20.07e-3),  # ±0.06 mm = ±187 FSR around ICOS 20.008 mm
    'R':      (0.35,       0.75),      # FlatSat 0.53 ± generous margin
    'alpha':  (1.4e-4,    1.8e-4),    # nominal ± 12.5%
    'I0':     (100.0,     15000.0),
    'I1':     (-0.5,       0.5),
    'I2':     (-0.5,       0.5),
    'sigma0': (0.1,        3.0),
    'sigma1': (-1.0,       1.0),
    'sigma2': (-1.0,       1.0),
    'B':      (50.0,       600.0),
}
```

**Tolansky-tightened bounds** (applied automatically when `tolansky` is
provided to `FitConfig`):

```python
TOLANSKY_TIGHTENED = {
    # t: Tolansky recovers d to < 1 µm. Use ±0.020 mm (±62 FSR) around
    # Tolansky result — tight enough to exclude all wrong periods, wide
    # enough to accommodate any residual Tolansky uncertainty.
    't_m':    lambda d_m: (d_m - 20e-6, d_m + 20e-6),

    # alpha: Tolansky recovers α to ~1–3%. Use ±5% around Tolansky result.
    'alpha':  lambda a: (a * 0.95, a * 1.05),
}
```

---

## 6. FitConfig dataclass

```python
from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np

@dataclass
class FitConfig:
    """
    User-facing configuration for M05 staged calibration inversion.

    All fields are optional. Defaults come from INSTRUMENT_DEFAULTS.
    User overrides (via *_bounds fields) are clamped to PHYSICS_BOUNDS —
    they can only tighten bounds, never relax them beyond physics.

    The recommended usage is to pass a TwoLineResult from S13:
        config = FitConfig(tolansky=tolansky_result)
    This automatically seeds t, α, and ε from Tolansky and tightens their
    bounds. All other fields use instrument defaults.
    """

    # Tolansky priors (preferred path — pass TwoLineResult from S13)
    tolansky: object = None   # TwoLineResult; if provided, seeds t, α, ε

    # Manual overrides (used only if tolansky is None)
    t_init_m:    Optional[float] = None   # None → use ETALON_GAP_M from S03
    t_bounds_m:  Optional[Tuple] = None   # None → INSTRUMENT_DEFAULTS

    R_init:       float = ETALON_R_INSTRUMENT  # 0.53 from FlatSat
    R_bounds:     Optional[Tuple] = None

    alpha_init:   Optional[float] = None   # None → ALPHA_RAD_PX from S03
    alpha_bounds: Optional[Tuple] = None

    sigma0_init:  float = 0.5    # pixels; Harding recommendation
    sigma0_bounds: Optional[Tuple] = None

    B_init:       Optional[float] = None   # None → auto from profile 5th pct
    B_bounds:     Optional[Tuple] = None

    # Optimiser settings
    max_nfev: int   = 50_000
    ftol:     float = 1e-14
    xtol:     float = 1e-14
    gtol:     float = 1e-14

    # Convergence guard settings
    n_convergence_perturbations: int  = 3
    perturbation_scale:          float = 0.05   # 5% of each parameter value
    require_convergence_guard:   bool = False   # warn only by default

    def resolve(self, profile) -> dict:
        """
        Return a dict of {param: (init_value, lo_bound, hi_bound)} for all
        10 parameters, merging Tolansky priors, manual overrides, and defaults.

        If self.tolansky is not None:
            t_init  = tolansky.d_m
            t_bounds = TOLANSKY_TIGHTENED['t_m'](tolansky.d_m)
            alpha_init  = tolansky.alpha_rad_px
            alpha_bounds = TOLANSKY_TIGHTENED['alpha'](tolansky.alpha_rad_px)
            B_init  = profile.profile[np.where(~profile.masked)].min() * 0.8
        Otherwise:
            t_init  = self.t_init_m or ETALON_GAP_M
            alpha_init = self.alpha_init or ALPHA_RAD_PX
        All bounds are clamped to PHYSICS_BOUNDS after merging.
        """
```

---

## 7. Quality flags

```python
class FitFlags:
    GOOD                 = 0x00
    TOLANSKY_NOT_PROVIDED = 0x01   # running without Tolansky priors (warn)
    T_ALPHA_DEGENERATE   = 0x02   # |corr(t, α)| > 0.98 after Stage 2
    R_SIGMA_DEGENERATE   = 0x04   # |corr(R, σ₀)| > 0.95 after Stage 3
    PARAM_AT_BOUND       = 0x08   # any parameter hit its effective bound
    CHI2_HIGH            = 0x10   # chi2_reduced > 3.0
    CHI2_VERY_HIGH       = 0x20   # chi2_reduced > 10.0
    CHI2_LOW             = 0x40   # chi2_reduced < 0.5
    MULTIPLE_MINIMA      = 0x80   # convergence guard found different minimum
    PSF_UNPHYSICAL       = 0x100  # sigma(r) < 0.01 px at any profile radius
    STDERR_NONE          = 0x200  # any parameter stderr is None after final fit
```

Note: `TOLANSKY_NOT_PROVIDED` is a warning, not a fatal flag. M05 can run
without Tolansky using the S03 defaults, but the FSR-period ambiguity risk
is then the user's responsibility.

---

## 8. Output dataclass

Per S04 — every fitted parameter must have `sigma_` and `two_sigma_` fields.

```python
@dataclass
class CalibrationResult:
    """
    Output of M05 calibration inversion.
    Passed directly to M06 as fixed instrument characterisation.
    All sigma_ and two_sigma_ fields required by S04.
    """
    # --- Fitted parameters (10 total) ---
    t_m:      float   # etalon gap, metres
    R_refl:   float   # effective plate reflectivity
    alpha:    float   # magnification constant, rad/px
    I0:       float   # intensity envelope peak, ADU
    I1:       float   # envelope linear coefficient
    I2:       float   # envelope quadratic coefficient
    sigma0:   float   # PSF base width, pixels
    sigma1:   float   # PSF sine variation, pixels
    sigma2:   float   # PSF cosine variation, pixels
    B:        float   # CCD bias, ADU

    # --- 1σ standard errors ---
    sigma_t_m:     float
    sigma_R_refl:  float
    sigma_alpha:   float
    sigma_I0:      float
    sigma_I1:      float
    sigma_I2:      float
    sigma_sigma0:  float
    sigma_sigma1:  float
    sigma_sigma2:  float
    sigma_B:       float

    # --- 2σ values (exactly 2 × sigma_*) ---
    two_sigma_t_m:     float
    two_sigma_R_refl:  float
    two_sigma_alpha:   float
    two_sigma_I0:      float
    two_sigma_I1:      float
    two_sigma_I2:      float
    two_sigma_sigma0:  float
    two_sigma_sigma1:  float
    two_sigma_sigma2:  float
    two_sigma_B:       float

    # --- Phase reference (critical for wind retrieval) ---
    epsilon_cal:   float   # fractional order at fringe centre, λ₁ = 640.2248 nm
                           # = (2 * t_m / NE_WAVELENGTH_1_M) mod 1
                           # This is the zero-wind phase reference for M06
    sigma_epsilon_cal:   float
    two_sigma_epsilon_cal: float

    # --- Fit quality ---
    chi2_reduced:    float
    n_bins_used:     int
    n_params_free:   int       # 10 in Stage 4
    covariance:      np.ndarray   # shape (10, 10)
    correlation:     np.ndarray   # normalised, shape (10, 10)
    converged:       bool
    quality_flags:   int          # FitFlags bitmask

    # --- Stage progression (for diagnostics) ---
    chi2_by_stage: list[float]    # chi2_reduced after each stage [S1, S2, S3, S4]
                                  # must be monotonically non-increasing

    # --- Tolansky priors used (for traceability) ---
    tolansky_d_m:        Optional[float]  # S13 d_m used to seed t
    tolansky_alpha:      Optional[float]  # S13 alpha_rad_px used to seed α
    tolansky_epsilon:    Optional[float]  # S13 eps1 used as reference

    # --- Configuration used (for reproducibility) ---
    fit_config: FitConfig
```

### 8.1 epsilon_cal computation

```python
epsilon_cal = (2.0 * t_m / NE_WAVELENGTH_1_M) % 1.0
```

This is the fractional interference order at the fringe centre for the
primary neon line. M06 uses this as the zero-wind reference: the science
Doppler shift ε_sci is measured relative to ε_cal projected to 630.0304 nm.

Uncertainty propagation (linear approximation):
```python
sigma_epsilon_cal = (2.0 / NE_WAVELENGTH_1_M) * sigma_t_m
```

---

## 9. The staged fitting sequence

Four fitting stages, preceded by a Tolansky seed step. Each stage must pass
its verification check before the next begins.

### Stage 0 — Tolansky seed (no optimisation)

**Purpose:** Initialize all parameters from Tolansky priors and profile
statistics. No LM fitting occurs.

```python
def _stage0_seed(
    profile: FringeProfile,
    config: FitConfig,
) -> dict:
    """
    Return initial parameter dict from Tolansky priors and profile statistics.

    If config.tolansky is provided:
        t_init  = config.tolansky.d_m
        alpha_init = config.tolansky.alpha_rad_px
        ε reference = config.tolansky.eps1
    Else:
        t_init  = ETALON_GAP_M  (S03 constant — still valid, but FSR risk)
        alpha_init = ALPHA_RAD_PX  (S03 constant)
        Set TOLANSKY_NOT_PROVIDED quality flag

    I0_init  = np.percentile(non-masked profile values, 75)
    B_init   = np.percentile(non-masked profile values, 5) × 0.8
    I1_init  = 0.0
    I2_init  = 0.0
    sigma0_init = 0.5 pixels  (Harding recommendation)
    sigma1_init = 0.0
    sigma2_init = 0.0
    R_init   = config.R_init  (default ETALON_R_INSTRUMENT = 0.53)
    """
```

**Stage 0 verification:**
- If Tolansky provided: assert `abs(config.tolansky.d_m - ETALON_GAP_M) < 0.1e-3`
  (Tolansky and S03 prior agree to within 0.1 mm). Log WARNING if they disagree
  by more than 0.02 mm. Do not raise an error — Tolansky wins.
- `I0_init > 0`, `B_init > 0`
- `t_init` within `INSTRUMENT_DEFAULTS['t_m']` bounds

### Stage 1 — Photometric baseline

**Free parameters:** `{I0, I1, I2, B}` — 4 free  
**Fixed:** `{t, R, alpha, sigma0, sigma1, sigma2}` at Stage 0 values

**Purpose:** Establish the photometric baseline (envelope and bias) before
any spectral parameters move. With `t` and `α` fixed by Tolansky to high
accuracy, the forward model fringe positions are correct, and the envelope
fit is unambiguous.

**Implementation:** Use `scipy.optimize.least_squares(method='lm')` with
residuals weighted by `1/sigma_profile`. Only non-masked bins are included.

```python
residuals = (profile.profile[~masked] - model[~masked]) / profile.sigma_profile[~masked]
```

**Stage 1 verification (all must pass before Stage 2):**
- All 4 `stderr` values finite and not None
- `chi2_reduced < 50` (loose — envelope-only fit won't be perfect)
- `I0 > 0` and `I0 × (1 + I1 + I2) > 0` (intensity positive at all radii)
- `B > 0`

### Stage 2 — Geometry + reflectivity

**Free parameters:** `{t, alpha, I0, I1, I2, B, R}` — 7 free  
**Fixed:** `{sigma0, sigma1, sigma2}` at Stage 0 values (0.5, 0, 0)

**Purpose:** Refine the geometric parameters (`t`, `α`) and introduce
reflectivity (`R`) simultaneously. With the photometric baseline established
in Stage 1, the fringe contrast (determined by `R`) can be reliably recovered.
`R` is introduced here — before PSF — because R and σ₀ are degenerate if
introduced together.

**Starting point:** Stage 1 result + `R = config.R_init` (default 0.53).

**Stage 2 verification:**
- All 7 `stderr` values finite
- `chi2_reduced < Stage 1 chi2_reduced` (must improve)
- `0.35 < R < 0.75` (within instrument defaults)
- `t_m` within Tolansky-tightened bounds (if Tolansky provided)
- Check condition number of (t, α) sub-block of Hessian:
  ```python
  H_sub = J.T @ J  # 2×2 sub-block for t and α
  cond = np.linalg.cond(H_sub)
  if cond > 100: quality_flags |= FitFlags.T_ALPHA_DEGENERATE
  ```

### Stage 3 — PSF base width

**Free parameters:** `{t, alpha, I0, I1, I2, B, R, sigma0}` — 8 free  
**Fixed:** `{sigma1, sigma2}` at 0.0

**Purpose:** Recover the PSF base width `σ₀`. This is introduced last because
it is the most degenerate with `R`. By the time Stage 3 runs, `R` is already
well-constrained from Stage 2, which breaks the R–σ₀ degeneracy.

**Starting point:** Stage 2 result + `sigma0 = config.sigma0_init` (0.5 px).

**Stage 3 verification:**
- All 8 `stderr` values finite
- `chi2_reduced < Stage 2 chi2_reduced`
- `sigma0 > 0.01` pixels (physical minimum)
- Check R–σ₀ correlation:
  ```python
  corr_R_sigma = correlation_matrix[R_idx, sigma0_idx]
  if abs(corr_R_sigma) > 0.95:
      quality_flags |= FitFlags.R_SIGMA_DEGENERATE
      log.warning(f"|corr(R,σ₀)| = {abs(corr_R_sigma):.3f} > 0.95. "
                  "Consider fixing R at 0.53 via R_bounds=(0.50, 0.56).")
  ```

### Stage 4 — Full free optimisation

**Free parameters:** all 10 `{t, alpha, I0, I1, I2, B, R, sigma0, sigma1, sigma2}`

**Purpose:** Final joint refinement over all parameters. `sigma1` and `sigma2`
are released here to capture any azimuthal PSF variation. With Stage 3
providing a good starting point, convergence is fast.

**Starting point:** Stage 3 result + `sigma1 = 0.0`, `sigma2 = 0.0`.

**Stage 4 verification (all mandatory per S04):**
- All 10 `stderr` values finite and not None (set `STDERR_NONE` and raise
  `RuntimeError` if any are None — covariance cannot be trusted)
- `0.5 < chi2_reduced < 10.0` (warn at > 3.0, raise at > 10.0)
- `chi2_reduced < Stage 3 chi2_reduced`
- `sigma(r) > 0.01` px at all radial positions in `profile.r_grid`:
  ```python
  sigma_r = sigma0 + sigma1*np.sin(np.pi*r/r_max) + sigma2*np.cos(np.pi*r/r_max)
  if np.any(sigma_r < 0.01): quality_flags |= FitFlags.PSF_UNPHYSICAL
  ```
- All parameters within `effective_bounds` — check `at_lower_bound` and
  `at_upper_bound` flags from scipy result; set `PARAM_AT_BOUND` if any hit

### Convergence guard (runs after Stage 4)

Re-run Stage 4 from `n_convergence_perturbations` (default 3) perturbed
starting points. Each starting point is Stage 3 result with each parameter
independently perturbed by ±`perturbation_scale` × parameter value.

```python
def _convergence_guard(
    profile: FringeProfile,
    stage4_result: CalibrationResult,
    config: FitConfig,
) -> bool:
    """
    Returns True if all perturbed runs converge to the same minimum (within 1σ).
    A different minimum: |param_perturbed - param_stage4| > 3 × stderr_stage4
    for any of the 10 parameters.
    Sets MULTIPLE_MINIMA flag and logs WARNING if any run diverges.
    Does NOT raise an error unless config.require_convergence_guard = True.
    """
```

---

## 10. Top-level function

```python
def fit_calibration_fringe(
    profile: FringeProfile,
    config: FitConfig = None,
) -> CalibrationResult:
    """
    Run the full M05 staged calibration inversion on a neon FringeProfile.

    Parameters
    ----------
    profile : FringeProfile
        From M03 reduce_calibration_frame(). Must have peak_fits populated
        (used by Tolansky before this function is called).
        profile.quality_flags is checked — a CENTRE_FAILED flag raises
        a ValueError unless explicitly overridden in config.
    config : FitConfig, optional
        If None, uses FitConfig() with all defaults.
        Recommended: pass FitConfig(tolansky=tolansky_result) where
        tolansky_result comes from TolanskyPipeline.run(profile).

    Returns
    -------
    CalibrationResult
        All 10 fitted parameters with sigma and two_sigma, epsilon_cal,
        chi2_reduced, chi2_by_stage, quality_flags, and the config used.

    Raises
    ------
    ValueError
        If profile.quality_flags & CENTRE_FAILED, or if fewer than
        20 unmasked bins remain after excluding masked bins.
    RuntimeError
        If any stage 4 parameter stderr is None (covariance failure).
        If Stage 2 or 3 chi2_reduced does not improve from previous stage.

    Notes
    -----
    Recommended call sequence:
        fp = reduce_calibration_frame(image, cx=145, cy=145)
        tol = TolanskyPipeline(fp).run()
        cfg = FitConfig(tolansky=tol)
        cal = fit_calibration_fringe(fp, cfg)
    """
```

---

## 11. Verification tests

All tests in `tests/test_s14_m05_calibration_inversion.py`. Tests T1–T4
require no real data. T5–T7 use synthetic images from M02. T8 uses real
FlatSat data if available.

### T1 — FitConfig resolves Tolansky priors correctly

```python
def test_fitconfig_resolves_tolansky():
    """
    When a TwoLineResult is passed, t_init and alpha_init must come from
    Tolansky, and bounds must be tightened relative to INSTRUMENT_DEFAULTS.
    """
    from src.fpi.tolansky_2026_04_05 import TwoLineResult
    # Build a minimal TwoLineResult stub
    tol = TwoLineResult.__new__(TwoLineResult)
    tol.d_m = 20.008e-3
    tol.alpha_rad_px = 1.607e-4
    tol.eps1 = 0.34

    config = FitConfig(tolansky=tol)
    resolved = config.resolve(profile=None)   # profile=None for this test

    assert abs(resolved['t_m'][0] - 20.008e-3) < 1e-9
    # Tolansky-tightened bounds should be narrower than INSTRUMENT_DEFAULTS
    t_lo, t_hi = resolved['t_m'][1], resolved['t_m'][2]
    assert (t_hi - t_lo) < (20.07e-3 - 19.95e-3), \
        "Tolansky-tightened t bounds should be narrower than instrument defaults"
    # Alpha bounds should be ±5% around Tolansky value
    a_lo, a_hi = resolved['alpha'][1], resolved['alpha'][2]
    assert abs(a_lo - 1.607e-4 * 0.95) < 1e-8
    assert abs(a_hi - 1.607e-4 * 1.05) < 1e-8
```

### T2 — chi2_by_stage is monotonically non-increasing

```python
def test_chi2_monotone(synthetic_cal_profile):
    """
    chi2_reduced must not increase between stages.
    Each stage must improve (or maintain) the fit quality.
    """
    result = fit_calibration_fringe(synthetic_cal_profile,
                                    FitConfig(tolansky=synthetic_tolansky))
    stages = result.chi2_by_stage
    assert len(stages) == 4
    for i in range(len(stages) - 1):
        assert stages[i+1] <= stages[i] * 1.05, \
            f"chi2 increased from stage {i+1} ({stages[i]:.3f}) to "
            f"stage {i+2} ({stages[i+1]:.3f})"
```

### T3 — All two_sigma fields are exactly 2 × sigma

```python
def test_two_sigma_convention(synthetic_cal_profile):
    """S04 compliance: two_sigma_X must equal exactly 2.0 × sigma_X."""
    result = fit_calibration_fringe(synthetic_cal_profile,
                                    FitConfig(tolansky=synthetic_tolansky))
    params = ['t_m', 'R_refl', 'alpha', 'I0', 'I1', 'I2',
              'sigma0', 'sigma1', 'sigma2', 'B', 'epsilon_cal']
    for p in params:
        sigma = getattr(result, f'sigma_{p}')
        two_sigma = getattr(result, f'two_sigma_{p}')
        assert abs(two_sigma - 2.0 * sigma) < 1e-15, \
            f"two_sigma_{p} = {two_sigma} ≠ 2 × sigma_{p} = {2*sigma}"
```

### T4 — epsilon_cal computed correctly from t_m

```python
def test_epsilon_cal_computation(synthetic_cal_profile):
    """epsilon_cal = (2 * t_m / lambda_1) mod 1."""
    from src.constants import NE_WAVELENGTH_1_M
    result = fit_calibration_fringe(synthetic_cal_profile,
                                    FitConfig(tolansky=synthetic_tolansky))
    expected = (2.0 * result.t_m / NE_WAVELENGTH_1_M) % 1.0
    assert abs(result.epsilon_cal - expected) < 1e-12
```

### T5 — Round-trip: inject known params, recover within tolerance

```python
def test_round_trip_recovery():
    """
    Synthesise a calibration image with known InstrumentParams (M02).
    Run M03 → Tolansky → M05. Recovered parameters must match to:
      t_m:    < 1e-9 m  (< 0.001 µm)
      R_refl: < 0.01
      alpha:  < 1e-6 rad/px (< 0.6%)
      sigma0: < 0.05 px
    chi2_reduced must be in [0.8, 1.5].
    """
    from src.fpi.m02_calibration_synthesis_2026_04_05 import synthesise_calibration_image
    from src.fpi.m01_airy_forward_model_2026_04_05 import InstrumentParams
    from src.fpi.m03_annular_reduction_2026_04_05 import reduce_calibration_frame
    from src.fpi.tolansky_2026_04_05 import TolanskyPipeline

    params = InstrumentParams()  # uses S03 defaults
    result_m02 = synthesise_calibration_image(params, image_size=256, add_noise=False)
    image = result_m02['image_2d']

    fp = reduce_calibration_frame(image, cx_seed=128, cy_seed=128)
    tol = TolanskyPipeline(fp).run()
    config = FitConfig(tolansky=tol)
    cal = fit_calibration_fringe(fp, config)

    assert abs(cal.t_m - params.t_m) < 1e-9
    assert abs(cal.R_refl - params.R_refl) < 0.01
    assert abs(cal.alpha - params.alpha) < 1e-6
    assert abs(cal.sigma0 - params.sigma0) < 0.05
    assert 0.8 < cal.chi2_reduced < 1.5
```

### T6 — Without Tolansky: TOLANSKY_NOT_PROVIDED flag set

```python
def test_no_tolansky_flag(synthetic_cal_profile):
    """Running without Tolansky sets the TOLANSKY_NOT_PROVIDED quality flag."""
    result = fit_calibration_fringe(synthetic_cal_profile, FitConfig())
    assert result.quality_flags & FitFlags.TOLANSKY_NOT_PROVIDED, \
        "TOLANSKY_NOT_PROVIDED flag must be set when no Tolansky result provided"
    # Should still converge to a reasonable result on noiseless data
    assert result.converged
    assert result.chi2_reduced < 5.0
```

### T7 — Noisy synthetic: chi2_reduced near 1, sigma_t < 1 µm

```python
def test_noisy_round_trip():
    """
    Synthesise with Poisson noise at realistic SNR (ΔS/σ_N ≈ 5).
    chi2_reduced should be in [0.7, 2.0].
    sigma_t_m < 1e-6 m (1 µm).
    """
    # ... synthesise with add_noise=True, run full pipeline
    assert 0.7 < cal.chi2_reduced < 2.0
    assert cal.sigma_t_m < 1e-6
```

### T8 — Real FlatSat data (skip if file absent)

```python
@pytest.mark.skipif(not Path('data/flatsat_cal_profile.npz').exists(),
                    reason='FlatSat data not available')
def test_flatsat_data():
    """
    Run on real FlatSat calibration profile.
    Expected: t_m near 20.008e-3 m, R near 0.53, chi2 in [0.8, 3.0].
    The FlatSat value of 20.670 mm must NOT appear in the result —
    that would indicate the FSR-period ambiguity has returned.
    """
    fp = FringeProfile.load('data/flatsat_cal_profile.npz')
    tol = TolanskyPipeline(fp).run()
    config = FitConfig(tolansky=tol)
    cal = fit_calibration_fringe(fp, config)

    assert abs(cal.t_m - 20.008e-3) < 0.05e-3, \
        f"t_m = {cal.t_m*1e3:.4f} mm; expected near 20.008 mm. " \
        f"If near 20.670, FSR-period ambiguity has re-emerged."
    assert 0.40 < cal.R_refl < 0.70
    assert 0.8 < cal.chi2_reduced < 3.0
```

---

## 12. Expected numerical values

For noiseless synthetic data from `InstrumentParams()` defaults (S03):

| Quantity | Expected | Notes |
|----------|----------|-------|
| `t_m` recovery | `params.t_m` ± 1e-9 m | Tolansky seeds to < 1 µm |
| `R_refl` recovery | `params.R_refl` ± 0.01 | |
| `alpha` recovery | `params.alpha` ± 1e-6 rad/px | |
| `epsilon_cal` | `(2*t/λ₁) mod 1` | Exact formula |
| `chi2_reduced` (noiseless) | 0.8–1.5 | Some model-vs-model residual expected |
| `chi2_reduced` (noisy, SNR≈5) | 0.7–2.0 | |
| `chi2_by_stage` | monotone non-increasing | Verified by T2 |
| `sigma_t_m` (noisy) | < 1 µm | |
| Stage 1 free params | 4 | I0, I1, I2, B |
| Stage 2 free params | 7 | + t, alpha, R |
| Stage 3 free params | 8 | + sigma0 |
| Stage 4 free params | 10 | + sigma1, sigma2 |

---

## 13. File locations

```
soc_sewell/
├── src/fpi/
│   └── m05_calibration_inversion_2026_04_05.py
└── tests/
    └── test_s14_m05_calibration_inversion.py
```

---

## 14. Instructions for Claude Code

1. Read this entire spec, S04, S12, and S13 before writing any code.
2. Confirm S12 and S13 tests pass:
   ```bash
   pytest tests/test_m03_annular_reduction_2026_04_05.py tests/test_tolansky_2026_04_05.py -v
   ```
3. Implement `src/fpi/m05_calibration_inversion_2026_04_05.py` in this
   strict order:
   `FitFlags` → `FitConfig` (with `resolve()`) →
   `CalibrationResult` → `_stage0_seed` → `_run_lm_stage` (shared LM
   helper) → `_stage1_photometric` → `_stage2_geometry` →
   `_stage3_psf` → `_stage4_full_free` → `_convergence_guard` →
   `_compute_epsilon_cal` → `fit_calibration_fringe`
4. Use `scipy.optimize.least_squares(method='lm')` for all four fitting
   stages. Use the same weighted residual formula in every stage:
   `residuals = (data - model) / sigma_profile` (non-masked bins only).
5. Derive `sigma_` values from the covariance matrix diagonal:
   `sigma = np.sqrt(np.diag(cov))` where
   `cov = sigma2_residual * np.linalg.inv(J.T @ J)`.
   If `J.T @ J` is singular, set all stderr to `np.inf` and set
   `STDERR_NONE` flag — do not raise.
6. Set `two_sigma_X = 2.0 * sigma_X` for every fitted parameter and
   for `epsilon_cal`. Never compute `two_sigma` independently.
7. `_compute_epsilon_cal`: compute as `(2 * t_m / NE_WAVELENGTH_1_M) % 1.0`
   and propagate uncertainty as `sigma_eps = (2 / NE_WAVELENGTH_1_M) * sigma_t`.
8. Record `chi2_reduced` into `chi2_by_stage` after each of the four stages.
   After Stage 4, check monotonicity and log WARNING (not error) if any
   stage increased chi2 by more than 5%.
9. Write all 8 tests. For T5–T7, use `InstrumentParams()` from M01 as truth.
10. Run module tests:
    ```bash
    pytest tests/test_s14_m05_calibration_inversion.py -v
    ```
11. Run full suite:
    ```bash
    pytest tests/ -v
    ```
    No regressions permitted.
12. Commit:
    ```
    feat(m05): implement staged calibration inversion with Tolansky priors, 8/8 tests pass
    Implements: S14_m05_calibration_inversion_2026-04-06.md
    ```

Module docstring header:
```python
"""
M05 — Staged calibration inversion for WindCube FPI neon fringe profile.

Spec:        docs/specs/S14_m05_calibration_inversion_2026-04-06.md
Spec date:   2026-04-05
Generated:   <today>
Tool:        Claude Code
Last tested: <today>  (8/8 tests pass)
Depends on:  src.constants, src.fpi.m01_*, src.fpi.m03_*, src.fpi.tolansky_*
"""
```
