# S_L01 — FPI Pipeline Coordination Library
## WindCube SOC Pipeline — Specification v1.0
**Spec ID:** S_L01  
**Spec file:** `specs/S_L01_fpi_pipeline_2026-05-14.md`  
**Date:** 2026-05-14  
**Status:** Authoritative  
**Module:** `windcube/fpi_pipeline.py`  
**Architecture ref:** `docs/WINDCUBE_ARCH_01_pipeline_synthesis_2026-05-14.md`

**Depends on:**
- `center_finder.py` — `find_centre()`, `CentreResult`
- `annular_reduction.py` — `annular_reduce()`, `FringeProfile`
- `src/fpi/tolansky_2026-05-13.py` — `run_tolansky_2line()`, `to_m05_priors()`, `TolanskyResult`
- `src/processing/H05_calibration_inversion_2026_05_12.py` — `run_staged_inversion()`, `save_cal_result()`, `FitResult`
- `src/processing/H06_airglow_inversion_2026_05_14.py` — `run_airglow_inversion()`, `AirglowResult`, `load_cal_result()` (S_H06_refactor, fc33dc7)
- `windcube/constants.py` — authoritative numerical constants
- `src/metadata/p01_image_metadata_2026_04_06.py` — `ImageMetadata`

**Consumed by:**
- `scripts/invert_wind_map.py` (Step 4, S_batch_v2)
- `scripts/invert_single_frame.py` (Step 5, S_single_v2)

---

## 1. Purpose and scope

This spec defines `windcube/fpi_pipeline.py` — a coordination library that:

1. Re-exports the core processing functions from all five pipeline modules
   under stable, canonical names
2. Defines the canonical `CalibrationResult` and `MasterCalibration`
   dataclasses used by the batch pipeline
3. Provides a `FringeProfile` adapter so H05's internal `_FringeProfile`
   accepts `annular_reduction.py`'s `FringeProfile` without modifying
   either source file
4. Implements `average_calibrations()` — arithmetic mean of N
   `CalibrationResult` objects → `MasterCalibration`
5. Implements `process_cal_frame()` — full calibration chain in one call
6. Implements `process_science_frame()` — full science chain in one call

**What this spec does NOT do:**
- Does not modify any existing script
- Does not duplicate any algorithm
- Does not define the calibration scheduler (that is Step 3, S_L02)
- Does not contain any interactive (tkinter/dialog) code
- Does not write any files to disk (callers handle I/O)

---

## 2. Canonical dataclasses

### 2.1 CalibrationResult

Wraps the output of H05 `run_staged_inversion()` (`FitResult`) into a
stable public type with clearly named fields. Used by `average_calibrations()`
and passed to H06 as the `cal` argument.

```python
@dataclass
class CalibrationResult:
    """
    Output of a single calibration frame inversion (H05).

    All parameters are the H05 10-parameter fit result for one
    dark-subtracted neon calibration frame. The 1σ uncertainty on each
    fitted parameter is stored alongside the value.

    This is the type averaged by average_calibrations() to produce
    a MasterCalibration.
    """
    # ── Etalon geometry ──────────────────────────────────────────────────
    t_m:          float   # etalon gap [m]
    sigma_t_m:    float   # 1σ uncertainty [m]
    alpha:        float   # plate scale [rad/px, 2×2 binned]
    sigma_alpha:  float   # 1σ uncertainty [rad/px]

    # ── Reflectivities ───────────────────────────────────────────────────
    R_refl:       float   # effective reflectivity at λ₁=640.2 nm → used by H06
    sigma_R_refl: float
    R2:           float   # reflectivity at λ₂=638.3 nm (reference only)
    sigma_R2:     float

    # ── Intensity model ──────────────────────────────────────────────────
    I0:           float   # mean intensity [ADU]
    sigma_I0:     float
    I1:           float   # linear vignetting coefficient
    sigma_I1:     float
    I2:           float   # quadratic vignetting coefficient
    sigma_I2:     float

    # ── PSF ─────────────────────────────────────────────────────────────
    sigma0:       float   # PSF base width [px]
    sigma_sigma0: float
    sigma1:       float   # PSF sin variation [px] — fixed=0 in H05
    sigma2:       float   # PSF cos variation [px] — fixed=0 in H05

    # ── Background ───────────────────────────────────────────────────────
    B:            float   # CCD bias pedestal [ADU]
    sigma_B:      float

    # ── Lamp ratio ───────────────────────────────────────────────────────
    ne_ratio:       float   # λ₂/λ₁ intensity ratio (fitted)
    sigma_ne_ratio: float

    # ── Zero-wind phase reference ─────────────────────────────────────────
    epsilon_cal:       float   # fractional interference order at centre
    sigma_epsilon_cal: float

    # ── Fit quality ──────────────────────────────────────────────────────
    chi2_red:    float   # reduced chi-squared
    converged:   bool    # True if LM converged
    n_bins_used: int     # number of profile bins used in fit

    # ── Provenance ───────────────────────────────────────────────────────
    source_file:       str = ""    # path to the _profile_vs_r2.npy used
    t_tolansky_mm:     float = 0.0 # Tolansky gap seed [mm]
    eps_a_tolansky:    float = 0.0 # Tolansky ε_a seed
    alpha_tolansky:    float = 0.0 # Tolansky alpha seed [rad/px]
```

### 2.2 MasterCalibration

The arithmetic mean of N `CalibrationResult` objects. Has the same
parameter fields as `CalibrationResult` plus provenance of the averaging.
Used directly as the `cal` argument to H06's `run_airglow_inversion()`.

```python
@dataclass
class MasterCalibration:
    """
    Arithmetic mean of N CalibrationResult objects from one orbit.

    sigma_ fields are the standard error of the mean: sigma / sqrt(N).

    Exposes the same parameter interface as _CalResult (H06 internal) so
    it can be passed directly to run_airglow_inversion() and H06 helpers.
    """
    # Same fields as CalibrationResult (values are means, sigmas are SEMs)
    t_m:          float
    sigma_t_m:    float
    alpha:        float
    sigma_alpha:  float
    R_refl:       float   # = mean(R_refl) across N frames
    sigma_R_refl: float
    R2:           float
    sigma_R2:     float
    I0:           float
    sigma_I0:     float
    I1:           float
    sigma_I1:     float
    I2:           float
    sigma_I2:     float
    sigma0:       float
    sigma_sigma0: float
    sigma1:       float   # always 0.0 (fixed in H05)
    sigma2:       float   # always 0.0 (fixed in H05)
    B:            float
    sigma_B:      float
    ne_ratio:       float
    sigma_ne_ratio: float
    epsilon_cal:       float
    sigma_epsilon_cal: float

    # ── Averaging provenance ──────────────────────────────────────────────
    n_frames_averaged: int      # number of CalibrationResult objects averaged
    chi2_red_mean:     float    # mean chi2_red across frames
    n_converged:       int      # number of frames where converged=True
    orbit_number:      int = -1 # orbit this master cal belongs to (-1 = unknown)
```

---

## 3. FringeProfile adapter

H05's `run_staged_inversion()` expects an internal `_FringeProfile` object
with fields: `profile`, `r_grid`, `sigma_profile`, `masked`, `r_max_px`.

`annular_reduction.py`'s `FringeProfile` has compatible data but different
class name and additional fields.

Provide this conversion function:

```python
def to_h05_fringe_profile(fp, r_max_px: float = 110.0):
    """
    Convert annular_reduction.FringeProfile to H05's internal _FringeProfile.

    Parameters
    ----------
    fp : annular_reduction.FringeProfile
        Output of annular_reduce().
    r_max_px : float
        Outer radius used in the inversion. Bins beyond this are excluded.

    Returns
    -------
    H05._FringeProfile-compatible object suitable for run_staged_inversion().
    """
    from src.processing.H05_calibration_inversion_2026_05_12 import _FringeProfile as H05Profile
    in_range = fp.r_grid <= r_max_px
    return H05Profile(
        profile       = fp.profile[in_range].copy(),
        r_grid        = fp.r_grid[in_range].copy(),
        sigma_profile = fp.sigma_profile[in_range].copy(),
        masked        = fp.masked[in_range].copy(),
        r_max_px      = float(r_max_px),
    )
```

---

## 4. Core functions

### 4.1 `average_calibrations()`

```python
def average_calibrations(
    results: list[CalibrationResult],
    orbit_number: int = -1,
) -> MasterCalibration:
    """
    Arithmetic mean of N CalibrationResult objects.

    Parameters
    ----------
    results : list[CalibrationResult]
        N results from independent inversion of N cal frames.
        Must have len >= 1. If len == 1, returns the single result
        wrapped as a MasterCalibration.
    orbit_number : int
        Orbit number for provenance. Default -1 (unknown).

    Returns
    -------
    MasterCalibration

    Notes
    -----
    Simple arithmetic mean is used (not inverse-variance weighting)
    because all N frames are acquired under identical conditions.
    The uncertainty on each mean parameter is the standard error
    of the mean: sigma_master = mean(sigma_i) / sqrt(N).

    Frames where converged=False are included in the average with a
    warning logged. Callers may wish to filter these before averaging.
    """
```

**Implementation:** For each scalar parameter field in `CalibrationResult`
that has a corresponding `sigma_` field, compute:
```python
mean_val   = np.mean([getattr(r, field) for r in results])
mean_sigma = np.mean([getattr(r, f"sigma_{field}") for r in results])
sem        = mean_sigma / np.sqrt(len(results))
```
Set `MasterCalibration.n_frames_averaged = len(results)`,
`chi2_red_mean = np.mean([r.chi2_red for r in results])`,
`n_converged = sum(r.converged for r in results)`.

Boolean and string fields (`converged`, `source_file`) are not averaged;
they go into `n_converged` and `orbit_number` provenance fields.

### 4.2 `process_cal_frame()`

```python
def process_cal_frame(
    pixels_ds: np.ndarray,
    r_max_px: float = 110.0,
    cx_seed: float | None = None,
    cy_seed: float | None = None,
    tolansky_n_pairs: int | None = None,
    h05_r_max_px: float | None = None,
) -> CalibrationResult:
    """
    Full calibration chain for one dark-subtracted cal frame.

    Runs in order:
      1. find_centre(pixels_ds)         → CentreResult
      2. annular_reduce(pixels_ds, ...) → FringeProfile
      3. run_tolansky_2line(peak_fits)  → TolanskyResult
      4. to_m05_priors(tol_result)      → dict
      5. run_staged_inversion(...)      → FitResult
      6. CalibrationResult from FitResult + Tolansky provenance

    Parameters
    ----------
    pixels_ds : np.ndarray, shape (H, W)
        Dark-subtracted calibration frame pixels. float32 or float64.
    r_max_px : float
        Outer radius for both annular reduction and H05 inversion.
        Default 110.0 (FlatSat/flight nominal).
    cx_seed, cy_seed : float or None
        Initial centre guess for find_centre(). Default: image centre.
    tolansky_n_pairs : int or None
        Number of ring pairs for Tolansky fit. None = use all.
    h05_r_max_px : float or None
        Override r_max for H05 inversion only. None = use r_max_px.

    Returns
    -------
    CalibrationResult

    Raises
    ------
    ValueError
        If run_staged_inversion fails to converge after 4 stages.
    InsufficientRingsError
        If Tolansky analysis finds fewer than 4 valid rings.
    """
```

**Implementation order:**

```python
# 1. Centre finding
centre = find_centre(pixels_ds, cx_seed=cx_seed, cy_seed=cy_seed)

# 2. Annular reduction
fp = annular_reduce(
    pixels_ds.astype(np.float32),
    centre.cx, centre.cy,
    centre.sigma_cx, centre.sigma_cy,
    r_max_px=r_max_px,
)

# 3. Tolansky analysis (uses peak_fits_r2 from FringeProfile)
tol = run_tolansky_2line(fp.peak_fits_r2, n_pairs=tolansky_n_pairs)

# 4. M05 priors
priors = to_m05_priors(tol)
t_eff  = phase_correct_gap(
    priors["t_init_mm"] * 1e-3,
    priors["epsilon_cal_a"],
    NE_WAVELENGTH_1_AIR_M,
)

# 5. H05 inversion
h05_rmax = h05_r_max_px if h05_r_max_px is not None else r_max_px
h05_fp   = to_h05_fringe_profile(fp, r_max_px=h05_rmax)
fit      = run_staged_inversion(
    h05_fp, t_eff,
    alpha_init   = priors["alpha_init"],
    eps_a        = priors["epsilon_cal_a"],
    R1_init      = 0.53,
    R2_init      = 0.53,
)

# 6. Wrap in CalibrationResult
return CalibrationResult(
    t_m           = fit.t_m,
    sigma_t_m     = fit.sigma_t_m,
    alpha         = fit.alpha,
    sigma_alpha   = fit.sigma_alpha,
    R_refl        = fit.R1,          # R1 → H06 R_refl per H05 convention
    sigma_R_refl  = fit.sigma_R1,
    R2            = fit.R2,
    sigma_R2      = fit.sigma_R2,
    I0            = fit.I0,
    sigma_I0      = fit.sigma_I0,
    I1            = fit.I1,
    sigma_I1      = fit.sigma_I1,
    I2            = fit.I2,
    sigma_I2      = fit.sigma_I2,
    sigma0        = fit.sigma0,
    sigma_sigma0  = fit.sigma_sigma0,
    sigma1        = 0.0,
    sigma2        = 0.0,
    B             = fit.B,
    sigma_B       = fit.sigma_B,
    ne_ratio      = fit.ne_ratio,
    sigma_ne_ratio = fit.sigma_ne_ratio,
    epsilon_cal   = fit.epsilon_cal,
    sigma_epsilon_cal = fit.sigma_epsilon_cal,
    chi2_red      = fit.chi2_reduced,
    converged     = fit.converged,
    n_bins_used   = fit.n_bins_used,
    t_tolansky_mm = priors["t_init_mm"],
    eps_a_tolansky = priors["epsilon_cal_a"],
    alpha_tolansky = priors["alpha_init"],
)
```

### 4.3 `master_cal_to_h06_cal()`

H06's `run_airglow_inversion()` and internal helpers expect H06's
`_CalResult` type. Provide a conversion:

```python
def master_cal_to_h06_cal(mc: MasterCalibration):
    """
    Convert MasterCalibration to H06's internal _CalResult shim.

    Allows MasterCalibration to be passed directly to
    run_airglow_inversion() and load_cal_result()-equivalent calls.
    """
    from src.processing.H06_airglow_inversion_2026_05_14 import _CalResult
    return _CalResult(
        t_m         = mc.t_m,
        alpha       = mc.alpha,
        R_refl      = mc.R_refl,
        I0          = mc.I0,
        I1          = mc.I1,
        I2          = mc.I2,
        sigma0      = mc.sigma0,
        sigma1      = mc.sigma1,
        sigma2      = mc.sigma2,
        B           = mc.B,
        epsilon_cal = mc.epsilon_cal,
        quality_flags = 0,
    )
```

### 4.4 `process_science_frame()`

```python
def process_science_frame(
    pixels_ds: np.ndarray,
    master_cal: MasterCalibration,
    v_los_prior_ms: float,
    r_max_px: float = 110.0,
    cx_seed: float | None = None,
    cy_seed: float | None = None,
) -> AirglowResult:
    """
    Full science frame processing chain for one dark-subtracted science frame.

    Runs in order:
      1. find_centre(pixels_ds)
      2. annular_reduce(pixels_ds, ...)
      3. Apply sigma floor
      4. run_airglow_inversion(r_grid, profile, sigma, h06_cal,
                               r_max_px, v_los_prior_ms)

    Parameters
    ----------
    pixels_ds : np.ndarray, shape (H, W)
        Dark-subtracted science frame pixels. float32 or float64.
    master_cal : MasterCalibration
        Master calibration result for this orbit.
    v_los_prior_ms : float
        A-priori LOS velocity from H07 geometry engine [m/s].
        Must be computed BEFORE calling this function:
            v_los_prior_ms = geom.V_sc_LOS + geom.v_earth_LOS
    r_max_px : float
        Outer fringe radius [px]. Default 110.0.
    cx_seed, cy_seed : float or None
        Centre seed. If None, use image centre.

    Returns
    -------
    AirglowResult
    """
```

**Implementation:**

```python
# 1. Centre finding
centre = find_centre(pixels_ds, cx_seed=cx_seed, cy_seed=cy_seed)

# 2. Annular reduction
fp = annular_reduce(
    pixels_ds.astype(np.float32),
    centre.cx, centre.cy,
    centre.sigma_cx, centre.sigma_cy,
    r_max_px=r_max_px,
)

# 3. Prepare profile arrays with sigma floor
in_range    = fp.r_grid <= r_max_px
r_grid      = fp.r_grid[in_range]
profile_adu = fp.profile[in_range]
sigma_adu   = fp.sigma_profile[in_range].copy()
bad         = ~np.isfinite(sigma_adu)
sigma_adu[bad] = float(np.nanmedian(fp.sigma_profile)) if np.any(np.isfinite(fp.sigma_profile)) else 1.0
s_floor     = max(1.0, float(np.median(profile_adu)) * 0.005)
sigma_adu   = np.maximum(sigma_adu, s_floor)

# 4. H06 inversion
h06_cal = master_cal_to_h06_cal(master_cal)
return run_airglow_inversion(
    r_grid         = r_grid,
    profile_adu    = profile_adu,
    sigma_adu      = sigma_adu,
    cal            = h06_cal,
    r_max_px       = r_max_px,
    v_los_prior_ms = v_los_prior_ms,
)
```

---

## 5. Re-exports

At the top of `windcube/fpi_pipeline.py`, import and re-export all public
functions and types that downstream code needs, so callers only need to
import from `windcube.fpi_pipeline`:

```python
# Centre finding
from center_finder import find_centre, CentreResult             # noqa: F401

# Annular reduction
from annular_reduction import annular_reduce, FringeProfile     # noqa: F401

# Tolansky
from src.fpi.tolansky_2026-05-13 import (                       # noqa: F401
    run_tolansky_2line, to_m05_priors, TolanskyResult,
    print_rectangular_array, plot_tolansky_result,
)

# H05
from src.processing.H05_calibration_inversion_2026_05_12 import (  # noqa: F401
    run_staged_inversion, save_cal_result,
)

# H06
from src.processing.H06_airglow_inversion_2026_05_14 import (   # noqa: F401
    run_airglow_inversion, AirglowResult, load_cal_result,
)
```

**Note on import paths:** The exact import syntax for `center_finder.py`,
`annular_reduction.py`, and `tolansky_2026-05-13.py` depends on where
those files live in the repo. Claude Code must verify the actual paths
before writing the imports. Use relative or absolute imports as appropriate
to the repo structure. If a module is not on `sys.path`, add the relevant
directory using the same pattern already used in H05/H06
(`REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent`).

---

## 6. Required constants

The following constants from `windcube/constants.py` are used internally:

```python
from windcube.constants import (
    NE_WAVELENGTH_1_AIR_M,   # 640.2248e-9
    EARTH_OMEGA_RAD_S,
)
```

The following import is needed for `process_cal_frame()`:

```python
from src.fpi.airy_forward_model_2026_05_05 import phase_correct_gap
```

---

## 7. Verification tests

### T1 — Module imports cleanly

```python
def test_fpi_pipeline_imports():
    from windcube import fpi_pipeline
    assert hasattr(fpi_pipeline, 'process_cal_frame')
    assert hasattr(fpi_pipeline, 'process_science_frame')
    assert hasattr(fpi_pipeline, 'average_calibrations')
    assert hasattr(fpi_pipeline, 'CalibrationResult')
    assert hasattr(fpi_pipeline, 'MasterCalibration')
    assert hasattr(fpi_pipeline, 'AirglowResult')
```

### T2 — average_calibrations: single frame

```python
def test_average_calibrations_single():
    from windcube.fpi_pipeline import average_calibrations, CalibrationResult
    cal = _make_test_cal_result()   # helper: returns minimal CalibrationResult
    mc = average_calibrations([cal])
    assert mc.n_frames_averaged == 1
    assert abs(mc.t_m - cal.t_m) < 1e-12
    assert mc.n_converged == 1
```

### T3 — average_calibrations: five frames, values averaged correctly

```python
def test_average_calibrations_five():
    from windcube.fpi_pipeline import average_calibrations, CalibrationResult
    import numpy as np
    cals = [_make_test_cal_result(t_m=20.106e-3 + i*1e-9) for i in range(5)]
    mc = average_calibrations(cals)
    assert mc.n_frames_averaged == 5
    expected_t_m = np.mean([c.t_m for c in cals])
    assert abs(mc.t_m - expected_t_m) < 1e-15
```

### T4 — master_cal_to_h06_cal: fields map correctly

```python
def test_master_cal_to_h06_cal():
    from windcube.fpi_pipeline import (
        average_calibrations, master_cal_to_h06_cal
    )
    mc = average_calibrations([_make_test_cal_result()])
    h06cal = master_cal_to_h06_cal(mc)
    assert abs(h06cal.t_m - mc.t_m) < 1e-12
    assert abs(h06cal.R_refl - mc.R_refl) < 1e-12
    assert abs(h06cal.epsilon_cal - mc.epsilon_cal) < 1e-12
```

### T5 — to_h05_fringe_profile: shape and r_max respected

```python
def test_to_h05_fringe_profile():
    import numpy as np
    from windcube.fpi_pipeline import to_h05_fringe_profile
    from annular_reduction import FringeProfile
    # Build a minimal FringeProfile with r_grid spanning 0..150 px
    fp = _make_test_fringe_profile(r_max=150.0)
    h05fp = to_h05_fringe_profile(fp, r_max_px=110.0)
    assert np.all(h05fp.r_grid <= 110.0)
    assert h05fp.r_max_px == 110.0
    assert len(h05fp.profile) == len(h05fp.r_grid)
```

---

## 8. File location

```
soc_sewell/
├── windcube/
│   ├── __init__.py              ← add fpi_pipeline to public exports
│   └── fpi_pipeline.py          ← new module (this spec)
└── tests/
    └── test_l01_fpi_pipeline.py ← new test file (T1–T5)
```

---

## 9. Instructions for Claude Code

Read this entire spec, WINDCUBE-ARCH-01 v2, and all source files listed
in the Dependencies section before writing any code. Pay particular attention
to the import path resolution note in §5 — the actual file locations of
`center_finder.py`, `annular_reduction.py`, and `tolansky_2026-05-13.py`
must be confirmed from the repo before writing any import statement.

**Step-by-step:**

1. Search the repo for the actual locations of:
   - `center_finder.py`
   - `annular_reduction.py`
   - `tolansky_2026-05-13.py`
   Report these paths in the implementation report.

2. Create `windcube/fpi_pipeline.py` with:
   - Module docstring referencing this spec
   - All re-exports from §5 (with correct import paths)
   - `CalibrationResult` dataclass (§2.1)
   - `MasterCalibration` dataclass (§2.2)
   - `to_h05_fringe_profile()` (§3)
   - `average_calibrations()` (§4.1)
   - `process_cal_frame()` (§4.2)
   - `master_cal_to_h06_cal()` (§4.3)
   - `process_science_frame()` (§4.4)

3. Add `fpi_pipeline` to `windcube/__init__.py` exports if applicable.

4. Create `tests/test_l01_fpi_pipeline.py` with:
   - A `_make_test_cal_result()` helper returning a minimal valid
     `CalibrationResult` with realistic WindCube values
   - A `_make_test_fringe_profile()` helper returning a minimal valid
     `FringeProfile`
   - Tests T1–T5 exactly as specified in §7

5. Run: `pytest tests/test_l01_fpi_pipeline.py -v`
   All 5 tests must pass.

6. Run full suite: `pytest tests/ -v` — no regressions.

7. Commit:
   ```
   feat(l01): add windcube/fpi_pipeline.py coordination library
   Implements: S_L01_fpi_pipeline_2026-05-14.md
   Adds CalibrationResult, MasterCalibration, AirglowResult re-export,
   process_cal_frame(), process_science_frame(), average_calibrations().
   5/5 tests pass.
   ```

**Do not:**
- Modify any existing source file
- Duplicate any algorithm — call existing functions only
- Add interactive (tkinter/matplotlib) code to `fpi_pipeline.py`
- Create any intermediate files on disk inside the library functions

**If an import fails** because a source file is not importable from
the standard repo root path, use the same `sys.path.insert` pattern
already present in H05/H06 to make it importable. Report which paths
required this treatment.

---

*End of S_L01 specification v1.0 — 2026-05-14*
