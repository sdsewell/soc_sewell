# SPEC-CAL01 — FPI Calibration Pipeline: Merged Library and Interactive Wrapper

**Version:** 1.1  
**Date:** 2026-05-16  
**Author:** Claude AI / Scott Sewell (NCAR/HAO)  
**Repo:** `soc_sewell`  
**Spec file:** `docs/specs/SPEC-CAL01_fpi_cal_pipeline_2026-05-16.md`

---

## 1. Purpose and Scope

### 1.1 Architectural intent

The pipeline is divided into two layers with a clean boundary at `OrbitCalResult`:

```
╔══════════════════════════════════════════════════════════════╗
║  fpi_cal_lib.py  —  INSTRUMENT CHARACTERISATION LAYER       ║
║                                                              ║
║  darks → master dark                                         ║
║        → dark-subtract cal frames                            ║
║        → centre find (per cal frame)                         ║
║        → annular reduce → radial profile (per cal frame)     ║
║        → peak find + Gaussian fits (per cal frame)           ║
║        → Tolansky 2-line WLS → seeds (per cal frame)         ║
║        → average seeds over orbit                            ║
║        → H05 staged LM fit ──────────────────────────────▶ OrbitCalResult
╚══════════════════════════════════════════════════════════════╝
                              │
                              ▼  (SPEC-CAL02, out of scope here)
╔══════════════════════════════════════════════════════════════╗
║  run_airglow_inversion.py  —  SCIENCE INVERSION LAYER       ║
║                                                              ║
║  OrbitCalResult + N airglow frames                           ║
║        → H06 Harding inversion ──────────────────────────▶ AirglowResult (per frame)
╚══════════════════════════════════════════════════════════════╝
```

**`fpi_cal_lib.py` is the heart of the system.** It contains all calibration physics from dark stacking through the final H05 Levenberg-Marquardt fit. The `OrbitCalResult` it produces is the complete, validated instrument description for an orbit — and potentially multiple orbits if the etalon proves stable. It is the sole input the science inversion layer needs.

**`run_cal_pipeline.py`** drives the calibration workflow: file-dialog UX, looping over cal frames, collecting figures at each stage, and writing outputs. It is pure orchestration — no physics.

**H06 airglow inversion** (`run_airglow_inversion.py`) is a separate script that takes `OrbitCalResult` and processes an arbitrary number of science frames. It is **out of scope for this spec** and will be addressed in SPEC-CAL02.

### 1.2 Deliverables

| File | Role |
|---|---|
| `src/fpi/fpi_cal_lib.py` | Merged calibration library — all physics through H05 |
| `src/processing/run_cal_pipeline.py` | Calibration workflow driver (UI + figures + file I/O) |
| `validation/test_cal01_smoke.py` | Smoke tests |

---

## 2. Source Modules and Function Inventory

All physics is **imported, not copied** from the following existing files. These are the single source of truth and must not be modified. `fpi_cal_lib.py` re-exports their public symbols and adds thin wrapper functions.

| Source file | Symbols imported and re-exported |
|---|---|
| `airy_forward_model_2026_05_05.py` | `airy_modified`, `phase_correct_gap`, `InstrumentParams` |
| `center_finder.py` | `azimuthal_variance_centre`, `estimate_centre_uncertainty`, `CentreResult` |
| `annular_reduction.py` | `annular_reduce`, `_find_and_fit_peaks_r2`, `FringeProfile`, `PeakFitR2`, `PeakFit` |
| `tolansky_2026-05-13.py` | `run_tolansky_2line`, `load_and_split_families`, `TolanskyResult`, `plot_tolansky_result`, `to_m05_priors` |
| `H05_calibration_inversion_2026_05_12.py` | `run_staged_inversion`, `save_cal_result`, `FitResult`, `_neon_model`, `_fd_jacobian`, `_run_stage`, `_model_components` |

> Claude Code must use `find` to locate the actual repo paths for each source file before writing any import statements.

---

## 3. `fpi_cal_lib.py` — Public API

### 3.1 New thin-wrapper functions

```python
def load_bin_frame(path: pathlib.Path,
                   shape: tuple[int,int] = (260, 276)) -> np.ndarray:
    """Load a single uint16 little-endian .bin frame → float64 array (H, W)."""

def make_master_dark(paths: list[pathlib.Path],
                     shape: tuple[int,int] = (260, 276)
                     ) -> tuple[np.ndarray, np.ndarray]:
    """
    Median-stack N dark frames.

    Returns
    -------
    master_dark : float64 (H, W) — median per pixel
    dark_sigma  : float64 (H, W) — std per pixel
    """

def dark_subtract(frame: np.ndarray,
                  master_dark: np.ndarray) -> np.ndarray:
    """Return frame.astype(float64) - master_dark  (float64)."""

def find_centre(image: np.ndarray,
                cx_seed: float | None = None,
                cy_seed: float | None = None,
                var_r_max_px: float | None = None) -> CentreResult:
    """
    Two-pass azimuthal variance centre finder.

    Seeds default to image centre if None.
    Calls azimuthal_variance_centre() then estimate_centre_uncertainty().
    Returns CentreResult with cx, cy, sigma_cx, sigma_cy,
    two_sigma_cx, two_sigma_cy, cost_at_min, grid_cx, grid_cy, grid_cost.
    """

def build_radial_profile(image: np.ndarray,
                         cx: float, cy: float,
                         n_bins: int = 150,
                         r_max_px: float = 110.0) -> FringeProfile:
    """
    Mulligan r²-binned annular reduction → 1-D radial profile.

    Wraps annular_reduce(). Returns FringeProfile with fields:
    r_grid, r2_grid, profile, sigma_profile, masked, cx, cy, r_max_px.
    """

def fit_peaks(fp: FringeProfile,
              distance: int = 5,
              prominence: float = 100.0,
              fit_half_window: int = 6) -> list[PeakFitR2]:
    """
    Detect peaks in radial profile and Gaussian-fit each in r² domain.

    Wraps _find_and_fit_peaks_r2().
    Expected: 20 peaks (10 strong 640.2 nm + 10 weak 638.3 nm pairs).
    """

def peaks_to_array(peaks: list[PeakFitR2]) -> np.ndarray:
    """
    Convert list[PeakFitR2] → float64 array (N, 10) for Tolansky input.

    Columns:
      0  peak_idx          (int, cast to float)
      1  r2_raw_px2
      2  r2_fit_px2        (NaN if fit failed)
      3  sigma_r2_fit_px2  (NaN if fit failed)
      4  r_raw_px
      5  0.0               (reserved)
      6  amplitude_adu
      7  width_r2_px2      (NaN if fit failed)
      8  reduced_chi2      (NaN if fit failed)
      9  line_id           (0.0 = 640.2 nm, 1.0 = 638.3 nm; assigned by median-amp split)
    """

def run_tolansky(peak_array: np.ndarray,
                 n_pairs: int | None = None) -> TolanskyResult:
    """
    Two-line Tolansky WLS analysis.

    peak_array : (N, 10) float64 from peaks_to_array().
    Wraps run_tolansky_2line().
    Key outputs: Delta_a, eps_a, alpha_mean, d_m, Y_B_obs, chi2_dof_a, chi2_dof_b.
    """

def average_tolansky_seeds(results: list[TolanskyResult],
                           chi2_threshold: float = 5.0
                           ) -> TolanskySeedMean:
    """
    Average Tolansky seeds over N cal frames.

    Frames where chi2_dof_a > threshold OR chi2_dof_b > threshold are
    excluded with a warning printed to stdout.
    Raises ValueError if no frames pass the filter.

    Computes mean and std/sqrt(N) for: d_m, alpha_mean, eps_a, Delta_a, Y_B_obs.
    All two_sigma_ fields = exactly 2 × sigma_.
    """

def run_h05(fp: FringeProfile,
            seeds: TolanskySeedMean,
            R1_init: float = 0.53,
            R2_init: float = 0.53,
            sigma0_init: float = 0.55) -> FitResult:
    """
    Full H05 staged Levenberg-Marquardt calibration fit.

    Builds the init vector from TolanskySeedMean (see §4 Stage 6 mapping),
    calls run_staged_inversion(), and returns FitResult.

    This is the culmination of the instrument characterisation pipeline.
    The returned FitResult is packed into OrbitCalResult by the wrapper
    and constitutes the sole input to the science inversion layer.
    """
```

### 3.2 New dataclasses

```python
@dataclass
class TolanskySeedMean:
    """Per-orbit mean of Tolansky seed quantities, averaged over N cal frames."""
    n_frames_total:        int
    n_frames_used:         int
    n_frames_rejected:     int

    d_m_mean:              float   # mean etalon gap (m)
    sigma_d_m_mean:        float   # std / sqrt(N_used), m
    two_sigma_d_m_mean:    float   # = 2 × sigma_d_m_mean

    alpha_mean:            float   # mean plate scale (rad/px)
    sigma_alpha_mean:      float
    two_sigma_alpha_mean:  float

    eps_a_mean:            float   # mean fractional order, 640.2 nm line
    sigma_eps_a_mean:      float
    two_sigma_eps_a_mean:  float

    Delta_a_mean:          float   # mean r²-spacing (px²/fringe)
    sigma_Delta_a_mean:    float
    two_sigma_Delta_a_mean: float

    Y_B_obs_mean:          float   # mean amplitude ratio (638/640)


@dataclass
class OrbitCalResult:
    """
    Complete orbit instrument calibration — handoff object between layers.

    Produced by run_cal_pipeline.py (calibration layer).
    Consumed by run_airglow_inversion.py (science layer, SPEC-CAL02).
    Saved as orbit_cal_result.npy (numpy dict, allow_pickle=True).
    """
    seeds:             TolanskySeedMean
    fit:               FitResult          # H05 result — 10 free params
    source_cal_files:  list[str]
    source_dark_files: list[str]
    date_utc:          str                # ISO-8601
    pipeline_version:  str = "SPEC-CAL01 v1.1"
```

### 3.3 Constraints

- `fpi_cal_lib.py` must import cleanly with no display (no `tkinter`, no `plt.show()`).
- All plotting functions from source modules (`plot_tolansky_result`, etc.) are re-exported but never called inside the library.
- `__all__` must be defined to expose the full public API surface.

---

## 4. Pipeline Stages

### Stage 0 — Master dark frame

**`fpi_cal_lib` call:**
```python
master_dark, dark_sigma = cal.make_master_dark(dark_paths)
```

**Saves:** `master_dark.npy` alongside first dark frame.

**Figure S0 — 2×3 panel dark QA:**
- [0,0] Mean of raw darks (imshow, gray, colorbar)
- [0,1] Master dark / median (imshow, gray, colorbar)
- [0,2] `dark_sigma` map (imshow, plasma colorbar)
- [1,0] Row-mean profile (mean ADU per row vs row index)
- [1,1] Histogram of master dark ADU values (50 bins, log y-scale)
- [1,2] Summary text: N frames, mean ADU, max σ, date, frame paths

---

### Stage 1 — Cal dark subtraction and centre finding

**`fpi_cal_lib` calls (per cal frame):**
```python
raw   = cal.load_bin_frame(path)
ds    = cal.dark_subtract(raw, master_dark)
ctr   = cal.find_centre(ds)              # CentreResult
```

**Saves:** `<stem>_center.npz` per frame (keys: cx, cy, sigma_cx, sigma_cy).

**Figure S1 (per frame) — 2×3 centre-finder QA**, replicating `center_finder.py` layout:
- [0,0] Raw image + lime-square grid-best mark
- [0,1] Variance vs cx (coarse grid, actual grid points as markers)
- [0,2] Variance vs cy (coarse grid)
- [1,0] Dark-subtracted image + cyan crosshair + yellow `+` at NM result
- [1,1] Variance vs cx (fine NM scan ±5 px, ±1σ/2σ spans)
- [1,2] Variance vs cy (fine NM scan ±5 px, ±1σ/2σ spans)

---

### Stage 2 — Annular integration → radial profile

**`fpi_cal_lib` call (per frame):**
```python
fp = cal.build_radial_profile(ds, ctr.cx, ctr.cy, n_bins=150, r_max_px=110.0)
```

**Saves:** `<stem>_L1.2.npz` and `<stem>_profile_vs_r2.npy` (shape 3×N_bins: r², profile, σ).

**Figure S2 (per frame) — 1×2 profile panels:**
- [0] I (ADU) vs r (px) with σ error bars
- [1] I (ADU) vs r² (px²) with σ error bars; fringes visually equally spaced

---

### Stage 3 — Peak finding and two-line Gaussian fits

**`fpi_cal_lib` calls (per frame):**
```python
peaks    = cal.fit_peaks(fp)
peak_arr = cal.peaks_to_array(peaks)    # (N, 10) float64
```

**Family assignment in `peaks_to_array`:** median-amplitude split — peaks above median → 640.2 nm (line_id=0.0); peaks at or below → 638.3 nm (line_id=1.0).

**Saves:** `<stem>_peak_fits_r2.npy` (shape N×10).

**Figure S3a (per frame) — full profile with peaks marked:**
- I vs r² with 20 vertical dashed lines (blue = 640.2 nm, orange = 638.3 nm)
- Best-fit Gaussians overlaid per family colour

**Figure S3b (per frame) — 5×4 individual peak fits:**
- Each panel: data + error bars, initial guess (gold dashed), LM fit (red)
- Green background = fit OK; red background = failed
- Title: peak N, λ, r²_fit ± σ_r², χ²_red

**Table S3 (stdout):** peak #, family, r²_fit, σ_r², amplitude, width, χ²_red, status.

---

### Stage 4 — Tolansky two-line WLS fit (per frame)

**`fpi_cal_lib` call (per frame):**
```python
tol = cal.run_tolansky(peak_arr, n_pairs=n_pairs)   # TolanskyResult
```

**Saves:** `<stem>_tolansky.npy` (pickled dict from TolanskyResult fields).

**Figure S4 (per frame) — 4-panel Tolansky:**
```python
cal.plot_tolansky_result(tol, save_path=png_path)
```
- A: r² vs p (both neon lines, WLS fits)
- B: WLS residuals
- C: Successive Δ(r²) with CV% annotation
- D: Summary text (d_m ± σ, alpha ± σ, eps_a ± σ, χ²/ν, N_Δ, Y_B_obs)

---

### Stage 5 — Seed averaging

**`fpi_cal_lib` call:**
```python
seeds = cal.average_tolansky_seeds(tol_results, chi2_threshold=5.0)
```

**Saves:** `orbit_seeds_mean.npy` to cal frame directory.

**Figure S5 — 4-panel seed averages:**
- One panel each for d_m (mm), alpha (rad/px), eps_a, Y_B_obs
- Per-frame values as scatter with ±1σ error bars; rejected frames as red ×
- Horizontal line at orbit mean; ±1σ shaded band

**Table S5 (stdout):** per-frame and orbit-mean values for all four quantities.

---

### Stage 6 — H05 full Harding LM calibration fit

**`fpi_cal_lib` call:**
```python
fit = cal.run_h05(fp_selected, seeds)    # FitResult
```

**Init vector mapping from `TolanskySeedMean` to `run_staged_inversion`:**

| Argument | Source |
|---|---|
| `t_eff` | `seeds.d_m_mean` |
| `alpha_init` | `seeds.alpha_mean` |
| `eps_a` | `seeds.eps_a_mean` |
| `ne_ratio_init` | `seeds.Y_B_obs_mean` |
| `R1_init`, `R2_init` | 0.53 (FlatSat default) |
| `sigma0_init` | 0.55 px |

**Orbit calibration result construction (in wrapper):**
```python
orbit_cal = cal.OrbitCalResult(
    seeds             = seeds,
    fit               = fit,
    source_cal_files  = [str(p) for p in cal_paths],
    source_dark_files = [str(p) for p in dark_paths],
    date_utc          = datetime.datetime.utcnow().isoformat(),
)
np.save(cal_dir / "orbit_cal_result.npy", orbit_cal.__dict__, allow_pickle=False)
# For nested dataclasses, save as a pickled dict:
np.save(cal_dir / "orbit_cal_result.npy",
        {"seeds": seeds.__dict__, "fit": fit.__dict__,
         "source_cal_files": orbit_cal.source_cal_files,
         "source_dark_files": orbit_cal.source_dark_files,
         "date_utc": orbit_cal.date_utc,
         "pipeline_version": orbit_cal.pipeline_version},
        allow_pickle=True)
```

**`orbit_cal_result.npy` is the final output of the calibration layer.** It is loaded by `run_airglow_inversion.py` (SPEC-CAL02) with `np.load(..., allow_pickle=True).item()`.

**Figure S6 — 4-panel H05 diagnostic:**
- [0] Data + composite model + individual 640 nm and 638 nm components vs r²
  - Call `cal._model_components(...)` for the two-component breakdown
- [1] Residual (data − model) with ±1σ Poisson band
- [2] Per-stage χ²_red bar chart (stages 1–4); acceptance band 0.5–2.0 shaded green
- [3] Parameter table — all 10 fitted params with 1σ; colour-coded by chi2

---

## 5. File I/O Contract

| File | Stage | Format |
|---|---|---|
| `master_dark.npy` | S0 | float64 (260, 276) |
| `<stem>_center.npz` | S1 per frame | keys: cx, cy, sigma_cx, sigma_cy |
| `<stem>_L1.2.npz` | S2 per frame | keys: r_grid, r2_grid, profile, sigma_profile, masked, cx, cy |
| `<stem>_profile_vs_r2.npy` | S2 per frame | float64 (3, N_bins) |
| `<stem>_peak_fits_r2.npy` | S3 per frame | float64 (N_peaks, 10) |
| `<stem>_tolansky.npy` | S4 per frame | pickled dict (allow_pickle=True) |
| `orbit_seeds_mean.npy` | S5 | pickled dict (allow_pickle=True) |
| `orbit_cal_result.npy` | S6 | pickled dict (allow_pickle=True) |

Figures: displayed with `plt.show(block=False)` and saved as `<stem>_fig_S<N>.png`.

---

## 6. User Interaction (wrapper only)

| Prompt | Dialog | Title |
|---|---|---|
| Dark frames | `askopenfilenames` | `"Select dark .bin frames"` |
| Cal frames | `askopenfilenames` | `"Select neon cal .bin frames"` |
| Ring pairs | `simpledialog.askinteger` | `"Tolansky ring pairs"` |
| Cal frame for H05 | `simpledialog.askinteger` | `"Cal frame index for H05 fit (1–N)"` |
| Stage exception | `messagebox.askyesno` | `"Stage N failed — continue?"` |

Stage banners: `print(f"\n{'='*60}\n  STAGE {N}: {desc}\n{'='*60}")`

---

## 7. Physical Constants

Imported from `windcube.constants` / `airy_forward_model_2026_05_05.py`. Never redefined.

| Constant | Value | Note |
|---|---|---|
| λ₁ | 640.2248e-9 m | Burns et al. (1950) |
| λ₂ | 638.2991e-9 m | Burns et al. (1950) |
| λ_OI | 630.0304e-9 m | |
| D_25C_MM | 20.0006 mm | Pat/Nir authoritative gap |
| ICOS as-built | 20.008 mm | N_Δ seeding only |
| R_refl init | 0.53 | FlatSat |
| Image shape | (260, 276) | 2×2 binned |
| r_max_px | 110 px | |

---

## 8. Uncertainty Standards

- Every fitted quantity has `sigma_<qty>` (1σ) and `two_sigma_<qty>` (= 2 × sigma).
- `TolanskySeedMean` follows the same convention.
- `stderr = None` or `inf` from any fit → `RuntimeError` before return.
- `chi2_reduced` ∈ [0.5, 2.0] is the acceptance criterion at every fit stage.

---

## 9. Out of Scope

Deferred to SPEC-CAL02:
- H06 Harding airglow inversion
- `AirglowResult` dataclass
- `run_airglow_inversion.py` wrapper script
- Multi-orbit calibration stability assessment

---

## 10. Revision History

| Version | Date | Summary |
|---|---|---|
| 1.0 | 2026-05-16 | Initial spec |
| 1.1 | 2026-05-16 | H05 fully inside `fpi_cal_lib`; H06 deferred to SPEC-CAL02; `OrbitCalResult` is the clean layer handoff; `peaks_to_array()` added; `__all__` required; save format for `OrbitCalResult` specified |

---

## 11. Claude Code Implementation Prompt

Commit this spec to `docs/specs/` first, then paste the following block into a Claude Code terminal from the repo root.

---

```
cat PIPELINE_STATUS.md

# ══════════════════════════════════════════════════════════════════════
# SPEC-CAL01 v1.1 — fpi_cal_lib.py + run_cal_pipeline.py
# Spec: docs/specs/SPEC-CAL01_fpi_cal_pipeline_2026-05-16.md
# ══════════════════════════════════════════════════════════════════════

# ── STEP 0: Locate source files ───────────────────────────────────────
# Run these finds and record the actual paths before writing any imports:
find . -name "airy_forward_model_2026_05_05.py" -not -path "*/\.*"
find . -name "center_finder.py"                  -not -path "*/\.*"
find . -name "annular_reduction.py"              -not -path "*/\.*"
find . -name "tolansky_2026-05-13.py"            -not -path "*/\.*"
find . -name "H05_calibration_inversion_2026_05_12.py" -not -path "*/\.*"

# ── STEP 1: Create src/fpi/fpi_cal_lib.py ────────────────────────────
#
# RULE: import, never copy-paste. fpi_cal_lib.py re-exports all source
# module symbols and adds only thin wrappers.  No tkinter.  No plt.show().
#
# Structure:
#
#   """fpi_cal_lib — WindCube FPI calibration library (SPEC-CAL01 v1.1)."""
#   from __future__ import annotations
#   import datetime, pathlib
#   from dataclasses import dataclass
#   import numpy as np
#
#   # ── Re-exports from existing source modules ──────────────────────
#   from <found_path>.airy_forward_model_2026_05_05 import (
#       airy_modified, phase_correct_gap, InstrumentParams,
#   )
#   from <found_path>.center_finder import (
#       azimuthal_variance_centre, estimate_centre_uncertainty, CentreResult,
#       _variance_cost,
#   )
#   from <found_path>.annular_reduction import (
#       annular_reduce, FringeProfile, PeakFitR2, PeakFit,
#       _find_and_fit_peaks_r2,
#   )
#   from <found_path>.tolansky_2026-05-13 import (   # use importlib if hyphen is a problem
#       run_tolansky_2line, load_and_split_families,
#       TolanskyResult, plot_tolansky_result, to_m05_priors,
#   )
#   from <found_path>.H05_calibration_inversion_2026_05_12 import (
#       run_staged_inversion, save_cal_result, FitResult,
#       _neon_model, _fd_jacobian, _run_stage, _model_components,
#   )
#
# NOTE on hyphen in filename: "tolansky_2026-05-13.py" cannot be
# imported with a plain import statement. Use importlib:
#
#   import importlib.util, sys
#   _spec = importlib.util.spec_from_file_location(
#       "tolansky_2line", "<found_path>/tolansky_2026-05-13.py")
#   _mod = importlib.util.module_from_spec(_spec)
#   _spec.loader.exec_module(_mod)
#   run_tolansky_2line  = _mod.run_tolansky_2line
#   TolanskyResult      = _mod.TolanskyResult
#   plot_tolansky_result = _mod.plot_tolansky_result
#   to_m05_priors       = _mod.to_m05_priors
#   load_and_split_families = _mod.load_and_split_families
#
# ── New dataclasses (Spec §3.2) ──
#   @dataclass class TolanskySeedMean  (all fields with two_sigma_)
#   @dataclass class OrbitCalResult    (seeds, fit, source lists, date_utc, pipeline_version)
#
# ── New thin wrappers (Spec §3.1) ──
#
# load_bin_frame(path, shape=(260,276)):
#   return np.fromfile(path, dtype='<u2').reshape(shape).astype(np.float64)
#
# make_master_dark(paths, shape=(260,276)):
#   frames = np.stack([load_bin_frame(p, shape) for p in paths])
#   return np.median(frames, axis=0), frames.std(axis=0)
#
# dark_subtract(frame, master_dark):
#   return frame.astype(np.float64) - master_dark
#
# find_centre(image, cx_seed=None, cy_seed=None, var_r_max_px=None):
#   H, W = image.shape
#   cx_seed = cx_seed if cx_seed is not None else (W-1)/2.0
#   cy_seed = cy_seed if cy_seed is not None else (H-1)/2.0
#   if var_r_max_px is None:
#       var_r_max_px = min(H, W)//2 - 10
#   cx, cy, cost_min, gcx, gcy, gcost = azimuthal_variance_centre(
#       image, cx_seed, cy_seed, var_r_max_px=var_r_max_px)
#   r_min_sq  = 5.0**2
#   r_max_sq  = var_r_max_px**2
#   cost_fn   = lambda xy: _variance_cost(xy[0], xy[1], image, r_min_sq, r_max_sq, 250)
#   sigma_cx, sigma_cy = estimate_centre_uncertainty(cx, cy, cost_fn)
#   return CentreResult(cx=cx, cy=cy,
#       sigma_cx=sigma_cx, sigma_cy=sigma_cy,
#       two_sigma_cx=2*sigma_cx, two_sigma_cy=2*sigma_cy,
#       cost_at_min=cost_min,
#       grid_cx=gcx, grid_cy=gcy, grid_cost=gcost)
#
# build_radial_profile(image, cx, cy, n_bins=150, r_max_px=110.0):
#   # Check actual annular_reduce() signature and pass the right args
#   return annular_reduce(image, cx, cy, n_bins=n_bins, r_max_px=r_max_px)
#
# fit_peaks(fp, distance=5, prominence=100.0, fit_half_window=6):
#   # Check actual _find_and_fit_peaks_r2() signature
#   return _find_and_fit_peaks_r2(
#       fp.r_grid, fp.r2_grid, fp.profile, fp.sigma_profile, fp.masked,
#       distance=distance, prominence=prominence, fit_half_window=fit_half_window)
#
# peaks_to_array(peaks):
#   # Build (N, 10) array; col 9 = line_id by median-amp split
#   amps = np.array([p.amplitude_adu for p in peaks])
#   threshold = np.median(amps)
#   rows = []
#   for p in peaks:
#       line_id = 0.0 if p.amplitude_adu > threshold else 1.0
#       rows.append([float(p.peak_idx), p.r2_raw_px2,
#                    p.r2_fit_px2, p.sigma_r2_fit_px2,
#                    p.r_raw_px, 0.0, p.amplitude_adu,
#                    p.width_r2_px2, p.reduced_chi2, line_id])
#   return np.array(rows, dtype=float)
#
# run_tolansky(peak_array, n_pairs=None):
#   return run_tolansky_2line(peak_array, n_pairs=n_pairs)
#
# average_tolansky_seeds(results, chi2_threshold=5.0):
#   kept = []
#   for i, r in enumerate(results):
#       if r.chi2_dof_a > chi2_threshold or r.chi2_dof_b > chi2_threshold:
#           print(f"  WARNING: frame {i} rejected (chi2_a={r.chi2_dof_a:.2f}, "
#                 f"chi2_b={r.chi2_dof_b:.2f} > {chi2_threshold})")
#       else:
#           kept.append(r)
#   if not kept:
#       raise ValueError("No Tolansky frames passed chi2 filter.")
#   N = len(kept)
#   def mean_sem(vals):
#       a = np.array(vals); return float(a.mean()), float(a.std()/np.sqrt(N))
#   d_m, sig_d = mean_sem([r.d_m for r in kept])
#   al, sig_al = mean_sem([r.alpha_mean for r in kept])
#   ep, sig_ep = mean_sem([r.eps_a for r in kept])
#   da, sig_da = mean_sem([r.Delta_a for r in kept])
#   yb = float(np.mean([r.Y_B_obs for r in kept]))
#   return TolanskySeedMean(
#       n_frames_total=len(results), n_frames_used=N,
#       n_frames_rejected=len(results)-N,
#       d_m_mean=d_m,  sigma_d_m_mean=sig_d,  two_sigma_d_m_mean=2*sig_d,
#       alpha_mean=al, sigma_alpha_mean=sig_al, two_sigma_alpha_mean=2*sig_al,
#       eps_a_mean=ep, sigma_eps_a_mean=sig_ep, two_sigma_eps_a_mean=2*sig_ep,
#       Delta_a_mean=da, sigma_Delta_a_mean=sig_da, two_sigma_Delta_a_mean=2*sig_da,
#       Y_B_obs_mean=yb)
#
# run_h05(fp, seeds, R1_init=0.53, R2_init=0.53, sigma0_init=0.55):
#   return run_staged_inversion(
#       fp,
#       t_eff         = seeds.d_m_mean,
#       alpha_init    = seeds.alpha_mean,
#       eps_a         = seeds.eps_a_mean,
#       ne_ratio_init = seeds.Y_B_obs_mean,
#       R1_init       = R1_init,
#       R2_init       = R2_init,
#       sigma0_init   = sigma0_init,
#   )
#
# ── STEP 2: Create src/processing/run_cal_pipeline.py ─────────────────
#
# Import fpi_cal_lib exclusively. Implement S0–S6. Full figure spec in
# Spec §4. plt.show(block=False) for all figures; save PNG alongside npy.
#
# main() structure:
#   stage_banner(0, "Master dark frame")
#   dark_paths = select_files("Select dark .bin frames")
#   master_dark, dark_sigma = cal.make_master_dark(dark_paths)
#   [save master_dark.npy, make Figure S0, save PNG]
#
#   stage_banner(1, "Cal dark subtraction and centre finding")
#   cal_paths = select_files("Select neon cal .bin frames")
#   centres, ds_frames = [], []
#   for path in cal_paths:
#       raw = cal.load_bin_frame(path)
#       ds  = cal.dark_subtract(raw, master_dark)
#       ctr = cal.find_centre(ds)
#       [save _center.npz, make Figure S1, save PNG]
#       centres.append(ctr); ds_frames.append(ds)
#
#   stage_banner(2, "Annular integration")
#   profiles = []
#   for ds, ctr, path in zip(ds_frames, centres, cal_paths):
#       fp = cal.build_radial_profile(ds, ctr.cx, ctr.cy)
#       [save _L1.2.npz, _profile_vs_r2.npy, Figure S2, PNG]
#       profiles.append(fp)
#
#   stage_banner(3, "Peak finding and Gaussian fits")
#   peak_arrays = []
#   for fp, path in zip(profiles, cal_paths):
#       peaks    = cal.fit_peaks(fp)
#       peak_arr = cal.peaks_to_array(peaks)
#       [save _peak_fits_r2.npy, Figures S3a + S3b, print Table S3]
#       peak_arrays.append(peak_arr)
#
#   n_pairs = ask_integer("Tolansky ring pairs", default=len(peaks)//2)
#   stage_banner(4, "Tolansky two-line WLS")
#   tol_results = []
#   for peak_arr, path in zip(peak_arrays, cal_paths):
#       tol = cal.run_tolansky(peak_arr, n_pairs=n_pairs)
#       [save _tolansky.npy, Figure S4 via cal.plot_tolansky_result(), PNG]
#       tol_results.append(tol)
#
#   stage_banner(5, "Seed averaging")
#   seeds = cal.average_tolansky_seeds(tol_results)
#   [save orbit_seeds_mean.npy, Figure S5, print Table S5]
#
#   idx = ask_integer("Cal frame for H05 fit (1–N)", default=1) - 1
#   stage_banner(6, "H05 full Harding LM fit")
#   fit = cal.run_h05(profiles[idx], seeds)
#   orbit_cal = cal.OrbitCalResult(seeds=seeds, fit=fit, ...)
#   [save orbit_cal_result.npy, Figure S6, print completion banner]
#
# Error handling: wrap each stage in run_with_error_handling() that
# prints traceback and asks via messagebox whether to continue.
#
# ── STEP 3: Smoke tests (validation/test_cal01_smoke.py) ──────────────
#
# 1. import fpi_cal_lib as cal  — must succeed without display
# 2. TolanskySeedMean and OrbitCalResult importable from cal
# 3. make_master_dark with 3 synthetic 260×276 uint16 frames:
#    - master.shape == (260, 276)
#    - 800 <= master.mean() <= 900
#    - dark_sigma.shape == (260, 276)
# 4. dark_subtract: result.dtype == float64, abs(result.mean()) < 10
# 5. run_cal_pipeline importable without tkinter launch
#    (check for "if __name__ == '__main__': main()" guard)
#
# Run: python -m pytest validation/test_cal01_smoke.py -v
#
# ── REPORT BACK ──────────────────────────────────────────────────────
# Report:
#   FILES CREATED : repo-relative paths of all created files
#   IMPORT CHECK  : output of: python -c "import src.fpi.fpi_cal_lib as cal; print(dir(cal))"
#   TESTS         : full pytest -v output
#   DEVIATIONS    : any spec section where implementation differs
#   OPEN ISSUES   : anything needing Claude AI review
#
# ── GIT COMMIT ───────────────────────────────────────────────────────
# Update PIPELINE_STATUS.md: SPEC-CAL01 status=IMPLEMENTED,
# tests=smoke_5/5, date=2026-05-16
#
# git add src/fpi/fpi_cal_lib.py \
#         src/processing/run_cal_pipeline.py \
#         validation/test_cal01_smoke.py \
#         docs/specs/SPEC-CAL01_fpi_cal_pipeline_2026-05-16.md \
#         PIPELINE_STATUS.md
# git commit -m "feat: SPEC-CAL01 v1.1 — fpi_cal_lib + run_cal_pipeline
#
# Merged calibration library: darks→centre→profile→peaks→Tolansky
# seeds→H05 LM fit→OrbitCalResult. Interactive wrapper S0-S6 with
# Windows dialogs and diagnostic figures. H06 deferred to SPEC-CAL02.
# Also updates PIPELINE_STATUS.md"
```

---

## 12. Report-back Format

Paste the full Claude Code report output into this project conversation. Claude AI will review deviations, produce an updated spec with a new datestamp if needed, and issue a follow-up Claude Code prompt for any fixes.
