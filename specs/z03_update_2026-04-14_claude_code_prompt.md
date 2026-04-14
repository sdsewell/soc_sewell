# Claude Code Prompt — Z03 Update: Expose All 10 Inversion Parameters
**Date:** 2026-04-14  
**Repo:** `soc_sewell`  
**Spec:** `specs/z03_synthetic_calibration_image_generator_spec_2026-04-14.md`  
**Target module:** `src/fpi/z03_synthetic_calibration_image_generator.py`  
**Test file:** `tests/test_z03.py`

---

## Objective

Update the Z03 synthetic calibration image generator to expose all 10 M05
inversion free parameters as interactive prompts, and replace the standalone
`airy()` helper with direct calls to `m01.airy_modified()` and
`m02.radial_profile_to_image()`.

---

## Pre-Flight Reads (do these first, before touching any code)

```
Read:  specs/z03_synthetic_calibration_image_generator_spec_2026-04-14.md
Read:  src/fpi/m01_airy_forward_model_2026_04_05.py
Read:  src/fpi/m02_calibration_synthesis_2026_04_05.py
Read:  src/fpi/z03_synthetic_calibration_image_generator.py   (current implementation)
Read:  tests/test_z03.py                                       (current tests)
```

Do **not** read any other source files unless a specific import error forces it.
Do **not** proceed to Task 1 until all five reads are complete.

---

## Baseline Check

Before making any changes, run the existing test suite and record the result:

```bash
cd /path/to/soc_sewell
pytest tests/test_z03.py -v 2>&1 | tail -20
```

Record the baseline pass/fail count in your final report.

---

## Tasks

Work through these tasks in order. Run `pytest tests/test_z03.py -v` after
**every** task. If any previously passing test breaks, stop and diagnose
before continuing to the next task.

---

### Task 1 — Update imports and module-level constants

In `z03_synthetic_calibration_image_generator.py`:

1. Add imports at the top of the file:
   ```python
   from src.fpi.m01_airy_forward_model_2026_04_05 import (
       InstrumentParams,
       airy_modified,
   )
   from src.fpi.m02_calibration_synthesis_2026_04_05 import radial_profile_to_image
   ```
   If the import paths differ from above, use whatever the existing file uses for
   its current M01/M02 imports (check the existing imports section first).

2. Remove (or comment out) any module-level `R`, `B_dc` constants that are
   currently hardcoded. These are now user-prompted parameters. Specifically:
   - Remove: `R = 0.82` (or equivalent)
   - Remove: `B_dc = 500` (or equivalent)
   Keep everything else that is currently at module level unchanged.

3. Add or update these module-level fixed constants (non-inversion parameters):
   ```python
   SIGMA_READ    = 50.0          # ADU — CCD97 EM gain regime read noise
   PIX_M         = 32.0e-6       # m   — CCD97 16 µm native × 2×2 binning
   CX_DEFAULT    = 137.5         # px  — geometric centre, 276-col array
   CY_DEFAULT    = 129.5         # px  — geometric centre, 260-row active region
   R_MAX_PX      = 110.0         # px  — FlatSat/flight max usable radius
   R_BINS        = 2000          # radial bins (must be ≥ 2000)
   N_REF         = 1.0           # refractive index, air gap
   NROWS, NCOLS  = 260, 276      # WindCube Level-0 image dimensions
   N_META_ROWS   = 4             # S19 header rows
   LAM_640       = 640.2248e-9   # m — Ne primary line (Burns et al. 1950)
   LAM_638       = 638.2991e-9   # m — Ne secondary line (Burns et al. 1950)
   ```
   If any of these already exist under different names, rename them to match
   exactly (update all call sites).

**Gate:** `pytest tests/test_z03.py -v` — all previously passing tests must
still pass before proceeding to Task 2.

---

### Task 2 — Expand the SynthParams dataclass

Locate the `SynthParams` dataclass (or namedtuple, or plain dict — whatever
the existing code uses to hold prompted parameter values). Expand it to hold
all 11 user-prompted parameters:

```python
@dataclass
class SynthParams:
    # Group 1 — etalon geometry
    d_mm:      float   # etalon gap, mm
    f_mm:      float   # imaging lens focal length, mm
    # Group 2 — reflectivity and PSF
    R:         float   # plate reflectivity
    sigma0:    float   # average PSF width, pixels
    sigma1:    float   # PSF sine variation, pixels
    sigma2:    float   # PSF cosine variation, pixels
    # Group 3 — intensity envelope and bias
    snr_peak:  float   # peak signal-to-noise ratio
    I1:        float   # linear vignetting coefficient
    I2:        float   # quadratic vignetting coefficient
    B_dc:      float   # bias pedestal, ADU
    # Group 4 — source (supplemental)
    rel_638:   float   # relative intensity of 638 nm line
```

If the existing code uses a plain dict, leave it as a dict and add the 7 new
keys with the correct defaults.

**Gate:** `pytest tests/test_z03.py -v`

---

### Task 3 — Replace `airy()` with `airy_modified()`; add helper functions

#### 3a. Remove the standalone `airy()` helper

Delete the function:
```python
def airy(theta, lam, d_mm, R):
    ...
```
If it is referenced anywhere outside Stage C synthesis, note it in your report
but proceed with deletion.

#### 3b. Add `check_psf_positive(sigma0, sigma1, sigma2) -> bool`

```python
def check_psf_positive(sigma0: float, sigma1: float, sigma2: float) -> bool:
    """Return True if sigma(r) >= 0 for all r.  Condition: sigma0 >= sqrt(sigma1^2 + sigma2^2)."""
    return sigma0 >= math.sqrt(sigma1**2 + sigma2**2)
```

#### 3c. Add `check_vignetting_positive(I0, I1, I2, r_max) -> bool`

```python
def check_vignetting_positive(I0: float, I1: float, I2: float, r_max: float) -> bool:
    """Return True if I(r) = I0*(1 + I1*(r/r_max) + I2*(r/r_max)^2) > 0 for r in [0, r_max]."""
    # Evaluate at r = r_max and at vertex of parabola if it falls in [0, r_max]
    import numpy as np
    r_test = np.array([0.0, r_max / 2.0, r_max])
    # Also check parabola vertex: d/dr[I1*u + I2*u^2] = 0 → u = -I1/(2*I2)
    if abs(I2) > 1e-12:
        u_vertex = -I1 / (2.0 * I2)
        if 0.0 <= u_vertex <= 1.0:
            r_test = np.append(r_test, u_vertex * r_max)
    vals = I0 * (1.0 + I1 * (r_test / r_max) + I2 * (r_test / r_max)**2)
    return bool(np.all(vals > 0))
```

#### 3d. Add `build_instrument_params(params, derived) -> InstrumentParams`

```python
def build_instrument_params(params: SynthParams, derived) -> InstrumentParams:
    """Construct an InstrumentParams object from SynthParams + DerivedParams."""
    return InstrumentParams(
        t       = params.d_mm * 1e-3,
        R_refl  = params.R,
        n       = N_REF,
        alpha   = derived.alpha_rad_per_px,
        I0      = derived.I0,
        I1      = params.I1,
        I2      = params.I2,
        sigma0  = params.sigma0,
        sigma1  = params.sigma1,
        sigma2  = params.sigma2,
        B       = params.B_dc,
        r_max   = R_MAX_PX,
    )
```

#### 3e. Add `synthesise_profile(inst_params, rel_638) -> tuple[np.ndarray, np.ndarray]`

```python
def synthesise_profile(
    inst_params: InstrumentParams,
    rel_638: float,
) -> tuple:
    """
    Build the noise-free 1D radial fringe profile using airy_modified().

    Returns (profile_1d, r_grid).
    profile_1d includes the bias (inst_params.B) and both Ne lines.
    """
    r_grid = np.linspace(0.0, R_MAX_PX, R_BINS)

    A640 = airy_modified(
        r_grid,
        LAM_640,
        inst_params.t,
        inst_params.R_refl,
        inst_params.alpha,
        inst_params.n,
        inst_params.r_max,
        inst_params.I0,
        inst_params.I1,
        inst_params.I2,
        inst_params.sigma0,
        inst_params.sigma1,
        inst_params.sigma2,
    )
    A638 = airy_modified(
        r_grid,
        LAM_638,
        inst_params.t,
        inst_params.R_refl,
        inst_params.alpha,
        inst_params.n,
        inst_params.r_max,
        inst_params.I0,
        inst_params.I1,
        inst_params.I2,
        inst_params.sigma0,
        inst_params.sigma1,
        inst_params.sigma2,
    )

    profile_1d = A640 + rel_638 * A638 + inst_params.B
    return profile_1d, r_grid
```

**Gate:** `pytest tests/test_z03.py -v`

---

### Task 4 — Update `derive_secondary()` and `synthesise_image()`

#### 4a. Update `derive_secondary(params)`

The function must now accept `B_dc` and `R` from `params` (they are no longer
module-level constants). Update the function to:

```python
@dataclass
class DerivedParams:
    alpha_rad_per_px: float
    I0: float           # derived from snr_peak via snr_to_ipeak()
    FSR_m: float
    finesse_F: float
    finesse_N: float

def derive_secondary(params: SynthParams) -> DerivedParams:
    alpha = PIX_M / (params.f_mm * 1e-3)
    I0    = snr_to_ipeak(params.snr_peak, params.B_dc, SIGMA_READ)
    d_m   = params.d_mm * 1e-3
    FSR   = LAM_640**2 / (2.0 * N_REF * d_m)
    F     = 4.0 * params.R / (1.0 - params.R)**2
    N_R   = math.pi * math.sqrt(params.R) / (1.0 - params.R)
    return DerivedParams(
        alpha_rad_per_px = alpha,
        I0               = I0,
        FSR_m            = FSR,
        finesse_F        = F,
        finesse_N        = N_R,
    )
```

#### 4b. Update `synthesise_image()`

Replace the existing synthesis logic (which called the standalone `airy()`
helper and built the image inline) with calls to the new helpers:

```python
def synthesise_image(params: SynthParams, derived: DerivedParams) -> np.ndarray:
    """
    Return noise-free calibration fringe image, shape (NROWS, NCOLS), float64.
    Uses airy_modified() via synthesise_profile() and radial_profile_to_image().
    """
    inst_params = build_instrument_params(params, derived)
    profile_1d, r_grid = synthesise_profile(inst_params, params.rel_638)

    # radial_profile_to_image uses a square grid; trim to NROWS after.
    image_sq = radial_profile_to_image(
        profile_1d, r_grid,
        image_size = NCOLS,    # 276 — use the wider dimension
        cx = CX_DEFAULT,
        cy = CY_DEFAULT,
        bias = params.B_dc,
    )

    # Trim to (NROWS, NCOLS) = (260, 276)
    # image_sq is (276, 276); keep the central 260 rows.
    row_start = (NCOLS - NROWS) // 2    # = 8
    image_out = image_sq[row_start : row_start + NROWS, :]
    return image_out
```

> **Implementation note:** If the current `synthesise_image()` already
> produces a (260, 276) output by a different method, verify that the new
> version also produces (260, 276). The trim arithmetic above assumes NCOLS=276
> and NROWS=260; (276−260)//2 = 8 rows cropped from each side.

**Gate:** `pytest tests/test_z03.py -v`

---

### Task 5 — Expand `prompt_all_params()`

Replace the existing prompt function (currently 4 prompts) with the full
11-prompt version organised into 4 groups. Each prompt must call
`_validated_prompt()` with the hard and warning bounds from spec Section 9.

Banner text update — replace the old banner with:

```
╔══════════════════════════════════════════════════════════════╗
║  Z03  Synthetic Calibration Image Generator                  ║
║  WindCube SOC — soc_sewell                                   ║
╚══════════════════════════════════════════════════════════════╝

This script synthesises a matched calibration + dark image pair
in authentic WindCube .bin format, suitable for ingestion by
Z01, Z02, or any S01-based pipeline module.

You will be prompted for 10 instrument/fringe parameters (all
free parameters of the M05 inversion) plus 1 source parameter.
Press <Enter> to accept the default shown in parentheses.
```

Prompt sequence and validation bounds:

```
GROUP 1 — ETALON GEOMETRY
  d_mm     default=20.106  hard=(15.0, 25.0)    warn=(19.0, 21.5)
  f_mm     default=199.12  hard=(100.0, 300.0)  warn=(180.0, 220.0)

GROUP 2 — REFLECTIVITY AND PSF
  R        default=0.53    hard=(0.01, 0.99)    warn=(0.3, 0.85)
  sigma0   default=0.5     hard=(0.0, 5.0)      warn=(0.0, 2.0)
  sigma1   default=0.1     hard=(-3.0, 3.0)     warn=(-1.0, 1.0)
  sigma2   default=-0.05   hard=(-3.0, 3.0)     warn=(-1.0, 1.0)

GROUP 3 — INTENSITY ENVELOPE AND BIAS
  snr_peak default=50.0    hard=(1.0, 500.0)    warn=(10.0, 200.0)
  I1       default=-0.1    hard=(-0.9, 0.9)     warn=(-0.5, 0.5)
  I2       default=0.005   hard=(-0.9, 0.9)     warn=(-0.5, 0.5)
  B_dc     default=300.0   hard=(0.0, 5000.0)   warn=(100.0, 1000.0)

GROUP 4 — SOURCE (supplemental)
  rel_638  default=0.8     hard=(0.0, 2.0)      warn=(0.3, 1.5)
```

After all prompts, echo a summary table showing all 11 parameters and ask
Y/n before proceeding. If the user enters "n", re-run the full prompt sequence.

After Y/n confirmation, call the two physics-consistency checks and abort with
a clear message if either fails:
```python
if not check_psf_positive(params.sigma0, params.sigma1, params.sigma2):
    print("ERROR: PSF sigma(r) goes negative. Increase sigma0 or reduce |sigma1|/|sigma2|.")
    sys.exit(1)

if not check_vignetting_positive(derived.I0, params.I1, params.I2, R_MAX_PX):
    print("ERROR: Vignetting envelope I(r) goes non-positive. Adjust I1/I2.")
    sys.exit(1)
```

**Gate:** `pytest tests/test_z03.py -v`

---

### Task 6 — Update `write_truth_json()`

The `_truth.json` sidecar must now record all 11 user params, the 5 derived
params, and the updated fixed defaults. Replace the existing function body with
output matching this schema exactly:

```python
truth = {
    "z03_version":  "1.2",
    "timestamp_utc": ...,
    "random_seed":   seed,
    "user_params": {
        "d_mm":      params.d_mm,
        "f_mm":      params.f_mm,
        "R":         params.R,
        "sigma0":    params.sigma0,
        "sigma1":    params.sigma1,
        "sigma2":    params.sigma2,
        "snr_peak":  params.snr_peak,
        "I1":        params.I1,
        "I2":        params.I2,
        "B_dc":      params.B_dc,
        "rel_638":   params.rel_638,
    },
    "derived_params": {
        "alpha_rad_per_px":    derived.alpha_rad_per_px,
        "I0_adu":              derived.I0,
        "FSR_m":               derived.FSR_m,
        "finesse_coefficient_F": derived.finesse_F,
        "finesse_N":           derived.finesse_N,
    },
    "fixed_defaults": {
        "sigma_read":  SIGMA_READ,
        "cx":          CX_DEFAULT,
        "cy":          CY_DEFAULT,
        "pix_m":       PIX_M,
        "r_max_px":    R_MAX_PX,
        "R_bins":      R_BINS,
        "nrows":       NROWS,
        "ncols":       NCOLS,
        "n_ref":       N_REF,
        "lam_640_m":   LAM_640,
        "lam_638_m":   LAM_638,
    },
    "output_cal_file":  str(path_cal.name),
    "output_dark_file": str(path_dark.name),
}
```

**Gate:** `pytest tests/test_z03.py -v`

---

### Task 7 — Update `make_diagnostic_figure()`

Update the figure title lines for the left panel (calibration image) to show
all key parameters across three lines:

```
Line 1: f"d={params.d_mm} mm   f={params.f_mm} mm   R={params.R}   SNR={params.snr_peak}"
Line 2: f"σ₀={params.sigma0}   I₁={params.I1}   I₂={params.I2}   B={params.B_dc} ADU"
Line 3: f"Ne: 640.2248 nm (×1.0)   638.2991 nm (×{params.rel_638})"
```

The right panel (dark image) title:
```
f"Dark image — B={params.B_dc} ADU, σ_read={SIGMA_READ} ADU"
```

No other changes to the figure layout.

**Gate:** `pytest tests/test_z03.py -v`

---

### Task 8 — Update tests in `tests/test_z03.py`

#### 8a. Fix any tests that hardcoded `R = 0.82` or `B_dc = 500`

Search for `0.82`, `500`, or any reference to the old fixed constants in
`tests/test_z03.py`. Update those tests to use the new defaults
(`R = 0.53`, `B_dc = 300`) or to pass explicit values.

#### 8b. Update `test_truth_json_complete`

The test must verify all 11 keys are present in `user_params`:

```python
def test_truth_json_complete(truth_json_path):
    with open(truth_json_path) as f:
        truth = json.load(f)
    expected_user_keys = {
        "d_mm", "f_mm", "R", "sigma0", "sigma1", "sigma2",
        "snr_peak", "I1", "I2", "B_dc", "rel_638"
    }
    assert expected_user_keys == set(truth["user_params"].keys())
    assert "output_cal_file"  in truth
    assert "output_dark_file" in truth
    assert "finesse_N"        in truth["derived_params"]
```

#### 8c. Add `test_psf_broadening_effect`

```python
def test_psf_broadening_effect(default_params):
    """Non-zero sigma0 must produce broader fringes than sigma0=0."""
    import numpy as np
    from src.fpi.z03_synthetic_calibration_image_generator import (
        synthesise_profile, build_instrument_params, derive_secondary, SynthParams
    )

    params_sharp = SynthParams(
        d_mm=20.106, f_mm=199.12, R=0.53,
        sigma0=0.0, sigma1=0.0, sigma2=0.0,
        snr_peak=50.0, I1=0.0, I2=0.0, B_dc=300.0, rel_638=0.8
    )
    params_broad = SynthParams(
        d_mm=20.106, f_mm=199.12, R=0.53,
        sigma0=2.0, sigma1=0.0, sigma2=0.0,
        snr_peak=50.0, I1=0.0, I2=0.0, B_dc=300.0, rel_638=0.8
    )

    derived_sharp = derive_secondary(params_sharp)
    derived_broad = derive_secondary(params_broad)

    inst_sharp = build_instrument_params(params_sharp, derived_sharp)
    inst_broad = build_instrument_params(params_broad, derived_broad)

    prof_sharp, r_grid = synthesise_profile(inst_sharp, 0.8)
    prof_broad, _      = synthesise_profile(inst_broad, 0.8)

    # Find FWHM of the first fringe peak in each profile
    # Compare standard deviation of the profiles as a proxy for broadening
    assert np.std(prof_broad) < np.std(prof_sharp), \
        "PSF broadening should smooth (reduce std of) the fringe profile"
```

#### 8d. Add `test_vignetting_effect`

```python
def test_vignetting_effect(default_params):
    """Non-zero I1 must produce a measurable radial intensity gradient."""
    import numpy as np
    from src.fpi.z03_synthetic_calibration_image_generator import (
        synthesise_image, derive_secondary, SynthParams
    )

    # Flat illumination (I1 = I2 = 0)
    params_flat = SynthParams(
        d_mm=20.106, f_mm=199.12, R=0.53,
        sigma0=0.5, sigma1=0.0, sigma2=0.0,
        snr_peak=50.0, I1=0.0, I2=0.0, B_dc=300.0, rel_638=0.8
    )
    # Strong vignetting (I1 = -0.4)
    params_vig = SynthParams(
        d_mm=20.106, f_mm=199.12, R=0.53,
        sigma0=0.5, sigma1=0.0, sigma2=0.0,
        snr_peak=50.0, I1=-0.4, I2=0.0, B_dc=300.0, rel_638=0.8
    )

    derived_flat = derive_secondary(params_flat)
    derived_vig  = derive_secondary(params_vig)

    img_flat = synthesise_image(params_flat, derived_flat)
    img_vig  = synthesise_image(params_vig,  derived_vig)

    # Mean intensity in the central 20×20 patch vs outer annulus should differ
    cx, cy = 137, 129
    r_half = 55   # half of r_max

    def annulus_mean(img, r_inner, r_outer):
        rows, cols = np.ogrid[:img.shape[0], :img.shape[1]]
        r_map = np.sqrt((cols - cx)**2 + (rows - cy)**2)
        mask = (r_map >= r_inner) & (r_map < r_outer)
        return float(img[mask].mean())

    inner_flat  = annulus_mean(img_flat, 0, r_half)
    outer_flat  = annulus_mean(img_flat, r_half, 110)
    inner_vig   = annulus_mean(img_vig,  0, r_half)
    outer_vig   = annulus_mean(img_vig,  r_half, 110)

    ratio_flat = inner_flat / outer_flat
    ratio_vig  = inner_vig  / outer_vig

    # Vignetting should make the inner/outer ratio more extreme
    assert ratio_vig > ratio_flat * 1.01, \
        "Vignetting (I1 < 0) should reduce outer intensity relative to inner"
```

**Gate:** `pytest tests/test_z03.py -v` — all 12 tests must pass.

---

## Stopping Rule

If tests are not passing after a total of ~15 minutes of active debugging,
**stop**. Do not continue modifying code in an attempt to force tests to pass.
Instead, proceed directly to the Final Report and document exactly what is
failing and why.

---

## Final Report

Paste back to Claude.ai using this exact format:

```
=== Z03 UPDATE REPORT ===

BASELINE:  X/10 tests passing before changes

TASKS COMPLETED:
  Task 1 (imports + constants):     [DONE / PARTIAL / SKIPPED]
  Task 2 (SynthParams dataclass):   [DONE / PARTIAL / SKIPPED]
  Task 3 (airy_modified helpers):   [DONE / PARTIAL / SKIPPED]
  Task 4 (derive_secondary + synthesise_image): [DONE / PARTIAL / SKIPPED]
  Task 5 (prompt_all_params):       [DONE / PARTIAL / SKIPPED]
  Task 6 (write_truth_json):        [DONE / PARTIAL / SKIPPED]
  Task 7 (diagnostic figure):       [DONE / PARTIAL / SKIPPED]
  Task 8 (tests):                   [DONE / PARTIAL / SKIPPED]

FINAL TEST RESULT:  X/12 tests passing

DEVIATIONS FROM SPEC:
  [List any place where the implementation differs from the spec,
   or "None" if implementation matches spec exactly.]

OPEN ISSUES:
  [List any failing tests, import errors, or unresolved questions,
   or "None" if everything passed.]

FILES MODIFIED:
  src/fpi/z03_synthetic_calibration_image_generator.py
  tests/test_z03.py
  [any others]
```

Do **not** include inline code diffs in the report — just the structured
summary above.
