# H02 — Calibration Fringe Synthesis

**Spec ID:** H02
**Spec file:** `docs/specs/H02_calibration_synthesis_2026-05-05.md`
**Project:** WindCube FPI Pipeline
**Institution:** NCAR / High Altitude Observatory (HAO)
**Status:** Authoritative
**Depends on:** S01, S02, S03, S04, H01 (Airy forward model — must pass all 15 tests first)
**Used by:**
  - S11 (M04 — airglow synthesis) — imports `radial_profile_to_image` from this module
  - S12 (M03 — annular reduction) — receives 2D calibration image as input
  - H05 (calibration inversion) — receives 1D calibration fringe profile
  - S16 (INT02) — visualises calibration fringe images
**References:**
  - Harding et al. (2014) Applied Optics 53(4), Section 2.A
  - Burns, Adams & Longwell (1950) — Ne I spectroscopic standards (air wavelengths)
**Last updated:** 2026-05-05

> **What changed from 2026-04-05:**
> 1. **Spec ID corrected** from S10 to H02 throughout. All references to
>    S09/M01 updated to H01/`airy_forward_model`.
> 2. **`NE_WAVELENGTH_1_M` / `NE_WAVELENGTH_2_M` renamed** to
>    `NE_WAVELENGTH_1_AIR_M` / `NE_WAVELENGTH_2_AIR_M` throughout, consistent
>    with H01 §3 canonical naming (explicit about air vs vacuum wavelengths).
> 3. **`NE_INTENSITY_2` corrected from 0.8 to 0.36** in Section 2.1 table and
>    Section 4 import block. The value 0.8 in the 2026-04-05 spec was a
>    carry-over from an early draft. The authoritative value in
>    `windcube/constants.py` (and H01 §3) is 0.36. If the implementation was
>    generated from the old spec, T4 (beat pattern) may still pass because a
>    beat exists regardless of ratio, but T8 (profile vs H01 direct) will catch
>    the discrepancy. Verify and correct if needed.
> 4. **Constants import block (Section 4)** now imports from
>    `windcube/constants.py` directly, consistent with H01 §9 (constants
>    placement rule). H02 no longer re-exports neon constants from H01;
>    both modules draw from the single source of truth.
> 5. **`airy_modified()` call in T8** updated to use `InstrumentParams` object
>    rather than positional arguments, matching H01 §6 current signature.
> 6. **`FOCAL_LENGTH_M` never appeared in H02** and does not appear here.
>    No change needed on that front, but confirmed absent.
> 7. **File locations and module docstring** updated to reflect new date suffix.
> 8. **Instructions for Claude Code** (Section 8) expanded to full structured
>    task sequence with report template, matching H01/H05/H06 pattern.

---

## 1. Purpose

H02 generates a synthetic 2D CCD calibration fringe image from the WindCube
neon emission lamp. It is the first module in the FPI chain to produce a 2D
image — everything in H01 works in 1D radial space.

H02 has two responsibilities:

1. **1D fringe profile synthesis** — evaluate `airy_modified()` from H01 at
   each of the two neon spectral lines, superimpose them with their known
   relative intensities, add the CCD bias. Produces the 1D radial fringe
   profile `S_cal(r)`.

2. **2D image generation** — wrap `S_cal(r)` into a 2D CCD image using
   circular symmetry, then add Poisson photon noise. Produces the synthetic
   L1B image passed to M03.

The `radial_profile_to_image()` function defined here is **shared with M04**.
M04 imports it directly — do not duplicate it.

---

## 2. Physical background

### 2.1 Why two neon lines — and why not a shared wavelength grid

The WindCube neon lamp emits at two known wavelengths (Burns et al. 1950
IAU standards, air wavelengths):

| Line | Wavelength | Relative intensity |
|------|------------|--------------------|
| Ne 1 | 640.2248 nm | 1.0 (reference) |
| Ne 2 | 638.2991 nm | 0.36 |

The two lines are separated by 1925.7 pm — approximately 188 FSRs apart
(FSR ≈ 10.2 pm at 640 nm for t = 20 mm). They are therefore in completely
separate spectral orders and **cannot share a wavelength grid**. A ±3 FSR
grid centred between the lines would contain neither line.

The correct approach is to call `airy_modified()` directly at each wavelength
— no instrument matrix, no wavelength grid, no matrix multiply. This is the
fundamental simplification that makes H02 much simpler than M04.

### 2.2 The radial beat pattern

When the two independent Airy ring systems are superimposed, they produce a
**radial beat envelope**: alternating regions of constructive and destructive
interference. The spatial period of this beat envelope encodes the etalon gap
`t` and is used by the Tolansky pipeline (S13) to obtain a precise initial
estimate of `t` before the full H05 inversion.

Beat period derivation:
```
Δr_beat ≈ FSR / |d(FSR)/d(r²)| ≈ function of t, α, λ₁, λ₂
```
The beat is visible as amplitude modulation of the fringe peaks in the 1D
profile — bright peaks alternating with suppressed peaks. T4 verifies this.

### 2.3 Source model — delta functions

Both neon lines are treated as perfect spectral delta functions. The natural
linewidth of neon emission is negligible compared to the FPI instrument
linewidth (~10 pm FSR / finesse ≈ 2 pm). No thermal broadening, no
wavelength grid needed:

```
S_cal(r) = airy_modified(r; λ₁, params)
         + NE_INTENSITY_2 × airy_modified(r; λ₂, params)
         + B

where airy_modified() is from H01,
      NE_INTENSITY_2 = 0.36 (from windcube/constants.py),
      B = params.B (CCD bias from InstrumentParams).
```

This is the 1D analogue of H01 §4.2's two-line neon source model. H02
implements it by direct function calls rather than through the instrument
matrix, because the two lines are too far apart in wavelength to share a
grid (see §2.1).

---

## 3. Function signatures

Implement in this order: `radial_profile_to_image` →
`add_poisson_noise` → `synthesise_calibration_image`.

### 3.1 `radial_profile_to_image`

```python
def radial_profile_to_image(
    profile_1d:  np.ndarray,  # S(r), shape (R,), CCD counts
    r_grid:      np.ndarray,  # radial bin centres, pixels, shape (R,)
    image_size:  int   = 256, # CCD active dimension, pixels
    cx:          float = None,# fringe centre x (default: (image_size-1)/2)
    cy:          float = None,# fringe centre y (default: (image_size-1)/2)
    bias:        float = 300.0,# value for pixels beyond r_max
) -> np.ndarray:
    """
    Wrap a 1D radial fringe profile into a 2D CCD image.

    For each pixel (row, col), compute r = sqrt((col-cx)²+(row-cy)²),
    then linearly interpolate profile_1d at r using np.interp. Pixels
    beyond max(r_grid) are set to `bias`.

    This function is shared between H02 and M04. M04 imports it from here.
    Do not duplicate it.

    Parameters
    ----------
    profile_1d : 1D radial fringe profile in CCD counts, shape (R,)
    r_grid     : radial bin centres in pixels, shape (R,). Must start near 0.
    image_size : CCD active pixels along one side. Default 256 (2×2 binned).
    cx, cy     : fringe centre coordinates in pixels.
                 Default: (image_size - 1) / 2.0  (geometric centre)
    bias       : fill value for pixels outside r_grid range. Should equal
                 params.B (CCD bias) so out-of-range pixels blend with the
                 dark background.

    Returns
    -------
    image : np.ndarray, shape (image_size, image_size), float64
    """
```

### 3.2 `add_poisson_noise`

```python
def add_poisson_noise(
    image_noiseless: np.ndarray,       # shape (N, N), float64, counts ≥ 0
    rng: np.random.Generator = None,   # default_rng() if None
) -> np.ndarray:
    """
    Add Poisson photon noise to a noiseless CCD image.

    Each pixel value v is replaced by a sample from Poisson(λ=v).
    Values < 0 are clipped to 0 before sampling (physically required).

    The neon calibration image is photon-noise limited — no dark current
    or read noise term is needed for the calibration frame.

    Parameters
    ----------
    image_noiseless : float64 array, CCD counts. Must be non-negative.
    rng             : numpy Generator. Pass default_rng(seed) for
                      reproducibility. If None, uses np.random.default_rng().

    Returns
    -------
    image_noisy : np.ndarray, same shape as image_noiseless, float64
    """
```

### 3.3 `synthesise_calibration_image`

```python
def synthesise_calibration_image(
    params:     'InstrumentParams',    # from H01
    image_size: int   = 256,           # CCD dimension, pixels
    cx:         float = None,          # fringe centre x (default: geometric centre)
    cy:         float = None,          # fringe centre y (default: geometric centre)
    R_bins:     int   = 500,           # radial bins in 1D profile
    add_noise:  bool  = True,          # add Poisson noise
    rng:        np.random.Generator = None,
) -> dict:
    """
    Generate a complete synthetic neon lamp calibration fringe image.

    Calls airy_modified() from H01 at NE_WAVELENGTH_1_AIR_M and
    NE_WAVELENGTH_2_AIR_M, superimposes with relative intensities from
    windcube/constants.py, adds bias, wraps to 2D, optionally adds
    Poisson noise.

    Parameters
    ----------
    params     : InstrumentParams from H01.
    image_size : CCD active dimension in pixels. Default 256 (2×2 binned).
    cx, cy     : fringe centre in pixels. Default: geometric centre.
    R_bins     : number of radial bins in 1D profile. Default 500.
    add_noise  : if True, add Poisson photon noise. Default True.
    rng        : numpy Generator for reproducibility.

    Returns
    -------
    dict with keys:
        'image_2d'        : np.ndarray (image_size, image_size) — noisy image
        'image_noiseless' : np.ndarray (image_size, image_size) — noiseless image
        'profile_1d'      : np.ndarray (R_bins,) — 1D fringe profile (no noise)
        'r_grid'          : np.ndarray (R_bins,) — radial bin centres, pixels
        'cx'              : float — fringe centre x used
        'cy'              : float — fringe centre y used
        'params'          : InstrumentParams used (for H05 reference)
    """
```

---

## 4. Constants

All numerical constants are imported from `windcube/constants.py`. H02
does not define or re-export any constants of its own.

```python
from windcube.constants import (
    NE_WAVELENGTH_1_AIR_M,  # 640.2248e-9 m — strong neon line (air wavelength)
    NE_WAVELENGTH_2_AIR_M,  # 638.2991e-9 m — weak neon line (air wavelength)
    NE_INTENSITY_1,          # 1.0 — reference intensity
    NE_INTENSITY_2,          # 0.36 — weak/strong ratio
)
```

H01 forward model objects are imported as:

```python
from fpi.airy_forward_model import (
    InstrumentParams,
    airy_modified,
)
```

**Note on constant naming:** The `_AIR_M` suffix in `NE_WAVELENGTH_*_AIR_M`
is deliberate — it distinguishes air wavelengths (as tabulated by Burns et al.
1950 and used by the FPI in atmospheric conditions) from vacuum wavelengths.
All wavelength constants in `windcube/constants.py` follow this convention.
Never substitute vacuum wavelengths here.

**Note on `NE_INTENSITY_2`:** The authoritative value is **0.36**, set in
`windcube/constants.py` per H01 §3. Earlier drafts of this spec carried 0.8 —
that value is incorrect and must not be used. T8 will catch any mismatch
between H02's synthesis and H01's forward model.

---

## 5. Verification tests

All 8 tests in `tests/test_h02_calibration_synthesis.py`.

### T1 — Output shapes correct

```python
def test_output_shapes():
    """All returned arrays must have the expected shapes."""
    from fpi.airy_forward_model import InstrumentParams
    params = InstrumentParams()
    result = synthesise_calibration_image(params, image_size=256,
                                          R_bins=500, add_noise=False)
    assert result['image_2d'].shape        == (256, 256)
    assert result['image_noiseless'].shape == (256, 256)
    assert result['profile_1d'].shape      == (500,)
    assert result['r_grid'].shape          == (500,)
    assert result['cx'] is not None
    assert result['cy'] is not None
```

### T2 — Noiseless image everywhere positive

```python
def test_image_positivity():
    """Noiseless calibration image must be everywhere positive."""
    from fpi.airy_forward_model import InstrumentParams
    params = InstrumentParams()
    result = synthesise_calibration_image(params, add_noise=False)
    assert np.all(result['image_noiseless'] > 0), \
        "Noiseless calibration image contains non-positive values"
```

### T3 — Circular symmetry

```python
def test_circular_symmetry():
    """
    At a fixed radius, noiseless pixel values must agree to within 1%.
    Tests that radial_profile_to_image correctly implements circular geometry.
    """
    from fpi.airy_forward_model import InstrumentParams
    params = InstrumentParams()
    result = synthesise_calibration_image(params, add_noise=False)
    img = result['image_noiseless']
    cx, cy = result['cx'], result['cy']
    r_test = 50.0
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    values = []
    for a in angles:
        row = int(np.round(cy + r_test * np.sin(a)))
        col = int(np.round(cx + r_test * np.cos(a)))
        row = np.clip(row, 0, img.shape[0] - 1)
        col = np.clip(col, 0, img.shape[1] - 1)
        values.append(img[row, col])
    values = np.array(values)
    cv = np.std(values) / np.mean(values)
    assert cv < 0.01, \
        f"Circular symmetry broken: std/mean = {cv:.4f} at r={r_test} px"
```

### T4 — Radial beat pattern present

```python
def test_beat_pattern_present():
    """
    The 1D profile must show amplitude modulation from the two neon lines.
    Peak heights must vary by more than 10% (peak ratio > 1.10).
    """
    from scipy.signal import find_peaks
    from fpi.airy_forward_model import InstrumentParams
    params = InstrumentParams()
    result = synthesise_calibration_image(params, add_noise=False)
    profile = result['profile_1d']
    peaks, _ = find_peaks(profile, height=0.3 * np.max(profile))
    assert len(peaks) >= 4, \
        f"Only {len(peaks)} peaks found — not enough to measure beat pattern"
    peak_heights = profile[peaks]
    ratio = np.max(peak_heights) / np.min(peak_heights)
    assert ratio > 1.10, \
        f"No beat modulation detected: max/min peak ratio = {ratio:.3f} (expect > 1.10)"
```

### T5 — Poisson noise statistics

```python
def test_poisson_noise_statistics():
    """
    Variance of noise should equal mean signal (Poisson: Var = Mean).
    Allow 20% tolerance.
    """
    from fpi.airy_forward_model import InstrumentParams
    params = InstrumentParams()
    r1 = synthesise_calibration_image(params, add_noise=True,
                                      rng=np.random.default_rng(42))
    r2 = synthesise_calibration_image(params, add_noise=False)
    noise  = r1['image_2d'] - r2['image_noiseless']
    signal = r2['image_noiseless']
    mask   = signal > 100
    assert mask.sum() >= 100, "Insufficient high-signal pixels for test"
    ratio = np.var(noise[mask]) / np.mean(signal[mask])
    assert 0.8 < ratio < 1.2, \
        f"Poisson noise check failed: Var/Mean = {ratio:.3f} (expect ≈ 1.0)"
```

### T6 — Reproducible with fixed seed

```python
def test_reproducible_with_seed():
    """Two calls with identical seeds must produce identical noisy images."""
    from fpi.airy_forward_model import InstrumentParams
    params = InstrumentParams()
    r1 = synthesise_calibration_image(params, add_noise=True,
                                      rng=np.random.default_rng(99))
    r2 = synthesise_calibration_image(params, add_noise=True,
                                      rng=np.random.default_rng(99))
    np.testing.assert_array_equal(r1['image_2d'], r2['image_2d'],
        err_msg="Same seed must produce identical images")
```

### T7 — Custom fringe centre respected

```python
def test_custom_centre():
    """
    Shifting the fringe centre by (10, 10) px must change the image.
    Verifies cx, cy are actually used in radial_profile_to_image.
    """
    from fpi.airy_forward_model import InstrumentParams
    params = InstrumentParams()
    r_default = synthesise_calibration_image(params, add_noise=False,
                                              cx=127.5, cy=127.5)
    r_shifted = synthesise_calibration_image(params, add_noise=False,
                                              cx=137.5, cy=137.5)
    assert not np.allclose(r_default['image_2d'], r_shifted['image_2d']), \
        "Shifting fringe centre had no effect on image"
```

### T8 — 1D profile matches direct H01 evaluation

```python
def test_profile_matches_h01():
    """
    synthesise_calibration_image 1D profile must equal the direct superposition
    of two airy_modified() calls from H01. Tests that H02 is a correct wrapper.

    Uses InstrumentParams object interface (H01 §6 current signature).
    Constants imported from windcube/constants.py, not from H01 module.
    """
    from fpi.airy_forward_model import InstrumentParams, airy_modified
    from windcube.constants import (
        NE_WAVELENGTH_1_AIR_M, NE_WAVELENGTH_2_AIR_M, NE_INTENSITY_2
    )
    params = InstrumentParams()
    R_bins = 500
    r = np.linspace(0, params.r_max, R_bins)

    A1 = airy_modified(r, NE_WAVELENGTH_1_AIR_M, params)
    A2 = airy_modified(r, NE_WAVELENGTH_2_AIR_M, params)
    expected = A1 + NE_INTENSITY_2 * A2 + params.B

    result = synthesise_calibration_image(params, R_bins=R_bins, add_noise=False)
    np.testing.assert_allclose(result['profile_1d'], expected, rtol=1e-10,
        err_msg="H02 profile does not match direct H01 superposition")
```

---

## 6. Expected numerical values

For `InstrumentParams()` defaults, `image_size=256`, `R_bins=500`:

| Quantity | Expected | Notes | Test |
|----------|----------|-------|------|
| Image shape | (256, 256) | 2×2 binned | T1 |
| Profile shape | (500,) | R_bins default | T1 |
| All noiseless pixel values | > 0 | bias floor | T2 |
| Circular symmetry CV | < 0.01 | interp quality | T3 |
| Beat peak ratio | > 1.10 | two-line modulation | T4 |
| Poisson Var/Mean | 0.8–1.2 | photon statistics | T5 |
| Profile vs H01 direct | rtol < 1e-10 | NE_INTENSITY_2 = 0.36 | T8 |

---

## 7. File locations in repository

```
soc_sewell/
├── windcube/
│   └── constants.py                              ← NE_WAVELENGTH_*_AIR_M, NE_INTENSITY_2
├── src/fpi/
│   ├── __init__.py
│   ├── airy_forward_model_2026_05_05.py          ← H01
│   └── m02_calibration_synthesis_2026_05_05.py   ← this module
├── tests/
│   └── test_h02_calibration_synthesis.py
└── docs/specs/
    ├── H02_calibration_synthesis_2026-05-05.md   ← this file
    └── archive/
        └── H02_calibration_synthesis_2026-04-05.md
```

---

## 8. Instructions for Claude Code

### Preamble — read before touching any file

1. Read this entire spec (H02).
2. Read H01 (`docs/specs/H01_airy_forward_model_2026-05-05.md`) in full —
   pay particular attention to §3 (constants), §5 (`InstrumentParams`),
   and §6 (`airy_modified` signature).
3. Read `windcube/constants.py` in full.
4. Read the current `src/fpi/m02_calibration_synthesis_*.py` (latest dated
   version, if it exists).
5. Read the current `tests/test_*calibration_synthesis*.py` (if it exists).

Report which dated implementation files you found for steps 4 and 5 before
proceeding.

### Task sequence

**TASK A — Confirm H01 tests pass**

```bash
pytest tests/test_airy_forward_model*.py -v --tb=short
```

All 15 H01 tests must pass before proceeding. Stop and report any failures.

**TASK B — Verify constants**

Confirm `windcube/constants.py` exports all of the following. Report
Yes/No for each:
- `NE_WAVELENGTH_1_AIR_M`  (640.2248e-9 m)
- `NE_WAVELENGTH_2_AIR_M`  (638.2991e-9 m)
- `NE_INTENSITY_1`          (1.0)
- `NE_INTENSITY_2`          (**0.36** — not 0.8; verify this value)

If the existing implementation uses `NE_INTENSITY_2 = 0.8`, it must be
corrected in `windcube/constants.py` before proceeding. Report the value
found and whether a correction was needed.

Confirm `FOCAL_LENGTH_M` is **not** imported by H02 (it may exist in
constants.py but must not appear in this module's imports).

**TASK C — Create or update module**

If `src/fpi/m02_calibration_synthesis_*.py` already exists:
- Copy it to `src/fpi/m02_calibration_synthesis_2026_05_05.py`.
- Apply the changes below.

If it does not exist yet:
- Create `src/fpi/m02_calibration_synthesis_2026_05_05.py` from scratch
  per this spec.

Changes to apply (whether creating or updating):

1. Update module docstring (template below).
2. Replace all imports of `NE_WAVELENGTH_1_M` / `NE_WAVELENGTH_2_M` with
   `NE_WAVELENGTH_1_AIR_M` / `NE_WAVELENGTH_2_AIR_M` from
   `windcube.constants` (not from `fpi.airy_forward_model`).
3. Import `NE_INTENSITY_2` from `windcube.constants`, not from
   `fpi.airy_forward_model`.
4. Confirm `NE_INTENSITY_2` is used as `0.36` everywhere (in the synthesis
   expression, in the `synthesise_calibration_image` docstring, in any
   inline comments).
5. Update `airy_modified()` call to the `(r, lam, params)` signature from
   H01 §6. If the existing code passes positional arguments for `t`, `R`,
   `alpha`, etc., replace with the `params` object.
6. Update all references to `S09`, `M01`, `m01_*` in comments and docstrings
   to `H01`, `airy_forward_model`.
7. `radial_profile_to_image` must use `np.interp` for the radius-to-pixel
   mapping. Fill value for out-of-range pixels must be `bias`, not zero.
8. `add_poisson_noise` must clip to zero before `rng.poisson()`.

Module docstring:
```python
"""
Module:      m02_calibration_synthesis_2026_05_05.py
Spec:        docs/specs/H02_calibration_synthesis_2026-05-05.md
Author:      Claude Code
Generated:   2026-05-05
Last tested: 2026-05-05
Project:     WindCube FPI Pipeline — NCAR/HAO
Repo:        soc_sewell

Changes from prior version:
  - NE_WAVELENGTH_1_M / NE_WAVELENGTH_2_M renamed to
    NE_WAVELENGTH_1_AIR_M / NE_WAVELENGTH_2_AIR_M throughout.
  - NE_INTENSITY_2 corrected to 0.36 (was 0.8 in early drafts).
  - All constants now imported from windcube.constants, not from H01.
  - airy_modified() call updated to (r, lam, params) signature.
  - No algorithmic changes.
"""
```

**TASK D — Create or update test file**

Create `tests/test_h02_calibration_synthesis.py` by copying the existing
test file (if present) and applying these changes:

1. Update imports: all `NE_WAVELENGTH_*_M` → `NE_WAVELENGTH_*_AIR_M`;
   import from `windcube.constants`, not from `fpi.airy_forward_model`.
2. T8: replace positional `airy_modified()` arguments with the
   `(r, lam, params)` form per H01 §6.
3. `NE_INTENSITY_2` imported from `windcube.constants`.
4. All module-level imports updated to point to the new dated module.

If no test file exists yet, write all 8 tests from Section 5 verbatim.

**TASK E — Run module tests**

```bash
pytest tests/test_h02_calibration_synthesis.py -v --tb=short
```

All 8 tests must pass. Stop and report if any fail. T8 failure at this
point almost certainly means `NE_INTENSITY_2` is still 0.8 in some path —
check constants.py and all call sites.

**TASK F — Full test suite**

```bash
pytest tests/ -v --tb=short
```

Report any failures. No regressions permitted.

**TASK G — Archive old files and commit**

1. Archive the old spec:
   ```bash
   git mv docs/specs/H02_calibration_synthesis_2026-04-05.md \
           docs/specs/archive/H02_calibration_synthesis_2026-04-05.md
   ```
2. Copy this spec to `docs/specs/H02_calibration_synthesis_2026-05-05.md`.
3. Commit:
   ```
   refactor(H02): AIR wavelength naming; NE_INTENSITY_2=0.36; H01 refs updated
   Implements: H02_calibration_synthesis_2026-05-05.md
   No algorithmic changes. 8/8 tests pass.
   ```

### Report format (paste back to Claude.ai)

```
TASK A — H01 tests
  Result: N/15 pass

TASK B — Constants check
  NE_WAVELENGTH_1_AIR_M present: Yes / No
  NE_WAVELENGTH_2_AIR_M present: Yes / No
  NE_INTENSITY_1 present: Yes / No
  NE_INTENSITY_2 value found: [value] — correction needed: Yes / No
  FOCAL_LENGTH_M not imported by H02: Yes / No

TASK C — Module created/updated
  Source file: src/fpi/m02_calibration_synthesis_2026_05_05.py
  NE_WAVELENGTH_*_AIR_M used throughout: Yes / No
  NE_INTENSITY_2 = 0.36 used: Yes / No
  airy_modified() uses (r, lam, params) signature: Yes / No
  Constants from windcube.constants (not H01 module): Yes / No
  Algorithmic changes: None / [list any]

TASK D — Test file created/updated
  Test file: tests/test_h02_calibration_synthesis.py
  T8 uses (r, lam, params) signature: Yes / No
  NE_INTENSITY_2 from windcube.constants: Yes / No

TASK E — Module tests
  Result: N/8 pass
  Failures: [list]

TASK F — Full suite
  Result: N/N pass
  Unexpected failures: [list]

TASK G — Commit hash: [hash]
```
