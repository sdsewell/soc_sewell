# S10 — M02 Calibration Fringe Synthesis Specification

**Spec ID:** S10
**Spec file:** `docs/specs/S10_m02_calibration_synthesis_2026-04-05.md`
**Project:** WindCube FPI Pipeline
**Institution:** NCAR / High Altitude Observatory (HAO)
**Status:** Specification — ready for implementation in VS Code
**Depends on:** S01, S02, S03, S04, S09 (M01 must pass all 8 tests first)
**Used by:**
  - S11 (M04) — imports `radial_profile_to_image` from this module
  - S12 (M03) — receives 2D calibration image as input
  - S13 (M05) — receives 1D calibration fringe profile
  - S16 (INT02) — visualises calibration fringe images
**References:**
  - Harding et al. (2014) Applied Optics 53(4), Section 2.A
  - WindCube ForwardModel Reference, Section 6.3 — neon lamp lines
**Last updated:** 2026-04-05

> **Note:** This spec supersedes `m02_calibration_synthesis_spec.md` from
> the legacy repo. Repo path is `soc_sewell`. All constants reference S09
> (M01) as the single source of truth. File naming follows S01 convention.

---

## 1. Purpose

M02 generates a synthetic 2D CCD calibration fringe image from the WindCube
neon emission lamp. It is the first module in the FPI chain to produce a 2D
image — everything in M01 works in 1D radial space.

M02 has two responsibilities:

1. **1D fringe profile synthesis** — evaluate `airy_modified()` from M01 at
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

The WindCube neon lamp emits at two known wavelengths:

| Line | Wavelength | Relative intensity |
|------|------------|-------------------|
| Ne 1 | 640.2248 nm | 1.0 (reference) |
| Ne 2 | 638.2991 nm | 0.8 |

The two lines are separated by 1925.7 pm — approximately 188 FSRs apart
(FSR ≈ 10.2 pm at 640 nm for t = 20 mm). They are therefore in completely
separate spectral orders and **cannot share a wavelength grid**. A ±3 FSR
grid centred between the lines would contain neither line.

The correct approach is to call `airy_modified()` directly at each wavelength
— no instrument matrix, no grid, no matrix multiply. This is the fundamental
simplification that makes M02 much simpler than M04.

### 2.2 The radial beat pattern

When the two independent Airy ring systems are superimposed, they produce a
**radial beat envelope**: alternating regions of constructive and destructive
interference. The spatial period of this beat envelope encodes the etalon gap
`t` and is used by M05 Stage 0 to obtain a precise initial estimate of `t`
before the full inversion.

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
S_cal(r) = A̜(r; λ₁, params)
         + NE_INTENSITY_2 × A̜(r; λ₂, params)
         + B

where A̜ = airy_modified() from M01.
```

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
    then linearly interpolate profile_1d at r. Pixels beyond max(r_grid)
    are set to `bias`.

    This function is shared between M02 and M04. M04 imports it from here.
    Do not duplicate it.

    Parameters
    ----------
    profile_1d : 1D radial fringe profile in CCD counts, shape (R,)
    r_grid     : radial bin centres in pixels, shape (R,). Must start near 0.
    image_size : CCD active pixels along one side. Default 256 (2×2 binned).
    cx, cy     : fringe centre coordinates in pixels.
                 Default: (image_size - 1) / 2.0  (geometric centre)
    bias       : fill value for pixels outside r_grid range.

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
    rng             : numpy Generator. Pass default_rng(seed) for reproducibility.
                      If None, uses np.random.default_rng().

    Returns
    -------
    image_noisy : np.ndarray, same shape as image_noiseless, float64
    """
```

### 3.3 `synthesise_calibration_image`

```python
def synthesise_calibration_image(
    params:     'InstrumentParams',    # from M01
    image_size: int   = 256,           # CCD dimension, pixels
    cx:         float = None,          # fringe centre x (default: geometric centre)
    cy:         float = None,          # fringe centre y (default: geometric centre)
    R_bins:     int   = 500,           # radial bins in 1D profile
    add_noise:  bool  = True,          # add Poisson noise
    rng:        np.random.Generator = None,
) -> dict:
    """
    Generate a complete synthetic neon lamp calibration fringe image.

    Calls airy_modified() at NE_WAVELENGTH_1_M and NE_WAVELENGTH_2_M,
    superimposes with relative intensities, adds bias, wraps to 2D,
    optionally adds Poisson noise.

    Parameters
    ----------
    params     : InstrumentParams from M01.
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
        'params'          : InstrumentParams used (for M05 reference)
    """
```

---

## 4. Neon wavelength constants

These constants are defined in M01 (S09) and imported by M02. Do not
redefine them in M02.

```python
from fpi.m01_airy_forward_model import (
    InstrumentParams,
    airy_modified,
    NE_WAVELENGTH_1_M,   # 640.2248e-9 m
    NE_WAVELENGTH_2_M,   # 638.2991e-9 m
    NE_INTENSITY_2,      # 0.8
)
```

---

## 5. Verification tests

All 8 tests in `tests/test_m02_calibration_synthesis_2026-04-05.py`.

### T1 — Output shapes correct

```python
def test_output_shapes():
    """All returned arrays must have the expected shapes."""
    params = InstrumentParams()
    result = synthesise_calibration_image(params, image_size=256,
                                          R_bins=500, add_noise=False)
    assert result['image_2d'].shape        == (256, 256)
    assert result['image_noiseless'].shape == (256, 256)
    assert result['profile_1d'].shape      == (500,)
    assert result['r_grid'].shape          == (500,)
```

### T2 — Noiseless image everywhere positive

```python
def test_image_positivity():
    """Noiseless calibration image must be everywhere positive."""
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
    Tests radial_profile_to_image correctly implements circular geometry.
    """
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
    params = InstrumentParams()
    r_default = synthesise_calibration_image(params, add_noise=False,
                                              cx=127.5, cy=127.5)
    r_shifted = synthesise_calibration_image(params, add_noise=False,
                                              cx=137.5, cy=137.5)
    assert not np.allclose(r_default['image_2d'], r_shifted['image_2d']), \
        "Shifting fringe centre had no effect on image"
```

### T8 — 1D profile matches direct M01 evaluation

```python
def test_profile_matches_m01():
    """
    synthesise_calibration_image 1D profile must equal direct superposition
    of two airy_modified() calls. Tests that M02 is a correct wrapper of M01.
    """
    from fpi.m01_airy_forward_model import (
        InstrumentParams, airy_modified,
        NE_WAVELENGTH_1_M, NE_WAVELENGTH_2_M, NE_INTENSITY_2
    )
    params = InstrumentParams()
    R_bins = 500
    r = np.linspace(0, params.r_max, R_bins)

    A1 = airy_modified(r, NE_WAVELENGTH_1_M, params.t, params.R_refl,
                       params.alpha, params.n, params.r_max,
                       params.I0, params.I1, params.I2,
                       params.sigma0, params.sigma1, params.sigma2)
    A2 = airy_modified(r, NE_WAVELENGTH_2_M, params.t, params.R_refl,
                       params.alpha, params.n, params.r_max,
                       params.I0, params.I1, params.I2,
                       params.sigma0, params.sigma1, params.sigma2)
    expected = A1 + NE_INTENSITY_2 * A2 + params.B

    result = synthesise_calibration_image(params, R_bins=R_bins, add_noise=False)
    np.testing.assert_allclose(result['profile_1d'], expected, rtol=1e-10,
        err_msg="M02 profile does not match direct M01 superposition")
```

---

## 6. Expected numerical values

For `InstrumentParams()` defaults, `image_size=256`, `R_bins=500`:

| Quantity | Expected | Test |
|----------|----------|------|
| Image shape | (256, 256) | T1 |
| Profile shape | (500,) | T1 |
| All noiseless pixel values | > 0 | T2 |
| Circular symmetry CV | < 0.01 | T3 |
| Beat peak ratio | > 1.10 | T4 |
| Poisson Var/Mean | 0.8–1.2 | T5 |
| Profile vs M01 direct | rtol < 1e-10 | T8 |

---

## 7. File locations in repository

```
soc_sewell/
├── fpi/
│   ├── __init__.py
│   ├── m01_airy_forward_model_2026-04-05.py   ← S09, already complete
│   └── m02_calibration_synthesis_2026-04-05.py ← this module
├── tests/
│   └── test_m02_calibration_synthesis_2026-04-05.py
└── docs/specs/
    └── S10_m02_calibration_synthesis_2026-04-05.md ← this file
```

---

## 8. Instructions for Claude Code

1. Read this entire spec AND S09 (M01) before writing any code.
2. Confirm M01 tests pass: `pytest tests/test_m01_airy_forward_model_*.py -v`
3. Implement `fpi/m02_calibration_synthesis_2026-04-05.py` with functions
   in this strict order:
   `radial_profile_to_image` → `add_poisson_noise` → `synthesise_calibration_image`
4. Import all constants from M01 — do not redefine `NE_WAVELENGTH_1_M`,
   `NE_WAVELENGTH_2_M`, `NE_INTENSITY_2` in M02.
5. `radial_profile_to_image` must use `np.interp` for the radius-to-pixel
   mapping. The fill value for out-of-range pixels must be `bias`, not zero.
6. `add_poisson_noise` must clip to zero before `rng.poisson()` — negative
   counts are unphysical.
7. Write all 8 tests in `tests/test_m02_calibration_synthesis_2026-04-05.py`.
8. Run: `pytest tests/test_m02_calibration_synthesis_2026-04-05.py -v`
   All 8 must pass.
9. Run full suite: `pytest tests/ -v` — M01 + M02 all pass.
10. Commit: `feat(m02): implement calibration fringe synthesis, 8/8 tests pass`
11. Do not implement S11 (M04) until this commit is confirmed.

Module docstring header:
```python
"""
Module:      m02_calibration_synthesis_2026-04-05.py
Spec:        docs/specs/S10_m02_calibration_synthesis_2026-04-05.md
Author:      Claude Code
Generated:   <today>
Last tested: <today>  (8/8 tests pass)
Project:     WindCube FPI Pipeline — NCAR/HAO
Repo:        soc_sewell
"""
```
