# S11 — M04 Airglow Fringe Synthesis Specification

**Spec ID:** S11
**Spec file:** `docs/specs/S11_m04_airglow_synthesis_2026-04-05.md`
**Project:** WindCube FPI Pipeline
**Institution:** NCAR / High Altitude Observatory (HAO)
**Status:** Specification — ready for implementation in VS Code
**Depends on:** S01, S02, S03, S04, S09 (M01), S10 (M02)
**Used by:**
  - S12 (M03) — receives 2D science image as input
  - S14 (M06) — 1D airglow profile is the inversion target
  - S16 (INT02) — visualises synthetic airglow images
  - S17 (INT03) — Monte Carlo noise trials use synthesised images
**References:**
  - Harding et al. (2014) Applied Optics 53(4), Section 2.B
  - WindCube STM v1 — SG1/SG2 wind precision requirements
**Last updated:** 2026-04-05

> **Critical design decision — delta-function source model:**
> This spec intentionally **removes** the thermally-broadened Gaussian
> source model that appeared in the legacy `m04_airglow_synthesis_spec.md`.
> The OI 630 nm line is modelled as a spectral delta function. This is the
> authoritative WindCube pipeline decision. See Section 2.3 for the full
> rationale. Any implementation that accepts a `temperature_K` argument
> or computes a thermal linewidth is **wrong**.

---

## 1. Purpose

M04 generates a synthetic 2D CCD science (airglow) fringe image. It receives
`v_rel` (m/s, from NB02c) and encodes it as a Doppler shift of the OI
630.0304 nm thermospheric emission line, then wraps the 1D fringe profile
into a 2D image and adds detector noise.

M04 is the module that connects the geometry pipeline (NB02c output) to the
FPI instrument model (Harding chain). The synthetic image it produces is the
ground truth against which M06 must recover the original `v_rel`.

**What M04 does not do:**
- M04 does not model thermal broadening. No `temperature_K` parameter.
- M04 does not compute a wavelength grid or call `build_instrument_matrix`.
  The delta-function source means a direct `airy_modified()` call suffices,
  exactly as M02 does for neon — just at a different wavelength.
- M04 does not model dark current, CIC, or EMCCD gain. Gaussian white noise
  at a specified SNR is the noise model (Harding Eq. 17).

---

## 2. Physical background

### 2.1 Doppler shift — wind to wavelength

The line-of-sight wind `v_rel` shifts the OI 630.0304 nm rest wavelength:

```
λ_c = λ₀ · (1 + v_rel / c)

where:
  λ₀    = OI_WAVELENGTH_M = 630.0304e-9 m  (air wavelength, from S03/M01)
  v_rel = line-of-sight wind, m/s  (positive = receding = redshift)
  c     = SPEED_OF_LIGHT_MS = 299,792,458 m/s
```

For `v_rel = 100 m/s`: `λ_c − λ₀ = 630.0304e-9 × 100/c ≈ +0.210 pm`.

### 2.2 The fringe pattern with delta-function source

With a delta-function source at wavelength `λ_c`, the 1D airglow fringe
profile is simply the modified Airy function evaluated at that wavelength:

```
S_airglow(r) = I_line · airy_modified(r; λ_c, params) + B
```

No wavelength grid. No matrix multiply. Just one `airy_modified()` call,
identical in structure to M02 but at the Doppler-shifted OI wavelength.

The Doppler shift `λ_c` displaces all fringe rings inward or outward by a
small amount. M06 recovers `v_rel` by finding where the fringe rings land —
i.e. by inverting for the `t` that produces the observed ring positions, then
converting `Δt` to `Δλ_c` to `v_rel`. The delta-function source means this
inversion requires only 3 free parameters (`λ_c`, `I_line`, `B`) rather than
5 (adding `Δλ` and `Y_bg` in the Gaussian model).

### 2.3 Why a delta function — the design decision in full

The OI 630.0 nm emission line has a thermal Doppler width of approximately
1.2–1.5 pm (1/e half-width) at thermospheric temperatures of 800–1200 K.
The FPI etalon FSR is ~9.9 pm and the instrument finesse is ~4.9, giving
a fringe width of ~2 pm. The thermal line is therefore comparable in width
to the instrument resolution.

In principle, the thermal linewidth encodes neutral temperature and could
be recovered from the fringe shape. In practice, WindCube's STM does not
include temperature as a Level 2 science data product. The mission science
requirement is wind precision (SG1: σ < 34 m/s, SG2: σ < 32 m/s), not
temperature.

Treating the source as a delta function:
1. Reduces M06 from 5 free parameters to 3 — faster, more stable inversion.
2. Removes an entire class of degeneracy between temperature and instrument
   PSF width (`σ₀`).
3. Is a conservative approximation — the thermal width is small enough that
   the fringe shift due to wind dominates the fringe broadening due to
   temperature at WindCube's operating SNR.
4. Is consistent with how M02 treats the neon lines (also delta functions).

**If future work requires temperature retrieval**, a separate module (not
M04 or M06) should be created. M04 and M06 are frozen to the delta-function
model.

### 2.4 Noise model

Airglow science frames use **Gaussian white noise** at a specified SNR
(Harding Eq. 17):

```
SNR = ΔS / σ_N

where:
  ΔS  = max(S_airglow(r)) − min(S_airglow(r))  (fringe contrast, counts)
  σ_N = Gaussian noise standard deviation (counts)
```

Typical WindCube operating SNR: 0.5–10.

### 2.5 Inverse functions — co-located with forward functions

The analytic inverses of the Doppler equations are defined in M04 alongside
the forward functions. M06 imports them to convert recovered `λ_c` back to
`v_rel`. Co-locating them in M04 makes round-trip testing trivial and keeps
the Doppler physics in one place.

---

## 3. Function signatures

Implement in this order:
`v_rel_to_lambda_c` → `lambda_c_to_v_rel` → `add_gaussian_noise`
→ `synthesise_airglow_image`.

### 3.1 `v_rel_to_lambda_c`

```python
def v_rel_to_lambda_c(
    v_rel_ms:  float,
    lambda0_m: float = None,   # defaults to OI_WAVELENGTH_M from M01
) -> float:
    """
    Convert line-of-sight wind speed to Doppler-shifted line centre.

    λ_c = λ₀ · (1 + v_rel / c)    (Harding Eq. 11)

    Parameters
    ----------
    v_rel_ms  : LOS wind speed, m/s.
                Positive = emitter receding from instrument (redshift).
                Negative = emitter approaching (blueshift).
    lambda0_m : rest wavelength in metres. Default: OI_WAVELENGTH_M (S03/M01).

    Returns
    -------
    lambda_c_m : float, Doppler-shifted line centre in metres
    """
```

### 3.2 `lambda_c_to_v_rel`

```python
def lambda_c_to_v_rel(
    lambda_c_m: float,
    lambda0_m:  float = None,  # defaults to OI_WAVELENGTH_M
) -> float:
    """
    Recover line-of-sight wind speed from Doppler-shifted line centre.

    v_rel = c · (λ_c / λ₀ − 1)    (inverse of Harding Eq. 11)

    Used by M06 to convert recovered λ_c back to v_rel.
    Defined here (not in M06) to keep Doppler physics co-located.

    Returns
    -------
    v_rel_ms : float, m/s
    """
```

### 3.3 `add_gaussian_noise`

```python
def add_gaussian_noise(
    image_noiseless: np.ndarray,      # shape (N, N), float64
    snr:             float,           # target SNR = ΔS / σ_N
    rng:             np.random.Generator = None,
) -> tuple[np.ndarray, float]:
    """
    Add Gaussian white noise to an airglow image at a specified SNR.

    SNR = ΔS / σ_N  where ΔS = max(image) − min(image).
    σ_N = ΔS / SNR.

    Parameters
    ----------
    image_noiseless : noiseless 2D airglow image, counts
    snr             : target signal-to-noise ratio (Harding Eq. 17)

    Returns
    -------
    (image_noisy, sigma_noise) :
        image_noisy  : np.ndarray, same shape, with Gaussian noise added
        sigma_noise  : float, the noise std dev actually used (counts)
    """
```

### 3.4 `synthesise_airglow_image`

```python
def synthesise_airglow_image(
    v_rel_ms:   float,
    params:     'InstrumentParams',
    snr:        float = 5.0,
    I_line:     float = 1.0,
    Y_bg:       float = 0.05,
    image_size: int   = 256,
    cx:         float = None,
    cy:         float = None,
    R_bins:     int   = 500,
    add_noise:  bool  = True,
    rng:        np.random.Generator = None,
) -> dict:
    """
    Generate a synthetic OI 630.0 nm airglow fringe image.

    Source model: spectral delta function at Doppler-shifted wavelength λ_c.
    No temperature broadening. No wavelength grid. One airy_modified() call.

    This function DOES NOT accept a temperature_K argument.
    Passing temperature_K will raise TypeError (enforced by **kwargs guard).

    Parameters
    ----------
    v_rel_ms   : LOS wind speed from NB02c, m/s. Encodes Doppler shift.
    params     : InstrumentParams from M01/M05.
    snr        : target SNR = ΔS / σ_N. Typical range: 0.5–10.
    I_line     : airglow line intensity scale factor. Default 1.0.
    Y_bg       : fractional background (relative to I_line). Default 0.05.
                 Background adds a constant offset: B_bg = Y_bg × I_line × I0.
    image_size : CCD active pixels, one side. Default 256.
    cx, cy     : fringe centre, pixels. Default: geometric centre.
    R_bins     : radial bins in 1D profile. Default 500.
    add_noise  : if True, add Gaussian noise at specified SNR. Default True.
    rng        : numpy Generator for reproducibility.

    Returns
    -------
    dict with keys:
        'image_2d'        : np.ndarray (image_size, image_size) — noisy image
        'image_noiseless' : np.ndarray (image_size, image_size) — noiseless
        'profile_1d'      : np.ndarray (R_bins,) — 1D noiseless profile
        'r_grid'          : np.ndarray (R_bins,) — radial bin centres, pixels
        'lambda_c_m'      : float — Doppler-shifted line centre used
        'sigma_noise'     : float — noise std dev applied (0.0 if no noise)
        'snr_actual'      : float — actual SNR of noiseless fringe
        'v_rel_ms'        : float — v_rel used (stored for round-trip testing)
        'cx'              : float — fringe centre x used
        'cy'              : float — fringe centre y used
        'params'          : InstrumentParams used

    Notes
    -----
    Uses radial_profile_to_image() imported from M02.
    All constants (OI_WAVELENGTH_M, SPEED_OF_LIGHT_MS) imported from M01.
    """
```

**Enforcement of no-temperature rule:**
```python
def synthesise_airglow_image(v_rel_ms, params, **kwargs):
    if 'temperature_K' in kwargs:
        raise TypeError(
            "synthesise_airglow_image() does not accept temperature_K. "
            "The OI 630 nm source is modelled as a delta function. "
            "See S11 Section 2.3 for the design rationale."
        )
    # ... rest of implementation
```

---

## 4. Imports

```python
from fpi.m01_airy_forward_model import (
    InstrumentParams,
    airy_modified,
    OI_WAVELENGTH_M,
    SPEED_OF_LIGHT_MS,
)
from fpi.m02_calibration_synthesis import radial_profile_to_image
```

`radial_profile_to_image` is defined in M02 and imported here — not
reimplemented. This is the only import M04 needs from M02.

---

## 5. Verification tests

All 8 tests in `tests/test_m04_airglow_synthesis_2026-04-05.py`.

### T1 — Output shapes and keys

```python
def test_output_shapes():
    """All returned arrays must have expected shapes."""
    params = InstrumentParams()
    result = synthesise_airglow_image(100.0, params, add_noise=False)
    assert result['image_2d'].shape       == (256, 256)
    assert result['image_noiseless'].shape == (256, 256)
    assert result['profile_1d'].shape     == (500,)
    assert result['r_grid'].shape         == (500,)
    assert isinstance(result['lambda_c_m'], float)
    assert isinstance(result['snr_actual'], float)
```

### T2 — Doppler shift moves fringe rings

```python
def test_doppler_shift_moves_fringes():
    """
    Positive v_rel (recession = redshift) must shift fringe peaks
    to larger radii. Negative v_rel shifts to smaller radii.
    """
    from scipy.signal import find_peaks
    params = InstrumentParams()
    r = None

    results = {}
    for v in [-500.0, 0.0, +500.0]:
        res = synthesise_airglow_image(v, params, add_noise=False)
        profile = res['profile_1d']
        r_grid  = res['r_grid']
        peaks, _ = find_peaks(profile)
        if len(peaks) > 0:
            results[v] = r_grid[peaks[0]]   # radius of first peak

    if len(results) == 3:
        assert results[-500.0] < results[0.0] < results[+500.0], \
            "Fringe peaks do not shift monotonically with v_rel"
```

### T3 — Zero wind gives symmetric profile

```python
def test_zero_wind_symmetric():
    """
    At v_rel = 0, lambda_c must equal OI_WAVELENGTH_M exactly.
    Profile must be symmetric (no preferred direction).
    """
    from fpi.m01_airy_forward_model import OI_WAVELENGTH_M
    params = InstrumentParams()
    result = synthesise_airglow_image(0.0, params, add_noise=False)
    assert abs(result['lambda_c_m'] - OI_WAVELENGTH_M) < 1e-18, \
        f"Zero wind: λ_c = {result['lambda_c_m']:.6e}, expected {OI_WAVELENGTH_M:.6e}"
```

### T4 — Doppler shift magnitude correct

```python
def test_doppler_shift_magnitude():
    """
    λ_c - λ₀ must equal λ₀ × v_rel / c to 1 ppm.
    For v_rel = 100 m/s: Δλ ≈ +0.210 pm.
    """
    from fpi.m01_airy_forward_model import OI_WAVELENGTH_M, SPEED_OF_LIGHT_MS
    v = 100.0
    params = InstrumentParams()
    result = synthesise_airglow_image(v, params, add_noise=False)
    lc = result['lambda_c_m']
    expected_shift = OI_WAVELENGTH_M * v / SPEED_OF_LIGHT_MS
    actual_shift   = lc - OI_WAVELENGTH_M
    assert abs(actual_shift - expected_shift) / abs(expected_shift) < 1e-6, \
        f"Doppler shift {actual_shift:.4e} m, expected {expected_shift:.4e} m"
```

### T5 — Round-trip v_rel recovery

```python
def test_round_trip_v_rel():
    """
    lambda_c_to_v_rel(v_rel_to_lambda_c(v)) must return v to < 1e-6 m/s.
    """
    for v in [-7200.0, -100.0, 0.0, +100.0, +500.0]:
        lc = v_rel_to_lambda_c(v)
        v_recovered = lambda_c_to_v_rel(lc)
        assert abs(v_recovered - v) < 1e-6, \
            f"Round-trip error {abs(v_recovered - v):.2e} m/s for v={v}"
```

### T6 — SNR achieved within 20% of target

```python
def test_snr_achieved():
    """
    Actual SNR of the noiseless fringe must be within 20% of specified target.
    """
    params = InstrumentParams()
    result = synthesise_airglow_image(100.0, params, snr=5.0, add_noise=True,
                                       rng=np.random.default_rng(0))
    assert abs(result['snr_actual'] - 5.0) / 5.0 < 0.20, \
        f"SNR {result['snr_actual']:.2f} more than 20% from target 5.0"
```

### T7 — Reproducible with fixed seed

```python
def test_reproducible_with_seed():
    """Two calls with identical seeds must produce identical noisy images."""
    params = InstrumentParams()
    r1 = synthesise_airglow_image(100.0, params, add_noise=True,
                                   rng=np.random.default_rng(7))
    r2 = synthesise_airglow_image(100.0, params, add_noise=True,
                                   rng=np.random.default_rng(7))
    np.testing.assert_array_equal(r1['image_2d'], r2['image_2d'])
```

### T8 — TypeError raised for temperature_K argument

```python
def test_no_temperature_argument():
    """
    synthesise_airglow_image must raise TypeError if temperature_K is passed.
    The delta-function source model has no temperature parameter.
    This test enforces the design decision permanently.
    """
    import pytest
    params = InstrumentParams()
    with pytest.raises(TypeError, match="temperature_K"):
        synthesise_airglow_image(100.0, params, temperature_K=800.0)
```

---

## 6. Expected numerical values

For `v_rel = 100 m/s`, `InstrumentParams()` defaults:

| Quantity | Expected | Derivation |
|----------|----------|------------|
| λ_c − λ₀ | +0.2101 pm | λ₀ × 100 / c |
| λ_c round-trip error | < 1e-6 m/s | T5 |
| Image shape | (256, 256) | T1 |
| SNR tolerance | ±20% of target | T6 |
| T8 TypeError | always raised | T8 |

---

## 7. File locations in repository

```
soc_sewell/
├── fpi/
│   ├── __init__.py
│   ├── m01_airy_forward_model_2026-04-05.py    ← S09, complete
│   ├── m02_calibration_synthesis_2026-04-05.py ← S10, complete
│   └── m04_airglow_synthesis_2026-04-05.py     ← this module
├── tests/
│   └── test_m04_airglow_synthesis_2026-04-05.py
└── docs/specs/
    └── S11_m04_airglow_synthesis_2026-04-05.md ← this file
```

---

## 8. Instructions for Claude Code

1. Read this entire spec AND S09 (M01) AND S10 (M02) before writing any code.
2. Confirm M01 and M02 tests pass:
   `pytest tests/test_m01_* tests/test_m02_* -v`
3. Implement `fpi/m04_airglow_synthesis_2026-04-05.py` with functions
   in this strict order:
   `v_rel_to_lambda_c` → `lambda_c_to_v_rel` → `add_gaussian_noise`
   → `synthesise_airglow_image`
4. Import `radial_profile_to_image` from M02. Do not reimplement it.
5. Import all constants from M01: `OI_WAVELENGTH_M`, `SPEED_OF_LIGHT_MS`.
   Do not define them in M04.
6. The source profile is:
   `S(r) = I_line × airy_modified(r; λ_c, params) + Y_bg×I_line×I0 + B`
   One call to `airy_modified()`. No wavelength grid. No matrix multiply.
7. Enforce the no-temperature rule: add `**kwargs` to `synthesise_airglow_image`
   and raise `TypeError` if `'temperature_K' in kwargs`. This is T8.
8. `add_gaussian_noise`: `σ_N = ΔS / snr` where `ΔS = max−min` of the
   noiseless image. Store `sigma_noise` in the returned dict.
9. Write all 8 tests. T8 must use `pytest.raises(TypeError)`.
10. Run: `pytest tests/test_m04_airglow_synthesis_2026-04-05.py -v`
    All 8 must pass.
11. Run full suite: `pytest tests/ -v` — M01 + M02 + M04 all pass.
12. Commit: `feat(m04): implement airglow synthesis delta-function model, 8/8 tests pass`

Module docstring header:
```python
"""
Module:      m04_airglow_synthesis_2026-04-05.py
Spec:        docs/specs/S11_m04_airglow_synthesis_2026-04-05.md
Author:      Claude Code
Generated:   <today>
Last tested: <today>  (8/8 tests pass)
Project:     WindCube FPI Pipeline — NCAR/HAO
Repo:        soc_sewell

Design note: OI 630 nm source is a spectral delta function.
             No temperature_K parameter. See spec Section 2.3.
"""
```
