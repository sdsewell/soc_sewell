# S09 — M01 Airy Forward Model Specification

**Spec ID:** S09
**Spec file:** `docs/specs/S09_m01_airy_forward_model_2026-04-13.md`
**Project:** WindCube FPI Pipeline
**Institution:** NCAR / High Altitude Observatory (HAO)
**Status:** Authoritative
**Supersedes:** `docs/specs/archive/S09_m01_airy_forward_model_2026-04-05.md`
**Depends on:** S01, S02, S03, S04
**Used by:**
  - S10 (M02) — imports `InstrumentParams`, `airy_modified`, Ne constants
  - S11 (M04) — imports `InstrumentParams`, OI constants
  - S12 (M03) — uses `InstrumentParams` for r_max
  - S13 (M05) — imports `build_instrument_matrix`, `make_wavelength_grid`
  - S14 (M06) — imports `build_instrument_matrix`, `make_wavelength_grid`
  - S07b (NB03) — imports `ALPHA_RAD_PER_PX` via S03 (no direct M01 import)
**References:**
  - Harding et al. (2014) Applied Optics 53(4) — Eqs. 2–16, instrument matrix
  - Tolansky (1948) — fringe analysis, two-line method
  - GNL4096-R iss1 WindCube Etalon Assembly (ICOS build report, Dec 2023)
  - Teledyne e2v CCD97 datasheet — pixel pitch, array size
**Last updated:** 2026-04-13
**Created/Modified by:** Claude AI

> **What changed from 2026-04-05:**
> 1. `InstrumentParams` gains a `bin_factor` field (default 2 for 2×2 binning).
> 2. `alpha` and `r_max` become derived via `__post_init__`, computed from
>    `bin_factor`, the Tolansky anchor `ALPHA_RAD_PER_PX` (S03), and the
>    field-stop FOV constant `FOV_DEG` (S03, new).
> 3. Three new physical constants added to S03: `FOV_DEG`, `PIXEL_SIZE_M`,
>    `CCD_PIXELS_UNBINNED`.
> 4. Two new tests added (T9, T10) for binning self-consistency.
> 5. Expected numerical values table updated for field-stop-limited r_max.
>
> **Backward compatibility:** `alpha` and `r_max` remain plain dataclass fields
> (not properties). They default to `None` and are resolved in `__post_init__`.
> Callers that explicitly pass `alpha=...` or `r_max=...` continue to work
> unchanged. `InstrumentParams()` with no arguments now yields `r_max ≈ 90 px`
> (field-stop limited) instead of `128 px` (detector limited). Any test that
> implicitly relied on `r_max = 128` must be updated.

---

## 1. Purpose

M01 is the mathematical core of the FPI instrument model. It computes the
ideal and PSF-broadened Airy transmission function for a given set of
instrument parameters, and constructs the instrument matrix A that maps a
source spectrum Y(λ) to a measured 1D fringe profile S(r):

```
S(r) = ∫ A(r, λ) Y(λ) dλ + B     (Harding Eq. 1, continuous form)
s    = A @ y + B                   (Harding Eq. 16, discrete matrix form)
```

M01 has no science-specific knowledge. It is wavelength-agnostic — it does
not know whether the source is neon (M02), OI 630 nm airglow (M04), or
anything else. That knowledge lives in the calling module's source spectrum
vector `y`. This separation is intentional and must be preserved.

**What M01 provides to the rest of the pipeline:**
- The `InstrumentParams` dataclass — the single shared container for all
  FPI hardware parameters, passed between M01, M02, M04, M05, M06.
- The `build_instrument_matrix()` function — the computational kernel of
  both M05 (calibration inversion) and M06 (airglow inversion).
- All physical constants for the FPI wavelengths and source lines, as the
  single source of truth imported by M02, M04, M06, M07.

---

## 2. Physical background

### 2.1 The Airy transmission function

The FPI etalon transmits light with intensity governed by the Airy function.
For a point source at angle θ from the optical axis:

```
A(r; λ) = I(r) / [1 + F · sin²(π · OPD / λ)]

where:
  OPD   = 2 · n · t · cos(θ(r))     optical path difference
  θ(r)  = arctan(α · r)              angle from optical axis
  F     = 4R / (1 − R)²              finesse coefficient
  I(r)  = I₀ · (1 + I₁·(r/r_max) + I₂·(r/r_max)²)   intensity envelope
```

Peak transmission occurs when OPD = mλ for integer m (interference order).
Peaks occur at radii where `2nt·cos(θ) = mλ`, giving concentric rings.

### 2.2 PSF broadening

Real optics broaden each Airy peak by a shift-variant Gaussian PSF whose
width σ(r) varies with radius:

```
σ(r) = σ₀ + σ₁·sin(π·r/r_max) + σ₂·cos(π·r/r_max)   (Harding Eq. 5)
```

The modified Airy function Ã(r; λ) is the convolution of the ideal Airy
with this PSF, implemented as a local Gaussian smoothing of the ideal
profile at each radial position.

### 2.3 The instrument matrix

Discretising wavelength into L bins and radius into R bins, the forward
model becomes a matrix equation. Column j of A is the modified Airy
profile Ã(r; λⱼ) evaluated at all R radial positions:

```
A[i, j] = Ã(rᵢ; λⱼ)     shape: (R, L)
s = A @ y + B
```

This is the central computational object. M05 and M06 both solve for y
given s and A using least-squares inversion.

### 2.4 Anti-inverse-crime rule

The wavelength grid used for synthesis (L_synth = 300) must differ from
the grid used for inversion (L = 101). Using identical grids in both
forward and inverse steps produces artificially perfect recoveries that
do not reflect real measurement noise. This is enforced by the spec:
`make_wavelength_grid()` accepts L as an argument; callers are responsible
for passing the correct value.

### 2.5 Detector binning and r_max

The CCD97 detector is 512×512 physical pixels. WindCube operates in 2×2
on-chip binning, producing a 256×256 effective image. The `bin_factor`
field of `InstrumentParams` (default 2) controls which mode is assumed.

`r_max` — the maximum usable fringe radius — is set by whichever limit is
smaller: the physical detector edge or the field stop.

```
r_max_detector  = (CCD_PIXELS_UNBINNED / bin_factor) / 2    [binned px]
r_max_field_stop = (FOV_DEG / 2) × (π/180) / alpha          [binned px]

r_max = min(r_max_detector, r_max_field_stop)
```

For all WindCube operating modes, the **field stop wins**: the 1.65° FOV
restricts the illuminated circle to a radius smaller than the physical
detector half-width. The field stop is therefore the authoritative upper
bound on r_max.

Numerical check (2×2 binned, all S03 defaults):
```
r_max_detector   = (512 / 2) / 2   = 128 px
r_max_field_stop = (0.825° × π/180) / 1.6071e-4  ≈ 89.7 px  → rounds to ~90 px
r_max            = min(128, 90) = 90 px
```

For 1×1 unbinned mode:
```
alpha            = 1.6071e-4 / 2 = 8.035e-5 rad/px
r_max_detector   = 512 / 2 = 256 px
r_max_field_stop = (0.825° × π/180) / 8.035e-5 ≈ 179.4 px → ~179 px
r_max            = min(256, 179) = 179 px
```

The same physical circle on the sky maps to 90 binned px or 179 unbinned px
— a factor of ~2, as expected. This self-consistency confirms the geometry
is correct.

---

## 3. New and updated constants for S03 / `windcube/constants.py`

The following three constants must be added to `windcube/constants.py`
**before** re-implementing M01. They are used by `InstrumentParams.__post_init__`
and must be importable from S03.

```python
# ---------------------------------------------------------------------------
# CCD geometry and optical FOV  (S09 v2 addendum, 2026-04-13)
# ---------------------------------------------------------------------------

# CCD97 physical pixel pitch.  From Teledyne e2v CCD97 datasheet.
PIXEL_SIZE_M = 16.0e-6              # m, unbinned pixel pitch

# CCD97 array size, unbinned.
CCD_PIXELS_UNBINNED = 512           # pixels per side, unbinned (512×512)

# FPI field stop full angle.  Sets r_max via field-stop limit.
# This is the 1.65° full FOV; half-angle used in r_max calculation.
FOV_DEG = 1.65                      # degrees, full field of view
```

**Note:** `ALPHA_RAD_PER_PX = 1.6071e-4` (the 2×2 binned Tolansky plate
scale) was already added to S03 in the S07b/NB03 session. It serves as
the anchor from which all binning-mode plate scales are derived. No change
needed to that constant.

---

## 4. Physical constants

All values from S03. Listed here for implementation convenience.
**Do not hardcode these — import from `m01_airy_forward_model.py`.**

### 4.1 FPI wavelength constants (defined in M01, imported by M02/M04/M06/M07)

```python
# OI 630 nm science line
OI_WAVELENGTH_M    = 630.0304e-9    # m, OI air wavelength (S03)
SPEED_OF_LIGHT_MS  = 299_792_458.0  # m/s, exact (S03)
BOLTZMANN_J_PER_K  = 1.380649e-23   # J/K, exact (S03)
OXYGEN_MASS_KG     = 2.6567e-26     # kg, one O-16 atom (S03)

# Neon calibration lamp lines
NE_WAVELENGTH_1_M  = 640.2248e-9    # m, primary Ne line (S03)
NE_WAVELENGTH_2_M  = 638.2991e-9    # m, secondary Ne line (S03)
NE_INTENSITY_2     = 0.8            # relative intensity of secondary line
```

### 4.2 `InstrumentParams` defaults

| Parameter | Default | Source | Notes |
|---|---|---|---|
| `bin_factor` | `2` | CCD97 ops mode | 1 = unbinned 512×512; 2 = 2×2 binned 256×256 |
| `t` | `20.008e-3 m` | ICOS build report | Physical spacer measurement |
| `R_refl` | `0.53` | FlatSat calibration | Effective R, not coating spec |
| `n` | `1.0` | — | Air gap |
| `alpha` | `None → derived` | Tolansky + bin_factor | `ALPHA_RAD_PER_PX / 2 × bin_factor` |
| `r_max` | `None → derived` | FOV_DEG + alpha | `min(r_det, r_fov)`, field-stop limited |
| `I0` | `1000.0 counts` | Harding | Average intensity |
| `I1` | `−0.1` | Harding | Linear vignetting |
| `I2` | `0.005` | Harding | Quadratic vignetting |
| `sigma0` | `0.5 px` | Harding | Average PSF width |
| `sigma1` | `0.1 px` | Harding | Sine variation |
| `sigma2` | `−0.05 px` | Harding | Cosine variation |
| `B` | `300.0 counts` | Harding | CCD bias pedestal |

**Resolved defaults for bin_factor=2 (standard operating mode):**
- `alpha` → `1.6071e-4 rad/px`  (Tolansky result, unchanged from prior spec)
- `r_max` → `~90 px`  (field-stop limited; was 128 px detector-limited in prior spec)

**Resolved defaults for bin_factor=1 (unbinned mode):**
- `alpha` → `8.035e-5 rad/px`
- `r_max` → `~179 px`  (field-stop limited)

**Override behaviour:** If `alpha` or `r_max` is explicitly set by the
caller (e.g. `InstrumentParams(r_max=110.0)` for flight data), the
explicit value is used and the derived value is discarded. `__post_init__`
only fills in values that are `None`.

---

## 5. Function signatures

Implement functions in this exact order — each depends on the previous.

### 5.1 `theta_from_r`

```python
def theta_from_r(
    r:     np.ndarray,  # radial positions, pixels, shape (R,)
    alpha: float,       # magnification constant, rad/pixel
) -> np.ndarray:
    """
    Map pixel radius to angle with optical axis.

    θ(r) = arctan(α · r)    (Harding Eq. 3)

    Parameters
    ----------
    r     : radial positions in pixels, shape (R,) or scalar
    alpha : magnification constant, rad/pixel.
            WindCube 2×2 binned: 1.6071e-4 rad/px (from S03 / Tolansky)

    Returns
    -------
    theta : angle in radians, same shape as r
    """
    return np.arctan(alpha * r)
```

### 5.2 `intensity_envelope`

```python
def intensity_envelope(
    r:     np.ndarray,  # radial positions, pixels, shape (R,)
    r_max: float,       # maximum radius, pixels
    I0:    float,       # average intensity, counts
    I1:    float,       # linear falloff coefficient
    I2:    float,       # quadratic falloff coefficient
) -> np.ndarray:
    """
    Quadratic intensity envelope accounting for optical vignetting.

    I(r) = I₀ · (1 + I₁·(r/r_max) + I₂·(r/r_max)²)    (Harding Eq. 4)

    The envelope must be positive everywhere for physically valid inputs.
    Caller is responsible for choosing I1, I2 such that I(r) > 0 for
    all r in [0, r_max].

    Returns
    -------
    I : intensity in counts, same shape as r
    """
    rn = r / r_max
    return I0 * (1.0 + I1 * rn + I2 * rn**2)
```

### 5.3 `airy_ideal`

```python
def airy_ideal(
    r:          np.ndarray,  # radial positions, pixels, shape (R,)
    wavelength: float,       # wavelength, metres
    t:          float,       # etalon gap, metres
    R_refl:     float,       # plate reflectivity, dimensionless
    alpha:      float,       # magnification constant, rad/pixel
    n:          float,       # index of refraction (1.0 for air gap)
    r_max:      float,       # maximum radius, pixels
    I0:         float,       # average intensity, counts
    I1:         float,       # linear vignetting coefficient
    I2:         float,       # quadratic vignetting coefficient
) -> np.ndarray:
    """
    Ideal (unbroadened) Airy transmission function at a single wavelength.

    A(r; λ) = I(r) / [1 + F · sin²(π · 2nt·cos(θ(r)) / λ)]

    Uses exact cosine (not small-angle approximation).
    Finesse coefficient F = 4R/(1-R)² computed internally.

    Returns
    -------
    A : CCD counts, shape (R,), values in [I(r)/(1+F), I(r)]
    """
    theta  = theta_from_r(r, alpha)
    I_env  = intensity_envelope(r, r_max, I0, I1, I2)
    F      = 4.0 * R_refl / (1.0 - R_refl)**2
    OPD    = 2.0 * n * t * np.cos(theta)
    phase  = np.pi * OPD / wavelength
    return I_env / (1.0 + F * np.sin(phase)**2)
```

### 5.4 `psf_sigma`

```python
def psf_sigma(
    r:      np.ndarray,  # radial positions, pixels, shape (R,)
    r_max:  float,       # maximum radius, pixels
    sigma0: float,       # average PSF width, pixels
    sigma1: float,       # sine variation amplitude, pixels
    sigma2: float,       # cosine variation amplitude, pixels
) -> np.ndarray:
    """
    Shift-variant Gaussian PSF width as a function of radius.

    σ(r) = σ₀ + σ₁·sin(π·r/r_max) + σ₂·cos(π·r/r_max)    (Harding Eq. 5)

    Returns
    -------
    sigma : PSF width in pixels, same shape as r. Always > 0 for valid inputs.
    """
    return sigma0 + sigma1 * np.sin(np.pi * r / r_max) \
                  + sigma2 * np.cos(np.pi * r / r_max)
```

### 5.5 `airy_modified`

```python
def airy_modified(
    r:          np.ndarray,
    wavelength: float,
    t:          float,
    R_refl:     float,
    alpha:      float,
    n:          float,
    r_max:      float,
    I0:         float,
    I1:         float,
    I2:         float,
    sigma0:     float,
    sigma1:     float,
    sigma2:     float,
) -> np.ndarray:
    """
    PSF-broadened Airy function at a single wavelength.

    Applies a shift-variant Gaussian convolution to the ideal Airy function.
    Uses the mean-sigma approximation (accurate to < 1% for smooth profiles).
    Returns airy_ideal() exactly when sigma_mean < 1e-6 (enforced by T3).

    Returns
    -------
    A_mod : PSF-broadened CCD counts, shape (R,)
    """
    A_ideal    = airy_ideal(r, wavelength, t, R_refl, alpha, n,
                            r_max, I0, I1, I2)
    sigma      = psf_sigma(r, r_max, sigma0, sigma1, sigma2)
    sigma_mean = float(np.mean(sigma))
    if sigma_mean < 1e-6:
        return A_ideal
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(A_ideal, sigma=sigma_mean)
```

### 5.6 `build_instrument_matrix`

```python
def build_instrument_matrix(
    r:           np.ndarray,  # radial bin centres, pixels, shape (R,)
    wavelengths: np.ndarray,  # wavelength bin centres, metres, shape (L,)
    t:           float,
    R_refl:      float,
    alpha:       float,
    n:           float,
    r_max:       float,
    I0:          float,
    I1:          float,
    I2:          float,
    sigma0:      float,
    sigma1:      float,
    sigma2:      float,
) -> np.ndarray:
    """
    Build the instrument matrix A of shape (R, L).

    Column j is airy_modified(r; λⱼ). Forward model: s = A @ y + B.
    Use L=101 for inversion, L=300 for synthesis (anti-inverse-crime rule).

    Returns
    -------
    A : np.ndarray, shape (R, L). All values >= 0, no NaN/Inf.
    """
    R_bins = len(r)
    L_bins = len(wavelengths)
    A = np.zeros((R_bins, L_bins))
    for j, lam in enumerate(wavelengths):
        A[:, j] = airy_modified(r, lam, t, R_refl, alpha, n,
                                 r_max, I0, I1, I2, sigma0, sigma1, sigma2)
    return A
```

### 5.7 `make_wavelength_grid`

```python
def make_wavelength_grid(
    center_wavelength: float,
    params:            'InstrumentParams',
    n_fsr:             float = 3.0,
    L:                 int   = 101,
) -> np.ndarray:
    """
    Wavelength grid spanning ±(n_fsr/2) FSRs about center_wavelength.

    Use L=101 for inversion, L=300 for synthesis (anti-inverse-crime rule).

    Returns
    -------
    wavelengths : np.ndarray, shape (L,), metres, monotonically increasing
    """
    fsr     = params.free_spectral_range(center_wavelength)
    lam_min = center_wavelength - n_fsr * fsr / 2.0
    lam_max = center_wavelength + n_fsr * fsr / 2.0
    return np.linspace(lam_min, lam_max, L)
```

### 5.8 `InstrumentParams`

```python
from dataclasses import dataclass, field
import numpy as np
from windcube.constants import (
    ALPHA_RAD_PER_PX,      # 2×2 Tolansky anchor, 1.6071e-4 rad/px
    CCD_PIXELS_UNBINNED,   # 512
    FOV_DEG,               # 1.65 degrees
)

@dataclass
class InstrumentParams:
    """
    Container for all WindCube FPI instrument parameters.
    Passed between M01, M02, M04, M05, M06 to avoid long argument lists.

    bin_factor controls the CCD binning mode:
        bin_factor=2  →  2×2 binned, 256×256 image  (standard, default)
        bin_factor=1  →  unbinned,   512×512 image

    alpha and r_max are derived from bin_factor via __post_init__ unless
    explicitly overridden by the caller. Setting alpha=<value> or
    r_max=<value> at construction bypasses the derived calculation and
    uses the explicit value directly. This allows flight data processing
    (r_max=110) and test overrides to work unchanged.
    """
    # Binning mode — set first, used by __post_init__ to derive alpha/r_max
    bin_factor: int   = 2          # 1 = unbinned (512×512); 2 = 2×2 (256×256)

    # Etalon
    t:          float = 20.008e-3  # gap, metres (ICOS build report Dec 2023)
    R_refl:     float = 0.53       # effective reflectivity (FlatSat cal)
    n:          float = 1.0        # refractive index (air gap)

    # Plate scale — None means "derive from bin_factor in __post_init__"
    alpha:      float = None       # rad/pixel; derived if None

    # Intensity envelope
    I0:         float =  1000.0    # average intensity, counts
    I1:         float =    -0.1    # linear vignetting coefficient
    I2:         float =   0.005    # quadratic vignetting coefficient

    # PSF
    sigma0:     float =  0.5       # average PSF width, pixels
    sigma1:     float =  0.1       # sine variation, pixels
    sigma2:     float = -0.05      # cosine variation, pixels

    # CCD
    B:          float = 300.0      # bias pedestal, counts

    # Max usable radius — None means "derive from FOV and alpha in __post_init__"
    r_max:      float = None       # pixels; derived if None

    def __post_init__(self):
        """
        Derive alpha and r_max from bin_factor if not explicitly set.

        alpha derivation:
            alpha_1x1 = ALPHA_RAD_PER_PX / 2    (Tolansky 2×2 anchor ÷ 2)
            alpha      = alpha_1x1 × bin_factor

        r_max derivation (field-stop limited):
            r_max_det  = (CCD_PIXELS_UNBINNED / bin_factor) / 2
            r_max_fov  = (FOV_DEG/2 × π/180) / alpha
            r_max      = min(r_max_det, r_max_fov)

        For bin_factor=2: alpha=1.6071e-4, r_max≈90 px (field-stop wins)
        For bin_factor=1: alpha=8.035e-5,  r_max≈179 px (field-stop wins)
        """
        if self.alpha is None:
            alpha_1x1  = ALPHA_RAD_PER_PX / 2.0
            self.alpha = alpha_1x1 * self.bin_factor
        if self.r_max is None:
            r_max_det  = (CCD_PIXELS_UNBINNED / self.bin_factor) / 2.0
            r_max_fov  = np.radians(FOV_DEG / 2.0) / self.alpha
            self.r_max = min(r_max_det, r_max_fov)

    def finesse_coefficient(self) -> float:
        """F = 4R / (1-R)²"""
        return 4.0 * self.R_refl / (1.0 - self.R_refl)**2

    def finesse(self) -> float:
        """Instrument finesse = π√R / (1-R)"""
        return np.pi * np.sqrt(self.R_refl) / (1.0 - self.R_refl)

    def free_spectral_range(self, wavelength: float) -> float:
        """FSR = λ² / (2nt),  metres"""
        return wavelength**2 / (2.0 * self.n * self.t)
```

---

## 6. Verification tests

All 10 tests in `tests/test_m01_airy_forward_model_2026-04-13.py`.
T1–T8 are carried over from the 2026-04-05 spec with minor updates where
`r_max` assertions are affected. T9–T10 are new binning tests.

### T1 — Airy peak positions uniform in r²

```python
def test_airy_peak_positions():
    """
    Peaks occur when 2nt·cos(θ) = mλ.
    In r² space, peak spacing must be approximately uniform.
    Expect 2–10 peaks across r_max for WindCube defaults.
    (r_max ≈ 90 px field-stop limited; expect ~4–6 fringes.)
    """
    from scipy.signal import find_peaks
    params = InstrumentParams()
    r  = np.linspace(0, params.r_max, 1000)
    A  = airy_ideal(r, OI_WAVELENGTH_M, params.t, params.R_refl,
                    params.alpha, params.n, params.r_max,
                    params.I0, params.I1, params.I2)
    peaks, _ = find_peaks(A, height=0.3 * params.I0)
    assert 2 <= len(peaks) <= 10, \
        f"Found {len(peaks)} peaks, expected 2–10 for WindCube defaults"
    r2_peaks = r[peaks]**2
    spacings = np.diff(r2_peaks)
    cv = np.std(spacings) / np.mean(spacings)
    assert cv < 0.20, \
        f"Peak spacing in r² not uniform (CV={cv:.3f})"
```

### T2 — PSF broadening increases FWHM

```python
def test_psf_broadens_fwhm():
    """Modified Airy must have larger FWHM than ideal Airy."""
    from scipy.signal import find_peaks
    params = InstrumentParams()
    r = np.linspace(0, params.r_max, 1000)

    A_ideal = airy_ideal(r, OI_WAVELENGTH_M, params.t, params.R_refl,
                          params.alpha, params.n, params.r_max,
                          params.I0, params.I1, params.I2)
    A_mod   = airy_modified(r, OI_WAVELENGTH_M, params.t, params.R_refl,
                             params.alpha, params.n, params.r_max,
                             params.I0, params.I1, params.I2,
                             params.sigma0, params.sigma1, params.sigma2)

    def first_peak_fwhm(A_arr, r_arr):
        peaks, _ = find_peaks(A_arr)
        if len(peaks) == 0:
            return None
        pk   = peaks[0]
        half = A_arr[pk] / 2.0
        left  = np.where(A_arr[:pk]  < half)[0]
        right = np.where(A_arr[pk:]  < half)[0]
        if len(left) == 0 or len(right) == 0:
            return None
        return r_arr[pk + right[0]] - r_arr[left[-1]]

    fwhm_i = first_peak_fwhm(A_ideal, r)
    fwhm_m = first_peak_fwhm(A_mod,   r)
    assert fwhm_i is not None and fwhm_m is not None
    assert fwhm_m > fwhm_i, \
        f"PSF did not broaden fringe: ideal={fwhm_i:.4f}, mod={fwhm_m:.4f} px"
```

### T3 — Zero PSF returns ideal Airy exactly

```python
def test_zero_psf_is_identity():
    """With sigma0=sigma1=sigma2=0, airy_modified must equal airy_ideal."""
    params = InstrumentParams(sigma0=0.0, sigma1=0.0, sigma2=0.0)
    r   = np.linspace(0, params.r_max, 500)
    lam = OI_WAVELENGTH_M
    A_i = airy_ideal(r, lam, params.t, params.R_refl, params.alpha,
                     params.n, params.r_max, params.I0, params.I1, params.I2)
    A_m = airy_modified(r, lam, params.t, params.R_refl, params.alpha,
                         params.n, params.r_max, params.I0, params.I1, params.I2,
                         0.0, 0.0, 0.0)
    np.testing.assert_allclose(A_m, A_i, rtol=1e-4,
        err_msg="Zero PSF did not return ideal Airy")
```

### T4 — Instrument matrix shape and non-negativity

```python
def test_instrument_matrix_shape():
    """A must have shape (R, L), all values >= 0, no NaN/Inf."""
    params = InstrumentParams()
    R, L   = 200, 101
    r      = np.linspace(0, params.r_max, R)
    wl     = make_wavelength_grid(OI_WAVELENGTH_M, params, L=L)
    A = build_instrument_matrix(r, wl, params.t, params.R_refl,
                                 params.alpha, params.n, params.r_max,
                                 params.I0, params.I1, params.I2,
                                 params.sigma0, params.sigma1, params.sigma2)
    assert A.shape == (R, L), f"Expected ({R},{L}), got {A.shape}"
    assert np.all(A >= 0),    "Instrument matrix has negative values"
    assert np.all(np.isfinite(A)), "Instrument matrix has NaN or Inf"
```

### T5 — Matrix forward model matches direct evaluation

```python
def test_matrix_forward_model_consistency():
    """
    s = A @ y + B must match direct airy_modified evaluation
    for a monochromatic (delta-function) source.
    Anti-inverse-crime: use L_synth=300 for synthesis, L=101 for matrix.
    """
    params = InstrumentParams()
    R, L_mat, L_synth = 200, 101, 300
    r    = np.linspace(0, params.r_max, R)
    lam0 = OI_WAVELENGTH_M

    wl_mat = make_wavelength_grid(lam0, params, L=L_mat)
    A_mat  = build_instrument_matrix(r, wl_mat, params.t, params.R_refl,
                                      params.alpha, params.n, params.r_max,
                                      params.I0, params.I1, params.I2,
                                      params.sigma0, params.sigma1, params.sigma2)
    j0   = np.argmin(np.abs(wl_mat - lam0))
    dlam = wl_mat[1] - wl_mat[0]
    y    = np.zeros(L_mat)
    y[j0] = 1.0 / dlam

    s_mat    = A_mat @ y + params.B
    s_direct = airy_modified(r, lam0, params.t, params.R_refl,
                              params.alpha, params.n, params.r_max,
                              params.I0, params.I1, params.I2,
                              params.sigma0, params.sigma1, params.sigma2) + params.B

    np.testing.assert_allclose(s_mat, s_direct, rtol=0.05,
        err_msg="Matrix forward model disagrees with direct evaluation")
```

### T6 — Intensity envelope positivity

```python
def test_intensity_envelope_positive():
    """I(r) must be positive everywhere for default parameters."""
    params = InstrumentParams()
    r = np.linspace(0, params.r_max, 500)
    I = intensity_envelope(r, params.r_max, params.I0, params.I1, params.I2)
    assert np.all(I > 0), \
        f"Intensity envelope non-positive; min={np.min(I):.2f}"
    assert I[0] >= I[-1], \
        "Intensity should fall off toward edge for I1 < 0 (vignetting)"
```

### T7 — Finesse and FSR physically reasonable

```python
def test_instrument_derived_quantities():
    """
    For WindCube defaults (R_refl=0.53, t=20.008 mm):
      Finesse coefficient F ≈ 9.6   (range 6–15 acceptable)
      Instrument finesse   ≈ 4.9    (range 3–20 acceptable)
      FSR at 630 nm        ≈ 9.92 pm (range 8–12 pm)
    """
    params  = InstrumentParams()
    F_coeff = params.finesse_coefficient()
    finesse = params.finesse()
    fsr     = params.free_spectral_range(OI_WAVELENGTH_M)

    assert 5 < F_coeff < 20,    f"Finesse coefficient {F_coeff:.2f} outside [5, 20]"
    assert 2 < finesse  < 20,   f"Finesse {finesse:.2f} outside [2, 20]"
    assert 8e-12 < fsr < 12e-12, f"FSR {fsr:.3e} m outside [8, 12] pm"
```

### T8 — PSF sigma always positive

```python
def test_psf_sigma_positive():
    """PSF sigma must be positive everywhere for default parameters."""
    params = InstrumentParams()
    r      = np.linspace(0, params.r_max, 500)
    sigma  = psf_sigma(r, params.r_max, params.sigma0,
                        params.sigma1, params.sigma2)
    assert np.all(sigma > 0), \
        f"PSF sigma non-positive; min={np.min(sigma):.4f} px"

    sigma_flat = psf_sigma(r, params.r_max, 2.0, 0.0, 0.0)
    np.testing.assert_allclose(sigma_flat, 2.0, rtol=1e-10,
        err_msg="Constant PSF (sigma1=sigma2=0) not returning sigma0")
```

### T9 — bin_factor=1 gives double the plate scale

```python
def test_binning_alpha_scales():
    """
    alpha(bin_factor=1) must equal 2 × alpha(bin_factor=2) to < 1 ppm.
    Both must equal alpha_1x1 × bin_factor.
    """
    p2 = InstrumentParams(bin_factor=2)
    p1 = InstrumentParams(bin_factor=1)
    assert abs(p1.alpha / p2.alpha - 0.5) < 1e-6, \
        f"alpha ratio {p1.alpha/p2.alpha:.6f}, expected 0.5 " \
        f"(unbinned should be half the 2×2 value)"
    # Tolansky anchor check: 2×2 alpha must match S03 constant
    from windcube.constants import ALPHA_RAD_PER_PX
    assert abs(p2.alpha - ALPHA_RAD_PER_PX) < 1e-12, \
        f"bin_factor=2 alpha {p2.alpha:.6e} != ALPHA_RAD_PER_PX {ALPHA_RAD_PER_PX:.6e}"
```

### T10 — r_max is field-stop limited in all binning modes

```python
def test_r_max_field_stop_limited():
    """
    For both binning modes, r_max must be set by the field stop
    (not the detector edge). The field-stop-limited radius scales
    as 1/bin_factor, so r_max(bin_factor=1) ≈ 2 × r_max(bin_factor=2).
    Both r_max values must be less than the half-detector size.
    """
    p1 = InstrumentParams(bin_factor=1)
    p2 = InstrumentParams(bin_factor=2)

    # Field stop must win over detector in both modes
    r_det_1 = 512 / 2   # = 256 px unbinned half-detector
    r_det_2 = 256 / 2   # = 128 px binned half-detector
    assert p1.r_max < r_det_1, \
        f"r_max(1×1)={p1.r_max:.1f} >= detector half-width {r_det_1}"
    assert p2.r_max < r_det_2, \
        f"r_max(2×2)={p2.r_max:.1f} >= detector half-width {r_det_2}"

    # The two r_max values should scale approximately as 2:1
    ratio = p1.r_max / p2.r_max
    assert 1.8 < ratio < 2.2, \
        f"r_max ratio (1×1)/(2×2) = {ratio:.3f}, expected ~2.0"

    # Explicit override must bypass field-stop limit
    p_flight = InstrumentParams(r_max=110.0)
    assert p_flight.r_max == 110.0, \
        "Explicit r_max=110.0 override not respected"
```

---

## 7. Expected numerical values

For `InstrumentParams()` defaults at λ = `OI_WAVELENGTH_M` = 630.0304 nm,
`bin_factor=2`:

| Quantity | Expected | Source | Test |
|---|---|---|---|
| alpha | 1.6071e-4 rad/px | Tolansky / S03 | T9 |
| r_max | ~90 px | FOV field-stop limit | T10 |
| Finesse coefficient F | ~9.6 | 4 × 0.53 / 0.47² | T7 |
| Instrument finesse | ~4.9 | π√0.53 / 0.47 | T7 |
| FSR at 630 nm | ~9.92 pm | (630.03e-9)² / (2 × 20.008e-3) | T7 |
| Fringes across r_max | 2–10 | geometry (expect ~4–6) | T1 |
| Peak spacing uniformity | CV < 0.20 | r² linearity | T1 |
| FWHM broadening | > 0 (measurable) | PSF convolution | T2 |
| A matrix shape | (200, 101) | — | T4 |
| All A values | ≥ 0 | physics | T4 |
| Matrix vs direct | < 5% RMS | anti-crime | T5 |
| alpha(1×1) / alpha(2×2) | 0.5 | binning geometry | T9 |
| r_max(1×1) / r_max(2×2) | ~2.0 | FOV / alpha scaling | T10 |

> **Note on r_max change:** The prior spec (2026-04-05) used `r_max=128 px`
> as the default, which was the detector half-width for 2×2 binning.
> The correct field-stop-limited value is ~90 px. This affects the number
> of fringes visible in synthetic images — fewer rings at smaller r_max is
> physically correct.

---

## 8. Constants placement rule

All constants defined in this module are the single source of truth for
the entire FPI pipeline. Import them as follows:

```python
# In M02 (calibration synthesis):
from fpi.m01_airy_forward_model import (
    InstrumentParams, airy_modified, build_instrument_matrix,
    make_wavelength_grid, NE_WAVELENGTH_1_M, NE_WAVELENGTH_2_M, NE_INTENSITY_2
)

# In M04 (airglow synthesis):
from fpi.m01_airy_forward_model import (
    InstrumentParams, airy_modified, build_instrument_matrix,
    make_wavelength_grid, OI_WAVELENGTH_M, SPEED_OF_LIGHT_MS,
    BOLTZMANN_J_PER_K, OXYGEN_MASS_KG
)

# In M05/M06 (inversion):
from fpi.m01_airy_forward_model import (
    InstrumentParams, build_instrument_matrix, make_wavelength_grid,
    OI_WAVELENGTH_M, NE_WAVELENGTH_1_M, NE_WAVELENGTH_2_M
)
```

---

## 9. Dependencies

```
numpy  >= 1.24
scipy  >= 1.10   # gaussian_filter1d, find_peaks (tests only)
```

`windcube.constants` must export `ALPHA_RAD_PER_PX`, `CCD_PIXELS_UNBINNED`,
and `FOV_DEG` before M01 is implemented (see Section 3).

---

## 10. File locations in repository

```
soc_sewell/
├── windcube/
│   └── constants.py         ← add 3 new constants from Section 3
├── src/fpi/
│   ├── __init__.py           ← update import to new dated filename
│   └── m01_airy_forward_model_2026_04_13.py   ← this module
├── tests/
│   └── test_m01_airy_forward_model_2026-04-13.py
└── docs/specs/
    ├── S09_m01_airy_forward_model_2026-04-13.md   ← this file
    └── archive/
        └── S09_m01_airy_forward_model_2026-04-05.md  ← retired
```

---

## 11. Instructions for Claude Code

1. Read this entire spec AND `windcube/constants.py` before writing any code.
2. Add the 3 new constants from Section 3 to `windcube/constants.py`.
   Commit this first:
   `feat(constants): add FOV_DEG, PIXEL_SIZE_M, CCD_PIXELS_UNBINNED for S09 binning`
3. Run `pytest tests/ -v --tb=short` to confirm all existing tests pass
   after the constants update. If they do not, stop and report.
4. Create `src/fpi/m01_airy_forward_model_2026_04_13.py` by copying the
   current dated implementation and applying these changes:
   a. Add `from windcube.constants import ALPHA_RAD_PER_PX, CCD_PIXELS_UNBINNED, FOV_DEG`
      to the imports.
   b. Replace the `InstrumentParams` dataclass definition entirely with
      the new definition from Section 5.8, including `__post_init__`.
   c. All other functions (`theta_from_r` through `make_wavelength_grid`)
      are **unchanged** — copy them verbatim.
5. Update `src/fpi/__init__.py` to re-export from the new dated filename.
6. Move the old test file to a new file `tests/test_m01_airy_forward_model_2026-04-13.py`.
   Add T9 and T10 from Section 6. Update T1 comment to reflect ~4–6 fringes
   at the new r_max. Do not change T2–T8.
7. Move the old spec to `docs/specs/archive/`:
   `git mv docs/specs/S09_m01_airy_forward_model_2026-04-05.md \
           docs/specs/archive/S09_m01_airy_forward_model_2026-04-05.md`
8. Run: `pytest tests/test_m01_airy_forward_model_2026-04-13.py -v`
   All 10 must pass.
9. Run the full suite: `pytest tests/ -v --tb=short`
   **Known impact:** any downstream test that implicitly relied on
   `InstrumentParams().r_max == 128` will now get `r_max ≈ 90`. Identify
   and update those assertions — they are incorrect, not a regression.
   If more than 5 downstream tests fail for this reason, stop and report
   the list to Claude.ai before fixing them.
10. Commit:
    `feat(m01): add bin_factor to InstrumentParams; alpha and r_max now field-stop derived, 10/10 tests pass`

Module docstring header:
```python
"""
Module:      m01_airy_forward_model_2026_04_13.py
Spec:        docs/specs/S09_m01_airy_forward_model_2026-04-13.md
Author:      Claude Code
Generated:   <today>
Last tested: <today>  (10/10 tests pass)
Project:     WindCube FPI Pipeline — NCAR/HAO
Repo:        soc_sewell

Changes from 2026_04_05:
  - InstrumentParams gains bin_factor field (default 2).
  - alpha and r_max derived in __post_init__ from bin_factor,
    ALPHA_RAD_PER_PX (Tolansky anchor), and FOV_DEG (field stop).
  - Explicit alpha= or r_max= overrides still work as before.
"""
```
