# S09 — M01 Airy Forward Model Specification

**Spec ID:** S09
**Spec file:** `docs/specs/S09_m01_airy_forward_model_2026-04-05.md`
**Project:** WindCube FPI Pipeline
**Institution:** NCAR / High Altitude Observatory (HAO)
**Status:** Specification — ready for implementation in VS Code
**Depends on:** S01, S02, S03, S04
**Used by:**
  - S10 (M02) — imports `InstrumentParams`, `airy_modified`, Ne constants
  - S11 (M04) — imports `InstrumentParams`, OI constants
  - S12 (M03) — uses `InstrumentParams` for r_max
  - S13 (M05) — imports `build_instrument_matrix`, `make_wavelength_grid`
  - S14 (M06) — imports `build_instrument_matrix`, `make_wavelength_grid`
**References:**
  - Harding et al. (2014) Applied Optics 53(4) — Eqs. 2–16, instrument matrix
  - Tolansky (1948) — fringe analysis, two-line method
  - GNL4096-R iss1 WindCube Etalon Assembly (ICOS build report, Dec 2023)
**Last updated:** 2026-04-05

> **Note:** This spec supersedes `m01_airy_forward_model_spec.md` from the
> legacy repo. The `alpha` default has been corrected from `8.5e-5` to
> `1.6071e-4` rad/px, reflecting the Tolansky analysis result for the
> 2×2 binned WindCube CCD. All constants now reference S03 explicitly.
> The repo path is `soc_sewell`, not `windcube-pipeline`.

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

---

## 3. Physical constants

All values from S03. Listed here for implementation convenience.
**Do not hardcode these — import from `m01_airy_forward_model.py`.**

### 3.1 FPI wavelength constants (defined in M01, imported by M02/M04/M06/M07)

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

### 3.2 InstrumentParams defaults

| Parameter | Default | Source | Notes |
|-----------|---------|--------|-------|
| `t` | 20.008e-3 m | ICOS build report Dec 2023 | Physical spacer measurement |
| `R_refl` | 0.53 | FlatSat calibration | Effective R, not coating spec |
| `n` | 1.0 | — | Air gap |
| `alpha` | 1.6071e-4 rad/px | Tolansky analysis | **2×2 binned** — old spec (8.5e-5) is wrong |
| `I0` | 1000.0 counts | Harding | Average intensity |
| `I1` | −0.1 | Harding | Linear vignetting |
| `I2` | 0.005 | Harding | Quadratic vignetting |
| `sigma0` | 0.5 px | Harding | Average PSF width |
| `sigma1` | 0.1 px | Harding | Sine variation |
| `sigma2` | −0.05 px | Harding | Cosine variation |
| `B` | 300.0 counts | Harding | CCD bias pedestal |
| `r_max` | 128.0 px | CCD geometry | Half-width of 2×2 binned 256×256 image |

**Critical note on `alpha`:** The old spec value `8.5e-5` rad/px was for an
unbinned configuration or a different instrument. The Tolansky two-line
analysis on the WindCube FlatSat calibration image recovered
`α = 1.6071 × 10⁻⁴ rad/px` for the 2×2 binned CCD. This is the authoritative
value. M05's `FitConfig` uses `alpha_init = 1.6e-4` with bounds `(1.4e-4, 1.8e-4)`,
consistent with this.

**Derived quantities from defaults:**
- Finesse coefficient F = 4 × 0.53 / (0.47)² ≈ 9.6
- Instrument finesse = π√0.53 / 0.47 ≈ 4.9  (low due to effective R=0.53)
- FSR at 630 nm = (630e-9)² / (2 × 20.008e-3) ≈ 9.92 pm
- Expected fringes across r_max: 3–7

---

## 4. Function signatures

Implement functions in this exact order — each depends on the previous.

### 4.1 `theta_from_r`

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

### 4.2 `intensity_envelope`

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

### 4.3 `airy_ideal`

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

### 4.4 `psf_sigma`

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

    The PSF captures optical defects in the etalon and imaging lens that
    cause fringe broadening beyond the ideal Airy function. σ(r) must
    be positive everywhere for physically valid parameters.

    Returns
    -------
    sigma : PSF width in pixels, same shape as r. Always > 0 for valid inputs.
    """
    return sigma0 + sigma1 * np.sin(np.pi * r / r_max) \
                  + sigma2 * np.cos(np.pi * r / r_max)
```

### 4.5 `airy_modified`

```python
def airy_modified(
    r:          np.ndarray,  # radial positions, pixels, shape (R,)
    wavelength: float,       # wavelength, metres
    t:          float,       # etalon gap, metres
    R_refl:     float,       # plate reflectivity
    alpha:      float,       # magnification constant, rad/pixel
    n:          float,       # index of refraction
    r_max:      float,       # maximum radius, pixels
    I0:         float,       # average intensity
    I1:         float,       # linear vignetting
    I2:         float,       # quadratic vignetting
    sigma0:     float,       # average PSF width, pixels
    sigma1:     float,       # sine PSF variation
    sigma2:     float,       # cosine PSF variation
) -> np.ndarray:
    """
    PSF-broadened Airy function at a single wavelength.

    Applies a shift-variant Gaussian convolution to the ideal Airy
    function. At each radial position i, the ideal profile is smoothed
    by a Gaussian kernel of width σ(rᵢ).

    Implementation uses scipy.ndimage.gaussian_filter1d with the local
    sigma at each point. For efficiency, use the mean sigma across the
    profile as the filter width (mean-sigma approximation), which is
    accurate to < 1% for the smooth sigma profiles expected from WindCube.

    When sigma0 = sigma1 = sigma2 = 0, returns exactly airy_ideal().
    This is enforced by T3.

    Returns
    -------
    A_mod : PSF-broadened CCD counts, shape (R,)
    """
    A_ideal = airy_ideal(r, wavelength, t, R_refl, alpha, n,
                         r_max, I0, I1, I2)
    sigma   = psf_sigma(r, r_max, sigma0, sigma1, sigma2)
    sigma_mean = float(np.mean(sigma))
    if sigma_mean < 1e-6:
        return A_ideal
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(A_ideal, sigma=sigma_mean)
```

### 4.6 `build_instrument_matrix`

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

    Column j of A is airy_modified(r; λⱼ) for wavelength wavelengths[j].
    The forward model is:
        s = A @ y + B    (Harding Eq. 16)
    where y is the source spectrum (counts/m), B is the CCD bias vector.

    This is the most computationally expensive function in M01.
    It is called once per fit in M05 and M06. Cache the result
    when instrument parameters do not change between calls.

    Parameters
    ----------
    r           : radial bin centres in pixels, shape (R,)
    wavelengths : wavelength bin centres in metres, shape (L,)
                  Use L=101 for inversion, L=300 for synthesis.

    Returns
    -------
    A : np.ndarray, shape (R, L)
        All values >= 0. No NaN or Inf for valid inputs.
    """
    R_bins = len(r)
    L_bins = len(wavelengths)
    A = np.zeros((R_bins, L_bins))
    for j, lam in enumerate(wavelengths):
        A[:, j] = airy_modified(r, lam, t, R_refl, alpha, n,
                                 r_max, I0, I1, I2, sigma0, sigma1, sigma2)
    return A
```

### 4.7 `make_wavelength_grid`

```python
def make_wavelength_grid(
    center_wavelength: float,         # metres
    params:            'InstrumentParams',
    n_fsr:             float = 3.0,   # FSR spans to cover
    L:                 int   = 101,   # number of wavelength bins
) -> np.ndarray:
    """
    Construct the wavelength grid for the instrument matrix.

    Spans ±(n_fsr/2) free spectral ranges about center_wavelength.

    Parameters
    ----------
    center_wavelength : metres. Use OI_WAVELENGTH_M for airglow (M04/M06),
                        NE_WAVELENGTH_1_M for calibration (M02/M05).
    n_fsr : number of FSRs to span. Default 3.0 covers the full beat
            pattern between the two neon lines.
    L     : number of bins. Use 101 for inversion, 300 for synthesis
            (anti-inverse-crime rule — see Section 2.4).

    Returns
    -------
    wavelengths : np.ndarray, shape (L,), units metres, monotonically increasing
    """
    fsr     = params.free_spectral_range(center_wavelength)
    lam_min = center_wavelength - n_fsr * fsr / 2.0
    lam_max = center_wavelength + n_fsr * fsr / 2.0
    return np.linspace(lam_min, lam_max, L)
```

### 4.8 `InstrumentParams`

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class InstrumentParams:
    """
    Container for all WindCube FPI instrument parameters.
    Passed between M01, M02, M04, M05, M06 to avoid long argument lists.

    Defaults reflect the actual WindCube instrument as characterised by
    the Tolansky analysis and FlatSat calibration.

    IMPORTANT: alpha = 1.6071e-4 rad/px is the 2x2 binned value from the
    Tolansky two-line analysis. The Harding paper value (8.5e-5) is for
    a different instrument configuration and must NOT be used here.
    """
    # Etalon
    t:       float = 20.008e-3   # gap, metres (ICOS build report Dec 2023)
    R_refl:  float = 0.53        # effective reflectivity (FlatSat cal)
    n:       float = 1.0         # refractive index (air gap)
    alpha:   float = 1.6071e-4   # rad/pixel, 2×2 binned (Tolansky 2026)

    # Intensity envelope
    I0:  float =  1000.0   # average intensity, counts
    I1:  float =    -0.1   # linear vignetting coefficient
    I2:  float =   0.005   # quadratic vignetting coefficient

    # PSF
    sigma0: float =  0.5    # average PSF width, pixels
    sigma1: float =  0.1    # sine variation, pixels
    sigma2: float = -0.05   # cosine variation, pixels

    # CCD
    B:     float = 300.0    # bias pedestal, counts
    r_max: float = 128.0    # max usable radius, pixels (256px image / 2)

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

## 5. Verification tests

All 8 tests in `tests/test_m01_airy_forward_model_2026-04-05.py`.

### T1 — Airy peak positions uniform in r²

```python
def test_airy_peak_positions():
    """
    Peaks occur when 2nt·cos(θ) = mλ.
    In r² space, peak spacing must be approximately uniform.
    Expect 3–7 peaks across r_max for WindCube defaults.
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
        f"Peak spacing in r² not uniform (CV={cv:.3f}); Airy function may be wrong"
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
        pk = peaks[0]
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

    # Build matrix with inversion grid (L=101)
    wl_mat = make_wavelength_grid(lam0, params, L=L_mat)
    A_mat  = build_instrument_matrix(r, wl_mat, params.t, params.R_refl,
                                      params.alpha, params.n, params.r_max,
                                      params.I0, params.I1, params.I2,
                                      params.sigma0, params.sigma1, params.sigma2)

    # Monochromatic source at central wavelength
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
    """
    PSF sigma must be positive everywhere for default parameters.
    With sigma1=sigma2=0, sigma must be constant = sigma0.
    """
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

---

## 6. Expected numerical values

For `InstrumentParams()` defaults at λ = `OI_WAVELENGTH_M` = 630.0304 nm:

| Quantity | Expected | Source | Test |
|----------|----------|--------|------|
| Finesse coefficient F | ~9.6 | 4 × 0.53 / 0.47² | T7 |
| Instrument finesse | ~4.9 | π√0.53 / 0.47 | T7 |
| FSR at 630 nm | ~9.92 pm | (630.03e-9)² / (2 × 20.008e-3) | T7 |
| Fringes across r_max | 3–7 | geometry | T1 |
| Peak spacing uniformity | CV < 0.20 | r² linearity | T1 |
| FWHM broadening | > 0 (measurable) | PSF convolution | T2 |
| A matrix shape | (200, 101) | — | T4 |
| All A values | ≥ 0 | physics | T4 |
| Matrix vs direct | < 5% RMS | anti-crime | T5 |

---

## 7. Constants placement rule

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

Never import these constants from S03 directly in downstream modules —
M01 is the re-export point for all FPI-specific constants.

---

## 8. Dependencies

```
numpy  >= 1.24   # array operations
scipy  >= 1.10   # gaussian_filter1d (airy_modified), find_peaks (tests only)
```

No new dependencies beyond what Tier 1 already installed.

---

## 9. File locations in repository

```
soc_sewell/
├── fpi/
│   ├── __init__.py
│   └── m01_airy_forward_model_2026-04-05.py   ← this module
├── tests/
│   └── test_m01_airy_forward_model_2026-04-05.py
└── docs/specs/
    └── S09_m01_airy_forward_model_2026-04-05.md   ← this file
```

---

## 10. Instructions for Claude Code

1. Read this entire spec AND S03 (physical constants) before writing any code.
2. No new dependencies required — numpy and scipy already installed.
3. Create `fpi/__init__.py` (empty) if it does not already exist.
4. Implement `fpi/m01_airy_forward_model_2026-04-05.py` with functions
   in this strict order:
   `theta_from_r` → `intensity_envelope` → `airy_ideal` → `psf_sigma`
   → `airy_modified` → `build_instrument_matrix` → `make_wavelength_grid`
   → `InstrumentParams`
5. Define all eight constants from Section 3.1 at module level immediately
   after imports. These are imported by M02, M04, M05, M06 — they must
   exist at module level, not inside functions.
6. Use `alpha = 1.6071e-4` rad/px as the `InstrumentParams` default.
   **Do not use 8.5e-5** — that value is wrong for the 2×2 binned CCD.
7. For `airy_modified`: use `scipy.ndimage.gaussian_filter1d` with the
   mean sigma across the profile. This is the mean-sigma approximation
   (accurate to < 1% for smooth PSF profiles). Return `airy_ideal` directly
   when `sigma_mean < 1e-6` to make T3 pass exactly.
8. Write all 8 tests in `tests/test_m01_airy_forward_model_2026-04-05.py`.
   Each test must import `OI_WAVELENGTH_M` from the module — no hardcoded
   wavelength literals in tests.
9. Run: `pytest tests/test_m01_airy_forward_model_2026-04-05.py -v`
   All 8 must pass.
10. Run full suite: `pytest tests/ -v` — all existing tests still pass.
11. Commit:
    `feat(m01): implement Airy forward model, 8/8 tests pass`
12. Do not implement S10 (M02) until this commit is confirmed.

Module docstring header:
```python
"""
Module:      m01_airy_forward_model_2026-04-05.py
Spec:        docs/specs/S09_m01_airy_forward_model_2026-04-05.md
Author:      Claude Code
Generated:   <today>
Last tested: <today>  (8/8 tests pass)
Project:     WindCube FPI Pipeline — NCAR/HAO
Repo:        soc_sewell
"""
```
