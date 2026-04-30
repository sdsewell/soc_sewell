# S06 — Airy Forward Model Specification

**Spec ID:** S06
**Spec file:** `docs/specs/S06_airy_forward_model_2026-04-26.md`
**Project:** WindCube FPI Pipeline
**Institution:** NCAR / High Altitude Observatory (HAO)
**Status:** Authoritative
**Supersedes:** `docs/specs/archive/S06_airy_forward_model_2026-04-13.md`
**Depends on:** S01, S02, S03, S04
**Used by:** calibration synthesis, airglow synthesis, calibration inversion, airglow inversion modules
**References:**
  - Harding et al. (2014) Applied Optics 53(4) — Eqs. 1–11, instrument matrix
  - Vaughan (1989) *The Fabry-Perot Interferometer*, Ch. 3
  - Tolansky (1948) — fringe analysis, two-line method
  - GNL4096-R iss1 WindCube Etalon Assembly (ICOS build report, Dec 2023)
  - Teledyne e2v CCD97 datasheet — pixel pitch, array size
  - Burns, Adams & Longwell (1950) — Ne I spectroscopic standards
**Last updated:** 2026-04-26
**Created/Modified by:** Claude AI

> **What changed from 2026-04-13:**
> 1. **Section 4 (forward models) completely rewritten** to explicitly follow
>    Harding et al. (2014) Equations 1–11 for both the neon calibration
>    and OI airglow source spectra.
> 2. **Neon calibration source model** (Section 4.2): replaces the
>    prior single-laser-line model with a two-line neon spectrum
>    (λ₁ = 640.2248 nm strong, λ₂ = 638.2991 nm weak), both wavelengths
>    imported from `windcube/constants.py`.  No PSF-from-laser
>    bootstrapping — instrument function is solved directly from neon fringes.
> 3. **OI airglow source model** (Section 4.3): explicitly uses Harding
>    Eqs. 10–11 (Gaussian source spectrum + Doppler shift for wind).
>    Temperature retrieval via Eq. 12 is **explicitly excluded** — WindCube
>    uses a delta-function OI source (no thermal broadening).
> 4. **Velocity range** enforced: −7700 m/s to +1000 m/s in `make_airglow_spectrum()`.
>    Negative end is dominated by the spacecraft orbital velocity component
>    along the limb line of sight; positive end allows for modest recession.
> 5. All physical constants (wavelengths, speed of light, rest wavelength)
>    imported from `windcube/constants.py`; no hardcoded numerical values.
> 6. Section 5.8 `InstrumentParams` unchanged from 2026-04-13.
> 7. New tests T11–T15 added for forward model correctness.

---

## 1. Purpose

S06 is the mathematical core of the FPI instrument model. It computes the
ideal and PSF-broadened Airy transmission function for a given set of
instrument parameters, and constructs the instrument matrix **A** that maps
a source spectrum Y(λ) to a measured 1D fringe profile S(r):

```
S(r) = ∫ A(r, λ) Y(λ) dλ + B     (Harding Eq. 1, continuous form)
s    = A @ y + B                   (Harding Eq. 16, discrete matrix form)
```

S06 has no science-specific knowledge. It is wavelength-agnostic — it does
not know whether the source is neon, OI 630 nm airglow, or
anything else. That knowledge lives in the calling module's source spectrum
vector `y`. This separation is intentional and must be preserved.

**What S06 provides to the rest of the pipeline:**
- The `InstrumentParams` dataclass — the single shared container for all
  FPI hardware parameters, passed between pipeline modules.
- The `build_instrument_matrix()` function — the computational kernel of
  both the calibration inversion and airglow inversion modules.
- The `make_ne_spectrum()` function — two-line neon source vector for the calibration modules.
- The `make_airglow_spectrum()` function — Doppler-shifted OI source vector
  for the airglow modules.
- All physical constants for the FPI wavelengths and source lines, as the
  single source of truth imported by all downstream pipeline modules.

---

## 2. Physical background

### 2.1 The Airy transmission function

The FPI etalon transmits light with intensity governed by the Airy function.
For a point source at angle θ from the optical axis (Harding Eq. 2):

```
A(r; λ) = I(r) / [1 + F · sin²(π · OPD / λ)]

where:
  OPD   = 2 · n · t · cos(θ(r))     optical path difference
  θ(r)  = arctan(α · r)              angle from optical axis  [Harding Eq. 3]
  F     = 4·R_eff / (1 − R_eff)²      finesse coefficient
  I(r)  = I₀ · (1 + I₁·(r/r_max) + I₂·(r/r_max)²)   intensity envelope  [Harding Eq. 4]
```

Peak transmission occurs when OPD = mλ for integer m (interference order).
Peaks occur at radii where `2nt·cos(θ) = mλ`, giving concentric rings.

### 2.2 PSF broadening

Real optics broaden each Airy peak by a shift-variant Gaussian PSF whose
width σ(r) varies with radius (Harding Eqs. 6–8):

```
b(s, r) = (1 / √(2π·σ(r)²)) · exp(−(s−r)² / σ(r)²)     [Harding Eq. 7]

σ(r) = σ₀ + σ₁·sin(π·r/r_max) + σ₂·cos(π·r/r_max)       [Harding Eq. 8]
```

The modified Airy function Ã(r; λ) is the convolution of the ideal Airy
with this PSF (Harding Eq. 5):

```
Ã(r; λ) = ∫₀^r_max b(s, r) · A(s; λ) ds
```

Implemented as a local Gaussian smoothing of the ideal profile at each
radial position.

### 2.3 The instrument matrix

Discretising wavelength into L bins and radius into R bins, the forward
model becomes a matrix equation (Harding Eqs. 14–16). Column j of A is
the modified Airy profile Ã(r; λⱼ) evaluated at all R radial positions:

```
A[i, j] = Ã(rᵢ; λⱼ)     shape: (R, L)
s = A @ y + B
```

This is the central computational object. The calibration and airglow inversion modules both solve for y
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
r_max_detector   = (CCD_PIXELS_UNBINNED / bin_factor) / 2    [binned px]
r_max_field_stop = (FOV_DEG / 2) × (π/180) / alpha          [binned px]

r_max = min(r_max_detector, r_max_field_stop)
```

For all WindCube operating modes, the **field stop wins**: the 1.65° FOV
restricts the illuminated circle to a radius smaller than the physical
detector half-width.

Numerical check (2×2 binned, all S03 defaults):
```
r_max_detector   = (512 / 2) / 2   = 128 px
r_max_field_stop = (0.825° × π/180) / 1.6000e-4  ≈ 89.7 px  → ~90 px
r_max            = min(128, 90) = 90 px
```

---

## 3. Constants from `windcube/constants.py`

**All numerical values must be imported from `windcube/constants.py`.**
No module in the pipeline may hardcode a value that appears in that file.

### 3.1 Constants required for S06 (already present in constants.py)

```python
from windcube.constants import (
    # Etalon / optics
    ETALON_GAP_M,           # 20.008e-3 m  — ICOS build report (authoritative)
    ETALON_N,               # 1.0          — refractive index of etalon gap
    ETALON_R_INSTRUMENT,    # 0.53         — effective reflectivity (FlatSat)
    FOCAL_LENGTH_M,         # 0.200 m   — imaging lens f
    ALPHA_RAD_PX,           # 1.6000e-4 rad/px — 2×2 binned plate scale (32 µm / 200 mm)
    # CCD
    CCD_PIXELS_UNBINNED,    # 512          — physical pixels per side
    FOV_DEG,                # 1.65         — full field of view, degrees
    # OI airglow target line
    OI_WAVELENGTH_AIR_M,    # 630.0304e-9 m — NIST ASD air wavelength (rest)
    # Neon calibration lines (Burns et al. 1950 IAU standards, air wavelengths)
    NE_WAVELENGTH_1_AIR_M,  # 640.2248e-9 m — strong line
    NE_WAVELENGTH_2_AIR_M,  # 638.2991e-9 m — weak line
    NE_INTENSITY_1,         # 1.0           — reference intensity ratio
    NE_INTENSITY_2,         # 0.8           — weak/strong ratio
    # Physical constants
    SPEED_OF_LIGHT_MS,      # 299_792_458.0 m/s
)
```


### 3.2 Constants to add to `windcube/constants.py` (if not already present)

These were introduced in the 2026-04-13 revision and should already be
present. Verify before implementing:

```python
PIXEL_SIZE_M        = 16.0e-6    # m, unbinned pixel pitch (CCD97 datasheet)
CCD_PIXELS_UNBINNED = 512        # pixels per side, unbinned
FOV_DEG             = 1.65       # degrees, full field of view
```

---

## 4. Forward models following Harding et al. (2014)

This section precisely follows Harding et al. (2014) Applied Optics 53(4)
Equations 1–11. Equation numbers are cited inline.

### 4.1 The general forward model (Harding Eq. 1)

Every radial bin in a measured 1D fringe profile is governed by a
Fredholm integral of the first kind:

```
S(r) = ∫_{-∞}^{+∞} A(r, λ) · Y(λ) dλ          [Harding Eq. 1]
```

where:
- `r` — radial distance from fringe center (pixels, after annular reduction)
- `λ` — wavelength
- `S(r)` — measured pixel count (ADU) at radius r
- `A(r, λ)` — instrument function (ideal or PSF-broadened Airy)
- `Y(λ)` — source spectrum (neon lamp or OI airglow)

This integral is discretised into the instrument matrix equation (Section 4.4).

### 4.2 Neon calibration source model

**Context:** WindCube calibrates using a neon lamp rather than a laser.
The neon lamp emits two closely spaced lines in the 638–641 nm band that
both fall within the FPI's free spectral range. This is analogous to
Harding's laser calibration (which uses a delta function), but with
**two** lines instead of one.

Each neon line is treated as a monochromatic (delta-function) source.
The two-line source spectrum is:

```
Y_Ne(λ) = I_line · [NE_INTENSITY_1 · δ(λ − λ₁)
                   + NE_INTENSITY_2 · δ(λ − λ₂)]
```

where:
- `λ₁ = NE_WAVELENGTH_1_AIR_M` (640.2248 nm, strong line — imported from constants.py)
- `λ₂ = NE_WAVELENGTH_2_AIR_M` (638.2991 nm, weak line — imported from constants.py)
- `NE_INTENSITY_1 = 1.0` (reference — imported from constants.py)
- `NE_INTENSITY_2 = 0.8` (relative intensity — imported from constants.py)
- `I_line` — overall brightness scale (free parameter, units: ADU)

**The delta-function approximation is exact for neon:** the natural
linewidth of atomic neon transitions is negligible compared to the FPI
resolution. No thermal broadening is applied.

In the discrete forward model, each delta function contributes a
single-wavelength column of the instrument matrix. The neon fringe model
forward evaluation is thus:

```python
S_Ne(r) = I_line · (NE_INTENSITY_1 · Ã(r; λ₁)
                   + NE_INTENSITY_2 · Ã(r; λ₂)) + B
```

**Instrumental parameters recovered from neon calibration fringes:**

The neon fringe pattern is used to recover the full set of instrument
parameters listed in Table 1 (Section 5). This is the WindCube analogue
of Harding's laser calibration step. The recovered parameters are then
held fixed during OI airglow inversion.

| Parameter | Description | Fixed/Fitted | Default | Units |
|---|---|---|---|---|
| `R_eff` | Effective plate reflectivity (combines coating R, scatter, absorption) | Fitted | 0.53 | — |
| `t`      | Etalon gap | Fitted (fine) | 20.008e-3 | m |
| `alpha`  | Plate scale | Fitted | 1.6000e-4 | rad/px |
| `I0`     | Mean intensity | Fitted | 1000 | ADU |
| `I1`     | Linear intensity falloff | Fitted | −0.1 | — |
| `I2`     | Quadratic intensity falloff | Fitted | 0.005 | — |
| `sigma0` | Mean PSF width | Fitted | 0.8 | px |
| `sigma1` | Sinusoidal PSF variation | Fitted | 0.1 | px |
| `sigma2` | Cosinusoidal PSF variation | Fitted | −0.05 | px |
| `B`      | CCD bias (background) | Fitted | 300 | ADU |

**Key difference from Harding's laser calibration:**
Harding uses a frequency-stabilized HeNe laser at 632.8 nm (a single,
perfectly monochromatic line at a known, stable wavelength). WindCube uses
a neon lamp with two emission lines. The two lines provide additional
constraints that allow the etalon gap `t` to be recovered more robustly
via the Tolansky two-line beat period. The two-line nature
of the calibration source means the neon fringe pattern has a beat
envelope that must be accounted for in the forward model.

### 4.3 OI airglow source model (Harding Eqs. 10–11)

The OI 630.0 nm airglow emission is modelled as a monochromatic line
(delta function) Doppler-shifted by the line-of-sight wind velocity.
**Temperature (thermal Doppler broadening) is explicitly excluded.**

**Rationale for delta-function approximation:**
The WindCube FPI has an instrument finesse of ~4.9 (reflectivity R = 0.53),
giving an instrument linewidth much broader than the thermal Doppler width
of the OI line at thermospheric temperatures (~800 K). At this resolution,
the Gaussian thermal broadening produces a negligible change in fringe
shape compared to the Airy profile. Temperature is therefore not a science
product, and Harding Eq. 12 (Doppler broadening → temperature) is not
implemented. This is an explicit, deliberate design choice.

**OI source spectrum (Harding Eq. 10, delta-function limit):**

```
Y_OI(λ) = Y_bg + Y_line · δ(λ − λ_c)
```

where:
- `Y_bg`   — spectrally flat sky background (free parameter, ADU per wavelength bin)
- `Y_line` — integrated line intensity (free parameter, ADU)
- `λ_c`    — Doppler-shifted line center (free parameter, derived from v_rel)

**Doppler shift (Harding Eq. 11):**

```
λ_c = λ₀ · (1 + v_rel / c)          [Harding Eq. 11]
```

where:
- `λ₀ = OI_WAVELENGTH_AIR_M` — rest wavelength (630.0304 nm, imported from constants.py)
- `v_rel` — line-of-sight velocity (m/s); positive = recession (redshift)
- `c = SPEED_OF_LIGHT_MS` — speed of light (imported from constants.py)

**Velocity sign convention:**
Positive `v_rel` means the emitting gas is moving away from the spacecraft
(recession), which shifts `λ_c` to longer wavelengths and moves fringes
**inward** (to smaller radius). Negative `v_rel` means approach
(blueshift), shifting fringes outward.

**Velocity range:**

WindCube observes thermospheric winds at limb geometry. The line-of-sight
velocity has two components:
1. Thermospheric horizontal wind projected onto the line of sight: typically
   ±500 m/s during geomagnetic storms.
2. Spacecraft orbital velocity projected onto the line of sight: up to
   ~−7200 m/s (negative = approach, blueshift) depending on orbit geometry
   and look direction.

The combined range is:

```
v_rel_min = −7700 m/s    (maximum blueshift, spacecraft approaching + storm wind)
v_rel_max = +1000 m/s    (moderate recession, tail-wind geometry)
```

The implementation must enforce this range:
```python
assert -7700 <= v_rel <= 1000, f"v_rel={v_rel} m/s out of range"
```

In the forward model, `make_airglow_spectrum()` uses `v_rel` to compute
`λ_c` and places the line intensity at the wavelength bin nearest to `λ_c`.

**OI airglow fringe model:**

```python
S_OI(r) = Y_bg · sum(Ã(r; λⱼ)) · Δλ
          + Y_line · Ã(r; λ_c)
          + B
```

The `Y_bg` term integrates the flat background over all wavelength bins.
The `Y_line` term is a single-column lookup into the instrument matrix
at the shifted wavelength.

**Free parameters for OI airglow inversion:**

| Parameter | Description | Derived quantity | Units |
|---|---|---|---|
| `B`      | CCD bias | — | ADU |
| `Y_bg`   | Sky background | — | ADU / wavelength bin |
| `Y_line` | Line intensity | — | ADU |
| `v_rel`  | Line-of-sight velocity | wind (after subtracting orbital component) | m/s |

All instrument parameters (R_eff, t, alpha, I0, I1, I2, sigma0, sigma1,
sigma2) are **fixed at values recovered from the neon calibration** and
are not refitted during airglow inversion, following Harding's procedure.

### 4.4 Discrete forward model and instrument matrix (Harding Eqs. 14–16)

The continuous Fredholm integral (Harding Eq. 1) is discretised into
a matrix equation. This is the computational core of S06:

```
s = A @ y + B·1          [Harding Eq. 16]
```

where:
- `s` — vector of length R; measured fringe profile in radial bins
- `A` — matrix of shape (R, L); instrument matrix
- `y` — vector of length L; discretised source spectrum
- `B` — scalar; CCD bias (uniform offset)
- `1` — length-R vector of ones

**Building column j of A (Harding Eqs. 14–15):**

Each column is the modified Airy profile Ã(r; λⱼ) evaluated at all R
radial bin centres:

```python
A[:, j] = airy_modified(r_bins, lambda_j, params)
```

The PSF convolution (Eq. 5) is discretised as a sum over all radial bins:

```python
A_tilde[i, j] = sum_k(b(r_k, r_i) * A_ideal[k, j] * delta_r)
```

For the neon calibration forward evaluation, the delta-function source
means the forward model collapses to a direct matrix column lookup rather
than a full matrix multiply. For the airglow forward model, the background
term requires evaluating the full matrix.

---

## 5. `InstrumentParams` dataclass

Unchanged from the 2026-04-13 revision. Reproduced here for completeness.

### 5.1 Field definitions

```python
@dataclass
class InstrumentParams:
    """
    All adjustable instrument parameters for the WindCube FPI.

    Default values are authoritative starting points.  Any parameter may be
    overridden at construction time with values recovered from calibration.

    All wavelength / distance constants are imported from windcube.constants;
    no numerical literals appear in this class.
    """
    # --- Etalon ---
    R_eff:   float = field(default_factory=lambda: ETALON_R_INSTRUMENT)  # 0.53 — effective reflectivity
    t:       float = field(default_factory=lambda: ETALON_GAP_M)         # 20.008e-3 — from ICOS build report via constants.py
    n:       float = ETALON_N                                              # 1.0

    # --- Imaging lens / detector ---
    bin_factor: int   = 2          # 2 for 2×2 binning (flight), 1 for unbinned
    alpha:      float = None       # rad/px; derived in __post_init__ if None
    r_max:      float = None       # px; derived in __post_init__ if None

    # --- Intensity envelope (Harding Eq. 4) ---
    I0: float = 1000.0   # ADU; mean fringe intensity
    I1: float = -0.1     # linear falloff coefficient
    I2: float =  0.005   # quadratic falloff coefficient

    # --- PSF width (Harding Eq. 8) ---
    sigma0: float =  0.8    # px; mean PSF width
    sigma1: float =  0.1    # px; sin variation
    sigma2: float = -0.05   # px; cos variation

    # --- Background ---
    B: float = 300.0    # ADU; CCD bias / dark pedestal

    def __post_init__(self):
        """Derive alpha and r_max from bin_factor if not explicitly set."""
        from windcube.constants import ALPHA_RAD_PX, CCD_PIXELS_UNBINNED, FOV_DEG
        import math
        if self.alpha is None:
            # ALPHA_RAD_PX is the 2×2 binned Tolansky plate scale.
            # For other bin_factors: alpha = ALPHA_RAD_PX * (2 / bin_factor)
            # (smaller pixels → finer plate scale per pixel)
            self.alpha = ALPHA_RAD_PX * (2.0 / self.bin_factor)
        if self.r_max is None:
            r_det = (CCD_PIXELS_UNBINNED / self.bin_factor) / 2.0
            r_fov = (FOV_DEG / 2.0) * (math.pi / 180.0) / self.alpha
            self.r_max = min(r_det, r_fov)
```

### 5.2 Default values table

| Parameter | Default | Source | Notes |
|---|---|---|---|
| `R_eff` | 0.53 | constants.py (ETALON_R_INSTRUMENT) | Effective reflectivity combining coating, scatter, absorption |
| `t`      | 20.008e-3 m | constants.py (ICOS) | Authoritative ICOS value; fine-tuned by calibration fit |
| `n`      | 1.0 | constants.py | Air gap refractive index |
| `bin_factor` | 2 | Flight config | 2×2 on-chip binning |
| `alpha`  | 1.6000e-4 rad/px | constants.py (derived from FOCAL_LENGTH_M, CCD_PIXEL_2X2_UM) | Derived from bin_factor if not set |
| `r_max`  | ~90 px | Field-stop limit | Derived if not set; override with 110 for flight |
| `I0`     | 1000 ADU | Typical neon lamp | |
| `I1`     | −0.1 | Typical | Negative = intensity falls toward edge |
| `I2`     | 0.005 | Typical | |
| `sigma0` | 0.8 px | Typical (Harding Table 1) | Mean PSF blur |
| `sigma1` | 0.1 px | Typical | |
| `sigma2` | −0.05 px | Typical | |
| `B`      | 300 ADU | Typical dark pedestal | |

---

## 6. Function signatures

### 6.1 Existing functions (unchanged from 2026-04-13)

```python
def theta_from_r(r: np.ndarray, alpha: float) -> np.ndarray:
    """Angle from optical axis: θ(r) = arctan(α·r).  [Harding Eq. 3]"""

def opd(r: np.ndarray, params: InstrumentParams, lam: float) -> np.ndarray:
    """Optical path difference: OPD = 2·n·t·cos(θ(r))."""

def airy_ideal(r: np.ndarray, lam: float, params: InstrumentParams) -> np.ndarray:
    """Ideal Airy function at wavelength lam.  [Harding Eq. 2 + Eq. 4]"""

def psf_sigma(r: np.ndarray, r_max: float, sigma0: float,
              sigma1: float, sigma2: float) -> np.ndarray:
    """PSF width as function of radius.  [Harding Eq. 8]"""

def airy_modified(r: np.ndarray, lam: float,
                  params: InstrumentParams) -> np.ndarray:
    """PSF-broadened Airy function.  [Harding Eqs. 5–8]"""

def build_instrument_matrix(r_bins: np.ndarray, lam_grid: np.ndarray,
                             params: InstrumentParams,
                             n_subpixels: int = 1) -> np.ndarray:
    """
    Build the R×L instrument matrix A.  [Harding Eqs. 14–16]

    Parameters
    ----------
    r_bins      : (R,) array of radial bin centres, pixels
    lam_grid    : (L,) array of wavelength grid points, metres
    params      : InstrumentParams
    n_subpixels : sub-pixel oversampling within each radial bin
                  (8 recommended for synthetic data; 1 for real data)

    Returns
    -------
    A : (R, L) instrument matrix; each column is airy_modified at lam_grid[j]
    """

def make_wavelength_grid(lam_centre: float, n_fsr: float,
                         L: int, params: InstrumentParams) -> np.ndarray:
    """
    Construct a wavelength grid spanning n_fsr free spectral ranges
    centred on lam_centre.

    Parameters
    ----------
    lam_centre : centre wavelength (m)
    n_fsr      : number of FSRs to span (typically 5)
    L          : number of wavelength bins
    params     : InstrumentParams (uses t, n)

    Returns
    -------
    lam_grid : (L,) array of wavelength values, metres
    """
```

### 6.2 New functions added in this revision

```python
def make_ne_spectrum(lam_grid: np.ndarray, I_line: float = 1.0) -> np.ndarray:
    """
    Construct the two-line neon source spectrum vector.

    Places the two neon lines (NE_WAVELENGTH_1_AIR_M, NE_WAVELENGTH_2_AIR_M)
    at the nearest wavelength bins in lam_grid, weighted by their
    intensity ratios (NE_INTENSITY_1, NE_INTENSITY_2) from constants.py.

    This implements the delta-function neon source model (Section 4.2):
      Y_Ne(λ) = I_line · [NE_INTENSITY_1·δ(λ−λ₁) + NE_INTENSITY_2·δ(λ−λ₂)]

    Both wavelengths must fall within lam_grid; raises ValueError otherwise.

    Parameters
    ----------
    lam_grid : (L,) wavelength grid (metres) from make_wavelength_grid()
    I_line   : overall brightness scale (ADU); default 1.0

    Returns
    -------
    y_ne : (L,) source spectrum vector; units match I_line

    Notes
    -----
    All wavelengths imported from windcube.constants:
      NE_WAVELENGTH_1_AIR_M = 640.2248e-9 m  (strong line)
      NE_WAVELENGTH_2_AIR_M = 638.2991e-9 m  (weak line)
      NE_INTENSITY_1 = 1.0
      NE_INTENSITY_2 = 0.8
    """


def make_airglow_spectrum(lam_grid: np.ndarray,
                          v_rel: float,
                          Y_line: float = 1.0,
                          Y_bg: float = 0.0) -> np.ndarray:
    """
    Construct the OI 630.0 nm airglow source spectrum vector.

    Implements the delta-function Doppler-shifted source model (Section 4.3):
      Y_OI(λ) = Y_bg + Y_line · δ(λ − λ_c)
      λ_c = OI_WAVELENGTH_AIR_M · (1 + v_rel / SPEED_OF_LIGHT_MS)

    Temperature broadening (Harding Eq. 12) is explicitly NOT applied.
    WindCube uses the delta-function approximation throughout; temperature
    is not a science product.

    Parameters
    ----------
    lam_grid : (L,) wavelength grid (metres) from make_wavelength_grid()
    v_rel    : line-of-sight velocity (m/s); positive = recession
               Valid range: −7700 m/s to +1000 m/s
    Y_line   : line intensity (ADU); default 1.0
    Y_bg     : spectrally flat background per wavelength bin (ADU); default 0.0

    Returns
    -------
    y_oi : (L,) source spectrum vector

    Raises
    ------
    ValueError : if v_rel is outside [−7700, +1000] m/s
    ValueError : if λ_c falls outside lam_grid

    Notes
    -----
    Velocity sign convention:
      Positive v_rel → recession → λ_c > λ₀ → fringes shift inward (smaller r)
      Negative v_rel → approach  → λ_c < λ₀ → fringes shift outward (larger r)

    Rest wavelength imported from windcube.constants:
      OI_WAVELENGTH_AIR_M = 630.0304e-9 m
    Speed of light imported from windcube.constants:
      SPEED_OF_LIGHT_MS = 299_792_458 m/s
    """
```

---

## 7. Tests

Tests T1–T10 are unchanged from the 2026-04-13 revision. New tests T11–T15
are added here.

### T11 — make_ne_spectrum places both lines within grid

```python
def test_ne_spectrum_line_positions():
    """
    make_ne_spectrum must place nonzero power at bins closest to
    NE_WAVELENGTH_1_AIR_M and NE_WAVELENGTH_2_AIR_M, and zero elsewhere.
    """
    from fpi.airy_forward_model import (
        make_ne_spectrum, make_wavelength_grid, InstrumentParams
    )
    from windcube.constants import (
        NE_WAVELENGTH_1_AIR_M, NE_WAVELENGTH_2_AIR_M,
        NE_INTENSITY_1, NE_INTENSITY_2
    )
    params = InstrumentParams()
    lam_grid = make_wavelength_grid(NE_WAVELENGTH_1_AIR_M, n_fsr=220,
                                    L=501, params=params)
    y = make_ne_spectrum(lam_grid, I_line=1.0)

    # Exactly two nonzero bins
    nonzero = np.where(y > 0)[0]
    assert len(nonzero) == 2, f"Expected 2 nonzero bins, got {len(nonzero)}"

    # Verify the two bins bracket the correct wavelengths
    lam1_idx = np.argmin(np.abs(lam_grid - NE_WAVELENGTH_1_AIR_M))
    lam2_idx = np.argmin(np.abs(lam_grid - NE_WAVELENGTH_2_AIR_M))
    assert nonzero[0] in [lam1_idx, lam2_idx]
    assert nonzero[1] in [lam1_idx, lam2_idx]

    # Verify intensity ratio
    i1 = y[lam1_idx]
    i2 = y[lam2_idx]
    ratio = i2 / i1
    np.testing.assert_allclose(ratio, NE_INTENSITY_2 / NE_INTENSITY_1,
                                rtol=1e-6, err_msg="Ne line intensity ratio wrong")
```

### T12 — make_airglow_spectrum: zero velocity places line at rest wavelength

```python
def test_airglow_zero_velocity():
    """
    At v_rel=0, λ_c must equal OI_WAVELENGTH_AIR_M to within one bin width.
    """
    from fpi.airy_forward_model import (
        make_airglow_spectrum, make_wavelength_grid, InstrumentParams
    )
    from windcube.constants import OI_WAVELENGTH_AIR_M
    params = InstrumentParams()
    lam_grid = make_wavelength_grid(OI_WAVELENGTH_AIR_M, n_fsr=5,
                                    L=201, params=params)
    y = make_airglow_spectrum(lam_grid, v_rel=0.0, Y_line=1000.0, Y_bg=0.0)
    peak_idx = np.argmax(y)
    lam_peak = lam_grid[peak_idx]
    bin_width = lam_grid[1] - lam_grid[0]
    assert abs(lam_peak - OI_WAVELENGTH_AIR_M) <= bin_width, \
        f"Peak at {lam_peak*1e9:.4f} nm, expected {OI_WAVELENGTH_AIR_M*1e9:.4f} nm"
```

### T13 — make_airglow_spectrum: Doppler shift is correct direction and magnitude

```python
def test_airglow_doppler_shift():
    """
    A positive v_rel must shift λ_c to a longer wavelength (redshift).
    A negative v_rel must shift λ_c to a shorter wavelength (blueshift).
    Magnitude: Δλ = λ₀ · v_rel / c
    """
    from fpi.airy_forward_model import (
        make_airglow_spectrum, make_wavelength_grid, InstrumentParams
    )
    from windcube.constants import OI_WAVELENGTH_AIR_M, SPEED_OF_LIGHT_MS
    params = InstrumentParams()
    lam_grid = make_wavelength_grid(OI_WAVELENGTH_AIR_M, n_fsr=5,
                                    L=501, params=params)

    v_test = 500.0   # m/s recession
    expected_shift = OI_WAVELENGTH_AIR_M * v_test / SPEED_OF_LIGHT_MS

    y_pos = make_airglow_spectrum(lam_grid, v_rel=+v_test, Y_line=1.0)
    y_neg = make_airglow_spectrum(lam_grid, v_rel=-v_test, Y_line=1.0)

    lam_pos = lam_grid[np.argmax(y_pos)]
    lam_neg = lam_grid[np.argmax(y_neg)]

    assert lam_pos > OI_WAVELENGTH_AIR_M, "Positive v_rel should redshift"
    assert lam_neg < OI_WAVELENGTH_AIR_M, "Negative v_rel should blueshift"

    bin_width = lam_grid[1] - lam_grid[0]
    np.testing.assert_allclose(lam_pos - OI_WAVELENGTH_AIR_M,
                                expected_shift, atol=bin_width,
                                err_msg="Doppler shift magnitude wrong")
```

### T14 — make_airglow_spectrum: velocity range enforcement

```python
def test_airglow_velocity_bounds():
    """
    v_rel outside [−7700, +1000] m/s must raise ValueError.
    Boundary values must not raise.
    """
    from fpi.airy_forward_model import (
        make_airglow_spectrum, make_wavelength_grid, InstrumentParams
    )
    from windcube.constants import OI_WAVELENGTH_AIR_M
    params = InstrumentParams()
    lam_grid = make_wavelength_grid(OI_WAVELENGTH_AIR_M, n_fsr=30,
                                    L=501, params=params)

    import pytest
    with pytest.raises(ValueError):
        make_airglow_spectrum(lam_grid, v_rel=-8000.0)
    with pytest.raises(ValueError):
        make_airglow_spectrum(lam_grid, v_rel=+2000.0)

    # Boundary values must succeed
    make_airglow_spectrum(lam_grid, v_rel=-7700.0)
    make_airglow_spectrum(lam_grid, v_rel=+1000.0)
```

### T15 — full neon forward model round-trip

```python
def test_ne_forward_model_roundtrip():
    """
    A · y_ne + B must produce a plausible fringe profile:
    - Non-negative everywhere
    - Has at least 2 peaks (two neon lines produce interleaved ring families)
    - Peak amplitude within 50% of I0
    """
    from fpi.airy_forward_model import (
        make_ne_spectrum, make_wavelength_grid,
        build_instrument_matrix, InstrumentParams
    )
    from windcube.constants import NE_WAVELENGTH_1_AIR_M
    from scipy.signal import find_peaks

    params = InstrumentParams()
    r_bins = np.linspace(5, params.r_max, 200)
    lam_grid = make_wavelength_grid(NE_WAVELENGTH_1_AIR_M, n_fsr=220,
                                    L=501, params=params)
    y_ne = make_ne_spectrum(lam_grid, I_line=params.I0)
    A = build_instrument_matrix(r_bins, lam_grid, params)
    s = A @ y_ne + params.B

    assert np.all(s >= 0), "Fringe profile has negative values"
    peaks, _ = find_peaks(s, height=params.B + 0.1 * params.I0)
    assert len(peaks) >= 2, f"Expected ≥2 fringe peaks, got {len(peaks)}"
    assert np.max(s) < 3 * params.I0, "Peak intensity unreasonably large"
```

---

## 8. Expected numerical values

For `InstrumentParams()` defaults at λ = `OI_WAVELENGTH_AIR_M` = 630.0304 nm,
`bin_factor=2`:

| Quantity | Expected | Source | Test |
|---|---|---|---|
| alpha | 1.6000e-4 rad/px | constants.py (ALPHA_RAD_PX) | T9 |
| r_max | ~90 px | FOV field-stop limit | T10 |
| Finesse coefficient F | ~9.6 | 4 × 0.53 / 0.47² | T7 |
| Instrument finesse | ~4.9 | π√R_eff / (1−R_eff) | T7 |
| FSR at 630 nm | ~9.92 pm | (630.03e-9)² / (2 × 20.008e-3) | T7 |
| FSR at 640 nm | ~10.24 pm | (640.22e-9)² / (2 × 20.008e-3) | T7 |
| Fringes across r_max | 2–10 | geometry | T1 |
| Ne lines separation | ~188 FSR | Δλ / FSR_Ne | T11 |
| A matrix shape | (200, 101) | — | T4 |
| All A values | ≥ 0 | physics | T4 |
| alpha(1×1) / alpha(2×2) | 0.5 | binning geometry | T9 |
| r_max(1×1) / r_max(2×2) | ~2.0 | FOV / alpha scaling | T10 |
| Ne spectrum nonzero bins | exactly 2 | two-line model | T11 |
| OI peak at v=0 | within 1 bin of 630.0304 nm | Doppler | T12 |
| OI shift at +500 m/s | +1.05 pm redshift | Doppler | T13 |

---

## 9. Constants placement rule

All constants used in this module are imported from `windcube/constants.py`.
Import pattern for each downstream module:

```python
# In the calibration synthesis module:
from fpi.airy_forward_model import (
    InstrumentParams, airy_modified, build_instrument_matrix,
    make_wavelength_grid, make_ne_spectrum
)
# Ne wavelengths and intensities accessed via windcube.constants directly
# or retrieved from InstrumentParams which imports them internally.

# In the airglow synthesis module:
from fpi.airy_forward_model import (
    InstrumentParams, airy_modified, build_instrument_matrix,
    make_wavelength_grid, make_airglow_spectrum
)

# In the neon calibration inversion module:
from fpi.airy_forward_model import (
    InstrumentParams, build_instrument_matrix, make_wavelength_grid,
    make_ne_spectrum
)

# In the airglow wind inversion module:
from fpi.airy_forward_model import (
    InstrumentParams, build_instrument_matrix, make_wavelength_grid,
    make_airglow_spectrum
)
```

---

## 10. Dependencies

```
numpy  >= 1.24
scipy  >= 1.10   # gaussian_filter1d, find_peaks (tests only)
```

`windcube.constants` must export all constants listed in Section 3 before
S06 is implemented.

---

## 11. File locations in repository

```
soc_sewell/
├── windcube/
│   └── constants.py          ← verify 3 constants from Section 3.2 are present
├── src/fpi/
│   ├── __init__.py            ← update import to new dated filename
│   └── airy_forward_model_2026_04_26.py   ← this module
├── tests/
│   └── test_airy_forward_model_2026-04-26.py
└── docs/specs/
    ├── S06_airy_forward_model_2026-04-26.md   ← this file
    └── archive/
        └── S06_airy_forward_model_2026-04-13.md  ← retired
```

---

## 12. Instructions for Claude Code

### Preamble — read before touching any file

1. Read this entire spec.
2. Read `windcube/constants.py` in full.
3. Read the current `src/fpi/airy_forward_model_*.py` (latest dated version).
4. Read the current `tests/test_airy_forward_model_*.py` (latest dated version).

Report which dated files you found for steps 3 and 4 before proceeding.

### Task sequence

**TASK A — Verify constants**

Check that `windcube/constants.py` exports all of the following. Report
Yes/No for each:
- `ETALON_GAP_M`
- `ETALON_N`
- `ETALON_R_INSTRUMENT`
- `FOCAL_LENGTH_M`
- `ALPHA_RAD_PX`
- `CCD_PIXELS_UNBINNED`
- `FOV_DEG`
- `OI_WAVELENGTH_AIR_M`
- `NE_WAVELENGTH_1_AIR_M`
- `NE_WAVELENGTH_2_AIR_M`
- `NE_INTENSITY_1`
- `NE_INTENSITY_2`
- `SPEED_OF_LIGHT_MS`

If any are missing, add them from the authoritative values in Section 3 and
commit:
`feat(constants): add missing constants for S06 v2026-04-26`

Run `pytest tests/ -v --tb=short`. All existing tests must pass.
If they do not, stop and report.

**TASK B — Create new module**

Create `src/fpi/airy_forward_model_2026_04_26.py` by copying the
current implementation and applying these additions:

1. Update the module docstring (template below).
2. Add imports for the new constants from `windcube.constants`.
3. Add function `make_ne_spectrum()` exactly as specified in Section 6.2.
4. Add function `make_airglow_spectrum()` exactly as specified in Section 6.2.
5. All existing functions (`theta_from_r`, `opd`, `airy_ideal`, `psf_sigma`,
   `airy_modified`, `build_instrument_matrix`, `make_wavelength_grid`,
   `InstrumentParams`) are **unchanged** — copy verbatim.

Module docstring:
```python
"""
Module:      airy_forward_model_2026_04_26.py
Spec:        docs/specs/S06_airy_forward_model_2026-04-26.md
Author:      Claude Code
Generated:   2026-04-26
Last tested: 2026-04-26
Project:     WindCube FPI Pipeline — NCAR/HAO
Repo:        soc_sewell

Changes from 2026_04_13:
  - Added make_ne_spectrum(): two-line neon source vector (Sec 4.2)
  - Added make_airglow_spectrum(): Doppler-shifted OI delta-function
    source vector (Sec 4.3, Harding Eqs 10-11); temperature broadening
    (Eq 12) deliberately excluded.
  - Velocity range enforced: [-7700, +1000] m/s.
  - All wavelengths and physical constants imported from windcube.constants.
  - InstrumentParams and all forward model functions unchanged.
"""
```

**TASK C — Create new test file**

Create `tests/test_airy_forward_model_2026-04-26.py` by copying the
current test file and:
1. Update the imports to point to the new dated module.
2. Add tests T11–T15 from Section 7 verbatim.
3. Do not modify T1–T10.

**TASK D — Run tests**

Run: `pytest tests/test_airy_forward_model_2026-04-26.py -v --tb=short`

All 15 tests must pass. If any fail, debug and fix before proceeding.
Do not proceed to Task E if tests are not passing.
Stop and report if you cannot fix failures within 10 minutes.

**TASK E — Update __init__.py and archive old files**

1. Update `src/fpi/__init__.py` to re-export from the new dated module.
2. Archive the old spec:
   ```
   git mv docs/specs/archive/S06_airy_forward_model_2026-04-13.md \
           docs/specs/archive/S06_airy_forward_model_2026-04-13.md
   ```
3. Copy this new spec to `docs/specs/S06_airy_forward_model_2026-04-26.md`.

**TASK F — Full test suite**

Run: `pytest tests/ -v --tb=short`

Report any failures. Known acceptable failures: none expected.
If more than 3 tests fail in files other than the S06 test file, stop
and report the full failure list before attempting any fixes.

**TASK G — Commit**

```
feat(m01): add make_ne_spectrum and make_airglow_spectrum; Harding Eqs 1-11 explicit; 15/15 tests pass
```

### Report format (paste back to Claude.ai)

```
TASK A — Constants check
  All present: Yes / No (list any missing)
  Existing tests after constants update: N/N pass

TASK B — Module created
  Source file: src/fpi/airy_forward_model_2026_04_26.py
  make_ne_spectrum: implemented / NOT implemented
  make_airglow_spectrum: implemented / NOT implemented
  Deviations from spec: [list any]

TASK C — Test file created
  Tests T11-T15: all present / [list missing]

TASK D — New tests
  Result: N/15 pass
  Failures: [list]

TASK E — Housekeeping
  __init__.py updated: Yes / No
  Old spec archived: Yes / No

TASK F — Full suite
  Result: N/N pass
  Unexpected failures: [list]

TASK G — Commit hash: [hash]
```
