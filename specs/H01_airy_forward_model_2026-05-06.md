# H01 — Airy Forward Model Specification

**Spec ID:** H01
**Spec file:** `docs/specs/H01_airy_forward_model_2026-05-06.md`
**Project:** WindCube FPI Pipeline
**Institution:** NCAR / High Altitude Observatory (HAO)
**Status:** Authoritative
**Depends on:** S01, S02, S03, S04
**Used by:** calibration synthesis, airglow synthesis, calibration inversion, airglow inversion modules
**References:**
  - Harding et al. (2014) Applied Optics 53(4) — Eqs. 1–11, instrument matrix
  - Vaughan (1989) *The Fabry-Perot Interferometer*, Ch. 3
  - Tolansky (1948) — fringe analysis, two-line method
  - GNL4096-R iss1 WindCube Etalon Assembly (ICOS build report, Dec 2023)
  - Teledyne e2v CCD97 datasheet — pixel pitch, array size
  - Burns, Adams & Longwell (1950) — Ne I spectroscopic standards

> **What changed from 2026-05-05:**
> 1. **`phase_correct_gap()` added as §8.** The Tolansky two-line analysis
>    returns a self-consistent pair (d, ε_a). Plugging d back into `2d/λ`
>    via floating-point arithmetic does not in general recover ε_a to the
>    required precision, because d has uncertainty ~200 nm ~ 0.6 FSR.
>    `phase_correct_gap()` nudges t by at most λ/4 (~160 nm) so that
>    `(2·t_eff/λ_a) % 1 == ε_a` exactly, anchoring the absolute fringe
>    position for synthesis. The function is validated (T16). The corrected
>    value `t_eff` is a synthesis convenience quantity only; `t_tolansky`
>    remains the authoritative physical gap for all other purposes.
> 2. **Constants section updated:** `ETALON_GAP_M` note clarified to
>    distinguish the ICOS spacer value (used only for N_Δ disambiguation)
>    from the Tolansky operational gap (used for fitting and synthesis).
> 3. **Default values table §5.2 updated** to reflect authoritative Tolansky
>    result: `t = 20.1070707e-3 m` (2026-05-05 run on
>    `1_cal_120sexp_swapped_ROI_L1.1`).
> 4. **§9 constants placement rule updated** to include `phase_correct_gap`
>    in the calibration synthesis import example.
> 5. **Task sequence §12 updated:** Task B module docstring updated; Task H
>    added for `phase_correct_gap` implementation and T16.

> **What changed from 2026-04-29:**
> 1. **`FOCAL_LENGTH_M` removed throughout.** The focal length is not a
>    parameter of the forward model. `alpha` (the plate scale, rad/px) fully
>    encapsulates the lens geometry for the Airy forward model.
> 2. **`alpha` and `t` (etalon gap) are now Fixed-from-Tolansky** in the neon
>    calibration parameter table (Section 4.2).
> 3. **Spec ID corrected** from H06/S06 to H01 throughout.
> 4. **File locations updated** to reflect new date suffix.

---

## 0. Relationship to H05 and H06 — why three specs remain

H01, H05, and H06 cover different parts of the pipeline and should remain
separate. Merging them would conflate two different concerns.

**H01 (this spec)** defines the *forward model*: how to compute a predicted
fringe profile from a given set of instrument and source parameters. It
specifies `InstrumentParams`, `build_instrument_matrix()`, `make_ne_spectrum()`,
`make_airglow_spectrum()`, and `phase_correct_gap()`. This is pure mathematics
and has no knowledge of optimisation strategy.

**H05** defines the *calibration inversion*: how to drive the H01 forward
model through a staged Levenberg-Marquardt optimisation to recover instrument
parameters from a neon fringe profile. It owns `FitConfig`, `CalibrationResult`,
`FitFlags`, and the 4-stage fitting sequence.

**H06** defines the *airglow inversion*: how to use the H01 forward model
(with instrument parameters fixed from H05) to recover the Doppler-shifted
line centre and hence the line-of-sight wind speed.

**Coverage of Harding (2014) equations across all three specs:**

| Harding eq. | Content | Spec |
|---|---|---|
| Eq. 1 | Fredholm integral: S(r) = ∫ A(r,λ) Y(λ) dλ | H01 §4.1 |
| Eq. 2 | Ideal Airy function A(r;λ) | H01 §2.1 |
| Eq. 3 | θ(r) = arctan(α·r) | H01 §2.1 |
| Eq. 4 | Intensity envelope I(r) | H01 §2.1 |
| Eq. 5 | PSF convolution: Ã = ∫ b(s,r)·A(s;λ)ds | H01 §2.2 |
| Eq. 6 | Gaussian PSF (constant width form) | H01 §2.2 |
| Eq. 7 | Shift-variant Gaussian PSF b(s,r) | H01 §2.2 |
| Eq. 8 | σ(r) = σ₀ + σ₁·sin + σ₂·cos | H01 §2.2 |
| Eq. 9 | Forward model with B: S = ∫Ã·Y dλ + B | H01 §4.1 |
| Eq. 10 | Airglow source spectrum Y(λ) (Gaussian) | H01 §4.3 (delta-fn limit) |
| Eq. 11 | Doppler shift: λ_c = λ₀(1 + v/c) | H01 §4.3 |
| Eq. 12 | Doppler broadening → temperature | *Deliberately excluded* — not a WindCube science product |
| Eq. 13 | χ² minimisation objective | H05 §9 |
| Eq. 14 | Discretised PSF convolution sum | H01 §4.4 |
| Eq. 15 | Discretised Fredholm integral (matrix row) | H01 §4.4 |
| Eq. 16 | Matrix equation: s = A·y + B | H01 §4.4 |

All Harding equations relevant to WindCube are covered between H01, H05, and
H06. H01 alone covers Eqs. 1–11 and 14–16. H05 owns the optimisation
procedure (Eq. 13). Eq. 12 is deliberately omitted.

---

## 1. Purpose

H01 is the mathematical core of the FPI instrument model. It computes the
ideal and PSF-broadened Airy transmission function for a given set of
instrument parameters, and constructs the instrument matrix **A** that maps
a source spectrum Y(λ) to a measured 1D fringe profile S(r):

```
S(r) = ∫ A(r, λ) Y(λ) dλ + B     (Harding Eq. 1, continuous form)
s    = A @ y + B                   (Harding Eq. 16, discrete matrix form)
```

H01 has no science-specific knowledge. It is wavelength-agnostic — it does
not know whether the source is neon, OI 630 nm airglow, or anything else.
That knowledge lives in the calling module's source spectrum vector `y`.
This separation is intentional and must be preserved.

**What H01 provides to the rest of the pipeline:**
- The `InstrumentParams` dataclass — the single shared container for all
  FPI hardware parameters, passed between pipeline modules.
- The `build_instrument_matrix()` function — the computational kernel of
  both the calibration inversion and airglow inversion modules.
- The `make_ne_spectrum()` function — two-line neon source vector for the calibration modules.
- The `make_airglow_spectrum()` function — Doppler-shifted OI source vector
  for the airglow modules.
- The `phase_correct_gap()` function — phase-anchors the etalon gap for
  faithful absolute fringe position synthesis (§8).
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

`α` (rad/px) is the plate scale: the angle subtended per binned pixel. It is
fully determined by the Tolansky two-line analysis and is not a free parameter
of the forward model. No focal length appears in the Airy function; `α`
encapsulates all relevant lens geometry.

`t` (the etalon gap) is similarly determined by the Tolansky two-line beat
period before any LM fitting occurs and is passed into the forward model
as a fixed (or tightly constrained) value.

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

This is the central computational object. The calibration and airglow inversion
modules both solve for y given s and A using least-squares inversion.

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

### 3.1 Constants required for H01

```python
from windcube.constants import (
    # Etalon / optics
    ETALON_GAP_M,           # 20.008e-3 m  — ICOS spacer measurement; used ONLY
                            #               to resolve N_Δ integer ambiguity in
                            #               the Tolansky analysis. NOT the
                            #               operational gap for fitting/synthesis.
                            #               Authoritative operational gap is the
                            #               Tolansky-recovered value (20.1070707e-3 m
                            #               as of 2026-05-05 run).
    ETALON_N,               # 1.0          — refractive index of etalon gap
    ETALON_R_INSTRUMENT,    # 0.53         — effective reflectivity (FlatSat)
    ALPHA_RAD_PX,           # 1.6071e-4 rad/px — Tolansky-recovered plate scale, 2×2 binned
    # CCD
    CCD_PIXELS_UNBINNED,    # 512          — physical pixels per side
    FOV_DEG,                # 1.65         — full field of view, degrees
    # OI airglow target line
    OI_WAVELENGTH_AIR_M,    # 630.0304e-9 m — NIST ASD air wavelength (rest)
    # Neon calibration lines (Burns et al. 1950 IAU standards, air wavelengths)
    NE_WAVELENGTH_1_AIR_M,  # 640.2248e-9 m — strong line
    NE_WAVELENGTH_2_AIR_M,  # 638.2991e-9 m — weak line
    NE_INTENSITY_1,         # 1.0           — reference intensity ratio
    NE_INTENSITY_2,         # 0.36          — weak/strong ratio
    # Physical constants
    SPEED_OF_LIGHT_MS,      # 299_792_458.0 m/s
)
```

**Note on the etalon gap:** `ETALON_GAP_M = 20.008e-3 m` is the ICOS spacer
mechanical measurement. It is used **only** to resolve the N_Δ integer
ambiguity in the Tolansky two-line analysis. The operational gap for all
forward model evaluations and synthesis is the Tolansky-recovered value,
which is passed into `InstrumentParams(t=...)` explicitly. The two values
differ by ~100 µm (~0.5 FSR) due to assembly compression and thermal
settling — this is normal and expected.

**Note:** `FOCAL_LENGTH_M` is NOT imported or used by H01. The plate scale
`alpha` (rad/px) is the fundamental geometric parameter for the Airy forward
model, recovered directly from the Tolansky analysis. Focal length and pixel
pitch are the physical origin of `alpha` but are not needed here.

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
- `NE_INTENSITY_2 = 0.36` (relative intensity — imported from constants.py)
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

**Instrumental parameters recovered from neon calibration fringes (M05):**

`t` (etalon gap) and `alpha` (plate scale) are determined by the Tolansky
two-line analysis **before** M05 runs. They enter M05 as seeds with
Tolansky-tightened bounds (see H05 §2.2). They are not free parameters
of the forward model; they characterise the instrument geometry and are
measured, not fitted.

The parameters that M05 actually fits from the neon fringe **shape** are:

| Parameter | Description | Fixed/Fitted in M05 | Default | Units |
|---|---|---|---|---|
| `t`      | Etalon gap | Fixed seed from Tolansky (fine-tuned with tight bounds) | 20.1070707e-3 | m |
| `alpha`  | Plate scale | Fixed seed from Tolansky (fine-tuned with tight bounds) | 1.6084e-4 | rad/px |
| `R_eff`  | Effective plate reflectivity | Fitted | 0.53 | — |
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

**Free parameters for OI airglow inversion (M06):**

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
a matrix equation. This is the computational core of H01:

```
s = A @ y + B·1          [Harding Eq. 16]
```

where:
- `s` — vector of length R; measured fringe profile in radial bins
- `A` — matrix of shape (R, L); instrument matrix
- `y` — vector of length L; discretised source spectrum
- `B` — scalar; CCD bias (uniform offset)
- `1` — length-R vector of ones

**Unit convention — Convention B (A absorbs Δλ):**

The `dλ` factor from the Fredholm integral is absorbed into the matrix:

```
A[i, j] = Ã(rᵢ; λⱼ) · Δλ
```

where `Δλ = lam_grid[1] - lam_grid[0]` is the uniform wavelength bin
width in metres.  With this convention:

- `y[j]` has units of **ADU per bin** (not ADU/m spectral density)
- For a delta-function source (neon line or OI airglow), the line bin
  holds `I_line` ADU and all other bins are zero — no division by Δλ needed
- The background term becomes `Y_bg` ADU per bin, uniform across all j
- `s = A @ y + B` has units of ADU with no trailing Δλ factor

This is the convention used by `build_instrument_matrix()`.  The
alternative (Convention A, A = pure Airy, y in ADU/m) is *not* used.

**I0 normalization:**

Each column of A is the PSF-broadened Airy profile `Ã(r; λⱼ)` scaled by
the intensity envelope `I(r)` from `params` (Harding Eq. 4).  The peak
value of column j is approximately `params.I0 · Δλ`.  The source vector
`y` is therefore in units of ADU per bin *relative to* `I0`; callers must
scale `I_line` and `Y_bg` consistently with the observed count level.

**Building column j of A (Harding Eqs. 14–15):**

Each column is the modified Airy profile Ã(r; λⱼ) multiplied by Δλ,
evaluated at all R radial bin centres:

```python
dlam = lam_grid[1] - lam_grid[0]
A[:, j] = airy_modified(r_bins, lambda_j, params) * dlam
```

The PSF convolution (Eq. 5) is discretised as a sum over all radial bins:

```python
A_tilde[i, j] = sum_k(b(r_k, r_i) * A_ideal[k, j] * delta_r)
```

For both neon and airglow forward evaluations, `s = A @ y + B` with y
as described above.

---

## 5. `InstrumentParams` dataclass

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

    Note: FOCAL_LENGTH_M is not a field of InstrumentParams. The plate scale
    alpha (rad/px) fully captures the lens geometry needed for the Airy
    forward model. alpha is set from ALPHA_RAD_PX (the Tolansky-recovered
    value) scaled by bin_factor; no focal length arithmetic is needed here.

    Note on t (etalon gap): the default t = ETALON_GAP_M = 20.008e-3 m is
    the ICOS spacer value, suitable as a prior only. For synthesis requiring
    correct absolute fringe positions, pass the Tolansky-recovered t and
    call phase_correct_gap() — see §8.
    """
    # --- Etalon ---
    R_eff:   float = field(default_factory=lambda: ETALON_R_INSTRUMENT)  # 0.53 — effective reflectivity
    t:       float = field(default_factory=lambda: ETALON_GAP_M)         # 20.008e-3 — ICOS prior; see note above
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
| `t`      | 20.008e-3 m | constants.py (ICOS spacer) | N_Δ disambiguation prior only; operational gap is Tolansky-recovered 20.1070707e-3 m (2026-05-05) |
| `n`      | 1.0 | constants.py | Air gap refractive index |
| `bin_factor` | 2 | Flight config | 2×2 on-chip binning |
| `alpha`  | 1.6084e-4 rad/px | constants.py (ALPHA_RAD_PX, Tolansky-recovered 2026-05-05) | Derived from bin_factor if not set; no focal length needed |
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

    Unit convention (Convention B):
      A[i, j] = Ã(rᵢ; λⱼ) · Δλ
    where Δλ = lam_grid[1] - lam_grid[0].  The Δλ factor is absorbed
    into A so that s = A @ y + B with y in ADU per bin (not ADU/m).
    See Section 4.4 for the full rationale.

    Parameters
    ----------
    r_bins      : (R,) array of radial bin centres, pixels
    lam_grid    : (L,) array of wavelength grid points, metres
                  Must be uniformly spaced.
    params      : InstrumentParams
    n_subpixels : sub-pixel oversampling within each radial bin
                  (8 recommended for synthetic data; 1 for real data)

    Returns
    -------
    A : (R, L) instrument matrix; A[i,j] = airy_modified(rᵢ; λⱼ) · Δλ
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

    Unit convention (Convention B):
      y[j] is in ADU per bin.  For a delta-function line, the single bin
      nearest the line wavelength holds I_line (scaled by the intensity
      ratio); all other bins are zero.  No division by Δλ is performed.
      This vector is intended for direct use as: s = A @ y_ne + B,
      where A is built by build_instrument_matrix() (which absorbs Δλ).

    Grid requirement:
      lam_grid must span both neon lines. The two lines are ~1.926 nm
      (~188 FSR) apart. Use make_wavelength_grid() centred at the midpoint
      (NE_WAVELENGTH_1_AIR_M + NE_WAVELENGTH_2_AIR_M) / 2 with n_fsr ≥ 200.
      A grid centred on either line alone with n_fsr=5 will NOT include
      the other line and will raise ValueError.

    Parameters
    ----------
    lam_grid : (L,) wavelength grid (metres) from make_wavelength_grid()
    I_line   : brightness of the strong line (ADU per bin); default 1.0

    Returns
    -------
    y_ne : (L,) source spectrum vector in ADU per bin

    Notes
    -----
    All wavelengths imported from windcube.constants:
      NE_WAVELENGTH_1_AIR_M = 640.2248e-9 m  (strong line)
      NE_WAVELENGTH_2_AIR_M = 638.2991e-9 m  (weak line)
      NE_INTENSITY_1 = 1.0
      NE_INTENSITY_2 = 0.36
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

    Unit convention (Convention B):
      y[j] is in ADU per bin.  The bin nearest λ_c holds Y_line ADU;
      all other bins hold Y_bg ADU (uniform background per bin).
      No division by Δλ is performed.  This vector is intended for
      direct use as: s = A @ y_oi + B.

    Parameters
    ----------
    lam_grid : (L,) wavelength grid (metres) from make_wavelength_grid()
    v_rel    : line-of-sight velocity (m/s); positive = recession
               Valid range: −7700 m/s to +1000 m/s
    Y_line   : line intensity (ADU per bin); default 1.0
    Y_bg     : spectrally flat background per wavelength bin (ADU per bin);
               default 0.0

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

Tests T1–T10 are unchanged from the 2026-04-13 revision. Tests T11–T15
were added in the 2026-04-29 revision and are unchanged here. T16 is new.

### T11 — make_ne_spectrum places both lines within grid

```python
def test_ne_spectrum_line_positions():
    """
    make_ne_spectrum must place nonzero power at bins closest to
    NE_WAVELENGTH_1_AIR_M and NE_WAVELENGTH_2_AIR_M, and zero elsewhere.

    Grid centering note: the two neon lines are ~188 FSR apart in order
    space (~1.926 nm in wavelength at 640 nm / 20.008 mm gap).  A grid
    centred on Ne1 with n_fsr=5 would only span ~51 pm — far too narrow
    to include Ne2.  We therefore centre the grid at the midpoint of the
    two lines and span n_fsr=200 FSRs (~2.05 nm), which comfortably
    brackets both lines with ~6 FSR of margin on each side.
    """
    from fpi.airy_forward_model import (
        make_ne_spectrum, make_wavelength_grid, InstrumentParams
    )
    from windcube.constants import (
        NE_WAVELENGTH_1_AIR_M, NE_WAVELENGTH_2_AIR_M,
        NE_INTENSITY_1, NE_INTENSITY_2
    )
    params = InstrumentParams()
    lam_mid = (NE_WAVELENGTH_1_AIR_M + NE_WAVELENGTH_2_AIR_M) / 2.0
    lam_grid = make_wavelength_grid(lam_mid, n_fsr=200,
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
    - Peak amplitude within 3× I0
    """
    from fpi.airy_forward_model import (
        make_ne_spectrum, make_wavelength_grid,
        build_instrument_matrix, InstrumentParams
    )
    from windcube.constants import NE_WAVELENGTH_1_AIR_M, NE_WAVELENGTH_2_AIR_M
    from scipy.signal import find_peaks

    params = InstrumentParams()
    r_bins = np.linspace(5, params.r_max, 200)

    # Centre grid at midpoint of the two Ne lines; span 200 FSR to include both
    lam_mid = (NE_WAVELENGTH_1_AIR_M + NE_WAVELENGTH_2_AIR_M) / 2.0
    lam_grid = make_wavelength_grid(lam_mid, n_fsr=200, L=501, params=params)

    y_ne = make_ne_spectrum(lam_grid, I_line=params.I0)
    A = build_instrument_matrix(r_bins, lam_grid, params)  # A absorbs dlam
    s = A @ y_ne + params.B

    assert np.all(s >= 0), "Fringe profile has negative values"
    peaks, _ = find_peaks(s, height=params.B + 0.01 * (s.max() - params.B))
    assert len(peaks) >= 2, f"Expected ≥2 fringe peaks, got {len(peaks)}"
    assert s.max() > params.B, "No fringe signal above bias"
```

### T16 — phase_correct_gap: corrected gap recovers ε_a exactly

```python
def test_phase_correct_gap():
    """
    phase_correct_gap(t_tolansky, eps_a, lam_a) must return t_eff such that:
      1. (2 * t_eff / lam_a) % 1 == eps_a  to within 1e-9 (floating-point exact)
      2. abs(t_eff - t_tolansky) < lam_a / 4  (correction < one quarter FSR in gap)
      3. ValueError raised for eps_a outside [0, 1)

    Validated against the 2026-05-05 Tolansky run on
    1_cal_120sexp_swapped_ROI_L1.1:
      t_tolansky = 20.1070707e-3 m
      eps_a      = 0.23286
      expected correction ≈ −96.1 nm  (t_eff = 20.1069746e-3 m)
    """
    from fpi.airy_forward_model import phase_correct_gap
    from windcube.constants import NE_WAVELENGTH_1_AIR_M

    t_tolansky = 20.1070707e-3
    eps_a      = 0.23286
    lam_a      = NE_WAVELENGTH_1_AIR_M

    t_eff = phase_correct_gap(t_tolansky, eps_a, lam_a)

    # Check 1: fractional part matches eps_a
    eps_recovered = (2.0 * t_eff / lam_a) % 1.0
    np.testing.assert_allclose(eps_recovered, eps_a, atol=1e-9,
                                err_msg="Recovered eps_a does not match input")

    # Check 2: correction is sub-quarter-FSR
    assert abs(t_eff - t_tolansky) < lam_a / 4.0, \
        f"Correction {(t_eff - t_tolansky)*1e9:.1f} nm exceeds lam/4"

    # Check 3: known numerical result from 2026-05-05 run
    np.testing.assert_allclose(t_eff * 1e3, 20.1069746, atol=0.0000001,
                                err_msg="t_eff does not match expected 2026-05-05 value")

    # Check 4: ValueError for out-of-range eps_a
    import pytest
    with pytest.raises(ValueError):
        phase_correct_gap(t_tolansky, -0.1, lam_a)
    with pytest.raises(ValueError):
        phase_correct_gap(t_tolansky, 1.0, lam_a)
```

---

## 8. `phase_correct_gap()` — absolute fringe phase anchor for synthesis

### 8.1 Motivation

The Tolansky two-line analysis (S13) returns a self-consistent pair
**(d, ε_a)** where:
- `d` — the recovered etalon gap in metres
- `ε_a` — the excess fraction for line a: the fractional interference order
  at the centre of the fringe pattern (r = 0) for λ_a

These two quantities together fully specify the absolute fringe position.
However, `d` has an absolute uncertainty of ±0.0002 mm (±200 nm ≈ 0.6 FSR).
When `d` alone is plugged back into `2d/λ_a` via floating-point arithmetic,
the recovered fractional part may differ from `ε_a` by up to 0.6 — enough
to shift the first fringe by many pixels and in extreme cases swap the
ordering of the two neon fringe families.

**The fundamental principle:** the Tolansky (d, ε_a) pair must always
travel together into synthesis. `d` alone is insufficient to set the
absolute fringe position.

### 8.2 Algorithm

Given Tolansky outputs `(t_tolansky, eps_a, lam_a)`:

1. Compute current fractional part: `eps_current = (2 * t_tolansky / lam_a) % 1`
2. Compute fractional error: `delta_eps = eps_a − eps_current`
3. Wrap into (−0.5, +0.5] to take the nearest-FSR correction:
   - if `delta_eps > 0.5`: `delta_eps -= 1.0`
   - if `delta_eps <= -0.5`: `delta_eps += 1.0`
4. Convert to gap correction: `delta_t = delta_eps * lam_a / 2`
5. Return `t_eff = t_tolansky + delta_t`

The correction |delta_t| is always < λ_a/4 ≈ 160 nm — well within the
Tolansky uncertainty. Fringe spacings are unchanged; only the absolute
phase anchor shifts.

### 8.3 Function signature

```python
def phase_correct_gap(
    t_tolansky: float,   # Tolansky-recovered gap, metres
    eps_a: float,        # Tolansky excess fraction for line a (0 <= eps_a < 1)
    lam_a: float,        # wavelength of line a, metres
) -> float:
    """
    Return a phase-corrected effective gap for fringe synthesis.

    The Tolansky analysis returns (t, eps_a) as a self-consistent pair, but
    floating-point evaluation of 2*t/lam_a does not in general recover eps_a
    because t has uncertainty ~200 nm ~ 0.6 FSR.  This function nudges t by
    at most lam_a/4 (~160 nm) so that (2*t_eff/lam_a) % 1 == eps_a exactly,
    anchoring the absolute fringe position for synthesis.

    The correction is purely a synthesis convenience — t_tolansky remains the
    authoritative physical gap for all other purposes (Tolansky priors for
    M05, FSR calculations, NetCDF output).

    Parameters
    ----------
    t_tolansky : Tolansky-recovered etalon gap, metres.
    eps_a      : Tolansky excess fraction for the anchor wavelength (line a).
                 Must satisfy 0 <= eps_a < 1.
    lam_a      : Anchor wavelength in metres. Use NE_WAVELENGTH_1_AIR_M for
                 neon calibration synthesis.

    Returns
    -------
    t_eff : float, phase-corrected gap in metres.
            Satisfies: abs(t_eff - t_tolansky) < lam_a / 4

    Raises
    ------
    ValueError : if eps_a is outside [0, 1)

    Example (2026-05-05 Tolansky run on 1_cal_120sexp_swapped_ROI_L1.1)
    -------------------------------------------------------------------
    t_tolansky = 20.1070707e-3 m
    eps_a      = 0.23286
    lam_a      = 640.2248e-9 m  (NE_WAVELENGTH_1_AIR_M)

    t_eff      = 20.1069746e-3 m   (correction = −96.1 nm)
    check:  (2 * t_eff / lam_a) % 1  →  0.23286  ✓
    """
```

### 8.4 Usage pattern

Call `phase_correct_gap()` once after each Tolansky run, before constructing
`InstrumentParams` for synthesis:

```python
from src.fpi.airy_forward_model_2026_05_06 import InstrumentParams, phase_correct_gap
from windcube.constants import NE_WAVELENGTH_1_AIR_M

# Values from the Tolansky run on this calibration image
t_tolansky = 20.1070707e-3   # m  — TolanskyResult.d_m
eps_a      = 0.23286          # —  — TolanskyResult.eps_a

t_eff = phase_correct_gap(t_tolansky, eps_a, NE_WAVELENGTH_1_AIR_M)

params = InstrumentParams(t=t_eff)   # t_eff used for synthesis only
```

**Do not** store `t_eff` as the authoritative gap in NetCDF output or
Tolansky result structures. `t_tolansky` is the physical measurement;
`t_eff` is a derived synthesis convenience quantity.

### 8.5 Why only λ_a is needed (not λ_b)

Once `t_eff` is anchored to ε_a for λ_a, the phase for λ_b is fully
determined by physics — there is only one physical gap. The relative
interleaving of the two neon fringe families follows automatically from
`2·t_eff/λ_b`, which will be consistent with ε_b to within the
Tolansky measurement precision.

---

## 9. Constants placement rule

All constants used in this module are imported from `windcube/constants.py`.
Import pattern for each downstream module:

```python
# In the calibration synthesis module (H02):
from fpi.airy_forward_model import (
    InstrumentParams, airy_modified, build_instrument_matrix,
    make_wavelength_grid, make_ne_spectrum, phase_correct_gap
)
# Ne wavelengths and intensities accessed via windcube.constants directly
# or retrieved from InstrumentParams which imports them internally.

# In the airglow synthesis module:
from fpi.airy_forward_model import (
    InstrumentParams, airy_modified, build_instrument_matrix,
    make_wavelength_grid, make_airglow_spectrum
)

# In the neon calibration inversion module (M05):
from fpi.airy_forward_model import (
    InstrumentParams, build_instrument_matrix, make_wavelength_grid,
    make_ne_spectrum
)

# In the airglow wind inversion module (M06):
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
H01 is implemented.

---

## 11. File locations in repository

```
soc_sewell/
├── windcube/
│   └── constants.py          ← verify constants from Section 3 are present
├── src/fpi/
│   ├── __init__.py            ← update import to new dated filename
│   └── airy_forward_model_2026_05_06.py   ← this module
├── tests/
│   └── test_airy_forward_model_2026-05-06.py
└── docs/specs/
    ├── H01_airy_forward_model_2026-05-06.md   ← this file
    └── archive/
        ├── H01_airy_forward_model_2026-05-05.md  ← retired
        └── H01_airy_forward_model_2026-04-29.md  ← retired
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
- `ALPHA_RAD_PX`
- `CCD_PIXELS_UNBINNED`
- `FOV_DEG`
- `OI_WAVELENGTH_AIR_M`
- `NE_WAVELENGTH_1_AIR_M`
- `NE_WAVELENGTH_2_AIR_M`
- `NE_INTENSITY_1`
- `NE_INTENSITY_2`
- `SPEED_OF_LIGHT_MS`

Note: `FOCAL_LENGTH_M` is intentionally absent from this list.

If any of the listed constants are missing, add them from the authoritative
values in Section 3 and commit:
`feat(constants): add missing constants for H01 v2026-05-06`

Run `pytest tests/ -v --tb=short`. All existing tests must pass.
If they do not, stop and report.

**TASK B — Create new module**

Create `src/fpi/airy_forward_model_2026_05_06.py` by copying
`src/fpi/airy_forward_model_2026_05_05.py` and adding the
`phase_correct_gap()` function from §8.3 of this spec. Place it
immediately after `airy_modified()` and before `build_instrument_matrix()`.

Module docstring:
```python
"""
Module:      airy_forward_model_2026_05_06.py
Spec:        docs/specs/H01_airy_forward_model_2026-05-06.md
Author:      Claude Code
Generated:   2026-05-06
Last tested: 2026-05-06
Project:     WindCube FPI Pipeline — NCAR/HAO
Repo:        soc_sewell

Changes from 2026_05_05:
  - phase_correct_gap() added (H01 §8). Anchors absolute fringe phase
    for synthesis using the Tolansky (t, eps_a) pair. The correction for
    the 2026-05-05 Tolansky run was -96.1 nm.
  - No other functional changes.
"""
```

**TASK C — Copy and update test file**

Create `tests/test_airy_forward_model_2026-05-06.py` by copying
`tests/test_airy_forward_model_2026-05-05.py` and:
1. Updating the import line to point to `airy_forward_model_2026_05_06`.
2. Adding T16 from §7 of this spec verbatim.

**TASK D — Run new tests**

Run: `pytest tests/test_airy_forward_model_2026-05-06.py -v --tb=short`

All 16 tests must pass. Stop and report if any fail.

**TASK E — Update __init__.py and archive old files**

1. Update `src/fpi/__init__.py` to re-export from the new dated module.
2. Move the old spec to archive:
   ```
   git mv docs/specs/H01_airy_forward_model_2026-05-05.md \
           docs/specs/archive/H01_airy_forward_model_2026-05-05.md
   ```
3. Copy this new spec to `docs/specs/H01_airy_forward_model_2026-05-06.md`.

**TASK F — Full test suite**

Run: `pytest tests/ -v --tb=short`

Report any failures. No regressions permitted.

**TASK G — Commit**

```
feat(H01): add phase_correct_gap() for absolute fringe phase anchor
Implements: H01_airy_forward_model_2026-05-06.md §8
Validated: correction = -96.1 nm for 2026-05-05 Tolansky run.
16/16 tests pass.
```

### Report format (paste back to Claude.ai)

```
TASK A — Constants check
  All present: Yes / No (list any missing)
  FOCAL_LENGTH_M present in constants.py but not imported: Yes / No
  Existing tests after constants update: N/N pass

TASK B — Module created
  Source file: src/fpi/airy_forward_model_2026_05_06.py
  phase_correct_gap() added: Yes / No
  Position in file: after airy_modified(), before build_instrument_matrix()
  Functional changes from 2026_05_05 other than phase_correct_gap: None / [list]

TASK C — Test file created
  Tests T1-T16: all present / [list missing]

TASK D — New tests
  Result: N/16 pass
  Failures: [list]

TASK E — Housekeeping
  __init__.py updated: Yes / No
  Old spec archived: Yes / No

TASK F — Full suite
  Result: N/N pass
  Unexpected failures: [list]

TASK G — Commit hash: [hash]
```
