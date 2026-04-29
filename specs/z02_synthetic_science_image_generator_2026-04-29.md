# Z02 — Synthetic Science Image Generator

**Spec ID:** Z02  
**Tier:** 9 — Validation Testing  
**Module:** `z02_synthetic_science_image_generator.py`  
**Status:** Initial spec — awaiting implementation  
**Date:** 2026-04-29  
**Version:** 1.0  
**Author:** Scott Sewell / HAO  
**Repo:** `soc_sewell`  
**Dependencies:** S03 (physical constants), M01 (airy_ideal), M02 (radial_profile_to_image), P01 (metadata schema)

---

## Revision History

| Date | Version | Change |
|------|---------|--------|
| **2026-04-29** | **1.0** | **Initial spec. Derived from Z03 v1.5 by replacing two-line neon source with single Doppler-shifted OI 630.2046 nm line. Adds v_los prompt (default −7500 m/s). Removes rel_638. All other physics, geometry, and infrastructure identical to Z03 v1.5.** |

---

## 1. Purpose

Z02 creates a **synthetic 'truth' science image pair** — an OI 630.2046 nm airglow fringe image
and a companion dark image — both in authentic WindCube `.bin` format (1-row header + pixel
data), suitable for direct ingestion by any downstream pipeline science module.

The script is **interactive**: it prompts the user for all instrument and source parameters,
including the line-of-sight velocity `v_los` that Doppler-shifts the 630.2046 nm emission line.
Synthesis uses the ideal Airy transmission function with a single effective reflectivity R.

Both output images constitute **ground-truth artefacts** — every parameter used in synthesis
is known exactly and recorded in a companion `_truth.json` sidecar. The sidecar is designed
to be compared parameter-for-parameter with the wind retrieval output from the science
inversion pipeline.

### Z02 vs Z03 — what changes and what does not

Z02 and Z03 are deliberately parallel in design. The **only physics difference** is the
source model:

| Aspect | Z03 (calibration) | Z02 (science) |
|--------|-------------------|---------------|
| Source | Two-line neon lamp: 640.2248 nm + 638.2991 nm | Single OI airglow line: 630.2046 nm Doppler-shifted |
| Free source params | `rel_638` = 0.344 (line ratio) | `v_los` in m/s (Doppler shift) |
| Composite profile | `A(λ₁) + rel_638 · A(λ₂) + offset` | `A(λ_obs) + offset` |
| I₀ derivation | `I_peak / (1 + rel_638)` | `I_peak` directly (single line) |
| Output file type label | `_cal_synth_z03_` | `_sci_synth_z02_` |
| Truth JSON key | `z03_version` | `z02_version` |

All other aspects — image geometry, BinningConfig, etalon parameters, noise model, dark
image, header format, file I/O, test structure — are **identical** to Z03 v1.5.

---

## 2. Relationship to other pipeline modules

| Module | Role relative to Z02 |
|--------|--------------------|
| Z03 | Parallel calibration image generator; shared physics infrastructure |
| G01 | Bulk campaign generator; produces science images at scale using same Airy physics |
| M06 | Science fringe inversion; consumes real or Z02-synthetic science `.bin` files |
| Z01 | Reads `.bin` files; produces radial profiles for downstream inversion |

Z02 and G01 `_generate_science_pixels()` use **identical Airy physics and identical
instrument constants** (§3 below). Any discrepancy between the two is a bug.

---

## 3. Authoritative Instrument Constants

Identical to Z03 v1.5 §3 and G01 v10, with the neon-specific constants replaced by OI constants.

| Constant | Symbol | Value | Source |
|----------|--------|-------|--------|
| OI rest wavelength | `λ₀` | **630.2046e-9 m** | OI ¹S₀ → ¹D₂ transition; vacuum wavelength (Edlén 1966) |
| Speed of light | `c` | **2.99792458e8 m/s** | CODATA |
| Etalon gap | `d` | **20.0005 mm** | Benoit excess-fraction two-line recovery |
| Plate scale (2×2 binned) | `α` | **1.6000e-4 rad/px** | Tolansky two-line WLS |
| Plate scale (1×1 unbinned) | `α` | **0.8000e-4 rad/px** | = 1.6000e-4 / 2 |
| Effective reflectivity | `R` | **0.725** | Gives finesse N_R = 10.0; default (prompted) |
| Electronic offset | `offset` | **5 ADU** | Bias + read noise combined |
| Dark reference rate | `dark_ref` | **0.05 ADU/px/s** | At T_ref = −20°C |
| Dark doubling interval | `T_double` | **6.5°C** | Standard CCD rule |
| Focal plane temperature | `T_fp` | **−20°C** | Default (prompted) |

> **OI 630.2046 nm rest wavelength note:** The vacuum wavelength 630.2046 nm is used (converted from air wavelength 630.0304 nm via Edlén 1966; all three standard formulae agree to < 0.001 nm). For a
> source at thermospheric altitudes with v_los = −7500 m/s (blueshift dominated by spacecraft
> velocity), the observed wavelength is:
> ```
> λ_obs = 630.2046 × (1 + (−7500) / 2.99792458e8) = 630.1888 nm
> ```
> This shifts the phase by δφ = 4πd(1/λ_obs − 1/λ₀) ≈ −0.1 rad, moving fringes outward
> by approximately 0.25 pixels at r = 100 px (2×2 binned). The wind science signal
> (±hundreds of m/s) rides on top of the large spacecraft velocity.

---

## 4. User-Prompted Parameters

The script prompts sequentially in four groups. Groups 0–2 are identical to Z03 v1.5.
Group 3 is identical to Z03 v1.5 except the SNR prompt label now refers to the OI line.
Group 4 replaces the neon line-ratio prompt with a line-of-sight velocity prompt.

### Group 0 — Image Geometry (identical to Z03 v1.5)

| Prompt | Variable | Default | Units |
|--------|----------|---------|-------|
| Detector binning | `binning` | `2` | — |
| Fringe centre column | `cx` | *(mode-dependent)* | px |
| Fringe centre row | `cy` | *(mode-dependent)* | px |

Mode-dependent defaults, hard limits, and warning ranges: identical to Z03 v1.5 §4 / §11.

### Group 1 — Etalon Geometry (identical to Z03 v1.5)

| Prompt | Variable | Default | Units | Notes |
|--------|----------|---------|-------|-------|
| Etalon gap `d` | `d_mm` | **20.0005** | mm | Benoit two-line recovery |
| Plate scale `α` | `alpha` | *(mode-dependent)* | rad/px | 1.6000e-4 binned; 0.8000e-4 unbinned |

### Group 2 — Etalon Reflectivity (identical to Z03 v1.5)

| Prompt | Variable | Default | Units | Notes |
|--------|----------|---------|-------|-------|
| Effective reflectivity `R` | `R` | **0.725** | — | Finesse N_R = 10; absorbs mirror quality + any PSF effect |

### Group 3 — Intensity Envelope and Detector (updated label only)

| Prompt | Variable | Default | Units | Notes |
|--------|----------|---------|-------|-------|
| Peak signal-to-noise ratio | `snr_peak` | `50.0` | — | OI 630.2046 nm fringe peak above offset |
| Linear vignetting `I₁` | `I1` | `-0.1` | — | I(r) = I₀·(1 + I₁·(r/r_max) + I₂·(r/r_max)²) |
| Quadratic vignetting `I₂` | `I2` | `0.005` | — | Must keep I(r) > 0 for all r ≤ r_max |
| Focal plane temperature | `T_fp_c` | **−20.0** | °C | Drives dark current model |

### Group 4 — Source Velocity (replaces Z03 Group 4)

| Prompt | Variable | Default | Units | Notes |
|--------|----------|---------|-------|-------|
| Line-of-sight velocity | `v_los_ms` | **−7500** | m/s | Doppler shift of OI 630.2046 nm; negative = blueshift |

The default −7500 m/s reflects the typical along-track spacecraft velocity contribution
to the line-of-sight Doppler, which dominates the total observed shift.

#### 4.1 Velocity hard limits and warnings

| Bound | Value | Rationale |
|-------|-------|-----------|
| Hard minimum | **−10000 m/s** | Well outside observable range; likely input error |
| Hard maximum | **+3000 m/s** | Well outside observable range; likely input error |
| Warning minimum | **−8500 m/s** | Below expected spacecraft + wind range |
| Warning maximum | **+1500 m/s** | Above expected spacecraft + wind range |

The science-relevant velocity range is approximately **−8000 to +1000 m/s**, spanning the
spacecraft velocity (≈ −7100 m/s along-track) combined with thermospheric winds
(typically ±200 m/s) and projection geometry. Values outside the warning range are accepted
with a console warning but not rejected.

#### 4.2 Velocity prompt implementation

```python
v_los_ms = _validated_prompt(
    "Line-of-sight velocity v_los [m/s]",
    default=-7500.0, units="m/s",
    hard_min=-10000.0, hard_max=3000.0,
    warn_min=-8500.0,  warn_max=1500.0,
)
```

The prompt should display a one-line physics note:
```
  [Negative = blueshift (toward spacecraft); typical range −8000 to +1000 m/s]
  [λ_obs = 630.2046 × (1 + v/c); at −7500 m/s: λ_obs ≈ 630.1888 nm]
```

---

## 5. Physics

### 5.1 Ideal Airy transmission function

Identical to Z03 v1.5 §5.1:

```
A_ideal(r; λ, d, R, α) = I(r) / [1 + F·sin²(π·2d·cos(θ(r)) / λ)]
```

where:
- `θ(r) = arctan(α · r)`
- `I(r) = I₀ · (1 + I₁·(r/r_max) + I₂·(r/r_max)²)`
- `F = 4R / (1 − R)²`

### 5.2 Doppler-shifted OI 630.2046 nm source

The OI 630.2046 nm airglow emission is treated as a **delta-function source** at the
Doppler-shifted observed wavelength. No thermal (temperature) broadening is applied —
temperature is not a science product of this instrument.

```
λ_obs = λ₀ × (1 + v_los / c)
```

where `v_los` is the total line-of-sight velocity (spacecraft + wind + Earth rotation),
positive for recession (redshift), negative for approach (blueshift).

| v_los (m/s) | λ_obs (nm) | Fringe shift direction |
|-------------|-----------|----------------------|
| −8000 | 630.1878 | Outward (larger r) |
| −7500 | 630.1888 | Outward |
| 0 | 630.2046 | Reference |
| +500 | 630.0315 | Inward (smaller r) |
| +1000 | 630.0325 | Inward |

> **Fringe shift direction convention:** Blueshift (negative v_los, approach) decreases λ_obs,
> which increases the phase `φ = 4πnd·cos(θ)/λ_obs` at fixed r, pushing fringes to larger
> radii (outward). This is the dominant effect: at −7500 m/s the OI fringe ring pattern is
> shifted outward by ~0.25 px (2×2 binned) relative to a stationary source.
>
> **Physical sign convention alignment:** This matches G01 `_generate_science_pixels()` where
> `lambda_obs = LAMBDA_OI_M × (1 + v_rel_ms / C_LIGHT_MS)` and positive v_rel (recession) is
> defined as moving fringes inward. Both conventions are consistent — Z02 uses the same formula.

### 5.3 SNR quadratic and I₀ derivation

For a **single-line** source, `I_peak = I₀` exactly (no composite factor). The SNR quadratic
simplifies from Z03:

```
SNR = I_peak / sqrt(I_peak + offset)

I_peak = I₀ = [SNR² + sqrt(SNR⁴ + 4·SNR²·offset)] / 2
```

```python
def snr_to_ipeak(snr: float, offset: float) -> float:
    """Identical to Z03 v1.5; valid for single-line source."""
    return (snr**2 + math.sqrt(snr**4 + 4 * snr**2 * offset)) / 2

I0 = snr_to_ipeak(snr_peak, OFFSET_ADU)   # I_peak = I₀; no (1+rel_638) denominator
```

With defaults snr_peak = 50, offset = 5:
```
I0 = I_peak = [2500 + sqrt(6250000 + 50000)] / 2 ≈ 2500 ADU
```

> **Contrast with Z03:** In Z03, `I_peak` is the composite (two-line) peak and
> `I₀ = I_peak / (1 + rel_638)`. In Z02, there is only one line, so `I_peak = I₀` directly.
> The `snr_to_ipeak()` function is unchanged — only the step `I₀ = I_peak / (1 + rel_638)`
> is replaced by `I₀ = I_peak`.

### 5.4 Science fringe profile

```
S_sci(r) = A_ideal(r; λ_obs, d, R, α, I₀, I₁, I₂) + offset
```

where `λ_obs = LAM_OI × (1 + v_los_ms / C_LIGHT_MS)` and `I₀ = snr_to_ipeak(snr_peak, OFFSET_ADU)`.

### 5.5 Noise model (identical to Z03 v1.5)

```python
# Science image pixel synthesis
dark_rate = DARK_REF_ADU_S * 2.0**((T_fp_c - T_REF_DARK_C) / T_DOUBLE_C)
mean_dark = dark_rate * exp_time_s

signal   = S_sci(r)                          # OI fringe + offset (no dark yet)
pixel    = Poisson(signal + mean_dark) + OFFSET_ADU
pixel    = clip(round(pixel), 0, 16383).astype(uint16)
```

```python
# Dark image pixel synthesis (identical to Z03 v1.5)
dark_rate = DARK_REF_ADU_S * 2.0**((T_fp_c - T_REF_DARK_C) / T_DOUBLE_C)
mean_dark = dark_rate * exp_time_s
pixel     = Poisson(mean_dark) + OFFSET_ADU
pixel     = clip(round(pixel), 0, 16383).astype(uint16)
```

No separate Gaussian read noise draw. Identical to Z03 v1.5.

### 5.6 Image layout — 1-row header

Identical to Z03 v1.5: 1-row P01 header at row 0, pixel data from row 1 onward.

---

## 6. BinningConfig — Image Geometry Constants

**Identical to Z03 v1.5 §6.** No changes. Reproduced here for completeness.

| Field | `binning=2` (2×2) | `binning=1` (1×1) |
|-------|-------------------|-------------------|
| `nrows` | 260 | 528 |
| `ncols` | 276 | 552 |
| `active_rows` | 259 | 527 |
| `n_meta_rows` | 1 | 1 |
| `cx_default` | 137.5 | 275.5 |
| `cy_default` | 130.0 | 264.0 |
| `r_max_px` | 110.0 | 220.0 |
| `alpha_default` | 1.6000e-4 | 0.8000e-4 |
| `pix_m` | 32.0e-6 | 16.0e-6 |
| `label` | `"2x2_binned"` | `"1x1_unbinned"` |

---

## 7. Fixed Constants (module-level, not prompted)

| Constant | Symbol | Value | Notes |
|----------|--------|-------|-------|
| OI rest wavelength | `LAM_OI` | `630.2046e-9` m | Replaces LAM_640, LAM_638 from Z03 |
| Speed of light | `C_LIGHT_MS` | `2.99792458e8` | m/s |
| Electronic offset | `OFFSET_ADU` | `5` | Bias + read noise; fixed |
| Dark reference rate | `DARK_REF_ADU_S` | `0.05` | ADU/px/s at T_REF_DARK_C |
| Dark reference temperature | `T_REF_DARK_C` | `−20.0` | °C |
| Dark doubling interval | `T_DOUBLE_C` | `6.5` | °C |
| Radial bins | `R_BINS` | `2000` | Avoids interpolation artefacts |
| Refractive index | `N_REF` | `1.0` | Air gap |

Constants **removed** relative to Z03: `LAM_640`, `LAM_638`.  
Constants **added** relative to Z03: `LAM_OI`, `C_LIGHT_MS`.

---

## 8. SynthParams Dataclass (v1.0)

```python
@dataclass
class SynthParams:
    # Group 0 — image geometry
    binning:    int     # 1 or 2
    cx:         float   # fringe centre column, pixels
    cy:         float   # fringe centre row, pixels
    # Group 1 — etalon geometry
    d_mm:       float   # etalon gap, mm  (default 20.0005)
    alpha:      float   # plate scale, rad/px  (mode-dependent default)
    # Group 2 — reflectivity
    R:          float   # effective reflectivity  (default 0.725)
    # Group 3 — intensity envelope and detector
    snr_peak:   float   # OI fringe peak SNR  (default 50.0)
    I1:         float   # linear vignetting  (default −0.1)
    I2:         float   # quadratic vignetting  (default 0.005)
    T_fp_c:     float   # focal plane temperature, °C  (default −20.0)
    # Group 4 — source velocity (replaces rel_638 from Z03)
    v_los_ms:   float   # line-of-sight velocity, m/s  (default −7500.0)
```

Changes vs Z03 v1.5 `SynthParams`:
- **Removed:** `rel_638`
- **Added:** `v_los_ms`

---

## 9. Stage Descriptions

### Stage A — Banner and Parameter Prompting

```
╔══════════════════════════════════════════════════════════════╗
║  Z02  Synthetic Science Image Generator  v1.0                ║
║  WindCube SOC — soc_sewell                                   ║
╚══════════════════════════════════════════════════════════════╝

Synthesises a matched science + dark image pair in authentic
WindCube .bin format. The OI 630.2046 nm Airy fringe pattern is
Doppler-shifted by the prompted line-of-sight velocity.
Press <Enter> to accept the default shown in parentheses.

──────────────────────────────────────────────────────────────
 GROUP 0  IMAGE GEOMETRY
──────────────────────────────────────────────────────────────
  Detector binning (1=unbinned 1×1, 2=binned 2×2)  (default 2):
  Fringe centre column cx [pixels]                  (default <mode>):
  Fringe centre row    cy [pixels]                  (default <mode>):

──────────────────────────────────────────────────────────────
 GROUP 1  ETALON GEOMETRY
──────────────────────────────────────────────────────────────
  Etalon gap d [mm]                                 (default 20.0005):
  Plate scale alpha [rad/px]                        (default <mode>):

──────────────────────────────────────────────────────────────
 GROUP 2  ETALON REFLECTIVITY
──────────────────────────────────────────────────────────────
  Effective reflectivity R                          (default 0.725):
    [Finesse N_R = π√R/(1−R); R=0.725 → N_R≈10.0]

──────────────────────────────────────────────────────────────
 GROUP 3  INTENSITY ENVELOPE AND DETECTOR
──────────────────────────────────────────────────────────────
  Peak SNR (OI 630.2046 nm fringe peak)                (default 50.0):
  Linear vignetting coefficient I_1                 (default -0.1):
  Quadratic vignetting coefficient I_2              (default 0.005):
  Focal plane temperature [°C]                      (default -20.0):

──────────────────────────────────────────────────────────────
 GROUP 4  SOURCE VELOCITY
──────────────────────────────────────────────────────────────
  Line-of-sight velocity v_los [m/s]                (default -7500):
    [Negative = blueshift; typical range -8000 to +1000 m/s]
    [λ_obs = 630.2046 × (1 + v/c); at -7500 m/s: λ_obs ≈ 630.1888 nm]
```

### Stage B — Derive Secondary Parameters

```python
cfg       = get_binning_config(binning)
lambda_obs = LAM_OI * (1.0 + v_los_ms / C_LIGHT_MS)
I0         = snr_to_ipeak(snr_peak, OFFSET_ADU)   # single line: I0 = I_peak
F_coef     = 4 * R / (1 - R)**2
N_R        = math.pi * math.sqrt(R) / (1 - R)
FSR_m      = LAM_OI**2 / (2.0 * d_mm * 1e-3)
dark_rate  = DARK_REF_ADU_S * 2.0**((T_fp_c - T_REF_DARK_C) / T_DOUBLE_C)
delta_lam  = lambda_obs - LAM_OI                  # for display
```

Console summary includes `v_los`, `λ_obs`, and finesse.

**Stage B validates:**
- Vignetting positivity: `I(r) > 0` for r ∈ {0, r_max/2, r_max} — hard error if violated

### Stage C — Synthesise Fringe Profile and Image

```python
r_grid     = np.linspace(0.0, cfg.r_max_px, R_BINS)
theta      = np.arctan(alpha * r_grid)
cos_theta  = np.cos(theta)
F_coef     = 4.0 * R / (1.0 - R)**2
vignette   = I0 * (1.0 + I1*(r_grid/cfg.r_max_px) + I2*(r_grid/cfg.r_max_px)**2)

phase      = 4.0 * np.pi * N_REF * (d_mm * 1e-3) * cos_theta / lambda_obs
A_oi       = vignette / (1.0 + F_coef * np.sin(phase / 2.0)**2)
profile_1d = A_oi + OFFSET_ADU

image_2d = radial_profile_to_image(
    profile_1d, r_grid,
    nrows=cfg.nrows, ncols=cfg.ncols,
    cx=cx, cy=cy,
    r_max=cfg.r_max_px,
    background=OFFSET_ADU,
)
```

### Stage D — Noise (identical to Z03 v1.5)

```python
exp_time_s  = exp_time_cts * TIMER_PERIOD_S
dark_rate   = DARK_REF_ADU_S * 2.0**((T_fp_c - T_REF_DARK_C) / T_DOUBLE_C)
mean_dark   = dark_rate * exp_time_s

# Science image
sci_signal  = image_2d + mean_dark
sci_noisy   = rng.poisson(np.maximum(sci_signal, 0)).astype(float) + OFFSET_ADU
sci_uint16  = np.clip(np.round(sci_noisy), 0, 16383).astype(np.uint16)

# Dark image
dark_noisy  = rng.poisson(np.full(image_2d.shape, mean_dark)).astype(float) + OFFSET_ADU
dark_uint16 = np.clip(np.round(dark_noisy), 0, 16383).astype(np.uint16)
```

### Stage E — Metadata (1-row header, identical to Z03 v1.5)

`write_header_row()` encodes `img_type = "science"` (vs `"cal"` in Z03). Everything else
about the header encoding is identical.

### Stages F–G — File Write, Diagnostics

Output filenames use `_sci_synth_z02_` in place of `_cal_synth_z03_`:

```
yyyymmddThhmmssZ_sci_synth_z02_<label>.bin
yyyymmddThhmmssZ_dark_synth_z02_<label>.bin
yyyymmddThhmmssZ_sci_synth_z02_<label>_truth.json
yyyymmddThhmmssZ_sci_synth_z02_<label>_diagnostic.png
```

Stage G title lines:
```python
f"d={d_mm} mm   α={alpha:.4e} rad/px   R={R:.3f} (N_R={N_R:.1f})   SNR={snr_peak}   [{cfg.label}]"
f"cx={cx:.1f}   cy={cy:.1f}   I₁={I1}   I₂={I2}   T_fp={T_fp_c}°C"
f"OI 630.2046 nm   v_los={v_los_ms:+.0f} m/s   λ_obs={lambda_obs*1e9:.4f} nm"
f"I₀ = {I0:.1f} ADU   (single line; I_peak = I₀)"
```

---

## 10. Output Files and `_truth.json` Schema (v1.0)

```json
{
  "z02_version": "1.0",
  "timestamp_utc": "2026-04-29T00:00:00Z",
  "random_seed": 12345678,
  "user_params": {
    "binning":    2,
    "cx":         137.5,
    "cy":         130.0,
    "d_mm":       20.0005,
    "alpha":      1.6000e-4,
    "R":          0.725,
    "snr_peak":   50.0,
    "I1":        -0.1,
    "I2":         0.005,
    "T_fp_c":    -20.0,
    "v_los_ms":  -7500.0
  },
  "derived_params": {
    "alpha_rad_per_px":      1.6000e-4,
    "lambda_obs_m":          6.299843e-7,
    "delta_lambda_m":       -1.575e-12,
    "I0_adu":                2500.0,
    "I_peak_adu":            2500.0,
    "finesse_N_R":           10.0,
    "finesse_coefficient_F": 9.63,
    "FSR_m":                 9.923e-12,
    "dark_rate_adu_px_s":    0.05
  },
  "fixed_constants": {
    "lam_oi_m":        6.302046e-7,
    "c_light_ms":      2.99792458e8,
    "offset_adu":      5,
    "dark_ref_adu_s":  0.05,
    "T_ref_dark_c":   -20.0,
    "T_double_c":      6.5,
    "R_bins":          2000,
    "n_ref":           1.0,
    "n_meta_rows":     1,
    "nrows":           260,
    "ncols":           276,
    "active_rows":     259,
    "r_max_px":        110.0,
    "pix_m":           3.2e-5,
    "label":           "2x2_binned"
  },
  "output_sci_file":  "20260429T000000Z_sci_synth_z02_2x2_binned.bin",
  "output_dark_file": "20260429T000000Z_dark_synth_z02_2x2_binned.bin"
}
```

**Differences from Z03 v1.5 `_truth.json`:**
- Key `z03_version` → `z02_version`
- `user_params`: `rel_638` removed; `v_los_ms` added
- `derived_params`: `Y_B`, `FSR_m` (neon-based) removed; `lambda_obs_m`, `delta_lambda_m` added;
  `I0_adu` = `I_peak_adu` (no composite denominator); `FSR_m` recomputed from `LAM_OI`
- `fixed_constants`: `lam_640_m`, `lam_638_m` removed; `lam_oi_m`, `c_light_ms` added
- Output keys: `output_cal_file` → `output_sci_file`

---

## 11. Parameter Validation and Bounds

### Groups 0–3 (identical to Z03 v1.5 §11)

See Z03 v1.5 §11. No changes.

### Group 4 — Source Velocity

| Parameter | Hard min | Hard max | Warning min | Warning max |
|-----------|----------|----------|-------------|-------------|
| `v_los_ms` | −10000 | +3000 | −8500 | +1500 |

---

## 12. Test Verification

The test suite for Z02 parallels Z03 v1.5 exactly, with calibration-specific tests replaced
by science-specific equivalents.

### Full test list (25 tests)

| Test ID | Description | Pass criterion |
|---------|-------------|---------------|
| `test_sci_output_shape_binned` | Sci `.bin` loads to (260, 276) uint16 | Shape matches |
| `test_dark_output_shape_binned` | Dark `.bin` loads to (260, 276) uint16 | Shape matches |
| `test_sci_output_shape_unbinned` | Sci `.bin` loads to (528, 552) uint16 | Shape matches |
| `test_dark_output_shape_unbinned` | Dark `.bin` loads to (528, 552) uint16 | Shape matches |
| `test_sci_header_round_trip` | 1-row header recoverable via `parse_header()` | All fields match; `img_type="science"` |
| `test_dark_header_round_trip` | Dark 1-row header correct | `img_type` correct; shutter closed |
| `test_fringe_peak_location` | Peak near (cx, cy) | Within 1 px of requested centre |
| `test_snr_achieved` | Peak SNR ≈ requested | Within ±20% of target |
| `test_dark_no_fringes` | Dark has no fringe structure | No periodic structure |
| `test_truth_json_complete` | All required keys in sidecar | See §12a |
| `test_default_params` | Script runs with all defaults | Both .bin + truth.json written |
| `test_I0_single_line` | I0_adu = snr_to_ipeak(snr_peak, OFFSET_ADU); no rel_638 division | Exact to 6 sig figs |
| `test_alpha_no_f_mm` | `f_mm` not in truth JSON | Key absent |
| `test_binning_config_binned` | BinningConfig for binning=2 | All fields match §6 table |
| `test_binning_config_unbinned` | BinningConfig for binning=1 | All fields match §6 table |
| `test_cx_cy_offset_binned` | Fringe centre displaced 10 px | Peak within 1 px of (cx+10, cy+10) |
| `test_cx_cy_offset_unbinned` | Fringe centre displaced 20 px | Peak within 1 px |
| `test_filename_label` | Output filename contains mode label | `"2x2_binned"` or `"1x1_unbinned"` in filename |
| `test_no_rel_638_param` | SynthParams has no `rel_638` field | AttributeError on access |
| `test_dark_current_scales_with_temperature` | Dark mean increases with T_fp_c | rate(−10°C) > 1.5 × rate(−20°C) |
| `test_offset_present_in_dark` | Dark floor ≈ OFFSET_ADU | min(dark) ≥ 4 ADU |
| `test_finesse_from_R` | N_R from default R ≈ 10.0 | |N_R − 10.0| < 0.1 |
| `test_1row_header` | Row 0 parses; row 1 is pixel data | parse_header() OK; pixel row 1 nonzero |
| `test_blueshift_moves_fringes_outward` | Fringe radius at v_los=−7500 > fringe radius at v_los=0 | First bright ring radius increases with blueshift |
| `test_redshift_moves_fringes_inward` | Fringe radius at v_los=+500 < fringe radius at v_los=0 | First bright ring radius decreases with redshift |

**Total: 25 tests.**

The last two tests (`test_blueshift_moves_fringes_outward`, `test_redshift_moves_fringes_inward`)
are new to Z02 and verify the Doppler fringe-shift direction. They have no analogue in Z03.

### 12a — `test_truth_json_complete`

```python
expected_user_keys = {
    "binning", "cx", "cy",
    "d_mm", "alpha", "R",
    "snr_peak", "I1", "I2", "T_fp_c",
    "v_los_ms"           # replaces rel_638
}
expected_derived_keys = {
    "alpha_rad_per_px", "lambda_obs_m", "delta_lambda_m",
    "I0_adu", "I_peak_adu",
    "finesse_N_R", "finesse_coefficient_F", "FSR_m", "dark_rate_adu_px_s"
}
expected_fixed_keys = {
    "lam_oi_m", "c_light_ms",   # replaces lam_640_m, lam_638_m
    "offset_adu", "dark_ref_adu_s", "T_ref_dark_c", "T_double_c",
    "R_bins", "n_ref", "n_meta_rows", "nrows", "ncols",
    "active_rows", "r_max_px", "pix_m", "label"
}
# Assert absent keys (must not appear anywhere in truth JSON)
for absent in ("rel_638", "lam_640_m", "lam_638_m", "sigma0", "sigma1",
               "sigma2", "B_dc", "f_mm", "sigma_read", "Y_B"):
    assert absent not in truth["user_params"]
    assert absent not in truth["fixed_constants"]
```

### 12b — Doppler fringe-shift direction tests

```python
def test_blueshift_moves_fringes_outward():
    """Blueshift (negative v_los) → λ_obs decreases → phase increases → fringes shift outward."""
    profile_rest  = synthesise_profile_1d(v_los_ms=0.0,    **default_etalon_params)
    profile_blue  = synthesise_profile_1d(v_los_ms=-7500.0, **default_etalon_params)
    r_peak_rest   = r_grid[np.argmax(profile_rest)]
    r_peak_blue   = r_grid[np.argmax(profile_blue)]
    assert r_peak_blue > r_peak_rest, (
        f"Blueshift should move fringes outward: "
        f"r_peak_blue={r_peak_blue:.2f} px, r_peak_rest={r_peak_rest:.2f} px"
    )

def test_redshift_moves_fringes_inward():
    """Redshift (positive v_los) → λ_obs increases → phase decreases → fringes shift inward."""
    profile_rest  = synthesise_profile_1d(v_los_ms=0.0,   **default_etalon_params)
    profile_red   = synthesise_profile_1d(v_los_ms=500.0, **default_etalon_params)
    r_peak_rest   = r_grid[np.argmax(profile_rest)]
    r_peak_red    = r_grid[np.argmax(profile_red)]
    assert r_peak_red < r_peak_rest, (
        f"Redshift should move fringes inward: "
        f"r_peak_red={r_peak_red:.2f} px, r_peak_rest={r_peak_rest:.2f} px"
    )
```

---

## 13. Known Limitations

- No thermal (temperature) broadening of the OI line. The source is treated as a
  delta function in wavelength. Temperature is not a science product of this instrument.
- No multiple-reflection contribution from the lower thermosphere or E-region. Single-layer
  airglow source assumed.
- No limb geometry or column emission rate modelling. `I₀` is the peak ADU in the fringe
  pattern, not a physical column emission rate.
- `r_max` is fixed by `BinningConfig`; not user-prompted.
- `v_los_ms` is the **total** line-of-sight velocity including spacecraft, wind, and Earth
  rotation contributions. Z02 does not decompose these — the truth JSON records the total.

---

## 14. Locked Decisions

**LD-1:** All instrument constants in §3 are shared with Z03 v1.5 and G01 v10. Any change
to a shared constant must be updated in all three specs simultaneously (per Z03 LD-5).

**LD-2:** The OI vacuum wavelength is 630.2046e-9 m (Edlén 1966 conversion from air wavelength 630.0304 nm). This is the authoritative value for Doppler calculations.
This is the vacuum wavelength computed from the measured air wavelength 630.0304 nm using the Edlén (1966) formula.
No further correction is applied.

**LD-3:** Blueshift (negative v_los) shifts fringes **outward** in radius. This is physically
correct and consistent with G01. The test `test_blueshift_moves_fringes_outward` encodes
this as an executable requirement.

**LD-4:** `I0 = I_peak` for the single-line source (no composite denominator). The
`snr_to_ipeak()` function is reused unchanged from Z03; only the downstream assignment changes.

---

## 15. Claude Code Implementation Prompt

> **Instructions for Claude Code:**  
> Read this entire spec first (§§1–14). Then read the current Z03 implementation as the
> reference — Z02 is Z03 with the source model replaced. Execute tasks in order.
> Gate on `pytest tests/test_z02.py -v` passing between tasks.
> Stop and report back after 10–15 minutes if tests are not passing.

```
cat PIPELINE_STATUS.md

# Read the Z02 spec in full before touching any code:
cat docs/specs/z02_synthetic_science_image_generator_2026-04-29.md

# Read Z03 as reference implementation:
cat src/fpi/z03_synthetic_calibration_image_generator.py

# Read shared infrastructure:
cat src/fpi/m02_calibration_synthesis.py
cat tests/test_z03.py   # use as template for test_z02.py

# ── TASK 1 of 5: Create z02_synthetic_science_image_generator.py ──────────
#
# Start by copying z03_synthetic_calibration_image_generator.py to
# z02_synthetic_science_image_generator.py, then make these targeted changes:
#
# 1a. MODULE CONSTANTS — replace neon constants with OI constants:
#     REMOVE: LAM_640, LAM_638
#     ADD:    LAM_OI = 630.2046e-9
#             C_LIGHT_MS = 2.99792458e8
#     All other constants (OFFSET_ADU, DARK_REF_ADU_S, T_REF_DARK_C,
#     T_DOUBLE_C, R_BINS, N_REF) unchanged.
#
# 1b. SynthParams dataclass:
#     REMOVE: rel_638: float
#     ADD:    v_los_ms: float   (after T_fp_c)
#
# 1c. BinningConfig._BINNING_CONFIGS: unchanged — copy verbatim.
#
# Gate: python -c "
#   from src.fpi.z02_synthetic_science_image_generator import SynthParams, LAM_OI
#   assert LAM_OI == 630.0e-9
#   p = SynthParams.__dataclass_fields__
#   assert 'v_los_ms' in p
#   assert 'rel_638' not in p
#   print('OK')
# "

# ── TASK 2 of 5: Update prompt_all_params() ───────────────────────────────
#
# 2a. Update banner text: "Synthetic Calibration Image Generator" →
#     "Synthetic Science Image Generator"
#
# 2b. Group 3: change SNR prompt label:
#     "Peak SNR (composite 640+638 nm peak)" →
#     "Peak SNR (OI 630.2046 nm fringe peak)"
#
# 2c. Group 4: REMOVE rel_638 prompt. ADD v_los_ms prompt:
#
#     print("\n──────────────────────────────────────────────────────────────")
#     print(" GROUP 4  SOURCE VELOCITY")
#     print("──────────────────────────────────────────────────────────────")
#     print("  [Negative = blueshift; typical range -8000 to +1000 m/s]")
#     print("  [λ_obs = 630.2046 × (1 + v/c); at -7500 m/s: λ_obs ≈ 630.1888 nm]")
#     v_los_ms = _validated_prompt(
#         "Line-of-sight velocity v_los [m/s]",
#         default=-7500.0, units="m/s",
#         hard_min=-10000.0, hard_max=3000.0,
#         warn_min=-8500.0,  warn_max=1500.0,
#     )
#
# 2d. Update SynthParams construction: remove rel_638=rel_638, add v_los_ms=v_los_ms.
#
# Gate: pytest tests/test_z02.py -v  (run even if test file not yet created)

# ── TASK 3 of 5: Update derive_secondary() and synthesise_profile() ───────
#
# 3a. derive_secondary():
#     REMOVE: I0 = I_peak / (1.0 + rel_638)  [composite denominator]
#     REPLACE WITH: I0 = I_peak              [single line; I_peak = I_0 directly]
#     ADD: lambda_obs = LAM_OI * (1.0 + params.v_los_ms / C_LIGHT_MS)
#          delta_lam  = lambda_obs - LAM_OI
#     UPDATE FSR: FSR_m = LAM_OI**2 / (2.0 * params.d_mm * 1e-3)
#     REMOVE: Y_B (was rel_638 alias)
#
# 3b. synthesise_profile():
#     REMOVE: A640, A638, rel_638 terms
#     REPLACE WITH single-line synthesis per §9 Stage C:
#
#       lambda_obs = LAM_OI * (1.0 + params.v_los_ms / C_LIGHT_MS)
#       phase      = 4.0 * np.pi * N_REF * (params.d_mm * 1e-3) * cos_theta / lambda_obs
#       A_oi       = vignette / (1.0 + F_coef * np.sin(phase / 2.0)**2)
#       profile_1d = A_oi + OFFSET_ADU
#
# Gate: pytest tests/test_z02.py -v

# ── TASK 4 of 5: Update output filenames, Stage G, truth JSON ─────────────
#
# 4a. Output filenames: "_cal_synth_z03_" → "_sci_synth_z02_"
#     Dark filenames: "_dark_synth_z03_" → "_dark_synth_z02_"
#
# 4b. Stage G title lines: update per §9 Stage G of this spec.
#     Show v_los, lambda_obs, and "I₀ = I_peak (single line)".
#
# 4c. write_truth_json(): update to v1.0 schema per §10:
#     Key: "z03_version" → "z02_version": "1.0"
#     user_params: remove rel_638; add v_los_ms
#     derived_params: remove Y_B; add lambda_obs_m, delta_lambda_m;
#                     set I_peak_adu = I0_adu (same value, no denominator);
#                     recompute FSR_m from LAM_OI
#     fixed_constants: remove lam_640_m, lam_638_m;
#                      add lam_oi_m = 6.302046e-7, c_light_ms = 2.99792458e8
#     Output key: "output_cal_file" → "output_sci_file"
#
# Gate: pytest tests/test_z02.py -v

# ── TASK 5 of 5: Create test_z02.py (25 tests) ────────────────────────────
#
# Copy tests/test_z03.py to tests/test_z02.py and adapt:
#
# 5a. Update all imports and module references: z03 → z02, SynthParams from z02.
#
# 5b. Update make_default_params():
#     Remove rel_638. Add v_los_ms=-7500.0.
#
# 5c. Rename/update tests:
#     test_cal_output_shape_binned → test_sci_output_shape_binned
#     test_cal_header_round_trip  → test_sci_header_round_trip
#       (assert img_type == "science", not "cal")
#     test_rel_638_ratio          → REMOVE (no neon ratio)
#     test_I0_option_a            → test_I0_single_line
#       (assert truth["derived_params"]["I0_adu"] ==
#               truth["derived_params"]["I_peak_adu"])
#     test_round_trip_I0          → test_round_trip_I0  (adapt for single line)
#
# 5d. Add 2 new Doppler direction tests per §12b:
#     test_blueshift_moves_fringes_outward
#     test_redshift_moves_fringes_inward
#
# 5e. Update test_truth_json_complete per §12a.
#
# Gate: pytest tests/test_z02.py -v — all 25 tests must pass.

# ── Commit ─────────────────────────────────────────────────────────────────
git add src/fpi/z02_synthetic_science_image_generator.py \
        tests/test_z02.py \
        docs/specs/z02_synthetic_science_image_generator_2026-04-29.md \
        PIPELINE_STATUS.md
git commit -m "feat: Z02 v1.0 — synthetic science image generator

Single Doppler-shifted OI 630.2046 nm source replaces two-line neon.
v_los prompt (default -7500 m/s) with range -10000 to +3000 m/s.
I0 = I_peak directly (no composite denominator).
All instrument constants, geometry, noise model identical to Z03 v1.5.
Blueshift moves fringes outward (verified by 2 new direction tests).
25/25 tests pass.

Derived from Z03 v1.5. Aligns with G01 v10 science pixel generator."

# ── REPORT BACK ────────────────────────────────────────────────────────────
# 1. Full pytest output for all 25 tests (pass/fail by name)
# 2. Derived values at default params: lambda_obs, I0, N_R, dark_rate
# 3. Full pytest summary line
# 4. Confirm fringe shift direction: at v_los=-7500 m/s, r_peak_blue > r_peak_rest?
#    Print the actual pixel values.
# 5. Any deviations from this spec
```

---

*End of Z02 Spec v1.0 — 2026-04-29*
