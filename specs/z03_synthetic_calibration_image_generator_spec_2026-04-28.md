# Z03 — Synthetic Calibration Image Generator

**Spec ID:** Z03  
**Tier:** 9 — Validation Testing  
**Module:** `z03_synthetic_calibration_image_generator.py`  
**Status:** Spec updated — awaiting implementation  
**Date:** 2026-04-28  
**Version:** 1.5  
**Author:** Scott Sewell / HAO  
**Repo:** `soc_sewell`  
**Dependencies:** S03 (physical constants), M01 (airy_ideal), M02 (radial_profile_to_image), P01 (metadata schema)

---

## Revision History

| Date | Version | Change |
|------|---------|--------|
| 2026-04-10 | 1.0 | Initial spec |
| 2026-04-10 | 1.0 | SNR quadratic corrected; Z03_OUTPUT_DIR env-var; 7 pytest cases |
| 2026-04-13 | 1.1 | Stage G simplified to 1×2 cal+dark; dark image added throughout |
| 2026-04-14 | 1.2 | Major: all 10 M05 inversion params user-prompted; `airy_modified()` replaces standalone helper; R and B_dc promoted from fixed to prompted |
| 2026-04-22 | 1.3 | `f_mm` → `alpha` prompt; Option A I₀ fix; rel_638 default 0.8→0.58; Claude Code prompt appended |
| 2026-04-22 | 1.3.1 | Implementation housekeeping: corrects illustrative I_peak example in §5.3a; records constants corrections (R_MAX_PX, CX/CY_DEFAULT, N_META_ROWS) as LD-1. No algorithm changes. 15/15 tests pass. |
| 2026-04-28 | 1.4 | Major: binning mode prompt added (1×1 unbinned vs 2×2 binned); arbitrary cx/cy prompt added for both modes; image-geometry constants refactored into BinningConfig; 7 new pytest cases. |
| **2026-04-28** | **1.5** | **Major physics overhaul. PSF broadening (σ₀,σ₁,σ₂) removed — absorbed into effective finesse/R. rel_638 updated to 0.344 (radial-profile measurement). Etalon gap default corrected to 20.0005 mm (Benoit). Plate scale default corrected to 1.6000e-4 rad/px. Bias and read noise unified as single electronic offset (5 ADU). Dark current model added, driven by prompted focal-plane temperature (default −20°C). Header reduced to 1 row (S19 retired). Dependency on `airy_modified()` replaced by `airy_ideal()`. All parameter defaults aligned with G01.** |

---

## 1. Purpose

Z03 creates a **synthetic 'truth' calibration image pair** — a neon fringe calibration image and
a companion dark image — both in authentic WindCube `.bin` format (1-row header + pixel data),
suitable for direct ingestion by Z01, F01, or any downstream pipeline module.

The script is **interactive**: it prompts the user for all instrument and calibration parameters
that the calibration inversion stage recovers from a real calibration image. Synthesis uses the
ideal Airy transmission function with a single effective reflectivity R (which captures both
mirror reflectivity and any unresolved instrumental contrast reduction). PSF broadening (σ₀,σ₁,σ₂)
is not applied separately because at the operational reflectivity R ≈ 0.53–0.73 it is degenerate
with an effective R and introduces fitting degeneracy without adding independent information.

Both output images constitute **ground-truth artefacts** — every parameter used in synthesis is
known exactly and recorded in a companion `_truth.json` sidecar.

As of v1.4, Z03 supports **two detector binning modes**:

| Mode | Binning | Image dimensions | Active Airy region | Pixel pitch |
|------|---------|-----------------|-------------------|-------------|
| `binned` | 2×2 | 260 rows × 276 cols | 256×256 px centred at (cx, cy) | 32 µm (2 × 16 µm) |
| `unbinned` | 1×1 | 528 rows × 552 cols | 512×512 px centred at (cx, cy) | 16 µm |

---

## 2. Relationship to Z01, F01, and G01

| Aspect | Z01 | F01 | Z03 | G01 |
|--------|-----|-----|-----|-----|
| Primary input | Real `.bin` + dark | 1D FringeProfile from Z01a | *(generates output)* | *(generates output)* |
| Output | Tolansky result (d, α, ε) | CalibrationResult (R, α, I₀, …) | `.bin` cal + dark + `_truth.json` | `.bin` cal + dark + science (bulk campaign) |
| Core physics | Ring-order WLS | Two-line Airy LM fit | Ideal Airy synthesis | Ideal Airy synthesis |
| Scale | Single interactive image pair | Single 1D profile | Single interactive image pair | ~120,000 frames per 30-day run |

Z03 and G01 use **identical Airy physics** and **identical instrument constants** (§3 below).
Any discrepancy between the two is a bug.

---

## 3. Authoritative Instrument Constants

These values are fixed across Z03 and G01. They are not user-prompted in G01 (they are
hardwired constants). They are user-prompted in Z03 with the values below as defaults.

| Constant | Symbol | Value | Source |
|----------|--------|-------|--------|
| Etalon gap | `d` | **20.0005 mm** | Benoit excess-fraction two-line recovery |
| Plate scale (2×2 binned) | `α` | **1.6000e-4 rad/px** | Tolansky two-line WLS |
| Plate scale (1×1 unbinned) | `α` | **0.8000e-4 rad/px** | = 1.6000e-4 / 2 |
| Effective reflectivity | `R` | **0.725** | Gives finesse N_R = 10.0; default (prompted) |
| Ne 640.2 nm wavelength | `λ₁` | **640.2248e-9 m** | Burns, Adams & Longwell (1950) IAU |
| Ne 638.3 nm wavelength | `λ₂` | **638.2991e-9 m** | Burns, Adams & Longwell (1950) IAU |
| 638/640 intensity ratio | `rel_638` | **0.344** | Radial-profile average of real cal images |
| Electronic offset | `offset` | **5 ADU** | Bias + read noise combined; real-image pedestal |
| Dark reference rate | `dark_ref` | **0.05 ADU/px/s** | At T_ref = −20°C |
| Dark doubling interval | `T_double` | **6.5°C** | Standard CCD rule |
| Focal plane temperature | `T_fp` | **−20°C** | Default operating temperature (prompted) |

> **Why a single electronic offset?**  
> The CCD97 operating in EM gain mode at −20°C with typical exposure times shows a residual
> pedestal of ~5 ADU after all dark subtraction. This bundles the deterministic bias offset and
> the rms read noise floor into a single additive constant. Both are small, fixed, and not
> science-relevant. The offset is added to every pixel; no per-pixel Gaussian read noise term
> is applied in synthesis. Dark current (exponential in temperature, linear in time) is handled
> separately because it is genuinely variable and physically distinct.

> **Why effective R rather than PSF broadening?**  
> At R = 0.53–0.73 the fringe FWHM spans several pixels and the Airy profile is broad. A
> shift-invariant Gaussian PSF blur is nearly degenerate with a reduction in effective R —
> both reduce fringe contrast by the same amount at every radius. Fitting both simultaneously
> is ill-conditioned. The effective R absorbs mirror quality, any unresolved PSF effect, and
> etalon plate defects into a single free parameter that the fitter can cleanly recover.
> The finesse N_R = π√R/(1−R) = 10 corresponds to R = 0.725.

---

## 4. User-Prompted Parameters

The script prompts sequentially in four groups. The Group 0 geometry prompts (binning, cx, cy)
are unchanged from v1.4. Groups 1–4 are updated as follows.

### Group 0 — Image Geometry (unchanged from v1.4)

| Prompt | Variable | Default | Units |
|--------|----------|---------|-------|
| Detector binning | `binning` | `2` | — |
| Fringe centre column | `cx` | *(mode-dependent)* | px |
| Fringe centre row | `cy` | *(mode-dependent)* | px |

See v1.4 §4.1a–§4.1b for mode-dependent defaults and hard limits.

### Group 1 — Etalon Geometry

| Prompt | Variable | Default | Units | Notes |
|--------|----------|---------|-------|-------|
| Etalon gap `d` | `d_mm` | **20.0005** | mm | Benoit two-line recovery |
| Plate scale `α` | `alpha` | *(mode-dependent, see §3)* | rad/px | Tolansky WLS result |

### Group 2 — Etalon Reflectivity

| Prompt | Variable | Default | Units | Notes |
|--------|----------|---------|-------|-------|
| Effective reflectivity `R` | `R` | **0.725** | — | Finesse N_R = 10; absorbs mirror quality + any PSF effect |

> The old Group 2 PSF parameters (σ₀, σ₁, σ₂) are **removed** in v1.5. They are no longer
> prompted and no longer appear in `SynthParams` or `_truth.json`.

### Group 3 — Intensity Envelope and Detector

| Prompt | Variable | Default | Units | Notes |
|--------|----------|---------|-------|-------|
| Peak signal-to-noise ratio | `snr_peak` | `50.0` | — | Sets composite I_peak via SNR quadratic (§5.3) |
| Linear vignetting `I₁` | `I1` | `-0.1` | — | I(r) = I₀·(1 + I₁·(r/r_max) + I₂·(r/r_max)²) |
| Quadratic vignetting `I₂` | `I2` | `0.005` | — | Must keep I(r) > 0 for all r ≤ r_max |
| Focal plane temperature | `T_fp_c` | **−20.0** | °C | Drives dark current model |

> **B_dc is removed** in v1.5. The electronic offset (5 ADU) is a fixed constant, not
> user-prompted. This avoids confusion between bias, read noise, and dark current.

### Group 4 — Source Parameter

| Prompt | Variable | Default | Units | Notes |
|--------|----------|---------|-------|-------|
| Intensity ratio 638 nm / 640 nm | `rel_638` | **0.344** | — | Measured from radial-profile averages of real cal images |

---

## 5. Physics

### 5.1 Ideal Airy transmission function

Z03 v1.5 uses a plain ideal Airy function — no PSF broadening:

```
A_ideal(r; λ, d, R, α) = I(r) / [1 + F·sin²(π·2d·cos(θ(r)) / λ)]
```

where:
- `θ(r) = arctan(α · r)`   (incidence angle at pixel radius r)
- `I(r) = I₀ · (1 + I₁·(r/r_max) + I₂·(r/r_max)²)`   (vignetting envelope)
- `F = 4R / (1 − R)²`   (Airy coefficient)

This replaces `M01.airy_modified()`. The new dependency is `M01.airy_ideal()` (or an
equivalent inline implementation). If `airy_ideal()` does not yet exist in M01, implement
it inline in Z03 and file a spec note to add it to M01.

### 5.2 Two neon reference lines

| Line | Wavelength λ | Relative intensity |
|------|-------------|--------------------|
| Ne 640.2 nm (strong) | 640.2248e-9 m | 1.0 (reference) |
| Ne 638.3 nm (weak) | 638.2991e-9 m | `rel_638` = 0.344 |

Source: Burns, Adams & Longwell (1950), IAU "S" standard. Ratio from radial-profile average
of real WindCube calibration images.

### 5.3 SNR quadratic and I₀ derivation

The `snr_peak` prompt specifies the composite peak SNR of the combined two-line profile above
the electronic offset:

```
SNR = I_peak / sqrt(I_peak + offset)

I_peak = [SNR² + sqrt(SNR⁴ + 4·SNR²·offset)] / 2   (positive root)
```

Note: the noise floor is just `offset` (5 ADU) because read noise is already bundled into
`offset` and dark current at the fringe peak timescale is negligible. The formula simplifies
relative to v1.3:

```python
def snr_to_ipeak(snr: float, offset: float) -> float:
    return (snr**2 + math.sqrt(snr**4 + 4 * snr**2 * offset)) / 2
```

The per-line amplitude I₀ follows from the Option A relationship (unchanged from v1.3):

```
I₀ = I_peak / (1 + rel_638)
```

With defaults snr_peak=50, offset=5, rel_638=0.344:
```
I_peak = [2500 + sqrt(6250000 + 4·2500·5)] / 2 ≈ 2500 ADU   (composite)
I₀     = 2500 / 1.344 ≈ 1860 ADU                              (per-line)
```

### 5.4 Composite profile

```
S_cal(r) = A_ideal(r; λ₆₄₀, d, R, α, I₀, I₁, I₂)
         + rel_638 × A_ideal(r; λ₆₃₈, d, R, α, I₀, I₁, I₂)
         + offset
```

where `I₀ = I_peak / (1 + rel_638)`.

### 5.5 Noise model

```python
# Calibration image pixel synthesis
dark_rate = DARK_REF_ADU_S * 2.0**((T_fp_c - T_REF_DARK_C) / T_DOUBLE_C)
mean_dark = dark_rate * exp_time_s           # typically negligible at −20°C

signal = S_cal(r)                            # neon fringe + offset
pixel  = Poisson(signal + mean_dark) + OFFSET_ADU
pixel  = clip(round(pixel), 0, 16383).astype(uint16)
```

```python
# Dark image pixel synthesis
dark_rate = DARK_REF_ADU_S * 2.0**((T_fp_c - T_REF_DARK_C) / T_DOUBLE_C)
mean_dark = dark_rate * exp_time_s
pixel     = Poisson(mean_dark) + OFFSET_ADU
pixel     = clip(round(pixel), 0, 16383).astype(uint16)
```

There is **no separate Gaussian read noise draw**. The electronic offset (5 ADU) is added
as a fixed constant; shot noise on the signal itself provides all the stochastic variation.

### 5.6 Image layout — 1-row header

As of v1.5, the binary format uses a **1-row header** (276 uint16 words) followed by pixel
data. The S19 4-row header is retired. The header row is row 0; pixel data begins at row 1.

This aligns Z03 with G01's `_write_bin_file()` layout and with the actual flight binary format.

`BinningConfig.n_meta_rows` is updated to `1` in both binning modes.

---

## 6. BinningConfig — Image Geometry Constants (updated from v1.4)

The `BinningConfig` NamedTuple is unchanged in structure. The `n_meta_rows` field changes
from 4 to 1.

### Authoritative BinningConfig values (v1.5)

| Field | `binning=2` (2×2) | `binning=1` (1×1) | Notes |
|-------|-------------------|-------------------|-------|
| `nrows` | 260 | 528 | Total rows including 1 header row |
| `ncols` | 276 | 552 | Total columns |
| `active_rows` | 259 | 527 | nrows − 1 |
| `n_meta_rows` | **1** | **1** | Updated from 4 in v1.4 |
| `cx_default` | 137.5 | 275.5 | (ncols − 1) / 2 |
| `cy_default` | 130.0 | 264.0 | (nrows − 1) / 2; updated for 1-row header |
| `r_max_px` | 110.0 | 220.0 | FlatSat/flight; unbinned = 2 × binned |
| `alpha_default` | **1.6000e-4** | **0.8000e-4** | Updated from 1.6133e-4 / 0.8067e-4 |
| `pix_m` | 32.0e-6 | 16.0e-6 | Physical pixel pitch |
| `label` | `"2x2_binned"` | `"1x1_unbinned"` | Appears in output filenames |

> **cy_default update:** With 1 metadata row instead of 4, the first pixel row is row 1
> (not row 4). The geometric centre of the active pixel area for binned mode is
> (1 + 259/2) = 130.5 → rounded to 130.0 px. For unbinned: (1 + 527/2) = 264.5 → 264.0 px.
> These are used as prompt defaults only; the user may enter any value within hard limits.

---

## 7. Fixed Constants (module-level, not prompted)

| Constant | Symbol | Value | Notes |
|----------|--------|-------|-------|
| Electronic offset | `OFFSET_ADU` | **5** | Bias + read noise; fixed, not prompted |
| Dark reference rate | `DARK_REF_ADU_S` | `0.05` | ADU/px/s at T_REF_DARK_C |
| Dark reference temperature | `T_REF_DARK_C` | `−20.0` | °C |
| Dark doubling interval | `T_DOUBLE_C` | `6.5` | °C |
| Radial bins | `R_BINS` | `2000` | Avoids interpolation artefacts |
| Refractive index | `N_REF` | `1.0` | Air gap |
| Ne 640.2 nm | `LAM_640` | `640.2248e-9` m | Burns et al. (1950) |
| Ne 638.3 nm | `LAM_638` | `638.2991e-9` m | Burns et al. (1950) |

`SIGMA_READ` and `B_dc` are **removed** as module-level constants.

---

## 8. SynthParams Dataclass (v1.5)

```python
@dataclass
class SynthParams:
    # Group 0 — image geometry
    binning:   int     # 1 or 2
    cx:        float   # fringe centre column, pixels
    cy:        float   # fringe centre row, pixels
    # Group 1 — etalon geometry
    d_mm:      float   # etalon gap, mm  (default 20.0005)
    alpha:     float   # plate scale, rad/px  (mode-dependent default)
    # Group 2 — reflectivity (PSF params removed)
    R:         float   # effective reflectivity  (default 0.725)
    # Group 3 — intensity envelope and detector
    snr_peak:  float   # composite peak SNR  (default 50.0)
    I1:        float   # linear vignetting  (default −0.1)
    I2:        float   # quadratic vignetting  (default 0.005)
    T_fp_c:    float   # focal plane temperature, °C  (default −20.0)
    # Group 4 — source
    rel_638:   float   # 638/640 intensity ratio  (default 0.344)
```

Removed fields vs v1.4: `sigma0`, `sigma1`, `sigma2`, `B_dc`.  
New fields vs v1.4: `T_fp_c`.

---

## 9. Stage Descriptions (changes from v1.4)

### Stage A — Banner and Parameter Prompting

Group 2 now shows only R. PSF prompts removed. B_dc prompt replaced by T_fp_c:

```
──────────────────────────────────────────────────────────────
 GROUP 2  ETALON REFLECTIVITY
──────────────────────────────────────────────────────────────
  Effective reflectivity R                         (default 0.725):
    [Finesse N_R = π√R/(1−R); R=0.725 → N_R≈10.0]

──────────────────────────────────────────────────────────────
 GROUP 3  INTENSITY ENVELOPE AND DETECTOR
──────────────────────────────────────────────────────────────
  Peak SNR (composite 640+638 nm peak)             (default 50.0):
  Linear vignetting coefficient I_1                (default -0.1):
  Quadratic vignetting coefficient I_2             (default 0.005):
  Focal plane temperature [°C]                     (default -20.0):

──────────────────────────────────────────────────────────────
 GROUP 4  SOURCE
──────────────────────────────────────────────────────────────
  Intensity ratio 638nm/640nm (rel_638)            (default 0.344):
```

### Stage B — Derive Secondary Parameters

```python
cfg      = get_binning_config(binning)
I_peak   = snr_to_ipeak(snr_peak, OFFSET_ADU)      # offset = 5 ADU
I0       = I_peak / (1.0 + rel_638)
F_coef   = 4 * R / (1 - R)**2
N_R      = math.pi * math.sqrt(R) / (1 - R)
FSR_m    = LAM_640**2 / (2.0 * d_mm * 1e-3)
dark_rate = DARK_REF_ADU_S * 2.0**((T_fp_c - T_REF_DARK_C) / T_DOUBLE_C)
```

Printed to terminal including finesse value computed from R.

**Stage B validates:**
- Vignetting positivity: `I(r) > 0` for r ∈ {0, r_max/2, r_max} — hard error if violated
- (PSF positivity check removed — no PSF parameters)

### Stage C — Synthesise Fringe Profile and Image

```python
r_grid    = np.linspace(0.0, cfg.r_max_px, R_BINS)
theta     = np.arctan(alpha * r_grid)
cos_theta = np.cos(theta)
F_coef    = 4.0 * R / (1.0 - R)**2
vignette  = I0 * (1.0 + I1 * (r_grid/cfg.r_max_px) + I2 * (r_grid/cfg.r_max_px)**2)

def airy_ideal(lam):
    phase = 4.0 * np.pi * N_REF * (d_mm * 1e-3) * cos_theta / lam
    return vignette / (1.0 + F_coef * np.sin(phase / 2.0)**2)

A640       = airy_ideal(LAM_640)
A638       = airy_ideal(LAM_638)
profile_1d = A640 + rel_638 * A638 + OFFSET_ADU

image_2d = radial_profile_to_image(
    profile_1d, r_grid,
    nrows=cfg.nrows, ncols=cfg.ncols,
    cx=cx, cy=cy,
    r_max=cfg.r_max_px,
    background=OFFSET_ADU,
)
```

### Stage D — Noise

```python
exp_time_s  = exp_time_cts * TIMER_PERIOD_S
dark_rate   = DARK_REF_ADU_S * 2.0**((T_fp_c - T_REF_DARK_C) / T_DOUBLE_C)
mean_dark   = dark_rate * exp_time_s

# Calibration image
cal_signal  = image_2d + mean_dark
cal_noisy   = rng.poisson(np.maximum(cal_signal, 0)).astype(float) + OFFSET_ADU
cal_uint16  = np.clip(np.round(cal_noisy), 0, 16383).astype(np.uint16)

# Dark image
dark_noisy  = rng.poisson(np.full(image_2d.shape, mean_dark)).astype(float) + OFFSET_ADU
dark_uint16 = np.clip(np.round(dark_noisy), 0, 16383).astype(np.uint16)
```

### Stage E — Metadata (1-row header)

`build_metadata()` writes a **single header row** (row 0, 276 uint16 words) using the P01
`_encode_header()` format. `n_meta_rows = 1`. Pixel data occupies rows 1 onward.

The S19 4-row header is no longer written. The `embed_metadata()` function from v1.4 is
replaced by `write_header_row()`.

### Stages F–G — File Write, Diagnostics

Unchanged in logic. Output filenames include the mode label as in v1.4.

Stage G title lines updated to show finesse N_R instead of σ₀:
```
f"d={d_mm} mm   α={alpha:.4e} rad/px   R={R:.3f} (N_R={N_R:.1f})   SNR={snr_peak}   [{cfg.label}]"
f"cx={cx:.1f}   cy={cy:.1f}   I₁={I1}   I₂={I2}   T_fp={T_fp_c}°C"
f"Ne: 640.2248 nm (×1.0)   638.2991 nm (×{rel_638})"
f"I₀ (per-line) = {I0:.1f} ADU   I_peak (composite) = {I_peak:.1f} ADU"
```

---

## 10. Output Files and `_truth.json` Schema (v1.5)

Output filenames unchanged from v1.4.

```json
{
  "z03_version": "1.5",
  "timestamp_utc": "2026-04-28T00:00:00Z",
  "random_seed": 12345678,
  "user_params": {
    "binning":   2,
    "cx":        137.5,
    "cy":        130.0,
    "d_mm":      20.0005,
    "alpha":     1.6000e-4,
    "R":         0.725,
    "snr_peak":  50.0,
    "I1":       -0.1,
    "I2":        0.005,
    "T_fp_c":   -20.0,
    "rel_638":   0.344
  },
  "derived_params": {
    "alpha_rad_per_px":       1.6000e-4,
    "I_peak_adu":             2500.0,
    "I0_adu":                 1860.0,
    "Y_B":                    0.344,
    "finesse_N_R":            10.0,
    "finesse_coefficient_F":  9.63,
    "FSR_m":                  1.0238e-11,
    "dark_rate_adu_px_s":     0.05
  },
  "fixed_constants": {
    "offset_adu":      5,
    "dark_ref_adu_s":  0.05,
    "T_ref_dark_c":   -20.0,
    "T_double_c":      6.5,
    "R_bins":          2000,
    "n_ref":           1.0,
    "lam_640_m":       6.402248e-7,
    "lam_638_m":       6.382991e-7,
    "n_meta_rows":     1,
    "nrows":           260,
    "ncols":           276,
    "active_rows":     259,
    "r_max_px":        110.0,
    "pix_m":           3.2e-5,
    "label":           "2x2_binned"
  },
  "output_cal_file":  "20260428T000000Z_cal_synth_z03_2x2_binned.bin",
  "output_dark_file": "20260428T000000Z_dark_synth_z03_2x2_binned.bin"
}
```

---

## 11. Parameter Validation and Bounds (updated from v1.4)

### Group 0 — unchanged from v1.4

### Group 1 — Etalon Geometry

| Parameter | Hard min | Hard max | Warning range |
|-----------|----------|----------|---------------|
| `d_mm` | 15.0 | 25.0 | Outside (19.5, 20.5) |
| `alpha` | 1e-5 | 1e-3 | Outside (0.5e-4, 5e-4) |

### Group 2 — Reflectivity (PSF params removed)

| Parameter | Hard min | Hard max | Warning range |
|-----------|----------|----------|---------------|
| `R` | 0.01 | 0.99 | Outside (0.4, 0.92) |

The warning range for R corresponds to finesse between ~3.2 and ~28. Values outside this
range are physically unusual for a thermospheric FPI.

### Group 3 — Intensity and Detector

| Parameter | Hard min | Hard max | Warning range |
|-----------|----------|----------|---------------|
| `snr_peak` | 1.0 | 500.0 | Outside (10.0, 200.0) |
| `I1` | -0.9 | 0.9 | Outside (-0.5, 0.5) |
| `I2` | -0.9 | 0.9 | Outside (-0.5, 0.5) |
| `T_fp_c` | -60.0 | 20.0 | Outside (-40.0, 0.0) |

### Group 4 — Source

| Parameter | Hard min | Hard max | Warning range |
|-----------|----------|----------|---------------|
| `rel_638` | 0.0 | 2.0 | Outside (0.1, 1.0) |

---

## 12. Test Verification (v1.5)

Tests removed: `test_psf_broadening_effect`, `test_vignetting_effect` (PSF params gone).  
Tests updated: all tests that referenced `sigma0`, `sigma1`, `sigma2`, `B_dc`, `SIGMA_READ`
must be updated to remove those parameters.

### Carry-forward tests (updated signatures)

| Test ID | Description | Pass criterion |
|---------|-------------|---------------|
| `test_cal_output_shape_binned` | Cal `.bin` loads to (260, 276) uint16 | Shape matches |
| `test_dark_output_shape_binned` | Dark `.bin` loads to (260, 276) uint16 | Shape matches |
| `test_cal_output_shape_unbinned` | Cal `.bin` loads to (528, 552) uint16 | Shape matches |
| `test_dark_output_shape_unbinned` | Dark `.bin` loads to (528, 552) uint16 | Shape matches |
| `test_cal_header_round_trip` | 1-row header recoverable via `parse_header()` | All written fields match |
| `test_dark_header_round_trip` | Dark 1-row header correct | `img_type="cal"` in dark, shutter closed |
| `test_fringe_peak_location` | Peak near (cx, cy) | Within 1 px of requested centre |
| `test_snr_achieved` | Peak SNR ≈ requested | Within ±20% of target |
| `test_rel_638_ratio` | Family amplitude ratio ≈ rel_638 | Within ±5% |
| `test_dark_no_fringes` | Dark has no fringe structure | No periodic structure |
| `test_truth_json_complete` | All required keys in sidecar | See §12a |
| `test_default_params` | Script runs with all defaults | Both .bin + truth.json written |
| `test_I0_option_a` | I0_adu = I_peak / (1 + rel_638) | Exact to 6 sig figs |
| `test_round_trip_I0` | Z03 truth I0_adu matches F01 recovered I0 | Within 5% |
| `test_alpha_no_f_mm` | f_mm not in truth JSON | Key absent |
| `test_binning_config_binned` | BinningConfig for binning=2 | All fields match §6 table |
| `test_binning_config_unbinned` | BinningConfig for binning=1 | All fields match §6 table |
| `test_cx_cy_offset_binned` | Fringe centre displaced 10 px | Peak within 1 px of (cx+10, cy+10) |
| `test_cx_cy_offset_unbinned` | Fringe centre displaced 20 px | Peak within 1 px |
| `test_filename_label` | Output filename contains mode label | String present |

### New tests in v1.5

| Test ID | Description | Pass criterion |
|---------|-------------|---------------|
| `test_no_sigma_params` | SynthParams has no sigma0/sigma1/sigma2/B_dc fields | AttributeError on access |
| `test_dark_current_scales_with_temperature` | Dark frame mean increases with T_fp_c | mean(T=−10) > mean(T=−20) by expected factor |
| `test_offset_present_in_dark` | Dark frame minimum ≈ OFFSET_ADU | min(dark) ≥ 4 ADU |
| `test_finesse_from_R` | N_R computed from default R ≈ 10.0 | |N_R − 10.0| < 0.1 |
| `test_1row_header` | File row 0 parses as valid header; row 1 is pixel data | parse_header() succeeds; pixel row 1 nonzero |

**Total: 25 tests must pass.**

### 12a — Updated `test_truth_json_complete`

```python
expected_user_keys = {
    "binning", "cx", "cy",
    "d_mm", "alpha", "R",
    "snr_peak", "I1", "I2", "T_fp_c", "rel_638"
}
expected_derived_keys = {
    "alpha_rad_per_px", "I_peak_adu", "I0_adu", "Y_B",
    "finesse_N_R", "finesse_coefficient_F", "FSR_m", "dark_rate_adu_px_s"
}
expected_fixed_keys = {
    "offset_adu", "dark_ref_adu_s", "T_ref_dark_c", "T_double_c",
    "R_bins", "n_ref", "lam_640_m", "lam_638_m",
    "n_meta_rows", "nrows", "ncols", "active_rows", "r_max_px", "pix_m", "label"
}
# Assert absent keys
for absent in ("sigma0", "sigma1", "sigma2", "B_dc", "f_mm", "sigma_read"):
    assert absent not in truth["user_params"]
    assert absent not in truth["fixed_constants"]
```

---

## 13. Known Limitations

- No EM gain excess noise factor.
- Vignetting is modelled as a smooth polynomial; spatially non-uniform illumination from
  real optical aberrations is not captured.
- `r_max` is fixed by `BinningConfig`; not user-prompted.
- `pix_m` retained in `BinningConfig` for metadata/documentation only; not used in any
  optical calculation.

---

## 14. Locked Decisions

**LD-1:** Authoritative fixed-constant values — carry forward from v1.3; updated for
1-row header and new defaults in v1.5.

| Constant | v1.4 value | v1.5 value | Source |
|----------|-----------|------------|--------|
| `n_meta_rows` | 4 | **1** | S19 retired; P01 1-row header |
| `cy_default` (binned) | 129.5 | **130.0** | Recentred for 1-row header |
| `cy_default` (unbinned) | 263.5 | **264.0** | Recentred for 1-row header |
| `alpha_default` (binned) | 1.6133e-4 | **1.6000e-4** | Corrected Tolansky value |
| `alpha_default` (unbinned) | 0.8067e-4 | **0.8000e-4** | Corrected Tolansky value |
| `d_mm` default | 20.0006 | **20.0005** | Corrected Benoit two-line recovery |
| `R` default | 0.53 | **0.725** | Effective finesse ≈ 10 |
| `rel_638` default | 0.58 | **0.344** | Real-image radial-profile measurement |

**LD-2:** SNR quadratic noise floor is now `offset` (5 ADU) only, not `B_dc + sigma_read²`.

**LD-3:** PSF broadening parameters σ₀, σ₁, σ₂ are permanently retired from Z03. They are
degenerate with effective R at operational finesse values and cause ill-conditioned fits.

**LD-4:** Unbinned r_max = 220 px (= 2 × 110 px). Do not change to 256 px.

**LD-5:** G01 and Z03 must use identical values for all constants in §3. Any future update
to one spec must be reflected in the other before implementation.

---

## 15. Claude Code Implementation Prompt

> **Instructions for Claude Code:**  
> Read this entire spec first (§§1–14). Then execute the tasks below in order.
> Gate on `pytest tests/test_z03.py -v` passing between tasks.
> Do not implement anything not described here.
> Stop and report back after 10–15 minutes if tests are not passing.

```
cat PIPELINE_STATUS.md

# Read the full spec before touching any code:
cat docs/specs/z03_synthetic_calibration_image_generator_spec_2026-04-28_v1.5.md

# Read these existing files for context:
cat src/fpi/z03_synthetic_calibration_image_generator.py
cat src/fpi/m01_airy_forward_model_2026_04_05.py
cat src/fpi/m02_calibration_synthesis.py
cat tests/test_z03.py

# ── TASK 1 of 6: Remove PSF parameters; add T_fp_c; update constants ──────
#
# 1a. In SynthParams dataclass:
#     REMOVE: sigma0, sigma1, sigma2, B_dc
#     ADD: T_fp_c: float  (after I2, before rel_638)
#     UPDATE: R default to 0.725
#
# 1b. Remove module-level constants: SIGMA_READ (and any B_dc constant)
#     Add: OFFSET_ADU = 5  (int, fixed)
#          DARK_REF_ADU_S = 0.05
#          T_REF_DARK_C = -20.0
#          T_DOUBLE_C = 6.5
#
# 1c. Update BinningConfig values per §6 table:
#     n_meta_rows: 4 → 1 in both modes
#     cy_default:  129.5 → 130.0  (binned);  263.5 → 264.0  (unbinned)
#     alpha_default: 1.6133e-4 → 1.6000e-4  (binned)
#                    0.8067e-4 → 0.8000e-4  (unbinned)
#
# Gate: python -c "from src.fpi.z03_synthetic_calibration_image_generator import
#        get_binning_config, OFFSET_ADU; c=get_binning_config(2);
#        assert c.n_meta_rows==1; assert OFFSET_ADU==5; print('OK')"

# ── TASK 2 of 6: Update prompt_all_params() ───────────────────────────────
#
# 2a. In Group 1, update d_mm default: 20.0006 → 20.0005
#     Update alpha defaults to use cfg.alpha_default (1.6000e-4 / 0.8000e-4)
#
# 2b. In Group 2, REMOVE sigma0, sigma1, sigma2 prompts.
#     Keep only R prompt. Update R default to 0.725.
#     Add finesse hint text after prompt:
#       print(f"    [Finesse N_R = π√R/(1−R); R=0.725 → N_R≈10.0]")
#
# 2c. In Group 3, REMOVE B_dc prompt.
#     ADD T_fp_c prompt after I2:
#       T_fp_c = _validated_prompt(
#           "Focal plane temperature [°C]",
#           default=-20.0, units="°C",
#           hard_min=-60.0, hard_max=20.0,
#           warn_min=-40.0, warn_max=0.0,
#       )
#
# 2d. In Group 4, update rel_638 default: 0.58 → 0.344
#
# 2e. Update SynthParams construction call to match new fields.
#
# Gate: pytest tests/test_z03.py -v  (some tests will fail — that is expected)

# ── TASK 3 of 6: Replace airy_modified with airy_ideal in synthesis ───────
#
# 3a. In synthesise_profile() (or equivalent):
#     REMOVE call to airy_modified() from M01.
#     REPLACE with inline airy_ideal() as specified in §9 Stage C:
#
#       r_grid    = np.linspace(0.0, cfg.r_max_px, R_BINS)
#       theta     = np.arctan(params.alpha * r_grid)
#       cos_theta = np.cos(theta)
#       F_coef    = 4.0 * params.R / (1.0 - params.R)**2
#       vignette  = I0 * (1.0 + params.I1*(r_grid/cfg.r_max_px)
#                                + params.I2*(r_grid/cfg.r_max_px)**2)
#
#       def _airy(lam):
#           phase = 4.0 * np.pi * N_REF * (params.d_mm * 1e-3) * cos_theta / lam
#           return vignette / (1.0 + F_coef * np.sin(phase / 2.0)**2)
#
#       A640       = _airy(LAM_640)
#       A638       = _airy(LAM_638)
#       profile_1d = A640 + derived.rel_638 * A638 + OFFSET_ADU
#
#     If airy_modified is no longer needed by any other function in z03, remove
#     its import. Do NOT modify M01 itself.
#
# 3b. Update derive_secondary() to:
#     - Remove sigma-related validation (check_psf_positive)
#     - Add finesse computation: N_R = math.pi * math.sqrt(R) / (1 - R)
#     - Use snr_to_ipeak(snr_peak, OFFSET_ADU)  [offset=5, not B_dc+sigma_read²]
#     - Add dark_rate = DARK_REF_ADU_S * 2**((T_fp_c - T_REF_DARK_C)/T_DOUBLE_C)
#
# Gate: pytest tests/test_z03.py -v

# ── TASK 4 of 6: Update noise model and header writer ─────────────────────
#
# 4a. In add_noise() (calibration image):
#     REMOVE Gaussian read noise draw.
#     Use: pixel = Poisson(signal + mean_dark) + OFFSET_ADU  (no Gaussian)
#     mean_dark = DARK_REF_ADU_S * 2**((T_fp_c - T_REF_DARK_C)/T_DOUBLE_C) * exp_time_s
#
# 4b. In synthesise_dark():
#     REMOVE Gaussian read noise draw.
#     Use: pixel = Poisson(mean_dark) + OFFSET_ADU
#
# 4c. In the header/metadata writing stage:
#     Change n_meta_rows from 4 to 1 (already handled via BinningConfig).
#     Replace embed_metadata() (4-row S19 writer) with write_header_row():
#       - Calls _encode_header(meta) → np.ndarray shape (276,) dtype ">u2"
#       - Writes as row 0 of the output array
#       - Pixel data occupies rows 1 onward
#     The _encode_header() function should be imported from or verified against
#     G01's implementation in validation/gen01_synthetic_metadata_generator_2026_04_16.py
#     to ensure identical binary layout.
#
# Gate: pytest tests/test_z03.py -v

# ── TASK 5 of 6: Update tests ─────────────────────────────────────────────
#
# 5a. Remove tests that reference removed parameters:
#     test_psf_broadening_effect, test_vignetting_effect
#     Any test fixture using sigma0/sigma1/sigma2/B_dc → update to use new SynthParams
#
# 5b. Update make_default_params() helper in test file:
#     Remove sigma0/sigma1/sigma2/B_dc fields.
#     Add T_fp_c=-20.0.
#     Update R=0.725, rel_638=0.344, d_mm=20.0005, alpha=cfg.alpha_default.
#
# 5c. Add 5 new tests per §12:
#
# TEST: test_no_sigma_params
#   params = make_default_params()
#   for attr in ("sigma0", "sigma1", "sigma2", "B_dc"):
#       assert not hasattr(params, attr), f"SynthParams should not have {attr}"
#
# TEST: test_dark_current_scales_with_temperature
#   from src.fpi.z03_synthetic_calibration_image_generator import (
#       DARK_REF_ADU_S, T_REF_DARK_C, T_DOUBLE_C
#   )
#   exp_time_s = 1.0
#   rate_cold = DARK_REF_ADU_S * 2**((-20.0 - T_REF_DARK_C) / T_DOUBLE_C)
#   rate_warm = DARK_REF_ADU_S * 2**((-10.0 - T_REF_DARK_C) / T_DOUBLE_C)
#   assert rate_warm > rate_cold * 1.5, "Dark rate should at least 1.5x higher at −10 vs −20°C"
#
# TEST: test_offset_present_in_dark
#   Synthesise dark image with T_fp_c=-20.0, exp_time very short (0.001 s)
#   so dark current ≈ 0. Assert np.min(dark_image) >= 4.
#
# TEST: test_finesse_from_R
#   import math
#   R = 0.725
#   N_R = math.pi * math.sqrt(R) / (1.0 - R)
#   assert abs(N_R - 10.0) < 0.1
#
# TEST: test_1row_header
#   Synthesise cal image with default params.
#   raw = np.frombuffer(open(cal_path, "rb").read(), dtype=">u2")
#   assert raw.shape == (260 * 276,)
#   header_row = raw[:276]
#   # rows (header_row[0]) and cols (header_row[1]) should match BinningConfig
#   assert header_row[0] == 260
#   assert header_row[1] == 276
#   # Row 1 should be pixel data (nonzero — should have fringe signal)
#   pixel_row1 = raw[276:552]
#   assert np.any(pixel_row1 > 10), "Expected pixel signal in row 1"
#
# Gate: pytest tests/test_z03.py -v — all 25 tests must pass.

# ── TASK 6 of 6: Update truth_json, Stage G, PIPELINE_STATUS, commit ──────
#
# 6a. write_truth_json(): update to v1.5 schema per §10.
#     Remove sigma0/sigma1/sigma2/B_dc/sigma_read from all sections.
#     Add T_fp_c to user_params.
#     Add finesse_N_R, dark_rate_adu_px_s to derived_params.
#     Add offset_adu, dark_ref_adu_s, T_ref_dark_c, T_double_c to fixed_constants.
#     Rename "fixed_defaults" key to "fixed_constants".
#
# 6b. make_diagnostic_figure(): update title strings per §9 Stage G.
#
# 6c. Update PIPELINE_STATUS.md: Z03 version 1.5, note physics overhaul.
#
git add src/fpi/z03_synthetic_calibration_image_generator.py \
        tests/test_z03.py \
        docs/specs/z03_synthetic_calibration_image_generator_spec_2026-04-28_v1.5.md \
        PIPELINE_STATUS.md
git commit -m "feat: Z03 v1.5 — physics overhaul: ideal Airy, effective R, 1-row header

Remove PSF broadening (sigma0/sigma1/sigma2) — degenerate with effective R.
Add T_fp_c prompt; dark current model now temperature-driven.
Bias + read noise unified as fixed OFFSET_ADU = 5 ADU.
Update defaults: d=20.0005 mm, alpha=1.6000e-4 rad/px, R=0.725,
rel_638=0.344. 1-row header (S19 retired). 25/25 tests pass.

Aligns physics with G01. LD-3, LD-5 recorded."

# ── REPORT BACK ────────────────────────────────────────────────────────────
# 1. Full pytest output for all 25 tests (pass/fail by name)
# 2. Derived values at default params: I_peak, I0, N_R, dark_rate
# 3. Full pytest summary line
# 4. Whether airy_ideal was added to M01 or kept inline — confirm which
# 5. Any deviations from this spec
```

---

*End of Z03 Spec v1.5 — 2026-04-28*
