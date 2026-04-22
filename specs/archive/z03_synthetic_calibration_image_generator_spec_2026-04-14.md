# Z03 — Synthetic Calibration Image Generator

**Spec ID:** Z03  
**Tier:** 9 — Validation Testing  
**Module:** `z03_synthetic_calibration_image_generator.py`  
**Status:** Implemented — updated spec  
**Date:** 2026-04-14  
**Author:** Scott Sewell / HAO  
**Repo:** `soc_sewell`  
**Dependencies:** S03 (physical constants), S10 (M02, calibration fringe synthesis), S19 (P01, metadata schema)

---

## 1. Purpose

Z03 creates a **synthetic 'truth' calibration image pair** — a neon fringe calibration image and a companion dark image — both in authentic WindCube `.bin` format (260 rows × 276 cols, uint16 little-endian), suitable for direct ingestion by S01, Z01, Z02, or any downstream pipeline module.

The script is **interactive**: it prompts the user for all 10 instrumental and calibration parameters that the M05 inversion stage recovers from a real calibration image. Synthesis uses the full PSF-broadened, vignetting-modulated Airy model (M01 `airy_modified()`) rather than an ideal Airy function. This ensures that every degree of freedom visible to the inverter can be independently set and recovered.

The dark image contains only the noise background (no fringe signal), using the same noise model as the calibration image. This provides a matched synthetic dark for use in the dark-subtraction step of S12/M03.

Both output images constitute **ground-truth artefacts** — every parameter used in synthesis is known exactly and is recorded in a companion `_truth.json` sidecar.

---

## 2. Relationship to Z01 and Z02

| Aspect | Z01 | Z02 | Z03 |
|--------|-----|-----|-----|
| Primary input | Real `.bin` + dark `.bin` | Real `.bin` or `.npy` | *(none — generates output)* |
| Interactivity | ROI, crop, dark selection | Dark selection, ROI | Parameter entry only |
| Output | Diagnostic figures + recovered d, f, α, ε | Diagnostic figures + residuals | `.bin` cal + dark images + `_truth.json` sidecar |
| Core physics | Runs S12 + S13 (analysis) | Runs S12 + S13 (analysis) | Runs S10/M02 (synthesis) |
| Metadata | Reads S19 header | Reads S19 header | **Writes** S19 header (both images) |

Z03 is the **upstream complement** to Z01/Z02: it produces the controlled synthetic cal+dark pair that those scripts are designed to ingest and analyse.

---

## 3. Script Overview

```
z03_synthetic_calibration_image_generator.py
│
├── Stage A  — Banner and parameter prompting (interactive, 10 + 1 parameters)
├── Stage B  — Derive secondary optical parameters
├── Stage C  — Build InstrumentParams; synthesise PSF-broadened two-line neon Airy
│             fringe image with vignetting (noise-free float)
├── Stage D  — Apply noise model → calibration image (uint16)
│             Synthesise dark image (noise-only, uint16)
├── Stage E  — Build S19-compliant metadata; embed into cal image rows 0–3
│             Build S19-compliant metadata; embed into dark image rows 0–3
├── Stage F  — Write cal .bin + dark .bin + _truth.json sidecar
└── Stage G  — Diagnostic display (2-panel figure)
```

---

## 4. User-Prompted Parameters

The script opens a terminal session and prompts sequentially in four groups. Defaults are shown; pressing `<Enter>` accepts the default. All 10 parameters map directly to the free parameters recovered by the M05 calibration fringe inversion, plus one supplemental source parameter (`rel_638`).

### Group 1 — Etalon Geometry

| Prompt | Variable | Default | Units | Notes |
|--------|----------|---------|-------|-------|
| Etalon gap `d` | `d_mm` | `20.106` | mm | Operational gap (Tolansky-derived) |
| Imaging lens focal length `f` | `f_mm` | `199.12` | mm | Lens effective focal length; sets α |

### Group 2 — Etalon Reflectivity and PSF

| Prompt | Variable | Default | Units | Notes |
|--------|----------|---------|-------|-------|
| Plate reflectivity `R` | `R` | `0.53` | — | FlatSat-measured effective reflectivity |
| Average PSF width `σ₀` | `sigma0` | `0.5` | pixels | Shift-variant Gaussian broadening, mean |
| PSF sine variation `σ₁` | `sigma1` | `0.1` | pixels | Amplitude of sin(πr/r_max) term |
| PSF cosine variation `σ₂` | `sigma2` | `-0.05` | pixels | Amplitude of cos(πr/r_max) term |

### Group 3 — Intensity Envelope and Bias

| Prompt | Variable | Default | Units | Notes |
|--------|----------|---------|-------|-------|
| Peak signal-to-noise ratio | `snr_peak` | `50.0` | — | Sets I₀ via SNR quadratic (Section 5.3) |
| Linear vignetting coefficient `I₁` | `I1` | `-0.1` | — | I(r) = I₀·(1 + I₁·(r/r_max) + I₂·(r/r_max)²) |
| Quadratic vignetting coefficient `I₂` | `I2` | `0.005` | — | Must keep I(r) > 0 for all r ≤ r_max |
| Bias pedestal `B` | `B_dc` | `300` | ADU | CCD bias + stray light floor |

### Group 4 — Source Parameters (supplemental, not an inversion free parameter)

| Prompt | Variable | Default | Units | Notes |
|--------|----------|---------|-------|-------|
| Relative intensity of 638 nm line | `rel_638` | `0.8` | — | I₆₃₈ / I₆₄₀; known a priori during inversion |

> **Design note — why 10 + 1?**  
> The 10 parameters in Groups 1–3 correspond exactly to the free parameters recovered by M05: {d, α, R, σ₀, σ₁, σ₂, I₀, I₁, I₂, B}. Setting all 10 to non-default values creates a maximally realistic inversion challenge and ensures that no parameter is inadvertently constrained by a fixed default.  
> `rel_638` is listed separately because it is a property of the Ne source, not of the instrument, and is treated as a known constant during inversion.

### 4.1 Prompt implementation pattern

```python
def _validated_prompt(label: str, default: float, units: str,
                      hard_min: float, hard_max: float,
                      warn_min: float, warn_max: float) -> float:
    ...
```

All prompts are wrapped in `try/except ValueError` with a re-prompt loop. Values outside hard limits are rejected immediately and force re-entry. Values inside hard limits but outside the warning range print an advisory and proceed after the Y/n confirmation step.

After all prompts, the script echoes a parameter summary table and asks the user to confirm (Y/n) before proceeding.

---

## 5. Physics: Two-Line Neon Fringe Synthesis

### 5.1 PSF-broadened Airy transmission function

Z03 now uses the full M01 `airy_modified()` function rather than a simplified ideal Airy helper. This ensures that the PSF broadening and vignetting present in the synthesis are the same degrees of freedom that M05 inverts.

The ideal on-axis Airy intensity at pixel radius r for a single wavelength λ is:

```
A_ideal(r; λ, d, R, α, I₀, I₁, I₂) = I(r) / [1 + F·sin²(π·2d·cos(θ(r)) / λ)]
```

where:
- `θ(r) = arctan(α · r)` — angle from optical axis
- `I(r) = I₀ · (1 + I₁·(r/r_max) + I₂·(r/r_max)²)` — quadratic vignetting envelope
- `F = 4R / (1 − R)²` — coefficient of finesse
- `α = pixel_pitch / f` — plate scale (rad/pixel)

PSF broadening is then applied via a shift-variant Gaussian convolution (mean-sigma approximation):

```
σ(r) = σ₀ + σ₁·sin(π·r/r_max) + σ₂·cos(π·r/r_max)
A_modified(r) = gaussian_filter1d(A_ideal(r), sigma=mean(σ))
```

Implemented by calling `m01.airy_modified(r_grid, lam, d, R, alpha, n, r_max, I0, I1, I2, sigma0, sigma1, sigma2)` directly.

### 5.2 Two neon reference lines

| Line | Wavelength λ (nm) | Wavenumber (cm⁻¹) | Relative intensity |
|------|-------------------|--------------------|--------------------|
| Ne 640.2 nm | 640.2248 | 15615.211 | 1.0 (reference) |
| Ne 638.3 nm | 638.2991 | 15662.315 | `rel_638` (user-set) |

Source: Burns, Adams & Longwell (1950), IAU "S" standard.  
Wavelengths are imported from `src.constants.NE_WAVELENGTH_1_M` / `NE_WAVELENGTH_2_M` at runtime, with fallback to module-level literals.

The composite noise-free 1D radial profile is:

```
S_cal(r) = airy_modified(r; λ₆₄₀, d, R, α, I₀, I₁, I₂, σ₀, σ₁, σ₂)
         + rel_638 · airy_modified(r; λ₆₃₈, d, R, α, I₀, I₁, I₂, σ₀, σ₁, σ₂)
         + B
```

The 2D noise-free image is built from this 1D profile via `m02.radial_profile_to_image()`:

```
I_float(x, y) = S_cal(sqrt((x − cx)² + (y − cy)²))
```

> **Important:** `I₀` in the formula above is not the same as `I_peak` in the old simplified model. `I₀` is the intensity envelope coefficient at r = 0; the actual peak fringe intensity depends on `I₀`, `I₁`, `I₂`, `R`, `σ₀`, and the fringe phase at r = 0. The relationship between `snr_peak` and `I₀` is therefore iterative — see Section 5.3.

### 5.3 Deriving I₀ from SNR

The peak SNR is measured at the first fringe maximum, which occurs at a radius r_peak determined by d, α, and λ₆₄₀. Because `I₀` enters through the vignetting envelope, the mapping from SNR to `I₀` is:

```
I_peak_approx = I_envelope(r_peak; I₀, I₁, I₂)  [evaluated at r = 0 if r_peak is unknown a priori]
```

For the purposes of SNR → I₀ conversion, the same quadratic as before is used on an effective `I_peak`:

```
SNR = I_peak / sqrt(I_peak + B + sigma_read²)

I_peak² - SNR²·I_peak - SNR²·(B + sigma_read²) = 0
```

Positive root:
```python
def snr_to_ipeak(snr: float, B_dc: float, sigma_read: float) -> float:
    noise_floor = B_dc + sigma_read**2
    return (snr**2 + math.sqrt(snr**4 + 4 * snr**2 * noise_floor)) / 2
```

`I₀` is then set equal to `I_peak` (i.e., the vignetting envelope is normalised to I₀ at r = 0). The effective peak intensity at r_peak will differ from `I_peak` by the vignetting factor `(1 + I₁·(r_peak/r_max) + I₂·(r_peak/r_max)²)`, which is close to 1.0 for the default vignetting coefficients.

> **Note:** The simplified approximation `I_peak ≈ SNR²·(B + σ_read²) / (SNR² - 1)` must not be used — it diverges at SNR = 1 and is negative for SNR < 1.

### 5.4 Plate scale α derivation

```
α [rad/pixel] = pixel_pitch [m] / f [m]
```

Default `pixel_pitch = 32.0 µm` (CCD97 native 16 µm × 2×2 binning).

### 5.5 Dark image model

The synthetic dark image contains only the noise background — no fringe signal. It is synthesised as:

```
I_dark(x, y) = B_dc  (uniform, before noise)
```

The same noise model as Stage D is applied: Poisson draws on `B_dc` (representing dark current and stray photons) plus Gaussian read noise, using an independent RNG draw from the same seed sequence.

```
dark_float = B_dc · ones(nrows, ncols)
dark_noisy = clip(Poisson(dark_float) + Gaussian(0, sigma_read), 0, 16383).astype(uint16)
```

---

## 6. Fixed Default Parameters

These are physically motivated constants that are **not** free parameters of the M05 inversion and are therefore not prompted. All are written into `_truth.json` for full traceability.

| Parameter | Symbol | Default Value | Source |
|-----------|--------|--------------|--------|
| Read noise | `sigma_read` | `50.0` ADU | CCD97 EM gain regime estimate |
| Image centre (col) | `cx` | `137.5` | Geometric centre of 276-col array |
| Image centre (row) | `cy` | `129.5` | Geometric centre of 260-row active region |
| Pixel pitch (binned) | `pix_m` | `32.0e-6` m | CCD97 16 µm × 2×2 binning |
| Max usable radius | `r_max` | `110.0` px | FlatSat/flight value |
| Radial bins | `R_bins` | `2000` | Avoids interpolation artefacts (must be ≥ 2000) |
| Image dimensions | `nrows, ncols` | `260, 276` | WindCube Level-0 standard |
| Metadata rows | `n_meta_rows` | `4` | S19 header occupies first 4 rows |
| Ne 640.2 nm wavelength | `lam_640` | `640.2248e-9` m | Burns et al. (1950) |
| Ne 638.3 nm wavelength | `lam_638` | `638.2991e-9` m | Burns et al. (1950) |
| Refractive index | `n_ref` | `1.0` | Air gap |
| Output dtype | — | `uint16` | 14-bit pixels, zero-padded to 16-bit |
| Byte order | — | little-endian | WindCube `.bin` standard |

> **Parameters removed from fixed defaults vs. previous spec:** `R` (now Group 2 prompt), `B_dc` (now Group 3 prompt). `sigma_read` remains fixed because it is a detector noise property, not an optical parameter recovered by M05.

---

## 7. Stage Descriptions

### Stage A — Banner and Parameter Prompting

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

──────────────────────────────────────────────────────────────
 GROUP 1  ETALON GEOMETRY
──────────────────────────────────────────────────────────────
  Etalon gap d [mm]                       (default 20.106):
  Imaging lens focal length f [mm]        (default 199.12):

──────────────────────────────────────────────────────────────
 GROUP 2  ETALON REFLECTIVITY AND PSF
──────────────────────────────────────────────────────────────
  Plate reflectivity R                    (default 0.53):
  Average PSF width sigma_0 [pixels]      (default 0.5):
  PSF sine variation sigma_1 [pixels]     (default 0.1):
  PSF cosine variation sigma_2 [pixels]   (default -0.05):

──────────────────────────────────────────────────────────────
 GROUP 3  INTENSITY ENVELOPE AND BIAS
──────────────────────────────────────────────────────────────
  Peak SNR                                (default 50.0):
  Linear vignetting coefficient I_1       (default -0.1):
  Quadratic vignetting coefficient I_2    (default 0.005):
  Bias pedestal B [ADU]                   (default 300):

──────────────────────────────────────────────────────────────
 GROUP 4  SOURCE (not an inversion free parameter)
──────────────────────────────────────────────────────────────
  Relative intensity 638 nm / 640 nm      (default 0.8):
```

After all prompts, the script echoes a parameter summary table and asks the user to confirm (Y/n) before proceeding.

### Stage B — Derive Secondary Parameters

Computed (not prompted) from user inputs plus fixed defaults:

- `alpha_rad_per_px = pix_m / (f_mm * 1e-3)` — plate scale
- `I0` — from SNR quadratic (Section 5.3, positive root)
- `FSR_mm = lam_640**2 / (2 * d_mm * 1e-3)` — free spectral range
- `finesse_F = 4*R / (1-R)**2`
- `finesse_N = pi * sqrt(R) / (1 - R)` — reflectivity finesse

These are printed to the terminal for the user's information.

### Stage C — Synthesise Fringe Image (noise-free float)

Stage C builds an `InstrumentParams` object (from M01) from all user-supplied and fixed parameters, then calls `m01.airy_modified()` directly for each Ne line:

```python
params = InstrumentParams(
    t       = d_mm * 1e-3,
    R_refl  = R,
    n       = n_ref,
    alpha   = alpha_rad_per_px,
    I0      = I0,           # derived from snr_peak
    I1      = I1,
    I2      = I2,
    sigma0  = sigma0,
    sigma1  = sigma1,
    sigma2  = sigma2,
    B       = B_dc,
    r_max   = r_max,
)

r_grid = np.linspace(0.0, r_max, R_bins)

A640 = airy_modified(r_grid, lam_640, params.t, params.R_refl, params.alpha,
                     params.n, params.r_max, params.I0, params.I1, params.I2,
                     params.sigma0, params.sigma1, params.sigma2)

A638 = airy_modified(r_grid, lam_638, params.t, params.R_refl, params.alpha,
                     params.n, params.r_max, params.I0, params.I1, params.I2,
                     params.sigma0, params.sigma1, params.sigma2)

profile_1d = A640 + rel_638 * A638 + params.B

I_float = radial_profile_to_image(profile_1d, r_grid,
                                  image_size=ncols, cx=cx, cy=cy, bias=params.B)
```

> **Note on image dimensions:** The detector is 276 × 260 (non-square). `radial_profile_to_image()` is called with a square 276 × 276 array; the result is then trimmed to 260 rows by discarding the outermost (r > 110) rows top and bottom. Pixels beyond `r_max` are filled with `B_dc`.

> Rows 0–3 are synthesised normally at this stage and overwritten with metadata in Stage E.

### Stage D — Apply Noise Model; Synthesise Dark Image

**Calibration image** — noise applied to `I_float`:

```python
ss   = np.random.SeedSequence()
seed = ss.entropy                        # logged to _truth.json

rng  = np.random.default_rng(ss)

# Poisson photon noise + Gaussian read noise
signal_counts = rng.poisson(np.clip(I_float, 0, None)).astype(np.float64)
read_noise    = rng.standard_normal(size=signal_counts.shape) * sigma_read
image_cal     = np.clip(signal_counts + read_noise, 0, 16383).astype(np.uint16)
```

**Dark image** — noise only, no fringe signal, drawn from the same RNG stream:

```python
dark_float  = np.full((nrows, ncols), B_dc, dtype=np.float64)
dark_counts = rng.poisson(dark_float).astype(np.float64)
dark_read   = rng.standard_normal(size=dark_float.shape) * sigma_read
image_dark  = np.clip(dark_counts + dark_read, 0, 16383).astype(np.uint16)
```

Both images use the **same seed sequence** so the synthesis is fully reproducible from `_truth.json`.

### Stage E — Build and Embed S19 Metadata

S19 defines a metadata structure embedded in the **first `n_meta_rows` rows** (rows 0–3) of each image array. Each row is 276 uint16 pixels = 552 bytes; total header = 2208 bytes.

#### 7.1 Calibration image metadata

| S19 Field | Value |
|-----------|-------|
| `image_type` | `"Cal"` |
| `n_rows` | `260` |
| `n_cols` | `276` |
| `binning` | `2` |
| `shutter_status` | `"Open"` |
| `cal_lamp_1` | `"C1_On"` |
| `cal_lamp_2` | `"C2_Off"` |
| `cal_lamp_3` | `"C3_Off"` |
| `date_utc` | system clock `yyyymmdd` |
| `time_utc` | system clock `hhmmss` |
| `exposure_ms` | `120000` (120 s) |
| `etalon_temp_1` | `24.00` °C |
| All other fields | `0` |

#### 7.2 Dark image metadata

Same as calibration metadata with the following differences:

| S19 Field | Value |
|-----------|-------|
| `image_type` | `"Dark"` |
| `shutter_status` | `"Closed"` |
| `cal_lamp_1` | `"C1_Off"` |
| `time_utc` | system clock `hhmmss` at dark write time |

#### 7.3 Serialisation into pixel rows

```python
meta_json   = json.dumps(meta, separators=(',', ':'))
meta_bytes  = meta_json.encode('utf-8')
target_bytes = n_meta_rows * ncols * 2          # 4 × 276 × 2 = 2208
meta_padded = meta_bytes.ljust(target_bytes, b'\x00')
meta_uint16 = np.frombuffer(meta_padded, dtype='<u2').reshape(n_meta_rows, ncols)
image[0:n_meta_rows, :] = meta_uint16
```

> **Single-write guarantee:** metadata is written exactly once per image, here in Stage E. No other stage writes to rows 0–3 after this point.

### Stage F — Write Output Files

#### Output filenames

```
yyyymmddThhmmssZ_cal_synth_z03.bin
yyyymmddThhmmssZ_dark_synth_z03.bin
yyyymmddThhmmssZ_cal_synth_z03_truth.json
```

The timestamp in the dark filename matches the cal filename (same synthesis session). The `_synth_z03` infix distinguishes synthetic files from real flight data.

#### .bin writes

```python
image_cal.astype('<u2').tofile(path_cal)
assert path_cal.stat().st_size  == 260 * 276 * 2

image_dark.astype('<u2').tofile(path_dark)
assert path_dark.stat().st_size == 260 * 276 * 2
```

#### _truth.json sidecar

```json
{
  "z03_version": "1.2",
  "timestamp_utc": "2026-04-14T00:00:00Z",
  "random_seed": 12345678,
  "user_params": {
    "d_mm":       20.106,
    "f_mm":       199.12,
    "R":          0.53,
    "sigma0":     0.5,
    "sigma1":     0.1,
    "sigma2":    -0.05,
    "snr_peak":   50.0,
    "I1":        -0.1,
    "I2":         0.005,
    "B_dc":       300,
    "rel_638":    0.8
  },
  "derived_params": {
    "alpha_rad_per_px":    1.6071e-4,
    "I0_adu":              2312.5,
    "FSR_mm":              1.017e-5,
    "finesse_coefficient_F": 5.73,
    "finesse_N":           3.84
  },
  "fixed_defaults": {
    "sigma_read":   50.0,
    "cx":           137.5,
    "cy":           129.5,
    "pix_m":        3.2e-5,
    "r_max_px":     110.0,
    "R_bins":       2000,
    "nrows":        260,
    "ncols":        276,
    "n_ref":        1.0,
    "lam_640_m":    6.402248e-7,
    "lam_638_m":    6.382991e-7
  },
  "output_cal_file":  "20260414T000000Z_cal_synth_z03.bin",
  "output_dark_file": "20260414T000000Z_dark_synth_z03.bin"
}
```

> **Note on finesse values:** With `R = 0.53` (FlatSat default), `F = 4×0.53/(0.47)² ≈ 9.6` and `N_R = π√0.53/0.47 ≈ 4.9`. These are substantially different from the `R = 0.82` values in the previous spec version (F ≈ 90, N_R ≈ 25). Downstream Z01/Z02 tests that compare to expected finesse values must use the user-entered `R`, not a hardcoded expected value.

#### Output directory

Default: `C:\Users\sewell\Documents\GitHub\soc_synthesized_data\`, created if absent.  
Override via `Z03_OUTPUT_DIR` environment variable for Linux/CI use:

```python
DEFAULT_OUTPUT_DIR = Path(
    os.environ.get("Z03_OUTPUT_DIR",
                   r"C:\Users\sewell\Documents\GitHub\soc_synthesized_data")
)
```

### Stage G — Diagnostic Display

A 1×2 matplotlib figure is saved as `{stem}_diagnostic.png` alongside the `.bin` files. `plt.show()` is never called — the figure is saved only.

| Panel | Content |
|-------|---------|
| **Left** | Synthetic calibration image (imshow, grey colormap, log-scaled). Title line 1: `d={d_mm} mm  f={f_mm} mm  R={R}  SNR={snr_peak}`. Title line 2: `σ₀={sigma0}  I₁={I1}  I₂={I2}  B={B_dc} ADU`. Title line 3: `Ne: 640.2248 nm (×1.0)  638.2991 nm (×{rel_638})` |
| **Right** | Synthetic dark image (imshow, grey colormap, linear scale). Title: `Dark image — B={B_dc} ADU, σ_read={sigma_read} ADU` |

---

## 8. Output Files Summary

| File | Description |
|------|-------------|
| `yyyymmddThhmmssZ_cal_synth_z03.bin` | Synthetic calibration image — fringe signal + noise, S19 metadata in rows 0–3 |
| `yyyymmddThhmmssZ_dark_synth_z03.bin` | Synthetic dark image — noise background only, S19 metadata in rows 0–3 |
| `yyyymmddThhmmssZ_cal_synth_z03_truth.json` | Ground-truth sidecar with all 10 synthesis parameters, seed, and both output filenames |
| `yyyymmddThhmmssZ_cal_synth_z03_diagnostic.png` | 1×2 side-by-side cal + dark display figure |

---

## 9. Parameter Validation and Bounds

### Group 1 — Etalon Geometry

| Parameter | Hard min | Hard max | Warning range |
|-----------|----------|----------|---------------|
| `d_mm` | 15.0 | 25.0 | Outside (19.0, 21.5) |
| `f_mm` | 100.0 | 300.0 | Outside (180.0, 220.0) |

### Group 2 — Reflectivity and PSF

| Parameter | Hard min | Hard max | Warning range |
|-----------|----------|----------|---------------|
| `R` | 0.01 | 0.99 | Outside (0.3, 0.85) |
| `sigma0` | 0.0 | 5.0 | Outside (0.0, 2.0) |
| `sigma1` | -3.0 | 3.0 | Outside (-1.0, 1.0) |
| `sigma2` | -3.0 | 3.0 | Outside (-1.0, 1.0) |

> `sigma0 + sigma1·sin(x) + sigma2·cos(x)` must remain ≥ 0 for all x. At Stage B, the script checks `sigma0 - sqrt(sigma1² + sigma2²) >= 0` and rejects the combination with a hard error if violated.

### Group 3 — Intensity Envelope and Bias

| Parameter | Hard min | Hard max | Warning range |
|-----------|----------|----------|---------------|
| `snr_peak` | 1.0 | 500.0 | Outside (10.0, 200.0) |
| `I1` | -0.9 | 0.9 | Outside (-0.5, 0.5) |
| `I2` | -0.9 | 0.9 | Outside (-0.5, 0.5) |
| `B_dc` | 0 | 5000 | Outside (100, 1000) |

> `I(r) = I₀·(1 + I₁·(r/r_max) + I₂·(r/r_max)²)` must remain > 0 for all r ≤ r_max. At Stage B, the script evaluates `I(r_max)` and `I(r_max/2)` and rejects if either is ≤ 0.

### Group 4 — Source

| Parameter | Hard min | Hard max | Warning range |
|-----------|----------|----------|---------------|
| `rel_638` | 0.0 | 2.0 | Outside (0.3, 1.5) |

---

## 10. Python Module Structure

```
z03_synthetic_calibration_image_generator.py
│
├── IMPORTS      from m01_airy_forward_model import InstrumentParams, airy_modified
│                from m02_calibration_synthesis import radial_profile_to_image
│
├── CONSTANTS (top-level)   sigma_read, pix_m, cx, cy, r_max, R_bins,
│                           lam_640, lam_638, n_ref
│
├── def _validated_prompt(...)        validated interactive float prompt
├── def prompt_all_params()           returns SynthParams dataclass (11 fields)
│                                     includes Y/n confirmation step
├── def snr_to_ipeak(...)             SNR quadratic positive root (Section 5.3)
├── def check_psf_positive(...)       validates sigma0 ≥ sqrt(sigma1²+sigma2²)
├── def check_vignetting_positive(...)validates I(r) > 0 for r in [0, r_max]
├── def derive_secondary(params)      → DerivedParams (α, I0, FSR, F, N_R)
├── def build_instrument_params(...)  → InstrumentParams from SynthParams + derived
├── def synthesise_profile(...)       calls airy_modified() × 2, sums lines + B
├── def synthesise_image(...)         radial_profile_to_image() → float64 (nrows, ncols)
├── def add_noise(...)                → uint16 cal image
├── def synthesise_dark(...)          → uint16 dark image
├── def build_s19_metadata(...)       → dict
├── def embed_metadata(...)           → uint16 image (in-place)
├── def write_bin(...)
├── def write_truth_json(...)
├── def make_diagnostic_figure(...)
└── def main()
```

**Key design change:** The standalone `airy()` helper from the previous spec version is removed. All Airy evaluation is delegated to `m01.airy_modified()` to guarantee that the synthesis and inversion use identical physics.

---

## 11. Integration with Downstream Pipeline

### Ingestion by Z01 / Z02

Both output `.bin` files are indistinguishable in format from real WindCube images:

```python
cal  = np.fromfile("20260414T000000Z_cal_synth_z03.bin",  dtype='<u2').reshape(260, 276)
dark = np.fromfile("20260414T000000Z_dark_synth_z03.bin", dtype='<u2').reshape(260, 276)
```

S19 `parse_metadata()` decodes the header rows of each file exactly as for real data.

### Validation workflow

```
Z03 → cal.bin + dark.bin  (known d, f, α, R, σ₀, σ₁, σ₂, I₀, I₁, I₂, B)
              ↓
        Z01 or Z02  (dark-subtract → reduce → fit)
              ↓
    Recovered d, f, α, R, σ₀, σ₁, σ₂, I₀, I₁, I₂, B
              ↓
    Compare to _truth.json  ← quantitative validation of all 10 parameters
```

---

## 12. Test Verification

Implemented in `tests/test_z03.py`.

| Test ID | Description | Pass criterion |
|---------|-------------|---------------|
| `test_cal_output_shape` | Cal `.bin` loads to (260, 276) uint16 | Shape matches exactly |
| `test_dark_output_shape` | Dark `.bin` loads to (260, 276) uint16 | Shape matches exactly |
| `test_cal_metadata_round_trip` | S19 metadata recoverable from cal rows 0–3 | All written fields match on read-back |
| `test_dark_metadata_round_trip` | S19 metadata recoverable from dark rows 0–3 | `image_type="Dark"`, `shutter_status="Closed"` |
| `test_fringe_peak_location` | Peak of cal image at image centre | Intensity maximum within 1 pixel of `(cy, cx)` |
| `test_snr_achieved` | Peak SNR of noisy cal image ≈ requested SNR | Within ±20% of target |
| `test_rel_638_ratio` | Amplitude ratio of the two Ne line families | Within ±5% of requested `rel_638` |
| `test_dark_no_fringes` | Dark image has no fringe structure | Std dev ≈ sqrt(B_dc + sigma_read²); no periodic structure |
| `test_truth_json_complete` | All required keys present in sidecar | All 10 `user_params` keys + `derived_params` + `fixed_defaults` present |
| `test_default_params` | Script runs with all defaults | Completes without error; both `.bin` files and `_truth.json` written |
| `test_psf_broadening_effect` | Non-zero σ₀ produces broader fringes than σ₀ = 0 | FWHM of central fringe ≥ FWHM of zero-PSF reference |
| `test_vignetting_effect` | Non-zero I₁ produces asymmetric radial intensity | Mean intensity in outer half-annulus differs from inner half-annulus by ≥ 1% |

---

## 13. Known Limitations and Future Work

- **Fixed image centre:** `(cx, cy)` is fixed at the geometric centre. A future parameter could offset the centre to test S12's centre-finding robustness.
- **No EM gain model:** The noise model uses Poisson + Gaussian read noise. A more complete model including EMCCD excess noise factor (F_EM ≈ √2) may be added for higher realism.
- **Output path portability:** The default output path is Windows-specific. Override via `Z03_OUTPUT_DIR` for Linux/CI use (Section 7, Stage F).
- **r_max not prompted:** The maximum usable radius is fixed at 110 px (flight value). A future revision may expose it as a synthesis parameter.

---

## 14. Spec Roadmap Position

```
Z01  Validate calibration using real images         [written]
Z02  Validate calibration: ring analysis + Tolansky [written]
Z03  Synthetic cal + dark image generator           [implemented]
Z04  (future) Automated Z03→Z01/Z02 round-trip comparison
```

---

## Revision History

| Date | Change |
|------|--------|
| 2026-04-10 | Initial spec written |
| 2026-04-10 | Corrected SNR quadratic (Section 5.3); added `Z03_OUTPUT_DIR` env-var; confirmed 7 pytest cases passing |
| 2026-04-10 | Stage G updated: wavelength labels added to cal image title and parameters text box |
| 2026-04-13 | Removed radial profile and centre-slice plots from Stage G; replaced 2×2 figure with 1×2 cal+dark side-by-side; added synthetic dark image throughout; updated `_truth.json` schema; expanded test suite to 10 cases |
| 2026-04-14 | **Major revision — all 10 M05 inversion parameters now user-prompted.** Added Groups 2–3 prompts: R, σ₀, σ₁, σ₂ (new), I₁, I₂ (new), B_dc (promoted from fixed). Updated Stage A banner, Stage C synthesis to use `airy_modified()` + `radial_profile_to_image()` from M01/M02 (replacing standalone `airy()` helper). Updated Section 6 fixed defaults (removed R and B_dc). Added `r_max`, `R_bins`, `n_ref` to fixed defaults table. Updated Section 9 validation bounds for 6 new parameters; added PSF positivity and vignetting positivity checks in Stage B. Updated `_truth.json` schema to 11 user params. Updated R default from 0.82 → 0.53 (FlatSat-measured value). Removed "instrument defects not modelled" limitation (now resolved). Added 2 new tests (test_psf_broadening_effect, test_vignetting_effect). Updated module structure: removed `airy()`, added `build_instrument_params()`, `check_psf_positive()`, `check_vignetting_positive()`, `synthesise_profile()`. |

---

*End of Z03 Spec*
