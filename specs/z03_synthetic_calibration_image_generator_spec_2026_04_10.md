# Z03 — Synthetic Calibration Image Generator

**Spec ID:** Z03  
**Tier:** 9 — Validation Testing  
**Module:** `z03_synthetic_calibration_image_generator.py`  
**Status:** Draft — ready for implementation  
**Date:** 2026-04-10  
**Author:** Scott Sewell / HAO  
**Repo:** `soc_sewell`  
**Dependencies:** S03 (physical constants), S10 (M02, calibration fringe synthesis), S19 (P01, metadata schema)

---

## 1. Purpose

Z03 creates a **synthetic 'truth' calibration image** in the authentic WindCube `.bin` format (260 rows × 276 cols, uint16 little-endian) that can be ingested by S01, Z01, Z02, or any downstream pipeline module.

The script is **interactive**: it prompts the user for key instrumental and calibration parameters, synthesises a two-line neon Airy fringe pattern on a CCD-sized array, injects Poisson + read noise at a user-specified SNR, embeds a complete S19-compliant metadata structure into the first image rows, and writes the result as a `.bin` file whose name follows the WindCube ISO-8601 naming convention.

The generated image constitutes a **ground-truth artefact** — every parameter used in synthesis is known exactly and is written into the metadata header and a companion `_truth.json` sidecar, enabling downstream validation by comparison against pipeline-recovered values.

---

## 2. Relationship to Z01 and Z02

| Aspect | Z01 | Z02 | Z03 |
|--------|-----|-----|-----|
| Primary input | Real `.bin` + dark `.bin` | Real `.bin` or `.npy` | *(none — generates output)* |
| Interactivity | ROI, crop, dark selection | Dark selection, ROI | Parameter entry only |
| Output | Diagnostic figures + recovered d, f, α, ε | Diagnostic figures + residuals | `.bin` image + `_truth.json` sidecar |
| Core physics | Runs S12 + S13 (analysis) | Runs S12 + S13 (analysis) | Runs S10/M02 (synthesis) |
| Metadata | Reads S19 header | Reads S19 header | **Writes** S19 header |

Z03 is the **upstream complement** to Z01/Z02: it produces the controlled synthetic inputs that those scripts are designed to ingest and analyse.

---

## 3. Script Overview

```
z03_synthetic_calibration_image_generator.py
│
├── Stage A  — Banner and parameter prompting (interactive)
├── Stage B  — Derive secondary optical parameters
├── Stage C  — Synthesise two-line neon Airy fringe image
├── Stage D  — Apply noise model (Poisson + Gaussian read noise)
├── Stage E  — Build S19-compliant metadata and embed into image rows
├── Stage F  — Write .bin file + _truth.json sidecar
└── Stage G  — Diagnostic display (4-panel figure)
```

---

## 4. User-Prompted Parameters

The script opens a terminal session and prompts sequentially. Defaults are shown in brackets; pressing `<Enter>` accepts the default.

| Prompt | Variable | Default | Units | Notes |
|--------|----------|---------|-------|-------|
| Etalon gap `d` | `d_mm` | `20.106` | mm | Operational gap (Tolansky-derived) |
| Imaging lens focal length `f` | `f_mm` | `199.12` | mm | Lens effective focal length |
| Signal-to-noise ratio (peak) | `snr_peak` | `50.0` | — | Peak fringe counts / √peak counts |
| Relative intensity of 638 nm line | `rel_638` | `0.8` | — | I₆₃₈ / I₆₄₀ |

All other synthesis parameters use physically motivated defaults defined in Section 6.

### 4.1 Prompt implementation pattern

```python
def prompt_float(label: str, default: float, units: str = "") -> float:
    unit_str = f" [{units}]" if units else ""
    raw = input(f"  {label}{unit_str} (default {default}): ").strip()
    return float(raw) if raw else default
```

All prompts are wrapped in `try/except ValueError` with a re-prompt loop. Out-of-range values trigger a warning and re-prompt (ranges defined in Section 9).

---

## 5. Physics: Two-Line Neon Fringe Synthesis

### 5.1 Airy transmission function

The ideal on-axis Airy function for a single monochromatic wavelength λ at etalon gap *d* and plate reflectivity *R* is:

```
T(θ; λ, d, R) = 1 / [1 + F·sin²(δ/2)]
```

where:
- `δ = (4π/λ) · d · cos(θ)` — round-trip phase
- `F = 4R / (1 − R)²` — coefficient of finesse
- `θ(r) = arctan(α · r)` — angle from optical axis, with plate scale `α = 1 / f` (rad/pixel, to first order)
- `r` — radial distance from image centre in pixels

### 5.2 Two neon reference lines

| Line | Wavelength λ (nm) | Wavenumber (cm⁻¹) | Relative intensity |
|------|-------------------|--------------------|--------------------|
| Ne 640.2 nm | 640.2248 | 15615.211 | 1.0 (reference) |
| Ne 638.3 nm | 638.2991 | 15662.315 | `rel_638` (user-set) |

Source: Burns, Adams & Longwell (1950), IAU "S" standard.

The composite image is:

```
I_synth(x, y) = I_peak · [T(r; λ₆₄₀, d, R) + rel_638 · T(r; λ₆₃₈, d, R)] + B_dc
```

where:
- `r = sqrt((x - cx)² + (y - cy)²)` — pixel radius from detector centre `(cx, cy)`
- `I_peak` is derived from `snr_peak` (Section 5.3)
- `B_dc` is a DC background offset (default 500 ADU, see Section 6)

### 5.3 Deriving I_peak from SNR

For a photon-noise-limited detector:

```
snr_peak = I_peak / sqrt(I_peak + B_dc + sigma_read²)
```

Solving for `I_peak` (positive root of quadratic):

```python
# snr_peak² · I_peak  =  I_peak + B_dc + sigma_read²
# I_peak(snr² - 1) = B_dc + sigma_read²
# (only valid when snr_peak² >> 1, otherwise full quadratic)
a = 1.0 - snr_peak**2
b = snr_peak**2
c = snr_peak**2 * (B_dc + sigma_read**2)
# I_peak = (-b ± sqrt(b²-4ac)) / (2a)  [select positive root]
```

### 5.4 Plate scale α derivation

```
α [rad/pixel] = pixel_pitch [m] / f [m]
```

Default `pixel_pitch = 16.0 µm` (CCD97, 2×2 binned → 32 µm effective; see Section 6).

> **Note:** The CCD97 native pixel pitch is 16 µm. In 2×2 binning the effective pitch is 32 µm. Z03 uses 2×2 binned geometry (256 × 256 active pixels within the 260 × 276 array) by default.

---

## 6. Fixed Default Parameters

These are physics-motivated constants not prompted from the user. They are written into `_truth.json` for full traceability.

| Parameter | Symbol | Default Value | Source |
|-----------|--------|--------------|--------|
| Plate reflectivity | `R` | `0.82` | WindCube etalon spec (GNL4096R) |
| Read noise | `sigma_read` | `50.0` ADU | CCD97 EM gain regime estimate |
| DC background | `B_dc` | `500` ADU | Dark current + stray light estimate |
| Image centre (col) | `cx` | `137.5` | Geometric centre of 276-col array |
| Image centre (row) | `cy` | `129.5` | Geometric centre of 260-row active region |
| Pixel pitch (binned) | `pix_m` | `32.0e-6` m | CCD97 16 µm × 2×2 binning |
| Image dimensions | `nrows, ncols` | `260, 276` | WindCube Level-0 standard |
| Active region rows | `n_active_rows` | `256` | After 4 dark rows top + bottom |
| Metadata rows | `n_meta_rows` | `4` | S19 header occupies first 4 rows |
| Ne 640.2 nm wavelength | `lam_640` | `640.2248e-9` m | Burns et al. (1950) |
| Ne 638.3 nm wavelength | `lam_638` | `638.2991e-9` m | Burns et al. (1950) |
| Output dtype | — | `uint16` | 14-bit pixels, zero-padded to 16-bit |
| Byte order | — | little-endian | WindCube `.bin` standard |

---

## 7. Stage Descriptions

### Stage A — Banner and Parameter Prompting

```
╔══════════════════════════════════════════════════════════════╗
║  Z03  Synthetic Calibration Image Generator                  ║
║  WindCube SOC — soc_sewell                                   ║
╚══════════════════════════════════════════════════════════════╝

This script synthesises a two-line neon Airy fringe calibration
image in authentic WindCube .bin format, suitable for ingestion
by Z01, Z02, or any S01-based pipeline module.

You will be prompted for 4 parameters.  Press <Enter> to accept
the default shown in parentheses.

──────────────────────────────────────────────────────────────
 INSTRUMENTAL PARAMETERS
──────────────────────────────────────────────────────────────
  Etalon gap d [mm] (default 20.106):
  Imaging lens focal length f [mm] (default 199.12):

──────────────────────────────────────────────────────────────
 NOISE AND SOURCE PARAMETERS
──────────────────────────────────────────────────────────────
  Peak SNR (default 50.0):
  Relative intensity of 638 nm line vs 640 nm (default 0.8):
```

After all prompts, the script echoes a parameter summary table and asks the user to confirm before proceeding.

### Stage B — Derive Secondary Parameters

Computed (not prompted) from user inputs plus fixed defaults:

- `alpha_rad_per_px = pix_m / (f_mm * 1e-3)`  — plate scale
- `I_peak` — from SNR quadratic (Section 5.3)
- `FSR_mm = lam_640**2 / (2 * d_mm * 1e-3)` — free spectral range
- `finesse_F = 4*R / (1-R)**2`

These are printed to the terminal for the user's information.

### Stage C — Synthesise Fringe Image

1. Build pixel coordinate grids `X, Y` over `(nrows, ncols)`.
2. Compute `r = sqrt((X - cx)² + (Y - cy)²)` for all pixels.
3. Compute `theta = arctan(alpha_rad_per_px * r)`.
4. Evaluate `T_640 = airy(theta, lam_640, d_mm, R)` and `T_638 = airy(theta, lam_638, d_mm, R)`.
5. `I_float = I_peak * (T_640 + rel_638 * T_638) + B_dc`

The `airy()` helper:

```python
def airy(theta: np.ndarray, lam: float, d_mm: float, R: float) -> np.ndarray:
    """Ideal Airy transmission, returns values in [0, 1]."""
    d = d_mm * 1e-3
    delta = (4 * np.pi / lam) * d * np.cos(theta)
    F = 4 * R / (1 - R)**2
    return 1.0 / (1.0 + F * np.sin(delta / 2)**2)
```

> **Important:** The centre region corresponding to metadata rows (rows 0–3 inclusive) is synthesised normally at this stage; it will be overwritten in Stage E.

### Stage D — Apply Noise Model

```python
# Poisson noise on signal
signal_counts = np.random.poisson(np.clip(I_float, 0, None)).astype(np.float64)

# Gaussian read noise
read_noise = np.random.normal(0, sigma_read, size=signal_counts.shape)

# Combine and clip to valid ADU range [0, 16383] (14-bit)
image_noisy = np.clip(signal_counts + read_noise, 0, 16383).astype(np.uint16)
```

The numpy random seed is logged in `_truth.json` (auto-generated via `np.random.SeedSequence`) to allow reproducibility.

### Stage E — Build and Embed S19 Metadata

S19 defines a metadata structure embedded in the **first `n_meta_rows` rows** (rows 0–3) of the image array. Each row is 276 uint16 pixels = 552 bytes.

#### 5.1 Metadata field mapping

Fields explicitly set from user input or synthesis parameters:

| S19 Field | Source | Value |
|-----------|--------|-------|
| `image_type` | Fixed | `"Cal"` |
| `n_rows` | Fixed | `260` |
| `n_cols` | Fixed | `276` |
| `binning` | Fixed | `2` |
| `shutter_status` | Fixed | `"Open"` |
| `cal_lamp_1` | Fixed | `"C1_On"` |
| `cal_lamp_2` | Fixed | `"C2_Off"` |
| `cal_lamp_3` | Fixed | `"C3_Off"` |
| `date_utc` | System clock | `yyyymmdd` |
| `time_utc` | System clock | `hhmmss` |
| `exposure_ms` | Fixed default | `120000` ms (120 s) |
| `etalon_temp_1` | Fixed default | `24.00` °C |
| All other fields | Default | `0` |

All fields not listed above are set to their S19-defined zero/null defaults.

#### 5.2 Serialisation into pixel rows

The metadata is serialised as a UTF-8 JSON string, zero-padded to fill exactly `n_meta_rows × ncols × 2` bytes, then reinterpreted as uint16 little-endian values and written into `image[0:n_meta_rows, :]`.

```python
import json, datetime
meta = build_s19_metadata(user_params, fixed_defaults)
meta_json = json.dumps(meta, separators=(',', ':'))
meta_bytes = meta_json.encode('utf-8')
# Pad to exact size
target_bytes = n_meta_rows * ncols * 2
meta_padded = meta_bytes.ljust(target_bytes, b'\x00')
meta_uint16 = np.frombuffer(meta_padded, dtype='<u2').reshape(n_meta_rows, ncols)
image[0:n_meta_rows, :] = meta_uint16
```

> **Consistency note:** This is the exact inverse operation of the metadata-read procedure used in Z01 and Z02, ensuring round-trip fidelity.

### Stage F — Write .bin File and _truth.json Sidecar

#### Output filename

```
yyyymmddThhmmssZ_cal_synth_z03.bin
yyyymmddThhmmssZ_cal_synth_z03_truth.json
```

The `_synth_z03` infix distinguishes synthetic files from real flight data.

#### .bin write

```python
image.astype('<u2').tofile(output_path_bin)
```

#### _truth.json sidecar

Contains all synthesis parameters (both user-entered and fixed defaults), derived secondary parameters, random seed, and synthesis timestamp. This is the ground-truth record for downstream validation.

```json
{
  "z03_version": "1.0",
  "timestamp_utc": "2026-04-10T18:30:00Z",
  "random_seed": 12345678,
  "user_params": {
    "d_mm": 20.106,
    "f_mm": 199.12,
    "snr_peak": 50.0,
    "rel_638": 0.8
  },
  "derived_params": {
    "alpha_rad_per_px": 1.6071e-4,
    "I_peak_adu": 2312.5,
    "FSR_mm": 1.017e-5,
    "finesse_coefficient_F": 90.3
  },
  "fixed_defaults": {
    "R": 0.82,
    "sigma_read": 50.0,
    "B_dc": 500,
    "cx": 137.5,
    "cy": 129.5,
    "pix_m": 3.2e-5,
    "nrows": 260,
    "ncols": 276,
    "lam_640_m": 6.402248e-7,
    "lam_638_m": 6.382991e-7
  },
  "output_file": "20260410T183000Z_cal_synth_z03.bin"
}
```

### Stage G — Diagnostic Display

A 4-panel matplotlib figure is shown after writing the files:

| Panel | Content |
|-------|---------|
| **Top-left** | Full synthetic image (imshow, grey colormap, log-scaled) with title showing key parameters |
| **Top-right** | Azimuthally averaged radial profile I(r²) — two-line composite, showing both Ne line families labelled |
| **Bottom-left** | Noise-free model vs noisy realisation overlay at a 1-pixel-wide horizontal slice through centre |
| **Bottom-right** | Text summary box: all input parameters, derived α, I_peak, FSR, output filename |

The figure is saved as `{stem}_diagnostic.png` alongside the `.bin` file.

---

## 8. Output Files Summary

| File | Description |
|------|-------------|
| `yyyymmddThhmmssZ_cal_synth_z03.bin` | Synthetic calibration image in WindCube Level-0 format |
| `yyyymmddThhmmssZ_cal_synth_z03_truth.json` | Ground-truth sidecar with all synthesis parameters |
| `yyyymmddThhmmssZ_cal_synth_z03_diagnostic.png` | 4-panel diagnostic figure |

---

## 9. Parameter Validation and Bounds

| Parameter | Hard min | Hard max | Warning range |
|-----------|----------|----------|---------------|
| `d_mm` | 15.0 | 25.0 | Outside (19.0, 21.5) |
| `f_mm` | 100.0 | 300.0 | Outside (180.0, 220.0) |
| `snr_peak` | 1.0 | 500.0 | Outside (10.0, 200.0) |
| `rel_638` | 0.0 | 2.0 | Outside (0.3, 1.5) |

If any value falls outside the warning range, the script prints a yellow warning but proceeds after the user confirms. Values outside the hard limits raise `ValueError` and force re-entry.

---

## 10. Python Module Structure

```
z03_synthetic_calibration_image_generator.py
│
├── CONSTANTS (top-level)   R, sigma_read, B_dc, pix_m, cx, cy, lam_640, lam_638
│
├── def prompt_float(...)   validated interactive float prompt
├── def prompt_all_params() returns dataclass SynthParams
├── def airy(...)           Airy transmission function (vectorised numpy)
├── def snr_to_ipeak(...)   quadratic solve for I_peak from SNR
├── def synthesise_image(params) → np.ndarray (float64, noise-free)
├── def add_noise(image, params, seed) → np.ndarray (uint16)
├── def build_s19_metadata(params) → dict
├── def embed_metadata(image, meta_dict) → np.ndarray (uint16, in-place)
├── def write_bin(image, path)
├── def write_truth_json(params, derived, path)
├── def make_diagnostic_figure(image_noisy, image_noisefree, params, derived, paths)
└── def main()
```

---

## 11. Integration with Downstream Pipeline

### Ingestion by Z01 / Z02

The output `.bin` file is indistinguishable in format from a real WindCube calibration image. Z01 and Z02 can load it directly using their standard `load_bin()` routines:

```python
image = np.fromfile("20260410T183000Z_cal_synth_z03.bin", dtype='<u2').reshape(260, 276)
```

The metadata rows are decoded by S19's `parse_metadata()` function exactly as for real data.

### Validation workflow

```
Z03 → .bin (known d, f, α, rel_638)
         ↓
       Z01 or Z02
         ↓
  Recovered d, f, α, ε₁, ε₂
         ↓
  Compare to _truth.json  ← quantitative validation
```

The `_truth.json` sidecar provides the ground truth for this comparison. A future Z04 spec may automate this comparison loop.

### Ingestion by S01

S01 reads `.bin` files according to the WindCube Level-0 data standard. The `_synth_z03` filename infix does not affect S01's file parser (which keys on the `_cal` suffix).

---

## 12. Test Verification

The following pytest cases shall be implemented in `tests/test_z03.py`:

| Test ID | Description | Pass criterion |
|---------|-------------|---------------|
| `test_output_shape` | `.bin` file loads to (260, 276) uint16 | Shape matches exactly |
| `test_metadata_round_trip` | S19 metadata recoverable from first 4 rows | All written fields match on read-back |
| `test_fringe_peak_location` | Peak of radial profile at r²=0 | Intensity maximum at centre within 1 pixel |
| `test_snr_achieved` | Peak SNR of noisy image ≈ requested SNR | Within ±20% of target |
| `test_rel_638_ratio` | Amplitude ratio of the two line families | Within ±5% of requested `rel_638` |
| `test_truth_json_complete` | All required keys present in sidecar | All keys from Section 8 present |
| `test_default_params` | Script runs with all defaults (no user input) | Completes without error, output files written |

---

## 13. Known Limitations and Future Work

- **Instrument defects not modelled:** The synthesised image uses an ideal Airy function. Etalon flatness defects, vignetting, and lens aberrations (σ₀, σ₁, σ₂ broadening terms from M05) are not included in the initial version. A `--include-defects` flag may be added in a future revision.
- **Single dark frame not generated:** Z03 does not synthesise a companion dark frame. The user should run the dark generator separately (future Z04) or use a real dark.
- **Fixed image centre:** `(cx, cy)` is fixed at the geometric centre. A future parameter could offset the centre to test S12's centre-finding robustness.
- **No EM gain model:** The noise model uses Poisson + Gaussian read noise. A more complete model including EMCCD excess noise factor (F_EM ≈ √2) may be added for higher realism.

---

## 14. Spec Roadmap Position

This spec sits in **Tier 9 — Validation Testing**, alongside Z01 and Z02.

```
Z01  Validate calibration using real images        [written]
Z02  Validate calibration: ring analysis + Tolansky [written]
Z03  Synthetic calibration image generator          [this spec]
Z04  (future) Automated Z03→Z01/Z02 round-trip comparison
```

Z03 is the first Z-series spec that *generates* rather than *analyses*, completing the synthetic ↔ real validation loop.

---

*End of Z03 Spec*
