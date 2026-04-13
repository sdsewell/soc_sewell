# Z03 — Synthetic Calibration Image Generator

**Spec ID:** Z03  
**Tier:** 9 — Validation Testing  
**Module:** `z03_synthetic_calibration_image_generator.py`  
**Status:** Implemented — updated spec  
**Date:** 2026-04-13  
**Author:** Scott Sewell / HAO  
**Repo:** `soc_sewell`  
**Dependencies:** S03 (physical constants), S10 (M02, calibration fringe synthesis), S19 (P01, metadata schema)

---

## 1. Purpose

Z03 creates a **synthetic 'truth' calibration image pair** — a neon fringe calibration image and a companion dark image — both in authentic WindCube `.bin` format (260 rows × 276 cols, uint16 little-endian), suitable for direct ingestion by S01, Z01, Z02, or any downstream pipeline module.

The script is **interactive**: it prompts the user for key instrumental and calibration parameters, synthesises a two-line neon Airy fringe pattern on a CCD-sized array, injects Poisson + read noise at a user-specified SNR, embeds complete S19-compliant metadata structures into the first image rows of both output files, and writes the results as `.bin` files whose names follow the WindCube ISO-8601 naming convention.

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
├── Stage A  — Banner and parameter prompting (interactive)
├── Stage B  — Derive secondary optical parameters
├── Stage C  — Synthesise two-line neon Airy fringe image (noise-free float)
├── Stage D  — Apply noise model → calibration image (uint16)
│             Synthesise dark image (noise-only, uint16)
├── Stage E  — Build S19-compliant metadata; embed into cal image rows 0–3
│             Build S19-compliant metadata; embed into dark image rows 0–3
├── Stage F  — Write cal .bin + dark .bin + _truth.json sidecar
└── Stage G  — Diagnostic display (2-panel figure)
```

---

## 4. User-Prompted Parameters

The script opens a terminal session and prompts sequentially. Defaults are shown; pressing `<Enter>` accepts the default.

| Prompt | Variable | Default | Units | Notes |
|--------|----------|---------|-------|-------|
| Etalon gap `d` | `d_mm` | `20.106` | mm | Operational gap (Tolansky-derived) |
| Imaging lens focal length `f` | `f_mm` | `199.12` | mm | Lens effective focal length |
| Signal-to-noise ratio (peak) | `snr_peak` | `50.0` | — | Peak fringe counts / √peak counts |
| Relative intensity of 638 nm line | `rel_638` | `0.8` | — | I₆₃₈ / I₆₄₀ |

All other synthesis parameters use physically motivated defaults defined in Section 6.

### 4.1 Prompt implementation pattern

```python
def _validated_prompt(label: str, default: float, units: str,
                      hard_min: float, hard_max: float,
                      warn_min: float, warn_max: float) -> float:
    ...
```

All prompts are wrapped in `try/except ValueError` with a re-prompt loop. Values outside hard limits are rejected immediately and force re-entry. Values inside hard limits but outside the warning range print an advisory and proceed after the Y/n confirmation step.

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
- `θ(r) = arctan(α · r)` — angle from optical axis, with plate scale `α = pixel_pitch / f` (rad/pixel)
- `r` — radial distance from image centre in pixels

### 5.2 Two neon reference lines

| Line | Wavelength λ (nm) | Wavenumber (cm⁻¹) | Relative intensity |
|------|-------------------|--------------------|--------------------|
| Ne 640.2 nm | 640.2248 | 15615.211 | 1.0 (reference) |
| Ne 638.3 nm | 638.2991 | 15662.315 | `rel_638` (user-set) |

Source: Burns, Adams & Longwell (1950), IAU "S" standard.  
Wavelengths are imported from `src.constants.NE_WAVELENGTH_1_M` / `NE_WAVELENGTH_2_M` at runtime, with fallback to module-level literals.

The composite noise-free signal image is:

```
I_synth(x, y) = I_peak · [T(r; λ₆₄₀, d, R) + rel_638 · T(r; λ₆₃₈, d, R)] + B_dc
```

where:
- `r = sqrt((x − cx)² + (y − cy)²)` — pixel radius from detector centre `(cx, cy)`
- `I_peak` is derived from `snr_peak` (Section 5.3)
- `B_dc` is a DC background offset (default 500 ADU, see Section 6)

### 5.3 Deriving I_peak from SNR

Peak SNR is defined as:

```
SNR = I_peak / sqrt(I_peak + B_dc + sigma_read²)
```

Rearranging gives the general quadratic in I_peak:

```
I_peak² - SNR²·I_peak - SNR²·(B_dc + sigma_read²) = 0
```

The correct positive root (valid for all SNR ≥ 0, including SNR ≤ 1) is:

```python
def snr_to_ipeak(snr: float, B_dc: float, sigma_read: float) -> float:
    noise_floor = B_dc + sigma_read**2
    return (snr**2 + math.sqrt(snr**4 + 4 * snr**2 * noise_floor)) / 2
```

> **Note:** The simplified approximation `I_peak ≈ SNR²·(B_dc + σ_read²) / (SNR² - 1)` is only valid for SNR >> 1 and must not be used in code — it diverges at SNR = 1 and is negative for SNR < 1.

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

The same noise model as Stage D is applied: Poisson draws on `B_dc` (representing dark current and stray photons) plus Gaussian read noise, using an independent RNG draw from the same seed sequence. The dark image therefore represents a physically realistic matched dark frame for use in the S12 dark-subtraction step.

```
dark_float = B_dc · ones(nrows, ncols)
dark_noisy = clip(Poisson(dark_float) + Gaussian(0, sigma_read), 0, 16383).astype(uint16)
```

---

## 6. Fixed Default Parameters

These are physics-motivated constants not prompted from the user. All are written into `_truth.json` for full traceability. Wavelength constants are imported from `src.constants` at runtime.

| Parameter | Symbol | Default Value | Source |
|-----------|--------|--------------|--------|
| Plate reflectivity | `R` | `0.82` | WindCube etalon spec (GNL4096R) |
| Read noise | `sigma_read` | `50.0` ADU | CCD97 EM gain regime estimate |
| DC background | `B_dc` | `500` ADU | Dark current + stray light estimate |
| Image centre (col) | `cx` | `137.5` | Geometric centre of 276-col array |
| Image centre (row) | `cy` | `129.5` | Geometric centre of 260-row active region |
| Pixel pitch (binned) | `pix_m` | `32.0e-6` m | CCD97 16 µm × 2×2 binning |
| Image dimensions | `nrows, ncols` | `260, 276` | WindCube Level-0 standard |
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

This script synthesises a matched calibration + dark image pair
in authentic WindCube .bin format, suitable for ingestion by
Z01, Z02, or any S01-based pipeline module.

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

After all prompts, the script echoes a parameter summary table and asks the user to confirm (Y/n) before proceeding.

### Stage B — Derive Secondary Parameters

Computed (not prompted) from user inputs plus fixed defaults:

- `alpha_rad_per_px = pix_m / (f_mm * 1e-3)` — plate scale
- `I_peak` — from SNR quadratic (Section 5.3, positive root)
- `FSR_mm = lam_640**2 / (2 * d_mm * 1e-3)` — free spectral range
- `finesse_F = 4*R / (1-R)**2`

These are printed to the terminal for the user's information.

### Stage C — Synthesise Fringe Image (noise-free float)

1. Build pixel coordinate grids `X, Y` over `(nrows, ncols)` via `np.meshgrid`.
2. Compute `r = sqrt((X − cx)² + (Y − cy)²)` for all pixels.
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

**Dark image** — noise only, no fringe signal, drawn from the same RNG stream (ensuring statistical independence from the cal draw):

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

The timestamp in the dark filename matches the cal filename (same synthesis session). The `_synth_z03` infix distinguishes synthetic files from real flight data while remaining parseable by S01 (which keys on `_cal` / `_dark` suffixes).

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
  "z03_version": "1.1",
  "timestamp_utc": "2026-04-13T00:00:00Z",
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
  "output_cal_file":  "20260413T000000Z_cal_synth_z03.bin",
  "output_dark_file": "20260413T000000Z_dark_synth_z03.bin"
}
```

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
| **Left** | Synthetic calibration image (imshow, grey colormap, log-scaled). Title line 1: key parameters (d, f, SNR, R). Title line 2: `Ne lines: 640.2248 nm (×1.0)  638.2991 nm (×{rel_638})` |
| **Right** | Synthetic dark image (imshow, grey colormap, linear scale). Title: `Dark image — B_dc={B_dc} ADU, σ_read={sigma_read} ADU` |

No radial profiles, centre slices, or parameter text boxes are included. The images speak for themselves as ground-truth artefacts; full parameter records are in `_truth.json`.

---

## 8. Output Files Summary

| File | Description |
|------|-------------|
| `yyyymmddThhmmssZ_cal_synth_z03.bin` | Synthetic calibration image — fringe signal + noise, S19 metadata in rows 0–3 |
| `yyyymmddThhmmssZ_dark_synth_z03.bin` | Synthetic dark image — noise background only, S19 metadata in rows 0–3 |
| `yyyymmddThhmmssZ_cal_synth_z03_truth.json` | Ground-truth sidecar with all synthesis parameters, seed, and both output filenames |
| `yyyymmddThhmmssZ_cal_synth_z03_diagnostic.png` | 1×2 side-by-side cal + dark display figure |

---

## 9. Parameter Validation and Bounds

| Parameter | Hard min | Hard max | Warning range |
|-----------|----------|----------|---------------|
| `d_mm` | 15.0 | 25.0 | Outside (19.0, 21.5) |
| `f_mm` | 100.0 | 300.0 | Outside (180.0, 220.0) |
| `snr_peak` | 1.0 | 500.0 | Outside (10.0, 200.0) |
| `rel_638` | 0.0 | 2.0 | Outside (0.3, 1.5) |

---

## 10. Python Module Structure

```
z03_synthetic_calibration_image_generator.py
│
├── CONSTANTS (top-level)   R, sigma_read, B_dc, pix_m, cx, cy, lam_640, lam_638
│
├── def _validated_prompt(...)       validated interactive float prompt
├── def prompt_all_params()          returns SynthParams dataclass; Y/n confirmation
├── def airy(...)                    ideal Airy transmission (vectorised numpy)
├── def snr_to_ipeak(...)            SNR quadratic positive root (Section 5.3)
├── def derive_secondary(params)     → DerivedParams (α, I_peak, FSR, F)
├── def synthesise_image(params, derived) → np.ndarray float64 (noise-free cal)
├── def add_noise(I_float, rng, sigma_read) → np.ndarray uint16 (cal image)
├── def synthesise_dark(rng, B_dc, sigma_read, nrows, ncols) → np.ndarray uint16
├── def build_s19_metadata(image_type, params) → dict
├── def embed_metadata(image, meta_dict) → np.ndarray uint16 (in-place)
├── def write_bin(image, path)
├── def write_truth_json(params, derived, seed, path_cal, path_dark, path_truth)
├── def make_diagnostic_figure(image_cal, image_dark, params, derived, path_png)
└── def main()
```

---

## 11. Integration with Downstream Pipeline

### Ingestion by Z01 / Z02

Both output `.bin` files are indistinguishable in format from real WindCube images:

```python
cal  = np.fromfile("20260413T000000Z_cal_synth_z03.bin",  dtype='<u2').reshape(260, 276)
dark = np.fromfile("20260413T000000Z_dark_synth_z03.bin", dtype='<u2').reshape(260, 276)
```

S19 `parse_metadata()` decodes the header rows of each file exactly as for real data.

### Validation workflow

```
Z03 → cal.bin + dark.bin  (known d, f, α, rel_638)
              ↓
        Z01 or Z02  (dark-subtract → reduce → fit)
              ↓
    Recovered d, f, α, ε₁, ε₂
              ↓
    Compare to _truth.json  ← quantitative validation
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
| `test_dark_no_fringes` | Dark image has no fringe structure | Std dev of dark ≈ sqrt(B_dc + sigma_read²); no periodic structure |
| `test_truth_json_complete` | All required keys present in sidecar | Includes `output_cal_file` and `output_dark_file` |
| `test_default_params` | Script runs with all defaults | Completes without error; both `.bin` files and `_truth.json` written |

---

## 13. Known Limitations and Future Work

- **Instrument defects not modelled:** The synthesised cal image uses an ideal Airy function. Etalon flatness defects, vignetting, and lens aberrations (σ₀, σ₁, σ₂ broadening terms from M05) are not included. A `--include-defects` flag may be added in a future revision.
- **Fixed image centre:** `(cx, cy)` is fixed at the geometric centre. A future parameter could offset the centre to test S12's centre-finding robustness.
- **No EM gain model:** The noise model uses Poisson + Gaussian read noise. A more complete model including EMCCD excess noise factor (F_EM ≈ √2) may be added for higher realism.
- **Output path portability:** The default output path is Windows-specific. Override via `Z03_OUTPUT_DIR` for Linux/CI use (Section 7, Stage F).

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
| 2026-04-13 | Removed radial profile and centre-slice plots from Stage G; replaced 2×2 figure with 1×2 cal+dark side-by-side; added synthetic dark image throughout (Sections 1, 3, 5.5, 7/Stage D, 7/Stage E §7.2, 7/Stage F, 8, 10, 12, 13); updated `_truth.json` schema to include `output_dark_file`; expanded test suite to 10 cases; removed "No companion dark frame" limitation note |

---

*End of Z03 Spec*
