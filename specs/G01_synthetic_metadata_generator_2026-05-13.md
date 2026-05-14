# G01 — Synthetic Metadata Generator Specification

**Spec ID:** G01  
**Spec file:** `docs/specs/G01_synthetic_metadata_generator_2026-05-13.md`  
**Script:** `src/processing/GEN01_synthesize_mission_dataset_2026_05_13.py`  
**Previous script:** `validation/gen01_synthetic_metadata_generator_2026_04_16.py`  
**Project:** WindCube FPI Science Operations Center Pipeline  
**Institution:** NCAR / High Altitude Observatory (HAO)  
**Status:** Authoritative — v11  
**Spec version:** 11  
**Date:** 2026-05-13  

**Depends on:**
- NB01 (`nb01_orbit_propagator_2026_04_16.py`) — `propagate_orbit(t_start, duration_s, dt_s)`
- NB02a (`nb02a_boresight_2026_04_16.py`) — `compute_los_eci(...)`
- NB02b (`nb02b_tangent_point_2026_04_16.py`) — `compute_tangent_point(...)`
- NB02c (`nb02c_los_projection_2026_04_16.py`) — `compute_v_rel(...)`
- P01 (`p01_image_metadata_2026_04_06.py`) — `ImageMetadata`, `AdcsQualityFlags`
- `windcube/constants.py` — authoritative numerical constants
- `tkinter` (stdlib) — native folder-browser dialog

**Used by:**
- Z02 (synthetic airglow image generator)
- Z03 (synthetic neon calibration image generator)
- Future dark frame synthesis
- Round-trip pipeline validation (GEN01 → M03 → M05 → M06/M07 → compare to truth CSV)

**Key references:**
- Tolansky two-line Benoit analysis of real WindCube cal images (S13 / H05)
  — `2_cal_120sexp_swapped_raw_L0_peak_fits_r2_tolansky_2line.png`
  — N_Δ = −189, ε_a − ε_b = −0.500127, d = 20.10702 ± 0.0002 mm
- Teledyne e2v CCD97 datasheet — dark current, gain, read noise
- Burns, Adams & Longwell (1950) IAU neon spectroscopic standards
- CONOPS WC-SE-0003 V8

---

## Revision history

| Version | Date | Summary |
|---------|------|---------|
| 1–9 | 2026-04-16 | See archive. Incremental development through binary synthesis. |
| 10 | 2026-04-28 | Physics alignment with Z03 v1.5. **Introduced regression:** `ETALON_GAP_M = 20.0005e-3` (ICOS spacer value, not Benoit result). `PLATE_SCALE_RPX = 1.6000e-4` (rounded, not Tolansky). Read noise removed, `OFFSET_ADU=5` added, `REL_638=0.344` added. |
| **11** | **2026-05-13** | **Correct Tolansky constants from real cal image analysis. Add Doppler shift to science pixel generator. Physical CCD97 noise model in science frames. Rename script to `GEN01_synthesize_mission_dataset_2026_05_13.py`.** |

---

## 1. Purpose

GEN01 is a standalone interactive Python script that pre-computes and saves
the complete AOCS/instrument metadata array **and** a corresponding set of
synthetic binary FPI image files for a WindCube observation campaign.

| Frame type | NB01 | NB02a/b/c | Image synthesis |
|------------|------|-----------|-----------------|
| science | ✓ orbit state | ✓ full LOS decomposition | Doppler-shifted Airy fringe at `v_rel_ms`; Poisson + dark noise |
| cal | ✓ orbit state | — | Two-line neon (λ₁, λ₂) composite; Poisson noise only |
| dark | ✓ orbit state | — | Temperature-dependent dark current + offset |

**Storage:** ~120,000 `.bin` files × 143,520 bytes ≈ 17.3 GB for a 30-day
run at 10 s cadence.

---

## 2. User interface — interactive prompts

Unchanged from v9. See `G01_synthetic_metadata_generator_2026-04-28.md` §2.

---

## 3. Wind map registry

Unchanged from v9.

---

## 4. Orbit propagation and CONOPS scheduling

Unchanged from v9.

---

## 5. Main metadata loop

Unchanged from v9. Note: `ccd_temp1` (drawn per-frame from
`Normal(CCD_TEMP_MEAN_C, CCD_TEMP_STD_C)`) is now passed through to
`_generate_science_pixels()` so the dark current in science frames varies
realistically with the per-frame simulated focal-plane temperature.

---

## 6. `ImageMetadata` field assignment

Unchanged from v9.

---

## 7. Binary image synthesis

### 7.1 Constants (v11 — authoritative)

```python
# ── FPI optical model — authoritative values from Tolansky two-line analysis ──

LAMBDA_OI_M      = 630.0e-9          # OI 630.0 nm source wavelength, m
LAMBDA_NE1_M     = 640.2248e-9       # Neon strong line (Burns et al. 1950 IAU)
LAMBDA_NE2_M     = 638.2991e-9       # Neon weak line  (Burns et al. 1950 IAU)

# Etalon gap — Tolansky two-line Benoit recovery from real WindCube cal images
# d = (N_Δ + ε_a − ε_b)·λ_a·λ_b / [2·n·(λ_b − λ_a)]
# N_Δ = −189, ε_a − ε_b = −0.500127
# Result: d = 20.10702 ± 0.0002 mm  (2σ: ±0.0004 mm)
# 20.106e-3 is the 6-significant-figure shorthand; use 20.10702e-3 for
# highest-precision work (e.g. M05 forward model priors).
# NOTE: 20.0005e-3 in G01 v10 was the ICOS spacer (pre-assembly gap) and
# was erroneous as a Benoit label — corrected here.
ETALON_GAP_M     = 20.106e-3

# Plate scale — Tolansky two-line result
# α = √(λ_a · n_air / (d · Δ_a))
# Δ_a = 1230.8 px²/fr,  result: α_a = 1.608313e-4 rad/px
# Consistency between two lines: 94.7 ppm [PASS]
PLATE_SCALE_RPX  = 1.6083e-4         # rad/px, 2×2 binned

# Effective reflectivity — chosen to match observed fringe sharpness in real images.
# FlatSat coating measurement gave R=0.53 (finesse ≈ 4.9), but in-flight fringes
# are sharper. R=0.725 gives reflective finesse N_R = π√R/(1−R) ≈ 9.73,
# which matches observed peak sharpness. PSF broadening is absorbed into R_eff.
R_REFL           = 0.725

N_GAP            = 1.0               # Refractive index of etalon gap (air/vacuum)
C_LIGHT_MS       = 2.99792458e8      # Speed of light, m/s

# Airy denominator coefficient: F = 4R/(1−R)²
# At R=0.725: F ≈ 38.35   (reflective finesse N_R = π√R/(1−R) ≈ 9.73)
FINESSE_F        = 4 * R_REFL / (1 - R_REFL) ** 2

# 638/640 neon intensity ratio — measured from real WindCube calibration images
# Convention: weak(638nm) / strong(640nm).  REL_638 = 0.344.
# (Previous CAL_NE_RATIO=3.0 used the opposite strong/weak convention and was
# a rough prior, not a measurement; 3.0 ≠ 1/0.344 = 2.91.)
REL_638          = 0.344

# ── CCD / pixel layout ──────────────────────────────────────────────────────
NX_PIX, NY_PIX   = 256, 256
N_ROWS_BIN       = 259
N_COLS_BIN       = 276
ROW_OFFSET_PIX   = 1
COL_OFFSET_PIX   = 10
ADU_MAX          = 16383             # 14-bit ceiling

# ── Electronic offset ────────────────────────────────────────────────────────
# Post-subtraction pedestal in real CCD97 images at −20°C operating point.
# Bias (deterministic DC) and read noise (stochastic rms) are both small and
# effectively constant; bundled into a single additive constant.
# Read noise is NOT modelled as a separate Gaussian draw — it is absorbed here.
OFFSET_ADU       = 5

# ── CCD97 physical noise model (Teledyne e2v datasheet) ────────────────────
# These values apply to the OSH (output signal high-gain) mode at 50 kHz CDS.
# They are used in _generate_science_pixels() to compute per-frame dark current.
GAIN_E_PER_ADU   = 1.0               # e-/ADU
READ_NOISE_E     = 2.2               # e- rms (informational; absorbed into OFFSET_ADU)
QDD_AT_20C       = 400.0            # dark current reference: e-/pix/s at +20°C
# Dark current doubling interval and reference temperature
T_DOUBLE_C       = 6.5               # °C per doubling (CCD97 datasheet)
# dark_rate(T) = QDD_AT_20C * GAIN_E_PER_ADU * 2^((T − 20) / T_DOUBLE_C)
# At T = −20°C:  rate ≈ 400 × 1.0 × 2^(−40/6.5) ≈ 0.033 e-/pix/s ≈ 0.033 ADU/pix/s

# ── Frame signal levels ──────────────────────────────────────────────────────
SCI_PEAK_ADU     = 5000              # OI 630 nm fringe peak above OFFSET_ADU
CAL_PEAK_ADU     = 12000             # Neon composite fringe peak above OFFSET_ADU

# Legacy dark reference constants (used by _generate_dark_pixels).
# Kept for backward compatibility; physically redundant with QDD_AT_20C above
# but in different units (ADU/pix/s vs e-/pix/s; equal because GAIN=1.0).
DARK_REF_ADU_S   = 0.05             # ADU/pix/s at T_REF_DARK_C
T_REF_DARK_C     = -20.0            # °C reference for dark frame generator
```

> **Constants removed in v11 (relative to v9):**
> - `FOCAL_LENGTH_M` — never used in pixel synthesis
> - `BIAS_ADU = 100` → `OFFSET_ADU = 5`
> - `SCI_READ_NOISE`, `CAL_READ_NOISE`, `DARK_READ_NOISE` — Gaussian reads removed
> - `CAL_NE_RATIO = 3.0` → `REL_638 = 0.344`
>
> **Constants corrected from v10:**
> - `ETALON_GAP_M`: `20.0005e-3` → `20.106e-3` (restores correct Tolansky Benoit value)
> - `PLATE_SCALE_RPX`: `1.6000e-4` → `1.6083e-4` (from Tolansky two-line α)

### 7.2 Mixed-endian encoding helpers

Unchanged from v9.

### 7.3 Header encoder — `_encode_header()`

Unchanged from v9.

### 7.4 Pixel image generators (v11)

#### `_generate_science_pixels(v_rel_ms, rng, nx, ny, plate_scale, cx, cy, ccd_temp_c, exp_time_s)`

```python
def _generate_science_pixels(v_rel_ms, rng, nx, ny, plate_scale,
                              cx, cy, ccd_temp_c, exp_time_s):
    """
    Generate OI 630 nm Airy fringe image with Doppler shift v_rel_ms.

    Noise model: Poisson photon noise + temperature-dependent dark current
    (CCD97 physical model).  No Gaussian read noise draw (absorbed into
    OFFSET_ADU = 5).

    Parameters
    ----------
    v_rel_ms    : float — LOS velocity, m/s (positive = recession = inward fringe shift)
    ccd_temp_c  : float — focal-plane temperature, °C (per-frame from metadata draw)
    exp_time_s  : float — exposure duration, s
    """
    lambda_obs = LAMBDA_OI_M * (1.0 + v_rel_ms / C_LIGHT_MS)

    x = np.arange(nx) - cx
    y = np.arange(ny) - cy
    XX, YY = np.meshgrid(x, y)
    r_px   = np.sqrt(XX**2 + YY**2)
    theta  = r_px * plate_scale
    delta  = 4.0 * np.pi * N_GAP * ETALON_GAP_M * np.cos(theta) / lambda_obs
    I_airy = 1.0 / (1.0 + FINESSE_F * np.sin(delta / 2.0)**2)

    signal = SCI_PEAK_ADU * I_airy

    # Physical dark current: CCD97 formula, reference at +20°C
    dark_rate = QDD_AT_20C * GAIN_E_PER_ADU * 2.0**((ccd_temp_c - 20.0) / T_DOUBLE_C)
    dark_mean = max(dark_rate * exp_time_s, 0.0)

    photon = rng.poisson(np.clip(signal, 0, None))
    dark   = rng.poisson(np.full(signal.shape, dark_mean))
    image  = np.round(photon + dark + OFFSET_ADU).astype(np.float32)
    return np.clip(image, 0, ADU_MAX).astype(np.uint16)
```

Key changes from v9:
- `lambda_obs` now depends on `v_rel_ms` — Doppler shift is encoded in the fringe
- `ccd_temp_c` and `exp_time_s` passed in from dispatcher (per-frame values)
- Dark current uses CCD97 formula referenced to `+20°C` and `T_DOUBLE_C = 6.5°C`
- Gaussian read noise draw removed

#### `_generate_cal_pixels(rng, nx, ny, plate_scale, cx, cy)`

```python
    I_cal = (_airy(LAMBDA_NE1_M) + REL_638 * _airy(LAMBDA_NE2_M)) / (1.0 + REL_638)
    signal = CAL_PEAK_ADU * I_cal
    photon = rng.poisson(np.clip(signal, 0, None))
    image  = np.round(photon + OFFSET_ADU).astype(np.float32)
    return np.clip(image, 0, ADU_MAX).astype(np.uint16)
```

Cal frames are photon-noise limited (per H02 §2.3). No dark current term
(120 s exposure at −20°C adds ~4 ADU/px, negligible vs. CAL_PEAK_ADU = 12000).

#### `_generate_dark_pixels(ccd_temp1_c, exp_time_s, rng, nx, ny)`

```python
    dark_rate = DARK_REF_ADU_S * 2.0**((ccd_temp1_c - T_REF_DARK_C) / T_DOUBLE_C)
    mean_dark = max(dark_rate * exp_time_s, 0.0)
    dark_arr  = rng.poisson(mean_dark, size=(ny, nx)).astype(float)
    image     = np.round(dark_arr + OFFSET_ADU).astype(np.float32)
    return np.clip(image, 0, ADU_MAX).astype(np.uint16)
```

Gaussian read noise draw removed. `DARK_REF_ADU_S` reference convention
(−20°C) is numerically consistent with `QDD_AT_20C` convention (+20°C)
because both use `T_DOUBLE_C = 6.5°C`:
`0.05 ADU/px/s ≈ 400 × 1.0 × 2^(−40/6.5) ≈ 0.033` — close but not exact
due to independent origin of the two constants. For dark frames, the legacy
`DARK_REF_ADU_S` path is used; science frames use the CCD97 path.
A future cleanup may unify these to a single constant.

#### `_generate_pixels()` dispatcher

```python
    if frame_type == "science":
        return _generate_science_pixels(v_rel_ms, rng, nx, ny, plate_scale,
                                         cx, cy, ccd_temp1_c, exp_time_s)
    elif frame_type == "cal":
        return _generate_cal_pixels(rng, nx, ny, plate_scale, cx, cy)
    elif frame_type == "dark":
        return _generate_dark_pixels(ccd_temp1_c, exp_time_s, rng, nx, ny)
```

`exp_time_s = exp_time_cts * TIMER_PERIOD_S` (computed before the frame loop).

#### `_write_bin_file()` fill value

```python
    pixel_array = np.full((n_pixel_rows, n_cols), OFFSET_ADU, dtype=np.uint16)
```

Pixels outside the science window are filled to `OFFSET_ADU = 5`, not `BIAS_ADU = 100`.

### 7.5 Binary filename convention

Unchanged from v9.

---

## 8. Output files

Unchanged from v9. README `.txt` sidecar, `.npy`, `.csv`, ground track `.png`,
and per-frame `.bin` files.

---

## 9. Progress reporting and verification

### 9.1 Console output

Unchanged from v9.

### 9.2 Verification checks (C1–C21)

All checks carry forward from v9 with one update:

**C20** (pixel floor): minimum pixel value in any frame must be ≥ `OFFSET_ADU - 1 = 4`.
(Was ≥ 4 with `BIAS_ADU = 100` previously; same numerical floor, different source.)

**New check C22 — Doppler round-trip (recommended manual check):**
For a science frame with `v_rel_ms` stored in the truth CSV, extract the
binary image, run M03 annular reduction, locate the first Airy peak radius r₁,
and verify that `r₁²` shifts in the direction expected for the signed velocity
(positive v_rel → inward shift → smaller r₁). Quantitative check: Δr² per
100 m/s ≈ `Δ_a / FSR × λ / c × 100 ≈ 1230 × 3.3e-7 ≈ 4×10⁻⁴ px²/fr per m/s`.
At `v_rel = 100 m/s`, expected Δr₁² ≈ 0.04 px² — detectable if r₁ ≈ 35 px.

---

## 10. File location in repository

```
soc_sewell/
├── src/
│   └── processing/
│       └── GEN01_synthesize_mission_dataset_2026_05_13.py   ← NEW location
├── validation/
│   └── gen01_synthetic_metadata_generator_2026_04_16.py    ← v9 (keep for reference)
└── docs/specs/
    ├── G01_synthetic_metadata_generator_2026-05-13.md       ← this file (v11)
    └── archive/
        └── G01_synthetic_metadata_generator_2026-04-28.md   ← v10 (archived)
```

---

## 11. Constants cross-reference — Z03 and H02 alignment

This table is the authoritative cross-check. Any future edit to a shared
constant must update **all three specs** in the same commit.

| Constant | GEN01 v11 | Z03 symbol | H02 symbol | Value |
|----------|-----------|------------|------------|-------|
| Etalon gap | `ETALON_GAP_M` | `d_mm` default | `params.t` default | **20.106 mm** (20.10702 mm precise) |
| Plate scale (2×2) | `PLATE_SCALE_RPX` | `alpha` default | `params.alpha` default | **1.6083e-4 rad/px** |
| Effective reflectivity | `R_REFL` | `R` default | `params.R_refl` default | **0.725** |
| Finesse coeff F | `FINESSE_F` | `F_coef` (derived) | derived | **38.35** at R=0.725 |
| Reflective finesse N_R | (derived) | (derived) | (derived) | **9.73** at R=0.725 |
| 638/640 ratio | `REL_638` | `rel_638` default | `NE_INTENSITY_2` | **0.344** |
| Electronic offset | `OFFSET_ADU` | `OFFSET_ADU` | `params.B` | **5 ADU** |
| Dark ref rate | `DARK_REF_ADU_S` | `DARK_REF_ADU_S` | — | 0.05 ADU/px/s |
| Dark ref temp | `T_REF_DARK_C` | `T_REF_DARK_C` | — | −20.0°C |
| Dark doubling | `T_DOUBLE_C` | `T_DOUBLE_C` | — | 6.5°C |
| Ne 640.2 nm | `LAMBDA_NE1_M` | `LAM_640` | `NE_WAVELENGTH_1_AIR_M` | 640.2248e-9 m |
| Ne 638.3 nm | `LAMBDA_NE2_M` | `LAM_638` | `NE_WAVELENGTH_2_AIR_M` | 638.2991e-9 m |
| ADU ceiling | `ADU_MAX` | `16383` | — | 16383 |
| CCD97 dark ref | `QDD_AT_20C` | `Qdd_at_20C` (M03 v3) | — | 400.0 e-/px/s |
| CCD97 gain | `GAIN_E_PER_ADU` | `gain_e_per_adu` | — | 1.0 e-/ADU |
| CCD97 read noise | `READ_NOISE_E` | `read_noise_e` | — | 2.2 e- rms |

> **Note on FINESSE_F vs reflective finesse:**
> `FINESSE_F = 4R/(1−R)²` is the coefficient in the Airy denominator
> `1 / (1 + F·sin²(δ/2))`. At R=0.725, F ≈ 38.35.
> The **reflective finesse** is `N_R = π√R/(1−R) ≈ 9.73` — this is the
> number most commonly cited (fringe width as a fraction of FSR).
> The two quantities are related by `F = (2·N_R/π)²`.
> Comments in code should cite N_R ≈ 9.73, not F ≈ 38.35, when describing
> fringe sharpness to avoid confusion.

> **Known inconsistency to resolve (not a bug):** `DARK_REF_ADU_S = 0.05` at
> −20°C and `QDD_AT_20C = 400.0` at +20°C are from different measurement
> sources. Converting QDD_AT_20C to −20°C:
> 400 × 2^(−40/6.5) ≈ 0.033 ADU/px/s — ≈35% lower than 0.05.
> Dark frames use the legacy path (0.05 at −20°C); science frames use the
> CCD97 datasheet path (400 at +20°C). A future spec revision should unify
> these once on-orbit dark measurements are available.

---

## 12. On-orbit etalon gap temperature correction (forward note)

On orbit, four etalon temperature sensors provide a correction to the etalon
gap via the coefficient of thermal expansion of the fused silica spacer
(α_fs ≈ 0.55 ppm/°C). The operational pipeline (M05/M06) will apply:

```
d_eff(T) = d_Tolansky × (1 + α_fs × (T_etalon − T_lab))
```

where `T_lab` is the lab temperature at which the Tolansky analysis was
performed and `T_etalon` is the on-orbit telemetry value. GEN01 does not
simulate this correction — it uses fixed `ETALON_GAP_M`. A future GEN01
revision may add etalon thermal drift to the science pixel generator once
the thermal coefficient is measured.

---

## 13. Instructions for Claude Code — next revision

The next GEN01 revision (v12) should address:

1. **FINESSE_F comment fix** (minor): update the inline comment from "≈26.5"
   to "≈38.35 (F coefficient); reflective finesse N_R ≈ 9.73" — one line.

2. **Separate science / cal exposure times**: split the single `exp_time_cts`
   prompt into `exp_time_sci_s` (default 10 s) and `exp_time_cal_s`
   (default 120 s). Pass the appropriate value to each pixel generator.
   This affects the dark current in science frames and the cal frame signal level.

3. **Geospatial sampling region**: replace the symmetric `lat_band_deg` prompt
   with `lat_min_deg` (default −40°) and `lat_max_deg` (default +40°) to
   allow asymmetric science band selection.

4. **Dark constant unification**: replace `DARK_REF_ADU_S / T_REF_DARK_C`
   with the CCD97 path (`QDD_AT_20C`) in `_generate_dark_pixels()` to
   eliminate the inconsistency noted in §11.

---

*End of G01 Spec v11 — 2026-05-13*
