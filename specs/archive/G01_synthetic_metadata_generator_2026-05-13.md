# G01 — Synthetic Metadata Generator Specification

**Spec ID:** G01  
**Spec file:** `specs/G01_synthetic_metadata_generator_2026-05-13.md`  
**Script:** `src/processing/GEN01_synthesize_mission_dataset_2026_05_13.py`  
**Previous script:** `validation/gen01_synthetic_metadata_generator_2026_04_16.py`  
**Project:** WindCube FPI Science Operations Center Pipeline  
**Institution:** NCAR / High Altitude Observatory (HAO)  
**Status:** Authoritative — v13  
**Spec version:** 13  
**Date:** 2026-05-13  
**Git commit:** `fbea032`  

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
| 11 | 2026-05-13 | Correct Tolansky constants from real cal image analysis. Add Doppler shift to science pixel generator. Physical CCD97 noise model in science frames. Rename script to `GEN01_synthesize_mission_dataset_2026_05_13.py`. |
| 12 | 2026-05-13 | Separate sci/cal exposure time prompts (`exp_time_sci_s`, `exp_time_cal_s`). Asymmetric lat band (`lat_min_deg`, `lat_max_deg`). Unified dark model (`QDD_AT_20C` path). C16 rewritten. Removed `DARK_REF_ADU_S`, `T_REF_DARK_C`. Commit `46a44af`. |
| **13** | **2026-05-13** | **Doppler shift verified: r₁ shift direction and magnitude confirmed (3% of analytic at v=±2000 m/s). `TimeVaryingStormWindMap`: trapezoidal ap ramp, wind map option 6. `_plot_vrel_histogram()`: 2-panel (3-panel for option 6) figure. `ap_current` added to per-frame vrel_list CSV column. Commit `fbea032`.** |

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

## 2. User interface — interactive prompts (v12)

The v12 prompt sequence replaces three v9 prompts and adds two new ones.
All other prompts are unchanged from v9.

**Replaced — science band (single symmetric → two asymmetric):**

```
# OLD (v9–v11):
lat_band_deg  — symmetric band ±lat_band_deg, default ±40°

# NEW (v12):
lat_min_deg   — southern science band edge, default −40.0°, range −89 to +89
lat_max_deg   — northern science band edge, default +40.0°, range −89 to +89
               (must satisfy lat_max_deg > lat_min_deg)
```

This allows asymmetric configurations such as 40°–70°N for storm studies (SQ1)
or ±20° for DE3 tidal characterisation (SQ2) without wasting synthetic frames
on unobserved latitudes.

**Replaced — single exposure time → two separate:**

```
# OLD (v9–v11):
exp_time_cts  — single exposure time in timing-register counts

# NEW (v12):
exp_time_sci_s  — science frame exposure time, seconds, default 10.0, range 1–300
exp_time_cal_s  — calibration frame exposure time, seconds, default 120.0, range 10–600
```

Science and calibration exposures have different durations in CONOPS and drive
different noise budgets (dark current at 10 s vs. 120 s differs by 12×).
Both values are written to the README sidecar and the metadata CSV.

**Added in v13 — wind map option 6:**

```
[6] HWM14 storm with onset ramp
    Prompts:
      day_of_year   [default 355]
      ut_hours      [default   3.0]
      f107          [default 180.0]
      ap_quiet      [default   4.0]   background Kp-equivalent
      ap_peak       [default 150.0]   storm main-phase peak
      onset_hour    [default  12.0]   hours from campaign start
      ramp_up_h     [default   3.0]   hours quiet→peak
      ramp_down_h   [default   9.0]   hours at peak; symmetric recovery follows
```

When option 6 is selected, `ap_current` is written to the per-frame CSV column
for science frames. Non-option-6 wind maps write `NaN` to that column.

---

## 3. Wind map registry (v13)

| Key | Label | Builder | Notes |
|-----|-------|---------|-------|
| `"1"` | Uniform constant | `_build_uniform` | v_zonal, v_merid |
| `"2"` | Analytic sine_lat | `_build_analytic_sine` | A_zonal, A_merid |
| `"3"` | Analytic wave4/DE3 | `_build_analytic_wave4` | A_zonal, A_merid, phase_rad |
| `"4"` | HWM14 quiet-time | `_build_hwm14` | doy, ut, f107, ap |
| `"5"` | HWM14 storm/DWM07 | `_build_storm` | doy, ut, f107, ap (fixed) |
| `"6"` | HWM14 storm with onset ramp | `_build_storm_onset` | returns `TimeVaryingStormWindMap` |

### `TimeVaryingStormWindMap` (v13)

A dataclass that wraps `StormWindMap` with a trapezoidal ap time profile.
The main frame loop calls `wind_map.set_ap(wind_map.ap_at(t_elapsed_h))`
before each `compute_v_rel()` call when this wind map type is active.

```
ap(t):
  t < onset_hour                         → ap_quiet
  onset_hour ≤ t < onset + ramp_up_h     → linear ramp ap_quiet → ap_peak
  onset+ramp_up ≤ t < onset+ramp_up+ramp_down_h  → ap_peak
  t ≥ onset+ramp_up+ramp_down_h          → linear ramp ap_peak → ap_quiet
                                            (over ramp_down_h, then stays quiet)
```

Methods: `ap_at(t_hours_from_start)`, `set_ap(ap)`, `wind_components(lat, lon)`.

Options 1–5 return a standard `WindMap`; only option 6 returns a
`TimeVaryingStormWindMap`. The main loop checks `isinstance` to decide
whether to call `set_ap()`.

---

## 4. Orbit propagation and CONOPS scheduling

Unchanged from v9.

---

## 5. Main metadata loop (v13 additions)

Two additions to the per-frame loop:

**ap ramp update (option 6 only):** Before each `compute_v_rel()` call:
```python
if isinstance(wind_map, TimeVaryingStormWindMap):
    t_elapsed_h = (row.epoch.timestamp() -
                   df_sched.iloc[0].epoch.timestamp()) / 3600.0
    wind_map.set_ap(wind_map.ap_at(t_elapsed_h))
```

**ap_current in vrel_list:** Each frame's `vrel_list` entry includes:
```python
"ap_current": wind_map._current_ap if isinstance(wind_map, TimeVaryingStormWindMap)
              else float("nan")
```
This column appears in the output CSV for all frame types; NaN for non-option-6
wind maps.

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
# (Previous CAL_NE_RATIO=3.0 used opposite strong/weak convention and was a
# rough prior; 3.0 ≠ 1/0.344 = 2.91.)
REL_638          = 0.344

# ── CCD / pixel layout ──────────────────────────────────────────────────────
NX_PIX, NY_PIX   = 256, 256
N_ROWS_BIN       = 259
N_COLS_BIN       = 276
ROW_OFFSET_PIX   = 1
COL_OFFSET_PIX   = 10
ADU_MAX          = 16383             # 14-bit ceiling

# ── Electronic offset ────────────────────────────────────────────────────────
# Post-subtraction pedestal in real CCD97 images at −20°C. Bias and read noise
# both absorbed here; no separate Gaussian draw.
OFFSET_ADU       = 5

# ── CCD97 physical noise model (Teledyne e2v datasheet) ────────────────────
# OSH mode, 50 kHz CDS. Used by ALL three pixel generators (v12 unification).
# dark_rate(T) = QDD_AT_20C * GAIN_E_PER_ADU * 2^((T − 20) / T_DOUBLE_C)
# At T = −20°C: ≈ 400 × 1.0 × 2^(−40/6.5) ≈ 0.033 ADU/pix/s
GAIN_E_PER_ADU   = 1.0               # e-/ADU
READ_NOISE_E     = 2.2               # e- rms (informational; absorbed into OFFSET_ADU)
QDD_AT_20C       = 400.0             # dark current reference, e-/pix/s at +20°C
T_DOUBLE_C       = 6.5               # °C per doubling (CCD97 datasheet)

# ── Frame signal levels ──────────────────────────────────────────────────────
SCI_PEAK_ADU     = 5000              # OI 630 nm fringe peak above OFFSET_ADU
CAL_PEAK_ADU     = 12000             # Neon composite fringe peak above OFFSET_ADU

# Removed in v12: DARK_REF_ADU_S, T_REF_DARK_C (legacy dark path retired;
# all frames now use QDD_AT_20C / T_DOUBLE_C).
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
>
> **Constants removed in v12:**
> - `DARK_REF_ADU_S` — legacy dark reference rate; replaced by `QDD_AT_20C` path
> - `T_REF_DARK_C` — legacy dark reference temperature; replaced by +20°C CCD97 convention

### 7.2 Mixed-endian encoding helpers

Unchanged from v9.

### 7.3 Header encoder — `_encode_header()`

Unchanged from v9.

### 7.4 Pixel image generators (v12)

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
    exp_time_s  : float — science exposure duration, s  (exp_time_sci_s from prompt)
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
- `lambda_obs` depends on `v_rel_ms` — Doppler shift encoded in fringe (v11)
- `ccd_temp_c` and `exp_time_s` passed per-frame from dispatcher (v11)
- Dark current uses unified CCD97 path at +20°C reference (v12)
- Gaussian read noise draw removed (v11)

#### `_generate_cal_pixels(rng, nx, ny, plate_scale, cx, cy)`

```python
    I_cal = (_airy(LAMBDA_NE1_M) + REL_638 * _airy(LAMBDA_NE2_M)) / (1.0 + REL_638)
    signal = CAL_PEAK_ADU * I_cal
    photon = rng.poisson(np.clip(signal, 0, None))
    image  = np.round(photon + OFFSET_ADU).astype(np.float32)
    return np.clip(image, 0, ADU_MAX).astype(np.uint16)
```

Cal frames are photon-noise limited (per H02 §2.3). No dark current term
(120 s at −20°C adds ~4 ADU/px, negligible vs. CAL_PEAK_ADU = 12000).

#### `_generate_dark_pixels(ccd_temp1_c, exp_time_s, rng, nx, ny)`

```python
    # Unified CCD97 path (same formula as science frames, v12)
    dark_rate = QDD_AT_20C * GAIN_E_PER_ADU * 2.0**((ccd_temp1_c - 20.0) / T_DOUBLE_C)
    mean_dark = max(dark_rate * exp_time_s, 0.0)
    dark_arr  = rng.poisson(mean_dark, size=(ny, nx)).astype(float)
    image     = np.round(dark_arr + OFFSET_ADU).astype(np.float32)
    return np.clip(image, 0, ADU_MAX).astype(np.uint16)
```

v12 change: replaces the legacy `DARK_REF_ADU_S / T_REF_DARK_C` path with
the unified `QDD_AT_20C` CCD97 datasheet formula, identical to the science
frame generator. The 35% discrepancy noted in v11 §11 is eliminated.
Dark frames receive `exp_time_cal_s` (120 s default) from the dispatcher.

#### `_generate_pixels()` dispatcher (v12)

```python
    sci_exp_s = exp_time_sci_s   # from prompt (default 10 s)
    cal_exp_s = exp_time_cal_s   # from prompt (default 120 s)

    if frame_type == "science":
        return _generate_science_pixels(v_rel_ms, rng, nx, ny, plate_scale,
                                         cx, cy, ccd_temp1_c, sci_exp_s)
    elif frame_type == "cal":
        return _generate_cal_pixels(rng, nx, ny, plate_scale, cx, cy)
    elif frame_type == "dark":
        return _generate_dark_pixels(ccd_temp1_c, cal_exp_s, rng, nx, ny)
```

Science and dark/cal frames use their respective exposure times. Cal frames
use `cal_exp_s` implicitly via `CAL_PEAK_ADU` scaling (signal is fixed;
noise is Poisson of signal level regardless of `exp_time_cal_s`).

#### `_write_bin_file()` fill value

```python
    pixel_array = np.full((n_pixel_rows, n_cols), OFFSET_ADU, dtype=np.uint16)
```

Pixels outside the science window filled to `OFFSET_ADU = 5`.

### 7.5 Binary filename convention

Unchanged from v9.

---

## 8. Output files (v13)

Per-run outputs (all saved to user-selected output folder):

| File | Description |
|------|-------------|
| `{stem}.npy` | NumPy object array of `ImageMetadata` records |
| `{stem}.csv` | Full-schedule CSV including `ap_current` column (v13) |
| `{stem}.txt` | README sidecar with all prompt values |
| `{stem}_ground_tracks.png` | S/C ground tracks + tangent points (existing) |
| `{stem}_vrel_histogram.png` | v_rel distribution by look mode (v13, see §8.1) |
| `bin/{timestamp}_{type}.bin` | Per-frame binary FPI images |

### 8.1 v_rel histogram figure (`_plot_vrel_histogram`)

Two-panel figure (three panels for wind map option 6), figsize=(10, 4):

**Left panel — along-track frames** (odd orbits, LOS ∥ track):
- Histogram of `v_rel_ms`, bins=40, color `#0057C2`
- Vertical dashed line at mean `v_wind_LOS` (truth wind LOS projection), color `#b5651d`
- Vertical dotted line at 0 m/s
- Text annotation: mean ± std of `v_rel`

**Right panel — cross-track frames** (even orbits, LOS ⊥ track):
- Same layout, color `#003479`

**Third panel (option 6 only) — ap vs. time:**
- Step plot of ap index vs. hours from campaign start
- Phase-colored: quiet=`#aaaaaa`, ramp-up=`#e67e22`, peak=`#c0392b`, recovery=`#f39c12`

Supertitle: `f"GEN01 v_rel distribution — {windmap_label}  (seed={rng_seed})"`

---

## 9. Progress reporting and verification

### 9.1 Console output

Unchanged from v9.

### 9.2 Verification checks (C1–C24)

All checks carry forward from v12. Additions:

**C16** (science band): `lat_min_deg − 5° ≤ tp_lat ≤ lat_max_deg + 5°` (v12).

**C20** (pixel floor): minimum pixel value ≥ `OFFSET_ADU − 1 = 4`.

**C22 — Doppler round-trip (confirmed v13):**
Verified at v = ±2000 m/s: r₁(0) = 26 px, r₁(+2000) = 13 px, r₁(−2000) = 34 px.
Observed Δr² = 507 px² vs. analytic 524 px² (ratio 0.97 — integer-bin rounding).
Direction: recession → inward ✓, approach → outward ✓.

**C23 — Storm onset ramp (option 6 only):**
`ap_current` column non-NaN for science frames; values span [ap_quiet, ap_peak];
peak ap occurs at `onset_hour + ramp_up_h` ± one obs cadence.

**C24 — v_rel histogram figure:**
`{stem}_vrel_histogram.png` exists after run. Along-track frames show wide
distribution centered near ±7000 m/s (spacecraft LOS velocity dominant);
cross-track frames show narrower distribution centered near truth wind.

---

## 10. File location in repository

```
soc_sewell/
├── src/
│   └── processing/
│       └── GEN01_synthesize_mission_dataset_2026_05_13.py   ← authoritative (v13)
├── validation/
│   └── gen01_synthetic_metadata_generator_2026_04_16.py    ← v9 archive
└── specs/
    ├── G01_synthetic_metadata_generator_2026-05-13.md       ← this file (v13)
    └── archive/
        └── G01_synthetic_metadata_generator_2026-04-28.md   ← v10 archive
```

---

## 11. Constants cross-reference — Z03 and H02 alignment

This table is the authoritative cross-check. Any future edit to a shared
constant must update **all three specs** in the same commit.

| Constant | GEN01 v13 | Z03 symbol | H02 symbol | Value |
|----------|-----------|------------|------------|-------|
| Etalon gap | `ETALON_GAP_M` | `d_mm` default | `params.t` default | **20.106 mm** (20.10702 mm precise) |
| Plate scale (2×2) | `PLATE_SCALE_RPX` | `alpha` default | `params.alpha` default | **1.6083e-4 rad/px** |
| Effective reflectivity | `R_REFL` | `R` default | `params.R_refl` default | **0.725** |
| Finesse coeff F | `FINESSE_F` | `F_coef` (derived) | derived | **38.35** at R=0.725 |
| Reflective finesse N_R | (derived) | (derived) | (derived) | **9.73** at R=0.725 |
| 638/640 ratio | `REL_638` | `rel_638` default | `NE_INTENSITY_2` | **0.344** |
| Electronic offset | `OFFSET_ADU` | `OFFSET_ADU` | `params.B` | **5 ADU** |
| Dark doubling | `T_DOUBLE_C` | `T_DOUBLE_C` | — | 6.5°C |
| Ne 640.2 nm | `LAMBDA_NE1_M` | `LAM_640` | `NE_WAVELENGTH_1_AIR_M` | 640.2248e-9 m |
| Ne 638.3 nm | `LAMBDA_NE2_M` | `LAM_638` | `NE_WAVELENGTH_2_AIR_M` | 638.2991e-9 m |
| ADU ceiling | `ADU_MAX` | `16383` | — | 16383 |
| CCD97 dark ref | `QDD_AT_20C` | `Qdd_at_20C` (M03 v3) | — | 400.0 e-/px/s at +20°C |
| CCD97 gain | `GAIN_E_PER_ADU` | `gain_e_per_adu` | — | 1.0 e-/ADU |
| CCD97 read noise | `READ_NOISE_E` | `read_noise_e` | — | 2.2 e- rms |

> **Removed from GEN01 in v12** (were in Z03 cross-reference until now):
> `DARK_REF_ADU_S` and `T_REF_DARK_C` — legacy dark path retired. Z03 and H02
> were never using these; they existed only in GEN01's dark frame generator.

> **Note on FINESSE_F vs reflective finesse:**
> `FINESSE_F = 4R/(1−R)²` is the coefficient in the Airy denominator
> `1 / (1 + F·sin²(δ/2))`. At R=0.725, F ≈ 38.35.
> The **reflective finesse** `N_R = π√R/(1−R) ≈ 9.73` is what is most commonly
> cited in optics texts (fringe FWHM as a fraction of FSR). Related by `F = (2N_R/π)²`.
> Code comments should cite N_R ≈ 9.73 when describing fringe sharpness.

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
revision may add per-frame etalon thermal drift once the thermal coefficient
is measured from real images.

---

*End of G01 Spec v13 — 2026-05-13 — commit fbea032*
