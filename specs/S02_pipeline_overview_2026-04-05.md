# S02 — Pipeline Overview and Data Flow

**Spec ID:** S02
**Spec file:** `specs/S02_pipeline_overview_2026-04-05.md`
**Project:** WindCube FPI Science Operations Center Pipeline
**Institution:** NCAR / High Altitude Observatory (HAO)
**Status:** Authoritative
**Depends on:** S01
**Used by:** S05, S06, S07, S08, S09, S10, S11, S12, S13, S14, S15, S16, S17, S18, S19
**Last updated:** 2026-04-05
**Created/Modified by:** Claude AI

---

## 1. Purpose

This document describes the complete WindCube FPI data reduction pipeline: every
processing level, every module, every interface contract, and the data types
that flow between modules. It is the authoritative reference for the overall
architecture. Any collaborator writing a spec for a specific module must first
read this document to understand where their module sits in the chain and what
data it receives and produces.

S02 does not define algorithms. It defines structure. Algorithm decisions are in
the module specs (S05–S17).

---

## 2. Overview: from raw image to wind vector

WindCube acquires two types of CCD fringe images on every orbit:
- **Calibration images** — produced with the onboard neon lamp; a two-wavelength
  (640.2248 nm, 638.2991 nm) fringe pattern used to characterise the etalon
- **Science images** — produced with the shutter open; the OI 630.0304 nm
  airglow fringe pattern from the thermosphere, Doppler-shifted by wind

The pipeline converts these raw images into geolocated horizontal wind vectors
through five processing levels and a supporting geometry chain.

---

## 3. Processing levels

| Level | Name | Content | Produced by |
|-------|------|---------|-------------|
| L0 | Raw image | CCD counts (uint16) + metadata header | Spacecraft downlink |
| L1a | Dark-subtracted image | float64 ADU, overscan removed | M02/M04 synthesis; real: dark subtraction step |
| L1b | FringeProfile | 1D radial profile in r² bins with σ per bin | M03 (annular reduction) |
| L1c | AirglowFitResult | λ_c (m) with σ; χ²_reduced | M06 (airglow inversion) |
| L2 | WindResult | v_zonal, v_meridional (m/s) with σ; geolocation | M07 (wind retrieval) |

**Important:** Science images are L0 raw fringe patterns. They are not wind maps.
Wind retrieval requires L0 → L1a → L1b → L1c → L2. Do not skip or conflate levels.

---

## 4. Module inventory

The pipeline is organised into two chains: a **geometry/simulation chain** (NB-series)
and an **instrument processing chain** (M-series). The NB chain runs first and
produces the synthetic inputs used to validate the M chain.

### 4.1 Geometry and simulation chain (NB-series)

| Module | Spec | Tier | Input | Output | Description |
|--------|------|------|-------|--------|-------------|
| NB00 | S05 | 1 | — | WindField T1–T4 | Synthetic thermospheric wind map |
| NB01 | S06 | 1 | TLE or Keplerian elements | OrbitState (ECI pos/vel per epoch) | SGP4 orbit propagator |
| NB02 | S07 | 1 | OrbitState, WindField | LOSGeometry, v_rel_ms | Boresight, tangent point, LOS projection |
| INT01 | S08 | 1 | NB00–NB02 | Geometry validation plots | Integration notebook |

### 4.2 Instrument chain (M-series)

| Module | Spec | Tier | Input | Output | Description |
|--------|------|------|-------|--------|-------------|
| M01 | S09 | 2 | InstrumentParams, λ, r | Ã(r; λ) profile values | Modified Airy forward model |
| M02 | S10 | 2 | InstrumentParams | CalibrationImage dict | Synthetic neon calibration image |
| M04 | S11 | 2 | InstrumentParams, v_rel_ms | AirglowImage dict | Synthetic OI 630 nm science image |
| M03 | S12 | 3 | 2D image (L1a) | FringeProfile (L1b) | Fringe centre finding + annular r² reduction |
| M05 | S13 | 4 | FringeProfile (cal) | CalibrationResult | Staged calibration inversion; recovers etalon params |
| M06 | S14 | 5 | FringeProfile (sci), CalibrationResult | AirglowFitResult (L1c) | Airglow fringe inversion; recovers λ_c |
| M07 | S15 | 6 | AirglowFitResult, LOSGeometry | WindResult (L2) | WLS vector wind retrieval |

### 4.3 Integration and data product modules

| Module | Spec | Tier | Description |
|--------|------|------|-------------|
| INT02 | S16 | 7 | FPI instrument chain validation notebook (M01→M07) |
| INT03 | S17 | 7 | Full end-to-end pipeline notebook (NB00→M07→L2) |
| P01 | S18 | 8 | Image sidecar JSON schema (metadata per L0 frame) |
| L2 product | S19 | 8 | NetCDF output schema for L2 wind vectors |

---

## 5. Data flow diagram

```
  ┌─────────────────────────────────────────────────────────────────┐
  │  GEOMETRY / SIMULATION CHAIN                                    │
  │                                                                 │
  │  NB00            NB01              NB02                         │
  │  (WindField)───► (OrbitState) ───► (LOSGeometry, v_rel_ms) ──┐ │
  │   T1–T4          SGP4/J2            boresight · tangent pt    │ │
  └──────────────────────────────────────────────────────────────│─┘
                                                                  │ v_rel_ms
                                                                  ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  FORWARD MODEL (SYNTHESIS)                                      │
  │                                                                 │
  │  M01 (Airy model) ◄──────────────── InstrumentParams           │
  │       │                                                         │
  │       ├──► M02 ──► CalibrationImage (L0 neon, uint16)          │
  │       │                                                         │
  │       └──► M04 ──► AirglowImage (L0 science, float64) ◄────────┘
  └─────────────────────────────────────────────────────────────────┘
                  │                         │
                  │ L0 cal image            │ L0 sci image
                  ▼                         ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  ANNULAR REDUCTION                                              │
  │                                                                 │
  │  M03 (cal)  ──► FringeProfile_cal (L1b)                        │
  │  M03 (sci)  ──► FringeProfile_sci (L1b)                        │
  └─────────────────────────────────────────────────────────────────┘
                  │                         │
        FringeProfile_cal          FringeProfile_sci
                  │                         │
                  ▼                         │
  ┌──────────────────────┐                  │
  │  M05 CALIBRATION     │                  │
  │  INVERSION           │                  │
  │  CalibrationResult   │──────────────────┤
  │  (etalon params)     │                  │
  └──────────────────────┘                  │
                                            ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  M06 AIRGLOW INVERSION                                          │
  │  Inputs: FringeProfile_sci + CalibrationResult                  │
  │  Output: AirglowFitResult → λ_c (m) + σ_λ_c (L1c)            │
  └─────────────────────────────────────────────────────────────────┘
                  │
                  │ AirglowFitResult (L1c)
                  │ + LOSGeometry (from NB02)
                  ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  M07 WIND RETRIEVAL                                             │
  │  λ_c → v_rel → WLS over spatial bins → v_zonal, v_meridional   │
  │  Output: WindResult (L2), geolocated                            │
  └─────────────────────────────────────────────────────────────────┘
```

---

## 6. Interface contracts

These are the fixed data types that cross module boundaries. Every module spec
must use exactly these field names and types. Changes require a revision to this
spec and to all affected downstream module specs.

### 6.1 InstrumentParams (M01 → M02, M04, M05, M06)

Defined in S09 (M01). Contains:

| Field | Type | Units | Description |
|-------|------|-------|-------------|
| `t_m` | float | m | Etalon gap. Nominal: 20.008e-3 m |
| `R_refl` | float | — | Effective etalon reflectivity. Nominal: 0.53 (FlatSat) |
| `alpha` | float | rad/px | Magnification constant = pixel_pitch / focal_length |
| `n` | float | — | Refractive index of gap medium (1.0 for vacuum) |
| `r_max` | float | px | Maximum usable radius (half image width for square arrays) |
| `I0` | float | ADU | Intensity envelope peak value |
| `I1` | float | — | Envelope linear coefficient |
| `I2` | float | — | Envelope quadratic coefficient |
| `sigma0` | float | px | PSF width base value |
| `sigma1` | float | px | PSF width sine coefficient |
| `sigma2` | float | px | PSF width cosine coefficient |
| `B` | float | ADU | CCD bias offset |

### 6.2 FringeProfile (M03 → M05, M06) — L1b

Defined in S12 (M03). Key fields:

| Field | Type | Shape | Units | Description |
|-------|------|-------|-------|-------------|
| `profile` | ndarray | (n_bins,) | ADU | Mean intensity per r² bin |
| `sigma_profile` | ndarray | (n_bins,) | ADU | 1σ SEM per bin |
| `two_sigma_profile` | ndarray | (n_bins,) | ADU | Exactly 2 × sigma_profile |
| `r_grid` | ndarray | (n_bins,) | px | Bin centre radii |
| `r2_grid` | ndarray | (n_bins,) | px² | Bin centre r² values |
| `n_pixels` | ndarray | (n_bins,) | — | Pixel count per bin |
| `masked` | ndarray | (n_bins,) | — | True = exclude from fitting |
| `cx`, `cy` | float | — | px | Fringe centre |
| `sigma_cx`, `sigma_cy` | float | — | px | 1σ centre uncertainty |
| `two_sigma_cx`, `two_sigma_cy` | float | — | px | Exactly 2 × sigma |
| `quality_flags` | int | — | — | Bitmask (S04 / S12) |

### 6.3 CalibrationResult (M05 → M06)

Defined in S13 (M05). Contains the fitted InstrumentParams with σ and 2σ
for each parameter, plus:

| Field | Type | Description |
|-------|------|-------------|
| `t_m`, `sigma_t_m`, `two_sigma_t_m` | float | Etalon gap (m) with uncertainties |
| `R_refl`, `sigma_R_refl`, `two_sigma_R_refl` | float | Reflectivity with uncertainties |
| `alpha`, `sigma_alpha`, `two_sigma_alpha` | float | Magnification with uncertainties |
| `I0`, `sigma_I0`, `two_sigma_I0` | float | Envelope peak with uncertainties |
| `I1`, `sigma_I1`, `two_sigma_I1` | float | Envelope linear coefficient |
| `I2`, `sigma_I2`, `two_sigma_I2` | float | Envelope quadratic coefficient |
| `sigma0`, `sigma_sigma0`, `two_sigma_sigma0` | float | PSF base width |
| `sigma1`, `sigma_sigma1`, `two_sigma_sigma1` | float | PSF sine coefficient |
| `sigma2`, `sigma_sigma2`, `two_sigma_sigma2` | float | PSF cosine coefficient |
| `B`, `sigma_B`, `two_sigma_B` | float | Bias offset |
| `chi2_reduced` | float | Reduced χ² of calibration fit |
| `quality_flags` | int | Bitmask (S04 / S13) |

### 6.4 AirglowFitResult (M06 → M07) — L1c

Defined in S14 (M06).

| Field | Type | Units | Description |
|-------|------|-------|-------------|
| `lambda_c_m` | float | m | Fitted line centre wavelength |
| `sigma_lambda_c_m` | float | m | 1σ uncertainty |
| `two_sigma_lambda_c_m` | float | m | Exactly 2 × sigma |
| `v_rel_ms` | float | m/s | LOS wind velocity (Doppler) |
| `sigma_v_rel_ms` | float | m/s | 1σ uncertainty |
| `two_sigma_v_rel_ms` | float | m/s | Exactly 2 × sigma |
| `chi2_reduced` | float | — | Reduced χ² of airglow fit |
| `quality_flags` | int | — | Bitmask (S04 / S14) |

### 6.5 LOSGeometry (NB02 → M07)

Defined in S07 (NB02).

| Field | Type | Units | Description |
|-------|------|-------|-------------|
| `los_eci` | ndarray | — | Unit LOS vector in ECI J2000 |
| `tangent_lat_deg` | float | deg | Geodetic latitude of tangent point |
| `tangent_lon_deg` | float | deg | Geodetic longitude of tangent point |
| `tangent_alt_km` | float | km | Altitude of tangent point above WGS84 |
| `v_rel_ms` | float | m/s | Projected LOS velocity (spacecraft + Earth rotation + wind) |
| `obs_mode` | str | — | `'along_track'` or `'cross_track'` |
| `epoch_utc` | float | s | Unix timestamp of observation |

### 6.6 WindResult (M07 output) — L2

Defined in S15 (M07).

| Field | Type | Units | Description |
|-------|------|-------|-------------|
| `v_zonal_ms` | float | m/s | Zonal wind component (positive = eastward) |
| `sigma_v_zonal_ms` | float | m/s | 1σ uncertainty |
| `two_sigma_v_zonal_ms` | float | m/s | Exactly 2 × sigma |
| `v_meridional_ms` | float | m/s | Meridional wind component (positive = northward) |
| `sigma_v_meridional_ms` | float | m/s | 1σ uncertainty |
| `two_sigma_v_meridional_ms` | float | m/s | Exactly 2 × sigma |
| `lat_deg`, `lon_deg` | float | deg | Geolocation of retrieval cell centre |
| `alt_km` | float | km | Altitude of retrieval cell (nominally 250 km) |
| `epoch_utc` | float | s | Unix timestamp |
| `n_obs_along`, `n_obs_cross` | int | — | Contributing observations per component |
| `quality_flags` | int | — | Bitmask (S04 / S15) |

---

## 7. Key design decisions (do not reopen)

The following decisions are final and must not be contradicted by any module spec:

**OI 630 nm source model is a spectral delta function.** Temperature retrieval
is not a WindCube science goal. M04 and M06 use a delta function source spectrum.
Do not add thermal broadening.

**OI rest wavelength is 630.0304 nm (air), not 630.0 nm.** All Doppler shift
calculations use this value. See S03.

**Etalon gap is 20.008 mm (ICOS measured).** The FlatSat value of 20.670 mm
is wrong (FSR-period ambiguity). See S13.

**Depression angle is 15.73°.** Derived from arccos(6621/6881) at 510 km
spacecraft altitude, 250 km tangent height. Earlier values of 23.4° are
superseded.

**Binning is 2×2 in nominal science mode.** CCD97 pixels are 16 µm native.
After 2×2 binning: 32 µm effective pixel, 256×256 active image.

**Uncertainty convention: σ and 2σ together for all fitted parameters.** Defined
in S04 and required in every module that outputs fitted quantities.

**Wind sign convention:** zonal positive eastward, meridional positive northward.
Reference frame: ENU at the tangent point. See S07.

---

## 8. Physical constants

All physical constants used by any module are defined in S03
(`src/constants.py`). No module may hardcode a numerical value for a physical
constant. Import from `src.constants` using the canonical symbol name.

See S03 for the full table of values and sources.

---

## 9. Uncertainty conventions

All uncertainty reporting follows the conventions in S04. In brief:
- Every fitted scalar must have both `sigma_X` and `two_sigma_X` companions
- `two_sigma_X` is always exactly `2.0 * sigma_X` (never independently computed)
- Reduced χ² is `chi2 / (n_points - n_free_params)`; acceptable range 0.5–3.0
- Quality flags use the bitmask convention defined in S04

---

## 10. Validation benchmarks

| What to validate | Source | Notes |
|-----------------|--------|-------|
| Orbit geometry | Astropy + SGP4 | Compare tangent points against ephemeris |
| Wind climatology | HWM14 (hwm14 Python package) | Used as T1 truth wind field |
| Wind observations | ICON/MIGHTI L3 wind product | NASA Earthdata; 2020–present |
| Airglow emission | Meneses et al. 2008 | Height profiles uploaded to project KB |
| Thermospheric wind model | TIE-GCM output | Public access via CEDAR archive |

For operational cross-validation of retrieved L2 winds, the primary external
benchmark is the ICON/MIGHTI L3 horizontal wind product. TIE-GCM provides
the physics-based reference for storm-time behaviour.

---

## 11. Repository layout (from S01)

```
soc_sewell/
├── specs/                ← S01–S19 spec files
├── src/
│   ├── constants.py      ← S03 — all physical constants
│   ├── fpi/              ← M01, M02, M03, M04, M05, M06, M07
│   └── geometry/         ← NB00, NB01, NB02
├── tests/                ← test_m01_*.py … test_m07_*.py
├── notebooks/            ← INT01, INT02, INT03
├── validation/           ← external data comparison scripts
└── data/
    ├── reference/        ← HWM14, ICON/MIGHTI, TIE-GCM extracts
    └── synthetic/        ← M02/M04 synthetic image outputs
```

---

## 12. Implementation order

Implement in tier order. Do not begin a tier until all specs in the previous
tier are `Authoritative` and all tests pass.

| Tier | Specs | Gate |
|------|-------|------|
| 0 | S01, S02, S03, S04 | All Authoritative before any code runs |
| 1 | S05, S06, S07, S08 | Requires S01–S04 Authoritative |
| 2 | S09, S10, S11 | Requires S01–S08 Authoritative |
| 3 | S12 | Requires S09–S11 |
| 4 | S13 | Requires S12 |
| 5 | S14 | Requires S13 |
| 6 | S15 | Requires S07, S14 |
| 7 | S16, S17 | Requires all prior tiers |
| 8 | S18, S19 | Can begin after S01 |

---

## 13. Instructions for Claude Code

S02 does not produce any executable code. There is no Python module to implement
from this spec. Claude Code should:

1. Read S02 in full before implementing any other module.
2. Verify that every module it implements uses the exact field names and types
   in Section 6.
3. If a module produces or consumes an interface not listed in Section 6,
   flag this to the project lead before proceeding.
4. Use the processing level definitions in Section 3 when naming or describing
   intermediate outputs.

There are no tests for S02 itself. The interfaces it defines are tested
indirectly by the integration tests in S16 (INT02) and S17 (INT03).
