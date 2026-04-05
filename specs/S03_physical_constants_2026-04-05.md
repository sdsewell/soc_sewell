# S03 — Physical Constants

**Spec ID:** S03
**Spec file:** `specs/S03_physical_constants_2026-04-05.md`
**Project:** WindCube FPI Science Operations Center Pipeline
**Institution:** NCAR / High Altitude Observatory (HAO)
**Status:** Authoritative
**Depends on:** S01, S02
**Used by:** S05, S06, S07, S08, S09, S10, S11, S12, S13, S14, S15, S16, S17
**Last updated:** 2026-04-05
**Created/Modified by:** Claude AI

---

## 1. Purpose

This spec defines every physical, spectroscopic, geodetic, orbital, and
instrument constant used by the WindCube pipeline. All constants live in a
single Python module, `src/constants.py`. Every other module imports from
this module using the canonical symbol name. No module may hardcode a
numerical value for a constant that appears in this table.

The canonical symbol names are the import names in Python. Use them exactly.

---

## 2. Design decisions

**Do not duplicate constants.** If a constant belongs in `src/constants.py`,
it must not also be defined in a module file. Remove any constants from
existing modules that duplicate entries here and replace them with imports.

**Source traceability.** Every constant entry carries a source citation.
This is not decoration — it is essential for anomaly investigation. When
a systematic error is traced to a wrong constant, the source field tells
you which document or database to check.

**Exact vs. measured.** Some constants are exact by definition (SI units,
CODATA 2018 fundamentals). Others are measured values from instrument build
reports or standard databases. The distinction is marked explicitly. Exact
values have zero uncertainty by definition; measured values have a tolerance
or confidence interval that is noted.

**Critical corrections from legacy code.** Several constants differ from
values used in the legacy `scotts/fpi_sim/` code. These are flagged
explicitly with a `LEGACY CORRECTION` note. The legacy values must not be
used in any new implementation.

---

## 3. Constant table

### 3.1 Fundamental physical constants

| Python symbol | Value | Units | Source | Notes |
|---------------|-------|-------|--------|-------|
| `SPEED_OF_LIGHT_MS` | 299_792_458.0 | m/s | CODATA 2018 (exact) | Exact by SI definition |
| `BOLTZMANN_J_PER_K` | 1.380649e-23 | J/K | CODATA 2018 (exact) | Exact by SI definition |
| `PLANCK_J_S` | 6.62607015e-34 | J·s | CODATA 2018 (exact) | Exact by SI definition |
| `EARTH_GRAV_PARAM_M3_S2` | 3.986004418e14 | m³/s² | EGM2008 | GM for WGS84 Earth |
| `EARTH_OMEGA_RAD_S` | 7.2921150e-5 | rad/s | WGS84 | Earth rotation rate |
| `EARTH_J2` | 1.08263e-3 | — | EGM2008 | J2 zonal harmonic coefficient |

### 3.2 WGS84 geodetic constants

| Python symbol | Value | Units | Source | Notes |
|---------------|-------|-------|--------|-------|
| `WGS84_A_M` | 6_378_137.0 | m | WGS84 (exact) | Equatorial semi-major axis |
| `WGS84_B_M` | 6_356_752.314_245 | m | WGS84 (derived) | Polar semi-minor axis |
| `WGS84_F` | 1.0 / 298.257_223_563 | — | WGS84 (exact) | Flattening parameter |
| `WGS84_E2` | 6.694379990141317e-3 | — | WGS84 (derived) | First eccentricity squared |

Note: `WGS84_E2 = 1 - (WGS84_B_M / WGS84_A_M)**2` — compute rather than
hardcode if high precision is needed.

### 3.3 Spectroscopic constants — OI airglow target line

| Python symbol | Value | Units | Source | Notes |
|---------------|-------|-------|--------|-------|
| `OI_WAVELENGTH_M` | 630.0304e-9 | m | NIST ASD (air wavelength) | **LEGACY CORRECTION:** legacy code uses 630.0e-9. The NIST value is 630.0304 nm in air. This is the rest wavelength for all Doppler shift calculations |
| `OI_WAVELENGTH_VAC_M` | 630.2010e-9 | m | NIST ASD (vacuum) | Vacuum wavelength; used only if operating in vacuum wavelength convention |
| `OXYGEN_MASS_KG` | 2.6567e-26 | kg | NIST; 16.0 u × 1.66054e-27 kg/u | Mass of one oxygen-16 atom; used for thermal broadening if needed |

**Which wavelength to use:** All Doppler shift calculations in this pipeline
use `OI_WAVELENGTH_M` (air wavelength 630.0304 nm). The Doppler formula is:

```
v_rel = c × (λ_c - OI_WAVELENGTH_M) / OI_WAVELENGTH_M
```

where `λ_c` is the fitted line centre wavelength from M06, and `v_rel` is
the line-of-sight wind speed. Positive `v_rel` means recession (redshift,
source moving away from spacecraft).

### 3.4 Spectroscopic constants — neon calibration lamp

| Python symbol | Value | Units | Source | Notes |
|---------------|-------|-------|--------|-------|
| `NE_WAVELENGTH_1_M` | 640.2248e-9 | m | NIST ASD (Ne I, air) | Primary neon line; relative intensity = 1.0 |
| `NE_WAVELENGTH_2_M` | 638.2991e-9 | m | NIST ASD (Ne I, air) | Secondary neon line; relative intensity 0.8 |
| `NE_INTENSITY_1` | 1.0 | — | NIST ASD | Reference intensity; arbitrary normalisation |
| `NE_INTENSITY_2` | 0.8 | — | NIST ASD | Ratio to primary line |

**Note on neon line selection:** The WindCube neon calibration lamp is a
standard spectral lamp. These two lines are the brightest Ne I lines in the
630–640 nm window and are well separated in the etalon transmission pattern.
The separation Δλ = 1.9257 nm produces a beat period in r² space that is
used in M05 Stage 0 to anchor the etalon gap `t` before the optimiser runs.

Beat period anchor calculation:
```
Δλ_Ne = NE_WAVELENGTH_1_M - NE_WAVELENGTH_2_M = 1.9257e-9 m
FSR   = NE_WAVELENGTH_1_M² / (2 × t) ≈ 10.24e-12 m at t = 20.008 mm
Number of FSRs between lines = Δλ_Ne / FSR ≈ 187.9
```
This means the two neon lines are ~188 FSR periods apart, providing a
long baseline for the beat-period `t` estimate.

### 3.5 Etalon and optical constants

| Python symbol | Value | Units | Source | Notes |
|---------------|-------|-------|--------|-------|
| `ETALON_GAP_M` | 20.008e-3 | m | ICOS build report GNL-4096-R iss1, §7.4 | **LEGACY CORRECTION:** FlatSat code used 20.670e-3 m — that value arises from an FSR-period ambiguity and is wrong. The ICOS mechanical measurement of 20.008 mm is correct |
| `ETALON_GAP_TOLERANCE_M` | 0.010e-3 | m | ICOS build report, §7.4 | Manufacturing tolerance ±0.010 mm |
| `ETALON_R_COATING` | 0.80 | — | ICOS build report, §5.3.1 | As-deposited coating reflectivity at 630 nm; upper bound on effective R |
| `ETALON_R_INSTRUMENT` | 0.53 | — | FlatSat calibration measurement | Effective instrument reflectivity from fringe contrast; lower than coating value due to plate flatness (λ/153–λ/170), parallelism (λ/30), and PSF. Use as default initial estimate in M05 |
| `ETALON_N` | 1.0 | — | Design (air/vacuum gap) | Refractive index of etalon gap medium |
| `FOCAL_LENGTH_M` | 0.200 | m | WindCube optical design | FPI imaging lens focal length |
| `CCD_PIXEL_UM` | 16.0 | µm | CCD97-00 datasheet | Native pixel pitch, unbinned |
| `CCD_PIXEL_2X2_UM` | 32.0 | µm | Derived (2×2 binning mode) | Effective pixel pitch in nominal science mode |
| `ALPHA_RAD_PX` | 1.6e-4 | rad/px | Derived | Magnification constant = CCD_PIXEL_2X2_UM / FOCAL_LENGTH_M = 32e-6 / 0.200 = 1.60e-4 rad/px |

**Magnification constant derivation:**
```
ALPHA_RAD_PX = CCD_PIXEL_2X2_UM × 1e-6 / FOCAL_LENGTH_M
             = 32.0e-6 / 0.200
             = 1.60e-4 rad/px
```
This maps pixel radius `r` (pixels) to angle from optical axis:
`θ(r) = arctan(ALPHA_RAD_PX × r) ≈ ALPHA_RAD_PX × r` (paraxial).

### 3.6 CCD detector constants

| Python symbol | Value | Units | Source | Notes |
|---------------|-------|-------|--------|-------|
| `CCD_PIXELS_NATIVE` | 512 | px | CCD97-00 datasheet | Active pixels per side, unbinned |
| `CCD_PIXELS_2X2` | 256 | px | Derived | Active pixels per side after 2×2 binning |
| `CCD_DARK_RATE_E_PX_S` | 400.0 | e⁻/px/s | CCD97-00 datasheet, 20°C | At 20°C; strongly temperature-dependent |
| `CCD_READ_NOISE_E` | 2.2 | e⁻ rms | CCD97-00 datasheet, 50 kHz, no EM gain | Conventional readout noise (no EM gain) |
| `CCD_READ_NOISE_EM_E` | 1.0 | e⁻ rms | CCD97-00 datasheet, 1 MHz, 1000× gain | Effectively noise-free with EM gain |
| `CCD_FULL_WELL_E` | 130_000 | e⁻ | CCD97-00 datasheet | Peak signal capacity per pixel |
| `CCD_EM_GAIN_DEFAULT` | 200 | — | WindCube operations | Default EM gain; do not exceed 300× without explicit instruction |
| `CCD_QE_PEAK` | 0.90 | — | CCD97-00 datasheet | Peak QE at ~550 nm |
| `CCD_QE_630` | 0.85 | — | CCD97-00 datasheet (estimated from QE curve) | QE at 630 nm; use for photon-to-electron conversion |

### 3.7 Mission and orbital constants

| Python symbol | Value | Units | Source | Notes |
|---------------|-------|-------|--------|-------|
| `SC_ALTITUDE_KM` | 510.0 | km | WC-SE-0003 v8 ConOps | Nominal spacecraft altitude above WGS84 |
| `SC_ALTITUDE_RANGE_KM` | (500.0, 550.0) | km | WC-SE-0003 v8 | Operational altitude range |
| `TP_ALTITUDE_KM` | 250.0 | km | WC-SE-0003 v8 ConOps | OI 630 nm tangent height |
| `TP_ALTITUDE_TOLERANCE_KM` | 5.0 | km | WC-SE-0003 v8, THRF requirement | THRF model error budget |
| `SC_VELOCITY_MS` | 7600.0 | m/s | Derived (circular orbit at 510 km) | Approximate; use orbit propagator for actual value |
| `SC_ORBITAL_PERIOD_S` | 5640.0 | s | WC-SE-0003 v8 | ~94 minutes |
| `DEPRESSION_ANGLE_DEG` | 15.73 | deg | Derived; arccos(6621/6881) | **LEGACY CORRECTION:** earlier documents used 23.4°. Correct calculation: arccos((WGS84_A_M/1e3 + TP_ALTITUDE_KM) / (WGS84_A_M/1e3 + SC_ALTITUDE_KM)) = arccos(6628.137/6888.137) ≈ 15.73°. Note: uses sea-level equatorial radius approximation |
| `ORBIT_INCLINATION_DEG` | 97.4 | deg | WC-SE-0003 v8 | Sun-synchronous inclination |
| `LTAN_HOURS` | 6.0 | hours | WC-SE-0003 v8 | Local time of ascending node (dawn-dusk) |
| `SCIENCE_CADENCE_S` | 10.0 | s | WC-SE-0003 v8 | Nominal image cadence |

**Depression angle derivation:**
```
R_sc = WGS84_A_M/1e3 + SC_ALTITUDE_KM  = 6378.137 + 510.0 = 6888.137 km
R_tp = WGS84_A_M/1e3 + TP_ALTITUDE_KM  = 6378.137 + 250.0 = 6628.137 km
DEPRESSION_ANGLE_DEG = degrees(arccos(R_tp / R_sc))
                     = degrees(arccos(6628.137 / 6888.137))
                     = degrees(arccos(0.96226))
                     = 15.73°
```

### 3.8 Wind measurement and error budget

| Python symbol | Value | Units | Source | Notes |
|---------------|-------|-------|--------|-------|
| `WIND_BIAS_BUDGET_MS` | 9.8 | m/s | STM v1 | Required 1σ wind precision from Monte Carlo |
| `WIND_MAX_STORM_MS` | 400.0 | m/s | STM v1 | Maximum wind speed to resolve (G2 storm) |
| `WIND_MIN_DETECTABLE_MS` | 20.0 | m/s | STM v1 | 5% of peak storm amplitude |
| `LAT_RANGE_DEG` | (-40.0, 40.0) | deg | STM v1, SG1+SG2 | Primary science latitude band |

---

## 4. Derived quantities — compute these, do not hardcode

The following quantities should be computed from the constants above rather
than hardcoded. This ensures consistency if a constant is ever updated.

```python
# Derived quantities — recommended to define in constants.py for convenience

# Etalon FSR at primary neon line (in wavelength)
# FSR = NE_WAVELENGTH_1_M**2 / (2 * ETALON_GAP_M)
ETALON_FSR_NE1_M = NE_WAVELENGTH_1_M**2 / (2 * ETALON_GAP_M)
# = (640.2248e-9)**2 / (2 * 20.008e-3)
# = 4.099e-16 / 4.002e-2 ≈ 1.024e-14 m = 10.24 pm

# Etalon FSR at OI 630 nm (in wavelength)
ETALON_FSR_OI_M = OI_WAVELENGTH_M**2 / (2 * ETALON_GAP_M)
# = (630.0304e-9)**2 / (2 * 20.008e-3) ≈ 9.922e-15 m = 9.92 pm

# Etalon finesse coefficient F (for Airy function)
# F = 4*R / (1-R)**2
# At R=0.53: F = 4*0.53 / (0.47)**2 ≈ 9.62
# At R=0.80: F = 4*0.80 / (0.20)**2 = 80.0

# Velocity equivalent of one FSR at OI 630 nm
# delta_v_FSR = SPEED_OF_LIGHT_MS * ETALON_FSR_OI_M / OI_WAVELENGTH_M
# = 299792458 * 9.922e-15 / 630.0304e-9 ≈ 4.723 km/s
VELOCITY_PER_FSR_MS = SPEED_OF_LIGHT_MS * ETALON_FSR_OI_M / OI_WAVELENGTH_M

# Neon line separation in FSR units (used in M05 beat-period t estimate)
NE_DELTA_LAMBDA_M = NE_WAVELENGTH_1_M - NE_WAVELENGTH_2_M  # 1.9257e-9 m
NE_SEPARATION_FSR = NE_DELTA_LAMBDA_M / ETALON_FSR_NE1_M   # ≈ 187.9 FSR

# CCD plate scale: radians per pixel (2×2 binned)
PLATE_SCALE_RAD_PX = CCD_PIXEL_2X2_UM * 1e-6 / FOCAL_LENGTH_M  # = ALPHA_RAD_PX
```

---

## 5. Module implementation

### 5.1 File location

```
soc_sewell/
└── src/
    └── constants.py   ← this module
```

There is no `__init__.py` re-export needed for this module — it is a flat
file of constants, not a package.

### 5.2 Module structure

The file is organised into clearly labelled sections matching the table
groups in Section 3. Each section begins with a comment block identifying
the group name and source document.

```python
# =============================================================================
# S03 — Physical Constants
# soc_sewell/src/constants.py
#
# Spec:        S03_physical_constants_2026-04-05.md
# Spec date:   2026-04-05
# Generated:   2026-04-05  (Claude Code)
# Tool:        Claude Code
# Last tested: YYYY-MM-DD  (N/N tests pass)
# Depends on:  nothing
# =============================================================================

# ---------------------------------------------------------------------------
# 3.1 Fundamental physical constants
# Source: CODATA 2018 (exact SI values)
# ---------------------------------------------------------------------------
SPEED_OF_LIGHT_MS    = 299_792_458.0     # m/s — exact by SI definition
BOLTZMANN_J_PER_K    = 1.380649e-23      # J/K — exact by SI definition
PLANCK_J_S           = 6.62607015e-34    # J·s — exact by SI definition
...
```

---

## 6. Verification tests

All tests in `tests/test_s03_constants.py`.

### T1 — OI wavelength is not legacy 630.0 nm

```python
def test_OI_wavelength_not_legacy():
    """Catch any regression to the legacy 630.0 nm value."""
    from src.constants import OI_WAVELENGTH_M
    assert abs(OI_WAVELENGTH_M - 630.0e-9) > 1e-12, \
        "OI_WAVELENGTH_M appears to be the legacy 630.0 nm value; should be 630.0304 nm"
    assert abs(OI_WAVELENGTH_M - 630.0304e-9) < 1e-14, \
        f"OI_WAVELENGTH_M = {OI_WAVELENGTH_M*1e9:.6f} nm; expected 630.0304 nm"
```

### T2 — Etalon gap is not legacy 20.670 mm

```python
def test_etalon_gap_not_legacy():
    """Catch any regression to the FlatSat FSR-error value."""
    from src.constants import ETALON_GAP_M
    assert abs(ETALON_GAP_M - 20.670e-3) > 1e-6, \
        "ETALON_GAP_M appears to be the legacy 20.670 mm FlatSat value; should be 20.008 mm"
    assert abs(ETALON_GAP_M - 20.008e-3) < 1e-9, \
        f"ETALON_GAP_M = {ETALON_GAP_M*1e3:.4f} mm; expected 20.008 mm"
```

### T3 — Depression angle is 15.73°, not legacy 23.4°

```python
def test_depression_angle():
    """Verify the depression angle is the correct derived value."""
    import numpy as np
    from src.constants import DEPRESSION_ANGLE_DEG, WGS84_A_M, SC_ALTITUDE_KM, TP_ALTITUDE_KM
    R_sc = WGS84_A_M / 1e3 + SC_ALTITUDE_KM
    R_tp = WGS84_A_M / 1e3 + TP_ALTITUDE_KM
    expected = np.degrees(np.arccos(R_tp / R_sc))
    assert abs(DEPRESSION_ANGLE_DEG - expected) < 0.01, \
        f"DEPRESSION_ANGLE_DEG = {DEPRESSION_ANGLE_DEG:.2f}°; expected {expected:.2f}°"
    assert abs(DEPRESSION_ANGLE_DEG - 23.4) > 1.0, \
        "DEPRESSION_ANGLE_DEG appears to be the legacy 23.4° value"
```

### T4 — FSR calculation is self-consistent

```python
def test_fsr_consistency():
    """Derived FSR values are consistent with primary constants."""
    from src.constants import (ETALON_FSR_OI_M, ETALON_FSR_NE1_M,
                                OI_WAVELENGTH_M, NE_WAVELENGTH_1_M, ETALON_GAP_M)
    fsr_oi_check   = OI_WAVELENGTH_M**2  / (2 * ETALON_GAP_M)
    fsr_ne1_check  = NE_WAVELENGTH_1_M**2 / (2 * ETALON_GAP_M)
    assert abs(ETALON_FSR_OI_M  - fsr_oi_check)  < 1e-18
    assert abs(ETALON_FSR_NE1_M - fsr_ne1_check) < 1e-18
```

### T5 — Neon line separation ≈ 187.9 FSR

```python
def test_neon_separation_fsr():
    """Beat period anchor: Ne lines are ~188 FSR apart."""
    from src.constants import NE_SEPARATION_FSR
    assert 185 < NE_SEPARATION_FSR < 191, \
        f"NE_SEPARATION_FSR = {NE_SEPARATION_FSR:.1f}; expected ≈ 187.9"
```

### T6 — Magnification constant is consistent with pixel pitch and focal length

```python
def test_alpha_consistency():
    """ALPHA_RAD_PX = pixel_pitch / focal_length."""
    from src.constants import ALPHA_RAD_PX, CCD_PIXEL_2X2_UM, FOCAL_LENGTH_M
    expected = CCD_PIXEL_2X2_UM * 1e-6 / FOCAL_LENGTH_M
    assert abs(ALPHA_RAD_PX - expected) < 1e-10, \
        f"ALPHA_RAD_PX = {ALPHA_RAD_PX:.3e}; expected {expected:.3e}"
```

### T7 — All constants are float or tuple, not string

```python
def test_constant_types():
    """Guard against string or None values from typos."""
    import src.constants as c
    non_tuple_names = [
        'SPEED_OF_LIGHT_MS', 'BOLTZMANN_J_PER_K', 'OI_WAVELENGTH_M',
        'NE_WAVELENGTH_1_M', 'NE_WAVELENGTH_2_M', 'ETALON_GAP_M',
        'FOCAL_LENGTH_M', 'ALPHA_RAD_PX', 'DEPRESSION_ANGLE_DEG',
        'SC_ALTITUDE_KM', 'TP_ALTITUDE_KM', 'WIND_BIAS_BUDGET_MS',
    ]
    for name in non_tuple_names:
        val = getattr(c, name)
        assert isinstance(val, (int, float)), \
            f"{name} has type {type(val).__name__}, expected float"
```

### T8 — Velocity per FSR is physically reasonable

```python
def test_velocity_per_fsr():
    """One FSR should correspond to ~4.7 km/s at OI 630 nm, 20 mm gap."""
    from src.constants import VELOCITY_PER_FSR_MS
    assert 4_500 < VELOCITY_PER_FSR_MS < 5_000, \
        f"VELOCITY_PER_FSR_MS = {VELOCITY_PER_FSR_MS:.0f} m/s; expected ~4720 m/s"
```

---

## 7. Expected numerical values

| Symbol | Expected value | Derivation |
|--------|----------------|-----------|
| `OI_WAVELENGTH_M` | 630.0304e-9 m | NIST ASD air wavelength, Ne I |
| `ETALON_GAP_M` | 20.008e-3 m | ICOS build report spacer measurement |
| `ETALON_FSR_OI_M` | ≈ 9.922e-15 m (9.92 pm) | λ²/(2t) at OI 630 nm |
| `ETALON_FSR_NE1_M` | ≈ 1.024e-14 m (10.24 pm) | λ²/(2t) at Ne 640.2 nm |
| `NE_SEPARATION_FSR` | ≈ 187.9 | Δλ_Ne / FSR_Ne1 |
| `ALPHA_RAD_PX` | 1.60e-4 rad/px | 32e-6 m / 0.200 m |
| `DEPRESSION_ANGLE_DEG` | 15.73° | arccos(6628.137/6888.137) |
| `VELOCITY_PER_FSR_MS` | ≈ 4723 m/s | c × FSR_OI / λ_OI |
| `WIND_BIAS_BUDGET_MS` | 9.8 m/s | STM v1 Monte Carlo result |

---

## 8. File location in repository

```
soc_sewell/
├── src/
│   └── constants.py          ← implementation of this spec
└── tests/
    └── test_s03_constants.py ← tests for this spec
```

---

## 9. Instructions for Claude Code

1. Read this entire spec before writing any code.
2. Create `src/constants.py` with sections in the order shown in Section 3.
3. Use the exact Python symbol names from the table — these are imported by
   name in every other module. Changing a symbol name breaks all imports.
4. Each section must begin with a comment block giving: group name,
   source document, and any LEGACY CORRECTION notes.
5. Implement the derived quantities in Section 4 at the bottom of the file,
   clearly separated from the primary constants.
6. Write `tests/test_s03_constants.py` with tests T1–T8 as specified above.
7. Run `pytest tests/test_s03_constants.py -v` — all 8 tests must pass.
8. After implementing: open `fpi/m01_airy_forward_model.py` and check whether
   any constants defined there duplicate entries in `src/constants.py`. If so,
   replace the local definitions with imports from `src.constants`. Re-run M01
   tests to confirm nothing broke.

**Commit message template:**
```
feat(constants): implement S03 physical constants module, 8/8 tests pass
Implements: S03_physical_constants_2026-04-05.md
```
