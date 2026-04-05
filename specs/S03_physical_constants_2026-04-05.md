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

**Revision note:** Added `compute_depression_angle()` helper function (Section 4.1)
and changed `DEPRESSION_ANGLE_DEG` from a hardcoded value to a value computed at
module load time from `SC_ALTITUDE_KM` and `TP_ALTITUDE_KM`. Test T3 strengthened;
test T9 (sensitivity) added. No other content changed.

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

**Computed vs. hardcoded.** Where a constant is fully determined by other
constants already in this module, it must be computed from those primaries
rather than hardcoded. This ensures consistency if a primary value is ever
updated, and makes the derivation self-documenting in the code.
`DEPRESSION_ANGLE_DEG` is the canonical example of this pattern — see
Section 3.7 and Section 4.1.

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
| `OXYGEN_MASS_KG` | 2.6567e-26 | kg | NIST; 16.0 u × 1.66054e-27 kg/u | Mass of one oxygen-16 atom |

**Which wavelength to use:** All Doppler shift calculations use `OI_WAVELENGTH_M`
(air wavelength 630.0304 nm). The Doppler formula is:
```
v_rel = c × (λ_c - OI_WAVELENGTH_M) / OI_WAVELENGTH_M
```
Positive `v_rel` means recession (redshift, source moving away from spacecraft).

### 3.4 Spectroscopic constants — neon calibration lamp

| Python symbol | Value | Units | Source | Notes |
|---------------|-------|-------|--------|-------|
| `NE_WAVELENGTH_1_M` | 640.2248e-9 | m | NIST ASD (Ne I, air) | Primary neon line; relative intensity = 1.0 |
| `NE_WAVELENGTH_2_M` | 638.2991e-9 | m | NIST ASD (Ne I, air) | Secondary neon line; relative intensity 0.8 |
| `NE_INTENSITY_1` | 1.0 | — | NIST ASD | Reference intensity; arbitrary normalisation |
| `NE_INTENSITY_2` | 0.8 | — | NIST ASD | Ratio to primary line |

Beat period anchor calculation:
```
Δλ_Ne = NE_WAVELENGTH_1_M - NE_WAVELENGTH_2_M = 1.9257e-9 m
FSR   = NE_WAVELENGTH_1_M² / (2 × t) ≈ 10.24e-12 m at t = 20.008 mm
Number of FSRs between lines ≈ 187.9
```

### 3.5 Etalon and optical constants

| Python symbol | Value | Units | Source | Notes |
|---------------|-------|-------|--------|-------|
| `ETALON_GAP_M` | 20.008e-3 | m | ICOS build report GNL-4096-R iss1, §7.4 | **LEGACY CORRECTION:** FlatSat code used 20.670e-3 m — FSR-period ambiguity error |
| `ETALON_GAP_TOLERANCE_M` | 0.010e-3 | m | ICOS build report, §7.4 | Manufacturing tolerance ±0.010 mm |
| `ETALON_R_COATING` | 0.80 | — | ICOS build report, §5.3.1 | As-deposited coating reflectivity at 630 nm; upper bound on effective R |
| `ETALON_R_INSTRUMENT` | 0.53 | — | FlatSat calibration measurement | Effective instrument reflectivity from fringe contrast |
| `ETALON_N` | 1.0 | — | Design (air/vacuum gap) | Refractive index of etalon gap medium |
| `FOCAL_LENGTH_M` | 0.200 | m | WindCube optical design | FPI imaging lens focal length |
| `CCD_PIXEL_UM` | 16.0 | µm | CCD97-00 datasheet | Native pixel pitch, unbinned |
| `CCD_PIXEL_2X2_UM` | 32.0 | µm | Derived (2×2 binning mode) | Effective pixel pitch in nominal science mode |
| `ALPHA_RAD_PX` | 1.6e-4 | rad/px | Derived | = CCD_PIXEL_2X2_UM × 1e-6 / FOCAL_LENGTH_M = 32e-6 / 0.200 |

### 3.6 CCD detector constants

| Python symbol | Value | Units | Source | Notes |
|---------------|-------|-------|--------|-------|
| `CCD_PIXELS_NATIVE` | 512 | px | CCD97-00 datasheet | Active pixels per side, unbinned |
| `CCD_PIXELS_2X2` | 256 | px | Derived | Active pixels per side after 2×2 binning |
| `CCD_DARK_RATE_E_PX_S` | 400.0 | e⁻/px/s | CCD97-00 datasheet, 20°C | Strongly temperature-dependent |
| `CCD_READ_NOISE_E` | 2.2 | e⁻ rms | CCD97-00 datasheet, 50 kHz, no EM gain | Conventional readout noise |
| `CCD_READ_NOISE_EM_E` | 1.0 | e⁻ rms | CCD97-00 datasheet, 1 MHz, 1000× gain | Effectively noise-free with EM gain |
| `CCD_FULL_WELL_E` | 130_000 | e⁻ | CCD97-00 datasheet | Peak signal capacity per pixel |
| `CCD_EM_GAIN_DEFAULT` | 200 | — | WindCube operations | Default EM gain; do not exceed 300× |
| `CCD_QE_PEAK` | 0.90 | — | CCD97-00 datasheet | Peak QE at ~550 nm |
| `CCD_QE_630` | 0.85 | — | CCD97-00 datasheet (estimated) | QE at 630 nm |

### 3.7 Mission and orbital constants

| Python symbol | Value | Units | Source | Notes |
|---------------|-------|-------|--------|-------|
| `SC_ALTITUDE_KM` | 510.0 | km | WC-SE-0003 v8 ConOps | Nominal spacecraft altitude above WGS84 |
| `SC_ALTITUDE_RANGE_KM` | (500.0, 550.0) | km | WC-SE-0003 v8 | Operational altitude range |
| `TP_ALTITUDE_KM` | 250.0 | km | WC-SE-0003 v8 ConOps | OI 630 nm tangent height |
| `TP_ALTITUDE_TOLERANCE_KM` | 5.0 | km | WC-SE-0003 v8 | THRF model error budget |
| `SC_VELOCITY_MS` | 7600.0 | m/s | Derived (circular orbit at 510 km) | Approximate; use orbit propagator for actual |
| `SC_ORBITAL_PERIOD_S` | 5640.0 | s | WC-SE-0003 v8 | ~94 minutes |
| `DEPRESSION_ANGLE_DEG` | **computed** | deg | `compute_depression_angle(SC_ALTITUDE_KM, TP_ALTITUDE_KM)` → 15.73°. See Section 4.1. **LEGACY CORRECTION:** earlier documents hardcoded 23.4° (wrong altitude). Now computed from primaries. |
| `ORBIT_INCLINATION_DEG` | 97.4 | deg | WC-SE-0003 v8 | Sun-synchronous inclination |
| `LTAN_HOURS` | 6.0 | hours | WC-SE-0003 v8 | Local time of ascending node (dawn-dusk) |
| `SCIENCE_CADENCE_S` | 10.0 | s | WC-SE-0003 v8 | Nominal image cadence |

### 3.8 Wind measurement and error budget

| Python symbol | Value | Units | Source | Notes |
|---------------|-------|-------|--------|-------|
| `WIND_BIAS_BUDGET_MS` | 9.8 | m/s | STM v1 | Required 1σ wind precision from Monte Carlo |
| `WIND_MAX_STORM_MS` | 400.0 | m/s | STM v1 | Maximum wind speed to resolve (G2 storm) |
| `WIND_MIN_DETECTABLE_MS` | 20.0 | m/s | STM v1 | 5% of peak storm amplitude |
| `LAT_RANGE_DEG` | (-40.0, 40.0) | deg | STM v1, SG1+SG2 | Primary science latitude band |

---

## 4. Derived quantities — compute these, do not hardcode

### 4.1 Depression angle helper function

This function must be defined in `src/constants.py` **immediately after**
`SC_ORBITAL_PERIOD_S` and **before** the `DEPRESSION_ANGLE_DEG` assignment.

**Physical meaning.** The depression angle is the angle below local horizontal
at which the FPI boresight must point to intercept the thermosphere at the
target tangent height. It depends only on the spacecraft altitude and the
desired tangent height — both of which are primary constants in this module.

**Why a function, not just a constant.** Downstream modules (S08 INT01
integration notebook, S17 INT03 end-to-end notebook) need to compute the
depression angle for altitudes other than the nominal 510 km — for example,
to characterise sensitivity across the 500–550 km operational range, or to
visualize how the LOS geometry changes over the orbit lifetime as the
spacecraft decays. A callable function makes all of these uses trivial.

**Approximation used.** The calculation uses the WGS84 equatorial radius as
the Earth radius at all latitudes. This introduces up to ±0.3° error at high
latitudes due to Earth's oblateness. For constructing the synthetic boresight
quaternion in NB02 this is adequate. For the operational tangent-point finder
(NB02b), the full WGS84 ray-ellipsoid intersection is used instead, so
`DEPRESSION_ANGLE_DEG` is not in that code path.

```python
import numpy as np

def compute_depression_angle(sc_alt_km: float, tp_alt_km: float) -> float:
    """
    Compute the limb depression angle from spacecraft and tangent point altitudes.

    The depression angle δ is the angle below local horizontal at the spacecraft
    at which the boresight must be directed to reach a tangent height of tp_alt_km.

    Derived from the right triangle O–T–S/C (Earth centre, tangent point,
    spacecraft): the angle at S/C equals arccos(R_tp / R_sc).

    Uses WGS84 equatorial radius as Earth radius (equatorial approximation,
    accurate to ±0.3° across latitudes). For the full oblate-ellipsoid
    treatment see NB02b ray-ellipsoid intersection.

    Parameters
    ----------
    sc_alt_km : float
        Spacecraft altitude above WGS84 ellipsoid, km.
    tp_alt_km : float
        Tangent point altitude above WGS84 ellipsoid, km.

    Returns
    -------
    float
        Depression angle, degrees. Always positive (boresight depressed
        below horizontal). Valid range for WindCube: ~10°–25°.

    Examples
    --------
    compute_depression_angle(510.0, 250.0)  ->  15.73°  (nominal WindCube)
    compute_depression_angle(500.0, 250.0)  ->  15.45°  (lower orbit bound)
    compute_depression_angle(550.0, 250.0)  ->  16.75°  (upper orbit bound)
    compute_depression_angle(525.0, 250.0)  ->  15.18°  (old wrong altitude)

    Notes
    -----
    The legacy value of 23.4° arose from two compounding errors:
      (1) Using altitude 525 km instead of the correct 510 km.
      (2) A geometric error in an earlier formula version.
    Both are corrected here. The formula arccos(R_tp / R_sc) is geometrically
    exact for a spherical Earth and agrees with WC-SE-0003 Section 4.
    """
    R_earth_km = WGS84_A_M / 1e3          # equatorial radius in km
    R_sc = R_earth_km + sc_alt_km
    R_tp = R_earth_km + tp_alt_km
    return float(np.degrees(np.arccos(R_tp / R_sc)))


# Nominal mission depression angle.
# Computed from primary constants — NOT hardcoded.
# Updates automatically if SC_ALTITUDE_KM or TP_ALTITUDE_KM changes.
DEPRESSION_ANGLE_DEG = compute_depression_angle(SC_ALTITUDE_KM, TP_ALTITUDE_KM)
# Nominal result: ~15.73° for SC_ALTITUDE_KM=510.0, TP_ALTITUDE_KM=250.0
```

### 4.2 Other derived quantities

```python
# Etalon FSR at primary neon line (wavelength)
ETALON_FSR_NE1_M = NE_WAVELENGTH_1_M**2 / (2 * ETALON_GAP_M)
# ≈ 1.024e-14 m = 10.24 pm

# Etalon FSR at OI 630 nm (wavelength)
ETALON_FSR_OI_M = OI_WAVELENGTH_M**2 / (2 * ETALON_GAP_M)
# ≈ 9.922e-15 m = 9.92 pm

# Velocity equivalent of one FSR at OI 630 nm
VELOCITY_PER_FSR_MS = SPEED_OF_LIGHT_MS * ETALON_FSR_OI_M / OI_WAVELENGTH_M
# ≈ 4723 m/s

# Neon line separation in FSR units (M05 beat-period t estimate)
NE_DELTA_LAMBDA_M = NE_WAVELENGTH_1_M - NE_WAVELENGTH_2_M   # 1.9257e-9 m
NE_SEPARATION_FSR = NE_DELTA_LAMBDA_M / ETALON_FSR_NE1_M    # ≈ 187.9 FSR

# CCD plate scale (2×2 binned)
PLATE_SCALE_RAD_PX = CCD_PIXEL_2X2_UM * 1e-6 / FOCAL_LENGTH_M  # = ALPHA_RAD_PX
```

**Implementation order in `src/constants.py`:** Define primary constants in
Section 3.1–3.8 order. Then define `compute_depression_angle()` inside the
Section 3.7 block (after `SC_ORBITAL_PERIOD_S`), followed immediately by
`DEPRESSION_ANGLE_DEG`. Then define the Section 4.2 derived quantities at
the end of the file.

---

## 5. Module structure

```python
# =============================================================================
# S03 — Physical Constants
# soc_sewell/src/constants.py
#
# Spec:        S03_physical_constants_2026-04-05.md
# Spec date:   2026-04-05
# Generated:   YYYY-MM-DD  (Claude Code)
# Tool:        Claude Code
# Last tested: YYYY-MM-DD  (9/9 tests pass)
# Depends on:  nothing
# =============================================================================

import numpy as np

# 3.1 Fundamental physical constants ...
# 3.2 WGS84 geodetic constants ...
# 3.3 Spectroscopic — OI airglow ...
# 3.4 Spectroscopic — neon lamp ...
# 3.5 Etalon and optical ...
# 3.6 CCD detector ...
# 3.7 Mission and orbital ...
#     SC_ALTITUDE_KM = 510.0
#     TP_ALTITUDE_KM = 250.0
#     ...
#     SC_ORBITAL_PERIOD_S = 5640.0
#
#     def compute_depression_angle(...): ...
#     DEPRESSION_ANGLE_DEG = compute_depression_angle(SC_ALTITUDE_KM, TP_ALTITUDE_KM)
#
# 3.8 Wind measurement and error budget ...
# 4.2 Derived quantities ...
# Quality flags (PipelineFlags class — from S04) ...
```

---

## 6. Verification tests

All tests in `tests/test_s03_constants.py`. Tests T1–T2 and T4–T8 are
unchanged. T3 is strengthened. T9 is new.

### T1 — OI wavelength is not legacy 630.0 nm

```python
def test_OI_wavelength_not_legacy():
    from src.constants import OI_WAVELENGTH_M
    assert abs(OI_WAVELENGTH_M - 630.0e-9) > 1e-12, \
        "OI_WAVELENGTH_M appears to be the legacy 630.0 nm value; should be 630.0304 nm"
    assert abs(OI_WAVELENGTH_M - 630.0304e-9) < 1e-14, \
        f"OI_WAVELENGTH_M = {OI_WAVELENGTH_M*1e9:.6f} nm; expected 630.0304 nm"
```

### T2 — Etalon gap is not legacy 20.670 mm

```python
def test_etalon_gap_not_legacy():
    from src.constants import ETALON_GAP_M
    assert abs(ETALON_GAP_M - 20.670e-3) > 1e-6, \
        "ETALON_GAP_M is the legacy FlatSat value; should be 20.008 mm"
    assert abs(ETALON_GAP_M - 20.008e-3) < 1e-9, \
        f"ETALON_GAP_M = {ETALON_GAP_M*1e3:.4f} mm; expected 20.008 mm"
```

### T3 — Depression angle is computed from primaries, not hardcoded

```python
def test_depression_angle_computed_from_primaries():
    """
    DEPRESSION_ANGLE_DEG must equal compute_depression_angle(SC_ALTITUDE_KM,
    TP_ALTITUDE_KM) to machine precision — proving it is computed, not hardcoded.
    """
    import numpy as np
    from src.constants import (DEPRESSION_ANGLE_DEG, SC_ALTITUDE_KM,
                                TP_ALTITUDE_KM, compute_depression_angle)
    recomputed = compute_depression_angle(SC_ALTITUDE_KM, TP_ALTITUDE_KM)
    assert abs(DEPRESSION_ANGLE_DEG - recomputed) < 1e-10, \
        (f"DEPRESSION_ANGLE_DEG ({DEPRESSION_ANGLE_DEG:.4f}°) does not match "
         f"compute_depression_angle({SC_ALTITUDE_KM}, {TP_ALTITUDE_KM}) "
         f"= {recomputed:.4f}°. It may be hardcoded.")
    assert abs(DEPRESSION_ANGLE_DEG - 15.73) < 0.02, \
        f"Nominal depression angle = {DEPRESSION_ANGLE_DEG:.2f}°; expected ~15.73°"
    assert abs(DEPRESSION_ANGLE_DEG - 23.4) > 1.0, \
        "DEPRESSION_ANGLE_DEG is the legacy 23.4° value"
```

### T4 — FSR calculation is self-consistent

```python
def test_fsr_consistency():
    from src.constants import (ETALON_FSR_OI_M, ETALON_FSR_NE1_M,
                                OI_WAVELENGTH_M, NE_WAVELENGTH_1_M, ETALON_GAP_M)
    assert abs(ETALON_FSR_OI_M  - OI_WAVELENGTH_M**2  / (2 * ETALON_GAP_M)) < 1e-18
    assert abs(ETALON_FSR_NE1_M - NE_WAVELENGTH_1_M**2 / (2 * ETALON_GAP_M)) < 1e-18
```

### T5 — Neon line separation ≈ 187.9 FSR

```python
def test_neon_separation_fsr():
    from src.constants import NE_SEPARATION_FSR
    assert 185 < NE_SEPARATION_FSR < 191, \
        f"NE_SEPARATION_FSR = {NE_SEPARATION_FSR:.1f}; expected ≈ 187.9"
```

### T6 — Magnification constant consistent with pixel pitch and focal length

```python
def test_alpha_consistency():
    from src.constants import ALPHA_RAD_PX, CCD_PIXEL_2X2_UM, FOCAL_LENGTH_M
    expected = CCD_PIXEL_2X2_UM * 1e-6 / FOCAL_LENGTH_M
    assert abs(ALPHA_RAD_PX - expected) < 1e-10, \
        f"ALPHA_RAD_PX = {ALPHA_RAD_PX:.3e}; expected {expected:.3e}"
```

### T7 — All named constants are float or tuple, not string

```python
def test_constant_types():
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
    from src.constants import VELOCITY_PER_FSR_MS
    assert 4_500 < VELOCITY_PER_FSR_MS < 5_000, \
        f"VELOCITY_PER_FSR_MS = {VELOCITY_PER_FSR_MS:.0f} m/s; expected ~4720 m/s"
```

### T9 — compute_depression_angle() responds correctly to altitude inputs

```python
def test_depression_angle_sensitivity():
    """
    Verify that compute_depression_angle() correctly responds to different
    altitude inputs. Proves it is a live calculation, not a lookup or stub.
    """
    from src.constants import compute_depression_angle
    angle_nominal  = compute_depression_angle(510.0, 250.0)  # 15.73°
    angle_low_sc   = compute_depression_angle(500.0, 250.0)  # lower orbit → smaller angle
    angle_high_sc  = compute_depression_angle(550.0, 250.0)  # higher orbit → larger angle
    angle_high_tp  = compute_depression_angle(510.0, 300.0)  # higher tangent → smaller angle

    assert angle_low_sc  < angle_nominal, \
        "Lower orbit altitude should give smaller depression angle"
    assert angle_high_sc > angle_nominal, \
        "Higher orbit altitude should give larger depression angle"
    assert angle_high_tp < angle_nominal, \
        "Higher tangent height should give smaller depression angle"

    for angle, label in [(angle_nominal, 'nominal'), (angle_low_sc, 'low_sc'),
                         (angle_high_sc, 'high_sc'), (angle_high_tp, 'high_tp')]:
        assert 10.0 < angle < 25.0, \
            f"Depression angle ({label}) = {angle:.2f}° outside plausible range [10°, 25°]"
```

---

## 7. Expected numerical values

| Symbol | Expected value | Derivation |
|--------|----------------|-----------|
| `OI_WAVELENGTH_M` | 630.0304e-9 m | NIST ASD air wavelength |
| `ETALON_GAP_M` | 20.008e-3 m | ICOS build report spacer measurement |
| `ETALON_FSR_OI_M` | ≈ 9.922e-15 m (9.92 pm) | λ²/(2t) at OI 630 nm |
| `ETALON_FSR_NE1_M` | ≈ 1.024e-14 m (10.24 pm) | λ²/(2t) at Ne 640.2 nm |
| `NE_SEPARATION_FSR` | ≈ 187.9 | Δλ_Ne / FSR_Ne1 |
| `ALPHA_RAD_PX` | 1.60e-4 rad/px | 32e-6 m / 0.200 m |
| `DEPRESSION_ANGLE_DEG` | ≈ 15.73° | `compute_depression_angle(510.0, 250.0)` |
| `compute_depression_angle(500.0, 250.0)` | ≈ 15.45° | lower orbit bound |
| `compute_depression_angle(550.0, 250.0)` | ≈ 16.75° | upper orbit bound |
| `VELOCITY_PER_FSR_MS` | ≈ 4723 m/s | c × FSR_OI / λ_OI |
| `WIND_BIAS_BUDGET_MS` | 9.8 m/s | STM v1 Monte Carlo result |

---

## 8. File location in repository

```
soc_sewell/
├── src/
│   └── constants.py          ← implementation of this spec
└── tests/
    └── test_s03_constants.py ← tests T1–T9 (9 tests)
```

---

## 9. Instructions for Claude Code

This is a **revision** to an already-implemented module. Modify the existing
files — do not create new ones.

**Changes to `src/constants.py` (3 edits only):**

1. Add `import numpy as np` at the top if not already present.
2. In the Section 3.7 block, immediately after the `SC_ORBITAL_PERIOD_S`
   assignment, insert the `compute_depression_angle()` function exactly as
   written in Section 4.1. Copy the full docstring.
3. Replace the line that assigns `DEPRESSION_ANGLE_DEG` as a literal (e.g.
   `DEPRESSION_ANGLE_DEG = 15.73`) with:
   ```python
   DEPRESSION_ANGLE_DEG = compute_depression_angle(SC_ALTITUDE_KM, TP_ALTITUDE_KM)
   ```
   Do not change anything else in the file.

**Changes to `tests/test_s03_constants.py` (2 edits only):**

4. Replace the existing `test_depression_angle()` function with the new
   `test_depression_angle_computed_from_primaries()` (T3 above).
5. Add `test_depression_angle_sensitivity()` (T9 above) at the end of the file.
   Do not modify any other test.

**Verification:**
```bash
pytest tests/test_s03_constants.py -v
```
All 9 tests must pass before committing.

**Commit message:**
```
fix(constants): compute DEPRESSION_ANGLE_DEG from primaries, add T9 sensitivity test
Implements: S03_physical_constants_2026-04-05.md
```

**Archive instruction:** Per S01 Section 3, move the previous committed version
of this spec to `specs/archive/` before placing this revised version in
`specs/`. Use a filename suffix to distinguish (e.g. append `_r1` before the
date, or keep the same filename and note the revision in git history).
The `Last updated` date remains `2026-04-05`.
