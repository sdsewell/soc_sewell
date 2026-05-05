# S03 — Physical Constants

**Spec ID:** S03
**Spec file:** `specs/S03_physical_constants_2026-05-05.md`
**Project:** WindCube FPI Science Operations Center Pipeline
**Institution:** NCAR / High Altitude Observatory (HAO)
**Status:** Authoritative
**Depends on:** S01, S02
**Used by:** S05, S06, S07, S08, S09, S10, S11, S12, S13, S14, S15, S16, S17
**Last updated:** 2026-05-05
**Created/Modified by:** Claude AI

**Revision note (2026-05-05):** Four structural changes from the 2026-04-29 revision:

1. **Vacuum wavelengths are now canonical for all emission lines.**  The
   primary symbols `OI_WAVELENGTH_M`, `NE_WAVELENGTH_1_M`, and
   `NE_WAVELENGTH_2_M` now carry the **vacuum** values (Edlén-derived from
   the NIST/Burns air values).  The air wavelengths are retained as named
   constants (`_AIR_M` suffix) for use in the Doppler note and FSR
   calculations, but they are no longer the default.  All deprecated
   backward-compatibility aliases removed.

2. **`FOCAL_LENGTH_M` removed.**  The imaging lens focal length is no
   longer a named constant.  The plate scale `ALPHA_RAD_PX` is now a
   **hardcoded primary** (1.6071e-4 rad/px) whose authoritative value comes
   from the Tolansky two-line analysis of a real neon calibration image
   (Z01 / S13).  The nominal design value (32 µm / 200 mm = 1.60e-4 rad/px)
   is retained only as an inline comment for reference.

3. **`PLATE_SCALE_RPX` and `PLATE_SCALE_RAD_PX` removed.**  `ALPHA_RAD_PX`
   is the single, uniquely-defined plate-scale symbol throughout the pipeline.

4. **Section 3.4b merged into Section 3.5.**  The gap / plate-scale /
   reflectivity calibration constants (`ETALON_GAP_M`, `ALPHA_RAD_PX`,
   `R_REFL_FLATSAT`, `R_MAX_PX`) now live in one place (§3.5).
   Section 3.4b is removed.  Note: `ETALON_GAP_M` is updated to the
   Tolansky-recovered operational value (20.106 mm); the ICOS spacer
   measurement (20.008 mm) is preserved as `ETALON_GAP_ICOS_M` for
   integer-disambiguation use only.

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
Section 3.7 and Section 4.1.  All vacuum wavelengths are likewise computed
from their air counterparts via `_edlen_n()` rather than hardcoded.

**Critical corrections from legacy code.** Several constants differ from
values used in the legacy `scotts/fpi_sim/` code. These are flagged
explicitly with a `LEGACY CORRECTION` note. The legacy values must not be
used in any new implementation.

**Vacuum wavelengths are canonical.** WindCube observes from orbit; the
thermospheric OI 630 nm photons travel through vacuum.  All pipeline
spectroscopic symbols therefore carry the **vacuum** wavelength as their
primary value.  Air wavelengths (NIST ASD / Burns et al. 1950 standards,
15 °C, 101 325 Pa) are retained as `_AIR_M` variants because:
(a) they are the laboratory reference standard used for line identification,
(b) the Edlén consistency tests require them, and
(c) FSR calculations performed at instrument design time used air values
    and are preserved unchanged.

The Doppler formula used throughout the pipeline is:
```
v_rel = c × (λ_c − OI_WAVELENGTH_M) / OI_WAVELENGTH_M
```
where `OI_WAVELENGTH_M` is the vacuum rest wavelength.  Positive `v_rel`
means recession (redshift, source moving away from spacecraft).

**Single plate-scale symbol.** `ALPHA_RAD_PX` is the one and only
plate-scale constant.  Its value (1.6071e-4 rad/px) comes from the
Tolansky two-line joint fit to a real neon calibration image (S13 / Z01).
The nominal design value (pixel pitch / focal length = 32 µm / 200 mm =
1.60e-4 rad/px) differs by ~0.4 % and must not be used in fitting code.

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

The OI 630 nm forbidden transition (O(¹D) → O(³P)) is the primary WindCube
science line.  The **vacuum** wavelength is the canonical pipeline value,
derived via the Edlén (1966) formula from the NIST ASD air wavelength.

| Python symbol | Value | Units | Source | Notes |
|---------------|-------|-------|--------|-------|
| `OI_WAVELENGTH_M` | 629.9582e-9 | m | Edlén (1966) from NIST ASD air value | **Canonical vacuum rest wavelength for all Doppler calculations**; computed via `_air_to_vac_nm(630.0304)` |
| `OI_WAVELENGTH_AIR_M` | 630.0304e-9 | m | NIST ASD (air, 15 °C, 101 325 Pa) | Retained for FSR calculations and Edlén consistency tests; Δ = +72.2 pm relative to vacuum |
| `OXYGEN_MASS_KG` | 2.6567e-26 | kg | NIST; 16.0 u × 1.66054e-27 kg/u | Mass of one oxygen-16 atom |

**Legacy corrections:**
- `OI_WAVELENGTH_VACUUM_M = 630.0304e-9` was the old name and value; it was
  **mislabelled** (the value 630.0304 nm is the air wavelength).  Symbol removed.
- `OI_WAVELENGTH_VAC_M = 630.2010e-9` was an unrecognised erroneous value.
  Replaced by the Edlén-derived 629.9582 nm.

### 3.4 Spectroscopic constants — neon calibration lamp

The two Ne I lines in the 630–640 nm window are used for FPI etalon gap
calibration (Tolansky two-line method).  **Vacuum** wavelengths are
canonical; air wavelengths are retained for FSR/beat-period calculations.

| Python symbol | Value | Units | Source | Notes |
|---------------|-------|-------|--------|-------|
| `NE_WAVELENGTH_1_M` | 640.1426e-9 | m | Edlén (1966) from NIST ASD / Burns et al. (1950) | **Canonical vacuum wavelength**, primary Ne line; computed via `_air_to_vac_nm(640.2248)`; Δ = −82.2 pm |
| `NE_WAVELENGTH_1_AIR_M` | 640.2248e-9 | m | NIST ASD / Burns et al. (1950), Ne I air | Retained for FSR and Tolansky calculations |
| `NE_WAVELENGTH_2_M` | 638.2189e-9 | m | Edlén (1966) from NIST ASD / Burns et al. (1950) | **Canonical vacuum wavelength**, secondary Ne line; computed via `_air_to_vac_nm(638.2991)`; Δ = −80.2 pm |
| `NE_WAVELENGTH_2_AIR_M` | 638.2991e-9 | m | NIST ASD / Burns et al. (1950), Ne I air | Retained for FSR and Tolansky calculations |
| `NE_INTENSITY_1` | 1.0 | — | NIST ASD | Reference intensity; arbitrary normalisation |
| `NE_INTENSITY_2` | 0.8 | — | NIST ASD | Ratio of secondary to primary line |

Beat period anchor calculation (uses air wavelengths to match instrument
design convention):
```
Δλ_Ne = NE_WAVELENGTH_1_AIR_M - NE_WAVELENGTH_2_AIR_M = 1.9257e-9 m
FSR   = NE_WAVELENGTH_1_AIR_M² / (2 × ETALON_GAP_M) ≈ 10.27e-12 m at d = 20.106 mm
Number of FSRs between lines ≈ 187.4
```

### 3.5 Etalon, optical, and calibration constants

Sources: ICOS build report GNL-4096-R iss1; FlatSat calibration; Tolansky
two-line analysis of real neon calibration image (S13 / Z01, 2026-04-21).

**Gap note:** Two gap values are defined.  `ETALON_GAP_M` is the
**operational gap** recovered from the Tolansky two-line analysis; this is
the value used in all fitting and FSR calculations.  `ETALON_GAP_ICOS_M` is
the ICOS spacer measurement used **only** to resolve the FSR-period integer
ambiguity (N_int = −189); it must not be used as the gap in any fit.

| Python symbol | Value | Units | Source | Notes |
|---------------|-------|-------|--------|-------|
| `ETALON_GAP_M` | 20.106e-3 | m | Tolansky two-line analysis (S13 / Z01, 2026-04-21) | **Authoritative operational gap**; use in all FSR / fitting calculations |
| `ETALON_GAP_ICOS_M` | 20.008e-3 | m | ICOS build report GNL-4096-R iss1, §7.4 | Spacer measurement; used **only** to resolve FSR-period integer N_int = −189; do not substitute for `ETALON_GAP_M` |
| `ETALON_GAP_TOLERANCE_M` | 0.010e-3 | m | ICOS build report, §7.4 | Manufacturing tolerance ±0.010 mm |
| `ETALON_R_COATING` | 0.80 | — | ICOS build report, §5.3.1 | As-deposited coating reflectivity at 630 nm |
| `ETALON_R_INSTRUMENT` | 0.53 | — | FlatSat calibration measurement | Effective instrument reflectivity from fringe contrast; same as `R_REFL_FLATSAT` |
| `R_REFL_FLATSAT` | 0.53 | — | FlatSat calibration measurement | Alias for `ETALON_R_INSTRUMENT`; retained for use in M01/M03 fringe model code |
| `ETALON_N` | 1.0 | — | Design (air/vacuum gap) | Refractive index of etalon gap medium |
| `ALPHA_RAD_PX` | 1.6071e-4 | rad/px | Tolansky two-line joint fit (S13 / Z01, 2026-04-21) | **Authoritative plate scale**; 2×2 binned; nominal design value 32 µm / 200 mm = 1.60e-4 rad/px (−0.4 % lower; do not use in fits) |
| `R_MAX_PX` | 110 | px | FlatSat / flight usable radius | Maximum ring radius included in annular reduction (M03) |
| `CCD_PIXEL_UM` | 16.0 | µm | CCD97-00 datasheet | Native pixel pitch, unbinned |
| `CCD_PIXEL_2X2_UM` | 32.0 | µm | Derived (2×2 binning mode) | Effective pixel pitch in nominal science mode |

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
| `SC_VELOCITY_MS` | 7600.0 | m/s | Derived (circular orbit at 510 km) | Approximate |
| `SC_ORBITAL_PERIOD_S` | 5640.0 | s | WC-SE-0003 v8 | ~94 minutes |
| `DEPRESSION_ANGLE_DEG` | **computed** | deg | `compute_depression_angle(SC_ALTITUDE_KM, TP_ALTITUDE_KM)` → 15.79°. See Section 4.1. **LEGACY CORRECTION:** earlier documents hardcoded 23.4° (wrong altitude). |
| `ORBIT_INCLINATION_DEG` | 97.4 | deg | WC-SE-0003 v8 | Sun-synchronous inclination |
| `LTAN_HOURS` | 6.0 | hours | WC-SE-0003 v8 | Local time of ascending node (dawn-dusk) |
| `SCIENCE_CADENCE_S` | 10.0 | s | WC-SE-0003 v8 | Nominal image cadence |

### 3.8 Wind measurement and error budget

| Python symbol | Value | Units | Source | Notes |
|---------------|-------|-------|--------|-------|
| `WIND_BIAS_BUDGET_MS` | 9.8 | m/s | STM v1 | Required 1σ wind precision from Monte Carlo |
| `WIND_MAX_STORM_MS` | 400.0 | m/s | STM v1 | Maximum wind speed to resolve (G2 storm) |
| `WIND_MIN_DETECTABLE_MS` | 20.0 | m/s | STM v1 | 5% of peak storm amplitude |
| `LAT_RANGE_DEG` | (−40.0, 40.0) | deg | STM v1 | Primary science latitude band (SG1+SG2) |

---

## 4. Derived quantities and helper functions

### 4.1 `compute_depression_angle(sc_alt_km, tp_alt_km)`

Computes the limb depression angle from spacecraft and tangent point
altitudes using:

```
δ = arccos(R_tp / R_sc)
```

where `R_sc = WGS84_A_M/1000 + sc_alt_km` and `R_tp = WGS84_A_M/1000 + tp_alt_km`.

The nominal depression angle `DEPRESSION_ANGLE_DEG` is computed at module load
time via `compute_depression_angle(SC_ALTITUDE_KM, TP_ALTITUDE_KM)` → 15.79°.

### 4.2 `_edlen_n(lambda_vac_nm)` — Edlén (1966) refractive index

Returns the refractive index of standard dry air at 15 °C, 101 325 Pa for a
given vacuum wavelength in nm.

```
s = 10⁴ / λ_vac (nm)
(n − 1) × 10⁸ = 8342.13 + 2406030 / (130 − s²) + 15997 / (38.9 − s²)
```

Vacuum-to-air conversion: `λ_air = λ_vac / n`

Air-to-vacuum inversion: iterate `λ_vac ← λ_air × n(_edlen_n(λ_vac))` to
convergence (typically 3–4 iterations; tolerance 1e-10 nm).

This function is used internally to derive all vacuum-wavelength constants
from their air-wavelength primaries.  It is exposed as a module-level
function for use by calibration modules that need to convert arbitrary
wavelengths.

**Reference:** Edlén, B. (1966). *The refractive index of air.* Metrologia,
2(2), 71–80.

### 4.3 Derived spectral and optical quantities

All FSR calculations use **air** wavelengths to preserve consistency with
instrument design-era calculations.

| Symbol | Formula | Approximate value |
|--------|---------|-------------------|
| `ETALON_FSR_NE1_M` | `NE_WAVELENGTH_1_AIR_M² / (2 × ETALON_GAP_M)` | ≈ 10.27 pm |
| `ETALON_FSR_OI_M` | `OI_WAVELENGTH_AIR_M² / (2 × ETALON_GAP_M)` | ≈ 9.90 pm |
| `VELOCITY_PER_FSR_MS` | `c × ETALON_FSR_OI_M / OI_WAVELENGTH_M` | ≈ 4712 m/s |
| `NE_DELTA_LAMBDA_M` | `NE_WAVELENGTH_1_AIR_M − NE_WAVELENGTH_2_AIR_M` | ≈ 1.9257 nm |
| `NE_SEPARATION_FSR` | `NE_DELTA_LAMBDA_M / ETALON_FSR_NE1_M` | ≈ 187.4 |

---

## 5. Quality flags (S04)

```python
class PipelineFlags:
    GOOD           = 0x00
    FIT_FAILED     = 0x01
    CHI2_HIGH      = 0x02
    CHI2_VERY_HIGH = 0x04
    CHI2_LOW       = 0x08
```

Bits 4–15 are module-specific; see S12–S15.

---

## 6. Test suite

### T1 — Speed of light is exact SI value

```python
def test_speed_of_light():
    from src.constants import SPEED_OF_LIGHT_MS
    assert SPEED_OF_LIGHT_MS == 299_792_458.0
```

### T2 — OI vacuum wavelength is self-consistent via Edlén round-trip

```python
def test_oi_wavelength_vac_consistency():
    from src.constants import OI_WAVELENGTH_M, OI_WAVELENGTH_AIR_M, _edlen_n
    # Recover air from vacuum: lambda_air = lambda_vac / n
    lv_nm = OI_WAVELENGTH_M * 1e9
    la_nm = OI_WAVELENGTH_AIR_M * 1e9
    n = _edlen_n(lv_nm)
    la_recovered = lv_nm / n
    assert abs(la_recovered - la_nm) < 1e-4, (
        f"Round-trip error: vac {lv_nm:.6f} nm → air {la_recovered:.6f} nm "
        f"(expected {la_nm:.6f} nm; residual {abs(la_recovered - la_nm)*1e6:.4f} fm)"
    )
    # Vacuum must be shorter than air; shift ~−72 pm
    delta_pm = (lv_nm - la_nm) * 1000
    assert -90 < delta_pm < -60, f"OI vac-air shift = {delta_pm:.1f} pm; expected ~−72 pm"
```

### T3 — Depression angle is computed from primary constants

```python
def test_depression_angle_computed_from_primaries():
    from src.constants import (DEPRESSION_ANGLE_DEG, SC_ALTITUDE_KM,
                                TP_ALTITUDE_KM, compute_depression_angle)
    recomputed = compute_depression_angle(SC_ALTITUDE_KM, TP_ALTITUDE_KM)
    assert abs(DEPRESSION_ANGLE_DEG - recomputed) < 1e-10
    assert abs(DEPRESSION_ANGLE_DEG - 15.79) < 0.02
    assert abs(DEPRESSION_ANGLE_DEG - 23.4) > 1.0
```

### T4 — FSR calculation is self-consistent (uses air wavelengths)

```python
def test_fsr_consistency():
    from src.constants import (ETALON_FSR_OI_M, ETALON_FSR_NE1_M,
                                OI_WAVELENGTH_AIR_M, NE_WAVELENGTH_1_AIR_M,
                                ETALON_GAP_M)
    assert abs(ETALON_FSR_OI_M  - OI_WAVELENGTH_AIR_M**2  / (2 * ETALON_GAP_M)) < 1e-18
    assert abs(ETALON_FSR_NE1_M - NE_WAVELENGTH_1_AIR_M**2 / (2 * ETALON_GAP_M)) < 1e-18
```

### T5 — Neon line separation ≈ 187 FSR

```python
def test_neon_separation_fsr():
    from src.constants import NE_SEPARATION_FSR
    assert 183 < NE_SEPARATION_FSR < 191, \
        f"NE_SEPARATION_FSR = {NE_SEPARATION_FSR:.1f}; expected ≈ 187.4"
```

### T6 — ALPHA_RAD_PX matches Tolansky authoritative value

```python
def test_alpha_rad_px_value():
    from src.constants import ALPHA_RAD_PX
    # Tolansky-recovered value; must match to 4 significant figures
    assert abs(ALPHA_RAD_PX - 1.6071e-4) < 1e-8, \
        f"ALPHA_RAD_PX = {ALPHA_RAD_PX:.6e}; expected 1.6071e-4 rad/px"
    # Sanity: must differ from the nominal design value (no longer used in fits)
    nominal_design = 32e-6 / 0.200
    assert abs(ALPHA_RAD_PX - nominal_design) > 1e-7, \
        "ALPHA_RAD_PX must be Tolansky value, not nominal design value"
```

### T7 — All named constants are float or tuple, not string

```python
def test_constant_types():
    import src.constants as c
    non_tuple_names = [
        'SPEED_OF_LIGHT_MS', 'BOLTZMANN_J_PER_K',
        'OI_WAVELENGTH_M', 'OI_WAVELENGTH_AIR_M',
        'NE_WAVELENGTH_1_M', 'NE_WAVELENGTH_1_AIR_M',
        'NE_WAVELENGTH_2_M', 'NE_WAVELENGTH_2_AIR_M',
        'ETALON_GAP_M', 'ETALON_GAP_ICOS_M', 'ALPHA_RAD_PX',
        'DEPRESSION_ANGLE_DEG', 'SC_ALTITUDE_KM', 'TP_ALTITUDE_KM',
        'WIND_BIAS_BUDGET_MS',
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
        f"VELOCITY_PER_FSR_MS = {VELOCITY_PER_FSR_MS:.0f} m/s; expected ~4712 m/s"
```

### T9 — `compute_depression_angle()` responds correctly to altitude inputs

```python
def test_depression_angle_sensitivity():
    from src.constants import compute_depression_angle
    angle_nominal  = compute_depression_angle(510.0, 250.0)
    angle_low_sc   = compute_depression_angle(500.0, 250.0)
    angle_high_sc  = compute_depression_angle(550.0, 250.0)
    angle_high_tp  = compute_depression_angle(510.0, 300.0)
    assert angle_low_sc  < angle_nominal
    assert angle_high_sc > angle_nominal
    assert angle_high_tp < angle_nominal
    for angle, label in [(angle_nominal, 'nominal'), (angle_low_sc, 'low_sc'),
                          (angle_high_sc, 'high_sc'), (angle_high_tp, 'high_tp')]:
        assert 10.0 < angle < 25.0, \
            f"Depression angle ({label}) = {angle:.2f}° outside plausible range"
```

### T10 — Neon vacuum wavelengths are internally consistent with Edlén

```python
def test_neon_vacuum_wavelengths():
    from src.constants import (NE_WAVELENGTH_1_M, NE_WAVELENGTH_1_AIR_M,
                                NE_WAVELENGTH_2_M, NE_WAVELENGTH_2_AIR_M,
                                _edlen_n)
    for lv_m, la_m, name in [
        (NE_WAVELENGTH_1_M, NE_WAVELENGTH_1_AIR_M, 'Ne1'),
        (NE_WAVELENGTH_2_M, NE_WAVELENGTH_2_AIR_M, 'Ne2'),
    ]:
        la_nm = la_m * 1e9
        lv_nm = lv_m * 1e9
        n = _edlen_n(lv_nm)
        la_recovered = lv_nm / n
        assert abs(la_recovered - la_nm) < 1e-4, (
            f"{name} round-trip residual = {abs(la_recovered - la_nm)*1e6:.4f} fm"
        )
        delta_pm = (lv_nm - la_nm) * 1000
        assert -90 < delta_pm < -70, \
            f"{name} shift = {delta_pm:.1f} pm; expected between −90 and −70 pm"
    # Vacuum wavelengths must be shorter than air wavelengths
    assert NE_WAVELENGTH_1_M < NE_WAVELENGTH_1_AIR_M
    assert NE_WAVELENGTH_2_M < NE_WAVELENGTH_2_AIR_M
    assert OI_WAVELENGTH_M   < OI_WAVELENGTH_AIR_M
```

### T11 — Removed symbols are absent (no deprecated aliases)

```python
def test_no_deprecated_symbols():
    import src.constants as c
    removed = [
        'FOCAL_LENGTH_M',       # removed; ALPHA_RAD_PX is primary
        'PLATE_SCALE_RPX',      # removed; use ALPHA_RAD_PX
        'PLATE_SCALE_RAD_PX',   # removed; use ALPHA_RAD_PX
    ]
    for name in removed:
        assert not hasattr(c, name), \
            f"Deprecated symbol '{name}' should have been removed from constants.py"
```

---

## 7. Expected numerical values

| Symbol | Expected value | Derivation |
|--------|----------------|-----------|
| `OI_WAVELENGTH_M` | 629.9582e-9 m | Edlén (1966) from NIST ASD 630.0304 nm air; Δ = −72.2 pm |
| `OI_WAVELENGTH_AIR_M` | 630.0304e-9 m | NIST ASD air wavelength |
| `NE_WAVELENGTH_1_M` | 640.1426e-9 m | Edlén (1966) from Burns et al. (1950) 640.2248 nm air; Δ = −82.2 pm |
| `NE_WAVELENGTH_1_AIR_M` | 640.2248e-9 m | NIST ASD / Burns et al. (1950) |
| `NE_WAVELENGTH_2_M` | 638.2189e-9 m | Edlén (1966) from Burns et al. (1950) 638.2991 nm air; Δ = −80.2 pm |
| `NE_WAVELENGTH_2_AIR_M` | 638.2991e-9 m | NIST ASD / Burns et al. (1950) |
| `ETALON_GAP_M` | 20.106e-3 m | Tolansky two-line analysis (S13/Z01, 2026-04-21) |
| `ETALON_GAP_ICOS_M` | 20.008e-3 m | ICOS build report spacer; integer disambiguation only |
| `ALPHA_RAD_PX` | 1.6071e-4 rad/px | Tolansky two-line joint fit (S13/Z01, 2026-04-21) |
| `ETALON_FSR_OI_M` | ≈ 9.90 pm | `OI_WAVELENGTH_AIR_M²/(2×ETALON_GAP_M)` at d = 20.106 mm |
| `ETALON_FSR_NE1_M` | ≈ 10.27 pm | `NE_WAVELENGTH_1_AIR_M²/(2×ETALON_GAP_M)` at d = 20.106 mm |
| `NE_SEPARATION_FSR` | ≈ 187.4 | `NE_DELTA_LAMBDA_M / ETALON_FSR_NE1_M` (air wavelengths) |
| `ALPHA_RAD_PX` | 1.6071e-4 rad/px | Tolansky two-line result |
| `DEPRESSION_ANGLE_DEG` | ≈ 15.79° | `compute_depression_angle(510.0, 250.0)` |
| `VELOCITY_PER_FSR_MS` | ≈ 4712 m/s | `c × ETALON_FSR_OI_M / OI_WAVELENGTH_M` (vacuum) |
| `WIND_BIAS_BUDGET_MS` | 9.8 m/s | STM v1 Monte Carlo result |

---

## 8. File location in repository

```
soc_sewell/
├── src/
│   └── constants.py          ← implementation of this spec
└── tests/
    └── test_s03_constants.py ← tests T1–T11 (11 tests)
```

---

## 9. Instructions for Claude Code

This is a **revision** to an already-implemented module. Modify the existing
files — do not create new ones.

### 9.1 Changes to `src/constants.py`

**Step 1 — Update Section 3.3** (OI airglow constants).

Replace the existing Section 3.3 block with:

```python
# ---------------------------------------------------------------------------
# 3.3 Spectroscopic constants — OI airglow target line
# Source: NIST Atomic Spectra Database (NIST ASD)
#
# Vacuum wavelength is canonical (WindCube observes from orbit).
# Air wavelength retained for FSR calculations and Edlén consistency tests.
#
# LEGACY CORRECTIONS:
#   (a) OI_WAVELENGTH_VACUUM_M = 630.0304e-9 was MISLABELLED;
#       630.0304 nm is the AIR wavelength. Symbol removed.
#   (b) OI_WAVELENGTH_VAC_M = 630.2010e-9 was an unrecognised erroneous
#       value. Removed.
# ---------------------------------------------------------------------------
OI_WAVELENGTH_AIR_M = 630.0304e-9      # m — NIST ASD air wavelength; retain for FSR calcs
OI_WAVELENGTH_M     = _air_to_vac_nm(OI_WAVELENGTH_AIR_M * 1e9) * 1e-9
# OI_WAVELENGTH_M ≈ 629.9582e-9 m — canonical vacuum rest wavelength (Edlén 1966)
# Doppler formula (vacuum convention):
# v_rel = SPEED_OF_LIGHT_MS * (lambda_c - OI_WAVELENGTH_M) / OI_WAVELENGTH_M
# Positive v_rel = recession (redshift, source moving away from spacecraft).

OXYGEN_MASS_KG = 2.6567e-26            # kg — one oxygen-16 atom
```

**Step 2 — Update Section 3.4** (neon calibration constants).

Replace the existing Section 3.4 block with:

```python
# ---------------------------------------------------------------------------
# 3.4 Spectroscopic constants — neon calibration lamp
# Source: NIST ASD (Ne I, air wavelengths); Burns, Adams & Longwell (1950)
#
# Vacuum wavelengths are canonical. Air wavelengths retained for FSR /
# beat-period calculations (consistent with FPI calibration convention).
# No deprecated aliases.
# ---------------------------------------------------------------------------
NE_WAVELENGTH_1_AIR_M = 640.2248e-9   # m — primary Ne line, air (Burns 1950 / NIST)
NE_WAVELENGTH_1_M     = _air_to_vac_nm(NE_WAVELENGTH_1_AIR_M * 1e9) * 1e-9
# NE_WAVELENGTH_1_M ≈ 640.1426e-9 m — canonical vacuum; Δ ≈ −82.2 pm

NE_WAVELENGTH_2_AIR_M = 638.2991e-9   # m — secondary Ne line, air (Burns 1950 / NIST)
NE_WAVELENGTH_2_M     = _air_to_vac_nm(NE_WAVELENGTH_2_AIR_M * 1e9) * 1e-9
# NE_WAVELENGTH_2_M ≈ 638.2189e-9 m — canonical vacuum; Δ ≈ −80.2 pm

NE_INTENSITY_1 = 1.0                  # — reference intensity
NE_INTENSITY_2 = 0.8                  # — ratio of secondary to primary
```

**Step 3 — Remove Section 3.4b entirely.**  Delete the lines:

```python
# 3.4b Authoritative gap and F01 calibration constants
D_25C_MM          = 20.008e-3
PLATE_SCALE_RPX   = 1.6000e-4
R_REFL_FLATSAT    = 0.53
```

**Step 4 — Replace Section 3.5** (etalon and optical constants).

Remove `FOCAL_LENGTH_M` and `ALPHA_RAD_PX = CCD_PIXEL_2X2_UM * 1e-6 / FOCAL_LENGTH_M`.
Replace the full Section 3.5 block with:

```python
# ---------------------------------------------------------------------------
# 3.5 Etalon, optical, and calibration constants
# Sources: ICOS build report GNL-4096-R iss1; FlatSat calibration;
#          Tolansky two-line analysis of real neon calibration image
#          (S13 / Z01, 2026-04-21).
#
# Gap note: ETALON_GAP_M is the Tolansky-recovered OPERATIONAL gap.
#   Use it for all FSR and fitting calculations.
#   ETALON_GAP_ICOS_M is the ICOS spacer measurement; use it ONLY to
#   resolve the FSR-period integer N_int = −189. Never substitute for
#   ETALON_GAP_M.
#
# Plate scale note: ALPHA_RAD_PX is hardcoded from the Tolansky two-line
#   joint fit (1.6071e-4 rad/px).  The nominal design value
#   (32 µm / 200 mm = 1.60e-4 rad/px) is ~0.4 % lower and must NOT be
#   used in fitting code.  FOCAL_LENGTH_M is not defined here; the etalon
#   plate scale is more directly and more accurately recovered from the
#   Tolansky analysis than from the COTS lens specification.
# ---------------------------------------------------------------------------
ETALON_GAP_M            = 20.106e-3   # m — Tolansky two-line (S13/Z01); AUTHORITATIVE
ETALON_GAP_ICOS_M       = 20.008e-3   # m — ICOS spacer; integer N_int disambiguation ONLY
ETALON_GAP_TOLERANCE_M  = 0.010e-3    # m — ICOS manufacturing tolerance ±0.010 mm
ETALON_R_COATING        = 0.80        # — — as-deposited coating reflectivity at 630 nm
ETALON_R_INSTRUMENT     = 0.53        # — — effective instrument R from FlatSat fringe contrast
R_REFL_FLATSAT          = ETALON_R_INSTRUMENT   # alias for M01/M03 fringe model code
ETALON_N                = 1.0         # — — refractive index of etalon gap (air/vacuum)
ALPHA_RAD_PX            = 1.6071e-4   # rad/px — Tolansky two-line joint fit; 2×2 binned
#   (nominal design: 32e-6 m / 0.200 m = 1.60e-4 rad/px — do NOT use in fits)
R_MAX_PX                = 110         # px — FlatSat/flight usable radius for M03
CCD_PIXEL_UM            = 16.0        # µm — CCD97-00 native pixel pitch (unbinned)
CCD_PIXEL_2X2_UM        = 32.0        # µm — effective pixel pitch after 2×2 binning
```

**Step 5 — Remove `PLATE_SCALE_RAD_PX` from Section 4** derived quantities.
Delete the line:
```python
PLATE_SCALE_RAD_PX = CCD_PIXEL_2X2_UM * 1e-6 / FOCAL_LENGTH_M
```

**Step 6 — Update `VELOCITY_PER_FSR_MS`** in Section 4 to use the vacuum
rest wavelength:
```python
VELOCITY_PER_FSR_MS = SPEED_OF_LIGHT_MS * ETALON_FSR_OI_M / OI_WAVELENGTH_M
```

### 9.2 Changes to `tests/test_s03_constants.py`

1. Replace `test_oi_wavelength_air_vac_consistency()` with
   `test_oi_wavelength_vac_consistency()` (T2 above).

2. Replace `test_alpha_consistency()` (T6) with
   `test_alpha_rad_px_value()` (T6 above).  Remove the `FOCAL_LENGTH_M`
   import from this test.

3. In `test_constant_types()` (T7): replace `'FOCAL_LENGTH_M'` with
   `'ETALON_GAP_ICOS_M'`; replace `'OI_WAVELENGTH_AIR_M'` and
   `'OI_WAVELENGTH_VAC_M'` with `'OI_WAVELENGTH_M'` and
   `'OI_WAVELENGTH_AIR_M'`; replace `'NE_WAVELENGTH_1_AIR_M'`,
   `'NE_WAVELENGTH_1_VAC_M'`, `'NE_WAVELENGTH_2_AIR_M'`,
   `'NE_WAVELENGTH_2_VAC_M'` with `'NE_WAVELENGTH_1_M'`,
   `'NE_WAVELENGTH_1_AIR_M'`, `'NE_WAVELENGTH_2_M'`,
   `'NE_WAVELENGTH_2_AIR_M'`.

4. Update `test_neon_vacuum_wavelengths()` (T10) to use `NE_WAVELENGTH_1_M`
   and `NE_WAVELENGTH_2_M` (vacuum) as the primary symbols, with `_AIR_M`
   variants as the air reference.

5. Add `test_no_deprecated_symbols()` (T11 above).

6. Do not modify T1, T3, T5, T8, T9.

### 9.3 Verification

```bash
pytest tests/test_s03_constants.py -v
```

All 11 tests must pass before committing.

### 9.4 Commit message

```
feat(constants): vacuum wavelengths canonical; remove FOCAL_LENGTH_M, PLATE_SCALE_* aliases; merge §3.4b into §3.5; update ETALON_GAP_M to Tolansky value

- OI_WAVELENGTH_M, NE_WAVELENGTH_{1,2}_M now carry vacuum values (Edlén 1966)
- _AIR_M variants retained for FSR calculations and Edlén consistency tests
- All deprecated backward-compat aliases removed (OI_WAVELENGTH_VAC_M etc.)
- FOCAL_LENGTH_M removed; ALPHA_RAD_PX = 1.6071e-4 rad/px is now a
  hardcoded primary from Tolansky two-line joint fit (S13/Z01)
- PLATE_SCALE_RPX and PLATE_SCALE_RAD_PX removed; ALPHA_RAD_PX is sole symbol
- Section 3.4b merged into Section 3.5; ETALON_GAP_M updated to 20.106 mm
  (Tolansky operational); ETALON_GAP_ICOS_M = 20.008 mm retained for N_int only
- R_REFL_FLATSAT retained as alias of ETALON_R_INSTRUMENT for M01/M03 code
- VELOCITY_PER_FSR_MS now uses OI_WAVELENGTH_M (vacuum) in denominator
- T6 updated, T11 added (11 tests total)

Implements: S03_physical_constants_2026-05-05.md
```
