# S03 — Physical Constants

**Spec ID:** S03
**Spec file:** `specs/S03_physical_constants_2026-04-29.md`
**Project:** WindCube FPI Science Operations Center Pipeline
**Institution:** NCAR / High Altitude Observatory (HAO)
**Status:** Authoritative
**Depends on:** S01, S02
**Used by:** S05, S06, S07, S08, S09, S10, S11, S12, S13, S14, S15, S16, S17
**Last updated:** 2026-04-29
**Created/Modified by:** Claude AI

**Revision note (2026-04-29):** Section 3.3 and 3.4 expanded to carry both air and
vacuum wavelengths for every spectral line used by the pipeline, with explicit
Edlén (1966) conversion.  Three corrections made:

1. `OI_WAVELENGTH_VACUUM_M` was *mislabelled* — the value 630.0304 nm is the NIST
   ASD **air** wavelength; the Edlén-derived vacuum value is 629.9582 nm.  The
   symbol is renamed `OI_WAVELENGTH_AIR_M` (= 630.0304 nm) and a new symbol
   `OI_WAVELENGTH_VAC_M` (= 629.9582 nm) is introduced.
2. The old `OI_WAVELENGTH_VAC_M = 630.2010e-9` was an unrecognised / erroneous
   value and is removed.
3. New symbols `NE_WAVELENGTH_1_AIR_M`, `NE_WAVELENGTH_1_VAC_M`,
   `NE_WAVELENGTH_2_AIR_M`, `NE_WAVELENGTH_2_VAC_M` replace the previous
   single-wavelength neon constants.  The air values are unchanged (Burns et al.
   1950 / NIST ASD); the vacuum values are Edlén-derived.
4. Section 3.4b gap constants are unchanged.
5. Tests T2 and T10 (new) added for wavelength consistency.

Backward-compatibility note: the legacy symbol `OI_WAVELENGTH_M` is retained
as an alias to `OI_WAVELENGTH_AIR_M` so that existing code that imports
`OI_WAVELENGTH_M` continues to work.  Similarly `NE_WAVELENGTH_1_M` and
`NE_WAVELENGTH_2_M` are retained as aliases.  These aliases are deprecated and
will be removed in a future revision.

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

**Air vs. vacuum wavelengths.** NIST ASD and Burns et al. (1950) report
visible-range spectral lines as **air wavelengths** (standard dry air,
15 °C, 101 325 Pa). Doppler formulae applied to ground-based instruments
operating in air should use air wavelengths. Vacuum wavelengths are required
when comparing to HITRAN/ExoMol databases or ab-initio calculations.
WindCube observes from a vacuum environment (space), so the thermospheric
OI 630 nm photons arrive with their vacuum wavelength; however, all
FPI calibration and inversion modules are written in the air-wavelength
convention adopted at instrument design time, and the rest-wavelength
`OI_WAVELENGTH_AIR_M` (630.0304 nm) must be used consistently unless the
entire chain is converted to vacuum.  Both values are now provided so that
modules can be explicit about which convention they use.

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
science line.  NIST ASD reports the air wavelength; the vacuum wavelength is
derived via the Edlén (1966) formula (see Section 4.2).

| Python symbol | Value | Units | Source | Notes |
|---------------|-------|-------|--------|-------|
| `OI_WAVELENGTH_AIR_M` | 630.0304e-9 | m | NIST ASD (air wavelength) | **Canonical rest wavelength for all Doppler/FSR calculations**; standard dry air 15 °C, 101 325 Pa |
| `OI_WAVELENGTH_VAC_M` | 629.9582e-9 | m | Edlén (1966) from `OI_WAVELENGTH_AIR_M` | Vacuum wavelength; Δ = −72.2 pm relative to air value |
| `OXYGEN_MASS_KG` | 2.6567e-26 | kg | NIST; 16.0 u × 1.66054e-27 kg/u | Mass of one oxygen-16 atom |

**Deprecated aliases (do not use in new code):**

| Alias | Points to | Deprecation note |
|-------|-----------|-----------------|
| `OI_WAVELENGTH_M` | `OI_WAVELENGTH_AIR_M` | Kept for backward compatibility; will be removed in a future spec revision |

**Legacy corrections:**
- `OI_WAVELENGTH_VACUUM_M = 630.0304e-9` was the old name and value; it was
  **mislabelled** (the value 630.0304 nm is the air wavelength).  Symbol removed.
- `OI_WAVELENGTH_VAC_M = 630.2010e-9` was an unrecognised erroneous value.
  Replaced by the Edlén-derived 629.9582 nm.

**Which wavelength to use:**

All Doppler shift calculations use `OI_WAVELENGTH_AIR_M` (630.0304 nm).
The Doppler formula is:
```
v_rel = c × (λ_c - OI_WAVELENGTH_AIR_M) / OI_WAVELENGTH_AIR_M
```
Positive `v_rel` means recession (redshift, source moving away from spacecraft).

Use `OI_WAVELENGTH_VAC_M` only when interfacing with HITRAN/ExoMol databases,
ab-initio transition probabilities, or other vacuum-convention references.

### 3.4 Spectroscopic constants — neon calibration lamp

The two Ne I lines in the 630–640 nm window are used for FPI etalon gap
calibration (Tolansky two-line method).  NIST ASD and Burns et al. (1950)
report air wavelengths; vacuum values are Edlén-derived (Section 4.2).

| Python symbol | Value | Units | Source | Notes |
|---------------|-------|-------|--------|-------|
| `NE_WAVELENGTH_1_AIR_M` | 640.2248e-9 | m | NIST ASD / Burns et al. (1950), Ne I air | Primary line; relative intensity 1.0 |
| `NE_WAVELENGTH_1_VAC_M` | 640.1426e-9 | m | Edlén (1966) from `NE_WAVELENGTH_1_AIR_M` | Vacuum counterpart; Δ = −82.2 pm |
| `NE_WAVELENGTH_2_AIR_M` | 638.2991e-9 | m | NIST ASD / Burns et al. (1950), Ne I air | Secondary line; relative intensity 0.8 |
| `NE_WAVELENGTH_2_VAC_M` | 638.2189e-9 | m | Edlén (1966) from `NE_WAVELENGTH_2_AIR_M` | Vacuum counterpart; Δ = −80.2 pm |
| `NE_INTENSITY_1` | 1.0 | — | NIST ASD | Reference intensity; arbitrary normalisation |
| `NE_INTENSITY_2` | 0.8 | — | NIST ASD | Ratio of secondary to primary line |

**Deprecated aliases (do not use in new code):**

| Alias | Points to | Deprecation note |
|-------|-----------|-----------------|
| `NE_WAVELENGTH_1_M` | `NE_WAVELENGTH_1_AIR_M` | Kept for backward compatibility |
| `NE_WAVELENGTH_2_M` | `NE_WAVELENGTH_2_AIR_M` | Kept for backward compatibility |

Beat period anchor calculation (unchanged; uses air wavelengths throughout,
consistent with FPI calibration convention):
```
Δλ_Ne = NE_WAVELENGTH_1_AIR_M - NE_WAVELENGTH_2_AIR_M = 1.9257e-9 m
FSR   = NE_WAVELENGTH_1_AIR_M² / (2 × t) ≈ 10.24e-12 m at t = 20.008 mm
Number of FSRs between lines ≈ 187.9
```

### 3.4b Authoritative gap and F01 calibration constants

Source: Z01a two-line Tolansky analysis (2026-04-21).

| Python symbol | Value | Units | Notes |
|---------------|-------|-------|-------|
| `D_25C_MM` | 20.0006e-3 | m | ICOS build − Pat/Nir pre-load correction |
| `PLATE_SCALE_RPX` | 1.6071e-4 | rad/px | 2×2 binned Tolansky joint fit result |
| `R_REFL_FLATSAT` | 0.53 | — | FlatSat effective reflectivity |
| `R_MAX_PX` | 110 | px | FlatSat/flight usable radius |

### 3.5 Etalon and optical constants

| Python symbol | Value | Units | Source | Notes |
|---------------|-------|-------|--------|-------|
| `ETALON_GAP_M` | 20.008e-3 | m | ICOS build report GNL-4096-R iss1, §7.4 | Spacer measurement |
| `ETALON_GAP_TOLERANCE_M` | 0.010e-3 | m | ICOS build report, §7.4 | Manufacturing tolerance ±0.010 mm |
| `ETALON_R_COATING` | 0.80 | — | ICOS build report, §5.3.1 | As-deposited coating reflectivity at 630 nm |
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

This function is used internally to derive all `_VAC_M` constants from their
`_AIR_M` primaries.  It is exposed as a module-level function for use by
calibration modules that need to convert arbitrary wavelengths.

**Reference:** Edlén, B. (1966). *The refractive index of air.* Metrologia,
2(2), 71–80.

### 4.3 Derived spectral and optical quantities

| Symbol | Formula | Approximate value |
|--------|---------|-------------------|
| `ETALON_FSR_NE1_M` | `NE_WAVELENGTH_1_AIR_M² / (2 × ETALON_GAP_M)` | ≈ 10.24 pm |
| `ETALON_FSR_OI_M` | `OI_WAVELENGTH_AIR_M² / (2 × ETALON_GAP_M)` | ≈ 9.92 pm |
| `VELOCITY_PER_FSR_MS` | `c × ETALON_FSR_OI_M / OI_WAVELENGTH_AIR_M` | ≈ 4723 m/s |
| `NE_DELTA_LAMBDA_M` | `NE_WAVELENGTH_1_AIR_M − NE_WAVELENGTH_2_AIR_M` | ≈ 1.9257 nm |
| `NE_SEPARATION_FSR` | `NE_DELTA_LAMBDA_M / ETALON_FSR_NE1_M` | ≈ 187.9 |
| `PLATE_SCALE_RAD_PX` | `CCD_PIXEL_2X2_UM × 1e-6 / FOCAL_LENGTH_M` | = `ALPHA_RAD_PX` |

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

### T2 — OI wavelength pair is self-consistent via Edlén

```python
def test_oi_wavelength_air_vac_consistency():
    from src.constants import OI_WAVELENGTH_AIR_M, OI_WAVELENGTH_VAC_M, _edlen_n
    # air → vacuum: lambda_vac = lambda_air * n_air (iterated)
    la_nm = OI_WAVELENGTH_AIR_M * 1e9
    lv_nm = OI_WAVELENGTH_VAC_M * 1e9
    # recover air from vacuum: lambda_air = lambda_vac / n
    n = _edlen_n(lv_nm)
    la_recovered = lv_nm / n
    assert abs(la_recovered - la_nm) < 1e-4, (
        f"Round-trip error: air {la_nm:.6f} nm → vac {lv_nm:.6f} nm → "
        f"air {la_recovered:.6f} nm (residual {abs(la_recovered - la_nm)*1e6:.4f} fm)"
    )
    # Shift is negative and in the range 60–90 pm
    delta_pm = (lv_nm - la_nm) * 1000
    assert -90 < delta_pm < -60, f"OI air-vac shift = {delta_pm:.1f} pm; expected ~−72 pm"
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

### T4 — FSR calculation is self-consistent

```python
def test_fsr_consistency():
    from src.constants import (ETALON_FSR_OI_M, ETALON_FSR_NE1_M,
                                OI_WAVELENGTH_AIR_M, NE_WAVELENGTH_1_AIR_M,
                                ETALON_GAP_M)
    assert abs(ETALON_FSR_OI_M  - OI_WAVELENGTH_AIR_M**2  / (2 * ETALON_GAP_M)) < 1e-18
    assert abs(ETALON_FSR_NE1_M - NE_WAVELENGTH_1_AIR_M**2 / (2 * ETALON_GAP_M)) < 1e-18
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
    assert abs(ALPHA_RAD_PX - expected) < 1e-10
```

### T7 — All named constants are float or tuple, not string

```python
def test_constant_types():
    import src.constants as c
    non_tuple_names = [
        'SPEED_OF_LIGHT_MS', 'BOLTZMANN_J_PER_K',
        'OI_WAVELENGTH_AIR_M', 'OI_WAVELENGTH_VAC_M',
        'NE_WAVELENGTH_1_AIR_M', 'NE_WAVELENGTH_1_VAC_M',
        'NE_WAVELENGTH_2_AIR_M', 'NE_WAVELENGTH_2_VAC_M',
        'ETALON_GAP_M', 'FOCAL_LENGTH_M', 'ALPHA_RAD_PX',
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
        f"VELOCITY_PER_FSR_MS = {VELOCITY_PER_FSR_MS:.0f} m/s; expected ~4723 m/s"
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

### T10 — Neon vacuum wavelengths are consistent with Edlén

```python
def test_neon_vacuum_wavelengths():
    from src.constants import (NE_WAVELENGTH_1_AIR_M, NE_WAVELENGTH_1_VAC_M,
                                NE_WAVELENGTH_2_AIR_M, NE_WAVELENGTH_2_VAC_M,
                                _edlen_n)
    for la_m, lv_m, name in [
        (NE_WAVELENGTH_1_AIR_M, NE_WAVELENGTH_1_VAC_M, 'Ne1'),
        (NE_WAVELENGTH_2_AIR_M, NE_WAVELENGTH_2_VAC_M, 'Ne2'),
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
    assert NE_WAVELENGTH_1_VAC_M < NE_WAVELENGTH_1_AIR_M
    assert NE_WAVELENGTH_2_VAC_M < NE_WAVELENGTH_2_AIR_M
    assert OI_WAVELENGTH_VAC_M   < OI_WAVELENGTH_AIR_M
```

---

## 7. Expected numerical values

| Symbol | Expected value | Derivation |
|--------|----------------|-----------|
| `OI_WAVELENGTH_AIR_M` | 630.0304e-9 m | NIST ASD air wavelength |
| `OI_WAVELENGTH_VAC_M` | 629.9582e-9 m | Edlén (1966); Δ = −72.2 pm |
| `NE_WAVELENGTH_1_AIR_M` | 640.2248e-9 m | NIST ASD / Burns et al. (1950) |
| `NE_WAVELENGTH_1_VAC_M` | 640.1426e-9 m | Edlén (1966); Δ = −82.2 pm |
| `NE_WAVELENGTH_2_AIR_M` | 638.2991e-9 m | NIST ASD / Burns et al. (1950) |
| `NE_WAVELENGTH_2_VAC_M` | 638.2189e-9 m | Edlén (1966); Δ = −80.2 pm |
| `ETALON_GAP_M` | 20.008e-3 m | ICOS build report spacer measurement |
| `ETALON_FSR_OI_M` | ≈ 9.922e-15 m (9.92 pm) | λ_air²/(2t) at OI 630 nm |
| `ETALON_FSR_NE1_M` | ≈ 1.024e-14 m (10.24 pm) | λ_air²/(2t) at Ne 640.2 nm |
| `NE_SEPARATION_FSR` | ≈ 187.9 | Δλ_Ne / FSR_Ne1 (air wavelengths) |
| `ALPHA_RAD_PX` | 1.60e-4 rad/px | 32e-6 m / 0.200 m |
| `DEPRESSION_ANGLE_DEG` | ≈ 15.79° | `compute_depression_angle(510.0, 250.0)` |
| `VELOCITY_PER_FSR_MS` | ≈ 4723 m/s | c × FSR_OI / λ_OI (air) |
| `WIND_BIAS_BUDGET_MS` | 9.8 m/s | STM v1 Monte Carlo result |

---

## 8. File location in repository

```
soc_sewell/
├── src/
│   └── constants.py          ← implementation of this spec
└── tests/
    └── test_s03_constants.py ← tests T1–T10 (10 tests)
```

---

## 9. Instructions for Claude Code

This is a **revision** to an already-implemented module. Modify the existing
files — do not create new ones.

### 9.1 Changes to `src/constants.py`

**Step 1 — Add `_edlen_n()` helper** (insert immediately after the imports block,
before Section 3.1):

```python
def _edlen_n(lambda_vac_nm: float) -> float:
    """
    Refractive index of standard dry air at 15 °C, 101 325 Pa.

    Parameters
    ----------
    lambda_vac_nm : float
        Vacuum wavelength in nanometres.

    Returns
    -------
    float
        n_air (always > 1).

    Reference
    ---------
    Edlén, B. (1966). The refractive index of air. Metrologia, 2(2), 71–80.
    """
    s = 1e4 / lambda_vac_nm           # wavenumber in µm⁻¹
    n_minus_1 = (8342.13
                 + 2406030.0 / (130.0 - s ** 2)
                 + 15997.0  / (38.9  - s ** 2)) * 1e-8
    return 1.0 + n_minus_1


def _air_to_vac_nm(lambda_air_nm: float, tol: float = 1e-10) -> float:
    """
    Convert an air wavelength to vacuum wavelength via Edlén (1966).

    Uses Newton iteration: λ_vac ← λ_air × n(_edlen_n(λ_vac)).
    Converges in 3–4 iterations to femtometre accuracy.

    Parameters
    ----------
    lambda_air_nm : float
        Air wavelength in nanometres.
    tol : float
        Convergence tolerance in nanometres (default 1e-10 nm = 0.1 fm).

    Returns
    -------
    float
        Vacuum wavelength in nanometres.
    """
    lv = lambda_air_nm
    for _ in range(20):
        lv_new = lambda_air_nm * _edlen_n(lv)
        if abs(lv_new - lv) < tol:
            return lv_new
        lv = lv_new
    return lv
```

**Step 2 — Replace Section 3.3** (OI airglow constants block).
Remove the old `OI_WAVELENGTH_M`, `OI_WAVELENGTH_VACUUM_M`, `OI_WAVELENGTH_VAC_M`
assignments and replace with:

```python
# ---------------------------------------------------------------------------
# 3.3 Spectroscopic constants — OI airglow target line
# Source: NIST Atomic Spectra Database (NIST ASD)
#
# NIST ASD and Burns et al. (1950) report air wavelengths for visible lines
# (standard dry air, 15 °C, 101 325 Pa). Vacuum values are Edlén-derived.
#
# LEGACY CORRECTIONS:
#   (a) OI_WAVELENGTH_VACUUM_M = 630.0304e-9 was MISLABELLED;
#       630.0304 nm is the AIR wavelength. Symbol renamed and correct vacuum
#       value (629.9582 nm) added.
#   (b) OI_WAVELENGTH_VAC_M = 630.2010e-9 was an unrecognised erroneous
#       value. Removed.
# ---------------------------------------------------------------------------
OI_WAVELENGTH_AIR_M = 630.0304e-9      # m — NIST ASD air wavelength (canonical)
OI_WAVELENGTH_VAC_M = _air_to_vac_nm(OI_WAVELENGTH_AIR_M * 1e9) * 1e-9
# OI_WAVELENGTH_VAC_M ≈ 629.9582e-9 m; Δ ≈ −72.2 pm

# Deprecated alias — do not use in new code
OI_WAVELENGTH_M = OI_WAVELENGTH_AIR_M  # backward-compat alias

OXYGEN_MASS_KG = 2.6567e-26            # kg — one oxygen-16 atom

# Doppler formula (air convention):
# v_rel = SPEED_OF_LIGHT_MS * (lambda_c - OI_WAVELENGTH_AIR_M) / OI_WAVELENGTH_AIR_M
# Positive v_rel = recession (redshift, source moving away from spacecraft).
```

**Step 3 — Replace Section 3.4** (neon calibration constants block).
Remove the old `NE_WAVELENGTH_1_M`, `NE_WAVELENGTH_2_M`, `NE_INTENSITY_1`,
`NE_INTENSITY_2` assignments and replace with:

```python
# ---------------------------------------------------------------------------
# 3.4 Spectroscopic constants — neon calibration lamp
# Source: NIST ASD (Ne I, air wavelengths); Burns, Adams & Longwell (1950)
#
# Both air and vacuum wavelengths are provided. All FSR / beat-period
# calculations use air wavelengths (consistent with FPI calibration convention).
# ---------------------------------------------------------------------------
NE_WAVELENGTH_1_AIR_M = 640.2248e-9   # m — primary Ne line, air; intensity 1.0
NE_WAVELENGTH_1_VAC_M = _air_to_vac_nm(NE_WAVELENGTH_1_AIR_M * 1e9) * 1e-9
# NE_WAVELENGTH_1_VAC_M ≈ 640.1426e-9 m; Δ ≈ −82.2 pm

NE_WAVELENGTH_2_AIR_M = 638.2991e-9   # m — secondary Ne line, air; intensity 0.8
NE_WAVELENGTH_2_VAC_M = _air_to_vac_nm(NE_WAVELENGTH_2_AIR_M * 1e9) * 1e-9
# NE_WAVELENGTH_2_VAC_M ≈ 638.2189e-9 m; Δ ≈ −80.2 pm

NE_INTENSITY_1 = 1.0                  # — reference intensity
NE_INTENSITY_2 = 0.8                  # — ratio of secondary to primary

# Deprecated aliases — do not use in new code
NE_WAVELENGTH_1_M = NE_WAVELENGTH_1_AIR_M
NE_WAVELENGTH_2_M = NE_WAVELENGTH_2_AIR_M
```

**Step 4 — Update Section 4 derived quantities** that reference old symbols.
Change every occurrence of `OI_WAVELENGTH_M` → `OI_WAVELENGTH_AIR_M` and
`NE_WAVELENGTH_1_M` → `NE_WAVELENGTH_1_AIR_M` in the derived-quantity
computations (`ETALON_FSR_OI_M`, `ETALON_FSR_NE1_M`, `VELOCITY_PER_FSR_MS`,
`NE_DELTA_LAMBDA_M`, `NE_SEPARATION_FSR`).

### 9.2 Changes to `tests/test_s03_constants.py`

1. In `test_constant_types()` (T7): replace `'OI_WAVELENGTH_M'` with
   `'OI_WAVELENGTH_AIR_M'`, `'OI_WAVELENGTH_VAC_M'`, `'NE_WAVELENGTH_1_AIR_M'`,
   `'NE_WAVELENGTH_1_VAC_M'`, `'NE_WAVELENGTH_2_AIR_M'`, `'NE_WAVELENGTH_2_VAC_M'`.
2. Add `test_oi_wavelength_air_vac_consistency()` (T2 above).
3. Add `test_neon_vacuum_wavelengths()` (T10 above).
4. In `test_fsr_consistency()` (T4): update imports to use `OI_WAVELENGTH_AIR_M`
   and `NE_WAVELENGTH_1_AIR_M`.
5. Do not modify T1, T3, T5, T6, T8, T9.

### 9.3 Verification

```bash
pytest tests/test_s03_constants.py -v
```

All 10 tests must pass before committing.

### 9.4 Commit message

```
feat(constants): add air+vacuum wavelengths for OI 630 nm and Ne lines (Edlén 1966)

- Rename OI_WAVELENGTH_M → OI_WAVELENGTH_AIR_M (was mislabelled as vacuum)
- Add OI_WAVELENGTH_VAC_M = 629.9582 nm (Edlén-derived)
- Remove erroneous OI_WAVELENGTH_VAC_M = 630.2010e-9
- Add NE_WAVELENGTH_{1,2}_{AIR,VAC}_M with Edlén vacuum companions
- Add _edlen_n() and _air_to_vac_nm() helper functions
- Backward-compat aliases retained: OI_WAVELENGTH_M, NE_WAVELENGTH_{1,2}_M
- Tests T2 and T10 added (10 tests total)

Implements: S03_physical_constants_2026-04-29.md

Also updates PIPELINE_STATUS.md
```
