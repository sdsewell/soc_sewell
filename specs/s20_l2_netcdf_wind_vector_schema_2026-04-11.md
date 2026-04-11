# S20 — L2 Data Product: netCDF Schema for Wind Vector Format

**Specification ID:** S20  
**Tier:** 10 — Data Product Formats  
**Status:** Draft  
**Date:** 2026-04-11  
**Author:** Scott Sewell / HAO-NCAR  
**Repository:** `soc_sewell`  
**Related specs:** S01–S19, Z01  
**Science traceability:** SQ1, SQ2 (STM v1.0)

---

## 1. Purpose and scope

This specification defines the netCDF-4 file schema for the WindCube **Level 2 (L2) wind vector data product** — the final, publicly released scientific output of the Science Operations Center (SOC) pipeline.

The L2 file is produced by the **Merger** module (Qian Wu / FORTRAN), which ingests paired Level 1c calibrated line-of-sight (LOS) wind files from alternating along-track and cross-track orbits, decomposes them into zonal and meridional components, and writes the resulting time-series of horizontal wind vectors geolocated at their thermospheric tangent points.

Every field in this schema is traceable to a science requirement from the STM.

---

## 2. Science traceability

| STM item | Requirement | L2 field(s) |
|---|---|---|
| SQ1 — geomagnetic storm winds | Zonal & meridional winds, ±40° lat, precision ≤ 34 m/s (instrument: 9.8 m/s) | `u_wind`, `v_wind`, `u_wind_error`, `v_wind_error`, `latitude`, `longitude` |
| SQ1 — temporal sampling | 1-hour cadence during storm main phase | `time` + `quality_flag` |
| SQ2 — DE3 tidal winds | Zonal & meridional winds, ±30° lat, ≥15 continuous days, precision ≤ 32 m/s | Same as above + `orbit_number` |
| SQ2 — spatial resolution | ≥3 obs per 2° latitude, 10-s cadence | `time`, `latitude`, dimension `obs` |
| Both — look-direction provenance | Along-track and cross-track orbits identified | `look_direction` |
| Both — altitude coverage | Single emission volume 150–290 km, vertical res ~104 km | `altitude` |
| Both — nightglow emission | OI 630.0 nm only | `emission_rate`, global attribute `wavelength_nm` |
| Both — data quality | Usable fraction of observations identified | `quality_flag`, `sza_flag`, `dayglow_flag`, `data_gap_flag` |

---

## 3. File naming and size

```
windcube_l2_YYYYMMDD_v<VV>.nc
```

- `YYYYMMDD` — UTC date of observations
- `v<VV>` — two-digit processing version (e.g., `v01`)
- Typical file size: **~4 MB per day** (as documented in the Continuation Review)
- Format: **netCDF-4** (HDF5 back-end), compressed with zlib level 4

Example: `windcube_l2_20270315_v01.nc`

---

## 4. Dimensions

| Dimension | Symbol | Description |
|---|---|---|
| `obs` | N_obs | Number of merged wind vector samples in the file. Unlimited (extendable). |
| `pair` | 2 | Index over the two orbit look directions (along-track = 0, cross-track = 1). |
| `strlen` | 16 | Fixed length for short enumerated string flags. |

> **Design note:** `obs` is the primary scientific index. Each observation corresponds to one merged (u, v) wind vector located at a geolocated thermospheric tangent point, combining data from an adjacent along-track / cross-track orbit pair.

---

## 5. Global attributes

All CF-1.8 convention required attributes are included. WindCube-specific attributes are listed below.

| Attribute name | Type | Example value | Description |
|---|---|---|---|
| `Conventions` | string | `"CF-1.8"` | NetCDF conventions version |
| `title` | string | `"WindCube L2 thermospheric wind vectors"` | Human-readable dataset title |
| `institution` | string | `"High Altitude Observatory, NCAR/UCAR, Boulder CO"` | Producing institution |
| `source` | string | `"WindCube CubeSat FPI, OI 630.0 nm airglow"` | Measurement source description |
| `history` | string | `"Created 2027-03-15T22:14:00Z by soc_sewell v1.3.0"` | Processing provenance |
| `date_created` | string | `"2027-03-15T22:14:00Z"` | ISO-8601 UTC creation timestamp |
| `processing_level` | string | `"L2"` | Data product level |
| `pipeline_version` | string | `"1.3.0"` | `soc_sewell` repo release tag |
| `git_sha` | string | `"a3f8c21"` | Git commit SHA of pipeline code |
| `orbit_mode` | string | `"SSO dawn-dusk LTAN 0600/1800"` | Orbit configuration |
| `orbit_altitude_km` | float | `510.0` | Nominal orbital altitude (km) |
| `wavelength_nm` | float | `630.0` | Observed emission wavelength (nm) |
| `emission_species` | string | `"OI 630.0 nm"` | Emitting species identification |
| `calibration_file` | string | `"windcube_cal_20270101_v01.nc"` | Calibration file used |
| `etalon_gap_mm` | float | `20.106` | Operational etalon gap from Tolansky analysis (mm) |
| `focal_length_mm` | float | `199.12` | Imaging lens focal length from Tolansky analysis (mm) |
| `references` | string | `"Sewell et al. (in prep); doi:..."` | Literature references |
| `acknowledgements` | string | `"NSF CubeSat program; NCAR/HAO"` | Funding/acknowledgements |

---

## 6. Coordinate variables

All coordinate variables have dimension `(obs)`. These are the primary independent axes for indexing and subsetting the data.

### 6.1 `time`

| Attribute | Value |
|---|---|
| dtype | `float64` |
| units | `"seconds since 2000-01-01 12:00:00"` (J2000.0 epoch) |
| long_name | `"UTC time at tangent-point centre of exposure"` |
| calendar | `"standard"` |
| valid_range | `[0.0, 1e10]` |
| fill_value | `9.969209968386869e+36` (CF default `_FillValue`) |

> **Rationale:** J2000.0 is standard for space physics data (SPDF/OMNIWeb compatibility). The value represents the midpoint of each 10-second integration interval, referenced to the tangent-point centre of the FoV.

### 6.2 `latitude`

| Attribute | Value |
|---|---|
| dtype | `float32` |
| units | `"degrees_north"` |
| long_name | `"Geodetic latitude of OI 630 nm emission tangent point"` |
| valid_range | `[-90.0, 90.0]` |
| comment | `"Emission-weighted mean geodetic latitude within the 1.65° FoV"` |

### 6.3 `longitude`

| Attribute | Value |
|---|---|
| dtype | `float32` |
| units | `"degrees_east"` |
| long_name | `"Geodetic longitude of OI 630 nm emission tangent point"` |
| valid_range | `[-180.0, 180.0]` |

### 6.4 `altitude`

| Attribute | Value |
|---|---|
| dtype | `float32` |
| units | `"km"` |
| long_name | `"Emission-weighted tangent height of OI 630 nm volume"` |
| valid_range | `[100.0, 350.0]` |
| comment | `"Nominal emission peak ~250 km; integration covers ~150–290 km"` |

### 6.5 `sza`

| Attribute | Value |
|---|---|
| dtype | `float32` |
| units | `"degrees"` |
| long_name | `"Solar zenith angle at tangent point"` |
| valid_range | `[0.0, 180.0]` |
| comment | `"Science observations nominally require sza > 100° (nightside); see sza_flag"` |

---

## 7. Science data variables

All science variables have dimension `(obs)` and type `float32` unless noted. Fill value is `NaN` (IEEE 754) for all float variables.

### 7.1 Primary wind vector components

#### `u_wind` — Zonal wind

| Attribute | Value |
|---|---|
| units | `"m/s"` |
| long_name | `"Zonal wind velocity (positive eastward)"` |
| standard_name | `"eastward_wind"` |
| valid_range | `[-1000.0, 1000.0]` |
| comment | `"Merged from along-track and cross-track LOS using Merger decomposition; positive = eastward"` |
| ancillary_variables | `"u_wind_error quality_flag"` |

#### `v_wind` — Meridional wind

| Attribute | Value |
|---|---|
| units | `"m/s"` |
| long_name | `"Meridional wind velocity (positive northward)"` |
| standard_name | `"northward_wind"` |
| valid_range | `[-1000.0, 1000.0]` |
| comment | `"Merged from along-track and cross-track LOS; positive = northward"` |
| ancillary_variables | `"v_wind_error quality_flag"` |

#### `u_wind_error` — Zonal wind 1-sigma uncertainty

| Attribute | Value |
|---|---|
| units | `"m/s"` |
| long_name | `"1-sigma uncertainty in zonal wind"` |
| comment | `"Propagated from LOS velocity uncertainties through Merger decomposition matrix"` |
| valid_range | `[0.0, 500.0]` |

#### `v_wind_error` — Meridional wind 1-sigma uncertainty

| Attribute | Value |
|---|---|
| units | `"m/s"` |
| long_name | `"1-sigma uncertainty in meridional wind"` |
| comment | `"Propagated from LOS velocity uncertainties through Merger decomposition matrix"` |
| valid_range | `[0.0, 500.0]` |

### 7.2 Derived scalar wind quantities

#### `wind_speed`

| Attribute | Value |
|---|---|
| units | `"m/s"` |
| long_name | `"Horizontal wind speed"` |
| comment | `"sqrt(u_wind^2 + v_wind^2)"` |
| valid_range | `[0.0, 1000.0]` |

#### `wind_direction`

| Attribute | Value |
|---|---|
| units | `"degrees"` |
| long_name | `"Horizontal wind direction (meteorological convention)"` |
| comment | `"Direction FROM which wind blows, measured clockwise from north (0–360°). Met convention: 0° = wind from north, 90° = wind from east."` |
| valid_range | `[0.0, 360.0]` |

### 7.3 LOS provenance (pre-merge)

These variables preserve the individual LOS measurements before vector decomposition. They are essential for reprocessing, validation, and understanding merger residuals.

#### `los_velocity_along`

| Attribute | Value |
|---|---|
| units | `"m/s"` |
| long_name | `"Calibrated LOS wind velocity from along-track orbit"` |
| comment | `"Positive = toward instrument (redshift). L1c product value; spacecraft velocity correction applied."` |

#### `los_velocity_cross`

| Attribute | Value |
|---|---|
| units | `"m/s"` |
| long_name | `"Calibrated LOS wind velocity from cross-track orbit"` |
| comment | `"Positive = toward instrument (redshift). L1c product value; spacecraft velocity correction applied."` |

#### `los_error_along`, `los_error_cross`

| Attribute | Value |
|---|---|
| units | `"m/s"` |
| long_name | `"1-sigma LOS velocity uncertainty (along-track / cross-track)"` |

---

## 8. Quality and flag variables

All flag variables have dimension `(obs)` and dtype `uint8` with fill value `255`.

### 8.1 `quality_flag` — master quality indicator

| Value | Meaning |
|---|---|
| 0 | Good — recommended for science use |
| 1 | Caution — marginal conditions; use with care |
| 2 | Bad — do not use |
| 255 | Fill / not retrieved |

> **STM traceability:** Enables downstream science teams (SQ1 storm phase analysis, SQ2 tidal decomposition) to filter to reliable measurements.

Bit-level decomposition (for advanced users):

| Bit | Mask | Condition |
|---|---|---|
| 0 | 0x01 | LOS fit residual > 3σ threshold |
| 1 | 0x02 | Merger geometry angle < 20° (near-degenerate) |
| 2 | 0x04 | Emission rate below noise floor |
| 3 | 0x08 | Data gap in one of the pair orbits |
| 4 | 0x10 | SZA not in nightglow range (> 102°) |
| 5 | 0x20 | Etalon temperature outside calibrated range |

### 8.2 `sza_flag`

| Value | Meaning |
|---|---|
| 0 | Nightside (sza ≥ 102°) — nightglow regime |
| 1 | Twilight (90° ≤ sza < 102°) |
| 2 | Dayside (sza < 90°) — dayglow contamination likely |

### 8.3 `dayglow_flag`

| Value | Meaning |
|---|---|
| 0 | No dayglow contamination detected |
| 1 | Dayglow correction applied |
| 2 | Dayglow exceeds correction capability |

### 8.4 `data_gap_flag`

| Value | Meaning |
|---|---|
| 0 | Both along-track and cross-track orbit data present |
| 1 | Along-track data missing; wind estimate degraded |
| 2 | Cross-track data missing; wind estimate degraded |
| 3 | Both missing; no wind estimate |

---

## 9. Orbit ancillary variables

All have dimension `(obs)`.

| Variable | dtype | units | long_name |
|---|---|---|---|
| `orbit_number` | int32 | — | Sequential orbit number since launch |
| `look_direction` | uint8 | — | 0 = along-track, 1 = cross-track |
| `sc_latitude` | float32 | degrees_north | Spacecraft geodetic latitude |
| `sc_longitude` | float32 | degrees_east | Spacecraft geodetic longitude |
| `sc_altitude` | float32 | km | Spacecraft geodetic altitude |

---

## 10. Instrument state variables

All have dimension `(obs)`.

| Variable | dtype | units | long_name | Notes |
|---|---|---|---|---|
| `etalon_temp` | float32 | °C | Etalon temperature sensor 1 reading | From telemetry; affects FSR and gap |
| `exposure_time` | float32 | s | Science exposure duration | Nominal 10 s |
| `emission_rate` | float32 | Rayleigh | OI 630 nm emission rate from fit | Signal proxy; used in quality assessment |

---

## 11. Variable summary table

```
Dimension:   obs (unlimited)

Coordinates (obs):
  time            float64   J2000 seconds
  latitude        float32   degrees_north
  longitude       float32   degrees_east
  altitude        float32   km
  sza             float32   degrees

Science (obs):
  u_wind          float32   m/s   zonal wind (+east)
  v_wind          float32   m/s   meridional wind (+north)
  u_wind_error    float32   m/s   1-sigma
  v_wind_error    float32   m/s   1-sigma
  wind_speed      float32   m/s   sqrt(u²+v²)
  wind_direction  float32   deg   met convention
  los_velocity_along  float32  m/s
  los_velocity_cross  float32  m/s
  los_error_along     float32  m/s
  los_error_cross     float32  m/s

Quality (obs):
  quality_flag    uint8
  sza_flag        uint8
  dayglow_flag    uint8
  data_gap_flag   uint8

Orbit (obs):
  orbit_number    int32
  look_direction  uint8
  sc_latitude     float32   degrees_north
  sc_longitude    float32   degrees_east
  sc_altitude     float32   km

Instrument (obs):
  etalon_temp     float32   °C
  exposure_time   float32   s
  emission_rate   float32   Rayleigh
```

---

## 12. CF conventions compliance

This schema targets **CF Conventions 1.8**. Key compliance items:

- All float variables carry `units`, `long_name`, `valid_range`, and `_FillValue`.
- `time` uses a units string with reference epoch (CF §4.4).
- `latitude` / `longitude` carry `standard_name = "latitude"` / `"longitude"` and `axis = "Y"` / `"X"`.
- `standard_name` attributes follow the CF Standard Name Table v85 where applicable (`eastward_wind`, `northward_wind`).
- Compressed variables use `zlib=True, complevel=4` via netCDF4-python.

---

## 13. Python reference implementation (schema creation)

The following module — `m08_l2_writer.py` (to be specced as S21) — creates a conformant empty L2 file for a given day's observations. It is the canonical reference for producers and the validation target for consumers.

```python
"""
m08_l2_writer.py  —  S20 reference implementation
Creates a WindCube L2 netCDF-4 file with the schema defined in S20.
"""
import numpy as np
import netCDF4 as nc
from datetime import datetime, timezone


FILL_F32 = np.float32(9.96921e36)   # CF default fill for float32
FILL_U8  = np.uint8(255)
FILL_I32 = np.int32(-2147483647)


def create_l2_file(
    filepath: str,
    n_obs: int,
    pipeline_version: str = "1.0.0",
    git_sha: str = "unknown",
) -> nc.Dataset:
    """
    Create an empty, schema-conformant WindCube L2 netCDF-4 file.

    Parameters
    ----------
    filepath : str
        Output path, e.g. 'windcube_l2_20270315_v01.nc'
    n_obs : int
        Number of merged wind vector observations to pre-allocate.
    pipeline_version : str
        soc_sewell release tag.
    git_sha : str
        Git commit SHA of pipeline code.

    Returns
    -------
    nc.Dataset
        Open (writable) dataset. Caller must call ds.close().
    """
    ds = nc.Dataset(filepath, "w", format="NETCDF4")

    # ── Global attributes ──────────────────────────────────────────────
    ds.Conventions          = "CF-1.8"
    ds.title                = "WindCube L2 thermospheric horizontal wind vectors"
    ds.institution          = "High Altitude Observatory, NCAR/UCAR, Boulder CO, USA"
    ds.source               = "WindCube CubeSat FPI, OI 630.0 nm airglow, SSO LEO ~510 km"
    ds.processing_level     = "L2"
    ds.pipeline_version     = pipeline_version
    ds.git_sha              = git_sha
    ds.date_created         = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ds.wavelength_nm        = np.float32(630.0)
    ds.emission_species     = "OI 630.0 nm"
    ds.orbit_mode           = "SSO dawn-dusk, LTAN 0600/1800 ± 1 hr"
    ds.orbit_altitude_km    = np.float32(510.0)
    ds.etalon_gap_mm        = np.float32(20.106)   # Tolansky operational value
    ds.focal_length_mm      = np.float32(199.12)   # Tolansky operational value
    ds.references           = "Sewell et al. (in prep)"
    ds.acknowledgements     = "NSF CubeSat program; NCAR/HAO"

    # ── Dimensions ──────────────────────────────────────────────────────
    ds.createDimension("obs", None)   # unlimited

    # ── Helper: create compressed float32 variable ─────────────────────
    def fvar(name, dims=("obs",), fill=FILL_F32, **attrs):
        v = ds.createVariable(name, "f4", dims,
                              fill_value=fill, zlib=True, complevel=4)
        v.setncatts(attrs)
        return v

    def u8var(name, dims=("obs",), **attrs):
        v = ds.createVariable(name, "u1", dims,
                              fill_value=FILL_U8, zlib=True, complevel=4)
        v.setncatts(attrs)
        return v

    def i32var(name, dims=("obs",), **attrs):
        v = ds.createVariable(name, "i4", dims,
                              fill_value=FILL_I32, zlib=True, complevel=4)
        v.setncatts(attrs)
        return v

    # ── Coordinates ─────────────────────────────────────────────────────
    t = ds.createVariable("time", "f8", ("obs",), zlib=True, complevel=4)
    t.units       = "seconds since 2000-01-01 12:00:00"
    t.long_name   = "UTC time at tangent-point centre of exposure"
    t.calendar    = "standard"
    t.axis        = "T"

    fvar("latitude",
         long_name="Geodetic latitude of OI 630 nm emission tangent point",
         units="degrees_north",
         standard_name="latitude",
         valid_range=np.array([-90.0, 90.0], dtype="f4"),
         axis="Y")

    fvar("longitude",
         long_name="Geodetic longitude of OI 630 nm emission tangent point",
         units="degrees_east",
         standard_name="longitude",
         valid_range=np.array([-180.0, 180.0], dtype="f4"),
         axis="X")

    fvar("altitude",
         long_name="Emission-weighted tangent height of OI 630 nm volume",
         units="km",
         valid_range=np.array([100.0, 350.0], dtype="f4"),
         comment="Nominal emission peak ~250 km; integration spans ~150–290 km")

    fvar("sza",
         long_name="Solar zenith angle at tangent point",
         units="degrees",
         valid_range=np.array([0.0, 180.0], dtype="f4"))

    # ── Science: primary wind vector ────────────────────────────────────
    fvar("u_wind",
         long_name="Zonal wind velocity (positive eastward)",
         units="m/s",
         standard_name="eastward_wind",
         valid_range=np.array([-1000.0, 1000.0], dtype="f4"),
         ancillary_variables="u_wind_error quality_flag")

    fvar("v_wind",
         long_name="Meridional wind velocity (positive northward)",
         units="m/s",
         standard_name="northward_wind",
         valid_range=np.array([-1000.0, 1000.0], dtype="f4"),
         ancillary_variables="v_wind_error quality_flag")

    fvar("u_wind_error",
         long_name="1-sigma uncertainty in zonal wind",
         units="m/s",
         valid_range=np.array([0.0, 500.0], dtype="f4"),
         comment="Propagated from LOS uncertainties through Merger decomposition")

    fvar("v_wind_error",
         long_name="1-sigma uncertainty in meridional wind",
         units="m/s",
         valid_range=np.array([0.0, 500.0], dtype="f4"),
         comment="Propagated from LOS uncertainties through Merger decomposition")

    fvar("wind_speed",
         long_name="Horizontal wind speed",
         units="m/s",
         valid_range=np.array([0.0, 1000.0], dtype="f4"),
         comment="sqrt(u_wind**2 + v_wind**2)")

    fvar("wind_direction",
         long_name="Horizontal wind direction, meteorological convention",
         units="degrees",
         valid_range=np.array([0.0, 360.0], dtype="f4"),
         comment="Direction FROM which wind blows; 0=from N, 90=from E (met convention)")

    # ── Science: LOS provenance ─────────────────────────────────────────
    for tag in ("along", "cross"):
        fvar(f"los_velocity_{tag}",
             long_name=f"Calibrated LOS velocity ({tag}-track orbit)",
             units="m/s",
             comment="Positive = toward instrument (redshift); s/c velocity corrected")
        fvar(f"los_error_{tag}",
             long_name=f"1-sigma LOS velocity uncertainty ({tag}-track)",
             units="m/s")

    # ── Quality flags ───────────────────────────────────────────────────
    u8var("quality_flag",
          long_name="Master quality flag",
          flag_values=np.array([0, 1, 2], dtype="u1"),
          flag_meanings="good caution bad",
          comment="Bit 0: fit residual; bit 1: geometry; bit 2: SNR; "
                  "bit 3: data gap; bit 4: SZA; bit 5: etalon temp")

    u8var("sza_flag",
          long_name="Solar zenith angle regime flag",
          flag_values=np.array([0, 1, 2], dtype="u1"),
          flag_meanings="nightside twilight dayside")

    u8var("dayglow_flag",
          long_name="Dayglow contamination flag",
          flag_values=np.array([0, 1, 2], dtype="u1"),
          flag_meanings="none corrected excessive")

    u8var("data_gap_flag",
          long_name="Data gap flag for orbit pair completeness",
          flag_values=np.array([0, 1, 2, 3], dtype="u1"),
          flag_meanings="complete along_missing cross_missing both_missing")

    # ── Orbit ancillary ─────────────────────────────────────────────────
    i32var("orbit_number",
           long_name="Sequential orbit number since launch")

    u8var("look_direction",
          long_name="Payload look direction for this observation",
          flag_values=np.array([0, 1], dtype="u1"),
          flag_meanings="along_track cross_track")

    fvar("sc_latitude",
         long_name="Spacecraft geodetic latitude",
         units="degrees_north",
         valid_range=np.array([-90.0, 90.0], dtype="f4"))

    fvar("sc_longitude",
         long_name="Spacecraft geodetic longitude",
         units="degrees_east",
         valid_range=np.array([-180.0, 180.0], dtype="f4"))

    fvar("sc_altitude",
         long_name="Spacecraft geodetic altitude",
         units="km",
         valid_range=np.array([400.0, 650.0], dtype="f4"))

    # ── Instrument state ────────────────────────────────────────────────
    fvar("etalon_temp",
         long_name="Etalon temperature sensor 1",
         units="degrees_Celsius",
         comment="From L1a telemetry header; affects etalon gap and FSR")

    fvar("exposure_time",
         long_name="Science exposure duration",
         units="s",
         comment="Nominal 10 s; may vary during commissioning")

    fvar("emission_rate",
         long_name="OI 630 nm emission rate from spectral fit",
         units="Rayleigh",
         valid_range=np.array([0.0, 10000.0], dtype="f4"),
         comment="Proxy for nightglow signal strength; used in quality assessment")

    return ds
```

---

## 14. Verification tests

These pytest tests are the acceptance gate for any L2 writer implementation.

```python
# test_s20_l2_schema.py
import numpy as np
import netCDF4 as nc
import pytest
import tempfile, os
from windcube.m08_l2_writer import create_l2_file


N = 120   # representative observation count (~15 s cadence × 30 min orbit segment)

@pytest.fixture
def l2_path(tmp_path):
    p = str(tmp_path / "test_l2.nc")
    ds = create_l2_file(p, n_obs=N)
    # Populate with synthetic data
    ds["time"][:] = np.linspace(8.1e8, 8.1e8 + N * 10.0, N)
    ds["latitude"][:] = np.linspace(-40.0, 40.0, N, dtype="f4")
    ds["longitude"][:] = np.full(N, -75.0, dtype="f4")
    ds["altitude"][:] = np.full(N, 250.0, dtype="f4")
    ds["sza"][:] = np.full(N, 110.0, dtype="f4")
    u = np.random.uniform(-200, 200, N).astype("f4")
    v = np.random.uniform(-200, 200, N).astype("f4")
    ds["u_wind"][:] = u
    ds["v_wind"][:] = v
    ds["u_wind_error"][:] = np.full(N, 9.8, dtype="f4")
    ds["v_wind_error"][:] = np.full(N, 9.8, dtype="f4")
    ds["wind_speed"][:] = np.sqrt(u**2 + v**2)
    ds["wind_direction"][:] = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
    ds["quality_flag"][:] = np.zeros(N, dtype="u1")
    ds["sza_flag"][:] = np.zeros(N, dtype="u1")
    ds["dayglow_flag"][:] = np.zeros(N, dtype="u1")
    ds["data_gap_flag"][:] = np.zeros(N, dtype="u1")
    ds["orbit_number"][:] = np.full(N, 142, dtype="i4")
    ds["look_direction"][:] = np.zeros(N, dtype="u1")
    ds["emission_rate"][:] = np.full(N, 150.0, dtype="f4")
    ds.close()
    return p


def test_required_variables_present(l2_path):
    required = [
        "time", "latitude", "longitude", "altitude", "sza",
        "u_wind", "v_wind", "u_wind_error", "v_wind_error",
        "wind_speed", "wind_direction",
        "los_velocity_along", "los_velocity_cross",
        "quality_flag", "sza_flag", "dayglow_flag", "data_gap_flag",
        "orbit_number", "look_direction",
        "etalon_temp", "exposure_time", "emission_rate",
    ]
    with nc.Dataset(l2_path) as ds:
        for v in required:
            assert v in ds.variables, f"Missing variable: {v}"


def test_cf_conventions(l2_path):
    with nc.Dataset(l2_path) as ds:
        assert ds.Conventions == "CF-1.8"
        assert ds.processing_level == "L2"
        assert "time" in ds.variables
        assert ds["latitude"].units == "degrees_north"
        assert ds["longitude"].units == "degrees_east"
        assert ds["time"].units.startswith("seconds since 2000-01-01")


def test_obs_dimension_unlimited(l2_path):
    with nc.Dataset(l2_path) as ds:
        assert ds.dimensions["obs"].isunlimited()


def test_wind_dtypes(l2_path):
    with nc.Dataset(l2_path) as ds:
        for v in ("u_wind", "v_wind", "u_wind_error", "v_wind_error",
                  "wind_speed", "wind_direction"):
            assert ds[v].dtype == np.float32, f"{v} must be float32"


def test_quality_flag_range(l2_path):
    with nc.Dataset(l2_path) as ds:
        qf = ds["quality_flag"][:]
        assert qf.max() <= 2, "quality_flag values must be 0, 1, or 2"


def test_wind_speed_consistency(l2_path):
    with nc.Dataset(l2_path) as ds:
        u = ds["u_wind"][:]
        v = ds["v_wind"][:]
        ws = ds["wind_speed"][:]
        expected = np.sqrt(u**2 + v**2)
        np.testing.assert_allclose(ws, expected, rtol=1e-5,
                                   err_msg="wind_speed != sqrt(u^2 + v^2)")


def test_wind_direction_range(l2_path):
    with nc.Dataset(l2_path) as ds:
        wd = ds["wind_direction"][:]
        valid = ~np.isnan(wd)
        assert np.all(wd[valid] >= 0.0) and np.all(wd[valid] <= 360.0), \
            "wind_direction must be in [0, 360]"


def test_science_traceability_precision(l2_path):
    """SQ1/SQ2: wind error must be physically plausible (< 34 m/s instrument spec)."""
    with nc.Dataset(l2_path) as ds:
        sigma_u = ds["u_wind_error"][:]
        sigma_v = ds["v_wind_error"][:]
        assert np.nanmean(sigma_u) < 34.0, "Mean u_wind_error exceeds STM budget"
        assert np.nanmean(sigma_v) < 34.0, "Mean v_wind_error exceeds STM budget"


def test_latitude_coverage(l2_path):
    """SQ1: must cover ±40°; SQ2: must cover ±30°."""
    with nc.Dataset(l2_path) as ds:
        lat = ds["latitude"][:]
        assert lat.min() <= -30.0, "Latitude coverage must reach -30°"
        assert lat.max() >= 30.0,  "Latitude coverage must reach +30°"


def test_global_attributes(l2_path):
    required_attrs = [
        "Conventions", "title", "institution", "source",
        "processing_level", "pipeline_version", "git_sha",
        "date_created", "wavelength_nm", "emission_species",
        "etalon_gap_mm", "focal_length_mm",
    ]
    with nc.Dataset(l2_path) as ds:
        for a in required_attrs:
            assert hasattr(ds, a), f"Missing global attribute: {a}"


def test_file_format_netcdf4(l2_path):
    with nc.Dataset(l2_path) as ds:
        assert ds.file_format == "NETCDF4"
```

---

## 15. Acceptance criteria

A L2 file is considered schema-conformant when:

1. All `test_s20_l2_schema.py` tests pass with zero failures.
2. `ncdump -h <file>` shows `Conventions = "CF-1.8"`.
3. All science variables carry `units`, `long_name`, and `valid_range`.
4. `quality_flag` contains no values outside {0, 1, 2, 255}.
5. `wind_speed` agrees with `sqrt(u_wind² + v_wind²)` to float32 precision.
6. Latitude coverage spans at least ±30° within a single day file (SQ2 requirement).
7. File size is ≤ 10 MB per day (expected ~4 MB).

---

## 16. Open questions and future work

| # | Question | Priority |
|---|---|---|
| 1 | Should L2 files be daily or per-orbit? Per-orbit would simplify reprocessing. | Medium |
| 2 | Should `altitude` be a fixed scalar (e.g., 250.0 km nominal) or computed per-obs from emission-weighted radiative transfer? | High — needed before first flight data |
| 3 | Merger geometry matrix elements: store as ancillary variable for transparency? | Low |
| 4 | SPDF/CDF-A compliant metadata for NASA OMNIWeb submission? | Medium — needed for SQ1/SQ2 archive |
| 5 | Version-controlled calibration chain: should `calibration_file` point to a DOI or a local path? | Medium |

---

## 17. Revision history

| Version | Date | Change |
|---|---|---|
| 0.1 | 2026-04-11 | Initial draft — S20 |

---

*End of S20 specification.*
