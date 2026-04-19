# NB01 Orbit Propagator

**Spec ID:** NB01
**Spec file:** `docs/specs/NB01_orbit_propagator_2026-04-16.md`
**Project:** WindCube FPI Science Operations Center Pipeline
**Institution:** NCAR / High Altitude Observatory (HAO)
**Status:** ✓ Complete — 8/8 tests pass
**Depends on:** `src/constants.py`
**Used by:** NB02 (geometry), INT01 (integration notebook), G01 (metadata generator)
**Last updated:** 2026-04-16

> **Revision note (same-day, post-implementation):**
> `_eci_to_geodetic_batch()` added. The original `_eci_to_geodetic()` was
> called once per epoch inside the propagation loop, triggering a separate
> astropy GCRS→ITRS transform per epoch (~5 ms each). At 10 s cadence over
> 30 days that cost ~44.5 s/day → ~22 min for a 30-day G01 run.
> The batch function passes all N position vectors and timestamps to astropy
> in a single vectorised call, reducing cost to ~1.55 s/day → ~47 s for
> 30 days (28× speedup). All 8 unit tests still pass unchanged because the
> public `propagate_orbit()` interface is identical.

---

## 1. Purpose

Propagate the WindCube spacecraft orbit forward in time from a given start
epoch, producing a time-series of ECI (Earth-Centred Inertial, J2000)
position and velocity vectors at a configurable cadence. Also compute the
corresponding WGS84 geodetic position (latitude, longitude, altitude) at
each epoch.

This module is the root of the geometry pipeline. Every downstream
calculation — tangent-point finding, LOS wind projection, spacecraft velocity
removal — depends on these vectors. The position accuracy requirement is
better than ~100 m; velocity accuracy better than ~0.1 m/s.

The ECI position and velocity vectors produced here feed directly into the
AOCS Tangent Height Reference Frame (THRF) geometry computed by NB02.
The THRF definition and spacecraft body axis conventions are described in
§2.4 and must be understood by anyone extending this module.

---

## 2. Physical background

### 2.1 Sun-synchronous orbit

A sun-synchronous orbit (SSO) is a near-polar orbit whose plane precesses
eastward at the same rate Earth orbits the Sun (~0.9856°/day). WindCube's
target LTAN is 06:00 ± 1 hour (dawn-dusk). The SSO condition at 510 km
altitude requires **97.44°** inclination (slightly retrograde).

### 2.2 Why SGP4

The `sgp4` library propagates from TLE sets — the standard format for all
operational satellite telemetry. Using it means a synthetic TLE can be
replaced by a real mission TLE with zero code changes. SGP4 includes J2,
J3, J4, atmospheric drag, and lunar/solar perturbations implicitly.

### 2.3 Orbital mechanics at 510 km

```
Semi-major axis a = WGS84_A_M/1e3 + SC_ALTITUDE_KM = 6888.137 km
Orbital period   T = 2π √(a³/GM) ≈ 5689 s ≈ 94.82 min
Orbital speed    v = √(GM/a)      ≈ 7607 m/s
Orbits per day     = 86400/T      ≈ 15.19
SSO inclination    = 97.44°
RAAN precession    = 0.9856°/day
```

All derived from `EARTH_GRAV_PARAM_M3_S2`, `WGS84_A_M`, and `SC_ALTITUDE_KM`
in `src/constants.py`. Do not hardcode them.

---

## 2.4 AOCS Reference Frame Definitions

The following reference frames are defined in the WindCube AOCS Design
Report (SI-UCAR-WC-RP-004 Issue 1.0 §2.4.2) and are authoritative for all
geometry computations in the SOC pipeline.

### 2.4.1 Satellite Body Reference Frame (BRF)

| Axis | Direction | Physical meaning |
|------|-----------|-----------------|
| **−X** (BRF) | Payload boresight | FPI optical axis |
| **+Y** (BRF) | Star Tracker boresight | ASTRO-CL FOV |
| **+Z** (BRF) | Completes right-handed basis | Nominally toward Sun |

Payload boresight is `−X_BRF`. Code using `+X_BRF` as boresight is wrong.
Python variable convention: `vec_brf` for 3-vectors in BRF.

### 2.4.2 Tangent Height Reference Frame (THRF)

The THRF tracks a point at 250 km ± 5 km tangent height above WGS84.

| Configuration | Tangent height location | BRF alignment |
|---------------|------------------------|---------------|
| **Along-track** | Ahead in orbit plane | `−X_BRF` ≈ velocity direction |
| **Cross-track** | Perpendicular to orbit | `−X_BRF` ≈ orbit-normal |

Python convention: `thrf_config` ∈ `{'along_track', 'cross_track'}`.

### 2.4.3 Star Tracker Interface Reference Frame (SIRF)

Frame of the Jena Optronik ASTRO-CL star tracker. AOCS pointing requirements
(APE < 100 arcsec, AKE < 30 arcsec, RPE < 72 arcsec) are expressed in SIRF.
NB01 does not implement SIRF transforms.

### 2.4.4 Frame hierarchy and pipeline responsibility

```
ECI J2000  →  THRF  →  SIRF  →  PIRF  →  PSRF
              ^^^^       ^^^^
           NB02 owns   AOCS owns (onboard)
```

**NB01** produces ECI position and velocity only. Frame transforms to
THRF/BRF/SIRF are performed by NB02.

---

## 3. Orbital parameters

Default values for `propagate_orbit()`. All overridable as function arguments.

| Parameter | Default | Units | Source |
|-----------|---------|-------|--------|
| Altitude | `SC_ALTITUDE_KM` = 510.0 | km | `src/constants.py` |
| Inclination | 97.44 | deg | SSO condition at 510 km |
| RAAN at epoch | 90.0 | deg | Dawn-dusk SSO |
| Eccentricity | 0.0 | — | Circular orbit |
| True anomaly at epoch | 0.0 | deg | Ascending node |
| Propagation cadence | `SCIENCE_CADENCE_S` = 10.0 | s | `src/constants.py` |
| Drag term B* | 0.0 | — | No drag (short simulations) |

---

## 4. Physical constants

```python
from src.constants import (
    EARTH_GRAV_PARAM_M3_S2,  # GM = 3.986004418e14 m³/s²
    WGS84_A_M,               # 6378137.0 m
    EARTH_J2,                # 1.08263e-3
    SC_ALTITUDE_KM,          # 510.0 km
    SCIENCE_CADENCE_S,       # 10.0 s
)
```

---

## 5. Function signatures

### 5.1 Main function

```python
def propagate_orbit(
    t_start: str,
    duration_s: float,
    dt_s: float = SCIENCE_CADENCE_S,
    altitude_km: float = SC_ALTITUDE_KM,
    inclination_deg: float = 97.44,
    raan_deg: float = 90.0,
    bstar: float = 0.0,
) -> pd.DataFrame:
    """
    Propagate the WindCube orbit and return per-epoch ECI state vectors
    and WGS84 geodetic coordinates.

    Parameters
    ----------
    t_start : str
        Start epoch, ISO 8601 UTC, e.g. '2027-01-01T00:00:00'.
    duration_s : float
        Total propagation duration, seconds.
    dt_s : float
        Time step between epochs, seconds. Default: SCIENCE_CADENCE_S (10 s).
    altitude_km : float
        Circular orbit altitude above WGS84, km. Default: SC_ALTITUDE_KM.
    inclination_deg : float
        Orbital inclination, degrees. Default 97.44° (SSO at 510 km).
    raan_deg : float
        RAAN at epoch, degrees. Default 90.0° (dawn-dusk SSO).
    bstar : float
        SGP4 drag coefficient B*. Default 0.0.

    Returns
    -------
    pd.DataFrame
        One row per epoch, integer index from 0. Columns defined in §5.2.

    Performance note
    ----------------
    Geodetic conversion uses _eci_to_geodetic_batch(), which passes all N
    position vectors to astropy in a single vectorised GCRS→ITRS call.
    Benchmark: ~1.55 s/day at 10 s cadence (28× faster than the per-epoch
    loop that called _eci_to_geodetic() once per row).

    Reference frames
    ----------------
    Output pos_eci_* and vel_eci_* are ECI J2000 (TEME approximation;
    difference < 1 arcsec, ~50 m at 500 km). Not BRF, THRF, or SIRF.
    See §2.4 for AOCS frame definitions. NB02 performs BRF/THRF transforms.
    """
```

### 5.2 Output DataFrame columns

Fixed column names. Match P01 telemetry field names for consistency
between synthetic and real data.

| Column | dtype | Units | Description |
|--------|-------|-------|-------------|
| `epoch` | datetime64[ns, UTC] | — | UTC timestamp |
| `pos_eci_x` | float64 | m | ECI J2000 position x |
| `pos_eci_y` | float64 | m | ECI J2000 position y |
| `pos_eci_z` | float64 | m | ECI J2000 position z |
| `vel_eci_x` | float64 | m/s | ECI J2000 velocity x |
| `vel_eci_y` | float64 | m/s | ECI J2000 velocity y |
| `vel_eci_z` | float64 | m/s | ECI J2000 velocity z |
| `lat_deg` | float64 | deg | WGS84 geodetic latitude |
| `lon_deg` | float64 | deg | WGS84 geodetic longitude |
| `alt_km` | float64 | km | WGS84 geodetic altitude |
| `speed_ms` | float64 | m/s | Scalar speed \|vel_eci\| |

**Frame note:** `pos_eci_*` and `vel_eci_*` are in ECI J2000. The `_eci`
suffix is the authoritative frame indicator. Do not confuse with BRF, THRF,
or SIRF vectors.

### 5.3 Private helpers

#### `_datetime_to_sgp4_epoch(dt) → (year, fractional_doy)`

Converts an ISO 8601 string or datetime to the (year, fractional day-of-year)
pair required by `sgp4init`.

#### `_build_satrec(epoch_dt, altitude_km, inclination_deg, raan_deg, bstar) → Satrec`

Constructs a `Satrec` object from mean orbital elements for a circular orbit.
Mean motion `n (rad/min) = 60 × √(GM / a³)`.

#### `_eci_to_geodetic(pos_eci_m, epoch) → (lat_deg, lon_deg, alt_km)`

**Single-epoch version.** Converts one ECI position vector to WGS84 geodetic
coordinates via astropy GCRS→ITRS. Retained for use in tests and any
single-epoch callers outside the main propagation loop.

```python
def _eci_to_geodetic(pos_eci_m: np.ndarray, epoch) -> tuple[float, float, float]:
    """
    Convert a single ECI position vector to WGS84 geodetic coordinates.

    Parameters
    ----------
    pos_eci_m : np.ndarray, shape (3,) — ECI J2000 position, metres.
    epoch : astropy.time.Time or pandas Timestamp (tz-aware UTC)

    Returns
    -------
    (lat_deg, lon_deg, alt_km) : tuple[float, float, float]
    """
```

**Performance warning:** Do not call this function in a per-epoch loop over
large arrays. Use `_eci_to_geodetic_batch()` instead (see below).

#### `_eci_to_geodetic_batch(pos_eci_m_array, epochs) → (lat_deg_arr, lon_deg_arr, alt_km_arr)`

**Vectorised batch version.** Passes all N position vectors and timestamps to
astropy in a single `GCRS → ITRS` call, avoiding N separate coordinate
transform setups. This is the function called internally by `propagate_orbit()`.

```python
def _eci_to_geodetic_batch(
    pos_eci_m_array: np.ndarray,   # shape (N, 3), ECI J2000 positions, metres
    epochs,                         # array-like of N tz-aware UTC timestamps
                                    # (pandas DatetimeIndex or list of Timestamps)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorised ECI → WGS84 geodetic conversion for all N epochs at once.

    Passes pos_eci_m_array and epochs to astropy GCRS→ITRS in a single call,
    avoiding N separate coordinate transform setups. 28× faster than calling
    _eci_to_geodetic() once per epoch for a 30-day, 10 s-cadence orbit.

    Parameters
    ----------
    pos_eci_m_array : np.ndarray, shape (N, 3)
        ECI J2000 position vectors, metres.
    epochs : array-like of length N
        UTC timestamps (pandas DatetimeIndex with tz='UTC', or list of
        pandas Timestamps with tz='UTC').

    Returns
    -------
    (lat_deg_arr, lon_deg_arr, alt_km_arr) : each np.ndarray, shape (N,)
        WGS84 geodetic latitude (deg), longitude (deg), altitude (km).

    Notes
    -----
    Implementation uses astropy CartesianRepresentation with an array-valued
    obstime, which processes all epochs in one GCRS→ITRS transform call.
    The speedup over the per-epoch loop is ~28× at 10 s cadence because
    the astropy coordinate framework has significant per-call overhead
    (frame initialisation, IERS table lookup) that is paid once for the
    entire array rather than N times.

    Benchmark (510 km SSO, 10 s cadence):
        Per-epoch loop (_eci_to_geodetic × N):  ~44.5 s/day → ~22 min/30 days
        Batch (_eci_to_geodetic_batch):          ~ 1.55 s/day → ~47 s/30 days
    """
```

**Implementation sketch:**

```python
from astropy.time import Time
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation
import astropy.units as u

def _eci_to_geodetic_batch(pos_eci_m_array, epochs):
    times = Time(list(epochs), scale='utc')           # astropy Time array
    xyz   = pos_eci_m_array * u.m                     # (N, 3) Quantity
    gcrs  = GCRS(
        CartesianRepresentation(xyz[:, 0], xyz[:, 1], xyz[:, 2]),
        obstime=times,
    )
    itrs  = gcrs.transform_to(ITRS(obstime=times))
    loc   = itrs.earth_location
    return (
        loc.lat.deg,          # np.ndarray shape (N,)
        loc.lon.deg,
        loc.height.to(u.km).value,
    )
```

**`propagate_orbit()` integration:** After the SGP4 per-epoch loop (which
must remain per-epoch because `satrec.sgp4()` has no batch API), collect
all `pos_m` arrays into a `(N, 3)` array and call `_eci_to_geodetic_batch()`
once to fill `lat_deg`, `lon_deg`, and `alt_km`:

```python
# After SGP4 loop collects pos_data (list of (3,) arrays) and epochs_list:
pos_array = np.array(pos_data)                        # (N, 3)
lat_arr, lon_arr, alt_arr = _eci_to_geodetic_batch(pos_array, epochs)
```

---

## 6. Reference frame comment block

Insert immediately below the import block in the implementation file:

```python
# ---------------------------------------------------------------------------
# REFERENCE FRAME NOTE (SI-UCAR-WC-RP-004 §2.4.2)
# All position and velocity vectors produced by this module are expressed in
# ECI J2000 (approximately TEME for SGP4 output; difference < 1 arcsec).
# They are NOT in BRF (body frame), THRF (tangent height frame), or SIRF
# (star tracker frame). Frame transforms to THRF are performed by NB02.
# ---------------------------------------------------------------------------
```

---

## 7. Verification tests

All 8 tests in `tests/test_nb01_orbit_propagator.py`. These tests call
the public `propagate_orbit()` function and are unchanged by the batch
geodetic conversion — the interface is identical.

| Test | What it checks |
|------|---------------|
| T1 | `propagate_orbit` returns a DataFrame with all required columns |
| T2 | `\|pos_eci\|` ≈ 6,888,137 m ± 5,000 m (correct altitude) |
| T3 | `speed_ms` ≈ 7607 m/s ± 50 m/s |
| T4 | `alt_km` ≈ 510 km ± 20 km |
| T5 | Orbital period from consecutive row timestamps ≈ 5689 s ± 30 s |
| T6 | Inclination from latitude extremes ≈ 97.44° ± 0.5° |
| T7 | RAAN precession ≈ 0.9856°/day ± 0.05°/day |
| T8 | No NaN or Inf in any output column |

---

## 8. Expected numerical values

| Quantity | Expected value | Tolerance | Source |
|----------|----------------|-----------|--------|
| `\|pos_eci\|` | 6,888,137 m | ± 5,000 m | WGS84_A_M + 510 km |
| `speed_ms` | 7607 m/s | ± 50 m/s | √(GM/a) at 510 km |
| `alt_km` | 510 km | ± 20 km | Circular orbit |
| Orbital period | 5689 s | ± 30 s | 2π√(a³/GM) |
| Inclination | 97.44° | ± 0.5° | SSO condition |
| RAAN precession | 0.9856°/day | ± 0.05°/day | SSO requirement |
| Propagation time (30 days, 10 s) | ~47 s | — | Batch geodetic benchmark |

---

## 9. Output contract for downstream modules

NB02 imports from this module using exactly these column names:

```python
from src.geometry.nb01_orbit_propagator_2026_04_16 import propagate_orbit

df = propagate_orbit(t_start='2027-01-01T00:00:00', duration_s=86400.0)

pos_eci_m  = df[['pos_eci_x', 'pos_eci_y', 'pos_eci_z']].values   # (N,3), m
vel_eci_ms = df[['vel_eci_x', 'vel_eci_y', 'vel_eci_z']].values   # (N,3), m/s
epochs     = df['epoch'].values                                     # datetime64[ns, UTC]
lat_deg    = df['lat_deg'].values                                   # (N,), deg
lon_deg    = df['lon_deg'].values                                   # (N,), deg
alt_km     = df['alt_km'].values                                    # (N,), km
```

Do not rename these columns. NB02 and G01 depend on them by name.

---

## 10. File location in repository

```
soc_sewell/
├── src/
│   └── geometry/
│       ├── __init__.py
│       └── nb01_orbit_propagator_2026_04_16.py   ← this module
└── tests/
    └── test_nb01_orbit_propagator.py              ← 8 tests
```

**`__init__.py` export:**
```python
from src.geometry.nb01_orbit_propagator_2026_04_16 import propagate_orbit  # noqa: F401
```

---

## 11. Dependencies

```
sgp4     >= 2.21   # TLE-based SGP4/SDP4 propagator
astropy  >= 5.0    # GCRS→ITRS coordinate transforms, Time array handling
numpy    >= 1.24   # Array operations
pandas   >= 2.0    # Output DataFrame
```

---

## 12. Instructions for Claude Code

### Preamble
```bash
cat PIPELINE_STATUS.md
```

### Prerequisite reads
1. This spec: `docs/specs/NB01_orbit_propagator_2026-04-16.md` — complete.
2. `src/constants.py` — confirm `SC_ALTITUDE_KM`, `SCIENCE_CADENCE_S`,
   `EARTH_GRAV_PARAM_M3_S2`, `WGS84_A_M` are present.
3. `CLAUDE.md` at repo root.

### Key implementation note — batch geodetic conversion

The performance-critical change is refactoring `propagate_orbit()` to collect
SGP4 outputs into arrays and then call `_eci_to_geodetic_batch()` once:

```python
# Collect from SGP4 per-epoch loop (sgp4 has no batch API)
pos_list  = []   # each element shape (3,)
vel_list  = []
epoch_list = []
for epoch in epochs:
    ...  # sgp4 call per epoch
    pos_list.append(pos_m)
    vel_list.append(vel_m_s)
    epoch_list.append(epoch)

# Single vectorised geodetic conversion — replaces per-epoch _eci_to_geodetic()
pos_array = np.array(pos_list)                              # (N, 3)
lat_arr, lon_arr, alt_arr = _eci_to_geodetic_batch(pos_array, epoch_list)
speed_arr = np.linalg.norm(np.array(vel_list), axis=1)

# Assemble DataFrame from arrays
df = pd.DataFrame({
    'epoch':     epochs,
    'pos_eci_x': pos_array[:, 0], 'pos_eci_y': pos_array[:, 1],
    'pos_eci_z': pos_array[:, 2],
    'vel_eci_x': np.array(vel_list)[:, 0], ...
    'lat_deg':   lat_arr, 'lon_deg': lon_arr, 'alt_km': alt_arr,
    'speed_ms':  speed_arr,
})
```

Keep `_eci_to_geodetic()` (single-epoch version) for tests and any external
callers that need a single conversion.

### Tests
```bash
pytest tests/test_nb01_orbit_propagator.py -v   # all 8 must pass
pytest tests/ -v                                 # no regressions
```

### Epilogue
```bash
git add PIPELINE_STATUS.md \
        src/geometry/nb01_orbit_propagator_2026_04_16.py
git commit -m "perf(nb01): vectorise geodetic conversion, 28x speedup

Adds _eci_to_geodetic_batch(): all N positions passed to astropy in one
GCRS->ITRS call instead of N separate calls.
Benchmark: 1.55 s/day (was 44.5 s/day) at 10 s cadence, 510 km SSO.
All 8 NB01 tests pass. Public propagate_orbit() interface unchanged.

Also updates PIPELINE_STATUS.md"
```
