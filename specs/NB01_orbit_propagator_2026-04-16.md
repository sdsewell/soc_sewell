# S06 — NB01 Orbit Propagator

**Spec ID:** S06
**Spec file:** `specs/S06_nb01_orbit_propagator_2026-04-16.md`
**Project:** WindCube FPI Science Operations Center Pipeline
**Institution:** NCAR / High Altitude Observatory (HAO)
**Status:** Authoritative
**Depends on:** S01, S02, S03, S04
**Used by:** S07 (NB02 geometry), S08 (INT01 integration notebook), S16 (INT02), S17 (INT03)
**Last updated:** 2026-04-16
**Created/Modified by:** Claude AI

---

## 1. Purpose

Propagate the WindCube spacecraft orbit forward in time from a given start
epoch, producing a time-series of ECI (Earth-Centred Inertial, J2000)
position and velocity vectors at a configurable cadence. Also compute the
corresponding WGS84 geodetic position (latitude, longitude, altitude) at each
epoch.

This module is the root of the geometry pipeline. Every downstream
calculation — tangent-point finding, LOS wind projection, spacecraft velocity
removal — depends on these vectors. The position accuracy requirement is
better than ~100 m; velocity accuracy better than ~0.1 m/s. A 0.1 m/s
velocity error projects to ~0.1 m/s wind error in the best case — comparable
to the STM budget — so this is not a loose requirement.

The ECI position and velocity vectors produced here feed directly into the
AOCS Tangent Height Reference Frame (THRF) geometry computed by S07 (NB02).
The THRF definition and the spacecraft body axis conventions used by the AOCS
are described in Section 2.4 and must be understood by anyone extending this
module.

---

## 2. Physical background

### 2.1 Sun-synchronous orbit

A sun-synchronous orbit (SSO) is a near-polar orbit whose plane precesses
eastward at the same rate Earth orbits the Sun (~0.9856°/day), keeping the
local time of the ascending node (LTAN) nearly constant throughout the year.
WindCube's target LTAN is 06:00 ± 1 hour (dawn-dusk). This keeps the
instrument looking away from the Sun and maximises solar panel illumination.

The SSO condition is achieved by choosing the orbital inclination so that
Earth's J2 oblateness causes exactly the right nodal precession. At 510 km
altitude the required inclination is **97.44°** (slightly retrograde). The
SGP4 propagator handles the J2 precession automatically through its
Brouwer mean-motion model.

### 2.2 Why SGP4

The `sgp4` library propagates from Two-Line Element (TLE) sets — the standard
format for all operational satellite telemetry. Using it means a synthetic
TLE can be replaced by a real mission TLE with zero code changes. SGP4
includes J2, J3, J4, atmospheric drag, and lunar/solar perturbations
implicitly. It is numerically validated against decades of operational use.
A custom Keplerian integrator would require explicit implementation of all
these effects.

### 2.3 Orbital mechanics at 510 km

Key parameters for WindCube's nominal 510 km circular orbit, derived from
the constants in S03:

```
Semi-major axis a = WGS84_A_M/1e3 + SC_ALTITUDE_KM = 6888.137 km
Orbital period   T = 2π √(a³/GM) = 5689 s = 94.82 min
Orbital speed    v = √(GM/a)      = 7607 m/s
Orbits per day     = 86400/T      = 15.19
SSO inclination    = 97.44°       (computed from J2 precession condition)
RAAN precession    = 0.9856°/day  (matches Earth's orbital rate)
```

All of these are derived from `EARTH_GRAV_PARAM_M3_S2`, `WGS84_A_M`,
`SC_ALTITUDE_KM`, and `EARTH_J2` in S03. Do not hardcode them.

---

## 2.4 AOCS Reference Frame Definitions

The following reference frames are defined in the WindCube AOCS Design Report
(SI-UCAR-WC-RP-004, Issue 1.0, §2.4.2) and are authoritative for all geometry
computations in the SOC pipeline. The orbit propagator (NB01) produces ECI
vectors that downstream modules (NB02, NB03) transform into these frames.
Variable names, axis labels, and frame abbreviations throughout the pipeline
**must** match the conventions below exactly.

### 2.4.1 Satellite Body Reference Frame (BRF)

The BRF is fixed to the spacecraft structure. Its axes are defined as (from
SI-UCAR-WC-RP-004 §2.4.2.1):

| Axis | Direction | Physical meaning |
|------|-----------|-----------------|
| **−X** (BRF) | Payload boresight | The FPI optical axis points in the **−X** direction |
| **+Y** (BRF) | Star Tracker boresight | The ASTRO-CL star tracker FOV points in **+Y** |
| **+Z** (BRF) | Completes right-handed basis | Nominally toward Sun in along-track fine pointing |

**Key conventions for pipeline code:**
- The payload boresight is `−X_BRF`, not `+X_BRF`. Any vector expressed in the
  BRF that points along the FPI optical axis has a negative X component.
- In the along-track fine pointing (FP) configuration the AOCS aligns:
  - Primary: `−X_BRF` → tangent height point (250 km airglow)
  - Secondary: `−Z_BRF` → approximately toward Sun
- In the cross-track fine pointing configuration:
  - Primary: `−X_BRF` → tangent height point (perpendicular to orbit plane)
  - Secondary: `+Z_BRF` → approximately toward Sun
- Variable name convention in Python: `vec_brf` for 3-vectors expressed in BRF.

### 2.4.2 Tangent Height Reference Frame (THRF)

The THRF is the dynamic reference frame used during all payload observation
periods. Its purpose is to track a point at 250 km ± 5 km tangent height above
the Earth's surface (the OI 630.0 nm airglow layer) as the spacecraft moves
along its orbit (SI-UCAR-WC-RP-004 §2.4.2.2; also reference document
THRF-NOTE 2024-09-20).

**Physical definition:** The tangent point P_t is the point on the 250 km
altitude surface (WGS84 ellipsoid + 250 km) where the FPI boresight ray is
tangent to that surface. This point moves continuously as the spacecraft
advances in its orbit.

The THRF has **two distinct observation configurations**; the AOCS alternates
between them one orbit at a time:

| Configuration | Tangent height location | BRF alignment |
|---------------|------------------------|---------------|
| **Along-track** (anti-track) | In the orbit plane, ahead of the spacecraft | `−X_BRF` ≈ velocity direction; `−Y_BRF` ≈ nadir; `−Z_BRF` ≈ toward Sun |
| **Cross-track** | Perpendicular to the orbit plane | `−X_BRF` ≈ orbit-normal; `−Y_BRF` ≈ nadir; `+Z_BRF` ≈ toward Sun; `+Z_BRF` ≈ velocity |

**Geometry for NB02:** NB01 delivers the ECI position **r_sc** and velocity
**v_sc** of the spacecraft. NB02 uses these to compute the unit boresight
vector **û_bore** in ECI for each observation configuration:

- The tracking direction **v_track** is the direction from the spacecraft
  toward the tangent point P_t.
- The elevation angle θ_t from horizontal to the boresight satisfies
  `cos(θ_t) = R_E / (R_E + h_t)` where `R_E` is the local Earth radius
  (WGS84) and `h_t` = 250 km.
- The AOCS model error requirement is < 5 km absolute deviation from the
  true 250 km tangent height (SI-UCAR-WC-RP-004 §2.1).

**Variable name convention in Python:**
- `thrf_config`: string, one of `'along_track'` or `'cross_track'`
- `vec_thrf`: 3-vector expressed in the THRF frame
- `tangent_point_eci_m`: ECI position of the tangent point, metres

### 2.4.3 Star Tracker Interface Reference Frame (SIRF)

The SIRF is the coordinate frame of the Jena Optronik ASTRO-CL star tracker
unit (SI-UCAR-WC-RP-004 §2.4.2.3). It is the frame in which all AOCS
performance requirements (APE, AKE, RPE) are expressed.

**Axis definition:**
| Axis | Direction |
|------|-----------|
| **+Z** (SIRF) | Normal to the aluminium lid surface, pointing in the direction of the FOV |
| **+X** (SIRF) | Parallel to the edge of the aluminium lid |
| **+Y** (SIRF) | Completes the right-handed coordinate system |

**Key relationships:**
- The SIRF is mounted such that `+Y_BRF` (spacecraft body) ≈ `+Z_SIRF` (star
  tracker boresight); the exact alignment rotation is given in the AOCS CAD
  model and the Franck diagram (SI-UCAR-WC-RP-004 Figure 3-1).
- AOCS performance requirements are all specified relative to SIRF:
  - APE < 100 arcsec RMS (SYS.52)
  - AKE < 30 arcsec RMS (SYS.108)
  - RPE < 72 arcsec RMS over 10 s integration (SYS.109)
- The pipeline does **not** need to implement SIRF transforms directly; these
  are handled by the onboard ADCS. However, any attitude knowledge metadata
  stored in the L1/L2 files (S18, S20) should note that reported attitude
  quaternions are expressed relative to SIRF.

**Variable name convention in Python:** `vec_sirf` for 3-vectors expressed in
SIRF.

### 2.4.4 Frame hierarchy and pipeline responsibility

The full pointing error chain (Franck diagram, SI-UCAR-WC-RP-004 Figure 3-1)
flows as:

```
ECI J2000  →  THRF  →  SIRF  →  PIRF  →  PSRF
              ^^^^       ^^^^
           NB02 owns   AOCS owns (onboard)
```

- **NB01** (this module): produces ECI position and velocity.
- **NB02** (S07): transforms ECI → THRF geometry; computes tangent point,
  boresight unit vector, and LOS wind projection.
- **AOCS onboard**: closes the THRF → SIRF → PIRF → PSRF chain in real time.
- **SOC pipeline**: does not reproduce the AOCS control loop; it uses the
  reported SIRF attitude quaternion (from spacecraft telemetry) to reconstruct
  the actual boresight pointing for wind retrieval.

---

## 3. Orbital parameters

Default values for `propagate_orbit()`. All are overridable as function
arguments so that sensitivity studies and non-nominal scenarios can be run.

| Parameter | Symbol | Default value | Units | Source |
|-----------|--------|---------------|-------|--------|
| Altitude (nominal) | h | `SC_ALTITUDE_KM` = 510.0 | km | S03 / WC-SE-0003 v8 |
| Inclination | i | 97.44 | deg | Computed for SSO at 510 km |
| RAAN at epoch | Ω₀ | 90.0 | deg | Dawn-dusk SSO (Sun at 90° from node) |
| Eccentricity | e | 0.0 | — | Circular orbit |
| Argument of perigee | ω | 0.0 | deg | Undefined for circular; set to 0 |
| True anomaly at epoch | ν₀ | 0.0 | deg | Spacecraft at ascending node |
| Propagation cadence | Δt | `SCIENCE_CADENCE_S` = 10.0 | s | S03 / WC-SE-0003 v8 |
| Drag term | B* | 0.0 | — | No atmospheric drag (short simulations) |

**Inclination note:** 97.44° is the SSO value at exactly 510 km. The old
NB01 spec used 97.4° (the value from 525 km). The difference of 0.04° is
small but the correct value for 510 km is used here to be consistent with
S03. The SGP4 propagator will maintain the correct RAAN precession rate if
the inclination is set correctly.

---

## 4. Physical constants from S03

Import all constants by symbol. Do not hardcode numerical values.

```python
from src.constants import (
    EARTH_GRAV_PARAM_M3_S2,  # GM = 3.986004418e14 m³/s²
    WGS84_A_M,               # Equatorial radius = 6378137.0 m
    EARTH_J2,                # J2 = 1.08263e-3
    SC_ALTITUDE_KM,          # 510.0 km
    SCIENCE_CADENCE_S,       # 10.0 s
)
```

---

## 5. Function signatures

### 5.1 Main function

```python
import pandas as pd
import numpy as np
from sgp4.api import Satrec, jday
from astropy.time import Time
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation
import astropy.units as u
from src.constants import SC_ALTITUDE_KM, SCIENCE_CADENCE_S

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
        Start epoch in ISO 8601 UTC format, e.g. '2027-01-01T00:00:00'.
    duration_s : float
        Total propagation duration in seconds.
    dt_s : float
        Time step between output epochs, seconds. Default: SCIENCE_CADENCE_S (10 s).
    altitude_km : float
        Circular orbit altitude above WGS84 ellipsoid, km.
        Default: SC_ALTITUDE_KM (510 km).
    inclination_deg : float
        Orbital inclination, degrees. Default 97.44° (SSO at 510 km).
    raan_deg : float
        Right ascension of ascending node at epoch, degrees.
        Default 90.0° (dawn-dusk, Sun near orbit plane).
    bstar : float
        SGP4 drag coefficient (B*). Default 0.0 (no drag).
        Use 0.0 for short simulations (< 1 day); drag matters over weeks.

    Returns
    -------
    pd.DataFrame
        One row per epoch. Columns defined in Section 5.2.
        Integer index starting at 0.

    Notes
    -----
    Output positions are in ECI J2000 (metres). Strictly speaking SGP4
    outputs in TEME (True Equator Mean Equinox), which differs from ECI
    J2000 by < 1 arcsec. For synthetic orbit simulations at the accuracy
    level required here (~100 m), this difference is negligible. The column
    name `pos_eci_*` is used for consistency with the telemetry field names
    in S18 (P01 metadata spec). If sub-metre accuracy is later needed,
    apply the TEME→GCRS rotation via astropy before storing.

    The output integer index (0, 1, 2, ...) makes integer-based row lookup
    straightforward in downstream modules. The epoch column carries the
    full timestamp for time-based operations.

    Reference frames: output vectors are in ECI J2000. Downstream module
    NB02 (S07) transforms these into the THRF (§2.4.2) for tangent-point
    geometry. The BRF (§2.4.1) and SIRF (§2.4.3) transforms are handled
    by the onboard AOCS and are not reproduced in this module.
    """
```

### 5.2 Output DataFrame columns

These column names are fixed. They match the telemetry field names from the
image metadata spec (S18) so that synthetic and real data use identical names.

| Column | dtype | Units | Description |
|--------|-------|-------|-------------|
| `epoch` | datetime64[ns, UTC] | — | UTC timestamp of observation |
| `pos_eci_x` | float64 | m | ECI J2000 position x-component |
| `pos_eci_y` | float64 | m | ECI J2000 position y-component |
| `pos_eci_z` | float64 | m | ECI J2000 position z-component |
| `vel_eci_x` | float64 | m/s | ECI J2000 velocity x-component |
| `vel_eci_y` | float64 | m/s | ECI J2000 velocity y-component |
| `vel_eci_z` | float64 | m/s | ECI J2000 velocity z-component |
| `lat_deg` | float64 | deg | WGS84 geodetic latitude |
| `lon_deg` | float64 | deg | WGS84 geodetic longitude |
| `alt_km` | float64 | km | WGS84 geodetic altitude |
| `speed_ms` | float64 | m/s | Scalar orbital speed \|vel_eci\| |

**Frame note:** `pos_eci_*` and `vel_eci_*` are expressed in ECI J2000. They
are **not** in BRF, THRF, or SIRF. Do not confuse these with AOCS-frame
vectors. The naming suffix `_eci` is the authoritative indicator of frame.

### 5.3 Private helper: ECI to geodetic

```python
def _eci_to_geodetic(pos_eci_m: np.ndarray, epoch) -> tuple[float, float, float]:
    """
    Convert ECI position vector to WGS84 geodetic coordinates.

    Parameters
    ----------
    pos_eci_m : np.ndarray, shape (3,)
        ECI J2000 position vector, metres.
    epoch : astropy.time.Time or datetime
        UTC epoch of the observation.

    Returns
    -------
    (lat_deg, lon_deg, alt_km) : tuple[float, float, float]
        WGS84 geodetic latitude (deg), longitude (deg), altitude (km).

    Implementation
    --------------
    Uses astropy GCRS→ITRS transform chain. GCRS is approximately
    equivalent to ECI J2000 for this application. The ITRS frame
    co-rotates with Earth, giving geodetic coordinates directly.
    """
    t = Time(epoch, scale='utc') if not isinstance(epoch, Time) else epoch
    gcrs = GCRS(
        CartesianRepresentation(pos_eci_m * u.m),
        obstime=t,
    )
    itrs = gcrs.transform_to(ITRS(obstime=t))
    loc = itrs.earth_location
    return (
        float(loc.lat.deg),
        float(loc.lon.deg),
        float(loc.height.to(u.km).value),
    )
```

### 5.4 TLE construction from orbital elements

SGP4 requires a TLE as input. Build a synthetic TLE from the orbital
elements using `sgp4.api.Satrec` with `sgp4init()`:

```python
from sgp4.api import Satrec, WGS84
from sgp4.conveniences import sat_epoch_datetime

def _build_satrec(
    epoch_dt,            # datetime object, UTC
    altitude_km: float,
    inclination_deg: float,
    raan_deg: float,
    bstar: float,
) -> Satrec:
    """
    Construct a Satrec object from mean orbital elements.

    Mean motion in TLE units is revolutions/day:
        n_rev_day = (86400 / (2π)) × √(GM / a³)

    For altitude_km = 510:
        a = 6888.137 km
        n = 1.1066e-3 rad/s
        n_rev_day = 15.186 rev/day
    """
    from sgp4.api import Satrec, WGS84
    import numpy as np
    from src.constants import EARTH_GRAV_PARAM_M3_S2, WGS84_A_M

    a_km = WGS84_A_M / 1e3 + altitude_km          # semi-major axis, km
    n_rad_s = np.sqrt(EARTH_GRAV_PARAM_M3_S2 / (a_km * 1e3)**3)
    n_rev_day = n_rad_s * 86400 / (2 * np.pi)     # TLE mean motion

    satrec = Satrec()
    satrec.sgp4init(
        WGS84,                  # gravity model
        'i',                    # 'i' = improved mode
        0,                      # satellite number (arbitrary for synthetic)
        _datetime_to_sgp4_epoch(epoch_dt),  # epoch in SGP4 fractional year format
        bstar,                  # drag term B*
        0.0,                    # first derivative of mean motion (ignored)
        0.0,                    # second derivative of mean motion (ignored)
        0.0,                    # eccentricity (circular)
        0.0,                    # argument of perigee (undefined for circular)
        np.radians(inclination_deg),
        0.0,                    # mean anomaly at epoch (spacecraft at ascending node)
        n_rev_day,
        np.radians(raan_deg),
    )
    return satrec
```

**Epoch format:** SGP4 uses a fractional year format internally. Use
`astropy.time.Time` to convert from ISO strings to Julian date, then from
Julian date to the SGP4 epoch format. Do not compute this by hand.

---

## 6. Known pitfalls

**SGP4 unit trap.** The `sgp4` library returns position in **km** and
velocity in **km/s**. This is the most common source of errors. Convert to
metres and m/s immediately after every SGP4 call, before storing anything:

```python
e, pos_km, vel_km_s = satrec.sgp4(jd, fr)
assert e == 0, f"SGP4 error code {e}"
pos_m   = np.array(pos_km)   * 1e3   # km → m
vel_m_s = np.array(vel_km_s) * 1e3   # km/s → m/s
# Defensive check:
assert np.linalg.norm(pos_m) > 1e6, \
    f"pos_eci looks like km not metres: |r| = {np.linalg.norm(pos_m):.1f}"
```

**SGP4 error codes.** `sgp4()` returns `(e, r, v)` where `e` is an integer
error code. Always assert `e == 0` before using `r` and `v`. Non-zero codes
indicate a propagation failure (typically: epoch too far from TLE epoch, or
physically impossible state).

**TEME vs ECI J2000.** SGP4 natively outputs in the TEME frame (True Equator,
Mean Equinox), not ECI J2000/GCRS. The difference is < 1 arcsec (~50 m at
500 km altitude) and is negligible for this application. Add a comment in
the code flagging this approximation. If sub-metre precision is needed in
future, apply the TEME→GCRS rotation from astropy before storing.

**Epoch format.** `jday()` requires the Julian date split into integer and
fractional parts: `jd, fr = jday(year, month, day, hour, minute, second)`.
Use `astropy.time.Time` to derive these values reliably rather than computing
Julian date arithmetic by hand.

**DataFrame index.** Use a simple integer index (0, 1, 2, ...), not the epoch
as the index. Integer-based row lookup in downstream modules is much simpler
than datetime-based lookup.

**Frame confusion.** The ECI vectors produced here are **not** in BRF, THRF,
or SIRF. Do not pass `pos_eci_*` columns into any function expecting a
BRF-frame vector without first applying the appropriate rotation. See §2.4.4
for the frame hierarchy. Column names carry the `_eci` suffix as the
authoritative frame indicator.

---

## 7. Verification tests

All tests in `tests/test_s06_nb01_orbit_propagator.py`.

### T1 — Output schema

```python
def test_output_schema():
    """Output DataFrame must have all required columns."""
    df = propagate_orbit('2027-01-01T00:00:00', duration_s=100)
    required = [
        'epoch', 'pos_eci_x', 'pos_eci_y', 'pos_eci_z',
        'vel_eci_x', 'vel_eci_y', 'vel_eci_z',
        'lat_deg', 'lon_deg', 'alt_km', 'speed_ms',
    ]
    for col in required:
        assert col in df.columns, f"Missing column: {col}"
    assert len(df) == 11   # epochs at t=0, 10, 20, ..., 100 s
    assert df.index[0] == 0 and df.index[-1] == 10
```

### T2 — Units are SI (metres, not km)

```python
def test_units_are_si():
    """Position must be in metres (> 1e6), not km."""
    df = propagate_orbit('2027-01-01T00:00:00', duration_s=10)
    pos_norm = np.linalg.norm(df[['pos_eci_x','pos_eci_y','pos_eci_z']].values[0])
    assert pos_norm > 1e6, \
        f"|pos_eci| = {pos_norm:.1f} — looks like km, expected metres (~6.9e6)"
    vel_norm = np.linalg.norm(df[['vel_eci_x','vel_eci_y','vel_eci_z']].values[0])
    assert 7400 < vel_norm < 7800, \
        f"|vel_eci| = {vel_norm:.1f} m/s — outside expected range [7400, 7800]"
```

### T3 — Orbital speed matches S03 derived value

```python
def test_orbital_speed():
    """Scalar orbital speed must be within ±50 m/s of theoretical value."""
    import numpy as np
    from src.constants import EARTH_GRAV_PARAM_M3_S2, WGS84_A_M, SC_ALTITUDE_KM
    a_m = WGS84_A_M + SC_ALTITUDE_KM * 1e3
    v_theory = np.sqrt(EARTH_GRAV_PARAM_M3_S2 / a_m)   # ~7607 m/s at 510 km
    df = propagate_orbit('2027-01-01T00:00:00', duration_s=10,
                         altitude_km=SC_ALTITUDE_KM)
    v_actual = float(df['speed_ms'].iloc[0])
    assert abs(v_actual - v_theory) < 50, \
        f"Speed {v_actual:.1f} m/s; theory {v_theory:.1f} m/s; diff {abs(v_actual-v_theory):.1f}"
```

### T4 — Orbital period matches theoretical value

```python
def test_orbital_period():
    """Period recovered from ascending node crossings must be 5660–5720 s."""
    import numpy as np
    df = propagate_orbit('2027-01-01T00:00:00', duration_s=12000, dt_s=10.0)
    # Ascending node: z-component of position crosses zero going positive
    z = df['pos_eci_z'].values
    crossings = np.where((z[:-1] < 0) & (z[1:] >= 0))[0]
    assert len(crossings) >= 2, "Could not find two ascending node crossings"
    # Period in seconds from epoch timestamps
    t0 = df['epoch'].iloc[crossings[0]]
    t1 = df['epoch'].iloc[crossings[1]]
    period_s = (t1 - t0).total_seconds()
    assert 5660 < period_s < 5720, \
        f"Period {period_s:.1f} s; expected ~5689 s (94.82 min) at 510 km"
```

### T5 — Inclination matches SSO requirement

```python
def test_inclination():
    """Recovered inclination from angular momentum must be 97.0°–97.9°."""
    import numpy as np
    df = propagate_orbit('2027-01-01T00:00:00', duration_s=100)
    pos = df[['pos_eci_x','pos_eci_y','pos_eci_z']].values[0]
    vel = df[['vel_eci_x','vel_eci_y','vel_eci_z']].values[0]
    L = np.cross(pos, vel)
    inc_deg = np.degrees(np.arccos(L[2] / np.linalg.norm(L)))
    assert 97.0 < inc_deg < 97.9, \
        f"Inclination {inc_deg:.2f}°; expected ~97.44° for SSO at 510 km"
```

### T6 — Altitude stays within operational range

```python
def test_altitude_consistency():
    """Geodetic altitude must stay within SC_ALTITUDE_RANGE_KM over 24 hours."""
    from src.constants import SC_ALTITUDE_RANGE_KM
    df = propagate_orbit('2027-01-01T00:00:00', duration_s=86400)
    lo, hi = SC_ALTITUDE_RANGE_KM
    in_range = df['alt_km'].between(lo - 20, hi + 20)   # ±20 km margin for SGP4 oscillation
    assert in_range.all(), \
        f"Altitude out of range: min={df['alt_km'].min():.1f}, max={df['alt_km'].max():.1f} km"
```

### T7 — RAAN precession rate matches SSO requirement

```python
def test_raan_precession():
    """
    RAAN must precess at 0.9856 ± 0.05 deg/day — the sun-synchronous rate.
    Verified over 7 days to get a stable rate estimate.
    """
    import numpy as np
    df = propagate_orbit('2027-01-01T00:00:00', duration_s=7 * 86400, dt_s=600)
    pos = df[['pos_eci_x','pos_eci_y','pos_eci_z']].values
    vel = df[['vel_eci_x','vel_eci_y','vel_eci_z']].values
    # Angular momentum vector at each epoch
    L = np.cross(pos, vel)          # shape (N, 3)
    L_norm = L / np.linalg.norm(L, axis=1, keepdims=True)
    # RAAN = atan2(L_x, -L_y) from the ascending node vector
    raan = np.degrees(np.arctan2(L_norm[:, 0], -L_norm[:, 1]))
    # Unwrap to remove ±180° jumps
    raan_unwrapped = np.unwrap(raan, period=360)
    # Linear fit to get precession rate
    t_days = np.arange(len(raan_unwrapped)) * 600 / 86400
    coeffs = np.polyfit(t_days, raan_unwrapped, 1)
    precession_rate = coeffs[0]   # deg/day
    assert 0.93 < precession_rate < 1.04, \
        f"RAAN precession {precession_rate:.4f} deg/day; expected 0.9856 ± 0.05 deg/day"
```

### T8 — Geodetic coordinates are self-consistent with ECI position

```python
def test_geodetic_consistency():
    """
    Geodetic altitude must be consistent with |pos_eci|.
    For a spherical-Earth approximation: |pos_eci| ≈ WGS84_A_M + alt_km*1e3 ± 25 km.
    """
    import numpy as np
    from src.constants import WGS84_A_M
    df = propagate_orbit('2027-01-01T00:00:00', duration_s=5700, dt_s=60)
    pos_norm = np.linalg.norm(
        df[['pos_eci_x','pos_eci_y','pos_eci_z']].values, axis=1
    )
    alt_from_pos_km = (pos_norm - WGS84_A_M) / 1e3   # rough spherical estimate
    alt_from_geodetic = df['alt_km'].values
    diff = np.abs(alt_from_pos_km - alt_from_geodetic)
    assert diff.max() < 25.0, \
        f"Geodetic and ECI altitudes disagree by {diff.max():.1f} km (max); expected < 25 km"
```

---

## 8. Expected numerical values

For `propagate_orbit('2027-01-01T00:00:00', duration_s=100, altitude_km=510.0)`:

| Quantity | Expected value | Tolerance | Derivation |
|----------|----------------|-----------|-----------|
| `\|pos_eci\|` | ≈ 6,888,137 m | ± 5,000 m | WGS84_A_M + 510 km |
| `speed_ms` | ≈ 7607 m/s | ± 50 m/s | √(GM/a) at 510 km |
| `alt_km` | ≈ 510 km | ± 20 km | Circular orbit |
| Orbital period | ≈ 5689 s (94.82 min) | ± 30 s | 2π√(a³/GM) |
| Inclination | ≈ 97.44° | ± 0.5° | SSO condition at 510 km |
| RAAN precession | 0.9856°/day | ± 0.05°/day | SSO requirement |
| Orbits per day | ≈ 15.19 | ± 0.3 | 86400/T |

---

## 9. Output contract for downstream modules

S07 (NB02) imports from this module using exactly these column names:

```python
from src.geometry.nb01_orbit_propagator_2026_04_05 import propagate_orbit

df = propagate_orbit(
    t_start='2027-01-01T00:00:00',
    duration_s=2 * 5689,        # two full orbits
    dt_s=10.0,
)

# S07 accesses these arrays:
pos_eci_m  = df[['pos_eci_x', 'pos_eci_y', 'pos_eci_z']].values   # shape (N, 3), metres, ECI J2000
vel_eci_ms = df[['vel_eci_x', 'vel_eci_y', 'vel_eci_z']].values   # shape (N, 3), m/s,   ECI J2000
epochs     = df['epoch'].values                                     # datetime64[ns, UTC]
lat_deg    = df['lat_deg'].values                                   # shape (N,), degrees
lon_deg    = df['lon_deg'].values                                   # shape (N,), degrees
alt_km     = df['alt_km'].values                                    # shape (N,), km

# S07 uses pos_eci_m and vel_eci_ms to construct the THRF geometry
# (see §2.4.2). These vectors are in ECI J2000, NOT in BRF or SIRF.
```

Do not rename these columns. S07 depends on them by name.

---

## 10. File location in repository

```
soc_sewell/
├── src/
│   └── geometry/
│       ├── __init__.py
│       └── nb01_orbit_propagator_2026_04_05.py    ← this module
└── tests/
    └── test_s06_nb01_orbit_propagator.py          ← 8 tests
```

**`__init__.py` content:**
```python
# src/geometry/__init__.py
from src.geometry.nb01_orbit_propagator_2026_04_05 import propagate_orbit  # noqa: F401
```

Note: Python module filenames use underscores for the date component
(`2026_04_05`) while spec filenames use hyphens (`2026-04-05`). See S01
Section 10.

---

## 11. Dependencies

```
sgp4 >= 2.21      # TLE-based SGP4/SDP4 propagator
astropy >= 5.0    # GCRS→ITRS coordinate transforms, time handling
numpy >= 1.24     # Array operations
pandas >= 2.0     # Output DataFrame
```

```bash
pip install sgp4 astropy numpy pandas
```

---

## 12. Instructions for Claude Code

### Prerequisite reads

Before writing any code, read the following files in order:

1. This spec: `specs/S06_nb01_orbit_propagator_2026-04-16.md` — complete read required.
2. `specs/S01_*.md` — project conventions (file naming, module headers, commit format).
3. `specs/S02_*.md` — pipeline architecture and tier structure.
4. `specs/S03_*.md` — authoritative constants; confirm `SC_ALTITUDE_KM`,
   `SCIENCE_CADENCE_S`, `EARTH_GRAV_PARAM_M3_S2`, `WGS84_A_M` are present.
5. `CLAUDE.md` at repo root — project-level instructions.

### Reference frame awareness (new — read carefully)

This spec (§2.4) defines three AOCS reference frames from SI-UCAR-WC-RP-004:

- **BRF** (Body Reference Frame): payload boresight is **−X_BRF**; star
  tracker boresight is **+Y_BRF**.
- **THRF** (Tangent Height Reference Frame): the dynamic frame tracking the
  250 km airglow tangent point. Two configurations: `along_track` and
  `cross_track`.
- **SIRF** (Star Tracker Interface Reference Frame): the frame in which AOCS
  pointing requirements (APE, AKE, RPE) are expressed.

**NB01 does not implement BRF, THRF, or SIRF transforms.** It only produces
ECI J2000 vectors. However:
- All docstrings for functions that output position or velocity vectors **must**
  state the frame explicitly: `"ECI J2000 (metres)"`.
- All variable names for ECI vectors **must** carry the `_eci` suffix.
- Add a module-level comment block immediately below the imports:

```python
# ---------------------------------------------------------------------------
# REFERENCE FRAME NOTE (SI-UCAR-WC-RP-004 §2.4.2)
# All position and velocity vectors produced by this module are expressed in
# ECI J2000 (approximately TEME for SGP4 output; difference < 1 arcsec).
# They are NOT in BRF (body frame), THRF (tangent height frame), or SIRF
# (star tracker frame). Frame transforms to THRF are performed by NB02 (S07).
# ---------------------------------------------------------------------------
```

### Implementation steps

1. Create `src/geometry/` as a Python package with `__init__.py` if it
   does not already exist.
2. Create `src/geometry/nb01_orbit_propagator_2026_04_05.py` with the
   module header:
   ```python
   """
   NB01 — Orbit propagator: ECI state vectors and geodetic coordinates.

   Spec:        S06_nb01_orbit_propagator_2026-04-16.md
   Spec date:   2026-04-16
   Generated:   YYYY-MM-DD
   Tool:        Claude Code
   Last tested: YYYY-MM-DD  (8/8 tests pass)
   Depends on:  src.constants

   Reference frames: all output vectors are ECI J2000 (TEME approximation).
   BRF / THRF / SIRF transforms are in NB02 (S07).
   See SI-UCAR-WC-RP-004 §2.4.2 for AOCS frame definitions.
   """
   ```
3. Insert the reference frame comment block (above) immediately below the
   import block.
4. Implement `_datetime_to_sgp4_epoch()`, `_build_satrec()`,
   `_eci_to_geodetic()`, and `propagate_orbit()` in that order.
5. Import all physical constants from `src.constants` using the canonical
   symbol names. Do not hardcode GM, Earth radius, or any other constant.
6. Add the defensive assertion `assert np.linalg.norm(pos_m) > 1e6` immediately
   after converting SGP4 output to metres. This catches the km/m unit trap
   during development and can be left in production as a sanity check.
7. In the `propagate_orbit()` docstring, explicitly state: *"Output vectors
   `pos_eci_*` and `vel_eci_*` are in ECI J2000. They are not in BRF, THRF,
   or SIRF. See §2.4 of the spec for AOCS frame definitions."*
8. Write `tests/test_s06_nb01_orbit_propagator.py` with tests T1–T8 exactly
   as written in §7.
9. Run `pytest tests/test_s06_nb01_orbit_propagator.py -v` — all 8 tests
   must pass before committing.
10. Update `src/geometry/__init__.py` to re-export `propagate_orbit`.

### Stop condition

If any test fails after two debugging attempts, stop and return the full
pytest output and a diagnosis note to Claude.ai. Do not loop more than
10 minutes on a failing test without returning.

### Report-back format

After all tests pass, return a report in this exact format for pasting
into Claude.ai:

```
NB01 IMPLEMENTATION REPORT
===========================
Spec version: S06_nb01_orbit_propagator_2026-04-16.md
Tests: T1–T8 all pass (8/8)
Deviations from spec: [none | list any]
Reference frame comment block: inserted at line NN
Module header spec date: 2026-04-16
Files created/modified:
  - src/geometry/nb01_orbit_propagator_2026_04_05.py  (created/modified)
  - src/geometry/__init__.py  (updated)
  - tests/test_s06_nb01_orbit_propagator.py  (created/modified)
```

**Commit message:**
```
feat(geometry): implement S06 NB01 orbit propagator, 8/8 tests pass
Implements: S06_nb01_orbit_propagator_2026-04-16.md
Adds AOCS frame nomenclature (BRF/THRF/SIRF) per SI-UCAR-WC-RP-004 §2.4.2
```
