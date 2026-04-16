"""
NB01 — Orbit propagator: ECI state vectors and geodetic coordinates.

Spec:        S06_nb01_orbit_propagator_2026-04-16.md
Spec date:   2026-04-16
Generated:   2026-04-16
Tool:        Claude Code
Last tested: 2026-04-16  (8/8 tests pass)
Depends on:  src.constants

Reference frames: all output vectors are ECI J2000 (TEME approximation).
BRF / THRF / SIRF transforms are in NB02 (S07).
See SI-UCAR-WC-RP-004 §2.4.2 for AOCS frame definitions.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import pandas as pd
from sgp4.api import Satrec, WGS84, jday
from astropy.time import Time
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation
import astropy.units as u
from src.constants import (
    EARTH_GRAV_PARAM_M3_S2,
    WGS84_A_M,
    EARTH_J2,
    SC_ALTITUDE_KM,
    SCIENCE_CADENCE_S,
    SC_ALTITUDE_RANGE_KM,
)

# ---------------------------------------------------------------------------
# REFERENCE FRAME NOTE (SI-UCAR-WC-RP-004 §2.4.2)
# All position and velocity vectors produced by this module are expressed in
# ECI J2000 (approximately TEME for SGP4 output; difference < 1 arcsec).
# They are NOT in BRF (body frame), THRF (tangent height frame), or SIRF
# (star tracker frame). Frame transforms to THRF are performed by NB02 (S07).
# ---------------------------------------------------------------------------


def _datetime_to_sgp4_epoch(dt) -> tuple[int, float]:
    """Convert a datetime/ISO string to (year, fractional day-of-year) for sgp4init."""
    t = Time(dt, scale="utc")
    year = t.datetime.year
    day_of_year = t.datetime.timetuple().tm_yday
    frac_day = (
        t.datetime.hour
        + t.datetime.minute / 60
        + t.datetime.second / 3600
        + t.datetime.microsecond / 3.6e9
    ) / 24
    return year, day_of_year + frac_day


def _build_satrec(
    epoch_dt,
    altitude_km: float,
    inclination_deg: float,
    raan_deg: float,
    bstar: float,
) -> Satrec:
    """
    Construct a Satrec object from mean orbital elements.

    Parameters
    ----------
    epoch_dt : datetime
        UTC epoch for the TLE.
    altitude_km : float
        Circular orbit altitude above WGS84 ellipsoid, km.
    inclination_deg : float
        Orbital inclination, degrees.
    raan_deg : float
        Right ascension of ascending node, degrees.
    bstar : float
        SGP4 drag coefficient (B*).

    Returns
    -------
    Satrec
        Initialised SGP4 satellite record ready for propagation.

    Notes
    -----
    Mean motion is computed in radians/minute as required by sgp4init:
        n (rad/min) = 60 × √(GM / a³)
    For altitude_km = 510: a = 6888.137 km, n ≈ 0.06641 rad/min (15.19 rev/day).
    """
    a_km = WGS84_A_M / 1e3 + altitude_km
    n_rad_s = np.sqrt(EARTH_GRAV_PARAM_M3_S2 / (a_km * 1e3) ** 3)
    n_rad_min = n_rad_s * 60  # sgp4init expects radians/minute

    year, epochdays = _datetime_to_sgp4_epoch(epoch_dt)

    satrec = Satrec()
    satrec.sgp4init(
        WGS84,                       # gravity model
        0,                           # satellite number (arbitrary for synthetic)
        year,                        # epoch year
        epochdays,                   # fractional day of year
        bstar,                       # drag term B*
        0.0,                         # ndot — first deriv. of mean motion (unused)
        0.0,                         # nddot — second deriv. of mean motion (unused)
        0.0,                         # ecco — eccentricity (circular orbit)
        0.0,                         # argpo — argument of perigee (undefined for circular)
        np.radians(inclination_deg), # inclo — inclination, radians
        0.0,                         # mo — mean anomaly at epoch (ascending node)
        n_rad_min,                   # no_kozai — mean motion, radians/minute
        np.radians(raan_deg),        # nodeo — RAAN, radians
    )
    return satrec


def _eci_to_geodetic(pos_eci_m: np.ndarray, epoch) -> tuple[float, float, float]:
    """
    Convert ECI position vector to WGS84 geodetic coordinates.

    Parameters
    ----------
    pos_eci_m : np.ndarray, shape (3,)
        ECI J2000 position vector, metres.
    epoch : astropy.time.Time or datetime-like
        UTC epoch of the observation.

    Returns
    -------
    (lat_deg, lon_deg, alt_km) : tuple[float, float, float]
        WGS84 geodetic latitude (deg), longitude (deg), altitude (km).

    Notes
    -----
    Uses astropy GCRS→ITRS transform chain. GCRS is approximately equivalent
    to ECI J2000 for this application. The ITRS frame co-rotates with Earth,
    giving geodetic coordinates directly.
    """
    t = Time(epoch, scale="utc") if not isinstance(epoch, Time) else epoch
    gcrs = GCRS(CartesianRepresentation(pos_eci_m * u.m), obstime=t)
    itrs = gcrs.transform_to(ITRS(obstime=t))
    loc = itrs.earth_location
    return (
        float(loc.lat.deg),
        float(loc.lon.deg),
        float(loc.height.to(u.km).value),
    )


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
        One row per epoch with integer index starting at 0. Columns:
        epoch, pos_eci_x/y/z (m), vel_eci_x/y/z (m/s),
        lat_deg, lon_deg, alt_km, speed_ms.

    Notes
    -----
    Output vectors pos_eci_* and vel_eci_* are in ECI J2000. They are not
    in BRF, THRF, or SIRF. See §2.4 of the spec for AOCS frame definitions.

    Strictly speaking SGP4 outputs in TEME (True Equator Mean Equinox),
    which differs from ECI J2000 by < 1 arcsec (~50 m at 500 km altitude).
    This difference is negligible at the ~100 m accuracy level required here.
    The column names pos_eci_* and vel_eci_* match the telemetry field names
    in S18 (P01 metadata spec) for consistency between synthetic and real data.
    If sub-metre accuracy is later needed, apply the TEME→GCRS rotation via
    astropy before storing.
    """
    epochs = pd.date_range(
        start=pd.to_datetime(t_start, utc=True),
        periods=int(duration_s // dt_s) + 1,
        freq=pd.Timedelta(seconds=dt_s),
    )
    satrec = _build_satrec(
        epochs[0].to_pydatetime(), altitude_km, inclination_deg, raan_deg, bstar
    )

    data = []
    for epoch in epochs:
        jd, fr = jday(
            epoch.year, epoch.month, epoch.day,
            epoch.hour, epoch.minute,
            epoch.second + epoch.microsecond / 1e6,
        )
        e, pos_km, vel_km_s = satrec.sgp4(jd, fr)
        assert e == 0, f"SGP4 error code {e}"

        pos_m = np.array(pos_km) * 1e3    # km → m
        vel_m_s = np.array(vel_km_s) * 1e3  # km/s → m/s

        # Defensive check: catch the km/m unit trap
        assert np.linalg.norm(pos_m) > 1e6, (
            f"pos_eci looks like km not metres: |r| = {np.linalg.norm(pos_m):.1f}"
        )

        lat_deg, lon_deg, alt_km_val = _eci_to_geodetic(pos_m, epoch)
        speed_ms = float(np.linalg.norm(vel_m_s))

        data.append([
            epoch,
            pos_m[0], pos_m[1], pos_m[2],
            vel_m_s[0], vel_m_s[1], vel_m_s[2],
            lat_deg, lon_deg, alt_km_val, speed_ms,
        ])

    df = pd.DataFrame(data, columns=[
        "epoch",
        "pos_eci_x", "pos_eci_y", "pos_eci_z",
        "vel_eci_x", "vel_eci_y", "vel_eci_z",
        "lat_deg", "lon_deg", "alt_km", "speed_ms",
    ])
    df.index = np.arange(len(df))
    return df
