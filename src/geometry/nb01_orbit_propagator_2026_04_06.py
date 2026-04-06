"""
NB01 — Orbit propagator: ECI state vectors and geodetic coordinates.

Spec:        S06_nb01_orbit_propagator_2026-04-06.md
Spec date:   2026-04-05
Generated:   2026-04-05
Tool:        Copilot
Last tested: YYYY-MM-DD  (8/8 tests pass)
Depends on:  src.constants
"""

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

def _datetime_to_sgp4_epoch(dt):
    t = Time(dt, scale='utc')
    year = t.datetime.year
    day_of_year = t.datetime.timetuple().tm_yday
    # SGP4 expects (year, fractional day of year)
    frac_day = (t.datetime.hour + t.datetime.minute/60 + t.datetime.second/3600 + t.datetime.microsecond/3.6e9)/24
    return year, day_of_year + frac_day

def _build_satrec(epoch_dt, altitude_km, inclination_deg, raan_deg, bstar):
    a_km = WGS84_A_M / 1e3 + altitude_km
    n_rad_s = np.sqrt(EARTH_GRAV_PARAM_M3_S2 / (a_km * 1e3)**3)
    n_rad_min = n_rad_s * 60  # SGP4 expects mean motion in radians per minute
    year, epochdays = _datetime_to_sgp4_epoch(epoch_dt)
    satrec = Satrec()
    satrec.sgp4init(
        WGS84,           # gravity model
        0,               # satellite number (arbitrary)
        year,            # epoch year
        epochdays,       # epoch day of year + fractional day
        bstar,           # drag term
        0.0,             # ndot (ignored)
        0.0,             # nddot (ignored)
        0.0,             # ecco (eccentricity, circular)
        0.0,             # argpo (argument of perigee)
        np.radians(inclination_deg), # inclo (inclination, radians)
        0.0,             # mo (mean anomaly at epoch)
        n_rad_min,       # no_kozai (mean motion, radians/minute)
        np.radians(raan_deg) # nodeo (RAAN, radians)
    )
    return satrec

def _eci_to_geodetic(pos_eci_m, epoch):
    t = Time(epoch, scale='utc') if not isinstance(epoch, Time) else epoch
    gcrs = GCRS(CartesianRepresentation(pos_eci_m * u.m), obstime=t)
    itrs = gcrs.transform_to(ITRS(obstime=t))
    loc = itrs.earth_location
    return float(loc.lat.deg), float(loc.lon.deg), float(loc.height.to(u.km).value)

def propagate_orbit(
    t_start: str,
    duration_s: float,
    dt_s: float = SCIENCE_CADENCE_S,
    altitude_km: float = SC_ALTITUDE_KM,
    inclination_deg: float = 97.44,
    raan_deg: float = 90.0,
    bstar: float = 0.0,
) -> pd.DataFrame:
    epochs = pd.date_range(start=pd.to_datetime(t_start, utc=True), periods=int(duration_s//dt_s)+1, freq=pd.Timedelta(seconds=dt_s))
    satrec = _build_satrec(epochs[0].to_pydatetime(), altitude_km, inclination_deg, raan_deg, bstar)
    data = []
    for epoch in epochs:
        jd, fr = jday(epoch.year, epoch.month, epoch.day, epoch.hour, epoch.minute, epoch.second + epoch.microsecond/1e6)
        e, pos_km, vel_km_s = satrec.sgp4(jd, fr)
        assert e == 0, f"SGP4 error code {e}"
        pos_m = np.array(pos_km) * 1e3
        vel_m_s = np.array(vel_km_s) * 1e3
        assert np.linalg.norm(pos_m) > 1e6, f"pos_eci looks like km not metres: |r| = {np.linalg.norm(pos_m):.1f}"
        lat_deg, lon_deg, alt_km = _eci_to_geodetic(pos_m, epoch)
        speed_ms = np.linalg.norm(vel_m_s)
        data.append([
            epoch,
            pos_m[0], pos_m[1], pos_m[2],
            vel_m_s[0], vel_m_s[1], vel_m_s[2],
            lat_deg, lon_deg, alt_km, speed_ms
        ])
    df = pd.DataFrame(data, columns=[
        'epoch', 'pos_eci_x', 'pos_eci_y', 'pos_eci_z',
        'vel_eci_x', 'vel_eci_y', 'vel_eci_z',
        'lat_deg', 'lon_deg', 'alt_km', 'speed_ms'
    ])
    df.index = np.arange(len(df))
    return df
