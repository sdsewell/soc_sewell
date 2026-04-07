"""
NB02c — LOS projection: ENU vectors, Earth rotation, and v_rel.

Spec:        S07_nb02_geometry_2026-04-05.md  Section 4.3
Spec date:   2026-04-05
Generated:   2026-04-07
Depends on:  src.constants, astropy
"""

from __future__ import annotations

import numpy as np
from astropy.time import Time
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation
import astropy.units as u

from src.constants import EARTH_OMEGA_RAD_S


# ---------------------------------------------------------------------------
# Private helper: rotate a direction vector from ITRS to GCRS (ECI)
# ---------------------------------------------------------------------------


def _itrs_to_gcrs_vector(vec_itrs: np.ndarray, epoch: Time) -> np.ndarray:
    """
    Rotate a direction unit vector from ITRS (ECEF) to GCRS (ECI).

    ITRS → GCRS is purely rotational (origin fixed at Earth centre), so
    transforming the Cartesian coordinates of a 1 m unit vector gives the
    ECI direction.  The result is then renormalised for numerical hygiene.

    Never cache between epochs — the rotation is epoch-dependent (precession
    + nutation).
    """
    cr = CartesianRepresentation(
        vec_itrs[0] * u.m, vec_itrs[1] * u.m, vec_itrs[2] * u.m
    )
    itrs_coord = ITRS(cr, obstime=epoch)
    gcrs_coord = itrs_coord.transform_to(GCRS(obstime=epoch))
    xyz = np.array([
        gcrs_coord.cartesian.x.to(u.m).value,
        gcrs_coord.cartesian.y.to(u.m).value,
        gcrs_coord.cartesian.z.to(u.m).value,
    ])
    return xyz / np.linalg.norm(xyz)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def enu_unit_vectors_eci(
    lat_deg: float,
    lon_deg: float,
    epoch,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute East, North, Up unit vectors in ECI at a geodetic location.

    Uses astropy ITRS→GCRS for the ECEF→ECI rotation.
    This transform is epoch-dependent (precession + nutation).
    Never cache across epochs.

    The ENU basis in ECEF (ITRS), for geodetic lat φ, lon λ:
        East  = [-sin(λ),            cos(λ),           0        ]
        North = [-sin(φ)·cos(λ), -sin(φ)·sin(λ),  cos(φ)  ]
        Up    = [ cos(φ)·cos(λ),  cos(φ)·sin(λ),  sin(φ)  ]

    Parameters
    ----------
    lat_deg : float — geodetic latitude, degrees
    lon_deg : float — geodetic longitude, degrees
    epoch   : astropy Time (or string parseable as utc)

    Returns
    -------
    (e_east_eci, e_north_eci, e_up_eci) — each shape (3,), unit vectors in ECI
    """
    t = epoch if isinstance(epoch, Time) else Time(epoch, scale="utc")

    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)
    sin_lat, cos_lat = np.sin(lat_rad), np.cos(lat_rad)
    sin_lon, cos_lon = np.sin(lon_rad), np.cos(lon_rad)

    # ENU unit vectors in ECEF (ITRS)
    e_east_itrs  = np.array([-sin_lon,               cos_lon,              0.0      ])
    e_north_itrs = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon,  cos_lat])
    e_up_itrs    = np.array([ cos_lat * cos_lon,  cos_lat * sin_lon,  sin_lat])

    # Rotate each vector to ECI
    e_east_eci  = _itrs_to_gcrs_vector(e_east_itrs,  t)
    e_north_eci = _itrs_to_gcrs_vector(e_north_itrs, t)
    e_up_eci    = _itrs_to_gcrs_vector(e_up_itrs,    t)

    return e_east_eci, e_north_eci, e_up_eci


def earth_rotation_velocity_eci(
    tp_eci: np.ndarray,
) -> np.ndarray:
    """
    Velocity of the atmosphere at the tangent point due to Earth rotation.

    v_earth = omega_E × r_tp  (cross product)
    omega_E = [0, 0, EARTH_OMEGA_RAD_S] in ECI (aligned with rotation axis)

    For a point at position r = [rx, ry, rz]:
        v_earth = [0, 0, ω] × [rx, ry, rz]
                = [-ω·ry,  ω·rx,  0]

    Parameters
    ----------
    tp_eci : np.ndarray, shape (3,), m — tangent point ECI position

    Returns
    -------
    v_earth_eci : np.ndarray, shape (3,), m/s
    """
    omega = EARTH_OMEGA_RAD_S
    omega_vec = np.array([0.0, 0.0, omega])
    return np.cross(omega_vec, tp_eci)


def compute_v_rel(
    wind_map,
    tp_lat_deg: float,
    tp_lon_deg: float,
    tp_eci: np.ndarray,
    vel_eci: np.ndarray,
    los_eci: np.ndarray,
    epoch,
) -> dict:
    """
    Compute v_rel — the FPI's measured Doppler shift observable.

    v_rel = v_wind_LOS - V_sc_LOS - v_earth_LOS

    where all three terms are LOS projections:
      v_wind_LOS  = (e_east·v_zonal + e_north·v_merid) · los_eci
      V_sc_LOS    = vel_eci · los_eci
      v_earth_LOS = (omega_E × tp_eci) · los_eci

    Sign convention: positive v_rel = emitting gas receding from instrument
    (redshift), corresponding to LOS wind away from the spacecraft.

    Parameters
    ----------
    wind_map    : WindMap — called as wind_map.sample(tp_lat_deg, tp_lon_deg)
                  Returns (v_zonal_ms, v_merid_ms).
    tp_lat_deg  : float — tangent point geodetic latitude, degrees
    tp_lon_deg  : float — tangent point geodetic longitude, degrees
    tp_eci      : np.ndarray, shape (3,), m — tangent point ECI position
    vel_eci     : np.ndarray, shape (3,), m/s — spacecraft ECI velocity
    los_eci     : np.ndarray, shape (3,) — unit LOS vector in ECI
    epoch       : astropy Time

    Returns
    -------
    dict with keys:
        'v_rel'        : float, m/s — full observable (what FPI measures)
        'v_wind_LOS'   : float, m/s — atmospheric wind LOS component
        'V_sc_LOS'     : float, m/s — spacecraft velocity LOS component
        'v_earth_LOS'  : float, m/s — Earth rotation LOS component
        'v_zonal_ms'   : float, m/s — zonal wind at tangent point
        'v_merid_ms'   : float, m/s — meridional wind at tangent point
    """
    t = epoch if isinstance(epoch, Time) else Time(epoch, scale="utc")

    # Sample wind at the tangent point
    v_zonal_ms, v_merid_ms = wind_map.sample(tp_lat_deg, tp_lon_deg)

    # ENU unit vectors in ECI at tangent point
    e_east_eci, e_north_eci, _ = enu_unit_vectors_eci(tp_lat_deg, tp_lon_deg, t)

    # Atmospheric wind velocity in ECI (no vertical component)
    v_wind_eci = v_zonal_ms * e_east_eci + v_merid_ms * e_north_eci

    # LOS projections
    v_wind_LOS  = float(np.dot(v_wind_eci, los_eci))
    V_sc_LOS    = float(np.dot(vel_eci, los_eci))
    v_earth_LOS = float(np.dot(earth_rotation_velocity_eci(tp_eci), los_eci))

    # Full Doppler observable
    v_rel = v_wind_LOS - V_sc_LOS - v_earth_LOS

    return {
        "v_rel":       v_rel,
        "v_wind_LOS":  v_wind_LOS,
        "V_sc_LOS":    V_sc_LOS,
        "v_earth_LOS": v_earth_LOS,
        "v_zonal_ms":  float(v_zonal_ms),
        "v_merid_ms":  float(v_merid_ms),
    }
