"""
NB02b — Tangent point: ray-WGS84 ellipsoid intersection.

Spec:        S07_nb02_geometry_2026-04-05.md  Section 4.2
Spec date:   2026-04-05
Generated:   2026-04-07
Depends on:  src.constants, astropy
"""

from __future__ import annotations

import numpy as np
from astropy.time import Time
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation
import astropy.units as u

from src.constants import WGS84_A_M, WGS84_B_M


# ---------------------------------------------------------------------------
# Private helper: ECI → geodetic via astropy
# ---------------------------------------------------------------------------


def _eci_to_geodetic(pos_eci_m: np.ndarray, epoch: Time) -> tuple[float, float, float]:
    """
    Convert an ECI (GCRS) position vector to geodetic lat, lon, alt.

    Returns
    -------
    (lat_deg, lon_deg, alt_km) : floats
    """
    gcrs = GCRS(CartesianRepresentation(pos_eci_m * u.m), obstime=epoch)
    itrs = gcrs.transform_to(ITRS(obstime=epoch))
    loc = itrs.earth_location
    lat_deg = float(loc.lat.deg)
    lon_deg = float(loc.lon.deg)
    alt_km  = float(loc.height.to(u.km).value)
    return lat_deg, lon_deg, alt_km


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def compute_tangent_point(
    pos_eci: np.ndarray,
    los_eci: np.ndarray,
    epoch,
    h_target_km: float = 250.0,
) -> dict:
    """
    Find the tangent point via ray-WGS84 ellipsoid intersection at the
    target emission altitude shell (ellipsoid inflated by h_target).

    The inflated ellipsoid has semi-axes:
        a_eff = WGS84_A_M + h_target_km * 1e3
        b_eff = WGS84_B_M + h_target_km * 1e3

    Parametric ray: P(t) = pos_eci + t * los_eci

    Substituting into the ellipsoid equation
        (Px/a_eff)² + (Py/a_eff)² + (Pz/b_eff)² = 1
    yields a quadratic in t.  The smaller positive root is the near-side
    (atmosphere-entry) intersection.

    If the ray does not intersect the inflated ellipsoid, raises ValueError.

    Parameters
    ----------
    pos_eci : np.ndarray, shape (3,), m — spacecraft ECI position
    los_eci : np.ndarray, shape (3,) — unit LOS vector in ECI
    epoch   : astropy Time — needed for ECI→ECEF→geodetic conversion
    h_target_km : float — target emission altitude, km

    Returns
    -------
    dict with keys:
        'tp_eci'     : np.ndarray, shape (3,) — tangent point ECI, m
        'tp_lat_deg' : float — geodetic latitude, degrees
        'tp_lon_deg' : float — geodetic longitude, degrees (-180 to +180)
        'tp_alt_km'  : float — geodetic altitude, km
    """
    t_epoch = epoch if isinstance(epoch, Time) else Time(epoch, scale="utc")

    h_m    = h_target_km * 1e3
    a_eff  = WGS84_A_M + h_m       # inflated equatorial semi-axis
    b_eff  = WGS84_B_M + h_m       # inflated polar semi-axis

    px, py, pz = pos_eci
    lx, ly, lz = los_eci

    a2 = a_eff ** 2
    b2 = b_eff ** 2

    # Quadratic coefficients: A*t² + B*t + C = 0
    A_coef = (lx**2 + ly**2) / a2 + lz**2 / b2
    B_coef = 2.0 * ((px * lx + py * ly) / a2 + pz * lz / b2)
    C_coef = (px**2 + py**2) / a2 + pz**2 / b2 - 1.0

    discriminant = B_coef**2 - 4.0 * A_coef * C_coef
    if discriminant < 0.0:
        raise ValueError(
            f"Ray does not intersect the inflated WGS84 ellipsoid at "
            f"h_target = {h_target_km} km. "
            f"Check that the LOS points toward Earth and the spacecraft "
            f"altitude is greater than h_target."
        )

    sqrt_disc = np.sqrt(discriminant)
    t1 = (-B_coef - sqrt_disc) / (2.0 * A_coef)
    t2 = (-B_coef + sqrt_disc) / (2.0 * A_coef)

    # Use the smaller positive root (near-side intersection)
    candidates = [t for t in (t1, t2) if t > 0.0]
    if not candidates:
        raise ValueError(
            f"Both ray-ellipsoid intersection roots are non-positive (t1={t1:.1f}, t2={t2:.1f}). "
            f"The spacecraft may be below the target altitude or the LOS points away from Earth."
        )
    t_near = min(candidates)

    tp_eci = pos_eci + t_near * los_eci

    lat_deg, lon_deg, alt_km = _eci_to_geodetic(tp_eci, t_epoch)

    return {
        "tp_eci":     tp_eci,
        "tp_lat_deg": lat_deg,
        "tp_lon_deg": lon_deg,
        "tp_alt_km":  alt_km,
    }
