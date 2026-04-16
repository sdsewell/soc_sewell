"""
NB02b — Tangent point: ray-WGS84 ellipsoid intersection at 250 km shell.

Spec:      NB02_geometry_2026-04-16.md
Spec date: 2026-04-16
Generated: 2026-04-16
Tool:      Claude Code

Input los_eci is the -X_BRF boresight rotated to ECI by NB02a.
"""

from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
from astropy.time import Time
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation
import astropy.units as u

from src.constants import WGS84_A_M, WGS84_B_M


def _eci_to_geodetic(pos_eci_m: np.ndarray, epoch: Time) -> tuple[float, float, float]:
    """Convert an ECI (GCRS) position vector to geodetic lat, lon, alt."""
    gcrs = GCRS(CartesianRepresentation(pos_eci_m * u.m), obstime=epoch)
    itrs = gcrs.transform_to(ITRS(obstime=epoch))
    loc = itrs.earth_location
    return float(loc.lat.deg), float(loc.lon.deg), float(loc.height.to(u.km).value)


def compute_tangent_point(
    pos_eci: np.ndarray,
    los_eci: np.ndarray,
    epoch,
    h_target_km: float = 250.0,
) -> dict:
    """
    Find the tangent point via ray-WGS84 ellipsoid intersection at the
    target emission altitude shell (ellipsoid inflated by h_target).

    Uses the smaller positive root (near-side intersection). If the ray
    does not intersect the inflated ellipsoid, raises ValueError.

    The los_eci input is produced by NB02a compute_los_eci() and points
    in the -X_BRF direction rotated to ECI. For along_track mode the
    tangent point will be ~923 km ahead of the spacecraft in the velocity
    direction (THRF along-track configuration). For cross_track mode it
    will be ~923 km to the side (THRF cross-track configuration).

    The AOCS THRF model error requirement is < 5 km from the true 250 km
    tangent height (SI-UCAR-WC-RP-004 §2.1). The WGS84 oblateness causes
    tp_alt_km to differ from h_target_km by up to ~5 km; this is expected
    and verified by test T5.

    Parameters
    ----------
    pos_eci : np.ndarray, shape (3,), m — spacecraft ECI position
    los_eci : np.ndarray, shape (3,) — unit LOS vector in ECI
    epoch   : astropy Time — needed for ECI→ECEF→geodetic conversion
    h_target_km : float — target emission altitude, km

    Returns
    -------
    dict with keys:
        'tp_eci'     : np.ndarray, shape (3,) — tangent point ECI position, m
        'tp_lat_deg' : float — geodetic latitude, degrees
        'tp_lon_deg' : float — geodetic longitude, degrees (-180 to +180)
        'tp_alt_km'  : float — geodetic altitude, km
                       (will differ slightly from h_target_km due to WGS84
                       oblateness; expected error < 5 km — see T5)
    """
    t_epoch = epoch if isinstance(epoch, Time) else Time(epoch, scale="utc")

    h_m   = h_target_km * 1e3
    a_eff = WGS84_A_M + h_m
    b_eff = WGS84_B_M + h_m

    px, py, pz = pos_eci
    lx, ly, lz = los_eci

    a2 = a_eff ** 2
    b2 = b_eff ** 2

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

    candidates = [t for t in (t1, t2) if t > 0.0]
    if not candidates:
        raise ValueError(
            f"Both ray-ellipsoid intersection roots are non-positive "
            f"(t1={t1:.1f}, t2={t2:.1f}). "
            f"The spacecraft may be below the target altitude or the LOS "
            f"points away from Earth."
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
