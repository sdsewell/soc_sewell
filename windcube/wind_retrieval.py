"""
H07 — Thermospheric wind vector retrieval from FPI LOS Doppler velocities.

Spec:        specs/H07_wind_vector_retrieval_2026-05-14_v03.md (v0.3)
Spec date:   2026-05-14
Generated:   2026-05-14
Tool:        Claude Code
Last tested: 2026-05-14  (8/8 tests pass)

Boresight:   -X_BRF per SI-UCAR-WC-RP-004 §2.4.2.1
Convention:  NB02c sign convention (nb02c_los_projection_2026_04_16.py)
             v_corrected = v_rel + V_sc_LOS + v_earth_LOS
             Design matrix: A[i] = [+L_E, +L_N]
Quaternion:  scalar-last [x,y,z,w] (P01 pipeline convention)
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.spatial.transform import Rotation

import astropy.units as u
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation
from astropy.time import Time

from src.constants import EARTH_OMEGA_RAD_S
from windcube.constants import (
    GDOP_MAX,
    H07_BORESIGHT_BODY,
    N_MIN_FRAMES,
    WGS84_A_M,
    WGS84_B_M,
)

if TYPE_CHECKING:
    from src.metadata import ImageMetadata

# ---------------------------------------------------------------------------
# Module-level constant
# ---------------------------------------------------------------------------

BORESIGHT_BODY: np.ndarray = np.array(H07_BORESIGHT_BODY)   # -X_BRF

# ---------------------------------------------------------------------------
# Data structures (§9.1)
# ---------------------------------------------------------------------------


@dataclass
class LOSGeometry:
    """Per-frame geometry computed from ImageMetadata."""

    l_hat_eci: np.ndarray       # (3,) SC→TP unit vector, ECI
    l_hat_ecef: np.ndarray      # (3,) SC→TP unit vector, ECEF (via astropy)
    L_E: float                  # dot(l̂, Ê)  eastward direction cosine
    L_N: float                  # dot(l̂, N̂)  northward direction cosine
    L_Z: float                  # dot(l̂, Ẑ)  upward direction cosine
    tangent_lat_deg: float      # geodetic latitude of tangent point, deg
    tangent_lon_deg: float      # geodetic longitude of tangent point, deg
    tangent_alt_km: float       # geodetic altitude of tangent point, km
    tangent_pos_eci: np.ndarray # (3,) tangent point ECI position, m
    V_sc_LOS: float             # dot(v_sc, l̂)    approach-positive, m/s
    v_earth_LOS: float          # dot(v_earth, l̂) approach-positive, m/s


@dataclass
class LOSObservation:
    """Corrected LOS velocity observation ready for wind inversion."""

    v_corrected: float     # v_wind_LOS = v_rel + V_sc_LOS + v_earth_LOS, m/s
    sigma_v: float         # 1-sigma uncertainty from M06, m/s
    L_E: float             # eastward direction cosine
    L_N: float             # northward direction cosine
    L_Z: float             # upward direction cosine
    tangent_lat_deg: float # deg
    tangent_lon_deg: float # deg
    tangent_alt_km: float  # km
    epoch_unix_ms: int     # Unix ms
    obs_mode: str          # 'along_track' or 'cross_track'


@dataclass
class WindSolution:
    """Wind vector retrieved from a spatiotemporal bin."""

    v_E: float                  # eastward wind, m/s (+ = eastward)
    v_N: float                  # northward wind, m/s (+ = northward)
    sigma_v_E: float            # 1-sigma uncertainty, m/s
    sigma_v_N: float            # 1-sigma uncertainty, m/s
    two_sigma_v_E: float        # = 2.0 * sigma_v_E exactly
    two_sigma_v_N: float        # = 2.0 * sigma_v_N exactly
    n_frames: int
    gdop_flag: bool
    n_frames_flag: bool
    condition_number: float
    mean_tangent_lat_deg: float
    mean_tangent_lon_deg: float
    mean_tangent_alt_km: float
    mean_epoch_unix_ms: int
    obs_modes: set


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _itrs_to_eci_unit(vec_itrs: np.ndarray, epoch: Time) -> np.ndarray:
    """Rotate an ITRS unit vector to ECI (GCRS) at the given epoch."""
    cr = CartesianRepresentation(
        vec_itrs[0] * u.m, vec_itrs[1] * u.m, vec_itrs[2] * u.m
    )
    itrs = ITRS(cr, obstime=epoch)
    gcrs = itrs.transform_to(GCRS(obstime=epoch))
    xyz = np.array([
        gcrs.cartesian.x.to(u.m).value,
        gcrs.cartesian.y.to(u.m).value,
        gcrs.cartesian.z.to(u.m).value,
    ])
    norm = np.linalg.norm(xyz)
    return xyz / norm


def _enu_eci(lat_rad: float, lon_rad: float, epoch: Time):
    """Return ENU unit vectors (Ê, N̂, Ẑ) in ECI at the given tangent point."""
    phi, lam = lat_rad, lon_rad
    E_itrs = np.array([-np.sin(lam),                       np.cos(lam),          0.0])
    N_itrs = np.array([-np.sin(phi) * np.cos(lam), -np.sin(phi) * np.sin(lam), np.cos(phi)])
    Z_itrs = np.array([ np.cos(phi) * np.cos(lam),  np.cos(phi) * np.sin(lam), np.sin(phi)])
    E_eci = _itrs_to_eci_unit(E_itrs, epoch)
    N_eci = _itrs_to_eci_unit(N_itrs, epoch)
    Z_eci = _itrs_to_eci_unit(Z_itrs, epoch)
    return E_eci, N_eci, Z_eci


def _ecef_to_geodetic(x: float, y: float, z: float):
    """
    ECEF to WGS84 geodetic (lat_rad, lon_rad, alt_m) using Bowring iteration.
    Accurate to sub-mm for altitudes 0–2000 km.
    """
    a, b = WGS84_A_M, WGS84_B_M
    e2 = 1.0 - (b / a) ** 2
    ep2 = (a / b) ** 2 - 1.0
    p = np.sqrt(x ** 2 + y ** 2)
    lon = np.arctan2(y, x)
    # Bowring iteration
    lat = np.arctan2(z, p * (1.0 - e2))
    for _ in range(10):
        N = a / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
        lat_new = np.arctan2(z + e2 * N * np.sin(lat), p)
        if abs(lat_new - lat) < 1e-12:
            lat = lat_new
            break
        lat = lat_new
    N = a / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - N if abs(np.cos(lat)) > 1e-10 else abs(z) / np.sin(lat) - N * (1 - e2)
    return lat, lon, alt


def _wgs84_radius(lat_rad: float) -> float:
    """Geocentric radius of WGS84 ellipsoid at geodetic latitude."""
    a, b = WGS84_A_M, WGS84_B_M
    cos_lat = np.cos(lat_rad)
    sin_lat = np.sin(lat_rad)
    num = (a ** 2 * cos_lat) ** 2 + (b ** 2 * sin_lat) ** 2
    den = (a * cos_lat) ** 2 + (b * sin_lat) ** 2
    return np.sqrt(num / den)


def _ray_geodetic_alt(s: float, r_sc: np.ndarray, l_hat: np.ndarray, h_target_m: float) -> float:
    """Residual: geodetic altitude at ray parameter s minus h_target_m."""
    r = r_sc + s * l_hat
    lat, _, alt = _ecef_to_geodetic(r[0], r[1], r[2])
    return alt - h_target_m


def _eci_to_ecef_vec(vec_eci: np.ndarray, epoch: Time) -> np.ndarray:
    """Rotate an ECI (GCRS) vector to ECEF (ITRS) at the given epoch."""
    cr = CartesianRepresentation(
        vec_eci[0] * u.m, vec_eci[1] * u.m, vec_eci[2] * u.m
    )
    gcrs = GCRS(cr, obstime=epoch)
    itrs = gcrs.transform_to(ITRS(obstime=epoch))
    return np.array([
        itrs.cartesian.x.to(u.m).value,
        itrs.cartesian.y.to(u.m).value,
        itrs.cartesian.z.to(u.m).value,
    ])


def _geodetic_altitude(r_ecef: np.ndarray) -> float:
    """Geodetic altitude above WGS84 in metres."""
    _, _, alt_m = _ecef_to_geodetic(r_ecef[0], r_ecef[1], r_ecef[2])
    return alt_m


def _parse_epoch(meta) -> Time:
    """Parse ImageMetadata.utc_timestamp to astropy Time."""
    ts = meta.utc_timestamp
    # Strip trailing +00:00 if present (astropy isot doesn't accept offset)
    if ts.endswith("+00:00"):
        ts = ts[:-6]
    return Time(ts, format="isot", scale="utc")


# ---------------------------------------------------------------------------
# Public API (§9.2)
# ---------------------------------------------------------------------------


def derive_obs_mode(
    attitude_quaternion: list,
    pos_eci: np.ndarray,
    vel_eci: np.ndarray,
    ambiguity_threshold: float = 0.707,
) -> str:
    """
    Derive 'along_track' or 'cross_track' from attitude quaternion.

    Projects the boresight [-1,0,0]_BRF (rotated to ECI) onto the forward
    velocity direction and the orbit normal. The larger projection identifies
    the look mode.

    Returns 'along_track', 'cross_track', or 'unknown'.
    """
    R = Rotation.from_quat(attitude_quaternion)
    boresight_eci = R.apply(BORESIGHT_BODY)
    boresight_eci /= np.linalg.norm(boresight_eci)

    nadir = -np.array(pos_eci) / np.linalg.norm(pos_eci)

    v_hat = np.array(vel_eci) / np.linalg.norm(vel_eci)
    v_horiz = v_hat - np.dot(v_hat, nadir) * nadir
    norm_vh = np.linalg.norm(v_horiz)
    if norm_vh < 1e-10:
        return "unknown"
    v_horiz /= norm_vh

    orbit_normal = np.cross(np.array(pos_eci), np.array(vel_eci))
    orbit_normal /= np.linalg.norm(orbit_normal)

    dot_v = abs(float(np.dot(boresight_eci, v_horiz)))
    dot_n = abs(float(np.dot(boresight_eci, orbit_normal)))

    if max(dot_v, dot_n) < ambiguity_threshold:
        return "unknown"

    return "along_track" if dot_v > dot_n else "cross_track"


def compute_los_geometry(meta) -> LOSGeometry:
    """
    Full Stage G: compute LOS geometry from ImageMetadata.

    Raises ValueError if meta.img_type != 'science'.
    Uses meta.h_target_km_obs if set; defaults to 250.0 km with a warning.
    For synthetic frames: uses stored tangent point if available.
    """
    if meta.img_type != "science":
        raise ValueError(
            f"compute_los_geometry requires img_type='science', got '{meta.img_type}'"
        )

    # ── Boresight → l̂ in ECI ──────────────────────────────────────────────
    q = meta.attitude_quaternion          # [x, y, z, w] scalar-last
    R_body2eci = Rotation.from_quat(q)
    l_hat_eci = R_body2eci.apply(BORESIGHT_BODY)
    l_hat_eci /= np.linalg.norm(l_hat_eci)

    # ── Target altitude ───────────────────────────────────────────────────
    if meta.h_target_km_obs is not None:
        h_target_km = float(meta.h_target_km_obs)
    else:
        warnings.warn(
            "h_target_km_obs not set in metadata (pre-v2 sidecar). "
            "Defaulting to 250.0 km. Re-ingest to populate correctly.",
            UserWarning,
            stacklevel=2,
        )
        h_target_km = 250.0

    # ── Epoch ─────────────────────────────────────────────────────────────
    epoch = _parse_epoch(meta)

    # ── Tangent point ─────────────────────────────────────────────────────
    pos_eci = np.array(meta.pos_eci_hat, dtype=float)

    if getattr(meta, "is_synthetic", False) and getattr(meta, "tangent_lat", None) is not None:
        # Synthetic frame: use stored tangent point, skip ray-trace
        tangent_lat_deg = float(meta.tangent_lat)
        tangent_lon_deg = float(meta.tangent_lon) if hasattr(meta, "tangent_lon") else 0.0
        tangent_alt_km = h_target_km
        lat_rad = np.radians(tangent_lat_deg)
        lon_rad = np.radians(tangent_lon_deg)
        # Approximate tangent ECI position from geodetic
        N_e = WGS84_A_M / np.sqrt(
            1.0 - (1.0 - (WGS84_B_M / WGS84_A_M) ** 2) * np.sin(lat_rad) ** 2
        )
        r_tp = (N_e + h_target_km * 1e3) * np.array([
            np.cos(lat_rad) * np.cos(lon_rad),
            np.cos(lat_rad) * np.sin(lon_rad),
            (WGS84_B_M ** 2 / WGS84_A_M ** 2 * N_e + h_target_km * 1e3) * np.sin(lat_rad)
            / (N_e + h_target_km * 1e3),
        ])
        # Restore to proper ECEF form
        r_tp_ecef = np.array([
            (N_e + h_target_km * 1e3) * np.cos(lat_rad) * np.cos(lon_rad),
            (N_e + h_target_km * 1e3) * np.cos(lat_rad) * np.sin(lon_rad),
            (WGS84_B_M ** 2 / WGS84_A_M ** 2 * N_e + h_target_km * 1e3) * np.sin(lat_rad),
        ])
        # Rotate ECEF→ECI via astropy
        cr = CartesianRepresentation(
            r_tp_ecef[0] * u.m, r_tp_ecef[1] * u.m, r_tp_ecef[2] * u.m
        )
        itrs_tp = ITRS(cr, obstime=epoch)
        gcrs_tp = itrs_tp.transform_to(GCRS(obstime=epoch))
        tangent_pos_eci = np.array([
            gcrs_tp.cartesian.x.to(u.m).value,
            gcrs_tp.cartesian.y.to(u.m).value,
            gcrs_tp.cartesian.z.to(u.m).value,
        ])
    else:
        # Real frame: ray-trace to find tangent point
        pos_ecef = _eci_to_ecef_vec(pos_eci, epoch)
        l_hat_ecef_tmp = _eci_to_ecef_vec(l_hat_eci * 1e6, epoch) / 1e6  # rotate direction
        l_hat_ecef_tmp /= np.linalg.norm(l_hat_ecef_tmp)
        h_target_m = h_target_km * 1e3
        s_max = 2.0 * np.linalg.norm(pos_ecef)
        try:
            s_star = brentq(
                _ray_geodetic_alt, 0.0, s_max,
                args=(pos_ecef, l_hat_ecef_tmp, h_target_m),
                xtol=1.0,
            )
        except ValueError:
            # Bracketing failure: ray floor is slightly above (or below) the
            # target altitude due to floating-point geometry.  Find the
            # minimum-altitude point along the ray and use it as the tangent
            # point regardless of whether it exactly equals h_target_km.
            def _ray_alt_m(s):
                r = pos_ecef + s * l_hat_ecef_tmp
                return _geodetic_altitude(r)

            res = minimize_scalar(_ray_alt_m, bounds=(0.0, s_max), method="bounded")
            s_star = res.x
            min_alt_km = res.fun / 1e3
            diff_km = abs(min_alt_km - h_target_km)
            warnings.warn(
                f"Ray-trace: brentq bracketing failed; minimum altitude "
                f"{min_alt_km:.2f} km differs from target {h_target_km:.2f} km "
                f"by {diff_km:.3f} km — using minimum point.",
                UserWarning,
                stacklevel=3,
            )
        tp_ecef = pos_ecef + s_star * l_hat_ecef_tmp
        lat_rad, lon_rad, alt_m = _ecef_to_geodetic(tp_ecef[0], tp_ecef[1], tp_ecef[2])
        tangent_lat_deg = float(np.degrees(lat_rad))
        tangent_lon_deg = float(np.degrees(lon_rad))
        tangent_alt_km = alt_m / 1e3
        # Rotate ECEF tangent point to ECI
        cr = CartesianRepresentation(tp_ecef[0] * u.m, tp_ecef[1] * u.m, tp_ecef[2] * u.m)
        itrs_tp = ITRS(cr, obstime=epoch)
        gcrs_tp = itrs_tp.transform_to(GCRS(obstime=epoch))
        tangent_pos_eci = np.array([
            gcrs_tp.cartesian.x.to(u.m).value,
            gcrs_tp.cartesian.y.to(u.m).value,
            gcrs_tp.cartesian.z.to(u.m).value,
        ])

    # ── l̂ in ECEF ──────────────────────────────────────────────────────
    l_ecef = _eci_to_ecef_vec(l_hat_eci * 1e6, epoch) / 1e6
    l_hat_ecef = l_ecef / np.linalg.norm(l_ecef)

    # ── ENU direction cosines ─────────────────────────────────────────────
    lat_rad = np.radians(tangent_lat_deg)
    lon_rad = np.radians(tangent_lon_deg)
    E_eci, N_eci, Z_eci = _enu_eci(lat_rad, lon_rad, epoch)
    L_E = float(np.dot(l_hat_eci, E_eci))
    L_N = float(np.dot(l_hat_eci, N_eci))
    L_Z = float(np.dot(l_hat_eci, Z_eci))

    # ── Spacecraft velocity contribution ──────────────────────────────────
    vel_eci = np.array(meta.vel_eci_hat, dtype=float)
    V_sc_LOS = float(np.dot(vel_eci, l_hat_eci))

    # ── Earth rotation contribution ────────────────────────────────────────
    omega_vec = np.array([0.0, 0.0, EARTH_OMEGA_RAD_S])
    v_earth_eci = np.cross(omega_vec, tangent_pos_eci)
    v_earth_LOS = float(np.dot(v_earth_eci, l_hat_eci))

    return LOSGeometry(
        l_hat_eci=l_hat_eci,
        l_hat_ecef=l_hat_ecef,
        L_E=L_E,
        L_N=L_N,
        L_Z=L_Z,
        tangent_lat_deg=tangent_lat_deg,
        tangent_lon_deg=tangent_lon_deg,
        tangent_alt_km=tangent_alt_km,
        tangent_pos_eci=tangent_pos_eci,
        V_sc_LOS=V_sc_LOS,
        v_earth_LOS=v_earth_LOS,
    )


def correct_los_velocity(v_rel: float, geom: LOSGeometry) -> float:
    """
    Remove spacecraft and Earth-rotation contributions from v_rel.

    From NB02c (authoritative):
        v_rel = v_wind_LOS - V_sc_LOS - v_earth_LOS

    Rearranging:
        v_wind_LOS = v_rel + V_sc_LOS + v_earth_LOS

    Returns v_corrected = v_wind_LOS (approach-positive, wind toward TP).
    Note: ADDITION of V_sc_LOS and v_earth_LOS — not subtraction.
    """
    return v_rel + geom.V_sc_LOS + geom.v_earth_LOS


def process_frame(
    meta,
    v_rel: float,
    sigma_v: float,
) -> LOSObservation:
    """
    Convenience wrapper: compute geometry, derive obs_mode, correct velocity.

    Raises ValueError if img_type != 'science' or SLEW_IN_PROGRESS is set.
    """
    from src.metadata.p01_image_metadata_2026_04_06 import AdcsQualityFlags

    if meta.img_type != "science":
        raise ValueError(
            f"process_frame requires img_type='science', got '{meta.img_type}'"
        )
    if meta.adcs_quality_flag & AdcsQualityFlags.SLEW_IN_PROGRESS:
        raise ValueError(
            "process_frame: SLEW_IN_PROGRESS flag set — frame rejected"
        )

    geom = compute_los_geometry(meta)

    pos_eci = np.array(meta.pos_eci_hat, dtype=float)
    vel_eci = np.array(meta.vel_eci_hat, dtype=float)
    derived_mode = derive_obs_mode(
        meta.attitude_quaternion, pos_eci, vel_eci
    )

    stored_mode = getattr(meta, "obs_mode", "unknown") or "unknown"
    if derived_mode != stored_mode and stored_mode != "unknown":
        if getattr(meta, "is_synthetic", False):
            raise ValueError(
                f"process_frame: derived obs_mode '{derived_mode}' disagrees "
                f"with stored '{stored_mode}' for synthetic frame — geometry bug."
            )
        warnings.warn(
            f"process_frame: derived obs_mode '{derived_mode}' disagrees with "
            f"stored '{stored_mode}'. Using stored value (authoritative for real data).",
            UserWarning,
            stacklevel=2,
        )

    obs_mode = stored_mode if stored_mode != "unknown" else derived_mode
    v_corrected = correct_los_velocity(v_rel, geom)

    # Epoch in Unix ms from utc_timestamp
    epoch = _parse_epoch(meta)
    epoch_unix_ms = int(epoch.unix * 1000)

    return LOSObservation(
        v_corrected=v_corrected,
        sigma_v=sigma_v,
        L_E=geom.L_E,
        L_N=geom.L_N,
        L_Z=geom.L_Z,
        tangent_lat_deg=geom.tangent_lat_deg,
        tangent_lon_deg=geom.tangent_lon_deg,
        tangent_alt_km=geom.tangent_alt_km,
        epoch_unix_ms=epoch_unix_ms,
        obs_mode=obs_mode,
    )


def invert_wind_vector(observations: list) -> WindSolution:
    """
    Stage I: weighted least-squares wind vector inversion.

    Returns WindSolution with gdop_flag=True if condition number > GDOP_MAX,
    or n_frames_flag=True if len(observations) < N_MIN_FRAMES.
    NaN winds are returned for flagged bins.
    """
    n = len(observations)
    n_frames_flag = n < N_MIN_FRAMES

    mean_lat = float(np.mean([o.tangent_lat_deg for o in observations]))
    mean_lon = float(np.mean([o.tangent_lon_deg for o in observations]))
    mean_alt = float(np.mean([o.tangent_alt_km for o in observations]))
    mean_epoch = int(np.mean([o.epoch_unix_ms for o in observations]))
    obs_modes = {o.obs_mode for o in observations}

    if n_frames_flag:
        return WindSolution(
            v_E=float("nan"), v_N=float("nan"),
            sigma_v_E=float("nan"), sigma_v_N=float("nan"),
            two_sigma_v_E=float("nan"), two_sigma_v_N=float("nan"),
            n_frames=n, gdop_flag=False, n_frames_flag=True,
            condition_number=float("nan"),
            mean_tangent_lat_deg=mean_lat,
            mean_tangent_lon_deg=mean_lon,
            mean_tangent_alt_km=mean_alt,
            mean_epoch_unix_ms=mean_epoch,
            obs_modes=obs_modes,
        )

    A = np.zeros((n, 2))
    b = np.zeros(n)
    w = np.zeros(n)

    for i, obs in enumerate(observations):
        A[i, 0] = obs.L_E    # +L_E coefficient for v_E (no minus sign)
        A[i, 1] = obs.L_N    # +L_N coefficient for v_N (no minus sign)
        b[i] = obs.v_corrected
        w[i] = 1.0 / obs.sigma_v ** 2

    W = np.diag(w)
    ATA = A.T @ W @ A
    cond = float(np.linalg.cond(ATA))
    gdop_flag = cond > GDOP_MAX

    if gdop_flag:
        return WindSolution(
            v_E=float("nan"), v_N=float("nan"),
            sigma_v_E=float("nan"), sigma_v_N=float("nan"),
            two_sigma_v_E=float("nan"), two_sigma_v_N=float("nan"),
            n_frames=n, gdop_flag=True, n_frames_flag=n_frames_flag,
            condition_number=cond,
            mean_tangent_lat_deg=mean_lat,
            mean_tangent_lon_deg=mean_lon,
            mean_tangent_alt_km=mean_alt,
            mean_epoch_unix_ms=mean_epoch,
            obs_modes=obs_modes,
        )

    ATb = A.T @ W @ b
    x = np.linalg.solve(ATA, ATb)
    C = np.linalg.inv(ATA)

    v_E = float(x[0])
    v_N = float(x[1])
    sigma_v_E = float(np.sqrt(C[0, 0]))
    sigma_v_N = float(np.sqrt(C[1, 1]))
    two_sigma_v_E = 2.0 * sigma_v_E   # exactly 2 × sigma_v_E
    two_sigma_v_N = 2.0 * sigma_v_N   # exactly 2 × sigma_v_N

    return WindSolution(
        v_E=v_E,
        v_N=v_N,
        sigma_v_E=sigma_v_E,
        sigma_v_N=sigma_v_N,
        two_sigma_v_E=two_sigma_v_E,
        two_sigma_v_N=two_sigma_v_N,
        n_frames=n,
        gdop_flag=False,
        n_frames_flag=False,
        condition_number=cond,
        mean_tangent_lat_deg=mean_lat,
        mean_tangent_lon_deg=mean_lon,
        mean_tangent_alt_km=mean_alt,
        mean_epoch_unix_ms=mean_epoch,
        obs_modes=obs_modes,
    )


def bin_observations(
    obs_list: list,
    dlat: float = 5.0,
    dlon: float = 5.0,
    dt_min: float = 0.0,
) -> dict:
    """
    Stage B: spatiotemporal binning of LOSObservation list.

    Returns dict keyed by (lat_centre_deg, lon_centre_deg, t_centre_unix_ms)
    mapping to lists of LOSObservation for that bin.

    If dt_min == 0: bin by geographic location only (ignore time).
    All observations within the same dlat×dlon cell accumulate together
    regardless of when they were acquired.  This is appropriate for
    single-day datasets where azimuthal diversity comes from different
    orbital passes (along_track and cross_track) over the same location.

    If dt_min > 0: bin by (lat, lon, time) as before.
    """
    bins: dict = {}
    for obs in obs_list:
        lat_centre = round(obs.tangent_lat_deg / dlat) * dlat
        lon_centre = round(obs.tangent_lon_deg / dlon) * dlon
        if dt_min == 0:
            key = (lat_centre, lon_centre, 0)
        else:
            dt_ms = dt_min * 60.0 * 1000.0
            t_centre = round(obs.epoch_unix_ms / dt_ms) * dt_ms
            key = (lat_centre, lon_centre, int(t_centre))
        bins.setdefault(key, []).append(obs)
    return bins


def gmst_rotation_matrix(unix_ms: int) -> np.ndarray:
    """
    3×3 ECI→ECEF rotation matrix at the given epoch (Unix ms).

    Provided for completeness. H07 preferentially uses astropy GCRS↔ITRS
    transforms directly (same as NB02c).
    """
    epoch = Time(unix_ms / 1000.0, format="unix", scale="utc")
    # Use astropy to rotate unit vectors
    r_mat = np.zeros((3, 3))
    for i, ax in enumerate([
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]):
        ecef = _eci_to_ecef_vec(ax, epoch)
        r_mat[:, i] = ecef
    return r_mat
