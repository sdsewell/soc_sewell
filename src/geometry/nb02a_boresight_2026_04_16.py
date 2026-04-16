"""
NB02a — Boresight: quaternion synthesis and LOS unit vector in ECI.

Spec:      NB02_geometry_2026-04-16.md
Spec date: 2026-04-16
Generated: 2026-04-16
Tool:      Claude Code

Boresight direction: -X_BRF (SI-UCAR-WC-RP-004 §2.4.2.1).
look_mode strings match THRF config names: 'along_track', 'cross_track'.
Quaternion convention: scalar-last [q1, q2, q3, q4].
"""

from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
from scipy.spatial.transform import Rotation

from src.constants import WGS84_A_M, WGS84_B_M

# ---------------------------------------------------------------------------
# BORESIGHT DIRECTION NOTE (SI-UCAR-WC-RP-004 §2.4.2.1)
# The FPI payload boresight is the -X_BRF axis. The body-frame boresight
# vector is always np.array([-1., 0., 0.]). Rotating this by the attitude
# quaternion q (BRF→ECI) gives los_eci pointing toward the tangent point.
# For along_track mode: los_eci ≈ forward velocity direction, depressed 15.73°.
# For cross_track mode: los_eci ≈ orbit-normal direction, depressed 15.73°.
# These correspond to the THRF along-track and cross-track configurations
# defined in SI-UCAR-WC-RP-004 §2.4.2.2. See NB01 spec §2.4.2.
# ---------------------------------------------------------------------------

DEPRESSION_ANGLE_DEG: float = 15.73   # arccos(6621/6881), 510 km alt, 250 km tangent
DEFAULT_ALTITUDE_KM:  float = 510.0   # confirmed mission altitude


def _compute_depression_angle(
    altitude_km: float,
    h_target_km: float,
) -> float:
    """
    Depression angle below local horizontal for limb observation.

    Uses right-triangle geometry (right angle at tangent point).
    depression = arccos((R_E + h_target) / (R_E + altitude))
    R_E = 6371.0 km (mean radius — adequate for angle geometry).
    """
    R_E_km = 6371.0
    R_sc = R_E_km + altitude_km
    R_tp = R_E_km + h_target_km
    return float(np.degrees(np.arccos(R_tp / R_sc)))


def _brf_axes_eci(
    pos_eci: np.ndarray,
    vel_eci: np.ndarray,
    look_mode: str,
    dep_rad: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute BRF axes (+X, +Y, +Z) expressed in ECI for -X_BRF boresight convention.

    +X_BRF = AFT direction (opposite of boresight; -X_BRF points toward tangent point).
    +Y_BRF ≈ star tracker boresight direction (orbit-normal for along_track).
    +Z_BRF = +X_BRF × +Y_BRF (right-hand complete).

    Returns (x_brf, y_brf, z_brf) each shape (3,), unit vectors in ECI.
    """
    nadir = -pos_eci / np.linalg.norm(pos_eci)       # toward Earth centre
    v_hat = vel_eci / np.linalg.norm(vel_eci)

    # Orbit normal: always ⊥ both pos and vel (never needs caching)
    orbit_normal = np.cross(pos_eci, vel_eci)
    orbit_normal /= np.linalg.norm(orbit_normal)

    # Forward horizontal component of velocity (orthogonalised against nadir)
    v_horiz = v_hat - np.dot(v_hat, nadir) * nadir
    n_vh = np.linalg.norm(v_horiz)
    v_horiz = v_horiz / n_vh if n_vh > 1e-10 else v_hat

    cos_d, sin_d = np.cos(dep_rad), np.sin(dep_rad)

    if look_mode == "along_track":
        # -X_BRF → forward horizontal, depressed sin_d toward nadir
        boresight_eci = cos_d * v_horiz + sin_d * nadir
        # +Y_BRF ≈ orbit normal (star tracker direction, cross-track)
        y_candidate = orbit_normal
    else:  # cross_track
        # orbit_normal is already ⊥ nadir (proven by cross-product construction)
        # -X_BRF → orbit-normal direction, depressed sin_d toward nadir
        boresight_eci = cos_d * orbit_normal + sin_d * nadir
        # +Y_BRF ≈ forward velocity direction
        y_candidate = v_horiz

    boresight_eci /= np.linalg.norm(boresight_eci)

    # +X_BRF = anti-boresight (AFT direction)
    x_brf = -boresight_eci

    # Orthogonalise +Y_BRF against +X_BRF (Gram–Schmidt)
    y_brf = y_candidate - np.dot(y_candidate, x_brf) * x_brf
    y_brf /= np.linalg.norm(y_brf)

    # +Z_BRF = +X_BRF × +Y_BRF (right-hand complete)
    z_brf = np.cross(x_brf, y_brf)
    z_brf /= np.linalg.norm(z_brf)

    return x_brf, y_brf, z_brf


def compute_synthetic_quaternion(
    pos_eci: np.ndarray,
    vel_eci: np.ndarray,
    look_mode: str,
    altitude_km: float = DEFAULT_ALTITUDE_KM,
    h_target_km: float = 250.0,
) -> np.ndarray:
    """
    Synthesise a unit attitude quaternion for a given look mode.

    The FPI boresight is the -X_BRF axis (SI-UCAR-WC-RP-004 §2.4.2.1).
    This function constructs the quaternion q rotating BRF→ECI such that
    R(q) @ [-1, 0, 0] points toward the tangent point.

    For along_track (THRF along-track configuration):
        -X_BRF rotated to ECI points forward in velocity direction,
        depressed 15.73° below local horizontal.
        Tangent point is ~923 km ahead of spacecraft.

    For cross_track (THRF cross-track configuration):
        -X_BRF rotated to ECI points perpendicular to orbit plane,
        depressed 15.73° below local horizontal.
        Tangent point is ~923 km to the side of the ground track.

    The look_mode strings 'along_track' and 'cross_track' match the
    THRF configuration names in SI-UCAR-WC-RP-004 §2.4.2.2 and the
    thrf_config convention established in NB01 spec §2.4.2.

    Parameters
    ----------
    pos_eci : np.ndarray, shape (3,)
        Spacecraft ECI position, m.
    vel_eci : np.ndarray, shape (3,), m/s
        Spacecraft ECI velocity.
    look_mode : str
        'along_track' or 'cross_track'.
    altitude_km : float
        Nominal spacecraft altitude, km. Default 510.0.
    h_target_km : float
        Target tangent altitude, km. Default 250.0.

    Returns
    -------
    q : np.ndarray, shape (4,)
        Scalar-last unit quaternion [q1, q2, q3, q4] rotating BRF to ECI.
    """
    if look_mode not in ("along_track", "cross_track"):
        raise ValueError(
            f"look_mode must be 'along_track' or 'cross_track', got '{look_mode}'"
        )

    # Iterate to find the depression angle that targets h_target_km geodetic altitude.
    # The tangent point can be at a significantly different geocentric latitude than
    # the spacecraft (up to ~16° for the 510 km / 250 km geometry), so using the
    # SC latitude to set R_tp causes a systematic tp_alt error of up to ~6 km.
    # Each iteration: compute boresight from current dep_rad, find the geometric
    # tangent point (closest-approach point along LOS), update R_tp at that latitude.
    # Converges in 3–4 iterations to < 0.1 km geodetic altitude error.
    sc_r_m = np.linalg.norm(pos_eci)
    a_m = float(WGS84_A_M)
    b_m = float(WGS84_B_M)
    h_eff_m = h_target_km * 1e3

    def _wgs84_r(lat_gc: float) -> float:
        return a_m * b_m / np.sqrt(
            (b_m * np.cos(lat_gc)) ** 2 + (a_m * np.sin(lat_gc)) ** 2
        )

    # Seed with SC geocentric latitude; iterate toward TP latitude
    lat_gc_seed = float(np.arcsin(pos_eci[2] / sc_r_m))
    R_tp = _wgs84_r(lat_gc_seed) + h_eff_m

    x_brf = y_brf = z_brf = None
    for _ in range(5):
        dep_rad = float(np.arccos(np.clip(R_tp / sc_r_m, -1.0, 1.0)))
        x_brf, y_brf, z_brf = _brf_axes_eci(pos_eci, vel_eci, look_mode, dep_rad)
        los_approx = -x_brf  # boresight = -X_BRF (unit vector)
        t_min = -np.dot(pos_eci, los_approx)  # closest-approach parameter
        if t_min <= 0.0:
            break  # boresight doesn't point toward Earth
        tp_approx = pos_eci + t_min * los_approx
        lat_gc_tp = float(np.arcsin(tp_approx[2] / np.linalg.norm(tp_approx)))
        R_tp_new = _wgs84_r(lat_gc_tp) + h_eff_m
        if abs(R_tp_new - R_tp) < 1.0:  # converged to 1 m
            R_tp = R_tp_new
            break
        R_tp = R_tp_new

    # Rotation matrix: columns are ECI coords of BRF +X, +Y, +Z axes
    # R_mat @ e_brf = e_eci  (BRF → ECI active rotation)
    R_mat = np.column_stack([x_brf, y_brf, z_brf])

    rot = Rotation.from_matrix(R_mat)
    return rot.as_quat()   # scalar-last: [q1, q2, q3, q4]


def compute_los_eci(
    pos_eci: np.ndarray,
    vel_eci: np.ndarray,
    look_mode: str,
    altitude_km: float = DEFAULT_ALTITUDE_KM,
    h_target_km: float = 250.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the LOS unit vector in ECI and the corresponding quaternion.

    Internally builds boresight_body = np.array([-1., 0., 0.]) (-X_BRF),
    then rotates to ECI using the attitude quaternion for the given look_mode.

    Parameters
    ----------
    pos_eci : np.ndarray, shape (3,), m
    vel_eci : np.ndarray, shape (3,), m/s
    look_mode : str — 'along_track' or 'cross_track'
    altitude_km : float — spacecraft altitude, km
    h_target_km : float — tangent point target altitude, km

    Returns
    -------
    (los_eci, q) :
        los_eci : np.ndarray, shape (3,) — unit LOS vector in ECI J2000
        q       : np.ndarray, shape (4,) — attitude quaternion used (BRF→ECI)
    """
    if look_mode not in ("along_track", "cross_track"):
        raise ValueError(
            f"look_mode must be 'along_track' or 'cross_track', got '{look_mode}'"
        )

    q = compute_synthetic_quaternion(pos_eci, vel_eci, look_mode, altitude_km, h_target_km)

    # Boresight is -X_BRF (SI-UCAR-WC-RP-004 §2.4.2.1)
    boresight_body = np.array([-1.0, 0.0, 0.0])

    rot = Rotation.from_quat(q)           # BRF → ECI
    los_eci = rot.apply(boresight_body)
    los_eci /= np.linalg.norm(los_eci)   # ensure unit norm (numerical hygiene)

    return los_eci, q
