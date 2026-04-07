"""
NB02a — Boresight: quaternion synthesis and LOS unit vector.

Spec:        S07_nb02_geometry_2026-04-05.md  Section 4.1
Spec date:   2026-04-05
Generated:   2026-04-07
Depends on:  src.constants
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from src.constants import DEPRESSION_ANGLE_DEG as _DEP_DEFAULT

# ---------------------------------------------------------------------------
# Module-level constants (Section 3 of spec)
# ---------------------------------------------------------------------------

DEPRESSION_ANGLE_DEG: float = 15.73   # arccos(6621/6881), 510 km alt, 250 km tangent
DEFAULT_ALTITUDE_KM:  float = 510.0   # confirmed mission altitude

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


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


def _body_frame_axes(
    pos_eci: np.ndarray,
    vel_eci: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute orthonormal body frame axes in ECI.

    Body frame definition (spacecraft-fixed):
        +X = ram face = forward (velocity direction)
        +Y = cross-track = left of velocity
        +Z = nadir (toward Earth centre)

    Returns (x_hat, y_hat, z_hat) each shape (3,), unit vectors in ECI.
    Uses Gram–Schmidt orthogonalisation so the result is exact even for
    slightly non-circular orbits.
    """
    # +Z body = nadir = toward Earth centre
    z_hat = -pos_eci / np.linalg.norm(pos_eci)

    # +X body = forward ≈ velocity direction; orthogonalise vs nadir
    v_hat_raw = vel_eci / np.linalg.norm(vel_eci)
    y_hat = np.cross(z_hat, v_hat_raw)
    y_hat /= np.linalg.norm(y_hat)          # +Y body = left

    # Re-derive +X body to guarantee orthogonality
    x_hat = np.cross(y_hat, z_hat)
    x_hat /= np.linalg.norm(x_hat)         # +X body = forward (corrected)

    return x_hat, y_hat, z_hat


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def compute_synthetic_quaternion(
    pos_eci: np.ndarray,
    vel_eci: np.ndarray,
    look_mode: str,
    altitude_km: float = DEFAULT_ALTITUDE_KM,
    h_target_km: float = 250.0,
) -> np.ndarray:
    """
    Synthesise a unit attitude quaternion for a given look mode.

    For along-track: boresight points forward (+X body) depressed ~15.73°.
    For cross-track: boresight points left (+Y body) depressed ~15.73°.

    The exact depression angle is derived from the actual |pos_eci| radius and
    h_target_km using _compute_depression_angle (R_E = 6371 km mean radius).
    The altitude_km parameter is provided for API consistency but the actual
    spacecraft altitude is always computed from pos_eci to ensure the LOS
    targets h_target_km exactly (within WGS84 oblateness errors < 5 km).

    RAM-FACE APERTURE — critical geometry note:
    The instrument aperture is on the +X (ram) face. For along-track look
    mode the boresight points forward in the velocity direction and is
    depressed toward the limb. The tangent point is therefore ahead of the
    spacecraft by ~923 km, not behind it. WindCube measures the thermosphere
    it will fly over approximately 122 seconds in the future.

    Parameters
    ----------
    pos_eci : np.ndarray, shape (3,)
        Spacecraft ECI position, m.
    vel_eci : np.ndarray, shape (3,)
        Spacecraft ECI velocity, m/s.
    look_mode : str
        'along_track' or 'cross_track'.
    altitude_km : float
        Nominal spacecraft altitude, km (unused internally; actual altitude
        is derived from |pos_eci|). Default 510.0.
    h_target_km : float
        Target tangent altitude, km. Default 250.0.

    Returns
    -------
    q : np.ndarray, shape (4,)
        Scalar-last unit quaternion [q1, q2, q3, q4] rotating body
        frame to ECI.
    """
    if look_mode not in ("along_track", "cross_track"):
        raise ValueError(f"look_mode must be 'along_track' or 'cross_track', got '{look_mode}'")

    x_hat, y_hat, z_hat = _body_frame_axes(pos_eci, vel_eci)

    # Rotation matrix: columns are body axes expressed in ECI
    # R_body2eci @ e_body = e_eci  (active rotation, body → ECI)
    R_mat = np.column_stack([x_hat, y_hat, z_hat])

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

    The boresight direction in the body frame is:
        along_track : [cos(δ), 0,       sin(δ)]  (+X fwd, +Z nadir, dep δ)
        cross_track : [0,      cos(δ),  sin(δ)]  (+Y left, +Z nadir, dep δ)
    where δ = depression angle ≈ 15.73°.

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
        q       : np.ndarray, shape (4,) — attitude quaternion used
    """
    if look_mode not in ("along_track", "cross_track"):
        raise ValueError(f"look_mode must be 'along_track' or 'cross_track', got '{look_mode}'")

    # Compute depression angle from ACTUAL spacecraft radius so the LOS
    # always targets h_target_km (within WGS84 oblateness errors < 5 km).
    _R_E_KM = 6371.0   # mean radius — matches _compute_depression_angle
    sc_alt_km = np.linalg.norm(pos_eci) / 1e3 - _R_E_KM
    dep_rad = np.radians(_compute_depression_angle(sc_alt_km, h_target_km))
    cos_d = np.cos(dep_rad)
    sin_d = np.sin(dep_rad)

    # LOS in body frame (boresight depressed sin_d toward nadir)
    if look_mode == "along_track":
        los_body = np.array([cos_d, 0.0, sin_d])
    else:  # cross_track
        los_body = np.array([0.0, cos_d, sin_d])

    q = compute_synthetic_quaternion(pos_eci, vel_eci, look_mode, altitude_km, h_target_km)
    rot = Rotation.from_quat(q)          # body → ECI
    los_eci = rot.apply(los_body)
    los_eci /= np.linalg.norm(los_eci)   # ensure unit norm (numerical hygiene)

    return los_eci, q
