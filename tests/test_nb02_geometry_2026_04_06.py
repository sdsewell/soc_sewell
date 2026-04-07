"""
Tests for NB02 geometry pipeline.
Spec: specs/S07_nb02_geometry_2026-04-05.md
"""

import numpy as np
import pytest
from astropy.time import Time

from src.geometry.nb02a_boresight_2026_04_06 import (
    compute_synthetic_quaternion,
    compute_los_eci,
)
from src.geometry.nb02b_tangent_point_2026_04_06 import compute_tangent_point
from src.geometry.nb02c_los_projection_2026_04_06 import (
    enu_unit_vectors_eci,
    earth_rotation_velocity_eci,
    compute_v_rel,
)
from src.geometry.nb02d_l1c_calibrator_2026_04_06 import (
    compose_v_rel,
    remove_spacecraft_velocity,
)
from src.windmap.nb00_wind_map_2026_04_06 import UniformWindMap

# Standard test geometry: equatorial crossing at Y-axis, sun-synchronous velocity
_POS = np.array([0.0, 6.896e6, 0.0])       # shape (3,), m
_VEL = np.array([977.2, 0.0, 7534.5])       # shape (3,), m/s
_EPOCH = Time("2027-01-01T00:00:00", scale="utc")


# ---------------------------------------------------------------------------
# T1 — Quaternion unit norm
# ---------------------------------------------------------------------------

def test_quaternion_unit_norm():
    """Synthetic quaternion must be unit norm to 1e-10."""
    for mode in ("along_track", "cross_track"):
        q = compute_synthetic_quaternion(_POS, _VEL, mode)
        assert abs(np.linalg.norm(q) - 1.0) < 1e-10, \
            f"[{mode}] |q| = {np.linalg.norm(q):.12f} ≠ 1"


# ---------------------------------------------------------------------------
# T2 — LOS unit norm
# ---------------------------------------------------------------------------

def test_los_unit_norm():
    """LOS vector must be unit norm to 1e-10."""
    for mode in ("along_track", "cross_track"):
        los, _ = compute_los_eci(_POS, _VEL, mode)
        assert abs(np.linalg.norm(los) - 1.0) < 1e-10, \
            f"[{mode}] |los| = {np.linalg.norm(los):.12f} ≠ 1"


# ---------------------------------------------------------------------------
# T3 — Along-track vs cross-track approximate orthogonality
# ---------------------------------------------------------------------------

def test_look_mode_orthogonality():
    """Along-track and cross-track LOS vectors must have |dot| < 0.1."""
    los_at, _ = compute_los_eci(_POS, _VEL, "along_track")
    los_ct, _ = compute_los_eci(_POS, _VEL, "cross_track")
    dot = abs(np.dot(los_at, los_ct))
    assert dot < 0.1, f"|los_at · los_ct| = {dot:.4f} ≥ 0.1"


# ---------------------------------------------------------------------------
# T4 — LOS depression angle ≈ 15.73°
# ---------------------------------------------------------------------------

def test_los_depression_angle():
    """
    LOS must be depressed ~15.73° below local horizontal toward Earth.
    Expected: 10° < depression < 22° (allows for orbit geometry variation).
    """
    nadir = -_POS / np.linalg.norm(_POS)
    for mode in ("along_track", "cross_track"):
        los, _ = compute_los_eci(_POS, _VEL, mode)
        dep_deg = np.degrees(np.arcsin(np.dot(los, nadir)))
        assert 10.0 < dep_deg < 22.0, \
            f"[{mode}] depression = {dep_deg:.2f}° outside [10°, 22°]; expect ~15.73°"


# ---------------------------------------------------------------------------
# T5 — Tangent point altitude within 5 km of target
# ---------------------------------------------------------------------------

def test_tangent_point_altitude():
    """Tangent point altitude must be within 5 km of 250 km target."""
    los, _ = compute_los_eci(_POS, _VEL, "along_track")
    tp = compute_tangent_point(_POS, los, _EPOCH, h_target_km=250.0)
    assert abs(tp["tp_alt_km"] - 250.0) < 5.0, \
        f"Tangent alt {tp['tp_alt_km']:.2f} km, expected 250 ± 5 km"


# ---------------------------------------------------------------------------
# T6 — Tangent point leads spacecraft (RAM-FACE geometry)
# ---------------------------------------------------------------------------

def test_tangent_point_leads_spacecraft():
    """
    Along-track tangent point must be AHEAD of spacecraft, not behind.
    The along-track offset must be > 500 km forward.
    """
    los, _ = compute_los_eci(_POS, _VEL, "along_track")
    tp = compute_tangent_point(_POS, los, _EPOCH)
    tp_pos = tp["tp_eci"]
    v_hat = _VEL / np.linalg.norm(_VEL)
    forward_offset_m = np.dot(tp_pos - _POS, v_hat)
    assert forward_offset_m > 500e3, \
        (f"Tangent point {forward_offset_m / 1e3:.0f} km from sc; "
         f"expected > 500 km FORWARD (RAM-face aperture)")


# ---------------------------------------------------------------------------
# T7 — v_rel round-trip < 1e-10 m/s
# ---------------------------------------------------------------------------

def test_v_rel_round_trip():
    """
    compose_v_rel / remove_spacecraft_velocity must be exact inverses.
    Error must be < 1e-10 m/s (machine precision).
    If this fails, there is a sign error — not a numerical issue.
    """
    cases = [
        (100.0, -7100.0, 460.0),
        (-50.0,  -200.0, 230.0),
        (  0.0,  7097.3, 461.1),
    ]
    for v_wind, V_sc, v_earth in cases:
        v_rel = compose_v_rel(v_wind, V_sc, v_earth)
        v_wind_recovered = remove_spacecraft_velocity(v_rel, V_sc, v_earth)
        error = abs(v_wind_recovered - v_wind)
        assert error < 1e-10, \
            (f"Round-trip error {error:.2e} m/s for "
             f"(v_wind={v_wind}, V_sc={V_sc}, v_earth={v_earth}) — check signs")


# ---------------------------------------------------------------------------
# T8 — Along-track V_sc_LOS >> cross-track V_sc_LOS
# ---------------------------------------------------------------------------

def test_vsc_los_ratio():
    """
    Along-track V_sc_LOS must be > 10× cross-track V_sc_LOS.
    Verifies that the look modes are correctly implemented.
    """
    wind_map = UniformWindMap(v_zonal_ms=0.0, v_merid_ms=0.0)

    results = {}
    for mode in ("along_track", "cross_track"):
        los, _ = compute_los_eci(_POS, _VEL, mode)
        tp = compute_tangent_point(_POS, los, _EPOCH)
        res = compute_v_rel(
            wind_map,
            tp["tp_lat_deg"], tp["tp_lon_deg"], tp["tp_eci"],
            _VEL, los, _EPOCH,
        )
        results[mode] = abs(res["V_sc_LOS"])

    ratio = results["along_track"] / (results["cross_track"] + 1e-6)
    assert ratio > 10.0, \
        f"V_sc_LOS ratio along/cross = {ratio:.1f}; expected > 10"


# ---------------------------------------------------------------------------
# T9 — ENU unit vectors are orthonormal
# ---------------------------------------------------------------------------

def test_enu_orthonormal():
    """East, North, Up unit vectors must be mutually orthogonal and unit norm."""
    for lat, lon in [(0.0, 0.0), (30.0, 45.0), (-60.0, -90.0)]:
        e, n, u = enu_unit_vectors_eci(lat, lon, _EPOCH)
        assert abs(np.linalg.norm(e) - 1.0) < 1e-10, f"[{lat},{lon}] |E| ≠ 1"
        assert abs(np.linalg.norm(n) - 1.0) < 1e-10, f"[{lat},{lon}] |N| ≠ 1"
        assert abs(np.linalg.norm(u) - 1.0) < 1e-10, f"[{lat},{lon}] |U| ≠ 1"
        assert abs(np.dot(e, n)) < 1e-10, f"[{lat},{lon}] E·N ≠ 0"
        assert abs(np.dot(e, u)) < 1e-10, f"[{lat},{lon}] E·U ≠ 0"
        assert abs(np.dot(n, u)) < 1e-10, f"[{lat},{lon}] N·U ≠ 0"


# ---------------------------------------------------------------------------
# T10 — Earth rotation velocity plausible
# ---------------------------------------------------------------------------

def test_earth_rotation_velocity():
    """
    Earth rotation velocity at the equator must be ~465 m/s eastward.
    At the pole it must be near zero.
    """
    # Equatorial tangent point — ECEF x-axis (lon=0, lat=0, alt=250 km)
    tp_equator_eci = np.array([6.621e6, 0.0, 0.0])
    v_eq = earth_rotation_velocity_eci(tp_equator_eci)
    assert 400 < np.linalg.norm(v_eq) < 530, \
        f"Earth rotation at equator {np.linalg.norm(v_eq):.0f} m/s; expected ~465"

    # Polar tangent point
    tp_pole_eci = np.array([0.0, 0.0, 6.621e6])
    v_pole = earth_rotation_velocity_eci(tp_pole_eci)
    assert np.linalg.norm(v_pole) < 10, \
        f"Earth rotation at pole {np.linalg.norm(v_pole):.1f} m/s; expected ~0"
