"""
Tests for H07 wind vector retrieval module.
Spec: specs/H07_wind_vector_retrieval_2026-05-14_v03.md (v0.3)
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.metadata import ImageMetadata
from windcube.constants import GDOP_MAX, N_MIN_FRAMES
from windcube.wind_retrieval import (
    LOSObservation,
    correct_los_velocity,
    compute_los_geometry,
    derive_obs_mode,
    invert_wind_vector,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

# Nominal spacecraft state: equatorial orbit at ~510 km altitude
_POS_ECI = np.array([0.0, 6_888_137.0, 0.0])   # m; equatorial, 510 km alt
_VEL_ECI = np.array([7_560.0, 0.0, 0.0])        # m/s; prograde
_EPOCH_STR = "2027-01-01T00:01:00.000000+00:00"


def _make_science_meta(
    attitude_quaternion,
    pos_eci=None,
    vel_eci=None,
    obs_mode="along_track",
    tangent_lat=0.0,
    tangent_lon=12.0,
    h_target_km_obs=250.0,
    is_synthetic=True,
):
    """Minimal science ImageMetadata for H07 geometry tests."""
    pos = list(pos_eci) if pos_eci is not None else list(_POS_ECI)
    vel = list(vel_eci) if vel_eci is not None else list(_VEL_ECI)
    return ImageMetadata(
        rows=260, cols=276, exp_time=600, exp_unit=1,
        binning=2, img_type="science",
        lua_timestamp=1798819260000,
        adcs_timestamp=1798819260012,
        utc_timestamp=_EPOCH_STR,
        spacecraft_latitude=0.0, spacecraft_longitude=0.0,
        spacecraft_altitude=510_000.0,
        pos_eci_hat=pos,
        vel_eci_hat=vel,
        attitude_quaternion=list(attitude_quaternion),
        pointing_error=[0.0, 0.0, 0.0, 1.0],
        obs_mode=obs_mode,
        ccd_temp1=-18.0,
        etalon_temps=[-5.0, -5.0, -5.0, -5.0],
        shutter_status="open",
        gpio_pwr_on=[0, 0, 0, 0],
        lamp_ch_array=[0, 0, 0, 0, 0, 0],
        lamp1_status="off", lamp2_status="off", lamp3_status="off",
        orbit_number=1, frame_sequence=0, orbit_parity="along_track",
        adcs_quality_flag=0,
        is_synthetic=is_synthetic,
        tangent_lat=tangent_lat,
        tangent_lon=tangent_lon,
        h_target_km_obs=h_target_km_obs,
    )


def _make_obs(L_E, L_N, v_corrected=0.0, sigma_v=10.0):
    """LOSObservation with the given direction cosines and corrected velocity."""
    L_Z = float(np.sqrt(max(0.0, 1.0 - L_E**2 - L_N**2)))
    return LOSObservation(
        v_corrected=v_corrected,
        sigma_v=sigma_v,
        L_E=L_E, L_N=L_N, L_Z=L_Z,
        tangent_lat_deg=0.0, tangent_lon_deg=0.0,
        tangent_alt_km=250.0,
        epoch_unix_ms=1798819260000,
        obs_mode="along_track",
    )


# ---------------------------------------------------------------------------
# T1 — derive_obs_mode: along_track
# ---------------------------------------------------------------------------


def test_t1_derive_obs_mode_along_track():
    """
    An attitude quaternion synthesised for along_track must be identified as
    'along_track' by derive_obs_mode.
    """
    from src.geometry.nb02a_boresight_2026_04_16 import compute_synthetic_quaternion
    q = compute_synthetic_quaternion(
        _POS_ECI, _VEL_ECI, "along_track", altitude_km=510.0, h_target_km=250.0
    )
    mode = derive_obs_mode(
        attitude_quaternion=list(q),
        pos_eci=_POS_ECI,
        vel_eci=_VEL_ECI,
    )
    assert mode == "along_track", f"Expected 'along_track', got '{mode}'"


# ---------------------------------------------------------------------------
# T2 — derive_obs_mode: cross_track
# ---------------------------------------------------------------------------


def test_t2_derive_obs_mode_cross_track():
    """
    An attitude quaternion synthesised for cross_track must be identified as
    'cross_track' by derive_obs_mode.
    """
    from src.geometry.nb02a_boresight_2026_04_16 import compute_synthetic_quaternion
    q = compute_synthetic_quaternion(
        _POS_ECI, _VEL_ECI, "cross_track", altitude_km=510.0, h_target_km=250.0
    )
    mode = derive_obs_mode(
        attitude_quaternion=list(q),
        pos_eci=_POS_ECI,
        vel_eci=_VEL_ECI,
    )
    assert mode == "cross_track", f"Expected 'cross_track', got '{mode}'"


# ---------------------------------------------------------------------------
# T3 — compute_los_geometry: boresight matches NB02a LOS vector to < 0.001 rad
# ---------------------------------------------------------------------------


def test_t3_boresight_matches_nb02a():
    """
    l_hat_eci from compute_los_geometry must agree with compute_los_eci (NB02a)
    to better than 0.001 rad for a known synthetic along_track frame.
    """
    from src.geometry.nb02a_boresight_2026_04_16 import compute_los_eci
    los_eci_expected, q = compute_los_eci(
        _POS_ECI, _VEL_ECI, "along_track", altitude_km=510.0, h_target_km=250.0
    )
    los_eci_expected = los_eci_expected / np.linalg.norm(los_eci_expected)

    meta = _make_science_meta(
        attitude_quaternion=q,
        obs_mode="along_track",
        is_synthetic=True,
        tangent_lat=0.0,
        tangent_lon=12.0,
    )
    geom = compute_los_geometry(meta)

    dot = float(np.dot(geom.l_hat_eci, los_eci_expected))
    dot = float(np.clip(dot, -1.0, 1.0))
    angle_rad = float(np.arccos(dot))
    assert angle_rad < 0.001, (
        f"l_hat_eci angle vs NB02a = {np.degrees(angle_rad):.4f}° "
        f"(should be < 0.057°)"
    )


# ---------------------------------------------------------------------------
# T4 — correct_los_velocity: null-wind gives exactly 0
# ---------------------------------------------------------------------------


def test_t4_null_wind_correction_zero():
    """
    For a null-wind frame: setting v_rel = -V_sc_LOS - v_earth_LOS then
    calling correct_los_velocity must return 0.0 exactly (algebraic identity).

    Implements the self-consistency check from H07 §11.3.
    """
    from src.geometry.nb02a_boresight_2026_04_16 import compute_synthetic_quaternion
    q = compute_synthetic_quaternion(
        _POS_ECI, _VEL_ECI, "along_track", altitude_km=510.0, h_target_km=250.0
    )
    meta = _make_science_meta(
        attitude_quaternion=q,
        obs_mode="along_track",
        is_synthetic=True,
        tangent_lat=0.0,
        tangent_lon=12.0,
    )
    geom = compute_los_geometry(meta)

    # Null wind: v_rel = v_wind_LOS - V_sc_LOS - v_earth_LOS = 0 - V_sc - v_earth
    v_rel_null = -geom.V_sc_LOS - geom.v_earth_LOS
    v_corrected = correct_los_velocity(v_rel_null, geom)

    assert abs(v_corrected) < 1e-10, (
        f"v_corrected for null-wind should be 0.0, got {v_corrected:.4e} m/s"
    )


# ---------------------------------------------------------------------------
# T5 — invert_wind_vector: null-wind bin recovers v_E≈0, v_N≈0
# ---------------------------------------------------------------------------


def test_t5_invert_null_wind():
    """
    A bin of observations with v_corrected=0 and diverse L_E, L_N spanning
    the compass must recover v_E ≈ 0 and v_N ≈ 0.
    """
    angles = np.linspace(0, 2 * np.pi, 9)[:-1]  # 8 equally-spaced directions
    scale = 0.96   # so that L_E^2 + L_N^2 < 1 (leaves room for L_Z)
    obs_list = [
        _make_obs(
            L_E=scale * np.cos(a),
            L_N=scale * np.sin(a),
            v_corrected=0.0,
            sigma_v=10.0,
        )
        for a in angles
    ]
    sol = invert_wind_vector(obs_list)
    assert not sol.gdop_flag, "Unexpected GDOP flag for diverse-direction bin"
    assert not sol.n_frames_flag, "Unexpected n_frames flag"
    assert abs(sol.v_E) < 0.5, f"v_E = {sol.v_E:.2f} m/s (expected ≈ 0)"
    assert abs(sol.v_N) < 0.5, f"v_N = {sol.v_N:.2f} m/s (expected ≈ 0)"


# ---------------------------------------------------------------------------
# T6 — two_sigma_v_E == 2.0 * sigma_v_E exactly
# ---------------------------------------------------------------------------


def test_t6_two_sigma_exact():
    """
    WindSolution.two_sigma_v_E must equal 2.0 * sigma_v_E as the same
    floating-point operation, not a separate calculation.
    """
    angles = np.linspace(0, 2 * np.pi, 9)[:-1]
    scale = 0.96
    obs_list = [
        _make_obs(scale * np.cos(a), scale * np.sin(a), 0.0, 10.0)
        for a in angles
    ]
    sol = invert_wind_vector(obs_list)
    assert not sol.gdop_flag
    # Floating-point identity: must be exactly 2.0 * sigma_v_E
    assert sol.two_sigma_v_E == 2.0 * sol.sigma_v_E, (
        f"two_sigma_v_E ({sol.two_sigma_v_E}) != 2.0 * sigma_v_E ({2.0 * sol.sigma_v_E})"
    )
    assert sol.two_sigma_v_N == 2.0 * sol.sigma_v_N, (
        f"two_sigma_v_N ({sol.two_sigma_v_N}) != 2.0 * sigma_v_N ({2.0 * sol.sigma_v_N})"
    )


# ---------------------------------------------------------------------------
# T7 — invert_wind_vector: gdop_flag=True for near-parallel LOS vectors
# ---------------------------------------------------------------------------


def test_t7_gdop_flag_near_parallel():
    """
    When all LOS vectors point nearly the same direction, ATA is
    near-singular (high condition number) → gdop_flag=True.
    """
    L_E_base = 0.92
    L_N_base = 0.01
    obs_list = [
        _make_obs(
            L_E=L_E_base + 1e-7 * i,
            L_N=L_N_base + 1e-7 * i,
            v_corrected=0.0,
            sigma_v=10.0,
        )
        for i in range(8)
    ]
    sol = invert_wind_vector(obs_list)
    assert sol.gdop_flag, (
        f"Expected gdop_flag=True for near-parallel LOS vectors; "
        f"condition_number={sol.condition_number:.1f} (GDOP_MAX={GDOP_MAX})"
    )


# ---------------------------------------------------------------------------
# T8 — invert_wind_vector: n_frames_flag=True for fewer than N_MIN_FRAMES
# ---------------------------------------------------------------------------


def test_t8_n_frames_flag():
    """
    A bin with fewer than N_MIN_FRAMES=4 observations must set n_frames_flag=True
    and return NaN for v_E, v_N.
    """
    obs_list = [
        _make_obs(np.cos(a), np.sin(a) * 0.96, 0.0, 10.0)
        for a in [0.0, np.pi / 2, np.pi]   # 3 frames < N_MIN_FRAMES=4
    ]
    assert len(obs_list) < N_MIN_FRAMES
    sol = invert_wind_vector(obs_list)
    assert sol.n_frames_flag, f"Expected n_frames_flag=True for {len(obs_list)} frames"
    assert sol.n_frames == len(obs_list)
    assert np.isnan(sol.v_E), "v_E should be NaN when n_frames_flag=True"
    assert np.isnan(sol.v_N), "v_N should be NaN when n_frames_flag=True"
