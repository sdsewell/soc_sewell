import numpy as np
import pytest
from src.geometry.nb01_orbit_propagator_2026_04_05 import propagate_orbit
from src.constants import EARTH_GRAV_PARAM_M3_S2, WGS84_A_M, SC_ALTITUDE_KM, SC_ALTITUDE_RANGE_KM

def test_output_schema():
    df = propagate_orbit('2027-01-01T00:00:00', duration_s=100)
    required = [
        'epoch', 'pos_eci_x', 'pos_eci_y', 'pos_eci_z',
        'vel_eci_x', 'vel_eci_y', 'vel_eci_z',
        'lat_deg', 'lon_deg', 'alt_km', 'speed_ms',
    ]
    for col in required:
        assert col in df.columns, f"Missing column: {col}"
    assert len(df) == 11
    assert df.index[0] == 0 and df.index[-1] == 10

def test_units_are_si():
    df = propagate_orbit('2027-01-01T00:00:00', duration_s=10)
    pos_norm = np.linalg.norm(df[['pos_eci_x','pos_eci_y','pos_eci_z']].values[0])
    assert pos_norm > 1e6
    vel_norm = np.linalg.norm(df[['vel_eci_x','vel_eci_y','vel_eci_z']].values[0])
    assert 7400 < vel_norm < 7800

def test_orbital_speed():
    a_m = WGS84_A_M + SC_ALTITUDE_KM * 1e3
    v_theory = np.sqrt(EARTH_GRAV_PARAM_M3_S2 / a_m)
    df = propagate_orbit('2027-01-01T00:00:00', duration_s=10, altitude_km=SC_ALTITUDE_KM)
    v_actual = float(df['speed_ms'].iloc[0])
    assert abs(v_actual - v_theory) < 50

def test_orbital_period():
    df = propagate_orbit('2027-01-01T00:00:00', duration_s=12000, dt_s=10.0)
    z = df['pos_eci_z'].values
    crossings = np.where((z[:-1] < 0) & (z[1:] >= 0))[0]
    assert len(crossings) >= 2
    t0 = df['epoch'].iloc[crossings[0]]
    t1 = df['epoch'].iloc[crossings[1]]
    period_s = (t1 - t0).total_seconds()
    assert 5660 < period_s < 5720

def test_inclination():
    df = propagate_orbit('2027-01-01T00:00:00', duration_s=100)
    pos = df[['pos_eci_x','pos_eci_y','pos_eci_z']].values[0]
    vel = df[['vel_eci_x','vel_eci_y','vel_eci_z']].values[0]
    L = np.cross(pos, vel)
    inc_deg = np.degrees(np.arccos(L[2] / np.linalg.norm(L)))
    assert 97.0 < inc_deg < 97.9

def test_altitude_consistency():
    df = propagate_orbit('2027-01-01T00:00:00', duration_s=86400)
    lo, hi = SC_ALTITUDE_RANGE_KM
    in_range = df['alt_km'].between(lo - 20, hi + 20)
    assert in_range.all()

def test_raan_precession():
    df = propagate_orbit('2027-01-01T00:00:00', duration_s=7 * 86400, dt_s=600)
    pos = df[['pos_eci_x','pos_eci_y','pos_eci_z']].values
    vel = df[['vel_eci_x','vel_eci_y','vel_eci_z']].values
    L = np.cross(pos, vel)
    L_norm = L / np.linalg.norm(L, axis=1, keepdims=True)
    raan = np.degrees(np.arctan2(L_norm[:, 0], -L_norm[:, 1]))
    raan_unwrapped = np.unwrap(raan, period=360)
    t_days = np.arange(len(raan_unwrapped)) * 600 / 86400
    coeffs = np.polyfit(t_days, raan_unwrapped, 1)
    precession_rate = coeffs[0]
    assert 0.93 < precession_rate < 1.04

def test_geodetic_consistency():
    df = propagate_orbit('2027-01-01T00:00:00', duration_s=5700, dt_s=60)
    pos_norm = np.linalg.norm(df[['pos_eci_x','pos_eci_y','pos_eci_z']].values, axis=1)
    alt_from_pos_km = (pos_norm - WGS84_A_M) / 1e3
    alt_from_geodetic = df['alt_km'].values
    diff = np.abs(alt_from_pos_km - alt_from_geodetic)
    assert diff.max() < 25.0
