"""
Tests for S05 NB00 truth wind map module.

Spec:        S05_nb00_wind_map_2026-04-05.md
Spec tests:  T1a–T4b
Run with:    pytest tests/test_s05_nb00_wind_map.py -v
"""

import numpy as np
import pytest

from src.windmap import (
    AnalyticWindMap,
    GridWindMap,
    HWM14WindMap,
    StormWindMap,
    UniformWindMap,
)


# =============================================================================
# T1 — UniformWindMap (3 tests, no external dependencies)
# =============================================================================

def test_T1a_uniform_basic():
    """T1a — Basic return value."""
    wm = UniformWindMap(v_zonal_ms=100.0, v_merid_ms=50.0)
    vz, vm = wm.sample(lat_deg=30.0, lon_deg=45.0)
    assert vz == 100.0
    assert vm == 50.0


def test_T1b_uniform_spatial_invariance():
    """T1b — Spatial invariance at poles and antimeridian."""
    wm = UniformWindMap(v_zonal_ms=100.0, v_merid_ms=0.0)
    test_points = [
        (90.0,  0.0),    # North pole
        (-90.0, 0.0),    # South pole
        (0.0,  180.0),   # Antimeridian east
        (0.0, -180.0),   # Antimeridian west
        (0.0,   0.0),    # Prime meridian / equator
        (45.0, -120.0),  # Mid-latitude, west
    ]
    for lat, lon in test_points:
        vz, vm = wm.sample(lat, lon)
        assert vz == 100.0, f"Failed at ({lat}, {lon}): vz = {vz}"
        assert vm == 0.0,   f"Failed at ({lat}, {lon}): vm = {vm}"


def test_T1c_uniform_sample_array():
    """T1c — sample_array vectorisation."""
    wm = UniformWindMap(v_zonal_ms=100.0, v_merid_ms=50.0)
    lats = np.array([-30.0, 0.0, 30.0])
    lons = np.array([-120.0, 0.0, 120.0])
    vz_arr, vm_arr = wm.sample_array(lats, lons)
    np.testing.assert_array_equal(vz_arr, [100.0, 100.0, 100.0])
    np.testing.assert_array_equal(vm_arr, [50.0, 50.0, 50.0])


# =============================================================================
# T2 — AnalyticWindMap (4 tests, no external dependencies)
# =============================================================================

def test_T2a_sine_lat_equator():
    """T2a — Sine-lat pattern at equator: vz ≈ 0, vm ≈ A_merid."""
    wm = AnalyticWindMap(pattern="sine_lat", A_zonal_ms=200.0, A_merid_ms=100.0)
    vz, vm = wm.sample(lat_deg=0.0, lon_deg=0.0)
    assert abs(vz) < 1.0, f"Equatorial zonal wind should be ~0, got {vz:.1f}"
    assert abs(vm - 100.0) < 1.0, f"Equatorial meridional should be ~100, got {vm:.1f}"


def test_T2b_sine_lat_pole():
    """T2b — Sine-lat pattern at 90°N: vz ≈ A_zonal, vm ≈ 0."""
    wm = AnalyticWindMap(pattern="sine_lat", A_zonal_ms=200.0, A_merid_ms=100.0)
    vz, vm = wm.sample(lat_deg=90.0, lon_deg=0.0)
    assert abs(vz - 200.0) < 2.0, f"Polar zonal should be ~200 m/s, got {vz:.1f}"
    assert abs(vm) < 2.0, f"Polar meridional should be ~0, got {vm:.1f}"


def test_T2c_wave4_longitude_cycles():
    """T2c — Wave-4 pattern has 4 longitude cycles (~8 zero-crossings)."""
    wm = AnalyticWindMap(pattern="wave4", A_zonal_ms=150.0, A_merid_ms=75.0)
    lons = np.linspace(-180, 180, 361)
    vz_vals = np.array([wm.sample(30.0, lon)[0] for lon in lons])
    crossings = np.sum(np.diff(np.sign(vz_vals)) != 0)
    assert 7 <= crossings <= 9, f"Wave-4 should have ~8 zero-crossings, got {crossings}"


def test_T2d_netcdf_roundtrip(tmp_path):
    """T2d — NetCDF round-trip preserves wind values to within 0.5 m/s."""
    wm = AnalyticWindMap(pattern="sine_lat")
    filepath = str(tmp_path / "test_T2.nc")
    wm.to_netcdf(filepath, metadata={
        "wind_map_type": "T2_analytic",
        "alt_km": 250.0,
        "description": "test",
        "pipeline_version": "dev",
    })
    wm2 = GridWindMap.from_netcdf(filepath)
    for lat, lon in [(30.0, 45.0), (-45.0, -90.0), (0.0, 180.0)]:
        vz1, vm1 = wm.sample(lat, lon)
        vz2, vm2 = wm2.sample(lat, lon)
        assert abs(vz1 - vz2) < 0.5, f"NetCDF round-trip vz mismatch at ({lat},{lon})"
        assert abs(vm1 - vm2) < 0.5, f"NetCDF round-trip vm mismatch at ({lat},{lon})"


# =============================================================================
# T3 — HWM14WindMap (3 tests, skip if hwm14 not installed)
# =============================================================================

def test_T3a_hwm14_plausible():
    """T3a — Output is physically plausible at 250 km."""
    pytest.importorskip("hwm14", reason="hwm14 not installed; skipping T3")
    wm = HWM14WindMap(alt_km=250.0, day_of_year=172, ut_hours=12.0, f107=150.0, ap=4)
    vz, vm = wm.sample(lat_deg=30.0, lon_deg=120.0)
    assert -500 < vz < 500, f"HWM14 v_zonal out of range: {vz:.1f} m/s"
    assert -300 < vm < 300, f"HWM14 v_merid out of range: {vm:.1f} m/s"


def test_T3b_hwm14_no_nan():
    """T3b — Global grid: no NaN or Inf values."""
    pytest.importorskip("hwm14", reason="hwm14 not installed; skipping T3")
    wm = HWM14WindMap(alt_km=250.0, day_of_year=172, ut_hours=12.0, f107=150.0, ap=4)
    assert not np.any(np.isnan(wm.v_zonal_grid)), "HWM14 grid contains NaN in v_zonal"
    assert not np.any(np.isnan(wm.v_merid_grid)), "HWM14 grid contains NaN in v_merid"
    assert not np.any(np.isinf(wm.v_zonal_grid)), "HWM14 grid contains Inf in v_zonal"


def test_T3c_hwm14_eastward_bias():
    """T3c — Zonal wind exceeds 50 m/s eastward somewhere in the equatorial band.

    HWM14 has strong DE3 wave-4 longitude structure at low latitudes; the
    zonal wind can be weakly westward at some longitudes and strongly eastward
    at others.  The test samples four quadrant longitudes and checks that the
    maximum exceeds 50 m/s, confirming the model is producing the expected
    thermospheric eastward jet.
    """
    pytest.importorskip("hwm14", reason="hwm14 not installed; skipping T3")
    wm = HWM14WindMap(alt_km=250.0, day_of_year=172, ut_hours=12.0, f107=150.0, ap=4)
    lats = np.arange(-20, 21, 5.0)
    max_zonal = -np.inf
    for lon in [0.0, 90.0, 180.0, 270.0]:
        lons = np.full_like(lats, lon)
        vz, _ = wm.sample_array(lats, lons)
        max_zonal = max(max_zonal, float(np.max(vz)))
    assert max_zonal > 50.0, (
        f"HWM14 equatorial v_zonal max over quadrant lons = {max_zonal:.1f} m/s; "
        f"expected > 50 m/s eastward at least at one longitude"
    )


# =============================================================================
# T4 — StormWindMap (2 tests, skip if hwm14 not installed)
# =============================================================================

def test_T4a_storm_equatorward_surge():
    """T4a — Storm minus quiet is equatorward (< -30 m/s) at 60°N somewhere.

    DWM07 disturbance winds have strong longitude dependence.  The test samples
    four quadrant longitudes and checks that the equatorward surge exceeds
    30 m/s at least at one longitude.
    """
    pytest.importorskip("hwm14", reason="hwm14 not installed; skipping T4")
    quiet = HWM14WindMap(alt_km=250.0, day_of_year=355, ut_hours=3.0,
                         f107=180.0, f107a=180.0, ap=4)
    storm = StormWindMap(alt_km=250.0, day_of_year=355, ut_hours=3.0,
                         f107=180.0, f107a=180.0, ap=80)
    min_delta = np.inf
    for lon in [0.0, 90.0, 180.0, 270.0]:
        _, vm_q = quiet.sample(60.0, lon)
        _, vm_s = storm.sample(60.0, lon)
        min_delta = min(min_delta, vm_s - vm_q)
    assert min_delta < -30.0, (
        f"Storm equatorward surge at 60°N should be < -30 m/s at some longitude, "
        f"got min delta_vm = {min_delta:.1f} m/s"
    )


# =============================================================================
# T1e — plot() four-mode smoke test (no display required)
# =============================================================================

def test_T1_plot_all_modes():
    """T1e — plot() completes without exception for all four modes (Agg backend)."""
    pytest.importorskip("cartopy", reason="cartopy not installed; skipping T1e")
    import matplotlib
    matplotlib.use('Agg')   # non-interactive — must be set before pyplot import
    import matplotlib.pyplot as plt

    wm_t1 = UniformWindMap(v_zonal_ms=150.0, v_merid_ms=-75.0)
    wm_t2 = AnalyticWindMap(pattern='wave4')

    for wm, label in [(wm_t1, 'T1'), (wm_t2, 'T2')]:
        for mode in ('separate', 'vector', 'stream', 'magnitude'):
            try:
                wm.plot(title=f'{label} test', mode=mode)
                plt.close('all')
            except Exception as exc:
                pytest.fail(
                    f"plot(mode='{mode}') raised {type(exc).__name__} "
                    f"for {label}: {exc}"
                )


def test_T4b_storm_magnitude_bounded():
    """T4b — Storm peak wind is physically bounded (< 1.5 × WIND_MAX_STORM_MS)."""
    pytest.importorskip("hwm14", reason="hwm14 not installed; skipping T4")
    from src.constants import WIND_MAX_STORM_MS
    storm = StormWindMap(ap=80)
    lats = np.arange(-60, 61, 10.0)
    lons = np.zeros_like(lats)
    vz, vm = storm.sample_array(lats, lons)
    max_speed = np.max(np.sqrt(vz ** 2 + vm ** 2))
    assert max_speed < WIND_MAX_STORM_MS * 1.5, (
        f"Storm wind speed {max_speed:.0f} m/s exceeds 1.5 × STM max ({WIND_MAX_STORM_MS} m/s)"
    )
