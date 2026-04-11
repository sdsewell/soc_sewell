"""
Acceptance tests for the WindCube L2 netCDF schema (S20).

Spec:  s20_l2_netcdf_wind_vector_schema_2026-04-11.md
Tests: §14 (10 spec tests) + 2 additional tests = 12 total
"""
import os

import netCDF4 as nc
import numpy as np
import pytest

from windcube.m08_l2_writer import create_l2_file, write_l2_file

# ---------------------------------------------------------------------------
# Synthetic data fixture
# ---------------------------------------------------------------------------
N = 120  # ~15 s cadence × 30-min orbit segment


@pytest.fixture
def l2_path(tmp_path):
    """
    Write a schema-conformant L2 file with N=120 synthetic observations.

    Uses write_l2_file() (convenience wrapper) so that test_11 can verify
    the two creation paths produce identical schemas.
    """
    np.random.seed(42)

    p = str(tmp_path / "test_l2.nc")
    u = np.random.uniform(-200, 200, N).astype("f4")
    v = np.random.uniform(-200, 200, N).astype("f4")

    data = {
        "time":               np.linspace(8.1e8, 8.1e8 + N * 10.0, N),
        "latitude":           np.linspace(-40.0, 40.0, N, dtype="f4"),
        "longitude":          np.full(N, -75.0, dtype="f4"),
        "altitude":           np.full(N, 250.0, dtype="f4"),
        "sza":                np.full(N, 110.0, dtype="f4"),
        "u_wind":             u,
        "v_wind":             v,
        "u_wind_error":       np.full(N, 9.8, dtype="f4"),
        "v_wind_error":       np.full(N, 9.8, dtype="f4"),
        "wind_speed":         np.sqrt(u ** 2 + v ** 2).astype("f4"),
        "wind_direction":     ((270.0 - np.degrees(np.arctan2(v, u))) % 360.0).astype("f4"),
        "quality_flag":       np.zeros(N, dtype="u1"),
        "sza_flag":           np.zeros(N, dtype="u1"),
        "dayglow_flag":       np.zeros(N, dtype="u1"),
        "data_gap_flag":      np.zeros(N, dtype="u1"),
        "look_direction":     np.zeros(N, dtype="u1"),
        "orbit_number":       np.full(N, 142, dtype="i4"),
        "emission_rate":      np.full(N, 150.0, dtype="f4"),
        "los_velocity_along": np.zeros(N, dtype="f4"),
        "los_velocity_cross": np.zeros(N, dtype="f4"),
        "los_error_along":    np.full(N, 5.0, dtype="f4"),
        "los_error_cross":    np.full(N, 5.0, dtype="f4"),
        "sc_latitude":        np.full(N, 0.0, dtype="f4"),
        "sc_longitude":       np.full(N, 0.0, dtype="f4"),
        "sc_altitude":        np.full(N, 510.0, dtype="f4"),
        "etalon_temp":        np.full(N, 22.0, dtype="f4"),
        "exposure_time":      np.full(N, 10.0, dtype="f4"),
    }
    write_l2_file(p, data)
    return p


# ---------------------------------------------------------------------------
# Test 1 — §14
# ---------------------------------------------------------------------------

def test_01_required_variables_present(l2_path):
    """All variables listed in §11 must be present."""
    required = [
        "time", "latitude", "longitude", "altitude", "sza",
        "u_wind", "v_wind", "u_wind_error", "v_wind_error",
        "wind_speed", "wind_direction",
        "los_velocity_along", "los_velocity_cross",
        "los_error_along", "los_error_cross",
        "quality_flag", "sza_flag", "dayglow_flag", "data_gap_flag",
        "orbit_number", "look_direction",
        "sc_latitude", "sc_longitude", "sc_altitude",
        "etalon_temp", "exposure_time", "emission_rate",
    ]
    with nc.Dataset(l2_path) as ds:
        for v in required:
            assert v in ds.variables, f"Missing variable: {v}"


# ---------------------------------------------------------------------------
# Test 2 — §14
# ---------------------------------------------------------------------------

def test_02_cf_conventions(l2_path):
    """CF-1.8 required attributes and coordinate metadata."""
    with nc.Dataset(l2_path) as ds:
        assert ds.Conventions == "CF-1.8"
        assert ds.processing_level == "L2"
        assert "time" in ds.variables
        assert ds["latitude"].units == "degrees_north"
        assert ds["longitude"].units == "degrees_east"
        assert ds["time"].units.startswith("seconds since 2000-01-01")


# ---------------------------------------------------------------------------
# Test 3 — §14
# ---------------------------------------------------------------------------

def test_03_obs_dimension_unlimited(l2_path):
    """The 'obs' dimension must be UNLIMITED (extendable)."""
    with nc.Dataset(l2_path) as ds:
        assert ds.dimensions["obs"].isunlimited()


# ---------------------------------------------------------------------------
# Test 4 — §14
# ---------------------------------------------------------------------------

def test_04_wind_dtypes(l2_path):
    """Primary wind variables must be float32."""
    with nc.Dataset(l2_path) as ds:
        for v in ("u_wind", "v_wind", "u_wind_error", "v_wind_error",
                  "wind_speed", "wind_direction"):
            assert ds[v].dtype == np.float32, f"{v} must be float32"


# ---------------------------------------------------------------------------
# Test 5 — §14
# ---------------------------------------------------------------------------

def test_05_quality_flag_range(l2_path):
    """quality_flag values must be in {0, 1, 2} (fill 255 excluded from data)."""
    with nc.Dataset(l2_path) as ds:
        qf = ds["quality_flag"][:]
        assert qf.max() <= 2, "quality_flag values must be 0, 1, or 2"


# ---------------------------------------------------------------------------
# Test 6 — §14
# ---------------------------------------------------------------------------

def test_06_wind_speed_consistency(l2_path):
    """wind_speed == sqrt(u_wind² + v_wind²) to float32 tolerance."""
    with nc.Dataset(l2_path) as ds:
        u = ds["u_wind"][:]
        v = ds["v_wind"][:]
        ws = ds["wind_speed"][:]
        expected = np.sqrt(u ** 2 + v ** 2)
        np.testing.assert_allclose(ws, expected, rtol=1e-5,
                                   err_msg="wind_speed != sqrt(u^2 + v^2)")


# ---------------------------------------------------------------------------
# Test 7 — §14
# ---------------------------------------------------------------------------

def test_07_wind_direction_range(l2_path):
    """wind_direction must be in [0, 360] for non-fill values."""
    with nc.Dataset(l2_path) as ds:
        wd = ds["wind_direction"][:]
        valid = ~np.isnan(wd)
        assert np.all(wd[valid] >= 0.0) and np.all(wd[valid] <= 360.0), \
            "wind_direction must be in [0, 360]"


# ---------------------------------------------------------------------------
# Test 8 — §14
# ---------------------------------------------------------------------------

def test_08_science_traceability_precision(l2_path):
    """SQ1/SQ2: wind error must be physically plausible (< 34 m/s STM budget)."""
    with nc.Dataset(l2_path) as ds:
        sigma_u = ds["u_wind_error"][:]
        sigma_v = ds["v_wind_error"][:]
        assert np.nanmean(sigma_u) < 34.0, "Mean u_wind_error exceeds STM budget"
        assert np.nanmean(sigma_v) < 34.0, "Mean v_wind_error exceeds STM budget"


# ---------------------------------------------------------------------------
# Test 9 — §14
# ---------------------------------------------------------------------------

def test_09_latitude_coverage(l2_path):
    """SQ1: ±40°; SQ2: ±30° — latitude range must span at least ±30°."""
    with nc.Dataset(l2_path) as ds:
        lat = ds["latitude"][:]
        assert lat.min() <= -30.0, "Latitude coverage must reach -30°"
        assert lat.max() >= 30.0, "Latitude coverage must reach +30°"


# ---------------------------------------------------------------------------
# Test 10 — §14
# ---------------------------------------------------------------------------

def test_10_global_attributes(l2_path):
    """All required global attributes from S20 §5 must be present."""
    required_attrs = [
        "Conventions", "title", "institution", "source",
        "processing_level", "pipeline_version", "git_sha",
        "date_created", "wavelength_nm", "emission_species",
        "orbit_mode", "orbit_altitude_km",
        "etalon_gap_mm", "focal_length_mm",
        "references", "acknowledgements",
    ]
    with nc.Dataset(l2_path) as ds:
        for a in required_attrs:
            assert hasattr(ds, a), f"Missing global attribute: {a}"


# ---------------------------------------------------------------------------
# Test 11 — additional: file_format
# ---------------------------------------------------------------------------

def test_11_write_l2_file_convenience(tmp_path):
    """
    write_l2_file() must produce an identical schema to create_l2_file()
    and the file must be properly closed (reopenable) after the call.
    """
    np.random.seed(42)
    N2 = 10
    u = np.random.uniform(-100, 100, N2).astype("f4")
    v = np.random.uniform(-100, 100, N2).astype("f4")

    data = {
        "time":               np.linspace(8.1e8, 8.1e8 + N2 * 10.0, N2),
        "latitude":           np.linspace(-10.0, 10.0, N2, dtype="f4"),
        "longitude":          np.full(N2, 0.0, dtype="f4"),
        "altitude":           np.full(N2, 250.0, dtype="f4"),
        "sza":                np.full(N2, 110.0, dtype="f4"),
        "u_wind":             u,
        "v_wind":             v,
        "u_wind_error":       np.full(N2, 9.8, dtype="f4"),
        "v_wind_error":       np.full(N2, 9.8, dtype="f4"),
        "wind_speed":         np.sqrt(u ** 2 + v ** 2).astype("f4"),
        "wind_direction":     ((270.0 - np.degrees(np.arctan2(v, u))) % 360.0).astype("f4"),
        "quality_flag":       np.zeros(N2, dtype="u1"),
        "sza_flag":           np.zeros(N2, dtype="u1"),
        "dayglow_flag":       np.zeros(N2, dtype="u1"),
        "data_gap_flag":      np.zeros(N2, dtype="u1"),
        "look_direction":     np.zeros(N2, dtype="u1"),
        "orbit_number":       np.full(N2, 1, dtype="i4"),
        "emission_rate":      np.full(N2, 150.0, dtype="f4"),
        "los_velocity_along": np.zeros(N2, dtype="f4"),
        "los_velocity_cross": np.zeros(N2, dtype="f4"),
        "los_error_along":    np.full(N2, 5.0, dtype="f4"),
        "los_error_cross":    np.full(N2, 5.0, dtype="f4"),
        "sc_latitude":        np.full(N2, 0.0, dtype="f4"),
        "sc_longitude":       np.full(N2, 0.0, dtype="f4"),
        "sc_altitude":        np.full(N2, 510.0, dtype="f4"),
        "etalon_temp":        np.full(N2, 22.0, dtype="f4"),
        "exposure_time":      np.full(N2, 10.0, dtype="f4"),
    }

    path_w = str(tmp_path / "via_write.nc")
    path_c = str(tmp_path / "via_create.nc")

    # write_l2_file path
    write_l2_file(path_w, data)

    # create_l2_file path — identical schema check
    ds_c = create_l2_file(path_c, n_obs=N2)
    vars_c = sorted(ds_c.variables.keys())
    dims_c = {k: v.isunlimited() for k, v in ds_c.dimensions.items()}
    ds_c.close()

    # File must be closed and reopenable
    with nc.Dataset(path_w) as ds_w:
        assert sorted(ds_w.variables.keys()) == vars_c, \
            "write_l2_file variables differ from create_l2_file"
        assert {k: v.isunlimited() for k, v in ds_w.dimensions.items()} == dims_c, \
            "write_l2_file dimensions differ from create_l2_file"
        assert ds_w.Conventions == "CF-1.8"


# ---------------------------------------------------------------------------
# Test 12 — additional: fill values
# ---------------------------------------------------------------------------

def test_12_fill_values_correct(tmp_path):
    """
    Float32 NaN fill is masked correctly; orbit_number fill is -2147483647.
    """
    from windcube.m08_l2_writer import FILL_I32

    p = str(tmp_path / "fill_test.nc")
    N3 = 5
    u = np.ones(N3, dtype="f4") * 10.0
    v = np.ones(N3, dtype="f4") * 10.0

    data = {
        "time":               np.linspace(8.1e8, 8.1e8 + N3 * 10.0, N3),
        "latitude":           np.zeros(N3, dtype="f4"),
        "longitude":          np.zeros(N3, dtype="f4"),
        "altitude":           np.full(N3, 250.0, dtype="f4"),
        "sza":                np.full(N3, 110.0, dtype="f4"),
        "u_wind":             u.copy(),
        "v_wind":             v.copy(),
        "u_wind_error":       np.full(N3, 9.8, dtype="f4"),
        "v_wind_error":       np.full(N3, 9.8, dtype="f4"),
        "wind_speed":         np.sqrt(u ** 2 + v ** 2).astype("f4"),
        "wind_direction":     np.zeros(N3, dtype="f4"),
        "quality_flag":       np.zeros(N3, dtype="u1"),
        "sza_flag":           np.zeros(N3, dtype="u1"),
        "dayglow_flag":       np.zeros(N3, dtype="u1"),
        "data_gap_flag":      np.zeros(N3, dtype="u1"),
        "look_direction":     np.zeros(N3, dtype="u1"),
        "orbit_number":       np.full(N3, 1, dtype="i4"),
        "emission_rate":      np.full(N3, 150.0, dtype="f4"),
        "los_velocity_along": np.zeros(N3, dtype="f4"),
        "los_velocity_cross": np.zeros(N3, dtype="f4"),
        "los_error_along":    np.full(N3, 5.0, dtype="f4"),
        "los_error_cross":    np.full(N3, 5.0, dtype="f4"),
        "sc_latitude":        np.zeros(N3, dtype="f4"),
        "sc_longitude":       np.zeros(N3, dtype="f4"),
        "sc_altitude":        np.full(N3, 510.0, dtype="f4"),
        "etalon_temp":        np.full(N3, 22.0, dtype="f4"),
        "exposure_time":      np.full(N3, 10.0, dtype="f4"),
    }
    write_l2_file(p, data)

    # Now open again and overwrite obs[0] of u_wind with NaN
    with nc.Dataset(p, "a") as ds:
        ds["u_wind"][0] = np.nan

    with nc.Dataset(p) as ds:
        u_arr = ds["u_wind"][:]
        # netCDF4 should mask the NaN element
        assert np.ma.is_masked(u_arr[0]), \
            "u_wind[0] set to NaN should be masked when read back"
        # orbit_number fill value check
        assert int(ds["orbit_number"]._FillValue) == int(FILL_I32), \
            f"orbit_number._FillValue should be {int(FILL_I32)}"
