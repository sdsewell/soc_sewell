"""
Creates a WindCube L2 netCDF-4 file with the schema defined in S20.

Spec:         s20_l2_netcdf_wind_vector_schema_2026-04-11.md
Spec date:    2026-04-11
Generated:    2026-04-11  (Claude Code)
Tool:         Claude Code
Last tested:  2026-04-11
Implements:   windcube/m08_l2_writer.py
Depends on:   netCDF4, numpy

Implements S20.

Parameters
----------
filepath : str
    Output path for the netCDF-4 file.
n_obs : int
    Number of merged wind-vector observations to pre-allocate.
pipeline_version : str
    soc_sewell release tag written to the global attribute.
git_sha : str
    Git commit SHA written to the global attribute.

Returns
-------
nc.Dataset
    Open (writable) Dataset. Caller must call ds.close().
"""
import logging
from datetime import datetime, timezone

import netCDF4 as nc
import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level fill-value constants (CF conventions)
# ---------------------------------------------------------------------------
FILL_F32 = np.float32(np.nan)        # IEEE 754 NaN fill for float32 (S20 §7)
FILL_U8 = np.uint8(255)              # fill for uint8 flag variables
FILL_I32 = np.int32(-2147483647)     # fill for int32 variables

# Required science variables checked by write_l2_file()
_REQUIRED_VARS = ("u_wind", "v_wind", "u_wind_error", "v_wind_error",
                  "time", "latitude", "longitude")


def create_l2_file(
    filepath: str,
    n_obs: int,
    pipeline_version: str = "1.0.0",
    git_sha: str = "unknown",
) -> nc.Dataset:
    """
    Create an empty, schema-conformant WindCube L2 netCDF-4 file.

    Parameters
    ----------
    filepath : str
        Output path, e.g. 'windcube_l2_20270315_v01.nc'.
    n_obs : int
        Number of merged wind vector observations to pre-allocate.
    pipeline_version : str
        soc_sewell release tag.
    git_sha : str
        Git commit SHA of pipeline code.

    Returns
    -------
    nc.Dataset
        Open (writable) dataset. Caller must call ds.close().
    """
    log.debug("Creating L2 file: %s  (n_obs=%d)", filepath, n_obs)
    ds = nc.Dataset(filepath, "w", format="NETCDF4")

    # ── Global attributes ───────────────────────────────────────────────────
    ds.Conventions = "CF-1.8"
    ds.title = "WindCube L2 thermospheric horizontal wind vectors"
    ds.institution = "High Altitude Observatory, NCAR/UCAR, Boulder CO, USA"
    ds.source = "WindCube CubeSat FPI, OI 630.0 nm airglow, SSO LEO ~510 km"
    ds.processing_level = "L2"
    ds.pipeline_version = pipeline_version
    ds.git_sha = git_sha
    ds.date_created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ds.wavelength_nm = np.float32(630.0)
    ds.emission_species = "OI 630.0 nm"
    ds.orbit_mode = "SSO dawn-dusk, LTAN 0600/1800 ± 1 hr"
    ds.orbit_altitude_km = np.float32(510.0)
    ds.etalon_gap_mm = np.float32(20.106)    # Tolansky operational value
    ds.focal_length_mm = np.float32(199.12)  # Tolansky operational value
    ds.references = "Sewell et al. (in prep)"
    ds.acknowledgements = "NSF CubeSat program; NCAR/HAO"

    # ── Dimension ───────────────────────────────────────────────────────────
    ds.createDimension("obs", None)  # unlimited

    # ── Private variable-creation helpers ───────────────────────────────────
    def _fvar(name: str, dims: tuple = ("obs",), fill: np.float32 = FILL_F32,
              **attrs) -> nc.Variable:
        """Create a compressed float32 variable."""
        v = ds.createVariable(name, "f4", dims,
                              fill_value=fill, zlib=True, complevel=4)
        v.setncatts(attrs)
        return v

    def _u8var(name: str, dims: tuple = ("obs",), **attrs) -> nc.Variable:
        """Create a compressed uint8 variable."""
        v = ds.createVariable(name, "u1", dims,
                              fill_value=FILL_U8, zlib=True, complevel=4)
        v.setncatts(attrs)
        return v

    def _i32var(name: str, dims: tuple = ("obs",), **attrs) -> nc.Variable:
        """Create a compressed int32 variable."""
        v = ds.createVariable(name, "i4", dims,
                              fill_value=FILL_I32, zlib=True, complevel=4)
        v.setncatts(attrs)
        return v

    # ── Coordinate variables ─────────────────────────────────────────────────
    t = ds.createVariable("time", "f8", ("obs",), zlib=True, complevel=4)
    t.units = "seconds since 2000-01-01 12:00:00"
    t.long_name = "UTC time at tangent-point centre of exposure"
    t.calendar = "standard"
    t.axis = "T"
    t.valid_range = np.array([0.0, 1e10], dtype="f8")

    _fvar("latitude",
          long_name="Geodetic latitude of OI 630 nm emission tangent point",
          units="degrees_north",
          standard_name="latitude",
          valid_range=np.array([-90.0, 90.0], dtype="f4"),
          axis="Y",
          comment="Emission-weighted mean geodetic latitude within the 1.65° FoV")

    _fvar("longitude",
          long_name="Geodetic longitude of OI 630 nm emission tangent point",
          units="degrees_east",
          standard_name="longitude",
          valid_range=np.array([-180.0, 180.0], dtype="f4"),
          axis="X")

    _fvar("altitude",
          long_name="Emission-weighted tangent height of OI 630 nm volume",
          units="km",
          valid_range=np.array([100.0, 350.0], dtype="f4"),
          comment="Nominal emission peak ~250 km; integration spans ~150–290 km")

    _fvar("sza",
          long_name="Solar zenith angle at tangent point",
          units="degrees",
          valid_range=np.array([0.0, 180.0], dtype="f4"),
          comment="Science observations nominally require sza > 100° (nightside); see sza_flag")

    # ── Science: primary wind vector ─────────────────────────────────────────
    _fvar("u_wind",
          long_name="Zonal wind velocity (positive eastward)",
          units="m/s",
          standard_name="eastward_wind",
          valid_range=np.array([-1000.0, 1000.0], dtype="f4"),
          comment="Merged from along-track and cross-track LOS using Merger decomposition; "
                  "positive = eastward",
          ancillary_variables="u_wind_error quality_flag")

    _fvar("v_wind",
          long_name="Meridional wind velocity (positive northward)",
          units="m/s",
          standard_name="northward_wind",
          valid_range=np.array([-1000.0, 1000.0], dtype="f4"),
          comment="Merged from along-track and cross-track LOS; positive = northward",
          ancillary_variables="v_wind_error quality_flag")

    _fvar("u_wind_error",
          long_name="1-sigma uncertainty in zonal wind",
          units="m/s",
          valid_range=np.array([0.0, 500.0], dtype="f4"),
          comment="Propagated from LOS velocity uncertainties through Merger decomposition matrix")

    _fvar("v_wind_error",
          long_name="1-sigma uncertainty in meridional wind",
          units="m/s",
          valid_range=np.array([0.0, 500.0], dtype="f4"),
          comment="Propagated from LOS velocity uncertainties through Merger decomposition matrix")

    _fvar("wind_speed",
          long_name="Horizontal wind speed",
          units="m/s",
          valid_range=np.array([0.0, 1000.0], dtype="f4"),
          comment="sqrt(u_wind**2 + v_wind**2)")

    _fvar("wind_direction",
          long_name="Horizontal wind direction, meteorological convention",
          units="degrees",
          valid_range=np.array([0.0, 360.0], dtype="f4"),
          comment="Direction FROM which wind blows, measured clockwise from north (0–360°). "
                  "Met convention: 0° = wind from north, 90° = wind from east.")

    # ── Science: LOS provenance ──────────────────────────────────────────────
    for tag in ("along", "cross"):
        _fvar(f"los_velocity_{tag}",
              long_name=f"Calibrated LOS wind velocity from {tag}-track orbit",
              units="m/s",
              comment="Positive = toward instrument (redshift). "
                      "L1c product value; spacecraft velocity correction applied.")
        _fvar(f"los_error_{tag}",
              long_name=f"1-sigma LOS velocity uncertainty ({tag}-track)",
              units="m/s")

    # ── Quality flags ────────────────────────────────────────────────────────
    _u8var("quality_flag",
           long_name="Master quality flag",
           flag_values=np.array([0, 1, 2], dtype="u1"),
           flag_meanings="good caution bad",
           comment="Bit 0 (0x01): LOS fit residual > 3σ threshold; "
                   "Bit 1 (0x02): Merger geometry angle < 20° (near-degenerate); "
                   "Bit 2 (0x04): Emission rate below noise floor; "
                   "Bit 3 (0x08): Data gap in one of the pair orbits; "
                   "Bit 4 (0x10): SZA not in nightglow range (> 102°); "
                   "Bit 5 (0x20): Etalon temperature outside calibrated range")

    _u8var("sza_flag",
           long_name="Solar zenith angle regime flag",
           flag_values=np.array([0, 1, 2], dtype="u1"),
           flag_meanings="nightside twilight dayside")

    _u8var("dayglow_flag",
           long_name="Dayglow contamination flag",
           flag_values=np.array([0, 1, 2], dtype="u1"),
           flag_meanings="none corrected excessive")

    _u8var("data_gap_flag",
           long_name="Data gap flag for orbit pair completeness",
           flag_values=np.array([0, 1, 2, 3], dtype="u1"),
           flag_meanings="complete along_missing cross_missing both_missing")

    # ── Orbit ancillary ──────────────────────────────────────────────────────
    _i32var("orbit_number",
            long_name="Sequential orbit number since launch")

    _u8var("look_direction",
           long_name="Payload look direction for this observation",
           flag_values=np.array([0, 1], dtype="u1"),
           flag_meanings="along_track cross_track")

    _fvar("sc_latitude",
          long_name="Spacecraft geodetic latitude",
          units="degrees_north",
          valid_range=np.array([-90.0, 90.0], dtype="f4"))

    _fvar("sc_longitude",
          long_name="Spacecraft geodetic longitude",
          units="degrees_east",
          valid_range=np.array([-180.0, 180.0], dtype="f4"))

    _fvar("sc_altitude",
          long_name="Spacecraft geodetic altitude",
          units="km",
          valid_range=np.array([400.0, 650.0], dtype="f4"))

    # ── Instrument state ─────────────────────────────────────────────────────
    _fvar("etalon_temp",
          long_name="Etalon temperature sensor 1",
          units="degrees_Celsius",
          comment="From L1a telemetry header; affects etalon gap and FSR")

    _fvar("exposure_time",
          long_name="Science exposure duration",
          units="s",
          comment="Nominal 10 s; may vary during commissioning")

    _fvar("emission_rate",
          long_name="OI 630 nm emission rate from spectral fit",
          units="Rayleigh",
          valid_range=np.array([0.0, 10000.0], dtype="f4"),
          comment="Proxy for nightglow signal strength; used in quality assessment")

    log.debug("L2 schema created: %d variables", len(ds.variables))
    return ds


def write_l2_file(
    filepath: str,
    data: dict,
    pipeline_version: str = "1.0.0",
    git_sha: str = "unknown",
) -> None:
    """
    Create a schema-conformant L2 file, write data arrays, and close.

    Wraps create_l2_file(), writes all arrays in *data* by variable name,
    then closes the Dataset.

    Parameters
    ----------
    filepath : str
        Output path for the netCDF-4 file.
    data : dict
        Mapping of variable name → numpy array.  All keys must match
        variable names defined in the S20 schema.  Required keys:
        u_wind, v_wind, u_wind_error, v_wind_error, time, latitude,
        longitude.
    pipeline_version : str
        soc_sewell release tag.
    git_sha : str
        Git commit SHA of pipeline code.

    Raises
    ------
    ValueError
        If any required science variable is absent from *data*.
    """
    missing = [k for k in _REQUIRED_VARS if k not in data]
    if missing:
        raise ValueError(f"write_l2_file: missing required variables: {missing}")

    n_obs = len(data["time"])
    ds = create_l2_file(filepath, n_obs=n_obs,
                        pipeline_version=pipeline_version, git_sha=git_sha)
    try:
        for varname, array in data.items():
            if varname in ds.variables:
                ds[varname][:] = array
            else:
                log.warning("write_l2_file: unknown variable '%s' — skipped", varname)
    finally:
        ds.close()
    log.debug("L2 file written and closed: %s", filepath)
