"""
NB00 — Truth wind map module.

Spec:        S05_nb00_wind_map_2026-04-06.md
Spec date:   2026-04-05
Generated:   2026-04-05
Tool:        Claude Code
Last tested: 2026-04-05  (7/7 T1+T2 tests pass; T3/T4 deferred — hwm14 not installed)
Depends on:  src.constants
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from abc import ABC, abstractmethod
from datetime import datetime, timezone

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from src.constants import (
    LAT_RANGE_DEG,
    TP_ALTITUDE_KM,
    WIND_BIAS_BUDGET_MS,
    WIND_MAX_STORM_MS,
)


# =============================================================================
# WindMap — abstract base class
# =============================================================================

class WindMap(ABC):
    """
    Abstract base class for thermospheric truth wind maps.

    All concrete subclasses must implement sample(). Grid convention:
        v_zonal : eastward wind component, m/s  (positive = eastward)
        v_merid : northward wind component, m/s (positive = northward)
        lat_grid : shape (180,), cell centres -89.5 to +89.5 deg
        lon_grid : shape (360,), cell centres -179.5 to +179.5 deg
        grids    : shape (180, 360), indexing [lat, lon]

    Wind sign conventions follow S02 Section 6.6:
        Zonal      positive = eastward
        Meridional positive = northward
    """

    @abstractmethod
    def sample(self, lat_deg: float, lon_deg: float) -> tuple[float, float]:
        """
        Return wind components at the given geodetic location.

        Parameters
        ----------
        lat_deg : float
            Geodetic latitude, degrees, -90 to +90.
        lon_deg : float
            Geodetic longitude, degrees, -180 to +180.
            Implementations must handle longitude wrapping (e.g. 181° → -179°).

        Returns
        -------
        (v_zonal_ms, v_merid_ms) : tuple[float, float]
            v_zonal_ms — eastward wind speed, m/s
            v_merid_ms — northward wind speed, m/s

        Notes
        -----
        Implementations must handle:
        - Longitude wrapping at ±180°
        - Pole clamping: lat_deg clamped to [-90, +90] if out of range
        - Bilinear interpolation for all grid-based backends (T2–T4)
        """

    def sample_array(
        self,
        lat_deg: np.ndarray,
        lon_deg: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Vectorised wrapper — call sample() over arrays of lat/lon.

        Parameters
        ----------
        lat_deg : np.ndarray, shape (N,)
        lon_deg : np.ndarray, shape (N,)

        Returns
        -------
        (v_zonal_ms, v_merid_ms) : tuple of np.ndarray, each shape (N,), float64
        """
        lat_deg = np.asarray(lat_deg, dtype=float)
        lon_deg = np.asarray(lon_deg, dtype=float)
        assert lat_deg.shape == lon_deg.shape
        v_z = np.empty(lat_deg.shape, dtype=float)
        v_m = np.empty(lat_deg.shape, dtype=float)
        for i in range(lat_deg.size):
            v_z.flat[i], v_m.flat[i] = self.sample(lat_deg.flat[i], lon_deg.flat[i])
        return v_z, v_m

    def plot(
        self,
        title: str = "WindMap",
        alt_km: float = 250.0,
        subsample: int = 5,
    ) -> None:
        """
        Diagnostic plot: filled-colour zonal and meridional wind maps on a
        global Plate Carrée projection, with wind vector quivers overplotted
        at every `subsample`-th grid point.

        Uses matplotlib. Tries cartopy for geographic axes; falls back to a
        plain imshow layout if cartopy is not installed.

        Parameters
        ----------
        title     : str — figure suptitle
        alt_km    : float — altitude label for the title
        subsample : int  — plot every Nth grid point for quivers
        """
        import matplotlib.pyplot as plt

        lat_grid = np.linspace(-89.5, 89.5, 180)
        lon_grid = np.linspace(-179.5, 179.5, 360)
        LON, LAT = np.meshgrid(lon_grid, lat_grid)

        vz, vm = self.sample_array(LAT.ravel(), LON.ravel())
        vz = vz.reshape(180, 360)
        vm = vm.reshape(180, 360)

        lim = max(np.abs(vz).max(), np.abs(vm).max(), 1.0)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, data, label in zip(axes, [vz, vm], ["v_zonal (m/s)", "v_merid (m/s)"]):
            im = ax.imshow(
                data,
                origin="lower",
                extent=[-180, 180, -90, 90],
                aspect="auto",
                vmin=-lim,
                vmax=lim,
                cmap="RdBu_r",
            )
            plt.colorbar(im, ax=ax, label=label)
            # quivers
            qs = subsample
            qlon = lon_grid[::qs]
            qlat = lat_grid[::qs]
            QLON, QLAT = np.meshgrid(qlon, qlat)
            qvz, qvm = self.sample_array(QLAT.ravel(), QLON.ravel())
            ax.quiver(
                QLON, QLAT,
                qvz.reshape(QLAT.shape), qvm.reshape(QLAT.shape),
                scale=lim * 20, width=0.002, color="k", alpha=0.4,
            )
            ax.set_xlabel("Longitude (deg)")
            ax.set_ylabel("Latitude (deg)")
            ax.set_title(label)

        fig.suptitle(f"{title}  |  alt = {alt_km:.0f} km")
        plt.tight_layout()
        plt.show()


# =============================================================================
# GridWindMap — grid-backed base class
# =============================================================================

class GridWindMap(WindMap):
    """
    WindMap backed by a 180×360 lat/lon grid with bilinear interpolation.

    Subclasses must populate self.v_zonal_grid and self.v_merid_grid
    (both shape (180, 360), float32) in their __init__ before calling
    self._build_interpolators().

    Grid layout:
        lat_grid[i] = -89.5 + i   degrees, i in 0..179
        lon_grid[j] = -179.5 + j  degrees, j in 0..359
    """

    LAT_GRID: np.ndarray = np.linspace(-89.5, 89.5, 180)   # cell centres, degrees
    LON_GRID: np.ndarray = np.linspace(-179.5, 179.5, 360)  # cell centres, degrees

    def _build_interpolators(self) -> None:
        """Build scipy RegularGridInterpolator objects from populated grids."""
        self._interp_zonal = RegularGridInterpolator(
            (self.LAT_GRID, self.LON_GRID),
            self.v_zonal_grid,
            method="linear",
            bounds_error=False,
            fill_value=None,    # extrapolate at poles
        )
        self._interp_merid = RegularGridInterpolator(
            (self.LAT_GRID, self.LON_GRID),
            self.v_merid_grid,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

    def sample(self, lat_deg: float, lon_deg: float) -> tuple[float, float]:
        """Bilinear interpolation at (lat_deg, lon_deg)."""
        # Wrap longitude to [-180, 180)
        lon_deg = ((lon_deg + 180.0) % 360.0) - 180.0
        # Clamp latitude to grid bounds
        lat_deg = float(np.clip(lat_deg, -90.0, 90.0))
        pt = np.array([[lat_deg, lon_deg]])
        vz = float(self._interp_zonal(pt).item())
        vm = float(self._interp_merid(pt).item())
        return (vz, vm)

    def to_netcdf(self, filepath: str, metadata: dict | None = None) -> None:
        """
        Save this wind map to a NetCDF4 file.

        File schema
        -----------
        Dimensions:
            lat : 180   (cell centres: -89.5, ..., +89.5 degrees)
            lon : 360   (cell centres: -179.5, ..., +179.5 degrees)
        Variables:
            lat      (lat,)      float32  degrees_north
            lon      (lon,)      float32  degrees_east   range -180 to +180
            v_zonal  (lat, lon)  float32  m/s  positive = eastward
            v_merid  (lat, lon)  float32  m/s  positive = northward
        Global attributes (all required):
            wind_map_type    : str
            created_utc      : str  ISO 8601 (auto-generated if absent)
            alt_km           : float
            description      : str
            pipeline_version : str

        Parameters
        ----------
        filepath : str — path to output .nc file
        metadata : dict | None — key-value pairs written as global attributes;
            must include wind_map_type, alt_km, description, pipeline_version.
            created_utc is auto-generated if not supplied.
        """
        import netCDF4 as nc

        md = dict(metadata) if metadata else {}

        # Auto-generate created_utc if not provided
        if "created_utc" not in md:
            md["created_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        with nc.Dataset(filepath, "w", format="NETCDF4") as ds:
            # Dimensions
            ds.createDimension("lat", 180)
            ds.createDimension("lon", 360)

            # Coordinate variables
            lat_var = ds.createVariable("lat", "f4", ("lat",))
            lat_var.units = "degrees_north"
            lat_var[:] = self.LAT_GRID.astype("f4")

            lon_var = ds.createVariable("lon", "f4", ("lon",))
            lon_var.units = "degrees_east"
            lon_var[:] = self.LON_GRID.astype("f4")

            # Wind variables
            vz_var = ds.createVariable("v_zonal", "f4", ("lat", "lon"))
            vz_var.units = "m/s"
            vz_var.long_name = "Zonal wind component (positive = eastward)"
            vz_var[:] = self.v_zonal_grid.astype("f4")

            vm_var = ds.createVariable("v_merid", "f4", ("lat", "lon"))
            vm_var.units = "m/s"
            vm_var.long_name = "Meridional wind component (positive = northward)"
            vm_var[:] = self.v_merid_grid.astype("f4")

            # Global attributes
            for key, val in md.items():
                setattr(ds, key, val)

    @classmethod
    def from_netcdf(cls, filepath: str) -> "GridWindMap":
        """
        Load a previously saved GridWindMap from a NetCDF4 file.

        Returns a GridWindMap instance with v_zonal_grid and v_merid_grid
        populated from the file and interpolators initialised.
        The returned instance has `wind_map_type` set from the file's
        global attribute of the same name.
        """
        import netCDF4 as nc

        obj = object.__new__(GridWindMap)
        with nc.Dataset(filepath, "r") as ds:
            obj.v_zonal_grid = np.array(ds.variables["v_zonal"][:], dtype=np.float32)
            obj.v_merid_grid = np.array(ds.variables["v_merid"][:], dtype=np.float32)
            obj.wind_map_type = getattr(ds, "wind_map_type", "unknown")
            # Copy all global attributes as instance attributes
            for attr in ds.ncattrs():
                setattr(obj, attr, getattr(ds, attr))
        obj._build_interpolators()
        return obj


# =============================================================================
# T1 — UniformWindMap
# =============================================================================

class UniformWindMap(WindMap):
    """
    Uniform wind field: identical v_zonal and v_merid at every location.

    Purpose
    -------
    Simplest possible pipeline sanity check. If the end-to-end pipeline
    cannot recover a spatially uniform 100 m/s zonal wind exactly,
    something is fundamentally wrong in the geometry or inversion.

    No external dependencies. No grid required.

    Parameters
    ----------
    v_zonal_ms : float
        Eastward wind, m/s, everywhere on Earth. Default 100.0.
    v_merid_ms : float
        Northward wind, m/s, everywhere on Earth. Default 0.0.

    Usage
    -----
    wind_map = UniformWindMap(v_zonal_ms=100.0, v_merid_ms=50.0)
    vz, vm = wind_map.sample(lat_deg=30.0, lon_deg=45.0)
    # Returns (100.0, 50.0) for any lat/lon
    """

    def __init__(self, v_zonal_ms: float = 100.0, v_merid_ms: float = 0.0):
        self.v_zonal_ms = float(v_zonal_ms)
        self.v_merid_ms = float(v_merid_ms)

    def sample(self, lat_deg: float, lon_deg: float) -> tuple[float, float]:
        return (self.v_zonal_ms, self.v_merid_ms)


# =============================================================================
# T2 — AnalyticWindMap
# =============================================================================

class AnalyticWindMap(GridWindMap):
    """
    Analytic wind field with known spatial structure.

    Purpose
    -------
    Tests that tangent-point sampling correctly reproduces spatial variation.
    Uses closed-form formulas — results are exact and fully reproducible
    with no external dependencies.

    Two patterns, selected by the `pattern` argument:

    Pattern 'sine_lat' (default):
        v_zonal(lat) = A_z * sin(lat_rad)
        v_merid(lat) = A_m * cos(lat_rad)
        with A_z = 200.0 m/s, A_m = 100.0 m/s (default)

    Pattern 'wave4' (DE3-like):
        v_zonal(lat, lon) = A_z * sin(lat_rad) * cos(4 * lon_rad + phi)
        v_merid(lat, lon) = A_m * cos(lat_rad) * sin(4 * lon_rad + phi)
        with A_z = 150.0 m/s, A_m = 75.0 m/s, phi = 0.0 rad (default)

    Parameters
    ----------
    pattern    : str   'sine_lat' or 'wave4'. Default 'sine_lat'.
    A_zonal_ms : float Zonal amplitude, m/s.
    A_merid_ms : float Meridional amplitude, m/s.
    phase_rad  : float Phase offset for wave4 pattern, radians. Default 0.0.
    alt_km     : float Altitude label for NetCDF output. Default 250.0.
    """

    def __init__(
        self,
        pattern: str = "sine_lat",
        A_zonal_ms: float = 200.0,
        A_merid_ms: float = 100.0,
        phase_rad: float = 0.0,
        alt_km: float = 250.0,
    ):
        if pattern not in ("sine_lat", "wave4"):
            raise ValueError(f"pattern must be 'sine_lat' or 'wave4', got '{pattern}'")

        self.pattern = pattern
        self.A_zonal_ms = float(A_zonal_ms)
        self.A_merid_ms = float(A_merid_ms)
        self.phase_rad = float(phase_rad)
        self.alt_km = float(alt_km)

        lat_rad = np.radians(self.LAT_GRID)    # shape (180,)
        lon_rad = np.radians(self.LON_GRID)    # shape (360,)

        if pattern == "sine_lat":
            # v_zonal(lat) = A_z * sin(lat_rad)  — same for all lon
            # v_merid(lat) = A_m * cos(lat_rad)  — same for all lon
            self.v_zonal_grid = (
                A_zonal_ms * np.sin(lat_rad)[:, np.newaxis] * np.ones((1, 360))
            ).astype(np.float32)
            self.v_merid_grid = (
                A_merid_ms * np.cos(lat_rad)[:, np.newaxis] * np.ones((1, 360))
            ).astype(np.float32)

        else:  # wave4
            LAT, LON = np.meshgrid(lat_rad, lon_rad, indexing="ij")   # (180, 360)
            self.v_zonal_grid = (
                A_zonal_ms * np.sin(LAT) * np.cos(4 * LON + phase_rad)
            ).astype(np.float32)
            self.v_merid_grid = (
                A_merid_ms * np.cos(LAT) * np.sin(4 * LON + phase_rad)
            ).astype(np.float32)

        self._build_interpolators()


# =============================================================================
# T3 — HWM14WindMap
# =============================================================================

class HWM14WindMap(GridWindMap):
    """
    Empirical quiet-time thermospheric wind map from HWM14.

    Requires the `hwm14` Python package (conda-forge: hwm14).
    Tests are skipped if hwm14 is not installed (pytest.importorskip).

    HWM14 reference: Drob et al. (2015), doi:10.1002/2014EA000089

    Parameters
    ----------
    alt_km      : float  Altitude for HWM14 query, km. Default TP_ALTITUDE_KM (250).
    day_of_year : int    Day of year (1–365). Default 172 (June 21).
    ut_hours    : float  Universal time in hours (0–24). Default 12.0.
    f107        : float  F10.7 solar flux index. Default 150.0.
    f107a       : float  81-day average F10.7. Default 150.0.
    ap          : float  Geomagnetic ap index. Default 4 (quiet, Kp ≈ 1).
    year        : int    Calendar year for iyd computation. Default 2027.
    """

    def __init__(
        self,
        alt_km: float = TP_ALTITUDE_KM,
        day_of_year: int = 172,
        ut_hours: float = 12.0,
        f107: float = 150.0,
        f107a: float = 150.0,
        ap: float = 4.0,
        year: int = 2027,
    ):
        import hwm14 as _hwm14

        self.alt_km = float(alt_km)
        self.day_of_year = int(day_of_year)
        self.ut_hours = float(ut_hours)
        self.f107 = float(f107)
        self.f107a = float(f107a)
        self.ap = float(ap)

        iyd = year * 1000 + day_of_year       # YYYYDDD integer
        sec = ut_hours * 3600.0               # seconds since midnight UTC
        ap_array = [ap] * 7

        vz_grid = np.empty((180, 360), dtype=np.float32)
        vm_grid = np.empty((180, 360), dtype=np.float32)

        for i, lat in enumerate(self.LAT_GRID):
            for j, lon in enumerate(self.LON_GRID):
                result = _hwm14.hwm14(
                    iyd, sec, alt_km, lat, lon, -1, f107a, f107, 0, ap_array
                )
                vm_grid[i, j] = result[0]   # HWM14 first output = meridional (northward)
                vz_grid[i, j] = result[1]   # HWM14 second output = zonal (eastward)

        self.v_zonal_grid = vz_grid
        self.v_merid_grid = vm_grid
        self._build_interpolators()


# =============================================================================
# T4 — StormWindMap
# =============================================================================

class StormWindMap(GridWindMap):
    """
    Storm-perturbed thermospheric wind map for geomagnetic storm simulation.

    Backend A — HWM14 + DWM07 (default): set ap=80 (Kp ≈ 6, G2 storm).
    HWM14 internally calls DWM07 for the disturbance wind component.

    Requires the `hwm14` Python package.

    Parameters
    ----------
    alt_km      : float  Altitude, km. Default TP_ALTITUDE_KM (250).
    day_of_year : int    Day of year. Default 355 (December 21).
    ut_hours    : float  UT hours. Default 3.0 (post-midnight, storm peak).
    f107        : float  F10.7 index. Default 180.0 (elevated activity).
    f107a       : float  81-day average F10.7. Default 180.0.
    ap          : float  ap index. Default 80 (Kp ≈ 6, G2 storm).
    year        : int    Calendar year. Default 2027.
    """

    def __init__(
        self,
        alt_km: float = TP_ALTITUDE_KM,
        day_of_year: int = 355,
        ut_hours: float = 3.0,
        f107: float = 180.0,
        f107a: float = 180.0,
        ap: float = 80.0,
        year: int = 2027,
    ):
        # Delegate to HWM14WindMap initialisation with storm parameters
        _hwm = HWM14WindMap(
            alt_km=alt_km,
            day_of_year=day_of_year,
            ut_hours=ut_hours,
            f107=f107,
            f107a=f107a,
            ap=ap,
            year=year,
        )
        self.alt_km = _hwm.alt_km
        self.v_zonal_grid = _hwm.v_zonal_grid
        self.v_merid_grid = _hwm.v_merid_grid
        self._build_interpolators()
