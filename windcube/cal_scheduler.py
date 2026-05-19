"""
windcube/cal_scheduler.py — Per-orbit calibration scheduler.

Spec:        specs/S_L02_cal_scheduler_2026-05-14.md
Spec date:   2026-05-14
Generated:   2026-05-15
Tool:        Claude Code

Groups WindCube .bin files by orbit, builds master darks and master
calibrations per orbit, and returns an OrbitSchedule for use by
invert_wind_map.py.

Does NOT write any files to disk.
Does NOT contain interactive (tkinter/matplotlib) code.
"""

from __future__ import annotations

import logging
import pathlib
import sys
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# sys.path setup — same REPO_ROOT pattern as fpi_pipeline.py
# ---------------------------------------------------------------------------
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Imports from dependencies
# ---------------------------------------------------------------------------
from windcube.fpi_pipeline import (                                  # noqa: E402
    process_cal_frame,
    average_calibrations,
    CalibrationResult,
    MasterCalibration,
)
from windcube.constants import ORBIT_PERIOD_MIN                      # noqa: E402
from src.metadata.p01_image_metadata_2026_04_06 import (            # noqa: E402
    ingest_real_image,
)

log = logging.getLogger("cal_scheduler")


# ---------------------------------------------------------------------------
# §3.1 FrameRecord
# ---------------------------------------------------------------------------

@dataclass
class FrameRecord:
    """Metadata for one .bin file, used by the scheduler."""
    path:          pathlib.Path
    img_type:      str           # 'dark', 'cal', 'science'
    lua_timestamp: int           # Unix ms
    orbit_number:  int           # assigned by scheduler
    obs_mode:      str           # 'along_track', 'cross_track', 'unknown'
    meta:          object        # ImageMetadata (kept for downstream use)


# ---------------------------------------------------------------------------
# §3.2 OrbitGroup
# ---------------------------------------------------------------------------

@dataclass
class OrbitGroup:
    """All frames and calibration products for one orbit."""
    orbit_number:   int

    # Raw frame lists (sorted by lua_timestamp)
    dark_frames:    list
    cal_frames:     list
    science_frames: list

    # Calibration products (None until build() is called)
    master_dark:    object = field(default=None)   # np.ndarray | None
    cal_results:    object = field(default=None)   # list[CalibrationResult] | None
    master_cal:     object = field(default=None)   # MasterCalibration | None

    # Quality flags
    cal_missing:    bool = field(default=False)
    dark_missing:   bool = field(default=False)
    n_cal_failed:   int  = field(default=0)


# ---------------------------------------------------------------------------
# §3.3 OrbitSchedule
# ---------------------------------------------------------------------------

@dataclass
class OrbitSchedule:
    """
    Complete calibration schedule for one dataset directory.

    Produced by build_orbit_schedule().
    Consumed by invert_wind_map.py to process science frames.
    """
    input_dir:     pathlib.Path
    orbits:        list       # list[OrbitGroup] sorted by orbit_number
    n_orbits:      int
    n_science:     int        # total science frames
    n_cal_missing: int        # orbits with cal_missing=True
    t_epoch_ms:    int        # lua_timestamp of earliest frame

    def get_orbit(self, orbit_number: int):
        """Return the OrbitGroup for a given orbit number, or None."""
        for og in self.orbits:
            if og.orbit_number == orbit_number:
                return og
        return None

    def iter_science_frames(self):
        """
        Yield (FrameRecord, OrbitGroup) for every science frame
        in all orbits, in timestamp order.
        """
        for og in self.orbits:
            for fr in og.science_frames:
                yield fr, og


# ---------------------------------------------------------------------------
# §4.1 scan_directory
# ---------------------------------------------------------------------------

def scan_directory(
    input_dir,
    h_target_km_obs: float = 250.0,
) -> list:
    """
    Scan a directory for *.bin files and build FrameRecords.

    Parameters
    ----------
    input_dir : str or Path
        Directory to scan (non-recursive).
    h_target_km_obs : float
        Passed to ingest_real_image() as h_target_km_obs. Default 250.0 km.

    Returns
    -------
    list[FrameRecord], sorted by lua_timestamp ascending.

    Raises
    ------
    FileNotFoundError : If input_dir does not exist.
    ValueError : If no .bin files found.
    """
    input_dir = pathlib.Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Directory not found: {input_dir}")

    bin_files = sorted(input_dir.glob("*.bin"))
    n = len(bin_files)
    print(f"Scanning: {n} .bin files found in {input_dir}")

    if n == 0:
        raise ValueError(f"No .bin files found in {input_dir}")

    records = []
    for path in bin_files:
        meta, _pixels = ingest_real_image(path, h_target_km_obs=h_target_km_obs)
        orbit_num = meta.orbit_number if meta.orbit_number is not None else 0
        rec = FrameRecord(
            path=path,
            img_type=meta.img_type,
            lua_timestamp=meta.lua_timestamp,
            orbit_number=orbit_num,
            obs_mode=meta.obs_mode,
            meta=meta,
        )
        records.append(rec)

    records.sort(key=lambda r: r.lua_timestamp)
    return records


# ---------------------------------------------------------------------------
# §4.2 assign_orbits
# ---------------------------------------------------------------------------

def assign_orbits(records: list) -> list:
    """
    Assign FrameRecords to orbits and return a list of OrbitGroup objects.

    Uses lua_timestamp and ORBIT_PERIOD_MIN to compute orbit numbers.
    If meta.orbit_number is already set, use it directly.

    Returns list[OrbitGroup] sorted by orbit_number.
    """
    if not records:
        return []

    t_epoch = min(r.lua_timestamp for r in records)
    orbit_period_ms = ORBIT_PERIOD_MIN * 60_000  # ms

    for r in records:
        meta_orbit = None
        if r.meta is not None:
            meta_orbit = r.meta.orbit_number

        if meta_orbit is not None and meta_orbit != -1 and meta_orbit != 0:
            orbit_num = meta_orbit
        else:
            orbit_num = int((r.lua_timestamp - t_epoch) / orbit_period_ms)

        r.orbit_number = orbit_num

    # Group records by orbit_number into OrbitGroup objects
    groups_dict: dict = {}
    for r in records:
        on = r.orbit_number
        if on not in groups_dict:
            groups_dict[on] = {"dark": [], "cal": [], "science": []}
        if r.img_type == "dark":
            groups_dict[on]["dark"].append(r)
        elif r.img_type == "cal":
            groups_dict[on]["cal"].append(r)
        else:
            groups_dict[on]["science"].append(r)

    groups = []
    for on in sorted(groups_dict.keys()):
        g = groups_dict[on]
        dark_frames    = sorted(g["dark"],    key=lambda r: r.lua_timestamp)
        cal_frames     = sorted(g["cal"],     key=lambda r: r.lua_timestamp)
        science_frames = sorted(g["science"], key=lambda r: r.lua_timestamp)
        groups.append(OrbitGroup(
            orbit_number=on,
            dark_frames=dark_frames,
            cal_frames=cal_frames,
            science_frames=science_frames,
        ))

    return groups


# ---------------------------------------------------------------------------
# §4.3 build_master_dark
# ---------------------------------------------------------------------------

def build_master_dark(
    dark_records: list,
    h_target_km_obs: float = 250.0,
):
    """
    Sum dark frame pixels to produce a master dark.

    Parameters
    ----------
    dark_records : list[FrameRecord]
        Dark frames for this orbit. If empty, returns None.
    h_target_km_obs : float
        Passed to ingest_real_image().

    Returns
    -------
    np.ndarray, shape (259, 276), float32 — pixel sum of all dark frames.
    None if dark_records is empty.

    Notes
    -----
    The master dark is the SUM (not mean) of all dark frames. When
    subtracting from a single science or cal frame, divide by
    len(dark_records) to get the per-frame equivalent:
        dark_subtracted = pixels - master_dark / n_dark_frames
    This preserves correct Poisson statistics.
    """
    if not dark_records:
        return None

    n = len(dark_records)
    master = None
    for record in dark_records:
        _, pixels = ingest_real_image(record.path, h_target_km_obs=h_target_km_obs)
        if master is None:
            master = pixels.astype(np.float32)
        else:
            master += pixels.astype(np.float32)

    log.info("Master dark: summed %d frames", n)
    return master


# ---------------------------------------------------------------------------
# §4.4 build_master_cal
# ---------------------------------------------------------------------------

def build_master_cal(
    cal_records: list,
    master_dark,
    n_dark_frames: int = 0,
    r_max_px: float = 110.0,
    h_target_km_obs: float = 250.0,
    orbit_number: int = -1,
) -> tuple:
    """
    Process all calibration frames and average to a master calibration.

    Parameters
    ----------
    cal_records : list[FrameRecord]
        Calibration frames for this orbit. If empty, returns ([], None).
    master_dark : np.ndarray or None
        Master dark (sum of dark frames). None to skip dark subtraction.
    n_dark_frames : int
        Number of dark frames summed in master_dark. 0 to skip subtraction.
    r_max_px : float
        Outer fringe radius passed to process_cal_frame(). Default 110.0.
    h_target_km_obs : float
        Passed to ingest_real_image(). Default 250.0 km.
    orbit_number : int
        Passed to average_calibrations() for provenance. Default -1.

    Returns
    -------
    (cal_results, master_cal)
        cal_results  : list of CalibrationResult (one per successful frame)
        master_cal   : MasterCalibration, or None if all frames failed

    Does NOT raise. All failures are logged as warnings and skipped.
    """
    if not cal_records:
        return ([], None)

    N = len(cal_records)
    successful = []

    for i, record in enumerate(cal_records):
        try:
            _, pixels = ingest_real_image(record.path, h_target_km_obs=h_target_km_obs)
            if master_dark is not None and n_dark_frames > 0:
                pixels_ds = pixels.astype(np.float32) - master_dark / n_dark_frames
            else:
                pixels_ds = pixels.astype(np.float32)
                if master_dark is None:
                    log.warning("No master dark — processing cal without subtraction")
            result = process_cal_frame(pixels_ds, r_max_px=r_max_px)
            successful.append(result)
            log.info(
                "  Cal frame %d/%d: chi2=%.3f  converged=%s",
                i + 1, N, result.chi2_red, result.converged,
            )
        except Exception as exc:
            log.warning("  Cal frame %d/%d FAILED: %s", i + 1, N, exc)
            continue

    if not successful:
        return ([], None)

    mc = average_calibrations(successful, orbit_number=orbit_number)
    return (successful, mc)


# ---------------------------------------------------------------------------
# §4.5 build_orbit_schedule — top-level entry point
# ---------------------------------------------------------------------------

def build_orbit_schedule(
    input_dir,
    r_max_px: float = 110.0,
    h_target_km_obs: float = 250.0,
    process_cals: bool = True,
) -> OrbitSchedule:
    """
    Scan a directory, group frames by orbit, and build all calibration
    products. Returns a complete OrbitSchedule ready for use by
    invert_wind_map.py.

    Parameters
    ----------
    input_dir : str or Path
        Directory containing *_dark.bin, *_cal_*.bin, *_science_*.bin.
    r_max_px : float
        Outer fringe radius for cal and science processing. Default 110.0.
    h_target_km_obs : float
        Emission layer altitude. Default 250.0 km.
    process_cals : bool
        If True (default), build master darks and master calibrations.
        If False, scan and group only — calibration products are None.
        Set False for fast directory inspection without H05 processing.

    Returns
    -------
    OrbitSchedule
    """
    records = scan_directory(input_dir, h_target_km_obs)
    groups  = assign_orbits(records)
    t_epoch = records[0].lua_timestamp if records else 0

    n_science_total = sum(len(og.science_frames) for og in groups)
    n_cal_total     = sum(len(og.cal_frames)     for og in groups)
    n_dark_total    = sum(len(og.dark_frames)    for og in groups)
    print(
        f"Found {n_science_total} science, {n_cal_total} cal, "
        f"{n_dark_total} dark frames across {len(groups)} orbits"
    )

    for og in groups:
        if not process_cals:
            continue

        # Master dark
        print(
            f"Orbit {og.orbit_number}: building master dark "
            f"({len(og.dark_frames)} frames)..."
        )
        og.master_dark = build_master_dark(og.dark_frames, h_target_km_obs)
        if og.master_dark is None:
            og.dark_missing = True
            log.warning("Orbit %d: no dark frames", og.orbit_number)

        # Master calibration
        if og.cal_frames:
            print(
                f"Orbit {og.orbit_number}: processing "
                f"{len(og.cal_frames)} cal frames..."
            )
            og.cal_results, og.master_cal = build_master_cal(
                og.cal_frames,
                og.master_dark,
                n_dark_frames=len(og.dark_frames),
                r_max_px=r_max_px,
                h_target_km_obs=h_target_km_obs,
                orbit_number=og.orbit_number,
            )
            og.n_cal_failed = sum(
                1 for c in (og.cal_results or []) if not c.converged
            )
            if og.master_cal is None:
                og.cal_missing = True
            else:
                n_converged = sum(c.converged for c in (og.cal_results or []))
                n_cal = len(og.cal_frames)
                print(
                    f"Orbit {og.orbit_number}: master cal ready  "
                    f"chi2={og.master_cal.chi2_red_mean:.2f}  "
                    f"converged={n_converged}/{n_cal}"
                )
        else:
            og.cal_missing = True

    print(
        f"Schedule complete: {len(groups)} orbits, "
        f"{n_science_total} science frames ready"
    )

    return OrbitSchedule(
        input_dir     = pathlib.Path(input_dir),
        orbits        = groups,
        n_orbits      = len(groups),
        n_science     = n_science_total,
        n_cal_missing = sum(og.cal_missing for og in groups),
        t_epoch_ms    = t_epoch,
    )
