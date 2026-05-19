"""
tests/test_l02_cal_scheduler.py — Verification tests for cal_scheduler (S_L02).

Spec:  specs/S_L02_cal_scheduler_2026-05-14.md  §5  (T1–T5)
"""

import pathlib

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frame_record(img_type, lua_timestamp, orbit_number=0):
    """Return a FrameRecord with dummy path and no metadata."""
    from windcube.cal_scheduler import FrameRecord
    return FrameRecord(
        path=pathlib.Path(f"fake_{img_type}_{lua_timestamp}.bin"),
        img_type=img_type,
        lua_timestamp=lua_timestamp,
        orbit_number=orbit_number,
        obs_mode="unknown",
        meta=None,
    )


def _make_orbit_group(n_science, orbit_number):
    """Return an OrbitGroup with n_science science frames and empty dark/cal."""
    from windcube.cal_scheduler import OrbitGroup
    science_frames = [
        _make_frame_record("science", i, orbit_number)
        for i in range(n_science)
    ]
    return OrbitGroup(
        orbit_number=orbit_number,
        dark_frames=[],
        cal_frames=[],
        science_frames=science_frames,
    )


# ---------------------------------------------------------------------------
# T1 — scan_directory raises on missing directory
# ---------------------------------------------------------------------------

def test_scan_directory_missing():
    """scan_directory raises FileNotFoundError for a non-existent path."""
    from windcube.cal_scheduler import scan_directory
    with pytest.raises(FileNotFoundError):
        scan_directory("/nonexistent/path")


# ---------------------------------------------------------------------------
# T2 — assign_orbits groups correctly
# ---------------------------------------------------------------------------

def test_assign_orbits_groups_by_period():
    """Frames separated by >= ORBIT_PERIOD_MIN go into different orbits."""
    from windcube.cal_scheduler import assign_orbits

    PERIOD_MS = int(95 * 60 * 1000)   # 5_700_000 ms
    base_ts   = 1_800_000_000_000

    records = [
        _make_frame_record("science", base_ts + 0,          0),
        _make_frame_record("science", base_ts + 60_000,     0),   # same orbit
        _make_frame_record("science", base_ts + PERIOD_MS,  0),   # new orbit
    ]
    groups = assign_orbits(records)
    assert len(groups) == 2
    assert len(groups[0].science_frames) == 2
    assert len(groups[1].science_frames) == 1


# ---------------------------------------------------------------------------
# T3 — build_master_dark returns None for empty list
# ---------------------------------------------------------------------------

def test_build_master_dark_empty():
    """build_master_dark([]) returns None."""
    from windcube.cal_scheduler import build_master_dark
    result = build_master_dark([])
    assert result is None


# ---------------------------------------------------------------------------
# T4 — build_orbit_schedule raises ValueError on empty directory
# ---------------------------------------------------------------------------

def test_build_orbit_schedule_scan_only(tmp_path):
    """
    With process_cals=False, scan_directory still runs.
    A tmp directory with zero .bin files must raise ValueError.
    """
    from windcube.cal_scheduler import build_orbit_schedule
    with pytest.raises(ValueError):
        build_orbit_schedule(tmp_path, process_cals=False)


# ---------------------------------------------------------------------------
# T5 — iter_science_frames yields all science frames
# ---------------------------------------------------------------------------

def test_iter_science_frames():
    """iter_science_frames yields (FrameRecord, OrbitGroup) for every science frame."""
    from windcube.cal_scheduler import OrbitSchedule

    og1 = _make_orbit_group(n_science=2, orbit_number=0)
    og2 = _make_orbit_group(n_science=1, orbit_number=1)
    schedule = OrbitSchedule(
        input_dir=pathlib.Path("."),
        orbits=[og1, og2],
        n_orbits=2,
        n_science=3,
        n_cal_missing=0,
        t_epoch_ms=0,
    )
    frames = list(schedule.iter_science_frames())
    assert len(frames) == 3
    assert all(fr.img_type == "science" for fr, _ in frames)
