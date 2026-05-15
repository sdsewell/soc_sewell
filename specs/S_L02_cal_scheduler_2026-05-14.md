# S_L02 — Calibration Scheduler
## WindCube SOC Pipeline — Specification v1.0
**Spec ID:** S_L02  
**Spec file:** `specs/S_L02_cal_scheduler_2026-05-14.md`  
**Date:** 2026-05-14  
**Status:** Authoritative  
**Module:** `windcube/cal_scheduler.py`  
**Architecture ref:** `docs/WINDCUBE_ARCH_01_pipeline_synthesis_2026-05-14.md`

**Depends on:**
- `windcube/fpi_pipeline.py` (S_L01, a4101f9) — `process_cal_frame()`,
  `average_calibrations()`, `CalibrationResult`, `MasterCalibration`
- `src/metadata/p01_image_metadata_2026_04_06.py` — `ingest_real_image()`
- `windcube/constants.py` — `ORBIT_PERIOD_MIN`, `PASSES_PER_DAY`

**Consumed by:**
- `scripts/invert_wind_map.py` (Step 4, S_batch_v2)

---

## 1. Purpose and scope

This module answers one question: **for each science frame, which dark
frames and calibration frames should be used to process it?**

Given a directory of WindCube `.bin` files (dark, cal, and science), the
scheduler groups frames by orbit, builds a master dark and master
calibration per orbit, and returns an `OrbitSchedule` that maps every
science frame to its calibration products.

**What this spec does NOT do:**
- Does not implement any FPI physics
- Does not read pixel data (metadata only, except when building master darks)
- Does not write any files to disk
- Does not interpolate calibrations between orbits (deferred per ARCH-01 §3.2)

---

## 2. Orbit grouping strategy

### 2.1 Orbit assignment from timestamps

Frames are assigned to orbits using their `lua_timestamp` (Unix ms).
The orbit period is `ORBIT_PERIOD_MIN` = 95.0 minutes = 5,700,000 ms.

Orbit number for a frame is:
```python
orbit_number = int(
    (meta.lua_timestamp - t_epoch_ms) / (ORBIT_PERIOD_MIN * 60_000)
)
```

where `t_epoch_ms` is the `lua_timestamp` of the **earliest frame** in
the dataset (the reference epoch). This gives orbit numbers 0, 1, 2, ...
relative to the dataset start.

If `meta.orbit_number` is already set (not None) in the metadata, use
that value directly instead of computing from timestamp.

### 2.2 Image type classification

Use `meta.img_type` from `ImageMetadata`:
- `"dark"` → dark frame
- `"cal"` → calibration frame  
- `"science"` → science frame

### 2.3 Expected per-orbit composition

Nominal: 5 dark frames + 5 cal frames + N science frames per orbit,
where N varies (~30–80 depending on the orbit's observation window).

The scheduler must be robust to non-nominal compositions:
- Fewer than 5 darks: use all available; log warning if < 2
- Fewer than 5 cals: use all available; log warning if < 2; 
  if 0 cals, flag the orbit as `cal_missing=True`
- 0 darks: skip dark subtraction for that orbit; log warning
- 0 science frames: skip the orbit silently

---

## 3. Data structures

### 3.1 FrameRecord

Lightweight metadata container for one `.bin` file:

```python
@dataclass
class FrameRecord:
    """Metadata for one .bin file, used by the scheduler."""
    path:          pathlib.Path
    img_type:      str           # 'dark', 'cal', 'science'
    lua_timestamp: int           # Unix ms
    orbit_number:  int           # assigned by scheduler
    obs_mode:      str           # 'along_track', 'cross_track', 'unknown'
    meta:          object        # ImageMetadata (kept for downstream use)
```

### 3.2 OrbitGroup

All frames and calibration products for one orbit:

```python
@dataclass
class OrbitGroup:
    """All frames and calibration products for one orbit."""
    orbit_number:   int

    # Raw frame lists (sorted by lua_timestamp)
    dark_frames:    list[FrameRecord]
    cal_frames:     list[FrameRecord]
    science_frames: list[FrameRecord]

    # Calibration products (None until build() is called)
    master_dark:    np.ndarray | None     # shape (H, W), float32; None if no darks
    cal_results:    list[CalibrationResult] | None  # one per cal frame
    master_cal:     MasterCalibration | None        # averaged result

    # Quality flags
    cal_missing:    bool = False    # True if no cal frames available
    dark_missing:   bool = False    # True if no dark frames available
    n_cal_failed:   int  = 0        # cal frames where converged=False
```

### 3.3 OrbitSchedule

The complete schedule for all orbits in a dataset:

```python
@dataclass
class OrbitSchedule:
    """
    Complete calibration schedule for one dataset directory.

    Produced by build_orbit_schedule().
    Consumed by invert_wind_map.py to process science frames.
    """
    input_dir:     pathlib.Path
    orbits:        list[OrbitGroup]       # sorted by orbit_number
    n_orbits:      int
    n_science:     int                    # total science frames
    n_cal_missing: int                    # orbits with cal_missing=True
    t_epoch_ms:    int                    # lua_timestamp of earliest frame

    def get_orbit(self, orbit_number: int) -> OrbitGroup | None:
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
```

---

## 4. Public functions

### 4.1 `scan_directory()`

```python
def scan_directory(
    input_dir: str | pathlib.Path,
    h_target_km_obs: float = 250.0,
) -> list[FrameRecord]:
    """
    Scan a directory for *_dark.bin, *_cal_*.bin, and *_science_*.bin files.

    Calls ingest_real_image() on each file to read metadata.
    Does NOT read pixel data — metadata only.

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
```

**Implementation note:** GEN01 filename convention:
- `YYYY-MM-DDTHH-MM-SSZ_dark.bin`
- `YYYY-MM-DDTHH-MM-SSZ_cal_NNNN.bin`  (or `*_cal_*.bin`)
- `YYYY-MM-DDTHH-MM-SSZ_science.bin`

The img_type is read from `meta.img_type` (set by P01 ingest from lamp
and shutter state), not from the filename. The filename pattern is only
used to glob for `.bin` files.

Print progress: `"Scanning: {n} .bin files found in {input_dir}"`

### 4.2 `assign_orbits()`

```python
def assign_orbits(
    records: list[FrameRecord],
) -> list[OrbitGroup]:
    """
    Assign FrameRecords to orbits and return a list of OrbitGroup objects.

    Uses lua_timestamp and ORBIT_PERIOD_MIN to compute orbit numbers.
    If meta.orbit_number is already set, use it directly.

    Returns list[OrbitGroup] sorted by orbit_number.
    """
```

### 4.3 `build_master_dark()`

```python
def build_master_dark(
    dark_records: list[FrameRecord],
    h_target_km_obs: float = 250.0,
) -> np.ndarray | None:
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
    np.ndarray, shape (259, 276), float32  — pixel sum of all dark frames.
    None if dark_records is empty.

    Notes
    -----
    The master dark is the SUM (not mean) of all dark frames. When
    subtracting from a single science or cal frame, divide by
    len(dark_records) to get the per-frame equivalent:
        dark_subtracted = pixels - master_dark / n_dark_frames
    This preserves correct Poisson statistics.
    """
```

### 4.4 `build_master_cal()`

```python
def build_master_cal(
    cal_records: list[FrameRecord],
    master_dark: np.ndarray | None,
    r_max_px: float = 110.0,
    h_target_km_obs: float = 250.0,
) -> tuple[list[CalibrationResult], MasterCalibration | None]:
    """
    Process all calibration frames and average to a master calibration.

    For each cal frame:
      1. Load pixel data via ingest_real_image()
      2. Dark-subtract: pixels_ds = pixels - master_dark / n_dark
         (if master_dark is None, skip dark subtraction with warning)
      3. Call process_cal_frame(pixels_ds, r_max_px=r_max_px)
         → CalibrationResult
      4. If process_cal_frame raises, log warning, skip frame,
         increment n_cal_failed

    Then: average_calibrations(successful_results) → MasterCalibration

    Returns
    -------
    (cal_results, master_cal)
        cal_results  : list of CalibrationResult (one per successful frame)
        master_cal   : MasterCalibration, or None if all frames failed

    Raises
    ------
    Does NOT raise. All failures are logged as warnings and skipped.
    """
```

### 4.5 `build_orbit_schedule()` — top-level entry point

```python
def build_orbit_schedule(
    input_dir: str | pathlib.Path,
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

    Prints progress:
        "Found N science, M cal, K dark frames across O orbits"
        "Orbit {n}: building master dark ({K} frames)..."
        "Orbit {n}: processing {M} cal frames..."
        "Orbit {n}: master cal ready  chi2={x:.2f}  converged={j}/{M}"
        "Schedule complete: O orbits, N science frames ready"
    """
```

**Implementation:**

```python
records  = scan_directory(input_dir, h_target_km_obs)
groups   = assign_orbits(records)
t_epoch  = records[0].lua_timestamp if records else 0

for og in groups:
    if not process_cals:
        continue

    # Master dark
    og.master_dark = build_master_dark(og.dark_frames, h_target_km_obs)
    if og.master_dark is None:
        og.dark_missing = True

    # Master calibration
    if og.cal_frames:
        og.cal_results, og.master_cal = build_master_cal(
            og.cal_frames, og.master_dark, r_max_px, h_target_km_obs)
        og.n_cal_failed = sum(1 for c in (og.cal_results or [])
                              if not c.converged)
        if og.master_cal is None:
            og.cal_missing = True
    else:
        og.cal_missing = True

return OrbitSchedule(
    input_dir     = pathlib.Path(input_dir),
    orbits        = groups,
    n_orbits      = len(groups),
    n_science     = sum(len(og.science_frames) for og in groups),
    n_cal_missing = sum(og.cal_missing for og in groups),
    t_epoch_ms    = t_epoch,
)
```

---

## 5. Verification tests

### T1 — scan_directory raises on missing directory

```python
def test_scan_directory_missing():
    import pytest
    from windcube.cal_scheduler import scan_directory
    with pytest.raises(FileNotFoundError):
        scan_directory("/nonexistent/path")
```

### T2 — assign_orbits groups correctly

```python
def test_assign_orbits_groups_by_period():
    """Frames separated by > ORBIT_PERIOD_MIN go into different orbits."""
    from windcube.cal_scheduler import assign_orbits, FrameRecord
    import pathlib

    PERIOD_MS = int(95 * 60 * 1000)  # 95 min in ms
    base_ts   = 1_800_000_000_000

    records = [
        _make_frame_record('science', base_ts + 0,          0),
        _make_frame_record('science', base_ts + 60_000,     0),   # same orbit
        _make_frame_record('science', base_ts + PERIOD_MS,  0),   # new orbit
    ]
    groups = assign_orbits(records)
    assert len(groups) == 2
    assert len(groups[0].science_frames) == 2
    assert len(groups[1].science_frames) == 1
```

### T3 — build_master_dark returns None for empty list

```python
def test_build_master_dark_empty():
    from windcube.cal_scheduler import build_master_dark
    result = build_master_dark([])
    assert result is None
```

### T4 — build_orbit_schedule scans real GEN01 output (skip if no data)

```python
def test_build_orbit_schedule_scan_only(tmp_path):
    """
    With process_cals=False, scan_directory runs without H05.
    Uses a temp directory with zero .bin files — expects ValueError.
    """
    import pytest
    from windcube.cal_scheduler import build_orbit_schedule
    with pytest.raises(ValueError):
        build_orbit_schedule(tmp_path, process_cals=False)
```

### T5 — iter_science_frames yields all science frames

```python
def test_iter_science_frames():
    from windcube.cal_scheduler import OrbitSchedule, OrbitGroup, FrameRecord
    import pathlib

    # Build a minimal OrbitSchedule with 2 orbits, 3 science frames total
    og1 = _make_orbit_group(n_science=2, orbit_number=0)
    og2 = _make_orbit_group(n_science=1, orbit_number=1)
    schedule = OrbitSchedule(
        input_dir=pathlib.Path('.'), orbits=[og1, og2],
        n_orbits=2, n_science=3, n_cal_missing=0, t_epoch_ms=0,
    )
    frames = list(schedule.iter_science_frames())
    assert len(frames) == 3
    assert all(fr.img_type == 'science' for fr, _ in frames)
```

---

## 6. File location

```
soc_sewell/
├── windcube/
│   ├── cal_scheduler.py          ← new module (this spec)
│   └── fpi_pipeline.py           ← existing (S_L01)
└── tests/
    └── test_l02_cal_scheduler.py ← new test file (T1–T5)
```

---

## 7. Instructions for Claude Code

Read this entire spec, S_L01, and WINDCUBE-ARCH-01 before writing any code.

**Step-by-step:**

1. Create `windcube/cal_scheduler.py` with module docstring:
   ```
   Spec: specs/S_L02_cal_scheduler_2026-05-14.md
   Spec date: 2026-05-14
   ```
   Implement in order: `FrameRecord`, `OrbitGroup`, `OrbitSchedule`,
   `scan_directory`, `assign_orbits`, `build_master_dark`,
   `build_master_cal`, `build_orbit_schedule`.

2. Create `tests/test_l02_cal_scheduler.py` with helpers
   `_make_frame_record()` and `_make_orbit_group()`, then T1–T5.

3. Run: `pytest tests/test_l02_cal_scheduler.py -v`
   All 5 tests must pass.

4. Run full suite: `pytest tests/ -v` — no regressions.

5. Commit:
   ```
   feat(l02): add windcube/cal_scheduler.py calibration scheduler
   Implements: S_L02_cal_scheduler_2026-05-14.md
   OrbitSchedule, OrbitGroup, build_orbit_schedule().
   5/5 tests pass.
   ```

**Report back with:**
- T1–T5 pass/fail
- Full pytest result
- Git commit hash

---

*End of S_L02 specification v1.0 — 2026-05-14*
