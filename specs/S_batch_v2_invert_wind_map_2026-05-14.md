# S_batch_v2 — invert_wind_map.py: Wire H06 Full Pipeline
## WindCube SOC Pipeline — Specification v1.0
**Spec ID:** S_batch_v2  
**Spec file:** `specs/S_batch_v2_invert_wind_map_2026-05-14.md`  
**Date:** 2026-05-14  
**Status:** Authoritative  
**Script:** `scripts/invert_wind_map.py`  
**Architecture ref:** `docs/WINDCUBE_ARCH_01_pipeline_synthesis_2026-05-14.md`

**Depends on:**
- `windcube/cal_scheduler.py` (S_L02, 099fef6) — `build_orbit_schedule()`
- `windcube/fpi_pipeline.py` (S_L01, a4101f9) — `process_science_frame()`
- `windcube/wind_retrieval.py` (H07, 185ff08) — `compute_los_geometry()`

---

## 1. Purpose and scope

This spec adds a **second v_rel path** to `invert_wind_map.py` — the full
H06 fringe fitting pipeline — while fully preserving the existing
`--v-rel-csv` path for synthetic data validation.

**What changes:**
- New `--use-h06` flag (default: auto — active when `--v-rel-csv` not supplied)
- New `_build_schedule()` helper that calls `build_orbit_schedule()`
- Updated `_process_one()` to accept a `OrbitGroup` and call
  `process_science_frame()` when in H06 mode
- `v_los_prior_ms` computation added (H07 geometry before H06)

**What does NOT change:**
- `--v-rel-csv` path is preserved 100% — all existing validation runs
  continue to work identically
- All CLI arguments unchanged (one new optional flag added)
- All output writers (`_write_results_csv`, `_write_summary`) unchanged
- All binning and inversion logic unchanged
- All progress printing unchanged

---

## 2. New CLI argument

Add one new optional argument to `_parse_args()`:

```python
parser.add_argument(
    "--use-h06",
    action="store_true",
    default=False,
    help=(
        "Use the full H06 fringe fitting pipeline to recover v_rel from "
        "raw pixels. Activated automatically when --v-rel-csv is not "
        "supplied. Requires cal and dark frames in the input folder."
    ),
)
```

**Activation logic** (in `main()`, after arg parsing):

```python
use_h06 = args.use_h06 or (args.v_rel_csv is None)
```

When `use_h06=True` and `args.v_rel_csv` is also supplied, the CSV path
takes precedence and `use_h06` is set to False with a printed note:
```python
if use_h06 and args.v_rel_csv:
    print("NOTE: --v-rel-csv supplied — using CSV path, ignoring --use-h06")
    use_h06 = False
```

---

## 3. Schedule building (H06 mode only)

Add a new helper `_build_schedule()`:

```python
def _build_schedule(input_folder: Path, args: argparse.Namespace):
    """
    Build the per-orbit calibration schedule using cal_scheduler.

    Called once at startup in H06 mode. Processes all dark and cal
    frames in input_folder. Returns OrbitSchedule.

    Prints progress from build_orbit_schedule() directly.
    On failure, prints error and calls sys.exit(1).
    """
    from windcube.cal_scheduler import build_orbit_schedule
    try:
        schedule = build_orbit_schedule(
            input_folder,
            r_max_px        = 110.0,
            h_target_km_obs = args.h_target_km,
            process_cals    = True,
        )
    except Exception as exc:
        print(f"ERROR: Failed to build calibration schedule: {exc}",
              file=sys.stderr)
        sys.exit(1)

    # Warn about orbits with missing calibration
    if schedule.n_cal_missing > 0:
        print(
            f"WARNING: {schedule.n_cal_missing}/{schedule.n_orbits} orbits "
            "have no master calibration — science frames in those orbits "
            "will be skipped."
        )
    return schedule
```

---

## 4. Updated `_process_one()` signature and H06 path

### 4.1 New signature

```python
def _process_one(
    bin_path: Path,
    vrel_lookup,                    # dict or None (existing)
    args: argparse.Namespace,
    dark_frames: list = None,       # existing (used in CSV mode only)
    orbit_group = None,             # NEW: OrbitGroup | None (H06 mode)
) -> tuple:
```

### 4.2 New skip reason

Add `'cal_missing'` to the list of skip reasons:

```
'cal_missing'  — H06 mode but orbit has no master calibration
```

### 4.3 H06 path logic

Replace the current `else: return None, "m06_missing"` block (around
line 373) with:

```python
    else:
        # No CSV supplied — use H06 fringe fitting
        if orbit_group is None or orbit_group.master_cal is None:
            log.warning(
                "Skipping %s: no master calibration for this orbit",
                bin_path.name,
            )
            return None, "cal_missing"

        # ── H07 geometry first (needed for v_los_prior_ms) ─────────────
        from windcube import wind_retrieval
        try:
            geom = wind_retrieval.compute_los_geometry(meta)
        except ValueError as exc:
            log.warning("Geometry failed for %s: %s", bin_path.name, exc)
            return None, "geometry_error"

        v_los_prior_ms = geom.V_sc_LOS + geom.v_earth_LOS

        # ── Dark subtraction using per-orbit master dark ────────────────
        if orbit_group.master_dark is not None:
            n_dark = len(orbit_group.dark_frames)
            pixels_ds = (
                pixels.astype(np.float32)
                - orbit_group.master_dark / n_dark
            )
            pixels_ds = np.clip(pixels_ds, 0, 16383).astype(np.float32)
        else:
            log.warning(
                "No master dark for orbit %d — processing %s without "
                "dark subtraction",
                orbit_group.orbit_number, bin_path.name,
            )
            pixels_ds = pixels.astype(np.float32)

        # ── H06 fringe fitting via fpi_pipeline ────────────────────────
        from windcube.fpi_pipeline import process_science_frame
        try:
            airglow = process_science_frame(
                pixels_ds      = pixels_ds,
                master_cal     = orbit_group.master_cal,
                v_los_prior_ms = v_los_prior_ms,
                r_max_px       = 110.0,
                cx_seed        = None,
                cy_seed        = None,
            )
        except Exception as exc:
            log.warning(
                "H06 inversion failed for %s: %s", bin_path.name, exc
            )
            return None, "geometry_error"

        v_rel  = airglow.v_rel_ms
        sigma_v = airglow.sigma_v_ms
        log.info(
            "H06  %-40s  v_rel=%+.1f  sigma_v=%.1f  chi2=%.2f  "
            "converged=%s",
            bin_path.name, v_rel, sigma_v,
            airglow.chi2_red, airglow.converged,
        )
```

After this block the code continues to:

```python
    # Step 2c — H07 geometry + correction
    # NOTE: In H06 mode, geometry was already computed above for v_los_prior_ms.
    # We call process_frame() regardless — it recomputes geometry internally.
    # This is a small redundancy accepted for code simplicity.
    try:
        obs = wind_retrieval.process_frame(meta, v_rel, sigma_v)
    except ValueError as exc:
        log.warning("process_frame failed for %s: %s", bin_path.name, exc)
        return None, "geometry_error"
```

### 4.4 Dark subtraction in CSV mode (unchanged)

The existing nearest-dark-frame subtraction block (around lines 376–400)
is preserved exactly for the CSV/validation path. It is only reached when
`vrel_lookup is not None`.

---

## 5. Updated `main()` — schedule and orbit_group routing

### 5.1 Schedule building (insert after Step 0b dark frame discovery)

```python
    # ── Step 0c — Build calibration schedule (H06 mode only) ──────────────
    schedule = None
    if use_h06:
        print("H06 mode: building per-orbit calibration schedule...")
        schedule = _build_schedule(input_folder, args)
        print(
            f"Schedule: {schedule.n_orbits} orbits, "
            f"{schedule.n_science} science frames"
        )
```

### 5.2 Frame loop — pass orbit_group to `_process_one()`

In the serial processing loop, replace:

```python
        obs, reason = _process_one(bin_path, vrel_lookup, args, dark_frames=dark_frames)
```

With:

```python
        # Resolve orbit group for H06 mode
        og = None
        if schedule is not None:
            # Find which OrbitGroup this science frame belongs to
            # by matching bin_path to schedule frame records
            for og_candidate in schedule.orbits:
                if any(fr.path == bin_path
                       for fr in og_candidate.science_frames):
                    og = og_candidate
                    break

        obs, reason = _process_one(
            bin_path, vrel_lookup, args,
            dark_frames=dark_frames,
            orbit_group=og,
        )
```

Apply the same change in the parallel (ThreadPoolExecutor) path.

### 5.3 Updated skip counts summary

Add `'cal_missing'` to the skip breakdown in `_write_summary()`:

```python
# In _write_summary, add alongside SLEW_IN_PROGRESS and vrel_missing:
f"  No master calibration  : {skip_counts.get('cal_missing', 0)} frames\n"
```

---

## 6. Updated docstring

Update the module docstring to document both modes:

```python
"""
invert_wind_map.py — WindCube H07 batch processing driver.

Two v_rel modes:

  CSV mode (--v-rel-csv):
    v_rel is read from a pre-computed GEN01 CSV. Used for synthetic data
    validation. All existing --v-rel-csv runs continue to work unchanged.

  H06 mode (default when --v-rel-csv not supplied):
    v_rel is recovered from raw pixels via the full H06 fringe fitting
    pipeline. Requires dark and cal frames in the input folder.
    Uses per-orbit master darks and master calibrations (5 dark + 5 cal
    frames per orbit, per WINDCUBE-ARCH-01).

[remainder unchanged]
"""
```

---

## 7. Verification

### 7.1 CSV mode regression test

Run the existing null-wind validation (already validated):

```bash
python scripts/run_wind_map.py
# Select the null-wind 5-day bin_frames folder
# Select the matching GEN01 CSV
```

Expected: identical output to previous runs (v_E=0, v_N=0).
This confirms the CSV path is 100% unaffected.

### 7.2 H06 mode smoke test (--help)

```bash
python scripts/invert_wind_map.py --help
```

Confirm `--use-h06` appears in the help text.

### 7.3 H06 mode scan test (process_cals=False equivalent)

```bash
python scripts/invert_wind_map.py <bin_frames_folder> \
    --use-h06 --verbose
```

With a folder containing dark + cal + science frames, the schedule should
build and the first few science frames should process through H06. Report
`v_rel`, `sigma_v`, `chi2_red` for the first processed frame.

---

## 8. File location

```
soc_sewell/
└── scripts/
    └── invert_wind_map.py   ← edit in place
```

---

## 9. Instructions for Claude Code

Read this entire spec, the current `scripts/invert_wind_map.py` source,
WINDCUBE-ARCH-01, S_L01, and S_L02 before writing any code.

**Step-by-step:**

1. Add `--use-h06` argument to `_parse_args()` (spec §2).

2. Add `_build_schedule()` helper (spec §3).

3. Update `_process_one()` signature and add H06 path (spec §4).
   CRITICAL: do not touch the `vrel_lookup is not None` branch.
   Only replace the final `else: return None, "m06_missing"`.

4. Update `main()`:
   - Add `use_h06` activation logic after arg parsing (spec §2)
   - Add Step 0c schedule building (spec §5.1)
   - Add orbit_group lookup and pass to `_process_one()` in both
     serial and parallel loops (spec §5.2)
   - Add `cal_missing` to skip counts (spec §5.3)

5. Update module docstring (spec §6).

6. Run verification 7.2 (--help check):
   ```
   python scripts/invert_wind_map.py --help
   ```
   Confirm `--use-h06` is present.

7. Run verification 7.1 (CSV mode regression):
   If the GEN01 null-wind CSV and bin_frames folder are accessible,
   run a short test (--n-days 1 or whatever is fast) and confirm
   the summary shows v_E ≈ 0, v_N ≈ 0.
   If not accessible, note this and skip.

8. Commit:
   ```
   feat(batch): wire H06 full pipeline into invert_wind_map.py
   Implements: S_batch_v2_invert_wind_map_2026-05-14.md
   --v-rel-csv path fully preserved. H06 mode active by default
   when --v-rel-csv not supplied.
   ```

**Do not:**
- Modify `_write_results_csv()` or `_write_summary()`
- Modify the binning or inversion logic
- Change any existing CLI argument defaults
- Remove or rename any existing function

---

*End of S_batch_v2 specification v1.0 — 2026-05-14*
