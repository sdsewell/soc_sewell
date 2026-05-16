# G01 — Synthetic Metadata Generator Specification
## WindCube SOC Pipeline — v16
**Spec ID:** G01  
**Spec file:** `specs/G01_synthetic_metadata_generator_2026-05-16_v16.md`  
**Previous spec:** `specs/G01_synthetic_metadata_generator_2026-05-14_v15.md` (v15)  
**Script:** `src/processing/GEN01_synthesize_mission_dataset_2026_05_13.py`  
**Date:** 2026-05-16  
**Status:** Authoritative — v16  

**Depends on:**
- All v15 dependencies (unchanged)
- `python-sgp4 >= 2.21` (`pip install sgp4`) — new runtime dependency
- H07-ADD-01 (`H07_addendum_coverage_criteria_2026-05-14.md`) — coverage analysis methodology

---

## Revision history

| Version | Date | Summary |
|---------|------|---------|
| 1–15 | see archive | — |
| **16** | **2026-05-16** | **Add TLE ingestion mode (§11) and weekly ops planning workflow (§12). New `propagate_orbit_from_state()` NB01 entry point (§11.3). Legacy altitude-only mode preserved. New constants (§13). New checks C31–C33.** |

---

## 1–10. Unchanged from v15

All sections 1–10 (user interface, wind maps, orbit propagation, metadata,
CSV format, binary synthesis, verification checks C1–C30, coverage
diagnostics) are unchanged from v15.

---

## 11. TLE ingestion mode (NEW in v16)

### 11.1 Motivation

Once WindCube is on orbit, Space-Track.org delivers updated TLEs weekly
(or more frequently during anomaly periods).  The current GEN01 circular
Keplerian propagator (NB01) uses a user-typed altitude and a synthetic
SSO epoch.  After launch, three things change:

- The actual orbital altitude drifts due to atmospheric drag (B* term).
- The true RAAN and inclination differ from the design values by injection
  error, typically ±0.1°.
- The simulation epoch must precisely match the ops planning window.

TLE mode replaces the typed epoch and altitude prompts with a file-picker
dialog that loads the current WindCube TLE.  All downstream pipeline logic
(NB02a/b/c, image synthesis, coverage diagnostics) is unchanged.

### 11.2 TLE mode activation

At the start of `main()`, before the existing epoch prompt, GEN01 asks:

```
Use a TLE file for orbit initialisation?  [y/N]: 
```

- If the user enters `y` or `Y`, TLE mode is activated.  A Windows
  file-picker dialog opens (same pattern as the existing `_pick_folder`
  helper) asking the user to locate the `.tle` or `.txt` file.
- If the user enters anything else (including blank → default `N`),
  the existing prompts for epoch and altitude run as before (legacy mode).

### 11.3 TLE file format accepted

GEN01 accepts the standard three-line element set format:

```
WINDCUBE-FPI          
1 55001U 24010A   24010.50000000  .00001234  00000-0  10234-4 0  9994 6
2 55001  97.4400 127.3456 0009182 101.2345 260.4321 14.57599490284315
```

The title line (line 0) is optional.  GEN01 also accepts a two-line TLE
(lines 1 and 2 only, no title).

Files may contain multiple TLE sets (e.g. a Space-Track download with
historical entries).  If more than one set is found, GEN01 prints a
numbered list and prompts the user to select by index (default: the
most recent by TLE epoch).

### 11.4 TLE parsing and state extraction

Use `sgp4.api.Satrec` to parse the TLE:

```python
from sgp4.api import Satrec, jday

sat = Satrec.twoline2rv(line1, line2)

# Propagate to the TLE epoch to get the initial state
jd_whole, jd_frac = jday(
    tle_epoch_dt.year, tle_epoch_dt.month, tle_epoch_dt.day,
    tle_epoch_dt.hour, tle_epoch_dt.minute,
    tle_epoch_dt.second + tle_epoch_dt.microsecond / 1e6
)
e, r_km, v_km_s = sat.sgp4(jd_whole, jd_frac)
```

Where `e` is the SGP4 error code (0 = success).  Raise `RuntimeError`
if `e != 0`.

**Derived quantities extracted from the TLE:**

| Quantity | Source | Used for |
|---|---|---|
| `t_start` | TLE epoch (as ISO-8601 UTC string) | Simulation start epoch |
| `altitude_km` | `(|r_km| - R_EARTH_KM)` at TLE epoch | README/logging only |
| `pos0_eci_m` | `np.array(r_km) * 1000` | NB01 state initialisation |
| `vel0_eci_m_s` | `np.array(v_km_s) * 1000` | NB01 state initialisation |
| `inclination_deg` | `sat.inclo * 180/π` | Printed summary; no algorithmic use |
| `bstar` | `sat.bstar` | Printed summary; not used in propagator |
| `mean_motion_rev_per_day` | `sat.no_kozai * (1440 / (2π))` | Replaces Keplerian T_ORBIT_S |

`R_EARTH_KM = 6371.0` (mean sphere radius, for altitude display only —
not used in orbit propagation math).

**Orbital period from TLE:**

```python
T_ORBIT_S = 86400.0 / mean_motion_rev_per_day
```

This replaces the circular Keplerian formula
`2π √(a³/μ)` used in legacy mode.

### 11.5 NB01 new entry point: `propagate_orbit_from_state()`

Add to `src/geometry/nb01_orbit_propagator_*.py`:

```python
def propagate_orbit_from_state(
    pos0_m: np.ndarray,
    vel0_m_s: np.ndarray,
    t0: astropy.time.Time,
    duration_s: float,
    dt_s: float = 10.0,
) -> pd.DataFrame:
    """
    Propagate a WindCube orbit from an initial ECI state vector.

    Parameters
    ----------
    pos0_m      : Initial ECI position, metres  (3,)
    vel0_m_s    : Initial ECI velocity, m/s     (3,)
    t0          : Astropy Time object, start epoch (UTC)
    duration_s  : Simulation duration, seconds
    dt_s        : Step size, seconds (default 10.0 — same as propagate_orbit)

    Returns
    -------
    pd.DataFrame — same schema as propagate_orbit() output:
        epoch, lat_deg, lon_deg, alt_km,
        pos_eci_x, pos_eci_y, pos_eci_z,
        vel_eci_x, vel_eci_y, vel_eci_z

    Implementation notes
    --------------------
    Uses the same two-body Keplerian stepping already in propagate_orbit():
    advance position and velocity by dt_s using the vis-viva equation and
    rotation of the state vector.  The TLE supplies the accurate initial
    state; thereafter, the two-body approximation accumulates only a small
    error over the 7-day planning horizon (< 1 km positional error per day
    for a ~500 km SSO with B* ≈ 1e-4).

    The function converts ECI position to geodetic lat/lon/alt using
    astropy GCRS→ITRS (same as propagate_orbit).
    """
```

**Design rationale for not using full SGP4 at every step:**

SGP4 propagation at every 10-second step over 7 days (60 480 calls) would
give higher fidelity but is not needed for the planning use case — the
goal is to predict which geographic bins will be observed, not to produce
a precise ephemeris.  The two-body approximation introduces < 1 km
positional error per day, which is negligible relative to the 5°×5°
coverage bin size (~550 km at the equator).

If higher fidelity is needed in a future version, `propagate_orbit_from_state()`
can be updated to call `sat.sgp4()` at each time step without changing the
GEN01 calling interface.

### 11.6 GEN01 `main()` TLE-mode calling path

When TLE mode is active, GEN01:

1. Skips the `t_start` and `altitude_km` prompts.
2. Sets `t_start` from the TLE epoch (ISO-8601 UTC string).
3. Sets `altitude_km` from `|r_km| - R_EARTH_KM` (display/logging only).
4. Calls `propagate_orbit_from_state()` instead of `propagate_orbit()`.
5. Computes `T_ORBIT_S` from the TLE mean motion.
6. All remaining prompts (duration, science band, cadence, wind map, etc.)
   run identically to legacy mode.

The README sidecar gains a new section in TLE mode:

```
--- TLE Source ---
TLE file           : <filename>
NORAD ID           : <sat_no>
TLE epoch          : <tle_epoch_utc>
Inclination        : <inclination_deg:.4f>°
BSTAR drag         : <bstar:.5e>
Mean motion        : <mean_motion_rev_per_day:.8f> rev/day
T_orbit (TLE)      : <T_ORBIT_S:.1f> s  (<T_ORBIT_S/60:.2f> min)
Altitude at epoch  : <altitude_km:.1f> km  (derived: |r| - R_earth)
```

### 11.7 Output filename disambiguation

In TLE mode, the output stem gains a `_tle` suffix to distinguish it from
legacy-mode runs at the same epoch:

```
GEN01_<YYYYMMDD>_<duration>d_<windmap_tag>_seed<NNNN>_tle
```

Example:
```
GEN01_20270101_001.0d_uniform_seed0042_tle.csv
```

---

## 12. Weekly ops planning workflow (NEW in v16)

This section documents the recommended weekly workflow for using GEN01
as a mission operations planning tool after WindCube is on orbit.

### 12.1 Inputs required each week

| Input | Source | How to obtain |
|---|---|---|
| Current WindCube TLE | Space-Track.org or Celestrak | Download `.tle` file |
| Ops planning start date | Mission timeline | Set in tool prompt |
| Duration | Typically 7 days | Default or prompt |
| Wind map | HWM14 (quiet) or Storm for event planning | Prompt choice |

### 12.2 Recommended weekly procedure

**Step 1 — Download fresh TLE**

Visit `https://celestrak.org/SOCRATES/` or `https://www.space-track.org`
and download the current WindCube TLE.  Save as `windcube_current.tle`.
TLEs age — use one no older than 3 days for planning.

**Step 2 — Run GEN01 in TLE mode**

```
python src/processing/GEN01_synthesize_mission_dataset_<date>.py
```

At the TLE prompt, enter `y` and select `windcube_current.tle`.
Set duration to 7.0 days.  Select wind map appropriate for the week
(HWM14 quiet for nominal, Storm for active periods).

**Step 3 — Review coverage diagnostics**

GEN01 prints the coverage report at run end.  Key number to check:
`Mixed AT+CT (good H07)` — should be > 80% for a 7-day run.

**Step 4 — Run coverage_map.py**

```
python scripts/coverage_map.py <output_csv> --save
```

Review the four figures:
- Figure 1 (coverage map): geographic gaps in AT+CT overlap
- Figure 2 (pass count): asymmetry between AT and CT passes
- Figure 3 (forecast curve): where this run sits on the coverage curve
- Figure 4 (ground track): raw tangent point density

**Step 5 — Archive and commit**

Commit the CSV, coverage report, and PNG figures to the SOC weekly
planning archive.  The output stem includes the TLE epoch date, making
each weekly run self-documenting.

### 12.3 TLE age warning

GEN01 prints a warning if the TLE epoch is more than 7 days before
`t_start`:

```
WARNING: TLE epoch is N.N days before simulation start.
         Propagation accuracy degrades at >3 days for LEO.
         Consider downloading a fresher TLE.
```

This is a warning only — the run proceeds.

### 12.4 Multiple-TLE file handling

Space-Track batch downloads often contain multiple historical TLEs for
the same satellite.  When GEN01 detects more than one TLE in the file:

1. Print a numbered list sorted by epoch (newest first).
2. Prompt: `Select TLE [1 = most recent]: `
3. Default to the most recent (index 1).

---

## 13. New constants (v16)

Add to `windcube/constants.py`:

```python
# TLE / SGP4 propagation
R_EARTH_MEAN_KM = 6371.0     # mean Earth radius for altitude display [km]
SGP4_MAX_AGE_DAYS = 7.0      # warn if TLE epoch older than this before t_start
```

Note: `R_EARTH_MEAN_KM` is used only for the altitude display in the
README sidecar and the TLE summary printout.  All geodetic calculations
continue to use the WGS84 ellipsoid via astropy.

---

## 14. Verification checks (v16 additions)

### C31 — SGP4 propagation returns no error

In TLE mode, after calling `sat.sgp4(jd_whole, jd_frac)`:
- `e == 0` (SGP4 error code is success)
- `|r_km|` is between 6500 and 7500 km (reasonable LEO range)
- `|v_km_s|` is between 6.5 and 8.5 km/s (reasonable LEO range)

### C32 — TLE-mode README contains TLE section

After a TLE-mode run, the README sidecar file must contain the string
`"TLE Source"`.

### C33 — Output stem contains `_tle` suffix

In TLE mode, the output CSV filename must end with `_tle.csv`.

---

## 15. Instructions for Claude Code

Read this entire spec (v16) AND v15 before touching any code.
Read the NB01 spec before modifying `nb01_orbit_propagator_*.py`.

**Dependency check first:**

```bash
python -c "from sgp4.api import Satrec; print('sgp4 OK')"
```

If this fails, install: `pip install sgp4`.

**Changes required:**

### 15.1 Add `_load_tle_dialog()` helper to GEN01

```python
def _load_tle_dialog() -> tuple[str, str, str]:
    """
    Open a file-picker dialog to select a TLE file.
    Returns (title_line, line1, line2) for the selected TLE set.
    Raises RuntimeError if the file cannot be parsed.
    """
```

Implementation:
- Use `filedialog.askopenfilename(...)` (same tkinter pattern as
  `_pick_folder`) with `filetypes=[("TLE files", "*.tle *.txt"), ("All", "*")]`
- Read the file, split into non-blank lines
- Parse into TLE sets (groups of 2 or 3 lines where line1 starts with "1 "
  and line2 starts with "2 ")
- If multiple sets found, print list and prompt for selection
- Return the chosen `(title, line1, line2)` tuple

### 15.2 Add `_parse_tle_epoch()` helper to GEN01

```python
def _parse_tle_epoch(line1: str) -> datetime.datetime:
    """
    Parse the epoch field from TLE line 1 (columns 19–32, 1-indexed).
    Returns a Python datetime in UTC.

    TLE epoch format: YYDDD.DDDDDDDD
      YY: 2-digit year (57–99 → 1957–1999; 00–56 → 2000–2056)
      DDD.DDDDDDDD: day of year with fractional day
    """
```

### 15.3 Add `propagate_orbit_from_state()` to NB01 module

Implement as described in §11.5.  The function signature must match
exactly — `main()` will call it as:

```python
from src.geometry.nb01_orbit_propagator_2026_04_16 import (
    propagate_orbit,
    propagate_orbit_from_state,   # new
)

df_sched = propagate_orbit_from_state(
    pos0_m    = pos0_eci_m,
    vel0_m_s  = vel0_eci_m_s,
    t0        = Time(t_start, format="isot", scale="utc"),
    duration_s = duration_s,
    dt_s       = SCHED_DT_S,
)
```

The returned DataFrame must have exactly the same column schema as
`propagate_orbit()` so that all downstream code is unchanged.

### 15.4 Modify `main()` in GEN01

Add the TLE mode prompt block at the very start of `main()`, before the
existing `t_start` prompt.  Use a flag `use_tle: bool`.

In the TLE path:
- Call `_load_tle_dialog()` to get `(title, line1, line2)`
- Parse with `Satrec.twoline2rv(line1, line2)`
- Propagate to TLE epoch to get `pos0_eci_m`, `vel0_eci_m_s`
- Derive `t_start`, `altitude_km`, `T_ORBIT_S`
- Print TLE summary table (see §11.6)
- Check TLE age and print warning if needed (§12.3)
- Skip the `t_start` and `altitude_km` prompts

The `altitude_km` variable must still exist in scope after the TLE block
because it is used in: README sidecar writing, `compute_los_eci()` call,
and `_build_readme_lines()`.  In TLE mode it is set to the derived value.

### 15.5 Add C31–C33 verification checks

Add after C30 in the verification section of `main()`.  C31 and C32 run
only when `use_tle is True`; C33 runs only in TLE mode.

### 15.6 Smoke test

**Legacy mode (regression):**
Run GEN01 with duration 0.1 days, uniform wind, default altitude.
Confirm all C1–C30 checks still pass.

**TLE mode:**
Create a minimal valid TLE file `test_windcube.tle`:
```
WINDCUBE-FPI
1 55001U 24010A   24010.50000000  .00001234  00000-0  10234-4 0  9994 6
2 55001  97.4400 127.3456 0009182 101.2345 260.4321 14.57599490284315
```

Run GEN01, enter `y` at the TLE prompt, select this file, duration 0.1 days.
Confirm:
- TLE summary prints correctly
- Simulation runs to completion
- C31, C32, C33 PASS
- Output CSV filename contains `_tle`

### 15.7 Commit

```
feat(gen01): v16 — TLE ingestion and weekly ops planning mode

Implements: G01_synthetic_metadata_generator_2026-05-16_v16.md (v16)
- GEN01: TLE mode (y/N prompt, file dialog, sgp4 state extraction)
- NB01: propagate_orbit_from_state() new entry point
- windcube/constants.py: R_EARTH_MEAN_KM, SGP4_MAX_AGE_DAYS
- C31, C32, C33 verification checks added

Legacy altitude-only mode fully preserved.
```

Also update PIPELINE_STATUS.md: G01 status → v16, date → 2026-05-16.

---

*End of G01 Specification v16 — 2026-05-16*
