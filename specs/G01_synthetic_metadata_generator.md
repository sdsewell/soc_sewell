# G01 — Synthetic Metadata Generator Specification
## WindCube SOC Pipeline — v17

<!-- Version: 17 | Date: 2026-05-27 | Commit: 1892b71 -->
<!-- Versioning via git — no datestamp in filename -->

**Spec ID:** G01  
**Spec file:** `specs/G01_synthetic_metadata_generator.md`  
**Previous spec file:** `specs/G01_synthetic_metadata_generator_2026-05-16_v16.md` (archived)  
**Script:** `src/processing/GEN01_synthesize_mission_dataset_2026_05_16.py`  
**Date:** 2026-05-27  
**Status:** Authoritative — v17 (implemented, commit `1892b71`)  

**Depends on:**
- All v15 dependencies (unchanged)
- `python-sgp4 >= 2.21` (`pip install sgp4`) — new runtime dependency
- H07-ADD-01 (`H07_addendum_coverage_criteria_2026-05-14.md`) — coverage analysis methodology

---

## Revision history

| Version | Date | Summary |
|---------|------|---------|
| 1–15 | see archive | — |
| 16 | 2026-05-16 | Add TLE ingestion mode (§11) and weekly ops planning workflow (§12). New `propagate_orbit_from_state()` NB01 entry point (§11.3). Legacy altitude-only mode preserved. New constants (§13). New checks C31–C33. |
| 16.1 | 2026-05-16 | Implementation note: `propagate_orbit_from_state()` uses symplectic Euler (leapfrog). Spec path corrected to `specs/`. C16 and C30 pre-existing failures confirmed unrelated to v16 scope. |
| **17** | **2026-05-27** | **All FPI optical constants removed from GEN01 and imported from `windcube/constants.py` (single source of truth). Wavelength policy: NIST vacuum wavelengths throughout (Edlén 1966 conversion from air). `windcube/constants.py` updated to H05 TBAL values (Denmark testing, 2026-05-24, commit `1892b71`). `src/fpi/fpi_cal_lib.py` `InstrumentParams` defaults updated to match. See §7 (constants policy) and §16 (constants table). Spec filename convention changed: no datestamp in filename — git tracks versions.** |

---

## 1–10. Unchanged from v15

All sections 1–10 (user interface, wind maps, orbit propagation, metadata,
CSV format, binary synthesis, verification checks C1–C30, coverage
diagnostics) are unchanged from v15.

---

## 7. FPI constants policy (NEW in v17)

### 7.1 Single source of truth

**All FPI optical constants in GEN01 are imported from `windcube/constants.py`.
No FPI optical constant may be hardcoded in the GEN01 script.**

The relevant import block in GEN01:

```python
from windcube.constants import (
    # existing imports
    R_EARTH_MEAN_KM, SGP4_MAX_AGE_DAYS, WGS84_A_M, EARTH_GRAV_PARAM_M3_S2,
    # FPI optical model — imported, not hardcoded
    ETALON_GAP_M        as _C_ETALON_GAP_M,
    ETALON_N            as _C_ETALON_N,
    ALPHA_RAD_PX        as _C_PLATE_SCALE_RPX,
    R_REFL              as _C_R_REFL,
    R_REFL_2            as _C_R_REFL_2,
    NE_INTENSITY_2      as _C_REL_638,
    PSF_SIGMA0_PX       as _C_SIGMA0_PX,
    OI_WAVELENGTH_VAC_M as _C_LAMBDA_OI_M,
    NE_WAVELENGTH_1_VAC_M as _C_LAMBDA_NE1_M,
    NE_WAVELENGTH_2_VAC_M as _C_LAMBDA_NE2_M,
    SPEED_OF_LIGHT_MS   as _C_SPEED_OF_LIGHT_MS,
    R_MAX_PX            as _C_R_MAX_PX,
    CCD_DARK_RATE_E_PX_S as _C_QDD_AT_20C,
)
```

Module-level names (`LAMBDA_OI_M`, `ETALON_GAP_M`, etc.) are then assigned
from the imported values so that internal GEN01 code is unchanged.
`FINESSE_F` remains derived inline: `4 * R_REFL / (1 - R_REFL) ** 2`.

CCD electronics constants (`BIAS_DN`, `GAIN_E_PER_DN`, `READ_NOISE_E`,
`OFFSET_ADU`, `ADU_MAX`, `SCI_PEAK_ADU`, `CAL_PEAK_ADU`) remain hardcoded
in GEN01 pending a separate reconciliation of the FM PTC measurement values
with `windcube/constants.py` (see §16, known inconsistencies).

### 7.2 Wavelength policy

**All forward-model wavelength calculations use NIST vacuum wavelengths,
derived via the Edlén (1966) formula from NIST air standards.**

| Usage | Constant | Value |
|---|---|---|
| OI airglow forward model | `OI_WAVELENGTH_VAC_M` | 630.204637e-9 m |
| Ne 640 nm forward model | `NE_WAVELENGTH_1_VAC_M` | 640.401775e-9 m |
| Ne 638 nm forward model | `NE_WAVELENGTH_2_VAC_M` | 638.475557e-9 m |
| H06/H07 Doppler reference | `OI_LAMBDA0_NM = 630.0` | 630.0 nm (Harding convention) |

The Harding `OI_LAMBDA0_NM = 630.0` is kept as a separate constant for the
H06/H07 velocity recovery convention. It is **not** used in GEN01 forward
model calculations. The 0.2 nm difference corresponds to a ~95 m/s systematic
offset in v_rel, which is accounted for in the H06/H07 zero-wind calibration
procedure — not by changing the wavelength here.

Air wavelengths (`OI_WAVELENGTH_AIR_M`, `NE_WAVELENGTH_1_AIR_M`,
`NE_WAVELENGTH_2_AIR_M`) remain in `windcube/constants.py` as the NIST source
values, but are not used directly in any forward model computation.

### 7.3 FSR derived quantities

The following derived quantities in `windcube/constants.py` now use vacuum
wavelengths per the policy above (updated in commit `1892b71`):

```python
ETALON_FSR_NE1_M    = NE_WAVELENGTH_1_VAC_M**2 / (2.0 * ETALON_GAP_M)
ETALON_FSR_OI_M     = OI_WAVELENGTH_VAC_M**2   / (2.0 * ETALON_GAP_M)
VELOCITY_PER_FSR_MS = SPEED_OF_LIGHT_MS * ETALON_FSR_OI_M / OI_WAVELENGTH_VAC_M
NE_DELTA_LAMBDA_M   = NE_WAVELENGTH_1_VAC_M - NE_WAVELENGTH_2_VAC_M
NE_SEPARATION_FSR   = NE_DELTA_LAMBDA_M / ETALON_FSR_NE1_M
```

Verified values (commit `1892b71`):

| Quantity | Value |
|---|---|
| FSR(Ne1 vac) | 10.1980 pm |
| FSR(OI vac) | 9.8758 pm |
| Velocity per FSR | 4697.98 m/s |
| NE_SEPARATION_FSR | 188.8823 |

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

**Integrator (as implemented):** Symplectic Euler (leapfrog) — velocity
is updated first from the gravitational acceleration, then position is
updated using the new velocity.  This is the standard choice for
conservative two-body systems; it conserves orbital energy better than
forward Euler and prevents secular drift in the semi-major axis over
multi-day runs.

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

## 16. FPI constants table (NEW in v17)

Authoritative values as of commit `1892b71` (2026-05-27).
Source for all FPI optical parameters: **H05 10-parameter Harding inversion,
`0_30exp_cal_swapped.bin`, TBAL Denmark testing, 2026-05-24.**

Any future update to these values must be made **only in `windcube/constants.py`**.
GEN01, `fpi_cal_lib.py`, H02, H05, H06, and Z03 must all import from there.
A cross-spec consistency check (equivalent to the LD-5 rule in earlier specs)
must be run after any constants.py edit — confirm derived FSR values match §7.3.

### 16.1 FPI optical model constants

| Constant in `constants.py` | GEN01 alias | Value | Source | 1σ |
|---|---|---|---|---|
| `ETALON_GAP_M` | `ETALON_GAP_M` | 20.1076267e-3 m | H05 TBAL | 1.7 nm |
| `ALPHA_RAD_PX` | `PLATE_SCALE_RPX` | 1.60854e-4 rad/px | H05 TBAL | 0.00014e-4 |
| `R_REFL` | `R_REFL` | 0.241 | H05 TBAL, R1 @ 640.2 nm | 0.010 |
| `R_REFL_2` | `R_REFL_2` | 0.28303 | H05 TBAL, R2 @ 638.3 nm | 0.014 |
| `NE_INTENSITY_2` | `REL_638` | 0.7537 | H05 TBAL ne_ratio | 0.012 |
| `PSF_SIGMA0_PX` | `SIGMA0_PX` | 0.5592 px | H05 TBAL σ₀ | 0.009 |
| `ETALON_N` | `N_GAP` | 1.0 | — (air gap) | — |
| `R_MAX_PX` | `R_MAX_PX` | 110 px | FlatSat/flight | — |

### 16.2 Wavelengths (NIST vacuum, Edlén 1966)

| Constant in `constants.py` | GEN01 alias | Value |
|---|---|---|
| `OI_WAVELENGTH_VAC_M` | `LAMBDA_OI_M` | 630.204637e-9 m |
| `NE_WAVELENGTH_1_VAC_M` | `LAMBDA_NE1_M` | 640.401775e-9 m |
| `NE_WAVELENGTH_2_VAC_M` | `LAMBDA_NE2_M` | 638.475557e-9 m |

### 16.3 Derived instrument quantities (from constants.py at load time)

| Quantity | Value | Notes |
|---|---|---|
| `FINESSE_F` (in GEN01) | 1.6734 | `4R/(1−R)²` at R=0.241 |
| Reflective finesse N_R | 2.03 | `π√R/(1−R)` — fringes are broad/shallow |
| FSR(Ne1 vac) | 10.1980 pm | |
| FSR(OI vac) | 9.8758 pm | |
| Velocity per FSR | 4697.98 m/s | |
| NE_SEPARATION_FSR | 188.8823 | |

> **Note on low finesse:** N_R ≈ 2.03 means fringe peak-to-trough contrast
> is only ~2.7:1 (compared to ~40:1 at R=0.725). The WindCube FlatSat
> etalon coatings produce inherently broad, shallow fringes — this is the
> real instrument. The H05 inversion accounts for this via the PSF parameter
> σ₀. All pipeline forward models must use R=0.241, not legacy values of
> 0.53 or 0.725 which were pre-measurement placeholders.

### 16.4 Known inconsistencies (open items)

| Item | GEN01 hardcoded | `constants.py` | Action required |
|---|---|---|---|
| `BIAS_DN` | 275.0 ADU | not present | Add `CCD_BIAS_DN` to constants.py from FM PTC report |
| `GAIN_E_PER_DN` | 3.29 e-/DN | not present | Add `CCD_GAIN_E_PER_DN` from FM PTC (WIND-XCAM-RE-00035) |
| `READ_NOISE_E` | 4.61 e- | `CCD_READ_NOISE_E = 2.2` | 4.61 is FM overscan measurement; 2.2 is pre-FM placeholder — update constants.py |
| `OFFSET_ADU` | 5 | not present (old convention) | Reconcile with `BIAS_DN=275` — these refer to different processing stages |

The CCD electronics reconciliation is a separate task. Until resolved, GEN01
hardcodes these four values with comments citing their FM measurement sources.

### 16.5 Files updated in commit `1892b71`

| File | Change |
|---|---|
| `windcube/constants.py` | Updated `ETALON_GAP_M`, `D_TOLANSKY_MM`, `ALPHA_RAD_PX` (×2 locations), `ETALON_R_INSTRUMENT`, `R_REFL`, `NE_INTENSITY_2`. Added `R_REFL_2`, `PSF_SIGMA0_PX`, `OI_WAVELENGTH_VAC_M`, `NE_WAVELENGTH_1_VAC_M`, `NE_WAVELENGTH_2_VAC_M`. FSR formulas switched to vacuum wavelengths. |
| `src/processing/GEN01_synthesize_mission_dataset_2026_05_16.py` | All FPI optical constants now imported from `windcube.constants`. No hardcoded FPI values remain. |
| `src/fpi/fpi_cal_lib.py` | `InstrumentParams` defaults (`t`, `R_refl`, `alpha`) updated to H05 TBAL values to satisfy `mc01_fpi_mc_engine.py` module-level consistency assertion. |

---

*End of G01 Specification v17 — 2026-05-27 — commit `1892b71`*
