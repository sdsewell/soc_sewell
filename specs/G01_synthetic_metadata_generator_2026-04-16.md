# G01 — Synthetic Metadata Generator Specification

**Spec ID:** G01
**Spec file:** `docs/specs/G01_synthetic_metadata_generator_2026-04-16.md`
**Project:** WindCube FPI Science Operations Center Pipeline
**Institution:** NCAR / High Altitude Observatory (HAO)
**Status:** ✓ Complete — all 17 checks pass (C1–C17); 30-day run confirmed
**Depends on:**
  - NB00 (`nb00_wind_map_2026_04_06.py`) — `WindMap`, `UniformWindMap`,
    `AnalyticWindMap`, `HWM14WindMap`, `StormWindMap`
  - NB01 (`nb01_orbit_propagator_2026_04_16.py`) — `propagate_orbit(t_start, duration_s, dt_s)`
  - NB02a (`nb02a_boresight_2026_04_16.py`) — `compute_los_eci(pos_eci, vel_eci, look_mode, h_target_km)`
  - NB02b (`nb02b_tangent_point_2026_04_16.py`) — `compute_tangent_point(pos_eci, los_eci, epoch, h_target_km)`
  - P01 (`p01_image_metadata_2026_04_06.py`) — `ImageMetadata`, `AdcsQualityFlags`,
    `compute_adcs_quality_flag()`
  - `src/constants.py` — `WGS84_A_M`, `EARTH_GRAV_PARAM_M3_S2`
  - `tkinter` (stdlib) — native folder-browser dialog
**Used by:**
  - Z02 (synthetic airglow image generator)
  - Z03 (synthetic neon calibration image generator)
  - Future dark frame synthesis
**References:**
  - **WindCube Mission CONOPS Document**
    Document ID: `[TBD — insert document number, e.g. WC-OPS-XXXX]`
    Issue/version: `[TBD — insert version, e.g. Issue 1.0, 2025-MM-DD]`
    Sections referenced: observation schedule, science band definition,
    calibration cadence, dark frame strategy, exposure time budget
    *(When this CONOPS document is updated, review §3.2–3.3 and the
    `exp_time_cts` defaults in §5 against the new schedule and update
    the version citation above before re-implementing.)*
  - SI-UCAR-WC-RP-004 Issue 1.0 — AOCS Design Report (BRF/THRF/SIRF frames)
  - Drob et al. (2015), doi:10.1002/2014EA000089 — HWM14 empirical model
**Last updated:** 2026-04-16

> **CONOPS version note:** §3 observation schedule parameters and §4.5
> exposure time defaults are CONOPS-driven. Any CONOPS update requires a
> dated revision of this spec before Claude Code is re-run.

> **Revision history:**
> - v1 (2026-04-16): Initial — latitude-threshold state, one record per epoch.
> - v2 (2026-04-16): CONOPS model — science cadence, ±lat_band, 60°N trigger.
> - v3 (2026-04-16): S-number references replaced with NB01/NB02a/P01 names;
>   CONOPS document citation added.
> - v4 (2026-04-16): Half-normal PE distribution; C6/C10/C11 corrected;
>   script relocated to `validation/`.
> - v5 (2026-04-16): `exp_time_cts` prompt; `exp_unit = 38500`; CCD temp
>   noise N(−10, 1°C); RNG draw order fixed.
> - v6 (2026-04-16): CSV reduced to 38 binary-header columns; `.npy` retains
>   full `ImageMetadata`; C12 column-count guard added.
> - v7 (2026-04-16): Output directory via `tkinter` folder-browser dialog;
>   `_pick_folder()` helper; default path `C:\Users\sewell\WindCube\G01_outputs`.
>   Fixed corrupted `_project_root` path.
> - v8 (2026-04-16): NB02b tangent point + NB00 wind sampling for science
>   frames. Wind map registry with 5 builders; extensible. `los_eci` retained
>   from NB02a. CSV extended to 42 columns (4 new; `NaN` for cal/dark).
>   Filename includes `windmap_tag`. C12 updated (38→42); C14–C17 added.
>   `exp_time` stored as centiseconds: `round(exp_time_cts × 0.001 × 100)`.
>   All 17 checks pass on confirmed 30-day run.

---

## 1. Purpose

G01 is a standalone, interactive Python script that pre-computes and saves
the complete AOCS/instrument metadata array for a synthetic WindCube FPI
observation campaign. Every downstream image synthesis module (Z02, Z03, dark
frames) consumes these `ImageMetadata` records rather than re-running the
geometry pipeline independently.

**What G01 produces per frame type:**

| Frame type | NB01 | NB02a | NB02b | NB00 | Notes |
|------------|------|-------|-------|------|-------|
| science | ✓ orbit state | ✓ attitude q + los_eci | ✓ tangent point | ✓ wind sample | Full truth geometry |
| cal | ✓ orbit state | ✓ attitude q + los_eci | — | — | Lamp/shutter state only |
| dark | ✓ orbit state | ✓ attitude q + los_eci | — | — | Shutter-closed only |

**What G01 does not produce:**
- `truth_v_los` — requires NB02c LOS projection; computed by Z02.
- Pixel data — Z02/Z03's responsibility.
- Calls to P01 `build_synthetic_metadata()` — requires NB02c wind outputs.

---

## 2. User interface — interactive prompts

Numeric prompts appear first. The wind map selection menu follows. The
folder-browser dialog appears last. Invalid entries are rejected and
re-prompted; blank input accepts the default.

```
=== G01 — WindCube Synthetic Metadata Generator ===

Start epoch          [2027-01-01T00:00:00 UTC]  : _
Duration             [days,  default  30       ] : _
Science band         [deg,   default  40       ] : _
Obs. cadence         [sec,   default  10       ] : _
Cal/dark frames (n)  [int,   default   5       ] : _
Exposure time        [cts,   default 8000      ] : _
S/C altitude         [km,    default 510       ] : _
Tangent height       [km,    default 250       ] : _
Random seed          [int,   default  42       ] : _

Select wind map for science frame truth winds:
  [1] Uniform constant        (v_zonal, v_merid — specify values)
  [2] Analytic sine_lat       (A_z·sin(lat), A_m·cos(lat))
  [3] Analytic wave4 / DE3    (4-wave lat×lon pattern)
  [4] HWM14 quiet-time        (requires hwm14 package)
  [5] HWM14 storm / DWM07     (requires hwm14 package)
Choice [default 1]: _

Select output folder (dialog opening — check taskbar if not visible)...
→ Selected: C:\Users\sewell\WindCube\G01_outputs
```

### 2.1 Numeric parameter table

| Symbol | Description | Default | Type | Valid range |
|--------|-------------|---------|------|-------------|
| `t_start` | Start epoch | `"2027-01-01T00:00:00"` | ISO 8601 UTC | Any valid UTC string |
| `duration_days` | Duration | 30.0 | float | 0.1 – 365.0 |
| `lat_band_deg` | Science band half-width | 40.0 | float | 5.0 – 89.0 |
| `obs_cadence_s` | Observation cadence | 10.0 | float | 10.0 – 3600.0 |
| `n_caldark` | Cal/dark frames per trigger | 5 | int | 1 – 50 |
| `exp_time_cts` | Exposure time, timer counts | 8000 | int | 100 – 100000 |
| `altitude_km` | S/C altitude | 510.0 | float | 400.0 – 700.0 |
| `h_target_km` | Tangent height | 250.0 | float | 100.0 – 400.0 |
| `rng_seed` | NumPy RNG seed | 42 | int | ≥ 0 |

### 2.2 Wind map sub-prompts

After the menu choice, display only the sub-prompts relevant to that map:

| Choice | Sub-prompts (defaults) |
|--------|----------------------|
| 1 — Uniform | `v_zonal [m/s, default 100]`, `v_merid [m/s, default 0]` |
| 2 — Analytic sine_lat | `A_zonal [m/s, default 200]`, `A_merid [m/s, default 100]` |
| 3 — Analytic wave4 | `A_zonal [m/s, default 150]`, `A_merid [m/s, default 75]`, `phase [rad, default 0.0]` |
| 4 — HWM14 quiet | `day_of_year [default 172]`, `ut_hours [default 12.0]`, `f107 [default 150.0]`, `ap [default 4]` |
| 5 — HWM14 storm | `day_of_year [default 355]`, `ut_hours [default 3.0]`, `f107 [default 180.0]`, `ap [default 80]` |

For choices 4 and 5, if `hwm14` is not importable, print an error and
re-display the menu. Do not crash.

### 2.3 Output folder (`_pick_folder()`)

After wind map construction, print:
```
Select output folder (dialog opening — check taskbar if not visible)...
```
and open a native `tkinter.filedialog.askdirectory()` dialog with the root
window hidden and brought to the front. Default starting location:
`C:\Users\sewell\WindCube\G01_outputs`. If the user cancels, use this default.
Echo the chosen path to the terminal.

**Constraint warnings** (console only, not errors):
- `lat_band_deg ≥ 60.0`: science band overlaps cal/dark trigger; cal/dark wins.
- `2 × n_caldark × obs_cadence_s > 1200`: sequence may extend past 60°N arc.

---

## 3. Wind map registry — extensible design

The wind map menu is driven by `WIND_MAP_REGISTRY` and `WIND_MAP_TAGS`,
defined at module level. Adding a new map type requires adding one entry
to each dict — no other changes are needed.

```python
# ---------------------------------------------------------------------------
# WIND_MAP_REGISTRY and WIND_MAP_TAGS — extend here to add new wind map types.
#
# WIND_MAP_REGISTRY:
#   key   : menu number string (keep sequential)
#   value : (display_label, builder_function)
#
# builder_function signature:
#   (rng: Generator, h_target_km: float, **user_params) -> WindMap
#   rng          — numpy RNG, available for any stochastic initialisation
#   h_target_km  — altitude for HWM14/Storm backends
#   **user_params — keyword args from per-map sub-prompts
#
# WIND_MAP_TAGS:
#   key   : same menu number string
#   value : short lowercase label used in output filenames
#
# NB00 class → registry entry:
#   UniformWindMap  → '1'   tag 'uniform'
#   AnalyticWindMap → '2'   tag 'sine_lat'
#   AnalyticWindMap → '3'   tag 'wave4'
#   HWM14WindMap    → '4'   tag 'hwm14'
#   StormWindMap    → '5'   tag 'storm'
# ---------------------------------------------------------------------------

WIND_MAP_REGISTRY: dict[str, tuple[str, callable]] = {
    '1': ('Uniform constant',    _build_uniform),
    '2': ('Analytic sine_lat',   _build_analytic_sine),
    '3': ('Analytic wave4/DE3',  _build_analytic_wave4),
    '4': ('HWM14 quiet-time',    _build_hwm14),
    '5': ('HWM14 storm/DWM07',   _build_storm),
}

WIND_MAP_TAGS: dict[str, str] = {
    '1': 'uniform',
    '2': 'sine_lat',
    '3': 'wave4',
    '4': 'hwm14',
    '5': 'storm',
}
```

**Builder function pattern:**

```python
def _build_uniform(rng, h_target_km, v_zonal_ms=100.0, v_merid_ms=0.0):
    from src.windmap.nb00_wind_map_2026_04_06 import UniformWindMap
    return UniformWindMap(v_zonal_ms=v_zonal_ms, v_merid_ms=v_merid_ms)

def _build_analytic_sine(rng, h_target_km, A_zonal_ms=200.0, A_merid_ms=100.0):
    from src.windmap.nb00_wind_map_2026_04_06 import AnalyticWindMap
    return AnalyticWindMap(pattern='sine_lat',
                           A_zonal_ms=A_zonal_ms, A_merid_ms=A_merid_ms)

def _build_analytic_wave4(rng, h_target_km,
                          A_zonal_ms=150.0, A_merid_ms=75.0, phase_rad=0.0):
    from src.windmap.nb00_wind_map_2026_04_06 import AnalyticWindMap
    return AnalyticWindMap(pattern='wave4', A_zonal_ms=A_zonal_ms,
                           A_merid_ms=A_merid_ms, phase_rad=phase_rad)

def _build_hwm14(rng, h_target_km, day_of_year=172, ut_hours=12.0,
                 f107=150.0, ap=4.0):
    from src.windmap.nb00_wind_map_2026_04_06 import HWM14WindMap
    return HWM14WindMap(alt_km=h_target_km, day_of_year=int(day_of_year),
                        ut_hours=ut_hours, f107=f107, ap=ap)

def _build_storm(rng, h_target_km, day_of_year=355, ut_hours=3.0,
                 f107=180.0, ap=80.0):
    from src.windmap.nb00_wind_map_2026_04_06 import StormWindMap
    return StormWindMap(alt_km=h_target_km, day_of_year=int(day_of_year),
                        ut_hours=ut_hours, f107=f107, ap=ap)
```

**`_build_wind_map(choice, rng, h_target_km, **user_params) → WindMap`:**

```python
def _build_wind_map(choice: str, rng, h_target_km: float, **user_params):
    """
    Construct a WindMap from a registry key and user-supplied parameters.
    Raises KeyError for unknown choice; ImportError if hwm14 not installed.
    """
    label, builder = WIND_MAP_REGISTRY[choice]
    return builder(rng, h_target_km, **user_params)
```

**To add a new wind map type** (e.g. TIEGCM-backed):
1. Write `_build_tiegcm(rng, h_target_km, **kw) → WindMap`.
2. Add `'6': ('TIEGCM', _build_tiegcm)` to `WIND_MAP_REGISTRY`.
3. Add `'6': 'tiegcm'` to `WIND_MAP_TAGS`.
4. Add a row to the sub-prompts table in §2.2.
No other changes required anywhere in the script.

---

## 4. Orbit propagation and CONOPS scheduling

### 4.1 Two-tier propagation

NB01 at `SCHED_DT_S = 10.0` s always; user cadence → step size:

```python
SCHED_DT_S       = 10.0
step             = max(1, round(obs_cadence_s / SCHED_DT_S))
actual_cadence_s = step * SCHED_DT_S
```

```python
from src.geometry.nb01_orbit_propagator_2026_04_16 import propagate_orbit

df_sched = propagate_orbit(
    t_start    = t_start,
    duration_s = duration_days * 86400.0,
    dt_s       = SCHED_DT_S,
)
```

Orbit number and look mode (INT01 method):

```python
a_m       = WGS84_A_M + altitude_km * 1e3
T_ORBIT_S = 2 * np.pi * np.sqrt(a_m**3 / EARTH_GRAV_PARAM_M3_S2)
t0        = pd.Timestamp(t_start, tz='UTC')
df_sched['elapsed_s']    = (df_sched['epoch'] - t0).dt.total_seconds()
df_sched['orbit_number'] = (df_sched['elapsed_s'] // T_ORBIT_S).astype(int) + 1
df_sched['look_mode']    = df_sched['orbit_number'].apply(
    lambda n: 'along_track' if n % 2 == 1 else 'cross_track'
)
```

### 4.2 Science frame selection

*(CONOPS reference: [TBD — §TBD science band and cadence])*

```python
science_indices = []
in_band, band_entry_i = False, None

for i, row in df_sched.iterrows():
    if abs(row.lat_deg) <= lat_band_deg:
        if not in_band:
            in_band, band_entry_i = True, i
        if (i - band_entry_i) % step == 0:
            science_indices.append(i)
    else:
        in_band, band_entry_i = False, None
```

Science instrument state: `gpio = [0,0,0,0]`, `lamp = [0,0,0,0,0,0]`.

### 4.3 Calibration and dark trigger

*(CONOPS reference: [TBD — §TBD calibration schedule])*

```python
CAL_TRIGGER_LAT_DEG = 60.0   # CONOPS fixed parameter — see header citation

cal_trigger_indices = []
lat = df_sched['lat_deg'].values
for i in range(1, len(lat)):
    if (lat[i] > CAL_TRIGGER_LAT_DEG
            and lat[i-1] <= CAL_TRIGGER_LAT_DEG
            and lat[i] > lat[i-1]):
        cal_trigger_indices.append(i)
```

Cal sequence: `t₀ + k·step` for `k = 0..n−1`. Dark: `t₀ + (n+k)·step`.
Skip indices ≥ `len(df_sched)`.

Cal state: `gpio = [0,1,1,0]`, `lamp = [1,1,1,1,1,1]`.
Dark state: `gpio = [1,0,0,1]`, `lamp = [0,0,0,0,0,0]`.

### 4.4 `img_type` derivation

```python
def _classify_img_type(lamp_ch_array, gpio_pwr_on):
    """P01 §2.5 logic — keep in sync with p01_image_metadata_2026_04_06.py."""
    if any(lamp_ch_array):
        return "cal"
    elif gpio_pwr_on[0] == 1 and gpio_pwr_on[3] == 1:
        return "dark"
    return "science"
```

### 4.5 Schedule assembly

```python
cal_dark_set  = set(cal_indices) | set(dark_indices)
science_final = [i for i in science_indices if i not in cal_dark_set]
obs_indices   = sorted(science_final + cal_indices + dark_indices)
```

`n_complete_orbits = len(cal_trigger_indices)` — used for C10/C11.
`frame_sequence`: 0-based among all observation frames per orbit, by epoch.

---

## 5. Main metadata loop — per-frame computation

```python
from src.geometry.nb02a_boresight_2026_04_16 import compute_los_eci
from src.geometry.nb02b_tangent_point_2026_04_16 import compute_tangent_point

metadata_list = []

for frame_i, (idx, frame_type) in enumerate(zip(obs_indices, frame_types)):
    row       = df_sched.loc[idx]
    pos       = np.array([row.pos_eci_x, row.pos_eci_y, row.pos_eci_z])
    vel       = np.array([row.vel_eci_x, row.vel_eci_y, row.vel_eci_z])
    look_mode = row.look_mode
    epoch_t   = Time(row.epoch)

    # ── NB02a: attitude quaternion + LOS vector ───────────────────────────
    los_eci, q = compute_los_eci(pos, vel, look_mode, h_target_km=h_target_km)
    # los_eci is retained (not discarded) — needed by NB02b for science frames

    # ── RNG draws (fixed order — do not change) ───────────────────────────
    theta        = rng.normal(0.0, SIGMA_POINTING_RAD)          # 1 draw
    raw          = rng.standard_normal(3)                        # 3 draws
    n_hat        = raw / np.linalg.norm(raw)
    half_t       = theta / 2.0
    qe           = [n_hat[0]*np.sin(half_t), n_hat[1]*np.sin(half_t),
                    n_hat[2]*np.sin(half_t), np.cos(half_t)]
    qe           = [c / np.linalg.norm(qe) for c in qe]
    etalon_temps = rng.normal(ETALON_TEMP_MEAN_C, ETALON_TEMP_STD_C, 4)  # 4 draws
    ccd_temp1    = float(rng.normal(CCD_TEMP_MEAN_C, CCD_TEMP_STD_C))    # 1 draw

    # ── NB02b + NB00: tangent point + truth wind (science frames only) ────
    tp_lat = tp_lon = tp_alt = v_zonal = v_merid = None
    if frame_type == 'science':
        tp     = compute_tangent_point(pos, los_eci, epoch_t, h_target_km=h_target_km)
        tp_lat = tp['tp_lat_deg']
        tp_lon = tp['tp_lon_deg']
        tp_alt = tp['tp_alt_km']
        v_zonal, v_merid = wind_map.sample(tp_lat, tp_lon)

    # ── Instrument state, ADCS flag, ImageMetadata ────────────────────────
    gpio, lamp, exp_time_cs = _instrument_state(frame_type)
    adcs_flag = compute_adcs_quality_flag({
        'pointing_error': qe,
        'pos_eci_hat':    pos.tolist(),
        'adcs_timestamp': int(row.epoch.timestamp() * 1000),
    })
    meta = ImageMetadata(
        # ... all fields per §6 table ...
        tangent_lat        = tp_lat,
        tangent_lon        = tp_lon,
        tangent_alt_km     = tp_alt,
        truth_v_zonal      = v_zonal,
        truth_v_meridional = v_merid,
        truth_v_los        = None,   # requires NB02c; populated by Z02
        noise_seed         = frame_i,
        is_synthetic       = True,
    )
    metadata_list.append(meta)
```

**Why `truth_v_los` stays `None`:** computing `v_wind_LOS` requires projecting
the wind vector onto the LOS in ECI coordinates, accounting for spacecraft
velocity and Earth rotation (NB02c). G01 provides the un-projected physical
wind; Z02 runs the full NB02c chain when embedding the wind into fringe
patterns.

---

## 6. `ImageMetadata` field assignment

Fields newly populated in v8 are marked **NEW**.

| Field | Source | Value / formula |
|-------|--------|-----------------|
| `rows` | constant | `260` |
| `cols` | constant | `276` |
| `exp_time` | §4.5 | `round(exp_time_cts × 0.001 × 100)` centiseconds |
| `exp_unit` | §4.5 | `38500` (hardware register, fixed) |
| `binning` | constant | `2` |
| `img_type` | §4.4 | `_classify_img_type(lamp_ch_array, gpio_pwr_on)` |
| `lua_timestamp` | NB01 `epoch` | `int(row.epoch.timestamp() * 1000)` ms |
| `adcs_timestamp` | NB01 `epoch` | `= lua_timestamp` |
| `utc_timestamp` | NB01 `epoch` | `row.epoch.isoformat()` |
| `spacecraft_latitude` | NB01 `lat_deg` | `np.radians(row.lat_deg)` rad |
| `spacecraft_longitude` | NB01 `lon_deg` | `np.radians(row.lon_deg)` rad |
| `spacecraft_altitude` | NB01 `alt_km` | `row.alt_km * 1e3` m |
| `pos_eci_hat` | NB01 `pos_eci_*` | `[pos_eci_x, pos_eci_y, pos_eci_z]` m |
| `vel_eci_hat` | NB01 `vel_eci_*` | `[vel_eci_x, vel_eci_y, vel_eci_z]` m/s |
| `attitude_quaternion` | NB02a `compute_los_eci()` | `q`, scalar-last `[x,y,z,w]` |
| `pointing_error` | noise §4.1 | Gaussian error quaternion, σ = 5 arcsec |
| `obs_mode` | NB01 parity | `'along_track'` / `'cross_track'` |
| `ccd_temp1` | noise | `rng.normal(-10.0, 1.0)` °C |
| `etalon_temps` | noise | `rng.normal(24.0, 0.1, 4).tolist()` °C |
| `shutter_status` | §4.4 | derived from `gpio_pwr_on` |
| `gpio_pwr_on` | §4.2–4.3 | frame-type dependent |
| `lamp_ch_array` | §4.2–4.3 | frame-type dependent |
| `lamp1/2/3_status` | P01 rule | derived from `lamp_ch_array` |
| `orbit_number` | §4.1 | 1-based from elapsed time |
| `frame_sequence` | §4.5 | 0-based within orbit observation list |
| `orbit_parity` | §4.1 | `'along_track'` / `'cross_track'` |
| `adcs_quality_flag` | P01 | `compute_adcs_quality_flag(...)` |
| `is_synthetic` | constant | `True` |
| `noise_seed` | frame index | 0-based in `obs_indices` |
| `tangent_lat` **NEW** | NB02b (science) | `tp['tp_lat_deg']`; `None` for cal/dark |
| `tangent_lon` **NEW** | NB02b (science) | `tp['tp_lon_deg']`; `None` for cal/dark |
| `tangent_alt_km` **NEW** | NB02b (science) | `tp['tp_alt_km']`; `None` for cal/dark |
| `truth_v_zonal` **NEW** | NB00 (science) | `wind_map.sample(...)[0]`; `None` for cal/dark |
| `truth_v_meridional` **NEW** | NB00 (science) | `wind_map.sample(...)[1]`; `None` for cal/dark |
| `truth_v_los` | — | `None` — NB02c; populated by Z02 |
| `etalon_gap_mm` | — | `None` |
| All other Optional fields | — | `None` / defaults |

---

## 7. Output files

### 7.1 Naming convention

```
{output_dir}/GEN01_{t_start_compact}_{duration_days:05.1f}d_{windmap_tag}_seed{rng_seed:04d}
```

`windmap_tag` from `WIND_MAP_TAGS[choice]`, e.g. `uniform`, `sine_lat`,
`wave4`, `hwm14`, `storm`. Example (defaults):

```
C:\Users\sewell\WindCube\G01_outputs\GEN01_20270101_030.0d_uniform_seed0042.npy
C:\Users\sewell\WindCube\G01_outputs\GEN01_20270101_030.0d_uniform_seed0042.csv
```

### 7.2 `.npy` — complete `ImageMetadata` object array (primary)

Full `ImageMetadata` for every frame, including all newly populated tangent
point and wind fields for science frames. All downstream pipeline modules
load from this file.

```python
records = [dataclasses.asdict(m) for m in metadata_list]
np.save(npy_path, np.array(records, dtype=object), allow_pickle=True)
```

Load: `meta_list = [ImageMetadata(**r) for r in np.load(path, allow_pickle=True)]`

### 7.3 `.csv` — binary-header + tangent/wind columns, 42 columns

**Columns 1–38** are the binary-header equivalent (unchanged from v6).
**Columns 39–42** are the four new tangent point and wind fields. All rows
have the same 42-column schema; `NaN` is written for cal/dark rows where
the fields are `None` in `ImageMetadata`.

| # | Column name | P01 field | Science | Cal / Dark |
|---|-------------|-----------|---------|-----------|
| 39 | `tp_lat_deg` | `tangent_lat` | NB02b geodetic lat, deg | `NaN` |
| 40 | `tp_lon_deg` | `tangent_lon` | NB02b geodetic lon, deg | `NaN` |
| 41 | `wind_v_zonal_ms` | `truth_v_zonal` | NB00 zonal wind at TP, m/s | `NaN` |
| 42 | `wind_v_merid_ms` | `truth_v_meridional` | NB00 meridional wind at TP, m/s | `NaN` |

```python
# Append to each row dict in the CSV construction loop:
row_dict.update({
    'tp_lat_deg':      d['tangent_lat']       if d['tangent_lat']       is not None else float('nan'),
    'tp_lon_deg':      d['tangent_lon']        if d['tangent_lon']        is not None else float('nan'),
    'wind_v_zonal_ms': d['truth_v_zonal']      if d['truth_v_zonal']      is not None else float('nan'),
    'wind_v_merid_ms': d['truth_v_meridional'] if d['truth_v_meridional'] is not None else float('nan'),
})
```

**What is deliberately absent from the CSV** (recovered from 38 header columns):
`img_type`, `binning`, `utc_timestamp`, `shutter_status`, `lamp1/2/3_status`,
`obs_mode`, `orbit_number`, `frame_sequence`, `orbit_parity`,
`adcs_quality_flag`, `is_synthetic`, `noise_seed`, `truth_v_los`,
`etalon_gap_mm`, all dark provenance fields, `grafana_record_id`.

---

## 8. Noise model and instrument constants

### 8.1 Pointing error quaternion

`θ ~ N(0, σ_θ)` where `σ_θ = 5 arcsec × (π/648000) ≈ 2.4241 × 10⁻⁵ rad`.

Observable magnitude |θ| follows a half-normal:
`E[|θ|] ≈ 3.99 arcsec`, `Std[|θ|] ≈ 3.01 arcsec`.

P01 `SLEW_IN_PROGRESS` threshold: 30 arcsec (6σ) — essentially never triggered.

### 8.2 Etalon temperatures

`rng.normal(24.0, 0.1, size=4)` °C per frame.

### 8.3 CCD temperature

`rng.normal(-10.0, 1.0)` °C per frame.

### 8.4 RNG draw order per frame (fixed — do not change)

```
1. theta        1 draw — rng.normal(0, SIGMA_POINTING_RAD)
2. axis raw     3 draws — rng.standard_normal(3)
3. etalon temps 4 draws — rng.normal(24.0, 0.1, 4)
4. ccd_temp1    1 draw — rng.normal(-10.0, 1.0)
```

Total: 9 draws per frame. NB02b and NB00 calls are deterministic (no RNG draws).

### 8.5 Exposure time and hardware register

```python
TIMER_PERIOD_S    = 0.001   # 1 ms per count — hardware constant
EXP_UNIT_REGISTER = 38500   # hardware timing register — always fixed
```

`exp_time` (centiseconds) = `round(exp_time_cts × 0.001 × 100)`.
`exp_unit` = `38500` always — not derived from `exp_time_cts`.

---

## 9. Progress reporting and verification

### 9.1 Console output

```
Parameters:
  ...
  Wind map       : Uniform constant  (v_zonal=100 m/s, v_merid=0 m/s)
  Exp. time      : 8000 counts × 1.0 ms/count = 8.000 s  (exp_unit=38500)
  ...

Building NB01 orbit schedule ... done

Observation schedule:
  Science frames : 115860
  Cal frames     : 2275    (5 per orbit × ~455 complete orbits)
  Dark frames    : 2275    (5 per orbit × ~455 complete orbits)
  Total frames   : 120410

Building NB02a + NB02b + NB00 metadata ...
  [====================] 120410/120410
  (NB02b tangent point + NB00 wind called for 115860 science frames)

Image type verification:
  science : 115860
  cal     : 2275
  dark    : 2275

ADCS quality flags (P01):
  GOOD             : 120410
  SLEW_IN_PROGRESS : 0   (expected ~0)

Pointing error stats (arcsec):
  Mean  :   3.99   (expected ~3.99, half-normal)
  Std   :   3.02   (expected ~3.01, half-normal)
  Max   :   21.2

Etalon temperature stats (°C):
  Mean  : 24.00   (expected ~24.00)
  Std   :  0.10   (expected ~ 0.10)

CCD temperature stats (°C):
  Mean  : -10.00   (expected ~-10.00)
  Std   :   1.00   (expected ~  1.00)

Output files:
  ...GEN01_20270101_030.0d_uniform_seed0042.npy  (XX MB — full ImageMetadata)
  ...GEN01_20270101_030.0d_uniform_seed0042.csv  (XX MB — 42-column export)

G01 complete.
```

### 9.2 Verification checks (C1–C17)

Non-blocking — print `PASS` or `FAIL — <reason>`. File column: `[.npy]` or `[.csv]`.

| Check | Criterion | File |
|-------|-----------|------|
| C1 | NB01 sched rows ≈ `duration_s / SCHED_DT_S` ± 2 | — |
| C2 | All `orbit_parity` ∈ `{'along_track', 'cross_track'}` | `.npy` |
| C3 | No NaN in `att_q_x/y/z/w` | `.csv` |
| C4 | Attitude quaternions unit norm to 1e-6 | `.csv` |
| C5 | Pointing error quaternions unit norm to 1e-6 | `.csv` |
| C6 | Mean unsigned PE within 20% of `σ·√(2/π)` ≈ 3.99 arcsec | `.csv` pe_q_* |
| C7 | Etalon mean within 0.05°C of 24.0°C | `.csv` |
| C8 | P01 `adcs_quality_flag == 0` for > 99.9% | `.npy` |
| C9 | `img_type` ∈ `{'science','cal','dark'}` via `_classify_img_type(gpio, lamp)` | `.csv` |
| C10 | Cal count ≈ `n_caldark × len(cal_trigger_indices)` ±5% | `.npy` |
| C11 | Dark count ≈ `n_caldark × len(cal_trigger_indices)` ±5% | `.npy` |
| C12 | CSV has exactly **42** columns | `.csv` |
| C13 | `.npy` round-trip: `ImageMetadata(**np.load(...)[0])` succeeds | `.npy` |
| C14 | `tp_lat_deg` is `NaN` for all cal/dark rows | `.csv` |
| C15 | `tp_lat_deg` is non-NaN for all science rows | `.csv` |
| C16 | `tp_lat_deg` within `lat_band_deg + 5°` of 0° for science rows | `.csv` |
| C17 | `wind_v_zonal_ms` is non-NaN for all science rows | `.csv` |

**C10/C11:** Use `len(cal_trigger_indices)` (actual detected triggers), not
`df_sched['orbit_number'].max()` — the terminal partial orbit may not have
reached 60°N before the campaign window closed.

**C16:** Tangent points lead the spacecraft by ~8–10° in the along-track
direction. Science frames are taken at `|sc_lat| ≤ lat_band_deg`, so
tangent points should lie within roughly `lat_band_deg + 10°` of the equator.
The 5° tolerance is conservative. A tangent point outside this range indicates
a bug in NB02b or the frame-type assignment.

---

## 10. File location in repository

```
soc_sewell/
├── validation/
│   ├── gen01_synthetic_metadata_generator_2026_04_16.py
│   └── outputs/   (or user-selected folder outside repo)
└── docs/specs/
    └── G01_synthetic_metadata_generator_2026-04-16.md
```

---

## 11. Instructions for Claude Code

### Preamble
```bash
cat PIPELINE_STATUS.md
```

### Prerequisite reads
1. This spec in full.
2. NB00 spec / `nb00_wind_map_2026_04_06.py` — `WindMap` ABC; `UniformWindMap`,
   `AnalyticWindMap` (patterns `sine_lat`, `wave4`), `HWM14WindMap`,
   `StormWindMap`; `sample(lat_deg, lon_deg) → (v_zonal_ms, v_merid_ms)`.
3. NB01 spec — `propagate_orbit` columns.
4. NB02a spec — `compute_los_eci` returns `(los_eci, q)`; pass `h_target_km`.
   **`los_eci` must be retained** — it is the input to NB02b.
5. NB02b spec / `nb02b_tangent_point_2026_04_16.py` — `compute_tangent_point
   (pos_eci, los_eci, epoch, h_target_km)` → dict with `tp_lat_deg`,
   `tp_lon_deg`, `tp_alt_km`, `tp_eci`.
6. P01 spec — `ImageMetadata` fields; `tangent_lat`, `tangent_lon`,
   `tangent_alt_km`, `truth_v_zonal`, `truth_v_meridional`.
7. `CLAUDE.md` at repo root.

### Prerequisite tests
```bash
pytest tests/test_nb01_orbit_propagator.py -v    # 8/8
pytest tests/test_nb02_geometry_2026_04_16.py -v # 10/10
pytest tests/test_s19_p01_metadata.py -v         # 8/8
```

### Constants block
```python
SCHED_DT_S            = 10.0
CAL_TRIGGER_LAT_DEG   = 60.0
SIGMA_POINTING_ARCSEC =  5.0
SIGMA_POINTING_RAD    = SIGMA_POINTING_ARCSEC * np.pi / 648000.0
ETALON_TEMP_MEAN_C    = 24.0
ETALON_TEMP_STD_C     =  0.1
CCD_TEMP_MEAN_C       = -10.0
CCD_TEMP_STD_C        =   1.0
TIMER_PERIOD_S        =  0.001
EXP_UNIT_REGISTER     = 38500
```

### Critical rules
- **`los_eci` retained** from `compute_los_eci()` — pass to `compute_tangent_point()`.
- **NB02b + NB00 called only for `frame_type == 'science'`**.
- **`truth_v_los = None`** — do not attempt to compute it in G01.
- **Wind map tag in filename** — `WIND_MAP_TAGS[choice]` between duration and seed.
- **`exp_time` = `round(exp_time_cts × 0.001 × 100)` centiseconds.**
- **`exp_unit` = `EXP_UNIT_REGISTER = 38500` always.**
- **C10/C11**: `len(cal_trigger_indices)`, not `orbit_number.max()`.
- **C12**: `assert len(df_csv.columns) == 42`.
- **RNG order**: θ(1) → axis(3) → etalon(4) → CCD(1); no RNG in NB02b/NB00.
- **NB01 at `SCHED_DT_S = 10.0` always.**
- **Timezone**: `t0 = pd.Timestamp(t_start, tz='UTC')`.
- **No `build_synthetic_metadata()`** — construct `ImageMetadata` directly.

### Epilogue
```bash
git add PIPELINE_STATUS.md \
        validation/gen01_synthetic_metadata_generator_2026_04_16.py
git commit -m "feat(g01): NB02b tangent point + NB00 wind sampling, C1-C17 pass

Science frames: NB02b tp + NB00.sample() → tangent_lat/lon, truth_v_zonal/merid
Wind map registry: 5 builders (uniform/sine_lat/wave4/hwm14/storm), extensible
CSV: 42 columns (38 header + tp_lat/lon + wind_v_zonal/merid; NaN cal/dark)
exp_time = round(cts × 0.001 × 100) cs; exp_unit = 38500 fixed
C12 → 42 cols; C14-C17 added; all 17 checks pass

Also updates PIPELINE_STATUS.md"
```
