# G01 — Synthetic Metadata Generator Specification

**Spec ID:** G01
**Spec file:** `docs/specs/G01_synthetic_metadata_generator_2026-04-16.md`
**Project:** WindCube FPI Science Operations Center Pipeline
**Institution:** NCAR / High Altitude Observatory (HAO)
**Status:** Draft — awaiting implementation (v8 adds tangent point + wind map)
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
    *(When this CONOPS document is updated, review §3.2–3.3 and §4.5 against
    the new schedule and update the version citation before re-implementing.)*
  - SI-UCAR-WC-RP-004 Issue 1.0 — AOCS Design Report (BRF/THRF/SIRF frames)
  - Drob et al. (2015), doi:10.1002/2014EA000089 — HWM14 empirical model
**Last updated:** 2026-04-16

> **CONOPS version note:** §3 observation schedule parameters and §4.5
> exposure time defaults are CONOPS-driven. Any CONOPS update requires a
> dated revision of this spec before Claude Code is re-run.

> **Revision history:**
> - v1–v6 (2026-04-16): See previous revision notes.
> - v7 (2026-04-16): Output directory via `tkinter` folder-browser dialog;
>   `_pick_folder()` helper; default path outside repo.
> - v8 (2026-04-16): **Tangent point geometry and wind sampling added for
>   science frames.** NB02b `compute_tangent_point()` called for every science
>   epoch. NB00 `wind_map.sample()` called at the tangent point to populate
>   `tangent_lat`, `tangent_lon`, `truth_v_zonal`, `truth_v_meridional` in
>   `ImageMetadata`. Wind map type selected interactively via an extensible
>   registry. Cal and dark frames leave these four fields `None`. CSV
>   extended from 38 to 42 columns (4 new columns, `NaN` for cal/dark rows).

---

## 1. Purpose

G01 is a standalone, interactive Python script that pre-computes and saves
the complete AOCS/instrument metadata array for a synthetic WindCube FPI
observation campaign. Every downstream image synthesis module (Z02, Z03, dark
frames) consumes these `ImageMetadata` records rather than re-running the
geometry pipeline independently.

As of v8, G01 also computes the tangent point position and samples the chosen
truth wind map at that location for every science frame, providing the ground-
truth wind vector that Z02 will later embed into synthetic fringe patterns.

**What G01 does for science frames (v8):**
1. NB02a → attitude quaternion `q` and LOS unit vector `los_eci`
2. NB02b → tangent point `(tp_lat_deg, tp_lon_deg, tp_alt_km, tp_eci)`
3. NB00 → `wind_map.sample(tp_lat_deg, tp_lon_deg)` → `(v_zonal_ms, v_merid_ms)`
4. All four values stored in `ImageMetadata` as `tangent_lat`, `tangent_lon`,
   `truth_v_zonal`, `truth_v_meridional`

**What G01 does not do:**
- Does not compute `v_rel` or `v_wind_LOS` (NB02c). Those require the
  spacecraft velocity projection and are computed by Z02/Z03.
- Does not generate pixel data.
- Does not call P01 `build_synthetic_metadata()`.

---

## 2. User interface — interactive prompts

All numeric prompts appear first; the wind map selection menu follows; the
folder dialog appears last. Invalid entries are rejected and re-prompted.
Blank input accepts the default.

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

After the wind map choice, additional sub-prompts appear for any parameters
required by that map type (see §2.2).

### 2.1 Numeric parameter table

| Symbol | Description | Default | Type | Valid range |
|--------|-------------|---------|------|-------------|
| `t_start` | Start epoch | `"2027-01-01T00:00:00"` | ISO 8601 UTC | Any valid UTC string |
| `duration_days` | Duration | 30.0 | float | 0.1 – 365.0 |
| `lat_band_deg` | Science band half-width | 40.0 | float | 5.0 – 89.0 |
| `obs_cadence_s` | Observation cadence | 10.0 | float | 10.0 – 3600.0 |
| `n_caldark` | Cal/dark frames per trigger | 5 | int | 1 – 50 |
| `exp_time_cts` | Exposure time in timer counts | 8000 | int | 100 – 100000 |
| `altitude_km` | S/C altitude | 510.0 | float | 400.0 – 700.0 |
| `h_target_km` | Tangent height | 250.0 | float | 100.0 – 400.0 |
| `rng_seed` | NumPy RNG seed | 42 | int | ≥ 0 |

### 2.2 Wind map sub-prompts

After the menu choice, display only the sub-prompts relevant to that map:

| Choice | Sub-prompts (default) |
|--------|----------------------|
| 1 — Uniform | `v_zonal [m/s, default 100]`, `v_merid [m/s, default 0]` |
| 2 — Analytic sine_lat | `A_zonal [m/s, default 200]`, `A_merid [m/s, default 100]` |
| 3 — Analytic wave4 | `A_zonal [m/s, default 150]`, `A_merid [m/s, default 75]`, `phase [rad, default 0.0]` |
| 4 — HWM14 quiet | `day_of_year [default 172]`, `ut_hours [default 12.0]`, `f107 [default 150.0]`, `ap [default 4]` |
| 5 — HWM14 storm | `day_of_year [default 355]`, `ut_hours [default 3.0]`, `f107 [default 180.0]`, `ap [default 80]` |

For choices 4 and 5, if `hwm14` is not importable, print an error and
re-display the menu. Do not crash.

### 2.3 Output folder

After the wind map is constructed (and verified importable), print:
```
Select output folder (dialog opening — check taskbar if not visible)...
```
and call `_pick_folder()`. If the user cancels, use the default
`C:\Users\sewell\WindCube\G01_outputs`. Echo the chosen path.

**Constraint warnings** (console only, not errors):
- `lat_band_deg ≥ 60.0`: science band overlaps cal/dark trigger.
- `2 × n_caldark × obs_cadence_s > 1200`: sequence may extend past 60°N arc.

---

## 3. Wind map registry — extensible design

The wind map menu is driven by a registry dict defined at module level.
Adding a new map type requires only adding one entry to `WIND_MAP_REGISTRY`
— no changes to the prompt loop, the metadata loop, or the CSV writer.

```python
# ---------------------------------------------------------------------------
# WIND_MAP_REGISTRY — extend here to add new wind map types.
#
# Each entry:
#   key   : menu number string (keep sequential)
#   value : (display_label, builder_function)
#
# builder_function signature: (rng, h_target_km, **user_params) -> WindMap
#   rng          : numpy.random.Generator — available if builder needs
#                  stochastic initialisation (not currently used)
#   h_target_km  : float — altitude for which the map is valid
#   **user_params: keyword args collected from sub-prompts for this entry
#
# NB00 class → registry entry mapping:
#   UniformWindMap   → key '1'
#   AnalyticWindMap  → keys '2' (sine_lat) and '3' (wave4)
#   HWM14WindMap     → key '4'
#   StormWindMap     → key '5'
# ---------------------------------------------------------------------------

WIND_MAP_REGISTRY: dict[str, tuple[str, callable]] = {
    '1': ('Uniform constant',     _build_uniform),
    '2': ('Analytic sine_lat',    _build_analytic_sine),
    '3': ('Analytic wave4/DE3',   _build_analytic_wave4),
    '4': ('HWM14 quiet-time',     _build_hwm14),
    '5': ('HWM14 storm/DWM07',    _build_storm),
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

    Parameters
    ----------
    choice       : str — key from WIND_MAP_REGISTRY (e.g. '1', '4')
    rng          : numpy.random.Generator
    h_target_km  : float — tangent height, passed to all builders
    **user_params: sub-prompt values collected by the interactive prompt loop

    Returns
    -------
    WindMap instance (any NB00 subclass)

    Raises
    ------
    KeyError    : unknown choice
    ImportError : hwm14 not installed (choices '4' and '5')
    """
    label, builder = WIND_MAP_REGISTRY[choice]
    return builder(rng, h_target_km, **user_params)
```

**To add a new wind map type** (e.g. a TIEGCM-backed map):
1. Write `_build_tiegcm(rng, h_target_km, **kw) → WindMap`.
2. Add `'6': ('TIEGCM', _build_tiegcm)` to `WIND_MAP_REGISTRY`.
3. Add a row to the sub-prompts table in §2.2.
No other changes required.

---

## 4. Orbit propagation and CONOPS scheduling

*(Unchanged from v7 — §3.1–3.5 of that version, renumbered here.)*

### 4.1 Two-tier propagation

NB01 at `SCHED_DT_S = 10.0` s always. User cadence → step size:

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

```python
CAL_TRIGGER_LAT_DEG = 60.0

cal_trigger_indices = []
lat = df_sched['lat_deg'].values
for i in range(1, len(lat)):
    if (lat[i] > CAL_TRIGGER_LAT_DEG
            and lat[i-1] <= CAL_TRIGGER_LAT_DEG
            and lat[i] > lat[i-1]):
        cal_trigger_indices.append(i)
```

Cal sequence: `t₀ + k·step` for `k = 0..n−1`. Dark: `t₀ + (n+k)·step`.
Cal state: `gpio = [0,1,1,0]`, `lamp = [1,1,1,1,1,1]`.
Dark state: `gpio = [1,0,0,1]`, `lamp = [0,0,0,0,0,0]`.

### 4.4 Schedule assembly

```python
cal_dark_set  = set(cal_indices) | set(dark_indices)
science_final = [i for i in science_indices if i not in cal_dark_set]
obs_indices   = sorted(science_final + cal_indices + dark_indices)
```

`n_complete_orbits = len(cal_trigger_indices)`.

### 4.5 `img_type` derivation

```python
def _classify_img_type(lamp_ch_array, gpio_pwr_on):
    if any(lamp_ch_array):
        return "cal"
    elif gpio_pwr_on[0] == 1 and gpio_pwr_on[3] == 1:
        return "dark"
    return "science"
```

---

## 5. Main metadata loop — per-frame computation

The loop iterates over `zip(obs_indices, frame_types)`. For every frame,
NB02a is called to get `(los_eci, q)`. For science frames only, NB02b
and NB00 are additionally called.

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

    # ── RNG draws (fixed order per frame) ─────────────────────────────────
    theta  = rng.normal(0.0, SIGMA_POINTING_RAD)         # 1 draw
    raw    = rng.standard_normal(3)                       # 3 draws
    n_hat  = raw / np.linalg.norm(raw)
    half_t = theta / 2.0
    qe     = [n_hat[0]*np.sin(half_t), n_hat[1]*np.sin(half_t),
              n_hat[2]*np.sin(half_t), np.cos(half_t)]
    qe     = [c / np.linalg.norm(qe) for c in qe]
    etalon_temps = rng.normal(ETALON_TEMP_MEAN_C, ETALON_TEMP_STD_C, 4)  # 4 draws
    ccd_temp1    = float(rng.normal(CCD_TEMP_MEAN_C, CCD_TEMP_STD_C))    # 1 draw

    # ── NB02b + NB00: tangent point and truth wind (science frames only) ──
    tp_lat = tp_lon = tp_alt = v_zonal = v_merid = None
    if frame_type == 'science':
        tp = compute_tangent_point(pos, los_eci, epoch_t, h_target_km=h_target_km)
        tp_lat  = tp['tp_lat_deg']
        tp_lon  = tp['tp_lon_deg']
        tp_alt  = tp['tp_alt_km']
        v_zonal, v_merid = wind_map.sample(tp_lat, tp_lon)

    # ── Instrument state ──────────────────────────────────────────────────
    gpio, lamp, exp_time_cs = _instrument_state(frame_type)

    # ── ADCS quality flag (P01) ───────────────────────────────────────────
    adcs_flag = compute_adcs_quality_flag({
        'pointing_error': qe,
        'pos_eci_hat':    pos.tolist(),
        'adcs_timestamp': int(row.epoch.timestamp() * 1000),
    })

    # ── Construct ImageMetadata ───────────────────────────────────────────
    meta = ImageMetadata(
        # ... (all fields as per §6 field table) ...
        tangent_lat         = tp_lat,
        tangent_lon         = tp_lon,
        tangent_alt_km      = tp_alt,
        truth_v_zonal       = v_zonal,
        truth_v_meridional  = v_merid,
        # truth_v_los remains None — requires NB02c (computed by Z02)
        noise_seed          = frame_i,
        is_synthetic        = True,
    )
    metadata_list.append(meta)
```

**Why `truth_v_los` stays `None` in G01:** `v_wind_LOS` requires projecting
the wind vector onto the LOS direction at the tangent point epoch, which
involves the spacecraft velocity and Earth rotation terms (NB02c). G01
deliberately stops at the un-projected wind components; Z02 will compute
`truth_v_los` using the full NB02c chain when it generates the fringe pattern.

---

## 6. `ImageMetadata` field assignment

Fields newly populated in v8 are marked **NEW**.

| Field | Source | Value / formula |
|-------|--------|-----------------|
| `rows` | constant | `260` |
| `cols` | constant | `276` |
| `exp_time` | §4.5 | `round(exp_time_cts × TIMER_PERIOD_S × 100)` cs |
| `exp_unit` | §4.5 | `38500` (hardware register, fixed) |
| `binning` | constant | `2` |
| `img_type` | §4.5 | `_classify_img_type(lamp_ch_array, gpio_pwr_on)` |
| `lua_timestamp` | NB01 `epoch` | `int(row.epoch.timestamp() * 1000)` ms |
| `adcs_timestamp` | NB01 `epoch` | `= lua_timestamp` |
| `utc_timestamp` | NB01 `epoch` | `row.epoch.isoformat()` |
| `spacecraft_latitude` | NB01 `lat_deg` | `np.radians(row.lat_deg)` rad |
| `spacecraft_longitude` | NB01 `lon_deg` | `np.radians(row.lon_deg)` rad |
| `spacecraft_altitude` | NB01 `alt_km` | `row.alt_km * 1e3` m |
| `pos_eci_hat` | NB01 `pos_eci_*` | `[pos_eci_x, pos_eci_y, pos_eci_z]` m |
| `vel_eci_hat` | NB01 `vel_eci_*` | `[vel_eci_x, vel_eci_y, vel_eci_z]` m/s |
| `attitude_quaternion` | NB02a `compute_los_eci()` | `q`, scalar-last `[x,y,z,w]` |
| `pointing_error` | §4.1 noise | Gaussian error quaternion, σ = 5 arcsec |
| `obs_mode` | NB01 orbit parity | `'along_track'` / `'cross_track'` |
| `ccd_temp1` | noise | `rng.normal(-10.0, 1.0)` °C |
| `etalon_temps` | noise | `rng.normal(24.0, 0.1, 4).tolist()` °C |
| `shutter_status` | §4.5 | derived from `gpio_pwr_on` |
| `gpio_pwr_on` | §4.2–4.3 | frame-type dependent |
| `lamp_ch_array` | §4.2–4.3 | frame-type dependent |
| `lamp1/2/3_status` | P01 rule | derived from `lamp_ch_array` |
| `orbit_number` | §4.1 | 1-based from elapsed time |
| `frame_sequence` | §4.4 | 0-based within orbit observation list |
| `orbit_parity` | §4.1 | `'along_track'` / `'cross_track'` |
| `adcs_quality_flag` | P01 | `compute_adcs_quality_flag(...)` |
| `is_synthetic` | constant | `True` |
| `noise_seed` | frame index | 0-based in `obs_indices` |
| `tangent_lat` **NEW** | NB02b (science only) | `tp['tp_lat_deg']`; `None` for cal/dark |
| `tangent_lon` **NEW** | NB02b (science only) | `tp['tp_lon_deg']`; `None` for cal/dark |
| `tangent_alt_km` **NEW** | NB02b (science only) | `tp['tp_alt_km']`; `None` for cal/dark |
| `truth_v_zonal` **NEW** | NB00 (science only) | `wind_map.sample(tp_lat, tp_lon)[0]`; `None` for cal/dark |
| `truth_v_meridional` **NEW** | NB00 (science only) | `wind_map.sample(tp_lat, tp_lon)[1]`; `None` for cal/dark |
| `truth_v_los` | — | `None` — requires NB02c; populated by Z02 |
| `etalon_gap_mm` | — | `None` |
| All other Optional fields | — | `None` / defaults |

---

## 7. Output files

### 7.1 Naming convention

```
{output_dir}/GEN01_{t_start_compact}_{duration_days:05.1f}d_{windmap_tag}_seed{rng_seed:04d}
```

`windmap_tag` is a short lowercase label from the registry, e.g.
`uniform`, `sine_lat`, `wave4`, `hwm14`, `storm`. Example (defaults):

```
C:\Users\sewell\WindCube\G01_outputs\GEN01_20270101_030.0d_uniform_seed0042.npy
C:\Users\sewell\WindCube\G01_outputs\GEN01_20270101_030.0d_uniform_seed0042.csv
```

Add `windmap_tag` as a constant in the constants block:
```python
WIND_MAP_TAGS = {
    '1': 'uniform', '2': 'sine_lat', '3': 'wave4',
    '4': 'hwm14',   '5': 'storm',
}
```
When adding a new registry entry, add the corresponding tag here.

### 7.2 `.npy` — complete `ImageMetadata` object array (primary)

Unchanged from v6. Full `ImageMetadata` for every frame, including all
newly populated tangent point and wind fields for science frames.

```python
records = [dataclasses.asdict(m) for m in metadata_list]
np.save(npy_path, np.array(records, dtype=object), allow_pickle=True)
```

### 7.3 `.csv` — binary-header + tangent/wind columns, 42 columns

The CSV is extended from 38 to **42 columns** by appending four new columns.
Columns 1–38 are unchanged (see v6 §6.3). The four new columns are:

| # | Column name | P01 field | Science frames | Cal / Dark |
|---|-------------|-----------|----------------|-----------|
| 39 | `tp_lat_deg` | `tangent_lat` | NB02b geodetic latitude, deg | `NaN` |
| 40 | `tp_lon_deg` | `tangent_lon` | NB02b geodetic longitude, deg | `NaN` |
| 41 | `wind_v_zonal_ms` | `truth_v_zonal` | NB00 zonal wind at TP, m/s | `NaN` |
| 42 | `wind_v_merid_ms` | `truth_v_meridional` | NB00 meridional wind at TP, m/s | `NaN` |

**CSV construction addition** (append to the `rows_out` dict in the loop):

```python
rows_out.append({
    # ... columns 1–38 unchanged ...
    'tp_lat_deg':       d['tangent_lat']        if d['tangent_lat']        is not None else float('nan'),
    'tp_lon_deg':       d['tangent_lon']         if d['tangent_lon']         is not None else float('nan'),
    'wind_v_zonal_ms':  d['truth_v_zonal']       if d['truth_v_zonal']       is not None else float('nan'),
    'wind_v_merid_ms':  d['truth_v_meridional']  if d['truth_v_meridional']  is not None else float('nan'),
})
```

**C12 updates:** `assert len(df_csv.columns) == 42` (was 38).

---

## 8. Progress reporting and verification

### 8.1 Console output additions (v8)

```
Wind map       : Uniform constant  (v_zonal=100 m/s, v_merid=0 m/s)

...

Building NB02a attitude quaternions + NB02b tangent points + NB00 wind sampling ...
  [====================] 120410/120410
  (NB02b + NB00 called for 115860 science frames only)
```

### 8.2 Verification checks

| Check | Criterion | File | v8 change? |
|-------|-----------|------|-----------|
| C1 | NB01 sched rows ≈ `duration_s / SCHED_DT_S` ± 2 | — | No |
| C2 | All `orbit_parity` valid | `.npy` | No |
| C3 | No NaN in `att_q_*` | `.csv` | No |
| C4 | Attitude quaternions unit norm to 1e-6 | `.csv` | No |
| C5 | PE quaternions unit norm to 1e-6 | `.csv` | No |
| C6 | Mean unsigned PE within 20% of `σ·√(2/π)` ≈ 3.99 arcsec | `.csv` | No |
| C7 | Etalon mean within 0.05°C of 24.0°C | `.csv` | No |
| C8 | P01 `adcs_quality_flag == 0` for > 99.9% | `.npy` | No |
| C9 | `img_type` valid (P01 from csv gpio/lamp) | `.csv` | No |
| C10 | Cal count ≈ `n_caldark × n_complete_orbits` ±5% | `.npy` | No |
| C11 | Dark count ≈ `n_caldark × n_complete_orbits` ±5% | `.npy` | No |
| C12 | CSV has exactly **42** columns | `.csv` | **Updated (was 38)** |
| C13 | `.npy` round-trip succeeds | `.npy` | No |
| C14 | `tp_lat_deg` is `NaN` for all cal/dark rows | `.csv` | **NEW** |
| C15 | `tp_lat_deg` is non-NaN for all science rows | `.csv` | **NEW** |
| C16 | `tp_lat_deg` within `lat_band_deg + 5°` of 0° (science rows) | `.csv` | **NEW** |
| C17 | `wind_v_zonal_ms` is non-NaN for all science rows | `.csv` | **NEW** |

**C16 rationale:** The tangent point leads the spacecraft by ~923 km
(~8–10° in latitude depending on inclination). Science frames are taken
at `|sc_lat| ≤ lat_band_deg`, so tangent points should lie within
roughly `lat_band_deg + 10°` of the equator. The 5° margin accommodates
the geometry. A tangent point far outside this band indicates a bug in
NB02b or the frame-type assignment.

---

## 9. File location in repository

```
soc_sewell/
├── validation/
│   ├── gen01_synthetic_metadata_generator_2026_04_16.py
│   └── outputs/   (or user-selected folder outside repo)
└── docs/specs/
    └── G01_synthetic_metadata_generator_2026-04-16.md
```

---

## 10. Instructions for Claude Code

### Preamble
```bash
cat PIPELINE_STATUS.md
```

### Prerequisite reads
1. This spec in full.
2. NB00 spec and `nb00_wind_map_2026_04_06.py` — `WindMap` abstract base,
   `UniformWindMap`, `AnalyticWindMap`, `HWM14WindMap`, `StormWindMap`;
   the `sample(lat_deg, lon_deg) → (v_zonal_ms, v_merid_ms)` interface.
3. NB01 spec — `propagate_orbit` output columns.
4. NB02a spec — `compute_los_eci` returns `(los_eci, q)`; pass `h_target_km`.
   **`los_eci` must now be retained** (not discarded) for passing to NB02b.
5. NB02b spec and `nb02b_tangent_point_2026_04_16.py` — `compute_tangent_point
   (pos_eci, los_eci, epoch, h_target_km)` returns dict with keys
   `tp_eci`, `tp_lat_deg`, `tp_lon_deg`, `tp_alt_km`.
6. P01 spec — `ImageMetadata` fields; particularly `tangent_lat`,
   `tangent_lon`, `tangent_alt_km`, `truth_v_zonal`, `truth_v_meridional`.
7. `CLAUDE.md` at repo root.

### Prerequisite tests
```bash
pytest tests/test_nb01_orbit_propagator.py -v    # 8/8
pytest tests/test_nb02_geometry_2026_04_16.py -v # 10/10
pytest tests/test_s19_p01_metadata.py -v         # 8/8
```

### Additional imports
```python
from src.windmap.nb00_wind_map_2026_04_06 import (
    WindMap, UniformWindMap, AnalyticWindMap, HWM14WindMap, StormWindMap,
)
from src.geometry.nb02b_tangent_point_2026_04_16 import compute_tangent_point
```

### Key implementation rules

- **`los_eci` must be kept**, not discarded, from `compute_los_eci()`.
  It is the input to `compute_tangent_point()`.
- **NB02b is called only for science frames** (`frame_type == 'science'`).
  Do not call it for cal or dark frames.
- **`wind_map.sample()` is called only for science frames**, immediately
  after `compute_tangent_point()`, at `(tp['tp_lat_deg'], tp['tp_lon_deg'])`.
- **`truth_v_los` remains `None`** in G01. Do not attempt to compute it;
  it requires NB02c and is Z02's responsibility.
- **Wind map tag in filename:** `WIND_MAP_TAGS[choice]` inserted between
  duration and seed in the filename stem.
- **C12:** `assert len(df_csv.columns) == 42`.
- **C14–C17:** run after CSV is built; use pandas column operations.
- **RNG draw order unchanged:** θ(1) → axis(3) → etalon(4) → CCD(1).
  NB02b and NB00 calls are deterministic (no RNG draws).
- **`exp_unit` = `38500` always.**
- **NB01 at `SCHED_DT_S = 10.0` always.**
- **Timezone:** `t0 = pd.Timestamp(t_start, tz='UTC')`.

### Epilogue
```bash
git add PIPELINE_STATUS.md \
        validation/gen01_synthetic_metadata_generator_2026_04_16.py
git commit -m "feat(g01): add NB02b tangent point + NB00 wind sampling for science frames

Science frames: NB02b tp + NB00.sample() → tangent_lat/lon, truth_v_zonal/merid
Wind map registry pattern — extensible, add entry to WIND_MAP_REGISTRY
CSV: 42 columns (38 header + 4 new; NaN for cal/dark)
Checks C12 updated (38→42), C14-C17 added
NB00 choices: Uniform, AnalyticSineLat, AnalyticWave4, HWM14, Storm

Also updates PIPELINE_STATUS.md"
```
