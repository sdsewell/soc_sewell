# G01 — Synthetic Metadata Generator Specification

**Spec ID:** G01
**Spec file:** `docs/specs/G01_synthetic_metadata_generator_2026-04-16.md`
**Project:** WindCube FPI Science Operations Center Pipeline
**Institution:** NCAR / High Altitude Observatory (HAO)
**Status:** ✓ Complete — all 21 checks pass; binary synthesis confirmed
**Depends on:**
  - NB00 (`nb00_wind_map_2026_04_06.py`) — `WindMap`, `UniformWindMap`,
    `AnalyticWindMap`, `HWM14WindMap`, `StormWindMap`
  - NB01 (`nb01_orbit_propagator_2026_04_16.py`) — `propagate_orbit(t_start, duration_s, dt_s)`
  - NB02a (`nb02a_boresight_2026_04_16.py`) — `compute_los_eci(pos_eci, vel_eci, look_mode, h_target_km)`
  - NB02b (`nb02b_tangent_point_2026_04_16.py`) — `compute_tangent_point(pos_eci, los_eci, epoch, h_target_km)`
  - NB02c (`nb02c_los_projection_2026_04_16.py`) — `compute_v_rel(wind_map, tp_lat_deg, tp_lon_deg, tp_eci, vel_eci, los_eci, epoch)`
  - P01 (`p01_image_metadata_2026_04_06.py`) — `ImageMetadata`, `AdcsQualityFlags`,
    `compute_adcs_quality_flag()`, `parse_header()` (used in C21 round-trip check)
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
  - Burns, Adams & Longwell (1950) IAU neon spectroscopic standards — λ₁, λ₂
**Last updated:** 2026-04-16

> **CONOPS version note:** §4.2–4.3 observation schedule parameters and §7.1
> exposure time defaults are CONOPS-driven. Any CONOPS update requires a dated
> revision of this spec before Claude Code is re-run.

> **Revision history:**
> - v1 (2026-04-16): Initial — latitude-threshold state, one record per epoch.
> - v2 (2026-04-16): CONOPS model — science cadence, ±lat_band, 60°N trigger.
> - v3 (2026-04-16): S-number references replaced with NB01/NB02a/P01 names;
>   CONOPS document citation added.
> - v4 (2026-04-16): Half-normal PE distribution; C6/C10/C11 corrected;
>   script relocated to `validation/`.
> - v5 (2026-04-16): `exp_time_cts` prompt; `exp_unit = 38500`; CCD temp
>   noise N(−10, 1°C); RNG draw order established.
> - v6 (2026-04-16): CSV reduced to 38 binary-header columns; `.npy` retains
>   full `ImageMetadata`; C12 column-count guard added.
> - v7 (2026-04-16): `tkinter` folder-browser dialog; `_pick_folder()`;
>   default path `C:\Users\sewell\WindCube\G01_outputs`.
> - v8 (2026-04-16): NB02b tangent point + NB00 wind sampling for science
>   frames; wind map registry; CSV 42 columns; C14–C17 added.
> - v9 (2026-04-16): **NB02c LOS decomposition + binary image synthesis.**
>   `compute_v_rel()` replaces direct `wind_map.sample()` call; returns all
>   six LOS velocity components. `truth_v_los` now populated from
>   `vr['v_wind_LOS']`. Four new CSV columns (43–46). Binary image synthesis:
>   `_encode_u64()`, `_encode_f64()`, `_encode_header()` (exact P01 inverses,
>   round-trip verified); `_generate_science_pixels()` (Airy fringe at v_rel),
>   `_generate_cal_pixels()` (two-line neon), `_generate_dark_pixels()`
>   (exponential dark current + read noise); `_write_bin_file()` (143,520-byte
>   P01 layout); `_bin_filename()`. C12 updated (42→46); C18–C21 added.
>   All 21 checks pass; 26 prerequisite tests pass.

---

## 1. Purpose

G01 is a standalone, interactive Python script that pre-computes and saves
the complete AOCS/instrument metadata array **and** a corresponding set of
synthetic binary FPI image files for a WindCube observation campaign.

**What G01 produces per frame type:**

| Frame type | NB01 | NB02a | NB02b | NB02c | Image synthesis |
|------------|------|-------|-------|-------|-----------------|
| science | ✓ orbit state | ✓ `q`, `los_eci` | ✓ tangent point | ✓ full LOS decomposition | Airy fringe at `v_rel` Doppler |
| cal | ✓ orbit state | ✓ `q`, `los_eci` | — | — | Two-line neon (λ₁, λ₂) |
| dark | ✓ orbit state | ✓ `q`, `los_eci` | — | — | Dark current + read noise |

**Storage note:** ~120,000 `.bin` files × 143,520 bytes ≈ **17.3 GB** for a
30-day run at 10 s cadence. Ensure adequate disk space before running.

**What G01 does not produce:**
- `truth_v_los` was `None` in v1–v8; it is now populated from NB02c
  `v_wind_LOS` for science frames.
- Pixel data from complex wind retrieval (v_rel inversion) — that is Z02/Z03.

---

## 2. User interface — interactive prompts

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

| Choice | Sub-prompts (defaults) |
|--------|----------------------|
| 1 — Uniform | `v_zonal [m/s, default 100]`, `v_merid [m/s, default 0]` |
| 2 — Analytic sine_lat | `A_zonal [m/s, default 200]`, `A_merid [m/s, default 100]` |
| 3 — Analytic wave4 | `A_zonal [m/s, default 150]`, `A_merid [m/s, default 75]`, `phase [rad, default 0.0]` |
| 4 — HWM14 quiet | `day_of_year [default 172]`, `ut_hours [default 12.0]`, `f107 [default 150.0]`, `ap [default 4]` |
| 5 — HWM14 storm | `day_of_year [default 355]`, `ut_hours [default 3.0]`, `f107 [default 180.0]`, `ap [default 80]` |

For choices 4 and 5, if `hwm14` is not importable, print an error and
re-display the menu without crashing.

### 2.3 Output folder — `_pick_folder()`

Print `"Select output folder (dialog opening — check taskbar if not visible)..."`
and open `tkinter.filedialog.askdirectory()` with root window hidden and
raised. Default starting location: `C:\Users\sewell\WindCube\G01_outputs`.
Cancel → use the default. Echo chosen path to terminal.

**Constraint warnings** (console only):
- `lat_band_deg ≥ 60.0`: cal/dark trigger within science band; cal/dark wins.
- `2 × n_caldark × obs_cadence_s > 1200`: sequence may span past 60°N arc.

---

## 3. Wind map registry — extensible design

```python
WIND_MAP_REGISTRY: dict[str, tuple[str, callable]] = {
    '1': ('Uniform constant',    _build_uniform),
    '2': ('Analytic sine_lat',   _build_analytic_sine),
    '3': ('Analytic wave4/DE3',  _build_analytic_wave4),
    '4': ('HWM14 quiet-time',    _build_hwm14),
    '5': ('HWM14 storm/DWM07',   _build_storm),
}

WIND_MAP_TAGS: dict[str, str] = {
    '1': 'uniform', '2': 'sine_lat', '3': 'wave4',
    '4': 'hwm14',   '5': 'storm',
}
```

**To add a new map type:** write `_build_X(rng, h_target_km, **kw) → WindMap`,
add `'N': ('Label', _build_X)` to `WIND_MAP_REGISTRY`, add `'N': 'tag'` to
`WIND_MAP_TAGS`, add a row to §2.2. No other changes needed.

Builder signature: `(rng: Generator, h_target_km: float, **user_params) → WindMap`.

---

## 4. Orbit propagation and CONOPS scheduling

### 4.1 Two-tier propagation

NB01 always at `SCHED_DT_S = 10.0` s. User cadence → step:

```python
SCHED_DT_S       = 10.0
step             = max(1, round(obs_cadence_s / SCHED_DT_S))
actual_cadence_s = step * SCHED_DT_S
```

```python
from src.geometry.nb01_orbit_propagator_2026_04_16 import propagate_orbit

df_sched = propagate_orbit(t_start=t_start,
                            duration_s=duration_days * 86400.0,
                            dt_s=SCHED_DT_S)
```

Orbit number and look mode (INT01 method):

```python
a_m       = WGS84_A_M + altitude_km * 1e3
T_ORBIT_S = 2 * np.pi * np.sqrt(a_m**3 / EARTH_GRAV_PARAM_M3_S2)
t0        = pd.Timestamp(t_start, tz='UTC')
df_sched['elapsed_s']    = (df_sched['epoch'] - t0).dt.total_seconds()
df_sched['orbit_number'] = (df_sched['elapsed_s'] // T_ORBIT_S).astype(int) + 1
df_sched['look_mode']    = df_sched['orbit_number'].apply(
    lambda n: 'along_track' if n % 2 == 1 else 'cross_track')
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

Science state: `gpio = [0,0,0,0]`, `lamp = [0,0,0,0,0,0]`.

### 4.3 Calibration and dark trigger

*(CONOPS reference: [TBD — §TBD calibration schedule])*

```python
CAL_TRIGGER_LAT_DEG = 60.0   # CONOPS fixed — see header citation

cal_trigger_indices = []
lat = df_sched['lat_deg'].values
for i in range(1, len(lat)):
    if lat[i] > CAL_TRIGGER_LAT_DEG and lat[i-1] <= CAL_TRIGGER_LAT_DEG \
            and lat[i] > lat[i-1]:
        cal_trigger_indices.append(i)
```

Cal sequence: `t₀ + k·step` for `k = 0..n−1`.
Dark sequence: `t₀ + (n+k)·step` for `k = 0..n−1`.
Skip indices ≥ `len(df_sched)`.

Cal state: `gpio = [0,1,1,0]`, `lamp = [1,1,1,1,1,1]`.
Dark state: `gpio = [1,0,0,1]`, `lamp = [0,0,0,0,0,0]`.

### 4.4 `img_type` derivation

```python
def _classify_img_type(lamp_ch_array, gpio_pwr_on):
    """P01 §2.5 logic — keep in sync with p01_image_metadata_2026_04_06.py."""
    if any(lamp_ch_array):           return "cal"
    elif gpio_pwr_on[0] == 1 \
         and gpio_pwr_on[3] == 1:   return "dark"
    return "science"
```

### 4.5 Schedule assembly

```python
cal_dark_set  = set(cal_indices) | set(dark_indices)
science_final = [i for i in science_indices if i not in cal_dark_set]
obs_indices   = sorted(science_final + cal_indices + dark_indices)
```

`n_complete_orbits = len(cal_trigger_indices)` (used for C10/C11).
`frame_sequence`: 0-based among all observation frames per orbit, by epoch.

---

## 5. Main metadata loop

```python
from src.geometry.nb02a_boresight_2026_04_16   import compute_los_eci
from src.geometry.nb02b_tangent_point_2026_04_16 import compute_tangent_point
from src.geometry.nb02c_los_projection_2026_04_16 import compute_v_rel

bin_dir = pathlib.Path(output_dir) / "bin_frames"
bin_dir.mkdir(parents=True, exist_ok=True)
metadata_list = []

for frame_i, (idx, frame_type) in enumerate(zip(obs_indices, frame_types)):
    row       = df_sched.loc[idx]
    pos       = np.array([row.pos_eci_x, row.pos_eci_y, row.pos_eci_z])
    vel       = np.array([row.vel_eci_x, row.vel_eci_y, row.vel_eci_z])
    epoch_t   = Time(row.epoch)
    look_mode = row.look_mode

    # ── NB02a ─────────────────────────────────────────────────────────────
    # los_eci is retained — required by both NB02b and NB02c
    los_eci, q = compute_los_eci(pos, vel, look_mode, h_target_km=h_target_km)

    # ── Metadata RNG draws (9 total; pixel draws follow after) ────────────
    theta        = rng.normal(0.0, SIGMA_POINTING_RAD)          # draw 1
    raw          = rng.standard_normal(3)                        # draws 2–4
    n_hat        = raw / np.linalg.norm(raw)
    half_t       = theta / 2.0
    qe           = [n_hat[0]*np.sin(half_t), n_hat[1]*np.sin(half_t),
                    n_hat[2]*np.sin(half_t), np.cos(half_t)]
    qe           = [c / np.linalg.norm(qe) for c in qe]
    etalon_temps = rng.normal(ETALON_TEMP_MEAN_C, ETALON_TEMP_STD_C, 4)  # draws 5–8
    ccd_temp1    = float(rng.normal(CCD_TEMP_MEAN_C, CCD_TEMP_STD_C))    # draw 9

    # ── NB02b + NB02c (science only) ──────────────────────────────────────
    tp_lat = tp_lon = tp_alt = None
    v_zonal = v_merid = v_wind_LOS = v_earth_LOS = V_sc_LOS = v_rel = None

    if frame_type == 'science':
        tp = compute_tangent_point(pos, los_eci, epoch_t, h_target_km=h_target_km)
        tp_lat, tp_lon, tp_alt = tp['tp_lat_deg'], tp['tp_lon_deg'], tp['tp_alt_km']

        # NB02c subsumes wind_map.sample() — no separate NB00 call needed
        vr = compute_v_rel(wind_map, tp_lat, tp_lon, tp['tp_eci'],
                           vel, los_eci, epoch_t)
        v_wind_LOS  = vr['v_wind_LOS']
        v_earth_LOS = vr['v_earth_LOS']
        V_sc_LOS    = vr['V_sc_LOS']
        v_rel       = vr['v_rel']
        v_zonal     = vr['v_zonal_ms']    # from wind_map.sample() via NB02c
        v_merid     = vr['v_merid_ms']

    # ── Instrument state and ADCS flag ────────────────────────────────────
    gpio, lamp, exp_time_cs = _instrument_state(frame_type)
    adcs_flag = compute_adcs_quality_flag({
        'pointing_error': qe,
        'pos_eci_hat':    pos.tolist(),
        'adcs_timestamp': int(row.epoch.timestamp() * 1000),
    })

    # ── ImageMetadata ─────────────────────────────────────────────────────
    meta = ImageMetadata(
        # ... all fields per §6 table ...
        truth_v_los        = v_wind_LOS,    # populated for science; None otherwise
        tangent_lat        = tp_lat,
        tangent_lon        = tp_lon,
        tangent_alt_km     = tp_alt,
        truth_v_zonal      = v_zonal,
        truth_v_meridional = v_merid,
        noise_seed         = frame_i,
        is_synthetic       = True,
    )
    metadata_list.append(meta)

    # ── Binary image synthesis and write (§7) ─────────────────────────────
    pixels_256 = _generate_pixels(frame_type, v_rel, ccd_temp1,
                                  exp_time_cts, rng)
    _write_bin_file(meta, pixels_256, bin_dir / _bin_filename(meta))
```

---

## 6. `ImageMetadata` field assignment

| Field | Source | Value / formula |
|-------|--------|-----------------|
| `rows` | constant | `260` |
| `cols` | constant | `276` |
| `exp_time` | §4.5 | `round(exp_time_cts × 0.001 × 100)` centiseconds |
| `exp_unit` | §4.5 | `38500` (hardware register, fixed) |
| `binning` | constant | `2` |
| `img_type` | §4.4 | `_classify_img_type(lamp_ch_array, gpio_pwr_on)` |
| `lua_timestamp` | NB01 `epoch` | `int(row.epoch.timestamp() × 1000)` ms |
| `adcs_timestamp` | NB01 `epoch` | `= lua_timestamp` |
| `utc_timestamp` | NB01 `epoch` | `row.epoch.isoformat()` |
| `spacecraft_latitude` | NB01 `lat_deg` | `np.radians(row.lat_deg)` rad |
| `spacecraft_longitude` | NB01 `lon_deg` | `np.radians(row.lon_deg)` rad |
| `spacecraft_altitude` | NB01 `alt_km` | `row.alt_km × 1e3` m |
| `pos_eci_hat` | NB01 `pos_eci_*` | `[pos_eci_x, pos_eci_y, pos_eci_z]` m |
| `vel_eci_hat` | NB01 `vel_eci_*` | `[vel_eci_x, vel_eci_y, vel_eci_z]` m/s |
| `attitude_quaternion` | NB02a | `q`, scalar-last `[x,y,z,w]` |
| `pointing_error` | noise | Gaussian error quaternion, σ = 5 arcsec |
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
| `tangent_lat` | NB02b (science) | `tp['tp_lat_deg']`; `None` for cal/dark |
| `tangent_lon` | NB02b (science) | `tp['tp_lon_deg']`; `None` for cal/dark |
| `tangent_alt_km` | NB02b (science) | `tp['tp_alt_km']`; `None` for cal/dark |
| `truth_v_zonal` | NB02c (science) | `vr['v_zonal_ms']`; `None` for cal/dark |
| `truth_v_meridional` | NB02c (science) | `vr['v_merid_ms']`; `None` for cal/dark |
| `truth_v_los` | NB02c (science) | `vr['v_wind_LOS']`; `None` for cal/dark |
| `etalon_gap_mm` | — | `None` |
| All other Optional fields | — | `None` / defaults |

---

## 7. Binary image synthesis

### 7.1 Constants

```python
# FPI optical model (S03 authoritative values)
LAMBDA_OI_M      = 630.0e-9       # OI 630.0 nm, m
LAMBDA_NE1_M     = 640.2248e-9    # Neon strong line (Burns et al. 1950)
LAMBDA_NE2_M     = 638.2991e-9    # Neon weak line  (Burns et al. 1950)
ETALON_GAP_M     = 20.106e-3      # Tolansky-recovered, m
FOCAL_LENGTH_M   = 0.19912        # Imaging lens, m
PLATE_SCALE_RPX  = 1.6071e-4      # rad/px (2×2 binned, Tolansky)
R_REFL           = 0.53           # Effective reflectivity (FlatSat)
N_GAP            = 1.0            # Refractive index, air gap
C_LIGHT_MS       = 2.99792458e8   # m/s
FINESSE_F        = 4*R_REFL / (1-R_REFL)**2    # ≈ 9.6

# CCD / pixel layout
NX_PIX, NY_PIX   = 256, 256       # science region (2×2 binned)
N_ROWS_BIN       = 259            # pixel rows in binary file
N_COLS_BIN       = 276            # pixel columns in binary file
ROW_OFFSET_PIX   = 1              # top-left row of science window in pixel array
COL_OFFSET_PIX   = 10             # top-left col of science window
BIAS_ADU         = 100            # CCD bias, ADU
ADU_MAX          = 16383          # 14-bit ceiling

# Frame signal levels
SCI_PEAK_ADU     = 5000           # OI 630 nm fringe peak
SCI_READ_NOISE   = 5.0            # ADU rms
CAL_PEAK_ADU     = 12000          # Neon bright fringe peak
CAL_NE_RATIO     = 3.0            # Strong λ1 : weak λ2 intensity ratio
CAL_READ_NOISE   = 5.0
DARK_REF_ADU_S   = 0.05           # Dark current at T_REF_DARK_C, ADU/px/s
T_REF_DARK_C     = -20.0          # Reference temperature for dark model, °C
T_DOUBLE_C       = 6.5            # Dark current doubling interval, °C
DARK_READ_NOISE  = 5.0
```

### 7.2 Mixed-endian encoding helpers

Exact inverses of P01 `_u64()` and `_f64()`. Round-trip requirement:
`_f64(np.array(_encode_f64(x), dtype=">u2"), 0) == x` for all finite float64.

```python
def _encode_u64(val: int) -> list[int]:
    """uint64 → 4 BE uint16 words in LE word order (LSW first)."""
    return [(val >> (16 * i)) & 0xFFFF for i in range(4)]

def _encode_f64(val: float) -> list[int]:
    """float64 → 4 BE uint16 words in LE word order (LSW first)."""
    b     = struct.pack(">d", val)        # 8 bytes, big-endian double
    words = struct.unpack(">4H", b)       # [MSW, w1, w2, LSW]
    return list(reversed(words))          # [LSW, w2, w1, MSW] — LE word order
```

### 7.3 Header encoder — `_encode_header()`

Exact inverse of P01 `parse_header()`. Returns `np.ndarray` shape `(276,)`
`dtype=">u2"`. Quaternion convention: pipeline `[x,y,z,w]` → binary `[w,x,y,z]`
(applied to both `attitude_quaternion` and `pointing_error`).

```python
def _encode_header(meta: ImageMetadata) -> np.ndarray:
    h = np.zeros(276, dtype=">u2")
    h[0], h[1], h[2], h[3] = meta.rows, meta.cols, meta.exp_time, meta.exp_unit
    for i, w in enumerate(_encode_f64(meta.ccd_temp1)):        h[4 + i]  = w
    for i, w in enumerate(_encode_u64(meta.lua_timestamp)):    h[8 + i]  = w
    for i, w in enumerate(_encode_u64(meta.adcs_timestamp)):   h[12 + i] = w
    for j, val in enumerate([meta.spacecraft_latitude,
                              meta.spacecraft_longitude,
                              meta.spacecraft_altitude]):
        for i, w in enumerate(_encode_f64(val)):               h[16 + j*4 + i] = w
    # attitude_quaternion: [x,y,z,w] → [w,x,y,z] for binary
    q = meta.attitude_quaternion
    for j, val in enumerate([q[3], q[0], q[1], q[2]]):
        for i, w in enumerate(_encode_f64(val)):               h[28 + j*4 + i] = w
    # pointing_error: same reorder
    qe = meta.pointing_error
    for j, val in enumerate([qe[3], qe[0], qe[1], qe[2]]):
        for i, w in enumerate(_encode_f64(val)):               h[44 + j*4 + i] = w
    for j, val in enumerate(meta.pos_eci_hat):
        for i, w in enumerate(_encode_f64(val)):               h[60 + j*4 + i] = w
    for j, val in enumerate(meta.vel_eci_hat):
        for i, w in enumerate(_encode_f64(val)):               h[72 + j*4 + i] = w
    for j, val in enumerate(meta.etalon_temps):
        for i, w in enumerate(_encode_f64(val)):               h[84 + j*4 + i] = w
    for j, val in enumerate(meta.gpio_pwr_on):                 h[100 + j] = int(val) & 0xFF
    for j, val in enumerate(meta.lamp_ch_array):               h[104 + j] = int(val) & 0xFF
    # Words 110–275: padding (zeros)
    return h
```

### 7.4 Pixel image generators

All return `np.ndarray` shape `(256, 256)` `dtype=np.uint16`, clipped to
`[0, ADU_MAX]`. The 256×256 array is embedded in the 259×276 binary pixel
area by `_write_bin_file()`.

#### Science — Airy fringe with Doppler shift

```python
def _generate_science_pixels(v_rel_ms: float, rng) -> np.ndarray:
    """
    OI 630 nm Airy fringe pattern with Doppler shift v_rel_ms.
    λ_obs = LAMBDA_OI_M × (1 + v_rel_ms / C_LIGHT_MS)
    Positive v_rel (recession) → λ increases → fringes shift inward.
    δ(r) = 4π·N_GAP·ETALON_GAP_M·cos(r·PLATE_SCALE_RPX) / λ_obs
    I_airy = 1 / (1 + FINESSE_F·sin²(δ/2))
    Signal = SCI_PEAK_ADU × I_airy + Poisson(photon) + Normal(0, READ_NOISE) + BIAS
    """
    lambda_obs = LAMBDA_OI_M * (1.0 + v_rel_ms / C_LIGHT_MS)
    x, y = np.arange(NX_PIX) - NX_PIX/2.0, np.arange(NY_PIX) - NY_PIX/2.0
    XX, YY = np.meshgrid(x, y)
    theta  = np.sqrt(XX**2 + YY**2) * PLATE_SCALE_RPX
    delta  = 4.0 * np.pi * N_GAP * ETALON_GAP_M * np.cos(theta) / lambda_obs
    I_airy = 1.0 / (1.0 + FINESSE_F * np.sin(delta / 2.0)**2)
    signal = SCI_PEAK_ADU * I_airy
    image  = rng.poisson(np.clip(signal, 0, None)).astype(float) \
             + rng.normal(0.0, SCI_READ_NOISE, signal.shape) + BIAS_ADU
    return np.clip(np.round(image), 0, ADU_MAX).astype(np.uint16)
```

#### Calibration — two-line neon

```python
def _generate_cal_pixels(rng) -> np.ndarray:
    """
    Two superimposed Airy patterns at λ1=640.2248 nm (strong) and
    λ2=638.2991 nm (weak), ratio CAL_NE_RATIO:1 (Burns et al. 1950).
    """
    x, y = np.arange(NX_PIX) - NX_PIX/2.0, np.arange(NY_PIX) - NY_PIX/2.0
    XX, YY = np.meshgrid(x, y)
    theta  = np.sqrt(XX**2 + YY**2) * PLATE_SCALE_RPX

    def _airy(lam):
        d = 4.0 * np.pi * N_GAP * ETALON_GAP_M * np.cos(theta) / lam
        return 1.0 / (1.0 + FINESSE_F * np.sin(d / 2.0)**2)

    I_cal  = (CAL_NE_RATIO * _airy(LAMBDA_NE1_M) + _airy(LAMBDA_NE2_M)) \
             / (CAL_NE_RATIO + 1.0)
    signal = CAL_PEAK_ADU * I_cal
    image  = rng.poisson(np.clip(signal, 0, None)).astype(float) \
             + rng.normal(0.0, CAL_READ_NOISE, signal.shape) + BIAS_ADU
    return np.clip(np.round(image), 0, ADU_MAX).astype(np.uint16)
```

#### Dark — exponential dark current model

```python
def _generate_dark_pixels(ccd_temp1_c: float, exp_time_s: float,
                           rng) -> np.ndarray:
    """
    Dark current rate doubles every T_DOUBLE_C = 6.5°C.
    dark_rate [ADU/px/s] = DARK_REF_ADU_S × 2^((T - T_REF_DARK_C) / T_DOUBLE_C)
    mean_dark = dark_rate × exp_time_s
    N_px ~ Poisson(mean_dark) + Normal(0, READ_NOISE) + BIAS_ADU
    """
    dark_rate = DARK_REF_ADU_S * 2.0**((ccd_temp1_c - T_REF_DARK_C) / T_DOUBLE_C)
    mean_dark = max(dark_rate * exp_time_s, 0.0)
    image = rng.poisson(mean_dark, size=(NY_PIX, NX_PIX)).astype(float) \
            + rng.normal(0.0, DARK_READ_NOISE, size=(NY_PIX, NX_PIX)) + BIAS_ADU
    return np.clip(np.round(image), 0, ADU_MAX).astype(np.uint16)
```

#### Dispatcher

```python
def _generate_pixels(frame_type, v_rel_ms, ccd_temp1_c,
                     exp_time_cts, rng) -> np.ndarray:
    exp_time_s = exp_time_cts * TIMER_PERIOD_S
    if   frame_type == 'science': return _generate_science_pixels(v_rel_ms, rng)
    elif frame_type == 'cal':     return _generate_cal_pixels(rng)
    elif frame_type == 'dark':    return _generate_dark_pixels(ccd_temp1_c, exp_time_s, rng)
    raise ValueError(f"Unknown frame_type: {frame_type!r}")
```

### 7.5 Binary file writer — `_write_bin_file()`

```python
def _write_bin_file(meta: ImageMetadata,
                    pixels_256: np.ndarray,
                    path: pathlib.Path) -> None:
    """
    P01 §2.1 binary layout: 260 × 276 × 2 = 143,520 bytes.
    Row 0:     276-word header  (_encode_header).
    Rows 1–259: 259 × 276 pixel region; 256×256 science window at
               [ROW_OFFSET_PIX:ROW_OFFSET_PIX+256, COL_OFFSET_PIX:COL_OFFSET_PIX+256];
               all other pixels = BIAS_ADU.
    """
    header      = _encode_header(meta)
    pixel_array = np.full((N_ROWS_BIN, N_COLS_BIN), BIAS_ADU, dtype=np.uint16)
    pixel_array[ROW_OFFSET_PIX:ROW_OFFSET_PIX + NY_PIX,
                COL_OFFSET_PIX:COL_OFFSET_PIX + NX_PIX] = pixels_256
    full_array  = np.vstack([header.reshape(1, N_COLS_BIN),
                             pixel_array.astype(">u2")]).astype(">u2")
    assert full_array.shape  == (260, 276)
    assert full_array.nbytes == 143_520
    path.write_bytes(full_array.tobytes())
```

### 7.6 Filename — `_bin_filename()`

```python
def _bin_filename(meta: ImageMetadata) -> str:
    """YYYYMMDDThhmmssZ_{img_type}.bin  (no colons — Windows compatible)."""
    from datetime import datetime, timezone
    dt = datetime.fromtimestamp(meta.lua_timestamp / 1000.0, tz=timezone.utc)
    return f"{dt.strftime('%Y%m%dT%H%M%SZ')}_{meta.img_type}.bin"
```

### 7.7 RNG draw order — complete per-frame sequence (fixed)

```
Metadata draws (9 total) — always first:
  draw 1    theta              rng.normal(0, SIGMA_POINTING_RAD)
  draws 2–4 axis raw           rng.standard_normal(3)
  draws 5–8 etalon temps       rng.normal(24.0, 0.1, 4)
  draw 9    ccd_temp1          rng.normal(-10.0, 1.0)

Pixel synthesis draws — always after draw 9:
  [science]  Poisson  65536    rng.poisson(signal, size=(256,256))
  [science]  Normal   65536    rng.normal(0, SCI_READ_NOISE, (256,256))
  [cal]      Poisson  65536    rng.poisson(signal, size=(256,256))
  [cal]      Normal   65536    rng.normal(0, CAL_READ_NOISE, (256,256))
  [dark]     Poisson  65536    rng.poisson(mean_dark, size=(256,256))
  [dark]     Normal   65536    rng.normal(0, DARK_READ_NOISE, (256,256))
```

Do not change this order. Metadata reproducibility is independent of
pixel synthesis changes so long as metadata draws precede pixel draws.

---

## 8. Output files

### 8.1 Naming convention

```
{output_dir}/
├── GEN01_{t_start_compact}_{duration_days:05.1f}d_{windmap_tag}_seed{rng_seed:04d}.npy
├── GEN01_{t_start_compact}_{duration_days:05.1f}d_{windmap_tag}_seed{rng_seed:04d}.csv
└── bin_frames/
    ├── 20270101T000000Z_science.bin
    ├── 20270101T000010Z_science.bin
    ...
    ├── 20270101T013420Z_cal.bin
    └── 20270101T013510Z_dark.bin
```

### 8.2 `.npy` — primary pipeline format

Full `ImageMetadata` for every frame, including `truth_v_los` (now populated).

### 8.3 `.csv` — 46 columns

Columns 1–42 unchanged from v8. Four new columns (43–46):

| # | Column name | P01 field | Science | Cal / Dark |
|---|-------------|-----------|---------|-----------|
| 43 | `v_wind_los_ms` | `truth_v_los` | NB02c `v_wind_LOS`, m/s | `NaN` |
| 44 | `v_earth_los_ms` | — (CSV only) | NB02c `v_earth_LOS`, m/s | `NaN` |
| 45 | `v_sc_los_ms` | — (CSV only) | NB02c `V_sc_LOS`, m/s | `NaN` |
| 46 | `v_rel_ms` | — (CSV only) | NB02c `v_rel`, m/s | `NaN` |

`V_sc_LOS`, `v_earth_LOS`, and `v_rel` have no dedicated `ImageMetadata`
fields; they live in the CSV only.

---

## 9. Progress reporting and verification

### 9.1 Console output (additions from v9)

```
Building NB02a + NB02b + NB02c + image synthesis ...
  [====================] 120410/120410
  NB02b+NB02c called for 115860 science frames.
  .bin files → bin_frames/ (120410 files, ~17.3 GB)

v_rel stats for science frames (m/s):
  Mean : XXXX.X   (along-track dominated by V_sc_LOS ≈ -7100 m/s)
  Std  :  XXX.X

v_wind_LOS stats (m/s):
  Mean :  XXX.X
  Std  :   XX.X
```

### 9.2 Verification checks (C1–C21)

| Check | Criterion | File |
|-------|-----------|------|
| C1 | NB01 sched rows ≈ `duration_s / SCHED_DT_S` ± 2 | — |
| C2 | All `orbit_parity` valid | `.npy` |
| C3 | No NaN in `att_q_*` | `.csv` |
| C4 | Attitude quaternions unit norm to 1e-6 | `.csv` |
| C5 | PE quaternions unit norm to 1e-6 | `.csv` |
| C6 | Mean unsigned PE within 20% of `σ·√(2/π)` ≈ 3.99 arcsec | `.csv` |
| C7 | Etalon mean within 0.05°C of 24.0°C | `.csv` |
| C8 | P01 `adcs_quality_flag == 0` for > 99.9% | `.npy` |
| C9 | `img_type` valid via `_classify_img_type(gpio, lamp)` | `.csv` |
| C10 | Cal count ≈ `n_caldark × len(cal_trigger_indices)` ±5% | `.npy` |
| C11 | Dark count ≈ `n_caldark × len(cal_trigger_indices)` ±5% | `.npy` |
| C12 | CSV has exactly **46** columns | `.csv` |
| C13 | `.npy` round-trip: `ImageMetadata(**np.load(...)[0])` | `.npy` |
| C14 | `tp_lat_deg` is `NaN` for all cal/dark rows | `.csv` |
| C15 | `tp_lat_deg` is non-NaN for all science rows | `.csv` |
| C16 | `tp_lat_deg` within `lat_band_deg + 5°` of 0° for science rows | `.csv` |
| C17 | `wind_v_zonal_ms` is non-NaN for all science rows | `.csv` |
| C18 | `v_rel_ms` is non-NaN for all science rows | `.csv` |
| C19 | `v_rel_ms` is `NaN` for all cal/dark rows | `.csv` |
| C20 | All `.bin` files exist; size = 143,520 bytes; count = `len(metadata_list)` | `bin_frames/` |
| C21 | Header round-trip: `parse_header()` on first science `.bin`; `lua_timestamp` matches CSV | `bin_frames/` |

**C21 implementation:**
```python
first_sci = next(f for f in sorted(bin_dir.glob("*.bin")) if 'science' in f.name)
hdr       = np.frombuffer(first_sci.read_bytes(), dtype=">u2")[:276]
d         = parse_header(hdr)
sci_row   = df_csv[df_csv.apply(
    lambda r: _classify_img_type(
        [int(r[f'lamp_{i}']) for i in range(6)],
        [int(r[f'gpio_{i}']) for i in range(4)]) == 'science', axis=1)].iloc[0]
c21 = (d['lua_timestamp'] == int(sci_row['lua_timestamp']))
```

---

## 10. File location in repository

```
soc_sewell/
├── validation/
│   ├── gen01_synthetic_metadata_generator_2026_04_16.py
│   └── (outputs in user-selected folder outside repo)
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
2. NB02c spec / `nb02c_los_projection_2026_04_16.py` — `compute_v_rel()`.
   Note capital `V` in `V_sc_LOS` key.
3. P01 `parse_header()` and `_f64()` / `_u64()` — verify your encode
   inverses produce exact round-trips before use.
4. All previous prerequisite reads (NB00, NB01, NB02a, NB02b, P01, `CLAUDE.md`).

### Prerequisite tests
```bash
pytest tests/test_nb01_orbit_propagator.py -v    # 8/8
pytest tests/test_nb02_geometry_2026_04_16.py -v # 10/10
pytest tests/test_s19_p01_metadata.py -v         # 8/8
```

### Critical rules
- **`los_eci` retained** from NB02a — input to both NB02b and NB02c.
- **NB02c replaces `wind_map.sample()`** — no separate NB00 call for science.
  `truth_v_los = vr['v_wind_LOS']` (was `None` in v8).
- **`tp['tp_eci']`** (ndarray) is the `tp_eci` arg to `compute_v_rel()`.
- **`_encode_f64` round-trip must be exact** before use; assert it once.
- **Quaternion reorder**: `[x,y,z,w]` → `[w,x,y,z]` for both quaternion
  fields in `_encode_header()`.
- **C12 = 46 columns.**
- **RNG order**: 9 metadata draws first; pixel draws after. Never interleave.
- **`bin_dir`** = `{output_dir}/bin_frames/`; create once before the loop.
- **`exp_unit` = `38500` always;** `exp_time` = `round(cts × 0.001 × 100)` cs.
- **NB01 at `SCHED_DT_S = 10.0` always.**
- **Timezone**: `t0 = pd.Timestamp(t_start, tz='UTC')`.

### Epilogue
```bash
git add PIPELINE_STATUS.md \
        validation/gen01_synthetic_metadata_generator_2026_04_16.py
git commit -m "docs: update G01 spec to v9 (NB02c + binary synthesis confirmed)

All 21 checks pass; 26 prerequisite tests pass.
truth_v_los populated; CSV 46 columns; .bin synthesis verified.

Also updates PIPELINE_STATUS.md"
```
