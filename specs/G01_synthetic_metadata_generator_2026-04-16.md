# G01 — Synthetic Metadata Generator Specification

**Spec ID:** G01
**Spec file:** `docs/specs/G01_synthetic_metadata_generator_2026-04-16.md`
**Project:** WindCube FPI Science Operations Center Pipeline
**Institution:** NCAR / High Altitude Observatory (HAO)
**Status:** ✓ Complete — all 13 checks pass (30-day run confirmed)
**Depends on:**
  - NB01 (`nb01_orbit_propagator_2026_04_16.py`) — `propagate_orbit(t_start, duration_s, dt_s)`
  - NB02a (`nb02a_boresight_2026_04_16.py`) — `compute_los_eci(pos_eci, vel_eci, look_mode, h_target_km)`
  - P01 (`p01_image_metadata_2026_04_06.py`) — `ImageMetadata`, `AdcsQualityFlags`,
    `compute_adcs_quality_flag()`
  - `src/constants.py` — `WGS84_A_M`, `EARTH_GRAV_PARAM_M3_S2`
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
**Last updated:** 2026-04-16

> **CONOPS version note:** §3 observation schedule parameters (science band,
> cal trigger latitude, sequence structure) and §5 exposure time defaults are
> CONOPS-driven. Any CONOPS update requires a dated revision of this spec
> before Claude Code is re-run.

> **Revision history:**
> - v1 (2026-04-16): Initial — latitude-threshold state, one record per epoch.
> - v2 (2026-04-16): CONOPS model — science cadence, ±lat_band, 60°N trigger.
> - v3 (2026-04-16): S-number references replaced with NB01/NB02a/P01 names;
>   CONOPS document citation added.
> - v4 (2026-04-16): Half-normal PE distribution; C6/C10/C11 corrected;
>   script relocated to `validation/`.
> - v5 (2026-04-16): `exp_time_cts` prompt; `exp_unit = 38500`; CCD temp
>   noise N(−10, 1°C); RNG draw order fixed.
> - v6 (2026-04-16): CSV format revised. The `.csv` now contains exactly
>   **38 columns** — the 17 binary-header fields expanded to scalars — and
>   no derived, calculated, or None-valued pipeline fields. The `.npy` retains
>   the complete `ImageMetadata` objects for all downstream pipeline use.

---

## 1. Purpose

G01 is a standalone, interactive Python script that pre-computes and saves
the complete AOCS/instrument metadata array for a synthetic WindCube FPI
observation campaign. Every downstream image synthesis module (Z02, Z03, dark
frames) consumes these `ImageMetadata` records rather than re-running the
geometry pipeline independently.

**WindCube CONOPS as modelled by G01 (see CONOPS citation in header):**

```
One orbit (≈ 94.8 min, 510 km SSO):

  Ascending node
      │
      ▼ lat increases →
  ┌───────────────────────────────────────────────────────────────┐
  │  SCIENCE BAND  │   gap   │ CAL+DARK │ max lat │  gap  │ SCI  │
  │  0° → +band°   │band→60° │  trigger │  ~82°N  │ desc. │ etc. │
  │  obs_cadence_s │ (idle)  │ n+n frms │         │       │      │
  └───────────────────────────────────────────────────────────────┘
```

- **Science frames:** one frame every `obs_cadence_s` s while `|lat| ≤ lat_band_deg`.
- **Cal/dark sequence:** `n_caldark` cal + `n_caldark` dark frames, triggered
  once per orbit when the spacecraft latitude first ascends through 60.0°N.
- **All other epochs:** no observation, no `ImageMetadata` generated.

**What G01 does not do:** tangent points, v_rel, wind (NB02b/c); pixel data;
calls to P01 `build_synthetic_metadata()` (requires NB02c wind outputs).

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
Output directory     [ default validation/outputs/ ] : _
Random seed          [int,   default  42       ] : _
```

**Parameter table:**

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
| `output_dir` | Output directory | `"validation/outputs/"` | str | Any writable path |
| `rng_seed` | NumPy RNG seed | 42 | int | ≥ 0 |

**Constraint warnings** (console only, not errors):
- `lat_band_deg ≥ 60.0`: science band overlaps cal/dark trigger; cal/dark wins.
- `2 × n_caldark × obs_cadence_s > 1200`: sequence may extend past 60°N arc.

---

## 3. Orbit propagation and CONOPS scheduling

*(All schedule logic derived from the CONOPS document cited in the header.)*

### 3.1 Two-tier propagation

NB01 is always called at `SCHED_DT_S = 10.0` s regardless of `obs_cadence_s`.

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

t0 = pd.Timestamp(t_start, tz='UTC')
df_sched['elapsed_s']    = (df_sched['epoch'] - t0).dt.total_seconds()
df_sched['orbit_number'] = (df_sched['elapsed_s'] // T_ORBIT_S).astype(int) + 1
df_sched['look_mode']    = df_sched['orbit_number'].apply(
    lambda n: 'along_track' if n % 2 == 1 else 'cross_track'
)
```

### 3.2 Science frame selection

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

**Science instrument state:**

| Field | Value |
|-------|-------|
| `gpio_pwr_on` | `[0, 0, 0, 0]` |
| `lamp_ch_array` | `[0, 0, 0, 0, 0, 0]` |

### 3.3 Calibration and dark trigger and sequence

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

Sequence from trigger `t₀`: cal at `t₀ + k·step` for `k = 0..n−1`;
dark at `t₀ + (n+k)·step` for `k = 0..n−1`. Skip indices ≥ `len(df_sched)`.

**Calibration instrument state:**

| Field | Value |
|-------|-------|
| `gpio_pwr_on` | `[0, 1, 1, 0]` |
| `lamp_ch_array` | `[1, 1, 1, 1, 1, 1]` |

**Dark instrument state:**

| Field | Value |
|-------|-------|
| `gpio_pwr_on` | `[1, 0, 0, 1]` |
| `lamp_ch_array` | `[0, 0, 0, 0, 0, 0]` |

### 3.4 `img_type` derivation

```python
def _classify_img_type(lamp_ch_array: list, gpio_pwr_on: list) -> str:
    """P01 §2.5 classification — keep in sync with p01_image_metadata_2026_04_06.py."""
    if any(lamp_ch_array):
        return "cal"
    elif gpio_pwr_on[0] == 1 and gpio_pwr_on[3] == 1:
        return "dark"
    return "science"
```

### 3.5 Schedule assembly and frame sequencing

```python
cal_dark_set  = set(cal_indices) | set(dark_indices)
science_final = [i for i in science_indices if i not in cal_dark_set]
obs_indices   = sorted(science_final + cal_indices + dark_indices)
```

`frame_sequence`: 0-based among all observation frames per orbit, ordered
by epoch. `n_complete_orbits = len(cal_trigger_indices)` (excludes any
terminal partial orbit that did not reach 60°N before the window closed).

---

## 4. Noise model and instrument constants

### 4.1 Pointing error quaternion

Signed rotation angle drawn from N(0, σ_θ):

```
SIGMA_POINTING_ARCSEC = 5.0
σ_θ = 5.0 × (π / 648000) ≈ 2.4241 × 10⁻⁵ rad
```

```python
theta  = rng.normal(0.0, sigma_theta_rad)
raw    = rng.standard_normal(3)
n_hat  = raw / np.linalg.norm(raw)
half_θ = theta / 2.0
qe     = [n_hat[0]*np.sin(half_θ), n_hat[1]*np.sin(half_θ),
          n_hat[2]*np.sin(half_θ), np.cos(half_θ)]
qe     = [c / np.linalg.norm(qe) for c in qe]
```

The observable rotation magnitude |θ| follows a **half-normal** distribution:

```
E[|θ|]   = σ_θ · √(2/π) ≈ 3.99 arcsec
Std[|θ|] = σ_θ · √(1−2/π) ≈ 3.01 arcsec
```

### 4.2 Etalon temperatures

```python
etalon_temps = rng.normal(24.0, 0.1, size=4).tolist()   # °C
```

### 4.3 CCD temperature

Drawn per frame from N(−10.0, 1.0) °C:

```python
ccd_temp1 = float(rng.normal(-10.0, 1.0))
```

### 4.4 RNG draw order per observation frame (fixed — do not change)

```
1. theta       : 1 draw  — rng.normal(0, sigma_theta_rad)
2. axis raw    : 3 draws — rng.standard_normal(3)
3. etalon temps: 4 draws — rng.normal(24.0, 0.1, 4)
4. ccd_temp1   : 1 draw  — rng.normal(-10.0, 1.0)
```

Total: 9 draws per frame. Order is a reproducibility contract.

### 4.5 Exposure time and hardware register

```python
TIMER_PERIOD_S    = 0.001   # 1 ms per count — hardware constant
EXP_UNIT_REGISTER = 38500   # hardware timing register — always fixed
```

`exp_time` (P01 centiseconds field) = `round(exp_time_cts × TIMER_PERIOD_S × 100)`.
`exp_unit` = `38500` always — not derived from `exp_time_cts`.
Same `exp_time_cts` applied to all frame types (science, cal, dark).

---

## 5. `ImageMetadata` field assignment

| Field | Source | Value / formula |
|-------|--------|-----------------|
| `rows` | constant | `260` |
| `cols` | constant | `276` |
| `exp_time` | §4.5 | `round(exp_time_cts × TIMER_PERIOD_S × 100)` cs |
| `exp_unit` | §4.5 | `38500` (hardware register, fixed) |
| `binning` | constant | `2` |
| `img_type` | §3.4 | `_classify_img_type(lamp_ch_array, gpio_pwr_on)` |
| `lua_timestamp` | NB01 `epoch` | `int(row.epoch.timestamp() * 1000)` ms |
| `adcs_timestamp` | NB01 `epoch` | `= lua_timestamp` |
| `utc_timestamp` | NB01 `epoch` | `row.epoch.isoformat()` |
| `spacecraft_latitude` | NB01 `lat_deg` | `np.radians(row.lat_deg)` rad |
| `spacecraft_longitude` | NB01 `lon_deg` | `np.radians(row.lon_deg)` rad |
| `spacecraft_altitude` | NB01 `alt_km` | `row.alt_km * 1e3` m |
| `pos_eci_hat` | NB01 `pos_eci_*` | `[pos_eci_x, pos_eci_y, pos_eci_z]` m |
| `vel_eci_hat` | NB01 `vel_eci_*` | `[vel_eci_x, vel_eci_y, vel_eci_z]` m/s |
| `attitude_quaternion` | NB02a `compute_los_eci()` | `q` return, scalar-last `[x,y,z,w]` |
| `pointing_error` | §4.1 | Gaussian error quaternion, σ = 5 arcsec |
| `obs_mode` | NB01 orbit parity | `'along_track'` / `'cross_track'` |
| `ccd_temp1` | §4.3 | `rng.normal(-10.0, 1.0)` °C |
| `etalon_temps` | §4.2 | `rng.normal(24.0, 0.1, 4).tolist()` °C |
| `shutter_status` | §3.4 | derived from `gpio_pwr_on` |
| `gpio_pwr_on` | §3.2–3.3 | frame-type dependent |
| `lamp_ch_array` | §3.2–3.3 | frame-type dependent |
| `lamp1_status` | P01 rule | `"on" if lamp_ch[0] or lamp_ch[1] else "off"` |
| `lamp2_status` | P01 rule | `"on" if lamp_ch[2] or lamp_ch[3] else "off"` |
| `lamp3_status` | P01 rule | `"on" if lamp_ch[4] or lamp_ch[5] else "off"` |
| `orbit_number` | §3.1 | 1-based from elapsed time |
| `frame_sequence` | §3.5 | 0-based within orbit observation list |
| `orbit_parity` | §3.1 | `'along_track'` / `'cross_track'` |
| `adcs_quality_flag` | P01 `compute_adcs_quality_flag()` | computed from pointing_error |
| `is_synthetic` | constant | `True` |
| `noise_seed` | frame index | 0-based in `obs_indices` |
| All synthetic truth fields | — | `None` (NB02b/c not run) |
| All dark provenance fields | defaults | `False` / `0` / `None` |

---

## 6. Output files

### 6.1 Naming convention

```
{output_dir}/GEN01_{t_start_compact}_{duration_days:05.1f}d_seed{rng_seed:04d}
```

Example (defaults):
```
validation/outputs/GEN01_20270101_030.0d_seed0042.npy   (52.8 MB)
validation/outputs/GEN01_20270101_030.0d_seed0042.csv   (73.9 MB)
```

### 6.2 `.npy` — complete `ImageMetadata` object array (primary)

The `.npy` file is the **authoritative output** consumed by all downstream
pipeline modules. It contains the full `ImageMetadata` dataclass for every
observation frame, including all derived, calculated, and pipeline-added
fields.

```python
records = [dataclasses.asdict(m) for m in metadata_list]
np.save(npy_path, np.array(records, dtype=object), allow_pickle=True)
```

Loading in downstream modules:
```python
records   = np.load(npy_path, allow_pickle=True)
meta_list = [ImageMetadata(**r) for r in records]
```

### 6.3 `.csv` — binary-header equivalent, 38 columns (human-readable export)

The `.csv` contains exactly **38 columns** corresponding to the 17 binary
header fields of the on-orbit image format (P01 §2.3), with list fields
expanded into named scalar columns. It is the synthetic equivalent of what
the real binary header produces at ingest.

**Design rationale:** Derived fields (`img_type`, `binning`, `utc_timestamp`,
`shutter_status`, `lamp1/2/3_status`, `obs_mode`, `orbit_*`,
`adcs_quality_flag`) are reproducibly computable from the 38 header columns
via P01 at any time and are therefore not stored in the CSV. Pipeline-added
fields (`is_synthetic`, `noise_seed`, all `None`-valued truth and provenance
fields) belong only in the `.npy`. This keeps the CSV compact, externally
readable, and structurally identical to real on-orbit ingest output.

**The 38 CSV columns in order:**

| # | Column name | P01 field | Source | Type |
|---|-------------|-----------|--------|------|
| 1 | `rows` | `rows` | constant 260 | int |
| 2 | `cols` | `cols` | constant 276 | int |
| 3 | `exp_time` | `exp_time` | §4.5 (centiseconds) | int |
| 4 | `exp_unit` | `exp_unit` | 38500 (fixed) | int |
| 5 | `ccd_temp1` | `ccd_temp1` | §4.3 N(−10, 1) °C | float |
| 6 | `lua_timestamp` | `lua_timestamp` | NB01 epoch, Unix ms | int |
| 7 | `adcs_timestamp` | `adcs_timestamp` | = lua_timestamp | int |
| 8 | `spacecraft_latitude` | `spacecraft_latitude` | NB01 lat_deg → rad | float |
| 9 | `spacecraft_longitude` | `spacecraft_longitude` | NB01 lon_deg → rad | float |
| 10 | `spacecraft_altitude` | `spacecraft_altitude` | NB01 alt_km → m | float |
| 11 | `att_q_x` | `attitude_quaternion[0]` | NB02a q[0] | float |
| 12 | `att_q_y` | `attitude_quaternion[1]` | NB02a q[1] | float |
| 13 | `att_q_z` | `attitude_quaternion[2]` | NB02a q[2] | float |
| 14 | `att_q_w` | `attitude_quaternion[3]` | NB02a q[3] | float |
| 15 | `pe_q_x` | `pointing_error[0]` | §4.1 qe[0] | float |
| 16 | `pe_q_y` | `pointing_error[1]` | §4.1 qe[1] | float |
| 17 | `pe_q_z` | `pointing_error[2]` | §4.1 qe[2] | float |
| 18 | `pe_q_w` | `pointing_error[3]` | §4.1 qe[3] | float |
| 19 | `pos_eci_x` | `pos_eci_hat[0]` | NB01 pos_eci_x, m | float |
| 20 | `pos_eci_y` | `pos_eci_hat[1]` | NB01 pos_eci_y, m | float |
| 21 | `pos_eci_z` | `pos_eci_hat[2]` | NB01 pos_eci_z, m | float |
| 22 | `vel_eci_x` | `vel_eci_hat[0]` | NB01 vel_eci_x, m/s | float |
| 23 | `vel_eci_y` | `vel_eci_hat[1]` | NB01 vel_eci_y, m/s | float |
| 24 | `vel_eci_z` | `vel_eci_hat[2]` | NB01 vel_eci_z, m/s | float |
| 25 | `etalon_t0` | `etalon_temps[0]` | §4.2 N(24, 0.1) °C | float |
| 26 | `etalon_t1` | `etalon_temps[1]` | §4.2 N(24, 0.1) °C | float |
| 27 | `etalon_t2` | `etalon_temps[2]` | §4.2 N(24, 0.1) °C | float |
| 28 | `etalon_t3` | `etalon_temps[3]` | §4.2 N(24, 0.1) °C | float |
| 29 | `gpio_0` | `gpio_pwr_on[0]` | frame-type dependent | int |
| 30 | `gpio_1` | `gpio_pwr_on[1]` | frame-type dependent | int |
| 31 | `gpio_2` | `gpio_pwr_on[2]` | frame-type dependent | int |
| 32 | `gpio_3` | `gpio_pwr_on[3]` | frame-type dependent | int |
| 33 | `lamp_0` | `lamp_ch_array[0]` | frame-type dependent | int |
| 34 | `lamp_1` | `lamp_ch_array[1]` | frame-type dependent | int |
| 35 | `lamp_2` | `lamp_ch_array[2]` | frame-type dependent | int |
| 36 | `lamp_3` | `lamp_ch_array[3]` | frame-type dependent | int |
| 37 | `lamp_4` | `lamp_ch_array[4]` | frame-type dependent | int |
| 38 | `lamp_5` | `lamp_ch_array[5]` | frame-type dependent | int |

**CSV construction:**

```python
rows_out = []
for m in metadata_list:
    d = dataclasses.asdict(m)
    rows_out.append({
        'rows':                 d['rows'],
        'cols':                 d['cols'],
        'exp_time':             d['exp_time'],
        'exp_unit':             d['exp_unit'],
        'ccd_temp1':            d['ccd_temp1'],
        'lua_timestamp':        d['lua_timestamp'],
        'adcs_timestamp':       d['adcs_timestamp'],
        'spacecraft_latitude':  d['spacecraft_latitude'],
        'spacecraft_longitude': d['spacecraft_longitude'],
        'spacecraft_altitude':  d['spacecraft_altitude'],
        'att_q_x':  d['attitude_quaternion'][0],
        'att_q_y':  d['attitude_quaternion'][1],
        'att_q_z':  d['attitude_quaternion'][2],
        'att_q_w':  d['attitude_quaternion'][3],
        'pe_q_x':   d['pointing_error'][0],
        'pe_q_y':   d['pointing_error'][1],
        'pe_q_z':   d['pointing_error'][2],
        'pe_q_w':   d['pointing_error'][3],
        'pos_eci_x': d['pos_eci_hat'][0],
        'pos_eci_y': d['pos_eci_hat'][1],
        'pos_eci_z': d['pos_eci_hat'][2],
        'vel_eci_x': d['vel_eci_hat'][0],
        'vel_eci_y': d['vel_eci_hat'][1],
        'vel_eci_z': d['vel_eci_hat'][2],
        'etalon_t0': d['etalon_temps'][0],
        'etalon_t1': d['etalon_temps'][1],
        'etalon_t2': d['etalon_temps'][2],
        'etalon_t3': d['etalon_temps'][3],
        'gpio_0': d['gpio_pwr_on'][0],
        'gpio_1': d['gpio_pwr_on'][1],
        'gpio_2': d['gpio_pwr_on'][2],
        'gpio_3': d['gpio_pwr_on'][3],
        'lamp_0': d['lamp_ch_array'][0],
        'lamp_1': d['lamp_ch_array'][1],
        'lamp_2': d['lamp_ch_array'][2],
        'lamp_3': d['lamp_ch_array'][3],
        'lamp_4': d['lamp_ch_array'][4],
        'lamp_5': d['lamp_ch_array'][5],
    })

pd.DataFrame(rows_out).to_csv(csv_path, index=False)
```

**What is deliberately absent from the CSV:**

| Omitted field | Reason |
|---------------|--------|
| `img_type`, `binning`, `shutter_status` | Derived from gpio/lamp via P01 §2.5 |
| `utc_timestamp` | Derived from `lua_timestamp` |
| `lamp1/2/3_status` | Derived from `lamp_ch_array` |
| `obs_mode`, `orbit_number`, `frame_sequence`, `orbit_parity` | Pipeline-added schedule fields |
| `adcs_quality_flag` | Computed from `pointing_error` via P01 |
| `is_synthetic`, `noise_seed` | Pipeline provenance fields |
| All truth fields (`truth_v_*`, `tangent_*`, `etalon_gap*`) | `None` in G01; belong in `.npy` |
| All dark provenance fields | `None`/default in G01; belong in `.npy` |
| `grafana_record_id` | `None` in G01 |

---

## 7. Progress reporting and verification

### 7.1 Console output

```
Parameters:
  ...
  Exp. time        : 8000 counts × 1.0 ms/count = 8.000 s  (exp_unit=38500)
  ...

Building NB01 orbit schedule ... done

Observation schedule:
  Science frames : 115860
  Cal frames     : 2275    (5 per orbit × ~455 complete orbits)
  Dark frames    : 2275    (5 per orbit × ~455 complete orbits)
  Total frames   : 120410

Building NB02a attitude quaternions + metadata ...
  [====================] 120410/120410

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
  validation/outputs/GEN01_20270101_030.0d_seed0042.npy  (52.8 MB — full ImageMetadata)
  validation/outputs/GEN01_20270101_030.0d_seed0042.csv  (73.9 MB — 38-column header export)

G01 complete.
```

### 7.2 Verification checks

Checks annotated with the file they operate on: `[.npy]` or `[.csv]`.
Non-blocking — print `PASS` or `FAIL — <reason>` for each.

| Check | Criterion | File |
|-------|-----------|------|
| C1 | NB01 sched rows ≈ `duration_s / SCHED_DT_S` ± 2 | — |
| C2 | All `orbit_parity` ∈ `{'along_track', 'cross_track'}` | `.npy` |
| C3 | No NaN in `att_q_x/y/z/w` columns | `.csv` |
| C4 | All attitude quaternions unit norm to 1e-6: `√(att_q_x²+att_q_y²+att_q_z²+att_q_w²) ≈ 1` | `.csv` |
| C5 | All pointing error quaternions unit norm to 1e-6: `√(pe_q_x²+…+pe_q_w²) ≈ 1` | `.csv` |
| C6 | Mean unsigned PE angle within 20% of `σ·√(2/π)` ≈ 3.99 arcsec | `.csv` pe_q_* columns |
| C7 | `mean(etalon_t0..t3)` within 0.05°C of 24.0°C | `.csv` |
| C8 | P01 `adcs_quality_flag == 0` for > 99.9% of frames | `.npy` |
| C9 | `img_type` ∈ `{'science', 'cal', 'dark'}` (derived from `.csv` gpio/lamp) | `.csv` → P01 |
| C10 | Cal count ≈ `n_caldark × n_complete_orbits` (±5%) | `.npy` |
| C11 | Dark count ≈ `n_caldark × n_complete_orbits` (±5%) | `.npy` |
| C12 | CSV has exactly 38 columns | `.csv` |
| C13 | `.npy` round-trip: `ImageMetadata(**np.load(path, allow_pickle=True)[0])` succeeds | `.npy` |

**C9 implementation note:** Derive `img_type` from the CSV's gpio/lamp columns
using `_classify_img_type()` and verify all values are in the valid set.
This also confirms that the gpio/lamp columns are self-consistent with P01
classification logic — any bug in the instrument state assignment will surface
here without needing to re-run the generation.

**C12 is the CSV structural guard.** If a future code change adds or removes a
column, C12 catches it immediately without needing to inspect column names.

---

## 8. File location in repository

```
soc_sewell/
├── validation/
│   ├── gen01_synthetic_metadata_generator_2026_04_16.py
│   └── outputs/
│       ├── GEN01_20270101_030.0d_seed0042.npy   (52.8 MB — full ImageMetadata)
│       └── GEN01_20270101_030.0d_seed0042.csv   (73.9 MB — 38-col header export)
└── docs/specs/
    └── G01_synthetic_metadata_generator_2026-04-16.md
```

---

## 9. Instructions for Claude Code

### Preamble
```bash
cat PIPELINE_STATUS.md
```

### Prerequisite reads
1. This spec in full.
2. NB01 spec — `propagate_orbit` signature; `_eci_to_geodetic_batch` is internal.
3. NB02a spec — `compute_los_eci(pos_eci, vel_eci, look_mode, h_target_km)`;
   always pass `h_target_km` explicitly.
4. P01 spec — `ImageMetadata` fields; `AdcsQualityFlags`;
   `compute_adcs_quality_flag()`.
5. `CLAUDE.md` at repo root.

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
ETALON_TEMP_MEAN_C    = 24.0
ETALON_TEMP_STD_C     =  0.1
CCD_TEMP_MEAN_C       = -10.0
CCD_TEMP_STD_C        =   1.0
TIMER_PERIOD_S        =  0.001
EXP_UNIT_REGISTER     = 38500
```

### CSV construction rule

Build each CSV row by **explicitly naming each of the 38 columns** from the
dict returned by `dataclasses.asdict(m)`. Do not use `dataclasses.asdict(m)`
directly as a CSV row, do not iterate over dict keys, and do not use
`df.drop()` to remove unwanted columns. The column list in §6.3 is the
authoritative definition — construct exactly those 38 columns and no others.

### Critical rules
- **`exp_unit` = `EXP_UNIT_REGISTER = 38500` always** — not derived from
  `exp_time_cts`.
- **`ccd_temp1`** = `rng.normal(CCD_TEMP_MEAN_C, CCD_TEMP_STD_C)` per frame.
- **RNG draw order**: θ (1) → axis (3) → etalon (4) → CCD (1). Never change.
- **C6**: expected mean = `SIGMA_POINTING_ARCSEC * np.sqrt(2 / np.pi)`.
- **C10/C11**: use `len(cal_trigger_indices)`, not `orbit_number.max()`.
- **C12**: `assert len(df_csv.columns) == 38`.
- **NB01 at `SCHED_DT_S = 10.0`** always.
- **No `build_synthetic_metadata()`** — construct `ImageMetadata` directly.
- **Timezone**: `t0 = pd.Timestamp(t_start, tz='UTC')`.

### Epilogue
```bash
git add PIPELINE_STATUS.md \
        validation/gen01_synthetic_metadata_generator_2026_04_16.py
git commit -m "feat(g01): CSV reduced to 38 binary-header columns, C1-C13 pass

CSV: exactly 38 columns = 17 P01 binary-header fields expanded to scalars.
Derived/calculated/None fields removed from CSV; retained in .npy only.
C12: assert len(df.columns) == 38. C9: img_type via _classify_img_type(gpio, lamp).
30-day run: 120410 frames, science=115860, cal=2275, dark=2275

Also updates PIPELINE_STATUS.md"
```
