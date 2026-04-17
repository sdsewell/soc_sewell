# G01 — Synthetic Metadata Generator Specification

**Spec ID:** G01
**Spec file:** `docs/specs/G01_synthetic_metadata_generator_2026-04-16.md`
**Project:** WindCube FPI Science Operations Center Pipeline
**Institution:** NCAR / High Altitude Observatory (HAO)
**Status:** Draft — awaiting implementation
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
    calibration cadence, dark frame strategy
    *(When this CONOPS document is updated to a new version, review
    §3.2–3.3 of this spec against the new observation schedule and
    update the version citation above and the CONOPS revision note below.)*
  - SI-UCAR-WC-RP-004 Issue 1.0 — AOCS Design Report (BRF/THRF/SIRF frames)
**Last updated:** 2026-04-16

> **CONOPS version note:** The observation schedule defined in §3 is derived
> from the WindCube Mission CONOPS Document cited above. If that document is
> superseded, this spec must be reviewed against the new CONOPS before any
> re-implementation. Specifically: science band boundaries (§3.2), calibration
> trigger latitude (§3.3), and the n cal + n dark sequence structure (§3.3)
> are all CONOPS-driven parameters. Changes to any of these require a dated
> revision of this spec file before Claude Code is re-run.

> **Revision history:**
> - v1 (2026-04-16): Initial spec — latitude-threshold instrument state, one
>   record per NB01 epoch.
> - v2 (2026-04-16): CONOPS model — science at user cadence within ±lat_band,
>   cal+dark triggered at 60°N ascending; added cadence, n_caldark, lat_band
>   prompts.
> - v3 (2026-04-16): Removed all S-number references (S06, S07, S19);
>   replaced with NB01, NB02a, P01 module names throughout. Added CONOPS
>   document citation with version placeholder.

---

## 1. Purpose

G01 is a standalone, interactive Python script that pre-computes and saves
the complete AOCS/instrument metadata array for a synthetic WindCube FPI
observation campaign. Every downstream image synthesis module (Z02, Z03, dark
frames) consumes these `ImageMetadata` records rather than re-running the
geometry pipeline independently.

**WindCube CONOPS as modelled by G01 (current — see CONOPS document citation
in header):**

```
One orbit (≈ 94.8 min, 510 km SSO):

  Ascending node (equator)
      │
      ▼ lat increases →
  ┌───────────────────────────────────────────────────────────────┐
  │  SCIENCE BAND  │   gap   │ CAL+DARK │ max lat │  gap  │ SCI  │
  │  0° → +band°   │band→60° │  trigger │  ~82°N  │ desc. │ etc. │
  │  obs_cadence_s │ (idle)  │ n+n frms │         │       │      │
  └───────────────────────────────────────────────────────────────┘
             ↑
   Science frames repeat on every science-band transit
   (2–4 transits per orbit depending on band width)
```

- **Science frames:** one frame every `obs_cadence_s` seconds while
  `|lat_deg| ≤ lat_band_deg`. `img_type = 'science'`, shutter open, no lamps.

- **Cal/dark sequence:** triggered once per orbit at the first epoch where
  the spacecraft geodetic latitude ascends through 60.0°N. A sequence of
  `n_caldark` calibration frames followed immediately by `n_caldark` dark
  frames, each spaced `obs_cadence_s` seconds apart.
  Total sequence duration = `2 × n_caldark × obs_cadence_s` seconds.

- **All other epochs:** no observation, no `ImageMetadata` record generated.

**What G01 does not do:**
- It does not compute tangent points, `v_rel`, or wind observations (NB02b,
  NB02c). Those are added by Z02/Z03.
- It does not generate pixel data.
- It does not call `build_synthetic_metadata()` from P01 — G01 constructs
  `ImageMetadata` objects directly because NB02c wind outputs are unavailable.

---

## 2. User interface — interactive prompts

All prompts appear before any computation. Invalid entries are rejected and
re-prompted. Blank input accepts the default.

```
=== G01 — WindCube Synthetic Metadata Generator ===

Start epoch          [2027-01-01T00:00:00 UTC]  : _
Duration             [days,  default  30       ] : _
Science band         [deg,   default  40       ] : _   ← |lat| ≤ this value
Obs. cadence         [sec,   default  10       ] : _   ← science, cal, dark spacing
Cal/dark frames (n)  [int,   default   5       ] : _   ← n cal + n dark per orbit
S/C altitude         [km,    default 510       ] : _
Tangent height       [km,    default 250       ] : _
Output directory     [       default outputs/  ] : _
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
| `altitude_km` | S/C altitude | 510.0 | float | 400.0 – 700.0 |
| `h_target_km` | Tangent height | 250.0 | float | 100.0 – 400.0 |
| `output_dir` | Output directory | `"outputs/"` | str | Any writable path |
| `rng_seed` | NumPy RNG seed | 42 | int | ≥ 0 |

**Constraint warnings** (printed to console, not errors):

- If `lat_band_deg ≥ 60.0`: warn that the science band overlaps the 60°N
  cal/dark trigger latitude. Cal/dark type takes precedence at trigger epochs.
- If `2 × n_caldark × obs_cadence_s > 1200`: warn that the cal/dark sequence
  duration (> 20 min) may extend into the orbit's descending high-latitude arc.

---

## 3. Orbit propagation and CONOPS scheduling

All observation schedule logic in this section is governed by the WindCube
Mission CONOPS Document cited in the header. Section and parameter references
below map to that document. When the CONOPS is revised, update those
references here before re-implementing.

### 3.1 Two-tier propagation

G01 runs NB01 at a fixed **schedule cadence** of `SCHED_DT_S = 10.0` s
regardless of `obs_cadence_s`. This ensures the 60°N trigger can be detected
to within 10 s (≈ 76 km along-track) regardless of the observation cadence.

The user's `obs_cadence_s` determines the **step size** in grid rows:

```python
SCHED_DT_S = 10.0                                 # always fixed
step = max(1, round(obs_cadence_s / SCHED_DT_S))  # grid rows per obs
actual_cadence_s = step * SCHED_DT_S              # reported to user
```

NB01 propagation call:

```python
from src.geometry.nb01_orbit_propagator_2026_04_16 import propagate_orbit

df_sched = propagate_orbit(
    t_start    = t_start,
    duration_s = duration_days * 86400.0,
    dt_s       = SCHED_DT_S,
)
```

After propagation, assign orbit number and look mode (identical to INT01 §1):

```python
a_m = WGS84_A_M + altitude_km * 1e3
T_ORBIT_S = 2 * np.pi * np.sqrt(a_m**3 / EARTH_GRAV_PARAM_M3_S2)

t0 = pd.Timestamp(t_start, tz='UTC')
df_sched['elapsed_s']    = (df_sched['epoch'] - t0).dt.total_seconds()
df_sched['orbit_number'] = (df_sched['elapsed_s'] // T_ORBIT_S).astype(int) + 1
df_sched['look_mode']    = df_sched['orbit_number'].apply(
    lambda n: 'along_track' if n % 2 == 1 else 'cross_track'
)
```

Odd orbit numbers → `along_track`; even → `cross_track`.

### 3.2 Science frame selection

*(CONOPS reference: [TBD — insert document section for science band
definition and observation cadence, e.g. §4.2])*

Science frames are generated from 10 s grid rows where `|lat_deg| ≤ lat_band_deg`,
at every `step`-th row from each science-band entry point:

```python
science_indices = []
in_band         = False
band_entry_i    = None

for i, row in df_sched.iterrows():
    if abs(row.lat_deg) <= lat_band_deg:
        if not in_band:
            in_band      = True
            band_entry_i = i
        if (i - band_entry_i) % step == 0:
            science_indices.append(i)
    else:
        in_band      = False
        band_entry_i = None
```

A single orbit has multiple science-band transits (ascending and descending
through the band on both hemispheres). Each transit resets `band_entry_i`
so spacing is consistent within each transit.

**Science frame instrument state:**

| Field | Value | Notes |
|-------|-------|-------|
| `gpio_pwr_on` | `[0, 0, 0, 0]` | Shutter open |
| `lamp_ch_array` | `[0, 0, 0, 0, 0, 0]` | No lamps |
| `shutter_status` | `'open'` | Derived |
| `img_type` | `'science'` | Derived via `_classify_img_type()` |
| `exp_time` | `500` cs (5 s) | Stub — to be confirmed against CONOPS |

### 3.3 Calibration and dark trigger and sequence

*(CONOPS reference: [TBD — insert document section for calibration schedule,
e.g. §4.3 and §4.4])*

The **cal/dark trigger** is defined in the CONOPS as the first epoch per orbit
where the spacecraft geodetic latitude ascends through `CAL_TRIGGER_LAT_DEG =
60.0°N`. This is a named constant in the implementation; if the CONOPS is
updated to use a different trigger latitude, update `CAL_TRIGGER_LAT_DEG`
and the version citation in the header.

**Trigger detection** (run once over `df_sched` after orbit assignment):

```python
CAL_TRIGGER_LAT_DEG = 60.0   # CONOPS fixed parameter — see header citation

cal_trigger_indices = []
lat = df_sched['lat_deg'].values

for i in range(1, len(lat)):
    if (lat[i]     >  CAL_TRIGGER_LAT_DEG
            and lat[i-1] <= CAL_TRIGGER_LAT_DEG
            and lat[i]   >  lat[i-1]):      # ascending: lat increasing
        cal_trigger_indices.append(i)
```

At most one trigger per orbit; the 97.44° SSO always crosses 60°N exactly
once ascending per orbit.

**Sequence from each trigger index `t₀`:**

```
Cal frames:   t₀,  t₀+step,  t₀+2·step, …,  t₀+(n-1)·step     (n_caldark frames)
Dark frames:  t₀+n·step,  …,  t₀+(2n-1)·step                   (n_caldark frames)
```

Skip any candidate index ≥ `len(df_sched)` (campaign end boundary).

**Calibration instrument state:**

| Field | Value | Notes |
|-------|-------|-------|
| `gpio_pwr_on` | `[0, 1, 1, 0]` | Shutter open |
| `lamp_ch_array` | `[1, 1, 1, 1, 1, 1]` | All lamps on (neon) |
| `shutter_status` | `'open'` | Derived |
| `img_type` | `'cal'` | Derived via `_classify_img_type()` |
| `exp_time` | `120` cs (1.2 s) | Stub — to be confirmed against CONOPS |

**Dark instrument state:**

| Field | Value | Notes |
|-------|-------|-------|
| `gpio_pwr_on` | `[1, 0, 0, 1]` | Shutter closed |
| `lamp_ch_array` | `[0, 0, 0, 0, 0, 0]` | No lamps |
| `shutter_status` | `'closed'` | Derived |
| `img_type` | `'dark'` | Derived via `_classify_img_type()` |
| `exp_time` | `500` cs (5 s) | Stub — to be confirmed against CONOPS |

### 3.4 `img_type` derivation — always use P01 logic

Regardless of what G01 intends for each frame, derive `img_type` from the
P01 classification function. Do not hardcode image type strings directly:

```python
def _classify_img_type(lamp_ch_array: list, gpio_pwr_on: list) -> str:
    """P01 classification logic — keep in sync with p01_image_metadata_2026_04_06.py."""
    if any(lamp_ch_array):
        return "cal"
    elif gpio_pwr_on[0] == 1 and gpio_pwr_on[3] == 1:
        return "dark"
    return "science"
```

The three instrument states in §3.2–3.3 are designed to produce exactly
`'science'`, `'cal'`, and `'dark'` through this function. If they do not,
there is a bug in the gpio/lamp values.

### 3.5 Schedule assembly and frame sequencing

Collect all scheduled observation indices into one ordered list, with cal/dark
taking precedence over science at any overlap (possible when `lat_band_deg ≥ 60°`):

```python
cal_dark_set  = set(cal_indices) | set(dark_indices)
science_final = [i for i in science_indices if i not in cal_dark_set]
obs_indices   = sorted(science_final + cal_indices + dark_indices)
```

Within each orbit, `frame_sequence` is the 0-based index among all observation
frames for that orbit (science + cal + dark together, ordered by epoch):

```python
orbit_counter = {}
frame_sequences = []
for idx in obs_indices:
    orb = df_sched.loc[idx, 'orbit_number']
    n   = orbit_counter.get(orb, 0)
    frame_sequences.append(n)
    orbit_counter[orb] = n + 1
```

---

## 4. Noise model

### 4.1 Pointing error quaternion

The AOCS pointing error is modelled as a small random rotation with σ = 5
arcsec (1σ):

```
σ_θ = 5 arcsec × (π / 648000) ≈ 2.4241 × 10⁻⁵ rad
```

Per observation frame:

1. `θ = rng.normal(0.0, σ_θ)` — rotation magnitude (signed)
2. `raw = rng.standard_normal(3);  n = raw / ‖raw‖` — random unit axis
3. `qe = [n[0]·sin(θ/2), n[1]·sin(θ/2), n[2]·sin(θ/2), cos(θ/2)]` — scalar-last
4. `qe = qe / ‖qe‖` — normalise

The SLEW_IN_PROGRESS threshold (30 arcsec, from P01 `AdcsQualityFlags`) is
6σ. Essentially no frames will be flagged over a 30-day campaign.

**RNG draw order per observation frame** (strictly this order — do not change):
1. `θ`: 1 draw via `rng.normal`
2. Axis components: 3 draws via `rng.standard_normal(3)`
3. Etalon temperatures: 4 draws via `rng.normal(24.0, 0.1, 4)`

### 4.2 Etalon temperatures

```python
etalon_temps = rng.normal(24.0, 0.1, size=4).tolist()   # °C
```

Mean 24.0 °C (ambient lab temperature); σ = 0.1 °C (thermistor noise).

### 4.3 CCD temperature

Constant: `ccd_temp1 = -18.0` °C (nominal on-orbit set point).

---

## 5. `ImageMetadata` field assignment

Fields sourced from NB01 use the `propagate_orbit()` output column names.
Fields sourced from NB02a use the `compute_los_eci()` return values.
Fields sourced from P01 use the `ImageMetadata` dataclass field names.

| Field | Source module | Value / formula |
|-------|--------------|-----------------|
| `rows` | constant | `260` |
| `cols` | constant | `276` |
| `exp_time` | §3.2–3.3 | science/dark: `500` cs; cal: `120` cs |
| `exp_unit` | constant | `1` |
| `binning` | constant | `2` |
| `img_type` | P01 `_classify_img_type()` | derived from gpio/lamp |
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
| `obs_mode` | NB01 orbit parity | `'along_track'` (odd) / `'cross_track'` (even) |
| `ccd_temp1` | §4.3 | `-18.0` °C |
| `etalon_temps` | §4.2 | `rng.normal(24.0, 0.1, 4).tolist()` |
| `shutter_status` | §3.4 | derived from `gpio_pwr_on` |
| `gpio_pwr_on` | §3.2–3.3 | frame-type dependent |
| `lamp_ch_array` | §3.2–3.3 | frame-type dependent |
| `lamp1_status` | P01 rule | `"on" if lamp_ch[0] or lamp_ch[1] else "off"` |
| `lamp2_status` | P01 rule | `"on" if lamp_ch[2] or lamp_ch[3] else "off"` |
| `lamp3_status` | P01 rule | `"on" if lamp_ch[4] or lamp_ch[5] else "off"` |
| `orbit_number` | NB01 §3.1 | 1-based, from elapsed time |
| `frame_sequence` | §3.5 | 0-based within orbit observation list |
| `orbit_parity` | NB01 §3.1 | `'along_track'` / `'cross_track'` |
| `adcs_quality_flag` | P01 `compute_adcs_quality_flag()` | computed from `pointing_error` |
| `is_synthetic` | constant | `True` |
| `noise_seed` | frame index | 0-based position in `obs_indices` |
| `truth_v_los` | — | `None` (NB02c not run in G01) |
| `truth_v_zonal` | — | `None` |
| `truth_v_meridional` | — | `None` |
| `tangent_lat` | — | `None` (NB02b not run in G01) |
| `tangent_lon` | — | `None` |
| `tangent_alt_km` | — | `None` |
| `etalon_gap_mm` | — | `None` |
| `etalon_gap_corrected_mm` | — | `None` |
| `grafana_record_id` | — | `None` |
| `dark_subtracted` | default | `False` |
| `dark_n_frames` | default | `0` |
| `dark_lua_timestamp` | — | `None` |
| `dark_etalon_temp_mean` | — | `None` |

**`attitude_quaternion` vs `pointing_error`:** `attitude_quaternion` is the
**ideal commanded** boresight quaternion from NB02a `compute_los_eci()`. 
`pointing_error` is the **residual error** quaternion from §4.1. This mirrors
the P01 binary header layout (words 28–43 and 44–59 respectively).

---

## 6. Output files

### 6.1 Naming convention

```
{output_dir}/GEN01_{t_start_compact}_{duration_days:05.1f}d_seed{rng_seed:04d}
```

where `t_start_compact = t_start[:10].replace('-', '')`. Example (defaults):

```
outputs/GEN01_20270101_030.0d_seed0042.npy
outputs/GEN01_20270101_030.0d_seed0042.csv
```

### 6.2 `.npy` — object array of dicts

```python
import dataclasses, numpy as np
records = [dataclasses.asdict(m) for m in metadata_list]
np.save(npy_path, np.array(records, dtype=object), allow_pickle=True)
```

Loading in downstream modules:
```python
records   = np.load(npy_path, allow_pickle=True)
meta_list = [ImageMetadata(**r) for r in records]
```

### 6.3 `.csv` — flat expanded columns

| P01 list field | CSV column names |
|----------------|-----------------|
| `pos_eci_hat` | `pos_eci_x`, `pos_eci_y`, `pos_eci_z` |
| `vel_eci_hat` | `vel_eci_x`, `vel_eci_y`, `vel_eci_z` |
| `attitude_quaternion` | `att_q_x`, `att_q_y`, `att_q_z`, `att_q_w` |
| `pointing_error` | `pe_q_x`, `pe_q_y`, `pe_q_z`, `pe_q_w` |
| `etalon_temps` | `etalon_t0`, `etalon_t1`, `etalon_t2`, `etalon_t3` |
| `gpio_pwr_on` | `gpio_0`, `gpio_1`, `gpio_2`, `gpio_3` |
| `lamp_ch_array` | `lamp_0`, `lamp_1`, `lamp_2`, `lamp_3`, `lamp_4`, `lamp_5` |

All other fields keep their P01 field names as column headers. Original
list-valued dict keys are dropped and replaced by the expanded columns.

---

## 7. Progress reporting and verification

### 7.1 Console output

```
=== G01 — WindCube Synthetic Metadata Generator ===

Parameters:
  Start epoch      : 2027-01-01T00:00:00 UTC
  Duration         : 30.0 days
  Science band     : ±40.0°
  Obs. cadence     : 10.0 s requested → 10.0 s actual (step=1)
  Cal/dark per orb : 5 + 5 = 10 frames  (100.0 s sequence)
  Cal trigger lat  : 60.0°N ascending  [CONOPS TBD document, §TBD]
  S/C altitude     : 510.0 km
  Tangent ht       : 250.0 km
  T_orbit          : 5689.4 s (94.82 min)
  NB01 sched rows  : 259201  (10 s grid, 30 days)
  Total orbits     : ~456
  RNG seed         : 42
  Output dir       : outputs/

Building NB01 orbit schedule ... done

Observation schedule:
  Science frames : NNNNNN
  Cal frames     : NNNN    (5 per orbit × ~456 orbits)
  Dark frames    : NNNN    (5 per orbit × ~456 orbits)
  Total frames   : NNNNNN

Building NB02a attitude quaternions + metadata ...
  [====================] NNNNNN/NNNNNN ... done

Image type verification:
  science : NNNNNN
  cal     : NNNN
  dark    : NNNN

ADCS quality flags (P01):
  GOOD             : NNNNNN
  SLEW_IN_PROGRESS : 0   (expected ~0)

Pointing error stats (arcsec):
  Mean  :  0.00   (expected ~0.00)
  Std   :  5.00   (expected ~5.00)
  Max   : XX.X

Etalon temperature stats (°C):
  Mean  : 24.00   (expected ~24.00)
  Std   :  0.10   (expected ~ 0.10)

Output files:
  outputs/GEN01_20270101_030.0d_seed0042.npy  (XX.X MB)
  outputs/GEN01_20270101_030.0d_seed0042.csv  (XX.X MB)

G01 complete.
```

### 7.2 Verification checks

Non-blocking — print `PASS` or `FAIL — <reason>` for each:

| Check | Criterion |
|-------|-----------|
| C1 | NB01 sched rows ≈ `duration_s / SCHED_DT_S` ± 2 |
| C2 | All `orbit_parity` values are `'along_track'` or `'cross_track'` |
| C3 | No NaN in any `att_q_*` CSV column (NB02a output) |
| C4 | All attitude quaternions unit norm to 1e-6 (NB02a) |
| C5 | All pointing error quaternions unit norm to 1e-6 |
| C6 | Pointing error angle std within 20% of 5 arcsec |
| C7 | Etalon temp mean within 0.05°C of 24.0°C |
| C8 | P01 `adcs_quality_flag == 0` for > 99.9% of frames |
| C9 | `img_type` values are only `'science'`, `'cal'`, `'dark'` |
| C10 | Cal frame count ≈ `n_caldark × n_orbits` (±5%) |
| C11 | Dark frame count ≈ `n_caldark × n_orbits` (±5%) |
| C12 | No science frame has `lat_deg > lat_band_deg + 1°` |
| C13 | P01 round-trip: `ImageMetadata(**np.load(npy_path, allow_pickle=True)[0])` succeeds |

---

## 8. File location in repository

```
soc_sewell/
├── scripts/
│   └── gen01_synthetic_metadata_generator_2026_04_16.py
├── outputs/
│   ├── GEN01_20270101_030.0d_seed0042.npy
│   └── GEN01_20270101_030.0d_seed0042.csv
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

1. This entire spec: `docs/specs/G01_synthetic_metadata_generator_2026-04-16.md`
2. NB01 spec and implementation — propagation signature and output column
   names (`epoch`, `lat_deg`, `lon_deg`, `alt_km`, `pos_eci_*`, `vel_eci_*`).
3. NB02a spec and implementation — `compute_los_eci(pos_eci, vel_eci,
   look_mode, h_target_km)` returns `(los_eci, q)`; iterative depression
   angle. Pass `h_target_km` from user input, never rely on the default.
4. P01 spec and implementation — `ImageMetadata` dataclass field definitions
   and types; `AdcsQualityFlags`; `compute_adcs_quality_flag()`.
5. `CLAUDE.md` at repo root.

### Prerequisite tests

```bash
pytest tests/test_nb01_orbit_propagator.py -v    # 8/8 must pass
pytest tests/test_nb02_geometry_2026_04_16.py -v # 10/10 must pass
pytest tests/test_s19_p01_metadata.py -v         # 8/8 must pass
```

### Implementation steps

1. Create `scripts/gen01_synthetic_metadata_generator_2026_04_16.py`:
   ```python
   """
   G01 — Synthetic Metadata Generator.

   Spec:      docs/specs/G01_synthetic_metadata_generator_2026-04-16.md
   Spec date: 2026-04-16
   Generated: YYYY-MM-DD
   Tool:      Claude Code
   CONOPS:    [TBD — insert document ID and version when known]
   Usage:     python scripts/gen01_synthetic_metadata_generator_2026_04_16.py
   """
   ```
   Copy the `_project_root` sys.path pattern from NB01 or NB02a verbatim.

2. **Imports** (minimum):
   ```python
   import sys, pathlib, dataclasses
   from datetime import timezone
   import numpy as np
   import pandas as pd

   from src.geometry.nb01_orbit_propagator_2026_04_16 import propagate_orbit
   from src.geometry.nb02a_boresight_2026_04_16 import compute_los_eci
   from src.metadata.p01_image_metadata_2026_04_06 import (
       ImageMetadata, AdcsQualityFlags, compute_adcs_quality_flag,
   )
   from src.constants import WGS84_A_M, EARTH_GRAV_PARAM_M3_S2
   ```

3. **Constants block** (right after imports):
   ```python
   SCHED_DT_S            = 10.0   # NB01 propagation cadence — always fixed
   CAL_TRIGGER_LAT_DEG   = 60.0   # CONOPS ascending trigger — see spec header
   SIGMA_POINTING_ARCSEC =  5.0
   ETALON_TEMP_MEAN_C    = 24.0
   ETALON_TEMP_STD_C     =  0.1
   CCD_TEMP_C            = -18.0
   ```

4. **`_prompt(msg, default, cast, lo, hi)` helper** — re-prompts on ValueError
   and out-of-range. Blank input → default. Used for all nine prompts.

5. **`_pointing_error_quat(rng, sigma_arcsec)` helper** — §4.1 exactly.
   Returns `[x, y, z, w]` normalised. Draw order: `rng.normal` (θ), then
   `rng.standard_normal(3)` (axis). Do not swap these two draws.

6. **`_classify_img_type(lamp_ch_array, gpio_pwr_on)` helper** — 3-line P01
   logic from §3.4. Self-contained; do not import from P01.

7. **`_instrument_state(frame_type)` helper** — returns
   `(gpio_pwr_on, lamp_ch_array, exp_time_cs)` for `'science'`, `'cal'`,
   `'dark'` per §3.2–3.3. Raises `ValueError` for unknown type.

8. **`_build_schedule(df_sched, lat_band_deg, n_caldark, step)` function** —
   returns `(obs_indices, frame_types)` sorted by index:
   - Science: §3.2 transit-counter loop.
   - Cal/dark: §3.3 ascending-crossing loop; for each trigger `t0`, append
     `t0 + k*step` (cal, k=0..n-1) then `t0 + (n+k)*step` (dark, k=0..n-1);
     skip any index ≥ `len(df_sched)`.
   - Remove science indices that are also cal/dark indices.
   - Sort and return.

9. **Main metadata loop** — iterate `zip(obs_indices, frame_types)`:
   - Extract NB01 row: `row = df_sched.loc[idx]`
   - Build `pos`, `vel` arrays from NB01 columns.
   - Call `compute_los_eci(pos, vel, look_mode, h_target_km=h_target_km)` →
     `(los_eci, q)`. Ignore `los_eci`.
   - Draw pointing error (§4.1, 4 draws total from `rng`).
   - Draw etalon temps (§4.2, 4 draws from `rng`).
   - Call `_instrument_state(frame_type)` → `(gpio, lamp, exp_time)`.
   - Compute `adcs_quality_flag` from P01 `compute_adcs_quality_flag()`.
   - Construct `ImageMetadata(...)` directly — **not** via P01
     `build_synthetic_metadata()`.

10. **Progress bar** (no external libraries):
    ```python
    if i % max(1, n_obs // 100) == 0 or i == n_obs - 1:
        pct = 100 * (i+1) / n_obs
        bar = '='*int(pct//5) + ' '*(20-int(pct//5))
        print(f'\r  [{bar}] {i+1}/{n_obs}', end='', flush=True)
    print()
    ```

11. **CSV construction** — §6.3 column expansion; write with
    `pd.DataFrame(rows).to_csv(csv_path, index=False)`.

12. **Verification checks C1–C13** — after saving; print `PASS`/`FAIL`,
    never raise.

13. **`if __name__ == '__main__': main()`** pattern.

### Critical rules

- **NB01 at `SCHED_DT_S = 10.0` always.** Never pass `obs_cadence_s` to
  `propagate_orbit`.
- **NB02a `compute_los_eci` needs explicit `h_target_km`** — the iterative
  depression angle requires it. Do not rely on the 250.0 default.
- **RNG draw order is fixed**: pointing error draws (4) before etalon
  draws (4), strictly per observation frame, every frame.
- **Do not call P01 `build_synthetic_metadata()`** — it requires NB02c
  wind outputs not computed in G01.
- **Timezone**: `t0 = pd.Timestamp(t_start, tz='UTC')`.
  Use `row.epoch.timestamp()` for Unix seconds.

### Stop condition

Stop and report to Claude.ai if any prerequisite test fails, or if
`_build_schedule` produces zero science frames or zero cal triggers for a
1-day default-parameters test run.

### Report-back format

```
G01 IMPLEMENTATION REPORT
==========================
Spec:       G01_synthetic_metadata_generator_2026-04-16.md
Test run:   1 day, lat_band=40°, cadence=10s, n=5, alt=510 km, h=250 km
NB01 rows:  N
Obs frames: N  (science=N, cal=N, dark=N)
Orbits:     N  (along_track=N, cross_track=N)
Cadence:    X.X s actual (step=N)
Checks:     C1–C13  [list any FAIL]
PE std (arcsec): X.XX
Etalon mean (°C): XX.XX
Files:
  outputs/GEN01_20270101_001.0d_seed0042.npy
  outputs/GEN01_20270101_001.0d_seed0042.csv
Deviations from spec: [none | list any]
```

### Epilogue

```bash
git add PIPELINE_STATUS.md \
        scripts/gen01_synthetic_metadata_generator_2026_04_16.py
git commit -m "feat(g01): synthetic metadata generator, C1-C13 pass

Implements: G01_synthetic_metadata_generator_2026-04-16.md
CONOPS: science at obs_cadence_s in ±lat_band, cal+dark at 60N trigger
NB01 + NB02a + P01; no S-number references
1-day test: N obs frames (science=N, cal=N, dark=N)

Also updates PIPELINE_STATUS.md"
```
