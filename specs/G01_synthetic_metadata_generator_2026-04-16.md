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
> - v4 (2026-04-16): Half-normal PE distribution documented; C6/C10/C11
>   corrected; script location changed to `validation/`.
> - v5 (2026-04-16): Implementation corrections from 30-day G01 report:
>   (a) New `exp_time_cts` prompt (default 8000 counts); `TIMER_PERIOD_S =
>   0.001` s/count; `exp_unit` fixed to hardware register value `38500`.
>   (b) `ccd_temp1` now drawn per-frame from N(−10, 1°C) rather than a
>   constant. RNG draw order updated: PE (4) → etalon (4) → CCD (1).
>   (c) Console output updated with CCD stats block and exp_time line.

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
  once per orbit when the spacecraft latitude first ascends through 60.0°N,
  frames spaced `obs_cadence_s` s apart.
- **All other epochs:** no observation, no `ImageMetadata` generated.

**What G01 does not do:** tangent points, v_rel, wind (NB02b/c); pixel data;
`build_synthetic_metadata()` from P01 (requires NB02c wind outputs).

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
The user cadence determines a step size in grid rows:

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
| `shutter_status` | `'open'` |
| `img_type` | `'science'` |

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

Sequence from each trigger index `t₀`: cal frames at `t₀, t₀+step, …, t₀+(n−1)·step`;
dark frames at `t₀+n·step, …, t₀+(2n−1)·step`. Skip indices ≥ `len(df_sched)`.

**Calibration instrument state:**

| Field | Value |
|-------|-------|
| `gpio_pwr_on` | `[0, 1, 1, 0]` |
| `lamp_ch_array` | `[1, 1, 1, 1, 1, 1]` |
| `shutter_status` | `'open'` |
| `img_type` | `'cal'` |

**Dark instrument state:**

| Field | Value |
|-------|-------|
| `gpio_pwr_on` | `[1, 0, 0, 1]` |
| `lamp_ch_array` | `[0, 0, 0, 0, 0, 0]` |
| `shutter_status` | `'closed'` |
| `img_type` | `'dark'` |

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

`frame_sequence`: 0-based index among all observation frames for each orbit,
ordered by epoch. `n_complete_orbits = len(cal_trigger_indices)` (used in
C10/C11 — does not include the terminal partial orbit).

---

## 4. Noise model and instrument constants

### 4.1 Pointing error quaternion

The signed rotation angle θ is drawn from N(0, σ_θ) with:

```
σ_θ = SIGMA_POINTING_ARCSEC × (π / 648000) ≈ 2.4241 × 10⁻⁵ rad
SIGMA_POINTING_ARCSEC = 5.0
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

**Observed statistics** — the physical rotation magnitude |θ| follows a
half-normal distribution:

```
E[|θ|]   = σ_θ · √(2/π) ≈ 3.99 arcsec
Std[|θ|] = σ_θ · √(1−2/π) ≈ 3.01 arcsec
```

P01 `compute_adcs_quality_flag()` sets `SLEW_IN_PROGRESS` at 30 arcsec (6σ).
Essentially no frames flagged over a 30-day campaign.

### 4.2 Etalon temperatures

```python
etalon_temps = rng.normal(24.0, 0.1, size=4).tolist()   # °C
```

### 4.3 CCD temperature

CCD temperature is drawn **per frame** from:

```python
ccd_temp1 = float(rng.normal(-10.0, 1.0))   # °C
```

Mean −10.0 °C represents the nominal on-orbit CCD set point.
σ = 1.0 °C represents thermal control noise and calibration uncertainty.

**Why stochastic rather than constant:** Real CCD temperatures vary with
orbital thermal cycling and heater duty cycle. Using a per-frame draw
provides realistic temperature diversity in the synthetic dataset, which
is important for thermal correction testing in downstream modules.

### 4.4 RNG draw order per observation frame

The following draw order is **fixed and must not be changed** — altering it
would invalidate the seed-reproducibility guarantee:

```
1. theta       : 1 draw  via rng.normal(0, sigma_theta_rad)
2. axis raw    : 3 draws via rng.standard_normal(3)
3. etalon temps: 4 draws via rng.normal(24.0, 0.1, 4)
4. ccd_temp1   : 1 draw  via rng.normal(-10.0, 1.0)
```

Total: 9 draws per observation frame.

### 4.5 Exposure time and hardware register

The user specifies exposure time as an integer count in units of timer ticks.
The hardware timer period is:

```python
TIMER_PERIOD_S = 0.001   # 1.0 ms per count (hardware constant)
exp_unit       = 38500   # fixed hardware register value (not a count)
```

The exposure duration in seconds is `exp_time_cts × TIMER_PERIOD_S`. This
value is stored in the `exp_time` field of `ImageMetadata` (in centiseconds
as the P01 binary format requires: `exp_time_cs = round(exp_time_cts * TIMER_PERIOD_S * 100)`).

The `exp_unit` field stores the hardware timing register value `38500`.
This is a **fixed hardware constant**, not derived from `exp_time_cts`.
It must always be `38500` regardless of the exposure duration.

The same `exp_time_cts` is applied uniformly to all frame types (science,
cal, dark). The CONOPS specifies separate exposure times per frame type;
a future revision will add per-type overrides when those values are confirmed
against the CONOPS document. For now a single user-specified value is used
for all types.

Console display:
```
  Exp. time : 8000 counts × 1.0 ms/count = 8.000 s  (exp_unit=38500)
```

---

## 5. `ImageMetadata` field assignment

| Field | Source | Value / formula |
|-------|--------|-----------------|
| `rows` | constant | `260` |
| `cols` | constant | `276` |
| `exp_time` | §4.5 | `round(exp_time_cts * TIMER_PERIOD_S * 100)` centiseconds |
| `exp_unit` | §4.5 | `38500` (hardware register, fixed) |
| `binning` | constant | `2` |
| `img_type` | P01 §3.4 | `_classify_img_type(lamp_ch_array, gpio_pwr_on)` |
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
| `orbit_number` | NB01 §3.1 | 1-based from elapsed time |
| `frame_sequence` | §3.5 | 0-based within orbit observation list |
| `orbit_parity` | NB01 §3.1 | `'along_track'` / `'cross_track'` |
| `adcs_quality_flag` | P01 `compute_adcs_quality_flag()` | from pointing_error |
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

Example (defaults): `validation/outputs/GEN01_20270101_030.0d_seed0042.npy`

### 6.2 `.npy` — object array of dicts

```python
records = [dataclasses.asdict(m) for m in metadata_list]
np.save(npy_path, np.array(records, dtype=object), allow_pickle=True)
```

Load: `meta_list = [ImageMetadata(**r) for r in np.load(path, allow_pickle=True)]`

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

---

## 7. Progress reporting and verification

### 7.1 Console output

```
Parameters:
  ...
  Exp. time        : 8000 counts × 1.0 ms/count = 8.000 s  (exp_unit=38500)
  ...

...

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
```

### 7.2 Verification checks

| Check | Criterion | Notes |
|-------|-----------|-------|
| C1 | NB01 sched rows ≈ `duration_s / SCHED_DT_S` ± 2 | |
| C2 | All `orbit_parity` ∈ `{'along_track', 'cross_track'}` | |
| C3 | No NaN in `att_q_*` (NB02a) | |
| C4 | All attitude quaternions unit norm to 1e-6 | |
| C5 | All pointing error quaternions unit norm to 1e-6 | |
| C6 | Mean unsigned PE angle within 20% of `σ·√(2/π)` ≈ 3.99 arcsec | Half-normal; not 5 arcsec |
| C7 | Etalon temp mean within 0.05°C of 24.0°C | |
| C8 | P01 `adcs_quality_flag == 0` for > 99.9% of frames | |
| C9 | `img_type` ∈ `{'science', 'cal', 'dark'}` | |
| C10 | Cal count ≈ `n_caldark × n_complete_orbits` (±5%) | `n_complete_orbits = len(cal_trigger_indices)` |
| C11 | Dark count ≈ `n_caldark × n_complete_orbits` (±5%) | Same — terminal partial orbit excluded |
| C12 | No science frame `lat_deg > lat_band_deg + 1°` | |
| C13 | P01 round-trip: `ImageMetadata(**np.load(npy_path, allow_pickle=True)[0])` succeeds | |

---

## 8. File location in repository

```
soc_sewell/
├── validation/
│   ├── gen01_synthetic_metadata_generator_2026_04_16.py
│   └── outputs/
│       ├── GEN01_20270101_030.0d_seed0042.npy   (52.8 MB, 30-day run)
│       └── GEN01_20270101_030.0d_seed0042.csv   (73.9 MB, 30-day run)
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
2. NB01 spec — `propagate_orbit` signature; note `_eci_to_geodetic_batch`
   is internal; public interface unchanged.
3. NB02a spec — `compute_los_eci(pos_eci, vel_eci, look_mode, h_target_km)`;
   iterative depression angle; pass `h_target_km` explicitly.
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
SCHED_DT_S            = 10.0    # NB01 cadence, always fixed
CAL_TRIGGER_LAT_DEG   = 60.0    # CONOPS ascending trigger
SIGMA_POINTING_ARCSEC =  5.0    # Gaussian σ for θ draw
ETALON_TEMP_MEAN_C    = 24.0
ETALON_TEMP_STD_C     =  0.1
CCD_TEMP_MEAN_C       = -10.0
CCD_TEMP_STD_C        =   1.0
TIMER_PERIOD_S        =  0.001  # 1 ms per count — hardware constant
EXP_UNIT_REGISTER     = 38500   # hardware timing register — always fixed
```

### Key implementation rules

- **`exp_time` field** = `round(exp_time_cts * TIMER_PERIOD_S * 100)` centiseconds.
  `exp_unit` field = `EXP_UNIT_REGISTER = 38500` always — not derived from
  `exp_time_cts`.
- **`ccd_temp1`** = `float(rng.normal(CCD_TEMP_MEAN_C, CCD_TEMP_STD_C))` per frame.
  Not a constant.
- **RNG draw order** per frame (strictly): θ (1) → axis (3) → etalon (4) → CCD (1).
  Do not reorder — would invalidate seed reproducibility.
- **C6 check**: use `SIGMA_POINTING_ARCSEC * np.sqrt(2 / np.pi)` ≈ 3.99 as
  expected mean, not `SIGMA_POINTING_ARCSEC` directly.
- **C10/C11**: use `n_caldark * len(cal_trigger_indices)`, never
  `df_sched['orbit_number'].max()`.
- **NB01 always at `SCHED_DT_S = 10.0`** — never pass `obs_cadence_s`.
- **NB02a** needs explicit `h_target_km`.
- **No `build_synthetic_metadata()`** — construct `ImageMetadata` directly.
- **Timezone**: `t0 = pd.Timestamp(t_start, tz='UTC')`.

### Epilogue
```bash
git add PIPELINE_STATUS.md \
        validation/gen01_synthetic_metadata_generator_2026_04_16.py
git commit -m "feat(g01): exp_time prompt, exp_unit=38500, CCD noise, C1-C13 pass

exp_time_cts prompt (default 8000); exp_unit fixed to 38500 (hw register)
ccd_temp1: N(-10, 1°C) per frame (was constant)
RNG order: PE(4) -> etalon(4) -> CCD(1)
30-day run: science=115860, cal=2275, dark=2275, total=120410

Also updates PIPELINE_STATUS.md"
```
