# G01 — Synthetic Metadata Generator Specification

**Spec ID:** G01
**Spec file:** `docs/specs/G01_synthetic_metadata_generator_2026-04-16.md`
**Project:** WindCube FPI Science Operations Center Pipeline
**Institution:** NCAR / High Altitude Observatory (HAO)
**Status:** ✓ Complete — all 13 checks pass (1-day test confirmed)
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
> - v1 (2026-04-16): Initial — latitude-threshold instrument state, one record
>   per NB01 epoch.
> - v2 (2026-04-16): CONOPS model — science at user cadence within ±lat_band;
>   cal+dark triggered at 60°N ascending; cadence, n_caldark, lat_band prompts.
> - v3 (2026-04-16): Removed all S-number references; replaced with NB01,
>   NB02a, P01 module names. Added CONOPS document citation.
> - v4 (2026-04-16): Implementation corrections from G01 report:
>   (a) §4.1 — pointing error angle follows a **half-normal** distribution
>   (|θ| not θ); corrected expected mean and std; C6 updated accordingly.
>   (b) §7.2 C10/C11 — expected frame counts now use n_complete_orbits
>   (orbits that reached 60°N within the window), not the raw orbit count
>   which includes the terminal partial orbit.
>   (c) Script location changed from `scripts/` to `validation/`; default
>   `output_dir` updated to `"validation/outputs/"`.

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
- Tangent points, `v_rel`, or wind observations (NB02b, NB02c) — added by
  Z02/Z03.
- Pixel data generation.
- Calls to P01 `build_synthetic_metadata()` — that function requires NB02c
  wind outputs not computed here. G01 constructs `ImageMetadata` directly.

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
Output directory     [  default validation/outputs/ ] : _
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
| `output_dir` | Output directory | `"validation/outputs/"` | str | Any writable path |
| `rng_seed` | NumPy RNG seed | 42 | int | ≥ 0 |

**Constraint warnings** (printed to console, not errors):

- If `lat_band_deg ≥ 60.0`: warn that the science band overlaps the 60°N
  cal/dark trigger latitude. Cal/dark type takes precedence at trigger epochs.
- If `2 × n_caldark × obs_cadence_s > 1200`: warn that the cal/dark sequence
  duration (> 20 min) may extend into the orbit's descending high-latitude arc.

---

## 3. Orbit propagation and CONOPS scheduling

All observation schedule logic in this section is governed by the WindCube
Mission CONOPS Document cited in the header. When the CONOPS is revised,
update the section references noted below before re-implementing.

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
a_m       = WGS84_A_M + altitude_km * 1e3
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

*(CONOPS reference: [TBD — insert document section for science band definition
and observation cadence, e.g. §4.2])*

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

A single orbit has multiple science-band transits. Each transit resets
`band_entry_i` so spacing is consistent within each transit.

**Science frame instrument state:**

| Field | Value | Notes |
|-------|-------|-------|
| `gpio_pwr_on` | `[0, 0, 0, 0]` | Shutter open |
| `lamp_ch_array` | `[0, 0, 0, 0, 0, 0]` | No lamps |
| `shutter_status` | `'open'` | Derived |
| `img_type` | `'science'` | Derived via `_classify_img_type()` |
| `exp_time` | `500` cs (5 s) | Stub — confirm against CONOPS |

### 3.3 Calibration and dark trigger and sequence

*(CONOPS reference: [TBD — insert document section for calibration schedule,
e.g. §4.3 and §4.4])*

The **cal/dark trigger** fires once per orbit at the first epoch where the
spacecraft geodetic latitude ascends through `CAL_TRIGGER_LAT_DEG = 60.0°N`.
This is a named constant; update it and the CONOPS version citation in the
header if the CONOPS changes this value.

**Trigger detection:**

```python
CAL_TRIGGER_LAT_DEG = 60.0   # CONOPS fixed parameter — see header citation

cal_trigger_indices = []
lat = df_sched['lat_deg'].values

for i in range(1, len(lat)):
    if (lat[i]     >  CAL_TRIGGER_LAT_DEG
            and lat[i-1] <= CAL_TRIGGER_LAT_DEG
            and lat[i]   >  lat[i-1]):         # ascending: lat increasing
        cal_trigger_indices.append(i)
```

At most one trigger per orbit; the 97.44° SSO always has exactly one
ascending 60°N crossing per orbit.

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
| `exp_time` | `120` cs (1.2 s) | Stub — confirm against CONOPS |

**Dark instrument state:**

| Field | Value | Notes |
|-------|-------|-------|
| `gpio_pwr_on` | `[1, 0, 0, 1]` | Shutter closed |
| `lamp_ch_array` | `[0, 0, 0, 0, 0, 0]` | No lamps |
| `shutter_status` | `'closed'` | Derived |
| `img_type` | `'dark'` | Derived via `_classify_img_type()` |
| `exp_time` | `500` cs (5 s) | Stub — confirm against CONOPS |

### 3.4 `img_type` derivation — always use P01 logic

```python
def _classify_img_type(lamp_ch_array: list, gpio_pwr_on: list) -> str:
    """P01 classification logic — keep in sync with p01_image_metadata_2026_04_06.py."""
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

Cal/dark takes precedence over science at any overlap (possible when
`lat_band_deg ≥ 60°`). `frame_sequence` is the 0-based index among all
observation frames for each orbit, ordered by epoch:

```python
orbit_counter   = {}
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

The AOCS pointing error is modelled as a small random rotation. The signed
rotation angle θ is drawn from a zero-mean Gaussian:

```
θ ~ N(0, σ_θ)    where σ_θ = 5 arcsec × (π/648000) ≈ 2.4241 × 10⁻⁵ rad
```

The quaternion is constructed as:

```python
SIGMA_POINTING_ARCSEC = 5.0
sigma_theta_rad = SIGMA_POINTING_ARCSEC * np.pi / 648000.0

theta = rng.normal(0.0, sigma_theta_rad)          # signed; may be negative
raw   = rng.standard_normal(3)
n_hat = raw / np.linalg.norm(raw)                 # uniform random axis
half_theta = theta / 2.0
qe = [n_hat[0]*np.sin(half_theta),
      n_hat[1]*np.sin(half_theta),
      n_hat[2]*np.sin(half_theta),
      np.cos(half_theta)]
qe = [c / np.linalg.norm(qe) for c in qe]        # normalise
```

**Distribution of the unsigned rotation angle |θ|:**

The signed angle θ ~ N(0, σ_θ), but the physical rotation magnitude is |θ|
since a negative θ about axis **n** is identical to a positive θ about −**n**
(and **n** is drawn uniformly on the sphere). Therefore the unsigned angle
follows a **half-normal** distribution:

```
|θ| ~ half-normal(σ = σ_θ)

E[|θ|]  = σ_θ · √(2/π)     ≈ 5 · 0.7979 ≈ 3.99 arcsec
Std[|θ|] = σ_θ · √(1−2/π)  ≈ 5 · 0.6028 ≈ 3.01 arcsec
```

P01's `compute_adcs_quality_flag()` extracts the unsigned angle from the
quaternion vector part as `2·arcsin(‖qe_xyz‖) ≈ |θ|` for small angles. The
reported pointing error statistics in the console output and C6 check both
refer to this half-normal quantity. The σ_θ = 5 arcsec parameter controls
the underlying Gaussian draw, not the directly observable half-normal
statistics.

**SLEW_IN_PROGRESS threshold:** P01 sets this flag when the unsigned angle
exceeds 30 arcsec (AKE budget, SYS.108 from SI-UCAR-WC-RP-004). With
σ_θ = 5 arcsec, 30 arcsec corresponds to a 6σ draw. Essentially no frames
will be flagged over a 30-day campaign.

**RNG draw order per observation frame** (fixed — do not change order):
1. `theta` — 1 draw via `rng.normal`
2. `raw` axis — 3 draws via `rng.standard_normal(3)`
3. `etalon_temps` — 4 draws via `rng.normal(24.0, 0.1, 4)`

### 4.2 Etalon temperatures

```python
etalon_temps = rng.normal(24.0, 0.1, size=4).tolist()   # °C
```

Mean 24.0 °C (ambient lab temperature); σ = 0.1 °C (thermistor noise).

### 4.3 CCD temperature

Constant: `ccd_temp1 = -18.0` °C (nominal on-orbit set point).

---

## 5. `ImageMetadata` field assignment

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
| `pointing_error` | §4.1 | Gaussian error quaternion, σ_θ = 5 arcsec |
| `obs_mode` | NB01 orbit parity | `'along_track'` / `'cross_track'` |
| `ccd_temp1` | §4.3 | `-18.0` °C |
| `etalon_temps` | §4.2 | `rng.normal(24.0, 0.1, 4).tolist()` |
| `shutter_status` | P01 §3.4 | derived from `gpio_pwr_on` |
| `gpio_pwr_on` | §3.2–3.3 | frame-type dependent |
| `lamp_ch_array` | §3.2–3.3 | frame-type dependent |
| `lamp1_status` | P01 rule | `"on" if lamp_ch[0] or lamp_ch[1] else "off"` |
| `lamp2_status` | P01 rule | `"on" if lamp_ch[2] or lamp_ch[3] else "off"` |
| `lamp3_status` | P01 rule | `"on" if lamp_ch[4] or lamp_ch[5] else "off"` |
| `orbit_number` | NB01 §3.1 | 1-based, from elapsed time |
| `frame_sequence` | §3.5 | 0-based within orbit observation list |
| `orbit_parity` | NB01 §3.1 | `'along_track'` / `'cross_track'` |
| `adcs_quality_flag` | P01 `compute_adcs_quality_flag()` | from pointing_error |
| `is_synthetic` | constant | `True` |
| `noise_seed` | frame index | 0-based position in `obs_indices` |
| `truth_v_los` | — | `None` (NB02c not run) |
| `truth_v_zonal` | — | `None` |
| `truth_v_meridional` | — | `None` |
| `tangent_lat` | — | `None` (NB02b not run) |
| `tangent_lon` | — | `None` |
| `tangent_alt_km` | — | `None` |
| `etalon_gap_mm` | — | `None` |
| `etalon_gap_corrected_mm` | — | `None` |
| `grafana_record_id` | — | `None` |
| `dark_subtracted` | default | `False` |
| `dark_n_frames` | default | `0` |
| `dark_lua_timestamp` | — | `None` |
| `dark_etalon_temp_mean` | — | `None` |

---

## 6. Output files

### 6.1 Naming convention

```
{output_dir}/GEN01_{t_start_compact}_{duration_days:05.1f}d_seed{rng_seed:04d}
```

where `t_start_compact = t_start[:10].replace('-', '')`. Example (defaults):

```
validation/outputs/GEN01_20270101_030.0d_seed0042.npy
validation/outputs/GEN01_20270101_030.0d_seed0042.csv
```

### 6.2 `.npy` — object array of dicts

```python
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

---

## 7. Progress reporting and verification

### 7.1 Console output

The PE statistics reported are for the unsigned rotation angle |θ|, which
follows a half-normal distribution. Expected values are shown in brackets:

```
=== G01 — WindCube Synthetic Metadata Generator ===

Parameters:
  Start epoch      : 2027-01-01T00:00:00 UTC
  Duration         : 30.0 days
  Science band     : ±40.0°
  Obs. cadence     : 10.0 s requested → 10.0 s actual (step=1)
  Cal/dark per orb : 5 + 5 = 10 frames  (100.0 s sequence)
  Cal trigger lat  : 60.0°N ascending  [CONOPS — see spec header]
  S/C altitude     : 510.0 km
  Tangent ht       : 250.0 km
  T_orbit          : 5689.4 s (94.82 min)
  NB01 sched rows  : 259201  (10 s grid, 30 days)
  Total orbits     : ~456  (complete orbits reaching 60°N: ~455)
  RNG seed         : 42
  Output dir       : validation/outputs/

Building NB01 orbit schedule ... done

Observation schedule:
  Science frames  : NNNNNN
  Cal frames      : NNNN    (5 per complete orbit × ~455 complete orbits)
  Dark frames     : NNNN    (5 per complete orbit × ~455 complete orbits)
  Total frames    : NNNNNN

Building NB02a attitude quaternions + P01 metadata ...
  [====================] NNNNNN/NNNNNN ... done

Image type verification:
  science : NNNNNN
  cal     : NNNN
  dark    : NNNN

ADCS quality flags (P01):
  GOOD              : NNNNNN
  SLEW_IN_PROGRESS  : 0    (expected ~0)

Pointing error — unsigned rotation angle |θ| (half-normal distribution):
  Mean  : X.XX arcsec   (expected σ·√(2/π) ≈ 3.99 arcsec for σ=5)
  Std   : X.XX arcsec   (expected σ·√(1−2/π) ≈ 3.01 arcsec for σ=5)
  Max   : XX.X arcsec

Etalon temperature stats (°C):
  Mean  : XX.XX   (expected 24.00)
  Std   :  X.XX   (expected  0.10)

Output files:
  validation/outputs/GEN01_20270101_030.0d_seed0042.npy  (XX.X MB)
  validation/outputs/GEN01_20270101_030.0d_seed0042.csv  (XX.X MB)

G01 complete.
```

### 7.2 Verification checks

Non-blocking — print `PASS` or `FAIL — <reason>` for each:

| Check | Criterion | Notes |
|-------|-----------|-------|
| C1 | NB01 sched rows ≈ `duration_s / SCHED_DT_S` ± 2 | |
| C2 | All `orbit_parity` values are `'along_track'` or `'cross_track'` | |
| C3 | No NaN in any `att_q_*` column (NB02a) | |
| C4 | All attitude quaternions unit norm to 1e-6 (NB02a) | |
| C5 | All pointing error quaternions unit norm to 1e-6 | |
| C6 | Mean of unsigned PE angle within 20% of `σ·√(2/π)` ≈ 3.99 arcsec | Half-normal; not 5 arcsec |
| C7 | Etalon temp mean within 0.05°C of 24.0°C | |
| C8 | P01 `adcs_quality_flag == 0` for > 99.9% of frames | |
| C9 | `img_type` values are only `'science'`, `'cal'`, `'dark'` | |
| C10 | Cal frame count ≈ `n_caldark × n_complete_orbits` (±5%) | Use `len(cal_trigger_indices)`, not `n_orbits` |
| C11 | Dark frame count ≈ `n_caldark × n_complete_orbits` (±5%) | Same; terminal partial orbit may not reach 60°N |
| C12 | No science frame has `lat_deg > lat_band_deg + 1°` | 1° tolerance for 10 s grid snap |
| C13 | P01 round-trip: `ImageMetadata(**np.load(npy_path, allow_pickle=True)[0])` succeeds | |

**C10/C11 implementation note:** Use `n_complete_orbits = len(cal_trigger_indices)`,
not the raw orbit count from the DataFrame. The terminal partial orbit — the
last orbit still in progress at `t_start + duration_days` — may not have
reached 60°N before the window closed and therefore has no cal/dark trigger.
Using `n_orbits_total` for the expected count would cause a spurious FAIL for
any simulation whose window ends mid-orbit before the trigger latitude.

---

## 8. File location in repository

```
soc_sewell/
├── validation/
│   ├── gen01_synthetic_metadata_generator_2026_04_16.py
│   └── outputs/
│       ├── GEN01_20270101_030.0d_seed0042.npy
│       └── GEN01_20270101_030.0d_seed0042.csv
└── docs/specs/
    └── G01_synthetic_metadata_generator_2026-04-16.md
```

The script lives in `validation/` alongside Z01–Z03 validation scripts.
It is a user-facing executable, not an importable module, and has no
`__init__.py`.

---

## 9. Instructions for Claude Code

### Preamble

```bash
cat PIPELINE_STATUS.md
```

### Prerequisite reads

1. This entire spec.
2. NB01 spec and implementation — `propagate_orbit` signature; output columns
   `epoch`, `lat_deg`, `lon_deg`, `alt_km`, `pos_eci_*`, `vel_eci_*`.
3. NB02a spec and implementation — `compute_los_eci(pos_eci, vel_eci,
   look_mode, h_target_km)` returns `(los_eci, q)`. Iterative depression
   angle (post-fix). Always pass `h_target_km` explicitly.
4. P01 spec and implementation — `ImageMetadata` dataclass; `AdcsQualityFlags`;
   `compute_adcs_quality_flag()`.
5. `CLAUDE.md` at repo root.

### Prerequisite tests

```bash
pytest tests/test_nb01_orbit_propagator.py -v    # 8/8 must pass
pytest tests/test_nb02_geometry_2026_04_16.py -v # 10/10 must pass
pytest tests/test_s19_p01_metadata.py -v         # 8/8 must pass
```

### Implementation steps

1. Create `validation/gen01_synthetic_metadata_generator_2026_04_16.py`
   with module header:
   ```python
   """
   G01 — Synthetic Metadata Generator.

   Spec:      docs/specs/G01_synthetic_metadata_generator_2026-04-16.md
   Spec date: 2026-04-16
   Generated: YYYY-MM-DD
   Tool:      Claude Code
   CONOPS:    [TBD — insert document ID and version when known]
   Usage:     python validation/gen01_synthetic_metadata_generator_2026_04_16.py
   """
   ```
   Copy the `_project_root` sys.path pattern from NB01 or NB02a verbatim.

2. **Imports:**
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

3. **Constants block:**
   ```python
   SCHED_DT_S              = 10.0   # NB01 cadence — fixed
   CAL_TRIGGER_LAT_DEG     = 60.0   # CONOPS ascending trigger
   SIGMA_POINTING_ARCSEC   =  5.0   # Gaussian σ for θ draw
   ETALON_TEMP_MEAN_C      = 24.0
   ETALON_TEMP_STD_C       =  0.1
   CCD_TEMP_C              = -18.0
   ```

4. **`_prompt(msg, default, cast, lo, hi)`** — re-prompts on bad input;
   blank → default.

5. **`_pointing_error_quat(rng, sigma_arcsec=5.0)` helper** — §4.1 exactly.
   Draw order: `rng.normal` (θ), then `rng.standard_normal(3)` (axis).
   Returns `[x, y, z, w]` normalised.

6. **`_classify_img_type(lamp_ch_array, gpio_pwr_on)`** — §3.4 P01 logic.

7. **`_instrument_state(frame_type)`** — returns
   `(gpio_pwr_on, lamp_ch_array, exp_time_cs)` per §3.2–3.3.

8. **`_build_schedule(df_sched, lat_band_deg, n_caldark, step)`** — returns
   `(obs_indices, frame_types)`:
   - Science: §3.2 transit-counter loop.
   - Cal/dark: §3.3 ascending-crossing loop; for each trigger `t0` append
     `t0 + k*step` (cal) then `t0 + (n+k)*step` (dark) for `k = 0..n-1`;
     skip indices ≥ `len(df_sched)`.
   - Remove science indices that are also cal/dark (§3.5).
   - Sort and return. Also return `n_complete_orbits = len(cal_trigger_indices)`
     for use in C10/C11.

9. **Main loop** — iterate `zip(obs_indices, frame_types)`:
   - NB01 row: `row = df_sched.loc[idx]`
   - NB02a: `compute_los_eci(pos, vel, look_mode, h_target_km=h_target_km)`
   - RNG draws: pointing error first (4 total), then etalon temps (4).
   - Construct `ImageMetadata(...)` directly.
   - `adcs_quality_flag` from P01 `compute_adcs_quality_flag()`.

10. **C6 check** — compute unsigned PE angles from quaternion vector parts,
    check mean is within 20% of `SIGMA_POINTING_ARCSEC * np.sqrt(2/np.pi)`.
    **Do not** check against `SIGMA_POINTING_ARCSEC` directly.

11. **C10/C11 check** — expected count = `n_caldark * n_complete_orbits`
    where `n_complete_orbits = len(cal_trigger_indices)`. **Do not** use
    `df_sched['orbit_number'].max()` — that includes the terminal partial orbit.

12. Progress bar, CSV construction, and `if __name__ == '__main__': main()`
    per earlier spec sections.

### Critical rules

- **NB01 always at `SCHED_DT_S = 10.0`** — never pass `obs_cadence_s`.
- **NB02a needs explicit `h_target_km`** — iterative depression angle.
- **RNG draw order**: θ (1) → axis (3) → etalon (4) per frame, always.
- **No P01 `build_synthetic_metadata()`** — construct `ImageMetadata` directly.
- **Timezone**: `t0 = pd.Timestamp(t_start, tz='UTC')`.

### Epilogue

```bash
git add PIPELINE_STATUS.md \
        validation/gen01_synthetic_metadata_generator_2026_04_16.py
git commit -m "feat(g01): synthetic metadata generator, C1-C13 pass

Implements: G01_synthetic_metadata_generator_2026-04-16.md
PE distribution: half-normal (σ=5 arcsec Gaussian draw, |θ| observed)
C10/C11 use n_complete_orbits from cal_trigger_indices
1-day test: science=3876, cal=75, dark=75

Also updates PIPELINE_STATUS.md"
```
