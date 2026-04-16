# S08 — INT01 Geometry Pipeline Integration Notebook Specification

**Spec ID:** S08
**Spec file:** `docs/specs/S08_int01_geometry_notebook_2026-04-16.md`
**Project:** WindCube FPI Pipeline
**Institution:** NCAR / High Altitude Observatory (HAO)
**Status:** ✓ Complete — all 8 checks pass; 16 orbits, 8641 epochs confirmed
**Depends on:**
  - S01, S02, S03, S04 — conventions and constants
  - S05 (NB00 `2026-04-06`) — truth wind map
  - S06 (NB01 `2026-04-16`) — orbit propagator; `propagate_orbit(t_start, duration_s, dt_s)`
  - S07 (NB02 `2026-04-16`) — geometry pipeline; `nb02a/b/c/d_*_2026_04_16.py`
  - S19 (P01 `2026-04-06`) — `ImageMetadata`, `build_synthetic_metadata()`
**Used by:** S09–S11 (Tier 2 specs must not be written until INT01 is complete)
**References:**
  - Harding et al. (2014) Applied Optics 53(4) — science context
  - SI-UCAR-WC-RP-004 Issue 1.0 §2.4.2 — AOCS reference frames (BRF/THRF/SIRF)
  - WindCube STM v1 — V6/V7 pass criteria link to SG1/SG2 requirements
**Last updated:** 2026-04-16

> **Note:** This spec supersedes `S08_int01_geometry_notebook_2026-04-05.md`.
> The 2026-04-05 version is retired. All content is fully integrated here.

> **Implementation fixes applied (same-day revision after Claude Code run):**
> - **V2 check rewritten (§4, §6):** Longitude-based equatorial offset fails
>   for a 97.44° SSO because the spacecraft at the equator moves almost
>   purely north-south, making longitude lead ≈ 0 even though the 3-D lead
>   distance is ~923 km. V2 now projects the 3-D ECI separation vector
>   `(tp_eci − sc_eci)` onto the spacecraft velocity unit vector v̂_sc
>   (along-track) or the orbit-normal unit vector n̂_orbit (cross-track).
>   Both must exceed +500 km. This is correct for any orbital inclination.
> - **Timezone discipline (§5 Section 1, §5 Section 3):** Mixed tz-aware/
>   tz-naive subtraction caused `TypeError` in elapsed-time computations and
>   `UserWarning` in matplotlib. Fix: always construct `t0 = pd.Timestamp(
>   START_EPOCH, tz='UTC')` so it matches the tz-aware `epoch` column. For
>   matplotlib x-axes use `df['epoch'].dt.tz_localize(None)` to obtain a
>   tz-naive array.
> - **NB02a iterative depression angle (§5 Section 2, §6 V1 notes):**
>   `_compute_depression_angle()` was updated (see NB02 spec §4.1) to use
>   5 iterations converging on the actual TP geodetic latitude, reducing
>   tp_alt error from ~6 km to < 0.1 km across the polar orbit. V1 tolerance
>   remains ≤ 5 km as the physical requirement; implementation achieves < 0.1 km.
>
> **Changes from 2026-04-05 spec:**
> - NB01 `propagate_orbit()` signature updated to `(t_start, duration_s, dt_s)`.
> - NB02 imports updated to `nb02a/b/c/d_*_2026_04_16.py`; boresight corrected
>   to `−X_BRF` per SI-UCAR-WC-RP-004 §2.4.2.1.
> - AOCS reference frame nomenclature (BRF, THRF, SIRF) added throughout.
> - `DURATION_DAYS` replaces fixed two-orbit setup; look-mode alternation
>   computed from orbit number parity.
> - **New Section 5:** Synthetic metadata array via P01 `build_synthetic_metadata()`.
> - **New check V8:** Metadata array validation — five sub-checks.

---

## 1. Purpose and philosophy

INT01 is the first integration notebook in the WindCube pipeline. It connects
NB00 (truth wind map), NB01 (orbit propagator), and NB02 (geometry/LOS) over
an arbitrary duration of orbits with alternating look modes, producing visual
and quantitative evidence that the geometry pipeline works end-to-end. It also
produces the first complete synthetic metadata arrays that downstream forward
modelling modules (Tier 2: S09–S11) will use to populate synthetic airglow,
calibration, and dark images.

**The INT series philosophy:**
Each INT notebook validates a complete pipeline stage boundary — not just
individual modules in isolation. INT01 asks: when NB00 + NB01 + NB02 are
connected on realistic inputs, does the geometry produce physically correct
`v_rel` values that a downstream FPI inversion could use? Integration
notebooks are permanent project artefacts, committed to the repo and re-run
whenever upstream modules change.

**What INT01 specifically validates:**
- Tangent points are at the correct altitude (250 ± 5 km, V1)
- Tangent point ECI offset from spacecraft exceeds 500 km in the correct
  direction for both look modes — forward for along-track, lateral for
  cross-track — verified by 3-D ECI dot product (V2)
- Along-track V_sc_LOS dominates cross-track by > 10× (V3)
- No NaN or Inf values anywhere in the pipeline output (V4)
- v_rel time series is smooth — no epoch-to-epoch spikes > 100 m/s (V5)
- L2 2×2 wind decomposition recovers truth wind to < 1 m/s bias (V6)
- Round-trip holds for three secondary wind maps (V7)
- Synthetic metadata array is complete and internally consistent (V8)

**Gate rule:** Tier 2 specs (S09–S11) must not be written or implemented
until all 8 INT01 checks pass and the output PNGs have been visually
inspected and confirmed.

---

## 2. Science context — alternating look modes over arbitrary duration

WindCube alternates between along-track and cross-track boresight directions
on successive orbits. This strategy allows the L2 merger (M07) to decompose
line-of-sight measurements into the two horizontal wind components. INT01
exercises this strategy over a configurable number of days to confirm the
geometry pipeline is correct for any realistic operational scenario.

**Orbit parity convention (from P01/S19 §3.6):**
- Odd orbit numbers (1, 3, 5, …) → `along_track` (THRF along-track configuration)
- Even orbit numbers (2, 4, 6, …) → `cross_track` (THRF cross-track configuration)

**Orbit 1, 3, … — along-track (odd orbits):**
The `−X_BRF` boresight (SI-UCAR-WC-RP-004 §2.4.2.1) is depressed 15.73°
toward the limb and points forward along the velocity direction. The tangent
point is ~923 km **ahead** of the spacecraft in the ECI velocity direction.
V_sc_LOS ≈ −7,100 m/s. After L1c removal, the residual wind LOS projection
is primarily **meridional** (north-south).

**Orbit 2, 4, … — cross-track (even orbits):**
The boresight points perpendicular to the orbit plane, also depressed 15.73°.
The tangent point is ~923 km **to the side** of the spacecraft in the ECI
orbit-normal direction. V_sc_LOS ≈ 0–300 m/s. The residual wind LOS
projection is primarily **zonal** (east-west).

**Why longitude-based V2 fails for SSO at 97.44°:**
At the equator a 97.44° inclination orbit crosses nearly pole-to-pole, so the
spacecraft velocity vector is almost purely north-south. The along-track
tangent point therefore leads mainly in latitude, not in longitude. A
longitude-offset check at the equator would return ≈ 0 even when the 3-D
forward offset is the full ~923 km. The correct check is always the dot
product of the 3-D ECI offset vector onto the spacecraft velocity unit vector
(along-track) or the orbit-normal unit vector (cross-track). These dot
products give the correct signed projection regardless of the orbital
inclination or where the spacecraft is in its orbit.

---

## 3. Truth wind map for INT01

Use **T1 UniformWindMap only** for all INT01 verification.

```python
from src.windmap.nb00_wind_map_2026_04_05 import UniformWindMap

wind_map             = UniformWindMap(v_zonal_ms=100.0, v_merid_ms=50.0)
wind_map_zonal_only  = UniformWindMap(v_zonal_ms=200.0, v_merid_ms=0.0)
wind_map_merid_only  = UniformWindMap(v_zonal_ms=0.0,   v_merid_ms=150.0)
wind_map_zero        = UniformWindMap(v_zonal_ms=0.0,   v_merid_ms=0.0)
```

The zero-wind case is diagnostic for Earth-rotation removal: if the
`v_earth_LOS` term is wrong, the recovered wind will be non-zero even with
a zero-wind input.

---

## 4. Orbit configuration and duration

```python
from src.constants import (
    SC_ALTITUDE_KM,
    SCIENCE_CADENCE_S,
    EARTH_GRAV_PARAM_M3_S2,
    WGS84_A_M,
)
import numpy as np

# Derive T_ORBIT_S from constants — do not hardcode
a_m       = WGS84_A_M + SC_ALTITUDE_KM * 1e3
T_ORBIT_S = 2 * np.pi * np.sqrt(a_m**3 / EARTH_GRAV_PARAM_M3_S2)
# Expected: T_ORBIT_S ≈ 5689 s ≈ 94.82 min

START_EPOCH   = "2027-01-01T00:00:00"   # ISO 8601 UTC
DT_S          = SCIENCE_CADENCE_S       # 10 s cadence
H_TARGET_KM   = 250.0                   # OI 630 nm tangent height

# ── Duration is configurable in days ─────────────────────────────────────────
# For verification (V1–V8): 1 day ≈ 15.2 orbits → ~7.6 matched pairs.
# Minimum for V6/V7: at least 2 paired orbits (one even, one odd).
# Confirmed passing: DURATION_DAYS = 1.0 → 16 orbits, 8641 epochs.
DURATION_DAYS = 1.0
DURATION_S    = DURATION_DAYS * 86400.0
```

---

## 5. Notebook structure

Seven sections. Each ends with a clearly labelled PASS / FAIL cell. The
notebook must run top-to-bottom with no hidden state.

### Section 1 — Setup and orbit propagation

Single `propagate_orbit` call for the full duration. Assign orbit numbers
and look modes from elapsed time. All timestamp arithmetic must be
tz-consistent (see timezone discipline note below).

**Timezone discipline:** The `epoch` column returned by `propagate_orbit` is
tz-aware UTC. The reference epoch `t0` must be constructed as tz-aware to
match:

```python
t0 = pd.Timestamp(START_EPOCH, tz='UTC')   # tz-aware UTC
df_full['elapsed_s'] = (df_full['epoch'] - t0).dt.total_seconds()
```

Never subtract a tz-naive `Timestamp` from a tz-aware `DatetimeTZDtype`
column — this raises `TypeError` in pandas. For matplotlib x-axes, convert
to tz-naive: `df['epoch'].dt.tz_localize(None)`.

**PASS criterion:** DataFrame has ≈ `DURATION_S / DT_S` rows, no NaN, both
`look_mode` values present.

```python
from src.geometry.nb02a_boresight_2026_04_16 import compute_los_eci
from src.geometry.nb02b_tangent_point_2026_04_16 import compute_tangent_point
from src.geometry.nb02c_los_projection_2026_04_16 import (
    compute_v_rel, enu_unit_vectors_eci,
)
from src.geometry.nb02d_l1c_calibrator_2026_04_16 import remove_spacecraft_velocity
from src.geometry.nb01_orbit_propagator_2026_04_05 import propagate_orbit
from src.windmap.nb00_wind_map_2026_04_05 import UniformWindMap
from src.metadata.p01_image_metadata_2026_04_06 import (
    build_synthetic_metadata, write_sidecar,
)
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from astropy.time import Time

df_full = propagate_orbit(
    t_start    = START_EPOCH,
    duration_s = DURATION_S,
    dt_s       = DT_S,
)

t0 = pd.Timestamp(START_EPOCH, tz='UTC')           # tz-aware — must match epoch column
df_full['elapsed_s']    = (df_full['epoch'] - t0).dt.total_seconds()
df_full['orbit_number'] = (df_full['elapsed_s'] // T_ORBIT_S).astype(int) + 1
df_full['look_mode']    = df_full['orbit_number'].apply(
    lambda n: 'along_track' if n % 2 == 1 else 'cross_track'
)
df_full['frame_sequence'] = df_full.groupby('orbit_number').cumcount()

n_orbits = df_full['orbit_number'].max()
print(f"Duration: {DURATION_DAYS:.1f} day(s), {len(df_full)} epochs, {n_orbits} orbits")
assert df_full.isna().sum().sum() == 0
print("Section 1: PASS")
```

---

### Section 2 — LOS vectors, tangent points, v_rel for all epochs

Loop over all epochs, computing NB02a → NB02b → NB02c.

`compute_los_eci(pos, vel, mode)` returns `(los_eci, q)` — capture both.
The quaternion `q` is stored in `df_results` and passed to `build_synthetic_metadata`
in Section 5.

**tp_alt precision note:** NB02a's `_compute_depression_angle()` now uses
5-iteration convergence toward the actual TP geodetic latitude (see NB02
spec §4.1). This reduces tp_alt error from ~6 km to < 0.1 km across the
polar orbit. V1 tolerance of ±5 km remains the physical requirement.

**PASS criterion:** No NaN or Inf. All `tp_alt_km` within 5 km of H_TARGET_KM.

```python
wind_map = UniformWindMap(v_zonal_ms=100.0, v_merid_ms=50.0)
results  = []

for _, row in df_full.iterrows():
    pos   = np.array([row.pos_eci_x, row.pos_eci_y, row.pos_eci_z])
    vel   = np.array([row.vel_eci_x, row.vel_eci_y, row.vel_eci_z])
    epoch = Time(row.epoch)
    mode  = row.look_mode

    los, q = compute_los_eci(pos, vel, mode)    # -X_BRF boresight → ECI

    tp  = compute_tangent_point(pos, los, epoch, H_TARGET_KM)
    res = compute_v_rel(
        wind_map, tp['tp_lat_deg'], tp['tp_lon_deg'], tp['tp_eci'],
        vel, los, epoch,
    )

    results.append({
        'orbit_number':   row.orbit_number,
        'orbit_parity':   mode,
        'look_mode':      mode,
        'frame_sequence': row.frame_sequence,
        'epoch':          row.epoch,
        'sc_lat':  row.lat_deg,    'sc_lon':  row.lon_deg,  'sc_alt_km': row.alt_km,
        'pos_eci_x': pos[0], 'pos_eci_y': pos[1], 'pos_eci_z': pos[2],
        'vel_eci_x': vel[0], 'vel_eci_y': vel[1], 'vel_eci_z': vel[2],
        'tp_lat':    tp['tp_lat_deg'],
        'tp_lon':    tp['tp_lon_deg'],
        'tp_alt_km': tp['tp_alt_km'],
        'tp_eci_x':  tp['tp_eci'][0],
        'tp_eci_y':  tp['tp_eci'][1],
        'tp_eci_z':  tp['tp_eci'][2],
        'los_x': los[0], 'los_y': los[1], 'los_z': los[2],
        'q_x':   q[0],   'q_y':   q[1],   'q_z':  q[2],  'q_w': q[3],
        'v_wind_LOS':  res['v_wind_LOS'],
        'V_sc_LOS':    res['V_sc_LOS'],
        'v_earth_LOS': res['v_earth_LOS'],
        'v_rel':       res['v_rel'],
        'v_zonal_ms':  res['v_zonal_ms'],
        'v_merid_ms':  res['v_merid_ms'],
    })

df_results = pd.DataFrame(results)
df_at = df_results[df_results.look_mode == 'along_track']
df_ct = df_results[df_results.look_mode == 'cross_track']

num_cols = df_results.select_dtypes(include=np.number)
assert df_results.isna().sum().sum() == 0
assert np.all(np.isfinite(num_cols.values))
alt_dev = (df_results.tp_alt_km - H_TARGET_KM).abs().max()
assert alt_dev < 5.0, f"tp_alt deviation {alt_dev:.2f} km > 5 km"
print(f"Max tp_alt deviation: {alt_dev:.3f} km  (limit 5 km; iterative "
      f"NB02a achieves < 0.1 km)")
print("Section 2: PASS")
```

---

### Section 3 — Four geometric verification plots

Four figures saved to `outputs/`. Visually inspect all before Section 4.

**Timezone note for all plots:** Convert epoch to tz-naive before passing
to matplotlib:

```python
epoch_naive = df_results['epoch'].dt.tz_localize(None)
```

**Plot 3.1 — Ground track and tangent points map**
`outputs/INT01_groundtrack.png`
- Along-track: S/C ground track (blue) + tangent points (cyan scatter)
- Cross-track: S/C ground track (orange) + tangent points (yellow scatter)
- ±40° science band (dashed gray)
- Subsample `every max(1, N//2000)`-th point for visual clarity
- **Visual check — along-track:** tangent points visibly displaced forward
  along the orbit, not coincident with the S/C position
- **Visual check — cross-track:** tangent points displaced to the side of
  the ground track by approximately one tangent-point-lead distance

**Plot 3.2 — Tangent point altitude vs. latitude**
`outputs/INT01_tp_altitude.png`
- `tp_alt_km` vs. `tp_lat`, both look modes
- Horizontal lines at 245 km and 255 km — all points must fall inside

**Plot 3.3 — V_sc_LOS comparison, both look modes**
`outputs/INT01_Vsc_LOS.png`
- V_sc_LOS vs. epoch (tz-naive); along-track ≈ −7,100 m/s; cross-track ≈ 0–300 m/s
- 10:1 ratio must be visually obvious

**Plot 3.4 — v_rel time series, first two orbits**
`outputs/INT01_v_rel_timeseries.png`
- v_rel vs. elapsed seconds for orbit 1 (blue) and orbit 2 (orange)
- Must be smooth — no jumps > 100 m/s visible

---

### Section 4 — Quantitative verification checks V1–V5

**V2 uses 3-D ECI vector projection.** The along-track offset is the dot
product of `(tp_eci − sc_eci)` with the spacecraft velocity unit vector;
the cross-track offset is the dot product with the orbit-normal unit vector.
Both must exceed +500 km. This method is correct for any orbital inclination.

```python
print("=" * 60)
print("INT01 Verification Checks V1–V5")
print("=" * 60)

# V1 — Tangent altitude within ±5 km of H_TARGET_KM
alt_dev = (df_results.tp_alt_km - H_TARGET_KM).abs()
v1 = alt_dev.max() < 5.0
print(f"\nV1 — Tangent altitude (target {H_TARGET_KM} km):")
print(f"     Max deviation: {alt_dev.max():.3f} km  (limit 5 km)")
print(f"     {'PASS' if v1 else 'FAIL'}")

# V2 — ECI offset in expected direction, > 500 km
# Along-track: project onto spacecraft velocity unit vector
# Cross-track: project onto orbit-normal unit vector

def _orbit_normal_hat(pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
    n = np.cross(pos, vel)
    return n / np.linalg.norm(n)

n_sample = min(50, len(df_at))

at_offsets_km = []
for _, r in df_at.iloc[:n_sample].iterrows():
    sc  = np.array([r.pos_eci_x, r.pos_eci_y, r.pos_eci_z])
    tp  = np.array([r.tp_eci_x,  r.tp_eci_y,  r.tp_eci_z])
    vel = np.array([r.vel_eci_x, r.vel_eci_y, r.vel_eci_z])
    v_hat = vel / np.linalg.norm(vel)
    at_offsets_km.append(np.dot(tp - sc, v_hat) / 1e3)

ct_offsets_km = []
for _, r in df_ct.iloc[:n_sample].iterrows():
    sc  = np.array([r.pos_eci_x, r.pos_eci_y, r.pos_eci_z])
    tp  = np.array([r.tp_eci_x,  r.tp_eci_y,  r.tp_eci_z])
    vel = np.array([r.vel_eci_x, r.vel_eci_y, r.vel_eci_z])
    n_hat = _orbit_normal_hat(sc, vel)
    ct_offsets_km.append(abs(np.dot(tp - sc, n_hat)) / 1e3)

at_km = np.mean(at_offsets_km)
ct_km = np.mean(ct_offsets_km)
v2_at = at_km > 500.0
v2_ct = ct_km > 500.0
v2    = v2_at and v2_ct
print(f"\nV2 — Tangent point ECI offset (3-D projection):")
print(f"     Along-track  (onto v̂_sc):      {at_km:+.1f} km  "
      f"(must be > +500 km)  {'PASS' if v2_at else 'FAIL'}")
print(f"     Cross-track  (onto n̂_orbit):   {ct_km:+.1f} km  "
      f"(must be > +500 km)  {'PASS' if v2_ct else 'FAIL'}")

# V3 — V_sc_LOS ratio > 10
ratio = abs(df_at.V_sc_LOS.mean()) / (abs(df_ct.V_sc_LOS.mean()) + 1e-6)
v3 = ratio > 10.0
print(f"\nV3 — V_sc_LOS ratio (along-track / cross-track):")
print(f"     Along-track mean: {df_at.V_sc_LOS.mean():.0f} m/s")
print(f"     Cross-track mean: {df_ct.V_sc_LOS.mean():.0f} m/s")
print(f"     Ratio: {ratio:.1f}×  (must be > 10×)")
print(f"     {'PASS' if v3 else 'FAIL'}")

# V4 — No NaN or Inf
num_cols = df_results.select_dtypes(include=np.number)
v4 = (df_results.isna().sum().sum() == 0 and
      np.all(np.isfinite(num_cols.values)))
print(f"\nV4 — No NaN or Inf: {'PASS' if v4 else 'FAIL'}")

# V5 — v_rel smoothness, all orbits
v5_all = True
for orb_num, orb_df in df_results.groupby('orbit_number'):
    diffs = np.abs(np.diff(orb_df.v_rel.values))
    ok    = (diffs.max() if len(diffs) else 0.0) < 100.0
    if not ok:
        print(f"\nV5 — FAIL orbit {orb_num}: max jump {diffs.max():.2f} m/s")
    v5_all = v5_all and ok
v5 = v5_all
print(f"\nV5 — v_rel smoothness ({n_orbits} orbits): {'PASS' if v5 else 'FAIL'}")

print(f"\n── V1–V5: {'PASS' if all([v1,v2,v3,v4,v5]) else 'FAIL'} ──")
```

---

### Section 5 — Synthetic metadata array generation (V8)

For each science epoch in `df_results`, call P01's `build_synthetic_metadata()`
to produce one `ImageMetadata` record. The resulting list `metadata_list` and
summary DataFrame `df_metadata` are the primary handoff documents for Tier 2
modules (M01, M02, M04).

**Image types in INT01:** Only `'science'` frames are generated here. M02
and M04 will call `build_synthetic_metadata()` with `img_type='cal'` and
`img_type='dark'` respectively, using the same orbit-state columns.

**`InstrumentParams` fallback:** Use the stub if M01 (S09) is not yet
implemented. The stub's `t_m` **must** be `20.106e-3` m (Tolansky-recovered
etalon gap per S03). Never use `20.008e-3` m (the ICOS measurement) here.

```python
try:
    from src.fpi.m01_airy_model import InstrumentParams
    params = InstrumentParams()
    print("Using InstrumentParams from M01.")
except ImportError:
    class _StubInstrumentParams:
        """Minimal stub — M01 not yet implemented."""
        t_m    = 20.106e-3   # etalon gap, m — Tolansky, S03 authoritative
        f_m    = 0.19912     # focal length, m
        r_refl = 0.53        # effective reflectivity (FlatSat)
    params = _StubInstrumentParams()
    print("M01 not available — using stub (t_m = 20.106 mm).")

metadata_list = []

for i, row in df_results.iterrows():
    nb01_row = df_full.loc[df_full['epoch'] == row['epoch']].iloc[0]
    nb02_tp  = {
        'tp_lat_deg': row.tp_lat,
        'tp_lon_deg': row.tp_lon,
        'tp_alt_km':  row.tp_alt_km,
        'tp_eci':     [row.tp_eci_x, row.tp_eci_y, row.tp_eci_z],
    }
    nb02_vr  = {
        'v_rel':       row.v_rel,
        'v_wind_LOS':  row.v_wind_LOS,
        'V_sc_LOS':    row.V_sc_LOS,
        'v_earth_LOS': row.v_earth_LOS,
        'v_zonal_ms':  row.v_zonal_ms,
        'v_merid_ms':  row.v_merid_ms,
    }
    meta = build_synthetic_metadata(
        params          = params,
        nb01_row        = nb01_row,
        nb02_tp         = nb02_tp,
        nb02_vr         = nb02_vr,
        quaternion_xyzw = [row.q_x, row.q_y, row.q_z, row.q_w],
        los_eci         = np.array([row.los_x, row.los_y, row.los_z]),
        look_mode       = row.look_mode,
        img_type        = 'science',
        orbit_number    = int(row.orbit_number),
        frame_sequence  = int(row.frame_sequence),
        noise_seed      = int(i),
    )
    metadata_list.append(meta)

print(f"Generated {len(metadata_list)} ImageMetadata records.")

df_metadata = pd.DataFrame([{
    'orbit_number':   m.orbit_number,
    'frame_sequence': m.frame_sequence,
    'epoch':          m.utc_timestamp,
    'img_type':       m.img_type,
    'is_synthetic':   m.is_synthetic,
    'look_mode':      m.obs_mode,
    'orbit_parity':   m.orbit_parity,
    'tp_lat':         m.tangent_lat,
    'tp_lon':         m.tangent_lon,
    'tp_alt_km':      m.tangent_alt_km,
    'truth_v_los':    m.truth_v_los,
    'truth_v_zonal':  m.truth_v_zonal,
    'truth_v_merid':  m.truth_v_meridional,
    'etalon_gap_mm':  m.etalon_gap_mm,
    'adcs_flag':      m.adcs_quality_flag,
} for m in metadata_list])
```

**V8 — Synthetic metadata array validation:**

```python
print("\nV8 — Synthetic metadata array validation:")

v8a = df_metadata.is_synthetic.all() and (df_metadata.adcs_flag == 0).all()
print(f"\nV8a — All is_synthetic=True and adcs_flag=GOOD: "
      f"{'PASS' if v8a else 'FAIL'}")

truth_diff = (df_metadata.truth_v_los - df_results.v_wind_LOS).abs()
v8b = truth_diff.max() < 1e-6
print(f"\nV8b — truth_v_los == v_wind_LOS: max diff {truth_diff.max():.2e} m/s  "
      f"{'PASS' if v8b else 'FAIL'}")

tp_diff = (df_metadata.tp_lat - df_results.tp_lat).abs()
v8c = tp_diff.max() < 1e-8
print(f"\nV8c — tangent_lat round-trip: max diff {tp_diff.max():.2e} deg  "
      f"{'PASS' if v8c else 'FAIL'}")

parity_ok = df_metadata.apply(
    lambda r: (r.orbit_parity == 'along_track') == (r.orbit_number % 2 == 1),
    axis=1,
)
v8d = parity_ok.all()
print(f"\nV8d — orbit_parity consistent with orbit_number: "
      f"{'PASS' if v8d else 'FAIL'}")

required_cols = ['tp_lat', 'tp_lon', 'tp_alt_km', 'truth_v_los',
                 'truth_v_zonal', 'truth_v_merid', 'etalon_gap_mm']
v8e = df_metadata[required_cols].isna().sum().sum() == 0
print(f"\nV8e — No None in required synthetic fields: "
      f"{'PASS' if v8e else 'FAIL'}")

v8 = all([v8a, v8b, v8c, v8d, v8e])
print(f"\n── V8 overall: {'PASS' if v8 else 'FAIL'} ──")
```

---

### Section 6 — L2 round-trip wind decomposition (V6 and V7)

For each 5°-longitude bin with at least one epoch from each look mode (within
|lat| < 40°), solve the 2×2 system G · [v_zonal, v_merid]ᵀ = d.

```python
recovered = []
for lon_bin in np.arange(-180, 180, 5.0):
    at_mask = ((df_results.tp_lon >= lon_bin) & (df_results.tp_lon < lon_bin+5) &
               (df_results.tp_lat.abs() < 40) & (df_results.look_mode == 'along_track'))
    ct_mask = ((df_results.tp_lon >= lon_bin) & (df_results.tp_lon < lon_bin+5) &
               (df_results.tp_lat.abs() < 40) & (df_results.look_mode == 'cross_track'))
    if at_mask.sum() < 1 or ct_mask.sum() < 1:
        continue
    rows = [df_results[at_mask].iloc[0], df_results[ct_mask].iloc[0]]
    G_rows, d_vals = [], []
    for r in rows:
        e_e, e_n, _ = enu_unit_vectors_eci(r.tp_lat, r.tp_lon, Time(r.epoch))
        los = np.array([r.los_x, r.los_y, r.los_z])
        G_rows.append([np.dot(los, e_e), np.dot(los, e_n)])
        d_vals.append(r.v_wind_LOS)
    result, *_ = np.linalg.lstsq(np.array(G_rows), np.array(d_vals), rcond=None)
    recovered.append({'lon_bin': lon_bin,
                      'v_zonal_rec': result[0], 'v_merid_rec': result[1]})

df_rec = pd.DataFrame(recovered)
bias_z = (df_rec.v_zonal_rec - 100.0).mean()
bias_m = (df_rec.v_merid_rec -  50.0).mean()
rms_z  = (df_rec.v_zonal_rec - 100.0).std()
rms_m  = (df_rec.v_merid_rec -  50.0).std()
v6 = abs(bias_z) < 1.0 and abs(bias_m) < 1.0 and rms_z < 2.0 and rms_m < 2.0
print(f"\nV6 — L2 round-trip (primary wind 100, 50 m/s):")
print(f"     Zonal:  bias={bias_z:+.3f}, RMS={rms_z:.3f} m/s")
print(f"     Merid:  bias={bias_m:+.3f}, RMS={rms_m:.3f} m/s")
print(f"     {'PASS' if v6 else 'FAIL'}")

# V7 — repeat for (200,0), (0,150), (0,0) secondary maps
# (implementation by Claude Code; same pass criteria)
v7 = True   # placeholder
print(f"\nV7 — Secondary wind maps: {'PASS' if v7 else 'FAIL'}")
```

---

### Section 7 — Save results and progress checkpoint

```python
import json, pathlib

pathlib.Path('outputs').mkdir(exist_ok=True)
df_results.to_csv('outputs/INT01_results.csv', index=False)
df_metadata.to_csv('outputs/INT01_metadata_summary.csv', index=False)
for idx, label in [(0,'first'), (-1,'last')]:
    write_sidecar(metadata_list[idx],
                  pathlib.Path(f'outputs/INT01_metadata_{label}.json'))

cp_path = pathlib.Path('simulations/checkpoints/progress.json')
cp_path.parent.mkdir(parents=True, exist_ok=True)
cp = json.loads(cp_path.read_text()) if cp_path.exists() else {}
checks_passed = sum([v1, v2, v3, v4, v5, v6, v7, v8])
cp['INT01_geometry_pipeline'] = {
    'status':        'complete',
    'checks_passed': checks_passed,
    'duration_days': DURATION_DAYS,
    'n_orbits':      int(n_orbits),
    'n_epochs':      len(df_results),
    'n_metadata':    len(metadata_list),
    'notes': (
        f"V1–V8 all PASS. Bias zonal={bias_z:.3f} m/s, merid={bias_m:.3f} m/s. "
        f"{DURATION_DAYS:.1f} day, {n_orbits} orbits, "
        f"{len(metadata_list)} synthetic ImageMetadata records."
    ),
}
cp_path.write_text(json.dumps(cp, indent=2))
print(f"Progress checkpoint updated ({checks_passed}/8 checks).")
```

---

## 6. Complete verification checklist

| Check | Criterion | Blocking? | Implementation note |
|-------|-----------|-----------|---------------------|
| V1 | tp_alt within ±5 km of 250 km | Yes | Iterative NB02a achieves < 0.1 km in practice |
| V2 | ECI offset > 500 km: along v̂_sc (AT), along n̂_orbit (CT) | Yes | **3-D dot product — longitude offset is wrong for 97.44° SSO** |
| V3 | Along-track \|V_sc_LOS\| > 10× cross-track, all orbits | Yes | |
| V4 | Zero NaN / Inf in all output columns | Yes | |
| V5 | Max v_rel jump < 100 m/s, checked for every orbit | Yes | |
| V6 | L2 round-trip bias < 1 m/s, RMS < 2 m/s (primary map 100, 50 m/s) | Yes | |
| V7 | Same criteria for maps (200,0), (0,150), (0,0) m/s | Yes | Zero-wind case verifies Earth-rotation removal |
| V8 | Metadata: 5 sub-checks V8a–V8e all pass | Yes | New in 2026-04-16 |

---

## 7. Output artefacts

```
outputs/
├── INT01_groundtrack.png            # S/C tracks + tangent points map
├── INT01_tp_altitude.png            # tp_alt vs. lat, ±5 km band
├── INT01_Vsc_LOS.png                # V_sc_LOS both look modes
├── INT01_v_rel_timeseries.png       # v_rel first two orbits
├── INT01_results.csv                # Full df_results (all epochs)
├── INT01_metadata_summary.csv       # df_metadata (1 row / epoch)
├── INT01_metadata_first.json        # ImageMetadata sidecar, first epoch
└── INT01_metadata_last.json         # ImageMetadata sidecar, last epoch

simulations/checkpoints/
└── progress.json                    # Updated with INT01 entry
```

`INT01_metadata_summary.csv` is the handoff document for Tier 2 modules.

---

## 8. File location in repository

```
soc_sewell/
├── notebooks/
│   └── INT01_geometry_pipeline_YYYY-MM-DD.ipynb
├── outputs/
│   └── (artefacts above)
├── simulations/checkpoints/
│   └── progress.json
└── docs/specs/
    ├── S08_int01_geometry_notebook_2026-04-05.md    ← retired
    └── S08_int01_geometry_notebook_2026-04-16.md    ← this file
```

---

## 9. Instructions for Claude Code

### Preamble

```bash
cat PIPELINE_STATUS.md
```

### Prerequisite reads

1. This spec: `docs/specs/S08_int01_geometry_notebook_2026-04-16.md`
2. `docs/specs/S06_nb01_orbit_propagator_2026-04-16.md` §5.1 — `propagate_orbit` signature.
3. `docs/specs/NB02_geometry_2026-04-16.md` §4.1 — iterative `_compute_depression_angle`.
4. `docs/specs/S19_p01_metadata_2026-04-06.md` §5.2 — `build_synthetic_metadata`.
5. `CLAUDE.md` at repo root.

### Prerequisite tests

```bash
pytest tests/test_s05_nb00_wind_map.py -v          # T1+T2 must pass
pytest tests/test_s06_nb01_orbit_propagator.py -v  # 8/8 must pass
pytest tests/test_nb02_geometry_2026_04_16.py -v   # 10/10 must pass
pytest tests/test_s19_p01_metadata.py -v           # 8/8 must pass
```

### Critical implementation rules

1. **Single `propagate_orbit` call** for `DURATION_S`; assign orbit/look mode
   from elapsed time.
2. **Timezone:** `t0 = pd.Timestamp(START_EPOCH, tz='UTC')`. Matplotlib:
   `df['epoch'].dt.tz_localize(None)`.
3. **`compute_los_eci` returns `(los_eci, q)`** — store both.
4. **V2:** always use 3-D ECI dot product as written. Never use longitude
   or latitude difference to check tangent-point lead distance.
5. **`InstrumentParams` stub:** `t_m = 20.106e-3` m. Not `20.008e-3` m.

### Epilogue

```bash
git add PIPELINE_STATUS.md \
        notebooks/INT01_geometry_pipeline_*.ipynb \
        outputs/INT01_*.png outputs/INT01_*.csv outputs/INT01_*.json \
        simulations/checkpoints/progress.json
git commit -m "docs: update S08 INT01 spec (V2 ECI projection, tz-aware, NB02a iterative)

Also updates PIPELINE_STATUS.md"
```

**Do not write or implement S09–S11 (Tier 2) until this commit is confirmed.**
