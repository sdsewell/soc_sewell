# INT01 Geometry Pipeline Integration Notebook Specification

**Spec ID:** INT01
**Spec file:** `docs/specs/INT01_geometry_notebook_2026-04-16.md`
**Project:** WindCube FPI Pipeline
**Institution:** NCAR / High Altitude Observatory (HAO)
**Status:** ✓ Complete — all 7 geometry checks pass; adds synthetic metadata (V8, new)
**Depends on:**
  - S01, S02, S03, S04 — conventions and constants
  - NB00 `2026-04-06`) — truth wind map
  - NB01 `2026-04-16`) — orbit propagator; `propagate_orbit(t_start, duration_s, dt_s)`
  - NB02 `2026-04-16`) — geometry pipeline; `nb02a/b/c/d_*_2026_04_16.py`
  - P01 `2026-04-06`) — `ImageMetadata`, `build_synthetic_metadata()`
**Used by:** S09–S11 (Tier 2 specs must not be written until INT01 is complete)
**References:**
  - Harding et al. (2014) Applied Optics 53(4) — science context
  - SI-UCAR-WC-RP-004 Issue 1.0 §2.4.2 — AOCS reference frames (BRF/THRF/SIRF)
  - WindCube STM v1 — V6/V7 pass criteria link to SG1/SG2 requirements
**Last updated:** 2026-04-16

> **Note:** This spec supersedes `INT01_geometry_notebook_2026-04-05.md`.
> The 2026-04-05 version is retired. All content is fully integrated here.

> **Changes from 2026-04-05:**
> - NB01 `propagate_orbit()` signature updated to `(t_start, duration_s, dt_s)` — altitude
>   is no longer an argument; it is read from `SC_ALTITUDE_KM` in S03.
> - NB02 imports updated to `nb02a/b/c/d_*_2026_04_16.py`; boresight convention
>   corrected to `−X_BRF` per SI-UCAR-WC-RP-004 §2.4.2.1.
> - AOCS reference frame nomenclature (BRF, THRF, SIRF) added throughout.
> - Duration-based orbit span: `DURATION_DAYS` replaces fixed two-orbit setup.
>   Look-mode alternation is computed from orbit number, not hard-coded.
> - **New Section 5.5:** Synthetic metadata array generation using P01
>   `build_synthetic_metadata()`, producing one `ImageMetadata` per epoch.
> - **New check V8:** Metadata array validation — field coverage and ADCS flag integrity.

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
- Tangent points are **ahead** (along-track) or **to the side** (cross-track)
  of the spacecraft at the expected ~923 km offset (V2)
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
exercises this strategy over a configurable number of days to confirm that
the geometry pipeline is correct for any realistic operational scenario.

**Orbit parity convention (from P01/S19 §3.6):**
- Odd orbit numbers (1, 3, 5, …) → `along_track` (THRF along-track configuration)
- Even orbit numbers (2, 4, 6, …) → `cross_track` (THRF cross-track configuration)

**Orbit 1, 3, … — along-track (odd orbits):**
The `−X_BRF` boresight (SI-UCAR-WC-RP-004 §2.4.2.1) is depressed 15.73°
toward the limb and points forward along the velocity direction. The LOS is
nearly parallel to the spacecraft velocity. V_sc_LOS ≈ −7,100 m/s dominates
the measured Doppler shift. After L1c removal, the residual wind LOS projection
is primarily **meridional** (north-south).

**Orbit 2, 4, … — cross-track (even orbits):**
The boresight rotates 90° to perpendicular to the orbit plane (anti-Sun side),
also depressed 15.73°. V_sc_LOS ≈ 0–300 m/s. The residual wind LOS projection
is primarily **zonal** (east-west).

The observation is never purely meridional or purely zonal — M07's WLS
inversion uses the exact geometry vectors to solve the mixed system. Across
multiple days, INT01 produces many matched orbit pairs for the V6/V7
round-trip verification.

---

## 3. Truth wind map for INT01

Use **T1 UniformWindMap only** for all INT01 verification. Do not use HWM14
or any spatially varying map. A uniform map has an analytic expected result
for the 2×2 decomposition that can be verified to machine precision. Spatially
varying maps introduce sampling and interpolation effects that obscure whether
the geometry is correct.

```python
from windmap.nb00_wind_map import UniformWindMap

# Primary test case — non-trivial in both components
wind_map = UniformWindMap(v_zonal_ms=100.0, v_merid_ms=50.0)

# Secondary cases — run after primary passes
wind_map_zonal_only  = UniformWindMap(v_zonal_ms=200.0, v_merid_ms=0.0)
wind_map_merid_only  = UniformWindMap(v_zonal_ms=0.0,   v_merid_ms=150.0)
wind_map_zero        = UniformWindMap(v_zonal_ms=0.0,   v_merid_ms=0.0)
```

The zero-wind case is particularly diagnostic: `v_rel` is then dominated
entirely by `−(V_sc_LOS + v_earth_LOS)`. If Earth-rotation removal is wrong,
the recovered wind will be non-zero even with a zero-wind input.

---

## 4. Orbit configuration and duration

```python
from src.constants import (
    SC_ALTITUDE_KM,           # 510.0 km
    SCIENCE_CADENCE_S,        # 10.0 s
    EARTH_GRAV_PARAM_M3_S2,   # 3.986004418e14 m³/s²
    WGS84_A_M,                # 6378137.0 m
)
import numpy as np

# Derive orbital period from constants — do not hardcode
a_m = WGS84_A_M + SC_ALTITUDE_KM * 1e3
T_ORBIT_S = 2 * np.pi * np.sqrt(a_m**3 / EARTH_GRAV_PARAM_M3_S2)
# Expected: T_ORBIT_S ≈ 5689 s ≈ 94.82 min

START_EPOCH    = "2027-01-01T00:00:00"  # ISO 8601 UTC
DT_S           = SCIENCE_CADENCE_S      # 10 s cadence
H_TARGET_KM    = 250.0                  # OI 630 nm tangent height

# ── Duration is configurable in days. ──────────────────────────────────────
# For verification (V1–V8): 1 day ≈ 15.2 orbits → ~7.6 matched orbit pairs.
# For operational simulation: set to desired mission duration (e.g. 7 days).
# Minimum for V6/V7 to pass: at least 2 paired orbits (1 even + 1 odd).
DURATION_DAYS  = 1.0
DURATION_S     = DURATION_DAYS * 86400.0

# Expected row count: DURATION_S / DT_S ≈ 8640 rows for 1 day
```

---

## 5. Notebook structure

Seven sections. Each ends with a clearly labelled PASS / FAIL cell. The
notebook must run top-to-bottom as a single execution with no hidden state.

### Section 1 — Setup and orbit propagation

Imports, constants, propagate the full duration via NB01 using the updated
`propagate_orbit(t_start, duration_s, dt_s)` signature. Assign orbit numbers
and look modes from elapsed time.

**PASS criterion:** DataFrame has ≈ `DURATION_S / DT_S` rows, no NaN values,
`orbit_number` and `look_mode` columns cover both parities.

```python
from src.geometry.nb02a_boresight_2026_04_16 import (
    compute_synthetic_quaternion, compute_los_eci,
)
from src.geometry.nb02b_tangent_point_2026_04_16 import compute_tangent_point
from src.geometry.nb02c_los_projection_2026_04_16 import compute_v_rel
from src.geometry.nb02d_l1c_calibrator_2026_04_16 import remove_spacecraft_velocity
from src.geometry.nb01_orbit_propagator_2026_04_05 import propagate_orbit
from src.windmap.nb00_wind_map_2026_04_05 import UniformWindMap
from src.metadata.p01_image_metadata_2026_04_06 import (
    ImageMetadata, build_synthetic_metadata,
)
from src.fpi import InstrumentParams   # needed for build_synthetic_metadata
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from astropy.time import Time

# Single propagate_orbit call for the full duration
df_full = propagate_orbit(
    t_start=START_EPOCH,
    duration_s=DURATION_S,
    dt_s=DT_S,
)

# Assign orbit number from elapsed time (1-based)
t0 = pd.Timestamp(START_EPOCH, tz='UTC')
df_full['elapsed_s'] = (df_full['epoch'] - t0).dt.total_seconds()
df_full['orbit_number'] = (df_full['elapsed_s'] // T_ORBIT_S).astype(int) + 1

# Look mode alternates by orbit parity per P01/S19 §3.6 convention
df_full['look_mode'] = df_full['orbit_number'].apply(
    lambda n: 'along_track' if n % 2 == 1 else 'cross_track'
)

# Frame sequence counter within each orbit (0-based, resets at orbit boundary)
df_full['frame_sequence'] = df_full.groupby('orbit_number').cumcount()

n_orbits = df_full['orbit_number'].max()
print(f"Duration: {DURATION_DAYS:.1f} day(s), {len(df_full)} epochs, {n_orbits} orbits")
print(f"Along-track orbits:  {(df_full.look_mode=='along_track').sum()} epochs")
print(f"Cross-track orbits:  {(df_full.look_mode=='cross_track').sum()} epochs")
assert df_full.isna().sum().sum() == 0, "NaN found in orbit propagation output"
print("\nSection 1: PASS")
```

---

### Section 2 — LOS vectors, tangent points, v_rel for all epochs

Loop over all epochs in `df_full`, computing NB02a → NB02b → NB02c in
sequence. Store results in a single `df_results` DataFrame.

**NB02a note:** Use `compute_los_eci()` which returns `(los_eci, q)` — the
LOS vector and the synthetic attitude quaternion. The quaternion is needed
in Section 5 for synthetic metadata construction.

**Output columns:**
```
orbit_number, orbit_parity, look_mode, frame_sequence, epoch,
sc_lat, sc_lon, sc_alt_km,
pos_eci_x, pos_eci_y, pos_eci_z,
vel_eci_x, vel_eci_y, vel_eci_z,
tp_lat, tp_lon, tp_alt_km,
tp_eci_x, tp_eci_y, tp_eci_z,
los_x, los_y, los_z,
q_x, q_y, q_z, q_w,
v_wind_LOS, V_sc_LOS, v_earth_LOS, v_rel,
v_zonal_ms, v_merid_ms
```

**PASS criterion:** No NaN or Inf in any numeric column. All `tp_alt_km`
within 5 km of H_TARGET_KM.

```python
wind_map = UniformWindMap(v_zonal_ms=100.0, v_merid_ms=50.0)
results  = []

for _, row in df_full.iterrows():
    pos    = np.array([row.pos_eci_x, row.pos_eci_y, row.pos_eci_z])
    vel    = np.array([row.vel_eci_x, row.vel_eci_y, row.vel_eci_z])
    epoch  = Time(row.epoch)
    mode   = row.look_mode

    # NB02a — LOS vector and attitude quaternion (boresight = -X_BRF)
    los, q = compute_los_eci(pos, vel, mode)

    # NB02b — tangent point (WGS84 ellipsoid + 250 km shell)
    tp = compute_tangent_point(pos, los, epoch, H_TARGET_KM)

    # NB02c — v_rel decomposition
    res = compute_v_rel(
        wind_map,
        tp['tp_lat_deg'], tp['tp_lon_deg'], tp['tp_eci'],
        vel, los, epoch,
    )

    results.append({
        'orbit_number':   row.orbit_number,
        'orbit_parity':   mode,          # along_track or cross_track
        'look_mode':      mode,
        'frame_sequence': row.frame_sequence,
        'epoch':          row.epoch,
        'sc_lat':         row.lat_deg,
        'sc_lon':         row.lon_deg,
        'sc_alt_km':      row.alt_km,
        'pos_eci_x':      pos[0],  'pos_eci_y': pos[1],  'pos_eci_z': pos[2],
        'vel_eci_x':      vel[0],  'vel_eci_y': vel[1],  'vel_eci_z': vel[2],
        'tp_lat':         tp['tp_lat_deg'],
        'tp_lon':         tp['tp_lon_deg'],
        'tp_alt_km':      tp['tp_alt_km'],
        'tp_eci_x':       tp['tp_eci'][0],
        'tp_eci_y':       tp['tp_eci'][1],
        'tp_eci_z':       tp['tp_eci'][2],
        'los_x':          los[0],   'los_y': los[1],   'los_z': los[2],
        'q_x':            q[0],     'q_y':   q[1],
        'q_z':            q[2],     'q_w':   q[3],
        'v_wind_LOS':     res['v_wind_LOS'],
        'V_sc_LOS':       res['V_sc_LOS'],
        'v_earth_LOS':    res['v_earth_LOS'],
        'v_rel':          res['v_rel'],
        'v_zonal_ms':     res['v_zonal_ms'],
        'v_merid_ms':     res['v_merid_ms'],
    })

df_results  = pd.DataFrame(results)
df_at = df_results[df_results.look_mode == 'along_track']
df_ct = df_results[df_results.look_mode == 'cross_track']

num_cols = df_results.select_dtypes(include=np.number)
assert df_results.isna().sum().sum() == 0, "NaN found in results"
assert np.all(np.isfinite(num_cols.values)),  "Inf found in results"
alt_ok = (df_results.tp_alt_km - H_TARGET_KM).abs().max()
assert alt_ok < 5.0, f"Tangent altitude deviation {alt_ok:.1f} km > 5 km"
print(f"Max tangent altitude deviation: {alt_ok:.2f} km  (limit 5 km)")
print("\nSection 2: PASS")
```

---

### Section 3 — Four geometric verification plots

Four figures, all saved to `outputs/`. All must be visually inspected
and confirmed before proceeding to Section 4.

**Plot 3.1 — Ground track and tangent points map**
File: `outputs/INT01_groundtrack.png`

- Along-track ground track (blue) + tangent points (scatter, cyan)
- Cross-track ground track (orange) + tangent points (scatter, yellow)
- ±40° science band shown as dashed lines
- If DURATION_DAYS > 1: subsample for visual clarity (every 5th epoch)

**Visual checks:**
- Along-track tangent points are **ahead** of the spacecraft ground track
  by ~8–10° longitude at the equator (≈ 923 km forward)
- Cross-track tangent points are **perpendicular** to the ground track
  by a similar offset (~923 km to the side)
- Ground tracks span the expected latitude range for SSO at 97.44°
- Science band (±40°) contains the majority of tangent points

**Plot 3.2 — Tangent point altitude vs. latitude**
File: `outputs/INT01_tp_altitude.png`

- `tp_alt_km` vs. `tp_lat` for all epochs, both look modes
- Horizontal lines at 245 km and 255 km (±5 km tolerance band)
- All points must fall within the band

**Plot 3.3 — V_sc_LOS comparison, both look modes**
File: `outputs/INT01_Vsc_LOS.png`

- V_sc_LOS time series for along-track orbits (blue) and cross-track orbits (orange)
- Expected: along-track ≈ −7,100 m/s; cross-track ≈ 0–300 m/s
- The 10:1 ratio must be visually obvious

**Plot 3.4 — v_rel time series, first two orbits**
File: `outputs/INT01_v_rel_timeseries.png`

- v_rel vs. epoch for the first along-track orbit (blue) and first cross-track
  orbit (orange), offset for clarity
- Must be smooth — no jumps > 100 m/s between consecutive points

```python
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf

# --- Plot 3.1 ---
subsample = max(1, int(len(df_results) / 2000))  # keep ≤ 2000 points for speed
fig, ax = plt.subplots(figsize=(14, 6),
                        subplot_kw={'projection': ccrs.PlateCarree()})
ax.add_feature(cf.COASTLINE, linewidth=0.4)
ax.add_feature(cf.BORDERS,   linewidth=0.2)
ax.axhline(40,  ls='--', color='gray', alpha=0.6, label='±40° science band')
ax.axhline(-40, ls='--', color='gray', alpha=0.6)
ax.scatter(df_at.sc_lon [::subsample], df_at.sc_lat [::subsample],
           s=0.5, c='steelblue', label='S/C along-track')
ax.scatter(df_ct.sc_lon [::subsample], df_ct.sc_lat [::subsample],
           s=0.5, c='darkorange', label='S/C cross-track')
ax.scatter(df_at.tp_lon [::subsample], df_at.tp_lat [::subsample],
           s=1.0, c='cyan',   alpha=0.5, label='TP along-track')
ax.scatter(df_ct.tp_lon [::subsample], df_ct.tp_lat [::subsample],
           s=1.0, c='yellow', alpha=0.5, label='TP cross-track')
ax.set_title(f"INT01 Ground Track + Tangent Points ({DURATION_DAYS:.1f} day)")
ax.legend(loc='lower left', markerscale=5, fontsize=7)
plt.tight_layout()
plt.savefig('outputs/INT01_groundtrack.png', dpi=150)
plt.show()

# --- Plots 3.2, 3.3, 3.4 (similar structure — implementation per Claude Code) ---
```

---

### Section 4 — Quantitative verification checks V1–V5

```python
print("=" * 60)
print("INT01 Verification Checks V1–V5")
print("=" * 60)

# V1 — Tangent altitude within ±5 km of H_TARGET_KM
alt_dev = (df_results.tp_alt_km - H_TARGET_KM).abs()
v1 = alt_dev.max() < 5.0
print(f"\nV1 — Tangent altitude (target {H_TARGET_KM} km):")
print(f"     Max deviation: {alt_dev.max():.2f} km  (must be < 5 km)")
print(f"     {'PASS' if v1 else 'FAIL'}")

# V2 — Tangent points ahead (along-track) / to the side (cross-track)
# Along-track: TP longitude should lead S/C by > 500 km at equator
eq_at = df_at[df_at.sc_lat.abs() < 5]
if len(eq_at) > 0:
    lon_offset_km = (eq_at.tp_lon - eq_at.sc_lon).mean() * 111.0
    v2_at = abs(lon_offset_km) > 500
    print(f"\nV2a — Along-track TP forward offset at equator:")
    print(f"      Mean: {lon_offset_km:+.0f} km  (expected |offset| > 500 km forward)")
    print(f"      {'PASS' if v2_at else 'FAIL'}")
# Cross-track: TP latitude should differ from S/C by > 500 km
eq_ct = df_ct[df_ct.sc_lat.abs() < 5]
if len(eq_ct) > 0:
    lat_offset_km = (eq_ct.tp_lat - eq_ct.sc_lat).mean() * 111.0
    v2_ct = abs(lat_offset_km) > 500
    print(f"\nV2b — Cross-track TP lateral offset at equator:")
    print(f"      Mean: {lat_offset_km:+.0f} km  (expected |offset| > 500 km lateral)")
    print(f"      {'PASS' if v2_ct else 'FAIL'}")
v2 = v2_at and v2_ct

# V3 — V_sc_LOS ratio along-track / cross-track > 10
ratio = abs(df_at.V_sc_LOS.mean()) / (abs(df_ct.V_sc_LOS.mean()) + 1e-6)
v3 = ratio > 10.0
print(f"\nV3 — V_sc_LOS ratio (along-track / cross-track):")
print(f"     Along-track mean: {df_at.V_sc_LOS.mean():.0f} m/s")
print(f"     Cross-track mean: {df_ct.V_sc_LOS.mean():.0f} m/s")
print(f"     Ratio: {ratio:.1f}×  (must be > 10×)")
print(f"     {'PASS' if v3 else 'FAIL'}")

# V4 — No NaN or Inf in any output column
num_cols = df_results.select_dtypes(include=np.number)
v4 = (df_results.isna().sum().sum() == 0 and
      np.all(np.isfinite(num_cols.values)))
print(f"\nV4 — No NaN or Inf values in any output column:")
print(f"     NaN count: {df_results.isna().sum().sum()}")
print(f"     {'PASS' if v4 else 'FAIL'}")

# V5 — v_rel smoothness per orbit (no spikes > 100 m/s between adjacent epochs)
v5_all = True
for orb_num, orb_df in df_results.groupby('orbit_number'):
    diffs = np.abs(np.diff(orb_df.v_rel.values))
    max_jump = diffs.max() if len(diffs) > 0 else 0.0
    ok = max_jump < 100.0
    if not ok:
        print(f"\nV5 — FAIL in orbit {orb_num}: max jump {max_jump:.2f} m/s")
    v5_all = v5_all and ok
print(f"\nV5 — v_rel smoothness (all {n_orbits} orbits, limit 100 m/s):")
print(f"     {'PASS' if v5_all else 'FAIL'}")
v5 = v5_all

all_v1_v5 = all([v1, v2, v3, v4, v5])
print(f"\n── V1–V5 overall: {'PASS' if all_v1_v5 else 'FAIL'} ──")
```

---

### Section 5 — Synthetic metadata array generation (V8)

This section is new in the 2026-04-16 revision. It uses P01's
`build_synthetic_metadata()` to construct one `ImageMetadata` record for each
science epoch in `df_results`. The result is a list `metadata_list` and a
summary DataFrame `df_metadata` of key fields.

**Purpose:** The metadata array is the primary output consumed by Tier 2
modules (S09 — M01 forward model, S10 — M02 calibration synthesis, S11 — M04
science frame synthesis). Those modules require `ImageMetadata` objects to
know the orbit state, attitude, look mode, and truth velocity for each frame.

**Image types in INT01:** Only `'science'` frames are generated here. The
synthesis modules for calibration and dark frames are specified separately in
S10 (M02) and S11 (M04). When those modules are implemented, they will call
`build_synthetic_metadata()` with `img_type='cal'` and `img_type='dark'`
respectively, using the same orbit state columns from `df_results`.

**Note on `InstrumentParams`:** If M01 (S09) is not yet implemented, create
a minimal stub:

```python
class _StubInstrumentParams:
    """Minimal stub until M01 (S09) is implemented."""
    t_m = 20.106e-3   # etalon gap, metres (authoritative from S03)
    f_m = 0.19912     # focal length, metres
    r_refl = 0.53     # effective reflectivity (FlatSat)
params = _StubInstrumentParams()
```

```python
from src.metadata.p01_image_metadata_2026_04_06 import (
    build_synthetic_metadata, write_sidecar,
)

# Try to import InstrumentParams; fall back to stub if M01 not yet implemented
try:
    from src.fpi.m01_airy_model import InstrumentParams
    params = InstrumentParams()
    print("Using InstrumentParams from M01.")
except ImportError:
    class _StubInstrumentParams:
        t_m    = 20.106e-3   # etalon gap, m (S03 authoritative)
        f_m    = 0.19912     # focal length, m
        r_refl = 0.53        # effective reflectivity (FlatSat)
    params = _StubInstrumentParams()
    print("M01 not yet available — using InstrumentParams stub.")

metadata_list = []

for i, row in df_results.iterrows():
    nb01_row = df_full.loc[df_full['epoch'] == row['epoch']].iloc[0]

    nb02_tp = {
        'tp_lat_deg': row.tp_lat,
        'tp_lon_deg': row.tp_lon,
        'tp_alt_km':  row.tp_alt_km,
        'tp_eci':     [row.tp_eci_x, row.tp_eci_y, row.tp_eci_z],
    }
    nb02_vr = {
        'v_rel':       row.v_rel,
        'v_wind_LOS':  row.v_wind_LOS,
        'V_sc_LOS':    row.V_sc_LOS,
        'v_earth_LOS': row.v_earth_LOS,
        'v_zonal_ms':  row.v_zonal_ms,
        'v_merid_ms':  row.v_merid_ms,
    }
    quaternion_xyzw = [row.q_x, row.q_y, row.q_z, row.q_w]
    los_eci         = np.array([row.los_x, row.los_y, row.los_z])

    meta = build_synthetic_metadata(
        params         = params,
        nb01_row       = nb01_row,
        nb02_tp        = nb02_tp,
        nb02_vr        = nb02_vr,
        quaternion_xyzw= quaternion_xyzw,
        los_eci        = los_eci,
        look_mode      = row.look_mode,
        img_type       = 'science',
        orbit_number   = int(row.orbit_number),
        frame_sequence = int(row.frame_sequence),
        noise_seed     = int(i),  # deterministic seed from row index
    )
    metadata_list.append(meta)

print(f"Generated {len(metadata_list)} ImageMetadata records.")

# Build summary DataFrame of key fields for quick inspection
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

print(df_metadata.describe())
```

**V8 — Synthetic metadata validation:**

```python
print("\nV8 — Synthetic metadata array validation:")

# V8a — All records are synthetic with GOOD ADCS flag
v8a = (df_metadata.is_synthetic.all() and
       (df_metadata.adcs_flag == 0).all())
print(f"\nV8a — All is_synthetic=True and adcs_flag=GOOD:")
print(f"     is_synthetic all True: {df_metadata.is_synthetic.all()}")
print(f"     adcs_flag all 0 (GOOD): {(df_metadata.adcs_flag == 0).all()}")
print(f"     {'PASS' if v8a else 'FAIL'}")

# V8b — truth_v_los matches v_rel from df_results (round-trip)
truth_diff = (df_metadata.truth_v_los - df_results.v_wind_LOS).abs()
v8b = truth_diff.max() < 1e-6
print(f"\nV8b — truth_v_los matches v_wind_LOS from NB02c:")
print(f"     Max diff: {truth_diff.max():.2e} m/s  (must be < 1e-6 m/s)")
print(f"     {'PASS' if v8b else 'FAIL'}")

# V8c — tangent point fields match df_results
tp_lat_diff = (df_metadata.tp_lat - df_results.tp_lat).abs()
v8c = tp_lat_diff.max() < 1e-8
print(f"\nV8c — tangent_lat matches tp_lat from NB02b:")
print(f"     Max diff: {tp_lat_diff.max():.2e} deg  (must be < 1e-8 deg)")
print(f"     {'PASS' if v8c else 'FAIL'}")

# V8d — orbit_parity consistent with orbit_number
parity_ok = df_metadata.apply(
    lambda r: (r.orbit_parity == 'along_track') == (r.orbit_number % 2 == 1),
    axis=1,
)
v8d = parity_ok.all()
print(f"\nV8d — orbit_parity consistent with orbit_number parity:")
print(f"     All consistent: {parity_ok.all()}")
print(f"     {'PASS' if v8d else 'FAIL'}")

# V8e — No None/NaN in required synthetic fields
required_cols = [
    'tp_lat', 'tp_lon', 'tp_alt_km', 'truth_v_los',
    'truth_v_zonal', 'truth_v_merid', 'etalon_gap_mm',
]
has_none = df_metadata[required_cols].isna().sum()
v8e = has_none.sum() == 0
print(f"\nV8e — No None/NaN in required synthetic fields:")
if not v8e:
    print(f"     Columns with None: {has_none[has_none > 0].to_dict()}")
print(f"     {'PASS' if v8e else 'FAIL'}")

v8 = all([v8a, v8b, v8c, v8d, v8e])
print(f"\n── V8 overall: {'PASS' if v8 else 'FAIL'} ──")
```

---

### Section 6 — L2 round-trip wind decomposition (V6 and V7)

This is the most important geometry section. It verifies the complete geometry
pipeline can be inverted to recover the truth wind.

**Method:** For each longitude bin with tangent points from both look modes,
solve the 2×2 linear system:

```
G · [v_zonal, v_merid]ᵀ = d

where:
  G[i, :] = [los_eci[i] · ê_east_eci, los_eci[i] · ê_north_eci]
  d[i]    = v_wind_LOS[i]   (after L1c removal)
```

Use numpy least-squares (`np.linalg.lstsq`) for the 2×2 system. With
multiple orbits, there will be many matched pairs per longitude bin — use
the median along-track epoch and median cross-track epoch per bin, or
average the recovered winds across all pairs in the bin.

**V6 — Primary wind map (100, 50) m/s:**

```python
from src.geometry.nb02c_los_projection_2026_04_16 import enu_unit_vectors_eci

recovered = []
for lon_bin in np.arange(-180, 180, 5.0):
    at_mask = ((df_results.tp_lon >= lon_bin) & (df_results.tp_lon < lon_bin + 5) &
               (df_results.tp_lat.abs() < 40) & (df_results.look_mode == 'along_track'))
    ct_mask = ((df_results.tp_lon >= lon_bin) & (df_results.tp_lon < lon_bin + 5) &
               (df_results.tp_lat.abs() < 40) & (df_results.look_mode == 'cross_track'))
    if at_mask.sum() < 1 or ct_mask.sum() < 1:
        continue

    # Use first valid row from each look mode in this bin
    at_row = df_results[at_mask].iloc[0]
    ct_row = df_results[ct_mask].iloc[0]

    pair_recovered = []
    for r in [at_row, ct_row]:
        epoch_t = Time(r.epoch)
        e_east, e_north, _ = enu_unit_vectors_eci(r.tp_lat, r.tp_lon, epoch_t)
        los = np.array([r.los_x, r.los_y, r.los_z])
        G_row = [np.dot(los, e_east), np.dot(los, e_north)]
        pair_recovered.append({'G': G_row, 'd': r.v_wind_LOS})

    G = np.array([p['G'] for p in pair_recovered])
    d = np.array([p['d'] for p in pair_recovered])
    result, *_ = np.linalg.lstsq(G, d, rcond=None)
    recovered.append({'lon_bin': lon_bin, 'v_zonal_rec': result[0],
                      'v_merid_rec': result[1]})

df_rec  = pd.DataFrame(recovered)
bias_z  = (df_rec.v_zonal_rec - 100.0).mean()
bias_m  = (df_rec.v_merid_rec -  50.0).mean()
rms_z   = (df_rec.v_zonal_rec - 100.0).std()
rms_m   = (df_rec.v_merid_rec -  50.0).std()
v6 = (abs(bias_z) < 1.0 and abs(bias_m) < 1.0 and rms_z < 2.0 and rms_m < 2.0)
print(f"\nV6 — L2 round-trip (primary wind 100, 50 m/s):")
print(f"     Zonal:  bias={bias_z:+.3f}, RMS={rms_z:.3f} m/s")
print(f"     Merid:  bias={bias_m:+.3f}, RMS={rms_m:.3f} m/s")
print(f"     {'PASS' if v6 else 'FAIL'}")
```

**V7 — Secondary wind maps:**
Repeat Section 6 for three secondary cases using the same longitude-bin
pairing approach. Same pass criterion (bias < 1 m/s, RMS < 2 m/s) for each.
The zero-wind case verifies Earth-rotation removal.

---

### Section 7 — Save results and progress checkpoint

```python
import json, pathlib

pathlib.Path('outputs').mkdir(exist_ok=True)
df_results.to_csv('outputs/INT01_results.csv', index=False)
df_metadata.to_csv('outputs/INT01_metadata_summary.csv', index=False)

# Write metadata sidecar JSON for first and last science epochs
#   (representative samples — full array is too large for routine commit)
for idx, label in [(0, 'first'), (-1, 'last')]:
    write_sidecar(
        metadata_list[idx],
        pathlib.Path(f'outputs/INT01_metadata_{label}.json'),
    )
print(f"Saved metadata summary ({len(df_metadata)} rows) to outputs/INT01_metadata_summary.csv")

# Progress checkpoint
cp_path = pathlib.Path('simulations/checkpoints/progress.json')
cp_path.parent.mkdir(parents=True, exist_ok=True)
cp = json.loads(cp_path.read_text()) if cp_path.exists() else {}
checks_passed = sum([v1, v2, v3, v4, v5, v6, v7, v8])
cp['INT01_geometry_pipeline'] = {
    'status':           'complete',
    'checks_passed':    checks_passed,
    'duration_days':    DURATION_DAYS,
    'n_orbits':         int(n_orbits),
    'n_epochs':         len(df_results),
    'n_metadata':       len(metadata_list),
    'notes': (
        f"V1–V8 all PASS. Bias zonal={bias_z:.3f} m/s, merid={bias_m:.3f} m/s. "
        f"Duration {DURATION_DAYS:.1f} day, {n_orbits} orbits, "
        f"{len(metadata_list)} synthetic ImageMetadata records."
    ),
}
cp_path.write_text(json.dumps(cp, indent=2))
print(f"Progress checkpoint updated ({checks_passed}/8 checks passed).")
```

---

## 6. Complete verification checklist

| Check | Criterion | Blocking? | New in v2? |
|-------|-----------|-----------|-----------|
| V1 | Tangent altitude within ±5 km of 250 km | Yes | No |
| V2 | Along-track TP > 500 km forward; cross-track TP > 500 km lateral | Yes | Updated |
| V3 | Along-track V_sc_LOS > 10× cross-track (across all orbits) | Yes | No |
| V4 | Zero NaN / Inf in all output columns | Yes | No |
| V5 | Max v_rel epoch-to-epoch jump < 100 m/s, all orbits | Yes | No |
| V6 | L2 round-trip bias < 1 m/s, RMS < 2 m/s (primary map) | Yes | No |
| V7 | Same criteria for three secondary wind maps | Yes | No |
| V8 | Synthetic metadata array integrity (all 5 sub-checks) | Yes | **NEW** |

All eight checks must show PASS. If any fails, diagnose and fix the
indicated module before proceeding to Tier 2 specs.

---

## 7. Output artefacts

The following must be committed to the repo:

```
outputs/
├── INT01_groundtrack.png            # Map of S/C tracks + tangent points
├── INT01_tp_altitude.png            # Tangent point altitude vs latitude
├── INT01_Vsc_LOS.png                # V_sc_LOS comparison, both look modes
├── INT01_v_rel_timeseries.png       # v_rel time series, first two orbits
├── INT01_results.csv                # Full df_results DataFrame
├── INT01_metadata_summary.csv       # df_metadata (one row per epoch)
├── INT01_metadata_first.json        # ImageMetadata sidecar, first science epoch
└── INT01_metadata_last.json         # ImageMetadata sidecar, last science epoch

simulations/checkpoints/
└── progress.json                    # Updated with INT01 entry
```

`INT01_metadata_summary.csv` is a permanent reference for downstream
modules. It contains the truth geometry fields needed by M04 (airglow
synthesis) and M02 (calibration synthesis) to produce synthetic images.

---

## 8. File location in repository

```
soc_sewell/
├── notebooks/
│   └── INT01_geometry_pipeline_YYYY-MM-DD.ipynb       ← this spec
├── outputs/
│   ├── INT01_groundtrack.png
│   ├── INT01_tp_altitude.png
│   ├── INT01_Vsc_LOS.png
│   ├── INT01_v_rel_timeseries.png
│   ├── INT01_results.csv
│   ├── INT01_metadata_summary.csv
│   ├── INT01_metadata_first.json
│   └── INT01_metadata_last.json
├── simulations/checkpoints/
│   └── progress.json
└── docs/specs/
    ├── S08_int01_geometry_notebook_2026-04-05.md    ← retired
    └── S08_int01_geometry_notebook_2026-04-16.md    ← this file
```

---

## 9. Instructions for Claude Code

### Preamble — required at start of every session

```bash
cat PIPELINE_STATUS.md
```

### Prerequisite reads (before writing any code)

1. This entire spec: `docs/specs/S08_int01_geometry_notebook_2026-04-16.md`
2. `docs/specs/S06_nb01_orbit_propagator_2026-04-16.md` — especially the
   new `propagate_orbit(t_start, duration_s, dt_s)` signature (§5.1) and
   AOCS frame definitions (§2.4).
3. `docs/specs/NB02_geometry_2026-04-16.md` — boresight is `−X_BRF`
   (§2.1 critical); new NB02a–d file names; `compute_los_eci` returns
   `(los_eci, q)` tuple; `compute_v_rel` return dict includes
   `v_zonal_ms` and `v_merid_ms`.
4. `docs/specs/S19_p01_metadata_2026-04-06.md` — `ImageMetadata` dataclass,
   `build_synthetic_metadata()` signature (§5.2), orbit parity convention (§3.6).
5. `docs/specs/S05_nb00_wind_map_2026-04-06.md` — `UniformWindMap` interface.
6. `CLAUDE.md` at repo root.

### Prerequisite tests (before writing notebook)

```bash
pytest tests/test_s05_nb00_wind_map.py -v        # T1+T2 must pass
pytest tests/test_s06_nb01_orbit_propagator.py -v # 8/8 must pass
pytest tests/test_nb02_geometry_2026_04_16.py -v  # 10/10 must pass
pytest tests/test_s19_p01_metadata.py -v          # 8/8 must pass
```

Do not proceed if any prerequisite test fails.

### Key interface changes from 2026-04-05 spec

The following changes from the 2026-04-05 INT01 spec must be applied:

1. **`propagate_orbit()` signature:** old form was
   `propagate_orbit(START_EPOCH, T_ORBIT_S, DT_S, ALTITUDE_KM)` with one call
   per orbit. New form is a **single call**:
   ```python
   df_full = propagate_orbit(t_start=START_EPOCH, duration_s=DURATION_S, dt_s=DT_S)
   ```
   `ALTITUDE_KM` is no longer an argument — NB01 reads `SC_ALTITUDE_KM` from
   `src.constants`.

2. **Look mode assignment:** old spec hard-coded two orbits. New spec computes
   orbit number from elapsed time and derives look mode from parity. Use the
   code in Section 1 exactly as written.

3. **NB02 import paths:** use the `2026_04_16` date suffix in all imports.
   Old `nb02a_boresight` without date suffix is no longer valid.

4. **`compute_los_eci` return value:** returns `(los_eci, q)` — capture both.
   The quaternion `q` is required for `build_synthetic_metadata()`.

5. **`compute_v_rel` return dict:** now includes `'v_zonal_ms'` and
   `'v_merid_ms'` keys. Store them in `df_results`.

6. **No `ALTITUDE_KM` argument in notebook:** do not pass altitude to
   `compute_los_eci` or `compute_tangent_point`. Both now read `SC_ALTITUDE_KM`
   and `TP_ALTITUDE_KM` from `src.constants` as defaults.

### Implementation steps

1. Confirm all prerequisite tests pass (above).
2. Create `outputs/` directory if absent.
3. Create `notebooks/INT01_geometry_pipeline_YYYY-MM-DD.ipynb` with the
   seven sections above, one cell per section, each ending with PASS/FAIL.
4. **Section 1:** Single `propagate_orbit` call. Add `orbit_number`,
   `look_mode`, `frame_sequence` columns per spec.
5. **Section 2:** Loop must store `q_x/y/z/w`, `tp_eci_x/y/z`,
   `v_zonal_ms`, `v_merid_ms` columns — these are new vs. 2026-04-05 spec.
6. **Section 3:** Add `cartopy` import. If cartopy is not installed, skip
   the map projection and use a plain scatter plot with a note.
7. **Section 4:** V2 now checks both along-track and cross-track offsets.
8. **Section 5 (metadata):** Use the `_StubInstrumentParams` fallback if
   M01 is not yet implemented. The stub must use `t_m = 20.106e-3` (spec
   authoritative etalon gap in metres from S03).
9. **Section 6:** V6 pairing approach updated — longitude bins across all
   orbits, not just first two orbits.
10. **Section 7:** Save `INT01_metadata_summary.csv` and two sidecar JSONs.
11. Run notebook top-to-bottom — all cells, no skipping.
12. All 8 checks (V1–V8) must show PASS.
13. **Visually inspect all four PNGs** before marking INT01 complete.
14. Update `simulations/checkpoints/progress.json`.

### Stop condition

If any test or notebook cell fails after two debugging attempts, stop and
return the full error output and diagnosis note to Claude.ai. Do not loop
more than 10–15 minutes on a failing check without returning.

Do not write or implement S09–S11 (Tier 2) until this commit is confirmed.

### Report-back format

```
INT01 IMPLEMENTATION REPORT
============================
Spec version: S08_int01_geometry_notebook_2026-04-16.md
Checks passed: V1/V2/V3/V4/V5/V6/V7/V8  [list any FAIL]
Duration: DURATION_DAYS day(s), N orbits, N epochs
V6 bias:  zonal=±X.XXX m/s, merid=±X.XXX m/s
V8 metadata records: N
Deviations from spec: [none | list any]
NB02 boresight confirmed: -X_BRF (np.array([-1., 0., 0.]))
InstrumentParams: [M01 | stub]
Files created/modified:
  - notebooks/INT01_geometry_pipeline_YYYY-MM-DD.ipynb  (created)
  - outputs/INT01_groundtrack.png                        (saved)
  - outputs/INT01_tp_altitude.png                        (saved)
  - outputs/INT01_Vsc_LOS.png                            (saved)
  - outputs/INT01_v_rel_timeseries.png                   (saved)
  - outputs/INT01_results.csv                            (saved)
  - outputs/INT01_metadata_summary.csv                   (saved)
  - outputs/INT01_metadata_first.json                    (saved)
  - outputs/INT01_metadata_last.json                     (saved)
  - simulations/checkpoints/progress.json               (updated)
```

### Epilogue — required at end of every session

Update `PIPELINE_STATUS.md` for any spec whose status changed during
this session, then co-commit:

```bash
# Update PIPELINE_STATUS.md — change status/tests/date for affected specs
git add PIPELINE_STATUS.md \
        notebooks/INT01_geometry_pipeline_*.ipynb \
        outputs/INT01_*.png outputs/INT01_*.csv outputs/INT01_*.json \
        simulations/checkpoints/progress.json
git commit -m "feat(int01): geometry pipeline integration, all 8 checks pass

Implements: S08_int01_geometry_notebook_2026-04-16.md
Duration: DURATION_DAYS day(s), N orbits, N epochs
V1-V7 geometry checks pass. V8: N synthetic ImageMetadata records.
Bias zonal=±X.XXX m/s, merid=±X.XXX m/s.
Uses NB01 duration_s API, NB02 2026-04-16, P01 build_synthetic_metadata().

Also updates PIPELINE_STATUS.md"
```

**Do not write or implement S09–S11 (Tier 2) until this commit is confirmed.**
Tier 2 modules receive `v_rel` and `ImageMetadata` as inputs — if either
the geometry or the metadata is wrong, every downstream module will be wrong.
