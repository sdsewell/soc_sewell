# S08 — INT01 Geometry Pipeline Integration Notebook Specification

**Spec ID:** S08
**Spec file:** `docs/specs/S08_int01_geometry_notebook_2026-04-05.md`
**Project:** WindCube FPI Pipeline
**Institution:** NCAR / High Altitude Observatory (HAO)
**Status:** ✓ Complete — all 7 checks pass (re-implement from this spec for clean repo)
**Depends on:** S01, S02, S03, S04, S05 (NB00 T1+T2), S06 (NB01 8/8), S07 (NB02 10/10)
**Used by:** S09–S11 (Tier 2 specs must not be written until INT01 is complete)
**References:**
  - Harding et al. (2014) Applied Optics 53(4) — science context
  - WindCube STM v1 — V6/V7 pass criteria link to SG1/SG2 requirements
**Last updated:** 2026-04-05

> **Note:** This spec supersedes `INT01_geometry_pipeline_spec.md`.
> That file is retired. All content is fully integrated here.

---

## 1. Purpose and philosophy

INT01 is the first integration notebook in the WindCube pipeline. It connects
NB00 (truth wind map), NB01 (orbit propagator), and NB02 (geometry/LOS) over
two complete orbits with alternating look modes, producing visual and
quantitative evidence that the geometry pipeline works end-to-end.

**The INT series philosophy:**
Each INT notebook validates a complete pipeline stage boundary — not just
individual modules in isolation. INT01 asks: when NB00 + NB01 + NB02 are
connected on realistic inputs, does the geometry produce physically correct
`v_rel` values that a downstream FPI inversion could use? Integration
notebooks are permanent project artefacts, committed to the repo and re-run
whenever upstream modules change.

**What INT01 specifically validates:**
- Tangent points are at the correct altitude (250 ± 5 km, V1)
- Tangent points are **ahead** of the spacecraft (ram-face geometry, V2)
- Along-track V_sc_LOS dominates cross-track by > 10× (V3)
- No NaN or Inf values anywhere in the pipeline output (V4)
- v_rel time series is smooth — no epoch-to-epoch spikes > 100 m/s (V5)
- L2 2×2 wind decomposition recovers truth wind to < 1 m/s bias (V6)
- Round-trip holds for three secondary wind maps (V7)

**Gate rule:** Tier 2 specs (S09–S11) must not be written or implemented
until all 7 INT01 checks pass and the four output PNGs have been visually
inspected and confirmed.

---

## 2. Science context — two orbits, alternating look modes

WindCube alternates between along-track and cross-track boresight directions
on successive orbits. This two-orbit strategy allows the L2 merger (M07) to
decompose line-of-sight measurements into the two horizontal wind components.

**Orbit 1 — along-track (odd orbits):**
The +X ram-face aperture looks forward along the velocity direction, depressed
15.73° toward the limb. The LOS is nearly parallel to the spacecraft velocity.
V_sc_LOS ≈ −7,100 m/s dominates the measured Doppler shift. After L1c removal,
the residual wind LOS projection is primarily **meridional** (north-south).

**Orbit 2 — cross-track (even orbits):**
The boresight rotates 90° to the anti-Sun side (+Y body), depressed 15.73°.
V_sc_LOS ≈ 0–300 m/s. The residual wind LOS projection is primarily **zonal**
(east-west).

The observation is never purely meridional or purely zonal — M07's WLS
inversion uses the exact geometry vectors to solve the mixed system.

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

## 4. Orbit configuration

```python
T_ORBIT_S    = 5700       # orbital period at 510 km, seconds
DT_S         = 10.0       # timestep, seconds (~570 rows per orbit)
START_EPOCH  = "2027-01-01T00:00:00"
ALTITUDE_KM  = 510.0      # confirmed mission altitude (S03)
H_TARGET_KM  = 250.0      # OI 630 nm tangent height (S03)
```

Orbit 2 starts at the final epoch of Orbit 1. Look mode switches at the
orbit boundary exactly — no transition epoch.

---

## 5. Notebook structure

Six sections. Each ends with a clearly labelled PASS / FAIL cell. The
notebook must run top-to-bottom as a single execution with no hidden state.

### Section 1 — Setup and orbit propagation

Imports, constants, propagate two full orbits via NB01.

**PASS criterion:** Both DataFrames have ~570 rows, no NaN values, Orbit 2
starts at Orbit 1's final epoch.

```python
from geometry.nb01_orbit_propagator import propagate_orbit
from geometry.nb02a_boresight import compute_los_eci
from geometry.nb02b_tangent_point import compute_tangent_point
from geometry.nb02c_los_projection import compute_v_rel
from geometry.nb02d_l1c_calibrator import remove_spacecraft_velocity
from windmap.nb00_wind_map import UniformWindMap
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from astropy.time import Time

df_orbit1 = propagate_orbit(START_EPOCH, T_ORBIT_S, DT_S, ALTITUDE_KM)
df_orbit2 = propagate_orbit(
    df_orbit1['epoch'].iloc[-1].isoformat(), T_ORBIT_S, DT_S, ALTITUDE_KM)
```

### Section 2 — LOS vectors, tangent points, v_rel for both orbits

Loop over all epochs in both orbits, computing NB02a → NB02b → NB02c
in sequence. Store results in a single `df_results` DataFrame.

**Output columns:**
`orbit, look_mode, epoch, sc_lat, sc_lon, tp_lat, tp_lon, tp_alt_km,
los_x, los_y, los_z, v_wind_LOS, V_sc_LOS, v_earth_LOS, v_rel`

**PASS criterion:** No NaN or Inf in any column. All `tp_alt_km` within
5 km of H_TARGET_KM.

```python
wind_map = UniformWindMap(v_zonal_ms=100.0, v_merid_ms=50.0)
results = []
for orbit_num, (df, look_mode) in enumerate(
        [(df_orbit1, 'along_track'), (df_orbit2, 'cross_track')], start=1):
    for _, row in df.iterrows():
        pos = np.array([row.pos_eci_x, row.pos_eci_y, row.pos_eci_z])
        vel = np.array([row.vel_eci_x, row.vel_eci_y, row.vel_eci_z])
        epoch = Time(row.epoch)
        los, q = compute_los_eci(pos, vel, look_mode, ALTITUDE_KM, H_TARGET_KM)
        tp = compute_tangent_point(pos, los, epoch, H_TARGET_KM)
        res = compute_v_rel(wind_map,
                            tp['tp_lat_deg'], tp['tp_lon_deg'], tp['tp_eci'],
                            vel, los, epoch)
        results.append({
            'orbit': orbit_num, 'look_mode': look_mode, 'epoch': row.epoch,
            'sc_lat': row.lat_deg, 'sc_lon': row.lon_deg,
            'tp_lat': tp['tp_lat_deg'], 'tp_lon': tp['tp_lon_deg'],
            'tp_alt_km': tp['tp_alt_km'],
            'los_x': los[0], 'los_y': los[1], 'los_z': los[2],
            'v_wind_LOS': res['v_wind_LOS'],
            'V_sc_LOS': res['V_sc_LOS'],
            'v_earth_LOS': res['v_earth_LOS'],
            'v_rel': res['v_rel'],
        })
df_results = pd.DataFrame(results)
orb1 = df_results[df_results.orbit == 1]
orb2 = df_results[df_results.orbit == 2]
```

### Section 3 — Four geometric verification plots

Four figures, all saved to `outputs/`. All must be visually inspected
and confirmed before proceeding to Section 4.

**Plot 3.1 — Ground track and tangent points map**
File: `outputs/INT01_groundtrack.png`

- Orbit 1 ground track (blue) + tangent points (scatter, blue)
- Orbit 2 ground track (green) + tangent points (scatter, green)
- ±40° science band shown as dashed orange lines

**Visual checks:**
- Along-track tangent points are **ahead** of the spacecraft ground track
  by ~8–10° longitude at the equator (≈ 923 km forward)
- Cross-track tangent points are displaced **laterally** to one side
- Both orbits cover the ±40° science latitude band
- No tangent points at the spacecraft position (indicates a failed
  ray intersection that returned the spacecraft location)

**Plot 3.2 — Tangent point altitude vs latitude**
File: `outputs/INT01_tp_altitude.png`

- Both orbits shown; y-axis limited to 240–260 km
- Horizontal dashed line at 250 km target

**Visual checks:**
- All altitudes within ±5 km of 250 km
- Slight latitude-dependent variation is acceptable (WGS84 oblateness)

**Plot 3.3 — V_sc_LOS comparison between look modes**
File: `outputs/INT01_Vsc_LOS.png`

- Time series of V_sc_LOS for both orbits on the same axes
- Along-track should hover around −7,100 m/s
- Cross-track should be near zero (< 300 m/s magnitude)

**Visual checks:**
- Clear > 10× separation in V_sc_LOS magnitude between look modes
- Along-track V_sc_LOS smooth and consistent across orbit
- No step discontinuities

**Plot 3.4 — v_rel time series**
File: `outputs/INT01_v_rel_timeseries.png`

- v_rel for both orbits vs time or latitude
- Annotate the dominant contributions (V_sc_LOS, v_earth_LOS, v_wind)

**Visual checks:**
- Smooth, physically plausible values
- Orbit 1 shows large negative v_rel (~−7,100 m/s + wind + Earth rotation)
- Orbit 2 shows small v_rel dominated by Earth rotation and wind projection

### Section 4 — Quantitative verification checks (V1–V5)

```python
print("=" * 60)
print("INT01 QUANTITATIVE VERIFICATION CHECKS")
print("=" * 60)

# V1 — Tangent altitude within 5 km of 250 km target
tp_alt = df_results.tp_alt_km.values
v1 = np.all(np.abs(tp_alt - H_TARGET_KM) < 5.0)
print(f"\nV1 — Tangent altitude within 5 km of {H_TARGET_KM} km:")
print(f"     Max deviation: {np.max(np.abs(tp_alt - H_TARGET_KM)):.2f} km")
print(f"     {'PASS' if v1 else 'FAIL'}")

# V2 — Along-track tangent point forward of spacecraft at equator
eq_mask = (orb1.sc_lat.abs() < 5)
if eq_mask.sum() > 0:
    tp_offset_km = (orb1[eq_mask].tp_lon - orb1[eq_mask].sc_lon).mean() * 111
    v2 = abs(tp_offset_km) > 500
    print(f"\nV2 — Tangent point forward offset at equator:")
    print(f"     Mean offset: {tp_offset_km:.0f} km  (expected > 500 km forward)")
    print(f"     {'PASS' if v2 else 'FAIL'}")

# V3 — V_sc_LOS ratio along-track / cross-track > 10
ratio = abs(orb1.V_sc_LOS.mean()) / (abs(orb2.V_sc_LOS.mean()) + 1e-6)
v3 = ratio > 10.0
print(f"\nV3 — V_sc_LOS ratio (along-track / cross-track):")
print(f"     Along-track mean: {orb1.V_sc_LOS.mean():.0f} m/s")
print(f"     Cross-track mean: {orb2.V_sc_LOS.mean():.0f} m/s")
print(f"     Ratio: {ratio:.1f}×  (must be > 10×)")
print(f"     {'PASS' if v3 else 'FAIL'}")

# V4 — No NaN or Inf in any output column
num_cols = df_results.select_dtypes(include=np.number)
v4 = (df_results.isna().sum().sum() == 0 and
      np.all(np.isfinite(num_cols.values)))
print(f"\nV4 — No NaN or Inf values in any output column:")
print(f"     NaN count: {df_results.isna().sum().sum()}")
print(f"     {'PASS' if v4 else 'FAIL'}")

# V5 — v_rel smoothness (no spikes > 100 m/s between adjacent epochs)
for orbit_num, orb_df in [(1, orb1), (2, orb2)]:
    diffs = np.abs(np.diff(orb_df.v_rel.values))
    v5 = np.max(diffs) < 100.0
    print(f"\nV5 — v_rel smoothness (Orbit {orbit_num}):")
    print(f"     Max jump: {np.max(diffs):.2f} m/s  (must be < 100 m/s)")
    print(f"     {'PASS' if v5 else 'FAIL'}")
```

### Section 5 — L2 round-trip wind decomposition (V6 and V7)

This is the most important section. It verifies the complete geometry
pipeline can be inverted to recover the truth wind.

**Method:** For each tangent point pair — one from Orbit 1 (along-track)
and one from Orbit 2 (cross-track) at approximately the same lat/lon —
solve the 2×2 linear system:

```
G · [v_zonal, v_merid]ᵀ = d

where:
  G[i, :] = [LOS_eci[i] · ê_east_eci, LOS_eci[i] · ê_north_eci]
  d[i]    = v_wind_LOS[i]   (after L1c removal)
```

Use numpy least-squares (`np.linalg.lstsq`) for the 2×2 system.

**V6 — Primary wind map (100, 50) m/s:**

```python
# Find matched pairs at ±40° latitude, group by 5° longitude bins
# For each bin with both look modes, solve 2×2 and record recovered wind
recovered = []
for lon_bin in np.arange(-180, 180, 5.0):
    at_mask = ((orb1.tp_lon >= lon_bin) & (orb1.tp_lon < lon_bin + 5) &
               (orb1.tp_lat.abs() < 40))
    ct_mask = ((orb2.tp_lon >= lon_bin) & (orb2.tp_lon < lon_bin + 5) &
               (orb2.tp_lat.abs() < 40))
    if at_mask.sum() < 1 or ct_mask.sum() < 1:
        continue
    # Build geometry matrix and solve
    # ... (implementation left to Claude Code per the M07 geometry formalism)
    recovered.append({'lon_bin': lon_bin, 'v_zonal_rec': ..., 'v_merid_rec': ...})

df_rec = pd.DataFrame(recovered)
bias_z = (df_rec.v_zonal_rec - 100.0).mean()
bias_m = (df_rec.v_merid_rec - 50.0).mean()
rms_z  = (df_rec.v_zonal_rec - 100.0).std()
rms_m  = (df_rec.v_merid_rec - 50.0).std()
v6 = (abs(bias_z) < 1.0 and abs(bias_m) < 1.0 and
      rms_z < 2.0 and rms_m < 2.0)
print(f"\nV6 — L2 round-trip (primary wind 100, 50 m/s):")
print(f"     Zonal:  bias={bias_z:+.3f}, RMS={rms_z:.3f} m/s")
print(f"     Merid:  bias={bias_m:+.3f}, RMS={rms_m:.3f} m/s")
print(f"     {'PASS' if v6 else 'FAIL'}")
```

**V7 — Secondary wind maps:**
Repeat Section 5 for three secondary cases. Same pass criterion (bias < 1 m/s,
RMS < 2 m/s) for each. The zero-wind case verifies Earth-rotation removal.

### Section 6 — Save results and progress checkpoint

```python
df_results.to_csv('outputs/INT01_results.csv', index=False)

import json, pathlib
cp_path = pathlib.Path('simulations/checkpoints/progress.json')
cp_path.parent.mkdir(parents=True, exist_ok=True)
cp = json.loads(cp_path.read_text()) if cp_path.exists() else {}
cp['INT01_geometry_pipeline'] = {
    'status': 'complete',
    'checks_passed': 7,
    'notes': f'V1–V7 all PASS. Bias zonal={bias_z:.3f} m/s, merid={bias_m:.3f} m/s.'
}
cp_path.write_text(json.dumps(cp, indent=2))
print("Progress checkpoint updated.")
```

---

## 6. Complete verification checklist

| Check | Criterion | Blocking? |
|-------|-----------|-----------|
| V1 | Tangent altitude within ±5 km of 250 km | Yes |
| V2 | Along-track TP > 500 km forward of spacecraft | Yes |
| V3 | Along-track V_sc_LOS > 10× cross-track | Yes |
| V4 | Zero NaN / Inf in all output columns | Yes |
| V5 | Max v_rel epoch-to-epoch jump < 100 m/s | Yes |
| V6 | L2 round-trip bias < 1 m/s, RMS < 2 m/s (primary map) | Yes |
| V7 | Same criteria for three secondary wind maps | Yes |

All seven checks must show PASS. If any fails, diagnose and fix the
indicated module before proceeding to Tier 2 specs.

---

## 7. Output artefacts

The following must be committed to the repo:

```
outputs/
├── INT01_groundtrack.png          # Map of S/C tracks + tangent points
├── INT01_tp_altitude.png          # Tangent point altitude vs latitude
├── INT01_Vsc_LOS.png              # V_sc_LOS comparison, both look modes
├── INT01_v_rel_timeseries.png     # v_rel time series, both orbits
└── INT01_results.csv              # Full df_results DataFrame

simulations/checkpoints/
└── progress.json                  # Updated with INT01 entry
```

These are permanent visual records. If a future module change breaks the
geometry, re-running INT01 and comparing to the committed PNGs immediately
reveals what changed.

---

## 8. File location in repository

```
windcube-pipeline/
├── notebooks/
│   └── INT01_geometry_pipeline_YYYY-MM-DD.ipynb   ← this spec
├── outputs/
│   ├── INT01_groundtrack.png
│   ├── INT01_tp_altitude.png
│   ├── INT01_Vsc_LOS.png
│   ├── INT01_v_rel_timeseries.png
│   └── INT01_results.csv
├── simulations/checkpoints/
│   └── progress.json
└── docs/specs/
    └── S08_int01_geometry_notebook_2026-04-05.md   ← this file
```

---

## 9. Instructions for Claude Code

1. Read this entire spec AND S07 (NB02) AND S05 (NB00) before writing any
   code. Understand the interface contracts before opening a notebook.
2. Confirm all prerequisites pass:
   ```bash
   pytest tests/test_nb00_wind_map_*.py -v   # T1+T2 must pass
   pytest tests/test_nb01_orbit_*.py -v      # all 8 must pass
   pytest tests/test_nb02_geometry_*.py -v   # all 10 must pass
   ```
   Do not proceed if any prerequisite fails.
3. Create `notebooks/INT01_geometry_pipeline_YYYY-MM-DD.ipynb` with the
   six sections above, one cell per section, each ending with PASS/FAIL.
4. Create `outputs/` directory if absent.
5. Run the notebook top-to-bottom — all cells, no skipping.
6. All seven checks must show PASS.
7. Save all four PNGs and the CSV to `outputs/`.
8. **Visually inspect all four PNGs** before marking INT01 complete:
   - Plot 3.1: tangent points must be visibly ahead of spacecraft (not coincident)
   - Plot 3.2: all points within the 240–260 km band
   - Plot 3.3: clear 10× separation between look modes
   - Plot 3.4: smooth, physically plausible v_rel
9. Update `simulations/checkpoints/progress.json` (Section 6).
10. Commit everything — notebook, outputs, progress.json:
    ```
    feat(int01): geometry pipeline integration — all 7 checks pass, two orbits
    ```
11. **Do not write or implement S09–S11 (Tier 2) until this commit is confirmed.**
    The Tier 2 modules receive `v_rel` as input — if the geometry is wrong,
    every downstream module will be wrong too.
