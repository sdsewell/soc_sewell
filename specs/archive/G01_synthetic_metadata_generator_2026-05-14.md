# G01 — Synthetic Metadata Generator Specification

**Spec ID:** G01  
**Spec file:** `specs/G01_synthetic_metadata_generator_2026-05-14.md`  
**Previous spec:** `specs/G01_synthetic_metadata_generator_2026-05-13.md`  
**Script:** `src/processing/GEN01_synthesize_mission_dataset_2026_05_13.py`  
**Project:** WindCube FPI Science Operations Center Pipeline  
**Institution:** NCAR / High Altitude Observatory (HAO)  
**Status:** Authoritative — v14  
**Spec version:** 14  
**Date:** 2026-05-14  
**Git commit:** TBD  

**Depends on:**
- NB01 (`nb01_orbit_propagator_2026_04_16.py`) — `propagate_orbit(...)`
- NB02a (`nb02a_boresight_2026_04_16.py`) — `compute_los_eci(...)`
- NB02b (`nb02b_tangent_point_2026_04_16.py`) — `compute_tangent_point(...)`
- NB02c (`nb02c_los_projection_2026_04_16.py`) — `compute_v_rel(...)`
- P01 (`p01_image_metadata_2026_04_06.py`) — `ImageMetadata`, `AdcsQualityFlags`  
  **P01 v2 required** (`S19_p01_metadata_2026-05-14.md`) — adds `h_target_km_obs`
- `windcube/constants.py` — authoritative numerical constants

---

## Revision history

| Version | Date | Summary |
|---------|------|---------|
| 1–9 | 2026-04-16 | See archive. |
| 10 | 2026-04-28 | Physics alignment with Z03. |
| 11–12 | 2026-05-13 | Tolansky constants; Doppler shift; separate sci/cal exposure prompts; asymmetric lat band. |
| 13 | 2026-05-13 | Doppler round-trip verified; `TimeVaryingStormWindMap`; `ap_current` column; `_plot_vrel_histogram`. Commit `fbea032`. |
| **14** | **2026-05-14** | **Four changes (see §1 below): (a) add `obs_mode` CSV column; (b) rename and annotate LOS component CSV columns for sign-convention clarity; (c) populate `h_target_km_obs` in `ImageMetadata`; (d) replace `ap_current` column with `obs_mode` for non-storm-onset wind maps, keeping `ap_current` only where meaningful. Companion: S19 v2 (`P01_metadata_2026-05-14.md`), H07 v0.2.** |

---

## 1. v14 changes — summary

### 1a. Add `obs_mode` column to CSV output

The per-frame CSV currently has no column indicating whether each frame was
acquired in along-track or cross-track pointing mode. H07 wind retrieval
requires this information, and downstream analysts need it for quality
control.

**Change:** Add `obs_mode` as a new CSV column, populated from
`meta.obs_mode` (already present in `ImageMetadata` — GEN01 sets it from
`look_mode` at line 1449 of the v13 script). The column contains the string
`"along_track"`, `"cross_track"`, or `NaN` for non-science frames (obs_type
`"none"`). For cal and dark frames it takes the value of `look_mode` at the
time of acquisition (the spacecraft attitude during cal/dark sequences is
determined by the preceding look mode switch).

**Column position:** Insert `obs_mode` immediately after `obs_type` (second
column), so the frame classification columns are grouped together:

```
obs_type, obs_mode, rows, cols, exp_time, ...
```

### 1b. Rename and annotate LOS component CSV columns

**Problem:** The current column names `v_wind_los_ms`, `v_earth_los_ms`,
`v_sc_los_ms` do not communicate their sign convention. The values are
dot products of the respective velocity vectors with the unit LOS vector
pointing **from spacecraft toward tangent point** (`l̂_sc→tp`), so a positive
value means motion **toward** the tangent point (approach). This is the
**opposite** of the Harding (1950) convention used everywhere else in the
pipeline, where positive velocity means recession from the instrument.

The derived quantity `v_rel` IS already in Harding convention:
```
v_rel = −( v_sc_los_approach + v_earth_los_approach + v_wind_los_approach )
```
A positive `v_rel` correctly encodes recession (redshift). The pixel
generator uses `v_rel` directly as a Harding recession velocity.

The component columns are diagnostic/truth values — they are NOT consumed
by H07 directly. However, to prevent implementor confusion, rename them to
make the approach-positive sign explicit.

**Renamed columns:**

| Old name | New name | Sign convention |
|----------|----------|-----------------|
| `v_wind_los_ms` | `v_wind_los_approach_ms` | + = wind component toward TP |
| `v_earth_los_ms` | `v_earth_los_approach_ms` | + = Earth rotation toward TP |
| `v_sc_los_ms` | `v_sc_los_approach_ms` | + = SC velocity toward TP |
| `v_rel_ms` | `v_rel_ms` | **unchanged** — Harding convention, + = recession |

The relationship between them, written explicitly in a CSV header comment
block in the README sidecar (not in the CSV file itself, which has no
comment syntax):

```
v_rel_ms = −(v_sc_los_approach_ms + v_earth_los_approach_ms + v_wind_los_approach_ms)
v_rel_ms > 0 → recession (redshift) — Harding convention, consistent with M06/H07
```

**H07 inversion formula** (for implementor clarity): to recover the
wind LOS component in Harding convention from the CSV truth columns:

```
v_wind_harding = −v_wind_los_approach_ms
```

Or equivalently, from the observable:

```
v_wind_harding = −v_rel_ms − v_sc_los_approach_ms − v_earth_los_approach_ms
                 corrected via: v_wind_harding = v_rel_ms + v_sc_harding + v_earth_harding
                 where v_sc_harding = −v_sc_los_approach_ms (recession-positive)
```

### 1c. Populate `h_target_km_obs` in ImageMetadata

GEN01 already prompts for `h_target_km` (default 250.0 km) and uses it
throughout NB02a/NB02b, but does not write it to `ImageMetadata`. P01 v2
adds the `h_target_km_obs` field for exactly this purpose.

**Change:** Pass `h_target_km_obs=h_target_km` to the `ImageMetadata`
constructor in the main frame loop. Also add `h_target_km_obs` as a column
in the CSV output.

### 1d. `ap_current` column — retained, position unchanged

The `ap_current` column (added in v13) is retained. It carries the
instantaneous Ap geomagnetic index for storm-onset wind maps (option 6),
and `NaN` for all other wind map options. This is scientifically meaningful
and should not be removed. Its position at the end of the CSV is unchanged.

---

## 2–4. Unchanged from v13

Sections 2 (User interface), 3 (Wind map registry), 4 (Orbit propagation
and CONOPS scheduling) are unchanged.

---

## 5. Main metadata loop — v14 additions

### 5.1 `ImageMetadata` constructor call

Add `h_target_km_obs=h_target_km` to the `ImageMetadata(...)` constructor
call in the main frame loop (currently at approximately line 1432 of the
v13 script):

```python
meta = ImageMetadata(
    ...                              # all existing fields unchanged
    h_target_km_obs = h_target_km,  # NEW: from user prompt (default 250.0 km)
    ...
)
```

### 5.2 `vrel_list` dict — no changes

The `vrel_list` dicts are unchanged. The rename of CSV column names is
applied only at the CSV construction step (§5.3), not in the internal
dict keys.

### 5.3 CSV construction — v14 column changes

The CSV row dict in the main `rows_csv.append({...})` block must be updated
as follows. Only the changed keys are shown; all other keys are unchanged.

**For observed frames (the `if idx in obs_data:` branch):**

```python
rows_csv.append({
    "obs_type":               ft,
    "obs_mode":               r["obs_mode"],          # NEW — 1a
    "rows":                   r["rows"],
    # ... (all existing middle columns unchanged) ...
    "tp_lat_deg":             r["tangent_lat"]       if r["tangent_lat"]       is not None else _nan,
    "tp_lon_deg":             r["tangent_lon"]        if r["tangent_lon"]        is not None else _nan,
    "h_target_km_obs":        r["h_target_km_obs"]    if r["h_target_km_obs"]    is not None else _nan,  # NEW — 1c
    "wind_v_zonal_ms":        r["truth_v_zonal"]      if r["truth_v_zonal"]      is not None else _nan,
    "wind_v_merid_ms":        r["truth_v_meridional"] if r["truth_v_meridional"] is not None else _nan,
    "v_wind_los_approach_ms": vd["v_wind_los_ms"],    # RENAMED — 1b
    "v_earth_los_approach_ms":vd["v_earth_los_ms"],   # RENAMED — 1b
    "v_sc_los_approach_ms":   vd["v_sc_los_ms"],      # RENAMED — 1b
    "v_rel_ms":               vd["v_rel_ms"],          # unchanged
    "ap_current":             vd["ap_current"],        # unchanged
})
```

**For non-observing steps (the `else:` branch):**

```python
rows_csv.append({
    "obs_type":               "none",
    "obs_mode":               _nan,                   # NEW — 1a
    # ... (all existing middle columns unchanged) ...
    "tp_lat_deg":             _nan, "tp_lon_deg":     _nan,
    "h_target_km_obs":        _nan,                   # NEW — 1c
    "wind_v_zonal_ms":        _nan, "wind_v_merid_ms": _nan,
    "v_wind_los_approach_ms": _nan,                   # RENAMED — 1b
    "v_earth_los_approach_ms":_nan,                   # RENAMED — 1b
    "v_sc_los_approach_ms":   _nan,                   # RENAMED — 1b
    "v_rel_ms":               _nan,
    "ap_current":             _nan,
})
```

---

## 6. `ImageMetadata` field assignment — v14 update

**Additional mapping (v14):**

| `ImageMetadata` field | Source |
|----------------------|--------|
| `h_target_km_obs` | `h_target_km` (from user prompt, default 250.0 km) |

All other mappings unchanged from v13.

---

## 7. Binary image synthesis — unchanged from v13

Section 7 (constants, pixel generators, file writer) is unchanged.

---

## 8. Output files — v14 CSV column layout

### 8.1 Complete CSV column order (v14)

The v14 CSV has **51 columns** (v13 had 48). The three additions are
`obs_mode` (position 2), `h_target_km_obs` (position 42), and the three
renames. Column numbering is 1-based.

| # | Column name | Type | Units | Notes |
|---|-------------|------|-------|-------|
| 1 | `obs_type` | str | — | `science`, `cal`, `dark`, `none` |
| 2 | `obs_mode` | str | — | `along_track`, `cross_track`, NaN for `none` rows |
| 3 | `rows` | int | px | — |
| 4 | `cols` | int | px | — |
| 5 | `exp_time` | int | cs | Centiseconds |
| 6 | `exp_unit` | int | reg | CCD timing register |
| 7 | `ccd_temp1` | float | °C | — |
| 8 | `lua_timestamp` | int | ms | Unix epoch |
| 9 | `adcs_timestamp` | int | ms | Unix epoch |
| 10 | `spacecraft_latitude` | float | rad | — |
| 11 | `spacecraft_longitude` | float | rad | — |
| 12 | `spacecraft_altitude` | float | m | — |
| 13 | `att_q_x` | float | — | Attitude quaternion x (scalar-last) |
| 14 | `att_q_y` | float | — | — |
| 15 | `att_q_z` | float | — | — |
| 16 | `att_q_w` | float | — | — |
| 17 | `pe_q_x` | float | — | Pointing error quaternion x |
| 18 | `pe_q_y` | float | — | — |
| 19 | `pe_q_z` | float | — | — |
| 20 | `pe_q_w` | float | — | — |
| 21 | `pos_eci_x` | float | m | ECI position x |
| 22 | `pos_eci_y` | float | m | — |
| 23 | `pos_eci_z` | float | m | — |
| 24 | `vel_eci_x` | float | m/s | ECI velocity x |
| 25 | `vel_eci_y` | float | m/s | — |
| 26 | `vel_eci_z` | float | m/s | — |
| 27 | `etalon_t0` | float | °C | — |
| 28 | `etalon_t1` | float | °C | — |
| 29 | `etalon_t2` | float | °C | — |
| 30 | `etalon_t3` | float | °C | — |
| 31 | `gpio_0` | int | — | — |
| 32 | `gpio_1` | int | — | — |
| 33 | `gpio_2` | int | — | — |
| 34 | `gpio_3` | int | — | — |
| 35 | `lamp_0` | int | — | — |
| 36 | `lamp_1` | int | — | — |
| 37 | `lamp_2` | int | — | — |
| 38 | `lamp_3` | int | — | — |
| 39 | `lamp_4` | int | — | — |
| 40 | `lamp_5` | int | — | — |
| 41 | `tp_lat_deg` | float | deg | Tangent point geodetic latitude |
| 42 | `tp_lon_deg` | float | deg | Tangent point geodetic longitude |
| 43 | `h_target_km_obs` | float | km | **NEW** Intended emission layer altitude |
| 44 | `wind_v_zonal_ms` | float | m/s | Truth zonal wind at tangent point |
| 45 | `wind_v_merid_ms` | float | m/s | Truth meridional wind at tangent point |
| 46 | `v_wind_los_approach_ms` | float | m/s | **RENAMED** Wind LOS component, approach-positive |
| 47 | `v_earth_los_approach_ms` | float | m/s | **RENAMED** Earth rotation LOS, approach-positive |
| 48 | `v_sc_los_approach_ms` | float | m/s | **RENAMED** SC velocity LOS, approach-positive |
| 49 | `v_rel_ms` | float | m/s | Harding recession velocity (positive = redshift) |
| 50 | `ap_current` | float | — | Ap index (option 6 only; NaN otherwise) |

**Sign convention note embedded in README sidecar (§8.2):**

```
--- LOS velocity sign conventions ---
v_*_los_approach_ms columns: dot(velocity, l̂_sc→tp)
  Positive = motion TOWARD tangent point (approach).
  These are diagnostic/truth columns; not consumed directly by H07.

v_rel_ms: Harding recession convention.
  Positive = source receding from spacecraft (redshift, λ increases).
  Used directly by M06 pixel generator and H07 wind inversion.
  Relationship: v_rel_ms = −(v_sc_los_approach_ms
                             + v_earth_los_approach_ms
                             + v_wind_los_approach_ms)
```

### 8.2 README sidecar — v14 additions

Add the sign convention note above to the `--- Output ---` section of the
README `.txt` sidecar, and add:

```
Tangent height      : {h_target_km:.1f} km   (h_target_km_obs in metadata)
```

(This line already exists as `Tangent ht : {h_target_km:.1f} km` in v13 —
update its label to match the field name.)

### 8.3 v_rel histogram figure — unchanged from v13

---

## 9. Verification checks — v14 additions

All checks C1–C24 carry forward from v13. Additions:

**C25 — obs_mode column present and valid:**
For all science and cal/dark frames: `obs_mode ∈ {"along_track", "cross_track"}`.
For `obs_type == "none"` rows: `obs_mode` is NaN (pandas reads as float NaN).
No science frame may have NaN `obs_mode`.

**C26 — h_target_km_obs column present and consistent:**
For all science frames: `h_target_km_obs == h_target_km` (the prompt value).
For non-science frames: NaN permitted (cal/dark frames do not have a tangent
point but the spacecraft was at the prompt altitude).
For frames where `tangent_alt_km` is populated: `|tangent_alt_km - h_target_km_obs| < 0.5 km`.

**C27 — Renamed columns present, old names absent:**
CSV header must contain `v_wind_los_approach_ms`, `v_earth_los_approach_ms`,
`v_sc_los_approach_ms`. Must NOT contain `v_wind_los_ms`, `v_earth_los_ms`,
`v_sc_los_ms` (old names). Raise an assertion error if old names detected.

**C28 — Sign convention self-consistency:**
For all science frames where `v_rel_ms` is not NaN:
```
|v_rel_ms − (−v_sc_los_approach_ms − v_earth_los_approach_ms − v_wind_los_approach_ms)| < 0.01 m/s
```
This confirms the renamed columns retain the same numerical values and the
sign convention note in the README is consistent with the data.

---

## 10. File location in repository

```
soc_sewell/
├── src/
│   └── processing/
│       └── GEN01_synthesize_mission_dataset_2026_05_13.py   ← edit in place (v14)
└── specs/
    ├── G01_synthetic_metadata_generator_2026-05-14.md       ← this file (v14)
    └── archive/
        └── G01_synthetic_metadata_generator_2026-05-13.md   ← v13 archive
```

The script filename is **not** renamed for v14 — only the spec date changes.
This avoids breaking any existing import references.

---

## 11. Instructions for Claude Code

Read this entire spec AND the previous v13 spec before touching any code.
Read `S19_p01_metadata_2026-05-14.md` (P01 v2) for the `h_target_km_obs`
field definition. All four changes are small and surgical — do not refactor
anything beyond what is specified.

**Changes required:**

1. **`h_target_km_obs` in ImageMetadata constructor** (§5.1):
   In the main frame loop, add `h_target_km_obs=h_target_km` to the
   `ImageMetadata(...)` constructor call. This requires P01 v2 to be
   already implemented (test: `hasattr(ImageMetadata(), 'h_target_km_obs')`).

2. **`obs_mode` CSV column** (§5.3 / §8.1):
   In both the observed-frame and non-observing-step branches of `rows_csv`:
   - Insert `"obs_mode": r["obs_mode"]` immediately after `"obs_type": ft`
     in the observed-frame branch.
   - Insert `"obs_mode": _nan` immediately after `"obs_type": "none"`
     in the non-observing-step branch.

3. **Rename LOS component columns** (§5.3 / §8.1):
   In both CSV branches, rename:
   - `"v_wind_los_ms"` → `"v_wind_los_approach_ms"`
   - `"v_earth_los_ms"` → `"v_earth_los_approach_ms"`
   - `"v_sc_los_ms"` → `"v_sc_los_approach_ms"`
   The values are unchanged; only the dict keys change.

4. **`h_target_km_obs` CSV column** (§5.3 / §8.1):
   In both CSV branches, add:
   - Observed: `"h_target_km_obs": r["h_target_km_obs"] if r["h_target_km_obs"] is not None else _nan`
     inserted after `"tp_lon_deg"` and before `"wind_v_zonal_ms"`.
   - Non-observing: `"h_target_km_obs": _nan` at the same position.

5. **README sidecar** (§8.2):
   Add the sign convention note block and update the tangent height label.
   Append after the existing `--- Output ---` section.

6. **Verification checks C25–C28** (§9):
   Add four new checks to the end of the verification block. These can be
   implemented as assertions with descriptive messages immediately after
   the existing C24 check.

7. **Run a short synthetic dataset to verify:**
   ```
   Start epoch: 2027-01-01T00:00:00
   Duration: 0.1 days
   Wind map: [1] Uniform, v_zonal=0, v_merid=0
   All other prompts: defaults
   ```
   Confirm:
   - CSV has 51 columns including `obs_mode` and `h_target_km_obs`
   - No old column names (`v_wind_los_ms` etc.) present
   - C25–C28 all pass
   - 9/9 P01 tests still pass (`pytest tests/test_s19_p01_metadata.py -v`)

8. **Commit:**
   ```
   feat(gen01): v14 — obs_mode column, h_target_km_obs, LOS column renames
   Implements: G01_synthetic_metadata_generator_2026-05-14.md (v14)
   Requires: S19_p01_metadata_2026-05-14.md (v2, already merged)
   ```

**Do not:**
- Rename the script file
- Change any physics, pixel generators, or wind map builders
- Modify the `.npy` output (it stores the full `ImageMetadata` object
  which already carries `h_target_km_obs` once P01 v2 is implemented)
- Change the `_plot_vrel_histogram` function (it uses `m.obs_mode`
  which is already correct)

---

## 12. Constants cross-reference — unchanged from v13

See v13 spec §11. No constants changed in v14.

---

*End of G01 Specification v14 — 2026-05-14*
