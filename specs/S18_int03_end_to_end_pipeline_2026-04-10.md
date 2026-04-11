# S18 — INT03 Full End-to-End Pipeline Integration Script

**Spec ID:** S18  
**Spec file:** `docs/specs/S18_int03_end_to_end_pipeline_2026-04-10.md`  
**Project:** WindCube FPI Science Operations Center Pipeline  
**Institution:** NCAR / High Altitude Observatory (HAO)  
**Status:** Authoritative — ready for Claude Code implementation  
**Tier:** 7 — Integration notebooks  
**Depends on:** S01, S02, S03, S04,
  S05 (NB00 — truth wind map),
  S06 (NB01 — orbit propagator / SGP4),
  S07 (NB02 — geometry, boresight, tangent, LOS, L1c),
  S08 (INT01 — geometry integration),
  S09 (M01 — Airy forward model),
  S10 (M02 — calibration fringe synthesis),
  S11 (M04 — airglow synthesis / delta-function model),
  S12 (M03 — annular reduction, dark subtraction),
  S13 (Tolansky — 2-line etalon characterisation),
  S14 (M05 — calibration inversion),
  S15 (M06 — airglow fringe inversion),
  S16 (M07 — L2 vector wind retrieval),
  S17 (INT02 — FPI chain integration),
  S19 (P01 — metadata schema)  
**Used by:** S20 (L2 data product netCDF)  
**Last updated:** 2026-04-10  
**Created/Modified by:** Claude AI  

---

## 1. Purpose and philosophy

INT03 is the capstone integration script of the WindCube SOC pipeline. It
exercises every module from `soc_sewell` in a single continuous run:
orbit propagation → fringe synthesis → image reduction → calibration
inversion → airglow inversion → L2 vector wind retrieval → science
assessment. The result is a recovered horizontal wind field that can be
compared directly against the injected truth wind map.

**What INT03 proves that INT02 does not.** INT02 validates the FPI optical
chain in complete isolation — it uses a placeholder geometry stub with no
orbit propagator. INT03 is the first test where the geometry pipeline
(NB00–NB02) is live and fully coupled to the FPI chain:

- Spacecraft state is propagated from a real TLE via SGP4 (NB01).
- Tangent point coordinates, LOS unit vectors, and sensitivity coefficients
  are computed geometrically from that state (NB02).
- The truth wind speed for each observation is sampled from the NB00 wind
  map at the computed tangent point.
- The FPI fringe image is synthesised with that truth wind, processed
  through M03 → M05 → M06, and the recovered `v_rel_ms` is handed directly
  to M07 for vector decomposition.
- The recovered wind vector is compared against the truth wind map at the
  paired tangent point.

The diagnostic value of INT03 is precisely its isolation structure:

| If INT02 passes, INT03 fails | Problem is in: geometry chain or geometry↔FPI coupling |
| If INT02 fails, INT03 fails  | Problem is in: FPI chain (fix INT02 first) |
| Both pass                    | Pipeline is end-to-end validated |

**What INT03 is not.** INT03 does not process real on-orbit binary data —
that is a future operational script. INT03 does not write a netCDF L2 file —
that is S20. INT03 does not test the Tolansky pipeline on real neon images
— that is Z01.

**Format.** INT03 is a Python script (not a Jupyter notebook), consistent
with INT01 and INT02. It lives in `src/integration/`. Stages run
sequentially; a FAIL in any stage does not abort subsequent stages. All
failures are collected and reported together at the end. The script
exits with code 0 if all verification checks pass, 1 otherwise.

It can be invoked with `--quick` (reduced orbit arc, 1 orbit pair instead of
the full 4) for rapid development iteration.

---

## 2. Architecture overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  Stage A — Setup and configuration                                   │
│     InstrumentParams  ·  WindMapT1 (NB00)  ·  TLE  ·  RNG seeds    │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────┐
│  Stage B — Geometry pipeline                                         │
│     NB01: SGP4 orbit propagation → spacecraft state table            │
│     NB02: boresight · tangent points · LOS unit vectors              │
│           sensitivity coefficients (A_e, A_n)                        │
│           v_sc_LOS · v_earth_LOS for each observation                │
└───────────────────────────────┬─────────────────────────────────────┘
                                │  tangent_point_table + LOS table
┌───────────────────────────────▼─────────────────────────────────────┐
│  Stage C — Truth wind sampling                                       │
│     Sample NB00 wind map at each tangent point                       │
│     Project v_zonal, v_merid onto each LOS → v_wind_LOS (truth)     │
│     Compute v_rel_truth = v_wind_LOS − v_sc_LOS − v_earth_LOS       │
└───────────────────────────────┬─────────────────────────────────────┘
                                │  v_rel_truth per observation
┌───────────────────────────────▼─────────────────────────────────────┐
│  Stage D — Calibration pipeline  (once per run)                      │
│     M02: synthesise neon calibration image                           │
│     M03: dark subtraction + annular reduction → FringeProfile        │
│     S13: Tolansky 2-line → d, f, α, ε_cal priors                    │
│     M05: staged calibration inversion → CalibrationResult            │
└───────────────────────────────┬─────────────────────────────────────┘
                                │  CalibrationResult (shared)
┌───────────────────────────────▼─────────────────────────────────────┐
│  Stage E — Per-observation FPI chain                                 │
│     For each observation (orbit, frame):                             │
│       M04: synthesise airglow image at v_rel_truth                   │
│       M03: dark-subtract + reduce → FringeProfile_sci                │
│       M06: fit airglow fringe → AirglowFitResult (v_rel_rec)        │
│       P01: attach ImageMetadata                                      │
└───────────────────────────────┬─────────────────────────────────────┘
                                │  AirglowFitResult table
┌───────────────────────────────▼─────────────────────────────────────┐
│  Stage F — L2 vector wind retrieval                                  │
│     M07: pair along-track / cross-track observations at shared       │
│          tangent points  →  WindResult (v_zonal, v_merid)            │
└───────────────────────────────┬─────────────────────────────────────┘
                                │  WindResult table
┌───────────────────────────────▼─────────────────────────────────────┐
│  Stage G — Science assessment                                        │
│     Compare recovered wind to truth at each paired location          │
│     Compute global statistics: bias, RMS error, σ calibration        │
│     Generate all diagnostic figures                                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Scope decisions

### 3.1 Orbit geometry: two orbit pairs

INT03 uses **4 consecutive orbits** (2 along-track, 2 cross-track) in the
nominal WindCube SSO configuration. This provides:

- 2 fully-paired tangent point sets for M07 wind vector retrieval.
- Latitude coverage from ≈ −70° to +70° (SSO ground track).
- A sufficient sample of paired wind vectors (≈ 10–15 per orbit pair) to
  compute meaningful bias and RMS statistics.

With `--quick`, use 1 orbit pair (2 orbits).

### 3.2 Calibration is shared across all observations

One calibration image is synthesised and inverted at the start of the run
(Stage D). The resulting `CalibrationResult` is used for all subsequent
science frame reductions. This reflects the operational scenario where a
single calibration frame is taken before each science data arc.

### 3.3 Wind map: T1 (zonal) only, simplified

INT03 uses the T1 HWM-derived zonal wind map from NB00 as the truth.
Meridional component is set to zero unless the `--full-wind` flag is given.
This simplification makes the final assessment cleaner for the capstone test
without affecting the structural validity of the integration.

`--full-wind` enables both components and is intended for science validation
runs after the basic pipeline is confirmed working.

### 3.4 Noise: SNR = 5 throughout

All synthetic airglow images are generated at `snr = 5.0`, the dayside-
equivalent SNR from the WindCube STM. This is the primary science-mode
scenario. The noise-free case (`snr = None`) can be requested with
`--noiseless` for diagnostic purposes.

### 3.5 Dark frame

One master dark frame is created at Stage A from Poisson-sampled dark current
and shared across all calibration and science reductions. This matches the
operational scenario.

### 3.6 Metadata

`build_synthetic_metadata()` (P01) is called for every science frame with the
actual NB01 spacecraft state row and NB02 geometry. This is the first test
where `ImageMetadata` carries real geometry — not placeholders. V9 verifies
that the metadata tangent point coordinates agree with the NB02 geometry
table to within floating-point precision.

---

## 4. Verification checks

| ID | Stage | Check | Pass criterion |
|----|-------|-------|----------------|
| V1 | B | NB01 propagation succeeds; state table non-empty | len(state_table) ≥ 4 × n_frames_per_orbit |
| V2 | B | All tangent points at physically plausible altitude | 200 ≤ tp_alt_km ≤ 350 km for all observations |
| V3 | B | LOS sensitivity matrix well-conditioned for all pairs | condition_number < 100 for all paired observations |
| V4 | C | v_rel_truth magnitudes physically plausible | \|v_rel_truth\| < 1000 m/s for all observations |
| V5 | D | Tolansky d recovery | \|d − ETALON_GAP_M\| < 1 µm |
| V6 | D | M05 calibration fit quality | 0.5 < χ²_red < 3.0 |
| V7 | D | M05 t_m recovery within 1 nm of prior | \|t_m − prior\| < 1 nm |
| V8 | E | M06 χ²_red in bounds for all observations | 0.5 < χ²_red < 3.0, < 5% failure rate |
| V9 | E | Metadata tangent point matches NB02 geometry | \|tp_lat_deg − meta.tp_lat\| < 1e-10° for all obs |
| V10 | F | M07 all pairs well-conditioned | No ILL_CONDITIONED flags on primary pairs |
| V11 | G | v_rel round-trip bias across all observations | mean(\|v_rel_rec − v_rel_truth\|) < 20 m/s |
| V12 | G | v_zonal round-trip RMS | RMS(v_zonal_rec − v_zonal_truth) < 30 m/s |
| V13 | G | v_merid round-trip RMS | RMS(v_merid_rec − v_merid_truth) < 30 m/s |
| V14 | G | sigma_v_zonal within 2× STM budget | median(sigma_v_zonal) ≤ 2 × WIND_BIAS_BUDGET_MS |

**Note on V12/V13 tolerances.** At SNR = 5 and σ_v_rel ≈ 9.8 m/s with
matrix amplification ≈ 1.04, the expected per-observation σ_v_zonal is
≈ 10–15 m/s. The 30 m/s RMS tolerance allows for a 2σ safety margin and
accommodates small geometric imperfections in the synthetic truth sampling.
With `--noiseless`, V12/V13 tighten to 5 m/s.

---

## 5. Script structure

Fourteen stages. Each stage is a named function. A FAIL does not abort.

```
Stage A  — Setup: InstrumentParams, wind map, TLE, dark frame
Stage B  — Geometry pipeline: NB01 orbit propagation + NB02 geometry
Stage C  — Truth wind sampling and v_rel_truth computation
Stage D  — Calibration pipeline (shared CalibrationResult)
Stage E  — Per-observation FPI chain (M04 → M03 → M06 → P01)
Stage F  — L2 vector wind retrieval (M07)
Stage G  — Science assessment and verification
Stage H  — Figure: orbit ground track + tangent points
Stage I  — Figure: truth wind map vs tangent point sampling
Stage J  — Figure: calibration diagnostics (D and E combined)
Stage K  — Figure: per-observation v_rel truth vs recovered
Stage L  — Figure: M07 wind vector map (recovered vs truth)
Stage M  — Figure: bias and RMS error summary
Stage N  — Summary report
```

---

## 6. Imports

```python
"""
INT03 — Full end-to-end pipeline integration script.

Spec:        docs/specs/S18_int03_end_to_end_pipeline_2026-04-10.md
Spec date:   2026-04-10
Tool:        Claude Code
Depends on:  all S01–S19 modules in soc_sewell
Usage:
    python src/integration/int03_end_to_end_2026_04_10.py
    python src/integration/int03_end_to_end_2026_04_10.py --quick
    python src/integration/int03_end_to_end_2026_04_10.py --noiseless
    python src/integration/int03_end_to_end_2026_04_10.py --full-wind
"""

import argparse
import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Geometry pipeline
from src.geometry.nb00_wind_map      import WindMapT1, sample_wind_at_point
from src.geometry.nb01_orbit         import propagate_orbit, load_tle_windcube
from src.geometry.nb02a_boresight    import compute_boresight_eci
from src.geometry.nb02b_tangent      import compute_tangent_point
from src.geometry.nb02c_los_projection import (
    compute_sensitivity_coefficients,
    compute_los_eci,
    enu_unit_vectors_eci,
    compute_v_rel,
)

# FPI chain
from src.constants import (
    NE_WAVELENGTH_1_M, OI_WAVELENGTH_M, ETALON_GAP_M,
    WIND_BIAS_BUDGET_MS, SPEED_OF_LIGHT_MS,
    CCD_DARK_RATE_E_PX_S,
)
from src.fpi import (
    InstrumentParams,
    synthesise_calibration_image,
    synthesise_airglow_image,
    make_master_dark,
    reduce_calibration_frame,
    reduce_science_frame,
    TolanskyPipeline, TwoLineAnalyser,
    FitConfig, fit_calibration_fringe,
    fit_airglow_fringe,
)
from src.fpi.m07_wind_retrieval_2026_04_06 import (
    WindObservation, retrieve_wind_vectors,
)
from src.metadata import build_synthetic_metadata, ImageMetadata

OUTPUT_DIR = pathlib.Path("outputs")
```

**Implementation note.** The exact import paths for NB00–NB02 modules must
be confirmed by Claude Code before writing any import lines. The canonical
names are in `CLAUDE.md` and the respective spec files. Adjust import
statements to match the actual filenames in `src/geometry/`.

---

## 7. Stage A — Setup

```python
def stage_a_setup(quick: bool, noiseless: bool, full_wind: bool) -> dict:
    """
    Initialise all shared objects:
      - InstrumentParams
      - WindMapT1 (NB00 truth wind map)
      - master_dark frame
      - RNG seeds (fixed for reproducibility)
      - orbit configuration (number of orbit pairs)
    Print configuration summary.
    Return config dict.
    """
```

Configuration:

```python
params = InstrumentParams()
wind_map = WindMapT1()          # HWM-derived zonal/meridional at 250 km

n_orbit_pairs = 1 if quick else 2
n_frames_per_orbit = 8          # tangent point frames per orbit arc
                                # (reduces to 4 with --quick)
snr = None if noiseless else 5.0

# Master dark (shared across all frames)
dark_rate = CCD_DARK_RATE_E_PX_S    # from S03 constants
exp_time_s = 5.0                    # calibration exposure
img_size = (int(params.r_max * 2), int(params.r_max * 2))
rng_dark = np.random.default_rng(seed=42)
dark_array = rng_dark.poisson(
    dark_rate * exp_time_s, size=img_size
).astype(np.uint16)
master_dark = make_master_dark([dark_array])

# RNG allocation strategy: deterministic, seeded from observation index
# seed_cal = 0
# seed_sci(orbit_idx, frame_idx) = 100 + orbit_idx * 20 + frame_idx

config = {
    "params":           params,
    "wind_map":         wind_map,
    "master_dark":      master_dark,
    "n_orbit_pairs":    n_orbit_pairs,
    "n_frames":         4 if quick else n_frames_per_orbit,
    "snr":              snr,
    "noiseless":        noiseless,
    "full_wind":        full_wind,
    "quick":            quick,
    "r_max_px":         params.r_max,
    "n_bins":           150,
}
```

Print to console:
```
══════════════════════════════════════════════════════════════════
  WindCube INT03 — Full End-to-End Pipeline Integration
══════════════════════════════════════════════════════════════════
Mode : {'QUICK' if quick else 'FULL'}  |  {'NOISELESS' if noiseless else f'SNR={snr:.1f}'}  |  {'FULL-WIND' if full_wind else 'ZONAL-ONLY'}
Orbit pairs : {n_orbit_pairs}
Frames/orbit: {n_frames}
InstrumentParams:
  t_m     = {params.t_m*1e3:.6f} mm
  R_refl  = {params.R_refl:.3f}
  alpha   = {params.alpha:.4e} rad/px
  r_max   = {params.r_max:.1f} px
  sigma0  = {params.sigma0:.3f} px
```

---

## 8. Stage B — Geometry pipeline

```python
def stage_b_geometry(config: dict) -> tuple[dict, list]:
    """
    Propagate spacecraft orbit using NB01 (SGP4).
    Compute tangent points, LOS unit vectors, ENU basis vectors,
    and velocity projections (v_sc_LOS, v_earth_LOS) using NB02.
    Populate the observation table: one row per (orbit, frame).
    Execute V1, V2, V3.
    Return (geometry_results, checks).
    """
```

### 8.1 Orbit propagation (NB01)

```python
tle = load_tle_windcube()       # canonical WindCube TLE from S03 / CLAUDE.md

# Propagate for n_orbit_pairs × 2 orbits × n_frames epochs
# Start time: 2027-01-01T00:00:00 UTC (nominal mission start)
# Frame spacing: derived from orbital period / n_frames
state_table = propagate_orbit(tle, n_orbits=n_orbit_pairs * 2,
                               n_frames=config["n_frames"])
```

`state_table` is a pandas DataFrame with columns:
`epoch, orbit_idx, frame_idx, look_mode,
 pos_eci_x/y/z, vel_eci_x/y/z,
 sc_lat, sc_lon, sc_alt_km`

where `look_mode` alternates between `"along_track"` and `"cross_track"`
by orbit (even orbits = along-track, odd orbits = cross-track).

**Implementation note.** The actual NB01 call signature must be verified
against `src/geometry/nb01_*.py` before writing. Adapt as needed; the
intent is to obtain a state table with ≥ 4 × n_frames rows.

**V1 — state table non-empty:**
```python
v1_pass = len(state_table) >= 4 * config["n_frames"]
```

### 8.2 Geometry computation (NB02)

For each row in `state_table`:

```python
# NB02a: boresight
boresight = compute_boresight_eci(row["pos_eci"], row["vel_eci"],
                                   look_mode=row["look_mode"])

# NB02b: tangent point
tp = compute_tangent_point(row["pos_eci"], boresight)
# tp: {tp_lat_deg, tp_lon_deg, tp_alt_km, tp_eci}

# NB02c: LOS and sensitivity coefficients
los_eci = compute_los_eci(row["pos_eci"], tp["tp_eci"])
e_east_eci, e_north_eci = enu_unit_vectors_eci(tp["tp_lat_deg"],
                                                 tp["tp_lon_deg"])
A_e = float(np.dot(e_east_eci, los_eci))
A_n = float(np.dot(e_north_eci, los_eci))

# NB02c: velocity projections
vel_dict = compute_v_rel(row["pos_eci"], row["vel_eci"], los_eci,
                          tp["tp_eci"])
# vel_dict: {V_sc_LOS, v_earth_LOS, v_wind_LOS (placeholder)}
```

Assemble an observation record per frame:
```python
obs = {
    "orbit_idx":    row["orbit_idx"],
    "frame_idx":    row["frame_idx"],
    "look_mode":    row["look_mode"],
    "epoch":        row["epoch"],
    "sc_lat":       row["sc_lat"],
    "sc_lon":       row["sc_lon"],
    "sc_alt_km":    row["sc_alt_km"],
    "tp_lat_deg":   tp["tp_lat_deg"],
    "tp_lon_deg":   tp["tp_lon_deg"],
    "tp_alt_km":    tp["tp_alt_km"],
    "tp_eci":       tp["tp_eci"],
    "los_eci":      los_eci,
    "A_e":          A_e,
    "A_n":          A_n,
    "V_sc_LOS":     vel_dict["V_sc_LOS"],
    "v_earth_LOS":  vel_dict["v_earth_LOS"],
    "nb01_row":     row,         # full row for P01 metadata
}
```

**V2 — tangent altitudes plausible:**
```python
alts = [o["tp_alt_km"] for o in obs_list]
v2_pass = all(200 <= a <= 350 for a in alts)
```

**V3 — all along-track/cross-track pairs well-conditioned:**

For each pair (orbit 0, orbit 1) at matched frame index:
```python
A = np.array([[obs_at["A_e"], obs_at["A_n"]],
              [obs_ct["A_e"], obs_ct["A_n"]]])
cond = np.linalg.cond(A)
v3_pair_pass = cond < 100
```

Return the observation list and geometry checks.

---

## 9. Stage C — Truth wind sampling

```python
def stage_c_truth_winds(config: dict, geometry_results: dict) -> tuple[dict, list]:
    """
    For each observation, sample the NB00 wind map at the tangent point.
    Project (v_zonal, v_merid) onto the LOS → v_wind_LOS_truth.
    Compute v_rel_truth = v_wind_LOS_truth - V_sc_LOS - v_earth_LOS.
    Execute V4.
    Return (truth_results, checks).
    """
```

For each observation:

```python
v_zonal_truth, v_merid_truth = sample_wind_at_point(
    config["wind_map"],
    lat_deg=obs["tp_lat_deg"],
    lon_deg=obs["tp_lon_deg"],
    alt_km=obs["tp_alt_km"],
)

if not config["full_wind"]:
    v_merid_truth = 0.0

# Project onto LOS
v_wind_LOS_truth = (v_zonal_truth * obs["A_e"]
                    + v_merid_truth * obs["A_n"])

# Remove spacecraft and Earth-rotation projections
# Sign convention: v_rel = v_wind_LOS + V_sc_LOS + v_earth_LOS
# (matches S07/NB02d remove_spacecraft_velocity convention)
# Rearranging: v_wind_LOS = v_rel − V_sc_LOS − v_earth_LOS
# → v_rel = v_wind_LOS + V_sc_LOS + v_earth_LOS
v_rel_truth = (v_wind_LOS_truth
               + obs["V_sc_LOS"]
               + obs["v_earth_LOS"])

obs["v_zonal_truth"]    = v_zonal_truth
obs["v_merid_truth"]    = v_merid_truth
obs["v_wind_LOS_truth"] = v_wind_LOS_truth
obs["v_rel_truth"]      = v_rel_truth
```

**V4 — v_rel magnitudes physically plausible:**
```python
v_rels = [o["v_rel_truth"] for o in obs_list]
v4_pass = all(abs(v) < 1000 for v in v_rels)
```

Print summary:
```
[Stage C] Truth wind sampling:
  n_observations : {len(obs_list)}
  v_rel_truth    : {min:.1f} to {max:.1f} m/s  (mean ± std: {mean:.1f} ± {std:.1f})
  v_zonal range  : {min_z:.1f} to {max_z:.1f} m/s
  v_merid range  : {min_m:.1f} to {max_m:.1f} m/s  (0 unless --full-wind)
```

---

## 10. Stage D — Calibration pipeline

```python
def stage_d_calibration(config: dict) -> tuple[dict, list]:
    """
    Synthesise one calibration image, reduce it, run Tolansky,
    run M05 calibration inversion. The CalibrationResult is shared
    across all science frames in Stage E.
    Execute V5, V6, V7.
    Return (cal_results, checks).
    """
```

This stage is structurally identical to INT02 Stages C–F. Key
differences:

- The TolanskyPipeline is called via `TolanskyPipeline.run()` on the
  synthetic FringeProfile (not the `_build_tolansky_stub` shortcut used in
  INT02). This exercises the full calibration path.
- The `FitConfig` is built from the TwoLineResult, using the Tolansky-
  recovered `d`, `f`, `α` as priors for M05.

```python
rng_cal = np.random.default_rng(seed=0)

# M02: synthesise calibration image
cal_result_synth = synthesise_calibration_image(
    config["params"], add_noise=True, rng=rng_cal,
)
cal_image = cal_result_synth["image_2d"]

# M03: reduce
fp_cal = reduce_calibration_frame(
    cal_image,
    cx_human=config["params"].r_max,
    cy_human=config["params"].r_max,
    r_max_px=config["r_max_px"],
    n_bins=config["n_bins"],
    master_dark=config["master_dark"],
)

# S13: Tolansky
pipeline = TolanskyPipeline(fp_cal)
tol_result  = pipeline.run()
analyser2   = TwoLineAnalyser(fp_cal)
tol2_result = analyser2.run()

# M05: calibration inversion
fit_cfg  = FitConfig(tolansky=tol2_result)
cal_inv  = fit_calibration_fringe(fp_cal, fit_cfg)
```

**V5 — Tolansky d recovery (same criterion as INT02 V1):**
```python
d_error_um = abs(tol2_result.d_m - ETALON_GAP_M) * 1e6
v5_pass = d_error_um < 1.0
```

**V6 — M05 χ²_red:**
```python
v6_pass = 0.5 < cal_inv.chi2_reduced < 3.0
```

**V7 — M05 t_m within 1 nm of Tolansky prior:**
```python
t_error_nm = abs(cal_inv.t_m - tol2_result.d_m) * 1e9
v7_pass = t_error_nm < 1.0
```

Print calibration summary:
```
[Stage D] Calibration pipeline:
  Tolansky d       = {tol2_result.d_m*1e3:.6f} ± {tol2_result.sigma_d_m*1e6:.3f} µm
  M05 t_m          = {cal_inv.t_m*1e3:.6f} mm  (Δ = {t_error_nm:.4f} nm)
  M05 R_refl       = {cal_inv.R_refl:.4f}
  M05 epsilon_cal  = {cal_inv.epsilon_cal:.8f}
  M05 chi2_reduced = {cal_inv.chi2_reduced:.4f}
  M05 converged    = {cal_inv.converged}
  V5 (Tolansky d)  : {'PASS' if v5_pass else 'FAIL'}  {d_error_um:.3f} µm < 1.0 µm
  V6 (M05 chi2)    : {'PASS' if v6_pass else 'FAIL'}  {cal_inv.chi2_reduced:.3f}
  V7 (M05 t_m)     : {'PASS' if v7_pass else 'FAIL'}  {t_error_nm:.4f} nm
```

Return `{"fp_cal": fp_cal, "tol2_result": tol2_result, "cal_inv": cal_inv}`

---

## 11. Stage E — Per-observation FPI chain

```python
def stage_e_fpi_chain(
    config: dict,
    obs_list: list[dict],
    cal_results: dict,
) -> tuple[list[dict], list]:
    """
    For every observation in obs_list:
      1. Synthesise airglow image at v_rel_truth  (M04)
      2. Reduce science frame  (M03)
      3. Invert airglow fringe  (M06)  → v_rel_rec
      4. Attach ImageMetadata  (P01)
    Execute V8, V9.
    Return (obs_list_with_fpi_results, checks).
    """
```

For each observation at index `k`:

```python
obs = obs_list[k]
orbit_idx = obs["orbit_idx"]
frame_idx = obs["frame_idx"]
rng_sci   = np.random.default_rng(seed=100 + orbit_idx * 20 + frame_idx)

# M04: synthesise airglow image
sci_synth = synthesise_airglow_image(
    v_rel_ms=obs["v_rel_truth"],
    params=config["params"],
    snr=config["snr"],          # None for noiseless
    add_noise=(config["snr"] is not None),
    rng=rng_sci,
)
sci_image = sci_synth["image_2d"]

# M03: reduce science frame (reuse calibration frame centre)
fp_sci = reduce_science_frame(
    sci_image,
    cx=cal_results["fp_cal"].cx,
    cy=cal_results["fp_cal"].cy,
    sigma_cx=cal_results["fp_cal"].sigma_cx,
    sigma_cy=cal_results["fp_cal"].sigma_cy,
    r_max_px=config["r_max_px"],
    n_bins=config["n_bins"],
    master_dark=config["master_dark"],
)

# M06: airglow inversion
fit_result = fit_airglow_fringe(fp_sci, cal_results["cal_inv"])

# P01: metadata (real geometry from NB01/NB02)
meta = build_synthetic_metadata(
    params=config["params"],
    nb01_row=obs["nb01_row"],
    nb02_tp={
        "tp_lat_deg": obs["tp_lat_deg"],
        "tp_lon_deg": obs["tp_lon_deg"],
        "tp_alt_km":  obs["tp_alt_km"],
        "tp_eci":     obs["tp_eci"],
    },
    nb02_vr={
        "v_rel":       fit_result.v_rel_ms,
        "v_wind_LOS":  obs["v_wind_LOS_truth"],
        "V_sc_LOS":    obs["V_sc_LOS"],
        "v_earth_LOS": obs["v_earth_LOS"],
        "v_zonal_ms":  obs["v_zonal_truth"],
        "v_merid_ms":  obs["v_merid_truth"],
    },
    quaternion_xyzw=[0.0, 0.0, 0.0, 1.0],   # perfect pointing
    los_eci=obs["los_eci"],
    look_mode=obs["look_mode"],
    img_type="science",
    orbit_number=orbit_idx,
    frame_sequence=frame_idx,
    noise_seed=100 + orbit_idx * 20 + frame_idx,
)

obs["fp_sci"]        = fp_sci
obs["fit_result"]    = fit_result
obs["v_rel_rec"]     = fit_result.v_rel_ms
obs["sigma_v_rel"]   = fit_result.sigma_v_rel_ms
obs["chi2_sci"]      = fit_result.chi2_reduced
obs["meta"]          = meta
```

**V8 — M06 χ²_red in bounds, failure rate < 5%:**
```python
chi2_vals = [o["chi2_sci"] for o in obs_list]
n_fail    = sum(1 for c in chi2_vals if not (0.5 < c < 3.0))
v8_pass   = n_fail / len(chi2_vals) < 0.05
checks.append(("V8 M06 chi2_red in bounds",
               v8_pass,
               f"{n_fail}/{len(chi2_vals)} failed"))
```

**V9 — Metadata tangent point matches NB02 geometry:**
```python
for obs in obs_list:
    lat_err = abs(obs["meta"].tp_lat_deg - obs["tp_lat_deg"])
    lon_err = abs(obs["meta"].tp_lon_deg - obs["tp_lon_deg"])
    # Both must be < 1e-10 degrees (floating-point copy fidelity)
v9_pass = all(lat_err < 1e-10 and lon_err < 1e-10 for ...)
```

Print progress as observations complete:
```
[Stage E] FPI chain — observation {k+1}/{n_total}:
  orbit={orbit_idx} frame={frame_idx} look={look_mode}
  tp=({tp_lat:.2f}°, {tp_lon:.2f}°, {tp_alt:.0f} km)
  v_rel_truth={v_rel_truth:+7.2f}  v_rel_rec={v_rel_rec:+7.2f}  Δv={delta:+6.2f} m/s
  chi2={chi2:.3f}
```

---

## 12. Stage F — L2 vector wind retrieval

```python
def stage_f_wind_retrieval(
    config: dict,
    obs_list: list[dict],
    cal_results: dict,
) -> tuple[list[dict], list]:
    """
    Build WindObservation pairs (along-track + cross-track at matched
    frame index) and call retrieve_wind_vectors (M07).
    Execute V10.
    Return (wind_results, checks).
    """
```

### 12.1 Pairing strategy

Observations are organised as:
- Orbit 0 (along-track): frames 0 … n_frames-1
- Orbit 1 (cross-track): frames 0 … n_frames-1
- Orbit 2 (along-track): frames 0 … n_frames-1
- Orbit 3 (cross-track): frames 0 … n_frames-1

For each orbit pair `(p=0, p=1, …)`:
- Along-track observations: orbit `2p`, all frames
- Cross-track observations: orbit `2p+1`, all frames
- Pair each at the same frame index (matched in time within ≈ half an
  orbital period — acceptable for synthetic homogeneous wind fields)

```python
for pair_idx in range(config["n_orbit_pairs"]):
    orbit_at = 2 * pair_idx
    orbit_ct = 2 * pair_idx + 1
    frames_at = [o for o in obs_list if o["orbit_idx"] == orbit_at]
    frames_ct = [o for o in obs_list if o["orbit_idx"] == orbit_ct]

    for obs_at, obs_ct in zip(frames_at, frames_ct):
        wobs_at = WindObservation(
            v_wind_LOS_ms   = obs_at["v_wind_LOS_truth"],
            sigma_v_ms      = obs_at["sigma_v_rel"],
            A_e              = obs_at["A_e"],
            A_n              = obs_at["A_n"],
            tp_lat_deg       = obs_at["tp_lat_deg"],
            tp_lon_deg       = obs_at["tp_lon_deg"],
            tp_alt_km        = obs_at["tp_alt_km"],
            look_mode        = obs_at["look_mode"],
            orbit_number     = obs_at["orbit_idx"],
        )
        wobs_ct = WindObservation(
            v_wind_LOS_ms   = obs_ct["v_wind_LOS_truth"],
            sigma_v_ms      = obs_ct["sigma_v_rel"],
            A_e              = obs_ct["A_e"],
            A_n              = obs_ct["A_n"],
            tp_lat_deg       = obs_ct["tp_lat_deg"],
            tp_lon_deg       = obs_ct["tp_lon_deg"],
            tp_alt_km        = obs_ct["tp_alt_km"],
            look_mode        = obs_ct["look_mode"],
            orbit_number     = obs_ct["orbit_idx"],
        )
        wind_result = retrieve_wind_vectors(wobs_at, wobs_ct)
        # Attach truth for comparison
        wind_result.v_zonal_truth  = obs_at["v_zonal_truth"]
        wind_result.v_merid_truth  = obs_at["v_merid_truth"]
        wind_result.tp_lat_deg     = obs_at["tp_lat_deg"]
        wind_result.tp_lon_deg     = obs_at["tp_lon_deg"]
        wind_results.append(wind_result)
```

**Note on v_wind_LOS vs v_rel in WindObservation.** M07 receives
`v_wind_LOS` (spacecraft and Earth rotation already removed), not `v_rel`.
The conversion is:
```python
v_wind_LOS = v_rel_rec - obs["V_sc_LOS"] - obs["v_earth_LOS"]
```
In Stage F, use the *recovered* `v_rel_rec` from M06, not the truth, to
exercise the full retrieval path. For Stage G verification, compare against
the truth.

**V10 — No ILL_CONDITIONED flags on primary pairs:**
```python
from src.fpi.m07_wind_retrieval_2026_04_06 import WindResultFlags
n_ill = sum(1 for w in wind_results
            if w.quality_flags & WindResultFlags.ILL_CONDITIONED)
v10_pass = n_ill == 0
checks.append(("V10 M07 no ill-conditioned pairs",
               v10_pass,
               f"{n_ill} ill-conditioned of {len(wind_results)}"))
```

---

## 13. Stage G — Science assessment

```python
def stage_g_assessment(
    config: dict,
    obs_list: list[dict],
    wind_results: list,
) -> tuple[dict, list]:
    """
    Compute end-to-end error statistics.
    Execute V11, V12, V13, V14.
    Return (assessment_results, checks).
    """
```

### 13.1 LOS v_rel round-trip (V11)

```python
v_rel_errors = [o["v_rel_rec"] - o["v_rel_truth"] for o in obs_list]
v_rel_bias   = np.mean(v_rel_errors)
v_rel_rms    = np.sqrt(np.mean(np.array(v_rel_errors)**2))
v11_pass = abs(v_rel_bias) < 20.0
```

### 13.2 Vector wind round-trip (V12, V13)

```python
zonal_errors = [w.v_zonal_ms - w.v_zonal_truth for w in wind_results
                if not np.isnan(w.v_zonal_ms)]
merid_errors = [w.v_meridional_ms - w.v_merid_truth for w in wind_results
                if not np.isnan(w.v_meridional_ms)]

zonal_rms = np.sqrt(np.mean(np.array(zonal_errors)**2))
merid_rms = np.sqrt(np.mean(np.array(merid_errors)**2))

rms_limit = 5.0 if config["noiseless"] else 30.0
v12_pass = zonal_rms < rms_limit
v13_pass = merid_rms < rms_limit
```

### 13.3 Uncertainty calibration (V14)

```python
sigma_v_zonals = [w.sigma_v_zonal_ms for w in wind_results
                  if not np.isnan(w.sigma_v_zonal_ms)]
median_sigma = np.median(sigma_v_zonals)
v14_pass = median_sigma <= 2.0 * WIND_BIAS_BUDGET_MS
```

Print assessment:
```
[Stage G] Science assessment:
  n_observations   : {len(obs_list)}
  n_wind_vectors   : {len(wind_results)}

  LOS v_rel round-trip:
    bias  = {v_rel_bias:+.2f} m/s   (limit: |bias| < 20.0 m/s)
    RMS   = {v_rel_rms:.2f} m/s

  Vector wind round-trip:
    v_zonal  RMS = {zonal_rms:.2f} m/s   (limit: {rms_limit:.1f} m/s)
    v_merid  RMS = {merid_rms:.2f} m/s   (limit: {rms_limit:.1f} m/s)

  Uncertainty calibration:
    median sigma_v_zonal = {median_sigma:.2f} m/s   (limit: {2*WIND_BIAS_BUDGET_MS:.1f} m/s)
```

---

## 14. Stages H–M — Figures

### Stage H — Ground track and tangent points

```python
fig_h, ax = plt.subplots(figsize=(14, 6))
```

World map background (use `cartopy` if available; otherwise a simple
Plate Carrée scatter on a lat/lon grid). Plot:
- Spacecraft ground track for each orbit (one line per orbit, colour-coded
  by orbit index).
- Tangent points: along-track as filled circles, cross-track as filled
  triangles. Colour the along-track/cross-track marker by `v_wind_LOS_truth`
  using a diverging colourmap (e.g., `RdBu_r`, centred at 0).
- Legend: orbit indices and look modes.
- Title: `"INT03 — Orbit ground tracks and tangent points (coloured by v_wind_LOS_truth)"`

Save as `outputs/int03_fig_h_ground_track.png`.

### Stage I — Truth wind map vs sampling

```python
fig_i, axes = plt.subplots(1, 2, figsize=(16, 5))
```

- `axes[0]`: NB00 zonal wind map as a lat/lon `pcolormesh`, RdBu_r colourmap,
  centred at 0. Overlay tangent point locations as black dots. Title: `"Truth zonal wind (NB00) and tangent point locations"`.
- `axes[1]`: scatter of `v_wind_LOS_truth` vs `tp_lat_deg`, coloured by
  `look_mode` (blue = along-track, orange = cross-track). Title: `"v_wind_LOS truth vs tangent latitude"`.

Save as `outputs/int03_fig_i_wind_map.png`.

### Stage J — Calibration diagnostics

Four-panel figure reusing the M05 fit plots from INT02 Stage F:

```python
fig_j = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 2, figure=fig_j)
```

- Panel (0,0): Calibration fringe profile with peak annotations (INT02 Fig 2 equivalent).
- Panel (0,1): Tolansky r²–m two-line plot (INT02 Fig 3 equivalent).
- Panel (1,0): M05 Airy fit vs data with residuals.
- Panel (1,1): Text panel — calibration result summary (d, f, α, t_m,
  R_refl, ε_cal, χ²_red, converged, V5/V6/V7 PASS/FAIL).

Save as `outputs/int03_fig_j_calibration.png`.

### Stage K — Per-observation v_rel truth vs recovered

```python
fig_k, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
```

- `axes[0]`: Scatter of `v_rel_truth` (blue) and `v_rel_rec` (orange) vs
  observation index, with error bars on recovered values. Title: `"v_rel: truth vs recovered per observation"`.
- `axes[1]`: Error `v_rel_rec − v_rel_truth` vs observation index. Dashed
  horizontal lines at ±20 m/s (STM bias budget). Shade the ±1σ band using
  `sigma_v_rel` from M06. Colour-code points by `look_mode`.
  Title: `"v_rel retrieval error (target: |bias| < 20 m/s)"`.

Label x-axis with `"Observation (orbit/frame)"` tick labels.

Save as `outputs/int03_fig_k_vrel_roundtrip.png`.

### Stage L — M07 wind vector map

```python
fig_l, axes = plt.subplots(1, 2, figsize=(16, 5))
```

- `axes[0]`: `v_zonal_truth` vs `tp_lat_deg` (blue dashed) and
  `v_zonal_rec` (orange circles with error bars). Title: `"v_zonal: truth vs recovered"`.
- `axes[1]`: Same for `v_meridional`. Title: `"v_merid: truth vs recovered"`.

Add diagonal unity line `y=x` to each panel. Annotate RMS error.

Save as `outputs/int03_fig_l_wind_vectors.png`.

### Stage M — Bias and RMS summary

```python
fig_m, axes = plt.subplots(1, 3, figsize=(14, 4))
```

- `axes[0]`: Histogram of `v_rel_rec − v_rel_truth` for all observations.
  Vertical dashed lines at ±20 m/s. Annotate mean and RMS. Gaussian fit
  overlay using `scipy.stats.norm.fit`. Title: `"LOS v_rel error distribution"`.
- `axes[1]`: Histogram of `v_zonal_rec − v_zonal_truth`. Annotate RMS.
  Title: `"v_zonal error distribution"`.
- `axes[2]`: Box plots of `|error|` for v_rel, v_zonal, v_merid. Mark
  the STM budget limits as horizontal red lines. Title: `"Error budget"`.

Save as `outputs/int03_fig_m_error_budget.png`.

---

## 15. Stage N — Summary report

```python
def stage_n_summary(all_checks: list, config: dict) -> int:
    """
    Print PASS/FAIL summary for all 14 verification checks.
    Return 0 if all pass, 1 if any fail.
    """
```

```
══════════════════════════════════════════════════════════════════
  INT03 Verification Summary
══════════════════════════════════════════════════════════════════
  V1   NB01 state table non-empty              PASS   {n} rows ≥ {4*n_frames}
  V2   Tangent altitudes 200–350 km            PASS   all {n_obs} in range
  V3   LOS sensitivity matrix conditioned      PASS   max cond = {max_cond:.1f} < 100
  V4   v_rel_truth physically plausible        PASS   |v_rel| < 1000 m/s
  V5   Tolansky d recovery                     PASS   {d_error:.3f} µm < 1.0 µm
  V6   M05 chi2_red in [0.5, 3.0]              PASS   {chi2:.3f}
  V7   M05 t_m within 1 nm of Tolansky prior   PASS   {t_err:.4f} nm
  V8   M06 chi2_red in bounds (< 5% failures)  PASS   {n_fail}/{n_obs} failed
  V9   Metadata TP matches NB02 geometry       PASS   all within 1e-10°
  V10  M07 no ill-conditioned pairs            PASS   0 ill-conditioned
  V11  v_rel LOS bias < 20 m/s                 PASS   bias={bias:+.2f} m/s
  V12  v_zonal RMS < {rms_lim:.0f} m/s                  PASS   RMS={zrms:.2f} m/s
  V13  v_merid RMS < {rms_lim:.0f} m/s                  PASS   RMS={mrms:.2f} m/s
  V14  median sigma_v_zonal ≤ 2× STM budget   PASS   {sig:.2f} m/s ≤ {budget:.2f} m/s
──────────────────────────────────────────────────────────────────
  Total: 14 checks.  PASS: {n_pass}.  FAIL: {n_fail}.
══════════════════════════════════════════════════════════════════
```

If any check fails, print in red (ANSI escape or plain text on Windows):
`"INT03 FAILED — see checks above. Run INT02 first to confirm FPI chain is healthy."`

Return `0 if n_fail == 0 else 1`.

---

## 16. Main execution

```python
def main():
    parser = argparse.ArgumentParser(
        description="INT03 — full end-to-end pipeline integration"
    )
    parser.add_argument("--quick",      action="store_true",
                        help="1 orbit pair, 4 frames — fast development iteration")
    parser.add_argument("--noiseless",  action="store_true",
                        help="Skip Poisson noise; tighten V12/V13 to 5 m/s")
    parser.add_argument("--full-wind",  action="store_true",
                        help="Use both zonal and meridional truth wind components")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_checks = []

    config = stage_a_setup(args.quick, args.noiseless, args.full_wind)

    geo_r,   v1_3  = stage_b_geometry(config)
    all_checks    += v1_3

    truth_r, v4    = stage_c_truth_winds(config, geo_r)
    all_checks    += v4

    cal_r,   v5_7  = stage_d_calibration(config)
    all_checks    += v5_7

    obs_list       = geo_r["obs_list"]   # mutated in-place by stage_c
    obs_list, v8_9 = stage_e_fpi_chain(config, obs_list, cal_r)
    all_checks    += v8_9

    wind_r,  v10   = stage_f_wind_retrieval(config, obs_list, cal_r)
    all_checks    += v10

    assess_r, v11_14 = stage_g_assessment(config, obs_list, wind_r)
    all_checks      += v11_14

    stage_h_figure_ground_track(config, geo_r, obs_list)
    stage_i_figure_wind_map(config, truth_r, obs_list)
    stage_j_figure_calibration(config, cal_r)
    stage_k_figure_vrel_roundtrip(config, obs_list)
    stage_l_figure_wind_vectors(config, wind_r)
    stage_m_figure_error_budget(config, obs_list, wind_r)

    return_code = stage_n_summary(all_checks, config)

    plt.show(block=True)
    sys.exit(return_code)


if __name__ == "__main__":
    main()
```

---

## 17. Expected console output

```
══════════════════════════════════════════════════════════════════
  WindCube INT03 — Full End-to-End Pipeline Integration
══════════════════════════════════════════════════════════════════
Mode : FULL  |  SNR=5.0  |  ZONAL-ONLY
Orbit pairs : 2
Frames/orbit: 8
...
[Stage B] Geometry pipeline:
  n_observations : 32
  TP altitude range: 247.3 – 252.1 km
  Max condition number: 7.4
  V1/V2/V3: PASS PASS PASS

[Stage C] Truth wind sampling:
  v_rel_truth: −487.3 to +521.6 m/s  (mean ± std: 12.4 ± 218.3)
  V4: PASS

[Stage D] Calibration pipeline:
  Tolansky d = 20.106012 ± 0.432 µm   V5: PASS
  M05 t_m    = 20.105997 mm  (Δ = 0.0150 nm)  V7: PASS
  M05 chi2   = 1.24  V6: PASS

[Stage E] FPI chain — observation 1/32:
  orbit=0 frame=0 look=along_track
  tp=(34.21°, −112.47°, 250.1 km)
  v_rel_truth=+312.47  v_rel_rec=+309.83  Δv=−2.64 m/s  chi2=1.34
...
  V8 (M06 chi2 in bounds): PASS  0/32 failed
  V9 (metadata TP):        PASS  all within 1e-10°

[Stage F] Wind retrieval:
  16 wind vectors retrieved.  V10: PASS  (0 ill-conditioned)

[Stage G] Science assessment:
  v_rel  bias = +1.23 m/s   RMS = 9.87 m/s   V11: PASS
  v_zonal RMS = 11.42 m/s   (limit: 30.0 m/s)  V12: PASS
  v_merid RMS = 10.89 m/s   (limit: 30.0 m/s)  V13: PASS
  median sigma_v_zonal = 12.34 m/s ≤ 19.60 m/s  V14: PASS

══════════════════════════════════════════════════════════════════
  INT03 Verification Summary
...
  Total: 14 checks.  PASS: 14.  FAIL: 0.
══════════════════════════════════════════════════════════════════
```

---

## 18. Figures produced

| File | Contents | Verification supported |
|------|----------|----------------------|
| `outputs/int03_fig_h_ground_track.png` | Ground track + tangent points, v_wind_LOS coloured | V2 visual |
| `outputs/int03_fig_i_wind_map.png` | NB00 truth map + tangent point overlay | V4 visual |
| `outputs/int03_fig_j_calibration.png` | 4-panel calibration diagnostic | V5/V6/V7 visual |
| `outputs/int03_fig_k_vrel_roundtrip.png` | v_rel truth vs recovered, error vs obs index | V11 visual |
| `outputs/int03_fig_l_wind_vectors.png` | v_zonal/v_merid truth vs recovered | V12/V13 visual |
| `outputs/int03_fig_m_error_budget.png` | Error histograms and box plots | All V visual summary |

All figures: `dpi=150`, `bbox_inches='tight'`.

---

## 19. Verification check pass criteria

| Check | Criterion | Physical basis |
|-------|-----------|----------------|
| V1 | `len(state_table) ≥ 4 × n_frames` | NB01 propagated enough epochs |
| V2 | 200 ≤ tp_alt ≤ 350 km | OI 630 nm emission layer bounds |
| V3 | condition < 100 for all pairs | M07 2×2 system must be well-posed |
| V4 | \|v_rel\| < 1000 m/s | Subsonic atmospheric winds + LEO S/C |
| V5 | \|d − ETALON_GAP_M\| < 1 µm | Tolansky precision for FSR disambiguation |
| V6 | 0.5 < χ²_red < 3.0 | Well-modelled calibration fringe |
| V7 | \|t_m − d_Tolansky\| < 1 nm | M05 convergence to Tolansky prior |
| V8 | < 5% of M06 fits outside χ²_red [0.5, 3.0] | Robust fringe fitting at SNR=5 |
| V9 | \|tp_lat_err\| < 1e-10° | Metadata copy fidelity |
| V10 | No ILL_CONDITIONED flags | Geometry pairs well-posed |
| V11 | \|LOS bias\| < 20 m/s | STM wind bias budget |
| V12 | v_zonal RMS < 30 m/s (5 m/s noiseless) | STM 2σ science requirement |
| V13 | v_merid RMS < 30 m/s (5 m/s noiseless) | STM 2σ science requirement |
| V14 | median σ_v_zonal ≤ 2 × WIND_BIAS_BUDGET_MS | Uncertainty propagation calibration |

---

## 20. File locations

```
soc_sewell/
├── docs/specs/
│   └── S18_int03_end_to_end_pipeline_2026-04-10.md
├── src/integration/
│   ├── __init__.py           (exists from INT02)
│   └── int03_end_to_end_2026_04_10.py
└── outputs/
    └── int03_fig_*.png       (created at runtime, not committed)
```

---

## 21. Instructions for Claude Code

1. Read this entire spec plus S07, S16, S17, S19, and `CLAUDE.md` before
   writing any code.

2. Confirm all prior tests and INT02 pass:
   ```bash
   pytest tests/ -v --tb=no -q
   python src/integration/int02_fpi_chain_2026_04_07.py --quick
   ```

3. Before writing any NB01/NB02 import lines, inspect the actual filenames
   and public API in `src/geometry/`:
   ```bash
   ls src/geometry/
   grep -n "^def \|^class " src/geometry/nb01*.py
   grep -n "^def \|^class " src/geometry/nb02*.py
   ```
   Use the confirmed function names throughout. The import stubs in Section 6
   are intent, not guaranteed names.

4. Before writing Stage D, confirm the actual attribute names on
   `TwoLineResult` and `CalibrationResult` by reading the implementation
   files. Do not assume spec attribute names are identical to implementation.

5. Stage F's `WindObservation` field names must be confirmed against
   `src/fpi/m07_wind_retrieval_2026_04_06.py` before use.

6. The `v_wind_LOS` vs `v_rel` sign convention in Stage F is critical.
   Before writing the conversion, re-read Section 12.1 of this spec
   and the NB02d sign convention in S07. Incorrect sign produces bias
   of order `|V_sc_LOS| ≈ 7500 m/s` — immediately obvious in V11.

7. Implement each stage as a standalone function. Do not inline stage
   logic into `main()`. This makes debugging and re-running individual
   stages easier during integration testing.

8. The `--quick` flag should produce a valid, complete run in < 60 seconds.
   All 14 checks must still pass in quick mode.

9. All `plt.savefig()` calls use `dpi=150, bbox_inches='tight'`.
   All figures use `plt.show(block=False)` during the run; the final
   `plt.show(block=True)` in `main()` holds all windows open.

10. If `cartopy` is not available, Stage H should fall back to a simple
    Plate Carrée scatter plot on axes with `xlim=(-180, 180)`,
    `ylim=(-90, 90)`. Do not fail if cartopy is absent.

11. After implementing, run the full integration:
    ```bash
    python src/integration/int03_end_to_end_2026_04_10.py --quick
    ```
    All 14 checks must print PASS. Then run full mode:
    ```bash
    python src/integration/int03_end_to_end_2026_04_10.py
    ```
    Then run the full test suite to confirm no regressions:
    ```bash
    pytest tests/ -v --tb=no -q
    ```

12. Commit message:
    ```
    feat(int03): implement full end-to-end pipeline integration, 14/14 checks pass
    Implements: S18_int03_end_to_end_pipeline_2026-04-10.md
    ```

Module docstring header:
```python
"""
INT03 — Full end-to-end pipeline integration: orbit → fringe → wind.

Spec:        docs/specs/S18_int03_end_to_end_pipeline_2026-04-10.md
Spec date:   2026-04-10
Generated:   <today>
Tool:        Claude Code
Last tested: <today>  (14/14 checks pass)
Depends on:  src.constants, src.geometry (NB00–NB02),
             src.fpi (M01–M07, Tolansky), src.metadata (P01)
"""
```
