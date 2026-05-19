# G01 — Synthetic Metadata Generator Specification
## WindCube SOC Pipeline — v15
**Spec ID:** G01  
**Spec file:** `specs/G01_synthetic_metadata_generator_2026-05-14_v15.md`  
**Previous spec:** `specs/G01_synthetic_metadata_generator_2026-05-14.md` (v14)  
**Script:** `src/processing/GEN01_synthesize_mission_dataset_2026_05_13.py`  
**Date:** 2026-05-14  
**Status:** Authoritative — v15  

**Depends on:**
- All v14 dependencies (unchanged)
- H07-ADD-01 (`H07_addendum_coverage_criteria_2026-05-14.md`) — coverage
  analysis methodology

---

## Revision history

| Version | Date | Summary |
|---------|------|---------|
| 1–14 | see archive | — |
| **15** | **2026-05-14** | **Add end-of-run coverage diagnostics (§9) and standalone `scripts/coverage_map.py` (§10). No changes to image synthesis, metadata, or CSV format. Companion: H07-ADD-01.** |

---

## 1–8. Unchanged from v14

All sections 1–8 (user interface, wind maps, orbit propagation, metadata,
CSV format, binary synthesis, verification checks C1–C28) are unchanged
from v14.

---

## 9. End-of-run coverage diagnostics (NEW in v15)

After the main synthesis loop completes and the CSV is written, GEN01 runs
a coverage analysis and prints a diagnostics report to stdout. The report
is also written to a sidecar text file alongside the CSV.

### 9.1 What is computed

Using the science-frame rows of the output CSV (obs_type == "science"),
GEN01 computes the following for each geographic bin (default 5°×5°):

**Per-bin quantities:**
- `n_along_track`: number of along_track science frames with tangent point
  in this bin
- `n_cross_track`: number of cross_track science frames with tangent point
  in this bin
- `n_total`: n_along_track + n_cross_track
- `has_both_modes`: True if n_along_track ≥ 1 AND n_cross_track ≥ 1
- `predicted_gdop_ok`: True if has_both_modes (simplified — actual GDOP
  depends on direction cosines, but mode diversity is the dominant factor)

**Global summary quantities:**
- `n_bins_sampled`: bins with n_total ≥ 1
- `n_bins_mixed`: bins with has_both_modes (predicted good H07 solutions)
- `pct_mixed`: n_bins_mixed / n_bins_sampled × 100
- `n_bins_at_only`: bins with only along_track frames
- `n_bins_ct_only`: bins with only cross_track frames

### 9.2 Coverage report format

Print this block to stdout after the CSV is written, and write it to
`<output_stem>_coverage_report.txt`:

```
================================================================
GEN01 Coverage Diagnostic Report
================================================================
Dataset         : <output_stem>
Duration        : {n_days:.2f} days  ({n_science_frames} science frames)
Bin size        : {dlat}° × {dlon}°
Orbit period    : ~95 min  ({passes_per_day:.1f} passes/day)
Ground track Δλ : {delta_lon:.2f}° between successive passes
----------------------------------------------------------------
Geographic coverage (5°×5° bins):
  Bins sampled (≥1 frame)  : {n_bins_sampled}
  Along-track only         : {n_bins_at_only}  ({pct_at:.1f}%)
  Cross-track only         : {n_bins_ct_only}  ({pct_ct:.1f}%)
  Mixed AT+CT (good H07)   : {n_bins_mixed}  ({pct_mixed:.1f}%)
----------------------------------------------------------------
H07 wind solution prediction:
  Predicted good solutions : {pct_mixed:.1f}%
  Expected for {n_days:.1f}-day dataset : {expected_pct:.1f}%
  (Expected from orbital mechanics: ~{days_for_80:.1f} days for >80%)
----------------------------------------------------------------
Coverage by latitude band:
  60°S–30°S  : {pct_south_hi:.0f}% mixed
  30°S–0°    : {pct_south_lo:.0f}% mixed
  0°–30°N    : {pct_north_lo:.0f}% mixed
  30°N–60°N  : {pct_north_hi:.0f}% mixed
  (Polar regions >60° not shown — sparse airglow coverage)
----------------------------------------------------------------
Recommendation:
  {recommendation_string}
================================================================
```

Where `recommendation_string` is generated as:
- If pct_mixed < 10%: "Extend to ≥5 days for useful H07 wind coverage."
- If pct_mixed < 50%: "Consider extending to ≥5 days for >75% H07 coverage."
- If pct_mixed < 80%: "Coverage adequate for regional analysis. Extend for global."
- If pct_mixed ≥ 80%: "Coverage sufficient for global H07 wind map validation."

### 9.3 Expected good-bin fraction formula

The expected fraction of mixed-mode bins for an n-day simulation:

```python
def expected_mixed_fraction(n_days, dlat=5.0, dlon=5.0):
    """
    Analytical estimate of mixed AT+CT bin fraction.
    Based on orbital mechanics: 23.75 deg longitude spacing per pass,
    15.2 passes/day, SSO dusk-dawn orbit.

    This is an approximation — actual coverage depends on the exact
    orbit geometry and whether AT/CT passes land in the same bin.
    """
    DELTA_LON_DEG = 23.75      # longitude spacing per pass
    passes_per_day = 15.2

    # Fraction of lon bins covered by AT in n_days:
    # Each day covers 15.2 passes × (dlon / 360) of all lon bins
    # with ascending + descending doubles the rate
    lon_coverage_at = min(1.0, n_days * passes_per_day * 2 * dlon / 360.0)
    lon_coverage_ct = min(1.0, n_days * passes_per_day * 2 * dlon / 360.0)

    # Mixed: probability both AT and CT have visited the same bin
    # Approximation: independent sampling
    mixed_frac = lon_coverage_at * lon_coverage_ct

    # Latitude weighting: science band ~±60 deg, fuller coverage at mid-lat
    # Simple approximation: uniform within science band
    return mixed_frac
```

Use this in the report as `expected_pct = expected_mixed_fraction(n_days) × 100`.

### 9.4 Coverage sidecar file

Write the coverage report text to:
`<output_stem>_coverage_report.txt`

alongside the CSV file. This file is the input for `coverage_map.py` (§10).

Also append to the existing README sidecar a one-line summary:
```
Coverage (5x5 deg, mixed AT+CT): {pct_mixed:.1f}%  ({n_bins_mixed}/{n_bins_sampled} bins)
```

---

## 10. Standalone coverage map script (NEW in v15)

Create `scripts/coverage_map.py` — a standalone visualization tool that
reads a GEN01 CSV and produces global map figures.

### 10.1 Command-line interface

```
python scripts/coverage_map.py <gen01_csv_path> [options]

Required:
  gen01_csv_path      Path to GEN01 output CSV (v14 or later format)

Optional:
  --dlat              Bin latitude size, degrees (default: 5.0)
  --dlon              Bin longitude size, degrees (default: 5.0)
  --output-dir        Directory for saved figures (default: same as CSV)
  --save              Save figures to PNG instead of displaying interactively
  --dpi               Figure DPI for saved files (default: 150)
```

### 10.2 Figures produced

All figures use an equirectangular projection (no cartopy required).
Draw simple coastlines using matplotlib's built-in path data if available,
or skip coastlines gracefully.

Use a consistent colour scheme across all panels:
- Along-track only: blue (#0057C2 — NCAR brand secondary blue)
- Cross-track only: orange (#FF7F0E)
- Mixed AT+CT: green (#2CA02C)
- No coverage: light grey (#EEEEEE)
- Land outlines (if available): dark grey (#444444)

**Figure 1 — Coverage map (main figure):**
`<stem>_coverage_map.png`

Single world map panel. Each 5°×5° bin coloured by:
- Grey: not sampled
- Blue: along_track only
- Orange: cross_track only
- Green: mixed AT+CT (predicted good H07 solution)

Title: f"WindCube Coverage Map — {n_days:.1f}-day simulation"
Subtitle: f"{pct_mixed:.0f}% mixed AT+CT  |  {n_science_frames} science frames  |  5°×5° bins"
Colorbar / legend: show the four categories with frame counts.

**Figure 2 — Pass count map:**
`<stem>_pass_count_map.png`

Two side-by-side panels:
  Left: n_along_track per bin (colormap: Blues, 0 to max)
  Right: n_cross_track per bin (colormap: Oranges, 0 to max)

Title: f"Along-track and Cross-track Pass Counts — {n_days:.1f} days"

**Figure 3 — Coverage vs. accumulation days (forecast curve):**
`<stem>_coverage_forecast.png`

Line plot showing:
- X axis: simulation duration in days (0 to 14)
- Y axis: predicted mixed-bin fraction (%)
- Three lines:
    2.5°×2.5° bins (dotted)
    5°×5° bins (solid, thicker — highlighted as default)
    10°×10° bins (dashed)
- Vertical dashed line at the actual n_days of this simulation
- Horizontal dashed lines at 50%, 80%, 90%
- Markers on the 5°×5° line at 1, 3, 5, 7, 14 days

Title: "Predicted H07 Coverage vs. Simulation Duration"
X label: "Simulation duration (days)"
Y label: "Predicted mixed AT+CT bins (%)"
Add text annotation at the actual n_days point:
  f"This run: {n_days:.1f} d → {pct_mixed:.0f}%"

**Figure 4 — Ground track map:**
`<stem>_ground_track_map.png`

Scatter plot of all tangent point locations from the CSV:
- Along-track points: blue dots, alpha=0.3, size=1
- Cross-track points: orange dots, alpha=0.3, size=1

Title: f"Tangent Point Coverage — {n_days:.1f}-day simulation"
This figure shows the raw sampling density before binning.

### 10.3 Summary printout

After generating figures, print:

```
Coverage Map Summary
====================
Input CSV    : <csv_path>
Science frames: {n_science_frames}  ({n_at} AT, {n_ct} CT)
Bin size     : {dlat}° × {dlon}°
-----------------------------------------
Bins sampled : {n_bins_sampled}
  AT only    : {n_bins_at_only}  ({pct:.0f}%)
  CT only    : {n_bins_ct_only}  ({pct:.0f}%)
  Mixed      : {n_bins_mixed}  ({pct_mixed:.0f}%)
-----------------------------------------
Figures written to: {output_dir}
```

---

## 11. Instructions for Claude Code

Read this entire spec (v15) AND v14 before touching any code.
Read H07-ADD-01 for context on why coverage matters.

**Changes required:**

### 11.1 GEN01 script — add coverage diagnostic

In `src/processing/GEN01_synthesize_mission_dataset_2026_05_13.py`:

After the CSV is written (after `df.to_csv(csv_path, ...)`), add a call to
a new function `_run_coverage_diagnostic(df, csv_path, h_target_km)` that:

1. Filters to science rows: `sci = df[df['obs_type'] == 'science']`
2. Assigns each row to a 5°×5° bin using `tp_lat_deg` and `tp_lon_deg`
3. Computes per-bin AT/CT counts
4. Computes global summary statistics
5. Prints the coverage report (§9.2)
6. Writes `<stem>_coverage_report.txt`
7. Appends the one-line summary to the README sidecar

Use `expected_mixed_fraction()` (§9.3) to fill in the "Expected" line.
Use the actual `n_days` from the GEN01 prompt for `days_for_80` calculation:
  `days_for_80 = (dlon / (15.2 * 2 * dlon / 360)) ** 2`
  (days until expected_mixed_fraction ≥ 0.80, solved analytically)
  Simplified: `days_for_80 = 360.0 / (15.2 * 2 * dlon)` ≈ 2.4 days for dlon=5
  Wait — this gives ~2.4 but our empirical result is ~5 days. Use the
  empirical formula: days_for_80 ≈ 360 / (15.2 * dlon) = 4.7 days for
  dlon=5. Use this value.

### 11.2 Create scripts/coverage_map.py

Implement the full script as specified in §10. Key implementation notes:

- Read the CSV with pandas. Required columns: obs_type, obs_mode,
  tp_lat_deg, tp_lon_deg. All present in v14 CSV format.
- For the ground track figure (Figure 4), use alpha=0.3 and markersize=1
  for legibility with thousands of points.
- For Figure 3, use expected_mixed_fraction() with n_days from 0 to 14
  in 0.1-day steps to make smooth curves.
- coastlines: try `from matplotlib.patches import PathPatch` with
  matplotlib's built-in world map data. If unavailable, skip silently.
- All four figures should work without cartopy.

### 11.3 Verification checks (add to existing C25–C28 suite)

**C29 — Coverage report written:**
After running a short synthetic dataset (0.1 days), confirm:
- `<stem>_coverage_report.txt` file exists
- File contains "Coverage Diagnostic Report" header
- `pct_mixed` line is present and is a float between 0 and 100

**C30 — coverage_map.py runs on existing CSV:**
Run:
  `python scripts/coverage_map.py <path_to_GEN01_csv> --save`
Confirm all 4 PNG files are written without error.

### 11.4 Smoke test

Run GEN01 with:
- Duration: 0.1 days
- Wind map: option 1, v_zonal=0, v_merid=0
- All other prompts: defaults

Confirm coverage report prints at end of run. Note the pct_mixed value
(expected ~0–2% for 0.1-day run).

Then run coverage_map.py on the existing 1-day CSV:
  `python scripts/coverage_map.py <GEN01_20270101_001_0d_uniform_seed0042.csv> --save`

Confirm 4 PNG files written. Report any errors.

### 11.5 Commit

```
feat(gen01): v15 — end-of-run coverage diagnostics and coverage_map.py
Implements: G01_synthetic_metadata_generator_2026-05-14_v15.md (v15)
Companion: H07_addendum_coverage_criteria_2026-05-14.md (H07-ADD-01)
- GEN01: _run_coverage_diagnostic() after CSV write
- scripts/coverage_map.py: 4-panel global coverage visualization
- C29, C30 verification checks added
```

---

## 12. Constants used in coverage analysis

All from `windcube/constants.py` (already defined):
- `OI_EMISSION_ALT_KM = 250.0`

New constants (add to `windcube/constants.py`):
```python
ORBIT_PERIOD_MIN     = 95.0    # approximate WindCube orbital period [min]
PASSES_PER_DAY       = 15.2    # approximate science passes per day
GROUND_TRACK_DELTA_LON_DEG = 23.75  # longitude spacing between passes [deg]
```

---

*End of G01 Specification v15 — 2026-05-14*
