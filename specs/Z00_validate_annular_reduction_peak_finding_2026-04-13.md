# Z00 — Validate Annular Reduction and Peak Finding

**Spec ID:** Z00
**Spec file:** `docs/specs/Z00_validate_annular_reduction_peak_finding_2026-04-13.md`
**Project:** WindCube FPI Pipeline
**Institution:** NCAR / High Altitude Observatory (HAO)
**Status:** Specification — ready for implementation in VS Code
**Depends on:** S12 (M03 `m03_annular_reduction`)
**Produces:**
  - Seven diagnostic figures (a)–(g), displayed interactively and saved to PNG
  - `{cal_roi_stem}_fringe_peaks.npy` — structured peak table, consumed later by Z04
    (Tolansky analysis validation)
**References:**
  - S12 — M03 Fringe Centre Finding and Annular Reduction Specification (2026-04-06)
  - Melissinos (1966) *Experiments in Modern Physics*, 1st ed., §7.3:
    Table 7.5 "Radii of Fabry-Perot Pattern for Single Line" and
    Fig. 7.28 "Ring order p vs. R² (mm²)"
  - Niciejewski et al. (1992) SPIE 1745 — r² reduction rationale
**Last updated:** 2026-04-13

---

## 1. Purpose

This script provides an end-to-end interactive validation of M03's four sequential
responsibilities using **real saved arrays** (calibration ROI + dark ROI) previously
extracted from a WindCube FlatSat or flight binary image:

0. Dark subtraction (`make_master_dark` + `subtract_dark`)
1. Sub-pixel centre finding (two-pass azimuthal variance minimisation)
2. Annular reduction (r²-binned mean intensity and SEM)
3. Peak finding and Gaussian fitting (`_find_and_fit_peaks` via `annular_reduce`)

All M03 functions are called via the production API defined in S12 —
no special test-only pathways.  The script is entirely self-contained,
requires no command-line arguments, and prompts the user for file locations
at runtime.

---

## 2. Script location and invocation

```
soc_sewell/
└── validation/
    └── z00_validate_annular_reduction_2026-04-13.py
```

```bash
# From the repo root:
python validation/z00_validate_annular_reduction_2026-04-13.py
```

No arguments.  All interactive prompts appear in the terminal.

---

## 3. Required inputs

The script asks for **two `.npy` files** via interactive file-selection dialogs
(see Section 5).  Both must have been saved previously — for example, by the
binary-image metadata pipeline (S19/P01) or by any earlier processing step that
extracted and saved the ROI sub-arrays.

| Prompt | Expected array | Shape | dtype |
|--------|---------------|-------|-------|
| "Select the **calibration ROI** `.npy` file" | `cal_roi` | `(N, N)` | any numeric; uint16 typical |
| "Select the **dark ROI** `.npy` file" | `dark_roi` | `(N, N)` same shape as cal_roi | any numeric; uint16 typical |

`N` is typically 256 for a full 2×2-binned FlatSat frame, or a smaller sub-array
if the ROI was cropped before saving.  The script infers `r_max_px` and the
initial centre seed from the array size (see Section 6.2).

### 3.1 Fringe-type detection

The script asks the user to declare the fringe type after the files are loaded:

```
Fringe type: [1] Neon calibration (2-line, 640.2 + 638.3 nm)
             [2] Airglow science  (1-line, 630.0 nm)
Enter 1 or 2:
```

This choice controls:
- The amplitude threshold used to separate the two neon peak families (Section 9.1)
- Column labels in Figure (f) and Figure (g) (Sections 10.6, 10.7)
- Whether a single-family or two-family Melissinos table is produced (Section 10.7)

---

## 4. Instrument / reduction parameters

All numerical defaults in the table below are **fixed** for FlatSat / flight arrays.
They may be overridden at the top of the script via clearly labelled constants.

| Parameter | Default | Source |
|-----------|---------|--------|
| `R_MAX_PX` | 110.0 | FlatSat confirmed (S12 §14) |
| `N_BINS` | 150 | S12 §14 |
| `N_SUBPIXELS` | 1 | S12 §14 (real data) |
| `PEAK_DISTANCE` | 5 | S12 §10.3 |
| `PEAK_PROMINENCE` | 100.0 ADU | S12 §10.3 |
| `PEAK_FIT_HALF_WINDOW` | 8 | S12 §10.3 |
| `MIN_PEAK_SEP_PX` | 3.0 | S12 §10.3 |
| `VAR_SEARCH_PX` | 15.0 | S12 §8 |
| `NEON_A_WAVELENGTH_NM` | 640.2248 | Burns et al. 1950 |
| `NEON_B_WAVELENGTH_NM` | 638.2991 | Burns et al. 1950 |
| `OI_WAVELENGTH_NM` | 630.0304 | rest wavelength (air) |
| `NEON_AMPLITUDE_SPLIT_ADU` | 1000.0 | FlatSat threshold (S13) |

---

## 5. File selection

Use `tkinter.filedialog.askopenfilename` for GUI file selection.  If `tkinter`
is unavailable (headless environment), fall back to a `input()` prompt for
a typed path.  The selection code pattern:

```python
import tkinter as tk
from tkinter import filedialog

def select_npy_file(prompt: str) -> str:
    """Open a file-selection dialog and return the chosen path."""
    root = tk.Tk()
    root.withdraw()          # hide the empty root window
    root.attributes('-topmost', True)
    print(f"\n{prompt}")
    path = filedialog.askopenfilename(
        title=prompt,
        filetypes=[("NumPy arrays", "*.npy"), ("All files", "*.*")],
    )
    root.destroy()
    if not path:
        raise FileNotFoundError("No file selected — aborting.")
    return path
```

Call sequence:

```python
cal_roi_path  = select_npy_file("Select the CALIBRATION ROI .npy file")
dark_roi_path = select_npy_file("Select the DARK ROI .npy file")
```

The **stem** of `cal_roi_path` (filename without extension) is used as the
base name for all saved output files:

```python
import pathlib
cal_roi_stem = pathlib.Path(cal_roi_path).stem
output_dir   = pathlib.Path(cal_roi_path).parent
```

---

## 6. Processing chain

Execute the following steps **in strict order**, importing only from the
production M03 module.

### 6.1 Import

```python
from fpi.m03_annular_reduction_2026_04_05 import (
    make_master_dark,
    subtract_dark,
    reduce_calibration_frame,
    reduce_science_frame,
    FringeProfile,
    PeakFit,
    QualityFlags,
)
```

Adjust the module filename/import path to match the current dated file in
`src/fpi/`.

### 6.2 Initial seed and r_max inference

```python
cal_roi  = np.load(cal_roi_path)
dark_roi = np.load(dark_roi_path)

N = cal_roi.shape[0]        # assume square ROI
cx_seed = (N - 1) / 2.0    # geometric centre of ROI
cy_seed = (N - 1) / 2.0

# r_max: use R_MAX_PX constant; warn if ROI is smaller than expected
if N < 2 * R_MAX_PX:
    print(f"WARNING: ROI size {N}×{N} is smaller than 2×r_max "
          f"({2*R_MAX_PX:.0f} px). Reducing r_max to {N/2 - 2:.0f} px.")
    r_max = N / 2.0 - 2.0
else:
    r_max = R_MAX_PX
```

### 6.3 Master dark

```python
master_dark = make_master_dark([dark_roi])
```

`dark_roi` is treated as a single-frame master.  If multiple dark frames
are available as a list, pass them all; `make_master_dark` handles 1–N
frames via median combination (S12 §3.2).

### 6.4 Reduction

```python
fp = reduce_calibration_frame(
    image       = cal_roi,
    master_dark = master_dark,
    cx_human    = cx_seed,
    cy_human    = cy_seed,
    r_max_px    = r_max,
    n_bins      = N_BINS,
    n_subpixels = N_SUBPIXELS,
    peak_distance         = PEAK_DISTANCE,
    peak_prominence       = PEAK_PROMINENCE,
    peak_fit_half_window  = PEAK_FIT_HALF_WINDOW,
    min_peak_sep_px       = MIN_PEAK_SEP_PX,
    var_search_px         = VAR_SEARCH_PX,
)
```

For a science (airglow) frame, use `reduce_science_frame` instead (fringe
type 2), providing `cx` and `cy` from a paired calibration frame or the
default seed.

### 6.5 Terminal summary

Print the following immediately after `reduce_calibration_frame` returns:

```
=== Z00 ANNULAR REDUCTION SUMMARY ===
  Input ROI     : {cal_roi_path}  shape={cal_roi.shape}
  Dark ROI      : {dark_roi_path}
  Dark subtracted: {fp.dark_subtracted}  ({fp.dark_n_frames} frame(s))
  Centre (fine) : cx={fp.cx:.4f} px,  cy={fp.cy:.4f} px
  Centre sigma  : σ_cx={fp.sigma_cx:.4f} px,  σ_cy={fp.sigma_cy:.4f} px
  Seed source   : {fp.seed_source}
  Quality flags : 0x{fp.quality_flags:02X}
  Cost at min   : {fp.cost_at_min:.4f}
  Grid seed cx  : {fp.grid_cx:.4f},  grid seed cy={fp.grid_cy:.4f}
  Peaks found   : {len(fp.peak_fits)}  ({sum(p.fit_ok for p in fp.peak_fits)} fit_ok)
  Profile bins  : {fp.n_bins}  (r_max={fp.r_max_px:.1f} px)
  Sparse bins   : {fp.sparse_bins}
=====================================
```

If `fp.quality_flags != 0`, print a decoded flag list below the summary.

---

## 7. Peak-family separation (neon calibration only)

For fringe type 1 (neon), separate the `fp.peak_fits` list into the
**Neon-A** (640.2248 nm, strong) and **Neon-B** (638.2991 nm, weak) families
using the amplitude threshold defined in Section 4:

```python
if fringe_type == 1:
    peaks_A = [p for p in fp.peak_fits if p.amplitude_adu >= NEON_AMPLITUDE_SPLIT_ADU]
    peaks_B = [p for p in fp.peak_fits if p.amplitude_adu <  NEON_AMPLITUDE_SPLIT_ADU]

    # Sort each family by ascending r_fit_px and assign 1-based ring indices
    peaks_A = sorted(peaks_A, key=lambda p: p.r_fit_px)
    peaks_B = sorted(peaks_B, key=lambda p: p.r_fit_px)

    n_A = len(peaks_A)
    n_B = len(peaks_B)
    print(f"  Neon-A peaks: {n_A}  (≥ {NEON_AMPLITUDE_SPLIT_ADU:.0f} ADU)")
    print(f"  Neon-B peaks: {n_B}  (< {NEON_AMPLITUDE_SPLIT_ADU:.0f} ADU)")
```

For fringe type 2 (airglow), there is a single family:

```python
else:
    peaks_OI = sorted(fp.peak_fits, key=lambda p: p.r_fit_px)
```

---

## 8. Derived quantities needed for tables and figures

For **each peak** in each family, compute:

| Symbol | Definition | Units |
|--------|-----------|-------|
| `r_fit_px` | Gaussian centroid radius | px |
| `r2_fit_px2` | `r_fit_px ** 2` | px² |
| `sigma_r_fit_px` | 1σ uncertainty on centroid (from Gaussian fit) | px |
| `two_sigma_r_px` | `2 × sigma_r_fit_px` | px |
| `sigma_r2_px2` | propagated: `2 × r_fit_px × sigma_r_fit_px` | px² |
| `two_sigma_r2_px2` | `2 × sigma_r2_px2` | px² |
| `amplitude_adu` | Gaussian amplitude above background | ADU |
| `baseline_adu` | background offset (`B0` from Gaussian fit; `profile_raw − amplitude_adu` if not stored separately) | ADU |
| `delta_r2` | `r2[i+1] − r2[i]` for successive peaks within each family | px² |

> **Note on baseline:** If `PeakFit` does not store `B0` directly, estimate it
> from the 20th-percentile of the fit window (consistent with S12 §10.3).
> Document the estimation method in the figure caption.

> **Note on `two_sigma` fields:** Per the uncertainty standards addendum
> (S04), `two_sigma_r_px = 2 × sigma_r_fit_px` exactly — no rounding.

---

## 9. Figures

All figures are created with `matplotlib`.  Each figure is:
1. **Displayed** via `plt.show()` (non-blocking where possible, so all
   figures can be open simultaneously).
2. **Saved** as a PNG to `output_dir` with the filename pattern
   `{cal_roi_stem}_z00_fig{letter}.png` (e.g., `flatsat_cal_roi_z00_figa.png`).

Use a consistent style: `plt.rcParams['font.size'] = 11`, `figure.dpi = 150`.

---

### Figure (a) — Dark-subtracted image and histogram

**Filename:** `{cal_roi_stem}_z00_figa.png`

**Layout:** 1 row × 2 panels.

**Left panel — image:**
- Display the dark-subtracted image `img_ds = fp image` (retrieved from the
  `subtract_dark` call before passing to `reduce_calibration_frame`; save it
  as a local variable `img_ds` in the processing chain).
- Use `imshow` with `origin='lower'`, colormap `'gray'`, and a `vmin`/`vmax`
  set to the 1st and 99th percentiles of `img_ds` (robust stretch).
- Overlay a red `+` marker at `(fp.cx, fp.cy)` (the recovered centre).
- Title: `"Dark-subtracted ROI"`.
- Add a colourbar labelled `"ADU"`.

**Right panel — histogram:**
- Plot `np.histogram(img_ds.ravel(), bins=256)` as a step histogram.
- x-axis: `"ADU"`, y-axis: `"Pixel count"`.
- Add a vertical dashed red line at the mean, and a vertical dashed green
  line at the median.  Label each in the legend.
- Title: `"Pixel histogram"`.

**Suptitle:** `"Fig (a): Dark-subtracted calibration ROI\n"
               f"(dark_n_frames={fp.dark_n_frames},  "
               f"cx={fp.cx:.3f} px,  cy={fp.cy:.3f} px)"`.

---

### Figure (b) — Coarse grid search: azimuthal variance vs. cx and cy

**Filename:** `{cal_roi_stem}_z00_figb.png`

**Purpose:** Show how the coarse ±`VAR_SEARCH_PX` grid search improves the
initial seed `(cx_seed, cy_seed)` = (N/2, N/2) to a coarsely determined
minimum.

**Data required:** The `azimuthal_variance_centre` function returns
`(cx_fine, cy_fine, cost_min, grid_cx, grid_cy, grid_cost)`.
The grid arrays `grid_cx`, `grid_cy`, `grid_cost` must be exposed by
`reduce_calibration_frame` or captured by temporarily patching the call.

> **Implementation note:** If `reduce_calibration_frame` does not currently
> return grid arrays, add an optional `return_grid=False` parameter to
> `azimuthal_variance_centre` (no change to public API of
> `reduce_calibration_frame`).  Call it directly from the validation script
> with `return_grid=True` to obtain the grid, then re-call
> `reduce_calibration_frame` normally.  Document this in the Claude Code
> prompt.

**Layout:** 1 row × 2 panels.

**Left panel — variance vs. cx:**
- For each row of the grid (fixed cy = grid row nearest cy_seed), plot
  variance cost vs. cx.
- If the grid is 2D (shape `[n_cy, n_cx]`), take the row corresponding to
  the cy value nearest `cy_seed`.
- Mark the initial seed `cx_seed` with a vertical dashed grey line.
- Mark the grid minimum `fp.grid_cx` with a vertical dashed red line.
- Mark the final Nelder-Mead result `fp.cx` with a vertical solid blue line.
- x-axis: `"cx (px)"`, y-axis: `"Azimuthal variance cost"`.
- Title: `"Coarse search: cost vs. cx"`.
- Legend: `["Initial seed", "Grid minimum", "Final (Nelder-Mead)"]`.

**Right panel — variance vs. cy:**
- Analogous: take the column of the grid nearest `cx_seed`.
- Same marker conventions for `cy_seed`, `fp.grid_cy`, `fp.cy`.
- x-axis: `"cy (px)"`, y-axis: `"Azimuthal variance cost"`.
- Title: `"Coarse search: cost vs. cy"`.

**Suptitle:** `"Fig (b): Coarse centre-finding grid search  "
               f"(search ±{VAR_SEARCH_PX:.1f} px around seed)"`.

---

### Figure (c) — Nelder-Mead minimisation convergence

**Filename:** `{cal_roi_stem}_z00_figc.png`

**Purpose:** Illustrate the fine-pass convergence from the coarse grid minimum
to the final sub-pixel centre.

**Data required:** Instrument the `_variance_cost` function (or the
`scipy.optimize.minimize` callback) to record the cost at each Nelder-Mead
iteration.  Pass a `callback` function to `minimize` that appends
`(cx, cy, cost)` to a list.

> **Implementation note:** `scipy.optimize.minimize(..., method='Nelder-Mead',
> callback=cb)` receives the current simplex vertex `x` at each iteration.
> In the callback, evaluate `_variance_cost(*x, ...)` to get the cost.  Store
> only the function evaluations that lie within ±5 px of the initial simplex
> point; this avoids artefacts from early large-simplex steps.

**Layout:** 1 row × 2 panels.

**Left panel — cost vs. iteration:**
- Plot cost (y) vs. iteration number (x).
- Add a horizontal dashed red line at `fp.cost_at_min`.
- x-axis: `"Nelder-Mead iteration"`, y-axis: `"Azimuthal variance cost"`.
- Title: `"Fine minimisation convergence"`.

**Right panel — trajectory in (cx, cy) space:**
- Plot cx vs. cy for each iteration as a connected scatter plot.
- Mark start (grid minimum) with a green circle, end (final cx, cy) with
  a red star.
- x-axis: `"cx (px)"`, y-axis: `"cy (px)"`.
- Title: `"Centre trajectory (Nelder-Mead)"`.

**Suptitle:** `"Fig (c): Nelder-Mead fine minimisation  "
               f"(cx={fp.cx:.4f},  cy={fp.cy:.4f},  "
               f"σ_cx={fp.sigma_cx:.4f},  σ_cy={fp.sigma_cy:.4f} px)"`.

---

### Figure (d) — Annular reduction profile

**Filename:** `{cal_roi_stem}_z00_figd.png`

**Layout:** Single panel.

**Content:**
- Plot `fp.profile` (mean ADU, y) vs. `fp.r_grid` (radius in px, x) as a
  solid blue line.
- Shade the 1σ envelope: `fp.profile ± fp.sigma_profile`, using `fill_between`
  with `alpha=0.25`.
- Mask out any bins where `fp.masked == True` (do not plot those points).
- Mark each detected peak position `p.r_fit_px` with a vertical dashed red
  line.
- x-axis: `"Radius (px)"`, y-axis: `"Mean intensity (ADU)"`.
- Title: `"Annular reduction profile"`.
- Add a text annotation in the upper right: `f"{len(fp.peak_fits)} peaks found"`.

**Suptitle:** `"Fig (d): r²-binned annular reduction profile\n"
               f"(N_bins={fp.n_bins},  r_max={fp.r_max_px:.1f} px,  "
               f"n_subpixels={fp.n_subpixels})"`.

---

### Figure (e) — Gaussian fits to fringe peaks

**Filename:** `{cal_roi_stem}_z00_fige.png`

**Purpose:** Show the Gaussian fit superimposed on the data for each individual
fringe peak.

**Layout:** Subplot grid of `ceil(sqrt(N_peaks)) × ceil(N_peaks / ceil(sqrt(N_peaks)))`.
For 20 peaks, a 4 × 5 grid works well.  For ≤ 7 peaks, use 2 × 4.

**Each sub-panel (one per peak):**
- Extract the profile window used for fitting:
  radius range `[p.r_raw_px − fit_half_window * dr, p.r_raw_px + fit_half_window * dr]`
  where `dr` is the median bin spacing.
- Plot the profile data as black circles with error bars (`fp.sigma_profile`).
- Overlay the best-fit Gaussian:
  `A * exp(-0.5 * ((r - mu)/sig)**2) + B0`
  using `p.amplitude_adu`, `p.r_fit_px`, `p.width_px`, and `baseline_adu`.
- If `p.fit_ok` is `True`, use a solid red curve; if `False`, use a dashed
  orange curve with label `"FIT FAILED"`.
- Mark `p.r_fit_px` with a vertical blue dotted line.
- Title of each sub-panel: `f"Ring {ring_label}"` where `ring_label` is:
  - `"A{idx}"` for Neon-A family (e.g., `"A1"`, `"A2"`, …)
  - `"B{idx}"` for Neon-B family
  - `"OI{idx}"` for airglow family
- x-axis: `"r (px)"`, y-axis: `"ADU"` (suppress tick labels on interior panels).

**Suptitle:** `"Fig (e): Gaussian peak fits"`.

---

### Figure (f) — Peak fit results table

**Filename:** `{cal_roi_stem}_z00_figf.png`

**Purpose:** Present all fit results in tabular form, rendered as a matplotlib
`table` inside a figure with no axes.

**Column definitions (neon calibration, fringe type 1):**

| Column header | Content | Notes |
|--------------|---------|-------|
| `Neon-A (640.2248 nm)` | 1-based ring index *i* for Neon-A peaks; blank for Neon-B rows | |
| `Neon-B (638.2991 nm)` | 1-based ring index *j* for Neon-B peaks; blank for Neon-A rows | |
| `r_fit (px) ± 2σ` | `f"{p.r_fit_px:.3f} ± {two_sigma_r_px:.3f}"` | |
| `r² (px²) ± 2σ` | `f"{r2_fit_px2:.2f} ± {two_sigma_r2_px2:.2f}"` | |
| `Amplitude (ADU)` | `f"{p.amplitude_adu:.1f}"` | |
| `Baseline (ADU)` | `f"{baseline_adu:.1f}"` | |
| `fit_ok` | `"✓"` or `"✗"` | |

Rows are sorted by ascending `r_fit_px` across all families.  The first
column (`Neon-A` or `Neon-B`) is non-blank only for its respective family.

**Column definitions (airglow, fringe type 2):**

| Column header | Content |
|--------------|---------|
| `OI (630.0304 nm air-rest)` | 1-based ring index |
| `r_fit (px) ± 2σ` | as above |
| `r² (px²) ± 2σ` | as above |
| `Amplitude (ADU)` | as above |
| `Baseline (ADU)` | as above |
| `fit_ok` | as above |

**Rendering:**

```python
fig_f, ax_f = plt.subplots(figsize=(12, max(4, 0.4 * n_rows + 1.5)))
ax_f.axis('off')
tbl = ax_f.table(
    cellText  = table_data,
    colLabels = col_headers,
    cellLoc   = 'center',
    loc       = 'center',
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.auto_set_column_width(col=list(range(len(col_headers))))
```

Shade rows belonging to Neon-A peaks in light blue (`'#ddeeff'`) and
Neon-B peaks in light yellow (`'#fffacc'`).  For airglow, alternate
white and light grey rows.

**Suptitle:** `"Fig (f): Peak fit results\n"
               f"({len(fp.peak_fits)} peaks, {sum(p.fit_ok for p in fp.peak_fits)} fit_ok)"`.

---

### Figure (g) — Melissinos Table 7.5 analog and Fig. 7.28 analog

**Filename:** `{cal_roi_stem}_z00_figg.png`

**Purpose:** Reproduce the Melissinos (1966) §7.3 analysis for the WindCube
data: (1) a table of ring radii and their squares, analogous to his Table 7.5
"Radii of Fabry-Perot Pattern for Single Line", and (2) a plot of ring order
*p* vs. *r²* (px²), analogous to his Fig. 7.28.

**Layout:** 1 row × 2 panels.  Left panel holds the table; right panel holds
the *p* vs. *r²* scatter plot with linear fit.

#### Left panel — Melissinos Table 7.5 analog

Render a matplotlib `table` for **each** peak family separately.  For neon
calibration, produce two sub-tables stacked vertically (Neon-A above Neon-B),
separated by a blank spacer row.  For airglow, produce one table.

**Columns per family:**

| Column | Content | Units |
|--------|---------|-------|
| `p` | 1-based ring index | — |
| `r_p (px)` | `p.r_fit_px` | px |
| `r²_p (px²)` | `r_fit_px²` | px² |
| `Δr²_{p,p+1} (px²)` | `r²_{p+1} − r²_p` (blank for last ring) | px² |

This directly mirrors Melissinos Table 7.5 columns:
`p | R_p (cm) | R²_p (mm²) | (R²_{p+1} − R²_p) (mm²)`.

Use consistent decimal formatting:
- `r_p`: 3 decimal places
- `r²_p`: 2 decimal places
- `Δr²`: 2 decimal places

Add a header row labelled:
- `"Neon-A  (640.2248 nm)"` or `"Neon-B  (638.2991 nm)"` or
  `"OI 630.0304 nm (air-rest)"`.

#### Right panel — Melissinos Fig. 7.28 analog

- x-axis: `r²_p` (px²), labelled `"r² (px²)"`.
- y-axis: ring order `p` (integer), labelled `"Ring order p"`.
- Plot each family as a separate scatter series:
  - Neon-A: blue filled circles
  - Neon-B: red filled squares
  - OI: green filled triangles
- For each family, perform an **ordinary least-squares linear fit**
  `p = m · r² + b` and overlay the fit line as a dashed line of the same
  colour.
- Annotate the slope `m` (px⁻² per order) and intercept `b` (order at
  r² = 0, related to the fractional order `ε`) in a legend box.

  ```
  Neon-A: m = {m_A:.6f} px⁻², b = {b_A:.4f}  (ε_A = 1 − b_A mod 1)
  Neon-B: m = {m_B:.6f} px⁻², b = {b_B:.4f}  (ε_B = 1 − b_B mod 1)
  ```

- The straight-line fit to *all* data points is the key diagnostic:
  good linearity (R² > 0.9999) confirms that fringes are correctly assigned
  to a single family and that the centre is well-determined.
- Add a text box in the lower-right corner reporting `R²_fit` for each family.

**Suptitle:** `"Fig (g): Melissinos Table 7.5 + Fig. 7.28 analogs\n"
               "(ring order p vs. r² demonstrates equal spacing in r²)"`.

---

## 10. Output file: `{cal_roi_stem}_fringe_peaks.npy`

After all figures are generated, save the complete peak table as a NumPy
**structured array** to:

```
{output_dir}/{cal_roi_stem}_fringe_peaks.npy
```

### 10.1 Structured array dtype

```python
dtype_cal = np.dtype([
    ('family',            'U8'),     # 'neon_A', 'neon_B', or 'oi'
    ('ring_index',        'i4'),     # 1-based ring index within family
    ('r_fit_px',          'f8'),     # Gaussian centroid radius (px)
    ('sigma_r_fit_px',    'f8'),     # 1σ uncertainty (px)
    ('two_sigma_r_px',    'f8'),     # exactly 2 × sigma_r_fit_px (px)
    ('r2_fit_px2',        'f8'),     # r_fit_px ** 2 (px²)
    ('sigma_r2_px2',      'f8'),     # propagated 1σ (px²)
    ('two_sigma_r2_px2',  'f8'),     # exactly 2 × sigma_r2_px2 (px²)
    ('amplitude_adu',     'f8'),     # Gaussian amplitude (ADU)
    ('baseline_adu',      'f8'),     # background offset (ADU)
    ('width_px',          'f8'),     # Gaussian sigma width (px)
    ('fit_ok',            'bool'),   # True if Gaussian fit converged
    ('peak_idx',          'i4'),     # bin index in annular profile
    ('r_raw_px',          'f8'),     # raw detected radius before Gaussian fit (px)
])
```

Rows are sorted by ascending `r_fit_px`.  Neon-A rows come before Neon-B
rows within the sort (tie-broken by family order A < B < OI).

### 10.2 Save call

```python
np.save(output_dir / f"{cal_roi_stem}_fringe_peaks.npy", peak_table)
print(f"\nSaved peak table: {output_dir / (cal_roi_stem + '_fringe_peaks.npy')}")
print(f"  {len(peak_table)} rows × {len(peak_table.dtype.names)} columns")
```

### 10.3 Downstream use

This file is the **primary input** to the Tolansky analysis validation
script (Z04, to be specified).  The `family`, `ring_index`, `r_fit_px`,
`two_sigma_r_px`, `r2_fit_px2`, and `two_sigma_r2_px2` fields map
directly onto the Tolansky two-line joint solver (S13 M05):

- Neon-A peaks → `r1_sq` array (λ₁ = 640.2248 nm family)
- Neon-B peaks → `r2_sq` array (λ₂ = 638.2991 nm family)
- `two_sigma_r2_px2` → weights for the Benoit excess-fraction solve

---

## 11. Acceptance criteria

The script is considered passing when **all** of the following are true:

| Check | Criterion |
|-------|-----------|
| Dark subtracted | `fp.dark_subtracted == True` |
| Dark frames | `fp.dark_n_frames >= 1` |
| Centre quality | `fp.quality_flags == 0x00` (GOOD) |
| Centre σ | `fp.sigma_cx < 0.5 px` and `fp.sigma_cy < 0.5 px` |
| Peaks (neon cal) | `len(fp.peak_fits) == 20` and `≥ 18 fit_ok` |
| Peaks (airglow) | `len(fp.peak_fits) >= 4` and all `fit_ok` |
| Neon family split | `len(peaks_A) == 10` and `len(peaks_B) == 10` |
| r² linearity (neon) | R²_fit > 0.9999 for both families in Fig. (g) |
| Output saved | `{cal_roi_stem}_fringe_peaks.npy` exists with correct row count |
| All 7 figures | Each PNG file exists and non-zero size |

If any criterion fails, the script prints a `FAIL` summary to the terminal
and exits with code 1.  If all pass, it prints `PASS` and exits with code 0.

---

## 12. File locations

```
soc_sewell/
├── validation/
│   └── z00_validate_annular_reduction_2026-04-13.py
└── docs/specs/
    └── Z00_validate_annular_reduction_peak_finding_2026-04-13.md
```

---

## 13. Instructions for Claude Code

This spec calls for a **new standalone script**.  Do not modify any
existing source file.

### Pre-implementation reads

Before writing any code, read in full:

1. `docs/specs/Z00_validate_annular_reduction_peak_finding_2026-04-13.md`
   (this file)
2. `docs/specs/S12_m03_annular_reduction_2026-04-06.md` (M03 API reference)

Identify the current dated module filename in `src/fpi/` and adjust the
import path accordingly.

### Task sequence

**Task 0 — Confirm M03 tests still pass**

```bash
pytest tests/test_m03_annular_reduction_2026_04_05.py -v
```

All 10 tests must pass before any new file is touched.

**Task 1 — Scaffold the script**

Create `validation/z00_validate_annular_reduction_2026-04-13.py` with:
- Imports (numpy, matplotlib, pathlib, tkinter, scipy, M03 module)
- `PARAMETERS` block (Section 4 constants)
- `select_npy_file()` helper (Section 5)
- Empty stubs for each figure function: `make_fig_a()`, …, `make_fig_g()`
- `main()` that calls file selection → fringe type query → processing chain
  (Section 6) → terminal summary → each figure function → save .npy →
  acceptance check (Section 11)

**Task 2 — Processing chain**

Implement Sections 6.1–6.5.  Confirm the terminal summary prints correctly
with a known `.npy` test pair if one is available.

**Task 3 — Peak-family separation**

Implement Section 7.  Print the per-family counts.

**Task 4 — Figures (a), (d)**

Implement `make_fig_a()` and `make_fig_d()` first — they require no grid or
callback data.  Verify visually.

**Task 5 — Figures (b), (c)**

Implement `make_fig_b()` and `make_fig_c()`.  These require instrumenting
`azimuthal_variance_centre`.  See the implementation notes in Sections 9b
and 9c.  If the grid data is not directly accessible from
`reduce_calibration_frame`, call `azimuthal_variance_centre` directly
(import it from M03) to capture grid data, then proceed with
`reduce_calibration_frame` normally.

**Task 6 — Figure (e)**

Implement `make_fig_e()`.  Confirm each sub-panel window correctly isolates
one peak.

**Task 7 — Figure (f)**

Implement `make_fig_f()` with the colour-coded table.

**Task 8 — Figure (g) and .npy save**

Implement `make_fig_g()`.  Compute OLS fits, annotate R² values.
Implement the structured array save (Section 10).

**Task 9 — Acceptance check**

Implement Section 11 checks.  Print `PASS` / `FAIL` summary.

**Task 10 — Full run and report**

Run the complete script end-to-end on the available FlatSat `.npy` arrays.
Report back in the fixed format below.

### Report format (paste back to Claude.ai)

```
=== Z00 CLAUDE CODE REPORT ===
Date: YYYY-MM-DD
Script: validation/z00_validate_annular_reduction_2026-04-13.py
M03 tests: NN/10 pass

INPUT FILES:
  cal_roi:  <path>  shape=(<N>,<N>)
  dark_roi: <path>  shape=(<N>,<N>)

PROCESSING:
  dark_subtracted: <True/False>
  cx=<val>, cy=<val>, sigma_cx=<val>, sigma_cy=<val>
  quality_flags=0x<XX>
  peaks found: <N> (<M> fit_ok)
  neon_A: <n_A>,  neon_B: <n_B>

FIGURES SAVED:
  (a) <filename>  <size KB>
  (b) <filename>  <size KB>
  (c) <filename>  <size KB>
  (d) <filename>  <size KB>
  (e) <filename>  <size KB>
  (f) <filename>  <size KB>
  (g) <filename>  <size KB>

NPY OUTPUT:
  <filename>  <N_rows> rows × <N_cols> cols

ACCEPTANCE:  PASS / FAIL
  [list any failing criteria]

DEVIATIONS FROM SPEC:
  [list any deviations, or "None"]
==============================
```

**Commit message:**
```
feat(z00): add annular reduction validation script, 7-panel diagnostic
Implements: Z00_validate_annular_reduction_peak_finding_2026-04-13.md
```

### Time limit

Stop and return the report if any task exceeds 15 minutes without all
relevant tests passing.  Do not attempt extended debugging loops.
Return findings to Claude.ai for diagnosis.
