# Z00 — Validate Annular Reduction and Peak Finding

**Spec ID:** Z00
**Spec file:** `docs/specs/Z00_validate_annular_reduction_peak_finding_2026-04-13.md`
**Project:** WindCube FPI Pipeline
**Institution:** NCAR / High Altitude Observatory (HAO)
**Status:** v0.2 — implemented and reconciled 2026-04-13
**Depends on:** S12 (M03 `m03_annular_reduction_2026_04_06`)
**Produces:**
  - Seven diagnostic figures (a)–(g), saved to PNG
  - `{cal_roi_stem}_fringe_peaks.npy` — structured peak table, consumed later by Z04
    (Tolansky analysis validation)
**References:**
  - S12 — M03 Fringe Centre Finding and Annular Reduction Specification (2026-04-06)
  - Melissinos (1966) *Experiments in Modern Physics*, 1st ed., §7.3:
    Table 7.5 "Radii of Fabry-Perot Pattern for Single Line" and
    Fig. 7.28 "Ring order p vs. R² (mm²)"
  - Niciejewski et al. (1992) SPIE 1745 — r² reduction rationale
**Last updated:** 2026-04-13

> **Revision note v0.2 (2026-04-13):**
> Reconciled against Claude Code implementation report.
>
> (1) **M03 module renamed** — `m03_annular_reduction_2026_04_05.py` →
>     `m03_annular_reduction_2026_04_06.py` as a result of adding
>     `return_grid=False` to `azimuthal_variance_centre`.  Import path in
>     Section 6.1 and file locations in Section 12 updated accordingly.
>     All 10 M03 tests continue to pass.
>
> (2) **Grid-data capture pattern confirmed** — `FringeProfile` does not
>     expose `grid_cx / grid_cy` directly (those fields live in
>     `CentreResult` only).  Section 6.4 now gives a **definitive instruction**
>     to call `azimuthal_variance_centre(return_grid=True)` separately before
>     calling `reduce_calibration_frame`, not a conditional note.
>
> (3) **`--synthetic` flag documented** — Claude Code added a `--synthetic`
>     command-line flag for headless dry-runs using M02 synthetic data.
>     Sections 2 and 5 now document this flag.
>
> (4) **Acceptance behaviour on synthetic data clarified** — Section 11
>     notes that a synthetic single-amplitude image will fail the neon-family
>     and R²_fit criteria by design.  PASS is expected only on real two-line
>     FlatSat / flight data.

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
requires no command-line arguments in normal use, and prompts the user
for file locations at runtime.

---

## 2. Script location and invocation

```
soc_sewell/
└── validation/
    └── z00_validate_annular_reduction_2026-04-13.py
```

**Normal use (real FlatSat data):**

```bash
python validation/z00_validate_annular_reduction_2026-04-13.py
```

**Headless dry-run (synthetic data, no GUI required):**

```bash
python validation/z00_validate_annular_reduction_2026-04-13.py --synthetic
```

With `--synthetic`, the script generates its own calibration and dark arrays
via M02 `synthesise_calibration_image` and skips the tkinter file dialogs.
All figures and the `.npy` output are still produced.  Acceptance criteria
are **not** expected to pass in synthetic mode (see Section 11).

---

## 3. Required inputs

The script asks for **two `.npy` files** via interactive file-selection dialogs
(see Section 5).  Both must have been saved previously — for example, by the
binary-image metadata pipeline (S19/P01) or by any earlier processing step
that extracted and saved the ROI sub-arrays.

| Prompt | Expected array | Shape | dtype |
|--------|---------------|-------|-------|
| "Select the **calibration ROI** `.npy` file" | `cal_roi` | `(N, N)` | any numeric; uint16 typical |
| "Select the **dark ROI** `.npy` file" | `dark_roi` | `(N, N)` same shape as cal_roi | any numeric; uint16 typical |

`N` is typically 256 for a full 2×2-binned FlatSat frame, or a smaller
sub-array if the ROI was cropped before saving.  The script infers
`r_max_px` and the initial centre seed from the array size (Section 6.2).

### 3.1 Fringe-type detection

The script asks the user to declare the fringe type after files are loaded
(skipped in `--synthetic` mode, which defaults to fringe type 1):

```
Fringe type: [1] Neon calibration (2-line, 640.2 + 638.3 nm)
             [2] Airglow science  (1-line, 630.0 nm)
Enter 1 or 2:
```

This choice controls:
- The amplitude threshold used to separate the two neon peak families (Section 7)
- Column labels in Figures (f) and (g) (Sections 9.6, 9.7)
- Whether a single-family or two-family Melissinos table is produced (Section 9.7)

---

## 4. Instrument / reduction parameters

All numerical defaults are **fixed** for FlatSat / flight arrays and may be
overridden via clearly labelled constants at the top of the script.

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

Use `tkinter.filedialog.askopenfilename` for GUI file selection with a
`input()` fallback for headless environments.  If `--synthetic` was passed,
skip file selection and generate arrays programmatically.

```python
def select_npy_file(prompt: str) -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        print(f"\n{prompt}")
        path = filedialog.askopenfilename(
            title=prompt,
            filetypes=[("NumPy arrays", "*.npy"), ("All files", "*.*")],
        )
        root.destroy()
    except Exception:
        path = input(f"{prompt}\nEnter path: ").strip()
    if not path:
        raise FileNotFoundError("No file selected — aborting.")
    return path
```

The **stem** of `cal_roi_path` is the base name for all saved outputs:

```python
cal_roi_stem = pathlib.Path(cal_roi_path).stem
output_dir   = pathlib.Path(cal_roi_path).parent
```

**matplotlib backend:** Set `matplotlib.use('TkAgg')` for interactive display;
set `matplotlib.use('Agg')` in `--synthetic` / headless mode so figures are
saved without requiring a display.

---

## 6. Processing chain

### 6.1 Import

```python
from fpi.m03_annular_reduction_2026_04_06 import (
    make_master_dark,
    subtract_dark,
    azimuthal_variance_centre,   # imported directly for grid capture (§6.4)
    reduce_calibration_frame,
    reduce_science_frame,
    FringeProfile,
    PeakFit,
    QualityFlags,
)
```

### 6.2 Load arrays and infer parameters

```python
cal_roi  = np.load(cal_roi_path)
dark_roi = np.load(dark_roi_path)
N        = cal_roi.shape[0]
cx_seed  = (N - 1) / 2.0
cy_seed  = (N - 1) / 2.0
r_max    = R_MAX_PX if N >= 2 * R_MAX_PX else N / 2.0 - 2.0
```

### 6.3 Master dark

```python
master_dark = make_master_dark([dark_roi])
```

### 6.4 Grid capture for figure (b)

`FringeProfile` does not expose grid-search arrays (those live in
`CentreResult` only).  Capture them via a direct call to
`azimuthal_variance_centre` **before** calling `reduce_calibration_frame`:

```python
img_ds        = subtract_dark(cal_roi, master_dark, clip_negative=True)
p99_5         = float(np.percentile(img_ds, 99.5))
img_for_cost  = np.clip(img_ds, None, p99_5)

cx_fine, cy_fine, cost_min, grid_cx, grid_cy, grid_cost = \
    azimuthal_variance_centre(
        img_for_cost, cx_seed, cy_seed,
        var_r_max_px  = r_max,
        var_search_px = VAR_SEARCH_PX,
        return_grid   = True,
    )
# grid_cx: 1-D array of tested cx values
# grid_cy: 1-D array of tested cy values
# grid_cost: 2-D array shape (n_cy, n_cx)
```

### 6.5 Full reduction

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

### 6.6 Terminal summary

```
=== Z00 ANNULAR REDUCTION SUMMARY ===
  Input ROI      : {cal_roi_path}  shape={cal_roi.shape}
  Dark ROI       : {dark_roi_path}
  Dark subtracted: {fp.dark_subtracted}  ({fp.dark_n_frames} frame(s))
  Centre (fine)  : cx={fp.cx:.4f} px,  cy={fp.cy:.4f} px
  Centre sigma   : σ_cx={fp.sigma_cx:.4f} px,  σ_cy={fp.sigma_cy:.4f} px
  Seed source    : {fp.seed_source}
  Quality flags  : 0x{fp.quality_flags:02X}
  Cost at min    : {fp.cost_at_min:.4f}
  Peaks found    : {len(fp.peak_fits)}  ({sum(p.fit_ok for p in fp.peak_fits)} fit_ok)
  Profile bins   : {fp.n_bins}  (r_max={fp.r_max_px:.1f} px)
  Sparse bins    : {fp.sparse_bins}
=====================================
```

---

## 7. Peak-family separation

```python
if fringe_type == 1:
    peaks_A = sorted(
        [p for p in fp.peak_fits if p.amplitude_adu >= NEON_AMPLITUDE_SPLIT_ADU],
        key=lambda p: p.r_fit_px)
    peaks_B = sorted(
        [p for p in fp.peak_fits if p.amplitude_adu <  NEON_AMPLITUDE_SPLIT_ADU],
        key=lambda p: p.r_fit_px)
else:
    peaks_OI = sorted(fp.peak_fits, key=lambda p: p.r_fit_px)
```

---

## 8. Derived quantities

For each peak in each family:

| Symbol | Definition | Units |
|--------|-----------|-------|
| `r_fit_px` | Gaussian centroid | px |
| `r2_fit_px2` | `r_fit_px²` | px² |
| `sigma_r_fit_px` | 1σ centroid uncertainty | px |
| `two_sigma_r_px` | `2 × sigma_r_fit_px` (exact per S04) | px |
| `sigma_r2_px2` | `2 × r_fit_px × sigma_r_fit_px` | px² |
| `two_sigma_r2_px2` | `2 × sigma_r2_px2` (exact per S04) | px² |
| `amplitude_adu` | Gaussian amplitude above background | ADU |
| `baseline_adu` | background offset B0 | ADU |
| `delta_r2` | `r²[i+1] − r²[i]` within family | px² |

---

## 9. Figures

All figures saved as `{output_dir}/{cal_roi_stem}_z00_fig{letter}.png` at
`figure.dpi = 150`, `font.size = 11`.  Displayed interactively unless
`--synthetic` / headless.

---

### Figure (a) — Dark-subtracted image and histogram

**Layout:** 1 row × 2 panels.

**Left:** `imshow(img_ds)`, `origin='lower'`, `cmap='gray'`, `vmin/vmax` at
1st/99th percentile.  Red `+` marker at `(fp.cx, fp.cy)`.  Colourbar `"ADU"`.

**Right:** 256-bin step histogram of `img_ds.ravel()`.  Vertical dashed lines
at mean (red) and median (green) with legend.

**Suptitle:** `"Fig (a): Dark-subtracted calibration ROI"` + centre coords.

---

### Figure (b) — Coarse grid search: azimuthal variance vs. cx and cy

**Layout:** 1 row × 2 panels.  Data: `grid_cx`, `grid_cy`, `grid_cost` from
Section 6.4.

**Left:** Cost vs. `grid_cx` at the `grid_cy` row nearest `cy_seed`.
Vertical lines: `cx_seed` dashed grey (`"Initial seed"`),
`cx_fine` dashed red (`"Grid minimum"`), `fp.cx` solid blue (`"Final"`).

**Right:** Cost vs. `grid_cy` at the `grid_cx` column nearest `cx_seed`.
Same marker convention for cy.

**Suptitle:** `"Fig (b): Coarse centre-finding grid search"` + search range.

---

### Figure (c) — Nelder-Mead minimisation convergence

**Layout:** 1 row × 2 panels.

Instrument `_variance_cost` via the `scipy.optimize.minimize` callback to
record `(cx, cy)` at each iteration; re-evaluate cost at each point for the
y-axis.

**Left:** Cost vs. iteration.  Horizontal dashed red line at `fp.cost_at_min`.

**Right:** cx vs. cy trajectory, connected scatter.  Green circle at start
(grid minimum), red star at end (final).

**Suptitle:** `"Fig (c): Nelder-Mead fine minimisation"` + final cx, cy, σ.

---

### Figure (d) — Annular reduction profile

**Layout:** Single panel.

`fp.profile` (y) vs. `fp.r_grid` (x), solid blue.  `fill_between` ±
`fp.sigma_profile`, `alpha=0.25`.  Masked bins omitted.  Vertical dashed red
lines at each `p.r_fit_px`.  Peak count annotation upper right.

**Suptitle:** `"Fig (d): r²-binned annular reduction profile"` + N_bins, r_max.

---

### Figure (e) — Gaussian fits to fringe peaks

**Layout:** `ceil(sqrt(N)) × ceil(N / ceil(sqrt(N)))` subplots.
20 peaks → 4 × 5; ≤ 7 peaks → 2 × 4.

Each sub-panel: profile data (black circles + error bars) and Gaussian fit
(solid red if `fit_ok`, dashed orange if failed, labelled `"FIT FAILED"`).
Vertical blue dotted line at `p.r_fit_px`.
Sub-panel title: `"A{i}"` / `"B{j}"` / `"OI{k}"`.

**Suptitle:** `"Fig (e): Gaussian peak fits"`.

---

### Figure (f) — Peak fit results table

**Layout:** `ax.axis('off')`, matplotlib `table`.

**Columns (neon calibration):**

| Header | Content |
|--------|---------|
| `Neon-A (640.2248 nm)` | 1-based index or blank |
| `Neon-B (638.2991 nm)` | 1-based index or blank |
| `r_fit (px) ± 2σ` | `"{:.3f} ± {:.3f}"` |
| `r² (px²) ± 2σ` | `"{:.2f} ± {:.2f}"` |
| `Amplitude (ADU)` | `"{:.1f}"` |
| `Baseline (ADU)` | `"{:.1f}"` |
| `fit_ok` | `"✓"` / `"✗"` |

Rows sorted by ascending `r_fit_px`.  Neon-A rows shaded `'#ddeeff'`;
Neon-B rows `'#fffacc'`.

**Columns (airglow):** Replace the two index columns with a single
`OI (630.0304 nm air-rest)` column; alternate white / `'#f0f0f0'` rows.

**Suptitle:** `"Fig (f): Peak fit results"` + total / fit_ok count.

---

### Figure (g) — Melissinos Table 7.5 analog and Fig. 7.28 analog

**Layout:** 1 row × 2 panels.

#### Left panel — Table 7.5 analog

Matplotlib `table`.  For neon: Neon-A sub-table above Neon-B sub-table,
blank spacer row between.  For airglow: single table.

**Columns per family:**

| Header | Content | Format |
|--------|---------|--------|
| `p` | 1-based ring index | integer |
| `r_p (px)` | `r_fit_px` | 3 d.p. |
| `r²_p (px²)` | `r_fit_px²` | 2 d.p. |
| `Δr²_{p,p+1} (px²)` | `r²[p+1] − r²[p]`; blank for last | 2 d.p. |

Header row labelled with family name and wavelength.

#### Right panel — Fig. 7.28 analog

- x-axis: `r²_p (px²)`, y-axis: `Ring order p`
- Neon-A: blue circles; Neon-B: red squares; OI: green triangles
- OLS fit `p = m·r² + b` per family, dashed line of matching colour
- Legend per family:
  `"m = {m:.6f} px⁻²,  b = {b:.4f}  (ε ≈ {1 - b % 1:.4f})"`
- Text box lower-right with R²_fit per family

**Suptitle:** `"Fig (g): Melissinos Table 7.5 + Fig. 7.28 analogs\n"
              "(ring order p vs. r² — linearity confirms correct family assignment)"`.

---

## 10. Output file: `{cal_roi_stem}_fringe_peaks.npy`

### 10.1 Structured array dtype

```python
dtype_peaks = np.dtype([
    ('family',            'U8'),     # 'neon_A', 'neon_B', or 'oi'
    ('ring_index',        'i4'),     # 1-based within family
    ('r_fit_px',          'f8'),
    ('sigma_r_fit_px',    'f8'),
    ('two_sigma_r_px',    'f8'),     # exactly 2 × sigma_r_fit_px (S04)
    ('r2_fit_px2',        'f8'),
    ('sigma_r2_px2',      'f8'),
    ('two_sigma_r2_px2',  'f8'),     # exactly 2 × sigma_r2_px2 (S04)
    ('amplitude_adu',     'f8'),
    ('baseline_adu',      'f8'),
    ('width_px',          'f8'),
    ('fit_ok',            'bool'),
    ('peak_idx',          'i4'),
    ('r_raw_px',          'f8'),
])
```

Rows sorted by ascending `r_fit_px`.

### 10.2 Downstream use

Primary input to Z04 (Tolansky analysis validation):
- Neon-A rows → `r1_sq` (λ₁ = 640.2248 nm)
- Neon-B rows → `r2_sq` (λ₂ = 638.2991 nm)
- `two_sigma_r2_px2` → Benoit excess-fraction solver weights

---

## 11. Acceptance criteria

Designed for **real two-line FlatSat / flight data**.  A `--synthetic` run
is expected to fail the neon-family split and R²_fit checks because M02
produces a single-amplitude line — this is not a code defect.

| Check | Criterion |
|-------|-----------|
| Dark subtracted | `fp.dark_subtracted == True` |
| Dark frames | `fp.dark_n_frames >= 1` |
| Centre quality | `fp.quality_flags == 0x00` |
| Centre σ | `fp.sigma_cx < 0.5 px` and `fp.sigma_cy < 0.5 px` |
| Peaks (neon cal) | `len(fp.peak_fits) == 20` and `≥ 18 fit_ok` |
| Peaks (airglow) | `len(fp.peak_fits) >= 4` and all `fit_ok` |
| Neon family split | `len(peaks_A) == 10` and `len(peaks_B) == 10` |
| r² linearity | R²_fit > 0.9999 for each family in Fig. (g) |
| Output saved | `{cal_roi_stem}_fringe_peaks.npy` exists, correct row count |
| All 7 figures | Each PNG exists and is non-zero size |

Script exits with code 0 on PASS, code 1 on FAIL.

---

## 12. File locations

```
soc_sewell/
├── src/fpi/
│   └── m03_annular_reduction_2026_04_06.py    ← updated (return_grid added to
│                                                  azimuthal_variance_centre)
├── tests/
│   └── test_m03_annular_reduction_2026_04_05.py  ← unchanged; 10/10 pass
├── validation/
│   └── z00_validate_annular_reduction_2026-04-13.py
└── docs/specs/
    └── Z00_validate_annular_reduction_peak_finding_2026-04-13.md
```
