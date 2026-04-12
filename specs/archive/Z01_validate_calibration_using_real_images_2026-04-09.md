# Z01 — Validate Calibration Using Real (or Synthetic) Images

**Spec ID:** Z01
**Spec file:** `docs/specs/Z01_validate_calibration_using_real_images_2026-04-09.md`
**Project:** WindCube FPI Pipeline
**Institution:** NCAR / High Altitude Observatory (HAO)
**Status:** Specification — ready for implementation in VS Code
**Tier:** 9 — Validation Testing
**Depends on:** S12 (M03 — annular reduction, dark subtraction, peak finding),
S13 (Tolansky — TwoLineResult), S19 (P01 — ImageMetadata, ingest_real_image)
**Used by:** Standalone validation; results feed into M05 (S14) manual inspection
**Last updated:** 2026-04-09
**Created/Modified by:** Claude AI

---

## 1. Purpose

`z01_validate_calibration_using_real_images_2026-04-09.py` is a short,
self-contained validation script that takes a user-selected calibration
image and dark image — either real binary frames from the WindCube payload
or synthetic images produced by the M02 forward model — and walks through
the following chain:

```
Load images → Inspect metadata → Select ROI → Inspect ROI →
Annular reduction + peak finding (S12) → Tolansky analysis (S13) →
Etalon characterisation figure
```

Each stage produces a matplotlib figure that the user inspects and closes
before the script advances. All figures are publication-quality, suitable
for sharing with team members as screenshots or saved PNGs.

**What this script is not.** Z01 does not run M05 (full staged calibration
inversion). It runs only M03 and S13 (Tolansky), which together characterise
the etalon gap `d`, focal length `f`, plate scale `α`, and fractional orders
`ε₁/ε₂`. This is the minimum necessary to validate that the instrument is
behaving correctly and that the reduction pipeline works on real data.

**Truth testing.** When run on synthetic image pairs (M02 cal + known dark),
the recovered `d`, `f`, and `α` should match the known synthesis parameters.
The script therefore serves both as a flight data inspector and as a ground-
truth verification harness.

---

## 2. Script structure overview

The script executes sequentially through six named stages. Each stage is
implemented as a standalone function so that future scripts can import and
re-use individual stages.

```
Stage A  load_images()             — file dialogs + binary ingest
Stage B  extract_metadata()        — P01 ImageMetadata extraction
Stage C  figure_image_pair()       — Figure 1: images + metadata tables
Stage D  get_roi_from_user()       — interactive centre seed + ROI selection
Stage E  figure_roi_inspection()   — Figure 2: ROI images + ADU histograms
Stage F  figure_reduction_peaks()  — Figure 3: r² profile + 20-peak overlay
Stage G  figure_tolansky()         — Figure 4: Tolansky r² fit + parameters
```

The `main()` function calls these in order, passing results forward as
plain Python dicts or dataclasses.

---

## 3. Stage A — `load_images()`

### 3.1 Function signature

```python
def load_images() -> dict:
    """
    Open two Windows file-picker dialogs (tkinter.filedialog.askopenfilename)
    and load the selected calibration image and dark image.

    Supports two file types:
        *.bin   — real WindCube binary (S19 / P01 ingest_real_image format)
                  260 rows × 276 cols × 2 bytes big-endian uint16
                  Row 0 is the metadata header; pixel data in rows 1–259.
        *.npy   — NumPy array file produced by M02 synthesise_calibration_image()
                  or any synthetic dark array. Shape must be (H, W), dtype float64.

    Returns
    -------
    dict with keys:
        'cal_image'  : np.ndarray, float64, shape (259, 276) for .bin
                       or (H, W) for .npy
        'dark_image' : np.ndarray, same shape, dtype consistent with cal_image
        'cal_path'   : pathlib.Path — absolute path to calibration file
        'dark_path'  : pathlib.Path — absolute path to dark file
        'cal_type'   : str, 'real' | 'synthetic'
        'dark_type'  : str, 'real' | 'synthetic'
        'cal_raw'    : np.ndarray uint16, shape (260, 276) — full raw array
                       including header row (row 0). None for .npy files.
        'dark_raw'   : np.ndarray uint16 or None
    """
```

### 3.2 Implementation notes

- Use `tkinter.Tk()` with `tk.withdraw()` to suppress the Tk root window.
  Call `askopenfilename` twice with appropriate title strings:
  `"Select WindCube Calibration Image"` and `"Select WindCube Dark Image"`.
- File type filter: `[("WindCube images", "*.bin *.npy"), ("All files", "*.*")]`
- If the user cancels either dialog, print a clear message and `sys.exit(0)`.
- For `.bin` files: call `ingest_real_image(path)` from S19/P01.
  Extract the pixel array (rows 1–259) as float64.
- For `.npy` files: call `np.load(path)`. Array must be 2D. If it is not,
  raise `ValueError` with a descriptive message.
- Do **not** perform dark subtraction here. Return raw pixel arrays.
- Validate that `cal_image.shape == dark_image.shape`. If not, raise
  `ValueError`.

---

## 4. Stage B — `extract_metadata()`

### 4.1 Function signature

```python
def extract_metadata(load_result: dict) -> dict:
    """
    Extract ImageMetadata from the loaded images using S19 P01 functions.

    Parameters
    ----------
    load_result : dict returned by load_images()

    Returns
    -------
    dict with keys:
        'cal_meta'  : ImageMetadata | None
        'dark_meta' : ImageMetadata | None
    """
```

### 4.2 Implementation notes

- For real `.bin` files: call `ingest_real_image(path)` which returns
  `(pixel_array, ImageMetadata)`. The metadata is already parsed by Stage A;
  Stage B re-uses the metadata object from that call (do not parse twice).
  **Implementation detail:** Stage A should store `ImageMetadata` in
  `load_result['cal_meta_raw']` and `load_result['dark_meta_raw']` so Stage
  B can retrieve it without re-opening the file.
- For `.npy` synthetic images: metadata is not embedded in the file.
  Set `cal_meta = None` and `dark_meta = None`. The figure in Stage C will
  display "Synthetic image — no embedded metadata" in the metadata table
  cell.
- Do not raise exceptions if metadata is None. Stage C handles the None case
  gracefully.

---

## 5. Stage C — `figure_image_pair()`

### 5.1 Function signature

```python
def figure_image_pair(
    load_result: dict,
    meta_result: dict,
) -> None:
    """
    Figure 1: side-by-side display of the raw calibration and dark images
    with metadata tables below each image.

    Blocks until the user closes the figure window.
    """
```

### 5.2 Layout

```
┌─────────────────────────────────────────────────────────────┐
│  Figure 1: Image Inspection                                 │
├─────────────────────┬───────────────────────────────────────┤
│  ax[0,0]            │  ax[0,1]                              │
│  CAL IMAGE          │  DARK IMAGE                           │
│  imshow (viridis)   │  imshow (viridis)                     │
│  colorbar           │  colorbar                             │
├─────────────────────┼───────────────────────────────────────┤
│  ax[1,0]            │  ax[1,1]                              │
│  CAL METADATA TABLE │  DARK METADATA TABLE                  │
│  (matplotlib table) │  (matplotlib table)                   │
└─────────────────────┴───────────────────────────────────────┘
```

Use `plt.subplots(2, 2, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 2]})`.

### 5.3 Title bars

- Calibration image subplot title:
  `f"CAL: {cal_path.name}  |  {cal_image.shape[1]}×{cal_image.shape[0]} px"`
- Dark image subplot title:
  `f"DARK: {dark_path.name}  |  {dark_image.shape[1]}×{dark_image.shape[0]} px"`
- Figure suptitle:
  `"Figure 1 — Image Pair Inspection  [close to continue]"`

### 5.4 Metadata table content

Display the following rows for each image (if metadata is available).
Left column is the field name; right column is the value with units.

| Row | Field name | Source field | Notes |
|-----|-----------|-------------|-------|
| 1 | Timestamp (UTC) | `utc_timestamp` | ISO format |
| 2 | Exposure time (cs) | `exposure_time_cs` | centiseconds |
| 3 | CCD temp (°C) | `ccd_temp1_c` | 1 decimal place |
| 4 | Etalon temps (°C) | `etalon_temps_c` | 4 values, 1 decimal place |
| 5 | Lamp channels on | `lamp_channels_on` | list of active channels |
| 6 | S/C latitude (°) | `spacecraft_latitude` | converted rad → deg, 2 d.p. |
| 7 | S/C longitude (°) | `spacecraft_longitude` | converted rad → deg, 2 d.p. |
| 8 | S/C altitude (km) | `spacecraft_altitude` | alt_m / 1000, 1 d.p. |
| 9 | Obs mode | `obs_mode` | string |
| 10 | Is synthetic | `is_synthetic` | True/False |

If `meta == None` (synthetic .npy with no sidecar): display a single-row
table with "Synthetic image — no embedded metadata".

Use `ax.table(cellText=..., loc='center')`. Turn off axis ticks and spines
for the table subplots (`ax.axis('off')`).

### 5.5 Image display

- Use `plt.cm.viridis` with `vmin=np.percentile(image, 1)`,
  `vmax=np.percentile(image, 99)` for each image independently.
- Add a colorbar to each image subplot with label "ADU".
- Display pixel coordinates on the axes (do not remove tick labels).

---

## 6. Stage D — `get_roi_from_user()`

### 6.1 Function signature

```python
def get_roi_from_user(
    cal_image: np.ndarray,
    default_roi: tuple[int, int] = (216, 216),
) -> dict:
    """
    Prompt the user for a fringe centre seed and ROI size via an interactive
    matplotlib figure and text console prompts.

    The user clicks once on the calibration image to set the centre seed.
    The ROI size is then prompted via the console (default 216×216 px).

    Returns
    -------
    dict with keys:
        'cx_seed'    : float — user-clicked x pixel coordinate
        'cy_seed'    : float — user-clicked y pixel coordinate
        'roi_cols'   : int   — ROI width in pixels
        'roi_rows'   : int   — ROI height in pixels
        'roi_x0'     : int   — ROI left edge (column index)
        'roi_y0'     : int   — ROI top edge (row index)
        'roi_x1'     : int   — ROI right edge (exclusive)
        'roi_y1'     : int   — ROI bottom edge (exclusive)
    """
```

### 6.2 Implementation

**Step 1 — Interactive click figure.**

Display the calibration image in a new matplotlib figure
(`figsize=(8, 8)`) with title:
`"Figure 1b — Click to mark fringe centre seed  [close after clicking]"`.

Use `fig.canvas.mpl_connect('button_press_event', on_click)` to capture the
click coordinates `(event.xdata, event.ydata)`. Store in a list `clicks = []`.
After the first click, draw a red `+` marker at the clicked location,
update the title to show `f"Seed set: ({x:.1f}, {y:.1f})  [close to continue]"`,
and refresh the canvas. The user then closes the window.

If no click was registered before window close (user just closed without clicking),
default to the image geometric centre: `(W/2, H/2)`.

**Step 2 — Console ROI prompt.**

After the window closes, print to the console:

```
  Fringe centre seed: (cx={cx_seed:.1f}, cy={cy_seed:.1f})
  Enter ROI size [default 216 216] (rows cols, or press ENTER for default):
```

Parse the user's input:
- Empty line → use default `(216, 216)`.
- Two integers → use as `(roi_rows, roi_cols)`.
- Any other input → warn and use default.

**Step 3 — Compute ROI bounds.**

```python
roi_x0 = max(0, int(round(cx_seed)) - roi_cols // 2)
roi_y0 = max(0, int(round(cy_seed)) - roi_rows // 2)
roi_x1 = min(cal_image.shape[1], roi_x0 + roi_cols)
roi_y1 = min(cal_image.shape[0], roi_y0 + roi_rows)
```

Clip to image boundaries. If the resulting ROI is smaller than requested
(edge case), warn the user but continue.

---

## 7. Stage E — `figure_roi_inspection()`

### 7.1 Function signature

```python
def figure_roi_inspection(
    cal_image:  np.ndarray,
    dark_image: np.ndarray,
    roi:        dict,
) -> np.ndarray:
    """
    Figure 2: Display cal ROI, dark ROI, and dark-subtracted cal ROI side
    by side, with ADU histograms below each image.

    Computes the visual dark-subtracted image for display purposes only.
    Does NOT use S12's subtract_dark() — that function is called internally
    by reduce_calibration_frame() in Stage F with the full images.

    Returns
    -------
    np.ndarray — dark-subtracted cal ROI (float64, clipped to ≥ 0)
                 for visual confirmation only. Not passed to S12.

    Blocks until the user closes the figure.
    """
```

### 7.2 Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  Figure 2: ROI Inspection                                       │
├───────────────┬───────────────┬─────────────────────────────────┤
│  ax[0,0]      │  ax[0,1]      │  ax[0,2]                        │
│  CAL ROI      │  DARK ROI     │  CAL − DARK (visual)            │
│  imshow       │  imshow       │  imshow                         │
│  colorbar     │  colorbar     │  colorbar                       │
├───────────────┼───────────────┼─────────────────────────────────┤
│  ax[1,0]      │  ax[1,1]      │  ax[1,2]                        │
│  CAL histogram│  DARK histogram│  (CAL−DARK) histogram          │
└───────────────┴───────────────┴─────────────────────────────────┘
```

Use `plt.subplots(2, 3, figsize=(18, 10), gridspec_kw={'height_ratios': [3, 1.5]})`.

### 7.3 Title bars

Each image subplot title:
- `f"CAL ROI  |  seed=({roi['cx_seed']:.1f}, {roi['cy_seed']:.1f})\n{roi['roi_cols']}×{roi['roi_rows']} px  |  {cal_path.name}"`
- `f"DARK ROI  |  {dark_path.name}"`
- `"CAL − DARK (visual only — not passed to S12)"`

Figure suptitle:
`"Figure 2 — ROI Inspection  [close to continue]"`

### 7.4 Histograms

For each ROI image panel, plot a histogram in the panel below:
- 128 bins, range = (image.min(), np.percentile(image, 99.9))
- Y-axis label: "Count"
- X-axis label: "ADU"
- Vertical dashed red line at `np.median(image)` with label `f"median={np.median(image):.0f}"`
- `plt.legend(fontsize=8)`

### 7.5 Dark subtraction for visual display

```python
cal_roi  = cal_image[roi['roi_y0']:roi['roi_y1'], roi['roi_x0']:roi['roi_x1']]
dark_roi = dark_image[roi['roi_y0']:roi['roi_y1'], roi['roi_x0']:roi['roi_x1']]
diff_roi = np.clip(cal_roi.astype(np.float64) - dark_roi.astype(np.float64), 0.0, None)
```

**Critical:** `diff_roi` is computed here solely for visual display in this
figure. The `diff_roi` array returned by this function must NOT be passed to
`reduce_calibration_frame()` in Stage F. Stage F receives the original
`cal_image` (full, unclipped, undifferenced) and a separate `master_dark`
argument so that M03's `subtract_dark()` function handles the subtraction
with its own clipping and float64 conversion logic. This prevents double
subtraction.

---

## 8. Stage F — `figure_reduction_peaks()`

### 8.1 Function signature

```python
def figure_reduction_peaks(
    cal_image:   np.ndarray,
    dark_image:  np.ndarray,
    roi:         dict,
    cal_path:    pathlib.Path,
) -> 'FringeProfile':
    """
    Run S12's reduce_calibration_frame() on the full cal image (not just the
    ROI) with master_dark provided, then plot the r²-binned profile with
    SEM error bars and the 20 identified neon peaks.

    Parameters
    ----------
    cal_image   : raw calibration pixel array (float64 or uint16)
    dark_image  : raw dark pixel array — passed as master_dark to S12
    roi         : dict from get_roi_from_user() — provides cx_seed, cy_seed
    cal_path    : used for figure title bar

    Returns
    -------
    FringeProfile — the full S12 output dataclass, used by Stage G
    """
```

### 8.2 S12 call

```python
from fpi.m03_annular_reduction_2026_04_05 import reduce_calibration_frame, make_master_dark

master_dark = make_master_dark([dark_image])   # single dark frame → master

fp = reduce_calibration_frame(
    image        = cal_image,
    master_dark  = master_dark,
    cx_human     = roi['cx_seed'],
    cy_human     = roi['cy_seed'],
    r_max_px     = min(roi['roi_rows'], roi['roi_cols']) / 2.0,
    n_bins       = 150,
)
```

**Note:** `reduce_calibration_frame()` performs dark subtraction as its
first internal step (Section 3.4 of S12). Z01 must not subtract dark from
`cal_image` before this call. `dark_image` is passed only as `master_dark`.

After the call, print a summary to the console:

```
  S12 annular reduction complete:
    Centre: ({fp.cx:.3f}, {fp.cy:.3f}) px  [σ=({fp.sigma_cx:.3f}, {fp.sigma_cy:.3f}) px]
    r_max: {fp.r_max:.1f} px
    Bins used: {fp.n_bins}
    Dark subtracted: {fp.dark_subtracted}
    Peaks found: {len([p for p in fp.peak_fits if p.fit_ok])} / {len(fp.peak_fits)}
    Chi²/dof: (reported by M05, not available here)
```

### 8.3 Figure layout

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Figure 3: Annular Reduction and Peak Identification                     │
├──────────────────────────────────────────────────────────────────────────┤
│  ax_profile (upper, ~60% height)                                         │
│  r²-binned mean intensity profile (blue line)                            │
│  SEM error bars (light blue shading or error bar caps)                   │
│  Vertical arrows + labels for each of 20 peaks                          │
│  X-axis: r² (px²)   Y-axis: Mean ADU                                    │
├──────────────────────────────────────────────────────────────────────────┤
│  ax_table (lower, ~40% height)                                           │
│  matplotlib table: peak fit results (20 rows)                            │
└──────────────────────────────────────────────────────────────────────────┘
```

Use `fig, (ax_profile, ax_table) = plt.subplots(2, 1, figsize=(18, 12),
gridspec_kw={'height_ratios': [3, 2]})`.

### 8.4 Profile plot

- Plot `fp.r_sq_centres` (x) vs `fp.profile` (y) as a solid blue line.
- Use `ax.fill_between(fp.r_sq_centres, fp.profile - fp.sem, fp.profile + fp.sem,
  alpha=0.3, color='steelblue', label='±1 SEM')` for the uncertainty band.
- X-axis label: `"r² (px²)"`
- Y-axis label: `"Mean intensity (ADU)"`

### 8.5 Peak overlay

For each `PeakFit` in `fp.peak_fits`:

If `pf.fit_ok == True`:
  - Draw a downward-pointing arrow (`ax.annotate`) from above the peak
    tip to the peak centre. Arrow style: `arrowstyle='->'`, color 'red'
    for high-amplitude peaks (Ne 640.2 nm family) and 'orange' for
    low-amplitude peaks (Ne 638.3 nm family). Classify by amplitude:
    peaks with `pf.amplitude > 0.5 * max_amplitude` are the 640.2 nm
    family (high); the remainder are the 638.3 nm family (low).
  - Add a small text label below the arrowhead:
    `f"r²={pf.centre:.0f}\n±{pf.sigma_centre:.1f}"`  in 7pt font.

If `pf.fit_ok == False`:
  - Mark with a grey `×` at the estimated peak location; label "failed".

### 8.6 Peak table

Display a table below the profile plot. One row per `PeakFit` (20 rows).

| Column | Content |
|--------|---------|
| # | Peak index (1–20) |
| Family | "Ne 640.2" (high) or "Ne 638.3" (low) |
| r² centre (px²) | `pf.centre` formatted to 1 decimal |
| σ(r²) (px²) | `pf.sigma_centre` formatted to 2 decimal |
| 2σ(r²) (px²) | `2 × pf.sigma_centre` formatted to 2 decimal |
| Amplitude (ADU) | `pf.amplitude` formatted to 1 decimal |
| σ(amp) (ADU) | `pf.sigma_amplitude` formatted to 2 decimal |
| Fit OK | ✓ or ✗ |

Use `ax_table.axis('off')`.
Use `ax_table.table(cellText=rows, colLabels=headers, loc='center',
cellLoc='center')` with `fontsize=8`.

### 8.7 Title bar

```python
suptitle = (
    f"Figure 3 — Annular Reduction and Peak Identification\n"
    f"File: {cal_path.name}  |  "
    f"Centre: ({fp.cx:.3f}, {fp.cy:.3f}) px  "
    f"[σ=({fp.sigma_cx:.3f}, {fp.sigma_cy:.3f}) px]  |  "
    f"r_max={fp.r_max:.1f} px  |  "
    f"Peaks: {n_ok}/20  [close to continue]"
)
```

---

## 9. Stage G — `figure_tolansky()`

### 9.1 Function signature

```python
def figure_tolansky(
    fp: 'FringeProfile',
    cal_path: pathlib.Path,
) -> 'TwoLineResult':
    """
    Run S13's TolanskyPipeline on the FringeProfile from Stage F and
    produce Figure 4: the two-line Tolansky r² characterisation.

    Returns
    -------
    TwoLineResult — the joint two-line fit result
    """
```

### 9.2 S13 call

```python
from fpi.tolansky_2026_04_05 import TolanskyPipeline

pipeline = TolanskyPipeline(
    fringe_profile = fp,
    d_prior_m      = 20.008e-3,   # ICOS spacer measurement — used only for N_int
    pixel_pitch_m  = 32e-6,       # 2×2 binned CCD
    lam1_nm        = 640.2248,
    lam2_nm        = 638.2991,
)
result = pipeline.run()
```

Print the full Tolansky summary to the console via `pipeline.print_summary()`.

### 9.3 Figure layout

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Figure 4: Tolansky Two-Line Etalon Characterisation                     │
├─────────────────────────────┬────────────────────────────────────────────┤
│  ax[0,0]                    │  ax[0,1]                                   │
│  r² vs fringe index         │  r² vs fringe index                        │
│  λ₁ = 640.2248 nm           │  λ₂ = 638.2991 nm                          │
│  Data + WLS fit line        │  Data + WLS fit line                       │
│  Residuals inset            │  Residuals inset                           │
├─────────────────────────────┴────────────────────────────────────────────┤
│  ax[1,:]  (spanning both columns)                                        │
│  Results parameter table                                                 │
└──────────────────────────────────────────────────────────────────────────┘
```

Use `fig = plt.figure(figsize=(18, 12))` with `GridSpec(2, 2,
height_ratios=[2.5, 1.5])`. Merge ax[1,0] and ax[1,1] for the table
(`fig.add_subplot(gs[1, :])`).

### 9.4 r² scatter plots (one per Ne line)

For each line (index 1 and 2):

- Plot measured r² values vs fringe index p as open circles with error
  bars (`xerr=None, yerr=result.sr1_sq` or `sr2_sq`).
  Marker: `'o'`, `markersize=6`, `mfc='white'`, `mec='steelblue'` (λ₁)
  or `'darkorange'` (λ₂).
- Overplot the WLS best-fit line as a solid line in the same colour.
- Label: `f"λ₁ = {result.lam1_nm} nm"` or `f"λ₂ = {result.lam2_nm} nm"`.
- X-axis label: `"Fringe index p"`
- Y-axis label: `"r² (px²)"`
- Add a text box in the upper-left corner showing:
  ```
  S₁ = {result.S1:.2f} ± {result.sigma_S1:.2f} px²/fringe
  ε₁ = {result.eps1:.5f} ± {result.sigma_eps1:.5f}
  R² = {r2_fit:.6f}
  ```
  (or ε₂ for λ₂ panel). Use `ax.text(0.03, 0.95, ..., transform=ax.transAxes,
  va='top', fontsize=9, bbox=dict(boxstyle='round', fc='lightyellow'))`.

### 9.5 Residuals inset

For each panel, add an inset axes (`ax.inset_axes([0.55, 0.05, 0.42, 0.3])`)
showing the residuals `r²_obs − r²_fit` in units of `σ(r²)` (normalised
residuals, should scatter around zero within ±3).

- Plot as a stem plot or scatter with horizontal dashed line at y=0.
- Y-label: "Residual (σ)"
- No x-label (shared with main panel context)

### 9.6 Results parameter table

Display the following recovered parameters in the bottom-spanning axes.
Turn off axis, use `ax.table(...)`.

| Parameter | Symbol | Value | 1σ uncertainty | 2σ uncertainty | Units | Notes |
|-----------|--------|-------|---------------|---------------|-------|-------|
| Etalon gap | d | `result.d_m * 1e3` | `result.sigma_d_m * 1e3` | `result.two_sigma_d_m * 1e3` | mm | ICOS prior: 20.008 mm |
| Focal length | f | `result.f_px * 32e-6 * 1e3` | `result.sigma_f_px * 32e-6 * 1e3` | `result.two_sigma_f_px * 32e-6 * 1e3` | mm | nominal 200 mm |
| Plate scale | α | `result.alpha_rad_px` | `result.sigma_alpha` | `result.two_sigma_alpha` | rad/px | Tolansky: ~1.607e-4 |
| Frac. order ε₁ | ε₁ | `result.epsilon_cal_1` | `result.sigma_eps1` | `2*result.sigma_eps1` | — | λ₁ = 640.2248 nm |
| Frac. order ε₂ | ε₂ | `result.epsilon_cal_2` | `result.sigma_eps2` | `2*result.sigma_eps2` | — | λ₂ = 638.2991 nm |
| Reduced χ² | χ²/ν | `result.chi2_dof` | — | — | — | acceptable: 0.5–3.0 |

Format numbers:
- d: 5 decimal places (mm) — resolves 0.001 mm = 1 µm
- f: 2 decimal places (mm)
- α: scientific notation, 4 significant figures
- ε: 5 decimal places

### 9.7 Title bar

```python
suptitle = (
    f"Figure 4 — Tolansky Two-Line Etalon Characterisation\n"
    f"File: {cal_path.name}  |  "
    f"d = {result.d_m*1e3:.4f} ± {result.sigma_d_m*1e3:.4f} mm  |  "
    f"f = {result.f_px*32e-6*1e3:.2f} ± {result.sigma_f_px*32e-6*1e3:.2f} mm  |  "
    f"α = {result.alpha_rad_px:.4e} ± {result.sigma_alpha:.4e} rad/px  |  "
    f"[close to exit]"
)
```

---

## 10. Main function

```python
def main():
    """
    Sequential execution of all six stages.
    Each figure blocks until the user closes it.
    """
    print("=" * 70)
    print("  WindCube FPI — Z01 Calibration Validation Script")
    print("=" * 70)

    # Stage A
    print("\nStage A: Loading images...")
    load = load_images()
    print(f"  CAL:  {load['cal_path'].name}  ({load['cal_type']})")
    print(f"  DARK: {load['dark_path'].name}  ({load['dark_type']})")

    # Stage B
    print("\nStage B: Extracting metadata...")
    meta = extract_metadata(load)

    # Stage C
    print("\nStage C: Displaying image pair (Figure 1)...")
    figure_image_pair(load, meta)

    # Stage D
    print("\nStage D: Collecting centre seed and ROI from user...")
    roi = get_roi_from_user(load['cal_image'])
    print(f"  Seed: ({roi['cx_seed']:.1f}, {roi['cy_seed']:.1f}) px")
    print(f"  ROI:  {roi['roi_cols']}×{roi['roi_rows']} px")

    # Stage E
    print("\nStage E: Displaying ROI inspection (Figure 2)...")
    figure_roi_inspection(load['cal_image'], load['dark_image'], roi)
    # Note: figure_roi_inspection returns the visual diff array but we
    # do NOT pass it to Stage F. Stage F gets the raw cal image + dark.

    # Stage F
    print("\nStage F: Running S12 annular reduction + peak finding (Figure 3)...")
    fp = figure_reduction_peaks(
        cal_image  = load['cal_image'],
        dark_image = load['dark_image'],
        roi        = roi,
        cal_path   = load['cal_path'],
    )

    # Stage G
    print("\nStage G: Running S13 Tolansky analysis (Figure 4)...")
    result = figure_tolansky(fp, load['cal_path'])

    print("\n" + "=" * 70)
    print("  Z01 complete. All figures closed. Tolansky result summary:")
    print(f"    d   = {result.d_m*1e3:.5f} ± {result.two_sigma_d_m*1e3:.5f} mm (2σ)")
    print(f"    f   = {result.f_px*32e-6*1e3:.3f} ± {result.two_sigma_f_px*32e-6*1e3:.3f} mm (2σ)")
    print(f"    α   = {result.alpha_rad_px:.4e} ± {result.two_sigma_alpha:.4e} rad/px (2σ)")
    print(f"    ε₁  = {result.epsilon_cal_1:.5f} ± {2*result.sigma_eps1:.5f} (2σ)")
    print(f"    ε₂  = {result.epsilon_cal_2:.5f} ± {2*result.sigma_eps2:.5f} (2σ)")
    print(f"    χ²/ν = {result.chi2_dof:.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
```

---

## 11. Dark subtraction architecture — clarification

This is the most important design decision in Z01. The following diagram
shows exactly where dark subtraction occurs and why it occurs there only once:

```
load_images()
  ├── cal_image (raw uint16 → float64, NO subtraction)
  └── dark_image (raw uint16 → float64, NO subtraction)
        │
        ▼
figure_roi_inspection()
  ├── cal_roi  = cal_image[ROI]
  ├── dark_roi = dark_image[ROI]
  └── diff_roi = clip(cal_roi − dark_roi, 0)  ← VISUAL ONLY, not propagated
        │
        ▼  [diff_roi is returned but DISCARDED by main()]
        │
        ▼
figure_reduction_peaks()
  ├── Calls: make_master_dark([dark_image])  → master_dark
  └── Calls: reduce_calibration_frame(
                 image       = cal_image,       ← RAW, not differenced
                 master_dark = master_dark,      ← S12 subtracts internally
                 cx_human    = roi['cx_seed'],
                 cy_human    = roi['cy_seed'],
                 r_max_px    = ...,
             )
      └── Inside S12: subtract_dark(cal_image, master_dark)  ← ONE subtraction
```

Z01 intentionally computes `diff_roi` for the visual confirmation step (Stage E)
using simple array arithmetic. This is not the same operation as `subtract_dark()`
from S12, which enforces float64 conversion, negative clipping, and provenance
tracking. Stage E's `diff_roi` is only for visual inspection — it is never
fed into any analysis function.

---

## 12. Key physical parameters used in this script

All physical constants must be imported from `windcube.constants` (S03). The
following are the relevant values referenced in Z01:

| Symbol | Value | Source | Notes |
|--------|-------|--------|-------|
| NE_WAVELENGTH_1_NM | 640.2248 | S03 / Burns 1950 | Primary neon line |
| NE_WAVELENGTH_2_NM | 638.2991 | S03 / Burns 1950 | Secondary neon line |
| CCD_PIXEL_PITCH_M | 32e-6 | S03 | 2×2 binned pixel pitch |
| ICOS_GAP_MM | 20.008 | S03 | Used only as TolanskyPipeline d_prior |
| F_NOMINAL_MM | 200.0 | S03 / reference only | Nominal; Tolansky will recover true value |

The recovered focal length from Tolansky is expected to be considerably less
than 200 mm (confirmed ~199.1 mm from FlatSat data, and could be 197–200 mm
for the flight instrument). The ICOS gap of 20.008 mm is used only to resolve
the integer order ambiguity N_int in the excess fractions step; the recovered
`d` will reflect the true operational gap (~20.106 mm from prior Tolansky work).

---

## 13. Verification tests

There are no automated pytest tests for Z01 — it is a fully interactive script.
However, the following manual checks are built into the workflow:

| Check | Figure | Criterion |
|-------|--------|-----------|
| V1 | Fig 1 | Images load correctly; metadata table populates with physically sensible values |
| V2 | Fig 1 | CCD temperature in range −30°C to +30°C; exposure time > 0 |
| V3 | Fig 2 | Dark ROI shows lower counts than cal ROI; diff ROI shows clear fringe pattern |
| V4 | Fig 2 | ADU histograms show reasonable distributions (no saturation in cal ROI) |
| V5 | Fig 3 | All 20 peaks identified (✓ symbols); no more than 2 failures (✗) acceptable |
| V6 | Fig 3 | Profile shape shows characteristic beat envelope from two Ne lines |
| V7 | Fig 4 | R² > 0.9999 for both Tolansky line fits |
| V8 | Fig 4 | Recovered d within 0.5 mm of 20.008 mm (resolves correct FSR period) |
| V9 | Fig 4 | Recovered f within 5 mm of 200 mm |
| V10| Fig 4 | χ²/ν in range [0.5, 3.0] |

---

## 14. Expected output (real FlatSat image)

Based on prior analysis of `cal_image_L1_1.npy` and related real images:

| Quantity | Expected |
|----------|----------|
| Fringe centre | ~(107.7, 109.9) px |
| r_max | ~103 px |
| Peaks found | 20 / 20 |
| Ne 640.2 nm peaks | 10 (high amplitude) |
| Ne 638.3 nm peaks | 10 (low amplitude, ~0.8×) |
| Tolansky d | ~20.106 ± 0.005 mm |
| Tolansky f | ~199.1 ± 1.0 mm |
| Tolansky α | ~1.65e-4 ± 0.01e-4 rad/px |
| χ²/ν | ~1.0–2.0 |

---

## 15. File location in repository

```
soc_sewell/
├── validation/
│   └── z01_validate_calibration_using_real_images_2026-04-09.py
└── docs/specs/
    └── Z01_validate_calibration_using_real_images_2026-04-09.md
```

The `validation/` directory is new (it did not exist in the S01–S20 layout).
Claude Code should create it if absent.

---

## 16. Dependencies

```
numpy       >= 1.24
scipy       >= 1.10
matplotlib  >= 3.7   # required for table, annotate, inset_axes
tkinter                # standard library; required for file dialogs
pathlib                # standard library
```

Internal imports (relative to `soc_sewell/`):
```python
from fpi.m03_annular_reduction_2026_04_05 import (
    reduce_calibration_frame,
    make_master_dark,
    FringeProfile,
)
from fpi.tolansky_2026_04_05 import TolanskyPipeline, TwoLineResult
from fpi.p01_metadata_2026_04_06 import ingest_real_image, ImageMetadata
from windcube.constants import (
    NE_WAVELENGTH_1_NM, NE_WAVELENGTH_2_NM,
    CCD_PIXEL_PITCH_M, ICOS_GAP_MM,
)
```

---

## 17. Instructions for Claude Code

1. Read this spec (Z01), S12 (M03), S13 (Tolansky), and S19 (P01) in full
   before writing any code.

2. Confirm that M03 and Tolansky tests pass before starting:
   ```bash
   pytest tests/test_m03_annular_reduction_*.py -v
   pytest tests/test_tolansky_*.py -v
   ```
   If either fails, stop and report the failure.

3. Create `validation/` directory if it does not exist.

4. Create `validation/z01_validate_calibration_using_real_images_2026-04-09.py`
   with the module-level docstring and header block matching S01 naming rules:
   ```python
   """
   Z01 — Validate Calibration Using Real (or Synthetic) Images
   WindCube FPI Pipeline — NCAR / High Altitude Observatory (HAO)
   Spec: docs/specs/Z01_validate_calibration_using_real_images_2026-04-09.md
   Tool: Claude Code
   Last updated: 2026-04-09
   """
   ```

5. Implement the six stage functions in the order:
   `load_images` → `extract_metadata` → `figure_image_pair` →
   `get_roi_from_user` → `figure_roi_inspection` → `figure_reduction_peaks`
   → `figure_tolansky` → `main`

6. **Dark subtraction critical rule (Section 11):** Never subtract the dark
   from `cal_image` before calling `reduce_calibration_frame()`. Always pass
   the raw `cal_image` + `master_dark` as separate arguments. The visual
   `diff_roi` in Stage E is computed from ROI slices for display only and
   is never passed to any analysis function.

7. Use `plt.show()` (blocking) after each figure to pause execution.
   Do not use `plt.ion()` or `plt.pause()`.

8. All `matplotlib.table()` calls must use `cellLoc='center'` and set the
   column widths explicitly to avoid text overflow.

9. For the interactive click in Stage D: if `event.xdata is None` (click
   outside axes), ignore it and wait for the next click.

10. Run the script manually with a real or synthetic image pair to confirm
    all four figures display correctly and all stage outputs print to console.
    Do not mark Z01 complete until a successful run has been confirmed.
