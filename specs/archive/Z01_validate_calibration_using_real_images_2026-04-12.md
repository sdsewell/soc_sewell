# Z01 — Validate Calibration Using Real (or Synthetic) Images

**Spec ID:** Z01
**Spec file:** `docs/specs/Z01_validate_calibration_using_real_images_2026-04-12.md`
**Project:** WindCube FPI Pipeline
**Institution:** NCAR / High Altitude Observatory (HAO)
**Status:** Implemented
**Tier:** 9 — Validation Testing
**Depends on:** S12 (M03 — annular reduction, dark subtraction, peak finding),
S13 (Tolansky — TwoLineResult), S19 (P01 — ImageMetadata, ingest_real_image)
**Used by:** Standalone validation; results feed into M05 (S14) manual inspection
**Last updated:** 2026-04-12
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
Annular reduction + peak finding (S12) → Dark subtraction comparison →
Tolansky analysis (S13) → Etalon characterisation figure
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

The script executes sequentially through seven named stages. Each stage is
implemented as a standalone function so that future scripts can import and
re-use individual stages.

```
Stage A  load_images()             — file dialogs + binary ingest
Stage B  extract_metadata()        — P01 ImageMetadata extraction
Stage C  figure_image_pair()       — Figure 1: images + metadata tables + click-to-seed
Stage D  figure_roi_inspection()   — Figure 2: ROI images + ADU histograms (2×2)
Stage F  run_s12_reduction()       — no figure: S12 reduction helper
Stage F  figure_dark_comparison()  — Figure 3: naive diff vs S12 dark-sub comparison
Stage F  figure_reduction_peaks()  — Figure 4: r² profile + 20-peak overlay
Stage G  figure_tolansky()         — Figure 5: Tolansky r² fit + parameters
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
) -> tuple[float, float]:
    """
    Figure 1: side-by-side display of the raw calibration and dark images
    with metadata tables below each image.

    Wires a click-to-seed handler on the cal image axis. Clicking places
    a red '+' marker, updates the subplot title with chosen coordinates,
    and records (cx_seed, cy_seed).

    Returns
    -------
    (cx_seed, cy_seed) : tuple[float, float]
        Clicked fringe centre seed in image pixel coordinates.
        Defaults to image geometric centre if no click was registered.

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
│  imshow (gray)      │  imshow (gray)                        │
│  colorbar           │  colorbar                             │
├─────────────────────┼───────────────────────────────────────┤
│  ax[1,0]            │  ax[1,1]                              │
│  CAL METADATA TABLE │  DARK METADATA TABLE                  │
│  (matplotlib table) │  (matplotlib table)                   │
└─────────────────────┴───────────────────────────────────────┘
```

Use `plt.subplots(2, 2, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 2]})`.

### 5.3 Title bars

- Calibration image subplot title (before click):
  `"Calibration frame — click to set fringe seed"`
- Calibration image subplot title (after click):
  `f"Calibration frame — seed: ({cx:.1f}, {cy:.1f})  [close to continue]"`
- Dark image subplot title:
  `"Master dark"`
- No pixel-size suffix (`| 276×259 px`) in titles.
- Figure suptitle: directory path of the two files. If both share the same
  parent directory, one line is sufficient. If they differ, two lines.
  The click instruction appears in the cal subplot title only, not in the
  suptitle.

### 5.4 Image display

- Use `cmap="gray"`, `vmin=0`, `vmax=16383` (full 14-bit range) for both
  cal and dark images. Do **not** use percentile-based scaling.
- Add a colorbar to each image subplot with label `"ADU"`.
- Display pixel coordinates on the axes (do not remove tick labels).

### 5.5 Click-to-seed

Wire a `button_press_event` handler to the cal image axis:

```python
clicks = []

def on_click(event):
    if event.inaxes is not ax_cal or event.xdata is None:
        return
    cx, cy = event.xdata, event.ydata
    clicks.append((cx, cy))
    ax_cal.plot(cx, cy, 'r+', markersize=14, markeredgewidth=2)
    ax_cal.set_title(f"Calibration frame — seed: ({cx:.1f}, {cy:.1f})  [close to continue]")
    fig.canvas.draw()

fig.canvas.mpl_connect('button_press_event', on_click)
plt.show()  # blocks

cx_seed = clicks[-1][0] if clicks else cal_image.shape[1] / 2.0
cy_seed = clicks[-1][1] if clicks else cal_image.shape[0] / 2.0
return cx_seed, cy_seed
```

### 5.6 Metadata table content

Display all `ImageMetadata` fields using `dataclasses.asdict()` — every
field is shown, not a hand-picked subset. Left column is the field name;
right column is the value with units. Apply the following conversions
in the value column:

- latitude / longitude: convert radians → degrees
- altitude: convert metres → km

AOCS fields to include (in addition to the standard fields):
- quaternion components (all four)
- pointing error
- ECI position (x, y, z)
- ECI velocity (x, y, z)
- ADCS quality flag: raw integer value **and** decoded bitmask names

If `meta == None` (synthetic .npy with no sidecar): display a single-row
table with `"Synthetic image — no embedded metadata"`.

Use `ax.table(cellText=..., loc='center')`. Turn off axis ticks and spines
for the table subplots (`ax.axis('off')`).

---

## 6. Stage D — `figure_roi_inspection()`

### 6.1 Function signature

```python
def figure_roi_inspection(
    cal_image:  np.ndarray,
    dark_image: np.ndarray,
    cx:         float,
    cy:         float,
    roi_half:   int = 108,
) -> None:
    """
    Figure 2: Display cal ROI and dark ROI side by side with ADU histograms.

    Returns nothing.

    Blocks until the user closes the figure.
    """
```

### 6.2 Layout

```
┌─────────────────────────────────────────────────────────────┐
│  Figure 2: ROI Inspection                                   │
├───────────────────────┬─────────────────────────────────────┤
│  ax[0,0]              │  ax[0,1]                            │
│  CAL ROI              │  DARK ROI                           │
│  imshow (gray)        │  imshow (gray)                      │
│  colorbar             │  colorbar                           │
├───────────────────────┼─────────────────────────────────────┤
│  ax[1,0]              │  ax[1,1]                            │
│  CAL ROI histogram    │  DARK ROI histogram                 │
└───────────────────────┴─────────────────────────────────────┘
```

Use `plt.subplots(2, 2, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1.5]})`.

No third column (visual diff panel) is produced.

### 6.3 Image display

- Use `cmap="gray"`, `vmin=0`, `vmax=16383` for both ROI panels.
- Add colorbars labelled `"ADU"`.

### 6.4 Histograms

For each ROI image panel, plot a histogram in the panel below:
- 128 bins, range = `(0, 16383)`
- Y-axis label: `"Count"`
- X-axis label: `"ADU"`
- Vertical dashed red line at `np.median(image)` with label
  `f"median={np.median(image):.0f}"`
- `plt.legend(fontsize=8)`

### 6.5 Title bars

- Cal ROI subplot: `f"CAL ROI  |  seed=({cx:.1f}, {cy:.1f})"`
- Dark ROI subplot: `"DARK ROI"`
- Figure suptitle: `"Figure 2 — ROI Inspection  [close to continue]"`

---

## 7. Stage F — `run_s12_reduction()` (no figure)

### 7.1 Function signature

```python
def run_s12_reduction(
    cal_image:  np.ndarray,
    dark_image: np.ndarray,
    cx:         float,
    cy:         float,
    roi:        dict,
) -> tuple['FringeProfile', np.ndarray]:
    """
    Stage F-1: encapsulates the S12 computation and exposes the full-frame
    dark-subtracted image for comparison in figure_dark_comparison().

    Execution order:
        1. Call reduce_calibration_frame(cal_image, master_dark, cx, cy, ...)
           → fp (FringeProfile)
        2. Call subtract_dark(cal_image, master_dark, clip_negative=True)
           → s12_dark_sub_image (full-frame, same call S12 makes internally)

    Returns
    -------
    (fp, s12_dark_sub_image) : tuple[FringeProfile, np.ndarray]
    """
```

### 7.2 Implementation notes

- `reduce_calibration_frame()` performs dark subtraction internally as its
  first step (Section 3.4 of S12). Z01 must not subtract dark from
  `cal_image` before this call.
- `subtract_dark()` is called a second time here solely to expose the
  full-frame result for Figure 3 comparison. This is not a double
  subtraction — the first call is internal to `reduce_calibration_frame()`
  and the second call here is for diagnostic output only.
- Print S12 summary to console after the call (see original Stage F
  console output specification in §8.2).

---

## 8. Stage F — `figure_dark_comparison()` (Figure 3, new)

### 8.1 Function signature

```python
def figure_dark_comparison(
    cal_roi:        np.ndarray,
    dark_roi:       np.ndarray,
    s12_dark_sub:   np.ndarray,
    roi_slice:      tuple[slice, slice],
) -> None:
    """
    Figure 3: compare the naive visual dark difference against the S12
    dark-subtracted image to confirm the correct dark frame is being routed
    into the pipeline.

    Returns nothing.

    Blocks until the user closes the figure.
    """
```

### 8.2 Layout

```
┌─────────────────────────────────────────────────────────────┐
│  Figure 3: Dark Subtraction Comparison                      │
├───────────────────────┬─────────────────────────────────────┤
│  ax[0,0]              │  ax[0,1]                            │
│  Naive ROI diff       │  S12 dark-sub ROI                   │
│  clip(cal−dark, 0)    │  s12_dark_sub[y0:y1, x0:x1]        │
│  imshow (gray)        │  imshow (gray)                      │
│  colorbar             │  colorbar                           │
├───────────────────────┼─────────────────────────────────────┤
│  ax[1,0]              │  ax[1,1]                            │
│  naive diff histogram │  S12 dark-sub histogram             │
└───────────────────────┴─────────────────────────────────────┘
```

Use `plt.subplots(2, 2, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1.5]})`.

### 8.3 Shared colour scale

```python
vmax = np.percentile(np.concatenate([naive_roi.ravel(), s12_roi.ravel()]), 99.5)
vmin = 0
```

Both top panels use the same `vmin`/`vmax` so they are directly comparable.

### 8.4 np.allclose check

```python
identical = np.allclose(naive_roi, s12_roi)
max_diff = np.max(np.abs(naive_roi - s12_roi))
check_str = (
    f"IDENTICAL (max |diff| = 0)"
    if identical
    else f"MISMATCH — max |diff| = {max_diff:.1f} ADU"
)
print(f"  Dark subtraction comparison: {check_str}")
```

The `check_str` is included in the figure suptitle.

### 8.5 Title bars

- Top-left: `"Naive ROI diff: clip(cal − dark, 0)"`
- Top-right: `"S12 dark-sub ROI"`
- Figure suptitle:
  `f"Figure 3 — Dark Subtraction Comparison  |  {check_str}  [close to continue]"`

---

## 9. Stage F — `figure_reduction_peaks()` (Figure 4, was Figure 3)

### 9.1 Function signature

```python
def figure_reduction_peaks(
    fp:       'FringeProfile',
    roi:      dict,
    cal_path: pathlib.Path,
) -> None:
    """
    Figure 4: r²-binned profile with SEM error bars and the 20 identified
    neon peaks.

    Takes the already-computed FringeProfile from run_s12_reduction() rather
    than recomputing it internally.

    Blocks until the user closes the figure.
    """
```

### 9.2 Figure layout

*(unchanged from original Stage F / Figure 3 — see §8.3–8.6 of the
2026-04-09 spec for full profile plot, peak overlay, and peak table
specifications)*

### 9.3 Title bar

```python
suptitle = (
    f"Figure 4 — Annular Reduction and Peak Identification\n"
    f"File: {cal_path.name}  |  "
    f"Centre: ({fp.cx:.3f}, {fp.cy:.3f}) px  "
    f"[σ=({fp.sigma_cx:.3f}, {fp.sigma_cy:.3f}) px]  |  "
    f"r_max={fp.r_max:.1f} px  |  "
    f"Peaks: {n_ok}/20  [close to continue]"
)
```

---

## 10. Stage G — `figure_tolansky()` (Figure 5, was Figure 4)

### 10.1 Function signature

```python
def figure_tolansky(
    fp:       'FringeProfile',
    cal_path: pathlib.Path,
) -> 'TwoLineResult':
    """
    Figure 5: Tolansky two-line etalon characterisation.

    *(unchanged from original Stage G — see §9 of the 2026-04-09 spec)*

    Returns
    -------
    TwoLineResult — the joint two-line fit result
    """
```

### 10.2 Title bar

```python
suptitle = (
    f"Figure 5 — Tolansky Two-Line Etalon Characterisation\n"
    f"File: {cal_path.name}  |  "
    f"d = {result.d_m*1e3:.4f} ± {result.sigma_d_m*1e3:.4f} mm  |  "
    f"f = {result.f_px*32e-6*1e3:.2f} ± {result.sigma_f_px*32e-6*1e3:.2f} mm  |  "
    f"α = {result.alpha_rad_px:.4e} ± {result.sigma_alpha:.4e} rad/px  |  "
    f"[close to exit]"
)
```

*(All other specifications unchanged from §9 of the 2026-04-09 spec.)*

---

## 11. Main function

```python
def main():
    """
    Sequential execution of all stages.
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

    # Stage C — Figure 1 (returns click seed)
    print("\nStage C: Displaying image pair (Figure 1)...")
    cx_seed, cy_seed = figure_image_pair(load, meta)
    print(f"  Fringe centre seed: ({cx_seed:.1f}, {cy_seed:.1f}) px")

    # Stage D — Figure 2 (ROI inspection, 2×2 layout)
    print("\nStage D: Displaying ROI inspection (Figure 2)...")
    figure_roi_inspection(
        load['cal_image'], load['dark_image'], cx_seed, cy_seed
    )

    # Stage F — sub-step 1: run S12 reduction (no figure)
    print("\nStage F-1: Running S12 annular reduction...")
    roi = {
        'cx_seed': cx_seed,
        'cy_seed': cy_seed,
        # roi bounds computed inside run_s12_reduction or passed from Stage D
    }
    fp, s12_dark_sub = run_s12_reduction(
        load['cal_image'], load['dark_image'], cx_seed, cy_seed, roi
    )

    # Stage F — sub-step 2: dark subtraction comparison (Figure 3)
    print("\nStage F-2: Dark subtraction comparison (Figure 3)...")
    roi_slice = (
        slice(roi['roi_y0'], roi['roi_y1']),
        slice(roi['roi_x0'], roi['roi_x1']),
    )
    cal_roi  = load['cal_image'][roi_slice]
    dark_roi = load['dark_image'][roi_slice]
    figure_dark_comparison(cal_roi, dark_roi, s12_dark_sub, roi_slice)

    # Stage F — sub-step 3: radial profile (Figure 4)
    print("\nStage F-3: Radial fringe profile (Figure 4)...")
    figure_reduction_peaks(fp, roi, load['cal_path'])

    # Stage G — Tolansky analysis (Figure 5)
    print("\nStage G: Running S13 Tolansky analysis (Figure 5)...")
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

## 12. Dark subtraction architecture — clarification

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
  └── [displayed only — no diff panel produced, no array returned]
        │
        ▼
run_s12_reduction()
  ├── Calls: make_master_dark([dark_image])  → master_dark
  ├── Calls: reduce_calibration_frame(
  │              image       = cal_image,       ← RAW, not differenced
  │              master_dark = master_dark,      ← S12 subtracts internally
  │              cx_human    = cx_seed,
  │              cy_human    = cy_seed,
  │              r_max_px    = ...,
  │          )
  │   └── Inside S12: subtract_dark(cal_image, master_dark)  ← ONE subtraction
  └── Calls: subtract_dark(cal_image, master_dark, clip_negative=True)
             → s12_dark_sub_image (diagnostic exposure for Figure 3 only)
        │
        ▼
figure_dark_comparison()
  ├── naive_roi = clip(cal_roi − dark_roi, 0)   ← visual reference
  └── s12_roi   = s12_dark_sub_image[ROI]        ← S12 pipeline output
      np.allclose(naive_roi, s12_roi) → "IDENTICAL" or "MISMATCH"
```

---

## 13. Key physical parameters used in this script

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

## 14. Verification tests

There are no automated pytest tests for Z01 — it is a fully interactive script.
However, the following manual checks are built into the workflow:

| Check | Figure | Criterion |
|-------|--------|-----------|
| V1 | Fig 1 | Images load correctly; metadata table shows all `dataclasses.asdict()` fields |
| V2 | Fig 1 | CCD temperature in range −30°C to +30°C; exposure time > 0 |
| V3 | Fig 2 | Cal ROI shows fringe rings; dark ROI shows low uniform counts |
| V4 | Fig 2 | ADU histograms show reasonable distributions (no saturation in cal ROI) |
| V5 | Fig 3 | Suptitle reads "IDENTICAL (max \|diff\| = 0)" — confirms dark routing |
| V6 | Fig 4 | All 20 peaks identified (✓ symbols); no more than 2 failures (✗) acceptable |
| V7 | Fig 4 | Profile shape shows characteristic beat envelope from two Ne lines |
| V8 | Fig 5 | R² > 0.9999 for both Tolansky line fits |
| V9 | Fig 5 | Recovered d within 0.5 mm of 20.008 mm (resolves correct FSR period) |
| V10 | Fig 5 | Recovered f within 5 mm of 200 mm |
| V11 | Fig 5 | χ²/ν in range [0.5, 3.0] |

---

## 15. Expected output (real FlatSat image)

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
| Fig 3 check | IDENTICAL (max \|diff\| = 0) |
| χ²/ν | ~1.0–2.0 |

---

## 16. File location in repository

```
soc_sewell/
├── validation/
│   └── z01_validate_calibration_using_real_images_2026-04-09.py
└── docs/specs/
    └── Z01_validate_calibration_using_real_images_2026-04-12.md
```

---

## 17. Dependencies

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
    subtract_dark,
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

## 18. Instructions for Claude Code

1. Read this spec (Z01), S12 (M03), S13 (Tolansky), and S19 (P01) in full
   before writing any code.

2. Confirm that M03 and Tolansky tests pass before starting:
   ```bash
   pytest tests/test_m03_annular_reduction_*.py -v
   pytest tests/test_tolansky_*.py -v
   ```
   If either fails, stop and report the failure.

3. The `validation/` directory already exists from the prior implementation.

4. The script filename is unchanged:
   `validation/z01_validate_calibration_using_real_images_2026-04-09.py`

5. Implement **only** the following changes relative to the 2026-04-09
   implementation:
   - `figure_image_pair`: change cmap `viridis` → `gray`; scale `vmin=0`,
     `vmax=16383`; add click-to-seed handler; return `(cx_seed, cy_seed)`;
     update suptitle to show directory path; metadata via `dataclasses.asdict()`
   - `figure_roi_inspection`: change layout 2×3 → 2×2 (remove diff panel);
     change cmap to `gray`, scale `vmin=0`, `vmax=16383`; return `None`
   - New `run_s12_reduction()`: extract S12 call from old `figure_reduction_peaks`;
     add second `subtract_dark()` call; return `(fp, s12_dark_sub_image)`
   - New `figure_dark_comparison()`: implement as Figure 3 per §8 above
   - `figure_reduction_peaks`: rename to Figure 4; accept pre-computed `fp`;
     simplify signature to `(fp, roi, cal_path)`
   - `figure_tolansky`: rename to Figure 5; update suptitle number only
   - `main()`: update stage F to three sub-steps per §11 above

6. **Dark subtraction critical rule (§12):** Never subtract the dark from
   `cal_image` before calling `reduce_calibration_frame()`. The second
   `subtract_dark()` call in `run_s12_reduction()` is for diagnostic
   exposure to Figure 3 only — it is not fed back into any analysis.

7. Use `plt.show()` (blocking) after each figure. Do not use `plt.ion()`
   or `plt.pause()`.

8. Run the script manually with a real or synthetic image pair to confirm
   all five figures display correctly and the Figure 3 suptitle reads
   "IDENTICAL". Report back with the console output from a successful run.

---

## 19. Change log

| Version | Date | Changes |
|---------|------|---------|
| v0.1 | 2026-04-09 | Initial specification and implementation |
| v0.2 | 2026-04-12 | Fig 1: cmap→gray, 14-bit scale, click-to-seed returns (cx,cy), dataclasses.asdict() metadata, directory path suptitle. Fig 2: 2×3→2×2 layout, gray cmap, 14-bit scale, returns None. New run_s12_reduction() helper. New Figure 3 dark comparison. Figs 3→4, 4→5 renumbered. figure_reduction_peaks takes pre-computed fp. |
