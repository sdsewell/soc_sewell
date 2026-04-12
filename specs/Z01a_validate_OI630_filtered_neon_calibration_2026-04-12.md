# Z01a — Validate OI 630 nm Filtered Neon Lamp Calibration Images (No Metadata Header)

**Spec ID:** Z01a
**Spec file:** `docs/specs/Z01a_validate_OI630_filtered_neon_calibration_2026-04-12.md`
**Project:** WindCube FPI Pipeline
**Institution:** NCAR / High Altitude Observatory (HAO)
**Status:** Specification — ready for implementation in VS Code
**Tier:** 9 — Validation Testing
**Depends on:** S12 (M03 — annular reduction, dark subtraction, peak finding),
S13 (Tolansky — SingleLineResult), S19 (P01 — ingest_real_image)
**Derived from:** Z01 (`Z01_validate_calibration_using_real_images_2026-04-12.md`)
**Used by:** Standalone OI-channel calibration; ε₀ output feeds zero-velocity
reference for M06 science inversion
**Last updated:** 2026-04-12
**Created/Modified by:** Claude AI

---

## 1. Purpose

`z01a_validate_OI630_filtered_neon_calibration_2026-04-12.py` is a
validation and calibration script for a specific class of images produced
by illuminating the FPI payload with an external neon lamp passed through a
630 nm bandpass filter.

### 1.1 What these images are

These are **calibration images of the OI 630 nm channel**, not science
airglow images. The optical configuration is:

```
Ne lamp → 630 nm bandpass filter → baffled field stop (1.65° full angle)
       → FPI etalon → imaging lens → CCD
```

The 630 nm filter isolates a single spectral family that mimics the OI
630.0 nm airglow emission line. The baffled field stop restricts the accepted
cone angle to 1.65° full angle (0.825° half-angle), matching the instrument's
nominal science FOV. As a result:

- **Only 6–7 fringes are visible** — fewer than the 20-fringe neon calibration
  images in Z01. This is correct and expected; it is not a failure condition.
- The signal level is considerably stronger than real airglow, but the fringe
  geometry is otherwise representative of a science observation.
- Because the neon lamp is a static ground-based source, the line-of-sight
  Doppler shift is **zero by construction**. These images therefore provide a
  zero-velocity reference for the OI channel.

### 1.2 What this script does

The images have had the 1-row binary metadata header removed and are therefore
a slightly different shape from the standard P01 format (typically 256×256 px
rather than 259×276 px). No `ImageMetadata` can be extracted.

The pipeline chain is:

```
Load headerless images → Display image pair (no metadata) →
Select ROI + inspect → Annular reduction (S12) →
Dark subtraction comparison → Radial profile (6–7 peaks) →
Single-line Tolansky (S13) → ε₀, λ_c, v_ref figure
```

### 1.3 Calibration output — ε₀

The primary new product from Z01a (beyond what Z01 already provides) is
`ε₀` — the **rest-frame fractional interference order** at OI 630.0 nm.
Because `v_rel = 0` by construction, the recovered fractional order `ε` from
this image IS `ε₀`:

```
ε₀ = ε_recovered   (valid only when source is at rest)
```

`ε₀` becomes the zero-velocity reference for all subsequent OI science
frames. For a science frame with an unknown wind, the velocity is:

```
v_rel = c · (ε_sci − ε₀) · (λ_OI / (2d · ∂ε/∂λ))
```

Z01a prints `ε₀` to the console and displays it prominently in Figure 5.

### 1.4 Relationship to Z01

Z01 uses a two-line Ne lamp (640.2248 nm + 638.2991 nm) to solve for `d`
and `f` as free parameters. Z01a uses those recovered values as fixed priors,
then solves only for `ε` at OI 630.0 nm. The two scripts are complementary
steps in a complete instrument characterisation sequence:

```
Z01 (2-line Ne, 640/638 nm)  →  d, f, α  (opto-mechanical constants)
Z01a (filtered Ne at 630 nm) →  ε₀        (OI-channel zero-velocity reference)
M06 (science airglow frames) →  ε_sci → v_rel = f(ε_sci, ε₀, d, f)
```

---

## 2. Differences from Z01 — summary

| Aspect | Z01 | Z01a |
|--------|-----|------|
| Image format | `.bin` 260×276 (incl. 1-row header) or `.npy` | `.npy` or `.bin` **without** header row |
| Image shape (typical) | 259×276 px | 256×256 px (or as supplied) |
| Embedded metadata | `ImageMetadata` via `ingest_real_image()` | None — header absent |
| Stage B | Extracts `ImageMetadata` | Always returns `None/None` |
| Fringe source | Ne 640.2248 nm + Ne 638.2991 nm (2 families) | Ne lamp → 630 nm filter → baffled FOV (1 family) |
| Expected peak count | 20 (10 per Ne family) | 6–7 (single family, 1.65° FOV limit) |
| Profile beat pattern | Yes — amplitude envelope from two Ne lines | No — uniform amplitude, single family |
| Doppler shift | N/A (calibration source, no v_rel output) | Zero by construction — static Ne lamp |
| Tolansky algorithm | Two-line (`TolanskyPipeline` / `TwoLineResult`) | Single-line (`SingleLineTolansky` / `SingleLineResult`) |
| Tolansky free parameters | `d`, `f`, `ε₁`, `ε₂` | `ε` only — `d` and `f` fixed priors |
| Primary new output | `d`, `f`, `α` | `ε₀` (rest-frame fractional order at OI 630 nm) |
| Figure 5 title | Two r² plots + parameter table | Single r² plot + ε₀ / v_ref table |

---

## 3. Script structure overview

```
Stage A  load_images()              — file dialogs; headerless .bin or .npy
Stage B  extract_metadata()         — always returns None/None
Stage C  figure_image_pair()        — Figure 1: images + "no metadata" table + click-to-seed
Stage D  figure_roi_inspection()    — Figure 2: ROI images + histograms (2×2)  [identical to Z01]
Stage F  run_s12_reduction()        — no figure: S12 reduction helper           [identical to Z01]
Stage F  figure_dark_comparison()   — Figure 3: dark subtraction check          [identical to Z01]
Stage F  figure_reduction_peaks()   — Figure 4: r² profile + 6–7 single-family peaks
Stage G  figure_tolansky_1line()    — Figure 5: single-line Tolansky → ε₀, λ_c, v_ref
```

---

## 4. Stage A — `load_images()` (Z01a variant)

### 4.1 Differences from Z01

- Supports `.npy` and headerless `.bin` files. For `.bin`, a new helper
  `load_headerless_bin(path, shape)` reads the raw pixel data with no
  header row (see §4.3).
- Does **not** call `ingest_real_image()` — no header row is present.
- `cal_type` / `dark_type` are `'headerless'` for `.bin`, `'synthetic'`
  for `.npy`.

### 4.2 Function signature

```python
def load_images(image_shape: tuple[int, int] = (256, 256)) -> dict:
    """
    Load a headerless calibration image and dark image.

    Parameters
    ----------
    image_shape : (rows, cols) expected shape. Used only for .bin files.
                  Default (256, 256). Supply explicitly if images differ.

    Returns
    -------
    dict with keys:
        'cal_image'  : np.ndarray, float64, shape image_shape
        'dark_image' : np.ndarray, float64, shape image_shape
        'cal_path'   : pathlib.Path
        'dark_path'  : pathlib.Path
        'cal_type'   : str, 'headerless' | 'synthetic'
        'dark_type'  : str, 'headerless' | 'synthetic'
        'cal_raw'    : np.ndarray uint16, shape image_shape, or None for .npy
        'dark_raw'   : np.ndarray uint16 or None
    """
```

### 4.3 `load_headerless_bin()` helper

```python
def load_headerless_bin(path: pathlib.Path, shape: tuple[int, int]) -> np.ndarray:
    """
    Load a headerless WindCube .bin file as a 2D float64 array.

    The file contains shape[0] × shape[1] big-endian uint16 values with no
    header row. Total expected file size = shape[0] * shape[1] * 2 bytes.

    Raises ValueError if the file size does not match the expected shape.
    """
    data = np.frombuffer(path.read_bytes(), dtype='>u2').astype(np.float64)
    expected = shape[0] * shape[1]
    if data.size != expected:
        raise ValueError(
            f"load_headerless_bin: expected {expected} pixels "
            f"({shape[0]}×{shape[1]}) but got {data.size} "
            f"in {path.name}"
        )
    return data.reshape(shape)
```

### 4.4 Shape validation

Validate `cal_image.shape == dark_image.shape`. If not, raise `ValueError`.
Do not enforce any absolute shape — these images may vary.

---

## 5. Stage B — `extract_metadata()` (Z01a variant)

Always returns `{'cal_meta': None, 'dark_meta': None}`. No attempt is made
to parse metadata from either file type.

```python
def extract_metadata(load_result: dict) -> dict:
    """
    Z01a: no metadata header present in any input file.
    Always returns None for both cal and dark.
    """
    return {'cal_meta': None, 'dark_meta': None}
```

---

## 6. Stage C — `figure_image_pair()` (Z01a variant)

### 6.1 Differences from Z01

- Both metadata table panels display:
  `"No embedded metadata — header row absent"`.
- Figure suptitle identifies the source:
  `f"Figure 1 — OI 630 nm Filtered Neon Lamp Calibration  |  {directory_str}"`
- All other behaviour — gray cmap, 14-bit scale, click-to-seed, 2×2 layout,
  return `(cx_seed, cy_seed)` — is **identical to Z01 §5**.

### 6.2 Metadata table implementation

```python
for ax_table in [ax_meta_cal, ax_meta_dark]:
    ax_table.axis('off')
    ax_table.table(
        cellText=[["No embedded metadata — header row absent"]],
        colLabels=["Status"],
        loc='center',
        cellLoc='center',
    )
```

---

## 7. Stages D, F (run_s12_reduction, figure_dark_comparison) — identical to Z01

These stages are **identical** to Z01 §§6–8 and receive images of a
potentially different shape — handled transparently since no fixed dimensions
are assumed.

---

## 8. Stage F — `figure_reduction_peaks()` (Figure 4, Z01a variant)

### 8.1 Differences from Z01

- **Expected peak count: 6–7.** This follows directly from the 1.65° full-
  angle FOV imposed by the baffled field stop. Fewer fringes fit within the
  reduced angular range. This is a correct, expected result — not a failure.
  Do not report "6/20 peaks found" or similar — there is no target of 20.
- **Single family only.** All detected peaks belong to the OI 630 nm family.
  There is no interleaved second spectral family, no amplitude classification,
  and no beat envelope in the profile. The amplitude envelope should be
  roughly uniform across the fringe pattern.
- **Peak overlay colour:** all peaks drawn in `'steelblue'` (single colour).
- **Peak table:** no "Family" column. Columns are:
  `#`, `r² centre (px²)`, `σ(r²) (px²)`, `2σ(r²) (px²)`,
  `Amplitude (ADU)`, `σ(amp) (ADU)`, `Fit OK`.

### 8.2 Title bar

```python
suptitle = (
    f"Figure 4 — Annular Reduction and Peak Identification\n"
    f"OI 630 nm filtered neon lamp  |  "
    f"File: {cal_path.name}  |  "
    f"Centre: ({fp.cx:.3f}, {fp.cy:.3f}) px  "
    f"[σ=({fp.sigma_cx:.3f}, {fp.sigma_cy:.3f}) px]  |  "
    f"r_max={fp.r_max:.1f} px  |  "
    f"Peaks found: {n_peaks_found} (expect 6–7)  [close to continue]"
)
```

---

## 9. Stage G — `figure_tolansky_1line()` (Figure 5, Z01a-specific)

### 9.1 Physical rationale

Because `d` and `f` are already known from Z01, the single OI 630 nm line
has only one free parameter: `ε`, the fractional interference order at the
OI rest wavelength. Since the neon lamp source is static, `v_rel = 0` by
construction. The recovered `ε` is therefore the **rest-frame fractional
order `ε₀`**, which becomes the zero-velocity reference for all future OI
science frames processed by M06.

This is the same algorithm M06 will apply to real science frames, but with
the ground truth that `v_rel = 0`. The check `λ_c ≈ OI_WAVELENGTH_NM` and
`v_rel ≈ 0 m/s` serves as a self-consistency validation of the `d` and `f`
priors recovered in Z01.

### 9.2 Function signature

```python
def figure_tolansky_1line(
    fp:       'FringeProfile',
    cal_path: pathlib.Path,
) -> 'SingleLineResult':
    """
    Figure 5: single-line Tolansky analysis on the OI 630 nm filtered neon
    lamp calibration fringe profile.

    Uses fixed priors d and f from windcube.constants (recovered from Z01).
    Fits only the fractional order ε. Since the source is at rest, the
    recovered ε is the zero-velocity reference ε₀ for the OI channel.

    Validates that v_rel ≈ 0 m/s (within measurement uncertainty).

    Returns
    -------
    SingleLineResult — dataclass including ε₀, λ_c, v_ref, σ_v, chi2_dof
    """
```

### 9.3 S13 single-line call

```python
from fpi.tolansky_2026_04_05 import SingleLineTolansky, SingleLineResult
from windcube.constants import (
    OI_WAVELENGTH_NM,       # 630.0 nm
    TOLANSKY_D_MM,          # 20.106 mm — fixed prior from Z01 two-line neon fit
    TOLANSKY_F_MM,          # 199.12 mm — fixed prior from Z01 two-line neon fit
    CCD_PIXEL_PITCH_M,
    ICOS_GAP_MM,
)

analyser = SingleLineTolansky(
    fringe_profile  = fp,
    lam_rest_nm     = OI_WAVELENGTH_NM,
    d_prior_m       = TOLANSKY_D_MM * 1e-3,
    f_prior_m       = TOLANSKY_F_MM * 1e-3,
    pixel_pitch_m   = CCD_PIXEL_PITCH_M,
    d_icos_m        = ICOS_GAP_MM * 1e-3,
)
result = analyser.run()
```

After the run, compute the zero-velocity consistency check:

```python
v_ref_check_pass = abs(result.v_rel_ms) < 3 * result.sigma_v_ms
print(f"  Zero-velocity check: v_rel = {result.v_rel_ms:.1f} ± {result.sigma_v_ms:.1f} m/s  "
      f"({'PASS' if v_ref_check_pass else 'WARN — non-zero velocity detected'})")
print(f"  ε₀ (rest-frame fractional order) = {result.epsilon:.6f} ± {result.sigma_eps:.6f}")
```

### 9.4 Figure layout

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Figure 5: Single-Line Tolansky — OI 630 nm Zero-Velocity Reference     │
├──────────────────────────────────────────────────────────────────────────┤
│  ax_scatter (~60% height)                                                │
│  r² vs fringe index — data (circles) + WLS fit line                     │
│  Residuals inset (lower right)                                           │
├──────────────────────────────────────────────────────────────────────────┤
│  ax_table (~40% height)                                                  │
│  Results parameter table (see §9.6)                                      │
└──────────────────────────────────────────────────────────────────────────┘
```

Use `fig, (ax_scatter, ax_table) = plt.subplots(2, 1, figsize=(14, 10),
gridspec_kw={'height_ratios': [3, 2]})`.

### 9.5 r² scatter plot

- Plot measured r² values vs fringe index `p` as open circles with error
  bars. Marker: `'o'`, `markersize=6`, `mfc='white'`, `mec='steelblue'`.
- Overplot the WLS best-fit line as a solid steelblue line.
- Label: `f"OI 630.0 nm  (λ_rest = {OI_WAVELENGTH_NM:.4f} nm)  —  zero-velocity reference"`.
- X-axis label: `"Fringe index p"`
- Y-axis label: `"r² (px²)"`
- Text box (upper left, inside axes):
  ```
  S   = {result.S:.2f} ± {result.sigma_S:.2f} px²/fringe
  ε₀  = {result.epsilon:.6f} ± {result.sigma_eps:.6f}
  R²  = {r2_fit:.6f}
  ```
  Use `ax.text(0.03, 0.95, ..., transform=ax.transAxes, va='top',
  fontsize=9, bbox=dict(boxstyle='round', fc='lightyellow'))`.

### 9.6 Residuals inset

Add an inset axes (`ax_scatter.inset_axes([0.55, 0.05, 0.42, 0.3])`)
showing normalised residuals `(r²_obs − r²_fit) / σ(r²)`. Should scatter
within ±3. Stem plot with dashed zero line. Y-label: `"Residual (σ)"`.

### 9.7 Results parameter table

| Parameter | Symbol | Value | 1σ | 2σ | Units | Notes |
|-----------|--------|-------|-----|-----|-------|-------|
| Rest fractional order | ε₀ | `result.epsilon` | `result.sigma_eps` | `result.two_sigma_eps` | — | zero-velocity reference for M06 |
| Calibrated λ | λ_c | `result.lam_c_nm` | `result.sigma_lam_c_nm` | `result.two_sigma_lam_c_nm` | nm | should equal OI_WAVELENGTH_NM |
| Reference velocity | v_ref | `result.v_rel_ms` | `result.sigma_v_ms` | `result.two_sigma_v_ms` | m/s | expect ≈ 0; source is at rest |
| Zero-v check | — | `PASS / WARN` | — | — | — | PASS if \|v_ref\| < 3σ_v |
| Integer order | N | `result.N_int` | — | — | — | resolved from d_prior |
| d (prior) | d | `result.d_prior_mm` | — | — | mm | fixed from Z01 |
| f (prior) | f | `result.f_prior_mm` | — | — | mm | fixed from Z01 |
| Reduced χ² | χ²/ν | `result.chi2_dof` | — | — | — | acceptable: 0.5–3.0 |

Format numbers:
- ε₀: **6** decimal places (extra precision — this value is stored as a
  calibration constant)
- λ_c: 5 decimal places (nm)
- v_ref: 1 decimal place (m/s)

### 9.8 Title bar

```python
v_check = "PASS" if abs(result.v_rel_ms) < 3 * result.sigma_v_ms else "WARN"
suptitle = (
    f"Figure 5 — Single-Line Tolansky  |  OI 630 nm zero-velocity reference\n"
    f"File: {cal_path.name}  |  "
    f"ε₀ = {result.epsilon:.6f} ± {result.sigma_eps:.6f}  |  "
    f"λ_c = {result.lam_c_nm:.5f} nm  |  "
    f"v_ref = {result.v_rel_ms:.1f} ± {result.sigma_v_ms:.1f} m/s  "
    f"[{v_check}]  |  [close to exit]"
)
```

---

## 10. Main function

```python
def main():
    print("=" * 70)
    print("  WindCube FPI — Z01a OI 630 nm Filtered Neon Lamp Calibration")
    print("  (Zero-velocity reference — single-line Tolansky)")
    print("=" * 70)

    # Stage A
    print("\nStage A: Loading headerless images...")
    load = load_images()
    print(f"  CAL:  {load['cal_path'].name}  shape={load['cal_image'].shape}  ({load['cal_type']})")
    print(f"  DARK: {load['dark_path'].name}  shape={load['dark_image'].shape}  ({load['dark_type']})")

    # Stage B
    print("\nStage B: Metadata extraction (none expected — header absent)...")
    meta = extract_metadata(load)

    # Stage C — Figure 1
    print("\nStage C: Displaying image pair (Figure 1)...")
    cx_seed, cy_seed = figure_image_pair(load, meta)
    print(f"  Fringe centre seed: ({cx_seed:.1f}, {cy_seed:.1f}) px")

    # Stage D — Figure 2
    print("\nStage D: ROI inspection (Figure 2)...")
    figure_roi_inspection(load['cal_image'], load['dark_image'], cx_seed, cy_seed)

    # Stage F-1: S12 reduction
    print("\nStage F-1: S12 annular reduction...")
    roi = {'cx_seed': cx_seed, 'cy_seed': cy_seed}
    fp, s12_dark_sub = run_s12_reduction(
        load['cal_image'], load['dark_image'], cx_seed, cy_seed, roi
    )

    # Stage F-2: dark subtraction comparison (Figure 3)
    print("\nStage F-2: Dark subtraction comparison (Figure 3)...")
    roi_slice = (
        slice(roi['roi_y0'], roi['roi_y1']),
        slice(roi['roi_x0'], roi['roi_x1']),
    )
    figure_dark_comparison(
        load['cal_image'][roi_slice],
        load['dark_image'][roi_slice],
        s12_dark_sub,
        roi_slice,
    )

    # Stage F-3: radial profile (Figure 4)
    print("\nStage F-3: Radial fringe profile (Figure 4) — expect 6–7 peaks...")
    figure_reduction_peaks(fp, roi, load['cal_path'])

    # Stage G: single-line Tolansky (Figure 5)
    print("\nStage G: Single-line Tolansky — OI 630 nm zero-velocity reference (Figure 5)...")
    result = figure_tolansky_1line(fp, load['cal_path'])

    print("\n" + "=" * 70)
    print("  Z01a complete — zero-velocity calibration summary:")
    print(f"    ε₀    = {result.epsilon:.6f} ± {result.two_sigma_eps:.6f}  (2σ)  ← store as calibration constant")
    print(f"    λ_c   = {result.lam_c_nm:.5f} ± {result.two_sigma_lam_c_nm:.5f} nm (2σ)")
    print(f"    v_ref = {result.v_rel_ms:.1f} ± {result.two_sigma_v_ms:.1f} m/s (2σ)  ← expect ≈ 0")
    v_check = abs(result.v_rel_ms) < 3 * result.sigma_v_ms
    print(f"    Zero-velocity check: {'PASS' if v_check else 'WARN — investigate'}")
    print(f"    N_int   = {result.N_int}")
    print(f"    d_prior = {result.d_prior_mm:.3f} mm  (fixed)")
    print(f"    f_prior = {result.f_prior_mm:.3f} mm  (fixed)")
    print(f"    χ²/ν    = {result.chi2_dof:.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
```

---

## 11. Dark subtraction architecture

Identical to Z01 §12. The absence of a metadata header has no impact on
dark subtraction logic, which operates entirely on the pixel arrays.

---

## 12. Key physical parameters

All constants imported from `windcube.constants` (S03).

### 12.1 General constants

| Symbol | Value | Source | Notes |
|--------|-------|--------|-------|
| OI_WAVELENGTH_NM | 630.0 | S03 / NIST | OI 630.0 nm rest wavelength |
| TOLANSKY_D_MM | 20.106 | S03 / Z01 output | Fixed prior — etalon gap (see §12.2) |
| TOLANSKY_F_MM | 199.12 | S03 / Z01 output | Fixed prior — focal length |
| CCD_PIXEL_PITCH_M | 32e-6 | S03 | 2×2 binned pixel pitch |
| ICOS_GAP_MM | 20.008 | S03 | ICOS mechanical measurement (see §12.2) |

### 12.2 Etalon gap — consolidated measurements

Four independent estimates of the etalon gap `d` exist. They are not all
consistent and the discrepancy between the mechanical and Tolansky values
is an open item.

| Label | Value (mm) | Source | Status |
|-------|-----------|--------|--------|
| `ICOS_GAP_MM` | 20.008 000 000 | ICOS mechanical spacer measurement | Used only to resolve FSR integer N_int |
| Pre-load compression | −0.000 070 803 | Pat & Nir clamping compression (70.803 nm) | Applied to ICOS to give 25°C estimate |
| `D_25C_MM` | 20.007 929 197 | ICOS − pre-load; best physical estimate at 25°C | Reference for thermal model (§12.3) |
| Delaney fit | 20.000 000 028 | Alfred's code fit to FlatSat data | ~8 µm below ICOS; provenance under review |
| `TOLANSKY_D_MM` | 20.106 | Z01 two-line neon Tolansky fit | **+98 µm above D_25C_MM — discrepancy unresolved** |

**Open item:** The Tolansky-recovered gap (20.106 mm) is ~98 µm larger than
the 25°C mechanical estimate (20.007929 mm). Possible causes include: the
FlatSat etalon assembly differing from the ICOS-measured spacer, or an
unmodelled systematic in the Tolansky fit. `TOLANSKY_D_MM` is used as the
Tolansky prior (self-consistent with the Z01 neon fit) until this discrepancy
is resolved. `D_25C_MM` is recorded for reference but is **not** used as
the Tolansky prior.

### 12.3 Etalon thermal model

The etalon gap varies with temperature according to the measured Zerodur
spacer thermal expansion coefficient.

| Parameter | Value | Notes |
|-----------|-------|-------|
| Thermal expansion coefficient | 18.585 nm/°C | Measured; Zerodur spacer |
| Reference temperature | 25°C | Etalon heater setpoint |
| Reference gap `D_25C_MM` | 20.007 929 197 mm | Best estimate at setpoint |
| Heater control band | ±0.1°C | Expected in-orbit stability |
| d variation at ±0.1°C | ±1.86 nm | Negligible — < 0.001 fringe |
| Worst-case orbital excursion | ±2°C | Etalon heater failure bound |
| d variation at ±2°C | ±37.2 nm | ~0.02 fringe; < 1 m/s velocity error |

The thermal contribution to systematic velocity error is negligible under
normal heater operation. Even at the ±2°C worst-case bound the d variation
of ±37 nm is small relative to the ~98 µm Tolansky/mechanical discrepancy.

For reference, the gap as a function of temperature offset ΔT from 25°C:

```
d(T) = D_25C_MM + 18.585e-6 * ΔT   [mm,  ΔT in °C]
```

**Calibration sequence dependency:** `TOLANSKY_D_MM` and `TOLANSKY_F_MM`
must be updated in `windcube.constants` if a new Z01 run produces improved
values. Z01a always reads the current values from constants at runtime —
there is no hardcoded fallback.

---

## 13. Verification tests

Manual checks only — Z01a is fully interactive.

| Check | Figure | Criterion |
|-------|--------|-----------|
| V1 | Fig 1 | Images load; both metadata panels show "No embedded metadata — header row absent" |
| V2 | Fig 1 | Image shapes match; no shape mismatch error |
| V3 | Fig 2 | Cal ROI shows 6–7 concentric fringe rings; dark ROI is uniformly low |
| V4 | Fig 2 | No saturation in cal ROI ADU histogram |
| V5 | Fig 3 | Suptitle reads "IDENTICAL (max \|diff\| = 0)" |
| V6 | Fig 4 | 6–7 peaks found; all `steelblue` (no amplitude family classification); no "Family" column in table |
| V7 | Fig 4 | Profile amplitude envelope is roughly uniform — no beat modulation |
| V8 | Fig 5 | R² > 0.9999 for Tolansky line fit |
| V9 | Fig 5 | Zero-velocity check: `PASS` — \|v_ref\| < 3σ_v (source is static Ne lamp) |
| V10 | Fig 5 | λ_c within 0.005 nm of 630.0 nm |
| V11 | Fig 5 | χ²/ν in range [0.5, 3.0] |
| V12 | Console | ε₀ printed to 6 decimal places with "← store as calibration constant" |

**V9 is the key new check.** A WARN result (non-zero detected velocity)
indicates either: (a) `d` or `f` priors need updating from a fresh Z01 run,
(b) a fringe centre seed error that biased the annular reduction, or (c) a
genuine instrument alignment shift. Investigate before using ε₀ as a
calibration reference.

---

## 14. Expected output (typical filtered neon lamp image)

| Quantity | Expected |
|----------|----------|
| Image shape | 256×256 px (or as supplied) |
| Embedded metadata | None |
| Peaks found | 6–7 |
| Profile envelope | Uniform — no beat modulation |
| ε₀ | ~0.3–0.7 (exact value depends on etalon temperature) |
| λ_c | 630.000 ± 0.002 nm |
| v_ref | −20 to +20 m/s (within measurement noise; expect ≈ 0) |
| Zero-velocity check | PASS |
| χ²/ν | ~1.0–2.0 |

---

## 15. File location in repository

```
soc_sewell/
├── validation/
│   ├── z01_validate_calibration_using_real_images_2026-04-09.py
│   └── z01a_validate_OI630_filtered_neon_calibration_2026-04-12.py
└── docs/specs/
    ├── Z01_validate_calibration_using_real_images_2026-04-12.md
    └── Z01a_validate_OI630_filtered_neon_calibration_2026-04-12.md
```

---

## 16. Dependencies

```
numpy       >= 1.24
scipy       >= 1.10
matplotlib  >= 3.7
tkinter                # standard library
pathlib                # standard library
```

Internal imports:
```python
from fpi.m03_annular_reduction_2026_04_05 import (
    reduce_calibration_frame,
    make_master_dark,
    subtract_dark,
    FringeProfile,
)
from fpi.tolansky_2026_04_05 import SingleLineTolansky, SingleLineResult
from windcube.constants import (
    OI_WAVELENGTH_NM,
    TOLANSKY_D_MM,
    TOLANSKY_F_MM,
    CCD_PIXEL_PITCH_M,
    ICOS_GAP_MM,
)
```

`TwoLineResult`, `TolanskyPipeline`, and `ImageMetadata` are not used in
Z01a.

---

## 17. Instructions for Claude Code

1. Read this spec (Z01a), Z01 (2026-04-12), S12 (M03), S13 (Tolansky), and
   S19 (P01) in full before writing any code.

2. Confirm M03 and Tolansky tests pass:
   ```bash
   pytest tests/test_m03_annular_reduction_*.py -v
   pytest tests/test_tolansky_*.py -v
   ```
   Stop and report any failures.

3. Create:
   `validation/z01a_validate_OI630_filtered_neon_calibration_2026-04-12.py`

   Do **not** modify the Z01 script.

4. Module-level docstring:
   ```python
   """
   Z01a — Validate OI 630 nm Filtered Neon Lamp Calibration Images
   WindCube FPI Pipeline — NCAR / High Altitude Observatory (HAO)
   Spec: docs/specs/Z01a_validate_OI630_filtered_neon_calibration_2026-04-12.md
   Derived from: Z01 (Z01_validate_calibration_using_real_images_2026-04-12.md)
   Source type: Ne lamp + 630 nm filter + 1.65° baffled field stop
   Zero-velocity reference: v_rel = 0 by construction
   Tool: Claude Code
   Last updated: 2026-04-12
   """
   ```

5. Implement Z01a-specific functions from scratch per this spec:
   `load_headerless_bin`, `load_images`, `extract_metadata`,
   `figure_image_pair`, `figure_reduction_peaks`, `figure_tolansky_1line`,
   `main`.

6. For unchanged stages (D, F-1, F-2), import from Z01 or copy verbatim
   with comment `# unchanged from Z01`.

7. Confirm `SingleLineTolansky` exists in `fpi/tolansky_*.py` before
   calling it. If absent, stop and report.

8. **Peak count expectation:** the script must NOT flag 6–7 peaks as a
   failure. The title bar reads `"Peaks found: N (expect 6–7)"` regardless
   of N. Only fewer than 4 peaks should trigger a console warning.

9. **Zero-velocity check:** print `PASS` or `WARN` to the console as shown
   in §9.3. A `WARN` should not abort the script — it is informational.

10. **ε₀ output:** the console final summary must print ε₀ to 6 decimal
    places with the annotation `"← store as calibration constant"` as shown
    in §10.

11. Run the script on a real or synthetic headerless OI image pair. Confirm:
    - Fig 1 metadata panels show "No embedded metadata — header row absent"
    - Fig 4 shows 6–7 peaks in `steelblue`, no "Family" column
    - Fig 5 title shows `PASS` for zero-velocity check
    - Console prints ε₀ to 6 decimal places

    Report back with the full console output of a successful run.

---

## 18. Change log

| Version | Date | Changes |
|---------|------|---------|
| v0.1 | 2026-04-12 | Initial specification. Source correctly identified as Ne lamp + 630 nm filter + 1.65° baffled field stop (not raw airglow). Zero-Doppler reference characterisation added. ε₀ as primary new calibration output. Expected peak count 6–7 (not variable). Zero-velocity consistency check (PASS/WARN) added to Fig 5 and console. Verification check V9 tightened from \|v\|<1500 m/s to \|v\|<3σ_v. |
| v0.2 | 2026-04-12 | §12 expanded with four-source etalon gap table (ICOS 20.008000 mm, Pat & Nir pre-load −70.803 nm, D_25C 20.007929197 mm, Delaney fit 20.000000028 mm, Tolansky 20.106 mm) and open-item flag on 98 µm Tolansky/mechanical discrepancy. Thermal model added: 18.585 nm/°C coefficient; ±0.1°C heater control → ±1.86 nm gap variation (negligible); ±2°C worst-case orbital excursion → ±37.2 nm. Corrected from prior erroneous ±60°C / ±1.115 µm figures. |
