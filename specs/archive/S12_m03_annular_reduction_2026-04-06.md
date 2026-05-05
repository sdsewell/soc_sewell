# S12 — M03 Fringe Centre Finding and Annular Reduction Specification

**Spec ID:** S12
**Spec file:** `docs/specs/S12_m03_annular_reduction_2026-04-06.md`
**Project:** WindCube FPI Pipeline
**Institution:** NCAR / High Altitude Observatory (HAO)
**Status:** Specification — ready for implementation in VS Code
**Depends on:** S01, S02, S03, S04, S09 (M01), S10 (M02), S11 (M04)
**Used by:**
  - S13 (Tolansky) — receives `FringeProfile.peak_fits` as input
  - S14 (M05) — receives calibration `FringeProfile`
  - S15 (M06) — receives science `FringeProfile`
  - S17 (INT02) — exercises full reduction pipeline
**References:**
  - Harding et al. (2014) Applied Optics 53(4), Section 3
  - Niciejewski et al. (1992) SPIE 1745 — r² reduction, software aperture
  - Mulligan (1986) J. Phys. E 19, 545 — software aperture technique
**Last updated:** 2026-04-06

> **Revision note vs earlier draft:** Three additions:
> (1) **Hot-pixel pre-clip** — image clipped to 99.5th percentile before
>     the azimuthal variance cost function to prevent saturated pixels
>     biasing the centre. Confirmed necessary on the real WindCube FlatSat
>     calibration binary image.
> (2) **Integrated peak finding** — `annular_reduce()` calls
>     `_find_and_fit_peaks()` on the completed 1D profile and stores
>     results in `FringeProfile.peak_fits`. Essential for S13 (Tolansky),
>     which consumes 20 peak radii (10 per Ne line). Adaptive window
>     clamping and a physics-grounded distance floor are required to
>     correctly find all 20/20 peaks on real FlatSat data.
> (3) **Dark frame subtraction** (this revision) — a master dark frame
>     must be subtracted from every raw image before any other processing.
>     This step is the first operation in both `reduce_calibration_frame()`
>     and `reduce_science_frame()`. See Section 3 (new) for the full
>     specification.

---

## 1. Purpose

M03 reduces a 2D CCD fringe image to a 1D radial intensity profile and
simultaneously locates and fits all fringe peaks for the Tolansky analysis.

Four sequential responsibilities, in strict order:

0. **Dark subtraction** — subtract the master dark frame from the raw image
   before any optical analysis. This is the first operation and must never
   be skipped or reordered.

1. **Centre finding** — locate `(cx, cy)` to sub-pixel accuracy.
   Hot-pixel pre-clip + two-pass azimuthal variance minimisation.

2. **Annular reduction** — r²-binned mean intensity and SEM.

3. **Peak finding** — Gaussian fit to each fringe peak. Results stored
   in `FringeProfile.peak_fits` for consumption by S13 (Tolansky).

**API separation:**
- `reduce_calibration_frame()` — subtracts dark AND finds centre AND reduces AND finds peaks.
- `reduce_science_frame()` — subtracts dark, uses provided centre, reduces only.
  No peak finding. `peak_fits = []`.

---

## 2. Why naive centre-finding approaches fail on WindCube data

Empirically confirmed on real FlatSat frames:

**Radial symmetry transform:** Vote scale ~10 px; FPI ring pixels at
r = 60 px need to vote 60 px inward. Overscan gradient dominates → 100+ px errors.

**Airy model nonlinear LSQ as centre finder:** Stray light makes the
brightest feature a ring at r ≈ 8–20 px, not the zeroth-order disc.
Fitter converges to a spurious minimum 1–2 px from truth.

**Single-pass Nelder-Mead:** Variance minimum is ~1 px wide. A large
initial simplex steps over it → local plateau convergence.

**Hough circles:** Clipped outer ring → fits the clipped arc, not the
common centre. Errors 50–100 px confirmed on FlatSat.

**Conclusion:** Human seed + two-pass azimuthal variance minimisation
is the only robust approach. Stage 1 CoM is a cross-check, not the
primary seed.

---

## 3. Dark frame subtraction

### 3.1 Physical motivation

Dark current accumulates in every CCD pixel during an exposure at a rate
that is strongly temperature-dependent (`CCD_DARK_RATE_E_PX_S` from S03).
It is spatially non-uniform — warm pixels, column defects, and thermal
gradients across the detector all produce structure in the dark frame. If
not removed, this structure contaminates both the centre-finding step
(the azimuthal variance cost function will be biased toward dark-current
gradients rather than the fringe centre) and the annular reduction profile
(introducing a spurious radial pedestal that distorts the fringe baseline
and degrades the chi² of the S14 Airy fit).

Dark subtraction is the **first** operation. It must precede the hot-pixel
clip, centre finding, and annular reduction.

### 3.2 Master dark frame construction

On orbit, the operations sequence (WC-SE-0003 §5) acquires 1–5 dark frames
with the shutter closed and neon lamp off immediately before or after the
calibration exposures. These frames are co-added into a master dark:

```python
def make_master_dark(dark_frames: list[np.ndarray]) -> np.ndarray:
    """
    Construct a master dark frame by median-combining a list of dark images.

    Parameters
    ----------
    dark_frames : list of np.ndarray
        Each array is a single dark exposure, same shape and dtype as the
        science/calibration image. Minimum 1 frame; median of 1 is itself.
        dtype is typically uint16 (raw CCD counts, Level 0).

    Returns
    -------
    np.ndarray, float64
        Master dark frame, same spatial shape as input frames.
        Returned as float64 to allow unbiased subtraction from float64 images.

    Notes
    -----
    Median combination is preferred over mean because it is robust against
    cosmic ray hits in individual dark frames. With 1–5 frames the gain
    in cosmic-ray rejection is modest, but median is correct by construction
    and costs nothing.

    If dark_frames contains exactly 1 frame, the median is that frame
    converted to float64 (no change in values).

    The master dark is NOT normalised by exposure time here. The dark
    frames must already be taken at the same exposure time as the
    science/calibration frame they will be subtracted from. This is
    guaranteed by the on-orbit operations sequence (WC-SE-0003 §5, steps
    3 and 8 acquire darks at the same cadence as science exposures).
    """
    if len(dark_frames) == 0:
        raise ValueError("dark_frames must contain at least one frame")
    stack = np.stack([f.astype(np.float64) for f in dark_frames], axis=0)
    return np.median(stack, axis=0)
```

### 3.3 Dark subtraction

```python
def subtract_dark(
    image: np.ndarray,
    master_dark: np.ndarray,
    clip_negative: bool = True,
) -> np.ndarray:
    """
    Subtract master dark from a raw image and return a float64 result.

    Parameters
    ----------
    image : np.ndarray
        Raw CCD image (uint16 or float). Same spatial shape as master_dark.
    master_dark : np.ndarray, float64
        Master dark from make_master_dark(). Must match image shape exactly.
    clip_negative : bool
        If True (default), clip negative values to 0.0 after subtraction.
        Negative values can arise from readout noise in dark pixels; they
        are unphysical as photon counts and would bias the annular reduction
        mean toward negative values in low-signal bins. Default True.

    Returns
    -------
    np.ndarray, float64
        Dark-subtracted image. Shape identical to input.

    Raises
    ------
    ValueError
        If image.shape != master_dark.shape.

    Notes
    -----
    Conversion to float64 occurs before subtraction. The raw uint16 image
    is never modified in place.
    """
    if image.shape != master_dark.shape:
        raise ValueError(
            f"Image shape {image.shape} does not match "
            f"master dark shape {master_dark.shape}"
        )
    dark_subtracted = image.astype(np.float64) - master_dark
    if clip_negative:
        dark_subtracted = np.clip(dark_subtracted, 0.0, None)
    return dark_subtracted
```

### 3.4 Integration into the top-level API

Both `reduce_calibration_frame()` and `reduce_science_frame()` accept a
`master_dark` parameter. Dark subtraction is the first line of each function:

```python
# First line of reduce_calibration_frame() and reduce_science_frame():
if master_dark is not None:
    image = subtract_dark(image, master_dark, clip_negative=True)
else:
    image = image.astype(np.float64)   # still convert to float64 for consistency
```

**`master_dark=None` is permitted** for synthetic images from M02/M04, which
are already noise-free float64 arrays with no dark current. When `None` is
passed, M03 skips subtraction but still converts to float64. Tests T1–T7
that use synthetic M02/M04 images should pass `master_dark=None`.

### 3.5 What dark subtraction does and does not remove

**Removes:**
- Thermal dark current (temperature-dependent, ~400 e⁻/px/s at 20°C per S03)
- Fixed-pattern dark structure (warm pixels, column defects)
- Bias level (the CCD97 bias offset B is largely captured in the dark)

**Does not remove:**
- Readout noise (random, zero-mean — unaffected by subtraction)
- Hot pixels with shot noise above the median (removed by sigma-clip in annular_reduce)
- Stray light from the neon lamp or Sun (structure in the optical path, not the dark)

**Relationship to the bias term B in the Airy model (S14):** After dark
subtraction the residual bias in the fringe profile is small but non-zero
(dominated by readout noise floor). The `B` parameter in the M05 Airy fit
absorbs this residual. The S14 Stage 1 initial estimate `B_init ≈ 0.8 ×
percentile(profile, 5)` must be understood as the residual post-dark bias,
not the raw CCD bias. Document this explicitly in S14 FitConfig.

### 3.6 FringeProfile provenance fields for dark subtraction

Add two fields to `FringeProfile` (Section 12):

```python
dark_subtracted:    bool    # True if master_dark was provided and subtracted
dark_n_frames:      int     # number of dark frames used in master (0 if none)
```

These support diagnostic verification and data provenance tracing in the
Level 2 product (S20).

---

## 4. Binning convention — r² not r

Fringes are equally spaced in r² (Niciejewski 1992). All binning uses r²:

```
r²_edges[i] = (i / N_bins) · r²_max
r_centre[i] = sqrt(0.5 · (r²_edges[i] + r²_edges[i+1]))
```

---

## 5. Hot-pixel pre-clip

Before computing the variance cost, clip the image to its 99.5th percentile:

```python
p99_5 = float(np.percentile(image, 99.5))
image_for_cost = np.clip(image, None, p99_5)
```

**Used only for centre finding.** The full unclipped image is used for
the annular reduction profile. T5 verifies this separation explicitly.

---

## 6. Stage 1 — Coarse centre: intensity-weighted CoM

```python
def coarse_centre_com(
    image:             np.ndarray,
    overscan_left:     int   = 8,    # 2×2 binned — NOT 1×1 value (15 px)
    overscan_bottom:   int   = 4,    # 2×2 binned — NOT 1×1 value (30 px)
    coarse_top_pct:    float = 0.5,
    coarse_inner_frac: float = 0.45,
) -> tuple[float, float] | tuple[None, None]:
    """
    Intensity-weighted CoM of the brightest pixels in the inner 45% of frame.
    Returns (cx_coarse, cy_coarse) or (None, None) on failure (STAGE1_FAILED).
    """
```

---

## 7. Human prior and seed resolution

```python
def resolve_seed(
    cx_human:   float | None,
    cy_human:   float | None,
    cx_com:     float | None,
    cy_com:     float | None,
    cx_history: float | None,
    cy_history: float | None,
    image_size: int,
    disagreement_threshold_px: float = 5.0,
) -> tuple[float, float, str, int]:
    """
    Priority: human → history → CoM → geometric centre (last resort).
    Returns (cx_seed, cy_seed, seed_source, quality_flags_partial).
    """
```

---

## 8. Stage 2 — Two-pass azimuthal variance minimisation

### 7.1 Variance cost function

```python
def _variance_cost(
    cx: float, cy: float,
    image: np.ndarray,    # pre-clipped to 99.5th percentile
    r_min_sq: float, r_max_sq: float,
    n_var_bins: int,
) -> float:
    """
    Sum of per-bin intensity variance over the annular region.
    Uses np.bincount — no Python loop over bins.
    Minimum at the true optical axis regardless of fringe amplitude.
    """
```

### 7.2 Two-pass algorithm (mandatory)

```python
def azimuthal_variance_centre(
    image:         np.ndarray,   # pre-clipped to 99.5th percentile
    cx_seed:       float,
    cy_seed:       float,
    var_r_min_px:  float = 5.0,
    var_r_max_px:  float = None,
    var_n_bins:    int   = 250,
    var_search_px: float = 15.0,
) -> tuple[float, float, float, float, float, float]:
    """
    Pass 1: coarse grid, spacing max(2.0, var_search_px/8), ±var_search_px.
    Pass 2: Nelder-Mead from Pass 1 minimum.
            xatol=0.02, fatol=1.0, maxiter=500.
    Returns (cx_fine, cy_fine, cost_min, grid_cx, grid_cy, grid_cost).
    Single-pass Nelder-Mead MUST NOT be used — it fails T2.
    """
```

---

## 9. Centre uncertainty estimation

```python
def estimate_centre_uncertainty(
    cx: float, cy: float,
    cost_fn: callable,    # closure capturing image_for_cost, r bounds, n_bins
    delta_px: float = 0.5,
) -> tuple[float, float]:
    """
    σ_cx = sqrt(2 / |d²C/dcx²|) from finite-difference Hessian.
    Clamped to [0.02, 5.0] px. Returns (sigma_cx, sigma_cy).
    """
```

---

## 10. Top-level centre API

```python
@dataclass
class CentreResult:
    cx:           float
    cy:           float
    sigma_cx:     float
    sigma_cy:     float
    two_sigma_cx: float
    two_sigma_cy: float
    cost_at_min:  float
    grid_cx:      float
    grid_cy:      float
    grid_cost:    float


def find_centre(
    image:        np.ndarray,
    cx_seed:      float = None,
    cy_seed:      float = None,
    var_r_min_px: float = 5.0,
    var_r_max_px: float = None,
    var_n_bins:   int   = 250,
    var_search_px: float = 15.0,
) -> CentreResult:
    """
    Two-pass azimuthal variance centre finder with 99.5th-percentile
    hot-pixel clip applied to the cost function only.
    Seeds default to image geometric centre if not provided.
    """
```

---

## 11. Peak finding

### 10.1 Why peak finding must be adaptive

The calibration profile contains 20 interleaved fringe peaks (10 per Ne
line). Two failure modes affect naive approaches:

**Good-bin compression:** `find_peaks` counts only unmasked bins. When
masked bins lie between real peaks the apparent separation is smaller
than the distance threshold → real peaks suppressed. Fix: physics-grounded
`min_sep_px` floor converts minimum physical ring separation to safe bin
distance.

**Window engulfment:** Default fitting window engulfs adjacent peaks →
`curve_fit` locks onto the flanking peak. Fix: adaptive half-window clamped
to `floor((nearest_neighbour_separation − 1) / 2)`.

Both fixes are required to achieve 20/20 peaks on real FlatSat data.

### 10.2 PeakFit dataclass

```python
@dataclass
class PeakFit:
    peak_idx:       int    # bin index
    r_raw_px:       float  # r_grid at detected bin (px)
    profile_raw:    float  # profile value at detected bin (ADU)
    r_fit_px:       float  # Gaussian centroid (px); falls back to r_raw if failed
    sigma_r_fit_px: float  # 1σ uncertainty on centroid (px); nan if failed
    amplitude_adu:  float  # Gaussian amplitude above background (ADU)
    width_px:       float  # Gaussian sigma (px); nan if failed
    fit_ok:         bool
```

### 10.3 `_find_and_fit_peaks`

```python
def _find_and_fit_peaks(
    r_grid:          np.ndarray,
    profile:         np.ndarray,
    sigma_profile:   np.ndarray,
    masked:          np.ndarray,
    distance:        int   = 5,
    prominence:      float = 100.0,
    fit_half_window: int   = 8,     # upper bound; adaptive clamp is effective value
    min_sep_px:      float = 3.0,
) -> list[PeakFit]:
    """
    Locate peaks and fit a Gaussian to each.

    1. Compute physics-grounded distance floor:
       median_dr_px = median spacing between good bins (px)
       safe_distance = max(1, floor(min_sep_px / median_dr_px))
       effective_distance = min(distance, safe_distance)

    2. Run find_peaks on good-bin subset only.

    3. For each peak:
       left_sep  = bin distance to left neighbour
       right_sep = bin distance to right neighbour
       adaptive_hw = max(2, (min(left_sep, right_sep) - 1) // 2)
       effective_hw = min(fit_half_window, adaptive_hw)

    4. Gaussian fit with SEM weights (absolute_sigma=True).
       Background B0 = 20th percentile of window (not minimum).
       Bounds: A > 0, mu within window, sig > 0.3*median_dr_px.

    Returns list[PeakFit] sorted by r_raw_px.
    """
```

---

## 12. Annular reduction

```python
def annular_reduce(
    image:                  np.ndarray,   # already dark-subtracted, float64
    cx:                     float,
    cy:                     float,
    sigma_cx:               float,
    sigma_cy:               float,
    r_min_px:               float = 0.0,
    r_max_px:               float = 128.0,
    n_bins:                 int   = 150,
    n_subpixels:            int   = 1,
    sigma_clip_threshold:   float = 3.0,
    min_pixels_per_bin:     int   = 3,
    bad_pixel_mask:         np.ndarray = None,
    peak_distance:          int   = 5,
    peak_prominence:        float = 100.0,
    peak_fit_half_window:   int   = 8,
    min_peak_sep_px:        float = 3.0,
) -> 'FringeProfile':
    """
    Reduce a 2D image to a 1D r²-binned profile, then find and fit peaks.

    The image parameter must already be dark-subtracted (float64).
    Dark subtraction is the caller's responsibility — see Section 3.

    r_max_px defaults:
      128.0 px — synthetic images (M02/M04, 256×256, centred fringe)
      110.0 px — FlatSat / flight frames (clipped outer ring at ~113 px)

    SEM denominator: N_pixels (CCD pixel count), NOT N_subpixels.
    Masked bins: sigma_profile = two_sigma_profile = np.inf.
    peak_fits: populated via _find_and_fit_peaks().
    """
```

---

## 13. FringeProfile dataclass

```python
@dataclass
class FringeProfile:
    # Profile arrays — shape (n_bins,)
    profile:           np.ndarray   # mean ADU per r² bin
    sigma_profile:     np.ndarray   # 1σ SEM (np.inf for masked)
    two_sigma_profile: np.ndarray   # exactly 2 × sigma_profile  (S04)
    r_grid:            np.ndarray   # bin centre radii, px
    r2_grid:           np.ndarray   # bin centre r², px²
    n_pixels:          np.ndarray   # CCD pixel count per bin
    masked:            np.ndarray   # bool, True = excluded

    # Centre (S04 uncertainty fields)
    cx:             float
    cy:             float
    sigma_cx:       float
    sigma_cy:       float
    two_sigma_cx:   float   # 2 × sigma_cx
    two_sigma_cy:   float   # 2 × sigma_cy

    # Provenance
    seed_source:   str    # 'human'|'history'|'com'|'geometric'|'provided'
    stage1_cx:     float
    stage1_cy:     float
    cost_at_min:   float
    quality_flags: int
    sparse_bins:   bool   # True if > 10% bins below min_pixels_per_bin

    # Reduction parameters
    r_min_px:    float
    r_max_px:    float
    n_bins:      int
    n_subpixels: int
    sigma_clip:  float
    image_shape: tuple

    # Peak fits — populated for calibration frames, empty for science frames
    peak_fits: list   # list[PeakFit], sorted by r_raw_px

    # Dark subtraction provenance (Section 3.6)
    dark_subtracted: bool   # True if master_dark was provided and applied
    dark_n_frames:   int    # number of dark frames combined into master (0 if none)


class QualityFlags:
    GOOD                    = 0x00
    STAGE1_FAILED           = 0x01
    SEED_DISAGREEMENT       = 0x02
    SEED_FALLBACK_GEOMETRIC = 0x04
    LOW_CONFIDENCE          = 0x08
    CENTRE_FAILED           = 0x10
    SPARSE_BINS             = 0x20
    CENTRE_JUMP             = 0x40
```

---

## 14. Confirmed operational parameters

| Parameter | Synthetic | FlatSat / flight |
|-----------|-----------|-----------------|
| `r_max_px` | 128.0 | 110.0 |
| `n_bins` | 150 | 150 |
| `n_subpixels` | 1 | 1 |
| `overscan_left` | 8 | 8 |
| `overscan_bottom` | 4 | 4 |
| Expected peaks | ≥ 6 | 20 (10 per Ne line) |

---

## 15. Verification tests

All 10 in `tests/test_m03_annular_reduction_2026-04-05.py`.
(T9 and T10 are new in this revision — verify dark subtraction.)

### T1 — Round-trip matches M01 ground truth (< 2%)
### T2 — Centre recovery < 0.05 px on known-offset synthetic image
### T3 — Wrong centre (0.5 px offset) gives lower fringe contrast
### T4 — SEM ratio n_subpixels=8 vs 1 in [0.7, 1.4]

### T5 — Hot-pixel clip: centre unaffected, profile unaffected

```python
def test_hot_pixel_clip():
    """
    Injecting a saturated pixel must not shift the centre > 0.1 px
    and must not change the mean profile by more than 0.5%.
    Verifies that the 99.5th-percentile clip is applied to cost only,
    not to the annular reduction.
    """
    params = InstrumentParams()
    result = synthesise_calibration_image(params, add_noise=False)
    img_clean = result['image_2d'].copy()
    img_hot   = img_clean.copy()
    img_hot[64, 64] = 65535
    fp_c = reduce_calibration_frame(img_clean, cx_human=127.5, cy_human=127.5,
                                     r_max_px=params.r_max)
    fp_h = reduce_calibration_frame(img_hot,   cx_human=127.5, cy_human=127.5,
                                     r_max_px=params.r_max)
    assert abs(fp_h.cx - 127.5) < 0.1, "Hot pixel shifted centre"
    mask = ~fp_c.masked & ~fp_h.masked
    diff_frac = np.mean(np.abs(fp_h.profile[mask] - fp_c.profile[mask])) \
                / np.mean(fp_c.profile[mask])
    assert diff_frac < 0.005, f"Hot pixel contaminated profile: {diff_frac:.4f}"
```

### T6 — Peak finding: ≥ 6 peaks found and fitted on synthetic cal image

```python
def test_peak_finding():
    """
    Synthetic calibration image must yield >= 6 peaks with fit_ok=True.
    All-fit requirement is relaxed by 1 to allow edge peak failure.
    """
    params = InstrumentParams()
    result = synthesise_calibration_image(params, add_noise=False)
    fp = reduce_calibration_frame(result['image_2d'], cx_human=127.5,
                                   cy_human=127.5, r_max_px=params.r_max)
    assert len(fp.peak_fits) >= 6, \
        f"Only {len(fp.peak_fits)} peaks found"
    n_ok = sum(1 for p in fp.peak_fits if p.fit_ok)
    assert n_ok >= len(fp.peak_fits) - 1, \
        f"Only {n_ok}/{len(fp.peak_fits)} Gaussian fits succeeded"
```

### T7 — Science frame: empty peaks, seed_source='provided'

```python
def test_science_frame_no_peaks():
    params = InstrumentParams()
    sci = synthesise_airglow_image(100.0, params, add_noise=True,
                                    rng=np.random.default_rng(42))
    fp = reduce_science_frame(sci['image_2d'], cx=127.5, cy=127.5,
                               sigma_cx=0.05, sigma_cy=0.05,
                               r_max_px=params.r_max)
    assert fp.seed_source == 'provided'
    assert fp.peak_fits == []
```

### T8 — Masked bins have sigma = inf and two_sigma = inf (S04)

### T9 — Dark subtraction removes known dark signal from profile

```python
def test_dark_subtraction_removes_signal():
    """
    Construct a synthetic calibration image. Inject a uniform dark frame
    with a known constant value. Verify that the post-subtraction profile
    mean is reduced by approximately that constant.
    """
    import numpy as np
    params = InstrumentParams()
    result = synthesise_calibration_image(params, add_noise=False)
    img = result['image_2d'].astype(np.float64)
    dark_level = 200.0
    dark_frame = np.full_like(img, dark_level)
    master_dark = make_master_dark([dark_frame])

    fp_no_dark = reduce_calibration_frame(img, cx_human=127.5, cy_human=127.5,
                                           r_max_px=params.r_max,
                                           master_dark=None)
    fp_dark    = reduce_calibration_frame(img, cx_human=127.5, cy_human=127.5,
                                           r_max_px=params.r_max,
                                           master_dark=master_dark)

    mask = ~fp_no_dark.masked & ~fp_dark.masked
    mean_diff = np.mean(fp_no_dark.profile[mask] - fp_dark.profile[mask])
    assert abs(mean_diff - dark_level) < 1.0, \
        f"Dark subtraction removed {mean_diff:.1f} ADU; expected {dark_level:.1f}"
    assert fp_dark.dark_subtracted is True
    assert fp_dark.dark_n_frames == 1
    assert fp_no_dark.dark_subtracted is False
    assert fp_no_dark.dark_n_frames == 0
```

### T10 — Master dark: median of multiple frames is robust to a cosmic ray

```python
def test_master_dark_median_cosmic_ray():
    """
    A single cosmic-ray hit in one of 3 dark frames must not contaminate
    the master dark at that pixel location.
    """
    import numpy as np
    base = np.full((256, 256), 150.0)
    frames = [base.copy() for _ in range(3)]
    frames[1][100, 100] = 60000.0   # cosmic ray in frame 2 only
    master = make_master_dark(frames)
    assert abs(master[100, 100] - 150.0) < 1.0, \
        f"Cosmic ray leaked into master dark: {master[100,100]:.1f}"
    assert abs(np.mean(master) - 150.0) < 0.1
```

---

## 16. Expected numerical values

| Quantity | Expected | Test |
|----------|----------|------|
| Centre recovery | < 0.05 px | T2 |
| sigma_cx, sigma_cy | < 0.1 px | T2 |
| Profile round-trip | < 2% | T1 |
| SEM ratio 8/1 | 0.7–1.4 | T4 |
| Hot pixel centre shift | < 0.1 px | T5 |
| Peaks on synthetic | ≥ 6 | T6 |
| Peaks on real FlatSat | 20 (10 per Ne line) | — |
| Dark removal (uniform 200 ADU dark) | mean diff = 200 ± 1 ADU | T9 |
| Cosmic ray in 1/3 dark frames | master dark unaffected at hit pixel | T10 |

---

## 17. Dependencies

```
numpy  >= 1.24
scipy  >= 1.10   # optimize.minimize, signal.find_peaks, optimize.curve_fit
```

---

## 18. File locations

```
soc_sewell/
├── fpi/
│   └── m03_annular_reduction_2026-04-05.py
├── tests/
│   └── test_m03_annular_reduction_2026-04-05.py
└── docs/specs/
    └── S12_m03_annular_reduction_2026-04-06.md
```

---

## 19. Instructions for Claude Code

This is a **revision** to an already-implemented module. The existing
`src/fpi/m03_annular_reduction_2026_04_05.py` and its test file must be
updated in place. Do not create new files.

**Changes required to `src/fpi/m03_annular_reduction_2026_04_05.py`:**

1. Read this revised spec, S04, and the original S12 spec in full before
   writing any code.
2. Confirm the existing 8 M03 tests still pass before making any changes:
   ```bash
   pytest tests/test_m03_annular_reduction_2026_04_05.py -v
   ```
3. Add two new functions at the top of the module, immediately after the
   imports and before any existing code:
   - `make_master_dark(dark_frames)` — Section 3.2
   - `subtract_dark(image, master_dark, clip_negative=True)` — Section 3.3
4. Add `master_dark: np.ndarray = None` as the **first** non-image parameter
   to both `reduce_calibration_frame()` and `reduce_science_frame()`.
5. Add the dark subtraction call as the **first lines** of both functions
   (Section 3.4 pattern).
6. Add `dark_subtracted: bool` and `dark_n_frames: int` fields to the
   `FringeProfile` dataclass (Section 3.6). Set them correctly in both
   `reduce_calibration_frame()` and `reduce_science_frame()`.
7. Add `annular_reduce` docstring note: "image must already be dark-subtracted".
   `annular_reduce` itself does NOT accept a `master_dark` parameter —
   dark subtraction is the caller's responsibility.
8. Update the implementation order in the module to:
   `make_master_dark` → `subtract_dark` → [existing functions in original
   order] → `reduce_calibration_frame` → `reduce_science_frame`

**Changes required to `tests/test_m03_annular_reduction_2026_04_05.py`:**

9. Add T9 (`test_dark_subtraction_removes_signal`) — Section 15.
10. Add T10 (`test_master_dark_median_cosmic_ray`) — Section 15.
    Both tests import `make_master_dark` from the module.

**Verification:**
```bash
pytest tests/test_m03_annular_reduction_2026_04_05.py -v
```
All 10 tests must pass (8 existing + 2 new).

**Full suite:**
```bash
pytest tests/ -v
```
No regressions permitted.

**Commit message:**
```
feat(m03): add dark frame subtraction, 10/10 tests pass
Implements: S12_m03_annular_reduction_2026-04-06.md (revised)
```
