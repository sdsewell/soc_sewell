# S12 — M03 Fringe Centre Finding and Annular Reduction Specification

**Spec ID:** S12
**Spec file:** `docs/specs/S12_m03_annular_reduction_2026-04-05.md`
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
**Last updated:** 2026-04-05

> **Revision note vs earlier draft:** Two additions:
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

---

## 1. Purpose

M03 reduces a 2D CCD fringe image to a 1D radial intensity profile and
simultaneously locates and fits all fringe peaks for the Tolansky analysis.

Three sequential responsibilities:

1. **Centre finding** — locate `(cx, cy)` to sub-pixel accuracy.
   Hot-pixel pre-clip + two-pass azimuthal variance minimisation.

2. **Annular reduction** — r²-binned mean intensity and SEM.

3. **Peak finding** — Gaussian fit to each fringe peak. Results stored
   in `FringeProfile.peak_fits` for consumption by S13 (Tolansky).

**API separation:**
- `reduce_calibration_frame()` — finds centre AND reduces AND finds peaks.
- `reduce_science_frame()` — uses provided centre, reduces only.
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

## 3. Binning convention — r² not r

Fringes are equally spaced in r² (Niciejewski 1992). All binning uses r²:

```
r²_edges[i] = (i / N_bins) · r²_max
r_centre[i] = sqrt(0.5 · (r²_edges[i] + r²_edges[i+1]))
```

---

## 4. Hot-pixel pre-clip

Before computing the variance cost, clip the image to its 99.5th percentile:

```python
p99_5 = float(np.percentile(image, 99.5))
image_for_cost = np.clip(image, None, p99_5)
```

**Used only for centre finding.** The full unclipped image is used for
the annular reduction profile. T5 verifies this separation explicitly.

---

## 5. Stage 1 — Coarse centre: intensity-weighted CoM

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

## 6. Human prior and seed resolution

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

## 7. Stage 2 — Two-pass azimuthal variance minimisation

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

## 8. Centre uncertainty estimation

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

## 9. Top-level centre API

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

## 10. Peak finding

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

## 11. Annular reduction

```python
def annular_reduce(
    image:                  np.ndarray,
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

    r_max_px defaults:
      128.0 px — synthetic images (M02/M04, 256×256, centred fringe)
      110.0 px — FlatSat / flight frames (clipped outer ring at ~113 px)

    SEM denominator: N_pixels (CCD pixel count), NOT N_subpixels.
    Masked bins: sigma_profile = two_sigma_profile = np.inf.
    peak_fits: populated via _find_and_fit_peaks().
    """
```

---

## 12. FringeProfile dataclass

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

## 13. Confirmed operational parameters

| Parameter | Synthetic | FlatSat / flight |
|-----------|-----------|-----------------|
| `r_max_px` | 128.0 | 110.0 |
| `n_bins` | 150 | 150 |
| `n_subpixels` | 1 | 1 |
| `overscan_left` | 8 | 8 |
| `overscan_bottom` | 4 | 4 |
| Expected peaks | ≥ 6 | 20 (10 per Ne line) |

---

## 14. Verification tests

All 8 in `tests/test_m03_annular_reduction_2026-04-05.py`.

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

---

## 15. Expected numerical values

| Quantity | Expected | Test |
|----------|----------|------|
| Centre recovery | < 0.05 px | T2 |
| sigma_cx, sigma_cy | < 0.1 px | T2 |
| Profile round-trip | < 2% | T1 |
| SEM ratio 8/1 | 0.7–1.4 | T4 |
| Hot pixel centre shift | < 0.1 px | T5 |
| Peaks on synthetic | ≥ 6 | T6 |
| Peaks on real FlatSat | 20 (10 per Ne line) | — |

---

## 16. Dependencies

```
numpy  >= 1.24
scipy  >= 1.10   # optimize.minimize, signal.find_peaks, optimize.curve_fit
```

---

## 17. File locations

```
soc_sewell/
├── fpi/
│   └── m03_annular_reduction_2026-04-05.py
├── tests/
│   └── test_m03_annular_reduction_2026-04-05.py
└── docs/specs/
    └── S12_m03_annular_reduction_2026-04-05.md
```

---

## 18. Instructions for Claude Code

1. Read this spec AND S04 before writing any code.
2. Confirm M01, M02, M04 tests pass first.
3. Implement in strict order:
   `QualityFlags` → `CentreResult` → `PeakFit` → `_gaussian` →
   `_variance_cost` → `coarse_centre_com` → `resolve_seed` →
   `azimuthal_variance_centre` → `estimate_centre_uncertainty` →
   `find_centre` → `_find_and_fit_peaks` → `FringeProfile` →
   `annular_reduce` → `reduce_calibration_frame` → `reduce_science_frame`
4. Hot-pixel clip in `find_centre()` only — never in `annular_reduce()`.
5. Two-pass structure mandatory — T2 fails with single-pass.
6. `_find_and_fit_peaks`: implement both adaptive fixes exactly as in
   Section 10.3. These are what achieve 20/20 peaks on real data.
7. SEM denominator = N_pixels, not N_subpixels. T4 verifies.
8. Masked bins: sigma_profile = two_sigma_profile = np.inf.
9. `reduce_science_frame`: seed_source='provided', peak_fits=[].
10. Run: `pytest tests/test_m03_annular_reduction_2026-04-05.py -v`
    All 8 must pass.
11. Run full suite: `pytest tests/ -v`
12. Commit: `feat(m03): implement annular reduction + peak finding, 8/8 tests pass`

```python
"""
Module:      m03_annular_reduction_2026-04-05.py
Spec:        docs/specs/S12_m03_annular_reduction_2026-04-05.md
Author:      Claude Code
Generated:   <today>
Last tested: <today>  (8/8 tests pass)
Project:     WindCube FPI Pipeline — NCAR/HAO
Repo:        soc_sewell
"""
```
