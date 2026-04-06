# S12 — M03 Fringe Centre Finding and Annular Reduction Specification

**Spec ID:** S12
**Spec file:** `docs/specs/S12_m03_annular_reduction_2026-04-05.md`
**Project:** WindCube FPI Pipeline
**Institution:** NCAR / High Altitude Observatory (HAO)
**Status:** Specification — ready for implementation in VS Code
**Depends on:** S01, S02, S03, S04, S09 (M01), S10 (M02), S11 (M04)
**Used by:**
  - S13 (M05) — receives calibration `FringeProfile`
  - S14 (M06) — receives science `FringeProfile`
  - S16 (INT02) — exercises full reduction pipeline
**References:**
  - Harding et al. (2014) Applied Optics 53(4), Section 3 — annular summation
  - Niciejewski et al. (1992) SPIE 1745 — r² reduction and software aperture
  - Mulligan (1986) J. Phys. E 19, 545 — software aperture technique
**Last updated:** 2026-04-05

> **Note:** This spec supersedes `m03_annular_reduction_spec.md` from the
> legacy repo. All references to `ADDENDUM_uncertainty_standards.md` now
> point to S04. The `r_max_px` default is 128 px for synthetic images
> (params.r_max) and 110 px for real FlatSat/flight frames. Repo is
> `soc_sewell`. File naming follows S01 convention.

---

## 1. Purpose

M03 reduces a 2D CCD fringe image to a 1D radial intensity profile (`FringeProfile`)
that M05 and M06 can invert. It has two sequential responsibilities:

1. **Centre finding** — locate the optical axis `(cx, cy)` to sub-pixel
   accuracy using a two-stage algorithm: coarse centre-of-mass (Stage 1)
   followed by two-pass azimuthal variance minimisation (Stage 2).

2. **Annular reduction** — bin all CCD pixels by r² distance from `(cx, cy)`,
   computing mean intensity and SEM per bin. The SEM feeds directly into
   the χ² weighting of M05 and M06.

M03 is the most operationally critical module because every downstream
calculation depends on centre accuracy. A 0.5 px centre error causes
visible peak doubling in the radial profile (Niciejewski 1992, Fig. 6).
The algorithm design here is derived from empirical testing on real
WindCube FlatSat calibration frames.

**The API enforces a strict separation:**
- `reduce_calibration_frame()` — finds the centre AND reduces the profile.
  Called on neon lamp frames.
- `reduce_science_frame()` — reduces the profile using a pre-supplied centre.
  Never attempts centre finding. Called on airglow frames.

This separation is mandatory. The airglow signal is too faint to find the
centre from a science frame directly.

---

## 2. Why naive centre-finding approaches fail on WindCube data

These failures are empirically confirmed on real FlatSat frames.
Understanding them is essential to implementing the correct algorithm.

**Radial symmetry transform (Loy & Zelinsky 2003):** Vote scale capped at
~10 px; FPI ring pixels at r = 60 px need to vote 60 px inward. The overscan
boundary gradient dominates votes → errors of 100+ px.

**Airy model nonlinear least-squares as centre finder:** Fails when stray
light makes the brightest feature a ring at r ≈ 8–20 px rather than the
zeroth-order disc at r = 0. Fitter shifts `(cx, cy)` to bring model peak
toward data peak → spurious minimum 1–2 px from true centre.

**Single-pass Nelder-Mead on the variance cost:** The azimuthal variance
minimum is only ~1 px wide. A single Nelder-Mead pass with a large simplex
steps over this basin → converges to a local plateau far from truth.

**Hough circles:** When the outer ring clips the CCD edge, Hough fits the
clipped arc rather than the common centre → errors of 50–100 px. Confirmed
on FlatSat frames where the outer ring clips the left edge.

**Conclusion:** The only robust approach is (a) accept a human visual
estimate or history-based seed as the primary input, then (b) refine
with the two-pass azimuthal variance minimiser. The Stage 1 CoM is a
useful cross-check but must never be the sole seed when the outer ring
is near the CCD edge.

---

## 3. Binning convention — r² not r

Under the paraxial approximation the FPI resonance condition gives
wavelength linear in r² (Niciejewski 1992):

```
λ = (μt / f²m) · r²
```

Fringes are equally spaced in r², not in r. **All binning in M03 uses r².**

Bin boundaries in r² space:
```
r²_edges[i] = (i / N_bins) · r²_max     i = 0, 1, ..., N_bins
```

Bin centre radii (for the `r_grid` output field):
```
r_centre[i] = sqrt(0.5 · (r²_edges[i] + r²_edges[i+1]))
```

This convention is used for both the variance cost function (Stage 2)
and the final annular reduction output.

---

## 4. Stage 1 — Coarse centre: intensity-weighted CoM

```python
def coarse_centre_com(
    image:               np.ndarray,
    overscan_left:       int   = 8,     # columns to zero (2×2 binned)
    overscan_bottom:     int   = 4,     # rows to zero (2×2 binned)
    coarse_top_pct:      float = 0.5,   # top % of pixels for CoM
    coarse_inner_frac:   float = 0.45,  # inner region as fraction of min dim
) -> tuple[float, float] | tuple[None, None]:
    """
    Locate the zeroth-order Airy disc by intensity-weighted CoM of the
    brightest pixels within the inner 45% of the frame.

    Returns (cx_coarse, cy_coarse) accurate to ±1–4 pixels,
    or (None, None) if the top-0.5% pixels are concentrated outside
    the inner 45% region (flag STAGE1_FAILED).

    Parameters
    ----------
    overscan_left   : columns to zero before CoM. 8 px for 2×2 binning.
                      Do NOT use 1×1 values (15 px) — they are for unbinned.
    overscan_bottom : rows to zero before CoM. 4 px for 2×2 binning.
    coarse_inner_frac : restricts CoM to inner 45% to exclude bright
                      vignette rim at frame edges. Critical fix for frames
                      where outer ring is brighter than the central disc.
    """
```

---

## 5. Human prior and centre history

```python
def resolve_seed(
    cx_human:   float | None,    # human visual estimate, ±2 px accuracy
    cy_human:   float | None,
    cx_com:     float | None,    # Stage 1 result (may be None if failed)
    cy_com:     float | None,
    cx_history: float | None,    # most recent verified centre
    cy_history: float | None,
    image_size: int,
    disagreement_threshold_px: float = 5.0,
) -> tuple[float, float, str, int]:
    """
    Resolve the best seed for Stage 2 from available sources.

    Priority order:
    1. Human prior — if provided, always used. Sets seed_source='human'.
    2. History — if no human prior but history available. seed_source='history'.
    3. CoM Stage 1 — if no human or history. seed_source='com'.
    4. Geometric centre — fallback of last resort. seed_source='geometric'.
       Sets flag SEED_FALLBACK_GEOMETRIC.

    If human and CoM both available and disagree by > disagreement_threshold_px,
    sets flag SEED_DISAGREEMENT but still uses human (not CoM).

    Returns
    -------
    (cx_seed, cy_seed, seed_source, quality_flags_partial)
    """
```

---

## 6. Stage 2 — Fine centre: two-pass azimuthal variance minimisation

### 6.1 Principle

The true optical axis is the unique point around which all FP rings are
perfectly circular. The cost function:

```
C(cx, cy) = Σ_bins  Var{ I(pixels) : r² ∈ [r²_bin_i] }
```

has its minimum at the true `(cx, cy)` regardless of fringe amplitude,
background, or stray light level.

### 6.2 Two-pass algorithm (mandatory — single-pass fails)

```
Pass 1 — coarse grid search:
  Grid spacing: max(2.0, search_radius / 8) px
  Grid extent:  ±search_radius px around seed (default 15 px)
  → reliably locates the correct minimum basin

Pass 2 — fine Nelder-Mead:
  Initial simplex radius: grid_step + 0.5 px around Pass 1 minimum
  scipy.optimize.minimize(method='Nelder-Mead')
  Options: xatol=0.02, fatol=1.0, maxiter=500
  → converges to sub-pixel accuracy (< 0.05 px error on synthetic images)
```

```python
def azimuthal_variance_centre(
    image:          np.ndarray,
    cx_seed:        float,
    cy_seed:        float,
    var_r_min_px:   float = 5.0,    # exclude central disc (too few pixels)
    var_r_max_px:   float = None,   # None → image_size//2 - 10
    var_n_bins:     int   = 250,    # r² bins for variance computation
    var_search_px:  float = 15.0,   # half-width of coarse grid search
) -> tuple[float, float, float]:
    """
    Two-pass azimuthal variance minimisation.

    var_r_max_px: set to 110 px for FlatSat/flight frames to exclude the
    clipped outer ring. For synthetic images from M02 (image_size=256,
    centred fringe), None or 118 px is appropriate.

    Returns
    -------
    (cx_fine, cy_fine, cost_at_min)
    """
```

---

## 7. Centre uncertainty estimation

```python
def estimate_centre_uncertainty(
    image:       np.ndarray,
    cx:          float,
    cy:          float,
    delta_px:    float = 0.5,   # perturbation size for numerical Hessian
    var_r_min_px: float = 5.0,
    var_r_max_px: float = None,
    var_n_bins:   int   = 250,
) -> tuple[float, float]:
    """
    Estimate 1σ uncertainties on the centre position from the local
    curvature of the azimuthal variance cost function.

    Uses a 2-point numerical second derivative:
      σ_cx ≈ sqrt(2 · C(cx, cy) / |d²C/dcx²|)

    Returns
    -------
    (sigma_cx, sigma_cy) in pixels

    Typical values for WindCube synthetic images: 0.01–0.05 px.
    Typical values for FlatSat frames: 0.01–0.03 px.
    If sigma > 0.3 px, sets LOW_CONFIDENCE flag.
    If sigma > 1.0 px, sets CENTRE_FAILED flag.
    """
```

---

## 8. Annular reduction

```python
def annular_reduce(
    image:                  np.ndarray,
    cx:                     float,
    cy:                     float,
    r_min_px:               float = 0.0,
    r_max_px:               float = 128.0,  # synthetic default; 110 for FlatSat
    n_bins:                 int   = 150,
    n_subpixels:            int   = 8,
    sigma_clip_threshold:   float = 3.0,
    min_pixels_per_bin:     int   = 3,
    bad_pixel_mask:         np.ndarray = None,
) -> 'FringeProfile':
    """
    Reduce a 2D CCD image to a 1D r²-binned radial intensity profile.

    Algorithm
    ---------
    1. Zero overscan columns/rows; apply bad_pixel_mask.
    2. For each active pixel (row_i, col_j):
       - Subdivide into N_sub × N_sub equal-area sub-pixels.
       - For each sub-pixel: compute r² = (col-cx)² + (row-cy)²,
         assign parent pixel ADU to the appropriate r² bin.
    3. Per bin: compute mean_I, std_I from contributing pixel ADU values.
    4. Sigma-clip once: reject pixels where |I − mean_I| > threshold × std_I,
       then recompute mean_I and std_I.
    5. SEM = std_I / sqrt(N_pixels)  ← N_pixels, NOT N_subpixels (see below).
    6. Mask bins with N_pixels < min_pixels_per_bin (set sigma = np.inf).
    7. Return FringeProfile.

    Parameters
    ----------
    r_max_px       : outer radius cutoff.
                     Default 128.0 px for synthetic images (params.r_max).
                     Use 110.0 px for FlatSat/flight frames (excludes clipped ring).
    n_subpixels    : sub-pixel subdivision (Mulligan 1986).
                     N_sub=8 → 64 sub-pixels per pixel, ~0.35 s runtime.
    sigma_clip     : reject pixels > N sigma from bin mean. Default 3.0.

    CRITICAL — SEM denominator
    --------------------------
    SEM = std_I / sqrt(N_pixels_in_bin)   ← CORRECT
    SEM = std_I / sqrt(N_subpixels_in_bin) ← WRONG (underestimates uncertainty)

    The N_sub² sub-pixels from one CCD pixel are fully correlated (identical ADU).
    The effective number of independent measurements is the count of CCD pixels,
    not sub-pixels. T4 verifies this explicitly.
    """
```

---

## 9. Output dataclass — FringeProfile

`FringeProfile` is the standard interface between M03, M05, and M06.
Every field follows the S04 uncertainty convention.

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class FringeProfile:
    """
    1D radial fringe profile produced by M03.
    Inputs to M05 (calibration inversion) and M06 (airglow inversion).
    """
    # Profile arrays — shape (n_bins,)
    profile:           np.ndarray  # mean intensity per r² bin, ADU
    sigma_profile:     np.ndarray  # 1σ SEM per bin (np.inf for masked bins)
    two_sigma_profile: np.ndarray  # exactly 2 × sigma_profile  (S04)
    r_grid:            np.ndarray  # bin centre radii, pixels
    r2_grid:           np.ndarray  # bin centre r² values, pixels²
    n_pixels:          np.ndarray  # CCD pixel count per bin (int)
    masked:            np.ndarray  # bool — True = excluded from fitting

    # Centre — with S04 uncertainties
    cx:             float   # fringe centre x, pixels
    cy:             float   # fringe centre y, pixels
    sigma_cx:       float   # 1σ uncertainty on cx, pixels
    sigma_cy:       float   # 1σ uncertainty on cy, pixels
    two_sigma_cx:   float   # exactly 2 × sigma_cx  (S04)
    two_sigma_cy:   float   # exactly 2 × sigma_cy  (S04)

    # Centre provenance
    seed_source:  str    # 'human' | 'history' | 'com' | 'geometric'
    stage1_cx:    float  # Stage 1 CoM result (for diagnostics)
    stage1_cy:    float
    cost_at_min:  float  # azimuthal variance cost at (cx, cy)

    # Quality flags
    quality_flags: int   # QualityFlags bitmask

    @property
    def is_good(self) -> bool:
        return self.quality_flags == 0

    @property
    def centre_low_confidence(self) -> bool:
        return bool(self.quality_flags & QualityFlags.LOW_CONFIDENCE)

    # Reduction parameters (for reproducibility)
    r_min_px:    float
    r_max_px:    float
    n_bins:      int
    n_subpixels: int
    sigma_clip:  float
    image_shape: tuple


class QualityFlags:
    GOOD                    = 0x00
    STAGE1_FAILED           = 0x01  # CoM Stage 1 returned None
    SEED_DISAGREEMENT       = 0x02  # human and CoM disagree by > 5 px
    SEED_FALLBACK_GEOMETRIC = 0x04  # no seed; used image centre
    LOW_CONFIDENCE          = 0x08  # sigma_cx or sigma_cy > 0.3 px
    CENTRE_FAILED           = 0x10  # sigma_cx or sigma_cy > 1.0 px
    SPARSE_BINS             = 0x20  # > 10% of bins have N_pixels < min
    CENTRE_JUMP             = 0x40  # centre differs from history by > 2 px
```

---

## 10. Top-level API

### 10.1 `reduce_calibration_frame`

```python
def reduce_calibration_frame(
    image:             np.ndarray,
    cx_human:          float = None,      # human estimate, ±2 px
    cy_human:          float = None,
    cx_history:        float = None,      # from previous calibration frame
    cy_history:        float = None,
    sigma_human:       float = 2.0,
    r_max_px:          float = 128.0,     # synthetic: params.r_max; FlatSat: 110
    n_bins:            int   = 150,
    n_subpixels:       int   = 8,
    sigma_clip:        float = 3.0,
    min_pixels_per_bin: int  = 3,
    bad_pixel_mask:    np.ndarray = None,
    overscan_left:     int   = 8,
    overscan_bottom:   int   = 4,
) -> FringeProfile:
    """
    Full pipeline for a neon lamp calibration frame:
    Stage 1 (CoM) → resolve_seed → Stage 2 (azimuthal variance)
    → estimate_centre_uncertainty → annular_reduce.

    The centre found here must be passed to reduce_science_frame()
    for all corresponding science frames. Never find the centre from
    a science frame directly — the signal is too faint.
    """
```

### 10.2 `reduce_science_frame`

```python
def reduce_science_frame(
    image:             np.ndarray,
    cx:                float,          # from reduce_calibration_frame()
    cy:                float,
    sigma_cx:          float,          # from reduce_calibration_frame()
    sigma_cy:          float,
    r_max_px:          float = 128.0,
    n_bins:            int   = 150,
    n_subpixels:       int   = 8,
    sigma_clip:        float = 3.0,
    min_pixels_per_bin: int  = 3,
    bad_pixel_mask:    np.ndarray = None,
    overscan_left:     int   = 8,
    overscan_bottom:   int   = 4,
) -> FringeProfile:
    """
    Annular reduction of a science (airglow) frame.
    NO centre finding. Centre is accepted directly from the paired
    calibration frame and stored in the output FringeProfile.

    seed_source is set to 'provided' (not 'human', 'com', etc.).
    """
```

---

## 11. Confirmed operational parameters

From WindCube FlatSat calibration testing:

| Parameter | Synthetic | FlatSat / flight | Notes |
|-----------|-----------|-----------------|-------|
| `r_max_px` | 128.0 | 110.0 | FlatSat outer ring clips at ~113 px |
| `n_bins` | 150 | 150 | ~15 bins per FSR at t=20 mm |
| `n_subpixels` | 8 | 8 | 64 sub-pixels/pixel, ~0.35 s |
| `overscan_left` | 8 | 8 | 2×2 binned value |
| `overscan_bottom` | 4 | 4 | 2×2 binned value |
| `coarse_inner_frac` | 0.45 | 0.45 | Excludes bright vignette rim |
| `var_r_max_px` | 118 | 110 | Used in Stage 2 only |
| Centre accuracy | < 0.05 px | 0.008–0.015 px | Verified |

---

## 12. Verification tests

All 8 tests in `tests/test_m03_annular_reduction_2026-04-05.py`.

### T1 — Round-trip: profile matches M01 ground truth

```python
def test_round_trip_matches_m01():
    """
    Reduce a noiseless synthetic calibration image with known exact centre.
    Recovered 1D profile must match M01 ground truth to within 2%.
    """
    from fpi.m01_airy_forward_model import InstrumentParams
    from fpi.m02_calibration_synthesis import synthesise_calibration_image
    from scipy.interpolate import interp1d

    params = InstrumentParams()
    result = synthesise_calibration_image(params, add_noise=False,
                                          cx=127.5, cy=127.5)
    profile = reduce_calibration_frame(
        result['image_2d'], cx_human=127.5, cy_human=127.5,
        r_max_px=params.r_max, n_bins=150, n_subpixels=8)

    m01_interp = interp1d(result['r_grid'], result['profile_1d'],
                          bounds_error=False, fill_value=params.B)
    m01_at_bins = m01_interp(profile.r_grid)
    mask = ~profile.masked
    np.testing.assert_allclose(
        profile.profile[mask], m01_at_bins[mask], rtol=0.02,
        err_msg="Annular profile disagrees with M01 ground truth by > 2%")
```

### T2 — Centre recovery accuracy < 0.05 px

```python
def test_centre_recovery_accuracy():
    """Stage 2 must recover a known offset centre to < 0.05 px."""
    from fpi.m01_airy_forward_model import InstrumentParams
    from fpi.m02_calibration_synthesis import synthesise_calibration_image

    params = InstrumentParams()
    true_cx, true_cy = 131.3, 124.7
    result = synthesise_calibration_image(params, add_noise=False,
                                          cx=true_cx, cy=true_cy)
    profile = reduce_calibration_frame(
        result['image_2d'], cx_human=130.0, cy_human=124.0,
        r_max_px=params.r_max)

    assert abs(profile.cx - true_cx) < 0.05, \
        f"cx error {abs(profile.cx - true_cx):.4f} px exceeds 0.05 px"
    assert abs(profile.cy - true_cy) < 0.05, \
        f"cy error {abs(profile.cy - true_cy):.4f} px exceeds 0.05 px"
```

### T3 — Wrong centre gives lower fringe contrast

```python
def test_wrong_centre_reduces_contrast():
    """
    A 0.5 px centre error must reduce the fringe peak-to-trough contrast
    relative to the correct centre.
    """
    from fpi.m01_airy_forward_model import InstrumentParams
    from fpi.m02_calibration_synthesis import synthesise_calibration_image

    params = InstrumentParams()
    result = synthesise_calibration_image(params, add_noise=False,
                                          cx=127.5, cy=127.5)
    img = result['image_2d']
    p_correct = reduce_calibration_frame(img, cx_human=127.5, cy_human=127.5,
                                          r_max_px=params.r_max)
    p_wrong   = reduce_calibration_frame(img, cx_human=128.0, cy_human=128.0,
                                          r_max_px=params.r_max)
    contrast_c = np.max(p_correct.profile) - np.min(p_correct.profile)
    contrast_w = np.max(p_wrong.profile)   - np.min(p_wrong.profile)
    assert contrast_c > contrast_w, \
        "Correct centre should give higher fringe contrast than 0.5 px offset"
```

### T4 — SEM uses pixel count not sub-pixel count

```python
def test_sem_uses_pixel_count():
    """
    SEM must not scale with n_subpixels.
    Profiles at n_subpixels=1 and n_subpixels=8 must have similar SEM
    (ratio in 0.7–1.4). If SEM scaled with sub-pixel count, ratio would
    be ~8.
    """
    from fpi.m01_airy_forward_model import InstrumentParams
    from fpi.m02_calibration_synthesis import synthesise_calibration_image

    params = InstrumentParams()
    result = synthesise_calibration_image(params, add_noise=True,
                                          rng=np.random.default_rng(42))
    p1 = reduce_calibration_frame(result['image_2d'],
                                   cx_human=127.5, cy_human=127.5,
                                   r_max_px=params.r_max, n_subpixels=1)
    p8 = reduce_calibration_frame(result['image_2d'],
                                   cx_human=127.5, cy_human=127.5,
                                   r_max_px=params.r_max, n_subpixels=8)
    ratio = np.median(p8.sigma_profile[~p8.masked] /
                      p1.sigma_profile[~p1.masked])
    assert 0.7 < ratio < 1.4, \
        f"SEM ratio n_sub=8/1 = {ratio:.2f}; expected 0.7–1.4 (not ~8)"
```

### T5 — Sigma clipping removes hot pixels

```python
def test_sigma_clipping_removes_hot_pixels():
    """
    Profile with injected hot pixels must be similar to clean profile.
    Sigma clipping removes the hot pixel contribution.
    """
    from fpi.m01_airy_forward_model import InstrumentParams
    from fpi.m02_calibration_synthesis import synthesise_calibration_image

    params = InstrumentParams()
    result = synthesise_calibration_image(params, add_noise=False)
    img_clean = result['image_2d'].copy()
    img_dirty = img_clean.copy()
    rng = np.random.default_rng(0)
    hot_r = rng.integers(20, 200, 10)
    hot_c = rng.integers(20, 200, 10)
    img_dirty[hot_r, hot_c] = 50000

    p_clean = reduce_calibration_frame(img_clean, cx_human=127.5,
                                        cy_human=127.5, r_max_px=params.r_max)
    p_dirty = reduce_calibration_frame(img_dirty, cx_human=127.5,
                                        cy_human=127.5, r_max_px=params.r_max)
    mask = ~p_clean.masked & ~p_dirty.masked
    np.testing.assert_allclose(p_dirty.profile[mask], p_clean.profile[mask],
        rtol=0.05, err_msg="Hot pixels contaminated the mean profile")
```

### T6 — Science frame uses provided centre

```python
def test_science_frame_uses_provided_centre():
    """
    reduce_science_frame must store the provided centre exactly in
    the output FringeProfile with seed_source='provided'.
    """
    from fpi.m01_airy_forward_model import InstrumentParams
    from fpi.m04_airglow_synthesis import synthesise_airglow_image

    params = InstrumentParams()
    sci = synthesise_airglow_image(100.0, params, add_noise=True,
                                    rng=np.random.default_rng(42))
    profile = reduce_science_frame(
        sci['image_2d'], cx=127.5, cy=127.5,
        sigma_cx=0.05, sigma_cy=0.05,
        r_max_px=params.r_max)

    assert profile.cx == 127.5
    assert profile.cy == 127.5
    assert profile.sigma_cx == 0.05
    assert profile.seed_source == 'provided'
```

### T7 — Quality flags triggered correctly

```python
def test_quality_flags_low_confidence():
    """
    LOW_CONFIDENCE flag must be set when sigma_cx > 0.3 px.
    When centre is badly wrong, either recovery or flag must be set.
    """
    from fpi.m01_airy_forward_model import InstrumentParams
    from fpi.m02_calibration_synthesis import synthesise_calibration_image

    params = InstrumentParams()
    result = synthesise_calibration_image(params, add_noise=True,
                                          rng=np.random.default_rng(42))
    profile = reduce_calibration_frame(
        result['image_2d'],
        cx_human=100.0, cy_human=100.0,  # bad seed: far from true centre
        r_max_px=params.r_max)

    if abs(profile.cx - 127.5) > 1.0:
        assert profile.quality_flags & QualityFlags.LOW_CONFIDENCE, \
            "Large centre error without LOW_CONFIDENCE flag set"
```

### T8 — Masked bins have sigma = inf and two_sigma = inf

```python
def test_masked_bins_sigma_inf():
    """
    Masked bins must have sigma_profile = np.inf and
    two_sigma_profile = np.inf. Enforces S04 uncertainty convention.
    """
    from fpi.m01_airy_forward_model import InstrumentParams
    from fpi.m02_calibration_synthesis import synthesise_calibration_image

    params = InstrumentParams()
    result = synthesise_calibration_image(params, add_noise=False)
    profile = reduce_calibration_frame(
        result['image_2d'], cx_human=127.5, cy_human=127.5,
        r_max_px=params.r_max, n_bins=150, min_pixels_per_bin=3)

    for i in range(len(profile.profile)):
        if profile.masked[i]:
            assert np.isinf(profile.sigma_profile[i]), \
                f"Masked bin {i}: sigma_profile not inf"
            assert np.isinf(profile.two_sigma_profile[i]), \
                f"Masked bin {i}: two_sigma_profile not inf"
        if np.isinf(profile.sigma_profile[i]):
            assert profile.masked[i], \
                f"Bin {i}: sigma=inf but not masked"
```

---

## 13. Expected numerical values

For `InstrumentParams()` defaults, 256×256 synthetic image, true centre (127.5, 127.5):

| Quantity | Expected | Test |
|----------|----------|------|
| Centre recovery error | < 0.05 px | T2 |
| sigma_cx, sigma_cy | < 0.1 px | T2 |
| Profile round-trip error | < 2% | T1 |
| SEM ratio (n_sub=8 vs 1) | 0.7–1.4 | T4 |
| Bins with data (n_bins=150) | ~148–150 | T1 |
| Hot pixel contamination | < 5% | T5 |

---

## 14. Dependencies

```
numpy  >= 1.24   # array operations
scipy  >= 1.10   # optimize.minimize (Nelder-Mead), interpolate.interp1d
```

No new dependencies beyond what previous tiers installed.

---

## 15. File locations in repository

```
soc_sewell/
├── fpi/
│   ├── __init__.py
│   ├── m01_airy_forward_model_2026-04-05.py     ← S09
│   ├── m02_calibration_synthesis_2026-04-05.py  ← S10
│   ├── m04_airglow_synthesis_2026-04-05.py      ← S11
│   └── m03_annular_reduction_2026-04-05.py      ← this module
├── tests/
│   └── test_m03_annular_reduction_2026-04-05.py
└── docs/specs/
    └── S12_m03_annular_reduction_2026-04-05.md  ← this file
```

---

## 16. Instructions for Claude Code

1. Read this entire spec AND S04 (uncertainty standards) before writing code.
2. Confirm M01, M02, M04 tests pass:
   `pytest tests/test_m01_* tests/test_m02_* tests/test_m04_* -v`
3. Implement `fpi/m03_annular_reduction_2026-04-05.py` in this strict order:
   `QualityFlags` → `CentreHistory` → `coarse_centre_com` → `resolve_seed`
   → `azimuthal_variance_centre` → `estimate_centre_uncertainty`
   → `FringeProfile` → `annular_reduce`
   → `reduce_calibration_frame` → `reduce_science_frame`
4. `azimuthal_variance_centre`: two-pass structure is mandatory.
   Do NOT implement as single-pass Nelder-Mead — it will fail on real data.
5. `annular_reduce`: binning must use r², not r. SEM denominator is
   `N_pixels` (CCD pixel count), never `N_subpixels`. T4 verifies this.
6. `two_sigma_profile` must be set to `2.0 * sigma_profile` element-wise.
   Masked bins: `sigma_profile = np.inf`, `two_sigma_profile = np.inf`.
   T8 verifies this — it is the S04 uncertainty convention.
7. `reduce_science_frame`: set `seed_source = 'provided'`. Do not call
   any centre-finding functions. T6 verifies this.
8. Write all 8 tests in `tests/test_m03_annular_reduction_2026-04-05.py`.
9. Run: `pytest tests/test_m03_annular_reduction_2026-04-05.py -v`
   All 8 must pass.
10. Run full suite: `pytest tests/ -v` — no regressions.
11. Commit: `feat(m03): implement annular reduction, 8/8 tests pass`

Module docstring header:
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
