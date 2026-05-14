# S13b — Tolansky Single-Line Analysis (Airglow Radial Profile)

**Spec ID:** S13b
**Spec file:** `docs/specs/S13b_tolansky_1_line_2026-05-05.md`
**Project:** WindCube FPI Pipeline
**Institution:** NCAR / High Altitude Observatory (HAO)
**Depends on:** `annular_reduction.py` — provides `{stem}_peak_fits.npy`
**Used by:**
  - M05 (S14) — optional pre-fit check of plate scale α consistency
  - Z04 (SNR sensitivity sweep) — may use α_tol output as quality gate
**References:**
  - Vaughan (1989) *The Fabry-Perot Interferometer*, §3.5.2, pp. 116–121.
    Equations (3.83)–(3.86) apply to the single-line case (Steps 1–4 only).
    Steps 5–7 (Benoit recovery) **do not apply** — they require two lines.
  - Melissinos & Napolitano (2003) *Experiments in Modern Physics*, Ch. 7,
    §7.5 "The Fabry-Perot Etalon", pp. 303–312.
  - Niciejewski et al. (1992) SPIE Vol. 1745, pp. 166–179 — r² reduction
    technique for FPI airglow images.
  - Harding et al. (2014) *Applied Optics* 53, 4, 666–677 — WindCube
    primary algorithmic source.
  - Mulligan (1986) *J. Phys. E* 19, 545 — r²-bin annular reduction.
**Sibling spec:** S13a (two-line Tolansky for neon calibration lamp images)
**Last updated:** 2026-05-05

> **Design rationale.**  The two-line Benoit method (S13a) cannot be used for
> airglow images because there is only one spectral line (OI 630.0 nm).  Without
> a second line, N_Δ is indeterminate and d cannot be recovered absolutely.
> However, the plate scale α (equivalently the focal length f) **can** be
> recovered from the spacing of successive r²-peaks of a single line if d is
> supplied from a contemporaneous neon calibration.  This module performs
> exactly that measurement — no more — and reports α with its uncertainty.

> **Input clarification.**  Peak finding and Gaussian centroid fitting are
> performed entirely within `annular_reduction.py` and stored in the
> `_peak_fits.npy` file saved alongside the L1.2 archive.  This module reads
> that pre-computed 9-column float64 table directly.  It performs no peak
> detection and no Gaussian fitting of its own.

---

## 1. Purpose

Given the pre-computed ring peak radii from an **airglow science image**
(single OI 630.0 nm emission line), recover:

| Parameter | Symbol | Physical meaning |
|-----------|--------|-----------------|
| r²-spacing | Δ | Mean successive difference of ring r²-values (px²) |
| Plate scale | α | Angular pixel scale (rad/px) = pixel_pitch / f |
| Fractional order | ε | Fractional interference order at optical axis |

> **What this module cannot recover.**  Without a second spectral line, the
> absolute plate spacing d is indeterminate (the FSR integer ambiguity N_int
> cannot be resolved from the radial data alone).  This module therefore does
> **not** report d.  For d, use S13a on a neon calibration image taken in the
> same thermal state.

---

## 2. Physical basis

### 2.1 The Tolansky r²-spacing relation

For a Fabry-Pérot fringe pattern at high interference order, the paraxial
approximation (Vaughan §3.5.2, Eq. 3.84) gives:

```
r²_p  =  Δ · (p − 1 + ε)                              [Vaughan 3.84]
```

where r_p is the radius of the p-th ring (pixels), p = 1 for the
innermost visible ring, ε ∈ [0,1) is the fractional order at the optical
axis, and Δ (px²) is the constant r²-spacing:

```
Δ  =  f²_px · λ / (n_air · d)                          [Vaughan 3.85]
```

Successive differences δ_p = r²_{p+1} − r²_p all equal Δ for a
well-behaved fringe pattern.

### 2.2 Recovering α from a single line

Rearranging Eq. 3.85 using α = 1/f_px (rad/px):

```
α  =  sqrt(λ_OI · n_air / (d_cal · Δ))                [derived from 3.85]
σ(α)/α  =  (1/2) · sqrt( (σ_Δ/Δ)² + (σ_d_cal/d_cal)² )
```

Because d is unknown from the single-line data alone, **d_cal from the
most recent S13a result is required for Mode A**.  If d_cal is not
available, the module operates in Mode B and reports only Δ and ε.

### 2.3 Spectral context: OI 630.0 nm airglow

The OI 630.0 nm emission originates from the thermosphere (~250–300 km
peak altitude).  The rest wavelength used to predict Δ is
`λ_OI_rest = 630.0304 nm` (vacuum).  The fractional order ε encodes the
Doppler shift of the line, but S13b makes no attempt to extract wind
velocity — that is the responsibility of M06/M07.

### 2.4 Comparison with neon (S13a)

Since Δ ∝ λ and λ_OI (630.0 nm) < λ_Ne_a (640.2 nm), for the same d
and f the OI Δ is slightly smaller than the neon Δₐ:

```
Δ_OI / Δₐ  =  λ_OI / λ_Ne_a  =  630.0304 / 640.2248  ≈  0.9841
```

For d = 20.0006 mm and f ≈ 6230 px: Δ_OI ≈ 0.984 × Δₐ ≈ 477 px².

---

## 3. Input

### 3.1 Peak fits array

S13b reads the `_peak_fits.npy` file saved by `annular_reduction.py`:

```python
peaks = np.load('{stem}_peak_fits.npy')   # shape (N_peaks, 9), float64
```

The 9 columns are (zero-indexed), exactly as written by `annular_reduction.py`
lines 610–620:

| Col | Name | Units | Description |
|-----|------|-------|-------------|
| 0 | `peak_num` | — | 1-based peak index (float) |
| 1 | `r_raw_px` | px | Detected bin-centre radius (`find_peaks`) |
| 2 | `r_fit_px` | px | TRF Gaussian centroid μ; NaN if fit failed |
| 3 | `sigma_r_fit_px` | px | 1σ uncertainty on μ; NaN if fit failed |
| 4 | `r_fit_sq_px2` | px² | μ²; NaN if fit failed |
| 5 | `sigma_r_fit_sq_px2` | px² | 2·μ·σ_μ (propagated); NaN if fit failed |
| 6 | `amplitude_adu` | ADU | Gaussian amplitude A above background |
| 7 | `width_sigma_px` | px | Gaussian σ-width; NaN if fit failed |
| 8 | `reduced_chi2` | — | χ²/(n_points−4); NaN if fit failed |

Load and filter to rows with valid (non-NaN) Gaussian fits:

```python
peaks = np.load(f'{stem}_peak_fits.npy')

# Retain only rows where the Gaussian fit succeeded (r_fit_px not NaN)
valid        = np.isfinite(peaks[:, 2])
n_nan_dropped = int((~valid).sum())
peaks_ok     = peaks[valid]

if peaks_ok.shape[0] < 2:
    raise InsufficientRingsError(
        f"Only {peaks_ok.shape[0]} valid fitted peak(s) after NaN filter "
        f"({n_nan_dropped} dropped); need ≥ 2 for Tolansky analysis."
    )
if n_nan_dropped > 0:
    warnings.warn(f"{n_nan_dropped} peak row(s) with failed Gaussian fit dropped.")

p_arr         = peaks_ok[:, 0]    # 1-based ring indices (float)
r_fit_px      = peaks_ok[:, 2]    # centroid radii (px)
sigma_r_px    = peaks_ok[:, 3]    # 1σ centroid uncertainty (px)
r2_px2        = peaks_ok[:, 4]    # r²_p  (px²)
sigma_r2_px2  = peaks_ok[:, 5]    # σ(r²_p) = 2·r·σ_r  (px²)
```

### 3.2 Ancillary scalars

| Parameter | Symbol | Default | Source |
|-----------|--------|---------|--------|
| Vacuum wavelength | λ_OI_m | 630.0304 × 10⁻⁹ m | rest-frame OI 630 nm |
| Pixel pitch | pixel_pitch_m | 32 × 10⁻⁶ m | 2×2 binned CCD97 |
| Air refractive index | n_air | 1.0 | etalon gap fill |
| Calibration gap (optional) | d_cal_m | None | S13a `TolanskyResult.d_m` |
| Calibration gap uncertainty | sigma_d_cal_m | None | S13a `TolanskyResult.sigma_d_m` |
| Calibration focal length (optional) | f_cal_px | None | S13a `TolanskyResult.f_px` |

---

## 4. Algorithm

### Step 1 — Successive r²-differences and simple mean Δ

```python
delta_arr    = np.diff(r2_px2)
Delta_simple = np.mean(delta_arr)
cv_delta     = delta_arr.std() / delta_arr.mean()
```

**Uniformity check:** warn if `cv_delta > 0.05` (5%).  A high CV may
indicate a spurious detection or a mis-identified ring in the upstream
peak finder.

### Step 2 — WLS linear fit for Δ and ε (Vaughan §3.5.2 Steps 3–4)

Fit the model `r²_p = S · p + b` using weights `w_p = 1 / σ(r²_p)²`.
The uncertainty σ(r²_p) is taken directly from col 5 of the peak table
(`sigma_r_fit_sq_px2 = 2·r_fit·σ_r_fit`), already computed by
`annular_reduction.py`:

```python
w = 1.0 / sigma_r2_px2**2

sum_w    = np.sum(w)
sum_wp   = np.sum(w * p_arr)
sum_wp2  = np.sum(w * p_arr**2)
sum_wr2  = np.sum(w * r2_px2)
sum_wpr2 = np.sum(w * p_arr * r2_px2)

Lambda = sum_w * sum_wp2 - sum_wp**2

S = (sum_w * sum_wpr2 - sum_wp * sum_wr2) / Lambda    # slope  = Δ
b = (sum_wp2 * sum_wr2 - sum_wp * sum_wpr2) / Lambda  # intercept

var_S = sum_w  / Lambda
var_b = sum_wp2 / Lambda
```

Fractional order and its uncertainty:

```
ε       =  1 + b / S
σ(ε)²  =  (σ_b / S)²  +  (b · σ_S / S²)²
```

Reduced χ²:

```
χ²_dof  =  Σ w_p · (r²_p − S·p − b)²  /  (N_rings − 2)
```

Values `χ²_dof ≫ 1` indicate under-estimated σ(r_p) in the upstream
Gaussian fits or a mis-assigned ring.

### Step 3 — Recover α (Mode A, requires d_cal_m)

```python
if d_cal_m is not None:
    Delta       = S
    sigma_Delta = np.sqrt(var_S)
    alpha       = np.sqrt(LAM_OI_M * n_air / (d_cal_m * Delta))
    sigma_alpha = 0.5 * alpha * np.sqrt(
        (sigma_Delta / Delta)**2 + (sigma_d_cal_m / d_cal_m)**2
    )
    f_px       = 1.0 / alpha
    sigma_f_px = sigma_alpha / alpha**2
else:
    alpha = sigma_alpha = f_px = sigma_f_px = None
```

### Step 4 — Δ consistency check against S13a calibration prediction

If the caller supplies both `f_cal_px` and `d_cal_m` from the S13a result:

```python
Delta_pred    = f_cal_px**2 * LAM_OI_M / (n_air * d_cal_m)
delta_frac    = abs(Delta - Delta_pred) / Delta_pred
if delta_frac > 0.01:
    warnings.warn(
        f"Δ_obs/Δ_pred discrepancy = {delta_frac*100:.2f}% > 1%  "
        "— possible scale change or ring misidentification."
    )
```

---

## 5. Output data class

```python
@dataclass
class TolanskyResult1L:
    """
    Output of the Tolansky single-line analysis (S13b).
    All two_sigma_ fields are exactly 2 × sigma_ (S04 convention).
    Mode-B fields are None when d_cal_m was not supplied.
    """
    # --- Ring data (from _peak_fits.npy, valid rows only) ---
    n_rings:       int             # number of valid fitted rings used
    n_nan_dropped: int             # number of NaN (failed-fit) rows discarded
    p_arr:         np.ndarray      # ring indices (1-based float), shape (n_rings,)
    r_fit_px:      np.ndarray      # Gaussian centroid radii  (px)
    sigma_r_px:    np.ndarray      # 1σ centroid uncertainty  (px)
    r2_px2:        np.ndarray      # r²_p  (px²)
    sigma_r2_px2:  np.ndarray      # σ(r²_p) = 2·r·σ_r  (px²)
    delta_arr:     np.ndarray      # successive differences δ_p (px²)
    cv_delta:      float           # CV = std(δ)/mean(δ)

    # --- WLS fit (Vaughan §3.5.2, Steps 3–4) ---
    Delta:           float         # Δ = best-fit r²-slope  (px²/ring)
    sigma_Delta:     float         # 1σ
    two_sigma_Delta: float         # exactly 2 × sigma_Delta (S04)
    eps:             float         # ε = fractional order at axis, ∈ [0,1)
    sigma_eps:       float         # 1σ
    two_sigma_eps:   float         # exactly 2 × sigma_eps (S04)
    chi2_dof:        float         # reduced χ²

    # --- Mode A outputs (None if d_cal_m not supplied) ---
    alpha_rad_px:    float | None  # α = 1/f_px  (rad/px)
    sigma_alpha:     float | None  # 1σ
    two_sigma_alpha: float | None  # exactly 2 × sigma_alpha (S04)
    f_px:            float | None  # focal length (pixels)
    sigma_f_px:      float | None  # 1σ

    # --- Δ consistency check (None if f_cal_px or d_cal_m not supplied) ---
    Delta_pred:          float | None  # predicted Δ_OI from S13a f_cal and d_cal
    Delta_consistency:   float | None  # |Δ_obs − Δ_pred| / Δ_pred

    # --- Provenance ---
    lam_OI_nm:  float         # 630.0304 (input wavelength, nm)
    d_cal_mm:   float | None  # d_cal used (mm), or None
```

---

## 6. Printed summary

The `print_tolansky_1line(result)` function prints:

```
=== TOLANSKY SINGLE-LINE ANALYSIS  (S13b) ===
Source line:  OI 630.0 nm  (λ = 630.0304 nm vacuum)
Rings used: N   (p = 1 … N)   [X NaN row(s) dropped]

  p  :       1         2         3      …       N
  r² :  XXXXX.XX  XXXXX.XX  XXXXX.XX  …  XXXXX.XX  (px²)
                 δ₁₂=XX.X   δ₂₃=XX.X  …
  CV(δ) = X.XXX   [PASS / WARN]

WLS FIT  (Vaughan §3.5.2)
  Δ       = XXXX.XX ± X.XX px²   (2σ = X.XX px²)
  ε       = X.XXXX  ± X.XXXX
  χ²_dof  = X.XX

MODE A  (d_cal: XX.XXXX mm / not supplied)
  α   = X.XXXXE-4 ± X.XXXXE-6 rad/px   (2σ = X.XXXXE-6)
  f   = XXXXX.X   ± X.X px

CONSISTENCY CHECK  (f_cal, d_cal supplied: YES/NO)
  Δ_pred = XXXX.XX px²
  |Δ_obs − Δ_pred| / Δ_pred = X.X%   [PASS / WARN]
==============================================
```

---

## 7. Integration with M05 (S14)

When `d_cal_m` is supplied, the single-line result can seed M05 priors as
a fallback or independent cross-check:

```python
def to_m05_priors_1line(result: TolanskyResult1L) -> dict | None:
    """
    Returns M05 prior dict if Mode A succeeded (d_cal available),
    else returns None and caller must use S13a priors.
    """
    if result.alpha_rad_px is None:
        return None
    return {
        'alpha_init':   result.alpha_rad_px,
        'alpha_bounds': (result.alpha_rad_px * 0.875,
                         result.alpha_rad_px * 1.125),   # ±12.5%
        'epsilon_sci':  result.eps,    # ε for OI 630 nm
    }
```

> **Important:** S13b does not supply `t_init_mm` or `t_bounds_mm` to M05.
> Those fields always come from S13a.

---

## 8. Verification tests

All 6 tests in `tests/test_tolansky_1line_2026-05-05.py`.

A shared helper `make_synthetic_peaks_array(Delta, eps, n, sigma_r=0.3)`
generates a valid `(n, 9)` float64 array matching the `annular_reduction.py`
column layout, for use across all tests:

```python
def make_synthetic_peaks_array(Delta, eps, n, sigma_r=0.3):
    p   = np.arange(1, n + 1, dtype=float)
    r2  = Delta * (p - 1 + eps)
    r   = np.sqrt(r2)
    sr  = np.full(n, sigma_r)
    sr2 = 2.0 * r * sr
    # cols: peak_num | r_raw_px | r_fit_px | sigma_r | r2 | sigma_r2 | amp | width | chi2
    return np.column_stack([p, r, r, sr, r2, sr2,
                            np.full(n, 500.0), np.full(n, 1.5), np.ones(n)])
```

### T1 — WLS recovers known Δ and ε on noise-free synthetic data

```python
def test_wls_single_line_known_answer():
    """Exact synthetic r²; Δ and ε must be recovered to < 0.01%."""
    Delta_true = 477.0
    eps_true   = 0.42
    arr = make_synthetic_peaks_array(Delta_true, eps_true, n=7)
    result = run_tolansky_1line(arr, d_cal_m=None)
    assert abs(result.Delta - Delta_true) / Delta_true < 1e-4
    assert abs(result.eps   - eps_true)                < 1e-4
```

### T2 — Successive differences uniform on exact data

```python
def test_successive_differences_uniform_1line():
    Delta_true = 477.0
    arr   = make_synthetic_peaks_array(Delta_true, 0.42, n=7)
    result = run_tolansky_1line(arr, d_cal_m=None)
    assert result.cv_delta < 1e-10
```

### T3 — Mode A α recovery with known d_cal

```python
def test_alpha_recovery_mode_a():
    """With known d_cal and Δ, α must be recovered to < 0.1%."""
    LAM_OI   = 630.0304e-9
    d_cal    = 20.0006e-3
    f_true   = 6230.0
    alpha_true = 1.0 / f_true
    Delta_true = f_true**2 * LAM_OI / d_cal
    arr    = make_synthetic_peaks_array(Delta_true, 0.42, n=7)
    result = run_tolansky_1line(arr, d_cal_m=d_cal, sigma_d_cal_m=5e-6)
    assert abs(result.alpha_rad_px - alpha_true) / alpha_true < 1e-3
```

### T4 — Mode B: α is None when d_cal not supplied

```python
def test_mode_b_no_alpha():
    arr    = make_synthetic_peaks_array(477.0, 0.42, n=5)
    result = run_tolansky_1line(arr, d_cal_m=None)
    assert result.alpha_rad_px is None
    assert result.Delta is not None
```

### T5 — NaN rows dropped with correct count

```python
def test_nan_rows_dropped():
    arr = make_synthetic_peaks_array(477.0, 0.42, n=5)
    arr[2, 2:] = np.nan   # invalidate row index 2 (peak 3)
    result = run_tolansky_1line(arr, d_cal_m=None)
    assert result.n_nan_dropped == 1
    assert result.n_rings == 4
```

### T6 — All two_sigma_ fields are exactly 2 × sigma_ (S04)

```python
def test_two_sigma_fields_1line():
    arr    = make_synthetic_peaks_array(477.0, 0.42, n=7)
    result = run_tolansky_1line(arr, d_cal_m=20.0006e-3, sigma_d_cal_m=5e-6)
    assert abs(result.two_sigma_Delta - 2.0 * result.sigma_Delta) < 1e-14
    assert abs(result.two_sigma_eps   - 2.0 * result.sigma_eps)   < 1e-14
    assert abs(result.two_sigma_alpha - 2.0 * result.sigma_alpha) < 1e-14
```

---

## 9. Expected numerical values (WindCube FlatSat, OI 630 nm)

Using D_25C_MM = 20.0006 mm, pixel_pitch = 32 µm:

| Quantity | Expected | Notes |
|---------|----------|-------|
| Δ (OI 630 nm) | ~477 px² | Δ_OI ≈ 0.984 × Δ_Ne_a |
| α | ~1.607 × 10⁻⁴ rad/px | same instrument scale as S13a |
| f_px | ~6230 px | sqrt(Δ·d/λ_OI) |
| ε | instrument-dependent | typically 0.2–0.8 |
| χ²_dof | ~1.0 for good data | WLS quality |
| CV(δ) | < 0.02 for clean fringes | uniformity |

---

## 10. File locations

```
soc_sewell/
├── src/fpi/
│   └── tolansky_1line_2026-05-05.py
├── tests/
│   └── test_tolansky_1line_2026-05-05.py
└── docs/specs/
    └── S13b_tolansky_1_line_2026-05-05.md
```

---

## 11. Instructions for Claude Code

### Pre-implementation reads

Before writing any code, read in full:

1. `docs/specs/S13b_tolansky_1_line_2026-05-05.md` (this file)
2. `docs/specs/S13a_tolansky_2_line_2026-05-05.md` §4 Step 4 — WLS
   formulation to reuse verbatim
3. `ingest/annular_reduction.py` lines 607–628 — exact column layout of
   `_peak_fits.npy`, and lines 610–620 for the precise save loop

Confirm annular_reduction tests pass first:

```bash
cat PIPELINE_STATUS.md
pytest tests/ -v -k "annular"
```

### Task sequence

**Task 1 — `InsufficientRingsError`**

Define a custom exception class for the case where fewer than 2 valid
fitted rings remain after NaN filtering.

**Task 2 — Input loading helper**

Implement `load_peak_fits(path_or_array)` that:
1. Accepts either a file path string/Path or a pre-loaded `(N, 9)` ndarray
   (the latter allows direct use in tests without touching the filesystem).
2. If a path, loads with `np.load`.
3. Filters to rows where `col 2` (r_fit_px) is `np.isfinite`.
4. Counts and warns on dropped NaN rows.
5. Returns `(p_arr, r_fit_px, sigma_r_px, r2_px2, sigma_r2_px2, n_nan_dropped)`.
6. Raises `InsufficientRingsError` if fewer than 2 valid rows remain.

Column mapping (zero-indexed):
- col 0 → `peak_num` → `p_arr`
- col 2 → `r_fit_px`
- col 3 → `sigma_r_fit_px` → `sigma_r_px`
- col 4 → `r_fit_sq_px2` → `r2_px2`
- col 5 → `sigma_r_fit_sq_px2` → `sigma_r2_px2`

**Task 3 — Single-line WLS helper**

Implement `run_single_line_wls(p, r2, sigma_r2)` using the closed-form
normal equations from §4 Step 2.  The weight denominator is
`sigma_r2**2` directly (col 5 values are already `2·r·σ_r`).

Return a dict: `Delta`, `sigma_Delta`, `eps`, `sigma_eps`, `chi2_dof`,
`delta_arr`, `cv_delta`.

**Task 4 — Mode A α recovery helper**

Implement `recover_alpha(Delta, sigma_Delta, d_cal_m, sigma_d_cal_m, lam_m)`.
Returns `(alpha, sigma_alpha, f_px, sigma_f_px)` or `(None,None,None,None)`
if `d_cal_m is None`.

**Task 5 — Top-level `run_tolansky_1line()`**

```python
def run_tolansky_1line(
    peaks_input:   np.ndarray | str | pathlib.Path,  # (N,9) array or file path
    lam_OI_m:      float = 630.0304e-9,
    pixel_pitch_m: float = 32e-6,
    n_air:         float = 1.0,
    d_cal_m:       float | None = None,
    sigma_d_cal_m: float | None = None,
    f_cal_px:      float | None = None,
) -> TolanskyResult1L:
```

Calls Tasks 1–4 in sequence.  Sets all `two_sigma_` fields to exactly
`2.0 × sigma_`.  Performs the Δ consistency check (§4 Step 4) when both
`f_cal_px` and `d_cal_m` are supplied.

**Task 6 — `print_tolansky_1line()`**

Implement the formatted table from §6.

**Task 7 — `to_m05_priors_1line()`**

Implement §7 exactly.

**Task 8 — Tests (6/6 must pass)**

Place `make_synthetic_peaks_array()` as a module-level helper in the test
file (not in the production module).

```bash
pytest tests/test_tolansky_1line_2026-05-05.py -v
pytest tests/ -v   # no regressions
```

**Task 9 — Commit**

```bash
# Update PIPELINE_STATUS.md — add S13b, set status and date
git add src/fpi/tolansky_1line_2026-05-05.py
git add tests/test_tolansky_1line_2026-05-05.py
git add PIPELINE_STATUS.md
git commit -m "feat(S13b): single-line Tolansky from annular_reduction peak_fits.npy, 6/6 tests pass

Also updates PIPELINE_STATUS.md"
```

### Module docstring

```python
"""
Module:      tolansky_1line_2026-05-05.py
Spec:        docs/specs/S13b_tolansky_1_line_2026-05-05.md
Reference:   Vaughan (1989) The Fabry-Perot Interferometer, §3.5.2
             Equations (3.83)–(3.86) — single-line r²-spacing only
             Harding et al. (2014) ApplOpt 53, 666 — WindCube forward model
             Mulligan (1986) J. Phys. E 19, 545 — annular r²-bin reduction
Author:      Claude Code
Project:     WindCube FPI Pipeline — NCAR/HAO
Repo:        soc_sewell
Input:       {stem}_peak_fits.npy  (9-column float64, from annular_reduction.py)
             Peak finding and Gaussian fitting are performed by annular_reduction.py.
             This module reads the pre-computed table directly — no peak detection here.
Note:        Single OI 630.0 nm airglow image analysis only.
             Recovers Δ, ε; recovers α in Mode A (d_cal supplied).
             Does NOT recover absolute d.
             For neon two-line Benoit d recovery, see S13a / tolansky_2line.py
"""
```

### Report format

```
=== S13b CLAUDE CODE REPORT ===
Date: YYYY-MM-DD
Module: src/fpi/tolansky_1line_2026-05-05.py
Tests: N/6 pass

TOLANSKY SINGLE-LINE RESULTS:
  Rings used: N  (NaN rows dropped: X)
  Δ       = XXXX.XX ± X.XX px²   (2σ = X.XX px²)
  ε       = X.XXXX  ± X.XXXX
  CV(δ)   = X.XXX   [PASS / WARN]
  χ²_dof  = X.XX
  Mode A  (d_cal supplied: YES/NO):
    α   = X.XXXXE-4 ± X.XXXXE-6 rad/px
    f   = XXXXX.X   ± X.X px
  Δ consistency: X.X%   [PASS / WARN]

DEVIATIONS FROM SPEC:
  [list any, or "None"]
================================
```

Stop and return this report if any task takes more than 15 minutes
without all relevant tests passing.
