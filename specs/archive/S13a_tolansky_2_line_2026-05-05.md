# S13a — Tolansky Two-Line Analysis (Neon Calibration Lamp)

**Spec ID:** S13a
**Spec file:** `docs/specs/S13a_tolansky_2_line_2026-05-05.md`
**Project:** WindCube FPI Pipeline
**Institution:** NCAR / High Altitude Observatory (HAO)
**Status:** v0.4 — 2026-05-05
**Depends on:** `annular_reduction.py` — provides `{stem}_peak_fits.npy`
**Used by:**
  - M05 (S14) — receives `TolanskyResult` as informed priors for α and ε
**References:**
  - Vaughan (1989) *The Fabry-Perot Interferometer*, §3.5.2 "Analysis of
    photographic recordings", pp. 116–121.  **Equations 3.83–3.97 are
    the authoritative derivation for all analysis steps below.**
  - Benoit (1898) — exact-fractions method for plate-spacing recovery
  - GNL4096-R iss1 WindCube Etalon Assembly (ICOS build report, Dec 2023)
  - Burns, Adams & Longwell (1950) — Ne IAU "S" standard wavelengths
**Sibling spec:** S13b (single-line Tolansky for airglow radial profile analysis)
**Supersedes:** S13a_tolansky_2_line_2026-05-05.md v0.3
**Last updated:** 2026-05-05

> **Revision note v0.4 (2026-05-05):**
> (1) Focal length f removed throughout.  α is the primary recovered plate-scale
> parameter; f is redundant because α = sqrt(λ/(d·Δ)) absorbs it completely.
> (2) Input changed from a structured `_fringe_peaks.npy` with a `family` string
> field to the plain 9-column float64 `_peak_fits.npy` produced by
> `annular_reduction.py`.  Family assignment is now done by amplitude threshold
> (640 nm line is ~3× brighter than 638 nm).
> (3) Expected Δ values updated from ~485 px² to ~1233 px² to match actual
> FlatSat data (wider fringe spacing than earlier synthetic estimates).
> (4) Reference plot added as §10.

---

## 1. Purpose

Given a set of calibration fringe ring radii for **two known neon wavelengths**,
recover the etalon physical parameters that M05 (S14) needs as starting priors:

| Parameter | Symbol | Physical meaning |
|-----------|--------|-----------------|
| Plate scale | α | Angular pixel scale (rad/px); primary geometric parameter |
| r²-spacing | Δₐ, Δᵦ | Mean successive r²-differences for each line (px²) |
| Fractional orders | εₐ, εᵦ | Fractional interference order at optical axis for each line |
| Plate spacing | d | Etalon mirror separation (metres); recovered via Benoit |

> **Why α, not f.**  The focal length f (pixels) and α (rad/px) carry identical
> information: α = 1/f_px.  Working in α avoids a derived intermediate quantity
> and is dimensionally consistent with the angular convention used throughout
> the pipeline.  f is not reported or stored anywhere in this module.

The analysis follows Tolansky's rectangular-array method as described in
Vaughan (1989) §3.5.2, equations (3.83)–(3.97), with WLS refinement to obtain
propagated 1σ uncertainties.

**Scope.**  This module operates on **neon calibration lamp images only**.
It requires two spectrally distinct emission lines (λₐ and λᵦ) so that
Benoit's exact-fractions method can resolve the plate-spacing ambiguity.
For airglow science images (single OI 630 nm line), see S13b.

---

## 2. Input

### 2.1 Peak fits array

S13a reads the `_peak_fits.npy` file saved by `annular_reduction.py`:

```python
peaks = np.load('{stem}_peak_fits.npy')   # shape (N_peaks, 9), float64
```

The 9 columns are (zero-indexed), exactly as written by `annular_reduction.py`:

| Col | Name | Units | Description |
|-----|------|-------|-------------|
| 0 | `peak_num` | — | 1-based sequential index across both families |
| 1 | `r_raw_px` | px | Detected bin-centre radius (`find_peaks`) |
| 2 | `r_fit_px` | px | TRF Gaussian centroid μ; NaN if fit failed |
| 3 | `sigma_r_fit_px` | px | 1σ uncertainty on μ; NaN if fit failed |
| 4 | `r_fit_sq_px2` | px² | μ²; NaN if fit failed |
| 5 | `sigma_r_fit_sq_px2` | px² | 2·μ·σ_μ (propagated); NaN if fit failed |
| 6 | `amplitude_adu` | ADU | Gaussian amplitude A above background |
| 7 | `width_sigma_px` | px | Gaussian σ-width; NaN if fit failed |
| 8 | `reduced_chi2` | — | χ²/(n_points−4); NaN if fit failed |

**Expected row count:** 20 rows for a typical FlatSat calibration image
(10 rings per line × 2 lines, interleaved in radius).

### 2.2 Family assignment by amplitude

The two neon lines are physically interleaved in the ring pattern and arrive
in the peak table sorted by radius, not by wavelength.  They are separated by
amplitude: the 640.2248 nm line is approximately **3× brighter** than the
638.2991 nm line (FlatSat intensity ratio Y_B ≈ 0.30; see §9).

```python
# 1. Filter to valid (non-NaN) rows
valid       = np.isfinite(peaks[:, 2])
n_nan_dropped = int((~valid).sum())
peaks_ok    = peaks[valid]

if peaks_ok.shape[0] < 4:
    raise InsufficientRingsError(
        f"Only {peaks_ok.shape[0]} valid fitted peaks; need ≥ 4 (≥2 per family)."
    )

amps = peaks_ok[:, 6]   # col 6: amplitude_adu

# 2. Split on median amplitude
amp_threshold = np.median(amps)
mask_a = amps > amp_threshold    # 640.2248 nm — brighter line
mask_b = amps <= amp_threshold   # 638.2991 nm — dimmer line

peaks_a = peaks_ok[mask_a]   # sorted by radius (ascending r)
peaks_b = peaks_ok[mask_b]

if peaks_a.shape[0] < 2 or peaks_b.shape[0] < 2:
    raise InsufficientRingsError(
        "Family split by amplitude produced < 2 rings in one family.  "
        "Check peak detection parameters."
    )

# 3. Assign 1-based ring indices within each family
p_a = np.arange(1, peaks_a.shape[0] + 1, dtype=float)
p_b = np.arange(1, peaks_b.shape[0] + 1, dtype=float)

r2_a      = peaks_a[:, 4]    # r²_p for λₐ  (px²)
sigma_r2_a = peaks_a[:, 5]   # σ(r²_p) for λₐ  (px²)
r2_b      = peaks_b[:, 4]
sigma_r2_b = peaks_b[:, 5]
```

> **Amplitude ratio sanity check:** compute
> `Y_B_obs = median(amps[mask_b]) / median(amps[mask_a])`.
> Warn if `Y_B_obs < 0.15` or `Y_B_obs > 0.60`; this range covers normal
> FlatSat operating conditions.  A value outside the range may indicate a
> bad exposure, wrong dark subtraction, or inverted family assignment.

### 2.3 WindCube numerical constants

| Constant | Value | Source |
|---------|-------|--------|
| λₐ | 640.2248 × 10⁻⁹ m | Burns et al. 1950; IAU standard |
| λᵦ | 638.2991 × 10⁻⁹ m | Burns et al. 1950; IAU standard |
| pixel_pitch | 32 × 10⁻⁶ m | 2×2 binned CCD97 (2 × 16 µm) |
| d_prior | 20.008 × 10⁻³ m | ICOS build report GNL4096-R (spacer measurement) |
| n_air | 1.0 | Air gap |

---

## 3. Notation and correspondence with Vaughan (1989) §3.5.2

Vaughan works with **ring diameters** D_p (mm) and their squares D²_p.
This module works with **ring radii** r_p (pixels) and their squares r²_p.
Since D_p = 2 r_p:

```
r²_p = D²_p / 4
```

All of Vaughan's equations involving D² translate to r² with the factor-of-4
substitution made once here and silently absorbed throughout.

| Vaughan symbol | Our symbol | Units | Meaning |
|---------------|-----------|-------|---------|
| ₐD²_p / 4 | ₐr²_p | px² | Radius-squared of pth ring, line a |
| Δₐ (Vaughan D² form) / 4 | Δₐ | px² | Mean successive r²-difference, line a |
| δₐ_p | δₐ_p | px² | p-th successive difference |
| ₐε_c | εₐ | — | Fractional interference order at centre, line a |
| nₐ | nₐ | — | Integer interference order at centre, line a |

**Ring indexing:** p = 1 for innermost ring (Vaughan uses p = 0; we offset by 1).

---

## 4. Analysis — equations (3.83) through (3.97)

### Step 1 — Successive r²-differences (Vaughan Eqs. 3.85, 3.87)

For a Fabry-Pérot fringe pattern at high interference order,
the paraxial approximation gives rings equally spaced in r²:

```
ₐr²_p  =  Δₐ · (p − 1 + εₐ)                          [3.84 in r² form]
```

Successive r²-differences are therefore constant:

```
Δₐ  =  f²_px · λₐ / (n_air · d)  =  λₐ / (n_air · d · α²)   [3.85]
Δᵦ  =  f²_px · λᵦ / (n_air · d)  =  λᵦ / (n_air · d · α²)   [3.87]
```

Note: both expressions on the right are equivalent; neither f nor α
appears in the Tolansky analysis itself — they are derived **from** Δ and d
after the Benoit recovery (Step 6).

Compute simple mean differences as a first check:

```python
delta_a = np.diff(r2_a)    # δₐ_p, length n_a − 1
delta_b = np.diff(r2_b)

Delta_a_simple = np.mean(delta_a)
Delta_b_simple = np.mean(delta_b)
```

**Consistency check (Eq. 3.85 / 3.87 ratio):**

```
Δₐ / Δᵦ  =  λₐ / λᵦ  =  640.2248 / 638.2991  =  1.003017   [3.85/3.87]
```

Deviation from this ratio in excess of ~0.2% signals a family
mis-assignment or a mis-measured ring.

### Step 2 — Rectangular array (Vaughan Table 3.1)

Tolansky's rectangular array lays out r²_p values as columns (ring
number 1, 2, …) and lines as rows (a, b).  Individual differences δ_p
are written between successive columns; any δ_p that differs markedly
from the mean Δ flags a bad ring.  The module renders this as a formatted
printed table (see §6).

### Step 3 — Fractional interference orders (Vaughan Eqs. 3.86, 3.88)

```
εₐ  =  ₐr²_p / Δₐ  −  (p − 1)    for any ring p          [3.86]
εᵦ  =  ᵦr²_p / Δᵦ  −  (p − 1)    for any ring p          [3.88]
```

Computed per-ring and averaged; WLS gives the best estimates (Step 4).

**Sanity check:** `0 ≤ εₐ < 1` and `0 ≤ εᵦ < 1`.

### Step 4 — WLS refinement

Fit model `r²_p = S · p + b` with weights `w_p = 1/σ(r²_p)²`:

```python
def _wls(p, r2, sigma_r2):
    w        = 1.0 / sigma_r2**2
    sum_w    = np.sum(w)
    sum_wp   = np.sum(w * p)
    sum_wp2  = np.sum(w * p**2)
    sum_wr2  = np.sum(w * r2)
    sum_wpr2 = np.sum(w * p * r2)
    Lambda   = sum_w * sum_wp2 - sum_wp**2
    S        = (sum_w * sum_wpr2 - sum_wp * sum_wr2) / Lambda
    b        = (sum_wp2 * sum_wr2 - sum_wp * sum_wpr2) / Lambda
    var_S    = sum_w  / Lambda
    var_b    = sum_wp2 / Lambda
    eps      = 1.0 + b / S
    sig_eps  = np.sqrt((np.sqrt(var_b)/S)**2 + (b*np.sqrt(var_S)/S**2)**2)
    chi2_dof = np.sum(w * (r2 - S*p - b)**2) / (len(p) - 2)
    return dict(Delta=S, sigma_Delta=np.sqrt(var_S),
                eps=eps, sigma_eps=sig_eps,
                chi2_dof=chi2_dof,
                intercept=b, sigma_intercept=np.sqrt(var_b))
```

Apply to each family independently.

**Reduced χ²:** values `χ²_dof ≫ 1` indicate under-estimated σ(r_p) or
a mis-assigned ring; `χ²_dof ≪ 1` indicates over-estimated σ(r_p).

### Step 5 — Identify integer order difference N_Δ (Vaughan Eq. 3.96)

```
N_Δ  ≡  nₐ − nᵦ  =  round(2 · d_prior · (1/λₐ − 1/λᵦ))   [3.96]
```

`d_prior = 20.008 mm` resolves the FSR-period integer ambiguity only; it
does **not** bias the recovered d.

For WindCube: **N_Δ = −189**

### Step 6 — Recover plate spacing d (Vaughan Eq. 3.97 / Benoit)

```
d  =  (N_Δ + εₐ − εᵦ) · λₐ·λᵦ / (2·n_air·(λᵦ − λₐ))     [3.97]
```

Note λᵦ < λₐ so (λᵦ − λₐ) < 0, and N_Δ = −189 < 0, giving d > 0.

Uncertainty (εₐ and εᵦ are independent):

```python
factor  = lam_a * lam_b / (2 * n_air * abs(lam_b - lam_a))
sigma_d = factor * np.sqrt(sig_eps_a**2 + sig_eps_b**2)
```

### Step 7 — Recover plate scale α

From Eq. (3.85) rearranged (eliminating f entirely):

```
α  =  sqrt(λₐ · n_air / (d · Δₐ))                         [from 3.85]
```

This is the **primary geometric output** of S13a.  Propagate uncertainty:

```python
alpha     = np.sqrt(lam_a * n_air / (d * Delta_a))
sig_alpha = 0.5 * alpha * np.sqrt((sig_Delta_a/Delta_a)**2 + (sig_d/d)**2)
```

**Cross-check α from line b:**

```
α_b  =  sqrt(λᵦ · n_air / (d · Δᵦ))
```

Acceptance criterion: `|α_a − α_b| / α_a < 0.001` (1000 ppm).

**α_mean** (reported, used as M05 prior):

```python
alpha_mean = 0.5 * (alpha_a + alpha_b)
```

> **Note on plot annotation.**  The reference plot (§10) labels the slopes
> of the P vs r² WLS fit as `α_640`, `α_638` in units of "orders/px²".
> These are `1/Δ` (the WLS slope), **not** the angular plate scale.  The
> plot then derives `α_rpx = sqrt(slope · λ / (2nd))`, which differs from
> our formula by a factor of `1/sqrt(2)`.  This module uses the form that
> follows directly from Vaughan Eq. 3.85 without the factor of 2 in the
> denominator: `α = sqrt(λ/(d·Δ))`.  The two forms correspond to different
> conventions for the interference condition (2nd cos θ vs nd cos θ); the
> Vaughan convention (2nd) is authoritative for this pipeline.  α values
> from this module (~1.607 × 10⁻⁴ rad/px) should not be directly compared
> with the plot's `α_rpx` (~1.14 × 10⁻⁴ rad/px).

---

## 5. Output data class

```python
@dataclass
class TolanskyResult:
    """
    Output of the Tolansky two-line analysis (S13a).
    All two_sigma_ fields are exactly 2 × sigma_ (S04 convention).
    Focal length f is not reported; α is the sole plate-scale parameter.
    """
    # --- Family assignment provenance ---
    n_peaks_total:   int    # total rows in _peak_fits.npy
    n_nan_dropped:   int    # rows dropped (NaN r_fit_px)
    n_rings_a:       int    # rings used for λₐ family
    n_rings_b:       int    # rings used for λᵦ family
    amp_threshold:   float  # median amplitude used for family split (ADU)
    Y_B_obs:         float  # median_amp_b / median_amp_a  (intensity ratio)

    # --- Line a  (λₐ = 640.2248 nm) ---
    Delta_a:         float        # Δₐ  (px²)                  [Eq. 3.85]
    sigma_Delta_a:   float        # 1σ
    two_sigma_Delta_a: float      # exactly 2 × sigma_Delta_a
    eps_a:           float        # εₐ                         [Eq. 3.86]
    sigma_eps_a:     float        # 1σ
    two_sigma_eps_a: float        # exactly 2 × sigma_eps_a
    chi2_dof_a:      float        # reduced χ²
    delta_a:         np.ndarray   # successive r²-differences (px²)

    # --- Line b  (λᵦ = 638.2991 nm) ---
    Delta_b:         float        # Δᵦ  (px²)                  [Eq. 3.87]
    sigma_Delta_b:   float
    two_sigma_Delta_b: float
    eps_b:           float        # εᵦ                         [Eq. 3.88]
    sigma_eps_b:     float
    two_sigma_eps_b: float
    chi2_dof_b:      float
    delta_b:         np.ndarray

    # --- Consistency check ---
    Delta_ratio_obs:      float   # Δₐ/Δᵦ observed
    Delta_ratio_expected: float   # λₐ/λᵦ = 1.003017
    Delta_ratio_residual: float   # |obs − expected| / expected

    # --- Integer disambiguation ---
    N_Delta: int                  # N_Δ = nₐ − nᵦ  [Eq. 3.96 / Benoit]

    # --- Plate spacing recovery  [Eq. 3.97] ---
    d_m:             float        # recovered d  (metres)
    sigma_d_m:       float        # 1σ
    two_sigma_d_m:   float        # exactly 2 × sigma_d_m

    # --- Plate scale α (primary geometric output) ---
    alpha_a:         float        # α from line a  (rad/px)    [from 3.85]
    alpha_b:         float        # α from line b  (rad/px)  cross-check
    alpha_mean:      float        # mean of α_a and α_b  (rad/px)
    sigma_alpha:     float        # 1σ on α_mean
    two_sigma_alpha: float        # exactly 2 × sigma_alpha
    alpha_consistency: float      # |α_a − α_b| / α_a  (accept if < 0.001)

    # --- Wavelengths (provenance) ---
    lam_a_nm: float               # 640.2248
    lam_b_nm: float               # 638.2991
```

---

## 6. Rectangular array table (Vaughan Table 3.1 analog)

```
=== TOLANSKY RECTANGULAR ARRAY  (Vaughan 1989, Table 3.1 analog) ===

Family assignment:  amp_threshold = XXXX.X ADU   Y_B_obs = X.XXX
  640.2248 nm (line a):  N rings = XX   median amp = XXXX ADU
  638.2991 nm (line b):  N rings = XX   median amp = XXXX ADU

Component a  (λₐ = 640.2248 nm)
  p  :      1           2           3       …      N
  r²  :  XXXXX.XX   XXXXX.XX   XXXXX.XX  …  XXXXX.XX   (px²)
                δ₁₂=XXXX.X   δ₂₃=XXXX.X  …
  Δₐ (WLS slope) = XXXXXXX.XX ± X.XX px²   χ²_dof = X.XX
  εₐ             = X.XXXX     ± X.XXXX

Component b  (λᵦ = 638.2991 nm)
  p  :      1           2           3       …      N
  r²  :  XXXXX.XX   XXXXX.XX   XXXXX.XX  …  XXXXX.XX   (px²)
                δ₁₂=XXXX.X   δ₂₃=XXXX.X  …
  Δᵦ (WLS slope) = XXXXXXX.XX ± X.XX px²   χ²_dof = X.XX
  εᵦ             = X.XXXX     ± X.XXXX

Ratio  Δₐ/Δᵦ observed = X.XXXXXX   expected (λₐ/λᵦ) = 1.003017
        residual = XX.X ppm   [PASS if < 200 ppm]

=== BENOIT RECOVERY  (Vaughan Eqs. 3.94–3.97) ===
  N_Δ = nₐ − nᵦ = -189   [from d_prior = 20.008 mm, Eq. 3.96]
  d   = XX.XXXX ± X.XXXX mm   (2σ = X.XXXX mm)

=== PLATE SCALE ===
  α_a  = X.XXXXE-4 ± X.XXXXE-6 rad/px   [from Δₐ, d, λₐ]
  α_b  = X.XXXXE-4 ± X.XXXXE-6 rad/px   [from Δᵦ, d, λᵦ; cross-check]
  α_mean = X.XXXXE-4 ± X.XXXXE-6 rad/px   (2σ = X.XXXXE-6)
  |α_a − α_b| / α_a = X.X ppm   [PASS if < 1000 ppm]
```

---

## 7. M05 priors handoff

```python
def to_m05_priors(result: TolanskyResult) -> dict:
    """Convert TolanskyResult to the prior dict expected by M05 FitConfig."""
    d_mm     = result.d_m * 1e3
    sig_d_mm = result.sigma_d_m * 1e3
    return {
        't_init_mm':      d_mm,
        't_bounds_mm':    (d_mm - 3*sig_d_mm, d_mm + 3*sig_d_mm),
        'alpha_init':     result.alpha_mean,
        'alpha_bounds':   (result.alpha_mean * 0.875,
                           result.alpha_mean * 1.125),   # ±12.5%
        'epsilon_cal_a':  result.eps_a,    # εₐ for λₐ = 640.2248 nm
        'epsilon_cal_b':  result.eps_b,    # εᵦ for λᵦ = 638.2991 nm
    }
```

---

## 8. Verification tests

All 7 tests in `tests/test_tolansky_2line_2026-05-05.py`.

A shared `make_synthetic_peaks_array(Delta_a, eps_a, Delta_b, eps_b, n, sigma_r=0.05)`
helper builds a valid `(2n, 9)` float64 array with two interleaved families
(640 nm rows with amplitude 1800 ADU, 638 nm rows with amplitude 600 ADU),
sorted by ascending radius.

### T1 — Successive differences uniform on exact synthetic data

```python
def test_successive_differences_uniform():
    arr = make_synthetic_peaks_array(1233.0, 0.22, 1228.0, 0.73, n=10)
    result = run_tolansky(arr)
    assert result.delta_a.std() / result.delta_a.mean() < 1e-10
    assert result.delta_b.std() / result.delta_b.mean() < 1e-10
```

### T2 — WLS recovers known Δ and ε to < 0.01%

```python
def test_wls_known_answer():
    Delta_a_true, eps_a_true = 1233.0, 0.22
    Delta_b_true, eps_b_true = 1228.0, 0.73
    arr = make_synthetic_peaks_array(Delta_a_true, eps_a_true,
                                     Delta_b_true, eps_b_true, n=10)
    result = run_tolansky(arr)
    assert abs(result.Delta_a - Delta_a_true) / Delta_a_true < 1e-4
    assert abs(result.eps_a   - eps_a_true)                  < 1e-4
    assert abs(result.Delta_b - Delta_b_true) / Delta_b_true < 1e-4
    assert abs(result.eps_b   - eps_b_true)                  < 1e-4
```

### T3 — Δ ratio constraint: Δₐ/Δᵦ = λₐ/λᵦ

```python
def test_delta_ratio_matches_wavelength_ratio():
    lam_a, lam_b = 640.2248e-9, 638.2991e-9
    d, alpha = 20.0e-3, 1.607e-4
    Delta_a = lam_a / (d * alpha**2)
    Delta_b = lam_b / (d * alpha**2)
    assert abs(Delta_a/Delta_b - lam_a/lam_b) / (lam_a/lam_b) < 1e-8
```

### T4 — N_Δ correctly identified from d_prior

```python
def test_N_Delta_from_prior():
    lam_a, lam_b = 640.2248e-9, 638.2991e-9
    d_prior = 20.008e-3
    N = round(2 * d_prior * (1/lam_a - 1/lam_b))
    assert N == -189, f"N_Δ = {N}, expected −189"
```

### T5 — Benoit d recovery to < 1 µm on synthetic data

```python
def test_benoit_d_recovery():
    lam_a, lam_b = 640.2248e-9, 638.2991e-9
    d_true  = 20.1e-3
    alpha   = 1.607e-4
    eps_a, eps_b = 0.22, 0.73
    Delta_a = lam_a / (d_true * alpha**2)
    Delta_b = lam_b / (d_true * alpha**2)
    arr = make_synthetic_peaks_array(Delta_a, eps_a, Delta_b, eps_b, n=10,
                                     sigma_r=0.05)
    result = run_tolansky(arr)
    assert abs(result.d_m - d_true) < 1e-6
```

### T6 — α recovered from d and Δₐ to < 0.1%

```python
def test_alpha_recovery():
    lam_a   = 640.2248e-9
    d_true  = 20.1e-3
    alpha_true = 1.607e-4
    Delta_a = lam_a / (d_true * alpha_true**2)
    alpha_rec = np.sqrt(lam_a / (d_true * Delta_a))
    assert abs(alpha_rec - alpha_true) / alpha_true < 1e-3
```

### T7 — All two_sigma_ fields equal exactly 2 × sigma_ (S04)

```python
def test_two_sigma_fields():
    arr    = make_synthetic_peaks_array(1233.0, 0.22, 1228.0, 0.73, n=10)
    result = run_tolansky(arr)
    assert abs(result.two_sigma_Delta_a - 2.0*result.sigma_Delta_a) < 1e-14
    assert abs(result.two_sigma_Delta_b - 2.0*result.sigma_Delta_b) < 1e-14
    assert abs(result.two_sigma_eps_a   - 2.0*result.sigma_eps_a)   < 1e-14
    assert abs(result.two_sigma_eps_b   - 2.0*result.sigma_eps_b)   < 1e-14
    assert abs(result.two_sigma_d_m     - 2.0*result.sigma_d_m)     < 1e-14
    assert abs(result.two_sigma_alpha   - 2.0*result.sigma_alpha)    < 1e-14
```

---

## 9. Expected numerical values (WindCube FlatSat)

From the FlatSat calibration image `1_cal_120sexp_swapped.bin`:

| Quantity | Expected | Equation |
|---------|----------|---------|
| Δₐ | ~1233 px² | Eq. 3.85 |
| Δᵦ | ~1229 px² | Eq. 3.87 |
| Δₐ/Δᵦ | 1.003017 | 3.85/3.87 |
| N_Δ | −189 | Eq. 3.96 |
| d | ~20.00–20.11 mm | Eq. 3.97 |
| εₐ | ~0.22 | Eq. 3.86 |
| εᵦ | ~0.73 | Eq. 3.88 |
| α_mean | ~1.607 × 10⁻⁴ rad/px | from Eq. 3.85 |
| α consistency | < 1000 ppm | cross-check |
| Y_B_obs | ~0.30 | median(amp_b)/median(amp_a) |
| χ²_dof (line a) | ~1–6 | WLS quality |
| χ²_dof (line b) | ~0.4–1 | WLS quality |

> **Note on χ²_dof.**  The FlatSat data shows χ²_dof ≈ 5.3 for line a
> and ≈ 0.4 for line b.  This asymmetry suggests systematic residuals in
> the brighter line (stronger signal, tighter constraint reveals model
> imperfection) rather than a gross error.  Values in this range are
> acceptable for S13a's purpose of seeding M05 priors.

---

## 10. Reference output figure

The plot `1_cal_120sexp_swapped_tolansky_joint_two_line.png` (committed to
`docs/figures/`) illustrates the expected output of this module for the
FlatSat 120 s calibration exposure.

**Top panel — P vs r² Tolansky plot:**
Ring order P (y-axis) vs r²_fit (x-axis, px²) with WLS fit lines for each
family.  The two families are visually separated in intercept (ε offset)
and nearly parallel in slope (slopes ≈ 1/Δ ≈ 8.1 × 10⁻⁴ orders/px²).

**Lower annotation panels:**
- **Yellow box** — N_Δ calculation showing d_prior, wavelengths, result N_Δ = −189
- **Green box** — WLS slopes in orders/px² (= 1/Δ) for each line and their mean;
  also `α_rpx` computed as `sqrt(slope·λ/(2nd))` (**note**: this uses a factor of
  2 in the denominator that differs from the Vaughan Eq. 3.85 convention; see §4
  Step 7 for the authoritative formula used in this module)
- **Blue box** — Benoit d recovery via Vaughan Eq. 3.97
- **Orange box** — intensity ratio Y_B = 638/640 amplitude ratio (~0.30)

The figure is a **reference diagnostic** produced by the standalone analysis
script; it is not generated by the pipeline during normal operation.

---

## 11. File locations

```
soc_sewell/
├── src/fpi/
│   └── tolansky_2line_2026-05-05.py
├── tests/
│   └── test_tolansky_2line_2026-05-05.py
└── docs/specs/
    └── S13a_tolansky_2_line_2026-05-05.md
└── docs/figures/
    └── 1_cal_120sexp_swapped_tolansky_joint_two_line.png
```

---

## 12. Instructions for Claude Code

### Pre-implementation reads

Before writing any code, read in full:

1. `docs/specs/S13a_tolansky_2_line_2026-05-05.md` (this file)
2. `ingest/annular_reduction.py` lines 607–628 — exact column layout of
   `_peak_fits.npy`

Confirm annular_reduction tests pass first:

```bash
cat PIPELINE_STATUS.md
pytest tests/ -v -k "annular"
```

### Task sequence

**Task 1 — `InsufficientRingsError`**

Define a custom exception class for cases where the peak table has fewer
than 4 valid rows, or either family has fewer than 2 rings after the
amplitude split.

**Task 2 — Input loading and family assignment**

Implement `load_and_split_families(path_or_array)` performing the NaN
filter and amplitude-threshold split from §2.  Returns
`(p_a, r2_a, sigma_r2_a, p_b, r2_b, sigma_r2_b, amp_threshold, Y_B_obs, n_nan_dropped)`.

**Task 3 — Single-line WLS helper**

Implement `_wls(p, r2, sigma_r2)` exactly as given in §4 Step 4.  Returns
the dict specified there.

**Task 4 — Benoit d recovery**

Implement `benoit_d(eps_a, sigma_eps_a, eps_b, sigma_eps_b, lam_a_m, lam_b_m, d_prior_m, n_air=1.0)`.

**Task 5 — α recovery**

Implement `recover_alpha(Delta_a, sigma_Delta_a, Delta_b, sigma_Delta_b, d_m, sigma_d_m, lam_a_m, lam_b_m, n_air=1.0)`.
Returns `(alpha_a, alpha_b, alpha_mean, sigma_alpha, alpha_consistency)`.
No focal length is computed or returned.

**Task 6 — Top-level `run_tolansky()`**

```python
def run_tolansky(
    peaks_input:   np.ndarray | str | pathlib.Path,
    lam_a_m:       float = 640.2248e-9,
    lam_b_m:       float = 638.2991e-9,
    d_prior_m:     float = 20.008e-3,
    n_air:         float = 1.0,
) -> TolanskyResult:
```

Calls Tasks 2–5 in sequence.  Sets all `two_sigma_` fields to exactly
`2.0 × sigma_`.

**Task 7 — `print_rectangular_array()`**

Implement the formatted table from §6.

**Task 8 — `to_m05_priors()`**

Implement §7 exactly.

**Task 9 — Tests (7/7 must pass)**

Place `make_synthetic_peaks_array()` as a module-level helper in the test file.

```bash
pytest tests/test_tolansky_2line_2026-05-05.py -v
pytest tests/ -v   # no regressions
```

**Task 10 — Commit**

```bash
# Update PIPELINE_STATUS.md — update S13a status/version/date
git add src/fpi/tolansky_2line_2026-05-05.py
git add tests/test_tolansky_2line_2026-05-05.py
git add PIPELINE_STATUS.md
git commit -m "feat(S13a): v0.4 — remove f, add peak_fits.npy ingestion, amplitude family split, 7/7 tests pass

Also updates PIPELINE_STATUS.md"
```

### Module docstring

```python
"""
Module:      tolansky_2line_2026-05-05.py
Spec:        docs/specs/S13a_tolansky_2_line_2026-05-05.md  v0.4
Reference:   Vaughan (1989) The Fabry-Perot Interferometer, §3.5.2
             Equations (3.83)–(3.97) — rectangular array / Benoit method
             Burns, Adams & Longwell (1950) — Ne IAU standard wavelengths
Author:      Claude Code
Project:     WindCube FPI Pipeline — NCAR/HAO
Repo:        soc_sewell
Input:       {stem}_peak_fits.npy  (9-column float64, from annular_reduction.py)
             Family assignment by amplitude threshold (640nm ~3× brighter).
Note:        Two-line neon calibration lamp analysis only.
             Focal length f is not computed or stored; α is the sole
             plate-scale output.
             For airglow single-line analysis, see S13b / tolansky_1line.py
"""
```

### Report format

```
=== S13a CLAUDE CODE REPORT ===
Date: YYYY-MM-DD
Module: src/fpi/tolansky_2line_2026-05-05.py
Tests: N/7 pass

INPUT
  Total peaks: XX   NaN dropped: X
  amp_threshold = XXXX ADU   Y_B_obs = X.XXX  [PASS/WARN]
  n_rings_a = XX   n_rings_b = XX

TOLANSKY TWO-LINE RESULTS
  Δₐ = XXXXXXX.XX ± X.XX px²    εₐ = X.XXXX ± X.XXXX (2σ = X.XXXX)
  Δᵦ = XXXXXXX.XX ± X.XX px²    εᵦ = X.XXXX ± X.XXXX (2σ = X.XXXX)
  Δₐ/Δᵦ = X.XXXXXX  (expected 1.003017, residual XX ppm)  [PASS/WARN]
  χ²_dof_a = X.XX    χ²_dof_b = X.XX
  N_Δ = -189
  d   = XX.XXXX ± X.XXXX mm   (2σ = X.XXXX mm)
  α_a = X.XXXXE-4   α_b = X.XXXXE-4
  α_mean = X.XXXXE-4 ± X.XXXXE-6 rad/px   (2σ = X.XXXXE-6)
  α consistency = X.X ppm  [PASS/WARN]

DEVIATIONS FROM SPEC:
  [list any, or "None"]
================================
```

Stop and return this report if any task takes more than 15 minutes
without all relevant tests passing.
