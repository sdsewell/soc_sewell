# S13 — Tolansky Two-Line Analysis

**Spec ID:** S13
**Spec file:** `docs/specs/S13_tolansky_analysis_2026-04-13.md`
**Project:** WindCube FPI Pipeline
**Institution:** NCAR / High Altitude Observatory (HAO)
**Status:** v0.2 — rewritten 2026-04-13 to follow Vaughan (1989) §3.5.2
**Depends on:** Z00 (provides `{stem}_fringe_peaks.npy`)
**Used by:**
  - S14 (M05) — receives `TolanskyResult` as informed priors
**References:**
  - Vaughan (1989) *The Fabry-Perot Interferometer*, §3.5.2 "Analysis of
    photographic recordings", pp. 116–121.  **Equations 3.83–3.97 are
    the authoritative derivation for all analysis steps below.**
  - Benoit (1898) — exact-fractions method for plate-spacing recovery
  - GNL4096-R iss1 WindCube Etalon Assembly (ICOS build report, Dec 2023)
**Supersedes:** S13_tolansky_analysis_2026-04-05.md
**Last updated:** 2026-04-13

> **Revision note v0.2:** This rewrite removes all references to dark
> subtraction, annular reduction, and peak finding.  Those responsibilities
> belong to M03 (S12) and are validated by Z00.  This module's sole input
> is the structured `_fringe_peaks.npy` array saved by Z00, and its sole
> purpose is to carry out the Vaughan §3.5.2 two-line Tolansky analysis
> to recover the plate spacing `d`, effective focal length `f`, plate-scale
> `α`, and fractional interference orders `εₐ`, `εᵦ`.

---

## 1. Purpose

Given a set of calibration fringe ring radii for two known neon wavelengths,
recover the etalon physical parameters that M05 (S14) needs as starting
priors:

| Parameter | Symbol | Physical meaning |
|-----------|--------|-----------------|
| Plate spacing | d | Etalon mirror separation (metres) |
| Focal length | f | Imaging lens focal length (pixels) |
| Plate scale | α | Angular pixel scale (rad/px) |
| Fractional orders | εₐ, εᵦ | Fractional interference order at optical axis for each neon line |

The analysis follows the rectangular-array method of Tolansky (1931) as
described in Vaughan (1989) §3.5.2, equations (3.83)–(3.97), with the
addition of a weighted least-squares (WLS) refinement step to obtain
properly propagated 1σ uncertainties on all recovered parameters.

**What this module is not.**  The Tolansky analysis uses only ring radii
(not the full fringe profile shape), so it cannot recover reflectivity R,
PSF width σ, or intensity envelope coefficients.  Its purpose is to anchor
d, f, and ε so M05 starts from physically correct initial values.

---

## 2. Input

```python
peak_table = np.load('{stem}_fringe_peaks.npy')
```

where `{stem}` is the base name of the calibration ROI array used by Z00.
The structured array dtype is defined in Z00 §10.1. For this module,
the relevant fields are:

| Field | Type | Description |
|-------|------|-------------|
| `family` | str | `'neon_A'` (λₐ = 640.2248 nm) or `'neon_B'` (λᵦ = 638.2991 nm) |
| `ring_index` | int | 1-based ring index p within family |
| `r_fit_px` | float | Gaussian centroid radius r_p (pixels) |
| `sigma_r_fit_px` | float | 1σ uncertainty σ(r_p) (pixels) |
| `r2_fit_px2` | float | r²_p = r_fit_px² (pixels²) |
| `two_sigma_r2_px2` | float | 2σ(r²_p) = 2 × 2·r_p·σ(r_p) (pixels²) |

Load the two families separately:

```python
rows_a = peak_table[peak_table['family'] == 'neon_A']   # line a: λₐ
rows_b = peak_table[peak_table['family'] == 'neon_B']   # line b: λᵦ

p_a  = rows_a['ring_index'].astype(float)   # 1, 2, 3, …
r2_a = rows_a['r2_fit_px2']                 # ₐr²_p  (px²)
sr_a = rows_a['sigma_r_fit_px']             # σ(r_p) (px)

p_b  = rows_b['ring_index'].astype(float)
r2_b = rows_b['r2_fit_px2']
sr_b = rows_b['sigma_r_fit_px']
```

---

## 3. Notation and correspondence with Vaughan (1989) §3.5.2

Vaughan works with **ring diameters** D_p (mm) and their squares D²_p.
This module works with **ring radii** r_p (pixels) and their squares r²_p.
Since D_p = 2 r_p:

```
r²_p = D²_p / 4
```

All of Vaughan's equations involving D² or Δ translate to r² and Δ/4.
For clarity, every equation below is written in **our r² pixel notation**,
with the corresponding Vaughan equation number cited.  The factor-of-4
substitution is made once here and silently absorbed throughout.

| Vaughan symbol | Our symbol | Units | Meaning |
|---------------|-----------|-------|---------|
| ₐD²_p | ₐr²_p | px² | Radius-squared of pth ring, line a |
| Δₐ (= ₐD²_{p+1} − ₐD²_p) | Δₐ (= ₐr²_{p+1} − ₐr²_p) | px² | Mean successive r²-difference, line a |
| δₐ_p (individual) | δₐ_p | px² | p-th successive difference = r²_{p+1} − r²_p |
| ₐε_c / 2π | εₐ | — | Fractional interference order at centre, line a |
| nₐ | nₐ | — | Integer interference order at centre, line a |
| f (focal length, mm) | f (focal length, px) | px | f_px = f_m / pixel_pitch_m |
| d | d | m | Plate spacing |
| λₐ | λₐ | m | Wavelength of line a |
| n (refractive index) | n_air = 1.0 | — | Refractive index of etalon gap |

**Ring indexing:** Vaughan numbers rings from p = 0 (innermost).
We number from p = 1.  Where Vaughan writes `− p`, we write `− (p − 1)`.

**WindCube numerical constants:**

| Constant | Value | Source |
|---------|-------|--------|
| λₐ | 640.2248 × 10⁻⁹ m | Burns et al. 1950; IAU standard |
| λᵦ | 638.2991 × 10⁻⁹ m | Burns et al. 1950; IAU standard |
| pixel_pitch | 32 × 10⁻⁶ m | 2×2 binned CCD97 (2 × 16 µm) |
| d_prior | 20.008 × 10⁻³ m | ICOS build report GNL4096-R (spacer measurement) |
| n_air | 1.0 | Air gap |

---

## 4. Analysis — equations (3.83) through (3.97)

### Step 1 — Successive r²-differences (Vaughan Eq. 3.85 / 3.87)

For a Fabry-Pérot fringe pattern at high interference order,
the paraxial approximation gives ring radii that are equally spaced in r²:

```
ₐr²_p = Δₐ · (p − 1 + εₐ)                           [3.84 in r² form]
```

Successive r²-differences are therefore constant:

```
δₐ_p  ≡  ₐr²_{p+1} − ₐr²_p  =  Δₐ  =  f² · λₐ / (n_air · d)   [3.85]
δᵦ_p  ≡  ᵦr²_{p+1} − ᵦr²_p  =  Δᵦ  =  f² · λᵦ / (n_air · d)   [3.87]
```

Compute for each family:

```python
delta_a = np.diff(r2_a)     # δₐ_1, δₐ_2, … (n_a − 1 values)
delta_b = np.diff(r2_b)

Delta_a = np.mean(delta_a)  # Δₐ — best simple estimate (px²)
Delta_b = np.mean(delta_b)  # Δᵦ
```

**Consistency check (Eq. 3.85 / 3.87 ratio):**

```
Δₐ / Δᵦ  =  λₐ / λᵦ  =  640.2248 / 638.2991  =  1.003014 …    [3.85/3.87]
```

Deviation from this ratio in excess of ~0.1% signals a family mis-assignment.

### Step 2 — Rectangular array (Vaughan Table 3.1)

Tolansky's rectangular array lays out r²_p values as columns
(ring number 1, 2, 3, …) and lines as rows (component a, component b).
Between successive columns the individual differences δ_p are written.
This layout makes gross errors in ring measurement immediately apparent
because any δ_p that differs markedly from Δ corresponds to a mis-measured
or mis-assigned ring (Vaughan pp. 118–119).

```
Component a  (λₐ = 640.2248 nm)
  Ring p :       1           2           3       …       nₐ
  ₐr²_p  :  ₐr²_1      ₐr²_2      ₐr²_3  …  ₐr²_nₐ
              δₐ_1₂        δₐ_2₃        δₐ_3₄  …
  Mean Δₐ = …

Component b  (λᵦ = 638.2991 nm)
  Ring p :       1           2           3       …       nᵦ
  ᵦr²_p  :  ᵦr²_1      ᵦr²_2      ᵦr²_3  …  ᵦr²_nᵦ
              δᵦ_1₂        δᵦ_2₃        δᵦ_3₄  …
  Mean Δᵦ = …

Ratio  Δₐ/Δᵦ (observed) = …   expected = λₐ/λᵦ = 1.003014
```

The module renders this array as a formatted printed table (see Section 6).

### Step 3 — Fractional interference orders (Vaughan Eq. 3.86 / 3.88)

The fractional interference order at the optical axis for each line is:

```
εₐ  =  ₐr²_p / Δₐ  −  (p − 1)    for any ring p          [3.86]
εᵦ  =  ᵦr²_p / Δᵦ  −  (p − 1)    for any ring p          [3.88]
```

In practice, compute εₐ from every ring and take the mean (or use WLS,
see Step 4):

```python
eps_a_per_ring = r2_a / Delta_a - (p_a - 1)
eps_b_per_ring = r2_b / Delta_b - (p_b - 1)
eps_a = np.mean(eps_a_per_ring)   # should lie in [0, 1)
eps_b = np.mean(eps_b_per_ring)
```

**Sanity check:** `0 ≤ εₐ < 1` and `0 ≤ εᵦ < 1`.  Values outside this
range indicate an error in ring indexing or family assignment.

### Step 4 — WLS refinement (enhancement over basic Vaughan method)

The mean-differences approach of Steps 1–3 gives equal weight to all rings.
A weighted least-squares (WLS) linear fit to `r²_p = Δ · p + b` provides
best-estimate Δ and ε with propagated uncertainties.

Propagate radial uncertainties to r²:

```
σ(r²_p)  =  2 · r_p · σ(r_p)                             [error propagation]
```

Weight each ring inversely by its variance:

```
w_p  =  1 / σ(r²_p)²
```

WLS normal equations for model `r²_p = S · p + b`:

```
Λ  =  Σw · Σw·p²  −  (Σw·p)²
S  =  (Σw · Σw·p·r² − Σw·p · Σw·r²) / Λ                 [slope = Δ]
b  =  (Σw·p² · Σw·r² − Σw·p · Σw·p·r²) / Λ              [intercept]

Var(S)  =  Σw / Λ
Var(b)  =  Σw·p² / Λ
```

Apply separately to each family (a and b).  Recover the fractional order:

```
εₐ  =  1  +  bₐ / Sₐ                                     [from Eq. 3.86]
σ(εₐ)²  =  (σ_b/Sₐ)²  +  (bₐ · σ_S / Sₐ²)²
```

The WLS slope is the best-estimate `Δₐ = Sₐ` (px²/ring).

**Reduced χ²:**

```
χ²_dof  =  Σ w_p · (r²_p − Sₐ·p − bₐ)²  /  (N_a − 2)
```

Values `χ²_dof ≫ 1` indicate under-estimated σ(r_p) or a mis-assigned ring.

### Step 5 — Identify integer order difference N_Δ (Vaughan Eqs. 3.95–3.96)

The central interference orders for lines a and b are:

```
nₐ + εₐ  =  2·d / λₐ                                      [3.94]
nᵦ + εᵦ  =  2·d / λᵦ                                      [3.95]
```

Their difference is an integer (Benoit exact-fractions):

```
N_Δ  ≡  nₐ − nᵦ  =  round(2 · d_prior · (1/λₐ − 1/λᵦ))  [3.96]
```

`d_prior` is the ICOS spacer measurement (20.008 mm).  Its sole function
here is to resolve the FSR-period integer ambiguity; it does **not** bias
the recovered d.

For WindCube: `N_Δ = round(2 × 20.008×10⁻³ × (1/640.2248×10⁻⁹ − 1/638.2991×10⁻⁹)) ≈ −189`

### Step 6 — Recover plate spacing d (Vaughan Eq. 3.97 / Benoit)

Subtracting the two expressions in Step 5:

```
N_Δ  +  εₐ − εᵦ  =  2d · (1/λₐ − 1/λᵦ)

d  =  (N_Δ + εₐ − εᵦ) · λₐ·λᵦ / (2·n_air·(λᵦ − λₐ))    [3.97 / Benoit]
```

Note: λᵦ < λₐ so λᵦ − λₐ < 0, and N_Δ ≈ −189 < 0, so d > 0 as required.

Propagate uncertainty:

```
σ(εₐ − εᵦ)  =  sqrt(σ_εₐ² + σ_εᵦ² − 2·Cov(εₐ, εᵦ))
```

where `Cov(εₐ, εᵦ) = 0` since the two families come from independent ring
measurements.

```
σ(d)  =  |λₐ·λᵦ / (2·n_air·(λᵦ − λₐ))| · σ(εₐ − εᵦ)
```

### Step 7 — Recover focal length f and plate scale α

From Eq. (3.85) rearranged:

```
f  =  sqrt(Δₐ · n_air · d / λₐ)        [pixels]           [from 3.85]
f_m  =  f · pixel_pitch                 [metres]
```

Uncertainty (treating Δₐ and d as independent):

```
σ(f)/f  =  (1/2) · sqrt( (σ_Δₐ/Δₐ)² + (σ_d/d)² )
```

Plate scale:

```
α  =  pixel_pitch / f_m  =  1 / f_px    [rad/px]
σ(α)  =  α · σ(f)/f
```

### Verification: self-consistency of f from both lines

As a cross-check, compute f also from line b:

```
f_b  =  sqrt(Δᵦ · n_air · d / λᵦ)
```

`|f_a − f_b| / f_a < 0.001` is the acceptance criterion.
Larger discrepancy indicates a bad ring in one family.

---

## 5. Output data class

```python
@dataclass
class TolanskyResult:
    """
    Output of the Tolansky two-line analysis.
    All two_sigma_ fields are exactly 2 × sigma_ (S04 convention).
    """
    # --- Single-line WLS fit results ---
    # Line a  (λₐ = 640.2248 nm)
    Delta_a:         float   # Δₐ = mean r²-step, px²          [Eq. 3.85]
    sigma_Delta_a:   float   # 1σ uncertainty on Δₐ,  px²
    eps_a:           float   # εₐ = fractional order at centre  [Eq. 3.86]
    sigma_eps_a:     float   # 1σ
    chi2_dof_a:      float   # reduced χ² for line-a WLS fit
    delta_a:         np.ndarray  # δₐ_p successive differences  (px²)

    # Line b  (λᵦ = 638.2991 nm)
    Delta_b:         float   # Δᵦ,  px²                         [Eq. 3.87]
    sigma_Delta_b:   float
    eps_b:           float   # εᵦ                               [Eq. 3.88]
    sigma_eps_b:     float
    chi2_dof_b:      float
    delta_b:         np.ndarray

    # --- Consistency check ---
    Delta_ratio_obs:      float  # Δₐ/Δᵦ (observed)
    Delta_ratio_expected: float  # λₐ/λᵦ = 1.003014…
    Delta_ratio_residual: float  # |obs − expected| / expected

    # --- Integer disambiguation ---
    N_Delta:  int    # N_Δ = nₐ − nᵦ  [Eq. 3.96 / Benoit]

    # --- Plate spacing recovery  [Eq. 3.97] ---
    d_m:             float  # recovered d  (metres)
    sigma_d_m:       float  # 1σ
    two_sigma_d_m:   float  # exactly 2 × sigma_d_m   (S04)

    # --- Focal length and plate scale ---
    f_px:            float  # f  (pixels)
    sigma_f_px:      float
    two_sigma_f_px:  float  # exactly 2 × sigma_f_px  (S04)
    f_b_px:          float  # cross-check from line b
    f_consistency:   float  # |f_a − f_b| / f_a  (accept if < 0.001)

    alpha_rad_px:    float  # α  (rad/px)
    sigma_alpha:     float
    two_sigma_alpha: float  # exactly 2 × sigma_alpha  (S04)

    # --- Inputs for M05 priors ---
    lam_a_nm:  float  # 640.2248
    lam_b_nm:  float  # 638.2991
    n_rings_a: int    # number of rings used
    n_rings_b: int
```

---

## 6. Rectangular array table (Vaughan Table 3.1 analog)

The `print_rectangular_array(result)` function prints the following layout
to stdout, matching Vaughan's Table 3.1 structure exactly:

```
=== TOLANSKY RECTANGULAR ARRAY (Vaughan 1989, Table 3.1 analog) ===

Component a  (λₐ = 640.2248 nm)
  p  :      1          2          3      ...     10
  r²  :  XXXXX.XX   XXXXX.XX   XXXXX.XX  ...  XXXXX.XX   (px²)
             δ₁₂=XX.X   δ₂₃=XX.X  ...
  Δₐ (mean δ) = XXXX.XX px²    σ = X.XX px²
  εₐ          = X.XXXX           σ = X.XXXXX

Component b  (λᵦ = 638.2991 nm)
  p  :      1          2          3      ...     10
  r²  :  XXXXX.XX   XXXXX.XX   XXXXX.XX  ...  XXXXX.XX   (px²)
             δ₁₂=XX.X   δ₂₃=XX.X  ...
  Δᵦ (mean δ) = XXXX.XX px²    σ = X.XX px²
  εᵦ          = X.XXXX           σ = X.XXXXX

Ratio  Δₐ/Δᵦ observed = X.XXXXXX   expected (λₐ/λᵦ) = X.XXXXXX   residual = X.X ppm

=== BENOIT RECOVERY (Vaughan Eqs. 3.94–3.97) ===
  N_Δ = nₐ − nᵦ = XXX   [from d_prior = 20.008 mm, Eq. 3.96]
  d   = XX.XXX ± X.XXX mm  (2σ = X.XXX mm)
  f   = XXXXX.X ± X.X px  (= XXX.X ± X.X mm)  (2σ = X.X px)
  f_b = XXXXX.X px  (cross-check)   |f_a − f_b|/f_a = X.X ppm
  α   = X.XXXXE-4 ± X.XXXXE-6 rad/px  (2σ = X.XXXXE-6)
```

---

## 7. M05 priors handoff

```python
def to_m05_priors(result: TolanskyResult) -> dict:
    """
    Convert TolanskyResult to the prior dict expected by M05 FitConfig.
    Direct mapping to S14 fields.
    """
    d_mm = result.d_m * 1e3
    sig_d_mm = result.sigma_d_m * 1e3
    return {
        't_init_mm':      d_mm,
        't_bounds_mm':    (d_mm - 3 * sig_d_mm, d_mm + 3 * sig_d_mm),
        'alpha_init':     result.alpha_rad_px,
        'alpha_bounds':   (result.alpha_rad_px * 0.875,
                           result.alpha_rad_px * 1.125),  # ±12.5%
        'epsilon_cal_1':  result.eps_a,    # εₐ for λₐ = 640.2248 nm
        'epsilon_cal_2':  result.eps_b,    # εᵦ for λᵦ = 638.2991 nm
    }
```

---

## 8. Verification tests

All 7 tests in `tests/test_tolansky_2026-04-13.py`.

### T1 — Successive differences are uniform on exact synthetic data

```python
def test_successive_differences_uniform():
    """
    For rings on the exact Tolansky r² = Δ·(p−1+ε) curve with no noise,
    all δ_p must equal Δ exactly (CV < 1e-10).
    """
    Delta_true = 485.0   # px²
    eps_true   = 0.37
    p          = np.arange(1, 11, dtype=float)
    r2         = Delta_true * (p - 1 + eps_true)
    delta      = np.diff(r2)
    cv         = delta.std() / delta.mean()
    assert cv < 1e-10, f"CV(δ) = {cv:.2e} for exact data; expected < 1e-10"
```

### T2 — WLS recovers known Δ and ε to high accuracy

```python
def test_wls_known_answer():
    """
    WLS fit on exact r² = Δ·p + b must recover Δ and ε to < 0.01%.
    """
    Delta_true = 485.0
    eps_true   = 0.37
    p   = np.arange(1, 11, dtype=float)
    r2  = Delta_true * (p - 1 + eps_true)
    sr  = np.full_like(r2, 0.3 * 2 * np.sqrt(r2[0]))  # σ(r²) = 2r·σ_r
    result = run_single_line_wls(p, r2, sr)
    assert abs(result['Delta'] - Delta_true) / Delta_true < 1e-4
    assert abs(result['eps']   - eps_true)                < 1e-4
    assert result['r2_fit'] > 0.9999
```

### T3 — Δ ratio constraint: Δₐ/Δᵦ = λₐ/λᵦ from same (d, f)

```python
def test_delta_ratio_matches_wavelength_ratio():
    """
    Rings from the same d and f must give Δₐ/Δᵦ = λₐ/λᵦ to < 10 ppm.
    """
    lam_a, lam_b = 640.2248e-9, 638.2991e-9
    d, f_px      = 20.106e-3, 6222.0
    p            = np.arange(1, 11, dtype=float)
    Delta_a_true = f_px**2 * lam_a / d
    Delta_b_true = f_px**2 * lam_b / d
    for lam, Delta_true in [(lam_a, Delta_a_true), (lam_b, Delta_b_true)]:
        r2 = Delta_true * (p - 1 + 0.4)
        # (check ratio)
    ratio_obs = Delta_a_true / Delta_b_true
    ratio_exp = lam_a / lam_b
    assert abs(ratio_obs - ratio_exp) / ratio_exp < 1e-8
```

### T4 — N_Δ correctly identified from d_prior

```python
def test_N_Delta_from_prior():
    """
    N_Δ = round(2 · d_prior · (1/λₐ − 1/λᵦ)) must equal −189 for
    d_prior = 20.008 mm (ICOS measurement).
    """
    lam_a, lam_b = 640.2248e-9, 638.2991e-9
    d_prior      = 20.008e-3
    N_Delta      = round(2 * d_prior * (1/lam_a - 1/lam_b))
    assert N_Delta == -189, f"N_Δ = {N_Delta}, expected −189"
```

### T5 — Benoit d recovery to < 1 µm on synthetic data

```python
def test_benoit_d_recovery():
    """
    Synthetic rings from d_true = 20.106 mm.
    Recovered d must match d_true to < 1 µm.
    """
    lam_a, lam_b = 640.2248e-9, 638.2991e-9
    d_true, f_px = 20.106e-3, 6222.0
    p = np.arange(1, 11, dtype=float)
    eps_a, eps_b = 0.37, 0.51
    Delta_a = f_px**2 * lam_a / d_true
    Delta_b = f_px**2 * lam_b / d_true
    r2_a = Delta_a * (p - 1 + eps_a)
    r2_b = Delta_b * (p - 1 + eps_b)
    sr_a = np.full_like(r2_a, 0.05)  # tight uncertainties
    sr_b = np.full_like(r2_b, 0.05)
    result = run_tolansky(
        p, r2_a, sr_a, p, r2_b, sr_b,
        lam_a_m=lam_a, lam_b_m=lam_b,
        d_prior_m=20.008e-3, pixel_pitch_m=32e-6
    )
    assert abs(result.d_m - d_true) < 1e-6, \
        f"|d_recovered − d_true| = {abs(result.d_m - d_true)*1e6:.3f} µm > 1 µm"
```

### T6 — f recovered from d via Δₐ · n_air · d / λₐ

```python
def test_f_recovery():
    """From exact Δₐ and known d, f must be recovered to < 0.1%."""
    lam_a = 640.2248e-9
    d_m   = 20.106e-3
    f_true_px = 6222.0
    Delta_a_true = f_true_px**2 * lam_a / d_m
    f_recovered  = np.sqrt(Delta_a_true * d_m / lam_a)
    assert abs(f_recovered - f_true_px) / f_true_px < 1e-3
```

### T7 — All two_sigma_ fields equal exactly 2 × sigma_ (S04)

```python
def test_two_sigma_fields():
    """S04 convention: every two_sigma_ field = exactly 2 × sigma_."""
    # (run run_tolansky on synthetic data from T5)
    assert abs(result.two_sigma_d_m   - 2.0 * result.sigma_d_m)   < 1e-15
    assert abs(result.two_sigma_f_px  - 2.0 * result.sigma_f_px)  < 1e-15
    assert abs(result.two_sigma_alpha - 2.0 * result.sigma_alpha)  < 1e-15
```

---

## 9. Expected numerical values (WindCube FlatSat)

For d ≈ 20.106 mm, f ≈ 199.12 mm, α ≈ 1.6071 × 10⁻⁴ rad/px:

| Quantity | Expected | Equation |
|---------|----------|---------|
| Δₐ | ~485 px² | 3.85 |
| Δᵦ | ~484 px² | 3.87 |
| Δₐ/Δᵦ | 1.003014 | 3.85/3.87 |
| N_Δ | −189 | 3.96 |
| d | 20.106 ± ~0.005 mm | 3.97 |
| f | ~6222 px (~199.1 mm) | from Δₐ |
| α | ~1.607 × 10⁻⁴ rad/px | pixel_pitch/f_m |
| χ²_dof (line a) | ~1.0 | WLS quality |
| f consistency | < 100 ppm | cross-check |

---

## 10. File locations

```
soc_sewell/
├── src/fpi/
│   └── tolansky_2026-04-13.py
├── tests/
│   └── test_tolansky_2026-04-13.py
└── docs/specs/
    └── S13_tolansky_analysis_2026-04-13.md
```

---

## 11. Instructions for Claude Code

### Pre-implementation reads

Before writing any code, read in full:

1. `docs/specs/S13_tolansky_analysis_2026-04-13.md` (this file)
2. `docs/specs/Z00_validate_annular_reduction_peak_finding_2026-04-13.md`
   §10 (fringe_peaks.npy dtype definition)

Confirm M03 and Z00 tests pass first:

```bash
pytest tests/ -v --ignore=tests/test_tolansky*.py
```

### Task sequence

**Task 1 — Helper: single-line WLS**

Implement `run_single_line_wls(p, r2, sr)` returning a dict with keys:
`Delta`, `sigma_Delta`, `eps`, `sigma_eps`, `chi2_dof`, `r2_fit`,
`delta` (array of successive differences), `intercept`, `sigma_intercept`.

Use the closed-form WLS normal equations from Section 4 Step 4 exactly.
σ(r²_p) = 2·r_p·σ(r_p); weights w_p = 1/σ(r²_p)².

**Task 2 — Benoit d recovery**

Implement `benoit_d(eps_a, sigma_eps_a, eps_b, sigma_eps_b, lam_a_m, lam_b_m, d_prior_m, n_air=1.0)` returning `(N_Delta, d_m, sigma_d_m)`.
Follow Section 4 Steps 5–6 exactly.

**Task 3 — f and α recovery**

Implement `recover_f_alpha(Delta_a, sigma_Delta_a, d_m, sigma_d_m, lam_a_m, pixel_pitch_m)` returning `(f_px, sigma_f_px, alpha, sigma_alpha)`.
Follow Section 4 Step 7.

**Task 4 — Top-level `run_tolansky()`**

```python
def run_tolansky(
    p_a, r2_a, sr_a,          # ring indices, r²-values (px²), σ(r) (px)
    p_b, r2_b, sr_b,
    lam_a_m  = 640.2248e-9,
    lam_b_m  = 638.2991e-9,
    d_prior_m = 20.008e-3,
    pixel_pitch_m = 32e-6,
    n_air    = 1.0,
) -> TolanskyResult:
```

Calls Tasks 1–3 in order, assembles TolanskyResult, sets all `two_sigma_`
fields to exactly `2.0 × sigma_` (never rounded, never truncated), computes
f_b cross-check.

**Task 5 — `print_rectangular_array()`**

Implement the formatted table from Section 6.  The δ values should appear
between their respective r² columns.  Align columns with f-strings to
4 decimal places.

**Task 6 — `to_m05_priors()`**

Implement Section 7 exactly.

**Task 7 — Tests**

Implement all 7 tests from Section 8 in `tests/test_tolansky_2026-04-13.py`.
Run: `pytest tests/test_tolansky_2026-04-13.py -v`  — all 7 must pass.

**Task 8 — Full suite**

```bash
pytest tests/ -v
```

No regressions permitted.

**Task 9 — Commit**

```bash
git add src/fpi/tolansky_2026-04-13.py
git add tests/test_tolansky_2026-04-13.py
git commit -m "feat(tolansky): implement Vaughan §3.5.2 two-line analysis, 7/7 tests pass
Implements: S13_tolansky_analysis_2026-04-13.md"
```

### Module docstring

```python
"""
Module:      tolansky_2026-04-13.py
Spec:        docs/specs/S13_tolansky_analysis_2026-04-13.md
Reference:   Vaughan (1989) The Fabry-Perot Interferometer, §3.5.2
             Equations (3.83)–(3.97) — rectangular array method
Author:      Claude Code
Project:     WindCube FPI Pipeline — NCAR/HAO
Repo:        soc_sewell
"""
```

### Report format

```
=== S13 CLAUDE CODE REPORT ===
Date: YYYY-MM-DD
Module: src/fpi/tolansky_2026-04-13.py
Tests: N/7 pass

TOLANSKY RESULTS:
  Δₐ = XXXX.XX ± X.XX px²   Δᵦ = XXXX.XX ± X.XX px²
  Δₐ/Δᵦ = X.XXXXXX  (expected X.XXXXXX,  residual XX ppm)
  N_Δ = XXX
  εₐ = X.XXXX ± X.XXXX     εᵦ = X.XXXX ± X.XXXX
  d  = XX.XXX ± X.XXX mm   (2σ = X.XXX mm)
  f  = XXXXX.X ± X.X px    (= XXX.X ± X.X mm)
  α  = X.XXXXE-4 ± X.XXXXE-6 rad/px
  f consistency = X.X ppm

DEVIATIONS FROM SPEC:
  [list any, or "None"]
==============================
```

Stop and return this report if any task takes more than 15 minutes without
all relevant tests passing.
