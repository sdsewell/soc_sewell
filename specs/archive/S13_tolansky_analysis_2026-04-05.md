# S13 — Tolansky Fringe Analysis Specification
## Single-Line and Two-Line Joint Analysis for FPI Instrument Characterisation

**Spec ID:** S13
**Spec file:** `docs/specs/S13_tolansky_analysis_2026-04-05.md`
**Project:** WindCube FPI Pipeline
**Institution:** NCAR / High Altitude Observatory (HAO)
**Status:** Specification — ready for implementation in VS Code
**Depends on:** S01, S02, S03, S04, S12 (M03 — provides `FringeProfile.peak_fits`)
**Used by:**
  - S14 (M05) — receives `TwoLineResult` as informed priors for Stage 0/1
  - S17 (INT02) — validates Tolansky priors against M05 inversion result
**References:**
  - Tolansky (1948) *Multiple-Beam Interferometry* — fringe order analysis
  - Benoit (1898) / Michelson & Benoit (1895) — excess fractions method
  - Harding et al. (2014) Applied Optics 53(4), Section 2
  - GNL4096-R iss1 WindCube Etalon Assembly (ICOS build report, Dec 2023)
**Last updated:** 2026-04-05

---

## 1. Purpose

The Tolansky module characterises the WindCube FPI etalon from the peak
radii measured by M03. It recovers the physical instrument parameters `d`
(etalon plate spacing), `f` (effective focal length), `α` (magnification
constant), and `ε` (fractional interference orders) that M05 needs as
informed priors for its staged Airy inversion.

This module sits between M03 (which produces peak radii) and M05 (which
inverts the full Airy function). Without Tolansky priors, M05 Stage 0
must guess `t_init_mm` from a beat-period estimate that is vulnerable to
the FSR-period ambiguity — the same ambiguity that produced the erroneous
FlatSat value of d = 20.670 mm (2068 FSR periods from the true gap).
The Tolansky two-line analysis resolves this ambiguity completely.

**What this module is not.** Tolansky is not an alternative to M05. It
uses ring radii only (not the full fringe shape), so it cannot recover
reflectivity R, PSF width σ, or intensity envelope coefficients. Its sole
purpose is to anchor d, f, and ε so that M05 starts from physically correct
initial values and converges to the right solution on the first attempt.

---

## 2. Physical background

### 2.1 The Tolansky r² relation

Constructive interference (Haidinger fringes) requires:

```
m λ = 2 n d cos θ
```

Writing `m = m₀ − (p − 1 + ε)` for fringe index p and fractional order ε,
and using the paraxial substitution `cos θ ≈ 1 − r²/(2f²)`:

```
r_p² = (f² λ / (n d)) · (p − 1 + ε)
```

So r² is **linear in fringe index p**:

```
Slope     S = f² λ / (n d)      [px²/fringe]
Intercept b = S · (ε − 1)
ε           = 1 + b/S            (fractional order at centre)
```

This is the single-line Tolansky analysis. Given λ and a known f or d,
the remaining unknown is recovered from S.

### 2.2 The f-d degeneracy and why one line is not enough

A single-line fit gives only the ratio `f²/d`. With λ known, we can compute
either `d` (if f is known) or `f` (if d is known), but not both independently.
The inexpensive COTS lens in WindCube has a poorly characterised focal length;
the ICOS mechanical spacer measurement of d is far more reliable. So in
principle d should anchor the analysis. But the FSR-period ambiguity means
the ICOS d value can only be trusted to within ±1 FSR before the analysis
is run — which means f is still underdetermined.

### 2.3 The two-line joint analysis — breaking the degeneracy

With two known wavelengths λ₁ and λ₂ through the **same** etalon:

**Slope ratio constraint** (enforced in joint fit):
```
S₁ / S₂ = λ₁ / λ₂
```
This ties both ring families to a single f and d, using all N₁+N₂ rings
to determine one shared slope S₁ and two independent fractional orders ε₁, ε₂.

**Excess fractions method** (Benoit 1898):
The central interference orders are `m₀,₁ = 2nd/λ₁` and `m₀,₂ = 2nd/λ₂`.
Their difference is:

```
m₀,₁ − m₀,₂ = 2nd(1/λ₁ − 1/λ₂) = N_int + (ε₁ − ε₂)
```

Rearranging for d:

```
d = (N_int + ε₁ − ε₂) · λ₁λ₂ / [2n(λ₂ − λ₁)]
```

N_int is identified by rounding `2 · d_prior · (1/λ₁ − 1/λ₂)` using the
ICOS mechanical measurement as the prior. This is the **only** role of the
prior — resolving the FSR-period integer ambiguity. Once N_int is fixed, d
is determined from ε₁ and ε₂ alone, with no dependence on f.

**Recovering f:**
```
f = sqrt(S₁ · n · d / λ₁)
```

**Recovering α:**
```
α = pixel_pitch / f    [rad/px]
```

For WindCube 2×2 binned CCD (16 µm × 2 = 32 µm pitch):
```
α = 32e-6 m / f_m    [rad/px]
```

### 2.4 Information flow

```
Single-line fit  →  S = f²λ/(nd)       (f and d entangled)
Joint fit        →  S₁, ε₁, ε₂         (shared instrument, shared d/f)
Excess fractions →  d                  (f-independent, prior resolves N_int)
d + S₁          →  f, α               (for M05 FitConfig.alpha_init)
```

### 2.5 Critical numerical constants (from S03)

| Constant | Value | Source |
|----------|-------|--------|
| `NE_WAVELENGTH_1_M` | 640.2248e-9 m | S03/M01 |
| `NE_WAVELENGTH_2_M` | 638.2991e-9 m | S03/M01 |
| `d_prior_m` | 20.008e-3 m | ICOS build report Dec 2023 |
| CCD pixel pitch (2×2) | 32e-6 m | 2 × 16 µm CCD97 native |

---

## 3. Module structure

Three classes, each building on the previous:

| Class | Purpose | Input | Output |
|-------|---------|-------|--------|
| `TolanskyAnalyser` | Single-line WLS fit | p, r, σ_r | slope S, ε, recovered d or λ |
| `TwoLineAnalyser` | Joint fit + excess fractions | two `TolanskyAnalyser` | d, f, α, ε₁, ε₂ |
| `TolanskyPipeline` | Full pipeline from `FringeProfile` | `FringeProfile` | `TwoLineResult` |

`TolanskyPipeline` is the function M05 calls. It handles peak splitting,
index assignment, unit conversion, and orchestrates the two analysers.

---

## 4. Data classes

### 4.1 `TolanskyResult` (single-line output)

```python
@dataclass
class TolanskyResult:
    p:            np.ndarray   # fringe indices
    r:            np.ndarray   # ring radii, pixels
    sigma_r:      np.ndarray   # 1σ uncertainties, pixels
    r_sq:         np.ndarray   # r²
    sigma_r_sq:   np.ndarray   # σ(r²) = 2r·σ_r

    slope:        float        # S = f²λ/(nd), px²/fringe
    sigma_slope:  float
    intercept:    float        # b = S(ε−1)
    sigma_int:    float
    r2_fit:       float        # coefficient of determination R²

    epsilon:       float       # fractional order at centre (0 ≤ ε < 1)
    sigma_epsilon: float

    delta_r_sq:    np.ndarray  # successive Δ(r²)
    sigma_delta:   np.ndarray

    recovered_d_m:   float | None   # plate spacing, metres
    sigma_d_m:       float | None
    recovered_f_px:  float | None   # focal length, pixels
    sigma_f_px:      float | None
```

### 4.2 `TwoLineResult` (joint two-line output)

```python
@dataclass
class TwoLineResult:
    # Joint fit
    S1:          float   # slope for λ₁, px²/fringe
    sigma_S1:    float
    S2:          float   # = S1 × λ₂/λ₁ (not free)
    eps1:        float   # ε₁ fractional order (λ₁ family)
    sigma_eps1:  float
    eps2:        float   # ε₂ fractional order (λ₂ family)
    sigma_eps2:  float
    cov_eps:     float   # covariance(ε₁, ε₂) from joint fit
    chi2_dof:    float   # reduced χ²
    lam_ratio:   float   # λ₂/λ₁

    # Excess fractions d recovery
    N_int:            int    # integer order difference used
    delta_eps:        float  # ε₁ − ε₂
    sigma_delta_eps:  float
    d_m:              float  # recovered plate spacing, metres  (S04)
    sigma_d_m:        float
    two_sigma_d_m:    float  # exactly 2 × sigma_d_m            (S04)

    # f and α recovery
    f_px:         float  # recovered focal length, pixels
    sigma_f_px:   float
    two_sigma_f_px: float
    alpha_rad_px: float  # magnification constant, rad/px
    sigma_alpha:  float
    two_sigma_alpha: float

    # ε for M05 handoff
    epsilon_cal_1: float  # ε₁ — fractional order at λ₁ (640.2 nm)
    epsilon_cal_2: float  # ε₂ — fractional order at λ₂ (638.3 nm)

    # Residuals for plotting
    p1:     np.ndarray
    r1_sq:  np.ndarray
    sr1_sq: np.ndarray
    pred1:  np.ndarray
    p2:     np.ndarray
    r2_sq:  np.ndarray
    sr2_sq: np.ndarray
    pred2:  np.ndarray

    lam1_nm: float
    lam2_nm: float
```

---

## 5. Class: `TolanskyAnalyser`

```python
class TolanskyAnalyser:
    """
    Single-line Tolansky r² analysis on measured FPI fringe ring radii.

    Parameters
    ----------
    p        : fringe indices (1 = innermost ring, 2, 3, …)
    r        : ring radii in pixels
    sigma_r  : 1σ uncertainty on each radius, pixels
    lam_nm   : wavelength in nm (set to None if recovering wavelength)
    n        : refractive index of etalon gap (1.0 for air)
    f_px     : effective focal length, pixels (None if unknown)
    d_m      : plate separation, metres (None if unknown)
    pixel_pitch_m : CCD pixel pitch in metres (32e-6 for 2×2 binned)

    Exactly one of (lam_nm × d_m) or (lam_nm × f_px) must be known;
    set the other to None.
    """

    def __init__(self, p, r, sigma_r, lam_nm,
                 n=1.0, f_px=None, d_m=None,
                 pixel_pitch_m=32e-6): ...

    def run(self) -> TolanskyResult:
        """
        1. r² = r², σ(r²) = 2r·σ_r
        2. Weighted least-squares r² = S·p + b:
               weights w_p = 1 / σ(r²_p)²
               Δ = Σw·Σwp² − (Σwp)²
               S = (Σw·Σwpr² − Σwp·Σwr²) / Δ
               b = (Σwp²·Σwr² − Σwp·Σwpr²) / Δ
               Var(S) = Σw/Δ,  Var(b) = Σwp²/Δ
        3. ε = (1 + b/S) mod 1
               σ_ε² = (σ_b/S)² + (b·σ_S/S²)²
        4. Δ(r²), σ(Δr²) successive differences
        5. Recover d or f from S (whichever is unknown)
        """
```

---

## 6. Class: `TwoLineAnalyser`

```python
class TwoLineAnalyser:
    """
    Joint two-line Tolansky analysis. Recovers d and f independently.

    Parameters
    ----------
    analyser1  : TolanskyAnalyser for λ₁ (longer wavelength, 640.2 nm)
    analyser2  : TolanskyAnalyser for λ₂ (shorter wavelength, 638.3 nm)
    lam1_nm    : wavelength 1, nm
    lam2_nm    : wavelength 2, nm
    d_prior_m  : rough prior on plate spacing, metres.
                 Used ONLY to identify N_int — does not bias recovered d.
                 Use ICOS measurement: 20.008e-3 m
    n          : refractive index of etalon gap
    pixel_pitch_m : CCD pixel pitch, metres
    """

    def run(self) -> TwoLineResult:
        """
        Step 1 — Joint weighted fit:
            Free parameters: (S₁, ε₁, ε₂)  [3 parameters]
            Constraint: S₂ = S₁ · λ₂/λ₁
            Data: N₁ + N₂ ring radii
            Residuals: [(r²₁ − S₁(p₁−1+ε₁)) / σ(r²₁),
                        (r²₂ − S₁·λ_ratio·(p₂−1+ε₂)) / σ(r²₂)]
            Method: scipy.optimize.least_squares(method='lm')
            Options: ftol=xtol=gtol=1e-15, max_nfev=100_000
            Covariance inflated by reduced χ².

        Step 2 — Excess fractions → d:
            lever = λ₁·λ₂ / (2n·(λ₂−λ₁))     [metres]
            N_int = round(d_prior / |lever|)
            d = |lever · (N_int + ε₁ − ε₂)|
            σ(d) = |lever| · σ(ε₁−ε₂)
            where σ(ε₁−ε₂) = sqrt(σ_ε₁² + σ_ε₂² − 2·cov(ε₁,ε₂))

        Step 3 — Recover f and α:
            f = sqrt(S₁ · n · d / λ₁)
            α = pixel_pitch / f    [rad/px]
            σ(f) via Gaussian propagation (S₁ and d treated as independent)
            σ(α) = α · σ(f) / f
        """
```

---

## 7. Class: `TolanskyPipeline`

This is the top-level function called by S14 (M05) and S17 (INT02).

```python
class TolanskyPipeline:
    """
    Full Tolansky pipeline from a FringeProfile to a TwoLineResult.

    Handles:
    - Peak splitting by amplitude threshold (λ₁ vs λ₂ families)
    - Fringe index assignment (p = 1, 2, … for each family)
    - Unit conversion (pixels throughout)
    - Orchestration of TolanskyAnalyser × 2 and TwoLineAnalyser

    Parameters
    ----------
    profile          : FringeProfile from M03 (must have peak_fits populated)
    d_prior_m        : plate spacing prior, metres. Default: 20.008e-3
    lam1_nm          : primary neon wavelength, nm. Default: 640.2248
    lam2_nm          : secondary neon wavelength, nm. Default: 638.2991
    amplitude_split_fraction : peaks below this fraction of the maximum
                      amplitude are assigned to the λ₂ (weaker) family.
                      Default 0.7 — works for NE_INTENSITY_2 = 0.8 with
                      typical SNR. Adjust if the two families have unusual
                      relative brightnesses.
    n                : refractive index. Default 1.0 (air gap).
    pixel_pitch_m    : CCD pixel pitch. Default 32e-6 (2×2 binned).
    sigma_r_default  : fallback σ_r when PeakFit.sigma_r_fit_px is nan.
                      Default 0.5 px.
    """

    def run(self) -> TwoLineResult:
        """
        1. Extract peaks with fit_ok=True from FringeProfile.peaks_ok.
        2. Split into two families by amplitude:
               family 1 (λ₁): amplitude >= amplitude_split_fraction × max_amp
               family 2 (λ₂): amplitude <  amplitude_split_fraction × max_amp
        3. Assign fringe indices: p = 1, 2, … sorted by r_fit_px ascending.
        4. Build TolanskyAnalyser for each family with lam_nm known, d=None.
           Run both analysers.
        5. Run TwoLineAnalyser with d_prior_m = 20.008e-3 m.
        6. Return TwoLineResult.

        Raises
        ------
        ValueError : if fewer than 3 peaks in either family after splitting.
        """

    def to_m05_priors(self) -> dict:
        """
        Convert TwoLineResult to the prior dict expected by M05 FitConfig.

        Returns
        -------
        dict with keys:
            't_init_mm'     : float — recovered d in mm  (= d_m * 1000)
            't_bounds_mm'   : (float, float) — (d_m*1000 - 0.020, d_m*1000 + 0.020)
            'alpha_init'    : float — recovered α, rad/px
            'alpha_bounds'  : (float, float) — α ± 12.5%
            'epsilon_cal_1' : float — ε₁ for the λ₁ line
            'epsilon_cal_2' : float — ε₂ for the λ₂ line
        """
```

---

## 8. Verification tests

All 8 tests in `tests/test_tolansky_2026-04-05.py`.

### T1 — Single-line WLS fit: synthetic data with known answer

```python
def test_single_line_wls_known_answer():
    """
    Synthetic rings from exact r² = S*p + b.
    Recovered slope must match input to < 0.01%.
    Recovered ε must match to < 1e-4.
    """
    S_true = 500.0   # px²/fringe
    eps_true = 0.3
    p = np.arange(1, 11, dtype=float)
    b = S_true * (eps_true - 1.0)
    r_sq = S_true * p + b
    r    = np.sqrt(r_sq)
    sigma_r = np.full_like(r, 0.3)

    a = TolanskyAnalyser(p, r, sigma_r, lam_nm=640.2248,
                          d_m=20.008e-3, pixel_pitch_m=32e-6)
    res = a.run()
    assert abs(res.slope - S_true) / S_true < 1e-4, \
        f"Slope error {abs(res.slope - S_true)/S_true:.2e}"
    assert abs(res.epsilon - eps_true) < 1e-4, \
        f"ε error {abs(res.epsilon - eps_true):.2e}"
    assert res.r2_fit > 0.9999
```

### T2 — Weighted fit upweights low-uncertainty rings

```python
def test_weighted_fit_uses_uncertainties():
    """
    Rings with smaller σ_r must have more influence on the fit.
    Add an outlier with large uncertainty — fit should be unaffected.
    """
    S_true = 500.0
    p = np.arange(1, 8, dtype=float)
    b = S_true * (-0.7)
    r = np.sqrt(S_true * p + b)
    sigma_r = np.full_like(r, 0.3)

    # Corrupt last ring, but give it large uncertainty
    r_corrupt = r.copy()
    r_corrupt[-1] *= 1.5
    sigma_r_corrupt = sigma_r.copy()
    sigma_r_corrupt[-1] = 50.0  # huge uncertainty → nearly zero weight

    a_clean   = TolanskyAnalyser(p, r, sigma_r, lam_nm=640.2248,
                                  d_m=20.008e-3, pixel_pitch_m=32e-6)
    a_corrupt = TolanskyAnalyser(p, r_corrupt, sigma_r_corrupt,
                                  lam_nm=640.2248, d_m=20.008e-3,
                                  pixel_pitch_m=32e-6)
    res_c = a_clean.run()
    res_x = a_corrupt.run()
    assert abs(res_x.slope - res_c.slope) / res_c.slope < 0.01, \
        "High-sigma outlier should not shift the slope by > 1%"
```

### T3 — Successive Δ(r²) coefficient of variation < 5%

```python
def test_delta_r2_uniformity():
    """For exact Tolansky data, Δ(r²) must be perfectly uniform (CV ≈ 0)."""
    S_true = 450.0
    p = np.arange(1, 12, dtype=float)
    r = np.sqrt(S_true * p + S_true * (-0.6))
    sigma_r = np.full_like(r, 0.2)
    a = TolanskyAnalyser(p, r, sigma_r, lam_nm=640.2248,
                          d_m=20.008e-3, pixel_pitch_m=32e-6)
    res = a.run()
    cv = res.delta_r_sq.std() / abs(res.delta_r_sq.mean()) * 100
    assert cv < 2.0, f"CV(Δr²) = {cv:.2f}% for exact data; expected < 2%"
```

### T4 — Two-line joint fit: slope ratio constraint enforced

```python
def test_two_line_slope_ratio():
    """
    In the joint fit, S₂ must equal S₁ × λ₂/λ₁ to < 1e-10.
    This is the key constraint that makes the two-line analysis work.
    """
    # Build synthetic rings for both lines from the same d and f
    lam1, lam2 = 640.2248e-9, 638.2991e-9
    d, f = 20.008e-3, 0.20  # metres
    n = 1.0
    px = 32e-6  # pixel pitch

    def make_rings(lam, n_rings=10):
        S = f**2 * lam / (n * d)
        eps = 0.4
        p = np.arange(1, n_rings + 1, dtype=float)
        r_sq = S * p + S * (eps - 1.0)
        r = np.sqrt(r_sq) / px   # convert to pixels
        return p, r, np.full_like(r, 0.3 / px)

    p1, r1, sr1 = make_rings(lam1)
    p2, r2, sr2 = make_rings(lam2)

    a1 = TolanskyAnalyser(p1, r1, sr1, lam_nm=lam1*1e9, d_m=d, pixel_pitch_m=px)
    a2 = TolanskyAnalyser(p2, r2, sr2, lam_nm=lam2*1e9, d_m=d, pixel_pitch_m=px)
    tla = TwoLineAnalyser(a1, a2, lam1_nm=lam1*1e9, lam2_nm=lam2*1e9,
                           d_prior_m=d, n=n, pixel_pitch_m=px)
    res = tla.run()

    expected_ratio = lam2 / lam1
    actual_ratio   = res.S2 / res.S1
    assert abs(actual_ratio - expected_ratio) < 1e-10, \
        f"Slope ratio {actual_ratio:.10f} ≠ λ₂/λ₁ = {expected_ratio:.10f}"
```

### T5 — Excess fractions recovers d to < 1 µm

```python
def test_excess_fractions_d_recovery():
    """
    Synthetic two-line data from known d = 20.008 mm.
    Recovered d must match to < 1 µm.
    """
    # (Use same setup as T4 but check d recovery)
    # ... (implementation follows T4 pattern)
    assert abs(res.d_m - 20.008e-3) < 1e-6, \
        f"|d_recovered - d_true| = {abs(res.d_m - 20.008e-3)*1e6:.3f} µm > 1 µm"
```

### T6 — N_int correctly identified from d_prior

```python
def test_N_int_from_prior():
    """
    N_int must be the correct integer for d = 20.008 mm.
    For λ₁=640.2248 nm, λ₂=638.2991 nm, n=1:
        N_int ≈ round(2 × 20.008e-3 × (1/640.2248e-9 − 1/638.2991e-9))
    """
    from fpi.tolansky_2026-04-05 import TwoLineAnalyser, TolanskyAnalyser
    # ... build analysers with exact synthetic data, check res.N_int
    expected_N_int = round(2 * 20.008e-3 * (1/640.2248e-9 - 1/638.2991e-9))
    assert res.N_int == expected_N_int, \
        f"N_int = {res.N_int}, expected {expected_N_int}"
```

### T7 — TolanskyPipeline from FringeProfile

```python
def test_pipeline_from_fringe_profile():
    """
    Build a FringeProfile with synthetic PeakFit entries (10 per family).
    TolanskyPipeline must recover d to < 0.1 mm and α to < 5%.
    """
    # Construct a minimal FringeProfile with 20 PeakFit entries
    # (10 at higher amplitude for λ₁, 10 at 0.8× for λ₂)
    # ... (see implementation note below)
    assert abs(result.d_m - 20.008e-3) < 1e-4, \
        f"d error = {abs(result.d_m - 20.008e-3)*1e3:.4f} mm"
    assert abs(result.alpha_rad_px - 1.6e-4) / 1.6e-4 < 0.05, \
        "α recovered to within 5%"
```

### T8 — TwoLineResult has all S04 two_sigma fields

```python
def test_two_sigma_fields_present():
    """All two_sigma_ fields must equal exactly 2 × sigma_ (S04 convention)."""
    # ... (run TwoLineAnalyser on synthetic data)
    assert abs(res.two_sigma_d_m   - 2.0 * res.sigma_d_m)   < 1e-15
    assert abs(res.two_sigma_f_px  - 2.0 * res.sigma_f_px)  < 1e-15
    assert abs(res.two_sigma_alpha - 2.0 * res.sigma_alpha)  < 1e-15
```

---

## 9. Expected numerical values

For the WindCube FlatSat configuration (d = 20.008 mm, f ≈ 200 mm, 2×2 binned):

| Quantity | Expected | Notes |
|----------|----------|-------|
| Slope S₁ (640 nm) | ~2500 px²/fringe | f=200 mm, d=20 mm → S=f²λ/d ≈ 200²×640e-9/0.020 |
| Slope ratio S₂/S₁ | 638.2991/640.2248 = 0.99699 | T4 |
| N_int | ~−189 | From d_prior and wavelength difference |
| d recovered | 20.008 ± ~0.001 mm | T5 |
| f recovered | ~200 mm (~6250 px) | Depends on actual lens |
| α recovered | ~1.607×10⁻⁴ rad/px | Tolansky result confirmed |
| CV(Δr²) | < 5% | Parallelism diagnostic |
| χ²/dof (joint fit) | ~1.0 | For correctly characterised σ_r |

---

## 10. `to_m05_priors` output

The dict returned by `TolanskyPipeline.to_m05_priors()` maps directly to
`M05FitConfig` fields:

```python
{
    't_init_mm':    20.108,          # d_m × 1000
    't_bounds_mm':  (20.088, 20.128),# ± 0.020 mm around recovered d
    'alpha_init':   1.607e-4,        # α rad/px
    'alpha_bounds': (1.407e-4, 1.807e-4),  # ± 12.5%
    'epsilon_cal_1': 0.34271,        # ε₁ for λ₁ = 640.2248 nm
    'epsilon_cal_2': 0.28965,        # ε₂ for λ₂ = 638.2991 nm
}
```

---

## 11. File locations in repository

```
soc_sewell/
├── fpi/
│   └── tolansky_2026-04-05.py
├── tests/
│   └── test_tolansky_2026-04-05.py
└── docs/specs/
    └── S13_tolansky_analysis_2026-04-05.md
```

---

## 12. Instructions for Claude Code

1. Read this entire spec AND S04 AND S12 before writing any code.
2. Confirm M03 tests pass: `pytest tests/test_m03_annular_reduction_*.py -v`
3. Implement `fpi/tolansky_2026-04-05.py` in this strict order:
   `TolanskyResult` → `TwoLineResult`
   → `TolanskyAnalyser` (with `_wls`, `_epsilon`, `_successive_diffs`,
      `_recover`, `run`, `print_table`)
   → `TwoLineAnalyser` (with `_joint_fit`, `_excess_fractions`,
      `_recover_f`, `run`, `print_summary`)
   → `TolanskyPipeline` (with `run`, `to_m05_priors`)
4. All `two_sigma_` fields must be set to exactly `2.0 × sigma_` (S04).
5. The joint fit uses `scipy.optimize.least_squares(method='lm')` with
   `ftol=xtol=gtol=1e-15, max_nfev=100_000`.
6. `_excess_fractions`: the sign convention is:
   `lever = λ₁·λ₂ / (2n·(λ₂−λ₁))` — this will be negative (λ₂ < λ₁).
   Take `d = abs(lever · (N_int + ε₁ − ε₂))` to ensure d > 0.
7. `TolanskyPipeline.run()` must raise `ValueError` if fewer than 3 peaks
   in either family after the amplitude split.
8. Write all 8 tests in `tests/test_tolansky_2026-04-05.py`.
9. Run: `pytest tests/test_tolansky_2026-04-05.py -v` — all 8 must pass.
10. Run full suite: `pytest tests/ -v` — no regressions.
11. Commit: `feat(tolansky): implement Tolansky two-line analysis, 8/8 tests pass`

Module docstring header:
```python
"""
Module:      tolansky_2026-04-05.py
Spec:        docs/specs/S13_tolansky_analysis_2026-04-05.md
Author:      Claude Code
Generated:   <today>
Last tested: <today>  (8/8 tests pass)
Project:     WindCube FPI Pipeline — NCAR/HAO
Repo:        soc_sewell
"""
```
