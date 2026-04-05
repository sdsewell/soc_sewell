# S04 — Uncertainty Standards and Reporting Conventions

**Spec ID:** S04
**Spec file:** `specs/S04_uncertainty_standards_2026-04-05.md`
**Project:** WindCube FPI Science Operations Center Pipeline
**Institution:** NCAR / High Altitude Observatory (HAO)
**Status:** Authoritative
**Depends on:** S01, S02, S03
**Used by:** S09, S10, S11, S12, S13, S14, S15, S16, S17
**Last updated:** 2026-04-05
**Created/Modified by:** Claude AI

---

## 1. Purpose

This document defines the conventions for reporting uncertainties, reduced
chi-squared statistics, and quality flags throughout the WindCube pipeline.
Every module that produces fitted or estimated quantities must follow these
conventions exactly. This ensures that σ values are meaningful, that
comparisons between modules are consistent, and that the L2 wind product
carries calibrated uncertainties traceable back to detector-level noise.

This spec does not define algorithms. It defines naming rules, units, and
quality flag patterns that every module spec must follow.

---

## 2. The σ and 2σ rule

**Every fitted or estimated parameter that is reported as an output must have
both a 1σ field and a 2σ field.** No exceptions.

The naming pattern is:

```python
# For a scalar parameter named X:
X              : float   # best-fit value
sigma_X        : float   # 1σ uncertainty (68% confidence interval, Gaussian)
two_sigma_X    : float   # exactly 2.0 * sigma_X (95% confidence interval)
```

The `two_sigma_X` field is always computed as:

```python
two_sigma_X = 2.0 * sigma_X   # ALWAYS — never independently computed
```

This rule applies to:
- Every fitted parameter in `CalibrationResult` (M05)
- Every fitted parameter in `AirglowFitResult` (M06)
- Every wind component in `WindResult` (M07)
- Every centre coordinate in `FringeProfile` (M03)

### 2.1 Rationale

Why carry 2σ explicitly rather than computing it from σ at the point of use?

1. **Traceability.** Downstream modules and data product schemas reference
   `two_sigma_X` by name. Having it as an explicit field ensures it cannot be
   accidentally computed with a different factor.
2. **Reporting standard.** The space weather and upper atmosphere community
   typically reports 2σ (≈95% confidence) in published results (e.g. ICON/MIGHTI
   wind products, TIMED/TIDI). Carrying 2σ explicitly makes the L2 output
   consistent with these benchmarks.
3. **Consistency with STM.** The STM wind precision requirement of 9.8 m/s
   is a 1σ Monte Carlo result. Both σ and 2σ are needed to assess compliance.

### 2.2 Example

```python
@dataclass
class AirglowFitResult:
    # Doppler-shifted line centre
    lambda_c_m          : float   # best-fit value, metres
    sigma_lambda_c_m    : float   # 1σ uncertainty, metres
    two_sigma_lambda_c_m: float   # exactly 2.0 * sigma_lambda_c_m

    # LOS wind velocity
    v_rel_ms            : float   # best-fit value, m/s
    sigma_v_rel_ms      : float   # 1σ uncertainty, m/s
    two_sigma_v_rel_ms  : float   # exactly 2.0 * sigma_v_rel_ms

    # Fit quality
    chi2_reduced        : float
    quality_flags       : int
```

And at construction time:

```python
result = AirglowFitResult(
    lambda_c_m           = lmfit_result.params['lambda_c'].value,
    sigma_lambda_c_m     = lmfit_result.params['lambda_c'].stderr,
    two_sigma_lambda_c_m = 2.0 * lmfit_result.params['lambda_c'].stderr,
    v_rel_ms             = compute_v_rel(lmfit_result),
    sigma_v_rel_ms       = propagate_sigma_v(lmfit_result),
    two_sigma_v_rel_ms   = 2.0 * propagate_sigma_v(lmfit_result),
    ...
)
```

---

## 3. Uncertainty sources and propagation chain

The uncertainty chain flows from detector noise upward through the pipeline:

```
CCD detector noise
  ├── Readout noise: σ_read = CCD_READ_NOISE_EM_E (S03)
  └── Dark current: σ_dark = sqrt(dark_rate × t_exp × gain)  [Poisson]
         ↓
  Per-pixel ADU noise
         ↓ M03 annular reduction
  FringeProfile.sigma_profile  (SEM = std / sqrt(N_pixels) per r² bin)
         ↓ M05 calibration inversion (LM fit)
  CalibrationResult.sigma_*   (from lmfit covariance matrix diagonal)
         ↓ M06 airglow inversion (LM fit, using cal result as fixed params)
  AirglowFitResult.sigma_lambda_c_m
         ↓ Doppler formula: v = c × (λ_c - λ₀) / λ₀
  AirglowFitResult.sigma_v_rel_ms
         ↓ M07 WLS wind retrieval
  WindResult.sigma_v_zonal_ms, sigma_v_meridional_ms
```

### 3.1 Per-bin SEM in FringeProfile

The standard error of the mean (SEM) per r² bin in M03:

```
sigma_profile[i] = std(ADU values in bin i) / sqrt(N_pixels[i])
```

where `N_pixels[i]` is the count of CCD pixels (not sub-pixels) contributing
to bin i. This is the input uncertainty to all subsequent fitting steps.

Bins with `N_pixels[i] < min_pixels_per_bin` are masked: `masked[i] = True`,
`sigma_profile[i] = np.inf`. Masked bins are excluded from chi-squared
calculation.

### 3.2 Uncertainty from LM fit (lmfit)

M05 and M06 use the `lmfit` package's Levenberg-Marquardt optimizer. After
a successful fit, parameter uncertainties come from the covariance matrix:

```python
sigma_param = result.params['param_name'].stderr
```

If `stderr` is `None` (fit did not converge or Hessian is singular),
the module must:
1. Set `sigma_param = np.inf`
2. Set the corresponding quality flag (see Section 5)
3. Not raise an exception — return the result with the quality flag set

### 3.3 Uncertainty propagation: λ_c → v_rel

The Doppler formula (see S03) gives:

```
v_rel = c × (λ_c - λ₀) / λ₀
```

Propagating uncertainty by linear approximation (valid because λ₀ has
negligible uncertainty relative to σ_λ_c):

```
σ_v_rel = c × σ_λ_c / λ₀
        = SPEED_OF_LIGHT_MS × sigma_lambda_c_m / OI_WAVELENGTH_M
```

This is implemented in M06. Do not implement it elsewhere.

**Numerical check:** For a typical σ_λ_c ≈ 2e-14 m (0.02 pm):
```
σ_v_rel = 299792458 × 2e-14 / 630.0304e-9 ≈ 9.5 m/s
```
This is close to the STM 1σ wind budget of 9.8 m/s, as expected.

---

## 4. Reduced chi-squared convention

Every fitting step (M05, M06) must report `chi2_reduced`:

```
chi2_reduced = sum[ ((data[i] - model[i]) / sigma[i])^2 ]  /  (N - P)
```

where:
- `data[i]` = `FringeProfile.profile[i]` (for non-masked bins only)
- `model[i]` = best-fit model evaluated at same r² bin
- `sigma[i]` = `FringeProfile.sigma_profile[i]`
- `N` = number of non-masked bins used in fit
- `P` = number of free parameters in the fit stage

### 4.1 Acceptable range

| chi2_reduced range | Interpretation | Action |
|-------------------|----------------|--------|
| 0.5 – 3.0 | Acceptable fit | No flag |
| 3.0 – 10.0 | Marginal fit | Set `CHI2_HIGH` flag; warn in log |
| > 10.0 | Poor fit | Set `CHI2_VERY_HIGH` flag; flag for review |
| < 0.5 | Overfit or noise underestimated | Set `CHI2_LOW` flag; warn in log |
| NaN or Inf | Fit failure | Set `FIT_FAILED` flag |

A `chi2_reduced` in range [0.5, 3.0] does not guarantee a good fit — it
only means the residuals are consistent with the noise model. Always inspect
the residual plot when diagnosing anomalies.

### 4.2 Why chi2_reduced matters for wind bias

The 9.8 m/s wind bias budget (STM) assumes that M05 and M06 produce
unbiased fits with `chi2_reduced ≈ 1.0`. If `chi2_reduced >> 1`, the
noise model is wrong or the model does not fit the data — the reported σ_v
underestimates the true uncertainty. If `chi2_reduced << 1`, the noise is
overestimated and the fit is excessively conservative.

The first time a new dataset is processed, always plot the chi2_reduced
distribution across frames and verify it peaks near 1.0.

---

## 5. Quality flags (bitmask convention)

Every output dataclass that carries quality information uses an integer
bitmask field named `quality_flags`. Individual flags are combined with
bitwise OR (`|`). Individual flags are checked with bitwise AND (`&`).

### 5.1 Global flag definitions

These flags apply across all modules. Module-specific flags are defined in
the module spec (S12 for M03, S13 for M05, S14 for M06, S15 for M07)
and must use values that do not conflict with the global flags below.

**Global flags (bits 0–3, values 0x01–0x08):**

```python
class PipelineFlags:
    """Global quality flags — defined in S04, used by all modules."""
    GOOD              = 0x00  # No issues; all tests passed
    FIT_FAILED        = 0x01  # Optimizer did not converge or Hessian singular
    CHI2_HIGH         = 0x02  # chi2_reduced > 3.0
    CHI2_VERY_HIGH    = 0x04  # chi2_reduced > 10.0 (implies CHI2_HIGH)
    CHI2_LOW          = 0x08  # chi2_reduced < 0.5
```

**Module-specific flags use bits 4–15:**
- M03 (FringeProfile): bits 4–10, defined in S12
- M05 (CalibrationResult): bits 4–10, defined in S13
- M06 (AirglowFitResult): bits 4–9, defined in S14
- M07 (WindResult): bits 4–9, defined in S15

Bits 11–15 are reserved for future use.

### 5.2 Flag usage pattern

```python
# Setting flags
result.quality_flags = PipelineFlags.GOOD
if chi2_reduced > 3.0:
    result.quality_flags |= PipelineFlags.CHI2_HIGH
if chi2_reduced > 10.0:
    result.quality_flags |= PipelineFlags.CHI2_VERY_HIGH

# Checking flags
if result.quality_flags & PipelineFlags.FIT_FAILED:
    print("Fit failed — do not use")

if result.quality_flags == PipelineFlags.GOOD:
    print("All quality tests passed")
```

### 5.3 Downstream handling of flagged results

- A result with `quality_flags != GOOD` is not automatically discarded.
  The pipeline continues and the flags are propagated to the L2 product.
- The L2 data product (S19) includes `quality_flags` for each wind vector.
  Users can filter on this field.
- Integration notebooks (S16, S17) must display the fraction of flagged
  frames as part of their summary statistics.

---

## 6. FringeProfile field naming addendum

The `FringeProfile` dataclass (defined in full in S12) must follow the
σ/2σ naming rule for every estimated quantity:

| Field | Type | Description |
|-------|------|-------------|
| `sigma_profile` | ndarray | 1σ SEM per r² bin |
| `two_sigma_profile` | ndarray | Exactly `2 × sigma_profile` |
| `sigma_cx` | float | 1σ centre uncertainty, x |
| `two_sigma_cx` | float | Exactly `2 × sigma_cx` |
| `sigma_cy` | float | 1σ centre uncertainty, y |
| `two_sigma_cy` | float | Exactly `2 × sigma_cy` |

These field names are fixed. S12 must define them with exactly these names.

---

## 7. Units table

All quantities in the pipeline use SI units unless explicitly noted.
The following are the standard units for wind-relevant quantities:

| Quantity | Unit | Symbol |
|----------|------|--------|
| Wavelength | metres | m |
| Wind speed | metres per second | m/s |
| Angle (geodetic) | degrees | deg |
| Angle (geometric, optical) | radians | rad |
| Pixel position | pixels | px |
| Pixel radius squared | pixels squared | px² |
| Time | seconds (Unix epoch) | s |
| Temperature | Kelvin | K |
| Altitude | kilometres | km |
| Image intensity | ADU (analogue-digital units) | ADU |

**Never mix units within a calculation.** The most common error is mixing
nm and m for wavelength. Use `OI_WAVELENGTH_M` (in metres) from S03
everywhere — never write `630.0304` as a bare number in a calculation.

---

## 8. Verification tests

All tests in `tests/test_s04_conventions.py`.

These tests do not test any specific module — they test the convention itself
by creating synthetic dataclass instances and checking the naming rules.

### T1 — two_sigma is exactly 2 × sigma

```python
def test_two_sigma_exact():
    """two_sigma must always be exactly 2.0 × sigma, never independently computed."""
    import numpy as np
    sigma_val = 3.7  # arbitrary
    two_sigma_val = 2.0 * sigma_val
    assert two_sigma_val == 2.0 * sigma_val
    # Check floating point: 2.0 * x must equal 2.0 * x (trivially true,
    # but this documents the intent)
    assert two_sigma_val / sigma_val == 2.0
```

### T2 — chi2_reduced acceptable range boundaries

```python
def test_chi2_flag_logic():
    """Verify flag assignment logic for chi2_reduced values."""
    def assign_flags(chi2):
        flags = 0x00
        if chi2 > 3.0:  flags |= 0x02
        if chi2 > 10.0: flags |= 0x04
        if chi2 < 0.5:  flags |= 0x08
        return flags

    assert assign_flags(1.0) == 0x00   # GOOD
    assert assign_flags(5.0) == 0x02   # CHI2_HIGH
    assert assign_flags(15.0) == 0x06  # CHI2_HIGH | CHI2_VERY_HIGH
    assert assign_flags(0.3) == 0x08   # CHI2_LOW
```

### T3 — sigma_v from sigma_lambda propagation

```python
def test_sigma_v_propagation():
    """Verify Doppler uncertainty propagation: sigma_v = c * sigma_lambda / lambda0."""
    from src.constants import SPEED_OF_LIGHT_MS, OI_WAVELENGTH_M
    sigma_lambda_c = 2.0e-14   # 0.02 pm, typical value
    sigma_v_expected = SPEED_OF_LIGHT_MS * sigma_lambda_c / OI_WAVELENGTH_M
    assert 8.0 < sigma_v_expected < 12.0, \
        f"sigma_v = {sigma_v_expected:.2f} m/s; expected ~9.5 m/s for sigma_lambda = 0.02 pm"
```

### T4 — Quality flag bitwise operations

```python
def test_quality_flag_operations():
    """Bitwise flag operations work correctly."""
    GOOD = 0x00
    FIT_FAILED = 0x01
    CHI2_HIGH = 0x02

    flags = GOOD
    assert flags == GOOD

    flags |= CHI2_HIGH
    assert flags & CHI2_HIGH
    assert not (flags & FIT_FAILED)

    flags |= FIT_FAILED
    assert flags & CHI2_HIGH
    assert flags & FIT_FAILED
    assert flags != GOOD
```

### T5 — Doppler formula sign convention

```python
def test_doppler_sign_convention():
    """Positive v_rel (recession) produces lambda_c > lambda_0."""
    from src.constants import SPEED_OF_LIGHT_MS, OI_WAVELENGTH_M
    v_rel = +100.0   # m/s, positive = recession = redshift
    lambda_c = OI_WAVELENGTH_M * (1 + v_rel / SPEED_OF_LIGHT_MS)
    assert lambda_c > OI_WAVELENGTH_M, \
        "Recession should produce redshift (lambda_c > lambda_0)"

    v_rel_check = SPEED_OF_LIGHT_MS * (lambda_c - OI_WAVELENGTH_M) / OI_WAVELENGTH_M
    assert abs(v_rel_check - v_rel) < 0.01, \
        "Round-trip Doppler formula failed"
```

### T6 — STM wind precision is achievable

```python
def test_stm_wind_budget():
    """sigma_v of 9.8 m/s corresponds to sigma_lambda ~ 2.06e-14 m at OI 630 nm."""
    from src.constants import SPEED_OF_LIGHT_MS, OI_WAVELENGTH_M, WIND_BIAS_BUDGET_MS
    sigma_lambda_required = WIND_BIAS_BUDGET_MS * OI_WAVELENGTH_M / SPEED_OF_LIGHT_MS
    # Should be ~2.06e-14 m = 0.0206 pm
    assert 1.5e-14 < sigma_lambda_required < 2.5e-14, \
        f"sigma_lambda required for STM = {sigma_lambda_required:.3e} m; expected ~2.06e-14 m"
```

---

## 9. Expected numerical values

| Quantity | Expected value | Derivation |
|----------|----------------|-----------|
| σ_v corresponding to σ_λ = 2e-14 m | ≈ 9.5 m/s | c × 2e-14 / 630.0304e-9 |
| σ_λ required for 9.8 m/s STM wind budget | ≈ 2.06e-14 m | 9.8 × 630.0304e-9 / c |
| chi2_reduced = 1.0 | Ideal fit; noise model exactly correct | — |
| Two-sigma coverage | ≈ 95.4% | Gaussian, ±2σ |
| One-sigma coverage | ≈ 68.3% | Gaussian, ±1σ |

---

## 10. File location in repository

```
soc_sewell/
├── src/
│   └── constants.py              ← PipelineFlags class may live here
└── tests/
    └── test_s04_conventions.py   ← tests T1–T6
```

The `PipelineFlags` class may be defined either in `src/constants.py` (as a
class of integer literals) or in `src/quality_flags.py`. The project lead
will decide at implementation time. Either choice is acceptable provided
all modules import from the same single location.

---

## 11. Instructions for Claude Code

1. Read this entire spec before writing any code.
2. Create `tests/test_s04_conventions.py` with tests T1–T6.
3. Define the `PipelineFlags` class in `src/constants.py` (appended at the
   end of the file, after all numerical constants). If the file is getting
   too long, create `src/quality_flags.py` instead — but check with the
   project lead first.
4. Run `pytest tests/test_s04_conventions.py -v` — all 6 tests must pass.
5. There is no separate Python module to create for S04 — the conventions
   it defines are enforced in the individual module specs (S12–S15).

**Commit message template:**
```
feat(conventions): implement S04 uncertainty standards and quality flags, 6/6 tests pass
Implements: S04_uncertainty_standards_2026-04-05.md
```
