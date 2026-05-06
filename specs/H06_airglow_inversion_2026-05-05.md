# H06 — Airglow Fringe Inversion

**Spec ID:** H06
**Spec file:** `docs/specs/H06_airglow_inversion_2026-05-05.md`
**Project:** WindCube FPI Science Operations Center Pipeline
**Institution:** NCAR / High Altitude Observatory (HAO)
**Status:** Authoritative
**Depends on:** S01, S02, S03, S04, H01 (Airy forward model),
S11 (M04 — airglow synthesis), S12 (M03 — FringeProfile),
H05 (calibration inversion — provides CalibrationResult)
**Used by:** S16 (M07 — wind retrieval receives AirglowFitResult),
S17 (INT02), S18 (INT03)
**Last updated:** 2026-05-05

> **What changed from 2026-04-06:**
> 1. **`FOCAL_LENGTH_M` removed** from the constants import block (Section 4).
>    H06 never needed focal length — `alpha` in the `CalibrationResult` received
>    from H05 encapsulates all lens geometry. The import was vestigial.
> 2. **References updated** throughout: S09/M01 → H01; S14/M05 → H05;
>    spec ID corrected from S15 to H06.
> 3. **Section 3 (physical background) updated** to note that `t` and `alpha`
>    in the `CalibrationResult` are the Tolansky-seeded, H05-refined values
>    from H05 §8 (Group A parameters), consistent with H01 §4.2 and H05 §1.
> 4. **Instructions for Claude Code added** (Section 13 — was absent from the
>    2026-04-06 version).
> 5. **File locations updated** to reflect new date suffix.
> 6. No changes to the fitting algorithm, forward model, dataclasses, or tests.

---

## 1. Purpose

H06 inverts the 1D OI 630 nm airglow `FringeProfile` from M03 to recover
the Doppler-shifted line centre `λ_c`. All ten instrument parameters are
fixed at the values determined by H05 from the neon calibration frame.
The only free parameters are the three airglow source parameters.

The recovered `λ_c` is converted to a line-of-sight wind speed `v_rel`:

```
v_rel = c × (λ_c − OI_WAVELENGTH_AIR_M) / OI_WAVELENGTH_AIR_M
```

This `v_rel` is the primary output of H06 and the primary input to M07.

**Relationship to H01 and H05:** H06 is a pure consumer of H01 and H05.
It calls `airy_modified()` from H01 (via `CalibrationResult` instrument
parameters) and receives all 10 instrument parameters from H05 as fixed
constants. H06 has no forward-model mathematics of its own.

**What H06 is not.** H06 does not decompose v_rel into zonal and meridional
components — that is M07's job. H06 does not characterise the etalon — that
is H05's job. H06 does not retrieve temperature — that is not a WindCube
science goal (see Section 2.1).

---

## 2. Key design decisions

### 2.1 Delta-function source model — do not reopen this decision

WindCube's science goals are wind speed and direction. Temperature retrieval
is explicitly **not** in the science requirements (STM v1). The OI 630 nm
emission line is therefore modelled as a **spectral delta function** — a
single infinitely narrow line at `λ_c`. This is equivalent to setting
`Δλ → 0` in the Harding Gaussian model (Harding Eq. 10).

This is consistent with H01 §4.3, which implements the same delta-function
approximation in `make_airglow_spectrum()` and explicitly excludes Harding
Eq. 12 (temperature broadening).

**Consequence:** H06 has **three free parameters** `{Y_line, B_sci, λ_c}`,
not Harding's five `{B, Y_bg, Y_line, λ_c, Δλ}`. Temperature broadening
and background sky emission are not fitted. This simplification:
- Reduces parameter degeneracy (no Δλ–λ_c correlation)
- Halves the number of free parameters compared to the full Harding model
- Is consistent with M04 (airglow synthesis) which also uses a delta
  function source when `temperature_K=None` is passed

**Implication for M04 consistency:** When running the synthetic round-trip
test (T5), M04 must generate the airglow image with the delta-function model
(i.e., pass `temperature_K=None` or set `Δλ → 0`). If M04 generates with
a thermally broadened line and H06 inverts with a delta function, there will
be a systematic model mismatch. For the operational pipeline this is
acceptable (the thermal broadening is small compared to the fringe width),
but the round-trip test should be self-consistent.

### 2.2 Instrument parameters are fixed — not refitted

All ten `CalibrationResult` parameters (`t_m`, `R_refl`, `alpha`, `I0`, `I1`,
`I2`, `sigma0`, `sigma1`, `sigma2`, `B_cal`) are received from H05 and used
to construct the forward model without refitting. Among these, `t_m` and
`alpha` are the Tolansky-seeded, H05-refined values (H05 Group A). The
remaining eight were fitted from the neon fringe shape (H05 Group B).
H06 treats all ten identically — fixed constants for the airglow inversion.

The `B` in H06 (`B_sci`) is a separate free parameter — the science frame
bias may differ from the calibration frame bias due to dark current variation.

### 2.3 Fine-grid forward model

H06 uses a dense uniform-r grid to evaluate the Airy model, then bin-averages
to match the M03 r² bins. This prevents the grid-spacing bias that inflates χ²
and corrupts parameter uncertainties. The same strategy is used by H05.

### 2.4 Soft-bound penalty residuals

LM (`scipy.optimize.least_squares(method='lm')`) does not enforce bounds.
Penalty residuals are added outside the effective bounds for each parameter,
exactly as in H05. See Section 7.3.

### 2.5 Lambda_c initialisation — brute-force scan over one FSR

The initial guess for `λ_c` is the most critical initialisation in H06.
A wrong initial λ_c by more than ~FSR/2 will cause the fit to converge to
a false minimum. The brute-force scan over one FSR finds the correct period
before any LM step begins. See Section 7.2 for the rationale and the ~4723 m/s
FSR velocity equivalent.

---

## 3. Physical background

### 3.1 Forward model for the airglow fringe

With a delta-function source at `λ_c`, the forward model (H01 Eq. 1 in the
delta-function limit) is:

```
s(r) = Y_line · Ã(r; λ_c) + B_sci
```

where `Ã(r; λ_c)` is the modified Airy function from H01, evaluated at the
fitted `λ_c` using the fixed `CalibrationResult` instrument parameters.

The full set of fixed parameters passed from `CalibrationResult` to the
H01 `airy_modified()` call are:

| Parameter | H05 group | Role in H06 |
|---|---|---|
| `t_m` | Group A (Tolansky-seeded) | OPD calculation |
| `alpha` | Group A (Tolansky-seeded) | θ(r) = arctan(α·r) |
| `R_refl` | Group B (fitted from fringe shape) | Finesse coefficient F |
| `I0`, `I1`, `I2` | Group B | Intensity envelope |
| `sigma0`, `sigma1`, `sigma2` | Group B | PSF width σ(r) |
| `B_cal` | Group B | Fixed bias (calibration frame only) |

Note: The `I0`, `I1`, `I2` intensity envelope from `CalibrationResult` is
absorbed into `Y_line` — H06 fits a single scalar intensity multiplier
rather than three envelope coefficients. The envelope shape is fixed; only
its overall scale is free. This keeps the model linear in `Y_line` and
reduces degeneracy.

```python
def _airglow_model(r_fine, lambda_c_m, Y_line, B_sci, cal):
    """
    Delta-function airglow fringe at lambda_c_m.

    r_fine     : fine uniform-r grid, pixels
    lambda_c_m : free parameter — Doppler-shifted line centre
    Y_line     : free parameter — line intensity scale factor
    B_sci      : free parameter — science frame CCD bias
    cal        : CalibrationResult — all 10 instrument parameters fixed

    Returns model profile on r_fine grid (before bin-averaging).
    """
    profile_fine = Y_line * airy_modified(
        r_fine, lambda_c_m,
        t=cal.t_m, R=cal.R_refl, alpha=cal.alpha, n=1.0,
        r_max=r_fine[-1],
        I0=cal.I0, I1=cal.I1, I2=cal.I2,
        sigma0=cal.sigma0, sigma1=cal.sigma1, sigma2=cal.sigma2,
    ) + B_sci
    return profile_fine
```

Then bin-average `profile_fine` to M03 r² bin centres before computing
residuals (identical pattern to H05's `_neon_model`).

### 3.2 Doppler shift — λ_c to v_rel (H01 Eq. 11, inverted)

```
v_rel = c × (λ_c − λ₀) / λ₀

where:
  λ₀ = OI_WAVELENGTH_AIR_M = 630.0304e-9 m  (from windcube/constants.py)
  c  = SPEED_OF_LIGHT_MS                     (from windcube/constants.py)
```

Positive `v_rel` = recession (redshift), consistent with H01 §4.3 velocity
sign convention.

### 3.3 Uncertainty propagation

From the LM covariance matrix, `sigma_lambda_c_m` is the 1σ uncertainty
on `λ_c`. The wind uncertainty propagates linearly:

```
sigma_v_rel = c × sigma_lambda_c_m / OI_WAVELENGTH_AIR_M
```

**Numerical check:** For the STM wind budget of 9.8 m/s:
```
sigma_lambda_c required = 9.8 × 630.0304e-9 / 299792458 ≈ 2.06e-14 m (0.021 pm)
```

---

## 4. Physical constants from `windcube/constants.py`

```python
from windcube.constants import (
    OI_WAVELENGTH_AIR_M,   # 630.0304e-9 m — OI rest wavelength (air)
    SPEED_OF_LIGHT_MS,     # 299_792_458.0 m/s
    ETALON_GAP_M,          # 20.008e-3 m — used to compute FSR for λ_c scan
)
```

The FSR at the OI wavelength is computed inline rather than imported as a
separate constant:

```python
FSR_OI_M = OI_WAVELENGTH_AIR_M**2 / (2.0 * ETALON_GAP_M)   # ≈ 9.922e-15 m
```

**Note:** `FOCAL_LENGTH_M` is **not** imported here. The `alpha` plate scale
needed by the H01 Airy forward model is provided by `CalibrationResult.alpha`,
which was recovered by H05 (seeded by Tolansky). No focal length arithmetic
is needed in H06.

The H01 forward model function is imported as:

```python
from fpi.airy_forward_model import airy_modified
```

The `CalibrationResult` dataclass is imported from H05:

```python
from src.fpi.m05_calibration_inversion import CalibrationResult
```

---

## 5. Free parameters and bounds

Three free parameters only:

| Parameter | Symbol | Physical meaning | Initial estimate | Bounds |
|-----------|--------|-----------------|-----------------|--------|
| Line centre | `lambda_c_m` | Doppler-shifted wavelength | Brute-force scan result | `OI_WAVELENGTH_AIR_M ± 1.5 × FSR_OI_M` |
| Line intensity | `Y_line` | Scale factor on Airy amplitude | `max(profile) / max(Airy(λ₀))` | (0, ∞) |
| Science bias | `B_sci` | CCD bias in science frame | `min(profile) × 0.8` | (0, `min(profile) × 1.5`) |

**Why λ_c bounds span ±1.5 FSR:** The Doppler shift for the maximum
expected storm wind of 400 m/s corresponds to:
```
Δλ = λ₀ × v/c = 630.0304e-9 × 400/299792458 ≈ 8.4e-13 m = 0.84 pm
FSR ≈ 9.92 pm at 20 mm gap
```
So 400 m/s is only 0.085 FSR. The ±1.5 FSR bound is generous and prevents
the fit from jumping to adjacent fringe orders while still excluding all
physically impossible states.

---

## 6. Output dataclass

Per S04 — every fitted parameter must have `sigma_` and `two_sigma_` fields.

```python
@dataclass
class AirglowFitResult:
    """
    Output of H06 airglow inversion.
    Passed to M07 (wind retrieval) for LOS-to-vector decomposition.
    """
    # Primary output — line centre and wind
    lambda_c_m:              float   # fitted line centre, metres
    sigma_lambda_c_m:        float   # 1σ uncertainty, metres
    two_sigma_lambda_c_m:    float   # exactly 2 × sigma_lambda_c_m  (S04)

    v_rel_ms:                float   # LOS wind speed, m/s
    sigma_v_rel_ms:          float   # 1σ uncertainty, m/s
    two_sigma_v_rel_ms:      float   # exactly 2 × sigma_v_rel_ms    (S04)

    # Other fitted parameters (for diagnostics)
    Y_line:                  float
    sigma_Y_line:            float
    two_sigma_Y_line:        float

    B_sci:                   float   # fitted science frame bias, ADU
    sigma_B_sci:             float
    two_sigma_B_sci:         float

    # Fit quality
    chi2_reduced:            float   # must be in [0.5, 3.0] for good fit
    n_bins_used:             int
    n_params_free:           int     # always 3 for H06
    converged:               bool
    quality_flags:           int     # AirglowFitFlags bitmask

    # Phase relationship to calibration (diagnostic)
    epsilon_sci:             float   # (2 × lambda_c_m / OI_WAVELENGTH_AIR_M) mod 1
    delta_epsilon:           float   # epsilon_sci − cal.epsilon_cal
                                     # fractional phase shift from wind

    # Input traceability
    calibration_t_m:         float   # t_m from CalibrationResult used
    calibration_epsilon_cal: float   # epsilon_cal from CalibrationResult used

    # LM scan diagnostics
    lambda_c_scan_init_m:    float   # λ_c at start of brute-force scan
    lambda_c_lm_init_m:      float   # λ_c passed to LM after scan


class AirglowFitFlags:
    """Bitmask quality flags for AirglowFitResult. Uses bits 4+ per S04."""
    GOOD                  = 0x00
    FIT_FAILED            = 0x01   # LM did not converge
    CHI2_HIGH             = 0x02   # chi2 > 3.0
    CHI2_VERY_HIGH        = 0x04   # chi2 > 10.0
    CHI2_LOW              = 0x08   # chi2 < 0.5
    SCAN_AMBIGUOUS        = 0x10   # brute-force scan has two minima < 10% apart
    LAMBDA_C_AT_BOUND     = 0x20   # lambda_c hit its bound (possible FSR jump)
    STDERR_NONE           = 0x40   # any stderr is None (singular covariance)
    LOW_SNR               = 0x80   # estimated SNR < 1.0 (Y_line / B_sci < 1)
    CAL_QUALITY_DEGRADED  = 0x100  # CalibrationResult had non-GOOD quality flags
```

---

## 7. Fit procedure

### 7.1 Step 0 — Validate inputs

```python
def fit_airglow_fringe(
    profile: FringeProfile,
    cal: CalibrationResult,
    n_fine: int = 500,
) -> AirglowFitResult:
    """
    Invert an OI 630 nm science FringeProfile to recover v_rel.

    Parameters
    ----------
    profile : FringeProfile
        From M03 reduce_science_frame(). Must have dark subtraction applied.
        profile.quality_flags is checked — CENTRE_FAILED raises ValueError.
    cal : CalibrationResult
        From H05 fit_calibration_fringe(). All 10 instrument parameters
        are used as fixed constants. If cal.quality_flags is non-GOOD,
        set CAL_QUALITY_DEGRADED flag in output but do not abort.
    n_fine : int
        Number of points in the fine uniform-r grid for forward model
        evaluation. Default 500. Do not reduce below 200.

    Returns
    -------
    AirglowFitResult

    Raises
    ------
    ValueError
        If profile.quality_flags & CENTRE_FAILED.
        If fewer than 10 unmasked bins remain.
    RuntimeError
        If any parameter stderr is None after fit.
    """
```

Check inputs:
```python
if profile.quality_flags & QualityFlags.CENTRE_FAILED:
    raise ValueError("FringeProfile has CENTRE_FAILED flag — cannot invert")
n_good = np.sum(~profile.masked)
if n_good < 10:
    raise ValueError(f"Only {n_good} unmasked bins — need ≥ 10")
if cal.quality_flags != 0:
    result_flags |= AirglowFitFlags.CAL_QUALITY_DEGRADED
```

### 7.2 Step 1 — Brute-force scan over λ_c

```python
def _lambda_c_scan(
    profile: FringeProfile,
    cal: CalibrationResult,
    n_scan: int = 200,
    n_fine: int = 500,
) -> tuple[float, float]:
    """
    Scan lambda_c over one FSR to find the initial guess.

    Scans n_scan evenly-spaced values of lambda_c across
    [OI_WAVELENGTH_AIR_M - FSR_OI_M/2,
     OI_WAVELENGTH_AIR_M + FSR_OI_M/2].

    At each candidate lambda_c:
    1. Build the model profile (fine grid → bin-average)
    2. Analytically solve for Y_line and B_sci via least-squares
       (linear in these parameters given fixed lambda_c)
    3. Compute chi2

    Returns (lambda_c_best, chi2_min).
    Sets SCAN_AMBIGUOUS flag if the second-best chi2 is within 10%
    of chi2_min (two plausible minima — possible fringe-order confusion).
    """
```

**Why this scan is necessary:** The OI 630 nm fringe pattern looks
almost identical at `λ_c = λ₀ + k×FSR` for any integer k. Without a scan,
the LM fit starting from `λ₀` will converge to whichever FSR period happens
to have the nearest local minimum, which may not be the correct one.
A wrong-period initial `λ_c` would produce a wind error of approximately:

```
v_FSR = c × FSR_OI_M / OI_WAVELENGTH_AIR_M
      = 299792458 × 9.922e-15 / 630.0304e-9
      ≈ 4723 m/s
```

This error is catastrophic and undetectable from χ² alone if the LM
converges well. The scan prevents it.

### 7.3 Step 2 — LM fit

```python
def _run_airglow_lm(
    profile: FringeProfile,
    cal: CalibrationResult,
    lambda_c_init_m: float,
    Y_line_init: float,
    B_sci_init: float,
    n_fine: int = 500,
) -> scipy.optimize.OptimizeResult:
    """
    Run LM fit over {lambda_c_m, Y_line, B_sci}.

    Uses scipy.optimize.least_squares(method='lm').
    Weighted residuals: (data - model) / sigma_profile (non-masked only).

    Sigma floor: max(sigma_profile[i], 0.005 × median(profile[non-masked]))
    (same floor as H05)

    Soft-bound penalty residuals: one extra residual per free parameter,
    firing linearly outside the effective bounds in Section 5. Penalty
    weight = 1.0. Same pattern as H05.

    Convergence tolerances: ftol=xtol=gtol=1e-12.
    Max function evaluations: 50_000.
    """
```

### 7.4 Step 3 — Covariance and uncertainties

```python
J = result.jac          # Jacobian at solution, shape (n_residuals, 3)
J_data = J[:n_good, :]  # remove penalty rows
s2 = chi2_unweighted / (n_good - 3)
try:
    cov = s2 * np.linalg.inv(J_data.T @ J_data)
except np.linalg.LinAlgError:
    cov = s2 * np.linalg.pinv(J_data.T @ J_data)
    result_flags |= AirglowFitFlags.STDERR_NONE

sigma = np.sqrt(np.diag(cov))
# sigma[0] = sigma_lambda_c_m
# sigma[1] = sigma_Y_line
# sigma[2] = sigma_B_sci
```

### 7.5 Step 4 — Compute v_rel and epsilon_sci

```python
v_rel_ms = SPEED_OF_LIGHT_MS * (lambda_c_m - OI_WAVELENGTH_AIR_M) / OI_WAVELENGTH_AIR_M
sigma_v_rel_ms = SPEED_OF_LIGHT_MS * sigma_lambda_c_m / OI_WAVELENGTH_AIR_M

epsilon_sci = (2.0 * lambda_c_m / OI_WAVELENGTH_AIR_M) % 1.0
delta_epsilon = epsilon_sci - cal.epsilon_cal
```

---

## 8. Quality checks

```python
if chi2_reduced > 10.0:
    flags |= AirglowFitFlags.CHI2_VERY_HIGH | AirglowFitFlags.CHI2_HIGH
elif chi2_reduced > 3.0:
    flags |= AirglowFitFlags.CHI2_HIGH
elif chi2_reduced < 0.5:
    flags |= AirglowFitFlags.CHI2_LOW

lambda_c_lo = OI_WAVELENGTH_AIR_M - 1.5 * FSR_OI_M
lambda_c_hi = OI_WAVELENGTH_AIR_M + 1.5 * FSR_OI_M
if abs(lambda_c_m - lambda_c_lo) < 1e-15 or abs(lambda_c_m - lambda_c_hi) < 1e-15:
    flags |= AirglowFitFlags.LAMBDA_C_AT_BOUND

snr_estimate = (max(profile.profile[~profile.masked]) -
                min(profile.profile[~profile.masked])) / B_sci
if snr_estimate < 1.0:
    flags |= AirglowFitFlags.LOW_SNR
```

---

## 9. Verification tests

All tests in `tests/test_h06_airglow_inversion.py`.

### T1 — Output dataclass S04 compliance

```python
def test_two_sigma_convention(synthetic_airglow_profile, synthetic_cal_result):
    """All two_sigma_ fields must equal exactly 2.0 × sigma_."""
    result = fit_airglow_fringe(synthetic_airglow_profile, synthetic_cal_result)
    pairs = [
        ('lambda_c_m', 'sigma_lambda_c_m', 'two_sigma_lambda_c_m'),
        ('v_rel_ms',   'sigma_v_rel_ms',   'two_sigma_v_rel_ms'),
        ('Y_line',     'sigma_Y_line',      'two_sigma_Y_line'),
        ('B_sci',      'sigma_B_sci',       'two_sigma_B_sci'),
    ]
    for _, s_name, ts_name in pairs:
        sigma = getattr(result, s_name)
        two_sigma = getattr(result, ts_name)
        assert abs(two_sigma - 2.0 * sigma) < 1e-15, \
            f"{ts_name} = {two_sigma} ≠ 2 × {s_name} = {2*sigma}"
```

### T2 — Doppler formula: v_rel recovered from lambda_c

```python
def test_doppler_formula_consistency(synthetic_airglow_profile, synthetic_cal_result):
    """v_rel must equal c × (lambda_c - lambda_0) / lambda_0."""
    from windcube.constants import SPEED_OF_LIGHT_MS, OI_WAVELENGTH_AIR_M
    result = fit_airglow_fringe(synthetic_airglow_profile, synthetic_cal_result)
    v_check = SPEED_OF_LIGHT_MS * (result.lambda_c_m - OI_WAVELENGTH_AIR_M) / OI_WAVELENGTH_AIR_M
    assert abs(result.v_rel_ms - v_check) < 1e-6
```

### T3 — Zero wind: v_rel near zero for v_rel_truth = 0

```python
def test_zero_wind_recovery(synthetic_cal_result):
    """A zero-wind airglow image must recover v_rel within ±5 m/s."""
    from src.fpi.m04_airglow_synthesis import synthesise_airglow_image
    from src.fpi.m03_annular_reduction import reduce_science_frame
    from fpi.airy_forward_model import InstrumentParams
    params = InstrumentParams()
    result_m04 = synthesise_airglow_image(
        v_rel_ms=0.0, params=params, add_noise=False)
    fp = reduce_science_frame(
        result_m04['image_2d'],
        cx=params.r_max, cy=params.r_max,
        sigma_cx=0.05, sigma_cy=0.05,
        r_max_px=params.r_max,
    )
    result = fit_airglow_fringe(fp, synthetic_cal_result)
    assert abs(result.v_rel_ms) < 5.0
    assert result.converged
```

### T4 — Known wind: round-trip recovery within 20 m/s (noiseless)

```python
def test_known_wind_round_trip(synthetic_cal_result):
    """Noiseless round-trip: inject v_rel=200 m/s, recover to within 20 m/s."""
    from src.fpi.m04_airglow_synthesis import synthesise_airglow_image
    from src.fpi.m03_annular_reduction import reduce_science_frame
    from fpi.airy_forward_model import InstrumentParams
    v_truth = 200.0
    params = InstrumentParams()
    result_m04 = synthesise_airglow_image(
        v_rel_ms=v_truth, params=params, add_noise=False)
    fp = reduce_science_frame(
        result_m04['image_2d'],
        cx=params.r_max, cy=params.r_max,
        sigma_cx=0.05, sigma_cy=0.05,
        r_max_px=params.r_max,
    )
    result = fit_airglow_fringe(fp, synthetic_cal_result)
    assert abs(result.v_rel_ms - v_truth) < 20.0
    assert result.converged
```

### T5 — Noisy round-trip: |error| < 3 × sigma_v

```python
def test_noisy_round_trip_uncertainty_calibrated(synthetic_cal_result):
    """Noisy round-trip at SNR ≈ 5: recovered v_rel within 3σ of truth."""
    from src.fpi.m04_airglow_synthesis import synthesise_airglow_image
    from src.fpi.m03_annular_reduction import reduce_science_frame
    from fpi.airy_forward_model import InstrumentParams
    v_truth = 150.0
    params = InstrumentParams()
    rng = np.random.default_rng(42)
    result_m04 = synthesise_airglow_image(
        v_rel_ms=v_truth, params=params, snr=5.0,
        add_noise=True, rng=rng)
    fp = reduce_science_frame(
        result_m04['image_2d'],
        cx=params.r_max, cy=params.r_max,
        sigma_cx=0.05, sigma_cy=0.05,
        r_max_px=params.r_max,
    )
    result = fit_airglow_fringe(fp, synthetic_cal_result)
    assert abs(result.v_rel_ms - v_truth) < 3.0 * result.sigma_v_rel_ms
    assert 0.5 < result.chi2_reduced < 3.0
```

### T6 — Scan prevents FSR-period confusion

```python
def test_scan_prevents_fsr_confusion(synthetic_cal_result):
    """
    Inject v_rel = -300 m/s (blueshifted). Without scan, a naive LM
    starting from lambda_0 would converge to the wrong FSR period.
    With the scan, the correct lambda_c must be recovered.
    """
    from src.fpi.m04_airglow_synthesis import synthesise_airglow_image
    from src.fpi.m03_annular_reduction import reduce_science_frame
    from fpi.airy_forward_model import InstrumentParams
    v_truth = -300.0
    params = InstrumentParams()
    result_m04 = synthesise_airglow_image(
        v_rel_ms=v_truth, params=params, add_noise=False)
    fp = reduce_science_frame(
        result_m04['image_2d'],
        cx=params.r_max, cy=params.r_max,
        sigma_cx=0.05, sigma_cy=0.05,
        r_max_px=params.r_max,
    )
    result = fit_airglow_fringe(fp, synthetic_cal_result)
    # Must not be off by one FSR (~4723 m/s)
    assert abs(result.v_rel_ms - v_truth) < 100.0, \
        f"FSR confusion: recovered v_rel = {result.v_rel_ms:.0f} m/s, " \
        f"truth = {v_truth:.0f} m/s"
```

### T7 — CENTRE_FAILED profile raises ValueError

```python
def test_centre_failed_raises(synthetic_cal_result):
    """fit_airglow_fringe must raise ValueError for CENTRE_FAILED profile."""
    from src.fpi.m03_annular_reduction import FringeProfile, QualityFlags
    bad_profile = _make_minimal_fringe_profile()
    bad_profile.quality_flags = QualityFlags.CENTRE_FAILED
    with pytest.raises(ValueError, match="CENTRE_FAILED"):
        fit_airglow_fringe(bad_profile, synthetic_cal_result)
```

### T8 — sigma_v consistent with STM wind budget

```python
def test_sigma_v_within_stm_budget(synthetic_cal_result):
    """
    At SNR ≈ 5, sigma_v_rel should be ≤ 2 × STM budget (9.8 m/s).
    """
    from src.fpi.m04_airglow_synthesis import synthesise_airglow_image
    from src.fpi.m03_annular_reduction import reduce_science_frame
    from fpi.airy_forward_model import InstrumentParams
    from windcube.constants import WIND_BIAS_BUDGET_MS
    params = InstrumentParams()
    rng = np.random.default_rng(7)
    result_m04 = synthesise_airglow_image(
        v_rel_ms=100.0, params=params, snr=5.0,
        add_noise=True, rng=rng)
    fp = reduce_science_frame(
        result_m04['image_2d'],
        cx=params.r_max, cy=params.r_max,
        sigma_cx=0.05, sigma_cy=0.05,
        r_max_px=params.r_max,
    )
    result = fit_airglow_fringe(fp, synthetic_cal_result)
    assert result.sigma_v_rel_ms < 2.0 * WIND_BIAS_BUDGET_MS
```

---

## 10. Expected numerical values

| Quantity | Expected value | Notes |
|----------|----------------|-------|
| v_rel recovery (noiseless, v_truth=200) | within ±20 m/s | grid discretisation floor |
| v_rel recovery (noisy SNR=5, v_truth=150) | within 3σ | T5 |
| sigma_v_rel at SNR=5 | ≤ 2 × 9.8 m/s = 19.6 m/s | T8; STM budget |
| chi2_reduced (noisy) | 0.5–3.0 | T5 |
| sigma_lambda_c at 9.8 m/s budget | ≈ 2.06e-14 m (0.021 pm) | S04 derivation |
| FSR velocity equivalent | ≈ 4723 m/s | computed from ETALON_GAP_M |
| Max storm wind (400 m/s) in FSR units | ≈ 0.085 FSR | negligible ambiguity risk |
| n_params_free | 3 | always for H06 |
| lambda_c scan points | 200 | n_scan default |

---

## 11. Conftest fixtures

Add these fixtures to `tests/conftest.py`:

```python
import pytest
import numpy as np
from fpi.airy_forward_model import InstrumentParams
from src.fpi.m02_calibration_synthesis import synthesise_calibration_image
from src.fpi.m03_annular_reduction import reduce_calibration_frame
from src.fpi.tolansky import TolanskyPipeline
from src.fpi.m05_calibration_inversion import fit_calibration_fringe, FitConfig

@pytest.fixture(scope='session')
def synthetic_cal_result():
    """
    A CalibrationResult from a noiseless synthetic calibration image.
    Uses _build_tolansky_stub rather than TolanskyPipeline.run() to avoid
    amplitude-split reliability issues on synthetic data.
    Computed once per test session.
    """
    params = InstrumentParams()
    cal_m02 = synthesise_calibration_image(params, add_noise=False)
    fp = reduce_calibration_frame(
        cal_m02['image_2d'], cx_human=params.r_max, cy_human=params.r_max,
        r_max_px=params.r_max)
    tol_stub = _build_tolansky_stub(params)
    config = FitConfig(tolansky=tol_stub)
    return fit_calibration_fringe(fp, config)


@pytest.fixture
def synthetic_airglow_profile():
    """A noiseless FringeProfile from a 100 m/s airglow image."""
    from src.fpi.m04_airglow_synthesis import synthesise_airglow_image
    from src.fpi.m03_annular_reduction import reduce_science_frame
    params = InstrumentParams()
    cal_fp = reduce_calibration_frame(
        synthesise_calibration_image(params, add_noise=False)['image_2d'],
        cx_human=params.r_max, cy_human=params.r_max,
        r_max_px=params.r_max)
    sci_m04 = synthesise_airglow_image(v_rel_ms=100.0, params=params, add_noise=False)
    return reduce_science_frame(
        sci_m04['image_2d'],
        cx=cal_fp.cx, cy=cal_fp.cy,
        sigma_cx=cal_fp.sigma_cx, sigma_cy=cal_fp.sigma_cy,
        r_max_px=params.r_max)
```

---

## 12. File locations

```
soc_sewell/
├── src/fpi/
│   └── m06_airglow_inversion_2026_05_05.py
└── tests/
    ├── conftest.py          ← add synthetic_cal_result fixture here
    └── test_h06_airglow_inversion.py
```

---

## 13. Instructions for Claude Code

### Preamble — read before touching any file

1. Read this entire spec (H06).
2. Read H01 (`docs/specs/H01_airy_forward_model_2026-05-05.md`) in full.
3. Read H05 (`docs/specs/H05_calibration_inversion_2026-05-05.md`) in full.
4. Read S04 (output dataclass and uncertainty conventions).
5. Read S11 (M04 airglow synthesis) and S12 (FringeProfile).
6. Read the current `src/fpi/m06_airglow_inversion_*.py` (latest dated version).
7. Read the current `tests/test_*airglow_inversion*.py`.

Report which dated implementation files you found for steps 6 and 7 before
proceeding.

### Task sequence

**TASK A — Confirm prior tests pass**

```bash
pytest tests/ -v --tb=no -q
```

All existing tests must pass before proceeding. Stop and report any failures.

**TASK B — Verify constants**

Confirm `windcube/constants.py` exports all constants listed in H06 §4.
In particular:
- `FOCAL_LENGTH_M` is **not** imported by this module
- `OI_WAVELENGTH_AIR_M`, `SPEED_OF_LIGHT_MS`, and `ETALON_GAP_M` are present

Report Yes/No for each required constant.

**TASK C — Create new module**

Create `src/fpi/m06_airglow_inversion_2026_05_05.py` by copying the
current implementation and applying these changes:

1. Update module docstring (template below).
2. Remove any import or use of `FOCAL_LENGTH_M`.
3. Replace any use of `OI_WAVELENGTH_M` with `OI_WAVELENGTH_AIR_M` throughout
   (the constant was renamed in H01 2026-05-05 to be explicit about air vs vacuum).
4. Update all imports of `CalibrationResult` to point to the new H05 dated module.
5. Update the import of `airy_modified` to point to the new H01 dated module.
6. In `_airglow_model()`, confirm `cal.alpha` is passed directly to
   `airy_modified()` — no focal length arithmetic.
7. In `_lambda_c_scan()` and quality checks: replace any hardcoded FSR constant
   with the inline computation:
   `FSR_OI_M = OI_WAVELENGTH_AIR_M**2 / (2.0 * ETALON_GAP_M)`
8. All other functions, fits, and tests are **unchanged**.

Module docstring:
```python
"""
Module:      m06_airglow_inversion_2026_05_05.py
Spec:        docs/specs/H06_airglow_inversion_2026-05-05.md
Author:      Claude Code
Generated:   2026-05-05
Last tested: 2026-05-05
Project:     WindCube FPI Pipeline — NCAR/HAO
Repo:        soc_sewell

Changes from prior version:
  - FOCAL_LENGTH_M removed from imports; not used in H06.
  - OI_WAVELENGTH_M renamed to OI_WAVELENGTH_AIR_M throughout (H01 naming).
  - FSR_OI_M computed inline from ETALON_GAP_M; no separate constant.
  - CalibrationResult and airy_modified imports updated to 2026_05_05 modules.
  - No algorithmic changes.
"""
```

**TASK D — Update conftest.py**

Add or update the `synthetic_cal_result` and `synthetic_airglow_profile`
fixtures in `tests/conftest.py` per Section 11. Use
`_build_tolansky_stub(params)` — not `TolanskyPipeline.run()` — to construct
the CalibrationResult for the session-scoped fixture.

**TASK E — Run module tests**

```bash
pytest tests/test_h06_airglow_inversion.py -v --tb=short
```

All 8 tests must pass. Stop and report if any fail.

**TASK F — Full test suite**

```bash
pytest tests/ -v --tb=short
```

No regressions permitted. Report any failures.

**TASK G — Archive old module and commit**

1. Archive the old spec:
   ```bash
   git mv docs/specs/H06_airglow_inversion_2026-04-06.md \
           docs/specs/archive/H06_airglow_inversion_2026-04-06.md
   ```
2. Copy this spec to `docs/specs/H06_airglow_inversion_2026-05-05.md`.
3. Commit:
   ```
   refactor(H06): remove FOCAL_LENGTH_M; OI_WAVELENGTH_AIR_M naming; H01/H05 refs updated
   Implements: H06_airglow_inversion_2026-05-05.md
   No algorithmic changes. 8/8 tests pass.
   ```

### Report format (paste back to Claude.ai)

```
TASK A — Prior tests
  Full suite: N/N pass

TASK B — Constants check
  All H06 §4 constants present: Yes / No (list any missing)
  FOCAL_LENGTH_M not imported: Yes / No
  OI_WAVELENGTH_AIR_M used (not OI_WAVELENGTH_M): Yes / No

TASK C — Module created
  Source file: src/fpi/m06_airglow_inversion_2026_05_05.py
  FOCAL_LENGTH_M removed: Yes / No
  OI_WAVELENGTH_AIR_M used throughout: Yes / No
  FSR_OI_M computed inline: Yes / No
  H01/H05 import paths updated: Yes / No
  Algorithmic changes: None / [list any]

TASK D — conftest.py updated
  synthetic_cal_result fixture: present / missing
  synthetic_airglow_profile fixture: present / missing

TASK E — Module tests
  Result: N/8 pass
  Failures: [list]

TASK F — Full suite
  Result: N/N pass
  Unexpected failures: [list]

TASK G — Commit hash: [hash]
```
