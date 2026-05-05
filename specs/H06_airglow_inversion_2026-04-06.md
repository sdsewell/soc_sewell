# S15 — M06 Airglow Fringe Inversion

**Spec ID:** S15
**Spec file:** `docs/specs/S15_m06_airglow_inversion_2026-04-06.md`
**Project:** WindCube FPI Science Operations Center Pipeline
**Institution:** NCAR / High Altitude Observatory (HAO)
**Status:** Authoritative
**Depends on:** S01, S02, S03, S04, S09 (M01), S11 (M04), S12 (M03),
S14 (M05 — provides CalibrationResult)
**Used by:** S16 (M07 — wind retrieval receives AirglowFitResult),
S17 (INT02), S18 (INT03)
**Last updated:** 2026-04-06
**Created/Modified by:** Claude AI

---

## 1. Purpose

M06 inverts the 1D OI 630 nm airglow `FringeProfile` from M03 to recover
the Doppler-shifted line centre `λ_c`. All ten instrument parameters are
fixed at the values determined by M05 from the neon calibration frame.
The only free parameters are the three airglow source parameters.

The recovered `λ_c` is converted to a line-of-sight wind speed `v_rel`:

```
v_rel = c × (λ_c − OI_WAVELENGTH_M) / OI_WAVELENGTH_M
```

This `v_rel` is the primary output of M06 and the primary input to M07.

**What M06 is not.** M06 does not decompose v_rel into zonal and meridional
components — that is M07's job. M06 does not characterise the etalon — that
is M05's job. M06 does not retrieve temperature — that is not a WindCube
science goal (see Section 2.1).

---

## 2. Key design decisions

### 2.1 Delta-function source model — do not reopen this decision

WindCube's science goals are wind speed and direction. Temperature retrieval
is explicitly **not** in the science requirements (STM v1). The OI 630 nm
emission line is therefore modelled as a **spectral delta function** — a
single infinitely narrow line at `λ_c`. This is equivalent to setting
`Δλ → 0` in the Harding Gaussian model.

**Consequence:** M06 has **three free parameters** `{Y_line, B, λ_c}`,
not Harding's five `{B, Y_bg, Y_line, λ_c, Δλ}`. Temperature broadening
and background sky emission are not fitted. This simplification:
- Reduces parameter degeneracy (no Δλ–λ_c correlation)
- Halves the number of free parameters compared to the full Harding model
- Is consistent with M04 (airglow synthesis) which also uses a delta
  function source when `temperature_K=None` is passed

**Implication for M04 consistency:** When running the synthetic round-trip
test (T5), M04 must generate the airglow image with the delta-function model
(i.e., pass `temperature_K=None` or set `Δλ → 0`). If M04 generates with
a thermally broadened line and M06 inverts with a delta function, there will
be a systematic model mismatch. For the operational pipeline this is
acceptable (the thermal broadening is small compared to the fringe width),
but the round-trip test should be self-consistent.

### 2.2 Instrument parameters are fixed — not refitted

All ten CalibrationResult parameters (`t_m`, `R_refl`, `alpha`, `I0`, `I1`,
`I2`, `sigma0`, `sigma1`, `sigma2`, `B_cal`) are received from M05 and used
to construct the forward model without refitting. The `B` in M06
(`B_sci`) is a separate free parameter — the science frame bias may differ
from the calibration frame bias due to dark current variation.

### 2.3 Fine-grid forward model — inherited from M05 experience

M06 uses the same fine-grid forward model strategy established during M05
implementation: evaluate the Airy model on a dense uniform-r grid then
bin-average to match the M03 r² bins. This prevents the grid-spacing bias
that inflates χ² and corrupts parameter uncertainties.

### 2.4 Soft-bound penalty residuals — inherited from M05

LM (`scipy.optimize.least_squares(method='lm')`) does not enforce bounds.
Penalty residuals are added outside the effective bounds for each parameter,
exactly as in M05. See Section 7.3.

### 2.5 Lambda_c initialisation — brute-force scan over one FSR

The initial guess for `λ_c` is the most critical initialisation in M06.
A wrong initial λ_c by more than ~FSR/2 will cause the fit to converge to
a false minimum. The brute-force scan over one FSR finds the correct period
before any LM step begins.

---

## 3. Physical background

### 3.1 Forward model for the airglow fringe

With a delta-function source at `λ_c`, the forward model is:

```
s(r) = Y_line · Ã(r; λ_c) + B_sci
```

where `Ã(r; λ_c)` is the modified Airy function from M01, evaluated at the
fitted `λ_c` using the fixed CalibrationResult instrument parameters.

Note: the `I0`, `I1`, `I2` intensity envelope from CalibrationResult is
absorbed into `Y_line` — M06 fits a single scalar intensity multiplier
rather than three envelope coefficients. The envelope shape is fixed; only
its overall scale is free. This keeps the model linear in `Y_line` and
reduces degeneracy.

Explicitly:

```python
def _airglow_model(r_fine, lambda_c_m, Y_line, B_sci, cal):
    """
    Delta-function airglow fringe at lambda_c_m.

    r_fine  : fine uniform-r grid, pixels
    lambda_c_m : free parameter — Doppler-shifted line centre
    Y_line  : free parameter — line intensity scale factor
    B_sci   : free parameter — science frame CCD bias
    cal     : CalibrationResult — all 10 instrument parameters fixed

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
residuals (identical pattern to M05's `_neon_model`).

### 3.2 Doppler shift — λ_c to v_rel

```
v_rel = c × (λ_c − λ₀) / λ₀        (Harding Eq. 11, inverted)

where:
  λ₀ = OI_WAVELENGTH_M = 630.0304e-9 m  (from S03)
  c  = SPEED_OF_LIGHT_MS                (from S03)
```

Positive `v_rel` = recession (redshift), consistent with M04's convention.

### 3.3 Uncertainty propagation

From the LM covariance matrix, `sigma_lambda_c_m` is the 1σ uncertainty
on `λ_c`. The wind uncertainty propagates linearly:

```
sigma_v_rel = c × sigma_lambda_c_m / OI_WAVELENGTH_M
```

**Numerical check:** For the STM wind budget of 9.8 m/s:
```
sigma_lambda_c required = 9.8 × 630.0304e-9 / 299792458 ≈ 2.06e-14 m (0.021 pm)
```

---

## 4. Physical constants from S03

```python
from src.constants import (
    OI_WAVELENGTH_M,       # 630.0304e-9 m — OI rest wavelength (air)
    SPEED_OF_LIGHT_MS,     # 299_792_458.0 m/s
    ETALON_FSR_OI_M,       # ≈ 9.922e-15 m — FSR at OI 630 nm for λ_c scan
)
```

---

## 5. Free parameters and bounds

Three free parameters only:

| Parameter | Symbol | Physical meaning | Initial estimate | Bounds |
|-----------|--------|-----------------|-----------------|--------|
| Line centre | `lambda_c_m` | Doppler-shifted wavelength | Brute-force scan result | `OI_WAVELENGTH_M ± 1.5 × ETALON_FSR_OI_M` |
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
    Output of M06 airglow inversion.
    Passed to M07 (wind retrieval) for LOS-to-vector decomposition.
    """
    # Primary output — line centre and wind
    lambda_c_m:              float   # fitted line centre, metres
    sigma_lambda_c_m:        float   # 1σ uncertainty, metres
    two_sigma_lambda_c_m:    float   # exactly 2 × sigma_lambda_c_m  (S04)

    v_rel_ms:                float   # LOS wind speed, m/s
    sigma_v_rel_ms:          float   # 1σ uncertainty, m/s
    two_sigma_v_rel_ms:      float   # exactly 2 × sigma_v_rel_ms    (S04)

    # Other fitted parameters (for diagnostics and physical truth checks)
    Y_line:                  float   # fitted line intensity scale
    sigma_Y_line:            float
    two_sigma_Y_line:        float

    B_sci:                   float   # fitted science frame bias, ADU
    sigma_B_sci:             float
    two_sigma_B_sci:         float

    # Fit quality
    chi2_reduced:            float   # must be in [0.5, 3.0] for good fit
    n_bins_used:             int     # non-masked bins used in fit
    n_params_free:           int     # always 3 for M06
    converged:               bool
    quality_flags:           int     # AirglowFitFlags bitmask

    # Phase relationship to calibration (diagnostic)
    epsilon_sci:             float   # (2 × lambda_c_m / OI_WAVELENGTH_M) mod 1
                                     # fractional order at science wavelength
    delta_epsilon:           float   # epsilon_sci − epsilon_cal (from CalibrationResult)
                                     # this is the fractional phase shift from wind

    # Input traceability
    calibration_t_m:         float   # t_m from CalibrationResult used
    calibration_epsilon_cal: float   # epsilon_cal from CalibrationResult used

    # LM scan diagnostics
    lambda_c_scan_init_m:    float   # λ_c at start of brute-force scan
    lambda_c_lm_init_m:      float   # λ_c passed to LM after scan


class AirglowFitFlags:
    """Bitmask quality flags for AirglowFitResult. Uses bits 4+ per S04."""
    GOOD                  = 0x00
    FIT_FAILED            = 0x01   # global S04 flag — LM did not converge
    CHI2_HIGH             = 0x02   # global S04 flag — chi2 > 3.0
    CHI2_VERY_HIGH        = 0x04   # global S04 flag — chi2 > 10.0
    CHI2_LOW              = 0x08   # global S04 flag — chi2 < 0.5
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
        From M05 fit_calibration_fringe(). All 10 instrument parameters
        are used as fixed constants. If cal.quality_flags is non-GOOD,
        set CAL_QUALITY_DEGRADED flag in output but do not abort.
    n_fine : int
        Number of points in the fine uniform-r grid for forward model
        evaluation. Default 500 (same as M05). Do not reduce below 200.

    Returns
    -------
    AirglowFitResult
        All fitted parameters with sigma and two_sigma, chi2_reduced,
        and quality flags.

    Raises
    ------
    ValueError
        If profile.quality_flags & CENTRE_FAILED.
        If fewer than 10 unmasked bins remain.
    RuntimeError
        If any parameter stderr is None after fit (pseudoinverse fallback
        exhausted — should not occur in practice).
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
    [OI_WAVELENGTH_M - ETALON_FSR_OI_M/2,
     OI_WAVELENGTH_M + ETALON_FSR_OI_M/2].

    At each candidate lambda_c:
    1. Build the model profile (fine grid → bin-average)
    2. Analytically solve for Y_line and B_sci via least-squares
       (linear in these parameters given fixed lambda_c)
    3. Compute chi2

    Returns (lambda_c_best, chi2_min).
    Sets SCAN_AMBIGUOUS flag if the second-best chi2 is within 10%
    of chi2_min (two plausible minima — possible fringe-order confusion).

    Notes
    -----
    The analytic solve for Y_line and B_sci at each scan point
    keeps the scan computationally cheap (no nested optimisation).
    With n_scan=200 this takes ~50 ms per frame.
    """
```

**Why this scan is necessary:** The OI 630 nm fringe pattern looks
almost identical at λ_c = λ₀ + k×FSR for any integer k. Without a scan,
the LM fit starting from λ₀ will converge to whichever FSR period happens
to have the nearest local minimum, which may not be the correct one.
The scan finds the correct period before LM refines within it.

**Wind speed corresponding to one FSR:**
```
v_FSR = c × FSR_OI / OI_WAVELENGTH_M
      = 299792458 × 9.922e-15 / 630.0304e-9
      ≈ 4723 m/s
```
A wrong-period initial λ_c would produce a wind error of ~4723 m/s —
catastrophic and undetectable from χ² alone if the LM converges well.

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
    Applied per bin to prevent near-zero SEM bins from dominating chi².
    (Same floor as M05 — see M05 implementation notes.)

    Soft-bound penalty residuals:
    One extra residual per free parameter, firing linearly outside the
    effective bounds defined in Section 5. Penalty weight = 1.0 (same
    order of magnitude as data residuals). This prevents lambda_c from
    drifting by more than 1.5 FSR from OI_WAVELENGTH_M and prevents
    Y_line or B_sci from going negative.

    Convergence tolerances: ftol=xtol=gtol=1e-12.
    Max function evaluations: 50_000.
    """
```

### 7.4 Step 3 — Covariance and uncertainties

Identical pattern to M05:

```python
J = result.jac          # Jacobian at solution, shape (n_residuals, 3)
# Remove penalty rows (last n_params rows) from J for covariance
J_data = J[:n_good, :]
s2 = chi2_unweighted / (n_good - 3)  # residual variance
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
v_rel_ms = SPEED_OF_LIGHT_MS * (lambda_c_m - OI_WAVELENGTH_M) / OI_WAVELENGTH_M
sigma_v_rel_ms = SPEED_OF_LIGHT_MS * sigma_lambda_c_m / OI_WAVELENGTH_M

epsilon_sci = (2.0 * lambda_c_m / OI_WAVELENGTH_M) % 1.0
delta_epsilon = epsilon_sci - cal.epsilon_cal
```

---

## 8. Quality checks

After the fit, check these conditions and set flags accordingly:

```python
# Chi2 range (global S04 flags)
if chi2_reduced > 10.0:
    flags |= AirglowFitFlags.CHI2_VERY_HIGH | AirglowFitFlags.CHI2_HIGH
elif chi2_reduced > 3.0:
    flags |= AirglowFitFlags.CHI2_HIGH
elif chi2_reduced < 0.5:
    flags |= AirglowFitFlags.CHI2_LOW

# lambda_c at bound (possible FSR-period confusion)
lambda_c_lo = OI_WAVELENGTH_M - 1.5 * ETALON_FSR_OI_M
lambda_c_hi = OI_WAVELENGTH_M + 1.5 * ETALON_FSR_OI_M
if abs(lambda_c_m - lambda_c_lo) < 1e-15 or abs(lambda_c_m - lambda_c_hi) < 1e-15:
    flags |= AirglowFitFlags.LAMBDA_C_AT_BOUND

# Low SNR
snr_estimate = (max(profile.profile[~profile.masked]) -
                min(profile.profile[~profile.masked])) / B_sci
if snr_estimate < 1.0:
    flags |= AirglowFitFlags.LOW_SNR
```

---

## 9. Verification tests

All tests in `tests/test_s15_m06_airglow_inversion.py`.

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
    from src.constants import SPEED_OF_LIGHT_MS, OI_WAVELENGTH_M
    result = fit_airglow_fringe(synthetic_airglow_profile, synthetic_cal_result)
    v_check = SPEED_OF_LIGHT_MS * (result.lambda_c_m - OI_WAVELENGTH_M) / OI_WAVELENGTH_M
    assert abs(result.v_rel_ms - v_check) < 1e-6, \
        f"v_rel = {result.v_rel_ms:.4f} m/s but Doppler formula gives {v_check:.4f}"
```

### T3 — Zero wind: v_rel near zero for v_rel_truth = 0

```python
def test_zero_wind_recovery(synthetic_cal_result):
    """A zero-wind airglow image must recover v_rel within ±5 m/s."""
    from src.fpi.m04_airglow_synthesis_2026_04_05 import synthesise_airglow_image
    from src.fpi.m03_annular_reduction_2026_04_06 import reduce_science_frame
    from src.fpi.m01_airy_forward_model_2026_04_05 import InstrumentParams
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
    assert abs(result.v_rel_ms) < 5.0, \
        f"Zero-wind recovery: |v_rel| = {abs(result.v_rel_ms):.2f} m/s > 5 m/s"
    assert result.converged
```

### T4 — Known wind: round-trip recovery within 20 m/s (noiseless)

```python
def test_known_wind_round_trip(synthetic_cal_result):
    """
    Noiseless round-trip: inject v_rel=200 m/s, recover to within 20 m/s.
    The 20 m/s tolerance accounts for grid discretisation in the synthetic
    image (same inverse-crime effect seen in M05 — acceptable for noiseless).
    """
    from src.fpi.m04_airglow_synthesis_2026_04_05 import synthesise_airglow_image
    from src.fpi.m03_annular_reduction_2026_04_06 import reduce_science_frame
    from src.fpi.m01_airy_forward_model_2026_04_05 import InstrumentParams
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
    error = abs(result.v_rel_ms - v_truth)
    assert error < 20.0, \
        f"v_rel error = {error:.2f} m/s > 20 m/s (noiseless round-trip)"
    assert result.converged
```

### T5 — Noisy round-trip: |error| < 3 × sigma_v

```python
def test_noisy_round_trip_uncertainty_calibrated(synthetic_cal_result):
    """
    Noisy round-trip at SNR ≈ 5: recovered v_rel within 3σ of truth.
    Also verify chi2_reduced ∈ [0.5, 3.0].
    """
    from src.fpi.m04_airglow_synthesis_2026_04_05 import synthesise_airglow_image
    from src.fpi.m03_annular_reduction_2026_04_06 import reduce_science_frame
    from src.fpi.m01_airy_forward_model_2026_04_05 import InstrumentParams
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
    error = abs(result.v_rel_ms - v_truth)
    assert error < 3.0 * result.sigma_v_rel_ms, \
        f"|error| = {error:.2f} m/s > 3σ = {3*result.sigma_v_rel_ms:.2f} m/s"
    assert 0.5 < result.chi2_reduced < 3.0, \
        f"chi2_reduced = {result.chi2_reduced:.3f} outside [0.5, 3.0]"
```

### T6 — Scan prevents FSR-period confusion

```python
def test_scan_prevents_fsr_confusion(synthetic_cal_result):
    """
    Inject v_rel = -300 m/s (blueshifted). Without scan, a naive LM
    starting from lambda_0 would converge to the wrong FSR period.
    With the scan, the correct lambda_c must be recovered.
    """
    from src.fpi.m04_airglow_synthesis_2026_04_05 import synthesise_airglow_image
    from src.fpi.m03_annular_reduction_2026_04_06 import reduce_science_frame
    from src.fpi.m01_airy_forward_model_2026_04_05 import InstrumentParams
    from src.constants import OI_WAVELENGTH_M, SPEED_OF_LIGHT_MS
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
        f"FSR confusion: recovered v_rel = {result.v_rel_ms:.0f} m/s, "
        f"truth = {v_truth:.0f} m/s, diff = {result.v_rel_ms - v_truth:.0f} m/s"
```

### T7 — CENTRE_FAILED profile raises ValueError

```python
def test_centre_failed_raises(synthetic_cal_result):
    """fit_airglow_fringe must raise ValueError for CENTRE_FAILED profile."""
    from src.fpi.m03_annular_reduction_2026_04_06 import FringeProfile, QualityFlags
    import pytest
    # Build a minimal bad profile
    bad_profile = _make_minimal_fringe_profile()
    bad_profile.quality_flags = QualityFlags.CENTRE_FAILED
    with pytest.raises(ValueError, match="CENTRE_FAILED"):
        fit_airglow_fringe(bad_profile, synthetic_cal_result)
```

### T8 — sigma_v consistent with STM wind budget

```python
def test_sigma_v_within_stm_budget(synthetic_cal_result):
    """
    At SNR ≈ 5 (dayside conditions), sigma_v_rel should be ≤ STM budget.
    STM: 9.8 m/s 1σ. Allow 2× margin for synthetic data.
    """
    from src.fpi.m04_airglow_synthesis_2026_04_05 import synthesise_airglow_image
    from src.fpi.m03_annular_reduction_2026_04_06 import reduce_science_frame
    from src.fpi.m01_airy_forward_model_2026_04_05 import InstrumentParams
    from src.constants import WIND_BIAS_BUDGET_MS
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
    assert result.sigma_v_rel_ms < 2.0 * WIND_BIAS_BUDGET_MS, \
        (f"sigma_v = {result.sigma_v_rel_ms:.2f} m/s > "
         f"2 × STM budget ({2*WIND_BIAS_BUDGET_MS:.1f} m/s)")
```

---

## 10. Expected numerical values

For `InstrumentParams()` defaults, noiseless synthetic image, SNR = 5
where noise is added:

| Quantity | Expected value | Notes |
|----------|----------------|-------|
| v_rel recovery (noiseless, v_truth=200) | within ±20 m/s | grid discretisation floor |
| v_rel recovery (noisy SNR=5, v_truth=150) | within 3σ | T5 |
| sigma_v_rel at SNR=5 | ≤ 2 × 9.8 m/s = 19.6 m/s | T8; STM budget |
| chi2_reduced (noisy) | 0.5–3.0 | T5 |
| sigma_lambda_c at 9.8 m/s budget | ≈ 2.06e-14 m (0.021 pm) | S04 derivation |
| FSR velocity equivalent | ≈ 4723 m/s | S03 VELOCITY_PER_FSR_MS |
| Max storm wind (400 m/s) in FSR units | ≈ 0.085 FSR | negligible ambiguity risk |
| n_params_free | 3 | always for M06 |
| lambda_c scan points | 200 | n_scan default |

---

## 11. Conftest fixtures

Add these fixtures to `tests/conftest.py` so they are available to all
tests in `test_s15_m06_airglow_inversion.py`:

```python
import pytest
import numpy as np
from src.fpi.m01_airy_forward_model_2026_04_05 import InstrumentParams
from src.fpi.m02_calibration_synthesis_2026_04_05 import synthesise_calibration_image
from src.fpi.m03_annular_reduction_2026_04_06 import reduce_calibration_frame
from src.fpi.tolansky_2026_04_05 import TolanskyPipeline
from src.fpi.m05_calibration_inversion_2026_04_06 import (
    fit_calibration_fringe, FitConfig
)

@pytest.fixture(scope='session')
def synthetic_cal_result():
    """
    A CalibrationResult from a noiseless synthetic calibration image.
    Computed once per test session (slow — ~30s).

    Uses _build_tolansky_stub (same pattern as M05 T5) rather than
    TolanskyPipeline.run() to avoid amplitude-split reliability issues
    on synthetic data.
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
    from src.fpi.m04_airglow_synthesis_2026_04_05 import synthesise_airglow_image
    from src.fpi.m03_annular_reduction_2026_04_06 import reduce_science_frame
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
│   └── m06_airglow_inversion_2026_04_06.py
└── tests/
    ├── conftest.py          ← add synthetic_cal_result fixture here
    └── test_s15_m06_airglow_inversion.py
```

---

## 13. Instructions for Claude Code

1. Read this entire spec, S04, S12, S14 before writing any code.
2. Confirm all prior tests pass:
   ```bash
   pytest tests/ -v --tb=no -q
   ```
3. Implement `src/fpi/m06_airglow_inversion_2026_04_06.py` in this order:
   `AirglowFitFlags` → `AirglowFitResult` → `_airglow_model` →
   `_bin_average` → `_lambda_c_scan` → `_run_airglow_lm` →
   `_compute_uncertainties` → `fit_airglow_fringe`
4. The forward model (`_airglow_model`) must evaluate on a fine uniform-r
   grid then bin-average to M03 r² bin centres — **identical pattern to
   M05's `_neon_model`**. Do not evaluate directly on `profile.r_grid`.
5. Apply the sigma floor: `sigma_eff = max(sigma, 0.005 × median(profile))`
   per bin — same as M05.
6. Apply soft-bound penalty residuals for all three parameters — same
   pattern as M05. Penalty weight = 1.0.
7. `two_sigma_X = 2.0 × sigma_X` for every field — never independently
   computed. Set this immediately after computing each sigma.
8. `epsilon_sci = (2.0 × lambda_c_m / OI_WAVELENGTH_M) % 1.0`
9. Add `synthetic_cal_result` and `synthetic_airglow_profile` fixtures to
   `tests/conftest.py`. Use `_build_tolansky_stub(params)` — not
   `TolanskyPipeline.run()` — to construct the CalibrationResult for
   fixtures. Copy the `_build_tolansky_stub` pattern from M05 tests.
10. Run module tests:
    ```bash
    pytest tests/test_s15_m06_airglow_inversion.py -v
    ```
    All 8 must pass.
11. Run full suite:
    ```bash
    pytest tests/ -v
    ```
    No regressions.
12. Commit:
    ```
    feat(m06): implement airglow inversion, delta-function model, 8/8 tests pass
    Implements: S15_m06_airglow_inversion_2026-04-06.md
    ```

Module docstring header:
```python
"""
M06 — Airglow fringe inversion: recovers v_rel from OI 630 nm FringeProfile.

Spec:        docs/specs/S15_m06_airglow_inversion_2026-04-06.md
Spec date:   2026-04-06
Generated:   <today>
Tool:        Claude Code
Last tested: <today>  (8/8 tests pass)
Depends on:  src.constants, src.fpi.m01_*, src.fpi.m03_*, src.fpi.m05_*
"""
```
