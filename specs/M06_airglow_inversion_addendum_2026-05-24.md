# M06 Addendum — Temperature Retrieval (`use_temperature` flag)

**Spec file:** `M06_airglow_inversion_addendum_2026-05-24.md`  
**Amends:** `m06_airglow_inversion_2026_05_05.py`  
**New library file:** `m06_airglow_inversion_2026-05-24.py`  
**Status:** Draft  
**Date:** 2026-05-24  
**Author:** SOC / S. Sewell  
**Required by:** MC01 (`MC01_fpi_mc_engine_2026-05-24.md`)  
**Depends on:** M03 addendum (`M03_airglow_synthesis_addendum_2026-05-24.md`) for  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;`delta_lambda_from_temperature()` (imported, not reimplemented)

---

## 1. Motivation

The existing M06 inversion fits **3 free parameters** (`λ_c`, `Y_line`, `B_sci`)
using the delta-function forward model. Temperature is not retrieved. This is
correct for operational WindCube science (the OI 630 nm line is unresolved at
WindCube's etalon resolution) and must remain the default behaviour.

However, the Harding (2014) Monte Carlo simulations require recovering both wind
velocity *and* temperature from thermally broadened synthetic airglow fringes, in
order to test for biases in both quantities across the expected parameter space
(Harding §4.B, §4.C). This addendum adds a `use_temperature` flag to
`fit_airglow_fringe()` that activates a **4-parameter fit** including `Δλ` as a
free parameter, from which temperature is recovered via Harding Eq. 12 (inverted).

When `use_temperature=False` (default), the function is **strictly unchanged** from
the existing implementation — same code path, same outputs, no regression.

---

## 2. Changes to `AirglowFitResult`

Add four new optional fields at the end of the dataclass. All default to `None`
so that existing code consuming `AirglowFitResult` is unaffected when
`use_temperature=False`.

```python
@dataclass
class AirglowFitResult:
    # ... all existing fields unchanged ...

    # Temperature retrieval (None when use_temperature=False)
    T_est_K:          Optional[float] = None
    sigma_T_K:        Optional[float] = None
    two_sigma_T_K:    Optional[float] = None
    delta_lambda_m:   Optional[float] = None   # fitted Δλ (diagnostic)
```

The `n_params_free` field must be set to `4` (not `3`) when `use_temperature=True`.

---

## 3. New flag bit in `AirglowFitFlags`

```python
class AirglowFitFlags:
    # ... existing bits unchanged ...
    DELTA_LAMBDA_AT_BOUND = 0x200   # Δλ hit its bound during LM fit
```

---

## 4. Changes to `fit_airglow_fringe()`

### 4.1 New parameter

```python
def fit_airglow_fringe(
    profile,
    cal,
    n_fine: int = 500,
    use_temperature: bool = False,    # NEW
) -> AirglowFitResult:
```

| New parameter | Type | Default | Description |
|---------------|------|---------|-------------|
| `use_temperature` | bool | `False` | If `True`, add `Δλ` as a 4th free parameter and retrieve temperature. If `False`, behaviour identical to existing implementation. |

### 4.2 New forward model helper: `_airglow_model_thermal()`

This replaces `_airglow_model_fine()` only on the temperature-enabled path.
The existing `_airglow_model_fine()` is **not modified**.

```python
def _airglow_model_thermal(
    r_fine: np.ndarray,
    lambda_c_m: float,
    delta_lambda_m: float,
    Y_line: float,
    B_sci: float,
    cal,
    L_invert: int = 101,
    n_fsr: float = 5.0,
) -> np.ndarray:
    """
    Thermally broadened airglow fringe model on a fine r grid.

    Implements the Harding Eq. 15 discrete spectral integral:
        S(r_i) ≈ Σ_j Ã(r_i, λ_j) Y(λ_j) Δλ + B

    where Y(λ) is the Gaussian source from M03._gaussian_source_spectrum().
    Uses L_invert=101 spectral bins (Harding §3 anti-inverse-crime rule:
    invert at L=101, even though synthesis used L=300).

    Parameters
    ----------
    r_fine         : fine uniform-r grid, shape (n_fine,)
    lambda_c_m     : Doppler-shifted line centre, metres
    delta_lambda_m : Doppler line half-width (1/e), metres
    Y_line         : line amplitude scale factor
    B_sci          : CCD bias, ADU
    cal            : CalibrationResult — all instrument parameters fixed
    L_invert       : spectral bins for inversion (default 101, per Harding §3)
    n_fsr          : number of FSRs to span in spectral grid (default 5.0)

    Returns
    -------
    model : ndarray shape (n_fine,)
    """
    from m03_airglow_synthesis_2026_05_24 import _gaussian_source_spectrum
    from windcube.constants import OI_WAVELENGTH_AIR_M

    r_max  = float(r_fine[-1])
    FSR_m  = OI_WAVELENGTH_AIR_M ** 2 / (2.0 * cal.t_m)
    lam_lo = lambda_c_m - (n_fsr / 2.0) * FSR_m
    lam_hi = lambda_c_m + (n_fsr / 2.0) * FSR_m
    lam_grid = np.linspace(lam_lo, lam_hi, L_invert)
    d_lam    = lam_grid[1] - lam_grid[0]

    Y_lam = _gaussian_source_spectrum(
        lam_grid, lambda_c_m, delta_lambda_m, Y_line, Y_bg=0.0
    )

    model = np.zeros_like(r_fine)
    for j, lam_j in enumerate(lam_grid):
        airy_j = airy_modified(
            r_fine, lam_j,
            t=cal.t_m, R_refl=cal.R_refl, alpha=cal.alpha, n=1.0,
            r_max=r_max,
            I0=cal.I0, I1=cal.I1, I2=cal.I2,
            sigma0=cal.sigma0, sigma1=cal.sigma1, sigma2=cal.sigma2,
        )
        model += airy_j * Y_lam[j] * d_lam
    return model + B_sci
```

> **Note:** `Y_bg` is excluded from the thermal model here (set to 0.0). The
> `B_sci` parameter absorbs the background floor in the inversion. This matches
> Harding's parameterisation where `B` is the bias/background combined term.

### 4.3 New LM helper: `_run_airglow_lm_thermal()`

A new LM fitting function operating on 4 parameters `[λ_c, Δλ, Y_line, B_sci]`.
The existing `_run_airglow_lm()` is **not modified**.

```python
def _run_airglow_lm_thermal(
    r_good: np.ndarray,
    profile_good: np.ndarray,
    sigma_good: np.ndarray,
    r_max: float,
    cal,
    lambda_c_init_m: float,
    delta_lambda_init_m: float,
    Y_line_init: float,
    B_sci_init: float,
    n_fine: int = 500,
    L_invert: int = 101,
):
    """
    LM fit over {lambda_c_m, delta_lambda_m, Y_line, B_sci}.

    Bounds:
      lambda_c_m     : OI_WAVELENGTH_AIR_M ± 1.5 × FSR_OI_M  (soft)
      delta_lambda_m : [delta_lambda_from_temperature(100K),
                        delta_lambda_from_temperature(3000K)]  (soft)
      Y_line         : [0, inf)   (soft lower only)
      B_sci          : [0, min(profile)*1.5]  (soft)
    """
```

Soft-bound penalty residuals applied identically to `_run_airglow_lm()`.
The Δλ bounds correspond to T ∈ [100 K, 3000 K] computed via
`delta_lambda_from_temperature()` (imported from M03 addendum).

### 4.4 New Jacobian helper: `_compute_jacobian_thermal()`

Extends `_compute_jacobian_analytical()` to 4 parameters. The `Δλ` column
uses a finite-difference step of `delta_lambda_init_m * 0.01` (1% of the
current Δλ estimate). The existing `_compute_jacobian_analytical()` is
**not modified**.

### 4.5 Temperature conversion: `temperature_from_delta_lambda()`

Public function — inverse of M03's `delta_lambda_from_temperature()`.

```python
def temperature_from_delta_lambda(
    delta_lambda_m: float,
    lambda0_m: float = None,
) -> float:
    """
    Harding Eq. 12 inverted: temperature from fitted line half-width.

        T = m * c² * (Δλ / λ₀)² / k

    Parameters
    ----------
    delta_lambda_m : fitted Doppler line half-width, metres
    lambda0_m      : rest wavelength, metres.
                     Default: OI_WAVELENGTH_AIR_M from windcube/constants.py.

    Returns
    -------
    T_K : float, Kelvin
    """
    from windcube.constants import (
        OI_WAVELENGTH_AIR_M, SPEED_OF_LIGHT_MS,
        BOLTZMANN_J_PER_K, OXYGEN_MASS_KG,
    )
    if lambda0_m is None:
        lambda0_m = OI_WAVELENGTH_AIR_M
    ratio = (delta_lambda_m * SPEED_OF_LIGHT_MS) / lambda0_m
    return OXYGEN_MASS_KG * ratio ** 2 / BOLTZMANN_J_PER_K
```

### 4.6 Modified `fit_airglow_fringe()` branching logic

The existing `fit_airglow_fringe()` function body is **unchanged** for the
`use_temperature=False` path. After the existing step 4 (Doppler wind and phase),
insert the following branch:

```python
# ---- Step 5 (new): Temperature retrieval if requested ----
T_est_K = None
sigma_T_K = None
two_sigma_T_K = None
delta_lambda_fit_m = None

if use_temperature:
    # Initial guess for Δλ from T=1000 K (Harding §3)
    from m03_airglow_synthesis_2026_05_24 import delta_lambda_from_temperature
    delta_lambda_init = delta_lambda_from_temperature(1000.0)

    lm_thermal = _run_airglow_lm_thermal(
        r_good, profile_good, sigma_good, r_max, cal,
        lambda_c_init_m      = lambda_c_m,   # seed from 3-param result
        delta_lambda_init_m  = delta_lambda_init,
        Y_line_init          = Y_line,
        B_sci_init           = B_sci,
        n_fine               = n_fine,
        L_invert             = 101,
    )

    lambda_c_m     = float(lm_thermal.x[0])   # refine λ_c with 4-param fit
    delta_lambda_fit_m = float(lm_thermal.x[1])
    Y_line         = float(lm_thermal.x[2])
    B_sci          = float(lm_thermal.x[3])

    # Recompute chi2 and uncertainties for 4-param fit
    # ... (analogous to existing steps 2–3, using _airglow_model_thermal
    #      and _compute_jacobian_thermal) ...

    T_est_K = temperature_from_delta_lambda(delta_lambda_fit_m)
    # sigma_T_K via error propagation: σ_T = dT/dΔλ × σ_Δλ
    # dT/dΔλ = 2 × m × c² × Δλ / (λ₀² × k)
    dT_dDlam = (2.0 * OXYGEN_MASS_KG * SPEED_OF_LIGHT_MS**2
                * delta_lambda_fit_m
                / (OI_WAVELENGTH_AIR_M**2 * BOLTZMANN_J_PER_K))
    sigma_T_K     = dT_dDlam * sigmas_thermal[1]   # sigmas_thermal[1] = σ_Δλ
    two_sigma_T_K = 2.0 * sigma_T_K

    # DELTA_LAMBDA_AT_BOUND flag
    dlam_lo = delta_lambda_from_temperature(100.0)
    dlam_hi = delta_lambda_from_temperature(3000.0)
    if (abs(delta_lambda_fit_m - dlam_lo) < 1e-18 or
            abs(delta_lambda_fit_m - dlam_hi) < 1e-18):
        result_flags |= AirglowFitFlags.DELTA_LAMBDA_AT_BOUND

    n_params_free_out = 4
else:
    n_params_free_out = 3   # existing behaviour
```

**Seeding strategy:** The 4-parameter LM is seeded from the converged 3-parameter
result (`lambda_c_m`, `Y_line`, `B_sci` from the existing fit) plus
`delta_lambda_init = delta_lambda_from_temperature(1000 K)`. This two-stage
approach mirrors Harding §3's staged optimisation and avoids cold-start
divergence of Δλ.

### 4.7 Updated `AirglowFitResult` construction

In the existing `return AirglowFitResult(...)` block, add:

```python
    # Temperature (None when use_temperature=False)
    T_est_K          = T_est_K,
    sigma_T_K        = sigma_T_K,
    two_sigma_T_K    = two_sigma_T_K,
    delta_lambda_m   = delta_lambda_fit_m,
    # Updated free-parameter count
    n_params_free    = n_params_free_out,
```

---

## 5. Acceptance tests

These extend the existing M06 test suite; do not remove existing tests.

| Test ID | Description | Pass criterion |
|---------|-------------|----------------|
| T-M06-T01 | `use_temperature=False` produces bit-identical output to pre-addendum | All `AirglowFitResult` fields (except the 4 new optional ones) unchanged |
| T-M06-T02 | `use_temperature=True` on synthetic T=800K profile | `\|T_est_K − 800\| < 50 K`, `converged=True` |
| T-M06-T03 | `temperature_from_delta_lambda(delta_lambda_from_temperature(800.0))` | Returns 800.0 ± 0.01 K |
| T-M06-T04 | `n_params_free = 4` when `use_temperature=True` | Field value = 4 |
| T-M06-T05 | `n_params_free = 3` when `use_temperature=False` | Field value = 3 (unchanged) |
| T-M06-T06 | `T_est_K` is `None` when `use_temperature=False` | Field is `None` |
| T-M06-T07 | `DELTA_LAMBDA_AT_BOUND` flag set on out-of-range Δλ | Flag present in `quality_flags` |
| T-M06-T08 | Error propagation: `sigma_T_K` reasonable | `sigma_T_K ∈ [1, 100] K` for SNR=5, T=800K |

---

## 6. Change log

| Version | Date | Summary |
|---------|------|---------|
| Addendum 1.0 | 2026-05-24 | Add `use_temperature` flag, 4-param LM fit, `T_est_K` output |
