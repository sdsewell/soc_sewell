# F01 — Full Airy Fit to Neon Calibration Image

**Spec ID:** F01  
**Title:** Full Airy Fit to Neon Calibration Image  
**Version:** v1  
**Date:** 2026-04-21  
**Author:** Claude AI / Scott Sewell  
**Repo:** `soc_sewell`  
**Spec file:** `docs/specs/F01_full_airy_fit_to_neon_image_2026-04-21.md`  
**Depends on:** Z01a (Tolansky analysis), M01, M03  
**Consumed by:** F02 (airglow inversion)

---

## 0. Pipeline context — 8-step calibration-to-wind chain

F01 implements **steps 1–4** of the calibration-to-wind pipeline.
F02 implements **steps 5–8** using the `CalibrationResult` output of F01.

| Step | Module | Description |
|------|--------|-------------|
| 1 | Z01a | Two-line neon exposure; annular-reduce calibration image → ring radii r²_fit for λ₆₄₀ and λ₆₃₈ |
| 2 | Z01a | Tolansky WLS: fit ring-order P vs r²_fit; recover fractional orders ε₆₄₀, ε₆₃₈; plate-scale α from slope |
| 3 | Z01a | Benoit two-line gap recovery (Vaughan Eq. 3.97): d = (N_Δ + ε_a − ε_b)·λa·λb / [2(λb − λa)] |
| **4** | **F01** | **Full modified-Airy fit to neon 1D fringe profile: fix d from step 3; free-fit R, α, I₀, I₁, I₂, σ₀, σ₁, σ₂, B → CalibrationResult** |
| 5 | F02 | Annular reduction of airglow frame (calls M03); FringeProfile S(r) |
| 6 | F02 | Brute-force χ² scan over ±½ FSR for initial λ_c |
| 7 | F02 | Levenberg–Marquardt inversion: free-fit λ_c, Y_line, B_sci; all 10 instrument params fixed from CalibrationResult |
| 8 | F02 | Doppler wind and 2σ uncertainty; AirglowFitResult → M07 |

Steps 1–3 (Tolansky) are **prerequisites**, not replacements, for F01.  
The Tolansky resolves the FSR-period integer ambiguity and pins `d` to ~70 nm accuracy from the
ICOS pre-load correction.  F01 then uses `d` as a fixed prior and fits the remaining 9 shape
parameters from the full neon fringe profile.

---

## 1. Purpose and scope

F01 fits the 10-parameter modified Airy function (Harding et al. 2014, Eq. 9) to the
annular-reduced 1D neon calibration fringe profile.  The sole output is a `CalibrationResult`
dataclass carrying all 10 instrument parameters with their statistical uncertainties.  This
object is the complete instrument-function description passed to F02 for airglow inversion.

F01 does **not** re-derive `d` from scratch.  It accepts `d` from the Tolansky result (Z01a
`epsilon_cal` output → `t_m = 20.0006e-3 m`) as a fixed parameter and fits around it.

---

## 2. Physical model

### 2.1 Forward model (neon calibration fringe)

Because the neon source is monochromatic (delta-function spectrum at λ_Ne), the Fredholm
integral (Harding Eq. 9) reduces to a direct evaluation of the modified Airy function:

```
S(r) = Ã(r; λ_Ne) + B
```

where `Ã(r; λ)` is the PSF-broadened Airy function from M01 `airy_modified()`.

The ideal Airy transmission is:

```
A(r; λ) = I(r) / [1 + F · sin²(π · 2nd·cos(θ(r)) / λ)]
```

with:
- `θ(r) = arctan(α · r)`  — Harding Eq. 3
- `I(r) = I₀ · (1 + I₁·(r/r_max) + I₂·(r/r_max)²)`  — Harding Eq. 4
- `F = 4R/(1−R)²`  — finesse coefficient
- PSF broadening via shift-variant Gaussian with `σ(r) = σ₀ + σ₁·sin(πr/r_max) + σ₂·cos(πr/r_max)`  — Harding Eq. 8

### 2.2 Neon source wavelength

The primary calibration line is λ₆₄₀ = 640.2248 nm (Burns, Adams & Longwell 1950, IAU "S"
standard).  This is the bright line that dominates the annular-reduced profile; the secondary
line at 638.2991 nm is separated by ~2 FSR and does not appear in the single-line neon profile
used here (the two-line Tolansky analysis in Z01a has already exploited the beat between them
to recover `d`).

### 2.3 Free and fixed parameters

| Parameter | Symbol | Fixed/Free in F01 | Source if fixed |
|-----------|--------|--------------------|-----------------|
| Etalon gap | `d` (= `t_m`) | **Fixed** | Z01a Tolansky: 20.0006 mm |
| Index of refraction | `n` | **Fixed** | 1.0 (air gap) |
| Neon wavelength | `λ_Ne` | **Fixed** | 640.2248 nm |
| Plate reflectivity | `R` | Free | Init: 0.53 |
| Magnification | `α` | Free | Init: 1.6071 × 10⁻⁴ rad/px (Tolansky slope) |
| Avg intensity | `I₀` | Free | Init: median(S) |
| Linear vignetting | `I₁` | Free | Init: −0.1 |
| Quadratic vignetting | `I₂` | Free | Init: 0.005 |
| Avg PSF width | `σ₀` | Free | Init: 0.5 px |
| Sine PSF variation | `σ₁` | Free | Init: 0.0 |
| Cosine PSF variation | `σ₂` | Free | Init: 0.0 |
| CCD bias | `B` | Free | Init: percentile(S, 5) |

**Total free parameters: 9.**  
**Total fixed parameters (at fit time): 3** (`d`, `n`, `λ_Ne`).

---

## 3. Input

### 3.1 `FringeProfile` (from M03 `reduce_calibration_frame()`)

| Field | Type | Description |
|-------|------|-------------|
| `r_grid` | `ndarray (R,)` | Radial bin centres, pixels |
| `r2_grid` | `ndarray (R,)` | r² values, px² |
| `profile` | `ndarray (R,)` | Mean CCD counts per bin |
| `sigma_profile` | `ndarray (R,)` | Bootstrapped std/√N per bin |
| `masked` | `ndarray bool (R,)` | True = bad bin |
| `r_max_px` | `float` | Maximum usable radius, pixels |
| `quality_flags` | `int` | M03 QualityFlags bitmask |

R = 500 bins (equal-pixel-count annuli over the 256×256 CCD).

### 3.2 `TolanskyResult` (from Z01a)

| Field | Type | Description |
|-------|------|-------------|
| `t_m` | `float` | Authoritative gap, metres (20.0006e-3) |
| `alpha_rpx` | `float` | Plate scale from Tolansky slope, rad/px |
| `epsilon_640` | `float` | Fractional order at λ₆₄₀ (0.7735) |
| `epsilon_638` | `float` | Fractional order at λ₆₃₈ (0.2711) |
| `epsilon_cal` | `float` | Rest-frame fractional order at λ_OI (630.0 nm), extrapolated |

---

## 4. Algorithm

### 4.1 Validate inputs

- Raise `ValueError` if `profile.quality_flags & CENTRE_FAILED`.
- Build `good_mask`: unmasked bins with finite, positive `sigma_profile` and finite `profile`.
- Require ≥ 50 good bins; raise `ValueError` otherwise.
- Apply sigma floor: `sigma_good = max(sigma_raw, max(1.0, median(profile) × 0.005))`.

### 4.2 Construct initial parameter guesses

Use data-driven estimates wherever possible (following Harding §3):

| Parameter | Initial guess |
|-----------|--------------|
| `R` | 0.53 |
| `α` | `tolansky.alpha_rpx` (from Z01a WLS slope) |
| `I₀` | `median(profile_good)` |
| `I₁` | −0.1 |
| `I₂` | 0.005 |
| `σ₀` | 0.5 |
| `σ₁` | 0.0 |
| `σ₂` | 0.0 |
| `B` | `percentile(profile_good, 5)` |

### 4.3 Staged Levenberg–Marquardt optimization

Following Harding §3 (staged convergence strategy): never optimize all 9 free parameters
simultaneously from the first stage.

**Stage A** — fit `{I₀, B}` only (linear parameters; fast, robust).  
**Stage B** — fit `{I₀, I₁, I₂, B}` (intensity envelope).  
**Stage C** — fit `{R, α, I₀, B}` (dominant Airy shape).  
**Stage D** — fit all 9 free parameters simultaneously (final refinement).

Each stage uses `scipy.optimize.least_squares` (Levenberg–Marquardt method, `method='lm'`).

The residual vector at each stage is:

```
residuals[i] = (profile_good[i] − S_model(r_good[i])) / sigma_good[i]
```

where `S_model(r)` evaluates `airy_modified()` on a fine uniform-r grid of 500 points and
interpolates to `r_good` (same strategy as M06).

### 4.4 χ² quality check

After Stage D:

```
χ²_red = Σ residuals² / (n_good − 9)
```

Set flags:
- `CHI2_HIGH` if χ²_red > 3.0
- `CHI2_VERY_HIGH` if χ²_red > 10.0
- `CHI2_LOW` if χ²_red < 0.5

### 4.5 Uncertainty estimation

Compute parameter covariance from the Jacobian of the final residual vector:

```
cov = χ²_red · (Jᵀ J)⁻¹
sigma_params = sqrt(diag(cov))
```

Use `np.linalg.pinv` with `rcond=1e-10` if condition number of JᵀJ > 10¹⁴; set
`STDERR_NONE` flag if any stderr is non-finite.

---

## 5. Output — `CalibrationResult`

All 10 instrument parameters (9 fitted + `d` fixed) plus their 1σ uncertainties.

| Field | Type | Description |
|-------|------|-------------|
| `t_m` | `float` | Etalon gap, metres (passed through from Tolansky) |
| `R_refl` | `float` | Fitted plate reflectivity |
| `sigma_R_refl` | `float` | 1σ uncertainty |
| `alpha` | `float` | Fitted magnification, rad/px |
| `sigma_alpha` | `float` | 1σ uncertainty |
| `I0` | `float` | Fitted average intensity |
| `sigma_I0` | `float` | 1σ |
| `I1` | `float` | Fitted linear vignetting |
| `sigma_I1` | `float` | 1σ |
| `I2` | `float` | Fitted quadratic vignetting |
| `sigma_I2` | `float` | 1σ |
| `sigma0` | `float` | Fitted avg PSF width, px |
| `sigma_sigma0` | `float` | 1σ |
| `sigma1` | `float` | Fitted sine PSF variation, px |
| `sigma_sigma1` | `float` | 1σ |
| `sigma2` | `float` | Fitted cosine PSF variation, px |
| `sigma_sigma2` | `float` | 1σ |
| `B` | `float` | Fitted CCD bias, counts |
| `sigma_B` | `float` | 1σ |
| `epsilon_cal` | `float` | Rest-frame ε₀ at λ_OI from Z01a TolanskyResult |
| `chi2_reduced` | `float` | Reduced χ² of Stage D fit |
| `n_bins_used` | `int` | Number of good radial bins |
| `n_params_free` | `int` | Always 9 for F01 |
| `converged` | `bool` | True if Stage D LM converged |
| `quality_flags` | `int` | CalibrationFitFlags bitmask |
| `lambda_ne_m` | `float` | Neon wavelength used, metres (640.2248e-9) |
| `timestamp` | `float` | POSIX timestamp of calibration frame |

---

## 6. Quality flags — `CalibrationFitFlags`

```python
class CalibrationFitFlags:
    GOOD              = 0x000
    FIT_FAILED        = 0x001   # LM did not converge in Stage D
    CHI2_HIGH         = 0x002   # chi2_red > 3.0
    CHI2_VERY_HIGH    = 0x004   # chi2_red > 10.0
    CHI2_LOW          = 0x008   # chi2_red < 0.5
    STDERR_NONE       = 0x010   # any stderr is None / non-finite
    R_AT_BOUND        = 0x020   # R hit bounds [0.1, 0.95]
    ALPHA_AT_BOUND    = 0x040   # α hit bounds [0.5×, 2×] init
    FEW_BINS          = 0x080   # n_good < 100
```

---

## 7. Parameter bounds for LM

| Parameter | Lower bound | Upper bound |
|-----------|------------|-------------|
| `R` | 0.10 | 0.95 |
| `α` | 0.5 × init | 2.0 × init |
| `I₀` | 1.0 | — |
| `I₁` | −0.5 | 0.5 |
| `I₂` | −0.5 | 0.5 |
| `σ₀` | 0.01 | 5.0 |
| `σ₁` | −3.0 | 3.0 |
| `σ₂` | −3.0 | 3.0 |
| `B` | 0.0 | — |

---

## 8. Constants (from `windcube/constants.py`)

| Name | Value | Source |
|------|-------|--------|
| `NE_WAVELENGTH_1_M` | 640.2248e-9 m | Burns, Adams & Longwell (1950) |
| `D_25C_MM` | 20.0006 mm | ICOS build − Pat/Nir pre-load correction |
| `PLATE_SCALE_RPX` (Tolansky) | 1.6071e-4 rad/px | Z01a two-line Tolansky result |
| `R_REFL_FLATSAT` | 0.53 | FlatSat effective reflectivity (Z01 calibration) |
| `R_MAX_PX` | 110 | FlatSat/flight usable radius |

---

## 9. Relationship to existing modules

| Existing module | Role in F01 |
|----------------|------------|
| `M01.airy_modified()` | Forward model called at each LM iteration |
| `M01.InstrumentParams` | Default container; `t` default must equal `D_25C_MM` |
| `M03.reduce_calibration_frame()` | Produces the `FringeProfile` input |
| `Z01a` (Tolansky) | Provides `TolanskyResult.t_m` and initial `α` |
| `M05` (existing) | F01 supersedes M05 for this step; M05 may be retired or kept as legacy |

**Note on M01 default:** `InstrumentParams.t` must be updated to `20.0006e-3 m` (from the
current incorrect default of `20.008e-3 m`) before F01 is implemented.

---

## 10. Test matrix

| ID | Description | Pass criterion |
|----|-------------|---------------|
| T01 | Synthetic neon fringe with known params → recovery | All 9 free params within 2σ of truth |
| T02 | `d` fixed at Tolansky value; fit does not move it | `result.t_m == tolansky.t_m` exactly |
| T03 | CHI2_HIGH flag set when noise inflated | `quality_flags & CHI2_HIGH` non-zero |
| T04 | Staged convergence: Stage D χ² ≤ Stage C χ² | Monotone improvement |
| T05 | Real 120s neon calibration image (cal_image_120s.png) | χ²_red < 3.0, converged=True |
| T06 | `sigma0 = sigma1 = sigma2 = 0` → `airy_modified` == `airy_ideal` | Max abs diff < 1e-10 |
| T07 | `R_AT_BOUND` flag fires when R hits upper bound | Confirmed with low-SNR synthetic |
| T08 | `CalibrationResult` fields: `two_sigma_` == 2×`sigma_` | Exact equality check |

---

## 11. Open issues

- M05 retirement decision: once F01 tests pass at T05 level, open a PR to deprecate M05
  and redirect its callers to F01.
- Temporal interpolation of `CalibrationResult` between bracketing neon exposures
  (Harding §3, final paragraph) is a separate spec (candidate: F01b) and is out of scope here.

---

*End of F01 spec v1 — 2026-04-21*
