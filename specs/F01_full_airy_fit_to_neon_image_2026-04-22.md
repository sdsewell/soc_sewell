# F01 — Two-Line Airy Fit to Neon Calibration Profile

**Spec ID:** F01  
**Title:** Two-Line Airy Fit to Neon Calibration Profile  
**Version:** v3  
**Date:** 2026-04-22  
**Author:** Claude AI / Scott Sewell  
**Repo:** `soc_sewell`  
**Spec file:** `docs/specs/F01_full_airy_fit_to_neon_image_2026-04-22.md`  
**Depends on:** Z01a (two-line Tolansky — provides d, α, ε₆₄₀, ε₆₃₈), M01  
**Consumed by:** F02 (airglow Doppler inversion)

---

## Change log

| Version | Date | Summary |
|---------|------|---------|
| v1 | 2026-04-21 | Initial spec (single-line model) |
| v2 | 2026-04-21 | Locked decisions from implementation (n_scan, constant split) |
| **v3** | **2026-04-22** | **Rewritten: two-line forward model required; Y_B free parameter; scope restricted to Step 4b only (Tolansky seeding moved to Z01a spec)** |

---

## 0. Scope and pipeline position

F01 implements **Step 4b** of the calibration-to-wind pipeline only.

Steps 1–4a (image loading, dark subtraction, annular reduction, two-line Tolansky
analysis, and Benoit gap recovery) are performed by **Z01a** and are out of scope here.
F01 receives the Z01a outputs as inputs and performs no image-level operations.

| Step | Module | Description |
|------|--------|-------------|
| 1 | Z01a | Load calibration + dark images; dark subtract |
| 2 | Z01a | Find fringe centre; annular-reduce to 1D FringeProfile (500 bins) |
| 3 | Z01a | Two-line Tolansky WLS → ε₆₄₀, ε₆₃₈, α |
| 3b | Z01a | Benoit gap recovery → d (Vaughan Eq. 3.97) |
| **4b** | **F01** | **Two-line Airy fit to FringeProfile → CalibrationResult (10 params + Y_B)** |
| 5 | F02 | Annular reduction of airglow frame |
| 6 | F02 | Brute-force λ_c scan |
| 7 | F02 | LM inversion: λ_c, Y_line, B_sci free; all instrument params fixed |
| 8 | F02 | Doppler wind and 2σ uncertainty |

---

## 1. Purpose

F01 fits a **two-line modified Airy forward model** to the 1D annular-reduced neon
calibration fringe profile. It recovers the 10 instrument parameters of the WindCube FPI
plus the intensity ratio between the two neon lamp lines.

The output `CalibrationResult` is the complete instrument-function description consumed
by F02 for airglow Doppler inversion.

---

## 2. Motivation for the two-line model

### 2.1 Why the single-line model fails on real data

The v1/v2 spec stated that the 638.2991 nm secondary neon line "does not appear in the
single-line neon profile." **This is incorrect for the WindCube FPI and neon lamp
combination.** Real calibration images from the WindCube FlatSat show both neon lines
at clearly comparable amplitudes.

The 1D annular-reduced profile of the real 120s calibration exposure shows a clear
alternating tall/short peak pattern with amplitude ratio of approximately 0.58. This
is the direct signature of two Airy fringe families interleaved in r-space. The two
families have slightly different FSRs (10.247 pm vs 10.185 pm at d = 20.0006 mm), so
their rings land at slightly different radii and alternate through the profile.

Attempting to fit a single-line model to this data leaves ~1400 ADU of systematic
residual at every secondary peak, driving the LM fitter to compensate by
misestimating R, I₀, and σ₀. The two-line model is **required**, not optional.

### 2.2 The intensity ratio is a free parameter

The relative intensity Y_B of the 638.2991 nm line with respect to the 640.2248 nm
line depends on the specific lamp spectrum, the narrowband interference filter
transmittance at each wavelength, and the etalon reflectivity curve. It cannot be
assumed from first principles or from a published lamp spectrum. It must be fitted
from the calibration data. Y_B is therefore a **10th free parameter** of the F01 fit.

### 2.3 Phase relationship between the two line families

The two families are separated by 187.93 FSR_A (or equivalently 189.07 FSR_B) in
interference order at d = 20.0006 mm. This is not an integer, so the phase offset
between the two families at the optical axis (r = 0) is determined by ε₆₄₀ and ε₆₃₈.
Both ε values come from Z01a and enter F01 as fixed inputs — they set the relative
phase of the two Airy families throughout the profile. The fit does not re-derive them.

---

## 3. Physical model

### 3.1 Two-line forward model

The measured 1D profile S(r) is the incoherent superposition of two Airy patterns at
the two neon wavelengths, plus a CCD bias pedestal:

```
S(r) = Y_A · Ã(r; λ_A, d, R, α, I₀, I₁, I₂, σ₀, σ₁, σ₂)
     + Y_B · Ã(r; λ_B, d, R, α, I₀, I₁, I₂, σ₀, σ₁, σ₂)
     + B
```

where:
- `λ_A = 640.2248 nm`  — primary (brighter) neon line
- `λ_B = 638.2991 nm`  — secondary neon line
- `Ã(r; λ, ...)` is `M01.airy_modified()` — the PSF-broadened Airy function
- `Y_A` is the primary line intensity scale (free parameter)
- `Y_B` is the secondary line intensity (free parameter; units same as Y_A)
- `B` is the CCD bias pedestal (free parameter)

Both Airy functions share the same instrument parameters (d, R, α, I₀, I₁, I₂, σ₀,
σ₁, σ₂) because the etalon and imaging optics are wavelength-independent over the
1.93 nm separation between the two lines.

**Implementation note:** In the current `M01.airy_modified()` implementation, I₀ enters
as an absolute intensity envelope amplitude. The ratio Y_B/Y_A is therefore the
physically meaningful intensity ratio between the two lines. Expected value from
real data: Y_B/Y_A ≈ 0.55–0.65 (not 0.8 as previously hardcoded).

### 3.2 Airy function components (from M01)

The ideal Airy transmission at wavelength λ is:

```
A(r; λ) = I(r) / [1 + F · sin²(π · 2nd·cos(θ(r)) / λ)]
```

with:
- `θ(r) = arctan(α · r)`                                  (Harding Eq. 3)
- `I(r) = I₀ · (1 + I₁·(r/r_max) + I₂·(r/r_max)²)`       (Harding Eq. 4)
- `F = 4R/(1−R)²`                                          (finesse coefficient)
- PSF: `σ(r) = σ₀ + σ₁·sin(πr/r_max) + σ₂·cos(πr/r_max)` (Harding Eq. 8)

### 3.3 Phase anchoring from Tolansky ε values

The fractional fringe orders ε₆₄₀ and ε₆₃₈ from Z01a fix the absolute phase of each
Airy family at r = 0. Because d is fixed from the Benoit result, the phase of each
family at every radius is fully determined by d and α. The ε values enter implicitly
through the fixed d — no explicit ε term appears in the forward model.

---

## 4. Free and fixed parameters

### 4.1 Summary table

| Parameter | Symbol | Status | Initial value source |
|-----------|--------|--------|---------------------|
| Etalon gap | d | **FIXED** | Z01a Benoit recovery: 20.0006 mm |
| Index of refraction | n | **FIXED** | 1.0 (air gap) |
| Primary wavelength | λ_A | **FIXED** | 640.2248 nm (Burns et al. 1950) |
| Secondary wavelength | λ_B | **FIXED** | 638.2991 nm (Burns et al. 1950) |
| Plate reflectivity | R | Free | Contrast scan of data profile |
| Magnification | α | Free | Z01a Tolansky WLS slope |
| Intensity envelope (avg) | I₀ | Free | 80th percentile of bright peak amplitudes |
| Linear vignetting | I₁ | Free | −0.1 (Harding prior) |
| Quadratic vignetting | I₂ | Free | 0.005 (Harding prior) |
| PSF width (avg) | σ₀ | Free | FWHM analysis of profile peaks |
| PSF sine variation | σ₁ | Free | 0.0 |
| PSF cosine variation | σ₂ | Free | 0.0 |
| Primary intensity | Y_A | Free | Median of bright peak heights minus B |
| Secondary intensity | Y_B | Free | Y_A × (median short-peak-height / median tall-peak-height) |
| CCD bias | B | Free | 2nd percentile of profile |

**Total free parameters: 11** (R, α, I₀, I₁, I₂, σ₀, σ₁, σ₂, Y_A, Y_B, B).  
**Total fixed: 4** (d, n, λ_A, λ_B).

### 4.2 Separability of Y_A, Y_B, and I₀

Note that Y_A, Y_B, and I₀ are partially degenerate: multiplying all three by a common
factor leaves the model unchanged. This degeneracy is broken by fixing the functional
form of I(r) to be normalised to 1.0 at r = 0 (standard Harding convention). Concretely,
`M01.airy_modified` is evaluated at intensity I₀, and Y_A/Y_B scale the whole Airy
function (including its radial envelope). This means I₀ sets the absolute ADU level of
the primary line at the peak of the first fringe near r = 0, and Y_B/Y_A is the true
intensity ratio between the lines. In practice, initialise Y_A = 1.0 and absorb the
absolute scaling entirely into I₀; set Y_B_init = estimated_ratio (§5.2).

---

## 5. Algorithm

### 5.1 Validate inputs

- Raise `ValueError` if `profile.quality_flags & CENTRE_FAILED`.
- Build `good_mask`: unmasked bins with finite positive `sigma_profile` and finite `profile`.
- Require ≥ 50 good bins.
- Apply sigma floor: `sigma_good = max(sigma_raw, max(1.0, median(profile) × 0.005))`.
- If `cal.quality_flags != 0`: set `CAL_QUALITY_DEGRADED` in output flags.

### 5.2 Data-driven initial parameter estimates

All initial estimates are derived from the 1D profile and must be printed to the
console and displayed in the pre-fit diagnostic figure so the user can verify them
before the fit runs.

**B (CCD bias):**  
`B_init = percentile(profile, 2)`. The Airy minimum approaches B closely between
fringe peaks; the 2nd percentile is robust against isolated noisy low bins.

**R (plate reflectivity):**  
Estimate from fringe contrast. Separate the profile into bright peaks (family A) and
dim peaks (family B) by amplitude. Using the bright peaks only:
```
C_A = (I_max_A − I_min_local) / (I_max_A + I_min_local − 2·B)
F_est = 2·C_A / (1 − C_A)
R_init from: F = 4R/(1−R)²  →  x = 2(√(1+F)−1)/F, R = 1−x
```
Where I_min_local is the profile value at the trough immediately adjacent to each
bright peak. Do NOT use the global minimum of the profile, which may sit on a dim
(family B) fringe peak.

**Separating the two fringe families for initial estimation:**  
Find all peaks with `scipy.signal.find_peaks`. Sort by amplitude. The top half by
amplitude are family A (640 nm, brighter); the bottom half are family B (638 nm, dimmer).
This amplitude-based separation is the same method used in Z01a and is locked as the
correct discriminator.

**I₀:**  
`I0_init = median(bright_peak_heights − B)`. The bright peaks approximate the Airy
maximum I₀ + B (with minor envelope tilt from I₁, I₂ which are small).

**Y_A:**  
`Y_A_init = 1.0`. The absolute scale is absorbed into I₀.

**Y_B (intensity ratio):**  
`Y_B_init = median(dim_peak_heights − B) / median(bright_peak_heights − B)`  
Expected value: 0.55–0.65 for the WindCube FlatSat lamp and filter combination.

**σ₀:**  
Measure the FWHM of the median bright-peak fringe. The ideal Airy FWHM in r-space
is approximately `r_spacing / (π · sqrt(R) / (1−R))`. The excess FWHM above ideal
is attributed to PSF: `σ₀_init = sqrt(max(0, FWHM_meas² − FWHM_ideal²)) / (2√(2ln2))`,
clipped to `[0.1, 4.0]` px.

**α:**  
Use `tolansky.alpha_rpx` directly. Cross-check against r² peak spacing:
`alpha_check = sqrt(λ_A / (2·d·median(Δr²)))`. If disagreement exceeds 5%, print a
warning — it indicates a possible centre-finding error upstream.

**I₁, I₂, σ₁, σ₂:**  
`I₁_init = −0.1`, `I₂_init = 0.005`, `σ₁_init = 0.0`, `σ₂_init = 0.0` (Harding
§3 defaults; these will be refined by the staged LM).

### 5.3 Staged Levenberg–Marquardt optimisation

The two-line model has 11 free parameters. Fitting all simultaneously from the first
stage is numerically unstable. The staging strategy is extended from the 9-parameter
case to account for Y_A and Y_B:

| Stage | Free parameters | Rationale |
|-------|----------------|-----------|
| A | Y_A, Y_B, B | Linear intensity scales; fast analytic-like solution |
| B | Y_A, Y_B, I₀, B | Expand to full intensity envelope base |
| C | Y_A, Y_B, I₀, I₁, I₂, B | Full intensity envelope |
| D | R, α, Y_A, Y_B, I₀, B | Dominant Airy shape + both line scales |
| E | All 11 free params | Final global refinement |

Use `scipy.optimize.least_squares(method='trf')` for all stages (not `method='lm'`,
which does not support bounds). The residual vector at each stage is:

```
residuals[i] = (profile_good[i] − S_model(r_good[i])) / sigma_good[i]
```

where `S_model` evaluates the two-line model on a fine grid of 500 r-points and
interpolates to `r_good` bin centres.

**Critical:** Between each stage, clip all parameter values to their bounds before
passing to the next stage. This prevents a parameter that drifted outside its bounds
in one stage from destabilising the next.

### 5.4 Parameter bounds

| Parameter | Lower | Upper | Notes |
|-----------|-------|-------|-------|
| R | 0.10 | 0.95 | |
| α | 0.5 × α_init | 2.0 × α_init | Dynamic; flags ALPHA_AT_BOUND if hit |
| I₀ | 1.0 | ∞ | |
| I₁ | −0.5 | 0.5 | |
| I₂ | −0.5 | 0.5 | |
| σ₀ | 0.01 | 5.0 | |
| σ₁ | −3.0 | 3.0 | |
| σ₂ | −3.0 | 3.0 | |
| Y_A | 0.01 | ∞ | Must be positive |
| Y_B | 0.01 | ∞ | Must be positive; ratio Y_B/Y_A expected 0.4–0.9 |
| B | 0.0 | ∞ | |

### 5.5 χ² quality check

After Stage E:

```
χ²_red = Σ residuals² / (n_good − 11)
```

Flags: `CHI2_VERY_HIGH` if > 10.0, `CHI2_HIGH` if > 3.0, `CHI2_LOW` if < 0.5.

A well-converged two-line fit on the real 120s calibration data should achieve
χ²_red < 3.0. χ²_red values between 3 and 10 indicate residual structure that may
arise from lamp intensity fluctuations, optical distortions, or a hot-pixel artifact
in the profile.

### 5.6 Uncertainty estimation

Compute parameter covariance from the Stage E Jacobian:

```
cov = χ²_red · (JᵀJ)⁻¹
sigma_params = sqrt(diag(cov))
```

Use column-scaling before inverting JᵀJ to handle the ∼7 orders of magnitude
difference in column norms between α (∼10⁻⁴) and I₀ (∼10³). Specifically:
divide each Jacobian column j by its L2 norm `c_j`; invert `(J_scaled)ᵀ(J_scaled)`;
then back-scale: `sigma_j = sqrt(diag_j of scaled_cov) / c_j`.

Use `np.linalg.pinv(rcond=1e-10)` if `cond(JᵀJ) > 10¹⁴`; set `STDERR_NONE` flag.

---

## 6. Inputs

### 6.1 `FringeProfile` (from Z01a annular reduction)

| Field | Type | Description |
|-------|------|-------------|
| `r_grid` | `ndarray (R,)` | Radial bin centres, px |
| `r2_grid` | `ndarray (R,)` | r² values, px² |
| `profile` | `ndarray (R,)` | Mean CCD counts per bin (dark-subtracted) |
| `sigma_profile` | `ndarray (R,)` | SEM per bin |
| `masked` | `ndarray bool (R,)` | True = bad bin |
| `r_max_px` | `float` | Maximum usable radius, px |
| `quality_flags` | `int` | QualityFlags bitmask |

R = 500 bins.

### 6.2 `TolanskyResult` (from Z01a)

| Field | Type | Description |
|-------|------|-------------|
| `t_m` | `float` | Benoit gap, m (20.0006e-3) |
| `alpha_rpx` | `float` | Plate scale, rad/px |
| `epsilon_640` | `float` | Fractional order ε at 640.2248 nm |
| `epsilon_638` | `float` | Fractional order ε at 638.2991 nm |
| `epsilon_cal` | `float` | Extrapolated ε at OI 630 nm |

---

## 7. Output — `CalibrationResult`

| Field | Type | Description |
|-------|------|-------------|
| `t_m` | `float` | Etalon gap, m (passed through from Tolansky) |
| `R_refl` | `float` | Fitted reflectivity |
| `sigma_R_refl` | `float` | 1σ |
| `two_sigma_R_refl` | `float` | Exactly 2 × sigma_R_refl |
| `alpha` | `float` | Fitted magnification, rad/px |
| `sigma_alpha` | `float` | 1σ |
| `two_sigma_alpha` | `float` | Exactly 2 × sigma_alpha |
| `I0` | `float` | Fitted intensity envelope amplitude, ADU |
| `sigma_I0` | `float` | 1σ |
| `two_sigma_I0` | `float` | Exactly 2 × sigma_I0 |
| `I1` | `float` | Linear vignetting |
| `sigma_I1` | `float` | 1σ |
| `two_sigma_I1` | `float` | Exactly 2 × sigma_I1 |
| `I2` | `float` | Quadratic vignetting |
| `sigma_I2` | `float` | 1σ |
| `two_sigma_I2` | `float` | Exactly 2 × sigma_I2 |
| `sigma0` | `float` | PSF width avg, px |
| `sigma_sigma0` | `float` | 1σ |
| `two_sigma_sigma0` | `float` | Exactly 2 × sigma_sigma0 |
| `sigma1` | `float` | PSF sine variation, px |
| `sigma_sigma1` | `float` | 1σ |
| `two_sigma_sigma1` | `float` | Exactly 2 × sigma_sigma1 |
| `sigma2` | `float` | PSF cosine variation, px |
| `sigma_sigma2` | `float` | 1σ |
| `two_sigma_sigma2` | `float` | Exactly 2 × sigma_sigma2 |
| `Y_A` | `float` | Primary line (640 nm) intensity scale |
| `sigma_Y_A` | `float` | 1σ |
| `two_sigma_Y_A` | `float` | Exactly 2 × sigma_Y_A |
| `Y_B` | `float` | Secondary line (638 nm) intensity |
| `sigma_Y_B` | `float` | 1σ |
| `two_sigma_Y_B` | `float` | Exactly 2 × sigma_Y_B |
| `intensity_ratio` | `float` | Y_B / Y_A (diagnostic; expected 0.55–0.65) |
| `B` | `float` | CCD bias, ADU |
| `sigma_B` | `float` | 1σ |
| `two_sigma_B` | `float` | Exactly 2 × sigma_B |
| `epsilon_cal` | `float` | ε at OI 630 nm, from TolanskyResult (diagnostic) |
| `chi2_reduced` | `float` | Reduced χ² of Stage E fit |
| `chi2_stages` | `list[float]` | χ²_red after each stage A–E |
| `n_bins_used` | `int` | Good bins in fit |
| `n_params_free` | `int` | Always 11 for F01 v3 |
| `converged` | `bool` | Stage E convergence status |
| `quality_flags` | `int` | CalibrationFitFlags bitmask |
| `lambda_A_m` | `float` | 640.2248e-9 m |
| `lambda_B_m` | `float` | 638.2991e-9 m |
| `timestamp` | `float` | POSIX timestamp |

The `two_sigma_` fields must equal **exactly** `2 × sigma_`; enforced by `__post_init__`.

---

## 8. Quality flags — `CalibrationFitFlags`

```python
class CalibrationFitFlags:
    GOOD              = 0x000
    FIT_FAILED        = 0x001   # Stage E LM did not converge
    CHI2_HIGH         = 0x002   # chi2_red > 3.0
    CHI2_VERY_HIGH    = 0x004   # chi2_red > 10.0
    CHI2_LOW          = 0x008   # chi2_red < 0.5
    STDERR_NONE       = 0x010   # any stderr non-finite
    R_AT_BOUND        = 0x020   # R hit [0.10, 0.95]
    ALPHA_AT_BOUND    = 0x040   # α hit [0.5×, 2×] init
    FEW_BINS          = 0x080   # n_good < 100
    YB_RATIO_LOW      = 0x100   # Y_B/Y_A < 0.3 (possible family misidentification)
    YB_RATIO_HIGH     = 0x200   # Y_B/Y_A > 1.0 (families may be swapped)
```

---

## 9. Constants (from `windcube/constants.py`)

| Name | Value | Notes |
|------|-------|-------|
| `NE_WAVELENGTH_1_M` | 640.2248e-9 m | Burns, Adams & Longwell (1950) |
| `NE_WAVELENGTH_2_M` | 638.2991e-9 m | Burns, Adams & Longwell (1950) |
| `D_25C_MM` | 20.0006e-3 m | Tolansky/Benoit authoritative gap |
| `PLATE_SCALE_RPX` | 1.6071e-4 rad/px | Z01a WLS result, 2×2 binned |
| `R_REFL_FLATSAT` | 0.53 | FlatSat prior — used only as sanity check bound, not as init |
| `R_MAX_PX` | 110 | FlatSat/flight usable radius |

---

## 10. Relationship to v1/v2 spec and existing code

| v2 element | Status in v3 |
|-----------|-------------|
| Single-line forward model | **Replaced** by two-line model |
| Hardcoded 0.8 intensity ratio | **Replaced** by free parameter Y_B |
| §2.2 claim "638 nm line does not appear" | **Retracted** — both lines are present |
| Stages A→B→C→D (4 stages) | **Extended** to A→B→C→D→E (5 stages) |
| 9 free parameters | **Extended** to 11 (added Y_A, Y_B) |
| Tolansky seeding within F01 | **Moved** entirely to Z01a |
| Locked decisions LD-1, LD-2, LD-3 | Carried forward unchanged |

The `f01_full_airy_fit_to_neon_image_2026_04_21.py` implementation requires a new
module with this spec as source of truth. The old module should be kept as
`f01_full_airy_fit_to_neon_image_2026_04_21_v2_DEPRECATED.py` until the new
implementation passes T05 on real data.

---

## 11. Test matrix

| ID | Description | Pass criterion |
|----|-------------|---------------|
| T01 | Synthetic two-line fringe with known params → recovery | All 11 free params within 2σ of truth |
| T02 | d fixed at Tolansky value; fit does not move it | `result.t_m == tolansky.t_m` exactly |
| T03 | CHI2_HIGH flag when noise inflated × 10 | Flag set |
| T04 | Stage monotone: χ²(E) ≤ χ²(D) ≤ χ²(C) ≤ χ²(B) ≤ χ²(A) | Monotone non-increasing |
| **T05** | **Real 120s neon calibration image** | **χ²_red < 3.0, converged=True** |
| T06 | σ₀=σ₁=σ₂=0 → airy_modified == airy_ideal | Max abs diff < 1e-10 |
| T07 | R_AT_BOUND flag fires when R hits 0.95 | Flag set |
| T08 | `two_sigma_` == 2 × `sigma_` for all fields | Exact equality |
| T09 | Y_B/Y_A recovered within 5% on synthetic data | `abs(ratio_fit - ratio_true) < 0.05` |
| T10 | YB_RATIO_LOW flag fires when Y_B/Y_A < 0.3 | Flag set |
| T11 | Single-line synthetic (Y_B=0) converges and Y_B_fit < 0.05 | Flag YB_RATIO_LOW set |

**T05 is the primary acceptance test.** The v2 spec never passed T05 on real data
because the single-line model could not fit the two-line calibration spectrum.

---

## 12. Locked decisions (carried forward from v2, plus new)

**LD-1: n_scan = 211 (odd) in F02.** Unchanged.

**LD-2: OI_WAVELENGTH_M / OI_WAVELENGTH_VACUUM_M constant split.** Unchanged.

**LD-3: InstrumentParams.t default = 20.0006e-3 m.** Unchanged.

**LD-4: N_delta must be seeded with the ICOS nominal gap (20.008 mm), not D_25C_MM.**  
The Benoit boundary for N_delta = −189 vs −188 sits at T = 20.000896 mm. The
Tolansky-recovered gap D_25C_MM = 20.0006 mm is only 0.3 µm from this boundary and
will flip N_delta from −189 to −188 due to floating-point proximity. The ICOS nominal
value of 20.008 mm sits 7.1 µm safely inside the correct N_delta = −189 window.
Physical consistency: N_delta = −189 → d = 20.0006 mm (78 nm pre-load compression
of 20.008 mm as-built). N_delta = −190 → d = 20.107 mm (impossible expansion).
**Use T_MM = 20.008 mm in Z01a for N_delta computation only. Do not replace with
D_25C_MM.**

**LD-5: The 638 nm secondary neon line is present and significant in real calibration data.**  
The single-line approximation is invalid for the WindCube FlatSat lamp/filter/etalon
combination. The intensity ratio Y_B/Y_A ≈ 0.58 from real data analysis. This ratio
must be fitted as a free parameter, not hardcoded.

---

## 13. Open issues

- M05 retirement: pending T05 passage with v3 two-line model.
- Temporal interpolation of CalibrationResult between bracketing neon frames: deferred
  to spec F01b.
- The mean-sigma approximation in `M01.airy_modified()` (using a single scalar σ for the
  Gaussian PSF rather than position-dependent σ(r)) may limit χ²_red at small r. This
  is a known limitation of the current M01 implementation and is not addressed here.

---

*End of F01 spec v3 — 2026-04-22*

---

---

## 14. Claude Code implementation prompt

> **Instructions for Claude Code:**  
> Read this entire file first. Then execute the tasks below in order.
> Gate each task on pytest passing before proceeding to the next.
> Do not implement anything not described in this spec.

```
cat PIPELINE_STATUS.md

# Read the full spec before touching any code:
cat docs/specs/F01_full_airy_fit_to_neon_image_2026-04-22.md

# Read these existing files for context — do NOT modify them:
cat src/fpi/f01_full_airy_fit_to_neon_image_2026_04_21.py
cat src/fpi/m01_airy_forward_model_2026_04_05.py
cat src/fpi/m03_annular_reduction_2026_04_06.py
cat src/constants.py | grep -E "NE_WAVELENGTH|D_25C|PLATE_SCALE|R_MAX|R_REFL"

# ── TASK 1 of 5: Ensure NE_WAVELENGTH_2_M exists in constants.py ──────────
#
# Check whether NE_WAVELENGTH_2_M = 638.2991e-9 already exists in
# src/constants.py. If it does not, add it immediately after NE_WAVELENGTH_1_M
# with the comment:
#   # m, secondary Ne line (Burns, Adams & Longwell 1950)
# Also ensure NE_INTENSITY_2 = 0.8 exists as a legacy constant (keep it;
# the new spec replaces it with a free parameter but other code may import it).
# Do not change any other constants.

# ── TASK 2 of 5: Create the new F01 v3 module ─────────────────────────────
#
# Create src/fpi/f01_full_airy_fit_to_neon_image_2026_04_22.py
#
# The new module is a REPLACEMENT for f01_full_airy_fit_to_neon_image_2026_04_21.py.
# Keep the old file in place — do not delete or rename it.
#
# WHAT TO KEEP from the old module (copy these structures unchanged):
#   - TolanskyResult dataclass (identical)
#   - The FringeProfile + QualityFlags import pattern
#   - The column-scaled Jacobian covariance approach (§5.6) — this is correct
#     and proven. Copy it verbatim, expanding from 9 to 11 columns.
#   - The _stage_bounds helper pattern
#   - The r_fine / np.interp forward model evaluation pattern
#   - The input validation block (§5.1) — identical
#   - The sigma_floor logic — identical
#
# WHAT CHANGES in the new module:
#
#   (a) CalibrationFitFlags — add two new flags:
#       YB_RATIO_LOW  = 0x100   # Y_B/Y_A < 0.3
#       YB_RATIO_HIGH = 0x200   # Y_B/Y_A > 1.0
#
#   (b) CalibrationResult dataclass — add Y_A, Y_B, and intensity_ratio fields
#       following the exact field pattern of the existing params (value + sigma_
#       + two_sigma_). Also rename lambda_ne_m → lambda_A_m and add lambda_B_m.
#       Set n_params_free = 11 (not 9). Add __post_init__ that asserts all
#       two_sigma_ fields == 2 * sigma_ (raises AssertionError if violated).
#
#   (c) Module-level constants — change:
#       _N_FREE = 11  (was 9)
#       Add:
#       _STAGE_E_ORDER = ["R","alpha","I0","I1","I2","sigma0","sigma1","sigma2",
#                         "Y_A","Y_B","B"]
#       (this is the column order of the Stage E Jacobian for covariance extraction)
#
#   (d) Stage definitions — replace _STAGES with the 5-stage scheme from spec §5.3:
#       Stage A: ["Y_A", "Y_B", "B"]
#       Stage B: ["Y_A", "Y_B", "I0", "B"]
#       Stage C: ["Y_A", "Y_B", "I0", "I1", "I2", "B"]
#       Stage D: ["R", "alpha", "Y_A", "Y_B", "I0", "B"]
#       Stage E: ["R","alpha","I0","I1","I2","sigma0","sigma1","sigma2","Y_A","Y_B","B"]
#
#   (e) Bounds — add to _STATIC_BOUNDS:
#       "Y_A": (0.01, np.inf)
#       "Y_B": (0.01, np.inf)
#
#   (f) Two-line forward model — replace the single-line _model() function with:
#
#       def _model(p: dict) -> np.ndarray:
#           A_fine_A = airy_modified(r_fine, NE_WAVELENGTH_1_M, t_fixed,
#                          p["R"], p["alpha"], n_refr, r_max,
#                          p["I0"], p["I1"], p["I2"],
#                          p["sigma0"], p["sigma1"], p["sigma2"])
#           A_fine_B = airy_modified(r_fine, NE_WAVELENGTH_2_M, t_fixed,
#                          p["R"], p["alpha"], n_refr, r_max,
#                          p["I0"], p["I1"], p["I2"],
#                          p["sigma0"], p["sigma1"], p["sigma2"])
#           combined = p["Y_A"] * A_fine_A + p["Y_B"] * A_fine_B + p["B"]
#           return np.interp(r_good, r_fine, combined)
#
#       Both lines share all instrument params (R, alpha, I0, I1, I2, sigma0,
#       sigma1, sigma2) — this is physically correct per spec §3.1.
#
#   (g) Data-driven initial guesses — replace the initial guess block with the
#       amplitude-based family separation from spec §5.2:
#
#       1. Find all peaks in s_good using scipy.signal.find_peaks with
#          distance = max(5, int(len(s_good)/25)) and
#          height = np.percentile(s_good, 70).
#       2. Sort peaks by amplitude (s_good[peak_idx]).
#       3. Top half by amplitude → family A (640 nm, brighter).
#          Bottom half → family B (638 nm, dimmer).
#       4. B_init     = float(np.percentile(s_good, 2))
#       5. bright_amps = s_good[family_A_idx] - B_init
#          dim_amps    = s_good[family_B_idx]  - B_init
#       6. I0_init    = float(np.percentile(bright_amps, 80))
#          Clip to max(I0_init, 10.0).
#       7. Y_A_init   = 1.0   (absolute scale absorbed into I0)
#       8. Y_B_init   = float(np.median(dim_amps) / np.median(bright_amps))
#          Clip Y_B_init to [0.1, 1.5].
#          If fewer than 2 peaks in either family, set Y_B_init = 0.6 and
#          print a warning.
#       9. R_init: compute contrast from bright peaks only:
#             C = median((bright_amps) / (bright_amps + 2*B_init))
#             clipped to [0.05, 0.999]
#             F_est = 2*C/(1-C)
#             x = 2*(sqrt(1+F_est)-1)/F_est
#             R_init = clip(1-x, 0.1, 0.95)
#          Use R_init directly (not the hardcoded 0.53).
#       10. sigma0_init = 0.5 (default; FWHM estimation is in the validation
#           script, not in the core fit module).
#       11. All other params: I1=-0.1, I2=0.005, sigma1=0.0, sigma2=0.0
#           (Harding §3 defaults; unchanged from v2).
#
#       curr dict keys are now:
#       ["R","alpha","I0","I1","I2","sigma0","sigma1","sigma2","Y_A","Y_B","B"]
#
#   (h) DOF — update to:
#       dof = max(n_good - 11, 1)
#
#   (i) Covariance extraction — apply the column-scaled Jacobian method
#       (identical logic to v2) but with 11 columns using _STAGE_E_ORDER.
#       The YB_RATIO_LOW and YB_RATIO_HIGH flags are set after covariance:
#         ratio = curr["Y_B"] / curr["Y_A"]
#         if ratio < 0.3:  quality_flags |= CalibrationFitFlags.YB_RATIO_LOW
#         if ratio > 1.0:  quality_flags |= CalibrationFitFlags.YB_RATIO_HIGH
#
#   (j) CalibrationResult construction — add Y_A, Y_B, intensity_ratio fields.
#       Set lambda_A_m = NE_WAVELENGTH_1_M, lambda_B_m = NE_WAVELENGTH_2_M.
#       Remove lambda_ne_m (renamed). n_params_free = 11.
#
# fit_neon_fringe() signature is unchanged:
#   def fit_neon_fringe(profile: FringeProfile,
#                       tolansky: TolanskyResult,
#                       R_init: float = None) -> CalibrationResult
# The R_init argument: if provided, use it in place of the contrast-derived
# R_init (allows external override for testing).

# ── TASK 3 of 5: Write tests ───────────────────────────────────────────────
#
# Create validation/test_f01_two_line_airy_fit.py with tests T01–T11 from
# spec §11. Use the following synthetic fringe generator for all tests:
#
#   def _make_synthetic_profile(R, alpha, I0, I1, I2, s0, s1, s2,
#                                Y_A, Y_B, B, d, r_max, rng_seed=42):
#       rng = np.random.default_rng(rng_seed)
#       r = np.linspace(1.0, r_max, 500)
#       A_fine_A = airy_modified(r, NE_WAVELENGTH_1_M, d, R, alpha, 1.0,
#                                r_max, I0, I1, I2, s0, s1, s2)
#       A_fine_B = airy_modified(r, NE_WAVELENGTH_2_M, d, R, alpha, 1.0,
#                                r_max, I0, I1, I2, s0, s1, s2)
#       signal = Y_A * A_fine_A + Y_B * A_fine_B + B
#       noisy  = rng.poisson(np.maximum(signal, 1)).astype(np.float32)
#       sigma  = np.maximum(np.sqrt(noisy) / 8.0, 1.0).astype(np.float32)
#       return r, noisy, sigma
#
# Build a FringeProfile-compatible SimpleNamespace for each test:
#   profile = SimpleNamespace(r_grid=r, r2_grid=r**2, profile=noisy,
#                             sigma_profile=sigma,
#                             masked=np.zeros(len(r), dtype=bool),
#                             r_max_px=float(r_max), quality_flags=0)
#
# Truth parameters for T01:
#   R=0.72, alpha=1.6133e-4, I0=2200.0, I1=-0.1, I2=0.005,
#   sigma0=0.5, sigma1=0.0, sigma2=0.0,
#   Y_A=1.0, Y_B=0.58, B=6100.0, d=20.0006e-3, r_max=110.0
#
# T01: All 11 free params recovered within 2σ of truth.
# T02: result.t_m == tolansky.t_m exactly (gap not moved).
# T03: CHI2_HIGH flag set when sigma inflated ×10.
# T04: chi2_stages is monotone non-increasing (5 values, A through E).
# T05: pytest.skip("requires real binary") unless tests/data/cal_image_120s.bin
#      exists. If it exists: load, reduce, fit, assert chi2_red < 3.0.
# T06: sigma0=sigma1=sigma2=0 → airy_modified == airy_ideal (max diff < 1e-10).
# T07: Force R to hit upper bound (R_init=0.94). Assert R_AT_BOUND flag set.
# T08: two_sigma_ == 2 * sigma_ for all 11 params (exact float equality via
#      __post_init__ — the test should confirm the assertion does not fire on a
#      good result, and DOES fire if a two_sigma field is manually corrupted).
# T09: Y_B/Y_A recovered within 0.05 of truth (0.58) on T01 synthetic.
# T10: Set Y_B_true=0.2 (below 0.3 threshold). Assert YB_RATIO_LOW flag set.
# T11: Set Y_B_true=0.0, Y_A_true=1.0 (single-line synthetic). Assert that
#      Y_B_fit < 0.05 and YB_RATIO_LOW is set.
#
# Run: pytest validation/test_f01_two_line_airy_fit.py -v
# Report full output.

# ── TASK 4 of 5: Run full test suite ──────────────────────────────────────
#
pytest validation/ tests/ -v --tb=short 2>&1 | tail -25
#
# Expected baseline going into this session:
#   3 failed (m02, m03, m04 — pre-existing), 134 passed, 8 skipped
# Any change to the failure count (other than new T01-T11 additions) must be
# explained. If the old f01 tests in validation/test_f01_full_airy_fit_to_neon.py
# still import from the v2 module they will continue to pass — do not break them.

# ── TASK 5 of 5: Update PIPELINE_STATUS.md and commit ──────────────────────
#
# Update PIPELINE_STATUS.md:
#   - Add F01 v3 entry with status = IMPLEMENTED if T01-T11 pass (T05 may skip)
#   - Note: old F01 v2 module kept in place as DEPRECATED
#
git add src/fpi/f01_full_airy_fit_to_neon_image_2026_04_22.py \
        src/constants.py \
        validation/test_f01_two_line_airy_fit.py \
        PIPELINE_STATUS.md
git commit -m "feat: implement F01 v3 — two-line Airy fit to neon profile

Replaces single-line model with two-line forward model (640+638 nm).
Y_B added as 11th free parameter; data-driven R_init from contrast.
5-stage LM scheme (A-E). 10/11 tests pass (T05 skipped, no real bin).
Old v2 module kept as f01_..._2026_04_21.py (DEPRECATED).

Also updates PIPELINE_STATUS.md"

# ── REPORT BACK ────────────────────────────────────────────────────────────
#
# Report the following:
# 1. Full pytest output for T01-T11 (pass/fail/skip for each)
# 2. T01 fitted values for all 11 params vs truth, with 1σ
# 3. T01 chi2_reduced
# 4. T01 Y_B/Y_A recovered ratio vs truth (0.58)
# 5. T09 Y_B/Y_A absolute error
# 6. Full pytest summary line for Task 4
# 7. Whether NE_WAVELENGTH_2_M already existed in constants.py
```
