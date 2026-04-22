# F02 — Full Airy Fit to Airglow Image

**Spec ID:** F02  
**Title:** Full Airy Fit to Airglow Image (Doppler Wind Inversion)  
**Version:** v1  
**Date:** 2026-04-21  
**Author:** Claude AI / Scott Sewell  
**Repo:** `soc_sewell`  
**Spec file:** `docs/specs/F02_full_airy_fit_to_airglow_image_2026-04-21.md`  
**Depends on:** F01 (CalibrationResult), M01, M03  
**Consumed by:** M07 (LOS-to-vector wind decomposition), S20 (L2 netCDF output)

---

## 0. Pipeline context — 8-step calibration-to-wind chain

F02 implements **steps 5–8** of the calibration-to-wind pipeline.
F01 implements **steps 1–4** and produces the `CalibrationResult` consumed here.

| Step | Module | Description |
|------|--------|-------------|
| 1 | Z01a | Two-line neon exposure; annular-reduce calibration image → ring radii r²_fit for λ₆₄₀ and λ₆₃₈ |
| 2 | Z01a | Tolansky WLS: fit ring-order P vs r²_fit; recover fractional orders ε₆₄₀, ε₆₃₈; plate-scale α from slope |
| 3 | Z01a | Benoit two-line gap recovery (Vaughan Eq. 3.97): d = (N_Δ + ε_a − ε_b)·λa·λb / [2(λb − λa)] |
| 4 | F01 | Full modified-Airy fit to neon 1D fringe profile: fix d; free-fit R, α, I₀, I₁, I₂, σ₀, σ₁, σ₂, B → CalibrationResult |
| **5** | **F02** | **Annular reduction of airglow frame (calls M03): FringeProfile S(r), ~500 equal-pixel-count radial bins** |
| **6** | **F02** | **Brute-force χ² scan over ±½ FSR around λ_OI; analytic solve for Y_line, B_sci at each of 200 grid points** |
| **7** | **F02** | **Levenberg–Marquardt inversion: free-fit λ_c, Y_line, B_sci (3 DOF); all 10 instrument params fixed from CalibrationResult** |
| **8** | **F02** | **Doppler wind v_rel and 2σ uncertainty; AirglowFitResult → M07** |

The critical constraint enforced across steps 5–8 is that **all 10 instrument parameters are
fixed at the values delivered by F01**.  F02 fits only 3 free parameters.  This separation —
10-parameter neon fit in F01, 3-parameter airglow fit in F02 — is the Harding et al. (2014)
architecture and is the reason the wind retrieval reaches sub-m/s precision.

---

## 1. Purpose and scope

F02 inverts an OI 630.0 nm airglow `FringeProfile` to recover the line-of-sight Doppler wind
velocity `v_rel` and its 2σ uncertainty.  The source spectrum is modelled as a
thermally-broadened, Doppler-shifted Gaussian (Harding Eq. 10) convolved with the instrument
function fixed by F01.

In WindCube's operational regime, winds of ±8000 m/s correspond to ±0.17 FSR at 630 nm, well
within the ±½ FSR scan window of step 6.  The scan unambiguously identifies the correct fringe
order before handing off to LM (step 7).

F02 is a refactor and formal specification of the existing `M06.fit_airglow_fringe()` function.
The algorithm is unchanged; this spec adds the pipeline-context header (§0), tightens the
constant-value requirements, and clarifies the interface to F01's `CalibrationResult`.

---

## 2. Physical model

### 2.1 Source spectrum — thermally broadened Gaussian

```
Y(λ) = Y_bg + Y_line · exp[−½ · ((λ − λ_c) / Δλ)²]
```

where (Harding Eqs. 10–12):

- `Y_bg` — background sky emission, wavelength-independent (not a free parameter in WindCube;
  absorbed into `B_sci` because the narrowband filter suppresses continuum)
- `Y_line` — line intensity scale factor (free)
- `λ_c = λ_OI · (1 + v_rel / c)` — Doppler-shifted line centre (free)
- `Δλ = (λ_OI / c) · √(k_B · T / m_O)` — Doppler breadth (not a free parameter; temperature
  is not a WindCube science product; Δλ is absorbed into the forward model via the delta-function
  approximation — see §2.2)

### 2.2 Delta-function source approximation

The OI 630 nm source is modelled as a **delta function** at `λ_c` (i.e., `Δλ → 0`).  This is
valid because at thermospheric temperatures (~800 K) the thermal width is ~0.003 nm ≪ 1 FSR
(~0.01 nm), so the convolution with the instrument function is negligible.  This reduces the
free parameters from 5 (Harding Table 2) to 3, removing the temperature degeneracy.

The forward model therefore collapses to:

```
S(r) = Y_line · Ã(r; λ_c) + B_sci
```

where `Ã(r; λ_c)` is `M01.airy_modified()` evaluated at `λ_c` with all 10 instrument
parameters fixed from `CalibrationResult`.

### 2.3 Free and fixed parameters

| Parameter | Symbol | Fixed/Free in F02 | Source if fixed |
|-----------|--------|--------------------|-----------------|
| All 10 instrument params | t, R, α, I₀, I₁, I₂, σ₀, σ₁, σ₂, B_cal | **Fixed** | F01 `CalibrationResult` |
| Line centre | `λ_c` | Free | Init: brute-force scan (§4.2) |
| Line intensity | `Y_line` | Free | Init: analytic solve at scan minimum |
| Science bias | `B_sci` | Free | Init: analytic solve at scan minimum |

**Total free parameters: 3.**

Note: `B_sci` is fitted separately from the calibration bias `B` in `CalibrationResult`
because the science frame may have a different pedestal level than the neon frame.

---

## 3. Input

### 3.1 `FringeProfile` (from M03 `reduce_science_frame()`)

Same schema as F01 §3.1.  The profile must have dark subtraction already applied before
passing to F02; see M03 spec for dark-frame handling.

### 3.2 `CalibrationResult` (from F01)

The full F01 output object (§5 of F01 spec).  F02 uses the following fields:

| Field | Used as |
|-------|---------|
| `t_m` | Fixed etalon gap in `airy_modified()` |
| `R_refl` | Fixed plate reflectivity |
| `alpha` | Fixed magnification |
| `I0`, `I1`, `I2` | Fixed intensity envelope |
| `sigma0`, `sigma1`, `sigma2` | Fixed PSF widths |
| `B` | Not used directly (B_sci is re-fitted) |
| `epsilon_cal` | Rest-frame ε₀ at λ_OI; diagnostic only |
| `quality_flags` | Sets `CAL_QUALITY_DEGRADED` flag in output if non-zero |

---

## 4. Algorithm

### 4.1 Validate inputs

- Raise `ValueError` if `profile.quality_flags & CENTRE_FAILED`.
- Build `good_mask`: unmasked bins with finite, positive `sigma_profile` and finite `profile`.
- Require ≥ 10 good bins; raise `ValueError` otherwise.
- Apply sigma floor: `sigma_good = max(sigma_raw, max(1.0, median(profile) × 0.005))`.
- If `cal.quality_flags != 0`: set `CAL_QUALITY_DEGRADED` in output flags.

### 4.2 Brute-force scan for λ_c (step 6)

Scan 200 evenly-spaced `λ_c` values across `[λ_OI − ½ FSR, λ_OI + ½ FSR]`.  At each grid
point analytically solve for `Y_line` and `B_sci` via weighted linear least squares (2×2
normal equations).  Record χ²_red.  Select `λ_c_best` at minimum χ².

Check for `SCAN_AMBIGUOUS`: if any other grid point has χ² within 10% of the minimum,
set the flag.

Scan bounds: `λ_c ∈ [λ_OI − ½ FSR, λ_OI + ½ FSR]`.  Because ±8000 m/s corresponds to
±0.17 FSR, this window comfortably contains all expected winds while remaining within a
single FSR period (avoiding order-ambiguity).

### 4.3 Levenberg–Marquardt inversion (step 7)

Free parameters: `x = [λ_c, Y_line, B_sci]`.  
Initial values: `λ_c_best`, `Y_line_init`, `B_sci_init` from scan step.

Forward model evaluation per LM iteration:
1. Evaluate `airy_modified(r_fine, λ_c, ...)` on fine grid of 500 points (instrument params
   from `CalibrationResult`).
2. Linearly interpolate to `r_good` bin centres.
3. Compute `model = Y_line × airy_bins + B_sci`.
4. Return residual vector `(profile_good − model) / sigma_good`.

Bounds:
- `λ_c`: `[λ_OI − 1.5 FSR, λ_OI + 1.5 FSR]`
- `Y_line`: `[0, ∞)`
- `B_sci`: `[0, ∞)`

Use `scipy.optimize.least_squares(method='lm')`.

### 4.4 χ² and convergence

```
χ²_red = Σ residuals² / (n_good − 3)
```

Flag thresholds: CHI2_VERY_HIGH > 10.0, CHI2_HIGH > 3.0, CHI2_LOW < 0.5.

### 4.5 Uncertainty estimation (step 8, uncertainty part)

Compute covariance from the Jacobian of the final 3-parameter residual vector:

```
cov = χ²_red · (Jᵀ J)⁻¹
[σ_λc, σ_Yline, σ_Bsci] = sqrt(diag(cov))
```

Use pseudoinverse fallback with `STDERR_NONE` flag if JᵀJ is near-singular (cond > 10¹⁴).

### 4.6 Doppler wind conversion (step 8, wind part)

```
v_rel   = c · (λ_c − λ_OI) / λ_OI
σ_v     = c · σ_λc / λ_OI
```

Sign convention (locked): positive `v_rel` = recession (redshift) = fringes shift to **smaller**
radius (inward).

The `lam_rest_nm` used to evaluate `λ_OI` must be set to **629.95 nm** (not 630.0304 nm) to
avoid the half-integer N_int boundary at 629.9974 nm, which causes velocity-sign failures.

### 4.7 Phase diagnostic

```
epsilon_sci   = (2 · λ_c / λ_OI) mod 1.0
delta_epsilon = epsilon_sci − cal.epsilon_cal
```

These are stored in `AirglowFitResult` for diagnostic use by M07 and L2 QC; they do not enter
the wind calculation.

---

## 5. Output — `AirglowFitResult`

| Field | Type | Description |
|-------|------|-------------|
| `lambda_c_m` | `float` | Fitted line centre, metres |
| `sigma_lambda_c_m` | `float` | 1σ uncertainty |
| `two_sigma_lambda_c_m` | `float` | Exactly 2 × sigma_lambda_c_m |
| `v_rel_ms` | `float` | LOS Doppler wind, m/s (+ = recession) |
| `sigma_v_rel_ms` | `float` | 1σ wind uncertainty, m/s |
| `two_sigma_v_rel_ms` | `float` | Exactly 2 × sigma_v_rel_ms |
| `Y_line` | `float` | Fitted line intensity scale |
| `sigma_Y_line` | `float` | 1σ |
| `two_sigma_Y_line` | `float` | Exactly 2 × sigma_Y_line |
| `B_sci` | `float` | Fitted science frame bias, counts |
| `sigma_B_sci` | `float` | 1σ |
| `two_sigma_B_sci` | `float` | Exactly 2 × sigma_B_sci |
| `chi2_reduced` | `float` | Reduced χ² of LM fit |
| `n_bins_used` | `int` | Number of good radial bins |
| `n_params_free` | `int` | Always 3 for F02 |
| `converged` | `bool` | LM convergence status |
| `quality_flags` | `int` | AirglowFitFlags bitmask |
| `epsilon_sci` | `float` | Fractional order at fitted λ_c |
| `delta_epsilon` | `float` | epsilon_sci − epsilon_cal (diagnostic) |
| `calibration_t_m` | `float` | cal.t_m, traceability |
| `calibration_epsilon_cal` | `float` | cal.epsilon_cal, traceability |
| `lambda_c_scan_init_m` | `float` | Scan centre (= λ_OI), diagnostic |
| `lambda_c_lm_init_m` | `float` | λ_c passed to LM after scan, diagnostic |

The `two_sigma_` fields must equal **exactly** `2 × sigma_` (enforced by T08).

---

## 6. Quality flags — `AirglowFitFlags`

```python
class AirglowFitFlags:
    GOOD                 = 0x000
    FIT_FAILED           = 0x001   # LM did not converge
    CHI2_HIGH            = 0x002   # chi2_red > 3.0
    CHI2_VERY_HIGH       = 0x004   # chi2_red > 10.0
    CHI2_LOW             = 0x008   # chi2_red < 0.5
    SCAN_AMBIGUOUS       = 0x010   # second-best scan chi2 within 10% of minimum
    LAMBDA_C_AT_BOUND    = 0x020   # lambda_c hit its bound
    STDERR_NONE          = 0x040   # any stderr non-finite
    LOW_SNR              = 0x080   # (max−min profile) / B_sci < 1.0
    CAL_QUALITY_DEGRADED = 0x100   # CalibrationResult had non-GOOD flags
```

---

## 7. Constants (from `windcube/constants.py`)

| Name | Value | Notes |
|------|-------|-------|
| `OI_WAVELENGTH_M` | 629.95e-9 m | Rest wavelength for inversion; avoids N_int boundary |
| `OI_WAVELENGTH_VACUUM_M` | 630.0304e-9 m | Physical vacuum wavelength; not used in inversion |
| `ETALON_FSR_OI_M` | λ²/(2nd) | Computed from D_25C_MM at runtime |
| `SPEED_OF_LIGHT_MS` | 299 792 458 m/s | Exact |

---

## 8. Relationship to existing modules

| Existing module | Role in F02 |
|----------------|------------|
| `M06.fit_airglow_fringe()` | F02 formalises and supersedes M06 |
| `M01.airy_modified()` | Forward model (unchanged) |
| `M03.reduce_science_frame()` | Produces FringeProfile input |
| `F01` | Produces CalibrationResult input |

The existing M06 implementation is algorithmically consistent with this spec.  The only
required code change is to ensure `OI_WAVELENGTH_M` in constants is set to 629.95e-9 m and
that `CalibrationResult` is populated from F01 (not from `InstrumentParams` defaults).

---

## 9. Test matrix

| ID | Description | Pass criterion |
|----|-------------|---------------|
| T01 | Synthetic airglow at v = 0 → recovery | \|v_rel\| < 1 m/s |
| T02 | Synthetic at v = +500 m/s → recovery | \|v_err\| < 2σ |
| T03 | Synthetic at v = −500 m/s → recovery | \|v_err\| < 2σ |
| T04 | v = +8000 m/s (near scan edge) → no FSR jump | Correct order recovered |
| T05 | SCAN_AMBIGUOUS flag fires for injected degenerate profile | Flag set |
| T06 | CalibrationResult with non-zero flags → CAL_QUALITY_DEGRADED | Flag set |
| T07 | `two_sigma_v_rel_ms == 2 × sigma_v_rel_ms` | Exact equality |
| T08 | sign convention: positive v_rel → fringes at smaller r | Verified in T02 residuals |
| T09 | Real airglow frame (if available) → chi2_red < 3.0 | Pass |

---

## 10. Open issues

- Temporal interpolation of `CalibrationResult` between bracketing neon frames (Harding §3):
  out of scope for F02; candidate spec F01b.
- The Harding 0.4 m/s systematic bias and zero-mean vertical wind correction
  (M07 responsibility) are not handled here.
- The `Δλ` (temperature) parameter is deliberately excluded from the 3-parameter fit.
  If temperature becomes a science product in a future mission phase, the free parameter
  count increases to 4 and a new spec version is required.

---

*End of F02 spec v1 — 2026-04-21*
