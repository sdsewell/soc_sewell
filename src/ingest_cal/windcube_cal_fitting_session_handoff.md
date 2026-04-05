# WindCube FPI — Calibration Fitting Session Handoff
**Date:** 2026-04-03  
**Topic:** Two-wavelength neon calibration image fitting — clean-slate plan  
**Status:** Plan complete, implementation not yet started  
**Next action:** Implement Stage 0–2 diagnostic script in Claude Code (VS Code)

---

## Context

This document captures a complete session reviewing all existing FPI calibration fitting code, diagnosing its weaknesses, examining a real calibration image (`cal_image_L1_1.npy`), and producing a clean 5-stage fitting plan from scratch. It is intended for handoff to a fresh Claude session with no prior context.

---

## The real data: `cal_image_L1_1.npy`

| Property | Value |
|---|---|
| Shape | 216 × 216 pixels |
| Dtype | `>u2` (big-endian uint16) — cast to float before use |
| Min / Max | 5275 / 11769 counts |
| Mean / Median | 7125 / 7080 counts |
| 5th percentile (background) | ~5883 counts (raw); **~6700 after center-corrected radial profile** |
| 99th percentile | ~9127 counts |
| Peak signal above background | **~2100 counts** (contrast ratio ~0.31) |
| Center of mass | approximately (cx=111.6, cy=107.2) pixels |
| Usable r_max | ~100 px |

**Fringe structure observed in radial profile:**
- Clear ring peaks at approximately r ≈ 16, 38, 52 px
- Further rings visible but spacing compresses and contrast drops at larger r
- ~29 local extrema detected across full radius — consistent with a two-line beat pattern
- Shot noise per bin: √6700 ≈ 82 counts; fringe SNR per bin ≈ 26

This is an **L1 image** (dark-subtracted, flat-corrected). It is **not** a raw L0 image. The background of ~6700 counts is CCD bias + any residual diffuse signal, not a dark-current level.

---

## Existing code review and problems found

### `fpi_airy_fit.py` (legacy migrated code)

**What it does correctly:**
- Wavelengths λ₁ = 640.2248 nm, λ₂ = 638.2991 nm are **correctly fixed** — they are captured in the residual closure before `least_squares` is called and never appear in the parameter vector `x0`. This is the right design.
- Uses `scipy.optimize.least_squares` with TRF method and tight tolerances (1e-14)
- Computes ε_cal and ε_sci (fractional fringe order) from fitted `d`
- Estimates parameter uncertainties from SVD of the Jacobian

**Problems:**
1. **Wrong vignetting model:** Uses `exp(-κr²)` — an exponential decay. This has no physical basis for this optical design. The correct form (Harding Eq. 4) is a **quadratic polynomial** `I₀(1 + I₁(r/r_max) + I₂(r/r_max)²)`.
2. **Wrong initial `d`:** Default `d_mm_init = 20.67 mm` is the known FlatSat FSR-period artifact — it is **2068 FSR periods away** from the physical etalon gap of ~20.008 mm. Any fresh run with this default will converge to the wrong valley.
3. **`A₂` floats by default:** The relative intensity of the two neon lines is a lamp property determinable from NIST transition data, not a free parameter. Floating it silently absorbs systematic errors rather than flagging them.
4. **No staged fitting:** All ~6 parameters float simultaneously from the start. The optimizer navigates a high-dimensional, multiply-periodic landscape blindly.
5. **No ring center pre-fit:** Center of mass is used as-is without a dedicated ring-geometry fit. A 0.5 px center error produces artificial fringe broadening that biases R and then ε.

### `m01_airy_forward_model.py` (production M01 module)

**What it is:** The production forward model implementing the full Harding (2014) Airy function chain. Not called by `fpi_airy_fit.py` — completely separate code path.

**Functions:**
- `theta_from_r(r, alpha)` — maps pixel radius to angle: θ = arctan(α·r)
- `intensity_envelope(r, r_max, I0, I1, I2)` — Harding Eq. 4 quadratic vignetting
- `airy_ideal(r, λ, t, R, α, n, r_max, I0, I1, I2)` — single-wavelength Airy (Harding Eq. 2)
- `psf_sigma(r, r_max, σ₀, σ₁, σ₂)` — Fourier PSF width profile (Harding Eq. 8)
- `airy_modified(r, λ, t, R, α, n, r_max, I0, I1, I2, σ₀, σ₁, σ₂)` — PSF-convolved Airy (Harding Eq. 5/14)
- `build_instrument_matrix(r, wavelengths, ...)` — full (R × L) instrument matrix

**Key design note:** `build_instrument_matrix` is for the general multi-wavelength inversion (M04/M06 airglow). For the two-line neon calibration, you do **not** need the full matrix — two calls to `airy_modified` at λ₁ and λ₂ are sufficient and much faster.

**Call map (from `fit_calibration_profile`):**
```
fit_calibration_profile()          ← public entry point
  ├── _auto_guess()                 ← init I_scale, I_bg from data percentiles
  ├── _run_lsq()                    ← scipy least_squares (TRF)
  │     └── resid() closure         ← called ~1000s of times per fit
  │           └── _model_cal()      ← two-line Airy model
  │                 ├── _phase_profile(r², d, λ₁, f, p)   ← δ(r²) = (4πd/λ)(1-r²/2f²)
  │                 ├── _airy_T(δ₁, R)                    ← T = 1/(1+F·sin²(δ/2))
  │                 ├── _phase_profile(r², d, λ₂, f, p)
  │                 └── _airy_T(δ₂, R)
  ├── _fit_quality()                ← rms, χ²/dof, residuals  [post-optimizer]
  ├── _covariance()                 ← SVD of Jacobian → stderr [post-optimizer]
  └── _epsilon(d, λ)  [×2]         ← m₀ = 2d/λ, ε = m₀ mod 1 [post-optimizer]
      → CalibrationFitResult
```

---

## The clean-slate fitting plan

### Core philosophy

The two-line neon calibration is fundamentally a problem with **two physically fundamental unknowns** (`t` and `R`) plus several nuisance parameters (vignetting, PSF, background) that can be pre-estimated or staged in separately. Give Levenberg-Marquardt the smallest, best-conditioned problem at each stage.

### Instrument constants (never float these)

| Constant | Value | Source |
|---|---|---|
| λ₁ (Ne line 1) | 640.2248 nm | NIST atomic database |
| λ₂ (Ne line 2) | 638.2991 nm | NIST atomic database |
| λ_sci (OI) | 630.0 nm | Science target |
| Pixel pitch | 32 µm | CCD97-00 datasheet |
| Focal length | 200 mm | Instrument design |
| α (magnification) | 1.6×10⁻⁴ rad/px | pitch/focal_length |
| n (index) | 1.0 | vacuum gap |
| A₂ (line ratio) | ~0.67 | NIST (see note below) |

**Note on A₂:** The NIST value for the Ne I 640.2/638.3 nm intensity ratio should be looked up and hardcoded. Do **not** float A₂ in routine fits. If the model fit is poor at the fixed NIST value, that is a diagnostic signal (non-equilibrium lamp, spectral contamination) to investigate — not a reason to let A₂ absorb the error silently.

**Note on α:** α = pixel_pitch/focal_length is the engineering-design value. Float it in Stage 5 with a ±12.5% bound only to check for focus or alignment changes, not as a routine free parameter.

---

### Stage 0 — Inspect and pre-process (no optimizer)

**Goal:** Extract known quantities directly from the data.

Steps:
1. Load `cal_image_L1_1.npy`, cast to float64
2. Compute center of mass as initial cx, cy estimate
3. Bin the image into **equal-area annular bins** (not equal-width) — ~150 bins for this 216×216 image. Equal-area bins give approximately equal photon noise per bin, which is the χ² assumption.
4. Compute uncertainty vector: `σᵢ = sqrt(Iᵢ)` (shot noise; L1 image so this is appropriate)
5. Read off directly:
   - **B ≈ 5th percentile of radial profile ≈ 6700 counts** — fix this, do not float it
   - **I₀ ≈ 99th percentile - B ≈ 2100 counts** — use as seed
   - **Ring peak positions** at r ≈ 16, 38, 52 px — record for Stage 2

**Why fix B:** The background is directly observable from the outermost/lowest bins of the profile. Floating it introduces a correlation with I₀ that slows convergence and can produce degenerate solutions where B rises and I₀ falls (or vice versa) with no change in χ².

---

### Stage 1 — Ring center fit (geometric, no Airy model)

**Goal:** Determine (cx, cy) to sub-pixel accuracy.

**Method:** For 3–4 of the sharpest fringe rings visible in the 2D image, threshold the image near each ring's peak, identify contiguous bright regions, and fit circles to each ring ridge. Take the median center across rings.

This is Harding's prescribed method and gives sub-0.1 px accuracy, which is required: a 0.3 px center error is sufficient to introduce a ~1% temperature bias (Harding), and propagates into ε through profile smearing.

**Why this is separate from the Airy fit:** The ring center is a geometric property of the optical axis on the detector, not a parameter of the Airy transmission function. Conflating them causes the Airy optimizer to absorb centering errors into σ₀ (PSF width), which then biases R, which biases ε. Always solve the center problem geometrically first.

**Output:** Fixed (cx, cy) for all subsequent stages.

---

### Stage 2 — Beat-period estimate of `t` (analytic, no optimizer)

**Goal:** Determine `t` to within ±0.5 FSR (~10 µm) to seed Stage 3 in the correct interference order valley.

**Method:** The ring maxima in r² space fall at positions:

```
r²_m = (f²/p²) × (1 - λ·m/(2nt))
```

where m is the interference order. The spacing between consecutive maxima in r² is linear in 1/t. By measuring r²_peak positions from the Stage 0 profile and fitting a line through (m, r²_m), you extract t analytically.

The two-line beat pattern has a characteristic spatial period in r² given by:

```
Δ(r²)_beat = (f²/p²) × (λ₁²/Δλ) / (2nt)
```

where Δλ = λ₁ - λ₂ = 1.9257 nm. This beat period is clearly visible in the profile (the alternating strong/weak ring amplitude pattern).

**Why this is critical:** The FlatSat code arrived at `d = 20.670 mm`, which is the ICOS measured value of 20.008 mm plus 2068 FSR periods (≈ 0.662 mm = 2068 × 320 nm). This is a well-known FSR-period ambiguity in FPI fitting — the Airy function repeats every FSR, so any valley is a valid local minimum. The beat-period analysis resolves this ambiguity because the beat period uniquely determines t without this degeneracy. **Do not skip Stage 2.**

**Output:** `t_seed` — initial estimate for Stage 3, guaranteed to be in the physically correct FSR valley.

---

### Stage 3 — Two-parameter fit: `t` and `R` only

**Goal:** Recover the two core Airy parameters cleanly.

**Free parameters:** `t` (seeded from Stage 2), `R`  
**Fixed:** everything else

```python
# Fixed values at this stage
B     = 6700          # from Stage 0
cx,cy = ...           # from Stage 1
alpha = 1.6e-4        # engineering value (rad/px)
I0    = 2100          # from Stage 0 (I_scale = peak - B)
I1, I2 = 0.0, 0.0    # flat envelope
sigma0, sigma1, sigma2 = 0.5, 0.0, 0.0   # narrow symmetric PSF
A2    = 0.67          # NIST line ratio (fix this)
```

**Bounds:**
- `t`: (19.95e-3, 20.07e-3) m — spans ±0.06 mm around ICOS value, safely excludes 20.670 mm
- `R`: (0.35, 0.75) — effective instrument range (coating spec 80% reduced by flatness losses)

**Forward model:** Two calls to `airy_modified(r, λ₁, ...)` and `airy_modified(r, λ₂, ...)`, combined as:
```
s(r) = B + I_scale × exp/envelope × (T(r,λ₁) + A₂·T(r,λ₂)) / (1 + A₂)
```

**Diagnostic:** After Stage 3, plot data vs model. If χ²_red ≈ 1.0, the core physics is right. If χ²_red > 2.0, the residuals will tell you what's missing (slow radial trend → vignetting; systematic fringe width error → PSF; coherent oscillation → wrong t period).

---

### Stage 4 — Intensity envelope and PSF

**Goal:** Clean up residuals from Stage 3 that reflect vignetting and PSF effects.

**Hold fixed:** `t`, `R` from Stage 3 results  
**Free parameters:** `I₀`, `I₁`, `I₂` (quadratic vignetting envelope), then `σ₀`

**Run in two sub-steps:**
1. Float I₀, I₁, I₂ only → fit vignetting profile
2. Then release σ₀ (average PSF width) → fit fringe broadening

**Decision rule for σ₁, σ₂:** Only add these (PSF radial variation) if sub-step 2 residuals show a systematic trend in fringe width as a function of radius. These are the most weakly constrained parameters and add parameter correlation. Default: fix σ₁ = σ₂ = 0.

**Why quadratic envelope, not exponential:** Harding Eq. 4 `I₀(1 + I₁·ρ + I₂·ρ²)` where ρ = r/r_max is physically motivated by the aperture stop geometry of the collimating lens. The exponential `exp(-κr²)` used in the legacy `fpi_airy_fit.py` has no physical basis here and produces a different functional shape that can confuse the fit.

---

### Stage 5 — Joint refinement

**Goal:** Final polish with all physically meaningful parameters floating simultaneously.

**Free parameters:** `t`, `R`, `α`, `I₀`, `I₁`, `I₂`, `σ₀`  
**Held fixed throughout:** `B` (data-determined), `A₂` (NIST), `σ₁`, `σ₂` (unless Stage 4 showed clear need)

**Seeds:** Stage 3 and Stage 4 best-fit values

**Bounds summary:**

| Parameter | Bounds | Rationale |
|---|---|---|
| `t` | (19.95e-3, 20.07e-3) m | ±0.06 mm around ICOS; excludes 20.67 mm |
| `R` | (0.35, 0.75) | Effective instrument range |
| `α` | (1.4e-4, 1.8e-4) rad/px | ±12.5% around engineering value |
| `I₀` | (100, 15000) | CCD97-00 operational range |
| `I₁` | (−0.5, 0.5) | Vignetting coefficient |
| `I₂` | (−0.5, 0.5) | Vignetting coefficient |
| `σ₀` | (0.1, 3.0) px | Realistic PSF range |

**Degeneracy warning:** `t` and `α` are partially degenerate — both affect ring spacing in r² through the product `t·α²`. Monitor the correlation coefficient between them in the Jacobian covariance. If |corr(t, α)| > 0.95, tighten the `α` bound further (it is better determined from engineering than from the fringe pattern).

**Primary output:**  
```
ε_cal = (2t/λ₁) mod 1    # fractional fringe order at λ₁ = 640.2248 nm
ε_sci = (2t/λ_sci) mod 1  # projected to 630.0 nm science wavelength
```
This is the zero-wind phase reference used by M06 for all subsequent science frames.

---

### Summary table

| Stage | Free params | Fixed | Purpose |
|---|---|---|---|
| 0 | — | — | Read B, I₀, ring positions directly from data |
| 1 | cx, cy (geometric) | all Airy params | Ring center from ellipse/circle fitting |
| 2 | — (analytic) | — | Beat-period estimate of t (correct FSR valley) |
| 3 | t, R | everything else | Core Airy physics |
| 4 | I₀, I₁, I₂, σ₀ | t, R from Stage 3 | Vignetting + PSF |
| 5 | t, R, α, I₀, I₁, I₂, σ₀ | B, A₂, σ₁, σ₂ | Joint refinement |

---

## Key numbers to have on hand

| Quantity | Value | Notes |
|---|---|---|
| Ne λ₁ | 640.2248 nm | Fix — never float |
| Ne λ₂ | 638.2991 nm | Fix — never float |
| Δλ (beat) | 1.9257 nm | λ₁ - λ₂ |
| OI science | 630.0 nm | — |
| FSR at λ₁, t=20mm | λ²/(2t) ≈ 10.25 pm | One interference order |
| α (engineering) | 1.6×10⁻⁴ rad/px | 32 µm / 200 mm |
| t_physical | ~20.008 mm | ICOS measurement |
| t_FlatSat (WRONG) | 20.670 mm | FSR-period artifact — never use |
| NIST A₂ | ~0.67 | Need to verify from NIST database |
| Wind bias from 1 pm error in t | ~0.014 m/s | Well within 9.8 m/s budget |
| Wind bias from wrong FSR period | ~7 km/s | Catastrophic — why Stage 2 is mandatory |

---

## What to implement next

The agreed first implementation step is a **standalone diagnostic script** (no pipeline wrapping) that:

1. Loads `cal_image_L1_1.npy`
2. Runs Stage 0: equal-area radial binning from center of mass, plots radial profile, prints B and I₀ estimates
3. Runs Stage 1: circle-fitting to ring ridges, plots found center overlaid on 2D image
4. Runs Stage 2: beat-period analysis, plots r² vs ring order with fitted line, prints t_seed
5. **Produces a 3-panel diagnostic figure:** 2D image with center marked | radial profile with ring peaks annotated | r² plot with beat period fit

This diagnostic should be run and inspected **before any optimizer is called**. If the ring center, background, and t_seed are wrong at this stage, no amount of LM fitting will fix it downstream.

**File to create:** `pipeline/m05_cal_diagnostic.py` (or similar — confirm with Scott)  
**Data file:** `cal_image_L1_1.npy` (already in project)  
**Dependencies:** numpy, scipy, matplotlib — no pipeline imports needed yet

---

## Relationship to existing pipeline modules

| Module | Role | Status |
|---|---|---|
| `fpi/m01_airy_forward_model.py` | Forward model — use `airy_modified()` in Stage 3+ | Implemented |
| `fpi/m03_annular_reduction.py` | Should provide equal-area binning | Check implementation |
| `fpi/m05_calibration_inversion.py` | Will wrap this staged plan when complete | Spec exists, implementation pending |
| `fpi_airy_fit.py` | Legacy — do not extend | Has wrong vignetting model, wrong t_init |

**Important:** M03's `reduce_calibration_frame()` should be used for the annular reduction in production, but for the diagnostic script, implement the binning directly to avoid pipeline dependencies and keep the script self-contained and debuggable.

---

## Caveats and open questions

1. **NIST A₂ value needs verification.** Look up the exact oscillator strengths for Ne I 640.225 nm (2p₅3s → 2p₅3p transition, configuration 1s²2s²2p⁵(²P°₃/₂)3s → ...) and Ne I 638.299 nm. The ratio may deviate from simple statistical weights due to the lamp operating conditions.

2. **Is M03 producing equal-area bins?** The Harding prescription is explicit about this — each bin should contain approximately equal pixel count. If M03 uses equal-width bins, the noise per bin is non-uniform and the χ² weighting will be wrong. Verify before using M03 output for the calibration fit.

3. **The `alpha` degeneracy with `t`.** Monitor correlation coefficient in Stage 5. If |corr(t,α)| > 0.95, fix α at its engineering value and do not float it.

4. **Image is big-endian uint16** (`dtype = >u2`). Always cast with `arr.astype(np.float64)` before any arithmetic — numpy will silently overflow on uint16 subtraction.

5. **L1 vs L0:** This image is already dark-subtracted. Do not apply a dark frame correction again. Confirm the L1 processing steps applied upstream before fitting.
