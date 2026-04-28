# Z03 — Synthetic Calibration Image Generator

**Spec ID:** Z03  
**Tier:** 9 — Validation Testing  
**Module:** `z03_synthetic_calibration_image_generator.py`  
**Status:** Implemented — updated spec  
**Date:** 2026-04-22  
**Version:** 1.3.1  
**Author:** Scott Sewell / HAO  
**Repo:** `soc_sewell`  
**Dependencies:** S03 (physical constants), M01 (airy_modified), M02 (radial_profile_to_image), S19 (P01, metadata schema)

---

## Revision History

| Date | Version | Change |
|------|---------|--------|
| 2026-04-10 | 1.0 | Initial spec |
| 2026-04-10 | 1.0 | SNR quadratic corrected; Z03_OUTPUT_DIR env-var; 7 pytest cases |
| 2026-04-13 | 1.1 | Stage G simplified to 1×2 cal+dark; dark image added throughout |
| 2026-04-14 | 1.2 | **Major:** all 10 M05 inversion params user-prompted; `airy_modified()` replaces standalone helper; R and B_dc promoted from fixed to prompted |
| **2026-04-22** | **1.3** | **`f_mm` → `alpha` prompt; Option A I₀ fix; rel_638 default 0.8→0.58; Claude Code prompt appended** |
| **2026-04-22** | **1.3.1** | **Implementation housekeeping: corrects illustrative I_peak example in §5.3a; records constants corrections (R_MAX_PX, CX/CY_DEFAULT, N_META_ROWS) as LD-1. No algorithm changes. 15/15 tests pass.** |

---

## 1. Purpose

Z03 creates a **synthetic 'truth' calibration image pair** — a neon fringe calibration image and
a companion dark image — both in authentic WindCube `.bin` format (260 rows × 276 cols, uint16
little-endian), suitable for direct ingestion by Z01, F01, or any downstream pipeline module.

The script is **interactive**: it prompts the user for all instrument and calibration parameters
that the F01 inversion stage recovers from a real calibration image. Synthesis uses the full
PSF-broadened, vignetting-modulated Airy model (`M01.airy_modified()`) rather than an ideal
Airy function. This ensures that every degree of freedom visible to the fitter can be
independently set and recovered.

Both output images constitute **ground-truth artefacts** — every parameter used in synthesis is
known exactly and recorded in a companion `_truth.json` sidecar. The sidecar is designed to be
compared parameter-for-parameter with a `CalibrationResult` from F01.

---

## 2. Relationship to Z01, F01, and F02

| Aspect | Z01 | F01 | Z03 |
|--------|-----|-----|-----|
| Primary input | Real `.bin` + dark | 1D FringeProfile from Z01a | *(generates output)* |
| Output | Tolansky result (d, α, ε) | CalibrationResult (R, α, I₀, …) | `.bin` cal + dark + `_truth.json` |
| Core physics | Ring-order WLS | Two-line Airy LM fit | Two-line Airy synthesis |

Z03 is the **upstream complement** to Z01 and F01: it produces the controlled synthetic cal+dark
pair that those modules are designed to ingest and analyse. The `_truth.json` maps directly to
`CalibrationResult` fields for end-to-end round-trip validation.

---

## 3. Script Overview

```
z03_synthetic_calibration_image_generator.py
│
├── Stage A  — Banner and parameter prompting (interactive, 10 + 1 parameters)
├── Stage B  — Derive secondary optical parameters (I₀, FSR, finesse)
├── Stage C  — Build InstrumentParams; synthesise PSF-broadened two-line neon Airy
│             fringe image with vignetting (noise-free float)
│             I₀ = I_peak / (1 + rel_638) applied to EACH line separately
│             profile = Ã(r; λ₆₄₀, I₀) + rel_638 × Ã(r; λ₆₃₈, I₀) + B
├── Stage D  — Apply noise model → calibration image (uint16)
│             Synthesise dark image (noise-only, uint16)
├── Stage E  — Build S19-compliant metadata; embed into cal image rows 0–3
│             Build S19-compliant metadata; embed into dark image rows 0–3
├── Stage F  — Write cal .bin + dark .bin + _truth.json sidecar
└── Stage G  — Diagnostic display (1×2 figure)
```

---

## 4. User-Prompted Parameters

The script opens a terminal session and prompts sequentially in four groups. All 10 parameters
map directly to the free parameters recovered by F01 (CalibrationResult), plus one supplemental
source parameter (`rel_638`). This one-to-one correspondence is the primary design goal of v1.3.

### Group 1 — Etalon Geometry

| Prompt | Variable | Default | Units | Notes |
|--------|----------|---------|-------|-------|
| Etalon gap `d` | `d_mm` | `20.0006` | mm | Authoritative Tolansky/Benoit value (D_25C_MM) |
| Plate scale `α` | `alpha` | `1.6133e-4` | rad/px | Tolansky two-line WLS result (2×2 binned). Replaces `f_mm` from v1.2. |

> **Why `alpha` instead of `f_mm`?**  
> F01 fits `α` directly. Z03 previously prompted `f_mm` and derived `α = pix_m / f_mm` internally,
> creating an unnecessary indirection. Users cross-checking Z03 against F01 output had to convert
> between the two. The Tolansky analysis (Z01a) measures `α` directly from the r²-vs-P slope and
> reports it in those units; it never reports `f_mm`. Using `α` as the primary parameter makes the
> Z03 ↔ F01 ↔ Z01a comparison parameter-for-parameter.  
>
> The `f_mm` parameter is **not** stored in `SynthParams` or `_truth.json`. If needed for display,
> it can be recovered from `alpha` as `f_mm = pix_m / alpha × 1e3` (for display only).

### Group 2 — Etalon Reflectivity and PSF

| Prompt | Variable | Default | Units | Notes |
|--------|----------|---------|-------|-------|
| Plate reflectivity `R` | `R` | `0.53` | — | FlatSat-measured effective reflectivity |
| Average PSF width `σ₀` | `sigma0` | `0.5` | pixels | Shift-variant Gaussian broadening, mean |
| PSF sine variation `σ₁` | `sigma1` | `0.1` | pixels | Amplitude of sin(πr/r_max) term |
| PSF cosine variation `σ₂` | `sigma2` | `-0.05` | pixels | Amplitude of cos(πr/r_max) term |

### Group 3 — Intensity Envelope and Bias

| Prompt | Variable | Default | Units | Notes |
|--------|----------|---------|-------|-------|
| Peak signal-to-noise ratio | `snr_peak` | `50.0` | — | Sets composite I_peak via SNR quadratic (§5.3). See §5.3a for I₀ derivation. |
| Linear vignetting `I₁` | `I1` | `-0.1` | — | I(r) = I₀·(1 + I₁·(r/r_max) + I₂·(r/r_max)²) |
| Quadratic vignetting `I₂` | `I2` | `0.005` | — | Must keep I(r) > 0 for all r ≤ r_max |
| Bias pedestal `B` | `B_dc` | `300` | ADU | CCD bias + stray light floor |

### Group 4 — Source Parameter (supplemental)

| Prompt | Variable | Default | Units | Notes |
|--------|----------|---------|-------|-------|
| Intensity ratio 638 nm / 640 nm | `rel_638` | `0.58` | — | Y_B / Y_A; **updated from 0.8 to 0.58** — real WindCube FlatSat measurement from F01 v3.1 analysis |

> **Design note — 10 + 1 parameters:**  
> The 10 parameters in Groups 1–3 correspond exactly to the free parameters in F01 `CalibrationResult`:
> {d, α, R, σ₀, σ₁, σ₂, I₀, I₁, I₂, B}. `rel_638` is listed separately because it is a property
> of the Ne source, not of the instrument, and is treated as F01's fitted parameter `Y_B` (= `rel_638`
> times `Y_A`, with `Y_A = 1.0` by convention).

### 4.1 Prompt implementation pattern

```python
def _validated_prompt(label: str, default: float, units: str,
                      hard_min: float, hard_max: float,
                      warn_min: float, warn_max: float) -> float: ...
```

All prompts are wrapped in `try/except ValueError` with a re-prompt loop. Values outside hard
limits are rejected and force re-entry. Values inside hard limits but outside warning range print
an advisory and proceed after Y/n confirmation.

After all prompts, the script echoes a parameter summary table and asks Y/n before proceeding.

---

## 5. Physics: Two-Line Neon Fringe Synthesis

### 5.1 PSF-broadened Airy transmission function

Z03 uses `M01.airy_modified()` directly. This ensures the PSF broadening and vignetting
present in the synthesis are identical degrees of freedom to those fitted by F01.

```
A_ideal(r; λ, d, R, α, I₀, I₁, I₂) = I(r) / [1 + F·sin²(π·2d·cos(θ(r)) / λ)]
```

where:
- `θ(r) = arctan(α · r)`
- `I(r) = I₀ · (1 + I₁·(r/r_max) + I₂·(r/r_max)²)`
- `F = 4R / (1 − R)²`

### 5.2 Two neon reference lines

| Line | Wavelength λ (nm) | Relative intensity |
|------|-------------------|--------------------|
| Ne 640.2 nm | 640.2248 | 1.0 (reference) |
| Ne 638.3 nm | 638.2991 | `rel_638` (user-set) |

Source: Burns, Adams & Longwell (1950), IAU "S" standard.

### 5.3 Deriving I_peak from SNR (composite peak)

The `snr_peak` prompt specifies the peak SNR of the **combined two-line composite profile**,
measured at the first bright fringe maximum. The SNR quadratic gives the composite peak signal
above background:

```
SNR = I_peak / sqrt(I_peak + B + sigma_read²)

I_peak² − SNR²·I_peak − SNR²·(B + sigma_read²) = 0

I_peak = [SNR² + sqrt(SNR⁴ + 4·SNR²·(B + sigma_read²))] / 2   (positive root)
```

```python
def snr_to_ipeak(snr: float, B_dc: float, sigma_read: float) -> float:
    noise_floor = B_dc + sigma_read**2
    return (snr**2 + math.sqrt(snr**4 + 4 * snr**2 * noise_floor)) / 2
```

### 5.3a  I₀ per-line — the Option A fix (v1.3)

`I_peak` from §5.3 is the **combined** peak of the 640 + 638 nm profile. Because both Airy
functions are evaluated with the same `I₀` and the composite profile is:

```
S(r) = Ã(r; λ₆₄₀, I₀) + rel_638 × Ã(r; λ₆₃₈, I₀) + B
```

the composite peak above B is `I₀ × (1 + rel_638)` at perfect phase alignment. Therefore:

```
I₀ = I_peak / (1 + rel_638)
```

This is the value passed to `airy_modified()` for **both** lines. The secondary line is scaled
by `rel_638` at the **profile level** (not by inflating `I₀`).

**Why this matters:** In v1.2 and earlier, `I₀` was set equal to `I_peak`, which meant each
line was synthesised with the full composite-peak amplitude, overestimating both lines by the
factor `(1 + rel_638)`. The F01 fitter (which correctly uses separate per-line scales Y_A and
Y_B) would recover an `I₀` that is `1/(1 + rel_638)` times the Z03 synthesis value — creating
a systematic discrepancy in the round-trip validation. The fix ensures that:

```
Z03 truth: I₀ = I_peak / (1 + rel_638)
F01 fitted: I₀  ≡  same quantity (CalibrationResult.I0)
```

With the default `rel_638 = 0.58` and `snr_peak = 50`, `B_dc = 300`, `sigma_read = 50`:

```
noise_floor = B_dc + sigma_read² = 300 + 2500 = 2800 ADU²
I_peak = [50² + sqrt(50⁴ + 4·50²·2800)] / 2 ≈ 4176 ADU   (composite)
I₀     = 4176 / (1 + 0.58) ≈ 2643 ADU                     (per-line)
```

These are the values confirmed by the implementation (commit `15/15 pass, 2026-04-22`).
The earlier illustrative example of `I_peak ≈ 2380 ADU` in the v1.3 draft was incorrect —
it omitted the `sigma_read² = 2500` contribution to the noise floor.

> **Note:** `snr_peak` in the truth JSON refers to the composite-peak SNR (what the user entered).
> `I0_adu` is the derived per-line envelope amplitude actually passed to `airy_modified()`.
> A round-trip test comparing Z03 truth to F01 output should compare `F01.I0` against
> `truth["derived_params"]["I0_adu"]`, not against `I_peak`.

### 5.4 Plate scale

`α` is prompted directly (Group 1). The `pix_m / f` conversion is no longer performed.
The prompted value is used as-is in `airy_modified()`.

```python
alpha_rad_per_px = alpha    # direct from prompt; no f_mm conversion
```

The derived-params section of `_truth.json` no longer records `f_mm`. It records `alpha_rad_per_px`.

### 5.5 Composite profile

```
S_cal(r) = Ã(r; λ₆₄₀, d, R, α, I₀, I₁, I₂, σ₀, σ₁, σ₂)
         + rel_638 × Ã(r; λ₆₃₈, d, R, α, I₀, I₁, I₂, σ₀, σ₁, σ₂)
         + B
```

where `I₀ = I_peak / (1 + rel_638)` as defined in §5.3a.

### 5.6 Dark image model

```
I_dark(x, y) = B_dc  (uniform, before noise)
dark_noisy = clip(Poisson(B_dc) + Gaussian(0, sigma_read), 0, 16383).astype(uint16)
```

---

## 6. Fixed Default Parameters

| Parameter | Symbol | Default Value | Source |
|-----------|--------|--------------|--------|
| Read noise | `sigma_read` | `50.0` ADU | CCD97 EM gain regime estimate |
| Image centre (col) | `cx` | `137.5` | Geometric centre of 276-col array (276/2 − 0.5) |
| Image centre (row) | `cy` | `129.5` | Geometric centre of 260-row active region (260/2 − 0.5) |
| Pixel pitch (binned) | `pix_m` | `32.0e-6` m | CCD97 16 µm × 2×2 binning (informational only in v1.3) |
| Max usable radius | `r_max` | `110.0` px | FlatSat/flight value |
| Radial bins | `R_bins` | `2000` | Avoids interpolation artefacts |
| Image dimensions | `nrows, ncols` | `260, 276` | WindCube Level-0 standard |
| Metadata rows | `n_meta_rows` | `4` | S19 header occupies first 4 rows |
| Ne 640.2 nm wavelength | `lam_640` | `640.2248e-9` m | Burns et al. (1950) |
| Ne 638.3 nm wavelength | `lam_638` | `638.2991e-9` m | Burns et al. (1950) |
| Refractive index | `n_ref` | `1.0` | Air gap |

> `pix_m` is retained as a fixed constant for documentation purposes but is no longer used in
> any calculation. `alpha` is now a direct user prompt.

---

## 7. Stage Descriptions

### Stage A — Banner and Parameter Prompting

```
╔══════════════════════════════════════════════════════════════╗
║  Z03  Synthetic Calibration Image Generator  v1.3            ║
║  WindCube SOC — soc_sewell                                   ║
╚══════════════════════════════════════════════════════════════╝

Synthesises a matched calibration + dark image pair in authentic
WindCube .bin format. Parameters correspond one-to-one with the
F01 CalibrationResult free parameters.
Press <Enter> to accept the default shown in parentheses.

──────────────────────────────────────────────────────────────
 GROUP 1  ETALON GEOMETRY
──────────────────────────────────────────────────────────────
  Etalon gap d [mm]                       (default 20.0006):
  Plate scale alpha [rad/px]              (default 1.6133e-4):

──────────────────────────────────────────────────────────────
 GROUP 2  ETALON REFLECTIVITY AND PSF
──────────────────────────────────────────────────────────────
  Plate reflectivity R                    (default 0.53):
  Average PSF width sigma_0 [pixels]      (default 0.5):
  PSF sine variation sigma_1 [pixels]     (default 0.1):
  PSF cosine variation sigma_2 [pixels]   (default -0.05):

──────────────────────────────────────────────────────────────
 GROUP 3  INTENSITY ENVELOPE AND BIAS
──────────────────────────────────────────────────────────────
  Peak SNR (composite 640+638 nm peak)    (default 50.0):
  Linear vignetting coefficient I_1       (default -0.1):
  Quadratic vignetting coefficient I_2    (default 0.005):
  Bias pedestal B [ADU]                   (default 300):

──────────────────────────────────────────────────────────────
 GROUP 4  SOURCE (not a direct inversion free parameter)
──────────────────────────────────────────────────────────────
  Intensity ratio 638nm/640nm (rel_638)   (default 0.58):
```

### Stage B — Derive Secondary Parameters

```python
I_peak = snr_to_ipeak(snr_peak, B_dc, sigma_read)   # composite peak above B
I0     = I_peak / (1.0 + rel_638)                   # per-line amplitude (Option A fix)
FSR_m  = lam_640**2 / (2.0 * d_mm * 1e-3)
F_coef = 4 * R / (1 - R)**2
N_R    = math.pi * math.sqrt(R) / (1 - R)
```

Printed to terminal for verification.

**Stage B also validates:**
- PSF positivity: `sigma0 >= sqrt(sigma1² + sigma2²)` — hard error if violated
- Vignetting positivity: `I(r) > 0` for r ∈ {0, r_max/2, r_max, vertex} — hard error if violated

### Stage C — Synthesise Fringe Image

```python
inst = InstrumentParams(
    t      = d_mm * 1e-3,
    R_refl = R,
    n      = n_ref,
    alpha  = alpha,          # direct from prompt — no f_mm conversion
    I0     = I0,             # I_peak / (1 + rel_638)  ← Option A fix
    I1     = I1,
    I2     = I2,
    sigma0 = sigma0,
    sigma1 = sigma1,
    sigma2 = sigma2,
    B      = B_dc,
    r_max  = R_MAX_PX,
)

r_grid = np.linspace(0.0, R_MAX_PX, R_BINS)

A640 = airy_modified(r_grid, lam_640, inst.t, inst.R_refl, inst.alpha,
                     inst.n, inst.r_max, inst.I0, inst.I1, inst.I2,
                     inst.sigma0, inst.sigma1, inst.sigma2)

A638 = airy_modified(r_grid, lam_638, inst.t, inst.R_refl, inst.alpha,
                     inst.n, inst.r_max, inst.I0, inst.I1, inst.I2,
                     inst.sigma0, inst.sigma1, inst.sigma2)

profile_1d = A640 + rel_638 * A638 + inst.B
```

Note that `I0` is the **per-line** amplitude. The composite peak above B is `I0 × (1 + rel_638)`,
which equals `I_peak` as computed from `snr_peak`. Both lines share the same `I0`, `I1`, `I2`,
`sigma0`, `sigma1`, `sigma2` — consistent with the F01 forward model.

### Stages D–G — unchanged from v1.2

(Noise model, metadata, file write, diagnostic display — see v1.2 spec for details.)

**Stage G title update for v1.3:**  
```
Line 1: f"d={d_mm} mm   α={alpha:.4e} rad/px   R={R}   SNR={snr_peak}"
Line 2: f"σ₀={sigma0}   I₁={I1}   I₂={I2}   B={B_dc} ADU"
Line 3: f"Ne: 640.2248 nm (×1.0)   638.2991 nm (×{rel_638})"
Line 4: f"I₀ (per-line) = {I0:.1f} ADU   I_peak (composite) = {I_peak:.1f} ADU"
```

---

## 8. Output Files Summary

| File | Description |
|------|-------------|
| `yyyymmddThhmmssZ_cal_synth_z03.bin` | Synthetic calibration image |
| `yyyymmddThhmmssZ_dark_synth_z03.bin` | Synthetic dark image |
| `yyyymmddThhmmssZ_cal_synth_z03_truth.json` | Ground-truth sidecar |
| `yyyymmddThhmmssZ_cal_synth_z03_diagnostic.png` | 1×2 figure |

---

## 9. `_truth.json` Schema (v1.3)

```json
{
  "z03_version": "1.3",
  "timestamp_utc": "2026-04-22T00:00:00Z",
  "random_seed": 12345678,
  "user_params": {
    "d_mm":      20.0006,
    "alpha":     1.6133e-4,
    "R":         0.53,
    "sigma0":    0.5,
    "sigma1":    0.1,
    "sigma2":   -0.05,
    "snr_peak":  50.0,
    "I1":       -0.1,
    "I2":        0.005,
    "B_dc":      300,
    "rel_638":   0.58
  },
  "derived_params": {
    "alpha_rad_per_px":      1.6133e-4,
    "I_peak_adu":            4176.2,
    "I0_adu":                2643.2,
    "Y_B":                   0.58,
    "FSR_m":                 1.0238e-11,
    "finesse_coefficient_F": 9.63,
    "finesse_N":             4.88
  },
  "fixed_defaults": {
    "sigma_read": 50.0,
    "cx":         137.5,
    "cy":         129.5,
    "pix_m":      3.2e-5,
    "r_max_px":   110.0,
    "R_bins":     2000,
    "nrows":      260,
    "ncols":      276,
    "n_ref":      1.0,
    "lam_640_m":  6.402248e-7,
    "lam_638_m":  6.382991e-7
  },
  "output_cal_file":  "20260422T000000Z_cal_synth_z03.bin",
  "output_dark_file": "20260422T000000Z_dark_synth_z03.bin"
}
```

**F01 round-trip comparison map:**

| `_truth.json` field | F01 `CalibrationResult` field | Notes |
|--------------------|-----------------------------|-------|
| `user_params.d_mm × 1e-3` | `result.t_m` | Fixed in F01; should match exactly |
| `user_params.alpha` | `result.alpha` | Free in F01; should agree within 2σ |
| `user_params.R` | `result.R_refl` | Free in F01 |
| `derived_params.I0_adu` | `result.I0` | **Use I0_adu, not snr_peak** |
| `user_params.I1` | `result.I1` | Free in F01 |
| `user_params.I2` | `result.I2` | Free in F01 |
| `user_params.sigma0` | `result.sigma0` | Free in F01 |
| `user_params.sigma1` | `result.sigma1` | Free in F01 |
| `user_params.sigma2` | `result.sigma2` | Free in F01 |
| `user_params.B_dc` | `result.B` | Free in F01 |
| `derived_params.Y_B` | `result.intensity_ratio` | = rel_638 in synthesis |

---

## 10. Parameter Validation and Bounds

### Group 1 — Etalon Geometry

| Parameter | Hard min | Hard max | Warning range |
|-----------|----------|----------|---------------|
| `d_mm` | 15.0 | 25.0 | Outside (19.0, 21.5) |
| `alpha` | 1e-5 | 1e-3 | Outside (0.5e-4, 5e-4) |

### Group 2 — Reflectivity and PSF

| Parameter | Hard min | Hard max | Warning range |
|-----------|----------|----------|---------------|
| `R` | 0.01 | 0.99 | Outside (0.3, 0.85) |
| `sigma0` | 0.0 | 5.0 | Outside (0.0, 2.0) |
| `sigma1` | -3.0 | 3.0 | Outside (-1.0, 1.0) |
| `sigma2` | -3.0 | 3.0 | Outside (-1.0, 1.0) |

### Group 3 — Intensity Envelope and Bias

| Parameter | Hard min | Hard max | Warning range |
|-----------|----------|----------|---------------|
| `snr_peak` | 1.0 | 500.0 | Outside (10.0, 200.0) |
| `I1` | -0.9 | 0.9 | Outside (-0.5, 0.5) |
| `I2` | -0.9 | 0.9 | Outside (-0.5, 0.5) |
| `B_dc` | 0 | 50000 | Outside (100, 10000) |

> B_dc upper hard limit raised from 5000 → 50000 to accommodate real WindCube data
> (real neon calibration B ≈ 6100 ADU).

### Group 4 — Source

| Parameter | Hard min | Hard max | Warning range |
|-----------|----------|----------|---------------|
| `rel_638` | 0.0 | 2.0 | Outside (0.2, 1.5) |

---

## 11. SynthParams Dataclass (v1.3)

```python
@dataclass
class SynthParams:
    # Group 1 — etalon geometry
    d_mm:      float   # etalon gap, mm
    alpha:     float   # plate scale, rad/px  ← replaces f_mm
    # Group 2 — reflectivity and PSF
    R:         float
    sigma0:    float
    sigma1:    float
    sigma2:    float
    # Group 3 — intensity envelope and bias
    snr_peak:  float
    I1:        float
    I2:        float
    B_dc:      float
    # Group 4 — source
    rel_638:   float
```

---

## 12. Python Module Structure

```
z03_synthetic_calibration_image_generator.py
│
├── IMPORTS      from m01_airy_forward_model import InstrumentParams, airy_modified
│                from m02_calibration_synthesis import radial_profile_to_image
│
├── CONSTANTS    SIGMA_READ, PIX_M, CX_DEFAULT, CY_DEFAULT, R_MAX_PX, R_BINS,
│                N_REF, NROWS, NCOLS, N_META_ROWS, LAM_640, LAM_638
│                (PIX_M retained for documentation; not used in calculations)
│
├── def _validated_prompt(...)
├── def prompt_all_params()           → SynthParams  (11 fields)
├── def snr_to_ipeak(...)             SNR quadratic positive root
├── def check_psf_positive(...)
├── def check_vignetting_positive(...)
├── def derive_secondary(params)      → DerivedParams
│                                       (alpha_rad_per_px, I_peak, I0, Y_B,
│                                        FSR_m, finesse_F, finesse_N)
├── def build_instrument_params(...)  → InstrumentParams
├── def synthesise_profile(...)       → (profile_1d, r_grid)
│                                       uses I0 = I_peak / (1 + rel_638)
├── def synthesise_image(...)
├── def add_noise(...)
├── def synthesise_dark(...)
├── def build_s19_metadata(...)
├── def embed_metadata(...)
├── def write_bin(...)
├── def write_truth_json(...)
├── def make_diagnostic_figure(...)
└── def main()
```

---

## 13. Test Verification

| Test ID | Description | Pass criterion |
|---------|-------------|---------------|
| `test_cal_output_shape` | Cal `.bin` loads to (260, 276) uint16 | Shape matches exactly |
| `test_dark_output_shape` | Dark `.bin` loads to (260, 276) uint16 | Shape matches exactly |
| `test_cal_metadata_round_trip` | S19 metadata recoverable | All written fields match |
| `test_dark_metadata_round_trip` | Dark metadata correct | `image_type="Dark"`, `shutter_status="Closed"` |
| `test_fringe_peak_location` | Peak at image centre | Within 1 px of (cy, cx) |
| `test_snr_achieved` | Peak SNR ≈ requested | Within ±20% of target |
| `test_rel_638_ratio` | Family amplitude ratio ≈ rel_638 | Within ±5% |
| `test_dark_no_fringes` | Dark has no fringe structure | No periodic structure |
| `test_truth_json_complete` | All required keys in sidecar | See §13a |
| `test_default_params` | Script runs with all defaults | Both .bin + truth.json written |
| `test_psf_broadening_effect` | sigma0 > 0 broadens fringes | std(broad) < std(sharp) |
| `test_vignetting_effect` | I1 ≠ 0 produces radial gradient | inner/outer ratio differs by ≥ 1% |
| **`test_I0_option_a`** | **I0_adu = I_peak / (1 + rel_638)** | **Exact to 6 significant figures** |
| **`test_round_trip_I0`** | **Z03 truth I0_adu matches F01 recovered I0** | **Within 5% on synthetic profile** |
| **`test_alpha_no_f_mm`** | **f_mm not present in truth JSON** | **Key absent from user_params and derived_params** |

### 13a — Updated `test_truth_json_complete`

```python
expected_user_keys = {
    "d_mm", "alpha", "R", "sigma0", "sigma1", "sigma2",
    "snr_peak", "I1", "I2", "B_dc", "rel_638"
}
expected_derived_keys = {
    "alpha_rad_per_px", "I_peak_adu", "I0_adu", "Y_B",
    "FSR_m", "finesse_coefficient_F", "finesse_N"
}
# assert "f_mm" not in truth["user_params"]
# assert "f_mm" not in truth["derived_params"]
```

---

## 14. Known Limitations

- Fixed image centre: `(cx, cy)` is fixed at the geometric centre.
- No EM gain excess noise factor.
- `r_max` not prompted; fixed at 110 px.
- `pix_m` retained as a constant but is no longer used in any calculation.

---

## 14a. Locked Decisions (implementation-phase findings)

**LD-1: Authoritative fixed-constant values (confirmed by implementation, 2026-04-22).**  
The following constants were corrected during the v1.3 implementation to match the
actual WindCube FlatSat geometry and S19 header format. Do not revert these values.

| Constant | v1.2 value | v1.3 correct value | Source |
|----------|-----------|-------------------|--------|
| `R_MAX_PX` | 175 | **110** | FlatSat/flight usable radius (spec §6) |
| `CX_DEFAULT` | 145 | **137.5** | Geometric centre of 276-col array (276/2 − 0.5) |
| `CY_DEFAULT` | 145 | **129.5** | Geometric centre of 260-row active region (260/2 − 0.5) |
| `N_META_ROWS` | 1 | **4** | S19 header occupies first 4 rows |

These corrections were required for `test_round_trip_I0` (uses `r_max_px=110.0` in the
FringeProfile passed to F01) and for `test_cal_metadata_round_trip` (reads back 4 header rows).

**LD-2: sigma_read = 50 ADU must be included in the noise floor for the SNR quadratic.**  
`noise_floor = B_dc + sigma_read²`. With `sigma_read = 50`, `sigma_read² = 2500 ADU²`,
which dominates the noise floor for typical `B_dc = 300 ADU`. Any illustrative calculation
of `I_peak` must include this term. The v1.3 draft erroneously showed `I_peak ≈ 2380 ADU`
(omitting sigma_read); the correct value at default params is `I_peak ≈ 4176 ADU`.

---

## 15. Claude Code Implementation Prompt

> **Instructions for Claude Code:**  
> Read this entire file first (§§1–14). Then execute the tasks below in order.
> Gate on pytest passing between tasks. Do not implement anything not described here.

```
cat PIPELINE_STATUS.md

# Read the full spec before touching any code:
cat docs/specs/z03_synthetic_calibration_image_generator_spec_2026-04-22.md

# Read these existing files for context — do NOT modify them unless a task says to:
cat src/fpi/z03_synthetic_calibration_image_generator.py
cat src/fpi/m01_airy_forward_model_2026_04_05.py
cat tests/test_z03.py
cat src/constants.py | grep -E "NE_WAVELENGTH|PLATE_SCALE|D_25C"

# ── TASK 1 of 5: Replace f_mm with alpha in SynthParams and prompt ─────────
#
# In z03_synthetic_calibration_image_generator.py:
#
# 1a. In SynthParams dataclass: replace field `f_mm: float` with `alpha: float`.
#     Update the docstring/comment to: "plate scale, rad/px".
#
# 1b. In prompt_all_params(): replace the f_mm prompt with:
#       alpha = _validated_prompt(
#           "Plate scale alpha [rad/px]",
#           default=1.6133e-4,
#           units="rad/px",
#           hard_min=1e-5, hard_max=1e-3,
#           warn_min=0.5e-4, warn_max=5e-4,
#       )
#     Remove the f_mm prompt entirely.
#
# 1c. In derive_secondary(): remove the line
#       alpha_rad_per_px = PIX_M / (params.f_mm * 1e-3)
#     Replace with:
#       alpha_rad_per_px = params.alpha
#     Update DerivedParams to remove f_mm if it was stored there.
#
# 1d. In build_instrument_params(): the alpha field already uses derived.alpha_rad_per_px
#     so no change needed there — just verify.
#
# 1e. In write_truth_json(): update user_params dict key from "f_mm" to "alpha".
#     The derived_params section already records "alpha_rad_per_px" — no change there.
#
# 1f. In make_diagnostic_figure(): update title Line 1 from
#       f"d={d_mm} mm   f={f_mm} mm   R={R}   SNR={snr_peak}"
#     to:
#       f"d={d_mm} mm   α={alpha:.4e} rad/px   R={R}   SNR={snr_peak}"
#
# Gate: pytest tests/test_z03.py -v
# Any test that references f_mm must be updated to use alpha instead.

# ── TASK 2 of 5: Apply the Option A I₀ fix ────────────────────────────────
#
# This is the core change. In derive_secondary():
#
#   BEFORE:
#     I_peak = snr_to_ipeak(params.snr_peak, params.B_dc, SIGMA_READ)
#     derived.I0 = I_peak                     # WRONG — inflated by (1 + rel_638)
#
#   AFTER:
#     I_peak = snr_to_ipeak(params.snr_peak, params.B_dc, SIGMA_READ)
#     I0     = I_peak / (1.0 + params.rel_638)  # per-line amplitude
#     derived.I_peak = I_peak                    # composite peak — store separately
#     derived.I0     = I0                        # per-line amplitude
#     derived.Y_B    = params.rel_638            # convenience alias for truth JSON
#
# Update DerivedParams (dataclass or namedtuple) to carry BOTH I_peak and I0
# as separate fields.
#
# In synthesise_profile(): the call to build_instrument_params() already passes
# derived.I0 into InstrumentParams.I0, so the airy_modified() calls will
# automatically use the corrected per-line I0. No change needed in synthesise_profile
# itself — just verify that inst.I0 == derived.I0 == I_peak/(1+rel_638).
#
# Update write_truth_json() to add both I_peak_adu and I0_adu to derived_params:
#   "I_peak_adu": derived.I_peak,
#   "I0_adu":     derived.I0,
#   "Y_B":        derived.Y_B,
#
# Update the parameter summary table printed after prompts to show:
#   I_peak (composite) = {I_peak:.1f} ADU
#   I0     (per-line)  = {I0:.1f}     ADU   [= I_peak / (1 + rel_638)]
#
# Update Stage G title line 4:
#   f"I₀ (per-line) = {derived.I0:.1f} ADU   I_peak (composite) = {derived.I_peak:.1f} ADU"
#
# Gate: pytest tests/test_z03.py -v

# ── TASK 3 of 5: Update rel_638 default ───────────────────────────────────
#
# Change the default for rel_638 from 0.8 to 0.58 everywhere it appears:
#   - In _validated_prompt() call for rel_638
#   - In any test fixtures that use the default
#   - In the banner text if it shows the default value
#
# Also update the B_dc hard_max from 5000 to 50000 (spec §10, Group 3).
#
# Gate: pytest tests/test_z03.py -v

# ── TASK 4 of 5: Add three new tests ──────────────────────────────────────
#
# In tests/test_z03.py, add these three tests:
#
# TEST: test_I0_option_a
#   from src.fpi.z03_synthetic_calibration_image_generator import (
#       snr_to_ipeak, derive_secondary, SynthParams, SIGMA_READ
#   )
#   params = SynthParams(d_mm=20.0006, alpha=1.6133e-4, R=0.53,
#                        sigma0=0.5, sigma1=0.0, sigma2=0.0,
#                        snr_peak=50.0, I1=-0.1, I2=0.005,
#                        B_dc=300.0, rel_638=0.58)
#   derived = derive_secondary(params)
#   I_peak_expected = snr_to_ipeak(50.0, 300.0, SIGMA_READ)
#   I0_expected = I_peak_expected / (1.0 + 0.58)
#   assert abs(derived.I0 - I0_expected) / I0_expected < 1e-6, \
#       f"I0 mismatch: {derived.I0} vs {I0_expected}"
#   assert abs(derived.I_peak - I_peak_expected) / I_peak_expected < 1e-6
#
# TEST: test_round_trip_I0
#   This test synthesises a 1D profile with Z03 parameters, then runs
#   the F01 fitter on it and checks that F01 recovers I0 within 5%
#   of the Z03 truth value.
#
#   from src.fpi.z03_synthetic_calibration_image_generator import (
#       derive_secondary, build_instrument_params, synthesise_profile, SynthParams
#   )
#   from src.fpi.f01_full_airy_fit_to_neon_image_2026_04_22 import (
#       fit_neon_fringe, TolanskyResult
#   )
#   from types import SimpleNamespace
#   import numpy as np
#
#   params = SynthParams(d_mm=20.0006, alpha=1.6133e-4, R=0.53,
#                        sigma0=0.5, sigma1=0.0, sigma2=0.0,
#                        snr_peak=50.0, I1=-0.1, I2=0.005,
#                        B_dc=300.0, rel_638=0.58)
#   derived = derive_secondary(params)
#   inst    = build_instrument_params(params, derived)
#   profile_1d, r_grid = synthesise_profile(inst, params.rel_638)
#
#   rng   = np.random.default_rng(42)
#   noisy = rng.poisson(np.maximum(profile_1d, 1)).astype(np.float32)
#   sigma = np.maximum(np.sqrt(noisy) / 8.0, 1.0).astype(np.float32)
#
#   fringe = SimpleNamespace(
#       r_grid=r_grid.astype(np.float32), r2_grid=(r_grid**2).astype(np.float32),
#       profile=noisy, sigma_profile=sigma,
#       masked=np.zeros(len(r_grid), dtype=bool),
#       r_max_px=110.0, quality_flags=0,
#   )
#   tolansky = TolanskyResult(t_m=20.0006e-3, alpha_rpx=1.6133e-4,
#                              epsilon_640=0.7735, epsilon_638=0.2711,
#                              epsilon_cal=0.22)
#   result = fit_neon_fringe(fringe, tolansky)
#
#   truth_I0 = derived.I0
#   assert abs(result.I0 - truth_I0) / truth_I0 < 0.05, \
#       f"F01 I0={result.I0:.1f} vs Z03 truth I0={truth_I0:.1f} (>{5}% error)"
#
# TEST: test_alpha_no_f_mm
#   from src.fpi.z03_synthetic_calibration_image_generator import (
#       derive_secondary, SynthParams
#   )
#   import json, tempfile, pathlib
#   # Run a minimal synthesis and check truth JSON has alpha not f_mm
#   params = SynthParams(d_mm=20.0006, alpha=1.6133e-4, R=0.53,
#                        sigma0=0.5, sigma1=0.0, sigma2=0.0,
#                        snr_peak=50.0, I1=0.0, I2=0.0,
#                        B_dc=300.0, rel_638=0.58)
#   derived = derive_secondary(params)
#   # Build a minimal truth dict directly (don't write files in unit test)
#   user_keys = set(params.__dataclass_fields__.keys())
#   assert "alpha"  in user_keys
#   assert "f_mm"  not in user_keys
#   assert derived.alpha_rad_per_px == params.alpha
#
# Gate: pytest tests/test_z03.py -v — all 15 tests must pass.

# ── TASK 5 of 5: Update PIPELINE_STATUS.md and commit ─────────────────────
#
# Update PIPELINE_STATUS.md: set Z03 version to 1.3, note the three changes.
#
git add src/fpi/z03_synthetic_calibration_image_generator.py \
        tests/test_z03.py \
        docs/specs/z03_synthetic_calibration_image_generator_spec_2026-04-22.md \
        PIPELINE_STATUS.md
git commit -m "feat: Z03 v1.3.1 — housekeeping, corrected constants and I_peak example

Corrects R_MAX_PX (175→110), CX/CY_DEFAULT (145→137.5/129.5),
N_META_ROWS (1→4) per LD-1. Corrects I_peak illustrative example
in spec §5.3a (sigma_read² omitted in v1.3 draft; correct value
4176 ADU not 2380 ADU). No algorithm changes. 15/15 tests pass.

Also updates PIPELINE_STATUS.md"

# ── REPORT BACK ────────────────────────────────────────────────────────────
#
# Report:
# 1. Full pytest output for all 15 tests (pass/fail for each)
# 2. The derived.I0 and derived.I_peak values for default params
#    (d=20.0006, alpha=1.6133e-4, R=0.53, snr_peak=50, B_dc=300, rel_638=0.58)
# 3. Full pytest summary line
# 4. Whether test_round_trip_I0 required importing F01 — confirm which
#    F01 module path was used
# 5. Any deviations from this spec
```

---

*End of Z03 Spec v1.3.1 — 2026-04-22*
