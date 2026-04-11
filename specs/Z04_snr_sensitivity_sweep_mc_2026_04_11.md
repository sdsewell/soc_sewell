# Z04 — SNR Sensitivity Sweep Monte Carlo Validation

**Spec:** Z04  
**Module:** `validation/z04_snr_sensitivity_sweep.py`  
**Tier:** 9 — Validation  
**Status:** Written — ready for Claude Code implementation  
**Date:** 2026-04-11  
**Author:** S. Sewell, HAO/NCAR  
**Depends on:** S09 (M01), S11 (M04), S12 (M03), S13 (Tolansky), S14 (M05), S15 (M06)  
**Analogous to:** Harding et al. 2014 §4, Figs. 7–8

---

## 1. Purpose

Z04 is a Monte Carlo validation script that characterises the WindCube FPI pipeline's velocity retrieval bias and uncertainty as a function of signal-to-noise ratio (SNR).  It answers the ORR question: *"Does the pipeline produce unbiased LOS velocity estimates over the SNR range expected on orbit, and are the reported uncertainties accurate?"*

The script is directly analogous to Harding et al. (2014) Simulations 1–3 (§4, Figs. 6–8), translated to the WindCube instrument model and the **delta-function source approximation** used throughout the WindCube pipeline (S11, M04).  Because the delta-function model eliminates the temperature free parameter, Z04 targets only LOS velocity recovery — simplifying Harding's 2D (v, T) bias plots to 1D bias-vs-SNR curves.

**ORR acceptance gates** (§7) must pass before launch operations begin (target: May 2026 ORR).

---

## 2. Scientific Background

### 2.1 SNR definition (Harding convention)

Following Harding et al. (2014) Eq. (17):

```
SNR = ΔS / σ_N
```

where `ΔS` is the peak-to-trough amplitude of the annular-reduced 1D fringe profile (in ADU), and `σ_N` is the standard deviation of the added Gaussian white noise (in ADU).

This definition is instrument-level: it reflects the contrast of the OI 630 nm fringe as seen by the detector, not the photon flux from the thermosphere.  Expected on-orbit SNR for the WindCube FPI at 250 km emission altitude is 1–5 based on radiometric modelling (Science Traceability Matrix).

### 2.2 Why noise model is Gaussian (not Poisson) for science frames

Calibration frames (neon lamp) are photon-noise limited → Poisson noise added in Z04's calibration channel.  Science (airglow) frames are dominated by detector dark current and read noise at typical exposure times → Gaussian white noise is the correct model for the science fringe, consistent with Harding et al. §4 and with WindCube's Teledyne CCD97 EMCCD noise characteristics.

### 2.3 Velocity path in the WindCube pipeline

The pipeline recovers LOS velocity via the following chain, all fixed in S13/S14/S15/M06:

```
Science frame (2D image)
  → M03: annular reduction → 1D fringe profile S(r²)
  → Tolansky WLS fit: recover ε_sci (fractional fringe order offset)
  → Integer order: m₀ = round(2 d n / λ₀)
  → Line centre: λ_c = 2 d n / (m₀ + ε_sci)
  → Doppler: v_LOS = c (λ_c − λ₀) / λ₀
```

Fixed priors fed in from S13/S14:
- `d` = 20.106 mm (operational gap, two-line neon calibration)
- `f` = 199.12 mm (measured focal length)
- `α` = 1.6071 × 10⁻⁴ rad/px (plate scale, 2×2 binned)
- `R_refl` = 0.53 (FlatSat reflectivity)
- `λ₀` = 630.0304 nm (OI 630 nm rest wavelength)

### 2.4 Expected performance (pre-simulation estimate)

At SNR = 5 (good airglow night), the fractional order sensitivity is:

```
Δε / ε ≈ 1/SNR × 1/√N_rings
```

For WindCube (~8 resolved rings, SNR = 5): σ_ε ≈ 0.003, propagating to σ_v ≈ 7–12 m/s.  The ORR requirement is σ_v ≤ 15 m/s at SNR ≥ 2.

---

## 3. Simulations

Z04 runs three independent Monte Carlo simulations in sequence, mirroring Harding §4.A–C.

### Sim-1: Uncertainty Calibration

**Purpose:** Verify that the pipeline's reported velocity uncertainty σ_est accurately matches the true scatter σ_v (i.e., that error bars are not over- or under-estimated).

| Parameter | Value |
|-----------|-------|
| True v_LOS | 100 m/s (fixed) |
| SNR | 5 (fixed) |
| N trials | 10 000 |
| Noise model | Gaussian white on science frame; Poisson on cal frame |

**Measured quantities per trial:**
- `v_rec`: recovered LOS velocity (m/s)
- `σ_est`: pipeline-reported 1σ uncertainty on v_rec

**Statistics computed:**
- `bias` = mean(v_rec − v_true)
- `σ_v` = std(v_rec − v_true)   (actual scatter)
- `mean_σ_est` = mean(σ_est)   (reported uncertainty)
- `σ_ratio` = mean_σ_est / σ_v  (should be 1.0 ± 0.20)
- Fraction of trials where |v_rec − v_true| < σ_est  (should be ~68%)
- K–S test of (v_rec − v_true) / σ_est against N(0,1)  (p > 0.05)

**Output panel:** Scatter plot of v_rec vs trial index; histogram of (v_rec − v_true) with Gaussian overlay; printed σ_ratio.

---

### Sim-2: Bias vs LOS Velocity

**Purpose:** Check that no velocity-dependent bias exists across the expected thermospheric wind range (−300 to +300 m/s).

| Parameter | Value |
|-----------|-------|
| True v_LOS | Uniform random ∈ [−300, +300] m/s |
| SNR | 5 (fixed) |
| N trials | 1 000 |

**Statistics computed per trial:**
- velocity error = v_rec − v_true

**Plot:** velocity error vs v_true (Harding Fig. 7 analogue).  Overplot ±5 m/s acceptance band.  A linear fit to the error-vs-velocity relationship is reported; slope must be < 0.01 (i.e., < 1% gain error).

---

### Sim-3: Bias vs SNR  *(ORR gate)*

**Purpose:** Characterise bias and σ_v as a function of SNR over the on-orbit range.  This is the primary ORR deliverable.

| Parameter | Value |
|-----------|-------|
| True v_LOS | Uniform random ∈ [−200, +200] m/s |
| SNR grid | [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0] |
| N trials per SNR bin | 10 000 / 9 bins ≈ 1 111 per bin; script allocates evenly |
| Total trials | 10 000 |

**Statistics computed per SNR bin:**
- `bias(SNR)` = mean(v_rec − v_true)
- `σ_v(SNR)` = std(v_rec − v_true)
- `mean_σ_est(SNR)` = mean(σ_est)
- 68th-percentile absolute error

**Plot:** Two-panel figure (Harding Fig. 8 analogue):
- Panel A: bias(SNR) with ±1σ error bars; dashed ±5 m/s ORR acceptance band
- Panel B: σ_v(SNR) solid; mean_σ_est(SNR) dashed; horizontal line at 15 m/s ORR limit

Both panels share a common x-axis (log scale, SNR).  Vertical grey band marks the ORR SNR threshold (SNR = 2).

---

## 4. Implementation Architecture

### 4.1 File location

```
validation/z04_snr_sensitivity_sweep.py
```

This follows the Z02/Z03 convention of placing validation scripts in `validation/`.

### 4.2 Module imports

Z04 calls existing production modules directly — it does not re-implement physics:

```python
from soc_sewell.m04_airglow_synthesis import synthesize_airglow_frame
from soc_sewell.m03_annular_reduction  import reduce_science_frame
from soc_sewell.m06_airglow_inversion  import invert_airglow_fringe
from soc_sewell.constants              import PHY   # λ₀, c, d, f, α, R_refl
```

If M06 is not yet fully implemented, Z04 falls back to the Tolansky sub-pipeline (S13 path) as used in Z02's `_run_tolansky_stage()`, with a compatibility shim documented in §4.4.

### 4.3 Per-trial function signature

```python
def _run_single_trial(
    v_true_ms: float,
    snr: float,
    rng: np.random.Generator,
    priors: dict,
) -> dict:
    """
    Synthesise one noisy airglow frame, reduce, invert, return results.

    Returns
    -------
    dict with keys:
        v_rec    : float  recovered LOS velocity (m/s)
        sigma_est: float  pipeline-reported 1σ uncertainty (m/s)
        eps      : float  recovered fractional order ε
        lam_c    : float  recovered line centre (nm)
        converged: bool   True if fit converged
    """
```

### 4.4 SNR injection

SNR is injected by scaling the Gaussian noise added to the synthetic science frame:

```python
# After synthesise_airglow_frame():
delta_S = profile.max() - profile.min()   # fringe contrast (ADU)
sigma_N = delta_S / snr                    # noise std (ADU)
noisy_frame = frame + rng.normal(0, sigma_N, frame.shape)
```

This exactly implements the Harding SNR definition (§2.1).

Poisson noise on the calibration frame is added as:

```python
cal_frame_noisy = rng.poisson(cal_frame.astype(float)).astype(np.float32)
```

### 4.5 Reproducibility

A global `rng = np.random.default_rng(seed=42)` is used for all three simulations, ensuring bit-exact reproducibility.  The seed is exposed as a CLI argument `--seed`.

### 4.6 Parallelism

`joblib.Parallel(n_jobs=-1)` wraps the per-trial loop for Sim-3 (largest simulation).  Each worker receives an independent derived RNG child generator via `rng.spawn(N_workers)`.

---

## 5. Output Artifacts

All outputs written to `validation/outputs/z04/`:

| File | Description |
|------|-------------|
| `z04_sim1_scatter.png` | Sim-1 scatter + histogram panel |
| `z04_sim2_bias_vs_velocity.png` | Harding Fig. 7 analogue |
| `z04_sim3_bias_vs_snr.png` | Harding Fig. 8 analogue — **primary ORR figure** |
| `z04_mc_results.csv` | Trial-level CSV: [sim_id, trial, v_true, snr, v_rec, sigma_est, converged] |
| `z04_acceptance.json` | Pass/fail dict keyed by gate name (§7) |
| `z04_summary.txt` | Human-readable summary printed to stdout and saved |

### 5.1 Figure style

All figures use `matplotlib` with NCAR brand colours where appropriate:
- Bias curve: `#003479` (NCAR navy)
- σ_v curve: `#0057C2` (NCAR blue)
- ORR limit lines: `#CC0000` dashed
- SNR acceptance zone: light grey fill (`alpha=0.15`)

Figure size: 8 × 10 inches at 150 dpi for Sim-3 two-panel; 10 × 5 inches for Sim-2.

---

## 6. Pytest Suite

Test file: `tests/test_z04.py`

| Test ID | Description | Pass Criterion |
|---------|-------------|----------------|
| T1 | Smoke test: N=10 trials, SNR=5 run completes without exception | No exception |
| T2 | ΔS > 0 for synthesised frame (fringe contrast non-zero) | `delta_S > 0` |
| T3 | Recovered v within ±50 m/s of truth at SNR=5 for 10 trials | All 10 pass |
| T4 | σ_ratio ∈ [0.5, 2.0] for N=100 trials at SNR=5 | Passes (loose bound for fast CI) |
| T5 | acceptance.json is valid JSON with all required gate keys | All 8 gate keys present |
| T6 | Output PNG files created for all three simulations | All 5 files exist |

CI runtime target: < 120 seconds (N kept low in CI; full run is `--full` flag).

---

## 7. ORR Acceptance Gates

The following conditions, evaluated from `z04_acceptance.json`, must all be `PASS` before the ORR:

| Gate ID | Metric | Threshold | Rationale |
|---------|--------|-----------|-----------|
| G01 | \|bias(SNR≥2)\| | < 5 m/s | STM wind accuracy requirement (L2 product) |
| G02 | σ_v(SNR=2) | ≤ 15 m/s | STM wind precision requirement at minimum SNR |
| G03 | σ_v(SNR=5) | ≤ 10 m/s | Expected good-conditions precision |
| G04 | σ_ratio (Sim-1) | ∈ [0.80, 1.20] | Error bars accurate to 20% |
| G05 | 68th-percentile coverage (Sim-1) | ∈ [63%, 73%] | Gaussian statistics |
| G06 | K–S p-value (Sim-1) | > 0.05 | Residuals consistent with Gaussian |
| G07 | Velocity-error slope (Sim-2) | < 0.01 m/s per m/s | No gain error |
| G08 | Convergence rate (all sims) | > 95% | Robust inversion |

---

## 8. CLI Interface

```
python validation/z04_snr_sensitivity_sweep.py [OPTIONS]

Options:
  --full          Run full N=10000 simulations (default: CI-scale N=100)
  --seed INT      RNG seed (default: 42)
  --outdir PATH   Output directory (default: validation/outputs/z04/)
  --plot          Show interactive matplotlib windows (default: save only)
  --sim {1,2,3}   Run only the specified simulation
  --verbose       Print per-bin statistics to stdout
```

---

## 9. Relationship to Other Specs

| Spec | Role in Z04 |
|------|-------------|
| S09 (M01) | Airy forward model used inside M04 synthesis |
| S11 (M04) | Synthesises 2D science frame with controllable SNR |
| S12 (M03) | Annular reduction: 2D → 1D fringe profile |
| S13 (Tolansky) | WLS fit → ε_sci → λ_c → v_LOS |
| S14 (M05) | Provides calibration priors (d, f, α) for inversion |
| S15 (M06) | Full airglow inversion (Z04 calls this if available) |
| Z02 | Proof-of-concept single-frame synthesis + Tolansky; Z04 generalises to MC |
| Z03 | Related validation (if implemented); Z04 is independent |

---

## 10. Known Limitations and Future Work

- **Delta-function approximation:** Z04 does not sweep temperature because the delta-function source model (S11) has no thermal broadening free parameter.  A future spec (Z05, post-ORR) will validate bias under different assumed temperatures when a Gaussian source model is optionally substituted.
- **Calibration noise coupling:** Z04 adds Poisson noise to the calibration frame but does not perturb the instrument parameters (d, f, α) between trials.  A full treatment would draw (d, f, α) from their posterior distributions on each trial; this is deferred to Z05.
- **Single LOS direction:** Z04 tests a single LOS (science frame centre).  Off-axis LOS geometry bias is not assessed here; that is an M07/Z06 concern.
- **No hot-pixel injection:** The M03 99.5th-percentile clip is not stressed in Z04.  Hot-pixel robustness is covered in Z01.

---

## 11. Revision History

| Date | Author | Change |
|------|--------|--------|
| 2026-04-11 | S. Sewell | Initial spec written |
