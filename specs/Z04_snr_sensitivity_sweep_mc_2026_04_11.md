# Z04 — SNR Sensitivity Sweep Monte Carlo Validation

**Spec:** Z04  
**Module:** `validation/z04_snr_sensitivity_sweep.py`  
**Tier:** 9 — Validation  
**Status:** Implemented — 6/6 tests passing  
**Date written:** 2026-04-11  
**Date implemented:** 2026-04-11  
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
validation/z04_snr_sensitivity_sweep.py   (~430 lines)
tests/test_z04.py
```

### 4.2 Module imports

Z04 calls existing production modules directly:

```python
from soc_sewell.m04_airglow_synthesis import synthesize_airglow_frame
from soc_sewell.m03_annular_reduction  import reduce_science_frame
from soc_sewell.m06_airglow_inversion  import invert_airglow_fringe
from soc_sewell.constants              import PHY
```

### 4.3 Key implementation functions

| Function | Description |
|----------|-------------|
| `_build_calibration_result()` | Noiseless synthetic `CalibrationResult` via M05. Uses conftest.py centre convention to avoid systematic offsets |
| `_run_single_trial()` | Synthesise → reduce (M03) → invert (M06). Returns `v_rec`, `sigma_est`, `converged`, `delta_s` |
| `_run_sim1()` | N=10 000 trials at fixed v=100 m/s, SNR=5 |
| `_run_sim2()` | N=1 000 trials with randomised v ∈ [−300, +300] m/s |
| `_run_sim3()` | N=10 000 trials across SNR grid; joblib-parallel via `_run_snr_bin()` |
| `_run_snr_bin()` | Module-level joblib worker — builds its own `CalibrationResult` to avoid pickling issues with complex dataclass fields |
| `_check_acceptance_gates()` | Evaluates all 8 ORR gates (G01–G08), returns dict |

### 4.4 SNR injection

```python
delta_S = profile.max() - profile.min()   # fringe contrast (ADU)
sigma_N = delta_S / snr                    # noise std (ADU)
noisy_frame = frame + rng.normal(0, sigma_N, frame.shape)
```

Poisson noise on calibration frame:

```python
cal_frame_noisy = rng.poisson(cal_frame.astype(float)).astype(np.float32)
```

### 4.5 Reproducibility

Global `rng = np.random.default_rng(seed=42)`.  Seed exposed as `--seed` CLI argument.

### 4.6 Parallelism

`joblib.Parallel(n_jobs=-1)` wraps Sim-3 SNR bins.  Each worker receives an independent RNG child via `rng.spawn(N_workers)`.  `joblib` added to `requirements.txt`.

**Implementation note:** `_run_snr_bin()` is defined at module level (not as a nested or lambda function) specifically to avoid joblib pickling failures with complex dataclass fields in `CalibrationResult`.  Each worker rebuilds its own `CalibrationResult` from scratch.

---

## 5. Output Artifacts

All outputs written to `validation/outputs/z04/`:

| File | Description |
|------|-------------|
| `z04_sim1_scatter.png` | Sim-1 scatter + histogram panel |
| `z04_sim2_bias_vs_velocity.png` | Harding Fig. 7 analogue |
| `z04_sim3_bias_vs_snr.png` | Harding Fig. 8 analogue — **primary ORR figure** |
| `z04_mc_results.csv` | Trial-level CSV: [sim_id, trial, v_true, snr, v_rec, sigma_est, converged] |
| `z04_acceptance.json` | Pass/fail dict keyed by gate name (G01–G08) |
| `z04_summary.txt` | Human-readable summary |

### 5.1 Figure style

All figures use `matplotlib` with NCAR brand colours:
- Bias curve: `#003479` (NCAR navy)
- σ_v curve: `#0057C2` (NCAR blue)
- ORR limit lines: `#CC0000` dashed
- SNR acceptance zone: light grey fill (`alpha=0.15`)

Figure size: 8 × 10 inches at 150 dpi for Sim-3 two-panel; 10 × 5 inches for Sim-2.

---

## 6. Pytest Suite

Test file: `tests/test_z04.py` — **6/6 passing, CI runtime 38 s**

| Test ID | Description | Result |
|---------|-------------|--------|
| T1 | Smoke test: N=10 trials, all sims, no exception | PASS |
| T2 | ΔS > 0 for synthesised frame (fringe contrast non-zero) | PASS |
| T3 | \|v_rec − v_true\| < 50 m/s at SNR=5 for 10 trials | PASS |
| T4 | σ_ratio ∈ [0.5, 2.0] for N=100 trials at SNR=5 | PASS |
| T5 | acceptance.json has all G01–G08 keys | PASS |
| T6 | All 5 output PNG/CSV/JSON files created | PASS |

---

## 7. ORR Acceptance Gates

The following conditions, evaluated from `z04_acceptance.json`, must all be `PASS` before the ORR.  The full-run evaluation (`--full`) is the authoritative ORR check:

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

**To run the full ORR evaluation:**
```bash
python validation/z04_snr_sensitivity_sweep.py --full --verbose
cat validation/outputs/z04/z04_acceptance.json
```

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

---

## 10. Known Limitations and Future Work

- **Delta-function approximation:** Z04 does not sweep temperature because the delta-function source model (S11) has no thermal broadening free parameter.  A future spec (Z05, post-ORR) will validate bias under different assumed temperatures when a Gaussian source model is optionally substituted.
- **Calibration noise coupling:** Z04 adds Poisson noise to the calibration frame but does not perturb the instrument parameters (d, f, α) between trials.  A full treatment would draw (d, f, α) from their posterior distributions; deferred to Z05.
- **Single LOS direction:** Z04 tests a single LOS (science frame centre).  Off-axis geometry bias is an M07/Z06 concern.
- **No hot-pixel injection:** Hot-pixel robustness is covered in Z01.

---

## 11. Revision History

| Date | Author | Change |
|------|--------|--------|
| 2026-04-11 | S. Sewell | Initial spec written |
| 2026-04-11 | S. Sewell | Updated after Claude Code implementation: added §4.3 function table, joblib pickling note (§4.6), T-series results (§6), revised status to Implemented |
