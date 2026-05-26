# MC02 — FPI Monte Carlo Simulation Drivers (Harding Replication)

**Spec file:** `MC02_fpi_mc_simulations_2026-05-24.md`  
**Module:** `windcube/mc02_fpi_mc_simulations.py`  
**Runner:** `run_mc_simulations.py`  
**Test file:** `tests/test_mc02_fpi_mc_simulations.py`  
**Status:** Draft  
**Date:** 2026-05-24  
**Author:** SOC / S. Sewell  
**Reference:** Harding, Gehrels & Makela (2014), *Applied Optics* 53(4), §4 and Figs. 6–8

### Dependencies

| Depends on | Module / file | Purpose |
|------------|--------------|---------|
| MC01 | `windcube/mc01_fpi_mc_engine.py` | All trial execution; T-MC01-xx must pass first |
| CAL01 | `fpi_cal_lib.py` | `InstrumentParams` dataclass construction |
| S03 | `windcube/constants.py` | Physical constants for `InstrumentParams` defaults |

---

## 1. Purpose

MC02 implements the **three Monte Carlo simulation drivers** that replicate
Harding et al. (2014) §4, adapted to WindCube. It calls MC01 for all trial
execution and is responsible solely for constructing `TrialInput` lists, invoking
`mc01.run_simulation()`, saving results, generating figures, and writing a summary
report. MC02 has **no physics implementation of its own**.

All three simulations use `use_temperature=True` internally (hardcoded in MC01).
MC02 does not need to pass this flag — it is transparent to the caller.

---

## 2. The three simulations

### 2.1 Simulation 1 — Uncertainty Estimates (Harding Fig. 6 analog)

**Purpose:** Verify that the inversion accurately estimates its own uncertainty.
The sample covariance of the (v_est, T_est) scatter must match the mean estimated
error ellipses.

| Parameter | Value | Harding §4.A |
|-----------|-------|--------------|
| `v_true` | 100 m/s | 100 m/s ✓ |
| `T_true` | 800 K | 800 K ✓ |
| `SNR` | 5 | 5 ✓ |
| N trials | 10,000 | 10,000 ✓ |

```python
inputs = [TrialInput(v_true_ms=100.0, T_true_K=800.0, snr=5.0)] * 10_000
```

**Expected:** σ_v ≈ 1.8 m/s, σ_T ≈ 6.5 K (Harding values; WindCube may differ
due to larger etalon gap 20.1 mm vs Harding's 15 mm). Blue and red ellipses align.

**Output:** `MC01_sim1_{YYYY-MM-DD}.npz`

---

### 2.2 Simulation 2 — Biases Over Wind and Temperature (Harding Fig. 7 analog)

**Purpose:** Test for systematic biases in v and T over the full operating range.

| Parameter | Distribution | Range | Harding §4.B |
|-----------|-------------|-------|--------------|
| `v_true` | Uniform | −300 to +300 m/s | ✓ |
| `T_true` | Uniform | 300 to 1500 K | ✓ |
| `SNR` | Fixed | 5 | ✓ |
| N trials | — | 1,000 | ✓ |

```python
rng_setup = np.random.default_rng(seed=42)
v_samples = rng_setup.uniform(-300.0, 300.0, size=1_000)
T_samples = rng_setup.uniform( 300.0, 1500.0, size=1_000)
inputs = [TrialInput(v_true_ms=v, T_true_K=T, snr=5.0)
          for v, T in zip(v_samples, T_samples)]
```

**Expected:** Constant ~0.4 m/s velocity bias (Harding); no T bias; scatter
increases with higher T.

**Output:** `MC01_sim2_{YYYY-MM-DD}.npz`

---

### 2.3 Simulation 3 — Biases Over SNR (Harding Fig. 8 analog)

**Purpose:** Test for SNR-dependent biases across the full operational SNR range.

| Parameter | Distribution | Range | Harding §4.C |
|-----------|-------------|-------|--------------|
| `v_true` | Uniform | −300 to +300 m/s | ✓ |
| `T_true` | Uniform | 300 to 1500 K | ✓ |
| `SNR` | Uniform | 0.5 to 10 | 0.5–5 (extended) |
| N trials | — | 10,000 | ✓ |

> **WindCube extension:** Upper SNR bound extended to 10 to cover storm-enhanced
> airglow (SQ1, `WIND_MAX_STORM_MS = 400 m/s`). To match Harding exactly, change
> `uniform(0.5, 10.0)` to `uniform(0.5, 5.0)`.

```python
rng_setup   = np.random.default_rng(seed=42)
v_samples   = rng_setup.uniform(-300.0,  300.0, size=10_000)
T_samples   = rng_setup.uniform( 300.0, 1500.0, size=10_000)
snr_samples = rng_setup.uniform(   0.5,   10.0, size=10_000)
inputs = [TrialInput(v_true_ms=v, T_true_K=T, snr=s)
          for v, T, s in zip(v_samples, T_samples, snr_samples)]
```

**Expected:** No bias in v or T at any SNR; scatter increases at low SNR;
non-converged trials concentrated at SNR < 1.

**Output:** `MC01_sim3_{YYYY-MM-DD}.npz`

---

## 3. Runner script (`run_mc_simulations.py`)

```
usage: run_mc_simulations.py [--sim {1,2,3,all}] [--workers N]
                              [--seed SEED] [--outdir DIR]
```

**Execution order:**
1. `cat PIPELINE_STATUS.md`
2. `pytest tests/test_mc01_fpi_mc_engine.py -q` — gate; abort if any fail
3. Build `InstrumentParams` from `windcube/constants.py`
4. Build `TrialInput` lists (§2)
5. `mc01.run_simulation()` for each selected sim
6. `mc01.save_simulation()` for each result
7. Generate figures (§4)
8. Write summary report (§5)
9. Update `PIPELINE_STATUS.md`

---

## 4. Figure specifications

All figures: 300 dpi PNG + PDF to `{outdir}/figures/`.

---

### 4.1 Figure 1 — Uncertainty Estimates (Harding Fig. 6 analog)

**Filename:** `MC02_fig1_sim1_uncertainty_{YYYY-MM-DD}.png`

**Single panel scatter plot.**

- All 10⁴ (v_est, T_est) pairs: semi-transparent grey dots (alpha=0.15, size=2 pt)
- **Blue ellipse:** 1-σ sample covariance of (v_est, T_est) scatter
- **Red dashed ellipse:** 1-σ from mean of per-trial estimated uncertainties
  (σ_v, σ_T); zero cross-covariance per Harding §4.A
- Crosshairs at (v_true=100, T_true=800)
- Infobox: mean σ_v (m/s), mean σ_T (K), fraction within 1-σ (expect 0.683)

```python
# Blue ellipse — sample covariance
cov = np.cov(v_est, T_est)
eigenvalues, eigenvectors = np.linalg.eigh(cov)
angle  = np.degrees(np.arctan2(*eigenvectors[:, 1][::-1]))
width, height = 2 * np.sqrt(eigenvalues)
blue_ell = Ellipse(xy=(np.mean(v_est), np.mean(T_est)),
                   width=width, height=height, angle=angle,
                   edgecolor='blue', facecolor='none', linewidth=2,
                   label='Sample covariance (1σ)')

# Red ellipse — mean estimated uncertainty
red_ell  = Ellipse(xy=(100.0, 800.0),
                   width=2*np.mean(sigma_v), height=2*np.mean(sigma_T),
                   angle=0,
                   edgecolor='red', facecolor='none', linewidth=2,
                   linestyle='--', label='Mean estimated uncertainty (1σ)')
```

**Title:** `Simulation 1: Uncertainty Estimates  (v = 100 m/s, T = 800 K, SNR = 5, N = 10,000)`

---

### 4.2 Figure 2 — Biases Over Wind and Temperature (Harding Fig. 7 analog)

**Filename:** `MC02_fig2_sim2_bias_vT_{YYYY-MM-DD}.png`

**2×2 subplot grid:**

| Panel | x-axis | y-axis |
|-------|--------|--------|
| (a) top-left | v_true (m/s) | v_error = v_est − v_true (m/s) |
| (b) top-right | T_true (K) | v_error (m/s) |
| (c) bottom-left | v_true (m/s) | T_error = T_est − T_true (K) |
| (d) bottom-right | T_true (K) | T_error (K) |

All panels: grey dots (alpha=0.4, size=3 pt); dashed red y=0 line; per-bin
median ± 1-σ band in orange (30 m/s bins for velocity panels, 100 K bins for
temperature panels).

**Title:** `Simulation 2: Biases Over Wind Speed and Temperature  (SNR = 5, N = 1,000)`

---

### 4.3 Figure 3 — Biases Over SNR (Harding Fig. 8 analog)

**Filename:** `MC02_fig3_sim3_bias_snr_{YYYY-MM-DD}.png`

**1×2 subplot grid:**

| Panel | x-axis | y-axis |
|-------|--------|--------|
| (a) left | SNR (log scale, [0.5, 10]) | v_error (m/s) |
| (b) right | SNR (log scale, [0.5, 10]) | T_error (K) |

Both panels: grey dots (alpha=0.15); dashed red y=0; orange median ± 1-σ band
(10 log-spaced SNR bins); dashed grey vertical at SNR=5.

**Title:** `Simulation 3: Biases Over SNR  (v ∈ [−300, 300] m/s, T ∈ [300, 1500] K, SNR ∈ [0.5, 10], N = 10,000)`

---

## 5. Summary report

Written to stdout and `{outdir}/MC02_summary_{YYYY-MM-DD}.txt`:

```
WindCube MC02 Monte Carlo Simulation Summary
============================================
Run date         : {YYYY-MM-DD HH:MM:SS UTC}
CAL01 git hash   : {hash}
MC01 git hash    : {hash}
RNG seed         : {seed}

Simulation 1 (Uncertainty Estimates)
  N trials             : 10,000
  N converged          : {n}  ({pct:.1f}%)
  Mean σ_v             : {x:.2f} m/s   [Harding: ~1.8 m/s]
  Mean σ_T             : {x:.1f} K     [Harding: ~6.5 K]
  Fraction within 1σ v : {x:.3f}       [expect 0.683]
  Fraction within 1σ T : {x:.3f}       [expect 0.683]
  Blue/red axis ratio v: {x:.3f}       [expect ~1.00]
  Blue/red axis ratio T: {x:.3f}       [expect ~1.00]

Simulation 2 (Biases Over Wind and Temperature)
  N trials             : 1,000
  N converged          : {n}  ({pct:.1f}%)
  Median v bias        : {x:.3f} m/s   [Harding: ~0.4 m/s]
  Median T bias        : {x:.2f} K
  Max |v bias| binned  : {x:.2f} m/s
  Max |T bias| binned  : {x:.1f} K

Simulation 3 (Biases Over SNR)
  N trials             : 10,000
  N converged          : {n}  ({pct:.1f}%)
  Median v bias (all)  : {x:.3f} m/s
  Median T bias (all)  : {x:.2f} K
  Convergence SNR < 1  : {x:.1f}%
  Convergence SNR > 3  : {x:.1f}%
```

---

## 6. Acceptance tests

| Test ID | Description | Pass criterion |
|---------|-------------|----------------|
| T-MC02-01 | Sim 1 smoke (N=100) | Convergence ≥ 95%; σ_v ∈ [1, 5] m/s; σ_T ∈ [1, 30] K |
| T-MC02-02 | Sim 2 smoke (N=50) | Convergence ≥ 90%; no exception |
| T-MC02-03 | Sim 3 smoke (N=100) | Convergence ≥ 70% overall; ≥ 95% at SNR > 3 |
| T-MC02-04 | Seed reproducibility | Two calls with seed=42 → identical TrialInput lists |
| T-MC02-05 | Fig 1 renders | File > 50 kB; two ellipses present |
| T-MC02-06 | Fig 2 renders | 2×2 grid; all four panels present |
| T-MC02-07 | Fig 3 renders | Log x-axis on both panels |
| T-MC02-08 | Summary report | File exists; all three simulation blocks present |
| T-MC02-09 | `--sim 1` isolation | Only sim1 .npz and Fig 1 written |
| T-MC02-10 | Ellipse ratio (Sim 1, N=500) | Blue/red semi-axis ratio ∈ [0.7, 1.5] for both v and T |

---

## 7. Estimated runtime

| Simulation | N | Expected (8 cores) |
|------------|---|--------------------|
| Sim 1 | 10,000 | ~20–40 min |
| Sim 2 | 1,000 | ~2–4 min |
| Sim 3 | 10,000 | ~20–40 min |

> The thermal spectral integral (L=300 wavelengths per synthesis, L=101 per
> inversion) adds ~3–5× overhead per trial vs the delta-function path. Set
> `OMP_NUM_THREADS=1` to avoid BLAS contention between workers.

---

## 8. Out of scope

- Any modification of `fpi_cal_lib.py`, `m03_*`, or `m06_*`
- ORR reporting
- LOS decomposition / wind vector retrieval (M07)
- HWM14 wind inputs (G01)
- Instrument-parameter sensitivity studies (future MC03)

---

## 9. Change log

| Version | Date | Summary |
|---------|------|---------|
| 1.0 | 2026-05-24 | First draft |
| 2.0 | 2026-05-24 | `use_temperature` flag removed from MC02 (now transparent via M03/M06); runtime estimate updated for thermal path overhead |
