# MC01 — FPI Monte Carlo Simulation Engine

**Spec file:** `MC01_fpi_mc_engine_2026-05-24.md`  
**Module:** `windcube/mc01_fpi_mc_engine.py`  
**Test file:** `tests/test_mc01_fpi_mc_engine.py`  
**Status:** Draft  
**Date:** 2026-05-24  
**Author:** SOC / S. Sewell  

### Dependencies

| Depends on | Module / file | Purpose |
|------------|--------------|---------|
| CAL01 | `fpi_cal_lib.py` | Neon calibration forward model and inversion |
| M03 + addendum | `m03_airglow_synthesis_2026-05-24.py` | `synthesise_airglow_image()` with `use_temperature` flag |
| M06 + addendum | `m06_airglow_inversion_2026-05-24.py` | `fit_airglow_fringe()` with `use_temperature` flag; `AirglowFitResult` |
| S03 | `windcube/constants.py` | All physical constants (sole authoritative source) |

**Prerequisite:** M03 addendum and M06 addendum must be implemented and pass their
respective T-M03-Txx and T-M06-Txx acceptance tests before MC01 may be implemented.

### Required by
MC02 (`MC02_fpi_mc_simulations_2026-05-24.md`)

---

## 1. Purpose

MC01 provides a **reusable Monte Carlo simulation engine** for the WindCube FPI
calibration pipeline. It wraps the existing calibration and airglow forward models
and inversions into a parallelisable single-trial harness, adds physics-correct
noise injection, and writes structured results to disk. It has **no opinion** about
which parameter space to sweep — that is the responsibility of the caller (MC02).

The engine replicates the Monte Carlo methodology of Harding, Gehrels & Makela
(2014), §4, adapted to WindCube instrument parameters. Unlike the earlier draft,
MC01 does **not** contain any bespoke inversion physics — it delegates entirely to
M03 (synthesis) and M06 (inversion), both of which now carry the `use_temperature`
flag.

---

## 2. Single-trial procedure

Each trial proceeds as follows (Harding §4):

1. **Synthesise a true neon calibration fringe** using `fpi_cal_lib.py` CAL01 at
   fixed instrument parameters (§4).
2. **Add Poisson noise** to the calibration fringe via M03's `add_poisson_noise()`
   (imported from `src.fpi.m02_calibration_synthesis_2026_05_05`).
3. **Invert the noisy calibration fringe** via the CAL01 staged LM inversion to
   recover instrument parameters (`t`, `R`, `α`, PSF params, intensity params, `B`).
4. **Synthesise a true airglow fringe** at caller-specified `(v_true, T_true)` via
   `m03_airglow_synthesis_2026-05-24.synthesise_airglow_image()` with
   `use_temperature=True` and `T_true_K=T_true`.
5. **Add Gaussian white noise** at the specified SNR via M03's
   `add_gaussian_noise()` (Harding Eq. 17: `ΔS = max − min`, `σ_N = ΔS / SNR`).
6. **Invert the noisy airglow fringe** via
   `m06_airglow_inversion_2026-05-24.fit_airglow_fringe()` with
   `use_temperature=True`, using the instrument parameters recovered in step 3.
7. **Return** `TrialResult` with `v_est`, `T_est`, `σ_v`, `σ_T`, `χ²`,
   `converged`.

> **WindCube adaptation:** Harding used a HeNe laser (632.8 nm) as the calibration
> source. WindCube uses a neon lamp (λ₁ = `NE_WAVELENGTH_1_AIR_M` = 640.2248 nm,
> λ₂ = `NE_WAVELENGTH_2_AIR_M` = 638.2991 nm from `windcube/constants.py`). The
> calibration fringe synthesis and inversion use the two-line neon forward model
> already implemented in `fpi_cal_lib.py`.

---

## 3. Module interface

### 3.1 Public API

```python
# windcube/mc01_fpi_mc_engine.py

def run_single_trial(
    v_true_ms: float,
    T_true_K: float,
    snr: float,
    instrument_params: "InstrumentParams",   # from fpi_cal_lib.py
    airglow_params: AirglowParams,
    rng: np.random.Generator,
) -> TrialResult:
    """One complete Monte Carlo trial. See §2 for procedure."""

def run_simulation(
    trial_inputs: list[TrialInput],
    instrument_params: "InstrumentParams",
    airglow_params: AirglowParams,
    n_workers: int = -1,
    seed: int = 42,
    progress: bool = True,
) -> SimulationResult:
    """Run N trials in parallel via ProcessPoolExecutor. See §7."""

def save_simulation(result: SimulationResult, path: str | Path) -> None:
    """Serialise SimulationResult to compressed .npz. See §8."""

def load_simulation(path: str | Path) -> SimulationResult:
    """Deserialise SimulationResult from .npz. See §8."""
```

### 3.2 Data classes

```python
@dataclass
class AirglowParams:
    """
    Airglow synthesis parameters — fixed across all trials in one simulation.
    Passed directly to m03_airglow_synthesis.synthesise_airglow_image().
    """
    B_sci:       float = 275.0   # CCD bias, counts (FM PTC value)
    Y_bg:        float = 50.0    # background sky emission, counts
    Y_line:      float = 1.0     # airglow line scale factor (dimensionless)
    L_synth:     int   = 300     # spectral bins for synthesis (anti-inverse-crime)
    L_invert:    int   = 101     # spectral bins for inversion (Harding §3)
    R_bins:      int   = 500     # radial bins (Harding §3: R=500)
    n_fsr:       float = 5.0     # FSR span for spectral grid
    image_size:  int   = 256     # CCD active dimension, px (2×2 binned)

@dataclass
class TrialInput:
    v_true_ms: float
    T_true_K:  float
    snr:       float

@dataclass
class TrialResult:
    v_true_ms:    float
    T_true_K:     float
    snr:          float
    v_est_ms:     float
    T_est_K:      float     # from AirglowFitResult.T_est_K
    sigma_v_ms:   float
    sigma_T_K:    float     # from AirglowFitResult.sigma_T_K
    chi2_airglow: float
    converged:    bool

@dataclass
class SimulationResult:
    trials:            list[TrialResult]
    instrument_params: "InstrumentParams"
    airglow_params:    AirglowParams
    seed:              int
    timestamp_utc:     str
    n_converged:       int
    n_total:           int
```

---

## 4. Authoritative instrument parameters

`InstrumentParams` must be constructed from `windcube/constants.py`. The caller
(MC02) must not override these defaults unless explicitly testing instrument-
parameter sensitivity.

| Parameter | Constant | Value | Source |
|-----------|----------|-------|--------|
| Cal λ₁ | `NE_WAVELENGTH_1_AIR_M` | 640.2248 nm | Burns et al. (1950), IAU "S" |
| Cal λ₂ | `NE_WAVELENGTH_2_AIR_M` | 638.2991 nm | Burns et al. (1950), IAU "S" |
| OI λ₀ | `OI_WAVELENGTH_AIR_M` | 630.0304 nm | NIST ASD |
| Reflectivity | `ETALON_R_INSTRUMENT` | 0.53 | FlatSat H05 calibration fit |
| Etalon gap | `ETALON_GAP_M` | 20.1071 mm | FlatSat H05 / CAL01 Tolansky |
| Plate scale | `ALPHA_RAD_PX` | 1.6085×10⁻⁴ rad/px | FlatSat H05 / CAL01 (2×2) |
| CCD bias | — | 275 DN | FM PTC measurement |
| Max radius | `R_MAX_PX` | 110 px | FlatSat/FM operational |
| Boltzmann k | `BOLTZMANN_J_PER_K` | 1.380649×10⁻²³ J/K | CODATA 2018 |
| O mass | `OXYGEN_MASS_KG` | 2.6567×10⁻²⁶ kg | `windcube/constants.py` |

---

## 5. Noise model

### 5.1 Calibration fringe — Poisson

```python
from src.fpi.m02_calibration_synthesis_2026_05_05 import add_poisson_noise
noisy_cal = add_poisson_noise(true_cal_profile, rng=rng)
```

### 5.2 Airglow fringe — Gaussian (Harding Eq. 17)

```python
from m03_airglow_synthesis_2026_05_24 import add_gaussian_noise
noisy_airglow_image = add_gaussian_noise(
    image_noiseless, snr, profile_1d, rng=rng
)
```

`ΔS = max(profile_1d) − min(profile_1d)`, `σ_N = ΔS / SNR`.

---

## 6. Inversion

### 6.1 Calibration inversion

Uses CAL01 (`fpi_cal_lib.py`) staged LM inversion. Initial guesses from
`windcube/constants.py` authoritative values (§4). Recovered instrument
parameters passed directly to step 6.2.

### 6.2 Airglow inversion

```python
from m06_airglow_inversion_2026_05_24 import fit_airglow_fringe

fit = fit_airglow_fringe(
    profile,
    cal,
    n_fine=500,
    use_temperature=True,   # always True inside MC01 trials
)

# Extract results
v_est_ms  = fit.v_rel_ms
T_est_K   = fit.T_est_K          # from 4-param fit via M06 addendum
sigma_v   = fit.sigma_v_rel_ms
sigma_T   = fit.sigma_T_K
chi2      = fit.chi2_reduced
converged = (fit.converged
             and fit.T_est_K is not None
             and fit.T_est_K > 0
             and abs(fit.v_rel_ms) < 2000.0)
```

`use_temperature=True` is **hardcoded** inside MC01 trial execution — MC01 always
retrieves temperature. There is no MC01-level `use_temperature` flag; the flag
lives in M03 and M06 where it belongs.

---

## 7. Parallelisation

`concurrent.futures.ProcessPoolExecutor`. Each worker receives its `TrialInput`
and a child `np.random.Generator` from `np.random.SeedSequence(master_seed)`.
Set `OMP_NUM_THREADS=1` before running to prevent BLAS thread pool contention.

---

## 8. I/O format

Compressed `.npz` — same schema as previous draft:

| Key | Shape | dtype |
|-----|-------|-------|
| `v_true` | (N,) | float64 |
| `T_true` | (N,) | float64 |
| `snr` | (N,) | float64 |
| `v_est` | (N,) | float64 |
| `T_est` | (N,) | float64 |
| `sigma_v` | (N,) | float64 |
| `sigma_T` | (N,) | float64 |
| `chi2` | (N,) | float64 |
| `converged` | (N,) | bool |
| `metadata` | scalar | JSON string |

Filename (set by MC02): `MC01_sim{N}_{YYYY-MM-DD}.npz`

---

## 9. Acceptance tests

All T-M03-Txx and T-M06-Txx tests must pass before running these.

| Test ID | Description | Pass criterion |
|---------|-------------|----------------|
| T-MC01-01 | Single trial, v=0, T=800K, SNR=10 | `\|v_est\| < 5 m/s`, `\|T_est − 800\| < 30 K`, `converged=True` |
| T-MC01-02 | Single trial, v=100, T=800K, SNR=5 | `\|v_est − 100\| < 10 m/s`, `T_est > 0`, `converged=True` |
| T-MC01-03 | Poisson noise: calibration frame | Mean of 1000 noisy profiles ≈ true within 1% |
| T-MC01-04 | Gaussian noise: airglow frame | Measured σ matches `ΔS / SNR` within 2% |
| T-MC01-05 | SNR round-trip | Injected SNR recoverable within 5% |
| T-MC01-06 | Reproducibility | Same seed → identical results |
| T-MC01-07 | Parallelism consistency | `n_workers=1` and `n_workers=4` identical for N=100 |
| T-MC01-08 | Save/load round-trip | All arrays recovered bit-exactly |
| T-MC01-09 | Diverged trial flagging | Noise σ→∞ → `converged=False`, no exception raised |
| T-MC01-10 | Constants traceability | `InstrumentParams` defaults asserted against `windcube/constants.py` at import |

---

## 10. Out of scope

- Parameter space sampling (MC02)
- Figure generation
- ORR reporting
- LOS decomposition / wind vector retrieval (M07)
- HWM14 wind inputs (G01)
- Any modification of `fpi_cal_lib.py`, `m03_airglow_synthesis_*.py`, or `m06_airglow_inversion_*.py`

---

## 11. Change log

| Version | Date | Summary |
|---------|------|---------|
| 1.0 | 2026-05-24 | First draft (bespoke thermal inversion in MC01) |
| 2.0 | 2026-05-24 | Bespoke thermal path removed; delegates to M03/M06 `use_temperature` flag |
