# WindCube FPI Pipeline Architecture — Processing Chain Synthesis
## From Raw Pixels to Thermospheric Wind Vectors

**Document ID:** WINDCUBE-ARCH-01  
**Version:** 2  
**Date:** 2026-05-14  
**Author:** Scott Sewell (HAO/NCAR)  
**Status:** Authoritative — project reference  
**Repo:** soc_sewell  
**Location:** `docs/WINDCUBE_ARCH_01_pipeline_synthesis_2026-05-14.md`

---

## 1. Purpose

This document describes the complete WindCube FPI science processing chain
from raw binary images to calibrated thermospheric wind vectors. It captures
design decisions made during the May 2026 pipeline development sprint and
serves as the authoritative reference for all subsequent spec authoring and
implementation work.

All five core processing modules have been audited against their source code.
The implementation plan in §8 reflects the confirmed state of each module.

---

## 2. Instrument and observation context

The WindCube FPI acquires three image types during normal operations:

**Dark frames** (`*_dark.bin`): shutter closed, lamp off. CCD thermal
background. Used to construct a master dark for bias/thermal subtraction.

**Calibration frames** (`*_cal_*.bin`): shutter open, two-line neon
calibration lamp on (640.2248 nm and 638.2991 nm per Burns, Adams &
Longwell 1950 IAU standards). Concentric Airy fringe rings from two known
wavelengths. Used to characterise the etalon gap, plate scale, reflectivity,
PSF, and establish the zero-wind phase reference `epsilon_cal`.

**Science frames** (`*_science_*.bin`): shutter open, lamp off. OI 630.0 nm
airglow fringe rings from the thermosphere at ~250 km altitude. The target
signal for wind retrieval.

The AOCS alternates between **along-track** and **cross-track** pointing
on an orbit-by-orbit basis per SI-UCAR-WC-RP-004 §2.4.2.2 (Tangent Height
Reference Frame). This provides the azimuthal diversity required for the
H07 wind vector inversion.

---

## 3. Calibration cadence and master calibration construction

### 3.1 Per-orbit calibration sequence

The nominal calibration cadence is **5 calibration frames + 5 dark frames
per orbit** (~95 minute orbital period). This cadence provides:

- Redundancy against individual frame failures
- Statistical averaging to reduce fit parameter uncertainty
- Per-orbit tracking of etalon thermal state

The processing sequence per orbit is:

```
Step 1 — Master dark construction
  5 dark frames → summed pixel-by-pixel → master_dark  [256×256, float32]
  (Sum, not mean. Divide by 5 when subtracting from any single frame to
   preserve correct photon-noise statistics.)

Step 2 — Dark-subtract each calibration frame
  cal_frame[i] − master_dark / 5   for i = 1..5
  → 5 dark-subtracted calibration frames

Step 3 — Independent inversion of each dark-subtracted cal frame
  For each of the 5 cal frames independently:
    find_centre        → CentreResult  (cx, cy, sigma_cx, sigma_cy)
    annular_reduce     → FringeProfile (profile, sigma, r_grid, peak_fits_r2, ...)
    run_tolansky_2line → TolanskyResult (d_m, eps_a, eps_b, alpha_mean, ...)
    to_m05_priors      → dict  (t_init_mm, alpha_init, epsilon_cal_a, ...)
    run_staged_inversion → CalibrationResult (t_m, alpha, R_refl, sigma0, B,
                                               epsilon_cal, + all 1σ uncertainties)
  Producing 5 independent CalibrationResult objects.

Step 4 — Average to master calibration
  Simple arithmetic mean of the 5 CalibrationResult parameter vectors:
    p_master[i] = mean([p1[i], p2[i], p3[i], p4[i], p5[i]])
    sigma_master[i] = mean([s1[i],...,s5[i]]) / sqrt(5)   [SEM]
  → MasterCalibration object

Step 5 — Apply to science frames
  For each science frame acquired during the same orbit:
    science_frame − master_dark / 5     → dark-subtracted science pixels
    find_centre (or use cal cx/cy as seed) → CentreResult
    annular_reduce                        → FringeProfile
    [H07 geometry first — see §5]
    run_airglow_inversion(fp, master_cal, v_los_prior_ms)
                                          → AirglowResult (v_rel_ms, sigma_v_ms)
    H07 correct + invert                  → WindSolution (v_E, v_N, ...)
```

### 3.2 Calibration stability assumption

The etalon gap is assumed stable within a single 95-minute orbit. A single
master calibration result is applied to all science frames in that orbit
without interpolation. If in-orbit thermal drift is subsequently found to
be significant (detectable as a systematic trend in `v_rel` residuals), this
assumption will be revisited. Interpolation between orbits is explicitly
deferred to a future spec.

### 3.3 Averaging rationale

Simple arithmetic mean is used (not inverse-variance weighting) because:
- All 5 cal frames are acquired consecutively under identical conditions
- SNR and fit quality are expected to be uniform across the 5 frames
- The uncertainty reduction follows standard SEM: `sigma / sqrt(5)`
- Simplicity is preferred; the marginal gain from weighting is small

---

## 4. The Tolansky stage — role and information flow

The Tolansky two-line analysis (`src/fpi/tolansky_2026-05-13.py`, public
function `run_tolansky_2line()`) provides **coarse etalon characterisation**
from neon fringe peak positions:

- `d_m`: etalon gap [m] (Benoit excess fractions, Vaughan 1989 Eq. 3.97)
- `eps_a`: fractional fringe order at λ_a = 640.2248 nm
- `eps_b`: fractional fringe order at λ_b = 638.2991 nm
- `alpha_mean`: plate scale [rad/px, 2×2 binned]

These are passed to H05 via `to_m05_priors()` as **initial parameter
guesses** only. The H05 output supersedes the Tolansky values and is the
authoritative instrument characterisation used by H06.

**Why Tolansky first?** The H05 LM optimisation has a 10-dimensional
parameter space. Tolansky provides physically motivated seeds for `t_m` and
`alpha` that reliably land the LM minimiser in the correct basin of
attraction. This has been validated against the FlatSat calibration data.

**Tolansky values update with every calibration acquisition**, tracking
long-term etalon drift and providing an independent instrument health record.

**File location:** `tolansky-2line.py` is a thin interactive launcher that
dynamically imports from `src/fpi/tolansky_2026-05-13.py` (located two
directory levels above the launcher). The real analysis functions live in
that canonical module:
- `run_tolansky_2line(peaks_input, ...) → TolanskyResult`
- `to_m05_priors(result) → dict`
- `print_rectangular_array(result)`
- `plot_tolansky_result(result, save_path)`

A backward-compatibility alias `run_tolansky = run_tolansky_2line` is at
the bottom of the canonical module.

---

## 5. The v_los_prior coupling point (H07 → H06)

H06's λ_c scan requires an a-priori LOS velocity seed to avoid FSR alias
ambiguity. The dominant contribution is spacecraft-atmosphere relative
motion (~km/s for along-track, ~0–500 m/s for cross-track).

**This prior is computed by H07's geometry engine** and passed to H06:

```python
v_los_prior_ms = geom.V_sc_LOS + geom.v_earth_LOS   # from LOSGeometry
```

H07 geometry must therefore run **before** H06 for each science frame.
The per-frame processing order is:

```
1. H07: compute_los_geometry(meta)
        → LOSGeometry (V_sc_LOS, v_earth_LOS, L_E, L_N, tangent_lat/lon/alt)

2.      v_los_prior_ms = geom.V_sc_LOS + geom.v_earth_LOS

3. H06: run_airglow_inversion(fp, master_cal, v_los_prior_ms)
        → AirglowResult (v_rel_ms, sigma_v_ms)

4. H07: correct_los_velocity(v_rel_ms, geom)
        → v_corrected = v_rel_ms + V_sc_LOS + v_earth_LOS

5.      Accumulate LOSObservation for wind vector inversion
```

For cross-track frames `v_los_prior_ms ≈ 0–500 m/s` and the scan finds
the correct minimum without the prior. For along-track frames
`v_los_prior_ms ≈ ±7000 m/s` and the prior is essential.

---

## 6. Module inventory — confirmed public APIs

All five core processing modules have been source-audited. Functions ready
to import today (before any refactoring):

| Module | File | Public function | Input | Output |
|--------|------|----------------|-------|--------|
| Centre finding | `center_finder.py` | `find_centre(image, ...)` | 2D ndarray | `CentreResult` |
| Annular reduction | `annular_reduction.py` | `annular_reduce(image, cx, cy, ...)` | 2D ndarray + centre | `FringeProfile` |
| Tolansky analysis | `src/fpi/tolansky_2026-05-13.py` | `run_tolansky_2line(peaks_input, ...)` | peak_fits_r2 array | `TolanskyResult` |
| Tolansky→H05 bridge | `src/fpi/tolansky_2026-05-13.py` | `to_m05_priors(result)` | `TolanskyResult` | dict |
| Calibration inversion | `H05_calibration_inversion_2026_05_12.py` | `run_staged_inversion(fp, t_eff, alpha, ...)` | `_FringeProfile` + params | `FitResult` |
| Calibration save | `H05_calibration_inversion_2026_05_12.py` | `save_cal_result(fit, path, ...)` | `FitResult` | `.npy` file |
| **Airglow inversion** | `H06_airglow_inversion_2026_05_14.py` | **No clean public function yet** | — | — |

**H06 is the only module requiring refactoring** before batch pipeline
integration. This is Step 1 of the implementation plan.

### FringeProfile dataclass mismatch

`annular_reduction.py` defines a `FringeProfile` dataclass with full fields
including `peak_fits_r2`, centre coordinates, and reduction parameters.

`H05_calibration_inversion_2026_05_12.py` defines an internal `_FringeProfile`
with compatible but fewer fields (profile, r_grid, sigma_profile, masked,
r_max_px) and a different class name.

**Resolution (Step 2):** `windcube/fpi_pipeline.py` will define one canonical
`FringeProfile` and provide a `to_h05_fringe_profile()` conversion shim.
No existing script is modified.

---

## 7. In-memory pipeline — complete information flow

All intermediate results pass in memory for batch processing efficiency.
Intermediate files written only in diagnostic mode.

```
Raw binary .bin file
    │
    ├─ P01: ingest_real_image()
    │       → (ImageMetadata, pixels: uint16 ndarray 259×276)
    │
    ├─ Dark subtraction (in memory)
    │       pixels_ds = pixels.astype(float32) − master_dark / n_dark
    │
    ├─ H07: compute_los_geometry(meta)           ← MUST run before H06
    │       → LOSGeometry
    │         (V_sc_LOS, v_earth_LOS, L_E, L_N,
    │          tangent_lat/lon/alt, l_hat_eci)
    │
    ├─ find_centre(pixels_ds)
    │       → CentreResult (cx, cy, sigma_cx, sigma_cy)
    │
    ├─ annular_reduce(pixels_ds, cx, cy, ...)
    │       → FringeProfile
    │         (profile, sigma_profile, r_grid, r2_grid,
    │          peak_fits_r2, masked, ...)
    │
    ╔══ CALIBRATION FRAMES ONLY ═══════════════════════════════════╗
    ║                                                               ║
    ║  run_tolansky_2line(fp.peak_fits_r2)                         ║
    ║    → TolanskyResult                                          ║
    ║      (d_m, eps_a, eps_b, alpha_mean, N_Delta, ...)           ║
    ║                                                               ║
    ║  to_m05_priors(tol_result)                                   ║
    ║    → dict {t_init_mm, alpha_init, epsilon_cal_a, ...}        ║
    ║                                                               ║
    ║  run_staged_inversion(fp, t_eff, alpha_init, ...)            ║
    ║    → CalibrationResult                                       ║
    ║      (t_m, alpha, R_refl, R1, R2,                            ║
    ║       I0, I1, I2, sigma0, B, ne_ratio, epsilon_cal,          ║
    ║       + 1σ uncertainty on each, chi2_red, converged)         ║
    ║                                                               ║
    ║  [After all 5 cal frames:]                                    ║
    ║  average_calibrations([c1..c5])                              ║
    ║    → MasterCalibration (arithmetic mean + SEM)               ║
    ║                                                               ║
    ╚══════════════════════════════════════════════════════════════╝
    │
    ╔══ SCIENCE FRAMES ONLY ════════════════════════════════════════╗
    ║                                                               ║
    ║  v_los_prior_ms = geom.V_sc_LOS + geom.v_earth_LOS           ║
    ║                                                               ║
    ║  run_airglow_inversion(fp, master_cal, v_los_prior_ms)       ║
    ║    → AirglowResult                                           ║
    ║      (v_rel_ms, sigma_v_ms, lc_m, Y_line, B_sci,             ║
    ║       chi2_red, converged, scan_ambiguous, fsr_oi_m)         ║
    ║                                                               ║
    ║  H07: correct_los_velocity(v_rel_ms, geom)                   ║
    ║    → v_corrected                                             ║
    ║      [= v_rel_ms + V_sc_LOS + v_earth_LOS]                   ║
    ║                                                               ║
    ║  Accumulate LOSObservation                                    ║
    ║                                                               ║
    ╚══════════════════════════════════════════════════════════════╝
    │
    └─ H07: invert_wind_vector(bin_observations)
            → WindSolution
              (v_E, v_N, sigma_v_E, sigma_v_N,
               two_sigma_v_E, two_sigma_v_N,
               n_frames, gdop_flag, condition_number, ...)
```

---

## 8. Implementation plan — 6 steps

Each step produces a datestamped spec committed to `soc_sewell/specs/`
before Claude Code implementation begins.

| Step | Spec ID | What | Scope | Blocks |
|------|---------|------|-------|--------|
| 1 | S_H06_refactor | Extract `run_airglow_inversion()` from H06. Add `AirglowResult` dataclass. Preserve `main()` unchanged. | Small | 2–6 |
| 2 | S_L01 | Create `windcube/fpi_pipeline.py`: re-export core functions, canonical dataclasses, `average_calibrations()`, `process_cal_frame()`, `process_science_frame()` | Medium | 3–6 |
| 3 | S_L02 | Calibration scheduler: `build_orbit_schedule(dir) → OrbitSchedule` grouping frames by orbit with master dark/cal per orbit | Small | 4–6 |
| 4 | S_batch_v2 | Update `invert_wind_map.py`: replace `--v-rel-csv` with real H06 pipeline; keep CSV path as fallback for synthetic validation | Medium | 6 |
| 5 | S_single_v2 | Update `invert_single_frame.py`: add H06 pipeline as preferred path; keep `--v-rel` fallback | Small | 6 |
| 6 | — | End-to-end validation: 5-day uniform wind (+100/+50 m/s) using real H06 fringe fitting | — | — |

### Step 1 — AirglowResult dataclass

```python
@dataclass
class AirglowResult:
    v_rel_ms:       float   # Harding convention: + = recession from SC
    sigma_v_ms:     float   # 1σ velocity uncertainty [m/s]
    lc_m:           float   # recovered line-centre wavelength [m]
    Y_line:         float   # airglow line intensity scale factor
    B_sci:          float   # science sky background [ADU]
    sigma_lc:       float   # 1σ uncertainty on lc_m [m]
    sigma_Y:        float
    sigma_B:        float
    chi2_red:       float   # reduced chi-squared of the fit
    converged:      bool
    scan_ambiguous: bool    # True if λ_c scan had two minima within 10%
    n_bins:         int     # profile bins used in fit
    fsr_oi_m:       float   # FSR used in scan [m]

def run_airglow_inversion(
    fringe_profile,              # FringeProfile from annular_reduce
    cal,                         # MasterCalibration or CalibrationResult
    r_max_px: float = 110.0,
    v_los_prior_ms: float = 0.0,
) -> AirglowResult:
    ...
```

### Step 2 — FringeProfile normalisation

`windcube/fpi_pipeline.py` defines one canonical `FringeProfile` dataclass
(based on `annular_reduction.py`'s version). A `to_h05_fringe_profile(fp)`
shim converts to H05's internal `_FringeProfile`. No existing scripts
are modified.

---

## 9. Standalone script compatibility

All existing interactive scripts are preserved unchanged:

| Script | Role |
|--------|------|
| `center_finder.py` | Interactive centre finding with diagnostic plot |
| `annular_reduction.py` | Interactive annular reduction with L1.2 output files |
| `tolansky-2line.py` | Launcher for `src/fpi/tolansky_2026-05-13.py` |
| `H05_calibration_inversion_2026_05_12.py` | Interactive cal inversion with diagnostic figure |
| `H06_airglow_inversion_2026_05_14.py` | Interactive airglow inversion (after Step 1 refactor, `main()` calls the new library function) |

The batch pipeline (`windcube/fpi_pipeline.py`) calls the same underlying
library functions. No logic is duplicated.

---

## 10. Open questions

| ID | Question | Priority |
|----|---------|----------|
| ARCH-01 | Should `find_centre` re-run per science frame, or use the cal frame centre as a fixed seed? Re-running is more accurate but ~3× slower per frame. | Medium |
| ARCH-02 | What criteria trigger rejection of a cal frame from the master average? (`chi2_red > 3`? `converged=False`?) | Medium |
| ARCH-03 | Fallback behaviour when fewer than 5 cal frames available for an orbit (use available; flag if < 3)? | Low |
| ARCH-04 | Should `TolanskyResult` be stored in `ImageMetadata` (extending P01 v3) or remain pipeline-internal? | Low |
| ARCH-05 | In diagnostic mode, which intermediate results write to disk? (centre `.npz`, profile `.npy`, Tolansky figure, H05 figure, H06 figure) | Low |

---

## 11. Key references

- Harding, B.J., Gehrels, T.W., Makela, J.J. (2014). *Nonlinear regression
  method for estimating neutral wind and temperature from Fabry-Perot
  spectrometer data.* Applied Optics 53(4), 666–672.
- Burns, K., Adams, B.G., Longwell, J. (1950). *The first spectrum of neon.*
  JOSA 40, 339–344. [IAU standard wavelengths: 640.2248 nm, 638.2991 nm]
- Vaughan, J.M. (1989). *The Fabry-Perot Interferometer.* Adam Hilger.
  §3.5.2 — rectangular array method; Eq. 3.97 — Benoit d recovery.
- Benoit, R. (1898). Application des phénomènes d'interférence à des
  déterminations métrologiques. [Excess fractions method]
- Mulligan, F.J. (1986). A new technique for the real-time recovery of
  Fabry-Perot line profiles. J. Phys. E 19, 545.
- SI-UCAR-WC-RP-004 WindCube AOCS Design Report v1.0 (Space Inventor, 2024).
  §2.4.2.2 — Tangent Height Reference Frame (THRF).
- IC Optical Systems Ltd, GNL-4096-R (ICOS build report). [d_prior = 20.008 mm]

---

## 12. Revision history

| Version | Date | Changes |
|---------|------|---------|
| 1 | 2026-05-14 | Initial draft |
| **2** | **2026-05-14** | **Full source audit of all 5 modules completed. Added: confirmed public APIs (§6), FringeProfile mismatch and resolution (§6), v_los_prior coupling point (§5), Tolansky file location clarification (§4), arithmetic mean decision for cal averaging (§3.3), AirglowResult dataclass (§8), FringeProfile normalisation approach (§8), H07→H06 processing order (§5, §7). Corrected: removed incorrect weighted-average description from v1.** |

---

*End of WINDCUBE-ARCH-01 v2 — 2026-05-14*
