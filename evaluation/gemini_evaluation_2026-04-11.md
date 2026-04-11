# Evaluation of Gemini's WindCube SOC Recommendations
**Date:** 2026-04-11  
**Author:** Claude AI (soc_sewell project)  
**Subject:** Critical assessment of Gemini's "Strategic Master Summary & Architectural Spec"  
**Context:** Pipeline is fully implemented (S01–S20 + Z-series, 21 specs, all passing). This
evaluation asks: what, if anything, from Gemini's document merits adoption now?

---

## 1. Overall Assessment

Gemini is thinking at the right level of abstraction for a *greenfield* system. Its document reads
as if written before any code existed — it proposes infrastructure that a thoughtful architect
would want to establish on day one. The problem is that day one was many months ago. The
WindCube SOC is not a blank slate; it is a working, tested, 21-spec pipeline with defined
interfaces, passing integration tests, and an operational readiness review in May 2026.

This changes the evaluation criteria. The question is not "is this good architecture?" (often,
yes) but "does the benefit of adopting this now outweigh the cost of touching working,
tested code before launch?"

**Summary verdict by proposal:**

| Gemini Proposal | Verdict | Priority |
|----------------|---------|----------|
| ScienceDataPacket unified envelope | Partially already done; full adoption too disruptive pre-launch | Defer |
| Idempotency / pure functions | Already the design; no action needed | None |
| Provenance / Lineage Registry | Partially done (P01/S19); gap is L2 sidecar (S20) | Low, already planned |
| Jacobian matrix requirement | Scientifically premature; wrong solver for this instrument | Reject |
| Monte Carlo sensitivity tooling | High value; partially done (INT03 noiseless mode); Z04 would extend it | Accept (scoped) |
| Static filenames / no date stamps | Reasonable for mature code; disruptive to adopt mid-stream | Defer to post-launch |
| Tiered implementation gates | Already our practice (PIPELINE_STATUS.md); no action needed | None |

---

## 2. Proposal-by-Proposal Analysis

### 2.1 ScienceDataPacket — "Data Envelopes and Sockets"

**What Gemini proposes.** Replace all inter-module interfaces with a single
`ScienceDataPacket` dataclass carrying (pixel array, σ array, 2σ array, metadata dict). No
module should receive a bare numpy array — always a packet.

**What the pipeline actually does.** The pipeline already has a principled interface
separation. `FringeProfile` carries the reduced 1D profile plus uncertainties (SEM per bin).
`CalibrationResult` carries `t_m ± σ`, `R_refl ± σ`, `alpha ± σ`, `sigma0 ± σ`, `epsilon_cal`,
`chi2_reduced`, `quality_flags`, and `converged`. `AirglowFitResult` carries `v_rel_ms ±
sigma_v_rel_ms`, `chi2_reduced`, `quality_flags`. `WindResult` carries `v_zonal_ms ±
sigma_v_zonal_ms`, `v_meridional_ms ± sigma_v_meridional_ms`, condition number, and flags.
`ImageMetadata` (P01/S19) carries the full provenance envelope per image. Every module
already propagates σ and 2σ consistently per S04.

**The gap.** The pipeline does not use a single unified envelope class — each module has its
own result type. This is by design (loose coupling, each module owns its own physics), and
it has worked well through all 21 specs and 14-check end-to-end integration.

**Verdict.** Gemini's `ScienceDataPacket` is a different architectural style, not an
improvement over what exists. Refactoring all 21 modules to a unified envelope type before
launch would: (a) touch every tested interface, (b) add a new failure mode without adding
scientific value, and (c) take weeks. The existing per-module result types serve the same
purpose more explicitly. **Defer to post-launch if Qian Wu's FORTRAN side ever needs a
unified ICD.**

---

### 2.2 Idempotency / Pure Functions

**What Gemini proposes.** Every module must be pure: same input → same output, always.
Required for Monte Carlo validity and multi-user debugging.

**What the pipeline actually does.** This is already the design. Every module takes explicit
inputs and returns explicit outputs. RNG seeding is deterministic and explicit (seed passed as
argument, never global state). The INT03 script runs identically across four modes with
bitwise-reproducible results. The `--noiseless` flag exists precisely to test the
deterministic path.

**Verdict.** Already done. No action needed.

---

### 2.3 Provenance / Lineage Registry

**What Gemini proposes.** Every data product carries a "Lineage Registry" — which AOCS
telemetry, physical constants, and code versions produced any given wind vector.

**What the pipeline actually does.** `ImageMetadata` (P01/S19) already carries spacecraft
quaternion, ephemeris, orbit number, frame sequence, `is_synthetic`, `etalon_gap_mm`,
`truth_v_los`, `adcs_quality_flag`, and `noise_seed`. The L0 sidecar JSON is written at
ingest. The planned L2 sidecar (S20) will carry `epsilon_cal`, `v_rel_ms`, `v_zonal`,
`v_meridional`, and fit quality flags.

**The genuine gap.** Code version provenance is not currently captured. If M06 is updated
and old data re-processed, there is no record of which version of `m06_airglow_inversion`
produced a given `v_rel`. This is the one area where Gemini is pointing at a real hole.

**Verdict.** The gap is real but small and has a simple fix: add a `pipeline_version` string
(from `git describe --tags`) to the L2 sidecar in S20. One field, one line of code. This is
worth doing and fits naturally into the S20 implementation. **Accept, scoped to S20.**

---

### 2.4 Jacobian Matrix Requirement

**What Gemini proposes.** The forward model must be legally required to provide a Jacobian
matrix (∂I/∂V, ∂I/∂d) to support future migration to gradient-based solvers like
Levenberg-Marquardt.

**Assessment.** This proposal contains a significant misunderstanding of the pipeline's
actual solver. M05 and M06 *already use* the Levenberg-Marquardt algorithm (via `lmfit`,
matching the Harding et al. 2014 approach exactly). The staged calibration inversion in M05
is LM throughout. M06's airglow inversion is LM. The `lmfit` library computes Jacobians
internally via finite differences — they do not need to be analytically supplied.

Furthermore, `lmfit`/`scipy.optimize` provide automatic Jacobian estimation that is
demonstrably sufficient: M05 converges reliably, chi²_red is in [0.5, 3.0], and the
noiseless round-trip in INT03 gives RMS ≈ 1.2 m/s. There is no evidence of convergence
pathology that would motivate supplying analytical Jacobians.

Analytical Jacobians for the Airy function would require differentiating through the
PSF convolution, the sub-pixel bin-averaging, and the two-line neon model — a substantial
implementation effort that would need its own validation suite. The scientific benefit is
marginal: the finite-difference Jacobians already work.

**Verdict.** Reject. This is solving a problem that does not exist in this pipeline. The
comment about "legally required" is appropriate for a ground-up system design document but
is not appropriate guidance for a tested, working pipeline. **Do not implement.**

---

### 2.5 Monte Carlo Sensitivity Tooling

**What Gemini proposes.** A "Stress Test" engine: inject stochastic perturbations (photon
shot noise, AOCS pointing jitter, detector read noise) into the forward model, run blind
inversion, analyse residuals to generate a definitive error map.

**What the pipeline actually does.** INT03's `--noiseless` and SNR=5 modes provide a
two-point Monte Carlo: (1) noiseless (geometric errors only, RMS ≈ 1.2 m/s), (2) SNR=5
(noise-dominated, RMS ≈ 10–15 m/s). This is a rudimentary but functional sensitivity test.
INT02 V7 checks σ_v against the STM budget. The pipeline does not currently sweep across
SNR values, pointing jitter amplitudes, or etalon temperature drifts.

**The genuine value.** Harding et al. (2014) — the primary literature reference for this
pipeline — runs three Monte Carlo simulations: (1) uncertainty calibration, (2) bias over
velocity/temperature range, (3) bias over SNR range. Their Fig. 7 (velocity bias vs true
velocity) and Fig. 8 (bias vs SNR) are the kind of output that reviewers and the mission
science team will expect before publication. The pipeline currently cannot produce these
figures.

**What this would actually look like for WindCube.** A Z04 validation script that:
- Loops over SNR ∈ {1, 2, 5, 10, 20, 50}
- At each SNR, runs N=100 trials (noise realizations) of Z02→M03→M06
- Plots mean bias and RMS(v_rel_rec − v_rel_truth) vs SNR
- Adds a second sweep over v_rel_truth ∈ {−400, −200, 0, +200, +400} m/s at SNR=5

This is the one Gemini suggestion with direct scientific value that is not already covered and
is tractable to implement. It does not require architectural changes — it is a validation script
that calls existing modules.

**Verdict.** Accept, scoped as Z04 (SNR sensitivity sweep) and Z05 (velocity bias
sweep). Low complexity, high science value. **Recommend implementing.**

---

### 2.6 Static Filenames / No Date Stamps

**What Gemini proposes.** Retire dated filenames (e.g., `m06_airglow_inversion_2026_04_05.py`).
Use Git history and file headers for versioning instead.

**Assessment.** This is a legitimate software engineering preference. Dated filenames are
unusual outside of HAO/NCAR conventions and do create minor friction (IDE autocomplete,
import statements that need updating when a file is revised). The argument for Git as the
version store is sound.

**However.** The pipeline has 21+ implemented modules all using dated filenames, all with
import chains that reference those names. Renaming them now would touch every import
statement in every module, every spec reference, every Claude Code prompt in the project
instructions, and the test suite. This is a purely cosmetic refactor with non-trivial risk of
introducing import errors before the ORR.

**Verdict.** The suggestion is reasonable for a new project. For this pipeline at this stage,
the cost/benefit is wrong. **Defer to a single post-launch cleanup commit.**

---

### 2.7 Tiered Implementation Gates

**What Gemini proposes.** No tier can be implemented until all dependency specs in the tier
below are marked Authoritative.

**What the pipeline actually does.** This has been the practice from the beginning.
`PIPELINE_STATUS.md` now formalises it. The spec roadmap graphic (A02) enforces it
visually. **Already done. No action needed.**

---

## 3. Recommendations: What to Actually Do

Three things from this evaluation are worth acting on, in priority order:

---

### Priority 1 — Z04: SNR Sensitivity Sweep (Monte Carlo validation)
**Complexity:** Low. New validation script calling existing Z02/M03/M06.  
**Scientific value:** High. Produces the Harding-style bias-vs-SNR curve that reviewers
will expect. Directly addresses the STM wind budget claim.  
**Spec:** Write Z04 as a new Tier 9 validation spec following the Z02/Z03 pattern.  

What it produces:
- Plot of RMS(v_rel error) vs SNR from 1 to 50 (log scale), with the STM 9.8 m/s budget
  marked as a horizontal line. Shows the minimum SNR required to meet spec.
- Plot of mean bias vs true wind speed at SNR=5 (−400 to +400 m/s range).
- Both figures are direct analogues of Harding et al. (2014) Figs. 7–8 for WindCube.

---

### Priority 2 — Add pipeline_version to S20 L2 sidecar
**Complexity:** Trivial. One field in the L2 sidecar JSON.  
**Value:** Closes the only genuine provenance gap Gemini identified.  

```python
import subprocess
def get_pipeline_version() -> str:
    try:
        return subprocess.check_output(
            ["git", "describe", "--tags", "--always", "--dirty"],
            cwd=pathlib.Path(__file__).parent,
        ).decode().strip()
    except Exception:
        return "unknown"
```

Add `pipeline_version: str` to the L2 sidecar dataclass in S20. One field, always populated
at write time. No downstream impact on science analysis.

---

### Priority 3 — Z05: Velocity Bias Sweep (optional, post-ORR)
**Complexity:** Low. Variant of Z04 with fixed SNR, swept v_rel_truth.  
**Value:** Confirms no systematic bias across the ±500 m/s science range.  
**Timing:** After ORR. The noiseless INT03 check (RMS ≈ 1.2 m/s across ±300 m/s)
already gives confidence here. Z05 is the publishable version of that check.

---

## 4. What Not to Do

To be explicit, these Gemini suggestions should **not** be acted on before launch:

- **Do not** refactor all modules to use a unified `ScienceDataPacket`. The existing
  per-module result types are working and tested.
- **Do not** implement analytical Jacobians. The LM solver already works. This is solving
  a non-problem at high cost.
- **Do not** rename dated source files. The cosmetic improvement does not justify the
  risk of touching every import statement before the ORR.

---

## 5. Summary Table

| Action | Source | Effort | When |
|--------|--------|--------|------|
| Z04: SNR sensitivity sweep | Gemini §5 (scoped) | ~1 session | Before ORR |
| S20: add `pipeline_version` field | Gemini §3 (scoped) | 30 min | With S20 impl |
| Z05: velocity bias sweep | Gemini §5 (scoped) | ~1 session | Post-ORR |
| ScienceDataPacket refactor | Gemini §2 | Weeks | Defer indefinitely |
| Analytical Jacobians | Gemini §4 | Weeks | Do not implement |
| Static filename refactor | Gemini §6 | ~1 day | Post-launch cleanup |
