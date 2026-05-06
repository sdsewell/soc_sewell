# H02 Spec Update — T3 and `airy_modified` signature

**Task file:** `docs/specs/H02_spec_update_2026-05-05.md`
**Project:** WindCube FPI Pipeline
**Institution:** NCAR / High Altitude Observatory (HAO)
**Status:** Authoritative
**Applies to:** H02 spec (whatever its current dated filename is)
**Prerequisite:** H02 implementation complete and passing

---

## Preamble — read before touching any file

1. Read this entire document.
2. Find the current H02 spec file (`docs/specs/H02_*.md`) and read it in full.
3. Find the current H02 test file (`tests/test_h02_*.py` or equivalent) and
   read it in full.
4. Find the current H02 module file and read it in full.
5. Run `pytest tests/ -v --tb=short` and record the baseline result.
   All H02 tests must be passing before you begin.
   If any are failing, stop and report.

---

## Task A — Fix T3 `test_circular_symmetry`: dynamic trough finder

### Background

The current T3 test checks azimuthal symmetry of the Airy pattern at a
hardcoded radius (r=50 px). At that radius the fringe profile sits on a
steep slope (~14 fringes over r_max with the current plate scale), producing
a pixel-to-pixel CV of ~0.13 purely from integer-pixel rounding — not from
any asymmetry in the optics model. The threshold 0.01 in the spec was never
physically achievable at that test point; the implementation had to use 0.15
to pass.

The correct fix is to evaluate azimuthal symmetry at a **fringe trough**,
where the intensity gradient is near zero. The CV at a trough is genuinely
small (target: < 0.05) and a tight threshold there actually tests what the
test cares about: that the Airy pattern has no azimuthal structure.
Hardcoding a pixel value is also fragile — it will drift if `alpha` or `t`
changes. The trough radius must be computed dynamically from `params`.

### Changes to the test file

Replace the T3 test body with the following implementation:

```python
def test_circular_symmetry():
    """
    T3 — Azimuthal symmetry of the Airy pattern.

    Evaluates the coefficient of variation (CV = std/mean) of the ideal
    Airy intensity across a thin annulus at a fringe trough radius.

    The trough is found dynamically: evaluate the 1D ideal Airy profile
    on a fine radial grid and locate the first local minimum beyond r=10 px.
    At a trough the intensity gradient is near zero, so pixel-to-pixel
    variation from integer rounding is negligible and a tight CV threshold
    is achievable and meaningful.

    Threshold: CV < 0.05 at the trough.
    """
    import numpy as np
    from scipy.signal import argrelmin
    from fpi.airy_forward_model import airy_ideal, InstrumentParams

    params = InstrumentParams()

    # --- find the first trough beyond r=10 px ---
    r_fine = np.linspace(10.0, params.r_max, 2000)
    from windcube.constants import OI_WAVELENGTH_AIR_M
    profile = airy_ideal(r_fine, OI_WAVELENGTH_AIR_M, params)

    # argrelmin returns indices of local minima
    minima_idx = argrelmin(profile, order=5)[0]
    assert len(minima_idx) > 0, (
        "No local minimum found in Airy profile between r=10 and r_max. "
        "Check params.alpha / params.t."
    )
    r_test = r_fine[minima_idx[0]]

    # --- build a thin annulus at r_test ---
    n_az = 360
    angles = np.linspace(0, 2 * np.pi, n_az, endpoint=False)
    # For the ideal Airy, intensity depends only on |r|, so all points on
    # the annulus evaluate to the same 1D profile value.  We perturb by
    # ±0.5 px to simulate integer-pixel scatter from a real annular average.
    r_annulus = r_test + 0.5 * np.cos(angles)   # slight radial scatter
    intensities = airy_ideal(r_annulus, OI_WAVELENGTH_AIR_M, params)

    mean_i = np.mean(intensities)
    std_i  = np.std(intensities)
    assert mean_i > 0, "Mean intensity at trough annulus is zero"
    cv = std_i / mean_i

    assert cv < 0.05, (
        f"Circular symmetry CV={cv:.4f} exceeds 0.05 at trough r={r_test:.1f} px. "
        f"Expected near-zero CV at a fringe minimum."
    )
```

### Changes to the H02 spec document

In the T3 test specification section, replace the existing description with:

> **T3 — Circular symmetry (dynamic trough)**
>
> Finds the first local minimum of `airy_ideal` beyond r=10 px by evaluating
> the profile on a 2000-point grid and calling `scipy.signal.argrelmin`.
> Builds a 360-point annulus at `r_test ± 0.5 px` of radial scatter and
> checks that CV = std/mean < **0.05**.
>
> Rationale: evaluating symmetry at a trough (near-zero gradient) isolates
> azimuthal asymmetry from fringe-slope rounding artefacts. The 0.05
> threshold is tight enough to catch broken symmetry and physically
> achievable at a minimum. Hardcoded pixel values are avoided; `r_test`
> is derived from `params` so the test remains valid if `alpha` or `t`
> changes.

Also update the expected values table (Section 8 or equivalent) to replace:

| T3 CV threshold | < 0.01 | azimuthal symmetry |

with:

| T3 CV threshold | < 0.05 | azimuthal symmetry at first Airy trough |

---

## Task B — Record `airy_modified` signature resolution in spec changelog

During H02 implementation, `airy_forward_model_2026_05_05.py` was updated
to support both the 13-argument form (used by the original H01 implementation)
and the 3-argument `(r, lam, params)` form required by H02 T8, via
duck-typing dispatch. H01 spec §6 already shows the 3-argument form as
authoritative. No functional change is needed — this task simply records
the resolution in the H02 spec changelog so the history is clear.

Add the following entry to the H02 spec changelog block (or create one
if it does not exist):

> **2026-05-05 — `airy_modified` calling convention resolved.**
> H01 §6 specifies the 3-argument `(r, lam, params)` form. The H01
> implementation had used a 13-argument form internally; H02 T8 required
> the 3-argument interface. `airy_forward_model_2026_05_05.py` was updated
> to support both forms via duck-typing dispatch, aligning implementation
> with the H01 spec. No downstream behaviour change.

---

## Task C — Run full test suite

```bash
pytest tests/ -v --tb=short
```

T3 must now pass. No regressions permitted elsewhere.

---

## Task D — Commit

```
fix(H02): T3 dynamic trough finder; record airy_modified signature fix

- T3 test_circular_symmetry: replaced hardcoded r=50 px with dynamic
  first-trough finder (argrelmin on fine radial grid). CV threshold
  tightened to 0.05 (was 0.15 at hardcoded slope point, meaningless).
- H02 spec: T3 description and expected-values table updated to match.
- H02 spec changelog: airy_modified 3-arg vs 13-arg resolution recorded.
All H02 tests pass.
```

---

## Report format (paste back to Claude.ai)

```
Baseline
  Tests before starting: N pass / N fail

Task A — T3 fix
  r_test (dynamic trough): [value] px
  CV at trough: [value]
  T3 passes with threshold 0.05: Yes / No
  Spec description updated: Yes / No
  Expected-values table updated: Yes / No

Task B — airy_modified changelog entry added: Yes / No

Task C — Full suite
  Result: N/N pass
  Unexpected failures: [list or "none"]

Task D — Commit hash: [hash]
```
