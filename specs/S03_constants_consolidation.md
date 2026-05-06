# S03-update — Constants Consolidation Task

**Task file:** `docs/specs/S03_constants_consolidation.md`
**Project:** WindCube FPI Pipeline
**Institution:** NCAR / High Altitude Observatory (HAO)
**Status:** Authoritative
**Prerequisite:** H01 implementation complete (airy_forward_model_2026_05_05.py passing 15/15 tests)

---

## Objective

Establish `windcube/constants.py` as the single source of truth for all
pipeline constants. Replace `src/constants.py` with a re-export shim so
that all existing imports continue to work without modification. Correct
two known value errors. Remove `FOCAL_LENGTH_M` from the pipeline entirely.

---

## Preamble — read before touching any file

1. Read this entire task document.
2. Read `windcube/constants.py` in full.
3. Read `src/constants.py` in full.
4. Read `tests/test_s03_constants.py` in full.
5. Run `pytest tests/ -v --tb=short` and record the baseline result.
   Three failures in `test_s03_constants.py` are expected and known
   (see Task C). All other tests must be passing before you begin.
   If any unexpected tests are failing, stop and report.

---

## Task A — Correct `NE_INTENSITY_2` in `windcube/constants.py`

The authoritative value for the weak/strong neon intensity ratio is **0.36**
(Burns, Adams & Longwell 1950, IAU spectroscopic standards). The current
`src/constants.py` carries 0.8, which is physically incorrect.

1. Open `windcube/constants.py`.
2. Find `NE_INTENSITY_2`. If it is already `0.36`, record "already correct"
   and skip to Task B.
3. If it is any other value, update it to `0.36` and add the comment:
   ```python
   NE_INTENSITY_2 = 0.36   # Burns et al. (1950) IAU standard; weak/strong ratio
   ```
4. Do **not** change any other value.

---

## Task B — Remove `FOCAL_LENGTH_M` from `windcube/constants.py`

The pipeline uses `alpha` (plate scale, rad/px) recovered from the Tolansky
two-line analysis as the sole geometric parameter for the Airy forward model.
`FOCAL_LENGTH_M` is not used by any current pipeline module and must be
removed to prevent future confusion.

1. Open `windcube/constants.py`.
2. Delete the line(s) defining `FOCAL_LENGTH_M` (the constant and its comment).
3. Search the entire repo for any `import` or reference to `FOCAL_LENGTH_M`:
   ```bash
   grep -r "FOCAL_LENGTH_M" . --include="*.py"
   ```
4. For each occurrence found:
   - In `src/constants.py`: remove it (the shim in Task D will not re-export it).
   - In any test file: see Task C.
   - In any other module: remove the import and any usage, replacing with
     `alpha` / `ALPHA_RAD_PX` where appropriate. Stop and report if a module
     uses `FOCAL_LENGTH_M` in a non-trivial calculation — do not silently
     delete load-bearing code.
5. Record every file modified.

---

## Task C — Fix the three pre-existing `test_s03_constants.py` failures

The three failing tests expect constants that either do not exist or are
being removed. Fix each one:

### C1 — `OI_WAVELENGTH_VAC_M`

This constant does not exist in either constants file. The pipeline uses
the air wavelength `OI_WAVELENGTH_AIR_M = 630.0304e-9 m` throughout.

**Action:** Update the failing test to import and check `OI_WAVELENGTH_AIR_M`
instead of `OI_WAVELENGTH_VAC_M`. Verify the expected value is `630.0304e-9`.

If the test was intentionally checking for a vacuum wavelength that the
pipeline genuinely needs, stop and report rather than silently dropping it.

### C2 — `FOCAL_LENGTH_M`

This constant is being removed (Task B). The test expecting it must be
removed too.

**Action:** Delete the test case(s) that check for `FOCAL_LENGTH_M`.
Add a comment:
```python
# FOCAL_LENGTH_M removed 2026-05-05: pipeline uses ALPHA_RAD_PX (Tolansky
# plate scale) exclusively. Focal length is not a pipeline parameter.
```

### C3 — `NE_WAVELENGTH_1_VAC_M`

This constant does not exist. The pipeline uses air wavelengths
(`NE_WAVELENGTH_1_AIR_M`, `NE_WAVELENGTH_2_AIR_M`) throughout, consistent
with the Burns et al. (1950) IAU standards which are tabulated in air.

**Action:** Update the failing test to import and check
`NE_WAVELENGTH_1_AIR_M` instead of `NE_WAVELENGTH_1_VAC_M`.
Verify the expected value is `640.2248e-9`.

---

## Task D — Replace `src/constants.py` with a re-export shim

This is the consolidation step. After Tasks A–C are complete:

1. Replace the entire contents of `src/constants.py` with the following:

```python
"""
src/constants.py — RE-EXPORT SHIM. DO NOT ADD CONSTANTS HERE.

All pipeline constants live in windcube/constants.py, which is the
single source of truth. This file exists solely for import compatibility:
any module using `from src.constants import X` will continue to work
without modification.

To add or change a constant: edit windcube/constants.py only.

Migration status: shim installed 2026-05-05 (S03 consolidation task).
Final goal: update all pipeline imports to `from windcube.constants import`
and delete this file.
"""
from windcube.constants import *  # noqa: F401, F403
```

2. Do **not** modify any other file's import statements. The shim means
   every existing `from src.constants import X` continues to resolve
   correctly through the wildcard re-export.

---

## Task E — Run the full test suite

```bash
pytest tests/ -v --tb=short
```

All tests must pass. The three previously failing `test_s03_constants.py`
tests must now pass. No regressions in any other test.

If any H01 tests fail after the `NE_INTENSITY_2` correction (Task A),
that is expected — the forward model now uses the physically correct 0.36
ratio. Inspect the failure: if the test was checking for the 0.8 value
explicitly, update the expected value in the test to 0.36. If the test
is a functional test (fringe shape, round-trip, etc.) and now fails on a
physics assertion, stop and report rather than patching blindly.

---

## Task F — Commit

```
fix(constants): consolidate to windcube/constants.py; correct NE_INTENSITY_2

- windcube/constants.py is now the single source of truth
- src/constants.py replaced with re-export shim (import compatibility preserved)
- NE_INTENSITY_2 corrected: 0.8 → 0.36 (Burns et al. 1950 IAU standard)
- FOCAL_LENGTH_M removed: pipeline uses ALPHA_RAD_PX (Tolansky) exclusively
- test_s03_constants.py: fixed 3 pre-existing failures (OI_WAVELENGTH_VAC_M,
  FOCAL_LENGTH_M, NE_WAVELENGTH_1_VAC_M)
Closes: S03-update
```

---

## Report format (paste back to Claude.ai)

```
Baseline
  Tests before starting: N pass / N fail
  Known failures: test_s03_constants.py (3) — confirmed

Task A — NE_INTENSITY_2
  Previous value: [value]
  Updated to 0.36: Yes / Already correct

Task B — FOCAL_LENGTH_M removal
  Removed from windcube/constants.py: Yes / Not present
  Other files containing FOCAL_LENGTH_M: [list or "none"]
  Any load-bearing usage found: Yes (STOPPED) / No

Task C — test_s03_constants.py fixes
  C1 OI_WAVELENGTH_VAC_M → OI_WAVELENGTH_AIR_M: Done / Skipped (reason)
  C2 FOCAL_LENGTH_M test deleted: Done / Skipped (reason)
  C3 NE_WAVELENGTH_1_VAC_M → NE_WAVELENGTH_1_AIR_M: Done / Skipped (reason)

Task D — src/constants.py shim installed: Yes / No

Task E — Full test suite
  Result: N/N pass
  Unexpected failures: [list or "none"]
  H01 tests after NE_INTENSITY_2 change: N/15 pass

Task F — Commit hash: [hash]
```
