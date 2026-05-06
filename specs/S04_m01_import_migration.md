# S04 — m01 Import Migration and Archive Cleanup

**Task file:** `docs/specs/S04_m01_import_migration.md`
**Project:** WindCube FPI Pipeline
**Institution:** NCAR / High Altitude Observatory (HAO)
**Status:** Authoritative
**Prerequisite:** S03 constants consolidation complete; H01 and H02 passing

---

## Objective

M03, M04, M05, M06, and M07 currently import from old dated `m01_*` module
files (`src/fpi/m01_airy_forward_model_2026_04_05.py` and/or
`src/fpi/m01_airy_forward_model_2026_04_26.py`). These files were moved to
`src/fpi/archive/` in a previous session but the move was never committed
and the imports were never updated, so the files were restored from git to
keep the import chain alive.

This task:
1. Updates all M03–M07 imports to use `fpi.airy_forward_model`
   (the clean `__init__.py` re-export path).
2. Commits the archive moves for the two old dated files.
3. Deletes the old dated files from `src/fpi/` entirely (they will
   exist only in git history after this commit).

---

## Preamble — read before touching any file

1. Read this entire document.
2. Run `pytest tests/ -v --tb=short` and record the baseline.
   All currently-passing tests must still be passing when you finish.
   Note any pre-existing failures — do not introduce new ones.
3. Confirm the two old files exist at their current locations:
   ```bash
   ls src/fpi/m01_airy_forward_model_2026_04_05.py
   ls src/fpi/m01_airy_forward_model_2026_04_26.py
   ```
   If either is missing, stop and report before proceeding.

---

## Task A — Audit all imports of old m01 files

Run a thorough grep across the entire repo:

```bash
grep -r "m01_airy_forward_model" . --include="*.py" -l
grep -r "from src.fpi.m01" . --include="*.py"
grep -r "import m01" . --include="*.py"
grep -r "airy_forward_model_2026_04" . --include="*.py"
```

Record every file found. The expected set is M03, M04, M05, M06, M07
plus possibly their test files. If any file outside that set appears,
stop and report before continuing — do not modify unexpected files.

---

## Task B — Update imports in M03, M04, M05, M06, M07

For each module identified in Task A, replace every import of the old
dated m01 path with the clean re-export path:

**Before (any variant of):**
```python
from src.fpi.m01_airy_forward_model_2026_04_05 import ...
from src.fpi.m01_airy_forward_model_2026_04_26 import ...
from fpi.m01_airy_forward_model_2026_04_05 import ...
from fpi.m01_airy_forward_model_2026_04_26 import ...
import src.fpi.m01_airy_forward_model_2026_04_05 as ...
```

**After:**
```python
from fpi.airy_forward_model import ...
```

Rules:
- Import only the names actually used in that module — do not use `*`.
- Preserve the original `as` alias if one was used.
- Do not change anything else in the file.
- If a module imports something from the old m01 that does not exist
  in the current `fpi.airy_forward_model`, stop and report — do not
  silently drop an import.

After updating each file, run its tests immediately to catch breakage
early:
```bash
pytest tests/test_<module_name>*.py -v --tb=short
```

---

## Task C — Run the full test suite (mid-task checkpoint)

```bash
pytest tests/ -v --tb=short
```

All tests that were passing at baseline must still pass.
Fix any import-related failures before proceeding to Task D.
If a non-import failure appears, stop and report.

---

## Task D — Archive and delete old dated m01 files

Now that no live code imports the old files, remove them cleanly.

1. Confirm `src/fpi/archive/` exists. Create it if not:
   ```bash
   mkdir -p src/fpi/archive
   ```

2. Move the two old files into archive:
   ```bash
   git mv src/fpi/m01_airy_forward_model_2026_04_05.py \
           src/fpi/archive/m01_airy_forward_model_2026_04_05.py
   git mv src/fpi/m01_airy_forward_model_2026_04_26.py \
           src/fpi/archive/m01_airy_forward_model_2026_04_26.py
   ```

3. Delete them from the archive immediately — they should live only
   in git history, not on disk:
   ```bash
   git rm src/fpi/archive/m01_airy_forward_model_2026_04_05.py
   git rm src/fpi/archive/m01_airy_forward_model_2026_04_26.py
   ```

   > Using `git mv` then `git rm` (rather than plain `rm`) gives git
   > a clean rename+delete record so the history remains navigable.
   > `git log --follow` will still find the old files by their original
   > path.

4. Confirm no stale references remain:
   ```bash
   grep -r "m01_airy_forward_model_2026_04" . --include="*.py"
   ```
   Must return nothing. If anything appears, fix it before committing.

---

## Task E — Final full test suite

```bash
pytest tests/ -v --tb=short
```

Result must be identical to (or better than) the Task C checkpoint.
No regressions permitted.

---

## Task F — Commit

Single atomic commit covering the import updates and the file deletions:

```
refactor(S04): migrate m01 imports to fpi.airy_forward_model; delete old dated files

- M03, M04, M05, M06, M07: imports updated from dated m01_* paths to
  clean fpi.airy_forward_model re-export (src/fpi/__init__.py)
- src/fpi/m01_airy_forward_model_2026_04_05.py: deleted (history preserved)
- src/fpi/m01_airy_forward_model_2026_04_26.py: deleted (history preserved)
- No functional changes to any module
All tests pass.
Closes: S04
```

---

## Report format (paste back to Claude.ai)

```
Baseline
  Tests before starting: N pass / N fail
  Old m01 files confirmed present: Yes / No

Task A — Import audit
  Files importing old m01 paths: [list]
  Unexpected files found: [list or "none"]

Task B — Import updates
  M03: updated / not needed / STOPPED (reason)
  M04: updated / not needed / STOPPED (reason)
  M05: updated / not needed / STOPPED (reason)
  M06: updated / not needed / STOPPED (reason)
  M07: updated / not needed / STOPPED (reason)
  Any missing symbols from new import path: [list or "none"]

Task C — Mid-task test suite
  Result: N/N pass
  Failures introduced: [list or "none"]

Task D — Archive and delete
  git mv + git rm completed: Yes / No
  Stale references after deletion: [list or "none"]

Task E — Final test suite
  Result: N/N pass
  Regressions vs baseline: [list or "none"]

Task F — Commit hash: [hash]
```
