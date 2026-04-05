# S01 — Repository Conventions and Workflow Rules

**Spec ID:** S01
**Spec file:** `docs/specs/S01_repo_conventions_2026-04-05.md`
**Project:** WindCube FPI Science Operations Center Pipeline
**Institution:** NCAR / High Altitude Observatory (HAO)
**Status:** Authoritative
**Depends on:** Nothing — this is the root document
**Used by:** All other specs (S02–S19) and all implementation files
**Last updated:** 2026-04-05

---

## 1. Purpose

This document defines the rules that keep the WindCube pipeline repository
consistent and reproducible as multiple contributors work on different specs
and modules — potentially in separate Claude AI accounts, separate VS Code
sessions, or across time.

Every other spec (S02–S19) and every Python implementation file must follow
the conventions defined here. When in doubt, S01 wins.

**Read this document in full before:**
- Writing any spec
- Asking Claude AI to generate or revise a spec
- Asking Claude Code to implement any module
- Opening a pull request

---

## 2. The two-tool workflow


This project uses two main AI tools for research, specification, and implementation, with collaborators permitted to use either tool for code development in VS Code.

### 2.1 Claude AI (claude.ai)

**Role:** Research, analysis, specification authoring, and planning.

Claude AI is used to:
- Research physical models and algorithms (Harding 2014, Niciejewski 1992, etc.)
- Design module interfaces and data flow
- Write and revise `.md` spec documents
- Plan test strategies and expected numerical values
- Generate reference implementations for self-contained modules

Claude AI **never** directly commits to the repository. Output from Claude AI is downloaded, reviewed, and manually placed in the repo by the project lead or a designated collaborator.

**The project knowledge base** in the designated Claude AI project is the canonical store of all spec documents. Upload every new or revised spec here immediately after committing it to the repo.

### 2.2 Claude Code and Copilot (VS Code extensions)

**Role:** Implementation, testing, debugging, and committing.

Both Claude Code and GitHub Copilot may be used by collaborators for:
- Reading specs from `specs/` before writing any code
- Implementing Python modules in `src/` or other designated folders
- Writing and running pytest test suites
- Fixing bugs discovered during testing
- Committing working, tested code to the repository

Collaborators may use either Claude Code or Copilot (or both) for code implementation and editing within VS Code. All contributors must follow the conventions in this document regardless of tool.

Neither Claude Code nor Copilot may directly modify a spec. If either tool discovers that a spec is ambiguous or incorrect, it must be flagged to the project lead, who revises the spec in Claude AI, updates the project knowledge base, and re-commits the spec before implementation resumes.

### 2.3 The handoff boundary

The only file types that cross from Claude AI to code implementation tools are `.md` spec files. The only file types that Claude Code or Copilot produce are `.py` source files, `.py` test files, `.ipynb` notebooks, and `.json`/`.nc`/`.npy` data outputs.

```
Claude AI                          Claude Code / Copilot
─────────────────                  ───────────────────────────────
Research                           Reads  specs/Sxx_*.md
  ↓                                  ↓
Write Sxx_*.md spec                Implements src/*.py
  ↓                                  ↓
Download + commit spec             Writes tests/test_*.py
  ↓                                  ↓
Upload spec to project KB          Runs pytest — all pass
                                     ↓
                                   Commits with standard message
```

---

## 3. Spec file naming convention

Every spec document follows this exact pattern:

```
Sxx_short_name_YYYY-MM-DD.md
```

Where:
- `Sxx` — two-digit spec number with leading zero: `S01`, `S02`, ... `S19`
- `short_name` — snake_case descriptor, max 4 words: `repo_conventions`,
  `airy_forward_model`, `airglow_image_synthesis`
- `YYYY-MM-DD` — the date in the `Last updated:` field inside the file,
  ISO 8601 format

**Examples:**
```
S01_repo_conventions_2026-04-05.md
S02_pipeline_overview_2026-04-05.md
S09_airy_forward_model_2026-03-27.md
S11_airglow_image_synthesis_2026-04-05.md
```

**Rules:**
- The date in the filename must always match the `Last updated:` field
  inside the file. If you change the content, change both.
- When a spec is revised, the old file is **retired to `docs/specs/archive/`**
  and the new file gets today's date. Never overwrite the old file in place —
  the archive preserves the history.
- Filenames use hyphens inside the date component (`2026-04-05`) and
  underscores everywhere else. No spaces anywhere.

---

## 4. Python implementation file naming convention

Every implementation file follows this exact pattern:

```
module_name_YYYY-MM-DD.py
```

Where the date is the date the **spec** that drove this implementation was
last updated — not the date the code was written. This creates an
unambiguous link between spec and implementation.

**Examples:**
```
fpi/m01_airy_forward_model_2026-03-27.py
fpi/m04_airglow_synthesis_2026-04-05.py
windmap/nb00_wind_map_2026-03-26.py
geometry/nb02_geometry_2026-03-27.py
```

**Exception:** If Claude AI generates a working, tested implementation in
its own sandbox (as happened with M04 in this project), the date is the
date of generation in Claude AI, and the internal header records both the
spec date and the generation date separately (see Section 5.2).

**Test files** mirror the module they test:
```
tests/test_m01_airy_forward_model_2026-03-27.py
tests/test_m04_airglow_synthesis_2026-04-05.py
```

**Demo scripts** live in `demos/` and follow the same convention:
```
demos/demo_m04_airglow_synthesis_2026-04-05.py
```

---

## 5. Internal file headers

### 5.1 Spec files (.md)


Every spec begins with this exact header block (adapt fields as needed):

```markdown
# Sxx — Short Descriptive Title

**Spec ID:** Sxx
**Spec file:** `specs/Sxx_short_name_YYYY-MM-DD.md`
**Project:** [Project Name]
**Institution:** [Institution]
**Status:** [Draft | Under review | Authoritative | Superseded]
**Depends on:** [comma-separated Sxx IDs, or "Nothing"]
**Used by:** [comma-separated Sxx IDs]
**Last updated:** YYYY-MM-DD
**Created/Modified by:** [Claude AI | Copilot | Claude Code | Manual]
```

**Status definitions:**
- `Draft` — being actively written, not yet ready for review
- `Under review` — complete but awaiting project lead sign-off
- `Authoritative` — approved; Claude Code may implement from this spec
- `Superseded` — replaced by a newer version; archived copy only

Claude Code must **refuse to implement** from a spec whose status is not
`Authoritative`. If the spec is `Draft` or `Under review`, implementation
waits until the project lead changes the status.

### 5.2 Python implementation files (.py)


Every Python module begins with this exact header block:

```python
"""
Module short description — one sentence.

Spec:         Sxx_short_name_YYYY-MM-DD.md
Spec date:    YYYY-MM-DD
Generated:    YYYY-MM-DD  (Claude AI | Copilot | Claude Code | Manual)
Tool:         [Claude AI | Copilot | Claude Code | Manual]
Last tested:  YYYY-MM-DD  (N/N tests pass, pytest X.Y)
Implements:   module/path/filename_YYYY-MM-DD.py
Depends on:   [comma-separated module names]
"""
```

The `Generated` and `Last tested` dates may differ from the `Spec date` —
that is normal and expected. The spec date is the stable anchor; the
other two dates track code and test history.

---


## 6. Repository directory structure

```
soc_sewell/
│
├── refs/                        ← reference documents, external standards, or key papers
│
├── specs/                       ← all Sxx_*.md spec files live here
│   ├── S01_repo_conventions_YYYY-MM-DD.md
│   ├── S02_pipeline_overview_YYYY-MM-DD.md
│   ├── ...
│   └── archive/                 ← superseded spec versions
│
├── src/                         ← main source code (Python modules, packages)
│   ├── __init__.py
│   └── ...
│
├── tests/                       ← all pytest test files
│   └── ...
│
├── data/                        ← reference data, calibration, and synthetic datasets
│   ├── reference/
│   └── synthetic/
│
├── notebooks/                   ← Jupyter notebooks for integration, demos, or analysis
│   └── ...
│
├── validation/                  ← scripts for validation and comparison
│
├── requirements.txt             ← pinned Python dependencies
└── README.md                    ← project overview and quick-start
```

**Import rule:** No module imports another module by its dated filename. Imports always use the base module name (e.g., `from src.module_name import SomeClass`). The dated filename is for file management only. Each package `__init__.py` re-exports from the current dated file (see Section 10).

---

## 7. Spec dependency and implementation order

Specs must be implemented in tier order. A module may not be implemented
until all specs it depends on are `Authoritative`.

| Tier | Specs | Gate condition |
|------|-------|----------------|
| 0 | S01–S04 | All Authoritative before any other spec is written |
| 1 | S05–S08 | Requires S01–S04 |
| 2 | S09–S11 | Requires S01–S08 |
| 3 | S12 | Requires S09–S11 |
| 4 | S13 | Requires S09–S12 |
| 5 | S14 | Requires S09–S13 |
| 6 | S15 | Requires S07, S14 |
| 7 | S16–S17 | Requires all prior tiers |
| 8 | S18–S19 | Can begin after S01 |

Parallel work within a tier is permitted — S05, S06, S07 can be
written simultaneously. However, Claude Code must not implement any of them
until S01–S04 are all `Authoritative` and committed to the repo.

See `docs/figures/windcube_spec_roadmap_2026-04-05.png` for the full visual
roadmap.

---

## 8. Git commit message conventions

**Spec commits:**
```
spec(Sxx): short description of what changed
Implements: S01_repo_conventions_2026-04-05.md
```

**Implementation commits:**
```
feat(module): what was implemented, N/N tests pass
Implements: Sxx_spec_name_YYYY-MM-DD.md
```

**Bug fix commits:**
```
fix(module): one-line description of the fix
```

**Test update commits:**
```
test(module): what was changed in tests
```

**Spec archive commits:**
```
archive(Sxx): retire old version YYYY-MM-DD, replace with YYYY-MM-DD
```

---

## 9. Test requirements

- All tests use `pytest`. No other framework.
- Every test file has the same date suffix as the module it tests.
- Tests must pass before Claude Code commits the implementation.
- The minimum test count per module is defined in that module's spec.
- Tests may not be skipped except for optional external dependencies
  (e.g. `hwm14`, `tiegcm`) where `pytest.importorskip` is used.
- After any change to any module, the full suite must pass:
  ```
  pytest tests/ -v
  ```

---

## 10. The `__init__.py` import pattern

Each package `__init__.py` re-exports from the current dated file so that
imports use base names only. Note that Python module filenames use
underscores (`2026_04_05`) while spec and data filenames use hyphens
(`2026-04-05`).

```python
# fpi/__init__.py  — updated each time a module gets a new dated version
from fpi.m01_airy_forward_model_2026_03_27 import *   # noqa: F401, F403
from fpi.m04_airglow_synthesis_2026_04_05  import *   # noqa: F401, F403
```

When a module is updated to a new dated version, the `__init__.py` is
updated in the same commit.

---

## 11. What to do when a spec needs to change

1. Identify the spec by its `Sxx` ID.
2. Open the current version in Claude AI (upload if not in project KB).
3. Make the changes, update the `Last updated:` date to today.
4. Download the revised spec with the new dated filename.
5. Move the old spec to `docs/specs/archive/`.
6. Commit both in one commit with message:
   `spec(Sxx): describe what changed`
7. Update the project knowledge base in Claude AI with the new version.
8. If the change affects an already-implemented module, re-implement via
   Claude Code against the revised spec.

---

## 12. Collaborator onboarding checklist

When any contributor (human or Claude AI instance) joins the project they
must complete the following before writing any spec or code:

- [ ] Read S01 (this document) in full
- [ ] Read S02 (pipeline overview) for the full data flow
- [ ] Read S03 (physical constants) to know what constants exist
- [ ] Read S04 (uncertainty standards) for σ/2σ conventions
- [ ] Review `docs/figures/windcube_spec_roadmap_*.png`
- [ ] Confirm which spec(s) they are assigned to work on
- [ ] Confirm all dependency specs for their assigned work are `Authoritative`

A Claude AI instance working on a spec should have S01–S04 **and** all
dependency specs uploaded to its project knowledge base before starting.
A Claude Code instance should have the relevant spec open in VS Code before
writing any code.

---

## 13. Spec quality checklist

Before marking any spec `Authoritative`, verify:

- [ ] Header block complete and all fields filled in
- [ ] Purpose section explains what the module does and why it exists
- [ ] All function signatures specified (parameter names, types, units)
- [ ] Return types and dict keys specified
- [ ] At least one table of expected numerical values with derivations
- [ ] Verification tests listed with explicit pass/fail criteria
- [ ] File location in repository shown
- [ ] "Instructions for Claude Code" section present and unambiguous
- [ ] Filename date matches `Last updated:` field
- [ ] All referenced dependency specs exist and are `Authoritative`

---

## 14. Project information

**Repository:** `windcube-pipeline` (private, NCAR/HAO GitHub)
**Primary contact:** Scott Farrell, HAO/NCAR
**Pipeline reference:** Harding, Gehrels & Makela (2014), Applied Optics 53(4)
**Instrument reference:** WindCube ICOS Etalon Assembly GNL-4096-R iss1
**STM reference:** WindCube Science Traceability Matrix v1
**Spec roadmap PNG:** `docs/figures/windcube_spec_roadmap_2026-04-05.png`
