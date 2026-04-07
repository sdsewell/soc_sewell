# WindCube FPI Pipeline — Spec Development Collaborator Guide

**Document type:** Collaborator onboarding and coordination guide
**Project:** WindCube FPI Pipeline — Science Operations Center (SOC)
**Institution:** NCAR / High Altitude Observatory (HAO)
**Maintained by:** Project lead (Scott Sewell, HAO)
**Last updated:** 2026-04-05

---

## What this document is for

This guide is for collaborators who are helping write specifications for the
WindCube FPI data reduction pipeline in their own Claude AI accounts. It gives
you everything you need to write a spec that is consistent with the rest of the
project — the right format, the right naming conventions, the right level of
detail, and the right dependencies to read first.

If you are a collaborator, read this entire document before opening a Claude AI
conversation to write your spec. It will save significant rework.

---

## 1. The big picture in two paragraphs

WindCube is a CubeSat FPI (Fabry-Pérot Interferometer) in low Earth orbit that
measures thermospheric wind speeds by observing the Doppler shift of the OI
630 nm airglow emission. The science pipeline takes raw CCD fringe images from
the spacecraft and converts them into geolocated horizontal wind vectors
(v_zonal, v_meridional) at ~250 km altitude.

The pipeline is being built entirely in Python. Every module is designed by
writing a specification document first (in Claude AI), then implementing and
testing the code in VS Code using Claude Code. This document governs the
specification writing phase.

---

## 2. The two-tool workflow

**Claude AI** (claude.ai) — used for research, design, and writing specs.
Claude AI never commits code to the repository. Collaborators use Claude AI in
their own accounts to write their assigned spec.

**Claude Code** (VS Code extension) — used by the project lead to implement
and test code from the specs. Claude Code reads the spec and writes Python.
Collaborators do not need to run Claude Code.

Your job as a collaborator: write a spec so complete and unambiguous that
Claude Code can implement it with zero guesswork.

---

## 3. The spec system: 19 specs in 9 tiers

The pipeline is documented as 19 specification files, numbered S01–S19, organised
into 9 tiers by dependency order. The tiers must be implemented in order because
each tier depends on the outputs of the tier above it.

```
Tier 0 — Foundational (S01–S04)    No dependencies. Must exist before anything else.
Tier 1 — Geometry (S05–S08)        Orbit propagation, LOS geometry, truth wind maps.
Tier 2 — FPI forward model (S09–S11) Airy physics, calibration synthesis, airglow synthesis.
Tier 3 — Reduction (S12)           2D image → 1D radial spectrum.
Tier 4 — Cal inversion (S13)       Ne lamp calibration fringe inversion.
Tier 5 — Airglow inversion (S14)   OI 630 nm science fringe inversion.
Tier 6 — Wind retrieval (S15)      Line-of-sight winds → vector wind field.
Tier 7 — Integration (S16–S17)     End-to-end validation notebooks.
Tier 8 — Data products (S18–S19)   Metadata and NetCDF output schemas.
```

The full spec roadmap with all 19 entries is attached as `windcube_spec_roadmap.png`.
Print it and keep it nearby while you work.

---

## 4. The complete spec inventory

| Spec ID | Module | Tier | Short description | Status |
|---------|--------|------|-------------------|--------|
| S01 | Repo conventions | 0 | Naming, workflow, directory structure | ✓ Written |
| S02 | Pipeline overview | 0 | Architecture, data flow, STM links | Needed |
| S03 | Physical constants | 0 | All numerical constants in one place | Needed |
| S04 | Uncertainty standards | 0 | σ, 2σ, FringeProfile conventions | Needed |
| S05 | NB00 wind map | 1 | Truth wind map T1–T4 backends | Needed |
| S06 | NB01 orbit | 1 | SGP4 orbit propagator | Needed |
| S07 | NB02 geometry | 1 | Boresight, tangent point, LOS, L1c | Needed |
| S08 | INT01 | 1 | Geometry integration notebook | Needed |
| S09 | M01 Airy model | 2 | Ideal + modified Airy function | Needed |
| S10 | M02 cal synthesis | 2 | Ne lamp calibration fringe synthesis | Needed |
| S11 | M04 + demo | 2 | OI 630 nm airglow synthesis + demo | ✓ Written |
| S12 | M03 reduction | 3 | Centre finding + annular r² reduction | Needed |
| S13 | M05 cal inversion | 4 | Staged calibration inversion | Needed |
| S14 | M06 airglow inv | 5 | Airglow fringe inversion, λ_c recovery | Needed |
| S15 | M07 wind retrieval | 6 | WLS L2 vector wind retrieval | Needed |
| S16 | INT02 | 7 | FPI instrument chain notebook | Needed |
| S17 | INT03 | 7 | Full end-to-end pipeline notebook | Needed |
| S18 | P01 metadata | 8 | Image sidecar JSON schema | Needed |
| S19 | L2 data product | 8 | NetCDF output schema | Needed |

---

## 5. What to read before writing your spec

Before writing any spec, you must read:

**Always (for every spec):**
- This document (you are reading it)
- S01 — Repo conventions (covers naming, file headers, workflow rules)
- S02 — Pipeline overview (covers the full data flow; understand where your
  module sits in the chain)

**In addition, read every spec listed in your module's "Depends on" field.**
You cannot specify an interface correctly without knowing what feeds into it
and what consumes its outputs.

**For Tier 1 specs (S05–S08):** read S01, S02, S03, S04.

**For Tier 2 specs (S09–S11):** read S01–S04 plus S05–S08.

**For Tier 3+ specs:** read everything above your tier.

If a spec you need to read is marked "Needed" in the table above (i.e. not yet
written), contact the project lead before starting. Do not guess at an interface
that hasn't been specified yet.

---

## 6. The spec format — mandatory fields


Every spec must contain the following header block at the top:

```markdown
# Sxx — [Module Name] Specification

**Spec ID:** Sxx
**Spec file:** `specs/Sxx_short_name_YYYY-MM-DD.md`
**Project:** [Project Name]
**Institution:** [Institution]
**Status:** Specification — ready for implementation
**Depends on:** [list every Sxx this spec depends on]
**Used by:** [list every Sxx or file that consumes this module's output]
**Last updated:** YYYY-MM-DD
**Created/Modified by:** [Claude AI | Copilot | Claude Code | Manual]
```

After the header, the spec body must contain these numbered sections:

```
1. Purpose         — what this module does and why it exists
2. Physical / mathematical background  — the science or engineering rationale
3. Function signatures  — every public function with full docstrings
4. Data structures  — every dataclass, dict key, or array shape
5. Verification tests  — 6–12 tests, each labelled T1, T2, …
6. Expected numerical values  — concrete numbers Claude Code can check against
7. File location in repository  — exact path in the directory tree
8. Instructions for Claude Code  — numbered implementation steps
```

Not every section will be long — a simple utility module might have a
two-sentence Purpose section — but all sections must be present.

---

## 7. The naming convention (critical)

Spec files:
```
Sxx_short_name_YYYY-MM-DD.md
```
Example: `S09_m01_airy_forward_model_2026-03-27.md`

Implementation files:
```
module_name_YYYY-MM-DD.py
```
Example: `m01_airy_forward_model_2026-03-27.py`

Both spec and implementation files must include a header field indicating which tool was used to create or modify the file: `Created/Modified by:` for specs, and `Tool:` for Python code. Allowed values are: Claude AI, Copilot, Claude Code, or Manual. Collaborators may use either Claude Code or Copilot for code implementation and editing within VS Code, but must always record the tool used.

The date in **both** the spec filename and the implementation filename is the date of the **last substantive edit to the spec**. This means when Claude Code or Copilot implements a module, the `.py` file's date matches the spec's date. If the spec
is later revised, both filename dates are updated together.

See S01 Section 2 and 3 for the full naming rules.

---

## 8. Key design decisions already made — do not reopen these

The following decisions were made during the project design phase and are
final. Do not propose alternatives in your spec without first discussing with
the project lead.

**OI 630 nm source model — delta function.**
The airglow emission line is modelled as a spectral delta function. Temperature
retrieval is not a WindCube science goal. This simplifies M06 from 5 free
parameters to 3 and keeps the forward model consistent with M02 (neon lamp).
Do not add thermal broadening to M04 or M06.

**OI rest wavelength — 630.0304 nm (air).**
Not 630.0 nm. Use the value from S03. All Doppler shift calculations use this
value.

**Etalon gap — 20.008 mm (ICOS measured).**
Not 20.0 mm. Not 20.67 mm (the FlatSat FSR-period error). The ICOS mechanical
measurement is the anchor for all M05/M06 inversion bounds.

**Depression angle — 15.73°.**
Earlier specs and presentations used 23.4° and 525 km. Both are superseded.
The correct value is arccos(6621/6881) = 15.73° at 510 km altitude with 250 km
tangent height.

**NB00 wind map classification — T1–T4 (not W1–W7).**
The T-classification maps directly to STM validation tests and is the
authoritative classification. W1–W7 labels should not appear in new specs.

**Uncertainty convention — always σ and 2σ together.**
Every fitted parameter must have both a `sigma_` field (1σ) and a `two_sigma_`
field (exactly 2 × sigma). This is defined in S04 and is non-negotiable. Any
spec that defines a fitted output must follow this convention.

**Coordinate system — ECI J2000 for orbit/LOS, ENU for wind.**
The spacecraft state is always in ECI J2000. Wind components are always in
ENU (East-North-Up) at the tangent point. The transform chain is
ENU → ECEF → ECI and is defined in NB02c. Do not invent alternative
conventions.

---

## 9. Physical constants — always cite S03

Never write a numerical value for a physical constant directly in a spec.
Instead write "from S03" and use the symbol. The values will be in S03 once
it is written. If you are writing a spec before S03 exists, use placeholders:

```
λ₀ = OI_WAVELENGTH_M  (S03 — 630.0304e-9 m)
c  = SPEED_OF_LIGHT_MS (S03 — 299,792,458 m/s)
```

The complete list of constants that will be in S03:

| Symbol | Value | Description |
|--------|-------|-------------|
| `OI_WAVELENGTH_M` | 630.0304e-9 m | OI rest wavelength, air |
| `SPEED_OF_LIGHT_MS` | 299,792,458 m/s | Speed of light, exact |
| `BOLTZMANN_J_PER_K` | 1.380649e-23 J/K | Boltzmann constant, exact |
| `OXYGEN_MASS_KG` | 2.6567e-26 kg | One oxygen-16 atom |
| `NE_WAVELENGTH_1_M` | 640.2248e-9 m | Ne primary calibration line |
| `NE_WAVELENGTH_2_M` | 638.2991e-9 m | Ne secondary calibration line |
| `NE_INTENSITY_2` | 0.8 | Ne line 2 relative intensity |
| `CCD_PIXEL_UM` | 16.0 µm | CCD97 native pixel pitch |
| `CCD_PIXELS` | 512 | CCD97 active pixels per side |
| `EARTH_OMEGA_RAD_S` | 7.2921150e-5 rad/s | Earth rotation rate |
| `WGS84_A_M` | 6,378,137.0 m | WGS84 equatorial radius |
| `WGS84_B_M` | 6,356,752.3 m | WGS84 polar radius |
| `SC_ALTITUDE_KM` | 510.0 km | WindCube nominal altitude |
| `TP_ALTITUDE_KM` | 250.0 km | OI 630 nm tangent height |
| `DEPRESSION_ANGLE_DEG` | 15.73° | arccos(6621/6881) |

---

## 10. Interface contracts between modules

The following interfaces are fixed. Your spec must be consistent with them.

**NB02c → M04 (v_rel):**
`v_rel_ms: float` — full LOS relative velocity in m/s, including spacecraft
velocity, Earth rotation, and atmospheric wind. Positive = recession (redshift).

**M04 → M03 (2D image):**
`image_2d: np.ndarray, shape (image_size, image_size), float64` — synthetic
L1A science image in ADU counts.

**M03 → M05/M06 (FringeProfile):**
A dataclass with fields `profile`, `sigma_profile`, `two_sigma_profile`,
`r_grid`, `r2_grid`, `n_pixels`, `masked`, `cx`, `cy`, `sigma_cx`, `sigma_cy`,
`two_sigma_cx`, `two_sigma_cy`, `quality_flags`. Defined fully in S12.

**M05 → M06 (CalibrationResult):**
A dataclass containing the 10 instrument parameters with σ and 2σ fields, plus
`epsilon_cal`, `epsilon_sci`, `chi2_reduced`, `quality_flags`. Defined in S13.

**M06 → M07 (AirglowFitResult):**
Contains `lambda_c_m`, `sigma_lambda_c_m`, `two_sigma_lambda_c_m`,
`v_rel_ms`, `sigma_v_rel_ms`, `chi2_reduced`, `quality_flags`. Defined in S14.

---

## 11. How to share your completed spec

When your spec is complete, provide it as a `.md` file named exactly:
```
Sxx_short_name_YYYY-MM-DD.md
```

Send the file to the project lead. Do not commit directly to the repository.
The project lead will review for consistency with S01–S04 and the interface
contracts above, then commit it to `docs/specs/` on the main branch.

If your spec requires a change to an already-committed spec (e.g. you discover
a missing constant in S03, or an interface mismatch with S12), flag this
explicitly in a comment at the top of your spec. Do not silently change
interfaces — every change to an upstream spec can break downstream modules.

---

## 12. Quality checklist before submitting your spec

Before sending your spec to the project lead, verify:

- [ ] Header block is complete (Spec ID, file, status, depends on, used by, last updated)
- [ ] All 8 required sections are present
- [ ] Every physical constant cites S03 rather than hardcoding a number
- [ ] Every fitted parameter has both `sigma_` and `two_sigma_` fields (S04)
- [ ] Function signatures include complete docstrings with parameter types and units
- [ ] The verification tests section has at least 6 tests labelled T1–T6
- [ ] Expected numerical values table is included with at least 3 entries
- [ ] File location in repository matches the directory structure in S01
- [ ] "Instructions for Claude Code" section ends with a specific `pytest` command
      and a commit message template
- [ ] No design decisions contradict Section 8 of this guide
- [ ] Filename follows `Sxx_short_name_YYYY-MM-DD.md` exactly

---

## 13. Example: what a good spec looks like

The best reference is S11 (`airglow_image_synthesis_spec.md`), which is already
in the project knowledge base. It is a merged spec covering both the M04 module
and the demo script. Study its structure — particularly how it:

- States the design decision (delta-function model) prominently before the
  physics sections, with explicit justification
- Gives the "Instructions for Claude Code" section as numbered steps ending
  with a commit message template
- Includes a table of expected numerical values with derivations
- Groups user-controllable inputs hierarchically with a clear rationale for
  what is and isn't exposed

A spec is ready for implementation when a competent Python developer who has
never seen the pipeline before could read it and write correct, tested code
without asking a single clarifying question.

---

## 14. Getting help

If you are uncertain about an interface, a design decision, or whether
something belongs in your spec or an adjacent spec, the best approach is:

1. Re-read S01 and S02 (the pipeline overview).
2. Re-read the spec for the module that produces your module's input.
3. If still unclear, write a brief summary of the ambiguity and share it
   with the project lead before proceeding. It is much easier to resolve an
   interface question before a spec is written than after.

---

*This guide is a living document. When S01 or S02 are updated, this guide
will be updated to match. Always check the `Last updated:` date at the top
against the one in your copy.*
