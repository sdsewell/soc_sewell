# SPEC-H03 — Airglow Synthesis Script (`H03_airglow_synthesis`)

**Spec version:** 3  
**Date:** 2026-05-13  
**Status:** READY FOR IMPLEMENTATION  
**Script:** `src/processing/H03_airglow_synthesis_2026_05_13.py`  
**Previous version:** `src/processing/H03_airglow_synthesis_2026_05_12.py`  
**Library dependency:** `src/fpi/m03_airglow_synthesis_2026_05_13.py`  (SPEC-M03 v3)

---

## 1. Purpose

`H03_airglow_synthesis` is the interactive driver script for generating
synthetic OI 630 nm airglow fringe images.  It prompts the user for all
variable parameters via text dialogs, calls `synthesise_airglow_image()`,
displays the result, and optionally saves the 2D image and 1D radial
profile to `.npy` files for downstream use by H06.

---

## 2. Motivation for this revision

The SNR prompt and Gaussian noise model are replaced by physically
motivated CCD noise parameters consistent with the Teledyne e2v CCD97
datasheet.  See SPEC-M03 v3 for the noise model derivation.  The user
now specifies focal-plane temperature and exposure time; all other noise
parameters (gain, read noise, Qdd reference) are held at CCD97 datasheet
defaults and noted in the prompt text so the user is aware of the assumed
values.

---

## 3. Dialog sequence

The existing dialog sequence is preserved with two changes: the **SNR
dialog is removed** and a **CCD noise block (two dialogs) is inserted**
in its place at the end of the source/noise step.

### Step 1 — Observation mode  *(unchanged)*

```
Dialog title: "Observation mode"
Prompt:       "Select observation mode:\n"
              "  cross_track — even orbits, LOS ⊥ track, |v| ≈ 500 m/s\n"
              "  along_track — odd orbits,  LOS ∥ track, v ≈ −7000 m/s"
Type:         string choice from {"cross_track", "along_track"}
Default:      "cross_track"
```

### Step 2 — Line-of-sight velocity  *(unchanged)*

```
Dialog title: "LOS velocity"
Prompt:       "Line-of-sight velocity v_rel (m/s)\n"
              "Cross-track default:  0 m/s\n"
              "Along-track default:  −7000 m/s"
Type:         float
Default:      0.0  (cross_track)  /  −7000.0  (along_track)
Range:        −15 000 to +15 000 m/s
```

### Step 3 — CCD binning  *(unchanged)*

```
Dialog title: "CCD binning"
Prompt:       "CCD binning mode:\n"
              "  2  — 2×2 binned (256×256, flight default, α=1.6084e-4 rad/px)\n"
              "  1  — 1×1 full-frame (512×512, ground test, α=8.042e-5 rad/px)"
Type:         integer choice from {1, 2}
Default:      2
```

### Step 4 — Etalon gap  *(unchanged)*

```
Dialog title: "Etalon gap"
Prompt:       "Etalon gap t (mm)\n"
              "Tolansky-recovered default: 20.1069746 mm\n"
              "Use phase-corrected t_eff from H05 for faithful fringe placement."
Type:         float
Default:      20.1069746
Range:        19.0 to 21.0 mm
```

### Step 5 — Effective reflectivity  *(unchanged)*

```
Dialog title: "Reflectivity"
Prompt:       "Effective reflectivity R\n"
              "Default: 0.239  (R1 from H05 at λ=640.2 nm)"
Type:         float
Default:      0.239
Range:        0.05 to 0.95
```

### Step 6 — OI line intensity  *(unchanged)*

```
Dialog title: "Line intensity"
Prompt:       "OI line intensity Y_line (ADU)\n"
              "Quiet-time range: 500–2000 ADU\n"
              "Default: 6480 ADU  (H03 reference value)"
Type:         float
Default:      6480.0
Range:        100 to 30000 ADU
```

### Step 7 — Exposure time  *(NEW — replaces SNR dialog)*

```
Dialog title: "Exposure time"
Prompt:       "Exposure time t_exp (s)\n"
              "Science (airglow) default:     10 s\n"
              "Calibration (neon) typical:   120 s\n"
              "Longer exposures increase dark current noise."
Type:         float
Default:      10.0
Range:        0.1 to 300.0 s
```

### Step 8 — Focal-plane temperature  *(NEW)*

```
Dialog title: "Focal-plane temperature"
Prompt:       "Focal-plane temperature T_fp (°C)\n"
              "Design operating point:  −20 °C\n"
              "Warm on-orbit risk:      −10 °C\n"
              "Cooling failure:         +20 °C\n"
              "Range: −40 to +20 °C\n"
              "\n"
              "Dark current is computed from the Teledyne e2v CCD97 formula\n"
              "(datasheet page 4): Qdd = 400 e-/pix/s at 20 °C.\n"
              "Gain = 1.0 e-/ADU   Read noise = 2.2 e- rms (OSH, 50 kHz CDS)."
Type:         float
Default:      −20.0
Range:        −40.0 to +20.0 °C
```

---

## 4. Defaults block changes

```python
_DEFAULTS = {
    # ... existing keys unchanged ...
    'add_noise':          True,
    # REMOVED: 'noise_type', 'snr'
    # NEW:
    'exp_time_s':         10.0,
    'T_focal_plane_C':   -20.0,
    'gain_e_per_adu':      1.0,
    'read_noise_e':        2.2,
    'Qdd_at_20C':        400.0,
}
```

---

## 5. Call-site changes

### 5.1 `_ask_source_and_noise()` return dict

```python
# OLD
return dict(Y_line=Y_line, Y_bg=Y_bg, snr=snr)

# NEW
return dict(Y_line=Y_line, Y_bg=Y_bg,
            exp_time_s=exp_time_s,
            T_focal_plane_C=T_focal_plane_C)
```

### 5.2 `synthesise_airglow_image()` call

```python
result = synthesise_airglow_image(
    params            = params,
    v_rel_ms          = v_rel_ms,
    Y_line            = src['Y_line'],
    Y_bg              = src['Y_bg'],
    image_size        = instr['image_size'],
    R_bins            = _DEFAULTS['R_bins'],
    L_synth           = _DEFAULTS['L_synth'],
    n_fsr             = _DEFAULTS['n_fsr'],
    observation_mode  = obs_mode,
    add_noise         = _DEFAULTS['add_noise'],
    # REMOVED: noise_type, snr
    # NEW:
    exp_time_s        = src['exp_time_s'],
    T_focal_plane_C   = src['T_focal_plane_C'],
    gain_e_per_adu    = _DEFAULTS['gain_e_per_adu'],
    read_noise_e      = _DEFAULTS['read_noise_e'],
    Qdd_at_20C        = _DEFAULTS['Qdd_at_20C'],
    rng               = np.random.default_rng(42),
)
```

---

## 6. Figure annotation changes

### 6.1 Image panel annotation (top-left text box)

Replace the SNR line with two lines:

```
# OLD
f"SNR = {snr:.1f} (requested)   SNR_actual = {snr_act:.2f}\n"

# NEW
f"t_exp = {exp_time_s:.1f} s   T_fp = {T_fp:.1f} °C\n"
f"dark = {result['dark_rate_e_per_s']:.4f} e-/pix/s   "
f"SNR_actual = {result['snr_actual']:.2f}\n"
```

### 6.2 Histogram panel title

```
# OLD
"Pixel value histogram  (Gaussian noise → approximately Gaussian)"

# NEW
"Pixel value histogram  (Poisson + read noise → approximately Gaussian)"
```

---

## 7. Terminal print changes

```python
# OLD
print(f"  Y_line={src['Y_line']:.0f}   SNR={src['snr']:.1f}   ...")

# NEW
print(f"  Y_line={src['Y_line']:.0f}   t_exp={src['exp_time_s']:.1f}s   "
      f"T_fp={src['T_focal_plane_C']:.1f}°C   "
      f"dark={result['dark_rate_e_per_s']:.4f} e-/pix/s   "
      f"SNR_actual={result['snr_actual']:.2f}")
```

---

## 8. Saved metadata changes

In the `.npy` sidecar / print block, replace `snr_requested` / `snr_actual`
with:

```python
'exp_time_s':          src['exp_time_s'],
'T_focal_plane_C':     src['T_focal_plane_C'],
'dark_rate_e_per_s':   result['dark_rate_e_per_s'],
'dark_e_per_pixel':    result['dark_e_per_pixel'],
'snr_actual':          result['snr_actual'],     # keep for diagnostics
```

---

## 9. Acceptance criteria

| ID | Check | Pass condition |
|----|-------|---------------|
| H03-C1 | Script runs to completion with all dialogs answered | No exception |
| H03-C2 | Annotation shows `t_exp` and `T_fp` values | Values match dialog input |
| H03-C3 | Histogram title updated | "Poisson + read noise" text present |
| H03-C4 | χ²/ν from H06 on saved profile (T=−20°C, t=10s) | 0.5 ≤ χ²/ν ≤ 2.0 |
| H03-C5 | `snr_actual` printed for T=−20°C, t=10s, Y_line=6480 | 30–60 |
| H03-C6 | Profile `.npy` sidecar contains `dark_rate_e_per_s` key | present and > 0 |

---

## 10. Revision history

| Version | Date | Author | Summary |
|---------|------|--------|---------|
| 1 | 2026-05-05 | SOC | Initial spec |
| 2 | 2026-05-12 | SOC | SNR dialog; Gaussian noise |
| 3 | 2026-05-13 | SOC | Replace SNR dialog with exposure time + focal-plane temperature dialogs.  Physical CCD noise model via M03 v3. |

---

## 11. Claude Code implementation prompt

```
cat PIPELINE_STATUS.md

# Implement the H03 dialog and call-site updates for the physical noise model.
# Spec: docs/specs/SPEC-H03_airglow_synthesis_script_2026-05-13.md
# Previous script: src/processing/H03_airglow_synthesis_2026_05_12.py
# New script:      src/processing/H03_airglow_synthesis_2026_05_13.py
# Requires M03 v3 already committed as m03_airglow_synthesis_2026_05_13.py

# Steps:
# 1. Copy H03_airglow_synthesis_2026_05_12.py to _2026_05_13.py.
# 2. Update the import to reference m03_airglow_synthesis_2026_05_13.
# 3. Replace _DEFAULTS noise keys per spec §4.
# 4. Replace _ask_source_and_noise(): remove SNR dialog, add Step 7
#    (exp_time_s) and Step 8 (T_focal_plane_C) dialogs per spec §3.
# 5. Update synthesise_airglow_image() call per spec §5.2.
# 6. Update make_figure() annotation and histogram title per spec §6.
# 7. Update terminal print and saved metadata per spec §7 and §8.
# 8. Run the script non-interactively with hard-coded defaults to confirm
#    no import errors or runtime exceptions before interactive testing.
#    Use: python -c "import src.processing.H03_airglow_synthesis_2026_05_13"
#    (import-only check; dialogs will not fire).

# Report back:
# - Import check result (pass/fail)
# - List of changed function signatures and line numbers
# - Any deviations from spec required by the existing code structure

# Update PIPELINE_STATUS.md — set H03 status, date, test count.
git add PIPELINE_STATUS.md \
        src/processing/H03_airglow_synthesis_2026_05_13.py \
        docs/specs/SPEC-H03_airglow_synthesis_script_2026-05-13.md
git commit -m "feat(H03): replace SNR dialog with exp_time + focal-plane temperature

Remove SNR/Gaussian noise prompt. Add two new dialogs:
  - exposure time t_exp (s), default 10 s
  - focal-plane temperature T_fp (°C), default -20 °C
Update synthesise_airglow_image() call to pass physical CCD parameters
from M03 v3 (T_focal_plane_C, exp_time_s, gain_e_per_adu, read_noise_e,
Qdd_at_20C). Update figure annotation, histogram title, terminal print,
and saved metadata accordingly.

Also updates PIPELINE_STATUS.md"
```
