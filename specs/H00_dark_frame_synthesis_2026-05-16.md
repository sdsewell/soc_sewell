# G01_dark — Dark Frame Binary Synthesis Specification

**Spec ID:** G01_dark  
**Version:** 1.2  
**Date:** 2026-05-16  
**Author:** Claude AI / Scott Sewell (NCAR/HAO)  
**Repo:** `soc_sewell`  
**Spec file:** `docs/specs/G01_dark_frame_synthesis_2026-05-16.md`

---

## 1. Purpose

This spec defines the physically correct synthesis of dark binary frames (`.bin`) by the G01
synthetic generator. It supersedes v1.0 and incorporates measured FM instrument parameters
from WIND-XCAM-RE-00035 (WindCube FM#2 Test Report, 23/07/2025).

The immediate motivation is that G01-generated dark frames showed mean ADU ~30,000 spanning
the full 16-bit range, consistent with a warm (+20°C) focal plane and no gain division applied.
Correctly synthesised darks at the nominal orbit operating temperature of −20°C should show
mean ~557 DN with σ ~9 DN.

---

## 2. Physical Model

A dark frame is acquired with the shutter closed. Every pixel records thermal dark current
accumulated during the exposure, plus a fixed bias pedestal, plus read noise added at readout.
No photon signal is present.

The pixel value in DN for a 2×2 binned superpixel is:

```
electrons_dark = Poisson( dark_rate(T_ccd) × t_exp × N_bin )
electrons_read  = Normal(0, σ_read²)          # one draw per superpixel
total_electrons = electrons_dark + electrons_read
signal_DN       = total_electrons / gain
bias_DN         = Normal(bias_adu, bias_sigma_adu²)   # per pixel
frame_DN        = clip( round( signal_DN + bias_DN ), 0, 65535 )
```

where all quantities are defined in §2.2.

### 2.1 Dark current temperature dependence (e2v CCD97 datasheet Fig. 3)

```
dark_rate(T) = Q_do × f(T) / f(T_ref)
f(T)         = T³ × exp(−9080 / T)
```

- `T` — focal plane temperature [Kelvin] = `T_ccd_C + 273.15`
- `T_ref = 293.15 K` (20°C, datasheet reference)
- `Q_do = 400 e⁻/px/s` — datasheet typical dark current at 20°C (1×1 pixel)

**T_ccd is read from the frame metadata header** (`ccd_temp` field, in °C).
It must not be hardcoded.

### 2.2 Authoritative instrument parameters

All parameters marked **MEASURED** come from WIND-XCAM-RE-00035, FM CCD serial 17195,
signal_sample=7, 2×2 binning, ~160 kHz pixel rate — the nominal flight operating point.

| Parameter | Symbol | Value | Source |
|---|---|---|---|
| System gain | `GAIN_E_PER_DN` | **3.29 e⁻/DN** | **MEASURED** — PTC, WIND-XCAM-RE-00035 §2.1.3 |
| Read noise | `READ_NOISE_E` | **4.61 e⁻ rms** | **MEASURED** — back-clocked overscan, WIND-XCAM-RE-00035 §2.2 |
| Dark current at 20°C (1×1 px) | `Q_DO_E_PER_PX_S` | **400 e⁻/px/s** | CCD97 datasheet (typical) |
| Reference temperature | `T_REF_K` | **293.15 K** | CCD97 datasheet |
| Binning factor | `N_BIN` | **4** | 2×2 |
| Bias pedestal | `BIAS_DN` | **~275 DN** | Estimated from Fig. 6 back-clocked image colorbar, WIND-XCAM-RE-00035 §2.2. **Pending calibration measurement.** |
| Bias scatter | `BIAS_SIGMA_DN` | **2 DN** | Placeholder — update when measured |

> **Bias note:** The 275 DN bias is read from the colorbar of the back-clocked image in
> WIND-XCAM-RE-00035 Fig. 6. This is an estimate, not a precision measurement. The FM
> units will be individually calibrated (per §2.12.3.1 note) and `BIAS_DN` should be updated
> from that calibration data when available.

> **v1.0 correction:** v1.0 used gain = 7.93 e⁻/DN (estimated from full-well/14-bit ADC) and
> bias = 800 DN (made up). Both are replaced by the values above.

### 2.3 Expected dark frame values at operating temperatures

| T_ccd | Dark rate (e⁻/px/s) | Dark signal per 2×2 px per 120s (e⁻) | Dark signal (DN) | Frame mean (DN) | Frame σ (DN) |
|---|---|---|---|---|---|
| +20°C | 400 | 192,000 | 58,359 | **saturated** | — |
| 0°C | 33.5 | 16,080 | 4,888 | 5,163 | 39 |
| −10°C | 8.5 | 4,065 | 1,236 | 1,511 | 19 |
| **−20°C** | **1.93** | **926** | **282** | **~557** | **~9** |
| −30°C | 0.39 | 188 | 57 | ~332 | ~4 |

Frame mean = bias_DN + dark_DN. Frame σ = sqrt(dark_e + read_noise_e²) / gain.

At −20°C the dark frames should look like a narrow spike centered at ~557 DN with σ ~9 DN —
not 30,000 DN spanning the 16-bit range.

---

## 3. Implementation

### 3.1 Function signature

```python
def synthesise_dark_frame(
    t_exp_s:          float,
    t_ccd_c:          float,
    shape:            tuple     = (260, 276),
    rng:              np.random.Generator = None,
    gain_e_per_dn:    float     = 3.29,
    read_noise_e:     float     = 4.61,
    q_do_e_per_s:     float     = 400.0,
    t_ref_k:          float     = 293.15,
    n_bin:            int       = 4,
    bias_dn:          float     = 275.0,
    bias_sigma_dn:    float     = 2.0,
) -> np.ndarray:
    """
    Synthesise a physically correct dark frame (uint16, shape=(260,276)).

    Steps:
    1. dark_rate_px_s = _dark_rate_e_per_px_s(t_ccd_c, q_do_e_per_s, t_ref_k)
    2. dark_e = dark_rate_px_s × t_exp_s × n_bin          # mean electrons per superpixel
    3. dark_electrons = rng.poisson(dark_e, size=shape)
    4. read_electrons  = rng.normal(0, read_noise_e, size=shape)
    5. signal_dn = (dark_electrons + read_electrons) / gain_e_per_dn
    6. bias_draw = rng.normal(bias_dn, bias_sigma_dn, size=shape)
    7. frame = clip(round(signal_dn + bias_draw), 0, 65535).astype(uint16)
    """
```

### 3.2 Dark rate formula

```python
def _dark_rate_e_per_px_s(t_ccd_c, q_do=400.0, t_ref_k=293.15):
    """
    e2v CCD97 dark signal temperature dependence (datasheet Fig. 3).
    Q_d / Q_do = 1.14e6 × T³ × exp(-9080/T) / [1.14e6 × T_ref³ × exp(-9080/T_ref)]
    Simplified: f(T)/f(T_ref) where f(T) = T³ × exp(-9080/T).
    Returns e-/px/s at t_ccd_c (for a 1×1 physical pixel).
    """
    T     = t_ccd_c + 273.15
    f     = lambda t: t**3 * np.exp(-9080.0 / t)
    return q_do * f(T) / f(t_ref_k)
```

### 3.3 G01 integration

- `t_ccd_c` comes from `ImageMetadata.ccd_temp` (°C). Never hardcode.
- `t_exp_s` comes from `ImageMetadata.exp_time_s`.
- RNG draws are appended after existing per-frame draws (PE×4, etalon×4, CCD×1).
- New draw order: dark Poisson (H×W), read noise (H×W), bias (H×W).

---

## 4. Validation Checks

| Check | Criterion |
|---|---|
| `_dark_rate_e_per_px_s(-20.0)` | 1.5 – 2.5 e⁻/px/s |
| `_dark_rate_e_per_px_s(20.0)` | 350 – 450 e⁻/px/s |
| Frame dtype | uint16 |
| Frame shape | (260, 276) |
| All pixels ≥ 0 | assert |
| Frame mean at −20°C, 120s | 480 – 650 DN |
| Frame σ at −20°C, 120s | 5 – 15 DN |
| Frame mean at +20°C, 120s | > 16383 (saturated) |

---

## 5. Open Items

| Item | Status |
|---|---|
| Bias pedestal precise measurement | Pending FM calibration data |
| Bias spatial non-uniformity (DSNU) | Not yet modelled; add when data available |
| Clock-induced charge (CIC) | Not yet modelled; may be needed for cooled operation |
| Per-pixel dark non-uniformity (DSNU = 60 e⁻/px/s at 20°C, datasheet) | Deferred |

---

## 6. Revision History

| Version | Date | Summary |
|---|---|---|
| 1.0 | 2026-05-16 | Initial spec. Identified G01 dark synthesis bug. |
| 1.2 | 2026-05-16 | Added §8 G01 integration contract and §9 binning mode support (1×1 and 2×2); shape, n_bin, file size, and validation criteria for both modes |
| 1.1 | 2026-05-16 | Updated with FM measured parameters from WIND-XCAM-RE-00035: gain 7.93→3.29 e⁻/DN (PTC measured), read noise 4.5→4.61 e⁻ (FM CCD measured), bias 800→~275 DN (Fig. 6 estimate). Correct dark frame at −20°C: ~557 DN mean, σ ~9 DN. |

---

## 7. Claude Code Implementation Prompt

Commit this spec to `docs/specs/` first, then paste the following.

---

```
cat PIPELINE_STATUS.md

# ══════════════════════════════════════════════════════════════════════
# G01_dark v1.1 — Fix dark frame synthesis using FM measured parameters
# Spec: docs/specs/G01_dark_frame_synthesis_2026-05-16.md
# ══════════════════════════════════════════════════════════════════════

# ── STEP 0: Find G01 and current dark synthesis code ─────────────────
find . -name "G01_synthetic*.py" -not -path "*/\.*"
# Read the file and find:
#   (a) where img_type == 'dark' frames are synthesised
#   (b) current dark/noise implementation
#   (c) ccd_temp field name and units in ImageMetadata
#   (d) exp_time_s field name

# ── STEP 1: Add _dark_rate_e_per_px_s() ──────────────────────────────
# def _dark_rate_e_per_px_s(t_ccd_c, q_do=400.0, t_ref_k=293.15):
#     T = t_ccd_c + 273.15
#     f = lambda t: t**3 * np.exp(-9080.0 / t)
#     return q_do * f(T) / f(t_ref_k)

# ── STEP 2: Add synthesise_dark_frame() per Spec §3.1 ────────────────
# Parameters with defaults:
#   gain_e_per_dn  = 3.29   (MEASURED: PTC, WIND-XCAM-RE-00035 §2.1.3)
#   read_noise_e   = 4.61   (MEASURED: back-clocked, WIND-XCAM-RE-00035 §2.2)
#   q_do_e_per_s   = 400.0  (CCD97 datasheet)
#   t_ref_k        = 293.15
#   n_bin          = 4
#   bias_dn        = 275.0  (estimated from WIND-XCAM-RE-00035 Fig. 6)
#   bias_sigma_dn  = 2.0
# Steps 1-7 exactly as in spec docstring.

# ── STEP 3: Wire into G01 ─────────────────────────────────────────────
#   frame = synthesise_dark_frame(
#       t_exp_s     = metadata.exp_time_s,
#       t_ccd_c     = metadata.ccd_temp,   # °C from metadata
#       shape       = (260, 276),
#       rng         = rng,
#   )
# Preserve RNG draw order per Spec §3.3.

# ── STEP 4: Validation tests (validation/test_g01_dark.py) ───────────
# test_dark_rate_minus20:  1.5 < rate < 2.5
# test_dark_rate_plus20:   350 < rate < 450
# test_dark_frame_minus20: dtype=uint16, shape=(260,276), 480<=mean<=650, 5<=std<=15
# test_dark_frame_plus20:  mean > 16383 (saturated)
# Run: python -m pytest validation/test_g01_dark.py -v

# ── REPORT BACK ──────────────────────────────────────────────────────
# Report:
#   DARK SYNTHESIS LOCATION : file + line numbers
#   CCD_TEMP SOURCE         : field name + units in ImageMetadata
#   T_EXP SOURCE            : field name + units
#   RNG ORDER               : confirm draw order preserved
#   TEST OUTPUT             : full pytest -v output
#   DEVIATIONS              : any differences from spec

# ── GIT COMMIT ───────────────────────────────────────────────────────
# Update PIPELINE_STATUS.md: G01_dark v1.1, tests=4/4, date=2026-05-16

git add <changed files> \
        validation/test_g01_dark.py \
        docs/specs/G01_dark_frame_synthesis_2026-05-16.md \
        PIPELINE_STATUS.md
git commit -m "fix: G01 dark synthesis v1.1 — FM measured gain and read noise

Was: electrons written as DN with wrong gain (7.93 e-/DN estimated).
Now: gain=3.29 e-/DN (PTC, WIND-XCAM-RE-00035), read_noise=4.61 e-
     (FM CCD 17195), bias=275 DN (Fig.6 estimate).
Result: -20C dark frame mean ~557 DN sigma ~9 DN (was ~30000 DN).
Also updates PIPELINE_STATUS.md"
```

---

## 8. G01 Integration Contract

This section defines exactly how `synthesise_dark_frame()` plugs into the
G01 generator pipeline. Claude Code must read `G01_synthetic_metadata_generator_*.py`
and `G01_*.md` alongside this spec before implementing.

### 8.1 Call site

`synthesise_dark_frame()` replaces the body of `_generate_dark_pixels()` in G01:

```python
def _generate_dark_pixels(
    ccd_temp1_c: float,
    exp_time_s:  float,
    rng:         np.random.Generator,
) -> np.ndarray:
    """
    Generate dark pixel array for one frame.

    Returns uint16 array shape (259, 276) — 259 image rows only.
    Row 0 (header) is NOT included; _write_bin_file() prepends it separately.
    All values clipped to 14-bit range [0, 16383] per the WindCube binary format.
    """
    frame = synthesise_dark_frame(
        t_exp_s   = exp_time_s,
        t_ccd_c   = ccd_temp1_c,
        shape     = (259, 276),      # 259 image rows, NOT 260
        rng       = rng,
    )
    # Enforce 14-bit ceiling — WindCube ADC is 14-bit (0–16383)
    return np.clip(frame, 0, 16383).astype(np.uint16)
```

### 8.2 Shape contract

| Layer | Shape | Notes |
|---|---|---|
| `synthesise_dark_frame()` output | `(259, 276)` | 259 image rows only |
| `_generate_dark_pixels()` return | `(259, 276)` uint16 | Same |
| `_write_bin_file()` pixel_array | `(259, 276)` | Receives this directly |
| Full binary frame on disk | `(260, 276)` uint16 | Row 0 = header, rows 1–259 = pixels |
| Binary file size | 143,520 bytes | 260 × 276 × 2 |

### 8.3 ADC range

The WindCube CCD is read via a **14-bit ADC** (valid range 0–16383), stored
in a 16-bit field. The top 2 bits are always zero. `synthesise_dark_frame()`
clips to 65535 (full uint16 range) and returns uint16; `_generate_dark_pixels()`
applies a second clip to 16383 to enforce the 14-bit ceiling before handing
pixels to `_write_bin_file()`.

This means that at +20°C the dark frame is genuinely saturated at the
14-bit ceiling — all pixels read 16383 DN. This is physically correct for
an uncooled detector.

### 8.4 Pixel value convention

The `BIAS_DN` default of 275 DN used in `synthesise_dark_frame()` must be
consistent with the `BIAS_ADU` constant already defined in G01 (used to fill
non-science-region pixels in all frame types). **Before implementing, verify
that `BIAS_ADU` in G01 matches the 275 DN value.** If G01 uses a different
value, use G01's `BIAS_ADU` as the `bias_dn` default in `synthesise_dark_frame()`
and document the discrepancy in the report-back.

### 8.5 Metadata fields consumed

| Field in `ImageMetadata` | Used as | Units |
|---|---|---|
| `ccd_temp1` (or equivalent) | `t_ccd_c` | °C |
| `exp_time_s` (decoded from `exp_time_cts × TIMER_PERIOD_S`) | `t_exp_s` | seconds |

Claude Code must confirm the exact field names by reading the `ImageMetadata`
dataclass definition before implementing.

### 8.6 RNG draw order

The RNG draw sequence per frame in G01 is fixed and must not be reordered:

```
Draws 1–4   : PE metadata noise (4 draws)
Draws 5–8   : Etalon metadata noise (4 draws)
Draw  9     : CCD metadata noise (1 draw, produces ccd_temp1)
              ← synthesise_dark_frame() draws begin here ←
Draw 10     : dark Poisson  — rng.poisson(dark_e, size=(259, 276))
Draw 11     : read noise    — rng.normal(0, read_noise_e, size=(259, 276))
Draw 12     : bias scatter  — rng.normal(bias_dn, bias_sigma_dn, size=(259, 276))
```

This order is a reproducibility contract. Do not change it.

### 8.7 Binary file writer unchanged

`_write_bin_file()` receives the `(259, 276)` uint16 pixel array unchanged.
No modifications to `_write_bin_file()`, `_encode_header()`, or
`_bin_filename()` are needed by this spec.


---

## 9. Binning Mode Support

`synthesise_dark_frame()` must support both binning modes. The binning mode
is read from `ImageMetadata.binning` (derived field: 2 if `cols==276`,
1 if `cols==552`).

### 9.1 Frame dimensions by binning mode

| Binning | Total frame | Image rows | Image cols | n_bin | Binary file size |
|---|---|---|---|---|---|
| 2×2 | 260 × 276 | **259** | **276** | **4** | 143,520 bytes |
| 1×1 | 528 × 552 | **527** | **552** | **1** | 582,912 bytes |

Source: WIND-XCAM-RE-00035 §2.8 (Figs. 22–23); P01 binary format spec.

The `shape` argument to `synthesise_dark_frame()` is always the **image
rows only** (no header row). `_write_bin_file()` prepends the header row.

### 9.2 n_bin — physical pixels per output pixel

`n_bin` is the number of physical CCD pixels that contribute to one output
pixel. Dark current accumulates independently in each physical pixel, so
the total dark electrons per output pixel scales with `n_bin`:

```
n_bin = binning ** 2        # 4 for 2×2,  1 for 1×1

dark_e_per_output_px = dark_rate(T_ccd) × t_exp_s × n_bin
```

At −20°C, 120s:
- 2×2 pixel: 926 e⁻ (n_bin=4)
- 1×1 pixel: 232 e⁻ (n_bin=1)

### 9.3 Gain across binning modes

The PTC gain of **3.29 e⁻/DN** was measured in 2×2 binning mode
(WIND-XCAM-RE-00035 §2.1). In 1×1 mode the physical pixel is smaller and
collects fewer photons per readout, but the same charge-to-voltage conversion
and ADC apply. The gain in e⁻/DN is therefore the same in both modes — the
difference is entirely captured by `n_bin` in the dark current calculation.

> **Pending:** XCAM has not provided a 1×1 PTC measurement. Use 3.29 e⁻/DN
> for both modes until a 1×1 PTC is available. Flag in report-back if this
> assumption is incorrect.

### 9.4 Updated `_generate_dark_pixels()` call site

```python
def _generate_dark_pixels(
    ccd_temp1_c: float,
    exp_time_s:  float,
    binning:     int,          # 1 or 2, from ImageMetadata.binning
    rng:         np.random.Generator,
) -> np.ndarray:
    """
    Returns uint16 image array (no header row), clipped to 14-bit range.

    Shape:
        binning=2 → (259, 276)   binary file: 143,520 bytes
        binning=1 → (527, 552)   binary file: 582,912 bytes
    """
    _SHAPES = {2: (259, 276), 1: (527, 552)}
    shape  = _SHAPES[binning]
    n_bin  = binning ** 2

    frame = synthesise_dark_frame(
        t_exp_s    = exp_time_s,
        t_ccd_c    = ccd_temp1_c,
        shape      = shape,
        rng        = rng,
        n_bin      = n_bin,
    )
    return np.clip(frame, 0, 16383).astype(np.uint16)
```

### 9.5 Updated RNG draw order

Draw counts scale with frame size:

| Binning | Draw 10 (Poisson) | Draw 11 (read noise) | Draw 12 (bias) |
|---|---|---|---|
| 2×2 | 259×276 = 71,484 draws | 71,484 draws | 71,484 draws |
| 1×1 | 527×552 = 290,904 draws | 290,904 draws | 290,904 draws |

The draw *count* changes but the *order* (Poisson → read → bias) is fixed
and must not change.

### 9.6 Updated validation checks

| Check | 2×2 criterion | 1×1 criterion |
|---|---|---|
| Frame shape | (259, 276) | (527, 552) |
| Binary size | 143,520 bytes | 582,912 bytes |
| All pixels in [0, 16383] | ✓ | ✓ |
| Mean at −20°C, 120s | 480–650 DN | 260–380 DN |
| σ at −20°C, 120s | 5–15 DN | 3–10 DN |

The 1×1 mean is lower because `n_bin=1` accumulates only ¼ the dark
current of a 2×2 superpixel.

