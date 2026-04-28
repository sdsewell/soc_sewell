# G01 — Synthetic Metadata Generator Specification

**Spec ID:** G01
**Spec file:** `docs/specs/G01_synthetic_metadata_generator_2026-04-28.md`
**Project:** WindCube FPI Science Operations Center Pipeline
**Institution:** NCAR / High Altitude Observatory (HAO)
**Status:** Spec updated — awaiting implementation
**Depends on:**
  - NB00 (`nb00_wind_map_2026_04_06.py`) — `WindMap`, `UniformWindMap`,
    `AnalyticWindMap`, `HWM14WindMap`, `StormWindMap`
  - NB01 (`nb01_orbit_propagator_2026_04_16.py`) — `propagate_orbit(t_start, duration_s, dt_s)`
  - NB02a (`nb02a_boresight_2026_04_16.py`) — `compute_los_eci(pos_eci, vel_eci, look_mode, h_target_km)`
  - NB02b (`nb02b_tangent_point_2026_04_16.py`) — `compute_tangent_point(pos_eci, los_eci, epoch, h_target_km)`
  - NB02c (`nb02c_los_projection_2026_04_16.py`) — `compute_v_rel(wind_map, tp_lat_deg, tp_lon_deg, tp_eci, vel_eci, los_eci, epoch)`
  - P01 (`p01_image_metadata_2026_04_06.py`) — `ImageMetadata`, `AdcsQualityFlags`,
    `compute_adcs_quality_flag()`, `parse_header()` (used in C21 round-trip check)
  - `src/constants.py` — `WGS84_A_M`, `EARTH_GRAV_PARAM_M3_S2`
  - `tkinter` (stdlib) — native folder-browser dialog
**Used by:**
  - Z02 (synthetic airglow image generator)
  - Z03 (synthetic neon calibration image generator)
  - Future dark frame synthesis
**References:**
  - **WindCube Mission CONOPS Document**
    Document ID: `[TBD — insert document number, e.g. WC-OPS-XXXX]`
    Issue/version: `[TBD — insert version, e.g. Issue 1.0, 2025-MM-DD]`
    Sections referenced: observation schedule, science band definition,
    calibration cadence, dark frame strategy, exposure time budget
    *(When this CONOPS document is updated, review §3.2–3.3 and §4.5 against
    the new schedule and update the version citation before re-implementing.)*
  - SI-UCAR-WC-RP-004 Issue 1.0 — AOCS Design Report (BRF/THRF/SIRF frames)
  - Drob et al. (2015), doi:10.1002/2014EA000089 — HWM14 empirical model
  - Burns, Adams & Longwell (1950) IAU neon spectroscopic standards — λ₁, λ₂
**Last updated:** 2026-04-28

> **CONOPS version note:** §4.2–4.3 observation schedule parameters and §7.1
> exposure time defaults are CONOPS-driven. Any CONOPS update requires a dated
> revision of this spec before Claude Code is re-run.

> **Revision history:**
> - v1–v9 (2026-04-16): See previous spec for full history through binary synthesis.
> - **v10 (2026-04-28): Physics alignment with Z03 v1.5. Updated constants: etalon gap
>   20.106 mm → 20.0005 mm (Benoit); plate scale 1.6071e-4 → 1.6000e-4 rad/px; R 0.53 →
>   0.725 (effective finesse ≈ 10); CAL_NE_RATIO 3.0 → rel_638 = 0.344 convention (weak/strong,
>   real-image measurement); BIAS_ADU 100 + read noise → OFFSET_ADU = 5 (combined electronic
>   floor); neon normalisation removed; CAL_PEAK_ADU retained as peak signal target.
>   All 21 checks must still pass after update.**

---

## 1. Purpose

G01 is a standalone, interactive Python script that pre-computes and saves
the complete AOCS/instrument metadata array **and** a corresponding set of
synthetic binary FPI image files for a WindCube observation campaign.

**What G01 produces per frame type:**

| Frame type | NB01 | NB02a | NB02b | NB02c | Image synthesis |
|------------|------|-------|-------|-------|-----------------|
| science | ✓ orbit state | ✓ `q`, `los_eci` | ✓ tangent point | ✓ full LOS decomposition | Airy fringe at `v_rel` Doppler |
| cal | ✓ orbit state | ✓ `q`, `los_eci` | — | — | Two-line neon (λ₁, λ₂) |
| dark | ✓ orbit state | ✓ `q`, `los_eci` | — | — | Dark current + electronic offset |

**Storage note:** ~120,000 `.bin` files × 143,520 bytes ≈ **17.3 GB** for a
30-day run at 10 s cadence. Ensure adequate disk space before running.

---

## 2. User interface — interactive prompts

Unchanged from v9. See previous spec for full prompt listing.

---

## 3. Wind map registry

Unchanged from v9. See previous spec.

---

## 4. Orbit propagation and CONOPS scheduling

Unchanged from v9. See previous spec.

---

## 5. Main metadata loop

Unchanged from v9. See previous spec.

---

## 6. `ImageMetadata` field assignment

Unchanged from v9. See previous spec.

---

## 7. Binary image synthesis

### 7.1 Constants (updated in v10)

The following block replaces the v9 constants block in its entirety. Changed values are
marked with `# UPDATED`.

```python
# ── FPI optical model — authoritative values, aligned with Z03 v1.5 ────────
LAMBDA_OI_M      = 630.0e-9          # OI 630.0 nm, m
LAMBDA_NE1_M     = 640.2248e-9       # Neon strong line (Burns et al. 1950)
LAMBDA_NE2_M     = 638.2991e-9       # Neon weak line  (Burns et al. 1950)
ETALON_GAP_M     = 20.0005e-3        # Benoit two-line recovery, m          # UPDATED (was 20.106e-3)
PLATE_SCALE_RPX  = 1.6000e-4         # rad/px (2×2 binned, Tolansky)        # UPDATED (was 1.6071e-4)
R_REFL           = 0.725             # Effective reflectivity; finesse ≈ 10 # UPDATED (was 0.53)
N_GAP            = 1.0               # Refractive index, air gap
C_LIGHT_MS       = 2.99792458e8      # m/s
FINESSE_F        = 4*R_REFL / (1-R_REFL)**2    # ≈ 9.63 at R=0.725

# 638/640 intensity ratio — measured from real calibration images             # UPDATED
REL_638          = 0.344             # weak (638 nm) / strong (640 nm)       # UPDATED (was CAL_NE_RATIO=3.0)

# ── CCD / pixel layout ──────────────────────────────────────────────────────
NX_PIX, NY_PIX   = 256, 256          # science region (2×2 binned)
N_ROWS_BIN       = 259               # pixel rows in binary file
N_COLS_BIN       = 276               # pixel columns in binary file
ROW_OFFSET_PIX   = 1                 # top-left row of science window in pixel array
COL_OFFSET_PIX   = 10                # top-left col of science window
OFFSET_ADU       = 5                 # electronic floor: bias + read noise combined # UPDATED (was BIAS_ADU=100)
ADU_MAX          = 16383             # 14-bit ceiling

# ── Frame signal levels ──────────────────────────────────────────────────────
SCI_PEAK_ADU     = 5000              # OI 630 nm fringe peak above offset
CAL_PEAK_ADU     = 12000             # Neon composite fringe peak above offset
DARK_REF_ADU_S   = 0.05             # Dark current at T_REF_DARK_C, ADU/px/s
T_REF_DARK_C     = -20.0            # Reference temperature for dark model, °C
T_DOUBLE_C       = 6.5              # Dark current doubling interval, °C
```

> **Constants removed in v10:**
> - `FOCAL_LENGTH_M` — not used in any pixel synthesis calculation
> - `BIAS_ADU = 100` — replaced by `OFFSET_ADU = 5`
> - `SCI_READ_NOISE`, `CAL_READ_NOISE`, `DARK_READ_NOISE` — read noise absorbed into `OFFSET_ADU`
> - `CAL_NE_RATIO = 3.0` — replaced by `REL_638 = 0.344` (opposite convention: weak/strong)

> **Why OFFSET_ADU = 5 instead of BIAS_ADU = 100?**  
> The CCD97 operating in EM gain mode at −20°C shows a post-subtraction pedestal of ~5 ADU
> in real images. The bias (deterministic DC offset) and read noise (stochastic rms) are both
> small and effectively constant at these operating conditions. Bundling them into a single
> additive constant `OFFSET_ADU` simplifies the noise model without loss of fidelity. Dark
> current is handled separately because it is genuinely variable (temperature- and time-dependent).

> **Why REL_638 = 0.344 with weak/strong convention?**  
> Previous versions used `CAL_NE_RATIO = 3.0` (strong/weak, so strong line was 3× weak).
> Z03 and the pipeline fitter use the opposite convention: `rel_638` = weak/strong intensity.
> Radial-profile averaging of real WindCube calibration images gives REL_638 = 0.344.
> The two conventions are not the same number (3.0 ≠ 1/0.344 = 2.91), and the old value
> appears to have been a rough prior rather than a measurement.

> **Why R_REFL = 0.725?**  
> The FlatSat-measured coating reflectivity was 0.53, giving finesse ≈ 4.9. Real images
> show sharper fringes than this would predict, suggesting either the coating performs better
> in flight or that other instrumental contrast effects reduce the effective finesse less than
> expected. An effective R = 0.725 gives finesse N_R = π√R/(1−R) ≈ 10, which matches the
> observed fringe sharpness. PSF broadening is not applied separately — it is absorbed into
> the effective R.

> **Why ETALON_GAP_M = 20.0005e-3?**  
> The Benoit excess-fraction two-line recovery gives d = 20.0005 mm. The previously used
> value of 20.106 mm was the Tolansky operational gap (which includes thermal and assembly
> compression effects relative to the ICOS spacer). For a full-mission simulation at the
> level of fidelity G01 provides, the Benoit value is the appropriate default. The difference
> (~0.1 mm) shifts the fringe pattern center phase and ring positions. Both specs now use
> the same default.

### 7.2 Mixed-endian encoding helpers

Unchanged from v9.

### 7.3 Header encoder — `_encode_header()`

Unchanged from v9.

### 7.4 Pixel image generators (updated in v10)

All generators return `np.ndarray` shape `(256, 256)` `dtype=np.uint16`, clipped to
`[0, ADU_MAX]`. The noise model is updated to match Z03 v1.5: no separate Gaussian read noise
draw; `OFFSET_ADU` added as a fixed constant.

#### Science — Airy fringe with Doppler shift (updated)

```python
def _generate_science_pixels(v_rel_ms: float, rng) -> np.ndarray:
    """
    OI 630 nm ideal Airy fringe pattern with Doppler shift v_rel_ms.
    λ_obs = LAMBDA_OI_M × (1 + v_rel_ms / C_LIGHT_MS)
    Positive v_rel (recession) → λ increases → fringes shift inward.
    Uses ideal Airy (no PSF broadening); effective R absorbs instrumental contrast.
    """
    lambda_obs = LAMBDA_OI_M * (1.0 + v_rel_ms / C_LIGHT_MS)
    x  = np.arange(NX_PIX) - NX_PIX / 2.0
    y  = np.arange(NY_PIX) - NY_PIX / 2.0
    XX, YY = np.meshgrid(x, y)
    theta     = np.sqrt(XX**2 + YY**2) * PLATE_SCALE_RPX
    cos_theta = np.cos(theta)
    phase     = 4.0 * np.pi * N_GAP * ETALON_GAP_M * cos_theta / lambda_obs
    I_airy    = 1.0 / (1.0 + FINESSE_F * np.sin(phase / 2.0)**2)
    signal    = SCI_PEAK_ADU * I_airy
    image     = rng.poisson(np.clip(signal, 0, None)).astype(float) + OFFSET_ADU
    return np.clip(np.round(image), 0, ADU_MAX).astype(np.uint16)
```

**Changes from v9:** Gaussian read noise draw removed. `BIAS_ADU` → `OFFSET_ADU`.

#### Calibration — two-line neon (updated)

```python
def _generate_cal_pixels(rng) -> np.ndarray:
    """
    Two superimposed ideal Airy patterns:
      λ₁ = 640.2248 nm (strong, amplitude 1.0)
      λ₂ = 638.2991 nm (weak,   amplitude REL_638 = 0.344)
    Composite profile:  I_cal = (A(λ₁) + REL_638 × A(λ₂)) / (1 + REL_638)
    Peak above offset targets CAL_PEAK_ADU.
    Source: Burns, Adams & Longwell (1950). Ratio from real-image measurement.
    """
    x  = np.arange(NX_PIX) - NX_PIX / 2.0
    y  = np.arange(NY_PIX) - NY_PIX / 2.0
    XX, YY = np.meshgrid(x, y)
    theta     = np.sqrt(XX**2 + YY**2) * PLATE_SCALE_RPX
    cos_theta = np.cos(theta)

    def _airy(lam):
        phase = 4.0 * np.pi * N_GAP * ETALON_GAP_M * cos_theta / lam
        return 1.0 / (1.0 + FINESSE_F * np.sin(phase / 2.0)**2)

    I_cal  = (_airy(LAMBDA_NE1_M) + REL_638 * _airy(LAMBDA_NE2_M)) / (1.0 + REL_638)
    signal = CAL_PEAK_ADU * I_cal
    image  = rng.poisson(np.clip(signal, 0, None)).astype(float) + OFFSET_ADU
    return np.clip(np.round(image), 0, ADU_MAX).astype(np.uint16)
```

**Changes from v9:**
- `CAL_NE_RATIO = 3.0` (strong/weak) replaced by `REL_638 = 0.344` (weak/strong).
- Normalisation denominator changed from `(CAL_NE_RATIO + 1.0)` to `(1.0 + REL_638)`.
  The composite fringe still peaks at 1.0 before scaling by `CAL_PEAK_ADU`, so the
  absolute signal level is unchanged — only the relative line amplitudes differ.
- Gaussian read noise draw removed. `BIAS_ADU` → `OFFSET_ADU`.

> **Normalisation note:** The `/ (1.0 + REL_638)` term keeps the composite peak at 1.0
> (same as a single ideal Airy peak) so that `CAL_PEAK_ADU` continues to mean the peak
> signal above offset in ADU. Z03 does **not** apply this normalisation — it instead works
> with per-line amplitude I₀ derived from SNR. This is a known structural difference between
> the two codes, acceptable because G01 targets a fixed ADU level while Z03 targets a user-
> specified SNR. The fringe shapes produced are identical given the same d, α, R.

#### Dark — exponential dark current model (updated)

```python
def _generate_dark_pixels(ccd_temp1_c: float, exp_time_s: float,
                           rng) -> np.ndarray:
    """
    Dark current rate doubles every T_DOUBLE_C = 6.5°C.
    dark_rate [ADU/px/s] = DARK_REF_ADU_S × 2^((T − T_REF_DARK_C) / T_DOUBLE_C)
    mean_dark = dark_rate × exp_time_s
    pixel ~ Poisson(mean_dark) + OFFSET_ADU
    """
    dark_rate = DARK_REF_ADU_S * 2.0**((ccd_temp1_c - T_REF_DARK_C) / T_DOUBLE_C)
    mean_dark = max(dark_rate * exp_time_s, 0.0)
    image = rng.poisson(mean_dark, size=(NY_PIX, NX_PIX)).astype(float) + OFFSET_ADU
    return np.clip(np.round(image), 0, ADU_MAX).astype(np.uint16)
```

**Changes from v9:** Gaussian read noise draw removed. `BIAS_ADU` → `OFFSET_ADU`.

#### Dispatcher

Unchanged from v9.

### 7.5 Binary file writer — `_write_bin_file()`

Unchanged from v9. The file layout (1-row header at row 0; 259 pixel rows at rows 1–259;
256×256 science window at [ROW_OFFSET_PIX:ROW_OFFSET_PIX+256, COL_OFFSET_PIX:COL_OFFSET_PIX+256])
**already matches Z03 v1.5** (1-row header). No change needed.

### 7.6 Filename — `_bin_filename()`

Unchanged from v9.

### 7.7 RNG draw order

Updated: Gaussian read noise draws are removed. The pixel synthesis section now makes
**2 draws per frame** (Poisson only) instead of 4 for science/cal and 2 for dark.

```
Metadata draws (9 total) — always first:
  draw 1    theta              rng.normal(0, SIGMA_POINTING_RAD)
  draws 2–4 axis raw           rng.standard_normal(3)
  draws 5–8 etalon temps       rng.normal(24.0, 0.1, 4)
  draw 9    ccd_temp1          rng.normal(-10.0, 1.0)

Pixel synthesis draws — always after draw 9:
  [science]  Poisson  65536    rng.poisson(signal, size=(256,256))
  [cal]      Poisson  65536    rng.poisson(signal, size=(256,256))
  [dark]     Poisson  65536    rng.poisson(mean_dark, size=(256,256))
```

> **Note:** Removing the Gaussian read noise draws changes the RNG sequence for all frames
> from v9. Any run intended to reproduce v9 output exactly must use the v9 code.
> New runs should use v10.

---

## 8. Output files

Unchanged from v9.

---

## 9. Progress reporting and verification

### 9.1 Console output

Unchanged from v9.

### 9.2 Verification checks (C1–C21)

Checks C1–C19 and C21 are unchanged from v9.

**C20** is updated: pixel values in calibration frames should now have a floor of `OFFSET_ADU`
(5 ADU) rather than `BIAS_ADU` (100 ADU). The minimum pixel value in any frame should be ≥ 4.

---

## 10. File location in repository

```
soc_sewell/
├── validation/
│   ├── gen01_synthetic_metadata_generator_2026_04_16.py
│   └── (outputs in user-selected folder outside repo)
└── docs/specs/
    └── G01_synthetic_metadata_generator_2026-04-28.md
```

---

## 11. Constants cross-reference — Z03 v1.5 alignment

This table is the authoritative cross-check. Any future edit to either spec that changes
a shared constant must update the other spec in the same commit (per Z03 LD-5).

| Constant | G01 v10 symbol | Z03 v1.5 symbol | Value |
|----------|---------------|----------------|-------|
| Etalon gap | `ETALON_GAP_M` | `d_mm` default | 20.0005 mm |
| Plate scale (2×2) | `PLATE_SCALE_RPX` | `alpha` default | 1.6000e-4 rad/px |
| Effective reflectivity | `R_REFL` | `R` default | 0.725 |
| Finesse coefficient | `FINESSE_F` | `F_coef` (derived) | 9.63 |
| 638/640 ratio | `REL_638` | `rel_638` default | 0.344 |
| Electronic offset | `OFFSET_ADU` | `OFFSET_ADU` | 5 ADU |
| Dark ref rate | `DARK_REF_ADU_S` | `DARK_REF_ADU_S` | 0.05 ADU/px/s |
| Dark ref temp | `T_REF_DARK_C` | `T_REF_DARK_C` | −20.0°C |
| Dark doubling interval | `T_DOUBLE_C` | `T_DOUBLE_C` | 6.5°C |
| Ne 640.2 nm | `LAMBDA_NE1_M` | `LAM_640` | 640.2248e-9 m |
| Ne 638.3 nm | `LAMBDA_NE2_M` | `LAM_638` | 638.2991e-9 m |
| Refractive index | `N_GAP` | `N_REF` | 1.0 |
| ADU ceiling | `ADU_MAX` | `16383` | 16383 |

**Known structural difference (not a bug):** G01 normalises the composite cal profile by
`(1 + REL_638)` to keep the peak at `CAL_PEAK_ADU`. Z03 works from user-specified SNR and
derives per-line I₀ via `I_peak / (1 + rel_638)`. The Airy shapes are identical; only the
absolute scaling route differs.

---

## 12. Instructions for Claude Code

### Preamble
```bash
cat PIPELINE_STATUS.md
```

### Prerequisite reads
1. This spec in full (§§1–11).
2. Z03 v1.5 spec — confirm constant alignment against §11 table.
3. The existing `gen01_synthetic_metadata_generator_2026_04_16.py` — §7 only (image synthesis).
4. P01 `parse_header()` — confirm 1-row header format still matches.
5. `CLAUDE.md` at repo root.

### Prerequisite tests
```bash
pytest tests/test_nb01_orbit_propagator.py -v    # 8/8
pytest tests/test_nb02_geometry_2026_04_16.py -v # 10/10
pytest tests/test_s19_p01_metadata.py -v         # 8/8
```

### Critical rules
- All rules from v9 carry forward.
- **Do not change orbit propagation, metadata assembly, or file layout** — only §7.1 and
  the three pixel generators change.
- **RNG sequence changes** — Gaussian draws removed from all three pixel generators.
  This is expected and acceptable; document it in the commit message.
- **OFFSET_ADU = 5 replaces BIAS_ADU = 100 everywhere in pixel generators.**
  Do not leave any reference to `BIAS_ADU` in pixel synthesis code.
- **CAL_NE_RATIO is removed.** Replace all references with `REL_638 = 0.344`.
- **Gaussian read noise draws removed** from all three pixel generators.
  The `SCI_READ_NOISE`, `CAL_READ_NOISE`, `DARK_READ_NOISE` constants are removed.

### Implementation tasks

```
# ── TASK 1 of 3: Update §7.1 constants block ─────────────────────────────
#
# In gen01_synthetic_metadata_generator_2026_04_16.py, find the constants
# block at the top of the binary image synthesis section and make these changes:
#
#   ETALON_GAP_M  = 20.106e-3  →  20.0005e-3
#   PLATE_SCALE_RPX = 1.6071e-4  →  1.6000e-4
#   R_REFL          = 0.53       →  0.725
#   FINESSE_F       = 4*R_REFL/(1-R_REFL)**2   (no change to formula; value updates automatically)
#   BIAS_ADU        = 100        →  REMOVE; add OFFSET_ADU = 5
#   CAL_NE_RATIO    = 3.0        →  REMOVE; add REL_638 = 0.344
#   SCI_READ_NOISE  = 5.0        →  REMOVE
#   CAL_READ_NOISE  = 5.0        →  REMOVE
#   DARK_READ_NOISE = 5.0        →  REMOVE
#
# Also remove FOCAL_LENGTH_M if it is present and not used elsewhere.
#
# Gate: python -c "
#   import importlib.util, pathlib
#   spec = importlib.util.spec_from_file_location('g01',
#       'validation/gen01_synthetic_metadata_generator_2026_04_16.py')
#   m = importlib.util.module_from_spec(spec)
#   # Don't execute (interactive prompts); just check parse
#   print('Syntax OK')
# "

# ── TASK 2 of 3: Update pixel generators ─────────────────────────────────
#
# Replace the three pixel generator functions (_generate_science_pixels,
# _generate_cal_pixels, _generate_dark_pixels) with the implementations
# shown in §7.4 of this spec. Key changes per function:
#
# _generate_science_pixels:
#   - Remove: rng.normal(0, SCI_READ_NOISE, ...) draw
#   - Change: + BIAS_ADU  →  + OFFSET_ADU
#
# _generate_cal_pixels:
#   - Change intensity ratio logic:
#     OLD: I_cal = (CAL_NE_RATIO * _airy(LAMBDA_NE1_M) + _airy(LAMBDA_NE2_M)) / (CAL_NE_RATIO + 1.0)
#     NEW: I_cal = (_airy(LAMBDA_NE1_M) + REL_638 * _airy(LAMBDA_NE2_M)) / (1.0 + REL_638)
#   - Remove: rng.normal(0, CAL_READ_NOISE, ...) draw
#   - Change: + BIAS_ADU  →  + OFFSET_ADU
#
# _generate_dark_pixels:
#   - Remove: rng.normal(0, DARK_READ_NOISE, ...) draw
#   - Change: + BIAS_ADU  →  + OFFSET_ADU
#
# Gate: Run a short test (1-day, default params) and inspect one cal .bin:
#   import numpy as np
#   raw = np.frombuffer(open('path/to/cal.bin','rb').read(), dtype='>u2')
#   img = raw.reshape(260,276)[1:257, 10:266]  # 256x256 science window
#   print(f"min={img.min()} max={img.max()} mean={img.mean():.1f}")
#   # Expect: min ≥ 4, max ≤ 16383, mean ~few hundred (fringe + offset)

# ── TASK 3 of 3: Verify checks C1–C21 and commit ─────────────────────────
#
# Run G01 with default parameters (1-day run for speed) and verify all
# applicable checks pass. Pay particular attention to:
#   C20: all .bin files exist and are 143,520 bytes
#   C21: header round-trip — lua_timestamp matches CSV
#
# Check that cal frame pixel floor is now ~5 ADU (not ~100 ADU).
# Check that the cal fringe pattern is visible (max >> 5 ADU).
#
git add validation/gen01_synthetic_metadata_generator_2026_04_16.py \
        docs/specs/G01_synthetic_metadata_generator_2026-04-28.md \
        PIPELINE_STATUS.md
git commit -m "feat: G01 v10 — physics alignment with Z03 v1.5

Updated constants: ETALON_GAP_M 20.106→20.0005 mm (Benoit),
PLATE_SCALE_RPX 1.6071e-4→1.6000e-4 rad/px, R_REFL 0.53→0.725
(effective finesse≈10), REL_638=0.344 (real-image measurement,
weak/strong convention, replaces CAL_NE_RATIO=3.0 strong/weak).
OFFSET_ADU=5 replaces BIAS_ADU=100; Gaussian read noise draws
removed from all three pixel generators. RNG sequence changes
from v9 are expected and documented. C1–C21 all pass.

Aligns with Z03 v1.5 per LD-5 cross-spec consistency rule."
```

---

*End of G01 Spec v10 — 2026-04-28*
