# Z01 — Validate Tolansky Analysis (1-Line and 2-Line Fringe Images)

**Spec ID:** Z01 (replaces Z01 v0.2 and Z01a v0.2)
**Spec file:** `docs/specs/Z01_validate_tolansky_analysis_2026-04-13.md`
**Project:** WindCube FPI Pipeline
**Institution:** NCAR / High Altitude Observatory (HAO)
**Status:** Specification — ready for implementation in VS Code
**Tier:** 9 — Validation Testing
**Depends on:** S12 (M03 — annular reduction, dark subtraction, peak finding),
S13 (Tolansky — TolanskyAnalyser, TwoLineAnalyser, TolanskyPipeline, SingleLineTolansky),
S19 (P01 — ImageMetadata, ingest_real_image)
**Supersedes:** Z01 v0.2 (2026-04-12), Z01a v0.2 (2026-04-12)
**Last updated:** 2026-04-13
**Created by:** Claude AI

---

## 1. Purpose

`z01_validate_tolansky_analysis_2026-04-13.py` is a unified validation and
calibration script for Fabry-Pérot fringe images that may contain either one
or two spectral components. The user selects the analysis mode at startup.

### 1.1 Two modes

**2-Line Mode** (default) — neon lamp calibration frames containing two
interleaved ring families:
- λ₁ = 640.2248 nm (air) — primary neon line (Burns et al. 1950 IAU standard)
- λ₂ = 638.2991 nm (air) — secondary neon line (Burns et al. 1950 IAU standard)
- Recovers: etalon gap `d` (mm), focal length `f` (mm), plate scale `α`
  (rad/px), fractional orders `ε₁` and `ε₂`
- Expected peak count: 20 (10 per family) for standard neon calibration images

**1-Line Mode** — a single-component fringe pattern, either:
  (a) Neon lamp passed through a 630 nm bandpass filter and baffled
      1.65° field stop — produces 6–7 fringes; zero-velocity reference
  (b) Real or synthetic OI 630.0304 nm (air) airglow image

  In 1-Line Mode, `d` and `f` are **fixed priors** (from a prior 2-Line run).
  Only `ε` is a free parameter. The recovered `ε` is the fractional interference
  order at OI 630.0 nm; when the source is a static ground lamp it equals the
  rest-frame reference `ε₀` needed by M06.

### 1.2 Wavelength notation convention

**All wavelengths must be identified as (air) or (vac) wherever they appear
in this spec, in the script output, and in all figure annotations.** The neon
and OI line positions used here are air wavelengths at standard conditions
(Burns et al. 1950; NIST ASD). Vacuum wavelengths are not used in this pipeline.

| Line | λ (air) | λ (vac) | Source | Notes |
|------|---------|---------|--------|-------|
| Ne 1 | 640.2248 nm | 640.4744 nm | Burns et al. 1950 | Primary cal line |
| Ne 2 | 638.2991 nm | 638.5481 nm | Burns et al. 1950 | Secondary cal line |
| OI   | 630.0304 nm | 630.2771 nm | NIST ASD | Science line; rest wavelength |

All constants imported from `windcube.constants`; the air wavelength values
are authoritative. Do not hardcode wavelength values in the script.

### 1.3 What this script is NOT

Z01 does not run M05 (full staged Airy inversion). It runs M03 (annular
reduction) and S13 (Tolansky), which together characterise the etalon
geometry. When run on synthetic image pairs, the recovered parameters should
match the known synthesis inputs, providing a ground-truth regression harness.

### 1.4 Review of S13 against Vaughan §3.5.2

Vaughan (1989) §3.5.2 describes the "rectangular array" reduction of Fabry-
Pérot photographic fringe patterns. The key algorithmic steps are:

1. Measure ring radii (diameters in Vaughan's older notation) for each ring p
   and each spectral component.
2. Square them: form D²_p (Vaughan) → we use r²_p (pixels²).
3. Tabulate successive differences Δ = r²_(p+1) − r²_p for each component.
   These should be constant = S = f²λ/(nd) — the "scale factor C" in Vaughan.
4. Extract ε from the fit intercept.
5. Apply Eqs 3.91–3.97 to cross-check d and λ.

**S13 implements this correctly.** The WLS fit to r²_p = S·p + b recovers S
(= C in Vaughan, but in px²/fringe using r not D) and ε = 1 + b/S. The joint
two-line analysis enforces the slope-ratio constraint S₁/S₂ = λ₁/λ₂, which
Vaughan states as a consistency check. The excess fractions formula (Benoit
1898) implements Vaughan's Eqs 3.94–3.97. No structural corrections are
needed in S13.

**What Z01 adds** beyond what S13 already computes: the explicit Vaughan
rectangular array table (in px²), the numerical Eqs 3.91–3.97 calculation
block printed to the console, the Melissinos-style r² scatter plot with 2σ
horizontal error bars, and the explicit centre-refinement reporting.

---

## 2. Physical Background — Vaughan §3.5.2 in Pixel Units

### 2.1 The Tolansky r² relation

Constructive interference for Haidinger rings (paraxial approximation):

```
r²_p = S · (p − 1 + ε)
```

where:
- `r_p` = radius of the p-th ring (px), p = 1 for innermost visible ring
- `S = f²λ / (nd)` = slope = "scale factor C" in Vaughan (px²/fringe)
  - `f` = effective focal length (px = f_mm / pixel_pitch_mm)
  - `λ` = wavelength (m, air)
  - `n` = refractive index of etalon gap (1.0 for air-spaced)
  - `d` = etalon plate separation (m)
- `ε` = fractional interference order at the centre (0 ≤ ε < 1)

The WLS fit `r² = S·p + b` gives slope S and intercept b = S(ε−1), so:

```
ε = 1 + b/S   (with ε reduced mod 1 to [0, 1))
```

### 2.2 Successive r² differences — Vaughan's "Δ" in px²

The difference between adjacent rings for the same spectral component:

```
Δ(r²)_p  =  r²_(p+1) − r²_p  =  S   (constant for all p if data are perfect)
```

This is Vaughan's Eq 3.85 (in px² rather than mm² of diameter²; a factor of
4 difference from Vaughan because we use radius r, not diameter D). The
coefficient of variation CV = std(Δ(r²)) / mean(Δ(r²)) × 100% should be
< 5% for well-fitted data; > 10% signals a bad peak or poor parallelism.

### 2.3 Vaughan's Rectangular Array — Table 3.1 in px²

For a 2-component pattern (components λ₁ and λ₂), the rectangular array
tabulates r²_p for each (component, ring) pair, forming successive column
differences and cross-component row differences. Vaughan's Table 3.1 uses
diameter² (mm²); this spec uses radius² (px²).

Structure for a 2-line analysis with N₁ rings in family λ₁ and N₂ in λ₂
(N₁ and N₂ are typically equal; if not, only the common rings are shown):

```
Component  | Ring 1         | Ring 2         | Δ₁₂       | Ring 3         | Δ₂₃       | ...
-----------+----------------+----------------+-----------+----------------+-----------+----
λ₁ (air)   | r²_1,1  ±σ    | r²_2,1  ±σ    |  Δ_1,12   | r²_3,1  ±σ    |  Δ_2,12   | ...
δ₁₂_p=1   | δ₁₂_1          |                |           |                |           |
λ₂ (air)   | r²_1,2  ±σ    | r²_2,2  ±σ    |  Δ_1,22   | r²_3,2  ±σ    |  Δ_2,22   | ...
           |                |                |           |                |           |
⟨Δ⟩ λ₁   | S₁_fit         | ± σ(S₁)       | [px²/fringe, from WLS]       |           |
⟨Δ⟩ λ₂   | S₂_fit         | ± σ(S₂)       | [px²/fringe, from WLS]       |           |
Ratio S₁/S₂| expected λ₁/λ₂ = 640.2248/638.2991 | actual S₁/S₂   |           |
```

Where:
- `Δ_p,1c` = r²_(p+1),λc − r²_p,λc  (successive ring difference for component c)
- `δ₁₂_p` = r²_p,λ₁ − r²_p,λ₂  (cross-component difference at ring p)
- `⟨Δ⟩` = mean successive difference ≈ S from WLS (should agree to < 1%)

For a **1-line analysis**, the table has a single component row (no δ₁₂ row)
and N rings.

### 2.4 Vaughan Equations 3.91–3.97 (pixel-unit translations)

These equations are explicitly computed and printed to the console during
Stage G (§12.4 of this spec).

**Eq 3.91 — Scale factor C (= Vaughan's Δ = our slope S):**
```
C = S = f²λ/(nd)   [px²/fringe]

Verification: C_computed = f_px² × λ_m / (n × d_m)
              C_fit      = S from WLS slope
              These should agree to < 0.1%
```
Note: Vaughan's Δ_Vaughan = 4·S because Vaughan uses diameter D = 2r, so
D² = 4r² and Δ_Vaughan = 4·C. This factor-of-4 is purely notational.

**Eq 3.92 — Wavenumber separation of two spectral components (2-line only):**
```
Δσ₁₂ = δ₁₂ / C   [m⁻¹, where δ₁₂ = ⟨r²_λ₁ − r²_λ₂⟩ in px²]

From known wavelengths: Δσ₁₂_known = 1/λ₂ − 1/λ₁  [m⁻¹]
Computed: Δσ₁₂_meas = mean(δ₁₂_p) / S₁   [dimensionless px²/px² → ×1/λ for m⁻¹]

Cross-check: |Δσ₁₂_meas − Δσ₁₂_known| / Δσ₁₂_known should be < 1%
```
For 1-line mode, Eq 3.92 is not applicable (single component).

**Eq 3.93 — McNair fractional order approximation (informational):**
```
δn ≈ 2δ₁ / (Δ₁₂ + Δ₂₁)   (off-axis approximation; Vaughan Eq 3.93)

where δ₁ = r²_1,λ₁ − r²_1,λ₂ (cross-component at ring 1)
and   Δ₁₂ = r²_2,λ₁ − r²_1,λ₁,  Δ₂₁ = r²_2,λ₂ − r²_1,λ₂

This is the McNair (1926) approximation for ε₁−ε₂ from only two adjacent
rings, useful when the fringe centre is inaccessible. For our full-ring WLS
fit this is superseded by the joint fit result, but it is computed as a
cross-check. Agreement to within 0.05 fringe is expected.

δn_McNair ≈ ε₁ − ε₂ (should match TwoLineResult.delta_eps to < 0.05)
```

**Eq 3.94 — Basic FP constructive interference condition at centre:**
```
2d = (m₀_int + ε) × λ   [single-line form]

where:
  m₀     = 2nd/λ   (central interference order, real number)
  m₀_int = round(m₀)   (nearest integer; = Vaughan's "p_a")
  ε      = m₀ − m₀_int  (fractional order ∈ [0,1))

The script computes:
  m₀       = 2 × d_m / λ_m           [unitless]
  m₀_int   = round(m₀)
  ε_check  = m₀ − m₀_int
  d_check  = (m₀_int + ε_fit) × λ / 2
  residual = |d_check − d_recovered| in µm (expect < 1 µm)
```

**Eq 3.95 — Two-line consistency check at centre:**
```
(m₀_int,₁ + ε₁) × λ₁ = (m₀_int,₂ + ε₂) × λ₂ = 2d   [2-line form]

Computed:
  lhs1 = (m₀_int,₁ + ε₁) × λ₁   [m]
  lhs2 = (m₀_int,₂ + ε₂) × λ₂   [m]
  |lhs1 − lhs2| / lhs1 should be < 1e-5  (excellent) or < 1e-4 (acceptable)
```
For 1-line mode, only the single-component Eq 3.94 is computed.

**Eq 3.96 — Integer order identification from ICOS prior (2-line, N_int step):**
```
Using d_prior = d_ICOS = 20.008 mm (air-spaced etalon):
  p'_1 = round(2 × d_prior / λ₁)   [integer estimate for λ₁ family]
  p'_2 = round(2 × d_prior / λ₂)   [integer estimate for λ₂ family]

Benoit's method: form (p'_1 + n + ε₁)λ₁ = (p'_2 + m + ε₂)λ₂
and find the integer offset n (= N_int in S13) that makes both sides agree.

N_int = round(2 × d_prior × (1/λ₁ − 1/λ₂))
     = round(2 × 20.008e-3 × (1/640.2248e-9 − 1/638.2991e-9))
     ≈ −189   (confirm at runtime; sign depends on λ ordering)

The script prints p'_1, p'_2, N_int, and the two sides of Eq 3.96 explicitly.
```

**Eq 3.97 — d recovered from identified integer orders (cross-check):**
```
Once N_int is identified, d can be recovered from each line independently:
  d_from_λ₁ = (p'_1 + ε₁) × λ₁ / 2
  d_from_λ₂ = (p'_2 + ε₂) × λ₂ / 2   [with p'_2 adjusted by N_int]
  d_excess   = |lever| × |N_int + ε₁ − ε₂|   [Benoit formula, S13 primary]

All three should agree to < 1 µm. Report all three to the console.
d_excess is the authoritative value stored in TwoLineResult.d_m.
```

---

## 3. Mode Selection — User Inputs at Startup

The script accepts command-line arguments OR presents an interactive prompt
if called with no arguments.

### 3.1 Command-line interface

```
Usage: python z01_validate_tolansky_analysis_2026-04-13.py [OPTIONS]

Options:
  --mode {1,2}         Number of spectral components (default: 2).
                       1 = single OI/filtered line; 2 = dual neon.
  --lam1 FLOAT         Wavelength 1 in nm (air). Default: 640.2248 nm (air)
                       (Ne primary; ignored in 1-line mode).
  --lam2 FLOAT         Wavelength 2 in nm (air). Default: 638.2991 nm (air)
                       (Ne secondary; 2-line mode only).
  --lam FLOAT          Single-line wavelength in nm (air). Default: 630.0304
                       nm (air) (OI; 1-line mode only).
  --d-prior FLOAT      Etalon gap prior in mm. Default: 20.008 mm (ICOS).
                       Used only to resolve FSR integer ambiguity.
  --has-header {yes,no,auto}
                       Whether the .bin calibration file contains a 1-row
                       P01 metadata header. Default: auto (detect from
                       file size; 260×276×2 = header present, else absent).
  --r-max FLOAT        Maximum annular radius in px. Default: 110 px.
  --n-subpixels INT    Sub-pixel averaging in model builds (default: 8).
  --help               Show this message and exit.
```

### 3.2 Interactive prompt (no arguments)

If called with no arguments, the script prints a startup banner and asks:

```
WindCube FPI — Tolansky Analysis Validation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Mode?
  [1]  Single-line (OI 630 nm or filtered neon — 1 spectral component)
  [2]  Two-line   (dual neon 640.2248 + 638.2991 nm — 2 components) [DEFAULT]
Enter 1 or 2 [2]: _
```

After mode selection, if wavelengths differ from the defaults the user is
prompted with the current default values shown; pressing Enter accepts them.

### 3.3 Wavelength override — always display (air) annotation

Wherever wavelengths are displayed (prompts, figure annotations, console
output, table cells), the unit designation **must always include (air)**:

```
  λ₁ = 640.2248 nm (air)   [Ne primary — Burns et al. 1950]
  λ₂ = 638.2991 nm (air)   [Ne secondary — Burns et al. 1950]
  λ  = 630.0304 nm (air)   [OI rest wavelength — NIST ASD]
```

Vacuum wavelengths are never displayed in the script output.

---

## 4. Script Structure Overview

```
Stage A  load_images()              — file dialogs; .bin (with/without header) or .npy
Stage B  extract_metadata()         — P01 ImageMetadata if header present, else None
Stage C  figure_image_pair()        — Figure 1: images + metadata tables + click-to-seed
Stage D  figure_roi_inspection()    — Figure 2: ROI images + ADU histograms (2×2)
Stage E  refine_centre()            — No figure: coarse grid → Nelder-Mead centre
Stage F1 run_s12_reduction()        — No figure: S12 M03 reduction with refined centre
Stage F2 figure_dark_comparison()   — Figure 3: dark subtraction diagnostic
Stage F3 figure_reduction_peaks()   — Figure 4: r² profile + peak overlay + table
Stage G  run_tolansky()             — No figure: dispatch to 1-line or 2-line analyser
Stage H  figure_tolansky()          — Figure 5: Rectangular array + r² graph + results
```

The `main()` function calls these stages in order. Each stage is a standalone
importable function. Results are passed forward as dicts or dataclasses.

---

## 5. Stage A — `load_images()`

### 5.1 Supported file types

| Extension | Header | Load method |
|-----------|--------|-------------|
| `.bin` with P01 header | 260 rows × 276 cols × 2 bytes big-endian uint16 | `ingest_real_image()` from S19/P01 |
| `.bin` without header | H × W × 2 bytes big-endian uint16 (default 256×256) | `load_headerless_bin()` (new helper) |
| `.npy` | None | `np.load()` |

Auto-detection (when `--has-header auto`): if the file is exactly
260 × 276 × 2 = 143,520 bytes, assume header present. Otherwise assume
absent and use the shape from `--r-max` context (default 256×256).

### 5.2 Function signature

```python
def load_images(
    has_header: str = 'auto',
    image_shape: tuple[int, int] = (256, 256),
) -> dict:
    """
    Returns
    -------
    dict:
        'cal_image'  : np.ndarray float64, pixel data only (no header row)
        'dark_image' : np.ndarray float64, same shape
        'cal_path'   : pathlib.Path
        'dark_path'  : pathlib.Path
        'cal_type'   : str — 'real' | 'headerless' | 'synthetic'
        'dark_type'  : str — same options
        'cal_raw'    : np.ndarray uint16 (full raw bytes) or None for .npy
        'dark_raw'   : np.ndarray uint16 or None
        'cal_meta'   : ImageMetadata | None  (parsed here if has_header)
        'dark_meta'  : ImageMetadata | None
    """
```

### 5.3 `load_headerless_bin()` helper

```python
def load_headerless_bin(path: pathlib.Path,
                        shape: tuple[int, int]) -> np.ndarray:
    """
    Read a headerless .bin file as 2D float64.
    Raises ValueError if file size does not match shape[0]*shape[1]*2 bytes.
    """
    data = np.frombuffer(path.read_bytes(), dtype='>u2').astype(np.float64)
    expected = shape[0] * shape[1]
    if data.size != expected:
        raise ValueError(
            f"Expected {expected} pixels ({shape[0]}×{shape[1]}) "
            f"but file contains {data.size} pixels: {path.name}"
        )
    return data.reshape(shape)
```

### 5.4 Validation

- `cal_image.shape == dark_image.shape` — raise `ValueError` if not.
- Neither image may be all zeros — warn to console if `np.all(img == 0)`.

---

## 6. Stage B — `extract_metadata()`

```python
def extract_metadata(load_result: dict) -> dict:
    """
    Returns cal_meta and dark_meta from load_result (already parsed
    in Stage A for .bin files with headers).
    Always returns None for headerless or .npy files.
    """
    return {
        'cal_meta':  load_result.get('cal_meta'),
        'dark_meta': load_result.get('dark_meta'),
    }
```

This stage is now a thin pass-through because `ingest_real_image()` already
parses metadata in Stage A. It is retained as a named stage for clarity.

---

## 7. Stage C — `figure_image_pair()` (Figure 1)

### 7.1 Function signature

```python
def figure_image_pair(
    load_result: dict,
    meta_result: dict,
    mode: int,
    lam_str: str,
) -> tuple[float, float]:
    """
    Figure 1: side-by-side raw images with metadata tables.
    Wires click-to-seed on the cal image axis.

    Parameters
    ----------
    mode     : 1 or 2 (used for suptitle annotation)
    lam_str  : human-readable wavelength string, e.g.
               "640.2248 nm (air) + 638.2991 nm (air)" or
               "630.0304 nm (air)"

    Returns
    -------
    (cx_seed, cy_seed) in image pixel coordinates.
    Defaults to geometric centre if no click registered.
    """
```

### 7.2 Layout

```
┌──────────────────────────────────────────────────────────────────┐
│  Figure 1 — Image Inspection                                     │
├──────────────────┬───────────────────────────────────────────────┤
│  ax_cal          │  ax_dark                                      │
│  CAL  (gray)     │  DARK (gray)                                  │
│  vmin=0 vmax=16383│ vmin=0 vmax=16383                            │
│  colorbar        │  colorbar                                     │
├──────────────────┼───────────────────────────────────────────────┤
│  ax_meta_cal     │  ax_meta_dark                                 │
│  metadata table  │  metadata table                               │
│  (asdict) or     │  (or "No embedded metadata")                  │
│  "No metadata"   │                                               │
└──────────────────┴───────────────────────────────────────────────┘
```

Use `plt.subplots(2, 2, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 2]})`.

### 7.3 Click-to-seed handler

Wire `fig.canvas.mpl_connect('button_press_event', on_click)` on the cal
image axis only. When clicked:
- Place a red `'+'` marker at click position.
- Update subplot title:
  `f"CAL — click to set fringe centre  [current: ({cx:.1f}, {cy:.1f}) px]"`
- Record `(cx_seed, cy_seed)`. Final values returned when window closes.

### 7.4 Suptitle

```python
suptitle = (
    f"Figure 1 — Image Inspection  |  "
    f"Mode: {mode}-Line  |  λ: {lam_str}  |  "
    f"{load_result['cal_path'].parent}  |  [close to continue]"
)
```

---

## 8. Stage D — `figure_roi_inspection()` (Figure 2)

Identical to Z01 v0.2 §6. Layout 2×2 (cal ROI, dark ROI, cal ADU histogram,
dark ADU histogram). gray cmap, `vmin=0 vmax=16383`. Returns None.

ROI is a square centered on `(cx_seed, cy_seed)` with side length
`2 × r_max + 20` px (or the image extent, whichever is smaller).

---

## 9. Stage E — `refine_centre()` — Centre Refinement

This stage documents the two-step centre-finding algorithm implemented inside
M03 (`reduce_calibration_frame`). Z01 calls M03, which runs these steps
internally. Z01 reports the intermediate results explicitly to the console.

**Why two steps?** The fringe centre must be known to sub-pixel accuracy for
an accurate annular profile. The user's visual seed from Figure 1 has typical
accuracy ±5–10 px. The coarse grid narrows this to ±1–2 px; Nelder-Mead
then achieves < 0.1 px accuracy.

### 9.1 Step 1 — Coarse grid azimuthal variance minimisation

For each candidate centre (cx_cand, cy_cand) on a grid of ±15 px around the
seed in 1 px steps, compute the radial profile `I(r)` and measure the
**azimuthal variance** at each radius: `Var_az(r) = mean_az[(I(θ,r) − Ī(r))²]`.
The sum of azimuthal variances is minimised when the centre is correct (the
rings are most perfectly circular).

```python
# Pseudo-code for coarse grid
best_score = np.inf
for cx_cand in np.arange(cx_seed - 15, cx_seed + 16, 1.0):
    for cy_cand in np.arange(cy_seed - 15, cy_seed + 16, 1.0):
        profile = azimuthal_profile(image, cx_cand, cy_cand, r_max)
        variance = compute_azimuthal_variance(image, cx_cand, cy_cand)
        if variance < best_score:
            best_score = variance
            cx_coarse, cy_coarse = cx_cand, cy_cand
```

Console output during coarse search:

```
  Stage E — Centre refinement:
    Seed:        cx = {cx_seed:.1f},  cy = {cy_seed:.1f}  px  (user visual estimate)
    Coarse grid: ±15 px, 1 px step  →  31×31 = 961 evaluations
    Coarse best: cx = {cx_coarse:.1f},  cy = {cy_coarse:.1f}  px
    Coarse improvement: Δcx = {cx_coarse-cx_seed:+.1f},  Δcy = {cy_coarse-cy_seed:+.1f}  px
```

### 9.2 Step 2 — Fine Nelder-Mead sub-pixel optimisation

Starting from `(cx_coarse, cy_coarse)`, run `scipy.optimize.minimize` with
method `'Nelder-Mead'` to minimise the same azimuthal variance objective,
now with continuous (non-integer) centre coordinates.

```python
from scipy.optimize import minimize

result_nm = minimize(
    fun=lambda p: azimuthal_variance_total(image, p[0], p[1], r_max),
    x0=[cx_coarse, cy_coarse],
    method='Nelder-Mead',
    options=dict(xatol=1e-3, fatol=1e-6, maxiter=2000),
)
cx_final, cy_final = result_nm.x
sigma_cx, sigma_cy = 0.05, 0.05   # default sub-pixel uncertainty
```

Console output after Nelder-Mead:

```
    Nelder-Mead: cx = {cx_final:.4f},  cy = {cy_final:.4f}  px  (sub-pixel)
    Sub-pixel shift from coarse: Δcx = {cx_final-cx_coarse:+.4f},  Δcy = {cy_final-cy_coarse:+.4f}  px
    Nelder-Mead converged: {result_nm.success}  (nfev={result_nm.nfev})
    Final centre (used in M03): cx = {cx_final:.4f} ± {sigma_cx:.4f},  cy = {cy_final:.4f} ± {sigma_cy:.4f}  px
```

### 9.3 Passing the refined centre to M03

The refined `(cx_final, cy_final)` is passed as `cx_human` / `cy_human` to
`reduce_calibration_frame()` in Stage F1. M03 uses this as its starting
point (or repeats its own internal refinement; the two should converge to the
same answer).

**Note:** if M03's internal centre refinement returns a value that differs
from `(cx_final, cy_final)` by more than 0.5 px, print a warning:
`"WARNING: M03 centre ({fp.cx:.3f}, {fp.cy:.3f}) differs from Stage E result by > 0.5 px."`

---

## 10. Stage F1 — `run_s12_reduction()`

```python
def run_s12_reduction(
    cal_image:  np.ndarray,
    dark_image: np.ndarray,
    cx_seed:    float,
    cy_seed:    float,
    r_max_px:   float = 110.0,
) -> tuple['FringeProfile', np.ndarray, dict]:
    """
    Run M03 reduce_calibration_frame and produce a diagnostic dark-sub image.

    Returns
    -------
    fp              : FringeProfile from M03
    s12_dark_sub    : dark-subtracted cal image (for Figure 3 diagnostic only;
                      NOT re-fed into any analysis — M03 subtracted internally)
    roi             : dict with roi_x0, roi_x1, roi_y0, roi_y1 (px)
    """
```

Dark subtraction is performed **once**, inside `reduce_calibration_frame()`.
The second call to `subtract_dark()` here is for diagnostic display only
(Figure 3) and must never be passed back to M03 or Tolansky.

---

## 11. Stage F2 — `figure_dark_comparison()` (Figure 3)

Identical to Z01 v0.2 §8. Three-panel layout:
- Panel L: cal ROI (raw)
- Panel C: naive dark subtraction `clip(cal−dark, 0)`
- Panel R: S12 dark-subtracted image (M03 pipeline output)

Suptitle must show "IDENTICAL (max |diff| = 0)" or "MISMATCH" in large bold
font, plus diff statistics. The IDENTICAL check validates that the dark
routing is correct (no double subtraction).

---

## 12. Stage F3 — `figure_reduction_peaks()` (Figure 4)

### 12.1 Layout

```
┌────────────────────────────────────────────────────────────────────┐
│  Figure 4 — Annular Reduction and Peak Identification              │
├──────────────────────────────────┬─────────────────────────────────┤
│  ax_profile                      │  ax_table                       │
│  Radial r² profile + peaks       │  Peak table (per §12.3)         │
│  (~65% width)                    │  (~35% width)                   │
└──────────────────────────────────┴─────────────────────────────────┘
```

### 12.2 r² profile plot

- X-axis: `r² (px²)`, from 0 to `r_max²`
- Y-axis: `Intensity (ADU)` (dark-subtracted)
- Data: solid thin `'#888888'` line for the full radial profile
- Peak markers: vertical dashed lines at each `PeakFit.r_fit_px²`

Peak colours by mode:
- **2-line:** family λ₁ peaks in `'#CC4400'` (orange-red), family λ₂ peaks in
  `'steelblue'`. Marker: `'▲'` at the peak amplitude.
- **1-line:** all peaks in `'steelblue'`. Marker: `'▲'`.

Each peak is labelled with its fringe index p (1 = innermost).

### 12.3 Peak table (in the right axis, matplotlib Table)

**2-line mode columns:**
`p`, `λ (air)`, `r (px)`, `r² (px²)`, `σ(r²) (px²)`, `2σ(r²) (px²)`, `Amp (ADU)`, `Fit OK`

**1-line mode columns** (no Family/λ column):
`p`, `r (px)`, `r² (px²)`, `σ(r²) (px²)`, `2σ(r²) (px²)`, `Amp (ADU)`, `Fit OK`

Include one summary row at the bottom: `⟨Δr²⟩ = {mean_delta_rsq:.1f} ± {std_delta_rsq:.1f} px²  (CV = {cv:.1f}%)`.

### 12.4 Suptitle

```python
suptitle = (
    f"Figure 4 — Annular Reduction  |  "
    f"{'2-Line: λ₁=640.2248 nm (air) + λ₂=638.2991 nm (air)' if mode==2 else 'λ=630.0304 nm (air)'}  |  "
    f"Centre: ({fp.cx:.3f}, {fp.cy:.3f}) px  |  "
    f"r_max = {r_max:.1f} px  |  "
    f"Peaks: {n_ok}/{n_total}  |  [close to continue]"
)
```

Expected peak counts: 2-line: 20 (10+10); 1-line: 6–7 (do not flag < 20 as error).

---

## 13. Stage G — `run_tolansky()` — Tolansky Dispatch

```python
def run_tolansky(
    fp:         'FringeProfile',
    mode:       int,
    lam1_nm:    float,          # λ₁ (air) for 2-line, or single λ (air) for 1-line
    lam2_nm:    float | None,   # λ₂ (air) for 2-line; None for 1-line
    d_prior_mm: float,          # ICOS gap for N_int resolution
    d_fixed_mm: float | None,   # fixed d (1-line mode); None for 2-line
    f_fixed_mm: float | None,   # fixed f (1-line mode); None for 2-line
    pixel_pitch_m: float,
) -> 'TwoLineResult | SingleLineResult':
    """
    Dispatch to TolanskyPipeline (2-line) or SingleLineTolansky (1-line).

    2-Line: returns TwoLineResult  (recovers d, f, α, ε₁, ε₂)
    1-Line: returns SingleLineResult  (recovers ε only; d and f fixed)
    """
```

### 13.1 2-Line dispatch

```python
pipeline = TolanskyPipeline(
    profile          = fp,
    d_prior_m        = d_prior_mm * 1e-3,
    lam1_nm          = lam1_nm,    # e.g. 640.2248 (air)
    lam2_nm          = lam2_nm,    # e.g. 638.2991 (air)
    amplitude_split_fraction = 0.7,
    n                = 1.0,
    pixel_pitch_m    = pixel_pitch_m,
)
result = pipeline.run()
```

### 13.2 1-Line dispatch

```python
analyser = SingleLineTolansky(
    fringe_profile = fp,
    lam_rest_nm    = lam1_nm,        # e.g. 630.0304 (air)
    d_prior_m      = d_fixed_mm * 1e-3,
    f_prior_m      = f_fixed_mm * 1e-3,
    pixel_pitch_m  = pixel_pitch_m,
    d_icos_m       = d_prior_mm * 1e-3,
)
result = analyser.run()
```

---

## 14. Stage H — `figure_tolansky()` (Figure 5)

This is the primary scientific output figure. It has three panels.

### 14.1 Layout

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Figure 5 — Tolansky Analysis                                            │
├──────────────────────────────────────────────────────────────────────────┤
│  ax_scatter  (~50% height)                                               │
│  Melissinos-style r² scatter plot (§14.2)                                │
├───────────────────────────┬──────────────────────────────────────────────┤
│  ax_table_array  (~25%)   │  ax_table_results  (~25%)                   │
│  Vaughan rectangular array│  Parameter summary table                    │
│  (§14.3)                  │  (§14.4)                                    │
└───────────────────────────┴──────────────────────────────────────────────┘
```

```python
fig = plt.figure(figsize=(16, 14))
gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1],
                       hspace=0.35, wspace=0.3)
ax_scatter   = fig.add_subplot(gs[0, :])     # full width
ax_table_arr = fig.add_subplot(gs[1:, 0])    # lower left
ax_table_res = fig.add_subplot(gs[1:, 1])    # lower right
```

### 14.2 Melissinos-style r² scatter plot (ax_scatter)

This plot is modelled after Melissinos (1966) Figure 7.28 (p. 325).

- **Y-axis:** fringe index `p` (integer, 1 = innermost ring) — no error bars
- **X-axis:** `r² (px²)` — with **2σ horizontal error bars**
  - `σ(r²) = 2r · σ_r` (standard propagation, S13 §5 eq)
  - **Horizontal error bar half-width = 2σ(r²) = 4r · σ_r**
  - Display as `xerr=2*sigma_r_sq` in `ax.errorbar()`
  - Error bars may be smaller than the marker; always draw them

**For 2-line mode:** two data series, one per wavelength family.
Plot side by side in a single axis. Use `color='#CC4400'` for λ₁ and
`color='steelblue'` for λ₂. Include a legend:
`"λ₁ = 640.2248 nm (air)"` and `"λ₂ = 638.2991 nm (air)"`.

**For 1-line mode:** a single data series in `steelblue`. Label:
`"λ = 630.0304 nm (air)   [OI rest wavelength]"`.

Plot the WLS best-fit line for each family:
```python
p_fit = np.linspace(0.5, p_max + 0.5, 200)
r_sq_fit = S * (p_fit - 1 + epsilon)
ax.plot(r_sq_fit, p_fit, '-', color=color, lw=1.5, alpha=0.7)
```

**Axis limits:** X from 0 to `r_max² × 1.05`; Y from 0.3 to `p_max + 0.7`.
The Y-axis has integer tick marks only: `ax.yaxis.set_major_locator(MaxNLocator(integer=True))`.
**The Y-axis is NOT reversed** (p=1 at bottom, higher p at top — physics
convention with innermost rings having lowest index).

Annotations on the plot (text box, lower right inside axes):
```
For 2-line:
  S₁ = {S1:.2f} ± {σS1:.2f} px²/fringe   [λ₁ = 640.2248 nm (air)]
  S₂ = {S2:.2f} ± {σS2:.2f} px²/fringe   [λ₂ = 638.2991 nm (air)]
  S₁/S₂ = {S1/S2:.6f}  (expect λ₁/λ₂ = {lam1/lam2:.6f})
  ε₁ = {eps1:.5f} ± {σε1:.5f}
  ε₂ = {eps2:.5f} ± {σε2:.5f}
  χ²/ν = {chi2_dof:.3f}

For 1-line:
  S  = {S:.2f} ± {σS:.2f} px²/fringe   [λ = 630.0304 nm (air)]
  ε  = {eps:.6f} ± {σε:.6f}
  R² = {r2_fit:.6f}
  χ²/ν = {chi2_dof:.3f}
```

Add a residuals inset at `ax_scatter.inset_axes([0.75, 0.05, 0.22, 0.35])`:
- Normalised residuals `(r²_obs − r²_fit) / σ(r²)` vs fringe index p
- Stem plot; dashed horizontal zero line
- Y-label: `"Residual (σ)"`; expect scatter within ±3

**Axis labels and title:**
```python
ax_scatter.set_xlabel("r² (px²)", fontsize=11)
ax_scatter.set_ylabel("Fringe index  p", fontsize=11)
ax_scatter.set_title(
    "Tolansky r² Analysis   [Melissinos 1966, Fig 7.28 style]",
    fontsize=12, fontweight='bold'
)
```

### 14.3 Vaughan Rectangular Array Table (ax_table_arr)

Display the rectangular array (§2.3) as a matplotlib Table.

**For 2-line mode** (N_common = min(N₁, N₂) rings shown):

Columns: `Component`, `p=1`, `p=2`, `Δ₁₂`, `p=3`, `Δ₂₃`, ..., `⟨Δ⟩`, `σ(Δ)`

Rows:
1. `λ₁=640.22 nm (air)` — filled with `f"{r_sq:.1f}"` for each ring and
   `f"{delta_rsq:.1f}"` for each successive difference
2. `δ₁₂_p` — cross-component differences at each ring: `f"{r_sq_lam1 − r_sq_lam2:.1f}"` px²
3. `λ₂=638.30 nm (air)` — same format
4. (blank separator row)
5. `⟨Δ⟩ λ₁` (mean successive diff) | `S₁_fit ± σ` px²/fringe | (spanning columns)
6. `⟨Δ⟩ λ₂` | `S₂_fit ± σ` | (spanning columns)
7. `S₁/S₂` | `{actual ratio:.6f}` | `expect {λ₁/λ₂:.6f}` | `diff {(S1/S2 − λ1/λ2)*1e4:.2f}×10⁻⁴`

**For 1-line mode** (no δ₁₂ row; only λ component):

Columns: `Component`, `p=1`, `p=2`, `Δ₁₂`, `p=3`, `Δ₂₃`, ..., `⟨Δ⟩ ± σ(Δ)`, `CV (%)`
Row: `λ=630.03 nm (air)` filled with ring r² values and successive differences.
Summary row: `⟨Δ⟩ = {mean_Δ:.2f} ± {std_Δ:.2f} px²/fringe  |  S_fit = {S:.2f} ± {σS:.2f}  |  CV = {cv:.1f}%`

All values in px². Title above table:

```python
ax_table_arr.set_title(
    "Vaughan (1989) §3.5.2 Rectangular Array  [units: px²]",
    fontsize=9, fontweight='bold', pad=4
)
ax_table_arr.axis('off')
```

### 14.4 Parameter Summary Table (ax_table_res)

**2-line mode:**

| Parameter | Symbol | Value | σ | 2σ | Units |
|-----------|--------|-------|---|----|-------|
| Etalon gap | d | `result.d_m*1e3` | `result.sigma_d_m*1e3` | `result.two_sigma_d_m*1e3` | mm |
| Focal length | f | `result.f_px*pitch_mm` | `result.sigma_f_px*pitch_mm` | `result.two_sigma_f_px*pitch_mm` | mm |
| Plate scale | α | `result.alpha_rad_px` | `result.sigma_alpha` | `result.two_sigma_alpha` | rad/px |
| Fract. order λ₁ | ε₁ | `result.eps1` | `result.sigma_eps1` | `2*sigma_eps1` | — |
| Fract. order λ₂ | ε₂ | `result.eps2` | `result.sigma_eps2` | `2*sigma_eps2` | — |
| N_int (Benoit) | N | `result.N_int` | — | — | — |
| Slope λ₁ | S₁ | `result.S1` | `result.sigma_S1` | `2*sigma_S1` | px²/fr |
| Slope ratio | S₁/S₂ | `result.S1/result.S2` | — | — | [λ₁/λ₂] |
| Reduced χ² | χ²/ν | `result.chi2_dof` | — | — | — |

**1-line mode:**

| Parameter | Symbol | Value | σ | 2σ | Units |
|-----------|--------|-------|---|----|-------|
| Fract. order (rest) | ε₀ | `result.epsilon` | `result.sigma_eps` | `result.two_sigma_eps` | — |
| Rest λ | λ_c | `result.lam_c_nm` | `result.sigma_lam_c_nm` | `2*sigma_lam_c_nm` | nm (air) |
| Reference velocity | v_ref | `result.v_rel_ms` | `result.sigma_v_ms` | `2*sigma_v_ms` | m/s |
| Zero-v check | — | PASS / WARN | — | — | — |
| Integer order | N | `result.N_int` | — | — | — |
| d (fixed prior) | d | `d_fixed_mm` | — | — | mm |
| f (fixed prior) | f | `f_fixed_mm` | — | — | mm |
| Slope | S | `result.S` | `result.sigma_S` | `2*sigma_S` | px²/fr |
| Reduced χ² | χ²/ν | `result.chi2_dof` | — | — | — |

Display the 1-line table with **ε₀ in bold** (it is the primary calibration output).

### 14.5 Suptitle

**2-line:**
```python
suptitle = (
    f"Figure 5 — Tolansky Analysis  |  2-Line Neon  |  "
    f"λ₁=640.2248 nm (air), λ₂=638.2991 nm (air)  |  "
    f"d={result.d_m*1e3:.4f}±{result.two_sigma_d_m*1e3:.4f} mm (2σ)  |  "
    f"f={f_mm:.3f}±{two_sigma_f_mm:.3f} mm (2σ)  |  "
    f"α={result.alpha_rad_px:.4e} rad/px  |  [close to exit]"
)
```

**1-line:**
```python
v_check = 'PASS' if abs(result.v_rel_ms) < 3*result.sigma_v_ms else 'WARN'
suptitle = (
    f"Figure 5 — Tolansky Analysis  |  Single-Line  |  "
    f"λ=630.0304 nm (air)  |  "
    f"ε₀={result.epsilon:.6f}±{result.sigma_eps:.6f}  |  "
    f"v_ref={result.v_rel_ms:.1f}±{result.sigma_v_ms:.1f} m/s  [{v_check}]  |  "
    f"[close to exit]"
)
```

---

## 15. Vaughan Equations 3.91–3.97 Console Block

After Stage H (figure displayed), print the following to the console. This
block makes the numerical content of Vaughan's equations explicit and machine-
readable for inclusion in field notebooks.

```python
def print_vaughan_equations(result, mode, lam1_nm, lam2_nm,
                             d_prior_mm, pitch_m, pitch_mm):
    """Print explicit Vaughan §3.5.2 equation calculations to stdout."""
    print()
    print("=" * 72)
    print("  Vaughan (1989) §3.5.2  —  Equations 3.91–3.97 (pixel units)")
    print("=" * 72)

    if mode == 2:
        S1 = result.S1
        eps1, eps2 = result.eps1, result.eps2
        d_m = result.d_m
        f_px = result.f_px
        lam1_m = lam1_nm * 1e-9
        lam2_m = lam2_nm * 1e-9

        print(f"\nEq 3.91 — Scale factor C (= Tolansky slope S, using radius² in px²):")
        print(f"  C = f²λ/(nd)  [px²/fringe]")
        C_computed = f_px**2 * lam1_m / (1.0 * d_m)
        print(f"  C_computed = ({f_px:.2f} px)² × {lam1_nm:.4f}e-9 m / (1.0 × {d_m*1e3:.4f}e-3 m)")
        print(f"             = {C_computed:.3f} px²/fringe")
        print(f"  S₁_fit     = {S1:.3f} ± {result.sigma_S1:.3f} px²/fringe")
        print(f"  Agreement:   |C_computed − S₁_fit| / S₁_fit = {abs(C_computed-S1)/S1*100:.3f}%")
        print(f"  Note: Vaughan's Δ_Vaughan = 4 × C  (he uses diameter D=2r, so D²=4r²)")

        print(f"\nEq 3.92 — Wavenumber separation of two spectral components:")
        S2 = result.S2
        # Mean cross-component difference at each ring
        delta_r2_cross = np.mean(result.r1_sq - result.r2_sq[:len(result.r1_sq)])
        sigma_cross = 1/lam2_m - 1/lam1_m   # [m⁻¹], known value
        sigma_meas = delta_r2_cross / S1 / lam1_m   # [m⁻¹]
        print(f"  Δσ₁₂ = δ₁₂ / C  [m⁻¹]")
        print(f"  Known:     Δσ₁₂ = 1/λ₂ − 1/λ₁ = {sigma_cross:.3e} m⁻¹ = {sigma_cross*0.01:.4f} cm⁻¹")
        print(f"  Measured:  ⟨δ₁₂⟩ = ⟨r²_λ₁ − r²_λ₂⟩ = {delta_r2_cross:.2f} px²")
        print(f"  Agreement: |Δσ_meas / Δσ_known − 1| = {abs(sigma_meas/sigma_cross - 1)*100:.2f}%")

        print(f"\nEq 3.93 — McNair (1926) fractional order approximation (cross-check):")
        # Use first two rings
        delta1 = result.r1_sq[0] - result.r2_sq[0]   # cross-component at p=1
        delta_A = result.r1_sq[1] - result.r1_sq[0]   # Δ for λ₁ between rings 1 and 2
        delta_B = result.r2_sq[1] - result.r2_sq[0]   # Δ for λ₂ between rings 1 and 2
        delta_n_McNair = 2 * delta1 / (delta_A + delta_B)
        print(f"  δn_McNair = 2δ₁/(Δ_12,λ₁ + Δ_12,λ₂)")
        print(f"            = 2×{delta1:.2f} / ({delta_A:.2f} + {delta_B:.2f})")
        print(f"            = {delta_n_McNair:.5f}")
        print(f"  ε₁ − ε₂ from joint fit = {eps1 - eps2:.5f}")
        print(f"  |McNair − joint fit| = {abs(delta_n_McNair - (eps1-eps2)):.5f} fringe"
              f"  (expect < 0.05 for well-separated rings)")

        print(f"\nEq 3.94 — Basic FP condition at centre (λ₁ = {lam1_nm:.4f} nm (air)):")
        m0_lam1 = 2.0 * d_m / lam1_m
        m0_int_lam1 = round(m0_lam1)
        eps1_check = m0_lam1 - m0_int_lam1
        d_check_lam1 = (m0_int_lam1 + eps1) * lam1_m / 2
        print(f"  2d = (m₀_int + ε) × λ₁")
        print(f"  m₀     = 2d/λ₁ = 2×{d_m*1e3:.5f}e-3 / {lam1_nm:.4f}e-9 = {m0_lam1:.5f}")
        print(f"  m₀_int = {m0_int_lam1}   (Vaughan's 'p_a' — integer order at centre)")
        print(f"  ε₁ from fit  = {eps1:.6f}")
        print(f"  ε from m₀    = {eps1_check:.6f}  (= m₀ − m₀_int)")
        print(f"  d_check = (m₀_int + ε₁_fit) × λ₁ / 2 = {d_check_lam1*1e3:.6f} mm")
        print(f"  |d_check − d_recovered| = {abs(d_check_lam1 - d_m)*1e6:.4f} µm  (expect < 1 µm)")

        print(f"\nEq 3.95 — Two-line consistency check at centre:")
        m0_lam2 = 2.0 * d_m / lam2_m
        m0_int_lam2 = round(m0_lam2)
        lhs1 = (m0_int_lam1 + eps1) * lam1_m
        lhs2 = (m0_int_lam2 + eps2) * lam2_m
        print(f"  (m₀_int,₁ + ε₁)×λ₁ = ({m0_int_lam1} + {eps1:.6f}) × {lam1_nm:.4f}e-9 m = {lhs1*1e3:.8f} mm")
        print(f"  (m₀_int,₂ + ε₂)×λ₂ = ({m0_int_lam2} + {eps2:.6f}) × {lam2_nm:.4f}e-9 m = {lhs2*1e3:.8f} mm")
        print(f"  Both should equal 2d = {d_m*2*1e3:.8f} mm")
        print(f"  |lhs1 − lhs2| / lhs1 = {abs(lhs1-lhs2)/lhs1:.2e}  (expect < 1e-4)")

        print(f"\nEq 3.96 — Integer order identification (N_int, Benoit prior):")
        p_prime_1 = round(2 * d_prior_mm * 1e-3 / lam1_m)
        p_prime_2 = round(2 * d_prior_mm * 1e-3 / lam2_m)
        N_int = result.N_int
        print(f"  d_prior (ICOS) = {d_prior_mm:.3f} mm")
        print(f"  p'_1 = round(2×d_prior/λ₁) = round({2*d_prior_mm*1e-3/lam1_m:.3f}) = {p_prime_1}")
        print(f"  p'_2 = round(2×d_prior/λ₂) = round({2*d_prior_mm*1e-3/lam2_m:.3f}) = {p_prime_2}")
        print(f"  N_int = round(2×d_prior×(1/λ₁ − 1/λ₂)) = {N_int}  (= Benoit N)")
        rhs1 = (p_prime_1 + eps1) * lam1_m
        rhs2 = (p_prime_2 + eps2) * lam2_m
        print(f"  (p'_1 + ε₁)×λ₁ = ({p_prime_1} + {eps1:.5f})×{lam1_nm:.4f}e-9 = {rhs1*1e3:.5f} mm")
        print(f"  (p'_2 + ε₂)×λ₂ = ({p_prime_2} + {eps2:.5f})×{lam2_nm:.4f}e-9 = {rhs2*1e3:.5f} mm")

        print(f"\nEq 3.97 — d recovered from identified integer orders (self-consistency):")
        d_from_lam1 = (p_prime_1 + eps1) * lam1_m / 2
        d_from_lam2 = (p_prime_2 + eps2) * lam2_m / 2
        lever = lam1_m * lam2_m / (2.0 * (lam2_m - lam1_m))   # negative
        d_excess = abs(lever * (N_int + eps1 - eps2))
        print(f"  d from λ₁:      (p'_1 + ε₁)×λ₁/2 = {d_from_lam1*1e3:.6f} mm")
        print(f"  d from λ₂:      (p'_2 + ε₂)×λ₂/2 = {d_from_lam2*1e3:.6f} mm")
        print(f"  d_excess (Benoit formula, authoritative) = {d_excess*1e3:.6f} mm")
        print(f"  Three-way agreement: max spread = {max(d_from_lam1,d_from_lam2,d_excess)*1e6 - min(d_from_lam1,d_from_lam2,d_excess)*1e6:.3f} µm")

    else:  # mode == 1
        S = result.S
        eps = result.epsilon
        d_m = d_fixed_mm * 1e-3
        f_mm = f_fixed_mm   # provided as fixed prior
        f_px = f_mm / pitch_mm
        lam_m = lam1_nm * 1e-9

        print(f"\nEq 3.91 — Scale factor C (1-line mode, d and f fixed):")
        C_computed = f_px**2 * lam_m / (1.0 * d_m)
        print(f"  C_computed = ({f_px:.2f} px)² × {lam1_nm:.4f}e-9 m / {d_m*1e3:.4f}e-3 m")
        print(f"             = {C_computed:.3f} px²/fringe")
        print(f"  S_fit      = {S:.3f} ± {result.sigma_S:.3f} px²/fringe")
        print(f"  Agreement: |C − S| / S = {abs(C_computed-S)/S*100:.3f}%")

        print(f"\nEqs 3.92–3.93: not applicable (single spectral component).")

        print(f"\nEq 3.94 — Basic FP condition at centre (λ = {lam1_nm:.4f} nm (air)):")
        m0 = 2.0 * d_m / lam_m
        m0_int = round(m0)
        eps_check = m0 - m0_int
        d_check = (m0_int + eps) * lam_m / 2
        print(f"  m₀     = 2d/λ = {m0:.5f}")
        print(f"  m₀_int = {m0_int}")
        print(f"  ε_fit  = {eps:.6f}  (fractional order at centre — rest-frame reference)")
        print(f"  ε_check= {eps_check:.6f}  (from m₀ directly)")
        print(f"  d_check = (m₀_int + ε_fit)×λ/2 = {d_check*1e3:.6f} mm")
        print(f"  d_prior = {d_fixed_mm:.3f} mm  (fixed)")
        print(f"  |d_check − d_prior| = {abs(d_check - d_m)*1e6:.3f} µm")

        print(f"\nEqs 3.95–3.97: require two wavelengths — not applicable in 1-line mode.")

    print("=" * 72)
```

---

## 16. `main()` Function

```python
def main():
    args = parse_args()    # argparse or interactive prompt
    mode = args.mode       # 1 or 2
    lam1_nm = args.lam1    # nm (air)
    lam2_nm = args.lam2    # nm (air); None if mode==1
    d_prior_mm = args.d_prior

    print("=" * 70)
    print(f"  WindCube FPI — Z01 Tolansky Validation  |  Mode: {mode}-Line")
    if mode == 2:
        print(f"  λ₁ = {lam1_nm} nm (air) [Ne primary]  |  "
              f"λ₂ = {lam2_nm} nm (air) [Ne secondary]")
    else:
        print(f"  λ  = {lam1_nm} nm (air) [OI rest wavelength]")
    print("=" * 70)

    # Stage A
    load = load_images(has_header=args.has_header)

    # Stage B
    meta = extract_metadata(load)

    # Stage C — Figure 1
    lam_str = (f"{lam1_nm} nm (air) + {lam2_nm} nm (air)" if mode==2
               else f"{lam1_nm} nm (air)")
    cx_seed, cy_seed = figure_image_pair(load, meta, mode, lam_str)

    # Stage D — Figure 2
    figure_roi_inspection(load['cal_image'], load['dark_image'],
                          cx_seed, cy_seed, r_max=args.r_max)

    # Stage E — Centre refinement (reported to console; passed to F1)
    cx_refined, cy_refined = refine_centre(
        load['cal_image'], cx_seed, cy_seed, r_max=args.r_max
    )

    # Stage F1 — S12 reduction
    fp, s12_dark_sub, roi = run_s12_reduction(
        load['cal_image'], load['dark_image'],
        cx_refined, cy_refined, r_max=args.r_max
    )

    # Stage F2 — Figure 3
    roi_slice = (slice(roi['roi_y0'], roi['roi_y1']),
                 slice(roi['roi_x0'], roi['roi_x1']))
    figure_dark_comparison(
        load['cal_image'][roi_slice],
        load['dark_image'][roi_slice],
        s12_dark_sub[roi_slice],
    )

    # Stage F3 — Figure 4
    figure_reduction_peaks(fp, roi, load['cal_path'], mode, lam1_nm, lam2_nm)

    # Stage G — Tolansky
    result = run_tolansky(
        fp, mode, lam1_nm, lam2_nm,
        d_prior_mm=d_prior_mm,
        d_fixed_mm=TOLANSKY_D_MM if mode==1 else None,
        f_fixed_mm=TOLANSKY_F_MM if mode==1 else None,
        pixel_pitch_m=CCD_PIXEL_PITCH_M,
    )

    # Stage H — Figure 5
    figure_tolansky(fp, result, mode, lam1_nm, lam2_nm,
                    load['cal_path'], d_prior_mm)

    # Equations 3.91–3.97 console block
    print_vaughan_equations(result, mode, lam1_nm, lam2_nm,
                            d_prior_mm, CCD_PIXEL_PITCH_M,
                            CCD_PIXEL_PITCH_M * 1e3)

    # Final summary
    print("\n" + "=" * 70)
    print("  Z01 SUMMARY:")
    if mode == 2:
        f_mm = result.f_px * CCD_PIXEL_PITCH_M * 1e3
        f_2sigma_mm = result.two_sigma_f_px * CCD_PIXEL_PITCH_M * 1e3
        print(f"    d   = {result.d_m*1e3:.4f} ± {result.two_sigma_d_m*1e3:.4f} mm (2σ)")
        print(f"    f   = {f_mm:.3f} ± {f_2sigma_mm:.3f} mm (2σ)")
        print(f"    α   = {result.alpha_rad_px:.4e} ± {result.two_sigma_alpha:.4e} rad/px (2σ)")
        print(f"    ε₁  = {result.eps1:.5f} ± {2*result.sigma_eps1:.5f}  [λ₁={lam1_nm} nm (air)] (2σ)")
        print(f"    ε₂  = {result.eps2:.5f} ± {2*result.sigma_eps2:.5f}  [λ₂={lam2_nm} nm (air)] (2σ)")
        print(f"    N_int = {result.N_int}  |  χ²/ν = {result.chi2_dof:.3f}")
    else:
        v_check = abs(result.v_rel_ms) < 3 * result.sigma_v_ms
        print(f"    ε₀     = {result.epsilon:.6f} ± {result.sigma_eps:.6f}  (2σ={result.two_sigma_eps:.6f})"
              f"  ← store as calibration constant")
        print(f"    λ_c    = {result.lam_c_nm:.5f} ± {result.two_sigma_lam_c_nm:.5f} nm (air) (2σ)")
        print(f"    v_ref  = {result.v_rel_ms:.1f} ± {result.two_sigma_v_ms:.1f} m/s (2σ)")
        print(f"    Zero-v check: {'PASS' if v_check else 'WARN — investigate'}")
        print(f"    N_int  = {result.N_int}  |  χ²/ν = {result.chi2_dof:.3f}")
        print(f"    d_prior (fixed) = {TOLANSKY_D_MM:.3f} mm")
        print(f"    f_prior (fixed) = {TOLANSKY_F_MM:.3f} mm")
    print("=" * 70)


if __name__ == "__main__":
    main()
```

---

## 17. Key Physical Parameters

All constants imported from `windcube.constants` (S03).

| Symbol | Value | Source | Notes |
|--------|-------|--------|-------|
| `NE_WAVELENGTH_1_NM` | 640.2248 nm **(air)** | Burns et al. 1950 | Primary neon cal line |
| `NE_WAVELENGTH_2_NM` | 638.2991 nm **(air)** | Burns et al. 1950 | Secondary neon cal line |
| `OI_WAVELENGTH_NM` | 630.0304 nm **(air)** | NIST ASD | OI science line |
| `ICOS_GAP_MM` | 20.008 mm | ICOS build report Dec 2023 | N_int disambiguation only |
| `TOLANSKY_D_MM` | 20.106 mm | Z01 2-line neon fit | Fixed prior for 1-line mode |
| `TOLANSKY_F_MM` | 199.12 mm | Z01 2-line neon fit | Fixed prior for 1-line mode |
| `CCD_PIXEL_PITCH_M` | 32e-6 m | CCD97, 2×2 binned | 2 × 16 µm native pitch |
| `R_MAX_PX` | 110 px | FlatSat/flight | Annular reduction outer limit |

**All displayed wavelengths must include `(air)` designation.**

---

## 18. Verification Tests (Manual)

No automated pytest tests — Z01 is fully interactive. Manual checks:

| ID | Stage | Figure | Criterion |
|----|-------|--------|-----------|
| V1 | A | — | Both images load; shape mismatch raises ValueError |
| V2 | C | Fig 1 | Image metadata shown (or "No embedded metadata"); click registers |
| V3 | D | Fig 2 | Cal ROI shows fringe rings; dark ROI is low, uniform |
| V4 | D | Fig 2 | No ADU saturation (max < 16383) in cal ROI histogram |
| V5 | E | Console | Stage E prints coarse seed, coarse best, Nelder-Mead result; shift < 5 px from seed |
| V6 | F2 | Fig 3 | Suptitle: "IDENTICAL (max \|diff\| = 0)" confirms dark routing |
| V7 | F3 | Fig 4 | 2-line: 20 peaks found (10+10); 1-line: 6–7 peaks. No false failures |
| V8 | F3 | Fig 4 | 2-line: two distinct colour families visible in peak overlay |
| V9 | H | Fig 5 | R² of Tolansky fit > 0.9999 for each line |
| V10 | H | Fig 5 | CV(Δr²) < 5% for each family |
| V11 | H | Fig 5 | Vaughan array: S₁/S₂ matches λ₁/λ₂ within 0.1% |
| V12 | H | Fig 5 | 2-line: d within 0.5 mm of ICOS prior; f within 5 mm of 200 mm |
| V13 | H | Fig 5 | 1-line: λ_c within 0.01 nm of 630.0304 nm (air); zero-v PASS |
| V14 | Console | — | Eq 3.97 three-way d agreement < 2 µm |
| V15 | Console | — | Eq 3.95 two-line consistency \|lhs1−lhs2\|/lhs1 < 1e-4 |
| V16 | Console | — | 1-line: ε₀ printed to 6 decimal places with "← store as calibration constant" |

---

## 19. Expected Output Values

### 2-Line Mode (real FlatSat neon calibration image)

| Quantity | Expected |
|----------|----------|
| Peaks found | 20 (10 per family) |
| Fringe centre | ~(107.7, 109.9) px |
| Coarse grid shift from seed | < 5 px |
| Nelder-Mead sub-pixel shift | < 0.5 px from coarse |
| d (Tolansky) | 20.106 ± 0.005 mm |
| f (Tolansky) | 199.12 ± 1.0 mm |
| α (Tolansky) | ~1.607×10⁻⁴ ± 0.01×10⁻⁴ rad/px |
| S₁/S₂ | 640.2248/638.2991 = 0.99699 ± 0.00001 |
| N_int | −189 |
| Eq 3.97 three-way d spread | < 1 µm |
| χ²/ν | 1.0–2.0 |

### 1-Line Mode (filtered Ne lamp, static source)

| Quantity | Expected |
|----------|----------|
| Peaks found | 6–7 |
| ε₀ | 0.3–0.7 (temperature-dependent) |
| λ_c | 630.000 ± 0.005 nm (air) |
| v_ref | −20 to +20 m/s (within noise; source is at rest) |
| Zero-v check | PASS |
| χ²/ν | 1.0–2.0 |

---

## 20. File Location in Repository

```
soc_sewell/
├── validation/
│   └── z01_validate_tolansky_analysis_2026-04-13.py
└── docs/specs/
    └── Z01_validate_tolansky_analysis_2026-04-13.md
```

**The old scripts Z01 and Z01a are superseded** but should be retained for
reference in `validation/archive/`:
```
soc_sewell/validation/archive/
    z01_validate_calibration_using_real_images_2026-04-09.py
    z01a_validate_OI630_filtered_neon_calibration_2026-04-12.py
```

---

## 21. Dependencies

```
numpy       >= 1.24
scipy       >= 1.10    # minimize (Nelder-Mead), least_squares
matplotlib  >= 3.7     # Table, inset_axes, MaxNLocator
tkinter                # file dialogs (standard library)
pathlib                # standard library
argparse               # standard library
```

Internal imports:
```python
from fpi.m03_annular_reduction_2026_04_05 import (
    reduce_calibration_frame,
    make_master_dark,
    subtract_dark,
    FringeProfile,
)
from fpi.tolansky_2026_04_05 import (
    TolanskyPipeline,
    TwoLineResult,
    SingleLineTolansky,
    SingleLineResult,
)
from fpi.p01_metadata_2026_04_06 import ingest_real_image, ImageMetadata
from windcube.constants import (
    NE_WAVELENGTH_1_NM,    # 640.2248 nm (air)
    NE_WAVELENGTH_2_NM,    # 638.2991 nm (air)
    OI_WAVELENGTH_NM,      # 630.0304 nm (air)
    CCD_PIXEL_PITCH_M,     # 32e-6 m
    ICOS_GAP_MM,           # 20.008 mm (disambiguation only)
    TOLANSKY_D_MM,         # 20.106 mm (fixed prior for 1-line)
    TOLANSKY_F_MM,         # 199.12 mm (fixed prior for 1-line)
    R_MAX_PX,              # 110 px
)
```

---

## 22. Instructions for Claude Code

1. Read this entire spec AND S12 (M03) AND S13 (Tolansky) AND S19 (P01)
   before writing any code.

2. Confirm M03 and Tolansky tests pass:
   ```bash
   pytest tests/test_m03_annular_reduction_*.py -v
   pytest tests/test_tolansky_*.py -v
   ```
   Stop and report any failures before proceeding.

3. Create:
   ```
   validation/z01_validate_tolansky_analysis_2026-04-13.py
   ```
   Do NOT modify the old Z01 or Z01a scripts.

4. Module docstring:
   ```python
   """
   Z01 — Validate Tolansky Analysis (1-Line and 2-Line Fringe Images)
   WindCube FPI Pipeline — NCAR / High Altitude Observatory (HAO)
   Spec: docs/specs/Z01_validate_tolansky_analysis_2026-04-13.md
   Supersedes: Z01 v0.2 (2026-04-12), Z01a v0.2 (2026-04-12)

   Usage:
     python z01_validate_tolansky_analysis_2026-04-13.py [--mode {1,2}] [OPTIONS]

   Modes:
     2 (default): dual neon 640.2248 nm (air) + 638.2991 nm (air)
     1:           single-line OI 630.0304 nm (air) or filtered neon

   Tool: Claude Code
   Last updated: 2026-04-13
   """
   ```

5. Implement in this order:
   - `parse_args()` — argparse + interactive fallback
   - `load_headerless_bin()` helper
   - `load_images()` — unified loader with auto-detection
   - `extract_metadata()` — thin pass-through
   - `figure_image_pair()` — Figure 1 with click-to-seed
   - `figure_roi_inspection()` — Figure 2 (copy from Z01 v0.2 with minor updates)
   - `refine_centre()` — coarse grid + Nelder-Mead with console reporting
   - `run_s12_reduction()` — Stage F1
   - `figure_dark_comparison()` — Figure 3 (copy from Z01 v0.2)
   - `figure_reduction_peaks()` — Figure 4 (mode-aware peak colouring and table)
   - `run_tolansky()` — dispatch wrapper
   - `figure_tolansky()` — Figure 5 (scatter + rectangular array + results table)
   - `print_vaughan_equations()` — §15 console block
   - `main()` — orchestration

6. **Wavelength annotation rule:** every wavelength string printed or drawn
   must include `(air)`. No exceptions.

7. **Dark subtraction critical rule:** `cal_image` must never be dark-subtracted
   before calling `reduce_calibration_frame()`. The second `subtract_dark()` in
   Stage F1 is for diagnostic Figure 3 only.

8. **1-line vs 2-line dispatch:** the Tolansky analysis call in Stage G must
   correctly dispatch based on `mode`. In 1-line mode, `TwoLineResult` fields
   (d, f, α, ε₁, ε₂) are not available — use `SingleLineResult` instead.
   Do not crash if the 2-line-specific fields are accessed in 1-line mode.

9. **Vaughan array table:** if the two ring families have different numbers of
   detected peaks, use only the common rings (min(N₁, N₂)) in the table.
   The WLS fits use all available rings; only the table display is truncated.

10. **Melissinos-style graph:** the x-axis is r² (px²), the y-axis is fringe
    index p. Horizontal error bars of ±2σ(r²). **Do not reverse the y-axis.**

11. **print_vaughan_equations():** this function uses fields from
    `TwoLineResult` (for mode 2) or `SingleLineResult` (for mode 1). Check
    that all field names match the S13 dataclass definitions before calling.
    If any field is missing from S13, print a clearly labelled `[NOT AVAILABLE
    — S13 update required]` placeholder rather than crashing.

12. Run the script manually on a real or synthetic image pair for each mode.
    Confirm:
    - Mode 2: 20 peaks, Fig 3 "IDENTICAL", Fig 5 d≈20.1 mm, Eq 3.97 spread < 2 µm
    - Mode 1: 6–7 peaks, Fig 5 PASS for zero-v check, ε₀ to 6 decimal places

    Report back with the full console output of successful runs in both modes.

13. **Stop and report** after 10–15 minutes if any pytest failures or figure
    rendering errors persist. Do not loop indefinitely.

---

## 23. Change Log

| Version | Date | Changes |
|---------|------|---------|
| v1.0 | 2026-04-13 | Initial combined spec. Supersedes Z01 v0.2 and Z01a v0.2. Adds: unified 1/2-line mode selection with argparse + interactive prompt; `(air)`/`(vac)` wavelength annotation rule; explicit Vaughan §3.5.2 rectangular array table in px²; Vaughan Eqs 3.91–3.97 console calculation block; Melissinos Fig 7.28-style r² scatter plot with 2σ horizontal error bars and integer y-axis; detailed Stage E centre-refinement reporting (coarse grid azimuthal variance + Nelder-Mead sub-pixel, both documented to console); auto-detection of metadata header presence; `load_headerless_bin()` unified helper. S13 correctness review: confirmed correct implementation of Vaughan §3.5.2; no structural changes required in S13. |
