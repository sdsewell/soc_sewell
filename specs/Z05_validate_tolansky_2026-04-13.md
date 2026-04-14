# Z05 — Validate Tolansky Two-Line Analysis

**Spec ID:** Z05
**Spec file:** `docs/specs/Z05_validate_tolansky_2026-04-13.md`
**Project:** WindCube FPI Pipeline
**Institution:** NCAR / High Altitude Observatory (HAO)
**Status:** Specification — ready for implementation in VS Code
**Depends on:**
  - Z00 (produces `{stem}_fringe_peaks.npy` — this script's only input)
  - S13 (M13 `tolansky_2026-04-13` — provides `run_tolansky`, `run_single_line_wls`,
    `TolanskyResult`, `to_m05_priors`)
**Produces:**
  - Terminal printout of the Vaughan rectangular array and Benoit recovery
  - `{stem}_z05_table31.png` — Vaughan Table 3.1 rendered as a matplotlib figure
  - `{stem}_z05_benoit.png` — Benoit recovery summary rendered as a matplotlib figure
  - `{stem}_tolansky_result.npy` — `TolanskyResult` fields saved as a structured array
    for downstream use by S14 (M05)
**References:**
  - Vaughan (1989) *The Fabry-Perot Interferometer*, §3.5.2, Table 3.1 and
    Equations (3.83)–(3.97)
  - S13 — Tolansky Two-Line Analysis Specification (2026-04-13)
  - Z00 — Validate Annular Reduction and Peak Finding (2026-04-13)
**Last updated:** 2026-04-13

---

## 1. Purpose

This script validates the S13 Tolansky analysis by:

1. Loading a `_fringe_peaks.npy` file produced by Z00.
2. Detecting whether the data contains one spectral line (airglow, OI 630 nm)
   or two (neon calibration, 640.2 + 638.3 nm).
3. Running the appropriate analysis from S13.
4. Rendering Vaughan's Table 3.1 rectangular array — both as terminal text
   and as a saved matplotlib figure — so the user can verify the ring
   positions and successive differences directly against the textbook layout.
5. Printing and saving the Benoit d/f/α recovery results.

The script is entirely self-contained and requires no command-line arguments.

---

## 2. Script location and invocation

```
soc_sewell/
└── validation/
    └── z05_validate_tolansky_2026-04-13.py
```

```bash
python validation/z05_validate_tolansky_2026-04-13.py
```

No arguments. One file-selection dialog at startup.

---

## 3. Input

### 3.1 File selection

Use the same `select_npy_file()` helper pattern as Z00 §5 (tkinter dialog
with `input()` fallback):

```python
peaks_path = select_npy_file("Select the fringe-peaks .npy file")
```

The file must be a structured array whose dtype matches Z00 §10.1.
The script reads:

```python
peak_table = np.load(peaks_path, allow_pickle=False)
```

### 3.2 Fringe-type detection

Inspect the `family` field values present in the array:

```python
families = set(peak_table['family'])

if 'neon_A' in families and 'neon_B' in families:
    fringe_type = 'two_line'       # neon calibration
elif 'oi' in families:
    fringe_type = 'one_line'       # airglow science
else:
    raise ValueError(f"Unrecognised family values: {families}")
```

Print the detected type and ring counts immediately:

```
Loaded: {peaks_path}
Fringe type detected: {fringe_type}
  neon_A rings: {n_A}   (λₐ = 640.2248 nm)
  neon_B rings: {n_B}   (λᵦ = 638.2991 nm)
```

or for single-line:

```
  OI rings: {n_OI}   (λ = 630.0304 nm)
```

---

## 4. Analysis

### 4.1 Two-line neon (fringe_type == 'two_line')

Extract ring data for each family:

```python
rows_a = peak_table[peak_table['family'] == 'neon_A']
rows_b = peak_table[peak_table['family'] == 'neon_B']

p_a  = rows_a['ring_index'].astype(float)
r2_a = rows_a['r2_fit_px2']
sr_a = rows_a['sigma_r_fit_px']    # σ(r_p) in pixels

p_b  = rows_b['ring_index'].astype(float)
r2_b = rows_b['r2_fit_px2']
sr_b = rows_b['sigma_r_fit_px']
```

Call the top-level S13 function:

```python
from fpi.tolansky_2026_04_13 import run_tolansky, to_m05_priors, TolanskyResult

result = run_tolansky(
    p_a, r2_a, sr_a,
    p_b, r2_b, sr_b,
    lam_a_m       = LAM_A_M,
    lam_b_m       = LAM_B_M,
    d_prior_m     = D_PRIOR_M,
    pixel_pitch_m = PIXEL_PITCH_M,
)
priors = to_m05_priors(result)
```

### 4.2 Single-line OI (fringe_type == 'one_line')

```python
from fpi.tolansky_2026_04_13 import run_single_line_wls

rows_oi = peak_table[peak_table['family'] == 'oi']
p_oi    = rows_oi['ring_index'].astype(float)
r2_oi   = rows_oi['r2_fit_px2']
sr_oi   = rows_oi['sigma_r_fit_px']

wls = run_single_line_wls(p_oi, r2_oi, sr_oi)
```

For single-line data, only the rectangular array and WLS summary are
produced.  The Benoit d recovery requires two lines and is skipped;
a note is printed:

```
NOTE: Single-line data — Benoit d recovery requires two lines.
      Only the rectangular array and WLS fit are shown.
```

---

## 5. Parameters block

```python
# ── Instrument constants ──────────────────────────────────────────────────
LAM_A_M       = 640.2248e-9   # m — neon line a  (Burns et al. 1950)
LAM_B_M       = 638.2991e-9   # m — neon line b
LAM_OI_M      = 630.0304e-9   # m — OI rest wavelength (air)
D_PRIOR_M     = 20.008e-3     # m — ICOS spacer (GNL4096-R); resolves N_Δ only
PIXEL_PITCH_M = 32e-6         # m — 2×2 binned CCD97  (2 × 16 µm)
```

---

## 6. Terminal output

Print the following blocks in order, using the exact headers shown:

### 6.1 Vaughan rectangular array (terminal)

```
=== VAUGHAN TABLE 3.1 RECTANGULAR ARRAY ===
(Vaughan 1989 §3.5.2 — r² values in px², successive differences δ below)

Component a  (λₐ = 640.2248 nm)
  p  :       1          2          3     ...     {n_A}
  r² :  XXXXXXX.XX XXXXXXX.XX XXXXXXX.XX  ...  XXXXXXX.XX   px²
                  δ=XXXX.XX  δ=XXXX.XX  ...
  Δₐ (mean δ) = XXXXX.XX px²    σ = XX.XX px²
  εₐ          =      X.XXXX      σ =  X.XXXX

Component b  (λᵦ = 638.2991 nm)
  [same layout]

Ratio  Δₐ/Δᵦ observed = X.XXXXXX   expected (λₐ/λᵦ) = X.XXXXXX
       residual = {residual_ppm:.2f} ppm    [accept if < 1000 ppm]
```

### 6.2 Benoit recovery (terminal, two-line only)

```
=== BENOIT RECOVERY  (Vaughan Eqs. 3.94–3.97) ===
  N_Δ = nₐ − nᵦ  = {N_Delta:d}   [from d_prior = {D_PRIOR_M*1e3:.3f} mm]

  d   = {d_mm:.4f} ± {sigma_d_mm:.4f} mm   (2σ = {two_sigma_d_mm:.4f} mm)
  f   = {f_px:.1f} ± {sigma_f_px:.1f} px
      = {f_mm:.3f} ± {sigma_f_mm:.3f} mm   (2σ = {two_sigma_f_mm:.3f} mm)
  α   = {alpha:.4e} ± {sigma_alpha:.2e} rad/px
        (2σ = {two_sigma_alpha:.2e} rad/px)
  f cross-check: f_b = {f_b_px:.1f} px   consistency = {consistency_ppm:.1f} ppm
```

### 6.3 M05 priors dict (two-line only)

```
=== M05 PRIORS  (to_m05_priors output) ===
  t_init_mm      = {t_init_mm:.4f}
  t_bounds_mm    = ({t_lo:.4f}, {t_hi:.4f})
  alpha_init     = {alpha_init:.4e}
  alpha_bounds   = ({alpha_lo:.4e}, {alpha_hi:.4e})
  epsilon_cal_1  = {eps1:.5f}   (εₐ, λₐ = 640.2248 nm)
  epsilon_cal_2  = {eps2:.5f}   (εᵦ, λᵦ = 638.2991 nm)
```

---

## 7. Figures

Two figures are produced.  Both are saved to the same directory as the input
`.npy` file. Filenames use the stem of the input file.

### Figure 1 — Vaughan Table 3.1  (`{stem}_z05_table31.png`)

**Purpose:** A clean, publication-ready rendering of Vaughan's rectangular
array as a matplotlib figure, suitable for a lab notebook or poster.

**Layout:** `axes.axis('off')` with a `matplotlib.table.Table`.

**Table structure (two-line neon):**

Row 0 — header:
```
Component | p = 1 | δ₁₂ | p = 2 | δ₂₃ | p = 3 | … | p = n | Δ (mean) | ε
```

Row 1 — line a:
```
λₐ = 640.2248 nm | r²₁ | δ₁₂ | r²₂ | δ₂₃ | … | r²ₙ | Δₐ ± σ | εₐ ± σ
```

Row 2 — line b:
```
λᵦ = 638.2991 nm | r²₁ | δ₁₂ | r²₂ | … | r²ₙ | Δᵦ ± σ | εᵦ ± σ
```

Row 3 — ratio line:
```
Δₐ/Δᵦ | {obs:.6f} | expected λₐ/λᵦ = {exp:.6f} | residual = {ppm:.1f} ppm
```

**Column formatting:**
- r²_p values: `f"{val:.2f}"` (2 decimal places, px² units)
- δ values: `f"{val:.2f}"` in a lighter grey font to visually distinguish
  them from the r² values (use `cellColours` with `'#f5f5f5'` for δ columns)
- Δ column: `f"{val:.2f} ± {sigma:.2f}"`
- ε column: `f"{val:.4f} ± {sigma:.4f}"`

**Header row:** bold, background `'#003479'` (NCAR blue), white text.
**Line a rows:** background `'#ddeeff'`.
**Line b rows:** background `'#fffacc'`.
**Ratio row:** background `'#e8e8e8'`, spanning appropriately.

**Title:** `"Vaughan (1989) Table 3.1 Analog\nr²-values (px²) and successive differences δ (px²)"`

For single-line OI, produce one data row labelled `λ_OI = 630.0304 nm`
with only Δ and ε columns (no Benoit ratio row).

**Figure size:** `figsize=(max(10, 1.4 * n_rings), 3.5)` so columns don't
crowd for 10 rings per family.

### Figure 2 — Benoit recovery summary  (`{stem}_z05_benoit.png`)

**Purpose:** Visualise the Δ values, the Δ ratio constraint, and the recovered
physical parameters in a compact three-panel figure.

**Layout:** 1 row × 3 panels.

**Left panel — successive differences δ_p per family:**
- x-axis: ring pair index (1–2, 2–3, …)
- y-axis: δ (px²)
- Neon-A: blue circles with error bars (σ propagated from σ(r²))
- Neon-B: red squares with error bars
- Horizontal dashed lines at Δₐ (blue) and Δᵦ (red)
- Shaded bands: ±1σ around each mean Δ
- x-axis label: `"Ring pair (p, p+1)"`, y-axis label: `"δ = r²_{p+1} − r²_p  (px²)"`
- Title: `"Successive r²-differences"`

**Middle panel — Δ ratio diagnostic:**
- Single horizontal axis showing observed Δₐ/Δᵦ vs. expected λₐ/λᵦ
- Plot a horizontal error bar for the observed ratio (centre ± 1σ propagated)
- Mark the expected ratio λₐ/λᵦ with a vertical dashed red line
- x-axis: `"Δₐ / Δᵦ"`, range centred on the expected ratio ±5×σ_ratio
- Annotate the residual in ppm inside the panel
- Title: `"Δ ratio constraint (Eq. 3.85/3.87)"`

**Right panel — recovered parameters text box:**
- `axes.axis('off')`
- Use `axes.text()` to render a multi-line summary:

```
Benoit recovery  (Eqs. 3.94–3.97)

N_Δ = {N_Delta}

d  = {d_mm:.4f} ± {sigma_d_mm:.4f} mm
     (2σ = {two_sigma_d_mm:.4f} mm)

f  = {f_mm:.3f} ± {sigma_f_mm:.3f} mm

α  = {alpha:.4e} rad/px
```

Monospace font (`family='monospace'`), fontsize 10.

**Suptitle:** `"Fig Z05: Tolansky Two-Line Analysis\n{stem}"`

---

## 8. Output file: `{stem}_tolansky_result.npy`

Save the scalar fields of `TolanskyResult` as a single-row structured array
so the result can be loaded by S14 (M05) without re-running the analysis.

```python
dtype_result = np.dtype([
    ('Delta_a',            'f8'),
    ('sigma_Delta_a',      'f8'),
    ('eps_a',              'f8'),
    ('sigma_eps_a',        'f8'),
    ('chi2_dof_a',         'f8'),
    ('Delta_b',            'f8'),
    ('sigma_Delta_b',      'f8'),
    ('eps_b',              'f8'),
    ('sigma_eps_b',        'f8'),
    ('chi2_dof_b',         'f8'),
    ('Delta_ratio_obs',    'f8'),
    ('Delta_ratio_expected','f8'),
    ('Delta_ratio_residual','f8'),
    ('N_Delta',            'i4'),
    ('d_m',                'f8'),
    ('sigma_d_m',          'f8'),
    ('two_sigma_d_m',      'f8'),
    ('f_px',               'f8'),
    ('sigma_f_px',         'f8'),
    ('two_sigma_f_px',     'f8'),
    ('f_b_px',             'f8'),
    ('f_consistency',      'f8'),
    ('alpha_rad_px',       'f8'),
    ('sigma_alpha',        'f8'),
    ('two_sigma_alpha',    'f8'),
    ('lam_a_nm',           'f8'),
    ('lam_b_nm',           'f8'),
    ('n_rings_a',          'i4'),
    ('n_rings_b',          'i4'),
])
```

The array arrays (`delta_a`, `delta_b`) are not stored here — they are
available in the terminal output and Figure 2.

For single-line data, fill all `_b` fields with `np.nan` and `n_rings_b = 0`.

---

## 9. Acceptance criteria

| Check | Criterion |
|-------|-----------|
| File loads | structured array with correct dtype |
| Family detection | `fringe_type` in `{'two_line', 'one_line'}` |
| Δ ratio residual (two-line) | `< 1000 ppm` |
| N_Δ (two-line) | `== −189` for WindCube data |
| d recovery (two-line) | `19.5 mm < d < 20.5 mm` |
| f recovery (two-line) | `f_consistency < 1000 ppm` |
| χ²_dof (each line) | `0.1 < χ²_dof < 10` |
| Both figures saved | PNG files exist and non-zero size |
| Result .npy saved | exists, 1 row, correct dtype |

Script exits with code 0 on PASS, code 1 on FAIL.

---

## 10. File locations

```
soc_sewell/
├── validation/
│   └── z05_validate_tolansky_2026-04-13.py
└── docs/specs/
    └── Z05_validate_tolansky_2026-04-13.md
```

---

## 11. Instructions for Claude Code

### Pre-implementation reads

```bash
cat docs/specs/Z05_validate_tolansky_2026-04-13.md   # this file
cat docs/specs/S13_tolansky_analysis_2026-04-13.md   # API reference
cat docs/specs/Z00_validate_annular_reduction_peak_finding_2026-04-13.md  # §10.1 dtype
```

### Confirm existing tests pass

```bash
pytest tests/ -v
```

All tests must pass before writing any new file.

### Task sequence

**Task 1 — Scaffold**
Create `validation/z05_validate_tolansky_2026-04-13.py` with:
- Parameters block (Section 5)
- `select_npy_file()` helper (same pattern as Z00 §5)
- `detect_fringe_type()` function (Section 3.2)
- Empty stubs: `make_table31_figure()`, `make_benoit_figure()`, `save_result_npy()`
- `main()` calling everything in order

**Task 2 — Analysis**
Implement the analysis calls (Section 4).
Print terminal output blocks 6.1–6.3.
Confirm terminal output is correct before proceeding to figures.

**Task 3 — Figure 1 (Table 3.1)**
Implement `make_table31_figure()` (Section 7, Figure 1).
Test with synthetic data if no .npy file is available.

**Task 4 — Figure 2 (Benoit summary)**
Implement `make_benoit_figure()` (Section 7, Figure 2).

**Task 5 — Save result .npy**
Implement `save_result_npy()` (Section 8).

**Task 6 — Acceptance check**
Implement Section 9 checks. Exit with code 0/1.

**Task 7 — Commit**
```bash
git add validation/z05_validate_tolansky_2026-04-13.py
git commit -m "feat(z05): add Tolansky validation script, Vaughan Table 3.1 rendering
Implements: Z05_validate_tolansky_2026-04-13.md"
```

### Report format

```
=== Z05 CLAUDE CODE REPORT ===
Date: YYYY-MM-DD
Script: validation/z05_validate_tolansky_2026-04-13.py

INPUT: {filename}  ({n_A} neon-A rings, {n_B} neon-B rings)

TOLANSKY RESULTS:
  Δₐ = XXXX.XX ± X.XX px²
  Δᵦ = XXXX.XX ± X.XX px²
  Δₐ/Δᵦ = X.XXXXXX  (expected X.XXXXXX,  residual XX.X ppm)
  N_Δ = XXX
  εₐ = X.XXXX ± X.XXXX
  εᵦ = X.XXXX ± X.XXXX
  d  = XX.XXXX ± X.XXXX mm  (2σ = X.XXXX mm)
  f  = XXXXX.X ± X.X px  (= XXX.XX ± X.XX mm)
  α  = X.XXXXE-4 ± X.XXXXE-6 rad/px
  f_consistency = X.X ppm

FIGURES SAVED:
  table31 : {filename}  {size KB}
  benoit  : {filename}  {size KB}

NPY SAVED: {filename}

ACCEPTANCE: PASS / FAIL
  [list any failing criteria]

DEVIATIONS FROM SPEC:
  [list any, or "None"]
==============================
```

Stop and return this report if any task takes more than 15 minutes.
