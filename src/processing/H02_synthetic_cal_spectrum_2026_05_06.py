"""
Script:  p02_synthetic_cal_spectrum_2026_05_06.py
Purpose: Synthesise the two-line neon calibration fringe spectrum expected from
         the WindCube FPI payload using the m02 calibration synthesis module.
         Plots the 1D radial spectrum showing both Ne components and displays
         all synthesis parameter values in a table below the spectrum.

Run from repo root:
    python src/processing/p02_synthetic_cal_spectrum_2026_05_06.py

Loads 1_cal_120sexp_swapped_ROI_L1.1_profile_vs_r.npy or some similar 1-D Averaged Radial Profile vs. R [px]
"""

import pathlib
import sys
import tkinter as tk
from tkinter import filedialog

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Make repo root importable regardless of working directory
# ---------------------------------------------------------------------------
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.fpi.airy_forward_model_2026_05_05 import InstrumentParams, airy_modified, phase_correct_gap  # noqa: E402
from src.fpi.m02_calibration_synthesis_2026_05_05 import synthesise_calibration_image  # noqa: E402
from src.constants import (  # noqa: E402
    NE_WAVELENGTH_1_AIR_M,
    NE_WAVELENGTH_2_AIR_M,
    NE_INTENSITY_2,
)

# ---------------------------------------------------------------------------
# Synthesis parameters
# Etalon gap and plate scale from S13a two-line Tolansky fit (2026-05-06).
# ---------------------------------------------------------------------------
_T_TOLANSKY = 20.1070707e-3   # m — S13a Tolansky physical gap
_EPS_A      = 0.23286         # — S13a Tolansky excess fraction, line a
_T_EFF      = phase_correct_gap(_T_TOLANSKY, _EPS_A, NE_WAVELENGTH_1_AIR_M)
print(f"phase_correct_gap: t_tolansky={_T_TOLANSKY*1e3:.7f} mm  "
      f"t_eff={_T_EFF*1e3:.7f} mm  "
      f"correction={(_T_EFF-_T_TOLANSKY)*1e9:.1f} nm")

PARAMS = InstrumentParams(
    t      = _T_EFF,           # m — phase-corrected for synthesis (see H01 §8)
    R_refl = 0.53,          # — effective reflectivity (FlatSat)
    n      = 1.0,           # — refractive index of air gap
    alpha  = 1.6085e-4,     # rad/px, 2×2 binned (S13a Tolansky 2026-05-06)
    I0     = 1000.0,        # ADU — average intensity envelope
    I1     =   -0.1,        # — linear vignetting coefficient
    I2     =    0.005,      # — quadratic vignetting coefficient
    sigma0 =    0.5,        # px — mean PSF width
    sigma1 =    0.1,        # px — PSF sine variation
    sigma2 =   -0.05,       # px — PSF cosine variation
    B      =  300.0,        # ADU — CCD bias pedestal
    r_max  =  128.0,        # px — max usable fringe radius
)
R_BINS     = 500    # radial bins in 1D profile
IMAGE_SIZE = 256    # CCD dimension (2×2 binned), pixels

# ---------------------------------------------------------------------------
# Optional: load a real radially-averaged calibration profile for comparison.
# Accepts three array shapes saved by annular_reduce / z01:
#   (N,)   — profile values only; r_grid inferred as linspace(0, r_max, N)
#   (2, N) — row 0 = r_grid (px), row 1 = profile
#   (N, 2) — col 0 = r_grid (px), col 1 = profile
# The real profile is scaled so its peak-above-baseline matches the synthetic,
# allowing visual comparison of fringe positions and shapes.
# Cancel the dialog to plot the synthetic spectrum alone.
# ---------------------------------------------------------------------------
_tk = tk.Tk()
_tk.withdraw()
_npy_path = filedialog.askopenfilename(
    title="Select real calibration profile (.npy)  —  Cancel to skip",
    filetypes=[("NumPy arrays", "*.npy"), ("All files", "*.*")],
)
_tk.destroy()

real_profile: np.ndarray | None = None
real_r_grid:  np.ndarray | None = None
real_label:   str               = ""

if _npy_path:
    _arr = np.load(_npy_path)
    if _arr.ndim == 1:
        real_profile = _arr
        real_r_grid  = np.linspace(0.0, PARAMS.r_max, len(_arr))
    elif _arr.ndim == 2 and _arr.shape[0] == 2:
        real_r_grid, real_profile = _arr[0], _arr[1]
    elif _arr.ndim == 2 and _arr.shape[1] == 2:
        real_r_grid, real_profile = _arr[:, 0], _arr[:, 1]
    else:
        print(f"Warning: unexpected array shape {_arr.shape} — skipping real profile.")
    if real_profile is not None:
        real_label = pathlib.Path(_npy_path).name

# ---------------------------------------------------------------------------
# Synthesise noiseless calibration profile
# ---------------------------------------------------------------------------
result = synthesise_calibration_image(
    params=PARAMS,
    image_size=IMAGE_SIZE,
    R_bins=R_BINS,
    add_noise=False,
)
r_grid     = result["r_grid"]       # (R_BINS,) pixels
profile_1d = result["profile_1d"]   # composite A1 + 0.36*A2 + B

# Compute the two Ne components separately for labelled overlay
A1 = airy_modified(r_grid, NE_WAVELENGTH_1_AIR_M, PARAMS)   # 640.22 nm, primary
A2 = airy_modified(r_grid, NE_WAVELENGTH_2_AIR_M, PARAMS)   # 638.30 nm, secondary

# ---------------------------------------------------------------------------
# Derived quantities for the table
# ---------------------------------------------------------------------------
FSR_NE1_M  = NE_WAVELENGTH_1_AIR_M ** 2 / (2.0 * PARAMS.t)
NE_SEP_FSR = (NE_WAVELENGTH_1_AIR_M - NE_WAVELENGTH_2_AIR_M) / FSR_NE1_M

finesse_coeff = PARAMS.finesse_coefficient()
finesse       = PARAMS.finesse()

# ---------------------------------------------------------------------------
# Figure layout: spectrum (top) + three grouped parameter sub-tables (bottom)
# Width ratios reflect row counts: fixed=15, variable=2, derived=4
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(16, 11))
gs  = gridspec.GridSpec(
    2, 3,
    height_ratios=[3, 2],
    width_ratios=[5, 1.5, 2.5],
    hspace=0.10, wspace=0.06,
    top=0.94, bottom=0.04, left=0.06, right=0.97,
)

ax_spec     = fig.add_subplot(gs[0, :])    # spectrum spans full width
ax_fixed    = fig.add_subplot(gs[1, 0])
ax_variable = fig.add_subplot(gs[1, 1])
ax_derived  = fig.add_subplot(gs[1, 2])

for ax in (ax_fixed, ax_variable, ax_derived):
    ax.axis("off")

# ---- Spectrum ---------------------------------------------------------------
ax_spec.plot(r_grid, profile_1d,
             color="black", lw=1.4, zorder=3,
             label="Composite  $A_1 + 0.36\\,A_2 + B$")
ax_spec.fill_between(r_grid, PARAMS.B, A1 + PARAMS.B,
                     color="steelblue", alpha=0.35,
                     label=f"Ne 640.22 nm  (primary,  $\\lambda_1$)")
ax_spec.fill_between(r_grid, PARAMS.B, NE_INTENSITY_2 * A2 + PARAMS.B,
                     color="firebrick", alpha=0.35,
                     label=f"Ne 638.30 nm  (secondary, $\\lambda_2$, ×{NE_INTENSITY_2:.2f})")
ax_spec.axhline(PARAMS.B, color="dimgrey", lw=0.8, ls="--",
                label=f"Bias  $B$ = {PARAMS.B:.0f} ADU")

if real_profile is not None:
    # Scale real profile: subtract its minimum (bias), match peak to synthetic
    _real_above = real_profile - real_profile.min()
    _synth_peak = profile_1d.max() - PARAMS.B
    _scale = _synth_peak / _real_above.max() if _real_above.max() > 0 else 1.0
    _real_scaled = _real_above * _scale + PARAMS.B
    ax_spec.plot(real_r_grid, _real_scaled,
                 color="darkorange", lw=1.1, ls="-", alpha=0.85, zorder=4,
                 label=f"Real  ({real_label},  peak-scaled)")

ax_spec.set_xlim(0.0, PARAMS.r_max)
ax_spec.set_xlabel("Radius  (pixels, 2×2 binned)", fontsize=11)
ax_spec.set_ylabel("CCD signal  (ADU)", fontsize=11)
ax_spec.set_title(
    "WindCube FPI — synthetic neon calibration spectrum  (noiseless)",
    fontsize=12, fontweight="bold",
)
ax_spec.legend(fontsize=9.5, loc="upper right")
ax_spec.grid(True, alpha=0.25)

# ---- Parameter tables: one per group ----------------------------------------
# Each row: (name, symbol, value, unit, type_tag)
# type_tag: 'fixed'    — direct InstrumentParams / source input to airy_modified
#           'variable' — synthesis resolution / control knob, not a physical param
#           'derived'  — computed from other parameters, not a synthesis input
all_rows = [
    # ── Etalon / optics ──
    ("Etalon gap",           "$t$",             f"{PARAMS.t * 1e3:.8f}",               "mm",     "fixed"),
    ("Reflectivity",         "$R_{\\rm refl}$", f"{PARAMS.R_refl:.3f}",               "—",      "fixed"),
    ("Refractive index",     "$n$",             f"{PARAMS.n:.1f}",                    "—",      "fixed"),
    ("Plate scale",          "$\\alpha$",       f"{PARAMS.alpha:.4e}",                "rad/px", "fixed"),
    # ── Intensity envelope ──
    ("Intensity (mean)",     "$I_0$",           f"{PARAMS.I0:.1f}",                   "ADU",    "fixed"),
    ("Vignetting (lin)",     "$I_1$",           f"{PARAMS.I1:.3f}",                   "—",      "fixed"),
    ("Vignetting (quad)",    "$I_2$",           f"{PARAMS.I2:.4f}",                   "—",      "fixed"),
    # ── PSF ──
    ("PSF width (mean)",     "$\\sigma_0$",     f"{PARAMS.sigma0:.2f}",               "px",     "fixed"),
    ("PSF variation",        "$\\sigma_1$",     f"{PARAMS.sigma1:.2f}",               "px",     "fixed"),
    ("PSF variation",        "$\\sigma_2$",     f"{PARAMS.sigma2:.3f}",               "px",     "fixed"),
    # ── CCD / geometry ──
    ("Bias pedestal",        "$B$",             f"{PARAMS.B:.1f}",                    "ADU",    "fixed"),
    ("Max radius",           "$r_{\\rm max}$",  f"{PARAMS.r_max:.1f}",                "px",     "fixed"),
    # ── Neon source ──
    ("Ne₁ wavelength (air)", "$\\lambda_1$",    f"{NE_WAVELENGTH_1_AIR_M * 1e9:.4f}", "nm",    "fixed"),
    ("Ne₂ wavelength (air)", "$\\lambda_2$",    f"{NE_WAVELENGTH_2_AIR_M * 1e9:.4f}", "nm",    "fixed"),
    ("Ne₂/Ne₁ intensity",    "",               f"{NE_INTENSITY_2:.2f}",              "—",      "fixed"),
    # ── Synthesis controls ──
    ("Radial bins",          "$R_{\\rm bins}$", f"{R_BINS}",                          "—",      "variable"),
    ("Image size",           "",               f"{IMAGE_SIZE}×{IMAGE_SIZE}",          "px",     "variable"),
    # ── Derived quantities ──
    ("Finesse coeff.",       "$F$",             f"{finesse_coeff:.2f}",               "—",      "derived"),
    ("Finesse",              "$\\mathcal{F}$",  f"{finesse:.2f}",                     "—",      "derived"),
    ("FSR @ Ne₁",            "",               f"{FSR_NE1_M * 1e12:.3f}",            "pm",     "derived"),
    ("Ne line separation",   "",               f"{NE_SEP_FSR:.2f}",                  "FSR",    "derived"),
]

fixed_rows    = [(n, s, v, u) for n, s, v, u, t in all_rows if t == "fixed"]
variable_rows = [(n, s, v, u) for n, s, v, u, t in all_rows if t == "variable"]
derived_rows  = [(n, s, v, u) for n, s, v, u, t in all_rows if t == "derived"]

_COL_LABELS = ["Parameter", "Symbol", "Value", "Unit"]
_COL_WIDTHS = [0.50, 0.16, 0.22, 0.12]

def _make_subtable(ax, rows, title, header_color):
    tbl = ax.table(
        cellText=rows,
        colLabels=_COL_LABELS,
        cellLoc="left",
        loc="upper center",
        colWidths=_COL_WIDTHS,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.35)
    for (row, _), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(fontweight="bold")
        elif row % 2 == 1:
            cell.set_facecolor("#f5f5f5")
    ax.set_title(title, fontsize=10, pad=4, loc="center", fontweight="bold")

_make_subtable(ax_fixed,    fixed_rows,    "Fixed",    "#c8d8f0")
_make_subtable(ax_variable, variable_rows, "Variable", "#c8e8c4")
_make_subtable(ax_derived,  derived_rows,  "Derived",  "#f0dcc0")

plt.show()
