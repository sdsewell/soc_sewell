"""
Script:  p02_synthetic_cal_spectrum_2026_05_06.py
Purpose: Synthesise the two-line neon calibration fringe spectrum expected from
         the WindCube FPI payload using the m02 calibration synthesis module.
         Plots the 1D radial spectrum showing both Ne components and displays
         all synthesis parameter values in a table below the spectrum.

Run from repo root:
    python src/processing/p02_synthetic_cal_spectrum_2026_05_06.py
"""

import pathlib
import sys

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Make repo root importable regardless of working directory
# ---------------------------------------------------------------------------
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.fpi.airy_forward_model_2026_05_05 import InstrumentParams, airy_modified  # noqa: E402
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
PARAMS = InstrumentParams(
    t      = 20.1071e-3,    # m — S13a Tolansky two-line fit (2026-05-06)
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
# Figure layout: spectrum (top, 60%) + parameter table (bottom, 40%)
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 2], hspace=0.08,
                        top=0.94, bottom=0.04, left=0.07, right=0.97)

ax_spec  = fig.add_subplot(gs[0])
ax_table = fig.add_subplot(gs[1])
ax_table.axis("off")

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

ax_spec.set_xlim(0.0, PARAMS.r_max)
ax_spec.set_xlabel("Radius  (pixels, 2×2 binned)", fontsize=11)
ax_spec.set_ylabel("CCD signal  (ADU)", fontsize=11)
ax_spec.set_title(
    "WindCube FPI — synthetic neon calibration spectrum  (noiseless)",
    fontsize=12, fontweight="bold",
)
ax_spec.legend(fontsize=9.5, loc="upper right")
ax_spec.grid(True, alpha=0.25)

# ---- Parameter table --------------------------------------------------------
table_rows = [
    # ── Etalon / optics ──
    ("Etalon gap",         "$t$",           f"{PARAMS.t * 1e3:.4f}",    "mm"),
    ("Reflectivity",       "$R_{\\rm refl}$", f"{PARAMS.R_refl:.3f}",    "—"),
    ("Refractive index",   "$n$",           f"{PARAMS.n:.1f}",          "—"),
    ("Plate scale",        "$\\alpha$",     f"{PARAMS.alpha:.4e}",      "rad/px"),
    ("Finesse coeff.",     "$F$",           f"{finesse_coeff:.2f}",     "—"),
    ("Finesse",            "$\\mathcal{F}$",f"{finesse:.2f}",           "—"),
    ("FSR @ Ne₁",         "",              f"{FSR_NE1_M * 1e12:.3f}",  "pm"),
    # ── Intensity envelope ──
    ("Intensity (mean)",   "$I_0$",         f"{PARAMS.I0:.1f}",        "ADU"),
    ("Vignetting (lin)",   "$I_1$",         f"{PARAMS.I1:.3f}",        "—"),
    ("Vignetting (quad)",  "$I_2$",         f"{PARAMS.I2:.4f}",        "—"),
    # ── PSF ──
    ("PSF width (mean)",   "$\\sigma_0$",   f"{PARAMS.sigma0:.2f}",    "px"),
    ("PSF variation",      "$\\sigma_1$",   f"{PARAMS.sigma1:.2f}",    "px"),
    ("PSF variation",      "$\\sigma_2$",   f"{PARAMS.sigma2:.3f}",    "px"),
    # ── CCD / geometry ──
    ("Bias pedestal",      "$B$",           f"{PARAMS.B:.1f}",         "ADU"),
    ("Max radius",         "$r_{\\rm max}$",f"{PARAMS.r_max:.1f}",     "px"),
    ("Radial bins",        "$R_{\\rm bins}$",f"{R_BINS}",              "—"),
    ("Image size",         "",              f"{IMAGE_SIZE}×{IMAGE_SIZE}", "px"),
    # ── Neon source ──
    ("Ne₁ wavelength (air)", "$\\lambda_1$", f"{NE_WAVELENGTH_1_AIR_M * 1e9:.4f}", "nm"),
    ("Ne₂ wavelength (air)", "$\\lambda_2$", f"{NE_WAVELENGTH_2_AIR_M * 1e9:.4f}", "nm"),
    ("Ne₂/Ne₁ intensity",  "",              f"{NE_INTENSITY_2:.2f}",  "—"),
    ("Ne line separation", "",              f"{NE_SEP_FSR:.2f}",       "FSR"),
]

n = len(table_rows)
n_half = (n + 1) // 2
left_rows  = table_rows[:n_half]
right_rows = table_rows[n_half:]
# Pad to equal length
while len(right_rows) < len(left_rows):
    right_rows.append(("", "", "", ""))

cell_text = [
    [lr[0], lr[1], lr[2], lr[3],
     "  ",
     rr[0], rr[1], rr[2], rr[3]]
    for lr, rr in zip(left_rows, right_rows)
]
col_labels = ["Parameter", "Symbol", "Value", "Unit",
              "  ",
              "Parameter", "Symbol", "Value", "Unit"]
col_widths = [0.175, 0.065, 0.08, 0.055,
              0.01,
              0.175, 0.065, 0.08, 0.055]

tbl = ax_table.table(
    cellText=cell_text,
    colLabels=col_labels,
    cellLoc="left",
    loc="center",
    colWidths=col_widths,
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.scale(1, 1.35)

# Style the header row and the spacer column
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor("#d0d8e8")
        cell.set_text_props(fontweight="bold")
    elif col == 4:   # spacer column
        cell.set_edgecolor("none")
        cell.set_facecolor("white")
    elif row % 2 == 1:
        cell.set_facecolor("#f5f5f5")

ax_table.set_title("Synthesis parameters", fontsize=10, pad=4, loc="left")

plt.show()
