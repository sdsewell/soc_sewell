"""
Script:  p05_calibration_inversion_2026_05_06.py
Purpose: Load a real two-line neon calibration radial profile (tabulated vs r²),
         run the H05 staged Levenberg-Marquardt calibration inversion, and
         produce a diagnostic figure modelled on Harding et al. (2014) Fig. 4:
           • Top panel:  data vs. best-fit modified Airy model (vs r²)
           • Middle panel: residual (data − model), with ±1σ band
           • Bottom panel: fitted parameter table

         The fit uses:
           • t_eff   — phase-corrected Tolansky gap (H01 §8), fixed as seed
             with tight ±20 µm Tolansky-tightened bounds
           • alpha   — Tolansky plate scale, same tight-bounds treatment
           • R, I0, I1, I2, sigma0, sigma1, sigma2, B — all freely fitted
             from the fringe shape (Harding Table 1 Group B parameters)

Input .npy file formats accepted (same as p02):
   (N,)      — profile values only; r² inferred as linspace(0, r_max²?, N)
               NOT RECOMMENDED — see note below.
   (2, N)    — row 0 = r² grid (px²), row 1 = profile (ADU)
   (N, 2)    — col 0 = r² grid (px²), col 1 = profile (ADU)

   The preferred format is (2, N) or (N, 2) with an explicit r² axis,
   matching the output of annular_reduction.py (_profile_vs_r2.npy files).

Run from repo root:
    python src/processing/p05_calibration_inversion_2026_05_06.py

Dependencies: src.fpi.m05_calibration_inversion_2026_05_05
              src.fpi.airy_forward_model_2026_05_06  (for phase_correct_gap)
              windcube.constants
"""

import pathlib
import sys
import tkinter as tk
from tkinter import filedialog, simpledialog
import logging

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Make repo root importable regardless of working directory
# ---------------------------------------------------------------------------
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.fpi.airy_forward_model_2026_05_06 import (   # noqa: E402
    InstrumentParams,
    airy_modified,
    phase_correct_gap,
)
from src.fpi.m05_calibration_inversion_2026_05_05 import (  # noqa: E402
    fit_calibration_fringe,
    FitConfig,
    FitFlags,
    CalibrationResult,
    _neon_model,         # forward model helper — used to reconstruct best-fit curve
    _PARAM_IDX,
)
from windcube.constants import (                       # noqa: E402
    NE_WAVELENGTH_1_AIR_M,
    NE_WAVELENGTH_2_AIR_M,
    NE_INTENSITY_2,
)

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s  %(name)s  %(message)s")
log = logging.getLogger("p05")

# ---------------------------------------------------------------------------
# Minimal FringeProfile shim
# ---------------------------------------------------------------------------
# M05's fit_calibration_fringe() expects a FringeProfile object with these
# fields.  Rather than importing M03 (which may not be available on all
# machines), we build a lightweight shim from the loaded .npy data.

@dataclass
class _FringeProfile:
    """Minimal FringeProfile compatible with M05 fit_calibration_fringe()."""
    profile:       np.ndarray   # 1D radial profile in ADU, shape (N,)
    r_grid:        np.ndarray   # radial bin centres in pixels, shape (N,)
    sigma_profile: np.ndarray   # 1σ uncertainty per bin, shape (N,)
    masked:        np.ndarray   # bool mask: True = exclude, shape (N,)
    r_max_px:      float        # maximum usable radius (pixels)


# ---------------------------------------------------------------------------
# Tolansky result shim
# ---------------------------------------------------------------------------
# M05's FitConfig(tolansky=...) expects an object with .d_m, .alpha_rad_px,
# .eps1 attributes (the TolanskyResult duck-typed interface).

@dataclass
class _TolanskyShim:
    d_m:         float   # Tolansky physical gap (metres) — for FitConfig bounds
    alpha_rad_px: float  # Tolansky plate scale (rad/px)
    eps1:         float  # excess fraction for λ₁ (used for epsilon_cal traceability)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_profile(path: pathlib.Path, r_max_px: float):
    """
    Load a .npy profile file tabulated vs r² and return (r_grid, profile_adu).

    Accepted shapes:
      (N,)   — profile only; r² inferred as linspace(0, r_max²*N/(N-1), N)
               This is a last resort — always prefer an explicit r² axis.
      (2, N) — row 0 = r² (px²), row 1 = profile (ADU)
      (N, 2) — col 0 = r² (px²), col 1 = profile (ADU)
    """
    arr = np.load(path)

    if arr.ndim == 1:
        log.warning(
            "Profile is 1D — no explicit r² axis. "
            "Inferring r² as linspace(0, r_max², N). "
            "Prefer saving as (2,N) with explicit r² column."
        )
        n    = len(arr)
        r2   = np.linspace(0.0, r_max_px ** 2, n)
        prof = arr.astype(float)
    elif arr.ndim == 2 and arr.shape[0] == 2:
        r2   = arr[0].astype(float)
        prof = arr[1].astype(float)
    elif arr.ndim == 2 and arr.shape[1] == 2:
        r2   = arr[:, 0].astype(float)
        prof = arr[:, 1].astype(float)
    else:
        raise ValueError(
            f"Unexpected array shape {arr.shape}. "
            "Expected (N,), (2,N), or (N,2)."
        )

    r_grid = np.sqrt(np.maximum(r2, 0.0))   # convert r² → r (pixels)
    return r_grid, prof


def _estimate_sigma(profile: np.ndarray) -> np.ndarray:
    """
    Estimate per-bin 1σ uncertainty from the local scatter of the profile.

    Uses a 5-bin rolling standard deviation as a proxy for photon + readout
    noise.  A minimum floor of max(1.0, 0.5% of median signal) is applied.
    """
    n = len(profile)
    half = 2
    sigma = np.empty(n)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        sigma[i] = float(np.std(profile[lo:hi]))
    floor = max(1.0, float(np.median(profile)) * 0.005)
    return np.maximum(sigma, floor)


def _flag_string(flags: int) -> str:
    """Human-readable summary of FitFlags bitmask."""
    names = {
        FitFlags.TOLANSKY_NOT_PROVIDED: "TOLANSKY_NOT_PROVIDED",
        FitFlags.T_ALPHA_DEGENERATE:    "T_ALPHA_DEGENERATE",
        FitFlags.R_SIGMA_DEGENERATE:    "R_SIGMA_DEGENERATE",
        FitFlags.PARAM_AT_BOUND:        "PARAM_AT_BOUND",
        FitFlags.CHI2_HIGH:             "CHI2_HIGH",
        FitFlags.CHI2_VERY_HIGH:        "CHI2_VERY_HIGH",
        FitFlags.CHI2_LOW:              "CHI2_LOW",
        FitFlags.MULTIPLE_MINIMA:       "MULTIPLE_MINIMA",
        FitFlags.PSF_UNPHYSICAL:        "PSF_UNPHYSICAL",
        FitFlags.STDERR_NONE:           "STDERR_NONE",
    }
    active = [v for k, v in names.items() if flags & k]
    return "GOOD" if not active else " | ".join(active)


# ---------------------------------------------------------------------------
# Figure builder — Harding Fig. 4 style
# ---------------------------------------------------------------------------

def make_harding_figure(
    r2_data:    np.ndarray,    # r² axis for data (px²)
    profile:    np.ndarray,    # measured profile (ADU)
    sigma:      np.ndarray,    # 1σ per-bin uncertainty (ADU)
    r2_model:   np.ndarray,    # r² axis for model curve (finer grid, px²)
    model_best: np.ndarray,    # best-fit composite model (ADU), same length as r2_model
    cal:        "CalibrationResult",
    source_name: str = "",
) -> plt.Figure:
    """
    Three-panel Harding Fig. 4 style diagnostic figure.

    Top:    Data (orange) + best-fit model (black) vs r²
    Middle: Residual (data − model) with ±1σ band
    Bottom: Fitted parameter table with values and 1σ uncertainties
    """
    # ---- Interpolate model to data r² grid for residuals ----------------
    model_at_data = np.interp(r2_data, r2_model, model_best)
    residual      = profile - model_at_data

    # ---- Layout ---------------------------------------------------------
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(
        3, 1,
        height_ratios=[3, 1.5, 2],
        hspace=0.08,
        top=0.93, bottom=0.04, left=0.09, right=0.97,
    )
    ax_fit  = fig.add_subplot(gs[0])
    ax_res  = fig.add_subplot(gs[1], sharex=ax_fit)
    ax_tbl  = fig.add_subplot(gs[2])
    ax_tbl.axis("off")

    # ---- Top panel: data + model ----------------------------------------
    ax_fit.errorbar(
        r2_data, profile, yerr=sigma,
        fmt="none", ecolor="darkorange", elinewidth=0.6,
        capsize=0, alpha=0.5, zorder=2, label="Data ±1σ",
    )
    ax_fit.plot(
        r2_data, profile,
        color="darkorange", lw=0.9, alpha=0.85, zorder=3,
        label=f"Data  ({source_name})",
    )
    ax_fit.plot(
        r2_model, model_best,
        color="black", lw=1.5, zorder=4,
        label="Best-fit modified Airy  (H05 Stage 4)",
    )
    ax_fit.set_ylabel("CCD signal  (ADU)", fontsize=11)
    ax_fit.legend(fontsize=9, loc="upper right")
    ax_fit.grid(True, alpha=0.2)
    ax_fit.tick_params(labelbottom=False)

    # Annotate chi² and convergence
    chi2_str  = f"χ²/ν = {cal.chi2_reduced:.3f}"
    conv_str  = "converged" if cal.converged else "NOT converged"
    flags_str = _flag_string(cal.quality_flags)
    ax_fit.text(
        0.02, 0.97,
        f"{chi2_str}   {conv_str}\nFlags: {flags_str}",
        transform=ax_fit.transAxes,
        va="top", ha="left", fontsize=8.5,
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="grey", alpha=0.85),
    )

    # ---- Middle panel: residual -----------------------------------------
    ax_res.axhline(0, color="black", lw=0.8, ls="--")
    ax_res.fill_between(r2_data, -sigma, sigma,
                        color="steelblue", alpha=0.25, label="±1σ")
    ax_res.plot(r2_data, residual,
                color="steelblue", lw=0.9, alpha=0.9, label="Residual")
    ax_res.set_xlabel(r"$r^2$  (pixels², 2×2 binned)", fontsize=11)
    ax_res.set_ylabel("Residual  (ADU)", fontsize=10)
    ax_res.legend(fontsize=8.5, loc="upper right")
    ax_res.grid(True, alpha=0.2)
    ax_res.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, symmetric=True))

    # ---- Bottom panel: parameter table ----------------------------------
    # Rows: parameter, fitted value, 1σ uncertainty, description
    rows = [
        # param,       fitted,    stderr,    description
        ("t",
         f"{cal.t_m * 1e3:.7f} mm",
         f"±{cal.sigma_t_m * 1e6:.2f} µm",
         "Etalon gap  [Tolansky-seeded, Group A]"),
        ("α",
         f"{cal.alpha:.5e} rad/px",
         f"±{cal.sigma_alpha:.2e}",
         "Plate scale  [Tolansky-seeded, Group A]"),
        ("R",
         f"{cal.R_refl:.4f}",
         f"±{cal.sigma_R_refl:.4f}",
         "Effective reflectivity  [Group B]"),
        ("I₀",
         f"{cal.I0:.1f} ADU",
         f"±{cal.sigma_I0:.1f}",
         "Mean intensity  [Group B]"),
        ("I₁",
         f"{cal.I1:.4f}",
         f"±{cal.sigma_I1:.4f}",
         "Linear vignetting  [Group B]"),
        ("I₂",
         f"{cal.I2:.5f}",
         f"±{cal.sigma_I2:.5f}",
         "Quadratic vignetting  [Group B]"),
        ("σ₀",
         f"{cal.sigma0:.4f} px",
         f"±{cal.sigma_sigma0:.4f}",
         "PSF base width  [Group B]"),
        ("σ₁",
         f"{cal.sigma1:.4f} px",
         f"±{cal.sigma_sigma1:.4f}",
         "PSF sin variation  [Group B]"),
        ("σ₂",
         f"{cal.sigma2:.4f} px",
         f"±{cal.sigma_sigma2:.4f}",
         "PSF cos variation  [Group B]"),
        ("B",
         f"{cal.B:.1f} ADU",
         f"±{cal.sigma_B:.2f}",
         "CCD bias pedestal  [Group B]"),
        ("ε_cal",
         f"{cal.epsilon_cal:.6f}",
         f"±{cal.sigma_epsilon_cal:.6f}",
         "Fractional order at centre  (zero-wind reference)"),
    ]

    col_labels = ["Param", "Fitted value", "1σ", "Description"]
    col_widths = [0.06, 0.22, 0.16, 0.56]

    tbl = ax_tbl.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="left",
        loc="upper center",
        colWidths=col_widths,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.32)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#c8d8f0")
            cell.set_text_props(fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f0f4ff")

    # χ² by stage below table
    stage_str = "  ".join(
        f"S{i+1}: {v:.2f}" for i, v in enumerate(cal.chi2_by_stage)
    )
    ax_tbl.text(
        0.01, 0.02,
        f"χ²/ν by stage:  {stage_str}    "
        f"   bins used: {cal.n_bins_used}   free params: {cal.n_params_free}",
        transform=ax_tbl.transAxes,
        va="bottom", ha="left", fontsize=8.5,
        fontfamily="monospace",
        color="dimgrey",
    )

    fig.suptitle(
        "WindCube FPI — Neon Calibration Fringe Inversion  (H05 / Harding 2014)",
        fontsize=12, fontweight="bold", y=0.97,
    )
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    # ---- 1. Load the .npy profile ----------------------------------------
    _tk = tk.Tk()
    _tk.withdraw()
    npy_path = filedialog.askopenfilename(
        title="Select neon calibration profile (.npy, tabulated vs r²)",
        filetypes=[("NumPy arrays", "*.npy"), ("All files", "*.*")],
    )
    _tk.destroy()

    if not npy_path:
        print("No file selected — exiting.")
        return

    npy_path = pathlib.Path(npy_path)
    print(f"\nLoaded: {npy_path.name}")

    # ---- 2. Ask for r_max and Tolansky values ----------------------------
    _tk2 = tk.Tk()
    _tk2.withdraw()

    r_max_px = simpledialog.askfloat(
        "r_max",
        "Maximum usable fringe radius (pixels, 2×2 binned).\n"
        "FlatSat/flight: 110   Synthetic: 128",
        initialvalue=110.0,
        minvalue=50.0, maxvalue=200.0,
    )
    if r_max_px is None:
        r_max_px = 110.0
        print(f"  r_max not set — using default {r_max_px} px")

    t_tolansky_mm = simpledialog.askfloat(
        "Tolansky gap",
        "Tolansky-recovered etalon gap (mm).\n"
        "Example: 20.1070707",
        initialvalue=20.1070707,
        minvalue=19.5, maxvalue=20.5,
    )
    if t_tolansky_mm is None:
        t_tolansky_mm = 20.1070707
        print(f"  t_tolansky not set — using default {t_tolansky_mm} mm")

    eps_a = simpledialog.askfloat(
        "Tolansky ε_a",
        "Tolansky excess fraction ε_a for λ₁ = 640.2248 nm.\n"
        "Example: 0.23286\n"
        "(Used by phase_correct_gap to anchor absolute fringe position.)",
        initialvalue=0.23286,
        minvalue=0.0, maxvalue=0.9999,
    )
    if eps_a is None:
        eps_a = 0.23286
        print(f"  eps_a not set — using default {eps_a}")

    alpha_tolansky = simpledialog.askfloat(
        "Tolansky alpha",
        "Tolansky plate scale α (rad/px, 2×2 binned).\n"
        "Example: 1.6084e-4",
        initialvalue=1.6084e-4,
        minvalue=1.0e-5, maxvalue=1.0e-3,
    )
    if alpha_tolansky is None:
        alpha_tolansky = 1.6084e-4
        print(f"  alpha not set — using default {alpha_tolansky:.4e} rad/px")

    _tk2.destroy()

    t_tolansky_m = t_tolansky_mm * 1e-3

    # ---- 3. Phase-correct the gap ----------------------------------------
    t_eff = phase_correct_gap(t_tolansky_m, eps_a, NE_WAVELENGTH_1_AIR_M)
    correction_nm = (t_eff - t_tolansky_m) * 1e9
    print(f"\nphase_correct_gap:")
    print(f"  t_tolansky = {t_tolansky_m * 1e3:.7f} mm")
    print(f"  t_eff      = {t_eff * 1e3:.7f} mm")
    print(f"  correction = {correction_nm:.1f} nm")
    print(f"  check:  (2*t_eff/λ₁) % 1 = {(2*t_eff/NE_WAVELENGTH_1_AIR_M) % 1:.6f}  "
          f"(should be ε_a = {eps_a:.6f})")

    # ---- 4. Load profile and build FringeProfile shim --------------------
    r_grid, profile_adu = _load_profile(npy_path, r_max_px)

    # Restrict to r <= r_max_px
    in_range = r_grid <= r_max_px
    r_grid      = r_grid[in_range]
    profile_adu = profile_adu[in_range]

    if len(r_grid) < 30:
        raise ValueError(
            f"Only {len(r_grid)} bins within r_max = {r_max_px} px — "
            "check r_max or file format."
        )

    sigma_adu = _estimate_sigma(profile_adu)
    masked    = np.zeros(len(r_grid), dtype=bool)  # no bins masked

    fp = _FringeProfile(
        profile       = profile_adu,
        r_grid        = r_grid,
        sigma_profile = sigma_adu,
        masked        = masked,
        r_max_px      = r_max_px,
    )

    print(f"\nProfile loaded: {len(r_grid)} bins, "
          f"r in [{r_grid.min():.1f}, {r_grid.max():.1f}] px, "
          f"signal in [{profile_adu.min():.0f}, {profile_adu.max():.0f}] ADU")

    # ---- 5. Build FitConfig with Tolansky shim ---------------------------
    # Use t_eff as the seed for M05.  t_eff and t_tolansky differ by < λ/4,
    # so the Tolansky ±20 µm bounds are centred on t_eff here.
    # Note: we pass t_eff as d_m so that FitConfig.resolve() places the
    # ±20 µm window around the phase-corrected value, not the physical gap.
    # This is intentional: for the LM fit, t_eff is the better starting point.
    tolansky_shim = _TolanskyShim(
        d_m          = t_eff,           # phase-corrected — seeds t and bounds
        alpha_rad_px = alpha_tolansky,
        eps1         = eps_a,
    )

    config = FitConfig(
        tolansky     = tolansky_shim,
        R_init       = 0.53,
        sigma0_init  = 0.5,
        max_nfev     = 100_000,
        ftol=1e-14, xtol=1e-14, gtol=1e-14,
        n_convergence_perturbations = 3,
        require_convergence_guard   = False,
    )

    # ---- 6. Run H05 staged inversion ------------------------------------
    print("\nRunning H05 staged LM inversion (Stages 1–4 + convergence guard)…")
    try:
        cal = fit_calibration_fringe(fp, config)
    except Exception as e:
        print(f"\nERROR: H05 inversion failed: {e}")
        raise

    print(f"\n{'='*60}")
    print("H05 CALIBRATION INVERSION RESULT")
    print(f"{'='*60}")
    print(f"  Converged:    {cal.converged}")
    print(f"  χ²/ν:         {cal.chi2_reduced:.4f}")
    print(f"  χ²/ν by stage: {[f'{v:.3f}' for v in cal.chi2_by_stage]}")
    print(f"  Quality flags: {_flag_string(cal.quality_flags)}")
    print(f"  Bins used:    {cal.n_bins_used}")
    print()
    print(f"  --- Group A (Tolansky-seeded) ---")
    print(f"  t      = {cal.t_m * 1e3:.7f} ± {cal.sigma_t_m * 1e6:.3f} µm  mm")
    print(f"  alpha  = {cal.alpha:.5e} ± {cal.sigma_alpha:.2e} rad/px")
    print()
    print(f"  --- Group B (fringe shape) ---")
    print(f"  R      = {cal.R_refl:.4f} ± {cal.sigma_R_refl:.4f}")
    print(f"  I0     = {cal.I0:.1f} ± {cal.sigma_I0:.1f} ADU")
    print(f"  I1     = {cal.I1:.4f} ± {cal.sigma_I1:.4f}")
    print(f"  I2     = {cal.I2:.5f} ± {cal.sigma_I2:.5f}")
    print(f"  sigma0 = {cal.sigma0:.4f} ± {cal.sigma_sigma0:.4f} px")
    print(f"  sigma1 = {cal.sigma1:.4f} ± {cal.sigma_sigma1:.4f} px")
    print(f"  sigma2 = {cal.sigma2:.4f} ± {cal.sigma_sigma2:.4f} px")
    print(f"  B      = {cal.B:.2f} ± {cal.sigma_B:.2f} ADU")
    print()
    print(f"  --- Phase reference ---")
    print(f"  ε_cal  = {cal.epsilon_cal:.6f} ± {cal.sigma_epsilon_cal:.6f}")
    print(f"{'='*60}")

    # ---- 7. Reconstruct best-fit model on a fine r grid -----------------
    r_fine   = np.linspace(0.0, r_max_px, 2000)
    r2_fine  = r_fine ** 2
    r2_data  = r_grid ** 2

    model_fine = _neon_model(
        r_fine, r_max_px,
        cal.t_m, cal.R_refl, cal.alpha,
        cal.I0, cal.I1, cal.I2,
        cal.sigma0, cal.sigma1, cal.sigma2,
    ) + cal.B

    # ---- 8. Build and show figure ----------------------------------------
    fig = make_harding_figure(
        r2_data    = r2_data,
        profile    = profile_adu,
        sigma      = sigma_adu,
        r2_model   = r2_fine,
        model_best = model_fine,
        cal        = cal,
        source_name = npy_path.name,
    )
    plt.show()


if __name__ == "__main__":
    main()
