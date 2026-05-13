"""
Script:  H06_airglow_inversion_2026_05_13.py
Purpose: Load an OI 630 nm airglow fringe profile (real or synthetic),
         load a calibration result from H05, run the H06 staged airglow
         inversion to recover the line-of-sight wind velocity, and produce
         a diagnostic figure modelled on Harding et al. (2014) Fig. 5.

═══════════════════════════════════════════════════════════════════════════════
 INPUTS
═══════════════════════════════════════════════════════════════════════════════

  1. Airglow fringe profile (.npy, tabulated vs r²)
     Accepted formats (same as H05/H06):
       (2, N) — row 0 = r² (px²), row 1 = profile (ADU)   ← preferred
       (3, N) — row 0 = r², row 1 = profile, row 2 = SEM
       (N, 2) — col 0 = r², col 1 = profile
       (N,)   — profile only; r² inferred (not recommended)

     Source: M03 annular reduction output, OR profile saved by
             H03_airglow_synthesis_2026_05_12.py.

  2. Calibration result file (<stem>_cal_result.npy)
     Produced by H05_calibration_inversion_2026_05_12.py.
     Contains all 10 instrument parameters fixed from the neon calibration.
     Load with: np.load(path, allow_pickle=True).item()

═══════════════════════════════════════════════════════════════════════════════
 METHOD (follows Harding 2014 §3)
═══════════════════════════════════════════════════════════════════════════════

  Forward model (3 free parameters):
      s(r) = Y_line · Ã(r; λ_c) + B_sci

  where Ã(r; λ_c) uses all 10 instrument parameters fixed from H05.
  Temperature broadening is excluded (delta-function OI source).

  Inversion stages:
    Stage 1 — Brute-force scan of λ_c over ±2.0 FSR (300 grid points)
              Analytic solve for Y_line, B_sci at each scan point.
              Covers both cross-track (offset=0) and along-track (offset=−1,−2).
    Stage 2 — LM refinement over {λ_c, Y_line, B_sci} from scan minimum.

═══════════════════════════════════════════════════════════════════════════════
 CHANGELOG
═══════════════════════════════════════════════════════════════════════════════

 2026-05-13  FSR bug fix (this version):
   - BUG FIX: FSR_OI_M was computed from ETALON_GAP_M (D_25C = 20.0006 mm),
     differing from cal.t_m (≈20.107 mm) by ~0.5% → 25 m/s FSR error and a
     scan spanning only ±7083 m/s, too narrow for along-track (±8000 m/s).
   - fsr_oi is now computed from cal.t_m inside the inversion function and
     passed explicitly to _lambda_c_scan, _run_lm, _compute_covariance, and
     make_figure.  The module-level FSR_OI_M is retained only as a placeholder.
   - Scan half-width widened from ±1.5 FSR to ±2.0 FSR (≈±9394 m/s).
   - n_scan increased from 200 to 300 to maintain grid density.

 2026-05-12  (previous version):
   - Initial release.

  Wind recovery:
    v_rel = c × (λ_c − λ₀) / λ₀    [Harding Eq. 11, inverted]

  STM wind budget: σ_v ≤ 9.8 m/s  →  σ_λ ≤ 2.06e-14 m (0.021 pm)

═══════════════════════════════════════════════════════════════════════════════
 OUTPUT FIGURE (3 panels)
═══════════════════════════════════════════════════════════════════════════════

  Top:    Data + best-fit model vs r²  (Harding Fig. 5 style)
  Middle: Residual (data − model) with ±1σ band
  Bottom: Parameter table with fitted values and 1σ uncertainties

Run from repo root:
    python src/processing/H06_airglow_inversion_2026_05_12.py
"""

import pathlib
import sys
import tkinter as tk
from tkinter import filedialog, simpledialog
import logging
from dataclasses import dataclass

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.optimize import least_squares

# ---------------------------------------------------------------------------
# Make repo root importable
# ---------------------------------------------------------------------------
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.fpi.airy_forward_model_2026_05_05 import airy_modified   # noqa
from windcube.constants import (                                    # noqa
    OI_WAVELENGTH_AIR_M,
    SPEED_OF_LIGHT_MS,
    ETALON_GAP_M,
)

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s  %(name)s  %(message)s")
log = logging.getLogger("H06")

# NOTE: FSR_OI_M is computed from cal.t_m inside the inversion (not from ETALON_GAP_M).
# The D_25C_MM constant (≈ 20.0006 mm) differs from the fitted gap (≈ 20.107 mm)
# by ~0.5%, giving a 25 m/s error in FSR and a too-narrow scan range.
# A module-level placeholder is kept only for the covariance step-size; it is
# overwritten per-call once cal.t_m is known.
FSR_OI_M = OI_WAVELENGTH_AIR_M ** 2 / (2.0 * ETALON_GAP_M)   # placeholder — overwritten in inversion

# ---------------------------------------------------------------------------
# Calibration result shim
# ---------------------------------------------------------------------------

@dataclass
class _CalResult:
    """Lightweight shim holding instrument parameters from H05 .npy file."""
    t_m: float;     alpha: float
    R_refl: float
    I0: float;      I1: float;    I2: float
    sigma0: float;  sigma1: float; sigma2: float
    B: float
    epsilon_cal: float
    quality_flags: int = 0


def load_cal_result(path: pathlib.Path) -> _CalResult:
    """Load H05 calibration result .npy dict and return _CalResult shim."""
    d = np.load(path, allow_pickle=True).item()
    return _CalResult(
        t_m         = float(d['t_m']),
        alpha       = float(d['alpha']),
        R_refl      = float(d['R_refl']),   # = R1 from H05
        I0          = float(d['I0']),
        I1          = float(d['I1']),
        I2          = float(d['I2']),
        sigma0      = float(d['sigma0']),
        sigma1      = float(d.get('sigma1', 0.0)),
        sigma2      = float(d.get('sigma2', 0.0)),
        B           = float(d['B']),
        epsilon_cal = float(d['epsilon_cal']),
        quality_flags = int(d.get('quality_flags', 0)),
    )


# ---------------------------------------------------------------------------
# Forward model
# ---------------------------------------------------------------------------

def _airglow_model(r_arr, r_max, lambda_c_m, Y_line, B_sci, cal,
                   n_fine=500):
    """
    Delta-function airglow fringe: s(r) = Y_line · Ã(r; λ_c) + B_sci
    All instrument parameters fixed from cal (CalibrationResult from H05).
    """
    r_fine = np.linspace(0.0, r_max, n_fine)
    airy   = airy_modified(
        r_fine, lambda_c_m,
        cal.t_m, cal.R_refl, cal.alpha, 1.0, r_max,
        cal.I0, cal.I1, cal.I2,
        cal.sigma0, cal.sigma1, cal.sigma2,
    )
    model_fine = Y_line * airy + B_sci
    return np.interp(r_arr, r_fine, model_fine)


def _airglow_model_fine(r_fine, lambda_c_m, Y_line, B_sci, cal):
    """Model on a pre-built fine grid (for plotting)."""
    airy = airy_modified(
        r_fine, lambda_c_m,
        cal.t_m, cal.R_refl, cal.alpha, 1.0, float(r_fine[-1]),
        cal.I0, cal.I1, cal.I2,
        cal.sigma0, cal.sigma1, cal.sigma2,
    )
    return Y_line * airy + B_sci


# ---------------------------------------------------------------------------
# Stage 1: brute-force λ_c scan
# ---------------------------------------------------------------------------

def _lambda_c_scan(r_good, prof_good, sigma_good, r_max, cal,
                   fsr_oi, n_scan=300, n_fine=500):
    """
    Scan λ_c over ±2.0 FSR around OI_WAVELENGTH_AIR_M.
    At each point, analytically solve for Y_line and B_sci (linear given λ_c).

    Returns (lambda_c_best, chi2_min, scan_ambiguous_flag).

    Scan half-width is ±2.0 FSR (≈ ±9394 m/s with cal.t_m), which covers
    both cross-track (|v| < 500 m/s) and along-track (|v| up to ~8000 m/s).
    fsr_oi must be computed from cal.t_m, NOT from the ETALON_GAP_M constant,
    to avoid a 25 m/s FSR error that shifts the scan edges.
    n_scan increased to 300 to maintain grid density over the wider range.
    """
    lc_lo = OI_WAVELENGTH_AIR_M - 2.0 * fsr_oi
    lc_hi = OI_WAVELENGTH_AIR_M + 2.0 * fsr_oi
    scan  = np.linspace(lc_lo, lc_hi, n_scan)

    r_fine   = np.linspace(0.0, r_max, n_fine)
    n_good   = len(r_good)
    chi2_arr = np.full(n_scan, np.inf)

    for k, lc in enumerate(scan):
        airy_fine = airy_modified(
            r_fine, lc,
            cal.t_m, cal.R_refl, cal.alpha, 1.0, r_max,
            cal.I0, cal.I1, cal.I2,
            cal.sigma0, cal.sigma1, cal.sigma2,
        )
        airy_bins = np.interp(r_good, r_fine, airy_fine)
        w  = 1.0 / sigma_good
        A  = np.column_stack([airy_bins * w, w])
        b  = prof_good * w
        try:
            p  = np.linalg.solve(A.T @ A, A.T @ b)
        except np.linalg.LinAlgError:
            continue
        Y_k, B_k  = p
        model_k   = Y_k * airy_bins + B_k
        resid_k   = (prof_good - model_k) / sigma_good
        chi2_arr[k] = float(np.sum(resid_k ** 2)) / max(n_good - 2, 1)

    best_idx       = int(np.argmin(chi2_arr))
    chi2_min       = float(chi2_arr[best_idx])
    lambda_c_best  = float(scan[best_idx])

    # Check for ambiguous scan (second minimum within 10%)
    alt_mask = np.ones(n_scan, dtype=bool); alt_mask[best_idx] = False
    scan_ambiguous = bool(len(chi2_arr[alt_mask]) > 0 and
                          np.nanmin(chi2_arr[alt_mask]) < chi2_min * 1.10)
    if scan_ambiguous:
        log.warning("SCAN_AMBIGUOUS: second minimum within 10% of best chi2")

    log.info(f"  Scan best: λ_c = {lambda_c_best*1e9:.6f} nm  "
             f"chi2 = {chi2_min:.3f}  "
             f"v_rel ≈ {SPEED_OF_LIGHT_MS*(lambda_c_best-OI_WAVELENGTH_AIR_M)/OI_WAVELENGTH_AIR_M:+.1f} m/s")

    return lambda_c_best, chi2_min, scan_ambiguous


# ---------------------------------------------------------------------------
# Stage 2: LM refinement
# ---------------------------------------------------------------------------

def _run_lm(r_good, prof_good, sigma_good, r_max, cal,
            fsr_oi, lc_init, Y_init, B_init, n_fine=500):
    """LM fit over {λ_c, Y_line, B_sci} with soft-bound penalties.
    Bounds match the ±2.0 FSR scan range (using cal.t_m-derived FSR).
    """
    lc_lo = OI_WAVELENGTH_AIR_M - 2.0 * fsr_oi
    lc_hi = OI_WAVELENGTH_AIR_M + 2.0 * fsr_oi
    Y_lo  = 0.0;  Y_hi  = 1e8
    B_lo  = 0.0;  B_hi  = float(np.min(prof_good)) * 2.0 if np.min(prof_good) > 0 else 1e6

    lo    = np.array([lc_lo, Y_lo,  B_lo])
    hi    = np.array([lc_hi, Y_hi,  B_hi])
    hi_f  = np.where(np.isfinite(hi), hi, lo + 1e3)
    rng_  = hi_f - lo + 1e-30
    pen   = rng_ * 0.01

    p0    = np.array([lc_init, Y_init, B_init])

    def _residuals(p):
        lc, Y, B = p
        model = _airglow_model(r_good, r_max, lc, Y, B, cal, n_fine)
        data_r = (prof_good - model) / sigma_good
        below  = np.maximum(0.0, lo - p) / pen
        above  = np.maximum(0.0, p - hi) / pen
        return np.append(data_r, below + above)

    return least_squares(_residuals, p0, method='lm',
                         ftol=1e-12, xtol=1e-12, gtol=1e-12,
                         max_nfev=50_000)


# ---------------------------------------------------------------------------
# Covariance via FD Jacobian (same approach as H05)
# ---------------------------------------------------------------------------

def _compute_covariance(r_good, prof_good, sigma_good, r_max, cal,
                        fsr_oi, lc, Y_line, B_sci, chi2_red, n_fine=500):
    """
    Compute 3×3 covariance for [λ_c, Y_line, B_sci] via FD Jacobian.

    Uses step = fsr_oi/1000 ≈ 4.7 m/s for λ_c — avoids the ≈1-FSR
    default step that averages out fringe sensitivity.
    Y_line and B_sci use analytic derivatives.
    fsr_oi must be the cal.t_m-derived FSR (not the module-level constant).

    Returns (stderrs [3], cov [3,3]).
    """
    step_lc = fsr_oi / 1000.0   # ≈ 9.87e-18 m → ≈ 4.7 m/s

    # FD for λ_c
    m_hi = _airglow_model(r_good, r_max, lc + step_lc, Y_line, B_sci, cal, n_fine)
    m_lo = _airglow_model(r_good, r_max, lc - step_lc, Y_line, B_sci, cal, n_fine)
    d_dlc = (m_hi - m_lo) / (2.0 * step_lc)

    # Analytic for Y_line (∂model/∂Y_line = Airy) and B_sci (=1)
    r_fine = np.linspace(0.0, r_max, n_fine)
    airy_f = airy_modified(r_fine, lc,
                           cal.t_m, cal.R_refl, cal.alpha, 1.0, r_max,
                           cal.I0, cal.I1, cal.I2,
                           cal.sigma0, cal.sigma1, cal.sigma2)
    airy_b = np.interp(r_good, r_fine, airy_f)

    # Jacobian: J_ij = -∂model_i/∂p_j / sigma_i
    J = np.column_stack([
        -d_dlc          / sigma_good,
        -airy_b         / sigma_good,
        -np.ones(len(r_good)) / sigma_good,
    ])

    try:
        _, s, VT  = np.linalg.svd(J, full_matrices=False)
        threshold = np.finfo(float).eps * max(J.shape) * s[0]
        s_inv     = np.where(s > threshold, 1.0 / s, 0.0)
        JTJ_inv   = (VT.T * s_inv**2) @ VT
        cov       = chi2_red * JTJ_inv
        stderrs   = np.sqrt(np.maximum(np.diag(cov), 0.0))
        if not np.all(np.isfinite(stderrs)):
            raise np.linalg.LinAlgError("non-finite stderrs")
    except (np.linalg.LinAlgError, ValueError) as exc:
        log.warning(f"  Covariance failed: {exc}")
        stderrs = np.full(3, np.inf)
        cov     = np.full((3, 3), np.inf)

    return stderrs, cov


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_profile(path, r_max_px):
    arr = np.load(path, allow_pickle=False)
    sigma = None
    if arr.ndim == 1:
        log.warning("1D array — inferring r²")
        r2, prof = np.linspace(0.0, r_max_px**2, len(arr)), arr.astype(float)
    elif arr.ndim == 2 and arr.shape[0] == 3:
        r2, prof = arr[0].astype(float), arr[1].astype(float)
        sigma    = arr[2].astype(float)
    elif arr.ndim == 2 and arr.shape[0] == 2:
        r2, prof = arr[0].astype(float), arr[1].astype(float)
    elif arr.ndim == 2 and arr.shape[1] == 2:
        r2, prof = arr[:, 0].astype(float), arr[:, 1].astype(float)
    else:
        raise ValueError(f"Unexpected shape {arr.shape}")
    return np.sqrt(np.maximum(r2, 0.0)), prof, sigma


def _estimate_sigma(profile):
    return np.maximum(np.sqrt(np.maximum(profile, 1.0)), 1.0)


def _fmt_unc(value, unit=""):
    if not np.isfinite(value): return "±∞"
    s = f"±{value:.2e}"
    return s + f" {unit}" if unit else s


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(r2_data, profile, sigma,
                r_fine, model_fine,
                lc_m, Y_line, B_sci,
                sigma_lc, sigma_Y, sigma_B,
                chi2_red, n_bins, converged, scan_ambiguous,
                cal: _CalResult, fsr_oi: float,
                source_name: str = "", source_path: str = "") -> plt.Figure:

    r2_fine       = r_fine ** 2
    model_at_data = np.interp(r2_data, r2_fine, model_fine)
    residual      = profile - model_at_data

    v_rel_ms      = SPEED_OF_LIGHT_MS * (lc_m - OI_WAVELENGTH_AIR_M) / OI_WAVELENGTH_AIR_M
    sigma_v_ms    = SPEED_OF_LIGHT_MS * sigma_lc / OI_WAVELENGTH_AIR_M
    forder        = int(round((lc_m - OI_WAVELENGTH_AIR_M) / fsr_oi))

    # STM budget check: σ_v ≤ 9.8 m/s
    budget_ok = sigma_v_ms <= 9.8
    budget_str = f"{'✓' if budget_ok else '✗'} STM budget (≤9.8 m/s)"

    fig = plt.figure(figsize=(14, 11.5))
    gs  = gridspec.GridSpec(3, 1, height_ratios=[3, 1.5, 2.5],
                            hspace=0.08, top=0.90, bottom=0.03,
                            left=0.09, right=0.97)
    ax_fit = fig.add_subplot(gs[0])
    ax_res = fig.add_subplot(gs[1], sharex=ax_fit)
    ax_tbl = fig.add_subplot(gs[2])
    ax_tbl.axis("off")

    # ---- Top panel ----
    ax_fit.errorbar(r2_data, profile, yerr=sigma, fmt="none",
                    ecolor="darkorange", elinewidth=0.5, alpha=0.35, zorder=2)
    ax_fit.plot(r2_data, profile, color="darkorange", lw=0.9, alpha=0.85,
                zorder=3, label=f"Data  ({source_name})")
    ax_fit.plot(r2_fine, model_fine, color="black", lw=1.5, zorder=4,
                label="Best-fit  Y_line·Ã(λ_c) + B_sci")
    ax_fit.set_ylabel("CCD signal  (ADU)", fontsize=11)
    ax_fit.legend(fontsize=8.5, loc="upper right")
    ax_fit.grid(True, alpha=0.2)
    ax_fit.tick_params(labelbottom=False)

    conv_str = "converged" if converged else "NOT converged"
    amb_str  = "  SCAN_AMBIGUOUS" if scan_ambiguous else ""
    col_budget = "darkgreen" if budget_ok else "darkred"

    ax_fit.text(0.02, 0.97,
        f"χ²/ν = {chi2_red:.3f}   {conv_str}{amb_str}\n"
        f"v_rel = {v_rel_ms:+.2f} ± {sigma_v_ms:.2f} m/s   "
        f"fringe_order_offset = {forder}\n"
        f"λ_c = {lc_m*1e9:.7f} nm   {_fmt_unc(sigma_lc*1e15, 'fm')}\n"
        f"σ_v = {sigma_v_ms:.3f} m/s   {budget_str}",
        transform=ax_fit.transAxes, va="top", ha="left", fontsize=8.5,
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="grey", alpha=0.88))

    # Colour the σ_v line green or red depending on budget
    # (we can't easily colour individual lines in the text box, so add a
    # separate annotation for the budget result)
    ax_fit.text(0.02, 0.02,
        f"σ_v = {sigma_v_ms:.3f} m/s   {budget_str}",
        transform=ax_fit.transAxes, va="bottom", ha="left", fontsize=9,
        fontfamily="monospace", color=col_budget, fontweight="bold")

    # ---- Residual panel ----
    ax_res.axhline(0, color="black", lw=0.8, ls="--")
    ax_res.fill_between(r2_data, -sigma, sigma,
                        color="steelblue", alpha=0.22, label="±1σ")
    ax_res.plot(r2_data, residual, color="steelblue", lw=0.8,
                alpha=0.9, label="Residual (data − model)")
    ax_res.set_xlabel(r"$r^2$  (pixels², 2×2 binned)", fontsize=11)
    ax_res.set_ylabel("Residual  (ADU)", fontsize=10)
    ax_res.legend(fontsize=8.5, loc="upper right")
    ax_res.grid(True, alpha=0.2)
    ax_res.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, symmetric=True))

    # ---- Parameter table ----
    # Show free parameters (fitted), then fixed instrument parameters (from cal)
    rows = [
        # Free parameters
        ("λ_c",      f"{lc_m*1e9:.7f} nm",
                     _fmt_unc(sigma_lc*1e15, "fm"),
                     "Doppler-shifted line centre  [FREE]"),
        ("v_rel",    f"{v_rel_ms:+.3f} m/s",
                     _fmt_unc(sigma_v_ms, "m/s"),
                     "Line-of-sight wind  (c·(λ_c−λ₀)/λ₀)"),
        ("Y_line",   f"{Y_line:.2f}",
                     _fmt_unc(sigma_Y),
                     "OI line intensity scale  [FREE]"),
        ("B_sci",    f"{B_sci:.1f} ADU",
                     _fmt_unc(sigma_B),
                     "Science frame bias  [FREE]"),
        # Fixed instrument parameters from H05
        ("t",        f"{cal.t_m*1e3:.7f} mm",
                     "fixed (H05)",
                     "Etalon gap  [H05 Group A]"),
        ("α",        f"{cal.alpha:.5e} rad/px",
                     "fixed (H05)",
                     "Plate scale  [H05 Group A]"),
        ("R_refl",   f"{cal.R_refl:.5f}",
                     "fixed (H05)",
                     "Effective reflectivity  [H05 R1]"),
        ("I₀",       f"{cal.I0:.1f} ADU",
                     "fixed (H05)", "Intensity envelope  [H05]"),
        ("σ₀",       f"{cal.sigma0:.4f} px",
                     "fixed (H05)", "PSF width  [H05 constant blur]"),
        ("B_cal",    f"{cal.B:.1f} ADU",
                     "fixed (H05)", "Calibration bias  [reference only]"),
        ("ε_cal",    f"{cal.epsilon_cal:.6f}",
                     "fixed (H05)",
                     "Zero-wind phase reference  [H05]"),
        ("Δε",       f"{((2*lc_m/OI_WAVELENGTH_AIR_M)%1 - cal.epsilon_cal):+.6f}",
                     "—",
                     "ε_sci − ε_cal  (phase shift, diagnostic)"),
    ]

    tbl = ax_tbl.table(cellText=rows,
        colLabels=["Param", "Fitted / fixed value", "1σ", "Description"],
        cellLoc="left", loc="upper center",
        colWidths=[0.09, 0.22, 0.14, 0.55])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.0)
    tbl.scale(1, 1.18)

    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#c8d8f0"); cell.set_text_props(fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f0f4ff")
        if row in (1, 2, 3, 4):   # free params — green tint
            cell.set_facecolor("#f4fff4")
        if row == 2:               # v_rel — highlight
            cell.set_facecolor(
                "#d4f4d4" if budget_ok else "#ffd4d4")

    ax_tbl.text(0.01, 0.01,
        f"bins used: {n_bins}   free params: 3   "
        f"FSR_OI = {fsr_oi*1e15:.2f} fm   "
        f"v_FSR ≈ {SPEED_OF_LIGHT_MS*fsr_oi/OI_WAVELENGTH_AIR_M:.0f} m/s   "
        f"STM σ_v budget: 9.8 m/s",
        transform=ax_tbl.transAxes, va="bottom", ha="left",
        fontsize=8.5, fontfamily="monospace", color="dimgrey")

    fig.suptitle("WindCube FPI — OI 630 nm Airglow Inversion  "
                 "(3-param: λ_c, Y_line, B_sci / Harding 2014)",
                 fontsize=12, fontweight="bold", y=0.980)
    if source_path:
        fig.text(0.5, 0.963, source_path, ha="center", va="top", fontsize=8,
                 fontfamily="monospace", color="dimgrey")
    fig.text(0.5, 0.948,
        r"$s(r) = Y_\mathrm{line}\cdot\tilde{A}(r;\,\lambda_c,"
        r"\,t,\alpha,R,\sigma_0) + B_\mathrm{sci}$"
        r"$\quad[\tilde{A}=\mathrm{Airy}\times I(r)\circledast\mathcal{G}(\sigma_0)]$"
        r"$\quad\lambda_c = \lambda_0(1+v_\mathrm{rel}/c)$",
        ha="center", va="top", fontsize=9, color="#1a1a2e")
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    # ---- 1. Select airglow profile ----
    _tk = tk.Tk(); _tk.withdraw()
    prof_path = filedialog.askopenfilename(
        title="Select airglow fringe profile (.npy, tabulated vs r²)",
        filetypes=[("NumPy arrays", "*.npy"), ("All files", "*.*")])
    _tk.destroy()
    if not prof_path:
        print("No profile selected — exiting."); return
    prof_path = pathlib.Path(prof_path)
    print(f"\nAirglow profile: {prof_path.name}")

    # ---- 2. Select calibration result ----
    _tk2 = tk.Tk(); _tk2.withdraw()

    # Suggest companion cal_result file
    cal_suggestion = prof_path.parent / (
        prof_path.stem.replace('_profile_vs_r2', '')
                      .replace('_profile_vs_r', '')
                      .replace('_airglow_profile', '') + '_cal_result.npy')
    if not cal_suggestion.exists():
        cal_suggestion = prof_path.parent   # fall back to directory

    cal_path = filedialog.askopenfilename(
        title="Select H05 calibration result file (<stem>_cal_result.npy)",
        initialdir=str(cal_suggestion.parent if cal_suggestion.is_file()
                       else cal_suggestion),
        filetypes=[("NumPy arrays", "*.npy"), ("All files", "*.*")])
    _tk2.destroy()
    if not cal_path:
        print("No calibration result selected — exiting."); return
    cal_path = pathlib.Path(cal_path)
    print(f"Calibration result: {cal_path.name}")

    # ---- 3. r_max ----
    _tk3 = tk.Tk(); _tk3.withdraw()
    r_max_px = simpledialog.askfloat("r_max",
        "Maximum usable fringe radius (px, 2×2 binned).\nFlatSat/flight: 110",
        initialvalue=110.0, minvalue=50.0, maxvalue=200.0,
        parent=_tk3) or 110.0
    _tk3.destroy()

    # ---- 4. Load calibration result ----
    cal = load_cal_result(cal_path)
    print(f"\nCalibration result loaded:")
    print(f"  t      = {cal.t_m*1e3:.7f} mm")
    print(f"  alpha  = {cal.alpha:.5e} rad/px")
    print(f"  R_refl = {cal.R_refl:.5f}  (R1 from H05)")
    print(f"  sigma0 = {cal.sigma0:.4f} px")
    print(f"  B      = {cal.B:.1f} ADU")
    print(f"  ε_cal  = {cal.epsilon_cal:.6f}")
    if cal.quality_flags != 0:
        print(f"  WARNING: quality_flags = {cal.quality_flags} (non-zero)")

    # ---- 5. Load airglow profile ----
    r_grid, profile_adu, sigma_loaded = _load_profile(prof_path, r_max_px)
    in_range    = r_grid <= r_max_px
    r_grid      = r_grid[in_range]
    profile_adu = profile_adu[in_range]

    if sigma_loaded is not None:
        sigma_loaded = sigma_loaded[in_range]
        bad = ~np.isfinite(sigma_loaded)
        if bad.any():
            sigma_loaded[bad] = _estimate_sigma(profile_adu)[bad]
        sigma_adu = sigma_loaded
        print(f"  SEM from file  ({bad.sum()} bins fallback)")
    else:
        sigma_adu = _estimate_sigma(profile_adu)
        print("  SEM: Poisson sqrt(signal)")

    # Sigma floor
    s_floor = max(1.0, float(np.median(profile_adu)) * 0.005)
    sigma_adu = np.maximum(sigma_adu, s_floor)

    print(f"  {len(r_grid)} bins, r∈[{r_grid.min():.1f},{r_grid.max():.1f}] px, "
          f"signal∈[{profile_adu.min():.0f},{profile_adu.max():.0f}] ADU")

    r_max = float(r_max_px)

    # ---- 5b. FSR from calibration gap (authoritative fitted gap) ----
    # Bug fix: the module-level FSR_OI_M uses ETALON_GAP_M (D_25C ≈ 20.0006 mm),
    # which differs from the H05-fitted cal.t_m (≈ 20.107 mm) by ~0.5%, causing a
    # 25 m/s FSR error.  The old ±1.5 FSR scan only spanned ±7083 m/s, missing
    # along-track velocities up to ±8000 m/s.  All scan/bound/step calculations
    # now use fsr_oi derived from cal.t_m.
    fsr_oi = OI_WAVELENGTH_AIR_M ** 2 / (2.0 * cal.t_m)
    v_fsr  = SPEED_OF_LIGHT_MS * fsr_oi / OI_WAVELENGTH_AIR_M
    log.info(f"FSR from cal.t_m={cal.t_m*1e3:.7f} mm: {fsr_oi*1e15:.2f} fm  ({v_fsr:.1f} m/s)")

    # ---- 6. Stage 1: λ_c scan ----
    print(f"\nStage 1 — brute-force λ_c scan (±2.0 FSR, 300 points)…")
    print(f"  FSR = {v_fsr:.1f} m/s  scan spans ±{2.0*v_fsr:.0f} m/s")
    lc_best, chi2_scan, scan_ambiguous = _lambda_c_scan(
        r_grid, profile_adu, sigma_adu, r_max, cal, fsr_oi)

    # Initial Y_line, B_sci from analytic solve at lc_best
    r_fine = np.linspace(0.0, r_max, 500)
    airy_f = airy_modified(r_fine, lc_best,
                           cal.t_m, cal.R_refl, cal.alpha, 1.0, r_max,
                           cal.I0, cal.I1, cal.I2,
                           cal.sigma0, cal.sigma1, cal.sigma2)
    airy_b    = np.interp(r_grid, r_fine, airy_f)
    airy_max  = float(np.max(airy_b)) if np.max(airy_b) > 0 else 1.0
    Y_init    = float(np.max(profile_adu)) / airy_max
    B_init    = max(1.0, float(np.percentile(profile_adu, 5)) * 0.8)

    # ---- 7. Stage 2: LM refinement ----
    print("Stage 2 — LM refinement over {λ_c, Y_line, B_sci}…")
    lm = _run_lm(r_grid, profile_adu, sigma_adu, r_max, cal,
                 fsr_oi, lc_best, Y_init, B_init)

    lc_m   = float(lm.x[0])
    Y_line = float(lm.x[1])
    B_sci  = float(lm.x[2])

    # chi² from data residuals only
    model_bins = _airglow_model(r_grid, r_max, lc_m, Y_line, B_sci, cal)
    data_r     = (profile_adu - model_bins) / sigma_adu
    dof        = max(len(r_grid) - 3, 1)
    chi2_red   = float(np.sum(data_r**2)) / dof
    converged  = bool(lm.success or lm.cost < 1e-10)

    # ---- 8. Covariance ----
    print("Computing covariance (FD Jacobian)…")
    stderrs, cov = _compute_covariance(
        r_grid, profile_adu, sigma_adu, r_max, cal,
        fsr_oi, lc_m, Y_line, B_sci, chi2_red)
    sigma_lc, sigma_Y, sigma_B = stderrs

    # ---- 9. Derived quantities ----
    v_rel_ms   = SPEED_OF_LIGHT_MS * (lc_m - OI_WAVELENGTH_AIR_M) / OI_WAVELENGTH_AIR_M
    sigma_v_ms = SPEED_OF_LIGHT_MS * sigma_lc / OI_WAVELENGTH_AIR_M
    forder     = int(round((lc_m - OI_WAVELENGTH_AIR_M) / fsr_oi))
    eps_sci    = (2.0 * lc_m / OI_WAVELENGTH_AIR_M) % 1.0
    delta_eps  = eps_sci - cal.epsilon_cal
    budget_ok  = sigma_v_ms <= 9.8

    # ---- 10. Print results ----
    print(f"\n{'='*68}")
    print("AIRGLOW INVERSION RESULT  (H06 / Harding 2014)")
    print(f"{'='*68}")
    print(f"  Converged: {converged}   χ²/ν: {chi2_red:.4f}   bins: {len(r_grid)}")
    if scan_ambiguous:
        print("  WARNING: scan was ambiguous (two minima within 10%)")
    print()
    print(f"  --- Free parameters ---")
    print(f"  λ_c    = {lc_m*1e9:.7f} nm   {_fmt_unc(sigma_lc*1e15, 'fm')}")
    print(f"  v_rel  = {v_rel_ms:+.3f} m/s   {_fmt_unc(sigma_v_ms, 'm/s')}")
    print(f"           fringe_order_offset = {forder}")
    print(f"  Y_line = {Y_line:.2f}   {_fmt_unc(sigma_Y)}")
    print(f"  B_sci  = {B_sci:.2f}   {_fmt_unc(sigma_B)} ADU")
    print()
    print(f"  --- STM budget check ---")
    print(f"  σ_v = {sigma_v_ms:.3f} m/s  "
          f"{'✓ within 9.8 m/s budget' if budget_ok else '✗ EXCEEDS 9.8 m/s budget'}")
    print(f"  σ_λ = {sigma_lc*1e15:.3f} fm  (budget: ≤ 20.6 fm)")
    print()
    print(f"  --- Phase diagnostics ---")
    print(f"  ε_sci  = {eps_sci:.6f}")
    print(f"  ε_cal  = {cal.epsilon_cal:.6f}  (from H05)")
    print(f"  Δε     = {delta_eps:+.6f}")
    print(f"{'='*68}")

    # ---- 11. Figure ----
    r_fine_plot = np.linspace(0.0, r_max, 2000)
    model_fine  = _airglow_model_fine(r_fine_plot, lc_m, Y_line, B_sci, cal)

    fig = make_figure(
        r2_data=r_grid**2, profile=profile_adu, sigma=sigma_adu,
        r_fine=r_fine_plot, model_fine=model_fine,
        lc_m=lc_m, Y_line=Y_line, B_sci=B_sci,
        sigma_lc=sigma_lc, sigma_Y=sigma_Y, sigma_B=sigma_B,
        chi2_red=chi2_red, n_bins=len(r_grid),
        converged=converged, scan_ambiguous=scan_ambiguous,
        cal=cal, fsr_oi=fsr_oi,
        source_name=prof_path.name, source_path=str(prof_path))
    plt.show()


if __name__ == "__main__":
    main()
