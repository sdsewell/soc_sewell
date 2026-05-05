"""
Module:      tolansky_1line_2026-05-05.py
Spec:        docs/specs/S13b_tolansky_1_line_2026-05-05.md
Reference:   Vaughan (1989) The Fabry-Perot Interferometer, §3.5.2
             Equations (3.83)–(3.86) — single-line r²-spacing only
             Harding et al. (2014) ApplOpt 53, 666 — WindCube forward model
             Mulligan (1986) J. Phys. E 19, 545 — annular r²-bin reduction
Author:      Claude Code
Project:     WindCube FPI Pipeline — NCAR/HAO
Repo:        soc_sewell
Input:       {stem}_peak_fits.npy  (9-column float64, from annular_reduction.py)
             Peak finding and Gaussian fitting are performed by annular_reduction.py.
             This module reads the pre-computed table directly — no peak detection here.
Note:        Single OI 630.0 nm airglow image analysis only.
             Recovers Δ, ε; recovers α in Mode A (d_cal supplied).
             Does NOT recover absolute d.
             For neon two-line Benoit d recovery, see S13a / tolansky_2line.py
"""

from __future__ import annotations

import warnings
import pathlib
from dataclasses import dataclass
from typing import Union

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Custom exception
# ─────────────────────────────────────────────────────────────────────────────

class InsufficientRingsError(ValueError):
    """Raised when fewer than 2 valid fitted rings remain after NaN filtering."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Output data class  (S13b §5)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TolanskyResult1L:
    """
    Output of the Tolansky single-line analysis (S13b).
    All two_sigma_ fields are exactly 2 × sigma_ (S04 convention).
    Mode-B fields are None when d_cal_m was not supplied.
    """

    # --- Ring data (from _peak_fits.npy, valid rows only) ---
    n_rings:       int              # number of valid fitted rings used
    n_nan_dropped: int              # number of NaN (failed-fit) rows discarded
    p_arr:         np.ndarray       # ring indices (1-based float), shape (n_rings,)
    r_fit_px:      np.ndarray       # Gaussian centroid radii  (px)
    sigma_r_px:    np.ndarray       # 1σ centroid uncertainty  (px)
    r2_px2:        np.ndarray       # r²_p  (px²)
    sigma_r2_px2:  np.ndarray       # σ(r²_p) = 2·r·σ_r  (px²)
    delta_arr:     np.ndarray       # successive differences δ_p  (px²)
    cv_delta:      float            # CV = std(δ) / mean(δ)

    # --- WLS fit (Vaughan §3.5.2, Steps 3–4) ---
    Delta:           float          # Δ = best-fit r²-slope  (px²/ring)
    sigma_Delta:     float          # 1σ
    two_sigma_Delta: float          # exactly 2 × sigma_Delta (S04)
    eps:             float          # ε = fractional order at axis, ∈ [0,1)
    sigma_eps:       float          # 1σ
    two_sigma_eps:   float          # exactly 2 × sigma_eps (S04)
    chi2_dof:        float          # reduced χ²

    # --- Mode A outputs (None if d_cal_m not supplied) ---
    alpha_rad_px:    Union[float, None]   # α = 1/f_px  (rad/px)
    sigma_alpha:     Union[float, None]   # 1σ
    two_sigma_alpha: Union[float, None]   # exactly 2 × sigma_alpha (S04)
    f_px:            Union[float, None]   # focal length (pixels)
    sigma_f_px:      Union[float, None]   # 1σ

    # --- Δ consistency check (None if f_cal_px or d_cal_m not supplied) ---
    Delta_pred:        Union[float, None]  # predicted Δ_OI from S13a f_cal and d_cal
    Delta_consistency: Union[float, None]  # |Δ_obs − Δ_pred| / Δ_pred

    # --- Provenance ---
    lam_OI_nm: float          # input wavelength in nm (630.0304)
    d_cal_mm:  Union[float, None]   # d_cal used (mm), or None


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 — Input loading helper
# ─────────────────────────────────────────────────────────────────────────────

def load_peak_fits(path_or_array):
    """
    Load and filter the 9-column _peak_fits.npy table.

    Accepts either a file path (str / pathlib.Path) or a pre-loaded (N,9)
    ndarray (the latter allows direct use in tests without touching the
    filesystem).

    Returns
    -------
    p_arr, r_fit_px, sigma_r_px, r2_px2, sigma_r2_px2 : np.ndarray
    n_nan_dropped : int

    Raises
    ------
    InsufficientRingsError  if fewer than 2 valid rows remain after filtering.
    ValueError              if the array shape is not (N, 9).
    """
    if isinstance(path_or_array, np.ndarray):
        peaks = path_or_array
    else:
        peaks = np.load(path_or_array)

    if peaks.ndim != 2 or peaks.shape[1] != 9:
        raise ValueError(
            f"Expected (N, 9) peak_fits array, got shape {peaks.shape}. "
            "Columns: peak_num | r_raw_px | r_fit_px | sigma_r_fit_px | "
            "r_fit_sq | sigma_r_fit_sq | amplitude_adu | width_px | reduced_chi2"
        )

    valid = np.isfinite(peaks[:, 2])
    n_nan_dropped = int((~valid).sum())
    peaks_ok = peaks[valid]

    if n_nan_dropped > 0:
        warnings.warn(
            f"{n_nan_dropped} peak row(s) with failed Gaussian fit dropped "
            "(NaN in r_fit_px)."
        )

    if peaks_ok.shape[0] < 2:
        raise InsufficientRingsError(
            f"Only {peaks_ok.shape[0]} valid fitted peak(s) after NaN filter "
            f"({n_nan_dropped} dropped); need ≥ 2 for Tolansky analysis."
        )

    p_arr        = peaks_ok[:, 0]   # peak_num  (1-based)
    r_fit_px     = peaks_ok[:, 2]   # r_fit (px)   — TRF Gaussian centroid μ
    sigma_r_px   = peaks_ok[:, 3]   # sigma_r_fit (px) — 1σ on μ
    r2_px2       = peaks_ok[:, 4]   # μ²  (px²)
    sigma_r2_px2 = peaks_ok[:, 5]   # 2·μ·σ_μ  (px²)

    return p_arr, r_fit_px, sigma_r_px, r2_px2, sigma_r2_px2, n_nan_dropped


# ─────────────────────────────────────────────────────────────────────────────
# Task 3 — Single-line WLS helper
# ─────────────────────────────────────────────────────────────────────────────

def run_single_line_wls(p, r2, sigma_r2):
    """
    Weighted least-squares fit of the model  r²_p = S·p + b.

    Weights  w_p = 1 / σ(r²_p)²  (col 5 values are already 2·r·σ_r).

    Uses the closed-form normal equations (Vaughan §3.5.2 Steps 3–4;
    Bevington & Robinson §6.3):

        Λ = Σw · Σwp² − (Σwp)²
        S = (Σw · Σwpr²  −  Σwp · Σwr²) / Λ        [= Δ]
        b = (Σwp² · Σwr²  −  Σwp · Σwpr²) / Λ

        Var(S) = Σw  / Λ
        Var(b) = Σwp² / Λ

    Returns
    -------
    dict with keys:
        Delta, sigma_Delta, eps, sigma_eps, chi2_dof, delta_arr, cv_delta
    """
    w = 1.0 / sigma_r2 ** 2

    sum_w    = np.sum(w)
    sum_wp   = np.sum(w * p)
    sum_wp2  = np.sum(w * p ** 2)
    sum_wr2  = np.sum(w * r2)
    sum_wpr2 = np.sum(w * p * r2)

    Lambda = sum_w * sum_wp2 - sum_wp ** 2

    S = (sum_w * sum_wpr2 - sum_wp * sum_wr2) / Lambda
    b = (sum_wp2 * sum_wr2 - sum_wp * sum_wpr2) / Lambda

    var_S   = sum_w   / Lambda
    var_b   = sum_wp2 / Lambda
    sigma_S = float(np.sqrt(var_S))
    sigma_b = float(np.sqrt(var_b))

    # Fractional order ε = 1 + b/S, wrapped to [0, 1)
    eps       = float((1.0 + b / S) % 1.0)
    sigma_eps = float(np.sqrt((sigma_b / S) ** 2 + (b * sigma_S / S ** 2) ** 2))

    # Reduced χ²  (undefined for N ≤ 2)
    N = len(p)
    r2_pred = S * p + b
    if N > 2:
        chi2_dof = float(np.sum(w * (r2 - r2_pred) ** 2) / (N - 2))
    else:
        chi2_dof = float("nan")

    # Successive differences and their CV
    delta_arr  = np.diff(r2)
    mean_delta = float(np.mean(delta_arr))
    std_delta  = float(np.std(delta_arr))
    cv_delta   = std_delta / mean_delta if mean_delta != 0.0 else float("nan")

    return {
        "Delta":       float(S),
        "sigma_Delta": float(sigma_S),
        "eps":         eps,
        "sigma_eps":   sigma_eps,
        "chi2_dof":    chi2_dof,
        "delta_arr":   delta_arr,
        "cv_delta":    float(cv_delta),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Task 4 — Mode A α recovery helper
# ─────────────────────────────────────────────────────────────────────────────

def recover_alpha(Delta, sigma_Delta, d_cal_m, sigma_d_cal_m, lam_m, n_air=1.0):
    """
    Recover plate scale α = 1/f_px from the measured Δ and a calibration d.

        α  =  sqrt(λ_OI · n_air / (d_cal · Δ))          [Vaughan 3.85 rearranged]
        σ(α)/α  =  (1/2) · sqrt( (σ_Δ/Δ)² + (σ_d/d)² )

    Returns (alpha, sigma_alpha, f_px, sigma_f_px) or (None,)*4 if d_cal_m
    is None.
    """
    if d_cal_m is None:
        return None, None, None, None

    sigma_d = sigma_d_cal_m if sigma_d_cal_m is not None else 0.0

    alpha      = float(np.sqrt(lam_m * n_air / (d_cal_m * Delta)))
    sigma_alpha = float(
        0.5 * alpha * np.sqrt(
            (sigma_Delta / Delta) ** 2 + (sigma_d / d_cal_m) ** 2
        )
    )
    f_px       = float(1.0 / alpha)
    sigma_f_px = float(sigma_alpha / alpha ** 2)

    return alpha, sigma_alpha, f_px, sigma_f_px


# ─────────────────────────────────────────────────────────────────────────────
# Task 5 — Top-level run function
# ─────────────────────────────────────────────────────────────────────────────

def run_tolansky_1line(
    peaks_input,
    lam_OI_m:      float = 630.0304e-9,
    pixel_pitch_m: float = 32e-6,
    n_air:         float = 1.0,
    d_cal_m=None,
    sigma_d_cal_m=None,
    f_cal_px=None,
) -> TolanskyResult1L:
    """
    Run the Tolansky single-line analysis (S13b).

    Parameters
    ----------
    peaks_input : np.ndarray | str | pathlib.Path
        Either a pre-loaded (N, 9) float64 array or a path to a
        _peak_fits.npy file produced by annular_reduction.py.
    lam_OI_m : float
        Vacuum wavelength of the airglow emission line [m].  Default OI 630.0304 nm.
    pixel_pitch_m : float
        Pixel pitch [m].  Not used in computation; accepted for API consistency.
    n_air : float
        Refractive index of etalon gap (default 1.0, vacuum).
    d_cal_m : float or None
        Calibration plate spacing from S13a [m].  Required for Mode A.
    sigma_d_cal_m : float or None
        1σ uncertainty on d_cal_m [m].  Used in Mode A uncertainty budget.
    f_cal_px : float or None
        Calibration focal length from S13a [px].  Used for Δ consistency check.

    Returns
    -------
    TolanskyResult1L
    """
    p_arr, r_fit_px, sigma_r_px, r2_px2, sigma_r2_px2, n_nan_dropped = \
        load_peak_fits(peaks_input)

    wls = run_single_line_wls(p_arr, r2_px2, sigma_r2_px2)

    if wls["cv_delta"] > 0.05:
        warnings.warn(
            f"CV(δ) = {wls['cv_delta']:.3f} > 5% "
            "— possible ring misidentification or spurious detection."
        )

    Delta       = wls["Delta"]
    sigma_Delta = wls["sigma_Delta"]

    alpha, sigma_alpha, f_px, sigma_f_px = recover_alpha(
        Delta, sigma_Delta, d_cal_m, sigma_d_cal_m, lam_OI_m, n_air
    )

    # Step 4 — Δ consistency check against S13a calibration prediction
    Delta_pred        = None
    Delta_consistency = None
    if f_cal_px is not None and d_cal_m is not None:
        Delta_pred = float(f_cal_px ** 2 * lam_OI_m / (n_air * d_cal_m))
        Delta_consistency = float(abs(Delta - Delta_pred) / Delta_pred)
        if Delta_consistency > 0.01:
            warnings.warn(
                f"Δ_obs/Δ_pred discrepancy = {Delta_consistency*100:.2f}% > 1% "
                "— possible scale change or ring misidentification."
            )

    return TolanskyResult1L(
        n_rings        = len(p_arr),
        n_nan_dropped  = n_nan_dropped,
        p_arr          = p_arr,
        r_fit_px       = r_fit_px,
        sigma_r_px     = sigma_r_px,
        r2_px2         = r2_px2,
        sigma_r2_px2   = sigma_r2_px2,
        delta_arr      = wls["delta_arr"],
        cv_delta       = wls["cv_delta"],
        Delta           = Delta,
        sigma_Delta     = sigma_Delta,
        two_sigma_Delta = 2.0 * sigma_Delta,
        eps             = wls["eps"],
        sigma_eps       = wls["sigma_eps"],
        two_sigma_eps   = 2.0 * wls["sigma_eps"],
        chi2_dof        = wls["chi2_dof"],
        alpha_rad_px    = alpha,
        sigma_alpha     = sigma_alpha,
        two_sigma_alpha = (2.0 * sigma_alpha) if sigma_alpha is not None else None,
        f_px            = f_px,
        sigma_f_px      = sigma_f_px,
        Delta_pred          = Delta_pred,
        Delta_consistency   = Delta_consistency,
        lam_OI_nm = lam_OI_m * 1e9,
        d_cal_mm  = (d_cal_m * 1e3) if d_cal_m is not None else None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Task 6 — Printed summary  (S13b §6)
# ─────────────────────────────────────────────────────────────────────────────

def print_tolansky_1line(result: TolanskyResult1L) -> None:
    """Print formatted summary table (S13b §6)."""
    SEP = "=" * 46
    n  = result.n_rings
    p1 = int(result.p_arr[0])
    pn = int(result.p_arr[-1])

    nan_note = (f"  [{result.n_nan_dropped} NaN row(s) dropped]"
                if result.n_nan_dropped > 0 else "")
    print(f"\n=== TOLANSKY SINGLE-LINE ANALYSIS  (S13b) ===")
    print(f"Source line:  OI 630.0 nm  "
          f"(λ = {result.lam_OI_nm:.4f} nm vacuum)")
    print(f"Rings used: {n}   (p = {p1} … {pn}){nan_note}")
    print()

    # Ring table — show up to 6 entries, truncate with … if longer
    MAX = 6
    if n <= MAX:
        idx = list(range(n))
        trunc = False
    else:
        idx = list(range(MAX - 1)) + [n - 1]
        trunc = True

    p_row  = ""
    r2_row = ""
    for k, i in enumerate(idx):
        if trunc and k == MAX - 2:
            p_row  += "   …"
            r2_row += "   …"
        p_row  += f"  {int(result.p_arr[i]):>8}"
        r2_row += f"  {result.r2_px2[i]:>8.2f}"

    print(f"  p  :{p_row}")
    print(f"  r² :{r2_row}  (px²)")

    # Successive δ values
    n_show = min(len(result.delta_arr), MAX - 1)
    d_parts = []
    for i in range(n_show):
        pl = int(result.p_arr[i])
        ph = int(result.p_arr[i + 1])
        d_parts.append(f"δ{pl}{ph}={result.delta_arr[i]:.1f}")
    if len(result.delta_arr) > MAX - 1:
        d_parts.append("…")
    if d_parts:
        print("          " + "   ".join(d_parts))

    cv_flag = "PASS" if result.cv_delta < 0.05 else "WARN"
    print(f"  CV(δ) = {result.cv_delta:.3f}   [{cv_flag}]")
    print()

    print("WLS FIT  (Vaughan §3.5.2)")
    print(f"  Δ       = {result.Delta:.2f} ± {result.sigma_Delta:.2f} px²"
          f"   (2σ = {result.two_sigma_Delta:.2f} px²)")
    print(f"  ε       = {result.eps:.4f}  ± {result.sigma_eps:.4f}")
    print(f"  χ²_dof  = {result.chi2_dof:.2f}")
    print()

    d_str = f"{result.d_cal_mm:.4f} mm" if result.d_cal_mm is not None \
            else "not supplied"
    print(f"MODE A  (d_cal: {d_str})")
    if result.alpha_rad_px is not None:
        print(f"  α   = {result.alpha_rad_px:.4E} ± {result.sigma_alpha:.4E} rad/px"
              f"   (2σ = {result.two_sigma_alpha:.4E})")
        print(f"  f   = {result.f_px:.1f}   ± {result.sigma_f_px:.1f} px")
    else:
        print("  (supply d_cal_m to enable Mode A)")
    print()

    cal_ok = result.Delta_pred is not None
    print(f"CONSISTENCY CHECK  "
          f"(f_cal, d_cal supplied: {'YES' if cal_ok else 'NO'})")
    if cal_ok:
        dc_flag = "PASS" if result.Delta_consistency < 0.01 else "WARN"
        print(f"  Δ_pred = {result.Delta_pred:.2f} px²")
        print(f"  |Δ_obs − Δ_pred| / Δ_pred = "
              f"{result.Delta_consistency*100:.1f}%   [{dc_flag}]")
    print(SEP)


# ─────────────────────────────────────────────────────────────────────────────
# Task 7 — M05 priors helper  (S13b §7)
# ─────────────────────────────────────────────────────────────────────────────

def to_m05_priors_1line(result: TolanskyResult1L):
    """
    Return M05 prior dict if Mode A succeeded (d_cal was available),
    else return None so the caller falls back to S13a priors.

    Does NOT supply t_init_mm or t_bounds_mm — those always come from S13a.
    """
    if result.alpha_rad_px is None:
        return None
    return {
        "alpha_init":   result.alpha_rad_px,
        "alpha_bounds": (result.alpha_rad_px * 0.875,
                         result.alpha_rad_px * 1.125),   # ±12.5 %
        "epsilon_sci":  result.eps,
    }
