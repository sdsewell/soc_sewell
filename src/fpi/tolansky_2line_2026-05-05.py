"""
Module:      tolansky_2line_2026-05-05.py
Spec:        docs/specs/S13a_tolansky_2_line_2026-05-05.md  v0.4
Reference:   Vaughan (1989) The Fabry-Perot Interferometer, §3.5.2
             Equations (3.83)-(3.97) -- rectangular array / Benoit method
             Burns, Adams & Longwell (1950) -- Ne IAU standard wavelengths
Author:      Claude Code
Project:     WindCube FPI Pipeline -- NCAR/HAO
Repo:        soc_sewell
Input:       {stem}_peak_fits.npy  (9-column float64, from annular_reduction.py)
             Family assignment by amplitude threshold (640nm ~3x brighter).
Note:        Two-line neon calibration lamp analysis only.
             Focal length f is not computed or stored; alpha is the sole
             plate-scale output.
             For airglow single-line analysis, see S13b / tolansky_1line.py
"""

from __future__ import annotations

import pathlib
import warnings
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Task 1 -- custom exception
# ---------------------------------------------------------------------------

class InsufficientRingsError(ValueError):
    """Raised when the peak table has too few valid rings for analysis."""


# ---------------------------------------------------------------------------
# Task 1 -- result dataclass  (spec §5)
# ---------------------------------------------------------------------------

@dataclass
class TolanskyResult:
    """
    Output of the Tolansky two-line analysis (S13a).
    All two_sigma_ fields are exactly 2 x sigma_ (S04 convention).
    Focal length f is not reported; alpha is the sole plate-scale parameter.
    """
    # --- Family assignment provenance ---
    n_peaks_total:     int
    n_nan_dropped:     int
    n_rings_a:         int
    n_rings_b:         int
    amp_threshold:     float
    Y_B_obs:           float

    # --- Line a  (lam_a = 640.2248 nm) ---
    Delta_a:           float        # [Eq. 3.85]
    sigma_Delta_a:     float
    two_sigma_Delta_a: float
    eps_a:             float        # [Eq. 3.86]
    sigma_eps_a:       float
    two_sigma_eps_a:   float
    chi2_dof_a:        float
    delta_a:           np.ndarray   # successive r^2-differences (px^2)
    r2_a:              np.ndarray   # r^2 values used in fit (for printing)

    # --- Line b  (lam_b = 638.2991 nm) ---
    Delta_b:           float        # [Eq. 3.87]
    sigma_Delta_b:     float
    two_sigma_Delta_b: float
    eps_b:             float        # [Eq. 3.88]
    sigma_eps_b:       float
    two_sigma_eps_b:   float
    chi2_dof_b:        float
    delta_b:           np.ndarray
    r2_b:              np.ndarray

    # --- Consistency check ---
    Delta_ratio_obs:      float
    Delta_ratio_expected: float
    Delta_ratio_residual: float     # |obs - expected| / expected

    # --- Integer disambiguation ---
    N_Delta: int                    # N_Delta = na - nb  [Eq. 3.96 / Benoit]

    # --- Plate spacing recovery  [Eq. 3.97] ---
    d_m:           float
    sigma_d_m:     float
    two_sigma_d_m: float

    # --- Plate scale alpha (primary geometric output) ---
    alpha_a:           float        # alpha from line a  (rad/px)  [from 3.85]
    alpha_b:           float        # alpha from line b  (rad/px)  cross-check
    alpha_mean:        float        # mean of alpha_a and alpha_b  (rad/px)
    sigma_alpha:       float
    two_sigma_alpha:   float
    alpha_consistency: float        # |alpha_a - alpha_b| / alpha_a

    # --- Wavelengths (provenance) ---
    lam_a_nm: float                 # 640.2248
    lam_b_nm: float                 # 638.2991


# ---------------------------------------------------------------------------
# Task 2 -- input loading and family assignment
# ---------------------------------------------------------------------------

def load_and_split_families(path_or_array):
    """
    Load _peak_fits.npy (or accept a pre-loaded array) and split into the
    two neon-line families by median-amplitude threshold.

    Returns
    -------
    p_a, r2_a, sigma_r2_a : arrays for line a (brighter, 640.2248 nm)
    p_b, r2_b, sigma_r2_b : arrays for line b (dimmer,   638.2991 nm)
    amp_threshold, Y_B_obs, n_nan_dropped, n_peaks_total
    """
    if isinstance(path_or_array, (str, pathlib.Path)):
        peaks = np.load(str(path_or_array))
    else:
        peaks = np.asarray(path_or_array, dtype=float)

    if peaks.ndim != 2 or peaks.shape[1] != 9:
        raise ValueError(
            f"Expected shape (N, 9), got {peaks.shape}.  "
            "Columns: peak_num | r_raw_px | r_fit_px | sigma_r_fit_px | "
            "r_fit_sq | sigma_r_fit_sq | amplitude_adu | width_px | reduced_chi2"
        )

    n_peaks_total = peaks.shape[0]

    valid = np.isfinite(peaks[:, 2])
    n_nan_dropped = int((~valid).sum())
    peaks_ok = peaks[valid]

    if peaks_ok.shape[0] < 4:
        raise InsufficientRingsError(
            f"Only {peaks_ok.shape[0]} valid fitted peaks; "
            "need >= 4 (>=2 per family)."
        )

    amps = peaks_ok[:, 6]
    amp_threshold = float(np.median(amps))

    mask_a = amps > amp_threshold    # 640.2248 nm -- brighter
    mask_b = amps <= amp_threshold   # 638.2991 nm -- dimmer

    peaks_a = peaks_ok[mask_a]
    peaks_b = peaks_ok[mask_b]

    if peaks_a.shape[0] < 2 or peaks_b.shape[0] < 2:
        raise InsufficientRingsError(
            "Family split by amplitude produced < 2 rings in one family.  "
            "Check peak detection parameters."
        )

    Y_B_obs = float(np.median(amps[mask_b]) / np.median(amps[mask_a]))
    if Y_B_obs < 0.15 or Y_B_obs > 0.60:
        warnings.warn(
            f"Y_B_obs = {Y_B_obs:.3f} is outside the expected range [0.15, 0.60].  "
            "Check exposure, dark subtraction, or family assignment.",
            RuntimeWarning,
            stacklevel=3,
        )

    p_a = np.arange(1, peaks_a.shape[0] + 1, dtype=float)
    p_b = np.arange(1, peaks_b.shape[0] + 1, dtype=float)

    r2_a       = peaks_a[:, 4]
    sigma_r2_a = peaks_a[:, 5]
    r2_b       = peaks_b[:, 4]
    sigma_r2_b = peaks_b[:, 5]

    return (p_a, r2_a, sigma_r2_a,
            p_b, r2_b, sigma_r2_b,
            amp_threshold, Y_B_obs, n_nan_dropped, n_peaks_total)


# ---------------------------------------------------------------------------
# Task 3 -- WLS helper  (spec §4 Step 4, equations given verbatim)
# ---------------------------------------------------------------------------

def _wls(p, r2, sigma_r2):
    """Weighted least-squares fit r^2 = S*p + b (spec §4 Step 4)."""
    w        = 1.0 / sigma_r2 ** 2
    sum_w    = np.sum(w)
    sum_wp   = np.sum(w * p)
    sum_wp2  = np.sum(w * p ** 2)
    sum_wr2  = np.sum(w * r2)
    sum_wpr2 = np.sum(w * p * r2)
    Lambda   = sum_w * sum_wp2 - sum_wp ** 2
    S        = (sum_w * sum_wpr2 - sum_wp * sum_wr2) / Lambda
    b        = (sum_wp2 * sum_wr2 - sum_wp * sum_wpr2) / Lambda
    var_S    = sum_w  / Lambda
    var_b    = sum_wp2 / Lambda
    eps      = 1.0 + b / S
    sig_eps  = np.sqrt((np.sqrt(var_b) / S) ** 2
                       + (b * np.sqrt(var_S) / S ** 2) ** 2)
    chi2_dof = np.sum(w * (r2 - S * p - b) ** 2) / max(len(p) - 2, 1)
    return dict(
        Delta=S,           sigma_Delta=np.sqrt(var_S),
        eps=eps % 1.0,     sigma_eps=sig_eps,
        chi2_dof=chi2_dof,
        intercept=b,       sigma_intercept=np.sqrt(var_b),
    )


# ---------------------------------------------------------------------------
# Task 4 -- Benoit d recovery  (Vaughan Eq. 3.97)
# ---------------------------------------------------------------------------

def benoit_d(eps_a, sigma_eps_a, eps_b, sigma_eps_b,
             lam_a_m, lam_b_m, d_prior_m, n_air=1.0):
    """
    Recover etalon plate spacing d from the two fractional orders.

    d = (N_Delta + eps_a - eps_b) * lam_a * lam_b / (2*n_air*(lam_b - lam_a))

    N_Delta is resolved from d_prior_m and fixes only the FSR-period ambiguity.

    Returns (d_m, sigma_d_m, N_Delta).
    """
    N_Delta = int(round(2.0 * d_prior_m * (1.0 / lam_a_m - 1.0 / lam_b_m)))

    d = ((N_Delta + eps_a - eps_b)
         * lam_a_m * lam_b_m
         / (2.0 * n_air * (lam_b_m - lam_a_m)))
    d = abs(d)

    factor  = lam_a_m * lam_b_m / (2.0 * n_air * abs(lam_b_m - lam_a_m))
    sigma_d = factor * np.sqrt(sigma_eps_a ** 2 + sigma_eps_b ** 2)

    return d, sigma_d, N_Delta


# ---------------------------------------------------------------------------
# Task 5 -- alpha recovery  (from Vaughan Eq. 3.85)
# ---------------------------------------------------------------------------

def recover_alpha(Delta_a, sigma_Delta_a, Delta_b, sigma_Delta_b,
                  d_m, sigma_d_m, lam_a_m, lam_b_m, n_air=1.0):
    """
    Recover angular pixel scale alpha (rad/px).

    alpha = sqrt(lam * n_air / (d * Delta))   [from Vaughan Eq. 3.85]

    Returns (alpha_a, alpha_b, alpha_mean, sigma_alpha, alpha_consistency).
    No focal length is computed or returned.
    """
    alpha_a = np.sqrt(lam_a_m * n_air / (d_m * Delta_a))
    alpha_b = np.sqrt(lam_b_m * n_air / (d_m * Delta_b))
    alpha_mean = 0.5 * (alpha_a + alpha_b)

    sig_alpha = 0.5 * alpha_a * np.sqrt(
        (sigma_Delta_a / Delta_a) ** 2 + (sigma_d_m / d_m) ** 2
    )
    alpha_consistency = abs(alpha_a - alpha_b) / alpha_a

    return alpha_a, alpha_b, alpha_mean, sig_alpha, alpha_consistency


# ---------------------------------------------------------------------------
# Task 6 -- top-level run_tolansky()
# ---------------------------------------------------------------------------

def run_tolansky(
    peaks_input,
    lam_a_m:   float = 640.2248e-9,
    lam_b_m:   float = 638.2991e-9,
    d_prior_m: float = 20.008e-3,
    n_air:     float = 1.0,
) -> TolanskyResult:
    """
    Run the full S13a two-line Tolansky analysis on a neon calibration image.

    Parameters
    ----------
    peaks_input : ndarray or path
        (N, 9) float64 array from annular_reduction.py, or path to the
        ``{stem}_peak_fits.npy`` file it writes.
    lam_a_m : float
        Brighter-line wavelength [m].  Default: 640.2248 nm.
    lam_b_m : float
        Dimmer-line wavelength [m].    Default: 638.2991 nm.
    d_prior_m : float
        Prior plate spacing [m]; used only to resolve N_Delta.
        Default: 20.008 mm (ICOS build report).
    n_air : float
        Refractive index of etalon gap.  Default: 1.0.

    Returns
    -------
    TolanskyResult
    """
    (p_a, r2_a, sigma_r2_a,
     p_b, r2_b, sigma_r2_b,
     amp_threshold, Y_B_obs,
     n_nan_dropped, n_peaks_total) = load_and_split_families(peaks_input)

    fit_a = _wls(p_a, r2_a, sigma_r2_a)
    fit_b = _wls(p_b, r2_b, sigma_r2_b)

    Delta_a      = fit_a["Delta"]
    sigma_Delta_a = fit_a["sigma_Delta"]
    eps_a        = fit_a["eps"]
    sigma_eps_a  = fit_a["sigma_eps"]
    chi2_dof_a   = fit_a["chi2_dof"]

    Delta_b      = fit_b["Delta"]
    sigma_Delta_b = fit_b["sigma_Delta"]
    eps_b        = fit_b["eps"]
    sigma_eps_b  = fit_b["sigma_eps"]
    chi2_dof_b   = fit_b["chi2_dof"]

    delta_a = np.diff(r2_a)
    delta_b = np.diff(r2_b)

    Delta_ratio_obs      = Delta_a / Delta_b
    Delta_ratio_expected = lam_a_m / lam_b_m
    Delta_ratio_residual = (abs(Delta_ratio_obs - Delta_ratio_expected)
                            / Delta_ratio_expected)
    if Delta_ratio_residual > 0.002:
        warnings.warn(
            f"Delta_a/Delta_b = {Delta_ratio_obs:.6f} deviates from "
            f"lam_a/lam_b = {Delta_ratio_expected:.6f} by "
            f"{Delta_ratio_residual*1e6:.0f} ppm (>200 ppm).  "
            "Check family assignment.",
            RuntimeWarning,
            stacklevel=2,
        )

    d_m, sigma_d_m, N_Delta = benoit_d(
        eps_a, sigma_eps_a, eps_b, sigma_eps_b,
        lam_a_m, lam_b_m, d_prior_m, n_air,
    )

    (alpha_a, alpha_b, alpha_mean,
     sigma_alpha, alpha_consistency) = recover_alpha(
        Delta_a, sigma_Delta_a, Delta_b, sigma_Delta_b,
        d_m, sigma_d_m, lam_a_m, lam_b_m, n_air,
    )

    if alpha_consistency > 0.001:
        warnings.warn(
            f"|alpha_a - alpha_b| / alpha_a = "
            f"{alpha_consistency*1e6:.0f} ppm (>1000 ppm).  "
            "Check family assignment or ring quality.",
            RuntimeWarning,
            stacklevel=2,
        )

    return TolanskyResult(
        n_peaks_total=n_peaks_total,
        n_nan_dropped=n_nan_dropped,
        n_rings_a=len(p_a),
        n_rings_b=len(p_b),
        amp_threshold=amp_threshold,
        Y_B_obs=Y_B_obs,

        Delta_a=Delta_a,
        sigma_Delta_a=sigma_Delta_a,
        two_sigma_Delta_a=2.0 * sigma_Delta_a,
        eps_a=eps_a,
        sigma_eps_a=sigma_eps_a,
        two_sigma_eps_a=2.0 * sigma_eps_a,
        chi2_dof_a=chi2_dof_a,
        delta_a=delta_a,
        r2_a=r2_a.copy(),

        Delta_b=Delta_b,
        sigma_Delta_b=sigma_Delta_b,
        two_sigma_Delta_b=2.0 * sigma_Delta_b,
        eps_b=eps_b,
        sigma_eps_b=sigma_eps_b,
        two_sigma_eps_b=2.0 * sigma_eps_b,
        chi2_dof_b=chi2_dof_b,
        delta_b=delta_b,
        r2_b=r2_b.copy(),

        Delta_ratio_obs=Delta_ratio_obs,
        Delta_ratio_expected=Delta_ratio_expected,
        Delta_ratio_residual=Delta_ratio_residual,

        N_Delta=N_Delta,

        d_m=d_m,
        sigma_d_m=sigma_d_m,
        two_sigma_d_m=2.0 * sigma_d_m,

        alpha_a=alpha_a,
        alpha_b=alpha_b,
        alpha_mean=alpha_mean,
        sigma_alpha=sigma_alpha,
        two_sigma_alpha=2.0 * sigma_alpha,
        alpha_consistency=alpha_consistency,

        lam_a_nm=lam_a_m * 1e9,
        lam_b_nm=lam_b_m * 1e9,
    )


# ---------------------------------------------------------------------------
# Task 7 -- rectangular array table  (spec §6)
# ---------------------------------------------------------------------------

def print_rectangular_array(result: TolanskyResult) -> None:
    """Print the Vaughan (1989) Table 3.1 analog and Benoit summary."""
    r = result
    sep = "=" * 68

    print(f"\n{sep}")
    print("=== TOLANSKY RECTANGULAR ARRAY  (Vaughan 1989, Table 3.1 analog) ===")
    print(sep)

    yb_status = "PASS" if 0.15 <= r.Y_B_obs <= 0.60 else "WARN"
    print(f"\nFamily assignment:  amp_threshold = {r.amp_threshold:.1f} ADU"
          f"   Y_B_obs = {r.Y_B_obs:.3f}  [{yb_status}]")
    print(f"  640.2248 nm (line a):  N rings = {r.n_rings_a:2d}")
    print(f"  638.2991 nm (line b):  N rings = {r.n_rings_b:2d}")

    for lbl, r2_arr, delta_arr, Delta, sDelta, eps, seps, chi2 in [
        ("a", r.r2_a, r.delta_a,
         r.Delta_a, r.sigma_Delta_a, r.eps_a, r.sigma_eps_a, r.chi2_dof_a),
        ("b", r.r2_b, r.delta_b,
         r.Delta_b, r.sigma_Delta_b, r.eps_b, r.sigma_eps_b, r.chi2_dof_b),
    ]:
        lam_str = "640.2248" if lbl == "a" else "638.2991"
        sub     = "a" if lbl == "a" else "b"
        print(f"\nComponent {lbl}  (lam_{sub} = {lam_str} nm)")

        n = len(r2_arr)
        p_strs  = "".join(f"  {i+1:>10}" for i in range(n))
        r2_strs = "".join(f"  {v:>10.2f}" for v in r2_arr)
        print(f"  p   :{p_strs}")
        print(f"  r^2 :{r2_strs}  (px^2)")

        if len(delta_arr) > 0:
            d_strs = "".join(
                f"  d{i+1}{i+2}={delta_arr[i]:.1f}"
                for i in range(len(delta_arr))
            )
            print(f"        {d_strs}")

        print(f"  Delta_{sub} (WLS slope) = {Delta:.2f} +/- {sDelta:.2f} px^2"
              f"   chi2_dof = {chi2:.2f}")
        print(f"  eps_{sub}               = {eps:.4f}   +/- {seps:.4f}")

    ratio_ppm = r.Delta_ratio_residual * 1e6
    ratio_ok  = "PASS" if ratio_ppm < 200 else "WARN"
    print(f"\nRatio  Delta_a/Delta_b observed = {r.Delta_ratio_obs:.6f}"
          f"   expected (lam_a/lam_b) = {r.Delta_ratio_expected:.6f}")
    print(f"        residual = {ratio_ppm:.1f} ppm"
          f"   [{ratio_ok} if < 200 ppm]")

    print(f"\n{sep}")
    print("=== BENOIT RECOVERY  (Vaughan Eqs. 3.94-3.97) ===")
    print(f"  N_Delta = na - nb = {r.N_Delta}"
          f"   [from d_prior = 20.008 mm, Eq. 3.96]")
    d_mm   = r.d_m * 1e3
    sd_mm  = r.sigma_d_m * 1e3
    sd2_mm = r.two_sigma_d_m * 1e3
    print(f"  d   = {d_mm:.4f} +/- {sd_mm:.4f} mm   (2sigma = {sd2_mm:.4f} mm)")

    print(f"\n{sep}")
    print("=== PLATE SCALE ===")
    alpha_ok = "PASS" if r.alpha_consistency < 0.001 else "WARN"
    print(f"  alpha_a  = {r.alpha_a:.4e} +/- {r.sigma_alpha:.4e} rad/px"
          f"   [from Delta_a, d, lam_a]")
    print(f"  alpha_b  = {r.alpha_b:.4e} rad/px"
          f"   [from Delta_b, d, lam_b; cross-check]")
    print(f"  alpha_mean = {r.alpha_mean:.4e} +/- {r.sigma_alpha:.4e} rad/px"
          f"   (2sigma = {r.two_sigma_alpha:.4e})")
    print(f"  |alpha_a - alpha_b| / alpha_a = {r.alpha_consistency*1e6:.1f} ppm"
          f"   [{alpha_ok} if < 1000 ppm]")
    print(sep)


# ---------------------------------------------------------------------------
# Task 8 -- M05 priors handoff  (spec §7)
# ---------------------------------------------------------------------------

def to_m05_priors(result: TolanskyResult) -> dict:
    """Convert TolanskyResult to the prior dict expected by M05 FitConfig."""
    d_mm     = result.d_m * 1e3
    sig_d_mm = result.sigma_d_m * 1e3
    return {
        "t_init_mm":    d_mm,
        "t_bounds_mm":  (d_mm - 3 * sig_d_mm, d_mm + 3 * sig_d_mm),
        "alpha_init":   result.alpha_mean,
        "alpha_bounds": (result.alpha_mean * 0.875,
                         result.alpha_mean * 1.125),
        "epsilon_cal_a": result.eps_a,
        "epsilon_cal_b": result.eps_b,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import tkinter as tk
    from tkinter import filedialog

    if len(sys.argv) > 1:
        peaks_path = pathlib.Path(sys.argv[1])
    else:
        root = tk.Tk()
        root.withdraw()
        p = filedialog.askopenfilename(
            title="Select *_peak_fits.npy",
            filetypes=[("NumPy array", "*.npy"), ("All files", "*.*")],
        )
        root.destroy()
        if not p:
            print("No file selected -- exiting.")
            sys.exit(0)
        peaks_path = pathlib.Path(p)

    result = run_tolansky(peaks_path)
    print_rectangular_array(result)

    priors = to_m05_priors(result)
    print("\nM05 priors:")
    for k, v in priors.items():
        print(f"  {k}: {v}")
