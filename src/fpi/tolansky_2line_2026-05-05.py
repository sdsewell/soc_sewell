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
    r2_a:              np.ndarray   # r^2 values used in fit (for plotting)
    sigma_r2_a:        np.ndarray   # 1σ uncertainty on r^2, line a

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
    sigma_r2_b:        np.ndarray   # 1σ uncertainty on r^2, line b

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
        sigma_r2_a=sigma_r2_a.copy(),

        Delta_b=Delta_b,
        sigma_Delta_b=sigma_Delta_b,
        two_sigma_Delta_b=2.0 * sigma_Delta_b,
        eps_b=eps_b,
        sigma_eps_b=sigma_eps_b,
        two_sigma_eps_b=2.0 * sigma_eps_b,
        chi2_dof_b=chi2_dof_b,
        delta_b=delta_b,
        r2_b=r2_b.copy(),
        sigma_r2_b=sigma_r2_b.copy(),

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
# Task 9 -- diagnostic figure
# ---------------------------------------------------------------------------

def plot_tolansky_result(
    result: TolanskyResult,
    save_path=None,
):
    """
    Four-panel diagnostic figure for the S13a two-line Tolansky analysis.

    A) Joint Tolansky plot — r² vs p for both neon families with individual
       WLS fit lines.  Confirms linearity and correct slope ratio.

    B) WLS fit residuals for both families.  Random scatter with no trend
       confirms good ring detection and correct family assignment.

    C) Successive Δ(r²) for both families with dashed mean-slope references.
       CV < 2 % confirms etalon parallelism.

    D) Summary text — recovered d, alpha, ε_a, ε_b, N_Δ, χ²/ν.

    Parameters
    ----------
    result   : TolanskyResult from run_tolansky()
    save_path: optional file path; if given the figure is saved at dpi=150.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    r = result
    u2 = "px²"

    BLUE   = "tab:blue"
    ORANGE = "tab:orange"
    GREEN  = "tab:green"
    RED    = "tab:red"
    GRAY   = "gray"
    BLACK  = "black"

    fig = plt.figure(figsize=(14, 10), facecolor="white")
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            hspace=0.44, wspace=0.37,
                            left=0.09, right=0.97,
                            top=0.91, bottom=0.08)
    ax_tol = fig.add_subplot(gs[0, 0])
    ax_res = fig.add_subplot(gs[1, 0])
    ax_dr2 = fig.add_subplot(gs[0, 1])
    ax_txt = fig.add_subplot(gs[1, 1])

    for ax in [ax_tol, ax_res, ax_dr2, ax_txt]:
        ax.set_facecolor("white")
        ax.tick_params(colors=BLACK, which="both", direction="in")
        for sp in ax.spines.values():
            sp.set_edgecolor(BLACK)
        ax.xaxis.label.set_color(BLACK)
        ax.yaxis.label.set_color(BLACK)
        ax.title.set_color(BLACK)

    p_a = np.arange(1, r.n_rings_a + 1, dtype=float)
    p_b = np.arange(1, r.n_rings_b + 1, dtype=float)
    int_a = r.Delta_a * (r.eps_a - 1.0)
    int_b = r.Delta_b * (r.eps_b - 1.0)
    p_all  = np.concatenate([p_a, p_b])
    # Extend fit lines from p=0 so x-intercepts (where r²=0) are visible
    p_fine = np.linspace(0.0, p_all.max() + 0.3, 300)
    fit_a  = r.Delta_a * p_fine + int_a
    fit_b  = r.Delta_b * p_fine + int_b

    # x-intercepts: r²=0 → p = 1 − ε
    p_int_a = 1.0 - r.eps_a
    p_int_b = 1.0 - r.eps_b
    data_max = max(r.r2_a.max(), r.r2_b.max()) * 1.05
    y_min_tol = -0.05 * data_max   # small negative margin to show intercepts

    # ── A: Joint Tolansky plot ────────────────────────────────────────────────
    ax_tol.axhline(0, color=GRAY, lw=0.7, ls=":", zorder=0)
    ax_tol.errorbar(p_a, r.r2_a, yerr=r.sigma_r2_a,
                    fmt="o", color=BLUE, ecolor=GRAY, capsize=4, ms=6,
                    lw=1.4, zorder=3,
                    label=f"λ_a = {r.lam_a_nm:.4f} nm  (n={r.n_rings_a})")
    ax_tol.errorbar(p_b, r.r2_b, yerr=r.sigma_r2_b,
                    fmt="s", color=ORANGE, ecolor=GRAY, capsize=4, ms=6,
                    lw=1.4, zorder=3,
                    label=f"λ_b = {r.lam_b_nm:.4f} nm  (n={r.n_rings_b})")
    ax_tol.plot(p_fine, fit_a, color=BLUE,   lw=1.8, ls="-",  zorder=2,
                label=f"Fit a:  Δ_a = {r.Delta_a:.4g}")
    ax_tol.plot(p_fine, fit_b, color=ORANGE, lw=1.8, ls="--", zorder=2,
                label=f"Fit b:  Δ_b = {r.Delta_b:.4g}")
    # Mark r²=0 intercepts with vertical tick marks and labels
    ax_tol.scatter([p_int_a], [0], color=BLUE,   s=60, zorder=5,
                   marker="|", linewidths=2.0)
    ax_tol.scatter([p_int_b], [0], color=ORANGE, s=60, zorder=5,
                   marker="|", linewidths=2.0)
    ax_tol.text(p_int_a, y_min_tol * 0.75,
                f"1−ε_a={p_int_a:.3f}", ha="center", va="top",
                fontsize=8, color=BLUE, fontfamily="monospace")
    ax_tol.text(p_int_b, y_min_tol * 0.75,
                f"1−ε_b={p_int_b:.3f}", ha="center", va="top",
                fontsize=8, color=ORANGE, fontfamily="monospace")
    ax_tol.set_xlim(-0.2, p_all.max() + 0.5)
    ax_tol.set_ylim(y_min_tol, data_max)
    ax_tol.set_xticks(range(0, int(p_all.max()) + 1))
    ax_tol.set_xlabel("Fringe index  $p$", fontsize=11)
    ax_tol.set_ylabel(f"$r^2$  [{u2}]", fontsize=11)
    ax_tol.set_title("A — Tolansky Plot  (both neon lines)",
                      fontsize=11, fontweight="bold", pad=7)
    ax_tol.legend(fontsize=8, facecolor="white", labelcolor=BLACK,
                  edgecolor=BLACK, framealpha=0.9, ncol=2)
    ax_tol.text(0.97, 0.05,
                f"$\\chi^2/\\nu$:  a={r.chi2_dof_a:.3f},  b={r.chi2_dof_b:.3f}",
                transform=ax_tol.transAxes,
                ha="right", va="bottom", fontsize=8.5, color=GREEN)

    # ── B: Residuals ──────────────────────────────────────────────────────────
    resid_a = r.r2_a - (r.Delta_a * p_a + int_a)
    resid_b = r.r2_b - (r.Delta_b * p_b + int_b)
    ax_res.axhline(0, color=GRAY, lw=1.0, ls="--", zorder=1)
    ax_res.errorbar(p_a, resid_a, yerr=r.sigma_r2_a,
                    fmt="o", color=BLUE, ecolor=GRAY, capsize=4, ms=6,
                    lw=1.4, zorder=3, label="Line a")
    ax_res.errorbar(p_b, resid_b, yerr=r.sigma_r2_b,
                    fmt="s", color=ORANGE, ecolor=GRAY, capsize=4, ms=6,
                    lw=1.4, zorder=3, label="Line b")
    ax_res.set_xlabel("Fringe index  $p$", fontsize=11)
    ax_res.set_ylabel(f"Residual  [{u2}]", fontsize=11)
    ax_res.set_title("B — WLS Residuals",
                      fontsize=11, fontweight="bold", pad=7)
    ax_res.legend(fontsize=9, facecolor="white", labelcolor=BLACK,
                  edgecolor=BLACK, framealpha=0.9)

    # ── C: Successive Δ(r²) ──────────────────────────────────────────────────
    p_mid_a = 0.5 * (p_a[:-1] + p_a[1:])
    p_mid_b = 0.5 * (p_b[:-1] + p_b[1:])
    sdelta_a = np.sqrt(r.sigma_r2_a[1:] ** 2 + r.sigma_r2_a[:-1] ** 2)
    sdelta_b = np.sqrt(r.sigma_r2_b[1:] ** 2 + r.sigma_r2_b[:-1] ** 2)

    cv_a = (r.delta_a.std() / abs(r.delta_a.mean()) * 100
            if r.delta_a.mean() != 0 and len(r.delta_a) > 1 else np.nan)
    cv_b = (r.delta_b.std() / abs(r.delta_b.mean()) * 100
            if r.delta_b.mean() != 0 and len(r.delta_b) > 1 else np.nan)

    ax_dr2.axhline(r.Delta_a, color=BLUE,   lw=1.2, ls="--",
                   label=f"Δ_a = {r.Delta_a:.4g} (WLS slope)")
    ax_dr2.axhline(r.Delta_b, color=ORANGE, lw=1.2, ls=":",
                   label=f"Δ_b = {r.Delta_b:.4g} (WLS slope)")
    if len(r.delta_a) > 0:
        ax_dr2.errorbar(p_mid_a, r.delta_a, yerr=sdelta_a,
                        fmt="o", color=BLUE, ecolor=GRAY,
                        capsize=4, ms=6, lw=1.4, zorder=3)
    if len(r.delta_b) > 0:
        ax_dr2.errorbar(p_mid_b, r.delta_b, yerr=sdelta_b,
                        fmt="s", color=ORANGE, ecolor=GRAY,
                        capsize=4, ms=6, lw=1.4, zorder=3)
    ax_dr2.set_xlabel("Fringe index  $p$  (midpoint)", fontsize=11)
    ax_dr2.set_ylabel(f"$\\Delta(r^2)$  [{u2}]", fontsize=11)
    ax_dr2.set_title("C — Successive  $\\Delta(r^2)$",
                      fontsize=11, fontweight="bold", pad=7)
    ax_dr2.legend(fontsize=8.5, facecolor="white", labelcolor=BLACK,
                  edgecolor=BLACK, framealpha=0.9)
    cv_col = GREEN if max(cv_a, cv_b) < 2 else ("goldenrod" if max(cv_a, cv_b) < 5 else RED)
    ax_dr2.text(0.97, 0.07,
                f"CV_a = {cv_a:.1f}%   CV_b = {cv_b:.1f}%",
                transform=ax_dr2.transAxes,
                ha="right", va="bottom", fontsize=8.5, color=cv_col)

    # ── D: Summary ────────────────────────────────────────────────────────────
    ax_txt.axis("off")

    d_mm      = r.d_m * 1e3
    sig_d_mm  = r.sigma_d_m * 1e3
    sig2_d_mm = r.two_sigma_d_m * 1e3
    ratio_ppm = r.Delta_ratio_residual * 1e6
    ratio_col = GREEN if ratio_ppm < 200 else RED
    yb_col    = GREEN if 0.15 <= r.Y_B_obs <= 0.60 else RED
    chi_col_a = GREEN if r.chi2_dof_a < 2 else "goldenrod"
    chi_col_b = GREEN if r.chi2_dof_b < 2 else "goldenrod"

    # Each entry: (text, color, size, weight, fontstyle, extra_gap_before)
    # extra_gap_before adds a small visual separator before section headers
    lines_txt = [
        ("TWO-LINE TOLANSKY SUMMARY",                                  BLACK,   10.5, "bold",   "normal", 0.00),
        ("── Family assignment ─────────────────────────────────────",  GRAY,    8.0,  "normal", "normal", 0.01),
        (f"  N rings: {r.n_peaks_total} total"
         f"  ({r.n_rings_a} line a + {r.n_rings_b} line b)"
         f"   NaN dropped: {r.n_nan_dropped}",                         GRAY,    8.5,  "normal", "normal", 0.00),
        (f"  Y_B_obs = {r.Y_B_obs:.3f}"
         f"   {'[PASS]' if 0.15<=r.Y_B_obs<=0.60 else '[WARN]'}"
         f"   amp_threshold = {r.amp_threshold:.0f} ADU",              yb_col,  8.5,  "normal", "normal", 0.00),
        ("── WLS fit results ───────────────────────────────────────",  GRAY,    8.0,  "normal", "normal", 0.01),
        (f"  Δ_a = {r.Delta_a:.5g} ± {r.sigma_Delta_a:.3g} px²/fr"
         f"   ε_a = {r.eps_a:.6f} ± {r.sigma_eps_a:.2g}"
         f"   χ²/ν = {r.chi2_dof_a:.3f}",                             BLUE,    8.5,  "normal", "normal", 0.00),
        (f"  Δ_b = {r.Delta_b:.5g} ± {r.sigma_Delta_b:.3g} px²/fr"
         f"   ε_b = {r.eps_b:.6f} ± {r.sigma_eps_b:.2g}"
         f"   χ²/ν = {r.chi2_dof_b:.3f}",                             ORANGE,  8.5,  "normal", "normal", 0.00),
        (f"  Δ_a/Δ_b = {r.Delta_ratio_obs:.6f}"
         f"   λ_a/λ_b = {r.Delta_ratio_expected:.6f}"
         f"   Δ = {ratio_ppm:.0f} ppm"
         f"   {'[PASS]' if ratio_ppm<200 else '[WARN]'}",             ratio_col, 8.5, "normal", "normal", 0.00),
        ("── Benoit recovery ───────────────────────────────────────",  GRAY,    8.0,  "normal", "normal", 0.01),
        ("  d = (N_Δ + ε_a−ε_b)·λ_a·λ_b / [2·n·(λ_b−λ_a)]",          GRAY,    8.0,  "normal", "italic", 0.00),
        (f"  N_Δ = {r.N_Delta}"
         f"   ε_a − ε_b = {r.eps_a - r.eps_b:+.6f}",                 BLACK,   8.5,  "normal", "normal", 0.00),
        (f"  d  = {d_mm:.5f} ± {sig_d_mm:.4f} mm"
         f"   (2σ: ±{sig2_d_mm:.4f} mm)",                             GREEN,   9.5,  "bold",   "normal", 0.00),
        ("── Plate scale ───────────────────────────────────────────",  GRAY,    8.0,  "normal", "normal", 0.01),
        ("  α_a = √(λ_a·n_air / (d·Δ_a))",                             GRAY,    8.0,  "normal", "italic", 0.00),
        (f"  α_a = {r.alpha_a:.4e} rad/px"
         f"   α_b = {r.alpha_b:.4e} rad/px",                          BLACK,   8.5,  "normal", "normal", 0.00),
        (f"  α   = {r.alpha_mean:.4e} ± {r.sigma_alpha:.2e} rad/px"
         f"   (2σ: ±{r.two_sigma_alpha:.2e})",                        "purple", 9.5, "bold",   "normal", 0.00),
        (f"  consistency: {r.alpha_consistency*1e6:.1f} ppm"
         f"   {'[PASS]' if r.alpha_consistency<0.001 else '[WARN]'}",
         GREEN if r.alpha_consistency < 0.001 else RED, 8.5, "normal", "normal", 0.00),
    ]

    y = 0.98
    for text, color, size, weight, style, gap in lines_txt:
        y -= gap
        ax_txt.text(0.02, y, text, transform=ax_txt.transAxes,
                    ha="left", va="top", fontsize=size,
                    color=color, fontweight=weight, fontstyle=style,
                    fontfamily="monospace")
        y -= size * 0.010 + 0.005

    fig.suptitle(
        "Tolansky Two-Line Analysis  "
        f"(λ_a = {r.lam_a_nm:.4f} nm,  λ_b = {r.lam_b_nm:.4f} nm)",
        color=BLACK, fontsize=13, fontweight="bold", y=0.97,
    )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  Figure saved → {save_path}")
    return fig


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
