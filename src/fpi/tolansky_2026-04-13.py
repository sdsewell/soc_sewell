"""
Module:      tolansky_2026-04-13.py
Spec:        docs/specs/S13_tolansky_analysis_2026-04-13.md
Reference:   Vaughan (1989) The Fabry-Perot Interferometer, §3.5.2
             Equations (3.83)–(3.97) — rectangular array method
Author:      Claude Code
Project:     WindCube FPI Pipeline — NCAR/HAO
Repo:        soc_sewell
"""

import math
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Output dataclass (S13 §5)
# ---------------------------------------------------------------------------

@dataclass
class TolanskyResult:
    """
    Output of the Tolansky two-line analysis.
    All two_sigma_ fields are exactly 2 × sigma_ (S04 convention).
    """
    # --- Single-line WLS fit results ---
    # Line a  (λₐ = 640.2248 nm)
    Delta_a:         float        # Δₐ = mean r²-step, px²          [Eq. 3.85]
    sigma_Delta_a:   float        # 1σ uncertainty on Δₐ,  px²
    eps_a:           float        # εₐ = fractional order at centre  [Eq. 3.86]
    sigma_eps_a:     float        # 1σ
    chi2_dof_a:      float        # reduced χ² for line-a WLS fit
    delta_a:         np.ndarray   # δₐ_p successive differences  (px²)

    # Line b  (λᵦ = 638.2991 nm)
    Delta_b:         float        # Δᵦ,  px²                         [Eq. 3.87]
    sigma_Delta_b:   float
    eps_b:           float        # εᵦ                               [Eq. 3.88]
    sigma_eps_b:     float
    chi2_dof_b:      float
    delta_b:         np.ndarray

    # --- Consistency check ---
    Delta_ratio_obs:      float   # Δₐ/Δᵦ (observed)
    Delta_ratio_expected: float   # λₐ/λᵦ = 1.003014…
    Delta_ratio_residual: float   # |obs − expected| / expected

    # --- Integer disambiguation ---
    N_Delta:  int                 # N_Δ = nₐ − nᵦ  [Eq. 3.96 / Benoit]

    # --- Plate spacing recovery  [Eq. 3.97] ---
    d_m:             float        # recovered d  (metres)
    sigma_d_m:       float        # 1σ
    two_sigma_d_m:   float        # exactly 2 × sigma_d_m   (S04)

    # --- Focal length and plate scale ---
    f_px:            float        # f  (pixels)
    sigma_f_px:      float
    two_sigma_f_px:  float        # exactly 2 × sigma_f_px  (S04)
    f_b_px:          float        # cross-check from line b
    f_consistency:   float        # |f_a − f_b| / f_a  (accept if < 0.001)

    alpha_rad_px:    float        # α  (rad/px)
    sigma_alpha:     float
    two_sigma_alpha: float        # exactly 2 × sigma_alpha  (S04)

    # --- Inputs for M05 priors ---
    lam_a_nm:  float              # 640.2248
    lam_b_nm:  float              # 638.2991
    n_rings_a: int                # number of rings used
    n_rings_b: int


# ---------------------------------------------------------------------------
# 1c — Single-line WLS normal equations  (S13 §4 Step 4)
# ---------------------------------------------------------------------------

def run_single_line_wls(
    p: np.ndarray,
    r2: np.ndarray,
    sr: np.ndarray,
) -> dict:
    """
    Closed-form WLS fit of model  r²_p = S·p + b  with weights w_p = 1/σ(r²_p)².

    Parameters
    ----------
    p : array-like
        Ring indices (1-based integers as float).
    r2 : array-like
        Measured r²_p values (px²).
    sr : array-like
        1σ radial uncertainties σ(r_p) in pixels.

    Returns
    -------
    dict with keys:
        Delta           : float  — WLS slope S = Δ (px²/ring)
        sigma_Delta     : float  — 1σ on Δ
        eps             : float  — fractional order ε ∈ [0, 1)
        sigma_eps       : float  — 1σ on ε
        chi2_dof        : float  — reduced χ²
        r2_fit          : float  — coefficient of determination R²
        delta           : ndarray — successive differences r²_{p+1} − r²_p
        intercept       : float  — WLS intercept b
        sigma_intercept : float  — 1σ on b
    """
    p   = np.asarray(p,  dtype=float)
    r2  = np.asarray(r2, dtype=float)
    sr  = np.asarray(sr, dtype=float)

    # Propagate σ(r_p) → σ(r²_p) = 2·r_p·σ(r_p)
    r_p      = np.sqrt(r2)
    sigma_r2 = 2.0 * r_p * sr

    # Avoid division by zero
    sigma_r2 = np.where(sigma_r2 > 0, sigma_r2, 1e-30)
    w = 1.0 / sigma_r2 ** 2

    # WLS normal equations
    Sw    = np.sum(w)
    Swp   = np.sum(w * p)
    Swp2  = np.sum(w * p ** 2)
    Swr2  = np.sum(w * r2)
    Swpr2 = np.sum(w * p * r2)

    Lambda = Sw * Swp2 - Swp ** 2
    S = (Sw * Swpr2 - Swp * Swr2) / Lambda
    b = (Swp2 * Swr2 - Swp * Swpr2) / Lambda

    var_S = Sw   / Lambda
    var_b = Swp2 / Lambda

    sigma_S = float(np.sqrt(var_S))
    sigma_b = float(np.sqrt(var_b))

    # Fractional order  ε = 1 + b/S  (from Eq. 3.86 in r² form)
    eps_raw = 1.0 + b / S
    eps = float(eps_raw % 1.0)

    # Propagate σ(ε)
    # ε = 1 + b/S  →  σ²(ε) = (σ_b/S)² + (b·σ_S/S²)²
    sigma_eps = float(
        math.sqrt((sigma_b / S) ** 2 + (b * sigma_S / S ** 2) ** 2)
    )

    # Reduced χ²
    N = len(p)
    resid = r2 - S * p - b
    chi2  = float(np.sum(w * resid ** 2))
    chi2_dof = chi2 / max(N - 2, 1)

    # R² (unweighted, standard coefficient of determination)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((r2 - np.mean(r2)) ** 2))
    r2_fit = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    delta = np.diff(r2)

    return {
        "Delta":           float(S),
        "sigma_Delta":     float(sigma_S),
        "eps":             eps,
        "sigma_eps":       sigma_eps,
        "chi2_dof":        chi2_dof,
        "r2_fit":          r2_fit,
        "delta":           delta,
        "intercept":       float(b),
        "sigma_intercept": float(sigma_b),
    }


# ---------------------------------------------------------------------------
# 1d — Benoit d recovery  (S13 §4 Steps 5–6)
# ---------------------------------------------------------------------------

def benoit_d(
    eps_a: float,
    sigma_eps_a: float,
    eps_b: float,
    sigma_eps_b: float,
    lam_a_m: float,
    lam_b_m: float,
    d_prior_m: float,
    n_air: float = 1.0,
) -> tuple:
    """
    Recover plate spacing d from Benoit exact-fractions method.

    Parameters
    ----------
    eps_a, eps_b      : fractional interference orders for lines a and b
    sigma_eps_a/b     : 1σ uncertainties on εₐ, εᵦ
    lam_a_m, lam_b_m  : wavelengths in metres
    d_prior_m         : prior plate spacing in metres (resolves integer ambiguity)
    n_air             : refractive index of gap (default 1.0)

    Returns
    -------
    (N_Delta, d_m, sigma_d_m) : int, float, float
    """
    # Step 5 — integer disambiguation  [Vaughan Eq. 3.96]
    N_Delta = round(2.0 * d_prior_m * (1.0 / lam_a_m - 1.0 / lam_b_m))

    # Step 6 — plate spacing  [Vaughan Eq. 3.97 / Benoit]
    # lever = λₐ·λᵦ / (2·n_air·(λᵦ − λₐ))   (negative since λᵦ < λₐ)
    lever = lam_a_m * lam_b_m / (2.0 * n_air * (lam_b_m - lam_a_m))
    d_m   = abs(lever * (N_Delta + eps_a - eps_b))

    # Uncertainty propagation (εₐ, εᵦ independent)
    sigma_d_m = abs(lever) * math.sqrt(sigma_eps_a ** 2 + sigma_eps_b ** 2)

    return (N_Delta, d_m, sigma_d_m)


# ---------------------------------------------------------------------------
# 1e — Focal length and plate scale  (S13 §4 Step 7)
# ---------------------------------------------------------------------------

def recover_f_alpha(
    Delta_a: float,
    sigma_Delta_a: float,
    d_m: float,
    sigma_d_m: float,
    lam_a_m: float,
    pixel_pitch_m: float,
    n_air: float = 1.0,
) -> tuple:
    """
    Recover effective focal length f (pixels) and plate scale α (rad/px).

    Parameters
    ----------
    Delta_a        : WLS slope Δₐ (px²)
    sigma_Delta_a  : 1σ on Δₐ (px²)
    d_m            : plate spacing (metres)
    sigma_d_m      : 1σ on d (metres)
    lam_a_m        : wavelength of line a (metres)
    pixel_pitch_m  : physical pixel pitch (metres)
    n_air          : refractive index (default 1.0)

    Returns
    -------
    (f_px, sigma_f_px, alpha_rad_px, sigma_alpha)
    """
    # f = sqrt(Δₐ · n_air · d / λₐ)  [Eq. 3.85 rearranged]
    f_px = math.sqrt(Delta_a * n_air * d_m / lam_a_m)

    # Relative uncertainty σ(f)/f = 0.5·sqrt((σ_Δ/Δ)² + (σ_d/d)²)
    rel_f = 0.5 * math.sqrt((sigma_Delta_a / Delta_a) ** 2
                             + (sigma_d_m   / d_m)     ** 2)
    sigma_f_px = f_px * rel_f

    # Plate scale α = 1 / f_px  (pixel_pitch / f_m = pixel_pitch / (f_px·pixel_pitch) = 1/f_px)
    alpha = 1.0 / f_px
    sigma_alpha = alpha * rel_f          # same relative uncertainty

    return (f_px, sigma_f_px, alpha, sigma_alpha)


# ---------------------------------------------------------------------------
# 1f — Top-level run_tolansky  (S13 §11 Task 4)
# ---------------------------------------------------------------------------

def run_tolansky(
    p_a,
    r2_a,
    sr_a,
    p_b,
    r2_b,
    sr_b,
    lam_a_m:       float = 640.2248e-9,
    lam_b_m:       float = 638.2991e-9,
    d_prior_m:     float = 20.008e-3,
    pixel_pitch_m: float = 32e-6,
    n_air:         float = 1.0,
) -> TolanskyResult:
    """
    Vaughan §3.5.2 two-line Tolansky analysis.

    Parameters
    ----------
    p_a, r2_a, sr_a : ring indices, r²-values (px²), σ(r) (px) for line a
    p_b, r2_b, sr_b : same for line b
    lam_a_m         : wavelength of line a (metres)
    lam_b_m         : wavelength of line b (metres)
    d_prior_m       : prior plate spacing (metres) — only for N_Δ disambiguation
    pixel_pitch_m   : physical pixel pitch (metres)
    n_air           : refractive index of gap

    Returns
    -------
    TolanskyResult
    """
    p_a  = np.asarray(p_a,  dtype=float)
    r2_a = np.asarray(r2_a, dtype=float)
    sr_a = np.asarray(sr_a, dtype=float)
    p_b  = np.asarray(p_b,  dtype=float)
    r2_b = np.asarray(r2_b, dtype=float)
    sr_b = np.asarray(sr_b, dtype=float)

    # Single-line WLS for each family
    wls_a = run_single_line_wls(p_a, r2_a, sr_a)
    wls_b = run_single_line_wls(p_b, r2_b, sr_b)

    Delta_a       = wls_a["Delta"]
    sigma_Delta_a = wls_a["sigma_Delta"]
    eps_a         = wls_a["eps"]
    sigma_eps_a   = wls_a["sigma_eps"]

    Delta_b       = wls_b["Delta"]
    sigma_Delta_b = wls_b["sigma_Delta"]
    eps_b         = wls_b["eps"]
    sigma_eps_b   = wls_b["sigma_eps"]

    # Δ ratio consistency check  [Eq. 3.85 / 3.87]
    Delta_ratio_obs      = Delta_a / Delta_b
    Delta_ratio_expected = lam_a_m / lam_b_m
    Delta_ratio_residual = abs(Delta_ratio_obs - Delta_ratio_expected) / Delta_ratio_expected

    # Benoit d recovery
    N_Delta, d_m, sigma_d_m = benoit_d(
        eps_a, sigma_eps_a,
        eps_b, sigma_eps_b,
        lam_a_m, lam_b_m, d_prior_m, n_air,
    )

    # Focal length and plate scale
    f_px, sigma_f_px, alpha, sigma_alpha = recover_f_alpha(
        Delta_a, sigma_Delta_a,
        d_m, sigma_d_m,
        lam_a_m, pixel_pitch_m, n_air,
    )

    # Cross-check f from line b
    f_b_px = math.sqrt(Delta_b * n_air * d_m / lam_b_m)
    f_consistency = abs(f_px - f_b_px) / f_px

    return TolanskyResult(
        # Line a
        Delta_a         = float(Delta_a),
        sigma_Delta_a   = float(sigma_Delta_a),
        eps_a           = float(eps_a),
        sigma_eps_a     = float(sigma_eps_a),
        chi2_dof_a      = float(wls_a["chi2_dof"]),
        delta_a         = wls_a["delta"],
        # Line b
        Delta_b         = float(Delta_b),
        sigma_Delta_b   = float(sigma_Delta_b),
        eps_b           = float(eps_b),
        sigma_eps_b     = float(sigma_eps_b),
        chi2_dof_b      = float(wls_b["chi2_dof"]),
        delta_b         = wls_b["delta"],
        # Consistency
        Delta_ratio_obs      = float(Delta_ratio_obs),
        Delta_ratio_expected = float(Delta_ratio_expected),
        Delta_ratio_residual = float(Delta_ratio_residual),
        # Integer order
        N_Delta         = int(N_Delta),
        # Plate spacing
        d_m             = float(d_m),
        sigma_d_m       = float(sigma_d_m),
        two_sigma_d_m   = 2.0 * float(sigma_d_m),
        # Focal length
        f_px            = float(f_px),
        sigma_f_px      = float(sigma_f_px),
        two_sigma_f_px  = 2.0 * float(sigma_f_px),
        f_b_px          = float(f_b_px),
        f_consistency   = float(f_consistency),
        # Plate scale
        alpha_rad_px    = float(alpha),
        sigma_alpha     = float(sigma_alpha),
        two_sigma_alpha = 2.0 * float(sigma_alpha),
        # Provenance
        lam_a_nm        = float(lam_a_m * 1e9),
        lam_b_nm        = float(lam_b_m * 1e9),
        n_rings_a       = int(len(p_a)),
        n_rings_b       = int(len(p_b)),
    )


# ---------------------------------------------------------------------------
# 1g — Rectangular array table  (S13 §6)
# ---------------------------------------------------------------------------

def print_rectangular_array(result: TolanskyResult) -> None:
    """
    Print Vaughan Table 3.1 analog to stdout.

    Displays r²_p values and δ_p successive differences for both spectral
    components, ratio check, and Benoit recovery block.
    """
    # We only have delta arrays; reconstruct r² from the WLS for display.
    # The display uses the raw delta arrays, not the WLS fit values.
    # For r² we can't recover individual values from TolanskyResult alone,
    # so we print delta arrays and summary statistics.

    def _component_block(lam_nm: str, delta: np.ndarray,
                         Delta: float, sigma_Delta: float,
                         eps: float, sigma_eps: float) -> None:
        n = len(delta) + 1   # number of rings
        print(f"  p  :", end="")
        for i in range(1, n + 1):
            print(f"  {i:>10d}", end="")
        print()

        # Print delta values between columns
        print(f"  δ  :", end="")
        for i, d in enumerate(delta):
            print(f"       δ{i+1}{i+2}={d:7.2f}", end="")
        print()
        print(f"  Δ (mean δ) = {Delta:8.4f} px²    σ = {sigma_Delta:.4f} px²")
        print(f"  ε          = {eps:.5f}         σ = {sigma_eps:.5f}")

    print()
    print("=== TOLANSKY RECTANGULAR ARRAY (Vaughan 1989, Table 3.1 analog) ===")
    print()

    print(f"Component a  (λₐ = {result.lam_a_nm:.4f} nm)")
    _component_block(
        f"{result.lam_a_nm:.4f} nm",
        result.delta_a,
        result.Delta_a, result.sigma_Delta_a,
        result.eps_a,   result.sigma_eps_a,
    )
    print()

    print(f"Component b  (λᵦ = {result.lam_b_nm:.4f} nm)")
    _component_block(
        f"{result.lam_b_nm:.4f} nm",
        result.delta_b,
        result.Delta_b, result.sigma_Delta_b,
        result.eps_b,   result.sigma_eps_b,
    )
    print()

    residual_ppm = result.Delta_ratio_residual * 1e6
    print(
        f"Ratio  Δₐ/Δᵦ observed = {result.Delta_ratio_obs:.6f}   "
        f"expected (λₐ/λᵦ) = {result.Delta_ratio_expected:.6f}   "
        f"residual = {residual_ppm:.1f} ppm"
    )

    pixel_pitch_m = 32e-6     # fallback; not stored in result
    f_m = result.f_px * pixel_pitch_m * 1e3   # mm
    sigma_f_m = result.sigma_f_px * pixel_pitch_m * 1e3
    two_sigma_f_m = result.two_sigma_f_px * pixel_pitch_m * 1e3

    print()
    print("=== BENOIT RECOVERY (Vaughan Eqs. 3.94–3.97) ===")
    print(f"  N_Δ = nₐ − nᵦ = {result.N_Delta}   "
          "[from d_prior = 20.008 mm, Eq. 3.96]")
    print(
        f"  d   = {result.d_m*1e3:.3f} ± {result.sigma_d_m*1e3:.3f} mm  "
        f"(2σ = {result.two_sigma_d_m*1e3:.3f} mm)"
    )
    print(
        f"  f   = {result.f_px:.1f} ± {result.sigma_f_px:.1f} px  "
        f"(= {f_m:.1f} ± {sigma_f_m:.1f} mm)  "
        f"(2σ = {result.two_sigma_f_px:.1f} px)"
    )
    f_consistency_ppm = result.f_consistency * 1e6
    print(
        f"  f_b = {result.f_b_px:.1f} px  (cross-check)   "
        f"|f_a − f_b|/f_a = {f_consistency_ppm:.1f} ppm"
    )
    alpha_fmt  = f"{result.alpha_rad_px:.4E}"
    sigma_a_fmt = f"{result.sigma_alpha:.4E}"
    two_s_a_fmt = f"{result.two_sigma_alpha:.4E}"
    print(
        f"  α   = {alpha_fmt} ± {sigma_a_fmt} rad/px  "
        f"(2σ = {two_s_a_fmt})"
    )
    print()


# ---------------------------------------------------------------------------
# 1h — M05 priors handoff  (S13 §7)
# ---------------------------------------------------------------------------

def to_m05_priors(result: TolanskyResult) -> dict:
    """
    Convert TolanskyResult to the prior dict expected by M05 FitConfig.
    Direct mapping to S14 fields.
    """
    d_mm     = result.d_m * 1e3
    sig_d_mm = result.sigma_d_m * 1e3
    return {
        "t_init_mm":     d_mm,
        "t_bounds_mm":   (d_mm - 3 * sig_d_mm, d_mm + 3 * sig_d_mm),
        "alpha_init":    result.alpha_rad_px,
        "alpha_bounds":  (result.alpha_rad_px * 0.875,
                          result.alpha_rad_px * 1.125),
        "epsilon_cal_1": result.eps_a,
        "epsilon_cal_2": result.eps_b,
    }


# ---------------------------------------------------------------------------
# __main__ — smoke test on T5 synthetic data
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    lam_a = 640.2248e-9
    lam_b = 638.2991e-9
    d_true = 20.106e-3
    f_px   = 6222.0
    eps_a_t, eps_b_t = 0.37, 0.51

    p      = np.arange(1, 11, dtype=float)
    Delta_a_t = f_px ** 2 * lam_a / d_true
    Delta_b_t = f_px ** 2 * lam_b / d_true
    r2_a   = Delta_a_t * (p - 1 + eps_a_t)
    r2_b   = Delta_b_t * (p - 1 + eps_b_t)
    sr_a   = np.full_like(r2_a, 0.05)
    sr_b   = np.full_like(r2_b, 0.05)

    result = run_tolansky(
        p, r2_a, sr_a,
        p, r2_b, sr_b,
        lam_a_m       = lam_a,
        lam_b_m       = lam_b,
        d_prior_m     = 20.008e-3,
        pixel_pitch_m = 32e-6,
    )

    print_rectangular_array(result)

    pixel_pitch = 32e-6
    f_m  = result.f_px * pixel_pitch * 1e3
    sf_m = result.sigma_f_px * pixel_pitch * 1e3

    print("=== TOLANSKY RESULTS (T5 synthetic, d_true = 20.106 mm) ===")
    print(f"  Δₐ = {result.Delta_a:.4f} ± {result.sigma_Delta_a:.4f} px²")
    print(f"  Δᵦ = {result.Delta_b:.4f} ± {result.sigma_Delta_b:.4f} px²")
    print(f"  Δₐ/Δᵦ observed = {result.Delta_ratio_obs:.6f}   "
          f"expected = {result.Delta_ratio_expected:.6f}   "
          f"residual = {result.Delta_ratio_residual*1e6:.1f} ppm")
    print(f"  N_Δ = {result.N_Delta}")
    print(f"  εₐ = {result.eps_a:.5f} ± {result.sigma_eps_a:.5f}")
    print(f"  εᵦ = {result.eps_b:.5f} ± {result.sigma_eps_b:.5f}")
    print(f"  d  = {result.d_m*1e3:.5f} ± {result.sigma_d_m*1e3:.5f} mm  "
          f"(2σ = {result.two_sigma_d_m*1e3:.5f} mm)  "
          f"|error| = {abs(result.d_m - d_true)*1e6:.1f} µm")
    print(f"  f  = {result.f_px:.1f} ± {result.sigma_f_px:.1f} px  "
          f"(= {f_m:.2f} ± {sf_m:.2f} mm)")
    print(f"  α  = {result.alpha_rad_px:.4E} ± {result.sigma_alpha:.4E} rad/px")
    print(f"  f_consistency = {result.f_consistency*1e6:.1f} ppm")
    print(f"  two_sigma checks:")
    print(f"    two_sigma_d_m   == 2×sigma_d_m  : "
          f"{abs(result.two_sigma_d_m - 2*result.sigma_d_m) < 1e-15}")
    print(f"    two_sigma_f_px  == 2×sigma_f_px : "
          f"{abs(result.two_sigma_f_px - 2*result.sigma_f_px) < 1e-15}")
    print(f"    two_sigma_alpha == 2×sigma_alpha : "
          f"{abs(result.two_sigma_alpha - 2*result.sigma_alpha) < 1e-15}")
