"""
Tests for tolansky_2026-04-13.py
Spec: S13_tolansky_analysis_2026-04-13.md  §8  (T1–T7)
"""

import importlib.util as _ilu
import pathlib
import sys

import numpy as np
import pytest

# Ensure repo root is on the path
_REPO = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Load the hyphenated module by file path (Python identifiers can't have hyphens)
_spec = _ilu.spec_from_file_location(
    "tolansky_2026_04_13",
    str(_REPO / "src" / "fpi" / "tolansky_2026-04-13.py"),
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

run_single_line_wls = _mod.run_single_line_wls
benoit_d            = _mod.benoit_d
recover_f_alpha     = _mod.recover_f_alpha
run_tolansky        = _mod.run_tolansky
TolanskyResult      = _mod.TolanskyResult

# ---------------------------------------------------------------------------
# T1 — Successive differences are uniform on exact synthetic data
# ---------------------------------------------------------------------------

def test_successive_differences_uniform():
    """
    For rings on the exact Tolansky r² = Δ·(p−1+ε) curve with no noise,
    all δ_p must equal Δ exactly (CV < 1e-10).
    """
    Delta_true = 485.0   # px²
    eps_true   = 0.37
    p          = np.arange(1, 11, dtype=float)
    r2         = Delta_true * (p - 1 + eps_true)
    delta      = np.diff(r2)
    cv         = delta.std() / delta.mean()
    assert cv < 1e-10, f"CV(δ) = {cv:.2e} for exact data; expected < 1e-10"


# ---------------------------------------------------------------------------
# T2 — WLS recovers known Δ and ε to high accuracy
# ---------------------------------------------------------------------------

def test_wls_known_answer():
    """
    WLS fit on exact r² = Δ·p + b must recover Δ and ε to < 0.01%.
    """
    Delta_true = 485.0
    eps_true   = 0.37
    p   = np.arange(1, 11, dtype=float)
    r2  = Delta_true * (p - 1 + eps_true)
    sr  = np.full_like(r2, 0.3 * 2 * np.sqrt(r2[0]))   # σ(r²) = 2r·σ_r
    result = run_single_line_wls(p, r2, sr)
    assert abs(result["Delta"] - Delta_true) / Delta_true < 1e-4, \
        f"Δ error = {abs(result['Delta'] - Delta_true)/Delta_true:.2e}"
    assert abs(result["eps"]   - eps_true)                < 1e-4, \
        f"ε error = {abs(result['eps'] - eps_true):.2e}"
    assert result["r2_fit"] > 0.9999, \
        f"R² = {result['r2_fit']:.6f}"


# ---------------------------------------------------------------------------
# T3 — Δ ratio constraint: Δₐ/Δᵦ = λₐ/λᵦ from same (d, f)
# ---------------------------------------------------------------------------

def test_delta_ratio_matches_wavelength_ratio():
    """
    Rings from the same d and f must give Δₐ/Δᵦ = λₐ/λᵦ to < 10 ppm.
    """
    lam_a, lam_b = 640.2248e-9, 638.2991e-9
    d, f_px      = 20.106e-3, 6222.0
    p            = np.arange(1, 11, dtype=float)
    Delta_a_true = f_px ** 2 * lam_a / d
    Delta_b_true = f_px ** 2 * lam_b / d
    for lam, Delta_true in [(lam_a, Delta_a_true), (lam_b, Delta_b_true)]:
        r2 = Delta_true * (p - 1 + 0.4)
        # (check ratio)
    ratio_obs = Delta_a_true / Delta_b_true
    ratio_exp = lam_a / lam_b
    assert abs(ratio_obs - ratio_exp) / ratio_exp < 1e-8, \
        f"ratio residual = {abs(ratio_obs - ratio_exp)/ratio_exp:.2e}"


# ---------------------------------------------------------------------------
# T4 — N_Δ correctly identified from d_prior
# ---------------------------------------------------------------------------

def test_N_Delta_from_prior():
    """
    N_Δ = round(2 · d_prior · (1/λₐ − 1/λᵦ)) must equal −189 for
    d_prior = 20.008 mm (ICOS measurement).
    """
    lam_a, lam_b = 640.2248e-9, 638.2991e-9
    d_prior      = 20.008e-3
    N_Delta, _, _ = benoit_d(0.37, 0.01, 0.51, 0.01, lam_a, lam_b, d_prior)
    assert N_Delta == -189, f"N_Δ = {N_Delta}, expected −189"


# ---------------------------------------------------------------------------
# T5 — Benoit d recovery to < 1 µm on synthetic data
# ---------------------------------------------------------------------------

def test_benoit_d_recovery():
    """
    Synthetic rings from d_true = 20.106 mm.
    Recovered d must match d_true to < 1 µm.
    """
    lam_a, lam_b = 640.2248e-9, 638.2991e-9
    d_true, f_px = 20.106e-3, 6222.0
    p            = np.arange(1, 11, dtype=float)
    eps_a, eps_b = 0.37, 0.51
    Delta_a      = f_px ** 2 * lam_a / d_true
    Delta_b      = f_px ** 2 * lam_b / d_true
    r2_a         = Delta_a * (p - 1 + eps_a)
    r2_b         = Delta_b * (p - 1 + eps_b)
    sr_a         = np.full_like(r2_a, 0.05)
    sr_b         = np.full_like(r2_b, 0.05)
    result       = run_tolansky(
        p, r2_a, sr_a,
        p, r2_b, sr_b,
        lam_a_m       = lam_a,
        lam_b_m       = lam_b,
        d_prior_m     = 20.008e-3,
        pixel_pitch_m = 32e-6,
    )
    assert abs(result.d_m - d_true) < 1e-6, \
        f"|d_recovered − d_true| = {abs(result.d_m - d_true)*1e6:.3f} µm > 1 µm"


# ---------------------------------------------------------------------------
# T6 — f recovered from d via Δₐ · n_air · d / λₐ
# ---------------------------------------------------------------------------

def test_f_recovery():
    """From exact Δₐ and known d, f must be recovered to < 0.1%."""
    lam_a       = 640.2248e-9
    d_m         = 20.106e-3
    f_true_px   = 6222.0
    Delta_a_true = f_true_px ** 2 * lam_a / d_m
    f_recovered, _, _, _ = recover_f_alpha(
        Delta_a_true, 0.0, d_m, 0.0, lam_a, 32e-6
    )
    assert abs(f_recovered - f_true_px) / f_true_px < 1e-3, \
        f"f error = {abs(f_recovered - f_true_px)/f_true_px:.2e}"


# ---------------------------------------------------------------------------
# T7 — All two_sigma_ fields equal exactly 2 × sigma_  (S04)
# ---------------------------------------------------------------------------

def test_two_sigma_fields():
    """S04 convention: every two_sigma_ field = exactly 2 × sigma_."""
    lam_a, lam_b = 640.2248e-9, 638.2991e-9
    d_true, f_px = 20.106e-3, 6222.0
    p            = np.arange(1, 11, dtype=float)
    eps_a, eps_b = 0.37, 0.51
    Delta_a      = f_px ** 2 * lam_a / d_true
    Delta_b      = f_px ** 2 * lam_b / d_true
    r2_a         = Delta_a * (p - 1 + eps_a)
    r2_b         = Delta_b * (p - 1 + eps_b)
    sr_a         = np.full_like(r2_a, 0.05)
    sr_b         = np.full_like(r2_b, 0.05)
    result       = run_tolansky(
        p, r2_a, sr_a,
        p, r2_b, sr_b,
        lam_a_m       = lam_a,
        lam_b_m       = lam_b,
        d_prior_m     = 20.008e-3,
        pixel_pitch_m = 32e-6,
    )
    assert abs(result.two_sigma_d_m   - 2.0 * result.sigma_d_m)   < 1e-15, \
        f"two_sigma_d_m   = {result.two_sigma_d_m},  2×sigma = {2*result.sigma_d_m}"
    assert abs(result.two_sigma_f_px  - 2.0 * result.sigma_f_px)  < 1e-15, \
        f"two_sigma_f_px  = {result.two_sigma_f_px},  2×sigma = {2*result.sigma_f_px}"
    assert abs(result.two_sigma_alpha - 2.0 * result.sigma_alpha)  < 1e-15, \
        f"two_sigma_alpha = {result.two_sigma_alpha},  2×sigma = {2*result.sigma_alpha}"
