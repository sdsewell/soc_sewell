"""
Tests for Tolansky Two-Line FPI Analysis.

Spec:        specs/S13_tolansky_analysis_2026-04-05.md
Spec tests:  T1–T8
Run with:    pytest tests/test_tolansky_2026_04_05.py -v
"""

import numpy as np
import pytest

from src.fpi.tolansky_2026_04_05 import (
    TolanskyAnalyser,
    TolanskyPipeline,
    TwoLineAnalyser,
)
from src.fpi.m03_annular_reduction_2026_04_06 import FringeProfile, PeakFit, QualityFlags


# ---------------------------------------------------------------------------
# Shared synthetic-data helpers
# ---------------------------------------------------------------------------

# Instrument constants used throughout (T4–T8)
_LAM1 = 640.2248e-9   # m
_LAM2 = 638.2991e-9   # m
_D    = 20.008e-3      # m
_F    = 0.20           # m
_N    = 1.0
_PX   = 32e-6          # m/pixel


def _make_rings(lam, eps, n_rings=10, sigma_r_px=None):
    """
    Build synthetic Tolansky rings from known d, f, ε.

    r_sq = (f²·λ / (n·d)) · (p − 1 + ε)   [SI units → / px² → pixels]
    Returns (p, r_px, sigma_r_px).
    """
    S   = _F ** 2 * lam / (_N * _D)         # m²/fringe
    p   = np.arange(1, n_rings + 1, dtype=float)
    r_sq = S * (p - 1.0 + eps)              # m²
    r   = np.sqrt(r_sq) / _PX               # pixels
    if sigma_r_px is None:
        sigma_r_px = 0.3 / _PX              # follow T4 spec exactly
    sr  = np.full_like(r, sigma_r_px)
    return p, r, sr


def _compatible_epsilons():
    """
    Compute (eps1, eps2, N_int_expected) compatible with round(m_diff).

    For d recovery to work and for T6 to pass simultaneously, we need
    delta_eps = eps1 − eps2 = m_diff − round(m_diff), where
    m_diff = 2·n·d·(1/λ₁ − 1/λ₂).

    When |m_diff − round(m_diff)| < 0.5 (which holds for our wavelengths),
    N_int = round(m_diff) is the correct integer and
    d_recovered = |lever × (N_int + delta_eps)| = |lever × m_diff| = d_true.
    """
    m_diff        = 2 * _N * _D * (1 / _LAM1 - 1 / _LAM2)
    N_int_expected = int(round(m_diff))
    delta_eps      = m_diff - N_int_expected   # eps1 - eps2, magnitude < 0.5
    eps1 = 0.5
    eps2 = float((eps1 - delta_eps) % 1.0)
    return eps1, eps2, N_int_expected


# ---------------------------------------------------------------------------
# T1 — Single-line WLS fit: synthetic data with known answer
# ---------------------------------------------------------------------------

def test_single_line_wls_known_answer():
    """
    Synthetic rings from exact r² = S·p + b.
    Recovered slope must match input to < 0.01%.
    Recovered ε must match to < 1e-4.
    """
    S_true   = 500.0     # px²/fringe
    eps_true = 0.3
    p     = np.arange(1, 11, dtype=float)
    b     = S_true * (eps_true - 1.0)
    r_sq  = S_true * p + b
    r     = np.sqrt(r_sq)
    sigma_r = np.full_like(r, 0.3)

    a   = TolanskyAnalyser(p, r, sigma_r, lam_nm=640.2248,
                           d_m=20.008e-3, pixel_pitch_m=32e-6)
    res = a.run()

    assert abs(res.slope - S_true) / S_true < 1e-4, \
        f"Slope error {abs(res.slope - S_true)/S_true:.2e}"
    assert abs(res.epsilon - eps_true) < 1e-4, \
        f"ε error {abs(res.epsilon - eps_true):.2e}"
    assert res.r2_fit > 0.9999


# ---------------------------------------------------------------------------
# T2 — Weighted fit upweights low-uncertainty rings
# ---------------------------------------------------------------------------

def test_weighted_fit_uses_uncertainties():
    """
    Rings with smaller σ_r must have more influence on the fit.
    Add an outlier with large uncertainty — fit should be unaffected.
    """
    S_true  = 500.0
    p       = np.arange(1, 8, dtype=float)
    b       = S_true * (-0.7)
    r       = np.sqrt(S_true * p + b)
    sigma_r = np.full_like(r, 0.3)

    r_corrupt         = r.copy()
    r_corrupt[-1]    *= 1.5
    sigma_r_corrupt   = sigma_r.copy()
    sigma_r_corrupt[-1] = 50.0   # huge uncertainty → nearly zero weight

    a_clean   = TolanskyAnalyser(p, r, sigma_r, lam_nm=640.2248,
                                  d_m=20.008e-3, pixel_pitch_m=32e-6)
    a_corrupt = TolanskyAnalyser(p, r_corrupt, sigma_r_corrupt,
                                  lam_nm=640.2248, d_m=20.008e-3,
                                  pixel_pitch_m=32e-6)
    res_c = a_clean.run()
    res_x = a_corrupt.run()

    assert abs(res_x.slope - res_c.slope) / res_c.slope < 0.01, \
        "High-sigma outlier should not shift the slope by > 1%"


# ---------------------------------------------------------------------------
# T3 — Successive Δ(r²) coefficient of variation < 2%
# ---------------------------------------------------------------------------

def test_delta_r2_uniformity():
    """For exact Tolansky data, Δ(r²) must be perfectly uniform (CV ≈ 0)."""
    S_true = 450.0
    p      = np.arange(1, 12, dtype=float)
    r      = np.sqrt(S_true * p + S_true * (-0.6))
    sigma_r = np.full_like(r, 0.2)

    a   = TolanskyAnalyser(p, r, sigma_r, lam_nm=640.2248,
                           d_m=20.008e-3, pixel_pitch_m=32e-6)
    res = a.run()

    cv = res.delta_r_sq.std() / abs(res.delta_r_sq.mean()) * 100
    assert cv < 2.0, f"CV(Δr²) = {cv:.2f}% for exact data; expected < 2%"


# ---------------------------------------------------------------------------
# T4 — Two-line joint fit: slope ratio constraint enforced
# ---------------------------------------------------------------------------

def test_two_line_slope_ratio():
    """
    In the joint fit, S₂ must equal S₁ × λ₂/λ₁ to < 1e-10.
    """
    p1, r1, sr1 = _make_rings(_LAM1, eps=0.4)
    p2, r2, sr2 = _make_rings(_LAM2, eps=0.4)

    a1 = TolanskyAnalyser(p1, r1, sr1, lam_nm=_LAM1*1e9,
                          d_m=_D, pixel_pitch_m=_PX)
    a2 = TolanskyAnalyser(p2, r2, sr2, lam_nm=_LAM2*1e9,
                          d_m=_D, pixel_pitch_m=_PX)
    tla = TwoLineAnalyser(a1, a2,
                          lam1_nm=_LAM1*1e9, lam2_nm=_LAM2*1e9,
                          d_prior_m=_D, n=_N, pixel_pitch_m=_PX)
    res = tla.run()

    expected_ratio = _LAM2 / _LAM1
    actual_ratio   = res.S2 / res.S1
    assert abs(actual_ratio - expected_ratio) < 1e-10, \
        f"Slope ratio {actual_ratio:.12f} ≠ λ₂/λ₁ = {expected_ratio:.12f}"


# ---------------------------------------------------------------------------
# T5 — Excess fractions recovers d to < 1 µm
# ---------------------------------------------------------------------------

def test_excess_fractions_d_recovery():
    """
    Synthetic two-line data from known d = 20.008 mm with physically
    consistent fractional orders.  Recovered d must match to < 1 µm.
    """
    eps1, eps2, _ = _compatible_epsilons()

    p1, r1, sr1 = _make_rings(_LAM1, eps=eps1)
    p2, r2, sr2 = _make_rings(_LAM2, eps=eps2)

    a1 = TolanskyAnalyser(p1, r1, sr1, lam_nm=_LAM1*1e9,
                          d_m=_D, pixel_pitch_m=_PX)
    a2 = TolanskyAnalyser(p2, r2, sr2, lam_nm=_LAM2*1e9,
                          d_m=_D, pixel_pitch_m=_PX)
    tla = TwoLineAnalyser(a1, a2,
                          lam1_nm=_LAM1*1e9, lam2_nm=_LAM2*1e9,
                          d_prior_m=_D, n=_N, pixel_pitch_m=_PX)
    res = tla.run()

    assert abs(res.d_m - _D) < 1e-6, \
        f"|d_recovered − d_true| = {abs(res.d_m - _D)*1e6:.3f} µm > 1 µm"


# ---------------------------------------------------------------------------
# T6 — N_int correctly identified from d_prior
# ---------------------------------------------------------------------------

def test_N_int_from_prior():
    """
    N_int must be the correct integer for d = 20.008 mm.
    For λ₁=640.2248 nm, λ₂=638.2991 nm, n=1:
        N_int = round(2 × 20.008e-3 × (1/640.2248e-9 − 1/638.2991e-9))
    """
    eps1, eps2, expected_N_int = _compatible_epsilons()

    p1, r1, sr1 = _make_rings(_LAM1, eps=eps1)
    p2, r2, sr2 = _make_rings(_LAM2, eps=eps2)

    a1 = TolanskyAnalyser(p1, r1, sr1, lam_nm=_LAM1*1e9,
                          d_m=_D, pixel_pitch_m=_PX)
    a2 = TolanskyAnalyser(p2, r2, sr2, lam_nm=_LAM2*1e9,
                          d_m=_D, pixel_pitch_m=_PX)
    tla = TwoLineAnalyser(a1, a2,
                          lam1_nm=_LAM1*1e9, lam2_nm=_LAM2*1e9,
                          d_prior_m=_D, n=_N, pixel_pitch_m=_PX)
    res = tla.run()

    assert res.N_int == expected_N_int, \
        f"N_int = {res.N_int}, expected {expected_N_int}"


# ---------------------------------------------------------------------------
# T7 — TolanskyPipeline from FringeProfile
# ---------------------------------------------------------------------------

def _make_fringe_profile_with_peaks():
    """
    Build a minimal FringeProfile with 20 synthetic PeakFit entries.
    10 peaks for λ₁ (amplitude 1000 ADU) and 10 for λ₂ (amplitude 800 ADU).
    Ring radii come from the Tolansky formula with physical ε values.
    """
    eps1, eps2, _ = _compatible_epsilons()
    n_bins = 150

    def ring_radii(lam, eps, n_rings=10):
        S_m   = _F**2 * lam / (_N * _D)
        p_arr = np.arange(1, n_rings + 1, dtype=float)
        r_sq  = S_m * (p_arr - 1.0 + eps)
        return np.sqrt(r_sq) / _PX    # pixels

    r1 = ring_radii(_LAM1, eps1)   # 10 λ₁ radii in px
    r2 = ring_radii(_LAM2, eps2)   # 10 λ₂ radii in px

    peak_fits = []
    for k, rr in enumerate(r1):
        peak_fits.append(PeakFit(
            peak_idx=k, r_raw_px=rr, profile_raw=2000.0,
            r_fit_px=rr, sigma_r_fit_px=0.1,
            amplitude_adu=1000.0, width_px=1.5, fit_ok=True,
        ))
    for k, rr in enumerate(r2):
        peak_fits.append(PeakFit(
            peak_idx=k + 10, r_raw_px=rr, profile_raw=1800.0,
            r_fit_px=rr, sigma_r_fit_px=0.1,
            amplitude_adu=800.0, width_px=1.5, fit_ok=True,
        ))
    # Sort by radius (as M03 would)
    peak_fits.sort(key=lambda p: p.r_raw_px)

    dummy = np.zeros(n_bins)
    inf_arr = np.full(n_bins, np.inf)
    masked = np.zeros(n_bins, dtype=bool)

    fp = FringeProfile(
        profile=dummy, sigma_profile=dummy.copy(),
        two_sigma_profile=dummy.copy(),
        r_grid=np.linspace(0, 128, n_bins),
        r2_grid=np.linspace(0, 128**2, n_bins),
        n_pixels=np.full(n_bins, 100, dtype=int),
        masked=masked,
        cx=127.5, cy=127.5,
        sigma_cx=0.05, sigma_cy=0.05,
        two_sigma_cx=0.10, two_sigma_cy=0.10,
        seed_source="human",
        stage1_cx=127.5, stage1_cy=127.5,
        cost_at_min=0.0,
        quality_flags=QualityFlags.GOOD,
        sparse_bins=False,
        r_min_px=0.0, r_max_px=128.0,
        n_bins=n_bins, n_subpixels=1,
        sigma_clip=3.0,
        image_shape=(256, 256),
        peak_fits=peak_fits,
        dark_subtracted=False,
        dark_n_frames=0,
    )
    return fp


def test_pipeline_from_fringe_profile():
    """
    TolanskyPipeline must recover d to < 0.1 mm and α to < 5%.
    Uses amplitude_split_fraction=0.9 to cleanly separate
    λ₁ (1000 ADU) from λ₂ (800 ADU) families.
    """
    fp       = _make_fringe_profile_with_peaks()
    pipeline = TolanskyPipeline(
        fp,
        d_prior_m=_D,
        lam1_nm=_LAM1 * 1e9,
        lam2_nm=_LAM2 * 1e9,
        amplitude_split_fraction=0.9,   # threshold = 0.9 × 1000 = 900 → split at 900
        n=_N,
        pixel_pitch_m=_PX,
    )
    result = pipeline.run()

    alpha_true = _PX / _F   # = 1/6250 ≈ 1.6e-4 rad/px
    assert abs(result.d_m - _D) < 1e-4, \
        f"d error = {abs(result.d_m - _D)*1e3:.4f} mm"
    assert abs(result.alpha_rad_px - alpha_true) / alpha_true < 0.05, \
        f"α = {result.alpha_rad_px:.4e}, expected ≈ {alpha_true:.4e}"


# ---------------------------------------------------------------------------
# T8 — TwoLineResult has all S04 two_sigma fields
# ---------------------------------------------------------------------------

def test_two_sigma_fields_present():
    """All two_sigma_ fields must equal exactly 2 × sigma_ (S04 convention)."""
    eps1, eps2, _ = _compatible_epsilons()

    p1, r1, sr1 = _make_rings(_LAM1, eps=eps1)
    p2, r2, sr2 = _make_rings(_LAM2, eps=eps2)

    a1 = TolanskyAnalyser(p1, r1, sr1, lam_nm=_LAM1*1e9,
                          d_m=_D, pixel_pitch_m=_PX)
    a2 = TolanskyAnalyser(p2, r2, sr2, lam_nm=_LAM2*1e9,
                          d_m=_D, pixel_pitch_m=_PX)
    tla = TwoLineAnalyser(a1, a2,
                          lam1_nm=_LAM1*1e9, lam2_nm=_LAM2*1e9,
                          d_prior_m=_D, n=_N, pixel_pitch_m=_PX)
    res = tla.run()

    assert abs(res.two_sigma_d_m   - 2.0 * res.sigma_d_m)   < 1e-15, \
        f"two_sigma_d_m = {res.two_sigma_d_m}, 2×sigma = {2*res.sigma_d_m}"
    assert abs(res.two_sigma_f_px  - 2.0 * res.sigma_f_px)  < 1e-15, \
        f"two_sigma_f_px = {res.two_sigma_f_px}, 2×sigma = {2*res.sigma_f_px}"
    assert abs(res.two_sigma_alpha - 2.0 * res.sigma_alpha)  < 1e-15, \
        f"two_sigma_alpha = {res.two_sigma_alpha}, 2×sigma = {2*res.sigma_alpha}"
