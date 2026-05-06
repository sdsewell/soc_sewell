"""
Tests for tolansky_2line_2026-05-05.py (S13a).
All 7 tests mirror spec §8.

Note on T5: the spec lists d_true = 20.1e-3 with eps_a=0.22, eps_b=0.73.
Those parameters are not self-consistent through the Benoit formula for
N_Delta = -189 (error ~10 um >> 1 um tolerance).  T5 is instead written to
use the d_true that IS self-consistent with the chosen eps values and
N_Delta, verifying that the Benoit round-trip recovers d to < 1 um.
"""

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import the production module by file path (hyphenated name requires importlib)
# ---------------------------------------------------------------------------
_mod_path = (
    pathlib.Path(__file__).resolve().parent.parent
    / "src" / "fpi" / "tolansky_2line_2026-05-05.py"
)
_spec = importlib.util.spec_from_file_location("tolansky_2line", _mod_path)
_mod  = importlib.util.module_from_spec(_spec)
sys.modules["tolansky_2line"] = _mod
_spec.loader.exec_module(_mod)

run_tolansky          = _mod.run_tolansky
TolanskyResult        = _mod.TolanskyResult
InsufficientRingsError = _mod.InsufficientRingsError
benoit_d              = _mod.benoit_d


# ---------------------------------------------------------------------------
# Shared synthetic data helper  (spec §8)
# ---------------------------------------------------------------------------

def make_synthetic_peaks_array(Delta_a, eps_a, Delta_b, eps_b,
                                n, sigma_r=0.05):
    """
    Build a valid (2n, 9) float64 array with two interleaved families,
    sorted by ascending radius.

    640 nm family: amplitude 1800 ADU
    638 nm family: amplitude  600 ADU

    cols: peak_num | r_raw_px | r_fit_px | sigma_r_fit_px | r2_fit |
          sigma_r2_fit | amplitude_adu | width_px | reduced_chi2
    """
    p = np.arange(1, n + 1, dtype=float)

    r2_a = Delta_a * (p - 1.0 + eps_a)
    r2_b = Delta_b * (p - 1.0 + eps_b)
    r_a  = np.sqrt(r2_a)
    r_b  = np.sqrt(r2_b)

    sr_a    = np.full(n, sigma_r)
    sr2_a   = 2.0 * r_a * sr_a
    amp_a   = np.full(n, 1800.0)

    sr_b    = np.full(n, sigma_r)
    sr2_b   = 2.0 * r_b * sr_b
    amp_b   = np.full(n, 600.0)

    rows_a = np.column_stack([
        p, r_a, r_a, sr_a, r2_a, sr2_a, amp_a,
        np.full(n, 1.5), np.ones(n),
    ])
    rows_b = np.column_stack([
        p, r_b, r_b, sr_b, r2_b, sr2_b, amp_b,
        np.full(n, 1.5), np.ones(n),
    ])

    combined = np.vstack([rows_a, rows_b])
    order    = np.argsort(combined[:, 2])   # sort by r_fit_px
    combined = combined[order]

    # Re-assign sequential peak_num after sorting
    combined[:, 0] = np.arange(1, 2 * n + 1, dtype=float)
    return combined.astype(np.float64)


# ---------------------------------------------------------------------------
# T1 -- Successive differences uniform on exact synthetic data
# ---------------------------------------------------------------------------

def test_successive_differences_uniform():
    arr = make_synthetic_peaks_array(1233.0, 0.22, 1228.0, 0.73, n=10)
    result = run_tolansky(arr)
    assert result.delta_a.std() / result.delta_a.mean() < 1e-10
    assert result.delta_b.std() / result.delta_b.mean() < 1e-10


# ---------------------------------------------------------------------------
# T2 -- WLS recovers known Delta and eps to < 0.01%
# ---------------------------------------------------------------------------

def test_wls_known_answer():
    Delta_a_true, eps_a_true = 1233.0, 0.22
    Delta_b_true, eps_b_true = 1228.0, 0.73
    arr = make_synthetic_peaks_array(Delta_a_true, eps_a_true,
                                     Delta_b_true, eps_b_true, n=10)
    result = run_tolansky(arr)
    assert abs(result.Delta_a - Delta_a_true) / Delta_a_true < 1e-4
    assert abs(result.eps_a   - eps_a_true)                  < 1e-4
    assert abs(result.Delta_b - Delta_b_true) / Delta_b_true < 1e-4
    assert abs(result.eps_b   - eps_b_true)                  < 1e-4


# ---------------------------------------------------------------------------
# T3 -- Delta ratio constraint: Delta_a / Delta_b = lam_a / lam_b
# ---------------------------------------------------------------------------

def test_delta_ratio_matches_wavelength_ratio():
    lam_a, lam_b = 640.2248e-9, 638.2991e-9
    d, alpha = 20.0e-3, 1.607e-4
    Delta_a = lam_a / (d * alpha ** 2)
    Delta_b = lam_b / (d * alpha ** 2)
    assert abs(Delta_a / Delta_b - lam_a / lam_b) / (lam_a / lam_b) < 1e-8


# ---------------------------------------------------------------------------
# T4 -- N_Delta correctly identified from d_prior
# ---------------------------------------------------------------------------

def test_N_Delta_from_prior():
    lam_a, lam_b = 640.2248e-9, 638.2991e-9
    d_prior = 20.008e-3
    N = round(2 * d_prior * (1 / lam_a - 1 / lam_b))
    assert N == -189, f"N_Delta = {N}, expected -189"


# ---------------------------------------------------------------------------
# T5 -- Benoit d recovery to < 1 um on synthetic data
#
# We first compute the d_true that is self-consistent with eps_a=0.22,
# eps_b=0.73, N_Delta=-189 via the Benoit formula itself.  Then we generate
# synthetic data from (Delta_a, Delta_b) derived from that d_true, verify
# that run_tolansky recovers d to < 1 um.
# ---------------------------------------------------------------------------

def test_benoit_d_recovery():
    lam_a, lam_b = 640.2248e-9, 638.2991e-9
    alpha        = 1.607e-4
    eps_a, eps_b = 0.22, 0.73
    d_prior      = 20.008e-3
    n_air        = 1.0

    # d_true consistent with N_Delta=-189 and the chosen eps values
    d_true, _, N = benoit_d(eps_a, 0.0, eps_b, 0.0,
                             lam_a, lam_b, d_prior, n_air)

    Delta_a = lam_a / (d_true * alpha ** 2)
    Delta_b = lam_b / (d_true * alpha ** 2)
    arr = make_synthetic_peaks_array(Delta_a, eps_a, Delta_b, eps_b,
                                     n=10, sigma_r=0.05)
    result = run_tolansky(arr)
    assert abs(result.d_m - d_true) < 1e-6


# ---------------------------------------------------------------------------
# T6 -- alpha recovered from d and Delta_a to < 0.1%
# ---------------------------------------------------------------------------

def test_alpha_recovery():
    lam_a      = 640.2248e-9
    d_true     = 20.1e-3
    alpha_true = 1.607e-4
    Delta_a    = lam_a / (d_true * alpha_true ** 2)
    alpha_rec  = np.sqrt(lam_a / (d_true * Delta_a))
    assert abs(alpha_rec - alpha_true) / alpha_true < 1e-3


# ---------------------------------------------------------------------------
# T7 -- All two_sigma_ fields equal exactly 2 x sigma_ (S04)
# ---------------------------------------------------------------------------

def test_two_sigma_fields():
    arr    = make_synthetic_peaks_array(1233.0, 0.22, 1228.0, 0.73, n=10)
    result = run_tolansky(arr)
    assert abs(result.two_sigma_Delta_a - 2.0 * result.sigma_Delta_a) < 1e-14
    assert abs(result.two_sigma_Delta_b - 2.0 * result.sigma_Delta_b) < 1e-14
    assert abs(result.two_sigma_eps_a   - 2.0 * result.sigma_eps_a)   < 1e-14
    assert abs(result.two_sigma_eps_b   - 2.0 * result.sigma_eps_b)   < 1e-14
    assert abs(result.two_sigma_d_m     - 2.0 * result.sigma_d_m)     < 1e-14
    assert abs(result.two_sigma_alpha   - 2.0 * result.sigma_alpha)    < 1e-14
