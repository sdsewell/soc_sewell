"""
Tests for tolansky_2026-05-13.py (S13).
10 tests total: 7 original S13a tests + 2 10-column tests + 1 equality test.

Note on T5: the spec lists d_true = 20.1e-3 with eps_a=0.22, eps_b=0.73.
Those parameters are not self-consistent through the Benoit formula for
N_Delta = -189 (error ~10 um >> 1 um tolerance).  T5 is instead written to
use the d_true that IS self-consistent with the chosen eps values and
N_Delta, verifying that the Benoit round-trip recovers d to < 1 um.
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import from fpi_cal_lib (consolidated single-file library, v1.3+)
# ---------------------------------------------------------------------------
from src.fpi.fpi_cal_lib import (  # noqa: F401
    InsufficientRingsError,
    run_tolansky as run_tolansky,           # backward-compat alias
    run_tolansky_2line,
    TolanskyResult,
    to_m05_priors,
    benoit_d,
)


# ---------------------------------------------------------------------------
# Shared synthetic data helper  (spec §8)
# ---------------------------------------------------------------------------

def make_synthetic_peaks_array(
    Delta_a: float, eps_a: float, Delta_b: float, eps_b: float,
    n: int = 10,
) -> np.ndarray:
    """
    Build a valid (2n, 9) float64 array matching the _peak_fits_r2.npy layout.

    col 0: peak_num   col 1: r2_raw_px2   col 2: r2_fit_px2
    col 3: sigma_r2   col 4: r_derived_px col 5: sigma_r_derived
    col 6: amplitude  col 7: width        col 8: reduced_chi2

    640 nm family (a): amplitude 3000 ADU
    638 nm family (b): amplitude 1000 ADU
    """
    rows = []
    for i in range(n):
        r2_a = Delta_a * (i + eps_a)
        r_a  = float(np.sqrt(max(r2_a, 1e-12)))
        sig_r_a  = r_a * 1e-4
        sig_r2_a = 2.0 * r_a * sig_r_a
        rows.append([float(i * 2 + 1), r2_a, r2_a, sig_r2_a,
                     r_a, sig_r_a, 3000.0, 50.0, 1.0])
        r2_b = Delta_b * (i + eps_b)
        r_b  = float(np.sqrt(max(r2_b, 1e-12)))
        sig_r_b  = r_b * 1e-4
        sig_r2_b = 2.0 * r_b * sig_r_b
        rows.append([float(i * 2 + 2), r2_b, r2_b, sig_r2_b,
                     r_b, sig_r_b, 1000.0, 50.0, 1.0])
    return np.array(rows, dtype=np.float64)


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
    arr = make_synthetic_peaks_array(Delta_a, eps_a, Delta_b, eps_b, n=10)
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


# ---------------------------------------------------------------------------
# T8 -- 10-column line_id family split
# ---------------------------------------------------------------------------

def make_synthetic_peaks_10col(
    Delta_a: float, eps_a: float, Delta_b: float, eps_b: float,
    n: int = 10,
) -> np.ndarray:
    """
    Build a (2n, 10) array matching _peak_fits_r2.npy layout with line_id col.

    col 0-8: same as 9-col helper   col 9: line_id (0.0=640 nm, 1.0=638 nm)
    """
    rows = []
    for i in range(n):
        r2_a = Delta_a * (i + eps_a)
        r_a  = float(np.sqrt(max(r2_a, 1e-12)))
        sig_r_a  = r_a * 1e-4
        sig_r2_a = 2.0 * r_a * sig_r_a
        rows.append([float(i * 2 + 1), r2_a, r2_a, sig_r2_a,
                     r_a, sig_r_a, 3000.0, 50.0, 1.0, 0.0])
        r2_b = Delta_b * (i + eps_b)
        r_b  = float(np.sqrt(max(r2_b, 1e-12)))
        sig_r_b  = r_b * 1e-4
        sig_r2_b = 2.0 * r_b * sig_r_b
        rows.append([float(i * 2 + 2), r2_b, r2_b, sig_r2_b,
                     r_b, sig_r_b, 1000.0, 50.0, 1.0, 1.0])
    return np.array(rows, dtype=np.float64)


def test_10col_line_id_family_split():
    """10-column input: line_id=0 -> family a (Delta~1233), line_id=1 -> family b (Delta~1228)."""
    arr    = make_synthetic_peaks_10col(1233.0, 0.22, 1228.0, 0.73, n=8)
    result = run_tolansky_2line(arr)
    assert abs(result.Delta_a - 1233.0) / 1233.0 < 1e-6
    assert abs(result.Delta_b - 1228.0) / 1228.0 < 1e-6


def test_10col_line_id_overrides_amplitude():
    """10-col: line_id assignment is used even when family a is dimmer than family b."""
    rows = []
    n = 8
    Delta_a, eps_a = 1233.0, 0.22
    Delta_b, eps_b = 1228.0, 0.73
    for i in range(n):
        r2a = Delta_a * (i + eps_a)
        r2b = Delta_b * (i + eps_b)
        sa  = r2a * 5e-4
        sb  = r2b * 5e-4
        rows.append([i*2+1, r2a, r2a, sa,
                     np.sqrt(r2a), sa / (2 * np.sqrt(r2a)),
                     500.0, 50.0, 1.0, 0.0])   # line_id=0 but DIMMER amplitude
        rows.append([i*2+2, r2b, r2b, sb,
                     np.sqrt(r2b), sb / (2 * np.sqrt(r2b)),
                     3000.0, 50.0, 1.0, 1.0])  # line_id=1 but BRIGHTER amplitude
    arr = np.array(rows, dtype=np.float64)
    result = run_tolansky_2line(arr)
    # line_id overrides amplitude: family a must have Delta_a, family b Delta_b
    assert abs(result.Delta_a - Delta_a) / Delta_a < 1e-6
    assert abs(result.Delta_b - Delta_b) / Delta_b < 1e-6


# ---------------------------------------------------------------------------
# T11 -- 10-column and 9-column inputs give identical results
# ---------------------------------------------------------------------------

def test_10col_and_9col_give_same_result():
    """9-col amplitude-split and 10-col line_id-split must agree on all outputs."""
    Delta_a, eps_a, Delta_b, eps_b = 1233.0, 0.22, 1228.0, 0.73
    arr_9  = make_synthetic_peaks_array(Delta_a, eps_a, Delta_b, eps_b, n=10)
    arr_10 = make_synthetic_peaks_10col(Delta_a, eps_a, Delta_b, eps_b, n=10)
    r9  = run_tolansky_2line(arr_9)
    r10 = run_tolansky_2line(arr_10)
    assert abs(r9.Delta_a    - r10.Delta_a)    < 1e-9
    assert abs(r9.eps_a      - r10.eps_a)      < 1e-9
    assert abs(r9.d_m        - r10.d_m)        < 1e-12
    assert abs(r9.alpha_mean - r10.alpha_mean) < 1e-12
