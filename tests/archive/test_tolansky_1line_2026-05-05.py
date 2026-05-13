"""
Tests for tolansky_1line_2026-05-05.py (S13b).
All 6 tests mirror the spec exactly (§8).
"""

import importlib.util
import pathlib
import sys
import warnings

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import the production module by file path (hyphenated filename is not a
# valid Python identifier, so we use importlib rather than a normal import).
# ---------------------------------------------------------------------------
_mod_path = (
    pathlib.Path(__file__).resolve().parent.parent
    / "src" / "fpi" / "tolansky_1line_2026-05-05.py"
)
_spec = importlib.util.spec_from_file_location("tolansky_1line", _mod_path)
_mod  = importlib.util.module_from_spec(_spec)
sys.modules["tolansky_1line"] = _mod   # required for @dataclass resolution
_spec.loader.exec_module(_mod)

run_tolansky_1line     = _mod.run_tolansky_1line
TolanskyResult1L       = _mod.TolanskyResult1L
InsufficientRingsError = _mod.InsufficientRingsError


# ---------------------------------------------------------------------------
# Shared synthetic data helper  (spec §8)
# ---------------------------------------------------------------------------

def make_synthetic_peaks_array(Delta, eps, n, sigma_r=0.3):
    """
    Generate a valid (n, 9) float64 array matching the annular_reduction.py
    column layout, for use across all tests.

    cols: peak_num | r_raw_px | r_fit_px | sigma_r | r2 | sigma_r2 |
          amp      | width    | chi2
    """
    p   = np.arange(1, n + 1, dtype=float)
    r2  = Delta * (p - 1 + eps)
    r   = np.sqrt(r2)
    sr  = np.full(n, sigma_r)
    sr2 = 2.0 * r * sr
    return np.column_stack([
        p, r, r, sr, r2, sr2,
        np.full(n, 500.0), np.full(n, 1.5), np.ones(n),
    ])


# ---------------------------------------------------------------------------
# T1 — WLS recovers known Δ and ε on noise-free synthetic data
# ---------------------------------------------------------------------------

def test_wls_single_line_known_answer():
    """Exact synthetic r²; Δ and ε must be recovered to < 0.01%."""
    Delta_true = 477.0
    eps_true   = 0.42
    arr    = make_synthetic_peaks_array(Delta_true, eps_true, n=7)
    result = run_tolansky_1line(arr, d_cal_m=None)
    assert abs(result.Delta - Delta_true) / Delta_true < 1e-4
    assert abs(result.eps   - eps_true)                < 1e-4


# ---------------------------------------------------------------------------
# T2 — Successive differences uniform on exact data
# ---------------------------------------------------------------------------

def test_successive_differences_uniform_1line():
    Delta_true = 477.0
    arr    = make_synthetic_peaks_array(Delta_true, 0.42, n=7)
    result = run_tolansky_1line(arr, d_cal_m=None)
    assert result.cv_delta < 1e-10


# ---------------------------------------------------------------------------
# T3 — Mode A α recovery with known d_cal
# ---------------------------------------------------------------------------

def test_alpha_recovery_mode_a():
    """With known d_cal and Δ, α must be recovered to < 0.1%."""
    LAM_OI     = 630.0304e-9
    d_cal      = 20.0006e-3
    f_true     = 6230.0
    alpha_true = 1.0 / f_true
    Delta_true = f_true ** 2 * LAM_OI / d_cal
    arr    = make_synthetic_peaks_array(Delta_true, 0.42, n=7)
    result = run_tolansky_1line(arr, d_cal_m=d_cal, sigma_d_cal_m=5e-6)
    assert abs(result.alpha_rad_px - alpha_true) / alpha_true < 1e-3


# ---------------------------------------------------------------------------
# T4 — Mode B: α is None when d_cal not supplied
# ---------------------------------------------------------------------------

def test_mode_b_no_alpha():
    arr    = make_synthetic_peaks_array(477.0, 0.42, n=5)
    result = run_tolansky_1line(arr, d_cal_m=None)
    assert result.alpha_rad_px is None
    assert result.Delta is not None


# ---------------------------------------------------------------------------
# T5 — NaN rows dropped with correct count
# ---------------------------------------------------------------------------

def test_nan_rows_dropped():
    arr = make_synthetic_peaks_array(477.0, 0.42, n=5)
    arr[2, 2:] = np.nan   # invalidate row index 2 (peak 3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_tolansky_1line(arr, d_cal_m=None)
    assert result.n_nan_dropped == 1
    assert result.n_rings       == 4


# ---------------------------------------------------------------------------
# T6 — All two_sigma_ fields are exactly 2 × sigma_ (S04)
# ---------------------------------------------------------------------------

def test_two_sigma_fields_1line():
    arr    = make_synthetic_peaks_array(477.0, 0.42, n=7)
    result = run_tolansky_1line(arr, d_cal_m=20.0006e-3, sigma_d_cal_m=5e-6)
    assert abs(result.two_sigma_Delta - 2.0 * result.sigma_Delta) < 1e-14
    assert abs(result.two_sigma_eps   - 2.0 * result.sigma_eps)   < 1e-14
    assert abs(result.two_sigma_alpha - 2.0 * result.sigma_alpha) < 1e-14
