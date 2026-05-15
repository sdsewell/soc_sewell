"""
Tests for windcube/fpi_pipeline.py — FPI Pipeline Coordination Library.
Spec:    specs/S_L01_fpi_pipeline_2026-05-14.md
Tests:   T1–T5
Run with: pytest tests/test_l01_fpi_pipeline.py -v
"""

import pathlib
import sys

import numpy as np
import pytest

# Ensure repo root and src/processing are on sys.path so that
# annular_reduction is importable as a flat module name (T5).
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_SRC_PROC  = _REPO_ROOT / "src" / "processing"
for _p in (str(_REPO_ROOT), str(_SRC_PROC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _make_test_cal_result(t_m: float = 20.106e-3) -> "CalibrationResult":
    """Return a CalibrationResult with realistic WindCube values."""
    from windcube.fpi_pipeline import CalibrationResult
    return CalibrationResult(
        t_m               = t_m,
        sigma_t_m         = 5e-9,
        alpha             = 1.6071e-4,
        sigma_alpha       = 2e-8,
        R_refl            = 0.53,
        sigma_R_refl      = 0.005,
        R2                = 0.625,
        sigma_R2          = 0.005,
        I0                = 2000.0,
        sigma_I0          = 10.0,
        I1                = 0.01,
        sigma_I1          = 0.001,
        I2                = 0.005,
        sigma_I2          = 0.001,
        sigma0            = 0.553,
        sigma_sigma0      = 0.01,
        sigma1            = 0.0,
        sigma2            = 0.0,
        B                 = 150.0,
        sigma_B           = 5.0,
        ne_ratio          = 0.509,
        sigma_ne_ratio    = 0.01,
        epsilon_cal       = 0.2332,
        sigma_epsilon_cal = 1e-4,
        chi2_red          = 1.02,
        converged         = True,
        n_bins_used       = 450,
    )


def _make_test_fringe_profile(r_max: float = 150.0):
    """
    Return a minimal FringeProfile-like object for T5.

    Uses SimpleNamespace so that FringeProfile's many required fields
    need not all be specified — to_h05_fringe_profile only accesses
    r_grid, profile, sigma_profile, and masked.
    """
    from types import SimpleNamespace
    n = 500
    r_grid = np.linspace(5.0, r_max, n)
    return SimpleNamespace(
        r_grid        = r_grid,
        profile       = np.full(n, 800.0),
        sigma_profile = np.full(n, 15.0),
        masked        = np.zeros(n, dtype=bool),
        peak_fits_r2  = [],
    )


# ---------------------------------------------------------------------------
# T1 — Module imports cleanly (spec §7 T1)
# ---------------------------------------------------------------------------

def test_fpi_pipeline_imports():
    """fpi_pipeline module must exist and expose the required public names."""
    from windcube import fpi_pipeline
    assert hasattr(fpi_pipeline, "process_cal_frame")
    assert hasattr(fpi_pipeline, "process_science_frame")
    assert hasattr(fpi_pipeline, "average_calibrations")
    assert hasattr(fpi_pipeline, "CalibrationResult")
    assert hasattr(fpi_pipeline, "MasterCalibration")
    assert hasattr(fpi_pipeline, "AirglowResult")


# ---------------------------------------------------------------------------
# T2 — average_calibrations: single frame (spec §7 T2)
# ---------------------------------------------------------------------------

def test_average_calibrations_single():
    """Averaging a single CalibrationResult must return it verbatim."""
    from windcube.fpi_pipeline import average_calibrations
    cal = _make_test_cal_result()
    mc  = average_calibrations([cal])
    assert mc.n_frames_averaged == 1
    assert abs(mc.t_m - cal.t_m) < 1e-12
    assert mc.n_converged == 1


# ---------------------------------------------------------------------------
# T3 — average_calibrations: five frames (spec §7 T3)
# ---------------------------------------------------------------------------

def test_average_calibrations_five():
    """Mean of five frames must match numpy arithmetic mean."""
    from windcube.fpi_pipeline import average_calibrations
    cals = [_make_test_cal_result(t_m=20.106e-3 + i * 1e-9) for i in range(5)]
    mc   = average_calibrations(cals)
    assert mc.n_frames_averaged == 5
    expected_t_m = np.mean([c.t_m for c in cals])
    assert abs(mc.t_m - expected_t_m) < 1e-15


# ---------------------------------------------------------------------------
# T4 — master_cal_to_h06_cal: fields map correctly (spec §7 T4)
# ---------------------------------------------------------------------------

def test_master_cal_to_h06_cal():
    """_CalResult fields must match the MasterCalibration they came from."""
    from windcube.fpi_pipeline import average_calibrations, master_cal_to_h06_cal
    mc     = average_calibrations([_make_test_cal_result()])
    h06cal = master_cal_to_h06_cal(mc)
    assert abs(h06cal.t_m          - mc.t_m)          < 1e-12
    assert abs(h06cal.R_refl       - mc.R_refl)        < 1e-12
    assert abs(h06cal.epsilon_cal  - mc.epsilon_cal)   < 1e-12


# ---------------------------------------------------------------------------
# T5 — to_h05_fringe_profile: shape and r_max respected (spec §7 T5)
# ---------------------------------------------------------------------------

def test_to_h05_fringe_profile():
    """r_max_px must clip the profile, r_max_px field must be set correctly."""
    from windcube.fpi_pipeline import to_h05_fringe_profile
    # Verify annular_reduction.FringeProfile is importable (re-export test)
    from annular_reduction import FringeProfile  # noqa: F401

    fp    = _make_test_fringe_profile(r_max=150.0)
    h05fp = to_h05_fringe_profile(fp, r_max_px=110.0)

    assert np.all(h05fp.r_grid <= 110.0)
    assert h05fp.r_max_px == 110.0
    assert len(h05fp.profile) == len(h05fp.r_grid)
