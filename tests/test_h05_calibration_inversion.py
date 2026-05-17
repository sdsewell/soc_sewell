"""
Tests for H05 Staged Calibration Inversion.
Spec:    specs/H05_calibration_inversion_2026-05-05.md
Tests:   T1–T8
Run with: pytest tests/test_h05_calibration_inversion.py -v
"""

import types
from pathlib import Path

import numpy as np
import pytest

from src.fpi import InstrumentParams
from src.fpi.m02_calibration_synthesis_2026_05_05 import synthesise_calibration_image
from src.fpi.archive.m03_annular_reduction_2026_04_06 import annular_reduce
from src.fpi.m05_calibration_inversion_2026_05_05 import (
    CalibrationResult,
    FitConfig,
    FitFlags,
    fit_calibration_fringe,
)
from windcube.constants import NE_WAVELENGTH_1_AIR_M


# ---------------------------------------------------------------------------
# Module-level fixtures (shared across tests to avoid repeated expensive fits)
# ---------------------------------------------------------------------------

def _build_tolansky_stub(params: InstrumentParams) -> types.SimpleNamespace:
    """Build a Tolansky stub with exact values from InstrumentParams."""
    tol = types.SimpleNamespace()
    tol.d_m          = float(params.t)
    tol.alpha_rad_px = float(params.alpha)
    tol.eps1         = float((2.0 * params.t / NE_WAVELENGTH_1_AIR_M) % 1.0)
    return tol


@pytest.fixture(scope="module")
def _base_params():
    return InstrumentParams()


@pytest.fixture(scope="module")
def cal_fringe_profile(_base_params):
    """Noiseless calibration FringeProfile from InstrumentParams defaults."""
    cal_m02 = synthesise_calibration_image(
        _base_params, add_noise=False,
        cx=_base_params.r_max, cy=_base_params.r_max,
    )
    return annular_reduce(
        cal_m02["image_2d"],
        cx=_base_params.r_max,
        cy=_base_params.r_max,
        sigma_cx=0.05,
        sigma_cy=0.05,
        r_max_px=_base_params.r_max,
        n_bins=150,
    )


@pytest.fixture(scope="module")
def cal_tolansky_stub(_base_params):
    return _build_tolansky_stub(_base_params)


@pytest.fixture(scope="module")
def cal_result(cal_fringe_profile, cal_tolansky_stub):
    """CalibrationResult from noiseless synthetic calibration (module scope)."""
    config = FitConfig(tolansky=cal_tolansky_stub)
    return fit_calibration_fringe(cal_fringe_profile, config)


# ---------------------------------------------------------------------------
# T1 — FitConfig resolves Tolansky priors correctly
# ---------------------------------------------------------------------------

def test_fitconfig_resolves_tolansky():
    """
    When a TwoLineResult is passed, t_init and alpha_init must come from
    Tolansky, and bounds must be tightened relative to INSTRUMENT_DEFAULTS.
    """
    tol = types.SimpleNamespace()
    tol.d_m          = 20.008e-3
    tol.alpha_rad_px = 1.607e-4
    tol.eps1         = 0.34

    config   = FitConfig(tolansky=tol)
    resolved = config.resolve(profile=None)

    # t_init must equal Tolansky d_m
    assert abs(resolved["t_m"][0] - 20.008e-3) < 1e-9

    # t bounds must be narrower than instrument defaults (19.95e-3 to 20.07e-3)
    t_lo, t_hi = resolved["t_m"][1], resolved["t_m"][2]
    assert (t_hi - t_lo) < (20.07e-3 - 19.95e-3), \
        "Tolansky-tightened t bounds should be narrower than instrument defaults"

    # alpha bounds must be ±5% of Tolansky alpha
    a_lo, a_hi = resolved["alpha"][1], resolved["alpha"][2]
    assert abs(a_lo - 1.607e-4 * 0.95) < 1e-8
    assert abs(a_hi - 1.607e-4 * 1.05) < 1e-8


# ---------------------------------------------------------------------------
# T2 — chi2_by_stage is monotonically non-increasing
# ---------------------------------------------------------------------------

def test_chi2_monotone(cal_result):
    """chi2 must not increase by more than 5% from stage to stage."""
    stages = cal_result.chi2_by_stage
    assert len(stages) == 4
    for i in range(len(stages) - 1):
        assert stages[i + 1] <= stages[i] * 1.05, (
            f"chi2 increased from stage {i+1} ({stages[i]:.4f}) "
            f"to stage {i+2} ({stages[i+1]:.4f})"
        )


# ---------------------------------------------------------------------------
# T3 — All two_sigma fields are exactly 2 × sigma (S04 compliance)
# ---------------------------------------------------------------------------

def test_two_sigma_convention(cal_result):
    """S04: two_sigma_X must equal exactly 2.0 × sigma_X for all 11 fields."""
    params_to_check = [
        "t_m", "R_refl", "alpha",
        "I0", "I1", "I2",
        "sigma0", "sigma1", "sigma2",
        "B", "epsilon_cal",
    ]
    for p in params_to_check:
        sigma     = getattr(cal_result, f"sigma_{p}")
        two_sigma = getattr(cal_result, f"two_sigma_{p}")
        assert abs(two_sigma - 2.0 * sigma) < 1e-15, (
            f"two_sigma_{p} = {two_sigma} ≠ 2 × sigma_{p} = {2 * sigma}"
        )


# ---------------------------------------------------------------------------
# T4 — epsilon_cal computed correctly from t_m
# ---------------------------------------------------------------------------

def test_epsilon_cal_computation(cal_result):
    """epsilon_cal = (2 * t_m / NE_WAVELENGTH_1_AIR_M) mod 1."""
    expected = (2.0 * cal_result.t_m / NE_WAVELENGTH_1_AIR_M) % 1.0
    assert abs(cal_result.epsilon_cal - expected) < 1e-12, (
        f"epsilon_cal={cal_result.epsilon_cal:.6f}, expected={expected:.6f}"
    )


# ---------------------------------------------------------------------------
# T5 — Round-trip: inject known params, recover within tolerance
# ---------------------------------------------------------------------------

def test_round_trip_recovery(cal_result, _base_params):
    """
    Noiseless synthetic round-trip. Recovered parameters must match
    InstrumentParams defaults to within instrument-relevant tolerances.

    Tolerance notes:
      t_m:    < 1e-9 m  (sub-µm) — Tolansky seeds to this accuracy
      R_refl: < 0.05 — pixel-averaging in 2D→1D reduction shifts effective
              R_refl by ~2% systematically (inverse-crime effect at integer cx)
      alpha:  < 1e-6 rad/px
      sigma0: < 0.1 px — pixel-averaging creates same systematic as R_refl
    chi2_reduced must be in [0.5, 3.0].
    """
    params = _base_params
    assert abs(cal_result.t_m - params.t) < 1e-9, \
        f"t_m error = {abs(cal_result.t_m - params.t):.2e} m"
    assert abs(cal_result.R_refl - params.R_refl) < 0.05, \
        f"R_refl error = {abs(cal_result.R_refl - params.R_refl):.4f}"
    assert abs(cal_result.alpha - params.alpha) < 1e-6, \
        f"alpha error = {abs(cal_result.alpha - params.alpha):.2e} rad/px"
    assert abs(cal_result.sigma0 - params.sigma0) < 0.1, \
        f"sigma0 error = {abs(cal_result.sigma0 - params.sigma0):.4f} px"
    assert 0.5 < cal_result.chi2_reduced < 3.0, \
        f"chi2_reduced = {cal_result.chi2_reduced:.3f} outside [0.5, 3.0]"


# ---------------------------------------------------------------------------
# T6 — Without Tolansky: TOLANSKY_NOT_PROVIDED flag set
# ---------------------------------------------------------------------------

def test_no_tolansky_flag(cal_fringe_profile):
    """
    Running without Tolansky must set TOLANSKY_NOT_PROVIDED.

    Note: InstrumentParams.t (20.0006e-3 m) differs from ETALON_GAP_M
    (20.008e-3 m) by 7.4 µm ≈ 23 FSR periods. Without Tolansky the LM
    optimizer starts in the wrong period and cannot converge — this is the
    FSR-ambiguity risk documented in H05 §2.1. The test only verifies the
    flag is set; chi2 and convergence are not checked in this scenario.
    """
    result = fit_calibration_fringe(cal_fringe_profile, FitConfig())
    assert result.quality_flags & FitFlags.TOLANSKY_NOT_PROVIDED, \
        "TOLANSKY_NOT_PROVIDED flag must be set when tolansky=None"


# ---------------------------------------------------------------------------
# T7 — Noisy synthetic: chi2_reduced near 1, sigma_t < 1 µm
# ---------------------------------------------------------------------------

def test_noisy_round_trip():
    """
    Synthesise with Poisson noise. chi2_reduced in [0.7, 2.0]; sigma_t_m < 1 µm.
    """
    params = InstrumentParams()
    cal_m02 = synthesise_calibration_image(
        params, add_noise=True,
        cx=params.r_max, cy=params.r_max,
        rng=np.random.default_rng(42),
    )
    fp = annular_reduce(
        cal_m02["image_2d"],
        cx=params.r_max,
        cy=params.r_max,
        sigma_cx=0.05,
        sigma_cy=0.05,
        r_max_px=params.r_max,
        n_bins=150,
    )
    tol_stub = _build_tolansky_stub(params)
    config   = FitConfig(tolansky=tol_stub)
    cal      = fit_calibration_fringe(fp, config)

    assert 0.7 < cal.chi2_reduced < 2.0, \
        f"chi2_reduced = {cal.chi2_reduced:.3f} outside [0.7, 2.0] for noisy data"
    assert cal.sigma_t_m < 1e-6, \
        f"sigma_t_m = {cal.sigma_t_m:.2e} m >= 1 µm"


# ---------------------------------------------------------------------------
# T8 — Real FlatSat data (skip if file absent)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not Path("data/flatsat_cal_profile.npz").exists(),
    reason="FlatSat calibration profile not available",
)
def test_flatsat_data():
    """
    Run on real FlatSat calibration profile.
    Expected: t_m near 20.008e-3 m, R near 0.53, chi2 in [0.8, 3.0].
    """
    from src.fpi.archive.m03_annular_reduction_2026_04_06 import FringeProfile
    from src.two_d_one_d_reduction.tolansky import TolanskyPipeline

    fp  = FringeProfile.load("data/flatsat_cal_profile.npz")
    tol = TolanskyPipeline(fp).run()
    cal = fit_calibration_fringe(fp, FitConfig(tolansky=tol))

    assert abs(cal.t_m - 20.008e-3) < 0.05e-3, \
        f"t_m = {cal.t_m*1e3:.4f} mm, expected near 20.008 mm"
    assert 0.40 < cal.R_refl < 0.70, \
        f"R_refl = {cal.R_refl:.3f} outside [0.40, 0.70]"
    assert 0.8 < cal.chi2_reduced < 3.0, \
        f"chi2_reduced = {cal.chi2_reduced:.3f} outside [0.8, 3.0]"
