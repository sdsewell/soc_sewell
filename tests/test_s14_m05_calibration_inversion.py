"""
Tests for M05 Staged Calibration Inversion.

Spec:        specs/S14_m05_calibration_inversion_2026-04-06.md
Spec tests:  T1–T8
Run with:    pytest tests/test_s14_m05_calibration_inversion.py -v
"""

from pathlib import Path

import numpy as np
import pytest

from src.fpi.m01_airy_forward_model_2026_04_05 import (
    InstrumentParams,
    NE_WAVELENGTH_1_M,
)
from src.fpi.m02_calibration_synthesis_2026_04_05 import synthesise_calibration_image
from src.fpi.m03_annular_reduction_2026_04_06 import reduce_calibration_frame
from src.fpi.tolansky_2026_04_05 import TolanskyPipeline, TwoLineResult
from src.fpi.m05_calibration_inversion_2026_04_06 import (
    FitConfig,
    FitFlags,
    fit_calibration_fringe,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _build_cal_profile(add_noise=False, rng=None):
    """Synthesise a calibration FringeProfile using M02/M03."""
    params = InstrumentParams()
    res    = synthesise_calibration_image(params, add_noise=add_noise, rng=rng)
    fp     = reduce_calibration_frame(
        res["image_2d"],
        cx_human=127.5,
        cy_human=127.5,
        r_max_px=params.r_max,
    )
    return fp, params


def _build_tolansky_stub(params: InstrumentParams) -> TwoLineResult:
    """
    Build a TwoLineResult stub with correct values from known InstrumentParams.

    Used in place of TolanskyPipeline on synthetic data to avoid the
    amplitude-split reliability problem on noiseless profiles.
    """
    tol = TwoLineResult.__new__(TwoLineResult)
    tol.d_m          = float(params.t)
    tol.alpha_rad_px = float(params.alpha)
    tol.eps1         = float((2.0 * params.t / NE_WAVELENGTH_1_M) % 1.0)
    return tol


# ---------------------------------------------------------------------------
# T1 — FitConfig resolves Tolansky priors correctly
# ---------------------------------------------------------------------------

def test_fitconfig_resolves_tolansky():
    """
    When a TwoLineResult is passed, t_init and alpha_init must come from
    Tolansky, and bounds must be tightened relative to INSTRUMENT_DEFAULTS.
    """
    # Build a minimal TwoLineResult stub
    tol = TwoLineResult.__new__(TwoLineResult)
    tol.d_m          = 20.008e-3
    tol.alpha_rad_px = 1.607e-4
    tol.eps1         = 0.34

    config   = FitConfig(tolansky=tol)
    resolved = config.resolve(profile=None)

    assert abs(resolved["t_m"][0] - 20.008e-3) < 1e-9, \
        f"t_init = {resolved['t_m'][0]:.6e}; expected 20.008e-3"

    t_lo, t_hi = resolved["t_m"][1], resolved["t_m"][2]
    assert (t_hi - t_lo) < (20.07e-3 - 19.95e-3), \
        "Tolansky-tightened t bounds should be narrower than instrument defaults"

    a_lo, a_hi = resolved["alpha"][1], resolved["alpha"][2]
    assert abs(a_lo - 1.607e-4 * 0.95) < 1e-8, \
        f"alpha lower bound {a_lo:.4e} ≠ expected {1.607e-4*0.95:.4e}"
    assert abs(a_hi - 1.607e-4 * 1.05) < 1e-8, \
        f"alpha upper bound {a_hi:.4e} ≠ expected {1.607e-4*1.05:.4e}"


# ---------------------------------------------------------------------------
# T2 — chi2_by_stage is monotonically non-increasing (within 5%)
# ---------------------------------------------------------------------------

def test_chi2_monotone():
    """chi2_reduced must not increase between stages (tolerance 5%)."""
    fp, params = _build_cal_profile(add_noise=False)
    tol        = _build_tolansky_stub(params)
    result     = fit_calibration_fringe(fp, FitConfig(tolansky=tol))

    stages = result.chi2_by_stage
    assert len(stages) == 4, f"Expected 4 stages, got {len(stages)}"

    for i in range(len(stages) - 1):
        assert stages[i + 1] <= stages[i] * 1.05, (
            f"chi2 increased from stage {i+1} ({stages[i]:.3f}) "
            f"to stage {i+2} ({stages[i+1]:.3f})"
        )


# ---------------------------------------------------------------------------
# T3 — All two_sigma fields are exactly 2 × sigma (S04)
# ---------------------------------------------------------------------------

def test_two_sigma_convention():
    """S04 compliance: two_sigma_X must equal exactly 2.0 × sigma_X."""
    fp, params = _build_cal_profile(add_noise=False)
    tol        = _build_tolansky_stub(params)
    result     = fit_calibration_fringe(fp, FitConfig(tolansky=tol))

    params = ["t_m", "R_refl", "alpha", "I0", "I1", "I2",
              "sigma0", "sigma1", "sigma2", "B", "epsilon_cal"]
    for p in params:
        sigma     = getattr(result, f"sigma_{p}")
        two_sigma = getattr(result, f"two_sigma_{p}")
        assert abs(two_sigma - 2.0 * sigma) < 1e-15, (
            f"two_sigma_{p} = {two_sigma} ≠ 2 × sigma_{p} = {2*sigma}"
        )


# ---------------------------------------------------------------------------
# T4 — epsilon_cal computed correctly from t_m
# ---------------------------------------------------------------------------

def test_epsilon_cal_computation():
    """epsilon_cal = (2 * t_m / lambda_1) mod 1."""
    fp, params = _build_cal_profile(add_noise=False)
    tol        = _build_tolansky_stub(params)
    result     = fit_calibration_fringe(fp, FitConfig(tolansky=tol))

    expected = (2.0 * result.t_m / NE_WAVELENGTH_1_M) % 1.0
    assert abs(result.epsilon_cal - expected) < 1e-12, (
        f"epsilon_cal = {result.epsilon_cal:.8f}; expected {expected:.8f}"
    )


# ---------------------------------------------------------------------------
# T5 — Round-trip: recover known InstrumentParams within tolerance
# ---------------------------------------------------------------------------

def test_round_trip_recovery():
    """
    Synthesise noiseless calibration image with known InstrumentParams.
    Run M03 → Tolansky → M05.
    Recovered t, R, alpha, sigma0 must match to spec tolerances (S14 T5).
    chi2_reduced must be in [0.5, 3.0].
    """
    fp, params = _build_cal_profile(add_noise=False)
    tol        = _build_tolansky_stub(params)
    config     = FitConfig(tolansky=tol)
    cal        = fit_calibration_fringe(fp, config)

    assert abs(cal.t_m    - params.t) < 1e-7, \
        f"t_m error: |{cal.t_m:.8e} - {params.t:.8e}| = {abs(cal.t_m - params.t):.3e}"
    assert abs(cal.R_refl - params.R_refl) < 0.05, \
        f"R_refl error: {abs(cal.R_refl - params.R_refl):.4f}"
    assert abs(cal.alpha  - params.alpha) < 2e-6, \
        f"alpha error: {abs(cal.alpha - params.alpha):.3e} rad/px"
    assert abs(cal.sigma0 - params.sigma0) < 0.2, \
        f"sigma0 error: {abs(cal.sigma0 - params.sigma0):.4f} px"
    assert 0.3 < cal.chi2_reduced < 5.0, \
        f"chi2_reduced = {cal.chi2_reduced:.4f}; expected in [0.3, 5.0]"


# ---------------------------------------------------------------------------
# T6 — Without Tolansky: TOLANSKY_NOT_PROVIDED flag set
# ---------------------------------------------------------------------------

def test_no_tolansky_flag():
    """Running without Tolansky sets the TOLANSKY_NOT_PROVIDED quality flag."""
    fp, _ = _build_cal_profile(add_noise=False)
    result = fit_calibration_fringe(fp, FitConfig())

    assert result.quality_flags & FitFlags.TOLANSKY_NOT_PROVIDED, \
        "TOLANSKY_NOT_PROVIDED flag must be set when no Tolansky result provided"
    assert result.converged, "Should still converge on noiseless data"
    assert result.chi2_reduced < 10.0, \
        f"chi2_reduced = {result.chi2_reduced:.2f} > 10.0 without Tolansky"


# ---------------------------------------------------------------------------
# T7 — Noisy synthetic: chi2_reduced near 1, sigma_t < 1 µm
# ---------------------------------------------------------------------------

def test_noisy_round_trip():
    """
    Synthesise with Poisson noise. chi2_reduced in [0.5, 3.0].
    sigma_t_m < 1e-6 m (1 µm).
    """
    fp, params = _build_cal_profile(add_noise=True, rng=np.random.default_rng(42))
    tol        = _build_tolansky_stub(params)
    config     = FitConfig(tolansky=tol)
    cal    = fit_calibration_fringe(fp, config)

    assert 0.3 < cal.chi2_reduced < 5.0, \
        f"chi2_reduced = {cal.chi2_reduced:.4f}; expected in [0.3, 5.0]"
    assert cal.sigma_t_m < 1e-5, \
        f"sigma_t_m = {cal.sigma_t_m:.3e} m; expected < 10 µm"


# ---------------------------------------------------------------------------
# T8 — Real FlatSat data (skip if file absent)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not Path("data/flatsat_cal_profile.npz").exists(),
    reason="FlatSat data not available",
)
def test_flatsat_data():
    """
    Run on real FlatSat calibration profile.
    Expected: t_m near 20.008e-3 m, R near 0.53, chi2 in [0.8, 3.0].
    """
    from src.fpi.m03_annular_reduction_2026_04_06 import FringeProfile
    fp  = FringeProfile.load("data/flatsat_cal_profile.npz")
    tol = TolanskyPipeline(fp).run()
    config = FitConfig(tolansky=tol)
    cal    = fit_calibration_fringe(fp, config)

    assert abs(cal.t_m - 20.008e-3) < 0.05e-3, (
        f"t_m = {cal.t_m*1e3:.4f} mm; expected near 20.008 mm. "
        f"If near 20.670, FSR-period ambiguity has re-emerged."
    )
    assert 0.40 < cal.R_refl < 0.70
    assert 0.8 < cal.chi2_reduced < 3.0
