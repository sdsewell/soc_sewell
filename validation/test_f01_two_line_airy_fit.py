"""
Tests for F01 v3 — Two-Line Airy Fit to Neon Calibration Profile.

Spec:    specs/F01_full_airy_fit_to_neon_image_2026-04-22.md
Tests:   T01–T11 (spec §11)
Run with:
    pytest validation/test_f01_two_line_airy_fit.py -v
"""

import os
from types import SimpleNamespace

import numpy as np
import pytest

from src.fpi.f01_full_airy_fit_to_neon_image_2026_04_22 import (
    CalibrationFitFlags,
    CalibrationResult,
    TolanskyResult,
    fit_neon_fringe,
)
from src.fpi.m01_airy_forward_model_2026_04_05 import airy_ideal, airy_modified
from src.constants import NE_WAVELENGTH_1_M, NE_WAVELENGTH_2_M

# ---------------------------------------------------------------------------
# Canonical truth parameters (T01 ground truth, spec §11)
# ---------------------------------------------------------------------------
_TRUE = dict(
    R      = 0.72,
    alpha  = 1.6133e-4,
    I0     = 2200.0,
    I1     = -0.1,
    I2     = 0.005,
    sigma0 = 0.5,
    sigma1 = 0.0,
    sigma2 = 0.0,
    Y_A    = 1.0,
    Y_B    = 0.58,
    B      = 6100.0,
    d      = 20.0006e-3,
    r_max  = 110.0,
)

_RNG_SEED = 42


# ---------------------------------------------------------------------------
# Synthetic fringe generator (spec §11 _make_synthetic_profile)
# ---------------------------------------------------------------------------

def _make_synthetic_profile(R, alpha, I0, I1, I2, s0, s1, s2,
                             Y_A, Y_B, B, d, r_max, rng_seed=42):
    rng     = np.random.default_rng(rng_seed)
    r       = np.linspace(1.0, r_max, 500)
    A_fine_A = airy_modified(r, NE_WAVELENGTH_1_M, d, R, alpha, 1.0,
                             r_max, I0, I1, I2, s0, s1, s2)
    A_fine_B = airy_modified(r, NE_WAVELENGTH_2_M, d, R, alpha, 1.0,
                             r_max, I0, I1, I2, s0, s1, s2)
    signal  = Y_A * A_fine_A + Y_B * A_fine_B + B
    noisy   = rng.poisson(np.maximum(signal, 1)).astype(np.float32)
    sigma   = np.maximum(np.sqrt(noisy), 1.0).astype(np.float32)
    return r, noisy, sigma


def _make_profile(r, noisy, sigma, r_max):
    """Wrap arrays in a FringeProfile-compatible SimpleNamespace."""
    return SimpleNamespace(
        r_grid        = r,
        r2_grid       = r ** 2,
        profile       = noisy,
        sigma_profile = sigma,
        masked        = np.zeros(len(r), dtype=bool),
        r_max_px      = float(r_max),
        quality_flags = 0,
    )


def _make_tolansky(d=20.0006e-3, alpha=1.6133e-4, eps640=0.0,
                   eps638=0.0, eps_cal=0.0):
    return TolanskyResult(
        t_m         = d,
        alpha_rpx   = alpha,
        epsilon_640 = eps640,
        epsilon_638 = eps638,
        epsilon_cal = eps_cal,
    )


def _run_t01():
    """Run the canonical T01 synthetic fit and return the result."""
    t = _TRUE
    r, noisy, sigma = _make_synthetic_profile(
        R=t["R"], alpha=t["alpha"], I0=t["I0"], I1=t["I1"], I2=t["I2"],
        s0=t["sigma0"], s1=t["sigma1"], s2=t["sigma2"],
        Y_A=t["Y_A"], Y_B=t["Y_B"], B=t["B"],
        d=t["d"], r_max=t["r_max"], rng_seed=_RNG_SEED,
    )
    profile  = _make_profile(r, noisy, sigma, t["r_max"])
    tolansky = _make_tolansky(d=t["d"], alpha=t["alpha"])
    return fit_neon_fringe(profile, tolansky)


# ---------------------------------------------------------------------------
# T01: All 11 free params recovered within 2σ of truth
# ---------------------------------------------------------------------------

def test_T01_param_recovery():
    result = _run_t01()
    t = _TRUE

    param_map = [
        ("R",      result.R_refl,   result.sigma_R_refl,   t["R"]),
        ("alpha",  result.alpha,    result.sigma_alpha,    t["alpha"]),
        ("I0",     result.I0,       result.sigma_I0,       t["I0"]),
        ("I1",     result.I1,       result.sigma_I1,       t["I1"]),
        ("I2",     result.I2,       result.sigma_I2,       t["I2"]),
        ("sigma0", result.sigma0,   result.sigma_sigma0,   t["sigma0"]),
        ("sigma1", result.sigma1,   result.sigma_sigma1,   t["sigma1"]),
        ("sigma2", result.sigma2,   result.sigma_sigma2,   t["sigma2"]),
        ("Y_A",    result.Y_A,      result.sigma_Y_A,      t["Y_A"]),
        ("Y_B",    result.Y_B,      result.sigma_Y_B,      t["Y_B"]),
        ("B",      result.B,        result.sigma_B,        t["B"]),
    ]

    print("\nT01 parameter recovery:")
    print(f"  chi2_reduced = {result.chi2_reduced:.4f}")
    for name, fit, sig, truth in param_map:
        deviation = abs(fit - truth)
        n_sigma   = deviation / sig if sig > 0 else float("inf")
        status    = "OK" if deviation <= 2.0 * sig else "FAIL"
        print(f"  {name:8s}: fit={fit:.6g}  truth={truth:.6g}  "
              f"1σ={sig:.3g}  |Δ|={deviation:.3g}  ({n_sigma:.1f}σ) [{status}]")

    for name, fit, sig, truth in param_map:
        assert abs(fit - truth) <= 2.0 * sig + 1e-10 * abs(truth), (
            f"{name}: fit={fit:.6g} truth={truth:.6g} "
            f"deviation={abs(fit-truth):.3g} > 2σ={2*sig:.3g}"
        )

    assert result.converged
    assert result.chi2_reduced < 10.0


# ---------------------------------------------------------------------------
# T02: Gap passed through exactly — not fitted
# ---------------------------------------------------------------------------

def test_T02_gap_passthrough():
    t        = _TRUE
    r, noisy, sigma = _make_synthetic_profile(
        **{k: t[k] for k in ["R","I0","I1","I2","Y_A","Y_B","B","d","r_max"]},
        alpha=t["alpha"], s0=t["sigma0"], s1=t["sigma1"], s2=t["sigma2"],
    )
    profile  = _make_profile(r, noisy, sigma, t["r_max"])
    tolansky = _make_tolansky(d=t["d"], alpha=t["alpha"])
    result   = fit_neon_fringe(profile, tolansky)
    assert result.t_m == tolansky.t_m


# ---------------------------------------------------------------------------
# T03: CHI2_HIGH flag set when data noise >> reported sigma
# ---------------------------------------------------------------------------

def test_T03_chi2_high_with_excess_noise():
    t = _TRUE
    r, noisy, sigma = _make_synthetic_profile(
        R=t["R"], alpha=t["alpha"], I0=t["I0"], I1=t["I1"], I2=t["I2"],
        s0=t["sigma0"], s1=t["sigma1"], s2=t["sigma2"],
        Y_A=t["Y_A"], Y_B=t["Y_B"], B=t["B"],
        d=t["d"], r_max=t["r_max"], rng_seed=_RNG_SEED,
    )
    # Deflate reported sigma by 10× → residuals ~10× larger → chi2 >> 3
    profile  = _make_profile(r, noisy, sigma / 10.0, t["r_max"])
    tolansky = _make_tolansky(d=t["d"], alpha=t["alpha"])
    result   = fit_neon_fringe(profile, tolansky)
    assert result.quality_flags & CalibrationFitFlags.CHI2_HIGH, (
        f"chi2_red={result.chi2_reduced:.2f}; expected CHI2_HIGH"
    )


# ---------------------------------------------------------------------------
# T04: chi2_stages monotone non-increasing (A through E, 5 values)
# ---------------------------------------------------------------------------

def test_T04_chi2_stages_monotone():
    result = _run_t01()
    stages = result.chi2_stages
    assert len(stages) == 5, f"Expected 5 chi2_stages, got {len(stages)}"
    print(f"\nT04 chi2_stages: {[f'{v:.4f}' for v in stages]}")
    for i in range(len(stages) - 1):
        assert stages[i] >= stages[i + 1] - 1e-6, (
            f"chi2_stages not non-increasing: stage {i}={stages[i]:.4f} "
            f"< stage {i+1}={stages[i+1]:.4f}"
        )


# ---------------------------------------------------------------------------
# T05: Real 120 s neon calibration image (skipped if binary absent)
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="requires real binary data — not available in CI")
def test_T05_real_neon_image():
    data_path = "tests/data/cal_image_120s.bin"
    if not os.path.exists(data_path):
        pytest.skip("requires real binary")
    # Placeholder: load, reduce, fit, assert chi2_red < 3.0
    raise NotImplementedError("T05 real-data path not yet wired")


# ---------------------------------------------------------------------------
# T06: sigma0=sigma1=sigma2=0 → airy_modified == airy_ideal (max diff < 1e-10)
# ---------------------------------------------------------------------------

def test_T06_airy_modified_equals_ideal_at_zero_psf():
    r     = np.linspace(1.0, 110.0, 500)
    lam   = NE_WAVELENGTH_1_M
    t     = 20.0006e-3
    R     = 0.72
    alpha = 1.6133e-4
    n     = 1.0
    r_max = 110.0
    I0, I1, I2 = 2200.0, -0.1, 0.005

    ideal  = airy_ideal(r, lam, t, R, alpha, n, r_max, I0, I1, I2)
    modif  = airy_modified(r, lam, t, R, alpha, n, r_max, I0, I1, I2,
                           sigma0=0.0, sigma1=0.0, sigma2=0.0)
    max_diff = float(np.max(np.abs(ideal - modif)))
    assert max_diff < 1e-10, f"max |airy_modified - airy_ideal| = {max_diff:.3e}"


# ---------------------------------------------------------------------------
# T07: R_AT_BOUND flag fires when true R > upper bound
# ---------------------------------------------------------------------------

def test_T07_R_AT_BOUND_flag():
    t = _TRUE
    # True R = 0.97 (above the 0.95 upper bound) — fitter hits the ceiling
    r, noisy, sigma = _make_synthetic_profile(
        R=0.97, alpha=t["alpha"], I0=t["I0"], I1=t["I1"], I2=t["I2"],
        s0=t["sigma0"], s1=t["sigma1"], s2=t["sigma2"],
        Y_A=t["Y_A"], Y_B=t["Y_B"], B=t["B"],
        d=t["d"], r_max=t["r_max"], rng_seed=_RNG_SEED,
    )
    profile  = _make_profile(r, noisy, sigma, t["r_max"])
    tolansky = _make_tolansky(d=t["d"], alpha=t["alpha"])
    result   = fit_neon_fringe(profile, tolansky, R_init=0.94)
    assert result.quality_flags & CalibrationFitFlags.R_AT_BOUND, (
        f"R_AT_BOUND not set; R_refl={result.R_refl:.4f}"
    )


# ---------------------------------------------------------------------------
# T08: two_sigma_ == 2 * sigma_ (exact float equality via __post_init__)
# ---------------------------------------------------------------------------

def test_T08_two_sigma_exact():
    result = _run_t01()
    # A good result must not raise in __post_init__ (it was called at construction)
    result.__post_init__()   # must not raise

    # Manually corrupt one two_sigma_ field and confirm assertion fires
    original = result.two_sigma_R_refl
    result.two_sigma_R_refl = result.sigma_R_refl * 3.0   # wrong
    with pytest.raises(AssertionError):
        result.__post_init__()
    result.two_sigma_R_refl = original  # restore


# ---------------------------------------------------------------------------
# T09: Y_B/Y_A recovered within 5% of truth (0.58)
# ---------------------------------------------------------------------------

def test_T09_YB_ratio_recovery():
    result    = _run_t01()
    ratio_fit = result.intensity_ratio
    ratio_true = _TRUE["Y_B"] / _TRUE["Y_A"]
    err = abs(ratio_fit - ratio_true)
    print(f"\nT09  Y_B/Y_A: fit={ratio_fit:.4f}  truth={ratio_true:.4f}  "
          f"|err|={err:.4f}")
    assert err < 0.05, (
        f"Y_B/Y_A error {err:.4f} exceeds 0.05 tolerance "
        f"(fit={ratio_fit:.4f}, true={ratio_true:.4f})"
    )


# ---------------------------------------------------------------------------
# T10: YB_RATIO_LOW flag fires when Y_B/Y_A < 0.3
# ---------------------------------------------------------------------------

def test_T10_YB_RATIO_LOW_flag():
    t = _TRUE
    # Y_B_true = 0.2 → ratio < 0.3
    r, noisy, sigma = _make_synthetic_profile(
        R=t["R"], alpha=t["alpha"], I0=t["I0"], I1=t["I1"], I2=t["I2"],
        s0=t["sigma0"], s1=t["sigma1"], s2=t["sigma2"],
        Y_A=1.0, Y_B=0.2, B=t["B"],
        d=t["d"], r_max=t["r_max"], rng_seed=_RNG_SEED,
    )
    profile  = _make_profile(r, noisy, sigma, t["r_max"])
    tolansky = _make_tolansky(d=t["d"], alpha=t["alpha"])
    result   = fit_neon_fringe(profile, tolansky)
    ratio    = result.intensity_ratio
    print(f"\nT10  Y_B/Y_A fitted = {ratio:.4f}")
    assert result.quality_flags & CalibrationFitFlags.YB_RATIO_LOW, (
        f"YB_RATIO_LOW not set; Y_B/Y_A={ratio:.4f}"
    )


# ---------------------------------------------------------------------------
# T11: Single-line synthetic (Y_B=0): Y_B_fit < 0.05 and YB_RATIO_LOW set
# ---------------------------------------------------------------------------

def test_T11_single_line_synthetic():
    t = _TRUE
    # Y_B_true = 0 → single-line data; fitter should hit Y_B lower bound
    r, noisy, sigma = _make_synthetic_profile(
        R=t["R"], alpha=t["alpha"], I0=t["I0"], I1=t["I1"], I2=t["I2"],
        s0=t["sigma0"], s1=t["sigma1"], s2=t["sigma2"],
        Y_A=1.0, Y_B=0.0, B=t["B"],
        d=t["d"], r_max=t["r_max"], rng_seed=_RNG_SEED,
    )
    profile  = _make_profile(r, noisy, sigma, t["r_max"])
    tolansky = _make_tolansky(d=t["d"], alpha=t["alpha"])
    result   = fit_neon_fringe(profile, tolansky)
    print(f"\nT11  Y_B_fit={result.Y_B:.4f}  ratio={result.intensity_ratio:.4f}")
    assert result.Y_B < 0.05, f"Y_B_fit={result.Y_B:.4f} should be < 0.05"
    assert result.quality_flags & CalibrationFitFlags.YB_RATIO_LOW, (
        f"YB_RATIO_LOW not set; ratio={result.intensity_ratio:.4f}"
    )
