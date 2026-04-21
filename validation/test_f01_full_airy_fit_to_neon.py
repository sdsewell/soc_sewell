"""
Tests for F01 — Full Airy Fit to Neon Calibration Image.

Spec:    specs/F01_full_airy_fit_to_neon_image_2026-04-21.md
Tests:   T01–T08 (spec §10)
Run with:
    pytest validation/test_f01_full_airy_fit_to_neon.py -v
"""

import os
import pytest
import numpy as np

from src.fpi.f01_full_airy_fit_to_neon_image_2026_04_21 import (
    fit_neon_fringe,
    TolanskyResult,
    CalibrationFitFlags,
    CalibrationResult,
)
from src.fpi.m01_airy_forward_model_2026_04_05 import airy_modified, airy_ideal
from src.fpi.m03_annular_reduction_2026_04_06 import FringeProfile, QualityFlags
from src.constants import NE_WAVELENGTH_1_M, D_25C_MM, PLATE_SCALE_RPX

# ---------------------------------------------------------------------------
# Canonical true parameters (T01 ground truth, spec §10 T01)
# ---------------------------------------------------------------------------
_TRUE = dict(
    R      = 0.53,
    alpha  = 1.6071e-4,
    sigma0 = 0.5,
    sigma1 = 0.0,
    sigma2 = 0.0,
    I0     = 1000.0,
    I1     = -0.1,
    I2     = 0.005,
    B      = 300.0,
    d      = 20.0006e-3,
    lam    = 640.2248e-9,
)

_R_MAX       = 110.0
_N_BINS      = 500
_RNG_SEED    = 42
_N_PIX_BIN   = 100   # realistic pixels-per-bin (≈ π·R_max² / N_bins)


# ---------------------------------------------------------------------------
# Helper: build a minimal FringeProfile from synthetic Airy data
# ---------------------------------------------------------------------------

def _make_fringe(
    true: dict = None,
    noise_scale: float = 1.0,
    rng_seed: int = _RNG_SEED,
    quality_flags: int = QualityFlags.GOOD,
) -> tuple:
    """
    Return (FringeProfile, profile_true) for the given true params.

    noise_scale multiplies the Poisson noise amplitude added to the data.
    sigma_profile is always estimated from the NOISELESS profile so that
    inflating noise_scale raises chi2_red without changing the uncertainty
    weights (T03 design).
    """
    if true is None:
        true = _TRUE

    rng = np.random.default_rng(rng_seed)
    r   = np.linspace(1.0, _R_MAX - 1.0, _N_BINS)

    profile_true = (
        airy_modified(
            r, true["lam"], true["d"], true["R"], true["alpha"], 1.0, _R_MAX,
            true["I0"], true["I1"], true["I2"],
            true["sigma0"], true["sigma1"], true["sigma2"],
        )
        + true["B"]
    )

    # Per-bin SEM: Poisson std / sqrt(N_pix).  sigma_profile is computed from
    # the NOISELESS profile so that inflating noise_scale (T03) raises chi2_red
    # without changing the uncertainty weights.
    sigma_profile = np.sqrt(np.maximum(profile_true, 1.0)) / np.sqrt(_N_PIX_BIN)

    # Noise added to the data: noise_scale × one SEM per bin
    noise = rng.normal(0.0, noise_scale * sigma_profile)
    profile_noisy = profile_true + noise

    n = _N_BINS
    fp = FringeProfile(
        profile           = profile_noisy,
        sigma_profile     = sigma_profile,
        two_sigma_profile = 2.0 * sigma_profile,
        r_grid            = r,
        r2_grid           = r ** 2,
        n_pixels          = np.full(n, _N_PIX_BIN, dtype=int),
        masked            = np.zeros(n, dtype=bool),
        cx                = 0.0,
        cy                = 0.0,
        sigma_cx          = 0.1,
        sigma_cy          = 0.1,
        two_sigma_cx      = 0.2,
        two_sigma_cy      = 0.2,
        seed_source       = "provided",
        stage1_cx         = 0.0,
        stage1_cy         = 0.0,
        cost_at_min       = 0.0,
        quality_flags     = quality_flags,
        sparse_bins       = False,
        r_min_px          = 0.0,
        r_max_px          = _R_MAX,
        n_bins            = n,
        n_subpixels       = 1,
        sigma_clip        = 3.0,
        image_shape       = (256, 256),
        peak_fits         = [],
        dark_subtracted   = False,
        dark_n_frames     = 0,
    )
    return fp, profile_true


def _make_tolansky(true: dict = None) -> TolanskyResult:
    if true is None:
        true = _TRUE
    return TolanskyResult(
        t_m         = true["d"],
        alpha_rpx   = true["alpha"],
        epsilon_640 = 0.7735,
        epsilon_638 = 0.2711,
        epsilon_cal = 0.5000,
    )


# ---------------------------------------------------------------------------
# T01 — Synthetic neon fringe recovery: all 9 params within 2σ of truth
# ---------------------------------------------------------------------------

def test_t01_parameter_recovery():
    """Recover all 9 free params from a Poisson-noise synthetic profile."""
    fp, _ = _make_fringe()
    tol   = _make_tolansky()
    res   = fit_neon_fringe(fp, tol)

    mapping = {
        "R":      (res.R_refl,    res.two_sigma_R_refl),
        "alpha":  (res.alpha,     res.two_sigma_alpha),
        "I0":     (res.I0,        res.two_sigma_I0),
        "I1":     (res.I1,        res.two_sigma_I1),
        "I2":     (res.I2,        res.two_sigma_I2),
        "sigma0": (res.sigma0,    res.two_sigma_sigma0),
        "sigma1": (res.sigma1,    res.two_sigma_sigma1),
        "sigma2": (res.sigma2,    res.two_sigma_sigma2),
        "B":      (res.B,         res.two_sigma_B),
    }

    for name, (fitted, two_sig) in mapping.items():
        truth = _TRUE[name]
        err   = abs(fitted - truth)
        assert err <= two_sig, (
            f"T01 FAIL: {name}: |{fitted:.6g} − {truth:.6g}| = {err:.4g} "
            f"> 2σ = {two_sig:.4g}"
        )

    # Report for task completion summary
    print(f"\nT01 chi2_red = {res.chi2_reduced:.4f}")
    for name, (fitted, two_sig) in mapping.items():
        print(f"  {name:8s}: fitted={fitted:.6g}  truth={_TRUE[name]:.6g}  "
              f"sigma={two_sig/2:.4g}")


# ---------------------------------------------------------------------------
# T02 — Gap d is passed through unchanged from TolanskyResult
# ---------------------------------------------------------------------------

def test_t02_gap_fixed():
    """result.t_m must equal tolansky.t_m exactly (not moved by fit)."""
    fp, _  = _make_fringe()
    tol    = _make_tolansky()
    res    = fit_neon_fringe(fp, tol)
    assert res.t_m == tol.t_m, (
        f"T02 FAIL: result.t_m={res.t_m} != tolansky.t_m={tol.t_m}"
    )


# ---------------------------------------------------------------------------
# T03 — CHI2_HIGH flag when noise is 10× larger than sigma weights
# ---------------------------------------------------------------------------

def test_t03_chi2_high_flag():
    """Adding 10× Poisson noise while keeping sigma at 1× level → CHI2_HIGH."""
    fp, _  = _make_fringe(noise_scale=10.0)
    tol    = _make_tolansky()
    res    = fit_neon_fringe(fp, tol)
    assert res.quality_flags & CalibrationFitFlags.CHI2_HIGH, (
        f"T03 FAIL: CHI2_HIGH not set; chi2_red={res.chi2_reduced:.4f}"
    )


# ---------------------------------------------------------------------------
# T04 — Staged LM: chi² is monotone non-increasing across A→B→C→D
# ---------------------------------------------------------------------------

def test_t04_monotone_chi2():
    """chi2 after each stage must be <= chi2 from previous stage."""
    fp, _  = _make_fringe()
    tol    = _make_tolansky()
    res    = fit_neon_fringe(fp, tol)
    cs     = res.chi2_stages   # [chi2_A, chi2_B, chi2_C, chi2_D]
    assert len(cs) == 4, f"Expected 4 chi2 stages, got {len(cs)}"
    for i in range(1, 4):
        assert cs[i] <= cs[i - 1] + 1e-9, (
            f"T04 FAIL: chi2 not monotone at stage {i}: "
            f"chi2[{i}]={cs[i]:.6f} > chi2[{i-1}]={cs[i-1]:.6f}"
        )


# ---------------------------------------------------------------------------
# T05 — Real 120 s neon calibration image (skip if file absent)
# ---------------------------------------------------------------------------

_CAL_IMAGE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "tests", "data", "cal_image_120s.bin"
)

@pytest.mark.skipif(
    not os.path.exists(_CAL_IMAGE_PATH),
    reason="requires real image: tests/data/cal_image_120s.bin",
)
def test_t05_real_image():
    """chi2_red < 3.0 and converged=True on the real 120 s calibration image."""
    import struct
    with open(_CAL_IMAGE_PATH, "rb") as f:
        data = f.read()
    n = len(data) // 2
    image = np.array(struct.unpack(f"{n}H", data), dtype=np.float64).reshape(256, 256)

    from src.fpi.m03_annular_reduction_2026_04_06 import reduce_calibration_frame
    fp  = reduce_calibration_frame(image, r_max_px=_R_MAX, n_bins=_N_BINS)
    tol = _make_tolansky()
    res = fit_neon_fringe(fp, tol)

    assert res.converged, "T05 FAIL: converged=False on real image"
    assert res.chi2_reduced < 3.0, (
        f"T05 FAIL: chi2_red={res.chi2_reduced:.4f} >= 3.0"
    )


# ---------------------------------------------------------------------------
# T06 — sigma0=sigma1=sigma2=0: airy_modified equals airy_ideal
# ---------------------------------------------------------------------------

def test_t06_zero_psf_identity():
    """airy_modified with sigma0=sigma1=sigma2=0 must equal airy_ideal."""
    from src.fpi.m01_airy_forward_model_2026_04_05 import InstrumentParams
    p   = InstrumentParams(sigma0=0.0, sigma1=0.0, sigma2=0.0)
    r   = np.linspace(0.0, p.r_max, 300)
    lam = NE_WAVELENGTH_1_M

    A_mod  = airy_modified(r, lam, p.t, p.R_refl, p.alpha, p.n, p.r_max,
                           p.I0, p.I1, p.I2, 0.0, 0.0, 0.0)
    A_ideal = airy_ideal(r, lam, p.t, p.R_refl, p.alpha, p.n, p.r_max,
                         p.I0, p.I1, p.I2)

    max_diff = float(np.max(np.abs(A_mod - A_ideal)))
    assert max_diff < 1e-10, (
        f"T06 FAIL: max|airy_modified − airy_ideal| = {max_diff:.2e} >= 1e-10"
    )


# ---------------------------------------------------------------------------
# T07 — R_AT_BOUND flag fires when R hits the upper bound
# ---------------------------------------------------------------------------

def test_t07_r_at_bound():
    """Profile generated with R_true=0.97 forces R above 0.95 → R_AT_BOUND."""
    true_high_R = dict(_TRUE, R=0.97)
    fp, _  = _make_fringe(true=true_high_R)
    tol    = _make_tolansky()
    # R_init=0.94 (close to true) to aid convergence
    res    = fit_neon_fringe(fp, tol, R_init=0.94)
    assert res.quality_flags & CalibrationFitFlags.R_AT_BOUND, (
        f"T07 FAIL: R_AT_BOUND not set; R_refl={res.R_refl:.4f}"
    )


# ---------------------------------------------------------------------------
# T08 — two_sigma_ fields equal exactly 2 × sigma_ for all params
# ---------------------------------------------------------------------------

def test_t08_two_sigma_exact():
    """two_sigma_X == 2.0 * sigma_X must hold exactly for all 9 params."""
    fp,  _ = _make_fringe()
    tol    = _make_tolansky()
    res    = fit_neon_fringe(fp, tol)

    pairs = [
        ("R_refl",  res.sigma_R_refl,  res.two_sigma_R_refl),
        ("alpha",   res.sigma_alpha,   res.two_sigma_alpha),
        ("I0",      res.sigma_I0,      res.two_sigma_I0),
        ("I1",      res.sigma_I1,      res.two_sigma_I1),
        ("I2",      res.sigma_I2,      res.two_sigma_I2),
        ("sigma0",  res.sigma_sigma0,  res.two_sigma_sigma0),
        ("sigma1",  res.sigma_sigma1,  res.two_sigma_sigma1),
        ("sigma2",  res.sigma_sigma2,  res.two_sigma_sigma2),
        ("B",       res.sigma_B,       res.two_sigma_B),
    ]
    for name, sigma, two_sigma in pairs:
        expected = 2.0 * sigma
        assert two_sigma == expected, (
            f"T08 FAIL: two_sigma_{name}={two_sigma!r} != 2×sigma={expected!r}"
        )
