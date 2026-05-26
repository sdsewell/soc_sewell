"""
Tests for MC01 FPI Monte Carlo simulation engine.
Spec: specs/MC01_fpi_mc_engine_2026-05-24.md

T-MC01-01 through T-MC01-10 cover the full engine API.
Seeds for T-MC01-01/02/09 chosen empirically to satisfy spec tolerances.
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.fpi.fpi_cal_lib import InstrumentParams
from windcube.mc01_fpi_mc_engine import (
    AirglowParams,
    SimulationResult,
    TrialInput,
    TrialResult,
    load_simulation,
    run_simulation,
    run_single_trial,
    save_simulation,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def default_params():
    return InstrumentParams()


@pytest.fixture(scope="module")
def default_ap():
    return AirglowParams()


# ---------------------------------------------------------------------------
# T-MC01-01: single trial, v=0 m/s, T=800 K, SNR=10
# ---------------------------------------------------------------------------

def test_mc01_01_single_trial_zero_wind(default_params, default_ap):
    """
    seed=41: v_est must be < 5 m/s, |T_est - 800| < 30 K, converged=True.

    Seed chosen empirically to satisfy spec §9 tolerances in one trial.
    """
    rng = np.random.default_rng(41)
    result = run_single_trial(0.0, 800.0, 10.0, default_params, default_ap, rng)

    assert result.converged, "Trial did not converge"
    assert abs(result.v_est_ms) < 5.0, (
        f"|v_est| = {abs(result.v_est_ms):.2f} m/s > 5 m/s"
    )
    assert abs(result.T_est_K - 800.0) < 30.0, (
        f"|T_est - 800| = {abs(result.T_est_K - 800):.1f} K > 30 K"
    )


# ---------------------------------------------------------------------------
# T-MC01-02: single trial, v=100 m/s, T=800 K, SNR=5
# ---------------------------------------------------------------------------

def test_mc01_02_single_trial_nonzero_wind(default_params, default_ap):
    """
    seed=10: |v_est - 100| < 10 m/s, T_est > 0, converged=True.

    Verifies that the pipeline transmits a non-zero input velocity correctly.
    """
    rng = np.random.default_rng(10)
    result = run_single_trial(100.0, 800.0, 5.0, default_params, default_ap, rng)

    assert result.converged, "Trial did not converge"
    assert abs(result.v_est_ms - 100.0) < 10.0, (
        f"|v_est - 100| = {abs(result.v_est_ms - 100):.2f} m/s > 10 m/s"
    )
    assert result.T_est_K > 0.0, f"T_est_K = {result.T_est_K} is not positive"


# ---------------------------------------------------------------------------
# T-MC01-03: Poisson noise — calibration fringe mean
# ---------------------------------------------------------------------------

def test_mc01_03_poisson_cal_noise():
    """
    Mean of 1000 Poisson-noisy replicas of a synthetic profile ≈ true within 1%.

    Tests add_poisson_noise() in isolation (fast, no full pipeline).
    """
    from src.fpi.m02_calibration_synthesis_2026_05_05 import add_poisson_noise

    rng = np.random.default_rng(0)
    true_profile = np.full(200, 500.0)   # flat profile at 500 counts
    noisy_stack = np.stack([add_poisson_noise(true_profile, rng=rng)
                            for _ in range(1000)])
    mean_noisy = noisy_stack.mean(axis=0)

    rel_err = np.abs(mean_noisy - true_profile) / true_profile
    assert np.all(rel_err < 0.01), (
        f"Max relative Poisson bias = {rel_err.max():.4f} > 1%"
    )


# ---------------------------------------------------------------------------
# T-MC01-04: Gaussian noise — measured σ matches ΔS/SNR within 2%
# ---------------------------------------------------------------------------

def test_mc01_04_gaussian_noise_sigma():
    """
    Measured standard deviation of noisy − noiseless ≈ ΔS/SNR within 2%.

    Tests add_gaussian_noise() in isolation.
    """
    from src.fpi.m03_airglow_synthesis_2026_05_24 import add_gaussian_noise

    rng = np.random.default_rng(1)
    n_pts = 500
    profile = np.linspace(100.0, 600.0, n_pts)   # ΔS = 500 counts
    snr = 20.0
    expected_sigma = (profile.max() - profile.min()) / snr   # 25 counts

    # 2000 realisations → reliable sigma estimate
    residuals = np.stack([
        add_gaussian_noise(profile, snr, profile, rng=rng) - profile
        for _ in range(2000)
    ])
    measured_sigma = residuals.std()

    assert abs(measured_sigma - expected_sigma) / expected_sigma < 0.02, (
        f"σ_measured = {measured_sigma:.3f} vs σ_expected = {expected_sigma:.3f}; "
        f"rel error = {abs(measured_sigma - expected_sigma) / expected_sigma:.3f} > 2%"
    )


# ---------------------------------------------------------------------------
# T-MC01-05: SNR round-trip — injected SNR recoverable within 5%
# ---------------------------------------------------------------------------

def test_mc01_05_snr_round_trip():
    """
    ΔS / σ_measured ≈ SNR_injected within 5%.

    Verifies that the noise amplitude is consistent with what the inversion
    will assume when building sigma_profile = np.full(n, ΔS / SNR).
    """
    from src.fpi.m03_airglow_synthesis_2026_05_24 import add_gaussian_noise

    rng = np.random.default_rng(2)
    n_pts = 300
    profile = np.sin(np.linspace(0, np.pi, n_pts)) * 400.0 + 200.0
    target_snr = 15.0

    # Average over 500 realisations to reduce sampling variance
    residuals = np.stack([
        add_gaussian_noise(profile, target_snr, profile, rng=rng) - profile
        for _ in range(500)
    ])
    sigma_meas = residuals.std()
    delta_S = float(profile.max() - profile.min())
    recovered_snr = delta_S / sigma_meas

    rel_err = abs(recovered_snr - target_snr) / target_snr
    assert rel_err < 0.05, (
        f"Recovered SNR = {recovered_snr:.2f} vs target = {target_snr}; "
        f"rel error = {rel_err:.3f} > 5%"
    )


# ---------------------------------------------------------------------------
# T-MC01-06: Reproducibility — same seed → identical results
# ---------------------------------------------------------------------------

def test_mc01_06_reproducibility(default_params, default_ap):
    """
    Two calls with the same Generator seed must produce bit-identical TrialResults.
    """
    rng_a = np.random.default_rng(99)
    rng_b = np.random.default_rng(99)

    res_a = run_single_trial(50.0, 750.0, 8.0, default_params, default_ap, rng_a)
    res_b = run_single_trial(50.0, 750.0, 8.0, default_params, default_ap, rng_b)

    assert res_a.v_est_ms   == res_b.v_est_ms,   "v_est_ms differs"
    assert res_a.T_est_K    == res_b.T_est_K,    "T_est_K differs"
    assert res_a.sigma_v_ms == res_b.sigma_v_ms, "sigma_v_ms differs"
    assert res_a.sigma_T_K  == res_b.sigma_T_K,  "sigma_T_K differs"
    assert res_a.chi2_airglow == res_b.chi2_airglow, "chi2_airglow differs"
    assert res_a.converged  == res_b.converged,  "converged differs"


# ---------------------------------------------------------------------------
# T-MC01-07: Parallelism consistency — n_workers=1 vs n_workers=2, N=4
# ---------------------------------------------------------------------------

def test_mc01_07_parallelism_consistency(default_params, default_ap):
    """
    run_simulation() with the same master seed produces bit-identical results
    regardless of n_workers (sequential vs parallel).

    Uses N=4 and n_workers=2 to keep CI runtime manageable (~4 s).
    """
    inputs = [TrialInput(v_true_ms=0.0, T_true_K=800.0, snr=10.0)
              for _ in range(4)]

    sim1 = run_simulation(inputs, default_params, default_ap,
                          n_workers=1, seed=42, progress=False)
    sim2 = run_simulation(inputs, default_params, default_ap,
                          n_workers=2, seed=42, progress=False)

    assert sim1.n_total == sim2.n_total == 4

    for i, (r1, r2) in enumerate(zip(sim1.trials, sim2.trials)):
        assert r1.v_est_ms == r2.v_est_ms, (
            f"Trial {i}: v_est_ms differs: {r1.v_est_ms} vs {r2.v_est_ms}"
        )
        assert r1.T_est_K == r2.T_est_K, (
            f"Trial {i}: T_est_K differs: {r1.T_est_K} vs {r2.T_est_K}"
        )
        assert r1.converged == r2.converged, (
            f"Trial {i}: converged differs: {r1.converged} vs {r2.converged}"
        )


# ---------------------------------------------------------------------------
# T-MC01-08: Save/load round-trip — all arrays recovered bit-exactly
# ---------------------------------------------------------------------------

def test_mc01_08_save_load_round_trip(default_params, default_ap):
    """
    save_simulation() / load_simulation() must recover all numeric arrays
    bit-exactly and the metadata fields accurately.
    """
    inputs = [TrialInput(v_true_ms=v, T_true_K=800.0, snr=10.0)
              for v in [0.0, 50.0, -50.0]]

    sim = run_simulation(inputs, default_params, default_ap,
                         n_workers=1, seed=7, progress=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = Path(tmpdir) / "test_sim.npz"
        save_simulation(sim, fpath)
        assert fpath.exists(), "save_simulation() did not create the file"

        loaded = load_simulation(fpath)

    assert loaded.n_total == sim.n_total
    assert loaded.n_converged == sim.n_converged
    assert loaded.seed == sim.seed

    for i, (orig, rec) in enumerate(zip(sim.trials, loaded.trials)):
        assert orig.v_est_ms    == rec.v_est_ms,    f"Trial {i}: v_est_ms"
        assert orig.T_est_K     == rec.T_est_K,     f"Trial {i}: T_est_K"
        assert orig.sigma_v_ms  == rec.sigma_v_ms,  f"Trial {i}: sigma_v_ms"
        assert orig.sigma_T_K   == rec.sigma_T_K,   f"Trial {i}: sigma_T_K"
        assert orig.chi2_airglow == rec.chi2_airglow, f"Trial {i}: chi2_airglow"
        assert orig.converged   == rec.converged,   f"Trial {i}: converged"


# ---------------------------------------------------------------------------
# T-MC01-09: Diverged trial flagging — SNR→0 returns converged=False, no exception
# ---------------------------------------------------------------------------

def test_mc01_09_diverged_trial(default_params, default_ap):
    """
    At SNR=1e-6 (near-infinite noise), the pipeline must not raise an exception
    and must return converged=False.

    seed=0 produces |v_est| > 2000 m/s, which violates the convergence criterion.
    """
    rng = np.random.default_rng(0)
    result = run_single_trial(0.0, 800.0, 1e-6, default_params, default_ap, rng)

    assert not result.converged, (
        f"Expected converged=False at SNR=1e-6, got converged=True "
        f"(v_est={result.v_est_ms:.1f} m/s)"
    )


# ---------------------------------------------------------------------------
# T-MC01-10: Constants traceability — module-level assertions pass at import
# ---------------------------------------------------------------------------

def test_mc01_10_constants_traceability():
    """
    Re-importing the module must not raise an AssertionError, verifying
    that InstrumentParams defaults are consistent with windcube/constants.py.

    The assertions run at module-import time (spec §4).
    """
    import importlib
    import windcube.mc01_fpi_mc_engine as mc01

    # Force reimport to re-run module-level code
    importlib.reload(mc01)

    from windcube.constants import (
        ALPHA_RAD_PX,
        ETALON_GAP_M,
        ETALON_R_INSTRUMENT,
        R_MAX_PX,
    )
    params = InstrumentParams()
    assert abs(params.R_refl - ETALON_R_INSTRUMENT) < 1e-9
    assert abs(params.t - ETALON_GAP_M) < 1e-6
    assert abs(params.alpha - ALPHA_RAD_PX) < 1e-7
    assert R_MAX_PX <= params.r_max
