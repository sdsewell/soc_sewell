"""
Tests for Z04 SNR Sensitivity Sweep Monte Carlo Validation.

Spec:  specs/Z04_snr_sensitivity_sweep_mc_2026_04_11.md
Script: validation/z04_snr_sensitivity_sweep.py

6 tests (T1–T6) per spec §6.
CI runtime target: < 120 seconds (N kept low).

Tests:
  T1  test_smoke             — N=10 trials, all sims run without exception
  T2  test_delta_s_positive  — noiseless fringe contrast ΔS > 0
  T3  test_velocity_recovery — |v_rec - v_true| < 50 m/s for 10 trials at SNR=5
  T4  test_sigma_ratio       — σ_ratio ∈ [0.5, 2.0] for N=100 trials at SNR=5
  T5  test_acceptance_json   — acceptance.json valid JSON with all 8 gate keys
  T6  test_output_files      — all 5 output files created (3 PNG, CSV, JSON)
"""

import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Load z04 module by file path (avoids __init__.py requirement for validation/)
# ---------------------------------------------------------------------------

_REPO_ROOT  = pathlib.Path(__file__).resolve().parent.parent
_Z04_PATH   = _REPO_ROOT / "validation" / "z04_snr_sensitivity_sweep.py"

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_spec = importlib.util.spec_from_file_location("z04_snr_sensitivity_sweep", _Z04_PATH)
_z04  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_z04)

# Convenience aliases
_build_calibration_result = _z04._build_calibration_result
_run_single_trial         = _z04._run_single_trial
_run_sim1                 = _z04._run_sim1
_run_sim2                 = _z04._run_sim2
_run_sim3                 = _z04._run_sim3
_SNR_GRID                 = _z04._SNR_GRID
_check_acceptance_gates   = _z04._check_acceptance_gates
_save_acceptance          = _z04._save_acceptance
_make_sim1_figure         = _z04._make_sim1_figure
_make_sim2_figure         = _z04._make_sim2_figure
_make_sim3_figure         = _z04._make_sim3_figure
_save_csv                 = _z04._save_csv


# ---------------------------------------------------------------------------
# Module-scoped fixtures (shared calibration to avoid repeated M05 fits)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def priors():
    """InstrumentParams + CalibrationResult, built once per test session."""
    from src.fpi.m01_airy_forward_model_2026_04_05 import InstrumentParams
    cal    = _build_calibration_result()
    params = InstrumentParams()
    return {"params": params, "cal": cal}


# ---------------------------------------------------------------------------
# T1 — Smoke test: N=10, all three sims complete without exception
# ---------------------------------------------------------------------------

def test_t1_smoke(priors):
    """N=10 trials at SNR=5, all three sims run without raising an exception."""
    rng = np.random.default_rng(100)

    # Sim-1
    trials1, stats1 = _run_sim1(10, 5.0, rng, priors)
    assert len(trials1) == 10

    # Sim-2
    trials2, stats2 = _run_sim2(10, 5.0, rng, priors)
    assert len(trials2) == 10

    # Sim-3: 1 trial per bin × 9 bins
    bin_results, stats3 = _run_sim3(9, _SNR_GRID, rng, priors)
    assert len(bin_results) == len(_SNR_GRID)


# ---------------------------------------------------------------------------
# T2 — ΔS > 0 for synthesised airglow fringe
# ---------------------------------------------------------------------------

def test_t2_delta_s_positive(priors):
    """Fringe contrast ΔS must be positive for a synthetic airglow frame."""
    rng    = np.random.default_rng(200)
    result = _run_single_trial(100.0, 5.0, rng, priors)
    assert result["delta_s"] > 0.0, (
        f"ΔS = {result['delta_s']:.4f} — expected positive fringe contrast"
    )


# ---------------------------------------------------------------------------
# T3 — Velocity recovery within ±50 m/s at SNR=5 (10 trials)
# ---------------------------------------------------------------------------

def test_t3_velocity_recovery(priors):
    """
    For 10 converged trials at SNR=5, |v_rec - v_true| < 50 m/s.

    The ±50 m/s bound is deliberately loose for CI (spec §6, T3).
    """
    rng = np.random.default_rng(300)
    n_checked = 0
    for _ in range(10):
        v_true = float(rng.uniform(-200.0, 200.0))
        r      = _run_single_trial(v_true, 5.0, rng, priors)
        if r["converged"]:
            n_checked += 1
            err = abs(r["v_rec"] - v_true)
            assert err < 50.0, (
                f"v_true={v_true:.1f} m/s, v_rec={r['v_rec']:.1f} m/s, "
                f"error={err:.1f} m/s > 50 m/s"
            )
    # At SNR=5 essentially all trials should converge
    assert n_checked >= 8, (
        f"Only {n_checked}/10 trials converged at SNR=5 — unexpectedly low"
    )


# ---------------------------------------------------------------------------
# T4 — σ_ratio ∈ [0.5, 2.0] for N=100 trials at SNR=5
# ---------------------------------------------------------------------------

def test_t4_sigma_ratio(priors):
    """
    σ_ratio = mean_σ_est / σ_v must lie in [0.5, 2.0] for N=100 at SNR=5.

    This is a loose CI-safe bound (full ORR gate G04 requires [0.80, 1.20]).
    """
    rng = np.random.default_rng(400)
    _, stats = _run_sim1(100, 5.0, rng, priors)

    sr = stats["sigma_ratio"]
    assert np.isfinite(sr), f"σ_ratio is not finite: {sr}"
    assert 0.5 <= sr <= 2.0, (
        f"σ_ratio = {sr:.4f} outside loose CI bound [0.5, 2.0]"
    )


# ---------------------------------------------------------------------------
# T5 — acceptance.json contains all 8 required gate keys
# ---------------------------------------------------------------------------

def test_t5_acceptance_json_keys(priors, tmp_path):
    """
    acceptance.json must be valid JSON containing all 8 gate keys:
    G01…G08 (spec §7).
    """
    rng = np.random.default_rng(500)
    N   = 50   # small N for speed

    _, sim1_stats = _run_sim1(N, 5.0, rng, priors)
    _, sim2_stats = _run_sim2(N, 5.0, rng, priors)
    _, sim3_stats = _run_sim3(N, _SNR_GRID, rng, priors)

    all_conv = (
        [sim1_stats["conv_rate"], sim2_stats["conv_rate"]]
        + [s["conv_rate"] for s in sim3_stats]
    )
    gates = _check_acceptance_gates(sim1_stats, sim2_stats, sim3_stats, all_conv)
    path  = _save_acceptance(gates, tmp_path)

    # Must be valid JSON
    with open(path) as f:
        loaded = json.load(f)

    required = {"G01", "G02", "G03", "G04", "G05", "G06", "G07", "G08"}
    missing  = required - set(loaded.keys())
    assert not missing, f"Missing gate keys in acceptance.json: {missing}"

    # Each entry must have 'pass' (bool), 'value', and 'threshold' fields
    for gid, entry in loaded.items():
        assert "pass"      in entry, f"{gid} missing 'pass' field"
        assert "value"     in entry, f"{gid} missing 'value' field"
        assert "threshold" in entry, f"{gid} missing 'threshold' field"
        assert isinstance(entry["pass"], bool), f"{gid}['pass'] is not bool"


# ---------------------------------------------------------------------------
# T6 — all 5 output files created (3 PNG + CSV + JSON)
# ---------------------------------------------------------------------------

def test_t6_output_files(priors, tmp_path):
    """
    All 5 output files must be created:
      z04_sim1_scatter.png
      z04_sim2_bias_vs_velocity.png
      z04_sim3_bias_vs_snr.png
      z04_mc_results.csv
      z04_acceptance.json
    """
    rng = np.random.default_rng(600)
    N   = 20

    sim1_trials, sim1_stats = _run_sim1(N, 5.0, rng, priors)
    sim2_trials, sim2_stats = _run_sim2(N, 5.0, rng, priors)
    sim3_bins,   sim3_stats = _run_sim3(N, _SNR_GRID, rng, priors)

    _make_sim1_figure(sim1_trials, sim1_stats, tmp_path)
    _make_sim2_figure(sim2_trials, sim2_stats, tmp_path)
    _make_sim3_figure(sim3_stats, tmp_path)

    all_conv = (
        [sim1_stats["conv_rate"], sim2_stats["conv_rate"]]
        + [s["conv_rate"] for s in sim3_stats]
    )
    gates = _check_acceptance_gates(sim1_stats, sim2_stats, sim3_stats, all_conv)
    _save_acceptance(gates, tmp_path)

    # CSV with sim1 rows
    csv_rows = [
        {
            "sim_id": 1, "trial": i,
            "v_true": t["v_true"], "snr": t["snr"],
            "v_rec": t["v_rec"], "sigma_est": t["sigma_est"],
            "converged": t["converged"],
        }
        for i, t in enumerate(sim1_trials)
    ]
    _save_csv(csv_rows, tmp_path)

    expected_files = [
        "z04_sim1_scatter.png",
        "z04_sim2_bias_vs_velocity.png",
        "z04_sim3_bias_vs_snr.png",
        "z04_acceptance.json",
        "z04_mc_results.csv",
    ]
    for fname in expected_files:
        fpath = tmp_path / fname
        assert fpath.exists(), f"Expected output file not found: {fname}"
        assert fpath.stat().st_size > 0, f"Output file is empty: {fname}"
