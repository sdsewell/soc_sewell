"""
Z04 — SNR Sensitivity Sweep Monte Carlo Validation

Spec:        specs/Z04_snr_sensitivity_sweep_mc_2026_04_11.md
Author:      Claude Code
Generated:   2026-04-11
Project:     WindCube FPI Pipeline — NCAR/HAO
Repo:        soc_sewell

Characterises pipeline velocity-retrieval bias and uncertainty vs SNR
(Harding et al. 2014 §4, Figs. 7–8 analogue for WindCube).

Three Monte Carlo simulations:
  Sim-1  Uncertainty Calibration   — σ_ratio and KS test at fixed SNR=5
  Sim-2  Bias vs LOS Velocity      — gain error check across ±300 m/s range
  Sim-3  Bias vs SNR (ORR gate)    — bias/σ_v profile over 9-point SNR grid

Usage:
  python validation/z04_snr_sensitivity_sweep.py [--full] [--seed N]
         [--outdir PATH] [--plot] [--sim {1,2,3}] [--verbose]
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import argparse
import csv
import json
import pathlib
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from joblib import Parallel, delayed

# ---------------------------------------------------------------------------
# Repo root on sys.path (needed when running directly or from subprocesses)
# ---------------------------------------------------------------------------

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Pipeline imports (after sys.path is set)
from src.fpi.m01_airy_forward_model_2026_04_05 import (  # noqa: E402
    InstrumentParams,
    NE_WAVELENGTH_1_M,
)
from src.fpi.m02_calibration_synthesis_2026_04_05 import synthesise_calibration_image
from src.fpi.m03_annular_reduction_2026_04_06 import annular_reduce, reduce_science_frame
from src.fpi.m04_airglow_synthesis_2026_04_05 import synthesise_airglow_image
from src.fpi.m05_calibration_inversion_2026_04_06 import fit_calibration_fringe, FitConfig
from src.fpi.m06_airglow_inversion_2026_04_06 import fit_airglow_fringe
from src.fpi.tolansky_2026_04_05 import TwoLineResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NCAR_NAVY = "#003479"
_NCAR_BLUE = "#0057C2"
_ORR_RED   = "#CC0000"

# SNR grid for Sim-3 (Harding Fig. 8 analogue)
_SNR_GRID = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0]

# ORR SNR threshold
_SNR_ORR_THRESHOLD = 2.0

# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------


def _build_tolansky_stub(params: InstrumentParams) -> TwoLineResult:
    """
    Build a TwoLineResult with correct values from known InstrumentParams.
    Bypasses TolanskyPipeline.run() amplitude-split reliability issues on
    synthetic data (same approach as tests/conftest.py).
    """
    tol = TwoLineResult.__new__(TwoLineResult)
    tol.d_m          = float(params.t)
    tol.alpha_rad_px = float(params.alpha)
    tol.eps1         = float((2.0 * params.t / NE_WAVELENGTH_1_M) % 1.0)
    return tol


def _build_calibration_result():
    """
    Build a CalibrationResult from a noiseless synthetic calibration image.

    Uses cx=params.r_max to match science frame synthesis centre — avoids
    the ~6 m/s systematic bias that arises from a centre offset between
    calibration and science reductions (see tests/conftest.py).

    Returns the same result each call (noiseless → deterministic).
    Per §10: instrument parameters are not perturbed between trials.
    """
    params = InstrumentParams()
    cx = params.r_max   # 128.0 px — matches science frame synthesis centre

    cal_m02 = synthesise_calibration_image(params, add_noise=False, cx=cx, cy=cx)
    fp = annular_reduce(
        cal_m02["image_2d"],
        cx=cx,
        cy=cx,
        sigma_cx=0.05,
        sigma_cy=0.05,
        r_max_px=params.r_max,
        n_bins=150,
    )
    tol_stub = _build_tolansky_stub(params)
    config   = FitConfig(tolansky=tol_stub)
    return fit_calibration_fringe(fp, config)


# ---------------------------------------------------------------------------
# Per-trial function (spec §4.3)
# ---------------------------------------------------------------------------


def _run_single_trial(
    v_true_ms: float,
    snr: float,
    rng: np.random.Generator,
    priors: dict,
) -> dict:
    """
    Synthesise one noisy airglow frame, reduce, invert, return results.

    Parameters
    ----------
    v_true_ms : float — true LOS velocity (m/s)
    snr       : float — target SNR = ΔS / σ_N (Harding Eq. 17)
    rng       : numpy Generator for reproducibility
    priors    : dict with keys 'params' (InstrumentParams) and 'cal' (CalibrationResult)

    Returns
    -------
    dict with keys:
        v_rec    : float  recovered LOS velocity (m/s); nan if fit failed
        sigma_est: float  pipeline-reported 1σ uncertainty (m/s); nan if fit failed
        converged: bool   True if LM fit converged
        eps      : float  recovered fractional order ε_sci; nan if fit failed
        lam_c    : float  recovered line centre (nm); nan if fit failed
        delta_s  : float  noiseless fringe contrast ΔS (ADU) — always finite
    """
    params = priors["params"]
    cal    = priors["cal"]
    cx     = params.r_max   # 128.0 px

    # Synthesise noisy airglow frame (Gaussian noise at specified SNR)
    m04 = synthesise_airglow_image(
        v_rel_ms=v_true_ms,
        params=params,
        snr=snr,
        add_noise=True,
        rng=rng,
        cx=cx,
        cy=cx,
    )

    # Fringe contrast from noiseless image (T2 check, §4.4)
    delta_s = float(
        np.max(m04["image_noiseless"]) - np.min(m04["image_noiseless"])
    )

    # Annular reduction: 2D → 1D fringe profile
    fp = reduce_science_frame(
        m04["image_2d"],
        cx=cx,
        cy=cx,
        sigma_cx=0.05,
        sigma_cy=0.05,
        r_max_px=params.r_max,
    )

    # M06 inversion
    try:
        fit       = fit_airglow_fringe(fp, cal)
        v_rec     = float(fit.v_rel_ms)
        sigma_est = float(fit.sigma_v_rel_ms)
        converged = bool(fit.converged)
        eps       = float(fit.epsilon_sci)
        lam_c     = float(fit.lambda_c_m)
    except Exception:
        v_rec     = np.nan
        sigma_est = np.nan
        converged = False
        eps       = np.nan
        lam_c     = np.nan

    return {
        "v_rec":     v_rec,
        "sigma_est": sigma_est,
        "converged": converged,
        "eps":       eps,
        "lam_c":     lam_c,
        "delta_s":   delta_s,
    }


# ---------------------------------------------------------------------------
# Simulation 1 — Uncertainty Calibration (spec §3, Sim-1)
# ---------------------------------------------------------------------------


def _run_sim1(
    n_trials: int,
    snr: float,
    rng: np.random.Generator,
    priors: dict,
    verbose: bool = False,
) -> tuple:
    """
    Sim-1: Fixed v_true=100 m/s, fixed SNR=5.

    Validates that σ_est accurately tracks σ_v (error bar calibration).

    Returns
    -------
    (trials, stats)
        trials : list of per-trial result dicts (with v_true, snr appended)
        stats  : dict of aggregate statistics
    """
    V_TRUE = 100.0
    trials = []
    log_interval = max(1, n_trials // 10)

    for i in range(n_trials):
        r = _run_single_trial(V_TRUE, snr, rng, priors)
        r["v_true"] = V_TRUE
        r["snr"]    = snr
        trials.append(r)
        if verbose and (i + 1) % log_interval == 0:
            print(f"  Sim-1: {i + 1}/{n_trials}")

    v_recs    = np.array([t["v_rec"]     for t in trials])
    sig_ests  = np.array([t["sigma_est"] for t in trials])
    converged = np.array([t["converged"] for t in trials], dtype=bool)

    finite_mask = np.isfinite(v_recs) & np.isfinite(sig_ests)
    v_err = v_recs[finite_mask] - V_TRUE
    s_est = sig_ests[finite_mask]

    bias       = float(np.mean(v_err))       if len(v_err) > 0 else np.nan
    sigma_v    = float(np.std(v_err, ddof=1)) if len(v_err) > 1 else np.nan
    mean_sigma = float(np.mean(s_est))        if len(s_est) > 0 else np.nan
    sigma_ratio = mean_sigma / sigma_v if (np.isfinite(sigma_v) and sigma_v > 0) else np.nan

    # 68th-percentile coverage: fraction of trials where |v_err| < σ_est
    coverage = float(np.mean(np.abs(v_err) < s_est)) if len(v_err) > 0 else np.nan

    # KS test: (v_err / σ_est) ~ N(0, 1)
    valid_z = s_est > 0
    if np.sum(valid_z) > 5:
        z = v_err[valid_z] / s_est[valid_z]
        ks_stat, ks_p = sp_stats.kstest(z, "norm")
        ks_stat, ks_p = float(ks_stat), float(ks_p)
    else:
        ks_stat, ks_p = np.nan, np.nan

    stats = {
        "bias":        bias,
        "sigma_v":     sigma_v,
        "mean_sigma":  mean_sigma,
        "sigma_ratio": sigma_ratio,
        "coverage":    coverage,
        "ks_stat":     ks_stat,
        "ks_p":        ks_p,
        "conv_rate":   float(np.mean(converged)),
        "n_finite":    int(np.sum(finite_mask)),
        "n_trials":    n_trials,
    }
    return trials, stats


# ---------------------------------------------------------------------------
# Simulation 2 — Bias vs LOS Velocity (spec §3, Sim-2)
# ---------------------------------------------------------------------------


def _run_sim2(
    n_trials: int,
    snr: float,
    rng: np.random.Generator,
    priors: dict,
    verbose: bool = False,
) -> tuple:
    """
    Sim-2: v_true ~ U(-300, +300) m/s, fixed SNR=5.

    Checks for velocity-dependent gain error (Harding Fig. 7 analogue).

    Returns
    -------
    (trials, stats)
    """
    trials = []
    log_interval = max(1, n_trials // 10)

    for i in range(n_trials):
        v_true = float(rng.uniform(-300.0, 300.0))
        r = _run_single_trial(v_true, snr, rng, priors)
        r["v_true"] = v_true
        r["snr"]    = snr
        trials.append(r)
        if verbose and (i + 1) % log_interval == 0:
            print(f"  Sim-2: {i + 1}/{n_trials}")

    v_trues   = np.array([t["v_true"] for t in trials])
    v_recs    = np.array([t["v_rec"]  for t in trials])
    converged = np.array([t["converged"] for t in trials], dtype=bool)

    finite_mask = np.isfinite(v_recs)
    v_err = v_recs[finite_mask] - v_trues[finite_mask]

    # Linear fit: error = slope × v_true + offset
    if np.sum(finite_mask) > 2:
        coeffs = np.polyfit(v_trues[finite_mask], v_err, 1)
        slope  = float(coeffs[0])
        offset = float(coeffs[1])
    else:
        slope  = np.nan
        offset = np.nan

    stats = {
        "slope":     slope,
        "offset":    offset,
        "conv_rate": float(np.mean(converged)),
        "n_finite":  int(np.sum(finite_mask)),
        "n_trials":  n_trials,
    }
    return trials, stats


# ---------------------------------------------------------------------------
# Sim-3 SNR-bin worker — module-level for joblib pickling safety
# ---------------------------------------------------------------------------


def _run_snr_bin(snr: float, n_trials: int, seed: int) -> list:
    """
    Worker for one SNR bin in Sim-3 (called by joblib).

    Builds its own CalibrationResult and RNG from the supplied seed,
    avoiding any pickling issues with the parent-process objects.
    """
    cal    = _build_calibration_result()
    params = InstrumentParams()
    priors = {"params": params, "cal": cal}
    rng    = np.random.default_rng(seed)

    results = []
    for _ in range(n_trials):
        v_true = float(rng.uniform(-200.0, 200.0))
        r = _run_single_trial(v_true, snr, rng, priors)
        r["v_true"] = v_true
        r["snr"]    = float(snr)
        results.append(r)
    return results


# ---------------------------------------------------------------------------
# Simulation 3 — Bias vs SNR (ORR gate, spec §3, Sim-3)
# ---------------------------------------------------------------------------


def _run_sim3(
    n_trials_total: int,
    snr_grid: list,
    rng: np.random.Generator,
    priors: dict,
    verbose: bool = False,
) -> tuple:
    """
    Sim-3: v_true ~ U(-200, +200) m/s, swept over snr_grid.

    Uses joblib.Parallel(n_jobs=-1) — each SNR bin runs in its own process.
    The priors argument is accepted for API consistency but workers build
    their own calibration (see _run_snr_bin).

    Returns
    -------
    (bin_results, stats_by_snr)
        bin_results   : list of lists — one list of trial dicts per SNR bin
        stats_by_snr  : list of per-bin aggregate stat dicts
    """
    n_bins  = len(snr_grid)
    n_base  = n_trials_total // n_bins
    n_extra = n_trials_total - n_base * n_bins
    n_per_bin = [n_base + (1 if i < n_extra else 0) for i in range(n_bins)]

    # Independent integer seeds for each bin (derived from parent RNG)
    seeds = rng.integers(0, 2**32, size=n_bins).tolist()

    if verbose:
        print(f"  Sim-3: {n_bins} SNR bins, ~{n_base} trials/bin, "
              f"seed={seeds[0]}…")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bin_results = Parallel(n_jobs=-1)(
            delayed(_run_snr_bin)(snr_grid[i], n_per_bin[i], seeds[i])
            for i in range(n_bins)
        )

    # Per-bin statistics
    stats_by_snr = []
    for snr, trials in zip(snr_grid, bin_results):
        v_trues   = np.array([t["v_true"]    for t in trials])
        v_recs    = np.array([t["v_rec"]     for t in trials])
        sig_ests  = np.array([t["sigma_est"] for t in trials])
        converged = np.array([t["converged"] for t in trials], dtype=bool)

        finite_mask = np.isfinite(v_recs) & np.isfinite(sig_ests)
        v_err = v_recs[finite_mask] - v_trues[finite_mask]
        n_fin = int(np.sum(finite_mask))

        bias_val  = float(np.mean(v_err))        if n_fin > 0 else np.nan
        sigma_v   = float(np.std(v_err, ddof=1)) if n_fin > 1 else np.nan
        mean_sig  = float(np.mean(sig_ests[finite_mask])) if n_fin > 0 else np.nan
        pct68     = float(np.percentile(np.abs(v_err), 68)) if n_fin > 0 else np.nan
        bias_err  = (sigma_v / np.sqrt(n_fin)
                     if (np.isfinite(sigma_v) and n_fin > 0) else np.nan)
        conv_rate = float(np.mean(converged))

        s = {
            "snr":             float(snr),
            "bias":            bias_val,
            "bias_err":        bias_err,
            "sigma_v":         sigma_v,
            "mean_sigma_est":  mean_sig,
            "pct68":           pct68,
            "conv_rate":       conv_rate,
            "n_finite":        n_fin,
            "n_trials":        len(trials),
        }
        stats_by_snr.append(s)

        if verbose:
            print(f"    SNR={snr:.2f}: bias={bias_val:+.1f} m/s, "
                  f"σ_v={sigma_v:.1f} m/s, conv={conv_rate:.1%}")

    return bin_results, stats_by_snr


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _make_sim1_figure(trials: list, stats: dict, outdir: pathlib.Path) -> pathlib.Path:
    """Scatter plot + residual histogram (Sim-1 output)."""
    v_recs   = np.array([t["v_rec"]     for t in trials])
    sig_ests = np.array([t["sigma_est"] for t in trials])
    mask     = np.isfinite(v_recs) & np.isfinite(sig_ests)
    v_err    = v_recs[mask] - 100.0   # v_true = 100 m/s

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Panel A — scatter: v_rec vs trial index
    ax = axes[0]
    ax.scatter(np.where(mask)[0], v_recs[mask],
               s=3, alpha=0.4, color=_NCAR_BLUE, rasterized=True)
    ax.axhline(100.0, color=_ORR_RED, lw=1.0, ls="--", label="v_true = 100 m/s")
    ax.set_xlabel("Trial index")
    ax.set_ylabel("Recovered $v_{LOS}$ (m/s)")
    ax.set_title("Sim-1: $v_{rec}$ scatter (SNR = 5)")
    ax.legend(fontsize=8)

    # Panel B — histogram of (v_rec - v_true) with Gaussian overlay
    ax = axes[1]
    ax.hist(v_err, bins=30, density=True,
            color=_NCAR_BLUE, alpha=0.65, label="Residuals")
    if np.isfinite(stats["sigma_v"]) and stats["sigma_v"] > 0:
        x = np.linspace(v_err.min() - 5, v_err.max() + 5, 300)
        gauss = (1.0 / (stats["sigma_v"] * np.sqrt(2 * np.pi))
                 * np.exp(-0.5 * (x / stats["sigma_v"]) ** 2))
        ax.plot(x, gauss, color=_NCAR_NAVY, lw=2,
                label=f"N(0, σ²)  σ={stats['sigma_v']:.1f} m/s")
    ax.set_xlabel("$v_{rec} - v_{true}$ (m/s)")
    ax.set_ylabel("Density")
    sr = stats["sigma_ratio"]
    ksp = stats["ks_p"]
    ax.set_title(f"σ_ratio = {sr:.3f}   KS p = {ksp:.3f}")
    ax.legend(fontsize=8)

    fig.tight_layout()
    path = outdir / "z04_sim1_scatter.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _make_sim2_figure(trials: list, stats: dict, outdir: pathlib.Path) -> pathlib.Path:
    """Velocity error vs true velocity (Harding Fig. 7 analogue)."""
    v_trues = np.array([t["v_true"] for t in trials])
    v_recs  = np.array([t["v_rec"]  for t in trials])
    mask    = np.isfinite(v_recs)
    v_err   = v_recs[mask] - v_trues[mask]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(v_trues[mask], v_err,
               s=4, alpha=0.4, color=_NCAR_BLUE, rasterized=True, label="Error")
    ax.axhline(0.0,   color="grey",  lw=0.8, ls="-",  alpha=0.5)
    ax.axhline(+5.0,  color=_ORR_RED, lw=1.2, ls=":",  label="±5 m/s band")
    ax.axhline(-5.0,  color=_ORR_RED, lw=1.2, ls=":")

    if np.isfinite(stats["slope"]):
        v_range = np.linspace(v_trues[mask].min(), v_trues[mask].max(), 200)
        ax.plot(v_range, stats["slope"] * v_range + stats["offset"],
                color=_NCAR_NAVY, lw=2,
                label=f"Fit slope = {stats['slope']:.4f} m/s per m/s")

    ax.set_xlabel("$v_{true}$ (m/s)")
    ax.set_ylabel("$v_{rec} - v_{true}$ (m/s)")
    ax.set_title("Sim-2: Velocity error vs true velocity (SNR = 5)")
    ax.legend(fontsize=9)

    fig.tight_layout()
    path = outdir / "z04_sim2_bias_vs_velocity.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _make_sim3_figure(stats_by_snr: list, outdir: pathlib.Path) -> pathlib.Path:
    """Two-panel bias and σ_v vs SNR (Harding Fig. 8 analogue — primary ORR figure)."""
    snrs      = np.array([s["snr"]           for s in stats_by_snr])
    biases    = np.array([s["bias"]          for s in stats_by_snr])
    bias_errs = np.array([s["bias_err"]      for s in stats_by_snr])
    sigma_vs  = np.array([s["sigma_v"]       for s in stats_by_snr])
    mean_sigs = np.array([s["mean_sigma_est"] for s in stats_by_snr])

    fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

    # Panel A — bias(SNR)
    ax = axes[0]
    ax.axvspan(0.0, _SNR_ORR_THRESHOLD,
               alpha=0.15, color="grey", label="SNR < 2 (below ORR threshold)")
    ax.errorbar(snrs, biases, yerr=bias_errs,
                color=_NCAR_NAVY, marker="o", ms=5, capsize=3, lw=1.5,
                label="Bias")
    ax.axhline(+5.0, color=_ORR_RED, ls="--", lw=1.5, label="±5 m/s ORR limit (G01)")
    ax.axhline(-5.0, color=_ORR_RED, ls="--", lw=1.5)
    ax.axhline(0.0,  color="grey", ls="-", lw=0.7, alpha=0.5)
    ax.set_ylabel("Bias (m/s)")
    ax.set_title("Sim-3: Bias and uncertainty vs SNR  (ORR deliverable)")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xscale("log")

    # Panel B — σ_v(SNR) and mean_σ_est(SNR)
    ax = axes[1]
    ax.axvspan(0.0, _SNR_ORR_THRESHOLD, alpha=0.15, color="grey")
    ax.plot(snrs, sigma_vs, color=_NCAR_BLUE, marker="o", ms=5, lw=1.5,
            label="$σ_v$ (actual scatter)")
    ax.plot(snrs, mean_sigs, color=_NCAR_NAVY, ls="--", marker="s", ms=4, lw=1.5,
            label="$\\overline{σ_{est}}$ (pipeline-reported)")
    ax.axhline(15.0, color=_ORR_RED, ls="--", lw=1.5,
               label="15 m/s ORR limit (G02)")
    ax.set_xlabel("SNR")
    ax.set_ylabel("Velocity uncertainty (m/s)")
    ax.set_xscale("log")
    ax.legend(fontsize=9, loc="upper right")

    fig.tight_layout()
    path = outdir / "z04_sim3_bias_vs_snr.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# ORR Acceptance Gates (spec §7)
# ---------------------------------------------------------------------------


def _check_acceptance_gates(
    sim1_stats: dict,
    sim2_stats: dict,
    sim3_stats_by_snr: list,
    all_conv_rates: list,
) -> dict:
    """
    Evaluate all 8 ORR acceptance gates.

    Returns
    -------
    dict mapping gate_id -> {"value": float, "threshold": str, "pass": bool}
    """
    gates = {}

    # G01: max |bias(SNR ≥ 2)| < 5 m/s
    biases_above_snr2 = [
        abs(s["bias"])
        for s in sim3_stats_by_snr
        if s["snr"] >= _SNR_ORR_THRESHOLD and np.isfinite(s["bias"])
    ]
    g01_val = float(max(biases_above_snr2)) if biases_above_snr2 else np.nan
    gates["G01"] = {
        "value":     g01_val,
        "threshold": "< 5.0 m/s",
        "pass":      bool(np.isfinite(g01_val) and g01_val < 5.0),
    }

    # G02: σ_v(SNR=2) ≤ 15 m/s
    snr2_bin = next(
        (s for s in sim3_stats_by_snr if abs(s["snr"] - 2.0) < 0.01), None
    )
    g02_val = float(snr2_bin["sigma_v"]) if snr2_bin else np.nan
    gates["G02"] = {
        "value":     g02_val,
        "threshold": "≤ 15.0 m/s",
        "pass":      bool(np.isfinite(g02_val) and g02_val <= 15.0),
    }

    # G03: σ_v(SNR=5) ≤ 10 m/s
    snr5_bin = next(
        (s for s in sim3_stats_by_snr if abs(s["snr"] - 5.0) < 0.01), None
    )
    g03_val = float(snr5_bin["sigma_v"]) if snr5_bin else np.nan
    gates["G03"] = {
        "value":     g03_val,
        "threshold": "≤ 10.0 m/s",
        "pass":      bool(np.isfinite(g03_val) and g03_val <= 10.0),
    }

    # G04: σ_ratio (Sim-1) ∈ [0.80, 1.20]
    g04_val = float(sim1_stats.get("sigma_ratio", np.nan))
    gates["G04"] = {
        "value":     g04_val,
        "threshold": "∈ [0.80, 1.20]",
        "pass":      bool(np.isfinite(g04_val) and 0.80 <= g04_val <= 1.20),
    }

    # G05: 68th-percentile coverage (Sim-1) ∈ [63%, 73%]
    g05_val = float(sim1_stats.get("coverage", np.nan))
    gates["G05"] = {
        "value":     g05_val,
        "threshold": "∈ [0.63, 0.73]",
        "pass":      bool(np.isfinite(g05_val) and 0.63 <= g05_val <= 0.73),
    }

    # G06: KS p-value (Sim-1) > 0.05
    g06_val = float(sim1_stats.get("ks_p", np.nan))
    gates["G06"] = {
        "value":     g06_val,
        "threshold": "> 0.05",
        "pass":      bool(np.isfinite(g06_val) and g06_val > 0.05),
    }

    # G07: |velocity-error slope| (Sim-2) < 0.01 m/s per m/s
    raw_slope = sim2_stats.get("slope", np.nan)
    g07_val   = float(abs(raw_slope)) if np.isfinite(raw_slope) else np.nan
    gates["G07"] = {
        "value":     g07_val,
        "threshold": "< 0.01 m/s per m/s",
        "pass":      bool(np.isfinite(g07_val) and g07_val < 0.01),
    }

    # G08: convergence rate (all sims) > 95%
    finite_rates = [r for r in all_conv_rates if np.isfinite(r)]
    g08_val = float(min(finite_rates)) if finite_rates else np.nan
    gates["G08"] = {
        "value":     g08_val,
        "threshold": "> 0.95",
        "pass":      bool(np.isfinite(g08_val) and g08_val > 0.95),
    }

    return gates


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _save_csv(rows: list, outdir: pathlib.Path) -> pathlib.Path:
    """Write trial-level CSV: [sim_id, trial, v_true, snr, v_rec, sigma_est, converged]."""
    path = outdir / "z04_mc_results.csv"
    fieldnames = ["sim_id", "trial", "v_true", "snr", "v_rec", "sigma_est", "converged"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


def _save_acceptance(gates: dict, outdir: pathlib.Path) -> pathlib.Path:
    """Write z04_acceptance.json — pass/fail dict keyed by gate name."""
    path = outdir / "z04_acceptance.json"
    out  = {}
    for gate_id, g in sorted(gates.items()):
        val = g["value"]
        out[gate_id] = {
            "value":     float(val) if np.isfinite(float(val)) else None,
            "threshold": str(g["threshold"]),
            "pass":      bool(g["pass"]),
        }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    return path


def _save_summary(
    sim1_stats: dict,
    sim2_stats: dict,
    sim3_stats_by_snr: list,
    gates: dict,
    outdir: pathlib.Path,
) -> pathlib.Path:
    """Print and save human-readable summary."""
    lines = [
        "=" * 62,
        "Z04 SNR Sensitivity Sweep — Monte Carlo Validation Summary",
        "=" * 62,
        "",
        "--- Sim-1: Uncertainty Calibration (SNR=5, v_true=100 m/s) ---",
        f"  bias           = {sim1_stats['bias']:+.2f} m/s",
        f"  σ_v            = {sim1_stats['sigma_v']:.2f} m/s",
        f"  mean σ_est     = {sim1_stats['mean_sigma']:.2f} m/s",
        f"  σ_ratio        = {sim1_stats['sigma_ratio']:.4f}",
        f"  coverage (68%) = {sim1_stats['coverage']:.1%}",
        f"  KS p-value     = {sim1_stats['ks_p']:.4f}",
        f"  convergence    = {sim1_stats['conv_rate']:.1%}",
        "",
        "--- Sim-2: Bias vs LOS Velocity (SNR=5) ---",
        f"  slope  = {sim2_stats['slope']:.6f} m/s per m/s",
        f"  offset = {sim2_stats['offset']:.2f} m/s",
        f"  convergence = {sim2_stats['conv_rate']:.1%}",
        "",
        "--- Sim-3: Bias vs SNR ---",
        f"  {'SNR':>6}  {'bias (m/s)':>10}  {'σ_v':>8}  {'σ̄_est':>8}  {'conv%':>6}",
        f"  {'-'*6}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*6}",
    ]
    for s in sim3_stats_by_snr:
        lines.append(
            f"  {s['snr']:>6.2f}  {s['bias']:>+10.2f}  {s['sigma_v']:>8.2f}"
            f"  {s['mean_sigma_est']:>8.2f}  {s['conv_rate']:>6.1%}"
        )
    lines += [
        "",
        "--- ORR Acceptance Gates ---",
        f"  {'Gate':>4}  {'Status':>6}  {'Value':>10}  Threshold",
        f"  {'----':>4}  {'------':>6}  {'----------':>10}  ---------",
    ]
    for gate_id, g in sorted(gates.items()):
        status = "PASS" if g["pass"] else "FAIL"
        val_s  = f"{g['value']:.4f}" if np.isfinite(g["value"]) else "nan"
        lines.append(f"  {gate_id:>4}  {status:>6}  {val_s:>10}  {g['threshold']}")
    all_pass = all(g["pass"] for g in gates.values())
    lines += [
        "",
        f"  Overall: {'ALL GATES PASS' if all_pass else 'ONE OR MORE GATES FAIL'}",
        "",
        "=" * 62,
    ]

    text = "\n".join(lines)
    print(text)
    path = outdir / "z04_summary.txt"
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# CLI main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Z04 SNR Sensitivity Sweep Monte Carlo Validation"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Run full N=10000 simulations (default: CI-scale N≈100)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed (default: 42)"
    )
    parser.add_argument(
        "--outdir", type=pathlib.Path,
        default=pathlib.Path("validation/outputs/z04"),
        help="Output directory (default: validation/outputs/z04/)"
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Show interactive matplotlib windows after saving"
    )
    parser.add_argument(
        "--sim", type=int, choices=[1, 2, 3],
        help="Run only the specified simulation number"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-trial / per-bin statistics to stdout"
    )
    args = parser.parse_args()

    # Trial counts
    if args.full:
        N_SIM1, N_SIM2, N_SIM3 = 10_000, 1_000, 10_000
    else:
        N_SIM1, N_SIM2, N_SIM3 = 100, 100, 1_000

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rng    = np.random.default_rng(seed=args.seed)
    params = InstrumentParams()

    print("Building calibration result…")
    cal    = _build_calibration_result()
    priors = {"params": params, "cal": cal}

    sim1_trials, sim2_trials = [], []
    sim1_stats,  sim2_stats  = {}, {}
    sim3_bins,   sim3_stats  = [], []

    # ---- Run simulations ---------------------------------------------------
    if args.sim is None or args.sim == 1:
        print(f"Sim-1: {N_SIM1} trials (v_true=100 m/s, SNR=5)…")
        sim1_trials, sim1_stats = _run_sim1(
            N_SIM1, 5.0, rng, priors, args.verbose
        )
        if args.verbose:
            print(f"  bias={sim1_stats['bias']:+.2f} m/s, "
                  f"σ_ratio={sim1_stats['sigma_ratio']:.3f}, "
                  f"KS p={sim1_stats['ks_p']:.3f}")

    if args.sim is None or args.sim == 2:
        print(f"Sim-2: {N_SIM2} trials (v_true~U(-300,+300) m/s, SNR=5)…")
        sim2_trials, sim2_stats = _run_sim2(
            N_SIM2, 5.0, rng, priors, args.verbose
        )
        if args.verbose:
            print(f"  slope={sim2_stats['slope']:.5f} m/s per m/s")

    if args.sim is None or args.sim == 3:
        print(f"Sim-3: {N_SIM3} trials over {len(_SNR_GRID)} SNR bins…")
        sim3_bins, sim3_stats = _run_sim3(
            N_SIM3, _SNR_GRID, rng, priors, args.verbose
        )

    # ---- Figures -----------------------------------------------------------
    if sim1_trials:
        p = _make_sim1_figure(sim1_trials, sim1_stats, outdir)
        print(f"  Saved {p}")

    if sim2_trials:
        p = _make_sim2_figure(sim2_trials, sim2_stats, outdir)
        print(f"  Saved {p}")

    if sim3_stats:
        p = _make_sim3_figure(sim3_stats, outdir)
        print(f"  Saved {p}")

    # ---- CSV ---------------------------------------------------------------
    csv_rows = []
    for i, t in enumerate(sim1_trials):
        csv_rows.append({
            "sim_id": 1, "trial": i,
            "v_true": t["v_true"], "snr": t["snr"],
            "v_rec": t["v_rec"], "sigma_est": t["sigma_est"],
            "converged": t["converged"],
        })
    for i, t in enumerate(sim2_trials):
        csv_rows.append({
            "sim_id": 2, "trial": i,
            "v_true": t["v_true"], "snr": t["snr"],
            "v_rec": t["v_rec"], "sigma_est": t["sigma_est"],
            "converged": t["converged"],
        })
    trial_idx = 0
    for bin_list in sim3_bins:
        for t in bin_list:
            csv_rows.append({
                "sim_id": 3, "trial": trial_idx,
                "v_true": t["v_true"], "snr": t["snr"],
                "v_rec": t["v_rec"], "sigma_est": t["sigma_est"],
                "converged": t["converged"],
            })
            trial_idx += 1
    if csv_rows:
        p = _save_csv(csv_rows, outdir)
        print(f"  Saved {p}")

    # ---- Acceptance gates --------------------------------------------------
    if sim1_stats and sim2_stats and sim3_stats:
        all_conv = (
            [sim1_stats["conv_rate"], sim2_stats["conv_rate"]]
            + [s["conv_rate"] for s in sim3_stats]
        )
        gates = _check_acceptance_gates(sim1_stats, sim2_stats, sim3_stats, all_conv)
        p = _save_acceptance(gates, outdir)
        print(f"  Saved {p}")
        _save_summary(sim1_stats, sim2_stats, sim3_stats, gates, outdir)
    else:
        print("Note: acceptance gates require all 3 simulations (run without --sim).")

    if args.plot:
        matplotlib.use("TkAgg")
        plt.show()

    print("Z04 complete.")


if __name__ == "__main__":
    main()
