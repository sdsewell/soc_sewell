"""
MC02 — FPI Monte Carlo Simulation Drivers (Harding Replication).

Spec: specs/MC02_fpi_mc_simulations_2026-05-24.md

Constructs TrialInput lists for three simulations (§2), drives MC01 execution,
generates publication-quality figures replicating Harding et al. (2014) Figs. 6–8,
and writes a summary report.  MC02 has no physics of its own.
"""

from __future__ import annotations

import subprocess
from datetime import date, datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from windcube.mc01_fpi_mc_engine import (
    SimulationResult,
    TrialInput,
)


# ---------------------------------------------------------------------------
# Input builders (spec §2)
# ---------------------------------------------------------------------------


def build_sim1_inputs(n: int = 10_000) -> list[TrialInput]:
    """
    Simulation 1 — Uncertainty Estimates (Harding §4.A / Fig. 6 analog).

    Returns N identical TrialInputs at v=100 m/s, T=800 K, SNR=5.
    """
    return [TrialInput(v_true_ms=100.0, T_true_K=800.0, snr=5.0)] * n


def build_sim2_inputs(n: int = 1_000, seed: int = 42) -> list[TrialInput]:
    """
    Simulation 2 — Biases Over Wind and Temperature (Harding §4.B / Fig. 7 analog).

    Samples v ∈ [−300, 300] m/s and T ∈ [300, 1500] K uniformly; SNR fixed at 5.
    """
    rng = np.random.default_rng(seed=seed)
    v_samples = rng.uniform(-300.0, 300.0, size=n)
    T_samples = rng.uniform(300.0, 1500.0, size=n)
    return [
        TrialInput(v_true_ms=float(v), T_true_K=float(T), snr=5.0)
        for v, T in zip(v_samples, T_samples)
    ]


def build_sim3_inputs(n: int = 10_000, seed: int = 42) -> list[TrialInput]:
    """
    Simulation 3 — Biases Over SNR (Harding §4.C / Fig. 8 analog).

    Samples v ∈ [−300, 300] m/s, T ∈ [300, 1500] K, SNR ∈ [0.5, 10] uniformly.
    WindCube extension: upper SNR bound is 10 (Harding used 5).
    """
    rng = np.random.default_rng(seed=seed)
    v_samples = rng.uniform(-300.0, 300.0, size=n)
    T_samples = rng.uniform(300.0, 1500.0, size=n)
    snr_samples = rng.uniform(0.5, 10.0, size=n)
    return [
        TrialInput(v_true_ms=float(v), T_true_K=float(T), snr=float(s))
        for v, T, s in zip(v_samples, T_samples, snr_samples)
    ]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _converged_arrays(result: SimulationResult) -> dict[str, np.ndarray]:
    """Extract arrays for converged trials only."""
    conv = np.array([t.converged for t in result.trials])
    return {
        "v_true":  np.array([t.v_true_ms  for t in result.trials])[conv],
        "T_true":  np.array([t.T_true_K   for t in result.trials])[conv],
        "snr":     np.array([t.snr        for t in result.trials])[conv],
        "v_est":   np.array([t.v_est_ms   for t in result.trials])[conv],
        "T_est":   np.array([t.T_est_K    for t in result.trials])[conv],
        "sigma_v": np.array([t.sigma_v_ms for t in result.trials])[conv],
        "sigma_T": np.array([t.sigma_T_K  for t in result.trials])[conv],
    }


def _make_covariance_ellipse(
    v: np.ndarray, T: np.ndarray
) -> tuple[float, float, float, float, float]:
    """
    Compute 1-σ sample covariance ellipse parameters from (v, T) samples.

    Follows spec §4.1 convention exactly:
      eigenvalues, eigenvectors = np.linalg.eigh(cov)
      angle  = np.degrees(np.arctan2(*eigenvectors[:, 1][::-1]))
      width, height = 2 * np.sqrt(eigenvalues)

    Returns (centre_v, centre_T, width, height, angle_deg).
    """
    cov = np.cov(v, T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(*eigenvectors[:, 1][::-1]))
    width, height = 2.0 * np.sqrt(eigenvalues)
    return float(np.mean(v)), float(np.mean(T)), float(width), float(height), float(angle)


def _bin_median_sigma(
    x: np.ndarray,
    y: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-bin median and ±1-σ (half of 16th–84th percentile range) of y over x bins.

    Returns (bin_centres, medians, half_sigmas).
    """
    centres, medians, half_sigmas = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (x >= lo) & (x < hi)
        if mask.sum() < 2:
            continue
        p16, p50, p84 = np.percentile(y[mask], [16, 50, 84])
        centres.append(0.5 * (lo + hi))
        medians.append(p50)
        half_sigmas.append(0.5 * (p84 - p16))
    return np.array(centres), np.array(medians), np.array(half_sigmas)


# ---------------------------------------------------------------------------
# Figure 1 — Uncertainty Estimates (spec §4.1)
# ---------------------------------------------------------------------------


def make_figure1(
    result: SimulationResult,
    outdir: str | Path,
    date_str: str | None = None,
) -> Path:
    """
    Single-panel scatter replicating Harding Fig. 6.

    Blue ellipse: 1-σ sample covariance of (v_est, T_est).
    Red dashed ellipse: 1-σ mean per-trial estimated uncertainties (zero cross-cov).
    """
    if date_str is None:
        date_str = date.today().isoformat()
    outdir = Path(outdir)
    fig_dir = outdir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    d = _converged_arrays(result)
    v_est, T_est = d["v_est"], d["T_est"]
    sigma_v, sigma_T = d["sigma_v"], d["sigma_T"]
    n_conv = len(v_est)

    mean_sv = float(np.nanmean(sigma_v))
    mean_sT = float(np.nanmean(sigma_T))

    # Fraction within 1-σ Mahalanobis distance
    cov_mat = np.cov(v_est, T_est)
    cov_inv = np.linalg.inv(cov_mat)
    dv = v_est - np.mean(v_est)
    dT = T_est - np.mean(T_est)
    delta = np.stack([dv, dT], axis=-1)
    mah2 = np.einsum("ni,ij,nj->n", delta, cov_inv, delta)
    frac_1sig = float(np.mean(mah2 < 1.0))

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(
        v_est, T_est,
        c="grey", alpha=0.15, s=2, linewidths=0, rasterized=True,
        zorder=1,
    )

    # Blue ellipse — sample covariance (spec §4.1)
    cov_data = np.cov(v_est, T_est)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_data)
    angle = np.degrees(np.arctan2(*eigenvectors[:, 1][::-1]))
    width, height = 2.0 * np.sqrt(eigenvalues)
    blue_ell = Ellipse(
        xy=(float(np.mean(v_est)), float(np.mean(T_est))),
        width=width, height=height, angle=angle,
        edgecolor="blue", facecolor="none", linewidth=2,
        label="Sample covariance (1σ)", zorder=3,
    )
    ax.add_patch(blue_ell)

    # Red dashed ellipse — mean estimated uncertainty, zero cross-covariance
    red_ell = Ellipse(
        xy=(100.0, 800.0),
        width=2.0 * mean_sv,
        height=2.0 * mean_sT,
        angle=0.0,
        edgecolor="red", facecolor="none", linewidth=2, linestyle="--",
        label="Mean estimated uncertainty (1σ)", zorder=4,
    )
    ax.add_patch(red_ell)

    # Crosshairs at true values
    ax.axvline(100.0, color="k", linewidth=0.5, alpha=0.5, zorder=2)
    ax.axhline(800.0, color="k", linewidth=0.5, alpha=0.5, zorder=2)

    info = (
        f"Mean σ_v = {mean_sv:.2f} m/s\n"
        f"Mean σ_T = {mean_sT:.1f} K\n"
        f"Fraction within 1σ = {frac_1sig:.3f}  [expect 0.683]\n"
        f"N converged = {n_conv:,}"
    )
    ax.text(
        0.03, 0.97, info,
        transform=ax.transAxes, va="top", ha="left", fontsize=8,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
        zorder=5,
    )

    ax.set_xlabel("v_est (m/s)")
    ax.set_ylabel("T_est (K)")
    ax.set_title(
        f"Simulation 1: Uncertainty Estimates\n"
        f"(v = 100 m/s, T = 800 K, SNR = 5, N = {result.n_total:,})"
    )
    ax.legend(loc="lower right", fontsize=8)

    png_path = fig_dir / f"MC02_fig1_sim1_uncertainty_{date_str}.png"
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"MC02_fig1_sim1_uncertainty_{date_str}.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    return png_path


# ---------------------------------------------------------------------------
# Figure 2 — Biases Over Wind and Temperature (spec §4.2)
# ---------------------------------------------------------------------------


def make_figure2(
    result: SimulationResult,
    outdir: str | Path,
    date_str: str | None = None,
) -> Path:
    """
    2×2 subplot grid replicating Harding Fig. 7.

    Panels: (a) v_error vs v_true, (b) v_error vs T_true,
            (c) T_error vs v_true, (d) T_error vs T_true.
    """
    if date_str is None:
        date_str = date.today().isoformat()
    outdir = Path(outdir)
    fig_dir = outdir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    d = _converged_arrays(result)
    v_true, T_true = d["v_true"], d["T_true"]
    v_err = d["v_est"] - v_true
    T_err = d["T_est"] - T_true

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        f"Simulation 2: Biases Over Wind Speed and Temperature\n"
        f"(SNR = 5, N = {result.n_total:,})"
    )

    v_edges = np.arange(-315.0, 315.1, 30.0)
    T_edges = np.arange(250.0, 1550.1, 100.0)

    panels = [
        (axes[0, 0], v_true, v_err, v_edges, "v_true (m/s)", "v_error (m/s)", "(a)"),
        (axes[0, 1], T_true, v_err, T_edges, "T_true (K)",   "v_error (m/s)", "(b)"),
        (axes[1, 0], v_true, T_err, v_edges, "v_true (m/s)", "T_error (K)",   "(c)"),
        (axes[1, 1], T_true, T_err, T_edges, "T_true (K)",   "T_error (K)",   "(d)"),
    ]

    for ax, x, y, edges, xlabel, ylabel, label in panels:
        ax.scatter(x, y, c="grey", alpha=0.4, s=3, linewidths=0, rasterized=True)
        ax.axhline(0.0, color="red", linewidth=1.0, linestyle="--")
        centres, medians, half_sigs = _bin_median_sigma(x, y, edges)
        if len(centres) > 0:
            ax.plot(centres, medians, color="orange", linewidth=1.5)
            ax.fill_between(
                centres,
                medians - half_sigs,
                medians + half_sigs,
                color="orange", alpha=0.35,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.text(0.03, 0.97, label, transform=ax.transAxes,
                va="top", ha="left", fontsize=10, fontweight="bold")

    fig.tight_layout()
    png_path = fig_dir / f"MC02_fig2_sim2_bias_vT_{date_str}.png"
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"MC02_fig2_sim2_bias_vT_{date_str}.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    return png_path


# ---------------------------------------------------------------------------
# Figure 3 — Biases Over SNR (spec §4.3)
# ---------------------------------------------------------------------------


def make_figure3(
    result: SimulationResult,
    outdir: str | Path,
    date_str: str | None = None,
) -> Path:
    """
    1×2 log-x subplot grid replicating Harding Fig. 8.

    Both panels have SNR on a log x-axis; orange median ± 1-σ band;
    dashed grey vertical at SNR = 5.
    """
    if date_str is None:
        date_str = date.today().isoformat()
    outdir = Path(outdir)
    fig_dir = outdir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    d = _converged_arrays(result)
    snr_conv = d["snr"]
    v_err = d["v_est"] - d["v_true"]
    T_err = d["T_est"] - d["T_true"]

    snr_min, snr_max = 0.5, 10.0
    log_edges = np.logspace(np.log10(snr_min), np.log10(snr_max), 11)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Simulation 3: Biases Over SNR\n"
        f"(v ∈ [−300, 300] m/s, T ∈ [300, 1500] K, "
        f"SNR ∈ [0.5, 10], N = {result.n_total:,})"
    )

    panels = [
        (axes[0], snr_conv, v_err, "v_error (m/s)", "(a)"),
        (axes[1], snr_conv, T_err, "T_error (K)",   "(b)"),
    ]

    for ax, x, y, ylabel, label in panels:
        ax.scatter(x, y, c="grey", alpha=0.15, s=1, linewidths=0, rasterized=True)
        ax.axhline(0.0, color="red", linewidth=1.0, linestyle="--")
        ax.axvline(5.0, color="grey", linewidth=1.0, linestyle="--")
        centres, medians, half_sigs = _bin_median_sigma(x, y, log_edges)
        if len(centres) > 0:
            ax.plot(centres, medians, color="orange", linewidth=1.5)
            ax.fill_between(
                centres,
                medians - half_sigs,
                medians + half_sigs,
                color="orange", alpha=0.35,
            )
        ax.set_xscale("log")
        ax.set_xlim(snr_min, snr_max)
        ax.set_xlabel("SNR")
        ax.set_ylabel(ylabel)
        ax.text(0.03, 0.97, label, transform=ax.transAxes,
                va="top", ha="left", fontsize=10, fontweight="bold")

    fig.tight_layout()
    png_path = fig_dir / f"MC02_fig3_sim3_bias_snr_{date_str}.png"
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"MC02_fig3_sim3_bias_snr_{date_str}.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    return png_path


# ---------------------------------------------------------------------------
# Summary report (spec §5)
# ---------------------------------------------------------------------------


def write_summary_report(
    sim1: SimulationResult | None,
    sim2: SimulationResult | None,
    sim3: SimulationResult | None,
    outdir: str | Path,
    seed: int = 42,
    date_str: str | None = None,
) -> Path:
    """
    Write the MC02 summary report to stdout and to a text file.

    Any of sim1/sim2/sim3 may be None if that simulation was not run.
    """
    if date_str is None:
        date_str = date.today().isoformat()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    def _git_hash(path_pattern: str) -> str:
        try:
            r = subprocess.run(
                ["git", "log", "-1", "--format=%H", "--", path_pattern],
                capture_output=True, text=True, timeout=5,
            )
            return r.stdout.strip()[:12] or "unknown"
        except Exception:
            return "unknown"

    cal01_hash = _git_hash("src/fpi/fpi_cal_lib.py")
    mc01_hash = _git_hash("windcube/mc01_fpi_mc_engine.py")

    lines: list[str] = [
        "WindCube MC02 Monte Carlo Simulation Summary",
        "============================================",
        f"Run date         : {run_ts}",
        f"CAL01 git hash   : {cal01_hash}",
        f"MC01 git hash    : {mc01_hash}",
        f"RNG seed         : {seed}",
        "",
    ]

    # --- Simulation 1 ---
    if sim1 is not None:
        d1 = _converged_arrays(sim1)
        pct1 = 100.0 * sim1.n_converged / sim1.n_total if sim1.n_total > 0 else 0.0
        mean_sv = float(np.nanmean(d1["sigma_v"]))
        mean_sT = float(np.nanmean(d1["sigma_T"]))
        frac_v = float(np.mean(np.abs(d1["v_est"] - 100.0) < mean_sv))
        frac_T = float(np.mean(np.abs(d1["T_est"] - 800.0) < mean_sT))
        _, _, bw, bh, _ = _make_covariance_ellipse(d1["v_est"], d1["T_est"])
        ratio_v = (bw / 2.0) / mean_sv
        ratio_T = (bh / 2.0) / mean_sT
        lines += [
            "Simulation 1 (Uncertainty Estimates)",
            f"  N trials             : {sim1.n_total:,}",
            f"  N converged          : {sim1.n_converged:,}  ({pct1:.1f}%)",
            f"  Mean σ_v             : {mean_sv:.2f} m/s   [Harding: ~1.8 m/s]",
            f"  Mean σ_T             : {mean_sT:.1f} K     [Harding: ~6.5 K]",
            f"  Fraction within 1σ v : {frac_v:.3f}       [expect 0.683]",
            f"  Fraction within 1σ T : {frac_T:.3f}       [expect 0.683]",
            f"  Blue/red axis ratio v: {ratio_v:.3f}       [expect ~1.00]",
            f"  Blue/red axis ratio T: {ratio_T:.3f}       [expect ~1.00]",
            "",
        ]
    else:
        lines += ["Simulation 1 (Uncertainty Estimates)", "  [not run]", ""]

    # --- Simulation 2 ---
    if sim2 is not None:
        d2 = _converged_arrays(sim2)
        pct2 = 100.0 * sim2.n_converged / sim2.n_total if sim2.n_total > 0 else 0.0
        v_err2 = d2["v_est"] - d2["v_true"]
        T_err2 = d2["T_est"] - d2["T_true"]
        med_v2 = float(np.nanmedian(v_err2))
        med_T2 = float(np.nanmedian(T_err2))
        v_edges = np.arange(-315.0, 315.1, 30.0)
        T_edges = np.arange(250.0, 1550.1, 100.0)
        _, mv_bins, _ = _bin_median_sigma(d2["v_true"], v_err2, v_edges)
        _, mT_bins, _ = _bin_median_sigma(d2["T_true"], T_err2, T_edges)
        max_v_bias = float(np.nanmax(np.abs(mv_bins))) if len(mv_bins) > 0 else float("nan")
        max_T_bias = float(np.nanmax(np.abs(mT_bins))) if len(mT_bins) > 0 else float("nan")
        lines += [
            "Simulation 2 (Biases Over Wind and Temperature)",
            f"  N trials             : {sim2.n_total:,}",
            f"  N converged          : {sim2.n_converged:,}  ({pct2:.1f}%)",
            f"  Median v bias        : {med_v2:.3f} m/s   [Harding: ~0.4 m/s]",
            f"  Median T bias        : {med_T2:.2f} K",
            f"  Max |v bias| binned  : {max_v_bias:.2f} m/s",
            f"  Max |T bias| binned  : {max_T_bias:.1f} K",
            "",
        ]
    else:
        lines += ["Simulation 2 (Biases Over Wind and Temperature)", "  [not run]", ""]

    # --- Simulation 3 ---
    if sim3 is not None:
        d3 = _converged_arrays(sim3)
        pct3 = 100.0 * sim3.n_converged / sim3.n_total if sim3.n_total > 0 else 0.0
        v_err3 = d3["v_est"] - d3["v_true"]
        T_err3 = d3["T_est"] - d3["T_true"]
        med_v3 = float(np.nanmedian(v_err3))
        med_T3 = float(np.nanmedian(T_err3))
        all_snr = np.array([t.snr for t in sim3.trials])
        all_conv = np.array([t.converged for t in sim3.trials])
        mask_lo = all_snr < 1.0
        mask_hi = all_snr > 3.0
        conv_lo = 100.0 * all_conv[mask_lo].mean() if mask_lo.sum() > 0 else float("nan")
        conv_hi = 100.0 * all_conv[mask_hi].mean() if mask_hi.sum() > 0 else float("nan")
        lines += [
            "Simulation 3 (Biases Over SNR)",
            f"  N trials             : {sim3.n_total:,}",
            f"  N converged          : {sim3.n_converged:,}  ({pct3:.1f}%)",
            f"  Median v bias (all)  : {med_v3:.3f} m/s",
            f"  Median T bias (all)  : {med_T3:.2f} K",
            f"  Convergence SNR < 1  : {conv_lo:.1f}%",
            f"  Convergence SNR > 3  : {conv_hi:.1f}%",
            "",
        ]
    else:
        lines += ["Simulation 3 (Biases Over SNR)", "  [not run]", ""]

    report = "\n".join(lines)
    print(report)

    out_path = outdir / f"MC02_summary_{date_str}.txt"
    out_path.write_text(report, encoding="utf-8")
    return out_path
