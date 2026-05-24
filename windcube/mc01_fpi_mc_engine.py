"""
MC01 — FPI Monte Carlo Simulation Engine.

Spec: specs/MC01_fpi_mc_engine_2026-05-24.md

Wraps CAL01 calibration inversion + M03 thermal airglow synthesis + M06
temperature-enabled airglow inversion into a parallelisable per-trial harness.
"""

from __future__ import annotations

import json
import os
import types
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.fpi.fpi_cal_lib import (
    InstrumentParams,
    _neon_model,
    airy_modified,  # noqa: F401 — kept for callers that inspect the namespace
    phase_correct_gap,
    run_staged_inversion,
)
from src.fpi.m02_calibration_synthesis_2026_05_05 import add_poisson_noise
from src.fpi.m03_airglow_synthesis_2026_05_24 import (
    add_gaussian_noise,
    synthesise_airglow_image,
)
from src.fpi.m06_airglow_inversion_2026_05_24 import fit_airglow_fringe
from windcube.constants import (
    ALPHA_RAD_PX,
    ETALON_GAP_M,
    ETALON_R_INSTRUMENT,
    NE_INTENSITY_2,
    NE_WAVELENGTH_1_AIR_M,
    R_MAX_PX,
)

# ---------------------------------------------------------------------------
# Module-level assertions: verify InstrumentParams defaults against
# windcube/constants.py authoritative values (spec §4, T-MC01-10).
# ---------------------------------------------------------------------------
_DEFAULT_PARAMS = InstrumentParams()
assert abs(_DEFAULT_PARAMS.R_refl - ETALON_R_INSTRUMENT) < 1e-9, (
    f"InstrumentParams.R_refl={_DEFAULT_PARAMS.R_refl} "
    f"!= ETALON_R_INSTRUMENT={ETALON_R_INSTRUMENT}"
)
assert abs(_DEFAULT_PARAMS.t - ETALON_GAP_M) < 1e-6, (
    f"InstrumentParams.t={_DEFAULT_PARAMS.t} != ETALON_GAP_M={ETALON_GAP_M} "
    f"(diff={abs(_DEFAULT_PARAMS.t - ETALON_GAP_M):.2e} m)"
)
assert abs(_DEFAULT_PARAMS.alpha - ALPHA_RAD_PX) < 1e-7, (
    f"InstrumentParams.alpha={_DEFAULT_PARAMS.alpha} "
    f"!= ALPHA_RAD_PX={ALPHA_RAD_PX}"
)
assert R_MAX_PX <= _DEFAULT_PARAMS.r_max, (
    f"R_MAX_PX={R_MAX_PX} > InstrumentParams.r_max={_DEFAULT_PARAMS.r_max}"
)
del _DEFAULT_PARAMS

# ---------------------------------------------------------------------------
# Data classes (spec §3.2)
# ---------------------------------------------------------------------------

_N_CAL_BINS = 1500  # radial bins for calibration profile synthesis


@dataclass
class AirglowParams:
    """Airglow synthesis parameters — fixed across all trials in one simulation."""

    B_sci: float = 275.0       # CCD bias, counts (FM PTC value)
    Y_bg: float = 50.0         # background sky emission, counts
    Y_line: float = 1.0        # airglow line scale factor (dimensionless)
    L_synth: int = 300         # spectral bins for synthesis (anti-inverse-crime)
    L_invert: int = 101        # spectral bins for inversion (Harding §3)
    R_bins: int = 500          # radial bins (Harding §3: R=500)
    n_fsr: float = 5.0         # FSR span for spectral grid
    image_size: int = 256      # CCD active dimension, px (2×2 binned)


@dataclass
class TrialInput:
    """Specification of one Monte Carlo trial."""

    v_true_ms: float
    T_true_K: float
    snr: float


@dataclass
class TrialResult:
    """Results from one Monte Carlo trial."""

    v_true_ms: float
    T_true_K: float
    snr: float
    v_est_ms: float
    T_est_K: float
    sigma_v_ms: float
    sigma_T_K: float
    chi2_airglow: float
    converged: bool


@dataclass
class SimulationResult:
    """Aggregated results from an N-trial Monte Carlo simulation."""

    trials: list[TrialResult]
    instrument_params: InstrumentParams
    airglow_params: AirglowParams
    seed: int
    timestamp_utc: str
    n_converged: int
    n_total: int


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_cal_fp(
    noisy_profile: np.ndarray, r_grid: np.ndarray, r_max: float
) -> types.SimpleNamespace:
    """Build a duck-typed FringeProfile for calibration inversion."""
    sigma = np.maximum(np.sqrt(np.maximum(noisy_profile, 1.0)), 1.0)
    return types.SimpleNamespace(
        profile=noisy_profile,
        r_grid=r_grid,
        sigma_profile=sigma,
        masked=np.zeros(len(r_grid), dtype=bool),
        r_max_px=float(r_max),
    )


def _fit_to_cal(fit) -> types.SimpleNamespace:
    """Convert a CAL01 FitResult to the duck-typed cal object expected by M06."""
    return types.SimpleNamespace(
        t_m=fit.t_m,
        R_refl=fit.R1,      # R1 (640 nm line) is closest to OI 630 nm
        alpha=fit.alpha,
        I0=fit.I0,
        I1=fit.I1,
        I2=fit.I2,
        sigma0=fit.sigma0,
        sigma1=0.0,         # staged inversion fixes sigma1 = sigma2 = 0
        sigma2=0.0,
        B=fit.B,
        epsilon_cal=fit.epsilon_cal,
        quality_flags=0,    # GOOD — synthetic data
    )


def _build_science_fp(
    profile_1d: np.ndarray,
    r_grid: np.ndarray,
    r_max: float,
    sigma_N: float,
) -> object:
    """Build a duck-typed FringeProfile for the science airglow inversion."""
    from src.fpi.archive.m03_annular_reduction_2026_04_06 import (
        FringeProfile,
        QualityFlags,
    )

    n_bins = len(profile_1d)
    sigma_data = np.full(n_bins, max(float(sigma_N), 1.0))

    fp = FringeProfile.__new__(FringeProfile)
    fp.profile           = profile_1d.copy()
    fp.sigma_profile     = sigma_data
    fp.two_sigma_profile = 2.0 * sigma_data
    fp.r_grid            = r_grid.copy()
    fp.r2_grid           = r_grid ** 2
    fp.n_pixels          = np.ones(n_bins, dtype=int) * 100
    fp.masked            = np.zeros(n_bins, dtype=bool)
    fp.quality_flags     = QualityFlags.GOOD
    fp.r_max_px          = float(r_max)
    fp.cx                = float(r_max)
    fp.cy                = float(r_max)
    fp.sigma_cx          = 0.05
    fp.sigma_cy          = 0.05
    fp.two_sigma_cx      = 0.1
    fp.two_sigma_cy      = 0.1
    fp.seed_source       = "mc01_synthetic"
    fp.stage1_cx         = float(r_max)
    fp.stage1_cy         = float(r_max)
    fp.cost_at_min       = 0.0
    fp.sparse_bins       = False
    fp.r_min_px          = 0.0
    fp.n_bins            = n_bins
    fp.n_subpixels       = 1
    fp.sigma_clip        = 3.0
    fp.image_shape       = (256, 256)
    fp.peak_fits         = []
    fp.dark_subtracted   = False
    fp.dark_n_frames     = 0
    return fp


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_single_trial(
    v_true_ms: float,
    T_true_K: float,
    snr: float,
    instrument_params: InstrumentParams,
    airglow_params: AirglowParams,
    rng: np.random.Generator,
) -> TrialResult:
    """
    One complete Monte Carlo trial (spec §2).

    Parameters
    ----------
    v_true_ms         : True line-of-sight velocity, m/s.
    T_true_K          : True thermospheric temperature, K.
    snr               : Target signal-to-noise ratio for Gaussian noise (Eq. 17).
    instrument_params : True instrument parameters (InstrumentParams from fpi_cal_lib).
    airglow_params    : Airglow synthesis parameters (AirglowParams).
    rng               : Seeded numpy Generator for reproducibility.
    """
    params = instrument_params
    ap = airglow_params
    r_max = params.r_max

    # --- Step 1: Synthesise true neon calibration fringe ---
    r_cal = np.linspace(0.0, r_max, _N_CAL_BINS)
    true_cal = _neon_model(
        r_cal, r_max, params.t, params.alpha,
        params.R_refl, params.R_refl,
        params.I0, params.I1, params.I2,
        params.sigma0, params.sigma1, params.sigma2,
        params.B, NE_INTENSITY_2,
    )

    # --- Step 2: Add Poisson noise to calibration fringe ---
    noisy_cal = add_poisson_noise(true_cal, rng=rng)

    # --- Step 3: Invert noisy calibration fringe (CAL01 staged LM) ---
    eps_a = (2.0 * params.t / NE_WAVELENGTH_1_AIR_M) % 1.0
    t_eff = phase_correct_gap(params.t, eps_a, NE_WAVELENGTH_1_AIR_M)
    fp_cal = _build_cal_fp(noisy_cal, r_cal, r_max)
    cal_fit = run_staged_inversion(
        fp_cal, t_eff, params.alpha, eps_a,
        R1_init=params.R_refl, R2_init=params.R_refl,
    )
    cal = _fit_to_cal(cal_fit)

    # --- Step 4: Synthesise true airglow fringe (M03, use_temperature=True) ---
    # Uses TRUE instrument_params for synthesis — recovered 'cal' is for inversion only.
    ag = synthesise_airglow_image(
        params=params,
        v_rel_ms=v_true_ms,
        Y_line=ap.Y_line,
        Y_bg=ap.Y_bg,
        R_bins=ap.R_bins,
        L_synth=ap.L_synth,
        n_fsr=ap.n_fsr,
        image_size=ap.image_size,
        add_noise=False,
        use_temperature=True,
        T_true_K=T_true_K,
        cx=r_max,
        cy=r_max,
    )
    profile_1d = ag["profile_1d"]
    r_grid = ag["r_grid"]

    # --- Step 5: Add Gaussian white noise (Harding Eq. 17: σ_N = ΔS / SNR) ---
    delta_S = float(profile_1d.max() - profile_1d.min())
    sigma_N = delta_S / snr if snr > 0.0 else 1e10
    noisy_profile = add_gaussian_noise(profile_1d, snr, profile_1d, rng=rng)

    # --- Step 6: Invert noisy airglow fringe (M06, use_temperature=True) ---
    # Uses RECOVERED 'cal' (not true params) — anti-inverse-crime design.
    fp_sci = _build_science_fp(noisy_profile, r_grid, r_max, sigma_N)
    try:
        fit = fit_airglow_fringe(fp_sci, cal, n_fine=500, use_temperature=True)
    except Exception:
        return TrialResult(
            v_true_ms=float(v_true_ms),
            T_true_K=float(T_true_K),
            snr=float(snr),
            v_est_ms=float("nan"),
            T_est_K=float("nan"),
            sigma_v_ms=float("nan"),
            sigma_T_K=float("nan"),
            chi2_airglow=float("nan"),
            converged=False,
        )

    # --- Step 7: Package and return TrialResult ---
    converged = (
        fit.converged
        and fit.T_est_K is not None
        and fit.T_est_K > 0.0
        and abs(fit.v_rel_ms) < 2000.0
    )
    T_est = fit.T_est_K if fit.T_est_K is not None else float("nan")
    sigma_T = fit.sigma_T_K if fit.sigma_T_K is not None else float("nan")

    return TrialResult(
        v_true_ms=float(v_true_ms),
        T_true_K=float(T_true_K),
        snr=float(snr),
        v_est_ms=float(fit.v_rel_ms),
        T_est_K=T_est,
        sigma_v_ms=float(fit.sigma_v_rel_ms),
        sigma_T_K=sigma_T,
        chi2_airglow=float(fit.chi2_reduced),
        converged=bool(converged),
    )


def _trial_worker(args: tuple) -> TrialResult:
    """
    Module-level picklable worker for ProcessPoolExecutor.

    Receives (v_true, T_true, snr, instrument_params, airglow_params, child_seed)
    where child_seed is an np.random.SeedSequence object.
    """
    v_true, T_true, snr, instrument_params, airglow_params, child_seed = args
    rng = np.random.default_rng(child_seed)
    return run_single_trial(v_true, T_true, snr, instrument_params, airglow_params, rng)


def run_simulation(
    trial_inputs: list[TrialInput],
    instrument_params: InstrumentParams,
    airglow_params: AirglowParams,
    n_workers: int = -1,
    seed: int = 42,
    progress: bool = True,
) -> SimulationResult:
    """
    Run N trials in parallel via ProcessPoolExecutor (spec §7).

    Parameters
    ----------
    trial_inputs      : List of TrialInput specifications.
    instrument_params : True instrument parameters shared across all trials.
    airglow_params    : Airglow synthesis parameters shared across all trials.
    n_workers         : Number of parallel workers. -1 uses all CPU cores.
                        1 runs sequentially (no subprocess overhead on Windows).
    seed              : Master seed for np.random.SeedSequence.
    progress          : Print a progress counter to stdout.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    n_trials = len(trial_inputs)
    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(n_trials)

    if n_workers < 0:
        n_workers = os.cpu_count() or 1

    work_args = [
        (
            inp.v_true_ms, inp.T_true_K, inp.snr,
            instrument_params, airglow_params, child_seeds[i],
        )
        for i, inp in enumerate(trial_inputs)
    ]

    if n_workers == 1:
        # Sequential path — avoids subprocess spawn overhead on Windows CI.
        trials: list[TrialResult] = []
        for k, args in enumerate(work_args):
            trials.append(_trial_worker(args))
            if progress:
                print(f"\rMC01: {k + 1}/{n_trials} trials complete", end="", flush=True)
        if progress:
            print()
    else:
        trials = [None] * n_trials
        done = 0
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            future_to_idx = {
                pool.submit(_trial_worker, args): i
                for i, args in enumerate(work_args)
            }
            for future in as_completed(future_to_idx):
                i = future_to_idx[future]
                trials[i] = future.result()
                done += 1
                if progress:
                    print(f"\rMC01: {done}/{n_trials} trials complete", end="", flush=True)
        if progress:
            print()

    n_converged = sum(1 for t in trials if t.converged)

    return SimulationResult(
        trials=trials,
        instrument_params=instrument_params,
        airglow_params=airglow_params,
        seed=seed,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        n_converged=n_converged,
        n_total=n_trials,
    )


def save_simulation(result: SimulationResult, path: str | Path) -> None:
    """
    Serialise SimulationResult to a compressed .npz file (spec §8).

    Arrays stored: v_true, T_true, snr, v_est, T_est, sigma_v, sigma_T,
    chi2, converged (all shape (N,)), plus a JSON metadata scalar.
    """
    path = Path(path)
    trials = result.trials

    metadata = {
        "seed": result.seed,
        "timestamp_utc": result.timestamp_utc,
        "n_converged": result.n_converged,
        "n_total": result.n_total,
        "airglow_params": {
            "B_sci": result.airglow_params.B_sci,
            "Y_bg": result.airglow_params.Y_bg,
            "Y_line": result.airglow_params.Y_line,
            "L_synth": result.airglow_params.L_synth,
            "L_invert": result.airglow_params.L_invert,
            "R_bins": result.airglow_params.R_bins,
            "n_fsr": result.airglow_params.n_fsr,
            "image_size": result.airglow_params.image_size,
        },
    }

    np.savez_compressed(
        path,
        v_true=np.array([t.v_true_ms for t in trials], dtype=np.float64),
        T_true=np.array([t.T_true_K for t in trials], dtype=np.float64),
        snr=np.array([t.snr for t in trials], dtype=np.float64),
        v_est=np.array([t.v_est_ms for t in trials], dtype=np.float64),
        T_est=np.array([t.T_est_K for t in trials], dtype=np.float64),
        sigma_v=np.array([t.sigma_v_ms for t in trials], dtype=np.float64),
        sigma_T=np.array([t.sigma_T_K for t in trials], dtype=np.float64),
        chi2=np.array([t.chi2_airglow for t in trials], dtype=np.float64),
        converged=np.array([t.converged for t in trials], dtype=bool),
        metadata=np.array([json.dumps(metadata)]),
    )


def load_simulation(path: str | Path) -> SimulationResult:
    """
    Deserialise a SimulationResult from a .npz file produced by save_simulation().
    """
    path = Path(path)
    data = np.load(path, allow_pickle=False)
    metadata = json.loads(str(data["metadata"][0]))

    n = len(data["v_true"])
    trials = [
        TrialResult(
            v_true_ms=float(data["v_true"][i]),
            T_true_K=float(data["T_true"][i]),
            snr=float(data["snr"][i]),
            v_est_ms=float(data["v_est"][i]),
            T_est_K=float(data["T_est"][i]),
            sigma_v_ms=float(data["sigma_v"][i]),
            sigma_T_K=float(data["sigma_T"][i]),
            chi2_airglow=float(data["chi2"][i]),
            converged=bool(data["converged"][i]),
        )
        for i in range(n)
    ]

    ap_d = metadata["airglow_params"]
    airglow_params = AirglowParams(
        B_sci=ap_d["B_sci"],
        Y_bg=ap_d["Y_bg"],
        Y_line=ap_d["Y_line"],
        L_synth=ap_d["L_synth"],
        L_invert=ap_d["L_invert"],
        R_bins=ap_d["R_bins"],
        n_fsr=ap_d["n_fsr"],
        image_size=ap_d["image_size"],
    )

    return SimulationResult(
        trials=trials,
        instrument_params=InstrumentParams(),  # reconstruct from defaults
        airglow_params=airglow_params,
        seed=metadata["seed"],
        timestamp_utc=metadata["timestamp_utc"],
        n_converged=metadata["n_converged"],
        n_total=metadata["n_total"],
    )
