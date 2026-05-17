"""fpi_cal_lib — WindCube FPI calibration library (SPEC-CAL01 v1.1).

All calibration physics from dark stacking through the final H05
Levenberg-Marquardt fit.  Produces OrbitCalResult — the sole handoff
object to the science inversion layer (SPEC-CAL02).

No tkinter.  No plt.show().  Safe to import in headless environments.
"""
from __future__ import annotations

import datetime
import importlib.util
import pathlib
from dataclasses import dataclass

import numpy as np

# ── Re-exports — airy forward model ──────────────────────────────────────────

from src.fpi.airy_forward_model_2026_05_05 import (  # noqa: F401
    airy_modified,
    phase_correct_gap,
    InstrumentParams,
)

# ── Re-exports — centre finder ────────────────────────────────────────────────

from src.processing.center_finder import (  # noqa: F401
    azimuthal_variance_centre,
    estimate_centre_uncertainty,
    CentreResult,
    _variance_cost,
    find_centre as _cf_find_centre,
)

# ── Re-exports — annular reduction ────────────────────────────────────────────

from src.processing.annular_reduction import (  # noqa: F401
    annular_reduce,
    FringeProfile,
    PeakFitR2,
    PeakFit,
    _find_and_fit_peaks_r2,
)

# ── Re-exports — H05 calibration inversion ───────────────────────────────────

from src.processing.H05_calibration_inversion_2026_05_12 import (  # noqa: F401
    run_staged_inversion,
    save_cal_result,
    FitResult,
    _neon_model,
    _fd_jacobian,
    _run_stage,
    _model_components,
)

# ── Tolansky — importlib required because filename contains a hyphen ──────────

_TOLANSKY_PATH = pathlib.Path(__file__).parent / "tolansky_2026-05-13.py"
_spec_t = importlib.util.spec_from_file_location("tolansky_2line", _TOLANSKY_PATH)
_mod_t = importlib.util.module_from_spec(_spec_t)
# Must be registered before exec_module so @dataclass can resolve cls.__module__
import sys as _sys
_sys.modules.setdefault("tolansky_2line", _mod_t)
_spec_t.loader.exec_module(_mod_t)

run_tolansky_2line      = _mod_t.run_tolansky_2line       # noqa: F841
TolanskyResult          = _mod_t.TolanskyResult            # noqa: F841
plot_tolansky_result    = _mod_t.plot_tolansky_result      # noqa: F841
to_m05_priors           = _mod_t.to_m05_priors             # noqa: F841
load_and_split_families = _mod_t.load_and_split_families   # noqa: F841


# ── New dataclasses ───────────────────────────────────────────────────────────

@dataclass
class TolanskySeedMean:
    """Per-orbit mean of Tolansky seed quantities, averaged over N cal frames."""
    n_frames_total:          int
    n_frames_used:           int
    n_frames_rejected:       int

    d_m_mean:                float   # mean etalon gap (m)
    sigma_d_m_mean:          float   # std / sqrt(N_used), m
    two_sigma_d_m_mean:      float   # = 2 × sigma_d_m_mean

    alpha_mean:              float   # mean plate scale (rad/px)
    sigma_alpha_mean:        float
    two_sigma_alpha_mean:    float

    eps_a_mean:              float   # mean fractional order, 640.2 nm line
    sigma_eps_a_mean:        float
    two_sigma_eps_a_mean:    float

    Delta_a_mean:            float   # mean r²-spacing (px²/fringe)
    sigma_Delta_a_mean:      float
    two_sigma_Delta_a_mean:  float

    Y_B_obs_mean:            float   # mean amplitude ratio (638 / 640)


@dataclass
class OrbitCalResult:
    """
    Complete orbit instrument calibration — handoff object between layers.

    Produced by run_cal_pipeline.py (calibration layer).
    Consumed by run_airglow_inversion.py (science layer, SPEC-CAL02).
    Saved as orbit_cal_result.npy (numpy dict, allow_pickle=True).
    """
    seeds:             TolanskySeedMean
    fit:               FitResult          # H05 result — 10 free params
    source_cal_files:  list[str]
    source_dark_files: list[str]
    date_utc:          str                # ISO-8601
    pipeline_version:  str = "SPEC-CAL01 v1.1"


# ── Thin wrapper functions ────────────────────────────────────────────────────

def load_bin_frame(
    path: pathlib.Path,
    shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Load a WindCube FPI binary frame → float64 image array (H, W).

    Big-endian uint16 ('>u2'). Dimensions read from header words 0 and 1.
    Header row 0 is stripped — only image pixel rows are returned.
    Handles both 2×2 (259×276) and 1×1 (527×552) automatically.
    The `shape` argument is ignored; kept for backward compatibility only.
    """
    with open(path, "rb") as f:
        first_words = np.frombuffer(f.read(4), dtype=">u2")
    n_rows_frame = int(first_words[0])
    n_cols_frame = int(first_words[1])
    expected = n_rows_frame * n_cols_frame * 2
    actual   = pathlib.Path(path).stat().st_size
    if actual != expected:
        raise ValueError(
            f"{pathlib.Path(path).name}: file size {actual} B, "
            f"expected {expected} for {n_rows_frame}×{n_cols_frame} uint16."
        )
    raw = np.frombuffer(open(path, "rb").read(), dtype=">u2")
    return raw[n_cols_frame:].reshape(n_rows_frame - 1, n_cols_frame).astype(np.float64)


def make_master_dark(
    paths: list[pathlib.Path],
    shape: tuple[int, int] = (260, 276),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Median-stack N dark frames.

    Returns
    -------
    master_dark : float64 (H, W) — median per pixel
    dark_sigma  : float64 (H, W) — std per pixel
    """
    frames = np.stack([load_bin_frame(p, shape) for p in paths])
    return np.median(frames, axis=0), frames.std(axis=0)


def dark_subtract(
    frame: np.ndarray,
    master_dark: np.ndarray,
) -> np.ndarray:
    """Return frame.astype(float64) - master_dark (float64)."""
    return frame.astype(np.float64) - master_dark


def find_centre(
    image: np.ndarray,
    cx_seed: float | None = None,
    cy_seed: float | None = None,
    var_r_max_px: float | None = None,
) -> CentreResult:
    """
    Two-pass azimuthal variance centre finder.

    Seeds default to image centre if None.
    Calls azimuthal_variance_centre() then estimate_centre_uncertainty().
    Returns CentreResult with cx, cy, sigma_cx, sigma_cy,
    two_sigma_cx, two_sigma_cy, cost_at_min, grid_cx, grid_cy, grid_cost.
    """
    kwargs: dict = {}
    if var_r_max_px is not None:
        kwargs["var_r_max_px"] = var_r_max_px
    return _cf_find_centre(image, cx_seed=cx_seed, cy_seed=cy_seed, **kwargs)


def build_radial_profile(
    image: np.ndarray,
    cx: float,
    cy: float,
    n_bins: int = 150,
    r_max_px: float = 110.0,
) -> FringeProfile:
    """
    Mulligan r²-binned annular reduction → 1-D radial profile.

    Wraps annular_reduce(). Returns FringeProfile with fields:
    r_grid, r2_grid, profile, sigma_profile, masked, cx, cy, r_max_px.
    sigma_cx / sigma_cy default to 0.5 px (typical centre uncertainty).
    """
    return annular_reduce(
        image, cx, cy,
        sigma_cx=0.5,
        sigma_cy=0.5,
        n_bins=n_bins,
        r_max_px=r_max_px,
    )


def fit_peaks(
    fp: FringeProfile,
    distance: int = 5,
    prominence: float = 100.0,
    fit_half_window: int = 6,
) -> list[PeakFitR2]:
    """
    Detect peaks in radial profile and Gaussian-fit each in r² domain.

    Wraps _find_and_fit_peaks_r2().
    Expected: 20 peaks (10 strong 640.2 nm + 10 weak 638.3 nm pairs).
    """
    return _find_and_fit_peaks_r2(
        fp.r_grid,
        fp.r2_grid,
        fp.profile,
        fp.sigma_profile,
        fp.masked,
        distance=distance,
        prominence=prominence,
        fit_half_window=fit_half_window,
    )


def peaks_to_array(peaks: list[PeakFitR2]) -> np.ndarray:
    """
    Convert list[PeakFitR2] → float64 array (N, 10) for Tolansky input.

    Columns:
      0  peak_idx          (int, cast to float)
      1  r2_raw_px2
      2  r2_fit_px2        (NaN if fit failed)
      3  sigma_r2_fit_px2  (NaN if fit failed)
      4  r_raw_px
      5  0.0               (reserved)
      6  amplitude_adu
      7  width_r2_px2      (NaN if fit failed)
      8  reduced_chi2      (NaN if fit failed)
      9  line_id           (0.0 = 640.2 nm, 1.0 = 638.3 nm; median-amp split)
    """
    # Family assignment: strict interleaving by peak index.
    # Even-indexed peaks are always 640.2 nm (strong), odd-indexed are always
    # 638.3 nm (weak).  This is the physical reality — the two neon lines
    # produce strictly alternating rings in r².  A median-amplitude split
    # fails at large r² where the lines partially overlap.
    rows = []
    for p in peaks:
        line_id = 0.0 if (p.peak_idx % 2 == 0) else 1.0
        rows.append([
            float(p.peak_idx),
            p.r2_raw_px2,
            p.r2_fit_px2,
            p.sigma_r2_fit_px2,
            p.r_raw_px,
            0.0,
            p.amplitude_adu,
            p.width_r2_px2,
            p.reduced_chi2,
            line_id,
        ])
    return np.array(rows, dtype=float)


def run_tolansky(
    peak_array: np.ndarray,
    n_pairs: int | None = None,
) -> "TolanskyResult":
    """
    Two-line Tolansky WLS analysis.

    peak_array : (N, 10) float64 from peaks_to_array().
    Wraps run_tolansky_2line().
    Key outputs: Delta_a, eps_a, alpha_mean, d_m, Y_B_obs, chi2_dof_a, chi2_dof_b.
    """
    return run_tolansky_2line(peak_array, n_pairs=n_pairs)


def average_tolansky_seeds(
    results: list,
    chi2_threshold: float = 5.0,
) -> TolanskySeedMean:
    """
    Average Tolansky seeds over N cal frames.

    Frames where chi2_dof_a > threshold OR chi2_dof_b > threshold are
    excluded with a warning printed to stdout.
    Raises ValueError if no frames pass the filter.

    Computes mean and std/sqrt(N) for: d_m, alpha_mean, eps_a, Delta_a, Y_B_obs.
    All two_sigma_ fields = exactly 2 × sigma_.
    """
    kept = []
    for i, r in enumerate(results):
        if r.chi2_dof_a > chi2_threshold or r.chi2_dof_b > chi2_threshold:
            print(
                f"  WARNING: frame {i} rejected "
                f"(chi2_a={r.chi2_dof_a:.2f}, chi2_b={r.chi2_dof_b:.2f} "
                f"> {chi2_threshold})"
            )
        else:
            kept.append(r)

    if not kept:
        raise ValueError("No Tolansky frames passed chi2 filter.")

    N = len(kept)

    def _mean_sem(vals: list[float]) -> tuple[float, float]:
        a = np.array(vals, dtype=float)
        return float(a.mean()), float(a.std() / np.sqrt(N))

    d_m,  sig_d  = _mean_sem([r.d_m       for r in kept])
    al,   sig_al = _mean_sem([r.alpha_mean for r in kept])
    ep,   sig_ep = _mean_sem([r.eps_a      for r in kept])
    da,   sig_da = _mean_sem([r.Delta_a    for r in kept])
    yb = float(np.mean([r.Y_B_obs for r in kept]))

    return TolanskySeedMean(
        n_frames_total=len(results),
        n_frames_used=N,
        n_frames_rejected=len(results) - N,
        d_m_mean=d_m,
        sigma_d_m_mean=sig_d,
        two_sigma_d_m_mean=2.0 * sig_d,
        alpha_mean=al,
        sigma_alpha_mean=sig_al,
        two_sigma_alpha_mean=2.0 * sig_al,
        eps_a_mean=ep,
        sigma_eps_a_mean=sig_ep,
        two_sigma_eps_a_mean=2.0 * sig_ep,
        Delta_a_mean=da,
        sigma_Delta_a_mean=sig_da,
        two_sigma_Delta_a_mean=2.0 * sig_da,
        Y_B_obs_mean=yb,
    )


def run_h05(
    fp: FringeProfile,
    seeds: TolanskySeedMean,
    R1_init: float = 0.53,
    R2_init: float = 0.53,
    sigma0_init: float = 0.55,
) -> FitResult:
    """
    Full H05 staged Levenberg-Marquardt calibration fit.

    Builds the init vector from TolanskySeedMean (see §4 Stage 6 mapping),
    calls run_staged_inversion(), and returns FitResult.

    This is the culmination of the instrument characterisation pipeline.
    The returned FitResult is packed into OrbitCalResult by the wrapper
    and constitutes the sole input to the science inversion layer.
    """
    return run_staged_inversion(
        fp,
        t_eff=seeds.d_m_mean,
        alpha_init=seeds.alpha_mean,
        eps_a=seeds.eps_a_mean,
        ne_ratio_init=seeds.Y_B_obs_mean,
        R1_init=R1_init,
        R2_init=R2_init,
        sigma0_init=sigma0_init,
    )


# ── Public API surface ────────────────────────────────────────────────────────

__all__ = [
    # Re-exports — airy_forward_model
    "airy_modified",
    "phase_correct_gap",
    "InstrumentParams",
    # Re-exports — center_finder
    "azimuthal_variance_centre",
    "estimate_centre_uncertainty",
    "CentreResult",
    # Re-exports — annular_reduction
    "annular_reduce",
    "FringeProfile",
    "PeakFitR2",
    "PeakFit",
    "_find_and_fit_peaks_r2",
    # Re-exports — tolansky
    "run_tolansky_2line",
    "TolanskyResult",
    "plot_tolansky_result",
    "to_m05_priors",
    "load_and_split_families",
    # Re-exports — H05
    "run_staged_inversion",
    "save_cal_result",
    "FitResult",
    "_neon_model",
    "_fd_jacobian",
    "_run_stage",
    "_model_components",
    # New dataclasses
    "TolanskySeedMean",
    "OrbitCalResult",
    # New thin wrappers
    "load_bin_frame",
    "make_master_dark",
    "dark_subtract",
    "find_centre",
    "build_radial_profile",
    "fit_peaks",
    "peaks_to_array",
    "run_tolansky",
    "average_tolansky_seeds",
    "run_h05",
]
