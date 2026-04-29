"""
Z02 — Synthetic Science Image Generator
Spec:    specs/z02_synthetic_science_image_generator_2026-04-29.md
Version: 1.0
Author:  Scott Sewell / HAO
Repo:    soc_sewell

Synthesises a matched science + dark image pair in authentic WindCube
.bin format (1-row header + pixel data), suitable for ingestion by Z01,
M06, or any downstream pipeline module.

Physics: ideal Airy transmission function with effective reflectivity R.
Single Doppler-shifted OI 630.2046 nm source.  I0 = I_peak (single line).
Dark current driven by focal-plane temperature T_fp_c.
Electronic offset fixed at OFFSET_ADU = 5 ADU (bias + read noise combined).
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import struct
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import NamedTuple, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import find_peaks

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.fpi.m02_calibration_synthesis_2026_04_05 import radial_profile_to_image

# ---------------------------------------------------------------------------
# Module-level fixed constants (§7)
# ---------------------------------------------------------------------------

OFFSET_ADU       = 5            # ADU — bias + read noise combined; fixed pedestal
DARK_REF_ADU_S   = 0.05        # ADU/px/s at T_REF_DARK_C
T_REF_DARK_C     = -20.0       # °C — dark reference temperature
T_DOUBLE_C       = 6.5         # °C — dark doubling interval
R_BINS           = 2000        # radial bins
N_REF            = 1.0         # refractive index, air gap
LAM_OI           = 630.2046e-9 # m — OI ¹S₀→¹D₂ vacuum wavelength (Edlén 1966)
C_LIGHT_MS       = 2.99792458e8 # m/s — speed of light (CODATA)
TIMER_PERIOD_S   = 0.001       # s — seconds per exposure-count unit


# ---------------------------------------------------------------------------
# BinningConfig NamedTuple (§6) — identical to Z03 v1.5
# ---------------------------------------------------------------------------

class BinningConfig(NamedTuple):
    nrows:         int
    ncols:         int
    active_rows:   int
    n_meta_rows:   int
    cx_default:    float
    cy_default:    float
    r_max_px:      float
    alpha_default: float
    pix_m:         float
    label:         str


_BINNING_CONFIGS: dict[int, BinningConfig] = {
    2: BinningConfig(
        nrows=260, ncols=276, active_rows=259, n_meta_rows=1,
        cx_default=137.5, cy_default=130.0, r_max_px=110.0,
        alpha_default=1.6000e-4, pix_m=32.0e-6, label="2x2_binned",
    ),
    1: BinningConfig(
        nrows=528, ncols=552, active_rows=527, n_meta_rows=1,
        cx_default=275.5, cy_default=264.0, r_max_px=220.0,
        alpha_default=0.8000e-4, pix_m=16.0e-6, label="1x1_unbinned",
    ),
}


def get_binning_config(binning: int) -> BinningConfig:
    """Return BinningConfig for binning factor 1 or 2."""
    if binning not in _BINNING_CONFIGS:
        raise ValueError(f"binning must be 1 or 2, got {binning}")
    return _BINNING_CONFIGS[binning]


# ---------------------------------------------------------------------------
# SynthParams dataclass — all user-prompted parameters (§8)
# ---------------------------------------------------------------------------

@dataclass
class SynthParams:
    # Group 0 — image geometry
    binning:   int     # 1 or 2
    cx:        float   # fringe centre column, pixels
    cy:        float   # fringe centre row, pixels (full-frame coords)
    # Group 1 — etalon geometry
    d_mm:      float   # etalon gap, mm  (default 20.0005)
    alpha:     float   # plate scale, rad/px  (mode-dependent default)
    # Group 2 — reflectivity
    R:         float   # effective reflectivity  (default 0.725)
    # Group 3 — intensity envelope and detector
    snr_peak:  float   # OI fringe peak SNR  (default 50.0)
    I1:        float   # linear vignetting  (default -0.1)
    I2:        float   # quadratic vignetting  (default 0.005)
    T_fp_c:    float   # focal plane temperature, °C  (default -20.0)
    # Group 4 — source velocity (replaces rel_638 from Z03)
    v_los_ms:  float   # line-of-sight velocity, m/s  (default -7500.0)


# ---------------------------------------------------------------------------
# DerivedParams dataclass — computed from SynthParams
# ---------------------------------------------------------------------------

@dataclass
class DerivedParams:
    alpha_rad_per_px: float
    I_peak:           float   # = I0 for single-line source
    I0:               float   # single-line amplitude = snr_to_ipeak(snr_peak, offset)
    lambda_obs:       float   # Doppler-shifted observed wavelength, m
    delta_lam:        float   # lambda_obs - LAM_OI, m
    FSR_m:            float
    finesse_F:        float
    finesse_N_R:      float
    dark_rate:        float   # ADU/px/s at T_fp_c


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def snr_to_ipeak(snr: float, offset: float) -> float:
    """Convert peak SNR to single-line peak intensity (positive root of quadratic).

    SNR = I_peak / sqrt(I_peak + offset)  →  I_peak² - snr²·I_peak - snr²·offset = 0
    """
    return (snr**2 + math.sqrt(snr**4 + 4.0 * snr**2 * offset)) / 2.0


def check_vignetting_positive(I0: float, I1: float, I2: float, r_max: float) -> bool:
    """Return True if I(r) = I0*(1+I1*(r/r_max)+I2*(r/r_max)²) > 0 for r in [0, r_max]."""
    r_test = np.array([0.0, r_max / 2.0, r_max])
    if abs(I2) > 1e-12:
        u_vertex = -I1 / (2.0 * I2)
        if 0.0 <= u_vertex <= 1.0:
            r_test = np.append(r_test, u_vertex * r_max)
    vals = I0 * (1.0 + I1 * (r_test / r_max) + I2 * (r_test / r_max) ** 2)
    return bool(np.all(vals > 0))


def derive_secondary(params: SynthParams) -> DerivedParams:
    """Compute derived optical parameters from user-prompted SynthParams."""
    I_peak      = snr_to_ipeak(params.snr_peak, OFFSET_ADU)
    I0          = I_peak          # single line: I_peak = I0 directly (§5.3)
    lambda_obs  = LAM_OI * (1.0 + params.v_los_ms / C_LIGHT_MS)
    delta_lam   = lambda_obs - LAM_OI
    d_m         = params.d_mm * 1e-3
    FSR         = LAM_OI**2 / (2.0 * N_REF * d_m)
    F           = 4.0 * params.R / (1.0 - params.R)**2
    N_R         = math.pi * math.sqrt(params.R) / (1.0 - params.R)
    dark_rate   = DARK_REF_ADU_S * 2.0**((params.T_fp_c - T_REF_DARK_C) / T_DOUBLE_C)
    return DerivedParams(
        alpha_rad_per_px = params.alpha,
        I_peak           = I_peak,
        I0               = I0,
        lambda_obs       = lambda_obs,
        delta_lam        = delta_lam,
        FSR_m            = FSR,
        finesse_F        = F,
        finesse_N_R      = N_R,
        dark_rate        = dark_rate,
    )


# ---------------------------------------------------------------------------
# Synthesis (§9 Stage C) — single OI line, inline Airy
# ---------------------------------------------------------------------------

def synthesise_profile(
    params: SynthParams,
    derived: DerivedParams,
) -> tuple:
    """Build the noise-free 1D radial fringe profile using inline airy_ideal.

    Returns (profile_1d, r_grid).
    profile_1d includes OFFSET_ADU background.  Single OI 630.2046 nm line.
    """
    cfg       = get_binning_config(params.binning)
    r_grid    = np.linspace(0.0, cfg.r_max_px, R_BINS)
    theta     = np.arctan(params.alpha * r_grid)
    cos_theta = np.cos(theta)
    F_coef    = 4.0 * params.R / (1.0 - params.R)**2
    u         = r_grid / cfg.r_max_px
    vignette  = derived.I0 * (1.0 + params.I1 * u + params.I2 * u**2)

    phase      = 4.0 * np.pi * N_REF * (params.d_mm * 1e-3) * cos_theta / derived.lambda_obs
    A_oi       = vignette / (1.0 + F_coef * np.sin(phase / 2.0)**2)
    profile_1d = A_oi + OFFSET_ADU
    return profile_1d, r_grid


def synthesise_image(params: SynthParams, derived: DerivedParams) -> np.ndarray:
    """Return noise-free pixel data (active_rows × ncols), float64.

    Excludes the 1-row header — caller adds header via build_full_frame().
    cy is adjusted for the 1-row header offset before passing to M02.
    """
    cfg        = get_binning_config(params.binning)
    profile_1d, r_grid = synthesise_profile(params, derived)

    cy_pix = params.cy - cfg.n_meta_rows

    image_sq = radial_profile_to_image(
        profile_1d, r_grid,
        image_size=cfg.ncols,
        cx=params.cx,
        cy=cy_pix,
        bias=float(OFFSET_ADU),
    )
    return image_sq[:cfg.active_rows, :]


# ---------------------------------------------------------------------------
# Noise model (§9 Stage D)
# ---------------------------------------------------------------------------

def _apply_sci_noise(
    image_float: np.ndarray,
    params: SynthParams,
    derived: DerivedParams,
    exp_time_s: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply photon + dark noise to science pixel array; return uint16."""
    mean_dark   = derived.dark_rate * exp_time_s
    sci_signal  = image_float + mean_dark
    sci_noisy   = rng.poisson(np.maximum(sci_signal, 0)).astype(float) + OFFSET_ADU
    return np.clip(np.round(sci_noisy), 0, 16383).astype(np.uint16)


def _apply_dark_noise(
    shape: tuple,
    derived: DerivedParams,
    exp_time_s: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Synthesise dark pixel array (no fringe signal); return uint16."""
    mean_dark  = derived.dark_rate * exp_time_s
    dark_noisy = rng.poisson(np.full(shape, max(mean_dark, 0.0))).astype(float) + OFFSET_ADU
    return np.clip(np.round(dark_noisy), 0, 16383).astype(np.uint16)


# ---------------------------------------------------------------------------
# Header writer (§9 Stage E) — 1-row header, P01-compatible layout
# ---------------------------------------------------------------------------

def _encode_u64(h: np.ndarray, w: int, value: int) -> None:
    """Store uint64 into h[w:w+4] in P01 LE-word-order encoding."""
    for i in range(4):
        h[w + i] = (value >> (16 * i)) & 0xFFFF


def _encode_f64(h: np.ndarray, w: int, value: float) -> None:
    """Store float64 into h[w:w+4] in P01 LE-word-order encoding."""
    b     = struct.pack(">d", value)
    words = struct.unpack(">4H", b)
    h[w + 0] = words[3]
    h[w + 1] = words[2]
    h[w + 2] = words[1]
    h[w + 3] = words[0]


def _write_header_row(full_array: np.ndarray, image_type: str) -> None:
    """Write P01-compatible 1-row header into row 0 of full_array (in-place).

    image_type: 'science' — shutter open, no lamp → parse_header returns 'science'
                'cal'     — lamp on → parse_header returns 'cal'
                'dark'    — shutter closed → parse_header returns 'dark'
    """
    nrows, ncols = full_array.shape
    h = np.zeros(ncols, dtype=np.uint16)

    h[0] = nrows
    h[1] = ncols
    h[2] = 12000   # 120 s exposure in centiseconds
    h[3] = 0       # exp_unit

    _encode_f64(h, 4, 24.0)

    ts_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    _encode_u64(h, 8, ts_ms)

    itype = image_type.lower()
    if itype.startswith("dark"):
        # shutter closed: gpio[0]=1 (shutter_closed), gpio[3]=1 (pwr_on)
        h[100] = 1
        h[103] = 1
    elif itype.startswith("cal"):
        # lamp on; shutter implicitly open (no closed bit set)
        h[104] = 1
    # science: no lamp, no shutter_closed bits → parse_header returns 'science'

    full_array[0, :] = h


def build_full_frame(
    pixel_uint16: np.ndarray,
    params: SynthParams,
    image_type: str,
) -> np.ndarray:
    """Combine pixel data and 1-row header into a full-frame uint16 array."""
    cfg  = get_binning_config(params.binning)
    full = np.zeros((cfg.nrows, cfg.ncols), dtype=np.uint16)
    full[cfg.n_meta_rows:, :] = pixel_uint16
    _write_header_row(full, image_type)
    return full


# ---------------------------------------------------------------------------
# High-level synthesis driver (non-interactive, used by tests and main)
# ---------------------------------------------------------------------------

def run_synthesis(
    params: SynthParams,
    out_dir: pathlib.Path,
    seed: int = 42,
    exp_time_cts: int = 12000,
) -> tuple:
    """Synthesise science+dark pair, write .bin files and truth JSON.

    Returns (sci_path, dark_path, truth_path, stem).
    """
    cfg         = get_binning_config(params.binning)
    derived     = derive_secondary(params)

    if not check_vignetting_positive(derived.I0, params.I1, params.I2, cfg.r_max_px):
        raise ValueError("Vignetting envelope I(r) goes non-positive. Adjust I1/I2.")

    image_float = synthesise_image(params, derived)
    rng         = np.random.default_rng(seed)
    exp_time_s  = exp_time_cts * TIMER_PERIOD_S

    sci_px  = _apply_sci_noise(image_float, params, derived, exp_time_s, rng)
    dark_px = _apply_dark_noise(image_float.shape, derived, exp_time_s, rng)

    sci_frame  = build_full_frame(sci_px,  params, "science")
    dark_frame = build_full_frame(dark_px, params, "dark")

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts        = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem      = f"{ts}_synth_z02_{cfg.label}"
    sci_name  = f"{ts}_sci_synth_z02_{cfg.label}.bin"
    dark_name = f"{ts}_dark_synth_z02_{cfg.label}.bin"

    sci_path  = out_dir / sci_name
    dark_path = out_dir / dark_name

    sci_frame.astype(">u2").tofile(str(sci_path))
    dark_frame.astype(">u2").tofile(str(dark_path))

    truth_path = out_dir / f"{stem}_truth.json"
    write_truth_json(params, derived, seed, sci_path, dark_path, truth_path)

    return sci_path, dark_path, truth_path, stem


# ---------------------------------------------------------------------------
# Truth JSON (§10)
# ---------------------------------------------------------------------------

def write_truth_json(
    params: SynthParams,
    derived: DerivedParams,
    seed: int,
    path_sci: pathlib.Path,
    path_dark: pathlib.Path,
    truth_path: pathlib.Path,
) -> None:
    """Write the _truth.json sidecar with v1.0 schema."""
    cfg = get_binning_config(params.binning)
    ts  = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    truth = {
        "z02_version":   "1.0",
        "timestamp_utc": ts,
        "random_seed":   seed,
        "user_params": {
            "binning":   params.binning,
            "cx":        params.cx,
            "cy":        params.cy,
            "d_mm":      params.d_mm,
            "alpha":     params.alpha,
            "R":         params.R,
            "snr_peak":  params.snr_peak,
            "I1":        params.I1,
            "I2":        params.I2,
            "T_fp_c":    params.T_fp_c,
            "v_los_ms":  params.v_los_ms,
        },
        "derived_params": {
            "alpha_rad_per_px":      derived.alpha_rad_per_px,
            "lambda_obs_m":          derived.lambda_obs,
            "delta_lambda_m":        derived.delta_lam,
            "I_peak_adu":            derived.I_peak,
            "I0_adu":                derived.I0,
            "finesse_N_R":           derived.finesse_N_R,
            "finesse_coefficient_F": derived.finesse_F,
            "FSR_m":                 derived.FSR_m,
            "dark_rate_adu_px_s":    derived.dark_rate,
        },
        "fixed_constants": {
            "lam_oi_m":       LAM_OI,
            "c_light_ms":     C_LIGHT_MS,
            "offset_adu":     OFFSET_ADU,
            "dark_ref_adu_s": DARK_REF_ADU_S,
            "T_ref_dark_c":   T_REF_DARK_C,
            "T_double_c":     T_DOUBLE_C,
            "R_bins":         R_BINS,
            "n_ref":          N_REF,
            "n_meta_rows":    cfg.n_meta_rows,
            "nrows":          cfg.nrows,
            "ncols":          cfg.ncols,
            "active_rows":    cfg.active_rows,
            "r_max_px":       cfg.r_max_px,
            "pix_m":          cfg.pix_m,
            "label":          cfg.label,
        },
        "output_sci_file":  str(path_sci.name),
        "output_dark_file": str(path_dark.name),
    }
    with open(truth_path, "w", encoding="utf-8") as fh:
        json.dump(truth, fh, indent=2)


# ---------------------------------------------------------------------------
# Diagnostic figure (§9 Stage G)
# ---------------------------------------------------------------------------

def make_diagnostic_figure(
    sci_img: np.ndarray,
    dark_img: np.ndarray,
    params: SynthParams,
    derived: DerivedParams,
    out_dir: pathlib.Path,
    stem: str,
) -> pathlib.Path:
    """Save a 2×2 diagnostic figure: images (top row), histograms (bottom row)."""
    cfg = get_binning_config(params.binning)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_sci, ax_dark, ax_hsci, ax_hdark = (
        axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    )

    sci_pixels = sci_img[cfg.n_meta_rows:, :]
    sci_max    = max(int(sci_pixels.max()), 1)

    im0 = ax_sci.imshow(sci_img, cmap="gray", vmin=0, vmax=sci_max)
    title_l1 = (
        f"d={params.d_mm} mm   α={params.alpha:.4e} rad/px   "
        f"R={params.R:.3f} (N_R={derived.finesse_N_R:.1f})   "
        f"SNR={params.snr_peak}   [{cfg.label}]"
    )
    title_l2 = (
        f"cx={params.cx:.1f}   cy={params.cy:.1f}   "
        f"I₁={params.I1}   I₂={params.I2}   T_fp={params.T_fp_c}°C"
    )
    title_l3 = (
        f"OI 630.2046 nm   v_los={params.v_los_ms:+.0f} m/s   "
        f"λ_obs={derived.lambda_obs * 1e9:.4f} nm"
    )
    title_l4 = (
        f"I₀ = {derived.I0:.1f} ADU   (single line; I_peak = I₀)"
    )
    ax_sci.set_title(f"{title_l1}\n{title_l2}\n{title_l3}\n{title_l4}", fontsize=7)
    fig.colorbar(im0, ax=ax_sci, fraction=0.046, pad=0.04)

    dark_pixels = dark_img[cfg.n_meta_rows:, :]
    dark_max    = max(int(dark_pixels.max()), 1)

    im1 = ax_dark.imshow(dark_img, cmap="gray", vmin=0, vmax=dark_max)
    ax_dark.set_title(
        f"Dark image — T_fp={params.T_fp_c}°C   "
        f"dark_rate={derived.dark_rate:.4f} ADU/px/s",
        fontsize=9,
    )
    fig.colorbar(im1, ax=ax_dark, fraction=0.046, pad=0.04)

    ax_hsci.hist(sci_pixels.ravel(), bins=256, range=(0, sci_max), color="C0", linewidth=0)
    ax_hsci.set_xlim(0, sci_max)
    ax_hsci.set_xlabel("ADU")
    ax_hsci.set_ylabel("Pixel count")
    ax_hsci.set_title("Science histogram")

    ax_hdark.hist(dark_pixels.ravel(), bins=128, range=(0, dark_max), color="C1", linewidth=0)
    ax_hdark.set_xlim(0, dark_max)
    ax_hdark.set_xlabel("ADU")
    ax_hdark.set_ylabel("Pixel count")
    ax_hdark.set_title("Dark histogram")

    fig.suptitle(f"Z02 diagnostic — {stem}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = out_dir / f"{stem}_diagnostic.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Radial profile from 2-D image
# ---------------------------------------------------------------------------

def compute_radial_profile_from_image(
    image: np.ndarray,
    cx: float,
    cy: float,
    r_max: float,
    n_bins: int = 350,
    exclude_rows: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute a radially averaged profile from a 2-D image.

    Returns (r_centers, means, sems, counts).
    """
    nrows, ncols = image.shape
    col_idx = np.arange(ncols)
    row_idx = np.arange(exclude_rows, nrows)
    col_grid, row_grid = np.meshgrid(col_idx, row_idx)

    r_flat = np.sqrt((col_grid - cx)**2 + (row_grid - cy)**2).ravel()
    v_flat = image[exclude_rows:, :].astype(np.float64).ravel()

    bin_edges = np.linspace(0.0, r_max, n_bins + 1)
    r_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    in_range = r_flat <= r_max
    r_in = r_flat[in_range]
    v_in = v_flat[in_range]

    bin_idx = np.searchsorted(bin_edges[1:], r_in)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    counts = np.bincount(bin_idx, minlength=n_bins).astype(int)
    sums   = np.bincount(bin_idx, weights=v_in,    minlength=n_bins)
    sum2s  = np.bincount(bin_idx, weights=v_in**2, minlength=n_bins)

    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(counts > 0, sums / counts, np.nan)
        var   = np.where(
            counts > 1,
            (sum2s - sums**2 / np.where(counts > 0, counts, 1)) / np.maximum(counts - 1, 1),
            np.nan,
        )
        sems = np.where(
            counts > 1,
            np.sqrt(np.maximum(var, 0.0) / counts),
            np.nan,
        )

    return r_centers, means, sems, counts


# ---------------------------------------------------------------------------
# Peak finding for single OI line
# ---------------------------------------------------------------------------

def _theoretical_peak_radii_inline(
    params: SynthParams,
    lam: float,
    cfg: BinningConfig,
) -> np.ndarray:
    """Return r-positions (px) of Airy brightness maxima for a given wavelength."""
    r_fine    = np.linspace(0.0, cfg.r_max_px, 8000)
    cos_theta = np.cos(np.arctan(params.alpha * r_fine))
    F_coef    = 4.0 * params.R / (1.0 - params.R)**2
    phase     = 4.0 * np.pi * N_REF * (params.d_mm * 1e-3) * cos_theta / lam
    profile   = 1.0 / (1.0 + F_coef * np.sin(phase / 2.0)**2)
    p_range   = profile.max() - profile.min()
    pks, _    = find_peaks(
        profile,
        height=profile.min() + 0.30 * p_range,
        distance=len(r_fine) // 60,
    )
    return r_fine[pks]


def find_labeled_peaks(
    r_centers: np.ndarray,
    means: np.ndarray,
    params: SynthParams,
    derived: DerivedParams,
) -> list:
    """Detect peaks in the binned radial profile and label each as OI 630.2046 nm."""
    cfg   = get_binning_config(params.binning)
    valid = np.isfinite(means)
    if valid.sum() < 10:
        return []

    bg      = float(np.nanpercentile(means[valid], 5))
    p_range = float(np.nanpercentile(means[valid], 98)) - bg
    if p_range <= 0:
        return []

    min_dist = max(2, len(r_centers) // 100)
    pk_idx, _ = find_peaks(
        np.where(valid, means, bg),
        height=bg + 0.08 * p_range,
        distance=min_dist,
        prominence=0.04 * p_range,
    )
    if len(pk_idx) == 0:
        return []

    n_guard = max(2, min_dist)
    kept = []
    for i in pk_idx:
        left_ok  = (i >= n_guard) and np.all(means[i - n_guard : i] < means[i])
        right_ok = (i + n_guard < len(means)) and np.all(means[i + 1 : i + n_guard + 1] < means[i])
        if left_ok and right_ok:
            kept.append(i)
    pk_idx = np.array(kept, dtype=int)
    if len(pk_idx) == 0:
        return []

    r_oi = _theoretical_peak_radii_inline(params, derived.lambda_obs, cfg)

    labeled = []
    for i in pk_idx:
        r_obs = float(r_centers[i])
        adu   = float(means[i])
        amp   = adu - bg
        d_oi  = float(np.abs(r_oi - r_obs).min()) if len(r_oi) else np.inf
        labeled.append(dict(
            r_px=r_obs, adu=adu, amplitude=amp,
            wavelength_nm=630.2046, dist_theory_px=d_oi,
        ))
    return labeled


# ---------------------------------------------------------------------------
# Radial profile diagnostic figure
# ---------------------------------------------------------------------------

def make_radial_profile_figure(
    r_centers: np.ndarray,
    means: np.ndarray,
    sems: np.ndarray,
    labeled_peaks: list,
    params: SynthParams,
    derived: DerivedParams,
    out_dir: pathlib.Path,
    stem: str,
) -> pathlib.Path:
    """Save a radial-profile diagnostic figure with SEM band and peak table."""
    cfg = get_binning_config(params.binning)
    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[2.2, 1.0],
        hspace=0.40,
        left=0.08, right=0.97, top=0.93, bottom=0.04,
    )
    ax_prof = fig.add_subplot(gs[0])
    ax_tbl  = fig.add_subplot(gs[1])

    valid = np.isfinite(means)
    r_v   = r_centers[valid]
    m_v   = means[valid]
    s_v   = np.where(np.isfinite(sems[valid]), sems[valid], 0.0)

    ax_prof.fill_between(r_v, m_v - s_v, m_v + s_v,
                         color="#4488CC", alpha=0.30, label="±1 SEM")
    ax_prof.plot(r_v, m_v, color="#1155AA", lw=1.2,
                 marker="o", markersize=2.5, markevery=1, markerfacecolor="#1155AA",
                 label="Radial average")

    # Theoretical noise-free profile
    r_fine    = np.linspace(0.0, cfg.r_max_px, 6000)
    cos_theta = np.cos(np.arctan(params.alpha * r_fine))
    F_coef    = 4.0 * params.R / (1.0 - params.R)**2
    u_fine    = r_fine / cfg.r_max_px
    vig_fine  = derived.I0 * (1.0 + params.I1 * u_fine + params.I2 * u_fine**2)
    phase_fine = 4.0 * np.pi * N_REF * (params.d_mm * 1e-3) * cos_theta / derived.lambda_obs
    theory     = vig_fine / (1.0 + F_coef * np.sin(phase_fine / 2.0)**2) + OFFSET_ADU
    ax_prof.plot(r_fine, theory, color="#FF8800", lw=0.9, ls="--",
                 alpha=0.65, label="Theory (noise-free)")

    seen_oi = False
    for pk in labeled_peaks:
        lbl = None
        if not seen_oi:
            lbl = "OI 630.2046 nm peaks"
            seen_oi = True
        ax_prof.axvline(pk["r_px"], color="#2266EE", lw=0.9, ls=":", alpha=0.70, label=lbl)

    ax_prof.set_xlabel("Radius  r  (px)", fontsize=11)
    ax_prof.set_ylabel("Counts  (ADU)", fontsize=11)
    ax_prof.set_title(
        f"Radial Profile  |  d={params.d_mm} mm, R={params.R}, "
        f"v_los={params.v_los_ms:+.0f} m/s, SNR={params.snr_peak}\n"
        f"α={derived.alpha_rad_per_px:.4e} rad/px,  "
        f"I₀={derived.I0:.0f} ADU,  "
        f"λ_obs={derived.lambda_obs * 1e9:.4f} nm  "
        f"|  {len(labeled_peaks)} OI peaks detected",
        fontsize=9,
    )
    ax_prof.legend(fontsize=8, loc="upper right")
    ax_prof.grid(True, lw=0.4, alpha=0.5)

    ax_tbl.axis("off")
    MAX_ROWS = 40
    display_peaks = labeled_peaks[:MAX_ROWS]
    truncated     = len(labeled_peaks) > MAX_ROWS

    if display_peaks:
        col_labels = ["#", "r (px)", "Mean (ADU)", "Amplitude (ADU)",
                      "λ (nm)", "Δr vs theory (px)"]
        table_data = []
        for k, pk in enumerate(display_peaks, start=1):
            table_data.append([
                str(k),
                f"{pk['r_px']:.2f}",
                f"{pk['adu']:.1f}",
                f"{pk['amplitude']:.1f}",
                f"{pk['wavelength_nm']:.4f}",
                f"{pk['dist_theory_px']:.3f}",
            ])

        tbl = ax_tbl.table(
            cellText=table_data, colLabels=col_labels,
            loc="center", cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        font_size = 7.5 if len(display_peaks) > 25 else 8.5
        tbl.set_fontsize(font_size)
        row_h = 1.6 / max(len(display_peaks) + 1, 5)
        tbl.scale(1.0, max(0.9, row_h * len(display_peaks)))
        for ri in range(len(display_peaks)):
            for ci in range(len(col_labels)):
                tbl[ri + 1, ci].set_facecolor("#DDEEFF")
        note = f"  (first {MAX_ROWS} of {len(labeled_peaks)} shown)" if truncated else ""
        fig.text(
            0.5, -0.1,
            f"Detected OI 630.2046 nm peaks: {len(labeled_peaks)} total{note}",
            ha="center", va="bottom", fontsize=8.5,
        )
    else:
        ax_tbl.text(0.5, 0.5, "No peaks detected in radial profile.",
                    ha="center", va="center", fontsize=11,
                    transform=ax_tbl.transAxes)

    fig.suptitle(f"Z02 Radial Profile Diagnostic — {stem}", fontsize=11)
    out_path = out_dir / f"{stem}_radial_profile.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Interactive prompt helpers
# ---------------------------------------------------------------------------

def _validated_prompt(
    label: str,
    default: float,
    units: str,
    hard_min: float,
    hard_max: float,
    warn_min: float,
    warn_max: float,
) -> float:
    """Prompt for a float; enforce hard bounds and warn if outside recommended range."""
    while True:
        try:
            prompt_str = f"  {label}"
            if units:
                prompt_str += f" [{units}]"
            prompt_str += f"  (default {default}): "
            resp = input(prompt_str).strip()
            val  = float(default) if resp == "" else float(resp)
        except ValueError:
            print("  Invalid number — please try again.")
            continue

        if val < hard_min or val > hard_max:
            print(f"  Value out of hard bounds [{hard_min}, {hard_max}] — try again.")
            continue

        if val < warn_min or val > warn_max:
            yn = input(
                f"  Warning: {val} outside recommended range "
                f"[{warn_min}, {warn_max}]. Continue? (Y/n) "
            ).strip().lower()
            if yn not in ("", "y", "yes"):
                continue

        return val


def _int_prompt(label: str, default: int, choices: list) -> int:
    """Prompt for an integer constrained to choices."""
    while True:
        resp = input(f"  {label} {choices}  (default {default}): ").strip()
        if resp == "":
            return default
        try:
            val = int(resp)
        except ValueError:
            print(f"  Invalid — enter one of {choices}.")
            continue
        if val not in choices:
            print(f"  Must be one of {choices}.")
            continue
        return val


def prompt_all_params() -> SynthParams:
    """Interactively prompt for all synthesis parameters.

    Displays a banner, prompts in 5 groups, echoes a summary, and confirms.
    """
    banner = (
        "\n"
        "╔" + "═" * 62 + "╗\n"
        "║  Z02  Synthetic Science Image Generator  v1.0                ║\n"
        "║  WindCube SOC — soc_sewell                                   ║\n"
        "╚" + "═" * 62 + "╝\n"
        "\nSynthesises a matched science + dark image pair in authentic\n"
        "WindCube .bin format. The OI 630.2046 nm Airy fringe pattern is\n"
        "Doppler-shifted by the prompted line-of-sight velocity.\n"
        "Press <Enter> to accept the default shown in parentheses.\n"
    )
    print(banner)

    while True:
        print("\n──────────────────────────────────────────────────────────────")
        print(" GROUP 0  IMAGE GEOMETRY")
        print("──────────────────────────────────────────────────────────────")
        binning = _int_prompt("Detector binning (1=1×1 unbinned, 2=2×2 binned)", 2, [1, 2])
        cfg     = get_binning_config(binning)

        cx_hard_min = cfg.cx_default - cfg.ncols / 2.0
        cx_hard_max = cfg.cx_default + cfg.ncols / 2.0
        cy_hard_min = cfg.cy_default - cfg.active_rows / 2.0
        cy_hard_max = cfg.cy_default + cfg.active_rows / 2.0

        cx = _validated_prompt("Fringe centre column", cfg.cx_default, "px",
                               cx_hard_min, cx_hard_max,
                               cfg.cx_default - 50, cfg.cx_default + 50)
        cy = _validated_prompt("Fringe centre row",   cfg.cy_default, "px",
                               cy_hard_min, cy_hard_max,
                               cfg.cy_default - 50, cfg.cy_default + 50)

        print("\n──────────────────────────────────────────────────────────────")
        print(" GROUP 1  ETALON GEOMETRY")
        print("──────────────────────────────────────────────────────────────")
        d_mm  = _validated_prompt("Etalon gap d",   20.0005,         "mm",    15.0,  25.0, 19.5, 20.5)
        alpha = _validated_prompt("Plate scale α", cfg.alpha_default, "rad/px", 1e-5, 1e-3, 0.5e-4, 5e-4)

        print("\n──────────────────────────────────────────────────────────────")
        print(" GROUP 2  ETALON REFLECTIVITY")
        print("──────────────────────────────────────────────────────────────")
        R   = _validated_prompt("Effective reflectivity R", 0.725, "", 0.01, 0.99, 0.4, 0.92)
        N_R_hint = math.pi * math.sqrt(R) / (1.0 - R)
        print(f"    [Finesse N_R = π√R/(1−R); R={R:.3f} → N_R≈{N_R_hint:.1f}]")

        print("\n──────────────────────────────────────────────────────────────")
        print(" GROUP 3  INTENSITY ENVELOPE AND DETECTOR")
        print("──────────────────────────────────────────────────────────────")
        snr_peak = _validated_prompt("Peak SNR (OI 630.2046 nm fringe peak)", 50.0,  "",    1.0,  500.0, 10.0, 200.0)
        I1       = _validated_prompt("Linear vignetting coefficient I_1",     -0.1,  "",   -0.9,   0.9,  -0.5,  0.5)
        I2       = _validated_prompt("Quadratic vignetting coefficient I_2",   0.005, "",  -0.9,   0.9,  -0.5,  0.5)
        T_fp_c   = _validated_prompt("Focal plane temperature",               -20.0, "°C", -60.0, 30.0, -40.0,  0.0)

        print("\n──────────────────────────────────────────────────────────────")
        print(" GROUP 4  SOURCE VELOCITY")
        print("──────────────────────────────────────────────────────────────")
        print("  [Negative = blueshift (toward spacecraft); typical range -8000 to +1000 m/s]")
        lam_hint = LAM_OI * 1e9 * (1.0 + (-7500.0) / C_LIGHT_MS)
        print(f"  [λ_obs = 630.2046 × (1 + v/c); at -7500 m/s: λ_obs ≈ {lam_hint:.4f} nm]")
        v_los_ms = _validated_prompt(
            "Line-of-sight velocity v_los", -7500.0, "m/s",
            hard_min=-10000.0, hard_max=3000.0,
            warn_min=-8500.0,  warn_max=1500.0,
        )

        # Derived summary
        I_peak     = snr_to_ipeak(snr_peak, OFFSET_ADU)
        I0         = I_peak
        N_R        = math.pi * math.sqrt(R) / (1.0 - R)
        lambda_obs = LAM_OI * (1.0 + v_los_ms / C_LIGHT_MS)

        print("\n── PARAMETER SUMMARY ──")
        print(f"  {'Parameter':<32} {'Value':>14}")
        print(f"  {'-'*48}")
        entries = [
            ("binning",           binning),
            ("cx [px]",           cx),
            ("cy [px]",           cy),
            ("d_mm [mm]",         d_mm),
            ("alpha [rad/px]",    alpha),
            ("R",                 R),
            ("snr_peak",          snr_peak),
            ("I1",                I1),
            ("I2",                I2),
            ("T_fp_c [°C]",       T_fp_c),
            ("v_los_ms [m/s]",    v_los_ms),
        ]
        for name, val in entries:
            print(f"  {name:<32} {val:>14g}")
        print(f"  {'-'*48}")
        print(f"  {'(derived) I0 = I_peak [ADU]':<32} {I0:>14.1f}  single line")
        print(f"  {'(derived) N_R (finesse)':<32} {N_R:>14.2f}")
        print(f"  {'(derived) λ_obs [nm]':<32} {lambda_obs*1e9:>14.4f}")

        yn = input("\nProceed with synthesis? (Y/n) ").strip().lower()
        if yn in ("", "y", "yes"):
            break
        print("Re-entering parameters...\n")

    return SynthParams(
        binning=binning, cx=cx, cy=cy,
        d_mm=d_mm, alpha=alpha, R=R,
        snr_peak=snr_peak, I1=I1, I2=I2, T_fp_c=T_fp_c,
        v_los_ms=v_los_ms,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    """Interactive entry point: prompt, synthesise, write files, display."""
    params  = prompt_all_params()
    cfg     = get_binning_config(params.binning)
    derived = derive_secondary(params)

    if not check_vignetting_positive(derived.I0, params.I1, params.I2, cfg.r_max_px):
        print("ERROR: Vignetting envelope I(r) goes non-positive. Adjust I1/I2.")
        sys.exit(1)

    print(f"\n  α         = {derived.alpha_rad_per_px:.4e} rad/px")
    print(f"  I0        = {derived.I0:.1f} ADU  (single line; I_peak = I0)")
    print(f"  λ_obs     = {derived.lambda_obs * 1e9:.4f} nm")
    print(f"  Δλ        = {derived.delta_lam * 1e12:.3f} pm")
    print(f"  FSR       = {derived.FSR_m * 1e12:.3f} pm")
    print(f"  F         = {derived.finesse_F:.2f}")
    print(f"  N_R       = {derived.finesse_N_R:.2f}  (finesse)")
    print(f"  dark      = {derived.dark_rate:.6f} ADU/px/s at {params.T_fp_c}°C")

    image_float = synthesise_image(params, derived)

    ss          = np.random.SeedSequence()
    seed        = int(ss.entropy & 0xFFFFFFFF)
    rng         = np.random.default_rng(ss)
    exp_time_s  = 12000 * TIMER_PERIOD_S

    sci_px  = _apply_sci_noise(image_float, params, derived, exp_time_s, rng)
    dark_px = _apply_dark_noise(image_float.shape, derived, exp_time_s, rng)

    sci_frame  = build_full_frame(sci_px,  params, "science")
    dark_frame = build_full_frame(dark_px, params, "dark")

    default_out = pathlib.Path(os.environ.get(
        "Z02_OUTPUT_DIR",
        str(pathlib.Path.home() / "soc_synthesized_data"),
    ))
    out_dir = pathlib.Path(os.environ.get("Z02_OUTPUT_DIR", default_out))
    out_dir.mkdir(parents=True, exist_ok=True)

    ts        = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem      = f"{ts}_synth_z02_{cfg.label}"
    sci_name  = f"{ts}_sci_synth_z02_{cfg.label}.bin"
    dark_name = f"{ts}_dark_synth_z02_{cfg.label}.bin"

    sci_path  = out_dir / sci_name
    dark_path = out_dir / dark_name

    sci_frame.astype(">u2").tofile(str(sci_path))
    dark_frame.astype(">u2").tofile(str(dark_path))

    truth_path = out_dir / f"{stem}_truth.json"
    write_truth_json(params, derived, seed, sci_path, dark_path, truth_path)

    try:
        diag_path = make_diagnostic_figure(
            sci_frame, dark_frame, params, derived, out_dir, stem
        )
    except Exception as exc:
        diag_path = None
        print(f"  [WARN] Diagnostic figure failed: {exc}")

    r_centers, r_means, r_sems, _ = compute_radial_profile_from_image(
        sci_frame, cx=params.cx, cy=params.cy,
        r_max=cfg.r_max_px, n_bins=350,
    )
    labeled_peaks = find_labeled_peaks(r_centers, r_means, params, derived)
    print(f"\n  Radial profile: {len(r_centers)} bins, "
          f"{len(labeled_peaks)} OI peaks detected")

    try:
        rad_path = make_radial_profile_figure(
            r_centers, r_means, r_sems, labeled_peaks,
            params, derived, out_dir, stem,
        )
    except Exception as exc:
        rad_path = None
        print(f"  [WARN] Radial profile figure failed: {exc}")

    print("\nWrote:")
    print(f"  {sci_path}")
    print(f"  {dark_path}")
    print(f"  {truth_path}")
    if diag_path:
        print(f"  {diag_path}")
    if rad_path:
        print(f"  {rad_path}")
    try:
        if diag_path:
            os.startfile(str(diag_path))
        if rad_path:
            os.startfile(str(rad_path))
    except Exception:
        pass
    print("Done.")


if __name__ == "__main__":
    main()
