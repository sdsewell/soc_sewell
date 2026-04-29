"""
Z03 — Synthetic Calibration Image Generator
Spec:    specs/z03_synthetic_calibration_image_generator_spec_2026-04-28.md
Version: 1.5
Author:  Scott Sewell / HAO
Repo:    soc_sewell

Synthesises a matched calibration + dark image pair in authentic WindCube
.bin format (1-row header + pixel data), suitable for ingestion by Z01,
F01, or any downstream pipeline module.

Physics: ideal Airy transmission function with effective reflectivity R.
PSF broadening (sigma0/sigma1/sigma2) removed — degenerate with effective R.
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

OFFSET_ADU       = 5          # ADU — bias + read noise combined; fixed pedestal
DARK_REF_ADU_S   = 0.5       # ADU/px/s at T_REF_DARK_C
T_REF_DARK_C     = 10.0      # °C — dark reference temperature
T_DOUBLE_C       = 4.0        # °C — dark doubling interval
R_BINS           = 2000       # radial bins
N_REF            = 1.0        # refractive index, air gap
LAM_640          = 640.2248e-9  # m — Ne primary line (Burns et al. 1950)
LAM_638          = 638.2991e-9  # m — Ne secondary line (Burns et al. 1950)
TIMER_PERIOD_S   = 0.001      # s — seconds per exposure-count unit


# ---------------------------------------------------------------------------
# BinningConfig NamedTuple (§6)
# ---------------------------------------------------------------------------

class BinningConfig(NamedTuple):
    nrows:         int    # total rows including 1 header row
    ncols:         int    # total columns
    active_rows:   int    # nrows - 1 (pixel rows)
    n_meta_rows:   int    # header rows (always 1 in v1.5)
    cx_default:    float  # default fringe centre column
    cy_default:    float  # default fringe centre row (full-frame coords)
    r_max_px:      float  # max usable fringe radius, px
    alpha_default: float  # plate scale, rad/px
    pix_m:         float  # physical pixel pitch, m
    label:         str    # mode label for filenames


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
    d_mm:      float   # etalon gap, mm  (default 20.008)
    alpha:     float   # plate scale, rad/px  (mode-dependent default)
    # Group 2 — reflectivity
    R:         float   # effective reflectivity  (default 0.725)
    # Group 3 — intensity envelope and detector
    snr_peak:  float   # composite peak SNR  (default 50.0)
    I1:        float   # linear vignetting  (default -0.1)
    I2:        float   # quadratic vignetting  (default 0.005)
    T_fp_c:    float   # focal plane temperature, °C  (default 20.0)
    # Group 4 — source
    rel_638:   float   # 638/640 intensity ratio  (default 0.344)


# ---------------------------------------------------------------------------
# DerivedParams dataclass — computed from SynthParams
# ---------------------------------------------------------------------------

@dataclass
class DerivedParams:
    alpha_rad_per_px: float
    I_peak:           float   # composite two-line peak from snr_to_ipeak()
    I0:               float   # per-line amplitude = I_peak / (1 + rel_638)
    Y_B:              float   # alias for rel_638 (for truth JSON)
    FSR_m:            float
    finesse_F:        float
    finesse_N_R:      float
    dark_rate:        float   # ADU/px/s at T_fp_c


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def snr_to_ipeak(snr: float, offset: float) -> float:
    """Convert peak SNR to composite peak intensity (positive root of quadratic).

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
    cfg         = get_binning_config(params.binning)
    I_peak      = snr_to_ipeak(params.snr_peak, OFFSET_ADU)
    I0          = I_peak / (1.0 + params.rel_638)
    d_m         = params.d_mm * 1e-3
    FSR         = LAM_640**2 / (2.0 * N_REF * d_m)
    F           = 4.0 * params.R / (1.0 - params.R)**2
    N_R         = math.pi * math.sqrt(params.R) / (1.0 - params.R)
    dark_rate   = DARK_REF_ADU_S * 2.0**((params.T_fp_c - T_REF_DARK_C) / T_DOUBLE_C)
    return DerivedParams(
        alpha_rad_per_px = params.alpha,
        I_peak           = I_peak,
        I0               = I0,
        Y_B              = params.rel_638,
        FSR_m            = FSR,
        finesse_F        = F,
        finesse_N_R      = N_R,
        dark_rate        = dark_rate,
    )


# ---------------------------------------------------------------------------
# Synthesis (§9 Stage C) — inline airy_ideal, no PSF broadening
# ---------------------------------------------------------------------------

def synthesise_profile(
    params: SynthParams,
    derived: DerivedParams,
) -> tuple:
    """Build the noise-free 1D radial fringe profile using inline airy_ideal.

    Returns (profile_1d, r_grid).
    profile_1d includes OFFSET_ADU background and both Ne lines.
    """
    cfg       = get_binning_config(params.binning)
    r_grid    = np.linspace(0.0, cfg.r_max_px, R_BINS)
    theta     = np.arctan(params.alpha * r_grid)
    cos_theta = np.cos(theta)
    F_coef    = 4.0 * params.R / (1.0 - params.R)**2
    u         = r_grid / cfg.r_max_px
    vignette  = derived.I0 * (1.0 + params.I1 * u + params.I2 * u**2)

    def _airy(lam: float) -> np.ndarray:
        phase = 4.0 * np.pi * N_REF * (params.d_mm * 1e-3) * cos_theta / lam
        return vignette / (1.0 + F_coef * np.sin(phase / 2.0)**2)

    A640       = _airy(LAM_640)
    A638       = _airy(LAM_638)
    profile_1d = A640 + params.rel_638 * A638 + OFFSET_ADU
    return profile_1d, r_grid


def synthesise_image(params: SynthParams, derived: DerivedParams) -> np.ndarray:
    """Return noise-free pixel data (active_rows × ncols), float64.

    Excludes the 1-row header — caller adds header via build_full_frame().
    cy is adjusted for the 1-row header offset before passing to M02.
    """
    cfg        = get_binning_config(params.binning)
    profile_1d, r_grid = synthesise_profile(params, derived)

    # cy in full-frame coords → cy in pixel-data coords (subtract header rows)
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

def _apply_cal_noise(
    image_float: np.ndarray,
    params: SynthParams,
    derived: DerivedParams,
    exp_time_s: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply photon + dark noise to calibration pixel array; return uint16."""
    mean_dark   = derived.dark_rate * exp_time_s
    cal_signal  = image_float + mean_dark
    cal_noisy   = rng.poisson(np.maximum(cal_signal, 0)).astype(float) + OFFSET_ADU
    return np.clip(np.round(cal_noisy), 0, 16383).astype(np.uint16)


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
    words = struct.unpack(">4H", b)   # [MSW, w1, w2, LSW]
    h[w + 0] = words[3]               # LSW
    h[w + 1] = words[2]
    h[w + 2] = words[1]
    h[w + 3] = words[0]               # MSW


def _write_header_row(full_array: np.ndarray, image_type: str) -> None:
    """Write P01-compatible 1-row header into row 0 of full_array (in-place)."""
    nrows, ncols = full_array.shape
    h = np.zeros(ncols, dtype=np.uint16)

    h[0] = nrows
    h[1] = ncols
    h[2] = 12000   # 120 s exposure in centiseconds
    h[3] = 0       # exp_unit

    # words 4-7: etalon temp 24.0 °C
    _encode_f64(h, 4, 24.0)

    # words 8-11: lua_timestamp (Unix ms)
    ts_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    _encode_u64(h, 8, ts_ms)

    is_cal = image_type.lower().startswith("cal")

    # words 100-103: gpio_pwr_on — dark: shutter closed (bits 0 and 3)
    if not is_cal:
        h[100] = 1   # gpio_pwr_on[0]
        h[103] = 1   # gpio_pwr_on[3]

    # words 104-109: lamp_ch_array — cal: lamp on (bit 0)
    if is_cal:
        h[104] = 1   # lamp_ch_array[0]

    full_array[0, :] = h


def build_full_frame(
    pixel_uint16: np.ndarray,
    params: SynthParams,
    image_type: str,
) -> np.ndarray:
    """Combine pixel data and 1-row header into a full-frame uint16 array.

    pixel_uint16 shape: (active_rows, ncols).
    Returns shape: (nrows, ncols).
    """
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
    """Synthesise cal+dark pair, write .bin files and truth JSON.

    Returns (cal_path, dark_path, truth_path, stem).
    """
    cfg         = get_binning_config(params.binning)
    derived     = derive_secondary(params)

    if not check_vignetting_positive(derived.I0, params.I1, params.I2, cfg.r_max_px):
        raise ValueError("Vignetting envelope I(r) goes non-positive. Adjust I1/I2.")

    image_float = synthesise_image(params, derived)
    rng         = np.random.default_rng(seed)
    exp_time_s  = exp_time_cts * TIMER_PERIOD_S

    cal_px  = _apply_cal_noise(image_float, params, derived, exp_time_s, rng)
    dark_px = _apply_dark_noise(image_float.shape, derived, exp_time_s, rng)

    cal_frame  = build_full_frame(cal_px,  params, "cal")
    dark_frame = build_full_frame(dark_px, params, "dark")

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts        = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem      = f"{ts}_synth_z03_{cfg.label}"
    cal_name  = f"{ts}_cal_synth_z03_{cfg.label}.bin"
    dark_name = f"{ts}_dark_synth_z03_{cfg.label}.bin"

    cal_path  = out_dir / cal_name
    dark_path = out_dir / dark_name

    cal_frame.astype(">u2").tofile(str(cal_path))
    dark_frame.astype(">u2").tofile(str(dark_path))

    truth_path = out_dir / f"{stem}_truth.json"
    write_truth_json(params, derived, seed, cal_path, dark_path, truth_path)

    return cal_path, dark_path, truth_path, stem


# ---------------------------------------------------------------------------
# Truth JSON (§10)
# ---------------------------------------------------------------------------

def write_truth_json(
    params: SynthParams,
    derived: DerivedParams,
    seed: int,
    path_cal: pathlib.Path,
    path_dark: pathlib.Path,
    truth_path: pathlib.Path,
) -> None:
    """Write the _truth.json sidecar with v1.5 schema."""
    cfg = get_binning_config(params.binning)
    ts  = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    truth = {
        "z03_version":   "1.5",
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
            "rel_638":   params.rel_638,
        },
        "derived_params": {
            "alpha_rad_per_px":      derived.alpha_rad_per_px,
            "I_peak_adu":            derived.I_peak,
            "I0_adu":                derived.I0,
            "Y_B":                   derived.Y_B,
            "finesse_N_R":           derived.finesse_N_R,
            "finesse_coefficient_F": derived.finesse_F,
            "FSR_m":                 derived.FSR_m,
            "dark_rate_adu_px_s":    derived.dark_rate,
        },
        "fixed_constants": {
            "offset_adu":    OFFSET_ADU,
            "dark_ref_adu_s": DARK_REF_ADU_S,
            "T_ref_dark_c":  T_REF_DARK_C,
            "T_double_c":    T_DOUBLE_C,
            "R_bins":        R_BINS,
            "n_ref":         N_REF,
            "lam_640_m":     LAM_640,
            "lam_638_m":     LAM_638,
            "n_meta_rows":   cfg.n_meta_rows,
            "nrows":         cfg.nrows,
            "ncols":         cfg.ncols,
            "active_rows":   cfg.active_rows,
            "r_max_px":      cfg.r_max_px,
            "pix_m":         cfg.pix_m,
            "label":         cfg.label,
        },
        "output_cal_file":  str(path_cal.name),
        "output_dark_file": str(path_dark.name),
    }
    with open(truth_path, "w", encoding="utf-8") as fh:
        json.dump(truth, fh, indent=2)


# ---------------------------------------------------------------------------
# Diagnostic figure (§9 Stage G)
# ---------------------------------------------------------------------------

def make_diagnostic_figure(
    cal_img: np.ndarray,
    dark_img: np.ndarray,
    params: SynthParams,
    derived: DerivedParams,
    out_dir: pathlib.Path,
    stem: str,
) -> pathlib.Path:
    """Save a 2×2 diagnostic figure: images (top row), histograms (bottom row)."""
    cfg = get_binning_config(params.binning)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_cal, ax_dark, ax_hcal, ax_hdark = (
        axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    )

    cal_pixels = cal_img[cfg.n_meta_rows:, :]
    cal_max    = max(int(cal_pixels.max()), 1)

    im0 = ax_cal.imshow(cal_img, cmap="gray", vmin=0, vmax=cal_max)
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
        f"Ne: 640.2248 nm (×1.0)   638.2991 nm (×{params.rel_638})"
    )
    title_l4 = (
        f"I₀ (per-line) = {derived.I0:.1f} ADU   "
        f"I_peak (composite) = {derived.I_peak:.1f} ADU"
    )
    ax_cal.set_title(f"{title_l1}\n{title_l2}\n{title_l3}\n{title_l4}", fontsize=7)
    fig.colorbar(im0, ax=ax_cal, fraction=0.046, pad=0.04)

    dark_pixels = dark_img[cfg.n_meta_rows:, :]
    dark_max    = max(int(dark_pixels.max()), 1)

    im1 = ax_dark.imshow(dark_img, cmap="gray", vmin=0, vmax=dark_max)
    ax_dark.set_title(
        f"Dark image — T_fp={params.T_fp_c}°C   "
        f"dark_rate={derived.dark_rate:.4f} ADU/px/s",
        fontsize=9,
    )
    fig.colorbar(im1, ax=ax_dark, fraction=0.046, pad=0.04)

    ax_hcal.hist(cal_pixels.ravel(), bins=256, range=(0, cal_max), color="C0", linewidth=0)
    ax_hcal.set_xlim(0, cal_max)
    ax_hcal.set_xlabel("ADU")
    ax_hcal.set_ylabel("Pixel count")
    ax_hcal.set_title("Calibration histogram")

    ax_hdark.hist(dark_pixels.ravel(), bins=128, range=(0, dark_max), color="C1", linewidth=0)
    ax_hdark.set_xlim(0, dark_max)
    ax_hdark.set_xlabel("ADU")
    ax_hdark.set_ylabel("Pixel count")
    ax_hdark.set_title("Dark histogram")

    fig.suptitle(f"Z03 diagnostic — {stem}")
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

    Pixels in the first `exclude_rows` rows are ignored (header).
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
    sums   = np.bincount(bin_idx, weights=v_in,        minlength=n_bins)
    sum2s  = np.bincount(bin_idx, weights=v_in**2,     minlength=n_bins)

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
# Peak finding (using inline airy_ideal)
# ---------------------------------------------------------------------------

def _theoretical_peak_radii_inline(
    params: SynthParams,
    lam: float,
    cfg: BinningConfig,
) -> np.ndarray:
    """Return r-positions (px) of Airy brightness maxima for a single wavelength."""
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
) -> list:
    """Detect peaks in the binned radial profile and label each as 640 or 638 nm."""
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

    # Require n_guard strictly-lower neighbours on each side; rejects edge artefacts.
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

    r_640 = _theoretical_peak_radii_inline(params, LAM_640, cfg)
    r_638 = _theoretical_peak_radii_inline(params, LAM_638, cfg)

    labeled = []
    for i in pk_idx:
        r_obs = float(r_centers[i])
        adu   = float(means[i])
        amp   = adu - bg
        d640  = float(np.abs(r_640 - r_obs).min()) if len(r_640) else np.inf
        d638  = float(np.abs(r_638 - r_obs).min()) if len(r_638) else np.inf
        wl_nm = 640.2248 if d640 <= d638 else 638.2991
        labeled.append(dict(
            r_px=r_obs, adu=adu, amplitude=amp,
            wavelength_nm=wl_nm, dist_theory_px=min(d640, d638),
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

    def _airy_fine(lam):
        phase = 4.0 * np.pi * N_REF * (params.d_mm * 1e-3) * cos_theta / lam
        return vig_fine / (1.0 + F_coef * np.sin(phase / 2.0)**2)

    theory = _airy_fine(LAM_640) + params.rel_638 * _airy_fine(LAM_638) + OFFSET_ADU
    ax_prof.plot(r_fine, theory, color="#FF8800", lw=0.9, ls="--",
                 alpha=0.65, label="Theory (noise-free)")

    clr_640 = "#2266EE"; clr_638 = "#DD3333"
    seen_640 = seen_638 = False
    for pk in labeled_peaks:
        is_640 = pk["wavelength_nm"] > 639.0
        c   = clr_640 if is_640 else clr_638
        lbl = None
        if is_640 and not seen_640:
            lbl = "640.2 nm peaks"; seen_640 = True
        elif not is_640 and not seen_638:
            lbl = "638.3 nm peaks"; seen_638 = True
        ax_prof.axvline(pk["r_px"], color=c, lw=0.9, ls=":", alpha=0.70, label=lbl)

    ax_prof.set_xlabel("Radius  r  (px)", fontsize=11)
    ax_prof.set_ylabel("Counts  (ADU)", fontsize=11)
    n_640 = sum(1 for p in labeled_peaks if p["wavelength_nm"] > 639.0)
    n_638 = len(labeled_peaks) - n_640
    ax_prof.set_title(
        f"Radial Profile  |  d={params.d_mm} mm, R={params.R}, "
        f"rel_638={params.rel_638}, SNR={params.snr_peak}\n"
        f"α={derived.alpha_rad_per_px:.4e} rad/px,  "
        f"I₀={derived.I0:.0f} ADU  "
        f"|  {len(labeled_peaks)} peaks detected "
        f"({n_640} × 640 nm,  {n_638} × 638 nm)",
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
        table_data  = []
        row_colors  = []
        for k, pk in enumerate(display_peaks, start=1):
            is_640 = pk["wavelength_nm"] > 639.0
            table_data.append([
                str(k),
                f"{pk['r_px']:.2f}",
                f"{pk['adu']:.1f}",
                f"{pk['amplitude']:.1f}",
                f"{pk['wavelength_nm']:.4f}",
                f"{pk['dist_theory_px']:.3f}",
            ])
            row_colors.append("#DDEEFF" if is_640 else "#FFDDDD")

        tbl = ax_tbl.table(
            cellText=table_data, colLabels=col_labels,
            loc="center", cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        font_size = 7.5 if len(display_peaks) > 25 else 8.5
        tbl.set_fontsize(font_size)
        row_h = 1.6 / max(len(display_peaks) + 1, 5)
        tbl.scale(1.0, max(0.9, row_h * len(display_peaks)))
        for ri, col in enumerate(row_colors):
            for ci in range(len(col_labels)):
                tbl[ri + 1, ci].set_facecolor(col)
        note = f"  (first {MAX_ROWS} of {len(labeled_peaks)} shown)" if truncated else ""
        fig.text(
            0.5, -0.1,
            f"Detected peaks: {len(labeled_peaks)} total  "
            f"({n_640} × 640.2 nm   {n_638} × 638.3 nm){note}  "
            f"|  blue = 640 nm, red = 638 nm",
            ha="center", va="bottom", fontsize=8.5,
        )
    else:
        ax_tbl.text(0.5, 0.5, "No peaks detected in radial profile.",
                    ha="center", va="center", fontsize=11,
                    transform=ax_tbl.transAxes)

    fig.suptitle(f"Z03 Radial Profile Diagnostic — {stem}", fontsize=11)
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

    Displays a banner, prompts in 4 groups, echoes a summary, and confirms.
    """
    banner = (
        "\n"
        "╔" + "═" * 62 + "╗\n"
        "║  Z03  Synthetic Calibration Image Generator (v1.5)          ║\n"
        "║  WindCube SOC — soc_sewell                                   ║\n"
        "╚" + "═" * 62 + "╝\n"
        "\nThis script synthesises a matched calibration + dark image pair\n"
        "in authentic WindCube .bin format, suitable for ingestion by\n"
        "Z01, F01, or any downstream pipeline module.\n"
        "\nPhysics: ideal Airy function with effective reflectivity R.\n"
        "Press <Enter> to accept the default shown in parentheses.\n"
    )
    print(banner)

    while True:
        print("\n── GROUP 0  IMAGE GEOMETRY ──")
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

        print("\n── GROUP 1  ETALON GEOMETRY ──")
        d_mm  = _validated_prompt("Etalon gap d",   20.008,          "mm",    15.0,  25.0, 19.5, 20.5)
        alpha = _validated_prompt("Plate scale α", cfg.alpha_default, "rad/px", 1e-5, 1e-3, 0.5e-4, 5e-4)

        print("\n── GROUP 2  ETALON REFLECTIVITY ──")
        R = _validated_prompt("Effective reflectivity R", 0.725, "", 0.01, 0.99, 0.4, 0.92)
        N_R_hint = math.pi * math.sqrt(R) / (1.0 - R)
        print(f"    [Finesse N_R = π√R/(1−R); R={R:.3f} → N_R≈10.0]")

        print("\n── GROUP 3  INTENSITY ENVELOPE AND DETECTOR ──")
        snr_peak = _validated_prompt("Peak SNR (composite 640+638 nm peak)",  50.0,   "",    1.0,  500.0, 10.0,  200.0)
        I1       = _validated_prompt("Linear vignetting coefficient I_1",     -0.1,   "",   -0.9,   0.9,  -0.5,   0.5)
        I2       = _validated_prompt("Quadratic vignetting coefficient I_2",   0.005, "",   -0.9,   0.9,  -0.5,   0.5)
        T_fp_c   = _validated_prompt("Focal plane temperature",              10.0,  "°C", -60.0,  30.0, -40.0,   0.0)

        print("\n── GROUP 4  SOURCE ──")
        rel_638 = _validated_prompt("Intensity ratio 638nm/640nm (rel_638)", 0.344, "", 0.0, 2.0, 0.1, 1.0)

        # Derived summary
        I_peak = snr_to_ipeak(snr_peak, OFFSET_ADU)
        I0     = I_peak / (1.0 + rel_638)
        N_R    = math.pi * math.sqrt(R) / (1.0 - R)

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
            ("T_fp_c [°C]", T_fp_c),
            ("rel_638",           rel_638),
        ]
        for name, val in entries:
            print(f"  {name:<32} {val:>14g}")
        print(f"  {'-'*48}")
        print(f"  {'(derived) I_peak [ADU]':<32} {I_peak:>14.1f}  composite")
        print(f"  {'(derived) I0 [ADU]':<32} {I0:>14.1f}  per-line")
        print(f"  {'(derived) N_R (finesse)':<32} {N_R:>14.2f}")

        yn = input("\nProceed with synthesis? (Y/n) ").strip().lower()
        if yn in ("", "y", "yes"):
            break
        print("Re-entering parameters...\n")

    return SynthParams(
        binning=binning, cx=cx, cy=cy,
        d_mm=d_mm, alpha=alpha, R=R,
        snr_peak=snr_peak, I1=I1, I2=I2, T_fp_c=T_fp_c,
        rel_638=rel_638,
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

    # Print derived parameters
    print(f"\n  α       = {derived.alpha_rad_per_px:.4e} rad/px")
    print(f"  I0      = {derived.I0:.1f} ADU  (per-line)")
    print(f"  I_peak  = {derived.I_peak:.1f} ADU  (composite)")
    print(f"  FSR     = {derived.FSR_m * 1e12:.3f} pm")
    print(f"  F       = {derived.finesse_F:.2f}")
    print(f"  N_R     = {derived.finesse_N_R:.2f}  (finesse)")
    print(f"  dark    = {derived.dark_rate:.4f} ADU/px/s at {params.T_fp_c}°C")

    # Synthesize
    image_float = synthesise_image(params, derived)

    ss          = np.random.SeedSequence()
    seed        = int(ss.entropy & 0xFFFFFFFF)
    rng         = np.random.default_rng(ss)
    exp_time_s  = 12000 * TIMER_PERIOD_S

    cal_px  = _apply_cal_noise(image_float, params, derived, exp_time_s, rng)
    dark_px = _apply_dark_noise(image_float.shape, derived, exp_time_s, rng)

    cal_frame  = build_full_frame(cal_px,  params, "cal")
    dark_frame = build_full_frame(dark_px, params, "dark")

    # Output paths
    default_out = pathlib.Path(os.environ.get(
        "Z03_OUTPUT_DIR",
        str(pathlib.Path.home() / "soc_synthesized_data"),
    ))
    out_dir = pathlib.Path(os.environ.get("Z03_OUTPUT_DIR", default_out))
    out_dir.mkdir(parents=True, exist_ok=True)

    ts        = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem      = f"{ts}_synth_z03_{cfg.label}"
    cal_name  = f"{ts}_cal_synth_z03_{cfg.label}.bin"
    dark_name = f"{ts}_dark_synth_z03_{cfg.label}.bin"

    cal_path  = out_dir / cal_name
    dark_path = out_dir / dark_name

    cal_frame.astype(">u2").tofile(str(cal_path))
    dark_frame.astype(">u2").tofile(str(dark_path))

    truth_path = out_dir / f"{stem}_truth.json"
    write_truth_json(params, derived, seed, cal_path, dark_path, truth_path)

    # Diagnostic figures
    try:
        diag_path = make_diagnostic_figure(
            cal_frame, dark_frame, params, derived, out_dir, stem
        )
    except Exception as exc:
        diag_path = None
        print(f"  [WARN] Diagnostic figure failed: {exc}")

    r_centers, r_means, r_sems, _ = compute_radial_profile_from_image(
        cal_frame, cx=params.cx, cy=params.cy,
        r_max=cfg.r_max_px, n_bins=350,
    )
    labeled_peaks = find_labeled_peaks(r_centers, r_means, params)
    n_640 = sum(1 for p in labeled_peaks if p["wavelength_nm"] > 639.0)
    n_638 = len(labeled_peaks) - n_640
    print(f"\n  Radial profile: {len(r_centers)} bins, "
          f"{len(labeled_peaks)} peaks ({n_640}×640 nm, {n_638}×638 nm)")

    try:
        rad_path = make_radial_profile_figure(
            r_centers, r_means, r_sems, labeled_peaks,
            params, derived, out_dir, stem,
        )
    except Exception as exc:
        rad_path = None
        print(f"  [WARN] Radial profile figure failed: {exc}")

    print("\nWrote:")
    print(f"  {cal_path}")
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
