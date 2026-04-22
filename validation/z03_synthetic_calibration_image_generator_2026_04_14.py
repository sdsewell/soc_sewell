"""
Z03 — Synthetic Calibration Image Generator
Spec:    specs/z03_synthetic_calibration_image_generator_spec_2026-04-14.md
Version: 1.2
Author:  Scott Sewell / HAO
Repo:    soc_sewell

Synthesises a matched calibration + dark image pair in authentic WindCube
.bin format (260 × 276, uint16 little-endian), suitable for ingestion by
Z01, Z02, or any S01-based pipeline module.

All 10 M05 inversion free parameters are interactively prompted.
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
from typing import Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import find_peaks

# Allow running as a script from any working directory.
# validation/ is one level below repo root, so parents[1] is correct here.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.fpi.m01_airy_forward_model_2026_04_05 import (
    InstrumentParams,
    airy_modified,
)
from src.fpi.m02_calibration_synthesis_2026_04_05 import radial_profile_to_image

# ---------------------------------------------------------------------------
# Module-level fixed constants (non-inversion parameters)
# ---------------------------------------------------------------------------

SIGMA_READ    = 50.0          # ADU — CCD97 EM gain regime read noise
PIX_M         = 32.0e-6       # m   — CCD97 16 µm native × 2×2 binning
CX_DEFAULT    = 145         # px  — geometric centre, 276-col array
CY_DEFAULT    = 145         # px  — geometric centre, 260-row active region
R_MAX_PX      = 175        # px  — FlatSat/flight max usable radius
R_BINS        = 2000          # radial bins (must be ≥ 2000)
N_REF         = 1.0           # refractive index, air gap
NROWS, NCOLS  = 260, 276      # WindCube Level-0 image dimensions
N_META_ROWS   = 1             # S19 header rows
LAM_640       = 640.2248e-9   # m — Ne primary line (Burns et al. 1950)
LAM_638       = 638.2991e-9   # m — Ne secondary line (Burns et al. 1950)


# ---------------------------------------------------------------------------
# SynthParams dataclass — all 11 user-prompted parameters
# ---------------------------------------------------------------------------

@dataclass
class SynthParams:
    # Group 1 — etalon geometry
    d_mm:      float   # etalon gap, mm
    f_mm:      float   # imaging lens focal length, mm
    # Group 2 — reflectivity and PSF
    R:         float   # plate reflectivity
    sigma0:    float   # average PSF width, pixels
    sigma1:    float   # PSF sine variation, pixels
    sigma2:    float   # PSF cosine variation, pixels
    # Group 3 — intensity envelope and bias
    snr_peak:  float   # peak signal-to-noise ratio
    I1:        float   # linear vignetting coefficient
    I2:        float   # quadratic vignetting coefficient
    B_dc:      float   # bias pedestal, ADU
    # Group 4 — source (supplemental)
    rel_638:   float   # relative intensity of 638 nm line


# ---------------------------------------------------------------------------
# DerivedParams dataclass — computed from SynthParams
# ---------------------------------------------------------------------------

@dataclass
class DerivedParams:
    alpha_rad_per_px: float
    I0:               float   # derived from snr_peak via snr_to_ipeak()
    FSR_m:            float
    finesse_F:        float
    finesse_N:        float


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def snr_to_ipeak(snr: float, B_dc: float, sigma_read: float) -> float:
    """Convert peak SNR to peak intensity I0 (positive root of quadratic)."""
    noise_floor = B_dc + sigma_read ** 2
    return (snr ** 2 + math.sqrt(snr ** 4 + 4.0 * snr ** 2 * noise_floor)) / 2.0


def check_psf_positive(sigma0: float, sigma1: float, sigma2: float) -> bool:
    """Return True if sigma(r) >= 0 for all r.  Condition: sigma0 >= sqrt(sigma1^2 + sigma2^2)."""
    return sigma0 >= math.sqrt(sigma1 ** 2 + sigma2 ** 2)


def check_vignetting_positive(I0: float, I1: float, I2: float, r_max: float) -> bool:
    """Return True if I(r) = I0*(1 + I1*(r/r_max) + I2*(r/r_max)^2) > 0 for r in [0, r_max]."""
    r_test = np.array([0.0, r_max / 2.0, r_max])
    # Also check parabola vertex: d/dr[I1*u + I2*u^2] = 0 → u = -I1/(2*I2)
    if abs(I2) > 1e-12:
        u_vertex = -I1 / (2.0 * I2)
        if 0.0 <= u_vertex <= 1.0:
            r_test = np.append(r_test, u_vertex * r_max)
    vals = I0 * (1.0 + I1 * (r_test / r_max) + I2 * (r_test / r_max) ** 2)
    return bool(np.all(vals > 0))


def derive_secondary(params: SynthParams) -> DerivedParams:
    """Compute derived optical parameters from user-prompted SynthParams."""
    alpha = PIX_M / (params.f_mm * 1e-3)
    I0    = snr_to_ipeak(params.snr_peak, params.B_dc, SIGMA_READ)
    d_m   = params.d_mm * 1e-3
    FSR   = LAM_640 ** 2 / (2.0 * N_REF * d_m)
    F     = 4.0 * params.R / (1.0 - params.R) ** 2
    N_R   = math.pi * math.sqrt(params.R) / (1.0 - params.R)
    return DerivedParams(
        alpha_rad_per_px = alpha,
        I0               = I0,
        FSR_m            = FSR,
        finesse_F        = F,
        finesse_N        = N_R,
    )


def build_instrument_params(params: SynthParams, derived: DerivedParams) -> InstrumentParams:
    """Construct an InstrumentParams object from SynthParams + DerivedParams."""
    return InstrumentParams(
        t       = params.d_mm * 1e-3,
        R_refl  = params.R,
        n       = N_REF,
        alpha   = derived.alpha_rad_per_px,
        I0      = derived.I0,
        I1      = params.I1,
        I2      = params.I2,
        sigma0  = params.sigma0,
        sigma1  = params.sigma1,
        sigma2  = params.sigma2,
        B       = params.B_dc,
        r_max   = R_MAX_PX,
    )


def synthesise_profile(
    inst_params: InstrumentParams,
    rel_638: float,
) -> tuple:
    """
    Build the noise-free 1D radial fringe profile using airy_modified().

    Returns (profile_1d, r_grid).
    profile_1d includes the bias (inst_params.B) and both Ne lines.
    """
    r_grid = np.linspace(0.0, R_MAX_PX, R_BINS)

    A640 = airy_modified(
        r_grid,
        LAM_640,
        inst_params.t,
        inst_params.R_refl,
        inst_params.alpha,
        inst_params.n,
        inst_params.r_max,
        inst_params.I0,
        inst_params.I1,
        inst_params.I2,
        inst_params.sigma0,
        inst_params.sigma1,
        inst_params.sigma2,
    )
    A638 = airy_modified(
        r_grid,
        LAM_638,
        inst_params.t,
        inst_params.R_refl,
        inst_params.alpha,
        inst_params.n,
        inst_params.r_max,
        inst_params.I0,
        inst_params.I1,
        inst_params.I2,
        inst_params.sigma0,
        inst_params.sigma1,
        inst_params.sigma2,
    )

    profile_1d = A640 + rel_638 * A638 + inst_params.B
    return profile_1d, r_grid


def synthesise_image(params: SynthParams, derived: DerivedParams) -> np.ndarray:
    """
    Return noise-free calibration fringe image, shape (NROWS, NCOLS), float64.
    Uses airy_modified() via synthesise_profile() and radial_profile_to_image().
    """
    inst_params = build_instrument_params(params, derived)
    profile_1d, r_grid = synthesise_profile(inst_params, params.rel_638)

    # radial_profile_to_image uses a square grid; trim to NROWS after.
    image_sq = radial_profile_to_image(
        profile_1d, r_grid,
        image_size = NCOLS,    # 276 — use the wider dimension
        cx = CX_DEFAULT,
        cy = CY_DEFAULT,
        bias = params.B_dc,
    )

    # Trim to (NROWS, NCOLS) = (260, 276)
    # image_sq is (276, 276); keep the central 260 rows.
    row_start = (NCOLS - NROWS) // 2    # = 8
    image_out = image_sq[row_start : row_start + NROWS, :]
    return image_out


# ---------------------------------------------------------------------------
# Metadata helpers (S19 format)
# ---------------------------------------------------------------------------

def _set_f64(h: np.ndarray, w: int, value: float) -> None:
    """Store float64 into h[w:w+4] in p01 LE-word-order encoding.

    p01 decode:  unpack(">d", pack(">4H", h[w+3], h[w+2], h[w+1], h[w+0]))
    So we reverse the big-endian word order when storing.
    """
    b = struct.pack(">d", value)
    words = struct.unpack(">4H", b)   # [W3, W2, W1, W0] most→least significant
    h[w + 0] = words[3]               # least significant word
    h[w + 1] = words[2]
    h[w + 2] = words[1]
    h[w + 3] = words[0]               # most significant word


def _set_u64(h: np.ndarray, w: int, value: int) -> None:
    """Store uint64 into h[w:w+4] in p01 LE-word-order encoding.

    p01 decode:  sum(h[w+i] << (16*i) for i in range(4))
    """
    for i in range(4):
        h[w + i] = (value >> (16 * i)) & 0xFFFF


def embed_s19_header(image: np.ndarray, image_type: str) -> None:
    """Overwrite row 0 of image with a p01-compatible S19 binary header.

    Encodes into native uint16.  The caller must write the file as big-endian
    (image.astype('>u2').tofile(...)) so p01 can read it with dtype='>u2'.

    img_type classification (mirrors p01.parse_header):
      cal  — any lamp_ch_array word nonzero
      dark — shutter closed (gpio_pwr_on[0]==1 and gpio_pwr_on[3]==1)
    """
    h = np.zeros(NCOLS, dtype=np.uint16)

    # words 0-3: geometry + exposure
    h[0] = NROWS           # rows = 260
    h[1] = NCOLS           # cols = 276
    h[2] = 12000           # exp_time: 120 s expressed in centiseconds
    h[3] = 0               # exp_unit

    # words 4-7: ccd_temp1 = 24.0 °C
    _set_f64(h, 4, 24.0)

    # words 8-11: lua_timestamp (Unix ms, used by p01 for utc_timestamp)
    ts_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    _set_u64(h, 8, ts_ms)

    # words 12-15: adcs_timestamp = 0 (zeros by default)

    # words 16-83: spacecraft pos / vel / attitude = 0 (zeros by default)

    # words 84-87: etalon_temps[0] = 24.0 °C
    _set_f64(h, 84, 24.0)
    # words 88-99: etalon_temps[1,2,3] = 0.0 (zeros by default)

    is_cal = image_type.lower().startswith("cal")

    # words 100-103: gpio_pwr_on
    # dark: gpio_pwr_on[0]==1 and gpio_pwr_on[3]==1 → shutter closed
    if not is_cal:
        h[100] = 1   # gpio_pwr_on[0]
        h[103] = 1   # gpio_pwr_on[3]

    # words 104-109: lamp_ch_array
    # cal: lamp_ch_array[0] nonzero → p01 sets img_type="cal"
    if is_cal:
        h[104] = 1   # lamp_ch_array[0]

    image[0, :] = h


# ---------------------------------------------------------------------------
# Truth JSON
# ---------------------------------------------------------------------------

def write_truth_json(
    params: SynthParams,
    derived: DerivedParams,
    seed: int,
    path_cal: pathlib.Path,
    path_dark: pathlib.Path,
    truth_path: pathlib.Path,
) -> None:
    """Write the _truth.json sidecar with all parameters and file references."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    truth = {
        "z03_version":   "1.2",
        "timestamp_utc": ts,
        "random_seed":   seed,
        "user_params": {
            "d_mm":      params.d_mm,
            "f_mm":      params.f_mm,
            "R":         params.R,
            "sigma0":    params.sigma0,
            "sigma1":    params.sigma1,
            "sigma2":    params.sigma2,
            "snr_peak":  params.snr_peak,
            "I1":        params.I1,
            "I2":        params.I2,
            "B_dc":      params.B_dc,
            "rel_638":   params.rel_638,
        },
        "derived_params": {
            "alpha_rad_per_px":      derived.alpha_rad_per_px,
            "I0_adu":                derived.I0,
            "FSR_m":                 derived.FSR_m,
            "finesse_coefficient_F": derived.finesse_F,
            "finesse_N":             derived.finesse_N,
        },
        "fixed_defaults": {
            "sigma_read":  SIGMA_READ,
            "cx":          CX_DEFAULT,
            "cy":          CY_DEFAULT,
            "pix_m":       PIX_M,
            "r_max_px":    R_MAX_PX,
            "R_bins":      R_BINS,
            "nrows":       NROWS,
            "ncols":       NCOLS,
            "n_ref":       N_REF,
            "lam_640_m":   LAM_640,
            "lam_638_m":   LAM_638,
        },
        "output_cal_file":  str(path_cal.name),
        "output_dark_file": str(path_dark.name),
    }
    with open(truth_path, "w", encoding="utf-8") as fh:
        json.dump(truth, fh, indent=2)


# ---------------------------------------------------------------------------
# Diagnostic figure
# ---------------------------------------------------------------------------

def make_diagnostic_figure(
    cal_img: np.ndarray,
    dark_img: np.ndarray,
    params: SynthParams,
    out_dir: pathlib.Path,
    stem: str,
) -> pathlib.Path:
    """Save a 2×2 diagnostic figure: images (top row), histograms (bottom row)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_cal, ax_dark, ax_hcal, ax_hdark = (
        axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    )

    # --- calibration image (fixed 14-bit scale) ---
    im0 = ax_cal.imshow(cal_img, cmap="gray", vmin=0, vmax=16383)
    title_line1 = (
        f"d={params.d_mm} mm   f={params.f_mm} mm   "
        f"R={params.R}   SNR={params.snr_peak}"
    )
    title_line2 = (
        f"\u03c3\u2080={params.sigma0}   I\u2081={params.I1}   "
        f"I\u2082={params.I2}   B={params.B_dc} ADU"
    )
    title_line3 = (
        f"Ne: 640.2248 nm (\u00d71.0)   "
        f"638.2991 nm (\u00d7{params.rel_638})"
    )
    ax_cal.set_title(f"{title_line1}\n{title_line2}\n{title_line3}", fontsize=8)
    fig.colorbar(im0, ax=ax_cal, fraction=0.046, pad=0.04)

    # Exclude S19 metadata rows from dark scaling — header bytes packed as
    # uint16 can exceed 16383 and would blow out autoscale.
    dark_pixels = dark_img[N_META_ROWS:, :]
    dark_max = int(dark_pixels.max())

    # --- dark image (autoscale 0 → max pixel value, metadata rows excluded) ---
    im1 = ax_dark.imshow(dark_img, cmap="gray", vmin=0, vmax=dark_max)
    ax_dark.set_title(
        f"Dark image \u2014 B={params.B_dc} ADU, "
        f"\u03c3_read={SIGMA_READ} ADU",
        fontsize=9,
    )
    fig.colorbar(im1, ax=ax_dark, fraction=0.046, pad=0.04)

    # --- calibration histogram (fixed 14-bit x-range) ---
    ax_hcal.hist(cal_img.ravel(), bins=256, range=(0, 16383), color="C0", linewidth=0)
    ax_hcal.set_xlim(0, 16383)
    ax_hcal.set_xlabel("ADU")
    ax_hcal.set_ylabel("Pixel count")
    ax_hcal.set_title("Calibration histogram")

    # --- dark histogram (0 → max pixel value, metadata rows excluded) ---
    ax_hdark.hist(dark_pixels.ravel(), bins=128, range=(0, dark_max), color="C1", linewidth=0)
    ax_hdark.set_xlim(0, dark_max)
    ax_hdark.set_xlabel("ADU")
    ax_hdark.set_ylabel("Pixel count")
    ax_hdark.set_title("Dark histogram")

    fig.suptitle(f"Z03 diagnostic \u2014 {stem}")
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
    exclude_rows: int = N_META_ROWS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a radially averaged profile from a 2-D image.

    Pixels in the first `exclude_rows` rows are ignored (S19 metadata header).
    Each bin spans r_max / n_bins pixels in radius.

    Returns
    -------
    r_centers : (n_bins,) float64 — bin centre radii in pixels
    means     : (n_bins,) float64 — per-bin mean ADU (NaN for empty bins)
    sems      : (n_bins,) float64 — per-bin SEM  (NaN if n < 2)
    counts    : (n_bins,) int     — number of pixels per bin
    """
    nrows, ncols = image.shape
    col_idx = np.arange(ncols)
    row_idx = np.arange(exclude_rows, nrows)
    col_grid, row_grid = np.meshgrid(col_idx, row_idx)

    r_flat = np.sqrt((col_grid - cx) ** 2 + (row_grid - cy) ** 2).ravel()
    v_flat = image[exclude_rows:, :].astype(np.float64).ravel()

    bin_edges = np.linspace(0.0, r_max, n_bins + 1)
    r_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    in_range = r_flat <= r_max
    r_in = r_flat[in_range]
    v_in = v_flat[in_range]

    bin_idx = np.searchsorted(bin_edges[1:], r_in)   # 0-based, clipped at n_bins-1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    counts = np.bincount(bin_idx, minlength=n_bins).astype(int)
    sums   = np.bincount(bin_idx, weights=v_in,      minlength=n_bins)
    sum2s  = np.bincount(bin_idx, weights=v_in ** 2, minlength=n_bins)

    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(counts > 0, sums / counts, np.nan)
        var   = np.where(
            counts > 1,
            (sum2s - sums ** 2 / np.where(counts > 0, counts, 1)) / np.maximum(counts - 1, 1),
            np.nan,
        )
        sems  = np.where(
            counts > 1,
            np.sqrt(np.maximum(var, 0.0) / counts),
            np.nan,
        )

    return r_centers, means, sems, counts


# ---------------------------------------------------------------------------
# Peak finding and 640 / 638 nm labelling
# ---------------------------------------------------------------------------

def _theoretical_peak_radii(inst_params: "InstrumentParams", lam: float) -> np.ndarray:
    """
    Return r-positions (px) of Airy brightness maxima for a single wavelength.

    Evaluates airy_modified on a fine grid and detects peaks with find_peaks.
    """
    r_fine = np.linspace(0.0, inst_params.r_max, 8000)
    profile = airy_modified(
        r_fine, lam,
        inst_params.t, inst_params.R_refl, inst_params.alpha, inst_params.n,
        inst_params.r_max, inst_params.I0, inst_params.I1, inst_params.I2,
        inst_params.sigma0, inst_params.sigma1, inst_params.sigma2,
    )
    p_range = profile.max() - profile.min()
    pks, _ = find_peaks(
        profile,
        height=profile.min() + 0.30 * p_range,
        distance=len(r_fine) // 60,
    )
    return r_fine[pks]


def find_labeled_peaks(
    r_centers: np.ndarray,
    means: np.ndarray,
    inst_params: "InstrumentParams",
    rel_638: float,
) -> list:
    """
    Detect peaks in the binned radial profile and label each as 640 or 638 nm.

    Each peak is assigned to whichever wavelength's nearest theoretical ring
    is closer in radius.

    Returns a list of dicts with keys:
        r_px, adu, amplitude, wavelength_nm, dist_theory_px
    """
    valid = np.isfinite(means)
    if valid.sum() < 10:
        return []

    bg = float(np.nanpercentile(means[valid], 5))
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

    r_640 = _theoretical_peak_radii(inst_params, LAM_640)
    r_638 = _theoretical_peak_radii(inst_params, LAM_638)

    labeled = []
    for i in pk_idx:
        r_obs = float(r_centers[i])
        adu   = float(means[i])
        amp   = adu - bg

        d640 = float(np.abs(r_640 - r_obs).min()) if len(r_640) else np.inf
        d638 = float(np.abs(r_638 - r_obs).min()) if len(r_638) else np.inf

        wl_nm = 640.2248 if d640 <= d638 else 638.2991
        labeled.append(
            dict(
                r_px=r_obs,
                adu=adu,
                amplitude=amp,
                wavelength_nm=wl_nm,
                dist_theory_px=min(d640, d638),
            )
        )

    return labeled


# ---------------------------------------------------------------------------
# Radial profile diagnostic figure
# ---------------------------------------------------------------------------

def make_radial_profile_figure(
    r_centers: np.ndarray,
    means: np.ndarray,
    sems: np.ndarray,
    labeled_peaks: list,
    inst_params: "InstrumentParams",
    params: "SynthParams",
    derived: "DerivedParams",
    out_dir: pathlib.Path,
    stem: str,
) -> pathlib.Path:
    """
    Save a radial-profile diagnostic figure with SEM band and peak table.

    Layout (tall figure):
      top 65%   — profile line + SEM band + theoretical curve + peak markers
      bottom 35% — table of detected peaks coloured by wavelength
    """
    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[2.2, 1.0],
        hspace=0.40,
        left=0.08, right=0.97, top=0.93, bottom=0.04,
    )
    ax_prof = fig.add_subplot(gs[0])
    ax_tbl  = fig.add_subplot(gs[1])

    # ── Profile panel ────────────────────────────────────────────────────────
    valid = np.isfinite(means)
    r_v   = r_centers[valid]
    m_v   = means[valid]
    s_v   = np.where(np.isfinite(sems[valid]), sems[valid], 0.0)

    # SEM band first (behind the line)
    ax_prof.fill_between(r_v, m_v - s_v, m_v + s_v,
                         color="#4488CC", alpha=0.30, label="±1 SEM")
    ax_prof.plot(r_v, m_v, color="#1155AA", lw=1.2, label="Radial average")

    # Theoretical noise-free profile for reference
    r_fine = np.linspace(0.0, inst_params.r_max, 6000)
    A640t = airy_modified(
        r_fine, LAM_640,
        inst_params.t, inst_params.R_refl, inst_params.alpha, inst_params.n,
        inst_params.r_max, inst_params.I0, inst_params.I1, inst_params.I2,
        inst_params.sigma0, inst_params.sigma1, inst_params.sigma2,
    )
    A638t = airy_modified(
        r_fine, LAM_638,
        inst_params.t, inst_params.R_refl, inst_params.alpha, inst_params.n,
        inst_params.r_max, inst_params.I0, inst_params.I1, inst_params.I2,
        inst_params.sigma0, inst_params.sigma1, inst_params.sigma2,
    )
    theory = A640t + params.rel_638 * A638t + inst_params.B
    ax_prof.plot(r_fine, theory, color="#FF8800", lw=0.9, ls="--",
                 alpha=0.65, label="Theory (noise-free)")

    # Vertical peak markers coloured by wavelength
    clr_640 = "#2266EE"; clr_638 = "#DD3333"
    seen_640 = seen_638 = False
    for pk in labeled_peaks:
        is_640 = pk["wavelength_nm"] > 639.0   # 640.2248 > 639; 638.2991 < 639
        c = clr_640 if is_640 else clr_638
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
        f"α={derived.alpha_rad_per_px:.4e} rad/px,  I₀={derived.I0:.0f} ADU  "
        f"|  {len(labeled_peaks)} peaks detected "
        f"({n_640} × 640 nm,  {n_638} × 638 nm)",
        fontsize=9,
    )
    ax_prof.legend(fontsize=8, loc="upper right")
    ax_prof.grid(True, lw=0.4, alpha=0.5)

    # ── Peak table ────────────────────────────────────────────────────────────
    ax_tbl.axis("off")

    MAX_ROWS = 40
    display_peaks = labeled_peaks[:MAX_ROWS]
    truncated = len(labeled_peaks) > MAX_ROWS

    if display_peaks:
        col_labels = ["#", "r (px)", "Mean (ADU)", "Amplitude (ADU)",
                      "λ (nm)", "Δr vs theory (px)"]
        table_data = []
        row_colors = []
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
            cellText=table_data,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
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
        ax_tbl.set_title(
            f"Detected peaks: {len(labeled_peaks)} total  "
            f"({n_640} × 640.2 nm   {n_638} × 638.3 nm){note}  "
            f"|  blue rows = 640 nm, red rows = 638 nm",
            fontsize=8.5,
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
    """Prompt the user for a float, enforcing hard bounds and warning range."""
    while True:
        try:
            prompt_str = f"  {label}"
            if units:
                prompt_str += f" [{units}]"
            prompt_str += f" (default {default}): "
            resp = input(prompt_str).strip()
            val = float(default) if resp == "" else float(resp)
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


def prompt_all_params() -> SynthParams:
    """
    Interactively prompt the user for all 11 synthesis parameters.

    Displays a banner, prompts in 4 groups, echoes a summary table,
    and asks Y/n before returning. Loops if the user enters 'n'.
    """
    banner = (
        "\n"
        "\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
        "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
        "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
        "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
        "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
        "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557\n"
        "\u2551  Z03  Synthetic Calibration Image Generator                  \u2551\n"
        "\u2551  WindCube SOC \u2014 soc_sewell                                   \u2551\n"
        "\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
        "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
        "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
        "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
        "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
        "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255d\n"
        "\nThis script synthesises a matched calibration + dark image pair\n"
        "in authentic WindCube .bin format, suitable for ingestion by\n"
        "Z01, Z02, or any S01-based pipeline module.\n"
        "\nYou will be prompted for 10 instrument/fringe parameters (all\n"
        "free parameters of the M05 inversion) plus 1 source parameter.\n"
        "Press <Enter> to accept the default shown in parentheses.\n"
    )
    print(banner)

    while True:
        print("\n\u2500\u2500 GROUP 1  ETALON GEOMETRY \u2500\u2500")
        d_mm    = _validated_prompt("Etalon gap d",                  20.0001,"mm",  19.0,  21.0, 19.9,  20.1)
        f_mm    = _validated_prompt("Imaging lens focal length f",  200, "mm", 100.0, 300.0, 180.0, 250.0)

        print("\n\u2500\u2500 GROUP 2  REFLECTIVITY AND PSF \u2500\u2500")
        R       = _validated_prompt("Plate reflectivity R",           0.50,  "",    0.01,  0.99,  0.3,   0.85)
        sigma0  = _validated_prompt("Average PSF width sigma_0",      0.5,   "px",  0.0,   5.0,   0.0,   2.0)
        sigma1  = _validated_prompt("PSF sine variation sigma_1",     0.1,   "px", -3.0,   3.0,  -1.0,   1.0)
        sigma2  = _validated_prompt("PSF cosine variation sigma_2",  -0.05,  "px", -3.0,   3.0,  -1.0,   1.0)

        print("\n\u2500\u2500 GROUP 3  INTENSITY ENVELOPE AND BIAS \u2500\u2500")
        snr_peak = _validated_prompt("Peak SNR",                     50.0,  "",    1.0,  500.0, 10.0, 200.0)
        I1       = _validated_prompt("Linear vignetting coeff I_1",  -0.1,  "",   -0.9,   0.9,  -0.5,  0.5)
        I2       = _validated_prompt("Quadratic vignetting coeff I_2", 0.005, "",  -0.9,  0.9,  -0.5,  0.5)
        B_dc     = _validated_prompt("Bias pedestal B",              6500.0, "ADU", 0.0, 10000.0, 100.0, 8000.0)

        print("\n\u2500\u2500 GROUP 4  SOURCE (not an inversion free parameter) \u2500\u2500")
        rel_638  = _validated_prompt("Relative intensity 638 nm / 640 nm", 0.3, "", 0.0, 2.0, 0.1, 1.0)

        # Summary table
        print("\n\u2500\u2500 PARAMETER SUMMARY \u2500\u2500")
        print(f"  {'Parameter':<30} {'Value':>12}")
        print(f"  {'-'*44}")
        entries = [
            ("d_mm [mm]",      d_mm),
            ("f_mm [mm]",      f_mm),
            ("R",              R),
            ("sigma0 [px]",    sigma0),
            ("sigma1 [px]",    sigma1),
            ("sigma2 [px]",    sigma2),
            ("snr_peak",       snr_peak),
            ("I1",             I1),
            ("I2",             I2),
            ("B_dc [ADU]",     B_dc),
            ("rel_638",        rel_638),
        ]
        for name, val in entries:
            print(f"  {name:<30} {val:>12g}")

        yn = input("\nProceed with synthesis? (Y/n) ").strip().lower()
        if yn in ("", "y", "yes"):
            break
        print("Re-entering parameters...\n")

    return SynthParams(
        d_mm=d_mm, f_mm=f_mm, R=R,
        sigma0=sigma0, sigma1=sigma1, sigma2=sigma2,
        snr_peak=snr_peak, I1=I1, I2=I2, B_dc=B_dc,
        rel_638=rel_638,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    """Interactive entry point: prompt, synthesise, write files, display."""
    params = prompt_all_params()
    derived = derive_secondary(params)

    # Physics-consistency checks
    if not check_psf_positive(params.sigma0, params.sigma1, params.sigma2):
        print(
            "ERROR: PSF sigma(r) goes negative. "
            "Increase sigma0 or reduce |sigma1|/|sigma2|."
        )
        sys.exit(1)

    if not check_vignetting_positive(derived.I0, params.I1, params.I2, R_MAX_PX):
        print("ERROR: Vignetting envelope I(r) goes non-positive. Adjust I1/I2.")
        sys.exit(1)

    # Print derived parameters
    print(f"\n  alpha = {derived.alpha_rad_per_px:.4e} rad/px")
    print(f"  I0    = {derived.I0:.1f} ADU")
    print(f"  FSR   = {derived.FSR_m * 1e12:.3f} pm")
    print(f"  F     = {derived.finesse_F:.2f}")
    print(f"  N_F   = {derived.finesse_N:.2f}")

    # Synthesise noise-free float image
    I_float = synthesise_image(params, derived)

    # Apply noise model
    ss   = np.random.SeedSequence()
    seed = int(ss.entropy)
    rng  = np.random.default_rng(ss)

    signal_counts = rng.poisson(np.clip(I_float, 0, None)).astype(np.float64)
    read_noise    = rng.standard_normal(size=signal_counts.shape) * SIGMA_READ
    image_cal     = np.clip(signal_counts + read_noise, 0, 16383).astype(np.uint16)

    dark_float  = np.full((NROWS, NCOLS), params.B_dc, dtype=np.float64)
    dark_counts = rng.poisson(dark_float).astype(np.float64)
    dark_read   = rng.standard_normal(size=dark_float.shape) * SIGMA_READ
    image_dark  = np.clip(dark_counts + dark_read, 0, 16383).astype(np.uint16)

    # Embed S19 binary header in row 0 (p01-compatible)
    embed_s19_header(image_cal, "Cal")
    embed_s19_header(image_dark, "Dark")

    # Output paths
    default_out = pathlib.Path(os.environ.get(
        "Z03_OUTPUT_DIR",
        str(pathlib.Path(__file__).resolve().parents[3] / "GitHub/soc_synthesized_data"),
    ))
    out_dir = pathlib.Path(os.environ.get("Z03_OUTPUT_DIR", default_out))
    out_dir.mkdir(parents=True, exist_ok=True)

    ts        = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem      = f"{ts}_synth_z03"
    cal_name  = f"{ts}_cal_synth_z03.bin"
    dark_name = f"{ts}_dark_synth_z03.bin"

    path_cal  = out_dir / cal_name
    path_dark = out_dir / dark_name

    # Write big-endian so p01 can read with dtype=">u2"
    image_cal.astype(">u2").tofile(str(path_cal))
    image_dark.astype(">u2").tofile(str(path_dark))

    truth_path = out_dir / f"{stem}_truth.json"
    write_truth_json(params, derived, seed, path_cal, path_dark, truth_path)

    # Diagnostic figure (images + histograms)
    try:
        diag_path = make_diagnostic_figure(image_cal, image_dark, params, out_dir, stem)
    except Exception:
        diag_path = None

    # Radial profile from the noisy calibration image
    # cy in the trimmed image: CY_DEFAULT was set in the 276×276 square image;
    # after removing row_start=(276-260)//2=8 rows from the top, the centre row shifts.
    cy_trimmed = float(CY_DEFAULT - (NCOLS - NROWS) // 2)
    r_centers, r_means, r_sems, r_counts = compute_radial_profile_from_image(
        image_cal,
        cx=float(CX_DEFAULT),
        cy=cy_trimmed,
        r_max=float(R_MAX_PX),
        n_bins=350,
    )

    inst_params = build_instrument_params(params, derived)
    labeled_peaks = find_labeled_peaks(r_centers, r_means, inst_params, params.rel_638)

    n_640 = sum(1 for p in labeled_peaks if p["wavelength_nm"] > 639.0)
    n_638 = len(labeled_peaks) - n_640
    print(f"\n  Radial profile: {len(r_centers)} bins, "
          f"{len(labeled_peaks)} peaks detected "
          f"({n_640} × 640 nm, {n_638} × 638 nm)")
    if labeled_peaks:
        print(f"  {'#':>3}  {'r (px)':>8}  {'ADU':>8}  {'Amp':>8}  {'λ (nm)':>10}  "
              f"{'Δr_theory':>10}")
        for k, pk in enumerate(labeled_peaks, start=1):
            print(f"  {k:>3}  {pk['r_px']:>8.2f}  {pk['adu']:>8.1f}  "
                  f"{pk['amplitude']:>8.1f}  {pk['wavelength_nm']:>10.4f}  "
                  f"{pk['dist_theory_px']:>10.3f}")

    try:
        rad_path = make_radial_profile_figure(
            r_centers, r_means, r_sems, labeled_peaks,
            inst_params, params, derived, out_dir, stem,
        )
    except Exception as exc:
        rad_path = None
        print(f"  [WARN] Radial profile figure failed: {exc}")

    print("\nWrote:")
    print("  ", path_cal)
    print("  ", path_dark)
    print("  ", truth_path)
    if diag_path:
        print("  ", diag_path)
    if rad_path:
        print("  ", rad_path)
    if diag_path:
        os.startfile(str(diag_path))
    if rad_path:
        os.startfile(str(rad_path))
    print("Done.")


if __name__ == "__main__":
    main()
