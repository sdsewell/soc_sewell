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
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
CX_DEFAULT    = 137.5         # px  — geometric centre, 276-col array
CY_DEFAULT    = 129.5         # px  — geometric centre, 260-row active region
R_MAX_PX      = 110.0         # px  — FlatSat/flight max usable radius
R_BINS        = 2000          # radial bins (must be ≥ 2000)
N_REF         = 1.0           # refractive index, air gap
NROWS, NCOLS  = 260, 276      # WindCube Level-0 image dimensions
N_META_ROWS   = 4             # S19 header rows
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

def build_metadata_dict(image_type: str, exposure_ms: int = 120000) -> dict:
    """Build S19-compliant metadata dict for a calibration or dark image."""
    now = datetime.now(timezone.utc)
    is_cal = image_type.lower().startswith("cal")
    meta = {
        "image_type":     image_type,
        "n_rows":         NROWS,
        "n_cols":         NCOLS,
        "binning":        2,
        "shutter_status": "Open" if is_cal else "Closed",
        "cal_lamp_1":     "C1_On" if is_cal else "C1_Off",
        "cal_lamp_2":     "C2_Off",
        "cal_lamp_3":     "C3_Off",
        "date_utc":       now.strftime("%Y%m%d"),
        "time_utc":       now.strftime("%H%M%S"),
        "exposure_ms":    exposure_ms,
        "etalon_temp_1":  24.00,
    }
    return meta


def embed_metadata_rows(image: np.ndarray, meta: dict) -> None:
    """Encode metadata as JSON bytes and write into the first N_META_ROWS rows."""
    meta_json = json.dumps(meta, separators=(",", ":"))
    meta_bytes = meta_json.encode("utf-8")
    target = N_META_ROWS * NCOLS * 2
    padded = meta_bytes.ljust(target, b"\x00")
    arr = np.frombuffer(padded, dtype="<u2").reshape(N_META_ROWS, NCOLS)
    image[0:N_META_ROWS, :] = arr


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
    """Save a 2-panel diagnostic figure: calibration image (left), dark image (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax0, ax1 = axes

    im0 = ax0.imshow(cal_img, cmap="gray", vmin=0, vmax=16383)
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
    ax0.set_title(f"{title_line1}\n{title_line2}\n{title_line3}", fontsize=8)
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

    im1 = ax1.imshow(dark_img, cmap="gray", vmin=0, vmax=16383)
    ax1.set_title(
        f"Dark image \u2014 B={params.B_dc} ADU, "
        f"\u03c3_read={SIGMA_READ} ADU",
        fontsize=9,
    )
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    fig.suptitle(f"Z03 diagnostic \u2014 {stem}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = out_dir / f"{stem}_diagnostic.png"
    fig.savefig(out_path, dpi=150)
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
        d_mm    = _validated_prompt("Etalon gap d",                  20.106, "mm",  15.0,  25.0, 19.0,  21.5)
        f_mm    = _validated_prompt("Imaging lens focal length f",  199.12, "mm", 100.0, 300.0, 180.0, 220.0)

        print("\n\u2500\u2500 GROUP 2  REFLECTIVITY AND PSF \u2500\u2500")
        R       = _validated_prompt("Plate reflectivity R",           0.53,  "",    0.01,  0.99,  0.3,   0.85)
        sigma0  = _validated_prompt("Average PSF width sigma_0",      0.5,   "px",  0.0,   5.0,   0.0,   2.0)
        sigma1  = _validated_prompt("PSF sine variation sigma_1",     0.1,   "px", -3.0,   3.0,  -1.0,   1.0)
        sigma2  = _validated_prompt("PSF cosine variation sigma_2",  -0.05,  "px", -3.0,   3.0,  -1.0,   1.0)

        print("\n\u2500\u2500 GROUP 3  INTENSITY ENVELOPE AND BIAS \u2500\u2500")
        snr_peak = _validated_prompt("Peak SNR",                     50.0,  "",    1.0,  500.0, 10.0, 200.0)
        I1       = _validated_prompt("Linear vignetting coeff I_1",  -0.1,  "",   -0.9,   0.9,  -0.5,  0.5)
        I2       = _validated_prompt("Quadratic vignetting coeff I_2", 0.005, "",  -0.9,  0.9,  -0.5,  0.5)
        B_dc     = _validated_prompt("Bias pedestal B",              300.0, "ADU", 0.0, 5000.0, 100.0, 1000.0)

        print("\n\u2500\u2500 GROUP 4  SOURCE (not an inversion free parameter) \u2500\u2500")
        rel_638  = _validated_prompt("Relative intensity 638 nm / 640 nm", 0.8, "", 0.0, 2.0, 0.3, 1.5)

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

    # Embed S19 metadata
    embed_metadata_rows(image_cal, build_metadata_dict("Cal"))
    embed_metadata_rows(image_dark, build_metadata_dict("Dark"))

    # Output paths
    default_out = pathlib.Path(os.environ.get(
        "Z03_OUTPUT_DIR",
        str(pathlib.Path(__file__).resolve().parents[3] / "soc_synthesized_data"),
    ))
    out_dir = pathlib.Path(os.environ.get("Z03_OUTPUT_DIR", default_out))
    out_dir.mkdir(parents=True, exist_ok=True)

    ts        = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem      = f"{ts}_synth_z03"
    cal_name  = f"{ts}_cal_synth_z03.bin"
    dark_name = f"{ts}_dark_synth_z03.bin"

    path_cal  = out_dir / cal_name
    path_dark = out_dir / dark_name

    image_cal.tofile(str(path_cal))
    image_dark.tofile(str(path_dark))

    truth_path = out_dir / f"{stem}_truth.json"
    write_truth_json(params, derived, seed, path_cal, path_dark, truth_path)

    # Diagnostic figure
    try:
        diag_path = make_diagnostic_figure(image_cal, image_dark, params, out_dir, stem)
    except Exception:
        diag_path = None

    print("Wrote:")
    print("  ", path_cal)
    print("  ", path_dark)
    print("  ", truth_path)
    if diag_path:
        print("  ", diag_path)
    print("Done.")


if __name__ == "__main__":
    main()
