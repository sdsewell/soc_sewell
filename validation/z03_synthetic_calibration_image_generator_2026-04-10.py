"""
Module:      z03_synthetic_calibration_image_generator.py
Spec:        specs/z03_synthetic_calibration_image_generator_spec_2026_04_10.md
Author:      Claude Code
Generated:   2026-04-10
Project:     WindCube FPI Pipeline — NCAR/HAO
Repo:        soc_sewell

Interactive script that synthesises a two-line neon Airy fringe calibration
image in authentic WindCube .bin format (260 rows × 276 cols, uint16 LE),
embeds S19-compliant JSON metadata into the first 4 rows, and writes a
companion _truth.json sidecar and _diagnostic.png figure.

Stages:
  A — Banner + parameter prompting
  B — Derive secondary optical parameters
  C — Synthesise two-line neon Airy fringe image (noise-free)
  D — Apply Poisson + Gaussian read noise
  E — Build and embed S19-compliant JSON metadata into rows 0-3
  F — Write .bin file + _truth.json sidecar
  G — Diagnostic display (4-panel figure, saved as PNG)
"""

import datetime
import json
import math
import pathlib
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Fixed output directory (same repo as Z02 synthesized data)
# ---------------------------------------------------------------------------
OUTPUT_DIR = pathlib.Path(r"C:\Users\sewell\Documents\GitHub\soc_synthesized_data")

# ---------------------------------------------------------------------------
# Fixed default parameters (Section 6 of spec)
# ---------------------------------------------------------------------------
R            = 0.82           # plate reflectivity (WindCube etalon spec GNL4096R)
SIGMA_READ   = 50.0           # ADU — CCD97 EM gain regime read noise estimate
B_DC         = 500.0          # ADU — DC background (dark current + stray light)
CX           = 137.5          # col — geometric centre of 276-col array
CY           = 129.5          # row — geometric centre of 260-row active region
PIX_M        = 32.0e-6        # m   — CCD97 16 µm × 2×2 binning
NROWS        = 260            # total rows (including 4 metadata rows)
NCOLS        = 276            # total cols
N_ACTIVE     = 256            # active pixel rows after metadata rows
N_META_ROWS  = 4              # S19 JSON metadata occupies first 4 rows
ADU_MAX      = 16383          # 14-bit ceiling

# These are imported from src.constants at runtime (see _setup_import_path)
# Wavelengths kept here as fallback literals matching spec Table 6.
LAM_640      = 640.2248e-9    # m — Ne 640.2 nm (primary line)
LAM_638      = 638.2991e-9    # m — Ne 638.3 nm (secondary line)


# ---------------------------------------------------------------------------
# Import path helper
# ---------------------------------------------------------------------------

def _setup_import_path() -> None:
    """Add repo root to sys.path so src.* imports work from any CWD."""
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    s = str(repo_root)
    if s not in sys.path:
        sys.path.insert(0, s)


# ---------------------------------------------------------------------------
# Parameter dataclass
# ---------------------------------------------------------------------------

class SynthParams:
    """All synthesis parameters, user-entered and fixed."""

    def __init__(
        self,
        d_mm:     float,
        f_mm:     float,
        snr_peak: float,
        rel_638:  float,
    ) -> None:
        self.d_mm     = d_mm
        self.f_mm     = f_mm
        self.snr_peak = snr_peak
        self.rel_638  = rel_638
        # Fixed defaults
        self.R          = R
        self.sigma_read = SIGMA_READ
        self.B_dc       = B_DC
        self.cx         = CX
        self.cy         = CY
        self.pix_m      = PIX_M
        self.nrows      = NROWS
        self.ncols      = NCOLS
        self.lam_640    = LAM_640
        self.lam_638    = LAM_638


# ---------------------------------------------------------------------------
# Stage A helpers — validated prompting
# ---------------------------------------------------------------------------

# Validation bounds: (hard_min, hard_max, warn_min, warn_max)
_BOUNDS = {
    "d_mm":     (15.0,  25.0,  19.0,  21.5),
    "f_mm":     (100.0, 300.0, 180.0, 220.0),
    "snr_peak": (1.0,   500.0, 10.0,  200.0),
    "rel_638":  (0.0,   2.0,   0.3,   1.5),
}


def prompt_float(label: str, default: float, units: str = "") -> float:
    """
    Prompt for a float, accepting Enter for the default.

    Re-prompts on non-numeric input.
    Caller is responsible for range validation.
    """
    unit_str = f" [{units}]" if units else ""
    while True:
        raw = input(f"  {label}{unit_str} (default {default}): ").strip()
        if raw == "":
            return default
        try:
            return float(raw)
        except ValueError:
            print(f"    ⚠  Invalid input '{raw}' — please enter a number.")


def _validated_prompt(name: str, label: str, default: float, units: str = "") -> float:
    """Prompt for a float with hard-limit rejection and in-range warning."""
    hard_min, hard_max, warn_min, warn_max = _BOUNDS[name]
    while True:
        val = prompt_float(label, default, units)
        if val < hard_min or val > hard_max:
            print(f"    ✗  {val} is outside the hard limit [{hard_min}, {hard_max}]. "
                  f"Please re-enter.")
            continue
        if val < warn_min or val > warn_max:
            print(f"    ⚠  {val} is outside the recommended range "
                  f"({warn_min}–{warn_max}). Proceeding.")
        return val


def prompt_all_params() -> SynthParams:
    """
    Stage A: display banner, prompt for all four parameters, confirm.

    Returns a SynthParams with user-entered and fixed values populated.
    """
    border = "╔" + "═" * 62 + "╗"
    print()
    print(border)
    print("║  Z03  Synthetic Calibration Image Generator" + " " * 18 + "║")
    print("║  WindCube SOC — soc_sewell" + " " * 35 + "║")
    print("╚" + "═" * 62 + "╝")
    print()
    print("This script synthesises a two-line neon Airy fringe calibration")
    print("image in authentic WindCube .bin format, suitable for ingestion")
    print("by Z01, Z02, or any S01-based pipeline module.")
    print()
    print("You will be prompted for 4 parameters.  Press <Enter> to accept")
    print("the default shown in parentheses.")
    print()

    while True:
        print("─" * 62)
        print(" INSTRUMENTAL PARAMETERS")
        print("─" * 62)
        d_mm = _validated_prompt(
            "d_mm", "Etalon gap d", 20.106, "mm"
        )
        f_mm = _validated_prompt(
            "f_mm", "Imaging lens focal length f", 199.12, "mm"
        )
        print()
        print("─" * 62)
        print(" NOISE AND SOURCE PARAMETERS")
        print("─" * 62)
        snr_peak = _validated_prompt(
            "snr_peak", "Peak SNR", 50.0
        )
        rel_638 = _validated_prompt(
            "rel_638", "Relative intensity of 638 nm line vs 640 nm", 0.8
        )
        print()

        # Parameter summary
        alpha = PIX_M / (f_mm * 1e-3)
        F_coeff = 4 * R / (1 - R) ** 2
        fsr_pm = LAM_640 ** 2 / (2 * d_mm * 1e-3) * 1e12

        print("─" * 62)
        print(" PARAMETER SUMMARY")
        print("─" * 62)
        print(f"  Etalon gap            d        = {d_mm:.4f} mm")
        print(f"  Focal length          f        = {f_mm:.2f} mm")
        print(f"  Plate scale           α        = {alpha:.4e} rad/px")
        print(f"  Peak SNR              snr_peak = {snr_peak:.1f}")
        print(f"  Ne 638/640 ratio      rel_638  = {rel_638:.3f}")
        print(f"  Reflectivity          R        = {R}")
        print(f"  Finesse coeff         F        = {F_coeff:.1f}")
        print(f"  FSR at 640.2 nm               = {fsr_pm:.3f} pm")
        print(f"  Read noise            σ_read   = {SIGMA_READ} ADU")
        print(f"  DC background         B_dc     = {B_DC} ADU")
        print()

        raw = input("  Proceed with these parameters? [Y/n]: ").strip().lower()
        if raw in ("", "y", "yes"):
            break
        print()
        print("  Re-entering parameters…")
        print()

    return SynthParams(d_mm=d_mm, f_mm=f_mm, snr_peak=snr_peak, rel_638=rel_638)


# ---------------------------------------------------------------------------
# Stage B — Derive secondary parameters
# ---------------------------------------------------------------------------

def derive_secondary(params: SynthParams) -> dict:
    """
    Compute alpha, I_peak, FSR, finesse from user params + fixed defaults.

    Returns a dict of derived quantities for use in synthesis and reporting.
    """
    alpha = params.pix_m / (params.f_mm * 1e-3)
    I_peak = snr_to_ipeak(params.snr_peak, params.B_dc, params.sigma_read)
    fsr_m = params.lam_640 ** 2 / (2.0 * params.d_mm * 1e-3)
    F_coeff = 4.0 * params.R / (1.0 - params.R) ** 2
    finesse = math.pi * math.sqrt(params.R) / (1.0 - params.R)

    return {
        "alpha_rad_per_px":    alpha,
        "I_peak_adu":          I_peak,
        "FSR_m":               fsr_m,
        "FSR_pm":              fsr_m * 1e12,
        "finesse_coefficient_F": F_coeff,
        "finesse":             finesse,
    }


# ---------------------------------------------------------------------------
# Stage C — Airy function and image synthesis
# ---------------------------------------------------------------------------

def airy(theta: np.ndarray, lam: float, d_mm: float, R: float) -> np.ndarray:
    """
    Ideal Airy transmission function, vectorised over angle array.

    Parameters
    ----------
    theta : angle from optical axis (rad), any shape
    lam   : wavelength (m)
    d_mm  : etalon gap (mm)
    R     : plate reflectivity (dimensionless)

    Returns
    -------
    Transmission in [0, 1], same shape as theta.
    """
    d     = d_mm * 1e-3
    delta = (4.0 * np.pi / lam) * d * np.cos(theta)
    F     = 4.0 * R / (1.0 - R) ** 2
    return 1.0 / (1.0 + F * np.sin(delta / 2.0) ** 2)


def snr_to_ipeak(snr_peak: float, B_dc: float, sigma_read: float) -> float:
    """
    Solve for I_peak given snr_peak = I_peak / sqrt(I_peak + B_dc + sigma_read²).

    Quadratic: I_peak² - snr² · I_peak - snr² · (B_dc + sigma_read²) = 0
    Positive root: I_peak = [snr² + sqrt(snr⁴ + 4·snr²·(B_dc + sigma_read²))] / 2
    """
    snr2 = snr_peak ** 2
    noise_var = B_dc + sigma_read ** 2
    I_peak = (snr2 + math.sqrt(snr2 ** 2 + 4.0 * snr2 * noise_var)) / 2.0
    return float(I_peak)


def synthesise_image(params: SynthParams, derived: dict) -> np.ndarray:
    """
    Stage C: synthesise noise-free two-line neon composite Airy image.

    Builds pixel coordinate grids over the full (NROWS, NCOLS) array,
    evaluates T_640 and T_638 at every pixel, returns float64 array.

    The first N_META_ROWS rows will be synthesised normally here; they
    are overwritten with JSON metadata in Stage E.

    Returns
    -------
    I_float : np.ndarray, shape (NROWS, NCOLS), float64, units ADU
    """
    alpha  = derived["alpha_rad_per_px"]
    I_peak = derived["I_peak_adu"]

    # Pixel coordinate grids (col = x, row = y)
    cols = np.arange(params.ncols, dtype=np.float64)
    rows = np.arange(params.nrows, dtype=np.float64)
    X, Y = np.meshgrid(cols, rows)  # shape (nrows, ncols)

    r     = np.sqrt((X - params.cx) ** 2 + (Y - params.cy) ** 2)
    theta = np.arctan(alpha * r)

    T_640 = airy(theta, params.lam_640, params.d_mm, params.R)
    T_638 = airy(theta, params.lam_638, params.d_mm, params.R)

    I_float = I_peak * (T_640 + params.rel_638 * T_638) + params.B_dc
    return I_float


# ---------------------------------------------------------------------------
# Stage D — Noise model
# ---------------------------------------------------------------------------

def add_noise(
    I_float: np.ndarray,
    params:  SynthParams,
    rng:     np.random.Generator,
) -> np.ndarray:
    """
    Stage D: apply Poisson + Gaussian read noise, clip to [0, ADU_MAX].

    Parameters
    ----------
    I_float : noise-free image, float64 ADU
    params  : SynthParams (for sigma_read)
    rng     : seeded numpy Generator (seed recorded in _truth.json)

    Returns
    -------
    image_noisy : np.ndarray, shape (NROWS, NCOLS), uint16
    """
    signal_counts = rng.poisson(
        np.clip(I_float, 0.0, None)
    ).astype(np.float64)

    read_noise = rng.normal(0.0, params.sigma_read, size=signal_counts.shape)

    image_noisy = np.clip(signal_counts + read_noise, 0.0, ADU_MAX).astype(np.uint16)
    return image_noisy


# ---------------------------------------------------------------------------
# Stage E — S19 metadata construction and embedding
# ---------------------------------------------------------------------------

def build_s19_metadata(params: SynthParams) -> dict:
    """
    Build the S19-compliant metadata dict for a synthetic calibration image.

    All fields not explicitly set are given their S19-defined zero/null defaults.
    """
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    return {
        "image_type":     "Cal",
        "n_rows":         params.nrows,
        "n_cols":         params.ncols,
        "binning":        2,
        "shutter_status": "Open",
        "cal_lamp_1":     "C1_On",
        "cal_lamp_2":     "C2_Off",
        "cal_lamp_3":     "C3_Off",
        "date_utc":       now.strftime("%Y%m%d"),
        "time_utc":       now.strftime("%H%M%S"),
        "exposure_ms":    120000,
        "etalon_temp_1":  24.00,
    }


def embed_metadata(image: np.ndarray, meta_dict: dict) -> np.ndarray:
    """
    Stage E: serialise meta_dict as compact UTF-8 JSON, zero-pad to
    N_META_ROWS × NCOLS × 2 bytes, reinterpret as little-endian uint16,
    and overwrite image[0:N_META_ROWS, :] in-place.

    The image must be writable uint16; this function modifies it in-place
    and also returns it for convenience.

    Raises
    ------
    ValueError if the JSON is too large to fit in the metadata rows.
    """
    target_bytes = N_META_ROWS * NCOLS * 2   # 4 × 276 × 2 = 2208

    meta_json  = json.dumps(meta_dict, separators=(",", ":"))
    meta_bytes = meta_json.encode("utf-8")

    if len(meta_bytes) > target_bytes:
        raise ValueError(
            f"Metadata JSON is {len(meta_bytes)} bytes, exceeds "
            f"target {target_bytes} bytes."
        )

    meta_padded = meta_bytes.ljust(target_bytes, b"\x00")
    meta_uint16 = np.frombuffer(meta_padded, dtype="<u2").reshape(N_META_ROWS, NCOLS)
    image[0:N_META_ROWS, :] = meta_uint16
    return image


# ---------------------------------------------------------------------------
# Stage F — File writers
# ---------------------------------------------------------------------------

def write_bin(image: np.ndarray, path: pathlib.Path) -> None:
    """
    Stage F: write image as little-endian uint16 binary file.

    Parameters
    ----------
    image : shape (NROWS, NCOLS), uint16
    path  : output .bin path
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    image.astype("<u2").tofile(path)
    expected = NROWS * NCOLS * 2
    actual   = path.stat().st_size
    assert actual == expected, (
        f"File size {actual} != expected {expected} bytes."
    )


def write_truth_json(
    params:  SynthParams,
    derived: dict,
    seed:    int,
    out_bin: pathlib.Path,
    path:    pathlib.Path,
) -> None:
    """
    Stage F: write _truth.json sidecar with full synthesis provenance.
    """
    now_utc = datetime.datetime.now(tz=datetime.timezone.utc)
    truth = {
        "z03_version": "1.0",
        "timestamp_utc": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "random_seed": seed,
        "user_params": {
            "d_mm":     params.d_mm,
            "f_mm":     params.f_mm,
            "snr_peak": params.snr_peak,
            "rel_638":  params.rel_638,
        },
        "derived_params": {
            "alpha_rad_per_px":      derived["alpha_rad_per_px"],
            "I_peak_adu":            derived["I_peak_adu"],
            "FSR_m":                 derived["FSR_m"],
            "FSR_pm":                derived["FSR_pm"],
            "finesse_coefficient_F": derived["finesse_coefficient_F"],
            "finesse":               derived["finesse"],
        },
        "fixed_defaults": {
            "R":          params.R,
            "sigma_read": params.sigma_read,
            "B_dc":       params.B_dc,
            "cx":         params.cx,
            "cy":         params.cy,
            "pix_m":      params.pix_m,
            "nrows":      params.nrows,
            "ncols":      params.ncols,
            "lam_640_m":  params.lam_640,
            "lam_638_m":  params.lam_638,
        },
        "output_file": out_bin.name,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(truth, fh, indent=2)


# ---------------------------------------------------------------------------
# Stage G — Diagnostic figure
# ---------------------------------------------------------------------------

def make_diagnostic_figure(
    image_noisy:     np.ndarray,
    image_noisefree: np.ndarray,
    params:          SynthParams,
    derived:         dict,
    paths:           dict,
) -> None:
    """
    Stage G: 4-panel 2×2 diagnostic figure.

    Panel [0,0] — Full synthetic image (imshow, grey, log scale)
    Panel [0,1] — Azimuthally averaged radial profile I(r²)
    Panel [1,0] — Noise-free vs noisy horizontal slice through centre
    Panel [1,1] — Text summary of all parameters

    Saves figure as {stem}_diagnostic.png.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        f"Z03 — Synthetic Calibration Image  |  {paths['bin'].name}",
        fontsize=11,
    )

    ax_img   = axes[0, 0]
    ax_prof  = axes[0, 1]
    ax_slice = axes[1, 0]
    ax_text  = axes[1, 1]

    # ── Panel [0,0]: full image, log scale ───────────────────────────────
    display = image_noisy.astype(np.float64)
    display = np.clip(display, 1.0, None)     # avoid log(0)
    norm = mcolors.LogNorm(vmin=1.0, vmax=float(ADU_MAX))
    im = ax_img.imshow(display, cmap="gray", norm=norm, origin="upper")
    plt.colorbar(im, ax=ax_img, label="ADU (log scale)")
    ax_img.set_title(
        f"Synthetic cal image  d={params.d_mm:.3f} mm  "
        f"f={params.f_mm:.1f} mm\n"
        f"Ne lines: {params.lam_640*1e9:.4f} nm (×1.0)  "
        f"{params.lam_638*1e9:.4f} nm (×{params.rel_638:.2f})"
    )
    ax_img.set_xlabel("Column (px)")
    ax_img.set_ylabel("Row (px)")

    # ── Panel [0,1]: azimuthal radial profile I(r²) ──────────────────────
    cols_arr = np.arange(params.ncols, dtype=np.float64)
    rows_arr = np.arange(params.nrows, dtype=np.float64)
    X, Y = np.meshgrid(cols_arr, rows_arr)
    r2_all = (X - params.cx) ** 2 + (Y - params.cy) ** 2

    r2_flat   = r2_all.ravel()
    nf_flat   = image_noisefree.ravel()
    ny_flat   = image_noisy.astype(np.float64).ravel()

    r2_max  = r2_flat.max()
    n_bins  = 300
    edges   = np.linspace(0.0, r2_max, n_bins + 1)
    bin_idx = np.searchsorted(edges, r2_flat, side="right") - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    sum_nf  = np.bincount(bin_idx, weights=nf_flat,  minlength=n_bins)
    sum_ny  = np.bincount(bin_idx, weights=ny_flat,  minlength=n_bins)
    counts  = np.bincount(bin_idx,                   minlength=n_bins).astype(float)
    valid   = counts > 0
    centres = 0.5 * (edges[:-1] + edges[1:])

    ax_prof.plot(
        centres[valid], sum_nf[valid] / counts[valid],
        "b-", linewidth=1.0, label="Noise-free",
    )
    ax_prof.plot(
        centres[valid], sum_ny[valid] / counts[valid],
        "r-", linewidth=0.8, alpha=0.6, label="Noisy",
    )
    ax_prof.set_xlabel("r²  (px²)")
    ax_prof.set_ylabel("Mean intensity  (ADU)")
    ax_prof.set_title(
        f"Radial profile  |  SNR_peak = {params.snr_peak:.0f}"
        f"  |  rel_638 = {params.rel_638:.2f}"
    )
    ax_prof.legend(fontsize=8)

    # ── Panel [1,0]: horizontal slice through image centre ────────────────
    row_c = int(round(params.cy))
    nf_row = image_noisefree[row_c, :].astype(np.float64)
    ny_row = image_noisy[row_c, :].astype(np.float64)
    col_px = np.arange(params.ncols)

    ax_slice.plot(col_px, nf_row, "b-",  linewidth=1.2, label="Noise-free")
    ax_slice.plot(col_px, ny_row, "r-",  linewidth=0.8, alpha=0.7, label="Noisy")
    ax_slice.set_xlabel("Column (px)")
    ax_slice.set_ylabel("Intensity  (ADU)")
    ax_slice.set_title(f"Horizontal slice at row {row_c}  (image centre)")
    ax_slice.legend(fontsize=8)

    # ── Panel [1,1]: text summary ─────────────────────────────────────────
    ax_text.axis("off")
    summary = (
        f"d        = {params.d_mm:.4f} mm\n"
        f"f        = {params.f_mm:.2f} mm\n"
        f"α        = {derived['alpha_rad_per_px']:.4e} rad/px\n"
        f"λ_640    = {params.lam_640*1e9:.4f} nm  (rel = 1.000)\n"
        f"λ_638    = {params.lam_638*1e9:.4f} nm  (rel = {params.rel_638:.3f})\n"
        f"snr_peak = {params.snr_peak:.1f}\n"
        f"I_peak   = {derived['I_peak_adu']:.1f} ADU\n"
        f"R        = {params.R}\n"
        f"F_coeff  = {derived['finesse_coefficient_F']:.1f}\n"
        f"Finesse  = {derived['finesse']:.1f}\n"
        f"FSR      = {derived['FSR_pm']:.3f} pm\n"
        f"σ_read   = {params.sigma_read} ADU\n"
        f"B_dc     = {params.B_dc} ADU\n"
        f"\n"
        f"Output:\n"
        f"  {paths['bin'].name}\n"
        f"  {paths['truth'].name}"
    )
    ax_text.text(
        0.05, 0.97,
        summary,
        transform=ax_text.transAxes,
        ha="left", va="top",
        fontsize=9,
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9),
    )
    ax_text.set_title("Synthesis parameters")

    plt.tight_layout()
    fig.savefig(paths["diag"], dpi=120, bbox_inches="tight")
    plt.show(block=False)
    print(f"  Diagnostic figure saved: {paths['diag'].name}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Interactive entry point for Z03.

    Flow:
      A — Banner + prompt 4 parameters + confirm
      B — Derive secondary parameters, print derived table
      C — Synthesise noise-free image over full (260, 276) grid
      D — Apply Poisson + Gaussian read noise
      E — Build S19 JSON metadata, embed into rows 0-3
      F — Write .bin + _truth.json
      G — Diagnostic figure (saved as PNG, displayed)
    """
    _setup_import_path()

    # Try to import canonical wavelengths from src.constants;
    # fall back to module-level literals if unavailable.
    try:
        from src.constants import NE_WAVELENGTH_1_M, NE_WAVELENGTH_2_M
        global LAM_640, LAM_638
        LAM_640 = NE_WAVELENGTH_1_M
        LAM_638 = NE_WAVELENGTH_2_M
    except ImportError:
        pass   # module-level literals remain

    # ── Stage A ──────────────────────────────────────────────────────────
    params = prompt_all_params()

    # ── Stage B ──────────────────────────────────────────────────────────
    derived = derive_secondary(params)
    print()
    print("── Derived parameters " + "─" * 40)
    print(f"  Plate scale        α = {derived['alpha_rad_per_px']:.4e} rad/px")
    print(f"  Peak signal    I_peak = {derived['I_peak_adu']:.1f} ADU")
    print(f"  Free spec. range FSR = {derived['FSR_pm']:.3f} pm")
    print(f"  Finesse coeff      F = {derived['finesse_coefficient_F']:.1f}")
    print(f"  Instrument finesse   = {derived['finesse']:.1f}")
    print()

    # ── Stage C ──────────────────────────────────────────────────────────
    print("  Synthesising noise-free Airy fringe image …")
    I_float = synthesise_image(params, derived)

    # ── Stage D ──────────────────────────────────────────────────────────
    ss       = np.random.SeedSequence()
    seed_int = int(ss.entropy) & 0xFFFF_FFFF   # store as 32-bit int in JSON
    rng      = np.random.default_rng(ss)

    print("  Applying Poisson + read noise …")
    image_noisy = add_noise(I_float, params, rng)
    # Keep noisefree as uint16 for panel comparison (rows 0-3 will differ after Stage E)
    image_noisefree = np.clip(I_float, 0.0, ADU_MAX).astype(np.uint16)

    # ── Stage E ──────────────────────────────────────────────────────────
    print("  Building and embedding S19 metadata into rows 0-3 …")
    meta_dict = build_s19_metadata(params)
    embed_metadata(image_noisy, meta_dict)

    # ── Stage F ──────────────────────────────────────────────────────────
    now_utc = datetime.datetime.now(tz=datetime.timezone.utc)
    stem    = now_utc.strftime("%Y%m%dT%H%M%SZ") + "_cal_synth_z03"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "bin":   OUTPUT_DIR / f"{stem}.bin",
        "truth": OUTPUT_DIR / f"{stem}_truth.json",
        "diag":  OUTPUT_DIR / f"{stem}_diagnostic.png",
    }

    print(f"  Writing .bin file: {paths['bin'].name} …")
    write_bin(image_noisy, paths["bin"])
    print(f"  Writing truth JSON: {paths['truth'].name} …")
    write_truth_json(params, derived, seed_int, paths["bin"], paths["truth"])

    print()
    print("╔" + "═" * 62 + "╗")
    print("║  Output files written successfully" + " " * 27 + "║")
    print("╚" + "═" * 62 + "╝")
    print(f"  .bin   : {paths['bin']}")
    print(f"  truth  : {paths['truth']}")
    print()

    # ── Stage G ──────────────────────────────────────────────────────────
    print("  Generating diagnostic figure …")
    make_diagnostic_figure(image_noisy, image_noisefree, params, derived, paths)

    input("\nDone. Press Enter to exit.")


if __name__ == "__main__":
    main()
