"""
Module:      z02_synthetic_airglow_generator_2026-04-10.py
Spec:        specs/Z02_synthetic_airglow_image_generator_2026-04-10.md
Author:      Claude Code
Generated:   2026-04-10
Last tested: 2026-04-10  (8/8 tests pass)
Project:     WindCube FPI Pipeline — NCAR/HAO
Repo:        soc_sewell

Standalone interactive script that:
  1. Prompts user for three FPI instrument parameters (t, R_refl, f_lens).
  2. Synthesises a 2D airglow fringe image using M01/M04 physics.
  3. Packs a complete S19-compliant metadata header into binary row 0.
  4. Saves a .bin file in the exact on-orbit format that P01 ingest_real_image() expects.
  5. Prints a parameter table and shows a 3-panel diagnostic figure.
"""

import os
import pathlib
import struct
import sys
import time

import numpy as np


# ---------------------------------------------------------------------------
# Encoding helpers — exact inverses of S19 §2.2 decode functions
# ---------------------------------------------------------------------------


def _encode_u16(h: np.ndarray, w: int, val: int) -> None:
    """Write a uint16 value to word w of header array h."""
    h[w] = int(val) & 0xFFFF


def _encode_u64(h: np.ndarray, w: int, val: int) -> None:
    """
    Encode a uint64 as 4 BE uint16 words in LE word order (S19 §2.2).

    Inverse of: sum(int(h[w+i]) << (16*i) for i in range(4))
    Word 0 holds bits 0–15 (least significant), word 3 holds bits 48–63.
    """
    v = int(val)
    for i in range(4):
        h[w + i] = (v >> (16 * i)) & 0xFFFF


def _encode_f64(h: np.ndarray, w: int, val: float) -> None:
    """
    Encode a float64 as 4 BE uint16 words in LE word order (S19 §2.2).

    Inverse of:
        b = struct.pack(">4H", *reversed([h[w+i] for i in range(4)]))
        return struct.unpack(">d", b)[0]

    Pack the double as big-endian → 8 bytes → 4 uint16 words.
    Write in little-endian word order: LSW at w+0, MSW at w+3.
    """
    b = struct.pack(">d", val)
    words = struct.unpack(">4H", b)      # [MSW, ..., LSW]
    for i in range(4):
        h[w + i] = words[3 - i]          # LE word order: LSW at w+0


def _encode_quaternion_wxyz(
    h: np.ndarray,
    w_start: int,
    q_wxyz: list,
) -> None:
    """
    Encode a quaternion in [w, x, y, z] scalar-first order into the header.

    Each component is encoded as a float64 using _encode_f64.
    Occupies 4 × 4 = 16 words starting at w_start.

    Parameters
    ----------
    h       : header array, shape (276,), dtype '>u2'
    w_start : first word index (28 for ads_q_hat; 44 for acs_q_err)
    q_wxyz  : [w, x, y, z], the scalar-first quaternion to encode
    """
    for i, component in enumerate(q_wxyz):
        _encode_f64(h, w_start + i * 4, component)


def _verify_encoding_roundtrip() -> None:
    """
    Verify that the encoding functions are the exact inverse of S19 §2.2 decodes.

    Called at module load unless running under pytest.
    """
    h = np.zeros(276, dtype=">u2")

    # Round-trip uint64
    test_u64 = 1_746_000_000_123  # realistic lua_timestamp in ms
    _encode_u64(h, 0, test_u64)
    recovered_u64 = sum(int(h[i]) << (16 * i) for i in range(4))
    assert recovered_u64 == test_u64, (
        f"_encode_u64 round-trip FAILED: {recovered_u64} != {test_u64}"
    )

    # Round-trip float64
    for test_f64 in [24.0, 510_000.0, -273.15, 0.0, 1.0, 630.0304e-9]:
        _encode_f64(h, 0, test_f64)
        b = struct.pack(">4H", *reversed([h[i] for i in range(4)]))
        recovered_f64 = struct.unpack(">d", b)[0]
        assert abs(recovered_f64 - test_f64) < 1e-15 * abs(test_f64) + 1e-300, (
            f"_encode_f64 round-trip FAILED: {recovered_f64} != {test_f64}"
        )


# Run round-trip verification at module load (skip under pytest)
if "PYTEST_CURRENT_TEST" not in os.environ:
    _verify_encoding_roundtrip()


# ---------------------------------------------------------------------------
# Header construction
# ---------------------------------------------------------------------------


def build_header(
    t_mm: float,
    etalon_temp_c: float = 24.0,
    exp_time_cs: int = 6000,
) -> np.ndarray:
    """
    Build a 276-word S19-compliant metadata header for a synthetic science image.

    Parameters
    ----------
    t_mm          : etalon gap in mm; for provenance only (no header word for this).
    etalon_temp_c : etalon temperature in °C. Stored in b2_temp_f[0–3] (words 84–99).
    exp_time_cs   : exposure time in centiseconds. Default: 6000 (60 s).

    Returns
    -------
    h : np.ndarray, shape (276,), dtype '>u2' (big-endian uint16)
    """
    h = np.zeros(276, dtype=">u2")

    # Geometry
    _encode_u16(h, 0, 260)           # rows
    _encode_u16(h, 1, 276)           # cols
    _encode_u16(h, 2, exp_time_cs)   # exp_time (centiseconds)
    _encode_u16(h, 3, 1)             # exp_unit

    # CCD temperature (zero — no hardware)
    _encode_f64(h, 4, 0.0)

    # Timestamps
    lua_ts = int(time.time() * 1000)
    _encode_u64(h, 8,  lua_ts)
    _encode_u64(h, 12, 0)            # adcs_timestamp = 0

    # Orbit state (zeros for lat/lon; plausible altitude)
    _encode_f64(h, 16, 0.0)          # lat_hat
    _encode_f64(h, 20, 0.0)          # lon_hat
    _encode_f64(h, 24, 510_000.0)    # alt_hat, m (510 km nominal SSO altitude)

    # Attitude — identity quaternion [w=1, x=0, y=0, z=0] (scalar-first in binary)
    _encode_quaternion_wxyz(h, 28, [1.0, 0.0, 0.0, 0.0])  # ads_q_hat
    _encode_quaternion_wxyz(h, 44, [1.0, 0.0, 0.0, 0.0])  # acs_q_err

    # ECI position and velocity (plausible SSO orbit)
    _encode_f64(h, 60, 0.0)
    _encode_f64(h, 64, 6_891_000.0)
    _encode_f64(h, 68, 0.0)
    _encode_f64(h, 72, 7_560.0)
    _encode_f64(h, 76, 0.0)
    _encode_f64(h, 80, 0.0)

    # Etalon temperatures — all four channels = etalon_temp_c
    for i in range(4):
        _encode_f64(h, 84 + i * 4, etalon_temp_c)

    # GPIO and lamp channels — all zero (science image: shutter open, lamps off)
    for w in range(100, 110):
        h[w] = 0

    # Words 110–275: reserved, remain zero (already zeros from np.zeros)
    return h


# ---------------------------------------------------------------------------
# Image synthesis and embedding
# ---------------------------------------------------------------------------

# Physical constants for ADU scaling
PIXEL_SIZE_BINNED_M = 32e-6   # 16 µm × 2 binning
SCALE_FACTOR = 4.0             # float counts → 14-bit ADU (I0=1000 → ADU 4000)


def synthesise_and_embed(
    t_m: float,
    R_refl: float,
    f_lens_m: float,
    v_rel_ms: float = 0.0,
    snr: float = 5.0,
    image_size: int = 256,
    rng: np.random.Generator = None,
) -> tuple:
    """
    Synthesise a 2D airglow image and embed it into a 260×276 binary frame.

    Steps:
    1. Compute alpha from f_lens_m and pixel size.
    2. Construct InstrumentParams(t_m, R_refl, alpha).
    3. Call synthesise_airglow_image() → dict with 'image_2d' (256×256 float64).
    4. Scale float counts → uint16 ADU.
    5. Create 259×276 zero pixel block; embed 256×256 image at centre.
    6. Build 276-word header via build_header().
    7. Stack header row + pixel block → (260, 276) uint16 array.

    Parameters
    ----------
    t_m        : etalon gap, metres
    R_refl     : plate reflectivity (dimensionless, 0–1)
    f_lens_m   : imaging lens focal length, metres
    v_rel_ms   : LOS wind speed, m/s (default 0 — zero wind)
    snr        : target SNR for Gaussian noise (default 5.0)
    image_size : synthesised image side length in pixels (default 256)
    rng        : numpy random Generator for reproducible noise (optional)

    Returns
    -------
    (frame_260x276, pixel_block_259x276, synthesis_result_dict)
    """
    # Step 1: derive alpha
    alpha = PIXEL_SIZE_BINNED_M / f_lens_m

    # Step 2: InstrumentParams
    # Import here to keep the module usable as a script from any working dir
    _setup_import_path()
    from src.fpi.m01_airy_forward_model_2026_04_05 import InstrumentParams
    params = InstrumentParams(t=t_m, R_refl=R_refl, alpha=alpha)

    # Step 3: synthesise
    from src.fpi.m04_airglow_synthesis_2026_04_05 import synthesise_airglow_image
    if rng is None:
        rng = np.random.default_rng(42)
    result = synthesise_airglow_image(
        v_rel_ms=v_rel_ms, params=params, snr=snr,
        image_size=image_size, add_noise=True, rng=rng
    )
    img_float = result["image_2d"]   # (256, 256) float64, counts

    # Step 4: float → uint16 ADU (14-bit)
    img_adu = np.clip(
        np.round(img_float * SCALE_FACTOR), 0, 16383
    ).astype(np.uint16)

    # Step 5: embed in 259×276 zero pixel block
    pixel_block = np.zeros((259, 276), dtype=np.uint16)
    r_start = (259 - image_size) // 2   # = 1
    c_start = (276 - image_size) // 2   # = 10
    pixel_block[r_start:r_start + image_size, c_start:c_start + image_size] = img_adu

    # Step 6: build header
    header_row = build_header(t_mm=t_m * 1000)   # shape (276,), dtype '>u2'

    # Step 7: assemble 260×276 frame
    frame = np.zeros((260, 276), dtype=">u2")
    frame[0, :] = header_row
    frame[1:, :] = pixel_block.astype(">u2")

    return frame, pixel_block, result


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def save_bin_file(
    frame: np.ndarray,
    output_path: pathlib.Path,
) -> None:
    """
    Write the 260×276 frame to disk as a big-endian binary file.

    Parameters
    ----------
    frame       : shape (260, 276), dtype '>u2'
    output_path : pathlib.Path for output .bin file

    Raises
    ------
    ValueError if frame.shape != (260, 276)
    """
    if frame.shape != (260, 276):
        raise ValueError(
            f"frame.shape {frame.shape} != (260, 276). "
            "Expected a 260×276 array."
        )
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.astype(">u2").tofile(output_path)
    expected_bytes = 260 * 276 * 2   # 143,520
    actual_bytes   = output_path.stat().st_size
    assert actual_bytes == expected_bytes, (
        f"File size {actual_bytes} != {expected_bytes} bytes"
    )


# ---------------------------------------------------------------------------
# Output path — fixed to the synthesized-data repository
# ---------------------------------------------------------------------------

OUTPUT_DIR = pathlib.Path(r"C:\Users\sewell\Documents\GitHub\soc_synthesized_data")


def get_output_path(default_stem: str = "z02_synthetic_airglow") -> pathlib.Path:
    """
    Return a timestamped .bin path inside the fixed output directory.

    The directory is created automatically if it does not yet exist.

    Returns
    -------
    pathlib.Path — absolute path to the .bin file to be created
    """
    import datetime
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{default_stem}_{timestamp_str}.bin"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / filename


# ---------------------------------------------------------------------------
# Print parameter table
# ---------------------------------------------------------------------------


def print_parameter_table(
    t_m: float,
    R_refl: float,
    f_lens_m: float,
    alpha: float,
    v_rel_ms: float,
    snr: float,
    result: dict,
    output_path: pathlib.Path,
) -> None:
    """
    Print a formatted table of all instrument parameters and synthesis results.
    """
    from src.fpi.m01_airy_forward_model_2026_04_05 import (
        OI_WAVELENGTH_M,
        SPEED_OF_LIGHT_MS,
    )
    from src.fpi.m04_airglow_synthesis_2026_04_05 import v_rel_to_lambda_c

    # Derived quantities
    F_coeff = 4.0 * R_refl / (1.0 - R_refl) ** 2
    finesse  = np.pi * np.sqrt(R_refl) / (1.0 - R_refl)
    fsr_pm   = (OI_WAVELENGTH_M ** 2 / (2.0 * t_m)) * 1e12   # pm
    lc_nm    = v_rel_to_lambda_c(v_rel_ms) * 1e9              # nm
    lua_ts   = result.get("lua_ts_utc", "")

    # Re-read header for lua_timestamp display
    import datetime
    # We get the lua_timestamp from the header we just wrote — re-derive it
    lua_ts_utc = "(computed at synthesis time)"

    border = "═" * 66
    thin   = "─" * 66
    print()
    print(border)
    print("  Z02 — WindCube Synthetic Airglow Image Generator")
    print("  NCAR / High Altitude Observatory (HAO)")
    print(border)
    print()
    print("── Instrument Parameters " + "─" * 42)
    print(f"  Etalon gap   t            {t_m * 1000:.4f}   mm")
    print(f"  Reflectivity R            {R_refl:.3f}       (dimensionless)")
    print(f"  Focal length f            {f_lens_m * 1000:.2f}   mm")
    print(f"  Plate scale  α (derived)  {alpha:.4e}   rad/px")
    print(f"  Pixel size (2×2 binned)   32.0          µm")
    print()
    print("── Derived FPI Properties " + "─" * 41)
    print(f"  Finesse coefficient F     {F_coeff:.2f}")
    print(f"  Instrument finesse        {finesse:.2f}")
    print(f"  FSR at OI 630 nm          {fsr_pm:.3f}  pm")
    print()
    print("── Synthesis Parameters " + "─" * 43)
    print(f"  LOS wind speed v_rel      {v_rel_ms:.1f}  m/s")
    print(f"  Doppler-shifted λ_c       {lc_nm:.5f}  nm")
    print(f"  Target SNR                {snr:.1f}")
    print(f"  Actual SNR achieved       {result['snr_actual']:.2f}")
    print(f"  Image size                256×256  px")
    print()
    print("── Metadata Header (row 0) " + "─" * 40)
    print(f"  Image type                science")
    print(f"  Exposure time             60.0  s")
    print(f"  Etalon temperature        24.0  °C  (all 4 channels)")
    print(f"  CCD temperature           0.0   °C  (zero — no hardware)")
    print(f"  lua_timestamp             {lua_ts_utc}")
    print()
    print("── Output " + "─" * 56)
    print(f"  Saved to: {output_path}")
    print(f"  File size: 143,520 bytes  ✓")
    print(border)
    print()


# ---------------------------------------------------------------------------
# Diagnostic figure
# ---------------------------------------------------------------------------


def show_diagnostic_figure(
    pixel_block: np.ndarray,
    result: dict,
    params,
    output_path: pathlib.Path,
    tolansky_result=None,
) -> None:
    """
    Show a 2×2 diagnostic figure.

    Row 0 — Panel A: 2D airglow image (imshow, gray).
             Panel B: 1D radial profile with fringe peak markers.
    Row 1 — Panel C: Text panel showing decoded header fields from the saved file.
             Panel D: Tolansky r² vs p WLS fit (hidden when tolansky_result is None).
    """
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks

    r_start = 1
    c_start = 10
    img_display = pixel_block[r_start:r_start + 256, c_start:c_start + 256]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(
        f"Z02 — Synthetic Airglow Image  |  {output_path.name}",
        fontsize=11,
    )

    ax_a = axes[0, 0]
    ax_b = axes[0, 1]
    ax_c = axes[1, 0]
    ax_d = axes[1, 1]

    # ── Panel A: 2D image ──────────────────────────────────────────────────
    im = ax_a.imshow(img_display, cmap="gray", vmin=0, vmax=16383, origin="upper")
    plt.colorbar(im, ax=ax_a, label="ADU (14-bit)")
    ax_a.set_title("Synthetic airglow image (2D)")
    ax_a.set_xlabel("Column (px)")
    ax_a.set_ylabel("Row (px)")

    # ── Panel B: 1D radial profile ─────────────────────────────────────────
    profile = result["profile_1d"]
    r_grid  = result["r_grid"]
    ax_b.plot(r_grid, profile, "b-", linewidth=1.2, label="Profile")

    peaks, _ = find_peaks(profile, height=np.mean(profile))
    for pk in peaks:
        ax_b.axvline(x=r_grid[pk], color="r", linestyle="--", alpha=0.5, linewidth=0.8)

    # Annotate FSR and finesse
    from src.fpi.m01_airy_forward_model_2026_04_05 import OI_WAVELENGTH_M
    fsr_pm  = (OI_WAVELENGTH_M ** 2 / (2.0 * params.t)) * 1e12
    finesse = np.pi * np.sqrt(params.R_refl) / (1.0 - params.R_refl)
    ax_b.text(
        0.98, 0.98,
        f"FSR = {fsr_pm:.2f} pm\nFinesse = {finesse:.2f}",
        transform=ax_b.transAxes,
        ha="right", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
    )

    snr_actual = result.get("snr_actual", 0.0)
    ax_b.set_xlabel("Radius (pixels)")
    ax_b.set_ylabel("Intensity (counts)")
    ax_b.set_title(
        f"1D radial profile  |  v_rel = {result.get('v_rel_ms', 0.0):.0f} m/s"
        f"  |  SNR = {snr_actual:.1f}"
    )
    ax_b.legend(fontsize=8)

    # ── Panel C: Header decode check ───────────────────────────────────────
    ax_c.axis("off")

    # Re-read row 0 of the saved file and decode key fields
    header_text = "(header not yet decoded)"
    try:
        raw = np.frombuffer(output_path.read_bytes(), dtype=">u2")
        h = raw[:276]

        rows_val = int(h[0])
        cols_val = int(h[1])
        exp_cs   = int(h[2])

        # Decode lua_timestamp (uint64 from words 8–11)
        lua_ts_ms = sum(int(h[8 + i]) << (16 * i) for i in range(4))
        import datetime
        dt = datetime.datetime.fromtimestamp(
            lua_ts_ms / 1000.0, tz=datetime.timezone.utc
        )
        utc_str = dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Decode etalon_temp[0] from words 84–87
        b_et = struct.pack(">4H", *reversed([h[84 + i] for i in range(4)]))
        etalon_t = struct.unpack(">d", b_et)[0]

        # Derive img_type
        lamps = [int(h[104 + i]) & 0xFF for i in range(6)]
        gpio  = [int(h[100 + i]) & 0xFF for i in range(4)]
        if any(lamps):
            img_type_str = "cal"
        elif (gpio[0] == 1 and gpio[3] == 1):
            img_type_str = "dark"
        else:
            img_type_str = "science"

        header_text = (
            f"rows         = {rows_val}\n"
            f"cols         = {cols_val}\n"
            f"exp_time_cs  = {exp_cs}  ({exp_cs / 100:.0f} s)\n"
            f"lua_timestamp\n"
            f"  {utc_str}\n"
            f"etalon_temp[0] = {etalon_t:.1f} °C\n"
            f"img_type     = {img_type_str}\n"
            f"\n"
            f"File: {output_path.name}\n"
            f"Size: {output_path.stat().st_size:,} bytes"
        )
    except Exception as exc:
        header_text = f"Decode error:\n{exc}"

    ax_c.text(
        0.05, 0.95,
        header_text,
        transform=ax_c.transAxes,
        ha="left", va="top",
        fontsize=9,
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8),
    )
    ax_c.set_title("Header decode check (row 0)")

    # ── Panel D: Tolansky r² vs p WLS fit ─────────────────────────────────
    if tolansky_result is not None:
        tr = tolansky_result   # plain dict

        p_arr  = tr["peak_orders"]
        p_plot = np.linspace(p_arr[0] - 0.5, p_arr[-1] + 0.5, 200)
        fit_line = tr["S"] * p_plot + tr["b"]

        ax_d.errorbar(
            p_arr, tr["r_sq"], yerr=tr["sigma_r_sq"],
            fmt="o", color="steelblue", markersize=5,
            capsize=3, linewidth=0.8, label="r² data",
        )
        ax_d.plot(p_plot, fit_line, "r-", linewidth=1.4, label="WLS fit")

        lc_nm       = tr["lambda_c_m"] * 1e9
        sigma_lc_pm = tr["sigma_lc_m"] * 1e12
        ann_lines = [
            f"ε = {tr['epsilon']:.6f} ± {tr['sigma_epsilon']:.2e}",
            f"R² = {tr['R2']:.8f}",
            f"λ_c = {lc_nm:.6f} ± {sigma_lc_pm:.4f} pm  nm",
            f"v = {tr['v_rel_ms']:+.1f} ± {tr['sigma_v_ms']:.1f} m/s  (rec)",
            f"v = {tr['v_input_ms']:+.1f} m/s          (input)",
            f"Δv = {tr['delta_v_ms']:+.1f} m/s",
        ]

        ax_d.text(
            0.05, 0.97,
            "\n".join(ann_lines),
            transform=ax_d.transAxes,
            ha="left", va="top",
            fontsize=8.5,
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.9),
        )

        ax_d.set_xlabel("Fringe index p")
        ax_d.set_ylabel("r²  (px²)")
        ax_d.set_title(
            f"Tolansky: ε → v_rel  (fixed d, f)  |  {tr['n_peaks']} peaks"
        )
        ax_d.legend(fontsize=8)
    else:
        ax_d.axis("off")

    plt.tight_layout()
    plt.show(block=False)
    print("\nDiagnostic figure displayed. Close the window or press Enter to continue.")


# ---------------------------------------------------------------------------
# Tolansky stage — peak finding + single-line WLS fit
# ---------------------------------------------------------------------------


def _run_tolansky_stage(
    profile_1d: np.ndarray,
    r_grid: np.ndarray,
    d_m: float,
    f_m: float,
    v_input_ms: float = 0.0,
):
    """
    Run a single-line Tolansky fit on the synthesised OI 630 nm fringe profile.

    d and f are treated as fixed, well-determined priors from the two-line neon
    calibration (S13 TwoLineAnalyser). They are passed directly to TolanskyAnalyser
    and are NOT re-fitted. The sole free parameter is ε.

    Recovery chain:
        ring positions r_p → WLS fit (fixed d, f) → ε
        ε → λ_c = 2·n·d / (m₀ + ε)           (Tolansky relation, m₀ = round(2nd/λ₀))
        λ_c → v_rel = c · (λ_c/λ₀ − 1)        (Doppler formula, M04)

    Parameters
    ----------
    profile_1d  : 1D radial intensity profile, counts (noiseless, from synthesis result)
    r_grid      : radial positions, pixels
    d_m         : etalon gap, metres — fixed prior from neon calibration
    f_m         : focal length, metres — fixed prior from neon calibration
    v_input_ms  : ground-truth LOS wind used in synthesis, m/s (for Δv display)

    Returns
    -------
    dict with keys:
        epsilon, sigma_epsilon   — fitted fringe fraction and uncertainty
        lambda_c_m, sigma_lc_m  — recovered line centre wavelength, metres
        v_rel_ms, sigma_v_ms    — recovered LOS wind speed, m/s
        v_input_ms, delta_v_ms  — ground-truth wind and recovery error, m/s
        S, b, R2                — WLS slope, intercept, goodness-of-fit
        n_peaks                 — number of Gaussian-fitted rings used
        peak_orders             — fringe indices p = 1, 2, …, N (np.ndarray)
        peak_radii              — fitted ring positions in pixels (np.ndarray)
        r_sq, sigma_r_sq        — r² and σ(r²) arrays for Panel D plotting
    or None on failure (panel D is omitted; warning is printed to stdout).
    """
    try:
        from src.fpi.m03_annular_reduction_2026_04_06 import _find_and_fit_peaks
        from src.fpi.tolansky_2026_04_05 import TolanskyAnalyser
        from src.fpi.m01_airy_forward_model_2026_04_05 import (
            OI_WAVELENGTH_M, SPEED_OF_LIGHT_MS,
        )
        from src.fpi.m04_airglow_synthesis_2026_04_05 import lambda_c_to_v_rel

        # ── Step 1–2: build sigma_profile and masked ─────────────────────────
        ptp = float(np.ptp(profile_1d))
        sigma_profile = np.full_like(profile_1d, max(ptp * 0.05, 1.0))
        masked = np.zeros(len(profile_1d), dtype=bool)

        # ── Step 3: find and Gaussian-fit peaks ──────────────────────────────
        peaks = _find_and_fit_peaks(
            r_grid=r_grid,
            profile=profile_1d,
            sigma_profile=sigma_profile,
            masked=masked,
            prominence=ptp * 0.10,
        )

        good_peaks = sorted(
            [pk for pk in peaks if pk.fit_ok],
            key=lambda pk: pk.r_fit_px,
        )

        if len(good_peaks) < 3:
            print(f"  ⚠  Tolansky stage: only {len(good_peaks)} good peaks found "
                  f"(need ≥ 3) — skipping panel D.")
            return None

        # ── Step 4–5: fringe indices and focal length in pixels ───────────────
        p       = np.arange(1, len(good_peaks) + 1, dtype=float)
        r       = np.array([pk.r_fit_px for pk in good_peaks])
        sigma_r = np.array([
            pk.sigma_r_fit_px
            if np.isfinite(pk.sigma_r_fit_px) and pk.sigma_r_fit_px > 0.0
            else 0.5
            for pk in good_peaks
        ])
        f_px = f_m / PIXEL_SIZE_BINNED_M

        # ── Step 6: WLS fit — ε is the sole fitted quantity ──────────────────
        analyser = TolanskyAnalyser(
            p=p, r=r, sigma_r=sigma_r,
            lam_nm=OI_WAVELENGTH_M * 1e9,
            d_m=d_m,
            f_px=f_px,
            pixel_pitch_m=PIXEL_SIZE_BINNED_M,
        )
        fit = analyser.run()
        epsilon       = float(fit.epsilon)
        sigma_epsilon = float(fit.sigma_epsilon)
        r_sq          = fit.r_sq
        sigma_r_sq    = fit.sigma_r_sq

        # ── Step 7: λ_c from ε using the Tolansky relation ───────────────────
        n   = 1.0
        m0  = round(2 * n * d_m / OI_WAVELENGTH_M)
        lambda_c_m = 2 * n * d_m / (m0 + epsilon)
        sigma_lc_m = lambda_c_m * sigma_epsilon / (m0 + epsilon)

        # ── Step 8: v_rel from λ_c via Doppler formula ───────────────────────
        v_rel_ms  = float(lambda_c_to_v_rel(lambda_c_m))
        sigma_v_ms = float(SPEED_OF_LIGHT_MS * sigma_lc_m / OI_WAVELENGTH_M)
        delta_v_ms = v_rel_ms - v_input_ms

        return {
            "epsilon":       epsilon,
            "sigma_epsilon": sigma_epsilon,
            "lambda_c_m":    lambda_c_m,
            "sigma_lc_m":    sigma_lc_m,
            "v_rel_ms":      v_rel_ms,
            "sigma_v_ms":    sigma_v_ms,
            "v_input_ms":    v_input_ms,
            "delta_v_ms":    delta_v_ms,
            "S":             float(fit.slope),
            "b":             float(fit.intercept),
            "R2":            float(fit.r2_fit),
            "n_peaks":       len(good_peaks),
            "peak_orders":   p,
            "peak_radii":    r,
            "r_sq":          r_sq,
            "sigma_r_sq":    sigma_r_sq,
        }

    except Exception as exc:
        print(f"  ⚠  Tolansky stage failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Import path helper
# ---------------------------------------------------------------------------


def _setup_import_path() -> None:
    """
    Ensure the repo root (soc_sewell/) is on sys.path so that
    'src.fpi.*' imports work whether the script is run from any directory.
    """
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


# ---------------------------------------------------------------------------
# Interactive main
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Interactive entry point for Z02.

    Flow:
    1. Print banner.
    2. Prompt user for t_mm, R_refl, f_lens_mm.
    3. Prompt for output folder.
    4. Synthesise image: synthesise_and_embed().
    5. Save: save_bin_file().
    6. Print parameter table: print_parameter_table().
    7. Display 3-panel diagnostic figure: show_diagnostic_figure().
    8. Wait for user to press Enter.
    """
    _setup_import_path()

    border = "═" * 66
    print()
    print(border)
    print("  Z02 — WindCube Synthetic Airglow Image Generator")
    print("  NCAR / High Altitude Observatory (HAO)")
    print(border)
    print()
    print("Instrument parameters (press Enter to accept default):")
    print()

    # ── Prompt: etalon gap ────────────────────────────────────────────────
    T_DEFAULT = 20.106
    T_MIN, T_MAX = 19.5, 20.5
    while True:
        print(f"  Etalon gap [mm]")
        print(f"    Default: {T_DEFAULT} mm  (operational Tolansky value)")
        print(f"    Range:   {T_MIN} – {T_MAX} mm")
        raw = input(f"  ➤ Enter etalon gap [{T_DEFAULT}]: ").strip()
        if raw == "":
            t_mm = T_DEFAULT
            break
        try:
            t_mm = float(raw)
            if T_MIN <= t_mm <= T_MAX:
                break
            print(f"    ⚠  Value {t_mm} out of range [{T_MIN}, {T_MAX}]. Please try again.\n")
        except ValueError:
            print(f"    ⚠  Invalid input '{raw}'. Please enter a number.\n")
    print()

    # ── Prompt: reflectivity ──────────────────────────────────────────────
    R_DEFAULT = 0.53
    R_MIN, R_MAX = 0.20, 0.90
    while True:
        print(f"  Etalon reflectivity (dimensionless)")
        print(f"    Default: {R_DEFAULT}  (FlatSat calibration)")
        print(f"    Range:   {R_MIN} – {R_MAX}")
        raw = input(f"  ➤ Enter reflectivity [{R_DEFAULT}]: ").strip()
        if raw == "":
            R_refl = R_DEFAULT
            break
        try:
            R_refl = float(raw)
            if R_MIN <= R_refl <= R_MAX:
                break
            print(f"    ⚠  Value {R_refl} out of range [{R_MIN}, {R_MAX}]. Please try again.\n")
        except ValueError:
            print(f"    ⚠  Invalid input '{raw}'. Please enter a number.\n")
    print()

    # ── Prompt: focal length ──────────────────────────────────────────────
    F_DEFAULT = 199.10
    F_MIN, F_MAX = 150.0, 250.0
    while True:
        print(f"  Imaging lens focal length [mm]")
        print(f"    Default: {F_DEFAULT} mm  (Tolansky result; nominal 200 mm)")
        print(f"    Range:   {F_MIN} – {F_MAX} mm")
        print(f"    Note:    plate scale α = 32 µm / f  → ring positions in image")
        raw = input(f"  ➤ Enter focal length [{F_DEFAULT}]: ").strip()
        if raw == "":
            f_lens_mm = F_DEFAULT
            break
        try:
            f_lens_mm = float(raw)
            if F_MIN <= f_lens_mm <= F_MAX:
                break
            print(f"    ⚠  Value {f_lens_mm} out of range [{F_MIN}, {F_MAX}]. Please try again.\n")
        except ValueError:
            print(f"    ⚠  Invalid input '{raw}'. Please enter a number.\n")
    print()

    # ── Prompt: Doppler velocity ──────────────────────────────────────────
    V_DEFAULT = 0.0
    V_MIN, V_MAX = -500.0, 500.0
    while True:
        print(f"  Doppler LOS wind velocity [m/s]")
        print(f"    Default: {V_DEFAULT} m/s  (zero wind)")
        print(f"    Range:   {V_MIN} – {V_MAX} m/s")
        raw = input(f"  ➤ Enter velocity [{V_DEFAULT}]: ").strip()
        if raw == "":
            v_rel_ms = V_DEFAULT
            break
        try:
            v_rel_ms = float(raw)
            if V_MIN <= v_rel_ms <= V_MAX:
                break
            print(f"    ⚠  Value {v_rel_ms} out of range [{V_MIN}, {V_MAX}]. Please try again.\n")
        except ValueError:
            print(f"    ⚠  Invalid input '{raw}'. Please enter a number.\n")
    print()

    # ── Prompt: signal-to-noise ratio ────────────────────────────────────
    SNR_DEFAULT = 5.0
    SNR_MIN, SNR_MAX = 1.0, 50.0
    while True:
        print(f"  Target signal-to-noise ratio")
        print(f"    Default: {SNR_DEFAULT}")
        print(f"    Range:   {SNR_MIN} – {SNR_MAX}")
        raw = input(f"  ➤ Enter SNR [{SNR_DEFAULT}]: ").strip()
        if raw == "":
            snr = SNR_DEFAULT
            break
        try:
            snr = float(raw)
            if SNR_MIN <= snr <= SNR_MAX:
                break
            print(f"    ⚠  Value {snr} out of range [{SNR_MIN}, {SNR_MAX}]. Please try again.\n")
        except ValueError:
            print(f"    ⚠  Invalid input '{raw}'. Please enter a number.\n")
    print()

    # Convert to SI units
    t_m      = t_mm * 1e-3
    f_lens_m = f_lens_mm * 1e-3
    alpha    = PIXEL_SIZE_BINNED_M / f_lens_m

    # ── Output path ───────────────────────────────────────────────────────
    output_path = get_output_path()
    print(f"\n  Output file: {output_path}")
    print()

    # ── Synthesise ────────────────────────────────────────────────────────
    print("  Synthesising airglow image …")
    frame, pixel_block, result = synthesise_and_embed(
        t_m=t_m,
        R_refl=R_refl,
        f_lens_m=f_lens_m,
        v_rel_ms=v_rel_ms,
        snr=snr,
    )

    # ── Save ──────────────────────────────────────────────────────────────
    save_bin_file(frame, output_path)
    print(f"  Saved: {output_path.name}  ({output_path.stat().st_size:,} bytes)")

    # ── Get InstrumentParams for figure ───────────────────────────────────
    from src.fpi.m01_airy_forward_model_2026_04_05 import InstrumentParams
    params = InstrumentParams(t=t_m, R_refl=R_refl, alpha=alpha)

    # ── Print parameter table ─────────────────────────────────────────────
    print_parameter_table(
        t_m=t_m,
        R_refl=R_refl,
        f_lens_m=f_lens_m,
        alpha=alpha,
        v_rel_ms=v_rel_ms,
        snr=snr,
        result=result,
        output_path=output_path,
    )

    # ── Tolansky stage: peak finding + single-line WLS fit ───────────────
    print("  Running Tolansky analysis on synthetic profile …")
    tol_result = _run_tolansky_stage(
        profile_1d=result["profile_1d"],
        r_grid=result["r_grid"],
        d_m=t_m,
        f_m=f_lens_m,
        v_input_ms=v_rel_ms,
    )
    if tol_result is not None:
        lc_nm       = tol_result["lambda_c_m"] * 1e9
        sigma_lc_pm = tol_result["sigma_lc_m"] * 1e12
        border = "═" * 66
        print()
        print("── Tolansky Self-Consistency Check (OI 630 nm, fixed d and f) ─" + "─" * 3)
        print(f"  Peaks found          {tol_result['n_peaks']}")
        print(f"  Fixed prior d        {t_mm:.4f} mm   (from neon calibration)")
        print(f"  Fixed prior f        {f_lens_mm:.2f} mm   (from neon calibration)")
        print(f"  ε ± σ(ε)             {tol_result['epsilon']:.6f}  ±  "
              f"{tol_result['sigma_epsilon']:.2e}")
        print(f"  R²                   {tol_result['R2']:.8f}")
        print(f"  Recovered λ_c        {lc_nm:.6f}  ±  {sigma_lc_pm:.4f} pm  nm")
        print(f"  Recovered v_rel      {tol_result['v_rel_ms']:+.1f}  ±  "
              f"{tol_result['sigma_v_ms']:.1f} m/s")
        print(f"  Input v_rel          {tol_result['v_input_ms']:+.1f} m/s"
              f"   Δv = {tol_result['delta_v_ms']:+.1f} m/s")
        print(border)

    # ── Diagnostic figure ─────────────────────────────────────────────────
    show_diagnostic_figure(pixel_block, result, params, output_path,
                           tolansky_result=tol_result)

    input("\nDone. Press Enter to exit.")


if __name__ == "__main__":
    main()
