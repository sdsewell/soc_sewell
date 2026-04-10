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
# Interactive output path selection
# ---------------------------------------------------------------------------


def get_output_path_from_user(default_stem: str = "z02_synthetic_airglow") -> pathlib.Path:
    """
    Prompt user for a save folder, then return a full .bin file path.

    On Windows: attempt to open a native folder-chooser dialog via
    tkinter.filedialog.askdirectory(). Falls back to a typed-path prompt
    if tkinter is unavailable or the dialog is cancelled.

    Returns
    -------
    pathlib.Path — absolute path to the .bin file to be created
    """
    import datetime
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{default_stem}_{timestamp_str}.bin"

    folder = None

    # Try tkinter folder dialog (Windows primary path)
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.lift()
        root.attributes("-topmost", True)
        folder = filedialog.askdirectory(
            title="Select folder to save synthetic airglow .bin file"
        )
        root.destroy()
    except Exception:
        folder = None

    # Fall back to typed path
    if not folder:
        print("\nEnter the full path to the folder where the .bin file should be saved.")
        print(r"Example (Windows): C:\Users\scott\WindCube\synthetic_images")
        print("Example (Linux):   /home/scott/windcube/synthetic_images")
        folder = input("Folder path: ").strip().strip('"').strip("'")

    if not folder:
        raise RuntimeError("No folder selected — aborting.")

    return pathlib.Path(folder) / filename


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
    params: object,
    output_path: pathlib.Path,
) -> None:
    """
    Show a 3-panel matplotlib diagnostic figure.

    Panel A: 2D airglow image (imshow, gray).
    Panel B: 1D radial profile with fringe peak markers.
    Panel C: Text panel showing decoded header fields from the saved file.
    """
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks

    r_start = 1
    c_start = 10
    img_display = pixel_block[r_start:r_start + 256, c_start:c_start + 256]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Z02 — Synthetic Airglow Image  |  {output_path.name}",
        fontsize=11,
    )

    # ── Panel A: 2D image ──────────────────────────────────────────────────
    ax_a = axes[0]
    im = ax_a.imshow(img_display, cmap="gray", vmin=0, vmax=16383, origin="upper")
    plt.colorbar(im, ax=ax_a, label="ADU (14-bit)")
    ax_a.set_title("Synthetic airglow image (2D)")
    ax_a.set_xlabel("Column (px)")
    ax_a.set_ylabel("Row (px)")

    # ── Panel B: 1D radial profile ─────────────────────────────────────────
    ax_b = axes[1]
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
    ax_c = axes[2]
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

    plt.tight_layout()
    plt.show(block=False)
    print("\nDiagnostic figure displayed. Close the window or press Enter to continue.")


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

    # Convert to SI units
    t_m      = t_mm * 1e-3
    f_lens_m = f_lens_mm * 1e-3
    alpha    = PIXEL_SIZE_BINNED_M / f_lens_m

    # ── Output path ───────────────────────────────────────────────────────
    output_path = get_output_path_from_user()
    print(f"\n  Output file: {output_path}")
    print()

    # ── Synthesise ────────────────────────────────────────────────────────
    print("  Synthesising airglow image …")
    frame, pixel_block, result = synthesise_and_embed(
        t_m=t_m,
        R_refl=R_refl,
        f_lens_m=f_lens_m,
        v_rel_ms=0.0,
        snr=5.0,
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
        v_rel_ms=0.0,
        snr=5.0,
        result=result,
        output_path=output_path,
    )

    # ── Diagnostic figure ─────────────────────────────────────────────────
    show_diagnostic_figure(pixel_block, result, params, output_path)

    input("\nDone. Press Enter to exit.")


if __name__ == "__main__":
    main()
