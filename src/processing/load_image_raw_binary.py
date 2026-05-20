"""
load_image_raw_binary.py — Load and display a WindCube FPI binary image.

Supports frames from the bin_frames folder (science / cal / dark).
Frame naming convention: YYYYMMDDThhmmssZ_{type}.bin

Outputs
-------
- <stem>_load_image.png  : diagnostic figure (images + histograms + metadata table)
- <stem>_raw_L0.npy      : 2-D uint16 array, full image (all pixel rows × cols)
- <stem>_ROI_L1.1.npy    : 2-D uint16 array, user-selected ROI centred on fringes

Binary file format (empirically verified against real calibration frames)
--------------------------------------------------------------------------
All uint16 words are LITTLE-ENDIAN.  Multi-byte fields (float64, uint64) are
stored as four consecutive LE uint16 words in BIG-ENDIAN word order (most-
significant word first, i.e. word[w] = MSW).

Header row — 276 uint16 words (row 0 of the frame)
  word  0       exp_unit      uint16  timer tick rate (ticks/s; nominal 3850)
  word  1       exp_time      uint16  exposure duration in timer ticks
                                      exposure_s = exp_time × 0.001 s
  word  2       cols          uint16  total columns per row (incl. header row)
  word  3       rows          uint16  total rows (incl. header row)
  words  4-7    ccd_temp1     float64 °C
  words  8-11   lua_timestamp uint64  ms, Unix epoch
  words 12-15   adcs_timestamp uint64 ms, Unix epoch (0 = not available)
  words 16-19   lat_hat       float64 rad  (spacecraft geodetic latitude)
  words 20-23   lon_hat       float64 rad  (spacecraft longitude)
  words 24-27   alt_hat       float64 m    (spacecraft altitude)
  words 28-43   ads_q_hat[4]  float64 each [w, x, y, z]  attitude quaternion
  words 44-59   acs_q_err[4]  float64 each [w, x, y, z]  pointing-error quaternion
  words 60-71   pos_eci_hat[3] float64 each m
  words 72-83   vel_eci_hat[3] float64 each m/s
  words 84-99   b2_temp_f[4]  float64 each °C  (etalon temperatures)
  words 100-103 gpio_pwr_on[4] uint8 in low byte of each uint16
  words 104-109 lamp_ch_on[6]  uint8 in low byte of each uint16
  words 110-275 (padding / reserved)

Pixel data — rows 1 to (rows-1), each 276 uint16 words, little-endian.
14-bit ADC; valid range 0–16383.

Known frame sizes
  2×2 binned : 260 rows × 276 cols = 143 520 bytes
  1×1 unbinned: 528 rows × 552 cols = 582 912 bytes

Bug-fix history
---------------
v2 (2026-05-20) — Three issues corrected from v1:
  1. Header words 0-3 field assignment was wrong.  Actual order is
     [exp_unit, exp_time, cols, rows], not [rows, cols, exp_time, exp_unit].
  2. _f64() and _u64() reversed the word order before packing, producing
     completely wrong values.  Correct convention: word[w] is MSW; pack
     directly as struct.pack(">4H", h[w], h[w+1], h[w+2], h[w+3]).
  3. imshow() now uses interpolation="none" to suppress Moiré aliasing that
     appeared when matplotlib downsampled fine (≈10 px) fringe rings.

Usage
-----
    python load_image_raw_binary.py
A Windows file-open dialog appears; select any .bin frame file.
"""

import os
import pathlib
import struct
import tkinter as tk
from datetime import datetime, timezone
from tkinter import filedialog

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# ── User settings ──────────────────────────────────────────────────────────────

# Half-width/height of the ROI in pixels  (ROI = 2×ROI_HALF × 2×ROI_HALF)
ROI_HALF = 130

# Initial fringe centre (row, col) within the pixel image (row 0 = first pixel
# row, i.e. the header row has already been stripped).
FRINGE_CENTER = (129, 138)

# Show masked (dark rows/columns removed) image alongside the unmasked one.
MASK_DARK = True

# Rows/cols whose mean falls below this fraction of the image median are dark.
DARK_THRESHOLD = 0.5

# ── Known frame geometries ────────────────────────────────────────────────────

_KNOWN_FRAME_SIZES = [(260, 276), (528, 552)]


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_raw(path: str):
    """
    Load the header row and pixel image from a WindCube FPI binary file.

    Frame dimensions are read from header words 2 (cols) and 3 (rows) with the
    corrected field mapping.  Falls back to the known frame size matching the
    file size if the header dimensions are inconsistent.

    Returns
    -------
    header : ndarray (n_cols,) uint16  — full 276-word header row
    image  : ndarray (n_rows-1, n_cols) uint16 — pixel data, little-endian
    """
    data = open(path, "rb").read()
    raw  = np.frombuffer(data, dtype="<u2")

    # Read cols (word 2) and rows (word 3) with the corrected field order.
    n_cols_frame = int(raw[2])
    n_rows_frame = int(raw[3])

    actual   = len(data)
    expected = n_rows_frame * n_cols_frame * 2
    if actual != expected:
        for rows, cols in _KNOWN_FRAME_SIZES:
            if rows * cols * 2 == actual:
                print(
                    f"WARNING: Header says {n_rows_frame}×{n_cols_frame} but "
                    f"file is {actual} bytes — using known geometry {rows}×{cols}."
                )
                n_rows_frame, n_cols_frame = rows, cols
                break
        else:
            raise ValueError(
                f"File size mismatch: {actual} bytes, "
                f"expected {expected} for {n_rows_frame}×{n_cols_frame}, "
                f"and size matches no known frame geometry."
            )

    header = raw[:n_cols_frame].copy()
    image  = raw[n_cols_frame:].reshape(n_rows_frame - 1, n_cols_frame)
    return header, image


# ─────────────────────────────────────────────────────────────────────────────
# Header-decoding helpers
# ─────────────────────────────────────────────────────────────────────────────

def _u16(h: np.ndarray, w: int) -> int:
    return int(h[w])


def _f64(h: np.ndarray, w: int) -> float:
    """
    Decode a float64 stored as four LE uint16 words in BE word order.
    word[w] = MSW; pack directly without reversal.
    """
    b = struct.pack(">4H", h[w], h[w + 1], h[w + 2], h[w + 3])
    return struct.unpack(">d", b)[0]


def _u64(h: np.ndarray, w: int) -> int:
    """
    Decode a uint64 stored as four LE uint16 words in BE word order.
    word[w] = MSW.
    """
    return sum(int(h[w + (3 - i)]) << (16 * i) for i in range(4))


def _u8arr(h: np.ndarray, w: int, n: int) -> list:
    return [int(h[w + i]) & 0xFF for i in range(n)]


# ─────────────────────────────────────────────────────────────────────────────
# Header parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_header(h: np.ndarray) -> dict:
    """
    Decode the 276-word header row into a metadata dict.

    Corrected field map (words 0-3):
        word[0] = exp_unit   (timer tick rate, ticks/s)
        word[1] = exp_time   (exposure duration in ticks)
        word[2] = cols
        word[3] = rows
    """
    exp_unit  = _u16(h, 0)
    exp_ticks = _u16(h, 1)
    # exp_unit is a raw timer register value (clock divider / pre-load).
    # The actual time unit is fixed at 1 ms = 0.001 s regardless of exp_unit.
    # exp_unit=3850 is the nominal register value stored by the firmware;
    # it does NOT mean 3850 ticks/s.  Do not divide by it.
    exp_s     = exp_ticks * 0.001

    lua_ms  = _u64(h, 8)
    adcs_ms = _u64(h, 12)

    try:
        utc = datetime.fromtimestamp(lua_ms / 1000.0, tz=timezone.utc).isoformat()
    except (OSError, ValueError, OverflowError):
        utc = "invalid"

    gpio  = _u8arr(h, 100, 4)
    lamps = _u8arr(h, 104, 6)

    # Attitude quaternion stored [w, x, y, z]; JSON convention is [x, y, z, w]
    q_wxyz = [_f64(h, 28 + i * 4) for i in range(4)]
    q_xyzw = [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]

    e_wxyz = [_f64(h, 44 + i * 4) for i in range(4)]
    e_xyzw = [e_wxyz[1], e_wxyz[2], e_wxyz[3], e_wxyz[0]]

    shutter  = "closed" if (gpio[0] == 1 and gpio[3] == 1) else "open"
    any_lamp = any(lamps)
    if any_lamp:
        img_type = "cal"
    elif shutter == "closed":
        img_type = "dark"
    else:
        img_type = "science"

    return {
        "rows":                  _u16(h, 3),
        "cols":                  _u16(h, 2),
        "exp_time":              exp_ticks,
        "exp_unit":              exp_unit,
        "exp_time_s":            round(exp_s, 4),
        "ccd_temp1":             round(_f64(h, 4), 4),
        "lua_timestamp":         lua_ms,
        "adcs_timestamp":        adcs_ms,
        "utc_timestamp":         utc,
        "attitude_quaternion":   q_xyzw,
        "pointing_error":        e_xyzw,
        "spacecraft_position":   [_f64(h, 60 + i * 4) for i in range(3)],
        "spacecraft_velocity":   [_f64(h, 72 + i * 4) for i in range(3)],
        "spacecraft_latitude":   _f64(h, 16),
        "spacecraft_longitude":  _f64(h, 20),
        "spacecraft_altitude":   _f64(h, 24),
        "etalon_temps":          [round(_f64(h, 84 + i * 4), 4) for i in range(4)],
        "gpio_pwr_on":           gpio,
        "shutter_status":        shutter,
        "lamp_ch_array":         lamps,
        "lamp1_status":          "on" if lamps[0] else "off",
        "lamp2_status":          "on" if lamps[1] else "off",
        "lamp3_status":          "on" if lamps[2] else "off",
        "img_type":              img_type,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Image utilities
# ─────────────────────────────────────────────────────────────────────────────

def mask_dark_borders(image: np.ndarray, threshold: float = 0.5):
    """Remove rows/columns whose mean is below threshold × image median."""
    med      = float(np.median(image))
    row_mask = image.mean(axis=1) >= threshold * med
    col_mask = image.mean(axis=0) >= threshold * med
    return image[np.ix_(row_mask, col_mask)], row_mask, col_mask


def extract_roi(image: np.ndarray, center: tuple, half: int) -> np.ndarray:
    """
    Extract a (2*half × 2*half) ROI centred at (row, col), clamped to the
    image boundary.
    """
    r0, c0 = center
    r_lo = max(0, r0 - half)
    r_hi = min(image.shape[0], r0 + half)
    c_lo = max(0, c0 - half)
    c_hi = min(image.shape[1], c0 + half)
    return image[r_lo:r_hi, c_lo:c_hi]


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

def _plot_image(ax, fig, image: np.ndarray, title: str) -> None:
    vlo = float(np.percentile(image,  1))
    vhi = float(np.percentile(image, 99))
    im  = ax.imshow(
        image, cmap="gray", origin="lower",
        vmin=vlo, vmax=vhi, aspect="equal",
        interpolation="none",   # prevent Moiré aliasing on fine fringes
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Counts  (ADU)", fontsize=8)
    ax.set_title(
        f"{title}\n"
        f"{image.shape[0]} rows × {image.shape[1]} cols  |  "
        f"ADU [{image.min()}, {image.max()}]  |  "
        f"mean {image.mean():.0f}  std {image.std():.1f}",
        fontsize=8.5,
    )
    ax.set_xlabel("Column  (pixel)", fontsize=8)
    ax.set_ylabel("Row  (pixel)",    fontsize=8)
    ax.tick_params(labelsize=7)


def _plot_hist(ax, image: np.ndarray, title: str) -> None:
    vlo = float(np.percentile(image,  1))
    vhi = float(np.percentile(image, 99))
    ax.hist(image.ravel(), bins=256, color="steelblue", edgecolor="none")
    ax.axvline(vlo, color="orange", linestyle="--", linewidth=1,
               label=f"1st pct  ({vlo:.0f})")
    ax.axvline(vhi, color="red",    linestyle="--", linewidth=1,
               label=f"99th pct ({vhi:.0f})")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("ADU  (uint16 counts)", fontsize=8)
    ax.set_ylabel("Number of pixels",     fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)


# ─────────────────────────────────────────────────────────────────────────────
# Metadata table helpers
# ─────────────────────────────────────────────────────────────────────────────

_FIELD_META = {
    "rows":                 ("Rows",                    "pixels",   None),
    "cols":                 ("Cols",                    "pixels",   None),
    "exp_time":             ("Exposure ticks",          "ticks",    None),
    "exp_unit":             ("Tick rate",               "ticks/s",  None),
    "exp_time_s":           ("Exposure time",           "s",
                             lambda v: f"{v:.4f} s"),
    "ccd_temp1":            ("CCD temperature",         "°C",       None),
    "lua_timestamp":        ("Lua timestamp",           "ms (Unix)",None),
    "adcs_timestamp":       ("ADCS timestamp",          "ms (Unix)",None),
    "utc_timestamp":        ("UTC timestamp",           "",         None),
    "attitude_quaternion":  ("Attitude quaternion",     "[x,y,z,w]",None),
    "pointing_error":       ("Pointing error",          "[x,y,z,w]",None),
    "spacecraft_position":  ("SC position (ECI)",       "m",        None),
    "spacecraft_velocity":  ("SC velocity (ECI)",       "m/s",      None),
    "spacecraft_latitude":  ("SC latitude",             "rad",      None),
    "spacecraft_longitude": ("SC longitude",            "rad",      None),
    "spacecraft_altitude":  ("SC altitude",             "m",        None),
    "etalon_temps":         ("Etalon temperatures",     "°C",       None),
    "gpio_pwr_on":          ("GPIO power on",           "[ch0–3]",  None),
    "shutter_status":       ("Shutter status",          "",         None),
    "lamp_ch_array":        ("Lamp channel array",      "",         None),
    "lamp1_status":         ("Lamp 1 status",           "",         None),
    "lamp2_status":         ("Lamp 2 status",           "",         None),
    "lamp3_status":         ("Lamp 3 status",           "",         None),
    "img_type":             ("Image type",              "",         None),
}


def _fmt_value(key: str, raw) -> str:
    meta = _FIELD_META.get(key)
    if meta and meta[2] is not None:
        return meta[2](raw)
    if isinstance(raw, list):
        return "[" + ",  ".join(
            f"{v:.5g}" if isinstance(v, float) else str(v) for v in raw
        ) + "]"
    if isinstance(raw, float):
        return f"{raw:.6g}"
    return str(raw)


# ─────────────────────────────────────────────────────────────────────────────
# Figure builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_figure(
    image:          np.ndarray,
    roi:            np.ndarray,
    fringe_center:  tuple,
    filename:       str,
    col_labels:     list,
    cell_text:      list,
    row_heights_in: list,
    table_h_in:     float,
    roi_half:       int,
) -> plt.Figure:
    _HDR_ROW  = 0.32
    img_row_h = 5.0
    total_h   = img_row_h * 2 + table_h_in

    fig = plt.figure(figsize=(14.0, total_h))
    gs  = GridSpec(3, 2, figure=fig,
                   height_ratios=[img_row_h, img_row_h, table_h_in])

    cr, cc    = fringe_center
    roi_title = (
        f"ROI  {roi.shape[0]}×{roi.shape[1]} px  "
        f"centred at (row={cr}, col={cc})"
    )

    ax00   = fig.add_subplot(gs[0, 0])
    ax01   = fig.add_subplot(gs[0, 1])
    ax10   = fig.add_subplot(gs[1, 0])
    ax11   = fig.add_subplot(gs[1, 1])
    ax_tbl = fig.add_subplot(gs[2, :])

    _plot_image(ax00, fig, image, "Unmasked (full frame)")
    _plot_image(ax01, fig, roi,   roi_title)

    # Crosshair and ROI rectangle on the full-frame panel
    ax00.axhline(cr, color="cyan",   linewidth=0.8, linestyle="--", alpha=0.9)
    ax00.axvline(cc, color="cyan",   linewidth=0.8, linestyle="--", alpha=0.9)
    _ARM = 15
    ax00.plot([cc - _ARM, cc + _ARM], [cr, cr], color="yellow",
              linewidth=1.5, linestyle="-", alpha=1.0)
    ax00.plot([cc, cc], [cr - _ARM, cr + _ARM], color="yellow",
              linewidth=1.5, linestyle="-", alpha=1.0)
    r_lo = max(0, cr - roi_half);  r_hi = min(image.shape[0], cr + roi_half)
    c_lo = max(0, cc - roi_half);  c_hi = min(image.shape[1], cc + roi_half)
    ax00.add_patch(mpatches.Rectangle(
        (c_lo - 0.5, r_lo - 0.5), c_hi - c_lo, r_hi - r_lo,
        linewidth=1.2, edgecolor="red", facecolor="none",
    ))

    _plot_hist(ax10, image, "Pixel Distribution — Unmasked")
    _plot_hist(ax11, roi,   "Pixel Distribution — ROI")

    # Metadata table
    ax_tbl.axis("off")
    tbl = ax_tbl.table(
        cellText=cell_text,
        colLabels=col_labels,
        colWidths=[0.03, 0.16, 0.16, 0.10, 0.53],
        loc="upper center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)

    hdr_bg = "#2C3E50"
    alt_bg = "#EBF5FB"
    n_cols = len(col_labels)

    for c in range(n_cols):
        tbl[0, c].set_height(_HDR_ROW / table_h_in)
        tbl[0, c].set_facecolor(hdr_bg)
        tbl[0, c].set_text_props(color="white", fontweight="bold")
        tbl[0, c].set_edgecolor("#CCCCCC")

    for r_idx, h_in in enumerate(row_heights_in):
        for c in range(n_cols):
            cell = tbl[r_idx + 1, c]
            cell.set_height(h_in / table_h_in)
            cell.set_edgecolor("#CCCCCC")
            if r_idx % 2 == 1:
                cell.set_facecolor(alt_bg)

    ax_tbl.set_title(
        f"WindCube FPI Metadata (from binary header row) — {filename}",
        fontsize=11, fontweight="bold", pad=8,
    )
    fig.suptitle(
        f"WindCube FPI — {filename}\n"
        f"ROI half-width: {roi_half} px  |  Centre: cx = {cc}, cy = {cr}",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Regression checks (run on import during testing)
# ─────────────────────────────────────────────────────────────────────────────

def _regression_check(path: str, expected: dict) -> None:
    """
    Assert that key metadata fields decode to expected values.
    Raises AssertionError with a descriptive message on failure.
    """
    h, _ = load_raw(path)
    meta = parse_header(h)
    for field, want in expected.items():
        got = meta[field]
        if isinstance(want, float):
            assert abs(got - want) < 0.5, \
                f"[REGRESSION {path}] {field}: got {got}, expected {want}"
        elif isinstance(want, tuple):
            lo, hi = want
            assert lo <= got <= hi, \
                f"[REGRESSION {path}] {field}={got} outside [{lo}, {hi}]"
        else:
            assert got == want, \
                f"[REGRESSION {path}] {field}: got {got!r}, expected {want!r}"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── File selection ─────────────────────────────────────────────────────
    root = tk.Tk()
    root.withdraw()
    raw_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        r"..\raw_images_with_metadata",
    )
    bin_file = filedialog.askopenfilename(
        title="Select FPI binary image",
        initialdir=os.path.normpath(raw_dir),
        filetypes=[("Binary image", "*.bin"), ("All files", "*.*")],
    )
    root.destroy()
    if not bin_file:
        print("No file selected — exiting.")
        return

    # ── Load + decode ──────────────────────────────────────────────────────
    header, image = load_raw(bin_file)
    filename      = os.path.basename(bin_file)
    metadata      = parse_header(header)

    print(f"File          : {filename}")
    print(f"Shape         : {image.shape[0]} rows × {image.shape[1]} cols")
    print(f"Pixel range   : {image.min()} – {image.max()}  ADU")
    print(f"Mean ± std    : {image.mean():.1f} ± {image.std():.1f}  ADU")
    print(f"UTC           : {metadata['utc_timestamp']}")
    print(f"Exp time      : {metadata['exp_time']} ticks / {metadata['exp_unit']} ticks/s"
          f" = {metadata['exp_time_s']:.3f} s")
    print(f"CCD temp      : {metadata['ccd_temp1']} °C")
    print(f"Etalon temps  : {metadata['etalon_temps']} °C")
    print(f"Image type    : {metadata['img_type']}")

    # ── Binning detection ──────────────────────────────────────────────────
    _, n_cols    = image.shape
    unbinned     = (n_cols >= 500)
    roi_half_def = 216 if unbinned else ROI_HALF
    fc_def       = FRINGE_CENTER
    print(f"Binning       : {'1×1 unbinned' if unbinned else '2×2 binned'}")
    print(f"ROI half-size : {roi_half_def} px  ({roi_half_def * 2}×{roi_half_def * 2})")

    # ── Table layout pre-computation ───────────────────────────────────────
    col_labels = ["#", "Field (key)", "Display name", "Units", "Value"]
    cell_text  = [
        [str(i), key,
         _FIELD_META.get(key, (key, "", None))[0],
         _FIELD_META.get(key, (key, "", None))[1],
         _fmt_value(key, val)]
        for i, (key, val) in enumerate(metadata.items(), start=1)
    ]
    _LINE_H  = 0.20
    _MIN_ROW = 0.28
    _HDR_ROW = 0.32
    row_heights_in = [
        max(_MIN_ROW, max(v.count("\n") + 1 for v in row) * _LINE_H)
        for row in cell_text
    ]
    table_h_in = max(6.0, sum(row_heights_in) + _HDR_ROW + 1.2)

    # ── Build figure ───────────────────────────────────────────────────────
    roi = extract_roi(image, fc_def, roi_half_def)
    print(f"Fringe centre : row {fc_def[0]}, col {fc_def[1]}")
    print(f"ROI shape     : {roi.shape[0]} rows × {roi.shape[1]} cols")

    fig = _build_figure(
        image, roi, fc_def, filename,
        col_labels, cell_text, row_heights_in, table_h_in, roi_half_def,
    )

    # ── Save outputs ───────────────────────────────────────────────────────
    src      = pathlib.Path(bin_file)
    stem     = src.stem.replace("_L0", "")

    png_path = src.with_name(stem + "_load_image.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved    : {png_path}")

    plt.show()

    raw_path = src.with_name(stem + "_raw_L0.npy")
    np.save(raw_path, image)
    print(f"Raw image saved : {raw_path}  "
          f"shape={image.shape}  dtype={image.dtype}  "
          f"range=[{image.min()}, {image.max()}]")

    roi_path = src.with_name(stem + "_ROI_L1.1.npy")
    np.save(roi_path, roi)
    print(f"ROI saved       : {roi_path}  "
          f"shape={roi.shape}  dtype={roi.dtype}  "
          f"range=[{roi.min()}, {roi.max()}]")


if __name__ == "__main__":
    main()
