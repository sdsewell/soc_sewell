"""
load_image_raw_binary.py — Load and display a WindCube FPI binary image.

Supports frames from the bin_frames folder (science / cal / dark).
Frame naming convention: YYYYMMDDThhmmssZ_{type}.bin

Outputs
-------
- <stem>_load_image.png  : diagnostic figure (images + histograms + metadata table)
- <stem>_raw_L0.npy      : 2-D uint16 array, full frame (all rows × cols, including
                           dark reference and overscan columns/rows)
- <stem>_active_L0.npy   : 2-D uint16 array, active pixels only (256×256 for 2×2
                           binned; 512×512 for 1×1 unbinned)
- <stem>_ROI_L1.1.npy    : 2-D uint16 array, user-selected ROI within active pixels

Binary file format (verified against WIND-XCAM-RE-00035 FM#2 test report)
--------------------------------------------------------------------------
All uint16 words are LITTLE-ENDIAN.  Multi-byte fields (float64, uint64) are
stored as four consecutive LE uint16 words in BIG-ENDIAN word order (most-
significant word first, i.e. word[w] = MSW).

Header row — 276 uint16 words (row 0 of the frame)
  word  0       exp_unit      uint16  timer register value (nominal 3850)
  word  1       exp_time      uint16  exposure in timer ticks;
                                      exposure_s = exp_time × 0.001 s
  word  2       cols          uint16  total columns (276 for 2×2 binned)
  word  3       rows          uint16  total rows    (260 for 2×2 binned)
  words  4-7    ccd_temp1     float64 °C
  words  8-11   lua_timestamp uint64  ms, Unix epoch
  words 12-15   adcs_timestamp uint64 ms, Unix epoch (0 = not available)
  words 16-19   lat_hat       float64 rad
  words 20-23   lon_hat       float64 rad
  words 24-27   alt_hat       float64 m
  words 28-43   ads_q_hat[4]  float64 each [w, x, y, z]
  words 44-59   acs_q_err[4]  float64 each [w, x, y, z]
  words 60-71   pos_eci_hat[3] float64 each m
  words 72-83   vel_eci_hat[3] float64 each m/s
  words 84-99   b2_temp_f[4]  float64 each °C  (etalon temperatures)
  words 100-103 gpio_pwr_on[4] uint8 in low byte of each uint16
  words 104-109 lamp_ch_on[6]  uint8 in low byte of each uint16
  words 110-275 (padding / reserved)

Pixel data — rows 1 to (rows-1), each 276 uint16 words, little-endian.
14-bit ADC; valid range 0–16383.

CCD pixel layout — 2×2 binned frame (276 cols × 259 pixel rows)
-----------------------------------------------------------------
Source: CCD97-00 Datasheet (Teledyne e2v) + WIND-XCAM-RE-00035 FM#2 test report.

Columns (serial direction):
  cols   0–11  : 12 dark reference columns  (24 unbinned dark-ref cols / 2)
  cols  12–267 : 256 active pixels          (512 unbinned active cols  / 2)
  cols 268–275 :  8 serial overscan columns (16 unbinned overscan      / 2)

Rows (parallel direction):
  rows   0–255 : 256 active rows            (512 unbinned active rows  / 2)
  row      256 : 1 transition/dark-ref row  (partially shielded)
  rows 257–258 : 2 parallel overscan rows

Active pixel region (science-quality data):
  ACTIVE_COL_START = 12
  ACTIVE_COL_END   = 268   (exclusive)
  ACTIVE_ROW_END   = 256   (exclusive)
  Active shape     = 256 rows × 256 cols  (square ✓)

Known frame sizes
  2×2 binned : 260 rows × 276 cols = 143 520 bytes
  1×1 unbinned: 528 rows × 552 cols = 582 912 bytes

Bug-fix history
---------------
v2 (2026-05-20) — Three issues corrected from v1:
  1. Header words 0-3 field assignment was wrong (rows/cols/exp_time/exp_unit
     were in the wrong order).  Corrected to [exp_unit, exp_time, cols, rows].
  2. _f64() and _u64() reversed word order — wrong multi-byte endianness.
     Correct: word[w] is MSW; pack directly as ">4H" without reversal.
  3. imshow() now uses interpolation="none" to suppress Moiré aliasing.
  Additionally: active pixel region now derived from XCAM/CCD97 spec and
  cropped before display and ROI extraction.  Full raw frame still saved.

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

# ── Active pixel region — from CCD97 datasheet + WIND-XCAM-RE-00035 ───────────
#
# CCD97-00: 512 active cols + 24 dark reference cols + 16 serial overscan cols
#           512 active rows + 8 dark reference rows (top) + 8 dark reference rows (bottom)
# 2×2 binned: all counts halved.
#
# Serial (column) layout per 276-col binned row:
#   cols  0-11  : 12 dark reference cols   (= 24/2)
#   cols 12-267 : 256 active pixel cols    (= 512/2)
#   cols 268-275:  8 serial overscan cols  (= 16/2)
#
# Parallel (row) layout per 259 pixel rows:
#   rows   0-255: 256 active rows          (= 512/2)
#   row      256: 1 transition/dark-ref row
#   rows 257-258: 2 parallel overscan rows
#
# Source: CCD97-00 Datasheet (Teledyne e2v, v6 2017), Figure 4 +
#         WIND-XCAM-RE-00035 Windcube Test Report FM#2 (XCAM, 23/07/2025)

# 2×2 binned
BINNED_ACTIVE_COL_START = 12
BINNED_ACTIVE_COL_END   = 268   # exclusive  → 256 active cols
BINNED_ACTIVE_ROW_END   = 256   # exclusive  → 256 active rows

# 1×1 unbinned  (24 dark ref + 512 active + 16 overscan = 552; 528 rows total)
UNBINNED_ACTIVE_COL_START = 24
UNBINNED_ACTIVE_COL_END   = 536  # exclusive  → 512 active cols
UNBINNED_ACTIVE_ROW_END   = 512  # exclusive  → 512 active rows


def _active_bounds(n_cols: int):
    """Return (col_start, col_end, row_end) for the detected binning mode."""
    if n_cols >= 500:
        return UNBINNED_ACTIVE_COL_START, UNBINNED_ACTIVE_COL_END, UNBINNED_ACTIVE_ROW_END
    return BINNED_ACTIVE_COL_START, BINNED_ACTIVE_COL_END, BINNED_ACTIVE_ROW_END


# ── ROI default half-width ─────────────────────────────────────────────────────
ROI_HALF = 120   # 120 px → 240×240 ROI, fits within 256×256 active region

# Initial fringe centre in ACTIVE-pixel coordinates (row, col).
FRINGE_CENTER_ACTIVE = (128, 128)   # geometric centre of 256×256 active region

# ── Known frame geometries ────────────────────────────────────────────────────
_KNOWN_FRAME_SIZES = [(260, 276), (528, 552)]


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_raw(path: str):
    """
    Load header row and pixel image from a WindCube FPI binary file.

    Uses the corrected field map: word[2]=cols, word[3]=rows.
    Falls back to the known frame size if header dimensions are inconsistent.

    Returns
    -------
    header : ndarray (n_cols,) uint16  — full 276-word header row
    image  : ndarray (n_rows-1, n_cols) uint16 — full pixel frame (incl. overscan)
    """
    data = open(path, "rb").read()
    raw  = np.frombuffer(data, dtype="<u2")

    n_cols_frame = int(raw[2])   # corrected: cols at word[2]
    n_rows_frame = int(raw[3])   # corrected: rows at word[3]

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
    """float64: 4 LE uint16 words in BE word order; word[w] = MSW."""
    b = struct.pack(">4H", h[w], h[w + 1], h[w + 2], h[w + 3])
    return struct.unpack(">d", b)[0]


def _u64(h: np.ndarray, w: int) -> int:
    """uint64: 4 LE uint16 words in BE word order; word[w] = MSW."""
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
        word[0] = exp_unit   (timer register value; nominal 3850)
        word[1] = exp_time   (exposure ticks; exposure_s = ticks × 0.001)
        word[2] = cols
        word[3] = rows
    """
    exp_unit  = _u16(h, 0)
    exp_ticks = _u16(h, 1)
    # exp_unit is a raw timer register value (clock divider / pre-load), NOT
    # a tick rate.  The actual time unit is fixed at 1 ms = 0.001 s.
    exp_s     = exp_ticks * 0.001

    lua_ms  = _u64(h, 8)
    adcs_ms = _u64(h, 12)

    try:
        utc = datetime.fromtimestamp(lua_ms / 1000.0, tz=timezone.utc).isoformat()
    except (OSError, ValueError, OverflowError):
        utc = "invalid"

    gpio  = _u8arr(h, 100, 4)
    lamps = _u8arr(h, 104, 6)

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

def extract_active(image: np.ndarray) -> np.ndarray:
    """
    Crop the full pixel frame to the active pixel region.

    Bounds from CCD97 datasheet + WIND-XCAM-RE-00035:
      2×2 binned  : rows 0-255, cols 12-267  → 256×256
      1×1 unbinned: rows 0-511, cols 24-535  → 512×512
    """
    c0, c1, r1 = _active_bounds(image.shape[1])
    return image[:r1, c0:c1]


def extract_roi(image: np.ndarray, center: tuple, half: int) -> np.ndarray:
    """
    Extract a (2*half × 2*half) ROI centred at (row, col) within the supplied
    array, clamped to the array boundary.
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

def _plot_image(ax, fig, image: np.ndarray, title: str,
                center: tuple = None, roi_half: int = None) -> None:
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
    if center is not None:
        cr, cc = center
        ax.axhline(cr, color="cyan",   linewidth=0.8, linestyle="--", alpha=0.9)
        ax.axvline(cc, color="cyan",   linewidth=0.8, linestyle="--", alpha=0.9)
        _ARM = 12
        ax.plot([cc-_ARM, cc+_ARM], [cr, cr], color="yellow", lw=1.5)
        ax.plot([cc, cc], [cr-_ARM, cr+_ARM], color="yellow", lw=1.5)
    if center is not None and roi_half is not None:
        cr, cc = center
        r_lo = max(0, cr - roi_half);  r_hi = min(image.shape[0], cr + roi_half)
        c_lo = max(0, cc - roi_half);  c_hi = min(image.shape[1], cc + roi_half)
        ax.add_patch(mpatches.Rectangle(
            (c_lo - 0.5, r_lo - 0.5), c_hi - c_lo, r_hi - r_lo,
            linewidth=1.2, edgecolor="red", facecolor="none",
        ))


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
    "exp_unit":             ("Timer register",          "—",        None),
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
    active:         np.ndarray,
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
        f"centred at (row={cr}, col={cc})  [active-pixel coords]"
    )
    active_title = (
        f"Active pixels  "
        f"[cols {BINNED_ACTIVE_COL_START}–{BINNED_ACTIVE_COL_END-1}, "
        f"rows 0–{BINNED_ACTIVE_ROW_END-1}  from raw frame]"
    )

    ax00   = fig.add_subplot(gs[0, 0])
    ax01   = fig.add_subplot(gs[0, 1])
    ax10   = fig.add_subplot(gs[1, 0])
    ax11   = fig.add_subplot(gs[1, 1])
    ax_tbl = fig.add_subplot(gs[2, :])

    _plot_image(ax00, fig, active, active_title,
                center=fringe_center, roi_half=roi_half)
    _plot_image(ax01, fig, roi,    roi_title)
    _plot_hist(ax10, active, "Pixel Distribution — Active region")
    _plot_hist(ax11, roi,    "Pixel Distribution — ROI")

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
        f"ROI half-width: {roi_half} px  |  Centre (active coords): "
        f"col={cc}, row={cr}",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


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
    active        = extract_active(image)

    _, n_cols     = image.shape
    unbinned      = (n_cols >= 500)
    c0, c1, r1    = _active_bounds(n_cols)

    print(f"File          : {filename}")
    print(f"Full frame    : {image.shape[0]} rows × {image.shape[1]} cols")
    print(f"Binning       : {'1×1 unbinned' if unbinned else '2×2 binned'}")
    print(f"Active region : rows 0–{r1-1}, cols {c0}–{c1-1}"
          f"  →  {active.shape[0]}×{active.shape[1]} px")
    print(f"Pixel range   : {active.min()} – {active.max()}  ADU")
    print(f"Mean ± std    : {active.mean():.1f} ± {active.std():.1f}  ADU")
    print(f"UTC           : {metadata['utc_timestamp']}")
    print(f"Exp time      : {metadata['exp_time']} ticks × 0.001 s"
          f" = {metadata['exp_time_s']:.3f} s")
    print(f"CCD temp      : {metadata['ccd_temp1']} °C")
    print(f"Etalon temps  : {metadata['etalon_temps']} °C")
    print(f"Image type    : {metadata['img_type']}")

    # ── Fringe centre and ROI ─────────────────────────────────────────────
    roi_half = 120 if not unbinned else 216
    fc       = FRINGE_CENTER_ACTIVE   # geometric centre of active region
    roi      = extract_roi(active, fc, roi_half)
    print(f"Fringe centre : row {fc[0]}, col {fc[1]}  (active-pixel coords)")
    print(f"ROI shape     : {roi.shape[0]} rows × {roi.shape[1]} cols")

    # ── Metadata table ─────────────────────────────────────────────────────
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

    # ── Figure ─────────────────────────────────────────────────────────────
    fig = _build_figure(
        active, roi, fc, filename,
        col_labels, cell_text, row_heights_in, table_h_in, roi_half,
    )

    # ── Save outputs ───────────────────────────────────────────────────────
    src  = pathlib.Path(bin_file)
    stem = src.stem.replace("_L0", "")

    png_path = src.with_name(stem + "_load_image.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved      : {png_path}")

    plt.show()

    raw_path = src.with_name(stem + "_raw_L0.npy")
    np.save(raw_path, image)
    print(f"Full frame saved  : {raw_path}  "
          f"shape={image.shape}  range=[{image.min()}, {image.max()}]")

    act_path = src.with_name(stem + "_active_L0.npy")
    np.save(act_path, active)
    print(f"Active px saved   : {act_path}  "
          f"shape={active.shape}  range=[{active.min()}, {active.max()}]")

    roi_path = src.with_name(stem + "_ROI_L1.1.npy")
    np.save(roi_path, roi)
    print(f"ROI saved         : {roi_path}  "
          f"shape={roi.shape}  range=[{roi.min()}, {roi.max()}]")


if __name__ == "__main__":
    main()
