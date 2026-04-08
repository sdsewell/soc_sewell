"""
ingest_raw_image.py — Load and display a WindCube FPI binary image from the
raw_images_with_metadata/ folder.

Binary file format
------------------
- uint16, big-endian (*_swapped.bin files).  14-bit ADC values; valid range 0–16383.
- Row 0  (276 uint16 words) : embedded metadata header.
- Rows 1-259 (259 × 276)   : image pixels.
- Total: 260 rows × 276 cols = 71 760 uint16 = 143 520 bytes.

Metadata binary layout (empirically verified against *_analyzed.txt JSON files)
--------------------------------------------------------------------------------
Multi-byte fields (uint64, float64) use mixed-endian encoding: each field is
stored as 4 consecutive big-endian uint16 words in little-endian word order
(least-significant word first).  To decode: collect the 4 words, reverse their
order to get MSW-first, pack as 4 big-endian uint16s, then unpack as a
big-endian float64 or construct as a LE uint64.

  Words  0      : rows            uint16  (total rows incl. header row)
  Word   1      : cols            uint16
  Word   2      : exp_time        uint16  (centiseconds)
  Word   3      : exp_unit        uint16
  Words  4-7    : ccd_temp1       float64 (°C)
  Words  8-11   : lua_timestamp   uint64  (ms, Unix epoch)
  Words  12-15  : adcs_timestamp  uint64  (ms, Unix epoch; 0 = not available)
  Words  16-19  : lat_hat         float64 (rad)
  Words  20-23  : lon_hat         float64 (rad)
  Words  24-27  : alt_hat         float64 (m)
  Words  28-43  : ads_q_hat[4]    float64 each  [w, x, y, z]
  Words  44-59  : acs_q_err[4]    float64 each  [w, x, y, z]
  Words  60-71  : pos_eci_hat[3]  float64 each  (m)
  Words  72-83  : vel_eci_hat[3]  float64 each  (m/s)
  Words  84-99  : b2_temp_f[4]    float64 each  (°C, etalon temps)
  Words  100-103: gpio_pwr_on[4]  uint8 in low byte of uint16
  Words  104-109: lamp_ch_on[6]   uint8 in low byte of uint16
  Words  110-275: (padding / reserved)

Usage
-----
    python Scott/1-ingest/ingest_raw_image.py
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

# ── User settings ─────────────────────────────────────────────────────────────

# Set True to show masked (dark rows/columns removed) alongside unmasked image.
MASK_DARK = True

# Rows or columns whose mean falls below this fraction of the image median are
# considered dark and excluded from the masked image.
DARK_THRESHOLD = 0.5

# Centre of the fringe pattern estimated by visual inspection (row, col).
# This is used to extract a fixed-size ROI for the second subplot.
# Adjust these values after examining the unmasked image.
FRINGE_CENTER = (142, 145)   # (row, col) — update after visual inspection

# Half-width/height of the ROI in pixels (ROI will be 2×ROI_HALF × 2×ROI_HALF).
# 240-pixel ROI → ROI_HALF = 120
ROI_HALF = 108

# ── Fixed geometry ─────────────────────────────────────────────────────────────

ROWS, COLS = 259, 276   # image pixels only (excludes header row)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_raw(path: str):
    """
    Load the header row and image pixel region from a big-endian FPI binary.

    Returns
    -------
    header_be : ndarray (COLS,) uint16  — header words, big-endian decoded
    image     : ndarray (ROWS, COLS) uint16 — pixel data
    """
    expected = (ROWS + 1) * COLS * 2
    actual = os.path.getsize(path)
    if actual != expected:
        raise ValueError(
            f"File size mismatch: got {actual} bytes, "
            f"expected {expected} for a ({ROWS}+1)×{COLS} uint16 image."
        )
    raw = np.frombuffer(open(path, "rb").read(), dtype=">u2")
    return raw[:COLS].copy(), raw[COLS:].reshape(ROWS, COLS)


# ---------------------------------------------------------------------------
# Header decoding helpers
# ---------------------------------------------------------------------------

def _u16(h: np.ndarray, w: int) -> int:
    return int(h[w])


def _u64(h: np.ndarray, w: int) -> int:
    """Mixed-endian uint64: 4 BE uint16 words in LE word order (LSW at w)."""
    return sum(int(h[w + i]) << (16 * i) for i in range(4))


def _f64(h: np.ndarray, w: int) -> float:
    """Mixed-endian float64: 4 BE uint16 words in LE word order (LSW at w)."""
    b = struct.pack(">4H", *reversed([h[w + i] for i in range(4)]))
    return struct.unpack(">d", b)[0]


def _u8arr(h: np.ndarray, w: int, n: int) -> list:
    return [int(h[w + i]) & 0xFF for i in range(n)]


# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------

def parse_header(h: np.ndarray) -> dict:
    """
    Decode the 276-word header row into a metadata dict.
    Keys and value types match the *_analyzed.txt JSON files.
    """
    lua_ms  = _u64(h, 8)
    adcs_ms = _u64(h, 12)

    # UTC timestamp derived from lua_timestamp (Unix ms)
    try:
        utc = datetime.fromtimestamp(lua_ms / 1000.0, tz=timezone.utc).isoformat()
    except (OSError, ValueError, OverflowError):
        utc = "invalid"

    gpio   = _u8arr(h, 100, 4)
    lamps  = _u8arr(h, 104, 6)

    # Attitude quaternion stored [w, x, y, z]; JSON convention is [x, y, z, w]
    q_wxyz = [_f64(h, 28 + i * 4) for i in range(4)]
    q_xyzw = [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]

    e_wxyz = [_f64(h, 44 + i * 4) for i in range(4)]
    e_xyzw = [e_wxyz[1], e_wxyz[2], e_wxyz[3], e_wxyz[0]]

    # Shutter: gpio_pwr_on[0]==1 and gpio_pwr_on[3]==1 → closed (empirical)
    shutter = "closed" if (gpio[0] == 1 and gpio[3] == 1) else "open"

    # Image type classification
    any_lamp = any(lamps)
    if any_lamp:
        img_type = "cal"
    elif shutter == "closed":
        img_type = "dark"
    else:
        img_type = "science"

    return {
        "rows":                  _u16(h, 0),
        "cols":                  _u16(h, 1),
        "exp_time":              _u16(h, 2),
        "exp_unit":              _u16(h, 3),
        "ccd_temp1":             round(_f64(h, 4), 4),
        "lua_timestamp":         lua_ms,
        "adcs_timestamp":        adcs_ms,
        "utc_timestamp":         utc,
        "attitude_quadternion":  q_xyzw,
        "pointing_error":        e_xyzw,
        "spacecraft_position":   [_f64(h, 60 + i * 4) for i in range(3)],
        "spacecraft_velocity":   [_f64(h, 72 + i * 4) for i in range(3)],
        "spacecraft_latitude":   _f64(h, 16),
        "spacecraft_longitude":  _f64(h, 20),
        "spacecraft_altitude":   _f64(h, 24),
        "etalon_temps":          [_f64(h, 84 + i * 4) for i in range(4)],
        "gpio_pwr_on":           gpio,
        "shutter_status":        shutter,
        "lamp_ch_array":         lamps,
        "lamp1_status":          "on" if lamps[0] else "off",
        "lamp2_status":          "on" if lamps[1] else "off",
        "lamp3_status":          "on" if lamps[2] else "off",
        "img_type":              img_type,
    }


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

def mask_dark_borders(image: np.ndarray, threshold: float = 0.5):
    """
    Crop rows and columns whose mean falls below threshold × image median.

    Returns
    -------
    cropped  : ndarray  — pixel data with dark borders removed
    row_mask : ndarray (bool) — True for kept rows
    col_mask : ndarray (bool) — True for kept cols
    """
    med = float(np.median(image))
    row_mask = image.mean(axis=1) >= threshold * med
    col_mask = image.mean(axis=0) >= threshold * med
    return image[np.ix_(row_mask, col_mask)], row_mask, col_mask


def extract_roi(image: np.ndarray, center: tuple, half: int) -> np.ndarray:
    """
    Extract a (2*half) × (2*half) ROI centred at (row, col).

    The window is clamped to the image boundary if the centre is close to an
    edge, so the caller always receives a contiguous sub-array (which may be
    smaller than 2*half × 2*half in that case).

    Parameters
    ----------
    image  : 2-D ndarray
    center : (row, col) in pixel coordinates of the full image
    half   : half-size of the ROI window in pixels

    Returns
    -------
    roi : ndarray — sub-array of image
    """
    r0, c0 = center
    r_lo = max(0, r0 - half)
    r_hi = min(image.shape[0], r0 + half)
    c_lo = max(0, c0 - half)
    c_hi = min(image.shape[1], c0 + half)
    return image[r_lo:r_hi, c_lo:c_hi]


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_image(ax, fig, image: np.ndarray, title: str) -> None:
    vlo = float(np.percentile(image,  1))
    vhi = float(np.percentile(image, 99))
    im = ax.imshow(image, cmap="gray", origin="lower",
                   vmin=vlo, vmax=vhi, aspect="equal")
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


# ---------------------------------------------------------------------------
# Metadata table figure  (matches ingest_metadata_txt.py layout)
# ---------------------------------------------------------------------------

_FIELD_META = {
    "rows":                  ("Rows",                   "pixels",      None),
    "cols":                  ("Cols",                   "pixels",      None),
    "exp_time":              ("Exposure time",           "cs",
                              lambda v: f"{v} cs  ({v / 100:.2f} s)"),
    "exp_unit":              ("Exposure unit",           "register",    None),
    "ccd_temp1":             ("CCD temperature",         "°C",          None),
    "lua_timestamp":         ("Lua timestamp",           "ms (Unix)",   None),
    "adcs_timestamp":        ("ADCS timestamp",          "ms (Unix)",   None),
    "utc_timestamp":         ("UTC timestamp",           "",            None),
    "attitude_quadternion":  ("Attitude quaternion",     "[x,y,z,w]",   None),
    "pointing_error":        ("Pointing error",          "[x,y,z,w]",   None),
    "spacecraft_position":   ("SC position (ECI)",       "m",           None),
    "spacecraft_velocity":   ("SC velocity (ECI)",       "m/s",         None),
    "spacecraft_latitude":   ("SC latitude",             "rad",         None),
    "spacecraft_longitude":  ("SC longitude",            "rad",         None),
    "spacecraft_altitude":   ("SC altitude",             "m",           None),
    "etalon_temps":          ("Etalon temperatures",     "°C",          None),
    "gpio_pwr_on":           ("GPIO power on",           "[ch0–3]",     None),
    "shutter_status":        ("Shutter status",          "",            None),
    "lamp_ch_array":         ("Lamp channel array",      "",            None),
    "lamp1_status":          ("Lamp 1 status",           "",            None),
    "lamp2_status":          ("Lamp 2 status",           "",            None),
    "lamp3_status":          ("Lamp 3 status",           "",            None),
    "img_type":              ("Image type",              "",            None),
}


def _fmt_value(key: str, raw) -> str:
    meta = _FIELD_META.get(key)
    if meta and meta[2] is not None:
        return meta[2](raw)
    if isinstance(raw, list):
        return "[" + ",  ".join(
            f"{v:.6g}" if isinstance(v, float) else str(v) for v in raw
        ) + "]"
    if isinstance(raw, float):
        return f"{raw:.6g}"
    return str(raw)


def build_metadata_figure(metadata: dict, filename: str):
    col_labels = ["#", "Field (key)", "Display name", "Units", "Value"]
    cell_text = [
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
    fig_h = max(6.0, sum(row_heights_in) + _HDR_ROW + 1.2)

    fig, ax = plt.subplots(figsize=(13, fig_h))
    ax.axis("off")

    tbl = ax.table(
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
        tbl[0, c].set_height(_HDR_ROW / fig_h)
        tbl[0, c].set_facecolor(hdr_bg)
        tbl[0, c].set_text_props(color="white", fontweight="bold")
        tbl[0, c].set_edgecolor("#CCCCCC")

    for r_idx, h_in in enumerate(row_heights_in):
        for c in range(n_cols):
            cell = tbl[r_idx + 1, c]
            cell.set_height(h_in / fig_h)
            cell.set_edgecolor("#CCCCCC")
            if r_idx % 2 == 1:
                cell.set_facecolor(alt_bg)

    ax.set_title(
        f"WindCube FPI Metadata (from binary header row) — {filename}",
        fontsize=11, fontweight="bold", pad=8,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    root = tk.Tk()
    root.withdraw()
    raw_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           r"..\raw_images_with_metadata")
    bin_file = filedialog.askopenfilename(
        title="Select FPI binary image",
        initialdir=os.path.normpath(raw_dir),
        filetypes=[("Binary image", "*.bin"), ("All files", "*.*")],
    )
    root.destroy()
    if not bin_file:
        print("No file selected — exiting.")
        return

    header_be, image = load_raw(bin_file)
    filename = os.path.basename(bin_file)
    metadata = parse_header(header_be)

    print(f"File        : {filename}")
    print(f"Shape       : {image.shape[0]} rows × {image.shape[1]} cols")
    print(f"Pixel range : {image.min()} – {image.max()}  ADU")
    print(f"Mean ± std  : {image.mean():.1f} ± {image.std():.1f}  ADU")
    print(f"UTC         : {metadata['utc_timestamp']}")
    print(f"Exp time    : {metadata['exp_time']} cs = {metadata['exp_time']/100:.2f} s")
    print(f"CCD temp    : {metadata['ccd_temp1']} °C")
    print(f"Image type  : {metadata['img_type']}")

    # ── Extract ROI centred on user-defined fringe centre ──────────────────
    roi = extract_roi(image, FRINGE_CENTER, ROI_HALF)
    roi_side = ROI_HALF * 2
    roi_cx = roi.shape[1] / 2.0 - 0.5
    roi_cy = roi.shape[0] / 2.0 - 0.5
    print(f"Fringe centre     : row {FRINGE_CENTER[0]}, col {FRINGE_CENTER[1]}")
    print(f"ROI shape         : {roi.shape[0]} rows × {roi.shape[1]} cols "
          f"(requested {roi_side}×{roi_side})")
    print(f"ROI centre pixel  : cx = {roi_cx:.1f}, cy = {roi_cy:.1f}  (pixel coords)")

    # ── Figure 1: image + histogram (2×2 if MASK_DARK) ────────────────────
    if MASK_DARK:
        roi_title = (
            f"ROI  {roi.shape[0]}×{roi.shape[1]} px  "
            f"centred at ({FRINGE_CENTER[0]}, {FRINGE_CENTER[1]})"
        )
        fig1, axes = plt.subplots(2, 2, figsize=(14, 10))
        _plot_image(axes[0, 0], fig1, image, "Unmasked (full frame)")
        _plot_image(axes[0, 1], fig1, roi,   roi_title)

        # ── Overlays on full-frame panel ───────────────────────────────────
        ax0 = axes[0, 0]
        cr, cc = FRINGE_CENTER

        # Crosshair 1 — full-span guide lines (cyan dashed)
        ax0.axhline(cr, color="cyan", linewidth=0.8, linestyle="--", alpha=0.9)
        ax0.axvline(cc, color="cyan", linewidth=0.8, linestyle="--", alpha=0.9)

        # Crosshair 2 — short-arm tick marks at the user-defined fringe centre
        # (yellow solid lines, ±15 px arms so they stay visible at any zoom)
        _ARM = 15
        ax0.plot([cc - _ARM, cc + _ARM], [cr, cr],
                 color="yellow", linewidth=1.5, linestyle="-", alpha=1.0)
        ax0.plot([cc, cc], [cr - _ARM, cr + _ARM],
                 color="yellow", linewidth=1.5, linestyle="-", alpha=1.0)

        # Red rectangle marking the ROI extent
        r_lo = max(0, cr - ROI_HALF)
        c_lo = max(0, cc - ROI_HALF)
        r_hi = min(image.shape[0], cr + ROI_HALF)
        c_hi = min(image.shape[1], cc + ROI_HALF)
        rect = mpatches.Rectangle(
            (c_lo - 0.5, r_lo - 0.5),          # (x, y) — col then row
            c_hi - c_lo, r_hi - r_lo,
            linewidth=1.2, edgecolor="red", facecolor="none",
        )
        ax0.add_patch(rect)

        _plot_hist( axes[1, 0], image, "Pixel Distribution — Unmasked")
        _plot_hist( axes[1, 1], roi,   "Pixel Distribution — ROI")
    else:
        fig1, axes = plt.subplots(2, 1, figsize=(7, 10))
        _plot_image(axes[0], fig1, image, "Unmasked")
        _plot_hist( axes[1], image, "Pixel Distribution — Unmasked")

    fig1.suptitle(f"WindCube FPI — {filename}", fontsize=12, fontweight="bold")
    fig1.tight_layout()
 
    # ── Figure 2: metadata table ───────────────────────────────────────────
    build_metadata_figure(metadata, filename)

    plt.show()

    # ── Save ROI as numpy array alongside the source binary ────────────────
    src = pathlib.Path(bin_file)
    roi_path = src.with_name(src.stem.replace("_L0", "") + "_L1.1.npy")
    np.save(roi_path, roi)
    print(f"ROI saved : {roi_path}")
    print(f"  shape   : {roi.shape}  dtype: {roi.dtype}")
    print(f"  range   : {roi.min()} – {roi.max()}  ADU")


if __name__ == "__main__":
    main()
