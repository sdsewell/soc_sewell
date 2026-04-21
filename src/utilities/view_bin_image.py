"""
view_bin_image.py
-----------------
Utility: open and inspect a WindCube FPI binary image file.

Prompts the user to select a .bin file via a native file-browser dialog,
then produces a three-panel figure suitable for printing on A4/Letter
landscape (11 × 8.5 in):

  Top-left  — raw CCD pixel image  (ADU, fixed range 0–16383)
  Top-right — histogram of ADU values  (same fixed range)
  Bottom    — formatted metadata table  (17 fields, 2-column layout)

The figure is saved as a PNG alongside the chosen .bin file and also
displayed interactively (if a display is available).

Usage:
    python utilities/view_bin_image.py
"""

import os
import sys
import pathlib
import struct

# ── conda DLL path fix (Windows / VS Code) ──────────────────────────────────
if sys.platform == "win32":
    _py = pathlib.Path(sys.executable).parent
    _conda_dirs = [
        str(_py),
        str(_py / "Library" / "bin"),
        str(_py / "Library" / "mingw-w64" / "bin"),
        str(_py / "Library" / "usr" / "bin"),
        str(_py / "Scripts"),
    ]
    _existing = os.environ.get("PATH", "").split(os.pathsep)
    _new = [d for d in _conda_dirs if d not in _existing and pathlib.Path(d).is_dir()]
    if _new:
        os.environ["PATH"] = os.pathsep.join(_new) + os.pathsep + os.environ.get("PATH", "")

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tkinter as tk
from tkinter import filedialog

# ── repo root on sys.path ────────────────────────────────────────────────────
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.metadata.p01_image_metadata_2026_04_06 import parse_header

# ── Constants ────────────────────────────────────────────────────────────────
ADU_MIN     = 0
ADU_MAX     = 16383   # 2^14 − 1
N_HDR_WORDS = 276     # header is always 276 words (zero-padded to n_cols_frame)

# Layout parameters keyed by total file size in bytes.
# Header occupies the first row (276 words, zero-padded to n_cols_frame).
# Science region is embedded in the pixel block at [row_off:row_off+sci_rows, col_off:col_off+sci_cols].
_LAYOUTS = {
    260 * 276 * 2: dict(   # 143 520 B — 2×2 binned
        binning    = 2,
        n_rows     = 260,
        n_cols     = 276,
        sci_row_off = 1,
        sci_col_off = 10,
        sci_rows   = 256,
        sci_cols   = 256,
    ),
    528 * 552 * 2: dict(   # 582 912 B — 1×1 unbinned
        binning    = 1,
        n_rows     = 528,
        n_cols     = 552,
        sci_row_off = 2,
        sci_col_off = 20,
        sci_rows   = 512,
        sci_cols   = 512,
    ),
}


# ── File picker ──────────────────────────────────────────────────────────────

def _pick_file() -> pathlib.Path:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Select a WindCube FPI binary image file",
        filetypes=[("Binary image", "*.bin"), ("All files", "*.*")],
    )
    root.destroy()
    if not path:
        sys.exit("No file selected — exiting.")
    return pathlib.Path(path)


# ── Binary reader ────────────────────────────────────────────────────────────

def _load_bin(path: pathlib.Path):
    """Return (layout, meta_dict, pixel_array) auto-detecting 1×1 or 2×2 binning."""
    raw_bytes = path.read_bytes()
    layout = _LAYOUTS.get(len(raw_bytes))
    if layout is None:
        valid = ", ".join(f"{b:,} B" for b in _LAYOUTS)
        raise ValueError(
            f"Unexpected file size {len(raw_bytes):,} bytes "
            f"(expected one of: {valid}).  Not a valid WindCube .bin file?"
        )
    raw    = np.frombuffer(raw_bytes, dtype=">u2")
    # Header: first 276 words of row 0 (rest of row is zero-padding for 1×1)
    header = raw[:N_HDR_WORDS]
    pixels = raw[layout["n_cols"] :].reshape(layout["n_rows"] - 1, layout["n_cols"])
    meta   = parse_header(header)
    return layout, meta, pixels


# ── Metadata → display rows ──────────────────────────────────────────────────

def _fmt_quat(q):
    return "[{:.4f},  {:.4f},  {:.4f},  {:.4f}]".format(*q)

def _fmt_list(lst, fmt="{:.3f}"):
    return "[" + ",  ".join(fmt.format(v) for v in lst) + "]"


def _build_table_rows(meta: dict, filename: str) -> list:
    """
    Return a list of (label, value) tuples — 17 entries arranged so that
    the caller can split them evenly across two table columns.
    """
    sc_lat_deg = float(np.degrees(meta["spacecraft_latitude"]))
    sc_lon_deg = float(np.degrees(meta["spacecraft_longitude"]))
    sc_alt_km  = meta["spacecraft_altitude"] / 1e3

    exp_s      = meta["exp_time"] / 100.0   # centiseconds → seconds

    et = meta["etalon_temps"]
    et_str = "{:.2f} / {:.2f} / {:.2f} / {:.2f}  °C".format(*et)

    lamp_str = "Lamp 1: {}   Lamp 2: {}   Lamp 3: {}".format(
        meta["lamp1_status"], meta["lamp2_status"], meta["lamp3_status"]
    )

    gpio_str = _fmt_list(meta["gpio_pwr_on"], fmt="{:d}")
    lamp_ch  = _fmt_list(meta["lamp_ch_array"], fmt="{:d}")

    return [
        # ── Left column (rows 0–8) ──
        ("File",               filename),
        ("Image type",         meta["img_type"].upper()),
        ("UTC timestamp",      meta["utc_timestamp"]),
        ("LUA timestamp",      f"{meta['lua_timestamp']}  ms"),
        ("ADCS timestamp",     f"{meta['adcs_timestamp']}  ms"),
        ("Image size",         f"{meta['rows']} rows × {meta['cols']} cols"),
        ("Exposure time",      f"{meta['exp_time']}  cs  ({exp_s:.3f} s)"),
        ("Binning",            f"{meta['binning']}×{meta['binning']}"),
        ("Shutter",            meta["shutter_status"].upper()),
        # ── Right column (rows 0–7) ──
        ("S/C latitude",       f"{sc_lat_deg:+.4f}  °  (negative = south)"),
        ("S/C longitude",      f"{sc_lon_deg:+.4f}  °  (negative = west of prime meridian)"),
        ("S/C altitude",       f"{sc_alt_km:.3f}  km"),
        ("CCD temperature",    f"{meta['ccd_temp1']:.2f}  °C"),
        ("Etalon temps T0–T3", et_str),
        ("Lamp status",        lamp_str),
        ("Attitude quat [xyzw]",  _fmt_quat(meta["attitude_quaternion"])),
        ("Pointing err [xyzw]",   _fmt_quat(meta["pointing_error"])),
        ("GPIO power state",      gpio_str),
    ]


# ── Figure builder ───────────────────────────────────────────────────────────

def _build_figure(path: pathlib.Path, meta: dict, pixels: np.ndarray,
                  layout: dict):
    """
    Construct the three-panel figure:
      [image | histogram]
      [     metadata table     ]
    """
    r0  = layout["sci_row_off"]
    c0  = layout["sci_col_off"]
    nr  = layout["sci_rows"]
    nc  = layout["sci_cols"]
    sci = pixels[r0 : r0 + nr, c0 : c0 + nc]

    # ── Layout ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 8.5))   # landscape Letter / A4
    gs  = gridspec.GridSpec(
        2, 2,
        figure     = fig,
        height_ratios = [1.0, 0.95],
        hspace     = 0.38,
        wspace     = 0.30,
        left       = 0.06,
        right      = 0.97,
        top        = 0.93,
        bottom     = 0.03,
    )

    ax_img  = fig.add_subplot(gs[0, 0])
    ax_hist = fig.add_subplot(gs[0, 1])
    ax_tbl  = fig.add_subplot(gs[1, :])
    ax_tbl.axis("off")

    img_type_str = meta["img_type"].upper()
    fig.suptitle(
        f"WindCube FPI Binary Image Viewer  —  {path.name}  [{img_type_str}]",
        fontsize=11, fontweight="bold", y=0.975,
    )

    # ── Top-left: raw image ──────────────────────────────────────────────────
    im = ax_img.imshow(
        sci,
        cmap   = "gray",
        vmin   = ADU_MIN,
        vmax   = ADU_MAX,
        origin = "upper",
        aspect = "equal",
        interpolation = "nearest",
    )
    plt.colorbar(im, ax=ax_img, label="ADU", fraction=0.046, pad=0.04)
    ax_img.set_title(
        f"Raw CCD Image  ({layout['sci_rows']} × {layout['sci_cols']} science region,"
        f"  {layout['binning']}×{layout['binning']} binning)", fontsize=9)
    ax_img.set_xlabel("Column (px)", fontsize=8)
    ax_img.set_ylabel("Row (px)", fontsize=8)
    ax_img.tick_params(labelsize=7)

    # ── Top-right: histogram ─────────────────────────────────────────────────
    ax_hist.hist(
        sci.ravel(),
        bins   = 256,
        range  = (ADU_MIN, ADU_MAX),
        color  = "#2166ac",
        alpha  = 0.8,
        log    = False,
        rwidth = 0.9,
    )
    ax_hist.set_xlim(ADU_MIN, ADU_MAX)
    ax_hist.set_title("ADU Histogram", fontsize=9)
    ax_hist.set_xlabel("ADU  (0 – 16 383)", fontsize=8)
    ax_hist.set_ylabel("Pixel count", fontsize=8)
    ax_hist.tick_params(labelsize=7)
    ax_hist.grid(True, axis="y", linewidth=0.4, alpha=0.6)

    _stats = (
        f"min={sci.min()}   max={sci.max()}\n"
        f"mean={sci.mean():.1f}   std={sci.std():.1f}"
    )
    ax_hist.text(
        0.97, 0.95, _stats,
        transform=ax_hist.transAxes,
        ha="right", va="top",
        fontsize=7, family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", lw=0.6, alpha=0.9),
    )

    # ── Bottom: metadata table ────────────────────────────────────────────────
    rows = _build_table_rows(meta, path.name)

    # Split into two equal-ish halves for a side-by-side layout
    mid        = (len(rows) + 1) // 2    # left col takes the larger half
    left_rows  = rows[:mid]
    right_rows = rows[mid:]

    # Pad the shorter column with blank entries
    while len(right_rows) < len(left_rows):
        right_rows.append(("", ""))

    n_rows = len(left_rows)

    # Interleave: [left_label, left_value, right_label, right_value]
    cell_text  = []
    col_labels = ["Parameter", "Value", "Parameter", "Value"]
    for i in range(n_rows):
        cell_text.append([
            left_rows[i][0],  left_rows[i][1],
            right_rows[i][0], right_rows[i][1],
        ])

    tbl = ax_tbl.table(
        cellText  = cell_text,
        colLabels = col_labels,
        colWidths = [0.13, 0.37, 0.13, 0.37],
        loc       = "center",
        cellLoc   = "left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1.0, 1.55)   # row height

    # Style: header row — dark background, white text
    _HEADER_BG  = "#1f4e79"
    _HEADER_FG  = "white"
    _LABEL_BG   = "#dce6f1"
    _VALUE_BG   = "white"
    _ALT_BG     = "#eef3f9"    # alternating row shade for label cells
    _DIVIDER_BG = "#c9d9ea"    # vertical divider between the two halves

    for (row_idx, col_idx), cell in tbl.get_celld().items():
        cell.set_edgecolor("0.75")
        cell.set_linewidth(0.5)
        if row_idx == 0:
            # Header row
            cell.set_facecolor(_HEADER_BG)
            cell.get_text().set_color(_HEADER_FG)
            cell.get_text().set_fontweight("bold")
        else:
            data_row = row_idx - 1
            if col_idx in (0, 2):
                # Label cells — alternating shade
                cell.set_facecolor(_ALT_BG if data_row % 2 == 0 else _LABEL_BG)
                cell.get_text().set_fontweight("semibold")
            else:
                # Value cells
                cell.set_facecolor(_VALUE_BG)
                cell.get_text().set_family("monospace")

    # Make the centre divider column slightly darker
    for (row_idx, col_idx), cell in tbl.get_celld().items():
        if col_idx == 2 and row_idx > 0:
            cell.set_facecolor(
                _ALT_BG if (row_idx - 1) % 2 == 0 else _LABEL_BG
            )

    ax_tbl.set_title(
        "Image Metadata  (decoded from binary header)",
        fontsize=9, pad=6,
    )

    return fig


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("WindCube FPI Binary Image Viewer")
    print("Select a .bin file in the dialog (check taskbar if not visible)...")

    path = _pick_file()
    print(f"  Loading: {path}")

    layout, meta, pixels = _load_bin(path)

    print(f"  Binning     : {layout['binning']}×{layout['binning']}")
    print(f"  Image type  : {meta['img_type'].upper()}")
    print(f"  UTC         : {meta['utc_timestamp']}")
    print(f"  Size        : {meta['rows']} × {meta['cols']}")

    fig = _build_figure(path, meta, pixels, layout)

    out_path = path.parent.parent / (path.stem + ".png")
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"  Saved PNG   : {out_path}")

    plt.show()


if __name__ == "__main__":
    main()
