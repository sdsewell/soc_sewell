"""
run_windcube_calibration.py
===========================
WindCube FPI neon calibration pipeline.

Workflow
--------
1.  File selection  — Windows dialog boxes to select one cal .bin and one
                      dark .bin file.
2.  Image loading   — load_raw() decodes the big-endian binary header and
                      pixel data; dark-subtracted image is cast to float64.
3.  Figure 1        — 2×2 subplot grid:
      [0,0] Dark-subtracted cal image (grey, with centre cross)
      [0,1] ADU histogram, x-axis fixed 0 – 16383
      [1,0] Coarse grid-search cost map with best-grid point marked
      [1,1] Fine Nelder-Mead result: fringe image with recovered centre
4.  Annular reduction — r²-binned 1-D profile; parabola-primary peak fits
                        (Gaussian fallback) using _R2_WINDOWS below.
5.  Peak table      — matplotlib table figure listing line ID, r²_fit,
                      ±σ, χ²_red for every identified peak.
6.  Figure 3        — 4-column × ⌈N/4⌉-row subplot showing fit window and
                      parabola fit for every peak (from _plot_all_fringe_
                      diagnostics_r2).
7.  Tolansky        — two-line analysis (640.2248 nm + 638.2991 nm):
                      recovers d, α, ε_a, ε_b, Δ_a, Δ_b; prints the
                      Vaughan rectangular-array table; shows four-panel
                      diagnostic figure.

Configuration
-------------
All quantities the user is likely to tune are collected in the
"USER-TUNABLE PARAMETERS" block immediately below the imports.  Edit
_R2_WINDOWS to shift fitting windows; adjust CENTRE_SEED, R_MAX_PX, etc.
as needed.

Libraries used (all from fpi_cal_lib or standard scientific Python)
---------------------------------------------------------------------
  fpi_cal_lib :  find_centre, annular_reduce, peaks_to_array,
                 run_tolansky_2line, plot_tolansky_result,
                 print_rectangular_array, _plot_all_fringe_diagnostics_r2,
                 FringeProfile, PeakFitR2, _R2_WINDOWS, _variance_cost
  numpy, matplotlib, scipy (via fpi_cal_lib), tkinter (dialog only)

Note: No single-line Tolansky analysis is performed; only the two-line
analysis (640.2248 nm + 638.2991 nm) is executed as per WindCube S13a spec.
"""

from __future__ import annotations

import os
import sys
import struct
import pathlib
import warnings
import tkinter as tk
from tkinter import filedialog

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# fpi_cal_lib imports
# All physics lives in the library; this script is UX + orchestration only.
#
# Path resolution — finds the repo root (soc_sewell/) automatically by
# searching upward for the 'windcube' sub-package, so this script works
# regardless of where inside the repo it is placed.
#
# Two sys.path entries are required:
#   soc_sewell/windcube/  → 'import fpi_cal_lib'          (bare module)
#   soc_sewell/           → 'from windcube.constants ...' (package import,
#                            used internally by fpi_cal_lib)
# ---------------------------------------------------------------------------
# fpi_cal_lib.py lives at  soc_sewell/src/fpi/fpi_cal_lib.py
# This script may live anywhere in the repo; we locate fpi_cal_lib.py
# by searching upward AND sideways (src/fpi/) from the script's directory.
_THIS_FILE = pathlib.Path(__file__).resolve()

def _find_fpi_cal_lib(start: pathlib.Path) -> pathlib.Path:
    """Return the directory containing fpi_cal_lib.py.
    Searches upward from *start*, and for each ancestor also checks
    the src/fpi/ sub-path, which is the authoritative location."""
    for parent in [start, *start.parents]:
        # Direct hit (script is in the same folder as the lib)
        if (parent / 'fpi_cal_lib.py').exists():
            return parent
        # Standard layout: soc_sewell/src/fpi/fpi_cal_lib.py
        candidate = parent / 'src' / 'fpi' / 'fpi_cal_lib.py'
        if candidate.exists():
            return candidate.parent
    raise FileNotFoundError(
        "Could not locate fpi_cal_lib.py.\n"
        f"Searched upward from: {start}\n"
        "Expected at soc_sewell/src/fpi/fpi_cal_lib.py"
    )

_FPI_DIR = _find_fpi_cal_lib(_THIS_FILE.parent)   # soc_sewell/src/fpi/
# fpi_cal_lib itself does 'from windcube.constants import ...'
# so the repo root (soc_sewell/) must also be on sys.path so that
# the windcube package (soc_sewell/windcube/__init__.py) is importable.
_REPO_ROOT = _FPI_DIR.parents[1]                  # soc_sewell/src/fpi -> soc_sewell/
for _p in (_FPI_DIR, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from fpi_cal_lib import (
    # Section A — Centre finder
    find_centre,
    _variance_cost,
    CentreResult,
    # Section C — Annular reduction
    annular_reduce,
    FringeProfile,
    PeakFitR2,
    _R2_WINDOWS,
    _plot_all_fringe_diagnostics_r2,
    _print_peak_table_r2,
    peaks_to_array,
    # Section D — Tolansky
    run_tolansky_2line,
    plot_tolansky_result,
    print_rectangular_array,
)


# ===========================================================================
# USER-TUNABLE PARAMETERS
# ===========================================================================
# Edit these values to adjust centre seed, fitting geometry, and peak windows.
# All _R2_WINDOWS entries below override the auto-detected windows inside
# _find_and_fit_peaks_r2 on a per-peak basis (peak index 0, 1, 2, ...).
# Peaks are ordered by r² ascending; even indices → 640.2 nm (strong),
# odd indices → 638.3 nm (weak).

# ── Centre finding ──────────────────────────────────────────────────────────
# Seed in (row, col) image coordinates.  find_centre() expects (cx, cy)
# where cx = col, cy = row — the conversion is handled automatically below.
CENTRE_SEED_ROW = 145      # row coordinate of fringe centre seed (ROI box centre)
CENTRE_SEED_COL = 144      # col coordinate of fringe centre seed (ROI box centre)

VAR_R_MIN_PX    =  5.0     # inner exclusion radius for cost computation (px)
VAR_R_MAX_PX    = None     # None → min(H,W)//2 - 10; or set e.g. 110.0
VAR_N_BINS      = 250      # r² bins used in azimuthal variance cost
VAR_SEARCH_PX   = 15.0     # ±half-width of coarse grid search (px)
# NOTE: centre finding uses the geometric centre of the ROI as its
# coarse seed, NOT CENTRE_SEED_ROW/COL.  CENTRE_SEED_ROW/COL only
# controls where the ROI box is placed in Figure 0.

# ── Annular reduction ────────────────────────────────────────────────────────
R_MAX_PX        = 150.0    # outer radius for annular integration (px); extended to use partial annuli
N_BINS          = 1500     # number of r² bins  (→ ~43 bins/FWHM)
PEAK_PROMINENCE = 20.0     # minimum peak prominence in ADU  [lowered for weak outer fringes]

# ── Peak fitting windows in r²  (px²) ───────────────────────────────────────
# These override the auto-detected windows peak-by-peak.
# Format:  { peak_index: (r2_lo, r2_hi), ... }
# Copy _R2_WINDOWS from fpi_cal_lib as the starting point; edit freely.
# Peak index 0 = innermost; even = 640.2 nm, odd = 638.3 nm.
_R2_WINDOWS.update({
    #  key: (lo, centre, hi)  -- 3-tuple: lo/hi = fitting window bounds,
    #                             centre  = initial guess for parabolic fit
    #  P0-P19: centre = r²_fit from last run (authoritative)
    #  P20+:   centre = predicted from Δ spacing (adjust after inspecting Figure 2)
     0: (     205,  330,    455),   # P0  640.2 nm
     1: (     797,  947,   1097),   # P1  638.3 nm
     2: (    1415, 1561,   1715),   # P2  640.2 nm
     3: (    1973, 2173,   2373),   # P3  638.3 nm
     4: (    2600, 2797,   3000),   # P4  640.2 nm
     5: (    3243, 3395,   3543),   # P5  638.3 nm
     6: (    3827, 4019,   4227),   # P6  640.2 nm
     7: (    4474, 4670,   4784),   # P7  638.3 nm  [right +10]
     8: (    5107, 5250,   5407),   # P8  640.2 nm
     9: (    5649, 5850,   6049),   # P9  638.3 nm
    10: (    6284, 6485,   6684),   # P10 640.2 nm
    11: (    6889, 7085,   7289),   # P11 638.3 nm
    12: (    7517, 7712,   7917),   # P12 640.2 nm
    13: (    8157, 8312,   8457),   # P13 638.3 nm
    14: (    8703, 8955,   9203),   # P14 640.2 nm
    15: (    9361, 9529,   9661),   # P15 638.3 nm
    16: (    9984,10191,  10384),   # P16 640.2 nm
    17: (   10616,10757,  10916),   # P17 638.3 nm
    18: (   11226,11420,  11626),   # P18 640.2 nm
    19: (   11839,11988,  12139),   # P19 638.3 nm
    # ── Extended outer fringes — centres are predictions, adjust as needed ────
    20: (   12427,12652,  12877),   # P20 640.2 nm  predicted
    21: (   12990,13215,  13440),   # P21 638.3 nm  predicted
    22: (   13659,13884,  14109),   # P22 640.2 nm  predicted
    23: (   14175,14400,  14625),   # P23 638.3 nm  observed≈14400
    24: (   14975,15200,  15425),   # P24 640.2 nm  observed≈15200
    25: (   15475,15700,  15925),   # P25 638.3 nm  observed≈15700
    26: (   16075,16300,  16525),   # P26 640.2 nm  observed≈16300
    27: (   16675,16900,  17125),   # P27 638.3 nm  observed≈16900
    28: (   17375,17600,  17825),   # P28 640.2 nm  observed≈17600
    29: (   17898,18123,  18348),   # P29 638.3 nm  predicted
    30: (   18587,18812,  19037),   # P30 640.2 nm  predicted
    31: (   19125,19350,  19575),   # P31 638.3 nm  predicted
    32: (   19819,20044,  20269),   # P32 640.2 nm  predicted
    33: (   20352,20577,  20802),   # P33 638.3 nm  predicted
})

# ── Tolansky two-line analysis ───────────────────────────────────────────────
LAM_A_M     = 640.2248e-9  # Ne strong line wavelength (Burns 1950, IAU 'S')
LAM_B_M     = 638.2991e-9  # Ne weak   line wavelength (Burns 1950, IAU 'S')
D_PRIOR_M   = 20.008e-3    # ICOS nominal gap — used ONLY to resolve N_Delta


# ===========================================================================
# END USER-TUNABLE PARAMETERS
# ===========================================================================


# ---------------------------------------------------------------------------
# Binary loader  (adapted from load_image_swapped.py)
# ---------------------------------------------------------------------------

_KNOWN_FRAME_SIZES = [(260, 276), (528, 552)]


def _load_raw(path: str):
    """
    Load a big-endian WindCube FPI binary.

    Returns
    -------
    header : ndarray (n_cols,) uint16 — decoded header row
    image  : ndarray (n_rows-1, n_cols) uint16 — pixel data
    """
    with open(path, "rb") as f:
        first_words = np.frombuffer(f.read(4), dtype=">u2")
    n_rows_frame = int(first_words[0])
    n_cols_frame = int(first_words[1])
    actual   = os.path.getsize(path)
    expected = n_rows_frame * n_cols_frame * 2
    if actual != expected:
        for rows, cols in _KNOWN_FRAME_SIZES:
            if rows * cols * 2 == actual:
                print(
                    f"  WARNING: header says {n_rows_frame}×{n_cols_frame} "
                    f"but file is {actual} B — using known size {rows}×{cols}."
                )
                n_rows_frame, n_cols_frame = rows, cols
                break
        else:
            raise ValueError(
                f"File size mismatch: got {actual} B, "
                f"expected {expected} for {n_rows_frame}×{n_cols_frame} uint16."
            )
    raw = np.frombuffer(open(path, "rb").read(), dtype=">u2")
    header = raw[:n_cols_frame].copy()
    image  = raw[n_cols_frame:].reshape(n_rows_frame - 1, n_cols_frame).copy()
    return header, image


def _f64(h: np.ndarray, w: int) -> float:
    """Mixed-endian float64: 4 BE uint16 words in LE word order."""
    b = struct.pack(">4H", *reversed([h[w + i] for i in range(4)]))
    return struct.unpack(">d", b)[0]


def _u64(h: np.ndarray, w: int) -> int:
    """Mixed-endian uint64: 4 BE uint16 words in LE word order."""
    return sum(int(h[w + i]) << (16 * i) for i in range(4))


def _exp_time_s(h: np.ndarray) -> float:
    """Exposure time in seconds (word 2 = ms)."""
    return float(int(h[2])) * 1e-3


def _image_type_str(h: np.ndarray) -> str:
    """Quick classification from lamp and shutter bits."""
    gpio  = [int(h[100 + i]) & 0xFF for i in range(4)]
    lamps = [int(h[104 + i]) & 0xFF for i in range(6)]
    shutter = "closed" if (gpio[0] == 1 and gpio[3] == 1) else "open"
    if any(lamps):
        return "cal"
    if shutter == "closed":
        return "dark"
    return "science"


# ---------------------------------------------------------------------------
# Full header decoder (subset of load_image_swapped.py parse_header)
# ---------------------------------------------------------------------------

def _u8arr(h: np.ndarray, w: int, n: int) -> list:
    return [int(h[w + i]) & 0xFF for i in range(n)]


_FIELD_META = [
    # (key,             display_name,             unit,       fmt_fn)
    ('rows',            'Rows',                   'pixels',   None),
    ('cols',            'Cols',                   'pixels',   None),
    ('exp_time_ms',     'Exposure time',           'ms',       lambda v: f'{v} ms  ({v/1000:.2f} s)'),
    ('ccd_temp1',       'CCD temperature',         'deg C',    lambda v: f'{v:.4f}'),
    ('utc_timestamp',   'UTC timestamp',           '',         None),
    ('img_type',        'Image type',              '',         None),
    ('shutter_status',  'Shutter',                 '',         None),
    ('lamp1_status',    'Lamp 1',                  '',         None),
    ('lamp2_status',    'Lamp 2',                  '',         None),
    ('lamp3_status',    'Lamp 3',                  '',         None),
    ('sc_lat_deg',      'SC latitude',             'deg',      lambda v: f'{v:.4f}'),
    ('sc_lon_deg',      'SC longitude',            'deg',      lambda v: f'{v:.4f}'),
    ('sc_alt_km',       'SC altitude',             'km',       lambda v: f'{v:.2f}'),
    ('etalon_temps',    'Etalon temps',            'deg C',    lambda v: '[' + ', '.join(f'{x:.2f}' for x in v) + ']'),
    ('lua_timestamp',   'Lua timestamp',           'ms Unix',  None),
]


def _parse_header_full(h: np.ndarray) -> dict:
    """Decode header row into the dict expected by _FIELD_META."""
    import struct as _struct
    from datetime import datetime as _dt, timezone as _tz

    def _f64_local(w):
        b = _struct.pack('>4H', *reversed([h[w + i] for i in range(4)]))
        return _struct.unpack('>d', b)[0]

    def _u64_local(w):
        return sum(int(h[w + i]) << (16 * i) for i in range(4))

    gpio  = _u8arr(h, 100, 4)
    lamps = _u8arr(h, 104, 6)
    shutter = 'closed' if (gpio[0] == 1 and gpio[3] == 1) else 'open'
    any_lamp = any(lamps)
    img_type = 'cal' if any_lamp else ('dark' if shutter == 'closed' else 'science')
    lua_ms = _u64_local(8)
    try:
        utc = _dt.fromtimestamp(lua_ms / 1000.0, tz=_tz.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    except Exception:
        utc = 'invalid'
    return {
        'rows':           int(h[0]),
        'cols':           int(h[1]),
        'exp_time_ms':    int(h[2]),
        'ccd_temp1':      round(_f64_local(4), 4),
        'utc_timestamp':  utc,
        'img_type':       img_type,
        'shutter_status': shutter,
        'lamp1_status':   'on' if lamps[0] else 'off',
        'lamp2_status':   'on' if lamps[1] else 'off',
        'lamp3_status':   'on' if lamps[2] else 'off',
        'sc_lat_deg':     float(np.degrees(_f64_local(16))),
        'sc_lon_deg':     float(np.degrees(_f64_local(20))),
        'sc_alt_km':      _f64_local(24) / 1000.0,
        'etalon_temps':   [_f64_local(84 + i * 4) for i in range(4)],
        'lua_timestamp':  lua_ms,
    }


def _fmt_meta_value(key: str, val) -> str:
    """Format a metadata value for table display."""
    for k, _, _, fn in _FIELD_META:
        if k == key:
            if fn is not None:
                return fn(val)
            break
    if isinstance(val, float):
        return f'{val:.6g}'
    if isinstance(val, list):
        return '[' + ', '.join(f'{v:.4g}' if isinstance(v, float) else str(v) for v in val) + ']'
    return str(val)


# ---------------------------------------------------------------------------
# Windows file-dialog helpers
# ---------------------------------------------------------------------------

def _pick_file(title: str, initialdir: str | None = None) -> str:
    """Open a Windows dialog box and return the selected path (or '')."""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title=title,
        initialdir=initialdir,
        filetypes=[("WindCube binary", "*.bin"), ("All files", "*.*")],
        parent=root,
    )
    root.destroy()
    return path


# ---------------------------------------------------------------------------
# Figure 0 -- raw images, metadata table, and dark-subtracted ROI
#
# Layout  (3 rows x 2 cols, table and ROI span full width):
#   [0,0]  Raw cal image (full frame) with ROI box
#   [0,1]  Raw dark image (full frame) with ROI box
#   [1,:]  Metadata table — cal column | dark column, side by side
#   [2,:]  Dark-subtracted ROI (220x220 centred on coarse seed) — working array
# ---------------------------------------------------------------------------

ROI_HALF = 110   # half-size: ROI will be 220x220 px (2*ROI_HALF x 2*ROI_HALF)


def _extract_roi(image: np.ndarray, row_c: int, col_c: int, half: int) -> tuple:
    """Extract ROI centred at (row_c, col_c); return (roi_array, r_lo, r_hi, c_lo, c_hi)."""
    r_lo = max(0, row_c - half)
    r_hi = min(image.shape[0], row_c + half)
    c_lo = max(0, col_c - half)
    c_hi = min(image.shape[1], col_c + half)
    return image[r_lo:r_hi, c_lo:c_hi], r_lo, r_hi, c_lo, c_hi


def _draw_roi_box(ax, r_lo, r_hi, c_lo, c_hi, color='red', lw=1.5, label='ROI'):
    """Draw a rectangle outline on ax in image (col, row) coordinates."""
    import matplotlib.patches as _mpatches
    rect = _mpatches.Rectangle(
        (c_lo - 0.5, r_lo - 0.5),
        c_hi - c_lo, r_hi - r_lo,
        linewidth=lw, edgecolor=color, facecolor='none', label=label,
    )
    ax.add_patch(rect)


def figure0_raw_images(
    img_cal:    np.ndarray,
    img_dark:   np.ndarray,
    hdr_cal:    np.ndarray,
    hdr_dark:   np.ndarray,
    roi_ds:     np.ndarray,
    roi_bounds: tuple,
    cal_path:   str,
    dark_path:  str,
) -> plt.Figure:
    """
    Figure 0: raw images, metadata table, and dark-subtracted ROI.

    Parameters
    ----------
    roi_ds     : dark-subtracted ROI array (float64), pre-computed by main().
    roi_bounds : (r_lo, r_hi, c_lo, c_hi) in full-frame pixel coordinates.

    Returns
    -------
    fig : matplotlib Figure
    """
    meta_cal  = _parse_header_full(hdr_cal)
    meta_dark = _parse_header_full(hdr_dark)

    r_lo, r_hi, c_lo, c_hi = roi_bounds

    # ── Figure layout: 3 rows, 2 cols; row 1 spans full width; row 2 split ──
    fig = plt.figure(figsize=(16, 22))
    gs  = gridspec.GridSpec(
        3, 2, figure=fig,
        height_ratios=[3.5, 4.5, 3.0],
        hspace=0.30, wspace=0.12,
        left=0.05, right=0.97, top=0.90, bottom=0.03,
    )
    ax_cal   = fig.add_subplot(gs[0, 0])
    ax_dark  = fig.add_subplot(gs[0, 1])
    ax_table = fig.add_subplot(gs[1, :])
    ax_roi   = fig.add_subplot(gs[2, 0])   # dark-subtracted image — left
    ax_hist  = fig.add_subplot(gs[2, 1])   # pixel histogram — right

    vlo_cal  = float(np.percentile(img_cal,  1))
    vhi_cal  = float(np.percentile(img_cal, 99))
    vlo_dark = float(np.percentile(img_dark,  1))
    vhi_dark = float(np.percentile(img_dark, 99))

    # ── [0,0] Raw cal image ─────────────────────────────────────────────────
    im0 = ax_cal.imshow(img_cal, cmap='gray', origin='lower',
                        vmin=vlo_cal, vmax=vhi_cal,
                        aspect='equal', interpolation='none')
    fig.colorbar(im0, ax=ax_cal, fraction=0.046, pad=0.04).set_label('ADU', fontsize=8)
    _draw_roi_box(ax_cal, r_lo, r_hi, c_lo, c_hi, color='red', label=f'ROI {roi_ds.shape[0]}x{roi_ds.shape[1]} px')
    ax_cal.plot(CENTRE_SEED_COL, CENTRE_SEED_ROW, '+', color='yellow', ms=10, mew=2.0)
    ax_cal.legend(fontsize=7, loc='upper right')
    ax_cal.set_title(
        f'Raw cal  |  {img_cal.shape[0]}x{img_cal.shape[1]} px  '
        f'ADU [{img_cal.min()}, {img_cal.max()}]  mean {img_cal.mean():.0f}\n'
        f'{os.path.basename(cal_path)}',
        fontsize=9)
    ax_cal.set_xlabel('Column (pixel)', fontsize=8)
    ax_cal.set_ylabel('Row (pixel)', fontsize=8)
    ax_cal.tick_params(labelsize=7)

    # ── [0,1] Raw dark image ────────────────────────────────────────────────
    im1 = ax_dark.imshow(img_dark, cmap='gray', origin='lower',
                         vmin=vlo_dark, vmax=vhi_dark,
                         aspect='equal', interpolation='none')
    fig.colorbar(im1, ax=ax_dark, fraction=0.046, pad=0.04).set_label('ADU', fontsize=8)
    _draw_roi_box(ax_dark, r_lo, r_hi, c_lo, c_hi, color='red', label=f'ROI {roi_ds.shape[0]}x{roi_ds.shape[1]} px')
    ax_dark.plot(CENTRE_SEED_COL, CENTRE_SEED_ROW, '+', color='yellow', ms=10, mew=2.0)
    ax_dark.legend(fontsize=7, loc='upper right')
    ax_dark.set_title(
        f'Raw dark  |  {img_dark.shape[0]}x{img_dark.shape[1]} px  '
        f'ADU [{img_dark.min()}, {img_dark.max()}]  mean {img_dark.mean():.0f}\n'
        f'{os.path.basename(dark_path)}',
        fontsize=9)
    ax_dark.set_xlabel('Column (pixel)', fontsize=8)
    ax_dark.set_ylabel('Row (pixel)', fontsize=8)
    ax_dark.tick_params(labelsize=7)

    # ── [1,:] Metadata table — cal left, dark right ─────────────────────────
    ax_table.axis('off')

    col_labels = ['Field', 'Unit', 'CAL value', 'DARK value']
    rows = []
    for key, display_name, unit, _ in _FIELD_META:
        val_cal  = meta_cal.get(key,  'n/a')
        val_dark = meta_dark.get(key, 'n/a')
        rows.append([
            display_name,
            unit,
            _fmt_meta_value(key, val_cal)  if val_cal  != 'n/a' else 'n/a',
            _fmt_meta_value(key, val_dark) if val_dark != 'n/a' else 'n/a',
        ])

    n_rows = len(rows)
    tbl = ax_table.table(
        cellText=rows,
        colLabels=col_labels,
        loc='center',
        cellLoc='left',
        colWidths=[0.22, 0.10, 0.32, 0.32],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.55)
    HDR_BG = '#2C3E50'
    for c in range(4):
        tbl[0, c].set_facecolor(HDR_BG)
        tbl[0, c].set_text_props(color='white', fontweight='bold')
        tbl[0, c].set_edgecolor('#888888')
    for r in range(1, n_rows + 1):
        bg = '#EBF5FB' if r % 2 == 0 else 'white'
        for c in range(4):
            tbl[r, c].set_facecolor(bg)
            tbl[r, c].set_edgecolor('#CCCCCC')
    ax_table.set_title(
        'Embedded metadata — decoded from binary header row',
        fontsize=10, fontweight='bold', pad=6)

    # ── [2,0] Dark-subtracted ROI image — left ───────────────────────────────
    vlo_roi = float(np.percentile(roi_ds,  1))
    vhi_roi = float(np.percentile(roi_ds, 99))
    im2 = ax_roi.imshow(
        roi_ds, cmap='gray', origin='lower',
        vmin=vlo_roi, vmax=vhi_roi,
        aspect='equal', interpolation='none',
        extent=[c_lo, c_hi, r_lo, r_hi],   # keep full-frame pixel coords on axes
    )
    fig.colorbar(im2, ax=ax_roi, fraction=0.046, pad=0.04).set_label('ADU (dark-subtracted)', fontsize=8)
    ax_roi.plot(CENTRE_SEED_COL, CENTRE_SEED_ROW, '+', color='yellow', ms=14, mew=2.5,
                label=f'Seed  cx={CENTRE_SEED_COL}, cy={CENTRE_SEED_ROW}')
    ax_roi.axhline(CENTRE_SEED_ROW, color='cyan', lw=0.8, ls='--', alpha=0.7)
    ax_roi.axvline(CENTRE_SEED_COL, color='cyan', lw=0.8, ls='--', alpha=0.7)
    ax_roi.legend(fontsize=8, loc='upper right')
    ax_roi.set_title(
        f'Dark-subtracted ROI  |  {roi_ds.shape[0]}x{roi_ds.shape[1]} px  '
        f'ADU [{roi_ds.min():.0f}, {roi_ds.max():.0f}]  '
        f'mean {roi_ds.mean():.1f}  std {roi_ds.std():.1f}',
        fontsize=9)
    ax_roi.set_xlabel('Column (full-frame pixel)', fontsize=8)
    ax_roi.set_ylabel('Row (full-frame pixel)', fontsize=8)
    ax_roi.tick_params(labelsize=7)

    # ── [2,1] Histogram of dark-subtracted ROI pixels — right ────────────────
    _ds_flat = roi_ds.ravel()
    _ds_p1   = float(np.percentile(_ds_flat,  1))
    _ds_p99  = float(np.percentile(_ds_flat, 99))
    _ds_clip = _ds_flat[(_ds_flat >= _ds_p1) & (_ds_flat <= _ds_p99)]
    ax_hist.hist(_ds_clip, bins=80, color='steelblue', alpha=0.75, edgecolor='none')
    ax_hist.axvline(float(np.mean(_ds_flat)), color='red', lw=1.4,
                    label=f'mean = {np.mean(_ds_flat):.1f} ADU')
    ax_hist.axvline(float(np.median(_ds_flat)), color='orange', lw=1.2, ls='--',
                    label=f'median = {np.median(_ds_flat):.1f} ADU')
    ax_hist.set_xlabel('ADU  (dark-subtracted)', fontsize=8)
    ax_hist.set_ylabel('Pixel count', fontsize=8)
    ax_hist.set_title(
        f'Dark-subtracted ROI histogram  |  '
        f'mean={np.mean(_ds_flat):.1f}  std={np.std(_ds_flat):.1f}  '
        f'[{_ds_flat.min():.0f}, {_ds_flat.max():.0f}] ADU',
        fontsize=9)
    ax_hist.legend(fontsize=8)
    ax_hist.tick_params(labelsize=7)

    fig.suptitle(
        f'WindCube FPI — Raw Images & Metadata\n'
        f'Cal: {os.path.basename(cal_path)}    Dark: {os.path.basename(dark_path)}',
        fontsize=11, fontweight='bold',
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 1 -- two-pass centre finding  (2 rows x 3 columns)
# ---------------------------------------------------------------------------

def _cost_slice_cx(
    image_clipped: np.ndarray,
    cx_vals: np.ndarray,
    cy_fixed: float,
    r_min_sq: float,
    r_max_sq: float,
    n_var_bins: int,
) -> np.ndarray:
    """Evaluate variance cost along a cx slice with cy fixed."""
    return np.array([
        _variance_cost(cx, cy_fixed, image_clipped, r_min_sq, r_max_sq, n_var_bins)
        for cx in cx_vals
    ])


def _cost_slice_cy(
    image_clipped: np.ndarray,
    cy_vals: np.ndarray,
    cx_fixed: float,
    r_min_sq: float,
    r_max_sq: float,
    n_var_bins: int,
) -> np.ndarray:
    """Evaluate variance cost along a cy slice with cx fixed."""
    return np.array([
        _variance_cost(cx_fixed, cy, image_clipped, r_min_sq, r_max_sq, n_var_bins)
        for cy in cy_vals
    ])


def figure1_centre_finding(
    cal_ds: np.ndarray,
    centre: CentreResult,
    cx_seed: float,
    cy_seed: float,
    cal_path: str,
    dark_path: str,
    var_r_max_px: float,
) -> plt.Figure:
    """
    2-row x 3-column centre-finding figure.

    cx_seed, cy_seed : coarse seed used for the Pass 1 grid slices
                       (should be the ROI geometric centre, in ROI-local coords).

    Row 0 -- Pass 1 coarse grid:
      col 0: fringe image, green crosshair at grid-best centre
      col 1: variance vs cx (20 grid points), cy fixed at seed
      col 2: variance vs cy (20 grid points), cx fixed at seed

    Row 1 -- Pass 2 Nelder-Mead:
      col 0: fringe image, cyan dashed crosshair + yellow + at NM centre
      col 1: variance vs cx (dense, +-5 px window), cy fixed at NM result
      col 2: variance vs cy (dense, +-5 px window), cx fixed at NM result
    """
    H, W     = cal_ds.shape

    # Pre-clip image (same as find_centre)
    p99_5    = float(np.percentile(cal_ds, 99.5))
    img_c    = np.clip(cal_ds, None, p99_5)
    r_min_sq = VAR_R_MIN_PX ** 2
    r_max_sq = var_r_max_px ** 2

    # -- Coarse grid: 20 points along each axis, other axis fixed at seed ----
    offsets        = np.linspace(-VAR_SEARCH_PX, VAR_SEARCH_PX, 20)
    cx_grid        = cx_seed + offsets
    cy_grid        = cy_seed + offsets
    cost_cx_coarse = _cost_slice_cx(img_c, cx_grid,  cy_seed,   r_min_sq, r_max_sq, VAR_N_BINS)
    cost_cy_coarse = _cost_slice_cy(img_c, cy_grid,  cx_seed,   r_min_sq, r_max_sq, VAR_N_BINS)

    # -- Fine NM window: 200 dense points +/-5 px around NM result ----------
    NM_WIN       = 5.0
    N_FINE       = 200
    cx_fine      = np.linspace(centre.cx - NM_WIN, centre.cx + NM_WIN, N_FINE)
    cy_fine      = np.linspace(centre.cy - NM_WIN, centre.cy + NM_WIN, N_FINE)
    cost_cx_fine = _cost_slice_cx(img_c, cx_fine, centre.cy, r_min_sq, r_max_sq, VAR_N_BINS)
    cost_cy_fine = _cost_slice_cy(img_c, cy_fine, centre.cx, r_min_sq, r_max_sq, VAR_N_BINS)

    # -- Figure layout -------------------------------------------------------
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(
        2, 3, figure=fig,
        width_ratios=[1.2, 1.0, 1.0],
        hspace=0.42, wspace=0.32,
        left=0.06, right=0.97, top=0.88, bottom=0.07,
    )
    ax = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)]

    vlo = float(np.percentile(cal_ds,  1))
    vhi = float(np.percentile(cal_ds, 99))
    fmt = plt.FuncFormatter(lambda v, _: f"{v/1e8:.2f}")
    gcx, gcy = centre.grid_cx, centre.grid_cy
    step_px  = 2.0 * VAR_SEARCH_PX / 19.0

    # -- [0,0] Image with coarse grid centre (lime green) -------------------
    ax[0][0].imshow(cal_ds, cmap="gray", origin="lower",
                    vmin=vlo, vmax=vhi, aspect="equal", interpolation="none")
    ax[0][0].axhline(gcy, color="lime", lw=0.9, ls="--", alpha=0.9)
    ax[0][0].axvline(gcx, color="lime", lw=0.9, ls="--", alpha=0.9)
    ax[0][0].plot(gcx, gcy, "s", color="lime", ms=7, mew=1.5, mfc="none")
    ax[0][0].set_title(
        f"Pass 1 -- coarse grid  (step {step_px:.1f} px)\n"
        f"cx = {gcx:.2f} px,  cy = {gcy:.2f} px", fontsize=9)
    ax[0][0].set_xlabel("Column (pixel)", fontsize=8)
    ax[0][0].set_ylabel("Row (pixel)", fontsize=8)
    ax[0][0].tick_params(labelsize=7)

    # -- [0,1] Variance vs cx, coarse --------------------------------------
    ax[0][1].plot(cx_grid, cost_cx_coarse, "-", color="steelblue", lw=1.0, alpha=0.5)
    ax[0][1].scatter(cx_grid, cost_cx_coarse, s=60, color="steelblue",
                     zorder=3, label="20 grid points")
    ax[0][1].axvline(gcx, color="limegreen", lw=1.5,
                     label=f"grid best cx = {gcx:.2f} px")
    ax[0][1].set_xlabel("cx (pixel)", fontsize=8)
    ax[0][1].set_ylabel("Azimuthal variance cost", fontsize=8)
    ax[0][1].set_title("Variance vs cx  [grid search, cy fixed at seed]", fontsize=9)
    ax[0][1].legend(fontsize=7)
    ax[0][1].tick_params(labelsize=7)
    ax[0][1].yaxis.set_major_formatter(fmt)

    # -- [0,2] Variance vs cy, coarse --------------------------------------
    ax[0][2].plot(cy_grid, cost_cy_coarse, "-", color="darkorange", lw=1.0, alpha=0.5)
    ax[0][2].scatter(cy_grid, cost_cy_coarse, s=60, color="darkorange",
                     zorder=3, label="20 grid points")
    ax[0][2].axvline(gcy, color="limegreen", lw=1.5,
                     label=f"grid best cy = {gcy:.2f} px")
    ax[0][2].set_xlabel("cy (pixel)", fontsize=8)
    ax[0][2].set_ylabel("Azimuthal variance cost", fontsize=8)
    ax[0][2].set_title("Variance vs cy  [grid search, cx fixed at seed]", fontsize=9)
    ax[0][2].legend(fontsize=7)
    ax[0][2].tick_params(labelsize=7)
    ax[0][2].yaxis.set_major_formatter(fmt)

    # -- [1,0] Image with NM fine centre (cyan + yellow) -------------------
    ax[1][0].imshow(cal_ds, cmap="gray", origin="lower",
                    vmin=vlo, vmax=vhi, aspect="equal", interpolation="none")
    ax[1][0].axhline(centre.cy, color="cyan", lw=0.9, ls="--", alpha=0.85)
    ax[1][0].axvline(centre.cx, color="cyan", lw=0.9, ls="--", alpha=0.85)
    ax[1][0].plot(centre.cx, centre.cy, "+", color="yellow", ms=12, mew=2.0)
    ax[1][0].set_title(
        f"Pass 2 -- Nelder-Mead refinement\n"
        f"cx = {centre.cx:.3f} +/- {centre.sigma_cx:.3f} px,  "
        f"cy = {centre.cy:.3f} +/- {centre.sigma_cy:.3f} px", fontsize=9)
    ax[1][0].set_xlabel("Column (pixel)", fontsize=8)
    ax[1][0].set_ylabel("Row (pixel)", fontsize=8)
    ax[1][0].tick_params(labelsize=7)

    # -- [1,1] Variance vs cx, dense NM window ----------------------------
    ax[1][1].plot(cx_fine, cost_cx_fine, ".", color="steelblue", ms=3, alpha=0.8)
    ax[1][1].axvline(centre.cx, color="red", lw=1.5,
                     label=f"NM cx = {centre.cx:.3f} px")
    ax[1][1].axvspan(centre.cx - centre.sigma_cx,
                     centre.cx + centre.sigma_cx,
                     alpha=0.18, color="red",
                     label=f"+-1s = {centre.sigma_cx:.3f} px")
    ax[1][1].axvspan(centre.cx - centre.two_sigma_cx,
                     centre.cx + centre.two_sigma_cx,
                     alpha=0.08, color="salmon",
                     label=f"+-2s = {centre.two_sigma_cx:.3f} px")
    ax[1][1].set_xlabel("cx (pixel)", fontsize=8)
    ax[1][1].set_ylabel("Azimuthal variance cost", fontsize=8)
    ax[1][1].set_title(f"Variance vs cx  [Nelder-Mead, +/-{NM_WIN:.0f} px window]", fontsize=9)
    ax[1][1].legend(fontsize=7)
    ax[1][1].tick_params(labelsize=7)
    ax[1][1].yaxis.set_major_formatter(fmt)

    # -- [1,2] Variance vs cy, dense NM window ----------------------------
    ax[1][2].plot(cy_fine, cost_cy_fine, ".", color="darkorange", ms=3, alpha=0.8)
    ax[1][2].axvline(centre.cy, color="red", lw=1.5,
                     label=f"NM cy = {centre.cy:.3f} px")
    ax[1][2].axvspan(centre.cy - centre.sigma_cy,
                     centre.cy + centre.sigma_cy,
                     alpha=0.18, color="red",
                     label=f"+-1s = {centre.sigma_cy:.3f} px")
    ax[1][2].axvspan(centre.cy - centre.two_sigma_cy,
                     centre.cy + centre.two_sigma_cy,
                     alpha=0.08, color="salmon",
                     label=f"+-2s = {centre.two_sigma_cy:.3f} px")
    ax[1][2].set_xlabel("cy (pixel)", fontsize=8)
    ax[1][2].set_ylabel("Azimuthal variance cost", fontsize=8)
    ax[1][2].set_title(f"Variance vs cy  [Nelder-Mead, +/-{NM_WIN:.0f} px window]", fontsize=9)
    ax[1][2].legend(fontsize=7)
    ax[1][2].tick_params(labelsize=7)
    ax[1][2].yaxis.set_major_formatter(fmt)

    # -- Super-title ---------------------------------------------------------
    fig.suptitle(
        f"Two-pass azimuthal variance centre finder -- {os.path.basename(cal_path)}\n"
        f"ROI {H}x{W} px  |  "
        f"seed: cx = {cx_seed:.1f},  cy = {cy_seed:.1f} px  |  "
        f"grid cost = {centre.grid_cost:.3e}  ->  NM cost = {centre.cost_at_min:.3e}",
        fontsize=10, fontweight="bold",
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 2 — peak identification table
# ---------------------------------------------------------------------------

def figure2_peak_table(fp: FringeProfile, peaks: list[PeakFitR2]) -> plt.Figure:
    """
    Figure 2: radial profile with fitting windows + peak centroid lines (top),
    and peak identification table (bottom).

    The profile panel lets you visually inspect and adjust _R2_WINDOWS: the
    shaded bands show exactly what window each peak was fit over; the thin
    vertical lines show the recovered r2_fit centroid position.

    Parameters
    ----------
    fp    : FringeProfile from annular_reduce() — provides r2_grid and profile.
    peaks : list[PeakFitR2] from fp.peak_fits_r2.
    """
    # ── colours by line family (even=640.2 nm blue, odd=638.3 nm amber) ──────
    C_A      = "#1f77b4"   # 640.2 nm — blue
    C_B      = "#ff7f0e"   # 638.3 nm — amber
    C_A_FILL = "#aec7e8"   # window shading, 640.2 nm
    C_B_FILL = "#ffbb78"   # window shading, 638.3 nm
    C_VLINE  = "#d62728"   # centroid line — red

    n_rows = len(peaks)

    # ── Figure layout: tall profile on top, table below ──────────────────────
    tbl_h = max(3.5, 0.38 * n_rows + 1.8)
    fig   = plt.figure(figsize=(22, 5.5 + tbl_h))
    gs    = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[5.5, tbl_h],
        hspace=0.08,
        left=0.05, right=0.98, top=0.93, bottom=0.04,
    )
    ax_prof  = fig.add_subplot(gs[0])
    ax_table = fig.add_subplot(gs[1])
    ax_table.axis("off")

    # ── Radial profile ────────────────────────────────────────────────────────
    good     = ~fp.masked
    r2_good  = fp.r2_grid[good]
    I_good   = fp.profile[good]
    sem_good = fp.sigma_profile[good]

    # Plot the profile with SEM error band
    ax_prof.plot(r2_good, I_good, color="steelblue", lw=0.9,
                 zorder=3, label="I(r²) profile")
    ax_prof.fill_between(r2_good,
                         I_good - sem_good, I_good + sem_good,
                         color="steelblue", alpha=0.18, zorder=2)

    # ── Shade fitting windows and draw centroid lines ─────────────────────────
    window_legend_a = window_legend_b = False
    centroid_legend = False
    midpoint_legend = False
    midpoint_legend = False

    for k, pf in enumerate(peaks):
        is_a    = (k % 2 == 0)
        fc      = C_A_FILL if is_a else C_B_FILL
        lc      = C_A      if is_a else C_B
        lam_str = "640.2" if is_a else "638.3"

        # Shaded fitting window from _R2_WINDOWS
        if k in _R2_WINDOWS:
            _wk   = _R2_WINDOWS[k]
            r2_lo, r2_hi = _wk[0], _wk[-1]
            # 3-tuple: use explicit centre; 2-tuple: fall back to midpoint
            r2_mid = _wk[1] if len(_wk) == 3 else 0.5 * (r2_lo + r2_hi)

            # Window shade label — only once per family
            if is_a and not window_legend_a:
                ax_prof.axvspan(r2_lo, r2_hi, alpha=0.18, color=fc,
                                zorder=1, label="640.2 nm fit window")
                window_legend_a = True
            elif not is_a and not window_legend_b:
                ax_prof.axvspan(r2_lo, r2_hi, alpha=0.18, color=fc,
                                zorder=1, label="638.3 nm fit window")
                window_legend_b = True
            else:
                ax_prof.axvspan(r2_lo, r2_hi, alpha=0.18, color=fc, zorder=1)

            # Dashed green line at the centre guess
            if not midpoint_legend:
                ax_prof.axvline(r2_mid, color="limegreen", lw=0.9, ls="--",
                                alpha=0.85, zorder=3, label="window centre (initial guess)")
                midpoint_legend = True
            else:
                ax_prof.axvline(r2_mid, color="limegreen", lw=0.9, ls="--",
                                alpha=0.85, zorder=3)

            # Peak index label at the centre guess
            ax_prof.text(
                r2_mid,
                ax_prof.get_ylim()[1] if ax_prof.get_ylim()[1] != 0 else 1,
                str(k),
                ha="center", va="bottom",
                fontsize=6.5, color=lc, fontweight="bold",
                clip_on=True,
            )

        # Thin centroid line
        if pf.fit_ok and np.isfinite(pf.r2_fit_px2):
            if not centroid_legend:
                ax_prof.axvline(pf.r2_fit_px2, color=C_VLINE, lw=0.8,
                                alpha=0.75, zorder=4,
                                label="r²_fit centroid")
                centroid_legend = True
            else:
                ax_prof.axvline(pf.r2_fit_px2, color=C_VLINE, lw=0.8,
                                alpha=0.75, zorder=4)

    ax_prof.set_xlabel("r²  (px²)", fontsize=9)
    ax_prof.set_ylabel("Mean intensity  (ADU)", fontsize=9)
    ax_prof.set_xlim(float(fp.r2_grid[0]), float(fp.r2_grid[-1]))
    ax_prof.tick_params(labelsize=8)
    ax_prof.legend(fontsize=8, loc="upper right")
    ax_prof.set_title(
        f"Radial profile I(r²)  |  {n_rows} peaks detected  |  "
        f"n_bins={fp.n_bins}, r_max={fp.r_max_px:.0f} px  |  "
        "shaded = _R2_WINDOWS fitting range  |  "
        "red lines = r²_fit centroids  |  "
        "green dashed = window midpoint (initial guess)",
        fontsize=9,
    )

    # Draw peak index labels at the correct y height after axes are populated
    # (re-draw now that ylim is known)
    ymax = ax_prof.get_ylim()[1]
    for k, pf in enumerate(peaks):
        if k in _R2_WINDOWS:
            _wk = _R2_WINDOWS[k]; r2_lo, r2_hi = _wk[0], _wk[-1]
            ax_prof.text(
                0.5 * (r2_lo + r2_hi), ymax * 0.995,
                str(k),
                ha="center", va="top",
                fontsize=6.5,
                color=C_A if k % 2 == 0 else C_B,
                fontweight="bold", clip_on=True,
            )

    # ── Peak table ────────────────────────────────────────────────────────────
    # Unified columns: win_lo/hi added; r_derived, ±σ_r, para_ok removed
    col_labels = [
        "Peak", "λ\n(nm)",
        "win_lo\n(px²)", "win_hi\n(px²)",
        "r²_raw\n(px²)", "r²_fit\n(px²)", "±σ r²\n(px²)",
        "Amp\n(ADU)", "Width σ r²\n(px²)", "χ²_red",
    ]

    rows        = []
    row_colours = []
    for k, pf in enumerate(peaks):
        lam_str = "640.2" if k % 2 == 0 else "638.3"
        win_lo_s, win_hi_s = "—", "—"
        if k in _R2_WINDOWS:
            _wk = _R2_WINDOWS[k]
            win_lo_s = f"{_wk[0]:.0f}"
            win_hi_s = f"{_wk[-1]:.0f}"
        if pf.fit_ok and np.isfinite(pf.r2_fit_px2) and pf.r2_fit_px2 > 0:
            r2_fit  = f"{pf.r2_fit_px2:.2f}"
            sig_r2  = (f"{pf.sigma_r2_fit_px2:.3f}"
                       if np.isfinite(pf.sigma_r2_fit_px2) else "—")
            wid_s   = (f"{pf.width_r2_px2:.2f}"
                       if np.isfinite(pf.width_r2_px2) else "—")
            chi2_s  = (f"{pf.reduced_chi2:.2f}"
                       if np.isfinite(pf.reduced_chi2) else "—")
        else:
            r2_fit = sig_r2 = wid_s = chi2_s = "—"

        rows.append([
            str(k), lam_str,
            win_lo_s, win_hi_s,
            f"{pf.r2_raw_px2:.1f}",
            r2_fit, sig_r2,
            f"{pf.amplitude_adu:.0f}", wid_s, chi2_s,
        ])
        row_colours.append("#EAF4FF" if k % 2 == 0 else "#FFF6EA")

    n_cols_tbl = len(col_labels)
    tbl = ax_table.table(
        cellText=rows,
        colLabels=col_labels,
        loc="upper center",
        cellLoc="center",
        colWidths=[0.04, 0.05, 0.07, 0.07,
                   0.09, 0.09, 0.07,
                   0.07, 0.09, 0.06],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.6)
    for c in range(n_cols_tbl):
        tbl[0, c].set_facecolor("#2C3E50")
        tbl[0, c].set_text_props(color="white", fontweight="bold")
        tbl[0, c].set_height(tbl[0, c].get_height() * 1.8)
    for r_idx, bg in enumerate(row_colours):
        fail = not peaks[r_idx].fit_ok
        for c in range(n_cols_tbl):
            tbl[r_idx + 1, c].set_facecolor("#FFE0E0" if fail else bg)
    ax_table.set_title(
        f"Peak Identification Table  |  {n_rows} peaks  |  "
        "Blue: 640.2 nm  |  Amber: 638.3 nm  |  Red: fit failed",
        fontsize=9, fontweight="bold", pad=4,
    )
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    print("=" * 68)
    print("  WindCube FPI Neon Calibration Pipeline")
    print("=" * 68)

    # ── Step 1: file selection ───────────────────────────────────────────────
    # HARDCODED FILES — set USE_DIALOG = True to re-enable the file picker
    USE_DIALOG = False
    _HARDCODED_CAL  = (
        r"C:\Users\sewell\Documents\GitHub\soc_sewell\\"
        r"data_reference\cals_19_may_2026\0_30exp_cal_swapped.bin"
    )
    _HARDCODED_DARK = (
        r"C:\Users\sewell\Documents\GitHub\soc_sewell\\"
        r"data_reference\darks_18_may_2026\10_light_leak_30sexp_dark_swapped.bin"
    )

    if USE_DIALOG:
        print("\n[1/7]  Select files via dialog...")
        cal_path  = _pick_file("Select CALIBRATION (.bin) image")
        if not cal_path:
            print("  No calibration file selected — exiting.")
            return
        dark_path = _pick_file(
            "Select DARK (.bin) image",
            initialdir=str(pathlib.Path(cal_path).parent),
        )
        if not dark_path:
            print("  No dark file selected — exiting.")
            return
    else:
        print("\n[1/7]  Using hardcoded files (set USE_DIALOG=True to use file picker)...")
        cal_path  = _HARDCODED_CAL
        dark_path = _HARDCODED_DARK
        for _p, _label in [(cal_path, 'Cal'), (dark_path, 'Dark')]:
            if not pathlib.Path(_p).exists():
                print(f"  ERROR: {_label} file not found: {_p}")
                return

    print(f"  Cal  : {cal_path}")
    print(f"  Dark : {dark_path}")

    # Output directory — one level up from the selected cal file
    output_dir = pathlib.Path(cal_path).parent.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Figures will be saved to: {output_dir}")

    # ── Step 2: load and dark-subtract ──────────────────────────────────────
    print("\n[2/7]  Loading images...")
    hdr_cal,  img_cal  = _load_raw(cal_path)
    hdr_dark, img_dark = _load_raw(dark_path)

    print(f"  Cal  : {img_cal.shape[0]}×{img_cal.shape[1]}  "
          f"ADU [{img_cal.min()}, {img_cal.max()}]  "
          f"mean {img_cal.mean():.1f}  "
          f"exp {_exp_time_s(hdr_cal)*1000:.0f} ms  "
          f"type '{_image_type_str(hdr_cal)}'")
    print(f"  Dark : {img_dark.shape[0]}×{img_dark.shape[1]}  "
          f"ADU [{img_dark.min()}, {img_dark.max()}]  "
          f"mean {img_dark.mean():.1f}  "
          f"exp {_exp_time_s(hdr_dark)*1000:.0f} ms  "
          f"type '{_image_type_str(hdr_dark)}'")

    if img_cal.shape != img_dark.shape:
        print("  ERROR: cal and dark images have different shapes — exiting.")
        return


    # Dark subtraction — full frame float64; floor at 0
    cal_ds = np.maximum(
        img_cal.astype(np.float64) - img_dark.astype(np.float64), 0.0
    )
    print(f"  Dark-subtracted full frame: ADU [{cal_ds.min():.1f}, {cal_ds.max():.1f}]  "
          f"mean {cal_ds.mean():.1f}")

    # Extract the 220x220 dark-subtracted ROI centred on the coarse seed.
    # This is the working array for ALL subsequent pipeline steps.
    r_lo = max(0, CENTRE_SEED_ROW - ROI_HALF)
    r_hi = min(cal_ds.shape[0], CENTRE_SEED_ROW + ROI_HALF)
    c_lo = max(0, CENTRE_SEED_COL - ROI_HALF)
    c_hi = min(cal_ds.shape[1], CENTRE_SEED_COL + ROI_HALF)
    roi_ds     = cal_ds[r_lo:r_hi, c_lo:c_hi]
    roi_bounds = (r_lo, r_hi, c_lo, c_hi)
    print(f"  ROI: {roi_ds.shape[0]}x{roi_ds.shape[1]} px  "
          f"rows {r_lo}-{r_hi}, cols {c_lo}-{c_hi}  "
          f"ADU [{roi_ds.min():.1f}, {roi_ds.max():.1f}]  mean {roi_ds.mean():.1f}")

    # -------------------------------------------------------------------------
    # Figures — numbered filenames so they sort in production order
    # 0_cal_raw_images_metadata.png
    # 1_cal_fringe_center_nelder_mead.png
    # 2_cal_peak_identification_table.png
    # 3_cal_peak_fit_diagnostics.png   (+ 3_cal_peak_fit_diagnostics_peak_table.png)
    # 4_cal_tolansky_two_line_analysis.png
    # -------------------------------------------------------------------------

    # -- Figure 0: raw images, metadata, dark-subtracted ROI ----------------
    print("\n  Building Figure 0 (raw images + metadata + dark-subtracted ROI)...")
    fig0 = figure0_raw_images(
        img_cal, img_dark, hdr_cal, hdr_dark,
        roi_ds, roi_bounds, cal_path, dark_path)
    _fig0_path = output_dir / "0_cal_raw_images_metadata.png"
    fig0.savefig(_fig0_path, dpi=150, bbox_inches="tight")
    print(f"  Saved : {_fig0_path}")
    fig0.show()

    # ── Step 3: centre finding on ROI ─────────────────────────────────────────
    print("\n[3/7]  Two-pass azimuthal variance centre finding...")

    # Coarse seed = geometric centre of the ROI (ROI-local pixel coords).
    # CENTRE_SEED_ROW/COL only controls the ROI box in Figure 0.
    H, W = roi_ds.shape
    roi_cx_seed = (W - 1) / 2.0   # col centre of ROI (local coords)
    roi_cy_seed = (H - 1) / 2.0   # row centre of ROI (local coords)
    print(f"  ROI size : {H}x{W} px")
    print(f"  Coarse seed (ROI centre) : cx={roi_cx_seed:.1f}, cy={roi_cy_seed:.1f}  (ROI-local col, row)")

    var_r_max_px = VAR_R_MAX_PX if VAR_R_MAX_PX is not None else float(min(H, W) // 2 - 10)

    centre = find_centre(
        roi_ds,
        cx_seed=roi_cx_seed,
        cy_seed=roi_cy_seed,
        var_r_min_px=VAR_R_MIN_PX,
        var_r_max_px=var_r_max_px,
        var_n_bins=VAR_N_BINS,
        var_search_px=VAR_SEARCH_PX,
    )
    print(f"  Pass 1 grid best: cx={centre.grid_cx:.3f} cy={centre.grid_cy:.3f}  "
          f"cost={centre.grid_cost:.1f}")
    print(f"  Pass 2 NM fine : cx={centre.cx:.4f} ± {centre.sigma_cx:.4f}  "
          f"cy={centre.cy:.4f} ± {centre.sigma_cy:.4f}  "
          f"cost={centre.cost_at_min:.1f}")

    # -- Figure 1: two-pass centre finding -----------------------------------
    print("\n  Building Figure 1 (centre finding)...")
    fig1       = figure1_centre_finding(
        roi_ds, centre, roi_cx_seed, roi_cy_seed, cal_path, dark_path, var_r_max_px)
    _fig1_path = output_dir / "1_cal_fringe_center_nelder_mead.png"
    fig1.savefig(_fig1_path, dpi=150, bbox_inches="tight")
    print(f"  Saved : {_fig1_path}")
    fig1.show()

    # -- Step 4: annular reduction + peak fitting ----------------------------
    print("\n[4/7]  Annular reduction and parabolic peak fitting...")
    print(f"  r_max={R_MAX_PX} px  n_bins={N_BINS}  "
          f"prominence={PEAK_PROMINENCE} ADU")

    fp = annular_reduce(
        roi_ds,
        cx=centre.cx,
        cy=centre.cy,
        sigma_cx=centre.sigma_cx,
        sigma_cy=centre.sigma_cy,
        r_max_px=R_MAX_PX,
        n_bins=N_BINS,
        peak_prominence=PEAK_PROMINENCE,
    )

    peaks  = fp.peak_fits_r2
    n_peaks = len(peaks)
    n_ok    = sum(1 for p in peaks if p.fit_ok)
    n_para  = sum(1 for p in peaks if p.para_ok)
    print(f"  Peaks detected : {n_peaks}")
    print(f"  Fit ok         : {n_ok}/{n_peaks}")
    print(f"  Parabola ok    : {n_para}/{n_peaks}")
    _print_peak_table_r2(peaks)

    # -- Figure 2: peak identification table ---------------------------------
    print("\n[5/7]  Building Figure 2 (peak identification table)...")
    fig2       = figure2_peak_table(fp, peaks)
    _fig2_path = output_dir / "2_cal_peak_identification_table.png"
    fig2.savefig(_fig2_path, dpi=150, bbox_inches="tight")
    print(f"  Saved : {_fig2_path}")
    fig2.show()

    # -- Figure 2b: zoomed radial profile r² > 12000 — for tuning outer peaks -
    print("  Building Figure 2b (zoomed outer-fringe profile)...")
    R2_ZOOM_LO = 12000.0   # px² — left edge of zoom window
    fig2b, ax2b = plt.subplots(figsize=(20, 5))

    # Plot the full r² profile clipped to the zoom region
    r2g   = fp.r2_grid
    prof  = fp.profile
    sig   = fp.sigma_profile
    mask  = r2g >= R2_ZOOM_LO
    ax2b.plot(r2g[mask], prof[mask], color="steelblue", lw=0.9, zorder=2,
              label="I(r²) profile")
    ax2b.fill_between(r2g[mask],
                      prof[mask] - sig[mask],
                      prof[mask] + sig[mask],
                      alpha=0.25, color="steelblue", zorder=1)

    # Compute y-range from zoomed data before the loop (needed for text labels)
    _zoom_prof = prof[mask]
    _ylo  = float(np.nanmin(_zoom_prof))
    _yhi  = float(np.nanmax(_zoom_prof))
    _ypad = 0.05 * (_yhi - _ylo)

    # Re-draw window shading, green centre lines and red centroid lines
    _midpt_done = False
    _ctr_done   = False
    for k, pf in enumerate(peaks):
        is_a = (k % 2 == 0)
        fc   = "#5588CC" if is_a else "#CC8844"
        lc   = "#2255AA" if is_a else "#AA5522"
        C_VL = "#CC2222"

        if k in _R2_WINDOWS:
            _wk      = _R2_WINDOWS[k]
            r2_lo    = _wk[0];  r2_hi = _wk[-1]
            r2_mid   = _wk[1] if len(_wk) == 3 else 0.5 * (r2_lo + r2_hi)
            if r2_hi < R2_ZOOM_LO:
                continue   # entirely left of zoom window
            ax2b.axvspan(r2_lo, r2_hi, alpha=0.18, color=fc, zorder=1)

            # Green dashed centre line
            lbl_mid = "window centre (initial guess)" if not _midpt_done else None
            ax2b.axvline(r2_mid, color="limegreen", lw=1.0, ls="--",
                         alpha=0.9, zorder=3, label=lbl_mid)
            _midpt_done = True

            # Peak index label at green line
            ax2b.text(r2_mid, _yhi + _ypad * 0.3,
                      str(k), ha="center", va="bottom",
                      fontsize=7.5, color=lc, fontweight="bold", clip_on=True)

        # Red fitted centroid
        if pf.fit_ok and np.isfinite(pf.r2_fit_px2) and pf.r2_fit_px2 >= R2_ZOOM_LO:
            lbl_ctr = "r²_fit centroid" if not _ctr_done else None
            ax2b.axvline(pf.r2_fit_px2, color=C_VL, lw=1.0,
                         alpha=0.8, zorder=4, label=lbl_ctr)
            _ctr_done = True

    ax2b.set_xlim(R2_ZOOM_LO, float(r2g[-1]))
    ax2b.set_ylim(_ylo - _ypad, _yhi + _ypad)
    ax2b.set_xlabel("r²  (px²)", fontsize=10)
    ax2b.set_ylabel("Mean intensity  (ADU)", fontsize=10)

    # Fine tick marks: major every 500 px², minor every 100 px²
    import matplotlib.ticker as mticker
    ax2b.xaxis.set_major_locator(mticker.MultipleLocator(500))
    ax2b.xaxis.set_minor_locator(mticker.MultipleLocator(100))
    ax2b.tick_params(axis="x", which="major", labelsize=8, length=6)
    ax2b.tick_params(axis="x", which="minor", length=3)
    ax2b.tick_params(axis="y", labelsize=8)
    ax2b.grid(axis="x", which="major", lw=0.4, alpha=0.4)
    ax2b.grid(axis="x", which="minor", lw=0.2, alpha=0.2)
    ax2b.legend(fontsize=8, loc="upper right")
    ax2b.set_title(
        f"Figure 2b — Outer-fringe radial profile  |  r² > {R2_ZOOM_LO:.0f} px²  |  "
        f"green dashed = window centre (initial guess)  |  red = r²_fit centroid\n"
        f"Adjust 3-tuple centre values in _R2_WINDOWS to align green lines with "
        f"visible fringe peaks",
        fontsize=9,
    )
    fig2b.tight_layout()
    _fig2b_path = output_dir / "2b_cal_outer_fringe_profile.png"
    fig2b.savefig(_fig2b_path, dpi=150, bbox_inches="tight")
    print(f"  Saved : {_fig2b_path}")
    fig2b.show()

    # -- Figures 3a/3b: per-peak fit diagnostics split inner / outer ---------
    # 3a: P0-P{INNER_MAX-1} (reliably fitted inner fringes)
    # 3b: P{INNER_MAX}+     (extended outer fringes, separate figure)
    print("\n[6/7]  Building Figure 3 (per-peak fit diagnostics)...")
    _INNER_MAX = 20

    def _plot_peak_subset(fp_obj, peak_slice, fit_half_window, n_cols, save_path, peak_offset=0):
        """Temporarily swap fp.peak_fits_r2 to plot a subset."""
        orig = fp_obj.peak_fits_r2
        try:
            fp_obj.peak_fits_r2 = peak_slice
            _plot_all_fringe_diagnostics_r2(
                fp_obj, fit_half_window=fit_half_window,
                n_cols=n_cols, save_path=save_path,
                peak_offset=peak_offset)
        finally:
            fp_obj.peak_fits_r2 = orig

    all_peaks = fp.peak_fits_r2

    # Assign each peak to its _R2_WINDOWS key by checking which window
    # contains its r2_raw_px2.  Peaks that fall in no defined window are
    # spurious find_peaks detections and are excluded from both figures.
    def _window_key(pf):
        """Return the _R2_WINDOWS key whose [lo,hi] contains pf.r2_raw_px2,
        or None if no window matches."""
        for k, w in _R2_WINDOWS.items():
            if w[0] <= pf.r2_raw_px2 <= w[-1]:
                return k
        return None

    inner_peaks = [p for p in all_peaks if (_window_key(p) is not None and
                                             _window_key(p) < _INNER_MAX)]
    outer_peaks = [p for p in all_peaks if (_window_key(p) is not None and
                                             _window_key(p) >= _INNER_MAX)]

    _fig3a_path = output_dir / "3a_cal_peak_fit_diagnostics_inner.png"
    _plot_peak_subset(fp, inner_peaks, fit_half_window=200, n_cols=4,
                      save_path=_fig3a_path)
    print(f"  Saved : {_fig3a_path}  ({len(inner_peaks)} inner peaks P0-P{_INNER_MAX-1})")

    # Figure 3b (outer fringe diagnostics) suppressed — no longer needed.


    # -- Step 7: Tolansky two-line analysis ----------------------------------
    print("\n[7/7]  Tolansky two-line analysis (640.2248 + 638.2991 nm)...")
    peak_array = peaks_to_array(peaks)
    n_valid    = np.sum(np.isfinite(peak_array[:, 2]))
    print(f"  Peaks in array : {len(peak_array)}  ({n_valid} with valid r2_fit)")

    if n_valid < 4:
        print("  ERROR: fewer than 4 valid fitted peaks -- "
              "Tolansky analysis skipped.")
        print("  -> Check _R2_WINDOWS and PEAK_PROMINENCE settings.")
    else:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = run_tolansky_2line(
                peak_array,
                lam_a_m=LAM_A_M,
                lam_b_m=LAM_B_M,
                d_prior_m=D_PRIOR_M,
            )
        for w in caught:
            print(f"  WARNING: {w.message}")

        print_rectangular_array(result)

        print(f"\n  -- Tolansky results ------------------------------------------")
        print(f"  d     = {result.d_m*1e3:.6f} +/- {result.sigma_d_m*1e3:.4f} mm  "
              f"(2s = +/-{result.two_sigma_d_m*1e3:.4f} mm)")
        print(f"  alpha = {result.alpha_mean:.4e} +/- {result.sigma_alpha:.4e} rad/px  "
              f"(2s = +/-{result.two_sigma_alpha:.4e})")
        print(f"  eps_a = {result.eps_a:.5f} +/- {result.sigma_eps_a:.5f}  "
              f"(640.2 nm fractional order)")
        print(f"  eps_b = {result.eps_b:.5f} +/- {result.sigma_eps_b:.5f}  "
              f"(638.3 nm fractional order)")
        print(f"  Delta_a = {result.Delta_a:.3f} +/- {result.sigma_Delta_a:.3f} px2/fr  "
              f"chi2/nu = {result.chi2_dof_a:.3f}")
        print(f"  Delta_b = {result.Delta_b:.3f} +/- {result.sigma_Delta_b:.3f} px2/fr  "
              f"chi2/nu = {result.chi2_dof_b:.3f}")
        ratio_ppm  = result.Delta_ratio_residual * 1e6
        ratio_flag = "PASS" if ratio_ppm  < 200  else "WARN"
        alpha_ppm  = result.alpha_consistency  * 1e6
        alpha_flag = "PASS" if alpha_ppm  < 1000 else "WARN"
        print(f"  Delta_a/Delta_b : {ratio_ppm:.1f} ppm  [{ratio_flag}]")
        print(f"  alpha consist.  : {alpha_ppm:.1f} ppm  [{alpha_flag}]")
        print(f"  N_Delta         : {result.N_Delta}")
        print(f"  -------------------------------------------------------------")

        # -- Figure 4: Tolansky diagnostic -----------------------------------
        print("\n  Building Figure 4 (Tolansky two-line diagnostic)...")
        fig4       = plot_tolansky_result(
            result, subtitle=f"{os.path.basename(cal_path)}")
        _fig4_path = output_dir / "4_cal_tolansky_two_line_analysis.png"
        fig4.savefig(_fig4_path, dpi=150, bbox_inches="tight")
        print(f"  Saved : {_fig4_path}")
        fig4.show()

        # -- n_pairs sensitivity sweep ---------------------------------------
        # Re-run Tolansky using only the innermost n_pairs ring pairs, where
        # n_pairs decreases from (all rings) down to MIN_PAIRS.  This tests
        # whether excluding the noisier outer fringes improves Δ precision or
        # shifts d / α systematically.
        #
        # Theory reminder:
        #   WLS σ_Δ ∝ 1/√(Σ w_i·(p_i − p̄)²)  — adding rings always reduces
        #   the formal error IF they are unbiased.  But outer rings may carry
        #   systematic centroid bias (low SNR, fitting window edge effects).
        #   χ²_red > 1 on either line is the smoking gun.
        #
        # Outputs:
        #   Console table + 5_cal_tolansky_npairs_sweep.png
        MIN_PAIRS = 4   # minimum ring pairs to try

        n_max = min(result.n_rings_a, result.n_rings_b)
        sweep_pairs   = list(range(n_max, MIN_PAIRS - 1, -1))

        # Storage
        sw_n      = []
        sw_d      = []
        sw_sd     = []
        sw_alpha  = []
        sw_sa     = []
        sw_Da     = []
        sw_sDa    = []
        sw_Db     = []
        sw_sDb    = []
        sw_c2a    = []
        sw_c2b    = []
        sw_ok     = []

        print(f"\n  -- n_pairs sensitivity sweep  (all → {MIN_PAIRS} innermost pairs) --")
        hdr = (f"  {'n':>3}  {'d (mm)':>12}  {'±σ_d (µm)':>10}  "
               f"{'alpha (rad/px)':>14}  {'±σ_α (ppm)':>10}  "
               f"{'Δ_a':>7}  {'±σΔ_a':>6}  {'χ²_a':>5}  "
               f"{'Δ_b':>7}  {'±σΔ_b':>6}  {'χ²_b':>5}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))

        for n_pairs_sw in sweep_pairs:
            try:
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter("always")
                    r_sw = run_tolansky_2line(
                        peak_array,
                        lam_a_m=LAM_A_M,
                        lam_b_m=LAM_B_M,
                        d_prior_m=D_PRIOR_M,
                        n_pairs=n_pairs_sw,
                    )
                flag = ""
                if r_sw.chi2_dof_a > 2.0 or r_sw.chi2_dof_b > 2.0:
                    flag = " [χ²!]"
                if abs(r_sw.d_m - result.d_m) * 1e6 > 3 * result.sigma_d_m * 1e6:
                    flag += " [d shift!]"
                alpha_ppm_sw = r_sw.sigma_alpha / r_sw.alpha_mean * 1e6
                print(
                    f"  {n_pairs_sw:>3}  "
                    f"{r_sw.d_m*1e3:>12.6f}  "
                    f"{r_sw.sigma_d_m*1e6:>10.3f}  "
                    f"{r_sw.alpha_mean:>14.6e}  "
                    f"{alpha_ppm_sw:>10.1f}  "
                    f"{r_sw.Delta_a:>7.2f}  {r_sw.sigma_Delta_a:>6.3f}  "
                    f"{r_sw.chi2_dof_a:>5.2f}  "
                    f"{r_sw.Delta_b:>7.2f}  {r_sw.sigma_Delta_b:>6.3f}  "
                    f"{r_sw.chi2_dof_b:>5.2f}"
                    f"{flag}"
                )
                sw_n.append(n_pairs_sw)
                sw_d.append(r_sw.d_m * 1e3)
                sw_sd.append(r_sw.sigma_d_m * 1e6)
                sw_alpha.append(r_sw.alpha_mean)
                sw_sa.append(r_sw.sigma_alpha / r_sw.alpha_mean * 1e6)
                sw_Da.append(r_sw.Delta_a)
                sw_sDa.append(r_sw.sigma_Delta_a)
                sw_Db.append(r_sw.Delta_b)
                sw_sDb.append(r_sw.sigma_Delta_b)
                sw_c2a.append(r_sw.chi2_dof_a)
                sw_c2b.append(r_sw.chi2_dof_b)
                sw_ok.append(True)
            except Exception as exc:
                print(f"  {n_pairs_sw:>3}  FAILED: {exc}")
                sw_ok.append(False)

        # -- Figure 5: n_pairs sweep plot ------------------------------------
        print("\n  Building Figure 5 (n_pairs sensitivity sweep)...")

        sw_n   = np.array(sw_n)
        sw_d   = np.array(sw_d)
        sw_sd  = np.array(sw_sd)
        sw_alpha = np.array(sw_alpha)
        sw_sa  = np.array(sw_sa)
        sw_Da  = np.array(sw_Da)
        sw_sDa = np.array(sw_sDa)
        sw_Db  = np.array(sw_Db)
        sw_sDb = np.array(sw_sDb)
        sw_c2a = np.array(sw_c2a)
        sw_c2b = np.array(sw_c2b)

        fig5, axes5 = plt.subplots(2, 3, figsize=(16, 9))
        fig5.suptitle(
            f"Tolansky n_pairs sensitivity sweep  |  {os.path.basename(cal_path)}\n"
            f"Each point uses only the innermost n ring pairs; all = {n_max}",
            fontsize=11, fontweight="bold",
        )
        _kw  = dict(marker="o", ms=5, lw=1.2)
        _kwA = dict(color="#1f77b4", **_kw)   # blue = line a / alpha
        _kwB = dict(color="#ff7f0e", **_kw)   # orange = line b / d

        # Panel [0,0]: d vs n_pairs
        ax = axes5[0, 0]
        ax.errorbar(sw_n, sw_d, yerr=sw_sd * 1e-3, **_kwB, capsize=3,
                    ecolor="#ff7f0e", elinewidth=0.8)
        ax.axhline(result.d_m * 1e3, color="gray", lw=0.8, ls="--",
                   label=f"full ({n_max}) = {result.d_m*1e3:.6f} mm")
        ax.set_xlabel("n_pairs (innermost rings used)", fontsize=9)
        ax.set_ylabel("d  (mm)", fontsize=9)
        ax.set_title("Etalon gap d", fontsize=9, fontweight="bold")
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

        # Panel [0,1]: σ_d vs n_pairs
        ax = axes5[0, 1]
        ax.plot(sw_n, sw_sd, **_kwB)
        ax.set_xlabel("n_pairs", fontsize=9)
        ax.set_ylabel("σ_d  (µm)", fontsize=9)
        ax.set_title("d uncertainty σ_d", fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)

        # Panel [0,2]: α uncertainty (ppm) vs n_pairs
        ax = axes5[0, 2]
        ax.plot(sw_n, sw_sa, **_kwA)
        ax.set_xlabel("n_pairs", fontsize=9)
        ax.set_ylabel("σ_α / α  (ppm)", fontsize=9)
        ax.set_title("Plate scale uncertainty σ_α (ppm)", fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)

        # Panel [1,0]: Δ_a and Δ_b vs n_pairs
        ax = axes5[1, 0]
        ax.errorbar(sw_n, sw_Da, yerr=sw_sDa,
                    label="Δ_a  640.2 nm", capsize=3,
                    elinewidth=0.8, **_kwA)
        ax.errorbar(sw_n, sw_Db, yerr=sw_sDb,
                    label="Δ_b  638.3 nm", capsize=3,
                    elinewidth=0.8, **_kwB)
        ax.axhline(result.Delta_a, color="#1f77b4", lw=0.7, ls="--", alpha=0.5)
        ax.axhline(result.Delta_b, color="#ff7f0e", lw=0.7, ls="--", alpha=0.5)
        ax.set_xlabel("n_pairs", fontsize=9)
        ax.set_ylabel("Δ  (px² / fringe)", fontsize=9)
        ax.set_title("FSR slope Δ_a and Δ_b", fontsize=9, fontweight="bold")
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

        # Panel [1,1]: σ_Δ vs n_pairs
        ax = axes5[1, 1]
        ax.plot(sw_n, sw_sDa, label="σ_Δ_a  640.2 nm", **_kwA)
        ax.plot(sw_n, sw_sDb, label="σ_Δ_b  638.3 nm", **_kwB)
        ax.set_xlabel("n_pairs", fontsize=9)
        ax.set_ylabel("σ_Δ  (px² / fringe)", fontsize=9)
        ax.set_title("Δ uncertainty — formal WLS error", fontsize=9, fontweight="bold")
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

        # Panel [1,2]: χ²_red vs n_pairs
        ax = axes5[1, 2]
        ax.plot(sw_n, sw_c2a, label="χ²_red  line a  640.2 nm", **_kwA)
        ax.plot(sw_n, sw_c2b, label="χ²_red  line b  638.3 nm", **_kwB)
        ax.axhline(1.0, color="green", lw=1.0, ls="--", alpha=0.7, label="χ²=1 (ideal)")
        ax.axhline(2.0, color="red",   lw=0.8, ls=":",  alpha=0.6, label="χ²=2 (warn)")
        ax.set_xlabel("n_pairs", fontsize=9)
        ax.set_ylabel("χ²_red", fontsize=9)
        ax.set_title("Goodness of fit χ²_red", fontsize=9, fontweight="bold")
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

        fig5.tight_layout()
        _fig5_path = output_dir / "5_cal_tolansky_npairs_sweep.png"
        fig5.savefig(_fig5_path, dpi=150, bbox_inches="tight")
        print(f"  Saved : {_fig5_path}")
        fig5.show()

    print("\n[done]  All figures saved and displayed.")
    print(f"  Output folder : {output_dir}")
    plt.show(block=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
