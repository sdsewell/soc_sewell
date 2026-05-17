"""run_cal_pipeline.py — WindCube FPI calibration workflow driver (SPEC-CAL01 v1.1).

Orchestrates stages S0–S6: dark stacking → centre finding → annular reduction
→ peak fitting → Tolansky seeds → H05 LM fit → OrbitCalResult.

All physics is performed by fpi_cal_lib.  This script handles:
  - Windows file-dialog UX (tkinter)
  - Per-stage diagnostic figures (saved as PNG + displayed non-blocking)
  - Intermediate .npy / .npz checkpoints
  - Stage error recovery with user confirmation

Usage
-----
    python src/processing/run_cal_pipeline.py

Output
------
    orbit_cal_result.npy   — pickled dict; loaded by run_airglow_inversion.py
"""
from __future__ import annotations

import datetime
import os
import pathlib
import sys
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# Add repo root to sys.path so src.* imports resolve when run as a script
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import src.fpi.fpi_cal_lib as cal

# ── Constants ─────────────────────────────────────────────────────────────────

_LAM_A_NM = 640.2248
_LAM_B_NM = 638.2991


# ── Frame loader ──────────────────────────────────────────────────────────────

def _load_bin_frame(path: pathlib.Path) -> np.ndarray:
    """
    Load a WindCube FPI binary frame → float64 image array (H, W).
    Big-endian uint16 ('>u2'). Dimensions read from header. Header row stripped.
    Matches load_image.py::load_raw() exactly.
    """
    with open(path, "rb") as f:
        first_words = np.frombuffer(f.read(4), dtype=">u2")
    n_rows_frame = int(first_words[0])
    n_cols_frame = int(first_words[1])
    expected = n_rows_frame * n_cols_frame * 2
    actual   = path.stat().st_size
    if actual != expected:
        raise ValueError(
            f"{path.name}: file size {actual} B, "
            f"expected {expected} for {n_rows_frame}×{n_cols_frame} uint16."
        )
    raw = np.frombuffer(open(path, "rb").read(), dtype=">u2")
    return raw[n_cols_frame:].reshape(n_rows_frame - 1, n_cols_frame).astype(np.float64)


# ── Plotting helpers (self-contained — copied from load_image.py) ─────────────

import struct as _struct

def _parse_header(h: np.ndarray) -> dict:
    """Decode the 276-word big-endian header row into a metadata dict."""
    from datetime import datetime, timezone

    def _u16(w):  return int(h[w])
    def _u64(w):  return sum(int(h[w+i]) << (16*i) for i in range(4))
    def _f64(w):
        b = _struct.pack(">4H", *reversed([h[w+i] for i in range(4)]))
        return _struct.unpack(">d", b)[0]
    def _u8arr(w, n): return [int(h[w+i]) & 0xFF for i in range(n)]

    lua_ms  = _u64(8)
    adcs_ms = _u64(12)
    try:
        utc = datetime.fromtimestamp(lua_ms/1000.0, tz=timezone.utc).isoformat()
    except Exception:
        utc = "invalid"

    gpio  = _u8arr(100, 4)
    lamps = _u8arr(104, 6)
    q_wxyz = [_f64(28 + i*4) for i in range(4)]
    q_xyzw = [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]
    e_wxyz = [_f64(44 + i*4) for i in range(4)]
    e_xyzw = [e_wxyz[1], e_wxyz[2], e_wxyz[3], e_wxyz[0]]
    shutter  = "closed" if (gpio[0]==1 and gpio[3]==1) else "open"
    img_type = "cal" if any(lamps) else ("dark" if shutter=="closed" else "science")

    return {
        "rows":                 _u16(0),
        "cols":                 _u16(1),
        "exp_time":             _u16(2),
        "exp_unit":             _u16(3),
        "ccd_temp1":            round(_f64(4), 4),
        "lua_timestamp":        lua_ms,
        "adcs_timestamp":       adcs_ms,
        "utc_timestamp":        utc,
        "attitude_quadternion": q_xyzw,
        "pointing_error":       e_xyzw,
        "spacecraft_position":  [_f64(60+i*4) for i in range(3)],
        "spacecraft_velocity":  [_f64(72+i*4) for i in range(3)],
        "spacecraft_latitude":  _f64(16),
        "spacecraft_longitude": _f64(20),
        "spacecraft_altitude":  _f64(24),
        "etalon_temps":         [_f64(84+i*4) for i in range(4)],
        "gpio_pwr_on":          gpio,
        "shutter_status":       shutter,
        "lamp_ch_array":        lamps,
        "lamp1_status":         "on" if lamps[0] else "off",
        "lamp2_status":         "on" if lamps[1] else "off",
        "lamp3_status":         "on" if lamps[2] else "off",
        "img_type":             img_type,
    }


_FIELD_META = {
    "rows":                 ("Rows",               "pixels",   None),
    "cols":                 ("Cols",               "pixels",   None),
    "exp_time":             ("Exposure time",       "cs",
                             lambda v: f"{v} cs  ({v/100:.2f} s)"),
    "exp_unit":             ("Exposure unit",       "register", None),
    "ccd_temp1":            ("CCD temperature",     "°C",       None),
    "lua_timestamp":        ("Lua timestamp",       "ms (Unix)",None),
    "adcs_timestamp":       ("ADCS timestamp",      "ms (Unix)",None),
    "utc_timestamp":        ("UTC timestamp",       "",         None),
    "attitude_quadternion": ("Attitude quaternion", "[x,y,z,w]",None),
    "pointing_error":       ("Pointing error",      "[x,y,z,w]",None),
    "spacecraft_position":  ("SC position (ECI)",   "m",        None),
    "spacecraft_velocity":  ("SC velocity (ECI)",   "m/s",      None),
    "spacecraft_latitude":  ("SC latitude",         "rad",      None),
    "spacecraft_longitude": ("SC longitude",        "rad",      None),
    "spacecraft_altitude":  ("SC altitude",         "m",        None),
    "etalon_temps":         ("Etalon temperatures", "°C",       None),
    "gpio_pwr_on":          ("GPIO power on",       "[ch0-3]",  None),
    "shutter_status":       ("Shutter status",      "",         None),
    "lamp_ch_array":        ("Lamp channel array",  "",         None),
    "lamp1_status":         ("Lamp 1 status",       "",         None),
    "lamp2_status":         ("Lamp 2 status",       "",         None),
    "lamp3_status":         ("Lamp 3 status",       "",         None),
    "img_type":             ("Image type",          "",         None),
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


def _plot_image(ax, fig, image: np.ndarray, title: str) -> None:
    """Imshow with 1st/99th percentile stretch and colorbar. Matches load_image.py."""
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
    """Histogram with 1st/99th percentile markers. Matches load_image.py."""
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


def _extract_roi(image: np.ndarray, center: tuple, half: int) -> np.ndarray:
    """Extract a square ROI centred at (row, col), clamped to image bounds."""
    r0, c0 = center
    return image[max(0,r0-half):min(image.shape[0],r0+half),
                 max(0,c0-half):min(image.shape[1],c0+half)]


def _make_metadata_table_data(metadata: dict):
    """Build cell_text and row_heights_in for the metadata table."""
    col_labels = ["#", "Field (key)", "Display name", "Units", "Value"]
    cell_text  = [
        [str(i), key,
         _FIELD_META.get(key, (key, "", None))[0],
         _FIELD_META.get(key, (key, "", None))[1],
         _fmt_value(key, val)]
        for i, (key, val) in enumerate(metadata.items(), start=1)
    ]
    _LINE_H  = 0.20; _MIN_ROW = 0.28; _HDR_ROW = 0.32
    row_heights_in = [
        max(_MIN_ROW, max(v.count("\n") + 1 for v in row) * _LINE_H)
        for row in cell_text
    ]
    table_h_in = max(6.0, sum(row_heights_in) + _HDR_ROW + 1.2)
    return col_labels, cell_text, row_heights_in, table_h_in


def _draw_metadata_table(ax, col_labels, cell_text, row_heights_in,
                          table_h_in, filename: str) -> None:
    """Draw the metadata table onto ax. Matches load_image.py table style."""
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
    hdr_bg = "#2C3E50"; alt_bg = "#EBF5FB"
    n_cols = len(col_labels)
    for c in range(n_cols):
        tbl[0, c].set_facecolor(hdr_bg)
        tbl[0, c].set_text_props(color="white", fontweight="bold")
        tbl[0, c].set_edgecolor("#CCCCCC")
    for r_idx, h_in in enumerate(row_heights_in):
        for c in range(n_cols):
            cell = tbl[r_idx+1, c]
            cell.set_height(h_in / table_h_in)
            cell.set_edgecolor("#CCCCCC")
            if r_idx % 2 == 1:
                cell.set_facecolor(alt_bg)
    ax.set_title(
        f"WindCube FPI Metadata — {filename}",
        fontsize=11, fontweight="bold", pad=8,
    )



# ── UI helpers ────────────────────────────────────────────────────────────────

def _make_root() -> tk.Tk:
    root = tk.Tk()
    root.withdraw()
    return root


def select_files(title: str) -> list[pathlib.Path]:
    """Open a multi-file dialog and return selected paths."""
    root = _make_root()
    paths = filedialog.askopenfilenames(
        title=title,
        filetypes=[("Binary frames", "*.bin"), ("All files", "*.*")],
    )
    root.destroy()
    if not paths:
        raise RuntimeError(f"No files selected: {title}")
    return sorted(pathlib.Path(p) for p in paths)


def ask_integer(prompt: str, title: str, default: int = 1, minvalue: int = 1) -> int:
    """Show an integer-input dialog; raise RuntimeError if cancelled."""
    root = _make_root()
    val = simpledialog.askinteger(title, prompt, initialvalue=default, minvalue=minvalue)
    root.destroy()
    if val is None:
        raise RuntimeError(f"No integer provided for: {title!r}")
    return val


def stage_banner(n: int, desc: str) -> None:
    """Print a prominent stage header to stdout."""
    print(f"\n{'='*60}\n  STAGE {n}: {desc}\n{'='*60}")


def run_with_error_handling(stage_n: int, fn):
    """
    Call fn(); on exception print the traceback and ask whether to continue.
    Returns fn()'s result, or None if the exception was dismissed.
    Exits if the user chooses not to continue.
    """
    try:
        return fn()
    except Exception:
        traceback.print_exc()
        root = _make_root()
        cont = messagebox.askyesno(
            f"Stage {stage_n} failed — continue?",
            f"Stage {stage_n} raised an exception.\n"
            "See the terminal for details.\n\n"
            "Continue to the next stage?",
        )
        root.destroy()
        if not cont:
            print("Pipeline aborted by user.")
            sys.exit(1)
        return None


def _save_and_show(fig: plt.Figure, path: pathlib.Path, show: bool = True) -> None:
    """Save figure as PNG; optionally display it."""
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path.name}")
    if show:
        plt.show(block=True)
    else:
        print(f"  [display suppressed — open {path.name} to view]")
    plt.close(fig)


# ── Figure S0a — One load_image-style figure per dark frame ─────────────────

def _figure_s0a(
    raw_stack: np.ndarray,
    dark_paths: list[pathlib.Path],
    save_dir: pathlib.Path,
) -> None:
    """
    One figure per dark frame: image (top-left), ROI (top-right),
    histograms (middle row), metadata table (bottom).
    Matches the load_image.py display exactly.
    """
    from matplotlib.gridspec import GridSpec
    _ROI_HALF = 100

    for frame, fpath in zip(raw_stack, dark_paths):
        # Read header from the bin file
        raw_be    = np.frombuffer(open(fpath, "rb").read(), dtype=">u2")
        n_cols    = int(raw_be[1])
        header_be = raw_be[:n_cols].copy()
        metadata  = _parse_header(header_be)

        img_u16 = frame.astype(np.uint16)
        cr, cc  = img_u16.shape[0] // 2, img_u16.shape[1] // 2
        roi     = _extract_roi(img_u16, (cr, cc), _ROI_HALF)

        col_labels, cell_text, row_heights_in, table_h_in = (
            _make_metadata_table_data(metadata)
        )

        # Build figure — same layout as load_image.py _build_figure()
        img_row_h = 5.0
        total_h   = img_row_h * 2 + table_h_in
        fig = plt.figure(figsize=(14.0, total_h))
        gs  = GridSpec(3, 2, figure=fig,
                       height_ratios=[img_row_h, img_row_h, table_h_in])

        ax00   = fig.add_subplot(gs[0, 0])
        ax01   = fig.add_subplot(gs[0, 1])
        ax10   = fig.add_subplot(gs[1, 0])
        ax11   = fig.add_subplot(gs[1, 1])
        ax_tbl = fig.add_subplot(gs[2, :])

        _plot_image(ax00, fig, img_u16,
                    f"Unmasked (full frame)")
        _plot_image(ax01, fig, roi,
                    f"ROI {roi.shape[0]}×{roi.shape[1]} px  "
                    f"centred at (row={cr}, col={cc})")
        _plot_hist(ax10, img_u16, "Pixel Distribution — Unmasked")
        _plot_hist(ax11, roi,     "Pixel Distribution — ROI")

        # Centre crosshair on full-frame image
        ax00.axhline(cr, color="cyan",   linewidth=0.8, linestyle="--", alpha=0.9)
        ax00.axvline(cc, color="cyan",   linewidth=0.8, linestyle="--", alpha=0.9)
        ax00.plot([cc-15, cc+15], [cr, cr], color="yellow", linewidth=1.5)
        ax00.plot([cc, cc], [cr-15, cr+15], color="yellow", linewidth=1.5)

        _draw_metadata_table(ax_tbl, col_labels, cell_text,
                             row_heights_in, table_h_in, fpath.name)
        fig.suptitle(f"Stage 0 — {fpath.name}", fontsize=12, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.97))

        save_path = save_dir / f"S0a_{fpath.stem}.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path.name}  [display suppressed — open file to view]")
        plt.close(fig)


# ── Figure S0b — Master dark ──────────────────────────────────────────────────

def _figure_s0b(
    master_dark: np.ndarray,
    dark_sigma: np.ndarray,
    dark_paths: list[pathlib.Path],
    master_dark_path: pathlib.Path,
    save_path: pathlib.Path,
) -> None:
    """1×3: master dark image | histogram | statistics table."""
    H, W       = master_dark.shape
    median_val = float(np.median(master_dark))
    img_u16    = master_dark.astype(np.uint16)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        f"Stage 0 — Master dark\n{master_dark_path}",
        fontsize=10, fontweight="bold",
    )

    _MAX14 = 2**14 - 1  # 16 383 — full 14-bit sensor range
    im = axes[0].imshow(img_u16, cmap="gray", origin="lower",
                        vmin=0, vmax=_MAX14, aspect="equal")
    cb = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    cb.set_label("Counts  (ADU)", fontsize=8)
    axes[0].set_title(
        f"Master dark (median stack)\n"
        f"{img_u16.shape[0]} rows × {img_u16.shape[1]} cols  |  "
        f"ADU [{img_u16.min()}, {img_u16.max()}]  |  "
        f"mean {img_u16.mean():.0f}  std {img_u16.std():.1f}",
        fontsize=8.5,
    )
    axes[0].set_xlabel("Column  (pixel)", fontsize=8)
    axes[0].set_ylabel("Row  (pixel)", fontsize=8)
    axes[0].tick_params(labelsize=7)

    axes[1].hist(img_u16.ravel(), bins=256, range=(0, _MAX14),
                 color="steelblue", edgecolor="none")
    axes[1].set_xlim(0, _MAX14)
    axes[1].axvline(median_val, color="red", linewidth=1.4, linestyle="--",
                    label=f"median = {median_val:.1f} ADU")
    axes[1].set_title("Histogram", fontsize=9)
    axes[1].set_xlabel("ADU  (uint16 counts)", fontsize=8)
    axes[1].set_ylabel("Number of pixels", fontsize=8)
    axes[1].tick_params(labelsize=7)
    axes[1].legend(fontsize=7)

    axes[2].axis("off")
    rows = [
        ["N frames stacked", str(len(dark_paths))],
        ["Mean ADU",         f"{master_dark.mean():.2f}"],
        ["Median ADU",       f"{median_val:.2f}"],
        ["Std ADU",          f"{master_dark.std():.2f}"],
        ["Min ADU",          f"{master_dark.min():.0f}"],
        ["Max ADU",          f"{master_dark.max():.0f}"],
        ["Max sigma (px)",   f"{dark_sigma.max():.2f}"],
        ["Shape",            f"{H} × {W} px"],
    ]
    tbl = axes[2].table(cellText=rows, colLabels=["Statistic", "Value"],
                        cellLoc="left", loc="upper left", edges="open")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    axes[2].set_title("Summary statistics", fontsize=9)

    fig.tight_layout()
    _save_and_show(fig, save_path)


# ── Figure S1 — Cal frame gallery (combined) ─────────────────────────────────

def _figure_s1_cal_gallery(
    raw_frames: list[np.ndarray],
    ds_frames: list[np.ndarray],
    centres: list,
    cal_paths: list[pathlib.Path],
    master_dark_path: pathlib.Path,
    save_path: pathlib.Path,
) -> None:
    """
    3 rows × N_cals columns:
      Row 0 — raw cal frames (shared vmin/vmax)
      Row 1 — dark-subtracted frames (shared vmin/vmax) + yellow '+' at centre
      Row 2 — histograms of dark-subtracted frames (log y, shared x range)
    """
    N = len(cal_paths)
    fig, axes = plt.subplots(3, N, figsize=(3.2 * N, 9), squeeze=False)
    fig.suptitle(
        f"Stage 1 — Cal frames ({N} frames)  |  master dark: {master_dark_path.name}",
        fontsize=11, fontweight="bold",
    )

    raw_stack = np.stack(raw_frames)
    ds_stack  = np.stack(ds_frames)

    # Shared image scales
    raw_vlo = float(np.percentile(raw_stack, 1))
    raw_vhi = float(np.percentile(raw_stack, 99))
    ds_vlo  = float(np.percentile(ds_stack, 1))
    ds_vhi  = float(np.percentile(ds_stack, 99))

    # Shared histogram x range
    ds_min = float(ds_stack.min())
    ds_max = float(ds_stack.max())

    for i, (raw, ds, ctr, path) in enumerate(
        zip(raw_frames, ds_frames, centres, cal_paths)
    ):
        # Row 0: raw cal
        ax0 = axes[0, i]
        ax0.imshow(raw, cmap="gray", origin="lower",
                   vmin=raw_vlo, vmax=raw_vhi, aspect="auto")
        ax0.set_title(path.name, fontsize=8)
        ax0.set_xlabel("x (px)", fontsize=8)
        ax0.set_ylabel("y (px)", fontsize=8)
        ax0.tick_params(labelsize=6)

        # Row 1: dark-subtracted + centre marker
        ax1 = axes[1, i]
        ax1.imshow(ds, cmap="gray", origin="lower",
                   vmin=ds_vlo, vmax=ds_vhi, aspect="auto")
        ax1.plot(ctr.cx, ctr.cy, "+", color="yellow",
                 markersize=10, markeredgewidth=1.5)
        ax1.set_title(
            f"Dark-subtracted\ncx={ctr.cx:.1f}  cy={ctr.cy:.1f}",
            fontsize=7,
        )
        ax1.set_xlabel("x (px)", fontsize=8)
        ax1.set_ylabel("y (px)", fontsize=8)
        ax1.tick_params(labelsize=6)

        # Row 2: histogram of dark-subtracted (log y, shared x)
        ax2 = axes[2, i]
        ax2.hist(ds.ravel(), bins=60, range=(ds_min, ds_max),
                 color="darkorange", edgecolor="none", alpha=0.8)
        ax2.set_yscale("log")
        ax2.axvline(0.0, color="red", linewidth=1.0, linestyle="--",
                    label="zero")
        ax2.set_xlabel("ADU", fontsize=8)
        ax2.set_ylabel("Pixel count", fontsize=8)
        ax2.tick_params(labelsize=6)
        if i == 0:
            ax2.legend(fontsize=7)

    fig.tight_layout()
    _save_and_show(fig, save_path, show=False)


# ── Figure S1 — Centre-finder QA (per frame) ─────────────────────────────────

def _figure_s1(
    raw: np.ndarray,
    ds: np.ndarray,
    ctr: "cal.CentreResult",
    stem: str,
    save_path: pathlib.Path,
) -> None:
    """2×3 centre-finder QA, replicating center_finder.py layout."""
    H, W = ds.shape
    var_r_max_px = float(min(H, W) // 2 - 10)
    r_min_sq = 5.0 ** 2
    r_max_sq = var_r_max_px ** 2
    n_var_bins = 250
    var_search_px = 15.0

    cx_seed = (W - 1) / 2.0
    cy_seed = (H - 1) / 2.0

    p99_5 = float(np.percentile(ds, 99.5))
    img_c = np.clip(ds, None, p99_5)

    # Coarse grid 1-D slices
    grid_offsets = np.linspace(-var_search_px, var_search_px, 20)
    grid_cx_pts  = cx_seed + grid_offsets
    grid_cy_pts  = cy_seed + grid_offsets
    grid_cost_cx = np.array([
        cal._variance_cost(cx, ctr.grid_cy, img_c, r_min_sq, r_max_sq, n_var_bins)
        for cx in grid_cx_pts
    ])
    grid_cost_cy = np.array([
        cal._variance_cost(ctr.grid_cx, cy, img_c, r_min_sq, r_max_sq, n_var_bins)
        for cy in grid_cy_pts
    ])
    coarse_cx_line = np.linspace(grid_cx_pts[0], grid_cx_pts[-1], 200)
    coarse_cy_line = np.linspace(grid_cy_pts[0], grid_cy_pts[-1], 200)
    coarse_cost_cx = np.array([
        cal._variance_cost(cx, ctr.grid_cy, img_c, r_min_sq, r_max_sq, n_var_bins)
        for cx in coarse_cx_line
    ])
    coarse_cost_cy = np.array([
        cal._variance_cost(ctr.grid_cx, cy, img_c, r_min_sq, r_max_sq, n_var_bins)
        for cy in coarse_cy_line
    ])

    # Fine NM scan
    fine_half = 5.0
    fine_pts  = 101
    fine_cx_scan = np.linspace(ctr.cx - fine_half, ctr.cx + fine_half, fine_pts)
    fine_cy_scan = np.linspace(ctr.cy - fine_half, ctr.cy + fine_half, fine_pts)
    fine_cost_cx = np.array([
        cal._variance_cost(cx, ctr.cy, img_c, r_min_sq, r_max_sq, n_var_bins)
        for cx in fine_cx_scan
    ])
    fine_cost_cy = np.array([
        cal._variance_cost(ctr.cx, cy, img_c, r_min_sq, r_max_sq, n_var_bins)
        for cy in fine_cy_scan
    ])

    fig = plt.figure(figsize=(17, 10))
    gs  = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.32)
    fig.suptitle(f"Stage 1 — Centre Finder QA: {stem}", fontsize=12, fontweight="bold")

    vlo = float(np.percentile(raw, 1))
    vhi = float(np.percentile(raw, 99))

    # [0,0] Raw image + lime-square grid-best mark
    ax00 = fig.add_subplot(gs[0, 0])
    ax00.imshow(raw, cmap="gray", origin="lower", vmin=vlo, vmax=vhi)
    ax00.axhline(ctr.grid_cy, color="lime", linewidth=0.8, linestyle="--", alpha=0.9)
    ax00.axvline(ctr.grid_cx, color="lime", linewidth=0.8, linestyle="--", alpha=0.9)
    ax00.plot(ctr.grid_cx, ctr.grid_cy, "s", color="lime",
              markersize=10, markeredgewidth=1.5, markerfacecolor="none",
              label=f"Grid best ({ctr.grid_cx:.1f}, {ctr.grid_cy:.1f})")
    ax00.set_title("Raw — coarse grid best")
    ax00.legend(fontsize=7, loc="upper right")
    ax00.set_xlabel("x (px)"); ax00.set_ylabel("y (px)")

    # [0,1] Variance vs cx (coarse grid)
    ax01 = fig.add_subplot(gs[0, 1])
    ax01.plot(coarse_cx_line, coarse_cost_cx, color="steelblue", linewidth=0.8)
    ax01.plot(grid_cx_pts, grid_cost_cx, "o", color="steelblue",
              markersize=4, label="Grid pts")
    ax01.axvline(ctr.grid_cx, color="lime", linewidth=1.0, linestyle="--")
    ax01.set_xlabel("cx (px)"); ax01.set_ylabel("Variance cost")
    ax01.set_title("Coarse variance vs cx")
    ax01.legend(fontsize=7)

    # [0,2] Variance vs cy (coarse grid)
    ax02 = fig.add_subplot(gs[0, 2])
    ax02.plot(coarse_cy_line, coarse_cost_cy, color="darkorange", linewidth=0.8)
    ax02.plot(grid_cy_pts, grid_cost_cy, "o", color="darkorange",
              markersize=4, label="Grid pts")
    ax02.axvline(ctr.grid_cy, color="lime", linewidth=1.0, linestyle="--")
    ax02.set_xlabel("cy (px)"); ax02.set_ylabel("Variance cost")
    ax02.set_title("Coarse variance vs cy")
    ax02.legend(fontsize=7)

    # [1,0] Dark-subtracted image + cyan crosshair + yellow + at NM result
    ax10 = fig.add_subplot(gs[1, 0])
    vlo_ds = float(np.percentile(ds, 1))
    vhi_ds = float(np.percentile(ds, 99))
    ax10.imshow(ds, cmap="gray", origin="lower", vmin=vlo_ds, vmax=vhi_ds)
    ax10.axhline(ctr.cy, color="cyan", linewidth=0.8, linestyle="-", alpha=0.8)
    ax10.axvline(ctr.cx, color="cyan", linewidth=0.8, linestyle="-", alpha=0.8)
    ax10.plot(ctr.cx, ctr.cy, "+", color="yellow",
              markersize=14, markeredgewidth=1.5,
              label=f"NM ({ctr.cx:.2f}, {ctr.cy:.2f})")
    ax10.set_title("Dark-subtracted — NM result")
    ax10.legend(fontsize=7, loc="upper right")
    ax10.set_xlabel("x (px)"); ax10.set_ylabel("y (px)")

    # [1,1] Variance vs cx (fine NM scan ±5 px, ±1σ/2σ spans)
    ax11 = fig.add_subplot(gs[1, 1])
    ax11.plot(fine_cx_scan, fine_cost_cx, color="steelblue", linewidth=1.0,
              marker="o", markersize=3, markevery=5)
    ax11.axvline(ctr.cx, color="red", linewidth=1.2, label=f"cx={ctr.cx:.2f}")
    ax11.axvspan(ctr.cx - ctr.sigma_cx, ctr.cx + ctr.sigma_cx,
                 alpha=0.15, color="red", label=f"±1σ ({ctr.sigma_cx:.2f} px)")
    ax11.axvspan(ctr.cx - ctr.two_sigma_cx, ctr.cx + ctr.two_sigma_cx,
                 alpha=0.07, color="red", label=f"±2σ ({ctr.two_sigma_cx:.2f} px)")
    ax11.set_xlabel("cx (px)"); ax11.set_ylabel("Variance cost")
    ax11.set_title("Fine variance vs cx")
    ax11.legend(fontsize=7)

    # [1,2] Variance vs cy (fine NM scan ±5 px, ±1σ/2σ spans)
    ax12 = fig.add_subplot(gs[1, 2])
    ax12.plot(fine_cy_scan, fine_cost_cy, color="darkorange", linewidth=1.0,
              marker="o", markersize=3, markevery=5)
    ax12.axvline(ctr.cy, color="red", linewidth=1.2, label=f"cy={ctr.cy:.2f}")
    ax12.axvspan(ctr.cy - ctr.sigma_cy, ctr.cy + ctr.sigma_cy,
                 alpha=0.15, color="red", label=f"±1σ ({ctr.sigma_cy:.2f} px)")
    ax12.axvspan(ctr.cy - ctr.two_sigma_cy, ctr.cy + ctr.two_sigma_cy,
                 alpha=0.07, color="red", label=f"±2σ ({ctr.two_sigma_cy:.2f} px)")
    ax12.set_xlabel("cy (px)"); ax12.set_ylabel("Variance cost")
    ax12.set_title("Fine variance vs cy")
    ax12.legend(fontsize=7)

    _save_and_show(fig, save_path, show=False)


# ── Figure S2 — Radial profiles (combined) ───────────────────────────────────

def _figure_s2(
    fps: list,
    stems: list[str],
    save_path: pathlib.Path,
) -> None:
    """Combined figure: 1 column × N rows, one row per radial profile."""
    n = len(fps)
    fig, axes = plt.subplots(n, 1, figsize=(14, 6 * n),
                             constrained_layout=True, squeeze=False)
    fig.suptitle("Stage 2 — Radial Profiles", fontsize=12, fontweight="bold")

    for row, (fp, stem) in enumerate(zip(fps, stems)):
        ax = axes[row, 0]
        good = ~fp.masked
        r2   = fp.r2_grid[good]
        prof = fp.profile[good]
        sig  = fp.sigma_profile[good]
        sig  = np.where(sig > 0, sig, np.nan)

        ax.fill_between(r2, prof - 2*sig, prof + 2*sig,
                        alpha=0.25, color="steelblue", label="±2σ SEM")
        ax.fill_between(r2, prof - sig, prof + sig,
                        alpha=0.40, color="steelblue", label="±1σ SEM")
        ax.plot(r2, prof, color="steelblue", linewidth=0.8,
                marker="o", markersize=2, zorder=3, label="Mean ADU")

        n_bins = int(good.sum())
        ax.set_xlabel("r²  (pixel²)", fontsize=10)
        ax.set_ylabel("Mean intensity  (ADU)", fontsize=10)
        ax.set_title(
            f"{stem}  —  radial profile vs r²  ({n_bins}/{len(fp.r2_grid)} bins)  |  "
            f"r_max = {fp.r_max_px:.0f} px",
            fontsize=9,
        )
        ax.legend(fontsize=8)

    _save_and_show(fig, save_path)


# ── Figure S3a — Profile with peaks marked (per frame) ───────────────────────

def _figure_s3a(
    fps: list,
    peaks_list: list,
    peak_arrs: list,
    stems: list[str],
    save_path: pathlib.Path,
) -> None:
    """
    One row per cal frame. Matches annular_reduction.py style:
    - ±2σ SEM shaded band behind profile
    - Red vertical lines at each peak centroid
    - Gaussian fit overlaid at each peak in family colour
    - Title shows peak count and status
    """
    n = len(fps)
    fig, axes = plt.subplots(n, 1, figsize=(14, 5 * n), squeeze=False)
    fig.suptitle("Stage 3a — Peak Detection", fontsize=12, fontweight="bold")

    for row, (fp, peaks, peak_arr, stem) in enumerate(
        zip(fps, peaks_list, peak_arrs, stems)
    ):
        ax = axes[row, 0]
        good = ~fp.masked
        r2   = fp.r2_grid[good]
        prof = fp.profile[good]
        sig  = fp.sigma_profile[good]
        sig  = np.where(sig > 0, sig, np.nan)

        bg_level = float(np.percentile(prof, 10))

        # Shaded SEM bands
        ax.fill_between(r2, prof - 2*sig, prof + 2*sig,
                        alpha=0.20, color="steelblue")
        ax.fill_between(r2, prof - sig, prof + sig,
                        alpha=0.35, color="steelblue")
        # Profile line
        ax.plot(r2, prof, color="steelblue", linewidth=0.8,
                marker="o", markersize=2, zorder=3, label="Profile")

        seen_labels: set = set()
        for i, p in enumerate(peaks):
            line_id  = peak_arr[i, 9]
            color    = "royalblue" if line_id == 0.0 else "darkorange"
            lam_str  = f"{_LAM_A_NM:.1f} nm" if line_id == 0.0 else f"{_LAM_B_NM:.1f} nm"

            # Red vertical centroid line
            ax.axvline(p.r2_fit_px2 if (p.fit_ok and np.isfinite(p.r2_fit_px2))
                       else p.r2_raw_px2,
                       color="red", linewidth=0.8, linestyle="-", alpha=0.6)

            # Dashed family colour line at raw position
            lbl = lam_str if lam_str not in seen_labels else None
            ax.axvline(p.r2_raw_px2, color=color, linewidth=0.6,
                       linestyle="--", alpha=0.5, label=lbl)
            if lbl:
                seen_labels.add(lam_str)

            # Gaussian fit overlay
            if p.fit_ok and np.isfinite(p.r2_fit_px2) and np.isfinite(p.width_r2_px2):
                r2_span = np.linspace(
                    p.r2_fit_px2 - 3 * p.width_r2_px2,
                    p.r2_fit_px2 + 3 * p.width_r2_px2,
                    80,
                )
                gauss = p.amplitude_adu * np.exp(
                    -0.5 * ((r2_span - p.r2_fit_px2) / p.width_r2_px2) ** 2
                )
                ax.plot(r2_span, gauss + bg_level, color=color,
                        linewidth=1.2, alpha=0.85, zorder=4)

        n_ok   = sum(1 for p in peaks if p.fit_ok)
        ax.set_xlabel("r²  (px²)", fontsize=9)
        ax.set_ylabel("Intensity  (ADU)", fontsize=9)
        ax.set_title(
            f"{stem} — radial profile with detected peaks  |  "
            f"{n_ok}/{len(peaks)} fits OK",
            fontsize=9,
        )
        ax.legend(fontsize=8)

    fig.tight_layout()
    _save_and_show(fig, save_path)


# ── Figure S3b — Individual peak fits (per frame) ────────────────────────────

def _figure_s3b(
    fp: "cal.FringeProfile",
    peaks: list,
    peak_arr: np.ndarray,
    stem: str,
    save_path: pathlib.Path,
    show: bool = True,
) -> None:
    """
    5×4 panel of individual peak Gaussian fits in r² domain.
    Window is tight around each peak — half the inter-peak spacing.
    Fit window shaded in green; initial guess shown in gold, LM fit in family colour.
    """
    n_peaks = len(peaks)
    ncols   = 4
    nrows   = max(1, (n_peaks + ncols - 1) // ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(9 * ncols, 3.5 * nrows))
    fig.suptitle(f"Stage 3b — Individual Peak Fits: {stem}",
                 fontsize=11, fontweight="bold")
    axes_flat = np.array(axes).ravel()

    r2_all   = fp.r2_grid
    prof_all = fp.profile
    sig_all  = fp.sigma_profile
    good_all = ~fp.masked

    # Estimate half inter-peak spacing once from median FSR
    r2_peaks = np.array([p.r2_fit_px2 if (p.fit_ok and np.isfinite(p.r2_fit_px2))
                         else p.r2_raw_px2 for p in peaks])
    same_fam_diffs = np.diff(r2_peaks[::2])          # every-other = same family
    half_fsr = float(np.median(same_fam_diffs)) / 2.0 if len(same_fam_diffs) else 300.0

    # Bin spacing in r² — used to reconstruct the fit window (default fit_half_window=6)
    r2_good = r2_all[good_all]
    r2_bin_size = float(np.median(np.diff(r2_good))) if len(r2_good) > 1 else 1.0
    fit_half_r2 = 6 * r2_bin_size  # matches fit_peaks(fit_half_window=6)

    for i, p in enumerate(peaks):
        ax = axes_flat[i]
        line_id   = peak_arr[i, 9]
        lam_str   = f"{_LAM_A_NM:.1f}" if line_id == 0.0 else f"{_LAM_B_NM:.1f}"
        fit_color = "royalblue" if line_id == 0.0 else "darkorange"

        # Tight window: ±half_fsr/2 around the peak centre
        peak_r2  = p.r2_fit_px2 if (p.fit_ok and np.isfinite(p.r2_fit_px2)) \
                   else p.r2_raw_px2
        r2_lo    = peak_r2 - half_fsr * 0.55
        r2_hi    = peak_r2 + half_fsr * 0.55
        mask_win = (r2_all >= r2_lo) & (r2_all <= r2_hi) & good_all
        r2_win   = r2_all[mask_win]
        pr_win   = prof_all[mask_win]
        sg_win   = sig_all[mask_win]
        valid    = np.isfinite(sg_win) & (sg_win > 0)

        bg_col = "#d4edda" if p.fit_ok else "#f8d7da"
        ax.set_facecolor(bg_col)

        # Fit window shading (drawn first so it sits behind data)
        fit_lo = p.r2_raw_px2 - fit_half_r2
        fit_hi = p.r2_raw_px2 + fit_half_r2
        ax.axvspan(fit_lo, fit_hi, alpha=0.20, color="limegreen", zorder=0,
                   label="Fit window" if i == 0 else None)

        if valid.any():
            ax.errorbar(r2_win[valid], pr_win[valid], yerr=sg_win[valid],
                        fmt=".", markersize=4, color="black", ecolor="gray",
                        linewidth=0.5, zorder=2, label="Data")

        bg = float(np.percentile(pr_win, 15)) if len(pr_win) else 0.0
        r2_fine = np.linspace(r2_lo, r2_hi, 120)

        # Initial guess — use the amplitude from the fit result (better than raw)
        amp_guess = p.amplitude_adu
        w_guess   = p.width_r2_px2 if (p.fit_ok and np.isfinite(p.width_r2_px2)) \
                    else half_fsr / 4.0
        guess = amp_guess * np.exp(
            -0.5 * ((r2_fine - p.r2_raw_px2) / max(w_guess, 1.0)) ** 2
        )
        ax.plot(r2_fine, guess + bg, "--", color="gold",
                linewidth=0.9, label="Guess", zorder=3)

        # LM fit curve
        if p.fit_ok and np.isfinite(p.r2_fit_px2) and np.isfinite(p.width_r2_px2):
            fit_curve = p.amplitude_adu * np.exp(
                -0.5 * ((r2_fine - p.r2_fit_px2) / p.width_r2_px2) ** 2
            ) + bg
            ax.plot(r2_fine, fit_curve, color=fit_color, linewidth=1.2,
                    label="Fit", zorder=4)

        chi2_str = f"{p.reduced_chi2:.2f}" if np.isfinite(p.reduced_chi2) else "—"
        r2f_str  = f"{p.r2_fit_px2:.1f}" if np.isfinite(p.r2_fit_px2) else "—"
        sr2_str  = f"{p.sigma_r2_fit_px2:.2f}" if np.isfinite(p.sigma_r2_fit_px2) else "—"
        ax.set_title(
            f"Peak {i}  λ={lam_str} nm\n"
            f"r²={r2f_str} ± {sr2_str}  χ²={chi2_str}",
            fontsize=7,
        )
        ax.set_xlabel("r²  (px²)", fontsize=6)
        ax.set_ylabel("ADU", fontsize=6)
        ax.tick_params(labelsize=5)
        ax.legend(fontsize=5)

    for j in range(n_peaks, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.tight_layout()
    _save_and_show(fig, save_path, show=show)


# ── Table S3 — Peak fit summary ───────────────────────────────────────────────

def _print_table_s3(peaks: list, peak_arr: np.ndarray) -> None:
    """Print per-peak Gaussian fit summary matching the annular_reduction.py table."""
    col_w = [5, 12, 16, 16, 14, 12, 12, 10, 10, 8]
    hdr = (f"{'Peak':>5}  {'λ (nm)':>8}  {'r²_raw (px²)':>13}  "
           f"{'r²_fit (px²)':>13}  {'+/-sig r² (px²)':>15}  "
           f"{'r_der (px)':>11}  {'+/-sig_r (px)':>13}  "
           f"{'Amp (ADU)':>10}  {'Width σr²':>10}  {'χ²_red':>7}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for i, p in enumerate(peaks):
        line_id = peak_arr[i, 9]
        lam     = _LAM_A_NM if line_id == 0.0 else _LAM_B_NM
        r2f     = p.r2_fit_px2        if np.isfinite(p.r2_fit_px2)        else float("nan")
        sr2     = p.sigma_r2_fit_px2  if np.isfinite(p.sigma_r2_fit_px2)  else float("nan")
        r_der   = np.sqrt(r2f)        if np.isfinite(r2f) and r2f >= 0   else float("nan")
        sig_r   = sr2 / (2 * r_der)   if (np.isfinite(sr2) and np.isfinite(r_der)
                                          and r_der > 0) else float("nan")
        wid     = p.width_r2_px2      if np.isfinite(p.width_r2_px2)      else float("nan")
        chi     = p.reduced_chi2      if np.isfinite(p.reduced_chi2)      else float("nan")
        print(
            f"{i+1:>5}  {lam:>8.4f}  {p.r2_raw_px2:>13.2f}  "
            f"{r2f:>13.3f}  {sr2:>15.3f}  "
            f"{r_der:>11.3f}  {sig_r:>13.3f}  "
            f"{p.amplitude_adu:>10.1f}  {wid:>10.2f}  {chi:>7.3f}"
        )


# ── Figure S5 — Seed averages ─────────────────────────────────────────────────

def _figure_s5(
    tol_results: list,
    seeds: "cal.TolanskySeedMean",
    chi2_threshold: float,
    save_path: pathlib.Path,
) -> None:
    """4-panel per-frame Tolansky seed values with orbit mean."""
    rejected = set()
    for i, r in enumerate(tol_results):
        if r.chi2_dof_a > chi2_threshold or r.chi2_dof_b > chi2_threshold:
            rejected.add(i)

    frames = np.arange(len(tol_results))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Stage 5 — Tolansky Seed Averages", fontsize=12, fontweight="bold")

    panels = [
        (axes[0, 0], [r.d_m * 1e3 for r in tol_results],
         seeds.d_m_mean * 1e3, seeds.sigma_d_m_mean * 1e3,
         "d_m (mm)", "Etalon gap"),
        (axes[0, 1], [r.alpha_mean for r in tol_results],
         seeds.alpha_mean, seeds.sigma_alpha_mean,
         "α (rad/px)", "Plate scale"),
        (axes[1, 0], [r.eps_a for r in tol_results],
         seeds.eps_a_mean, seeds.sigma_eps_a_mean,
         "ε_a", "Fractional order"),
        (axes[1, 1], [r.Y_B_obs for r in tol_results],
         seeds.Y_B_obs_mean, 0.0,
         "Y_B_obs", "Amplitude ratio"),
    ]

    for ax, vals, mean_val, sigma_val, ylabel, title in panels:
        for i, v in enumerate(vals):
            if i in rejected:
                ax.plot(i, v, "rx", markersize=10, markeredgewidth=2, zorder=3)
            else:
                ax.errorbar(i, v, fmt="o", color="steelblue", markersize=5, zorder=3)
        ax.axhline(mean_val, color="red", linewidth=1.5, linestyle="-",
                   label=f"Mean: {mean_val:.6g}")
        if sigma_val > 0:
            ax.axhspan(mean_val - sigma_val, mean_val + sigma_val,
                       alpha=0.15, color="red", label=f"±1σ: {sigma_val:.3g}")
        ax.set_xlabel("Cal frame index")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)

    fig.tight_layout()
    _save_and_show(fig, save_path)


# ── Table S5 — Seed averages ──────────────────────────────────────────────────

def _print_table_s5(
    tol_results: list,
    seeds: "cal.TolanskySeedMean",
    chi2_threshold: float,
) -> None:
    """Print per-frame and orbit-mean Tolansky seed values to stdout."""
    print(f"\n{'Frame':>6}  {'d_m (mm)':>12}  {'α (rad/px)':>14}  "
          f"{'ε_a':>10}  {'Y_B_obs':>10}  status")
    print("-" * 72)
    for i, r in enumerate(tol_results):
        ok = not (r.chi2_dof_a > chi2_threshold or r.chi2_dof_b > chi2_threshold)
        status = "used" if ok else "REJECTED"
        print(f"{i:>6}  {r.d_m * 1e3:>12.6f}  {r.alpha_mean:>14.8f}  "
              f"{r.eps_a:>10.6f}  {r.Y_B_obs:>10.4f}  {status}")
    print("-" * 72)
    print(f"{'MEAN':>6}  {seeds.d_m_mean * 1e3:>12.6f}  "
          f"{seeds.alpha_mean:>14.8f}  {seeds.eps_a_mean:>10.6f}  "
          f"{seeds.Y_B_obs_mean:>10.4f}  "
          f"(N={seeds.n_frames_used}/{seeds.n_frames_total})")
    print(f"{'±1σ':>6}  {seeds.sigma_d_m_mean * 1e3:>12.6f}  "
          f"{seeds.sigma_alpha_mean:>14.8f}  {seeds.sigma_eps_a_mean:>10.6f}")


# ── Figure S6 — H05 diagnostic ───────────────────────────────────────────────

def _figure_s6(
    fp: "cal.FringeProfile",
    fit: "cal.FitResult",
    save_path: pathlib.Path,
) -> None:
    """4-panel H05 calibration fit diagnostic."""
    good = ~fp.masked & np.isfinite(fp.sigma_profile) & (fp.sigma_profile > 0)
    r_good  = fp.r_grid[good]
    p_good  = fp.profile[good]
    s_good  = fp.sigma_profile[good]
    r2_good = fp.r2_grid[good]
    r_max   = float(fp.r_max_px)

    # Model components
    comp, lam1, lam2 = cal._model_components(
        r_good, r_max,
        fit.t_m, fit.alpha, fit.R1, fit.R2,
        fit.I0, fit.I1, fit.I2,
        fit.sigma0, fit.sigma1, fit.sigma2,
        fit.B, fit.ne_ratio,
    )

    fig = plt.figure(figsize=(14, 10))
    gs  = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)
    fig.suptitle("Stage 6 — H05 Calibration Fit Diagnostic",
                 fontsize=12, fontweight="bold")

    # [0] Data + composite + components
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(r2_good, p_good, ".", markersize=2, color="lightgray",
             label="Data", zorder=1)
    ax0.plot(r2_good, comp, color="red", linewidth=1.2,
             label="Composite model", zorder=4)
    ax0.plot(r2_good, lam1, "--", color="royalblue", linewidth=0.8,
             label=f"{_LAM_A_NM:.1f} nm + B", zorder=3)
    ax0.plot(r2_good, lam2, "--", color="darkorange", linewidth=0.8,
             label=f"{_LAM_B_NM:.1f} nm × ne_ratio + B", zorder=3)
    ax0.set_xlabel("r² (px²)")
    ax0.set_ylabel("Intensity (ADU)")
    ax0.set_title("Data + model")
    ax0.legend(fontsize=7)

    # [1] Residual
    ax1 = fig.add_subplot(gs[0, 1])
    resid = p_good - comp
    ax1.plot(r2_good, resid, ".", markersize=2, color="steelblue", label="Residual")
    # ±1σ Poisson band (sqrt of model counts)
    poisson_sigma = np.sqrt(np.maximum(comp, 1.0))
    ax1.fill_between(r2_good, -poisson_sigma, poisson_sigma,
                     alpha=0.2, color="gray", label="±1σ Poisson")
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.set_xlabel("r² (px²)")
    ax1.set_ylabel("Residual (ADU)")
    ax1.set_title(f"Residual  (χ²_red = {fit.chi2_reduced:.3f})")
    ax1.legend(fontsize=7)

    # [2] Per-stage χ²_red bar chart
    ax2 = fig.add_subplot(gs[1, 0])
    stages = np.arange(1, len(fit.chi2_by_stage) + 1)
    colors = [
        "green" if 0.5 <= c <= 2.0 else "red"
        for c in fit.chi2_by_stage
    ]
    ax2.bar(stages, fit.chi2_by_stage, color=colors, edgecolor="black",
            linewidth=0.5)
    ax2.axhspan(0.5, 2.0, alpha=0.12, color="green", label="Accept [0.5, 2.0]")
    ax2.set_xlabel("Stage")
    ax2.set_ylabel("χ²_red")
    ax2.set_title("Per-stage χ²_red")
    ax2.legend(fontsize=7)

    # [3] Parameter table
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")
    chi2_ok = 0.5 <= fit.chi2_reduced <= 2.0

    param_lines = [
        f"t_m        = {fit.t_m * 1e3:.6f} ± {fit.sigma_t_m * 1e3:.6f} mm",
        f"alpha      = {fit.alpha:.8f} ± {fit.sigma_alpha:.8f} rad/px",
        f"R1         = {fit.R1:.4f} ± {fit.sigma_R1:.4f}",
        f"R2         = {fit.R2:.4f} ± {fit.sigma_R2:.4f}",
        f"I0         = {fit.I0:.1f} ± {fit.sigma_I0:.1f} ADU",
        f"I1         = {fit.I1:.4f} ± {fit.sigma_I1:.4f}",
        f"I2         = {fit.I2:.4f} ± {fit.sigma_I2:.4f}",
        f"sigma0     = {fit.sigma0:.4f} ± {fit.sigma_sigma0:.4f} px",
        f"B          = {fit.B:.1f} ± {fit.sigma_B:.1f} ADU",
        f"ne_ratio   = {fit.ne_ratio:.4f} ± {fit.sigma_ne_ratio:.4f}",
        f"ε_cal      = {fit.epsilon_cal:.6f} ± {fit.sigma_epsilon_cal:.6f}",
        f"χ²_red     = {fit.chi2_reduced:.4f}  ({'OK' if chi2_ok else 'WARN'})",
        f"N_bins     = {fit.n_bins_used}",
        f"Converged  = {fit.converged}",
    ]
    txt = "\n".join(param_lines)
    ax3.text(
        0.05, 0.95, txt,
        transform=ax3.transAxes,
        va="top", ha="left",
        fontsize=8,
        fontfamily="monospace",
        color="darkgreen" if chi2_ok else "darkred",
    )
    ax3.set_title("Fitted parameters (all 10 free + ε)")

    _save_and_show(fig, save_path)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main() -> None:
    """Run the calibration pipeline stages S0–S6."""

    # ── Stage 0: Master dark frame ────────────────────────────────────────────
    stage_banner(0, "Master dark frame")

    dark_paths = select_files("Select dark .bin frames")
    print(f"  {len(dark_paths)} dark frame(s) selected.")

    raw_stack = np.stack([_load_bin_frame(p) for p in dark_paths])
    master_dark, dark_sigma = cal.make_master_dark(dark_paths)

    cal_dir = dark_paths[0].parent
    dark_npy = cal_dir / "master_dark.npy"
    np.save(dark_npy, master_dark)
    print(f"  Saved: {dark_npy}")

    def _s0():
        _figure_s0a(raw_stack, dark_paths, cal_dir)
        _figure_s0b(master_dark, dark_sigma, dark_paths, dark_npy,
                    cal_dir / "S0b_master_dark.png")
    run_with_error_handling(0, _s0)

    # ── Stage 1: Cal dark subtraction and centre finding ──────────────────────
    stage_banner(1, "Cal dark subtraction and centre finding")

    cal_paths = select_files("Select neon cal .bin frames")
    print(f"  {len(cal_paths)} cal frame(s) selected.")

    # Phase 1a: load + dark subtract all frames (fast — no scipy optimisation yet)
    raw_frames: list[np.ndarray] = []
    ds_frames:  list[np.ndarray] = []
    for path in cal_paths:
        raw = _load_bin_frame(path)
        ds  = cal.dark_subtract(raw, master_dark)
        raw_frames.append(raw)
        ds_frames.append(ds)
        print(f"  Dark-subtracted: {path.name}")

    # Phase 1b: centre finding per frame (slower — Nelder-Mead)
    _FALLBACK_CTR = cal.CentreResult(
        cx=138.0, cy=130.0,
        sigma_cx=1.0, sigma_cy=1.0,
        two_sigma_cx=2.0, two_sigma_cy=2.0,
        cost_at_min=0.0,
        grid_cx=138.0, grid_cy=130.0, grid_cost=0.0,
    )
    centres: list[cal.CentreResult] = []

    for raw, ds, path in zip(raw_frames, ds_frames, cal_paths):
        stem = path.stem
        print(f"  Finding centre: {path.name}")

        def _s1_centre(raw=raw, ds=ds, path=path, stem=stem):
            ctr = cal.find_centre(ds)
            np.savez(
                path.parent / f"{stem}_center.npz",
                cx=np.array(ctr.cx),
                cy=np.array(ctr.cy),
                sigma_cx=np.array(ctr.sigma_cx),
                sigma_cy=np.array(ctr.sigma_cy),
            )
            print(f"    cx={ctr.cx:.3f}  cy={ctr.cy:.3f}  "
                  f"σ_cx={ctr.sigma_cx:.3f}  σ_cy={ctr.sigma_cy:.3f}")
            _figure_s1(raw, ds, ctr, stem,
                       path.parent / f"{stem}_S1_centre.png")
            return ctr

        result = run_with_error_handling(1, _s1_centre)
        centres.append(result if result is not None else _FALLBACK_CTR)

    # Combined gallery — all cals / dark-subtracted / histograms in one figure
    def _s1_gallery():
        _figure_s1_cal_gallery(
            raw_frames, ds_frames, centres, cal_paths, dark_npy,
            cal_dir / "S1_cal_gallery.png",
        )
    run_with_error_handling(1, _s1_gallery)

    # ── Stage 2: Annular integration ─────────────────────────────────────────
    stage_banner(2, "Annular integration → radial profile")

    profiles: list[cal.FringeProfile] = []

    for ds, ctr, path in zip(ds_frames, centres, cal_paths):
        stem = path.stem
        print(f"  Reducing: {path.name}")

        def _s2(ds=ds, ctr=ctr, path=path, stem=stem):
            fp = cal.build_radial_profile(ds, ctr.cx, ctr.cy)

            np.savez(
                path.parent / f"{stem}_L1.2.npz",
                r_grid=fp.r_grid,
                r2_grid=fp.r2_grid,
                profile=fp.profile,
                sigma_profile=fp.sigma_profile,
                masked=fp.masked,
                cx=np.array(fp.cx),
                cy=np.array(fp.cy),
            )
            np.save(
                path.parent / f"{stem}_profile_vs_r2.npy",
                np.stack([fp.r2_grid, fp.profile, fp.sigma_profile]),
            )
            print(f"    Bins: {fp.n_bins}  r_max: {fp.r_max_px:.1f} px  "
                  f"sparse: {fp.sparse_bins}")
            return fp

        result = run_with_error_handling(2, _s2)
        profiles.append(result)

    # Combined S2 figure — all frames in one column-per-row layout
    def _s2_combined():
        valid = [
            (fp, path.stem)
            for fp, path in zip(profiles, cal_paths)
            if fp is not None
        ]
        if valid:
            fps_v, stems_v = zip(*valid)
            _figure_s2(list(fps_v), list(stems_v), cal_dir / "S2_combined.png")
    run_with_error_handling(2, _s2_combined)

    # ── Stage 3: Peak finding and Gaussian fits ───────────────────────────────
    stage_banner(3, "Peak finding and Gaussian fits")

    peak_arrays: list[np.ndarray] = []
    peaks_list:  list             = []
    s3b_shown   = [False]  # show only the first S3b figure

    for fp, path in zip(profiles, cal_paths):
        if fp is None:
            peak_arrays.append(None)
            peaks_list.append(None)
            continue
        stem = path.stem
        print(f"  Fitting peaks: {path.name}")

        def _s3(fp=fp, path=path, stem=stem, s3b_shown=s3b_shown):
            peaks    = cal.fit_peaks(fp)
            peak_arr = cal.peaks_to_array(peaks)

            np.save(path.parent / f"{stem}_peak_fits_r2.npy", peak_arr)
            print(f"    {len(peaks)} peaks detected.")
            _print_table_s3(peaks, peak_arr)

            show_s3b = not s3b_shown[0]
            _figure_s3b(fp, peaks, peak_arr, stem,
                        path.parent / f"{stem}_fig_S3b.png",
                        show=show_s3b)
            s3b_shown[0] = True
            return peaks, peak_arr

        result = run_with_error_handling(3, _s3)
        if result is None:
            peak_arrays.append(None)
            peaks_list.append(None)
        else:
            peak_arrays.append(result[1])
            peaks_list.append(result[0])

    # Combined S3a figure — all frames in one column-per-row layout
    def _s3a_combined():
        valid = [
            (fp, pks, pa, path.stem)
            for fp, pks, pa, path in zip(profiles, peaks_list, peak_arrays, cal_paths)
            if fp is not None and pks is not None and pa is not None
        ]
        if valid:
            fps_v, pks_v, pas_v, stems_v = zip(*valid)
            _figure_s3a(
                list(fps_v), list(pks_v), list(pas_v), list(stems_v),
                cal_dir / "S3a_combined.png",
            )
    run_with_error_handling(3, _s3a_combined)

    # ── Stage 4: Tolansky two-line WLS ───────────────────────────────────────
    stage_banner(4, "Tolansky two-line WLS fit")

    # Ask for ring pairs before starting
    n_valid_peaks = next(
        (len(pa) for pa in peak_arrays if pa is not None), 20
    )
    n_pairs = ask_integer(
        "Number of ring pairs (one per neon line) for Tolansky WLS:",
        "Tolansky ring pairs",
        default=n_valid_peaks // 2,
    )
    print(f"  Using n_pairs = {n_pairs}")

    tol_results: list = []

    for peak_arr, path in zip(peak_arrays, cal_paths):
        if peak_arr is None:
            tol_results.append(None)
            continue
        stem = path.stem
        print(f"  Tolansky: {path.name}")

        def _s4(peak_arr=peak_arr, path=path, stem=stem):
            tol = cal.run_tolansky(peak_arr, n_pairs=n_pairs)

            tol_dict = {
                k: getattr(tol, k)
                for k in tol.__dataclass_fields__
            }
            np.save(path.parent / f"{stem}_tolansky.npy",
                    tol_dict, allow_pickle=True)

            print(f"    d_m={tol.d_m * 1e3:.5f} mm  "
                  f"α={tol.alpha_mean:.8f} rad/px  "
                  f"ε_a={tol.eps_a:.6f}")

            tol_fig_path = path.parent / f"{stem}_fig_S4.png"
            cal.plot_tolansky_result(tol, save_path=tol_fig_path)
            plt.close("all")
            print(f"  Saved: {tol_fig_path}")
            print(f"  Saved: {tol_fig_path.name}")
            plt.show(block=True)
            plt.close("all")
            return tol

        result = run_with_error_handling(4, _s4)
        tol_results.append(result)

    # Filter out None results
    valid_tol = [r for r in tol_results if r is not None]
    if not valid_tol:
        print("ERROR: No valid Tolansky results. Aborting.")
        sys.exit(1)

    # ── Stage 5: Seed averaging ───────────────────────────────────────────────
    stage_banner(5, "Seed averaging")
    CHI2_THRESH = 5.0

    def _s5():
        seeds = cal.average_tolansky_seeds(valid_tol, chi2_threshold=CHI2_THRESH)
        np.save(cal_dir / "orbit_seeds_mean.npy", seeds.__dict__, allow_pickle=True)
        _print_table_s5(valid_tol, seeds, CHI2_THRESH)
        _figure_s5(
            valid_tol, seeds, CHI2_THRESH,
            cal_dir / "fig_S5_seeds.png",
        )
        return seeds

    seeds = run_with_error_handling(5, _s5)
    if seeds is None:
        print("ERROR: Seed averaging failed. Aborting.")
        sys.exit(1)

    print(f"\n  d_m_mean   = {seeds.d_m_mean * 1e3:.6f} ± "
          f"{seeds.sigma_d_m_mean * 1e3:.6f} mm")
    print(f"  alpha_mean = {seeds.alpha_mean:.8f} ± "
          f"{seeds.sigma_alpha_mean:.8f} rad/px")
    print(f"  eps_a_mean = {seeds.eps_a_mean:.6f} ± "
          f"{seeds.sigma_eps_a_mean:.6f}")
    print(f"  Y_B_obs    = {seeds.Y_B_obs_mean:.4f}")

    # ── Stage 6: H05 full Harding LM calibration fit ─────────────────────────
    stage_banner(6, "H05 full Harding LM calibration fit")

    n_cal = len(profiles)
    idx_1based = ask_integer(
        f"Cal frame index for H05 fit (1–{n_cal}):",
        "Cal frame index for H05 fit (1–N)",
        default=1,
        minvalue=1,
    )
    idx = idx_1based - 1
    print(f"  Using cal frame {idx_1based}: {cal_paths[idx].name}")

    def _s6():
        fp_sel = profiles[idx]
        if fp_sel is None:
            raise ValueError(f"Profile for frame {idx_1based} is not available.")

        fit = cal.run_h05(fp_sel, seeds)

        orbit_cal = cal.OrbitCalResult(
            seeds=seeds,
            fit=fit,
            source_cal_files=[str(p) for p in cal_paths],
            source_dark_files=[str(p) for p in dark_paths],
            date_utc=datetime.datetime.utcnow().isoformat(),
        )
        np.save(
            cal_dir / "orbit_cal_result.npy",
            {
                "seeds": seeds.__dict__,
                "fit": fit.__dict__,
                "source_cal_files": orbit_cal.source_cal_files,
                "source_dark_files": orbit_cal.source_dark_files,
                "date_utc": orbit_cal.date_utc,
                "pipeline_version": orbit_cal.pipeline_version,
            },
            allow_pickle=True,
        )
        print(f"\n  Saved: {cal_dir / 'orbit_cal_result.npy'}")
        print(f"  χ²_red = {fit.chi2_reduced:.4f}  converged = {fit.converged}")

        _figure_s6(fp_sel, fit, cal_dir / "fig_S6_h05.png")
        return orbit_cal

    orbit_cal = run_with_error_handling(6, _s6)

    # ── Done ──────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  CALIBRATION PIPELINE COMPLETE")
    print(f"  Output: {cal_dir / 'orbit_cal_result.npy'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
