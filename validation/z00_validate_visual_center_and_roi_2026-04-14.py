"""
Z01 — Validate Tolansky Analysis (1-Line and 2-Line Fringe Images)
WindCube FPI Pipeline — NCAR / High Altitude Observatory (HAO)
Spec: specs/Z01_validate_tolansky_analysis_2026-04-13.md
Supersedes: Z01 v0.2 (2026-04-12), Z01a v0.2 (2026-04-12)

Usage:
  python z01_validate_tolansky_analysis_2026-04-13.py [--mode {1,2}] [OPTIONS]

Modes:
  2 (default): dual neon 640.2248 nm (air) + 638.2991 nm (air)
  1:           single-line OI 630.0304 nm (air) or filtered neon

Tool: Claude Code
Last updated: 2026-04-13
"""

# ── Standard library ──────────────────────────────────────────────────────────
import argparse
import pathlib
import sys
import tkinter as tk
from tkinter import filedialog

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import scipy.optimize as opt
from datetime import datetime

# ── Repo root on path ─────────────────────────────────────────────────────────
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))


# ── Stage A ───────────────────────────────────────────────────────────────────

def load_headerless_bin(path: pathlib.Path,
                        shape: tuple) -> np.ndarray:
    """
    Load a raw headerless big-endian uint16 binary image and return as float64.

    Parameters
    ----------
    path : pathlib.Path
        Path to the .bin file.
    shape : tuple
        (rows, cols) of the image.

    Returns
    -------
    np.ndarray
        2D float64 array, shape == shape.
    """
    expected_bytes = shape[0] * shape[1] * 2
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"File size mismatch: expected {expected_bytes} bytes for shape "
            f"{shape}, got {actual_bytes} bytes: {path}"
        )
    raw = np.fromfile(str(path), dtype=">u2")
    return raw.reshape(shape).astype(np.float64)


def load_images(has_header: str = "auto",
                image_shape: tuple = (256, 256)) -> dict:
    """
    Prompt the user via Tkinter file dialogs to select calibration and dark images.

    Returns a dict with keys:
      cal_image, dark_image, cal_path, dark_path, cal_meta, dark_meta
    """
    from src.metadata.p01_image_metadata_2026_04_06 import ingest_real_image

    root = tk.Tk()
    root.withdraw()

    _FILT = [("WindCube images", "*.bin *.npy"), ("All files", "*.*")]

    cal_path_str = filedialog.askopenfilename(
        title="Select WindCube Calibration Image",
        filetypes=_FILT,
    )
    if not cal_path_str:
        print("User cancelled file selection.")
        sys.exit(0)
    cal_path = pathlib.Path(cal_path_str)

    dark_path_str = filedialog.askopenfilename(
        title="Select WindCube Dark Image",
        filetypes=_FILT,
    )
    if not dark_path_str:
        print("User cancelled file selection.")
        sys.exit(0)
    dark_path = pathlib.Path(dark_path_str)

    root.destroy()

    _WINDCUBE_HEADER_BYTES = 260 * 276 * 2  # 143,520 bytes

    def _load_one(p: pathlib.Path):
        meta = None
        if p.suffix.lower() == ".npy":
            arr = np.load(str(p))
            if arr.ndim != 2:
                raise ValueError(
                    f"Expected 2D array from {p.name}, got shape {arr.shape}"
                )
            return arr.astype(np.float64), meta

        if has_header == "yes":
            meta, arr = ingest_real_image(p)
        elif has_header == "no":
            arr = load_headerless_bin(p, image_shape)
        else:  # auto
            size = p.stat().st_size
            if size == _WINDCUBE_HEADER_BYTES:
                meta, arr = ingest_real_image(p)
            else:
                arr = load_headerless_bin(p, image_shape)

        return arr.astype(np.float64), meta

    cal_image, cal_meta = _load_one(cal_path)
    dark_image, dark_meta = _load_one(dark_path)

    if cal_image.shape != dark_image.shape:
        raise ValueError(
            f"Calibration image shape {cal_image.shape} != "
            f"dark image shape {dark_image.shape}"
        )

    return {
        "cal_image":  cal_image,
        "dark_image": dark_image,
        "cal_path":   cal_path,
        "dark_path":  dark_path,
        "cal_meta":   cal_meta,
        "dark_meta":  dark_meta,
    }


# ── Stage B ───────────────────────────────────────────────────────────────────

def extract_metadata(load_result: dict) -> dict:
    """Pass-through: extract metadata fields from load result."""
    return {
        "cal_meta":  load_result.get("cal_meta"),
        "dark_meta": load_result.get("dark_meta"),
    }


# ── Stage C — Figure 1 ────────────────────────────────────────────────────────

_DEFAULT_ROI_SIZE = 210   # fallback if user skips the second click


def figure_image_pair(load_result: dict,
                      meta_result: dict,
                      mode: int,
                      lam_str: str) -> tuple:
    """
    Display calibration and dark images side-by-side (Figure 1).

    Two-click interaction in the calibration image:
      Click 1 — fringe centre seed  (red cross)
      Click 2 — ROI edge point      (blue cross + square preview)
                The ROI half-size is the Chebyshev distance from click 1
                to click 2, i.e.  ROI_SIZE = 2 * max(|Δx|, |Δy|),
                rounded to the nearest even integer.

    Returns
    -------
    (cx_seed, cy_seed, roi_size) : (float, float, int)
        cx_seed, cy_seed — fringe centre in full-image pixel coordinates
        roi_size         — side length of the square ROI to extract (px)
    """
    cal_image  = load_result["cal_image"]
    dark_image = load_result["dark_image"]
    cal_path   = load_result["cal_path"]

    vmax = float(np.percentile(cal_image, 99.5))

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    ax_cal, ax_dark = axes

    ax_cal.imshow(cal_image, cmap="gray", vmin=0, vmax=vmax, origin="upper")
    ax_cal.set_title(f"Calibration: {cal_path.name}\nλ = {lam_str}", fontsize=9)
    ax_cal.set_xlabel("Column (px)"); ax_cal.set_ylabel("Row (px)")

    ax_dark.imshow(dark_image, cmap="gray",
                   vmin=0, vmax=np.percentile(dark_image, 99.5), origin="upper")
    ax_dark.set_title(f"Dark: {load_result['dark_path'].name}", fontsize=9)
    ax_dark.set_xlabel("Column (px)"); ax_dark.set_ylabel("Row (px)")

    h, w = cal_image.shape
    cx_seed_def, cy_seed_def = w / 2.0, h / 2.0

    clicks = []                              # accumulates (x, y) in order
    _roi_rect: list[mpatches.Rectangle | None] = [None]  # preview patch

    def _update_title():
        n = len(clicks)
        if n == 0:
            msg = "Click 1: fringe centre"
        elif n == 1:
            msg = (f"Centre: ({clicks[0][0]:.1f}, {clicks[0][1]:.1f})  |  "
                   f"Click 2: ROI edge point")
        else:
            half = max(abs(clicks[1][0] - clicks[0][0]),
                       abs(clicks[1][1] - clicks[0][1]))
            size = 2 * max(1, round(half))
            msg = (f"Centre: ({clicks[0][0]:.1f}, {clicks[0][1]:.1f})  |  "
                   f"ROI size: {size} × {size} px  |  [close to confirm]")
        fig.suptitle(f"Figure 1 — Image Pair  |  {msg}",
                     fontsize=10, fontweight="bold")
        fig.canvas.draw_idle()

    def _onclick(event):
        if event.inaxes is not ax_cal or event.button != 1:
            return
        x, y = float(event.xdata), float(event.ydata)
        clicks.append((x, y))

        if len(clicks) == 1:
            # First click: draw centre cross
            ax_cal.plot(x, y, "r+", markersize=14, mew=2)

        elif len(clicks) == 2:
            # Second click: draw edge cross and square preview
            ax_cal.plot(x, y, "b+", markersize=12, mew=2)
            cx, cy = clicks[0]
            half = max(abs(x - cx), abs(y - cy))
            size = 2 * max(1, round(half))
            # Remove previous rectangle if re-clicking
            if _roi_rect[0] is not None:
                _roi_rect[0].remove()
            rect = mpatches.Rectangle(
                (cx - size / 2, cy - size / 2), size, size,
                linewidth=1.5, edgecolor="cyan", facecolor="none",
                linestyle="--",
            )
            ax_cal.add_patch(rect)
            _roi_rect[0] = rect

        else:
            # Any further click resets and starts over
            clicks.clear()
            if _roi_rect[0] is not None:
                _roi_rect[0].remove()
                _roi_rect[0] = None
            # Clear all artists added after the image
            while len(ax_cal.lines) > 0:
                ax_cal.lines[-1].remove()
            while len(ax_cal.patches) > 0:
                ax_cal.patches[-1].remove()
            clicks.append((x, y))
            ax_cal.plot(x, y, "r+", markersize=14, mew=2)

        _update_title()

    cid = fig.canvas.mpl_connect("button_press_event", _onclick)
    _update_title()
    plt.tight_layout()
    plt.show()
    fig.canvas.mpl_disconnect(cid)

    # Resolve centre
    if clicks:
        cx_seed, cy_seed = clicks[0]
    else:
        cx_seed, cy_seed = cx_seed_def, cy_seed_def
        print(f"  No click detected — using image centre "
              f"({cx_seed:.1f}, {cy_seed:.1f}) px as seed.")

    # Resolve ROI size
    if len(clicks) >= 2:
        half = max(abs(clicks[1][0] - cx_seed), abs(clicks[1][1] - cy_seed))
        roi_size = 2 * max(1, round(half))
        # Round up to next even number so ROI_HALF is an integer
        if roi_size % 2 != 0:
            roi_size += 1
    else:
        roi_size = _DEFAULT_ROI_SIZE
        print(f"  No ROI edge click — using default ROI size {roi_size} px.")

    return cx_seed, cy_seed, roi_size


# ── Stage D — Figure 2 ────────────────────────────────────────────────────────

def figure_roi_inspection(cal_image: np.ndarray,
                          dark_image: np.ndarray,
                          cx_seed: float,
                          cy_seed: float,
                          roi_size: int = _DEFAULT_ROI_SIZE,
                          cal_path: pathlib.Path | None = None,
                          dark_path: pathlib.Path | None = None,
                          save_dir: pathlib.Path | None = None) -> tuple:
    """Show a zoomed ROI around the fringe centre (Figure 2).

    Extracts a square ROI of roi_size × roi_size pixels centred on the
    user-selected fringe centre.  If the centre lies within roi_size//2 px
    of an image edge the box is shifted inward so the full extent is always
    available (no clamping artefacts).

    Additionally, saves the calibration and dark ROIs as .npy arrays when
    `save_dir` is provided (or use the parent folder of the calibration
    image otherwise).

    Returns
    -------
    (cal_roi, dark_roi) : tuple of np.ndarray, each (roi_size, roi_size)
    """
    ROI_SIZE = roi_size
    ROI_HALF = ROI_SIZE // 2

    h, w = cal_image.shape
    cx_int = int(round(cx_seed))
    cy_int = int(round(cy_seed))

    # Clamp start so that start + ROI_SIZE never exceeds the image boundary.
    x0 = max(0, min(cx_int - ROI_HALF, w - ROI_SIZE))
    y0 = max(0, min(cy_int - ROI_HALF, h - ROI_SIZE))
    x1 = x0 + ROI_SIZE
    y1 = y0 + ROI_SIZE

    cal_roi  = cal_image[y0:y1, x0:x1]
    dark_roi = dark_image[y0:y1, x0:x1]
    vmax = float(np.percentile(cal_roi, 99.5))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, img, title in zip(axes,
                               [cal_roi, dark_roi],
                               ["Calibration ROI (raw)", "Dark ROI (raw)"]):
        ax.imshow(img, cmap="gray", vmin=0, vmax=vmax, origin="upper")
        ax.set_title(title, fontsize=9)
        # Draw the seed cross
        ax.plot(cx_seed - x0, cy_seed - y0, "r+", markersize=16, mew=2)
        # Draw a circle at the ROI half-radius for reference
        theta = np.linspace(0, 2 * np.pi, 360)
        ax.plot(ROI_HALF * np.cos(theta) + (cx_seed - x0),
                ROI_HALF * np.sin(theta) + (cy_seed - y0),
                "r--", lw=1, alpha=0.7)
        ax.set_xlabel("Column"); ax.set_ylabel("Row")

    fig.suptitle(
        f"Figure 2 — ROI Inspection  |  Seed: ({cx_seed:.1f}, {cy_seed:.1f}) px  |  "
        f"ROI: {ROI_SIZE} × {ROI_SIZE} px  |  [close to continue]",
        fontsize=10, fontweight="bold"
    )
    plt.tight_layout()
    plt.show()

    # Save ROIs as .npy for downstream inspection/archival
    try:
        # Destination directory: explicit save_dir if given, otherwise use
        # the parent folder of the calibration image if available, else CWD.
        if save_dir is not None:
            out_dir = pathlib.Path(save_dir)
        elif cal_path is not None:
            out_dir = pathlib.Path(cal_path).parent
        else:
            out_dir = pathlib.Path.cwd()

        out_dir.mkdir(parents=True, exist_ok=True)

        def _roi_name_from_image(p: pathlib.Path) -> pathlib.Path:
            name = p.name
            if "." in name:
                base, ext = name.rsplit(".", 1)
                return out_dir / f"{base}_roi.npy"
            else:
                return out_dir / f"{name}_roi.npy"

        cal_fname = _roi_name_from_image(pathlib.Path(cal_path)) if cal_path is not None else (
            out_dir / "cal_roi.npy"
        )
        dark_fname = _roi_name_from_image(pathlib.Path(dark_path)) if dark_path is not None else (
            out_dir / "dark_roi.npy"
        )

        np.save(cal_fname, cal_roi.astype(np.float32))
        np.save(dark_fname, dark_roi.astype(np.float32))
        print(f"  Saved ROI arrays:\n    {cal_fname}\n    {dark_fname}")
    except Exception as exc:  # pragma: no cover - best-effort save
        print(f"  Warning: failed to save ROI arrays: {exc}")

    return cal_roi, dark_roi


# ── CLI and entry point ───────────────────────────────────────────────────────

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Z00 — WindCube FPI visual centre and ROI extraction"
    )
    parser.add_argument(
        "--has-header", choices=["yes", "no", "auto"], default="auto",
        help="Whether the binary image has a WindCube header. Default: auto."
    )
    return parser.parse_args()


def main():
    """Run Stages A–D: load images, show pair, click centre, inspect and save ROI."""
    args = parse_args()

    print("=" * 70)
    print("  WindCube FPI — Z00: Visual Centre and ROI Extraction")
    print("=" * 70)

    # Stage A — load images
    print("\nStage A: Loading images...")
    load = load_images(has_header=args.has_header)
    print(f"  CAL:  {load['cal_path'].name}  shape={load['cal_image'].shape}")
    print(f"  DARK: {load['dark_path'].name}  shape={load['dark_image'].shape}")

    # Stage B — extract metadata
    meta = extract_metadata(load)

    # Stage C — Figure 1: image pair, click centre then ROI edge
    print("\nStage C: Image pair (Figure 1)...")
    cx_seed, cy_seed, roi_size = figure_image_pair(
        load, meta, mode=2, lam_str="neon calibration"
    )
    print(f"  Fringe centre seed: ({cx_seed:.1f}, {cy_seed:.1f}) px")
    print(f"  ROI size selected:  {roi_size} × {roi_size} px")

    # Stage D — Figure 2: ROI inspection and save npy arrays
    print("\nStage D: ROI inspection and save (Figure 2)...")
    cal_roi, dark_roi = figure_roi_inspection(
        load["cal_image"], load["dark_image"],
        cx_seed, cy_seed, roi_size=roi_size,
        cal_path=load["cal_path"],
        dark_path=load["dark_path"],
    )
    print(f"  cal_roi  shape: {cal_roi.shape}")
    print(f"  dark_roi shape: {dark_roi.shape}")
    print("\nDone.")


if __name__ == "__main__":
    main()
