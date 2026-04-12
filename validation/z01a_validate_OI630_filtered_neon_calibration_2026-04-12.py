"""
Z01a — Validate OI 630 nm Filtered Neon Lamp Calibration Images
WindCube FPI Pipeline — NCAR / High Altitude Observatory (HAO)
Spec: docs/specs/Z01a_validate_OI630_filtered_neon_calibration_2026-04-12.md
Derived from: Z01 (Z01_validate_calibration_using_real_images_2026-04-12.md)
Source type: Ne lamp + 630 nm filter + 1.65° baffled field stop
Zero-velocity reference: v_rel = 0 by construction
Tool: Claude Code
Last updated: 2026-04-12
"""

import math
import pathlib
import sys
import tkinter as tk
from tkinter import filedialog

# Ensure the project root (soc_sewell/) is on sys.path regardless of where
# the script is invoked from.
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from src.fpi.m03_annular_reduction_2026_04_06 import (
    FringeProfile,
    make_master_dark,
    reduce_calibration_frame,
    subtract_dark,
)
from src.fpi.tolansky_2026_04_05 import SingleLineTolansky, SingleLineResult
from windcube.constants import (
    OI_WAVELENGTH_NM,
    TOLANSKY_D_MM,
    TOLANSKY_F_MM,
    CCD_PIXEL_PITCH_M,
    ICOS_GAP_MM,
)


# ---------------------------------------------------------------------------
# Helper — load_headerless_bin  (spec §4.3)
# ---------------------------------------------------------------------------

def load_headerless_bin(path: pathlib.Path, shape: tuple) -> np.ndarray:
    """
    Load a headerless WindCube .bin file as a 2D float64 array.

    The file contains shape[0] × shape[1] big-endian uint16 values with no
    header row.  Total expected file size = shape[0] * shape[1] * 2 bytes.

    Raises ValueError if the file size does not match the expected shape.
    """
    data = np.frombuffer(path.read_bytes(), dtype=">u2").astype(np.float64)
    expected = shape[0] * shape[1]
    if data.size != expected:
        raise ValueError(
            f"load_headerless_bin: expected {expected} pixels "
            f"({shape[0]}×{shape[1]}) but got {data.size} "
            f"in {path.name}"
        )
    return data.reshape(shape)


# ---------------------------------------------------------------------------
# Stage A — load_images  (Z01a variant)
# ---------------------------------------------------------------------------

def load_images(image_shape: tuple = (256, 256)) -> dict:
    """
    Load a headerless calibration image and dark image via file dialogs.

    Parameters
    ----------
    image_shape : (rows, cols) expected shape. Used only for .bin files.
                  Default (256, 256).

    Returns
    -------
    dict with keys:
        'cal_image'  : np.ndarray, float64, shape image_shape
        'dark_image' : np.ndarray, float64, shape image_shape
        'cal_path'   : pathlib.Path
        'dark_path'  : pathlib.Path
        'cal_type'   : str, 'headerless' | 'synthetic'
        'dark_type'  : str, 'headerless' | 'synthetic'
        'cal_raw'    : np.ndarray uint16, shape image_shape, or None for .npy
        'dark_raw'   : np.ndarray uint16 or None
    """
    root = tk.Tk()
    root.withdraw()

    file_types = [("WindCube images", "*.bin *.npy"), ("All files", "*.*")]

    cal_path_str = filedialog.askopenfilename(
        title="Select OI 630 nm Calibration Image",
        filetypes=file_types,
    )
    if not cal_path_str:
        print("No calibration image selected. Exiting.")
        sys.exit(0)

    dark_path_str = filedialog.askopenfilename(
        title="Select Dark Image",
        filetypes=file_types,
    )
    if not dark_path_str:
        print("No dark image selected. Exiting.")
        sys.exit(0)

    root.destroy()

    cal_path  = pathlib.Path(cal_path_str)
    dark_path = pathlib.Path(dark_path_str)

    def _load_file(path):
        """Load a .bin or .npy file. Return (pixel_array, raw_array, type_str)."""
        suffix = path.suffix.lower()
        if suffix == ".bin":
            pixels = load_headerless_bin(path, image_shape)
            raw = np.frombuffer(path.read_bytes(), dtype=">u2").reshape(image_shape)
            return pixels, raw, "headerless"
        elif suffix == ".npy":
            arr = np.load(path)
            if arr.ndim != 2:
                raise ValueError(
                    f"NumPy file {path.name} has shape {arr.shape}; expected 2D array."
                )
            return arr.astype(np.float64), None, "synthetic"
        else:
            raise ValueError(f"Unsupported file type '{suffix}' for {path.name}")

    cal_image, cal_raw, cal_type   = _load_file(cal_path)
    dark_image, dark_raw, dark_type = _load_file(dark_path)

    if cal_image.shape != dark_image.shape:
        raise ValueError(
            f"Calibration image shape {cal_image.shape} does not match "
            f"dark image shape {dark_image.shape}."
        )

    return {
        "cal_image":  cal_image,
        "dark_image": dark_image,
        "cal_path":   cal_path,
        "dark_path":  dark_path,
        "cal_type":   cal_type,
        "dark_type":  dark_type,
        "cal_raw":    cal_raw,
        "dark_raw":   dark_raw,
    }


# ---------------------------------------------------------------------------
# Stage B — extract_metadata  (Z01a variant)
# ---------------------------------------------------------------------------

def extract_metadata(load_result: dict) -> dict:
    """
    Z01a: no metadata header present in any input file.
    Always returns None for both cal and dark.
    """
    return {"cal_meta": None, "dark_meta": None}


# ---------------------------------------------------------------------------
# Stage C — figure_image_pair  (Z01a variant — no embedded metadata)
# ---------------------------------------------------------------------------

def figure_image_pair(
    load_result: dict,
    meta_result: dict,
) -> tuple:
    """
    Figure 1: side-by-side display of raw calibration and dark images.
    Both metadata panels show a 'no header' message.

    The user clicks on the CAL image to mark the initial fringe centre seed.
    Blocks until the figure is closed.

    Returns
    -------
    (cx_seed, cy_seed) : tuple[float, float]
        Pixel coordinates selected by the user, or image centre if no click.
    """
    cal_image  = load_result["cal_image"]
    dark_image = load_result["dark_image"]
    cal_path   = load_result["cal_path"]
    dark_path  = load_result["dark_path"]

    ADU_MAX = 2 ** 14 - 1   # 16 383

    # ── Figure layout ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        2, 2, figsize=(16, 12),
        gridspec_kw={"height_ratios": [3, 2]},
    )
    ax_cal_img,  ax_dark_img  = axes[0, 0], axes[0, 1]
    ax_meta_cal, ax_meta_dark = axes[1, 0], axes[1, 1]

    # ── Image display — grayscale, full 14-bit scale ─────────────────────────
    im_cal = ax_cal_img.imshow(cal_image, cmap="gray", vmin=0, vmax=ADU_MAX, origin="upper")
    plt.colorbar(im_cal, ax=ax_cal_img, label="ADU")
    ax_cal_img.set_title(
        "Calibration frame — click to set fringe seed",
        fontsize=9,
    )

    im_dark = ax_dark_img.imshow(dark_image, cmap="gray", vmin=0, vmax=ADU_MAX, origin="upper")
    plt.colorbar(im_dark, ax=ax_dark_img, label="ADU")
    ax_dark_img.set_title("Master dark", fontsize=9)

    # ── Metadata table panels — no header present ────────────────────────────
    for ax_table in [ax_meta_cal, ax_meta_dark]:
        ax_table.axis("off")
        ax_table.table(
            cellText=[["No embedded metadata — header row absent"]],
            colLabels=["Status"],
            loc="center",
            cellLoc="center",
        )

    # ── Interactive click on CAL image to collect cx/cy seed ─────────────────
    clicks = []

    def on_click(event):
        if event.inaxes is not ax_cal_img:
            return
        if event.xdata is None or event.ydata is None:
            return
        clicks.clear()
        clicks.append((event.xdata, event.ydata))
        cx, cy = event.xdata, event.ydata
        # Remove any previous marker
        for _ln in ax_cal_img.lines[:]:
            _ln.remove()
        ax_cal_img.plot(cx, cy, "r+", markersize=14, markeredgewidth=2)
        ax_cal_img.set_title(
            f"Calibration frame — seed: ({cx:.1f}, {cy:.1f})  [close to continue]",
            fontsize=9,
        )
        fig.canvas.draw()

    fig.canvas.mpl_connect("button_press_event", on_click)

    # ── Figure title — include image directory path(s) ───────────────────────
    cal_dir  = cal_path.parent
    dark_dir = dark_path.parent
    if cal_dir == dark_dir:
        directory_str = str(cal_dir)
    else:
        directory_str = f"CAL: {cal_dir}  |  DARK: {dark_dir}"

    fig.suptitle(
        f"Figure 1 — OI 630 nm Filtered Neon Lamp Calibration  |  {directory_str}",
        fontsize=11,
    )
    plt.tight_layout()
    plt.show()

    # ── Return seed (default to image centre if no click recorded) ───────────
    H, W = cal_image.shape
    if clicks:
        cx_seed, cy_seed = clicks[0]
        print(f"  Fringe centre seed from Figure 1 click: "
              f"(cx={cx_seed:.1f}, cy={cy_seed:.1f}) px")
    else:
        cx_seed, cy_seed = W / 2.0, H / 2.0
        print(f"  No click recorded — defaulting to image centre "
              f"(cx={cx_seed:.1f}, cy={cy_seed:.1f}) px")

    return cx_seed, cy_seed


# ---------------------------------------------------------------------------
# Stage D — figure_roi_inspection  (unchanged from Z01)
# ---------------------------------------------------------------------------

def figure_roi_inspection(
    cal_image:  np.ndarray,
    dark_image: np.ndarray,
    cx:         float,
    cy:         float,
    roi_half:   int = 108,
) -> None:
    """
    Figure 2: Cal ROI and dark ROI side by side with histograms below each.

    Both images use grayscale and the full 14-bit ADU scale (0–16 383).
    """
    # unchanged from Z01
    ADU_MAX = 2 ** 14 - 1
    H, W = cal_image.shape
    x0 = max(0, int(round(cx)) - roi_half)
    y0 = max(0, int(round(cy)) - roi_half)
    x1 = min(W, x0 + 2 * roi_half)
    y1 = min(H, y0 + 2 * roi_half)

    cal_roi  = cal_image[y0:y1, x0:x1]
    dark_roi = dark_image[y0:y1, x0:x1]

    fig, axes = plt.subplots(
        2, 2, figsize=(14, 10),
        gridspec_kw={"height_ratios": [3, 1.5]},
    )

    pairs = [
        (axes[0, 0], axes[1, 0], cal_roi,
         f"CAL ROI  |  centre=({cx:.1f}, {cy:.1f})"),
        (axes[0, 1], axes[1, 1], dark_roi,
         "DARK ROI"),
    ]

    for ax_img, ax_hist, img, title in pairs:
        im = ax_img.imshow(img, cmap="gray", vmin=0, vmax=ADU_MAX, origin="upper")
        plt.colorbar(im, ax=ax_img, label="ADU  (0 – 16 383)")
        ax_img.set_title(title, fontsize=9)

        hist_max = float(np.percentile(img, 99.9))
        hist_min = float(img.min())
        ax_hist.hist(
            img.ravel(), bins=128,
            range=(hist_min, hist_max) if hist_max > hist_min else (hist_min, hist_min + 1),
            color="steelblue", alpha=0.7,
        )
        med = float(np.median(img))
        ax_hist.axvline(med, color="red", linestyle="--", label=f"median={med:.0f}")
        ax_hist.set_xlabel("ADU")
        ax_hist.set_ylabel("Count")
        ax_hist.legend(fontsize=8)

    fig.suptitle("Figure 2 \u2014 ROI Inspection  [close to continue]", fontsize=13)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Stage F-1 — run_s12_reduction  (unchanged from Z01)
# ---------------------------------------------------------------------------

def run_s12_reduction(
    cal_image:  np.ndarray,
    dark_image: np.ndarray,
    cx_seed:    float,
    cy_seed:    float,
) -> tuple:
    """
    Run S12's reduce_calibration_frame() on the full cal image.

    Returns
    -------
    (FringeProfile, s12_dark_sub_image)
    """
    # unchanged from Z01
    master_dark = make_master_dark([dark_image])
    H, W = cal_image.shape
    r_max = min(H, W) / 2.0

    fp = reduce_calibration_frame(
        image       = cal_image,
        master_dark = master_dark,
        cx_human    = cx_seed,
        cy_human    = cy_seed,
        r_max_px    = r_max,
        n_bins      = 150,
    )

    s12_dark_sub = subtract_dark(cal_image, master_dark, clip_negative=True)

    n_ok = sum(1 for p in fp.peak_fits if p.fit_ok)
    print(f"  S12 annular reduction complete:")
    print(f"    Centre: ({fp.cx:.3f}, {fp.cy:.3f}) px  "
          f"[\u03c3=({fp.sigma_cx:.3f}, {fp.sigma_cy:.3f}) px]")
    print(f"    r_max: {fp.r_max_px:.1f} px")
    print(f"    Bins used: {fp.n_bins}")
    print(f"    Dark subtracted: {fp.dark_subtracted}")
    print(f"    Peaks found: {n_ok} / {len(fp.peak_fits)}")

    return fp, s12_dark_sub


# ---------------------------------------------------------------------------
# Stage F-2 — figure_dark_comparison  (Figure 3, unchanged from Z01)
# ---------------------------------------------------------------------------

def figure_dark_comparison(
    cal_image:    np.ndarray,
    dark_image:   np.ndarray,
    cx_seed:      float,
    cy_seed:      float,
    s12_dark_sub: np.ndarray,
    roi_half:     int = 108,
) -> None:
    """
    Figure 3: side-by-side comparison of visual vs S12 dark-subtracted ROI.
    """
    # unchanged from Z01
    H, W = cal_image.shape
    x0 = max(0, int(round(cx_seed)) - roi_half)
    y0 = max(0, int(round(cy_seed)) - roi_half)
    x1 = min(W, x0 + 2 * roi_half)
    y1 = min(H, y0 + 2 * roi_half)

    visual_diff = np.clip(
        cal_image[y0:y1, x0:x1].astype(np.float64)
        - dark_image[y0:y1, x0:x1].astype(np.float64),
        0.0, None,
    )
    s12_diff = s12_dark_sub[y0:y1, x0:x1]

    max_abs_diff = float(np.max(np.abs(visual_diff - s12_diff)))
    are_equal    = np.allclose(visual_diff, s12_diff, atol=0.0, rtol=0.0)
    equality_str = (
        "IDENTICAL (max |diff| = 0)"
        if are_equal
        else f"DIFFER  max |diff| = {max_abs_diff:.4g} ADU"
    )
    print(f"  Dark comparison: visual diff vs S12 diff — {equality_str}")

    combined = np.concatenate([visual_diff.ravel(), s12_diff.ravel()])
    vmax = float(np.percentile(combined, 99.5))

    fig, axes = plt.subplots(
        2, 2, figsize=(14, 10),
        gridspec_kw={"height_ratios": [3, 1.5]},
    )

    panels = [
        (axes[0, 0], axes[1, 0], visual_diff,
         "CAL \u2212 DARK  (visual: simple numpy clip \u2265 0)"),
        (axes[0, 1], axes[1, 1], s12_diff,
         "CAL \u2212 DARK  (S12: subtract_dark, same master_dark)"),
    ]

    for ax_img, ax_hist, img, title in panels:
        im = ax_img.imshow(img, cmap="gray", vmin=0, vmax=vmax, origin="upper")
        plt.colorbar(im, ax=ax_img, label="ADU")
        ax_img.set_title(title, fontsize=9)

        hist_max = float(np.percentile(img, 99.9))
        hist_min = float(img.min())
        ax_hist.hist(
            img.ravel(), bins=128,
            range=(hist_min, hist_max) if hist_max > hist_min else (hist_min, hist_min + 1),
            color="steelblue", alpha=0.7,
        )
        med = float(np.median(img))
        ax_hist.axvline(med, color="red", linestyle="--", label=f"median={med:.0f}")
        ax_hist.set_xlabel("ADU")
        ax_hist.set_ylabel("Count")
        ax_hist.legend(fontsize=8)

    fig.suptitle(
        f"Figure 3 \u2014 Dark Subtraction Comparison: Visual vs S12\n"
        f"ROI [{x0}:{x1}, {y0}:{y1}]  |  {equality_str}  [close to continue]",
        fontsize=11,
    )
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Stage F-3 — figure_reduction_peaks  (Figure 4 — Z01a variant)
# ---------------------------------------------------------------------------

def figure_reduction_peaks(
    fp:       FringeProfile,
    roi:      dict,
    cal_path: pathlib.Path,
) -> None:
    """
    Figure 4: r²-binned radial profile with single-family peak overlay and
    peak table.  All peaks drawn in steelblue — no family classification.

    Parameters
    ----------
    fp       : FringeProfile from run_s12_reduction()
    roi      : dict with 'roi_x0', 'roi_y0', 'roi_x1', 'roi_y1' (for context only)
    cal_path : Path to the calibration image (for title)
    """
    n_peaks_found = sum(1 for p in fp.peak_fits if p.fit_ok)

    if n_peaks_found < 4:
        print(f"  WARNING: only {n_peaks_found} good peaks found — expected 6–7.")

    fig, (ax_profile, ax_table) = plt.subplots(
        2, 1, figsize=(18, 12),
        gridspec_kw={"height_ratios": [3, 2]},
    )

    # ── Radial profile plot ──────────────────────────────────────────────────
    good_mask = ~fp.masked & np.isfinite(fp.sigma_profile)
    r2   = fp.r2_grid
    prof = fp.profile
    sem  = fp.sigma_profile

    ax_profile.plot(r2[good_mask], prof[good_mask], color="steelblue", lw=1.5,
                    label="Mean intensity")
    ax_profile.fill_between(
        r2[good_mask],
        prof[good_mask] - sem[good_mask],
        prof[good_mask] + sem[good_mask],
        alpha=0.3, color="steelblue", label="\u00b11 SEM",
    )
    ax_profile.set_xlabel("r\u00b2 (px\u00b2)", fontsize=11)
    ax_profile.set_ylabel("Mean intensity (ADU)", fontsize=11)
    ax_profile.legend(fontsize=9)

    # ── Peak overlay — all in steelblue, no family classification ────────────
    if fp.peak_fits:
        prof_range = prof[good_mask].max() - prof[good_mask].min()
        for pf in fp.peak_fits:
            r2_centre = pf.r_fit_px ** 2
            y_peak    = float(np.interp(r2_centre, r2[good_mask], prof[good_mask]))
            y_arrow   = y_peak + 0.12 * prof_range

            if pf.fit_ok:
                sigma_r2 = (
                    2.0 * pf.r_fit_px * pf.sigma_r_fit_px
                    if not math.isnan(pf.sigma_r_fit_px)
                    else float("nan")
                )
                ax_profile.annotate(
                    "",
                    xy=(r2_centre, y_peak),
                    xytext=(r2_centre, y_arrow),
                    arrowprops=dict(arrowstyle="->", color="steelblue", lw=1.2),
                )
                lbl = (
                    f"r\u00b2={r2_centre:.0f}\n\u00b1{sigma_r2:.1f}"
                    if not math.isnan(sigma_r2)
                    else f"r\u00b2={r2_centre:.0f}"
                )
                ax_profile.text(
                    r2_centre,
                    y_arrow + 0.01 * prof_range,
                    lbl, ha="center", va="bottom", fontsize=6, color="steelblue",
                )
            else:
                ax_profile.plot(r2_centre, y_peak, "x", color="grey",
                                markersize=8, markeredgewidth=1.5)
                ax_profile.text(r2_centre, y_peak, "failed", ha="center",
                                va="bottom", fontsize=6, color="grey")

    suptitle = (
        f"Figure 4 \u2014 Annular Reduction and Peak Identification\n"
        f"OI 630 nm filtered neon lamp  |  "
        f"File: {cal_path.name}  |  "
        f"Centre: ({fp.cx:.3f}, {fp.cy:.3f}) px  "
        f"[\u03c3=({fp.sigma_cx:.3f}, {fp.sigma_cy:.3f}) px]  |  "
        f"r_max={fp.r_max_px:.1f} px  |  "
        f"Peaks found: {n_peaks_found} (expect 6\u20137)  [close to continue]"
    )
    fig.suptitle(suptitle, fontsize=10)

    # ── Peak table — no Family column ────────────────────────────────────────
    ax_table.axis("off")
    if fp.peak_fits:
        headers = [
            "#", "r\u00b2 centre (px\u00b2)", "\u03c3(r\u00b2) (px\u00b2)",
            "2\u03c3(r\u00b2) (px\u00b2)", "Amplitude (ADU)",
            "\u03c3(amp) (ADU)", "Fit OK",
        ]
        rows = []
        for i, pf in enumerate(fp.peak_fits, start=1):
            r2_c = pf.r_fit_px ** 2
            if not math.isnan(pf.sigma_r_fit_px):
                s_r2    = 2.0 * pf.r_fit_px * pf.sigma_r_fit_px
                two_s_r2 = 2.0 * s_r2
            else:
                s_r2    = float("nan")
                two_s_r2 = float("nan")
            # sigma_amplitude_adu: use attribute if present, else dash
            try:
                s_amp = pf.sigma_amplitude_adu
                s_amp_str = f"{s_amp:.1f}" if not math.isnan(s_amp) else "\u2014"
            except AttributeError:
                s_amp_str = "\u2014"
            rows.append([
                str(i),
                f"{r2_c:.1f}",
                f"{s_r2:.2f}"    if not math.isnan(s_r2)    else "\u2014",
                f"{two_s_r2:.2f}" if not math.isnan(two_s_r2) else "\u2014",
                f"{pf.amplitude_adu:.1f}",
                s_amp_str,
                "\u2713" if pf.fit_ok else "\u2717",
            ])
        tbl = ax_table.table(
            cellText=rows,
            colLabels=headers,
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
    else:
        ax_table.text(0.5, 0.5, "No peaks found.", ha="center", va="center",
                      transform=ax_table.transAxes, fontsize=12)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Stage G — figure_tolansky_1line  (Figure 5 — Z01a specific)
# ---------------------------------------------------------------------------

def figure_tolansky_1line(
    fp:       FringeProfile,
    cal_path: pathlib.Path,
) -> SingleLineResult:
    """
    Run S13's SingleLineTolansky on the FringeProfile and produce Figure 5:
    the single-line Tolansky r² characterisation with ε₀ and v_ref.

    Returns
    -------
    SingleLineResult
    """
    # ── S13 call (spec §9.3) ─────────────────────────────────────────────────
    analyser = SingleLineTolansky(
        fringe_profile = fp,
        lam_rest_nm    = OI_WAVELENGTH_NM,
        d_prior_m      = TOLANSKY_D_MM * 1e-3,
        f_prior_m      = TOLANSKY_F_MM * 1e-3,
        pixel_pitch_m  = CCD_PIXEL_PITCH_M,
        d_icos_m       = ICOS_GAP_MM * 1e-3,
    )
    result = analyser.run()

    # ── Zero-velocity check (spec §9.4) ──────────────────────────────────────
    v_ref_check_pass = abs(result.v_rel_ms) < 3 * result.sigma_v_ms
    print(f"  Zero-velocity check: v_rel = {result.v_rel_ms:.1f} \u00b1 {result.sigma_v_ms:.1f} m/s  "
          f"({'PASS' if v_ref_check_pass else 'WARN \u2014 non-zero velocity detected'})")
    print(f"  \u03b5\u2080 (rest-frame fractional order) = {result.epsilon:.6f} \u00b1 {result.sigma_eps:.6f}")

    # ── Build figure ─────────────────────────────────────────────────────────
    fig, (ax_scatter, ax_table) = plt.subplots(
        2, 1, figsize=(14, 10),
        gridspec_kw={"height_ratios": [3, 2]},
    )

    # Gather good peaks sorted by radius (innermost first)
    good_peaks = sorted(
        [pf for pf in fp.peak_fits if pf.fit_ok],
        key=lambda pf: pf.r_fit_px,
    )
    p_vals   = np.arange(len(good_peaks), dtype=float)
    r2_obs   = np.array([pf.r_fit_px ** 2 for pf in good_peaks])
    sr2_obs  = np.array([
        2.0 * pf.r_fit_px * pf.sigma_r_fit_px
        if not math.isnan(pf.sigma_r_fit_px) else 1.0
        for pf in good_peaks
    ])
    r2_fit   = result.S * (p_vals + result.epsilon)

    # ── r² scatter plot (spec §9.5) ──────────────────────────────────────────
    ax_scatter.errorbar(
        p_vals, r2_obs, yerr=sr2_obs,
        fmt="o", markersize=6, mfc="white", mec="steelblue",
        ecolor="steelblue", capsize=4,
    )
    p_line = np.linspace(p_vals[0] - 0.2, p_vals[-1] + 0.2, 200)
    ax_scatter.plot(p_line, result.S * (p_line + result.epsilon),
                    "-", color="steelblue", lw=1.5)
    ax_scatter.set_xlabel("Fringe index p", fontsize=11)
    ax_scatter.set_ylabel("r\u00b2 (px\u00b2)", fontsize=11)

    # Residuals to compute R²
    weights = 1.0 / (sr2_obs ** 2)
    ss_res  = float(np.sum(weights * (r2_obs - r2_fit) ** 2))
    ss_tot  = float(np.sum(weights * (r2_obs - np.average(r2_obs, weights=weights)) ** 2))
    r2_gof  = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    ax_scatter.text(
        0.03, 0.97,
        f"S = {result.S:.2f} px\u00b2/fringe  [fixed]\n"
        f"\u03b5\u2080 = {result.epsilon:.6f} \u00b1 {result.sigma_eps:.6f}\n"
        f"\u03bb_c = {result.lam_c_nm:.5f} nm\n"
        f"v_ref = {result.v_rel_ms:.1f} \u00b1 {result.sigma_v_ms:.1f} m/s\n"
        f"R\u00b2 = {r2_gof:.6f}",
        transform=ax_scatter.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round", fc="lightyellow"),
    )

    # ── Residuals inset (spec §9.6) ──────────────────────────────────────────
    ax_inset = ax_scatter.inset_axes([0.55, 0.05, 0.42, 0.3])
    norm_resid = (r2_obs - r2_fit) / sr2_obs
    ax_inset.stem(p_vals, norm_resid, linefmt="steelblue", markerfmt="o",
                  basefmt="k--")
    ax_inset.axhline(0, color="k", lw=0.8, ls="--")
    ax_inset.set_ylabel("Residual (\u03c3)", fontsize=7)
    ax_inset.tick_params(labelsize=7)

    # ── Results table (spec §9.7) ─────────────────────────────────────────────
    ax_table.axis("off")
    tab_headers = ["Parameter", "Symbol", "Value", "1\u03c3 uncertainty",
                   "2\u03c3 uncertainty", "Units", "Notes"]
    tab_rows = [
        ["Frac. order",         "\u03b5\u2080",
         f"{result.epsilon:.6f}",
         f"{result.sigma_eps:.6f}",
         f"{result.two_sigma_eps:.6f}",
         "\u2014", "rest-frame OI 630 nm; store as cal. constant"],
        ["Tolansky slope",      "S",
         f"{result.S:.3f}",
         "\u2014  (fixed)",
         "\u2014  (fixed)",
         "px\u00b2/fringe", f"d={TOLANSKY_D_MM:.3f} mm, f={TOLANSKY_F_MM:.2f} mm"],
        ["Line-centre \u03bb",  "\u03bb_c",
         f"{result.lam_c_nm:.5f}",
         f"{result.sigma_lam_c_nm:.5f}",
         f"{result.two_sigma_lam_c_nm:.5f}",
         "nm", f"\u03bb_rest = {OI_WAVELENGTH_NM:.1f} nm"],
        ["LOS velocity",        "v_ref",
         f"{result.v_rel_ms:.1f}",
         f"{result.sigma_v_ms:.1f}",
         f"{result.two_sigma_v_ms:.1f}",
         "m/s", "expect \u2248 0 (static Ne lamp)"],
        ["Integer order",       "N_int",
         str(result.N_int),
         "\u2014", "\u2014",
         "\u2014", "resolved from ICOS prior"],
        ["d prior",             "d",
         f"{result.d_prior_mm:.3f}",
         "\u2014", "\u2014",
         "mm", "fixed — from Z01 Tolansky"],
        ["f prior",             "f",
         f"{result.f_prior_mm:.2f}",
         "\u2014", "\u2014",
         "mm", "fixed — from Z01 Tolansky"],
        ["Reduced \u03c7\u00b2", "\u03c7\u00b2/\u03bd",
         f"{result.chi2_dof:.3f}",
         "\u2014", "\u2014",
         "\u2014", "acceptable: 0.5\u20133.0"],
    ]
    tbl = ax_table.table(
        cellText=tab_rows,
        colLabels=tab_headers,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)

    # ── Title bar (spec §9.8) ─────────────────────────────────────────────────
    v_check = "PASS" if abs(result.v_rel_ms) < 3 * result.sigma_v_ms else "WARN"
    suptitle = (
        f"Figure 5 \u2014 Single-Line Tolansky  |  OI 630 nm zero-velocity reference\n"
        f"File: {cal_path.name}  |  "
        f"\u03b5\u2080 = {result.epsilon:.6f} \u00b1 {result.sigma_eps:.6f}  |  "
        f"\u03bb_c = {result.lam_c_nm:.5f} nm  |  "
        f"v_ref = {result.v_rel_ms:.1f} \u00b1 {result.sigma_v_ms:.1f} m/s  "
        f"[{v_check}]  |  [close to exit]"
    )
    fig.suptitle(suptitle, fontsize=9)
    plt.tight_layout()
    plt.show()

    return result


# ---------------------------------------------------------------------------
# Stage D helper — ROI from cx/cy  (used internally; no user prompt)
# ---------------------------------------------------------------------------

def _make_roi(cal_image: np.ndarray, cx: float, cy: float, roi_half: int = 108) -> dict:
    """Build a simple ROI dict centred on (cx, cy)."""
    H, W = cal_image.shape
    x0 = max(0, int(round(cx)) - roi_half)
    y0 = max(0, int(round(cy)) - roi_half)
    x1 = min(W, x0 + 2 * roi_half)
    y1 = min(H, y0 + 2 * roi_half)
    return {
        "cx_seed":  cx,
        "cy_seed":  cy,
        "roi_cols": x1 - x0,
        "roi_rows": y1 - y0,
        "roi_x0":   x0,
        "roi_y0":   y0,
        "roi_x1":   x1,
        "roi_y1":   y1,
    }


# ---------------------------------------------------------------------------
# Main  (spec §10)
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Sequential execution of all stages.
    Each figure blocks until the user closes it.
    """
    print("=" * 70)
    print("  WindCube FPI \u2014 Z01a OI 630 nm Calibration Validation Script")
    print("=" * 70)

    # Stage A
    print("\nStage A: Loading images...")
    load = load_images()
    print(f"  CAL:  {load['cal_path'].name}  ({load['cal_type']})")
    print(f"  DARK: {load['dark_path'].name}  ({load['dark_type']})")

    # Stage B
    print("\nStage B: Extracting metadata...")
    meta = extract_metadata(load)

    # Stage C
    print("\nStage C: Displaying image pair (Figure 1)...")
    cx_seed, cy_seed = figure_image_pair(load, meta)

    # Build ROI dict for downstream stages
    roi = _make_roi(load["cal_image"], cx_seed, cy_seed)
    print(f"  ROI:  {roi['roi_cols']}×{roi['roi_rows']} px  "
          f"[{roi['roi_x0']}:{roi['roi_x1']}, {roi['roi_y0']}:{roi['roi_y1']}]")

    # Stage D — ROI inspection (Figure 2)
    print("\nStage D: Displaying ROI inspection (Figure 2)...")
    figure_roi_inspection(
        load["cal_image"], load["dark_image"], cx_seed, cy_seed,
    )

    # Stage F-1: S12 annular reduction
    print("\nStage F: Running S12 annular reduction + peak finding...")
    fp, s12_dark_sub = run_s12_reduction(
        cal_image  = load["cal_image"],
        dark_image = load["dark_image"],
        cx_seed    = cx_seed,
        cy_seed    = cy_seed,
    )

    # Stage F-2: dark subtraction comparison (Figure 3)
    print("\nStage F: Displaying dark subtraction comparison (Figure 3)...")
    figure_dark_comparison(
        cal_image    = load["cal_image"],
        dark_image   = load["dark_image"],
        cx_seed      = cx_seed,
        cy_seed      = cy_seed,
        s12_dark_sub = s12_dark_sub,
    )

    # Stage F-3: radial profile + peak table (Figure 4)
    print("\nStage F: Displaying radial profile and peaks (Figure 4)...")
    figure_reduction_peaks(fp, roi, load["cal_path"])

    # Stage G: Single-line Tolansky (Figure 5)
    print("\nStage G: Running S13 single-line Tolansky analysis (Figure 5)...")
    result = figure_tolansky_1line(fp, load["cal_path"])

    # Final console summary
    print("\n" + "=" * 70)
    print("  Z01a complete. All figures closed.")
    print("  OI 630 nm zero-velocity calibration result:")
    print(f"    \u03b5\u2080    = {result.epsilon:.6f} \u00b1 {result.two_sigma_eps:.6f}  (2\u03c3)  \u2190 store as calibration constant")
    print(f"    \u03bb_c   = {result.lam_c_nm:.5f} \u00b1 {result.two_sigma_lam_c_nm:.5f} nm (2\u03c3)")
    print(f"    v_ref = {result.v_rel_ms:.1f} \u00b1 {result.two_sigma_v_ms:.1f} m/s (2\u03c3)  \u2190 expect \u2248 0")
    v_check = abs(result.v_rel_ms) < 3 * result.sigma_v_ms
    print(f"    Zero-velocity check: {'PASS' if v_check else 'WARN \u2014 investigate'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
