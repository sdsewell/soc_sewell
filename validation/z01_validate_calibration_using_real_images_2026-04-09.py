"""
Z01 — Validate Calibration Using Real (or Synthetic) Images
WindCube FPI Pipeline — NCAR / High Altitude Observatory (HAO)
Spec: docs/specs/Z01_validate_calibration_using_real_images_2026-04-09.md
Tool: Claude Code
Last updated: 2026-04-09
"""

import math
import pathlib
import sys
import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# Internal imports — paths relative to soc_sewell/ (project root must be on sys.path)
from src.fpi.m03_annular_reduction_2026_04_06 import (
    FringeProfile,
    make_master_dark,
    reduce_calibration_frame,
)
from src.fpi.tolansky_2026_04_05 import TolanskyPipeline, TwoLineResult
from src.metadata.p01_image_metadata_2026_04_06 import ingest_real_image, ImageMetadata

# Physical constants (S03 values used directly to avoid import path ambiguity)
NE_WAVELENGTH_1_NM  = 640.2248    # nm — primary Ne line
NE_WAVELENGTH_2_NM  = 638.2991    # nm — secondary Ne line
CCD_PIXEL_PITCH_M   = 32e-6       # m  — 2×2 binned pixel pitch
ICOS_GAP_MM         = 20.008      # mm — ICOS build report spacer measurement
ICOS_GAP_M          = ICOS_GAP_MM * 1e-3


# ---------------------------------------------------------------------------
# Stage A — load_images
# ---------------------------------------------------------------------------

def load_images() -> dict:
    """
    Open two Windows file-picker dialogs (tkinter) and load the selected
    calibration image and dark image.

    Supports:
        *.bin  — real WindCube binary (260×276 uint16 big-endian)
        *.npy  — NumPy array produced by M02 / synthetic pipeline

    Returns
    -------
    dict with keys:
        'cal_image'     : np.ndarray float64, pixel data only
        'dark_image'    : np.ndarray float64, pixel data only
        'cal_path'      : pathlib.Path
        'dark_path'     : pathlib.Path
        'cal_type'      : 'real' | 'synthetic'
        'dark_type'     : 'real' | 'synthetic'
        'cal_raw'       : np.ndarray uint16 (260×276) or None
        'dark_raw'      : np.ndarray uint16 or None
        'cal_meta_raw'  : ImageMetadata or None
        'dark_meta_raw' : ImageMetadata or None
    """
    root = tk.Tk()
    root.withdraw()

    file_types = [("WindCube images", "*.bin *.npy"), ("All files", "*.*")]

    cal_path_str = filedialog.askopenfilename(
        title="Select WindCube Calibration Image",
        filetypes=file_types,
    )
    if not cal_path_str:
        print("No calibration image selected. Exiting.")
        sys.exit(0)

    dark_path_str = filedialog.askopenfilename(
        title="Select WindCube Dark Image",
        filetypes=file_types,
    )
    if not dark_path_str:
        print("No dark image selected. Exiting.")
        sys.exit(0)

    root.destroy()

    cal_path  = pathlib.Path(cal_path_str)
    dark_path = pathlib.Path(dark_path_str)

    def _load_file(path):
        """Load a .bin or .npy file. Return (pixel_array, raw_array, meta, type_str)."""
        suffix = path.suffix.lower()
        if suffix == ".bin":
            meta, pixels = ingest_real_image(path)
            raw = np.frombuffer(path.read_bytes(), dtype=">u2").reshape(260, 276)
            return pixels.astype(np.float64), raw, meta, "real"
        elif suffix == ".npy":
            arr = np.load(path)
            if arr.ndim != 2:
                raise ValueError(
                    f"NumPy file {path.name} has shape {arr.shape}; expected 2D array."
                )
            return arr.astype(np.float64), None, None, "synthetic"
        else:
            raise ValueError(f"Unsupported file type '{suffix}' for {path.name}")

    cal_image, cal_raw, cal_meta, cal_type   = _load_file(cal_path)
    dark_image, dark_raw, dark_meta, dark_type = _load_file(dark_path)

    if cal_image.shape != dark_image.shape:
        raise ValueError(
            f"Calibration image shape {cal_image.shape} does not match "
            f"dark image shape {dark_image.shape}."
        )

    return {
        "cal_image":     cal_image,
        "dark_image":    dark_image,
        "cal_path":      cal_path,
        "dark_path":     dark_path,
        "cal_type":      cal_type,
        "dark_type":     dark_type,
        "cal_raw":       cal_raw,
        "dark_raw":      dark_raw,
        "cal_meta_raw":  cal_meta,
        "dark_meta_raw": dark_meta,
    }


# ---------------------------------------------------------------------------
# Stage B — extract_metadata
# ---------------------------------------------------------------------------

def extract_metadata(load_result: dict) -> dict:
    """
    Extract ImageMetadata from the loaded images.

    For .bin files: re-uses the ImageMetadata already parsed in load_images()
    and stored in 'cal_meta_raw' / 'dark_meta_raw'.
    For .npy synthetic files: metadata is None.

    Returns
    -------
    dict with keys:
        'cal_meta'  : ImageMetadata | None
        'dark_meta' : ImageMetadata | None
    """
    return {
        "cal_meta":  load_result.get("cal_meta_raw"),
        "dark_meta": load_result.get("dark_meta_raw"),
    }


# ---------------------------------------------------------------------------
# Stage C — figure_image_pair
# ---------------------------------------------------------------------------

def figure_image_pair(
    load_result: dict,
    meta_result: dict,
) -> None:
    """
    Figure 1: side-by-side display of raw calibration and dark images
    with metadata tables below each. Blocks until the user closes the figure.
    """
    cal_image  = load_result["cal_image"]
    dark_image = load_result["dark_image"]
    cal_path   = load_result["cal_path"]
    dark_path  = load_result["dark_path"]
    cal_meta   = meta_result["cal_meta"]
    dark_meta  = meta_result["dark_meta"]

    fig, axes = plt.subplots(
        2, 2, figsize=(16, 12),
        gridspec_kw={"height_ratios": [3, 2]},
    )
    ax_cal_img, ax_dark_img = axes[0, 0], axes[0, 1]
    ax_cal_tab, ax_dark_tab = axes[1, 0], axes[1, 1]

    # Image display
    for ax, img, title in [
        (ax_cal_img, cal_image,
         f"CAL: {cal_path.name}  |  {cal_image.shape[1]}×{cal_image.shape[0]} px"),
        (ax_dark_img, dark_image,
         f"DARK: {dark_path.name}  |  {dark_image.shape[1]}×{dark_image.shape[0]} px"),
    ]:
        vmin = float(np.percentile(img, 1))
        vmax = float(np.percentile(img, 99))
        im = ax.imshow(img, cmap="viridis", vmin=vmin, vmax=vmax, origin="upper")
        plt.colorbar(im, ax=ax, label="ADU")
        ax.set_title(title, fontsize=10)

    # Metadata tables
    for ax, meta in [(ax_cal_tab, cal_meta), (ax_dark_tab, dark_meta)]:
        ax.axis("off")
        if meta is None:
            table_data = [["Synthetic image — no embedded metadata", ""]]
            col_labels = ["Field", "Value"]
        else:
            lamp_on = [i for i, v in enumerate(meta.lamp_ch_array) if v]
            table_data = [
                ["Timestamp (UTC)",       str(meta.utc_timestamp)],
                ["Exposure time (cs)",    str(meta.exp_time)],
                ["CCD temp (°C)",         f"{meta.ccd_temp1:.1f}"],
                ["Etalon temps (°C)",     ", ".join(f"{t:.1f}" for t in meta.etalon_temps)],
                ["Lamp channels on",      str(lamp_on) if lamp_on else "none"],
                ["S/C latitude (°)",      f"{math.degrees(meta.spacecraft_latitude):.2f}"],
                ["S/C longitude (°)",     f"{math.degrees(meta.spacecraft_longitude):.2f}"],
                ["S/C altitude (km)",     f"{meta.spacecraft_altitude / 1000:.1f}"],
                ["Obs mode",              str(meta.obs_mode)],
                ["Is synthetic",          str(meta.is_synthetic)],
            ]
            col_labels = ["Field", "Value"]
        tbl = ax.table(
            cellText=table_data,
            colLabels=col_labels,
            loc="center",
            cellLoc="left",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)

    fig.suptitle("Figure 1 — Image Pair Inspection  [close to continue]", fontsize=13)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Stage D — get_roi_from_user
# ---------------------------------------------------------------------------

def get_roi_from_user(
    cal_image: np.ndarray,
    default_roi: tuple = (216, 216),
) -> dict:
    """
    Prompt the user for a fringe centre seed via an interactive matplotlib
    click and a console ROI size prompt.

    Returns
    -------
    dict with keys:
        'cx_seed', 'cy_seed', 'roi_cols', 'roi_rows',
        'roi_x0', 'roi_y0', 'roi_x1', 'roi_y1'
    """
    H, W = cal_image.shape
    clicks = []

    fig, ax = plt.subplots(figsize=(8, 8))
    vmin = float(np.percentile(cal_image, 1))
    vmax = float(np.percentile(cal_image, 99))
    ax.imshow(cal_image, cmap="viridis", vmin=vmin, vmax=vmax, origin="upper")
    ax.set_title("Figure 1b — Click to mark fringe centre seed  [close after clicking]")

    def on_click(event):
        if event.inaxes is not ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        clicks.clear()
        clicks.append((event.xdata, event.ydata))
        ax.plot(event.xdata, event.ydata, "r+", markersize=14, markeredgewidth=2)
        ax.set_title(
            f"Seed set: ({event.xdata:.1f}, {event.ydata:.1f})  [close to continue]"
        )
        fig.canvas.draw()

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.tight_layout()
    plt.show()

    if clicks:
        cx_seed, cy_seed = clicks[0]
    else:
        cx_seed, cy_seed = W / 2.0, H / 2.0
        print(f"  No click registered — defaulting to image centre "
              f"({cx_seed:.1f}, {cy_seed:.1f})")

    print(f"  Fringe centre seed: (cx={cx_seed:.1f}, cy={cy_seed:.1f})")
    roi_input = input(
        f"  Enter ROI size [default {default_roi[0]} {default_roi[1]}] "
        "(rows cols, or press ENTER for default): "
    ).strip()

    if roi_input == "":
        roi_rows, roi_cols = default_roi
    else:
        parts = roi_input.split()
        if len(parts) == 2:
            try:
                roi_rows, roi_cols = int(parts[0]), int(parts[1])
            except ValueError:
                print("  Invalid input — using default ROI.")
                roi_rows, roi_cols = default_roi
        else:
            print("  Invalid input — using default ROI.")
            roi_rows, roi_cols = default_roi

    roi_x0 = max(0, int(round(cx_seed)) - roi_cols // 2)
    roi_y0 = max(0, int(round(cy_seed)) - roi_rows // 2)
    roi_x1 = min(W, roi_x0 + roi_cols)
    roi_y1 = min(H, roi_y0 + roi_rows)

    actual_cols = roi_x1 - roi_x0
    actual_rows = roi_y1 - roi_y0
    if actual_cols != roi_cols or actual_rows != roi_rows:
        print(
            f"  Warning: ROI clipped to image boundary "
            f"({actual_cols}×{actual_rows} instead of {roi_cols}×{roi_rows})"
        )
        roi_cols, roi_rows = actual_cols, actual_rows

    return {
        "cx_seed":  cx_seed,
        "cy_seed":  cy_seed,
        "roi_cols": roi_cols,
        "roi_rows": roi_rows,
        "roi_x0":   roi_x0,
        "roi_y0":   roi_y0,
        "roi_x1":   roi_x1,
        "roi_y1":   roi_y1,
    }


# ---------------------------------------------------------------------------
# Stage E — figure_roi_inspection
# ---------------------------------------------------------------------------

def figure_roi_inspection(
    cal_image:  np.ndarray,
    dark_image: np.ndarray,
    roi:        dict,
) -> np.ndarray:
    """
    Figure 2: Display cal ROI, dark ROI, and visual dark-subtracted cal ROI
    side by side with ADU histograms below each panel.

    Computes diff_roi using simple array arithmetic FOR VISUAL DISPLAY ONLY.
    Does NOT use S12's subtract_dark() and the returned array is NOT passed
    to reduce_calibration_frame() in Stage F.

    Returns
    -------
    np.ndarray — visual diff_roi (float64, clipped >= 0). For display only.
    """
    x0, y0, x1, y1 = roi["roi_x0"], roi["roi_y0"], roi["roi_x1"], roi["roi_y1"]
    cal_roi  = cal_image[y0:y1, x0:x1]
    dark_roi = dark_image[y0:y1, x0:x1]
    diff_roi = np.clip(
        cal_roi.astype(np.float64) - dark_roi.astype(np.float64), 0.0, None
    )

    fig, axes = plt.subplots(
        2, 3, figsize=(18, 10),
        gridspec_kw={"height_ratios": [3, 1.5]},
    )

    images   = [cal_roi, dark_roi, diff_roi]
    # Title strings include cal_path and dark_path info when available
    # (roi dict doesn't carry paths; use generic labels)
    titles = [
        f"CAL ROI  |  seed=({roi['cx_seed']:.1f}, {roi['cy_seed']:.1f})\n"
        f"{roi['roi_cols']}×{roi['roi_rows']} px",
        "DARK ROI",
        "CAL \u2212 DARK (visual only \u2014 not passed to S12)",
    ]

    for col, (img, title) in enumerate(zip(images, titles)):
        ax_img  = axes[0, col]
        ax_hist = axes[1, col]

        vmin = float(np.percentile(img, 1))
        vmax = float(np.percentile(img, 99))
        im = ax_img.imshow(img, cmap="viridis", vmin=vmin, vmax=vmax, origin="upper")
        plt.colorbar(im, ax=ax_img, label="ADU")
        ax_img.set_title(title, fontsize=9)

        # Histogram
        hist_max = float(np.percentile(img, 99.9))
        hist_min = float(img.min())
        ax_hist.hist(img.ravel(), bins=128,
                     range=(hist_min, hist_max) if hist_max > hist_min else (hist_min, hist_min + 1),
                     color="steelblue", alpha=0.7)
        med = float(np.median(img))
        ax_hist.axvline(med, color="red", linestyle="--",
                        label=f"median={med:.0f}")
        ax_hist.set_xlabel("ADU")
        ax_hist.set_ylabel("Count")
        ax_hist.legend(fontsize=8)

    fig.suptitle("Figure 2 \u2014 ROI Inspection  [close to continue]", fontsize=13)
    plt.tight_layout()
    plt.show()

    return diff_roi


# ---------------------------------------------------------------------------
# Stage F — figure_reduction_peaks
# ---------------------------------------------------------------------------

def figure_reduction_peaks(
    cal_image:  np.ndarray,
    dark_image: np.ndarray,
    roi:        dict,
    cal_path:   pathlib.Path,
) -> FringeProfile:
    """
    Run S12's reduce_calibration_frame() on the full cal image (not the ROI)
    with master_dark provided, then plot the r²-binned profile + 20-peak overlay.

    Stage F passes the raw cal_image and master_dark as SEPARATE arguments to
    reduce_calibration_frame() so that S12 performs the one and only dark
    subtraction internally. diff_roi from Stage E is NOT used here.

    Returns
    -------
    FringeProfile — full S12 output used by Stage G
    """
    # Build master dark from the single dark frame
    master_dark = make_master_dark([dark_image])

    r_max = min(roi["roi_rows"], roi["roi_cols"]) / 2.0

    fp = reduce_calibration_frame(
        image       = cal_image,
        master_dark = master_dark,
        cx_human    = roi["cx_seed"],
        cy_human    = roi["cy_seed"],
        r_max_px    = r_max,
        n_bins      = 150,
    )

    n_ok = sum(1 for p in fp.peak_fits if p.fit_ok)
    print(f"  S12 annular reduction complete:")
    print(f"    Centre: ({fp.cx:.3f}, {fp.cy:.3f}) px  "
          f"[\u03c3=({fp.sigma_cx:.3f}, {fp.sigma_cy:.3f}) px]")
    print(f"    r_max: {fp.r_max_px:.1f} px")
    print(f"    Bins used: {fp.n_bins}")
    print(f"    Dark subtracted: {fp.dark_subtracted}")
    print(f"    Peaks found: {n_ok} / {len(fp.peak_fits)}")

    # --- Build figure -------------------------------------------------------
    fig, (ax_profile, ax_table) = plt.subplots(
        2, 1, figsize=(18, 12),
        gridspec_kw={"height_ratios": [3, 2]},
    )

    # Profile plot
    good = ~fp.masked & np.isfinite(fp.sigma_profile)
    r2   = fp.r2_grid
    prof = fp.profile
    sem  = fp.sigma_profile

    ax_profile.plot(r2[good], prof[good], color="steelblue", lw=1.5,
                    label="Mean intensity")
    ax_profile.fill_between(
        r2[good],
        prof[good] - sem[good],
        prof[good] + sem[good],
        alpha=0.3, color="steelblue", label="\u00b11 SEM",
    )
    ax_profile.set_xlabel("r\u00b2 (px\u00b2)", fontsize=11)
    ax_profile.set_ylabel("Mean intensity (ADU)", fontsize=11)
    ax_profile.legend(fontsize=9)

    # Peak overlay
    if fp.peak_fits:
        max_amp = max(pf.amplitude_adu for pf in fp.peak_fits)
        for pf in fp.peak_fits:
            # r² of the fitted peak centre
            r2_centre = pf.r_fit_px ** 2
            y_peak    = float(np.interp(r2_centre, r2[good], prof[good]))
            y_arrow   = y_peak + 0.12 * (prof[good].max() - prof[good].min())

            is_high  = pf.amplitude_adu >= 0.5 * max_amp
            clr      = "red" if is_high else "darkorange"

            if pf.fit_ok:
                sigma_r2 = 2.0 * pf.r_fit_px * pf.sigma_r_fit_px if not math.isnan(pf.sigma_r_fit_px) else float("nan")
                ax_profile.annotate(
                    "",
                    xy=(r2_centre, y_peak),
                    xytext=(r2_centre, y_arrow),
                    arrowprops=dict(arrowstyle="->", color=clr, lw=1.2),
                )
                lbl = f"r\u00b2={r2_centre:.0f}\n\u00b1{sigma_r2:.1f}" if not math.isnan(sigma_r2) else f"r\u00b2={r2_centre:.0f}"
                ax_profile.text(
                    r2_centre, y_arrow + 0.01 * (prof[good].max() - prof[good].min()),
                    lbl, ha="center", va="bottom", fontsize=6, color=clr,
                )
            else:
                ax_profile.plot(r2_centre, y_peak, "x", color="grey",
                                markersize=8, markeredgewidth=1.5)
                ax_profile.text(r2_centre, y_peak, "failed", ha="center",
                                va="bottom", fontsize=6, color="grey")

    suptitle = (
        f"Figure 3 \u2014 Annular Reduction and Peak Identification\n"
        f"File: {cal_path.name}  |  "
        f"Centre: ({fp.cx:.3f}, {fp.cy:.3f}) px  "
        f"[\u03c3=({fp.sigma_cx:.3f}, {fp.sigma_cy:.3f}) px]  |  "
        f"r_max={fp.r_max_px:.1f} px  |  "
        f"Peaks: {n_ok}/20  [close to continue]"
    )
    fig.suptitle(suptitle, fontsize=10)

    # Peak table
    ax_table.axis("off")
    if fp.peak_fits:
        max_amp = max(pf.amplitude_adu for pf in fp.peak_fits)
        headers = ["#", "Family", "r\u00b2 centre (px\u00b2)", "\u03c3(r\u00b2) (px\u00b2)",
                   "2\u03c3(r\u00b2) (px\u00b2)", "Amplitude (ADU)", "Fit OK"]
        rows = []
        for i, pf in enumerate(fp.peak_fits, start=1):
            family = "Ne 640.2" if pf.amplitude_adu >= 0.5 * max_amp else "Ne 638.3"
            r2_c   = pf.r_fit_px ** 2
            if not math.isnan(pf.sigma_r_fit_px):
                s_r2 = 2.0 * pf.r_fit_px * pf.sigma_r_fit_px
            else:
                s_r2 = float("nan")
            rows.append([
                str(i),
                family,
                f"{r2_c:.1f}",
                f"{s_r2:.2f}" if not math.isnan(s_r2) else "\u2014",
                f"{2.0 * s_r2:.2f}" if not math.isnan(s_r2) else "\u2014",
                f"{pf.amplitude_adu:.1f}",
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

    return fp


# ---------------------------------------------------------------------------
# Stage G — figure_tolansky
# ---------------------------------------------------------------------------

def figure_tolansky(
    fp:       FringeProfile,
    cal_path: pathlib.Path,
) -> TwoLineResult:
    """
    Run S13's TolanskyPipeline on the FringeProfile from Stage F and
    produce Figure 4: the two-line Tolansky r² characterisation.

    Returns
    -------
    TwoLineResult — the joint two-line fit result
    """
    pipeline = TolanskyPipeline(
        profile       = fp,
        d_prior_m     = ICOS_GAP_M,
        pixel_pitch_m = CCD_PIXEL_PITCH_M,
        lam1_nm       = NE_WAVELENGTH_1_NM,
        lam2_nm       = NE_WAVELENGTH_2_NM,
    )
    result = pipeline.run()

    # Print summary (TolanskyPipeline delegates to TwoLineAnalyser internals)
    print("\n  Tolansky two-line analysis summary:")
    print(f"    S\u2081  = {result.S1:.4f} \u00b1 {result.sigma_S1:.4f} px\u00b2/fringe")
    print(f"    S\u2082/S\u2081 = {result.S2 / result.S1:.10f}  (\u03bb\u2082/\u03bb\u2081 = {result.lam_ratio:.10f})")
    print(f"    \u03b5\u2081  = {result.eps1:.8f} \u00b1 {result.sigma_eps1:.2e}")
    print(f"    \u03b5\u2082  = {result.eps2:.8f} \u00b1 {result.sigma_eps2:.2e}")
    print(f"    N_int = {result.N_int}")
    print(f"    d   = {result.d_m * 1e3:.6f} \u00b1 {result.sigma_d_m * 1e6:.3f} \u03bcm")
    print(f"    f   = {result.f_px:.2f} \u00b1 {result.sigma_f_px:.2f} px")
    print(f"    \u03b1   = {result.alpha_rad_px:.6e} \u00b1 {result.sigma_alpha:.2e} rad/px")
    print(f"    \u03c7\u00b2/dof = {result.chi2_dof:.4f}")

    # --- Build figure -------------------------------------------------------
    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[2.5, 1.5])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax_tab = fig.add_subplot(gs[1, :])

    # Helper: compute R² for a set of predicted vs observed values
    def _r2_fit(obs, pred, weights):
        ss_res = float(np.sum(weights * (obs - pred) ** 2))
        ss_tot = float(np.sum(weights * (obs - np.average(obs, weights=weights)) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # Line 1 (λ₁)
    w1 = 1.0 / (result.sr1_sq ** 2) if result.sr1_sq is not None else np.ones_like(result.r1_sq)
    r2_fit_1 = _r2_fit(result.r1_sq, result.pred1, w1)

    ax1.errorbar(result.p1, result.r1_sq, yerr=result.sr1_sq,
                 fmt="o", markersize=6, mfc="white", mec="steelblue",
                 ecolor="steelblue", capsize=4, label=f"\u03bb\u2081 = {result.lam1_nm} nm")
    ax1.plot(result.p1, result.pred1, "-", color="steelblue", lw=1.5)
    ax1.set_xlabel("Fringe index p", fontsize=11)
    ax1.set_ylabel("r\u00b2 (px\u00b2)", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.text(
        0.03, 0.97,
        f"S\u2081 = {result.S1:.2f} \u00b1 {result.sigma_S1:.2f} px\u00b2/fringe\n"
        f"\u03b5\u2081 = {result.eps1:.5f} \u00b1 {result.sigma_eps1:.5f}\n"
        f"R\u00b2 = {r2_fit_1:.6f}",
        transform=ax1.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round", fc="lightyellow"),
    )

    # Residuals inset — line 1
    ax1_inset = ax1.inset_axes([0.55, 0.05, 0.42, 0.30])
    resid1 = (result.r1_sq - result.pred1) / result.sr1_sq
    ax1_inset.stem(result.p1, resid1, linefmt="steelblue", markerfmt="o",
                   basefmt="k--")
    ax1_inset.axhline(0, color="k", lw=0.8, ls="--")
    ax1_inset.set_ylabel("Residual (\u03c3)", fontsize=7)
    ax1_inset.tick_params(labelsize=7)

    # Line 2 (λ₂)
    S2 = result.S2
    w2 = 1.0 / (result.sr2_sq ** 2) if result.sr2_sq is not None else np.ones_like(result.r2_sq)
    r2_fit_2 = _r2_fit(result.r2_sq, result.pred2, w2)

    ax2.errorbar(result.p2, result.r2_sq, yerr=result.sr2_sq,
                 fmt="o", markersize=6, mfc="white", mec="darkorange",
                 ecolor="darkorange", capsize=4, label=f"\u03bb\u2082 = {result.lam2_nm} nm")
    ax2.plot(result.p2, result.pred2, "-", color="darkorange", lw=1.5)
    ax2.set_xlabel("Fringe index p", fontsize=11)
    ax2.set_ylabel("r\u00b2 (px\u00b2)", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.text(
        0.03, 0.97,
        f"S\u2082 = {S2:.2f} px\u00b2/fringe  [constrained]\n"
        f"\u03b5\u2082 = {result.eps2:.5f} \u00b1 {result.sigma_eps2:.5f}\n"
        f"R\u00b2 = {r2_fit_2:.6f}",
        transform=ax2.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round", fc="lightyellow"),
    )

    # Residuals inset — line 2
    ax2_inset = ax2.inset_axes([0.55, 0.05, 0.42, 0.30])
    resid2 = (result.r2_sq - result.pred2) / result.sr2_sq
    ax2_inset.stem(result.p2, resid2, linefmt="darkorange", markerfmt="o",
                   basefmt="k--")
    ax2_inset.axhline(0, color="k", lw=0.8, ls="--")
    ax2_inset.set_ylabel("Residual (\u03c3)", fontsize=7)
    ax2_inset.tick_params(labelsize=7)

    # Results table
    ax_tab.axis("off")
    f_mm    = result.f_px  * CCD_PIXEL_PITCH_M * 1e3
    sf_mm   = result.sigma_f_px * CCD_PIXEL_PITCH_M * 1e3
    s2f_mm  = result.two_sigma_f_px * CCD_PIXEL_PITCH_M * 1e3

    tab_headers = ["Parameter", "Symbol", "Value", "1\u03c3 uncertainty",
                   "2\u03c3 uncertainty", "Units", "Notes"]
    tab_rows = [
        ["Etalon gap",    "d",
         f"{result.d_m * 1e3:.5f}",
         f"{result.sigma_d_m * 1e3:.5f}",
         f"{result.two_sigma_d_m * 1e3:.5f}",
         "mm", f"ICOS prior: {ICOS_GAP_MM:.3f} mm"],
        ["Focal length",  "f",
         f"{f_mm:.2f}",
         f"{sf_mm:.2f}",
         f"{s2f_mm:.2f}",
         "mm", "nominal 200 mm"],
        ["Plate scale",   "\u03b1",
         f"{result.alpha_rad_px:.4e}",
         f"{result.sigma_alpha:.4e}",
         f"{result.two_sigma_alpha:.4e}",
         "rad/px", "Tolansky: ~1.607e-4"],
        ["Frac. order \u03b5\u2081", "\u03b5\u2081",
         f"{result.epsilon_cal_1:.5f}",
         f"{result.sigma_eps1:.5f}",
         f"{2 * result.sigma_eps1:.5f}",
         "\u2014", f"\u03bb\u2081 = {NE_WAVELENGTH_1_NM} nm"],
        ["Frac. order \u03b5\u2082", "\u03b5\u2082",
         f"{result.epsilon_cal_2:.5f}",
         f"{result.sigma_eps2:.5f}",
         f"{2 * result.sigma_eps2:.5f}",
         "\u2014", f"\u03bb\u2082 = {NE_WAVELENGTH_2_NM} nm"],
        ["Reduced \u03c7\u00b2",    "\u03c7\u00b2/\u03bd",
         f"{result.chi2_dof:.3f}",
         "\u2014", "\u2014",
         "\u2014", "acceptable: 0.5\u20133.0"],
    ]
    tbl = ax_tab.table(
        cellText=tab_rows,
        colLabels=tab_headers,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)

    suptitle = (
        f"Figure 4 \u2014 Tolansky Two-Line Etalon Characterisation\n"
        f"File: {cal_path.name}  |  "
        f"d = {result.d_m * 1e3:.4f} \u00b1 {result.sigma_d_m * 1e3:.4f} mm  |  "
        f"f = {f_mm:.2f} \u00b1 {sf_mm:.2f} mm  |  "
        f"\u03b1 = {result.alpha_rad_px:.4e} \u00b1 {result.sigma_alpha:.4e} rad/px  |  "
        f"[close to exit]"
    )
    fig.suptitle(suptitle, fontsize=9)
    plt.tight_layout()
    plt.show()

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """
    Sequential execution of all six stages.
    Each figure blocks until the user closes it.
    """
    print("=" * 70)
    print("  WindCube FPI \u2014 Z01 Calibration Validation Script")
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
    figure_image_pair(load, meta)

    # Stage D
    print("\nStage D: Collecting centre seed and ROI from user...")
    roi = get_roi_from_user(load["cal_image"])
    print(f"  Seed: ({roi['cx_seed']:.1f}, {roi['cy_seed']:.1f}) px")
    print(f"  ROI:  {roi['roi_cols']}×{roi['roi_rows']} px")

    # Stage E
    print("\nStage E: Displaying ROI inspection (Figure 2)...")
    figure_roi_inspection(load["cal_image"], load["dark_image"], roi)
    # Note: figure_roi_inspection returns the visual diff array but we
    # do NOT pass it to Stage F. Stage F gets the raw cal image + dark.

    # Stage F
    print("\nStage F: Running S12 annular reduction + peak finding (Figure 3)...")
    fp = figure_reduction_peaks(
        cal_image  = load["cal_image"],
        dark_image = load["dark_image"],
        roi        = roi,
        cal_path   = load["cal_path"],
    )

    # Stage G
    print("\nStage G: Running S13 Tolansky analysis (Figure 4)...")
    result = figure_tolansky(fp, load["cal_path"])

    print("\n" + "=" * 70)
    print("  Z01 complete. All figures closed. Tolansky result summary:")
    print(f"    d   = {result.d_m * 1e3:.5f} \u00b1 {result.two_sigma_d_m * 1e3:.5f} mm (2\u03c3)")
    print(f"    f   = {result.f_px * CCD_PIXEL_PITCH_M * 1e3:.3f} \u00b1 "
          f"{result.two_sigma_f_px * CCD_PIXEL_PITCH_M * 1e3:.3f} mm (2\u03c3)")
    print(f"    \u03b1   = {result.alpha_rad_px:.4e} \u00b1 {result.two_sigma_alpha:.4e} rad/px (2\u03c3)")
    print(f"    \u03b5\u2081  = {result.epsilon_cal_1:.5f} \u00b1 {2 * result.sigma_eps1:.5f} (2\u03c3)")
    print(f"    \u03b5\u2082  = {result.epsilon_cal_2:.5f} \u00b1 {2 * result.sigma_eps2:.5f} (2\u03c3)")
    print(f"    \u03c7\u00b2/\u03bd = {result.chi2_dof:.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
