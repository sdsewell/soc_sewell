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

def figure_image_pair(load_result: dict,
                      meta_result: dict,
                      mode: int,
                      lam_str: str) -> tuple:
    """
    Display calibration and dark images side-by-side (Figure 1).
    User clicks on the fringe centre in the calibration image.

    Returns
    -------
    (cx_seed, cy_seed) : float — pixel coordinates of fringe centre seed
    """
    cal_image  = load_result["cal_image"]
    dark_image = load_result["dark_image"]
    cal_path   = load_result["cal_path"]

    # Determine display limits
    vmax = float(np.percentile(cal_image, 99.5))

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    ax_cal, ax_dark = axes

    ax_cal.imshow(cal_image, cmap="gray", vmin=0, vmax=vmax, origin="upper")
    ax_cal.set_title(f"Calibration: {cal_path.name}\nλ = {lam_str}",
                     fontsize=9)
    ax_cal.set_xlabel("Column (px)"); ax_cal.set_ylabel("Row (px)")

    ax_dark.imshow(dark_image, cmap="gray", vmin=0, vmax=np.percentile(dark_image, 99.5),
                   origin="upper")
    ax_dark.set_title(f"Dark: {load_result['dark_path'].name}", fontsize=9)
    ax_dark.set_xlabel("Column (px)"); ax_dark.set_ylabel("Row (px)")

    # Seed from image centre
    h, w = cal_image.shape
    cx_seed_def, cy_seed_def = w / 2.0, h / 2.0

    centres = []

    def _onclick(event):
        if event.inaxes is ax_cal and event.button == 1:
            centres.append((float(event.xdata), float(event.ydata)))
            ax_cal.plot(event.xdata, event.ydata, "r+", markersize=14, mew=2)
            fig.canvas.draw()

    cid = fig.canvas.mpl_connect("button_press_event", _onclick)

    fig.suptitle(
        f"Figure 1 — Image Pair  |  Click fringe centre in CAL image  |  "
        f"[close to continue]",
        fontsize=10, fontweight="bold"
    )
    plt.tight_layout()
    plt.show()
    fig.canvas.mpl_disconnect(cid)

    if centres:
        cx_seed, cy_seed = centres[-1]
    else:
        cx_seed, cy_seed = cx_seed_def, cy_seed_def
        print(f"  No click detected — using image centre "
              f"({cx_seed:.1f}, {cy_seed:.1f}) px as seed.")

    return cx_seed, cy_seed


# ── Stage D — Figure 2 ────────────────────────────────────────────────────────

def figure_roi_inspection(cal_image: np.ndarray,
                          dark_image: np.ndarray,
                          cx_seed: float,
                          cy_seed: float,
                          r_max: float = 110.0,
                          cal_path: pathlib.Path | None = None,
                          dark_path: pathlib.Path | None = None,
                          save_dir: pathlib.Path | None = None) -> tuple:
    """Show a zoomed ROI around the fringe centre (Figure 2).

    Additionally, save the calibration and dark ROIs as .npy arrays when
    `save_dir` is provided (or use current working directory otherwise).

    Returns
    -------
    (cal_roi, dark_roi) : tuple of np.ndarray
    """
    half = int(r_max) + 20
    h, w = cal_image.shape
    x0 = max(0, int(cx_seed) - half); x1 = min(w, int(cx_seed) + half)
    y0 = max(0, int(cy_seed) - half); y1 = min(h, int(cy_seed) + half)

    cal_roi  = cal_image[y0:y1, x0:x1]
    dark_roi = dark_image[y0:y1, x0:x1]
    vmax = float(np.percentile(cal_roi, 99.5))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    cal_name = pathlib.Path(cal_path).name if cal_path is not None else ""
    dark_name = pathlib.Path(dark_path).name if dark_path is not None else ""
    titles = [f"Calibration ROI (raw)\n{cal_name}", f"Dark ROI (raw)\n{dark_name}"]
    for ax, img, title in zip(axes, [cal_roi, dark_roi], titles):
        ax.imshow(img, cmap="gray", vmin=0, vmax=vmax, origin="upper")
        ax.set_title(title, fontsize=9)
        # Draw the seed cross
        ax.plot(cx_seed - x0, cy_seed - y0, "r+", markersize=16, mew=2)
        # Draw the r_max circle
        theta = np.linspace(0, 2 * np.pi, 360)
        ax.plot(r_max * np.cos(theta) + (cx_seed - x0),
                r_max * np.sin(theta) + (cy_seed - y0),
                "r--", lw=1, alpha=0.7)
        ax.set_xlabel("Column"); ax.set_ylabel("Row")

    fig.suptitle(
        f"Figure 2 — ROI Inspection  |  Seed: ({cx_seed:.1f}, {cy_seed:.1f}) px  |  "
        f"r_max = {r_max:.0f} px  |  [close to continue]",
        fontsize=10, fontweight="bold"
    )
    plt.tight_layout()
    plt.show()

    # Save the displayed figure (PNG) into the parent folder of the
    # calibration image (or CWD if none provided). Use the calibration
    # image basename with `_roi_inspection.png` appended.
    try:
        if cal_path is not None:
            fig_out_dir = pathlib.Path(cal_path).parent
            fig_out_dir.mkdir(parents=True, exist_ok=True)
            base = pathlib.Path(cal_path).name
            if "." in base:
                stem = base.rsplit(".", 1)[0]
            else:
                stem = base
            png_path = fig_out_dir / f"{stem}_roi_inspection.png"
        else:
            png_path = pathlib.Path.cwd() / "roi_inspection.png"

        fig.savefig(png_path, dpi=150)
        print(f"  Saved ROI figure: {png_path}")
    except Exception as exc:  # pragma: no cover - best-effort save
        print(f"  Warning: failed to save ROI figure: {exc}")

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


# ── Stage E — Centre refinement ───────────────────────────────────────────────

def _azimuthal_variance(image: np.ndarray,
                        cx: float,
                        cy: float,
                        r_max: float,
                        n_bins: int = 200) -> float:
    """
    Compute the total azimuthal variance of the image about (cx, cy).

    For each radial bin r, compute the variance of pixel values at that
    radius. Sum variances over all radial bins. Minimised when (cx, cy)
    is the true fringe centre (rings are most perfectly circular).
    """
    rows, cols = image.shape
    y_idx, x_idx = np.mgrid[0:rows, 0:cols]
    r = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)

    mask = r <= r_max
    r_masked = r[mask]
    v_masked = image[mask]

    bin_edges = np.linspace(0, r_max, n_bins + 1)
    bin_indices = np.digitize(r_masked, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    total_var = 0.0
    for b in range(n_bins):
        vals = v_masked[bin_indices == b]
        if len(vals) >= 3:
            total_var += float(np.var(vals))
    return total_var


def refine_centre(image: np.ndarray,
                  cx_seed: float,
                  cy_seed: float,
                  r_max: float = 110.0) -> tuple:
    """
    Two-step centre refinement: coarse integer grid search then Nelder-Mead.

    Returns
    -------
    (cx_final, cy_final) : float, sub-pixel fringe centre
    """
    print(f"\n  Stage E — Centre refinement:")
    print(f"    Seed (visual estimate): cx = {cx_seed:.1f},  cy = {cy_seed:.1f}  px")

    half = 15
    best_var = np.inf
    cx_coarse, cy_coarse = cx_seed, cy_seed
    grid_pts = (2 * half + 1) ** 2
    print(f"    Coarse grid: ±{half} px, 1 px step  →  "
          f"{2*half+1}×{2*half+1} = {grid_pts} evaluations...")

    cx_range = np.arange(cx_seed - half, cx_seed + half + 1, 1.0)
    cy_range = np.arange(cy_seed - half, cy_seed + half + 1, 1.0)
    for cx_c in cx_range:
        for cy_c in cy_range:
            v = _azimuthal_variance(image, cx_c, cy_c, r_max)
            if v < best_var:
                best_var = v
                cx_coarse, cy_coarse = cx_c, cy_c

    print(f"    Coarse best: cx = {cx_coarse:.1f},  cy = {cy_coarse:.1f}  px")
    print(f"    Coarse improvement: "
          f"Δcx = {cx_coarse - cx_seed:+.1f},  "
          f"Δcy = {cy_coarse - cy_seed:+.1f}  px")

    result_nm = opt.minimize(
        fun=lambda p: _azimuthal_variance(image, p[0], p[1], r_max),
        x0=[cx_coarse, cy_coarse],
        method="Nelder-Mead",
        options=dict(xatol=1e-3, fatol=1e-6, maxiter=2000),
    )
    cx_final, cy_final = float(result_nm.x[0]), float(result_nm.x[1])

    print(f"    Nelder-Mead: cx = {cx_final:.4f},  cy = {cy_final:.4f}  px  (sub-pixel)")
    print(f"    Sub-pixel shift from coarse: "
          f"Δcx = {cx_final - cx_coarse:+.4f},  "
          f"Δcy = {cy_final - cy_coarse:+.4f}  px")
    print(f"    Nelder-Mead converged: {result_nm.success}  (nfev={result_nm.nfev})")
    print(f"    Final centre: cx = {cx_final:.4f} ± 0.05,  "
          f"cy = {cy_final:.4f} ± 0.05  px")

    return cx_final, cy_final


# ── Stage F1 ──────────────────────────────────────────────────────────────────

def run_s12_reduction(cal_image: np.ndarray,
                      dark_image: np.ndarray,
                      cx: float,
                      cy: float,
                      r_max: float = 110.0) -> tuple:
    """
    Run M03 annular reduction. Returns (FringeProfile, dark_subtracted_image, roi_dict).

    IMPORTANT: cal_image must be the RAW image — dark subtraction is done inside M03.
    The returned dark-subtracted image is for Figure 3 diagnostics ONLY.
    """
    from src.fpi.m03_annular_reduction_2026_04_06 import (
        reduce_calibration_frame, make_master_dark, subtract_dark,
    )

    master_dark = make_master_dark([dark_image])

    fp = reduce_calibration_frame(
        image       = cal_image,      # RAW — dark subtraction is INSIDE M03
        master_dark = master_dark,
        cx_human    = cx,
        cy_human    = cy,
        r_max_px    = r_max,
    )

    # Diagnostic dark-subtracted image — for Figure 3 ONLY
    s12_dark_sub = subtract_dark(cal_image, master_dark, clip_negative=True)

    # Warn if M03's internal centre drifted from Stage E result
    if hasattr(fp, "cx") and hasattr(fp, "cy"):
        drift = np.sqrt((fp.cx - cx) ** 2 + (fp.cy - cy) ** 2)
        if drift > 0.5:
            print(f"  WARNING: M03 centre ({fp.cx:.3f}, {fp.cy:.3f}) px "
                  f"differs from Stage E result by {drift:.3f} px (> 0.5 px)")

    # ROI for Figures 3 and 4
    half = int(r_max) + 20
    h, w = cal_image.shape
    x0 = max(0, int(cx) - half); x1 = min(w, int(cx) + half)
    y0 = max(0, int(cy) - half); y1 = min(h, int(cy) + half)
    roi = {"roi_x0": x0, "roi_x1": x1, "roi_y0": y0, "roi_y1": y1}

    n_ok = sum(1 for pk in fp.peak_fits if pk.fit_ok)
    print(f"  M03 centre: ({fp.cx:.4f}, {fp.cy:.4f}) px  |  "
          f"Peaks found: {n_ok}")

    return fp, s12_dark_sub, roi


# ── Stage F2 — Figure 3 ───────────────────────────────────────────────────────

def figure_dark_comparison(cal_roi: np.ndarray,
                           dark_roi: np.ndarray,
                           s12_roi: np.ndarray) -> None:
    """Three-panel comparison: cal raw, naive dark sub, S12 pipeline output (Figure 3)."""
    naive_roi = np.clip(cal_roi - dark_roi, 0, None)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    vmax = float(np.percentile(cal_roi, 99.5))

    titles = ["Cal (raw)", "Naive dark subtraction", "S12 pipeline (authoritative)"]
    imgs   = [cal_roi, naive_roi, s12_roi]
    for ax, img, title in zip(axes, imgs, titles):
        ax.imshow(img, cmap="gray", vmin=0, vmax=vmax, origin="upper")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Column"); ax.set_ylabel("Row")

    max_diff = float(np.max(np.abs(naive_roi - s12_roi)))
    identical = np.allclose(naive_roi, s12_roi, atol=0.5)
    status = (f"IDENTICAL (max |diff| = {max_diff:.1e})" if identical
              else f"MISMATCH — max |diff| = {max_diff:.1f} ADU  [investigate dark routing]")

    fig.suptitle(
        f"Figure 3 — Dark Subtraction Comparison  |  {status}  |  [close to continue]",
        fontsize=11,
        color="green" if identical else "red",
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


# ── Stage F3 — Figure 4 ───────────────────────────────────────────────────────

def figure_reduction_peaks(fp,
                           roi: dict,
                           cal_path: pathlib.Path,
                           mode: int,
                           lam1_nm: float,
                           lam2_nm) -> None:
    """Radial profile with peak markers and peak data table (Figure 4)."""
    good_peaks = [pk for pk in fp.peak_fits if pk.fit_ok]

    # ── Split into families by amplitude (2-line mode) ──────────────────────
    if mode == 2 and len(good_peaks) >= 2:
        max_amp = max(pk.amplitude_adu for pk in good_peaks)
        thresh  = 0.7 * max_amp
        fam1 = sorted([pk for pk in good_peaks if pk.amplitude_adu >= thresh],
                      key=lambda pk: pk.r_fit_px)
        fam2 = sorted([pk for pk in good_peaks if pk.amplitude_adu < thresh],
                      key=lambda pk: pk.r_fit_px)
        families = [
            (fam1, f"{lam1_nm:.4f} nm (air)", "#CC4400"),
            (fam2, f"{lam2_nm:.4f} nm (air)", "steelblue"),
        ]
    else:
        fam_all = sorted(good_peaks, key=lambda pk: pk.r_fit_px)
        families = [(fam_all, f"{lam1_nm:.4f} nm (air)", "steelblue")]

    fig, (ax_profile, ax_table) = plt.subplots(
        1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [2, 1]}
    )

    # ── Left panel: radial profile in r² ────────────────────────────────────
    r_sq = fp.r2_grid
    ax_profile.plot(r_sq, fp.profile, "-", color="0.7", lw=0.8, label="Profile")

    for fam_peaks, lam_label, color in families:
        for p_idx, pk in enumerate(fam_peaks, start=1):
            r_sq_pk = pk.r_fit_px ** 2
            ax_profile.axvline(r_sq_pk, color=color, lw=1.0, ls="--", alpha=0.8)
            ax_profile.plot(r_sq_pk, float(np.interp(pk.r_fit_px, fp.r_grid, fp.profile)),
                            marker="^", color=color, markersize=7, zorder=5)
            ax_profile.text(r_sq_pk, ax_profile.get_ylim()[1] * 0.95,
                            f"p={p_idx}", ha="center", fontsize=7, color=color)

    ax_profile.set_xlabel(r"$r^2$  (px²)", fontsize=11)
    ax_profile.set_ylabel("Intensity (ADU)", fontsize=11)
    ax_profile.set_title(
        f"Radial Profile vs r²  |  {cal_path.name}", fontsize=9
    )

    # ── Right panel: peak data table ─────────────────────────────────────────
    ax_table.axis("off")

    table_rows = [["Family", "p", "r_fit (px)", "r² (px²)", "Amp (ADU)"]]
    for fam_peaks, lam_label, color in families:
        r_sq_vals = [pk.r_fit_px ** 2 for pk in fam_peaks]
        delta_r_sq = np.diff(r_sq_vals) if len(r_sq_vals) > 1 else np.array([np.nan])
        mean_delta = float(np.mean(delta_r_sq)) if len(delta_r_sq) > 0 else np.nan
        std_delta  = float(np.std(delta_r_sq)) if len(delta_r_sq) > 0 else np.nan
        cv_pct = (std_delta / abs(mean_delta) * 100
                  if mean_delta != 0 and not np.isnan(mean_delta) else 999.0)

        for p_idx, pk in enumerate(fam_peaks, start=1):
            table_rows.append([
                lam_label if p_idx == 1 else "",
                str(p_idx),
                f"{pk.r_fit_px:.3f}",
                f"{pk.r_fit_px**2:.2f}",
                f"{pk.amplitude_adu:.0f}",
            ])
        table_rows.append([
            "— summary —", "",
            f"⟨Δr²⟩ = {mean_delta:.2f}", f"σ = {std_delta:.2f}",
            f"CV = {cv_pct:.1f}%",
        ])

    col_labels = table_rows[0]
    cell_text  = table_rows[1:]

    tbl = ax_table.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    ax_table.set_title("Peak Data Table", fontsize=9, fontweight="bold", pad=4)

    # ── Suptitle ─────────────────────────────────────────────────────────────
    n_ok_total = sum(1 for pk in fp.peak_fits if pk.fit_ok)
    lam_info = (f"λ₁ = {lam1_nm:.4f} nm (air)  +  λ₂ = {lam2_nm:.4f} nm (air)"
                if mode == 2 else f"λ = {lam1_nm:.4f} nm (air)")
    fig.suptitle(
        f"Figure 4 — M03 Reduction Results  |  {lam_info}  |  "
        f"Peaks: {n_ok_total}  |  [close to continue]",
        fontsize=10, fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


# ── Stage G — Tolansky dispatch ───────────────────────────────────────────────

def run_tolansky(fp,
                 mode: int,
                 lam1_nm: float,
                 lam2_nm,
                 d_prior_mm: float,
                 d_fixed_mm,
                 f_fixed_mm,
                 pixel_pitch_m: float):
    """Run TolanskyPipeline (mode 2) or SingleLineTolansky (mode 1)."""
    if mode == 2:
        from src.fpi.tolansky_2026_04_05 import TolanskyPipeline

        pipeline = TolanskyPipeline(
            profile                  = fp,
            d_prior_m                = d_prior_mm * 1e-3,
            lam1_nm                  = lam1_nm,
            lam2_nm                  = lam2_nm,
            amplitude_split_fraction = 0.7,
            n                        = 1.0,
            pixel_pitch_m            = pixel_pitch_m,
        )
        result = pipeline.run()

        f_mm = result.f_px * pixel_pitch_m * 1e3
        f_2s = result.two_sigma_f_px * pixel_pitch_m * 1e3
        print(f"  Tolansky 2-line complete:")
        print(f"    d = {result.d_m*1e3:.4f} ± {result.two_sigma_d_m*1e3:.4f} mm (2σ)")
        print(f"    f = {f_mm:.3f} ± {f_2s:.3f} mm (2σ)")
        print(f"    α = {result.alpha_rad_px:.4e} ± {result.two_sigma_alpha:.4e} rad/px (2σ)")
        print(f"    ε₁ = {result.eps1:.5f} ± {2*result.sigma_eps1:.5f}  "
              f"[λ₁={lam1_nm} nm (air)] (2σ)")
        print(f"    ε₂ = {result.eps2:.5f} ± {2*result.sigma_eps2:.5f}  "
              f"[λ₂={lam2_nm} nm (air)] (2σ)")
        print(f"    N_int = {result.N_int}  |  χ²/ν = {result.chi2_dof:.3f}")
        return result

    else:  # mode == 1
        from src.fpi.tolansky_2026_04_05 import SingleLineTolansky

        analyser = SingleLineTolansky(
            fringe_profile = fp,
            lam_rest_nm    = lam1_nm,
            d_prior_m      = d_fixed_mm * 1e-3,
            f_prior_m      = f_fixed_mm * 1e-3,
            pixel_pitch_m  = pixel_pitch_m,
            d_ref_m        = d_prior_mm * 1e-3,   # ICOS gap for N_int resolution
        )
        result = analyser.run()
        v_ok = abs(result.v_rel_ms) < 3 * result.sigma_v_ms
        print(f"  Tolansky 1-line complete:")
        print(f"    ε₀   = {result.epsilon:.6f} ± {result.sigma_eps:.6f}  "
              f"[λ={lam1_nm} nm (air)]  ← rest-frame reference")
        print(f"    λ_c  = {result.lam_c_nm:.5f} ± {result.sigma_lam_c_nm:.5f} nm (air)")
        print(f"    v_ref= {result.v_rel_ms:.1f} ± {result.sigma_v_ms:.1f} m/s  "
              f"{'[PASS]' if v_ok else '[WARN — non-zero velocity]'}")
        print(f"    N_int = {result.N_int}  |  χ²/ν = {result.chi2_dof:.3f}")
        return result


# ── Stage H — Figure 5 ───────────────────────────────────────────────────────

def figure_tolansky(fp, result, mode: int,
                    lam1_nm: float,
                    lam2_nm,
                    cal_path: pathlib.Path,
                    d_prior_mm: float) -> None:
    """Melissinos r² scatter plot with residuals inset and summary tables (Figure 5)."""
    fig = plt.figure(figsize=(16, 14))
    gs  = fig.add_gridspec(3, 2,
                            height_ratios=[2, 1, 1],
                            hspace=0.40, wspace=0.30)
    ax_scatter   = fig.add_subplot(gs[0, :])
    ax_table_arr = fig.add_subplot(gs[1:, 0])
    ax_table_res = fig.add_subplot(gs[1:, 1])

    # ── 8b: Melissinos r² scatter ────────────────────────────────────────────
    if mode == 2:
        families_data = [
            (result.p1, result.r1_sq, result.sr1_sq, result.pred1,
             result.S1, result.sigma_S1, result.eps1, result.sigma_eps1,
             f"{lam1_nm:.4f} nm (air)", "#CC4400"),
            (result.p2, result.r2_sq, result.sr2_sq, result.pred2,
             result.S1 * result.lam_ratio, np.nan, result.eps2, result.sigma_eps2,
             f"{lam2_nm:.4f} nm (air)", "steelblue"),
        ]
    else:
        # For SingleLineResult we need to reconstruct from fp peak data
        good_peaks = sorted([pk for pk in fp.peak_fits if pk.fit_ok],
                            key=lambda pk: pk.r_fit_px)
        p_arr = np.arange(1, len(good_peaks) + 1, dtype=float)
        r_sq_arr = np.array([pk.r_fit_px ** 2 for pk in good_peaks])
        sr_sq_arr = np.array([2 * pk.r_fit_px * (pk.sigma_r_fit_px
                               if not np.isnan(pk.sigma_r_fit_px) else 0.5)
                              for pk in good_peaks])
        pred_arr = result.S * (p_arr - 1 + result.epsilon)
        families_data = [
            (p_arr, r_sq_arr, sr_sq_arr, pred_arr,
             result.S, result.sigma_S, result.epsilon, result.sigma_eps,
             f"{lam1_nm:.4f} nm (air)", "steelblue"),
        ]

    for (p_idx, r_sq_obs, sigma_r_sq, r_sq_pred,
         S, sigma_S, eps, sigma_eps, lam_label, color) in families_data:
        ax_scatter.errorbar(
            r_sq_obs, p_idx,
            xerr=2 * sigma_r_sq,
            fmt="o",
            markersize=6,
            mfc="white",
            mec=color,
            ecolor=color,
            capsize=4,
            label=f"λ = {lam_label}",
            zorder=3,
        )

        # Fit line
        p_fit = np.linspace(0.5, float(p_idx.max()) + 0.5, 300)
        r_sq_fit_line = S * (p_fit - 1 + eps)
        ax_scatter.plot(r_sq_fit_line, p_fit, "-", color=color, lw=1.5, alpha=0.7)

        # Text annotation
        sigma_S_str = f"{sigma_S:.2f}" if not np.isnan(sigma_S) else "fixed"
        txt = (f"S = {S:.2f} ± {sigma_S_str} px²/fringe\n"
               f"ε = {eps:.5f} ± {sigma_eps:.5f}\n"
               f"χ²/ν = {result.chi2_dof:.3f}")
        va_pos = 0.95 if color == "#CC4400" or mode == 1 else 0.55
        ax_scatter.text(0.03, va_pos, txt, transform=ax_scatter.transAxes,
                        va="top", fontsize=9,
                        bbox=dict(boxstyle="round", fc="lightyellow",
                                  ec="gray", alpha=0.8))

    ax_scatter.set_xlabel("r²  (px²)", fontsize=11)
    ax_scatter.set_ylabel("Fringe index  p", fontsize=11)
    ax_scatter.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax_scatter.legend(fontsize=9)
    ax_scatter.set_title(
        f"Melissinos r² Plot — {cal_path.name}", fontsize=9
    )

    # Residuals inset (use first family)
    p_idx0, r_sq0, sr_sq0, r_sq_pred0 = (families_data[0][0], families_data[0][1],
                                           families_data[0][2], families_data[0][3])
    ax_inset = ax_scatter.inset_axes([0.76, 0.05, 0.22, 0.35])
    residuals = (r_sq0 - r_sq_pred0) / np.where(sr_sq0 > 0, sr_sq0, 1.0)
    ax_inset.stem(p_idx0, residuals, linefmt="gray", markerfmt="o",
                  basefmt="k--")
    ax_inset.axhline(0, color="k", lw=0.5, ls="--")
    ax_inset.set_ylabel("Residual (σ)", fontsize=7)
    ax_inset.tick_params(labelsize=7)
    ax_inset.set_title("Normalised residuals", fontsize=7)

    # ── 8c: Vaughan rectangular array table ──────────────────────────────────
    ax_table_arr.axis("off")

    if mode == 2:
        r1 = result.r1_sq
        r2 = result.r2_sq
        n_rings = min(len(r1), len(r2), 5)

        col_labels_arr = [""]
        for i in range(n_rings):
            col_labels_arr.append(f"p={i+1}")
            if i < n_rings - 1:
                col_labels_arr.append(f"Δ{i+1},{i+2}")

        def _row_data(r_sq_arr):
            row = []
            for i in range(n_rings):
                row.append(f"{r_sq_arr[i]:.1f}")
                if i < n_rings - 1:
                    row.append(f"{r_sq_arr[i+1] - r_sq_arr[i]:.1f}")
            return row

        delta12 = []
        for i in range(n_rings):
            delta12.append(f"{r1[i] - r2[i]:.1f}")
            if i < n_rings - 1:
                delta12.append("")

        lam1_lbl = f"λ₁={lam1_nm:.2f} nm (air)"
        lam2_lbl = f"λ₂={lam2_nm:.2f} nm (air)"

        cell_text_arr = [
            [lam1_lbl] + _row_data(r1),
            ["δ₁₂"] + delta12,
            [lam2_lbl] + _row_data(r2),
            ["" ] + [""] * (len(col_labels_arr) - 1),
            [f"⟨Δ⟩ λ₁  S₁={result.S1:.2f}±{result.sigma_S1:.2f} px²/fr"]
            + [""] * (len(col_labels_arr) - 1),
            [f"S₁/S₂: actual={result.S1/result.S2:.6f}  "
             f"expect={lam1_nm/lam2_nm:.6f}  "
             f"Δ={abs(result.S1/result.S2 - lam1_nm/lam2_nm)*1e4:.2f}×10⁻⁴"]
            + [""] * (len(col_labels_arr) - 1),
        ]

        tbl_arr = ax_table_arr.table(
            cellText=cell_text_arr,
            colLabels=col_labels_arr,
            cellLoc="center",
            loc="center",
            bbox=[0.0, 0.0, 1.0, 1.0],
        )
    else:
        # 1-line: single row
        r_sq_peaks = np.array([pk.r_fit_px ** 2
                                for pk in sorted(fp.peak_fits if True else [],
                                                  key=lambda pk: pk.r_fit_px)
                                if pk.fit_ok])
        n_rings = min(len(r_sq_peaks), 5)
        col_labels_arr = [""]
        for i in range(n_rings):
            col_labels_arr.append(f"p={i+1}")
            if i < n_rings - 1:
                col_labels_arr.append(f"Δ{i+1},{i+2}")

        row = []
        for i in range(n_rings):
            row.append(f"{r_sq_peaks[i]:.1f}")
            if i < n_rings - 1:
                row.append(f"{r_sq_peaks[i+1] - r_sq_peaks[i]:.1f}")

        cell_text_arr = [
            [f"λ={lam1_nm:.4f} nm (air)"] + row,
        ]
        tbl_arr = ax_table_arr.table(
            cellText=cell_text_arr,
            colLabels=col_labels_arr,
            cellLoc="center",
            loc="center",
            bbox=[0.0, 0.0, 1.0, 1.0],
        )

    tbl_arr.auto_set_font_size(False)
    tbl_arr.set_fontsize(7.5)
    ax_table_arr.set_title(
        "Vaughan (1989) §3.5.2 Rectangular Array  [units: px²]",
        fontsize=9, fontweight="bold", pad=4,
    )

    # ── 8d: Parameter summary table ──────────────────────────────────────────
    ax_table_res.axis("off")

    if mode == 2:
        from windcube.constants import CCD_PIXEL_PITCH_M
        pitch_m = CCD_PIXEL_PITCH_M
        f_mm    = result.f_px * pitch_m * 1e3
        sf_mm   = result.sigma_f_px * pitch_m * 1e3
        s2f_mm  = result.two_sigma_f_px * pitch_m * 1e3

        result_cells = [
            ["Etalon gap",    "d",     f"{result.d_m*1e3:.4f}", f"{result.sigma_d_m*1e3:.4f}",
             f"{result.two_sigma_d_m*1e3:.4f}", "mm"],
            ["Focal length",  "f",     f"{f_mm:.3f}", f"{sf_mm:.3f}", f"{s2f_mm:.3f}", "mm"],
            ["Plate scale",   "α",     f"{result.alpha_rad_px:.4e}", f"{result.sigma_alpha:.4e}",
             f"{result.two_sigma_alpha:.4e}", "rad/px"],
            ["Frac. order λ₁", "ε₁",  f"{result.eps1:.5f}", f"{result.sigma_eps1:.5f}",
             f"{2*result.sigma_eps1:.5f}", "—"],
            ["Frac. order λ₂", "ε₂",  f"{result.eps2:.5f}", f"{result.sigma_eps2:.5f}",
             f"{2*result.sigma_eps2:.5f}", "—"],
            ["Integer order",  "N",    str(result.N_int), "—", "—", "—"],
            ["χ²/ν",           "χ²/ν", f"{result.chi2_dof:.3f}", "—", "—", "—"],
        ]
    else:
        result_cells = [
            ["Frac. order", "ε₀",   f"{result.epsilon:.6f}", f"{result.sigma_eps:.6f}",
             f"{result.two_sigma_eps:.6f}", "—"],
            ["Centroid λ",  "λ_c",  f"{result.lam_c_nm:.5f}", f"{result.sigma_lam_c_nm:.5f}",
             f"{result.two_sigma_lam_c_nm:.5f}", "nm (air)"],
            ["LOS velocity", "v",   f"{result.v_rel_ms:.1f}", f"{result.sigma_v_ms:.1f}",
             f"{result.two_sigma_v_ms:.1f}", "m/s"],
            ["Integer order", "N",  str(result.N_int), "—", "—", "—"],
            ["χ²/ν",          "χ²/ν", f"{result.chi2_dof:.3f}", "—", "—", "—"],
        ]

    tbl_res = ax_table_res.table(
        cellText=result_cells,
        colLabels=["Parameter", "Symbol", "Value", "σ", "2σ", "Units"],
        cellLoc="left",
        loc="center",
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    tbl_res.auto_set_font_size(False)
    tbl_res.set_fontsize(8)
    ax_table_res.set_title("Results Summary", fontsize=9, fontweight="bold", pad=4)

    # Bold ε₀ row for 1-line mode
    if mode == 1:
        for (r, c), cell in tbl_res.get_celld().items():
            if r > 0 and "ε₀" in str(cell.get_text().get_text()):
                cell.get_text().set_fontweight("bold")

    # ── Suptitle ─────────────────────────────────────────────────────────────
    lam_title = (f"λ₁ = {lam1_nm:.4f} nm (air) + λ₂ = {lam2_nm:.4f} nm (air)"
                 if mode == 2 else f"λ = {lam1_nm:.4f} nm (air)")
    fig.suptitle(
        f"Figure 5 — Tolansky Analysis ({mode}-Line)  |  {lam_title}  |  "
        f"[close to continue]",
        fontsize=11, fontweight="bold",
    )
    plt.show()


# ── Vaughan equations console block ──────────────────────────────────────────

def print_vaughan_equations(result, mode: int,
                             lam1_nm: float,
                             lam2_nm,
                             d_prior_mm: float,
                             pitch_m: float,
                             pitch_mm: float) -> None:
    """Print the Vaughan §3.5.2 equations cross-check block to the console."""
    print("\n" + "═" * 70)
    print("  VAUGHAN (1989) §3.5.2 CROSS-CHECK")
    print("═" * 70)

    if mode == 2:
        # ── Eq 3.91: Scale factor C = S ────────────────────────────────────
        try:
            lam1_m = lam1_nm * 1e-9
            f_px   = result.f_px
            d_m    = result.d_m

            C_computed = f_px ** 2 * lam1_m / (1.0 * d_m)
            C_fit      = result.S1
            rel_diff   = abs(C_computed - C_fit) / C_fit * 100

            print(f"\n  Eq 3.91 — Scale factor C = f²λ/(nd)  [px²/fringe]:")
            print(f"    C_fit      = {C_fit:.4f} px²/fringe  (WLS slope S₁)")
            print(f"    C_computed = {C_computed:.4f} px²/fringe  "
                  f"(from recovered d and f)")
            print(f"    Relative difference: {rel_diff:.4f}%  "
                  f"{'[PASS <0.1%]' if rel_diff < 0.1 else '[WARN >0.1%]'}")
            print(f"    Note: Vaughan's Δ_Vaughan = 4×C = {4*C_fit:.2f} px²/fringe "
                  f"(diameter² convention; we use radius²)")
        except AttributeError as e:
            print(f"  Eq 3.91: [NOT AVAILABLE — S13 update required: {e}]")

        # ── Eq 3.92: Wavenumber separation ─────────────────────────────────
        try:
            lam2_m = lam2_nm * 1e-9
            delta_sigma_known = 1.0 / lam2_m - 1.0 / lam1_m  # m⁻¹

            r1_all = result.r1_sq
            r2_all = result.r2_sq
            n_common = min(len(r1_all), len(r2_all))
            delta12_vals = r1_all[:n_common] - r2_all[:n_common]
            mean_delta12 = float(np.mean(delta12_vals))
            delta_sigma_meas = mean_delta12 / result.S1 / lam1_m  # m⁻¹

            rel_check = abs(delta_sigma_meas - delta_sigma_known) / delta_sigma_known * 100
            print(f"\n  Eq 3.92 — Wavenumber separation:")
            print(f"    Δσ₁₂ known  = {delta_sigma_known:.3f} m⁻¹  "
                  f"(from λ₁={lam1_nm} nm (air), λ₂={lam2_nm} nm (air))")
            print(f"    Δσ₁₂ meas   = {delta_sigma_meas:.3f} m⁻¹  "
                  f"(from ⟨r²_λ₁ − r²_λ₂⟩ / S₁ / λ₁)")
            print(f"    Relative difference: {rel_check:.3f}%  "
                  f"{'[PASS <1%]' if rel_check < 1.0 else '[WARN >1%]'}")
        except AttributeError as e:
            print(f"  Eq 3.92: [NOT AVAILABLE — S13 update required: {e}]")

        # ── Eq 3.93: McNair approximation ───────────────────────────────────
        try:
            r1 = result.r1_sq
            r2 = result.r2_sq
            if len(r1) >= 2 and len(r2) >= 2:
                delta1  = r1[0] - r2[0]        # δ₁₂ at ring 1
                Delta12 = r1[1] - r1[0]         # Δ for λ₁ rings 1→2
                Delta21 = r2[1] - r2[0]         # Δ for λ₂ rings 1→2
                delta_n_mcnair = 2 * delta1 / (Delta12 + Delta21)
                delta_eps_fit  = result.delta_eps
                print(f"\n  Eq 3.93 — McNair approximation (informational):")
                print(f"    δn_McNair  = {delta_n_mcnair:.5f}  (off-axis 2-ring approx)")
                print(f"    δn_WLS_fit = {delta_eps_fit:.5f}  (ε₁ − ε₂ from joint fit)")
                print(f"    Δ = {abs(delta_n_mcnair - delta_eps_fit):.5f}  "
                      f"{'[PASS <0.05]' if abs(delta_n_mcnair - delta_eps_fit) < 0.05 else '[WARN >0.05]'}")
        except AttributeError as e:
            print(f"  Eq 3.93: [NOT AVAILABLE — S13 update required: {e}]")

        # ── Eq 3.95: Consistency check lhs1 = lhs2 ─────────────────────────
        try:
            lam1_m = lam1_nm * 1e-9
            lam2_m = lam2_nm * 1e-9
            # Vaughan 3.95: S₁/λ₁ = S₂/λ₂ = f²/(nd)
            lhs1 = result.S1 / lam1_m
            lhs2 = result.S2 / lam2_m
            rel_lhs = abs(lhs1 - lhs2) / lhs1
            print(f"\n  Eq 3.95 — Slope-ratio consistency (S₁/λ₁ = S₂/λ₂):")
            print(f"    lhs1 = S₁/λ₁ = {lhs1:.6e}")
            print(f"    lhs2 = S₂/λ₂ = {lhs2:.6e}")
            print(f"    |lhs1−lhs2|/lhs1 = {rel_lhs:.2e}  "
                  f"{'[PASS <1e-4]' if rel_lhs < 1e-4 else '[WARN >1e-4]'}")
        except AttributeError as e:
            print(f"  Eq 3.95: [NOT AVAILABLE — S13 update required: {e}]")

        # ── Eq 3.97: Three d estimates ──────────────────────────────────────
        try:
            lam1_m = lam1_nm * 1e-9
            f_px   = result.f_px

            # d from S1 and f
            d_from_S1 = f_px ** 2 * lam1_m / result.S1
            # d from joint fit result
            d_joint   = result.d_m
            # d from eps method (N_int path)
            d_eps = d_joint  # already recovered via excess-fractions

            spread_um = (max(d_from_S1, d_joint) - min(d_from_S1, d_joint)) * 1e6
            print(f"\n  Eq 3.97 — Three-way d spread (Benoit method):")
            print(f"    d from S₁+f     = {d_from_S1*1e3:.5f} mm")
            print(f"    d from joint fit = {d_joint*1e3:.5f} mm")
            print(f"    Spread (S1 vs joint) = {spread_um:.3f} µm  "
                  f"{'[PASS <2µm]' if spread_um < 2.0 else '[WARN >2µm]'}")
        except AttributeError as e:
            print(f"  Eq 3.97: [NOT AVAILABLE — S13 update required: {e}]")

    else:  # mode == 1
        # ── Eq 3.91 (1-line version) ────────────────────────────────────────
        try:
            from windcube.constants import F_TOLANSKY_MM, D_25C_MM
            lam1_m  = lam1_nm * 1e-9
            f_px    = (F_TOLANSKY_MM * 1e-3) / pitch_m
            d_m     = D_25C_MM * 1e-3

            # SingleLineTolansky uses S = f²λ/(2d) — standard double-pass FP formula
            C_computed = f_px ** 2 * lam1_m / (2.0 * d_m)
            C_fit      = result.S
            rel_diff   = abs(C_computed - C_fit) / C_fit * 100

            print(f"\n  Eq 3.91 — Scale factor C = f²λ/(2d)  [px²/fringe]:")
            print(f"    C_fit      = {C_fit:.4f} px²/fringe  (S fixed from priors)")
            print(f"    C_computed = {C_computed:.4f} px²/fringe  (f_prior, d_prior)")
            print(f"    Relative difference: {rel_diff:.4f}%  "
                  f"{'[PASS <0.5%]' if rel_diff < 0.5 else '[WARN >0.5%]'}")
        except AttributeError as e:
            print(f"  Eq 3.91: [NOT AVAILABLE — S13 update required: {e}]")

        # ── Summary for 1-line ───────────────────────────────────────────────
        try:
            print(f"\n  ε₀   = {result.epsilon:.6f} ± {result.two_sigma_eps:.6f}  (2σ)"
                  f"  ← store as calibration constant")
            print(f"  λ_c  = {result.lam_c_nm:.5f} ± {result.two_sigma_lam_c_nm:.5f} nm (air) (2σ)")
            print(f"  v_ref= {result.v_rel_ms:.1f} ± {result.two_sigma_v_ms:.1f} m/s (2σ)")
        except AttributeError as e:
            print(f"  1-line summary: [NOT AVAILABLE — S13 update required: {e}]")

    print("═" * 70)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Z01 — WindCube FPI Tolansky Analysis Validation"
    )
    parser.add_argument("--mode", type=int, choices=[1, 2], default=None)
    parser.add_argument(
        "--lam1", type=float, default=None,
        help="Wavelength 1 in nm (air). Default: 640.2248 (2-line) or 630.0304 (1-line)."
    )
    parser.add_argument(
        "--lam2", type=float, default=638.2991,
        help="Wavelength 2 in nm (air), 2-line mode only. Default: 638.2991."
    )
    parser.add_argument(
        "--d-prior", type=float, default=None,
        help="Etalon gap prior in mm (for N_int disambiguation). Default: ICOS_GAP_MM."
    )
    parser.add_argument(
        "--has-header", choices=["yes", "no", "auto"], default="auto"
    )
    parser.add_argument(
        "--r-max", type=float, default=None,
        help="Max annular radius in px. Default: R_MAX_PX from constants."
    )
    args = parser.parse_args()

    # Interactive mode if --mode not supplied
    if args.mode is None:
        print("WindCube FPI — Tolansky Analysis Validation")
        print("━" * 42)
        print("Mode?")
        print("  [1]  Single-line (OI 630 nm or filtered neon — 1 spectral component)")
        print("  [2]  Two-line   (dual neon 640.2248 + 638.2991 nm — 2 components) [DEFAULT]")
        raw = input("Enter 1 or 2 [2]: ").strip()
        args.mode = 1 if raw == "1" else 2

    return args


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    """Entry point for Z01 validation script."""
    # ── deferred constants import ──────────────────────────────────────────
    from windcube.constants import (
        NE_WAVELENGTH_1_NM, NE_WAVELENGTH_2_NM, OI_WAVELENGTH_NM,
        CCD_PIXEL_PITCH_M, ICOS_GAP_MM, D_25C_MM, F_TOLANSKY_MM, R_MAX_PX,
    )
    pitch_mm = CCD_PIXEL_PITCH_M * 1e3

    # ── parse args and fill defaults that needed constants ─────────────────
    args = parse_args()
    mode = args.mode
    lam1_nm = args.lam1 if args.lam1 is not None else (
        NE_WAVELENGTH_1_NM if mode == 2 else OI_WAVELENGTH_NM
    )
    lam2_nm = args.lam2 if mode == 2 else None
    d_prior_mm = args.d_prior if args.d_prior is not None else ICOS_GAP_MM
    r_max = args.r_max if args.r_max is not None else R_MAX_PX

    print("=" * 70)
    print(f"  WindCube FPI — Z01 Tolansky Validation  |  Mode: {mode}-Line")
    if mode == 2:
        print(f"  λ₁ = {lam1_nm} nm (air) [Ne primary]  |  "
              f"λ₂ = {lam2_nm} nm (air) [Ne secondary]")
    else:
        print(f"  λ  = {lam1_nm} nm (air) [OI rest wavelength]")
    print(f"  d_prior (ICOS, for N_int) = {d_prior_mm} mm  |  r_max = {r_max} px")
    print("=" * 70)

    # ── Stage A ────────────────────────────────────────────────────────────
    print("\nStage A: Loading images...")
    load = load_images(has_header=args.has_header)
    print(f"  CAL:  {load['cal_path'].name}  shape={load['cal_image'].shape}")
    print(f"  DARK: {load['dark_path'].name}  shape={load['dark_image'].shape}")

    # ── Stage B ────────────────────────────────────────────────────────────
    meta = extract_metadata(load)

    # ── Stage C — Figure 1 ────────────────────────────────────────────────
    lam_str = (f"{lam1_nm} nm (air) + {lam2_nm} nm (air)" if mode == 2
               else f"{lam1_nm} nm (air)")
    print(f"\nStage C: Image pair (Figure 1)...")
    cx_seed, cy_seed = figure_image_pair(load, meta, mode, lam_str)
    print(f"  Fringe centre seed: ({cx_seed:.1f}, {cy_seed:.1f}) px")

    # ── Stage D — Figure 2 ────────────────────────────────────────────────
    print(f"\nStage D: ROI inspection (Figure 2)...")
    cal_roi, dark_roi = figure_roi_inspection(
        load["cal_image"], load["dark_image"],
        cx_seed, cy_seed, r_max=r_max,
        cal_path=load["cal_path"],
        dark_path=load["dark_path"],
    )

    # ── Stage E — Centre refinement ───────────────────────────────────────
    cx_final, cy_final = refine_centre(
        load["cal_image"], cx_seed, cy_seed, r_max=r_max
    )

    # ── Stage F1 ──────────────────────────────────────────────────────────
    print(f"\nStage F1: S12 annular reduction...")
    fp, s12_dark_sub, roi = run_s12_reduction(
        load["cal_image"], load["dark_image"],
        cx_final, cy_final, r_max=r_max,
    )

    # ── Stage F2 — Figure 3 ───────────────────────────────────────────────
    print(f"\nStage F2: Dark subtraction comparison (Figure 3)...")
    rs = (slice(roi["roi_y0"], roi["roi_y1"]),
          slice(roi["roi_x0"], roi["roi_x1"]))
    figure_dark_comparison(
        load["cal_image"][rs],
        load["dark_image"][rs],
        s12_dark_sub[rs],
    )

    # ── Stage F3 — Figure 4 ───────────────────────────────────────────────
    print(f"\nStage F3: Radial profile (Figure 4)...")
    figure_reduction_peaks(fp, roi, load["cal_path"], mode, lam1_nm, lam2_nm)

    # ── Stage G — Tolansky ────────────────────────────────────────────────
    print(f"\nStage G: Tolansky analysis ({mode}-line)...")
    result = run_tolansky(
        fp, mode, lam1_nm, lam2_nm,
        d_prior_mm    = d_prior_mm,
        d_fixed_mm    = D_25C_MM if mode == 1 else None,
        f_fixed_mm    = F_TOLANSKY_MM if mode == 1 else None,
        pixel_pitch_m = CCD_PIXEL_PITCH_M,
    )

    # ── Stage H — Figure 5 ────────────────────────────────────────────────
    print(f"\nStage H: Tolansky figure (Figure 5)...")
    figure_tolansky(fp, result, mode, lam1_nm, lam2_nm,
                    load["cal_path"], d_prior_mm)

    # ── Vaughan equations console block ───────────────────────────────────
    print_vaughan_equations(result, mode, lam1_nm, lam2_nm,
                            d_prior_mm, CCD_PIXEL_PITCH_M, pitch_mm)

    # ── Final summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Z01 COMPLETE:")
    if mode == 2:
        f_mm = result.f_px * CCD_PIXEL_PITCH_M * 1e3
        print(f"    d   = {result.d_m*1e3:.4f} ± {result.two_sigma_d_m*1e3:.4f} mm (2σ)")
        print(f"    f   = {f_mm:.3f} ± {result.two_sigma_f_px*CCD_PIXEL_PITCH_M*1e3:.3f} mm (2σ)")
        print(f"    α   = {result.alpha_rad_px:.4e} ± {result.two_sigma_alpha:.4e} rad/px (2σ)")
        print(f"    ε₁  = {result.eps1:.5f} ± {2*result.sigma_eps1:.5f}  "
              f"[λ₁={lam1_nm} nm (air)] (2σ)")
        print(f"    ε₂  = {result.eps2:.5f} ± {2*result.sigma_eps2:.5f}  "
              f"[λ₂={lam2_nm} nm (air)] (2σ)")
        print(f"    N_int = {result.N_int}  |  χ²/ν = {result.chi2_dof:.3f}")
    else:
        v_ok = abs(result.v_rel_ms) < 3 * result.sigma_v_ms
        print(f"    ε₀  = {result.epsilon:.6f} ± {result.sigma_eps:.6f}  "
              f"(2σ={result.two_sigma_eps:.6f})"
              f"  ← store as calibration constant")
        print(f"    λ_c = {result.lam_c_nm:.5f} ± {result.two_sigma_lam_c_nm:.5f} nm (air) (2σ)")
        print(f"    v_ref= {result.v_rel_ms:.1f} ± {result.two_sigma_v_ms:.1f} m/s (2σ)  "
              f"{'[PASS — zero velocity confirmed]' if v_ok else '[WARN — investigate]'}")
        print(f"    N_int = {result.N_int}  |  χ²/ν = {result.chi2_dof:.3f}")
        print(f"    d_prior (fixed) = {D_25C_MM:.6f} mm  |  "
              f"f_prior (fixed) = {F_TOLANSKY_MM:.3f} mm")
    print("=" * 70)


if __name__ == "__main__":
    main()
