"""
Quick viewer: cal + dark binary images with embedded metadata tables.
Performs two-line Tolansky analysis and outputs TolanskyResult fields
(alpha_rpx in rad/px, Y_B intensity ratio) to the annular profile CSV.

Run from the validation/ subfolder:
    python z01_analyze_cal_dark_pair_2026_04_22.py

Changes 2026-04-22:
  - alpha_rpx now correctly converted from Tolansky slope (orders/px²)
    to rad/px via alpha_rpx = sqrt(slope * lambda / (2*n*d)).
    The raw slope (orders/px²) is kept on the figure for traceability
    but the CSV header now carries alpha_rpx in the correct units.
  - Y_B intensity ratio added: computed as median(amp_638) / median(amp_640)
    from the Gaussian-fit amplitudes in the peak table (all peaks included;
    peak 1 amplitude is reliable with the global fixed-baseline fit).
    Written to the CSV header and printed to terminal.
  - Both alpha_rpx and Y_B are written to the CSV header so F01 step4b
    can read them directly without re-deriving from scratch.
"""

import sys
import re
import pathlib
import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.table as mtable
import matplotlib.ticker as mticker

# ── Make src importable from validation/ ────────────────────────────────────
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.metadata.p01_image_metadata_2026_04_06 import ingest_real_image  # noqa: E402
from src.fpi.m03_annular_reduction_2026_04_06 import (  # noqa: E402
    azimuthal_variance_centre,
    estimate_centre_uncertainty,
    _variance_cost,
    annular_reduce,
)

# ── File selection via dialog ─────────────────────────────────────────────────
_root = tk.Tk()
_root.withdraw()   # hide the empty root window

cal_path = filedialog.askopenfilename(
    title="Select calibration image",
    filetypes=[("Cal images", "*cal*.bin"), ("All files", "*.*")],
)
if not cal_path:
    sys.exit("No calibration file selected — exiting.")

dark_path = filedialog.askopenfilename(
    title="Select dark image",
    filetypes=[("Dark images", "*dark*.bin"), ("All files", "*.*")],
    initialdir=str(pathlib.Path(cal_path).parent),
)
if not dark_path:
    sys.exit("No dark file selected — exiting.")

FILES = [pathlib.Path(cal_path), pathlib.Path(dark_path)]

# ── Metadata fields to display (scalar only) ─────────────────────────────────
FIELDS = [
    ("img_type",             "Image type"),
    ("rows",                 "Rows"),
    ("cols",                 "Cols"),
    ("binning",              "Binning"),
    ("exp_time",             "Exp time (cs)"),
    ("utc_timestamp",        "UTC timestamp"),
    ("ccd_temp1",            "CCD temp 1 (°C)"),
    ("etalon_temps",         "Etalon temps (°C)"),
    ("shutter_status",       "Shutter"),
    ("lamp1_status",         "Lamp 1"),
    ("lamp2_status",         "Lamp 2"),
    ("lamp3_status",         "Lamp 3"),
    ("adcs_quality_flag",    "ADCS quality flag"),
    ("spacecraft_altitude",  "SC altitude (m)"),
]


def fmt_value(val):
    """Format a metadata value for table display."""
    if isinstance(val, float):
        return f"{val:.4g}"
    if isinstance(val, list):
        return "[" + ", ".join(f"{v:.3g}" if isinstance(v, float) else str(v) for v in val) + "]"
    return str(val)


# ── Load both images ──────────────────────────────────────────────────────────
metas, images = [], []
for fp in FILES:
    meta, img = ingest_real_image(fp)
    metas.append(meta)
    images.append(img)

# ── ROI / seed parameters (needed for first figure markers) ──────────────────
# Synthetic files follow YYYY-MM-DDTHH-MM-SSZ_cal.bin (hyphens in time portion)
_SYNTHETIC_PAT = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z_")
_is_synthetic  = bool(_SYNTHETIC_PAT.match(pathlib.Path(cal_path).name))
CX, CY   = (138, 129) if _is_synthetic else (145, 145)
ROI_SIZE = 220

# ── Plot figure 1 — loop until user accepts ROI size ─────────────────────────
_img_max_dim = min(images[0].shape)
while True:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10),
                             gridspec_kw={"height_ratios": [3, 2]})
    fig.subplots_adjust(hspace=0.35, wspace=0.15)

    for col, (fp, meta, img) in enumerate(zip(FILES, metas, images)):
        # ── Top row: image ────────────────────────────────────────────────────
        ax_img = axes[0][col]
        vmin, vmax = np.percentile(img, [1, 99])
        ax_img.imshow(img, origin="upper", cmap="gray", vmin=vmin, vmax=vmax,
                      aspect="auto", interpolation="nearest")
        ax_img.set_title(fp.name, fontsize=9, pad=6)
        ax_img.set_xlabel("Column (px)", fontsize=8)
        ax_img.set_ylabel("Row (px)", fontsize=8)
        ax_img.tick_params(labelsize=7)

        # Absolute image centre
        abs_cx = (img.shape[1] - 1) / 2.0
        abs_cy = (img.shape[0] - 1) / 2.0
        ax_img.plot(abs_cx, abs_cy, "+", color="cyan",
                    markersize=14, markeredgewidth=1.5,
                    label=f"Image centre  ({abs_cx:.1f}, {abs_cy:.1f})")

        # Seed centre (CX = col, CY = row)
        ax_img.plot(CX, CY, "x", color="yellow",
                    markersize=12, markeredgewidth=1.5,
                    label=f"Seed  ({CX}, {CY})")

        # ROI box — shown on cal image only
        if col == 0:
            half = ROI_SIZE // 2
            roi_rect = mpatches.Rectangle(
                (CX - half, CY - half), ROI_SIZE, ROI_SIZE,
                linewidth=1.2, edgecolor="yellow", facecolor="none",
                linestyle="--", label=f"ROI  {ROI_SIZE}×{ROI_SIZE} px",
            )
            ax_img.add_patch(roi_rect)

        ax_img.legend(fontsize=6.5, loc="lower right",
                      framealpha=0.7, edgecolor="0.5")

        # ── Bottom row: metadata table ────────────────────────────────────────
        ax_tbl = axes[1][col]
        ax_tbl.axis("off")

        meta_dict = meta.__dict__
        cell_text = [[label, fmt_value(meta_dict[key])] for key, label in FIELDS]

        tbl = ax_tbl.table(
            cellText=cell_text,
            colLabels=["Field", "Value"],
            loc="upper center",
            cellLoc="left",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7.5)
        tbl.auto_set_column_width([0, 1])

        # Style header row
        for j in range(2):
            tbl[0, j].set_facecolor("#4472C4")
            tbl[0, j].set_text_props(color="white", fontweight="bold")

        # Alternating row shading
        for i in range(1, len(cell_text) + 1):
            color = "#EEF2FF" if i % 2 == 0 else "white"
            for j in range(2):
                tbl[i, j].set_facecolor(color)

    plt.suptitle("Cal & Dark Image Viewer", fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()
    plt.close(fig)

    # Prompt to accept or change ROI size
    raw = input(
        f"\nROI size: {ROI_SIZE}×{ROI_SIZE} px.  "
        f"Press Enter to accept, or type a new square size: "
    ).strip()
    if raw == "":
        break
    try:
        new_size = int(raw)
        if new_size < 10 or new_size > _img_max_dim:
            print(f"  Must be between 10 and {_img_max_dim}. Try again.")
        else:
            ROI_SIZE = new_size
    except ValueError:
        print("  Please enter an integer (e.g. 220). Try again.")

# ── ROI extraction ────────────────────────────────────────────────────────────
half = ROI_SIZE // 2

r0, r1 = CY - half, CY + half
c0, c1 = CX - half, CX + half

cal_roi  = images[0][r0:r1, c0:c1].astype(np.int32)
dark_roi = images[1][r0:r1, c0:c1].astype(np.int32)
diff_roi = cal_roi - dark_roi

# ── Stage 2: two-pass azimuthal variance minimisation ────────────────────────
cx_seed = float(half)   # midpoint of the ROI array (col)
cy_seed = float(half)   # midpoint of the ROI array (row)

p99_5        = float(np.percentile(cal_roi, 99.5))
cal_roi_clip = np.clip(cal_roi, None, p99_5)

VAR_R_MIN_PX = 5.0
VAR_N_BINS   = 250
r_min_sq = VAR_R_MIN_PX ** 2
r_max_sq = (min(cal_roi_clip.shape) / 2.0) ** 2  # default from azimuthal_variance_centre

cx_fine, cy_fine, cost_min, grid_cx, grid_cy, grid_cost = azimuthal_variance_centre(
    cal_roi_clip, cx_seed, cy_seed,
    var_r_min_px=VAR_R_MIN_PX,
    var_n_bins=VAR_N_BINS,
)

cost_fn = lambda cx, cy: _variance_cost(  # noqa: E731
    cx, cy, cal_roi_clip, r_min_sq, r_max_sq, VAR_N_BINS
)
sigma_cx, sigma_cy = estimate_centre_uncertainty(cx_fine, cy_fine, cost_fn)

# Match precision of displayed value to uncertainty (1 decimal place)
def _fmt_with_unc(val, sigma):
    """Format val ± sigma with precision matched to sigma."""
    if sigma >= 1.0:
        return f"{val:.0f} ± {sigma:.0f}"
    decimals = max(0, -int(np.floor(np.log10(sigma))))
    return f"{val:.{decimals}f} ± {sigma:.{decimals}f}"

print(f"Seed:        cx={cx_seed:.1f}, cy={cy_seed:.1f}")
print(f"Fine centre: cx={_fmt_with_unc(cx_fine, sigma_cx)},  cy={_fmt_with_unc(cy_fine, sigma_cy)}  (cost={cost_min:.4f})")

rois   = [cal_roi, dark_roi, diff_roi]
labels = ["Cal ROI", "Dark ROI", "Cal − Dark ROI"]

# ── ROI figure: 3 cols × 2 rows ───────────────────────────────────────────────
fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8))
fig2.suptitle(
    f"ROI  |  center = ({CX}, {CY})  |  size = {ROI_SIZE} × {ROI_SIZE} px",
    fontsize=12, fontweight="bold",
)

DN_MAX = 2**14 - 1  # 16383

for col, (roi, label) in enumerate(zip(rois, labels)):
    # Top row — image
    ax_img = axes2[0, col]
    if col == 1:  # dark ROI: autoscale
        vmin, vmax = np.percentile(roi, [1, 99])
    else:         # cal ROI and cal-dark ROI: fixed scale
        vmin, vmax = 0, DN_MAX
    im = ax_img.imshow(roi, origin="upper", cmap="gray",
                       vmin=vmin, vmax=vmax,
                       aspect="auto", interpolation="nearest")
    if col == 2:
        title = (f"{label}\n"
                 f"cx = {_fmt_with_unc(cx_fine, sigma_cx)},  "
                 f"cy = {_fmt_with_unc(cy_fine, sigma_cy)}")
        ax_img.plot(cx_fine, cy_fine, "+", color="red",
                    markersize=14, markeredgewidth=1.5)
    else:
        title = label
    ax_img.set_title(title, fontsize=10)
    ax_img.axis("off")
    fig2.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04)

    # Bottom row — histogram
    ax_hist = axes2[1, col]
    flat = roi.ravel()
    ax_hist.hist(flat, bins=128, color="#4472C4", edgecolor="none", alpha=0.85,
                 range=(0, DN_MAX))
    ax_hist.set_xlim(0, DN_MAX)
    ax_hist.set_xlabel("DN", fontsize=9)
    ax_hist.set_ylabel("Count", fontsize=9)
    ax_hist.set_title(f"{label} histogram", fontsize=9)
    ax_hist.tick_params(labelsize=8)

    # Annotate mean ± std
    mu, sigma = flat.mean(), flat.std()
    ax_hist.axvline(mu, color="red", linewidth=1.2, linestyle="--", label=f"μ={mu:.1f}")
    ax_hist.legend(fontsize=7.5, title=f"σ={sigma:.1f}", title_fontsize=7.5)

plt.tight_layout()
plt.show()

# ── Annular reduction ─────────────────────────────────────────────────────────
R_MAX_PX = 110.0
N_BINS   = 150

fp = annular_reduce(
    diff_roi.astype(float),
    cx=cx_fine,
    cy=cy_fine,
    sigma_cx=sigma_cx,
    sigma_cy=sigma_cy,
    r_max_px=R_MAX_PX,
    n_bins=N_BINS,
)

print(f"\nAnnular reduce: {len(fp.peak_fits)} peaks found")
for i, pk in enumerate(fp.peak_fits):
    two_s = 2.0 * pk.sigma_r_fit_px if np.isfinite(pk.sigma_r_fit_px) else float("nan")
    print(f"  Peak {i+1:2d}: r_fit={pk.r_fit_px:.3f} ± {two_s:.3f} px (2σ)  "
          f"amp={pk.amplitude_adu:.1f}  width={pk.width_px:.3f}  ok={pk.fit_ok}")

# ── Radial profile figure ─────────────────────────────────────────────────────
good = ~fp.masked & np.isfinite(fp.profile) & np.isfinite(fp.sigma_profile)

n_peaks = len(fp.peak_fits)
tbl_height = max(1.5, n_peaks * 0.28)
fig3 = plt.figure(figsize=(13, 6 + tbl_height))
gs   = fig3.add_gridspec(2, 1, height_ratios=[6, tbl_height], hspace=0.45)
ax_prof = fig3.add_subplot(gs[0])
ax_tbl  = fig3.add_subplot(gs[1])

# ── Profile: data points + connecting line ────────────────────────────────────
r_good = fp.r_grid[good]
p_good = fp.profile[good]

ax_prof.plot(r_good, p_good, "-", color="#4472C4", linewidth=0.8, alpha=0.5, zorder=1)
ax_prof.scatter(r_good, p_good, s=6, color="#4472C4", zorder=2, label="Binned profile")

# ── Peak markers, error bars, and text boxes ──────────────────────────────────
# Compute y-range after drawing data so offsets are meaningful
y_all  = p_good
y_span = y_all.max() - y_all.min() if len(y_all) else 1.0

for i, pk in enumerate(fp.peak_fits):
    r_x    = float(pk.r_fit_px)
    y_mark = float(fp.profile[pk.peak_idx])
    two_s  = (2.0 * pk.sigma_r_fit_px
              if (pk.fit_ok and np.isfinite(pk.sigma_r_fit_px)) else 0.0)

    ax_prof.errorbar(r_x, y_mark, xerr=two_s,
                     fmt="none", ecolor="red", elinewidth=1.2, capsize=3, zorder=4)
    ax_prof.plot(r_x, y_mark, "v", color="red", markersize=6, zorder=5)

    y_off = y_span * (0.12 if i % 2 == 0 else 0.22)
    txt   = (f"r={r_x:.2f}\n±{two_s:.2f} px" if two_s > 0
             else f"r={r_x:.2f}\n(no fit)")
    ax_prof.annotate(
        txt, xy=(r_x, y_mark),
        xytext=(r_x, y_mark + y_off),
        fontsize=5.5, ha="center", va="bottom",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="red", alpha=0.75, lw=0.6),
        arrowprops=dict(arrowstyle="-", color="red", lw=0.5),
    )

ax_prof.set_xlabel("Radius [px]", fontsize=10)
ax_prof.set_ylabel("Mean intensity (ADU)", fontsize=10)
ax_prof.set_title(
    f"Cal−Dark annular profile  |  cx={cx_fine:.2f}, cy={cy_fine:.2f}  |  "
    f"r_max={R_MAX_PX:.0f} px,  n_bins={N_BINS},  peaks={n_peaks}",
    fontsize=9,
)
ax_prof.set_ylim(0, 16383)
ax_prof.legend(fontsize=8)

# ── Peak table ────────────────────────────────────────────────────────────────
ax_tbl.axis("off")

# Ne line wavelengths: odd peak numbers → 640.2248 nm, even → 638.2991 nm
NE_LINES = {1: 640.2248, 0: 638.2991}   # keyed by (peak_number - 1) % 2

col_labels = [
    "#", "Line [nm]", "r_fit [px]", "2σ_r [px]",
    "r²_fit [px²]", "Δ_640 [px²]", "Δ_638 [px²]",
    "2σ_r² [px²]", "Amp [ADU]", "Baseline [ADU]", "Width [px]",
]

# Pre-compute r²_fit for every peak so delta look-ahead is simple
r2_all = [pk.r_fit_px ** 2 for pk in fp.peak_fits]

cell_text = []
d640_vals: list = []
d638_vals: list = []
for i, pk in enumerate(fp.peak_fits):
    peak_num   = i + 1
    wavelength = NE_LINES[peak_num % 2]   # odd → 640.2248, even → 638.2991

    two_sr = (2.0 * pk.sigma_r_fit_px
              if (pk.fit_ok and np.isfinite(pk.sigma_r_fit_px)) else float("nan"))
    r2_fit  = r2_all[i]
    # σ_{r²} = 2·r·σ_r  →  2σ_{r²} = 4·r·σ_r
    two_sr2 = (4.0 * pk.r_fit_px * pk.sigma_r_fit_px
               if (pk.fit_ok and np.isfinite(pk.sigma_r_fit_px)) else float("nan"))

    # Δ_640: difference to next 640 nm peak (i+2) for odd peaks only
    if peak_num % 2 == 1 and (i + 2) < len(r2_all):
        d640_val = r2_all[i + 2] - r2_fit
        d640_str = f"{d640_val:.2f}"
    else:
        d640_val = float("nan")
        d640_str = "—"

    # Δ_638: difference to next 638 nm peak (i+2) for even peaks only
    if peak_num % 2 == 0 and (i + 2) < len(r2_all):
        d638_val = r2_all[i + 2] - r2_fit
        d638_str = f"{d638_val:.2f}"
    else:
        d638_val = float("nan")
        d638_str = "—"

    two_sr_str   = f"{two_sr:.3f}"  if np.isfinite(two_sr)  else "—"
    two_sr2_str  = f"{two_sr2:.2f}" if np.isfinite(two_sr2) else "—"
    width_str    = f"{pk.width_px:.3f}"    if np.isfinite(pk.width_px)    else "—"
    base_str     = f"{pk.baseline_adu:.1f}" if np.isfinite(pk.baseline_adu) else "—"

    d640_vals.append(d640_val)
    d638_vals.append(d638_val)
    cell_text.append([
        str(peak_num),
        f"{wavelength:.4f}",
        f"{pk.r_fit_px:.3f}",
        two_sr_str,
        f"{r2_fit:.2f}",
        d640_str,
        d638_str,
        two_sr2_str,
        f"{pk.amplitude_adu:.1f}",
        base_str,
        width_str,
    ])

# ── Mean row ─────────────────────────────────────────────────────────────────
mean_640 = np.nanmean(d640_vals) if d640_vals else float("nan")
mean_638 = np.nanmean(d638_vals) if d638_vals else float("nan")
mean_row = [
    "Mean", "", "", "",
    "",
    f"{mean_640:.2f}" if np.isfinite(mean_640) else "—",
    f"{mean_638:.2f}" if np.isfinite(mean_638) else "—",
    "", "", "", "",
]
cell_text.append(mean_row)

# ── Y_B intensity ratio from Gaussian-fit amplitudes ─────────────────────────
amp_640_list = []   # amplitudes of 640 nm peaks (odd peak numbers)
amp_638_list = []   # amplitudes of 638 nm peaks (even peak numbers)
for i, pk in enumerate(fp.peak_fits):
    peak_num = i + 1
    if not pk.fit_ok or not np.isfinite(pk.amplitude_adu):
        continue
    if peak_num % 2 == 1:   # odd → 640 nm
        amp_640_list.append(pk.amplitude_adu)
    else:                    # even → 638 nm
        amp_638_list.append(pk.amplitude_adu)

if amp_640_list and amp_638_list:
    median_amp_640 = float(np.median(amp_640_list))
    median_amp_638 = float(np.median(amp_638_list))
    Y_B_estimate   = median_amp_638 / median_amp_640
else:
    median_amp_640 = float("nan")
    median_amp_638 = float("nan")
    Y_B_estimate   = float("nan")

print(f"\nIntensity ratio (Y_B = 638nm / 640nm amplitudes):")
print(f"  median amp 640nm = {median_amp_640:.1f} ADU  (n={len(amp_640_list)})")
print(f"  median amp 638nm = {median_amp_638:.1f} ADU  (n={len(amp_638_list)})")
print(f"  Y_B = {Y_B_estimate:.4f}")

if cell_text:
    tbl = ax_tbl.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="upper center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.auto_set_column_width(list(range(len(col_labels))))

    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#4472C4")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    mean_row_idx = len(cell_text)   # 1-based table index of the mean row
    for i in range(1, len(cell_text) + 1):
        if i == mean_row_idx:
            bg = "#D0E8FF"   # distinct colour for the mean row
            for j in range(len(col_labels)):
                tbl[i, j].set_facecolor(bg)
                tbl[i, j].set_text_props(fontweight="bold")
        else:
            bg = "#EEF2FF" if i % 2 == 0 else "white"
            for j in range(len(col_labels)):
                tbl[i, j].set_facecolor(bg)
else:
    ax_tbl.text(0.5, 0.5, "No peaks found", ha="center", va="center",
                transform=ax_tbl.transAxes, fontsize=11)

plt.show()

# ── Ring Order P vs r²_fit figure ────────────────────────────────────────────
# Separate the interleaved peaks into the two Ne line sets
r2_640, r2_638 = [], []
for i, pk in enumerate(fp.peak_fits):
    if (i + 1) % 2 == 1:          # odd peak number → 640.2248 nm
        r2_640.append(pk.r_fit_px ** 2)
    else:                           # even peak number → 638.2991 nm
        r2_638.append(pk.r_fit_px ** 2)

p_640 = np.arange(1, len(r2_640) + 1, dtype=float)
p_638 = np.arange(1, len(r2_638) + 1, dtype=float)
r2_640 = np.array(r2_640)
r2_638 = np.array(r2_638)

# Fit P = a·r² + b  →  intercept b = fractional fringe order ε ∈ [0, 1]
def _linfit(x, y):
    """Return (slope, intercept, 2σ_slope, 2σ_intercept) via np.polyfit cov."""
    coeffs, cov = np.polyfit(x, y, 1, cov=True)
    two_sig = 2.0 * np.sqrt(np.diag(cov))
    return coeffs[0], coeffs[1], two_sig[0], two_sig[1]

sl_640, ep_640, sl_640_2s, ep_640_2s = _linfit(r2_640, p_640)
sl_638, ep_638, sl_638_2s, ep_638_2s = _linfit(r2_638, p_638)

print(f"\n640 nm fit:  ε_640 = {ep_640:.4f} ± {ep_640_2s:.4f}  (2σ)")
print(f"638 nm fit:  ε_638 = {ep_638:.4f} ± {ep_638_2s:.4f}  (2σ)")

# ── Tolansky etalon gap calculation ─────────────────────────────────────────
LA_NM  = 640.2248    # λₐ
LB_NM  = 638.2991    # λᵦ


T_MM = 20.008   # ICOS as-built gap (mm) — used ONLY to compute N_delta integer.
                # Do NOT replace with the Tolansky-recovered gap D_25C_MM here:
                # D_25C_MM = 20.0006 mm sits only 0.3 um from the N_delta
                # rounding boundary and will flip N_delta from -189 to -188.
                # The ICOS nominal value sits 7.1 um inside the correct window.
                # Physical consistency: N_delta=-189 → d=20.0006 mm (78 nm
                # pre-load compression of 20.008 mm). N_delta=-190 would imply
                # a 99 um expansion — physically impossible. Locked 2026-04-22.

N_delta = round(2.0 * T_MM * 1e6 * (1.0 / LA_NM - 1.0 / LB_NM))
C_nm    = LA_NM * LB_NM / (2.0 * (LB_NM - LA_NM))   # negative (λb < λa)
d_nm    = (N_delta + ep_640 - ep_638) * C_nm
d_mm    = d_nm * 1e-6
two_sigma_d_nm = abs(C_nm) * np.sqrt(ep_640_2s**2 + ep_638_2s**2)
two_sigma_d_mm = two_sigma_d_nm * 1e-6

print(f"N_Δ  = {N_delta}")
print(f"d    = {d_mm:.4f} ± {two_sigma_d_mm:.4f} mm  (2σ)")

fig4, ax4 = plt.subplots(figsize=(9, 6))

# Data points — x = r²_fit, y = P
ax4.scatter(r2_640, p_640, color="#E84040", marker="o", s=50, zorder=3,
            label="640.2248 nm")
ax4.scatter(r2_638, p_638, color="#4472C4", marker="s", s=50, zorder=3,
            label="638.2991 nm")

# Fit lines extended to r²=0 so y-intercept (ε) is visible
r2_ext = np.array([0.0, max(r2_640.max(), r2_638.max()) * 1.04])
ax4.plot(r2_ext, sl_640 * r2_ext + ep_640, color="#E84040",
         linewidth=1.2, linestyle="--", zorder=2)
ax4.plot(r2_ext, sl_638 * r2_ext + ep_638, color="#4472C4",
         linewidth=1.2, linestyle="--", zorder=2)

ax4.set_xlabel("r²_fit  [px²]", fontsize=11)
ax4.set_ylabel("Ring Order  P", fontsize=11)
ax4.set_title(
    "Determination of Fractonal Fringe Orders and Estimate of Etalon Gap 'd'",
    fontsize=11, pad=22,
)
ax4.text(0.5, 1.002,
         f"cal: {FILES[0].name}   |   dark: {FILES[1].name}",
         transform=ax4.transAxes, fontsize=8.5,
         ha="center", va="bottom", color="#555555",
         clip_on=False)
ax4.legend(fontsize=9, loc="lower right", bbox_to_anchor=(1.0, 0.16))
ax4.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

# ── Lower-left: calibration image metadata ───────────────────────────────────
cal_meta   = metas[0]
exp_s      = cal_meta.exp_time / 100.0        # centiseconds → seconds
et         = cal_meta.etalon_temps            # list of 4 floats
et_str     = "[" + ", ".join(f"{v:.2f}" for v in et) + "]"

meta_annot = (
    f"Exposure: {int(round(exp_s))} s   |   "
    f"CCD Temp: {cal_meta.ccd_temp1:.2f} °C   |   "
    f"Etalon Temps: {et_str} °C"
)
ax4.text(0.5, 0.97, meta_annot,
         transform=ax4.transAxes, fontsize=8.5,
         va="top", ha="center", family="monospace",
         bbox=dict(boxstyle="round,pad=0.45", fc="#F5FFF5", ec="#3A7D44",
                   alpha=0.95, lw=0.8))

# ── Lower-right: ε values only ───────────────────────────────────────────────
eps_annot = (
    f"ε_640 = {ep_640:.4f} ± {ep_640_2s:.4f}\n"
    f"ε_638 = {ep_638:.4f} ± {ep_638_2s:.4f}"
)
ax4.text(0.97, 0.05, eps_annot,
         transform=ax4.transAxes, fontsize=9,
         va="bottom", ha="right",
         bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#888888",
                   alpha=0.92, lw=0.8),
         family="monospace")

# ── Helper: two-part box (bold heading + monospace body) ─────────────────────
# matplotlib can't mix font weights in one text() call, so we use two calls:
# the heading draws the box (with padding below), the body overlaps it below.

BOX_FS   = 8.0   # body font size (pt)
LINE_H   = 0.038  # estimated axes-fraction height per line at BOX_FS in this figure

def _two_part_box(ax, x, y_top, heading, body, n_body_lines,
                  fc, ec, heading_fc=None):
    """
    Draw a bold heading text box immediately above a monospace body text box.
    Returns the estimated y coordinate of the bottom of the body box.
    """
    if heading_fc is None:
        heading_fc = fc
    # Count heading lines to estimate its height
    n_head_lines = heading.count("\n") + 1
    head_h = n_head_lines * LINE_H + 0.017  # add bbox padding

    # Bold heading
    ax.text(x, y_top, heading,
            transform=ax.transAxes, fontsize=BOX_FS, fontweight="bold",
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.35", fc=heading_fc, ec=ec,
                      alpha=0.97, lw=1.0))

    # Monospace body positioned directly below
    y_body = y_top - head_h
    ax.text(x, y_body, body,
            transform=ax.transAxes, fontsize=BOX_FS,
            va="top", ha="left", family="monospace",
            bbox=dict(boxstyle="round,pad=0.35", fc=fc, ec=ec,
                      alpha=0.95, lw=0.8))

    body_h = n_body_lines * LINE_H + 0.025
    return y_body - body_h   # bottom of body box

# ── Yellow box (N_Δ) — placed well below the legend ─────────────────────────
nd_heading = (
    "Integer Difference Between Orders\n"
    "at the Optical Axis for Two\n"
    "Neon Calibration Lines"
)
nd_body = (
    f"N_Δ = round(2 · {T_MM} mm · (1/λa − 1/λb))\n"
    f"    = round(2 · {T_MM}×10⁶ nm ·\n"
    f"      (1/{LA_NM} − 1/{LB_NM}) nm⁻¹)\n"
    f"    = {N_delta}"
)
y_after_yellow = _two_part_box(ax4, 0.03, 0.90,
                               nd_heading, nd_body, n_body_lines=4,
                               fc="#FFF8E7", ec="#C8A000",
                               heading_fc="#FFE680")

# ── Blue box (d recovery) — placed below yellow box ──────────────────────────
d_heading = "Benoit 'd' recovery (Vaughan Eq. 3.97)"
d_body = (
    f"d = (N_Δ + ε_640nm − ε_638nm) · λa·λb / [2(λb − λa)]\n"
    f"  = ({N_delta} + {ep_640:.4f} − {ep_638:.4f})\n"
    f"    · {LA_NM}·{LB_NM} / [2·({LB_NM}−{LA_NM})] nm\n"
    f"  = {d_mm:.4f} ± {two_sigma_d_mm:.4f} mm  (Benoit Gap, 2σ)"
)
y_after_blue = _two_part_box(ax4, 0.03, y_after_yellow - 0.02,
                             d_heading, d_body, n_body_lines=4,
                             fc="#F0F8FF", ec="#4472C4",
                             heading_fc="#C8DEFF")

# ── Green box (plate scale α) — placed below blue box ────────────────────────
alpha_mean   = (sl_640 + sl_638) / 2.0
alpha_2s     = np.sqrt(sl_640_2s ** 2 + sl_638_2s ** 2) / 2.0

# Convert Tolansky slope (orders/px²) to plate scale alpha in rad/px.
# slope (orders/px²) = dP/d(r²) = alpha² * 2*n*d / lambda
# => alpha (rad/px) = sqrt(slope * lambda / (2*n*d))
# This is the value F01 needs. The raw slope is kept on the figure for
# traceability but must NOT be passed directly to F01 as alpha_rpx.
_n_ref    = 1.0
_d_m      = d_mm * 1e-3
_lam_a_m  = LA_NM * 1e-9
alpha_rpx_640 = float(np.sqrt(sl_640 * _lam_a_m / (2.0 * _n_ref * _d_m)))
alpha_rpx_638 = float(np.sqrt(sl_638 * LB_NM * 1e-9 / (2.0 * _n_ref * _d_m)))
alpha_rpx     = (alpha_rpx_640 + alpha_rpx_638) / 2.0
# Propagate uncertainty: σ(alpha) = alpha/(2*slope) * σ(slope)
alpha_rpx_2s  = float(0.5 * alpha_rpx / alpha_mean * alpha_2s * 2.0)

print(f"α_640  = {sl_640:.6f} ± {sl_640_2s:.6f}  orders/px²  (2σ)")
print(f"α_638  = {sl_638:.6f} ± {sl_638_2s:.6f}  orders/px²  (2σ)")
print(f"α_mean = {alpha_mean:.6f} ± {alpha_2s:.6f}  orders/px²  (2σ)")
print(f"α_rpx  = {alpha_rpx:.5e} ± {alpha_rpx_2s:.2e}  rad/px  (converted, 2σ)")

alpha_heading = "Plate Scale  α  (P vs r² slope)"
alpha_body = (
    f"α_640  = {sl_640:.6f} ± {sl_640_2s:.6f}  orders/px²\n"
    f"α_638  = {sl_638:.6f} ± {sl_638_2s:.6f}  orders/px²\n"
    f"α_mean = {alpha_mean:.6f} ± {alpha_2s:.6f}  orders/px²\n"
    f"\n"
    f"α_rpx  = √(slope·λ / 2nd)\n"
    f"       = {alpha_rpx:.5e} ± {alpha_rpx_2s:.2e}  rad/px\n"
    f"         ← use this value in F01 step4b"
)
y_after_green = _two_part_box(ax4, 0.03, y_after_blue - 0.02,
              alpha_heading, alpha_body, n_body_lines=6,
              fc="#F5FFF5", ec="#3A7D44",
              heading_fc="#B8FFB8")

# ── Orange box (Y_B intensity ratio) — placed below green box ────────────────
yb_heading = "Intensity Ratio  Y_B = 638nm / 640nm"
if np.isfinite(Y_B_estimate):
    yb_body = (
        f"median amp 640nm = {median_amp_640:.1f} ADU  (peak 1 excluded)\n"
        f"median amp 638nm = {median_amp_638:.1f} ADU\n"
        f"Y_B = {Y_B_estimate:.4f}  ← use as Y_B initial guess in F01"
    )
    n_yb_lines = 3
else:
    yb_body = "Insufficient peaks to estimate Y_B"
    n_yb_lines = 1
_two_part_box(ax4, 0.03, y_after_green - 0.02,
              yb_heading, yb_body, n_body_lines=n_yb_lines,
              fc="#FFF3E0", ec="#E65100",
              heading_fc="#FFCC80")

plt.tight_layout()
plt.show()

# ── Save annular profile CSV ──────────────────────────────────────────────────
import pandas as pd

# alpha_rpx was computed above in rad/px via sqrt(slope * lambda / (2*n*d)).
# Y_B_estimate was computed above from peak amplitude ratios.
# Both are written to the CSV header so F01 step4b can read them directly.

csv_path = pathlib.Path(cal_path).with_suffix("").parent / (
    pathlib.Path(cal_path).stem + "_annular_profile.csv"
)
_df = pd.DataFrame({
    "r_grid":        fp.r_grid,
    "profile":       fp.profile,
    "sigma_profile": fp.sigma_profile,
})
with open(csv_path, "w", newline="") as _fh:
    _fh.write(f"# alpha_rad_px: {alpha_rpx:.6e}\n")
    _fh.write(f"# alpha_rad_px_2sigma: {alpha_rpx_2s:.2e}\n")
    _fh.write(f"# d_mm: {d_mm:.6f}\n")
    _fh.write(f"# d_mm_2sigma: {two_sigma_d_mm:.6f}\n")
    _fh.write(f"# epsilon_640: {ep_640:.6f}\n")
    _fh.write(f"# epsilon_638: {ep_638:.6f}\n")
    _fh.write(f"# Y_B: {Y_B_estimate:.6f}\n")
    _fh.write(f"# Y_B_note: median(amp_638)/median(amp_640), all peaks included\n")
    _fh.write(f"# slope_orders_per_px2: {alpha_mean:.6f}\n")
    _fh.write(f"# cx: {cx_fine:.4f}\n")
    _fh.write(f"# cy: {cy_fine:.4f}\n")
    _fh.write(f"# r_max_px: {R_MAX_PX:.1f}\n")
    _df.to_csv(_fh, index=False)

print(f"\nAnnular profile saved → {csv_path}")
print(f"  α_rpx  (rad/px)     : {alpha_rpx:.6e}  ← correct value for F01")
print(f"  slope  (orders/px²) : {alpha_mean:.6f}  ← Tolansky raw slope (do NOT pass to F01 as alpha)")
print(f"  Y_B                 : {Y_B_estimate:.4f}  ← initial guess for F01 Y_B parameter")
print(f"  d                   : {d_mm:.6f} mm")
