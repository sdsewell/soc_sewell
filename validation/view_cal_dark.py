"""
Quick viewer: cal + dark binary images with embedded metadata tables.

Run from the validation/ subfolder:
    python view_cal_dark.py
"""

import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.table as mtable

# ── Make src importable from validation/ ────────────────────────────────────
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.metadata.p01_image_metadata_2026_04_06 import ingest_real_image  # noqa: E402

# ── File paths ────────────────────────────────────────────────────────────────
DATA_DIR = pathlib.Path(r"C:\Users\sewell\Documents\GitHub\soc_sample_data\2026_03_20")
FILES = [
    DATA_DIR / "1_cal_120sexp_swapped.bin",
    DATA_DIR / "1_dark_120sexp_swapped.bin",
]

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

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10),
                         gridspec_kw={"height_ratios": [3, 2]})
fig.subplots_adjust(hspace=0.35, wspace=0.15)

for col, (fp, meta, img) in enumerate(zip(FILES, metas, images)):
    # ── Top row: image ────────────────────────────────────────────────────────
    ax_img = axes[0][col]
    vmin, vmax = np.percentile(img, [1, 99])
    ax_img.imshow(img, origin="upper", cmap="gray", vmin=vmin, vmax=vmax,
                  aspect="auto", interpolation="nearest")
    ax_img.set_title(fp.name, fontsize=9, pad=6)
    ax_img.axis("off")

    # ── Bottom row: metadata table ────────────────────────────────────────────
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

# ── ROI extraction ────────────────────────────────────────────────────────────
CX, CY = 145, 145
ROI_SIZE = 220
half = ROI_SIZE // 2

r0, r1 = CY - half, CY + half
c0, c1 = CX - half, CX + half

cal_roi  = images[0][r0:r1, c0:c1].astype(np.int32)
dark_roi = images[1][r0:r1, c0:c1].astype(np.int32)
diff_roi = cal_roi - dark_roi

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
    ax_img.set_title(label, fontsize=10)
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
