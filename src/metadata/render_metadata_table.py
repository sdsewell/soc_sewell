"""
Render the ImageMetadata field reference table as a PNG.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ---------------------------------------------------------------------------
# Table data  (index auto-assigned)
# ---------------------------------------------------------------------------
# fmt: off
FIELDS = [
    # (field_name, pixels, description, default, units)

    # Section 3.1
    ("rows",                    "0",        "Total rows including header row",                          "260",    "—"),
    ("cols",                    "1",        "Total columns",                                            "276",    "—"),
    ("exp_time",                "2",        "Exposure duration",                                        "—",      "centiseconds"),
    ("exp_unit",                "3",        "CCD timing register value",                                "—",      "—"),
    ("binning",                 "derived",  "Pixel binning mode (derived from cols)",                   "2",      "—"),
    ("img_type",                "derived",  "Image type: 'cal', 'dark', or 'science'",                  "—",      "—"),

    # Section 3.2
    ("lua_timestamp",           "8–11",     "End-of-exposure time (primary reference)",                 "—",      "Unix ms"),
    ("adcs_timestamp",          "12–15",    "ADCS clock time at acquisition; 0 if unavailable",         "0",      "Unix ms"),
    ("utc_timestamp",           "derived",  "ISO 8601 string derived from lua_timestamp",               "—",      "—"),

    # Section 3.3
    ("spacecraft_latitude",     "16–19",    "Spacecraft geodetic latitude",                             "—",      "rad"),
    ("spacecraft_longitude",    "20–23",    "Spacecraft geodetic longitude",                            "—",      "rad"),
    ("spacecraft_altitude",     "24–27",    "Spacecraft altitude above WGS84 ellipsoid",                "—",      "m"),
    ("pos_eci_hat",             "60–71",    "ECI position vector [x, y, z]",                            "—",      "m"),
    ("vel_eci_hat",             "72–83",    "ECI velocity vector [vx, vy, vz]",                         "—",      "m/s"),

    # Section 3.4
    ("attitude_quaternion",     "28–43",    "Body-to-ECI quaternion [x, y, z, w] scalar-last",          "—",      "—"),
    ("pointing_error",          "44–59",    "Residual attitude error quaternion [x, y, z, w]",           "—",      "—"),
    ("obs_mode",                "—",        "Observation mode: along_track / cross_track / unknown",    "unknown","—"),

    # Section 3.5
    ("ccd_temp1",               "4–7",      "CCD temperature",                                          "—",      "°C"),
    ("etalon_temps",            "84–99",    "Etalon temperature array [T0, T1, T2, T3]",                 "—",      "°C"),
    ("shutter_status",          "derived",  "Shutter state: 'open' or 'closed'",                        "open",   "—"),
    ("gpio_pwr_on",             "100–103",  "GPIO power register values [ch0–ch3]",                     "—",      "—"),
    ("lamp_ch_array",           "104–109",  "Lamp channel states [ch0–ch5]",                            "—",      "—"),
    ("lamp1_status",            "derived",  "Lamp 1 state: 'on' or 'off'",                              "off",    "—"),
    ("lamp2_status",            "derived",  "Lamp 2 state: 'on' or 'off'",                              "off",    "—"),
    ("lamp3_status",            "derived",  "Lamp 3 state: 'on' or 'off'",                              "off",    "—"),

    # Section 3.6
    ("orbit_number",            "—",        "Absolute orbit counter",                                   "None",   "—"),
    ("frame_sequence",          "—",        "Frame index within the orbit",                             "None",   "—"),
    ("orbit_parity",            "derived",  "Look direction derived from orbit_number parity",          "None",   "—"),

    # Section 3.7
    ("adcs_quality_flag",       "derived",  "ADCS quality bitmask (0 = good)",                          "0",      "—"),

    # Section 3.8
    ("dark_subtracted",         "—",        "Whether dark subtraction has been applied",                "False",  "—"),
    ("dark_n_frames",           "—",        "Number of dark frames averaged for subtraction",           "0",      "—"),
    ("dark_lua_timestamp",      "—",        "Timestamp of the dark frame used",                         "None",   "Unix ms"),
    ("dark_etalon_temp_mean",   "—",        "Mean etalon temperature of the dark frame",                "None",   "°C"),

    # Section 3.9
    ("is_synthetic",            "—",        "True if image was simulated, not from hardware",           "False",  "—"),
    ("truth_v_los",             "—",        "True line-of-sight wind speed (synthetic only)",           "None",   "m/s"),
    ("truth_v_zonal",           "—",        "True zonal wind component (synthetic only)",               "None",   "m/s"),
    ("truth_v_meridional",      "—",        "True meridional wind component (synthetic only)",          "None",   "m/s"),
    ("tangent_lat",             "—",        "Tangent point geodetic latitude (synthetic only)",         "None",   "deg"),
    ("tangent_lon",             "—",        "Tangent point geodetic longitude (synthetic only)",        "None",   "deg"),
    ("tangent_alt_km",          "—",        "Tangent point altitude (synthetic only)",                  "None",   "km"),
    ("etalon_gap_mm",           "—",        "Etalon gap used in synthesis",                             "None",   "mm"),
    ("noise_seed",              "—",        "RNG seed used for noise generation (synthetic only)",      "None",   "—"),

    # Section 3.10
    ("etalon_gap_corrected_mm", "—",        "Thermally corrected etalon gap (pipeline-added)",          "None",   "mm"),

    # Section 3.11
    ("grafana_record_id",       "—",        "Grafana telemetry record identifier",                      "None",   "—"),
]
# fmt: on

SECTION_BREAKS = {
    0:  "3.1  Image geometry & classification",
    6:  "3.2  Timing",
    9:  "3.3  Orbit state",
    15: "3.4  Attitude",
    18: "3.5  Instrument state",
    26: "3.6  Orbit & sequence identification",
    29: "3.7  Pointing quality gate",
    30: "3.8  Dark subtraction provenance",
    34: "3.9  Synthetic fields",
    43: "3.10 Etalon thermal correction hook",
    44: "3.11 Grafana telemetry hook",
}

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
HEADERS   = ("#", "Field", "Pixels", "Description", "Default", "Units")
COL_FRAC  = [0.03, 0.18, 0.08, 0.44, 0.12, 0.15]   # must sum to 1.0
ROW_H     = 0.016          # axes fraction per data row
HDR_H     = 0.022          # header row height
SEC_H     = 0.018          # section-label row height

BG_EVEN   = "#f7f9fc"
BG_ODD    = "#ffffff"
BG_HDR    = "#1a3a5c"
BG_SEC    = "#dce6f0"
FG_HDR    = "white"
FG_SEC    = "#1a3a5c"
FG_DATA   = "#1a1a1a"
MONO      = "DejaVu Sans Mono"
SANS      = "DejaVu Sans"

HERE     = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "metadata_table.png")

# ---------------------------------------------------------------------------
# Build row list: interleave section headers and data rows
# ---------------------------------------------------------------------------
rows = []   # each entry: ("header", label) | ("data", idx, field, pixels, desc, default, units)
for i, entry in enumerate(FIELDS):
    if i in SECTION_BREAKS:
        rows.append(("section", SECTION_BREAKS[i]))
    rows.append(("data", i + 1, *entry))

# ---------------------------------------------------------------------------
# Figure sizing
# ---------------------------------------------------------------------------
total_h_frac = sum(SEC_H if r[0] == "section" else ROW_H for r in rows) + HDR_H
fig_h = total_h_frac / 0.92   # leave a little margin top+bottom
fig_w = 16

fig, ax = plt.subplots(figsize=(fig_w, fig_h * fig_w * 0.72))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
fig.patch.set_facecolor("white")

# ---------------------------------------------------------------------------
# Column x-positions (left edges, plus right edge sentinel)
# ---------------------------------------------------------------------------
xs = [0.0]
for f in COL_FRAC:
    xs.append(xs[-1] + f)
PAD = 0.006   # text left-pad within cell

# ---------------------------------------------------------------------------
# Draw header
# ---------------------------------------------------------------------------
y = 1.0 - HDR_H / total_h_frac

for j, (label, x0, x1) in enumerate(zip(HEADERS, xs[:-1], xs[1:])):
    ax.add_patch(plt.Rectangle((x0, y), x1 - x0, HDR_H / total_h_frac,
                                transform=ax.transAxes,
                                color=BG_HDR, zorder=1, clip_on=False))
    ax.text(x0 + PAD, y + (HDR_H / total_h_frac) * 0.5, label,
            transform=ax.transAxes, ha="left", va="center",
            fontsize=8.5, fontweight="bold", color=FG_HDR,
            fontfamily=SANS, clip_on=False)

y -= HDR_H / total_h_frac

# ---------------------------------------------------------------------------
# Draw data and section rows
# ---------------------------------------------------------------------------
data_row = 0
for r in rows:
    if r[0] == "section":
        h = SEC_H / total_h_frac
        ax.add_patch(plt.Rectangle((0, y), 1, h,
                                   transform=ax.transAxes,
                                   color=BG_SEC, zorder=1, clip_on=False))
        ax.text(PAD * 1.5, y + h * 0.5, r[1],
                transform=ax.transAxes, ha="left", va="center",
                fontsize=7.8, fontweight="bold", color=FG_SEC,
                fontfamily=SANS, clip_on=False)
        y -= h

    else:
        _, idx, fname, pixels, desc, default, units = r
        h = ROW_H / total_h_frac
        bg = BG_EVEN if data_row % 2 == 0 else BG_ODD
        ax.add_patch(plt.Rectangle((0, y), 1, h,
                                   transform=ax.transAxes,
                                   color=bg, zorder=1, clip_on=False))

        cells = (str(idx), fname, pixels, desc, default, units)
        fonts = (SANS, MONO, MONO, SANS, MONO, SANS)
        for j, (txt, x0, x1, ff) in enumerate(zip(cells, xs[:-1], xs[1:], fonts)):
            ax.text(x0 + PAD, y + h * 0.5, txt,
                    transform=ax.transAxes, ha="left", va="center",
                    fontsize=7.5, color=FG_DATA, fontfamily=ff,
                    clip_on=False)

        # thin row separator
        ax.axhline(y, color="#cccccc", linewidth=0.3, zorder=2)
        y -= h
        data_row += 1

# outer border
ax.add_patch(plt.Rectangle((0, y), 1, 1 - y,
                            transform=ax.transAxes, fill=False,
                            edgecolor="#1a3a5c", linewidth=1.2,
                            zorder=3, clip_on=False))

fig.suptitle("ImageMetadata — field reference",
             fontsize=11, fontweight="bold", color=BG_HDR,
             y=0.995, fontfamily=SANS)

plt.savefig(OUT_PATH, dpi=180, bbox_inches="tight",
            facecolor="white", edgecolor="none")
print(f"Saved: {OUT_PATH}")
