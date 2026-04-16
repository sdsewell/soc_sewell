"""
extract_metadata.py
-------------------
Prompt the user to select a WindCube FPI .bin image, parse the embedded
header fields, and render a formatted PNG table saved alongside the binary.

Only fields that are directly encoded in the binary header row are shown.
Derived quantities (binning, img_type, utc_timestamp, lamp/shutter status,
quality flags, pipeline fields) are excluded.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# --- file dialog (must happen before matplotlib imports on some backends) ---
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()
bin_path = filedialog.askopenfilename(
    title="Select a WindCube FPI .bin image",
    filetypes=[("Binary image", "*.bin"), ("All files", "*.*")],
)
root.destroy()

if not bin_path:
    print("No file selected — exiting.")
    sys.exit(0)

# --- parse header --------------------------------------------------------
import numpy as np
import struct

def _u16(h, w):
    return int(h[w])

def _u64(h, w):
    return sum(int(h[w + i]) << (16 * i) for i in range(4))

def _f64(h, w):
    b = struct.pack(">4H", *reversed([h[w + i] for i in range(4)]))
    return struct.unpack(">d", b)[0]

def _u8arr(h, w, n):
    return [int(h[w + i]) & 0xFF for i in range(n)]

raw = np.frombuffer(open(bin_path, "rb").read(), dtype=">u2")
h = raw[:276]

rows           = _u16(h, 0)
cols           = _u16(h, 1)
exp_time       = _u16(h, 2)
exp_unit       = _u16(h, 3)
ccd_temp1      = _f64(h, 4)
lua_timestamp  = _u64(h, 8)
adcs_timestamp = _u64(h, 12)
sc_lat         = _f64(h, 16)
sc_lon         = _f64(h, 20)
sc_alt         = _f64(h, 24)

q_wxyz = [_f64(h, 28 + i * 4) for i in range(4)]
att_q  = [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]   # → [x,y,z,w]

pe_wxyz = [_f64(h, 44 + i * 4) for i in range(4)]
pt_err  = [pe_wxyz[1], pe_wxyz[2], pe_wxyz[3], pe_wxyz[0]]

pos_eci  = [_f64(h, 60), _f64(h, 64), _f64(h, 68)]
vel_eci  = [_f64(h, 72), _f64(h, 76), _f64(h, 80)]
et_temps = [_f64(h, 84), _f64(h, 88), _f64(h, 92), _f64(h, 96)]
gpio     = _u8arr(h, 100, 4)
lamp_ch  = _u8arr(h, 104, 6)

# --- value formatting helpers -------------------------------------------
def _fmt_f(v, dp=4):
    return f"{v:.{dp}f}"

def _fmt_vec(lst, dp=4):
    inner = ",  ".join(f"{v:.{dp}f}" for v in lst)
    return f"[{inner}]"

def _fmt_int_list(lst):
    return "[" + ", ".join(str(v) for v in lst) + "]"

# --- table rows: (field, pixels, description, value, units) -------------
TABLE = [
    ("rows",               "0",        "Total image rows (incl. header row)",         str(rows),                    "—"),
    ("cols",               "1",        "Total image columns",                          str(cols),                    "—"),
    ("exp_time",           "2",        "Exposure duration",                            str(exp_time),                "centiseconds"),
    ("exp_unit",           "3",        "CCD timing register value",                    str(exp_unit),                "—"),
    ("ccd_temp1",          "4–7",      "CCD temperature",                              _fmt_f(ccd_temp1, 3),         "°C"),
    ("lua_timestamp",      "8–11",     "End-of-exposure time",                         str(lua_timestamp),           "Unix ms"),
    ("adcs_timestamp",     "12–15",    "ADCS clock time at acquisition",               str(adcs_timestamp),          "Unix ms"),
    ("sc_latitude",        "16–19",    "Spacecraft geodetic latitude",                 _fmt_f(sc_lat, 6),            "rad"),
    ("sc_longitude",       "20–23",    "Spacecraft geodetic longitude",                _fmt_f(sc_lon, 6),            "rad"),
    ("sc_altitude",        "24–27",    "Spacecraft altitude above WGS84",              _fmt_f(sc_alt, 1),            "m"),
    ("attitude_quaternion","28–43",    "Body-to-ECI quaternion [x, y, z, w]",          _fmt_vec(att_q, 6),           "—"),
    ("pointing_error",     "44–59",    "Attitude error quaternion [x, y, z, w]",       _fmt_vec(pt_err, 6),          "—"),
    ("pos_eci_hat",        "60–71",    "ECI position vector [x, y, z]",                _fmt_vec(pos_eci, 1),         "m"),
    ("vel_eci_hat",        "72–83",    "ECI velocity vector [vx, vy, vz]",             _fmt_vec(vel_eci, 3),         "m/s"),
    ("etalon_temps",       "84–99",    "Etalon temperatures [T0, T1, T2, T3]",         _fmt_vec(et_temps, 3),        "°C"),
    ("gpio_pwr_on",        "100–103",  "GPIO power register values [ch0–ch3]",         _fmt_int_list(gpio),          "—"),
    ("lamp_ch_array",      "104–109",  "Lamp channel states [ch0–ch5]",                _fmt_int_list(lamp_ch),       "—"),
]

# --- render table -------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HEADERS   = ("#", "Field", "Pixels", "Description", "Value", "Units")
COL_FRAC  = [0.03, 0.17, 0.08, 0.35, 0.28, 0.09]

ROW_H  = 0.038
HDR_H  = 0.052

BG_EVEN = "#f7f9fc"
BG_ODD  = "#ffffff"
BG_HDR  = "#1a3a5c"
FG_HDR  = "white"
FG_DATA = "#1a1a1a"
MONO    = "DejaVu Sans Mono"
SANS    = "DejaVu Sans"
PAD     = 0.008

n_rows     = len(TABLE)
total_h    = HDR_H + n_rows * ROW_H
fig_h_in   = max(5, total_h * 18)
fig_w_in   = 15

fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
fig.patch.set_facecolor("white")

xs = [0.0]
for f in COL_FRAC:
    xs.append(xs[-1] + f)

# header bar
y = 1.0 - HDR_H / total_h
for label, x0, x1 in zip(HEADERS, xs[:-1], xs[1:]):
    ax.add_patch(plt.Rectangle((x0, y), x1 - x0, HDR_H / total_h,
                                transform=ax.transAxes, color=BG_HDR,
                                zorder=1, clip_on=False))
    ax.text(x0 + PAD, y + (HDR_H / total_h) * 0.5, label,
            transform=ax.transAxes, ha="left", va="center",
            fontsize=9, fontweight="bold", color=FG_HDR,
            fontfamily=SANS, clip_on=False)
y -= HDR_H / total_h

# data rows
for i, (fname, pixels, desc, value, units) in enumerate(TABLE):
    h = ROW_H / total_h
    bg = BG_EVEN if i % 2 == 0 else BG_ODD
    ax.add_patch(plt.Rectangle((0, y), 1, h,
                                transform=ax.transAxes, color=bg,
                                zorder=1, clip_on=False))

    cells = (str(i + 1), fname, pixels, desc, value, units)
    fonts  = (SANS, MONO, MONO, SANS, MONO, SANS)
    for txt, x0, x1, ff in zip(cells, xs[:-1], xs[1:], fonts):
        ax.text(x0 + PAD, y + h * 0.5, txt,
                transform=ax.transAxes, ha="left", va="center",
                fontsize=8, color=FG_DATA, fontfamily=ff, clip_on=False)

    ax.axhline(y, color="#cccccc", linewidth=0.3, zorder=2)
    y -= h

# outer border
ax.add_patch(plt.Rectangle((0, y), 1, 1 - y,
                            transform=ax.transAxes, fill=False,
                            edgecolor="#1a3a5c", linewidth=1.2,
                            zorder=3, clip_on=False))

src_name = os.path.basename(bin_path)
fig.suptitle(f"Embedded header metadata  —  {src_name}",
             fontsize=10, fontweight="bold", color=BG_HDR,
             y=0.998, fontfamily=SANS)

out_path = os.path.splitext(bin_path)[0] + "_metadata.png"
plt.savefig(out_path, dpi=180, bbox_inches="tight",
            facecolor="white", edgecolor="none")
print(f"Saved: {out_path}")
