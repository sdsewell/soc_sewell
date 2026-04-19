"""
G01 — Synthetic Metadata Generator.

Spec:      docs/specs/G01_synthetic_metadata_generator_2026-04-16.md
Spec date: 2026-04-16
Generated: 2026-04-17
Tool:      Claude Code
CONOPS:    WC-SE-0003 WindCube Concept of Operations, V8
Usage:     python validation/gen01_synthetic_metadata_generator_2026_04_16.py
"""

import os
import struct
import sys
import pathlib
import dataclasses

# On Windows, conda DLLs (OpenBLAS, etc.) are in Library\bin inside the env.
# VS Code sets the Python exe but not PATH, so prepend conda dirs before any
# scientific package is imported.
if sys.platform == "win32":
    _py = pathlib.Path(sys.executable).parent
    _conda_dirs = [
        str(_py),
        str(_py / "Library" / "bin"),
        str(_py / "Library" / "mingw-w64" / "bin"),
        str(_py / "Library" / "usr" / "bin"),
        str(_py / "Scripts"),
    ]
    _existing = os.environ.get("PATH", "").split(os.pathsep)
    _new_dirs = [d for d in _conda_dirs if d not in _existing and pathlib.Path(d).is_dir()]
    if _new_dirs:
        os.environ["PATH"] = os.pathsep.join(_new_dirs) + os.pathsep + os.environ.get("PATH", "")
import tkinter as tk
from tkinter import filedialog

_project_root = pathlib.Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import pandas as pd
from astropy.time import Time

from src.geometry.nb01_orbit_propagator_2026_04_16 import propagate_orbit
from src.geometry.nb02a_boresight_2026_04_16 import compute_los_eci
from src.geometry.nb02b_tangent_point_2026_04_16 import compute_tangent_point
from src.geometry.nb02c_los_projection_2026_04_16 import compute_v_rel
from src.metadata.p01_image_metadata_2026_04_06 import (
    ImageMetadata,
    AdcsQualityFlags,
    compute_adcs_quality_flag,
)
from src.constants import WGS84_A_M, EARTH_GRAV_PARAM_M3_S2

# ---------------------------------------------------------------------------
# Constants — scheduling / instrument
# ---------------------------------------------------------------------------

SCHED_DT_S            = 10.0   # NB01 propagation cadence — always fixed
CAL_TRIGGER_LAT_DEG   = 70.0   # CONOPS ascending trigger — see spec header
SIGMA_POINTING_ARCSEC =  5.0
ETALON_TEMP_MEAN_C    = 24.0
ETALON_TEMP_STD_C     =  0.1
CCD_TEMP_MEAN_C       = -10.0
CCD_TEMP_STD_C        =   1.0
EXP_UNIT              = 38500  # timing register value
TIMER_PERIOD_S        = 0.001  # seconds per count when exp_unit = 38500

WIND_MAP_TAGS = {
    "1": "uniform",
    "2": "sine_lat",
    "3": "wave4",
    "4": "hwm14",
    "5": "storm",
}

# ---------------------------------------------------------------------------
# Constants — FPI optical model (§7.1)
# ---------------------------------------------------------------------------

LAMBDA_OI_M      = 630.0e-9       # OI 630.0 nm source wavelength, m
LAMBDA_NE1_M     = 640.2248e-9    # Neon strong line (Burns et al. 1950)
LAMBDA_NE2_M     = 638.2991e-9    # Neon weak line  (Burns et al. 1950)
ETALON_GAP_M     = 20.106e-3      # Tolansky-recovered gap (S03)
FOCAL_LENGTH_M   = 0.19912        # Imaging lens focal length, m
PLATE_SCALE_RPX  = 1.6071e-4      # rad/px (2×2 binned, Tolansky)
R_REFL           = 0.53           # Effective reflectivity (FlatSat)
N_GAP            = 1.0            # Refractive index of etalon gap (air)
C_LIGHT_MS       = 2.99792458e8   # Speed of light, m/s

FINESSE_F        = 4 * R_REFL / (1 - R_REFL) ** 2   # ≈ 9.6 for R=0.53

# CCD / pixel layout
NX_PIX           = 256
NY_PIX           = 256
N_ROWS_BIN       = 259
N_COLS_BIN       = 276
ROW_OFFSET_PIX   = 1
COL_OFFSET_PIX   = 10
BIAS_ADU         = 100
ADU_MAX          = 16383

# Signal levels
SCI_PEAK_ADU     = 5000
SCI_READ_NOISE   = 5.0
CAL_PEAK_ADU     = 12000
CAL_NE_RATIO     = 3.0
CAL_READ_NOISE   = 5.0

# Dark model
DARK_REF_ADU_S   = 0.05
T_REF_DARK_C     = -20.0
T_DOUBLE_C       = 6.5
DARK_READ_NOISE  = 5.0


# ---------------------------------------------------------------------------
# Helper: interactive text prompt
# ---------------------------------------------------------------------------

def _prompt(msg: str, default, cast, lo=None, hi=None):
    """Re-prompts on ValueError or out-of-range. Blank → default."""
    while True:
        raw = input(msg).strip()
        if raw == "":
            return default
        try:
            val = cast(raw)
        except (ValueError, TypeError):
            print(f"  Invalid input — expected {cast.__name__}. Try again.")
            continue
        if lo is not None and val < lo:
            print(f"  Value {val} below minimum {lo}. Try again.")
            continue
        if hi is not None and val > hi:
            print(f"  Value {val} above maximum {hi}. Try again.")
            continue
        return val


# ---------------------------------------------------------------------------
# Helper: Windows folder-picker dialog
# ---------------------------------------------------------------------------

def _pick_folder(title: str, default: str) -> str:
    """
    Open a native Windows folder-browser dialog.
    Returns the selected path, or default if the user cancels.
    """
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder = filedialog.askdirectory(
        title=title,
        initialdir=default,
    )
    root.destroy()
    return folder if folder else default


# ---------------------------------------------------------------------------
# Helper: pointing error quaternion  (§4.1)
# ---------------------------------------------------------------------------

def _pointing_error_quat(rng: np.random.Generator, sigma_arcsec: float) -> list:
    """Return a small random rotation quaternion [x, y, z, w] (scalar-last)."""
    sigma_rad = sigma_arcsec * (np.pi / 648000.0)
    theta = rng.normal(0.0, sigma_rad)          # draw 1: rotation magnitude
    raw   = rng.standard_normal(3)              # draws 2-4: axis components
    n     = np.linalg.norm(raw)
    raw   = raw / n if n > 1e-12 else np.array([1.0, 0.0, 0.0])
    s     = np.sin(theta / 2.0)
    qe    = np.array([raw[0] * s, raw[1] * s, raw[2] * s, np.cos(theta / 2.0)])
    qe    = qe / np.linalg.norm(qe)
    return qe.tolist()


# ---------------------------------------------------------------------------
# Helper: img_type classification  (§3.4 — mirrors P01 logic)
# ---------------------------------------------------------------------------

def _classify_img_type(lamp_ch_array: list, gpio_pwr_on: list) -> str:
    """P01 classification logic — keep in sync with p01_image_metadata_2026_04_06.py."""
    if any(lamp_ch_array):
        return "cal"
    elif gpio_pwr_on[0] == 1 and gpio_pwr_on[3] == 1:
        return "dark"
    return "science"


# ---------------------------------------------------------------------------
# Helper: instrument state per frame type  (§3.2–3.3)
# ---------------------------------------------------------------------------

def _instrument_state(frame_type: str) -> tuple:
    """Returns (gpio_pwr_on, lamp_ch_array)."""
    if frame_type == "science":
        return [0, 0, 0, 0], [0, 0, 0, 0, 0, 0]
    elif frame_type == "cal":
        return [0, 1, 1, 0], [1, 1, 1, 1, 1, 1]
    elif frame_type == "dark":
        return [1, 0, 0, 1], [0, 0, 0, 0, 0, 0]
    else:
        raise ValueError(f"Unknown frame_type: {frame_type!r}")


# ---------------------------------------------------------------------------
# Wind map builder functions  (§3)
# ---------------------------------------------------------------------------

def _build_uniform(rng, h_target_km, v_zonal_ms=100.0, v_merid_ms=0.0):
    from src.windmap import UniformWindMap
    return UniformWindMap(v_zonal_ms=v_zonal_ms, v_merid_ms=v_merid_ms)


def _build_analytic_sine(rng, h_target_km, A_zonal_ms=200.0, A_merid_ms=100.0):
    from src.windmap import AnalyticWindMap
    return AnalyticWindMap(pattern="sine_lat",
                           A_zonal_ms=A_zonal_ms, A_merid_ms=A_merid_ms)


def _build_analytic_wave4(rng, h_target_km,
                          A_zonal_ms=150.0, A_merid_ms=75.0, phase_rad=0.0):
    from src.windmap import AnalyticWindMap
    return AnalyticWindMap(pattern="wave4", A_zonal_ms=A_zonal_ms,
                           A_merid_ms=A_merid_ms, phase_rad=phase_rad)


def _build_hwm14(rng, h_target_km, day_of_year=172, ut_hours=12.0,
                 f107=150.0, ap=4.0):
    from src.windmap import HWM14WindMap
    return HWM14WindMap(alt_km=h_target_km, day_of_year=int(day_of_year),
                        ut_hours=ut_hours, f107=f107, ap=float(ap))


def _build_storm(rng, h_target_km, day_of_year=355, ut_hours=3.0,
                 f107=180.0, ap=80.0):
    from src.windmap import StormWindMap
    return StormWindMap(alt_km=h_target_km, day_of_year=int(day_of_year),
                        ut_hours=ut_hours, f107=f107, ap=float(ap))


# ---------------------------------------------------------------------------
# Wind map registry  (§3)
# ---------------------------------------------------------------------------

WIND_MAP_REGISTRY: dict = {
    "1": ("Uniform constant",   _build_uniform),
    "2": ("Analytic sine_lat",  _build_analytic_sine),
    "3": ("Analytic wave4/DE3", _build_analytic_wave4),
    "4": ("HWM14 quiet-time",   _build_hwm14),
    "5": ("HWM14 storm/DWM07",  _build_storm),
}


def _build_wind_map(choice: str, rng, h_target_km: float, **user_params):
    """Construct a WindMap from a registry key and user-supplied parameters."""
    label, builder = WIND_MAP_REGISTRY[choice]
    return builder(rng, h_target_km, **user_params)


# ---------------------------------------------------------------------------
# Schedule builder  (§3.2–3.5)
# ---------------------------------------------------------------------------

def _build_schedule(
    df_sched: pd.DataFrame,
    lat_band_deg: float,
    n_caldark: int,
    step: int,
) -> tuple:
    """
    Returns (obs_indices, frame_types, cal_trigger_indices) sorted by grid row index.
    Cal/dark takes precedence over science at any overlap.
    """
    n   = len(df_sched)
    lat = df_sched["lat_deg"].values

    science_indices = []
    in_band         = False
    band_entry_i    = None

    for i in range(n):
        if abs(lat[i]) <= lat_band_deg:
            if not in_band:
                in_band      = True
                band_entry_i = i
            if (i - band_entry_i) % step == 0:
                science_indices.append(i)
        else:
            in_band      = False
            band_entry_i = None

    cal_trigger_indices = []
    for i in range(1, n):
        if (lat[i]     >  CAL_TRIGGER_LAT_DEG
                and lat[i - 1] <= CAL_TRIGGER_LAT_DEG
                and lat[i]     >  lat[i - 1]):
            cal_trigger_indices.append(i)

    cal_indices  = []
    dark_indices = []
    for t0 in cal_trigger_indices:
        for k in range(n_caldark):
            idx = t0 + k * step
            if idx < n:
                cal_indices.append(idx)
        for k in range(n_caldark):
            idx = t0 + (n_caldark + k) * step
            if idx < n:
                dark_indices.append(idx)

    cal_dark_set  = set(cal_indices) | set(dark_indices)
    science_final = [i for i in science_indices if i not in cal_dark_set]

    all_tagged = (
        [(i, "science") for i in science_final]
        + [(i, "cal")     for i in cal_indices]
        + [(i, "dark")    for i in dark_indices]
    )
    all_tagged.sort(key=lambda x: x[0])

    obs_indices = [x[0] for x in all_tagged]
    frame_types = [x[1] for x in all_tagged]
    return obs_indices, frame_types, cal_trigger_indices


# ---------------------------------------------------------------------------
# Binary header encoder helpers (§7.2)
# ---------------------------------------------------------------------------

def _encode_u64(val: int) -> list:
    """uint64 → 4 BE uint16 words in LE word order (LSW first)."""
    return [(val >> (16 * i)) & 0xFFFF for i in range(4)]


def _encode_f64(val: float) -> list:
    """float64 → 4 BE uint16 words in LE word order (LSW first)."""
    b     = struct.pack(">d", val)
    words = struct.unpack(">4H", b)        # [MSW, w1, w2, LSW]
    return list(reversed(words))           # [LSW, w2, w1, MSW]


def _encode_header(meta: ImageMetadata) -> np.ndarray:
    """
    Encode ImageMetadata into the 276-word binary header row.

    Quaternion convention: pipeline [x,y,z,w] → binary [w,x,y,z].
    Applied to both attitude_quaternion and pointing_error (words 28–59).
    """
    h = np.zeros(276, dtype=">u2")

    h[0] = meta.rows
    h[1] = meta.cols
    h[2] = meta.exp_time
    h[3] = meta.exp_unit

    for i, w in enumerate(_encode_f64(meta.ccd_temp1)):
        h[4 + i] = w

    for i, w in enumerate(_encode_u64(meta.lua_timestamp)):
        h[8 + i] = w

    for i, w in enumerate(_encode_u64(meta.adcs_timestamp)):
        h[12 + i] = w

    for j, val in enumerate([meta.spacecraft_latitude,
                              meta.spacecraft_longitude,
                              meta.spacecraft_altitude]):
        for i, w in enumerate(_encode_f64(val)):
            h[16 + j * 4 + i] = w

    # Pipeline [x,y,z,w] → binary [w,x,y,z]
    q = meta.attitude_quaternion
    q_bin = [q[3], q[0], q[1], q[2]]
    for j, val in enumerate(q_bin):
        for i, w in enumerate(_encode_f64(val)):
            h[28 + j * 4 + i] = w

    qe = meta.pointing_error
    qe_bin = [qe[3], qe[0], qe[1], qe[2]]
    for j, val in enumerate(qe_bin):
        for i, w in enumerate(_encode_f64(val)):
            h[44 + j * 4 + i] = w

    for j, val in enumerate(meta.pos_eci_hat):
        for i, w in enumerate(_encode_f64(val)):
            h[60 + j * 4 + i] = w

    for j, val in enumerate(meta.vel_eci_hat):
        for i, w in enumerate(_encode_f64(val)):
            h[72 + j * 4 + i] = w

    for j, val in enumerate(meta.etalon_temps):
        for i, w in enumerate(_encode_f64(val)):
            h[84 + j * 4 + i] = w

    for j, val in enumerate(meta.gpio_pwr_on):
        h[100 + j] = int(val) & 0xFF

    for j, val in enumerate(meta.lamp_ch_array):
        h[104 + j] = int(val) & 0xFF

    return h


# ---------------------------------------------------------------------------
# Pixel image generators (§7.3)
# ---------------------------------------------------------------------------

def _generate_science_pixels(v_rel_ms: float, rng) -> np.ndarray:
    """Generate OI 630 nm Airy fringe image with Doppler shift v_rel_ms."""
    lambda_obs = LAMBDA_OI_M * (1.0 + v_rel_ms / C_LIGHT_MS)

    cx, cy = NX_PIX / 2.0, NY_PIX / 2.0
    x = np.arange(NX_PIX) - cx
    y = np.arange(NY_PIX) - cy
    XX, YY = np.meshgrid(x, y)
    r_px = np.sqrt(XX**2 + YY**2)

    theta = r_px * PLATE_SCALE_RPX
    delta = 4.0 * np.pi * N_GAP * ETALON_GAP_M * np.cos(theta) / lambda_obs
    I_airy = 1.0 / (1.0 + FINESSE_F * np.sin(delta / 2.0)**2)

    signal = SCI_PEAK_ADU * I_airy
    photon = rng.poisson(np.clip(signal, 0, None))
    read   = rng.normal(0.0, SCI_READ_NOISE, size=signal.shape)
    image  = np.round(photon + read + BIAS_ADU).astype(np.float32)
    return np.clip(image, 0, ADU_MAX).astype(np.uint16)


def _generate_cal_pixels(rng) -> np.ndarray:
    """Generate two-line neon calibration fringe image."""
    cx, cy = NX_PIX / 2.0, NY_PIX / 2.0
    x = np.arange(NX_PIX) - cx
    y = np.arange(NY_PIX) - cy
    XX, YY = np.meshgrid(x, y)
    r_px   = np.sqrt(XX**2 + YY**2)
    theta  = r_px * PLATE_SCALE_RPX

    def _airy(lam):
        delta = 4.0 * np.pi * N_GAP * ETALON_GAP_M * np.cos(theta) / lam
        return 1.0 / (1.0 + FINESSE_F * np.sin(delta / 2.0)**2)

    I_cal = (CAL_NE_RATIO * _airy(LAMBDA_NE1_M) + _airy(LAMBDA_NE2_M)) \
            / (CAL_NE_RATIO + 1.0)

    signal = CAL_PEAK_ADU * I_cal
    photon = rng.poisson(np.clip(signal, 0, None))
    read   = rng.normal(0.0, CAL_READ_NOISE, size=signal.shape)
    image  = np.round(photon + read + BIAS_ADU).astype(np.float32)
    return np.clip(image, 0, ADU_MAX).astype(np.uint16)


def _generate_dark_pixels(ccd_temp1_c: float, exp_time_s: float, rng) -> np.ndarray:
    """Generate dark frame based on CCD temperature and exposure time."""
    dark_rate  = DARK_REF_ADU_S * 2.0**((ccd_temp1_c - T_REF_DARK_C) / T_DOUBLE_C)
    mean_dark  = max(dark_rate * exp_time_s, 0.0)
    dark_arr   = rng.poisson(mean_dark, size=(NY_PIX, NX_PIX)).astype(float)
    read       = rng.normal(0.0, DARK_READ_NOISE, size=(NY_PIX, NX_PIX))
    image      = np.round(dark_arr + read + BIAS_ADU).astype(np.float32)
    return np.clip(image, 0, ADU_MAX).astype(np.uint16)


def _generate_pixels(frame_type: str, v_rel_ms, ccd_temp1_c: float,
                     exp_time_cts: int, rng) -> np.ndarray:
    """Dispatch to the appropriate pixel generator. Pixel draws always after 9 metadata draws."""
    exp_time_s = exp_time_cts * TIMER_PERIOD_S
    if frame_type == "science":
        return _generate_science_pixels(v_rel_ms, rng)
    elif frame_type == "cal":
        return _generate_cal_pixels(rng)
    elif frame_type == "dark":
        return _generate_dark_pixels(ccd_temp1_c, exp_time_s, rng)
    else:
        raise ValueError(f"Unknown frame_type: {frame_type!r}")


# ---------------------------------------------------------------------------
# Binary file writer (§7.4)
# ---------------------------------------------------------------------------

def _write_bin_file(meta: ImageMetadata,
                    pixels_256: np.ndarray,
                    path: pathlib.Path) -> None:
    """
    Write a WindCube FPI binary image file.

    Layout: row 0 = 276-word header; rows 1–259 = 259×276 pixel data.
    Total = 260 × 276 × 2 = 143,520 bytes.
    The 256×256 science region is embedded at rows [1:257], cols [10:266].
    """
    header      = _encode_header(meta)
    pixel_array = np.full((N_ROWS_BIN, N_COLS_BIN), BIAS_ADU, dtype=np.uint16)
    pixel_array[ROW_OFFSET_PIX : ROW_OFFSET_PIX + NY_PIX,
                COL_OFFSET_PIX : COL_OFFSET_PIX + NX_PIX] = pixels_256

    full_array = np.vstack([
        header.reshape(1, N_COLS_BIN),
        pixel_array.astype(">u2"),
    ]).astype(">u2")

    assert full_array.shape  == (260, 276),  f"Unexpected shape: {full_array.shape}"
    assert full_array.nbytes == 143_520,     f"Unexpected size: {full_array.nbytes}"
    path.write_bytes(full_array.tobytes())


# ---------------------------------------------------------------------------
# Binary filename convention (§7.5)
# ---------------------------------------------------------------------------

def _bin_filename(meta: ImageMetadata) -> str:
    """
    Return the binary filename for this frame.
    Format: YYYYMMDDThhmmssZ_{img_type}.bin  (colons excluded for Windows)
    """
    from datetime import datetime, timezone
    dt     = datetime.fromtimestamp(meta.lua_timestamp / 1000.0, tz=timezone.utc)
    ts_str = dt.strftime("%Y%m%dT%H%M%SZ")
    return f"{ts_str}_{meta.img_type}.bin"


# ---------------------------------------------------------------------------
# Ground track figure
# ---------------------------------------------------------------------------

def _plot_ground_tracks(
    df_sched:      pd.DataFrame,
    metadata_list: list,
    obs_indices:   list,
    frame_types:   list,
    lat_band_deg:  float,
    switch_at:     list,
    save_path:     pathlib.Path,
) -> None:
    """
    Plot spacecraft and tangent-point ground tracks for three attitude segments:
      Seg 0 — partial along-track  (t=0 → first cal/dark sequence ends)
      Seg 1 — full cross-track     (first → second attitude switch)
      Seg 2 — full along-track     (second → third attitude switch)

    Tangent points plotted only for science frames within the science band.
    A dashed connector is drawn between the spacecraft position and its
    tangent point for every 10th science observation.
    Annotated event markers:
      * TLE Acquisition                   — first timestep overall
      * Change attitude to cross-track    — switch_at[0]
      * Change attitude to along-track    — switch_at[1]
      ▲ Cal/dark sequence start           — first cal frame per segment
      ▼ Cal/dark sequence end             — last dark frame per segment
    """
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines

    lats_all = df_sched["lat_deg"].values
    lons_all = df_sched["lon_deg"].values
    n        = len(df_sched)

    # --- Three attitude-based segments ---
    sw0 = min(switch_at[0], n) if len(switch_at) > 0 else n
    sw1 = min(switch_at[1], n) if len(switch_at) > 1 else n
    sw2 = min(switch_at[2], n) if len(switch_at) > 2 else n

    seg_bounds = [(0, sw0), (sw0, sw1), (sw1, sw2)]
    seg_labels = [
        "Orbit 1  (partial, along-track)",
        "Orbit 2  (cross-track)",
        "Orbit 3  (along-track)",
    ]
    seg_sc_cols  = ["#6baed6", "#d6604d", "#2166ac"]   # SC track colours
    seg_tp_cols  = ["#c6dbef", "#fdae61", "#74add1"]   # tangent point colours
    seg_cn_cols  = ["#6baed6", "#d6604d", "#2166ac"]   # connector colours

    # Science observations per segment
    idx_arr = np.arange(n)
    sci_by_seg = []
    for s_start, s_end in seg_bounds:
        sci_by_seg.append([
            (obs_indices[i], metadata_list[i])
            for i, ft in enumerate(frame_types)
            if ft == "science" and s_start <= obs_indices[i] < s_end
        ])

    # Cal/dark sequence bounds (first cal idx, last dark idx) per segment
    def _cd_bounds(s_start, s_end):
        idxs = sorted(
            obs_indices[i]
            for i, ft in enumerate(frame_types)
            if ft in ("cal", "dark") and s_start <= obs_indices[i] < s_end
        )
        return (idxs[0], idxs[-1]) if idxs else None

    cd_bounds = [_cd_bounds(s, e) for s, e in seg_bounds]

    # --- Figure setup ---
    _cartopy = False
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        fig = plt.figure(figsize=(16, 8))
        ax  = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color="0.35")
        ax.add_feature(cfeature.BORDERS,   linewidth=0.3, color="0.55")
        ax.set_global()
        ax.gridlines(draw_labels=True, linewidth=0.3, color="0.7", alpha=0.7)
        _tr = ccrs.PlateCarree()
        _cartopy = True
    except ImportError:
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
        ax.grid(True, linewidth=0.3, color="0.7", alpha=0.7)
        _tr = None

    def _sc(x, y, **kw):
        kw_tr = {"transform": _tr} if _cartopy else {}
        ax.scatter(x, y, **kw, **kw_tr)

    def _ln(xs, ys, **kw):
        kw_tr = {"transform": _tr} if _cartopy else {}
        ax.plot(xs, ys, **kw, **kw_tr)

    def _ann(lon, lat, label, marker, color, text_offset=(12, 10)):
        kw_tr = {"transform": _tr} if _cartopy else {}
        ax.scatter([lon], [lat], marker=marker, c=color, s=110, zorder=10,
                   edgecolors="white", linewidths=0.9, **kw_tr)
        ann_kw = {}
        if _cartopy:
            ann_kw["xycoords"] = _tr._as_mpl_transform(ax)
        ax.annotate(
            label,
            xy=(lon, lat),
            xytext=text_offset,
            textcoords="offset points",
            fontsize=6.5,
            color=color,
            arrowprops=dict(arrowstyle="-", color=color, lw=0.6, alpha=0.8),
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec=color, lw=0.7, alpha=0.92),
            zorder=11,
            **ann_kw,
        )

    # --- SC ground tracks + tangent points ---
    for seg, (s_start, s_end) in enumerate(seg_bounds):
        if s_start >= n:
            continue
        seg_mask = (idx_arr >= s_start) & (idx_arr < s_end)
        _sc(lons_all[seg_mask], lats_all[seg_mask],
            c=seg_sc_cols[seg], s=1.5, alpha=0.45, zorder=3, linewidths=0)

        for k, (idx, m) in enumerate(sci_by_seg[seg]):
            if k == 0:
                # Scatter all tangent pts for this segment in one call
                tp_lats = [mm.tangent_lat for _, mm in sci_by_seg[seg]]
                tp_lons = [mm.tangent_lon for _, mm in sci_by_seg[seg]]
                _sc(tp_lons, tp_lats,
                    c=seg_tp_cols[seg], s=6, alpha=0.8, zorder=5, linewidths=0)
            if k % 10 == 0:
                sc_lat = df_sched.loc[idx, "lat_deg"]
                sc_lon = df_sched.loc[idx, "lon_deg"]
                _ln([sc_lon, m.tangent_lon], [sc_lat, m.tangent_lat],
                    linestyle="--", color=seg_cn_cols[seg],
                    lw=0.9, alpha=0.55, zorder=4)

    # Science band boundaries
    for bnd in [lat_band_deg, -lat_band_deg]:
        _ln([-180, 180], [bnd, bnd], linestyle=":", color="k", lw=0.8, alpha=0.55)

    # --- Event markers ---
    _CD_COL   = "#b5651d"
    _ATT_COLS = ["#7b2fa8", "#0e6655"]   # cross-track switch, along-track switch

    # TLE Acquisition: very first timestep
    _ann(float(lons_all[0]), float(lats_all[0]),
         "TLE Acquisition", "*", "#1a7d1a", text_offset=(12, 10))

    # Attitude switch markers
    att_events = [
        (sw0, "Change attitude to cross-track",  _ATT_COLS[0], (12, -20)),
        (sw1, "Change attitude to along-track",  _ATT_COLS[1], (12,  10)),
    ]
    for sw_idx, label, col, off in att_events:
        if sw_idx < n:
            _ann(float(lons_all[sw_idx]), float(lats_all[sw_idx]),
                 label, "*", col, text_offset=off)

    # Cal/dark start (▲) and end (▼) per segment
    seg_labels_short = ["Orb 1", "Orb 2", "Orb 3"]
    for seg, bounds in enumerate(cd_bounds):
        if bounds is None:
            continue
        idx_start, idx_end = bounds
        _ann(float(df_sched.loc[idx_start, "lon_deg"]),
             float(df_sched.loc[idx_start, "lat_deg"]),
             f"Cal/dark start ({seg_labels_short[seg]})",
             "^", _CD_COL, text_offset=(10, 10))
        _ann(float(df_sched.loc[idx_end, "lon_deg"]),
             float(df_sched.loc[idx_end, "lat_deg"]),
             f"Cal/dark end ({seg_labels_short[seg]})",
             "v", _CD_COL, text_offset=(10, -18))

    # --- Legend ---
    handles = []
    for seg in range(3):
        handles += [
            mlines.Line2D([], [], color=seg_sc_cols[seg], marker="o", ls="None",
                          markersize=5, label=f"{seg_labels[seg]}  S/C"),
            mlines.Line2D([], [], color=seg_tp_cols[seg], marker="o", ls="None",
                          markersize=5, label=f"{seg_labels[seg]}  tangent pts"),
            mlines.Line2D([], [], color=seg_cn_cols[seg], ls="--", lw=1.1,
                          label=f"{seg_labels[seg]}  S/C → tangent pt  (every 10th)"),
        ]
    handles += [
        mlines.Line2D([], [], color="k", ls=":", lw=0.9,
                      label=f"Science band  ±{lat_band_deg:.0f}°"),
        mlines.Line2D([], [], color="#1a7d1a", marker="*", ls="None",
                      markersize=8, label="TLE Acquisition"),
        mlines.Line2D([], [], color=_ATT_COLS[0], marker="*", ls="None",
                      markersize=8, label="Change attitude to cross-track"),
        mlines.Line2D([], [], color=_ATT_COLS[1], marker="*", ls="None",
                      markersize=8, label="Change attitude to along-track"),
        mlines.Line2D([], [], color=_CD_COL, marker="^", ls="None",
                      markersize=6, label="Cal/dark sequence start"),
        mlines.Line2D([], [], color=_CD_COL, marker="v", ls="None",
                      markersize=6, label="Cal/dark sequence end"),
    ]
    ax.legend(handles=handles, fontsize=6.5, loc="lower left",
              framealpha=0.85, edgecolor="0.6", ncol=2)
    ax.set_title(
        "G01 Ground Tracks — Orbits 1, 2 & 3  "
        "(partial along-track stub + two complete orbits)\n"
        "S/C positions (all latitudes)  +  observed-volume tangent pts (science band only)"
    )
    plt.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== G01 — WindCube Synthetic Metadata Generator ===\n")

    t_start       = _prompt(
        "Start epoch          [2027-01-01T00:00:00 UTC]  : ",
        "2027-01-01T00:00:00", str)
    duration_days = _prompt(
        "Duration             [days,  default   1       ] : ",
        1.0, float, 0.1, 365.0)
    lat_band_deg  = _prompt(
        "Science band         [deg,   default  40       ] : ",
        40.0, float, 5.0, 89.0)
    obs_cadence_s = _prompt(
        "Obs. cadence         [sec,   default  10       ] : ",
        10.0, float, 10.0, 3600.0)
    n_caldark     = _prompt(
        "Cal/dark frames (n)  [int,   default   5       ] : ",
        5, int, 1, 50)
    exp_time_cts  = _prompt(
        "Exposure time        [cts,   default 8000      ] : ",
        8000, int, 100, 100000)
    altitude_km   = _prompt(
        "S/C altitude         [km,    default 510       ] : ",
        510.0, float, 400.0, 700.0)
    h_target_km   = _prompt(
        "Tangent height       [km,    default 250       ] : ",
        250.0, float, 100.0, 400.0)
    rng_seed      = _prompt(
        "Random seed          [int,   default  42       ] : ",
        42, int, 0, None)

    exp_time_cs = round(exp_time_cts * TIMER_PERIOD_S * 100)

    # -----------------------------------------------------------------------
    # Wind map selection menu
    # -----------------------------------------------------------------------
    wind_map = None
    wm_choice = None
    while True:
        print("\nSelect wind map for science frame truth winds:")
        for key, (label, _) in WIND_MAP_REGISTRY.items():
            print(f"  [{key}] {label}")
        wm_choice = _prompt("Choice [default 1]: ", "1", str)
        if wm_choice not in WIND_MAP_REGISTRY:
            print("  Invalid choice. Try again.")
            continue

        wm_params = {}
        if wm_choice == "1":
            wm_params["v_zonal_ms"] = _prompt(
                "  v_zonal  [m/s, default 100] : ", 100.0, float)
            wm_params["v_merid_ms"] = _prompt(
                "  v_merid  [m/s, default   0] : ", 0.0, float)
        elif wm_choice == "2":
            wm_params["A_zonal_ms"] = _prompt(
                "  A_zonal  [m/s, default 200] : ", 200.0, float)
            wm_params["A_merid_ms"] = _prompt(
                "  A_merid  [m/s, default 100] : ", 100.0, float)
        elif wm_choice == "3":
            wm_params["A_zonal_ms"] = _prompt(
                "  A_zonal  [m/s, default 150] : ", 150.0, float)
            wm_params["A_merid_ms"] = _prompt(
                "  A_merid  [m/s, default  75] : ", 75.0, float)
            wm_params["phase_rad"]  = _prompt(
                "  phase    [rad, default 0.0] : ", 0.0, float)
        elif wm_choice == "4":
            wm_params["day_of_year"] = _prompt(
                "  day_of_year [default 172 ] : ", 172, int, 1, 366)
            wm_params["ut_hours"]    = _prompt(
                "  ut_hours    [default  12.0]: ", 12.0, float, 0.0, 24.0)
            wm_params["f107"]        = _prompt(
                "  f107        [default 150.0]: ", 150.0, float, 60.0, 300.0)
            wm_params["ap"]          = _prompt(
                "  ap          [default   4  ]: ", 4.0, float, 0.0, 400.0)
        elif wm_choice == "5":
            wm_params["day_of_year"] = _prompt(
                "  day_of_year [default 355 ] : ", 355, int, 1, 366)
            wm_params["ut_hours"]    = _prompt(
                "  ut_hours    [default   3.0]: ", 3.0, float, 0.0, 24.0)
            wm_params["f107"]        = _prompt(
                "  f107        [default 180.0]: ", 180.0, float, 60.0, 300.0)
            wm_params["ap"]          = _prompt(
                "  ap          [default  80  ]: ", 80.0, float, 0.0, 400.0)

        try:
            wind_map = _build_wind_map(wm_choice, None, h_target_km, **wm_params)
            break
        except ImportError as exc:
            print(f"\n  ERROR: {exc}")
            print("  Cannot use this wind map. Please select a different option.")
            continue

    windmap_tag   = WIND_MAP_TAGS[wm_choice]
    windmap_label = WIND_MAP_REGISTRY[wm_choice][0]

    _default_out = str(pathlib.Path.home() / "WindCube" / "G01_outputs")
    print("\nSelect output folder (dialog opening — check taskbar if not visible)...")
    output_dir = _pick_folder(
        title="Select output folder for G01 files",
        default=_default_out,
    )
    print(f"  Output directory : {output_dir}")

    if lat_band_deg >= CAL_TRIGGER_LAT_DEG:
        print(
            f"\n  WARNING: science band (±{lat_band_deg}°) overlaps the "
            f"{CAL_TRIGGER_LAT_DEG}°N cal/dark trigger latitude. "
            "Cal/dark takes precedence at trigger epochs."
        )
    seq_duration_s = 2 * n_caldark * obs_cadence_s
    if seq_duration_s > 1200:
        print(
            f"\n  WARNING: cal/dark sequence duration ({seq_duration_s:.0f} s = "
            f"{seq_duration_s / 60:.1f} min) > 20 min and may extend into "
            "the orbit's descending high-latitude arc."
        )

    step           = max(1, round(obs_cadence_s / SCHED_DT_S))
    actual_cadence = step * SCHED_DT_S
    duration_s     = duration_days * 86400.0

    a_m       = WGS84_A_M + altitude_km * 1e3
    T_ORBIT_S = 2 * np.pi * np.sqrt(a_m**3 / EARTH_GRAV_PARAM_M3_S2)

    sched_rows_approx = int(duration_s / SCHED_DT_S) + 1

    print(f"\nParameters:")
    print(f"  Start epoch      : {t_start} UTC")
    print(f"  Duration         : {duration_days:.1f} days")
    print(f"  Science band     : ±{lat_band_deg:.1f}°")
    print(f"  Obs. cadence     : {obs_cadence_s:.1f} s requested → "
          f"{actual_cadence:.1f} s actual (step={step})")
    print(f"  Cal/dark per orb : {n_caldark} + {n_caldark} = {2*n_caldark} frames  "
          f"({seq_duration_s:.1f} s sequence)")
    print(f"  Cal trigger lat  : {CAL_TRIGGER_LAT_DEG:.1f}°N ascending  "
          "[CONOPS TBD document, §TBD]")
    print(f"  Exp. time        : {exp_time_cts} counts × {TIMER_PERIOD_S*1000:.1f} ms/count "
          f"= {exp_time_cts * TIMER_PERIOD_S:.3f} s  ({exp_time_cs} cs in P01,  "
          f"exp_unit={EXP_UNIT})")
    print(f"  S/C altitude     : {altitude_km:.1f} km")
    print(f"  Tangent ht       : {h_target_km:.1f} km")
    print(f"  Wind map         : {windmap_label}  "
          f"(tag={windmap_tag!r})")
    print(f"  T_orbit          : {T_ORBIT_S:.1f} s ({T_ORBIT_S / 60:.2f} min)")
    print(f"  NB01 sched rows  : ~{sched_rows_approx}  "
          f"(10 s grid, {duration_days:.0f} days)")
    print(f"  RNG seed         : {rng_seed}")
    print(f"  Output dir       : {output_dir}")

    # -----------------------------------------------------------------------
    # Step 1: NB01 orbit propagation
    # -----------------------------------------------------------------------
    print("\nBuilding NB01 orbit schedule ...", end=" ", flush=True)

    df_sched = propagate_orbit(
        t_start     = t_start,
        duration_s  = duration_s,
        dt_s        = SCHED_DT_S,
        altitude_km = altitude_km,
    )

    t0 = pd.Timestamp(t_start, tz="UTC")
    df_sched["elapsed_s"]    = (df_sched["epoch"] - t0).dt.total_seconds()
    df_sched["orbit_number"] = (df_sched["elapsed_s"] // T_ORBIT_S).astype(int) + 1
    # look_mode is assigned after the schedule is built (needs cal_trigger_indices)
    df_sched["look_mode"] = "along_track"

    print("done")

    # -----------------------------------------------------------------------
    # Step 2: Build observation schedule
    # -----------------------------------------------------------------------
    obs_indices, frame_types, cal_trigger_indices = _build_schedule(
        df_sched, lat_band_deg, n_caldark, step
    )

    # Attitude switches just after each cal/dark sequence completes.
    # switch_at[k] is the first df_sched row index at which the new attitude is in effect.
    switch_at = [t0_i + 2 * n_caldark * step for t0_i in cal_trigger_indices]
    _n   = len(df_sched)
    _lk  = np.empty(_n, dtype=object)
    _md  = "along_track"
    _sw  = 0
    for _ii in range(_n):
        while _sw < len(switch_at) and _ii >= switch_at[_sw]:
            _md = "cross_track" if _md == "along_track" else "along_track"
            _sw += 1
        _lk[_ii] = _md
    df_sched["look_mode"] = _lk
    del _lk, _md, _sw, _ii, _n
    n_obs             = len(obs_indices)
    n_complete_orbits = len(cal_trigger_indices)

    n_science = frame_types.count("science")
    n_cal     = frame_types.count("cal")
    n_dark    = frame_types.count("dark")

    print(f"\nObservation schedule:")
    print(f"  Science frames : {n_science}")
    print(f"  Cal frames     : {n_cal}    "
          f"({n_caldark} per orbit × {n_complete_orbits} complete orbits)")
    print(f"  Dark frames    : {n_dark}    "
          f"({n_caldark} per orbit × {n_complete_orbits} complete orbits)")
    print(f"  Total frames   : {n_obs}")

    if n_science == 0:
        print("\nFATAL: zero science frames produced. Check lat_band_deg and schedule.")
        return
    if n_cal == 0:
        print("\nFATAL: zero cal frames produced. Check CAL_TRIGGER_LAT_DEG and orbit propagation.")
        return

    orbit_counter   = {}
    frame_sequences = []
    for idx in obs_indices:
        orb = int(df_sched.loc[idx, "orbit_number"])
        cnt = orbit_counter.get(orb, 0)
        frame_sequences.append(cnt)
        orbit_counter[orb] = cnt + 1

    # -----------------------------------------------------------------------
    # Step 3: Build ImageMetadata list + binary image files
    # -----------------------------------------------------------------------
    print("\nBuilding NB02a + NB02b + NB02c + image synthesis ...")

    rng      = np.random.default_rng(rng_seed)
    bin_dir  = pathlib.Path(output_dir) / "bin_frames"
    bin_dir.mkdir(parents=True, exist_ok=True)

    metadata_list  = []
    vrel_list      = []   # CSV-only LOS velocity components per frame

    for i, (idx, frame_type) in enumerate(zip(obs_indices, frame_types)):
        row = df_sched.loc[idx]

        pos       = np.array([row.pos_eci_x, row.pos_eci_y, row.pos_eci_z])
        vel       = np.array([row.vel_eci_x, row.vel_eci_y, row.vel_eci_z])
        look_mode = str(row.look_mode)

        # NB02a: attitude quaternion + LOS vector (los_eci retained for NB02b/c)
        los_eci, q = compute_los_eci(
            pos, vel, look_mode,
            altitude_km = altitude_km,
            h_target_km = h_target_km,
        )

        # RNG draw order: PE (4 draws), etalon temps (4 draws), CCD temp (1 draw) = 9 total
        pointing_error = _pointing_error_quat(rng, SIGMA_POINTING_ARCSEC)
        etalon_temps   = rng.normal(ETALON_TEMP_MEAN_C, ETALON_TEMP_STD_C, size=4).tolist()
        ccd_temp1      = float(rng.normal(CCD_TEMP_MEAN_C, CCD_TEMP_STD_C))

        # NB02b + NB02c: tangent point + full LOS decomposition (science frames only)
        tp_lat = tp_lon = tp_alt = None
        v_zonal = v_merid = None
        v_wind_LOS = v_earth_LOS = V_sc_LOS = v_rel_val = None

        if frame_type == "science":
            epoch_t = Time(row.epoch.to_pydatetime(), scale="utc")
            tp = compute_tangent_point(pos, los_eci, epoch_t, h_target_km=h_target_km)
            tp_lat = tp["tp_lat_deg"]
            tp_lon = tp["tp_lon_deg"]
            tp_alt = tp["tp_alt_km"]

            vr = compute_v_rel(
                wind_map,
                tp_lat, tp_lon, tp["tp_eci"],
                vel, los_eci, epoch_t,
            )
            v_wind_LOS  = vr["v_wind_LOS"]
            v_earth_LOS = vr["v_earth_LOS"]
            V_sc_LOS    = vr["V_sc_LOS"]
            v_rel_val   = vr["v_rel"]
            v_zonal     = vr["v_zonal_ms"]
            v_merid     = vr["v_merid_ms"]

        # Instrument state
        gpio, lamp = _instrument_state(frame_type)

        # Derived fields
        img_type       = _classify_img_type(lamp, gpio)
        shutter_status = "closed" if (gpio[0] == 1 and gpio[3] == 1) else "open"
        lamp1_status   = "on" if (lamp[0] or lamp[1]) else "off"
        lamp2_status   = "on" if (lamp[2] or lamp[3]) else "off"
        lamp3_status   = "on" if (lamp[4] or lamp[5]) else "off"

        lua_ts       = int(row.epoch.timestamp() * 1000)
        orbit_num    = int(row.orbit_number)
        orbit_parity = str(row.look_mode)

        adcs_flag = compute_adcs_quality_flag({
            "pointing_error": pointing_error,
            "pos_eci_hat":    pos.tolist(),
            "adcs_timestamp": lua_ts,
        })

        meta = ImageMetadata(
            rows                 = 260,
            cols                 = 276,
            exp_time             = exp_time_cs,
            exp_unit             = EXP_UNIT,
            binning              = 2,
            img_type             = img_type,
            lua_timestamp        = lua_ts,
            adcs_timestamp       = lua_ts,
            utc_timestamp        = row.epoch.isoformat(),
            spacecraft_latitude  = float(np.radians(row.lat_deg)),
            spacecraft_longitude = float(np.radians(row.lon_deg)),
            spacecraft_altitude  = float(row.alt_km * 1e3),
            pos_eci_hat          = pos.tolist(),
            vel_eci_hat          = vel.tolist(),
            attitude_quaternion  = q.tolist(),
            pointing_error       = pointing_error,
            obs_mode             = look_mode,
            ccd_temp1            = ccd_temp1,
            etalon_temps         = etalon_temps,
            shutter_status       = shutter_status,
            gpio_pwr_on          = gpio,
            lamp_ch_array        = lamp,
            lamp1_status         = lamp1_status,
            lamp2_status         = lamp2_status,
            lamp3_status         = lamp3_status,
            orbit_number         = orbit_num,
            frame_sequence       = frame_sequences[i],
            orbit_parity         = orbit_parity,
            adcs_quality_flag    = adcs_flag,
            is_synthetic         = True,
            noise_seed           = i,
            tangent_lat          = tp_lat,
            tangent_lon          = tp_lon,
            tangent_alt_km       = tp_alt,
            truth_v_zonal        = v_zonal,
            truth_v_meridional   = v_merid,
            truth_v_los          = v_wind_LOS,   # populated for science frames (v9)
        )
        metadata_list.append(meta)

        vrel_list.append({
            "v_wind_los_ms":  v_wind_LOS  if frame_type == "science" else float("nan"),
            "v_earth_los_ms": v_earth_LOS if frame_type == "science" else float("nan"),
            "v_sc_los_ms":    V_sc_LOS    if frame_type == "science" else float("nan"),
            "v_rel_ms":       v_rel_val   if frame_type == "science" else float("nan"),
        })

        # Binary image synthesis — pixel draws follow the 9 metadata draws
        pixels_256 = _generate_pixels(frame_type, v_rel_val, ccd_temp1,
                                      exp_time_cts, rng)
        _write_bin_file(meta, pixels_256, bin_dir / _bin_filename(meta))

        if i % max(1, n_obs // 100) == 0 or i == n_obs - 1:
            pct = 100 * (i + 1) / n_obs
            bar = "=" * int(pct // 5) + " " * (20 - int(pct // 5))
            print(f"\r  [{bar}] {i+1}/{n_obs}", end="", flush=True)

    print()
    bin_gb = len(metadata_list) * 143_520 / 1e9
    print(f"  NB02b+NB02c called for {n_science} science frames.")
    print(f"  .bin files written to: {bin_dir}  "
          f"({len(metadata_list)} files, ~{bin_gb:.1f} GB)")

    # -----------------------------------------------------------------------
    # Step 4: Console summaries
    # -----------------------------------------------------------------------
    img_types_out = [m.img_type for m in metadata_list]
    print(f"\nImage type verification:")
    for t in ("science", "cal", "dark"):
        print(f"  {t:7s} : {img_types_out.count(t)}")

    flags_out = [m.adcs_quality_flag for m in metadata_list]
    n_good  = flags_out.count(AdcsQualityFlags.GOOD)
    n_slew  = sum(1 for f in flags_out if f & AdcsQualityFlags.SLEW_IN_PROGRESS)
    print(f"\nADCS quality flags (P01):")
    print(f"  GOOD             : {n_good}")
    print(f"  SLEW_IN_PROGRESS : {n_slew}   (expected ~0)")

    pe_angles = []
    for m in metadata_list:
        qe       = m.pointing_error
        vn       = min(np.sqrt(qe[0]**2 + qe[1]**2 + qe[2]**2), 1.0)
        angle_as = 2.0 * np.degrees(np.arcsin(vn)) * 3600.0
        pe_angles.append(angle_as)
    pe_arr = np.array(pe_angles)
    pe_expected_mean = SIGMA_POINTING_ARCSEC * np.sqrt(2.0 / np.pi)
    pe_expected_std  = SIGMA_POINTING_ARCSEC * np.sqrt(1 - 2.0 / np.pi)
    print(f"\nPointing error stats (arcsec):")
    print(f"  Mean  : {pe_arr.mean():6.2f}   (expected ~{pe_expected_mean:.2f}, half-normal)")
    print(f"  Std   : {pe_arr.std():6.2f}   (expected ~{pe_expected_std:.2f}, half-normal)")
    print(f"  Max   : {pe_arr.max():6.1f}")

    et_all = np.array([m.etalon_temps for m in metadata_list]).flatten()
    print(f"\nEtalon temperature stats (°C):")
    print(f"  Mean  : {et_all.mean():.2f}   (expected ~{ETALON_TEMP_MEAN_C:.2f})")
    print(f"  Std   : {et_all.std():.2f}   (expected ~{ETALON_TEMP_STD_C:.2f})")

    ccd_all = np.array([m.ccd_temp1 for m in metadata_list])
    print(f"\nCCD temperature stats (°C):")
    print(f"  Mean  : {ccd_all.mean():.2f}   (expected ~{CCD_TEMP_MEAN_C:.2f})")
    print(f"  Std   : {ccd_all.std():.2f}   (expected ~{CCD_TEMP_STD_C:.2f})")

    sci_meta = [m for m in metadata_list if m.img_type == "science"]
    if sci_meta:
        tp_lats   = np.array([m.tangent_lat for m in sci_meta])
        vz_all    = np.array([m.truth_v_zonal for m in sci_meta])
        vm_all    = np.array([m.truth_v_meridional for m in sci_meta])
        vlos_all  = np.array([m.truth_v_los for m in sci_meta])
        vrel_sci  = np.array([d["v_rel_ms"] for d in vrel_list
                               if not np.isnan(d["v_rel_ms"])])
        print(f"\nTangent point stats (science frames):")
        print(f"  tp_lat  : mean={tp_lats.mean():.1f}°, "
              f"min={tp_lats.min():.1f}°, max={tp_lats.max():.1f}°")
        print(f"\nWind stats at tangent point (science frames):")
        print(f"  v_zonal : mean={vz_all.mean():.1f}, "
              f"min={vz_all.min():.1f}, max={vz_all.max():.1f} m/s")
        print(f"  v_merid : mean={vm_all.mean():.1f}, "
              f"min={vm_all.min():.1f}, max={vm_all.max():.1f} m/s")
        print(f"\nv_rel stats for science frames (m/s):")
        print(f"  Mean  : {vrel_sci.mean():8.1f}   (dominated by V_sc_LOS)")
        print(f"  Std   : {vrel_sci.std():8.1f}")
        print(f"  Min   : {vrel_sci.min():8.1f}   Max: {vrel_sci.max():.1f}")
        print(f"\nv_wind_LOS stats (m/s):")
        print(f"  Mean  : {vlos_all.mean():8.1f}   (truth wind projected onto LOS)")
        print(f"  Std   : {vlos_all.std():8.1f}")

    # -----------------------------------------------------------------------
    # Step 5: Save outputs
    # -----------------------------------------------------------------------
    out_path = pathlib.Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    t_start_compact = t_start[:10].replace("-", "")
    stem     = (f"GEN01_{t_start_compact}_{duration_days:05.1f}d_"
                f"{windmap_tag}_seed{rng_seed:04d}")
    npy_path = out_path / f"{stem}.npy"
    csv_path = out_path / f"{stem}.csv"

    records = [dataclasses.asdict(m) for m in metadata_list]
    np.save(str(npy_path), np.array(records, dtype=object), allow_pickle=True)

    # Build lookup: df_sched index → (record_dict, vrel_dict, frame_type)
    obs_data = {
        idx: (records[i], vrel_list[i], frame_types[i])
        for i, idx in enumerate(obs_indices)
    }

    # CSV: one row per obs-cadence step across the full schedule.
    # Observed frames carry full metadata (47 cols: obs_type + 46 data cols).
    # Non-observing steps have exp_time=0, obs_type='none', NaN for frame fields.
    all_indices = sorted(set(range(0, len(df_sched), step)) | set(obs_indices))

    _nan = float("nan")
    rows_csv = []
    for idx in all_indices:
        srow = df_sched.loc[idx]
        lua_ts = int(srow.epoch.timestamp() * 1000)

        if idx in obs_data:
            r, vd, ft = obs_data[idx]
            rows_csv.append({
                "obs_type":             ft,
                "rows":                 r["rows"],
                "cols":                 r["cols"],
                "exp_time":             r["exp_time"],
                "exp_unit":             r["exp_unit"],
                "ccd_temp1":            r["ccd_temp1"],
                "lua_timestamp":        r["lua_timestamp"],
                "adcs_timestamp":       r["adcs_timestamp"],
                "spacecraft_latitude":  r["spacecraft_latitude"],
                "spacecraft_longitude": r["spacecraft_longitude"],
                "spacecraft_altitude":  r["spacecraft_altitude"],
                "att_q_x":  r["attitude_quaternion"][0],
                "att_q_y":  r["attitude_quaternion"][1],
                "att_q_z":  r["attitude_quaternion"][2],
                "att_q_w":  r["attitude_quaternion"][3],
                "pe_q_x":   r["pointing_error"][0],
                "pe_q_y":   r["pointing_error"][1],
                "pe_q_z":   r["pointing_error"][2],
                "pe_q_w":   r["pointing_error"][3],
                "pos_eci_x": r["pos_eci_hat"][0],
                "pos_eci_y": r["pos_eci_hat"][1],
                "pos_eci_z": r["pos_eci_hat"][2],
                "vel_eci_x": r["vel_eci_hat"][0],
                "vel_eci_y": r["vel_eci_hat"][1],
                "vel_eci_z": r["vel_eci_hat"][2],
                "etalon_t0": r["etalon_temps"][0],
                "etalon_t1": r["etalon_temps"][1],
                "etalon_t2": r["etalon_temps"][2],
                "etalon_t3": r["etalon_temps"][3],
                "gpio_0": r["gpio_pwr_on"][0],
                "gpio_1": r["gpio_pwr_on"][1],
                "gpio_2": r["gpio_pwr_on"][2],
                "gpio_3": r["gpio_pwr_on"][3],
                "lamp_0": r["lamp_ch_array"][0],
                "lamp_1": r["lamp_ch_array"][1],
                "lamp_2": r["lamp_ch_array"][2],
                "lamp_3": r["lamp_ch_array"][3],
                "lamp_4": r["lamp_ch_array"][4],
                "lamp_5": r["lamp_ch_array"][5],
                "tp_lat_deg":      r["tangent_lat"]       if r["tangent_lat"]       is not None else _nan,
                "tp_lon_deg":      r["tangent_lon"]        if r["tangent_lon"]        is not None else _nan,
                "wind_v_zonal_ms": r["truth_v_zonal"]      if r["truth_v_zonal"]      is not None else _nan,
                "wind_v_merid_ms": r["truth_v_meridional"] if r["truth_v_meridional"] is not None else _nan,
                "v_wind_los_ms":   vd["v_wind_los_ms"],
                "v_earth_los_ms":  vd["v_earth_los_ms"],
                "v_sc_los_ms":     vd["v_sc_los_ms"],
                "v_rel_ms":        vd["v_rel_ms"],
            })
        else:
            # Non-observing step: orbital state only, exp_time=0
            rows_csv.append({
                "obs_type":             "none",
                "rows":                 0,
                "cols":                 0,
                "exp_time":             0,
                "exp_unit":             0,
                "ccd_temp1":            _nan,
                "lua_timestamp":        lua_ts,
                "adcs_timestamp":       lua_ts,
                "spacecraft_latitude":  float(np.radians(srow.lat_deg)),
                "spacecraft_longitude": float(np.radians(srow.lon_deg)),
                "spacecraft_altitude":  float(srow.alt_km * 1e3),
                "att_q_x":  _nan, "att_q_y":  _nan,
                "att_q_z":  _nan, "att_q_w":  _nan,
                "pe_q_x":   _nan, "pe_q_y":   _nan,
                "pe_q_z":   _nan, "pe_q_w":   _nan,
                "pos_eci_x": float(srow.pos_eci_x),
                "pos_eci_y": float(srow.pos_eci_y),
                "pos_eci_z": float(srow.pos_eci_z),
                "vel_eci_x": float(srow.vel_eci_x),
                "vel_eci_y": float(srow.vel_eci_y),
                "vel_eci_z": float(srow.vel_eci_z),
                "etalon_t0": _nan, "etalon_t1": _nan,
                "etalon_t2": _nan, "etalon_t3": _nan,
                "gpio_0": 0, "gpio_1": 0, "gpio_2": 0, "gpio_3": 0,
                "lamp_0": 0, "lamp_1": 0, "lamp_2": 0,
                "lamp_3": 0, "lamp_4": 0, "lamp_5": 0,
                "tp_lat_deg":      _nan, "tp_lon_deg":      _nan,
                "wind_v_zonal_ms": _nan, "wind_v_merid_ms": _nan,
                "v_wind_los_ms":   _nan, "v_earth_los_ms":  _nan,
                "v_sc_los_ms":     _nan, "v_rel_ms":        _nan,
            })

    df_csv = pd.DataFrame(rows_csv)
    df_csv.to_csv(str(csv_path), index=False)

    if wm_choice == "1":
        _wm_title = (
            f"G01  Uniform  "
            f"({wm_params['v_zonal_ms']:+.0f} m/s eastward,  "
            f"{wm_params['v_merid_ms']:+.0f} m/s southward)"
        )
    else:
        _wm_title = f"G01  {windmap_label}"

    png_path = out_path / f"{stem}_windmap_vector.png"
    print("Saving wind map vector plot ...", end=" ", flush=True)
    wind_map.plot(
        title=_wm_title,
        alt_km=h_target_km,
        subsample=8,
        mode="vector",
        save_path=str(png_path),
    )
    print("done")

    gt_path = out_path / f"{stem}_ground_tracks.png"
    print("Saving ground track figure ...", end=" ", flush=True)
    _plot_ground_tracks(
        df_sched      = df_sched,
        metadata_list = metadata_list,
        obs_indices   = obs_indices,
        frame_types   = frame_types,
        lat_band_deg  = lat_band_deg,
        switch_at     = switch_at,
        save_path     = gt_path,
    )
    print("done")

    npy_mb = npy_path.stat().st_size / 1e6
    csv_mb = csv_path.stat().st_size / 1e6
    png_kb = png_path.stat().st_size / 1e3
    gt_kb  = gt_path.stat().st_size  / 1e3
    print(f"\nOutput files:")
    print(f"  {npy_path}  ({npy_mb:.1f} MB)")
    print(f"  {csv_path}  ({csv_mb:.1f} MB)")
    print(f"  {png_path}  ({png_kb:.0f} KB)")
    print(f"  {gt_path}  ({gt_kb:.0f} KB)")

    # -----------------------------------------------------------------------
    # Step 6: Verification checks C1–C21
    # -----------------------------------------------------------------------
    print(f"\nVerification checks:")

    def _chk(label: str, passed: bool, detail: str = ""):
        status = "PASS" if passed else f"FAIL — {detail}"
        print(f"  {label}: {status}")
        return passed

    expected_rows = int(duration_s / SCHED_DT_S) + 1
    actual_rows   = len(df_sched)
    c1  = _chk("C1  NB01 sched rows",
                abs(actual_rows - expected_rows) <= 2,
                f"expected {expected_rows}, got {actual_rows}")

    valid_parities = {"along_track", "cross_track"}
    c2  = _chk("C2  orbit_parity values",
                all(m.orbit_parity in valid_parities for m in metadata_list),
                "invalid parity found")

    att_q = np.array([m.attitude_quaternion for m in metadata_list])
    c3  = _chk("C3  No NaN in att_q_*",
                not np.any(np.isnan(att_q)),
                "NaN found in attitude quaternions")

    att_norms   = np.linalg.norm(att_q, axis=1)
    max_att_dev = float(np.max(np.abs(att_norms - 1.0)))
    c4  = _chk("C4  att_q unit norm",
                max_att_dev < 1e-6,
                f"max deviation {max_att_dev:.2e}")

    pe_q      = np.array([m.pointing_error for m in metadata_list])
    pe_norms  = np.linalg.norm(pe_q, axis=1)
    max_pe_dev = float(np.max(np.abs(pe_norms - 1.0)))
    c5  = _chk("C5  pointing_error unit norm",
                max_pe_dev < 1e-6,
                f"max deviation {max_pe_dev:.2e}")

    pe_expected_std_v = SIGMA_POINTING_ARCSEC * np.sqrt(1.0 - 2.0 / np.pi)
    pe_std_dev = float(abs(pe_arr.std() - pe_expected_std_v) / pe_expected_std_v)
    c6  = _chk("C6  PE std within 20% of expected half-normal",
                pe_std_dev < 0.20,
                f"std={pe_arr.std():.2f} arcsec, expected {pe_expected_std_v:.2f} "
                f"({pe_std_dev*100:.1f}% off)")

    et_mean_dev = float(abs(et_all.mean() - ETALON_TEMP_MEAN_C))
    c7  = _chk("C7  Etalon temp mean within 0.05°C",
                et_mean_dev < 0.05,
                f"mean={et_all.mean():.3f}°C")

    frac_good = n_good / max(n_obs, 1)
    c8  = _chk("C8  ADCS GOOD > 99.9%",
                frac_good > 0.999,
                f"{frac_good*100:.2f}% good")

    valid_img = {"science", "cal", "dark"}
    c9  = _chk("C9  img_type values valid",
                all(m.img_type in valid_img for m in metadata_list),
                "unexpected img_type found")

    exp_cal  = n_caldark * n_complete_orbits
    c10_dev  = abs(n_cal - exp_cal) / max(exp_cal, 1)
    c10 = _chk(f"C10 Cal count within 5% of expected ({exp_cal})",
               c10_dev <= 0.05,
               f"got {n_cal} ({c10_dev*100:.1f}% off)")

    exp_dark = n_caldark * n_complete_orbits
    c11_dev  = abs(n_dark - exp_dark) / max(exp_dark, 1)
    c11 = _chk(f"C11 Dark count within 5% of expected ({exp_dark})",
               c11_dev <= 0.05,
               f"got {n_dark} ({c11_dev*100:.1f}% off)")

    c12 = _chk("C12 CSV has exactly 47 columns",
               len(df_csv.columns) == 47,
               f"got {len(df_csv.columns)}")

    try:
        loaded = np.load(str(npy_path), allow_pickle=True)
        ImageMetadata(**loaded[0])
        c13 = _chk("C13 P01 round-trip", True)
    except Exception as exc:
        c13 = _chk("C13 P01 round-trip", False, str(exc))

    # Derive masks from the obs_type column (CSV now contains non-obs rows too)
    obs_type_col  = df_csv["obs_type"].values
    sci_mask      = obs_type_col == "science"
    cal_dark_mask = np.isin(obs_type_col, ["cal", "dark"])

    tp_col = df_csv["tp_lat_deg"].values

    c14 = _chk("C14 tp_lat_deg NaN for all cal/dark rows",
               bool(np.all(np.isnan(tp_col[cal_dark_mask]))),
               f"{np.sum(~np.isnan(tp_col[cal_dark_mask]))} non-NaN cal/dark rows found")

    c15 = _chk("C15 tp_lat_deg non-NaN for all science rows",
               bool(np.all(~np.isnan(tp_col[sci_mask]))),
               f"{np.sum(np.isnan(tp_col[sci_mask]))} NaN science rows found")

    if sci_mask.any():
        max_tp_lat = float(np.nanmax(np.abs(tp_col[sci_mask])))
        c16 = _chk(f"C16 tp_lat_deg within lat_band+5° ({lat_band_deg+5:.1f}°) of equator",
                   max_tp_lat <= lat_band_deg + 5.0,
                   f"max |tp_lat|={max_tp_lat:.2f}°")
    else:
        c16 = _chk("C16 tp_lat_deg within lat_band+5° of equator", True)

    wv_col = df_csv["wind_v_zonal_ms"].values
    c17 = _chk("C17 wind_v_zonal_ms non-NaN for all science rows",
               bool(np.all(~np.isnan(wv_col[sci_mask]))),
               f"{np.sum(np.isnan(wv_col[sci_mask]))} NaN science rows found")

    vrel_col = df_csv["v_rel_ms"].values
    c18 = _chk("C18 v_rel_ms non-NaN for all science rows",
               bool(np.all(~np.isnan(vrel_col[sci_mask]))),
               f"{np.sum(np.isnan(vrel_col[sci_mask]))} NaN science rows found")

    c19 = _chk("C19 v_rel_ms NaN for all cal/dark rows",
               bool(np.all(np.isnan(vrel_col[cal_dark_mask]))),
               f"{np.sum(~np.isnan(vrel_col[cal_dark_mask]))} non-NaN cal/dark rows found")

    bin_files = list(bin_dir.glob("*.bin"))
    sizes_ok  = all(f.stat().st_size == 143_520 for f in bin_files)
    count_ok  = len(bin_files) == len(metadata_list)
    c20 = _chk("C20 All .bin files exist and are 143520 bytes",
               sizes_ok and count_ok,
               f"count={len(bin_files)} (expected {len(metadata_list)}), "
               f"all_correct_size={sizes_ok}")

    try:
        first_sci_bin = next(f for f in sorted(bin_dir.glob("*.bin"))
                             if "science" in f.name)
        raw = np.frombuffer(first_sci_bin.read_bytes(), dtype=">u2")
        hdr = raw[:276]
        from src.metadata.p01_image_metadata_2026_04_06 import parse_header
        d = parse_header(hdr)
        sci_csv = df_csv[df_csv["obs_type"] == "science"].iloc[0]
        c21 = _chk("C21 Header round-trip lua_timestamp matches CSV",
                   d["lua_timestamp"] == int(sci_csv["lua_timestamp"]),
                   f"header={d['lua_timestamp']}, csv={int(sci_csv['lua_timestamp'])}")
    except Exception as exc:
        c21 = _chk("C21 Header round-trip lua_timestamp matches CSV", False, str(exc))

    all_pass = all([c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13,
                    c14, c15, c16, c17, c18, c19, c20, c21])
    print(f"\n  {'All checks PASS.' if all_pass else 'Some checks FAILED — see above.'}")

    print("\nG01 complete.")


if __name__ == "__main__":
    main()
