"""
G01 — Synthetic Metadata Generator.

Spec:      docs/specs/G01_synthetic_metadata_generator_2026-04-16.md
Spec date: 2026-04-16
Generated: 2026-04-17
Tool:      Claude Code
CONOPS:    [TBD — insert document ID and version when known]
Usage:     python validation/gen01_synthetic_metadata_generator_2026_04_16.py
"""

import sys
import pathlib
import dataclasses
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
from src.metadata.p01_image_metadata_2026_04_06 import (
    ImageMetadata,
    AdcsQualityFlags,
    compute_adcs_quality_flag,
)
from src.constants import WGS84_A_M, EARTH_GRAV_PARAM_M3_S2

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHED_DT_S            = 10.0   # NB01 propagation cadence — always fixed
CAL_TRIGGER_LAT_DEG   = 60.0   # CONOPS ascending trigger — see spec header
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
    from src.windmap.nb00_wind_map_2026_04_06 import UniformWindMap
    return UniformWindMap(v_zonal_ms=v_zonal_ms, v_merid_ms=v_merid_ms)


def _build_analytic_sine(rng, h_target_km, A_zonal_ms=200.0, A_merid_ms=100.0):
    from src.windmap.nb00_wind_map_2026_04_06 import AnalyticWindMap
    return AnalyticWindMap(pattern="sine_lat",
                           A_zonal_ms=A_zonal_ms, A_merid_ms=A_merid_ms)


def _build_analytic_wave4(rng, h_target_km,
                          A_zonal_ms=150.0, A_merid_ms=75.0, phase_rad=0.0):
    from src.windmap.nb00_wind_map_2026_04_06 import AnalyticWindMap
    return AnalyticWindMap(pattern="wave4", A_zonal_ms=A_zonal_ms,
                           A_merid_ms=A_merid_ms, phase_rad=phase_rad)


def _build_hwm14(rng, h_target_km, day_of_year=172, ut_hours=12.0,
                 f107=150.0, ap=4.0):
    from src.windmap.nb00_wind_map_2026_04_06 import HWM14WindMap
    return HWM14WindMap(alt_km=h_target_km, day_of_year=int(day_of_year),
                        ut_hours=ut_hours, f107=f107, ap=float(ap))


def _build_storm(rng, h_target_km, day_of_year=355, ut_hours=3.0,
                 f107=180.0, ap=80.0):
    from src.windmap.nb00_wind_map_2026_04_06 import StormWindMap
    return StormWindMap(alt_km=h_target_km, day_of_year=int(day_of_year),
                        ut_hours=ut_hours, f107=f107, ap=float(ap))


# ---------------------------------------------------------------------------
# Wind map registry  (§3)
# ---------------------------------------------------------------------------

# Extend here to add new wind map types.
# builder signature: (rng, h_target_km, **user_params) -> WindMap
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

    # Science indices (§3.2): one per step from each band-entry point
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

    # Cal/dark trigger detection (§3.3): ascending crossing of 60°N
    cal_trigger_indices = []
    for i in range(1, n):
        if (lat[i]     >  CAL_TRIGGER_LAT_DEG
                and lat[i - 1] <= CAL_TRIGGER_LAT_DEG
                and lat[i]     >  lat[i - 1]):
            cal_trigger_indices.append(i)

    # Build cal and dark index lists from each trigger
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

    # Cal/dark takes precedence over science
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
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== G01 — WindCube Synthetic Metadata Generator ===\n")

    t_start       = _prompt(
        "Start epoch          [2027-01-01T00:00:00 UTC]  : ",
        "2027-01-01T00:00:00", str)
    duration_days = _prompt(
        "Duration             [days,  default  30       ] : ",
        30.0, float, 0.1, 365.0)
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

    # exp_time in centiseconds (P01 unit)
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

    # Folder picker dialog for output directory
    _default_out = str(pathlib.Path.home() / "WindCube" / "G01_outputs")
    print("\nSelect output folder (dialog opening — check taskbar if not visible)...")
    output_dir = _pick_folder(
        title="Select output folder for G01 files",
        default=_default_out,
    )
    print(f"  Output directory : {output_dir}")

    # Constraint warnings
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

    # Derived parameters
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
    df_sched["look_mode"]    = df_sched["orbit_number"].apply(
        lambda n: "along_track" if n % 2 == 1 else "cross_track"
    )

    print("done")

    # -----------------------------------------------------------------------
    # Step 2: Build observation schedule
    # -----------------------------------------------------------------------
    obs_indices, frame_types, cal_trigger_indices = _build_schedule(
        df_sched, lat_band_deg, n_caldark, step
    )
    n_obs            = len(obs_indices)
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

    # Stop condition: check for zero science or zero cal
    if n_science == 0:
        print("\nFATAL: zero science frames produced. Check lat_band_deg and schedule.")
        return
    if n_cal == 0:
        print("\nFATAL: zero cal frames produced. Check CAL_TRIGGER_LAT_DEG and orbit propagation.")
        return

    # frame_sequence: 0-based within each orbit's observation list
    orbit_counter   = {}
    frame_sequences = []
    for idx in obs_indices:
        orb = int(df_sched.loc[idx, "orbit_number"])
        cnt = orbit_counter.get(orb, 0)
        frame_sequences.append(cnt)
        orbit_counter[orb] = cnt + 1

    # -----------------------------------------------------------------------
    # Step 3: Build ImageMetadata list
    # -----------------------------------------------------------------------
    print(
        "\nBuilding NB02a attitude quaternions + NB02b tangent points "
        "+ NB00 wind sampling ..."
    )

    rng           = np.random.default_rng(rng_seed)
    metadata_list = []

    for i, (idx, frame_type) in enumerate(zip(obs_indices, frame_types)):
        row = df_sched.loc[idx]

        pos       = np.array([row.pos_eci_x, row.pos_eci_y, row.pos_eci_z])
        vel       = np.array([row.vel_eci_x, row.vel_eci_y, row.vel_eci_z])
        look_mode = str(row.look_mode)

        # NB02a: ideal attitude quaternion + LOS vector (los_eci retained for NB02b)
        los_eci, q = compute_los_eci(
            pos, vel, look_mode,
            altitude_km = altitude_km,
            h_target_km = h_target_km,
        )

        # RNG draw order: PE (4 draws), etalon temps (4 draws), CCD temp (1 draw)
        pointing_error = _pointing_error_quat(rng, SIGMA_POINTING_ARCSEC)
        etalon_temps   = rng.normal(ETALON_TEMP_MEAN_C, ETALON_TEMP_STD_C, size=4).tolist()
        ccd_temp1      = float(rng.normal(CCD_TEMP_MEAN_C, CCD_TEMP_STD_C))

        # NB02b + NB00: tangent point and truth wind (science frames only)
        tp_lat = tp_lon = tp_alt = v_zonal = v_merid = None
        if frame_type == "science":
            epoch_t = Time(row.epoch.to_pydatetime(), scale="utc")
            tp = compute_tangent_point(pos, los_eci, epoch_t, h_target_km=h_target_km)
            tp_lat  = tp["tp_lat_deg"]
            tp_lon  = tp["tp_lon_deg"]
            tp_alt  = tp["tp_alt_km"]
            v_zonal, v_merid = wind_map.sample(tp_lat, tp_lon)

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
        orbit_parity = "along_track" if orbit_num % 2 == 1 else "cross_track"

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
        )
        metadata_list.append(meta)

        if i % max(1, n_obs // 100) == 0 or i == n_obs - 1:
            pct = 100 * (i + 1) / n_obs
            bar = "=" * int(pct // 5) + " " * (20 - int(pct // 5))
            print(f"\r  [{bar}] {i+1}/{n_obs}", end="", flush=True)

    print()
    print(f"  (NB02b + NB00 called for {n_science} science frames only)")

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

    # Pointing error angle stats
    pe_angles = []
    for m in metadata_list:
        qe       = m.pointing_error
        vn       = min(np.sqrt(qe[0]**2 + qe[1]**2 + qe[2]**2), 1.0)
        angle_as = 2.0 * np.degrees(np.arcsin(vn)) * 3600.0
        pe_angles.append(angle_as)
    pe_arr = np.array(pe_angles)
    # For unsigned rotation angle drawn from |N(0,σ)|: mean = σ√(2/π), std = σ√(1−2/π)
    pe_expected_mean = SIGMA_POINTING_ARCSEC * np.sqrt(2.0 / np.pi)
    pe_expected_std  = SIGMA_POINTING_ARCSEC * np.sqrt(1 - 2.0 / np.pi)
    print(f"\nPointing error stats (arcsec):")
    print(f"  Mean  : {pe_arr.mean():6.2f}   (expected ~{pe_expected_mean:.2f}, half-normal)")
    print(f"  Std   : {pe_arr.std():6.2f}   (expected ~{pe_expected_std:.2f}, half-normal)")
    print(f"  Max   : {pe_arr.max():6.1f}")

    # Etalon temperature stats
    et_all = np.array([m.etalon_temps for m in metadata_list]).flatten()
    print(f"\nEtalon temperature stats (°C):")
    print(f"  Mean  : {et_all.mean():.2f}   (expected ~{ETALON_TEMP_MEAN_C:.2f})")
    print(f"  Std   : {et_all.std():.2f}   (expected ~{ETALON_TEMP_STD_C:.2f})")

    # CCD temperature stats
    ccd_all = np.array([m.ccd_temp1 for m in metadata_list])
    print(f"\nCCD temperature stats (°C):")
    print(f"  Mean  : {ccd_all.mean():.2f}   (expected ~{CCD_TEMP_MEAN_C:.2f})")
    print(f"  Std   : {ccd_all.std():.2f}   (expected ~{CCD_TEMP_STD_C:.2f})")

    # Tangent point / wind stats (science frames only)
    sci_meta = [m for m in metadata_list if m.img_type == "science"]
    if sci_meta:
        tp_lats   = np.array([m.tangent_lat for m in sci_meta])
        vz_all    = np.array([m.truth_v_zonal for m in sci_meta])
        vm_all    = np.array([m.truth_v_meridional for m in sci_meta])
        print(f"\nTangent point stats (science frames):")
        print(f"  tp_lat  : mean={tp_lats.mean():.1f}°, "
              f"min={tp_lats.min():.1f}°, max={tp_lats.max():.1f}°")
        print(f"\nWind stats at tangent point (science frames):")
        print(f"  v_zonal : mean={vz_all.mean():.1f}, "
              f"min={vz_all.min():.1f}, max={vz_all.max():.1f} m/s")
        print(f"  v_merid : mean={vm_all.mean():.1f}, "
              f"min={vm_all.min():.1f}, max={vm_all.max():.1f} m/s")

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

    # CSV: 17 binary-header fields (38 scalar columns) + 4 new tangent/wind columns
    # = 42 columns total. Cal/dark rows have NaN for the 4 new columns.
    rows_csv = []
    for r in records:
        rows_csv.append({
            # ── pixels 0-3: geometry / timing scalars ──────────────────────
            "rows":                  r["rows"],
            "cols":                  r["cols"],
            "exp_time":              r["exp_time"],
            "exp_unit":              r["exp_unit"],
            # ── pixels 4-7: CCD temperature ────────────────────────────────
            "ccd_temp1":             r["ccd_temp1"],
            # ── pixels 8-15: timestamps ─────────────────────────────────────
            "lua_timestamp":         r["lua_timestamp"],
            "adcs_timestamp":        r["adcs_timestamp"],
            # ── pixels 16-27: spacecraft geodetic state ─────────────────────
            "spacecraft_latitude":   r["spacecraft_latitude"],
            "spacecraft_longitude":  r["spacecraft_longitude"],
            "spacecraft_altitude":   r["spacecraft_altitude"],
            # ── pixels 28-43: attitude quaternion [x, y, z, w] ─────────────
            "att_q_x":  r["attitude_quaternion"][0],
            "att_q_y":  r["attitude_quaternion"][1],
            "att_q_z":  r["attitude_quaternion"][2],
            "att_q_w":  r["attitude_quaternion"][3],
            # ── pixels 44-59: pointing error quaternion [x, y, z, w] ────────
            "pe_q_x":   r["pointing_error"][0],
            "pe_q_y":   r["pointing_error"][1],
            "pe_q_z":   r["pointing_error"][2],
            "pe_q_w":   r["pointing_error"][3],
            # ── pixels 60-71: ECI position [x, y, z] m ──────────────────────
            "pos_eci_x": r["pos_eci_hat"][0],
            "pos_eci_y": r["pos_eci_hat"][1],
            "pos_eci_z": r["pos_eci_hat"][2],
            # ── pixels 72-83: ECI velocity [vx, vy, vz] m/s ─────────────────
            "vel_eci_x": r["vel_eci_hat"][0],
            "vel_eci_y": r["vel_eci_hat"][1],
            "vel_eci_z": r["vel_eci_hat"][2],
            # ── pixels 84-99: etalon temperatures [T0-T3] °C ────────────────
            "etalon_t0": r["etalon_temps"][0],
            "etalon_t1": r["etalon_temps"][1],
            "etalon_t2": r["etalon_temps"][2],
            "etalon_t3": r["etalon_temps"][3],
            # ── pixels 100-103: GPIO power register ──────────────────────────
            "gpio_0": r["gpio_pwr_on"][0],
            "gpio_1": r["gpio_pwr_on"][1],
            "gpio_2": r["gpio_pwr_on"][2],
            "gpio_3": r["gpio_pwr_on"][3],
            # ── pixels 104-109: lamp channel states ──────────────────────────
            "lamp_0": r["lamp_ch_array"][0],
            "lamp_1": r["lamp_ch_array"][1],
            "lamp_2": r["lamp_ch_array"][2],
            "lamp_3": r["lamp_ch_array"][3],
            "lamp_4": r["lamp_ch_array"][4],
            "lamp_5": r["lamp_ch_array"][5],
            # ── new v8: tangent point + truth wind (NaN for cal/dark) ────────
            "tp_lat_deg":      r["tangent_lat"]       if r["tangent_lat"]       is not None else float("nan"),
            "tp_lon_deg":      r["tangent_lon"]        if r["tangent_lon"]        is not None else float("nan"),
            "wind_v_zonal_ms": r["truth_v_zonal"]      if r["truth_v_zonal"]      is not None else float("nan"),
            "wind_v_merid_ms": r["truth_v_meridional"] if r["truth_v_meridional"] is not None else float("nan"),
        })

    df_csv = pd.DataFrame(rows_csv)
    df_csv.to_csv(str(csv_path), index=False)

    npy_mb = npy_path.stat().st_size / 1e6
    csv_mb = csv_path.stat().st_size / 1e6
    print(f"\nOutput files:")
    print(f"  {npy_path}  ({npy_mb:.1f} MB)")
    print(f"  {csv_path}  ({csv_mb:.1f} MB)")

    # -----------------------------------------------------------------------
    # Step 6: Verification checks C1–C17
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

    att_norms = np.linalg.norm(att_q, axis=1)
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

    # Rotation angle magnitude follows half-normal: std = σ·√(1−2/π) ≈ 3.01 arcsec
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

    # n_complete_orbits = len(cal_trigger_indices): actual detected triggers
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

    c12 = _chk("C12 CSV has exactly 42 columns",
               len(df_csv.columns) == 42,
               f"got {len(df_csv.columns)}")

    try:
        loaded = np.load(str(npy_path), allow_pickle=True)
        ImageMetadata(**loaded[0])
        c13 = _chk("C13 P01 round-trip", True)
    except Exception as exc:
        c13 = _chk("C13 P01 round-trip", False, str(exc))

    # C14–C17: tangent point / wind checks using the CSV
    cal_dark_mask = df_csv["gpio_0"].values  # not perfect but use img_type from npy
    # Use img_type from metadata_list for exact mask
    img_type_arr = np.array([m.img_type for m in metadata_list])
    sci_mask  = img_type_arr == "science"
    nons_mask = ~sci_mask

    tp_col = df_csv["tp_lat_deg"].values

    c14 = _chk("C14 tp_lat_deg NaN for all cal/dark rows",
               bool(np.all(np.isnan(tp_col[nons_mask]))),
               f"{np.sum(~np.isnan(tp_col[nons_mask]))} non-NaN cal/dark rows found")

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

    all_pass = all([c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13,
                    c14, c15, c16, c17])
    print(f"\n  {'All checks PASS.' if all_pass else 'Some checks FAILED — see above.'}")

    print("\nG01 complete.")


if __name__ == "__main__":
    main()
