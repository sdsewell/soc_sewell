"""
P01 — ImageMetadata dataclass, binary ingest, and Level-0 sidecar JSON.

Spec:        docs/specs/S19_p01_metadata_2026-04-06.md
Spec date:   2026-04-06
Generated:   2026-04-07
Tool:        Claude Code
Last tested: 2026-04-07  (8/8 tests pass)
Depends on:  src.constants, src.geometry (NB01/NB02 outputs),
             src.fpi.m01 (InstrumentParams)
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Reference constant — nominal etalon temperature for synthetic images
# ---------------------------------------------------------------------------

_T_REF_ETALON_C: float = -5.0   # °C — nominal etalon temperature (synthetic)


# ---------------------------------------------------------------------------
# AdcsQualityFlags bitmask
# ---------------------------------------------------------------------------


class AdcsQualityFlags:
    """
    Bitmask for adcs_quality_flag in ImageMetadata.
    Multiple flags may be set simultaneously.
    """
    GOOD             = 0x00  # No issues detected
    SLEW_IN_PROGRESS = 0x01  # pointing_error magnitude > 30 arcsec RMS
    STR_UNAVAILABLE  = 0x02  # Star tracker not contributing to attitude solution
    GNSS_UNAVAILABLE = 0x04  # GNSS position/velocity invalid at acquisition time
    ADCS_DEGRADED    = 0x08  # Generic ADCS degradation not covered by above flags
    POINTING_UNKNOWN = 0x10  # adcs_quality_flag cannot be determined


# ---------------------------------------------------------------------------
# ImageMetadata dataclass
# ---------------------------------------------------------------------------


@dataclass
class ImageMetadata:
    """
    Complete metadata record for one WindCube FPI image.

    Applies to all image types: dark, cal, science.
    Applies to both real (on-orbit) and synthetic images.
    Synthetic-only fields are None for real images.
    Pipeline-added fields are None at ingest and populated during processing.
    """

    # ── Section 3.1: Image geometry and classification ──────────────────────
    rows:         int    # Total rows including header row (on-orbit: 260)
    cols:         int    # Total columns (on-orbit: 276 for 2×2 binned)
    exp_time:     int    # Exposure duration, centiseconds
    exp_unit:     int    # CCD timing register value
    binning:      int    # 1 = 1×1, 2 = 2×2; derived from cols
    img_type:     str    # 'cal', 'dark', or 'science'

    # ── Section 3.2: Timing ─────────────────────────────────────────────────
    lua_timestamp:   int    # Unix epoch ms, end of exposure (primary reference)
    adcs_timestamp:  int    # ADCS clock Unix ms; 0 if unavailable
    utc_timestamp:   str    # ISO 8601 string derived from lua_timestamp

    # ── Section 3.3: Orbit state ────────────────────────────────────────────
    spacecraft_latitude:   float         # SC geodetic latitude, rad
    spacecraft_longitude:  float         # SC geodetic longitude, rad
    spacecraft_altitude:   float         # SC altitude above WGS84, m
    pos_eci_hat:  list                   # ECI position [x, y, z], m
    vel_eci_hat:  list                   # ECI velocity [vx, vy, vz], m/s

    # ── Section 3.4: Attitude ───────────────────────────────────────────────
    attitude_quaternion:  list           # [x, y, z, w] scalar-last
    pointing_error:       list           # [x, y, z, w] residual attitude error
    obs_mode:             str            # 'along_track', 'cross_track', or 'unknown'

    # ── Section 3.5: Instrument state ───────────────────────────────────────
    ccd_temp1:      float                # CCD temperature, °C
    etalon_temps:   list                 # [T0, T1, T2, T3], °C
    shutter_status: str                  # 'open' or 'closed'
    gpio_pwr_on:    list                 # GPIO register values [ch0–ch3]
    lamp_ch_array:  list                 # Lamp channel states [0–5]
    lamp1_status:   str                  # 'on' or 'off'
    lamp2_status:   str                  # 'on' or 'off'
    lamp3_status:   str                  # 'on' or 'off'

    # ── Section 3.6: Orbit and sequence identification ──────────────────────
    orbit_number:   Optional[int] = None
    frame_sequence: Optional[int] = None
    orbit_parity:   Optional[str] = None  # 'along_track' (odd) or 'cross_track' (even)

    # ── Section 3.7: Pointing quality gate ──────────────────────────────────
    adcs_quality_flag: int = 0

    # ── Section 3.8: Dark subtraction provenance ────────────────────────────
    dark_subtracted:        bool           = False
    dark_n_frames:          int            = 0
    dark_lua_timestamp:     Optional[int]  = None
    dark_etalon_temp_mean:  Optional[float] = None

    # ── Section 3.9: Synthetic fields ───────────────────────────────────────
    is_synthetic:       bool           = False
    truth_v_los:        Optional[float] = None
    truth_v_zonal:      Optional[float] = None
    truth_v_meridional: Optional[float] = None
    tangent_lat:        Optional[float] = None
    tangent_lon:        Optional[float] = None
    tangent_alt_km:     Optional[float] = None
    etalon_gap_mm:      Optional[float] = None
    noise_seed:         Optional[int]   = None

    # ── Section 3.10: Etalon thermal correction hook ────────────────────────
    etalon_gap_corrected_mm: Optional[float] = None

    # ── Section 3.11: Grafana telemetry hook ────────────────────────────────
    grafana_record_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Private binary decode helpers (Section 2.2)
# ---------------------------------------------------------------------------


def _u16(h: np.ndarray, w: int) -> int:
    """Single big-endian uint16 word."""
    return int(h[w])


def _u64(h: np.ndarray, w: int) -> int:
    """4 BE uint16 words in LE word order → uint64."""
    return sum(int(h[w + i]) << (16 * i) for i in range(4))


def _f64(h: np.ndarray, w: int) -> float:
    """4 BE uint16 words in LE word order → float64."""
    b = struct.pack(">4H", *reversed([h[w + i] for i in range(4)]))
    return struct.unpack(">d", b)[0]


def _u8arr(h: np.ndarray, w: int, n: int) -> list:
    """n consecutive uint16 words → list of uint8 from low byte."""
    return [int(h[w + i]) & 0xFF for i in range(n)]


# ---------------------------------------------------------------------------
# Header parser (Section 2)
# ---------------------------------------------------------------------------


def parse_header(h: np.ndarray) -> dict:
    """
    Parse a 276-word big-endian uint16 header row into a metadata dict.

    Performs:
    - All field decodes per the header word map (Section 2.3)
    - Quaternion re-ordering [w,x,y,z] → [x,y,z,w] for both
      attitude_quaternion and pointing_error (Section 2.4)
    - Derived fields: binning, shutter_status, lamp statuses,
      img_type, utc_timestamp (Section 2.5–2.6)
    - Legacy typo normalisation: the output dict always uses
      'attitude_quaternion' (never 'attitude_quadternion')

    Parameters
    ----------
    h : np.ndarray, shape (276,), dtype '>u2'
        Row 0 of the binary file.

    Returns
    -------
    dict
        All parsed and derived fields, keyed by ImageMetadata field names.
    """
    rows     = _u16(h, 0)
    cols     = _u16(h, 1)
    exp_time = _u16(h, 2)
    exp_unit = _u16(h, 3)

    ccd_temp1       = _f64(h, 4)
    lua_timestamp   = _u64(h, 8)
    adcs_timestamp  = _u64(h, 12)

    spacecraft_latitude  = _f64(h, 16)
    spacecraft_longitude = _f64(h, 20)
    spacecraft_altitude  = _f64(h, 24)

    # Quaternions: binary order [w, x, y, z] → pipeline order [x, y, z, w]
    ads_q_wxyz = [_f64(h, 28 + i * 4) for i in range(4)]
    attitude_quaternion = [ads_q_wxyz[1], ads_q_wxyz[2],
                           ads_q_wxyz[3], ads_q_wxyz[0]]  # [x,y,z,w]

    acs_q_wxyz = [_f64(h, 44 + i * 4) for i in range(4)]
    pointing_error = [acs_q_wxyz[1], acs_q_wxyz[2],
                      acs_q_wxyz[3], acs_q_wxyz[0]]       # [x,y,z,w]

    pos_eci_hat = [_f64(h, 60), _f64(h, 64), _f64(h, 68)]
    vel_eci_hat = [_f64(h, 72), _f64(h, 76), _f64(h, 80)]

    etalon_temps = [_f64(h, 84), _f64(h, 88), _f64(h, 92), _f64(h, 96)]

    gpio_pwr_on  = _u8arr(h, 100, 4)
    lamp_ch_array = _u8arr(h, 104, 6)

    # Derived fields
    binning = 1 if cols == 552 else 2

    shutter_closed  = (gpio_pwr_on[0] == 1 and gpio_pwr_on[3] == 1)
    shutter_status  = "closed" if shutter_closed else "open"

    lamp1_status = "on" if (lamp_ch_array[0] or lamp_ch_array[1]) else "off"
    lamp2_status = "on" if (lamp_ch_array[2] or lamp_ch_array[3]) else "off"
    lamp3_status = "on" if (lamp_ch_array[4] or lamp_ch_array[5]) else "off"

    if any(lamp_ch_array):
        img_type = "cal"
    elif shutter_closed:
        img_type = "dark"
    else:
        img_type = "science"

    # utc_timestamp from lua_timestamp (milliseconds → ISO 8601)
    dt = datetime.fromtimestamp(lua_timestamp / 1000.0, tz=timezone.utc)
    utc_timestamp = dt.isoformat()

    return {
        "rows":                 rows,
        "cols":                 cols,
        "exp_time":             exp_time,
        "exp_unit":             exp_unit,
        "binning":              binning,
        "img_type":             img_type,
        "lua_timestamp":        lua_timestamp,
        "adcs_timestamp":       adcs_timestamp,
        "utc_timestamp":        utc_timestamp,
        "spacecraft_latitude":  spacecraft_latitude,
        "spacecraft_longitude": spacecraft_longitude,
        "spacecraft_altitude":  spacecraft_altitude,
        "pos_eci_hat":          pos_eci_hat,
        "vel_eci_hat":          vel_eci_hat,
        "attitude_quaternion":  attitude_quaternion,  # [x,y,z,w]
        "pointing_error":       pointing_error,       # [x,y,z,w]
        "ccd_temp1":            ccd_temp1,
        "etalon_temps":         etalon_temps,
        "shutter_status":       shutter_status,
        "gpio_pwr_on":          gpio_pwr_on,
        "lamp_ch_array":        lamp_ch_array,
        "lamp1_status":         lamp1_status,
        "lamp2_status":         lamp2_status,
        "lamp3_status":         lamp3_status,
    }


# ---------------------------------------------------------------------------
# ADCS quality flag computation (Section 4)
# ---------------------------------------------------------------------------


def compute_adcs_quality_flag(meta: dict) -> int:
    """
    Compute the ADCS quality bitmask from raw parsed metadata.

    Parameters
    ----------
    meta : dict from parse_header() — contains pointing_error as [x,y,z,w]

    Returns
    -------
    int : AdcsQualityFlags bitmask
    """
    flag = AdcsQualityFlags.GOOD

    # Pointing error magnitude — convert quaternion to rotation angle (arcsec)
    qe = meta.get("pointing_error", [0.0, 0.0, 0.0, 1.0])
    vec_norm = np.sqrt(qe[0]**2 + qe[1]**2 + qe[2]**2)
    vec_norm = min(vec_norm, 1.0)   # clamp for numerical safety
    angle_arcsec = 2.0 * np.degrees(np.arcsin(vec_norm)) * 3600.0

    AKE_BUDGET_ARCSEC = 30.0   # SYS.108 from SI-UCAR-WC-RP-004
    if angle_arcsec > AKE_BUDGET_ARCSEC:
        flag |= AdcsQualityFlags.SLEW_IN_PROGRESS

    # GNSS availability — pos_eci_hat == [0,0,0] indicates no valid fix
    pos = meta.get("pos_eci_hat", [1.0, 0.0, 0.0])
    if all(p == 0.0 for p in pos):
        flag |= AdcsQualityFlags.GNSS_UNAVAILABLE

    # adcs_timestamp == 0 means ADCS telemetry was not available
    if meta.get("adcs_timestamp", 0) == 0:
        flag |= AdcsQualityFlags.ADCS_DEGRADED

    return flag


# ---------------------------------------------------------------------------
# Private helper: epoch to Unix milliseconds
# ---------------------------------------------------------------------------


def _epoch_to_unix_ms(epoch) -> int:
    """
    Convert an epoch value to Unix milliseconds integer.
    Accepts: astropy Time, pandas Timestamp, or numeric (already Unix ms).
    """
    try:
        # astropy Time
        return int(epoch.unix * 1000)
    except AttributeError:
        pass
    try:
        # pandas Timestamp or datetime
        return int(epoch.timestamp() * 1000)
    except AttributeError:
        pass
    # Numeric fallback (already ms)
    return int(epoch)


# ---------------------------------------------------------------------------
# Real image ingest (Section 5.1)
# ---------------------------------------------------------------------------


def ingest_real_image(
    path,
    obs_mode: str = "unknown",
    orbit_number: Optional[int] = None,
    frame_sequence: Optional[int] = None,
) -> tuple:
    """
    Load a WindCube FPI binary image and parse its metadata header.

    Parameters
    ----------
    path : str or Path
        Path to the `*_swapped.bin` file.
    obs_mode : str
        'along_track', 'cross_track', or 'unknown'.
    orbit_number : int, optional
        Absolute orbit counter.
    frame_sequence : int, optional
        Frame index within this orbit.

    Returns
    -------
    (ImageMetadata, np.ndarray)
        The metadata record and the pixel array (259 × 276, uint16).

    Raises
    ------
    ValueError
        If file size != 143,520 bytes.
    FileNotFoundError
        If path does not exist.
    """
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    raw_bytes = path.read_bytes()
    expected = 260 * 276 * 2  # 143,520 bytes
    if len(raw_bytes) != expected:
        raise ValueError(
            f"File size mismatch: expected {expected} bytes, "
            f"got {len(raw_bytes)} bytes: {path}"
        )

    # Parse as big-endian uint16
    raw = np.frombuffer(raw_bytes, dtype=">u2")

    header = raw[:276]
    image  = raw[276:].reshape(259, 276)

    d = parse_header(header)

    # Legacy typo normalisation (just in case)
    if "attitude_quadternion" in d:
        d["attitude_quaternion"] = d.pop("attitude_quadternion")

    # Derive orbit_parity from orbit_number
    orbit_parity = None
    if orbit_number is not None:
        orbit_parity = "along_track" if (orbit_number % 2 == 1) else "cross_track"

    adcs_quality_flag = compute_adcs_quality_flag(d)

    meta = ImageMetadata(
        rows                = d["rows"],
        cols                = d["cols"],
        exp_time            = d["exp_time"],
        exp_unit            = d["exp_unit"],
        binning             = d["binning"],
        img_type            = d["img_type"],
        lua_timestamp       = d["lua_timestamp"],
        adcs_timestamp      = d["adcs_timestamp"],
        utc_timestamp       = d["utc_timestamp"],
        spacecraft_latitude  = d["spacecraft_latitude"],
        spacecraft_longitude = d["spacecraft_longitude"],
        spacecraft_altitude  = d["spacecraft_altitude"],
        pos_eci_hat         = d["pos_eci_hat"],
        vel_eci_hat         = d["vel_eci_hat"],
        attitude_quaternion = d["attitude_quaternion"],
        pointing_error      = d["pointing_error"],
        obs_mode            = obs_mode,
        ccd_temp1           = d["ccd_temp1"],
        etalon_temps        = d["etalon_temps"],
        shutter_status      = d["shutter_status"],
        gpio_pwr_on         = d["gpio_pwr_on"],
        lamp_ch_array       = d["lamp_ch_array"],
        lamp1_status        = d["lamp1_status"],
        lamp2_status        = d["lamp2_status"],
        lamp3_status        = d["lamp3_status"],
        orbit_number        = orbit_number,
        frame_sequence      = frame_sequence,
        orbit_parity        = orbit_parity,
        adcs_quality_flag   = adcs_quality_flag,
        # All synthetic fields → None / defaults
        is_synthetic        = False,
    )
    return meta, image


# ---------------------------------------------------------------------------
# Synthetic metadata construction (Section 5.2)
# ---------------------------------------------------------------------------


def build_synthetic_metadata(
    params,
    nb01_row,
    nb02_tp: dict,
    nb02_vr: dict,
    quaternion_xyzw: list,
    los_eci,
    look_mode: str,
    img_type: str,
    orbit_number: Optional[int] = None,
    frame_sequence: Optional[int] = None,
    noise_seed: Optional[int] = None,
    snr: float = 5.0,
) -> ImageMetadata:
    """
    Construct an ImageMetadata for a synthetic image from geometry pipeline outputs.

    Parameters
    ----------
    params : InstrumentParams
        Instrument configuration. Provides etalon gap (params.t, metres).
    nb01_row : pd.Series
        One row from NB01's propagate_orbit() output. Provides pos_eci,
        vel_eci, epoch, sc_lat (deg), sc_lon (deg), sc_alt_km.
    nb02_tp : dict
        From compute_tangent_point(). Keys: tp_lat_deg, tp_lon_deg, tp_alt_km.
    nb02_vr : dict
        From compute_v_rel(). Keys: v_wind_LOS, v_zonal_ms, v_merid_ms.
    quaternion_xyzw : list[float]
        Body-to-ECI quaternion [x,y,z,w].
    los_eci : array-like, shape (3,)
        Unit LOS vector in ECI.
    look_mode : str
        'along_track' or 'cross_track'.
    img_type : str
        'cal', 'dark', or 'science'.
    orbit_number : int, optional
    frame_sequence : int, optional
    noise_seed : int, optional
    snr : float
        Signal-to-noise ratio (stored for provenance via noise_seed).

    Returns
    -------
    ImageMetadata
        Fully populated record. is_synthetic=True.
    """
    lua_timestamp = _epoch_to_unix_ms(nb01_row.epoch)
    dt = datetime.fromtimestamp(lua_timestamp / 1000.0, tz=timezone.utc)
    utc_timestamp = dt.isoformat()

    # Spacecraft position and velocity
    pos_eci_hat = [float(nb01_row.pos_eci_x),
                   float(nb01_row.pos_eci_y),
                   float(nb01_row.pos_eci_z)]
    vel_eci_hat = [float(nb01_row.vel_eci_x),
                   float(nb01_row.vel_eci_y),
                   float(nb01_row.vel_eci_z)]

    # Spacecraft geodetic position (sc_lat/sc_lon in degrees → radians)
    spacecraft_latitude  = float(np.radians(float(nb01_row.sc_lat)))
    spacecraft_longitude = float(np.radians(float(nb01_row.sc_lon)))
    spacecraft_altitude  = float(nb01_row.sc_alt_km) * 1000.0  # km → m

    # Orbit parity
    orbit_parity = None
    if orbit_number is not None:
        orbit_parity = "along_track" if (orbit_number % 2 == 1) else "cross_track"

    # Etalon gap
    etalon_gap_mm = float(params.t) * 1000.0   # metres → mm

    # Nominal synthetic instrument state
    etalon_temps = [_T_REF_ETALON_C] * 4

    # Lamp / shutter state from img_type
    if img_type == "cal":
        lamp_ch_array = [1, 0, 0, 0, 0, 0]
        lamp1_status  = "on"
        gpio_pwr_on   = [0, 0, 0, 0]
        shutter_status = "open"
    elif img_type == "dark":
        lamp_ch_array = [0, 0, 0, 0, 0, 0]
        lamp1_status  = "off"
        gpio_pwr_on   = [1, 0, 0, 1]   # shutter closed
        shutter_status = "closed"
    else:  # science
        lamp_ch_array = [0, 0, 0, 0, 0, 0]
        lamp1_status  = "off"
        gpio_pwr_on   = [0, 0, 0, 0]
        shutter_status = "open"

    return ImageMetadata(
        rows                = 256,
        cols                = 256,
        exp_time            = 500,
        exp_unit            = 1,
        binning             = 2,
        img_type            = img_type,
        lua_timestamp       = lua_timestamp,
        adcs_timestamp      = lua_timestamp,  # no clock jitter in simulation
        utc_timestamp       = utc_timestamp,
        spacecraft_latitude  = spacecraft_latitude,
        spacecraft_longitude = spacecraft_longitude,
        spacecraft_altitude  = spacecraft_altitude,
        pos_eci_hat         = pos_eci_hat,
        vel_eci_hat         = vel_eci_hat,
        attitude_quaternion = list(quaternion_xyzw),
        pointing_error      = [0.0, 0.0, 0.0, 1.0],   # identity = perfect pointing
        obs_mode            = look_mode,
        ccd_temp1           = -18.0,
        etalon_temps        = etalon_temps,
        shutter_status      = shutter_status,
        gpio_pwr_on         = gpio_pwr_on,
        lamp_ch_array       = lamp_ch_array,
        lamp1_status        = lamp1_status,
        lamp2_status        = "off",
        lamp3_status        = "off",
        orbit_number        = orbit_number,
        frame_sequence      = frame_sequence,
        orbit_parity        = orbit_parity,
        adcs_quality_flag   = AdcsQualityFlags.GOOD,
        is_synthetic        = True,
        truth_v_los         = float(nb02_vr["v_wind_LOS"]),
        truth_v_zonal       = float(nb02_vr["v_zonal_ms"]),
        truth_v_meridional  = float(nb02_vr["v_merid_ms"]),
        tangent_lat         = float(nb02_tp["tp_lat_deg"]),
        tangent_lon         = float(nb02_tp["tp_lon_deg"]),
        tangent_alt_km      = float(nb02_tp["tp_alt_km"]),
        etalon_gap_mm       = etalon_gap_mm,
        noise_seed          = noise_seed,
    )


# ---------------------------------------------------------------------------
# JSON sidecar I/O (Section 6.3)
# ---------------------------------------------------------------------------


def write_sidecar(meta: ImageMetadata, path) -> None:
    """Write ImageMetadata to a Level-0 JSON sidecar file (UTF-8, indent=2)."""
    d = dataclasses.asdict(meta)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)


def read_sidecar(path) -> ImageMetadata:
    """
    Read a Level-0 JSON sidecar and return an ImageMetadata.
    Applies legacy typo normalisation automatically.
    """
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    # Legacy normalisation
    if "attitude_quadternion" in d:
        d["attitude_quaternion"] = d.pop("attitude_quadternion")
    return ImageMetadata(**d)
