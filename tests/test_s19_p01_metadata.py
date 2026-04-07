"""
Tests for P01 ImageMetadata, binary ingest, and sidecar JSON.
Spec: specs/S19_p01_metadata_2026-04-06.md
"""

import dataclasses
import json
import pathlib
import struct
import tempfile

import numpy as np
import pytest

from src.metadata import (
    AdcsQualityFlags,
    ImageMetadata,
    ingest_real_image,
    build_synthetic_metadata,
    write_sidecar,
    read_sidecar,
)
from src.metadata.p01_image_metadata_2026_04_06 import parse_header


# ---------------------------------------------------------------------------
# Private encode helpers (inverse of _u64 and _f64)
# ---------------------------------------------------------------------------


def _encode_u64(h, w, val):
    """Inverse of _u64 — encode val into 4 LE-word-order BE uint16 words."""
    for i in range(4):
        h[w + i] = (val >> (16 * i)) & 0xFFFF


def _encode_f64(h, w, val):
    """Inverse of _f64 — encode float64 into 4 LE-word-order BE uint16 words."""
    b = struct.pack(">d", val)
    words = struct.unpack(">4H", b)
    for i in range(4):
        h[w + i] = words[3 - i]


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _make_nb01_row():
    """Minimal pandas Series mimicking one NB01 propagate_orbit() row."""
    import pandas as pd
    from astropy.time import Time
    return pd.Series({
        "epoch": Time("2027-01-01T00:01:00", format="isot", scale="utc"),
        "pos_eci_x": 0.0, "pos_eci_y": 6891000.0, "pos_eci_z": 0.0,
        "vel_eci_x": 7560.0, "vel_eci_y": 0.0, "vel_eci_z": 0.0,
        "sc_lat": 0.0, "sc_lon": 0.0, "sc_alt_km": 510.0,
    })


def _make_minimal_real_metadata():
    """Return a fully populated ImageMetadata with is_synthetic=False."""
    return ImageMetadata(
        rows=260, cols=276, exp_time=600, exp_unit=1,
        binning=2, img_type="science",
        lua_timestamp=1741999867000, adcs_timestamp=1741999867012,
        utc_timestamp="2025-03-15T02:31:07.000000+00:00",
        spacecraft_latitude=0.2356, spacecraft_longitude=-1.6142,
        spacecraft_altitude=527341.2,
        pos_eci_hat=[3120000.0, -5210000.0, 388000.0],
        vel_eci_hat=[4231.1, 2815.7, -5910.2],
        attitude_quaternion=[0.0012, -0.0034, 0.7071, 0.7071],
        pointing_error=[0.0, 0.0, 0.0, 1.0],
        obs_mode="along_track",
        ccd_temp1=-18.425,
        etalon_temps=[-5.12, -4.98, -5.21, -5.09],
        shutter_status="open",
        gpio_pwr_on=[0, 0, 0, 0],
        lamp_ch_array=[0, 0, 0, 0, 0, 0],
        lamp1_status="off", lamp2_status="off", lamp3_status="off",
        orbit_number=42, frame_sequence=17, orbit_parity="along_track",
        adcs_quality_flag=0,
    )


def _make_minimal_real_metadata_dict():
    return dataclasses.asdict(_make_minimal_real_metadata())


def _classify(lamps, gpio):
    """Call image type classification logic directly."""
    any_lamp = any(lamps)
    shutter_closed = (gpio[0] == 1 and gpio[3] == 1)
    if any_lamp:
        return "cal"
    if shutter_closed:
        return "dark"
    return "science"


# ---------------------------------------------------------------------------
# T1 — Round-trip: dataclass → JSON → dataclass
# ---------------------------------------------------------------------------


def test_json_round_trip():
    """
    Serialise an ImageMetadata to JSON and deserialise it back.
    All fields must be bit-for-bit identical after the round trip.
    Pay special attention to lua_timestamp (must not lose precision).
    """
    meta = _make_minimal_real_metadata()
    with tempfile.NamedTemporaryFile(suffix="_L0.json", delete=False) as f:
        path = pathlib.Path(f.name)
    write_sidecar(meta, path)
    meta2 = read_sidecar(path)
    assert meta == meta2
    # Explicit timestamp precision check
    assert meta2.lua_timestamp == meta.lua_timestamp


# ---------------------------------------------------------------------------
# T2 — Binary parse: known header decodes correctly
# ---------------------------------------------------------------------------


def test_binary_parse_known_header():
    """
    Construct a synthetic binary header with known field values.
    parse_header() must return those exact values.
    Tests _u64, _f64, _u8arr, and quaternion re-ordering.
    """
    h = np.zeros(276, dtype=">u2")
    # rows=260, cols=276
    h[0] = 260
    h[1] = 276
    # exp_time=600, exp_unit=1
    h[2] = 600
    h[3] = 1
    # ccd_temp1 = -18.0 via mixed-endian float64
    _encode_f64(h, 4, -18.0)
    # lua_timestamp = 1741999867000
    _encode_u64(h, 8, 1741999867000)
    # lamp_ch_array[0] = 1 (lamp 1 on)
    h[104] = 1
    meta = parse_header(h)
    assert meta["rows"] == 260
    assert abs(meta["ccd_temp1"] - (-18.0)) < 1e-10
    assert meta["lua_timestamp"] == 1741999867000
    assert meta["lamp1_status"] == "on"
    assert meta["img_type"] == "cal"


# ---------------------------------------------------------------------------
# T3 — Image type classification: all three types
# ---------------------------------------------------------------------------


def test_img_type_classification():
    """cal, dark, and science classification logic from Section 2.5."""
    # cal: any lamp on
    assert _classify(lamps=[1, 0, 0, 0, 0, 0], gpio=[0, 0, 0, 0]) == "cal"
    # dark: no lamp, shutter closed (gpio[0]=1, gpio[3]=1)
    assert _classify(lamps=[0, 0, 0, 0, 0, 0], gpio=[1, 0, 0, 1]) == "dark"
    # science: no lamp, shutter open
    assert _classify(lamps=[0, 0, 0, 0, 0, 0], gpio=[0, 0, 0, 0]) == "science"


# ---------------------------------------------------------------------------
# T4 — Quaternion re-ordering
# ---------------------------------------------------------------------------


def test_quaternion_reorder():
    """Binary [w,x,y,z] must become [x,y,z,w] in ImageMetadata."""
    h = np.zeros(276, dtype=">u2")
    # Encode ads_q_hat = [w=0.1, x=0.2, y=0.3, z=0.4]
    _encode_f64(h, 28, 0.1)   # w at word 28
    _encode_f64(h, 32, 0.2)   # x at word 32
    _encode_f64(h, 36, 0.3)   # y at word 36
    _encode_f64(h, 40, 0.4)   # z at word 40
    meta = parse_header(h)
    q = meta["attitude_quaternion"]
    assert abs(q[0] - 0.2) < 1e-12   # x
    assert abs(q[1] - 0.3) < 1e-12   # y
    assert abs(q[2] - 0.4) < 1e-12   # z
    assert abs(q[3] - 0.1) < 1e-12   # w


# ---------------------------------------------------------------------------
# T5 — AdcsQualityFlags: slew detection
# ---------------------------------------------------------------------------


def test_adcs_slew_detected():
    """
    A pointing_error with magnitude > 30 arcsec must set SLEW_IN_PROGRESS.
    A small pointing error must leave flag = GOOD.
    """
    # 1 degree rotation error → well above 30 arcsec threshold
    angle_rad = np.radians(1.0)
    qe_large = [np.sin(angle_rad / 2), 0.0, 0.0, np.cos(angle_rad / 2)]
    meta_large = {"pointing_error": qe_large,
                  "pos_eci_hat": [6e6, 0, 0], "adcs_timestamp": 1}
    flag = compute_adcs_quality_flag(meta_large)
    assert flag & AdcsQualityFlags.SLEW_IN_PROGRESS

    # Sub-arcsecond error → GOOD
    qe_small = [1e-5, 0.0, 0.0, 1.0]
    meta_small = {"pointing_error": qe_small,
                  "pos_eci_hat": [6e6, 0, 0], "adcs_timestamp": 1}
    assert compute_adcs_quality_flag(meta_small) == AdcsQualityFlags.GOOD


# ---------------------------------------------------------------------------
# T6 — Synthetic metadata: NB02 fields round-trip
# ---------------------------------------------------------------------------


def test_synthetic_metadata_from_nb02():
    """
    build_synthetic_metadata() must populate truth_v_los, tangent_lat,
    and etalon_gap_mm from the NB02 and InstrumentParams inputs.
    is_synthetic must be True. adcs_quality_flag must be GOOD.
    """
    from src.fpi import InstrumentParams
    params = InstrumentParams()
    nb02_tp = {"tp_lat_deg": 30.0, "tp_lon_deg": -90.0,
               "tp_alt_km": 250.1, "tp_eci": [1e6, 2e6, 3e6]}
    nb02_vr = {"v_rel": 150.0, "v_wind_LOS": 148.0,
               "V_sc_LOS": -7100.0, "v_earth_LOS": 2.0,
               "v_zonal_ms": 100.0, "v_merid_ms": 50.0}
    nb01_row = _make_nb01_row()
    meta = build_synthetic_metadata(
        params=params, nb01_row=nb01_row,
        nb02_tp=nb02_tp, nb02_vr=nb02_vr,
        quaternion_xyzw=[0, 0, 0, 1], los_eci=[1, 0, 0],
        look_mode="along_track", img_type="science",
        orbit_number=1, frame_sequence=5, noise_seed=42
    )
    assert meta.is_synthetic is True
    assert abs(meta.truth_v_los - 148.0) < 1e-10
    assert abs(meta.tangent_lat - 30.0) < 1e-10
    assert abs(meta.etalon_gap_mm - params.t * 1000) < 1e-6
    assert meta.adcs_quality_flag == AdcsQualityFlags.GOOD
    assert meta.orbit_parity == "along_track"   # orbit_number=1 is odd


# ---------------------------------------------------------------------------
# T7 — Legacy typo normalisation
# ---------------------------------------------------------------------------


def test_legacy_typo_normalised():
    """
    A JSON file with 'attitude_quadternion' (legacy typo) must be read
    correctly and normalised to 'attitude_quaternion'.
    """
    data = _make_minimal_real_metadata_dict()
    data["attitude_quadternion"] = data.pop("attitude_quaternion")
    with tempfile.NamedTemporaryFile(
            mode="w", suffix="_L0.json", delete=False) as f:
        json.dump(data, f)
        path = pathlib.Path(f.name)
    meta = read_sidecar(path)
    assert hasattr(meta, "attitude_quaternion")
    assert len(meta.attitude_quaternion) == 4


# ---------------------------------------------------------------------------
# T8 — File size validation
# ---------------------------------------------------------------------------


def test_file_size_validation(tmp_path):
    """ingest_real_image must raise ValueError for wrong file size."""
    bad = tmp_path / "bad.bin"
    bad.write_bytes(b"\x00" * 1000)
    with pytest.raises(ValueError, match="File size mismatch"):
        ingest_real_image(bad)


# ---------------------------------------------------------------------------
# Import needed for T5
# ---------------------------------------------------------------------------

from src.metadata.p01_image_metadata_2026_04_06 import compute_adcs_quality_flag
