"""
Print a reference table of all ImageMetadata fields.

Columns: index | field name | header pixels | description | default | units
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# fmt: off
FIELDS = [
    # (field_name, pixels, description, default, units)

    # Section 3.1 — Image geometry and classification
    ("rows",                  "0",        "Total rows including header row",                     "260",              "—"),
    ("cols",                  "1",        "Total columns",                                       "276",              "—"),
    ("exp_time",              "2",        "Exposure duration",                                   "—",                "centiseconds"),
    ("exp_unit",              "3",        "CCD timing register value",                           "—",                "—"),
    ("binning",               "derived",  "Pixel binning mode (derived from cols)",              "2",                "—"),
    ("img_type",              "derived",  "Image type: 'cal', 'dark', or 'science'",             "—",                "—"),

    # Section 3.2 — Timing
    ("lua_timestamp",         "8–11",     "End-of-exposure time (primary reference)",            "—",                "Unix ms"),
    ("adcs_timestamp",        "12–15",    "ADCS clock time at acquisition; 0 if unavailable",   "0",                "Unix ms"),
    ("utc_timestamp",         "derived",  "ISO 8601 string derived from lua_timestamp",          "—",                "—"),

    # Section 3.3 — Orbit state
    ("spacecraft_latitude",   "16–19",    "Spacecraft geodetic latitude",                        "—",                "rad"),
    ("spacecraft_longitude",  "20–23",    "Spacecraft geodetic longitude",                       "—",                "rad"),
    ("spacecraft_altitude",   "24–27",    "Spacecraft altitude above WGS84 ellipsoid",           "—",                "m"),
    ("pos_eci_hat",           "60–71",    "ECI position vector [x, y, z]",                       "—",                "m"),
    ("vel_eci_hat",           "72–83",    "ECI velocity vector [vx, vy, vz]",                    "—",                "m/s"),

    # Section 3.4 — Attitude
    ("attitude_quaternion",   "28–43",    "Body-to-ECI quaternion [x, y, z, w] scalar-last",    "—",                "—"),
    ("pointing_error",        "44–59",    "Residual attitude error quaternion [x, y, z, w]",     "—",                "—"),
    ("obs_mode",              "—",        "Observation mode: 'along_track', 'cross_track', or 'unknown'", "unknown", "—"),

    # Section 3.5 — Instrument state
    ("ccd_temp1",             "4–7",      "CCD temperature",                                     "—",                "°C"),
    ("etalon_temps",          "84–99",    "Etalon temperature array [T0, T1, T2, T3]",           "—",                "°C"),
    ("shutter_status",        "derived",  "Shutter state: 'open' or 'closed'",                   "open",             "—"),
    ("gpio_pwr_on",           "100–103",  "GPIO power register values [ch0–ch3]",                "—",                "—"),
    ("lamp_ch_array",         "104–109",  "Lamp channel states [ch0–ch5]",                       "—",                "—"),
    ("lamp1_status",          "derived",  "Lamp 1 state: 'on' or 'off'",                         "off",              "—"),
    ("lamp2_status",          "derived",  "Lamp 2 state: 'on' or 'off'",                         "off",              "—"),
    ("lamp3_status",          "derived",  "Lamp 3 state: 'on' or 'off'",                         "off",              "—"),

    # Section 3.6 — Orbit and sequence identification
    ("orbit_number",          "—",        "Absolute orbit counter",                              "None",             "—"),
    ("frame_sequence",        "—",        "Frame index within the orbit",                        "None",             "—"),
    ("orbit_parity",          "derived",  "Orbit look direction derived from orbit_number parity","None",            "—"),

    # Section 3.7 — Pointing quality gate
    ("adcs_quality_flag",     "derived",  "ADCS quality bitmask (0 = good)",                    "0",                "—"),

    # Section 3.8 — Dark subtraction provenance
    ("dark_subtracted",       "—",        "Whether dark subtraction has been applied",           "False",            "—"),
    ("dark_n_frames",         "—",        "Number of dark frames averaged for subtraction",      "0",                "—"),
    ("dark_lua_timestamp",    "—",        "Timestamp of the dark frame used",                   "None",             "Unix ms"),
    ("dark_etalon_temp_mean", "—",        "Mean etalon temperature of the dark frame",           "None",             "°C"),

    # Section 3.9 — Synthetic fields
    ("is_synthetic",          "—",        "True if image was simulated, not from hardware",      "False",            "—"),
    ("truth_v_los",           "—",        "True line-of-sight wind speed (synthetic only)",      "None",             "m/s"),
    ("truth_v_zonal",         "—",        "True zonal wind component (synthetic only)",          "None",             "m/s"),
    ("truth_v_meridional",    "—",        "True meridional wind component (synthetic only)",     "None",             "m/s"),
    ("tangent_lat",           "—",        "Tangent point geodetic latitude (synthetic only)",    "None",             "deg"),
    ("tangent_lon",           "—",        "Tangent point geodetic longitude (synthetic only)",   "None",             "deg"),
    ("tangent_alt_km",        "—",        "Tangent point altitude (synthetic only)",             "None",             "km"),
    ("etalon_gap_mm",         "—",        "Etalon gap used in synthesis",                        "None",             "mm"),
    ("noise_seed",            "—",        "RNG seed used for noise generation (synthetic only)", "None",             "—"),

    # Section 3.10 — Etalon thermal correction hook
    ("etalon_gap_corrected_mm", "—",      "Thermally corrected etalon gap (pipeline-added)",     "None",             "mm"),

    # Section 3.11 — Grafana telemetry hook
    ("grafana_record_id",     "—",        "Grafana telemetry record identifier",                 "None",             "—"),
]
# fmt: on

COL_WIDTHS = (3, 28, 9, 52, 9, 14)
HEADERS = ("#", "Field", "Pixels", "Description", "Default", "Units")
SEP = "  "


def _row(cols):
    return SEP.join(str(c).ljust(w) for c, w in zip(cols, COL_WIDTHS))


def _rule():
    return SEP.join("-" * w for w in COL_WIDTHS)


if __name__ == "__main__":
    print(_row(HEADERS))
    print(_rule())
    for idx, (name, pixels, desc, default, units) in enumerate(FIELDS, start=1):
        print(_row((idx, name, pixels, desc, default, units)))
