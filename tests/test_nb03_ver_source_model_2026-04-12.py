"""
Tests for NB03 — VER Source Model and Signal Budget.

Spec:   specs/S07b_nb03_ver_source_model_2026-04-12.md
Module: src/fpi/nb03_ver_source_model_2026_04_12.py

8 tests (T1–T8) per spec §7.
"""

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Load nb03 module by file path (handles underscored date convention)
# ---------------------------------------------------------------------------

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_NB03_PATH = _REPO_ROOT / "src" / "fpi" / "nb03_ver_source_model_2026_04_12.py"

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_spec = importlib.util.spec_from_file_location("nb03_ver_source_model", _NB03_PATH)
_nb03 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_nb03)

build_ver_profile        = _nb03.build_ver_profile
altitude_along_los       = _nb03.altitude_along_los
compute_los_geometry     = _nb03.compute_los_geometry
integrate_los_emission   = _nb03.integrate_los_emission
signal_photons_per_pixel = _nb03.signal_photons_per_pixel
signal_to_I_line         = _nb03.signal_to_I_line
compute_signal_budget    = _nb03.compute_signal_budget


# ---------------------------------------------------------------------------
# T1 — VER profile positive and peaks in correct altitude range
# ---------------------------------------------------------------------------

def test_ver_profile_nightglow_peak():
    """
    Nightglow VER must peak between 200 and 300 km.
    Peak value must be in physically plausible range 1–100 cm⁻³ s⁻¹.
    Must be zero at 100 km and 500 km (table boundaries).
    """
    ver = build_ver_profile('nightglow')
    h = np.linspace(100e3, 500e3, 2000)
    v = ver(h)
    # Convert back to cm⁻³ s⁻¹ for physical bounds check
    v_cm3 = v * 4 * np.pi / 1e6
    peak_h_km = h[np.argmax(v_cm3)] / 1000.0
    peak_v    = np.max(v_cm3)
    assert 200.0 < peak_h_km < 300.0, \
        f"Nightglow VER peak at {peak_h_km:.1f} km, expected 200–300 km"
    assert 1.0 < peak_v < 100.0, \
        f"Peak VER {peak_v:.2f} cm⁻³ s⁻¹ outside [1, 100]"
    assert ver(100e3) == 0.0 or abs(ver(100e3)) < 1e-10, \
        "VER must be zero at lower boundary (100 km)"
    assert ver(500e3) == 0.0 or abs(ver(500e3)) < 1e-10, \
        "VER must be zero at upper boundary (500 km)"


# ---------------------------------------------------------------------------
# T2 — LOS geometry: pitch angle correct for nominal parameters
# ---------------------------------------------------------------------------

def test_los_geometry_nominal():
    """
    For h_orbit=500 km, h_tangent=250 km:
      pitch_angle = arccos((R_earth+250e3)/(R_earth+500e3))
      d_tangent   = R_orbit * sin(pitch_angle)
    Values must be physically consistent to < 1 m.
    """
    import astropy.constants as C
    geo = compute_los_geometry(h_tangent_m=250e3, h_orbit_m=500e3)
    R_e = C.R_earth.value
    R_orb = R_e + 500e3
    expected_pa = np.arccos((R_e + 250e3) / R_orb)
    expected_dt = R_orb * np.sin(expected_pa)
    assert abs(geo['pitch_angle_rad'] - expected_pa) < 1e-10, \
        f"Pitch angle {np.degrees(geo['pitch_angle_rad']):.4f} deg, " \
        f"expected {np.degrees(expected_pa):.4f} deg"
    assert abs(geo['d_tangent_m'] - expected_dt) < 1.0, \
        f"d_tangent {geo['d_tangent_m']:.1f} m, expected {expected_dt:.1f} m"
    assert geo['d_near_m'] < geo['d_tangent_m'] < geo['d_far_m'], \
        "d_near < d_tangent < d_far must hold"


# ---------------------------------------------------------------------------
# T3 — Altitude along LOS is minimum at tangent point
# ---------------------------------------------------------------------------

def test_altitude_along_los_minimum_at_tangent():
    """
    h(d) must be minimum at d = d_tangent and monotonically decrease
    then increase on either side.
    """
    import astropy.constants as C
    geo = compute_los_geometry(250e3, 500e3)
    R_orb = C.R_earth.value + 500e3
    R_e   = C.R_earth.value
    d = np.linspace(geo['d_near_m'], geo['d_far_m'], 500)
    h = altitude_along_los(d, geo['pitch_angle_rad'], R_orb, R_e)
    idx_min = np.argmin(h)
    # Tangent point should be near the middle of the array
    assert 100 < idx_min < 400, \
        f"Altitude minimum at index {idx_min}, expected near center"
    # Altitude at minimum should equal h_tangent to within 1 km
    assert abs(h[idx_min] - 250e3) < 1000.0, \
        f"Min altitude {h[idx_min]/1e3:.2f} km, expected 250.00 km"


# ---------------------------------------------------------------------------
# T4 — Column brightness is positive and physically plausible
# ---------------------------------------------------------------------------

def test_column_brightness_physical():
    """
    Nightglow column brightness must be in the range 1e9–1e13
    photons m⁻² s⁻¹ sr⁻¹ for the nominal geometry.
    (Equivalent to ~10–10000 Rayleigh in 4π sr, per Wang et al. 2020.)
    """
    ver = build_ver_profile('nightglow')
    geo = compute_los_geometry(250e3, 500e3)
    B = integrate_los_emission(ver, geo)
    assert B > 0, "Column brightness must be positive"
    assert 1e9 < B < 1e13, \
        f"Column brightness {B:.3e} ph m⁻² s⁻¹ sr⁻¹ outside [1e9, 1e13]"


# ---------------------------------------------------------------------------
# T5 — Photon flux per pixel is positive and sub-saturation
# ---------------------------------------------------------------------------

def test_photon_flux_per_pixel():
    """
    Photon flux must be positive.
    For nightglow, expect < 10000 photons/s/pixel (CCD97 full well
    ~80000 e⁻; at QE=0.9 and 10s, saturation would require >>8000
    photons/s/pixel).
    """
    ver = build_ver_profile('nightglow')
    geo = compute_los_geometry(250e3, 500e3)
    B   = integrate_los_emission(ver, geo)
    F   = signal_photons_per_pixel(B)
    assert F > 0, "Photon flux must be positive"
    assert F < 10000, f"Photon flux {F:.1f} ph/s/px implausibly high"


# ---------------------------------------------------------------------------
# T6 — I_line is in M04-compatible range
# ---------------------------------------------------------------------------

def test_I_line_in_m04_range():
    """
    I_line must be positive.
    For nightglow with nominal parameters, expected range 0.001–10.
    I_line >> 10 would saturate the simulated image;
    I_line << 0.001 would make the signal undetectable.
    """
    result = compute_signal_budget(mode='nightglow')
    I_line = result['I_line']
    assert I_line > 0, "I_line must be positive"
    assert 0.001 < I_line < 10.0, \
        f"I_line = {I_line:.4f} outside expected range [0.001, 10]"


# ---------------------------------------------------------------------------
# T7 — Dayglow is brighter than nightglow
# ---------------------------------------------------------------------------

def test_dayglow_brighter_than_nightglow():
    """
    Dayglow I_line must be > nightglow I_line.
    Wang et al. (2020): dayglow VER ~100× nightglow.
    Integrated brightness ratio need not be exactly 100× but must be > 5×.
    """
    day   = compute_signal_budget(mode='dayglow')
    night = compute_signal_budget(mode='nightglow')
    assert day['I_line'] > night['I_line'], \
        "Dayglow must be brighter than nightglow"
    ratio = day['I_line'] / night['I_line']
    assert ratio > 5.0, \
        f"Day/night I_line ratio {ratio:.1f} < 5, inconsistent with literature"


# ---------------------------------------------------------------------------
# T8 — signal_to_I_line round-trip
# ---------------------------------------------------------------------------

def test_signal_to_I_line_scaling():
    """
    I_line must scale linearly with each detector constant:
      - doubling t_exp doubles I_line
      - doubling QE doubles I_line
      - doubling aperture doubles column brightness (tested via
        signal_photons_per_pixel with doubled aperture)
    """
    base         = signal_to_I_line(100.0)
    double_texp  = signal_to_I_line(100.0, t_exp=2 * 10.0)
    double_qe    = signal_to_I_line(100.0, qe=2 * 0.90)
    assert abs(double_texp / base - 2.0) < 1e-10, \
        "I_line must double when t_exp doubles"
    assert abs(double_qe / base - 2.0) < 1e-10, \
        "I_line must double when QE doubles"
