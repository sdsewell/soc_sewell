"""
Tests for S03 physical constants module.

Spec:        S03_physical_constants_2026-04-05.md
Spec tests:  T1–T8
Run with:    pytest tests/test_s03_constants.py -v
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# T1 — OI wavelength is not legacy 630.0 nm
# ---------------------------------------------------------------------------
def test_OI_wavelength_not_legacy():
    """Catch any regression to the legacy 630.0 nm value."""
    from src.constants import OI_WAVELENGTH_M
    assert abs(OI_WAVELENGTH_M - 630.0e-9) > 1e-12, (
        "OI_WAVELENGTH_M appears to be the legacy 630.0 nm value; should be 630.0304 nm"
    )
    assert abs(OI_WAVELENGTH_M - 630.0304e-9) < 1e-14, (
        f"OI_WAVELENGTH_M = {OI_WAVELENGTH_M * 1e9:.6f} nm; expected 630.0304 nm"
    )


# ---------------------------------------------------------------------------
# T2 — Etalon gap is not legacy 20.670 mm
# ---------------------------------------------------------------------------
def test_etalon_gap_not_legacy():
    """Catch any regression to the FlatSat FSR-error value."""
    from src.constants import ETALON_GAP_M
    assert abs(ETALON_GAP_M - 20.670e-3) > 1e-6, (
        "ETALON_GAP_M appears to be the legacy 20.670 mm FlatSat value; should be 20.008 mm"
    )
    assert abs(ETALON_GAP_M - 20.008e-3) < 1e-9, (
        f"ETALON_GAP_M = {ETALON_GAP_M * 1e3:.4f} mm; expected 20.008 mm"
    )


# ---------------------------------------------------------------------------
# T3 — Depression angle is computed from primaries, not hardcoded
# ---------------------------------------------------------------------------
def test_depression_angle_computed_from_primaries():
    """
    DEPRESSION_ANGLE_DEG must equal compute_depression_angle(SC_ALTITUDE_KM,
    TP_ALTITUDE_KM) to machine precision — proving it is computed, not hardcoded.
    """
    from src.constants import (DEPRESSION_ANGLE_DEG, SC_ALTITUDE_KM,
                                TP_ALTITUDE_KM, compute_depression_angle)
    recomputed = compute_depression_angle(SC_ALTITUDE_KM, TP_ALTITUDE_KM)
    assert abs(DEPRESSION_ANGLE_DEG - recomputed) < 1e-10, (
        f"DEPRESSION_ANGLE_DEG ({DEPRESSION_ANGLE_DEG:.4f}°) does not match "
        f"compute_depression_angle({SC_ALTITUDE_KM}, {TP_ALTITUDE_KM}) "
        f"= {recomputed:.4f}°. It may be hardcoded."
    )
    assert abs(DEPRESSION_ANGLE_DEG - 15.79) < 0.02, \
        f"Nominal depression angle = {DEPRESSION_ANGLE_DEG:.2f}°; expected ~15.79°"
    assert abs(DEPRESSION_ANGLE_DEG - 23.4) > 1.0, \
        "DEPRESSION_ANGLE_DEG is the legacy 23.4° value"


# ---------------------------------------------------------------------------
# T4 — FSR calculation is self-consistent
# ---------------------------------------------------------------------------
def test_fsr_consistency():
    """Derived FSR values are consistent with primary constants."""
    from src.constants import (
        ETALON_FSR_OI_M, ETALON_FSR_NE1_M,
        OI_WAVELENGTH_M, NE_WAVELENGTH_1_M, ETALON_GAP_M,
    )
    fsr_oi_check  = OI_WAVELENGTH_M ** 2  / (2.0 * ETALON_GAP_M)
    fsr_ne1_check = NE_WAVELENGTH_1_M ** 2 / (2.0 * ETALON_GAP_M)
    assert abs(ETALON_FSR_OI_M  - fsr_oi_check)  < 1e-18
    assert abs(ETALON_FSR_NE1_M - fsr_ne1_check) < 1e-18


# ---------------------------------------------------------------------------
# T5 — Neon line separation ≈ 187.9 FSR
# ---------------------------------------------------------------------------
def test_neon_separation_fsr():
    """Beat period anchor: Ne lines are ~188 FSR apart."""
    from src.constants import NE_SEPARATION_FSR
    assert 185 < NE_SEPARATION_FSR < 191, (
        f"NE_SEPARATION_FSR = {NE_SEPARATION_FSR:.1f}; expected ≈ 187.9"
    )


# ---------------------------------------------------------------------------
# T6 — Magnification constant is consistent with pixel pitch and focal length
# ---------------------------------------------------------------------------
def test_alpha_consistency():
    """ALPHA_RAD_PX = CCD_PIXEL_2X2_UM * 1e-6 / FOCAL_LENGTH_M."""
    from src.constants import ALPHA_RAD_PX, CCD_PIXEL_2X2_UM, FOCAL_LENGTH_M
    expected = CCD_PIXEL_2X2_UM * 1e-6 / FOCAL_LENGTH_M
    assert abs(ALPHA_RAD_PX - expected) < 1e-10, (
        f"ALPHA_RAD_PX = {ALPHA_RAD_PX:.3e}; expected {expected:.3e}"
    )


# ---------------------------------------------------------------------------
# T7 — All constants are float or tuple, not string
# ---------------------------------------------------------------------------
def test_constant_types():
    """Guard against string or None values from typos."""
    import src.constants as c
    non_tuple_names = [
        "SPEED_OF_LIGHT_MS", "BOLTZMANN_J_PER_K", "OI_WAVELENGTH_M",
        "NE_WAVELENGTH_1_M", "NE_WAVELENGTH_2_M", "ETALON_GAP_M",
        "FOCAL_LENGTH_M", "ALPHA_RAD_PX", "DEPRESSION_ANGLE_DEG",
        "SC_ALTITUDE_KM", "TP_ALTITUDE_KM", "WIND_BIAS_BUDGET_MS",
    ]
    for name in non_tuple_names:
        val = getattr(c, name)
        assert isinstance(val, (int, float)), (
            f"{name} has type {type(val).__name__}, expected float"
        )


# ---------------------------------------------------------------------------
# T8 — Velocity per FSR is physically reasonable
# ---------------------------------------------------------------------------
def test_velocity_per_fsr():
    """One FSR should correspond to ~4.7 km/s at OI 630 nm, 20 mm gap."""
    from src.constants import VELOCITY_PER_FSR_MS
    assert 4_500 < VELOCITY_PER_FSR_MS < 5_000, (
        f"VELOCITY_PER_FSR_MS = {VELOCITY_PER_FSR_MS:.0f} m/s; expected ~4720 m/s"
    )


# ---------------------------------------------------------------------------
# T9 — compute_depression_angle() responds correctly to altitude inputs
# ---------------------------------------------------------------------------
def test_depression_angle_sensitivity():
    """
    Verify that compute_depression_angle() correctly responds to different
    altitude inputs. Proves it is a live calculation, not a lookup or stub.
    """
    from src.constants import compute_depression_angle
    angle_nominal = compute_depression_angle(510.0, 250.0)   # 15.73°
    angle_low_sc  = compute_depression_angle(500.0, 250.0)   # lower orbit → smaller angle
    angle_high_sc = compute_depression_angle(550.0, 250.0)   # higher orbit → larger angle
    angle_high_tp = compute_depression_angle(510.0, 300.0)   # higher tangent → smaller angle

    assert angle_low_sc  < angle_nominal, \
        "Lower orbit altitude should give smaller depression angle"
    assert angle_high_sc > angle_nominal, \
        "Higher orbit altitude should give larger depression angle"
    assert angle_high_tp < angle_nominal, \
        "Higher tangent height should give smaller depression angle"

    for angle, label in [(angle_nominal, "nominal"), (angle_low_sc, "low_sc"),
                         (angle_high_sc, "high_sc"), (angle_high_tp, "high_tp")]:
        assert 10.0 < angle < 25.0, \
            f"Depression angle ({label}) = {angle:.2f}° outside plausible range [10°, 25°]"
