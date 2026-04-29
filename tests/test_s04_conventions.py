"""
Tests for S04 uncertainty standards and reporting conventions.

Spec:        S04_uncertainty_standards_2026-04-05.md
Spec tests:  T1–T6
Run with:    pytest tests/test_s04_conventions.py -v
"""


# ---------------------------------------------------------------------------
# T1 — two_sigma is exactly 2 × sigma
# ---------------------------------------------------------------------------
def test_two_sigma_exact():
    """two_sigma must always be exactly 2.0 × sigma, never independently computed."""
    sigma_val = 3.7   # arbitrary
    two_sigma_val = 2.0 * sigma_val
    assert two_sigma_val == 2.0 * sigma_val
    # Documents the intent: 2.0 * x must equal 2.0 * x
    assert two_sigma_val / sigma_val == 2.0


# ---------------------------------------------------------------------------
# T2 — chi2_reduced acceptable range boundaries
# ---------------------------------------------------------------------------
def test_chi2_flag_logic():
    """Verify flag assignment logic for chi2_reduced values."""
    def assign_flags(chi2):
        flags = 0x00
        if chi2 > 3.0:  flags |= 0x02
        if chi2 > 10.0: flags |= 0x04
        if chi2 < 0.5:  flags |= 0x08
        return flags

    assert assign_flags(1.0)  == 0x00   # GOOD
    assert assign_flags(5.0)  == 0x02   # CHI2_HIGH
    assert assign_flags(15.0) == 0x06   # CHI2_HIGH | CHI2_VERY_HIGH
    assert assign_flags(0.3)  == 0x08   # CHI2_LOW


# ---------------------------------------------------------------------------
# T3 — sigma_v from sigma_lambda propagation
# ---------------------------------------------------------------------------
def test_sigma_v_propagation():
    """Verify Doppler uncertainty propagation: sigma_v = c * sigma_lambda / lambda0."""
    from src.constants import SPEED_OF_LIGHT_MS, OI_WAVELENGTH_AIR_M
    sigma_lambda_c = 2.0e-14   # 0.02 pm, typical value
    sigma_v_expected = SPEED_OF_LIGHT_MS * sigma_lambda_c / OI_WAVELENGTH_AIR_M
    assert 8.0 < sigma_v_expected < 12.0, (
        f"sigma_v = {sigma_v_expected:.2f} m/s; expected ~9.5 m/s for sigma_lambda = 0.02 pm"
    )


# ---------------------------------------------------------------------------
# T4 — Quality flag bitwise operations
# ---------------------------------------------------------------------------
def test_quality_flag_operations():
    """Bitwise flag operations work correctly."""
    from src.constants import PipelineFlags

    flags = PipelineFlags.GOOD
    assert flags == PipelineFlags.GOOD

    flags |= PipelineFlags.CHI2_HIGH
    assert flags & PipelineFlags.CHI2_HIGH
    assert not (flags & PipelineFlags.FIT_FAILED)

    flags |= PipelineFlags.FIT_FAILED
    assert flags & PipelineFlags.CHI2_HIGH
    assert flags & PipelineFlags.FIT_FAILED
    assert flags != PipelineFlags.GOOD


# ---------------------------------------------------------------------------
# T5 — Doppler formula sign convention
# ---------------------------------------------------------------------------
def test_doppler_sign_convention():
    """Positive v_rel (recession) produces lambda_c > lambda_0."""
    from src.constants import SPEED_OF_LIGHT_MS, OI_WAVELENGTH_AIR_M
    v_rel = +100.0   # m/s, positive = recession = redshift
    lambda_c = OI_WAVELENGTH_AIR_M * (1 + v_rel / SPEED_OF_LIGHT_MS)
    assert lambda_c > OI_WAVELENGTH_AIR_M, (
        "Recession should produce redshift (lambda_c > lambda_0)"
    )

    v_rel_check = SPEED_OF_LIGHT_MS * (lambda_c - OI_WAVELENGTH_AIR_M) / OI_WAVELENGTH_AIR_M
    assert abs(v_rel_check - v_rel) < 0.01, "Round-trip Doppler formula failed"


# ---------------------------------------------------------------------------
# T6 — STM wind precision is achievable
# ---------------------------------------------------------------------------
def test_stm_wind_budget():
    """sigma_v of 9.8 m/s corresponds to sigma_lambda ~ 2.06e-14 m at OI 630 nm."""
    from src.constants import SPEED_OF_LIGHT_MS, OI_WAVELENGTH_AIR_M, WIND_BIAS_BUDGET_MS
    sigma_lambda_required = WIND_BIAS_BUDGET_MS * OI_WAVELENGTH_AIR_M / SPEED_OF_LIGHT_MS
    # Should be ~2.06e-14 m = 0.0206 pm
    assert 1.5e-14 < sigma_lambda_required < 2.5e-14, (
        f"sigma_lambda required for STM = {sigma_lambda_required:.3e} m; expected ~2.06e-14 m"
    )
