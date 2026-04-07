"""
Shared pytest fixtures for the soc_sewell test suite.

Fixtures defined here are available to all test modules automatically.
"""
import pytest
import numpy as np

from src.fpi.m01_airy_forward_model_2026_04_05 import InstrumentParams, NE_WAVELENGTH_1_M
from src.fpi.m02_calibration_synthesis_2026_04_05 import synthesise_calibration_image
from src.fpi.m03_annular_reduction_2026_04_06 import reduce_calibration_frame
from src.fpi.tolansky_2026_04_05 import TwoLineResult
from src.fpi.m05_calibration_inversion_2026_04_06 import (
    fit_calibration_fringe,
    FitConfig,
)


# ---------------------------------------------------------------------------
# Helper: Tolansky stub (bypasses TolanskyPipeline.run() amplitude-split
# reliability issue on synthetic data)
# ---------------------------------------------------------------------------

def _build_tolansky_stub(params: InstrumentParams) -> TwoLineResult:
    """
    Build a TwoLineResult stub with correct values from known InstrumentParams.
    """
    tol = TwoLineResult.__new__(TwoLineResult)
    tol.d_m          = float(params.t)
    tol.alpha_rad_px = float(params.alpha)
    tol.eps1         = float((2.0 * params.t / NE_WAVELENGTH_1_M) % 1.0)
    return tol


# ---------------------------------------------------------------------------
# Session-scoped: CalibrationResult from noiseless synthetic calibration image
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def synthetic_cal_result():
    """
    A CalibrationResult from a noiseless synthetic calibration image.

    Computed once per test session (slow, ~30 s).
    Uses _build_tolansky_stub — not TolanskyPipeline.run() — to avoid
    amplitude-split reliability issues on synthetic data.
    """
    params = InstrumentParams()
    cal_m02 = synthesise_calibration_image(params, add_noise=False)
    fp = reduce_calibration_frame(
        cal_m02["image_2d"],
        cx_human=params.r_max,
        cy_human=params.r_max,
        r_max_px=params.r_max,
    )
    tol_stub = _build_tolansky_stub(params)
    config   = FitConfig(tolansky=tol_stub)
    return fit_calibration_fringe(fp, config)


# ---------------------------------------------------------------------------
# Function-scoped: noiseless airglow FringeProfile at 100 m/s
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_airglow_profile():
    """A noiseless FringeProfile from a 100 m/s airglow image."""
    from src.fpi.m04_airglow_synthesis_2026_04_05 import synthesise_airglow_image
    from src.fpi.m03_annular_reduction_2026_04_06 import reduce_science_frame

    params = InstrumentParams()
    cal_fp = reduce_calibration_frame(
        synthesise_calibration_image(params, add_noise=False)["image_2d"],
        cx_human=params.r_max,
        cy_human=params.r_max,
        r_max_px=params.r_max,
    )
    sci_m04 = synthesise_airglow_image(v_rel_ms=100.0, params=params, add_noise=False)
    return reduce_science_frame(
        sci_m04["image_2d"],
        cx=cal_fp.cx,
        cy=cal_fp.cy,
        sigma_cx=cal_fp.sigma_cx,
        sigma_cy=cal_fp.sigma_cy,
        r_max_px=params.r_max,
    )
