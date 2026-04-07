"""
Shared pytest fixtures for the soc_sewell test suite.

Fixtures defined here are available to all test modules automatically.
"""
import pytest
import numpy as np

from src.fpi.m01_airy_forward_model_2026_04_05 import InstrumentParams, NE_WAVELENGTH_1_M
from src.fpi.m02_calibration_synthesis_2026_04_05 import synthesise_calibration_image
from src.fpi.m03_annular_reduction_2026_04_06 import annular_reduce
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

    Computed once per test session.  The calibration image is reduced using
    cx = params.r_max (= 128.0 px) — the same centre used in T3–T8 science
    frame reductions — so M05's fitted effective instrument parameters
    (R_refl, sigma0, I0, …) absorb the same 2D pixel-averaging bias that
    appears in the science frames.  If the calibration were reduced at the
    true image centre (127.5 px, as found by reduce_calibration_frame's
    variance minimiser), the effective params would differ slightly and
    introduce a ~20 m/s systematic wind bias in T3.

    Uses _build_tolansky_stub — not TolanskyPipeline.run() — to avoid
    amplitude-split reliability issues on synthetic data.
    """
    params = InstrumentParams()
    # Synthesize calibration image at the same centre as the M06 science tests
    # (cx = params.r_max = 128.0 px) and reduce it with the same cx.
    # This eliminates the synthesis–reduction centre offset that would otherwise
    # cause a ~6 m/s systematic wind bias in T3 (the offset shifts the effective
    # fringe radius in M05's fitted params relative to M06's science data).
    cal_m02 = synthesise_calibration_image(
        params, add_noise=False,
        cx=params.r_max, cy=params.r_max,
    )
    fp = annular_reduce(
        cal_m02["image_2d"],
        cx=params.r_max,
        cy=params.r_max,
        sigma_cx=0.05,
        sigma_cy=0.05,
        r_max_px=params.r_max,
        n_bins=150,
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

    params  = InstrumentParams()
    sci_m04 = synthesise_airglow_image(
        v_rel_ms=100.0, params=params, add_noise=False,
        cx=params.r_max, cy=params.r_max,
    )
    return reduce_science_frame(
        sci_m04["image_2d"],
        cx=params.r_max,
        cy=params.r_max,
        sigma_cx=0.05,
        sigma_cy=0.05,
        r_max_px=params.r_max,
    )
