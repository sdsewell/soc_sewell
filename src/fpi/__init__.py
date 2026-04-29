# fpi/__init__.py — FPI instrument chain subpackage
# Implements: M01 (S09), M02 (S10), M03 (S12), M04 (S11),
#             M05 (S13), M06 (S14), M07 (S15), Tolansky
#
# Re-export from current dated implementation files per S01 Section 10.
from src.fpi.m01_airy_forward_model_2026_04_26 import (  # noqa: F401
    InstrumentParams,
    airy_ideal,
    airy_modified,
    build_instrument_matrix,
    intensity_envelope,
    make_wavelength_grid,
    make_airglow_spectrum,
    make_ne_spectrum,
    psf_sigma,
    theta_from_r,
)
from src.fpi.m02_calibration_synthesis_2026_04_05 import (  # noqa: F401
    add_poisson_noise,
    radial_profile_to_image,
    synthesise_calibration_image,
)
from src.fpi.m03_annular_reduction_2026_04_06 import (  # noqa: F401
    CentreResult,
    FringeProfile,
    PeakFit,
    QualityFlags,
    annular_reduce,
    azimuthal_variance_centre,
    coarse_centre_com,
    estimate_centre_uncertainty,
    find_centre,
    make_master_dark,
    reduce_calibration_frame,
    reduce_science_frame,
    resolve_seed,
    subtract_dark,
)
from src.fpi.m04_airglow_synthesis_2026_04_05 import (  # noqa: F401
    add_gaussian_noise,
    lambda_c_to_v_rel,
    synthesise_airglow_image,
    v_rel_to_lambda_c,
)
from src.fpi.m05_calibration_inversion_2026_04_06 import (  # noqa: F401
    CalibrationResult,
    FitConfig,
    FitFlags,
    fit_calibration_fringe,
)
from src.fpi.m06_airglow_inversion_2026_04_06 import (  # noqa: F401
    AirglowFitFlags,
    AirglowFitResult,
    fit_airglow_fringe,
)
from src.fpi.m07_wind_retrieval_2026_04_06 import (  # noqa: F401
    WindObservation,
    WindResult,
    WindResultFlags,
    compute_sensitivity_coefficients,
    pair_observations,
    retrieve_wind_vectors,
)
