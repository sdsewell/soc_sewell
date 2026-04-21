"""
Module:      m01_airy_forward_model_2026_04_05.py
Spec:        specs/S09_m01_airy_forward_model_2026-04-05.md
Author:      Claude Code
Generated:   2026-04-05
Last tested: 2026-04-05  (8/8 tests pass)
Project:     WindCube FPI Pipeline — NCAR/HAO
Repo:        soc_sewell

M01 is the mathematical core of the FPI instrument model. It computes
the ideal and PSF-broadened Airy transmission function and constructs
the instrument matrix A that maps a source spectrum Y(λ) to a measured
1D fringe profile S(r).

This module is wavelength-agnostic. All FPI-specific physical constants
are defined here and imported by M02, M04, M05, M06.
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

# ---------------------------------------------------------------------------
# Physical constants — single source of truth for FPI pipeline.
# Import these from here; do not redefine in downstream modules.
# ---------------------------------------------------------------------------

# OI 630 nm science line
OI_WAVELENGTH_M    = 630.0304e-9    # m, OI air wavelength (S03)
SPEED_OF_LIGHT_MS  = 299_792_458.0  # m/s, exact (S03)
BOLTZMANN_J_PER_K  = 1.380649e-23   # J/K, exact (S03)
OXYGEN_MASS_KG     = 2.6567e-26     # kg, one O-16 atom (S03)

# Neon calibration lamp lines
NE_WAVELENGTH_1_M  = 640.2248e-9    # m, primary Ne line (S03)
NE_WAVELENGTH_2_M  = 638.2991e-9    # m, secondary Ne line (S03)
NE_INTENSITY_2     = 0.8            # relative intensity of secondary line


# ---------------------------------------------------------------------------
# InstrumentParams dataclass
# ---------------------------------------------------------------------------

@dataclass
class InstrumentParams:
    """
    Container for all WindCube FPI instrument parameters.

    Passed between M01, M02, M04, M05, M06 to avoid long argument lists.
    Defaults reflect the actual WindCube instrument as characterised by
    the Tolansky analysis and FlatSat calibration.

    IMPORTANT: alpha = 1.6071e-4 rad/px is the 2×2 binned value from the
    Tolansky two-line analysis. The Harding paper value (8.5e-5) is for
    a different instrument configuration and must NOT be used here.
    """

    # Etalon
    t:       float = 20.0006e-3  # gap, metres; authoritative Tolansky result (Z01a 2026-04-21)
    R_refl:  float = 0.53        # effective reflectivity (FlatSat cal)
    n:       float = 1.0         # refractive index (air gap)
    alpha:   float = 1.6071e-4   # rad/pixel, 2×2 binned (Tolansky 2026)

    # Intensity envelope
    I0:  float =  1000.0   # average intensity, counts
    I1:  float =    -0.1   # linear vignetting coefficient
    I2:  float =   0.005   # quadratic vignetting coefficient

    # PSF
    sigma0: float =  0.5    # average PSF width, pixels
    sigma1: float =  0.1    # sine variation, pixels
    sigma2: float = -0.05   # cosine variation, pixels

    # CCD
    B:     float = 300.0    # bias pedestal, counts
    r_max: float = 128.0    # max usable radius, pixels (256px image / 2)

    def finesse_coefficient(self) -> float:
        """F = 4R / (1-R)²"""
        return 4.0 * self.R_refl / (1.0 - self.R_refl) ** 2

    def finesse(self) -> float:
        """Instrument finesse = π√R / (1-R)"""
        return np.pi * np.sqrt(self.R_refl) / (1.0 - self.R_refl)

    @property
    def t_m(self) -> float:
        """Alias for t (etalon gap, metres). Matches CalibrationResult.t_m naming."""
        return self.t

    def free_spectral_range(self, wavelength: float) -> float:
        """FSR = λ² / (2nt),  metres"""
        return wavelength ** 2 / (2.0 * self.n * self.t)


# ---------------------------------------------------------------------------
# Core functions — implement in dependency order
# ---------------------------------------------------------------------------

def theta_from_r(
    r: np.ndarray,   # radial positions, pixels, shape (R,)
    alpha: float,    # magnification constant, rad/pixel
) -> np.ndarray:
    """
    Map pixel radius to angle with optical axis.

    θ(r) = arctan(α · r)    (Harding Eq. 3)

    Parameters
    ----------
    r     : radial positions in pixels, shape (R,) or scalar
    alpha : magnification constant, rad/pixel.
            WindCube 2×2 binned: 1.6071e-4 rad/px (from S03 / Tolansky)

    Returns
    -------
    theta : angle in radians, same shape as r
    """
    return np.arctan(alpha * r)


def intensity_envelope(
    r: np.ndarray,   # radial positions, pixels, shape (R,)
    r_max: float,    # maximum radius, pixels
    I0: float,       # average intensity, counts
    I1: float,       # linear falloff coefficient
    I2: float,       # quadratic falloff coefficient
) -> np.ndarray:
    """
    Quadratic intensity envelope accounting for optical vignetting.

    I(r) = I₀ · (1 + I₁·(r/r_max) + I₂·(r/r_max)²)    (Harding Eq. 4)

    The envelope must be positive everywhere for physically valid inputs.
    Caller is responsible for choosing I1, I2 such that I(r) > 0 for
    all r in [0, r_max].

    Returns
    -------
    I : intensity in counts, same shape as r
    """
    rn = r / r_max
    return I0 * (1.0 + I1 * rn + I2 * rn ** 2)


def airy_ideal(
    r: np.ndarray,        # radial positions, pixels, shape (R,)
    wavelength: float,    # wavelength, metres
    t: float,             # etalon gap, metres
    R_refl: float,        # plate reflectivity, dimensionless
    alpha: float,         # magnification constant, rad/pixel
    n: float,             # index of refraction (1.0 for air gap)
    r_max: float,         # maximum radius, pixels
    I0: float,            # average intensity, counts
    I1: float,            # linear vignetting coefficient
    I2: float,            # quadratic vignetting coefficient
) -> np.ndarray:
    """
    Ideal (unbroadened) Airy transmission function at a single wavelength.

    A(r; λ) = I(r) / [1 + F · sin²(π · 2nt·cos(θ(r)) / λ)]

    Uses exact cosine (not small-angle approximation).
    Finesse coefficient F = 4R/(1-R)² computed internally.

    Returns
    -------
    A : CCD counts, shape (R,), values in [I(r)/(1+F), I(r)]
    """
    theta = theta_from_r(r, alpha)
    I_env = intensity_envelope(r, r_max, I0, I1, I2)
    F = 4.0 * R_refl / (1.0 - R_refl) ** 2
    OPD = 2.0 * n * t * np.cos(theta)
    phase = np.pi * OPD / wavelength
    return I_env / (1.0 + F * np.sin(phase) ** 2)


def psf_sigma(
    r: np.ndarray,    # radial positions, pixels, shape (R,)
    r_max: float,     # maximum radius, pixels
    sigma0: float,    # average PSF width, pixels
    sigma1: float,    # sine variation amplitude, pixels
    sigma2: float,    # cosine variation amplitude, pixels
) -> np.ndarray:
    """
    Shift-variant Gaussian PSF width as a function of radius.

    σ(r) = σ₀ + σ₁·sin(π·r/r_max) + σ₂·cos(π·r/r_max)    (Harding Eq. 5)

    The PSF captures optical defects in the etalon and imaging lens that
    cause fringe broadening beyond the ideal Airy function. σ(r) must
    be positive everywhere for physically valid parameters.

    Returns
    -------
    sigma : PSF width in pixels, same shape as r. Always > 0 for valid inputs.
    """
    return sigma0 + sigma1 * np.sin(np.pi * r / r_max) \
                  + sigma2 * np.cos(np.pi * r / r_max)


def airy_modified(
    r: np.ndarray,        # radial positions, pixels, shape (R,)
    wavelength: float,    # wavelength, metres
    t: float,             # etalon gap, metres
    R_refl: float,        # plate reflectivity
    alpha: float,         # magnification constant, rad/pixel
    n: float,             # index of refraction
    r_max: float,         # maximum radius, pixels
    I0: float,            # average intensity
    I1: float,            # linear vignetting
    I2: float,            # quadratic vignetting
    sigma0: float,        # average PSF width, pixels
    sigma1: float,        # sine PSF variation
    sigma2: float,        # cosine PSF variation
) -> np.ndarray:
    """
    PSF-broadened Airy function at a single wavelength.

    Applies a shift-variant Gaussian convolution to the ideal Airy
    function. Uses the mean sigma across the profile as the filter width
    (mean-sigma approximation, accurate to < 1% for smooth sigma profiles).

    When sigma0 = sigma1 = sigma2 = 0, returns exactly airy_ideal().
    This is enforced by T3.

    Returns
    -------
    A_mod : PSF-broadened CCD counts, shape (R,)
    """
    A_ideal = airy_ideal(r, wavelength, t, R_refl, alpha, n, r_max, I0, I1, I2)
    sigma = psf_sigma(r, r_max, sigma0, sigma1, sigma2)
    sigma_mean = float(np.mean(sigma))
    if sigma_mean < 1e-6:
        return A_ideal
    # gaussian_filter1d sigma is in array elements; convert from pixel units
    r = np.asarray(r)
    dr = float((r[-1] - r[0]) / (len(r) - 1)) if len(r) > 1 else 1.0
    sigma_samples = sigma_mean / dr if dr > 0 else sigma_mean
    return gaussian_filter1d(A_ideal, sigma=sigma_samples)


def build_instrument_matrix(
    r: np.ndarray,           # radial bin centres, pixels, shape (R,)
    wavelengths: np.ndarray, # wavelength bin centres, metres, shape (L,)
    t: float,
    R_refl: float,
    alpha: float,
    n: float,
    r_max: float,
    I0: float,
    I1: float,
    I2: float,
    sigma0: float,
    sigma1: float,
    sigma2: float,
) -> np.ndarray:
    """
    Build the instrument matrix A of shape (R, L).

    Column j of A is airy_modified(r; λⱼ) for wavelength wavelengths[j].
    The forward model is:
        s = A @ y + B    (Harding Eq. 16)
    where y is the source spectrum (counts/m), B is the CCD bias vector.

    Parameters
    ----------
    r           : radial bin centres in pixels, shape (R,)
    wavelengths : wavelength bin centres in metres, shape (L,)
                  Use L=101 for inversion, L=300 for synthesis.

    Returns
    -------
    A : np.ndarray, shape (R, L)
        All values >= 0. No NaN or Inf for valid inputs.
    """
    R_bins = len(r)
    L_bins = len(wavelengths)
    A = np.zeros((R_bins, L_bins))
    # Scale each column by Δλ so that A @ y integrates: s = A @ y + B
    # where y is a spectral density (counts/m) and A has units counts·m.
    dlam = float(wavelengths[1] - wavelengths[0]) if L_bins > 1 else 1.0
    for j, lam in enumerate(wavelengths):
        A[:, j] = airy_modified(r, lam, t, R_refl, alpha, n,
                                r_max, I0, I1, I2, sigma0, sigma1, sigma2) * dlam
    return A


def make_wavelength_grid(
    center_wavelength: float,           # metres
    params: "InstrumentParams",
    n_fsr: float = 3.0,                 # FSR spans to cover
    L: int = 101,                       # number of wavelength bins
) -> np.ndarray:
    """
    Construct the wavelength grid for the instrument matrix.

    Spans ±(n_fsr/2) free spectral ranges about center_wavelength.

    Parameters
    ----------
    center_wavelength : metres. Use OI_WAVELENGTH_M for airglow (M04/M06),
                        NE_WAVELENGTH_1_M for calibration (M02/M05).
    n_fsr : number of FSRs to span. Default 3.0 covers the full beat
            pattern between the two neon lines.
    L     : number of bins. Use 101 for inversion, 300 for synthesis
            (anti-inverse-crime rule).

    Returns
    -------
    wavelengths : np.ndarray, shape (L,), units metres, monotonically increasing
    """
    fsr = params.free_spectral_range(center_wavelength)
    lam_min = center_wavelength - n_fsr * fsr / 2.0
    lam_max = center_wavelength + n_fsr * fsr / 2.0
    return np.linspace(lam_min, lam_max, L)
