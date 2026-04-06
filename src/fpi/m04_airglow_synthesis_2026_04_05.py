"""
Module:      m04_airglow_synthesis_2026_04_05.py
Spec:        specs/S11_m04_airglow_synthesis_2026-04-05.md
Author:      Claude Code
Generated:   2026-04-05
Last tested: 2026-04-05  (8/8 tests pass)
Project:     WindCube FPI Pipeline — NCAR/HAO
Repo:        soc_sewell

Design note: OI 630 nm source is a spectral delta function.
             No temperature_K parameter. See spec Section 2.3.
"""

import numpy as np

from src.fpi.m01_airy_forward_model_2026_04_05 import (
    InstrumentParams,
    OI_WAVELENGTH_M,
    SPEED_OF_LIGHT_MS,
    airy_modified,
)
from src.fpi.m02_calibration_synthesis_2026_04_05 import radial_profile_to_image


def v_rel_to_lambda_c(
    v_rel_ms: float,
    lambda0_m: float = None,   # defaults to OI_WAVELENGTH_M from M01
) -> float:
    """
    Convert line-of-sight wind speed to Doppler-shifted line centre.

    λ_c = λ₀ · (1 + v_rel / c)    (Harding Eq. 11)

    Parameters
    ----------
    v_rel_ms  : LOS wind speed, m/s.
                Positive = emitter receding from instrument (redshift).
                Negative = emitter approaching (blueshift).
    lambda0_m : rest wavelength in metres. Default: OI_WAVELENGTH_M (S03/M01).

    Returns
    -------
    lambda_c_m : float, Doppler-shifted line centre in metres
    """
    if lambda0_m is None:
        lambda0_m = OI_WAVELENGTH_M
    return lambda0_m * (1.0 + v_rel_ms / SPEED_OF_LIGHT_MS)


def lambda_c_to_v_rel(
    lambda_c_m: float,
    lambda0_m: float = None,   # defaults to OI_WAVELENGTH_M
) -> float:
    """
    Recover line-of-sight wind speed from Doppler-shifted line centre.

    v_rel = c · (λ_c / λ₀ − 1)    (inverse of Harding Eq. 11)

    Used by M06 to convert recovered λ_c back to v_rel.
    Defined here (not in M06) to keep Doppler physics co-located.

    Returns
    -------
    v_rel_ms : float, m/s
    """
    if lambda0_m is None:
        lambda0_m = OI_WAVELENGTH_M
    return SPEED_OF_LIGHT_MS * (lambda_c_m / lambda0_m - 1.0)


def add_gaussian_noise(
    image_noiseless: np.ndarray,          # shape (N, N), float64
    snr: float,                           # target SNR = ΔS / σ_N
    rng: np.random.Generator = None,
) -> tuple:
    """
    Add Gaussian white noise to an airglow image at a specified SNR.

    SNR = ΔS / σ_N  where ΔS = max(image) − min(image).
    σ_N = ΔS / SNR.

    Parameters
    ----------
    image_noiseless : noiseless 2D airglow image, counts
    snr             : target signal-to-noise ratio (Harding Eq. 17)

    Returns
    -------
    (image_noisy, sigma_noise) :
        image_noisy  : np.ndarray, same shape, with Gaussian noise added
        sigma_noise  : float, the noise std dev actually used (counts)
    """
    if rng is None:
        rng = np.random.default_rng()

    delta_s = float(np.max(image_noiseless) - np.min(image_noiseless))
    sigma_n = delta_s / snr
    noise = rng.normal(0.0, sigma_n, image_noiseless.shape)
    image_noisy = image_noiseless + noise
    return image_noisy.astype(np.float64), float(sigma_n)


def synthesise_airglow_image(
    v_rel_ms: float,
    params: "InstrumentParams",
    snr: float = 5.0,
    I_line: float = 1.0,
    Y_bg: float = 0.05,
    image_size: int = 256,
    cx: float = None,
    cy: float = None,
    R_bins: int = 500,
    add_noise: bool = True,
    rng: np.random.Generator = None,
    **kwargs,
) -> dict:
    """
    Generate a synthetic OI 630.0 nm airglow fringe image.

    Source model: spectral delta function at Doppler-shifted wavelength λ_c.
    No temperature broadening. No wavelength grid. One airy_modified() call.

    This function DOES NOT accept a temperature_K argument.
    Passing temperature_K will raise TypeError (enforced by **kwargs guard).

    Parameters
    ----------
    v_rel_ms   : LOS wind speed from NB02c, m/s. Encodes Doppler shift.
    params     : InstrumentParams from M01/M05.
    snr        : target SNR = ΔS / σ_N. Typical range: 0.5–10.
    I_line     : airglow line intensity scale factor. Default 1.0.
    Y_bg       : fractional background (relative to I_line). Default 0.05.
                 Background adds a constant offset: B_bg = Y_bg × I_line × I0.
    image_size : CCD active pixels, one side. Default 256.
    cx, cy     : fringe centre, pixels. Default: geometric centre.
    R_bins     : radial bins in 1D profile. Default 500.
    add_noise  : if True, add Gaussian noise at specified SNR. Default True.
    rng        : numpy Generator for reproducibility.

    Returns
    -------
    dict with keys:
        'image_2d'        : np.ndarray (image_size, image_size) — noisy image
        'image_noiseless' : np.ndarray (image_size, image_size) — noiseless
        'profile_1d'      : np.ndarray (R_bins,) — 1D noiseless profile
        'r_grid'          : np.ndarray (R_bins,) — radial bin centres, pixels
        'lambda_c_m'      : float — Doppler-shifted line centre used
        'sigma_noise'     : float — noise std dev applied (0.0 if no noise)
        'snr_actual'      : float — actual SNR of noiseless fringe
        'v_rel_ms'        : float — v_rel used (stored for round-trip testing)
        'cx'              : float — fringe centre x used
        'cy'              : float — fringe centre y used
        'params'          : InstrumentParams used

    Notes
    -----
    Uses radial_profile_to_image() imported from M02.
    All constants (OI_WAVELENGTH_M, SPEED_OF_LIGHT_MS) imported from M01.
    """
    # Enforce no-temperature rule (S11 Section 2.3 and 3.4)
    if "temperature_K" in kwargs:
        raise TypeError(
            "synthesise_airglow_image() does not accept temperature_K. "
            "The OI 630 nm source is modelled as a delta function. "
            "See S11 Section 2.3 for the design rationale."
        )

    # Resolve defaults
    if cx is None:
        cx = (image_size - 1) / 2.0
    if cy is None:
        cy = (image_size - 1) / 2.0

    # Doppler-shifted line centre
    lambda_c_m = v_rel_to_lambda_c(v_rel_ms)

    # 1D radial grid
    r_grid = np.linspace(0.0, params.r_max, R_bins)

    # 1D fringe profile: one airy_modified call at Doppler-shifted wavelength
    # S(r) = I_line × airy_modified(r; λ_c, params)
    #       + Y_bg × I_line × params.I0
    #       + params.B
    airy = airy_modified(
        r_grid,
        lambda_c_m,
        params.t,
        params.R_refl,
        params.alpha,
        params.n,
        params.r_max,
        params.I0,
        params.I1,
        params.I2,
        params.sigma0,
        params.sigma1,
        params.sigma2,
    )
    profile_1d = I_line * airy + Y_bg * I_line * params.I0 + params.B

    # 2D noiseless image (shared function from M02)
    image_noiseless = radial_profile_to_image(
        profile_1d, r_grid, image_size=image_size, cx=cx, cy=cy, bias=params.B
    )

    # SNR accounting
    delta_s = float(np.max(image_noiseless) - np.min(image_noiseless))
    sigma_noise_val = delta_s / snr   # σ_N = ΔS / target_snr
    snr_actual = delta_s / sigma_noise_val  # = target_snr by construction

    # Optionally add Gaussian noise
    if add_noise:
        image_2d, sigma_noise_applied = add_gaussian_noise(
            image_noiseless, snr=snr, rng=rng
        )
    else:
        image_2d = image_noiseless.copy()
        sigma_noise_applied = 0.0

    return {
        "image_2d": image_2d,
        "image_noiseless": image_noiseless,
        "profile_1d": profile_1d,
        "r_grid": r_grid,
        "lambda_c_m": float(lambda_c_m),
        "sigma_noise": float(sigma_noise_applied),
        "snr_actual": float(snr_actual),
        "v_rel_ms": float(v_rel_ms),
        "cx": float(cx),
        "cy": float(cy),
        "params": params,
    }
