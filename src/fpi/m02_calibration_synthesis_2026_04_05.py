"""
Module:      m02_calibration_synthesis_2026_04_05.py
Spec:        specs/S10_m02_calibration_synthesis_2026-04-05.md
Author:      Claude Code
Generated:   2026-04-05
Last tested: 2026-04-05  (8/8 tests pass)
Project:     WindCube FPI Pipeline — NCAR/HAO
Repo:        soc_sewell

M02 generates a synthetic 2D CCD calibration fringe image from the WindCube
neon emission lamp. It wraps M01 airy_modified() calls for both neon lines,
superimposes them, converts to a 2D image via circular symmetry, and adds
Poisson photon noise.

The radial_profile_to_image() function defined here is shared with M04 —
do not duplicate it in M04; import from here.

All Ne constants imported from M01 (single source of truth).
"""

import numpy as np

from src.fpi.m01_airy_forward_model_2026_04_05 import (
    InstrumentParams,
    NE_INTENSITY_2,
    NE_WAVELENGTH_1_M,
    NE_WAVELENGTH_2_M,
    airy_modified,
)


def radial_profile_to_image(
    profile_1d: np.ndarray,   # S(r), shape (R,), CCD counts
    r_grid: np.ndarray,       # radial bin centres, pixels, shape (R,)
    image_size: int = 256,    # CCD active dimension, pixels
    cx: float = None,         # fringe centre x (default: (image_size-1)/2)
    cy: float = None,         # fringe centre y (default: (image_size-1)/2)
    bias: float = 300.0,      # value for pixels beyond r_max
) -> np.ndarray:
    """
    Wrap a 1D radial fringe profile into a 2D CCD image.

    For each pixel (row, col), compute r = sqrt((col-cx)²+(row-cy)²),
    then linearly interpolate profile_1d at r. Pixels beyond max(r_grid)
    are set to `bias`.

    This function is shared between M02 and M04. M04 imports it from here.
    Do not duplicate it.

    Parameters
    ----------
    profile_1d : 1D radial fringe profile in CCD counts, shape (R,)
    r_grid     : radial bin centres in pixels, shape (R,). Must start near 0.
    image_size : CCD active pixels along one side. Default 256 (2×2 binned).
    cx, cy     : fringe centre coordinates in pixels.
                 Default: (image_size - 1) / 2.0  (geometric centre)
    bias       : fill value for pixels outside r_grid range.

    Returns
    -------
    image : np.ndarray, shape (image_size, image_size), float64
    """
    if cx is None:
        cx = (image_size - 1) / 2.0
    if cy is None:
        cy = (image_size - 1) / 2.0

    cols = np.arange(image_size, dtype=float)
    rows = np.arange(image_size, dtype=float)
    col_grid, row_grid = np.meshgrid(cols, rows)  # shape (N, N)

    r_pixel = np.sqrt((col_grid - cx) ** 2 + (row_grid - cy) ** 2)

    # np.interp: out-of-range values use left/right fill; set both to bias
    r_max = float(r_grid[-1])
    image = np.interp(r_pixel, r_grid, profile_1d, left=bias, right=bias)

    return image.astype(np.float64)


def add_poisson_noise(
    image_noiseless: np.ndarray,          # shape (N, N), float64, counts >= 0
    rng: np.random.Generator = None,      # default_rng() if None
) -> np.ndarray:
    """
    Add Poisson photon noise to a noiseless CCD image.

    Each pixel value v is replaced by a sample from Poisson(λ=v).
    Values < 0 are clipped to 0 before sampling (physically required).

    The neon calibration image is photon-noise limited — no dark current
    or read noise term is needed for the calibration frame.

    Parameters
    ----------
    image_noiseless : float64 array, CCD counts. Must be non-negative.
    rng             : numpy Generator. Pass default_rng(seed) for reproducibility.
                      If None, uses np.random.default_rng().

    Returns
    -------
    image_noisy : np.ndarray, same shape as image_noiseless, float64
    """
    if rng is None:
        rng = np.random.default_rng()

    clipped = np.clip(image_noiseless, 0.0, None)
    return rng.poisson(clipped).astype(np.float64)


def synthesise_calibration_image(
    params: "InstrumentParams",           # from M01
    image_size: int = 256,               # CCD dimension, pixels
    cx: float = None,                    # fringe centre x (default: geometric centre)
    cy: float = None,                    # fringe centre y (default: geometric centre)
    R_bins: int = 500,                   # radial bins in 1D profile
    add_noise: bool = True,              # add Poisson noise
    rng: np.random.Generator = None,
) -> dict:
    """
    Generate a complete synthetic neon lamp calibration fringe image.

    Calls airy_modified() at NE_WAVELENGTH_1_M and NE_WAVELENGTH_2_M,
    superimposes with relative intensities, adds bias, wraps to 2D,
    optionally adds Poisson noise.

    The 1D profile formula (no wavelength grid, no matrix multiply):
        S_cal(r) = airy_modified(r; NE_WAVELENGTH_1_M, params)
                 + NE_INTENSITY_2 × airy_modified(r; NE_WAVELENGTH_2_M, params)
                 + params.B

    Parameters
    ----------
    params     : InstrumentParams from M01.
    image_size : CCD active dimension in pixels. Default 256 (2×2 binned).
    cx, cy     : fringe centre in pixels. Default: geometric centre.
    R_bins     : number of radial bins in 1D profile. Default 500.
    add_noise  : if True, add Poisson photon noise. Default True.
    rng        : numpy Generator for reproducibility.

    Returns
    -------
    dict with keys:
        'image_2d'        : np.ndarray (image_size, image_size) — noisy image
        'image_noiseless' : np.ndarray (image_size, image_size) — noiseless image
        'profile_1d'      : np.ndarray (R_bins,) — 1D fringe profile (no noise)
        'r_grid'          : np.ndarray (R_bins,) — radial bin centres, pixels
        'cx'              : float — fringe centre x used
        'cy'              : float — fringe centre y used
        'params'          : InstrumentParams used (for M05 reference)
    """
    # Resolve defaults
    if cx is None:
        cx = (image_size - 1) / 2.0
    if cy is None:
        cy = (image_size - 1) / 2.0

    # 1D radial grid
    r_grid = np.linspace(0.0, params.r_max, R_bins)

    # 1D fringe profile: two Ne lines superimposed, plus bias
    A1 = airy_modified(
        r_grid,
        NE_WAVELENGTH_1_M,
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
    A2 = airy_modified(
        r_grid,
        NE_WAVELENGTH_2_M,
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
    profile_1d = A1 + NE_INTENSITY_2 * A2 + params.B

    # 2D noiseless image
    image_noiseless = radial_profile_to_image(
        profile_1d, r_grid, image_size=image_size, cx=cx, cy=cy, bias=params.B
    )

    # Optionally add Poisson noise
    if add_noise:
        image_2d = add_poisson_noise(image_noiseless, rng=rng)
    else:
        image_2d = image_noiseless.copy()

    return {
        "image_2d": image_2d,
        "image_noiseless": image_noiseless,
        "profile_1d": profile_1d,
        "r_grid": r_grid,
        "cx": cx,
        "cy": cy,
        "params": params,
    }
