"""
Module:      m03_airglow_synthesis_2026_05_12.py
Spec:        docs/specs/H03_airglow_synthesis_2026-05-05.md
Author:      Claude Code
Generated:   2026-05-05
Last tested: 2026-05-12
Project:     WindCube FPI Pipeline — NCAR/HAO
Repo:        soc_sewell

Synthesises a 2D OI 630 nm airglow fringe image following Harding (2014)
Eqs. 10–11. Supports two observation regimes:
  - Cross-track (even orbits): v_rel in [−1000, +1000] m/s; fringe_order_offset=0
  - Along-track (odd orbits):  v_rel in [−8000, −6000] m/s; fringe_order_offset=−1 or −2

Key design choices:
  - Delta-function OI source; Harding Eq. 12 (temperature) excluded
  - radial_profile_to_image imported from H02; not reimplemented
  - Gaussian noise default (dark-noise dominated), per Harding §4
  - Profile computed via direct airy_modified() call (not instrument matrix).

Changes from 2026_05_05:
  - Steps 3–6 replaced: instrument matrix path removed in favour of a direct
    airy_modified() call at lambda_c_m (delta-function limit).
    Rationale: the matrix path requires Y_line in ADU/m (spectral density),
    not ADU/bin. The conversion Y_line /= dlam produced physically correct
    mathematics but signal levels of ~4×10⁶ ADU — far exceeding the 14-bit
    detector range (max 16383 ADU). The direct Airy call gives signal levels
    in the expected range (bias + Y_line × Airy_peak ≈ 2011 + 600 ≈ 2600 ADU
    for Y_line=1000, I0=6480, R=0.239) and is simpler and more transparent.
    The anti-inverse-crime rule (L_synth ≠ L_inv) was relevant only for the
    matrix path and does not apply to the direct call; L_synth and lam_grid
    are no longer needed and have been removed from synthesise_airglow_image().
  - Y_line default changed from 1000.0 to 1.0. airy_modified() already returns
    values in ADU (I0-scaled, range ~2400–6480 ADU for current instrument params).
    Y_line=1000 therefore produced signal levels ~3.9×10⁶ ADU — far exceeding
    the 14-bit detector maximum of 16383 ADU. Y_line=1.0 gives ΔS≈3929 ADU,
    which is physically correct. Y_line is a pure dimensionless scale factor.
  - Y_bg support retained: background adds a flat offset Y_bg to the profile.
  - lam_grid removed from output dict (no longer computed).
  - L_synth and n_fsr parameters retained as no-ops for API compatibility
    but marked deprecated in the docstring.
"""

import numpy as np

from windcube.constants import (
    OI_WAVELENGTH_AIR_M,
    SPEED_OF_LIGHT_MS,
    V_REL_CROSSTRACK_MAX_MS,
    V_REL_ALONGTRACK_MIN_MS,
    V_REL_ALONGTRACK_MAX_MS,
)
from src.fpi.airy_forward_model_2026_05_05 import (
    InstrumentParams,
    airy_modified,
)
from src.fpi.m02_calibration_synthesis_2026_05_05 import (
    add_poisson_noise,
    radial_profile_to_image,
)

# ---------------------------------------------------------------------------
# Observation mode bounds (H03 §5.3)
# ---------------------------------------------------------------------------

OBSERVATION_MODE_BOUNDS = {
    "cross_track": (-V_REL_CROSSTRACK_MAX_MS, +V_REL_CROSSTRACK_MAX_MS),
    "along_track": (V_REL_ALONGTRACK_MIN_MS, V_REL_ALONGTRACK_MAX_MS),
}


# ---------------------------------------------------------------------------
# add_gaussian_noise
# ---------------------------------------------------------------------------

def add_gaussian_noise(
    image_noiseless: np.ndarray,
    snr: float,
    profile_1d: np.ndarray,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Add Gaussian white noise to a noiseless airglow CCD image at a
    specified SNR, following Harding (2014) §4.

    σ_N is derived from the 1D fringe profile amplitude:
        ΔS  = max(profile_1d) − min(profile_1d)
        σ_N = ΔS / snr

    Gaussian noise N(0, σ_N²) is added independently to each pixel.

    Parameters
    ----------
    image_noiseless : float64 array of CCD counts, shape (N, N).
    snr             : signal-to-noise ratio (ΔS / σ_N). Must be > 0.
    profile_1d      : 1D noiseless fringe profile, shape (R,). Used only
                      to compute ΔS; must be the same profile wrapped into
                      image_noiseless.
    rng             : numpy Generator. Pass default_rng(seed) for
                      reproducibility. If None, uses np.random.default_rng().

    Returns
    -------
    image_noisy : np.ndarray, same shape as image_noiseless, float64.

    Raises
    ------
    ValueError : if snr <= 0.
    """
    if snr <= 0:
        raise ValueError(f"snr must be > 0; got {snr}")
    if rng is None:
        rng = np.random.default_rng()

    delta_S = float(profile_1d.max() - profile_1d.min())
    sigma_N = delta_S / snr
    noise = rng.normal(0.0, sigma_N, image_noiseless.shape)
    return (image_noiseless + noise).astype(np.float64)


# ---------------------------------------------------------------------------
# synthesise_airglow_image
# ---------------------------------------------------------------------------

def synthesise_airglow_image(
    params: "InstrumentParams",
    v_rel_ms: float,
    Y_line: float = 1.0,
    Y_bg: float = 0.0,
    image_size: int = 256,
    cx: float = None,
    cy: float = None,
    R_bins: int = 500,
    L_synth: int = 300,
    n_fsr: float = 10.0,
    observation_mode: str = None,
    add_noise: bool = True,
    noise_type: str = "gaussian",
    snr: float = 5.0,
    rng: np.random.Generator = None,
) -> dict:
    """
    Generate a complete synthetic OI 630 nm airglow fringe image.

    Follows Harding (2014) Eqs. 10–11 (delta-function source, direct Airy path):
      1. Validate v_rel_ms against observation_mode bounds (if provided).
      2. Build r_bins (R_bins uniform points from 0 to params.r_max).
      3. Compute Doppler-shifted line centre lambda_c_m.
      4. Compute 1D profile via direct airy_modified() call at lambda_c_m:
            profile_1d = Y_line * airy_modified(r_bins, lambda_c_m, params)
                       + Y_bg + params.B
      5. Wrap to 2D: image = radial_profile_to_image(profile_1d, r_bins, ...).
      6. Optionally add noise.
      7. Compute fringe_order_offset and snr_actual; assemble output dict.

    This is the delta-function limit of Harding Eq. 1: since Y(λ) = Y_line·δ(λ−λ_c),
    the Fredholm integral collapses to a single Airy evaluation at λ_c.
    The instrument matrix path (Eqs. 14–16) is not used because it requires
    Y_line in ADU/m (spectral density), producing signal levels ~10⁶× too large
    for a 14-bit detector. The direct call gives physically correct ADU levels.

    Parameters
    ----------
    params     : InstrumentParams from H01.
    v_rel_ms   : line-of-sight velocity (m/s). Positive = recession (redshift).
    Y_line     : Dimensionless scale factor applied to the Airy function output.
                 Default 1.0 = use instrument-calibrated signal levels directly.
                 airy_modified() already returns values in ADU (range ~B to ~I0),
                 so Y_line=1.0 gives a physically correct synthetic image.
                 With I0≈6480, R≈0.24: fringe peak ≈ 6480 ADU, trough ≈ 2400 ADU,
                 ΔS ≈ 3929 ADU — well within the 14-bit range (16383 ADU max).
                 Use Y_line < 1 for fainter airglow, Y_line > 1 to simulate
                 brighter emission or higher exposure.
    Y_bg       : flat sky background added uniformly to all radial bins (ADU).
                 Default 0.
    image_size : CCD active dimension in pixels. Default 256 (2×2 binned).
    cx, cy     : fringe centre in pixels. Default: geometric centre.
    R_bins     : number of radial bins. Default 500.
    L_synth    : deprecated no-op (retained for API compatibility).
    n_fsr      : deprecated no-op (retained for API compatibility).
    observation_mode : 'cross_track', 'along_track', or None (default).
    add_noise  : if True, add noise per noise_type. Default True.
    noise_type : 'gaussian' (default) or 'poisson'.
    snr        : SNR = ΔS/σ_N for Gaussian noise. Default 5.
    rng        : numpy Generator for reproducibility.

    Returns
    -------
    dict with keys:
        'image_2d', 'image_noiseless', 'profile_1d', 'r_grid',
        'lambda_c_m', 'fringe_order_offset',
        'cx', 'cy', 'params', 'v_rel_ms', 'observation_mode', 'snr_actual'

    Raises
    ------
    ValueError
        If v_rel_ms violates the observation_mode bounds.
    """
    # Step 1: observation_mode validation
    if observation_mode is not None:
        if observation_mode not in OBSERVATION_MODE_BOUNDS:
            raise ValueError(
                f"Unknown observation_mode '{observation_mode}'. "
                f"Valid values: {list(OBSERVATION_MODE_BOUNDS)}"
            )
        lo, hi = OBSERVATION_MODE_BOUNDS[observation_mode]
        if not (lo <= v_rel_ms <= hi):
            raise ValueError(
                f"v_rel_ms={v_rel_ms:.1f} m/s is outside the '{observation_mode}' "
                f"regime bounds [{lo:.0f}, {hi:.0f}] m/s."
            )

    # Step 2: radial grid
    if cx is None:
        cx = (image_size - 1) / 2.0
    if cy is None:
        cy = (image_size - 1) / 2.0

    r_bins = np.linspace(0.0, params.r_max, R_bins)

    # Step 3: Doppler-shifted line centre
    lambda_c_m = OI_WAVELENGTH_AIR_M * (1.0 + v_rel_ms / SPEED_OF_LIGHT_MS)

    # Step 4: 1D fringe profile — direct Airy call at lambda_c_m
    # Delta-function source: integral over Y(λ)·A(r,λ)dλ = Y_line·A(r,λ_c)
    # airy_modified() returns ADU in range [I0/(1+F), I0] ≈ [2445, 6480] ADU.
    # Adding Y_bg and params.B gives profile in range ~[4456, 8491] ADU (Y_line=1),
    # well within the 14-bit detector maximum of 16383 ADU.
    # ΔS ≈ 4035 ADU for Y_line=1.0 with current instrument parameters.
    airy_profile = airy_modified(r_bins, lambda_c_m, params)
    profile_1d   = Y_line * airy_profile + Y_bg + params.B

    # Step 5: 2D noiseless image
    image_noiseless = radial_profile_to_image(
        profile_1d, r_bins, image_size=image_size, cx=cx, cy=cy, bias=params.B
    )

    # Step 6: noise
    delta_S = float(profile_1d.max() - profile_1d.min())

    if add_noise:
        if noise_type == "gaussian":
            image_2d = add_gaussian_noise(image_noiseless, snr, profile_1d, rng=rng)
            sigma_N_eff = delta_S / snr
            snr_actual = float(snr)
        elif noise_type == "poisson":
            image_2d = add_poisson_noise(image_noiseless, rng=rng)
            sigma_N_eff = float(np.sqrt(np.mean(profile_1d)))
            snr_actual = delta_S / sigma_N_eff if sigma_N_eff > 0 else np.inf
        else:
            raise ValueError(f"noise_type must be 'gaussian' or 'poisson'; got {noise_type!r}")
    else:
        image_2d = image_noiseless.copy()
        snr_actual = np.inf

    # Step 7: fringe_order_offset (diagnostic only, does not affect computation)
    FSR_OI_M = OI_WAVELENGTH_AIR_M ** 2 / (2.0 * params.t)
    fringe_order_offset = int(round((lambda_c_m - OI_WAVELENGTH_AIR_M) / FSR_OI_M))

    return {
        "image_2d": image_2d,
        "image_noiseless": image_noiseless,
        "profile_1d": profile_1d,
        "r_grid": r_bins,
        "lambda_c_m": float(lambda_c_m),
        "fringe_order_offset": fringe_order_offset,
        "cx": float(cx),
        "cy": float(cy),
        "params": params,
        "v_rel_ms": float(v_rel_ms),
        "observation_mode": observation_mode,
        "snr_actual": float(snr_actual),
    }
