"""
Module:      m03_airglow_synthesis_2026_05_05.py
Spec:        docs/specs/H03_airglow_synthesis_2026-05-05.md
Author:      Claude Code
Generated:   2026-05-05
Last tested: 2026-05-05
Project:     WindCube FPI Pipeline — NCAR/HAO
Repo:        soc_sewell

Synthesises a 2D OI 630 nm airglow fringe image following Harding (2014)
Eqs. 10–11, 14–16 (instrument matrix path). Supports two observation regimes:
  - Cross-track (even orbits): v_rel in [−1000, +1000] m/s; fringe_order_offset=0
  - Along-track (odd orbits):  v_rel in [−8000, −6000] m/s; fringe_order_offset=−1 or −2

Key design choices:
  - L_synth=300; H06 inversion uses L=101 (anti-inverse-crime, H01 §2.4)
  - n_fsr=10 default covers both regimes with ~3× margin
  - Delta-function OI source; Harding Eq. 12 (temperature) excluded
  - radial_profile_to_image imported from H02; not reimplemented
  - Gaussian noise default (dark-noise dominated), per Harding §4
  - Source spectrum built inline (not via make_airglow_spectrum) so that
    along-track velocities down to −8000 m/s work without hitting H01's
    v_rel bounds check, which is enforced at −7700 m/s.
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
    build_instrument_matrix,
    make_wavelength_grid,
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
    Y_line: float = 1000.0,
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

    Follows Harding (2014) Eqs. 14–16 (instrument matrix path):
      1. Validate v_rel_ms against observation_mode bounds (if provided).
      2. Build r_bins (R_bins uniform points from 0 to params.r_max).
      3. Build lam_grid centred on OI_WAVELENGTH_AIR_M, spanning n_fsr FSRs
         with L_synth bins (anti-inverse-crime: L_synth=300 ≠ L_inv=101).
      4. Build instrument matrix A = build_instrument_matrix(r_bins,
         lam_grid, params, n_subpixels=8) — shape (R_bins, L_synth).
      5. Build source spectrum y (delta-function at Doppler-shifted λ_c).
      6. Compute 1D profile: s = A @ y + params.B.
      7. Wrap to 2D: image = radial_profile_to_image(s, r_bins, ...).
      8. Optionally add noise.
      9. Compute fringe_order_offset and snr_actual; assemble output dict.

    Parameters
    ----------
    params     : InstrumentParams from H01.
    v_rel_ms   : line-of-sight velocity (m/s). Positive = recession (redshift).
    Y_line     : OI line intensity in ADU per wavelength bin. Default 1000.
    Y_bg       : spectrally flat sky background in ADU per bin. Default 0.
    image_size : CCD active dimension in pixels. Default 256 (2×2 binned).
    cx, cy     : fringe centre in pixels. Default: geometric centre.
    R_bins     : number of radial bins. Default 500.
    L_synth    : wavelength bins for synthesis. Default 300.
    n_fsr      : FSRs spanned by the wavelength grid. Default 10.
    observation_mode : 'cross_track', 'along_track', or None (default).
    add_noise  : if True, add noise per noise_type. Default True.
    noise_type : 'gaussian' (default) or 'poisson'.
    snr        : SNR = ΔS/σ_N for Gaussian noise. Default 5.
    rng        : numpy Generator for reproducibility.

    Returns
    -------
    dict with keys:
        'image_2d', 'image_noiseless', 'profile_1d', 'r_grid',
        'lam_grid', 'lambda_c_m', 'fringe_order_offset',
        'cx', 'cy', 'params', 'v_rel_ms', 'observation_mode', 'snr_actual'

    Raises
    ------
    ValueError
        If v_rel_ms violates the observation_mode bounds.
        If λ_c falls outside lam_grid.
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

    # Step 3: wavelength grid
    lam_grid = make_wavelength_grid(
        OI_WAVELENGTH_AIR_M, params, n_fsr=n_fsr, L=L_synth
    )

    # Step 4: instrument matrix — n_subpixels=8 per H03 spec (anti-inverse-crime)
    A = build_instrument_matrix(r_bins, lam_grid, params, n_subpixels=8)

    # Step 5: Doppler-shifted line centre and source spectrum
    # Built inline (not via make_airglow_spectrum) so that along-track
    # velocities to -8000 m/s work without hitting H01's -7700 m/s bound.
    lambda_c_m = OI_WAVELENGTH_AIR_M * (1.0 + v_rel_ms / SPEED_OF_LIGHT_MS)

    lam_min, lam_max = lam_grid[0], lam_grid[-1]
    if not (lam_min <= lambda_c_m <= lam_max):
        raise ValueError(
            f"Doppler-shifted λ_c={lambda_c_m * 1e9:.6f} nm outside "
            f"lam_grid [{lam_min * 1e9:.6f}, {lam_max * 1e9:.6f}] nm. "
            f"Increase n_fsr (currently {n_fsr})."
        )

    y = np.full(L_synth, Y_bg)
    idx_c = int(np.argmin(np.abs(lam_grid - lambda_c_m)))
    y[idx_c] += Y_line

    # Step 6: 1D fringe profile (Harding Eq. 16)
    profile_1d = A @ y + params.B

    # Step 7: 2D noiseless image
    image_noiseless = radial_profile_to_image(
        profile_1d, r_bins, image_size=image_size, cx=cx, cy=cy, bias=params.B
    )

    # Step 8: noise
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

    # Step 9: fringe_order_offset (diagnostic only, does not affect computation)
    FSR_OI_M = OI_WAVELENGTH_AIR_M ** 2 / (2.0 * params.t)
    fringe_order_offset = int(round((lambda_c_m - OI_WAVELENGTH_AIR_M) / FSR_OI_M))

    return {
        "image_2d": image_2d,
        "image_noiseless": image_noiseless,
        "profile_1d": profile_1d,
        "r_grid": r_bins,
        "lam_grid": lam_grid,
        "lambda_c_m": float(lambda_c_m),
        "fringe_order_offset": fringe_order_offset,
        "cx": float(cx),
        "cy": float(cy),
        "params": params,
        "v_rel_ms": float(v_rel_ms),
        "observation_mode": observation_mode,
        "snr_actual": float(snr_actual),
    }
