"""
Module:      m03_airglow_synthesis_2026_05_24.py
Spec:        specs/M03_airglow_synthesis_lib_2026-05-13.md
             specs/M03_airglow_synthesis_addendum_2026-05-24.md
Author:      Claude Code
Generated:   2026-05-24
Last tested: 2026-05-24
Project:     WindCube FPI Pipeline — NCAR/HAO
Repo:        soc_sewell

Extends m03_airglow_synthesis_2026_05_13.py with thermal line broadening.

Changes from 2026_05_13:
  - Added delta_lambda_from_temperature(): Harding Eq. 12 public function.
  - Added _gaussian_source_spectrum(): Harding Eq. 10 private helper.
  - synthesise_airglow_image(): new use_temperature and T_true_K parameters.
    use_temperature=False (default) is byte-identical to existing behaviour.
  - L_synth and n_fsr un-deprecated: govern spectral grid when
    use_temperature=True.  n_fsr default changed 10.0 -> 5.0 (spec addendum).
  - Return dict: added T_true_K and delta_lambda_m keys (None when
    use_temperature=False).
"""

import numpy as np

from windcube.constants import (
    OI_WAVELENGTH_AIR_M,
    SPEED_OF_LIGHT_MS,
    V_REL_CROSSTRACK_MAX_MS,
    V_REL_ALONGTRACK_MIN_MS,
    V_REL_ALONGTRACK_MAX_MS,
    BOLTZMANN_J_PER_K,
    OXYGEN_MASS_KG,
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
# Observation mode bounds
# ---------------------------------------------------------------------------

OBSERVATION_MODE_BOUNDS = {
    "cross_track": (-V_REL_CROSSTRACK_MAX_MS, +V_REL_CROSSTRACK_MAX_MS),
    "along_track": (V_REL_ALONGTRACK_MIN_MS,  V_REL_ALONGTRACK_MAX_MS),
}


# ---------------------------------------------------------------------------
# add_gaussian_noise — retained for API compatibility; not used internally
# ---------------------------------------------------------------------------

def add_gaussian_noise(
    image_noiseless: np.ndarray,
    snr: float,
    profile_1d: np.ndarray,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    [DEPRECATED — use physical CCD noise model via synthesise_airglow_image()]

    Add Gaussian white noise to a noiseless airglow image at specified SNR.
    Retained for API compatibility with callers that import this function.

    Parameters
    ----------
    image_noiseless : float64 array of CCD counts, shape (N, N).
    snr             : signal-to-noise ratio (ΔS / σ_N). Must be > 0.
    profile_1d      : 1D noiseless fringe profile, shape (R,).
    rng             : numpy Generator for reproducibility.

    Returns
    -------
    image_noisy : np.ndarray, same shape as image_noiseless, float64.
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
# Dark current model — Teledyne e2v CCD97 datasheet, page 4
# ---------------------------------------------------------------------------

def _dark_rate_e2v(T_celsius: float, Qdd_at_20C: float = 400.0) -> float:
    """
    CCD97 dark signal.  Source: Teledyne e2v datasheet page 4.
    Returns dark signal in e-/pixel/s.
    Valid for T in [-123, +27] degrees C (150-300 K).

    Formula: Q_d(T) = Qdd * f(T_K) / f(T_ref)
             f(T)   = 1.14e6 * T^3 * exp(-9080 / T)
             T_ref  = 293.15 K  (20 degrees C reference)
    """
    T_k   = T_celsius + 273.15
    T_ref = 293.15
    _f    = lambda T: 1.14e6 * T ** 3 * np.exp(-9080.0 / T)
    return float(Qdd_at_20C * _f(T_k) / _f(T_ref))


# ---------------------------------------------------------------------------
# Thermal line-width helper — Harding Eq. 12
# ---------------------------------------------------------------------------

def delta_lambda_from_temperature(T_K: float, lambda0_m: float = None) -> float:
    """
    Harding Eq. 12: Doppler line half-width from thermospheric temperature.

        Δλ = (λ₀ / c) * sqrt(k * T / m)

    Parameters
    ----------
    T_K       : temperature in K. Must be > 0.
    lambda0_m : rest wavelength in metres.
                Default: OI_WAVELENGTH_AIR_M from windcube/constants.py.

    Returns
    -------
    delta_lambda_m : float, metres
    """
    if lambda0_m is None:
        lambda0_m = OI_WAVELENGTH_AIR_M
    if T_K <= 0:
        raise ValueError(f"T_K must be > 0; got {T_K}")
    return (lambda0_m / SPEED_OF_LIGHT_MS) * np.sqrt(BOLTZMANN_J_PER_K * T_K / OXYGEN_MASS_KG)


# ---------------------------------------------------------------------------
# Gaussian source spectrum helper — Harding Eq. 10
# ---------------------------------------------------------------------------

def _gaussian_source_spectrum(
    lambda_grid_m: np.ndarray,
    lambda_c_m: float,
    delta_lambda_m: float,
    Y_line: float,
    Y_bg: float,
) -> np.ndarray:
    """
    Harding Eq. 10: thermally broadened Gaussian source spectrum.

        Y(λ) = Y_bg + Y_line * exp(-0.5 * ((λ - λ_c) / Δλ)²)

    Parameters
    ----------
    lambda_grid_m  : wavelength grid, metres, shape (L,)
    lambda_c_m     : Doppler-shifted line centre, metres
    delta_lambda_m : Doppler line half-width (1/e), metres (Harding Eq. 12)
    Y_line         : line amplitude (dimensionless scale factor)
    Y_bg           : flat background emission (ADU)

    Returns
    -------
    Y : ndarray shape (L,) — source spectrum in ADU units
    """
    return Y_bg + Y_line * np.exp(
        -0.5 * ((lambda_grid_m - lambda_c_m) / delta_lambda_m) ** 2
    )


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
    n_fsr: float = 5.0,
    observation_mode: str = None,
    add_noise: bool = True,
    T_focal_plane_C: float = -20.0,
    exp_time_s: float = 10.0,
    gain_e_per_adu: float = 1.0,
    read_noise_e: float = 2.2,
    Qdd_at_20C: float = 400.0,
    rng: np.random.Generator = None,
    use_temperature: bool = False,
    T_true_K: float = 800.0,
) -> dict:
    """
    Generate a complete synthetic OI 630 nm airglow fringe image with
    physical CCD noise (Teledyne e2v CCD97 model).

    Signal chain (per pixel, when add_noise=True):
        signal_e  = image_ideal_ADU * gain_e_per_adu   [bias already embedded]
        dark_rate = _dark_rate_e2v(T_focal_plane_C, Qdd_at_20C)  [e-/pix/s]
        dark_e    = dark_rate * exp_time_s              [e- per pixel]
        total_e   = Poisson(signal_e + dark_e) + Normal(0, read_noise_e^2)
        pixel_ADU = round(total_e / gain_e_per_adu).clip(0, 65535).uint16

    bias_ADU is already in image_ideal_ADU (via params.B in profile_1d);
    it is NOT added again in the noise draw.

    Parameters
    ----------
    params           : InstrumentParams from H01.
    v_rel_ms         : line-of-sight velocity (m/s). Positive = recession.
    Y_line           : dimensionless Airy scale factor (default 1.0).
    Y_bg             : flat sky background added to all bins (ADU).
    image_size       : CCD dimension in pixels (default 256, 2x2 binned).
    cx, cy           : fringe centre in pixels (default: geometric centre).
    R_bins           : number of radial bins (default 500).
    L_synth          : spectral grid points for use_temperature=True path
                       (anti-inverse-crime: synthesise at 300, invert at 101).
    n_fsr            : spectral window half-width in FSRs for use_temperature=True.
    observation_mode : 'cross_track', 'along_track', or None.
    add_noise        : if False, return noiseless image (float64, no clipping).
    T_focal_plane_C  : focal-plane temperature in degrees C (default -20.0).
    exp_time_s       : exposure time in seconds (default 10.0).
    gain_e_per_adu   : CCD gain in e-/ADU (default 1.0).
    read_noise_e     : read noise rms in e- (default 2.2).
    Qdd_at_20C       : dark reference at 293 K in e-/pix/s (default 400.0).
    rng              : numpy Generator for reproducibility.
    use_temperature  : if True, use thermally broadened Gaussian source
                       (Harding Eqs. 10-12) instead of delta-function path.
    T_true_K         : true thermospheric temperature (K), used only when
                       use_temperature=True. Must be > 0.

    Returns
    -------
    dict with keys:
        image_2d            -- noisy uint16 (or noiseless float64 if not add_noise)
        image_noiseless     -- noiseless float64 image
        profile_1d          -- 1D radial profile (ADU), shape (R_bins,)
        r_grid              -- radial bin centres (pixels), shape (R_bins,)
        lambda_c_m          -- Doppler-shifted line centre (m)
        fringe_order_offset -- integer FSR offset from OI rest wavelength
        cx, cy              -- fringe centre (pixels)
        params              -- InstrumentParams (pass-through)
        v_rel_ms            -- input velocity (m/s)
        observation_mode    -- input mode string
        snr_actual          -- delta_S / sigma_mean (physical estimate)
        dark_rate_e_per_s   -- dark rate at T_focal_plane_C (e-/pix/s)
        dark_e_per_pixel    -- total dark per pixel (e-)
        T_focal_plane_C     -- input temperature (degrees C)
        exp_time_s          -- input exposure time (s)
        T_true_K            -- true temperature (K) if use_temperature else None
        delta_lambda_m      -- Doppler half-width (m) if use_temperature else None

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
                f"Valid: {list(OBSERVATION_MODE_BOUNDS)}"
            )
        lo, hi = OBSERVATION_MODE_BOUNDS[observation_mode]
        if not (lo <= v_rel_ms <= hi):
            raise ValueError(
                f"v_rel_ms={v_rel_ms:.1f} m/s outside '{observation_mode}' "
                f"bounds [{lo:.0f}, {hi:.0f}] m/s."
            )

    # Step 2: radial grid and RNG
    if cx is None:
        cx = (image_size - 1) / 2.0
    if cy is None:
        cy = (image_size - 1) / 2.0
    if rng is None:
        rng = np.random.default_rng()

    r_bins = np.linspace(0.0, params.r_max, R_bins)

    # Step 3: Doppler-shifted line centre
    lambda_c_m = OI_WAVELENGTH_AIR_M * (1.0 + v_rel_ms / SPEED_OF_LIGHT_MS)

    # Step 4: 1D fringe profile
    delta_lambda_m = None
    if not use_temperature:
        # --- Original delta-function path (unchanged) ---
        airy_profile = airy_modified(r_bins, lambda_c_m, params)
        profile_1d   = Y_line * airy_profile + Y_bg + params.B

    else:
        # --- Thermally broadened spectral integral path (Harding Eqs. 10-12) ---
        # Anti-inverse-crime: synthesise at L_synth=300, invert at L_invert=101
        delta_lambda_m = delta_lambda_from_temperature(T_true_K)

        # Spectral grid: n_fsr FSRs centred on lambda_c_m
        FSR_m    = OI_WAVELENGTH_AIR_M ** 2 / (2.0 * params.t)
        lam_lo   = lambda_c_m - (n_fsr / 2.0) * FSR_m
        lam_hi   = lambda_c_m + (n_fsr / 2.0) * FSR_m
        lam_grid = np.linspace(lam_lo, lam_hi, L_synth)   # shape (L_synth,)
        d_lam    = lam_grid[1] - lam_grid[0]

        # Source spectrum: Harding Eq. 10 (unnormalized shape).
        # Normalize the line component via the Riemann sum so that
        # integral(Y_line_part * d_lam) = Y_line, matching the delta-function
        # path's total line flux.  Without this the product d_lam (in metres)
        # makes the spectral-integral contribution ~1e-9 ADU — effectively zero.
        Y_lam_raw = _gaussian_source_spectrum(
            lam_grid, lambda_c_m, delta_lambda_m, Y_line, Y_bg
        )  # shape (L_synth,), unnormalized
        line_sum = ((Y_lam_raw - Y_bg) * d_lam).sum()
        if line_sum > 0:
            # Scale so the pure-line part integrates to Y_line
            Y_lam = Y_bg + (Y_lam_raw - Y_bg) * (Y_line / line_sum)
        else:
            Y_lam = Y_lam_raw

        # Vectorised Airy matrix: shape (R_bins, L_synth)
        # Harding Eq. 15: S(r_i) ≈ Σ_j Ã(r_i, λ_j) Y(λ_j) Δλ + B
        airy_matrix = np.stack(
            [airy_modified(r_bins, lj, params) for lj in lam_grid], axis=1
        )  # shape (R_bins, L_synth)
        profile_1d = airy_matrix @ (Y_lam * d_lam) + params.B

    # Step 5: 2D noiseless image (float64, bias embedded via params.B fill)
    image_noiseless = radial_profile_to_image(
        profile_1d, r_bins, image_size=image_size, cx=cx, cy=cy, bias=params.B
    )

    # Step 6: physical CCD noise model
    # Always compute dark_rate for the return dict; zero it for noiseless path.
    dark_rate = _dark_rate_e2v(T_focal_plane_C, Qdd_at_20C)
    dark_e    = dark_rate * exp_time_s

    if add_noise:
        signal_e = image_noiseless * gain_e_per_adu
        lambda_p = np.maximum(signal_e + dark_e, 0.0)   # Poisson requires >= 0
        total_e  = rng.poisson(lambda_p).astype(float)
        total_e += rng.normal(0.0, read_noise_e, size=image_noiseless.shape)
        # BIAS AUDIT: params.B is already embedded in image_noiseless via
        # profile_1d (Step 4 above). Do NOT add bias_ADU again.
        image_2d = np.round(
            total_e / gain_e_per_adu
        ).clip(0, 65535).astype(np.uint16)
    else:
        image_2d  = image_noiseless.copy()
        dark_rate = 0.0
        dark_e    = 0.0

    # Step 7: physical SNR diagnostic
    sigma_mean_adu = (np.sqrt(
        np.mean(image_noiseless) * gain_e_per_adu
        + dark_e
        + read_noise_e ** 2
    ) / gain_e_per_adu)
    delta_S    = float(profile_1d.max() - profile_1d.min())
    snr_actual = delta_S / sigma_mean_adu if sigma_mean_adu > 0 else np.inf

    # Step 8: fringe_order_offset (diagnostic)
    FSR_OI_M            = OI_WAVELENGTH_AIR_M ** 2 / (2.0 * params.t)
    fringe_order_offset = int(round((lambda_c_m - OI_WAVELENGTH_AIR_M) / FSR_OI_M))

    return {
        "image_2d":             image_2d,
        "image_noiseless":      image_noiseless,
        "profile_1d":           profile_1d,
        "r_grid":               r_bins,
        "lambda_c_m":           float(lambda_c_m),
        "fringe_order_offset":  fringe_order_offset,
        "cx":                   float(cx),
        "cy":                   float(cy),
        "params":               params,
        "v_rel_ms":             float(v_rel_ms),
        "observation_mode":     observation_mode,
        "snr_actual":           float(snr_actual),
        "dark_rate_e_per_s":    float(dark_rate),
        "dark_e_per_pixel":     float(dark_e),
        "T_focal_plane_C":      float(T_focal_plane_C),
        "exp_time_s":           float(exp_time_s),
        "T_true_K":             float(T_true_K) if use_temperature else None,
        "delta_lambda_m":       float(delta_lambda_m) if use_temperature else None,
    }


# ---------------------------------------------------------------------------
# Acceptance checks  M03-C1 through M03-C7  (spec §6)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("M03 acceptance checks  (spec §6)")
    print("=" * 60)

    # ---- M03-C2: dark_rate at T = -20 C ----------------------------------
    dr_minus20 = _dark_rate_e2v(-20.0)
    c2_pass = 1.90 <= dr_minus20 <= 2.05
    print(f"\nM03-C2  dark_rate at -20 C: {dr_minus20:.4f} e-/pix/s")
    print(f"        expected [1.90, 2.05]  ->  {'PASS' if c2_pass else 'FAIL'}")
    assert c2_pass, f"M03-C2 FAILED: dark_rate={dr_minus20:.4f}"

    # ---- M03-C3: dark_rate at T = +20 C ----------------------------------
    dr_plus20 = _dark_rate_e2v(+20.0)
    c3_pass = 390.0 <= dr_plus20 <= 420.0
    print(f"\nM03-C3  dark_rate at +20 C: {dr_plus20:.4f} e-/pix/s")
    print(f"        expected [390, 420]    ->  {'PASS' if c3_pass else 'FAIL'}")
    assert c3_pass, f"M03-C3 FAILED: dark_rate={dr_plus20:.4f}"

    # ---- Build H05-calibration InstrumentParams for C4-C7 ----------------
    params_h05 = InstrumentParams(
        t      = 20.1070e-3,
        R_refl = 0.239,
        alpha  = 1.6084e-4,
        I0     = 6479.9,
        I1     = -0.04684,
        I2     = -0.02599,
        sigma0 = 0.5528,
        sigma1 = 0.0,
        sigma2 = 0.0,
        B      = 2010.7,
        r_max  = 110.0,
    )

    rng_fixed = np.random.default_rng(42)

    # ---- M03-C4: snr_actual at T=-20 C, t=10 s, Y_line=1 (I0=6480) ------
    result_noisy = synthesise_airglow_image(
        params           = params_h05,
        v_rel_ms         = 0.0,
        Y_line           = 1.0,
        add_noise        = True,
        T_focal_plane_C  = -20.0,
        exp_time_s       = 10.0,
        gain_e_per_adu   = 1.0,
        read_noise_e     = 2.2,
        Qdd_at_20C       = 400.0,
        rng              = rng_fixed,
    )
    snr_val = result_noisy["snr_actual"]
    c4_pass = 30.0 <= snr_val <= 65.0
    print(f"\nM03-C4  snr_actual (T=-20C, t=10s, Y_line=1, I0=6480): {snr_val:.2f}")
    print(f"        expected [30, ~60]     ->  {'PASS' if c4_pass else 'FAIL'}")
    assert c4_pass, f"M03-C4 FAILED: snr_actual={snr_val:.2f}"

    # ---- M03-C5: add_noise=False returns noiseless image unchanged --------
    result_noiseless = synthesise_airglow_image(
        params           = params_h05,
        v_rel_ms         = 0.0,
        Y_line           = 1.0,
        add_noise        = False,
        T_focal_plane_C  = -20.0,
        exp_time_s       = 10.0,
        gain_e_per_adu   = 1.0,
        read_noise_e     = 2.2,
        Qdd_at_20C       = 400.0,
        rng              = np.random.default_rng(0),
    )
    c5_pass = np.all(
        result_noiseless["image_2d"] == result_noiseless["image_noiseless"]
    )
    print(f"\nM03-C5  add_noise=False => image_2d == image_noiseless: {c5_pass}")
    print(f"        expected True         ->  {'PASS' if c5_pass else 'FAIL'}")
    assert c5_pass, "M03-C5 FAILED: noiseless image differs from ideal"

    # ---- M03-C6: pixel dtype of noisy image is uint16 --------------------
    c6_pass = result_noisy["image_2d"].dtype == np.uint16
    print(f"\nM03-C6  image_2d.dtype: {result_noisy['image_2d'].dtype}")
    print(f"        expected uint16       ->  {'PASS' if c6_pass else 'FAIL'}")
    assert c6_pass, f"M03-C6 FAILED: dtype={result_noisy['image_2d'].dtype}"

    # ---- M03-C7: no pixel exceeds 65535 ----------------------------------
    max_val  = int(result_noisy["image_2d"].max())
    delta_S  = float(result_noisy["profile_1d"].max() - result_noisy["profile_1d"].min())
    c7_pass  = max_val <= 65535
    c7b_pass = 6000 <= max_val <= 12000
    c7c_pass = delta_S > 1000
    print(f"\nM03-C7  image_2d.max(): {max_val}")
    print(f"        expected <= 65535     ->  {'PASS' if c7_pass else 'FAIL'}")
    print(f"        fringe peak [6000,12000]: {max_val}  ->  {'PASS' if c7b_pass else 'FAIL'}")
    print(f"        fringe ΔS > 1000 ADU: {delta_S:.1f}  ->  {'PASS' if c7c_pass else 'FAIL'}")
    assert c7_pass,  f"M03-C7 FAILED: max={max_val}"
    assert c7b_pass, f"M03-C7b FAILED: peak={max_val} outside [6000,12000] (clipped or flat)"
    assert c7c_pass, f"M03-C7c FAILED: ΔS={delta_S:.1f} <= 1000 (no fringes)"

    # ---- delta_lambda_from_temperature spot check ------------------------
    dl = delta_lambda_from_temperature(800.0)
    dl_pass = 1.30e-12 <= dl <= 1.40e-12
    print(f"\nAddendum  delta_lambda(800 K): {dl:.4e} m")
    print(f"          expected [1.30e-12, 1.40e-12]  ->  {'PASS' if dl_pass else 'FAIL'}")
    assert dl_pass, f"delta_lambda(800K) FAILED: {dl:.4e}"

    # ---- M03-C1: note (requires H06 integration pipeline) ----------------
    print(f"\nM03-C1  chi2/nu from H06 inversion: NOT TESTED HERE")
    print(f"        Requires H03 + H06 integration run.")
    print(f"        Expected: 0.5 <= chi2/nu <= 2.0")

    # ---- Summary ---------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"Spot-checks:")
    print(f"  dark_rate at -20 C = {dr_minus20:.4f} e-/pix/s")
    print(f"  dark_rate at +20 C = {dr_plus20:.4f} e-/pix/s")
    print(f"  snr_actual (T=-20C, t=10s, Y_line=1, I0=6480) = {snr_val:.2f}")
    print(f"  image peak ADU     = {max_val}")
    print(f"  profile ΔS         = {delta_S:.1f} ADU")
    print(f"  delta_lambda(800K) = {dl:.4e} m")
    print(f"\nAll inline checks passed.")
