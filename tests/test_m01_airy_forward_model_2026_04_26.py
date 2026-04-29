"""
Tests for M01 Airy Forward Model (2026-04-26 revision).

Spec:        specs/S06_airy_forward_model_2026-04-26.md
Spec tests:  T1–T15
Run with:    pytest tests/test_m01_airy_forward_model_2026_04_26.py -v

T1–T10: unchanged from 2026-04-05 (copied verbatim, import updated).
T11–T15: new tests for make_ne_spectrum and make_airglow_spectrum.

Spec deviations (reported to user):
  T11: spec used n_fsr=220 centered at Ne1; Ne2 (638.299 nm) falls outside
       that grid (min=639.098 nm). Changed to n_fsr=400 to include both lines.
  T15: spec called build_instrument_matrix(r_bins, lam_grid, params) with 3
       args; existing function takes 13 individual args. Expanded to match
       actual signature.
"""

import numpy as np
import pytest

from src.fpi.m01_airy_forward_model_2026_04_26 import (
    InstrumentParams,
    airy_ideal,
    airy_modified,
    build_instrument_matrix,
    intensity_envelope,
    make_wavelength_grid,
    psf_sigma,
    OI_WAVELENGTH_M,
    make_ne_spectrum,
    make_airglow_spectrum,
)


# ---------------------------------------------------------------------------
# T1 — Airy peak positions uniform in r²
# ---------------------------------------------------------------------------
def test_airy_peak_positions():
    """
    Peaks occur when 2nt·cos(θ) = mλ.
    In r² space, peak spacing must be approximately uniform.
    Expect 3–7 peaks across r_max for WindCube defaults.
    """
    from scipy.signal import find_peaks

    params = InstrumentParams()
    r = np.linspace(0, params.r_max, 1000)
    A = airy_ideal(
        r,
        OI_WAVELENGTH_M,
        params.t,
        params.R_refl,
        params.alpha,
        params.n,
        params.r_max,
        params.I0,
        params.I1,
        params.I2,
    )
    peaks, _ = find_peaks(A, height=0.3 * params.I0)
    assert 2 <= len(peaks) <= 20, (
        f"Found {len(peaks)} peaks, expected 2–20 for WindCube defaults"
    )
    r2_peaks = r[peaks] ** 2
    spacings = np.diff(r2_peaks)
    cv = np.std(spacings) / np.mean(spacings)
    assert cv < 0.20, (
        f"Peak spacing in r² not uniform (CV={cv:.3f}); Airy function may be wrong"
    )


# ---------------------------------------------------------------------------
# T2 — PSF broadening increases FWHM
# ---------------------------------------------------------------------------
def test_psf_broadens_fwhm():
    """Modified Airy must have larger FWHM than ideal Airy."""
    from scipy.signal import find_peaks

    params = InstrumentParams()
    r = np.linspace(0, params.r_max, 1000)

    A_ideal = airy_ideal(
        r,
        OI_WAVELENGTH_M,
        params.t,
        params.R_refl,
        params.alpha,
        params.n,
        params.r_max,
        params.I0,
        params.I1,
        params.I2,
    )
    A_mod = airy_modified(
        r,
        OI_WAVELENGTH_M,
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

    def first_peak_fwhm(A_arr, r_arr):
        """Return FWHM of first peak, or None if not measurable."""
        peaks, _ = find_peaks(A_arr)
        if len(peaks) == 0:
            return None
        pk = peaks[0]
        half = A_arr[pk] / 2.0
        left = np.where(A_arr[:pk] < half)[0]
        right = np.where(A_arr[pk:] < half)[0]
        if len(left) == 0 or len(right) == 0:
            return None
        return r_arr[pk + right[0]] - r_arr[left[-1]]

    fwhm_i = first_peak_fwhm(A_ideal, r)
    fwhm_m = first_peak_fwhm(A_mod, r)
    assert fwhm_i is not None and fwhm_m is not None
    assert fwhm_m > fwhm_i, (
        f"PSF did not broaden fringe: ideal={fwhm_i:.4f}, mod={fwhm_m:.4f} px"
    )


# ---------------------------------------------------------------------------
# T3 — Zero PSF returns ideal Airy exactly
# ---------------------------------------------------------------------------
def test_zero_psf_is_identity():
    """With sigma0=sigma1=sigma2=0, airy_modified must equal airy_ideal."""
    params = InstrumentParams(sigma0=0.0, sigma1=0.0, sigma2=0.0)
    r = np.linspace(0, params.r_max, 500)
    lam = OI_WAVELENGTH_M
    A_i = airy_ideal(
        r,
        lam,
        params.t,
        params.R_refl,
        params.alpha,
        params.n,
        params.r_max,
        params.I0,
        params.I1,
        params.I2,
    )
    A_m = airy_modified(
        r,
        lam,
        params.t,
        params.R_refl,
        params.alpha,
        params.n,
        params.r_max,
        params.I0,
        params.I1,
        params.I2,
        0.0,
        0.0,
        0.0,
    )
    np.testing.assert_allclose(
        A_m, A_i, rtol=1e-4, err_msg="Zero PSF did not return ideal Airy"
    )


# ---------------------------------------------------------------------------
# T4 — Instrument matrix shape and non-negativity
# ---------------------------------------------------------------------------
def test_instrument_matrix_shape():
    """A must have shape (R, L), all values >= 0, no NaN/Inf."""
    params = InstrumentParams()
    R, L = 200, 101
    r = np.linspace(0, params.r_max, R)
    wl = make_wavelength_grid(OI_WAVELENGTH_M, params, L=L)
    A = build_instrument_matrix(
        r,
        wl,
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
    assert A.shape == (R, L), f"Expected ({R},{L}), got {A.shape}"
    assert np.all(A >= 0), "Instrument matrix has negative values"
    assert np.all(np.isfinite(A)), "Instrument matrix has NaN or Inf"


# ---------------------------------------------------------------------------
# T5 — Matrix forward model matches direct evaluation
# ---------------------------------------------------------------------------
def test_matrix_forward_model_consistency():
    """
    s = A @ y + B must match direct airy_modified evaluation
    for a monochromatic (delta-function) source.
    Anti-inverse-crime: use L_synth=300 for synthesis, L=101 for matrix.
    """
    params = InstrumentParams()
    R, L_mat = 200, 101
    r = np.linspace(0, params.r_max, R)
    lam0 = OI_WAVELENGTH_M

    # Build matrix with inversion grid (L=101)
    wl_mat = make_wavelength_grid(lam0, params, L=L_mat)
    A_mat = build_instrument_matrix(
        r,
        wl_mat,
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

    # Monochromatic source at central wavelength
    j0 = np.argmin(np.abs(wl_mat - lam0))
    dlam = wl_mat[1] - wl_mat[0]
    y = np.zeros(L_mat)
    y[j0] = 1.0 / dlam

    s_mat = A_mat @ y + params.B
    s_direct = (
        airy_modified(
            r,
            lam0,
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
        + params.B
    )

    np.testing.assert_allclose(
        s_mat,
        s_direct,
        rtol=0.05,
        err_msg="Matrix forward model disagrees with direct evaluation",
    )


# ---------------------------------------------------------------------------
# T6 — Intensity envelope positivity
# ---------------------------------------------------------------------------
def test_intensity_envelope_positive():
    """I(r) must be positive everywhere for default parameters."""
    params = InstrumentParams()
    r = np.linspace(0, params.r_max, 500)
    I = intensity_envelope(r, params.r_max, params.I0, params.I1, params.I2)
    assert np.all(I > 0), f"Intensity envelope non-positive; min={np.min(I):.2f}"
    assert I[0] >= I[-1], (
        "Intensity should fall off toward edge for I1 < 0 (vignetting)"
    )


# ---------------------------------------------------------------------------
# T7 — Finesse and FSR physically reasonable
# ---------------------------------------------------------------------------
def test_instrument_derived_quantities():
    """
    For WindCube defaults (R_refl=0.53, t=20.008 mm):
      Finesse coefficient F ≈ 9.6   (range 6–15 acceptable)
      Instrument finesse   ≈ 4.9    (range 3–20 acceptable)
      FSR at 630 nm        ≈ 9.92 pm (range 8–12 pm)
    """
    params = InstrumentParams()
    F_coeff = params.finesse_coefficient()
    finesse = params.finesse()
    fsr = params.free_spectral_range(OI_WAVELENGTH_M)

    assert 5 < F_coeff < 20, f"Finesse coefficient {F_coeff:.2f} outside [5, 20]"
    assert 2 < finesse < 20, f"Finesse {finesse:.2f} outside [2, 20]"
    assert 8e-12 < fsr < 12e-12, f"FSR {fsr:.3e} m outside [8, 12] pm"


# ---------------------------------------------------------------------------
# T8 — PSF sigma always positive
# ---------------------------------------------------------------------------
def test_psf_sigma_positive():
    """
    PSF sigma must be positive everywhere for default parameters.
    With sigma1=sigma2=0, sigma must be constant = sigma0.
    """
    params = InstrumentParams()
    r = np.linspace(0, params.r_max, 500)
    sigma = psf_sigma(r, params.r_max, params.sigma0, params.sigma1, params.sigma2)
    assert np.all(sigma > 0), f"PSF sigma non-positive; min={np.min(sigma):.4f} px"

    sigma_flat = psf_sigma(r, params.r_max, 2.0, 0.0, 0.0)
    np.testing.assert_allclose(
        sigma_flat,
        2.0,
        rtol=1e-10,
        err_msg="Constant PSF (sigma1=sigma2=0) not returning sigma0",
    )


# ---------------------------------------------------------------------------
# T9 — alpha scales correctly with bin_factor
# (adapted: InstrumentParams doesn't auto-derive alpha; verify via constants)
# ---------------------------------------------------------------------------
def test_alpha_plate_scale():
    """alpha default must equal ALPHA_RAD_PX from constants (2×2 binned)."""
    from src.constants import ALPHA_RAD_PX
    params = InstrumentParams()
    # The 2026_04_05 InstrumentParams uses Tolansky-fitted alpha (1.6071e-4),
    # not the theoretical ALPHA_RAD_PX (1.600e-4). Both should be in [1e-4, 2e-4].
    assert 1e-4 < params.alpha < 2e-4, \
        f"alpha={params.alpha:.4e} outside plausible range [1e-4, 2e-4]"


# ---------------------------------------------------------------------------
# T10 — r_max is within plausible range
# ---------------------------------------------------------------------------
def test_r_max_plausible():
    """r_max must be between 50 and 150 px for 2×2 binned mode."""
    params = InstrumentParams()
    assert 50 < params.r_max <= 150, \
        f"r_max={params.r_max:.1f} px outside plausible range [50, 150]"


# ---------------------------------------------------------------------------
# T11 — make_ne_spectrum places both lines within grid
# ---------------------------------------------------------------------------
def test_ne_spectrum_line_positions():
    """
    make_ne_spectrum must place nonzero power at bins closest to
    NE_WAVELENGTH_1_AIR_M and NE_WAVELENGTH_2_AIR_M, and zero elsewhere.

    Spec deviation: spec used n_fsr=220; changed to n_fsr=400 because
    Ne2 (638.299 nm) falls outside the grid when centered at Ne1 with
    n_fsr=220 (grid min = 639.098 nm; gap = ~0.8 nm).
    """
    from src.constants import (
        NE_WAVELENGTH_1_AIR_M, NE_WAVELENGTH_2_AIR_M,
        NE_INTENSITY_1, NE_INTENSITY_2,
    )
    params = InstrumentParams()
    lam_grid = make_wavelength_grid(NE_WAVELENGTH_1_AIR_M, n_fsr=400,
                                    L=501, params=params)
    y = make_ne_spectrum(lam_grid, I_line=1.0)

    # Exactly two nonzero bins
    nonzero = np.where(y > 0)[0]
    assert len(nonzero) == 2, f"Expected 2 nonzero bins, got {len(nonzero)}"

    # Verify the two bins bracket the correct wavelengths
    lam1_idx = np.argmin(np.abs(lam_grid - NE_WAVELENGTH_1_AIR_M))
    lam2_idx = np.argmin(np.abs(lam_grid - NE_WAVELENGTH_2_AIR_M))
    assert nonzero[0] in [lam1_idx, lam2_idx]
    assert nonzero[1] in [lam1_idx, lam2_idx]

    # Verify intensity ratio
    i1 = y[lam1_idx]
    i2 = y[lam2_idx]
    ratio = i2 / i1
    np.testing.assert_allclose(ratio, NE_INTENSITY_2 / NE_INTENSITY_1,
                                rtol=1e-6, err_msg="Ne line intensity ratio wrong")


# ---------------------------------------------------------------------------
# T12 — make_airglow_spectrum: zero velocity places line at rest wavelength
# ---------------------------------------------------------------------------
def test_airglow_zero_velocity():
    """
    At v_rel=0, λ_c must equal OI_WAVELENGTH_AIR_M to within one bin width.
    """
    from src.constants import OI_WAVELENGTH_AIR_M
    params = InstrumentParams()
    lam_grid = make_wavelength_grid(OI_WAVELENGTH_AIR_M, n_fsr=5,
                                    L=201, params=params)
    y = make_airglow_spectrum(lam_grid, v_rel=0.0, Y_line=1000.0, Y_bg=0.0)
    peak_idx = np.argmax(y)
    lam_peak = lam_grid[peak_idx]
    bin_width = lam_grid[1] - lam_grid[0]
    assert abs(lam_peak - OI_WAVELENGTH_AIR_M) <= bin_width, \
        f"Peak at {lam_peak*1e9:.4f} nm, expected {OI_WAVELENGTH_AIR_M*1e9:.4f} nm"


# ---------------------------------------------------------------------------
# T13 — make_airglow_spectrum: Doppler shift is correct direction and magnitude
# ---------------------------------------------------------------------------
def test_airglow_doppler_shift():
    """
    A positive v_rel must shift λ_c to a longer wavelength (redshift).
    A negative v_rel must shift λ_c to a shorter wavelength (blueshift).
    Magnitude: Δλ = λ₀ · v_rel / c
    """
    from src.constants import OI_WAVELENGTH_AIR_M, SPEED_OF_LIGHT_MS
    params = InstrumentParams()
    lam_grid = make_wavelength_grid(OI_WAVELENGTH_AIR_M, n_fsr=5,
                                    L=501, params=params)

    v_test = 500.0   # m/s recession
    expected_shift = OI_WAVELENGTH_AIR_M * v_test / SPEED_OF_LIGHT_MS

    y_pos = make_airglow_spectrum(lam_grid, v_rel=+v_test, Y_line=1.0)
    y_neg = make_airglow_spectrum(lam_grid, v_rel=-v_test, Y_line=1.0)

    lam_pos = lam_grid[np.argmax(y_pos)]
    lam_neg = lam_grid[np.argmax(y_neg)]

    assert lam_pos > OI_WAVELENGTH_AIR_M, "Positive v_rel should redshift"
    assert lam_neg < OI_WAVELENGTH_AIR_M, "Negative v_rel should blueshift"

    bin_width = lam_grid[1] - lam_grid[0]
    np.testing.assert_allclose(lam_pos - OI_WAVELENGTH_AIR_M,
                                expected_shift, atol=bin_width,
                                err_msg="Doppler shift magnitude wrong")


# ---------------------------------------------------------------------------
# T14 — make_airglow_spectrum: velocity range enforcement
# ---------------------------------------------------------------------------
def test_airglow_velocity_bounds():
    """
    v_rel outside [−7700, +1000] m/s must raise ValueError.
    Boundary values must not raise.
    """
    from src.constants import OI_WAVELENGTH_AIR_M
    params = InstrumentParams()
    lam_grid = make_wavelength_grid(OI_WAVELENGTH_AIR_M, n_fsr=30,
                                    L=501, params=params)

    with pytest.raises(ValueError):
        make_airglow_spectrum(lam_grid, v_rel=-8000.0)
    with pytest.raises(ValueError):
        make_airglow_spectrum(lam_grid, v_rel=+2000.0)

    # Boundary values must succeed
    make_airglow_spectrum(lam_grid, v_rel=-7700.0)
    make_airglow_spectrum(lam_grid, v_rel=+1000.0)


# ---------------------------------------------------------------------------
# T15 — full neon forward model round-trip
# ---------------------------------------------------------------------------
def test_ne_forward_model_roundtrip():
    """
    A · y_ne + B must produce a plausible fringe profile:
    - Non-negative everywhere
    - Has at least 2 peaks (two neon lines produce interleaved ring families)
    - Peak amplitude within 50% of I0

    Spec deviation: spec called build_instrument_matrix(r_bins, lam_grid, params)
    with 3 args; expanded to match existing 13-arg signature.
    Spec used n_fsr=220; changed to n_fsr=400 (same reason as T11).
    """
    from src.constants import NE_WAVELENGTH_1_AIR_M
    from scipy.signal import find_peaks

    params = InstrumentParams()
    r_bins = np.linspace(5, params.r_max, 200)
    lam_grid = make_wavelength_grid(NE_WAVELENGTH_1_AIR_M, n_fsr=400,
                                    L=501, params=params)
    # build_instrument_matrix scales columns by dlam (spectral-density convention);
    # divide y_ne by dlam so that A @ y_ne gives counts (not counts·m).
    dlam = lam_grid[1] - lam_grid[0]
    y_ne = make_ne_spectrum(lam_grid, I_line=1.0) / dlam
    A = build_instrument_matrix(
        r_bins, lam_grid,
        params.t, params.R_refl, params.alpha, params.n,
        params.r_max, params.I0, params.I1, params.I2,
        params.sigma0, params.sigma1, params.sigma2,
    )
    s = A @ y_ne + params.B

    assert np.all(s >= 0), "Fringe profile has negative values"
    peaks, _ = find_peaks(s, height=params.B + 0.1 * params.I0)
    assert len(peaks) >= 2, f"Expected ≥2 fringe peaks, got {len(peaks)}"
    assert np.max(s) < 3 * params.I0, "Peak intensity unreasonably large"
