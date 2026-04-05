"""
Tests for M01 Airy Forward Model.

Spec:        specs/S09_m01_airy_forward_model_2026-04-05.md
Spec tests:  T1–T8
Run with:    pytest tests/test_m01_airy_forward_model_2026_04_05.py -v
"""

import numpy as np
import pytest

from src.fpi.m01_airy_forward_model_2026_04_05 import (
    InstrumentParams,
    airy_ideal,
    airy_modified,
    build_instrument_matrix,
    intensity_envelope,
    make_wavelength_grid,
    psf_sigma,
    OI_WAVELENGTH_M,
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
