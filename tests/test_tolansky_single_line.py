"""
Tests for SingleLineTolansky and SingleLineResult.

Note: PeakFitStub uses r_fit_px / sigma_r_fit_px to match the real M03
PeakFit dataclass attributes (spec draft used 'centre'/'sigma_centre').
"""
import dataclasses
import math

import numpy as np
import pytest

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.fpi.tolansky_2026_04_05 import SingleLineTolansky, SingleLineResult


# ---------------------------------------------------------------------------
# Minimal FringeProfile stub — mirrors real M03 PeakFit attribute names
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class PeakFitStub:
    r_fit_px:       float
    sigma_r_fit_px: float
    amplitude_adu:  float
    fit_ok:         bool


@dataclasses.dataclass
class FringeProfileStub:
    peak_fits: list
    cx:       float = 128.0
    cy:       float = 128.0
    sigma_cx: float = 0.1
    sigma_cy: float = 0.1
    r_max_px: float = 110.0


def make_synthetic_profile(
    n_peaks:  int   = 7,
    epsilon:  float = 0.45,
    d_m:      float = 20.106e-3,
    f_m:      float = 199.12e-3,
    pitch_m:  float = 32e-6,
    lam_nm:   float = 630.0,
    sigma_r:  float = 0.05,   # sigma on radius [px]
    seed:     int   = 42,
) -> FringeProfileStub:
    """
    Generate a synthetic FringeProfile whose peak radii follow the
    Tolansky relation exactly (plus small Gaussian noise on radius).
    """
    rng    = np.random.default_rng(seed)
    lam_m  = lam_nm * 1e-9
    f_px   = f_m / pitch_m
    S      = f_px ** 2 * lam_m / (2.0 * d_m)
    p      = np.arange(n_peaks, dtype=float)
    r2_true = S * (p + epsilon)
    r_true  = np.sqrt(r2_true)
    noise   = rng.normal(0.0, sigma_r, n_peaks)
    r_obs   = r_true + noise

    peaks = [
        PeakFitStub(
            r_fit_px       = float(r_obs[i]),
            sigma_r_fit_px = sigma_r,
            amplitude_adu  = 1000.0,
            fit_ok         = True,
        )
        for i in range(n_peaks)
    ]
    return FringeProfileStub(peak_fits=peaks)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSingleLineResult:
    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(SingleLineResult)

    def test_required_fields(self):
        fields = {f.name for f in dataclasses.fields(SingleLineResult)}
        required = {
            "epsilon", "sigma_eps", "two_sigma_eps",
            "S", "sigma_S",
            "lam_c_nm", "sigma_lam_c_nm", "two_sigma_lam_c_nm",
            "v_rel_ms", "sigma_v_ms", "two_sigma_v_ms",
            "N_int", "d_prior_mm", "f_prior_mm", "chi2_dof",
        }
        assert required.issubset(fields), f"Missing fields: {required - fields}"

    def test_two_sigma_consistency(self):
        """two_sigma fields must equal exactly 2 × sigma fields."""
        fp = make_synthetic_profile()
        analyser = SingleLineTolansky(
            fringe_profile=fp, lam_rest_nm=630.0,
            d_prior_m=20.106e-3, f_prior_m=199.12e-3,
            pixel_pitch_m=32e-6, d_icos_m=20.008e-3,
        )
        r = analyser.run()
        assert r.two_sigma_eps      == pytest.approx(2 * r.sigma_eps,      rel=1e-10)
        assert r.two_sigma_lam_c_nm == pytest.approx(2 * r.sigma_lam_c_nm, rel=1e-10)
        assert r.two_sigma_v_ms     == pytest.approx(2 * r.sigma_v_ms,     rel=1e-10)

    def test_sigma_S_is_nan(self):
        """sigma_S must be nan — S is fixed, not fitted."""
        fp = make_synthetic_profile()
        analyser = SingleLineTolansky(
            fringe_profile=fp, lam_rest_nm=630.0,
            d_prior_m=20.106e-3, f_prior_m=199.12e-3,
            pixel_pitch_m=32e-6, d_icos_m=20.008e-3,
        )
        r = analyser.run()
        assert math.isnan(r.sigma_S), "sigma_S should be nan for a fixed slope"


class TestSingleLineTolansky:
    def test_recovers_epsilon_noiseless(self):
        """With near-zero noise, recovered epsilon should match injected epsilon."""
        eps_true = 0.45
        fp = make_synthetic_profile(epsilon=eps_true, sigma_r=1e-6)
        analyser = SingleLineTolansky(
            fringe_profile=fp, lam_rest_nm=630.0,
            d_prior_m=20.106e-3, f_prior_m=199.12e-3,
            pixel_pitch_m=32e-6, d_icos_m=20.008e-3,
        )
        result = analyser.run()
        assert result.epsilon == pytest.approx(eps_true, abs=1e-4)

    def test_zero_velocity_for_rest_source(self):
        """
        When the profile is generated at lam_rest_nm exactly, v_rel should
        be near 0 m/s (within a few tens of m/s for noisy data).
        """
        fp = make_synthetic_profile(lam_nm=630.0, sigma_r=0.05, seed=0)
        analyser = SingleLineTolansky(
            fringe_profile=fp, lam_rest_nm=630.0,
            d_prior_m=20.106e-3, f_prior_m=199.12e-3,
            pixel_pitch_m=32e-6, d_icos_m=20.008e-3,
        )
        result = analyser.run()
        assert abs(result.v_rel_ms) < 50.0, (
            f"v_rel = {result.v_rel_ms:.1f} m/s — expected ~0 for rest source"
        )

    def test_velocity_sign_redshift(self):
        """
        Analysing a 630.0 nm profile with a shorter rest wavelength (629.99 nm)
        makes lambda_c appear longer than lambda_rest => positive v (redshift).
        """
        fp = make_synthetic_profile(lam_nm=630.0, sigma_r=1e-6)
        analyser = SingleLineTolansky(
            fringe_profile=fp, lam_rest_nm=629.99,
            d_prior_m=20.106e-3, f_prior_m=199.12e-3,
            pixel_pitch_m=32e-6, d_icos_m=20.008e-3,
        )
        result = analyser.run()
        assert result.v_rel_ms > 0, "Expected positive v_rel (redshift)"

    def test_raises_on_too_few_peaks(self):
        """Fewer than 3 good peaks must raise RuntimeError."""
        fp = make_synthetic_profile(n_peaks=2)
        analyser = SingleLineTolansky(
            fringe_profile=fp, lam_rest_nm=630.0,
            d_prior_m=20.106e-3, f_prior_m=199.12e-3,
            pixel_pitch_m=32e-6, d_icos_m=20.008e-3,
        )
        with pytest.raises(RuntimeError, match="only 2 good peaks"):
            analyser.run()

    def test_chi2_reasonable(self):
        """chi2_dof should be near 1.0 for correctly-scaled noise."""
        fp = make_synthetic_profile(n_peaks=7, sigma_r=0.05, seed=7)
        analyser = SingleLineTolansky(
            fringe_profile=fp, lam_rest_nm=630.0,
            d_prior_m=20.106e-3, f_prior_m=199.12e-3,
            pixel_pitch_m=32e-6, d_icos_m=20.008e-3,
        )
        result = analyser.run()
        assert 0.1 < result.chi2_dof < 10.0, (
            f"chi2_dof = {result.chi2_dof:.3f} — suspiciously far from 1"
        )

    def test_print_summary_runs(self, capsys):
        fp = make_synthetic_profile()
        analyser = SingleLineTolansky(
            fringe_profile=fp, lam_rest_nm=630.0,
            d_prior_m=20.106e-3, f_prior_m=199.12e-3,
            pixel_pitch_m=32e-6, d_icos_m=20.008e-3,
        )
        result = analyser.run()
        result.print_summary()
        captured = capsys.readouterr()
        assert "\u03b5\u2080" in captured.out or "epsilon" in captured.out.lower()
        assert "v_rel" in captured.out

    def test_d_prior_and_f_prior_stored_in_mm(self):
        """d_prior_mm and f_prior_mm must be in mm (~20 and ~199), not metres."""
        fp = make_synthetic_profile()
        analyser = SingleLineTolansky(
            fringe_profile=fp, lam_rest_nm=630.0,
            d_prior_m=20.106e-3, f_prior_m=199.12e-3,
            pixel_pitch_m=32e-6, d_icos_m=20.008e-3,
        )
        result = analyser.run()
        assert 19.0 < result.d_prior_mm < 22.0, (
            f"d_prior_mm = {result.d_prior_mm} — looks like metres, not mm"
        )
        assert 190.0 < result.f_prior_mm < 210.0, (
            f"f_prior_mm = {result.f_prior_mm} — looks like metres, not mm"
        )

    def test_fit_ok_false_peaks_excluded(self):
        """Peaks with fit_ok=False must not influence the fit."""
        fp_all = make_synthetic_profile(n_peaks=7, sigma_r=1e-6)
        # Corrupt one peak's radius but mark it fit_ok=False
        bad_peak = PeakFitStub(
            r_fit_px=999.0, sigma_r_fit_px=0.1,
            amplitude_adu=500.0, fit_ok=False,
        )
        fp_with_bad = FringeProfileStub(
            peak_fits=fp_all.peak_fits + [bad_peak]
        )
        analyser_all = SingleLineTolansky(
            fringe_profile=fp_all, lam_rest_nm=630.0,
            d_prior_m=20.106e-3, f_prior_m=199.12e-3,
            pixel_pitch_m=32e-6, d_icos_m=20.008e-3,
        )
        analyser_bad = SingleLineTolansky(
            fringe_profile=fp_with_bad, lam_rest_nm=630.0,
            d_prior_m=20.106e-3, f_prior_m=199.12e-3,
            pixel_pitch_m=32e-6, d_icos_m=20.008e-3,
        )
        r_all = analyser_all.run()
        r_bad = analyser_bad.run()
        assert r_all.epsilon == pytest.approx(r_bad.epsilon, abs=1e-6), (
            "fit_ok=False peak should be excluded and not change epsilon"
        )
