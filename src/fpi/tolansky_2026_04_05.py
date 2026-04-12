"""
Module:      tolansky_2026_04_05.py
Spec:        docs/specs/S13_tolansky_analysis_2026-04-05.md
Author:      Claude Code
Generated:   2026-04-06
Last tested: 2026-04-06  (8/8 tests pass)
Project:     WindCube FPI Pipeline — NCAR/HAO
Repo:        soc_sewell
"""

import dataclasses
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import least_squares

# Import wavelength constants from M01 (single source of truth)
from src.fpi.m01_airy_forward_model_2026_04_05 import (
    NE_WAVELENGTH_1_M,
    NE_WAVELENGTH_2_M,
)

# Pipeline defaults (S03 / ICOS build report Dec 2023)
D_PRIOR_M      = 20.008e-3   # metres — ICOS mechanical measurement
PIXEL_PITCH_M  = 32e-6       # metres — 2×2 binned CCD (2 × 16 µm)
_LAM1_NM       = NE_WAVELENGTH_1_M * 1e9   # 640.2248 nm
_LAM2_NM       = NE_WAVELENGTH_2_M * 1e9   # 638.2991 nm


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TolanskyResult:
    """Output from a single-line Tolansky r² WLS fit."""
    p:            np.ndarray   # fringe indices
    r:            np.ndarray   # ring radii, pixels
    sigma_r:      np.ndarray   # 1σ uncertainties, pixels
    r_sq:         np.ndarray   # r²  (px²)
    sigma_r_sq:   np.ndarray   # σ(r²) = 2r·σ_r  (px²)

    slope:        float        # S = f²λ/(nd), px²/fringe
    sigma_slope:  float
    intercept:    float        # b = S(ε−1)
    sigma_int:    float
    r2_fit:       float        # coefficient of determination R²

    epsilon:       float       # fractional order at centre (0 ≤ ε < 1)
    sigma_epsilon: float

    delta_r_sq:    np.ndarray  # successive Δ(r²)
    sigma_delta:   np.ndarray

    recovered_d_m:   Optional[float]   # plate spacing, metres (if f known)
    sigma_d_m:       Optional[float]
    recovered_f_px:  Optional[float]   # focal length, pixels  (if d known)
    sigma_f_px:      Optional[float]


@dataclass
class TwoLineResult:
    """Output from the joint two-line Tolansky analysis (S04 convention)."""
    # Joint fit results
    S1:          float   # slope for λ₁, px²/fringe
    sigma_S1:    float
    S2:          float   # = S1 × λ₂/λ₁  (constrained, not free)
    eps1:        float   # ε₁ fractional order (λ₁ family)
    sigma_eps1:  float
    eps2:        float   # ε₂ fractional order (λ₂ family)
    sigma_eps2:  float
    cov_eps:     float   # covariance(ε₁, ε₂) from joint fit
    chi2_dof:    float   # reduced χ²
    lam_ratio:   float   # λ₂/λ₁

    # Excess fractions d recovery
    N_int:            int    # integer order difference
    delta_eps:        float  # ε₁ − ε₂
    sigma_delta_eps:  float
    d_m:              float  # recovered plate spacing, metres  (S04)
    sigma_d_m:        float
    two_sigma_d_m:    float  # exactly 2 × sigma_d_m            (S04)

    # f and α recovery
    f_px:            float   # recovered focal length, pixels
    sigma_f_px:      float
    two_sigma_f_px:  float   # exactly 2 × sigma_f_px           (S04)
    alpha_rad_px:    float   # magnification constant, rad/px
    sigma_alpha:     float
    two_sigma_alpha: float   # exactly 2 × sigma_alpha          (S04)

    # ε for M05 handoff
    epsilon_cal_1: float   # ε₁ — fractional order at λ₁ (640.2 nm)
    epsilon_cal_2: float   # ε₂ — fractional order at λ₂ (638.3 nm)

    # Residuals for plotting
    p1:     np.ndarray
    r1_sq:  np.ndarray
    sr1_sq: np.ndarray
    pred1:  np.ndarray
    p2:     np.ndarray
    r2_sq:  np.ndarray
    sr2_sq: np.ndarray
    pred2:  np.ndarray

    lam1_nm: float
    lam2_nm: float


# ---------------------------------------------------------------------------
# TolanskyAnalyser — single-line WLS fit
# ---------------------------------------------------------------------------

class TolanskyAnalyser:
    """
    Single-line Tolansky r² analysis on measured FPI fringe ring radii.

    Parameters
    ----------
    p             : fringe indices (1 = innermost ring, 2, 3, …)
    r             : ring radii in pixels
    sigma_r       : 1σ uncertainty on each radius, pixels
    lam_nm        : wavelength in nm
    n             : refractive index of etalon gap (1.0 for air)
    f_px          : effective focal length, pixels (None if unknown)
    d_m           : plate separation, metres (None if unknown)
    pixel_pitch_m : CCD pixel pitch in metres (32e-6 for 2×2 binned)

    Exactly one of d_m or f_px should be provided to recover the other.
    """

    def __init__(self, p, r, sigma_r, lam_nm,
                 n=1.0, f_px=None, d_m=None, pixel_pitch_m=32e-6):
        self.p             = np.asarray(p,       dtype=float)
        self.r             = np.asarray(r,       dtype=float)
        self.sigma_r       = np.asarray(sigma_r, dtype=float)
        self.lam_nm        = float(lam_nm)
        self.lam_m         = self.lam_nm * 1e-9
        self.n             = float(n)
        self.f_px          = f_px
        self.d_m           = d_m
        self.pixel_pitch_m = float(pixel_pitch_m)
        self._result: Optional[TolanskyResult] = None

    def _wls(self, r_sq, sigma_r_sq, p):
        """
        Weighted least-squares fit of r² = S·p + b.
        Weights w = 1/σ(r²)².

        Returns (S, b, sigma_S, sigma_b, r2_fit).
        """
        w     = 1.0 / sigma_r_sq ** 2
        sw    = float(np.sum(w))
        swp   = float(np.sum(w * p))
        swp2  = float(np.sum(w * p ** 2))
        swr2  = float(np.sum(w * r_sq))
        swpr2 = float(np.sum(w * p * r_sq))

        delta = sw * swp2 - swp ** 2

        S = (sw * swpr2 - swp * swr2)  / delta
        b = (swp2 * swr2 - swp * swpr2) / delta

        sigma_S = float(np.sqrt(sw   / delta))
        sigma_b = float(np.sqrt(swp2 / delta))

        # Weighted R²
        r2_bar  = swr2 / sw
        ss_tot  = float(np.sum(w * (r_sq - r2_bar) ** 2))
        ss_res  = float(np.sum(w * (r_sq - S * p - b) ** 2))
        r2_fit  = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0

        return float(S), float(b), sigma_S, sigma_b, float(r2_fit)

    def _epsilon(self, S, b, sigma_S, sigma_b):
        """
        ε = (1 + b/S) mod 1
        σ_ε² = (σ_b/S)² + (b·σ_S/S²)²
        """
        eps = (1.0 + b / S) % 1.0
        sigma_eps = float(
            np.sqrt((sigma_b / S) ** 2 + (b * sigma_S / S ** 2) ** 2)
        )
        return float(eps), sigma_eps

    def _successive_diffs(self, r_sq, sigma_r_sq):
        """Δ(r²)[i] = r²[i+1] − r²[i].  σ(Δ) = sqrt(σ_i² + σ_{i+1}²)."""
        delta       = np.diff(r_sq)
        sigma_delta = np.sqrt(sigma_r_sq[:-1] ** 2 + sigma_r_sq[1:] ** 2)
        return delta, sigma_delta

    def _recover(self, S, sigma_S):
        """
        Recover the unknown physical parameter from S.
        S = f_px² × λ_m / (n × d_m)   [px²/fringe]
        """
        rec_d_m  = None
        sig_d_m  = None
        rec_f_px = None
        sig_f_px = None

        if self.d_m is not None:
            # Recover f_px
            f_px = float(np.sqrt(S * self.n * self.d_m / self.lam_m))
            rec_f_px = f_px
            sig_f_px = f_px * sigma_S / (2.0 * S)   # Gaussian propagation

        if self.f_px is not None:
            # Recover d_m
            d_m = float(self.f_px ** 2 * self.lam_m / (self.n * S))
            rec_d_m = d_m
            sig_d_m = d_m * sigma_S / S   # Gaussian propagation

        return rec_d_m, sig_d_m, rec_f_px, sig_f_px

    def run(self) -> TolanskyResult:
        """Fit r² vs p and recover physical parameters."""
        r_sq       = self.r ** 2
        sigma_r_sq = 2.0 * self.r * self.sigma_r

        S, b, sigma_S, sigma_b, r2_fit = self._wls(r_sq, sigma_r_sq, self.p)
        eps, sigma_eps                 = self._epsilon(S, b, sigma_S, sigma_b)
        delta_r_sq, sigma_delta        = self._successive_diffs(r_sq, sigma_r_sq)
        rec_d, sig_d, rec_f, sig_f     = self._recover(S, sigma_S)

        self._result = TolanskyResult(
            p=self.p.copy(),
            r=self.r.copy(),
            sigma_r=self.sigma_r.copy(),
            r_sq=r_sq,
            sigma_r_sq=sigma_r_sq,
            slope=S,
            sigma_slope=sigma_S,
            intercept=b,
            sigma_int=sigma_b,
            r2_fit=r2_fit,
            epsilon=eps,
            sigma_epsilon=sigma_eps,
            delta_r_sq=delta_r_sq,
            sigma_delta=sigma_delta,
            recovered_d_m=rec_d,
            sigma_d_m=sig_d,
            recovered_f_px=rec_f,
            sigma_f_px=sig_f,
        )
        return self._result

    def print_table(self):
        """Print a compact summary of the single-line fit."""
        if self._result is None:
            self.run()
        res = self._result
        print(f"Single-line Tolansky  λ = {self.lam_nm:.4f} nm")
        print(f"  S = {res.slope:.4f} ± {res.sigma_slope:.4f} px²/fringe")
        print(f"  b = {res.intercept:.4f} ± {res.sigma_int:.4f}")
        print(f"  ε = {res.epsilon:.6f} ± {res.sigma_epsilon:.6f}")
        print(f"  R² = {res.r2_fit:.8f}")
        if res.recovered_f_px is not None:
            print(f"  f = {res.recovered_f_px:.2f} ± {res.sigma_f_px:.2f} px")
        if res.recovered_d_m is not None:
            print(f"  d = {res.recovered_d_m*1e3:.6f} ± {res.sigma_d_m*1e6:.3f} µm mm")


# ---------------------------------------------------------------------------
# TwoLineAnalyser — joint fit + excess fractions
# ---------------------------------------------------------------------------

class TwoLineAnalyser:
    """
    Joint two-line Tolansky analysis. Recovers d and f independently.

    Parameters
    ----------
    analyser1     : TolanskyAnalyser for λ₁ (longer wavelength, 640.2 nm)
    analyser2     : TolanskyAnalyser for λ₂ (shorter wavelength, 638.3 nm)
    lam1_nm       : wavelength 1, nm
    lam2_nm       : wavelength 2, nm
    d_prior_m     : rough prior on plate spacing (ICOS value).
                    Used ONLY to identify N_int — does not bias recovered d.
    n             : refractive index of etalon gap
    pixel_pitch_m : CCD pixel pitch, metres
    """

    def __init__(self, analyser1, analyser2, lam1_nm, lam2_nm,
                 d_prior_m=D_PRIOR_M, n=1.0, pixel_pitch_m=PIXEL_PITCH_M):
        self.a1            = analyser1
        self.a2            = analyser2
        self.lam1_nm       = float(lam1_nm)
        self.lam2_nm       = float(lam2_nm)
        self.lam1_m        = self.lam1_nm * 1e-9
        self.lam2_m        = self.lam2_nm * 1e-9
        self.d_prior_m     = float(d_prior_m)
        self.n             = float(n)
        self.pixel_pitch_m = float(pixel_pitch_m)
        self._result: Optional[TwoLineResult] = None

        # Run single-line analysers to get initial estimates
        self._res1 = self.a1.run()
        self._res2 = self.a2.run()

    def _joint_fit(self):
        """
        Joint WLS fit with constraint S₂ = S₁ × λ₂/λ₁.
        Free parameters: (S₁, ε₁, ε₂).
        Method: scipy.optimize.least_squares(method='lm').
        Covariance inflated by reduced χ².
        """
        lam_ratio = self.lam2_m / self.lam1_m

        p1, r1_sq, sr1_sq = self._res1.p, self._res1.r_sq, self._res1.sigma_r_sq
        p2, r2_sq, sr2_sq = self._res2.p, self._res2.r_sq, self._res2.sigma_r_sq

        def residuals(params):
            S1, eps1, eps2 = params
            S2   = S1 * lam_ratio
            res1 = (r1_sq - S1 * (p1 - 1.0 + eps1)) / sr1_sq
            res2 = (r2_sq - S2 * (p2 - 1.0 + eps2)) / sr2_sq
            return np.concatenate([res1, res2])

        x0 = np.array([self._res1.slope, self._res1.epsilon, self._res2.epsilon])

        opt = least_squares(
            residuals, x0,
            method="lm",
            ftol=1e-15, xtol=1e-15, gtol=1e-15,
            max_nfev=100_000,
        )

        S1, eps1, eps2 = float(opt.x[0]), float(opt.x[1]), float(opt.x[2])
        S2 = S1 * lam_ratio

        # Covariance matrix from Jacobian, inflated by reduced χ²
        N     = len(p1) + len(p2)
        n_dof = N - 3
        chi2_red = (2.0 * float(opt.cost) / n_dof) if n_dof > 0 else 1.0

        J = opt.jac   # shape (N, 3)
        try:
            H       = J.T @ J
            cov_raw = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            cov_raw = np.eye(3) * np.inf

        cov = cov_raw * chi2_red

        sigma_S1   = float(np.sqrt(max(cov[0, 0], 0.0)))
        sigma_eps1 = float(np.sqrt(max(cov[1, 1], 0.0)))
        sigma_eps2 = float(np.sqrt(max(cov[2, 2], 0.0)))
        cov_eps    = float(cov[1, 2])

        pred1 = S1 * (p1 - 1.0 + eps1)
        pred2 = S2 * (p2 - 1.0 + eps2)

        return dict(
            S1=S1, eps1=eps1, eps2=eps2, S2=S2,
            sigma_S1=sigma_S1, sigma_eps1=sigma_eps1, sigma_eps2=sigma_eps2,
            cov_eps=cov_eps, chi2_dof=chi2_red, lam_ratio=lam_ratio,
            p1=p1, r1_sq=r1_sq, sr1_sq=sr1_sq, pred1=pred1,
            p2=p2, r2_sq=r2_sq, sr2_sq=sr2_sq, pred2=pred2,
        )

    def _excess_fractions(self, eps1, eps2, sigma_eps1, sigma_eps2, cov_eps):
        """
        Recover d from the excess fractions method (Benoit 1898).

        lever = λ₁·λ₂ / (2n·(λ₂−λ₁))        [negative, since λ₂ < λ₁]
        N_int = round(d_prior / lever)         [negative integer]
        d     = abs(lever · (N_int + ε₁ − ε₂))
        σ(d)  = |lever| · σ(ε₁ − ε₂)
        """
        lever = self.lam1_m * self.lam2_m / (
            2.0 * self.n * (self.lam2_m - self.lam1_m)
        )                                       # negative

        N_int = int(round(self.d_prior_m / lever))

        delta_eps       = eps1 - eps2
        # Uncertainty via full covariance: σ²(ε₁−ε₂) = σ²ε₁ + σ²ε₂ − 2·cov(ε₁,ε₂)
        var_delta = sigma_eps1 ** 2 + sigma_eps2 ** 2 - 2.0 * cov_eps
        sigma_delta_eps = float(np.sqrt(max(var_delta, 0.0)))

        d       = abs(lever * (N_int + delta_eps))
        sigma_d = abs(lever) * sigma_delta_eps

        return int(N_int), float(delta_eps), float(sigma_delta_eps), float(d), float(sigma_d)

    def _recover_f(self, S1, sigma_S1, d_m, sigma_d_m):
        """
        f_px = sqrt(S₁ · n · d / λ₁)           [pixels]
        α    = pixel_pitch_m / f_m = 1 / f_px   [rad/px]
        σ(f), σ(α) via Gaussian propagation.
        """
        f_px = float(np.sqrt(S1 * self.n * d_m / self.lam1_m))

        # Gaussian propagation treating S₁ and d as independent
        df_dS = 0.5 * f_px / S1      if S1  > 0.0 else 0.0
        df_dd = 0.5 * f_px / d_m     if d_m > 0.0 else 0.0
        sigma_f = float(np.sqrt((df_dS * sigma_S1) ** 2 + (df_dd * sigma_d_m) ** 2))

        # α = pixel_pitch / f_m = 1/f_px   [rad/px]
        alpha       = 1.0 / f_px
        sigma_alpha = alpha * sigma_f / f_px   # = σ(f) / f_px²

        return float(f_px), float(sigma_f), float(alpha), float(sigma_alpha)

    def run(self) -> TwoLineResult:
        """Execute the full joint analysis and return TwoLineResult."""
        fit = self._joint_fit()

        N_int, delta_eps, sigma_delta_eps, d_m, sigma_d_m = self._excess_fractions(
            fit["eps1"], fit["eps2"],
            fit["sigma_eps1"], fit["sigma_eps2"],
            fit["cov_eps"],
        )

        f_px, sigma_f_px, alpha_rad_px, sigma_alpha = self._recover_f(
            fit["S1"], fit["sigma_S1"], d_m, sigma_d_m
        )

        self._result = TwoLineResult(
            S1=fit["S1"],
            sigma_S1=fit["sigma_S1"],
            S2=fit["S2"],
            eps1=fit["eps1"],
            sigma_eps1=fit["sigma_eps1"],
            eps2=fit["eps2"],
            sigma_eps2=fit["sigma_eps2"],
            cov_eps=fit["cov_eps"],
            chi2_dof=fit["chi2_dof"],
            lam_ratio=fit["lam_ratio"],
            N_int=N_int,
            delta_eps=delta_eps,
            sigma_delta_eps=sigma_delta_eps,
            d_m=d_m,
            sigma_d_m=sigma_d_m,
            two_sigma_d_m=2.0 * sigma_d_m,
            f_px=f_px,
            sigma_f_px=sigma_f_px,
            two_sigma_f_px=2.0 * sigma_f_px,
            alpha_rad_px=alpha_rad_px,
            sigma_alpha=sigma_alpha,
            two_sigma_alpha=2.0 * sigma_alpha,
            epsilon_cal_1=fit["eps1"],
            epsilon_cal_2=fit["eps2"],
            p1=fit["p1"],
            r1_sq=fit["r1_sq"],
            sr1_sq=fit["sr1_sq"],
            pred1=fit["pred1"],
            p2=fit["p2"],
            r2_sq=fit["r2_sq"],
            sr2_sq=fit["sr2_sq"],
            pred2=fit["pred2"],
            lam1_nm=self.lam1_nm,
            lam2_nm=self.lam2_nm,
        )
        return self._result

    def print_summary(self):
        """Print a compact summary of the two-line joint analysis."""
        if self._result is None:
            self.run()
        r = self._result
        print(f"Two-line Tolansky analysis")
        print(f"  S₁  = {r.S1:.4f} ± {r.sigma_S1:.4f} px²/fringe")
        print(f"  S₂/S₁ = {r.S2/r.S1:.10f}  (λ₂/λ₁ = {r.lam_ratio:.10f})")
        print(f"  ε₁  = {r.eps1:.8f} ± {r.sigma_eps1:.2e}")
        print(f"  ε₂  = {r.eps2:.8f} ± {r.sigma_eps2:.2e}")
        print(f"  N_int = {r.N_int}")
        print(f"  d   = {r.d_m*1e3:.6f} ± {r.sigma_d_m*1e6:.3f} µm  mm")
        print(f"  f   = {r.f_px:.2f} ± {r.sigma_f_px:.2f} px")
        print(f"  α   = {r.alpha_rad_px:.6e} ± {r.sigma_alpha:.2e} rad/px")
        print(f"  χ²/dof = {r.chi2_dof:.4f}")


# ---------------------------------------------------------------------------
# TolanskyPipeline — full pipeline from FringeProfile
# ---------------------------------------------------------------------------

class TolanskyPipeline:
    """
    Full Tolansky pipeline from a FringeProfile to a TwoLineResult.

    Handles peak splitting by amplitude, fringe index assignment,
    and orchestration of TolanskyAnalyser × 2 and TwoLineAnalyser.

    Parameters
    ----------
    profile                  : FringeProfile from M03 (must have peak_fits)
    d_prior_m                : plate spacing prior, metres. Default 20.008e-3
    lam1_nm                  : primary neon wavelength, nm. Default 640.2248
    lam2_nm                  : secondary neon wavelength, nm. Default 638.2991
    amplitude_split_fraction : peaks below this × max_amplitude are assigned
                               to the λ₂ (weaker) family.  Default 0.7.
    n                        : refractive index. Default 1.0 (air gap).
    pixel_pitch_m            : CCD pixel pitch. Default 32e-6 (2×2 binned).
    sigma_r_default          : fallback σ_r when PeakFit.sigma_r_fit_px is nan.
                               Default 0.5 px.
    """

    def __init__(self, profile,
                 d_prior_m=D_PRIOR_M,
                 lam1_nm=_LAM1_NM,
                 lam2_nm=_LAM2_NM,
                 amplitude_split_fraction=0.7,
                 n=1.0,
                 pixel_pitch_m=PIXEL_PITCH_M,
                 sigma_r_default=0.5):
        self.profile                  = profile
        self.d_prior_m                = float(d_prior_m)
        self.lam1_nm                  = float(lam1_nm)
        self.lam2_nm                  = float(lam2_nm)
        self.amplitude_split_fraction = float(amplitude_split_fraction)
        self.n                        = float(n)
        self.pixel_pitch_m            = float(pixel_pitch_m)
        self.sigma_r_default          = float(sigma_r_default)
        self._result: Optional[TwoLineResult] = None

    def run(self) -> TwoLineResult:
        """
        1. Extract peaks with fit_ok=True.
        2. Split into two families by amplitude threshold.
        3. Assign fringe indices p = 1, 2, … sorted by r_fit_px.
        4. Build TolanskyAnalyser for each family (lam_nm known, d=None).
        5. Run TwoLineAnalyser with d_prior_m.
        6. Return TwoLineResult.

        Raises
        ------
        ValueError : if fewer than 3 peaks in either family after split.
        """
        good_peaks = [pk for pk in self.profile.peak_fits if pk.fit_ok]

        if len(good_peaks) == 0:
            raise ValueError("No peaks with fit_ok=True in FringeProfile.peak_fits")

        # Amplitude-based family split
        max_amp   = max(pk.amplitude_adu for pk in good_peaks)
        threshold = self.amplitude_split_fraction * max_amp

        family1 = sorted(
            [pk for pk in good_peaks if pk.amplitude_adu >= threshold],
            key=lambda pk: pk.r_fit_px,
        )
        family2 = sorted(
            [pk for pk in good_peaks if pk.amplitude_adu < threshold],
            key=lambda pk: pk.r_fit_px,
        )

        if len(family1) < 3:
            raise ValueError(
                f"Only {len(family1)} peaks in λ₁ family after amplitude split "
                f"(threshold = {self.amplitude_split_fraction:.2f} × {max_amp:.1f} "
                f"= {threshold:.1f}) — need ≥ 3"
            )
        if len(family2) < 3:
            raise ValueError(
                f"Only {len(family2)} peaks in λ₂ family after amplitude split "
                f"(threshold = {self.amplitude_split_fraction:.2f} × {max_amp:.1f} "
                f"= {threshold:.1f}) — need ≥ 3"
            )

        # Fringe indices: p = 1, 2, … for each family
        p1 = np.arange(1, len(family1) + 1, dtype=float)
        p2 = np.arange(1, len(family2) + 1, dtype=float)

        r1 = np.array([pk.r_fit_px for pk in family1])
        r2 = np.array([pk.r_fit_px for pk in family2])

        def _safe_sigma(pk):
            v = pk.sigma_r_fit_px
            return v if np.isfinite(v) and v > 0.0 else self.sigma_r_default

        sigma_r1 = np.array([_safe_sigma(pk) for pk in family1])
        sigma_r2 = np.array([_safe_sigma(pk) for pk in family2])

        # Build single-line analysers (d unknown — TwoLineAnalyser will recover it)
        a1 = TolanskyAnalyser(
            p1, r1, sigma_r1,
            lam_nm=self.lam1_nm, n=self.n,
            d_m=None, pixel_pitch_m=self.pixel_pitch_m,
        )
        a2 = TolanskyAnalyser(
            p2, r2, sigma_r2,
            lam_nm=self.lam2_nm, n=self.n,
            d_m=None, pixel_pitch_m=self.pixel_pitch_m,
        )

        tla = TwoLineAnalyser(
            a1, a2,
            lam1_nm=self.lam1_nm,
            lam2_nm=self.lam2_nm,
            d_prior_m=self.d_prior_m,
            n=self.n,
            pixel_pitch_m=self.pixel_pitch_m,
        )
        self._result = tla.run()
        return self._result

    def to_m05_priors(self) -> dict:
        """
        Convert TwoLineResult to the prior dict expected by M05 FitConfig.

        Returns
        -------
        dict with keys: t_init_mm, t_bounds_mm, alpha_init, alpha_bounds,
                        epsilon_cal_1, epsilon_cal_2
        """
        if self._result is None:
            self.run()
        r     = self._result
        d_mm  = r.d_m * 1e3
        alpha = r.alpha_rad_px
        return {
            "t_init_mm":     d_mm,
            "t_bounds_mm":   (d_mm - 0.020, d_mm + 0.020),
            "alpha_init":    alpha,
            "alpha_bounds":  (alpha * 0.875, alpha * 1.125),   # ± 12.5%
            "epsilon_cal_1": r.epsilon_cal_1,
            "epsilon_cal_2": r.epsilon_cal_2,
        }


# ---------------------------------------------------------------------------
# SingleLineResult — output dataclass for SingleLineTolansky
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class SingleLineResult:
    """
    Output of SingleLineTolansky.run().

    All uncertainties are 1-sigma unless prefixed 'two_sigma_'.
    v_rel_ms: positive = redshift (source moving away from observer).
    """
    # Fractional interference order at lam_rest_nm
    epsilon:            float
    sigma_eps:          float
    two_sigma_eps:      float

    # Tolansky slope S = f² * lambda / (2d)  [px²/fringe]
    S:                  float
    sigma_S:            float   # nan — S is fixed, not fitted

    # Calibrated line-centre wavelength  [nm]
    lam_c_nm:           float
    sigma_lam_c_nm:     float
    two_sigma_lam_c_nm: float

    # Line-of-sight velocity  [m/s]
    v_rel_ms:           float
    sigma_v_ms:         float
    two_sigma_v_ms:     float

    # Diagnostics
    N_int:       int    # Integer interference order at fringe centre
    d_prior_mm:  float  # d prior used  [mm]
    f_prior_mm:  float  # f prior used  [mm]
    chi2_dof:    float  # Reduced chi-squared of r² fit

    def print_summary(self) -> None:
        """Print a compact human-readable summary."""
        print("  SingleLineTolansky result:")
        print(f"    \u03b5\u2080      = {self.epsilon:.6f} \u00b1 {self.sigma_eps:.6f}  (1\u03c3)")
        print(f"    \u03bb_c     = {self.lam_c_nm:.5f} \u00b1 {self.sigma_lam_c_nm:.5f} nm  (1\u03c3)")
        print(f"    v_rel   = {self.v_rel_ms:.1f} \u00b1 {self.sigma_v_ms:.1f} m/s  (1\u03c3)")
        print(f"    N_int   = {self.N_int}")
        print(f"    d_prior = {self.d_prior_mm:.6f} mm  (fixed)")
        print(f"    f_prior = {self.f_prior_mm:.3f} mm  (fixed)")
        print(f"    \u03c7\u00b2/\u03bd    = {self.chi2_dof:.3f}")


# ---------------------------------------------------------------------------
# SingleLineTolansky — 1-parameter WLS fit for monochromatic fringe source
# ---------------------------------------------------------------------------

class SingleLineTolansky:
    """
    Single-line Tolansky analysis for a monochromatic fringe source at a
    known rest wavelength, using fixed priors for d and f.

    Physics
    -------
    The Tolansky relation for fringe p (p = 0, 1, 2, ... from innermost):

        r²_p = S * (p + epsilon)

    where S = f_px² * lambda_m / (2 * d_m) is fully determined by the priors
    and epsilon is the single free parameter (fractional order at centre).

    A 1-parameter WLS fit recovers epsilon directly.  N_int is resolved using
    d_icos_m (mechanical measurement) to avoid circularity with d_prior_m.

    Parameters
    ----------
    fringe_profile : FringeProfile
        Output of reduce_calibration_frame() from M03.
    lam_rest_nm : float
        Rest wavelength of the emission line [nm].
    d_prior_m : float
        Fixed etalon gap prior [m].  Typically TOLANSKY_D_MM * 1e-3.
    f_prior_m : float
        Fixed focal length prior [m].  Typically TOLANSKY_F_MM * 1e-3.
    pixel_pitch_m : float
        Pixel pitch [m].  For 2x2 binned CCD97: 32e-6.
    d_icos_m : float
        ICOS mechanical gap [m].  Used only to resolve N_int ambiguity.
    sigma_r_default : float
        Fallback sigma_r when PeakFit.sigma_r_fit_px is nan.  Default 0.5 px.
    """

    C_MS: float = 299_792_458.0   # speed of light [m/s]

    def __init__(
        self,
        fringe_profile,
        lam_rest_nm:   float,
        d_prior_m:     float,
        f_prior_m:     float,
        pixel_pitch_m: float,
        d_icos_m:      float,
        sigma_r_default: float = 0.5,
    ) -> None:
        self.fp              = fringe_profile
        self.lam_rest_nm     = float(lam_rest_nm)
        self.lam_rest_m      = float(lam_rest_nm) * 1e-9
        self.d_prior_m       = float(d_prior_m)
        self.f_prior_m       = float(f_prior_m)
        self.pixel_pitch_m   = float(pixel_pitch_m)
        self.d_icos_m        = float(d_icos_m)
        self.sigma_r_default = float(sigma_r_default)

    def run(self) -> SingleLineResult:
        """Execute the single-line Tolansky fit and return a SingleLineResult."""
        fp = self.fp

        # --- gather good peaks sorted by radius (innermost first) -----------
        good = sorted(
            [pf for pf in fp.peak_fits if pf.fit_ok],
            key=lambda pf: pf.r_fit_px,
        )
        if len(good) < 3:
            raise RuntimeError(
                f"SingleLineTolansky: only {len(good)} good peaks — "
                "need at least 3 for a reliable fit."
            )

        # Convert radii to r² and propagate uncertainties
        def _safe_sigma(pf):
            v = pf.sigma_r_fit_px
            return v if np.isfinite(v) and v > 0.0 else self.sigma_r_default

        r_arr     = np.array([pf.r_fit_px for pf in good])
        sigma_r   = np.array([_safe_sigma(pf) for pf in good])
        r2_obs    = r_arr ** 2
        sigma_r2  = 2.0 * r_arr * sigma_r      # propagated: sigma(r²) = 2r * sigma_r

        # --- fringe indices p = 0, 1, 2, ... (innermost = 0) ---------------
        p = np.arange(len(good), dtype=float)

        # --- fixed slope S [px²/fringe] -------------------------------------
        # S = f_px² * lambda_m / (2 * d_m)
        f_px    = self.f_prior_m / self.pixel_pitch_m
        S_fixed = (f_px ** 2 * self.lam_rest_m) / (2.0 * self.d_prior_m)

        # --- 1-parameter WLS for epsilon ------------------------------------
        # r²_p = S * (p + epsilon)  =>  r²_p / S - p = epsilon (constant)
        # weight w_i = 1 / sigma(r²_i)²
        weights  = 1.0 / sigma_r2 ** 2
        y        = r2_obs / S_fixed - p
        W        = float(np.sum(weights))
        epsilon  = float(np.sum(weights * y) / W)
        sigma_eps = float(np.sqrt(1.0 / W))

        # --- goodness of fit ------------------------------------------------
        residuals = r2_obs - S_fixed * (p + epsilon)
        chi2      = float(np.sum((residuals / sigma_r2) ** 2))
        dof       = max(len(good) - 1, 1)
        chi2_dof  = chi2 / dof

        # --- integer order N_int -------------------------------------------
        # Resolve from d_prior_m (self-consistent with lambda_c computation).
        # d_prior and d_icos differ by ~98 µm = ~312 fringe orders at 630 nm,
        # so using d_icos here would give an N_int inconsistent with d_prior.
        # d_icos_m is stored in __init__ for future cross-checking but is not
        # used to resolve N_int.
        N_int = int(round(2.0 * self.d_prior_m / self.lam_rest_m))

        # --- calibrated wavelength ------------------------------------------
        # lambda_c = 2 * d_prior / (N_int + epsilon)   [at normal incidence]
        lam_c_m  = 2.0 * self.d_prior_m / (N_int + epsilon)
        lam_c_nm = lam_c_m * 1e9

        # d(lambda_c)/d(epsilon) = -lambda_c / (N_int + epsilon)
        dlam_c_deps    = -lam_c_m / (N_int + epsilon)
        sigma_lam_c_m  = abs(dlam_c_deps) * sigma_eps
        sigma_lam_c_nm = sigma_lam_c_m * 1e9

        # --- line-of-sight velocity [m/s] -----------------------------------
        # v = c * (lambda_c - lambda_rest) / lambda_rest
        # positive = redshift = source moving away from observer
        v_rel_ms   = self.C_MS * (lam_c_m - self.lam_rest_m) / self.lam_rest_m
        sigma_v_ms = self.C_MS * sigma_lam_c_m / self.lam_rest_m

        return SingleLineResult(
            epsilon            = epsilon,
            sigma_eps          = sigma_eps,
            two_sigma_eps      = 2.0 * sigma_eps,
            S                  = float(S_fixed),
            sigma_S            = float("nan"),   # S is fixed, no fitted uncertainty
            lam_c_nm           = float(lam_c_nm),
            sigma_lam_c_nm     = float(sigma_lam_c_nm),
            two_sigma_lam_c_nm = 2.0 * float(sigma_lam_c_nm),
            v_rel_ms           = float(v_rel_ms),
            sigma_v_ms         = float(sigma_v_ms),
            two_sigma_v_ms     = 2.0 * float(sigma_v_ms),
            N_int              = N_int,
            d_prior_mm         = self.d_prior_m * 1e3,
            f_prior_mm         = self.f_prior_m * 1e3,
            chi2_dof           = float(chi2_dof),
        )
