"""
tolansky-1line.py
=================
Single-line Tolansky method for Fabry-Perot Interferometer characterisation.
Intended for airglow science images (one emission line, all peaks from one family).

Loads a _peak_fits.npy file produced by annular_reduction.py — all valid peaks
are treated as belonging to ONE spectral line (no amplitude-based splitting).

INPUT FILE
----------
<stem>_peak_fits.npy  — 2-D float64 array, one row per detected fringe peak.
9 columns:

  col 0 : peak_num
  col 1 : r_raw (px)          — detected bin centre (find_peaks)       — not used
  col 2 : r_fit (px)          — TRF Gaussian centroid μ                ← r input
  col 3 : sigma_r_fit (px)    — 1-sigma uncertainty on μ               ← sigma_r input
  col 4 : r_fit (px²)         — μ², for use in r²-domain calibration   — not used here
  col 5 : sigma_r_fit (px²)   — 2·μ·σ_μ  (propagated uncertainty)     — not used here
  col 6 : amplitude (ADU)     — Gaussian amplitude A above background  — not used here
  col 7 : width_sigma (px)    — Gaussian width σ                       — not used
  col 8 : reduced_chi2        — χ²/(n_points − 4)                      — not used
Cols 2–8 are NaN when the Gaussian fit failed for that peak (those rows are
dropped before analysis).

OUTPUT
------
  - Weighted least-squares fit of r² vs p
  - Slope S and intercept with 1σ uncertainties
  - Fractional order ε at the geometric centre
  - Successive Δ(r²) and σ(Δr²) — linearity / parallelism diagnostic
  - Recovered plate spacing d  (λ, n, f known)
  - One four-panel diagnostic figure saved alongside the input file

PHYSICAL MODEL
--------------
Constructive interference (Haidinger fringes):

    m λ = 2 n d cos θ

At the pattern centre the order is m₀ = 2nd/λ.
Writing m = m₀ - (p - 1 + ε) for fringe index p and fractional order ε,
and using the paraxial substitution cos θ ≈ 1 - r²/(2f²):

    r_p² = (f² λ / (n d)) · (p − 1 + ε)

So r² is LINEAR in fringe index p, with:

    Slope      S = f² λ / (n d)         [unit² / fringe,  unit = unit of r and f]
    Intercept  b = S · (ε − 1)
    ε            = fractional order at centre  (0 ≤ ε < 1)

Recovering the unknown from the measured slope:

    d = f² λ / (n S)          if  λ, n, f  are known
    λ = n d S / f²            if  d, n, f  are known

UNIT CONVENTION
---------------
r, f, and d must all be in the SAME unit.  This script works entirely in
pixels so that r (from the detector), f, and d share one unit:

    f_px  = f_mm  / pixel_size_mm
    d_px  = d_mm  / pixel_size_mm
    λ_px  = λ_nm  × 1e-9 / pixel_size_m     (lam_unit_per_nm = 1e-9 / pixel_m)

WEIGHTED LEAST-SQUARES
-----------------------
σ(r²) = 2r σ_r varies across rings, so a weighted fit is used:

    weights w_p = 1 / σ(r²_p)²

This gives the correct covariance matrix and propagated parameter uncertainties.

WINDCUBE INSTRUMENT CONSTANTS
------------------------------
  pixel pitch   :  32 µm  (CCD97-00, 2×2 binned)
  focal length  : 200 mm  (→ f_px = 200/0.032 = 6250 px)
  n (gap)       :   1.0   (vacuum gap)
  λ (airglow)   : set LAM_NM in the __main__ block to the airglow emission [nm]
  d  (ICOS)     :  20.008 mm  (spacer measurement, build report §7.4)
                   → d_px = 20.008/0.032 = 625.25 px

One analysis is run:
  λ known  →  recover d   (compare to ICOS calibration value)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TolanskyResult:
    """All outputs from a single Tolansky analysis."""

    # ── Derived input columns ───────────────────────────────────────────────
    p:          np.ndarray   # fringe index
    r:          np.ndarray   # radius
    sigma_r:    np.ndarray   # 1σ uncertainty on r
    r_sq:       np.ndarray   # r²
    sigma_r_sq: np.ndarray   # 1σ uncertainty on r²  = 2r·σ_r

    # ── Weighted least-squares fit ──────────────────────────────────────────
    slope:        float      # S = f² λ / (n d)
    sigma_slope:  float      # 1σ uncertainty on S
    intercept:    float      # b = S(ε − 1)
    sigma_int:    float      # 1σ uncertainty on b
    r2_fit:       float      # coefficient of determination R²

    # ── Derived diagnostics ─────────────────────────────────────────────────
    epsilon:       float       # fractional order at centre  (0 ≤ ε < 1)
    sigma_epsilon: float       # 1σ uncertainty on ε
    delta_r_sq:    np.ndarray  # successive Δ(r²)
    sigma_delta:   np.ndarray  # 1σ uncertainty on each Δ(r²)

    # ── Recovered physical parameter ────────────────────────────────────────
    recovered_d:    Optional[float] = None   # plate spacing  [r_unit]
    sigma_d:        Optional[float] = None
    recovered_lam_nm: Optional[float] = None # wavelength     [nm]
    sigma_lam_nm:   Optional[float] = None


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class TolanskyAnalyser:
    """
    Tolansky r² analysis on measured FPI fringe ring radii.

    Parameters
    ----------
    p        : array_like    fringe indices  (1 = innermost)
    r        : array_like    ring radii
    sigma_r  : array_like    1σ uncertainty on each radius
    r_unit   : str           display label for the radius unit

    Provide all known physical parameters; set the unknown to None:
    lam_nm   : float | None  wavelength [nm]
    n        : float         refractive index of etalon gap
    f        : float | None  effective focal length in the SAME unit as r
    d        : float | None  plate separation in the SAME unit as r

    lam_unit_per_nm : float
        Conversion factor from nm to whatever unit r and f are in.
        Default 1e-6 (nm → mm).  Override when r is in pixels:
            lam_unit_per_nm = 1e-9 / pixel_size_metres
        Example: 32 µm pixels → lam_unit_per_nm = 1e-9 / 32e-6 = 3.125e-5
    """

    def __init__(
        self,
        p:       "array_like",
        r:       "array_like",
        sigma_r: "array_like",
        r_unit:  str             = "mm",
        lam_nm:  Optional[float] = None,
        n:       float           = 1.0,
        f:       Optional[float] = None,
        d:       Optional[float] = None,
        lam_unit_per_nm: float   = 1e-6,   # default: nm → mm
    ):
        self.p       = np.asarray(p,       dtype=float)
        self.r       = np.asarray(r,       dtype=float)
        self.sigma_r = np.asarray(sigma_r, dtype=float)
        self.r_unit  = r_unit
        self.lam_nm  = lam_nm
        self.n       = float(n)
        self.f       = f
        self.d       = d
        self.lam_unit_per_nm = lam_unit_per_nm
        self.result: Optional[TolanskyResult] = None
        self._validate()

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate(self):
        if not (len(self.p) == len(self.r) == len(self.sigma_r)):
            raise ValueError("p, r, sigma_r must all have the same length.")
        if len(self.p) < 3:
            raise ValueError("Need at least 3 rings for a meaningful fit.")
        if np.any(self.sigma_r <= 0):
            raise ValueError("All sigma_r must be > 0.")
        if np.any(self.r <= 0):
            raise ValueError("All radii must be > 0.")
        if self.lam_nm is not None and self.d is not None:
            raise ValueError(
                "Both lam_nm and d are provided — set exactly one to None."
            )
        if self.lam_nm is None and self.d is None:
            raise ValueError(
                "Both lam_nm and d are None — at least one must be known."
            )

    # ── Step 1: r → r² with uncertainty propagation ──────────────────────────

    def _derive_r_squared(self):
        """
        r_sq       = r²
        sigma_r_sq = 2r · sigma_r

        Derivation:
            y = r²  →  dy/dr = 2r
            σ_y = |dy/dr| · σ_r = 2r · σ_r

        Valid provided σ_r << r (well-resolved rings).
        """
        r_sq       = self.r ** 2
        sigma_r_sq = 2.0 * self.r * self.sigma_r
        return r_sq, sigma_r_sq

    # ── Step 2: weighted least-squares r² = S·p + b ──────────────────────────

    @staticmethod
    def _wls(x, y, w):
        """
        Weighted least-squares  y = S·x + b.

        Normal equations (see e.g. Bevington & Robinson §6.3):

            Δ = Σw · Σwx² − (Σwx)²
            S = (Σw · Σwxy  −  Σwx · Σwy) / Δ
            b = (Σwx² · Σwy  −  Σwx · Σwxy) / Δ

            Var(S) = Σw  / Δ
            Var(b) = Σwx² / Δ

        Returns (slope, sigma_slope, intercept, sigma_intercept, R²)
        """
        sw   = w.sum()
        swx  = (w * x).sum()
        swy  = (w * y).sum()
        swxx = (w * x**2).sum()
        swxy = (w * x * y).sum()

        delta = sw * swxx - swx**2
        if delta == 0:
            raise ValueError("Degenerate fit: check fringe indices are distinct.")

        S = (sw * swxy - swx * swy) / delta
        b = (swxx * swy - swx * swxy) / delta

        sigma_S = np.sqrt(sw   / delta)
        sigma_b = np.sqrt(swxx / delta)

        y_hat  = S * x + b
        ss_res = (w * (y - y_hat)**2).sum()
        ss_tot = (w * (y - swy / sw)**2).sum()
        R2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return S, sigma_S, b, sigma_b, R2

    # ── Step 3: fractional order ε ────────────────────────────────────────────

    @staticmethod
    def _epsilon(S, b, sigma_S, sigma_b):
        """
        From  b = S(ε − 1):

            ε = 1 + b/S       (wrapped to [0, 1))

        Error propagation (S and b are correlated from the same fit, but here
        we use the conservative uncorrelated approximation):

            σ_ε² = (σ_b / S)²  +  (b · σ_S / S²)²
        """
        eps_raw  = 1.0 + b / S
        epsilon  = eps_raw % 1.0
        sigma_eps = np.sqrt((sigma_b / S)**2 + (b * sigma_S / S**2)**2)
        return epsilon, sigma_eps

    # ── Step 4: successive Δ(r²) ─────────────────────────────────────────────

    @staticmethod
    def _successive_diffs(r_sq, sigma_r_sq):
        """
        Δ(r²)_k   = r²_{k+1} − r²_k
        σ(Δr²)_k  = sqrt(σ_{k+1}² + σ_k²)

        In a perfect system all Δ(r²) equal the slope S, and the coefficient
        of variation CV = std / mean × 100 % should be < ~2 %.
        Larger CV indicates non-parallelism or systematic measurement error.
        """
        return np.diff(r_sq), np.sqrt(sigma_r_sq[1:]**2 + sigma_r_sq[:-1]**2)

    # ── Step 5: recover physical parameter ───────────────────────────────────

    def _recover(self, S, sigma_S):
        """
        S = f² λ / (n d)

        Solve for the unknown:
            d = f² λ / (n S)        σ_d = d · σ_S / S
            λ = n d S / f²          σ_λ = λ · σ_S / S   (same relative error)

        λ is stored in nm; converted to r_unit via self.lam_unit_per_nm
        before use, and the result converted back to nm for display.
        """
        if self.f is None:
            return None, None, None, None

        rec_d = rec_sd = rec_lam = rec_slam = None

        if self.lam_nm is not None:
            lam_u = self.lam_nm * self.lam_unit_per_nm   # nm → r_unit
            rec_d  = self.f**2 * lam_u / (self.n * S)
            rec_sd = rec_d * sigma_S / S

        elif self.d is not None:
            lam_u   = self.n * self.d * S / self.f**2    # in r_unit
            rec_lam = lam_u / self.lam_unit_per_nm        # → nm
            rec_slam = rec_lam * sigma_S / S

        return rec_d, rec_sd, rec_lam, rec_slam

    # ── Public: run ───────────────────────────────────────────────────────────

    def run(self) -> TolanskyResult:
        """Execute all five steps and return a TolanskyResult."""
        r_sq, sigma_r_sq = self._derive_r_squared()
        w                = 1.0 / sigma_r_sq**2
        S, sS, b, sb, R2 = self._wls(self.p, r_sq, w)
        eps, seps        = self._epsilon(S, b, sS, sb)
        delta, sdelta    = self._successive_diffs(r_sq, sigma_r_sq)
        rec_d, sd, rec_lam, slam = self._recover(S, sS)

        self.result = TolanskyResult(
            p=self.p, r=self.r, sigma_r=self.sigma_r,
            r_sq=r_sq, sigma_r_sq=sigma_r_sq,
            slope=S, sigma_slope=sS,
            intercept=b, sigma_int=sb,
            r2_fit=R2,
            epsilon=eps, sigma_epsilon=seps,
            delta_r_sq=delta, sigma_delta=sdelta,
            recovered_d=rec_d,  sigma_d=sd,
            recovered_lam_nm=rec_lam, sigma_lam_nm=slam,
        )
        return self.result

    # ── Public: print_table ───────────────────────────────────────────────────

    def print_table(self):
        """Print the full Tolansky data table and fit summary."""
        if self.result is None:
            self.run()
        res = self.result
        u, u2 = self.r_unit, f"{self.r_unit}²"

        hdr = (f"{'p':>4}  {'r':>10}  {'σ_r':>10}  "
               f"{'r²':>13}  {'σ(r²)':>12}  "
               f"{'Δ(r²)':>13}  {'σ(Δr²)':>13}")
        sep = "─" * (len(hdr) + 2)

        print(f"\n{sep}")
        print(f"  Tolansky Data Table   [r in {u},  r² in {u2}]")
        print(sep)
        print(hdr)
        print(sep)
        for i, pi in enumerate(res.p):
            ds  = f"{res.delta_r_sq[i-1]:>13.5f}" if i > 0 else f"{'—':>13}"
            sds = f"{res.sigma_delta[i-1]:>13.5f}" if i > 0 else f"{'—':>13}"
            print(f"  {int(pi):>2}  "
                  f"{res.r[i]:>10.4f}  {res.sigma_r[i]:>10.4f}  "
                  f"{res.r_sq[i]:>13.5f}  {res.sigma_r_sq[i]:>12.5f}  "
                  f"{ds}  {sds}")
        print(sep)
        print(f"\n  Weighted linear fit:   r² = S · p + b")
        print(f"    Slope       S = {res.slope:.6g} ± {res.sigma_slope:.6g}  {u2}/fringe")
        print(f"    Intercept   b = {res.intercept:.6g} ± {res.sigma_int:.6g}  {u2}")
        print(f"    R²            = {res.r2_fit:.7f}")
        print(f"    ε (frac. order at centre) = {res.epsilon:.5f} ± {res.sigma_epsilon:.5f}")
        cv = res.delta_r_sq.std() / abs(res.delta_r_sq.mean()) * 100
        print(f"    Δ(r²) mean  = {res.delta_r_sq.mean():.6g}  "
              f"std = {res.delta_r_sq.std():.6g}  "
              f"CV = {cv:.1f} %  {'✓' if cv < 5 else '⚠'}")
        if res.recovered_d is not None:
            print(f"\n  → Recovered plate spacing:  "
                  f"d = {res.recovered_d:.6g} ± {res.sigma_d:.6g}  {u}")
        if res.recovered_lam_nm is not None:
            print(f"\n  → Recovered wavelength:  "
                  f"λ = {res.recovered_lam_nm:.4f} ± {res.sigma_lam_nm:.4f}  nm")
        print()

    # ── Public: plot ──────────────────────────────────────────────────────────

    def plot(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Four-panel diagnostic figure:
          A) Tolansky plot: r² vs p with weighted fit and ±1σ error bars
          B) Fit residuals
          C) Successive Δ(r²) — linearity / parallelism diagnostic
          D) Summary text
        """
        if self.result is None:
            self.run()
        res = self.result
        u, u2 = self.r_unit, f"{self.r_unit}²"

        DARK   = '#0d1117'
        PANEL  = '#161b22'
        BORDER = '#30363d'
        ACCENT = '#58a6ff'
        GREEN  = '#3fb950'
        RED    = '#f85149'
        YELLOW = '#d29922'
        GRAY   = '#8b949e'
        WHITE  = '#e6edf3'

        fig = plt.figure(figsize=(14, 10), facecolor=DARK)
        fig.patch.set_facecolor(DARK)
        gs  = gridspec.GridSpec(2, 2, figure=fig,
                                hspace=0.44, wspace=0.37,
                                left=0.09, right=0.97,
                                top=0.91, bottom=0.08)
        ax_tol = fig.add_subplot(gs[0, 0])
        ax_res = fig.add_subplot(gs[1, 0])
        ax_dr2 = fig.add_subplot(gs[0, 1])
        ax_txt = fig.add_subplot(gs[1, 1])

        for ax in [ax_tol, ax_res, ax_dr2, ax_txt]:
            ax.set_facecolor(PANEL)
            ax.tick_params(colors=WHITE, which='both', direction='in')
            for sp in ax.spines.values():
                sp.set_edgecolor(BORDER)
            ax.xaxis.label.set_color(WHITE)
            ax.yaxis.label.set_color(WHITE)
            ax.title.set_color(WHITE)

        p_fine    = np.linspace(res.p[0] - 0.3, res.p[-1] + 0.3, 300)
        fit_line  = res.slope * p_fine + res.intercept
        residuals = res.r_sq - (res.slope * res.p + res.intercept)

        # ── A: Tolansky plot ──────────────────────────────────────────────────
        ax_tol.errorbar(res.p, res.r_sq, yerr=res.sigma_r_sq,
                        fmt='o', color=ACCENT, ecolor=GRAY,
                        capsize=4, ms=6, lw=1.4, zorder=3,
                        label="Measured  $r^2$")
        ax_tol.plot(p_fine, fit_line, color=GREEN, lw=1.8, zorder=2,
                    label=(f"Fit: $r^2 = {res.slope:.4g}\\,p "
                           f"{res.intercept:+.4g}$"))
        ax_tol.set_xlabel("Fringe index  $p$", fontsize=11)
        ax_tol.set_ylabel(f"$r^2$  [{u2}]", fontsize=11)
        ax_tol.set_title("A — Tolansky Plot", fontsize=11, fontweight='bold', pad=7)
        ax_tol.legend(fontsize=8.5, facecolor=PANEL, labelcolor=WHITE,
                      edgecolor=BORDER, framealpha=0.9)
        ax_tol.text(0.97, 0.05, f"$R^2 = {res.r2_fit:.6f}$",
                    transform=ax_tol.transAxes,
                    ha='right', va='bottom', fontsize=9, color=GREEN)

        # ── B: Residuals ──────────────────────────────────────────────────────
        ax_res.axhline(0, color=GRAY, lw=1.0, ls='--', zorder=1)
        ax_res.errorbar(res.p, residuals, yerr=res.sigma_r_sq,
                        fmt='s', color=YELLOW, ecolor=GRAY,
                        capsize=4, ms=6, lw=1.4, zorder=3)
        ax_res.set_xlabel("Fringe index  $p$", fontsize=11)
        ax_res.set_ylabel(f"Residual  [{u2}]", fontsize=11)
        ax_res.set_title("B — Fit Residuals", fontsize=11, fontweight='bold', pad=7)

        # ── C: Successive Δ(r²) ──────────────────────────────────────────────
        p_mid = 0.5 * (res.p[:-1] + res.p[1:])
        dmean = res.delta_r_sq.mean()
        cv    = res.delta_r_sq.std() / abs(dmean) * 100 if dmean != 0 else np.nan

        ax_dr2.axhline(dmean, color=GREEN, lw=1.3, ls='--',
                       label=f"Mean = {dmean:.4g}")
        ax_dr2.axhline(res.slope, color=ACCENT, lw=1.0, ls=':',
                       label=f"Slope S = {res.slope:.4g}")
        ax_dr2.errorbar(p_mid, res.delta_r_sq, yerr=res.sigma_delta,
                        fmt='^', color=RED, ecolor=GRAY,
                        capsize=4, ms=7, lw=1.4, zorder=3)
        ax_dr2.set_xlabel("Fringe index  $p$  (midpoint)", fontsize=11)
        ax_dr2.set_ylabel(f"$\\Delta(r^2)$  [{u2}]", fontsize=11)
        ax_dr2.set_title("C — Successive  $\\Delta(r^2)$",
                          fontsize=11, fontweight='bold', pad=7)
        ax_dr2.legend(fontsize=8.5, facecolor=PANEL, labelcolor=WHITE,
                      edgecolor=BORDER, framealpha=0.9)
        cv_col   = GREEN if cv < 2 else (YELLOW if cv < 5 else RED)
        cv_label = "✓ parallel" if cv < 5 else "⚠ check alignment"
        ax_dr2.text(0.97, 0.07, f"CV = {cv:.1f}%\n({cv_label})",
                    transform=ax_dr2.transAxes,
                    ha='right', va='bottom', fontsize=9, color=cv_col,
                    multialignment='right')

        # ── D: Summary ────────────────────────────────────────────────────────
        ax_txt.axis('off')
        known_str = (f"λ = {self.lam_nm:.2f} nm  (known)"
                     if self.lam_nm is not None
                     else f"d = {self.d:.6g} {u}  (known)")
        f_str = (f"f = {self.f:.6g} {u}"
                 if self.f is not None else "f = not provided")
        if res.recovered_d is not None:
            rec_line = (f"d  =  {res.recovered_d:.6g} "
                        f"± {res.sigma_d:.4g}  {u}")
        elif res.recovered_lam_nm is not None:
            rec_line = (f"λ  =  {res.recovered_lam_nm:.4f} "
                        f"± {res.sigma_lam_nm:.4f}  nm")
        else:
            rec_line = "(provide f to recover physical param)"

        lines = [
            ("TOLANSKY SUMMARY",          WHITE,  11,   'bold'),
            ("",                          WHITE,   3,   'normal'),
            (f"N rings : {len(res.p)}",   GRAY,   9.5, 'normal'),
            (f"n (gap) : {self.n:.3f}",   GRAY,   9.5, 'normal'),
            (f"{f_str}",                  GRAY,   9.5, 'normal'),
            (f"{known_str}",              GRAY,   9.5, 'normal'),
            ("",                          WHITE,   3,   'normal'),
            ("── Fit ──────────────────", BORDER, 8.5, 'normal'),
            (f"S  = {res.slope:.5g} ± {res.sigma_slope:.3g}  {u2}/fr",
             ACCENT, 9.5, 'normal'),
            (f"b  = {res.intercept:.5g} ± {res.sigma_int:.3g}  {u2}",
             ACCENT, 9.5, 'normal'),
            (f"R² = {res.r2_fit:.7f}",
             GREEN if res.r2_fit > 0.9999 else YELLOW, 9.5, 'normal'),
            (f"ε  = {res.epsilon:.5f} ± {res.sigma_epsilon:.5f}",
             ACCENT, 9.5, 'normal'),
            (f"CV(Δr²) = {cv:.2f} %  "
             f"{'✓' if cv < 5 else '⚠'}",
             cv_col, 9.5, 'normal'),
            ("",                          WHITE,   3,   'normal'),
            ("── Recovered ───────────", BORDER,  8.5, 'normal'),
            (rec_line,                    GREEN,  10,   'bold'),
        ]
        y = 0.97
        for text, color, size, weight in lines:
            ax_txt.text(0.04, y, text, transform=ax_txt.transAxes,
                        ha='left', va='top', fontsize=size,
                        color=color, fontweight=weight,
                        fontfamily='monospace')
            y -= size * 0.013 + 0.010

        fig.suptitle("Tolansky Method  —  FPI Fringe Ring Analysis",
                     color=WHITE, fontsize=13, fontweight='bold', y=0.97)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK)
            print(f"  Figure saved → {save_path}")
        return fig


# ─────────────────────────────────────────────────────────────────────────────
# Entry point  (python tolansky-1line.py  [path/to/<stem>_peak_fits.npy])
# ─────────────────────────────────────────────────────────────────────────────
#
# Ingests the _peak_fits.npy file produced by annular_reduction.py.
# All valid (non-NaN) peaks are treated as a single spectral line.
# 2-D float64 array, one row per detected fringe peak.  9 columns:
#
#   col 0 : peak_num
#   col 1 : r_raw (px)          — detected bin centre (find_peaks)       — not used
#   col 2 : r_fit (px)          — TRF Gaussian centroid μ                ← r input
#   col 3 : sigma_r_fit (px)    — 1-sigma uncertainty on μ               ← sigma_r input
#   col 4 : r_fit (px²)         — μ², for use in r²-domain calibration   — not used here
#   col 5 : sigma_r_fit (px²)   — 2·μ·σ_μ  (propagated uncertainty)     — not used here
#   col 6 : amplitude (ADU)     — Gaussian amplitude A above background  — not used here
#   col 7 : width_sigma (px)    — Gaussian width σ                       — not used
#   col 8 : reduced_chi2        — χ²/(n_points − 4)                      — not used
# Cols 2–8 are NaN when the Gaussian fit failed for that peak.
#
# ── WindCube instrument constants ────────────────────────────────────────────
#
#   pixel pitch   :  32 µm  (CCD97-00, 2×2 binned)
#   focal length  : 200 mm  (→ f_px = 200/0.032 = 6250 px)
#   n (gap)       :   1.0   (vacuum gap)
#   λ (airglow)   : set LAM_NM below to the airglow emission wavelength [nm]
#   d  (ICOS)     :  20.008 mm  → d_px = 625.25 px
#
# One analysis is run:
#   λ known  →  recover d   (compare to ICOS calibration value)

if __name__ == "__main__":
    import sys
    import pathlib
    import tkinter as tk
    from tkinter import filedialog

    # ── Locate the peaks file ─────────────────────────────────────────────────
    if len(sys.argv) > 1:
        peaks_path = pathlib.Path(sys.argv[1])
    else:
        root = tk.Tk()
        root.withdraw()
        path_str = filedialog.askopenfilename(
            title="Select *_peak_fits.npy",
            filetypes=[("NumPy array", "*.npy"), ("All files", "*.*")],
        )
        root.destroy()
        if not path_str:
            print("No file selected — exiting.")
            sys.exit(0)
        peaks_path = pathlib.Path(path_str)

    peaks = np.load(peaks_path)
    if peaks.ndim != 2 or peaks.shape[1] != 9:
        raise ValueError(
            f"Expected shape (N, 9) from _peak_fits.npy, "
            f"got {peaks.shape}.  "
            f"Columns: peak_num | r_raw_px | r_fit_px | sigma_r_fit_px | "
            f"r_fit_sq | sigma_r_fit_sq | amplitude_adu | width_px | reduced_chi2"
        )

    print(f"\n  Loaded  {peaks_path}")
    print(f"  Total peaks : {len(peaks)}")

    # ── Drop peaks where the Gaussian fit failed (NaN in cols 2 or 3) ────────
    valid = ~(np.isnan(peaks[:, 2]) | np.isnan(peaks[:, 3]) | (peaks[:, 3] <= 0))
    n_dropped = int((~valid).sum())
    if n_dropped:
        print(f"  Dropped {n_dropped} peak(s) with failed Gaussian fits (NaN / σ≤0)")
    peaks = peaks[valid]

    if len(peaks) < 3:
        print(f"  ERROR: only {len(peaks)} valid peak(s) after filtering — "
              f"need at least 3 for a fit.")
        sys.exit(1)

    # ── Extract inputs; re-index p from 1 ────────────────────────────────────
    p       = np.arange(1, len(peaks) + 1, dtype=float)
    r       = peaks[:, 2]   # r_fit (px)         — TRF Gaussian centroid μ
    sigma_r = peaks[:, 3]   # sigma_r_fit (px)   — 1σ uncertainty on μ

    print(f"  Valid peaks : {len(peaks)}")
    print(f"  r range     : {r[0]:.2f} – {r[-1]:.2f} px")

    # ── Instrument constants (pixels) ─────────────────────────────────────────
    PIXEL_M         = 32e-6             # pixel pitch  [m]
    F_PX            = 200e-3 / PIXEL_M  # focal length  [px] = 6250.00
    N_GAP           = 1.0               # refractive index (vacuum gap)
    LAM_NM          = 630.0             # airglow emission wavelength [nm] — set as needed
    D_ICOS_MM       = 20.008            # ICOS mechanical measurement  [mm]
    D_PX            = D_ICOS_MM * 1e-3 / PIXEL_M   # [px] = 625.25
    LAM_UNIT_PER_NM = 1e-9 / PIXEL_M   # nm → px  = 3.125e-5

    print(f"\n  Instrument constants:")
    print(f"    pixel pitch  = {PIXEL_M*1e6:.0f} µm")
    print(f"    f            = {F_PX:.2f} px  =  {F_PX*PIXEL_M*1e3:.1f} mm")
    print(f"    d (ICOS)     = {D_PX:.4f} px  =  {D_ICOS_MM:.4f} mm")
    print(f"    λ (airglow)  = {LAM_NM:.4f} nm")
    print(f"    n            = {N_GAP:.1f}")

    sep = "═" * 65

    # ── Run: known λ → recover d ──────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"  known λ = {LAM_NM:.4f} nm  →  recover d")
    print(sep)
    ana = TolanskyAnalyser(
        p=p, r=r, sigma_r=sigma_r,
        r_unit="px",
        lam_nm=LAM_NM, n=N_GAP, f=F_PX, d=None,
        lam_unit_per_nm=LAM_UNIT_PER_NM,
    )
    res = ana.run()
    ana.print_table()
    d_mm     = res.recovered_d * PIXEL_M * 1e3
    sig_d_mm = res.sigma_d     * PIXEL_M * 1e3
    pull_d   = abs(d_mm - D_ICOS_MM) / sig_d_mm
    print(f"  → d         = {d_mm:.6f} ± {sig_d_mm:.6f} mm")
    print(f"  ICOS  d     = {D_ICOS_MM:.6f} mm")
    print(f"  Δ           = {d_mm - D_ICOS_MM:+.6f} mm   (|Δ|/σ = {pull_d:.1f})")

    # ── Save figure ───────────────────────────────────────────────────────────
    out_dir  = peaks_path.parent
    stem     = peaks_path.stem.replace("_peak_fits", "")
    fig_path = str(out_dir / f"{stem}_tolansky_1line.png")
    ana.plot(save_path=fig_path)

    plt.show()
