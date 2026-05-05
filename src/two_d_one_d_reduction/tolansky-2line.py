"""
tolansky.py
===========
Standalone Tolansky method for Fabry-Perot Interferometer characterisation.

(example calibration data contained in radial_fringe_peaks.npy)

INPUT
-----
A table of fringe measurements :
    p        : fringe index array  (1 = innermost ring, 2, 3, ...)
    r        : ring radius array   [any consistent length unit, e.g. mm or px]
    sigma_r  : 1-sigma uncertainty on each radius measurement (same unit as r)

The code derives:
    r_sq       = r²
    sigma_r_sq = 2 r · sigma_r        (first-order Gaussian propagation)

OUTPUT
------
  - Weighted least-squares fit of r² vs p
  - Slope S and intercept with 1σ uncertainties
  - Fractional order ε at the geometric centre
  - Successive Δ(r²) and σ(Δr²) — linearity / parallelism diagnostic
  - Recovered physical parameter:  d  (if λ,n,f given)  OR  λ  (if d,n,f given)
  - Four-panel diagnostic figure

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
r, f, and d must all be in the SAME unit (e.g. all in mm, or all in pixels).
λ is provided in nanometres and converted internally to the r/f/d unit.
The conversion factor used is:

    λ_unit = λ_nm × 1e-6          (valid when r_unit = mm)

If r and f are in pixels, pre-convert λ yourself:
    λ_px  = λ_nm × 1e-9 / pixel_size_m

WEIGHTED LEAST-SQUARES
-----------------------
σ(r²) = 2r σ_r varies across rings, so a weighted fit is used:

    weights w_p = 1 / σ(r²_p)²

This gives the correct covariance matrix and propagated parameter uncertainties.

USAGE EXAMPLE
-------------
    import numpy as np
    from tolansky import TolanskyAnalyser

    # Measured ring radii on a CCD detector (units: pixels)
    p       = np.arange(1, 10, dtype=float)
    r_px    = np.array([12.8, 24.0, 31.5, 37.5, 42.5, 46.9, 50.9, 54.6, 58.1])
    sig_r   = np.full_like(r_px, 0.3)    # ±0.3 px on every measurement

    # Camera: 24 µm pixels, f_eff = 75 mm → f_px = 75/0.024 = 3125 px
    # d and λ must share the unit of r (pixels here):
    #   d_px = d_mm / pixel_size_mm = 15 / 0.024 = 625 px
    #   λ_px = λ_nm × 1e-9 / pixel_size_m = 630e-9 / 24e-6 = 0.02625 px

    analyser = TolanskyAnalyser(
        p        = p,
        r        = r_px,
        sigma_r  = sig_r,
        r_unit   = "px",
        lam_nm   = None,      # will be recovered
        n        = 1.0,
        f        = 3125.0,    # px
        d        = 625.0,     # px   (known)
        lam_unit_per_nm = 1e-9 / 24e-6,   # nm → px conversion
    )

    result = analyser.run()
    analyser.print_table()
    analyser.plot()

    # ── OR: work entirely in mm ─────────────────────────────────────────────
    analyser2 = TolanskyAnalyser(
        p        = p_mm,
        r        = r_mm,
        sigma_r  = sig_r_mm,
        r_unit   = "mm",
        lam_nm   = 630.0,    # known → recover d
        n        = 1.0,
        f        = 150.0,    # mm
        d        = None,     # unknown
    )
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
        Example: 24 µm pixels → lam_unit_per_nm = 1e-9 / 24e-6 = 0.04167
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
# Two-line joint analyser
# ─────────────────────────────────────────────────────────────────────────────
#
# THEORY — what a second wavelength buys you
# ------------------------------------------
# A single-line Tolansky fit measures the slope  S = f²λ/(nd).
# The two unknowns f and d cannot be separated: knowing S and λ gives only
# the ratio f²/d, not f or d individually.
#
# With two known wavelengths λ₁ and λ₂ observed simultaneously through the
# SAME etalon (same f, same d), three extra constraints appear:
#
#   1. Slope ratio constraint  (S₁/S₂ = λ₁/λ₂)
#      The slopes of the two r²-vs-p lines must be in the same ratio as
#      their wavelengths.  A joint fit enforces this exactly, using all
#      N₁+N₂ rings to determine one shared slope (S₁) and two independent
#      fractional orders (ε₁, ε₂).
#
#   2. Excess-fractions method  (Benoit 1898, Michelson & Benoit 1895)
#      The central interference orders of the two lines are:
#          m₀,₁ = 2nd/λ₁      m₀,₂ = 2nd/λ₂
#      Their difference is:
#          m₀,₁ − m₀,₂  =  2nd(1/λ₁ − 1/λ₂)  =  N_int + (ε₁ − ε₂)
#      where N_int is an integer and (ε₁ − ε₂) is the measurable fractional
#      remainder.  Rearranging for d:
#
#          d = (N_int + ε₁ − ε₂) · λ₁λ₂ / [2n(λ₂ − λ₁)]        ... (EF)
#
#      N_int is identified by rounding  2·d_prior·(1/λ₁−1/λ₂)  using any
#      rough prior for d (here the ICOS mechanical measurement resolves the
#      FSR-period ambiguity — it is the ONLY role of the prior).
#      Once N_int is fixed, d is determined from ε₁, ε₂ ALONE, with no
#      dependence on f whatsoever.
#
#   3. Recover f independently
#      With d now known, the slope S₁ = f²λ₁/(nd) gives:
#          f = sqrt(S₁ · n · d / λ₁)
#      The magnification constant α = pixel_pitch / f  [rad/px] follows
#      immediately and feeds directly into M05 as the initial guess.
#
# Summary of information flow:
#
#   Single-line fit  →  S = f²λ/(nd)       (f and d entangled)
#   Joint fit        →  S₁, S₂, ε₁, ε₂    (same instrument, same d/f)
#   Excess fractions →  d                  (no f required)
#   d + S₁          →  f                  (α for M05)
#
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TwoLineResult:
    """Outputs from the joint two-line Tolansky analysis."""

    # ── Joint fit ───────────────────────────────────────────────────────────
    S1:          float    # slope for line 1  [r_unit²/fringe]
    sigma_S1:    float
    S2:          float    # = S1 × λ₂/λ₁  (not free)
    eps1:        float    # fractional order at centre, line 1
    sigma_eps1:  float
    eps2:        float    # fractional order at centre, line 2
    sigma_eps2:  float
    cov_eps:     float    # covariance(ε₁, ε₂) from joint fit
    chi2_dof:    float    # reduced chi-squared of joint fit
    lam_ratio:   float    # λ₂/λ₁ (enforced in fit)

    # ── Excess-fractions d recovery ─────────────────────────────────────────
    N_int:       int      # integer order difference used in EF formula
    delta_eps:   float    # ε₁ − ε₂
    sigma_delta_eps: float
    d:           float    # recovered plate spacing  [r_unit]
    sigma_d:     float

    # ── f recovery ──────────────────────────────────────────────────────────
    f:           float    # recovered focal length  [r_unit]
    sigma_f:     float

    # ── Residuals for plotting ───────────────────────────────────────────────
    p1:          np.ndarray
    r1_sq:       np.ndarray
    sr1_sq:      np.ndarray
    pred1:       np.ndarray   # joint-fit predicted r² for line 1

    p2:          np.ndarray
    r2_sq:       np.ndarray
    sr2_sq:      np.ndarray
    pred2:       np.ndarray   # joint-fit predicted r² for line 2

    lam1_nm:     float
    lam2_nm:     float
    r_unit:      str


class TwoLineAnalyser:
    """
    Joint two-line Tolansky analysis: recovers d and f independently.

    Parameters
    ----------
    analyser1, analyser2 : TolanskyAnalyser
        Pre-built single-line analysers, each already run().
        analyser1  should correspond to λ₁ (the longer wavelength).
        analyser2  should correspond to λ₂ (the shorter wavelength).
    lam1_nm, lam2_nm : float
        Wavelengths in nm.
    d_prior : float
        Rough prior on plate spacing in the SAME unit as r.
        Used ONLY to identify the integer N_int in the excess-fractions
        formula; it does not bias the recovered d.
    lam_unit_per_nm : float
        Same conversion factor as used in the individual analysers
        (default 1e-6 for nm → mm; override for pixels).
    n : float
        Refractive index of etalon gap.
    """

    def __init__(
        self,
        analyser1:       TolanskyAnalyser,
        analyser2:       TolanskyAnalyser,
        lam1_nm:         float,
        lam2_nm:         float,
        d_prior:         float,
        lam_unit_per_nm: float = 1e-6,
        n:               float = 1.0,
    ):
        # Ensure individual fits have been run
        if analyser1.result is None:
            analyser1.run()
        if analyser2.result is None:
            analyser2.run()

        self.a1  = analyser1
        self.a2  = analyser2
        self.lam1_nm = float(lam1_nm)
        self.lam2_nm = float(lam2_nm)
        self.d_prior = float(d_prior)
        self.conv    = float(lam_unit_per_nm)
        self.n       = float(n)
        self.result: Optional[TwoLineResult] = None

        if analyser1.r_unit != analyser2.r_unit:
            raise ValueError("Both analysers must use the same r_unit.")
        self.r_unit = analyser1.r_unit

    # ── Step 1: joint weighted fit ────────────────────────────────────────────

    def _joint_fit(self):
        """
        Fit both ring families simultaneously with  S₂ = S₁·(λ₂/λ₁)  enforced.

        Free parameters:  (S₁, ε₁, ε₂)   [3 parameters]
        Data:             N₁ + N₂ rings

        The residual vector is:
            [ (r²₁ − S₁(p₁−1+ε₁)) / σ(r²₁) ,
              (r²₂ − S₁·λ_ratio·(p₂−1+ε₂)) / σ(r²₂) ]

        Returns  x = [S1, eps1, eps2],  covariance 3×3,  chi²/dof
        """
        from scipy.optimize import least_squares as _lsq

        res1, res2  = self.a1.result, self.a2.result
        lam_ratio   = (self.lam2_nm * self.conv) / (self.lam1_nm * self.conv)

        p1, r1_sq, sr1_sq = res1.p, res1.r_sq, res1.sigma_r_sq
        p2, r2_sq, sr2_sq = res2.p, res2.r_sq, res2.sigma_r_sq

        def _resid(params):
            S1, e1, e2 = params
            pred1 = S1 * (p1 - 1.0 + e1)
            pred2 = S1 * lam_ratio * (p2 - 1.0 + e2)
            return np.concatenate([(r1_sq - pred1) / sr1_sq,
                                   (r2_sq - pred2) / sr2_sq])

        x0  = [res1.slope, res1.epsilon, res2.epsilon]
        fit = _lsq(_resid, x0, method='lm',
                   ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=100_000)

        S1, eps1, eps2 = fit.x
        eps1 = eps1 % 1.0
        eps2 = eps2 % 1.0
        S2   = S1 * lam_ratio

        ndof     = len(fit.fun) - 3
        chi2_dof = float(np.sum(fit.fun ** 2)) / max(ndof, 1)

        # Covariance from Jacobian, inflated by reduced χ²
        try:
            cov = np.linalg.inv(fit.jac.T @ fit.jac) * chi2_dof
        except np.linalg.LinAlgError:
            cov = np.diag([np.nan, np.nan, np.nan])

        perr = np.sqrt(np.diag(cov))

        pred1 = S1 * (p1 - 1.0 + eps1)
        pred2 = S2 * (p2 - 1.0 + eps2)

        return (S1, perr[0], S2, eps1, perr[1], eps2, perr[2],
                cov[1, 2], chi2_dof, lam_ratio,
                p1, r1_sq, sr1_sq, pred1,
                p2, r2_sq, sr2_sq, pred2)

    # ── Step 2: excess fractions → d ─────────────────────────────────────────

    def _excess_fractions(self, S1, eps1, sigma_eps1, eps2, sigma_eps2, cov_eps):
        """
        d = (N_int + ε₁ − ε₂) · λ₁λ₂ / [2n(λ₂ − λ₁)]

        N_int identified from  d_prior  (resolves FSR-period ambiguity only).
        σ(d) propagated from σ(ε₁), σ(ε₂), and cov(ε₁,ε₂).
        """
        lam1_u = self.lam1_nm * self.conv   # in r_unit
        lam2_u = self.lam2_nm * self.conv

        # Lever arm [r_unit per order]
        lever  = lam1_u * lam2_u / (2.0 * self.n * (lam2_u - lam1_u))
        # Note: λ₂ < λ₁ → (λ₂ − λ₁) < 0 → lever < 0.
        # We want d > 0, so use |lever| and ensure sign is consistent.
        # The full formula handles sign automatically:
        # d = lever × (N_int + ε₁ − ε₂)
        # where N_int + (ε₁ − ε₂) > 0 when λ₁ > λ₂.

        m_diff_prior = self.d_prior / abs(lever)   # ≈ N_int + (ε₁ − ε₂)
        N_int        = int(np.round(m_diff_prior))

        delta_eps       = eps1 - eps2
        sigma_delta_eps = np.sqrt(sigma_eps1**2 + sigma_eps2**2
                                  - 2.0 * cov_eps)

        order_sum = N_int + delta_eps
        d         = lever * order_sum          # d in r_unit  (will be negative
                                               # if lever < 0 & order_sum > 0)
        d         = abs(d)                     # physical d > 0

        sigma_d   = abs(lever) * sigma_delta_eps

        return N_int, delta_eps, sigma_delta_eps, d, sigma_d

    # ── Step 3: recover f ─────────────────────────────────────────────────────

    def _recover_f(self, S1, sigma_S1, d, sigma_d):
        """
        f² = S₁ · n · d / λ₁      →     f = sqrt(S₁ · n · d / λ₁)

        S₁ and d are nearly independent (d comes from ε₁,ε₂ which are
        orthogonal to S₁ in the joint fit covariance), so we treat them
        as uncorrelated in the uncertainty propagation.

        σ(f²) = sqrt[ (n·d/λ₁·σ_S1)² + (S₁·n/λ₁·σ_d)² ]
        σ(f)  = σ(f²) / (2f)
        """
        lam1_u = self.lam1_nm * self.conv
        f_sq   = S1 * self.n * d / lam1_u
        f      = np.sqrt(abs(f_sq))

        dfsq_dS1 = self.n * d  / lam1_u
        dfsq_dd  = S1 * self.n / lam1_u
        sigma_fsq = np.sqrt((dfsq_dS1 * sigma_S1)**2 + (dfsq_dd * sigma_d)**2)
        sigma_f   = sigma_fsq / (2.0 * f) if f > 0 else np.nan

        return f, sigma_f

    # ── Public: run ───────────────────────────────────────────────────────────

    def run(self) -> TwoLineResult:
        """Execute all three steps and return a TwoLineResult."""
        (S1, sS1, S2, eps1, seps1, eps2, seps2,
         cov_eps, chi2_dof, lam_ratio,
         p1, r1_sq, sr1_sq, pred1,
         p2, r2_sq, sr2_sq, pred2) = self._joint_fit()

        (N_int, delta_eps, sigma_delta_eps,
         d, sigma_d) = self._excess_fractions(
            S1, eps1, seps1, eps2, seps2, cov_eps)

        f, sigma_f = self._recover_f(S1, sS1, d, sigma_d)

        self.result = TwoLineResult(
            S1=S1, sigma_S1=sS1, S2=S2,
            eps1=eps1, sigma_eps1=seps1,
            eps2=eps2, sigma_eps2=seps2,
            cov_eps=cov_eps,
            chi2_dof=chi2_dof,
            lam_ratio=lam_ratio,
            N_int=N_int,
            delta_eps=delta_eps,
            sigma_delta_eps=sigma_delta_eps,
            d=d, sigma_d=sigma_d,
            f=f, sigma_f=sigma_f,
            p1=p1, r1_sq=r1_sq, sr1_sq=sr1_sq, pred1=pred1,
            p2=p2, r2_sq=r2_sq, sr2_sq=sr2_sq, pred2=pred2,
            lam1_nm=self.lam1_nm,
            lam2_nm=self.lam2_nm,
            r_unit=self.r_unit,
        )
        return self.result

    # ── Public: print_summary ─────────────────────────────────────────────────

    def print_summary(self):
        """Print the joint-fit and recovered-parameter summary to stdout."""
        if self.result is None:
            self.run()
        res = self.result
        u   = self.r_unit

        sep = "═" * 65
        print(f"\n{sep}")
        print("  TWO-LINE JOINT TOLANSKY ANALYSIS")
        print(sep)
        print(f"\n  Step 1 — joint fit  ({len(res.p1)+len(res.p2)} rings,"
              f"  3 free params):")
        print(f"    S₁  = {res.S1:.6g} ± {res.sigma_S1:.4g}  {u}²/fringe"
              f"   [λ₁ = {res.lam1_nm:.4f} nm]")
        print(f"    S₂  = {res.S2:.6g}  (= S₁ × λ₂/λ₁,"
              f"  not a free param)   [λ₂ = {res.lam2_nm:.4f} nm]")
        print(f"    ε₁  = {res.eps1:.7f} ± {res.sigma_eps1:.7f}")
        print(f"    ε₂  = {res.eps2:.7f} ± {res.sigma_eps2:.7f}")
        print(f"    χ²/dof = {res.chi2_dof:.4f}")
        print(f"\n  Step 2 — excess fractions  (N_int = {res.N_int}):")
        print(f"    ε₁ − ε₂  = {res.delta_eps:+.7f}"
              f" ± {res.sigma_delta_eps:.7f}")
        print(f"    d   = {res.d:.6g} ± {res.sigma_d:.4g}  {u}"
              f"   (f-independent)")
        print(f"\n  Step 3 — recover f:")
        print(f"    f   = {res.f:.6g} ± {res.sigma_f:.4g}  {u}")
        print(sep)

    # ── Public: plot_joint ────────────────────────────────────────────────────

    def plot_joint(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Four-panel joint two-line diagnostic figure (Figure 3):

          A) Joint Tolansky plot — both line families on ONE r² vs p axes,
             each with its own fit line (S₂ = S₁·λ₂/λ₁ enforced).
             The slope ratio is visually verified by the parallel lines.

          B) Joint fit residuals for both lines on one panel.
             Systematic trends in one family but not the other would
             signal a fringe-index assignment error.

          C) Slope-ratio verification bar chart.
             Shows  S₂/S₁  vs  λ₂/λ₁  — these must agree within σ.
             Also shows individual-fit slopes vs joint-fit slopes.

          D) Summary: joint fit parameters + recovered d and f.
        """
        if self.result is None:
            self.run()
        res = self.result
        u, u2 = self.r_unit, f"{self.r_unit}²"


        # Use standard matplotlib colors and white background
        BLUE   = 'tab:blue'   # line 1 (640 nm)
        ORANGE = 'tab:orange' # line 2 (638 nm)
        GREEN  = 'tab:green'
        RED    = 'tab:red'
        YELLOW = 'goldenrod'
        GRAY   = 'gray'
        WHITE  = 'black'  # for text
        PURPLE = 'purple'

        fig = plt.figure(figsize=(14, 10), facecolor='white')
        fig.patch.set_facecolor('white')
        gs  = gridspec.GridSpec(2, 2, figure=fig,
                                hspace=0.44, wspace=0.37,
                                left=0.09, right=0.97,
                                top=0.91, bottom=0.08)
        ax_tol = fig.add_subplot(gs[0, 0])   # A
        ax_res = fig.add_subplot(gs[1, 0])   # B
        ax_rat = fig.add_subplot(gs[0, 1])   # C
        ax_txt = fig.add_subplot(gs[1, 1])   # D

        for ax in [ax_tol, ax_res, ax_rat, ax_txt]:
            ax.set_facecolor('white')
            ax.tick_params(colors='black', which='both', direction='in')
            for sp in ax.spines.values():
                sp.set_edgecolor('black')
            ax.xaxis.label.set_color('black')
            ax.yaxis.label.set_color('black')
            ax.title.set_color('black')

        # ── Shared p range for the fit lines ─────────────────────────────────
        p_all  = np.concatenate([res.p1, res.p2])
        p_fine = np.linspace(p_all.min() - 0.3, p_all.max() + 0.3, 300)
        # Reconstruct fit lines from joint parameters
        fit1   = res.S1 * (p_fine - 1.0 + res.eps1)
        fit2   = res.S2 * (p_fine - 1.0 + res.eps2)

        # ── A: joint Tolansky plot ────────────────────────────────────────────
        ax_tol.errorbar(res.p1, res.r1_sq, yerr=res.sr1_sq,
                        fmt='o', color=BLUE, ecolor=GRAY,
                        capsize=4, ms=6, lw=1.4, zorder=3,
                        label=f"$\\lambda_1$ = {res.lam1_nm:.2f} nm")
        ax_tol.errorbar(res.p2, res.r2_sq, yerr=res.sr2_sq,
                        fmt='s', color=ORANGE, ecolor=GRAY,
                        capsize=4, ms=6, lw=1.4, zorder=3,
                        label=f"$\\lambda_2$ = {res.lam2_nm:.2f} nm")
        ax_tol.plot(p_fine, fit1, color=BLUE,   lw=1.8, zorder=2, ls='-',
                    label=f"Fit $S_1$ = {res.S1:.4g}")
        ax_tol.plot(p_fine, fit2, color=ORANGE, lw=1.8, zorder=2, ls='--',
                    label=f"Fit $S_2$ = {res.S2:.4g}")

        ax_tol.set_xlabel("Fringe index  $p$", fontsize=11)
        ax_tol.set_ylabel(f"$r^2$  [{u2}]", fontsize=11)
        ax_tol.set_title("A — Joint Tolansky Plot  (both lines)",
                          fontsize=11, fontweight='bold', pad=7)
        ax_tol.legend(fontsize=8, facecolor='white', labelcolor='black',
                  edgecolor='black', framealpha=0.9, ncol=2)
        ax_tol.text(0.97, 0.05,
                    f"$\\chi^2/\\nu = {res.chi2_dof:.3f}$",
                    transform=ax_tol.transAxes,
                    ha='right', va='bottom', fontsize=9, color=GREEN)

        # ── B: joint residuals ────────────────────────────────────────────────
        resid1 = res.r1_sq - res.pred1
        resid2 = res.r2_sq - res.pred2

        ax_res.axhline(0, color=GRAY, lw=1.0, ls='--', zorder=1)
        ax_res.errorbar(res.p1, resid1, yerr=res.sr1_sq,
                        fmt='o', color=BLUE, ecolor=GRAY,
                        capsize=4, ms=6, lw=1.4, zorder=3,
                        label=f"$\\lambda_1$")
        ax_res.errorbar(res.p2, resid2, yerr=res.sr2_sq,
                        fmt='s', color=ORANGE, ecolor=GRAY,
                        capsize=4, ms=6, lw=1.4, zorder=3,
                        label=f"$\\lambda_2$")
        ax_res.set_xlabel("Fringe index  $p$", fontsize=11)
        ax_res.set_ylabel(f"Residual  [{u2}]", fontsize=11)
        ax_res.set_title("B — Joint Fit Residuals",
                          fontsize=11, fontweight='bold', pad=7)
        ax_res.legend(fontsize=9, facecolor='white', labelcolor='black',
                  edgecolor='black', framealpha=0.9)

        # ── C: slope-ratio verification ───────────────────────────────────────
        # Gather: individual slopes from each single-line fit,
        # joint slopes, and the theoretical λ ratio.
        S1_ind  = self.a1.result.slope
        S2_ind  = self.a2.result.slope
        ratio_ind  = S2_ind  / S1_ind      # from independent fits
        ratio_jnt  = res.S2  / res.S1      # from joint fit (enforced = lam_ratio)
        ratio_lam  = res.lam_ratio         # λ₂/λ₁ — the physical truth

        # Uncertainty on ratio_ind via error propagation
        sig_S1_ind = self.a1.result.sigma_slope
        sig_S2_ind = self.a2.result.sigma_slope
        sigma_ratio_ind = ratio_ind * np.sqrt(
            (sig_S1_ind / S1_ind)**2 + (sig_S2_ind / S2_ind)**2)

        categories = ['Independent\nfit ratio\n$S_2/S_1$',
                      'Joint fit\nenforced\n$\\lambda_2/\\lambda_1$',
                      'Physical\nwavelength\nratio']
        values  = [ratio_ind,  ratio_jnt,  ratio_lam]
        errors  = [sigma_ratio_ind, 0.0, 0.0]
        colours = [YELLOW, BLUE, GREEN]

        bars = ax_rat.bar(categories, values, color=colours, alpha=0.75,
                          width=0.5, zorder=2)
        ax_rat.errorbar(categories, values, yerr=errors,
                        fmt='none', color=WHITE, capsize=6, lw=1.8, zorder=3)

        # Mark the physical ratio as a horizontal reference
        ax_rat.axhline(ratio_lam, color=GREEN, lw=1.2, ls='--', zorder=1,
                       label=f"$\\lambda_2/\\lambda_1$ = {ratio_lam:.6f}")

        # Zoom y-axis to the interesting range
        spread = max(abs(ratio_ind - ratio_lam) * 3, 1e-5)
        ax_rat.set_ylim(ratio_lam - spread, ratio_lam + spread)
        ax_rat.set_ylabel("Slope ratio  $S_2 / S_1$", fontsize=11)
        ax_rat.set_title("C — Slope Ratio Consistency Check",
                          fontsize=11, fontweight='bold', pad=7)
        ax_rat.legend(fontsize=8.5, facecolor='white', labelcolor='black',
                  edgecolor='black', framealpha=0.9)
        ax_rat.tick_params(axis='x', labelsize=8.5)
        for label in ax_rat.get_xticklabels():
            label.set_color(WHITE)

        # Annotate deviation in units of sigma
        if sigma_ratio_ind > 0:
            pull = abs(ratio_ind - ratio_lam) / sigma_ratio_ind
            pull_col = GREEN if pull < 2 else (YELLOW if pull < 3 else RED)
            ax_rat.text(0.5, 0.06,
                        f"Independent fit: |Δratio|/σ = {pull:.2f}",
                        transform=ax_rat.transAxes,
                        ha='center', va='bottom', fontsize=9,
                        color=pull_col)

        # ── D: summary text ───────────────────────────────────────────────────
        ax_txt.axis('off')


        # Convert d, f to mm, alpha to rad/px, and compute two_sigma values
        PIXEL_M = 32e-6
        d_mm = res.d * PIXEL_M * 1e3
        sigma_d_mm = res.sigma_d * PIXEL_M * 1e3
        two_sigma_d_mm = 2 * sigma_d_mm
        f_mm = res.f * PIXEL_M * 1e3
        sigma_f_mm = res.sigma_f * PIXEL_M * 1e3
        two_sigma_f_mm = 2 * sigma_f_mm
        alpha = PIXEL_M / (f_mm * 1e-3) if f_mm != 0 else float('nan')
        sigma_alpha = abs(alpha * sigma_f_mm / f_mm) if f_mm != 0 else float('nan')
        two_sigma_alpha = 2 * sigma_alpha

        lines_txt = [
            ("JOINT TWO-LINE SUMMARY",         WHITE,  11,   'bold'),
            ("",                               WHITE,   3,   'normal'),
            (f"N rings : {len(res.p1)+len(res.p2)}"
             f"  ({len(res.p1)} + {len(res.p2)})", GRAY, 9.5, 'normal'),
            (f"n (gap) : {self.n:.3f}",        GRAY,   9.5, 'normal'),
            ("",                               WHITE,   3,   'normal'),
            ("── Instrument parameters ────────", BORDER, 8.5, 'normal'),
            (f"d = {d_mm:.5f} ± {sigma_d_mm:.3g} mm (2σ: ±{two_sigma_d_mm:.3g})", GREEN, 9.5, 'bold'),
            (f"f = {f_mm:.2f} ± {sigma_f_mm:.2g} mm (2σ: ±{two_sigma_f_mm:.2g})", GREEN, 9.5, 'bold'),
            (f"α = {alpha:.3e} ± {sigma_alpha:.1e} rad/px (2σ: ±{two_sigma_alpha:.1e})", PURPLE, 9.5, 'bold'),
            ("",                               WHITE,   3,   'normal'),
            ("── Fractional orders ────────────", BORDER, 8.5, 'normal'),
            (f"ε₁ = {res.eps1:.6f} ± {res.sigma_eps1:.2g}", BLUE, 9.5, 'normal'),
            (f"ε₂ = {res.eps2:.6f} ± {res.sigma_eps2:.2g}", ORANGE, 9.5, 'normal'),
            (f"ε₁−ε₂ = {res.delta_eps:+.6f} ± {res.sigma_delta_eps:.2g}", PURPLE, 9.5, 'normal'),
            ("",                               WHITE,   3,   'normal'),
            ("── Fit diagnostics ──────────────", BORDER, 8.5, 'normal'),
            (f"S₁ = {res.S1:.5g} ± {res.sigma_S1:.3g}  {u2}/fr", BLUE, 9.5, 'normal'),
            (f"S₂ = {res.S2:.5g}  (enforced)", ORANGE, 9.5, 'normal'),
            (f"χ²/ν = {res.chi2_dof:.4f}", GREEN if res.chi2_dof < 2 else YELLOW, 9.5, 'normal'),
            (f"N_int = {res.N_int}", GRAY, 9.5, 'normal'),
        ]
        y = 0.97
        for text, color, size, weight in lines_txt:
            ax_txt.text(0.04, y, text, transform=ax_txt.transAxes,
                        ha='left', va='top', fontsize=size,
                        color=color, fontweight=weight,
                        fontfamily='monospace')
            y -= size * 0.013 + 0.010

        fig.suptitle(
            "Tolansky Method  —  Joint Two-Line Analysis"
            f"  ($\\lambda_1$ = {res.lam1_nm:.2f} nm,"
            f"  $\\lambda_2$ = {res.lam2_nm:.2f} nm)",
            color='black', fontsize=13, fontweight='bold', y=0.97)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK)
            print(f"  Figure saved → {save_path}")
        return fig


# ─────────────────────────────────────────────────────────────────────────────
# Entry point  (python tolansky.py  [path/to/radial_profile_peaks.npy])
# ─────────────────────────────────────────────────────────────────────────────
#
# Ingests radial_profile_peaks.npy produced by annular_reduction.py.
# Expected columns (float64, shape N×6):
#
#   col 0  peak_num      detection index  (1 = innermost)
#   col 1  r_raw_px      raw detected radius  [px]       — not used
#   col 2  r_fit_px      Gaussian centroid    [px]       ← r input
#   col 3  sigma_r_fit   1σ uncertainty on centroid [px] ← sigma_r input
#   col 4  amplitude_adu peak amplitude [ADU]  ← used to split the two Ne lines
#   col 5  width_px      Gaussian sigma width [px]       — not used
#
# ── Two-line neon structure ──────────────────────────────────────────────────
#
# The WindCube neon calibration lamp excites two strong lines that are
# 188 free spectral ranges apart and therefore produce two INTERLEAVED sets
# of concentric rings on the detector.  The stronger line (640.2248 nm)
# produces the larger-amplitude peaks; the weaker (638.2990 nm) the smaller.
#
# The Tolansky method requires that p = 1, 2, 3, … indexes consecutive orders
# of a SINGLE spectral line.  Mixing both lines doubles the apparent order
# spacing, halving the measured slope S and recovering d ≈ 2 × d_true.
#
# Fix: split the peaks by amplitude (threshold = 1000 ADU) and run a
# separate Tolansky analysis on each line.
#
# ── WindCube instrument constants ────────────────────────────────────────────
#
#   pixel pitch   :  32 µm  (CCD97-00, 2×2 binned)
#   focal length  : 200 mm  (→ α = 32e-6/0.200 = 1.6e-4 rad/px)
#   n (gap)       :   1.0   (vacuum gap)
#   λ₁ (Ne)       : 640.2248 nm  — strong line   (IC Optical GNL-4096-R)
#   λ₂ (Ne)       : 638.2990 nm  — secondary line (IC Optical GNL-4096-R)
#   d  (ICOS)     :  20.008 mm   (spacer measurement, build report §7.4)
#
# All quantities are kept in PIXELS so that r, f, and d share one unit:
#   f_px  = 200 mm  / 0.032 mm  = 6250.00 px
#   d_px  = 20.008 mm / 0.032 mm = 625.25 px
#   λ_px  = λ_nm × 1e-9 / 32e-6  (conversion factor = 3.125e-5 px/nm)
#
# Four analyses are run (one per line × two modes):
#   Run 1a  λ₁ known  → recover d        (sanity-check vs ICOS)
#   Run 1b  d  known  → recover λ₁       (compare to Ne atlas)
#   Run 2a  λ₂ known  → recover d
#   Run 2b  d  known  → recover λ₂

if __name__ == "__main__":
    import sys
    import pathlib
    import tkinter as tk
    from tkinter import filedialog

    # ── Locate the peaks file ─────────────────────────────────────────────────
    if len(sys.argv) > 1:
        peaks_path = sys.argv[1]
    else:
        root = tk.Tk()
        root.withdraw()
        peaks_path = filedialog.askopenfilename(
            title="Select radial_profile_peaks.npy",
            filetypes=[("NumPy array", "*.npy"), ("All files", "*.*")],
        )
        root.destroy()
        if not peaks_path:
            print("No file selected — exiting.")
            sys.exit(0)

    peaks = np.load(peaks_path)
    if peaks.ndim != 2 or peaks.shape[1] != 6:
        raise ValueError(
            f"Expected shape (N, 6) from radial_profile_peaks.npy, "
            f"got {peaks.shape}.  "
            f"Columns: peak_num | r_raw_px | r_fit_px | "
            f"sigma_r_fit_px | amplitude_adu | width_px"
        )

    print(f"\n  Loaded  {peaks_path}")
    print(f"  Total peaks : {len(peaks)}")

    # ── Split into the two Ne emission lines by amplitude ─────────────────────
    # The stronger Ne line (640.2248 nm) produces higher-amplitude peaks.
    # A threshold of 1000 ADU cleanly separates the two populations for
    # this dataset; adjust AMP_THRESHOLD below if needed.
    AMP_THRESHOLD = 1000.0    # ADU — peaks above → line 1, below → line 2

    mask1 = peaks[:, 4] >= AMP_THRESHOLD   # strong line (640.2248 nm)
    mask2 = ~mask1                          # weak   line (638.2990 nm)

    if mask1.sum() == 0 or mask2.sum() == 0:
        print(f"\n  WARNING: amplitude split at {AMP_THRESHOLD} ADU produced "
              f"an empty group ({mask1.sum()} / {mask2.sum()} peaks).  "
              f"Adjust AMP_THRESHOLD and rerun.")
        sys.exit(1)

    def _extract(mask):
        """Return (p, r_px, sig_r) re-indexed from 1 for the selected peaks."""
        r   = peaks[mask, 2]
        sig = peaks[mask, 3]
        p   = np.arange(1, mask.sum() + 1, dtype=float)
        return p, r, sig

    p1, r1, s1 = _extract(mask1)
    p2, r2, s2 = _extract(mask2)

    print(f"\n  Line 1 (amp ≥ {AMP_THRESHOLD:.0f} ADU) : "
          f"{len(p1)} rings,  r = {r1[0]:.2f} – {r1[-1]:.2f} px")
    print(f"  Line 2 (amp <  {AMP_THRESHOLD:.0f} ADU) : "
          f"{len(p2)} rings,  r = {r2[0]:.2f} – {r2[-1]:.2f} px")

    # ── Instrument constants (pixels) ─────────────────────────────────────────
    PIXEL_M         = 32e-6             # pixel pitch  [m]
    F_PX            = 200e-3 / PIXEL_M  # focal length   [px]  = 6250.00
    N_GAP           = 1.0               # refractive index (vacuum gap)
    LAM1_NM         = 640.2248          # Ne strong line  [nm]
    LAM2_NM         = 638.2990          # Ne weak   line  [nm]
    D_ICOS_MM       = 20.008            # ICOS mechanical measurement  [mm]
    D_PX            = D_ICOS_MM * 1e-3 / PIXEL_M   # [px]  = 625.25
    LAM_UNIT_PER_NM = 1e-9 / PIXEL_M   # nm → px  = 3.125e-5

    print(f"\n  Instrument constants:")
    print(f"    pixel pitch  = {PIXEL_M*1e6:.0f} µm")
    print(f"    f            = {F_PX:.2f} px  =  {F_PX*PIXEL_M*1e3:.1f} mm")
    print(f"    d (ICOS)     = {D_PX:.4f} px  =  {D_ICOS_MM:.4f} mm")
    print(f"    λ₁ (Ne)      = {LAM1_NM:.4f} nm")
    print(f"    λ₂ (Ne)      = {LAM2_NM:.4f} nm")
    print(f"    n            = {N_GAP:.1f}")

    # ── Helper: run one Tolansky pair (recover d  and  recover λ) ─────────────
    def run_pair(p, r, sig, lam_nm, lam_label, run_prefix):
        sep = "═" * 65

        # (a) known λ → recover d
        print(f"\n{sep}")
        print(f"  {run_prefix}a:  known λ = {lam_nm:.4f} nm  →  recover d")
        print(sep)
        ana = TolanskyAnalyser(
            p=p, r=r, sigma_r=sig,
            r_unit="px",
            lam_nm=lam_nm, n=N_GAP, f=F_PX, d=None,
            lam_unit_per_nm=LAM_UNIT_PER_NM,
        )
        res = ana.run()
        ana.print_table()
        d_mm     = res.recovered_d * PIXEL_M * 1e3
        sig_d_mm = res.sigma_d     * PIXEL_M * 1e3
        pull_d   = abs(d_mm - D_ICOS_MM) / sig_d_mm
        print(f"  → d         = {d_mm:.6f} ± {sig_d_mm:.6f} mm")
        print(f"  ICOS  d     = {D_ICOS_MM:.6f} mm")
        print(f"  Δ           = {d_mm - D_ICOS_MM:+.6f} mm   "
              f"(|Δ|/σ = {pull_d:.1f})")

        # (b) known d → recover λ
        print(f"\n{sep}")
        print(f"  {run_prefix}b:  known d = {D_ICOS_MM:.4f} mm  →  recover λ")
        print(sep)
        ana2 = TolanskyAnalyser(
            p=p, r=r, sigma_r=sig,
            r_unit="px",
            lam_nm=None, n=N_GAP, f=F_PX, d=D_PX,
            lam_unit_per_nm=LAM_UNIT_PER_NM,
        )
        res2 = ana2.run()
        ana2.print_table()
        pull_lam = abs(res2.recovered_lam_nm - lam_nm) / res2.sigma_lam_nm
        print(f"  → λ         = {res2.recovered_lam_nm:.4f} "
              f"± {res2.sigma_lam_nm:.4f} nm")
        print(f"  {lam_label}  = {lam_nm:.4f} nm")
        print(f"  Δ           = {res2.recovered_lam_nm - lam_nm:+.4f} nm   "
              f"(|Δ|/σ = {pull_lam:.1f})")

        return ana, res   # return the "recover d" analyser for plotting

    ana1, res1 = run_pair(p1, r1, s1, LAM1_NM, "Ne atlas λ₁", "RUN 1")
    ana2, res2 = run_pair(p2, r2, s2, LAM2_NM, "Ne atlas λ₂", "RUN 2")

    # ════════════════════════════════════════════════════════════════════════
    # RUN 3 — Joint two-line analysis (TwoLineAnalyser)
    #
    # Uses TwoLineAnalyser to jointly fit both ring families, enforce the
    # slope ratio S₂/S₁ = λ₂/λ₁, recover d via excess fractions (Benoit
    # 1898), and derive f from S₁ + d.  See TwoLineAnalyser docstring for
    # the full derivation.
    # ════════════════════════════════════════════════════════════════════════

    tla = TwoLineAnalyser(
        analyser1       = ana1,
        analyser2       = ana2,
        lam1_nm         = LAM1_NM,
        lam2_nm         = LAM2_NM,
        d_prior         = D_PX,              # ICOS prior in pixels
        lam_unit_per_nm = LAM_UNIT_PER_NM,
        n               = N_GAP,
    )
    tla_res = tla.run()
    tla.print_summary()

    # Convert pixel results to mm for the α calculation
    d_px_joint  = tla_res.d
    f_px_joint  = tla_res.f
    d_mm_joint  = d_px_joint * PIXEL_M * 1e3
    f_mm_joint  = f_px_joint * PIXEL_M * 1e3
    sigma_d_mm  = tla_res.sigma_d * PIXEL_M * 1e3
    sigma_f_mm  = tla_res.sigma_f * PIXEL_M * 1e3

    alpha_fit   = PIXEL_M / (f_mm_joint * 1e-3)       # rad/px
    sigma_alpha = alpha_fit * sigma_f_mm / f_mm_joint

    print(f"\n  Derived in mm / rad:")
    print(f"    d   = {d_mm_joint:.5f} ± {sigma_d_mm:.5f} mm"
          f"   (ICOS = {D_ICOS_MM:.5f} mm,"
          f"  Δd = {d_mm_joint - D_ICOS_MM:+.5f} mm)")
    print(f"    f   = {f_mm_joint:.5f} ± {sigma_f_mm:.5f} mm"
          f"   (nominal 200.000 mm,"
          f"  {(f_mm_joint-200.0)/200.0*100:+.3f}%)")
    print(f"    α   = {alpha_fit:.6e} ± {sigma_alpha:.2e} rad/px"
          f"   (M05 initial guess)")

    # ── Save all three figures ────────────────────────────────────────────────
    out_dir = pathlib.Path(peaks_path).parent

    fig1_path = str(out_dir / "tolansky_line1_640nm.png")
    fig1 = ana1.plot(save_path=fig1_path)

    fig2_path = str(out_dir / "tolansky_line2_638nm.png")
    fig2 = ana2.plot(save_path=fig2_path)

    fig3_path = str(out_dir / "tolansky_joint_two_line.png")
    fig3 = tla.plot_joint(save_path=fig3_path)

    plt.show()
