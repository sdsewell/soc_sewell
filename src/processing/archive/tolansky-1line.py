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
  col 4–8 : not used here
Cols 2–8 are NaN when the Gaussian fit failed for that peak (those rows are
dropped before analysis).

OUTPUT
------
  - Δ (slope of r² vs p) with 2σ uncertainty
  - ε (fractional fringe order at the geometric centre) with 2σ uncertainty
  - χ²_ν (reduced chi-square of the WLS fit)
  - Successive Δ(r²) and CV — linearity / parallelism diagnostic
  - Four-panel diagnostic figure saved alongside the input file

Physical interpretation of Δ (recovering d, λ, or f) is handled by S13a /
tolansky-2line.py.

PHYSICAL MODEL
--------------
Constructive interference (Haidinger fringes):

    m λ = 2 n d cos θ

Writing m = m₀ - (p - 1 + ε) for fringe index p and fractional order ε,
and using the paraxial substitution cos θ ≈ 1 - r²/(2f²):

    r_p² = Δ · (p − 1 + ε)        where  Δ = f² λ / (n d)

So r² is LINEAR in fringe index p:

    Slope      Δ = f² λ / (n d)        [r_unit² / fringe]
    Intercept  b = Δ · (ε − 1)
    ε            = fractional order at centre  (0 ≤ ε < 1)

WEIGHTED LEAST-SQUARES
-----------------------
σ(r²) = 2r σ_r varies across rings, so a weighted fit is used:

    weights w_p = 1 / σ(r²_p)²

This gives the correct covariance matrix and propagated parameter uncertainties.
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
    """All outputs from a single Tolansky WLS analysis."""

    # ── Derived input columns ───────────────────────────────────────────────
    p:          np.ndarray   # fringe index
    r:          np.ndarray   # radius
    sigma_r:    np.ndarray   # 1σ uncertainty on r
    r_sq:       np.ndarray   # r²
    sigma_r_sq: np.ndarray   # 1σ uncertainty on r²  = 2r·σ_r

    # ── Weighted least-squares fit ──────────────────────────────────────────
    slope:        float      # Δ — slope of r² vs p
    sigma_slope:  float      # 1σ uncertainty on Δ
    intercept:    float      # b = Δ(ε − 1)
    sigma_int:    float      # 1σ uncertainty on b
    r2_fit:       float      # coefficient of determination R²
    chi2_dof:     float      # reduced chi-square  χ²/(N−2)

    # ── Derived quantities ──────────────────────────────────────────────────
    epsilon:       float       # fractional order at centre  (0 ≤ ε < 1)
    sigma_epsilon: float       # 1σ uncertainty on ε
    delta_r_sq:    np.ndarray  # successive Δ(r²)
    sigma_delta:   np.ndarray  # 1σ uncertainty on each Δ(r²)


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class TolanskyAnalyser:
    """
    Tolansky r² WLS analysis on measured FPI fringe ring radii.

    Fits r² = Δ·p + b and returns slope Δ and fractional order ε with
    their uncertainties.  Physical interpretation of Δ is left to the caller.

    Parameters
    ----------
    p        : array_like    fringe indices  (1 = innermost)
    r        : array_like    ring radii  [any consistent unit]
    sigma_r  : array_like    1σ uncertainty on each radius
    r_unit   : str           display label for the radius unit  (default "px")
    """

    def __init__(
        self,
        p:       "array_like",
        r:       "array_like",
        sigma_r: "array_like",
        r_unit:  str = "px",
    ):
        self.p       = np.asarray(p,       dtype=float)
        self.r       = np.asarray(r,       dtype=float)
        self.sigma_r = np.asarray(sigma_r, dtype=float)
        self.r_unit  = r_unit
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

    # ── Step 1: r → r² with uncertainty propagation ──────────────────────────

    def _derive_r_squared(self):
        r_sq       = self.r ** 2
        sigma_r_sq = 2.0 * self.r * self.sigma_r
        return r_sq, sigma_r_sq

    # ── Step 2: weighted least-squares r² = Δ·p + b ─────────────────────────

    @staticmethod
    def _wls(x, y, w):
        """
        Weighted least-squares  y = S·x + b.

        Normal equations (Bevington & Robinson §6.3):

            Λ = Σw · Σwx² − (Σwx)²
            S = (Σw · Σwxy  −  Σwx · Σwy) / Λ
            b = (Σwx² · Σwy  −  Σwx · Σwxy) / Λ

            Var(S) = Σw  / Λ
            Var(b) = Σwx² / Λ

        Returns (slope, sigma_slope, intercept, sigma_intercept, R², χ²_ν)
        """
        sw   = w.sum()
        swx  = (w * x).sum()
        swy  = (w * y).sum()
        swxx = (w * x**2).sum()
        swxy = (w * x * y).sum()

        lam = sw * swxx - swx**2
        if lam == 0:
            raise ValueError("Degenerate fit: check fringe indices are distinct.")

        S = (sw * swxy - swx * swy) / lam
        b = (swxx * swy - swx * swxy) / lam

        sigma_S = np.sqrt(sw   / lam)
        sigma_b = np.sqrt(swxx / lam)

        y_hat    = S * x + b
        ss_res   = (w * (y - y_hat)**2).sum()
        ss_tot   = (w * (y - swy / sw)**2).sum()
        R2       = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        chi2_dof = ss_res / max(len(x) - 2, 1)

        return S, sigma_S, b, sigma_b, R2, chi2_dof

    # ── Step 3: fractional order ε ────────────────────────────────────────────

    @staticmethod
    def _epsilon(S, b, sigma_S, sigma_b):
        """
        From  b = Δ(ε − 1):   ε = 1 + b/Δ  (wrapped to [0, 1))

        σ_ε² = (σ_b / S)²  +  (b · σ_S / S²)²   (conservative, uncorrelated)
        """
        epsilon   = (1.0 + b / S) % 1.0
        sigma_eps = np.sqrt((sigma_b / S)**2 + (b * sigma_S / S**2)**2)
        return epsilon, sigma_eps

    # ── Step 4: successive Δ(r²) ─────────────────────────────────────────────

    @staticmethod
    def _successive_diffs(r_sq, sigma_r_sq):
        """
        Δ(r²)_k  = r²_{k+1} − r²_k
        σ_k      = sqrt(σ_{k+1}² + σ_k²)

        CV = std/mean × 100 % should be < ~2 % for a parallel etalon.
        """
        return np.diff(r_sq), np.sqrt(sigma_r_sq[1:]**2 + sigma_r_sq[:-1]**2)

    # ── Public: run ───────────────────────────────────────────────────────────

    def run(self) -> TolanskyResult:
        """Execute WLS fit and return a TolanskyResult."""
        r_sq, sigma_r_sq = self._derive_r_squared()
        w                = 1.0 / sigma_r_sq**2
        S, sS, b, sb, R2, chi2_dof = self._wls(self.p, r_sq, w)
        eps, seps        = self._epsilon(S, b, sS, sb)
        delta, sdelta    = self._successive_diffs(r_sq, sigma_r_sq)

        self.result = TolanskyResult(
            p=self.p, r=self.r, sigma_r=self.sigma_r,
            r_sq=r_sq, sigma_r_sq=sigma_r_sq,
            slope=S, sigma_slope=sS,
            intercept=b, sigma_int=sb,
            r2_fit=R2, chi2_dof=chi2_dof,
            epsilon=eps, sigma_epsilon=seps,
            delta_r_sq=delta, sigma_delta=sdelta,
        )
        return self.result

    # ── Public: print_table ───────────────────────────────────────────────────

    def print_table(self):
        """Print the full Tolansky data table and fit summary."""
        if self.result is None:
            self.run()
        res = self.result
        assert res is not None
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
        print(f"\n  Weighted linear fit:   r² = Δ · p + b")
        print(f"    Δ    = {res.slope:.6g} ± {2*res.sigma_slope:.6g}  {u2}/fringe  (2σ)")
        print(f"    ε    = {res.epsilon:.5f} ± {2*res.sigma_epsilon:.5f}  (2σ)")
        print(f"    χ²_ν = {res.chi2_dof:.4f}")
        cv = res.delta_r_sq.std() / abs(res.delta_r_sq.mean()) * 100
        print(f"    CV(Δr²) = {cv:.1f} %  {'✓' if cv < 5 else '⚠'}")
        print()

    # ── Public: plot ──────────────────────────────────────────────────────────

    def plot(self, save_path: Optional[str] = None,
             peaks_filename: Optional[str] = None) -> plt.Figure:
        """
        Four-panel diagnostic figure:
          A) Tolansky plot: r² vs p with weighted fit and ±1σ error bars
          B) Fit residuals
          C) Successive Δ(r²) — linearity / parallelism diagnostic
          D) Summary: Δ ± 2σ, ε ± 2σ, χ²_ν, CV
        """
        if self.result is None:
            self.run()
        res = self.result
        assert res is not None
        u, u2 = self.r_unit, f"{self.r_unit}²"

        BG     = 'white'
        PANEL  = '#f6f8fa'
        BORDER = '#d0d7de'
        ACCENT = '#0969da'
        GREEN  = '#1a7f37'
        RED    = '#cf222e'
        YELLOW = '#9a6700'
        GRAY   = '#57606a'
        TEXT   = '#1f2328'

        fig = plt.figure(figsize=(14, 10), facecolor=BG)
        fig.patch.set_facecolor(BG)
        gs  = gridspec.GridSpec(2, 2, figure=fig,
                                hspace=0.44, wspace=0.37,
                                left=0.09, right=0.97,
                                top=0.88, bottom=0.08)
        ax_tol = fig.add_subplot(gs[0, 0])
        ax_res = fig.add_subplot(gs[1, 0])
        ax_dr2 = fig.add_subplot(gs[0, 1])
        ax_txt = fig.add_subplot(gs[1, 1])

        for ax in [ax_tol, ax_res, ax_dr2, ax_txt]:
            ax.set_facecolor(PANEL)
            ax.tick_params(colors=TEXT, which='both', direction='in')
            for sp in ax.spines.values():
                sp.set_edgecolor(BORDER)
            ax.xaxis.label.set_color(TEXT)
            ax.yaxis.label.set_color(TEXT)
            ax.title.set_color(TEXT)

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
        ax_tol.legend(fontsize=8.5, facecolor=PANEL, labelcolor=TEXT,
                      edgecolor=BORDER, framealpha=0.9)
        ax_tol.text(0.97, 0.05, f"$\\chi^2_\\nu = {res.chi2_dof:.4f}$",
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
                       label=f"Δ = {res.slope:.4g}")
        ax_dr2.errorbar(p_mid, res.delta_r_sq, yerr=res.sigma_delta,
                        fmt='^', color=RED, ecolor=GRAY,
                        capsize=4, ms=7, lw=1.4, zorder=3)
        ax_dr2.set_xlabel("Fringe index  $p$  (midpoint)", fontsize=11)
        ax_dr2.set_ylabel(f"$\\Delta(r^2)$  [{u2}]", fontsize=11)
        ax_dr2.set_title("C — Successive  $\\Delta(r^2)$",
                          fontsize=11, fontweight='bold', pad=7)
        ax_dr2.legend(fontsize=8.5, facecolor=PANEL, labelcolor=TEXT,
                      edgecolor=BORDER, framealpha=0.9)
        cv_col   = GREEN if cv < 2 else (YELLOW if cv < 5 else RED)
        cv_label = "✓ parallel" if cv < 5 else "⚠ check alignment"
        ax_dr2.text(0.97, 0.07, f"CV = {cv:.1f}%\n({cv_label})",
                    transform=ax_dr2.transAxes,
                    ha='right', va='bottom', fontsize=9, color=cv_col,
                    multialignment='right')

        # ── D: Summary ────────────────────────────────────────────────────────
        ax_txt.axis('off')
        chi2_col = GREEN if 0.5 < res.chi2_dof < 2.0 else YELLOW
        lines = [
            ("TOLANSKY SUMMARY",                                TEXT,     11, 'bold'),
            ("",                                                TEXT,      3, 'normal'),
            (f"N rings : {len(res.p)}",                         GRAY,    9.5, 'normal'),
            ("",                                                TEXT,      3, 'normal'),
            ("── WLS Fit ─────────────────────",                BORDER,  8.5, 'normal'),
            (f"Δ  = {res.slope:.5g} ± {2*res.sigma_slope:.3g}"
             f"  {u2}/fr  (2σ)",                                ACCENT,  9.5, 'normal'),
            (f"ε  = {res.epsilon:.5f} ± {2*res.sigma_epsilon:.5f}"
             f"  (2σ)",                                         ACCENT,  9.5, 'normal'),
            (f"χ²_ν = {res.chi2_dof:.4f}",                     chi2_col, 9.5, 'normal'),
            (f"CV(Δr²) = {cv:.2f} %  {'✓' if cv < 5 else '⚠'}",
             cv_col, 9.5, 'normal'),
        ]
        y = 0.97
        for text, color, size, weight in lines:
            ax_txt.text(0.04, y, text, transform=ax_txt.transAxes,
                        ha='left', va='top', fontsize=size,
                        color=color, fontweight=weight,
                        fontfamily='monospace')
            y -= size * 0.013 + 0.010

        fig.suptitle("Tolansky Method  —  FPI Fringe Ring Analysis",
                     color=TEXT, fontsize=13, fontweight='bold', y=0.99)
        if peaks_filename:
            fig.text(0.5, 0.955, peaks_filename,
                     ha='center', va='top', fontsize=9, color=GRAY,
                     fontfamily='monospace')

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BG)
            print(f"  Figure saved → {save_path}")
        return fig


# ─────────────────────────────────────────────────────────────────────────────
# Entry point  (python tolansky-1line.py  [path/to/<stem>_peak_fits.npy])
# ─────────────────────────────────────────────────────────────────────────────
#
# Ingests the _peak_fits.npy file produced by annular_reduction.py.
# All valid (non-NaN) peaks are treated as a single spectral line.
# Outputs Δ ± 2σ and ε ± 2σ from the WLS fit r² = Δ·p + b.
#
# Columns used from the .npy array:
#   col 2 : r_fit (px)        — TRF Gaussian centroid μ  ← r input
#   col 3 : sigma_r_fit (px)  — 1σ uncertainty on μ      ← sigma_r input
# Cols 2–8 are NaN when the Gaussian fit failed for that peak.

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
    r       = peaks[:, 2]   # r_fit (px)       — TRF Gaussian centroid μ
    sigma_r = peaks[:, 3]   # sigma_r_fit (px) — 1σ uncertainty on μ

    print(f"  Valid peaks : {len(peaks)}")
    print(f"  r range     : {r[0]:.2f} – {r[-1]:.2f} px")

    # ── Run WLS fit ───────────────────────────────────────────────────────────
    ana = TolanskyAnalyser(p=p, r=r, sigma_r=sigma_r, r_unit="px")
    ana.run()
    ana.print_table()

    # ── Save figure ───────────────────────────────────────────────────────────
    out_dir  = peaks_path.parent
    stem     = peaks_path.stem.replace("_peak_fits", "")
    fig_path = str(out_dir / f"{stem}_tolansky_1line.png")
    ana.plot(save_path=fig_path, peaks_filename=peaks_path.name)

    plt.show()
