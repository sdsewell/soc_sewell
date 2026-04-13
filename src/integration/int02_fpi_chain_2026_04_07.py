"""
INT02 — FPI instrument chain integration script.

Spec:        docs/specs/S17_int02_fpi_chain_2026-04-07.md
Spec date:   2026-04-07
Tool:        Claude Code
Depends on:  src.constants, src.fpi (M01–M06, Tolansky), src.metadata (P01)
Usage:
    python src/integration/int02_fpi_chain_2026_04_07.py
    python src/integration/int02_fpi_chain_2026_04_07.py --quick
"""

import argparse
import pathlib
import sys

# Ensure project root on sys.path
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.constants import (
    NE_WAVELENGTH_1_M,
    NE_WAVELENGTH_2_M,
    NE_INTENSITY_2,
    OI_WAVELENGTH_M,
    ETALON_GAP_M,
    WIND_BIAS_BUDGET_MS,
    SPEED_OF_LIGHT_MS,
    CCD_DARK_RATE_E_PX_S,
)
from src.fpi import (
    InstrumentParams,
    airy_modified,
    synthesise_calibration_image,
    synthesise_airglow_image,
    make_master_dark,
    subtract_dark,
    reduce_calibration_frame,
    reduce_science_frame,
    TolanskyPipeline,
    FitConfig,
    fit_calibration_fringe,
    fit_airglow_fringe,
)
from src.metadata import build_synthetic_metadata, ImageMetadata

OUTPUT_DIR = pathlib.Path("outputs")


# ===========================================================================
# Stage A — Setup and instrument parameters
# ===========================================================================

def stage_a_setup(quick: bool) -> dict:
    """
    Initialise InstrumentParams and test configuration.
    Print instrument parameter summary.
    Return config dict used by all subsequent stages.
    """
    params   = InstrumentParams()
    rng_cal  = np.random.default_rng(seed=0)
    rng_dark = np.random.default_rng(seed=1)
    rng_sci  = {v: np.random.default_rng(seed=100 + i)
                for i, v in enumerate([-300.0, 0.0, 300.0])}

    config = {
        "params":     params,
        "rng_cal":    rng_cal,
        "rng_dark":   rng_dark,
        "rng_sci":    rng_sci,
        "r_max_px":   params.r_max,
        "n_bins":     150,
        "snr":        5.0,
        "v_truth_ms": [-300.0, 0.0, 300.0],
        "quick":      quick,
    }

    print("\u2550" * 54)
    print("  WindCube INT02 \u2014 FPI Instrument Chain Integration")
    print("\u2550" * 54)
    print("InstrumentParams:")
    print(f"  t (etalon gap) : {params.t_m * 1e3:.6f} mm")
    print(f"  R (reflectivity): {params.R_refl:.3f}")
    print(f"  alpha           : {params.alpha:.4e} rad/px")
    print(f"  r_max           : {params.r_max:.1f} px")
    print(f"  sigma0 (PSF)    : {params.sigma0:.3f} px")
    print(f"  B (bias)        : {params.B:.1f} ADU")
    print(f"Running in {'QUICK' if quick else 'FULL'} mode.")

    return config


# ===========================================================================
# Stage B — Dark frame synthesis and M03 dark subtraction
# ===========================================================================

def stage_b_dark(config: dict) -> dict:
    """
    Synthesise a dark frame at the nominal CCD operating temperature.
    Build master dark. Verify dark level is physically reasonable.
    Return dark arrays for use in Stages C and G.
    """
    params = config["params"]

    # InstrumentParams does not define exp_time_s — use 5.0 s nominal
    exp_time_s = getattr(params, "exp_time_s", 5.0)

    dark_rate_adu_per_s = CCD_DARK_RATE_E_PX_S   # 400.0 e/px/s from S03
    dark_level = dark_rate_adu_per_s * exp_time_s

    img_side = int(params.r_max * 2)
    dark_array = config["rng_dark"].poisson(
        dark_level,
        size=(img_side, img_side),
    ).astype(np.uint16)

    master_dark = make_master_dark([dark_array])

    print(f"\n[Stage B] Dark frame:")
    print(f"  Expected dark level : {dark_level:.1f} ADU (Poisson mean)")
    print(f"  Actual dark mean    : {dark_array.mean():.1f} \u00b1 {dark_array.std():.1f} ADU")
    print(f"  Master dark mean    : {master_dark.mean():.1f} ADU")

    return {
        "dark_array":  dark_array,
        "master_dark": master_dark,
        "dark_level":  dark_level,
        "exp_time_s":  exp_time_s,
    }


# ===========================================================================
# Stage C — Calibration image synthesis
# ===========================================================================

def stage_c_synthesise_cal(config: dict, dark_results: dict) -> dict:
    """
    Synthesise a neon calibration fringe image using M02.
    Add Poisson noise. Display Figure 1 (raw cal image + dark side by side).
    Return synthetic image dict and figure.
    """
    params = config["params"]

    cal_result = synthesise_calibration_image(
        params,
        add_noise=True,
        rng=config["rng_cal"],
    )
    cal_image_clean = cal_result["image_2d"]   # pure fringe (no dark)

    # Simulate the raw CCD output by adding the dark frame to the fringe image.
    # Real CCD images = fringe + dark + noise.  M02 produces fringe only, so we
    # add the dark here.  Stage D will then subtract it via reduce_calibration_frame.
    dark_array = dark_results["dark_array"]
    dark_level = dark_results["dark_level"]
    cal_image  = (cal_image_clean.astype(np.float64)
                  + dark_array.astype(np.float64))   # raw CCD image

    print(f"\n[Stage C] Calibration image (with dark added, simulating raw CCD):")
    print(f"  Shape  : {cal_image.shape}")
    print(f"  Range  : {cal_image.min():.0f} \u2013 {cal_image.max():.0f} ADU")
    print(f"  Mean   : {cal_image.mean():.1f} \u00b1 {cal_image.std():.1f} ADU")

    # Figure 1 — Raw calibration image and dark frame side by side
    fig1, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, img, title in [
        (axes[0], cal_image,
         "Synthetic calibration image (neon 640/638 nm, raw with dark)"),
        (axes[1], dark_array.astype(float),
         f"Synthetic dark frame ({dark_level:.0f} ADU expected)"),
    ]:
        vmin = float(np.percentile(img, 1))
        vmax = float(np.percentile(img, 99))
        im = ax.imshow(img, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, label="ADU")
        ax.set_title(title, fontsize=10)

    fig1.suptitle("INT02 \u2014 Stage C: Synthetic images", fontsize=12)
    fig1.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig1.savefig(OUTPUT_DIR / "int02_fig1_raw_images.png", dpi=100)
    print("  Saved outputs/int02_fig1_raw_images.png")

    return {"cal_result": cal_result, "cal_image": cal_image, "fig1": fig1}


# ===========================================================================
# Stage D — Calibration frame reduction
# ===========================================================================

def stage_d_reduce_cal(config: dict, cal_results: dict,
                        dark_results: dict) -> dict:
    """
    Subtract dark frame then run reduce_calibration_frame() on the image.
    Return FringeProfile.
    """
    params = config["params"]

    fp_cal = reduce_calibration_frame(
        cal_results["cal_image"],
        master_dark=dark_results["master_dark"],
        cx_human=params.r_max,
        cy_human=params.r_max,
        r_max_px=config["r_max_px"],
        n_bins=config["n_bins"],
    )

    print(f"\n[Stage D] Calibration frame reduction:")
    print(f"  Centre (cx, cy)    : ({fp_cal.cx:.3f}, {fp_cal.cy:.3f}) px")
    print(f"  sigma_cx, sigma_cy : ({fp_cal.sigma_cx:.4f}, {fp_cal.sigma_cy:.4f}) px")
    print(f"  Quality flags      : {fp_cal.quality_flags:#04x}")
    print(f"  Good bins          : {np.sum(~fp_cal.masked)}/{fp_cal.n_bins}")
    print(f"  Peak fits found    : {len(fp_cal.peak_fits)}")
    print(f"  dark_subtracted    : {fp_cal.dark_subtracted}")

    # Figure 2 — Calibration fringe profile
    fig2, ax = plt.subplots(figsize=(10, 4))

    good = ~fp_cal.masked & np.isfinite(fp_cal.sigma_profile)
    r    = fp_cal.r_grid[good]
    p    = fp_cal.profile[good]
    s    = fp_cal.sigma_profile[good]

    ax.errorbar(r, p, yerr=s, fmt="-", color="steelblue", ecolor="lightblue",
                alpha=0.5, label="Profile \u00b11\u03c3")
    ax.plot(r, p, color="steelblue", lw=1.5)

    # Mark peaks
    for i, pf in enumerate(fp_cal.peak_fits):
        if pf.fit_ok:
            # Interpolate profile at peak radius
            y_at_peak = float(np.interp(pf.r_fit_px, fp_cal.r_grid[good], p))
            ax.plot(pf.r_fit_px, pf.amplitude_adu + y_at_peak,
                    "v", color="red", markersize=6)
            ax.annotate(f"P{i + 1}",
                        xy=(pf.r_fit_px, pf.amplitude_adu + y_at_peak),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=7, color="red")

    ax.set_title(f"Calibration fringe profile \u2014 {len(fp_cal.peak_fits)} peaks found",
                 fontsize=10)
    ax.set_xlabel("Radius (px)")
    ax.set_ylabel("Mean intensity (ADU)")
    ax.legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig(OUTPUT_DIR / "int02_fig2_cal_profile.png", dpi=100)
    print("  Saved outputs/int02_fig2_cal_profile.png")

    return {"fp_cal": fp_cal, "fig2": fig2}


# ===========================================================================
# Stage E — Tolansky analysis
# ===========================================================================

def stage_e_tolansky(config: dict, reduction_results: dict) -> tuple:
    """
    Run TolanskyPipeline on the calibration FringeProfile.
    TolanskyPipeline.run() returns a TwoLineResult (joint two-line analysis).
    Execute V1 verification check.
    Return (tolansky_results_dict, v1_results_list).
    """
    fp_cal = reduction_results["fp_cal"]

    # TolanskyPipeline.run() → TwoLineResult.
    # With NE_INTENSITY_2=0.8, the ideal split fraction is (1+0.8)/2 = 0.9.
    # However, radial vignetting causes λ₁ peak amplitudes to vary by ~10%,
    # so some λ₁ peaks fall below a 0.9×max threshold.  We try a sequence of
    # fractions (all > NE_INTENSITY_2 = 0.8) until one produces ≥3 peaks in
    # each family.
    tol2_result = None
    tol_exception = None
    used_frac = None
    for frac in [0.9, 0.87, 0.85, 0.83, 0.81]:
        try:
            pipeline    = TolanskyPipeline(fp_cal, amplitude_split_fraction=frac)
            tol2_result = pipeline.run()   # TwoLineResult
            used_frac   = frac
            break
        except ValueError as exc:
            tol_exception = exc

    print(f"\n[Stage E] Tolansky analysis:")
    if tol2_result is not None:
        print(f"  amplitude_split_fraction used : {used_frac}")
        print(f"  Two-line joint fit:")
        print(f"    d    = {tol2_result.d_m * 1e3:.6f} \u00b1 {tol2_result.sigma_d_m * 1e3:.6f} mm")
        print(f"    f    = {tol2_result.f_px:.2f} \u00b1 {tol2_result.sigma_f_px:.2f} px")
        print(f"    \u03b1    = {tol2_result.alpha_rad_px:.5e} rad/px")
        print(f"    \u03b51   = {tol2_result.eps1:.6f} \u00b1 {tol2_result.sigma_eps1:.6f}")
        print(f"    \u03b52   = {tol2_result.eps2:.6f} \u00b1 {tol2_result.sigma_eps2:.6f}")
        print(f"    S1   = {tol2_result.S1:.4f} \u00b1 {tol2_result.sigma_S1:.4f} px\u00b2/fringe")
        print(f"    \u03c7\u00b2/dof = {tol2_result.chi2_dof:.4f}")
        # V1 — Tolansky recovers d within 1 µm of ETALON_GAP_M
        d_error_um = abs(tol2_result.d_m - ETALON_GAP_M) * 1e6
        v1_pass = d_error_um < 1.0
        d_err_str = f"{d_error_um:.3f} \u00b5m < 1.0 \u00b5m"
    else:
        print(f"  TolanskyPipeline raised: {tol_exception}")
        d_error_um = float("nan")
        v1_pass = False
        d_err_str = f"pipeline failed: {tol_exception}"
    print(f"  V1: d error = {d_err_str}  {'PASS' if v1_pass else 'FAIL'}")

    # Figure 3 — Tolansky r² vs ring order plots
    fig3, axes = plt.subplots(1, 2, figsize=(12, 5))

    if tol2_result is not None:
        # Panel 0 — primary line (λ₁, 640.2 nm)
        ax0 = axes[0]
        ax0.errorbar(tol2_result.p1, tol2_result.r1_sq, yerr=tol2_result.sr1_sq,
                     fmt="o", color="black", markersize=5, capsize=3,
                     label=f"\u03bb\u2081 = {tol2_result.lam1_nm:.4f} nm")
        ax0.plot(tol2_result.p1, tol2_result.pred1, "-", color="red", lw=1.5,
                 label="WLS fit")
        ax0.set_xlabel("Ring order p")
        ax0.set_ylabel("r\u00b2 (px\u00b2)")
        ax0.set_title(f"Tolansky: {tol2_result.lam1_nm:.4f} nm", fontsize=10)
        ax0.legend(fontsize=8)
        ax0.text(0.05, 0.05,
                 f"\u03b5\u2081 = {tol2_result.eps1:.6f}",
                 transform=ax0.transAxes, fontsize=9,
                 bbox=dict(boxstyle="round", fc="lightyellow"))

        # Panel 1 — two-line joint (both families)
        ax1 = axes[1]
        ax1.errorbar(tol2_result.p1, tol2_result.r1_sq, yerr=tol2_result.sr1_sq,
                     fmt="o", color="black", markersize=5, capsize=3,
                     label=f"\u03bb\u2081 = {tol2_result.lam1_nm:.4f} nm")
        ax1.plot(tol2_result.p1, tol2_result.pred1, "-", color="black", lw=1.5)
        ax1.errorbar(tol2_result.p2, tol2_result.r2_sq, yerr=tol2_result.sr2_sq,
                     fmt="^", color="steelblue", markersize=5, capsize=3,
                     label=f"\u03bb\u2082 = {tol2_result.lam2_nm:.4f} nm")
        ax1.plot(tol2_result.p2, tol2_result.pred2, "-", color="steelblue", lw=1.5)
        ax1.set_xlabel("Ring order p")
        ax1.set_ylabel("r\u00b2 (px\u00b2)")
        ax1.set_title("Two-line joint analysis", fontsize=10)
        ax1.legend(fontsize=8)
        ax1.text(
            0.05, 0.95,
            f"d = {tol2_result.d_m * 1e3:.5f} mm\n"
            f"f = {tol2_result.f_px:.2f} px\n"
            f"\u03b1 = {tol2_result.alpha_rad_px:.4e} rad/px\n"
            f"\u03b5\u2081 = {tol2_result.eps1:.6f}\n"
            f"\u03b5\u2082 = {tol2_result.eps2:.6f}",
            transform=ax1.transAxes, va="top", fontsize=8,
            bbox=dict(boxstyle="round", fc="lightyellow"),
        )
    else:
        axes[0].text(0.5, 0.5, "Pipeline failed", ha="center",
                     va="center", transform=axes[0].transAxes, color="red")
        axes[1].text(0.5, 0.5, str(tol_exception), ha="center",
                     va="center", transform=axes[1].transAxes, color="red",
                     fontsize=8, wrap=True)

    fig3.suptitle("INT02 \u2014 Stage E: Tolansky analysis", fontsize=11)
    fig3.tight_layout()
    fig3.savefig(OUTPUT_DIR / "int02_fig3_tolansky.png", dpi=100)
    print("  Saved outputs/int02_fig3_tolansky.png")

    return (
        {"tol2_result": tol2_result, "fig3": fig3},
        [("V1 Tolansky d recovery", v1_pass, f"{d_err_str}")],
    )


# ===========================================================================
# Stage F — Calibration inversion (M05)
# ===========================================================================

def stage_f_calibration_inversion(
    config: dict,
    reduction_results: dict,
    tolansky_results: dict,
) -> tuple:
    """
    Build FitConfig from Tolansky result. Run fit_calibration_fringe (M05).
    Execute V2, V3, V4 verification checks.
    Return (cal_inv_results_dict, v_checks_list).
    """
    fp_cal    = reduction_results["fp_cal"]
    params    = config["params"]
    # Per spec Section 2.2: use truth instrument params directly to bypass
    # the amplitude-split reliability problem on synthetic data.
    config_m05 = FitConfig(
        t_init_m     = params.t_m,
        t_bounds_m   = (params.t_m - 20e-6, params.t_m + 20e-6),
        alpha_init   = params.alpha,
        alpha_bounds = (params.alpha * 0.95, params.alpha * 1.05),
    )
    cal_result = fit_calibration_fringe(fp_cal, config_m05)

    print(f"\n[Stage F] Calibration inversion (M05):")
    print(f"  t_m recovered  : {cal_result.t_m * 1e3:.6f} \u00b1 {cal_result.sigma_t_m * 1e6:.3f} \u00b5m")
    print(f"  R_refl         : {cal_result.R_refl:.4f} \u00b1 {cal_result.sigma_R_refl:.4f}")
    print(f"  alpha          : {cal_result.alpha:.5e} \u00b1 {cal_result.sigma_alpha:.2e} rad/px")
    print(f"  sigma0         : {cal_result.sigma0:.4f} \u00b1 {cal_result.sigma_sigma0:.4f} px")
    print(f"  epsilon_cal    : {cal_result.epsilon_cal:.8f}")
    print(f"  chi2_reduced   : {cal_result.chi2_reduced:.4f}")
    print(f"  chi2_by_stage  : {[f'{x:.3f}' for x in cal_result.chi2_by_stage]}")
    print(f"  converged      : {cal_result.converged}")
    print(f"  quality_flags  : {cal_result.quality_flags:#04x}")

    # V2 — chi2_reduced in [0.5, 3.0]
    v2_pass = 0.5 < cal_result.chi2_reduced < 3.0
    print(f"  V2: chi2_red = {cal_result.chi2_reduced:.3f}  {'PASS' if v2_pass else 'FAIL'}")

    # V3 — t_m within 1 nm of truth
    t_error_nm = abs(cal_result.t_m - config["params"].t_m) * 1e9
    v3_pass = t_error_nm < 1.0
    print(f"  V3: t_m error = {t_error_nm:.4f} nm  {'PASS' if v3_pass else 'FAIL'}")

    # V4 — epsilon_cal consistent with t_m
    eps_expected = (2.0 * cal_result.t_m / NE_WAVELENGTH_1_M) % 1.0
    eps_error = abs(cal_result.epsilon_cal - eps_expected)
    v4_pass = eps_error < 1e-10
    print(f"  V4: epsilon_cal error = {eps_error:.2e}  {'PASS' if v4_pass else 'FAIL'}")

    # Figure 4 — M05 calibration fit diagnostics
    fig4 = plt.figure(figsize=(14, 8))
    gs4  = gridspec.GridSpec(2, 3, figure=fig4)

    ax_fit  = fig4.add_subplot(gs4[0, :2])
    ax_res  = fig4.add_subplot(gs4[1, :2])
    ax_text = fig4.add_subplot(gs4[:, 2])

    # Good-bin arrays
    good = ~fp_cal.masked & np.isfinite(fp_cal.sigma_profile)
    r_good   = fp_cal.r_grid[good]
    p_good   = fp_cal.profile[good]
    s_good   = fp_cal.sigma_profile[good]

    # Evaluate M05 model on fine grid then interpolate to r_good
    # (matches M05 _neon_model exactly — two-line neon + bias B)
    r_max    = fp_cal.r_max_px
    r_fine   = np.linspace(0.0, r_max, 500)
    t        = cal_result.t_m
    R        = cal_result.R_refl
    alpha    = cal_result.alpha
    I0       = cal_result.I0
    I1       = cal_result.I1
    I2       = cal_result.I2
    sigma0   = cal_result.sigma0
    sigma1   = cal_result.sigma1
    sigma2   = cal_result.sigma2
    B        = cal_result.B

    A1_fine = airy_modified(r_fine, NE_WAVELENGTH_1_M, t, R, alpha,
                             1.0, r_max, I0, I1, I2, sigma0, sigma1, sigma2)
    A2_fine = airy_modified(r_fine, NE_WAVELENGTH_2_M, t, R, alpha,
                             1.0, r_max, I0, I1, I2, sigma0, sigma1, sigma2)
    model_fine = A1_fine + NE_INTENSITY_2 * A2_fine

    # Interpolate fine model to good-bin radii
    model_at_good = np.interp(r_good, r_fine, model_fine) + B

    # Airy fit panel
    ax_fit.errorbar(r_good, p_good, yerr=s_good,
                    fmt="none", ecolor="lightgrey", alpha=0.5, zorder=1)
    ax_fit.plot(r_good, p_good, ".", color="grey", markersize=2, zorder=2,
                label="Data")
    ax_fit.plot(r_good, model_at_good, "-", color="red", lw=1.5, zorder=3,
                label="M05 model")
    ax_fit.set_xlabel("Radius (px)")
    ax_fit.set_ylabel("Mean intensity (ADU)")
    ax_fit.set_title(
        f"M05 Airy fit  (\u03c7\u00b2_red = {cal_result.chi2_reduced:.3f})",
        fontsize=10)
    ax_fit.legend(fontsize=8)

    # Residuals panel
    resid = (p_good - model_at_good) / s_good
    ax_res.fill_between(r_good, resid, step="mid", color="steelblue", alpha=0.4)
    ax_res.plot(r_good, resid, "-", color="steelblue", lw=0.8)
    ax_res.axhline(0,   color="black",  lw=0.8, ls="-")
    ax_res.axhline(+1,  color="orange", lw=0.8, ls="--")
    ax_res.axhline(-1,  color="orange", lw=0.8, ls="--")
    ax_res.axhline(+2,  color="red",    lw=0.8, ls=":")
    ax_res.axhline(-2,  color="red",    lw=0.8, ls=":")
    ax_res.set_xlabel("Radius (px)")
    ax_res.set_ylabel("(data \u2212 model) / \u03c3")
    ax_res.set_title("Normalised residuals  (should scatter in \u00b12)", fontsize=10)

    # Parameter recovery text panel
    ax_text.axis("off")
    truth_p = config["params"]
    lines = [
        "Parameter Recovery",
        "─" * 28,
        f"t_m  : {cal_result.t_m * 1e3:.6f} ± {cal_result.two_sigma_t_m * 1e6:.3f} µm",
        f"       (truth: {truth_p.t_m * 1e3:.6f} mm)",
        f"R    : {cal_result.R_refl:.4f} ± {cal_result.two_sigma_R_refl:.4f}",
        f"       (truth: {truth_p.R_refl:.4f})",
        f"α    : {cal_result.alpha:.5e} ± {cal_result.two_sigma_alpha:.2e}",
        f"       (truth: {truth_p.alpha:.5e})",
        f"σ₀   : {cal_result.sigma0:.4f} ± {cal_result.two_sigma_sigma0:.4f} px",
        f"       (truth: {truth_p.sigma0:.4f})",
        f"ε_cal: {cal_result.epsilon_cal:.8f}",
        f"χ²_r : {cal_result.chi2_reduced:.4f}",
    ]
    ax_text.text(0.05, 0.95, "\n".join(lines),
                 transform=ax_text.transAxes, va="top", fontsize=8,
                 fontfamily="monospace",
                 bbox=dict(boxstyle="round", fc="lightyellow"))

    fig4.suptitle("INT02 \u2014 Stage F: M05 calibration inversion", fontsize=11)
    fig4.tight_layout()
    fig4.savefig(OUTPUT_DIR / "int02_fig4_m05_fit.png", dpi=100)
    print("  Saved outputs/int02_fig4_m05_fit.png")

    return (
        {"cal_result": cal_result, "fig4": fig4},
        [
            ("V2 chi2_reduced in [0.5, 3.0]", v2_pass,
             f"{cal_result.chi2_reduced:.3f}"),
            ("V3 t_m within 1 nm of truth",   v3_pass,
             f"{t_error_nm:.4f} nm"),
            ("V4 epsilon_cal consistent",      v4_pass,
             f"{eps_error:.2e}"),
        ],
    )


# ===========================================================================
# Stage G — Airglow image synthesis
# ===========================================================================

def stage_g_synthesise_sci(config: dict, dark_results: dict) -> dict:
    """
    Synthesise airglow fringe images at three injected wind speeds.
    Adds the synthetic dark to each image to simulate raw CCD output;
    Stage H will subtract it via reduce_science_frame(master_dark=...).
    Return dict keyed by v_truth_ms.
    """
    params = config["params"]
    sci_results = {}

    # NB03: physically grounded I_line (replaces I_line=1.0 placeholder)
    from src.fpi.nb03_ver_source_model_2026_04_12 import compute_signal_budget
    _I_line = compute_signal_budget()['I_line']

    print(f"\n[Stage G] Airglow synthesis:")
    for v_truth in config["v_truth_ms"]:
        sci = synthesise_airglow_image(
            v_rel_ms=v_truth,
            params=params,
            snr=config["snr"],
            add_noise=True,
            rng=config["rng_sci"][v_truth],
            I_line=_I_line,
        )
        # Add dark to science image to simulate raw CCD output (Stage H subtracts it)
        dark_arr = dark_results["dark_array"].astype(np.float64)
        sci_raw  = sci.copy()
        sci_raw["image_2d"] = (sci["image_2d"].astype(np.float64) + dark_arr)
        sci_results[v_truth] = sci_raw
        print(f"  v_truth = {v_truth:+7.1f} m/s  \u2192 image shape {sci_raw['image_2d'].shape}, "
              f"mean={sci_raw['image_2d'].mean():.1f} ADU")

    # Figure 5 — Three airglow images
    fig5, axes = plt.subplots(1, 3, figsize=(14, 4))
    vmin_all = min(float(np.percentile(sci_results[v]["image_2d"], 1))
                   for v in config["v_truth_ms"])
    vmax_all = max(float(np.percentile(sci_results[v]["image_2d"], 99))
                   for v in config["v_truth_ms"])

    ims = []
    for ax, v in zip(axes, config["v_truth_ms"]):
        img = sci_results[v]["image_2d"]
        im  = ax.imshow(img, cmap="gray", origin="lower",
                        vmin=vmin_all, vmax=vmax_all)
        ax.set_title(f"v_rel = {v:+.0f} m/s", fontsize=10)
        ims.append(im)

    fig5.colorbar(ims[-1], ax=axes.ravel().tolist(), label="ADU", shrink=0.8)
    fig5.suptitle("INT02 \u2014 Stage G: Airglow images (Doppler shift visible)",
                  fontsize=11)
    fig5.tight_layout()
    fig5.savefig(OUTPUT_DIR / "int02_fig5_airglow_images.png", dpi=100)
    print("  Saved outputs/int02_fig5_airglow_images.png")

    return {"sci_results": sci_results, "fig5": fig5}


# ===========================================================================
# Stage H — Airglow inversion
# ===========================================================================

def stage_h_airglow_inversion(
    config: dict,
    dark_results: dict,
    cal_inv_results: dict,
    sci_results: dict,
    reduction_r: dict,
) -> tuple:
    """
    For each injected wind speed: reduce_science_frame + fit_airglow_fringe.
    Execute V5, V6, V7 verification checks.
    Return (inversion_results_dict, v_checks_list).
    """
    cal_result = cal_inv_results["cal_result"]
    fp_cal     = reduction_r["fp_cal"]
    cx         = fp_cal.cx
    cy         = fp_cal.cy
    sigma_cx   = fp_cal.sigma_cx
    sigma_cy   = fp_cal.sigma_cy

    fits    = {}
    fps_sci = {}

    print(f"\n[Stage H] Airglow inversion:")
    for v_truth in config["v_truth_ms"]:
        sci_img = sci_results["sci_results"][v_truth]["image_2d"]

        fp_sci = reduce_science_frame(
            sci_img,
            master_dark=dark_results["master_dark"],
            cx=cx,
            cy=cy,
            sigma_cx=sigma_cx,
            sigma_cy=sigma_cy,
            r_max_px=config["r_max_px"],
            n_bins=config["n_bins"],
        )
        fps_sci[v_truth] = fp_sci

        fit = fit_airglow_fringe(fp_sci, cal_result)
        fits[v_truth] = fit

        print(f"  v_truth = {v_truth:+7.1f} m/s:")
        print(f"    v_rec  = {fit.v_rel_ms:+7.2f} \u00b1 {fit.sigma_v_rel_ms:.2f} m/s")
        print(f"    error  = {fit.v_rel_ms - v_truth:+6.2f} m/s")
        print(f"    chi2   = {fit.chi2_reduced:.3f}")
        print(f"    flags  = {fit.quality_flags:#04x}")

    # V5 — |v_rel error| < max(20, 3σ) for each wind speed
    # V6 — chi2_reduced in [0.5, 3.0] for each wind speed
    checks = []
    for v_truth, fit in fits.items():
        error     = fit.v_rel_ms - v_truth
        tolerance = max(20.0, 3.0 * fit.sigma_v_rel_ms)
        v5_pass   = abs(error) < tolerance
        checks.append((f"V5 v_rel recovery at {v_truth:+.0f} m/s",
                        v5_pass,
                        f"|error|={abs(error):.2f} < {tolerance:.2f} m/s"))

    for v_truth, fit in fits.items():
        v6_pass = 0.5 < fit.chi2_reduced < 3.0
        checks.append((f"V6 chi2 at {v_truth:+.0f} m/s",
                        v6_pass,
                        f"chi2_red = {fit.chi2_reduced:.3f}"))

    # V7 — sigma_v_rel at v=0 ≤ 2 × WIND_BIAS_BUDGET_MS
    fit_zero = fits[0.0]
    v7_pass  = fit_zero.sigma_v_rel_ms <= 2.0 * WIND_BIAS_BUDGET_MS
    checks.append(("V7 sigma_v within 2\u00d7 STM budget",
                    v7_pass,
                    f"sigma_v = {fit_zero.sigma_v_rel_ms:.2f} m/s "
                    f"(limit: {2 * WIND_BIAS_BUDGET_MS:.1f} m/s)"))

    # Figure 6 — M06 airglow inversion diagnostics
    colours = {-300.0: "steelblue", 0.0: "green", 300.0: "darkorange"}

    fig6 = plt.figure(figsize=(15, 10))
    gs6  = gridspec.GridSpec(2, 4, figure=fig6, hspace=0.4, wspace=0.35)

    for col, v_truth in enumerate(config["v_truth_ms"]):
        fp_sci = fps_sci[v_truth]
        fit    = fits[v_truth]
        colour = colours[v_truth]

        good  = ~fp_sci.masked & np.isfinite(fp_sci.sigma_profile)
        r_g   = fp_sci.r_grid[good]
        p_g   = fp_sci.profile[good]
        s_g   = fp_sci.sigma_profile[good]

        # Evaluate M06 forward model on fine grid for the overlay
        r_max_s  = fp_sci.r_max_px
        r_fine_s = np.linspace(0.0, r_max_s, 500)
        lam_c    = fit.lambda_c_m
        Y_line   = fit.Y_line
        B_sci    = fit.B_sci

        cal = cal_inv_results["cal_result"]
        A_sci_fine = airy_modified(
            r_fine_s, lam_c, cal.t_m, cal.R_refl, cal.alpha,
            1.0, r_max_s, cal.I0, cal.I1, cal.I2,
            cal.sigma0, cal.sigma1, cal.sigma2,
        )
        model_sci_fine = Y_line * A_sci_fine + B_sci
        model_sci_at_g = np.interp(r_g, r_fine_s, model_sci_fine)

        # Top row — profile + fit
        ax_top = fig6.add_subplot(gs6[0, col])
        ax_top.errorbar(r_g, p_g, yerr=s_g,
                        fmt="none", ecolor="lightgrey", alpha=0.5, zorder=1)
        ax_top.plot(r_g, p_g, ".", color="grey", markersize=2, zorder=2)
        ax_top.plot(r_g, model_sci_at_g, "-", color=colour, lw=1.5, zorder=3)
        ax_top.set_title(f"v_truth = {v_truth:+.0f} m/s", fontsize=9)
        ax_top.set_xlabel("Radius (px)", fontsize=8)
        ax_top.set_ylabel("ADU", fontsize=8)

        # Bottom row — normalised residuals
        ax_bot = fig6.add_subplot(gs6[1, col])
        resid_sci = (p_g - model_sci_at_g) / s_g
        ax_bot.fill_between(r_g, resid_sci, step="mid", color=colour, alpha=0.4)
        ax_bot.plot(r_g, resid_sci, "-", color=colour, lw=0.8)
        ax_bot.axhline(0,  color="black",  lw=0.8)
        ax_bot.axhline(+1, color="orange", lw=0.8, ls="--")
        ax_bot.axhline(-1, color="orange", lw=0.8, ls="--")
        ax_bot.set_xlabel("Radius (px)", fontsize=8)
        ax_bot.set_ylabel("Resid (\u03c3)", fontsize=8)

    # Summary scatter — v_rec vs v_truth (fourth column, spanning both rows)
    ax_scatter = fig6.add_subplot(gs6[:, 3])
    v_truths   = config["v_truth_ms"]
    v_recs     = [fits[v].v_rel_ms for v in v_truths]
    v_sigs     = [fits[v].sigma_v_rel_ms for v in v_truths]
    ax_scatter.errorbar(v_truths, v_recs, yerr=v_sigs,
                        fmt="o", color="black", capsize=4)
    ax_scatter.plot(v_truths, v_truths, "--", color="grey", lw=1, label="y=x (perfect)")
    for v_t, v_r in zip(v_truths, v_recs):
        ax_scatter.annotate(f"{v_r:+.1f}", xy=(v_t, v_r),
                            xytext=(4, 0), textcoords="offset points", fontsize=8)
    ax_scatter.set_xlabel("v_truth (m/s)")
    ax_scatter.set_ylabel("v_recovered (m/s)")
    ax_scatter.set_title("v_rec vs v_truth", fontsize=10)
    ax_scatter.legend(fontsize=8)

    fig6.suptitle("INT02 \u2014 Stage H: M06 airglow inversion", fontsize=11)
    fig6.savefig(OUTPUT_DIR / "int02_fig6_m06_fits.png", dpi=100)
    print("  Saved outputs/int02_fig6_m06_fits.png")

    return ({"fits": fits, "fps_sci": fps_sci, "fig6": fig6}, checks)


# ===========================================================================
# Stage I — Metadata validation (P01)
# ===========================================================================

def stage_i_metadata(
    config: dict,
    sci_results: dict,
    inversion_results: dict,
) -> tuple:
    """
    Build ImageMetadata for the v=0 m/s science image using
    build_synthetic_metadata(). Verify V8.
    """
    import pandas as pd

    # Use pandas.Timestamp since astropy is not available in this venv.
    # _epoch_to_unix_ms in P01 accepts pandas Timestamp via .timestamp().
    epoch = pd.Timestamp("2027-01-01T00:00:00", tz="UTC")

    # Minimal NB01-compatible row — uses orbit_number=1 (odd → along_track)
    nb01_row = pd.Series({
        "epoch":     epoch,
        "pos_eci_x": 0.0,
        "pos_eci_y": 6_891_000.0,   # ~510 km altitude, equatorial
        "pos_eci_z": 0.0,
        "vel_eci_x": 7560.0,
        "vel_eci_y": 0.0,
        "vel_eci_z": 0.0,
        "sc_lat":    0.0,
        "sc_lon":    0.0,
        "sc_alt_km": 510.0,
    })

    fit_zero = inversion_results["fits"][0.0]
    nb02_tp  = {
        "tp_lat_deg": 0.0,
        "tp_lon_deg": 0.0,
        "tp_alt_km":  250.0,
        "tp_eci":     [0.0, 6_621_000.0, 0.0],
    }
    nb02_vr = {
        "v_rel":       fit_zero.v_rel_ms,
        "v_wind_LOS":  fit_zero.v_rel_ms,
        "V_sc_LOS":    0.0,
        "v_earth_LOS": 0.0,
        "v_zonal_ms":  0.0,
        "v_merid_ms":  0.0,
    }

    # orbit_number=1 → odd → orbit_parity='along_track' (per P01 implementation)
    meta = build_synthetic_metadata(
        params=config["params"],
        nb01_row=nb01_row,
        nb02_tp=nb02_tp,
        nb02_vr=nb02_vr,
        quaternion_xyzw=[0.0, 0.0, 0.0, 1.0],
        los_eci=np.array([0.0, -1.0, 0.0]),
        look_mode="along_track",
        img_type="science",
        orbit_number=1,
        frame_sequence=0,
        noise_seed=100,
    )

    print(f"\n[Stage I] ImageMetadata validation:")
    print(f"  is_synthetic     : {meta.is_synthetic}")
    print(f"  img_type         : {meta.img_type}")
    print(f"  obs_mode         : {meta.obs_mode}")
    print(f"  orbit_number     : {meta.orbit_number}")
    print(f"  orbit_parity     : {meta.orbit_parity}")
    print(f"  truth_v_los      : {meta.truth_v_los:.4f} m/s")
    print(f"  etalon_gap_mm    : {meta.etalon_gap_mm:.6f} mm")
    print(f"  adcs_quality_flag: {meta.adcs_quality_flag:#04x}")

    checks = []

    # V8a — truth_v_los must match injected v_wind_LOS
    v_err = abs(meta.truth_v_los - nb02_vr["v_wind_LOS"])
    checks.append(("V8a truth_v_los correct",
                    v_err < 1e-10,
                    f"|error| = {v_err:.2e}"))

    # V8b — etalon_gap_mm must match params.t_m * 1000
    gap_err = abs(meta.etalon_gap_mm - config["params"].t_m * 1000)
    checks.append(("V8b etalon_gap_mm correct",
                    gap_err < 1e-6,
                    f"|error| = {gap_err:.2e} mm"))

    # V8c — is_synthetic must be True
    checks.append(("V8c is_synthetic = True",
                    meta.is_synthetic is True,
                    str(meta.is_synthetic)))

    # V8d — adcs_quality_flag must be GOOD (0)
    checks.append(("V8d adcs_quality_flag = GOOD",
                    meta.adcs_quality_flag == 0,
                    f"flags = {meta.adcs_quality_flag:#04x}"))

    # V8e — orbit_parity = 'along_track' for orbit_number=1 (odd)
    checks.append(("V8e orbit_parity = along_track (orbit_number=1 is odd)",
                    meta.orbit_parity == "along_track",
                    str(meta.orbit_parity)))

    return ({"meta": meta}, checks)


# ===========================================================================
# Stage J — Summary report
# ===========================================================================

def stage_j_summary(all_checks: list) -> int:
    """
    Print PASS/FAIL summary for all verification checks.
    Return 0 if all pass, 1 if any fail.
    """
    n_pass = sum(1 for _, ok, _ in all_checks if ok)
    n_fail = len(all_checks) - n_pass

    print("\n" + "\u2550" * 54)
    print("  INT02 Verification Summary")
    print("\u2550" * 54)
    for name, ok, detail in all_checks:
        status = "PASS" if ok else "FAIL"
        print(f"  {name:<42s} {status}   {detail}")
    print("\u2500" * 54)
    print(f"  Total: {len(all_checks)} checks.  PASS: {n_pass}.  FAIL: {n_fail}.")
    print("\u2550" * 54)

    if n_fail > 0:
        fail_msg = "INT02 FAILED \u2014 see checks above."
        # ANSI red on Unix/Windows terminal
        try:
            print(f"\033[91m{fail_msg}\033[0m")
        except Exception:
            print(fail_msg)

    return 0 if n_fail == 0 else 1


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="INT02 FPI chain integration")
    parser.add_argument("--quick", action="store_true",
                        help="Reduce noise trials for faster run")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_checks = []

    config      = stage_a_setup(args.quick)
    dark_r      = stage_b_dark(config)
    cal_r       = stage_c_synthesise_cal(config, dark_r)
    reduction_r = stage_d_reduce_cal(config, cal_r, dark_r)

    tol_r, v1   = stage_e_tolansky(config, reduction_r)
    all_checks += v1

    calinv_r, v2_4 = stage_f_calibration_inversion(config, reduction_r, tol_r)
    all_checks    += v2_4

    sci_r = stage_g_synthesise_sci(config, dark_r)

    inv_r, v5_7 = stage_h_airglow_inversion(
                      config, dark_r, calinv_r, sci_r, reduction_r)
    all_checks += v5_7

    meta_r, v8  = stage_i_metadata(config, sci_r, inv_r)
    all_checks += v8

    return_code = stage_j_summary(all_checks)

    plt.close("all")
    sys.exit(return_code)


if __name__ == "__main__":
    main()
