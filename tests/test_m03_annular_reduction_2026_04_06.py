"""
Tests for M03 Annular Reduction and Peak Finding.

Spec:        specs/S12_m03_annular_reduction_2026-04-06.md
Spec tests:  T1–T10
Run with:    pytest tests/test_m03_annular_reduction_2026_04_06.py -v
"""

import numpy as np
import pytest

from src.fpi.m01_airy_forward_model_2026_04_05 import InstrumentParams
from src.fpi.m02_calibration_synthesis_2026_04_05 import synthesise_calibration_image
from src.fpi.m04_airglow_synthesis_2026_04_05 import synthesise_airglow_image
from src.fpi.m03_annular_reduction_2026_04_06 import (
    FringeProfile,
    PeakFit,
    QualityFlags,
    annular_reduce,
    find_centre,
    make_master_dark,
    reduce_calibration_frame,
    reduce_science_frame,
)


# ---------------------------------------------------------------------------
# T1 — Round-trip matches M01 ground truth (< 2%)
# ---------------------------------------------------------------------------

def test_round_trip_profile():
    """
    Annular reduction of a noiseless synthetic calibration image must recover
    the ground-truth 1D profile to within 2% (mean relative error).
    """
    params = InstrumentParams()
    result = synthesise_calibration_image(params, add_noise=False)

    fp = reduce_calibration_frame(
        result["image_2d"],
        cx_human  = 127.5,
        cy_human  = 127.5,
        r_max_px  = params.r_max,
    )

    # Ground-truth: M02 profile_1d interpolated onto the annular r_grid.
    # Do NOT re-evaluate airy_modified here — the PSF broadening in
    # airy_modified depends on the grid spacing, so evaluating on a coarser
    # 150-point r²-grid gives different peak heights than the 500-point
    # r-uniform grid used by M02 to generate the 2D image.
    ref_profile = np.interp(fp.r_grid, result["r_grid"], result["profile_1d"])

    good = ~fp.masked
    assert np.any(good), "All bins masked — reduction failed completely"

    rel_err = np.mean(
        np.abs(fp.profile[good] - ref_profile[good]) / np.abs(ref_profile[good])
    )
    assert rel_err < 0.02, \
        f"Round-trip error {rel_err:.4f} exceeds 2% threshold"


# ---------------------------------------------------------------------------
# T2 — Centre recovery < 0.05 px on known-offset synthetic image
# ---------------------------------------------------------------------------

def test_centre_recovery():
    """
    find_centre must recover a known fringe centre to within 0.05 px.
    The image is synthesised with the centre offset 3 px in x and y from the
    geometric centre; the seed is the geometric centre (not the truth).
    sigma_cx and sigma_cy must be < 0.1 px.
    """
    params  = InstrumentParams()
    cx_true = 127.5 + 3.0
    cy_true = 127.5 + 3.0

    result = synthesise_calibration_image(
        params, add_noise=False, cx=cx_true, cy=cy_true
    )

    # Seed from the geometric centre (not the truth)
    cr = find_centre(
        result["image_2d"],
        cx_seed      = 127.5,
        cy_seed      = 127.5,
        var_r_max_px = params.r_max,
        var_search_px = 15.0,
    )

    assert abs(cr.cx - cx_true) < 0.05, \
        f"cx error {abs(cr.cx - cx_true):.4f} px; expected < 0.05 px"
    assert abs(cr.cy - cy_true) < 0.05, \
        f"cy error {abs(cr.cy - cy_true):.4f} px; expected < 0.05 px"
    assert cr.sigma_cx < 0.1, \
        f"sigma_cx = {cr.sigma_cx:.4f} px; expected < 0.1 px"
    assert cr.sigma_cy < 0.1, \
        f"sigma_cy = {cr.sigma_cy:.4f} px; expected < 0.1 px"


# ---------------------------------------------------------------------------
# T3 — Wrong centre (0.5 px offset) gives lower fringe contrast
# ---------------------------------------------------------------------------

def test_wrong_centre_lower_contrast():
    """
    Using the wrong centre (0.5 px offset) must yield lower fringe contrast
    than the true centre.  Contrast = (max - min) / (max + min) of profile.
    """
    params = InstrumentParams()
    result = synthesise_calibration_image(params, add_noise=False)

    cx_true = 127.5
    cy_true = 127.5

    def contrast(profile, masked):
        good = ~masked
        if not np.any(good):
            return 0.0
        mn  = float(np.min(profile[good]))
        mx  = float(np.max(profile[good]))
        denom = mx + mn
        return (mx - mn) / denom if denom > 0 else 0.0

    fp_true = annular_reduce(
        result["image_2d"],
        cx=cx_true, cy=cy_true, sigma_cx=0.05, sigma_cy=0.05,
        r_max_px=params.r_max,
    )
    fp_wrong = annular_reduce(
        result["image_2d"],
        cx=cx_true + 0.5, cy=cy_true, sigma_cx=0.05, sigma_cy=0.05,
        r_max_px=params.r_max,
    )

    c_true  = contrast(fp_true.profile,  fp_true.masked)
    c_wrong = contrast(fp_wrong.profile, fp_wrong.masked)

    assert c_true > c_wrong, \
        f"True-centre contrast ({c_true:.4f}) not greater than " \
        f"wrong-centre contrast ({c_wrong:.4f})"


# ---------------------------------------------------------------------------
# T4 — SEM ratio n_subpixels=8 vs 1 in [0.7, 1.4]
# ---------------------------------------------------------------------------

def test_sem_subpixel_ratio():
    """
    The ratio of mean SEM (n_subpixels=8) / mean SEM (n_subpixels=1) must
    lie in [0.7, 1.4].  SEM denominator is N_pixels, not N_subpixels.
    """
    params = InstrumentParams()
    rng    = np.random.default_rng(0)
    result = synthesise_calibration_image(params, add_noise=True, rng=rng)

    cx, cy = 127.5, 127.5

    fp1 = annular_reduce(result["image_2d"], cx=cx, cy=cy,
                         sigma_cx=0.05, sigma_cy=0.05,
                         r_max_px=params.r_max, n_subpixels=1)
    fp8 = annular_reduce(result["image_2d"], cx=cx, cy=cy,
                         sigma_cx=0.05, sigma_cy=0.05,
                         r_max_px=params.r_max, n_subpixels=8)

    # Use bins that are good in both
    good = ~fp1.masked & ~fp8.masked
    assert np.any(good), "No common good bins between n_subpixels=1 and 8"

    sem1 = float(np.mean(fp1.sigma_profile[good]))
    sem8 = float(np.mean(fp8.sigma_profile[good]))

    assert sem1 > 0, "SEM with n_subpixels=1 is zero (no noise?)"
    ratio = sem8 / sem1

    assert 0.7 <= ratio <= 1.4, \
        f"SEM ratio (n_sub=8)/(n_sub=1) = {ratio:.4f}; expected in [0.7, 1.4]"


# ---------------------------------------------------------------------------
# T5 — Hot-pixel clip: centre unaffected, profile unaffected
# ---------------------------------------------------------------------------

def test_hot_pixel_clip():
    """
    Injecting a saturated pixel must not shift the centre > 0.1 px
    and must not change the mean profile by more than 0.5%.
    Verifies that the 99.5th-percentile clip is applied to cost only,
    not to the annular reduction.
    """
    params    = InstrumentParams()
    result    = synthesise_calibration_image(params, add_noise=False)
    img_clean = result["image_2d"].copy()
    img_hot   = img_clean.copy()
    img_hot[64, 64] = 65535

    fp_c = reduce_calibration_frame(img_clean, cx_human=127.5, cy_human=127.5,
                                    r_max_px=params.r_max)
    fp_h = reduce_calibration_frame(img_hot,   cx_human=127.5, cy_human=127.5,
                                    r_max_px=params.r_max)

    assert abs(fp_h.cx - 127.5) < 0.1, \
        f"Hot pixel shifted centre by {abs(fp_h.cx - 127.5):.4f} px"

    mask      = ~fp_c.masked & ~fp_h.masked
    diff_frac = np.mean(np.abs(fp_h.profile[mask] - fp_c.profile[mask])) \
                / np.mean(fp_c.profile[mask])
    assert diff_frac < 0.005, \
        f"Hot pixel contaminated profile: diff_frac = {diff_frac:.4f}"


# ---------------------------------------------------------------------------
# T6 — Peak finding: >= 6 peaks found and fitted on synthetic cal image
# ---------------------------------------------------------------------------

def test_peak_finding():
    """
    Synthetic calibration image must yield >= 6 peaks with fit_ok=True.
    All-fit requirement is relaxed by 1 to allow edge peak failure.
    """
    params = InstrumentParams()
    result = synthesise_calibration_image(params, add_noise=False)
    fp     = reduce_calibration_frame(result["image_2d"],
                                      cx_human=127.5, cy_human=127.5,
                                      r_max_px=params.r_max)

    assert len(fp.peak_fits) >= 6, \
        f"Only {len(fp.peak_fits)} peaks found; expected >= 6"

    n_ok = sum(1 for p in fp.peak_fits if p.fit_ok)
    assert n_ok >= len(fp.peak_fits) - 1, \
        f"Only {n_ok}/{len(fp.peak_fits)} Gaussian fits succeeded"


# ---------------------------------------------------------------------------
# T7 — Science frame: empty peaks, seed_source='provided'
# ---------------------------------------------------------------------------

def test_science_frame_no_peaks():
    """
    reduce_science_frame must set peak_fits=[] and seed_source='provided'.
    """
    params = InstrumentParams()
    sci    = synthesise_airglow_image(100.0, params, add_noise=True,
                                      rng=np.random.default_rng(42))

    fp = reduce_science_frame(sci["image_2d"], cx=127.5, cy=127.5,
                               sigma_cx=0.05, sigma_cy=0.05,
                               r_max_px=params.r_max)

    assert fp.seed_source == "provided", \
        f"seed_source = {fp.seed_source!r}; expected 'provided'"
    assert fp.peak_fits == [], \
        f"peak_fits not empty: {fp.peak_fits}"


# ---------------------------------------------------------------------------
# T8 — Masked bins have sigma = inf and two_sigma = inf (S04)
# ---------------------------------------------------------------------------

def test_masked_bins_inf_sigma():
    """
    Bins with fewer than min_pixels_per_bin pixels must have
    sigma_profile = np.inf and two_sigma_profile = np.inf (S04 convention).
    """
    params = InstrumentParams()
    result = synthesise_calibration_image(params, add_noise=False)

    # Use a bad_pixel_mask that forces some bins to be under-populated
    # by masking a wide annular band
    img   = result["image_2d"]
    bpm   = np.zeros(img.shape, dtype=bool)
    cx, cy = 127.5, 127.5
    cols_g, rows_g = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
    r2    = (cols_g - cx) ** 2 + (rows_g - cy) ** 2
    # Mask pixels in an annular band → those r² bins will be empty → masked
    bpm[(r2 >= 60.0 ** 2) & (r2 < 65.0 ** 2)] = True

    fp = annular_reduce(
        img, cx=cx, cy=cy, sigma_cx=0.05, sigma_cy=0.05,
        r_max_px=params.r_max,
        bad_pixel_mask=bpm,
        min_pixels_per_bin=3,
    )

    masked_bins = np.where(fp.masked)[0]
    assert len(masked_bins) > 0, \
        "No masked bins created — bad_pixel_mask had no effect"

    assert np.all(np.isinf(fp.sigma_profile[fp.masked])), \
        "sigma_profile not inf for all masked bins"
    assert np.all(np.isinf(fp.two_sigma_profile[fp.masked])), \
        "two_sigma_profile not inf for all masked bins"

# ---------------------------------------------------------------------------
# T9 — Dark subtraction removes known dark signal from profile
# ---------------------------------------------------------------------------

def test_dark_subtraction_removes_signal():
    """
    Construct a synthetic calibration image. Inject a uniform dark frame
    with a known constant value. Verify that the post-subtraction profile
    mean is reduced by approximately that constant.
    """
    params = InstrumentParams()
    result = synthesise_calibration_image(params, add_noise=False)
    img = result["image_2d"].astype(np.float64)
    dark_level = 200.0
    dark_frame = np.full_like(img, dark_level)
    master_dark = make_master_dark([dark_frame])

    fp_no_dark = reduce_calibration_frame(img, cx_human=127.5, cy_human=127.5,
                                           r_max_px=params.r_max,
                                           master_dark=None)
    fp_dark    = reduce_calibration_frame(img, cx_human=127.5, cy_human=127.5,
                                           r_max_px=params.r_max,
                                           master_dark=master_dark)

    mask = ~fp_no_dark.masked & ~fp_dark.masked
    mean_diff = np.mean(fp_no_dark.profile[mask] - fp_dark.profile[mask])
    assert abs(mean_diff - dark_level) < 1.0, \
        f"Dark subtraction removed {mean_diff:.1f} ADU; expected {dark_level:.1f}"
    assert fp_dark.dark_subtracted is True
    assert fp_dark.dark_n_frames == 1
    assert fp_no_dark.dark_subtracted is False
    assert fp_no_dark.dark_n_frames == 0


# ---------------------------------------------------------------------------
# T10 — Master dark: median of multiple frames is robust to a cosmic ray
# ---------------------------------------------------------------------------

def test_master_dark_median_cosmic_ray():
    """
    A single cosmic-ray hit in one of 3 dark frames must not contaminate
    the master dark at that pixel location.
    """
    base = np.full((256, 256), 150.0)
    frames = [base.copy() for _ in range(3)]
    frames[1][100, 100] = 60000.0   # cosmic ray in frame 2 only
    master = make_master_dark(frames)
    assert abs(master[100, 100] - 150.0) < 1.0, \
        f"Cosmic ray leaked into master dark: {master[100, 100]:.1f}"
    assert abs(np.mean(master) - 150.0) < 0.1
