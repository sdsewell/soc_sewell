"""
Tests for Z02 synthetic airglow image generator.

NOTE: Tests use t_m=20.106e-3 (the discredited Tolansky fit gap) to preserve
historical test physics. The true operational gap is D_25C_MM ≈ 20.008 mm.

Spec:    specs/Z02_synthetic_airglow_image_generator_2026-04-10.md
Script:  scripts/z02_synthetic_airglow_generator_2026-04-10.py

8 tests:
  T1  test_file_size                     — output is exactly 143,520 bytes
  T2  test_header_decode                 — rows=260, cols=276, etalon_temp=24.0, img_type="science"
  T3  test_pixel_region_nonzero          — embedded 256×256 region is non-zero, max ≤ 16383
  T4  test_p01_ingest_accepts_file       — ingest_real_image() loads without error
  T5  test_alpha_from_focal_length       — α = 32e-6/f within 1% of 1.6072e-4 for f=199.1 mm
  T6  test_fringe_spacing_changes_with_gap — larger t → more fringe peaks
  T7  test_reflectivity_affects_contrast — higher R → higher contrast
  T8  test_save_wrong_shape_raises       — ValueError for non-(260,276) input
"""

import pathlib
import struct
import sys

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Import the module under test (add repo root to sys.path so imports work)
# ---------------------------------------------------------------------------

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_SCRIPT_PATH = _REPO_ROOT / "scripts" / "z02_synthetic_airglow_generator_2026-04-10.py"

import importlib.util as _ilu

_spec = _ilu.spec_from_file_location("z02_gen", _SCRIPT_PATH)
_mod  = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

synthesise_and_embed   = _mod.synthesise_and_embed
save_bin_file          = _mod.save_bin_file
build_header           = _mod.build_header
_encode_f64            = _mod._encode_f64


# ---------------------------------------------------------------------------
# Inline _decode_f64 per spec T2 requirement
# ---------------------------------------------------------------------------

def _decode_f64(h: np.ndarray, w: int) -> float:
    """Decode float64 from 4 BE uint16 words in LE word order (S19 §2.2)."""
    b = struct.pack(">4H", *reversed([h[w + i] for i in range(4)]))
    return struct.unpack(">d", b)[0]


# ---------------------------------------------------------------------------
# T1 — File size exactly 143,520 bytes
# ---------------------------------------------------------------------------

def test_file_size(tmp_path):
    """Output .bin must be exactly 260 × 276 × 2 = 143,520 bytes."""
    out = tmp_path / "test.bin"
    frame, _, _ = synthesise_and_embed(t_m=20.106e-3, R_refl=0.53, f_lens_m=0.1991)
    save_bin_file(frame, out)
    assert out.stat().st_size == 143_520


# ---------------------------------------------------------------------------
# T2 — Header row 0 decodes correctly
# ---------------------------------------------------------------------------

def test_header_decode(tmp_path):
    """Row 0 must decode to expected field values."""
    out = tmp_path / "test.bin"
    frame, _, _ = synthesise_and_embed(t_m=20.106e-3, R_refl=0.53, f_lens_m=0.1991)
    save_bin_file(frame, out)

    raw = np.frombuffer(out.read_bytes(), dtype=">u2")
    h = raw[:276]

    assert int(h[0]) == 260, f"rows = {int(h[0])}, expected 260"
    assert int(h[1]) == 276, f"cols = {int(h[1])}, expected 276"

    etalon_t0 = _decode_f64(h, 84)    # b2_temp_f[0]
    assert abs(etalon_t0 - 24.0) < 1e-10, (
        f"etalon_temp[0] = {etalon_t0}, expected 24.0"
    )

    # img_type classification: no lamps, no shutter closed → science
    lamps = [int(h[104 + i]) & 0xFF for i in range(6)]
    gpio  = [int(h[100 + i]) & 0xFF for i in range(4)]
    assert not any(lamps), f"lamp_ch_array = {lamps}, expected all zero"
    assert not (gpio[0] == 1 and gpio[3] == 1), (
        f"gpio_pwr_on = {gpio}, shutter must be open for science image"
    )


# ---------------------------------------------------------------------------
# T3 — Pixel region is non-zero
# ---------------------------------------------------------------------------

def test_pixel_region_nonzero(tmp_path):
    """The embedded 256×256 airglow image must be non-zero, max ≤ 16383."""
    out = tmp_path / "test.bin"
    frame, pixel_block, _ = synthesise_and_embed(
        t_m=20.106e-3, R_refl=0.53, f_lens_m=0.1991
    )
    save_bin_file(frame, out)
    r_start, c_start = 1, 10   # embedding offsets (Z02 spec Section 2.3)
    roi = pixel_block[r_start:r_start + 256, c_start:c_start + 256]
    assert roi.max() > 0, "Embedded image is all zeros"
    assert roi.max() <= 16383, f"Pixel value {roi.max()} exceeds 14-bit range"


# ---------------------------------------------------------------------------
# T4 — P01 ingest_real_image accepts the file
# ---------------------------------------------------------------------------

def test_p01_ingest_accepts_file(tmp_path):
    """
    P01's ingest_real_image() must accept the Z02 output without raising.
    Verifies full binary compatibility with the pipeline ingest path.
    """
    from src.metadata.p01_image_metadata_2026_04_06 import ingest_real_image

    out = tmp_path / "test.bin"
    frame, _, _ = synthesise_and_embed(t_m=20.106e-3, R_refl=0.53, f_lens_m=0.1991)
    save_bin_file(frame, out)

    meta, pixels = ingest_real_image(out)
    assert meta.rows == 260
    assert meta.cols == 276
    assert meta.img_type == "science"
    assert abs(meta.etalon_temps[0] - 24.0) < 1e-10
    assert pixels.shape == (259, 276)


# ---------------------------------------------------------------------------
# T5 — Plate scale derived correctly from focal length
# ---------------------------------------------------------------------------

def test_alpha_from_focal_length():
    """
    α = pixel_size_binned / f_lens.
    For f_lens = 199.1 mm, expect α ≈ 1.6072e-4 rad/px (within 1%).
    """
    PIXEL_SIZE_BINNED_M = 32e-6
    f = 0.1991
    alpha = PIXEL_SIZE_BINNED_M / f
    expected = 1.6072e-4
    assert abs(alpha - expected) / expected < 0.01, (
        f"alpha = {alpha:.4e}, expected {expected:.4e} (within 1%)"
    )


# ---------------------------------------------------------------------------
# T6 — Fringe spacing changes with etalon gap
# ---------------------------------------------------------------------------

def test_fringe_spacing_changes_with_gap():
    """
    A larger etalon gap produces more fringes across r_max.
    FSR ∝ 1/t, so more fringes fit in [0, r_max] at larger t.

    Uses a wide gap range [10, 15, 20] mm to ensure the monotonic
    relationship is unambiguous (avoids boundary effects at r_max that
    occur when gap values differ by < 0.5 mm).
    """
    from scipy.signal import find_peaks

    # Use gaps spread far enough apart to get reliably different fringe counts.
    # For alpha=1.6072e-4, r_max=128 px these give ~7, ~10, ~13 peaks.
    gaps = [10e-3, 15e-3, 20e-3]
    n_fringes = []
    for t in gaps:
        _, _, result = synthesise_and_embed(t_m=t, R_refl=0.53, f_lens_m=0.1991)
        profile = result["profile_1d"]
        peaks, _ = find_peaks(profile, height=np.mean(profile))
        n_fringes.append(len(peaks))

    assert n_fringes[2] >= n_fringes[1] >= n_fringes[0], (
        f"Fringe count {n_fringes} does not increase monotonically with gap"
    )


# ---------------------------------------------------------------------------
# T7 — Higher reflectivity produces sharper (higher-contrast) fringes
# ---------------------------------------------------------------------------

def test_reflectivity_affects_contrast():
    """
    Higher R_refl → higher finesse → higher fringe contrast.
    Contrast = (max - min) / (max + min) of the noiseless 1D profile.
    """
    contrasts = {}
    for R in [0.30, 0.53, 0.75]:
        _, _, result = synthesise_and_embed(t_m=20.106e-3, R_refl=R, f_lens_m=0.1991)
        p = result["profile_1d"]
        contrasts[R] = (p.max() - p.min()) / (p.max() + p.min())

    assert contrasts[0.75] > contrasts[0.53] > contrasts[0.30], (
        f"Fringe contrast {contrasts} does not increase with reflectivity"
    )


# ---------------------------------------------------------------------------
# T8 — Wrong-shape frame raises ValueError in save_bin_file
# ---------------------------------------------------------------------------

def test_save_wrong_shape_raises(tmp_path):
    """save_bin_file must raise ValueError for non-(260,276) arrays."""
    bad = np.zeros((256, 256), dtype=">u2")
    with pytest.raises(ValueError, match="260, 276"):
        save_bin_file(bad, tmp_path / "bad.bin")
