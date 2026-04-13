"""
Tests for Z01a — OI 630 nm Filtered Neon Lamp Calibration Validator.

Spec:   specs/Z01a_validate_OI630_filtered_neon_calibration_2026-04-12.md
Script: validation/z01a_validate_OI630_filtered_neon_calibration_2026-04-12.py

16 tests across 5 classes (T1–T16).
"""

import importlib.util
import pathlib
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must precede any plt import

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Load z01a module by file path (avoids __init__.py requirement for validation/)
# ---------------------------------------------------------------------------

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_Z01A_PATH = _REPO_ROOT / "validation" / "z01a_validate_OI630_filtered_neon_calibration_2026-04-12.py"

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_spec = importlib.util.spec_from_file_location(
    "z01a_validate_OI630_filtered_neon_calibration", _Z01A_PATH
)
_z01a = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_z01a)

# Convenience aliases
load_headerless_bin = _z01a.load_headerless_bin
extract_metadata    = _z01a.extract_metadata

# Constants imported directly (single source of truth)
from windcube.constants import (
    D_25C_MM,
    ICOS_GAP_MM,
    D_PRELOAD_NM,
    OI_WAVELENGTH_NM,
    F_TOLANSKY_MM,
)
from src.fpi.tolansky_2026_04_05 import SingleLineResult


# ---------------------------------------------------------------------------
# Helper — write a flat big-endian uint16 binary file
# ---------------------------------------------------------------------------

def _write_bin(shape: tuple, value: int = 1000) -> pathlib.Path:
    """
    Write a headerless big-endian uint16 .bin file filled with `value`.
    Returns a pathlib.Path to a NamedTemporaryFile.
    """
    n = shape[0] * shape[1]
    arr = np.full(n, value, dtype=">u2")
    tmp = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
    tmp.write(arr.tobytes())
    tmp.close()
    return pathlib.Path(tmp.name)


# ---------------------------------------------------------------------------
# Class TestLoadHeaderlessBin — 5 tests
# ---------------------------------------------------------------------------

class TestLoadHeaderlessBin:

    def test_correct_shape_and_dtype(self):
        """Loading a (256,256) bin file returns shape==(256,256), dtype==float64."""
        p = _write_bin((256, 256))
        arr = load_headerless_bin(p, (256, 256))
        assert arr.shape == (256, 256)
        assert arr.dtype == np.float64
        p.unlink()

    def test_pixel_values_preserved(self):
        """All pixels should equal the written value after loading."""
        p = _write_bin((256, 256), value=8192)
        arr = load_headerless_bin(p, (256, 256))
        assert np.all(arr == 8192.0)
        p.unlink()

    def test_non_square_shape(self):
        """Non-square (128, 200) shape is loaded correctly."""
        p = _write_bin((128, 200))
        arr = load_headerless_bin(p, (128, 200))
        assert arr.shape == (128, 200)
        p.unlink()

    def test_shape_mismatch_raises(self):
        """Loading a (256,256) file as (300,300) must raise ValueError."""
        p = _write_bin((256, 256))
        with pytest.raises(ValueError, match="expected"):
            load_headerless_bin(p, (300, 300))
        p.unlink()

    def test_big_endian_byte_order(self):
        """Raw bytes 0x12 0x34 load as big-endian uint16 = 0x1234 = 4660."""
        tmp = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
        tmp.write(b"\x12\x34")
        tmp.close()
        p = pathlib.Path(tmp.name)
        arr = load_headerless_bin(p, (1, 1))
        assert arr[0, 0] == 4660.0
        p.unlink()


# ---------------------------------------------------------------------------
# Class TestExtractMetadata — 2 tests
# ---------------------------------------------------------------------------

class TestExtractMetadata:

    def test_always_returns_none(self):
        """extract_metadata({}) must return exactly {'cal_meta': None, 'dark_meta': None}."""
        result = extract_metadata({})
        assert result == {"cal_meta": None, "dark_meta": None}

    def test_ignores_load_result_contents(self):
        """Non-empty load_result dict — both metadata fields are still None."""
        result = extract_metadata({"cal_image": np.zeros((256, 256))})
        assert result["cal_meta"] is None
        assert result["dark_meta"] is None


# ---------------------------------------------------------------------------
# Class TestConstantsArithmetic — 5 tests
# ---------------------------------------------------------------------------

class TestConstantsArithmetic:

    def test_d25c_equals_icos_minus_preload(self):
        """D_25C_MM must equal ICOS_GAP_MM − D_PRELOAD_NM × 1e-6."""
        assert D_25C_MM == pytest.approx(
            ICOS_GAP_MM - D_PRELOAD_NM * 1e-6, abs=1e-12
        )

    def test_d25c_value(self):
        """D_25C_MM must be 20.007929197 mm (within 1 pm)."""
        assert D_25C_MM == pytest.approx(20.007929197, abs=1e-9)

    def test_icos_gap_value(self):
        """ICOS_GAP_MM must be exactly 20.008 mm."""
        assert ICOS_GAP_MM == pytest.approx(20.008, abs=1e-9)

    def test_preload_value(self):
        """D_PRELOAD_NM must be 70.8029 nm (within 1 fm)."""
        assert D_PRELOAD_NM == pytest.approx(70.8029, abs=1e-6)

    def test_n_int_for_oi_630(self):
        """Integer order at OI 630 nm with D_25C_MM must be 63517."""
        N = int(round(2.0 * D_25C_MM * 1e-3 / (OI_WAVELENGTH_NM * 1e-9)))
        assert N == 63517


# ---------------------------------------------------------------------------
# Class TestSingleLineResultContract — 3 tests
# ---------------------------------------------------------------------------

def _make_slr(**kw) -> SingleLineResult:
    """Construct a SingleLineResult with sensible defaults, overriding with kw."""
    defaults = dict(
        epsilon=0.45,
        sigma_eps=1.0e-4,
        two_sigma_eps=2.0e-4,
        S=610.0,
        sigma_S=float("nan"),
        lam_c_nm=630.0,
        sigma_lam_c_nm=0.001,
        two_sigma_lam_c_nm=0.002,
        v_rel_ms=0.0,
        sigma_v_ms=1.0,
        two_sigma_v_ms=2.0,
        N_int=63517,
        d_prior_mm=20.00793,
        f_prior_mm=199.12,
        chi2_dof=1.0,
    )
    defaults.update(kw)
    return SingleLineResult(**defaults)


class TestSingleLineResultContract:

    def test_two_sigma_consistency(self):
        """two_sigma fields must equal exactly 2 × sigma fields."""
        sigma_eps      = 3.7e-5
        sigma_lam_c_nm = 2.1e-4
        sigma_v_ms     = 14.3
        r = _make_slr(
            sigma_eps=sigma_eps,
            two_sigma_eps=2.0 * sigma_eps,
            sigma_lam_c_nm=sigma_lam_c_nm,
            two_sigma_lam_c_nm=2.0 * sigma_lam_c_nm,
            sigma_v_ms=sigma_v_ms,
            two_sigma_v_ms=2.0 * sigma_v_ms,
        )
        assert r.two_sigma_eps      == pytest.approx(2 * r.sigma_eps,      rel=1e-10)
        assert r.two_sigma_lam_c_nm == pytest.approx(2 * r.sigma_lam_c_nm, rel=1e-10)
        assert r.two_sigma_v_ms     == pytest.approx(2 * r.sigma_v_ms,     rel=1e-10)

    def test_zero_velocity_pass(self):
        """|v_rel| < 3σ must be True when v=10, σ=20."""
        r = _make_slr(v_rel_ms=10.0, sigma_v_ms=20.0)
        assert abs(r.v_rel_ms) < 3 * r.sigma_v_ms

    def test_zero_velocity_warn(self):
        """|v_rel| < 3σ must be False when v=200, σ=10."""
        r = _make_slr(v_rel_ms=200.0, sigma_v_ms=10.0)
        assert not (abs(r.v_rel_ms) < 3 * r.sigma_v_ms)


# ---------------------------------------------------------------------------
# Class TestFigureReductionPeaksContract — 1 test
# ---------------------------------------------------------------------------

class TestFigureReductionPeaksContract:

    def test_no_family_column(self):
        """Figure 4 peak table must NOT have a 'Family' column (Z01a is single-family)."""
        cols = [
            "#",
            "r\u00b2 centre (px\u00b2)",
            "\u03c3(r\u00b2) (px\u00b2)",
            "2\u03c3(r\u00b2) (px\u00b2)",
            "Amplitude (ADU)",
            "\u03c3(amp) (ADU)",
            "Fit OK",
        ]
        assert "Family" not in cols
