"""
Smoke tests for SPEC-CAL01 v1.1 — fpi_cal_lib and run_cal_pipeline.

Tests
-----
1. fpi_cal_lib imports cleanly (no display)
2. TolanskySeedMean and OrbitCalResult importable from fpi_cal_lib
3. make_master_dark with 3 synthetic 260×276 uint16 frames
4. dark_subtract: dtype float64, result near zero
5. run_cal_pipeline importable without tkinter launch (main guard present)

Run:
    python -m pytest validation/test_cal01_smoke.py -v
"""
import importlib
import importlib.util
import pathlib
import sys
import types

import numpy as np
import pytest

# Ensure repo root is on sys.path so src.* resolves
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _synthetic_uint16_frame(seed: int, shape=(260, 276), base_adu=850) -> np.ndarray:
    """Return a synthetic uint16 frame with Gaussian noise around base_adu."""
    rng = np.random.default_rng(seed)
    raw = rng.normal(loc=base_adu, scale=20.0, size=shape)
    return np.clip(raw, 0, 65535).astype(np.uint16)


# ── Test 1: fpi_cal_lib imports cleanly ──────────────────────────────────────

def test_fpi_cal_lib_imports():
    """fpi_cal_lib must import without raising and without launching a display."""
    import src.fpi.fpi_cal_lib as cal  # noqa: F401
    assert hasattr(cal, "__all__"), "fpi_cal_lib must define __all__"
    # Spot-check a few expected symbols
    for sym in [
        "make_master_dark", "dark_subtract", "find_centre",
        "build_radial_profile", "fit_peaks", "peaks_to_array",
        "run_tolansky", "average_tolansky_seeds", "run_h05",
        "TolanskySeedMean", "OrbitCalResult", "FitResult",
    ]:
        assert sym in cal.__all__, f"{sym!r} missing from cal.__all__"
        assert hasattr(cal, sym), f"{sym!r} missing from cal namespace"


# ── Test 2: Dataclasses importable ───────────────────────────────────────────

def test_dataclasses_importable():
    """TolanskySeedMean and OrbitCalResult must be importable from fpi_cal_lib."""
    from src.fpi.fpi_cal_lib import TolanskySeedMean, OrbitCalResult
    import dataclasses

    assert dataclasses.is_dataclass(TolanskySeedMean)
    assert dataclasses.is_dataclass(OrbitCalResult)

    # Check key fields exist
    tsm_fields = {f.name for f in dataclasses.fields(TolanskySeedMean)}
    for field in ("d_m_mean", "sigma_d_m_mean", "two_sigma_d_m_mean",
                  "alpha_mean", "eps_a_mean", "Y_B_obs_mean",
                  "n_frames_total", "n_frames_used", "n_frames_rejected"):
        assert field in tsm_fields, f"TolanskySeedMean missing field {field!r}"

    ocr_fields = {f.name for f in dataclasses.fields(OrbitCalResult)}
    for field in ("seeds", "fit", "source_cal_files", "source_dark_files",
                  "date_utc", "pipeline_version"):
        assert field in ocr_fields, f"OrbitCalResult missing field {field!r}"


# ── Test 3: make_master_dark with synthetic frames ───────────────────────────

def test_make_master_dark(tmp_path):
    """make_master_dark stacks 3 uint16 .bin frames into a float64 median."""
    import src.fpi.fpi_cal_lib as cal

    shape = (260, 276)
    n_frames = 3
    paths = []
    for seed in range(n_frames):
        frame = _synthetic_uint16_frame(seed, shape, base_adu=850)
        p = tmp_path / f"dark_{seed:02d}.bin"
        frame.tofile(str(p))
        paths.append(p)

    master, sigma = cal.make_master_dark(paths, shape=shape)

    assert master.shape == shape, f"master.shape={master.shape} expected {shape}"
    assert sigma.shape == shape, f"sigma.shape={sigma.shape} expected {shape}"
    assert master.dtype == np.float64
    assert sigma.dtype == np.float64
    assert 800 <= master.mean() <= 900, (
        f"master.mean()={master.mean():.1f} outside [800, 900]"
    )
    assert sigma.max() < 200, f"sigma.max()={sigma.max():.1f} suspiciously large"


# ── Test 4: dark_subtract ─────────────────────────────────────────────────────

def test_dark_subtract():
    """dark_subtract must return float64 with mean close to zero."""
    import src.fpi.fpi_cal_lib as cal

    shape = (260, 276)
    rng = np.random.default_rng(42)

    # Identical frame and dark → residual should be ~0
    base = rng.normal(850.0, 20.0, shape)
    frame = np.clip(base, 0, 65535).astype(np.uint16)
    master_dark = base.copy()  # float64

    result = cal.dark_subtract(frame, master_dark)

    assert result.dtype == np.float64, f"dtype={result.dtype}"
    assert result.shape == shape
    assert abs(result.mean()) < 10.0, (
        f"result.mean()={result.mean():.3f} — expected near zero"
    )


# ── Test 5: run_cal_pipeline importable, guarded by __main__ ─────────────────

def test_run_cal_pipeline_importable():
    """run_cal_pipeline must import without launching tkinter dialogs."""
    # Load the module source and check that main() is guarded
    module_path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "src" / "processing" / "run_cal_pipeline.py"
    )
    assert module_path.exists(), f"Module not found: {module_path}"

    source = module_path.read_text(encoding="utf-8")
    assert 'if __name__ == "__main__"' in source or "if __name__ == '__main__'" in source, (
        "run_cal_pipeline.py must guard main() with 'if __name__ == \"__main__\"'"
    )

    # Import via importlib to avoid tkinter pop-up in CI
    spec = importlib.util.spec_from_file_location(
        "run_cal_pipeline", module_path
    )
    mod = importlib.util.module_from_spec(spec)  # sets __file__ correctly
    sys.modules["run_cal_pipeline"] = mod
    import unittest.mock as mock
    with mock.patch("matplotlib.pyplot.show"), \
         mock.patch("matplotlib.use"):           # suppress backend switch
        spec.loader.exec_module(mod)

    assert hasattr(mod, "main"), "run_cal_pipeline must define main()"
    assert hasattr(mod, "stage_banner"), "run_cal_pipeline must define stage_banner()"
    assert hasattr(mod, "select_files"), "run_cal_pipeline must define select_files()"
