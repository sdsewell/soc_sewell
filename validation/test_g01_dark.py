"""
Validation tests for G01 dark frame synthesis.

Spec: specs/H00_dark_frame_synthesis_2026-05-16.md §4 / §9.6
Run: python -m pytest validation/test_g01_dark.py -v
"""
import sys
import pathlib

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from src.processing.GEN01_synthesize_mission_dataset_2026_05_16 import (
    _dark_rate_e_per_px_s,
    synthesise_dark_frame,
)


def test_dark_rate_minus20():
    """CCD97 dark rate at −20°C must be 1.5–2.5 e⁻/px/s."""
    rate = _dark_rate_e_per_px_s(-20.0)
    assert 1.5 < rate < 2.5, f"Dark rate at −20°C = {rate:.3f} e⁻/px/s (expected 1.5–2.5)"


def test_dark_rate_plus20():
    """CCD97 dark rate at +20°C must recover Q_do = 400 e⁻/px/s within ±12.5%."""
    rate = _dark_rate_e_per_px_s(20.0)
    assert 350 < rate < 450, f"Dark rate at +20°C = {rate:.1f} e⁻/px/s (expected 350–450)"


def test_dark_frame_minus20():
    """2×2 dark frame at −20°C, 120s: shape, dtype, mean 480–650 DN, σ 5–15 DN."""
    rng = np.random.default_rng(42)
    frame = synthesise_dark_frame(
        t_exp_s = 120.0,
        t_ccd_c = -20.0,
        shape   = (259, 276),
        rng     = rng,
        n_bin   = 4,
    )
    assert frame.dtype == np.uint16, f"dtype={frame.dtype}, expected uint16"
    assert frame.shape == (259, 276), f"shape={frame.shape}, expected (259, 276)"
    assert np.all(frame >= 0), "Pixels below 0"
    mean = float(frame.mean())
    std  = float(frame.std())
    assert 480 <= mean <= 650, f"Frame mean={mean:.1f} DN (expected 480–650)"
    assert 5   <= std  <= 15,  f"Frame σ={std:.2f} DN (expected 5–15)"


def test_dark_frame_plus20():
    """2×2 dark frame at +20°C, 120s must be saturated (mean > 16383 DN)."""
    rng = np.random.default_rng(42)
    frame = synthesise_dark_frame(
        t_exp_s = 120.0,
        t_ccd_c = 20.0,
        shape   = (259, 276),
        rng     = rng,
        n_bin   = 4,
    )
    mean = float(frame.mean())
    assert mean > 16383, f"Frame mean at +20°C = {mean:.0f} DN (expected > 16383, saturated)"


def test_dark_frame_1x1_minus20():
    """1×1 dark frame at −20°C, 120s: shape (527,552), mean 260–380 DN, σ 3–10 DN."""
    rng = np.random.default_rng(42)
    frame = synthesise_dark_frame(
        t_exp_s = 120.0,
        t_ccd_c = -20.0,
        shape   = (527, 552),
        rng     = rng,
        n_bin   = 1,
    )
    assert frame.dtype == np.uint16
    assert frame.shape == (527, 552), f"shape={frame.shape}"
    mean = float(frame.mean())
    std  = float(frame.std())
    assert 260 <= mean <= 380, f"1×1 frame mean={mean:.1f} DN (expected 260–380)"
    assert 3   <= std  <= 10,  f"1×1 frame σ={std:.2f} DN (expected 3–10)"
