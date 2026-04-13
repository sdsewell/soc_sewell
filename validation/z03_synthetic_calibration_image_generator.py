"""Z03 — Synthetic Calibration Image Generator (validation/)

Moved version: default outputs go to sibling 'soc_synthesized_data'
under the user's GitHub folder (e.g. C:/Users/sewell/Documents/GitHub/soc_synthesized_data).
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys
from datetime import datetime, timezone
from typing import Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from src.fpi.m01_airy_forward_model_2026_04_05 import NE_WAVELENGTH_1_M as LAM_640_M
    from src.fpi.m01_airy_forward_model_2026_04_05 import NE_WAVELENGTH_2_M as LAM_638_M
except Exception:
    LAM_640_M = 640.2248e-9
    LAM_638_M = 638.2991e-9

# Fixed defaults from spec
R_DEFAULT = 0.82
SIG_READ_DEFAULT = 50.0
B_DC_DEFAULT = 500.0
PIX_M = 32.0e-6
NROWS = 260
NCOLS = 276
N_META_ROWS = 4
CX_DEFAULT = 137.5
CY_DEFAULT = 129.5


def _validated_prompt(label: str, default: float, units: str,
                      hard_min: float, hard_max: float,
                      warn_min: float, warn_max: float) -> float:
    while True:
        try:
            resp = input(f"{label} [{default} {units}]: ").strip()
            if resp == "":
                val = float(default)
            else:
                val = float(resp)
        except ValueError:
            print("  Invalid number — please try again.")
            continue

        if val < hard_min or val > hard_max:
            print(f"  Value out of hard bounds [{hard_min}, {hard_max}] — try again.")
            continue

        if val < warn_min or val > warn_max:
            yn = input(f"  Warning: {val} outside recommended range [{warn_min},{warn_max}]. Continue? (Y/n) ").strip().lower()
            if yn not in ("", "y", "yes"):
                continue

        return val


def snr_to_ipeak(snr: float, B_dc: float, sigma_read: float) -> float:
    noise_floor = B_dc + sigma_read ** 2
    return (snr ** 2 + math.sqrt(snr ** 4 + 4.0 * snr ** 2 * noise_floor)) / 2.0


def airy(theta: np.ndarray, lam_m: float, d_mm: float, R: float) -> np.ndarray:
    d_m = d_mm * 1e-3
    delta = (4.0 * np.pi / lam_m) * d_m * np.cos(theta)
    F = 4.0 * R / (1.0 - R) ** 2
    return 1.0 / (1.0 + F * np.sin(delta / 2.0) ** 2)


def build_metadata_dict(image_type: str, exposure_ms: int = 120000) -> dict:
    now = datetime.now(timezone.utc)
    return {
        "image_type": image_type,
        "n_rows": NROWS,
        "n_cols": NCOLS,
        "binning": 2,
        "shutter_status": "Open" if image_type.lower().startswith("cal") else "Closed",
        "date_utc": now.strftime("%Y%m%d"),
        "time_utc": now.strftime("%H%M%S"),
        "exposure_ms": exposure_ms,
        "etalon_temp_1": 24.0,
    }


def embed_metadata_rows(image: np.ndarray, meta: dict) -> None:
    meta_json = json.dumps(meta, separators=(",", ":"))
    meta_bytes = meta_json.encode("utf-8")
    target = N_META_ROWS * NCOLS * 2
    padded = meta_bytes.ljust(target, b"\x00")
    arr = np.frombuffer(padded, dtype="<u2").reshape(N_META_ROWS, NCOLS)
    image[0:N_META_ROWS, :] = arr


def _save_diagnostic_figure(cal_img: np.ndarray, dark_img: np.ndarray, out_dir: pathlib.Path, stem: str) -> pathlib.Path:
    """Save a 2x2 diagnostic: images on top row, histograms below."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax00, ax01 = axes[0]
    ax10, ax11 = axes[1]

    im0 = ax00.imshow(cal_img, cmap="gray", vmin=0, vmax=16383)
    ax00.set_title("Calibration (noisy)")
    fig.colorbar(im0, ax=ax00, fraction=0.046, pad=0.04)

    im1 = ax01.imshow(dark_img, cmap="gray", vmin=0, vmax=16383)
    ax01.set_title("Dark (noisy)")
    fig.colorbar(im1, ax=ax01, fraction=0.046, pad=0.04)

    ax10.hist(cal_img.ravel(), bins=256, range=(0, 16383), color="C0")
    ax10.set_title("Calibration histogram")
    ax10.set_xlabel("ADU")
    ax10.set_ylabel("Count")

    ax11.hist(dark_img.ravel(), bins=256, range=(0, 16383), color="C1")
    ax11.set_title("Dark histogram")
    ax11.set_xlabel("ADU")
    ax11.set_ylabel("Count")

    fig.suptitle(f"Z03 diagnostic — {stem}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = out_dir / f"{stem}_diagnostic.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def synthesize(d_mm: float, f_mm: float, snr_peak: float, rel_638: float,
               out_dir: pathlib.Path) -> dict:
    # Derived params
    alpha = PIX_M / (f_mm * 1e-3)
    I_peak = snr_to_ipeak(snr_peak, B_DC_DEFAULT, SIG_READ_DEFAULT)

    ys = np.arange(NROWS)
    xs = np.arange(NCOLS)
    X, Y = np.meshgrid(xs, ys)
    r = np.sqrt((X - CX_DEFAULT) ** 2 + (Y - CY_DEFAULT) ** 2)
    theta = np.arctan(alpha * r)

    T1 = airy(theta, LAM_640_M, d_mm, R_DEFAULT)
    T2 = airy(theta, LAM_638_M, d_mm, R_DEFAULT)

    # Build signal component and add DC background.
    signal_comp = I_peak * (T1 + rel_638 * T2)

    # Automatic re-scaling to produce a calibration-image peak near target_peak.
    # Target defaults chosen to match observed real images (peak ~7000 ADU).
    target_peak = 7000.0
    # compute current peak (signal + background)
    current_peak = float(signal_comp.max() + B_DC_DEFAULT)
    # scale only the signal component so background remains B_DC_DEFAULT
    if current_peak > 0:
        scale = (target_peak - B_DC_DEFAULT) / float(signal_comp.max())
        # avoid extreme scaling
        scale = float(max(0.1, min(scale, 10.0)))
    else:
        scale = 1.0

    signal_comp = signal_comp * scale
    I_float = signal_comp + B_DC_DEFAULT

    ss = np.random.SeedSequence()
    rng = np.random.default_rng(ss)
    seed_entropy = int(ss.entropy)

    signal = rng.poisson(np.clip(I_float, 0, None)).astype(np.float64)
    read_noise = rng.standard_normal(size=signal.shape) * SIG_READ_DEFAULT
    cal_img = np.clip(signal + read_noise, 0, 16383).astype("<u2")

    dark_float = np.full((NROWS, NCOLS), B_DC_DEFAULT, dtype=np.float64)
    dark_counts = rng.poisson(dark_float).astype(np.float64)
    dark_read = rng.standard_normal(size=dark_float.shape) * SIG_READ_DEFAULT
    dark_img = np.clip(dark_counts + dark_read, 0, 16383).astype("<u2")

    cal_meta = build_metadata_dict("Cal")
    dark_meta = build_metadata_dict("Dark")
    embed_metadata_rows(cal_img.view(np.uint16).reshape(NROWS, NCOLS), cal_meta)
    embed_metadata_rows(dark_img.view(np.uint16).reshape(NROWS, NCOLS), dark_meta)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = f"{ts}_synth_z03"
    cal_name = f"{ts}_cal_synth_z03.bin"
    dark_name = f"{ts}_dark_synth_z03.bin"
    truth = {
        "z03_version": "1.0",
        "timestamp_utc": ts,
        "random_seed": seed_entropy,
        "user_params": {
            "d_mm": d_mm,
            "f_mm": f_mm,
            "snr_peak": snr_peak,
            "rel_638": rel_638,
        },
        "derived_params": {
            "alpha_rad_per_px": alpha,
            "I_peak_adu": float(I_peak),
            "intensity_scale": float(scale),
            "target_peak_adu": float(target_peak),
            "finesse_F": float(4 * R_DEFAULT / (1 - R_DEFAULT) ** 2),
        },
        "fixed_defaults": {
            "R": R_DEFAULT,
            "sigma_read": SIG_READ_DEFAULT,
            "B_dc": B_DC_DEFAULT,
            "cx": CX_DEFAULT,
            "cy": CY_DEFAULT,
            "pix_m": PIX_M,
            "nrows": NROWS,
            "ncols": NCOLS,
            "lam_640_m": float(LAM_640_M),
            "lam_638_m": float(LAM_638_M),
        },
        "output_cal_file": cal_name,
        "output_dark_file": dark_name,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    cal_path = out_dir / cal_name
    dark_path = out_dir / dark_name
    cal_img.tofile(str(cal_path))
    dark_img.tofile(str(dark_path))

    truth_path = out_dir / f"{stem}_truth.json"
    with open(truth_path, "w", encoding="utf-8") as fh:
        json.dump(truth, fh, indent=2)

    np.save(out_dir / f"{stem}_cal_preview.npy", I_float.astype(np.float32))

    # Save diagnostic figure
    try:
        diag_path = _save_diagnostic_figure(cal_img, dark_img, out_dir, stem)
    except Exception:
        diag_path = None

    return {
        "cal_path": cal_path,
        "dark_path": dark_path,
        "truth_path": truth_path,
        "diagnostic_path": diag_path,
        "truth": truth,
    }


def main(argv=None):
    print("Z03 Synthetic Calibration Image Generator — validation version")
    d_mm = _validated_prompt("Etalon gap d [mm]", 20.0079, "mm", 10.0, 30.0, 18.0, 22.0)
    f_mm = _validated_prompt("Imaging lens focal length f [mm]", 230.0, "mm", 50.0, 500.0, 150.0, 300.0)
    snr = _validated_prompt("Peak SNR", 50.0, "", 0.1, 1e6, 2.0, 2000.0)
    rel_638 = _validated_prompt("Relative intensity of 638 nm line", 0.3, "", 0.0, 10.0, 0.01, 5.0)

    # Default output dir: sibling GitHub/soc_synthesized_data (outside repo)
    default_out = pathlib.Path(os.environ.get(
        "Z03_OUTPUT_DIR",
        str(pathlib.Path(__file__).resolve().parents[2] / "soc_synthesized_data")
    ))
    out_dir = pathlib.Path(os.environ.get("Z03_OUTPUT_DIR", default_out))

    print(f"\nParameters: d_mm={d_mm}, f_mm={f_mm}, snr={snr}, rel_638={rel_638}")
    ok = input("Proceed with synthesis? (Y/n) ").strip().lower()
    if ok not in ("", "y", "yes"):
        print("Aborted by user.")
        sys.exit(0)

    res = synthesize(d_mm, f_mm, snr, rel_638, out_dir)
    print("Wrote:")
    print("  ", res["cal_path"]) 
    print("  ", res["dark_path"]) 
    print("  ", res["truth_path"]) 
    if res.get("diagnostic_path"):
        print("  ", res["diagnostic_path"]) 
    print("Done.")


if __name__ == "__main__":
    main()
