#!/usr/bin/env python
"""
invert_single_frame.py — WindCube FPI single-frame diagnostic validation tool.

Processes one WindCube FPI binary science image end-to-end through H07 and
displays every intermediate quantity so the operator can verify the geometry
and velocity decomposition are correct.

Usage:
    python scripts/invert_single_frame.py <path_to_bin_file> [options]

Spec: H07_wind_vector_retrieval_2026-05-14_v03.md §12 (diagnostic output)
"""

from __future__ import annotations

import argparse
import importlib
import sys
import warnings
from pathlib import Path

# ── Project root on sys.path ───────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION NOTES
#
# This script is the primary validation tool for H07 (wind vector retrieval).
# For null-wind synthetic data (GEN01 wind map option 1, v_zonal=v_merid=0):
#
#   Expected: V_sc_LOS    ≈ +7300 m/s  (SC approaching airglow layer)
#             v_earth_LOS ≈ ±60 m/s    (Earth rotation, sign varies with geometry)
#             v_rel       ≈ -7250 m/s  (large blueshift from SC approach)
#             v_corrected ≈  0.0 m/s   (wind is zero — this is the key check)
#
#   If v_corrected is NOT near zero for null-wind synthetic data, there is
#   a geometry error: check the boresight direction, quaternion convention,
#   or the ECI↔ECEF rotation.
#
# Sign convention (NB02c authoritative):
#   v_rel = v_wind_LOS − V_sc_LOS − v_earth_LOS
#   v_corrected = v_rel + V_sc_LOS + v_earth_LOS   (addition, not subtraction)
#   Positive v_corrected = wind blowing toward tangent point (away from SC)
# ─────────────────────────────────────────────────────────────────────────────

# ── Optional matplotlib (graceful degradation) ─────────────────────────────
_HAS_MPL = False
try:
    import matplotlib as _mpl
    _HAS_MPL = True
except ImportError:
    pass

# ── Optional pipeline modules (graceful degradation) ───────────────────────
_HAS_M03 = False
_m03_module = None
try:
    _m03_module = importlib.import_module("src.processing.m03_reduce_calibration_frame")
    _HAS_M03 = True
except (ImportError, ModuleNotFoundError):
    pass

_HAS_M06 = False
_m06_module = None
try:
    _m06_module = importlib.import_module("src.processing.m06_airglow_fringe_fit")
    _HAS_M06 = True
except (ImportError, ModuleNotFoundError):
    pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WindCube FPI single-frame diagnostic validation tool.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "path_to_bin_file",
        help="Path to a *_science.bin (or *_swapped.bin) file",
    )
    parser.add_argument(
        "--h-target-km",
        type=float,
        default=250.0,
        metavar="KM",
        help="Emission layer altitude in km (default: 250.0)",
    )
    parser.add_argument(
        "--obs-mode",
        choices=["along_track", "cross_track"],
        default=None,
        metavar="MODE",
        help="Force obs_mode: 'along_track' or 'cross_track'",
    )
    parser.add_argument(
        "--v-rel",
        type=float,
        default=None,
        metavar="M_S",
        help="Override v_rel in m/s (skip M06 fringe fit)",
    )
    parser.add_argument(
        "--sigma-v",
        type=float,
        default=10.0,
        metavar="M_S",
        help="Override sigma_v in m/s (default: 10.0)",
    )
    parser.add_argument(
        "--sidecar",
        default=None,
        metavar="PATH",
        help="Path to JSON sidecar file (default: auto-detect <stem>_L0.json)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Print diagnostics only, suppress all matplotlib output",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save plots to PNG alongside the input .bin file",
    )
    return parser.parse_args()


def _adcs_flag_names(flag: int) -> list:
    """Return list of ADCS quality flag names set in the bitmask."""
    from src.metadata.p01_image_metadata_2026_04_06 import AdcsQualityFlags
    names = []
    if flag & AdcsQualityFlags.SLEW_IN_PROGRESS:
        names.append("SLEW_IN_PROGRESS")
    if flag & AdcsQualityFlags.STR_UNAVAILABLE:
        names.append("STR_UNAVAILABLE")
    if flag & AdcsQualityFlags.GNSS_UNAVAILABLE:
        names.append("GNSS_UNAVAILABLE")
    if flag & AdcsQualityFlags.ADCS_DEGRADED:
        names.append("ADCS_DEGRADED")
    if flag & AdcsQualityFlags.POINTING_UNKNOWN:
        names.append("POINTING_UNKNOWN")
    return names


def _make_plots(
    meta,
    geom,
    image,
    derived_mode: str,
    v_rel,
    v_corrected,
    path: Path,
    save_plots: bool,
) -> None:
    """Produce 5-panel diagnostic figure (2 rows × 3 cols, bottom row spans all)."""
    if not _HAS_MPL:
        print("WARNING: matplotlib not available — skipping plots.")
        return

    import numpy as np

    if save_plots:
        _mpl.use("Agg")

    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, :])

    # ── Panel 1: World map ────────────────────────────────────────────────────
    sc_lat = float(np.degrees(meta.spacecraft_latitude))
    sc_lon = float(np.degrees(meta.spacecraft_longitude))
    tp_lat = geom.tangent_lat_deg
    tp_lon = geom.tangent_lon_deg

    ax1.set_xlim(-180, 180)
    ax1.set_ylim(-90, 90)
    ax1.set_xlabel("Longitude (°)")
    ax1.set_ylabel("Latitude (°)")
    ax1.set_title(f"Ground Track  {meta.utc_timestamp[:10]}")
    ax1.set_facecolor("#e8f4f8")
    ax1.grid(True, linewidth=0.3, color="gray", alpha=0.5)

    # Attempt coastlines via basemap (skip silently if unavailable)
    try:
        from mpl_toolkits.basemap import Basemap
        m = Basemap(
            projection="cyl",
            llcrnrlat=-90, urcrnrlat=90,
            llcrnrlon=-180, urcrnrlon=180,
            ax=ax1,
        )
        m.drawcoastlines(linewidth=0.5, color="gray")
        m.fillcontinents(color="#d4c5a0", lake_color="#e8f4f8", zorder=0)
    except Exception:
        pass

    ax1.plot(sc_lon, sc_lat, "bo", markersize=8, label="Spacecraft", zorder=5)
    ax1.plot(tp_lon, tp_lat, "r*", markersize=12, label="Tangent Point", zorder=5)
    ax1.plot([sc_lon, tp_lon], [sc_lat, tp_lat], "k-", linewidth=1, zorder=4)
    ax1.legend(fontsize=7, loc="lower left")
    ax1.text(
        0.02, 0.98,
        "Spacecraft (blue) → Tangent Point (red)",
        transform=ax1.transAxes,
        va="top", fontsize=7, color="black",
    )

    # ── Panel 2: LOS velocity budget bar chart ────────────────────────────────
    v_corr_plot = v_corrected if v_corrected is not None else 0.0
    labels = ["V_sc_LOS", "v_earth_LOS", "v_corrected"]
    values = [geom.V_sc_LOS, geom.v_earth_LOS, v_corr_plot]
    colors = ["steelblue", "orange", "green"]
    ax2.barh(labels, values, color=colors)
    ax2.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_xlabel("Velocity (m/s)")
    ax2.set_title("LOS Velocity Budget")
    if v_corrected is None:
        ax2.text(
            0.5, 0.08, "v_corrected: N/A",
            transform=ax2.transAxes, ha="center", fontsize=8, color="gray",
        )

    # ── Panel 3: Direction cosines bar chart ──────────────────────────────────
    dc_labels = ["L_E", "L_N", "L_Z"]
    dc_values = [geom.L_E, geom.L_N, geom.L_Z]
    dc_colors = ["red", "blue", "grey"]
    ax3.bar(dc_labels, dc_values, color=dc_colors)
    ax3.set_ylim(-1, 1)
    ax3.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax3.set_title(f"Direction Cosines  ({derived_mode})")
    ax3.set_ylabel("Cosine value")

    # ── Panel 4: Raw FPI image (spans full bottom row) ────────────────────────
    im = ax4.imshow(image, cmap="gray", vmin=0, vmax=16383, aspect="auto")
    ax4.set_title(
        f"Raw FPI Image  exp={meta.exp_time/100:.1f}s  T={meta.ccd_temp1:.1f}°C"
    )
    ax4.set_xlabel("Column (px)")
    ax4.set_ylabel("Row (px)")
    cb = plt.colorbar(im, ax=ax4, fraction=0.02, pad=0.01)
    cb.set_label("ADU")

    fig.suptitle(
        f"WindCube FPI Single-Frame Diagnostic — {meta.utc_timestamp}",
        fontsize=12,
        fontweight="bold",
    )

    if save_plots:
        out = path.with_name(path.stem + "_h07_diagnostic.png")
        fig.savefig(str(out), dpi=150, bbox_inches="tight")
        print(f"Plots saved to: {out}")
    else:
        plt.show()


def _run(args: argparse.Namespace, path: Path) -> None:
    """Main processing pipeline — invoked from main() inside a try/except."""
    import numpy as np

    from src.metadata.p01_image_metadata_2026_04_06 import (
        ingest_real_image,
        read_sidecar,
    )
    from windcube import wind_retrieval

    # ── Stage 0 — Ingest ──────────────────────────────────────────────────────
    meta, image = ingest_real_image(path, h_target_km_obs=args.h_target_km)

    # Apply CLI obs_mode override immediately
    if args.obs_mode is not None:
        meta.obs_mode = args.obs_mode

    # Sidecar: explicit path or auto-detect <stem>_L0.json
    sidecar_path = None
    if args.sidecar:
        sidecar_path = Path(args.sidecar)
    else:
        candidate = path.with_name(path.stem + "_L0.json")
        if candidate.exists():
            sidecar_path = candidate

    if sidecar_path is not None:
        if not sidecar_path.exists():
            print(f"WARNING: Sidecar not found: {sidecar_path}")
        else:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                meta_sc = read_sidecar(sidecar_path)
            # Merge sidecar fields that override binary-header defaults
            if args.obs_mode is None and meta_sc.obs_mode not in (None, "unknown"):
                meta.obs_mode = meta_sc.obs_mode
            if meta_sc.h_target_km_obs is not None:
                meta.h_target_km_obs = meta_sc.h_target_km_obs
            # Carry over synthetic ground-truth fields
            for field in (
                "is_synthetic", "truth_v_los", "truth_v_zonal",
                "truth_v_meridional", "tangent_lat", "tangent_lon",
                "tangent_alt_km", "etalon_gap_mm", "noise_seed",
            ):
                val = getattr(meta_sc, field, None)
                if val is not None:
                    setattr(meta, field, val)

    # Derive obs_mode from attitude quaternion if still unknown
    if meta.obs_mode == "unknown":
        pos_eci = np.array(meta.pos_eci_hat, dtype=float)
        vel_eci = np.array(meta.vel_eci_hat, dtype=float)
        meta.obs_mode = wind_retrieval.derive_obs_mode(
            meta.attitude_quaternion, pos_eci, vel_eci
        )

    # ── Sanity warnings ───────────────────────────────────────────────────────
    if meta.img_type != "science":
        print(
            f"WARNING: Image type is '{meta.img_type}', not 'science'. "
            "Geometry stages will be attempted."
        )

    if meta.adcs_quality_flag != 0:
        flag_names = _adcs_flag_names(meta.adcs_quality_flag)
        print(
            f"WARNING: adcs_quality_flag = 0x{meta.adcs_quality_flag:02X}"
            f" ({', '.join(flag_names)}) — results may be unreliable."
        )

    # ── Stage 1 — Dark subtraction (optional) ─────────────────────────────────
    dark_candidates = sorted(path.parent.glob("*_dark.bin"))
    if dark_candidates and _HAS_M03:
        try:
            image = _m03_module.subtract_dark(image, dark_candidates[0])
            print(f"Dark subtracted using: {dark_candidates[0].name}")
        except Exception as exc:
            print(f"WARNING: Dark subtraction failed: {exc}")
    else:
        print("WARNING: No dark frame found — skipping dark subtraction")

    # ── Stage 2 — v_rel recovery ──────────────────────────────────────────────
    v_rel = None
    sigma_v = args.sigma_v

    if _HAS_M06:
        try:
            v_rel, sigma_v = _m06_module.fit_fringe(image, meta)
        except Exception as exc:
            print(f"WARNING: M06 fringe fit failed: {exc}")

    if v_rel is None and args.v_rel is not None:
        v_rel = args.v_rel
        sigma_v = args.sigma_v

    if v_rel is None:
        print("WARNING: M06 not available and --v-rel not supplied.")
        print("         Cannot compute v_rel from pixel data.")
        print("         Re-run with --v-rel <value> to proceed past this stage.")

    # ── Stage 3 — H07 geometry (always runs) ─────────────────────────────────
    geom = None
    derived_mode = "unknown"
    try:
        # Temporarily set img_type='science' to bypass H07's type guard when
        # running on cal/dark frames for diagnostic purposes (operator mode).
        saved_type = meta.img_type
        meta.img_type = "science"
        try:
            geom = wind_retrieval.compute_los_geometry(meta)
        finally:
            meta.img_type = saved_type

        pos_eci = np.array(meta.pos_eci_hat, dtype=float)
        vel_eci = np.array(meta.vel_eci_hat, dtype=float)
        derived_mode = wind_retrieval.derive_obs_mode(
            meta.attitude_quaternion, pos_eci, vel_eci
        )
    except Exception as exc:
        print(f"WARNING: Geometry computation failed: {exc}")

    # ── Stage 4 — H07 velocity correction ────────────────────────────────────
    v_corrected = None
    if v_rel is not None and geom is not None:
        v_corrected = wind_retrieval.correct_los_velocity(v_rel, geom)

    # ── Printed output (spec §12 exact format) ────────────────────────────────
    print("=" * 65)
    print(f"WindCube FPI — Single Frame Diagnostic")
    print(f"File: {path}")
    print("=" * 65)
    print()
    print("[ METADATA ]")
    print(f"  UTC timestamp  : {meta.utc_timestamp}")
    print(f"  Image type     : {meta.img_type}")
    print(f"  Exposure time  : {meta.exp_time / 100:.1f} s")
    print(f"  CCD temp       : {meta.ccd_temp1:.2f} °C")
    print(f"  Orbit number   : {meta.orbit_number}")
    print(f"  obs_mode       : {meta.obs_mode}")
    print(f"  h_target_km_obs: {meta.h_target_km_obs} km")
    print()
    print("[ SPACECRAFT STATE ]")
    print(f"  Position ECI   : [{meta.pos_eci_hat[0]:.1f}, "
                               f"{meta.pos_eci_hat[1]:.1f}, "
                               f"{meta.pos_eci_hat[2]:.1f}] m")
    print(f"  Velocity ECI   : [{meta.vel_eci_hat[0]:.3f}, "
                               f"{meta.vel_eci_hat[1]:.3f}, "
                               f"{meta.vel_eci_hat[2]:.3f}] m/s")
    print(f"  SC altitude    : {meta.spacecraft_altitude/1e3:.1f} km")
    print(f"  Attitude q     : [{meta.attitude_quaternion[0]:.6f}, "
                               f"{meta.attitude_quaternion[1]:.6f}, "
                               f"{meta.attitude_quaternion[2]:.6f}, "
                               f"{meta.attitude_quaternion[3]:.6f}]  [x,y,z,w]")

    if geom is not None:
        print()
        print("[ GEOMETRY (H07 Stage G) ]")
        print(f"  Boresight BRF  : [-1, 0, 0]  (-X_BRF)")
        print(f"  l_hat ECI      : [{geom.l_hat_eci[0]:.6f}, "
                                   f"{geom.l_hat_eci[1]:.6f}, "
                                   f"{geom.l_hat_eci[2]:.6f}]")
        print(f"  Tangent point  : lat={geom.tangent_lat_deg:.4f}°  "
                                 f"lon={geom.tangent_lon_deg:.4f}°  "
                                 f"alt={geom.tangent_alt_km:.2f} km")
        print(f"  Direction cos  : L_E={geom.L_E:+.6f}  "
                                 f"L_N={geom.L_N:+.6f}  "
                                 f"L_Z={geom.L_Z:+.6f}")
        print()
        print("[ OBS MODE CROSS-CHECK ]")
        print(f"  Stored  obs_mode : {meta.obs_mode}")
        match_str = "✓" if derived_mode == meta.obs_mode else "✗ MISMATCH"
        print(f"  Derived obs_mode : {derived_mode}  {match_str}")
        print()
        print("[ VELOCITY DECOMPOSITION (H07 Stage C) ]")
        print(f"  V_sc_LOS    (SC toward TP, m/s)    : {geom.V_sc_LOS:+12.3f}")
        print(f"  v_earth_LOS (Earth rot, m/s)        : {geom.v_earth_LOS:+12.3f}")

        if v_rel is not None:
            print(f"  v_rel       (FPI measured, m/s)     : {v_rel:+12.3f}")
            print(f"  sigma_v     (uncertainty, m/s)       :  {sigma_v:11.3f}")
            print(f"  v_corrected (wind LOS, m/s)          : {v_corrected:+12.3f}")
            print(f"")
            print(f"  CHECK: v_corrected = v_rel + V_sc_LOS + v_earth_LOS")
            check_val = v_rel + geom.V_sc_LOS + geom.v_earth_LOS
            print(f"       = {v_rel:.3f} + {geom.V_sc_LOS:.3f} + {geom.v_earth_LOS:.3f} = {check_val:.6f}")
            # For synthetic frames with truth available:
            if meta.is_synthetic and meta.truth_v_los is not None:
                print(f"  Truth v_wind_LOS (m/s)               : {meta.truth_v_los:+12.3f}")
                print(f"  Residual (v_corrected - truth, m/s)  : {v_corrected - meta.truth_v_los:+12.6f}")
        else:
            print(f"  v_rel       : NOT AVAILABLE (M06 not run / --v-rel not supplied)")
            print(f"  v_corrected : NOT AVAILABLE")
    else:
        print()
        print("[ GEOMETRY ] NOT AVAILABLE — see warnings above")
        print()
        print("[ VELOCITY DECOMPOSITION ]")
        print("  NOT AVAILABLE — geometry computation failed")

    print("=" * 65)

    # ── Plots ─────────────────────────────────────────────────────────────────
    if not args.no_plots and geom is not None:
        _make_plots(
            meta=meta,
            geom=geom,
            image=image,
            derived_mode=derived_mode,
            v_rel=v_rel,
            v_corrected=v_corrected,
            path=path,
            save_plots=args.save_plots,
        )


def main() -> None:
    # Ensure UTF-8 output on Windows terminals
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    args = _parse_args()
    path = Path(args.path_to_bin_file)

    # File existence check
    if not path.exists():
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        sys.exit(1)

    # File size check (must be exactly 143,520 bytes = 260×276×2)
    expected_bytes = 260 * 276 * 2
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        print(
            f"ERROR: File size mismatch: expected {expected_bytes} bytes, "
            f"got {actual_bytes} bytes: {path}",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        _run(args, path)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
