#!/usr/bin/env python
"""
invert_single_frame.py — WindCube FPI single-frame diagnostic validation tool.

Processes one WindCube FPI binary science image end-to-end through H07 and
displays every intermediate quantity so the operator can verify the geometry
and velocity decomposition are correct.

Usage:
    python scripts/invert_single_frame.py <path_to_bin_file> [options]

Spec: H07_wind_vector_retrieval_2026-05-14_v03.md §12 (diagnostic output)

H06 mode (default when --v-rel not supplied):
  v_rel is recovered from raw pixels via H06 fringe fitting.
  Requires *_cal*.bin files in the same folder.
  Builds a master calibration from all available cal frames,
  then runs process_science_frame() for the science frame.
  Falls back to --v-rel if no cal frames found or H06 fails.
"""

from __future__ import annotations

import argparse
import importlib
import re
import sys
import warnings
from datetime import datetime, timezone
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


def _find_nearest_dark_frame(lua_timestamp_ms: int, dark_paths: list):
    """
    Find the dark frame with the closest timestamp to lua_timestamp_ms.

    Parameters
    ----------
    lua_timestamp_ms : int
        Science frame timestamp in Unix milliseconds.
    dark_paths : list[Path]
        All *_dark.bin paths in the same folder (sorted order acceptable).

    Returns
    -------
    (Path, float) — (nearest_dark_path, dt_seconds), or None if no match.
    """
    best_path = None
    best_dt_ms = float("inf")
    for p in dark_paths:
        m = re.match(
            r"(\d{4})-(\d{2})-(\d{2})T(\d{2})-(\d{2})-(\d{2})Z", p.name
        )
        if not m:
            continue
        dt_obj = datetime(
            int(m.group(1)), int(m.group(2)), int(m.group(3)),
            int(m.group(4)), int(m.group(5)), int(m.group(6)),
            tzinfo=timezone.utc,
        )
        dark_ms = int(dt_obj.timestamp() * 1000)
        dt_ms = abs(lua_timestamp_ms - dark_ms)
        if dt_ms < best_dt_ms:
            best_dt_ms = dt_ms
            best_path = p
    if best_path is None:
        return None
    return best_path, best_dt_ms / 1000.0


def _find_cal_frames(science_path: Path) -> list[Path]:
    """
    Find *_cal*.bin files in the same folder as the science frame.
    Returns sorted list of Path objects. Empty list if none found.
    """
    return sorted(science_path.parent.glob("*_cal*.bin"))


def _build_master_dark_from_folder(science_path: Path, lua_timestamp_ms: int):
    """
    Find all *_dark.bin files in the same folder, load them, and return
    a float32 master dark array (sum of all dark frames) plus the count.

    Returns (master_dark_array, n_dark_frames) or (None, 0) if no darks found.
    Uses ingest_real_image() to load pixels.
    """
    import numpy as _np
    from src.metadata.p01_image_metadata_2026_04_06 import ingest_real_image as _ingest

    dark_paths = sorted(science_path.parent.glob("*_dark.bin"))
    if not dark_paths:
        return None, 0

    master_dark_sum = None
    n_dark = 0
    for dark_path in dark_paths:
        try:
            _, dark_pixels = _ingest(dark_path)
            if master_dark_sum is None:
                master_dark_sum = dark_pixels.astype(_np.float32)
            else:
                master_dark_sum += dark_pixels.astype(_np.float32)
            n_dark += 1
        except Exception as exc:
            print(f"  WARNING: Dark frame failed: {dark_path.name}: {exc}")

    if n_dark == 0:
        return None, 0
    return master_dark_sum, n_dark


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WindCube FPI single-frame diagnostic validation tool.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "path_to_bin_file",
        nargs="?",
        default=None,
        help="Path to *_science.bin file. If omitted, a file picker dialog opens.",
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
        "--use-h06",
        action="store_true",
        default=False,
        help=(
            "Use H06 fringe fitting to recover v_rel from raw pixels. "
            "Requires cal frames in the same folder as the science frame. "
            "Activated automatically when --v-rel is not supplied."
        ),
    )
    parser.add_argument(
        "--n-cal",
        type=int,
        default=5,
        metavar="N",
        help=(
            "Maximum number of cal frames to use when building the master "
            "calibration in H06 mode. Default: 5 (one orbit cadence). "
            "Use 0 for no limit (processes all cal frames in the folder)."
        ),
    )
    parser.add_argument(
        "--sidecar",
        default=None,
        metavar="PATH",
        help=(
            "Path to JSON sidecar file (real data only). "
            "GEN01 synthetic data does not produce JSON sidecars — "
            "use --obs-mode to supply obs_mode directly for synthetic frames."
        ),
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
    v_corrected,
    path: Path,
    save_plots: bool,
    airglow_result=None,
) -> None:
    """Produce diagnostic figure.

    Layout: 3×3 grid.
      Row 0: world map | LOS velocity budget | direction cosines
      Row 1-2 col 0-1: raw FPI image (spans 2 rows × 2 cols)
      Row 1 col 2: H06 fringe profile + model overlay (Panel A)
      Row 2 col 2: H06 scan chi2 curve (Panel B)
    Panels A and B are shown only when airglow_result is not None.
    """
    if not _HAS_MPL:
        print("WARNING: matplotlib not available — skipping plots.")
        return

    import numpy as np

    if save_plots:
        _mpl.use("Agg")

    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0])       # world map
    ax2 = fig.add_subplot(gs[0, 1])       # LOS velocity budget
    ax3 = fig.add_subplot(gs[0, 2])       # direction cosines
    ax4 = fig.add_subplot(gs[1:, :2])     # raw image (spans rows 1-2, cols 0-1)
    axA = fig.add_subplot(gs[1, 2])       # Panel A: fringe profile + model
    axB = fig.add_subplot(gs[2, 2])       # Panel B: scan chi2 curve

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

    # ── Panel 4: Raw FPI image (spans rows 1-2, cols 0-1) ────────────────────
    im = ax4.imshow(image, cmap="gray", vmin=0, vmax=16383, aspect="equal")
    ax4.set_title(
        f"Raw FPI Image  exp={meta.exp_time/100:.1f}s  T={meta.ccd_temp1:.1f}°C"
    )
    ax4.set_xlabel("Column (px)")
    ax4.set_ylabel("Row (px)")
    cb = plt.colorbar(im, ax=ax4, fraction=0.02, pad=0.01)
    cb.set_label("ADU")

    # ── Panel A: H06 fringe profile + best-fit model overlay ─────────────────
    if airglow_result is not None and airglow_result.profile_r_grid is not None:
        r_px  = airglow_result.profile_r_grid
        p_adu = airglow_result.profile_adu
        m_adu = airglow_result.profile_model
        axA.plot(r_px, p_adu, color="steelblue", lw=0.9, alpha=0.85,
                 label="Data (dark-sub)")
        if m_adu is not None:
            axA.plot(r_px, m_adu, color="darkorange", lw=1.4, ls="--",
                     label="H06 model")
        axA.set_xlabel("Radius (px)")
        axA.set_ylabel("ADU")
        chi2_str = f"{airglow_result.chi2_red:.2f}"
        R_str    = f"{airglow_result.Y_line:.0f}"
        axA.set_title(f"H06 Fringe Fit  χ²={chi2_str}  Y={R_str}")
        axA.legend(fontsize=7, loc="upper right")
        axA.grid(True, alpha=0.2)
    else:
        axA.set_visible(False)

    # ── Panel B: H06 scan chi2 curve ─────────────────────────────────────────
    if airglow_result is not None and airglow_result.scan_v_ms is not None:
        v_scan  = airglow_result.scan_v_ms
        c2_scan = airglow_result.scan_chi2
        axB.plot(v_scan, c2_scan, color="steelblue", lw=0.9)
        axB.axvline(airglow_result.v_rel_ms, color="red", lw=1.2,
                    label=f"v_rel={airglow_result.v_rel_ms:+.0f} m/s")
        if geom is not None:
            v_prior_val = -(geom.V_sc_LOS + geom.v_earth_LOS)
            axB.axvline(v_prior_val, color="green", lw=1.2, ls="--",
                        label=f"v_prior={v_prior_val:+.0f} m/s")
        ambig_str = str(airglow_result.scan_ambiguous)
        axB.set_title(f"H06 Scan  scan_ambig={ambig_str}")
        axB.set_xlabel("v (m/s, Harding conv.)")
        axB.set_ylabel("χ²/ν")
        axB.legend(fontsize=7, loc="upper right")
        axB.grid(True, alpha=0.2)
    else:
        axB.set_visible(False)

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
    _ONE_ORBIT_SEC = 6000.0
    if dark_candidates:
        nearest = _find_nearest_dark_frame(meta.lua_timestamp, dark_candidates)
        if nearest is not None:
            dark_path, dt_sec = nearest
            if dt_sec <= _ONE_ORBIT_SEC:
                if _HAS_M03:
                    try:
                        image = _m03_module.subtract_dark(image, dark_path)
                        print(f"Dark subtracted using: {dark_path.name} (dt={dt_sec:.0f}s)")
                    except Exception as exc:
                        print(f"WARNING: Dark subtraction (M03) failed: {exc}")
                else:
                    # M03 not available — native subtraction via ingest
                    try:
                        _, dark_pixels = ingest_real_image(dark_path)
                        image = np.clip(
                            image.astype(np.float32) - dark_pixels.astype(np.float32),
                            0, 16383,
                        ).astype(np.uint16)
                        print(f"Dark subtracted using: {dark_path.name} (dt={dt_sec:.0f}s)")
                    except Exception as exc:
                        print(f"WARNING: Dark subtraction failed: {exc}")
            else:
                print(
                    f"WARNING: Nearest dark {dark_path.name} is {dt_sec:.0f}s away "
                    f"(> 1 orbit) — skipping dark subtraction"
                )
        else:
            print("WARNING: No dark frame found — skipping dark subtraction")
    else:
        print("WARNING: No dark frame found — skipping dark subtraction")

    # ── use_h06 activation ────────────────────────────────────────────────────
    use_h06 = args.use_h06 or (args.v_rel is None)
    if use_h06 and args.v_rel is not None:
        print("NOTE: --v-rel supplied — using override, ignoring --use-h06")
        use_h06 = False

    # ── Stage 3 — H07 geometry (always runs; needed before H06) ──────────────
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

    # ── Stage 2 — v_rel recovery ─────────────────────────────────────────────
    # Priority order:
    #   1. H06 fringe fitting (if use_h06=True and cal frames available)
    #   2. --v-rel override (if supplied)
    #   3. Not available (print guidance)
    #
    # NOTE: geom must be computed before this stage (see Stage 3 above).
    #       v_los_prior_ms = geom.V_sc_LOS + geom.v_earth_LOS

    v_rel = None
    sigma_v = args.sigma_v
    airglow_result = None   # AirglowResult if H06 was run, else None

    if use_h06:
        cal_frames = _find_cal_frames(path)
        if not cal_frames:
            print(
                "WARNING: H06 requested but no *_cal*.bin files found in "
                f"{path.parent}\n"
                "         Falling back to --v-rel if supplied."
            )
        else:
            # Build per-folder master calibration from all available cal frames
            n_cal_limit = args.n_cal if args.n_cal > 0 else len(cal_frames)
            print(f"H06 mode: found {len(cal_frames)} cal frame(s) in folder "
                  f"(using first {n_cal_limit})")
            try:
                from windcube.fpi_pipeline import (
                    process_cal_frame,
                    average_calibrations,
                    process_science_frame,
                )
                from src.metadata.p01_image_metadata_2026_04_06 import ingest_real_image as _ingest

                # Dark-subtract and invert each cal frame
                cal_results = []
                master_dark_arr, n_dark = _build_master_dark_from_folder(
                    path, meta.lua_timestamp
                )
                for cal_path in cal_frames[:n_cal_limit]:
                    try:
                        _, cal_pixels = _ingest(cal_path,
                                                h_target_km_obs=args.h_target_km)
                        if master_dark_arr is not None and n_dark > 0:
                            cal_ds = (cal_pixels.astype(np.float32)
                                      - master_dark_arr / n_dark)
                        else:
                            cal_ds = cal_pixels.astype(np.float32)
                        cal_result = process_cal_frame(cal_ds, r_max_px=110.0)
                        cal_results.append(cal_result)
                        print(f"  Cal: {cal_path.name}  "
                              f"chi2={cal_result.chi2_red:.3f}  "
                              f"converged={cal_result.converged}")
                    except Exception as exc:
                        print(f"  WARNING: Cal frame failed: {cal_path.name}: {exc}")

                if not cal_results:
                    print("WARNING: All cal frames failed — falling back to --v-rel")
                else:
                    master_cal = average_calibrations(cal_results)
                    print(
                        f"Master cal: {len(cal_results)} frames averaged  "
                        f"t_m={master_cal.t_m*1e3:.6f} mm  "
                        f"epsilon_cal={master_cal.epsilon_cal:.6f}"
                    )

                    # v_los_prior from geometry (already computed in Stage 3)
                    # H06 uses Harding recession-positive convention; V_sc_LOS
                    # is approach-positive so must be negated.
                    v_los_prior = (-(geom.V_sc_LOS + geom.v_earth_LOS)
                                   if geom is not None else 0.0)

                    # Dark-subtract science frame
                    if master_dark_arr is not None and n_dark > 0:
                        pixels_ds = np.clip(
                            image.astype(np.float32) - master_dark_arr / n_dark,
                            0, 16383,
                        ).astype(np.float32)
                    else:
                        pixels_ds = image.astype(np.float32)

                    # H06 inversion
                    airglow_result = process_science_frame(
                        pixels_ds      = pixels_ds,
                        master_cal     = master_cal,
                        v_los_prior_ms = v_los_prior,
                        r_max_px       = 110.0,
                    )
                    v_rel   = airglow_result.v_rel_ms
                    sigma_v = airglow_result.sigma_v_ms
                    print(
                        f"H06 result: v_rel={v_rel:+.2f} m/s  "
                        f"sigma_v={sigma_v:.2f} m/s  "
                        f"chi2={airglow_result.chi2_red:.3f}  "
                        f"converged={airglow_result.converged}"
                    )
            except Exception as exc:
                print(f"WARNING: H06 pipeline failed: {exc}")
                print("         Falling back to --v-rel if supplied.")

    # --v-rel override (fallback or explicit)
    if v_rel is None and args.v_rel is not None:
        v_rel  = args.v_rel
        sigma_v = args.sigma_v

    if v_rel is None:
        print("WARNING: v_rel not available.")
        print("  H06 requires cal frames in the same folder, OR")
        print("  supply --v-rel <value> to proceed past this stage.")

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

        if airglow_result is not None:
            print()
            print("[ H06 FRINGE FIT ]")
            print(f"  chi2_red   : {airglow_result.chi2_red:.4f}")
            print(f"  converged  : {airglow_result.converged}")
            print(f"  scan_ambig : {airglow_result.scan_ambiguous}")
            print(f"  n_bins     : {airglow_result.n_bins}")
            print(f"  lc_m       : {airglow_result.lc_m*1e9:.7f} nm")
            print(f"  Y_line     : {airglow_result.Y_line:.2f} ADU")
            print(f"  B_sci      : {airglow_result.B_sci:.2f} ADU")
            budget = "✓" if airglow_result.budget_ok else "✗"
            print(f"  sigma_v    : {airglow_result.sigma_v_ms:.3f} m/s  "
                  f"{budget} STM budget (≤9.8 m/s)")

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
            v_corrected=v_corrected,
            path=path,
            save_plots=args.save_plots,
            airglow_result=airglow_result,
        )


def main() -> None:
    # Ensure UTF-8 output on Windows terminals
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    args = _parse_args()

    # If no path supplied on command line, open a Windows file picker
    if not args.path_to_bin_file:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        chosen = filedialog.askopenfilename(
            title="Select a WindCube FPI science .bin file",
            initialdir=r"C:\Users\sewell\WindCube\G01_outputs",
            filetypes=[
                ("WindCube binary", "*.bin"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
        if not chosen:
            print("No file selected — exiting.")
            return
        args.path_to_bin_file = chosen

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
