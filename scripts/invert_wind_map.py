#!/usr/bin/env python
"""
invert_wind_map.py — WindCube H07 batch processing driver.

Two v_rel modes:

  CSV mode (--v-rel-csv):
    v_rel is read from a pre-computed GEN01 CSV. Used for synthetic data
    validation. All existing --v-rel-csv runs continue to work unchanged.

  H06 mode (default when --v-rel-csv not supplied):
    v_rel is recovered from raw pixels via the full H06 fringe fitting
    pipeline. Requires dark and cal frames in the input folder.
    Uses per-orbit master darks and master calibrations (5 dark + 5 cal
    frames per orbit, per WINDCUBE-ARCH-01).

Usage:
    python scripts/invert_wind_map.py <input_folder> [options]

Spec: H07_wind_vector_retrieval_2026-05-14_v03.md (batch mode)

NOTE FOR FUTURE DEVELOPMENT:
This script currently writes results to CSV for simplicity and visual
validation. The production output format is netCDF-4 as defined in
S20/M08 (specs/S20_m08_l2_netcdf_writer.md). When M08 is available,
replace the CSV writer (Step: write_results_csv) with a call to
windcube.m08.write_l2_netcdf(wind_solutions, output_path).
"""

from __future__ import annotations

import argparse
import bisect
import logging
import re
import sys
import time
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ── Project root on sys.path ───────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ─────────────────────────────────────────────────────────────────────────────
# NULL-WIND VALIDATION
#
# To validate H07 with GEN01 synthetic data (null wind):
#
#   1. Generate a synthetic dataset with GEN01 (wind map option 1,
#      v_zonal=0, v_merid=0). This produces:
#        - A folder of *_science.bin files
#        - A CSV with v_rel_ms pre-computed (GEN01_*_uniform_*.csv)
#
#   2. Run this script with --v-rel-csv pointing to the GEN01 CSV:
#        python scripts/invert_wind_map.py <folder> \
#            --v-rel-csv <GEN01_csv_path> --sigma-v 10.0
#
#   3. Check the summary report:
#        v_E mean should be ~0.0 m/s (< 5 m/s systematic = PASS)
#        v_N mean should be ~0.0 m/s (< 5 m/s systematic = PASS)
#
#   4. Open the output CSV in a spreadsheet or with pandas to inspect
#      individual bin solutions.
#
# If v_E or v_N mean is far from zero, check:
#   - That --v-rel-csv timestamps match the .bin file lua_timestamps
#   - That obs_mode is being correctly read from sidecar or derived
#   - The geometry by running invert_single_frame.py on one frame
# ─────────────────────────────────────────────────────────────────────────────

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WindCube H07 batch wind-map processing driver.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_folder",
        help="Directory containing *_science.bin files (non-recursive)",
    )
    parser.add_argument(
        "--h-target-km",
        type=float,
        default=250.0,
        metavar="KM",
        help="Emission layer altitude in km (default: 250.0)",
    )
    parser.add_argument(
        "--dlat",
        type=float,
        default=5.0,
        metavar="DEG",
        help="Bin latitude width in degrees (default: 5.0)",
    )
    parser.add_argument(
        "--dlon",
        type=float,
        default=5.0,
        metavar="DEG",
        help="Bin longitude width in degrees (default: 5.0)",
    )
    parser.add_argument(
        "--dt-min",
        type=float,
        default=0.0,
        metavar="MIN",
        help=(
            "Bin time width in minutes. "
            "Use 0 for geographic-only binning (recommended for single-day "
            "datasets — accumulates both orbital modes per location). "
            "Default: 0"
        ),
    )
    parser.add_argument(
        "--n-days",
        type=int,
        default=None,
        metavar="N",
        help="Number of days to accumulate (default: process all files found)",
    )
    parser.add_argument(
        "--v-rel-csv",
        default=None,
        metavar="PATH",
        help=(
            "GEN01 CSV file with v_rel_ms column keyed by lua_timestamp. "
            "When supplied, v_rel is read from the CSV instead of M06."
        ),
    )
    parser.add_argument(
        "--sigma-v",
        type=float,
        default=10.0,
        metavar="M_S",
        help="Constant per-frame sigma_v in m/s when --v-rel-csv is supplied (default: 10.0)",
    )
    parser.add_argument(
        "--use-h06",
        action="store_true",
        default=False,
        help=(
            "Use the full H06 fringe fitting pipeline to recover v_rel from "
            "raw pixels. Activated automatically when --v-rel-csv is not "
            "supplied. Requires cal and dark frames in the input folder."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="PATH",
        help="Output directory for CSV and summary files (default: input_folder)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        metavar="N",
        help="Parallel worker threads (default: 1 = serial; see astropy note in code)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Set logging level to INFO for per-frame debug messages",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# v_rel CSV loader
# ---------------------------------------------------------------------------


def _load_vrel_csv(csv_path: Path, sigma_v: float) -> dict:
    """
    Read a GEN01 CSV and build a per-frame metadata lookup dict.

    Returns {lua_timestamp_int: dict} where each dict has keys:
        v_rel, sigma_v, obs_mode, tangent_lat, tangent_lon,
        h_target_km_obs, is_synthetic.
    Only science rows are included when the obs_type column is present.
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    if "obs_type" in df.columns:
        df = df[df["obs_type"] == "science"].copy()
    if "v_rel_ms" not in df.columns:
        raise ValueError(
            f"Column 'v_rel_ms' not found in {csv_path}. "
            "Check that this is a GEN01 v14 CSV."
        )
    if "lua_timestamp" not in df.columns:
        raise ValueError(f"Column 'lua_timestamp' not found in {csv_path}.")

    lookup = {}
    has_tangent = "tp_lat_deg" in df.columns and "tp_lon_deg" in df.columns
    has_h_target = "h_target_km_obs" in df.columns
    has_obs_mode = "obs_mode" in df.columns

    for _, row in df.iterrows():
        ts = int(row["lua_timestamp"])
        entry: dict = {
            "v_rel": float(row["v_rel_ms"]),
            "sigma_v": float(sigma_v),
            "obs_mode": None,
            "tangent_lat": None,
            "tangent_lon": None,
            "h_target_km_obs": None,
            "is_synthetic": has_tangent,  # GEN01 CSV implies synthetic
        }
        if has_obs_mode:
            om = row["obs_mode"]
            entry["obs_mode"] = str(om) if om == om else None  # NaN guard
        if has_tangent:
            lat = row["tp_lat_deg"]
            lon = row["tp_lon_deg"]
            entry["tangent_lat"] = float(lat) if lat == lat else None
            entry["tangent_lon"] = float(lon) if lon == lon else None
        if has_h_target:
            htk = row["h_target_km_obs"]
            entry["h_target_km_obs"] = float(htk) if htk == htk else None
        lookup[ts] = entry
    return lookup


# ---------------------------------------------------------------------------
# Dark-frame helpers (Fix 3)
# ---------------------------------------------------------------------------


def _parse_filename_timestamp_ms(fname: str) -> int | None:
    """
    Parse YYYY-MM-DDTHH-MM-SSZ from a filename stem.
    Returns Unix milliseconds or None if the pattern is not found.
    """
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})T(\d{2})-(\d{2})-(\d{2})Z", fname)
    if not m:
        return None
    dt = datetime(
        int(m.group(1)), int(m.group(2)), int(m.group(3)),
        int(m.group(4)), int(m.group(5)), int(m.group(6)),
        tzinfo=timezone.utc,
    )
    return int(dt.timestamp() * 1000)


def _load_dark_frames(folder: Path) -> list:
    """
    Scan folder for *_dark.bin files and return a sorted list of
    (unix_ms, Path) tuples.  Returns [] if none found.
    """
    result = []
    for p in sorted(folder.glob("*_dark.bin")):
        ts = _parse_filename_timestamp_ms(p.name)
        if ts is not None:
            result.append((ts, p))
    result.sort(key=lambda x: x[0])
    return result


def _find_nearest_dark(ts_ms: int, dark_frames: list) -> tuple:
    """
    Return the (unix_ms, Path) entry from dark_frames whose timestamp
    is closest to ts_ms.  dark_frames must be sorted by timestamp.
    """
    times = [t for t, _ in dark_frames]
    idx = bisect.bisect_left(times, ts_ms)
    candidates = []
    if idx < len(dark_frames):
        candidates.append(dark_frames[idx])
    if idx > 0:
        candidates.append(dark_frames[idx - 1])
    return min(candidates, key=lambda x: abs(x[0] - ts_ms))


# ---------------------------------------------------------------------------
# Sidecar merge helper
# ---------------------------------------------------------------------------


def _apply_sidecar(meta, bin_path: Path) -> None:
    """
    Try to merge companion JSON sidecar into meta (in-place).
    Silently skips if no sidecar found or if reading fails.
    """
    from src.metadata.p01_image_metadata_2026_04_06 import read_sidecar
    sidecar = bin_path.with_name(bin_path.stem + "_L0.json")
    if not sidecar.exists():
        return
    try:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            meta_sc = read_sidecar(sidecar)
        if meta_sc.obs_mode not in (None, "unknown"):
            meta.obs_mode = meta_sc.obs_mode
        if meta_sc.h_target_km_obs is not None:
            meta.h_target_km_obs = meta_sc.h_target_km_obs
        for field in (
            "is_synthetic", "truth_v_los", "truth_v_zonal",
            "truth_v_meridional", "tangent_lat", "tangent_lon",
            "tangent_alt_km", "etalon_gap_mm", "noise_seed",
        ):
            val = getattr(meta_sc, field, None)
            if val is not None:
                setattr(meta, field, val)
    except Exception as exc:
        log.warning("Failed to read sidecar %s: %s", sidecar.name, exc)


# ---------------------------------------------------------------------------
# Per-frame processor
# ---------------------------------------------------------------------------


def _process_one(
    bin_path: Path,
    vrel_lookup,
    args: argparse.Namespace,
    dark_frames: list = None,
    orbit_group=None,
) -> tuple:
    """
    Process a single *_science.bin file through H07.

    Returns (LOSObservation, None) on success,
    or (None, skip_reason_str) if the frame should be skipped.

    skip_reason values:
        'ingest_error'   — binary read failed
        'not_science'    — img_type != 'science'
        'slew'           — SLEW_IN_PROGRESS flag set
        'vrel_missing'   — lua_timestamp not in v_rel CSV
        'cal_missing'    — H06 mode but orbit has no master calibration
        'geometry_error' — process_frame raised ValueError
    """
    import numpy as np
    from src.metadata.p01_image_metadata_2026_04_06 import (
        AdcsQualityFlags,
        ingest_real_image,
    )
    from windcube import wind_retrieval

    # Step 2a — Ingest
    try:
        meta, pixels = ingest_real_image(bin_path, h_target_km_obs=args.h_target_km)
    except Exception as exc:
        log.warning("Ingest failed for %s: %s", bin_path.name, exc)
        return None, "ingest_error"

    _apply_sidecar(meta, bin_path)

    if meta.img_type != "science":
        log.debug("Skipping %s: img_type=%s", bin_path.name, meta.img_type)
        return None, "not_science"

    if meta.adcs_quality_flag & AdcsQualityFlags.SLEW_IN_PROGRESS:
        log.warning("Skipping %s: SLEW_IN_PROGRESS flag set", bin_path.name)
        return None, "slew"

    # Step 2b — Get v_rel and apply per-frame metadata from CSV
    if vrel_lookup is not None:
        entry = vrel_lookup.get(meta.lua_timestamp)
        if entry is None:
            log.warning(
                "v_rel not found for lua_timestamp=%d (%s)",
                meta.lua_timestamp,
                bin_path.name,
            )
            return None, "vrel_missing"
        v_rel = entry["v_rel"]
        sigma_v = entry["sigma_v"]
        # Apply metadata fields from GEN01 CSV (fixes obs_mode and tangent point)
        if entry.get("obs_mode") not in (None, "unknown"):
            meta.obs_mode = entry["obs_mode"]
        if entry.get("tangent_lat") is not None:
            meta.tangent_lat = entry["tangent_lat"]
            meta.tangent_lon = entry.get("tangent_lon")
            meta.is_synthetic = True
        if entry.get("h_target_km_obs") is not None:
            meta.h_target_km_obs = entry["h_target_km_obs"]
    else:
        # No CSV supplied — use H06 fringe fitting
        if orbit_group is None or orbit_group.master_cal is None:
            log.warning(
                "Skipping %s: no master calibration for this orbit",
                bin_path.name,
            )
            return None, "cal_missing"

        # ── H07 geometry first (needed for v_los_prior_ms) ─────────────
        from windcube import wind_retrieval
        try:
            geom = wind_retrieval.compute_los_geometry(meta)
        except ValueError as exc:
            log.warning("Geometry failed for %s: %s", bin_path.name, exc)
            return None, "geometry_error"

        v_los_prior_ms = geom.V_sc_LOS + geom.v_earth_LOS

        # ── Dark subtraction using per-orbit master dark ────────────────
        if orbit_group.master_dark is not None:
            n_dark = len(orbit_group.dark_frames)
            pixels_ds = (
                pixels.astype(np.float32)
                - orbit_group.master_dark / n_dark
            )
            pixels_ds = np.clip(pixels_ds, 0, 16383).astype(np.float32)
        else:
            log.warning(
                "No master dark for orbit %d — processing %s without "
                "dark subtraction",
                orbit_group.orbit_number, bin_path.name,
            )
            pixels_ds = pixels.astype(np.float32)

        # ── H06 fringe fitting via fpi_pipeline ────────────────────────
        from windcube.fpi_pipeline import process_science_frame
        try:
            airglow = process_science_frame(
                pixels_ds      = pixels_ds,
                master_cal     = orbit_group.master_cal,
                v_los_prior_ms = v_los_prior_ms,
                r_max_px       = 110.0,
                cx_seed        = None,
                cy_seed        = None,
            )
        except Exception as exc:
            log.warning(
                "H06 inversion failed for %s: %s", bin_path.name, exc
            )
            return None, "geometry_error"

        v_rel  = airglow.v_rel_ms
        sigma_v = airglow.sigma_v_ms
        log.info(
            "H06  %-40s  v_rel=%+.1f  sigma_v=%.1f  chi2=%.2f  "
            "converged=%s",
            bin_path.name, v_rel, sigma_v,
            airglow.chi2_red, airglow.converged,
        )

    # Step 2a2 — Dark subtraction (nearest dark frame within 1 orbit)
    _ONE_ORBIT_MS = 6_000_000
    if dark_frames:
        dark_ms, dark_path = _find_nearest_dark(meta.lua_timestamp, dark_frames)
        dt_ms = abs(meta.lua_timestamp - dark_ms)
        if dt_ms <= _ONE_ORBIT_MS:
            try:
                _, dark_pixels = ingest_real_image(dark_path)
                pixels_corrected = np.clip(
                    pixels.astype(np.float32) - dark_pixels.astype(np.float32),
                    0, 16383,
                ).astype(np.uint16)
                log.info(
                    "Dark subtracted: %s (dt=%.0fs)",
                    dark_path.name, dt_ms / 1000.0,
                )
            except Exception as exc:
                log.warning(
                    "Dark subtraction failed for %s using %s: %s",
                    bin_path.name, dark_path.name, exc,
                )
        else:
            log.warning(
                "No dark frame within 1 orbit of %s — skipping dark subtraction",
                bin_path.name,
            )

    # Step 2c — H07 geometry + correction
    try:
        obs = wind_retrieval.process_frame(meta, v_rel, sigma_v)
    except ValueError as exc:
        log.warning("process_frame failed for %s: %s", bin_path.name, exc)
        return None, "geometry_error"

    log.info(
        "OK  %-40s  v_rel=%+.1f  v_corr=%+.1f  mode=%s",
        bin_path.name, v_rel, obs.v_corrected, obs.obs_mode,
    )
    return obs, None


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _unix_ms_to_utc(unix_ms: int) -> str:
    dt = datetime.fromtimestamp(unix_ms / 1000.0, tz=timezone.utc)
    return dt.isoformat()


def _write_results_csv(
    bin_keys: list,
    solutions: list,
    output_path: Path,
) -> None:
    """Write wind solutions to CSV. One row per spatiotemporal bin."""
    import csv
    import math

    columns = [
        "bin_lat_deg", "bin_lon_deg", "bin_t_centre_unix_ms", "bin_t_centre_utc",
        "n_frames", "obs_modes",
        "v_E_ms", "v_N_ms", "sigma_v_E_ms", "sigma_v_N_ms",
        "two_sigma_v_E_ms", "two_sigma_v_N_ms",
        "condition_number", "gdop_flag", "n_frames_flag",
        "mean_tangent_lat_deg", "mean_tangent_lon_deg", "mean_tangent_alt_km",
        "mean_epoch_unix_ms",
    ]

    def _fmt(v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return ""
        return v

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for (lat_c, lon_c, t_c), sol in zip(bin_keys, solutions):
            modes_str = "|".join(sorted(sol.obs_modes)) if sol.obs_modes else ""
            writer.writerow({
                "bin_lat_deg":          lat_c,
                "bin_lon_deg":          lon_c,
                "bin_t_centre_unix_ms": t_c,
                "bin_t_centre_utc":     _unix_ms_to_utc(t_c),
                "n_frames":             sol.n_frames,
                "obs_modes":            modes_str,
                "v_E_ms":               _fmt(sol.v_E),
                "v_N_ms":               _fmt(sol.v_N),
                "sigma_v_E_ms":         _fmt(sol.sigma_v_E),
                "sigma_v_N_ms":         _fmt(sol.sigma_v_N),
                "two_sigma_v_E_ms":     _fmt(sol.two_sigma_v_E),
                "two_sigma_v_N_ms":     _fmt(sol.two_sigma_v_N),
                "condition_number":     _fmt(sol.condition_number),
                "gdop_flag":            sol.gdop_flag,
                "n_frames_flag":        sol.n_frames_flag,
                "mean_tangent_lat_deg": sol.mean_tangent_lat_deg,
                "mean_tangent_lon_deg": sol.mean_tangent_lon_deg,
                "mean_tangent_alt_km":  sol.mean_tangent_alt_km,
                "mean_epoch_unix_ms":   sol.mean_epoch_unix_ms,
            })


def _write_summary(
    input_folder: Path,
    n_found: int,
    n_processed: int,
    skip_counts: dict,
    solutions: list,
    dlat: float,
    dlon: float,
    dt_min: float,
    csv_path: Path,
    elapsed: float,
    output_path: Path,
) -> str:
    """Build, write, and return the summary report text."""
    import numpy as np

    n_skipped = n_found - n_processed
    n_bins = len(solutions)
    good = [s for s in solutions if not s.gdop_flag and not s.n_frames_flag]
    n_good = len(good)
    n_gdop = sum(1 for s in solutions if s.gdop_flag)
    n_sparse = sum(1 for s in solutions if s.n_frames_flag)

    def pct(n):
        return 100.0 * n / n_bins if n_bins > 0 else 0.0

    n_vrel_missing = skip_counts.get("vrel_missing", 0) + skip_counts.get("m06_missing", 0)

    lines = [
        "=" * 64,
        "WindCube H07 Wind Map — Batch Processing Summary",
        "=" * 64,
        f"Input folder   : {input_folder}",
        (
            f"Science frames : {n_found} found, {n_processed} processed, "
            f"{n_skipped} skipped"
        ),
        "Skipped breakdown:",
        f"  SLEW_IN_PROGRESS flag : {skip_counts.get('slew', 0)} frames",
        (
            f"  v_rel not available   : {n_vrel_missing} frames"
            f"  (M06 missing or not in CSV)"
        ),
        f"  No master calibration  : {skip_counts.get('cal_missing', 0)} frames",
        f"  Geometry error        : {skip_counts.get('geometry_error', 0)} frames",
        "-" * 64,
        (
            f"Bin parameters : dlat={dlat}°  dlon={dlon}°  dt=geographic-only"
            if dt_min == 0 else
            f"Bin parameters : dlat={dlat}°  dlon={dlon}°  dt={dt_min} min"
        ),
        f"Total bins     : {n_bins}",
        f"Good solutions : {n_good}  ({pct(n_good):.1f}%)",
        f"GDOP flagged   : {n_gdop}  ({pct(n_gdop):.1f}%)",
        f"Too few frames : {n_sparse}  ({pct(n_sparse):.1f}%)",
        "-" * 64,
    ]

    if good:
        ve = np.array([s.v_E for s in good])
        vn = np.array([s.v_N for s in good])
        sve = np.array([s.sigma_v_E for s in good])
        svn = np.array([s.sigma_v_N for s in good])
        lines += [
            "Wind statistics (good bins only):",
            (
                f"  v_E  : mean={np.mean(ve):.1f}  std={np.std(ve):.1f}  "
                f"min={np.min(ve):.1f}  max={np.max(ve):.1f}  m/s"
            ),
            (
                f"  v_N  : mean={np.mean(vn):.1f}  std={np.std(vn):.1f}  "
                f"min={np.min(vn):.1f}  max={np.max(vn):.1f}  m/s"
            ),
            f"  sigma_v_E : mean={np.mean(sve):.1f}  median={np.median(sve):.1f}  m/s",
            f"  sigma_v_N : mean={np.mean(svn):.1f}  median={np.median(svn):.1f}  m/s",
        ]
    else:
        lines.append("Wind statistics: no good bins to report")

    lines += [
        "-" * 64,
        (
            "For null-wind validation: v_E mean and v_N mean should both be\n"
            "near 0.0 m/s. Systematic offset > 5 m/s indicates a geometry error."
        ),
        "-" * 64,
        f"Output CSV     : {csv_path}",
        f"Processing time: {elapsed:.1f} s",
        "=" * 64,
    ]

    text = "\n".join(lines) + "\n"
    output_path.write_text(text, encoding="utf-8")
    return text


# ---------------------------------------------------------------------------
# H06 schedule builder
# ---------------------------------------------------------------------------


def _build_schedule(input_folder: Path, args: argparse.Namespace):
    """
    Build the per-orbit calibration schedule using cal_scheduler.

    Called once at startup in H06 mode. Processes all dark and cal
    frames in input_folder. Returns OrbitSchedule.

    Prints progress from build_orbit_schedule() directly.
    On failure, prints error and calls sys.exit(1).
    """
    from windcube.cal_scheduler import build_orbit_schedule
    try:
        schedule = build_orbit_schedule(
            input_folder,
            r_max_px        = 110.0,
            h_target_km_obs = args.h_target_km,
            process_cals    = True,
        )
    except Exception as exc:
        print(f"ERROR: Failed to build calibration schedule: {exc}",
              file=sys.stderr)
        sys.exit(1)

    # Warn about orbits with missing calibration
    if schedule.n_cal_missing > 0:
        print(
            f"WARNING: {schedule.n_cal_missing}/{schedule.n_orbits} orbits "
            "have no master calibration — science frames in those orbits "
            "will be skipped."
        )
    return schedule


# ---------------------------------------------------------------------------
# n_days file-list filter
# ---------------------------------------------------------------------------


def _filter_by_n_days(bin_files: list, n_days: int) -> list:
    """
    Return only files whose filename date falls within the first n_days
    of the dataset.  Assumes filenames begin with YYYY-MM-DD (GEN01 format).
    Falls back to returning all files if date extraction fails.
    """
    if not bin_files:
        return bin_files
    try:
        first_date_str = bin_files[0].name[:10]
        first_date = datetime.strptime(first_date_str, "%Y-%m-%d").date()
        cutoff = first_date + timedelta(days=n_days)
        cutoff_str = cutoff.strftime("%Y-%m-%d")
        return [f for f in bin_files if f.name[:10] < cutoff_str]
    except (ValueError, IndexError):
        log.warning(
            "--n-days: could not parse date from filename '%s'; "
            "processing all files.",
            bin_files[0].name,
        )
        return bin_files


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # Ensure UTF-8 output on Windows terminals
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    args = _parse_args()

    use_h06 = args.use_h06 or (args.v_rel_csv is None)
    if use_h06 and args.v_rel_csv:
        print("NOTE: --v-rel-csv supplied — using CSV path, ignoring --use-h06")
        use_h06 = False

    logging.basicConfig(
        format="%(levelname)s %(name)s: %(message)s",
        level=logging.INFO if args.verbose else logging.WARNING,
        stream=sys.stderr,
    )

    input_folder = Path(args.input_folder).resolve()
    if not input_folder.is_dir():
        print(f"ERROR: Input folder not found: {input_folder}", file=sys.stderr)
        sys.exit(1)

    output_dir = (
        Path(args.output_dir).resolve() if args.output_dir else input_folder
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    datestamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    stem = input_folder.name
    csv_path = output_dir / f"{stem}_wind_solutions_{datestamp}.csv"
    summary_path = output_dir / f"{stem}_wind_summary_{datestamp}.txt"

    t0 = time.monotonic()

    # ── Step 0 — Discover files ────────────────────────────────────────────────
    bin_files = sorted(input_folder.glob("*_science.bin"))
    if args.n_days is not None:
        bin_files = _filter_by_n_days(bin_files, args.n_days)

    n_found = len(bin_files)
    print(f"Found {n_found} science frames in {input_folder}")
    if n_found == 0:
        print("ERROR: No *_science.bin files found in input folder.", file=sys.stderr)
        sys.exit(1)

    # ── Step 0b — Discover dark frames ────────────────────────────────────────
    dark_frames = _load_dark_frames(input_folder)
    if dark_frames:
        print(f"Found {len(dark_frames)} dark frames for subtraction")
    else:
        log.warning("No dark frames found in %s — skipping dark subtraction", input_folder)

    # ── Step 0c — Build calibration schedule (H06 mode only) ──────────────────
    schedule = None
    if use_h06:
        print("H06 mode: building per-orbit calibration schedule...")
        schedule = _build_schedule(input_folder, args)
        print(
            f"Schedule: {schedule.n_orbits} orbits, "
            f"{schedule.n_science} science frames"
        )

    # ── Step 1 — Load v_rel lookup ─────────────────────────────────────────────
    vrel_lookup = None
    if args.v_rel_csv:
        vrel_csv_path = Path(args.v_rel_csv)
        if not vrel_csv_path.exists():
            print(f"ERROR: v_rel CSV not found: {vrel_csv_path}", file=sys.stderr)
            sys.exit(1)
        try:
            vrel_lookup = _load_vrel_csv(vrel_csv_path, args.sigma_v)
        except Exception as exc:
            print(f"ERROR: Failed to load v_rel CSV: {exc}", file=sys.stderr)
            sys.exit(1)
        print(f"Loaded v_rel lookup from CSV: {len(vrel_lookup)} rows")

    # ── Step 2 — Process frames ────────────────────────────────────────────────
    obs_list = []
    skip_counts: dict = {}

    if args.max_workers > 1 and schedule is not None:
        print("H06 mode: parallel processing not supported — using serial")

    if args.max_workers > 1 and schedule is None:
        log.warning(
            "max_workers=%d: astropy thread-safety not verified — "
            "use --max-workers 1 for production runs.",
            args.max_workers,
        )
        from concurrent.futures import ThreadPoolExecutor
        import functools
        fn = functools.partial(
            _process_one, vrel_lookup=vrel_lookup, args=args, dark_frames=dark_frames
        )
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            n_done = 0
            n_skipped_so_far = 0
            for obs, reason in pool.map(fn, bin_files):
                n_done += 1
                if obs is not None:
                    obs_list.append(obs)
                else:
                    skip_counts[reason] = skip_counts.get(reason, 0) + 1
                    n_skipped_so_far += 1
                if n_done % 100 == 0:
                    print(f"  Processed {n_done}/{n_found}  ({n_skipped_so_far} skipped)")
    else:
        n_skipped_so_far = 0
        for i, bin_path in enumerate(bin_files, 1):
            # Resolve orbit group for H06 mode
            og = None
            if schedule is not None:
                for og_candidate in schedule.orbits:
                    if any(fr.path == bin_path
                           for fr in og_candidate.science_frames):
                        og = og_candidate
                        break

            obs, reason = _process_one(
                bin_path, vrel_lookup, args,
                dark_frames=dark_frames,
                orbit_group=og,
            )
            if obs is not None:
                obs_list.append(obs)
            else:
                skip_counts[reason] = skip_counts.get(reason, 0) + 1
                n_skipped_so_far += 1
            if i % 100 == 0:
                print(f"  Processed {i}/{n_found}  ({n_skipped_so_far} skipped)")

    n_processed = len(obs_list)

    if n_processed == 0:
        print(
            "WARNING: No frames produced valid LOSObservations. "
            "Check v_rel source and skip reasons above.",
            file=sys.stderr,
        )
        # Still write an empty summary
        elapsed = time.monotonic() - t0
        report = _write_summary(
            input_folder=input_folder,
            n_found=n_found,
            n_processed=0,
            skip_counts=skip_counts,
            solutions=[],
            dlat=args.dlat,
            dlon=args.dlon,
            dt_min=args.dt_min,
            csv_path=csv_path,
            elapsed=elapsed,
            output_path=summary_path,
        )
        print(report, end="")
        sys.exit(0)

    # ── Step 3 — Bin and invert ────────────────────────────────────────────────
    from windcube import wind_retrieval
    bins = wind_retrieval.bin_observations(
        obs_list, dlat=args.dlat, dlon=args.dlon, dt_min=args.dt_min
    )
    n_bins = len(bins)
    print(f"Binning: {n_processed} observations → {n_bins} bins")

    bin_keys = sorted(bins.keys())
    solutions = []
    for key in bin_keys:
        sol = wind_retrieval.invert_wind_vector(bins[key])
        solutions.append(sol)

    n_good = sum(1 for s in solutions if not s.gdop_flag and not s.n_frames_flag)
    n_gdop = sum(1 for s in solutions if s.gdop_flag)
    n_sparse = sum(1 for s in solutions if s.n_frames_flag)
    print(f"  Good solutions  : {n_good}")
    print(f"  GDOP flagged    : {n_gdop}")
    print(f"  Too few frames  : {n_sparse}")

    # ── Step 4 — Write results CSV ─────────────────────────────────────────────
    _write_results_csv(bin_keys, solutions, csv_path)
    print(f"Results written to: {csv_path}")

    # ── Step 5 — Write summary report ─────────────────────────────────────────
    elapsed = time.monotonic() - t0
    report = _write_summary(
        input_folder=input_folder,
        n_found=n_found,
        n_processed=n_processed,
        skip_counts=skip_counts,
        solutions=solutions,
        dlat=args.dlat,
        dlon=args.dlon,
        dt_min=args.dt_min,
        csv_path=csv_path,
        elapsed=elapsed,
        output_path=summary_path,
    )
    print(report, end="")
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
