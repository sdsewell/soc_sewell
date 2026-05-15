"""
coverage_map.py — Global coverage visualization for WindCube GEN01 datasets.

Reads a GEN01 CSV (v14+ format) and produces 4 diagnostic figures:
  1. Coverage map (AT-only / CT-only / mixed / unsampled per 5-deg bin)
  2. Pass count map (AT and CT counts side by side)
  3. Coverage forecast curve (predicted mixed% vs simulation days)
  4. Ground track map (all tangent point locations)

Usage:
  python scripts/coverage_map.py <gen01_csv_path> [options]

Spec: specs/G01_synthetic_metadata_generator_2026-05-14_v15.md §10
"""

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend; overridden below for --show
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle

# ---------------------------------------------------------------------------
# Colour scheme (G01 v15 spec §10.2)
# ---------------------------------------------------------------------------
COLOR_AT    = "#0057C2"   # NCAR brand blue — along_track only
COLOR_CT    = "#FF7F0E"   # orange — cross_track only
COLOR_MIXED = "#2CA02C"   # green — mixed AT+CT
COLOR_EMPTY = "#EEEEEE"   # light grey — no coverage
COLOR_LAND  = "#CCCCCC"   # grey for background / coastlines


# ---------------------------------------------------------------------------
# Expected mixed-fraction formula (G01 v15 §9.3)
# ---------------------------------------------------------------------------
def _expected_mixed_fraction(n_days: float, dlon: float = 5.0) -> float:
    """Analytical estimate of mixed AT+CT bin fraction."""
    passes_per_day = 15.2
    lon_coverage = min(1.0, n_days * passes_per_day * 2 * dlon / 360.0)
    return lon_coverage * lon_coverage


# ---------------------------------------------------------------------------
# Bin computation
# ---------------------------------------------------------------------------
def _compute_bins(sci: pd.DataFrame, dlat: float, dlon: float) -> pd.DataFrame:
    """Return per-bin AT/CT counts and mode flags."""
    s = sci.copy()
    s["bin_lat"] = (s["tp_lat_deg"] / dlat).round() * dlat
    s["bin_lon"] = (s["tp_lon_deg"] / dlon).round() * dlon

    grp = s.groupby(["bin_lat", "bin_lon"])
    bin_stats = grp.apply(
        lambda g: pd.Series({
            "n_at": (g["obs_mode"] == "along_track").sum(),
            "n_ct": (g["obs_mode"] == "cross_track").sum(),
        })
    ).reset_index()
    bin_stats["has_both"] = (bin_stats["n_at"] >= 1) & (bin_stats["n_ct"] >= 1)
    bin_stats["at_only"]  = (bin_stats["n_at"] >= 1) & (bin_stats["n_ct"] == 0)
    bin_stats["ct_only"]  = (bin_stats["n_ct"] >= 1) & (bin_stats["n_at"] == 0)
    return bin_stats


# ---------------------------------------------------------------------------
# Figure 1 — Coverage map
# ---------------------------------------------------------------------------
def _fig_coverage_map(
    bin_stats: pd.DataFrame,
    dlat: float,
    dlon: float,
    n_days: float,
    pct_mixed: float,
    n_science: int,
) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    ax.set_facecolor(COLOR_LAND)

    for _, row in bin_stats.iterrows():
        x = row["bin_lon"] - dlon / 2
        y = row["bin_lat"] - dlat / 2
        if row["has_both"]:
            color = COLOR_MIXED
        elif row["at_only"]:
            color = COLOR_AT
        else:
            color = COLOR_CT
        ax.add_patch(Rectangle((x, y), dlon, dlat, color=color, linewidth=0))

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")

    # Grid lines every 30 deg
    for lon in range(-180, 181, 30):
        ax.axvline(lon, color="white", lw=0.4, alpha=0.5)
    for lat in range(-90, 91, 30):
        ax.axhline(lat, color="white", lw=0.4, alpha=0.5)

    legend_patches = [
        mpatches.Patch(color=COLOR_MIXED, label="Mixed AT+CT (good H07)"),
        mpatches.Patch(color=COLOR_AT,    label="Along-track only"),
        mpatches.Patch(color=COLOR_CT,    label="Cross-track only"),
        mpatches.Patch(color=COLOR_EMPTY, label="No coverage"),
    ]
    ax.legend(handles=legend_patches, loc="lower left", fontsize=9)

    ax.set_title(
        f"WindCube Coverage Map — {n_days:.1f}-day simulation\n"
        f"{pct_mixed:.0f}% mixed AT+CT  |  {n_science} science frames  |  "
        f"{dlat}°×{dlon}° bins",
        fontsize=11,
    )
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 2 — Pass count map
# ---------------------------------------------------------------------------
def _fig_pass_count_map(
    bin_stats: pd.DataFrame,
    dlat: float,
    dlon: float,
    n_days: float,
) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    for ax, col, cmap, label in [
        (ax1, "n_at", "Blues",   "Along-track"),
        (ax2, "n_ct", "Oranges", "Cross-track"),
    ]:
        ax.set_facecolor(COLOR_LAND)
        vmax = max(1, int(bin_stats[col].max()))
        sc = ax.scatter(
            bin_stats["bin_lon"],
            bin_stats["bin_lat"],
            c=bin_stats[col],
            cmap=cmap,
            vmin=0,
            vmax=vmax,
            s=(dlon * 3) ** 2,
            marker="s",
        )
        cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
        cbar.set_label("Number of passes")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
        ax.set_title(f"{label} pass count")
        for lon in range(-180, 181, 30):
            ax.axvline(lon, color="grey", lw=0.3, alpha=0.5)
        for lat in range(-90, 91, 30):
            ax.axhline(lat, color="grey", lw=0.3, alpha=0.5)

    fig.suptitle(
        f"Along-track and Cross-track Pass Counts — {n_days:.1f} days",
        fontsize=12,
    )
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 3 — Coverage forecast
# ---------------------------------------------------------------------------
def _fig_coverage_forecast(
    n_days: float,
    pct_mixed: float,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    days = np.arange(0, 14.1, 0.1)

    bin_configs = [
        (2.5, "dotted",  "2.5°×2.5° bins"),
        (5.0, "solid",   "5°×5° bins (default)"),
        (10.0, "dashed", "10°×10° bins"),
    ]
    for dlon_fc, ls, lbl in bin_configs:
        frac = [_expected_mixed_fraction(d, dlon=dlon_fc) * 100 for d in days]
        lw = 2.5 if dlon_fc == 5.0 else 1.5
        ax.plot(days, frac, linestyle=ls, lw=lw, label=lbl)

    # Markers on the 5°×5° line at specific days
    marker_days = [1, 3, 5, 7, 14]
    marker_vals = [_expected_mixed_fraction(d, dlon=5.0) * 100 for d in marker_days]
    ax.plot(marker_days, marker_vals, "o", color=COLOR_MIXED, zorder=5)

    # Vertical line at actual n_days
    ax.axvline(n_days, color="red", lw=1.2, linestyle="--", label=f"This run ({n_days:.1f} d)")

    # Horizontal reference lines
    for pct_ref in [50, 80, 90]:
        ax.axhline(pct_ref, color="grey", lw=0.8, linestyle="--", alpha=0.7)
        ax.text(14.1, pct_ref + 0.5, f"{pct_ref}%", fontsize=8, color="grey", va="bottom")

    # Annotation at actual run point
    actual_expected = _expected_mixed_fraction(n_days, dlon=5.0) * 100
    ax.annotate(
        f"This run:\n{n_days:.1f}d → {pct_mixed:.0f}%",
        xy=(n_days, pct_mixed),
        xytext=(n_days + 0.5, max(5, pct_mixed + 5)),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="red"),
        color="red",
    )

    ax.set_xlim(0, 14.5)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Simulation duration (days)")
    ax.set_ylabel("Predicted mixed AT+CT bins (%)")
    ax.set_title("Predicted H07 Coverage vs. Simulation Duration")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 4 — Ground track scatter
# ---------------------------------------------------------------------------
def _fig_ground_track(
    at_frames: pd.DataFrame,
    ct_frames: pd.DataFrame,
    n_days: float,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_facecolor(COLOR_LAND)

    if len(at_frames):
        ax.scatter(
            at_frames["tp_lon_deg"], at_frames["tp_lat_deg"],
            c=COLOR_AT, s=1, alpha=0.3, linewidths=0,
            label=f"Along-track ({len(at_frames)})",
        )
    if len(ct_frames):
        ax.scatter(
            ct_frames["tp_lon_deg"], ct_frames["tp_lat_deg"],
            c=COLOR_CT, s=1, alpha=0.3, linewidths=0,
            label=f"Cross-track ({len(ct_frames)})",
        )

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    for lon in range(-180, 181, 30):
        ax.axvline(lon, color="white", lw=0.4, alpha=0.5)
    for lat in range(-90, 91, 30):
        ax.axhline(lat, color="white", lw=0.4, alpha=0.5)
    ax.legend(fontsize=9, markerscale=5)
    ax.set_title(f"Tangent Point Coverage — {n_days:.1f}-day simulation")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="WindCube GEN01 global coverage visualization (G01 v15 §10)"
    )
    parser.add_argument("gen01_csv_path", help="Path to GEN01 output CSV (v14+ format)")
    parser.add_argument("--dlat",       type=float, default=5.0,  help="Bin latitude size in degrees (default: 5.0)")
    parser.add_argument("--dlon",       type=float, default=5.0,  help="Bin longitude size in degrees (default: 5.0)")
    parser.add_argument("--output-dir", default=None,             help="Directory for saved figures (default: same as CSV)")
    parser.add_argument("--save",       action="store_true",      help="Save figures to PNG instead of displaying")
    parser.add_argument("--dpi",        type=int,   default=150,  help="Figure DPI for saved files (default: 150)")
    args = parser.parse_args()

    csv_path = pathlib.Path(args.gen01_csv_path).resolve()
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = pathlib.Path(args.output_dir).resolve() if args.output_dir else csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.save:
        matplotlib.use("TkAgg")    # interactive backend for display

    # Load data
    df = pd.read_csv(csv_path)
    sci = df[df["obs_type"] == "science"].dropna(subset=["tp_lat_deg", "tp_lon_deg"])
    at_frames = sci[sci["obs_mode"] == "along_track"]
    ct_frames = sci[sci["obs_mode"] == "cross_track"]

    ts_col = sci["lua_timestamp"] if "lua_timestamp" in sci.columns else None
    if ts_col is not None and len(ts_col) > 1:
        n_days = (ts_col.max() - ts_col.min()) / 86_400_000.0
    else:
        n_days = 0.0

    dlat, dlon = args.dlat, args.dlon
    bin_stats = _compute_bins(sci, dlat, dlon)

    n_bins_sampled = len(bin_stats)
    n_bins_mixed   = int(bin_stats["has_both"].sum())
    n_bins_at_only = int(bin_stats["at_only"].sum())
    n_bins_ct_only = int(bin_stats["ct_only"].sum())
    pct_mixed = n_bins_mixed / n_bins_sampled * 100 if n_bins_sampled > 0 else 0.0
    n_science = len(sci)
    stem = csv_path.stem

    # Generate figures
    fig1 = _fig_coverage_map(bin_stats, dlat, dlon, n_days, pct_mixed, n_science)
    fig2 = _fig_pass_count_map(bin_stats, dlat, dlon, n_days)
    fig3 = _fig_coverage_forecast(n_days, pct_mixed)
    fig4 = _fig_ground_track(at_frames, ct_frames, n_days)

    if args.save:
        paths = {
            f"{stem}_coverage_map.png":       fig1,
            f"{stem}_pass_count_map.png":     fig2,
            f"{stem}_coverage_forecast.png":  fig3,
            f"{stem}_ground_track_map.png":   fig4,
        }
        for fname, fig in paths.items():
            out = output_dir / fname
            fig.savefig(str(out), dpi=args.dpi, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {out}")
    else:
        plt.show()

    # Summary printout (G01 v15 §10.3)
    pct_at = 100 * n_bins_at_only / max(n_bins_sampled, 1)
    pct_ct = 100 * n_bins_ct_only / max(n_bins_sampled, 1)
    print("")
    print("Coverage Map Summary")
    print("====================")
    print(f"Input CSV    : {csv_path}")
    print(f"Science frames: {n_science}  ({len(at_frames)} AT, {len(ct_frames)} CT)")
    print(f"Bin size     : {dlat}° x {dlon}°")
    print("-----------------------------------------")
    print(f"Bins sampled : {n_bins_sampled}")
    print(f"  AT only    : {n_bins_at_only}  ({pct_at:.0f}%)")
    print(f"  CT only    : {n_bins_ct_only}  ({pct_ct:.0f}%)")
    print(f"  Mixed      : {n_bins_mixed}  ({pct_mixed:.0f}%)")
    print("-----------------------------------------")
    print(f"Figures written to: {output_dir}")


if __name__ == "__main__":
    main()
