"""
Plot the ground track of the first N orbits from a G01 synthetic metadata CSV.

Usage:
    python src/utilities/plot_ground_track.py

Map rendering priority:
  1. cartopy  (best — Mercator projection)
  2. geopandas naturalearth_lowres  (good — flat lat/lon)
  3. plain matplotlib  (fallback — ocean colour only)
"""

import sys
import pathlib
import tkinter as tk
from tkinter import filedialog
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Any

if TYPE_CHECKING:
    import cartopy.crs as ccrs          # noqa: F401
    import cartopy.feature as cfeature  # noqa: F401

# ---------------------------------------------------------------------------
# Optional dependency probing (done once at import time)
# ---------------------------------------------------------------------------
try:
    import cartopy.crs as ccrs          # type: ignore[import-untyped]
    import cartopy.feature as cfeature  # type: ignore[import-untyped]
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

try:
    import geopandas as gpd  # type: ignore[import-untyped]
    HAS_GPD = True
except ImportError:
    gpd = None  # type: ignore[assignment]
    HAS_GPD = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_ORBITS = 2

WGS84_A_M        = 6_378_137.0
EARTH_GRAV_PARAM = 3.986004418e14   # m^3/s^2

ORBIT_COLOURS = ["gold", "deepskyblue", "lime", "hotpink", "orange"]

OCEAN_COLOUR = "#1a6e9e"
LAND_COLOUR  = "#4a7c3f"


# ---------------------------------------------------------------------------
# File picker
# ---------------------------------------------------------------------------
def pick_csv() -> pathlib.Path:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Select G01 synthetic metadata CSV",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    root.destroy()
    if not path:
        print("No file selected. Exiting.")
        sys.exit(0)
    return pathlib.Path(path)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def orbital_period_s(alt_m: float) -> float:
    a = WGS84_A_M + alt_m
    return 2.0 * np.pi * np.sqrt(a**3 / EARTH_GRAV_PARAM)


def load_and_prep(csv_path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns from {csv_path.name}")
    print(f"Columns: {list(df.columns)}")

    df["sc_lat_deg"] = np.degrees(df["spacecraft_latitude"])
    df["sc_lon_deg"] = np.degrees(df["spacecraft_longitude"])
    df["elapsed_s"]  = (df["lua_timestamp"] - df["lua_timestamp"].iloc[0]) / 1000.0

    mean_alt_m = df["spacecraft_altitude"].median()
    T_s = orbital_period_s(mean_alt_m)
    df["orbit_idx"] = (df["elapsed_s"] // T_s).astype(int)

    print(f"  S/C altitude  : {mean_alt_m/1e3:.1f} km")
    print(f"  Orbital period: {T_s:.1f} s  ({T_s/60:.2f} min)")
    print(f"  Total orbits  : {df['orbit_idx'].max() + 1}")

    if "tp_lat_deg" in df.columns:
        df["is_science"] = df["tp_lat_deg"].notna()
    else:
        print("  Note: 'tp_lat_deg' not found — frame-type colouring unavailable.")
        df["is_science"] = False
        df["tp_lat_deg"] = np.nan
        df["tp_lon_deg"] = np.nan

    return df


def extract_n_orbits(df: pd.DataFrame, n: int) -> pd.DataFrame:
    subset = df[df["orbit_idx"] < n].copy()
    for i in range(n):
        print(f"  Orbit {i + 1} rows: {(subset['orbit_idx'] == i).sum()}")
    return subset


# ---------------------------------------------------------------------------
# Map background helpers
# ---------------------------------------------------------------------------
def _draw_cartopy_background(ax: "Axes") -> None:  # type: ignore[type-arg]
    ax.set_global()  # type: ignore[attr-defined]
    ax.add_feature(cfeature.OCEAN,     facecolor=OCEAN_COLOUR, edgecolor="none", zorder=0)  # type: ignore[attr-defined]
    ax.add_feature(cfeature.LAND,      facecolor=LAND_COLOUR,  edgecolor="none", zorder=1)  # type: ignore[attr-defined]
    ax.add_feature(cfeature.LAKES,     facecolor=OCEAN_COLOUR, edgecolor="none", zorder=2)  # type: ignore[attr-defined]
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="white", alpha=0.8, zorder=3)  # type: ignore[attr-defined]
    ax.add_feature(cfeature.BORDERS,   linewidth=0.3, edgecolor="white",  # type: ignore[attr-defined]
                   linestyle=":", alpha=0.5, zorder=3)
    ax.gridlines(draw_labels=True, linewidth=0.4, color="white",  # type: ignore[attr-defined]
                 alpha=0.5, linestyle="--", zorder=3)


def _draw_geopandas_background(ax: Axes) -> None:
    assert gpd is not None
    from geodatasets import get_path  # type: ignore[import-untyped]
    world = gpd.read_file(get_path("naturalearth.land"))
    world.plot(ax=ax, color=LAND_COLOUR, edgecolor="white",
               linewidth=0.4, zorder=1)
    ax.set_facecolor(OCEAN_COLOUR)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-85, 85)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.grid(linewidth=0.3, alpha=0.3, color="white", zorder=0)


def _draw_plain_background(ax: Axes) -> None:
    ax.set_facecolor(OCEAN_COLOUR)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-85, 85)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.axhline(0, color="white", linewidth=0.5, linestyle="--", alpha=0.4)
    ax.grid(linewidth=0.3, alpha=0.3, color="white")


# ---------------------------------------------------------------------------
# Track plotting (shared logic)
# ---------------------------------------------------------------------------
def _draw_tracks(ax: Axes, data: pd.DataFrame, n_orbits: int,
                 transform: Any | None = None) -> None:
    plot_kw: dict = {"zorder": 4}
    scat_kw: dict = {"zorder": 5}
    if transform is not None:
        plot_kw["transform"] = transform
        scat_kw["transform"] = transform

    for i in range(n_orbits):
        orb = data[data["orbit_idx"] == i]
        if orb.empty:
            continue
        colour = ORBIT_COLOURS[i % len(ORBIT_COLOURS)]
        label  = f"Orbit {i + 1} ({'along' if i % 2 == 0 else 'cross'}-track)"

        ax.plot(orb["sc_lon_deg"], orb["sc_lat_deg"],
                color=colour, linewidth=1.4, alpha=0.9, label=label, **plot_kw)

        sci = orb[orb["is_science"]]
        if not sci.empty:
            ax.scatter(sci["sc_lon_deg"], sci["sc_lat_deg"],
                       c=colour, s=14, edgecolors="white",
                       linewidths=0.3, **scat_kw)

        if sci["tp_lat_deg"].notna().any():
            ax.scatter(sci["tp_lon_deg"], sci["tp_lat_deg"],
                       c=colour, s=8, marker="x", **scat_kw)

    first = data[data["orbit_idx"] == 0]
    ax.plot(first["sc_lon_deg"].iloc[0], first["sc_lat_deg"].iloc[0],
            "w*", ms=12, zorder=7, label="Start", **({} if transform is None
                                                      else {"transform": transform}))


# ---------------------------------------------------------------------------
# Main plot entry point
# ---------------------------------------------------------------------------
def plot_ground_track(data: pd.DataFrame, csv_path: pathlib.Path,
                      n_orbits: int) -> None:
    fig = plt.figure(figsize=(16, 8))
    fig.patch.set_facecolor("#111111")

    if HAS_CARTOPY:
        print("  Using cartopy (Mercator projection).")
        ax = fig.add_subplot(1, 1, 1,
                             projection=ccrs.Mercator(central_longitude=0))  # type: ignore[attr-defined]
        _draw_cartopy_background(ax)
        _draw_tracks(ax, data, n_orbits,
                     transform=ccrs.PlateCarree())  # type: ignore[attr-defined]
    elif HAS_GPD:
        print("  cartopy not found — using geopandas Natural Earth.")
        ax = fig.add_subplot(1, 1, 1)
        _draw_geopandas_background(ax)
        _draw_tracks(ax, data, n_orbits)
    else:
        print("  cartopy and geopandas not found — plain background only.")
        ax = fig.add_subplot(1, 1, 1)
        _draw_plain_background(ax)
        _draw_tracks(ax, data, n_orbits)

    n_sci  = int(data["is_science"].sum())
    has_tp = data["tp_lat_deg"].notna().any()
    type_str = (f"{n_sci} science obs" if has_tp
                else f"{len(data)} epochs (frame type not available)")
    ax.set_title(
        f"G01 Ground Track — Orbits 1–{n_orbits} — {csv_path.name}\n{type_str}",
        fontsize=10, color="white", pad=8,
    )
    ax.legend(loc="lower left", fontsize=8, framealpha=0.7,
              facecolor="#222222", labelcolor="white")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    csv_path = pick_csv()
    df   = load_and_prep(csv_path)
    data = extract_n_orbits(df, N_ORBITS)
    plot_ground_track(data, csv_path, N_ORBITS)


if __name__ == "__main__":
    main()
