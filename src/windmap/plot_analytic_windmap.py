"""
Quick script: create a DE3-like AnalyticWindMap (wave4 pattern) and save a plot.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import geodatasets

from src.windmap.nb00_wind_map_2026_04_06 import AnalyticWindMap

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "analytic_windmap_de3.png")

# DE3-like wave: zonal wavenumber 4, eastward-propagating tidal structure
wind_map = AnalyticWindMap(
    pattern="wave4",
    A_zonal_ms=150.0,
    A_merid_ms=75.0,
    phase_rad=0.0,
)

# Evaluate on the full global grid
lat_grid = np.linspace(-89.5, 89.5, 180)
lon_grid = np.linspace(-179.5, 179.5, 360)
LON, LAT = np.meshgrid(lon_grid, lat_grid)

vz, vm = wind_map.sample_array(LAT.ravel(), LON.ravel())
vz = vz.reshape(180, 360)
vm = vm.reshape(180, 360)

lim = max(float(abs(vz).max()), float(abs(vm).max()), 1.0)

# Continent outlines (downloaded once, then cached)
world = gpd.read_file(geodatasets.get_path("naturalearth.land"))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, data, label in zip(axes, [vz, vm], ["v_zonal (m/s)", "v_merid (m/s)"]):
    im = ax.imshow(
        data,
        origin="lower",
        extent=[-180, 180, -90, 90],
        aspect="auto",
        vmin=-lim,
        vmax=lim,
        cmap="RdBu_r",
    )
    plt.colorbar(im, ax=ax, label=label)
    world.boundary.plot(ax=ax, linewidth=0.5, color="k")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_title(label)

fig.suptitle(
    "AnalyticWindMap — DE3-like wave4  |  "
    r"$A_z$ = 150 m/s, $A_m$ = 75 m/s  |  alt = 250 km"
)
plt.tight_layout()
fig.savefig(OUT_PATH, dpi=150)
print(f"Saved: {OUT_PATH}")
