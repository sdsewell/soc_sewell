"""
Quick script: create a UniformWindMap and save a plot to this folder.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib
matplotlib.use("Agg")  # non-interactive backend so savefig works without a display
import matplotlib.pyplot as plt
import geopandas as gpd

from src.windmap.nb00_wind_map_2026_04_06 import UniformWindMap

# Load Natural Earth continent outlines via geodatasets
import geodatasets
_world = gpd.read_file(geodatasets.get_path("naturalearth.land"))

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "uniform_windmap.png")

wind_map = UniformWindMap(v_zonal_ms=100.0, v_merid_ms=0.0)

# Reproduce the base-class plot() logic so we can save the figure
import numpy as np

lat_grid = np.linspace(-89.5, 89.5, 180)
lon_grid = np.linspace(-179.5, 179.5, 360)
LON, LAT = np.meshgrid(lon_grid, lat_grid)

vz, vm = wind_map.sample_array(LAT.ravel(), LON.ravel())
vz = vz.reshape(180, 360)
vm = vm.reshape(180, 360)

lim = max(float(abs(vz).max()), float(abs(vm).max()), 1.0)

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
    _world.boundary.plot(ax=ax, linewidth=0.5, color="k")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_title(label)

fig.suptitle("UniformWindMap  |  v_zonal = 100 m/s, v_merid = 0 m/s  |  alt = 250 km")
plt.tight_layout()
fig.savefig(OUT_PATH, dpi=150)
print(f"Saved: {OUT_PATH}")
