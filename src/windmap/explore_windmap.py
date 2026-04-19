"""
explore_windmap.py
------------------
Interactive script: select a WindMap backend, then plot it in all four modes.

Usage (from anywhere inside the soc_sewell repository):
    python explore_windmap.py
    python scripts/explore_windmap.py
    python ../../explore_windmap.py

The script locates the repository root automatically by walking up the
directory tree from its own location until it finds a folder that contains
both a 'src/' subdirectory and a '.git' directory (the two markers that
uniquely identify the soc_sewell root).  It then inserts that root at the
front of sys.path so that 'from src.windmap import ...' works regardless
of where you invoke it from.

Spec reference: NB00_wind_map_2026-04-18.md
Depends on:     src/windmap  (NB00 implementation)
                src/constants
                numpy, matplotlib, cartopy
                hwm14  (optional — T3/T4 skipped if not installed)
"""

import sys
import os
import importlib
import importlib.util
from pathlib import Path

# On Windows, conda DLLs (OpenBLAS, Tcl/Tk, etc.) live in Library\bin
# inside the env.  VS Code sets the Python exe but not PATH, so those
# DLLs are invisible and numpy / matplotlib crash silently before any
# output.  Prepend the conda env directories to PATH now, before any
# scientific package is imported.
if sys.platform == "win32":
    _py = Path(sys.executable).parent          # e.g. …\envs\soc
    _conda_dirs = [
        str(_py),
        str(_py / "Library" / "bin"),
        str(_py / "Library" / "mingw-w64" / "bin"),
        str(_py / "Library" / "usr" / "bin"),
        str(_py / "Scripts"),
    ]
    _existing = os.environ.get("PATH", "").split(os.pathsep)
    _new_dirs = [d for d in _conda_dirs if d not in _existing and Path(d).is_dir()]
    if _new_dirs:
        os.environ["PATH"] = os.pathsep.join(_new_dirs) + os.pathsep + os.environ.get("PATH", "")

import numpy as np
# matplotlib imported lazily below, after the menu, so a backend crash
# does not prevent the text menu from appearing.

# ---------------------------------------------------------------------------
# 1.  Locate the soc_sewell repo root and insert it into sys.path
# ---------------------------------------------------------------------------

def _find_repo_root(start: Path) -> Path:
    """
    Walk up from `start` until we find a directory that looks like the
    soc_sewell repo root.  The root is identified by the simultaneous
    presence of:
        - a 'src/' subdirectory   (contains the pipeline source)
        - a '.git' directory      (git repository root marker)

    Falls back to the directory containing this script if the markers are
    never found (e.g. if the file has been copied outside the repo).
    """
    candidate = start.resolve()
    while True:
        has_src  = (candidate / 'src').is_dir()
        has_git  = (candidate / '.git').exists()
        if has_src and has_git:
            return candidate
        parent = candidate.parent
        if parent == candidate:
            # Reached filesystem root without finding the markers
            return start.resolve()
        candidate = parent


_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT  = _find_repo_root(_SCRIPT_DIR)

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# 2.  Import WindMap classes — graceful fallback if hwm14 is absent
# ---------------------------------------------------------------------------
try:
    from src.windmap import (
        UniformWindMap,
        AnalyticWindMap,
        HWM14WindMap,
        StormWindMap,
    )
except ModuleNotFoundError as exc:
    sys.exit(
        f"\nCould not import src.windmap: {exc}\n\n"
        f"Repo root detected as: {_REPO_ROOT}\n"
        "If that looks wrong, make sure this script lives somewhere inside\n"
        "the soc_sewell repository (which must contain both 'src/' and '.git/').\n"
    )

_hwm14_available = importlib.util.find_spec('hwm14') is not None

# ---------------------------------------------------------------------------
# 3.  Menu definition
# ---------------------------------------------------------------------------
MENU = [
    {
        'id':    'T1a',
        'label': 'T1 — Uniform  (100 m/s eastward, 0 m/s northward)',
        'build': lambda: UniformWindMap(v_zonal_ms=100.0, v_merid_ms=0.0),
        'needs_hwm14': False,
    },
    {
        'id':    'T1b',
        'label': 'T1 — Uniform  (150 m/s eastward, −75 m/s southward)',
        'build': lambda: UniformWindMap(v_zonal_ms=150.0, v_merid_ms=-75.0),
        'needs_hwm14': False,
    },
    {
        'id':    'T2a',
        'label': 'T2 — Analytic  sine-lat  (zonal jet + cross-equatorial flow)',
        'build': lambda: AnalyticWindMap(
            pattern='sine_lat', A_zonal_ms=200.0, A_merid_ms=100.0
        ),
        'needs_hwm14': False,
    },
    {
        'id':    'T2b',
        'label': 'T2 — Analytic  wave-4  (DE3 non-migrating tidal pattern)',
        'build': lambda: AnalyticWindMap(
            pattern='wave4', A_zonal_ms=150.0, A_merid_ms=75.0
        ),
        'needs_hwm14': False,
    },
    {
        'id':    'T3',
        'label': 'T3 — HWM14  quiet-time  (250 km, June solstice, UT 12:00)',
        'build': lambda: HWM14WindMap(
            alt_km=250.0, day_of_year=172, ut_hours=12.0, f107=150.0, ap=4
        ),
        'needs_hwm14': True,
    },
    {
        'id':    'T4',
        'label': 'T4 — Storm  (250 km, December solstice, UT 03:00, ap=80 / Kp≈6)',
        'build': lambda: StormWindMap(
            alt_km=250.0, day_of_year=355, ut_hours=3.0,
            f107=180.0, f107a=180.0, ap=80
        ),
        'needs_hwm14': True,
    },
]

# ---------------------------------------------------------------------------
# 4.  Print menu and get user selection
# ---------------------------------------------------------------------------
print()
print("=" * 65)
print("  WindCube SOC — WindMap Explorer")
print("  NB00 spec  |  HAO/NCAR")
print("=" * 65)
print(f"  Repo root : {_REPO_ROOT}")
print()
print("Available wind map backends:\n")

available = []
for i, entry in enumerate(MENU):
    if entry['needs_hwm14'] and not _hwm14_available:
        status = '  [unavailable — hwm14 not installed]'
    else:
        status = ''
    print(f"  [{i + 1}]  {entry['label']}{status}")
    if not (entry['needs_hwm14'] and not _hwm14_available):
        available.append(i)

print()
if not available:
    sys.exit("No wind map backends available. Install hwm14 to enable T3/T4.")

while True:
    raw = input(f"Select a wind map [1–{len(MENU)}]: ").strip()
    if not raw.isdigit():
        print("    Please enter a number.")
        continue
    choice = int(raw) - 1
    if choice not in available:
        if 0 <= choice < len(MENU):
            print("    That option requires hwm14. Choose a different backend.")
        else:
            print(f"    Please enter a number between 1 and {len(MENU)}.")
        continue
    break

entry = MENU[choice]
print(f"\nBuilding:  {entry['label']}")
print("(This may take a few seconds for HWM14/storm backends...)")

try:
    wm = entry['build']()
except Exception as exc:
    sys.exit(f"\nFailed to build wind map: {exc}")

print("Done.\n")

# ---------------------------------------------------------------------------
# 5.  Plot in all four modes
# ---------------------------------------------------------------------------
MODES = [
    ('separate',  'Two-panel U / V colour maps  +  quiver arrows'),
    ('vector',    'Single panel — speed colour  +  direction arrows'),
    ('stream',    'Single panel — streamlines coloured by speed'),
    ('magnitude', 'Single panel — speed magnitude  +  STM threshold contour'),
]

base_title = f"{entry['id']}  {entry['label'].split('—')[1].strip()}"

print("Plotting all four modes.")
print("Close each figure window to proceed to the next.\n")

import matplotlib
import matplotlib.pyplot as plt
_headless = matplotlib.get_backend().lower() == 'agg'
if _headless:
    print("NOTE: no interactive display — figures will be saved as PNG files.\n")

for mode, description in MODES:
    print(f"  [{mode:12s}]  {description}")
    save_path = f"{entry['id']}_{mode}.png" if _headless else None
    try:
        wm.plot(
            title=base_title,
            alt_km=250.0,
            subsample=8,
            mode=mode,
            save_path=save_path,
        )
        if save_path:
            print(f"             → saved to {save_path}")
    except Exception as exc:
        print(f"    WARNING: plot(mode='{mode}') failed — {type(exc).__name__}: {exc}")
        plt.close('all')
        continue

print("\nAll plots complete.")
