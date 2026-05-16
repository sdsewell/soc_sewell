"""
Smoke test: GEN01 v16 legacy mode (no TLE).
Monkey-patches input() and filedialog to run non-interactively.
"""
import sys, pathlib, tempfile, builtins
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import tkinter.filedialog as _fd

_out = tempfile.mkdtemp(prefix="gen01_legacy_")

# Responses in order: TLE prompt (N), t_start, duration, lat_min, lat_max,
# cadence, n_caldark, exp_sci, exp_cal, binning, cx_off, cy_off,
# altitude_km, h_target_km, rng_seed, wind_map choice, v_zonal, v_merid
_inputs = iter([
    "N",                    # TLE prompt
    "2027-01-01T00:00:00",  # t_start
    "",                     # altitude (default 510)
    "0.1",                  # duration_days
    "",                     # lat_min (-40)
    "",                     # lat_max (+40)
    "",                     # cadence (10 s)
    "",                     # n_caldark (5)
    "",                     # exp_sci (10 s)
    "",                     # exp_cal (120 s)
    "",                     # binning (2)
    "",                     # cx_offset (0)
    "",                     # cy_offset (0)
    "",                     # h_target_km (250)
    "",                     # rng_seed (42)
    "1",                    # wind map: uniform
    "",                     # v_zonal (100)
    "",                     # v_merid (0)
])

builtins._real_input = builtins.input
builtins.input = lambda prompt="": (print(prompt, end="", flush=True), next(_inputs, ""))[1]

# Patch folder dialog to return temp dir
_fd._real_askdir = _fd.askdirectory
_fd.askdirectory = lambda **kw: _out

print(f"Output dir: {_out}")
from src.processing.GEN01_synthesize_mission_dataset_2026_05_13 import main
main()
