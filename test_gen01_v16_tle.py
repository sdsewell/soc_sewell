"""
Smoke test: GEN01 v16 TLE mode.
Monkey-patches input() and filedialog to run non-interactively.
"""
import sys, pathlib, tempfile, builtins
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import tkinter.filedialog as _fd

_out = tempfile.mkdtemp(prefix="gen01_tle_")
_tle_path = str(pathlib.Path(__file__).parent / "test_windcube.tle")

# Responses in order: TLE prompt (y), then after TLE dialog:
# duration, lat_min, lat_max, cadence, n_caldark, exp_sci, exp_cal,
# binning, cx_off, cy_off, h_target_km, rng_seed, wind map, v_zonal, v_merid
_inputs = iter([
    "y",    # TLE prompt
    # (no t_start or altitude prompts in TLE mode)
    "0.1",  # duration_days
    "",     # lat_min (-40)
    "",     # lat_max (+40)
    "",     # cadence (10 s)
    "",     # n_caldark (5)
    "",     # exp_sci (10 s)
    "",     # exp_cal (120 s)
    "",     # binning (2)
    "",     # cx_offset (0)
    "",     # cy_offset (0)
    "",     # h_target_km (250)
    "",     # rng_seed (42)
    "1",    # wind map: uniform
    "",     # v_zonal (100)
    "",     # v_merid (0)
])

builtins.input = lambda prompt="": (print(prompt, end="", flush=True), next(_inputs, ""))[1]

# Patch filedialog: folder -> temp out dir, file -> TLE path
_fd.askdirectory = lambda **kw: _out
_fd.askopenfilename = lambda **kw: _tle_path

print(f"Output dir: {_out}")
print(f"TLE file  : {_tle_path}")
from src.processing.GEN01_synthesize_mission_dataset_2026_05_13 import main
main()
