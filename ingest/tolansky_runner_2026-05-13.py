"""
tolansky_runner_2026-05-13.py
=============================
Entry-point for the WindCube two-line neon calibration Tolansky analysis.

Supersedes:  src/processing/tolansky-2line.py   (renamed + updated import path)
             src/processing/tolansky-1line.py   (retired -- single-line airglow
             Tolansky is not a pipeline step; see S13 spec for rationale)

Loads a _peak_fits_r2.npy file produced by annular_reduction_2026-05-13.py,
runs the two-line analysis, prints the rectangular array, prints M05 priors,
and saves the diagnostic figure.
"""

import sys
import importlib.util
import pathlib
import tkinter as tk
from tkinter import filedialog

# ---------------------------------------------------------------------------
# Load canonical S13 implementation
# ---------------------------------------------------------------------------
_fpi_path = (pathlib.Path(__file__).resolve().parent.parent
             / "src" / "fpi" / "tolansky_2026-05-13.py")
_s13_spec = importlib.util.spec_from_file_location(
    "tolansky_unified", str(_fpi_path)
)
assert _s13_spec is not None, f"Cannot locate {_fpi_path}"
_s13_mod = importlib.util.module_from_spec(_s13_spec)
sys.modules["tolansky_unified"] = _s13_mod
assert _s13_spec.loader is not None
_s13_spec.loader.exec_module(_s13_mod)  # type: ignore[union-attr]

_run_tolansky            = _s13_mod.run_tolansky        # alias for run_tolansky_2line
_print_rectangular_array = _s13_mod.print_rectangular_array
_to_m05_priors           = _s13_mod.to_m05_priors
_plot_tolansky_result    = _s13_mod.plot_tolansky_result

# ---------------------------------------------------------------------------
# Locate the peaks file
# ---------------------------------------------------------------------------
if len(sys.argv) > 1:
    peaks_path = pathlib.Path(sys.argv[1])
else:
    root = tk.Tk()
    root.withdraw()
    _p = filedialog.askopenfilename(
        title="Select *_peak_fits_r2.npy",
        filetypes=[("NumPy array", "*.npy"), ("All files", "*.*")],
    )
    root.destroy()
    if not _p:
        print("No file selected -- exiting.")
        sys.exit(0)
    peaks_path = pathlib.Path(_p)

# ---------------------------------------------------------------------------
# Count peaks and prompt for number of pairs to fit
# ---------------------------------------------------------------------------
import numpy as _np
from tkinter import simpledialog as _simpledialog

_raw = _np.load(str(peaks_path))
_n_total = _raw.shape[0]
_n_pairs_max = _n_total // 2          # drop last peak if odd total
print(f"Peaks file    : {peaks_path.name}")
print(f"Total peaks   : {_n_total}  ->  max pairs = {_n_pairs_max}")
_root2 = tk.Tk()
_root2.withdraw()
_n_pairs = _simpledialog.askinteger(
    "Ring pairs",
    f"Found {_n_pairs_max} peak pairs ({_n_total} total peaks).\n"
    f"How many pairs should be used in the WLS fit?",
    initialvalue=_n_pairs_max,
    minvalue=2,
    maxvalue=_n_pairs_max,
    parent=_root2,
)
_root2.destroy()
if _n_pairs is None:
    print("Ring-pair count not set -- exiting.")
    sys.exit(0)
print(f"Using {_n_pairs} pairs per neon line")

# ---------------------------------------------------------------------------
# Run S13 analysis and print rectangular array
# ---------------------------------------------------------------------------
result = _run_tolansky(peaks_path, n_pairs=_n_pairs)
_print_rectangular_array(result)

# ---------------------------------------------------------------------------
# M05 priors handoff
# ---------------------------------------------------------------------------
priors = _to_m05_priors(result)
print("\nM05 priors:")
for k, v in priors.items():
    print(f"  {k}: {v}")

# ---------------------------------------------------------------------------
# Diagnostic figure
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt

_plot_tolansky_result(result)
fig_path = peaks_path.with_name(peaks_path.stem + "_tolansky_2line.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"  Figure saved -> {fig_path}")
plt.show()
