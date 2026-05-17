"""
diagnose_matplotlib.py
Run this FIRST before touching run_cal_pipeline.py.

    python diagnose_matplotlib.py

It will tell us exactly which backends are available and which one works.
"""
import sys
print(f"Python: {sys.executable}")
print(f"Version: {sys.version}")
print()

import matplotlib
print(f"matplotlib version : {matplotlib.__version__}")
print(f"matplotlib location: {matplotlib.__file__}")
print()

# ── 1. What backend is currently active? ─────────────────────────────
print(f"Current backend    : {matplotlib.get_backend()}")
print()

# ── 2. Which GUI backends are importable? ────────────────────────────
backends_to_try = ["TkAgg", "Qt5Agg", "QtAgg", "WXAgg", "Agg"]
print("Backend availability:")
for b in backends_to_try:
    try:
        import importlib
        importlib.import_module(f"matplotlib.backends.backend_{b.lower()}")
        print(f"  {b:12s}  IMPORTABLE")
    except ImportError as e:
        print(f"  {b:12s}  MISSING  ({e})")
print()

# ── 3. Try to actually open a window with each interactive backend ────
interactive = ["TkAgg", "Qt5Agg", "QtAgg", "WXAgg"]
print("Window open test (will try to show a blank figure for 1 second each):")
for b in interactive:
    try:
        matplotlib.use(b, force=True)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_title(f"Backend test: {b}")
        fig.canvas.draw()
        plt.pause(0.5)
        plt.close(fig)
        print(f"  {b:12s}  WORKS")
        # Use the first working one
        print(f"\n>>> USE THIS BACKEND: {b}")
        break
    except Exception as e:
        print(f"  {b:12s}  FAILED   ({type(e).__name__}: {e})")
else:
    print("\n>>> NO INTERACTIVE BACKEND WORKS on this machine.")
    print("    Will need to use Agg + os.startfile (save-and-open) approach.")
