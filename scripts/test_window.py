"""
test_window.py  — minimal matplotlib window test.
Run from repo root:
    & c:\Users\sewell\.conda\envs\windcube\python.exe test_window.py
"""
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

print("Opening figure window...")
print("(Check all open windows and taskbar — it may be behind VS Code)")

fig, ax = plt.subplots(figsize=(6, 4))
ax.set_title("TEST — close this window to continue")
ax.text(0.5, 0.5, "If you can see this,\nTkAgg is working.",
        ha="center", va="center", fontsize=16, transform=ax.transAxes)

print("Calling plt.show(block=True) now...")
plt.show(block=True)
print("Window was closed — TkAgg blocking works.")
