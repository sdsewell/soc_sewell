"""
run_wind_map.py — Windows launcher for invert_wind_map.py.
Opens folder and file pickers then runs the batch inversion.
"""

import subprocess
import sys
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

DEFAULT_DIR = r"C:\Users\sewell\Documents\GitHub\data_synthetic\GEN1_null_wind_map\bin_frames"

def main():
    root = tk.Tk()
    root.withdraw()

    # Pick input folder
    input_folder = filedialog.askdirectory(
        title="Select folder containing *_science.bin files",
        initialdir=DEFAULT_DIR,
    )
    if not input_folder:
        print("No folder selected — exiting.")
        return

    # Pick GEN01 CSV
    csv_path = filedialog.askopenfilename(
        title="Select GEN01 CSV file (v_rel_ms column required)",
        initialdir=input_folder,
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    if not csv_path:
        print("No CSV selected — exiting.")
        return

    root.destroy()

    script = Path(__file__).parent / "invert_wind_map.py"

    cmd = [
        sys.executable, str(script),
        input_folder,
        "--v-rel-csv", csv_path,
        "--sigma-v", "10.0",
        "--verbose",
    ]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
