"""Open an .npy file via Windows dialog and display array headings/structure."""

import tkinter as tk
from tkinter import filedialog
import numpy as np
from pathlib import Path


def open_npy_file() -> Path | None:
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Select .npy file",
        filetypes=[("NumPy array", "*.npy"), ("All files", "*.*")],
    )
    root.destroy()
    return Path(path) if path else None


def describe_array(arr: np.ndarray, path: Path) -> None:
    print(f"\nFile : {path.name}")
    print(f"Shape: {arr.shape}")
    print(f"Dtype: {arr.dtype}")

    if arr.dtype.names:
        print(f"\nFields ({len(arr.dtype.names)}):")
        for name in arr.dtype.names:
            sub = arr.dtype[name]
            print(f"  {name:<30} {sub}")
    else:
        print("\nNo named fields (not a structured array).")
        if arr.ndim == 2:
            print(f"Columns: {arr.shape[1]}  (no header names stored in .npy)")


def main() -> None:
    path = open_npy_file()
    if path is None:
        print("No file selected.")
        return

    arr = np.load(path, allow_pickle=True)

    # np.load on a .npy with allow_pickle may return an object array wrapping a dict
    if arr.ndim == 0:
        obj = arr.item()
        if isinstance(obj, dict):
            print(f"\nFile : {path.name}")
            print("Content: dict-like object array")
            print(f"\nKeys ({len(obj)}):")
            for k in obj:
                print(f"  {k}")
            return

    describe_array(arr, path)


if __name__ == "__main__":
    main()
