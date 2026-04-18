"""
strip_science_labels.py
-----------------------
Selects a folder, finds files whose names contain _science, _calibration, or _dark,
copies them into a new subfolder ("stripped") with those substrings removed.
"""

import os
import re
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox

# ── Substrings to strip (order matters: longest / most specific first) ──────
STRIP_PATTERNS = [
    "_calibration",
    "_science",
    "_dark",
    "_cal",
]

COMPILED = re.compile(
    "(" + "|".join(re.escape(p) for p in STRIP_PATTERNS) + ")",
    flags=re.IGNORECASE,
)

OUTPUT_SUBDIR = "stripped"


def clean_name(filename: str) -> str:
    """Return filename with all target substrings removed."""
    return COMPILED.sub("", filename)


def process_folder(src_folder: str) -> tuple[int, int, list[str], list[str]]:
    """
    Copy files from src_folder into src_folder/stripped with cleaned names.

    Returns
    -------
    (copied, skipped, warnings, skipped_files)
    """
    dst_folder = os.path.join(src_folder, OUTPUT_SUBDIR)
    os.makedirs(dst_folder, exist_ok=True)

    copied = skipped = 0
    warnings: list[str] = []
    skipped_files: list[str] = []

    entries = [
        e for e in os.scandir(src_folder)
        if e.is_file()           # files only
    ]

    for entry in entries:
        original = entry.name
        cleaned  = clean_name(original)

        if cleaned == original:
            skipped += 1
            skipped_files.append(original)
            continue                          # nothing to strip – skip

        dst_path = os.path.join(dst_folder, cleaned)

        if os.path.exists(dst_path):
            warnings.append(
                f"SKIP (collision): '{original}' → '{cleaned}' already exists."
            )
            continue

        shutil.copy2(entry.path, dst_path)
        copied += 1
        print(f"  {original}  →  {cleaned}")

    return copied, skipped, warnings, skipped_files


def main() -> None:
    # ── Hide the root Tk window ──────────────────────────────────────────────
    root = tk.Tk()
    root.withdraw()

    src_folder = filedialog.askdirectory(title="Select source folder")
    if not src_folder:
        print("No folder selected – exiting.")
        return

    print(f"\nSource : {src_folder}")
    print(f"Output : {os.path.join(src_folder, OUTPUT_SUBDIR)}\n")

    copied, skipped, warnings, skipped_files = process_folder(src_folder)

    # ── Summary ─────────────────────────────────────────────────────────────
    summary_lines = [
        f"Files copied  : {copied}",
        f"Files skipped (no match) : {skipped}",
    ]
    if skipped_files:
        summary_lines.append(f"\nSkipped files (no match):")
        summary_lines.extend(f"  {f}" for f in skipped_files)
    if warnings:
        summary_lines.append(f"\nWarnings ({len(warnings)}):")
        summary_lines.extend(f"  {w}" for w in warnings)

    summary = "\n".join(summary_lines)
    print("\n" + summary)

    messagebox.showinfo(
        "Done",
        f"Finished!\n\n{summary}\n\nOutput folder:\n"
        f"{os.path.join(src_folder, OUTPUT_SUBDIR)}",
    )


if __name__ == "__main__":
    main()