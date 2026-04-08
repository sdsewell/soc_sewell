"""
Z01_Validate_Calibration_Using_Real_Image.py

Standalone script to validate calibration using a real calibration image and a real dark image.
Follows the workflow in specs/Z01_Validate_Calibration_Using_Real_Image.md.

Assumes all required modules are available in src/ and that the script is run from the repo root.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Ensure repo root is on sys.path for src imports
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --- Import repo modules ---
from src.two_d_one_d_reduction import load_real_image, center_finder, annular_reduction, tolansky
import src.metadata as metadata_mod


# --- (b) Prompt user for file paths using file dialog ---
import tkinter as tk
from tkinter import filedialog

def get_file(title):
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title=title)
    root.destroy()
    return file_path

cal_img_path = get_file("Select real calibration image file")
dark_img_path = get_file("Select real dark image file")


# --- (b) Load images and extract metadata ---
cal_header, cal_img = load_real_image.load_raw(cal_img_path)
dark_header, dark_img = load_real_image.load_raw(dark_img_path)
cal_meta = load_real_image.parse_header(cal_header)
dark_meta = load_real_image.parse_header(dark_header)


# --- (c) Prepare metadata for display ---
cal_meta_row = cal_meta
dark_meta_row = dark_meta

# --- (d) Show raw images and metadata ---
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(cal_img, cmap='gray')
axes[0].set_title('Calibration Image')
axes[1].imshow(dark_img, cmap='gray')
axes[1].set_title('Dark Image')
plt.tight_layout()
plt.show()

meta_table = pd.DataFrame([cal_meta_row, dark_meta_row], index=['Calibration', 'Dark'])
print("\nMetadata (first row):\n", meta_table)


# --- (e) User selects center and ROI ---
def get_center_and_roi(img, default_size=216):
    plt.imshow(img, cmap='gray')
    plt.title('Click to select center of fringe pattern')
    center = plt.ginput(1)[0]  # (x, y)
    plt.close()
    print(f"Selected center: {center}")
    roi_size = 216  # Hard-coded default
    cx_seed, cy_seed = int(center[0]), int(center[1])
    return (cx_seed, cy_seed), roi_size

(center, roi_size) = get_center_and_roi(cal_img)
cx_seed, cy_seed = center


# --- (e) Extract ROI from both images ---
def extract_roi(img, center, size):
    x, y = int(center[0]), int(center[1])
    half = size // 2
    return img[y-half:y+half, x-half:x+half]


cal_roi = extract_roi(cal_img, (cx_seed, cy_seed), roi_size)
dark_roi = extract_roi(dark_img, (cx_seed, cy_seed), roi_size)

# After extracting ROI, reset seed to center of ROI
roi_cx_seed = roi_size // 2
roi_cy_seed = roi_size // 2


# --- (f) Show ROI images and histograms (now with dark-subtracted ROI) ---
dark_subtracted = cal_roi.astype(float) - dark_roi.astype(float)


fig, axes = plt.subplots(2, 3, figsize=(15, 8))
roi_title = f"Calibration ROI (cx_seed={cx_seed}, cy_seed={cy_seed})"
dark_title = f"Dark ROI (cx_seed={cx_seed}, cy_seed={cy_seed})"
sub_title = f"Dark-Subtracted ROI (cx_seed={cx_seed}, cy_seed={cy_seed})"
axes[0, 0].imshow(cal_roi, cmap='gray')
axes[0, 0].set_title(roi_title)
axes[0, 1].imshow(dark_roi, cmap='gray')
axes[0, 1].set_title(dark_title)
axes[0, 2].imshow(dark_subtracted, cmap='gray')
axes[0, 2].set_title(sub_title)
for i, data in enumerate([cal_roi, dark_roi, dark_subtracted]):
    axes[1, i].hist(data.ravel(), bins=50, color=['blue','red','green'][i], alpha=0.7)
    axes[1, i].set_xlim(0, 16383)
axes[1, 0].set_title('Calibration ROI Histogram')
axes[1, 1].set_title('Dark ROI Histogram')
axes[1, 2].set_title('Dark-Subtracted ROI Histogram')
plt.tight_layout()
plt.show()

# --- (g) Dark subtraction already done above ---




# --- (h) Center finding (intermediate diagnostic plot) ---

fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(dark_subtracted, cmap='gray')
ax.scatter([roi_cx_seed], [roi_cy_seed], color='orange', label='Seed center (ROI)', s=80)
ax.set_title(f'Seed Center for Coarse Grid: (cx_seed={roi_cx_seed}, cy_seed={roi_cy_seed})')
ax.legend()
plt.show()

# --- (h) Center finding (coarse and fine) ---
cx_fine, cy_fine, cost_min, best_cx, best_cy, best_cost = center_finder.azimuthal_variance_centre(dark_subtracted, roi_cx_seed, roi_cy_seed)
fine_result = center_finder.find_centre(dark_subtracted, cx_fine, cy_fine)


# --- (i) Plot center finding results (final center only) ---
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(dark_subtracted, cmap='gray')
ax.scatter([fine_result.cx], [fine_result.cy], color='red', label='Final center', s=80)
ax.set_title(f'Final Center: ({fine_result.cx:.2f}, {fine_result.cy:.2f})')
ax.legend()
plt.show()
print(f"Final center: ({fine_result.cx:.2f}, {fine_result.cy:.2f}), Uncertainty: {getattr(fine_result, 'uncertainty', 'N/A')}, Chi2: {getattr(fine_result, 'chi2', 'N/A')}")

# --- (j) Annular reduction and peak finding ---
profile, sem = annular_reduction.annular_reduce(dark_subtracted, fine_result.center)
peaks = annular_reduction.find_peaks(profile)

fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(np.arange(len(profile)), profile, yerr=sem, fmt='-o', label='Radial profile')
for pk in peaks:
    ax.annotate('', xy=(pk.position, profile[int(pk.position)]), xytext=(pk.position, profile[int(pk.position)]+sem[int(pk.position)]),
                arrowprops=dict(facecolor='black', shrink=0.05))
ax.set_title('Radial Profile with Peaks')
plt.show()

# Table of peaks
peak_table = pd.DataFrame([{**vars(pk), '2sigma': 2*pk.sigma} for pk in peaks])
print("\nPeak Fit Results:\n", peak_table)

# --- (k) Tolansky analysis ---
tol1 = tolansky.TolanskyAnalyser(profile, peaks, line='640nm')
tol2 = tolansky.TolanskyAnalyser(profile, peaks, line='638nm')
fig, ax = plt.subplots(figsize=(8, 5))
tol1.plot(ax=ax, label='640nm')
tol2.plot(ax=ax, label='638nm')
ax.legend()
plt.title('Tolansky 1-line Analysis')
plt.show()

# 2-line analysis
tol2line = tolansky.TwoLineAnalyser(profile, peaks)
fig, ax = plt.subplots(figsize=(8, 5))
tol2line.plot(ax=ax)
plt.title('Tolansky 2-line Analysis')
plt.show()

print(f"1-line epsilon (640nm): {tol1.epsilon}, (638nm): {tol2.epsilon}")
print(f"2-line focal length: {tol2line.focal_length}, etalon gap: {tol2line.etalon_gap}")
print(f"Uncertainties: {tol2line.uncertainties}")

# --- End of script ---
