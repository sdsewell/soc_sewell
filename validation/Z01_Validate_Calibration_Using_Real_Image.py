
# All imports at the top
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
from scipy.optimize import minimize



import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
from scipy.optimize import minimize

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





# --- (b) Hard-coded file paths for calibration and dark images ---
cal_img_path = r"C:/Users/sewell/Documents/GitHub/soc_sample_data/0_cal_120sexp_swapped.bin"
dark_img_path = r"C:/Users/sewell/Documents/GitHub/soc_sample_data/0_dark_120sexp_swapped.bin"

# --- (b) Load images and extract metadata ---
cal_header, cal_img = load_real_image.load_raw(cal_img_path)
dark_header, dark_img = load_real_image.load_raw(dark_img_path)
cal_meta = load_real_image.parse_header(cal_header)
dark_meta = load_real_image.parse_header(dark_header)


# --- (c) Prepare metadata for display ---
cal_meta_row = cal_meta
dark_meta_row = dark_meta



# --- (d) Show raw images and metadata, and select center ---
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(cal_img, cmap='gray')
axes[0].set_title('Raw Calibration Image')
axes[1].imshow(dark_img, cmap='gray')
axes[1].set_title('Raw Dark Image')
plt.tight_layout(rect=[0, 0.08, 1, 0.95])
# Add file names as text beneath the plots
fig.text(0.5, 0.02, f'Calibration: {cal_img_path}\nDark: {dark_img_path}', ha='center', va='bottom', fontsize=9, wrap=True)
print("\nMetadata (first row):\n", pd.DataFrame([cal_meta_row, dark_meta_row], index=['Calibration', 'Dark']))
print("\nClick on the center of the fringe pattern in the Raw Calibration Image (left panel)...")
plt.sca(axes[0])
center = plt.ginput(1)[0]  # (x, y)
plt.close(fig)
print(f"Selected center: {center}")
roi_size = 216  # Hard-coded default
cx_seed, cy_seed = int(center[0]), int(center[1])


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







# --- (h) Center finding (coarse grid diagnostic) ---
def variance_cost(xy):
    r_min_sq = 5.0 ** 2
    r_max_sq = (dark_subtracted.shape[0] // 2 - 10) ** 2
    return center_finder._variance_cost(xy[0], xy[1], dark_subtracted, r_min_sq, r_max_sq, 250)

var_search_px = 15.0
grid_step = max(2.0, var_search_px / 8.0)
offsets = np.arange(-var_search_px, var_search_px + grid_step * 0.5, grid_step)
coarse_points = []
for dy in offsets:
    for dx in offsets:
        cx = roi_cx_seed + dx
        cy = roi_cy_seed + dy
        cost = variance_cost([cx, cy])
        coarse_points.append((cx, cy, cost))
coarse_points = np.array(coarse_points)




# --- (h) Center finding (Nelder-Mead path diagnostic) ---
from scipy.optimize import minimize
nm_path = []
def nm_callback(xk):
    nm_path.append((xk[0], xk[1], variance_cost(xk)))

best_idx = np.argmin(coarse_points[:,2])
best_cx = coarse_points[best_idx,0]
best_cy = coarse_points[best_idx,1]
x0 = np.array([best_cx, best_cy])
simplex_r = grid_step + 0.5
initial_simplex = np.array([
    x0,
    x0 + [simplex_r, 0.0],
    x0 + [0.0, simplex_r],
])


result = minimize(
    variance_cost, x0,
    method="Nelder-Mead",
    options={
        "initial_simplex": initial_simplex,
        "xatol": 0.02,
        "fatol": 1.0,
        "maxiter": 500,
    },
    callback=nm_callback
)
cx_fine, cy_fine = result.x



# --- (h+i) Combined plot: coarse grid/Nelder-Mead and final center ---
# (Moved after fine_result is defined)



# --- (h) Fine result object for downstream ---
class FineResult:

    def __init__(self, cx, cy, sigma_cx=None, sigma_cy=None):
        self.cx = cx
        self.cy = cy
        self.sigma_cx = sigma_cx
        self.sigma_cy = sigma_cy
    @property
    def center(self):
        return (self.cx, self.cy)

# Robust numerical Hessian and uncertainty estimation at the minimum
def numerical_hessian(f, x, eps=1e-2):
    """Numerically estimate the Hessian matrix of f at x using central differences."""
    x = np.asarray(x, dtype=float)
    n = x.size
    hess = np.zeros((n, n))
    fx = f(x)
    for i in range(n):
        x_ip = x.copy(); x_ip[i] += eps
        x_im = x.copy(); x_im[i] -= eps
        f_ip = f(x_ip)
        f_im = f(x_im)
        hess[i, i] = (f_ip - 2*fx + f_im) / (eps**2)
        for j in range(i+1, n):
            x_ipp = x.copy(); x_ipp[i] += eps; x_ipp[j] += eps
            x_ipm = x.copy(); x_ipm[i] += eps; x_ipm[j] -= eps
            x_imp = x.copy(); x_imp[i] -= eps; x_imp[j] += eps
            x_imm = x.copy(); x_imm[i] -= eps; x_imm[j] -= eps
            f_ipp = f(x_ipp)
            f_ipm = f(x_ipm)
            f_imp = f(x_imp)
            f_imm = f(x_imm)
            hess_ij = (f_ipp - f_ipm - f_imp + f_imm) / (4 * eps**2)
            hess[i, j] = hess[j, i] = hess_ij
    return hess

try:
    hess = numerical_hessian(variance_cost, [cx_fine, cy_fine])
    cov = np.linalg.inv(hess)
    sigma_cx = np.sqrt(np.abs(cov[0,0]))
    sigma_cy = np.sqrt(np.abs(cov[1,1]))
except Exception as e:
    sigma_cx = sigma_cy = None


fine_result = FineResult(cx_fine, cy_fine, sigma_cx, sigma_cy)

# --- (i) Combined plot: coarse grid search and final center on ROI ---

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Left: Coarse grid search cost landscape
sc = axes[0].scatter(coarse_points[:,0], coarse_points[:,1], c=coarse_points[:,2], cmap='viridis', s=40)
axes[0].scatter([best_cx], [best_cy], color='red', marker='x', s=100, label='Coarse Min')
# Add starting position marker
axes[0].scatter([roi_cx_seed], [roi_cy_seed], color='orange', marker='*', s=120, label='Start (Seed)')
axes[0].set_title('Coarse Grid Search (cost)')
axes[0].set_xlabel('cx')
axes[0].set_ylabel('cy')
cbar = fig.colorbar(sc, ax=axes[0], label='Cost')
axes[0].legend()
# Add subtitle with seed and algorithm
axes[0].text(0.5, -0.18, f"Seed: cx={roi_cx_seed}, cy={roi_cy_seed}\nAlgorithm: Grid + Nelder-Mead",
             ha='center', va='top', transform=axes[0].transAxes, fontsize=10)

# Right: ROI with fine center and 2-sigma ellipse
axes[1].imshow(dark_subtracted, cmap='gray')
axes[1].scatter([fine_result.cx], [fine_result.cy], color='red', marker='o', s=60, label='Fine Center')

# Draw 2-sigma uncertainty ellipse if available
from matplotlib.patches import Ellipse
if fine_result.sigma_cx is not None and fine_result.sigma_cy is not None:
    ellipse = Ellipse((fine_result.cx, fine_result.cy),
                     width=2*2*fine_result.sigma_cx, height=2*2*fine_result.sigma_cy,
                     edgecolor='red', facecolor='none', linestyle='--', linewidth=2, label='2-sigma')
    axes[1].add_patch(ellipse)

# Title with ± notation and 2 significant figures
if fine_result.sigma_cx is not None and fine_result.sigma_cy is not None:
    cx_unc = f"±{2*fine_result.sigma_cx:.2g}"
    cy_unc = f"±{2*fine_result.sigma_cy:.2g}"
    title_str = (f"Final Center: ({fine_result.cx:.2f}, {fine_result.cy:.2f})\n"
                 f"2σ: cx {cx_unc}, cy {cy_unc}")
else:
    title_str = f"Final Center: ({fine_result.cx:.2f}, {fine_result.cy:.2f})"
axes[1].set_title(title_str)
axes[1].legend()

plt.tight_layout()
plt.show()

print(f"Final center: ({fine_result.cx:.2f}, {fine_result.cy:.2f}), "
    f"2-sigma uncertainty: cx={2*fine_result.sigma_cx if fine_result.sigma_cx is not None else 'N/A'}, "
    f"cy={2*fine_result.sigma_cy if fine_result.sigma_cy is not None else 'N/A'}, "
    f"Chi2: {getattr(fine_result, 'chi2', 'N/A')}")

# --- (j) Annular reduction and peak finding ---

fringe_result = annular_reduction.annular_reduce(dark_subtracted, fine_result.cx, fine_result.cy, fine_result.sigma_cx, fine_result.sigma_cy)
profile = fringe_result.profile
sem = fringe_result.sigma_profile
peak_fits = fringe_result.peak_fits

# Only show up to 20 peaks for the table
peak_fits_20 = peak_fits[:20]

import matplotlib.gridspec as gridspec
fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])

# Top: Radial profile with peaks (smaller markers)
ax1 = fig.add_subplot(gs[0])
ax1.errorbar(np.arange(len(profile)), profile, yerr=sem, fmt='-o', label='Radial profile', markersize=3)
for pk in [pf.peak_idx for pf in peak_fits_20]:
    ax1.annotate('', xy=(pk, profile[int(pk)]), xytext=(pk, profile[int(pk)]+sem[int(pk)]),
                arrowprops=dict(facecolor='black', shrink=0.05))
ax1.set_title('Radial Profile with Peaks')
ax1.set_xlabel('Bin Index')
ax1.set_ylabel('Intensity (ADU)')
ax1.legend()

# Add center annotation between plots (just above the table)

# Move the final center annotation below the table, at the bottom of the figure
center_str = f"Final center (Nelder-Mead): ({fine_result.cx:.2f}, {fine_result.cy:.2f})"
fig.text(0.5, 0.01, center_str, ha='center', va='bottom', fontsize=11,
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))


# Bottom: Table of peak fit results (first 20), with alternating line labels and no title
import matplotlib.table as mtable
ax2 = fig.add_subplot(gs[1])
ax2.axis('off')

if peak_fits_20:
    table_data = []
    col_labels = [
        '\u2013'.join(['**Peak #**', '']),  # workaround for bold
        '**Line**',
        '**r_raw_px**',
        '**profile_raw**',
        '**r_fit_px**',
        '**2σ_r_fit_px**',
        '**amplitude_adu**',
        '**width_px**',
        '**fit_ok**'
    ]
    for i, pf in enumerate(peak_fits_20):
        two_sigma = pf.sigma_r_fit_px * 2 if pf.sigma_r_fit_px is not None and not np.isnan(pf.sigma_r_fit_px) else np.nan
        two_sigma_str = f"±{two_sigma:.2g}" if not np.isnan(two_sigma) else 'N/A'
        line_label = '640.2248nm' if (i % 2 == 0) else '638.2991nm'
        table_data.append([
            i+1,
            line_label,
            f"{pf.r_raw_px:.2f}",
            f"{pf.profile_raw:.1f}",
            f"{pf.r_fit_px:.2f}",
            two_sigma_str,
            f"{pf.amplitude_adu:.1f}",
            f"{pf.width_px:.2f}",
            pf.fit_ok
        ])
    table = ax2.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    # Bold font for column headings
    for (key, cell) in table.get_celld().items():
        if key[0] == 0:
            cell.set_text_props(weight='bold')
else:
    ax2.text(0.5, 0.5, 'No peaks found', ha='center', va='center', fontsize=12)

plt.tight_layout()
plt.show()

# --- (k) Tolansky analysis ---

# Prepare 10 peaks for each 1-line Tolansky analysis
peak_fits_640 = [pf for i, pf in enumerate(fringe_result.peak_fits) if i % 2 == 0][:10]
peak_fits_638 = [pf for i, pf in enumerate(fringe_result.peak_fits) if i % 2 == 1][:10]

def tolansky_arrays(peak_fits):
    p = np.arange(1, len(peak_fits)+1)
    r = np.array([pf.r_fit_px for pf in peak_fits])
    sigma_r = np.array([pf.sigma_r_fit_px if pf.sigma_r_fit_px is not None and not np.isnan(pf.sigma_r_fit_px) else 1.0 for pf in peak_fits])
    return p, r, sigma_r

p_640, r_640, sigma_r_640 = tolansky_arrays(peak_fits_640)
p_638, r_638, sigma_r_638 = tolansky_arrays(peak_fits_638)


tol1 = tolansky.TolanskyAnalyser(p_640, r_640, sigma_r_640, r_unit="px", lam_nm=640.2248, n=1.0, f=None, d=None)
tol2 = tolansky.TolanskyAnalyser(p_638, r_638, sigma_r_638, r_unit="px", lam_nm=638.2991, n=1.0, f=None, d=None)
tol1.run()
tol2.run()

# Plot: peak index (y) vs r^2 (x) for each 1-line analysis


fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(tol1.p, tol1.r**2, yerr=2*tol1.r*tol1.sigma_r, fmt='o', label='640.2248nm', color='blue')
ax.errorbar(tol2.p, tol2.r**2, yerr=2*tol2.r*tol2.sigma_r, fmt='o', label='638.2991nm', color='orange')

pfit_638 = np.linspace(min(tol2.p)-0.5, max(tol2.p)+0.5, 100)
# Draw best fit lines (dashed, showing intercept, extended to x-axis)

# Extend fit lines to ensure x-intercept is visible in [0,1] axes
def fit_line_xrange(slope, intercept, p_min, p_max):
    x0 = -intercept / slope if slope != 0 else p_min
    # Always include x0 and [p_min-1, p_max+1]
    x_vals = np.array([x0, p_min-1, p_max+1])
    x_start = np.min(x_vals) - 1
    x_end = np.max(x_vals) + 1
    return np.linspace(x_start, x_end, 300)

pfit_640 = fit_line_xrange(tol1.result.slope, tol1.result.intercept, min(tol1.p), max(tol1.p))
fit_640 = tol1.result.slope * pfit_640 + tol1.result.intercept
ax.plot(pfit_640, fit_640, '--', color='blue', linewidth=1, alpha=0.8, label='Fit 640.2248nm')

pfit_638 = fit_line_xrange(tol2.result.slope, tol2.result.intercept, min(tol2.p), max(tol2.p))
fit_638 = tol2.result.slope * pfit_638 + tol2.result.intercept
ax.plot(pfit_638, fit_638, '--', color='orange', linewidth=1, alpha=0.8, label='Fit 638.2991nm')

ax.set_xlabel('Peak Index (p)')
ax.set_ylabel(r'$r^2$ [px$^2$]')
ax.set_title('Tolansky 1-line Analysis: $r^2$ vs Peak Index')
ax.legend()

# Annotate epsilon values and uncertainties
eps_640 = tol1.result.epsilon
eps_640_unc = tol1.result.sigma_epsilon
eps_638 = tol2.result.epsilon
eps_638_unc = tol2.result.sigma_epsilon


textstr = (f"ε₆₄₀ = {eps_640:.4f} ± {eps_640_unc:.4f}  |  ε₆₃₈ = {eps_638:.4f} ± {eps_638_unc:.4f}")
ax.text(0, 0, textstr, transform=ax.transAxes, fontsize=11,
    verticalalignment='bottom', horizontalalignment='left',
    bbox=dict(boxstyle='round,pad=0', facecolor='white', alpha=0.7),
    clip_on=True)

plt.tight_layout()
plt.show()




# 2-line analysis (unchanged)
tol2line = tolansky.TwoLineAnalyser(tol1, tol2, lam1_nm=640.2248, lam2_nm=638.2991, d_prior=0)
tol2line.run()

# Print 1-line epsilons and uncertainties
print(f"1-line epsilon (640.2248nm): {tol1.result.epsilon:.5f} ± {tol1.result.sigma_epsilon:.5f}, (638.2991nm): {tol2.result.epsilon:.5f} ± {tol2.result.sigma_epsilon:.5f}")

if hasattr(tol2line, 'result') and tol2line.result is not None:
    res = tol2line.result
    print(f"2-line recovered d: {res.d:.6g} ± {res.sigma_d:.4g} [{res.r_unit}]")
    print(f"2-line recovered f: {res.f:.6g} ± {res.sigma_f:.4g} [{res.r_unit}]")
    print(f"2-line eps1: {res.eps1:.7f} ± {res.sigma_eps1:.7f}, eps2: {res.eps2:.7f} ± {res.sigma_eps2:.7f}")
    print(f"2-line delta_eps: {res.delta_eps:+.7f} ± {res.sigma_delta_eps:.7f}")

    # Diagnostics: print before/after, print result fields, catch exceptions
    print("Calling tol2line.plot_joint()...")
    import pprint
    pprint.pprint({
        'd': res.d, 'sigma_d': res.sigma_d, 'f': res.f, 'sigma_f': res.sigma_f,
        'p1': res.p1, 'r1_sq': res.r1_sq, 'sr1_sq': res.sr1_sq, 'pred1': res.pred1,
        'p2': res.p2, 'r2_sq': res.r2_sq, 'sr2_sq': res.sr2_sq, 'pred2': res.pred2
    })
    try:
        fig_joint = tol2line.plot_joint()
        print("plot_joint() completed.")
        import matplotlib.pyplot as plt
        plt.show()
    except Exception as e:
        print(f"Exception in plot_joint: {e}")

# --- End of script ---
