"""
H06 diagnostic script — finds the eps_OI_exp bug.
Run from repo root with the venv python.
"""
import sys, pathlib, numpy as np
sys.path.insert(0, '.')

# ── 1. Print the key sections of the committed H06 source ─────────────────
print("=" * 60)
print("COMMITTED H06 SOURCE — key sections")
print("=" * 60)
lines = pathlib.Path('src/processing/H06_airglow_inversion_2026_05_14.py').read_text(encoding='utf-8').split('\n')

# Find run_airglow_inversion
rai_start = next(i for i, l in enumerate(lines) if 'def run_airglow_inversion' in l)
print(f"\ndef run_airglow_inversion starts at line {rai_start+1}")
print("Lines around _lambda_c_scan call:")
for i, line in enumerate(lines[rai_start:rai_start+120], start=rai_start+1):
    if any(kw in line for kw in ['_lambda_c_scan', 'lc_seed', 'eps_OI', 'N_int_OI', 'fsr_oi', 't_m']):
        print(f"  {i}: {line}")

# ── 2. Print the full _lambda_c_scan seed block ───────────────────────────
scan_start = next(i for i, l in enumerate(lines) if 'def _lambda_c_scan' in l)
print(f"\n\ndef _lambda_c_scan starts at line {scan_start+1}")
print("Seed computation block:")
for i, line in enumerate(lines[scan_start:scan_start+60], start=scan_start+1):
    if any(kw in line for kw in ['N_int_OI', 'eps_OI', 'lc_seed', 'v_los_prior', 'cal.t_m', 'fsr']):
        print(f"  {i}: {line}")

# ── 3. Verify math: what eps_OI_exp should be for the master cal t_m ──────
print("\n\n" + "=" * 60)
print("MATH CHECK")
print("=" * 60)
OI = 630.0e-9
t_master = 20.107167e-3   # from master cal output
N = round(2*t_master/OI)
eps = (2*t_master/OI) % 1.0
lc_0wind = 2*t_master / (N + eps)
print(f"t_master = {t_master*1e3:.9f} mm")
print(f"2*t/OI   = {2*t_master/OI:.9f}")
print(f"N_int_OI = {N}")
print(f"eps_OI   = {eps:.9f}  (should be ~0.276)")
print(f"lc_0wind = {lc_0wind*1e9:.9f} nm  (should be ~630.000000 nm)")
print(f"delta from 630nm = {(lc_0wind-OI)/OI * 3e8:.3f} m/s  (should be ~0)")

# ── 4. Verify master_cal_to_h06_cal passes t_m correctly ──────────────────
print("\n\n" + "=" * 60)
print("master_cal_to_h06_cal CONVERSION CHECK")
print("=" * 60)
from windcube.fpi_pipeline import master_cal_to_h06_cal, MasterCalibration, CalibrationResult
mc = MasterCalibration(
    t_m=t_master, sigma_t_m=3e-10,
    alpha=1.60885e-4, sigma_alpha=2e-9,
    R_refl=0.260, sigma_R_refl=0.002,
    R2=0.283, sigma_R2=0.004,
    I0=7700.0, sigma_I0=80.0,
    I1=0.0, sigma_I1=0.003,
    I2=0.0, sigma_I2=0.003,
    sigma0=0.01, sigma_sigma0=1e-13,
    sigma1=0.0, sigma2=0.0,
    B=100.0, sigma_B=100.0,
    ne_ratio=0.45, sigma_ne_ratio=0.004,
    epsilon_cal=0.2352, sigma_epsilon_cal=5e-4,
    n_frames_averaged=5, chi2_red_mean=4.0, n_converged=5,
)
h06cal = master_cal_to_h06_cal(mc)
print(f"mc.t_m        = {mc.t_m*1e3:.9f} mm")
print(f"h06cal.t_m    = {h06cal.t_m*1e3:.9f} mm")
print(f"h06cal fields: {list(h06cal.__dataclass_fields__.keys()) if hasattr(h06cal, '__dataclass_fields__') else dir(h06cal)}")
N2 = round(2*h06cal.t_m/OI)
eps2 = (2*h06cal.t_m/OI) % 1.0
print(f"eps_OI from h06cal.t_m = {eps2:.9f}  (should be ~0.276)")

# ── 5. Call run_airglow_inversion directly with synthetic data ────────────
print("\n\n" + "=" * 60)
print("DIRECT run_airglow_inversion CALL")
print("=" * 60)
from src.processing.archive.H06_airglow_inversion_2026_05_14 import run_airglow_inversion

# Build a synthetic profile that looks like a science fringe
r_grid = np.linspace(1.0, 110.0, 500)
# Simple Airy-like profile at v_rel=0
theta = r_grid * 1.60885e-4
delta = 4*np.pi*1.0*t_master*np.cos(theta)/OI
F = 4*0.26/(1-0.26)**2
profile = 3000.0/(1 + F*np.sin(delta/2)**2) + 500.0 + np.random.default_rng(42).normal(0, 20, len(r_grid))
sigma = np.full(len(r_grid), 20.0)

v_prior = -7255.0  # along-track prior
result = run_airglow_inversion(r_grid, profile, sigma, h06cal,
                                r_max_px=110.0, v_los_prior_ms=v_prior)
print(f"v_rel_ms   = {result.v_rel_ms:.2f} m/s  (should be near 0 for null wind)")
print(f"sigma_v    = {result.sigma_v_ms:.2f} m/s")
print(f"chi2_red   = {result.chi2_red:.3f}")
print(f"converged  = {result.converged}")
print(f"scan_ambig = {result.scan_ambiguous}")
print(f"Y_line     = {result.Y_line:.2f} ADU  (should be > 0)")
print(f"B_sci      = {result.B_sci:.2f} ADU")
