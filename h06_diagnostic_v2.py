"""
h06_diagnostic_v2.py — Complete H06 eps_OI_exp bug hunt.

Run from repo root:
  .venv\Scripts\python.exe h06_diagnostic_v2.py

Prints everything needed to find why eps_OI_exp=0.196 instead of 0.276.
"""
import sys, pathlib, numpy as np
sys.path.insert(0, '.')

SEP = "=" * 65

# ── 1. Print run_airglow_inversion source in full ─────────────────────────
print(SEP)
print("SECTION 1 — run_airglow_inversion full source")
print(SEP)
lines = pathlib.Path(
    'src/processing/H06_airglow_inversion_2026_05_14.py'
).read_text(encoding='utf-8').split('\n')

start = next(i for i, l in enumerate(lines) if 'def run_airglow_inversion' in l)
end   = start + 1
while end < len(lines):
    if end > start + 5 and lines[end].startswith('def '):
        break
    end += 1
print(f"(lines {start+1} – {end})")
for i, line in enumerate(lines[start:end], start=start+1):
    print(f"  {i}: {line}")

# ── 2. Print _lambda_c_scan seed block ────────────────────────────────────
print()
print(SEP)
print("SECTION 2 — _lambda_c_scan seed block (first 55 lines)")
print(SEP)
scan_start = next(i for i, l in enumerate(lines) if 'def _lambda_c_scan' in l)
for i, line in enumerate(lines[scan_start:scan_start+55], start=scan_start+1):
    print(f"  {i}: {line}")

# ── 3. Verify math ────────────────────────────────────────────────────────
print()
print(SEP)
print("SECTION 3 — Math verification")
print(SEP)
OI  = 630.0e-9
t   = 20.107167e-3
N   = round(2*t/OI)
eps = (2*t/OI) % 1.0
lc  = 2*t / (N + eps)
print(f"t_m          = {t*1e3:.9f} mm")
print(f"2*t/OI       = {2*t/OI:.9f}")
print(f"N_int_OI     = {N}")
print(f"eps_OI       = {eps:.9f}  (expect ~0.276)")
print(f"lc_0wind     = {lc*1e9:.9f} nm  (expect 630.000000000)")
print(f"v offset     = {(lc-OI)/OI*3e8:.3f} m/s  (expect 0.000)")

# ── 4. Verify master_cal_to_h06_cal ──────────────────────────────────────
print()
print(SEP)
print("SECTION 4 — master_cal_to_h06_cal passes t_m correctly")
print(SEP)
from windcube.fpi_pipeline import master_cal_to_h06_cal, MasterCalibration
mc = MasterCalibration(
    t_m=t, sigma_t_m=3e-10,
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
eps2 = (2*h06cal.t_m/OI) % 1.0
print(f"mc.t_m       = {mc.t_m*1e3:.9f} mm")
print(f"h06cal.t_m   = {h06cal.t_m*1e3:.9f} mm")
print(f"h06cal fields: {[f for f in dir(h06cal) if not f.startswith('_')]}")
print(f"eps_OI       = {eps2:.9f}  (expect ~0.276)")

# ── 5. Intercept _lambda_c_scan to print the cal.t_m it actually receives ─
print()
print(SEP)
print("SECTION 5 — Intercept _lambda_c_scan: what cal.t_m does it see?")
print(SEP)
import src.processing.H06_airglow_inversion_2026_05_14 as _h06mod

_original_scan = _h06mod._lambda_c_scan

def _intercepted_scan(r_good, prof_good, sigma_good, r_max, cal,
                      fsr_oi, n_scan=300, n_fine=500, v_los_prior_ms=0.0):
    N_dbg   = round(2*cal.t_m/OI)
    eps_dbg = (2*cal.t_m/OI) % 1.0
    print(f"  >>> _lambda_c_scan received cal.t_m = {cal.t_m*1e3:.9f} mm")
    print(f"  >>> N_int_OI={N_dbg}  eps_OI_exp={eps_dbg:.9f}")
    return _original_scan(r_good, prof_good, sigma_good, r_max, cal,
                          fsr_oi, n_scan, n_fine, v_los_prior_ms)

# Also intercept run_airglow_inversion to print the cal it receives
_original_rai = _h06mod.run_airglow_inversion

def _intercepted_rai(r_grid, profile_adu, sigma_adu, cal,
                     r_max_px=110.0, v_los_prior_ms=0.0,
                     n_scan=300, n_fine=500):
    print(f"  >>> run_airglow_inversion received cal.t_m = {cal.t_m*1e3:.9f} mm")
    # Temporarily patch _lambda_c_scan in the module
    _h06mod._lambda_c_scan = _intercepted_scan
    try:
        result = _original_rai(r_grid, profile_adu, sigma_adu, cal,
                               r_max_px, v_los_prior_ms, n_scan, n_fine)
    finally:
        _h06mod._lambda_c_scan = _original_scan
    return result

# Build synthetic profile
rng     = np.random.default_rng(42)
r_grid  = np.linspace(1.0, 110.0, 500)
theta   = r_grid * 1.60885e-4
delta   = 4*np.pi*t*np.cos(theta)/OI
F       = 4*0.26/(1-0.26)**2
profile = 3000.0/(1+F*np.sin(delta/2)**2) + 500.0 + rng.normal(0, 20, len(r_grid))
sigma   = np.full(len(r_grid), 20.0)

# Call via the intercepted wrapper
result = _intercepted_rai(r_grid, profile, sigma, h06cal,
                          r_max_px=110.0, v_los_prior_ms=-7255.0)

print()
print(f"H06 result:")
print(f"  v_rel_ms   = {result.v_rel_ms:.2f} m/s  (expect ~0 for null wind)")
print(f"  sigma_v    = {result.sigma_v_ms:.2f} m/s")
print(f"  chi2_red   = {result.chi2_red:.3f}")
print(f"  converged  = {result.converged}")
print(f"  scan_ambig = {result.scan_ambiguous}")
print(f"  Y_line     = {result.Y_line:.2f} ADU")

# ── 6. What t_m would give eps=0.196? ────────────────────────────────────
print()
print(SEP)
print("SECTION 6 — What t_m gives eps_OI_exp=0.196178?")
print(SEP)
eps_bad = 0.196178
for dN in [-1, 0, 1]:
    t_bad = (N + dN + eps_bad) * OI / 2
    diff_nm = (t_bad - t)*1e9
    print(f"  N+{dN:+d}: t={t_bad*1e3:.9f} mm  diff={diff_nm:+.3f} nm from master cal")

# phase_correct_gap effect
NE1    = 640.2248e-9
eps_cal = 0.235246
N_ne   = round(2*t/NE1)
t_pcg  = (N_ne + eps_cal) * NE1 / 2
eps_pcg = (2*t_pcg/OI) % 1.0
print(f"\n  phase_correct_gap(t={t*1e3:.6f}mm, eps_cal={eps_cal:.6f}, Ne1):")
print(f"    t_eff = {t_pcg*1e3:.9f} mm")
print(f"    eps_OI from t_eff = {eps_pcg:.9f}")
print(f"    N_int_OI from t_eff = {round(2*t_pcg/OI)}")

print()
print("DONE")
