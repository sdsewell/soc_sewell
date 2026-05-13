# SPEC-M03 — Airglow Synthesis Library (`m03_airglow_synthesis`)

**Spec version:** 3  
**Date:** 2026-05-13  
**Status:** READY FOR IMPLEMENTATION  
**Module:** `src/fpi/m03_airglow_synthesis_2026_05_13.py`  
**Previous version:** `src/fpi/m03_airglow_synthesis_2026_05_12.py`

---

## 1. Purpose

`m03_airglow_synthesis` provides `synthesise_airglow_image()`, the sole
function responsible for generating a synthetic WindCube FPI OI 630 nm
airglow fringe image with physically correct CCD noise.  It is called
exclusively by `H03_airglow_synthesis_2026_05_13.py`.

---

## 2. Motivation for this revision

The previous version modelled noise as a signal-independent Gaussian:

```
σ_N = ΔS / SNR
pixel += Normal(0, σ_N²)
```

This is inconsistent with real CCD noise, where σ depends on local
signal level.  The inversion module H06 estimates per-bin sigma as
`sqrt(profile_ADU)` (Poisson), so when H03 adds SNR-based Gaussian noise:

- H03 adds σ ≈ 786 e⁻ (SNR=5, ΔS≈3929 ADU)
- H06 assumes σ ≈ 89 e⁻ (sqrt of ~8000 ADU peak)
- χ²/ν = (89/786)² ≈ 0.013  →  χ²/ν << 1 in all synthetic tests

The fix replaces the SNR abstraction with a physical four-component CCD
noise model parameterised by focal-plane temperature, gain, read noise,
and exposure time.  Dark current is derived from the Teledyne e2v CCD97
datasheet temperature formula.

---

## 3. Physical noise model

### 3.1 Signal chain (per pixel)

```
signal_e  = image_ideal_ADU × gain_e_per_adu          # photon-generated e⁻

dark_rate = Qdd × f(T_K) / f(T_ref)                    # e2v formula (§3.2)
dark_e    = dark_rate × exp_time_s                      # thermal e⁻ per pixel

total_e   = Poisson(signal_e + dark_e)                  # shot noise draw
total_e  += Normal(0, read_noise_e²)                    # read noise

pixel_ADU = round( total_e / gain_e_per_adu + bias_ADU )
pixel_ADU = clip(pixel_ADU, 0, 65535).astype(uint16)
```

`image_ideal_ADU` is the noiseless Airy fringe image computed by the
existing forward model.  The noise draws are per-pixel (full 2D array,
not per radial bin).

### 3.2 Dark current temperature formula

Source: Teledyne e2v CCD97 datasheet, page 4, Figure 3 caption / Note 8.

```
Q_d(T) = Q_dd × f(T_K) / f(T_ref)

f(T)  = 1.14 × 10⁶ × T³ × exp(−9080 / T)

T_ref = 293.15 K  (20 °C reference)
Q_dd  = 400 e⁻/pixel/s  (typical)  or  800 e⁻/pixel/s  (maximum)
Valid range: 150 K – 300 K  (−123 °C to +27 °C)
```

This formula gives dark signal in **e⁻/pixel/s**.  Multiply by
`exp_time_s` to get total dark electrons per pixel per exposure.

Note 8 of the datasheet states there is also a clock-induced charge (CIC)
component independent of integration time.  CIC is not modelled here;
it will be added when a measured value is available from FlatSat testing.

### 3.3 Noise budget at design point

Design point: T = −20 °C, t_exp = 10 s, gain = 1 e⁻/ADU, peak ≈ 8000 ADU.

| Source | σ (e⁻ rms) | fraction |
|--------|-----------|---------|
| Signal shot noise | 89.4 | 99.7 % |
| Dark shot noise | 4.4 | 0.2 % |
| Read noise | 2.2 | < 0.1 % |
| **Total** | **89.6** | |

At +20 °C (cooling failure scenario), dark shot noise rises to 63.8 e⁻
and dominates; total σ increases to ~110 e⁻.

---

## 4. Function signature change

### 4.1 Removed parameters

| Parameter | Reason |
|-----------|--------|
| `noise_type` | Always physical CCD model; no longer selectable |
| `snr` | Replaced by physical parameters; SNR is a derived output |

### 4.2 New parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `T_focal_plane_C` | float | −20.0 | Focal-plane temperature (°C).  Used to compute dark rate via e2v formula.  Range −40 to +20 °C. |
| `exp_time_s` | float | 10.0 | Exposure time (s).  Science default 10 s; calibration (neon) default 120 s. |
| `gain_e_per_adu` | float | 1.0 | CCD conversion gain (e⁻/ADU).  TBC from FlatSat characterisation. |
| `read_noise_e` | float | 2.2 | Read noise (e⁻ rms).  Source: CCD97 datasheet, OSH amp, 50 kHz CDS. |
| `Qdd_at_20C` | float | 400.0 | Dark signal reference at 293 K (e⁻/pix/s).  Typical = 400, maximum = 800 per datasheet. |

### 4.3 Unchanged parameters

All existing parameters (`params`, `v_rel_ms`, `Y_line`, `Y_bg`,
`image_size`, `R_bins`, `L_synth`, `n_fsr`, `observation_mode`,
`add_noise`, `rng`) are retained with unchanged semantics.

`add_noise=False` bypasses the entire noise block and returns the ideal
noiseless image (useful for debugging).

### 4.4 Updated return dict

| Key | Change |
|-----|--------|
| `snr_actual` | Now computed as `ΔS / σ_mean` where `σ_mean = sqrt(mean(image_ideal_ADU * gain_e_per_adu + dark_e + read_noise_e²)) / gain_e_per_adu`.  Remains in return dict for figure annotation. |
| `dark_rate_e_per_s` | **New.**  Computed dark rate at `T_focal_plane_C` (e⁻/pix/s). |
| `dark_e_per_pixel` | **New.**  Total dark per pixel = `dark_rate_e_per_s × exp_time_s`. |
| `T_focal_plane_C` | **New.**  Echo of input temperature. |
| `exp_time_s` | **New.**  Echo of input exposure time. |
| `snr_requested` | **Removed** (no longer an input). |

---

## 5. Internal implementation notes

### 5.1 Dark current function

Define a module-level helper (not exported):

```python
def _dark_rate_e2v(T_celsius: float, Qdd_at_20C: float = 400.0) -> float:
    """
    CCD97 dark signal.  Source: Teledyne e2v datasheet page 4.
    Returns dark signal in e-/pixel/s.
    Valid for T in [−123, +27] °C (150–300 K).
    """
    T_k   = T_celsius + 273.15
    T_ref = 293.15
    _f    = lambda T: 1.14e6 * T**3 * np.exp(-9080.0 / T)
    return float(Qdd_at_20C * _f(T_k) / _f(T_ref))
```

### 5.2 Noise draw (replace existing noise block entirely)

```python
if add_noise:
    dark_rate  = _dark_rate_e2v(T_focal_plane_C, Qdd_at_20C)
    dark_e     = dark_rate * exp_time_s                         # scalar

    signal_e   = image_ideal_ADU * gain_e_per_adu               # 2D array
    lambda_p   = signal_e + dark_e                              # Poisson mean per pixel
    total_e    = rng.poisson(lambda_p).astype(float)
    total_e   += rng.normal(0.0, read_noise_e,
                            size=image_ideal_ADU.shape)
    image_noisy = np.round(
        total_e / gain_e_per_adu + bias_ADU
    ).clip(0, 65535).astype(np.uint16)
else:
    image_noisy = image_ideal_ADU.copy()
    dark_rate   = 0.0
    dark_e      = 0.0
```

`bias_ADU` is already present in `image_ideal_ADU` (set by `Y_bg` / the
existing background term); do **not** add it a second time.  Confirm this
against the existing code before implementing.

### 5.3 SNR diagnostic (replace existing SNR calculation)

```python
sigma_mean_adu = (np.sqrt(
    np.mean(image_ideal_ADU) * gain_e_per_adu
    + dark_e
    + read_noise_e**2
) / gain_e_per_adu)

delta_S     = float(profile_1d.max() - profile_1d.min())
snr_actual  = delta_S / sigma_mean_adu if sigma_mean_adu > 0 else np.inf
```

---

## 6. Acceptance criteria

| ID | Check | Pass condition |
|----|-------|---------------|
| M03-C1 | χ²/ν from H06 inversion of H03 synthetic at T=−20°C, t=10s | 0.5 ≤ χ²/ν ≤ 2.0 |
| M03-C2 | `dark_rate_e_per_s` at T=−20°C | 1.90 – 2.05 e⁻/pix/s |
| M03-C3 | `dark_rate_e_per_s` at T=+20°C | 390 – 420 e⁻/pix/s |
| M03-C4 | `snr_actual` at T=−20°C, t=10s, Y_line=6480 | 30–60 (signal-shot dominated) |
| M03-C5 | `add_noise=False` returns noiseless image unchanged | image == image_ideal_ADU |
| M03-C6 | Pixel dtype of noisy image | uint16 |
| M03-C7 | No pixel value exceeds 65535 | assert image_noisy.max() ≤ 65535 |

---

## 7. Revision history

| Version | Date | Author | Summary |
|---------|------|--------|---------|
| 1 | 2026-05-05 | SOC | Initial spec |
| 2 | 2026-05-12 | SOC | SNR dialog and Gaussian noise model |
| 3 | 2026-05-13 | SOC | Replace SNR/Gaussian with physical CCD noise model.  Add `T_focal_plane_C`, `exp_time_s`, `gain_e_per_adu`, `read_noise_e`, `Qdd_at_20C`.  Remove `noise_type`, `snr`. |

---

## 8. Claude Code implementation prompt

```
cat PIPELINE_STATUS.md

# Implement the M03 physical CCD noise model update.
# Spec: docs/specs/SPEC-M03_airglow_synthesis_lib_2026-05-13.md
# Previous module: src/fpi/m03_airglow_synthesis_2026_05_12.py
# New module:      src/fpi/m03_airglow_synthesis_2026_05_13.py

# Steps:
# 1. Copy the previous module to the new filename.
# 2. Add _dark_rate_e2v() helper (spec §5.1).
# 3. Replace the existing noise block with the physical model (spec §5.2).
#    IMPORTANT: confirm whether bias_ADU is already in image_ideal_ADU
#    before adding it in the noise draw — do not double-add.
# 4. Replace the SNR calculation with the physical σ_mean formula (spec §5.3).
# 5. Update synthesise_airglow_image() signature:
#    - Remove: noise_type, snr
#    - Add:    T_focal_plane_C=-20.0, exp_time_s=10.0,
#              gain_e_per_adu=1.0, read_noise_e=2.2, Qdd_at_20C=400.0
# 6. Update return dict: add dark_rate_e_per_s, dark_e_per_pixel,
#    T_focal_plane_C, exp_time_s; remove snr_requested.
# 7. Run acceptance checks M03-C1 through M03-C7 (spec §6) using a
#    minimal inline test — no external test file needed, just assert
#    statements at module bottom guarded by if __name__ == '__main__'.

# Report back:
# - The assert output for each M03-C check (pass/fail)
# - The dark_rate_e_per_s value at T=-20°C and T=+20°C
# - One sample snr_actual value (T=-20°C, t=10s, Y_line=6480)

# Update PIPELINE_STATUS.md — set M03 status, date, test count.
git add PIPELINE_STATUS.md \
        src/fpi/m03_airglow_synthesis_2026_05_13.py \
        docs/specs/SPEC-M03_airglow_synthesis_lib_2026-05-13.md
git commit -m "feat(M03): replace SNR/Gaussian noise with physical CCD model

Add _dark_rate_e2v() using Teledyne e2v CCD97 datasheet formula
(page 4, Fig 3): Qd(T) = Qdd * f(T)/f(293K), f(T)=1.14e6*T^3*exp(-9080/T).
Replace signal-independent Gaussian noise with per-pixel Poisson shot
noise (signal + dark) plus Gaussian read noise.  New parameters:
T_focal_plane_C (default -20 C), exp_time_s (default 10 s),
gain_e_per_adu (1.0), read_noise_e (2.2 e-), Qdd_at_20C (400 e-/pix/s).
Removes noise_type and snr input parameters.

Expected result: chi2/nu from H06 inversion rises from ~0.05 to 0.5-2.0.

Also updates PIPELINE_STATUS.md"
```
