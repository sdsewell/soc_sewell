# M03 Addendum — Thermal Line Broadening (`use_temperature` flag)

**Spec file:** `M03_airglow_synthesis_addendum_2026-05-24.md`  
**Amends:** `H03_airglow_synthesis_script_2026-05-13.md` (H03 script) and  
&emsp;&emsp;&emsp;&emsp;&emsp;`m03_airglow_synthesis_2026_05_12.py` (M03 library)  
**New library file:** `m03_airglow_synthesis_2026-05-24.py`  
**Status:** Draft  
**Date:** 2026-05-24  
**Author:** SOC / S. Sewell  
**Required by:** MC01 (`MC01_fpi_mc_engine_2026-05-24.md`)

---

## 1. Motivation

The existing M03 library synthesises airglow fringes using the **delta-function
source approximation**: the OI 630 nm emission line is treated as infinitely
narrow, so the Fredholm integral (Harding Eq. 1) collapses to a single Airy
evaluation at the Doppler-shifted line centre λ_c. Temperature plays no role.

The Harding (2014) Monte Carlo simulations (§4) require the full **thermally
broadened Gaussian source** (Harding Eq. 10) in which the line width Δλ encodes
the thermospheric temperature T via Harding Eq. 12:

$$\Delta\lambda = \frac{\lambda_0}{c}\sqrt{\frac{kT}{m}}$$

where k = `BOLTZMANN_J_PER_K`, m = `OXYGEN_MASS_KG`, c = `SPEED_OF_LIGHT_MS`
(all from `windcube/constants.py`). MC01 needs to synthesise truth images at
known (v_true, T_true) and then recover both, so temperature synthesis must be
supported.

This addendum adds a `use_temperature` flag to `synthesise_airglow_image()`.
When `False` (default), behaviour is **identical to the existing code** — no
regression. When `True`, the spectral integral path is activated.

---

## 2. Changes to `synthesise_airglow_image()`

### 2.1 New parameters

```python
def synthesise_airglow_image(
    params: InstrumentParams,
    v_rel_ms: float,
    # --- existing parameters unchanged ---
    Y_line: float = 1.0,
    Y_bg: float = 0.0,
    image_size: int = 256,
    cx: float = None,
    cy: float = None,
    R_bins: int = 500,
    L_synth: int = 300,       # un-deprecated: used when use_temperature=True
    n_fsr: float = 5.0,       # un-deprecated: used when use_temperature=True
    observation_mode: str = None,
    add_noise: bool = True,
    noise_type: str = "gaussian",
    snr: float = 5.0,
    rng: np.random.Generator = None,
    # --- NEW ---
    use_temperature: bool = False,
    T_true_K: float = 800.0,
) -> dict:
```

| New parameter | Type | Default | Description |
|---------------|------|---------|-------------|
| `use_temperature` | bool | `False` | If `True`, synthesise a thermally broadened Gaussian line profile via the spectral integral path (Harding Eqs. 10–12). If `False`, use the existing delta-function path unchanged. |
| `T_true_K` | float | 800.0 | True thermospheric temperature in K. Used only when `use_temperature=True`. Must be > 0. |

`L_synth` and `n_fsr` are **un-deprecated** — they now govern the spectral
discretisation for the `use_temperature=True` path. Their defaults (300 and 5.0)
are unchanged from the original spec and match Harding's anti-inverse-crime rule
(synthesise at L=300, invert at L=101).

### 2.2 New helper: `_gaussian_source_spectrum()`

```python
def _gaussian_source_spectrum(
    lambda_grid_m: np.ndarray,
    lambda_c_m: float,
    delta_lambda_m: float,
    Y_line: float,
    Y_bg: float,
) -> np.ndarray:
    """
    Harding Eq. 10: thermally broadened Gaussian source spectrum.

        Y(λ) = Y_bg + Y_line * exp(-0.5 * ((λ - λ_c) / Δλ)²)

    Parameters
    ----------
    lambda_grid_m  : wavelength grid, metres, shape (L,)
    lambda_c_m     : Doppler-shifted line centre, metres
    delta_lambda_m : Doppler line half-width (1/e), metres (Harding Eq. 12)
    Y_line         : line amplitude (dimensionless scale factor)
    Y_bg           : flat background emission (ADU)

    Returns
    -------
    Y : ndarray shape (L,) — source spectrum in ADU units
    """
    return Y_bg + Y_line * np.exp(
        -0.5 * ((lambda_grid_m - lambda_c_m) / delta_lambda_m) ** 2
    )
```

### 2.3 New helper: `delta_lambda_from_temperature()`

Public function — also imported by M06 addendum (§ below).

```python
def delta_lambda_from_temperature(T_K: float, lambda0_m: float = None) -> float:
    """
    Harding Eq. 12: Doppler line half-width from thermospheric temperature.

        Δλ = (λ₀ / c) * sqrt(k * T / m)

    Parameters
    ----------
    T_K       : temperature in K. Must be > 0.
    lambda0_m : rest wavelength in metres.
                Default: OI_WAVELENGTH_AIR_M from windcube/constants.py.

    Returns
    -------
    delta_lambda_m : float, metres
    """
    from windcube.constants import (
        OI_WAVELENGTH_AIR_M, SPEED_OF_LIGHT_MS,
        BOLTZMANN_J_PER_K, OXYGEN_MASS_KG,
    )
    if lambda0_m is None:
        lambda0_m = OI_WAVELENGTH_AIR_M
    if T_K <= 0:
        raise ValueError(f"T_K must be > 0; got {T_K}")
    return (lambda0_m / SPEED_OF_LIGHT_MS) * np.sqrt(BOLTZMANN_J_PER_K * T_K / OXYGEN_MASS_KG)
```

### 2.4 Modified step 4 inside `synthesise_airglow_image()`

Replace the existing step 4 block with the following branched logic:

```python
# Step 4: 1D fringe profile
if not use_temperature:
    # --- Original delta-function path (unchanged) ---
    airy_profile = airy_modified(r_bins, lambda_c_m, params)
    profile_1d   = Y_line * airy_profile + Y_bg + params.B

else:
    # --- Thermally broadened spectral integral path (Harding Eqs. 10–12) ---
    # Anti-inverse-crime rule: synthesise at L_synth=300, invert at L_invert=101
    delta_lambda_m = delta_lambda_from_temperature(T_true_K)

    # Spectral grid: n_fsr FSRs centred on lambda_c_m
    FSR_m     = OI_WAVELENGTH_AIR_M ** 2 / (2.0 * params.t)
    lam_lo    = lambda_c_m - (n_fsr / 2.0) * FSR_m
    lam_hi    = lambda_c_m + (n_fsr / 2.0) * FSR_m
    lam_grid  = np.linspace(lam_lo, lam_hi, L_synth)   # shape (L_synth,)
    d_lam     = lam_grid[1] - lam_grid[0]

    # Source spectrum: Harding Eq. 10
    Y_lam = _gaussian_source_spectrum(
        lam_grid, lambda_c_m, delta_lambda_m, Y_line, Y_bg
    )  # shape (L_synth,)

    # Instrument function A(r, λ) evaluated at each wavelength: shape (R_bins, L_synth)
    # Harding Eq. 15 discrete approximation: S(r_i) ≈ Σ_j Ã(r_i, λ_j) Y(λ_j) Δλ + B
    profile_1d = np.zeros(R_bins)
    for j, lam_j in enumerate(lam_grid):
        airy_j     = airy_modified(r_bins, lam_j, params)
        profile_1d += airy_j * Y_lam[j] * d_lam
    profile_1d += params.B
```

> **Performance note:** The loop over `L_synth=300` wavelengths is acceptable for
> single-image synthesis but is the dominant cost per MC01 trial. Claude Code may
> vectorise this as `airy_matrix = np.stack([airy_modified(r_bins, lj, params) for lj in lam_grid])`,
> then `profile_1d = airy_matrix.T @ (Y_lam * d_lam) + params.B` if profiling shows
> it to be a bottleneck.

### 2.5 Return dict additions

Add two new keys to the returned dict when `use_temperature=True`; set to `None`
when `use_temperature=False`:

```python
"T_true_K":        float(T_true_K) if use_temperature else None,
"delta_lambda_m":  float(delta_lambda_m) if use_temperature else None,
```

---

## 3. Validation check: `delta_lambda_from_temperature()`

At T = 800 K, λ₀ = 630.0304 nm:

```
Δλ = (630.0304e-9 / 299792458) * sqrt(1.380649e-23 * 800 / 2.6567e-26)
   = (2.101e-15) * sqrt(4.158e5)
   = (2.101e-15) * 644.8
   ≈ 1.354e-12 m  ≈ 1.354 pm
```

This corresponds to a 1/e Doppler half-width of ~645 m/s at 800 K — consistent
with thermospheric values in the literature.

---

## 4. Acceptance tests

These extend the existing M03 test suite; do not remove existing tests.

| Test ID | Description | Pass criterion |
|---------|-------------|----------------|
| T-M03-T01 | `use_temperature=False` produces identical output to pre-addendum code | Max absolute difference < 1e-10 ADU |
| T-M03-T02 | `delta_lambda_from_temperature(800.0)` | Result in [1.30e-12, 1.40e-12] m |
| T-M03-T03 | `delta_lambda_from_temperature(T_K=0)` | Raises `ValueError` |
| T-M03-T04 | `use_temperature=True`, T=800K produces wider fringes | FWHM of profile_1d > FWHM of delta-function profile at same v |
| T-M03-T05 | `use_temperature=True` return dict contains `T_true_K` and `delta_lambda_m` | Both keys present and finite |
| T-M03-T06 | `use_temperature=True` profile integral ≈ `use_temperature=False` integral | Within 10% for Y_line=1, T=800K, v=0 |
| T-M03-T07 | Anti-inverse-crime: L_synth=300 vs L_synth=200 | Max profile difference < 1% |

---

## 5. Change log

| Version | Date | Summary |
|---------|------|---------|
| Addendum 1.0 | 2026-05-24 | Add `use_temperature` flag and `delta_lambda_from_temperature()` |
