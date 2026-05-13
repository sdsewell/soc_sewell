# H03 — Airglow Fringe Synthesis

**Spec ID:** H03
**Spec file:** `docs/specs/H03_airglow_synthesis_2026-05-05.md`
**Project:** WindCube FPI Pipeline
**Institution:** NCAR / High Altitude Observatory (HAO)
**Status:** Authoritative
**Depends on:** S01, S02, S03, S04,
  H01 (Airy forward model — must pass all 15 tests first),
  H02 (calibration synthesis — provides `radial_profile_to_image`)
**Used by:**
  - S12 (M03 — annular reduction) — receives 2D airglow image as input
  - H06 (airglow inversion) — receives 1D airglow fringe profile for round-trip tests
  - S16 (INT02) — visualises airglow fringe images
**References:**
  - Harding et al. (2014) Applied Optics 53(4), Sections 2.B and 4
  - NIST Atomic Spectra Database — OI 630.0304 nm air wavelength

> **What changed from the initial 2026-05-05 draft:**
> 1. **Two observation regimes documented** (Section 2.7). WindCube operates
>    in two distinct velocity regimes corresponding to different observation
>    geometries. The cross-track regime (±1000 m/s, even orbits) and the
>    along-track regime (−6000 to −8000 m/s, odd orbits) produce fringe
>    patterns that differ by approximately one to two integer fringe orders.
>    The physical origin, fringe-order arithmetic, and implications for the
>    wavelength grid are fully documented.
> 2. **`n_fsr` default increased from 5 to 10.** The original default of 5
>    provided only ~1.5× margin at the along-track extreme (−8000 m/s,
>    Δλ ≈ 16.8 pm). The new default of 10 gives ~3× margin for both regimes
>    on the same grid without per-call configuration.
> 3. **`observation_mode` parameter added** to `synthesise_airglow_image()`.
>    Passing `'cross_track'` or `'along_track'` validates `v_rel_ms` against
>    the expected range for that mode and records it in the output dict.
>    Passing `None` (default) accepts the full H01 velocity range.
> 4. **`fringe_order_offset` added to output dict** — the integer number of
>    FSRs by which λ_c is offset from λ₀. Zero for cross-track; −1 or −2 for
>    along-track. This is the quantity H06's brute-force scan must identify
>    correctly to avoid a ~4700 m/s wind error per missed order.
> 5. **T9 added** — verifies fringe order offsets for both regimes and that
>    `observation_mode` validation raises correctly for out-of-range velocities.
> 6. **Three new constants added** to `windcube/constants.py` (Section 4):
>    `V_REL_CROSSTRACK_MAX_MS`, `V_REL_ALONGTRACK_MIN_MS`,
>    `V_REL_ALONGTRACK_MAX_MS`.

---

## 1. Purpose

H03 generates a synthetic 2D CCD airglow fringe image from a Doppler-shifted
OI 630 nm emission line. It is the airglow counterpart to H02, and together
they provide the synthetic inputs needed for round-trip tests of the complete
H05 → H06 inversion pipeline.

H03 has two responsibilities:

1. **1D fringe profile synthesis** — build the H01 instrument matrix for a
   wavelength grid centred on the OI rest wavelength, construct the
   Doppler-shifted source spectrum vector via `make_airglow_spectrum()`, and
   evaluate `s = A @ y + B` (Harding Eq. 16). Produces the 1D radial fringe
   profile `S_OI(r)`.

2. **2D image generation** — wrap `S_OI(r)` into a 2D CCD image using
   `radial_profile_to_image()` imported from H02, then add noise.

**Key structural difference from H02:** H02 calls `airy_modified()` directly
at each neon wavelength because the two lines are ~188 FSRs apart and cannot
share a wavelength grid. H03 uses the full `build_instrument_matrix()` +
`make_airglow_spectrum()` path from H01 because the OI source is a single
narrow line in a well-defined spectral neighbourhood, and the matrix approach
(a) exactly mirrors Harding's formulation and (b) ensures the synthesis and
inversion models use the same mathematical machinery, differing only in grid
resolution (anti-inverse-crime rule).

**`radial_profile_to_image` is imported from H02, not reimplemented.**

---

## 2. Physical background

### 2.1 The OI 630 nm airglow emission

The primary WindCube science target is the OI forbidden line at 630.0304 nm
(air wavelength, NIST ASD). This emission originates from dissociative
recombination of O₂⁺ in the thermosphere at approximately 250 km altitude.
The line-of-sight velocity Doppler-shifts the line centre by:

```
λ_c = λ₀ · (1 + v_rel / c)          [Harding Eq. 11]

where:
  λ₀    = OI_WAVELENGTH_AIR_M = 630.0304e-9 m
  v_rel = line-of-sight velocity (m/s); positive = recession
  c     = SPEED_OF_LIGHT_MS
```

### 2.2 Source model — delta function in the WindCube limit

WindCube's instrument finesse is ~4.9 (R_eff = 0.53), giving an instrument
linewidth of approximately FSR / finesse ≈ 9.92 pm / 4.9 ≈ 2 pm. The
thermal Doppler width of the OI line at thermospheric temperatures (~800 K) is:

```
Δλ_thermal = λ₀/c · √(kT/m_O)
           = 630.0304e-9 / 299792458 · √(1.38e-23 × 800 / 2.66e-26)
           ≈ 0.57 pm
```

Since the instrument linewidth (~2 pm) is broader than the thermal width
(~0.57 pm), temperature broadening changes the fringe shape negligibly.
The OI source is therefore modelled as a **spectral delta function**
(Harding Eq. 10, delta-function limit):

```
Y_OI(λ) = Y_bg + Y_line · δ(λ − λ_c)   [Harding Eq. 10, Δλ→0]
```

Harding Eq. 12 (Doppler broadening → temperature) is **deliberately
excluded** — temperature is not a WindCube science product. This is
consistent with H01 §4.3 and H06 §2.1; do not reopen this decision.

### 2.3 Using the instrument matrix (Harding Eqs. 14–16)

The 1D fringe profile is computed via the H01 matrix equation:

```
s = A @ y + B·1          [Harding Eq. 16]
```

where:
- `A` — (R, L) instrument matrix from `build_instrument_matrix()` [H01 §6]
- `y` — (L,) source spectrum vector from `make_airglow_spectrum()` [H01 §6]
- `B` — scalar CCD bias from `params.B`
- `s` — (R,) predicted fringe profile in ADU

The wavelength grid is centred on `OI_WAVELENGTH_AIR_M` and spans `n_fsr`
free spectral ranges. The choice of `L` is governed by the anti-inverse-crime
rule (Section 2.4).

This path is deliberately parallel to how H06 evaluates its forward model —
the synthesis and inversion use the same functions from H01, with L_synth ≠ L_inv
being the only intentional difference.

### 2.4 Anti-inverse-crime rule

Per H01 §2.4, the wavelength grid resolution used for synthesis must differ
from the grid used by the inversion module (H06). Using identical grids
produces artificially perfect recoveries.

```
L_synth = 300    (synthesis — used by H03)
L_inv   = 101    (inversion — used by H06)
```

H03 always passes `L = L_SYNTH = 300` to `make_wavelength_grid()`.
H06 always passes `L = 101`. This asymmetry is **intentional and required**.
Do not "fix" it.

### 2.5 Noise model — Gaussian white noise for airglow

The neon calibration image is photon-noise limited (Poisson), which is why
H02 uses `add_poisson_noise()`. The OI airglow image is different: the
airglow signal is very faint relative to the dark current, making the
dominant noise source dark noise — well modelled as **Gaussian white noise**.
This is consistent with Harding §4, which explicitly uses Gaussian white
noise of a specified SNR for Monte Carlo simulations of airglow frames.

SNR is defined following Harding Eq. 17:
```
SNR = ΔS / σ_N

where:
  ΔS   = max(S_OI(r)) − min(S_OI(r))   (peak-to-trough fringe amplitude)
  σ_N  = standard deviation of the Gaussian noise (ADU)
```

H03 adds Gaussian noise at a caller-specified SNR. It also supports Poisson
noise as an option (e.g. for bright airglow scenarios), but Gaussian is
the default.

### 2.6 Sky background

The `Y_bg` parameter (ADU per wavelength bin, uniform across all λ) adds a
spectrally flat background to the source spectrum. This represents diffuse
sky glow or scattered light that is spectrally unresolved at FPI resolution.
Setting `Y_bg = 0` (the default) produces a pure line profile.

### 2.7 Two observation regimes and fringe order offsets

WindCube operates in two physically distinct velocity regimes determined by
orbital geometry. Both regimes are supported by the same synthesis code; only
the caller-supplied `v_rel_ms` differs. Understanding the fringe-order
implications is essential for verifying that H06's brute-force scan correctly
identifies the interference order.

#### The FSR velocity equivalent

One free spectral range (FSR) corresponds to a shift of λ_c by one
interference order. The Doppler velocity that shifts λ_c by exactly one FSR is:

```
FSR_OI   = λ₀² / (2 · n · t) ≈ 9.922e-15 m  (at λ₀ = 630.0304 nm, t = 20.008 mm)
v_FSR    = c × FSR_OI / λ₀
         = 299792458 × 9.922e-15 / 630.0304e-9
         ≈ 4723 m/s per FSR
```

#### Regime 1 — Cross-track (even orbits), thermospheric wind

| Property | Value |
|---|---|
| Geometry | LOS approximately perpendicular to orbit track; limb-viewing |
| Velocity source | Thermospheric horizontal wind projected onto LOS |
| Typical range | ±200 m/s (quiet), up to ±500 m/s (storm) |
| Valid synthesis range | `−V_REL_CROSSTRACK_MAX_MS` to `+V_REL_CROSSTRACK_MAX_MS` = ±1000 m/s |
| FSR fraction at ±1000 m/s | ±0.21 FSR |
| Integer order offset | **0** — λ_c remains within the same FSR period as λ₀ |
| λ_c range | 630.0304 ± 2.1 pm |

The fringe pattern is qualitatively identical to the zero-wind pattern; only
the radial positions of peaks shift slightly. H06 can find λ_c from a scan
over ±0.5 FSR around λ₀.

#### Regime 2 — Along-track (odd orbits), spacecraft + wind

| Property | Value |
|---|---|
| Geometry | LOS approximately aligned with the spacecraft velocity vector |
| Velocity source | Spacecraft orbital velocity (~7.5 km/s) projected onto LOS, plus thermospheric wind |
| Typical range | −6000 to −8000 m/s (blueshift; spacecraft approaches the thermosphere) |
| Valid synthesis range | `V_REL_ALONGTRACK_MIN_MS` to `V_REL_ALONGTRACK_MAX_MS` = −8000 to −6000 m/s |
| FSR fraction | −1.27 to −1.69 FSR |
| Integer order offset | **−1 or −2** — λ_c shifted by approximately one to two full FSR periods below λ₀ |
| λ_c range | 630.0304 − (12.7 to 16.8) pm ≈ 629.987 to 629.991 nm |

**The fringe pattern for along-track observations has the same Airy shape** as
cross-track — the reflectivity, PSF, and etalon geometry are unchanged. What
differs is which radial positions on the detector the peaks occupy, because
λ_c is shifted by one to two integer orders. An inversion that scans only
±0.5 FSR around λ₀ will alias to the wrong order and return a wind error of
approximately n × 4723 m/s (n = 1 or 2). This is exactly the failure mode
that H06's brute-force scan (§7.2, scanning ±1.5 FSR) is designed to prevent.

#### Fringe order offset arithmetic

The integer number of FSRs by which λ_c is displaced from λ₀ is:

```python
FSR_OI_M        = OI_WAVELENGTH_AIR_M**2 / (2.0 * params.t)
lambda_c_m      = OI_WAVELENGTH_AIR_M * (1.0 + v_rel_ms / SPEED_OF_LIGHT_MS)
delta_lam       = lambda_c_m - OI_WAVELENGTH_AIR_M
fringe_order_offset = int(round(delta_lam / FSR_OI_M))
```

Expected values for representative velocities:

| v_rel (m/s) | Δλ (pm) | Δλ / FSR | `fringe_order_offset` |
|---|---|---|---|
| 0 | 0.0 | 0.00 | 0 |
| ±1000 | ±2.1 | ±0.21 | 0 |
| −6000 | −12.6 | −1.27 | −1 |
| −7000 | −14.7 | −1.48 | −1 |
| −7400 | −15.6 | −1.57 | −2 (boundary) |
| −8000 | −16.8 | −1.69 | −2 |

The −1/−2 transition occurs near v = −7400 m/s (1.5 FSR boundary). Both
integer offsets are physically valid along-track observations.

#### Fringe shape invariance under integer order shifts

A key consequence of the Airy function's periodicity: **the fringe shape —
peak width, contrast, and envelope — is identical regardless of which integer
order λ_c occupies**. The same `airy_modified()` function describes both
regimes. The only difference is which radial angle satisfies the constructive
interference condition 2nt·cos(θ) = mλ_c. This means:

1. H03's synthesis code is regime-independent — the same matrix path applies.
2. H06's inversion model is correct for both regimes — no regime-specific
   forward model is needed.
3. The only regime-specific logic lives in H06's brute-force scan, which must
   cover ±1.5 FSR to handle both cases without prior knowledge of the regime.
4. `fringe_order_offset` in the output dict is a diagnostic quantity; it does
   not affect any calculation.

#### Wavelength grid width by regime

The grid must contain λ_c. With the grid centred on λ₀:

| Regime | Max |Δλ| | Required half-width | Minimum n_fsr |
|---|---|---|---|
| Cross-track only | 2.1 pm | > 2.1 pm | ≥ 1 FSR |
| Along-track only | 16.8 pm | > 16.8 pm | ≥ 2 FSR (barely); 4 FSR safe |
| Both (default) | 16.8 pm | > 16.8 pm | **10 FSR recommended** (~3× margin) |

The default `n_fsr = 10` covers both regimes on the same grid. For
cross-track-only synthesis, `n_fsr = 5` halves matrix build time but should
not be used as the default because both regimes share the same code path.

---

## 3. Function signatures

Implement in this order:
`add_gaussian_noise` → `synthesise_airglow_image`.

`radial_profile_to_image` and `add_poisson_noise` are **imported from H02**;
do not reimplement them.

### 3.1 `add_gaussian_noise`

```python
def add_gaussian_noise(
    image_noiseless: np.ndarray,       # shape (N, N), float64, ADU
    snr:             float,            # SNR = ΔS / σ_N per Harding Eq. 17
    profile_1d:      np.ndarray,       # 1D fringe profile used to compute ΔS
    rng: np.random.Generator = None,   # default_rng() if None
) -> np.ndarray:
    """
    Add Gaussian white noise to a noiseless airglow CCD image at a
    specified SNR, following Harding (2014) §4.

    σ_N is derived from the 1D fringe profile amplitude:
        ΔS  = max(profile_1d) − min(profile_1d)
        σ_N = ΔS / snr

    Gaussian noise N(0, σ_N²) is added independently to each pixel.

    Parameters
    ----------
    image_noiseless : float64 array of CCD counts, shape (N, N).
    snr             : signal-to-noise ratio (ΔS / σ_N). Must be > 0.
                      Typical operational value: 5 (Harding §4.C).
    profile_1d      : 1D noiseless fringe profile, shape (R,). Used only
                      to compute ΔS; must be the same profile that was
                      wrapped into image_noiseless.
    rng             : numpy Generator. Pass default_rng(seed) for
                      reproducibility. If None, uses np.random.default_rng().

    Returns
    -------
    image_noisy : np.ndarray, same shape as image_noiseless, float64.

    Raises
    ------
    ValueError : if snr <= 0.
    """
```

### 3.2 `synthesise_airglow_image`

```python
def synthesise_airglow_image(
    params:           'InstrumentParams',  # from H01
    v_rel_ms:         float,               # line-of-sight velocity, m/s
    Y_line:           float     = 1000.0,  # line intensity, ADU per bin
    Y_bg:             float     = 0.0,     # flat sky background, ADU per bin
    image_size:       int       = 256,     # CCD dimension, pixels
    cx:               float     = None,    # fringe centre x (default: geometric)
    cy:               float     = None,    # fringe centre y (default: geometric)
    R_bins:           int       = 500,     # radial bins in 1D profile
    L_synth:          int       = 300,     # wavelength bins for synthesis grid
    n_fsr:            float     = 10.0,    # FSRs spanned by wavelength grid
    observation_mode: str       = None,    # 'cross_track', 'along_track', or None
    add_noise:        bool      = True,    # add noise to image
    noise_type:       str       = 'gaussian',  # 'gaussian' or 'poisson'
    snr:              float     = 5.0,     # SNR for Gaussian noise
    rng: np.random.Generator    = None,
) -> dict:
    """
    Generate a complete synthetic OI 630 nm airglow fringe image.

    Follows Harding (2014) Eqs. 14–16 (instrument matrix path):
      1. Validate v_rel_ms against observation_mode bounds (if provided).
      2. Build r_bins (R_bins uniform points from 0 to params.r_max).
      3. Build lam_grid centred on OI_WAVELENGTH_AIR_M, spanning n_fsr FSRs
         with L_synth bins (anti-inverse-crime: L_synth=300 ≠ L_inv=101).
      4. Build instrument matrix A = build_instrument_matrix(r_bins,
         lam_grid, params, n_subpixels=8) — shape (R_bins, L_synth).
      5. Build source spectrum y = make_airglow_spectrum(lam_grid,
         v_rel_ms, Y_line, Y_bg).
      6. Compute 1D profile: s = A @ y + params.B.
      7. Wrap to 2D: image = radial_profile_to_image(s, r_bins, ...).
      8. Optionally add noise.
      9. Compute fringe_order_offset and snr_actual; assemble output dict.

    Parameters
    ----------
    params     : InstrumentParams from H01.
    v_rel_ms   : line-of-sight velocity (m/s). Positive = recession (redshift).
                 Full valid range per H01: −7700 to +1000.
                 If observation_mode is set, a tighter check is applied:
                   'cross_track' → must be in [−1000, +1000] m/s
                   'along_track' → must be in [−8000, −6000] m/s
    Y_line     : OI line intensity in ADU per wavelength bin. Default 1000.
                 Scale consistently with params.I0 (see H01 §4.4 and §5.5).
    Y_bg       : spectrally flat sky background in ADU per bin. Default 0.
    image_size : CCD active dimension in pixels. Default 256 (2×2 binned).
    cx, cy     : fringe centre in pixels. Default: geometric centre.
    R_bins     : number of radial bins. Default 500.
    L_synth    : wavelength bins for synthesis. Default 300.
                 MUST differ from H06 inversion L=101 (anti-inverse-crime).
    n_fsr      : FSRs spanned by the wavelength grid. Default 10.
                 Covers both cross-track and along-track regimes with ~3×
                 margin. Do not reduce below 4 for along-track synthesis.
    observation_mode : Optional velocity regime label.
                 'cross_track' → validates v_rel_ms in [−1000, +1000] m/s.
                 'along_track' → validates v_rel_ms in [−8000, −6000] m/s.
                 None (default) → no mode validation; full H01 range accepted.
                 Raises ValueError if v_rel_ms is outside the mode bounds.
                 Stored in output dict for traceability.
    add_noise  : if True, add noise per noise_type. Default True.
    noise_type : 'gaussian' (default, Harding §4 dark-noise model) or 'poisson'.
    snr        : SNR = ΔS/σ_N for Gaussian noise, per Harding Eq. 17. Default 5.
                 Ignored when noise_type='poisson'.
    rng        : numpy Generator for reproducibility.

    Returns
    -------
    dict with keys:
        'image_2d'            : np.ndarray (image_size, image_size) — noisy image
        'image_noiseless'     : np.ndarray (image_size, image_size) — noiseless
        'profile_1d'          : np.ndarray (R_bins,) — 1D profile (no noise)
        'r_grid'              : np.ndarray (R_bins,) — radial bin centres, px
        'lam_grid'            : np.ndarray (L_synth,) — wavelength grid, metres
        'lambda_c_m'          : float — Doppler-shifted line centre (metres),
                                computed as OI_WAVELENGTH_AIR_M*(1+v_rel/c)
        'fringe_order_offset' : int — integer FSR offset of λ_c from λ₀;
                                0 for cross-track, −1 or −2 for along-track
        'cx'                  : float — fringe centre x used
        'cy'                  : float — fringe centre y used
        'params'              : InstrumentParams used
        'v_rel_ms'            : float — v_rel used (traceability echo)
        'observation_mode'    : str or None — mode passed in
        'snr_actual'          : float — ΔS/σ_N of noiseless profile
                                (== snr if Gaussian; estimated if Poisson;
                                 np.inf if add_noise=False)

    Raises
    ------
    ValueError
        If v_rel_ms violates the observation_mode bounds.
        If v_rel_ms is outside the H01 global range [−7700, +1000].
        Propagated from make_airglow_spectrum() if λ_c falls outside lam_grid.
    """
```

---

## 4. Constants

All numerical constants are imported from `windcube/constants.py`.

```python
from windcube.constants import (
    OI_WAVELENGTH_AIR_M,         # 630.0304e-9 m — OI rest wavelength (air)
    SPEED_OF_LIGHT_MS,           # 299_792_458.0 m/s
    V_REL_CROSSTRACK_MAX_MS,     # 1000.0  m/s — cross-track bound (symmetric)
    V_REL_ALONGTRACK_MIN_MS,     # −8000.0 m/s — along-track lower bound
    V_REL_ALONGTRACK_MAX_MS,     # −6000.0 m/s — along-track upper bound
)
```

**The three regime constants must be added to `windcube/constants.py`** if
not already present (Task B verifies this). Suggested placement alongside the
existing velocity range constants, with comments:

```python
# --- Observation regime velocity bounds (H03) ---
# Cross-track (even orbits): thermospheric wind projected onto LOS
V_REL_CROSSTRACK_MAX_MS  =  1000.0   # m/s; symmetric: valid range [-1000, +1000]
# Along-track (odd orbits): spacecraft orbital velocity + wind projected onto LOS
V_REL_ALONGTRACK_MIN_MS  = -8000.0   # m/s; lower bound (maximum blueshift)
V_REL_ALONGTRACK_MAX_MS  = -6000.0   # m/s; upper bound (minimum blueshift)
```

H01 forward model objects are imported as:

```python
from fpi.airy_forward_model import (
    InstrumentParams,
    build_instrument_matrix,
    make_wavelength_grid,
    make_airglow_spectrum,
)
```

H02 image utility is imported as:

```python
from fpi.m02_calibration_synthesis import (
    radial_profile_to_image,
    add_poisson_noise,       # needed only when noise_type='poisson'
)
```

`FOCAL_LENGTH_M` is not imported. `airy_modified()` is not imported directly —
H03 uses the matrix path exclusively.

---

## 5. Implementation notes

### 5.1 L_synth and the anti-inverse-crime rule

Always pass `L = L_SYNTH = 300` to `make_wavelength_grid()`. Never pass `101`.

### 5.2 Wavelength grid width

`n_fsr = 10` default. FSR_OI ≈ 9.922 pm gives a grid half-width of 49.6 pm,
providing ~3× margin over the 16.8 pm maximum along-track Doppler shift.
For cross-track-only batches, `n_fsr = 5` is sufficient but not the default.

### 5.3 `observation_mode` validation

```python
OBSERVATION_MODE_BOUNDS = {
    'cross_track': (-V_REL_CROSSTRACK_MAX_MS, +V_REL_CROSSTRACK_MAX_MS),
    'along_track': (V_REL_ALONGTRACK_MIN_MS,   V_REL_ALONGTRACK_MAX_MS),
}

if observation_mode is not None:
    if observation_mode not in OBSERVATION_MODE_BOUNDS:
        raise ValueError(f"Unknown observation_mode '{observation_mode}'. "
                         f"Valid values: {list(OBSERVATION_MODE_BOUNDS)}")
    lo, hi = OBSERVATION_MODE_BOUNDS[observation_mode]
    if not (lo <= v_rel_ms <= hi):
        raise ValueError(
            f"v_rel_ms={v_rel_ms:.1f} m/s is outside the '{observation_mode}' "
            f"regime bounds [{lo:.0f}, {hi:.0f}] m/s."
        )
```

### 5.4 `fringe_order_offset` computation

```python
FSR_OI_M            = OI_WAVELENGTH_AIR_M**2 / (2.0 * params.t)
lambda_c_m          = OI_WAVELENGTH_AIR_M * (1.0 + v_rel_ms / SPEED_OF_LIGHT_MS)
fringe_order_offset = int(round((lambda_c_m - OI_WAVELENGTH_AIR_M) / FSR_OI_M))
```

This is a diagnostic output only — it does not affect any calculation.
Store it in the output dict alongside `lambda_c_m`.

### 5.5 Relationship between Y_line, I0, and ADU counts

With Convention B (A absorbs Δλ), the peak fringe amplitude is approximately:

```
s_peak ≈ params.I0 · Δλ · Y_line + params.B
```

For `n_fsr = 10`, `Δλ ≈ 6.6e-13 m` (10 FSR / 300 bins). With `params.I0 = 1000`
and `Y_line = 1000`, `s_peak ≈ 0.66 ADU + 300 ADU ≈ 300.66 ADU` — barely above
the bias floor. To achieve a fringe amplitude of ~100 ADU above bias, set
`Y_line ≈ 100 / (params.I0 · Δλ) ≈ 1.5e14`. For round-trip tests, any
`Y_line` that produces `snr_actual ≥ 1` suffices; adjusting `Y_line` until
a realistic SNR is achieved is the caller's responsibility.

### 5.6 SNR computation

```python
delta_S = profile_1d.max() - profile_1d.min()
if add_noise and noise_type == 'gaussian':
    sigma_N    = delta_S / snr
    snr_actual = snr          # by construction
elif add_noise and noise_type == 'poisson':
    sigma_N_eff = np.sqrt(np.mean(profile_1d))
    snr_actual  = delta_S / sigma_N_eff
else:
    snr_actual = np.inf
```

---

## 6. Verification tests

All 9 tests in `tests/test_h03_airglow_synthesis.py`.

### T1 — Output shapes and keys correct

```python
def test_output_shapes():
    """All returned arrays must have the expected shapes and the dict
    must contain all required keys, including regime-specific ones."""
    from fpi.airy_forward_model import InstrumentParams
    params = InstrumentParams()
    result = synthesise_airglow_image(
        params, v_rel_ms=0.0, add_noise=False)
    assert result['image_2d'].shape        == (256, 256)
    assert result['image_noiseless'].shape == (256, 256)
    assert result['profile_1d'].shape      == (500,)
    assert result['r_grid'].shape          == (500,)
    assert result['lam_grid'].shape        == (300,)
    for key in ('lambda_c_m', 'fringe_order_offset', 'v_rel_ms',
                'observation_mode', 'snr_actual'):
        assert key in result, f"Missing output key: '{key}'"
    assert result['v_rel_ms'] == 0.0
    assert result['observation_mode'] is None
    assert result['snr_actual'] == np.inf
```

### T2 — Noiseless image everywhere non-negative

```python
def test_image_non_negative():
    """Noiseless airglow image must be non-negative everywhere."""
    from fpi.airy_forward_model import InstrumentParams
    params = InstrumentParams()
    result = synthesise_airglow_image(
        params, v_rel_ms=100.0, add_noise=False)
    assert np.all(result['image_noiseless'] >= 0)
    assert result['image_noiseless'].min() >= params.B * 0.99
```

### T3 — Circular symmetry

```python
def test_circular_symmetry():
    """At a fixed radius, noiseless pixel values must agree to within 1%."""
    from fpi.airy_forward_model import InstrumentParams
    params = InstrumentParams()
    result = synthesise_airglow_image(
        params, v_rel_ms=0.0, add_noise=False)
    img = result['image_noiseless']
    cx, cy = result['cx'], result['cy']
    r_test = 40.0
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    values = [img[int(np.clip(np.round(cy + r_test * np.sin(a)), 0, img.shape[0]-1)),
                  int(np.clip(np.round(cx + r_test * np.cos(a)), 0, img.shape[1]-1))]
              for a in angles]
    cv = np.std(values) / np.mean(values)
    assert cv < 0.01, f"Circular symmetry broken: std/mean = {cv:.4f}"
```

### T4 — Doppler shift moves fringe inward/outward correctly

```python
def test_doppler_fringe_shift_direction():
    """
    Positive v_rel (recession) must shift fringe inward (smaller r).
    Negative v_rel must shift fringe outward (larger r).
    Consistent with H01 §4.3 velocity sign convention.
    """
    from fpi.airy_forward_model import InstrumentParams
    from scipy.signal import find_peaks
    params = InstrumentParams()

    def first_peak_r(v):
        res = synthesise_airglow_image(params, v_rel_ms=v, add_noise=False)
        peaks, _ = find_peaks(res['profile_1d'],
                              height=0.5 * res['profile_1d'].max())
        assert len(peaks) >= 1, f"No peaks found for v={v} m/s"
        return res['r_grid'][peaks[0]]

    r0 = first_peak_r(0.0)
    assert first_peak_r(+500.0) < r0, "Positive v_rel should shift fringe inward"
    assert first_peak_r(-500.0) > r0, "Negative v_rel should shift fringe outward"
```

### T5 — lambda_c computed correctly from v_rel

```python
def test_lambda_c_doppler_formula():
    """lambda_c_m must equal OI_WAVELENGTH_AIR_M × (1 + v_rel/c). [H01 Eq. 11]"""
    from fpi.airy_forward_model import InstrumentParams
    from windcube.constants import OI_WAVELENGTH_AIR_M, SPEED_OF_LIGHT_MS
    params = InstrumentParams()
    v_test = 300.0
    result = synthesise_airglow_image(params, v_rel_ms=v_test, add_noise=False)
    expected = OI_WAVELENGTH_AIR_M * (1.0 + v_test / SPEED_OF_LIGHT_MS)
    assert abs(result['lambda_c_m'] - expected) < 1e-16
```

### T6 — Gaussian noise statistics match requested SNR

```python
def test_gaussian_noise_snr():
    """Gaussian noise at snr=5 must produce σ_N = ΔS/5 to within 20%."""
    from fpi.airy_forward_model import InstrumentParams
    params = InstrumentParams()
    snr_target = 5.0
    r_noisy = synthesise_airglow_image(
        params, v_rel_ms=0.0, add_noise=True,
        noise_type='gaussian', snr=snr_target,
        rng=np.random.default_rng(42))
    r_clean = synthesise_airglow_image(
        params, v_rel_ms=0.0, add_noise=False)
    noise    = r_noisy['image_2d'] - r_clean['image_noiseless']
    profile  = r_clean['profile_1d']
    sigma_N_expected = (profile.max() - profile.min()) / snr_target
    sigma_N_actual   = np.std(noise)
    ratio = sigma_N_actual / sigma_N_expected
    assert 0.8 < ratio < 1.2, \
        f"Noise σ ratio={ratio:.3f} outside [0.8, 1.2]"
    assert abs(r_noisy['snr_actual'] - snr_target) < 0.01
```

### T7 — Reproducible with fixed seed

```python
def test_reproducible_with_seed():
    """Two calls with identical seeds must produce identical noisy images."""
    from fpi.airy_forward_model import InstrumentParams
    params = InstrumentParams()
    r1 = synthesise_airglow_image(
        params, v_rel_ms=100.0, add_noise=True, rng=np.random.default_rng(77))
    r2 = synthesise_airglow_image(
        params, v_rel_ms=100.0, add_noise=True, rng=np.random.default_rng(77))
    np.testing.assert_array_equal(r1['image_2d'], r2['image_2d'])
```

### T8 — 1D profile matches direct H01 matrix evaluation

```python
def test_profile_matches_h01_matrix():
    """
    H03 1D profile must equal A @ y_oi + B from direct H01 calls.
    Both paths must use L_synth=300 and n_fsr=10 (anti-inverse-crime).
    """
    from fpi.airy_forward_model import (
        InstrumentParams, build_instrument_matrix,
        make_wavelength_grid, make_airglow_spectrum,
    )
    from windcube.constants import OI_WAVELENGTH_AIR_M
    params  = InstrumentParams()
    R_bins  = 500
    L_synth = 300
    n_fsr   = 10.0
    v_test  = 200.0

    r_bins   = np.linspace(0, params.r_max, R_bins)
    lam_grid = make_wavelength_grid(OI_WAVELENGTH_AIR_M, n_fsr, L_synth, params)
    y_oi     = make_airglow_spectrum(lam_grid, v_rel=v_test,
                                     Y_line=1000.0, Y_bg=0.0)
    A        = build_instrument_matrix(r_bins, lam_grid, params, n_subpixels=8)
    expected = A @ y_oi + params.B

    result = synthesise_airglow_image(
        params, v_rel_ms=v_test, Y_line=1000.0, Y_bg=0.0,
        R_bins=R_bins, L_synth=L_synth, n_fsr=n_fsr, add_noise=False)
    np.testing.assert_allclose(result['profile_1d'], expected, rtol=1e-10,
        err_msg="H03 profile does not match direct H01 matrix evaluation")
```

### T9 — Two-regime fringe order offset arithmetic

```python
def test_fringe_order_offset_by_regime():
    """
    Cross-track velocities must give fringe_order_offset = 0.
    Along-track velocities must give fringe_order_offset = −1 or −2
    depending on whether the shift is closer to 1 or 2 FSRs.
    observation_mode validation must raise ValueError for out-of-range inputs.

    Verifies §2.7 fringe order offset arithmetic, the fringe_order_offset
    output key, and observation_mode input validation.
    """
    from fpi.airy_forward_model import InstrumentParams
    import pytest
    params = InstrumentParams()

    # Cross-track: all velocities in ±1000 m/s → order offset = 0
    for v in [0.0, +500.0, -500.0, +1000.0, -1000.0]:
        r = synthesise_airglow_image(params, v_rel_ms=v,
                                     observation_mode='cross_track',
                                     add_noise=False)
        assert r['fringe_order_offset'] == 0, \
            f"Cross-track v={v} m/s: expected offset 0, got {r['fringe_order_offset']}"

    # Along-track in the −1 order range (−6000 to ~−7400 m/s)
    for v in [-6000.0, -6500.0, -7000.0]:
        r = synthesise_airglow_image(params, v_rel_ms=v,
                                     observation_mode='along_track',
                                     add_noise=False)
        assert r['fringe_order_offset'] == -1, \
            f"Along-track v={v} m/s: expected offset −1, got {r['fringe_order_offset']}"

    # Along-track in the −2 order range (~−7400 to −8000 m/s)
    for v in [-7500.0, -8000.0]:
        r = synthesise_airglow_image(params, v_rel_ms=v,
                                     observation_mode='along_track',
                                     add_noise=False)
        assert r['fringe_order_offset'] == -2, \
            f"Along-track v={v} m/s: expected offset −2, got {r['fringe_order_offset']}"

    # observation_mode validation: wrong velocity for mode must raise ValueError
    with pytest.raises(ValueError, match="along_track"):
        synthesise_airglow_image(params, v_rel_ms=+200.0,
                                 observation_mode='along_track',
                                 add_noise=False)
    with pytest.raises(ValueError, match="cross_track"):
        synthesise_airglow_image(params, v_rel_ms=-7000.0,
                                 observation_mode='cross_track',
                                 add_noise=False)
```

---

## 7. Expected numerical values

For `InstrumentParams()` defaults, `image_size=256`, `R_bins=500`,
`L_synth=300`, `n_fsr=10`, `Y_line=1000`, `Y_bg=0`:

| Quantity | Expected | Notes | Test |
|---|---|---|---|
| Image shape | (256, 256) | 2×2 binned | T1 |
| Profile shape | (500,) | R_bins default | T1 |
| lam_grid shape | (300,) | L_synth | T1 |
| All noiseless pixel values | ≥ 0 | bias floor | T2 |
| Circular symmetry CV | < 0.01 | | T3 |
| +500 m/s innermost peak | smaller r than v=0 | recession → inward | T4 |
| λ_c at v=300 m/s | 630.0304 × (1 + 300/c) nm | Eq. 11 | T5 |
| Gaussian noise σ/σ_expected | 0.8–1.2 | Harding Eq. 17 | T6 |
| Profile vs H01 direct matrix | rtol < 1e-10 | Eq. 16 | T8 |
| FSR at OI wavelength | ≈ 9.922 pm | λ²/(2t) at t=20.008 mm | T9 |
| `fringe_order_offset` at v=0 | 0 | cross-track | T9 |
| `fringe_order_offset` at −6000 to −7400 m/s | −1 | along-track | T9 |
| `fringe_order_offset` at −7400 to −8000 m/s | −2 | along-track | T9 |
| −1/−2 order boundary | ≈ −7400 m/s | 1.5 FSR threshold | T9 |
| Max Doppler shift (8000 m/s) | ≈ 16.8 pm ≈ 1.69 FSR | within 10-FSR grid | §5.2 |

---

## 8. File locations in repository

```
soc_sewell/
├── windcube/
│   └── constants.py                              ← add V_REL_CROSSTRACK_MAX_MS,
│                                                    V_REL_ALONGTRACK_MIN/MAX_MS
├── src/fpi/
│   ├── airy_forward_model_2026_05_05.py          ← H01
│   ├── m02_calibration_synthesis_2026_05_05.py   ← H02
│   └── m03_airglow_synthesis_2026_05_05.py       ← this module
├── tests/
│   └── test_h03_airglow_synthesis.py
└── docs/specs/
    └── H03_airglow_synthesis_2026-05-05.md       ← this file
```

---

## 9. Instructions for Claude Code

### Preamble — read before touching any file

1. Read this entire spec (H03).
2. Read H01 (`docs/specs/H01_airy_forward_model_2026-05-05.md`) — §2.4
   (anti-inverse-crime), §4.3 (OI source model, Eqs. 10–11), §4.4
   (instrument matrix, Eqs. 14–16), and §6 (function signatures).
3. Read H02 (`docs/specs/H02_calibration_synthesis_2026-05-05.md`) — H03
   imports `radial_profile_to_image` and `add_poisson_noise` from H02.
4. Read `windcube/constants.py` in full.
5. Check whether `src/fpi/m03_airglow_synthesis_*.py` already exists.

Report findings from steps 4 and 5 before proceeding.

### Task sequence

**TASK A — Confirm H01 and H02 tests pass**

```bash
pytest tests/test_airy_forward_model*.py tests/test_h02_calibration_synthesis*.py -v --tb=short
```

All 23 tests (15 + 8) must pass. Stop and report any failures.

**TASK B — Verify and add constants**

Confirm `windcube/constants.py` exports:
- `OI_WAVELENGTH_AIR_M`       — report current value
- `SPEED_OF_LIGHT_MS`         — report current value
- `V_REL_CROSSTRACK_MAX_MS`   (1000.0 m/s) — add if absent
- `V_REL_ALONGTRACK_MIN_MS`   (−8000.0 m/s) — add if absent
- `V_REL_ALONGTRACK_MAX_MS`   (−6000.0 m/s) — add if absent

If any of the three regime constants are missing, add them with the comments
from Section 4. Commit separately:
```
feat(constants): add observation regime velocity bounds for H03
```

Confirm `FOCAL_LENGTH_M` is **not** imported by H03.

**TASK C — Create module**

Create `src/fpi/m03_airglow_synthesis_2026_05_05.py`. Functions in order:
1. `add_gaussian_noise` (§3.1)
2. `synthesise_airglow_image` (§3.2)

Critical implementation checklist:
- [ ] `radial_profile_to_image` and `add_poisson_noise` imported from H02 — not reimplemented
- [ ] `build_instrument_matrix`, `make_wavelength_grid`, `make_airglow_spectrum` from H01
- [ ] Five constants from `windcube.constants` per Section 4
- [ ] Default `n_fsr = 10.0` (not 5.0)
- [ ] Default `L_synth = 300` (never 101)
- [ ] `observation_mode` validation via `OBSERVATION_MODE_BOUNDS` dict (§5.3)
- [ ] `fringe_order_offset` computed per §5.4 and included in output dict
- [ ] `lambda_c_m` from formula `OI_WAVELENGTH_AIR_M * (1.0 + v_rel_ms / SPEED_OF_LIGHT_MS)`, not from grid argmax
- [ ] `n_subpixels=8` passed to `build_instrument_matrix()`
- [ ] `snr_actual = np.inf` when `add_noise=False`
- [ ] `observation_mode` echoed in output dict

Module docstring:
```python
"""
Module:      m03_airglow_synthesis_2026_05_05.py
Spec:        docs/specs/H03_airglow_synthesis_2026-05-05.md
Author:      Claude Code
Generated:   2026-05-05
Last tested: 2026-05-05
Project:     WindCube FPI Pipeline — NCAR/HAO
Repo:        soc_sewell

Synthesises a 2D OI 630 nm airglow fringe image following Harding (2014)
Eqs. 10–11, 14–16 (instrument matrix path). Supports two observation regimes:
  - Cross-track (even orbits): v_rel in [−1000, +1000] m/s; fringe_order_offset=0
  - Along-track (odd orbits):  v_rel in [−8000, −6000] m/s; fringe_order_offset=−1 or −2

Key design choices:
  - L_synth=300; H06 inversion uses L=101 (anti-inverse-crime, H01 §2.4)
  - n_fsr=10 default covers both regimes with ~3× margin
  - Delta-function OI source; Harding Eq. 12 (temperature) excluded
  - radial_profile_to_image imported from H02; not reimplemented
  - Gaussian noise default (dark-noise dominated), per Harding §4
"""
```

**TASK D — Write tests**

Create `tests/test_h03_airglow_synthesis.py` with all 9 tests from Section 6.

For T9: verify the −1/−2 order boundary numerically before submitting.
With `params.t = 20.008e-3 m`, `FSR_OI ≈ 9.922 pm`, and the 1.5-FSR
threshold at `v ≈ −7388 m/s`. The test uses −7500 m/s and −8000 m/s for
the −2 region — confirm these are above the threshold magnitude.

**TASK E — Run module tests**

```bash
pytest tests/test_h03_airglow_synthesis.py -v --tb=short
```

All 9 tests must pass. Failure guide:

| Test | Likely cause |
|---|---|
| T1 (shapes/keys) | Missing `fringe_order_offset` or `observation_mode` in output |
| T2 (non-negative) | B not added to profile; or Y_bg sign error |
| T3 (symmetry) | `radial_profile_to_image` reimplemented incorrectly |
| T4 (shift direction) | v_rel sign convention inverted |
| T5 (lambda_c) | `lambda_c_m` taken from `lam_grid[argmax(y)]` instead of formula |
| T6 (noise SNR) | `sigma_N` computed from image std instead of profile ΔS |
| T7 (seed) | `rng` not passed through to `add_gaussian_noise` |
| T8 (matrix match) | `n_subpixels` or `n_fsr` differs between H03 and direct path |
| T9 (regime offsets) | FSR_OI uses wrong `t`; or `round()` applied to wrong quantity |

**TASK F — Full test suite**

```bash
pytest tests/ -v --tb=short
```

No regressions permitted.

**TASK G — Commit**

```
feat(H03): airglow fringe synthesis, two observation regimes, 9/9 tests pass
Implements: H03_airglow_synthesis_2026-05-05.md
Cross-track (offset=0) and along-track (offset=-1/-2) regimes documented and tested.
```

### Report format (paste back to Claude.ai)

```
TASK A — Prior tests
  H01 tests: N/15 pass
  H02 tests: N/8 pass

TASK B — Constants check
  OI_WAVELENGTH_AIR_M: present — value: [value]
  SPEED_OF_LIGHT_MS: present
  V_REL_CROSSTRACK_MAX_MS: present / added — value: [value]
  V_REL_ALONGTRACK_MIN_MS: present / added — value: [value]
  V_REL_ALONGTRACK_MAX_MS: present / added — value: [value]
  Constants commit (if additions): [hash or 'not needed']
  FOCAL_LENGTH_M not imported: Yes / No

TASK C — Module created
  Source file: src/fpi/m03_airglow_synthesis_2026_05_05.py
  n_fsr default = 10.0: Yes / No
  L_synth default = 300: Yes / No
  observation_mode validation: Yes / No
  fringe_order_offset in output: Yes / No
  lambda_c_m from formula: Yes / No
  n_subpixels=8 passed: Yes / No
  snr_actual=inf when no noise: Yes / No

TASK D — Test file
  All 9 tests present: Yes / No
  T9 boundary confirmed numerically: Yes / No

TASK E — Module tests
  Result: N/9 pass
  Failures: [list]

TASK F — Full suite
  Result: N/N pass
  Unexpected failures: [list]

TASK G — Commit hash: [hash]
```
