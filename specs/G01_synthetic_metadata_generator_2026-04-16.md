# G01 — Synthetic Metadata Generator Specification

**Spec ID:** G01
**Spec file:** `docs/specs/G01_synthetic_metadata_generator_2026-04-16.md`
**Project:** WindCube FPI Science Operations Center Pipeline
**Institution:** NCAR / High Altitude Observatory (HAO)
**Status:** Draft — awaiting v9 implementation (C18–C21 new; binary synthesis added)
**Depends on:**
  - NB00 (`nb00_wind_map_2026_04_06.py`) — `WindMap`, `UniformWindMap`,
    `AnalyticWindMap`, `HWM14WindMap`, `StormWindMap`
  - NB01 (`nb01_orbit_propagator_2026_04_16.py`) — `propagate_orbit(t_start, duration_s, dt_s)`
  - NB02a (`nb02a_boresight_2026_04_16.py`) — `compute_los_eci(pos_eci, vel_eci, look_mode, h_target_km)`
  - NB02b (`nb02b_tangent_point_2026_04_16.py`) — `compute_tangent_point(pos_eci, los_eci, epoch, h_target_km)`
  - NB02c (`nb02c_los_projection_2026_04_16.py`) — `compute_v_rel(wind_map, tp_lat_deg, tp_lon_deg, tp_eci, vel_eci, los_eci, epoch)`
  - P01 (`p01_image_metadata_2026_04_06.py`) — `ImageMetadata`, `AdcsQualityFlags`,
    `compute_adcs_quality_flag()`
  - `src/constants.py` — `WGS84_A_M`, `EARTH_GRAV_PARAM_M3_S2`
  - `tkinter` (stdlib) — native folder-browser dialog
**Used by:**
  - Z02 (synthetic airglow image generator) — may now be superseded by G01 for
    simple uniform/analytic wind maps; Z02 remains authoritative for HWM14/storm
  - Z03 (synthetic neon calibration image generator) — now inline in G01
**References:**
  - **WindCube Mission CONOPS Document**
    Document ID: `[TBD — insert document number, e.g. WC-OPS-XXXX]`
    Issue/version: `[TBD — insert version, e.g. Issue 1.0, 2025-MM-DD]`
    Sections referenced: observation schedule, science band, calibration cadence,
    dark frame strategy, exposure time budget
  - SI-UCAR-WC-RP-004 Issue 1.0 — AOCS Design Report (BRF/THRF/SIRF frames)
  - Drob et al. (2015), doi:10.1002/2014EA000089 — HWM14 empirical model
  - Burns, Adams & Longwell (1950) IAU neon spectroscopic standards — λ₁, λ₂
**Last updated:** 2026-04-16

> **CONOPS version note:** §3 observation schedule parameters and §5 exposure
> time defaults are CONOPS-driven. Any CONOPS update requires a dated revision
> of this spec before Claude Code is re-run.

> **Revision history:**
> - v1–v6 (2026-04-16): See previous revision notes.
> - v7 (2026-04-16): `tkinter` folder-browser dialog; `_pick_folder()`.
> - v8 (2026-04-16): NB02b tangent point + NB00 wind sampling for science
>   frames. Wind map registry. CSV 42 columns. C12 updated; C14–C17 added.
> - v9 (2026-04-16): **NB02c LOS velocity decomposition + binary image
>   synthesis.** NB02c `compute_v_rel()` called for science frames; replaces
>   the direct `wind_map.sample()` call (NB02c returns wind components too).
>   Four new CSV columns (43–46): `v_wind_los_ms`, `v_earth_los_ms`,
>   `v_sc_los_ms`, `v_rel_ms`. `truth_v_los` in `ImageMetadata` now populated.
>   CSV extended to 46 columns. New §7: binary image synthesis — one `.bin`
>   file per frame (science: Airy fringe at v_rel; cal: two-line neon; dark:
>   dark current + read noise). `_encode_header()` is the exact inverse of P01
>   `parse_header()`. C12 updated (42→46); C18–C21 added.

---

## 1. Purpose

G01 is a standalone, interactive Python script that pre-computes and saves
the complete AOCS/instrument metadata array **and** a corresponding set of
synthetic binary FPI image files for a WindCube observation campaign.

**What G01 produces per frame type (v9):**

| Frame type | NB01 | NB02a | NB02b | NB02c | Image synthesis |
|------------|------|-------|-------|-------|-----------------|
| science | ✓ orbit state | ✓ `q`, `los_eci` | ✓ tangent point | ✓ `v_rel`, LOS decomposition | Airy fringe at `v_rel` Doppler |
| cal | ✓ orbit state | ✓ `q`, `los_eci` | — | — | Two-line neon Airy pattern |
| dark | ✓ orbit state | ✓ `q`, `los_eci` | — | — | Dark current + read noise |

**What G01 does not produce:**
- `truth_v_los` was previously `None`; it is now populated from NB02c
  `v_wind_LOS` for science frames.
- Nothing else from the "not computed" list changes.

**Storage note:** At 10 s cadence over 30 days, G01 produces ~120,000 `.bin`
files × 143,520 bytes each ≈ **17.3 GB** of binary image data. Ensure
adequate disk space before running with `duration_days = 30`.

---

## 2. User interface — interactive prompts

Unchanged from v8. Shown here for completeness.

```
=== G01 — WindCube Synthetic Metadata Generator ===

Start epoch          [2027-01-01T00:00:00 UTC]  : _
Duration             [days,  default  30       ] : _
Science band         [deg,   default  40       ] : _
Obs. cadence         [sec,   default  10       ] : _
Cal/dark frames (n)  [int,   default   5       ] : _
Exposure time        [cts,   default 8000      ] : _
S/C altitude         [km,    default 510       ] : _
Tangent height       [km,    default 250       ] : _
Random seed          [int,   default  42       ] : _

Select wind map for science frame truth winds:
  [1] Uniform constant        (v_zonal, v_merid — specify values)
  [2] Analytic sine_lat       (A_z·sin(lat), A_m·cos(lat))
  [3] Analytic wave4 / DE3    (4-wave lat×lon pattern)
  [4] HWM14 quiet-time        (requires hwm14 package)
  [5] HWM14 storm / DWM07     (requires hwm14 package)
Choice [default 1]: _

Select output folder (dialog opening — check taskbar if not visible)...
→ Selected: C:\Users\sewell\WindCube\G01_outputs
```

Sub-prompts and parameter table are unchanged from v8 (§2.1–§2.3).

---

## 3. Wind map registry

Unchanged from v8 (§3). `WIND_MAP_REGISTRY` and `WIND_MAP_TAGS` with
5 builders. Extension by adding one entry to each dict.

---

## 4. Orbit propagation and CONOPS scheduling

Unchanged from v8 (§4.1–§4.5). NB01 at `SCHED_DT_S = 10.0` s; science at
`±lat_band_deg`; cal/dark triggered at `CAL_TRIGGER_LAT_DEG = 60.0°N`
ascending.

---

## 5. Main metadata loop — per-frame computation

The loop is updated in v9 to replace the direct `wind_map.sample()` call with
`compute_v_rel()`, which returns the wind components alongside the full LOS
decomposition. Only science frames call NB02b and NB02c.

```python
from src.geometry.nb02a_boresight_2026_04_16 import compute_los_eci
from src.geometry.nb02b_tangent_point_2026_04_16 import compute_tangent_point
from src.geometry.nb02c_los_projection_2026_04_16 import compute_v_rel

metadata_list = []
bin_dir = pathlib.Path(output_dir) / "bin_frames"
bin_dir.mkdir(parents=True, exist_ok=True)

for frame_i, (idx, frame_type) in enumerate(zip(obs_indices, frame_types)):
    row       = df_sched.loc[idx]
    pos       = np.array([row.pos_eci_x, row.pos_eci_y, row.pos_eci_z])
    vel       = np.array([row.vel_eci_x, row.vel_eci_y, row.vel_eci_z])
    look_mode = row.look_mode
    epoch_t   = Time(row.epoch)

    # ── NB02a: attitude quaternion + LOS vector ───────────────────────────
    # los_eci is retained — it is an input to both NB02b and NB02c
    los_eci, q = compute_los_eci(pos, vel, look_mode, h_target_km=h_target_km)

    # ── RNG draws — metadata noise (fixed order, 9 draws total) ──────────
    theta        = rng.normal(0.0, SIGMA_POINTING_RAD)          # draw 1
    raw          = rng.standard_normal(3)                        # draws 2-4
    n_hat        = raw / np.linalg.norm(raw)
    half_t       = theta / 2.0
    qe           = [n_hat[0]*np.sin(half_t), n_hat[1]*np.sin(half_t),
                    n_hat[2]*np.sin(half_t), np.cos(half_t)]
    qe           = [c / np.linalg.norm(qe) for c in qe]
    etalon_temps = rng.normal(ETALON_TEMP_MEAN_C, ETALON_TEMP_STD_C, 4)  # draws 5-8
    ccd_temp1    = float(rng.normal(CCD_TEMP_MEAN_C, CCD_TEMP_STD_C))    # draw 9

    # ── NB02b + NB02c: tangent point + full LOS decomposition ────────────
    # (science frames only; NB02c replaces the direct wind_map.sample() call)
    tp_lat = tp_lon = tp_alt = None
    v_zonal = v_merid = None
    v_wind_LOS = v_earth_LOS = V_sc_LOS = v_rel = None

    if frame_type == 'science':
        tp = compute_tangent_point(pos, los_eci, epoch_t, h_target_km=h_target_km)
        tp_lat = tp['tp_lat_deg']
        tp_lon = tp['tp_lon_deg']
        tp_alt = tp['tp_alt_km']

        vr = compute_v_rel(
            wind_map,
            tp_lat, tp_lon, tp['tp_eci'],
            vel, los_eci, epoch_t,
        )
        # NB02c returns all six keys; use them to populate both metadata and CSV
        v_wind_LOS = vr['v_wind_LOS']
        v_earth_LOS = vr['v_earth_LOS']
        V_sc_LOS   = vr['V_sc_LOS']
        v_rel      = vr['v_rel']
        v_zonal    = vr['v_zonal_ms']   # wind_map.sample() result via NB02c
        v_merid    = vr['v_merid_ms']

    # ── Instrument state, ADCS flag ───────────────────────────────────────
    gpio, lamp, exp_time_cs = _instrument_state(frame_type)
    adcs_flag = compute_adcs_quality_flag({
        'pointing_error': qe,
        'pos_eci_hat':    pos.tolist(),
        'adcs_timestamp': int(row.epoch.timestamp() * 1000),
    })

    # ── Construct ImageMetadata ───────────────────────────────────────────
    meta = ImageMetadata(
        # ... all fields per §6 table ...
        truth_v_los        = v_wind_LOS,   # now populated for science frames
        tangent_lat        = tp_lat,
        tangent_lon        = tp_lon,
        tangent_alt_km     = tp_alt,
        truth_v_zonal      = v_zonal,
        truth_v_meridional = v_merid,
        noise_seed         = frame_i,
        is_synthetic       = True,
    )
    metadata_list.append(meta)

    # ── Binary image synthesis and file write (§7) ────────────────────────
    pixels_256 = _generate_pixels(frame_type, v_rel, ccd_temp1,
                                  exp_time_cts, rng)  # pixel RNG draws after draw 9
    _write_bin_file(meta, pixels_256, bin_dir / _bin_filename(meta))
```

---

## 6. `ImageMetadata` field assignment

Fields changed from `None` in v8 are marked **UPDATED**.

| Field | Source | Value / formula |
|-------|--------|-----------------|
| `rows` | constant | `260` |
| `cols` | constant | `276` |
| `exp_time` | §4.5 | `round(exp_time_cts × 0.001 × 100)` centiseconds |
| `exp_unit` | §4.5 | `38500` (hardware register, fixed) |
| `binning` | constant | `2` |
| `img_type` | §4.4 | `_classify_img_type(lamp_ch_array, gpio_pwr_on)` |
| `lua_timestamp` | NB01 `epoch` | `int(row.epoch.timestamp() * 1000)` ms |
| `adcs_timestamp` | NB01 `epoch` | `= lua_timestamp` |
| `utc_timestamp` | NB01 `epoch` | `row.epoch.isoformat()` |
| `spacecraft_latitude` | NB01 `lat_deg` | `np.radians(row.lat_deg)` rad |
| `spacecraft_longitude` | NB01 `lon_deg` | `np.radians(row.lon_deg)` rad |
| `spacecraft_altitude` | NB01 `alt_km` | `row.alt_km × 1e3` m |
| `pos_eci_hat` | NB01 `pos_eci_*` | `[pos_eci_x, pos_eci_y, pos_eci_z]` m |
| `vel_eci_hat` | NB01 `vel_eci_*` | `[vel_eci_x, vel_eci_y, vel_eci_z]` m/s |
| `attitude_quaternion` | NB02a | `q`, scalar-last `[x,y,z,w]` |
| `pointing_error` | §4.1 noise | Gaussian error quaternion, σ = 5 arcsec |
| `obs_mode` | NB01 parity | `'along_track'` / `'cross_track'` |
| `ccd_temp1` | noise | `rng.normal(-10.0, 1.0)` °C |
| `etalon_temps` | noise | `rng.normal(24.0, 0.1, 4).tolist()` °C |
| `shutter_status` | §4.4 | derived from `gpio_pwr_on` |
| `gpio_pwr_on` | §4.2–4.3 | frame-type dependent |
| `lamp_ch_array` | §4.2–4.3 | frame-type dependent |
| `lamp1/2/3_status` | P01 rule | derived from `lamp_ch_array` |
| `orbit_number` | §4.1 | 1-based |
| `frame_sequence` | §4.5 | 0-based within orbit |
| `orbit_parity` | §4.1 | `'along_track'` / `'cross_track'` |
| `adcs_quality_flag` | P01 | `compute_adcs_quality_flag(...)` |
| `is_synthetic` | constant | `True` |
| `noise_seed` | frame index | 0-based in `obs_indices` |
| `tangent_lat` | NB02b (science) | `tp['tp_lat_deg']`; `None` for cal/dark |
| `tangent_lon` | NB02b (science) | `tp['tp_lon_deg']`; `None` for cal/dark |
| `tangent_alt_km` | NB02b (science) | `tp['tp_alt_km']`; `None` for cal/dark |
| `truth_v_zonal` | NB02c (science) | `vr['v_zonal_ms']`; `None` for cal/dark |
| `truth_v_meridional` | NB02c (science) | `vr['v_merid_ms']`; `None` for cal/dark |
| `truth_v_los` **UPDATED** | NB02c (science) | `vr['v_wind_LOS']`; `None` for cal/dark |
| `etalon_gap_mm` | — | `None` |
| All other Optional fields | — | `None` / defaults |

---

## 7. Binary image synthesis

### 7.1 Constants

```python
# FPI optical model (S03 authoritative values)
LAMBDA_OI_M      = 630.0e-9       # OI 630.0 nm source wavelength, m
LAMBDA_NE1_M     = 640.2248e-9    # Neon strong line (Burns et al. 1950)
LAMBDA_NE2_M     = 638.2991e-9    # Neon weak line  (Burns et al. 1950)
ETALON_GAP_M     = 20.106e-3      # Tolansky-recovered gap (S03, mm → m)
FOCAL_LENGTH_M   = 0.19912        # Imaging lens focal length, m
PLATE_SCALE_RPX  = 1.6071e-4      # rad/px (2×2 binned, Tolansky)
R_REFL           = 0.53           # Effective reflectivity (FlatSat)
N_GAP            = 1.0            # Refractive index of etalon gap (air)
C_LIGHT_MS       = 2.99792458e8   # Speed of light, m/s

# Finesse coefficient: F = 4R/(1-R)²
FINESSE_F        = 4 * R_REFL / (1 - R_REFL)**2    # ≈ 9.6 for R=0.53

# CCD / pixel layout
NX_PIX           = 256   # science region width (2×2 binned)
NY_PIX           = 256   # science region height
N_ROWS_BIN       = 259   # pixel rows in binary file (rows 1–259)
N_COLS_BIN       = 276   # pixel columns in binary file
ROW_OFFSET_PIX   = 1     # row within pixel array where science image starts
COL_OFFSET_PIX   = 10    # column offset (centres 256 in 276)
BIAS_ADU         = 100   # CCD bias level, ADU (uniform background)
ADU_MAX          = 16383 # 14-bit ADC maximum

# Science frame signal level
SCI_PEAK_ADU     = 5000  # peak Airy fringe ADU for OI 630 nm at default exposure
SCI_READ_NOISE   = 5.0   # read noise, ADU rms

# Calibration frame signal level
CAL_PEAK_ADU     = 12000 # peak ADU for neon bright fringes
CAL_NE_RATIO     = 3.0   # intensity ratio strong (λ1) : weak (λ2)
CAL_READ_NOISE   = 5.0

# Dark frame noise model
DARK_REF_ADU_S   = 0.05  # dark current, ADU/px/s at T_REF_DARK_C
T_REF_DARK_C     = -20.0 # reference temperature for dark model, °C
T_DOUBLE_C       = 6.5   # temperature interval for dark current to double, °C
DARK_READ_NOISE  = 5.0
```

### 7.2 Binary header encoder — `_encode_header()`

The exact inverse of P01 `parse_header()`. Takes an `ImageMetadata` and
returns a `np.ndarray` of shape `(276,)`, `dtype=">u2"` (big-endian uint16).

**Mixed-endian encoding helpers:**

```python
import struct

def _encode_u64(val: int) -> list[int]:
    """uint64 → 4 BE uint16 words in LE word order (LSW first)."""
    return [(val >> (16 * i)) & 0xFFFF for i in range(4)]

def _encode_f64(val: float) -> list[int]:
    """float64 → 4 BE uint16 words in LE word order (LSW first)."""
    b     = struct.pack(">d", val)            # 8 bytes, big-endian double
    words = struct.unpack(">4H", b)           # [MSW, w1, w2, LSW]
    return list(reversed(words))              # [LSW, w2, w1, MSW] — LE word order
```

**Verification:** `_f64(h, w)` in P01 does `struct.pack(">4H", *reversed(h[w:w+4]))` then
`struct.unpack(">d", ...)`. Encoding reverses this exactly. Round-trip error must be zero.

```python
def _encode_header(meta: ImageMetadata) -> np.ndarray:
    """
    Encode ImageMetadata into the 276-word binary header row.

    Quaternion convention: P01 pipeline uses scalar-last [x,y,z,w];
    binary header stores scalar-first [w,x,y,z]. Re-ordering performed here
    for both attitude_quaternion and pointing_error (words 28–59).
    """
    h = np.zeros(276, dtype=">u2")

    # Words 0–3: image dimensions and timing register
    h[0] = meta.rows
    h[1] = meta.cols
    h[2] = meta.exp_time          # centiseconds
    h[3] = meta.exp_unit          # hardware register (38500)

    # Words 4–7: ccd_temp1 (float64)
    for i, w in enumerate(_encode_f64(meta.ccd_temp1)):
        h[4 + i] = w

    # Words 8–11: lua_timestamp (uint64)
    for i, w in enumerate(_encode_u64(meta.lua_timestamp)):
        h[8 + i] = w

    # Words 12–15: adcs_timestamp (uint64)
    for i, w in enumerate(_encode_u64(meta.adcs_timestamp)):
        h[12 + i] = w

    # Words 16–27: geodetic position (lat, lon, alt — each float64)
    for j, val in enumerate([meta.spacecraft_latitude,
                              meta.spacecraft_longitude,
                              meta.spacecraft_altitude]):
        for i, w in enumerate(_encode_f64(val)):
            h[16 + j*4 + i] = w

    # Words 28–43: attitude quaternion (4 × float64)
    # Pipeline [x,y,z,w] → binary [w,x,y,z]
    q = meta.attitude_quaternion               # [x, y, z, w]
    q_bin = [q[3], q[0], q[1], q[2]]          # [w, x, y, z]
    for j, val in enumerate(q_bin):
        for i, w in enumerate(_encode_f64(val)):
            h[28 + j*4 + i] = w

    # Words 44–59: pointing error quaternion (4 × float64)
    qe = meta.pointing_error                   # [x, y, z, w]
    qe_bin = [qe[3], qe[0], qe[1], qe[2]]
    for j, val in enumerate(qe_bin):
        for i, w in enumerate(_encode_f64(val)):
            h[44 + j*4 + i] = w

    # Words 60–71: ECI position (3 × float64)
    for j, val in enumerate(meta.pos_eci_hat):
        for i, w in enumerate(_encode_f64(val)):
            h[60 + j*4 + i] = w

    # Words 72–83: ECI velocity (3 × float64)
    for j, val in enumerate(meta.vel_eci_hat):
        for i, w in enumerate(_encode_f64(val)):
            h[72 + j*4 + i] = w

    # Words 84–99: etalon temperatures (4 × float64)
    for j, val in enumerate(meta.etalon_temps):
        for i, w in enumerate(_encode_f64(val)):
            h[84 + j*4 + i] = w

    # Words 100–103: gpio_pwr_on (4 × uint8 in low byte)
    for j, val in enumerate(meta.gpio_pwr_on):
        h[100 + j] = int(val) & 0xFF

    # Words 104–109: lamp_ch_array (6 × uint8 in low byte)
    for j, val in enumerate(meta.lamp_ch_array):
        h[104 + j] = int(val) & 0xFF

    # Words 110–275: padding (already zero from np.zeros)
    return h
```

### 7.3 Pixel image generators

All generators return a `np.ndarray` of shape `(256, 256)`, `dtype=np.uint16`,
with values clipped to `[0, ADU_MAX]`. The 256×256 array is embedded into the
259×276 binary pixel area by `_write_bin_file()`.

#### 7.3.1 Science frame — Airy fringe at `v_rel` Doppler

Uses the delta-function OI source model (no temperature broadening; temperature
is not a science product). The fringe shift is applied through the observed
wavelength:

```
λ_obs = LAMBDA_OI_M × (1 + v_rel_ms / C_LIGHT_MS)
```

Positive `v_rel` (recession) → λ increases → fringes shift to smaller radii.

```python
def _generate_science_pixels(v_rel_ms: float, rng) -> np.ndarray:
    """
    Generate OI 630 nm Airy fringe image with Doppler shift v_rel_ms.

    Parameters
    ----------
    v_rel_ms : float — Doppler velocity from NB02c, m/s.
    rng      : numpy.random.Generator

    Physics
    -------
    Phase at pixel radius r_px:
        θ = r_px × PLATE_SCALE_RPX              (angle from optical axis, rad)
        λ_obs = LAMBDA_OI_M × (1 + v_rel_ms/C)
        δ = 4π × N_GAP × ETALON_GAP_M × cos(θ) / λ_obs
        I_airy = 1 / (1 + FINESSE_F × sin²(δ/2))

    Signal model:
        signal(r) = SCI_PEAK_ADU × I_airy(r)
        N_px ~ Poisson(signal(r))   (photon noise)
        + Normal(0, SCI_READ_NOISE) (read noise)
        + BIAS_ADU                  (CCD bias)

    Fringe centre: (NX_PIX/2, NY_PIX/2) in the 256×256 image.
    """
    lambda_obs = LAMBDA_OI_M * (1.0 + v_rel_ms / C_LIGHT_MS)

    cx, cy = NX_PIX / 2.0, NY_PIX / 2.0
    x = np.arange(NX_PIX) - cx
    y = np.arange(NY_PIX) - cy
    XX, YY = np.meshgrid(x, y)
    r_px = np.sqrt(XX**2 + YY**2)

    theta = r_px * PLATE_SCALE_RPX
    delta = 4.0 * np.pi * N_GAP * ETALON_GAP_M * np.cos(theta) / lambda_obs
    I_airy = 1.0 / (1.0 + FINESSE_F * np.sin(delta / 2.0)**2)

    signal = SCI_PEAK_ADU * I_airy
    photon = rng.poisson(np.clip(signal, 0, None))          # Poisson shot noise
    read   = rng.normal(0.0, SCI_READ_NOISE, size=signal.shape)
    image  = np.round(photon + read + BIAS_ADU).astype(np.float32)
    return np.clip(image, 0, ADU_MAX).astype(np.uint16)
```

#### 7.3.2 Calibration frame — two-line neon Airy pattern

Two superimposed Airy patterns at the neon wavelengths (Burns et al. 1950
IAU standards, confirmed). Weighted sum with `CAL_NE_RATIO : 1` (strong:weak).

```python
def _generate_cal_pixels(rng) -> np.ndarray:
    """
    Generate two-line neon calibration fringe image.

    Lines: λ1 = 640.2248 nm (strong), λ2 = 638.2991 nm (weak).
    Combined intensity: (CAL_NE_RATIO × I_airy(λ1) + I_airy(λ2)) /
                        (CAL_NE_RATIO + 1)
    Peak scaled to CAL_PEAK_ADU.
    """
    cx, cy = NX_PIX / 2.0, NY_PIX / 2.0
    x = np.arange(NX_PIX) - cx
    y = np.arange(NY_PIX) - cy
    XX, YY = np.meshgrid(x, y)
    r_px   = np.sqrt(XX**2 + YY**2)
    theta  = r_px * PLATE_SCALE_RPX

    def _airy(lam):
        delta = 4.0 * np.pi * N_GAP * ETALON_GAP_M * np.cos(theta) / lam
        return 1.0 / (1.0 + FINESSE_F * np.sin(delta / 2.0)**2)

    I_cal = (CAL_NE_RATIO * _airy(LAMBDA_NE1_M) + _airy(LAMBDA_NE2_M)) \
            / (CAL_NE_RATIO + 1.0)

    signal = CAL_PEAK_ADU * I_cal
    photon = rng.poisson(np.clip(signal, 0, None))
    read   = rng.normal(0.0, CAL_READ_NOISE, size=signal.shape)
    image  = np.round(photon + read + BIAS_ADU).astype(np.float32)
    return np.clip(image, 0, ADU_MAX).astype(np.uint16)
```

#### 7.3.3 Dark frame — dark current + read noise

Dark current scales exponentially with CCD temperature; the rate doubles
every `T_DOUBLE_C = 6.5°C`. The mean dark signal per pixel is:

```
dark_rate [ADU/px/s] = DARK_REF_ADU_S × 2^((ccd_temp1 - T_REF_DARK_C) / T_DOUBLE_C)
N_dark = dark_rate × exp_time_s
```

```python
def _generate_dark_pixels(ccd_temp1_c: float, exp_time_s: float, rng) -> np.ndarray:
    """
    Generate dark frame based on CCD temperature and exposure time.

    Model:
        dark_rate = DARK_REF_ADU_S × 2^((T - T_REF_DARK_C) / T_DOUBLE_C)
        mean dark signal per pixel = dark_rate × exp_time_s
        N_px ~ Poisson(mean_dark) + Normal(0, DARK_READ_NOISE) + BIAS_ADU
    """
    dark_rate  = DARK_REF_ADU_S * 2.0**((ccd_temp1_c - T_REF_DARK_C) / T_DOUBLE_C)
    mean_dark  = max(dark_rate * exp_time_s, 0.0)
    dark_arr   = rng.poisson(mean_dark, size=(NY_PIX, NX_PIX)).astype(float)
    read       = rng.normal(0.0, DARK_READ_NOISE, size=(NY_PIX, NX_PIX))
    image      = np.round(dark_arr + read + BIAS_ADU).astype(np.float32)
    return np.clip(image, 0, ADU_MAX).astype(np.uint16)
```

#### 7.3.4 Dispatcher

```python
def _generate_pixels(frame_type: str, v_rel_ms, ccd_temp1_c: float,
                     exp_time_cts: int, rng) -> np.ndarray:
    """
    Dispatch to the appropriate pixel generator.
    RNG draws for pixel synthesis always occur after the 9 metadata draws.
    """
    exp_time_s = exp_time_cts * TIMER_PERIOD_S
    if frame_type == 'science':
        return _generate_science_pixels(v_rel_ms, rng)
    elif frame_type == 'cal':
        return _generate_cal_pixels(rng)
    elif frame_type == 'dark':
        return _generate_dark_pixels(ccd_temp1_c, exp_time_s, rng)
    else:
        raise ValueError(f"Unknown frame_type: {frame_type!r}")
```

### 7.4 Binary file writer — `_write_bin_file()`

Assembles the 276-word header row and the 259×276 pixel array into the
143,520-byte P01 binary format.

```python
def _write_bin_file(meta: ImageMetadata,
                    pixels_256: np.ndarray,
                    path: pathlib.Path) -> None:
    """
    Write a WindCube FPI binary image file.

    Binary layout (P01 §2.1):
        Row 0       : 276 × uint16 — encoded ImageMetadata header
        Rows 1–259  : 259 × 276 × uint16 — pixel data (14-bit, 0–16383)
        Total bytes : 260 × 276 × 2 = 143,520

    The 256×256 science region is embedded at:
        rows [ROW_OFFSET_PIX : ROW_OFFSET_PIX+256]
        cols [COL_OFFSET_PIX : COL_OFFSET_PIX+256]
    All other pixels in the 259×276 region are set to BIAS_ADU.

    Parameters
    ----------
    meta       : ImageMetadata
    pixels_256 : np.ndarray, shape (256, 256), uint16
    path       : destination .bin file path (parent directory must exist)
    """
    header       = _encode_header(meta)                          # (276,) ">u2"
    pixel_array  = np.full((N_ROWS_BIN, N_COLS_BIN),
                           BIAS_ADU, dtype=np.uint16)
    pixel_array[ROW_OFFSET_PIX : ROW_OFFSET_PIX + NY_PIX,
                COL_OFFSET_PIX : COL_OFFSET_PIX + NX_PIX] = pixels_256

    full_array   = np.vstack([
        header.reshape(1, N_COLS_BIN),
        pixel_array.astype(">u2"),
    ]).astype(">u2")

    assert full_array.shape  == (260, 276),  f"Unexpected shape: {full_array.shape}"
    assert full_array.nbytes == 143_520,     f"Unexpected size: {full_array.nbytes}"
    path.write_bytes(full_array.tobytes())
```

### 7.5 Filename convention — `_bin_filename()`

```python
def _bin_filename(meta: ImageMetadata) -> str:
    """
    Return the binary filename for this frame.

    Format: YYYYMMDDThhmmssZ_{img_type}.bin
    Colons are excluded for Windows filesystem compatibility.

    Examples:
        20270101T000000Z_science.bin
        20270101T013420Z_cal.bin
        20270101T013510Z_dark.bin
    """
    from datetime import datetime, timezone
    dt     = datetime.fromtimestamp(meta.lua_timestamp / 1000.0, tz=timezone.utc)
    ts_str = dt.strftime('%Y%m%dT%H%M%SZ')
    return f"{ts_str}_{meta.img_type}.bin"
```

### 7.6 RNG draw order — complete per-frame sequence

The pixel synthesis draws (step 5) always follow the 9 metadata draws
(steps 1–4). This ensures that the metadata noise fields are reproducible
regardless of whether image synthesis is later modified.

```
Per observation frame — total RNG draws:
  1.  theta           1 draw  — rng.normal(0, SIGMA_POINTING_RAD)
  2.  axis raw        3 draws — rng.standard_normal(3)
  3.  etalon temps    4 draws — rng.normal(24.0, 0.1, 4)
  4.  ccd_temp1       1 draw  — rng.normal(-10.0, 1.0)
  ── metadata draws complete (9 total) ──────────────────────────
  5a. [science] photon noise  65536 draws — rng.poisson(signal, size=(256,256))
  5b. [science] read noise    65536 draws — rng.normal(0, READ_NOISE, (256,256))
  5a. [cal]     photon noise  65536 draws — rng.poisson(signal, size=(256,256))
  5b. [cal]     read noise    65536 draws — rng.normal(0, READ_NOISE, (256,256))
  5a. [dark]    dark Poisson  65536 draws — rng.poisson(mean_dark, size=(256,256))
  5b. [dark]    read noise    65536 draws — rng.normal(0, READ_NOISE, (256,256))
```

---

## 8. Output files

### 8.1 Naming convention

```
{output_dir}/
├── GEN01_{t_start_compact}_{duration_days:05.1f}d_{windmap_tag}_seed{rng_seed:04d}.npy
├── GEN01_{t_start_compact}_{duration_days:05.1f}d_{windmap_tag}_seed{rng_seed:04d}.csv
└── bin_frames/
    ├── 20270101T000000Z_science.bin
    ├── 20270101T000010Z_science.bin
    ...
    ├── 20270101T012950Z_cal.bin
    ├── 20270101T013000Z_cal.bin
    ...
    └── 20270101T235950Z_dark.bin
```

All `.bin` files land in `{output_dir}/bin_frames/`. The directory is
created by G01; it must not pre-exist or must be empty to avoid mixing
outputs from different runs.

### 8.2 `.npy` — primary pipeline format (unchanged)

Full `ImageMetadata` for every frame, including `truth_v_los` (now populated).

### 8.3 `.csv` — 46 columns

Columns 1–42 are unchanged from v8. Four new columns appended:

| # | Column name | P01 field | Science | Cal / Dark |
|---|-------------|-----------|---------|-----------|
| 43 | `v_wind_los_ms` | `truth_v_los` | NB02c `v_wind_LOS`, m/s | `NaN` |
| 44 | `v_earth_los_ms` | — (CSV only) | NB02c `v_earth_LOS`, m/s | `NaN` |
| 45 | `v_sc_los_ms` | — (CSV only) | NB02c `V_sc_LOS`, m/s | `NaN` |
| 46 | `v_rel_ms` | — (CSV only) | NB02c `v_rel`, m/s | `NaN` |

`V_sc_LOS`, `v_earth_LOS`, and `v_rel` have no dedicated `ImageMetadata`
fields; they exist in the CSV only. A future P01 spec revision could add
them to `ImageMetadata` if needed by downstream modules.

```python
row_dict.update({
    'v_wind_los_ms':  vr['v_wind_LOS']  if frame_type == 'science' else float('nan'),
    'v_earth_los_ms': vr['v_earth_LOS'] if frame_type == 'science' else float('nan'),
    'v_sc_los_ms':    vr['V_sc_LOS']    if frame_type == 'science' else float('nan'),
    'v_rel_ms':       vr['v_rel']       if frame_type == 'science' else float('nan'),
})
```

---

## 9. Progress reporting and verification

### 9.1 Console additions (v9)

```
Building NB02a + NB02b + NB02c + image synthesis ...
  [====================] 120410/120410
  NB02b+NB02c called for 115860 science frames.
  .bin files written to: .../bin_frames/ (120410 files, ~17.3 GB)

v_rel stats for science frames (m/s):
  Mean  : XXXX.X   (dominated by V_sc_LOS ≈ -7100 m/s for along-track)
  Std   :  XXX.X
  Min   : XXXX.X   Max: XXXX.X

v_wind_LOS stats (m/s):
  Mean  :  XXX.X   (truth wind projected onto LOS)
  Std   :   XX.X
```

### 9.2 Verification checks (C1–C21)

| Check | Criterion | File |
|-------|-----------|------|
| C1 | NB01 sched rows ≈ `duration_s / SCHED_DT_S` ± 2 | — |
| C2 | All `orbit_parity` valid | `.npy` |
| C3 | No NaN in `att_q_*` | `.csv` |
| C4 | Attitude quaternions unit norm to 1e-6 | `.csv` |
| C5 | PE quaternions unit norm to 1e-6 | `.csv` |
| C6 | Mean unsigned PE within 20% of `σ·√(2/π)` ≈ 3.99 arcsec | `.csv` |
| C7 | Etalon mean within 0.05°C of 24.0°C | `.csv` |
| C8 | P01 `adcs_quality_flag == 0` for > 99.9% | `.npy` |
| C9 | `img_type` valid via `_classify_img_type(gpio, lamp)` | `.csv` |
| C10 | Cal count ≈ `n_caldark × len(cal_trigger_indices)` ±5% | `.npy` |
| C11 | Dark count ≈ `n_caldark × len(cal_trigger_indices)` ±5% | `.npy` |
| C12 | CSV has exactly **46** columns | `.csv` |
| C13 | `.npy` round-trip succeeds | `.npy` |
| C14 | `tp_lat_deg` is `NaN` for all cal/dark rows | `.csv` |
| C15 | `tp_lat_deg` is non-NaN for all science rows | `.csv` |
| C16 | `tp_lat_deg` within `lat_band_deg + 5°` of 0° for science rows | `.csv` |
| C17 | `wind_v_zonal_ms` is non-NaN for all science rows | `.csv` |
| C18 | `v_rel_ms` is non-NaN for all science rows | `.csv` |
| C19 | `v_rel_ms` is NaN for all cal/dark rows | `.csv` |
| C20 | All `.bin` files exist and have size 143,520 bytes exactly | `bin_frames/` |
| C21 | Header round-trip: parse header of first science `.bin` via P01 `parse_header()`; `lua_timestamp` matches CSV row | `bin_frames/` |

**C20 implementation:**
```python
bin_files = list(bin_dir.glob("*.bin"))
sizes_ok  = all(f.stat().st_size == 143_520 for f in bin_files)
count_ok  = len(bin_files) == len(metadata_list)
c20 = sizes_ok and count_ok
```

**C21 implementation:**
```python
# Find first science .bin, load it, decode header, compare timestamp
first_sci_bin = next(f for f in sorted(bin_dir.glob("*.bin"))
                     if 'science' in f.name)
raw  = np.frombuffer(first_sci_bin.read_bytes(), dtype=">u2")
hdr  = raw[:276]
from src.metadata.p01_image_metadata_2026_04_06 import parse_header
d    = parse_header(hdr)
# Find matching CSV row
sci_csv = df_csv[df_csv.img_type_derived == 'science'].iloc[0]
c21 = (d['lua_timestamp'] == int(sci_csv['lua_timestamp']))
```

---

## 10. File location in repository

```
soc_sewell/
├── validation/
│   ├── gen01_synthetic_metadata_generator_2026_04_16.py
│   └── outputs/   (or user-selected folder outside repo)
│       ├── GEN01_20270101_030.0d_uniform_seed0042.npy
│       ├── GEN01_20270101_030.0d_uniform_seed0042.csv
│       └── bin_frames/
│           ├── 20270101T000000Z_science.bin
│           ├── 20270101T013420Z_cal.bin
│           └── ...  (~120k files, ~17 GB for 30-day run)
└── docs/specs/
    └── G01_synthetic_metadata_generator_2026-04-16.md
```

---

## 11. Instructions for Claude Code

### Preamble
```bash
cat PIPELINE_STATUS.md
```

### Prerequisite reads
1. This spec in full.
2. NB02c spec / `nb02c_los_projection_2026_04_16.py` — `compute_v_rel(wind_map,
   tp_lat_deg, tp_lon_deg, tp_eci, vel_eci, los_eci, epoch)` → dict with
   keys `v_rel`, `v_wind_LOS`, `V_sc_LOS`, `v_earth_LOS`, `v_zonal_ms`,
   `v_merid_ms`. Note capital `V` in `V_sc_LOS`.
3. P01 spec / `p01_image_metadata_2026_04_06.py` — `parse_header()` (for
   C21 round-trip check); `_f64()` and `_u64()` decode logic (to verify
   your encode inverses are correct).
4. All other prerequisite reads from v8 (NB00, NB01, NB02a, NB02b, P01,
   `CLAUDE.md`).

### Prerequisite tests
```bash
pytest tests/test_nb01_orbit_propagator.py -v    # 8/8
pytest tests/test_nb02_geometry_2026_04_16.py -v # 10/10
pytest tests/test_s19_p01_metadata.py -v         # 8/8
```

### Additional import
```python
from src.geometry.nb02c_los_projection_2026_04_16 import compute_v_rel
```

### New constants to add to constants block
```python
# FPI optical model
LAMBDA_OI_M    = 630.0e-9;   LAMBDA_NE1_M = 640.2248e-9;  LAMBDA_NE2_M = 638.2991e-9
ETALON_GAP_M   = 20.106e-3;  FOCAL_LENGTH_M = 0.19912;    PLATE_SCALE_RPX = 1.6071e-4
R_REFL         = 0.53;        N_GAP = 1.0;                 C_LIGHT_MS = 2.99792458e8
FINESSE_F      = 4*R_REFL / (1-R_REFL)**2   # ≈ 9.6
# Pixel layout
NX_PIX=256; NY_PIX=256; N_ROWS_BIN=259; N_COLS_BIN=276
ROW_OFFSET_PIX=1; COL_OFFSET_PIX=10; BIAS_ADU=100; ADU_MAX=16383
# Signal levels
SCI_PEAK_ADU=5000; SCI_READ_NOISE=5.0
CAL_PEAK_ADU=12000; CAL_NE_RATIO=3.0; CAL_READ_NOISE=5.0
# Dark model
DARK_REF_ADU_S=0.05; T_REF_DARK_C=-20.0; T_DOUBLE_C=6.5; DARK_READ_NOISE=5.0
```

### Critical rules (additions over v8)

- **NB02c replaces `wind_map.sample()`** in the science-frame path. Do not
  call `wind_map.sample()` directly for science frames; use `compute_v_rel()`
  which calls it internally and returns `v_zonal_ms` and `v_merid_ms`.
- **`truth_v_los` = `vr['v_wind_LOS']`** for science frames. This field is
  now populated (was `None` in v7/v8).
- **`los_eci` must be retained** from `compute_los_eci()`. It is needed by
  both NB02b and NB02c.
- **`tp['tp_eci']`** (a `numpy.ndarray`) is the `tp_eci` argument to
  `compute_v_rel()`, not `tp_eci_x/y/z` scalars.
- **`_encode_f64` round-trip must be exact.** Verify once:
  `assert struct.unpack(">d", struct.pack(">4H", *reversed(_encode_f64(x))))[0] == x`
  for x = 1.234567890123456e7 before using it.
- **Quaternion reorder in `_encode_header`**: pipeline `[x,y,z,w]` → binary
  `[w,x,y,z]`. Apply to both `attitude_quaternion` and `pointing_error`.
- **C21 derives `img_type`** by running `_classify_img_type(gpio, lamp)` on
  the decoded header; do not read `img_type` from the header dict directly
  (the parse_header function in P01 does derive it, but checking it explicitly
  is more robust as a verification step).
- **C12 = 46 columns** (was 42).
- **RNG order**: 9 metadata draws first; pixel draws always after. Never
  interleave.
- **`bin_dir`** = `{output_dir}/bin_frames/`. Create with `mkdir(parents=True,
  exist_ok=True)` before the loop. Do not create a new directory per orbit.
- **Storage estimate** in console: `len(metadata_list) × 143_520 / 1e9` GB.

### Epilogue
```bash
git add PIPELINE_STATUS.md \
        validation/gen01_synthetic_metadata_generator_2026_04_16.py
git commit -m "feat(g01): NB02c LOS decomposition + binary image synthesis, C1-C21

NB02c: v_rel, v_wind_LOS, V_sc_LOS, v_earth_LOS for science frames
truth_v_los now populated in ImageMetadata
CSV: 46 columns (+v_wind_los_ms/v_earth_los_ms/v_sc_los_ms/v_rel_ms)
Binary synthesis: science=Airy(v_rel), cal=two-line neon, dark=dark+noise
_encode_header() inverse of P01 parse_header(); C21 round-trip verified
~17 GB .bin output for 30-day run (143520 bytes/file)

Also updates PIPELINE_STATUS.md"
```
