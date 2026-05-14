# H07 — Wind Vector Retrieval
## WindCube SOC Pipeline — Specification v0.2
**Spec ID:** H07  
**Spec file:** `specs/H07_wind_vector_retrieval_2026-05-14.md`  
**Previous version:** `specs/H07_wind_vector_retrieval_2025-05-14.md` (v0.1 — draft, superseded)  
**Date:** 2026-05-14  
**Author:** Scott Sewell (HAO/NCAR)  
**Repo:** soc_sewell  
**Status:** Authoritative — v0.2  

**Depends on:**
- S19/P01 v2 (`P01_metadata_2026-05-14.md`) — `ImageMetadata` including `h_target_km_obs`
- G01 v14 (`G01_synthetic_metadata_generator_2026-05-14.md`) — synthetic dataset format
- S20 (M08 L2 netCDF-4 writer) — output format for wind solutions
- NB02a (`nb02a_boresight_2026_04_16.py`) — boresight convention reference

**Key reference documents:**
- SI-UCAR-WC-RP-004 WindCube AOCS Design Report v1.0 (Space Inventor, 2024-09-20)
- Harding, Gehrels & Makela (2014), *Applied Optics* 53(4), 666–672

---

## Revision history

| Version | Date | Change |
|---------|------|--------|
| v0.1 | 2025-05-14 | Initial draft. Boresight assumed `[0,0,−1]_BRF` (incorrect). Open issues H07-01 and H07-02 unresolved. |
| **v0.2** | **2026-05-14** | **Complete rewrite based on confirmed AOCS design report, NB02a source, and GEN01 CSV analysis. Boresight corrected to `[−1,0,0]_BRF`. Sign convention resolved against Harding and GEN01. obs_mode derivation algorithm specified. h_target_km_obs integrated from P01 v2. All v0.1 open issues closed.** |

---

## 1. Purpose

H07 defines **Module H07** (`windcube/wind_retrieval.py`), which converts a
time-tagged sequence of calibrated line-of-sight (LOS) Doppler velocities —
one per airglow science frame — into horizontal thermospheric wind vectors
(zonal v_E, meridional v_N) at the OI 630.0 nm emission tangent point.

The module solves the three-component LOS velocity decomposition problem:
the FPI-measured Doppler shift encodes a scalar projection of the combined
spacecraft, Earth-rotation, and thermospheric-wind velocity fields. H07
corrects for the two known (calculable) contributions and inverts the
residuals over multiple frames to recover the wind vector.

---

## 2. Science context and signal decomposition

### 2.1 The measurement

The OI 630.0 nm emission layer peaks at approximately **h = 250 km** altitude
(configurable via `h_target_km_obs`). For each science frame, M06 recovers a
centre wavelength λ_c from the Airy fringe fit. The Harding (2014) convention
defines the LOS velocity as:

```
v_rel = (λ_c − λ_0) / λ_0  ×  c        [m/s]
```

where λ_0 = 630.0 nm is the OI rest wavelength and c = 299,792,458 m/s.
**Positive v_rel means the airglow source is receding from the spacecraft
(redshift); negative means approaching (blueshift).** This is the Harding
convention used throughout the pipeline.

### 2.2 Three-component decomposition

The measured v_rel is the Harding-convention total LOS velocity:

```
v_rel = −( v_sc · l̂  +  v_earth · l̂  +  v_wind · l̂ )
```

where **l̂** is the unit vector pointing **from spacecraft toward tangent
point** (SC→TP direction), and the negative sign converts from the
approach-positive dot product to the recession-positive Harding convention.

Equivalently:

```
v_rel = v_sc_harding + v_earth_harding + v_wind_harding
```

where each Harding component is defined as:

```
v_X_harding = −dot(v_X, l̂)     [recession-positive]
```

The three contributions are:

| Term | Symbol | Typical magnitude | Calculable? |
|------|--------|------------------|-------------|
| Spacecraft orbital velocity | v_sc_harding | ±7500 m/s | Yes — from metadata |
| Earth rotation at tangent point | v_earth_harding | ±465 m/s (equator) | Yes — from geometry |
| Thermospheric wind (target) | v_wind_harding | 0–500 m/s | No — science product |

### 2.3 Corrected LOS velocity

After subtracting the two known terms:

```
v_corrected = v_rel − v_sc_harding − v_earth_harding
            = v_wind_harding
            = −dot(v_wind, l̂)
```

This scalar is what H07 inverts to recover (v_E, v_N).

### 2.4 Relationship to GEN01 CSV truth columns

The GEN01 v14 CSV uses approach-positive component columns. The mapping to
Harding convention used in H07 is:

```
v_sc_harding    = −v_sc_los_approach_ms
v_earth_harding = −v_earth_los_approach_ms
v_wind_harding  = −v_wind_los_approach_ms  =  truth_v_los  (in ImageMetadata)
```

The CSV column `v_rel_ms` is already in Harding convention and equals
`v_sc_harding + v_earth_harding + v_wind_harding`.

---

## 3. Instrument boresight and coordinate frames

### 3.1 Payload boresight in body frame

Per SI-UCAR-WC-RP-004 §2.4.2.1 (Satellite Body Reference Frame):

```
−X_BRF = payload boresight (FPI optical axis)
+Y_BRF = Star Tracker boresight
+Z_BRF = completes right-handed basis
```

The boresight unit vector in the spacecraft body frame is:

```python
BORESIGHT_BODY = np.array([-1.0, 0.0, 0.0])   # -X_BRF
```

This is the **single most safety-critical constant** in H07. Any error here
produces a systematic velocity bias of order |v_sc| ≈ 7500 m/s.

### 3.2 Quaternion convention

The `attitude_quaternion` field in `ImageMetadata` is **scalar-last**
`[x, y, z, w]` throughout the pipeline. The binary `.bin` header stores
`[w, x, y, z]` (scalar-first); P01 re-orders at ingest. H07 always reads
from `ImageMetadata`, never from raw binary.

```python
from scipy.spatial.transform import Rotation
q = meta.attitude_quaternion          # [x, y, z, w] scalar-last
R_body2eci = Rotation.from_quat(q)    # scipy default is scalar-last
los_eci = R_body2eci.apply(BORESIGHT_BODY)
los_eci /= np.linalg.norm(los_eci)    # unit vector SC→TP in ECI
```

### 3.3 Geometry confirmation

For the nominal 510 km altitude / 250 km tangent height mission geometry:
- Depression below local horizontal: **~15.8°**
- Angle from nadir: **~74.2°**
- Along-track mode: boresight projects strongly onto the velocity direction
- Cross-track mode: boresight projects strongly onto the orbit normal

These values are confirmed against the GEN01 CSV (off-nadir = 74.07° for
rows 0, 100, 500, 1000, 2000).

### 3.4 Coordinate frames

Three frames are used. All frame conversions must be explicit.

**ECI (Earth-Centered Inertial, J2000):**
- Origin: Earth centre of mass
- Axes: fixed to inertial space
- Used for: spacecraft position/velocity, attitude quaternion output,
  LOS vector `l̂_eci`

**ECEF (Earth-Centered Earth-Fixed, WGS84):**
- Origin: Earth centre of mass
- Axes: rotate with Earth
- Used for: Earth rotation velocity, tangent point geodetic coordinates
- Rotation from ECI: GMST rotation about z-axis at image epoch

**ENU (East-North-Up at tangent point):**
- Origin: tangent point on WGS84 ellipsoid
- Axes: Ê (eastward), N̂ (northward), Ẑ (radially outward)
- Used for: decomposing v_wind into (v_E, v_N) direction cosines

---

## 4. Geometry engine (Stage G)

### 4.1 LOS vector in ECI

```python
BORESIGHT_BODY = np.array([-1.0, 0.0, 0.0])   # -X_BRF per AOCS report §2.4.2.1

q = meta.attitude_quaternion                    # [x, y, z, w] scalar-last
R = Rotation.from_quat(q)                       # BRF → ECI
l_hat_eci = R.apply(BORESIGHT_BODY)
l_hat_eci /= np.linalg.norm(l_hat_eci)
```

`l_hat_eci` points from spacecraft toward tangent point (SC→TP).

### 4.2 Tangent point computation

The tangent point is the point along the LOS ray that achieves geodetic
altitude `h_target_km_obs` above the WGS84 ellipsoid.

**Target altitude:** Use `meta.h_target_km_obs` if not None; otherwise
default to 250.0 km and log a `UserWarning` (backward compatibility with
pre-v2 sidecars per P01 §3.10).

**Ray parameterisation:**

```
r(s) = r_sc + s × l_hat_eci      (s in metres, s > 0 toward TP)
```

**Ellipsoid intersection:** Find s* such that the geodetic altitude of
`r(s*)` equals `h_target_km_obs`. Use `scipy.optimize.brentq` on the
interval `[0, 2 × |r_sc|]`. The geodetic altitude is computed via the
Bowring iterative method or `astropy.coordinates`.

**For synthetic frames:** If `meta.is_synthetic is True` and
`meta.tangent_lat is not None`, the stored NB02b tangent point is
authoritative. Use it directly. Still compute `l_hat_eci` from the
quaternion independently — the two should agree to < 0.1 km geodetic
distance (use this as a validation check, not a hard gate).

**Output of Stage G per frame:**

```python
@dataclass
class LOSGeometry:
    l_hat_eci:       np.ndarray    # shape (3,) SC→TP unit vector, ECI J2000
    l_hat_ecef:      np.ndarray    # shape (3,) SC→TP unit vector, ECEF
    L_E:             float         # l̂ · Ê  eastward direction cosine
    L_N:             float         # l̂ · N̂  northward direction cosine
    L_Z:             float         # l̂ · Ẑ  upward direction cosine
    tangent_lat_deg: float         # geodetic latitude of tangent point, deg
    tangent_lon_deg: float         # geodetic longitude of tangent point, deg
    tangent_alt_km:  float         # geodetic altitude of tangent point, km
    tangent_pos_eci: np.ndarray    # shape (3,) tangent point position, m, ECI
    v_sc_harding:    float         # spacecraft LOS velocity, Harding convention, m/s
    v_earth_harding: float         # Earth rotation LOS velocity, Harding convention, m/s
```

### 4.3 ENU basis vectors at tangent point

Given geodetic coordinates (φ, λ) at the tangent point (φ = latitude in
radians, λ = longitude in radians):

```python
E_hat = np.array([-np.sin(λ),          np.cos(λ),         0.0         ])
N_hat = np.array([-np.sin(φ)*np.cos(λ), -np.sin(φ)*np.sin(λ), np.cos(φ)])
Z_hat = np.array([ np.cos(φ)*np.cos(λ),  np.cos(φ)*np.sin(λ), np.sin(φ)])
```

These are ECEF unit vectors. Convert `l_hat_eci` to ECEF via the GMST
rotation matrix before taking dot products.

Direction cosines:

```python
L_E = float(np.dot(l_hat_ecef, E_hat))
L_N = float(np.dot(l_hat_ecef, N_hat))
L_Z = float(np.dot(l_hat_ecef, Z_hat))
```

### 4.4 Earth rotation contribution (Harding convention)

```python
OMEGA_EARTH = np.array([0.0, 0.0, 7.2921150e-5])   # rad/s, ECI z-axis

# Earth rotation velocity at tangent point (ECI)
v_earth_eci = np.cross(OMEGA_EARTH, tangent_pos_eci)

# Harding convention: recession-positive = −dot(v, l̂)
v_earth_harding = -float(np.dot(v_earth_eci, l_hat_eci))
```

### 4.5 Spacecraft velocity contribution (Harding convention)

```python
vel_eci = np.array(meta.vel_eci_hat)    # ECI, m/s, from metadata

# Harding convention: recession-positive = −dot(v_sc, l̂)
v_sc_harding = -float(np.dot(vel_eci, l_hat_eci))
```

### 4.6 GMST rotation matrix

```python
def gmst_rotation_matrix(unix_ms: int) -> np.ndarray:
    """
    3×3 rotation matrix from ECI (J2000) to ECEF at the given epoch.

    Uses astropy for accuracy. Falls back to simple GMST formula if
    astropy is unavailable (accuracy ~0.1 arcsec over mission lifetime).
    """
    from astropy.time import Time
    from astropy.coordinates import GCRS, ITRS, EarthLocation
    import astropy.units as u
    t = Time(unix_ms / 1000.0, format="unix", scale="utc")
    # Build rotation from GCRS (≈ ECI J2000) to ITRS (≈ ECEF)
    gcrs = GCRS(obstime=t)
    itrs = ITRS(obstime=t)
    # 3×3 rotation matrix columns: ECI unit vectors expressed in ECEF
    R = gcrs.matrix_eci_to_ecef(t)   # use astropy's built-in if available
    return R
```

**Fallback (simple GMST):**

```python
def _gmst_simple(unix_ms: int) -> np.ndarray:
    """Simple GMST rotation; accurate to ~0.1 arcsec over ±50 yr."""
    import math
    jd = unix_ms / 86400000.0 + 2440587.5
    d = jd - 2451545.0                          # days from J2000.0
    gmst_rad = (280.46061837 + 360.98564736629 * d) * math.pi / 180.0
    gmst_rad = gmst_rad % (2 * math.pi)
    c, s = math.cos(gmst_rad), math.sin(gmst_rad)
    return np.array([[c, s, 0],[-s, c, 0],[0, 0, 1]])
```

---

## 5. obs_mode derivation from attitude quaternion (Stage M)

H07 can derive `obs_mode` independently from the attitude quaternion. This
serves two purposes: (a) cross-checking the stored `meta.obs_mode`; (b)
processing real images where `obs_mode` may be `"unknown"`.

**Algorithm:**

```python
def derive_obs_mode(
    attitude_quaternion: list[float],   # [x, y, z, w] scalar-last
    pos_eci: np.ndarray,                # spacecraft ECI position, m
    vel_eci: np.ndarray,                # spacecraft ECI velocity, m/s
    ambiguity_threshold: float = 0.707, # cos(45°) — minimum dot product
) -> str:
    """
    Derive 'along_track' or 'cross_track' from attitude quaternion.

    Method: rotate the boresight [-1,0,0]_BRF to ECI, then project onto
    the forward velocity direction and the orbit normal. The larger
    projection identifies the look mode.

    For nominal 510km/250km geometry, the expected projection magnitude
    is cos(15.8°) ≈ 0.962 for the correct mode and ~sin(15.8°) ≈ 0.272
    for the perpendicular mode — unambiguous classification.
    """
    from scipy.spatial.transform import Rotation

    # Boresight in ECI
    q = attitude_quaternion
    R = Rotation.from_quat(q)
    boresight_eci = R.apply(np.array([-1.0, 0.0, 0.0]))
    boresight_eci /= np.linalg.norm(boresight_eci)

    # Nadir unit vector (toward Earth)
    nadir = -pos_eci / np.linalg.norm(pos_eci)

    # Forward horizontal (velocity, orthogonalised against nadir)
    v_hat = vel_eci / np.linalg.norm(vel_eci)
    v_horiz = v_hat - np.dot(v_hat, nadir) * nadir
    norm_vh = np.linalg.norm(v_horiz)
    if norm_vh < 1e-10:
        return "unknown"
    v_horiz /= norm_vh

    # Orbit normal
    orbit_normal = np.cross(pos_eci, vel_eci)
    orbit_normal /= np.linalg.norm(orbit_normal)

    # Project boresight onto each reference direction
    dot_v = abs(float(np.dot(boresight_eci, v_horiz)))
    dot_n = abs(float(np.dot(boresight_eci, orbit_normal)))

    if max(dot_v, dot_n) < ambiguity_threshold:
        return "unknown"   # boresight doesn't align with either direction

    return "along_track" if dot_v > dot_n else "cross_track"
```

**Use in H07:** Before processing each frame, call `derive_obs_mode()` and
compare with `meta.obs_mode`. If they disagree and `meta.obs_mode` is not
`"unknown"`, log a warning but proceed with `meta.obs_mode` (the stored
value is authoritative for real data). For synthetic data, the two should
always agree.

---

## 6. Velocity correction (Stage C)

```python
def correct_los_velocity(
    v_rel: float,       # Harding LOS velocity from M06, m/s
    geom: LOSGeometry,  # output of compute_los_geometry()
) -> float:
    """
    Remove spacecraft and Earth-rotation contributions from v_rel.

    Returns v_corrected: the thermospheric wind LOS projection in
    Harding convention (positive = wind component receding from SC).

    v_corrected = v_rel − v_sc_harding − v_earth_harding
                = v_wind_harding
                = −dot(v_wind, l̂)
    """
    return v_rel - geom.v_sc_harding - geom.v_earth_harding
```

For the **null-wind validation test** (GEN01 wind map option 1,
v_zonal=v_merid=0): `v_corrected` should equal zero within the per-frame
noise level σ_v from M06. Any systematic offset indicates a geometry error.

---

## 7. Wind vector inversion (Stage I)

### 7.1 Design matrix

The horizontal wind assumption (v_Z = 0) is valid because:
- The vertical wind component at 250 km altitude is typically < 10 m/s
- `L_Z` for the WindCube geometry is sin(15.8°) ≈ 0.272, so the
  vertical wind contribution to v_corrected is < 2.7 m/s

For N frames within one spatiotemporal bin:

```python
A = np.zeros((N, 2))
b = np.zeros(N)
w = np.zeros(N)

for i, obs in enumerate(observations):
    A[i, 0] = -obs.L_E    # −L_E coefficient for v_E
    A[i, 1] = -obs.L_N    # −L_N coefficient for v_N
    b[i]    = obs.v_corrected
    w[i]    = 1.0 / obs.sigma_v**2
```

The sign convention: `v_corrected = −dot(v_wind, l̂) = −(v_E·L_E + v_N·L_N)`
therefore `v_corrected = −v_E·L_E − v_N·L_N`, giving design matrix
coefficients `[−L_E, −L_N]`.

### 7.2 Weighted least squares

```python
W = np.diag(w)
ATA = A.T @ W @ A
ATb = A.T @ W @ b

# Condition number check before inversion
cond = np.linalg.cond(ATA)
if cond > GDOP_MAX:
    return WindSolution(gdop_flag=True, ...)   # ill-conditioned

x = np.linalg.solve(ATA, ATb)                 # [v_E, v_N] in m/s
C = np.linalg.inv(ATA)                        # 2×2 covariance matrix
```

Solution:

```python
v_E = x[0]                        # eastward wind, m/s (positive = eastward)
v_N = x[1]                        # northward wind, m/s (positive = northward)
sigma_v_E = np.sqrt(C[0, 0])      # 1-sigma uncertainty, m/s
sigma_v_N = np.sqrt(C[1, 1])      # 1-sigma uncertainty, m/s
two_sigma_v_E = 2.0 * sigma_v_E   # must equal exactly 2 × sigma_v_E
two_sigma_v_N = 2.0 * sigma_v_N   # must equal exactly 2 × sigma_v_N
```

**Contract:** `two_sigma_v_E == 2.0 * sigma_v_E` exactly (per uncertainty
standards addendum — no rounding or separate calculation).

### 7.3 Geometric dilution of precision (GDOP)

```python
GDOP_MAX = 100.0   # condition number threshold (from windcube/constants.py)
```

Physical interpretation: GDOP is large when all LOS vectors in a bin point
in nearly the same direction. This occurs when all frames in a bin are from
the same look mode and the orbit has not provided sufficient azimuthal
diversity. Expected to be rare in nominal operations (along-track and
cross-track alternate every orbit).

### 7.4 Minimum frame count

```python
N_MIN_FRAMES = 4   # minimum frames per bin (from windcube/constants.py)
```

---

## 8. Spatiotemporal binning (Stage B)

### 8.1 Bin definition

Each output wind vector is associated with a bin defined by:
- **Geographic location:** tangent point (lat, lon) gridded to
  `dlat × dlon` degree cells (default 5° × 5°)
- **Time:** bin centre UTC, with width `dt_min` minutes (default 30 min)

A frame is assigned to the bin whose centre is closest to its tangent point
and whose time window contains its image epoch. Each frame contributes to
exactly one bin.

### 8.2 Multi-day accumulation mode

For DE3 tidal analysis (SQ2), data are accumulated over `n_days` (1–30)
with bins defined by (lat, local\_solar\_time, longitude) rather than
(lat, lon, UTC). The `accumulate_days` parameter switches this mode.
Default: `accumulate_days=False` (UTC binning).

### 8.3 Bin quality flags

Each `WindSolution` carries:

| Flag / field | Type | Meaning |
|---|---|---|
| `n_frames` | int | Frames contributing to this bin |
| `gdop_flag` | bool | True if condition number > GDOP_MAX |
| `n_frames_flag` | bool | True if n_frames < N_MIN_FRAMES |
| `mean_tangent_alt_km` | float | Mean tangent altitude of contributing frames |
| `mean_epoch_unix_ms` | int | Mean image epoch of contributing frames |
| `obs_modes` | set[str] | Look modes present in this bin |

---

## 9. Module interface

All functions live in `windcube/wind_retrieval.py`.

### 9.1 Data structures

```python
@dataclass
class LOSGeometry:
    l_hat_eci:       np.ndarray    # (3,) SC→TP unit vector, ECI
    l_hat_ecef:      np.ndarray    # (3,) SC→TP unit vector, ECEF
    L_E:             float         # l̂ · Ê  eastward direction cosine
    L_N:             float         # l̂ · N̂  northward direction cosine
    L_Z:             float         # l̂ · Ẑ  upward direction cosine
    tangent_lat_deg: float         # geodetic latitude, deg
    tangent_lon_deg: float         # geodetic longitude, deg
    tangent_alt_km:  float         # geodetic altitude, km
    tangent_pos_eci: np.ndarray    # (3,) tangent point position, ECI, m
    v_sc_harding:    float         # SC contribution, Harding convention, m/s
    v_earth_harding: float         # Earth rotation contribution, Harding, m/s


@dataclass
class LOSObservation:
    v_corrected:     float         # wind LOS velocity, Harding convention, m/s
    sigma_v:         float         # 1-sigma uncertainty from M06, m/s
    L_E:             float         # eastward direction cosine
    L_N:             float         # northward direction cosine
    L_Z:             float         # upward direction cosine
    tangent_lat_deg: float         # deg
    tangent_lon_deg: float         # deg
    tangent_alt_km:  float         # km
    epoch_unix_ms:   int           # image epoch, Unix ms
    obs_mode:        str           # 'along_track' or 'cross_track'


@dataclass
class WindSolution:
    v_E:                  float    # eastward wind, m/s (+ = eastward)
    v_N:                  float    # northward wind, m/s (+ = northward)
    sigma_v_E:            float    # 1-sigma uncertainty, m/s
    sigma_v_N:            float    # 1-sigma uncertainty, m/s
    two_sigma_v_E:        float    # = 2 × sigma_v_E exactly
    two_sigma_v_N:        float    # = 2 × sigma_v_N exactly
    n_frames:             int
    gdop_flag:            bool
    n_frames_flag:        bool
    condition_number:     float
    mean_tangent_lat_deg: float
    mean_tangent_lon_deg: float
    mean_tangent_alt_km:  float
    mean_epoch_unix_ms:   int
    obs_modes:            set      # e.g. {'along_track', 'cross_track'}
```

### 9.2 Public functions

---

#### `compute_los_geometry(meta: ImageMetadata) → LOSGeometry`

Compute the full LOS geometry for a single science frame. Implements
Stages G (§4) in full. Raises `ValueError` if `meta.img_type != "science"`.

---

#### `derive_obs_mode(attitude_quaternion, pos_eci, vel_eci, ambiguity_threshold=0.707) → str`

Derive `obs_mode` from quaternion geometry. Returns `"along_track"`,
`"cross_track"`, or `"unknown"`. See §5 for full algorithm.

---

#### `correct_los_velocity(v_rel: float, geom: LOSGeometry) → float`

Apply spacecraft and Earth-rotation corrections. Returns `v_corrected`
in Harding convention. See §6.

---

#### `process_frame(meta: ImageMetadata, v_rel: float, sigma_v: float) → LOSObservation`

Convenience wrapper combining `compute_los_geometry` and
`correct_los_velocity`. Also calls `derive_obs_mode` for cross-check.
Returns a fully populated `LOSObservation`. Raises `ValueError` if
`meta.img_type != "science"` or `meta.adcs_quality_flag & SLEW_IN_PROGRESS`.

---

#### `invert_wind_vector(observations: list[LOSObservation]) → WindSolution`

Weighted least-squares inversion over a list of `LOSObservation` objects
within one bin. Implements Stage I (§7). Returns `WindSolution` with
`gdop_flag=True` and NaN winds if condition number > GDOP_MAX, or
`n_frames_flag=True` if len(observations) < N_MIN_FRAMES.

---

#### `bin_observations(obs_list, dlat=5.0, dlon=5.0, dt_min=30.0) → dict`

Assign `LOSObservation` objects to spatiotemporal bins. Returns a dict
keyed by `(lat_centre_deg, lon_centre_deg, t_centre_unix_ms)` with values
`list[LOSObservation]`.

---

#### `gmst_rotation_matrix(unix_ms: int) → np.ndarray`

3×3 ECI→ECEF rotation matrix at given epoch. Uses astropy if available,
falls back to simple GMST formula. See §4.6.

---

## 10. Constants (in `windcube/constants.py`)

The following constants must be present in `windcube/constants.py`.
Add any that are missing; do not modify existing values.

```python
OMEGA_EARTH_RAD_S   = 7.2921150e-5    # Earth rotation rate [rad/s]
WGS84_A_M           = 6_378_137.0     # WGS84 equatorial radius [m]
WGS84_B_M           = 6_356_752.3142  # WGS84 polar radius [m]
OI_LAMBDA0_NM       = 630.0           # OI 630.0 nm rest wavelength [nm]
OI_EMISSION_ALT_KM  = 250.0           # Nominal emission layer altitude [km]
C_M_S               = 299_792_458.0   # Speed of light [m/s]
GDOP_MAX            = 100.0           # Condition number threshold
N_MIN_FRAMES        = 4               # Minimum frames per bin
H07_BORESIGHT_BODY  = [-1.0, 0.0, 0.0]  # -X_BRF per AOCS report §2.4.2.1
```

---

## 11. Null-wind validation test

For a synthetic dataset generated by GEN01 with wind map option 1,
v_zonal = 0, v_merid = 0 (e.g. `GEN01_20270101_001_0d_uniform_seed0042.csv`):

### 11.1 Expected behaviour at each stage

**Stage G:** `v_sc_harding` varies across frames over ±7500 m/s range.
`v_earth_harding` varies over approximately ±465×cos(lat) m/s range.
`l_hat_eci` should match the boresight computed directly from NB02a to
< 0.001 rad for synthetic frames.

**Stage C:** `v_corrected` should be zero within per-frame noise. For the
null-wind test with ideal (noiseless) geometry: `v_corrected = 0.0` exactly.
For frames with M06 noise: `|v_corrected| < 3 × sigma_v` for 99.7% of frames.

**Stage I:** With all `v_corrected ≈ 0`, recovered `v_E ≈ 0`, `v_N ≈ 0`.

### 11.2 Acceptance criteria

| Metric | Pass condition |
|--------|---------------|
| `mean(v_corrected)` over all frames | `< 5 m/s` systematic offset |
| `std(v_corrected)` | Consistent with mean per-frame σ_v from M06 |
| Recovered `v_E` per bin | Within 2σ of 0 for ≥ 95% of well-conditioned bins |
| Recovered `v_N` per bin | Within 2σ of 0 for ≥ 95% of well-conditioned bins |
| GDOP flags | `< 20%` of bins flagged |
| `derive_obs_mode` vs `meta.obs_mode` agreement | 100% for synthetic data |

### 11.3 CSV truth cross-check

For each science frame, the following identity must hold to < 0.01 m/s:

```
v_corrected_H07 ≈ −truth_v_los   (from meta.truth_v_los = 0.0)
```

And from the GEN01 CSV:

```
v_corrected_H07 ≈ −v_wind_los_approach_ms   (= 0.0 for null wind)
```

---

## 12. Diagnostic output — single-frame mode

The `invert_single_frame.py` driver script (specified separately in Step 8)
will produce the following for each frame:

```
═══════════════════════════════════════════════════════
Frame: 2027-01-01T00:00:00Z  (science, along_track)
───────────────────────────────────────────────────────
Spacecraft position (ECI):   [-1918.9, 6891215.9, -14683.9] m
Spacecraft velocity (ECI):   [984.4, 8.2, 7543.2] m/s
Attitude quaternion [x,y,z,w]: [0.5352, 0.5905, -0.4622, 0.3890]
Boresight [-1,0,0] in ECI:   [0.1245, -0.2724, 0.9541]
───────────────────────────────────────────────────────
Tangent point:  lat=15.58°  lon=-12.11°  alt=250.03 km
Direction cosines: L_E=+0.XXX  L_N=+0.XXX  L_Z=+0.XXX
───────────────────────────────────────────────────────
LOS velocity budget (Harding convention, + = recession):
  v_rel (measured by FPI):   XXXXX.X m/s
  v_sc_harding:              XXXXX.X m/s
  v_earth_harding:              +XX.X m/s
  v_corrected (wind):            +X.X m/s  (σ = X.X m/s)
───────────────────────────────────────────────────────
obs_mode stored:   along_track
obs_mode derived:  along_track  ✓
═══════════════════════════════════════════════════════
```

Diagnostic plots (5 panels):
1. World map with spacecraft ground track and tangent point locations
2. Velocity budget bar chart: v_sc, v_earth, v_corrected per frame
3. Direction cosines (L_E, L_N, L_Z) vs frame index
4. v_corrected residual histogram (null-wind test: centred on zero)
5. Wind solution confidence ellipses (1σ and 2σ) in (v_E, v_N) space

---

## 13. Dependencies

| Package | Use | Required |
|---------|-----|----------|
| `numpy` | Array math | Yes |
| `scipy.spatial.transform.Rotation` | Quaternion → rotation | Yes |
| `scipy.optimize.brentq` | LOS ray-trace to ellipsoid | Yes |
| `scipy.linalg.solve` | WLS inversion | Yes |
| `astropy.time.Time` | Unix ms → JD → GMST | Preferred |
| `matplotlib` | Diagnostic plots | Single-frame mode only |
| `cartopy` | World map | Optional (falls back to basic map) |

All matplotlib/cartopy imports are conditional on a `DIAGNOSTIC_MODE` flag.
Batch mode (`invert_wind_map.py`) has zero matplotlib imports.

---

## 14. Open issues

All v0.1 open issues (H07-01 through H07-06) are now resolved:

| ID | Resolution |
|----|-----------|
| H07-01 | Boresight confirmed: `[−1, 0, 0]_BRF` per AOCS report §2.4.2.1 and NB02a source |
| H07-02 | Quaternion convention confirmed: scalar-last `[x,y,z,w]` in pipeline/metadata; P01 handles binary re-ordering |
| H07-03 | GMST: use astropy with simple-GMST fallback (§4.6) |
| H07-04 | Vertical wind: L_Z × v_Z < 3 m/s — neglected with documentation (§7.1) |
| H07-05 | Bin defaults (5°×5°, 30 min): retained as defaults, configurable |
| H07-06 | Tidal binning (local-time × longitude): deferred to v0.3; `accumulate_days` parameter reserved |

No new open issues.

---

## 15. File locations

```
soc_sewell/
├── windcube/
│   ├── wind_retrieval.py          ← new module (this spec)
│   └── constants.py               ← add H07 constants if missing
├── tests/
│   └── test_h07_wind_retrieval.py ← new test file
└── specs/
    └── H07_wind_vector_retrieval_2026-05-14.md   ← this file
```

---

## 16. Instructions for Claude Code

Read this entire spec, P01 v2 (`P01_metadata_2026-05-14.md`), G01 v14
(`G01_synthetic_metadata_generator_2026-05-14.md`), NB02a
(`nb02a_boresight_2026_04_16.py`), and the AOCS report summary in this
spec before writing any code.

**Implementation order:**

1. Add any missing H07 constants to `windcube/constants.py` (§10).

2. Implement `windcube/wind_retrieval.py` in this order:
   `LOSGeometry` → `LOSObservation` → `WindSolution` →
   `gmst_rotation_matrix` → `derive_obs_mode` → `compute_los_geometry` →
   `correct_los_velocity` → `process_frame` → `invert_wind_vector` →
   `bin_observations`

3. Write `tests/test_h07_wind_retrieval.py` with at minimum:
   - **T1:** `derive_obs_mode` returns `"along_track"` for a synthetic
     along-track quaternion from NB02a
   - **T2:** `derive_obs_mode` returns `"cross_track"` for a synthetic
     cross-track quaternion from NB02a
   - **T3:** `compute_los_geometry` boresight matches NB02a LOS vector
     to < 0.001 rad for a known synthetic frame
   - **T4:** `correct_los_velocity` returns exactly 0.0 for a null-wind
     synthetic frame (v_rel = v_sc_harding + v_earth_harding)
   - **T5:** `invert_wind_vector` returns v_E ≈ 0, v_N ≈ 0 for a bin of
     null-wind corrected observations with diverse L_E, L_N direction cosines
   - **T6:** `two_sigma_v_E == 2.0 * sigma_v_E` exactly (no rounding)
   - **T7:** `invert_wind_vector` sets `gdop_flag=True` for a
     near-parallel set of observations (ill-conditioned ATA)
   - **T8:** `invert_wind_vector` sets `n_frames_flag=True` for fewer
     than N_MIN_FRAMES observations

4. Run: `pytest tests/test_h07_wind_retrieval.py -v`  
   All 8 tests must pass.

5. Run full suite: `pytest tests/ -v` — no regressions.

6. Commit:
   ```
   feat(h07): implement wind vector retrieval module, 8/8 tests pass
   Implements: H07_wind_vector_retrieval_2026-05-14.md (v0.2)
   Requires: S19 v2 (merged), G01 v14 (merged)
   ```

Module docstring header:
```python
"""
H07 — Thermospheric wind vector retrieval from FPI LOS Doppler velocities.

Spec:        specs/H07_wind_vector_retrieval_2026-05-14.md
Spec date:   2026-05-14
Generated:   <today>
Tool:        Claude Code
Last tested: <today>  (8/8 tests pass)

Boresight:   -X_BRF per SI-UCAR-WC-RP-004 §2.4.2.1 (AOCS Design Report)
Convention:  Harding (2014) — positive velocity = recession from spacecraft
Quaternion:  scalar-last [x,y,z,w] throughout (P01 pipeline convention)
"""
```

---

*End of H07 specification v0.2 — 2026-05-14*
