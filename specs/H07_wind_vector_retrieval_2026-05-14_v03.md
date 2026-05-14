# H07 — Wind Vector Retrieval
## WindCube SOC Pipeline — Specification v0.3
**Spec ID:** H07  
**Spec file:** `specs/H07_wind_vector_retrieval_2026-05-14_v03.md`  
**Previous version:** `specs/H07_wind_vector_retrieval_2026-05-14.md` (v0.2 — superseded)  
**Date:** 2026-05-14  
**Author:** Scott Sewell (HAO/NCAR)  
**Repo:** soc_sewell  
**Status:** Authoritative — v0.3  

**Depends on:**
- S19/P01 v2 (`P01_metadata_2026-05-14.md`) — `ImageMetadata` including `h_target_km_obs`
- G01 v14 (`G01_synthetic_metadata_generator_2026-05-14.md`) — synthetic dataset format, NB02c sign convention
- S20 (M08 L2 netCDF-4 writer) — output format for wind solutions
- NB02a (`nb02a_boresight_2026_04_16.py`) — boresight convention reference
- NB02c (`nb02c_los_projection_2026_04_16.py`) — **authoritative sign convention reference**

**Key reference documents:**
- SI-UCAR-WC-RP-004 WindCube AOCS Design Report v1.0 (Space Inventor, 2024-09-20)
- Harding, Gehrels & Makela (2014), *Applied Optics* 53(4), 666–672

---

## Revision history

| Version | Date | Change |
|---------|------|--------|
| v0.1 | 2025-05-14 | Initial draft. Boresight assumed `[0,0,−1]_BRF` (incorrect). |
| v0.2 | 2026-05-14 | Boresight corrected to `[−1,0,0]_BRF`. Sign convention resolved against Harding and GEN01 CSV. All v0.1 issues closed. **Superseded: correction formula §6 and design matrix §7.1 contained sign errors discovered by reading NB02c source.** |
| **v0.3** | **2026-05-14** | **Sign convention corrected throughout by reading NB02c source directly. Three sections changed: §2 (decomposition), §6 (correction formula), §7.1 (design matrix). All other sections unchanged from v0.2.** |

### v0.3 — what changed and why

Reading `nb02c_los_projection_2026_04_16.py` revealed the authoritative formula:

```
v_rel = v_wind_LOS − V_sc_LOS − v_earth_LOS
```

where all three terms are `dot(velocity, los_eci)` with `los_eci` pointing
SC→TP. The wind term has the **opposite sign** from the SC and Earth terms
because wind receding from the SC increases v_rel (redshift) while the SC
approaching the source decreases v_rel (blueshift). This sign asymmetry
propagates into two places in H07:

1. **§6 correction formula:** `v_corrected = v_rel + V_sc_LOS + v_earth_LOS`
   (not minus, because V_sc and v_earth are subtracted in the NB02c formula)

2. **§7.1 design matrix:** coefficients are `[+L_E, +L_N]` (not `[−L_E, −L_N]`)
   because `v_corrected = v_wind_LOS = dot(v_wind, l̂) = v_E·L_E + v_N·L_N`

v0.2 had both signs wrong, which would have produced winds with the correct
magnitude but **opposite direction** — a 180° azimuth error.

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
defines the measured LOS velocity as:

```
v_rel = (λ_c − λ_0) / λ_0  ×  c        [m/s]
```

where λ_0 = 630.0 nm is the OI rest wavelength and c = 299,792,458 m/s.
**Positive v_rel means the airglow source is receding from the spacecraft
(redshift, λ_c > λ_0). Negative v_rel means approaching (blueshift).**
This is the Harding convention used throughout the pipeline.

### 2.2 Three-component decomposition (authoritative — from NB02c)

All three LOS projections use the unit vector **l̂** pointing **from
spacecraft toward tangent point** (SC→TP), so a positive dot product means
motion toward the tangent point.

```
v_wind_LOS  = dot(v_wind_eci,  l̂)    [m/s]   + = wind toward TP
V_sc_LOS    = dot(vel_eci,     l̂)    [m/s]   + = SC toward TP
v_earth_LOS = dot(v_earth_eci, l̂)    [m/s]   + = Earth rotation toward TP
```

The NB02c formula (authoritative source, `nb02c_los_projection_2026_04_16.py`):

```
v_rel = v_wind_LOS − V_sc_LOS − v_earth_LOS
```

**Physical interpretation of the sign pattern:**
- Wind toward TP (away from SC) = source receding = redshift → v_wind_LOS > 0
  increases v_rel ✓
- SC toward TP (approaching source) = blueshift → V_sc_LOS > 0
  decreases v_rel ✓
- Earth rotation toward TP → v_earth_LOS > 0 decreases v_rel ✓

The three contributions:

| Term | Variable | Typical magnitude | Calculable? |
|------|----------|------------------|-------------|
| Thermospheric wind (target) | `v_wind_LOS` | 0–500 m/s | No — science product |
| Spacecraft orbital velocity | `V_sc_LOS` | ±7500 m/s | Yes — from metadata |
| Earth rotation at tangent point | `v_earth_LOS` | ±465 m/s (equator) | Yes — from geometry |

### 2.3 Corrected LOS velocity (wind component)

Rearranging the NB02c formula to isolate the wind:

```
v_corrected = v_wind_LOS = v_rel + V_sc_LOS + v_earth_LOS
```

`v_corrected` is positive when the wind blows **toward the tangent point**
(away from the spacecraft). This is the scalar quantity H07 inverts.

### 2.4 Relationship to GEN01 v14 CSV truth columns

The G01 v14 CSV stores all three dot-product components with the same
approach-positive sign (SC→TP positive). The mapping is direct — no sign
flip needed for the wind column:

```
v_wind_los_approach_ms  =  v_wind_LOS   (same sign, positive = toward TP)
v_sc_los_approach_ms    =  V_sc_LOS     (same sign)
v_earth_los_approach_ms =  v_earth_LOS  (same sign)
v_rel_ms                =  v_rel        (same sign, Harding convention)
```

Self-consistency check (C28 in G01 v14):
```
v_rel_ms = v_wind_los_approach_ms − v_sc_los_approach_ms − v_earth_los_approach_ms
```

For the null-wind validation: `v_wind_los_approach_ms = 0`, so
`v_corrected = v_rel + V_sc_LOS + v_earth_LOS = 0` exactly. ✓

### 2.5 Truth cross-check against ImageMetadata

`meta.truth_v_los` (set by GEN01) stores the wind LOS component in the
same convention as `v_wind_LOS`:

```
meta.truth_v_los = v_wind_los_approach_ms = v_corrected  (for noiseless data)
```

For null-wind dataset: `meta.truth_v_los = 0.0` and `v_corrected ≈ 0.0`. ✓

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

Rotating this by the attitude quaternion (BRF→ECI) gives `l̂`, the SC→TP
unit vector in ECI — the same `los_eci` used in NB02a and NB02c.

**This is the single most safety-critical constant in H07.** Any error here
produces a systematic velocity bias of order |V_sc| ≈ 7500 m/s.

### 3.2 Quaternion convention

The `attitude_quaternion` field in `ImageMetadata` is **scalar-last**
`[x, y, z, w]` throughout the pipeline. The binary `.bin` header stores
`[w, x, y, z]` (scalar-first); P01 re-orders at ingest. H07 always reads
from `ImageMetadata`, never from raw binary.

```python
from scipy.spatial.transform import Rotation
q = meta.attitude_quaternion          # [x, y, z, w] scalar-last
R_body2eci = Rotation.from_quat(q)    # scipy default is scalar-last
l_hat_eci = R_body2eci.apply(BORESIGHT_BODY)
l_hat_eci /= np.linalg.norm(l_hat_eci)    # unit vector SC→TP in ECI
```

### 3.3 Geometry confirmation

For the nominal 510 km altitude / 250 km tangent height mission geometry:
- Depression below local horizontal: **~15.8°**
- Angle from nadir: **~74.2°**
- Along-track mode: boresight projects strongly onto the velocity direction
- Cross-track mode: boresight projects strongly onto the orbit normal

Confirmed against GEN01 CSV: off-nadir = 74.07° for rows 0, 100, 500,
1000, 2000.

### 3.4 Coordinate frames

Three frames are used. All frame conversions must be explicit.

**ECI (Earth-Centered Inertial, J2000):**
- Origin: Earth centre of mass; axes fixed to inertial space
- Used for: spacecraft position/velocity, attitude quaternion, `l̂_eci`

**ECEF (Earth-Centered Earth-Fixed, WGS84):**
- Origin: Earth centre of mass; axes rotate with Earth
- Used for: Earth rotation velocity, tangent point geodetic coordinates
- Rotation from ECI: astropy GCRS↔ITRS transform (epoch-dependent;
  NB02c uses `_itrs_to_gcrs_vector` — H07 must use the same approach)

**ENU (East-North-Up at tangent point):**
- Origin: tangent point on WGS84 ellipsoid
- Axes: Ê (eastward), N̂ (northward), Ẑ (radially outward)
- Used for: decomposing v_wind into (v_E, v_N) direction cosines

---

## 4. Geometry engine (Stage G)

### 4.1 LOS vector in ECI

```python
BORESIGHT_BODY = np.array([-1.0, 0.0, 0.0])   # -X_BRF

q = meta.attitude_quaternion                    # [x, y, z, w] scalar-last
R = Rotation.from_quat(q)                       # BRF → ECI
l_hat_eci = R.apply(BORESIGHT_BODY)
l_hat_eci /= np.linalg.norm(l_hat_eci)         # unit SC→TP vector
```

### 4.2 Tangent point computation

**Target altitude:** Use `meta.h_target_km_obs` if not None; otherwise
default to 250.0 km and log a `UserWarning` (backward compatibility with
pre-v2 sidecars per P01 §3.10).

**Ray parameterisation:**
```
r(s) = r_sc + s × l_hat_eci      (s > 0 along SC→TP direction)
```

Find s* such that geodetic altitude of r(s*) equals `h_target_km_obs`.
Use `scipy.optimize.brentq` on interval `[0, 2 × |r_sc|]`.

**For synthetic frames:** If `meta.is_synthetic` and `meta.tangent_lat`
is not None, use the stored NB02b tangent point directly. Still compute
`l_hat_eci` from the quaternion and verify agreement with the stored
tangent point to < 0.1 km (validation check, not a hard gate).

**Output per frame** — `LOSGeometry` dataclass (§9.1).

### 4.3 ENU basis vectors at tangent point

NB02c uses astropy ITRS→GCRS for the ECEF→ECI rotation (epoch-dependent).
H07 must use the same approach for consistency. Given geodetic (φ, λ) at
the tangent point and the image epoch:

```python
# ECEF (ITRS) ENU unit vectors:
E_hat_itrs = np.array([-sin(λ),              cos(λ),          0.0     ])
N_hat_itrs = np.array([-sin(φ)*cos(λ), -sin(φ)*sin(λ),  cos(φ)])
Z_hat_itrs = np.array([ cos(φ)*cos(λ),  cos(φ)*sin(λ),  sin(φ)])

# Rotate to ECI using astropy (same as NB02c enu_unit_vectors_eci):
from astropy.time import Time
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation
import astropy.units as u

def itrs_to_eci(vec_itrs, epoch):
    cr = CartesianRepresentation(
        vec_itrs[0]*u.m, vec_itrs[1]*u.m, vec_itrs[2]*u.m)
    itrs = ITRS(cr, obstime=epoch)
    gcrs = itrs.transform_to(GCRS(obstime=epoch))
    xyz = np.array([gcrs.cartesian.x.to(u.m).value,
                    gcrs.cartesian.y.to(u.m).value,
                    gcrs.cartesian.z.to(u.m).value])
    return xyz / np.linalg.norm(xyz)

E_hat_eci = itrs_to_eci(E_hat_itrs, epoch)
N_hat_eci = itrs_to_eci(N_hat_itrs, epoch)
Z_hat_eci = itrs_to_eci(Z_hat_itrs, epoch)
```

Direction cosines (all dot products in ECI):

```python
L_E = float(np.dot(l_hat_eci, E_hat_eci))
L_N = float(np.dot(l_hat_eci, N_hat_eci))
L_Z = float(np.dot(l_hat_eci, Z_hat_eci))
```

### 4.4 Earth rotation contribution

Matches NB02c `earth_rotation_velocity_eci` exactly:

```python
from src.constants import EARTH_OMEGA_RAD_S
omega_vec = np.array([0.0, 0.0, EARTH_OMEGA_RAD_S])
v_earth_eci = np.cross(omega_vec, tangent_pos_eci)   # m/s in ECI
v_earth_LOS = float(np.dot(v_earth_eci, l_hat_eci))  # approach-positive
```

### 4.5 Spacecraft velocity contribution

```python
vel_eci  = np.array(meta.vel_eci_hat)                # ECI, m/s
V_sc_LOS = float(np.dot(vel_eci, l_hat_eci))         # approach-positive
```

Both `v_earth_LOS` and `V_sc_LOS` are stored in `LOSGeometry` with their
natural (approach-positive) sign, consistent with NB02c naming. The sign
convention is applied at correction time (§6).

---

## 5. obs_mode derivation from attitude quaternion (Stage M)

H07 can derive `obs_mode` independently from the attitude quaternion as a
cross-check against `meta.obs_mode`.

```python
def derive_obs_mode(
    attitude_quaternion: list[float],
    pos_eci: np.ndarray,
    vel_eci: np.ndarray,
    ambiguity_threshold: float = 0.707,   # cos(45°)
) -> str:
    """
    Derive 'along_track' or 'cross_track' from attitude quaternion.

    Projects the boresight [-1,0,0]_BRF (rotated to ECI) onto the forward
    velocity direction and the orbit normal. The larger projection identifies
    the look mode.

    For nominal 510km/250km geometry: expected projection ≈ cos(15.8°) ≈ 0.962
    for the correct mode vs ~sin(15.8°) ≈ 0.272 for the perpendicular.
    Classification margin: 0.962 − 0.707 = 0.255 (robust).
    """
    R = Rotation.from_quat(attitude_quaternion)
    boresight_eci = R.apply(np.array([-1.0, 0.0, 0.0]))
    boresight_eci /= np.linalg.norm(boresight_eci)

    nadir = -pos_eci / np.linalg.norm(pos_eci)

    v_hat = vel_eci / np.linalg.norm(vel_eci)
    v_horiz = v_hat - np.dot(v_hat, nadir) * nadir
    norm_vh = np.linalg.norm(v_horiz)
    if norm_vh < 1e-10:
        return "unknown"
    v_horiz /= norm_vh

    orbit_normal = np.cross(pos_eci, vel_eci)
    orbit_normal /= np.linalg.norm(orbit_normal)

    dot_v = abs(float(np.dot(boresight_eci, v_horiz)))
    dot_n = abs(float(np.dot(boresight_eci, orbit_normal)))

    if max(dot_v, dot_n) < ambiguity_threshold:
        return "unknown"

    return "along_track" if dot_v > dot_n else "cross_track"
```

**Use:** Call before processing each frame. If derived mode disagrees with
`meta.obs_mode` (and `meta.obs_mode != "unknown"`), log a warning and
proceed with `meta.obs_mode` (authoritative for real data). For synthetic
data, disagreement indicates a geometry bug and should raise an error.

---

## 6. Velocity correction (Stage C)

```python
def correct_los_velocity(
    v_rel: float,
    geom: LOSGeometry,
) -> float:
    """
    Remove spacecraft and Earth-rotation contributions from v_rel.

    From NB02c (authoritative):
        v_rel = v_wind_LOS − V_sc_LOS − v_earth_LOS

    Rearranging:
        v_wind_LOS = v_rel + V_sc_LOS + v_earth_LOS

    Returns v_corrected = v_wind_LOS:
        Positive = wind component toward tangent point (away from SC).
        This is the quantity projected by the design matrix in Stage I.

    Note the ADDITION of V_sc_LOS and v_earth_LOS — not subtraction.
    Both are stored approach-positive in LOSGeometry, consistent with NB02c.
    """
    return v_rel + geom.V_sc_LOS + geom.v_earth_LOS
```

**Null-wind validation:** For v_wind = 0 and noiseless geometry:
`v_corrected = v_rel + V_sc_LOS + v_earth_LOS = 0` exactly, because
NB02c defines `v_rel = 0 − V_sc_LOS − v_earth_LOS` when wind is zero. ✓

---

## 7. Wind vector inversion (Stage I)

### 7.1 Design matrix

`v_corrected = v_wind_LOS = dot(v_wind_eci, l̂) = v_E·L_E + v_N·L_N`

(The vertical term v_Z·L_Z is neglected: L_Z ≈ sin(15.8°) ≈ 0.272 and
v_Z < 10 m/s at 250 km, contributing < 2.7 m/s — acceptable.)

For N frames within one spatiotemporal bin:

```python
A = np.zeros((N, 2))
b = np.zeros(N)
w = np.zeros(N)

for i, obs in enumerate(observations):
    A[i, 0] = obs.L_E    # +L_E coefficient for v_E (NO minus sign)
    A[i, 1] = obs.L_N    # +L_N coefficient for v_N (NO minus sign)
    b[i]    = obs.v_corrected
    w[i]    = 1.0 / obs.sigma_v**2
```

**Sign note:** The coefficients are `+L_E, +L_N` because `v_corrected` is
the approach-positive wind projection, not the recession-positive Harding
v_rel. The inversion recovers geographic wind components directly:
`v_E > 0` = eastward wind, `v_N > 0` = northward wind.

### 7.2 Weighted least squares

```python
W   = np.diag(w)
ATA = A.T @ W @ A
ATb = A.T @ W @ b

cond = np.linalg.cond(ATA)
if cond > GDOP_MAX:
    return WindSolution(gdop_flag=True, ...)

x = np.linalg.solve(ATA, ATb)    # [v_E, v_N] in m/s
C = np.linalg.inv(ATA)           # 2×2 covariance matrix

v_E = x[0]                       # eastward wind, m/s (+ = eastward)
v_N = x[1]                       # northward wind, m/s (+ = northward)
sigma_v_E     = np.sqrt(C[0, 0])
sigma_v_N     = np.sqrt(C[1, 1])
two_sigma_v_E = 2.0 * sigma_v_E  # exactly 2 × sigma_v_E
two_sigma_v_N = 2.0 * sigma_v_N  # exactly 2 × sigma_v_N
```

**Contract:** `two_sigma_v_E == 2.0 * sigma_v_E` exactly (per uncertainty
standards addendum — no rounding or separate calculation).

### 7.3 Geometric dilution of precision (GDOP)

```python
GDOP_MAX = 100.0   # condition number threshold
```

Large GDOP occurs when all LOS vectors in a bin are nearly parallel —
the inversion cannot separate v_E from v_N. Expected to be rare in
nominal operations where along-track and cross-track orbits alternate.

### 7.4 Minimum frame count

```python
N_MIN_FRAMES = 4   # minimum frames per bin
```

---

## 8. Spatiotemporal binning (Stage B)

### 8.1 Bin definition

Each output wind vector is associated with a bin defined by:
- **Geographic location:** tangent point (lat, lon) gridded to
  `dlat × dlon` degree cells (default 5° × 5°)
- **Time:** bin centre UTC with width `dt_min` minutes (default 30 min)

A frame is assigned to the bin whose centre is closest to its tangent
point and whose time window contains its image epoch. Each frame
contributes to exactly one bin.

### 8.2 Multi-day accumulation mode

For DE3 tidal analysis (SQ2), data are accumulated over `n_days` (1–30)
with bins defined by (lat, local\_solar\_time, longitude). The
`accumulate_days` parameter switches this mode.
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

All functions in `windcube/wind_retrieval.py`.

### 9.1 Data structures

```python
@dataclass
class LOSGeometry:
    l_hat_eci:       np.ndarray    # (3,) SC→TP unit vector, ECI
    l_hat_ecef:      np.ndarray    # (3,) SC→TP unit vector, ECEF (via astropy)
    L_E:             float         # dot(l̂, Ê)  eastward direction cosine
    L_N:             float         # dot(l̂, N̂)  northward direction cosine
    L_Z:             float         # dot(l̂, Ẑ)  upward direction cosine
    tangent_lat_deg: float         # geodetic latitude of tangent point, deg
    tangent_lon_deg: float         # geodetic longitude of tangent point, deg
    tangent_alt_km:  float         # geodetic altitude of tangent point, km
    tangent_pos_eci: np.ndarray    # (3,) tangent point ECI position, m
    V_sc_LOS:        float         # dot(v_sc, l̂)    approach-positive, m/s
    v_earth_LOS:     float         # dot(v_earth, l̂) approach-positive, m/s


@dataclass
class LOSObservation:
    v_corrected:     float         # v_wind_LOS = v_rel + V_sc_LOS + v_earth_LOS, m/s
    sigma_v:         float         # 1-sigma uncertainty from M06, m/s
    L_E:             float         # eastward direction cosine
    L_N:             float         # northward direction cosine
    L_Z:             float         # upward direction cosine
    tangent_lat_deg: float         # deg
    tangent_lon_deg: float         # deg
    tangent_alt_km:  float         # km
    epoch_unix_ms:   int           # Unix ms
    obs_mode:        str           # 'along_track' or 'cross_track'


@dataclass
class WindSolution:
    v_E:                  float    # eastward wind, m/s (+ = eastward)
    v_N:                  float    # northward wind, m/s (+ = northward)
    sigma_v_E:            float    # 1-sigma uncertainty, m/s
    sigma_v_N:            float    # 1-sigma uncertainty, m/s
    two_sigma_v_E:        float    # = 2.0 * sigma_v_E exactly
    two_sigma_v_N:        float    # = 2.0 * sigma_v_N exactly
    n_frames:             int
    gdop_flag:            bool
    n_frames_flag:        bool
    condition_number:     float
    mean_tangent_lat_deg: float
    mean_tangent_lon_deg: float
    mean_tangent_alt_km:  float
    mean_epoch_unix_ms:   int
    obs_modes:            set
```

### 9.2 Public functions

**`compute_los_geometry(meta: ImageMetadata) → LOSGeometry`**  
Full Stage G (§4). Raises `ValueError` if `meta.img_type != "science"`.

**`derive_obs_mode(attitude_quaternion, pos_eci, vel_eci, ambiguity_threshold=0.707) → str`**  
Stage M (§5). Returns `"along_track"`, `"cross_track"`, or `"unknown"`.

**`correct_los_velocity(v_rel: float, geom: LOSGeometry) → float`**  
Stage C (§6). Returns `v_corrected = v_rel + V_sc_LOS + v_earth_LOS`.

**`process_frame(meta: ImageMetadata, v_rel: float, sigma_v: float) → LOSObservation`**  
Convenience wrapper: calls `compute_los_geometry`, `derive_obs_mode`, and
`correct_los_velocity`. Raises `ValueError` if `img_type != "science"` or
`adcs_quality_flag & SLEW_IN_PROGRESS`.

**`invert_wind_vector(observations: list[LOSObservation]) → WindSolution`**  
Stage I (§7). Returns NaN winds with flags set for ill-conditioned or
under-populated bins.

**`bin_observations(obs_list, dlat=5.0, dlon=5.0, dt_min=30.0) → dict`**  
Stage B (§8). Returns dict keyed by
`(lat_centre_deg, lon_centre_deg, t_centre_unix_ms)`.

**`gmst_rotation_matrix(unix_ms: int) → np.ndarray`**  
3×3 ECI→ECEF at given epoch. Provided for completeness; H07 preferentially
uses astropy GCRS↔ITRS transforms directly (same as NB02c).

---

## 10. Constants (in `windcube/constants.py`)

```python
EARTH_OMEGA_RAD_S   = 7.2921150e-5    # Earth rotation rate [rad/s]
                                       # Match NB02c: src.constants.EARTH_OMEGA_RAD_S
WGS84_A_M           = 6_378_137.0     # WGS84 equatorial radius [m]
WGS84_B_M           = 6_356_752.3142  # WGS84 polar radius [m]
OI_LAMBDA0_NM       = 630.0           # OI rest wavelength [nm]
OI_EMISSION_ALT_KM  = 250.0           # Nominal emission layer altitude [km]
C_M_S               = 299_792_458.0   # Speed of light [m/s]
GDOP_MAX            = 100.0           # Condition number threshold
N_MIN_FRAMES        = 4               # Minimum frames per bin
H07_BORESIGHT_BODY  = [-1.0, 0.0, 0.0]  # -X_BRF, AOCS report §2.4.2.1
```

**Note on `EARTH_OMEGA_RAD_S`:** NB02c imports this as
`src.constants.EARTH_OMEGA_RAD_S`. Verify the name matches; H07 must use
the same constant from the same module.

---

## 11. Null-wind validation test

For a GEN01 synthetic dataset with v_zonal = v_merid = 0 (e.g.
`GEN01_20270101_001_0d_uniform_seed0042.csv`):

### 11.1 Expected behaviour per stage

**Stage G:** `V_sc_LOS` varies ±7500 m/s across frames. `v_earth_LOS`
varies ±465×cos(lat) m/s. `l_hat_eci` matches NB02a LOS vector to < 0.001
rad for synthetic frames.

**Stage C:** `v_corrected = v_rel + V_sc_LOS + v_earth_LOS = 0.0` exactly
for noiseless synthetic frames. For frames with M06 photon noise:
`|v_corrected| < 3 × sigma_v` for 99.7% of frames.

**Stage I:** Recovered `v_E ≈ 0`, `v_N ≈ 0` for all well-conditioned bins.

### 11.2 Acceptance criteria

| Metric | Pass condition |
|--------|---------------|
| `mean(v_corrected)` over all frames | < 5 m/s systematic |
| `std(v_corrected)` | Consistent with mean σ_v from M06 |
| Recovered `v_E` per bin | Within 2σ of 0 for ≥ 95% of well-conditioned bins |
| Recovered `v_N` per bin | Within 2σ of 0 for ≥ 95% of well-conditioned bins |
| GDOP flags | < 20% of bins flagged |
| `derive_obs_mode` vs `meta.obs_mode` | 100% agreement for synthetic data |

### 11.3 Algebraic self-consistency check (T4)

For any synthetic science frame with null wind:

```python
V_sc_LOS    = np.dot(np.array(meta.vel_eci_hat), l_hat_eci)
v_earth_LOS = np.dot(np.cross([0,0,EARTH_OMEGA_RAD_S], tp_eci), l_hat_eci)
v_corrected = v_rel + V_sc_LOS + v_earth_LOS
assert abs(v_corrected) < 1e-6   # exact for noiseless geometry
```

---

## 12. Diagnostic output — single-frame mode

```
═══════════════════════════════════════════════════════════════
Frame: 2027-01-01T00:00:00Z  (science, along_track)
───────────────────────────────────────────────────────────────
Spacecraft position (ECI):     [-1918.9, 6891215.9, -14683.9] m
Spacecraft velocity (ECI):     [984.4, 8.2, 7543.2] m/s
Attitude quaternion [x,y,z,w]: [0.5352, 0.5905, -0.4622, 0.3890]
Boresight [-1,0,0]_BRF in ECI: [0.1245, -0.2724, 0.9541]
───────────────────────────────────────────────────────────────
Tangent point:  lat=15.58°  lon=-12.11°  alt=250.03 km
Direction cosines: L_E=+X.XXX  L_N=+X.XXX  L_Z=+X.XXX
───────────────────────────────────────────────────────────────
LOS velocity budget (approach-positive dot products):
  V_sc_LOS    (SC toward TP):         +XXXX.X m/s
  v_earth_LOS (Earth rot toward TP):    +XX.X m/s
  v_rel       (measured, Harding +):  -XXXX.X m/s
  v_corrected = v_rel + V_sc + v_earth:  +X.X m/s  (σ = X.X m/s)
───────────────────────────────────────────────────────────────
obs_mode stored:   along_track
obs_mode derived:  along_track  ✓
═══════════════════════════════════════════════════════════════
```

Diagnostic plots (5 panels):
1. World map — spacecraft ground track and tangent point locations
2. Velocity budget bar chart — V_sc_LOS, v_earth_LOS, v_corrected per frame
3. Direction cosines (L_E, L_N, L_Z) vs frame index
4. v_corrected residual histogram (null-wind: centred on zero)
5. Wind solution confidence ellipses (1σ, 2σ) in (v_E, v_N) space

---

## 13. Dependencies

| Package | Use | Required |
|---------|-----|----------|
| `numpy` | Array math | Yes |
| `scipy.spatial.transform.Rotation` | Quaternion → rotation | Yes |
| `scipy.optimize.brentq` | LOS ray-trace to ellipsoid | Yes |
| `scipy.linalg.solve` | WLS inversion | Yes |
| `astropy.time.Time` | Epoch handling | Yes |
| `astropy.coordinates` | GCRS↔ITRS (ECI↔ECEF) | Yes |
| `matplotlib` | Diagnostic plots | Single-frame mode only |
| `cartopy` | World map | Optional |

All matplotlib/cartopy imports conditional on `DIAGNOSTIC_MODE` flag.

---

## 14. Open issues

All issues resolved. None outstanding.

---

## 15. File locations

```
soc_sewell/
├── windcube/
│   ├── wind_retrieval.py          ← new module (this spec)
│   └── constants.py               ← verify EARTH_OMEGA_RAD_S present
├── tests/
│   └── test_h07_wind_retrieval.py ← new test file
└── specs/
    ├── H07_wind_vector_retrieval_2026-05-14_v03.md   ← this file
    └── archive/
        ├── H07_wind_vector_retrieval_2025-05-14.md   ← v0.1 archive
        └── H07_wind_vector_retrieval_2026-05-14.md   ← v0.2 archive
```

---

## 16. Instructions for Claude Code

Read this entire spec (v0.3), P01 v2, G01 v14, NB02a, and NB02c before
writing any code. Pay particular attention to the sign convention in §2
and §6 — v0.2 had these wrong.

**Implementation order:**

1. Verify `windcube/constants.py` has `EARTH_OMEGA_RAD_S` (same name as
   `src.constants.EARTH_OMEGA_RAD_S` used by NB02c). Add any missing
   constants from §10.

2. Implement `windcube/wind_retrieval.py` in this order:
   `LOSGeometry` → `LOSObservation` → `WindSolution` →
   `derive_obs_mode` → `compute_los_geometry` →
   `correct_los_velocity` → `process_frame` →
   `invert_wind_vector` → `bin_observations`

3. Write `tests/test_h07_wind_retrieval.py` with these 8 tests:
   - **T1:** `derive_obs_mode` → `"along_track"` for along-track quaternion
   - **T2:** `derive_obs_mode` → `"cross_track"` for cross-track quaternion
   - **T3:** `compute_los_geometry` boresight matches NB02a LOS vector
     to < 0.001 rad for a known synthetic frame
   - **T4:** `correct_los_velocity` returns 0.0 for null-wind synthetic
     frame: `v_corrected = v_rel + V_sc_LOS + v_earth_LOS = 0` exactly
   - **T5:** `invert_wind_vector` returns v_E ≈ 0, v_N ≈ 0 for null-wind
     bin with diverse L_E, L_N direction cosines
   - **T6:** `two_sigma_v_E == 2.0 * sigma_v_E` exactly
   - **T7:** `invert_wind_vector` sets `gdop_flag=True` for near-parallel
     observations (ill-conditioned ATA)
   - **T8:** `invert_wind_vector` sets `n_frames_flag=True` for fewer
     than N_MIN_FRAMES observations

4. Run: `pytest tests/test_h07_wind_retrieval.py -v` — all 8 must pass.

5. Run full suite: `pytest tests/ -v` — no regressions.

6. Commit:
   ```
   feat(h07): implement wind vector retrieval module, 8/8 tests pass
   Implements: H07_wind_vector_retrieval_2026-05-14_v03.md (v0.3)
   Key: correction formula v_corrected = v_rel + V_sc_LOS + v_earth_LOS
   Design matrix coefficients [+L_E, +L_N] (not minus)
   ```

Module docstring:
```python
"""
H07 — Thermospheric wind vector retrieval from FPI LOS Doppler velocities.

Spec:        specs/H07_wind_vector_retrieval_2026-05-14_v03.md (v0.3)
Spec date:   2026-05-14
Generated:   <today>
Tool:        Claude Code
Last tested: <today>  (8/8 tests pass)

Boresight:   -X_BRF per SI-UCAR-WC-RP-004 §2.4.2.1
Convention:  NB02c sign convention (nb02c_los_projection_2026_04_16.py)
             v_corrected = v_rel + V_sc_LOS + v_earth_LOS
             Design matrix: A[i] = [+L_E, +L_N]
Quaternion:  scalar-last [x,y,z,w] (P01 pipeline convention)
"""
```

---

*End of H07 specification v0.3 — 2026-05-14*
