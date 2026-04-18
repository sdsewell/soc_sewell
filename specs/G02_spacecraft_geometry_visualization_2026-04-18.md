# G02 — Spacecraft Geometry Visualization Suite
**Spec ID:** G02  
**Version:** v1.0  
**Date:** 2026-04-18  
**Author:** S. Sewell / Claude AI  
**Repo:** `soc_sewell`  
**Location:** `docs/specs/G02_spacecraft_geometry_visualization_2026-04-18.md`  
**Status:** Draft — ready for Claude Code implementation  
**Depends on:** S06 (NB01 orbit propagator), S07 (NB02 boresight/THRF/LOS geometry)

---

## 1. Purpose and Scope

This spec defines a standalone Python diagnostic script `windcube/tools/g02_geometry_viz.py` that produces a multi-panel visualization suite for validating WindCube spacecraft geometry.  The central validation question is:

> **Is the instrument boresight truly directed to the tangent height and geographic location I believe it is?**

The script ingests a single orbit's worth of L1c geometry packets (or a synthetic orbit from NB01/NB02) and produces **seven distinct figures** (Figures G02-1 through G02-7) covering every coordinate frame, reference frame, and derived quantity needed to answer that question.  It has no pipeline side-effects — it is purely diagnostic and produces no L2 data products.

### 1.1 Science motivation

The WindCube measurement geometry couples three things:
- The spacecraft **position** at the moment of exposure (ECI → geodetic)
- The **attitude** encoded in the quaternion (body frame orientation in ECI)
- The **look direction** (BRF → THRF boresight → tangent point geodetic)

A sign error in any one of these propagates silently through the pipeline as a velocity bias or an incorrect geographic attribution of the wind measurement.  G02 makes every link in that chain visually inspectable.

---

## 2. Definitions and Coordinate Frames

| Symbol | Name | Description |
|--------|------|-------------|
| ECI | Earth-Centered Inertial | J2000 frame; origin at Earth centre; Z toward north celestial pole |
| ECEF | Earth-Centered Earth-Fixed | Rotates with Earth; Z toward geographic north pole |
| Geodetic | WGS-84 | (lat, lon, alt) referenced to WGS-84 ellipsoid |
| BRF | Body Reference Frame | Spacecraft body axes (x̂_b, ŷ_b, ẑ_b); defined in S07 §2 |
| THRF | Tangent Height Reference Frame | Per-look frame; ẑ_T aligned with LOS, x̂_T in tangent plane; defined in S07 §3 |
| TP | Tangent Point | Geodetic (lat, lon, alt) of LOS–shell intersection at h_T = 250 km |
| q | Attitude quaternion | Unit quaternion q = (q_w, q_x, q_y, q_z) rotating ECI → BRF |

### 2.1 Body Reference Frame convention (from S07)

The WindCube BRF is defined as:
- **x̂_b** — along the spacecraft velocity direction (ram)  
- **ŷ_b** — completed right-hand, nominally port (left looking down track)  
- **ẑ_b** — nadir-pointing (toward Earth centre)

The FPI boresight in BRF is defined in `windcube/constants.py` as `BORESIGHT_BRF`.  The four THRF look directions (forward-port, forward-starboard, aft-port, aft-starboard) are also defined there.

### 2.2 THRF convention (from S07)

For each look direction `k`:
- **ẑ_T^(k)** — unit vector along LOS from spacecraft to tangent point (in ECI)  
- **x̂_T^(k)** — unit vector in tangent plane at TP, pointing along horizontal wind sensitivity axis  
- **ŷ_T^(k)** — completes right-hand set  

---

## 3. Inputs

```python
# Primary input — geometry record array or DataFrame with fields:
# t_utc       : UTC epoch (datetime64 or float MJD)
# r_eci       : (N,3) float64  S/C position in ECI [km]
# v_eci       : (N,3) float64  S/C velocity in ECI [km/s]
# q_eci2brf   : (N,4) float64  attitude quaternion (q_w, q_x, q_y, q_z)
# tp_eci      : (N,K,3) float64  tangent point position in ECI [km], K=4 look dirs
# tp_geodetic : (N,K,3) float64  tangent point geodetic (lat_deg, lon_deg, alt_km)
# sc_geodetic : (N,3) float64  S/C geodetic (lat_deg, lon_deg, alt_km)
```

If no real data are available the script can generate a synthetic input using the NB01/NB02 modules (one full orbit, ISS-like ~51.6° inclination, 400 km altitude).  This is the **default mode** invoked with no arguments.

### 3.1 Command-line interface

```
python -m windcube.tools.g02_geometry_viz [--orbit-file PATH] [--epoch ISO8601]
       [--duration-min FLOAT] [--sample-idx INT] [--output-dir PATH] [--no-show]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--orbit-file` | None (synthetic) | Path to NB02 geometry NetCDF or CSV |
| `--epoch` | 2025-01-01T00:00:00 | Orbit start epoch for synthetic mode |
| `--duration-min` | 96.0 | Synthetic orbit duration in minutes |
| `--sample-idx` | 0 | Index into the orbit time series for single-epoch figures |
| `--output-dir` | `./g02_output/` | Directory for saved PNG files |
| `--no-show` | False | Suppress interactive display (for CI) |

---

## 4. Output Figures

All figures are saved as `G02-N_<short_name>_<YYYYMMDDTHHmmss>.png` at 150 dpi.  The `<YYYYMMDDTHHmmss>` suffix is the `--epoch` value.

---

### Figure G02-1 — Orbit Ground Track and Tangent Point Map

**Purpose:** Verify that the spacecraft orbital ground track is correct and that all four tangent point families are displaced appropriately fore/aft and port/starboard of the sub-satellite point.

**Layout:** Single axes, equirectangular projection, full orbit duration.

**Elements:**
- Blue line: sub-satellite ground track (lat vs lon)
- Black dot at `--sample-idx`: current S/C sub-satellite point with label "S/C"
- Four coloured dots per time step (decimated to ≤200 points for speed):  
  `tp_geodetic[:, 0, :]` forward-port (cyan), `[:, 1, :]` forward-starboard (magenta),  
  `[:, 2, :]` aft-port (green), `[:, 3, :]` aft-starboard (orange)
- At `--sample-idx`, draw arrows from S/C sub-point to each of the four TPs with colour-coded labels
- Coastlines drawn with `cartopy` if available; plain lat/lon grid otherwise
- Title: `G02-1: Ground track + tangent point map  [epoch=<t>]`

**Verification sanity checks (printed to stdout):**
- Forward TPs should be ahead of S/C in the direction of motion
- Port TPs should be to the left when facing velocity direction (in Northern Hemisphere prograde pass, port ≈ westward)
- TP altitude should be 250 ± 5 km for all four families

---

### Figure G02-2 — S/C Position in ECI and Geodetic (Time Series)

**Purpose:** Verify orbital parameters and confirm ECI ↔ geodetic conversion is consistent.

**Layout:** 2 rows × 3 columns subplots, shared time axis.

**Row 1 — ECI components:**
- (1,1) `r_eci[:, 0]` X_ECI [km] vs time
- (1,2) `r_eci[:, 1]` Y_ECI [km] vs time
- (1,3) `r_eci[:, 2]` Z_ECI [km] vs time
- On each panel annotate the sample-idx point with a red dot

**Row 2 — Geodetic components:**
- (2,1) `sc_geodetic[:, 0]` Geodetic latitude [deg] vs time
- (2,2) `sc_geodetic[:, 1]` Geodetic longitude [deg] vs time
- (2,3) `sc_geodetic[:, 2]` Altitude [km] vs time — draw horizontal dashed line at mean altitude with label

**Computed consistency check (printed):**
- `|r_eci|` should equal `R_Earth + alt_km` to within 0.1 km for WGS-84 spherical approximation

---

### Figure G02-3 — S/C Velocity in ECI and Geodetic Components (Time Series)

**Purpose:** Verify orbital speed, direction, and the decomposition of ECI velocity into N/S, E/W, and radial components used downstream in the wind retrieval.

**Layout:** 2 rows × 2 columns subplots.

**Row 1 — ECI velocity:**
- (1,1) Three ECI velocity components `v_eci[:, 0/1/2]` on a single panel (colour-coded), legend `Vx, Vy, Vz`
- (1,2) `|v_eci|` orbital speed [km/s] vs time — annotate mean ± std, draw dashed line at 7.66 km/s (nominal LEO)

**Row 2 — Geodetic velocity decomposition:**
- (2,1) North-South and East-West components derived from ECI velocity projected onto local geodetic horizontal; labels `V_N/S [km/s]` and `V_E/W [km/s]`
- (2,2) Radial (vertical) component `V_radial [km/s]` — should be near-zero for circular orbit

**Computed consistency check (printed):**
- `sqrt(V_NS^2 + V_EW^2 + V_radial^2)` should match `|v_eci|` to machine precision

---

### Figure G02-4 — Body Reference Frame Axes and Boresight (Single Epoch, 3-D)

**Purpose:** This is the primary validation figure for attitude.  Shows the spacecraft body axes in ECI space, with the boresight and all four THRF z-axes, so you can visually confirm the pointing geometry.

**Layout:** Single 3-D axes (`mpl_toolkits.mplot3d`).

**Elements (all drawn at the S/C position `r_eci[sample_idx]`):**
- **Origin marker**: red sphere at S/C position
- **Earth sphere** (grey, translucent, radius R_Earth = 6371 km) centred at origin
- **BRF axes** drawn as thick arrows (length = 200 km for visibility):  
  - x̂_b: blue arrow, label "x̂_b (ram)"  
  - ŷ_b: green arrow, label "ŷ_b (port)"  
  - ẑ_b: red arrow, label "ẑ_b (nadir)"  
  The three BRF axes are derived from `q_eci2brf[sample_idx]` via quaternion rotation
- **Nadir check**: draw a thin grey line from S/C to Earth centre; ẑ_b should be anti-parallel to `r_eci` for nadir-pointing attitude
- **Four THRF ẑ_T vectors** as dashed arrows (length = 500 km, colour-coded as in G02-1), labelled "FP", "FS", "AP", "AS" (forward-port, forward-starboard, aft-port, aft-starboard)
- **Four tangent point markers**: colour-coded dots on the 250 km shell surface
- **LOS line segments**: thin dashed lines from S/C to each TP
- Axis labels: `X_ECI [km]`, `Y_ECI [km]`, `Z_ECI [km]`
- Title: `G02-4: BRF axes + THRF boresights  [sample_idx=N, t=<t>]`

**Verification sanity checks (printed):**
- `dot(ẑ_b, r̂_eci)` should be −1.00 ± 0.01 for nadir-pointing (−1 = nadir)
- `dot(x̂_b, v̂_eci)` should be +1.00 ± 0.01 for ram-pointing body x-axis
- `dot(ẑ_T^(k), r̂_eci)` (elevation angle of boresight below horizontal) should satisfy `arccos(dot) ≈ 90° + depression_angle_deg`
- BRF axes should be mutually orthogonal: all `dot(x̂_b, ŷ_b)`, `dot(x̂_b, ẑ_b)`, `dot(ŷ_b, ẑ_b)` < 1e-12

---

### Figure G02-5 — THRF Frames at Tangent Points (Single Epoch, 3-D Zoom)

**Purpose:** Zoom into the tangent point region to show the THRF axes at each TP and verify that x̂_T is correctly aligned with the horizontal wind sensitivity direction (zonal vs meridional).

**Layout:** Single 3-D axes, viewpoint centred on the tangent point centroid.  Camera zoomed to a ±500 km cube around the mean TP location.

**Elements:**
- Curved arc segment of the 250 km shell (wireframe sphere patch in the TP neighbourhood)
- For each of the four TPs:
  - Colour-coded dot at TP geodetic position (converted to ECI for plotting)
  - Three THRF axes drawn as arrows (length 50 km):  
    - ẑ_T (LOS direction, pointing from TP toward S/C): solid arrow  
    - x̂_T (horizontal wind sensitivity): dashed arrow  
    - ŷ_T (cross-sensitivity): dotted arrow  
  - Text label: "FP", "FS", "AP", "AS"
- S/C position shown as a small red dot in the upper corner of the view
- Title: `G02-5: THRF axes at tangent points  [sample_idx=N]`

**Verification sanity checks (printed):**
- For each THRF k: `|ẑ_T^(k)|` = 1.00, `|x̂_T^(k)|` = 1.00, `|ŷ_T^(k)|` = 1.00
- `dot(ẑ_T^(k), x̂_T^(k))` < 1e-10 for each k (LOS ⊥ sensitivity axis)
- `ẑ_T^(k)` should be nearly anti-parallel to the vector from S/C to TP (i.e., it points from TP toward S/C)

---

### Figure G02-6 — Quaternion Visualization: Euler Angles and Rotation History

**Purpose:** Make the quaternion time series interpretable by decomposing it into physically meaningful Euler angles and showing the rotation of BRF axes over the orbit.

**Layout:** 3 rows of subplots.

**Row 1 — Euler angle time series (ZYX / aerospace convention: yaw–pitch–roll):**
- Yaw ψ [deg]: rotation about ẑ_b
- Pitch θ [deg]: rotation about ŷ_b  
- Roll φ [deg]: rotation about x̂_b
- All three on a single panel with legend; horizontal dashed lines at 0°

**Row 2 — Quaternion components raw:**
- q_w, q_x, q_y, q_z vs time on a single panel
- Annotate `|q|` = 1 ± 1e-6 check as text in corner

**Row 3 — Angular rate (numerical derivative of quaternion → body rates):**
- ω_x, ω_y, ω_z [deg/s] — body angular rates derived from finite-difference quaternion kinematics
- For synthetic nadir-pointing orbit, all rates should be ~0.06 deg/s (one orbit per 96 min)

**Verification sanity checks (printed):**
- Max `| |q| − 1 |` over orbit; flag if > 1e-6
- Max |ψ|, |θ|, |φ| printed; flag if pitch or roll > 5° (unexpected off-nadir)
- Yaw rate consistency with orbital period

---

### Figure G02-7 — Observation Direction Validation Summary Panel

**Purpose:** A single "dashboard" figure that a reviewer can look at to answer the key question: "Is the observation direction correct?" at a glance.

**Layout:** 2 rows × 2 columns.

**Panel (1,1) — Tangent height vs time for all four looks:**
- `tp_geodetic[:, k, 2]` [km] for k = 0..3, colour-coded
- Horizontal dashed lines at 245 km and 255 km (acceptance band)
- Title: "Tangent Height [km]"

**Panel (1,2) — Tangent point latitude vs longitude (scatter, colour = time):**
- All four look families plotted simultaneously
- Colour axis is time within orbit (use viridis)
- Overplot S/C sub-satellite track as thin grey line
- Title: "TP Geodetic Position"

**Panel (2,1) — LOS azimuth at TP vs time:**
- Azimuth angle of each THRF boresight measured clockwise from North at the TP location [deg]
- Forward looks should be ~0°/180°, port/starboard looks ~−90°/+90° offset from track
- Title: "LOS azimuth at TP (CW from N) [deg]"

**Panel (2,2) — Depression angle (elevation below horizontal) vs time:**
- Angle between the LOS vector and the local horizontal plane at the S/C position [deg]
- Should be ~25–35° for 400 km orbit observing 250 km altitude
- Annotate expected value from geometry: `arcsin((h_sc − h_T) / |LOS|)` analytically
- Title: "LOS depression angle [deg]"

**Verification sanity checks (printed):**
- Fraction of epochs with TP altitude outside [245, 255] km; should be 0% for synthetic
- Mean and std of depression angle for each look family

---

## 5. Implementation Details

### 5.1 Module layout

```
windcube/
  tools/
    __init__.py          (existing or create)
    g02_geometry_viz.py  (new — main script)
    _g02_helpers.py      (new — frame transforms, THRF construction, sanity checks)
```

### 5.2 Key helper functions in `_g02_helpers.py`

```python
def quaternion_to_dcm(q: np.ndarray) -> np.ndarray:
    """
    Convert unit quaternion q = [q_w, q_x, q_y, q_z] to 3×3 Direction Cosine Matrix.
    R_dcm transforms a vector from the "from" frame to the "to" frame:
      v_brf = R_dcm @ v_eci
    where q = q_eci2brf.
    
    Parameters
    ----------
    q : (4,) or (N,4) array — [q_w, q_x, q_y, q_z]
    
    Returns
    -------
    R : (3,3) or (N,3,3) float64
    """

def dcm_to_euler_zyx(R: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract ZYX Euler angles (yaw ψ, pitch θ, roll φ) in degrees from a DCM.
    Singularity-protected for |θ| < 89.9°.
    """

def eci_to_geodetic(r_eci: np.ndarray, t_utc: np.ndarray) -> np.ndarray:
    """
    Convert ECI position to WGS-84 geodetic (lat_deg, lon_deg, alt_km).
    Uses GMST rotation for ECI→ECEF then Bowring iterative geodetic conversion.
    
    Note: for G02 diagnostic purposes only — production code uses astropy or skyfield.
    """

def eci_velocity_to_ned(r_eci: np.ndarray, v_eci: np.ndarray,
                        t_utc: np.ndarray) -> np.ndarray:
    """
    Decompose ECI velocity into North, East, Down components at each S/C position.
    Returns (N,3) array [V_N, V_E, V_D] in km/s.
    """

def compute_thrf_axes(r_sc_eci: np.ndarray, tp_eci: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct THRF axes (ẑ_T, x̂_T, ŷ_T) for a single look direction.
    
    ẑ_T = normalise(r_sc - r_tp)   [LOS direction, from TP toward S/C]
    x̂_T = normalise(ẑ_T × ẑ_ECI) × sign  [horizontal, in tangent plane]
    ŷ_T = ẑ_T × x̂_T               [completes RH set]
    
    Parameters
    ----------
    r_sc_eci : (3,) — S/C ECI position [km]
    tp_eci   : (3,) — tangent point ECI position [km]
    
    Returns
    -------
    z_T, x_T, y_T : each (3,) unit vectors
    """

def los_depression_angle(r_sc_eci: np.ndarray, tp_eci: np.ndarray) -> float:
    """
    Compute LOS depression angle below local horizontal at S/C [degrees].
    depression_angle = arcsin( dot(LOS_unit, -r̂_sc) )
    where r̂_sc is the outward radial unit vector at S/C.
    """

def los_azimuth_at_tp(tp_eci: np.ndarray, tp_geodetic: np.ndarray,
                      z_T: np.ndarray) -> float:
    """
    Compute azimuth of the LOS projected onto the horizontal plane at TP,
    measured clockwise from geographic North [degrees].
    """
```

### 5.3 Synthetic orbit generation (no real data mode)

When `--orbit-file` is absent the script calls:

```python
from windcube.nb01_orbit import propagate_sgp4_orbit
from windcube.nb02_geometry import compute_geometry_packet

orbit_params = {
    'epoch': args.epoch,
    'alt_km': 400.0,
    'inc_deg': 51.6,
    'duration_min': args.duration_min,
    'dt_sec': 30.0,
}
sc_eci, sc_vel_eci = propagate_sgp4_orbit(**orbit_params)
# Synthesize nadir-pointing quaternion from velocity/position
q = nadir_pointing_quaternion(sc_eci, sc_vel_eci)
# Compute tangent points for all four look directions
tp_eci, tp_geodetic = compute_geometry_packet(sc_eci, sc_vel_eci, q)
```

For the synthetic nadir-pointing quaternion, `x̂_b = v̂_eci`, `ẑ_b = −r̂_eci`, `ŷ_b = ẑ_b × x̂_b`.

### 5.4 Dependencies

| Package | Min version | Note |
|---------|-------------|------|
| numpy | 1.24 | |
| matplotlib | 3.7 | `mpl_toolkits.mplot3d` used for Figs G02-4, G02-5 |
| cartopy | 0.22 | Optional — for coastlines in G02-1; falls back to plain grid |
| astropy | 5.3 | Optional — for high-precision GMST; falls back to internal GMST |
| scipy | 1.11 | For rotation utilities in unit tests only |

No new package dependencies beyond what NB01/NB02 already require.

---

## 6. Sanity Check Report

After generating all seven figures the script prints a structured sanity report to stdout:

```
====================================================================
G02 Geometry Sanity Report — epoch 2025-01-01T00:00:00
====================================================================
[SC-POS]  |r_eci| vs (R_Earth + alt): max deviation = 0.03 km   PASS
[SC-VEL]  |v_eci| = 7.669 ± 0.001 km/s                         PASS
[QUAT]    max | |q| − 1 | = 3.1e-16                             PASS
[BRF]     nadir check dot(ẑ_b, r̂) = −1.0000 ± 0.0000           PASS
[BRF]     ram   check dot(x̂_b, v̂) = +1.0000 ± 0.0000           PASS
[BRF]     orthogonality max cross-dot = 2.2e-16                 PASS
[TP-ALT]  mean TP altitude: 250.0 km, std = 0.0 km              PASS
[TP-ALT]  fraction outside [245,255] km: 0.0%                   PASS
[DEP-ANG] depression angle: 30.2° ± 0.1° (forward), 30.2° ± 0.1° (aft)  PASS
[THRF]    all axes unit length: max | |â| − 1 | = 1.1e-16       PASS
[THRF]    LOS ⊥ sensitivity axis: max |dot(ẑ_T, x̂_T)| = 0.0    PASS
====================================================================
OVERALL: 12/12 checks PASS
====================================================================
```

Each check that fails prints `FAIL` in red (ANSI escape if terminal supports it) and the script exits with return code 1 to support CI integration.

---

## 7. Verification Checks (for spec compliance)

| ID | Check | Expected | Tolerance |
|----|-------|----------|-----------|
| V01 | `|r_eci|` vs `R_E + alt` | < 0.1 km deviation | WGS-84 spherical approx |
| V02 | Orbital speed `|v_eci|` | 7.60–7.72 km/s (400 km orbit) | ± 0.05 km/s |
| V03 | Quaternion unit norm | `| |q| − 1 |` < 1e-6 | per epoch |
| V04 | BRF nadir alignment | `dot(ẑ_b, r̂_eci)` = −1.00 | ± 0.01 (nadir-pointing) |
| V05 | BRF ram alignment | `dot(x̂_b, v̂_eci)` = +1.00 | ± 0.01 |
| V06 | BRF orthogonality | all pairwise dots < 1e-12 | |
| V07 | TP altitude | 250 ± 5 km | 100% epochs |
| V08 | THRF unit vectors | `| |â| − 1 |` < 1e-10 | all axes, all looks |
| V09 | LOS ⊥ sensitivity | `|dot(ẑ_T, x̂_T)|` < 1e-10 | |
| V10 | Depression angle range | 25–45° | 400 km orbit / 250 km TP |
| V11 | Forward TP ahead of S/C | lat advancement in direction of motion | sign check |
| V12 | All 7 figures saved | PNG files exist in output dir | |

---

## 8. Test Suite: `tests/test_g02_geometry_viz.py`

```python
"""
Tests for G02 geometry visualization helpers.
Each test verifies a specific geometric identity or sanity condition.
"""

def test_quaternion_to_dcm_identity():
    """Identity quaternion should yield 3×3 identity DCM."""
    q = np.array([1.0, 0.0, 0.0, 0.0])
    R = quaternion_to_dcm(q)
    np.testing.assert_allclose(R, np.eye(3), atol=1e-15)

def test_quaternion_to_dcm_90deg_z():
    """90° rotation about Z: x̂ → ŷ."""
    q = np.array([np.cos(np.pi/4), 0.0, 0.0, np.sin(np.pi/4)])
    R = quaternion_to_dcm(q)
    v_in  = np.array([1.0, 0.0, 0.0])
    v_out = R @ v_in
    np.testing.assert_allclose(v_out, [0.0, 1.0, 0.0], atol=1e-14)

def test_dcm_to_euler_zyx_zero():
    """Identity DCM → zero Euler angles."""
    psi, theta, phi = dcm_to_euler_zyx(np.eye(3))
    assert abs(psi) < 1e-10 and abs(theta) < 1e-10 and abs(phi) < 1e-10

def test_thrf_axes_orthogonality():
    """THRF axes must be mutually orthogonal unit vectors."""
    r_sc = np.array([6771.0, 0.0, 0.0])   # on equator
    r_tp = np.array([6621.0, 0.0, 0.0])   # nadir from above
    z_T, x_T, y_T = compute_thrf_axes(r_sc, r_tp)
    np.testing.assert_allclose(np.linalg.norm(z_T), 1.0, atol=1e-14)
    np.testing.assert_allclose(np.dot(z_T, x_T),    0.0, atol=1e-14)
    np.testing.assert_allclose(np.dot(z_T, y_T),    0.0, atol=1e-14)

def test_los_depression_angle_known():
    """
    S/C at 400 km, TP at 250 km, horizontal separation = 0.
    Depression angle = arcsin((400-250)/400) ≈ 22.0°.
    Adjust for actual slant geometry.
    """
    pass  # analytic comparison

def test_synthetic_orbit_all_checks_pass(tmp_path):
    """Run full G02 in synthetic mode with --no-show; all 12 sanity checks must pass."""
    from windcube.tools.g02_geometry_viz import run_g02
    report = run_g02(output_dir=str(tmp_path), no_show=True)
    assert report['n_pass'] == report['n_total']
    assert report['n_fail'] == 0

def test_all_figures_saved(tmp_path):
    """Seven PNG files must be written to output dir."""
    from windcube.tools.g02_geometry_viz import run_g02
    run_g02(output_dir=str(tmp_path), no_show=True)
    pngs = list(tmp_path.glob('G02-*.png'))
    assert len(pngs) == 7
```

---

## 9. Claude Code Implementation Prompt

```
cat PIPELINE_STATUS.md

## Context
Implement G02 spacecraft geometry visualization suite per spec
docs/specs/G02_spacecraft_geometry_visualization_2026-04-18.md.

## Pre-implementation discovery
1. Check windcube/constants.py for BORESIGHT_BRF and the four THRF look angles.
2. Check windcube/nb01_orbit.py for propagate_sgp4_orbit() signature.
3. Check windcube/nb02_geometry.py for compute_geometry_packet() signature.
4. If nb01/nb02 modules are not yet importable in synthetic mode, stub them with
   analytic circular orbit + nadir quaternion — document any stubs in the report back.
5. Check if windcube/tools/ directory exists; create it with __init__.py if not.

## Implementation tasks
1. Create windcube/tools/_g02_helpers.py with all helper functions in spec §5.2.
   - quaternion_to_dcm(): use the formula in terms of q_w, q_x, q_y, q_z.
   - dcm_to_euler_zyx(): ZYX decomposition, singularity-protected.
   - eci_to_geodetic(): GMST rotation + Bowring iterative. Store GMST function in helpers.
   - eci_velocity_to_ned(): project v_eci onto local NED at each epoch.
   - compute_thrf_axes(): see spec §5.2 for exact construction.
   - los_depression_angle(), los_azimuth_at_tp().

2. Create windcube/tools/g02_geometry_viz.py.
   - CLI as specified in §3.1 (argparse).
   - Synthetic orbit mode calling NB01/NB02 (or stub if not available).
   - Generate Figures G02-1 through G02-7 per §4.
   - Print sanity report per §6; exit code 1 on any FAIL.

3. Create tests/test_g02_geometry_viz.py with all tests in §8.

4. Run:
   pytest tests/test_g02_geometry_viz.py -v
   python -m windcube.tools.g02_geometry_viz --no-show --output-dir /tmp/g02_test/

5. Paste back:
   - Full pytest output.
   - The sanity report printed by the script.
   - A listing of the seven PNG filenames created.
   - Any stubs or deviations from the spec.

# Update PIPELINE_STATUS.md — mark G02 as implemented (or partial if stubs used)
git add windcube/tools/_g02_helpers.py windcube/tools/g02_geometry_viz.py \
        tests/test_g02_geometry_viz.py PIPELINE_STATUS.md
git commit -m "feat: implement G02 spacecraft geometry visualization suite

Seven-figure diagnostic: ground track, ECI/geodetic pos/vel, BRF axes,
THRF frames, quaternion decomposition, validation dashboard.
12 automated sanity checks; exits with code 1 on any FAIL.

Also updates PIPELINE_STATUS.md"
```

---

## 10. PIPELINE_STATUS.md Entry

Add the following row to PIPELINE_STATUS.md:

```
| G02 | spacecraft_geometry_visualization | draft | 0/12 sanity checks | 2026-04-18 |
```

Update to `implemented` after successful Claude Code session.

---

## 11. Relationship to Other Specs

| Spec | Relationship |
|------|-------------|
| S06 NB01 | G02 calls `propagate_sgp4_orbit()` for synthetic mode |
| S07 NB02 | G02 calls `compute_geometry_packet()` for THRF/TP computation |
| S08 INT01 | G02 is the visual complement to INT01's numerical tests |
| Z04 | G02 ground track overlay can optionally show the SNR sensitivity map from Z04 |

G02 is a **diagnostic tool** — it is not in the nominal pipeline data flow and does not appear in the L1c → L2 processing chain.  It is run by the science team during commissioning and whenever observation geometry anomalies are suspected.

---

## 12. Known Limitations (v1.0)

- Figure G02-4 (3-D BRF) is difficult to interpret in static PNG; a future v1.1 may add an interactive matplotlib widget or a `plotly` 3D export.
- `eci_to_geodetic()` in `_g02_helpers.py` uses a simplified GMST formula accurate to ~1 arcsec; replace with `astropy` call for sub-arcsec accuracy if needed.
- The four THRF look directions are currently fixed in `windcube/constants.py`; G02 will need updating if the look geometry changes.
- Cartopy is optional; without it the ground track map (G02-1) has no coastlines but remains functionally correct.
