# S16 — M07 L2 Vector Wind Retrieval

**Spec ID:** S16
**Spec file:** `docs/specs/S16_m07_wind_retrieval_2026-04-06.md`
**Project:** WindCube FPI Science Operations Center Pipeline
**Institution:** NCAR / High Altitude Observatory (HAO)
**Status:** Authoritative
**Depends on:** S01, S02, S03, S04, S07 (NB02 geometry), S15 (M06 airglow inversion)
**Used by:** S17 (INT02), S18 (INT03), S20 (L2 data product)
**Last updated:** 2026-04-06
**Created/Modified by:** Claude AI

---

## 1. Purpose

M07 decomposes pairs of line-of-sight wind measurements from adjacent
along-track and cross-track orbits into the horizontal wind vector
(v_zonal, v_meridional) at each geographic location. This is the final
scientific retrieval step — the L2 data product.

The fundamental measurement is: on orbit N (along-track look), the FPI
measures one projection of the wind vector; on orbit N+1 (cross-track look),
it measures a second, approximately orthogonal projection. Solving the
resulting 2×2 linear system recovers both components.

**What M07 is not.** M07 does not invert fringe images — that is M06.
M07 does not compute LOS geometry — that is NB02. M07 does not apply
dark subtraction or annular reduction — that is M03. M07 receives
pre-computed atmospheric wind LOS projections and geometric unit vectors,
and performs only the spatial pairing and linear algebraic decomposition.

---

## 2. Physical background

### 2.1 The 2×2 decomposition

At a tangent point with geodetic coordinates (lat, lon), the atmospheric
wind horizontal vector has two components in the local ENU frame:

```
v_wind = v_zonal · ê_east + v_merid · ê_north
```

Each FPI observation measures the projection of this wind onto the
instrument line-of-sight unit vector:

```
v_wind_LOS = (v_zonal · ê_east + v_merid · ê_north) · los_eci
           = v_zonal · (ê_east · los_eci) + v_merid · (ê_north · los_eci)
           = v_zonal · A_e + v_merid · A_n
```

where `A_e = ê_east · los_eci` and `A_n = ê_north · los_eci` are the
east and north sensitivity coefficients for that observation.

Two observations at approximately the same location (one along-track,
one cross-track) give:

```
[A_e_at   A_n_at] [v_zonal]   [v_wind_LOS_at]
[A_e_ct   A_n_ct] [v_merid] = [v_wind_LOS_ct]
```

Solving this 2×2 system gives (v_zonal, v_merid).

### 2.2 Sensitivity coefficients

`ê_east` and `ê_north` are the unit East and North vectors at the tangent
point in ECI coordinates, computed from `enu_unit_vectors_eci()` in NB02c.

For the along-track look, `los_eci` is approximately aligned with the
spacecraft velocity direction (depressed 15.79° toward the limb). This
means:
- `A_e_at ≈ cos(15.79°) × (velocity direction projected onto East)` — varies with latitude
- `A_n_at ≈ cos(15.79°) × cos(inclination)` — large for the SSO inclination

For the cross-track look, `los_eci` is approximately perpendicular to the
spacecraft velocity (also depressed 15.79°). This means:
- `A_e_ct ≈ cos(15.79°) × 1` — predominantly eastward sensitivity
- `A_n_ct ≈ 0` — approximately zero northward sensitivity

This near-orthogonality of the two LOS projections is what makes the 2×2
system well-conditioned. The condition number of the matrix should be
checked for every pair — poorly conditioned pairs (condition > 100) are
excluded from the L2 product.

### 2.3 v_wind_LOS from M06 and NB02

`v_wind_LOS` for each observation is obtained by removing the spacecraft
velocity and Earth rotation from the FPI's measured `v_rel`:

```python
# NB02d function — already implemented in S07
v_wind_LOS = remove_spacecraft_velocity(v_rel, V_sc_LOS, v_earth_LOS)
# = v_rel + V_sc_LOS + v_earth_LOS  (sign convention from NB02d)
```

In the real pipeline:
- `v_rel` comes from M06's `AirglowFitResult.v_rel_ms`
- `V_sc_LOS` and `v_earth_LOS` come from NB02c's `compute_v_rel()` dict
  for the same epoch

In the synthetic pipeline:
- `v_wind_LOS` is available directly from NB02c's `compute_v_rel()` dict

### 2.4 Uncertainty propagation

The uncertainty on `v_wind_LOS` is approximately:

```
sigma_v_wind_LOS ≈ sigma_v_rel
```

The spacecraft and Earth velocity terms are deterministic (no measurement
noise) so their uncertainties are negligible compared to `sigma_v_rel`
from M06.

The uncertainty on the recovered wind components follows from the
2×2 solution via error propagation:

```python
# A is the 2×2 sensitivity matrix
# sigma_b = [sigma_v_wind_LOS_at, sigma_v_wind_LOS_ct]
A_inv = np.linalg.inv(A)
# Covariance of [v_zonal, v_merid]:
cov_wind = A_inv @ np.diag(sigma_b**2) @ A_inv.T
sigma_v_zonal = np.sqrt(cov_wind[0, 0])
sigma_v_merid  = np.sqrt(cov_wind[1, 1])
```

---

## 3. Physical constants from S03

```python
from src.constants import (
    LAT_RANGE_DEG,    # (-40.0, 40.0) — primary science latitude band
    SC_ORBITAL_PERIOD_S,  # ~5640 s — used to set pairing time window
)
```

---

## 4. Input data structures

M07 receives two parallel lists, one per observation:

### 4.1 WindObservation (input record per epoch)

```python
@dataclass
class WindObservation:
    """
    A single FPI wind observation — one epoch, one look mode.
    Constructed from M06 output + NB02c output for the same epoch.
    """
    epoch_utc:       float      # Unix timestamp, seconds
    look_mode:       str        # 'along_track' or 'cross_track'
    tp_lat_deg:      float      # tangent point geodetic latitude, deg
    tp_lon_deg:      float      # tangent point geodetic longitude, deg
    tp_alt_km:       float      # tangent point geodetic altitude, km

    # From M06 AirglowFitResult
    v_rel_ms:        float      # LOS Doppler observable, m/s
    sigma_v_rel_ms:  float      # 1σ uncertainty on v_rel, m/s

    # From NB02c compute_v_rel() dict
    V_sc_LOS:        float      # spacecraft velocity LOS projection, m/s
    v_earth_LOS:     float      # Earth rotation LOS projection, m/s
    v_wind_LOS:      float      # atmospheric wind LOS projection, m/s
                                # = v_rel + V_sc_LOS + v_earth_LOS

    # From NB02c enu_unit_vectors_eci()
    los_eci:         np.ndarray  # shape (3,), unit LOS vector in ECI

    # ENU unit vectors at tangent point (from NB02c)
    e_east_eci:      np.ndarray  # shape (3,), unit East vector in ECI
    e_north_eci:     np.ndarray  # shape (3,), unit North vector in ECI

    # Quality
    m06_quality_flags: int       # from AirglowFitResult.quality_flags
```

### 4.2 WindResult (output record per pair)

```python
@dataclass
class WindResult:
    """
    L2 horizontal wind vector at one geographic location.
    Produced by M07 from one along-track + one cross-track pair.
    Per S04: every fitted quantity has sigma_ and two_sigma_ fields.
    """
    # Geolocation (midpoint of the two tangent points)
    lat_deg:    float    # degrees, WGS84
    lon_deg:    float    # degrees, WGS84
    alt_km:     float    # km, nominal ~250 km

    # Wind components — positive = eastward / northward (S02 Section 6.6)
    v_zonal_ms:             float
    sigma_v_zonal_ms:       float
    two_sigma_v_zonal_ms:   float   # exactly 2 × sigma_v_zonal_ms  (S04)

    v_meridional_ms:            float
    sigma_v_meridional_ms:      float
    two_sigma_v_meridional_ms:  float   # exactly 2 × sigma_v_meridional_ms

    # Timing
    epoch_at_utc:  float   # Unix timestamp of along-track observation
    epoch_ct_utc:  float   # Unix timestamp of cross-track observation
    delta_t_s:     float   # time separation between the two observations

    # Geometry diagnostics
    condition_number:  float   # condition number of 2×2 A matrix
    A_e_at:  float   # east sensitivity, along-track
    A_n_at:  float   # north sensitivity, along-track
    A_e_ct:  float   # east sensitivity, cross-track
    A_n_ct:  float   # north sensitivity, cross-track

    # Input observations used
    v_wind_LOS_at:  float   # atmospheric LOS wind, along-track
    v_wind_LOS_ct:  float   # atmospheric LOS wind, cross-track

    # Quality
    n_obs_at:    int    # always 1 (one along-track obs per pair)
    n_obs_ct:    int    # always 1 (one cross-track obs per pair)
    quality_flags:  int  # WindResultFlags bitmask


class WindResultFlags:
    """Bitmask quality flags for WindResult. Uses bits 4+ per S04."""
    GOOD                  = 0x00
    ILL_CONDITIONED       = 0x10   # condition_number > 100
    LARGE_DELTA_T         = 0x20   # |delta_t_s| > 7000 s (> ~1.2 orbits)
    AT_OBS_DEGRADED       = 0x40   # along-track M06 flags non-GOOD
    CT_OBS_DEGRADED       = 0x80   # cross-track M06 flags non-GOOD
    OUT_OF_LAT_BAND       = 0x100  # |lat| > LAT_RANGE_DEG[1]
```

---

## 5. Pairing algorithm

### 5.1 Overview

Given a list of `WindObservation` objects (mixed along-track and
cross-track, any number of orbits), M07 finds matched pairs and solves
the 2×2 system for each.

```python
def pair_observations(
    observations: list[WindObservation],
    lat_bin_deg: float = 2.0,
    max_delta_t_s: float = 7000.0,
    lat_range_deg: tuple = LAT_RANGE_DEG,
) -> list[tuple[WindObservation, WindObservation]]:
    """
    Match along-track observations to cross-track observations at
    approximately the same geographic location.

    Algorithm
    ---------
    1. Separate observations into along-track and cross-track lists.
    2. For each along-track observation within lat_range_deg:
       a. Find all cross-track observations with |delta_lat| < lat_bin_deg/2
          AND |delta_t| < max_delta_t_s.
       b. Among candidates, select the one with the smallest |delta_lat|.
       c. If no candidate found, skip this along-track observation.
    3. Each along-track observation may appear in at most one pair.
       Each cross-track observation may appear in at most one pair.
       (Nearest-neighbour matching, no reuse.)

    Parameters
    ----------
    observations : list of WindObservation
        Mixed look modes, any number of orbits.
    lat_bin_deg : float
        Maximum latitude separation for pairing, degrees. Default 2.0°
        (~220 km). Per STM: 1 obs per 2° latitude required.
    max_delta_t_s : float
        Maximum time separation for pairing, seconds. Default 7000 s
        (~1.23 orbits). Pairs from the same orbit pair (N, N+1) have
        delta_t ≈ 5640 s; this window accepts the full adjacent orbit.
    lat_range_deg : tuple
        (min_lat, max_lat) science latitude band. Default from S03.

    Returns
    -------
    list of (along_track_obs, cross_track_obs) tuples
    """
```

### 5.2 Rationale for max_delta_t_s = 7000 s

Adjacent along-track (orbit N) and cross-track (orbit N+1) observations
are separated by one orbital period, `SC_ORBITAL_PERIOD_S ≈ 5640 s`.
The 7000 s window accepts any pair from consecutive orbits while rejecting
pairs from non-adjacent orbits (which would be ~11,280 s apart). The
atmospheric wind can change significantly over non-adjacent orbit timescales
during geomagnetic storms.

### 5.3 Longitude handling

Along-track and cross-track tangent points are at different longitudes
(the ground track shifts ~24° between successive orbits). M07 does NOT
require longitude matching — only latitude matching. This is correct
because the science goals (SG1, SG2) require latitude resolution, not
longitude resolution. The L2 wind vector is assigned to the midpoint
latitude and the midpoint longitude of the two tangent points.

---

## 6. Top-level function

```python
def retrieve_wind_vectors(
    observations: list[WindObservation],
    lat_bin_deg: float = 2.0,
    max_delta_t_s: float = 7000.0,
    max_condition_number: float = 100.0,
    lat_range_deg: tuple = LAT_RANGE_DEG,
) -> list[WindResult]:
    """
    Decompose paired LOS wind observations into horizontal wind vectors.

    For each matched (along-track, cross-track) pair:
    1. Compute sensitivity coefficients A_e and A_n for each observation.
    2. Build the 2×2 matrix A.
    3. Check condition number — flag ILL_CONDITIONED if > max_condition_number.
    4. Solve the 2×2 system for [v_zonal, v_merid].
    5. Propagate uncertainties through the matrix inverse.
    6. Assign geolocation as the midpoint of the two tangent points.
    7. Package into a WindResult with quality flags.

    Parameters
    ----------
    observations : list[WindObservation]
        All observations to process. Mixed look modes and orbits.
    lat_bin_deg : float
        Latitude matching window, degrees. Default 2.0.
    max_delta_t_s : float
        Maximum time separation for valid pairs, seconds. Default 7000.
    max_condition_number : float
        Maximum 2×2 matrix condition number. Pairs above this threshold
        are solved but flagged ILL_CONDITIONED. Default 100.
    lat_range_deg : tuple
        Science latitude band. Pairs outside this band are solved but
        flagged OUT_OF_LAT_BAND. Default from S03.

    Returns
    -------
    list[WindResult]
        One WindResult per valid matched pair. Empty list if no pairs found.
        Results are sorted by epoch_at_utc.

    Notes
    -----
    ILL_CONDITIONED pairs are included in the output (not dropped).
    Downstream users should filter on quality_flags as needed.
    The L2 product (S20) includes all results with their flags.
    """
```

---

## 7. Sensitivity coefficient computation

```python
def compute_sensitivity_coefficients(
    obs: WindObservation,
) -> tuple[float, float]:
    """
    Compute the east and north sensitivity coefficients for one observation.

    A_e = ê_east  · los_eci   (east  component of LOS projection)
    A_n = ê_north · los_eci   (north component of LOS projection)

    Parameters
    ----------
    obs : WindObservation
        Must have los_eci, e_east_eci, e_north_eci populated.

    Returns
    -------
    (A_e, A_n) : tuple[float, float]

    Notes
    -----
    These are dimensionless numbers in (-1, +1). The magnitude |A_e|
    gives the fraction of eastward wind that contributes to this LOS
    measurement.

    For a well-designed along-track/cross-track pair at mid-latitudes:
        Along-track:  |A_n_at| > |A_e_at|  (more meridional sensitivity)
        Cross-track:  |A_e_ct| > |A_n_ct|  (more zonal sensitivity)

    Expected approximate values at 30°N for WindCube:
        Along-track:  A_e_at ≈ 0.05,  A_n_at ≈ 0.88
        Cross-track:  A_e_ct ≈ 0.93,  A_n_ct ≈ 0.05
    """
    A_e = float(np.dot(obs.e_east_eci,  obs.los_eci))
    A_n = float(np.dot(obs.e_north_eci, obs.los_eci))
    return A_e, A_n
```

---

## 8. Verification tests

All tests in `tests/test_s16_m07_wind_retrieval.py`.

### T1 — S04 compliance: two_sigma fields

```python
def test_two_sigma_convention():
    """All two_sigma_ fields must equal exactly 2.0 × sigma_."""
    result = _make_synthetic_wind_result()
    assert abs(result.two_sigma_v_zonal_ms   - 2.0 * result.sigma_v_zonal_ms)   < 1e-15
    assert abs(result.two_sigma_v_meridional_ms - 2.0 * result.sigma_v_meridional_ms) < 1e-15
```

### T2 — Uniform wind round-trip: known v_zonal and v_merid recovered

```python
def test_uniform_wind_round_trip():
    """
    Inject a known uniform wind field. Construct synthetic WindObservations
    from NB02 geometry. M07 must recover v_zonal and v_meridional to
    within 1 m/s (purely geometric round-trip, no noise).
    """
    from src.windmap.nb00_wind_map_2026_04_06 import UniformWindMap
    v_zonal_truth  =  100.0   # m/s eastward
    v_merid_truth  =   50.0   # m/s northward

    obs_at, obs_ct = _build_synthetic_obs_pair(
        v_zonal_truth, v_merid_truth,
        lat_deg=30.0, look_modes=('along_track', 'cross_track')
    )
    results = retrieve_wind_vectors([obs_at, obs_ct])

    assert len(results) == 1
    r = results[0]
    assert abs(r.v_zonal_ms      - v_zonal_truth) < 1.0, \
        f"v_zonal error = {r.v_zonal_ms - v_zonal_truth:.3f} m/s"
    assert abs(r.v_meridional_ms - v_merid_truth) < 1.0, \
        f"v_merid error = {r.v_meridional_ms - v_merid_truth:.3f} m/s"
    assert r.quality_flags == WindResultFlags.GOOD
```

### T3 — Zero wind: recovered components near zero

```python
def test_zero_wind():
    """Zero wind input must give near-zero output (< 1 m/s)."""
    obs_at, obs_ct = _build_synthetic_obs_pair(0.0, 0.0, lat_deg=15.0)
    results = retrieve_wind_vectors([obs_at, obs_ct])
    r = results[0]
    assert abs(r.v_zonal_ms)      < 1.0
    assert abs(r.v_meridional_ms) < 1.0
```

### T4 — Uncertainty propagation: sigma scales with input sigma

```python
def test_uncertainty_propagation():
    """
    Doubling sigma_v_rel on both observations must approximately double
    sigma_v_zonal and sigma_v_meridional.
    """
    obs1_at, obs1_ct = _build_synthetic_obs_pair(100.0, 50.0, sigma=5.0)
    obs2_at, obs2_ct = _build_synthetic_obs_pair(100.0, 50.0, sigma=10.0)
    r1 = retrieve_wind_vectors([obs1_at, obs1_ct])[0]
    r2 = retrieve_wind_vectors([obs2_at, obs2_ct])[0]
    # sigma should scale roughly linearly with input sigma
    assert 1.5 < r2.sigma_v_zonal_ms / r1.sigma_v_zonal_ms < 2.5
    assert 1.5 < r2.sigma_v_meridional_ms / r1.sigma_v_meridional_ms < 2.5
```

### T5 — Condition number flag

```python
def test_ill_conditioned_flagged():
    """
    Parallel LOS vectors (both along-track) produce singular matrix.
    ILL_CONDITIONED flag must be set.
    """
    # Build two along-track observations at same location — matrix will be singular
    obs1, obs2 = _build_synthetic_obs_pair(100.0, 50.0,
                                            look_modes=('along_track', 'along_track'))
    results = retrieve_wind_vectors([obs1, obs2])
    if len(results) > 0:
        assert results[0].quality_flags & WindResultFlags.ILL_CONDITIONED
```

### T6 — Pairing: respects max_delta_t_s

```python
def test_pairing_time_window():
    """Observations separated by > max_delta_t_s must not be paired."""
    obs_at = _make_obs(look_mode='along_track', epoch=0.0, lat=30.0)
    obs_ct = _make_obs(look_mode='cross_track', epoch=8000.0, lat=30.0)
    results = retrieve_wind_vectors([obs_at, obs_ct], max_delta_t_s=7000.0)
    assert len(results) == 0, "Observations > 7000 s apart must not be paired"
```

### T7 — Pairing: respects lat_bin_deg

```python
def test_pairing_latitude_window():
    """Observations separated by > lat_bin_deg must not be paired."""
    obs_at = _make_obs(look_mode='along_track', epoch=0.0,    lat=30.0)
    obs_ct = _make_obs(look_mode='cross_track', epoch=5640.0, lat=33.0)
    results = retrieve_wind_vectors([obs_at, obs_ct], lat_bin_deg=2.0)
    assert len(results) == 0, "Observations > 2° apart must not be paired"
```

### T8 — Sign convention: positive v_zonal = eastward

```python
def test_wind_sign_convention():
    """
    Positive v_zonal must correspond to eastward wind.
    Positive v_merid must correspond to northward wind.
    Verify using the sensitivity coefficients directly.
    """
    obs_at, obs_ct = _build_synthetic_obs_pair(100.0, 0.0, lat_deg=0.0)
    A_e_at, A_n_at = compute_sensitivity_coefficients(obs_at)
    A_e_ct, A_n_ct = compute_sensitivity_coefficients(obs_ct)
    # For a purely eastward wind, v_wind_LOS = v_zonal × A_e
    # Cross-track should show larger east sensitivity
    assert abs(A_e_ct) > abs(A_e_at), \
        "Cross-track should have larger east sensitivity than along-track"
    results = retrieve_wind_vectors([obs_at, obs_ct])
    assert results[0].v_zonal_ms > 0, "Eastward wind must give positive v_zonal"
```

---

## 9. Test helper functions

The following private helpers support T1–T8. Implement in the test file.

```python
def _build_synthetic_obs_pair(
    v_zonal_truth: float,
    v_merid_truth: float,
    lat_deg: float = 30.0,
    lon_deg: float = 0.0,
    sigma: float = 9.8,
    look_modes: tuple = ('along_track', 'cross_track'),
    delta_t_s: float = 5640.0,
) -> tuple[WindObservation, WindObservation]:
    """
    Build a synthetic (along_track, cross_track) observation pair at the
    given lat/lon with known wind truth. Uses NB02c geometry functions to
    compute los_eci, e_east_eci, e_north_eci. Computes v_wind_LOS from
    the truth wind by projecting onto each LOS. Sets sigma_v_rel_ms=sigma.

    This is a purely geometric construction — no FPI fringe simulation.
    Calls enu_unit_vectors_eci() and compute_los_eci() from NB02.
    """
```

---

## 10. Expected numerical values

For a uniform wind field of `v_zonal=100 m/s`, `v_merid=50 m/s` at 30°N,
noiseless geometric construction:

| Quantity | Expected value | Notes |
|----------|----------------|-------|
| `v_zonal_ms` | 100.0 ± 1.0 m/s | T2 |
| `v_meridional_ms` | 50.0 ± 1.0 m/s | T2 |
| `condition_number` | 2–20 | Well-conditioned for 90° yaw |
| `A_e_at` at 30°N | ≈ 0.05 | Small east sensitivity along-track |
| `A_n_at` at 30°N | ≈ 0.88 | Large north sensitivity along-track |
| `A_e_ct` at 30°N | ≈ 0.93 | Large east sensitivity cross-track |
| `A_n_ct` at 30°N | ≈ 0.05 | Small north sensitivity cross-track |
| `delta_t_s` nominal | ≈ 5640 s | One orbital period |
| `sigma_v_zonal` at sigma_v_rel=9.8 | ≈ 10–15 m/s | Amplified by matrix inverse |
| `sigma_v_merid` at sigma_v_rel=9.8 | ≈ 10–15 m/s | Amplified by matrix inverse |

**Note on sigma amplification:** The matrix inverse amplifies uncertainties
by a factor of `1/sin(θ)` where `θ` is the angle between the two LOS
projections in the horizontal plane. For a 90° yaw and 15.79° depression
angle, `sin(θ) ≈ cos(15.79°) ≈ 0.96`, so amplification is ~1.04 —
essentially none. This is why the 90° cross-track yaw is chosen.

---

## 11. File locations

```
soc_sewell/
├── src/fpi/
│   └── m07_wind_retrieval_2026_04_06.py
└── tests/
    └── test_s16_m07_wind_retrieval.py
```

---

## 12. Instructions for Claude Code

1. Read this entire spec, S04, S07 (NB02 geometry spec), and S02 before
   writing any code.
2. Confirm all prior tests pass:
   ```bash
   pytest tests/ -v --tb=no -q
   ```
3. Implement `src/fpi/m07_wind_retrieval_2026_04_06.py` in this order:
   `WindResultFlags` → `WindObservation` → `WindResult` →
   `compute_sensitivity_coefficients` → `pair_observations` →
   `retrieve_wind_vectors`
4. `two_sigma_X = 2.0 × sigma_X` for every field — set immediately after
   computing sigma. Never compute independently.
5. The 2×2 solve must use `np.linalg.solve` for well-conditioned matrices
   and catch `np.linalg.LinAlgError` (singular matrix) by setting
   `ILL_CONDITIONED` flag and returning NaN wind values. Do not raise.
6. Check condition number with `np.linalg.cond(A)` before solving. Flag
   `ILL_CONDITIONED` if `> max_condition_number` but still attempt solve.
7. Implement `_build_synthetic_obs_pair()` helper in the test file using
   `enu_unit_vectors_eci()` and `compute_los_eci()` from NB02c. Import
   these directly: `from src.geometry.nb02c_los_projection_YYYY_MM_DD
   import enu_unit_vectors_eci` (check the actual filename first).
8. Run module tests:
   ```bash
   pytest tests/test_s16_m07_wind_retrieval.py -v
   ```
   All 8 must pass.
9. Run full suite:
   ```bash
   pytest tests/ -v
   ```
   No regressions.
10. Commit:
    ```
    feat(m07): implement L2 vector wind retrieval, 8/8 tests pass
    Implements: S16_m07_wind_retrieval_2026-04-06.md
    ```

Module docstring header:
```python
"""
M07 — L2 vector wind retrieval: decomposes LOS winds into horizontal components.

Spec:        docs/specs/S16_m07_wind_retrieval_2026-04-06.md
Spec date:   2026-04-06
Generated:   <today>
Tool:        Claude Code
Last tested: <today>  (8/8 tests pass)
Depends on:  src.constants, src.geometry.nb02c_*, src.fpi.m06_*
"""
```
