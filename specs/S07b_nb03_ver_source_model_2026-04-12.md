# S07b — NB03 Volume Emission Rate Source Model & Signal Budget

**Spec ID:** S07b
**Spec file:** `docs/specs/S07b_nb03_ver_source_model_2026-04-12.md`
**Project:** WindCube FPI Pipeline
**Institution:** NCAR / High Altitude Observatory (HAO)
**Status:** Specification — ready for implementation in VS Code
**Tier:** 1 (Geometry & Signal Chain)
**Depends on:** S01, S02, S03 (constants), S06 (NB01 orbital geometry), S07 (NB02 look direction)
**Used by:**
  - S08 (INT01) — integration test of geometry + signal chain
  - S11 (M04) — `I_line` parameter for `synthesise_airglow_image()`
  - S17 (INT02) — physically grounded SNR for Monte Carlo noise trials
**References:**
  - signal_calculator.py (A. Ridley, HAO) — LOS geometry and integration architecture
  - Meneses et al. (2008) Geofísica Internacional 47(3), 161–166 — nightglow VER profiles
  - Solomon & Abreu (1989) JGR 94(A6), 6817–6824 — dayglow VER profiles
  - Wang et al. (2020) European Journal of Remote Sensing 53(1), 145–155 — satellite FPI signal budget
  - Teledyne e2v CCD97 datasheet — QE at 630 nm
**Last updated:** 2026-04-12

> **Design note:** NB03 is a Tier 1 module that is deliberately FPI-agnostic. It
> has no knowledge of the etalon, Airy function, or fringe pattern. Its sole
> responsibility is to compute a physically grounded photon count at the detector
> from a line-of-sight geometry and a tabulated VER profile. The dimensionless
> `I_line` it produces is the bridge to M04's delta-function source model.
>
> All new instrument constants introduced here (aperture, throughput, QE, EM gain,
> integration time) are defined in `windcube/constants.py` (S03). This is the
> single authoritative location. Do not hardcode them in NB03.

---

## 1. Purpose

NB03 replaces the arbitrary placeholder `I_line = 1.0` in M04
(`synthesise_airglow_image`) with a value computed from first principles:
the OI 630.0 nm volume emission rate (VER) profile, the spacecraft
line-of-sight (LOS) geometry, the detector aperture, and the WindCube
instrument constants.

**What NB03 provides:**

1. A tabulated, spline-interpolated OI 630.0 nm VER profile as a function
   of altitude, in two modes: `'nightglow'` (primary science case) and
   `'dayglow'` (reference only — WindCube is a nighttime mission).
2. A LOS geometry solver that accepts tangent height and orbital altitude
   and returns pitch angle and integration limits.
3. A LOS column brightness integrator in physical units.
4. A photon flux calculator per detector pixel per second.
5. A signal budget function that converts photons/s to the dimensionless
   `I_line` expected by M04, applying all detector constants from S03.

**What NB03 does not do:**

- NB03 does not model the FPI instrument transfer function.
- NB03 does not model shot noise or detector read noise — those are M04's
  responsibility via the `snr` parameter.
- NB03 does not compute wind or temperature — it is purely a photometric
  forward model.

---

## 2. Physical background

### 2.1 The OI 630.0 nm emission mechanism

The OI 630.0 nm (red line) emission arises from the forbidden transition
O(¹D → ³P). In the nighttime F region (180–350 km altitude), the dominant
production mechanism is dissociative recombination of O₂⁺:

```
O₂⁺ + e⁻  →  O(¹D) + O(³P)
O(¹D)      →  O(³P) + hν (630.0 nm)
```

The volume emission rate V(h) [photons m⁻³ s⁻¹] describes the number of
630.0 nm photons emitted per unit volume per second at altitude h. It is
isotropic, so the VER per steradian is V(h) / 4π.

The nightglow VER profile peaks near 220–260 km altitude at approximately
10–45 photons cm⁻³ s⁻¹ depending on ionospheric conditions
(Meneses et al. 2008; Link & Cogger 1988). For dayglow, the peak is
roughly 100× stronger (Solomon & Abreu 1989), but WindCube observes
in darkness, so the nightglow profile is the operative case.

### 2.2 Line-of-sight geometry

WindCube orbits at altitude `h_orbit` (nominally 500 km) and observes the
thermosphere at tangent height `h_tangent` (nominally 250 km) by pointing
its FPI slightly below horizontal. The pitch angle `pa` (angle below
horizontal at the spacecraft) is:

```
pa = arccos( (R_earth + h_tangent) / (R_earth + h_orbit) )
```

The distance from the spacecraft to any point along the LOS at distance
`d` [m] has altitude:

```
h(d) = sqrt( R_orbit² + d² - 2·R_orbit·d·sin(pa) ) - R_earth
```

where `R_orbit = R_earth + h_orbit`. The tangent point is at:

```
d_tangent = R_orbit · sin(pa)
```

Integration limits `(d_near, d_far)` are the two distances where the LOS
crosses the boundary altitude `h_max` (default 490 km), found by solving
the altitude equation as a quadratic in `d`.

### 2.3 Column brightness

The column brightness B [photons m⁻² s⁻¹ sr⁻¹] along a single LOS is:

```
B = ∫_{d_near}^{d_far}  V(h(d)) / (4π)  dd
```

where the 4π converts isotropic VER to per-steradian radiance.

### 2.4 Photon flux at detector

The photon flux F [photons s⁻¹] collected by a single binned pixel is:

```
F = B × A_aperture × Ω_pixel
```

where:
- `A_aperture` = telescope entrance pupil area [m²]
- `Ω_pixel`    = solid angle subtended by one binned pixel [sr]
                = α² where α = ALPHA_RAD_PER_PX (from S03/S09)

### 2.5 Detector signal chain

Converting photons/s to CCD counts, applying all WindCube detector
constants from S03:

```
counts = F × t_exp × QE × T_opt × G_em
```

where:
- `t_exp`  = integration time [s]           — `INTEGRATION_TIME_S`
- `QE`     = CCD quantum efficiency at 630 nm  — `CCD97_QE_630NM`
- `T_opt`  = total optical throughput         — `OPTICAL_THROUGHPUT`
- `G_em`   = EMCCD electron multiplication gain — `EM_GAIN`

### 2.6 Dimensionless I_line for M04

M04's `synthesise_airglow_image()` uses `I_line` as a dimensionless scaling
factor relative to `InstrumentParams.I0` (default 1000.0 counts). The
conversion is:

```
I_line = counts / I0
```

This is the number NB03 delivers to the M04 call. An `I_line` of 1.0
corresponds to exactly `I0 = 1000` counts at peak. Typical nightglow
values are expected in the range 0.01–5.0 depending on ionospheric
conditions and exposure time.

---

## 3. New constants for S03 / `windcube/constants.py`

The following constants must be added to `windcube/constants.py` (S03)
**before** implementing NB03. NB03 imports them from there; they must
never be hardcoded in the module itself.

```python
# ---------------------------------------------------------------------------
# WindCube instrument / detector constants  (S07b / NB03 addendum)
# ---------------------------------------------------------------------------

# Entrance pupil area.  Placeholder value — update when aperture is
# confirmed from the as-built optical design.
APERTURE_M2 = 0.06 * 0.03          # m²  (60 mm × 30 mm rectangular pupil)

# Total optical throughput: filters, mirrors, beamsplitters, etalon
# transmission losses.  From WindCube.py (A. Ridley).  Treat as
# provisional until measured on FlatSat.
OPTICAL_THROUGHPUT = 0.85           # dimensionless

# CCD97 quantum efficiency at 630 nm.  From Teledyne e2v CCD97 datasheet.
# Treat as provisional until device-level characterisation.
CCD97_QE_630NM = 0.90               # electrons / photon

# EMCCD electron multiplication gain.  Set to 1.0 (conventional CCD mode)
# until EM gain calibration is available.  Update when flight gain
# setting is determined.
EM_GAIN = 1.0                       # dimensionless

# Nominal science frame integration time.
INTEGRATION_TIME_S = 10.0           # seconds

# Plate scale (2×2 binned).  Authoritative Tolansky value from S09/S13.
# Reproduced here so NB03 can compute pixel solid angle without
# importing from M01 (which would create a circular Tier dependency).
ALPHA_RAD_PER_PX = 1.6071e-4        # rad / binned pixel

# Orbital and observation geometry defaults
ORBIT_ALTITUDE_M    = 500_000.0     # m, nominal WindCube orbit
TANGENT_ALTITUDE_M  = 250_000.0     # m, nominal thermospheric tangent height
VER_LAYER_TOP_M     = 490_000.0     # m, upper bound of emission layer for LOS integration
```

**Important notes:**
- `ALPHA_RAD_PER_PX` is intentionally duplicated from S09/M01. The canonical
  value lives in M01 / `InstrumentParams`; the S03 copy exists solely so
  NB03 (Tier 1) can compute `Ω_pixel` without importing from Tier 2.
  If the Tolansky result is ever revised, **both** locations must be updated.
- All five instrument constants (`APERTURE_M2`, `OPTICAL_THROUGHPUT`,
  `CCD97_QE_630NM`, `EM_GAIN`, `INTEGRATION_TIME_S`) are explicitly marked
  provisional in comments. The spec version or date should be updated
  whenever any of these values is revised.

---

## 4. Tabulated VER profiles

### 4.1 Nightglow profile (authoritative default)

Source: digitised from Meneses et al. (2008) Fig. 3, consistent with
Link & Cogger (1988) and with `NightglowEmissionData` in
`signal_calculator.py` (A. Ridley, HAO). This is the operative profile
for WindCube science simulations.

Units as stored: [VER in cm⁻³ s⁻¹, altitude in km].
Conversion to SI at load time: `VER_m3_s1_sr1 = VER_cm3_s1 × 1e6 / (4π)`

```python
NIGHTGLOW_VER_TABLE = np.array([
    # [VER cm⁻³ s⁻¹,  altitude km]
    [0.000,  100.0],
    [0.030,  150.0],
    [0.073,  191.98],
    [0.145,  196.08],
    [0.363,  199.86],
    [0.653,  204.59],
    [1.016,  209.32],
    [1.742,  213.42],
    [2.612,  216.57],
    [3.483,  219.40],
    [4.354,  221.93],
    [5.225,  225.08],
    [6.168,  226.65],
    [7.039,  228.23],
    [7.837,  230.12],
    [8.708,  231.70],
    [9.579,  233.27],
    [10.522, 234.85],
    [11.248, 238.63],
    [11.393, 242.42],
    [10.958, 247.15],
    [10.740, 251.56],
    [11.321, 255.34],
    [11.611, 259.75],
    [11.248, 264.80],
    [11.393, 268.58],
    [11.176, 272.36],
    [10.522, 276.46],
    [9.652,  279.93],
    [8.781,  282.14],
    [7.910,  284.34],
    [7.039,  287.18],
    [6.096,  290.96],
    [5.370,  295.06],
    [4.499,  298.84],
    [3.774,  302.63],
    [3.266,  307.04],
    [2.830,  311.45],
    [2.322,  315.55],
    [1.887,  319.96],
    [1.524,  324.06],
    [1.306,  328.16],
    [1.016,  332.57],
    [0.726,  336.99],
    [0.653,  339.51],
    [0.500,  345.0 ],
    [0.300,  350.0 ],
    [0.200,  360.0 ],
    [0.100,  380.0 ],
    [0.050,  400.0 ],
    [0.000,  500.0 ],
])  # shape (51, 2)
```

Peak VER ≈ 11.6 cm⁻³ s⁻¹ at ~260 km. The profile is zero outside
[100, 500] km; the spline must be clamped to zero outside this range.

### 4.2 Dayglow profile (reference only)

Source: `DayglowEmissionModel` in `signal_calculator.py` (A. Ridley, HAO),
consistent with Solomon & Abreu (1989). Included for completeness and
comparison. WindCube does not observe in dayglow conditions.

Not reproduced here in full — implementation should copy the table from
`signal_calculator.py` verbatim, with the same conversion factor.
Peak VER ≈ 209 cm⁻³ s⁻¹ at ~215 km. Roughly 18× brighter than nightglow
at peak, consistent with the ~1/100 ratio cited in Wang et al. (2020) for
the integrated LOS signal.

---

## 5. Function signatures

Implement in this strict order. Each function depends on the previous.

### 5.1 `build_ver_profile`

```python
def build_ver_profile(
    mode: str = 'nightglow',   # 'nightglow' or 'dayglow'
) -> callable:
    """
    Build a callable VER(h) spline interpolator for OI 630.0 nm.

    Parameters
    ----------
    mode : str
        'nightglow' (default, operative for WindCube science) or
        'dayglow' (reference only).

    Returns
    -------
    ver_func : callable
        CubicSpline interpolator.  Input: altitude in metres.
        Output: VER in m⁻³ s⁻¹ sr⁻¹ (isotropic, per steradian).
        Returns 0.0 for altitudes outside the tabulated range —
        enforced via CubicSpline extrapolate=False + np.clip.

    Notes
    -----
    Conversion from table units (cm⁻³ s⁻¹) to SI per steradian:
        VER_m3_s1_sr1 = VER_cm3_s1 * 1e6 / (4 * pi)
    The 1e6 = (100)³ converts cm⁻³ to m⁻³.
    The 4π distributes isotropic emission into steradians.
    """
```

### 5.2 `compute_los_geometry`

```python
def compute_los_geometry(
    h_tangent_m: float,                       # tangent altitude, m
    h_orbit_m:   float = ORBIT_ALTITUDE_M,    # spacecraft altitude, m
    h_max_m:     float = VER_LAYER_TOP_M,     # upper emission boundary, m
    R_earth_m:   float = None,                # defaults to astropy R_earth
) -> dict:
    """
    Compute line-of-sight geometry for a limb-viewing spacecraft.

    Parameters
    ----------
    h_tangent_m : tangent point altitude, metres
    h_orbit_m   : spacecraft orbital altitude above Earth's surface, metres
    h_max_m     : altitude of upper boundary of emission layer, metres
    R_earth_m   : Earth radius in metres (default: astropy Constant)

    Returns
    -------
    dict with keys:
        'pitch_angle_rad'  : float  — angle below horizontal at spacecraft
        'R_orbit_m'        : float  — geocentric orbital radius, m
        'd_tangent_m'      : float  — LOS distance to tangent point, m
        'd_near_m'         : float  — LOS distance to near emission boundary, m
        'd_far_m'          : float  — LOS distance to far emission boundary, m

    Raises
    ------
    ValueError if h_tangent_m >= h_orbit_m or h_tangent_m < 0.
    ValueError if the LOS never intersects the emission layer (h_max too low).
    """
```

### 5.3 `altitude_along_los`

```python
def altitude_along_los(
    d:         np.ndarray,   # LOS distances, metres, shape (N,)
    pa:        float,        # pitch angle, radians
    R_orbit_m: float,        # geocentric orbital radius, metres
    R_earth_m: float,        # Earth radius, metres
) -> np.ndarray:
    """
    Altitude above Earth's surface at LOS distance d.

    h(d) = sqrt( R_orbit² + d² - 2·R_orbit·d·sin(pa) ) - R_earth

    Clamped to [0, 1e6] m to prevent negative altitudes at very
    large d values from spline extrapolation.

    Parameters
    ----------
    d         : LOS distance array, metres
    pa        : pitch angle below horizontal, radians
    R_orbit_m : geocentric orbital radius of spacecraft, metres
    R_earth_m : Earth radius, metres

    Returns
    -------
    h : np.ndarray, altitude in metres, same shape as d
    """
```

### 5.4 `integrate_los_emission`

```python
def integrate_los_emission(
    ver_func:    callable,   # from build_ver_profile()
    geometry:    dict,       # from compute_los_geometry()
    n_quad:      int = 200,  # quadrature points (fixed_quad)
) -> float:
    """
    Integrate VER along the LOS to obtain column brightness.

    B = ∫_{d_near}^{d_far}  ver_func(h(d))  dd

    where ver_func already includes the 1/(4π) sr⁻¹ factor.

    Parameters
    ----------
    ver_func : callable, VER(altitude_m) → m⁻³ s⁻¹ sr⁻¹
    geometry : dict from compute_los_geometry()
    n_quad   : number of quadrature points for scipy.integrate.fixed_quad

    Returns
    -------
    B : float, column brightness in photons m⁻² s⁻¹ sr⁻¹
    """
```

### 5.5 `signal_photons_per_pixel`

```python
def signal_photons_per_pixel(
    column_brightness_m2_s1_sr1: float,   # from integrate_los_emission()
    aperture_m2:      float = None,       # defaults to APERTURE_M2 from S03
    alpha_rad_per_px: float = None,       # defaults to ALPHA_RAD_PER_PX from S03
) -> float:
    """
    Photon flux collected by a single binned detector pixel per second.

    F = B × A_aperture × Ω_pixel
    Ω_pixel = alpha²   (solid angle of one binned pixel, sr)

    Parameters
    ----------
    column_brightness_m2_s1_sr1 : column brightness, photons m⁻² s⁻¹ sr⁻¹
    aperture_m2      : entrance pupil area, m² (default: APERTURE_M2)
    alpha_rad_per_px : plate scale, rad/px (default: ALPHA_RAD_PER_PX)

    Returns
    -------
    photons_per_s : float, photons s⁻¹ per binned pixel
    """
```

### 5.6 `signal_to_I_line`

```python
def signal_to_I_line(
    photons_per_s: float,           # from signal_photons_per_pixel()
    t_exp:   float = None,          # s,  defaults to INTEGRATION_TIME_S
    qe:      float = None,          # e⁻/photon, defaults to CCD97_QE_630NM
    t_opt:   float = None,          # dimensionless, defaults to OPTICAL_THROUGHPUT
    g_em:    float = None,          # dimensionless, defaults to EM_GAIN
    I0:      float = 1000.0,        # counts, InstrumentParams.I0 default
) -> float:
    """
    Convert photons/s to the dimensionless I_line expected by M04.

    counts  = photons_per_s × t_exp × qe × t_opt × g_em
    I_line  = counts / I0

    All detector constants default to the authoritative S03 values.
    Callers may override any parameter to perform sensitivity studies.

    Parameters
    ----------
    photons_per_s : photon flux per binned pixel, photons s⁻¹
    t_exp   : integration time, seconds
    qe      : quantum efficiency at 630 nm, electrons/photon
    t_opt   : total optical throughput, dimensionless
    g_em    : EM gain, dimensionless
    I0      : M04 normalisation constant (InstrumentParams.I0), counts

    Returns
    -------
    I_line : float, dimensionless signal scale factor for M04
    """
```

### 5.7 `compute_signal_budget` (convenience wrapper)

```python
def compute_signal_budget(
    h_tangent_m: float = TANGENT_ALTITUDE_M,
    h_orbit_m:   float = ORBIT_ALTITUDE_M,
    mode:        str   = 'nightglow',
    verbose:     bool  = False,
) -> dict:
    """
    End-to-end signal budget: VER profile → I_line in one call.

    Calls build_ver_profile → compute_los_geometry →
    integrate_los_emission → signal_photons_per_pixel → signal_to_I_line.

    Parameters
    ----------
    h_tangent_m : tangent altitude, metres
    h_orbit_m   : spacecraft orbital altitude, metres
    mode        : 'nightglow' or 'dayglow'
    verbose     : if True, print each intermediate quantity with units

    Returns
    -------
    dict with keys:
        'mode'                  : str
        'h_tangent_m'           : float
        'h_orbit_m'             : float
        'pitch_angle_deg'       : float
        'column_brightness'     : float, photons m⁻² s⁻¹ sr⁻¹
        'photons_per_pixel_s'   : float, photons s⁻¹
        'counts_per_pixel'      : float, after full detector chain
        'I_line'                : float, dimensionless, for M04
        'geometry'              : dict, full output of compute_los_geometry()
    """
```

---

## 6. Imports

```python
import numpy as np
from scipy.integrate import fixed_quad
from scipy.interpolate import CubicSpline
import astropy.constants as C

from windcube.constants import (
    APERTURE_M2,
    OPTICAL_THROUGHPUT,
    CCD97_QE_630NM,
    EM_GAIN,
    INTEGRATION_TIME_S,
    ALPHA_RAD_PER_PX,
    ORBIT_ALTITUDE_M,
    TANGENT_ALTITUDE_M,
    VER_LAYER_TOP_M,
)
```

`astropy.constants` is used only for `C.R_earth.value`. This is already
a pipeline dependency (used in NB01/NB02). No new package installs required.

---

## 7. Verification tests

All 8 tests in `tests/test_nb03_ver_source_model_2026-04-12.py`.

### T1 — VER profile positive and peaks in correct altitude range

```python
def test_ver_profile_nightglow_peak():
    """
    Nightglow VER must peak between 200 and 300 km.
    Peak value must be in physically plausible range 1–100 cm⁻³ s⁻¹.
    Must be zero at 100 km and 500 km (table boundaries).
    """
    from fpi.nb03_ver_source_model import build_ver_profile
    ver = build_ver_profile('nightglow')
    h = np.linspace(100e3, 500e3, 2000)
    v = ver(h)
    # Convert back to cm⁻³ s⁻¹ for physical bounds check
    v_cm3 = v * 4 * np.pi / 1e6
    peak_h_km = h[np.argmax(v_cm3)] / 1000.0
    peak_v    = np.max(v_cm3)
    assert 200.0 < peak_h_km < 300.0, \
        f"Nightglow VER peak at {peak_h_km:.1f} km, expected 200–300 km"
    assert 1.0 < peak_v < 100.0, \
        f"Peak VER {peak_v:.2f} cm⁻³ s⁻¹ outside [1, 100]"
    assert ver(100e3) == 0.0 or abs(ver(100e3)) < 1e-10, \
        "VER must be zero at lower boundary (100 km)"
    assert ver(500e3) == 0.0 or abs(ver(500e3)) < 1e-10, \
        "VER must be zero at upper boundary (500 km)"
```

### T2 — LOS geometry: pitch angle correct for nominal parameters

```python
def test_los_geometry_nominal():
    """
    For h_orbit=500 km, h_tangent=250 km:
      pitch_angle = arccos((R_earth+250e3)/(R_earth+500e3))
      d_tangent   = R_orbit * sin(pitch_angle)
    Values must be physically consistent to < 1 m.
    """
    import astropy.constants as C
    from fpi.nb03_ver_source_model import compute_los_geometry
    geo = compute_los_geometry(h_tangent_m=250e3, h_orbit_m=500e3)
    R_e = C.R_earth.value
    R_orb = R_e + 500e3
    expected_pa = np.arccos((R_e + 250e3) / R_orb)
    expected_dt = R_orb * np.sin(expected_pa)
    assert abs(geo['pitch_angle_rad'] - expected_pa) < 1e-10, \
        f"Pitch angle {np.degrees(geo['pitch_angle_rad']):.4f} deg, " \
        f"expected {np.degrees(expected_pa):.4f} deg"
    assert abs(geo['d_tangent_m'] - expected_dt) < 1.0, \
        f"d_tangent {geo['d_tangent_m']:.1f} m, expected {expected_dt:.1f} m"
    assert geo['d_near_m'] < geo['d_tangent_m'] < geo['d_far_m'], \
        "d_near < d_tangent < d_far must hold"
```

### T3 — Altitude along LOS is minimum at tangent point

```python
def test_altitude_along_los_minimum_at_tangent():
    """
    h(d) must be minimum at d = d_tangent and monotonically decrease
    then increase on either side.
    """
    from fpi.nb03_ver_source_model import compute_los_geometry, altitude_along_los
    import astropy.constants as C
    geo = compute_los_geometry(250e3, 500e3)
    R_orb = C.R_earth.value + 500e3
    R_e   = C.R_earth.value
    d = np.linspace(geo['d_near_m'], geo['d_far_m'], 500)
    h = altitude_along_los(d, geo['pitch_angle_rad'], R_orb, R_e)
    idx_min = np.argmin(h)
    # Tangent point should be near the middle of the array
    assert 100 < idx_min < 400, \
        f"Altitude minimum at index {idx_min}, expected near center"
    # Altitude at minimum should equal h_tangent to within 1 km
    assert abs(h[idx_min] - 250e3) < 1000.0, \
        f"Min altitude {h[idx_min]/1e3:.2f} km, expected 250.00 km"
```

### T4 — Column brightness is positive and physically plausible

```python
def test_column_brightness_physical():
    """
    Nightglow column brightness must be in the range 1e9–1e13
    photons m⁻² s⁻¹ sr⁻¹ for the nominal geometry.
    (Equivalent to ~10–10000 Rayleigh in 4π sr, per Wang et al. 2020.)
    """
    from fpi.nb03_ver_source_model import (
        build_ver_profile, compute_los_geometry, integrate_los_emission
    )
    ver = build_ver_profile('nightglow')
    geo = compute_los_geometry(250e3, 500e3)
    B = integrate_los_emission(ver, geo)
    assert B > 0, "Column brightness must be positive"
    assert 1e9 < B < 1e13, \
        f"Column brightness {B:.3e} ph m⁻² s⁻¹ sr⁻¹ outside [1e9, 1e13]"
```

### T5 — Photon flux per pixel is positive and sub-saturation

```python
def test_photon_flux_per_pixel():
    """
    Photon flux must be positive.
    For nightglow, expect < 10000 photons/s/pixel (CCD97 full well
    ~80000 e⁻; at QE=0.9 and 10s, saturation would require >>8000
    photons/s/pixel).
    """
    from fpi.nb03_ver_source_model import (
        build_ver_profile, compute_los_geometry,
        integrate_los_emission, signal_photons_per_pixel
    )
    ver = build_ver_profile('nightglow')
    geo = compute_los_geometry(250e3, 500e3)
    B   = integrate_los_emission(ver, geo)
    F   = signal_photons_per_pixel(B)
    assert F > 0, "Photon flux must be positive"
    assert F < 10000, f"Photon flux {F:.1f} ph/s/px implausibly high"
```

### T6 — I_line is in M04-compatible range

```python
def test_I_line_in_m04_range():
    """
    I_line must be positive.
    For nightglow with nominal parameters, expected range 0.001–10.
    I_line >> 10 would saturate the simulated image;
    I_line << 0.001 would make the signal undetectable.
    """
    from fpi.nb03_ver_source_model import compute_signal_budget
    result = compute_signal_budget(mode='nightglow')
    I_line = result['I_line']
    assert I_line > 0, "I_line must be positive"
    assert 0.001 < I_line < 10.0, \
        f"I_line = {I_line:.4f} outside expected range [0.001, 10]"
```

### T7 — Dayglow is brighter than nightglow

```python
def test_dayglow_brighter_than_nightglow():
    """
    Dayglow I_line must be > nightglow I_line.
    Wang et al. (2020): dayglow VER ~100× nightglow.
    Integrated brightness ratio need not be exactly 100× but must be > 5×.
    """
    from fpi.nb03_ver_source_model import compute_signal_budget
    day   = compute_signal_budget(mode='dayglow')
    night = compute_signal_budget(mode='nightglow')
    assert day['I_line'] > night['I_line'], \
        "Dayglow must be brighter than nightglow"
    ratio = day['I_line'] / night['I_line']
    assert ratio > 5.0, \
        f"Day/night I_line ratio {ratio:.1f} < 5, inconsistent with literature"
```

### T8 — signal_to_I_line round-trip

```python
def test_signal_to_I_line_scaling():
    """
    I_line must scale linearly with each detector constant:
      - doubling t_exp doubles I_line
      - doubling QE doubles I_line
      - doubling aperture doubles column brightness (tested via
        signal_photons_per_pixel with doubled aperture)
    """
    from fpi.nb03_ver_source_model import signal_to_I_line
    base = signal_to_I_line(100.0)
    double_texp = signal_to_I_line(100.0, t_exp=2*10.0)
    double_qe   = signal_to_I_line(100.0, qe=2*0.90)
    assert abs(double_texp / base - 2.0) < 1e-10, \
        "I_line must double when t_exp doubles"
    assert abs(double_qe / base - 2.0) < 1e-10, \
        "I_line must double when QE doubles"
```

---

## 8. Expected numerical values

For nominal parameters (nightglow, h_tangent=250 km, h_orbit=500 km,
all S03 defaults):

| Quantity | Expected range | Derivation / check |
|---|---|---|
| Pitch angle | ~21–22° | arccos(6621/6871) |
| d_tangent | ~2400–2500 km | R_orbit × sin(pa) |
| Column brightness B | 1e10–1e12 ph m⁻² s⁻¹ sr⁻¹ | LOS integral |
| Photons/pixel/s F | 1–500 | B × A × Ω |
| counts/pixel | 10–5000 | F × 10 s × 0.9 × 0.85 × 1.0 |
| I_line | 0.01–5.0 | counts / 1000 |

These are order-of-magnitude bounds, not tight tolerances. The VER profile
is a representative quiet-night model, not a specific event prediction.

---

## 9. Connection to M04

After NB03 is implemented, the recommended call pattern in simulation
scripts and integration tests is:

```python
from fpi.nb03_ver_source_model import compute_signal_budget
from fpi.m04_airglow_synthesis import synthesise_airglow_image
from fpi.m01_airy_forward_model import InstrumentParams

budget = compute_signal_budget(
    h_tangent_m=250e3,
    h_orbit_m=500e3,
    mode='nightglow',
    verbose=True,
)

params = InstrumentParams()
result = synthesise_airglow_image(
    v_rel_ms=100.0,
    params=params,
    I_line=budget['I_line'],      # ← physically grounded, replaces 1.0
    snr=None,                     # ← set to None to use physical noise (future)
)
```

Note: M04's `snr` parameter remains active for now — physically grounded
Poisson noise is a future enhancement (post S07b). NB03 provides the
correct `I_line`; the noise model upgrade is a separate spec item.

---

## 10. File locations in repository

```
soc_sewell/
├── windcube/
│   └── constants.py          ← add 9 new constants from Section 3
├── fpi/
│   ├── __init__.py
│   └── nb03_ver_source_model_2026-04-12.py   ← this module
├── tests/
│   └── test_nb03_ver_source_model_2026-04-12.py
└── docs/specs/
    └── S07b_nb03_ver_source_model_2026-04-12.md   ← this file
```

---

## 11. Instructions for Claude Code

1. Read this entire spec AND S03 (`windcube/constants.py`) before writing
   any code.
2. Open `windcube/constants.py` and add the 9 new constants from Section 3,
   with the comments exactly as written. Commit this change first, before
   touching any `fpi/` code.
3. Run `pytest tests/ -v` to confirm all existing tests still pass after
   the constants update.
4. Implement `fpi/nb03_ver_source_model_2026-04-12.py` with functions
   in this strict order:
   `build_ver_profile` → `altitude_along_los` → `compute_los_geometry`
   → `integrate_los_emission` → `signal_photons_per_pixel`
   → `signal_to_I_line` → `compute_signal_budget`
5. The nightglow VER table (Section 4.1) must be defined as a module-level
   constant `NIGHTGLOW_VER_TABLE`. The dayglow table is copied verbatim
   from `signal_calculator.py`. Both are converted to SI/sr units inside
   `build_ver_profile()`, not at module level.
6. `build_ver_profile()` must clamp output to zero outside the table
   altitude range. Use `CubicSpline(..., extrapolate=False)` and then
   `np.nan_to_num(result, nan=0.0)` to handle out-of-range queries.
7. `compute_los_geometry()` must raise `ValueError` for
   `h_tangent_m >= h_orbit_m`. It must use `astropy.constants.R_earth.value`
   for Earth radius — do not hardcode 6371000.
8. `integrate_los_emission()` must use `scipy.integrate.fixed_quad` with
   `n=200` quadrature points, consistent with `signal_calculator.py`.
9. All five instrument constants in `signal_to_I_line()` must default to
   `None` and resolve to the S03 values inside the function body, not in
   the function signature. This ensures that if S03 values are later
   updated, the defaults automatically follow.
10. Write all 8 tests.
11. Run: `pytest tests/test_nb03_ver_source_model_2026-04-12.py -v`
    All 8 must pass.
12. Run full suite: `pytest tests/ -v` — all existing tests still pass.
13. Commit in two steps:
    `feat(constants): add NB03 instrument constants (aperture, QE, gain, throughput)`
    `feat(nb03): implement VER source model and signal budget, 8/8 tests pass`

Module docstring header:
```python
"""
Module:      nb03_ver_source_model_2026-04-12.py
Spec:        docs/specs/S07b_nb03_ver_source_model_2026-04-12.md
Author:      Claude Code
Generated:   <today>
Last tested: <today>  (8/8 tests pass)
Project:     WindCube FPI Pipeline — NCAR/HAO
Repo:        soc_sewell

Computes the OI 630.0 nm photon signal at the WindCube detector from
a tabulated volume emission rate profile integrated along the spacecraft
line of sight. Produces I_line for M04 synthesise_airglow_image().

Instrument constants (aperture, QE, throughput, EM gain, t_exp) are
imported from windcube/constants.py. Update them there, not here.
"""
```
