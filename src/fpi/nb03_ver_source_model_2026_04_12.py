"""
Module:      nb03_ver_source_model_2026-04-12.py
Spec:        docs/specs/S07b_nb03_ver_source_model_2026-04-12.md
Author:      Claude Code
Generated:   2026-04-12
Last tested: 2026-04-12  (8/8 tests pass)
Project:     WindCube FPI Pipeline — NCAR/HAO
Repo:        soc_sewell

Computes the OI 630.0 nm photon signal at the WindCube detector from
a tabulated volume emission rate profile integrated along the spacecraft
line of sight. Produces I_line for M04 synthesise_airglow_image().

Instrument constants (aperture, QE, throughput, EM gain, t_exp) are
imported from windcube/constants.py. Update them there, not here.
"""

import numpy as np
from scipy.integrate import fixed_quad
from scipy.interpolate import CubicSpline

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


# ---------------------------------------------------------------------------
# Module-level VER tables
# Units as stored: [VER cm⁻³ s⁻¹, altitude km]
# Conversion to SI per steradian happens inside build_ver_profile().
# ---------------------------------------------------------------------------

# Nightglow VER profile (ascending altitude order)
# Source: digitised from Meneses et al. (2008) Fig. 3 / NightglowEmissionData
# in signal_calculator.py (A. Ridley, HAO)
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

# Dayglow VER profile (descending altitude order, as stored in
# signal_calculator.py DayglowEmissionModel — A. Ridley, HAO)
# Source: Solomon & Abreu (1989) JGR 94(A6), 6817–6824.
# Peak VER ≈ 209 cm⁻³ s⁻¹ at ~215 km; roughly 18× brighter than nightglow.
# NOTE: This table is stored descending; build_ver_profile() flips it.
DAYGLOW_VER_TABLE = np.array([
    # [VER cm⁻³ s⁻¹,  altitude km]   (descending altitude)
    [0.000,  500.0],
    [0.050,  400.0],
    [0.100,  380.0],
    [0.300,  360.0],
    [0.800,  350.0],
    [1.500,  340.0],
    [3.000,  330.0],
    [5.000,  320.0],
    [9.000,  310.0],
    [14.00,  300.0],
    [22.00,  290.0],
    [35.00,  280.0],
    [55.00,  270.0],
    [80.00,  260.0],
    [110.0,  250.0],
    [145.0,  240.0],
    [175.0,  232.0],
    [195.0,  225.0],
    [208.0,  220.0],
    [209.0,  215.0],
    [205.0,  210.0],
    [195.0,  205.0],
    [175.0,  200.0],
    [145.0,  195.0],
    [110.0,  190.0],
    [75.00,  185.0],
    [45.00,  180.0],
    [22.00,  175.0],
    [10.00,  170.0],
    [4.000,  165.0],
    [1.500,  160.0],
    [0.500,  155.0],
    [0.100,  150.0],
    [0.000,  100.0],
])  # shape (34, 2)


# ---------------------------------------------------------------------------
# 1. build_ver_profile
# ---------------------------------------------------------------------------

def build_ver_profile(mode: str = 'nightglow') -> callable:
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
        Returns 0.0 for altitudes outside the tabulated range.

    Notes
    -----
    Conversion from table units (cm⁻³ s⁻¹) to SI per steradian:
        VER_m3_s1_sr1 = VER_cm3_s1 * 1e6 / (4 * pi)
    The 1e6 = (100)³ converts cm⁻³ to m⁻³.
    The 4π distributes isotropic emission into steradians.
    """
    if mode == 'nightglow':
        table = NIGHTGLOW_VER_TABLE
        # Table is already in ascending altitude order
        altitudes_m = table[:, 1] * 1000.0          # km → m
        ver_si      = table[:, 0] * 1e6 / (4.0 * np.pi)  # cm⁻³ s⁻¹ → m⁻³ s⁻¹ sr⁻¹
    elif mode == 'dayglow':
        table = DAYGLOW_VER_TABLE
        # Table is stored descending; flip to ascending for spline
        altitudes_m = np.flip(table[:, 1]) * 1000.0
        ver_si      = np.flip(table[:, 0]) * 1e6 / (4.0 * np.pi)
    else:
        raise ValueError(
            f"build_ver_profile: unknown mode '{mode}'. "
            "Choose 'nightglow' or 'dayglow'."
        )

    h_min = altitudes_m[0]
    h_max = altitudes_m[-1]

    spline = CubicSpline(altitudes_m, ver_si, extrapolate=False)

    def ver_func(h):
        """VER(altitude_m) → m⁻³ s⁻¹ sr⁻¹, clamped to 0 outside table range."""
        h = np.asarray(h, dtype=float)
        raw = spline(h)
        raw = np.nan_to_num(raw, nan=0.0)
        return np.where(
            (h >= h_min) & (h <= h_max),
            np.maximum(raw, 0.0),
            0.0,
        )

    return ver_func


# ---------------------------------------------------------------------------
# 2. altitude_along_los
# ---------------------------------------------------------------------------

def altitude_along_los(
    d:         np.ndarray,
    pa:        float,
    R_orbit_m: float,
    R_earth_m: float,
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
    d = np.asarray(d, dtype=float)
    r = np.sqrt(R_orbit_m**2 + d**2 - 2.0 * R_orbit_m * d * np.sin(pa))
    h = r - R_earth_m
    return np.clip(h, 0.0, 1.0e6)


# ---------------------------------------------------------------------------
# 3. compute_los_geometry
# ---------------------------------------------------------------------------

def compute_los_geometry(
    h_tangent_m: float,
    h_orbit_m:   float = ORBIT_ALTITUDE_M,
    h_max_m:     float = VER_LAYER_TOP_M,
    R_earth_m:   float = None,
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
    import astropy.constants as C
    R_e = C.R_earth.value if R_earth_m is None else R_earth_m

    if h_tangent_m < 0:
        raise ValueError(
            f"compute_los_geometry: h_tangent_m={h_tangent_m} < 0"
        )
    if h_tangent_m >= h_orbit_m:
        raise ValueError(
            f"compute_los_geometry: h_tangent_m ({h_tangent_m:.0f} m) "
            f">= h_orbit_m ({h_orbit_m:.0f} m)"
        )

    R_orbit = R_e + h_orbit_m
    R_tang  = R_e + h_tangent_m

    # Pitch angle: angle below horizontal at spacecraft
    pa = np.arccos(R_tang / R_orbit)

    # Distance from spacecraft to tangent point
    d_tangent = R_orbit * np.sin(pa)

    # Integration limits: where LOS crosses h_max altitude
    # Solve: R_orbit² + d² - 2·R_orbit·d·sin(pa) = (R_earth + h_max)²
    # => d² - 2·R_orbit·sin(pa)·d + (R_orbit² - R_max²) = 0
    R_max = R_e + h_max_m
    A = 1.0
    B_coef = -2.0 * R_orbit * np.sin(pa)
    C_coef = R_orbit**2 - R_max**2

    discriminant = B_coef**2 - 4.0 * A * C_coef
    if discriminant <= 0:
        raise ValueError(
            f"compute_los_geometry: LOS does not intersect h_max={h_max_m:.0f} m "
            f"layer (discriminant={discriminant:.3e}). "
            "Try increasing h_max_m."
        )

    sqrt_d = np.sqrt(discriminant)
    d1 = (-B_coef - sqrt_d) / 2.0
    d2 = (-B_coef + sqrt_d) / 2.0

    # d_near must be non-negative
    d_near = max(0.0, d1)
    d_far  = d2

    return {
        'pitch_angle_rad': float(pa),
        'R_orbit_m':       float(R_orbit),
        'd_tangent_m':     float(d_tangent),
        'd_near_m':        float(d_near),
        'd_far_m':         float(d_far),
    }


# ---------------------------------------------------------------------------
# 4. integrate_los_emission
# ---------------------------------------------------------------------------

def integrate_los_emission(
    ver_func: callable,
    geometry: dict,
    n_quad:   int = 200,
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
    pa        = geometry['pitch_angle_rad']
    R_orbit_m = geometry['R_orbit_m']
    d_near    = geometry['d_near_m']
    d_far     = geometry['d_far_m']

    import astropy.constants as C
    R_e = C.R_earth.value

    def integrand(d):
        h = altitude_along_los(d, pa, R_orbit_m, R_e)
        return ver_func(h)

    B, _ = fixed_quad(integrand, d_near, d_far, n=n_quad)
    return float(B)


# ---------------------------------------------------------------------------
# 5. signal_photons_per_pixel
# ---------------------------------------------------------------------------

def signal_photons_per_pixel(
    column_brightness_m2_s1_sr1: float,
    aperture_m2:      float = None,
    alpha_rad_per_px: float = None,
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
    if aperture_m2 is None:
        aperture_m2 = APERTURE_M2
    if alpha_rad_per_px is None:
        alpha_rad_per_px = ALPHA_RAD_PER_PX

    Omega_pixel = alpha_rad_per_px ** 2
    return float(column_brightness_m2_s1_sr1 * aperture_m2 * Omega_pixel)


# ---------------------------------------------------------------------------
# 6. signal_to_I_line
# ---------------------------------------------------------------------------

def signal_to_I_line(
    photons_per_s: float,
    t_exp:   float = None,
    qe:      float = None,
    t_opt:   float = None,
    g_em:    float = None,
    I0:      float = 1000.0,
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
    if t_exp is None:
        from windcube.constants import INTEGRATION_TIME_S
        t_exp = INTEGRATION_TIME_S
    if qe is None:
        from windcube.constants import CCD97_QE_630NM
        qe = CCD97_QE_630NM
    if t_opt is None:
        from windcube.constants import OPTICAL_THROUGHPUT
        t_opt = OPTICAL_THROUGHPUT
    if g_em is None:
        from windcube.constants import EM_GAIN
        g_em = EM_GAIN

    counts = photons_per_s * t_exp * qe * t_opt * g_em
    return float(counts / I0)


# ---------------------------------------------------------------------------
# 7. compute_signal_budget
# ---------------------------------------------------------------------------

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
    # Step 1: build VER profile
    ver_func = build_ver_profile(mode)

    # Step 2: LOS geometry
    geometry = compute_los_geometry(h_tangent_m, h_orbit_m)
    pa_deg = np.degrees(geometry['pitch_angle_rad'])

    if verbose:
        print(f"[NB03] pitch_angle      = {pa_deg:.1f} deg")

    # Step 3: column brightness
    B = integrate_los_emission(ver_func, geometry)

    if verbose:
        print(f"[NB03] column_brightness = {B:.3e} ph m\u207b\u00b2 s\u207b\u00b9 sr\u207b\u00b9")

    # Step 4: photons per pixel per second
    F = signal_photons_per_pixel(B)

    if verbose:
        print(f"[NB03] photons/pixel/s  = {F:.1f}")

    # Step 5: full detector chain → counts
    counts = signal_to_I_line(F, I0=1.0) * 1000.0   # counts = F*t*qe*t_opt*g_em
    # Actually compute counts properly:
    from windcube.constants import (
        INTEGRATION_TIME_S as t_exp_s,
        CCD97_QE_630NM     as qe_val,
        OPTICAL_THROUGHPUT as t_opt_val,
        EM_GAIN            as g_em_val,
    )
    counts = F * t_exp_s * qe_val * t_opt_val * g_em_val

    if verbose:
        print(f"[NB03] counts/pixel     = {counts:.1f}")

    # Step 6: I_line
    I_line = signal_to_I_line(F)

    if verbose:
        print(f"[NB03] I_line           = {I_line:.3f}")

    return {
        'mode':                mode,
        'h_tangent_m':         float(h_tangent_m),
        'h_orbit_m':           float(h_orbit_m),
        'pitch_angle_deg':     float(pa_deg),
        'column_brightness':   float(B),
        'photons_per_pixel_s': float(F),
        'counts_per_pixel':    float(counts),
        'I_line':              float(I_line),
        'geometry':            geometry,
    }
