# =============================================================================
# S03 — Physical Constants
# soc_sewell/src/constants.py
#
# Spec:        S03_physical_constants_2026-04-05.md
# Spec date:   2026-04-05
# Generated:   2026-04-05  (Claude Code)
# Tool:        Claude Code
# Last tested: 2026-04-05  (8/8 tests pass, pytest 8.x)
# Depends on:  nothing
#
# Single source of truth for every physical, spectroscopic, geodetic, orbital,
# and instrument constant used by the WindCube FPI pipeline.
# No other module may hardcode a value that appears in this table.
# =============================================================================

import math

# ---------------------------------------------------------------------------
# 3.1 Fundamental physical constants
# Source: CODATA 2018 (exact SI values)
# ---------------------------------------------------------------------------
SPEED_OF_LIGHT_MS    = 299_792_458.0     # m/s  — exact by SI definition
BOLTZMANN_J_PER_K    = 1.380649e-23      # J/K  — exact by SI definition
PLANCK_J_S           = 6.62607015e-34   # J·s  — exact by SI definition
EARTH_GRAV_PARAM_M3_S2 = 3.986004418e14  # m³/s² — EGM2008 GM for WGS84 Earth
EARTH_OMEGA_RAD_S    = 7.2921150e-5      # rad/s — WGS84 Earth rotation rate
EARTH_J2             = 1.08263e-3        # —     — EGM2008 J2 zonal harmonic

# ---------------------------------------------------------------------------
# 3.2 WGS84 geodetic constants
# Source: WGS84 standard (NGA.STND.0036)
# ---------------------------------------------------------------------------
WGS84_A_M  = 6_378_137.0              # m  — equatorial semi-major axis (exact)
WGS84_B_M  = 6_356_752.314_245        # m  — polar semi-minor axis (derived)
WGS84_F    = 1.0 / 298.257_223_563    # —  — flattening parameter (exact)
WGS84_E2   = 1.0 - (WGS84_B_M / WGS84_A_M) ** 2  # —  — first eccentricity squared
# WGS84_E2 = 6.694379990141317e-3; computed to maintain consistency with B and A

# ---------------------------------------------------------------------------
# 3.3 Spectroscopic constants — OI airglow target line
# Source: NIST Atomic Spectra Database (NIST ASD)
#
# LEGACY CORRECTION: legacy code uses OI_WAVELENGTH_M = 630.0e-9.
# The correct NIST air wavelength is 630.0304 nm. This is the rest wavelength
# for all Doppler shift calculations in this pipeline.
# ---------------------------------------------------------------------------
OI_WAVELENGTH_M     = 630.0304e-9     # m — OI 630 nm air wavelength (NIST ASD)
OI_WAVELENGTH_VAC_M = 630.2010e-9     # m — OI 630 nm vacuum wavelength (NIST ASD)
OXYGEN_MASS_KG      = 2.6567e-26      # kg — one oxygen-16 atom (16 u × 1.66054e-27 kg/u)

# Doppler formula: v_rel = SPEED_OF_LIGHT_MS * (lambda_c - OI_WAVELENGTH_M) / OI_WAVELENGTH_M
# Positive v_rel = recession (redshift, source moving away from spacecraft).

# ---------------------------------------------------------------------------
# 3.4 Spectroscopic constants — neon calibration lamp
# Source: NIST Atomic Spectra Database (Ne I, air wavelengths)
#
# These two Ne I lines are the brightest in the 630–640 nm window.
# Their separation (~188 FSR) is used in M05 Stage 0 to anchor the etalon gap.
# ---------------------------------------------------------------------------
NE_WAVELENGTH_1_M  = 640.2248e-9   # m — primary Ne line; relative intensity = 1.0
NE_WAVELENGTH_2_M  = 638.2991e-9   # m — secondary Ne line; relative intensity = 0.8
NE_INTENSITY_1     = 1.0           # — — reference (arbitrary normalisation)
NE_INTENSITY_2     = 0.8           # — — ratio of secondary to primary line

# ---------------------------------------------------------------------------
# 3.5 Etalon and optical constants
# Sources: ICOS build report GNL-4096-R iss1; FlatSat calibration; WindCube optical design
#
# LEGACY CORRECTION (etalon gap): FlatSat code used ETALON_GAP_M = 20.670e-3 m.
# That value arises from an FSR-period ambiguity and is wrong. The ICOS
# mechanical measurement of 20.008 mm is the correct value.
#
# LEGACY CORRECTION (depression angle): see Section 3.7.
# ---------------------------------------------------------------------------
ETALON_GAP_M           = 20.008e-3   # m — ICOS build report §7.4 (spacer measurement)
ETALON_GAP_TOLERANCE_M = 0.010e-3    # m — manufacturing tolerance ±0.010 mm
ETALON_R_COATING       = 0.80        # — — as-deposited coating reflectivity at 630 nm
ETALON_R_INSTRUMENT    = 0.53        # — — effective instrument R from FlatSat fringe contrast
ETALON_N               = 1.0         # — — refractive index of etalon gap (air/vacuum)
FOCAL_LENGTH_M         = 0.200       # m — FPI imaging lens focal length
CCD_PIXEL_UM           = 16.0        # µm — CCD97-00 native pixel pitch (unbinned)
CCD_PIXEL_2X2_UM       = 32.0        # µm — effective pixel pitch after 2×2 binning
ALPHA_RAD_PX           = CCD_PIXEL_2X2_UM * 1e-6 / FOCAL_LENGTH_M
# ALPHA_RAD_PX = 32e-6 / 0.200 = 1.60e-4 rad/px
# Maps pixel radius r (px) to angle from optical axis: θ(r) ≈ ALPHA_RAD_PX * r

# ---------------------------------------------------------------------------
# 3.6 CCD detector constants
# Source: CCD97-00 datasheet (Teledyne e2v)
# ---------------------------------------------------------------------------
CCD_PIXELS_NATIVE    = 512           # px — active pixels per side, unbinned
CCD_PIXELS_2X2       = 256           # px — active pixels per side after 2×2 binning
CCD_DARK_RATE_E_PX_S = 400.0        # e⁻/px/s — at 20°C; temperature-dependent
CCD_READ_NOISE_E     = 2.2           # e⁻ rms — conventional readout (no EM gain, 50 kHz)
CCD_READ_NOISE_EM_E  = 1.0           # e⁻ rms — EM gain mode (1 MHz, 1000× gain)
CCD_FULL_WELL_E      = 130_000       # e⁻ — peak signal capacity per pixel
CCD_EM_GAIN_DEFAULT  = 200           # — — default EM gain; do not exceed 300× without instruction
CCD_QE_PEAK          = 0.90          # — — peak QE at ~550 nm
CCD_QE_630           = 0.85          # — — QE at 630 nm (from QE curve)

# ---------------------------------------------------------------------------
# 3.7 Mission and orbital constants
# Source: WC-SE-0003 v8 ConOps; WGS84
#
# LEGACY CORRECTION (depression angle): earlier documents used 23.4°.
# The correct value is derived below from the spacecraft and tangent altitudes.
# ---------------------------------------------------------------------------
SC_ALTITUDE_KM         = 510.0              # km — nominal spacecraft altitude above WGS84
SC_ALTITUDE_RANGE_KM   = (500.0, 550.0)     # km — operational altitude range
TP_ALTITUDE_KM         = 250.0              # km — OI 630 nm tangent height
TP_ALTITUDE_TOLERANCE_KM = 5.0             # km — THRF model error budget
SC_VELOCITY_MS         = 7600.0             # m/s — approximate circular orbit at 510 km
SC_ORBITAL_PERIOD_S    = 5640.0             # s — ~94 minutes
ORBIT_INCLINATION_DEG  = 97.4              # deg — sun-synchronous inclination
LTAN_HOURS             = 6.0               # hours — local time of ascending node (dawn-dusk)
SCIENCE_CADENCE_S      = 10.0              # s — nominal image cadence

# Depression angle: arccos(R_tp / R_sc) where R = WGS84 equatorial radius + altitude
# Uses sea-level equatorial radius approximation.
# LEGACY CORRECTION: 23.4° is superseded.
_R_SC_KM = WGS84_A_M / 1e3 + SC_ALTITUDE_KM   # 6888.137 km
_R_TP_KM = WGS84_A_M / 1e3 + TP_ALTITUDE_KM   # 6628.137 km
DEPRESSION_ANGLE_DEG = math.degrees(math.acos(_R_TP_KM / _R_SC_KM))
# = degrees(arccos(6628.137 / 6888.137)) = 15.73°

# ---------------------------------------------------------------------------
# 3.8 Wind measurement and error budget
# Source: WindCube Science Traceability Matrix (STM) v1
# ---------------------------------------------------------------------------
WIND_BIAS_BUDGET_MS   = 9.8     # m/s — required 1σ wind precision (STM Monte Carlo)
WIND_MAX_STORM_MS     = 400.0   # m/s — maximum wind speed to resolve (G2 storm)
WIND_MIN_DETECTABLE_MS = 20.0   # m/s — minimum detectable wind (5% of peak storm)
LAT_RANGE_DEG         = (-40.0, 40.0)  # deg — primary science latitude band (SG1+SG2)

# =============================================================================
# Section 4 — Derived quantities
# Computed from primary constants above for convenience.
# Do not redefine these in other modules — import from here.
# =============================================================================

# Etalon FSR at primary neon line (m)
# FSR = lambda^2 / (2 * t)
ETALON_FSR_NE1_M = NE_WAVELENGTH_1_M ** 2 / (2.0 * ETALON_GAP_M)
# = (640.2248e-9)^2 / (2 * 20.008e-3) ≈ 1.024e-11 m = 10.24 pm

# Etalon FSR at OI 630.0304 nm (m)
ETALON_FSR_OI_M = OI_WAVELENGTH_M ** 2 / (2.0 * ETALON_GAP_M)
# = (630.0304e-9)^2 / (2 * 20.008e-3) ≈ 9.92e-12 m = 9.92 pm

# Velocity equivalent of one FSR at OI 630 nm (m/s)
# delta_v = c * FSR / lambda
VELOCITY_PER_FSR_MS = SPEED_OF_LIGHT_MS * ETALON_FSR_OI_M / OI_WAVELENGTH_M
# ≈ 4720 m/s

# Neon line separation in wavelength (m) and in FSR units
NE_DELTA_LAMBDA_M  = NE_WAVELENGTH_1_M - NE_WAVELENGTH_2_M   # 1.9257e-9 m
NE_SEPARATION_FSR  = NE_DELTA_LAMBDA_M / ETALON_FSR_NE1_M    # ≈ 187.9 FSR

# CCD plate scale (= ALPHA_RAD_PX; provided here under the plate-scale name as well)
PLATE_SCALE_RAD_PX = CCD_PIXEL_2X2_UM * 1e-6 / FOCAL_LENGTH_M   # = ALPHA_RAD_PX
