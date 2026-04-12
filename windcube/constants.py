"""
windcube/constants.py
WindCube FPI Pipeline — instrument and physical constants.
All values are authoritative for the pipeline; modules must import from
here rather than hardcoding.

Sources:
  OI_WAVELENGTH_NM    : NIST atomic line database
  TOLANSKY_D_MM       : Z01 two-line neon Tolansky fit (FlatSat)
  TOLANSKY_F_MM       : Z01 two-line neon Tolansky fit (FlatSat)
  ALPHA_RAD_PX        : Z01 two-line neon Tolansky fit (2x2 binned)
  ICOS_GAP_MM         : ICOS mechanical spacer measurement
  D_25C_MM            : ICOS_GAP_MM minus Pat & Nir pre-load compression
  D_PRELOAD_NM        : Pat & Nir clamping compression (Zerodur spacer)
  ETALON_THERMAL_NM_C : Measured Zerodur thermal expansion coefficient
  CCD_PIXEL_PITCH_M   : 2x2 binned pixel pitch (16 um native x 2)
  NE_WAVELENGTH_1_NM  : Ne 6402.2460 A, IAU standard "S" (Burns 1950)
  NE_WAVELENGTH_2_NM  : Ne 6382.9914 A, IAU standard "S" (Burns 1950)
  F_NOMINAL_MM        : COTS lens nominal focal length
  R_REFL              : FlatSat effective etalon reflectivity
  R_MAX_PX            : FlatSat/flight maximum fringe radius
"""

# ---------------------------------------------------------------------------
# Physical / astronomical constants
# ---------------------------------------------------------------------------

# OI 630.0 nm rest wavelength [nm]
OI_WAVELENGTH_NM: float = 630.0

# ---------------------------------------------------------------------------
# Neon calibration wavelengths (Z01 two-line source)
# IAU standard "S" lines, Burns, Adams & Longwell (1950)
# ---------------------------------------------------------------------------

# Ne 6402.2460 A = 640.22460 nm  (primary, high-amplitude family)
NE_WAVELENGTH_1_NM: float = 640.2248   # rounded to 4 d.p. per S03

# Ne 6382.9914 A = 638.29914 nm  (secondary, low-amplitude family)
NE_WAVELENGTH_2_NM: float = 638.2991   # rounded to 4 d.p. per S03

# ---------------------------------------------------------------------------
# Opto-mechanical calibration constants
# Recovered from Z01 two-line neon Tolansky fit on FlatSat data
# ---------------------------------------------------------------------------

# Etalon gap recovered by Tolansky two-line fit [mm]
# NOTE: ~98 um larger than D_25C_MM -- discrepancy under investigation
TOLANSKY_D_MM: float = 20.106

# Focal length recovered by Tolansky two-line fit [mm]
# COTS lens ~0.68% short of nominal 200 mm
TOLANSKY_F_MM: float = 199.12

# Plate scale (2x2 binned) recovered by Tolansky two-line fit [rad/px]
# Old value 8.5e-5 rad/px permanently retired
ALPHA_RAD_PX: float = 1.6071e-4

# COTS imaging lens nominal focal length [mm]
F_NOMINAL_MM: float = 200.0

# FlatSat effective etalon reflectivity (dimensionless)
R_REFL: float = 0.53

# Maximum fringe radius used in annular reduction [px] (FlatSat / flight)
R_MAX_PX: int = 110

# ---------------------------------------------------------------------------
# Etalon gap -- mechanical measurements and thermal model
# Reference temperature: 25 degrees C (etalon heater setpoint)
# ---------------------------------------------------------------------------

# ICOS mechanical spacer measurement [mm]
# Used ONLY to resolve FSR integer-order ambiguity N_int; not the Tolansky prior
ICOS_GAP_MM: float = 20.008

# Pat & Nir pre-load clamping compression [nm]
# Applied to ICOS measurement to obtain D_25C_MM
D_PRELOAD_NM: float = 70.8029

# Best-estimate etalon gap at 25 degrees C setpoint [mm]
# = ICOS_GAP_MM - D_PRELOAD_NM * 1e-6
D_25C_MM: float = 20.0079291971   # = 20.008000000 - 0.0000708029 (D_PRELOAD_NM * 1e-6)

# Zerodur spacer thermal expansion coefficient [nm/degrees C]
# Measured from lab testing
ETALON_THERMAL_NM_C: float = 18.585

# ---------------------------------------------------------------------------
# CCD / detector constants
# ---------------------------------------------------------------------------

# 2x2 binned pixel pitch [m]  (16 um native x 2)
CCD_PIXEL_PITCH_M: float = 32e-6
