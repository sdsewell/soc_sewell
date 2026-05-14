"""
windcube/constants.py
WindCube FPI Pipeline — instrument and physical constants.
All values are authoritative for the pipeline; modules must import from
here rather than hardcoding.

Sources:
  OI_WAVELENGTH_NM    : NIST ASD (https://physics.nist.gov/PhysRefData/ASD/lines_form.html)
  OI_WAVELENGTH_VAC_NM: derived via Edlén (1966) air-to-vacuum formula
  ALPHA_RAD_PX        : S13a two-line neon Tolansky fit (2x2 binned, 2026-05-06)
  D_TOLANSKY_MM       : S13a two-line neon Tolansky fit (Benoit, 2026-05-06)
  ICOS_GAP_MM         : ICOS mechanical spacer measurement
  D_25C_MM            : ICOS_GAP_MM minus Pat & Nir pre-load compression
  D_PRELOAD_NM        : Pat & Nir clamping compression (Zerodur spacer)
  ETALON_THERMAL_NM_C : Measured Zerodur thermal expansion coefficient
  CCD_PIXEL_PITCH_M   : 2x2 binned pixel pitch (16 um native x 2)
  NE_WAVELENGTH_1_NM      : Ne 6402.2460 A, IAU standard "S" (Burns 1950), air
  NE_WAVELENGTH_1_VAC_NM  : derived via Edlén (1966) air-to-vacuum formula
  NE_WAVELENGTH_2_NM      : Ne 6382.9914 A, IAU standard "S" (Burns 1950), air
  NE_WAVELENGTH_2_VAC_NM  : derived via Edlén (1966) air-to-vacuum formula
  R_REFL              : FlatSat effective etalon reflectivity
  R_MAX_PX            : FlatSat/flight maximum fringe radius
"""

# ---------------------------------------------------------------------------
# H01 — constants required by airy_forward_model (Section 3 of H01 spec)
# All values are authoritative for the H01 Airy forward model.
# ---------------------------------------------------------------------------

# Etalon / optics
ETALON_GAP_M        : float = 20.1071e-3    # m  — S13a Tolansky Benoit (2026-05-06); 1σ = 0.0002 mm
ETALON_N            : float = 1.0           # —  — refractive index of etalon gap (air)
ETALON_R_INSTRUMENT : float = 0.53         # —  — effective reflectivity (FlatSat)
# ALPHA_RAD_PX defined below in opto-mechanical section (1.6085e-4 rad/px)

# CCD / FOV
CCD_PIXELS_UNBINNED : int   = 512           # px — physical pixels per side (CCD97)
FOV_DEG             : float = 1.65          # deg — full field of view

# OI airglow target line
OI_WAVELENGTH_AIR_M : float = 630.0304e-9  # m  — NIST ASD air wavelength (rest)

# Neon calibration lines — air wavelengths (Burns 1950) and vacuum (NIST ASD / Edlén 1966)
NE_WAVELENGTH_1_AIR_M : float = 640.2248e-9   # m  — strong line, air
NE_WAVELENGTH_1_VAC_M : float = 640.4018e-9   # m  — strong line, vacuum (±0.0001 nm)
NE_WAVELENGTH_2_AIR_M : float = 638.2991e-9   # m  — weak line, air
NE_WAVELENGTH_2_VAC_M : float = 638.47560e-9  # m  — weak line, vacuum (±0.00005 nm)
NE_INTENSITY_1        : float = 1.0           # —  — reference intensity ratio
NE_INTENSITY_2        : float = 0.36          # —  — weak/strong ratio

# Physical constants
SPEED_OF_LIGHT_MS : float = 299_792_458.0   # m/s — exact SI value

# ---------------------------------------------------------------------------
# Physical / astronomical constants
# ---------------------------------------------------------------------------

# OI 630.0 nm rest wavelength
# Air wavelength from NIST Atomic Spectra Database (standard air, 15°C, 1 atm):
#   https://physics.nist.gov/PhysRefData/ASD/lines_form.html
# Vacuum wavelength derived via Edlén (1966) formula:
#   n_air = 1 + (8342.13 + 2406030/(130 - σ²) + 15997/(38.9 - σ²)) × 10⁻⁸
#   σ = 1/λ_air(µm) = 1.587225 µm⁻¹  →  n_air = 1.00027656
#   λ_vac = λ_air × n_air = 630.0304 × 1.00027656 = 630.2046 nm
OI_WAVELENGTH_NM:     float = 630.0304   # nm, air  (NIST ASD)
OI_WAVELENGTH_VAC_NM: float = 630.2046   # nm, vacuum  (Edlén 1966 conversion)

# ---------------------------------------------------------------------------
# Neon calibration wavelengths (Z01 two-line source)
# Air wavelengths: IAU standard "S" lines, Burns, Adams & Longwell (1950)
# NIST Atomic Spectra Database: https://physics.nist.gov/PhysRefData/ASD/lines_form.html
# Vacuum wavelengths derived via Edlén (1966) formula (same as OI above):
#   λ_vac = λ_air × n_air,  n_air = 1 + (8342.13 + 2406030/(130−σ²) + 15997/(38.9−σ²))×10⁻⁸
# ---------------------------------------------------------------------------

# Ne 6402.2460 Å = 640.22460 nm (air)  — primary / high-amplitude family (640 nm)
# σ = 1/0.6402248 µm⁻¹ = 1.56197 µm⁻¹  →  n_air = 1.00027643
# λ_vac = 640.2248 × 1.00027643 = 640.4018 nm  (NIST: 640.4018 ± 0.0001 nm)
NE_WAVELENGTH_1_NM:     float = 640.2248   # nm, air   (Burns 1950, rounded to 4 d.p.)
NE_WAVELENGTH_1_VAC_NM: float = 640.4018   # nm, vacuum (NIST ASD; ±0.0001 nm)

# Ne 6382.9914 Å = 638.29914 nm (air)  — secondary / low-amplitude family (638 nm)
# σ = 1/0.63829914 µm⁻¹ = 1.56668 µm⁻¹  →  n_air = 1.00027645
# λ_vac = 638.29914 × 1.00027645 = 638.4756 nm  (NIST: 638.47560 ± 0.00005 nm)
NE_WAVELENGTH_2_NM:     float = 638.2991   # nm, air   (Burns 1950, rounded to 4 d.p.)
NE_WAVELENGTH_2_VAC_NM: float = 638.47560  # nm, vacuum (NIST ASD; ±0.00005 nm)

# ---------------------------------------------------------------------------
# Opto-mechanical calibration constants
# Recovered from two-line neon Tolansky fit on FlatSat data
# ---------------------------------------------------------------------------

# Etalon plate spacing recovered by S13a two-line Tolansky fit (Benoit method) [mm]
# d = 20.1071 ± 0.0002 mm  (1σ),  2σ = 0.0004 mm   (2026-05-06)
# NOTE: disagrees with D_25C_MM (mechanical prior) by ~99 µm; discrepancy unresolved.
# All pipeline code must use ICOS_GAP_MM / D_25C_MM for N_int resolution only.
D_TOLANSKY_MM:       float = 20.1071
SIGMA_D_TOLANSKY_MM: float = 0.0002   # 1σ [mm]

# Plate scale (2x2 binned) recovered by S13a two-line Tolansky fit [rad/px]
# alpha = 1.6085e-4 ± 1.3478e-8 rad/px  (1σ),  2σ = 2.6955e-8   (2026-05-06)
# Old value 1.6071e-4 rad/px superseded.
ALPHA_RAD_PX:       float = 1.6085e-4
SIGMA_ALPHA_RAD_PX: float = 1.3478e-8   # 1σ [rad/px]

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

# ---------------------------------------------------------------------------
# WindCube instrument / detector constants  (S07b / NB03 addendum)
# ---------------------------------------------------------------------------

# Entrance pupil area.  Placeholder value — update when aperture is
# confirmed from the as-built optical design.
APERTURE_M2: float = 0.06 * 0.03          # m²  (60 mm × 30 mm rectangular pupil)

# Total optical throughput: filters, mirrors, beamsplitters, etalon
# transmission losses.  From WindCube.py (A. Ridley).  Treat as
# provisional until measured on FlatSat.
OPTICAL_THROUGHPUT: float = 0.85           # dimensionless

# CCD97 quantum efficiency at 630 nm.  From Teledyne e2v CCD97 datasheet.
# Treat as provisional until device-level characterisation.
CCD97_QE_630NM: float = 0.90               # electrons / photon

# EMCCD electron multiplication gain.  Set to 1.0 (conventional CCD mode)
# until EM gain calibration is available.  Update when flight gain
# setting is determined.
EM_GAIN: float = 1.0                       # dimensionless

# Nominal science frame integration time.
INTEGRATION_TIME_S: float = 10.0           # seconds

# Plate scale (2×2 binned).  Authoritative Tolansky value from S13a (2026-05-06).
# Reproduced here so NB03 can compute pixel solid angle without
# importing from M01 (which would create a circular Tier dependency).
ALPHA_RAD_PER_PX: float = 1.6085e-4        # rad / binned pixel

# Orbital and observation geometry defaults
ORBIT_ALTITUDE_M:   float = 500_000.0     # m, nominal WindCube orbit
TANGENT_ALTITUDE_M: float = 250_000.0     # m, nominal thermospheric tangent height
VER_LAYER_TOP_M:    float = 490_000.0     # m, upper bound of emission layer for LOS integration

# --- Observation regime velocity bounds (H03) ---
# Cross-track (even orbits): thermospheric wind projected onto LOS
V_REL_CROSSTRACK_MAX_MS  =  1000.0   # m/s; symmetric: valid range [-1000, +1000]
# Along-track (odd orbits): spacecraft orbital velocity + wind projected onto LOS
V_REL_ALONGTRACK_MIN_MS  = -8000.0   # m/s; lower bound (maximum blueshift)
V_REL_ALONGTRACK_MAX_MS  = -6000.0   # m/s; upper bound (minimum blueshift)

# ---------------------------------------------------------------------------
# H07 — Wind vector retrieval constants (H07_wind_vector_retrieval_2026-05-14_v03.md §10)
# ---------------------------------------------------------------------------

# Earth rotation rate [rad/s] — same value as src.constants.EARTH_OMEGA_RAD_S
# H07 imports from src.constants directly; this alias allows windcube-only use.
EARTH_OMEGA_RAD_S  : float = 7.2921150e-5

# WGS84 reference ellipsoid semi-axes [m]
WGS84_A_M          : float = 6_378_137.0     # equatorial radius
WGS84_B_M          : float = 6_356_752.3142  # polar radius

# OI 630.0 nm rest wavelength (nominal Harding convention value) [nm]
OI_LAMBDA0_NM      : float = 630.0

# Nominal OI emission layer altitude [km]
OI_EMISSION_ALT_KM : float = 250.0

# Speed of light [m/s] — alias; same value as SPEED_OF_LIGHT_MS above
C_M_S              : float = 299_792_458.0

# Wind vector inversion thresholds (Stage I)
GDOP_MAX           : float = 100.0   # condition number threshold for ill-conditioning flag
N_MIN_FRAMES       : int   = 4       # minimum frames per bin; fewer → n_frames_flag=True

# FPI payload boresight in spacecraft body frame (-X_BRF per SI-UCAR-WC-RP-004 §2.4.2.1)
H07_BORESIGHT_BODY : list  = [-1.0, 0.0, 0.0]
