# WindCube FPI Science Operations Center (SOC) Pipeline — Spec Inventory
## Project: soc_sewell | As of 2026-04-17

### MISSION CONTEXT
WindCube CubeSat (510 km SSO, 97.44°) measures thermospheric horizontal winds (~10–400 m/s)
via Doppler shift of OI 630.0304 nm airglow emission using an onboard Fabry-Pérot Interferometer.
Pipeline converts raw 512×512 fringe images (2×2-binned → 256×256) into geolocated L2 wind
vectors (v_zonal, v_meridional) at ~250 km tangent altitude.

### KEY PHYSICAL CONSTANTS (src/constants.py)
- OI rest wavelength: 630.0304 nm (air, NIST ASD) — NOT 630.0 nm
- Neon cal lines: λ₁=640.2248 nm, λ₂=638.2991 nm
- Etalon gap: 20.008 mm (FlatSat build report) — NOT legacy 20.670 mm
- Etalon reflectivity: 0.53 (effective, FlatSat)
- Tangent height: 250 km (OI 630 nm peak emission)
- CCD: 2×2 binned → 256×256 px, 32 µm binned pitch
- Depression angle: 15.73° (equatorial nominal)

---

## TIER 0 — FOUNDATIONAL SPECS

**S01** `specs/S01_repo_conventions_2026-04-05.md`
Repo workflow, naming conventions, two-tool workflow (Claude AI for specs → Claude Code for impl).
Status: ✓ Authoritative

**S02** `specs/S02_pipeline_overview_2026-04-05.md`
Complete pipeline architecture, data-flow diagram, interface contracts for all module boundaries.
Status: ✓ Authoritative

**S03** `specs/S03_physical_constants_2026-04-06.md`
Single-source-of-truth for all physical constants. Tests: 8/8 pass.
Status: ✓ Authoritative

**S04** `specs/S04_uncertainty_standards_2026-04-05.md`
Uncertainty reporting: σ and 2σ together for all fitted params. Quality flag bitmask definitions.
Status: ✓ Authoritative

---

## TIER 1 — GEOMETRY (NB-series)

**NB00** `specs/NB00_wind_map_2026-04-06.md`
Truth wind map interface; 4 backends: T1=uniform, T2=analytic, T3=HWM14 climatology, T4=storm-perturbed.
Impl: `src/windmap/nb00_wind_map_2026_04_06.py`
Status: ✓ Implemented, passing tests

**NB01** `specs/NB01_orbit_propagator_2026-04-16.md`
SGP4 orbit propagation; ECI position/velocity + WGS84 geodetic per epoch. Batch vectorized (28× speedup).
Impl: `src/geometry/nb01_orbit_propagator_2026_04_16.py`
Status: ✓ Implemented, 8/8 tests pass

**NB02** `specs/NB02_geometry_2026-04-16.md`
Four sub-modules — all ✓ implemented:
- NB02a `nb02a_boresight_2026_04_16.py` — quaternion → LOS unit vector (ECI J2000)
- NB02b `nb02b_tangent_point_2026_04_16.py` — ray–ellipsoid intersection at 250 km
- NB02c `nb02c_los_projection_2026_04_16.py` — wind + S/C velocity + Earth rotation → v_rel (Doppler observable)
- NB02d `nb02d_l1c_calibrator_2026_04_16.py` — L1c inversion; removes S/C + Earth rotation → physical wind
Status: ✓ All passing

**NB03** `specs/S07b_nb03_ver_source_model_2026-04-12.md`
Volume emission rate (VER) model for OI 630 nm as function of altitude.
Impl: `src/fpi/nb03_ver_source_model_2026_04_12.py`
Status: ✓ Implemented, 8/8 tests pass

**INT01** `specs/INT01_geometry_notebook_2026-04-16.md`
Geometry integration notebook: validates NB00+NB01+NB02 end-to-end over 16 orbits.
Impl: `notebooks/INT01_geometry_pipeline_2026-04-16.ipynb`
8 validation checks incl. V2 (ECI dot-product lead distance), V8 (metadata array).
Status: ✓ Complete, all 8 checks pass

---

## TIER 2 — FPI FORWARD MODEL (M-series)

**S09/M01** `specs/S09_m01_airy_forward_model_2026-04-13.md`
Modified Airy diffraction model: etalon fringe physics, PSF convolution, envelope polynomial.
Output: Ã(r; λ) radial fringe profiles.
Impl: `src/fpi/m01_airy_forward_model_2026_04_05.py`
Status: ✓ Implemented, passing tests

**S10/M02** `specs/S10_m02_calibration_synthesis_2026-04-05.md`
Synthesizes neon calibration images (λ₁=640.2248 nm, λ₂=638.2991 nm).
Impl: `src/fpi/m02_calibration_synthesis_2026_04_05.py`
Status: ✓ Implemented, passing tests

**S11/M04** `specs/S11_m04_airglow_synthesis_2026-04-05.md`
Synthesizes OI 630 nm science images with Doppler shift from v_rel.
Impl: `src/fpi/m04_airglow_synthesis_2026_04_05.py`
Status: ✓ Implemented, passing tests

---

## TIER 3 — REDUCTION

**S12/M03** `specs/S12_m03_annular_reduction_2026-04-06.md`
2D→1D: fringe centre detection + radial profile extraction via r² annular binning.
Impl: `src/fpi/m03_annular_reduction_2026_04_06.py`
Status: ✓ Implemented, passing tests

---

## TIER 4 — CALIBRATION INVERSION

**S13/Tolansky** `specs/S13_tolansky_analysis_2026-04-13.md`
Tolansky spatial-heterodyne method for single-line etalon gap measurement.
Impl: `src/two_d_one_d_reduction/tolansky.py`
Status: ✓ Implemented, 3 test variants pass

**S14/M05** `specs/S14_m05_calibration_inversion_2026-04-06.md`
Staged inversion: recovers etalon gap, reflectivity, magnification, envelope, PSF width from neon fringe profile.
Impl: `src/fpi/m05_calibration_inversion_2026_04_06.py`
Status: ✓ Implemented, passing tests

---

## TIER 5 — AIRGLOW INVERSION

**S15/M06** `specs/S15_m06_airglow_inversion_2026-04-06.md`
Fits OI line centre (λ_c) from science fringe profile. Yields Doppler shift → v_rel.
Impl: `src/fpi/m06_airglow_inversion_2026_04_06.py`
Status: ✓ Implemented, passing tests

---

## TIER 6 — WIND RETRIEVAL

**S16/M07** `specs/S16_m07_wind_retrieval_2026-04-06.md`
Weighted least-squares (WLS) vector wind retrieval: along-track + cross-track LOS winds → v_zonal, v_meridional.
Impl: `src/fpi/m07_wind_retrieval_2026_04_06.py`
Status: ✓ Implemented, 8/8 tests pass

---

## TIER 7 — INTEGRATION VALIDATION

**S17/INT02** `specs/S17_int02_fpi_chain_2026-04-07.md`
FPI instrument chain validation (M01→M07): forward model + reduction + inversion + wind retrieval.
Impl: `src/integration/int02_fpi_chain_2026_04_07.py`
Status: ✓ Complete, 16/16 tests pass

**S18/INT03** `specs/S18_int03_end_to_end_pipeline_2026-04-10.md`
Full end-to-end pipeline validation (NB00→NB01→NB02→M01→M07→L2 product).
Impl: `src/integration/int03_end_to_end_2026_04_10.py`
Status: ✓ Complete, 14/14 tests pass

---

## TIER 8 — DATA PRODUCTS

**S19/P01** `specs/P01_metadata_2026-04-06.md`
Level-0 image metadata sidecar schema: ImageMetadata dataclass, binary format, JSON I/O.
Captures AOCS state, instrument config, timing, provenance.
Impl: `src/metadata/p01_image_metadata_2026_04_06.py`
Status: ✓ Authoritative, passing tests

**S20** `specs/s20_l2_netcdf_wind_vector_schema_2026-04-11.md`
L2 NetCDF wind product schema: v_zonal, v_meridional, geolocation, uncertainties, quality flags.
Impl: `src/netCDF/m08_l2_writer.py`
Status: ✓ Complete, passing tests

---

## AUXILIARY SPECS

**A01** `specs/A01_windcube_collaborator_guide_2026-04-05.md`
Onboarding guide for spec-writing collaborators; workflow overview.
Status: ✓ Complete

**G01** `specs/G01_synthetic_metadata_generator_2026-04-16.md`
Interactive metadata array + binary synthetic image generator (CONOPS-driven).
Ties together NB01/NB02 + P01 + M02/M04. Produces 42-column CSV + binary frame files.
Impl: `validation/gen01_synthetic_metadata_generator_2026_04_16.py`
Status: ✓ Complete, 21 checks pass

---

## SCIENCE GOALS
- **SG1**: Detect stratospheric forcing signatures (waves, tides) in thermospheric winds
- **SG2**: Characterize tidal-driven winds from diurnal and semi-diurnal oscillations

## OPEN ITEMS (scotts_to_do_list.txt)
1. Field-of-view limiting for airglow synthesis (currently ±1.65° — under review)
2. Single-line Tolansky validation against lab neon lamp images
3. More accurate VER source strength model (altitude-integrated emission)

## TEST SUITE SUMMARY
21+ test files; majority passing. Known pre-existing failures:
- test_z04.py — missing joblib package
- test_s06_nb01_orbit_propagator.py — path/import issue (module itself passes)
- test_z02_synthetic_airglow_generator.py — wrong script path in test fixture
