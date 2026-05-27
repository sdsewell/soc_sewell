| Spec | Module       | Status | Tests    | Last verified |
|------|-------------|--------|----------|---------------|
| S01  | —           | impl   | n/a      | 2026-04-06    |
| S02  | —           | impl   | n/a      | 2026-04-06    |
| S03  | —           | v4     | 10/10    | 2026-04-29    |
| S04  | —           | impl   | n/a      | 2026-04-06    |
| S05  | NB00        | FIXED  | n/a      | 2026-05-16    |
| S06  | NB01        | impl   | n/a      | 2026-04-06    |
| S07  | NB02        | impl   | n/a      | 2026-04-06    |
| S07b | NB03        | impl   | 8/8      | 2026-04-12    |
| S08  | INT01       | impl   | 8/8      | 2026-04-16    |
| S09  | M01         | impl   | passing  | 2026-04-06    |
| S10  | M02         | impl   | passing  | 2026-04-06    |
| S11  | M04         | impl   | passing  | 2026-04-06    |
| S12  | M03         | PASS   | 6/7      | 2026-05-13    |
| M03-add | m03_airglow_synthesis_2026_05_24 | IMPLEMENTED | 7/7 | 2026-05-24 |
| M06-add | m06_airglow_inversion_2026_05_24 | IMPLEMENTED | 8/8 | 2026-05-24 |
| S13  | Tolansky    | impl   | passing  | 2026-04-06    |
| S13a | tolansky_2line_2026-05-05 | impl | 7/7 | 2026-05-05 |
| S13b | tolansky_1line_2026-05-05 | impl | 6/6 | 2026-05-05 |
| S14  | M05         | impl   | passing  | 2026-04-06    |
| S15  | M06         | impl   | 8/8      | 2026-04-21    |
| S16  | M07         | impl   | 8/8      | 2026-04-06    |
| S17  | INT02       | impl   | 16/16    | 2026-04-07    |
| S18  | INT03       | impl   | 14/14    | 2026-04-11    |
| S19  | P01         | impl   | 9/9      | 2026-05-14    |
| S20  | L2 product  | impl   | passing  | 2026-04-11    |
| Z01  | validate-cal| impl   | 6-stage  | 2026-04-11    |
| Z01a | OI630-cal   | impl   | 16/16    | 2026-04-12    |
| Z02  | airglow-gen | impl   | 8/8      | 2026-04-10    |
| Z03  | cal-gen     | impl   | 25/25    | 2026-04-28    |
| Z04  | snr-sweep   | impl   | 6/6      | 2026-04-11    |
| F01  | neon-fit    | impl   | 10/10+skip | 2026-04-22  |
| F02  | airglow-fit | impl   | 8/8+skip | 2026-04-21    |
| G01  | GEN01 mission-dataset-syn | PASS v16 | smoke+HWM14 e2e | 2026-05-16 |
| G01_dark | dark frame synthesis | PASS v1.2 | 5/5 | 2026-05-16 |
| H03  | airglow-syn | PASS   | 2/2      | 2026-05-13    |
| H06  | airglow-inv | PASS   | 2/2      | 2026-05-14    |
| CAL01 | fpi_cal_pipeline | COMPLETE | 2026-05-27 | 11-param H05 (I3 cubic vignetting); r2_max_fit=12000 px² (r_max≈109.5 px, 800 bins); chi2/nu=1.587; single Figure 6; .npy includes r2_max_fit provenance |
| MC01 | mc01_fpi_mc_engine | IMPLEMENTED | 10/10 | 2026-05-24 |
| MC02 | mc02_fpi_mc_simulations | IMPLEMENTED | 10/10 | 2026-05-24 |

## Known pre-existing test failures (not introduced by current work)

| Test file                                    | Reason                        |
|----------------------------------------------|-------------------------------|
| test_z04.py                                  | joblib now installed; re-check pending |
| test_s06_nb01_orbit_propagator.py            | missing module (NB01 not impl)|
| test_z02_synthetic_airglow_generator.py      | wrong script path             |

These failures pre-date the S07b session and are excluded from regression
assessment until the relevant modules are fixed or installed.

## Output file policy

`data_reference/` PNGs and `.npy` files are pipeline outputs regenerated
on each run. They are NOT committed source files and should remain
unstaged. The authoritative calibration result is
`data_reference/<stem>_cal_result.npy` produced by the single
production H05 run (n_pairs=16, r2_max_fit=12000 px²).

## H05 parameter selection (2026-05-27)

Two sweep analyses established the production parameters:

- **Sweep A** (n_pairs=10..16): two-basin LM structure at 13→14 pairs.
  Pairs 14–16 give physically correct R1≈0.25, R2≈0.29. n_pairs=16
  is the authoritative Tolansky seed.

- **Sweep B** (r2_max=10000..22500 px²): chi2/nu minimum at 12000 px²
  (1.587). I3 sign flip at r2>18000 identifies the 638.3 nm bleed-
  through contamination boundary. t_fit stable ±0.5 nm across all
  cutoffs. r2_max_fit=12000 is the production cutoff.

## Notes

- NB00 HWM14WindMap (spec v2026-05-16): uses `pyhwm2014` backend (pyHWM14/pyhwm2014/hwm14.cp312-win_amd64.pyd).
  T3 quiet-time and T4 storm wind maps verified at 250 km (84c5250).
  lgpedersen/hwm14 not compatible (numpy.distutils removed in NumPy 2.x; different API).

- pyhwm2014 in windcube env (2026-05-16): windcube uses Python 3.11 (MSVC); pyhwm2014 required >=3.12.
  Resolved by building hwm14.cp311-win_amd64.pyd with f2py --backend meson + hwm14_env gfortran.
  DLL resolution: libgfortran-5.dll already in windcube Library\bin; sitecustomize.py added to
  windcube Lib\ to call os.add_dll_directory(Library\bin) at startup.
  pyhwm2014 made importable via pyhwm2014_dev.pth in windcube site-packages.
  T3 quiet-time equator: U=-80.9 m/s V=-32.1 m/s PASS.
  T4 storm 60N: U=+111.5 m/s V=-62.2 m/s PASS.
  GEN01 wind map options 4 (HWM14 quiet) and 5 (HWM14 storm) operational in windcube.

- pyHWM14 submodule (2026-05-16): tracked as git submodule at 500d7dd; outer repo pointer unchanged.
  Local working-tree changes (dirty) are WIP inside the submodule — no outer-repo commit required.

- GEN01 HWM14 end-to-end integration (2026-05-16): wind map option 4 (quiet-time, 250 km) verified on
  GEN01-V2/GEN01_20270101_001.0d_hwm14_seed0042.csv (8641 rows). v_wind_los_approach_ms:
  mean=+3.0, std=56.9, min=-126.6, max=+142.1 m/s — SPATIALLY VARYING: PASS (std >> 5 m/s threshold).
  Wind columns present: wind_v_zonal_ms, wind_v_merid_ms, v_wind_los_approach_ms.
  matplotlib savefig outside conda-activated env: native LoadLibrary calls bypass os.add_dll_directory;
  fix is to prepend conda env dirs to os.environ["PATH"] before importing matplotlib.

- windcube env GEN01 dependencies (2026-05-16): all third-party packages verified present.
  numpy 2.4.5, pandas 3.0.3, scipy 1.17.1, astropy 7.2.0, sgp4 2.25, matplotlib 3.10.9,
  cartopy 0.25.0 (conda-forge, new), netCDF4 1.7.4 (conda-forge, new), pyhwm2014 0.0.0,
  joblib (already present), lmfit (already present), tkinter (stdlib).
  GEN01 import-only check: PASS (2026-05-16).
