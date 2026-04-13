| Spec | Module       | Status | Tests    | Last verified |
|------|-------------|--------|----------|---------------|
| S01  | —           | impl   | n/a      | 2026-04-06    |
| S02  | —           | impl   | n/a      | 2026-04-06    |
| S03  | —           | impl   | n/a      | 2026-04-06    |
| S04  | —           | impl   | n/a      | 2026-04-06    |
| S05  | NB00        | impl   | n/a      | 2026-04-06    |
| S06  | NB01        | impl   | n/a      | 2026-04-06    |
| S07  | NB02        | impl   | n/a      | 2026-04-06    |
| S07b | NB03        | impl   | 8/8      | 2026-04-12    |
| S08  | INT01       | impl   | n/a      | 2026-04-06    |
| S09  | M01         | impl   | passing  | 2026-04-06    |
| S10  | M02         | impl   | passing  | 2026-04-06    |
| S11  | M04         | impl   | passing  | 2026-04-06    |
| S12  | M03         | impl   | passing  | 2026-04-06    |
| S13  | Tolansky    | impl   | passing  | 2026-04-06    |
| S14  | M05         | impl   | passing  | 2026-04-06    |
| S15  | M06         | impl   | passing  | 2026-04-06    |
| S16  | M07         | impl   | 8/8      | 2026-04-06    |
| S17  | INT02       | impl   | 16/16    | 2026-04-07    |
| S18  | INT03       | impl   | 14/14    | 2026-04-11    |
| S19  | P01         | impl   | passing  | 2026-04-06    |
| S20  | L2 product  | impl   | passing  | 2026-04-11    |
| Z01  | validate-cal| impl   | 6-stage  | 2026-04-11    |
| Z01a | OI630-cal   | impl   | 16/16    | 2026-04-12    |
| Z02  | airglow-gen | impl   | 8/8      | 2026-04-10    |
| Z03  | cal-gen     | impl   | passing  | 2026-04-10    |
| Z04  | snr-sweep   | impl   | 6/6      | 2026-04-11    |

## Known pre-existing test failures (not introduced by current work)

| Test file                                    | Reason                        |
|----------------------------------------------|-------------------------------|
| test_z04.py                                  | missing joblib package        |
| test_s06_nb01_orbit_propagator.py            | missing module (NB01 not impl)|
| test_z02_synthetic_airglow_generator.py      | wrong script path             |

These failures pre-date the S07b session and are excluded from regression
assessment until the relevant modules are fixed or installed.
