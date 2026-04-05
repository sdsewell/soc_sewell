# NB01 Orbit Propagator — User Guide

**Module:** src/geometry/nb01_orbit_propagator_2026_04_05.py  
**Spec:** specs/S06_nb01_orbit_propagator_2026-04-05.md

---

## Overview
This module propagates the WindCube spacecraft orbit using SGP4, producing a time-series of ECI (Earth-Centered Inertial) position and velocity vectors, as well as WGS84 geodetic coordinates, at a configurable cadence. All constants are imported from src/constants for reproducibility.

## Main Function
### propagate_orbit
```
def propagate_orbit(
    t_start: str,
    duration_s: float,
    dt_s: float = SCIENCE_CADENCE_S,
    altitude_km: float = SC_ALTITUDE_KM,
    inclination_deg: float = 97.44,
    raan_deg: float = 90.0,
    bstar: float = 0.0,
) -> pd.DataFrame:
```
- **t_start**: Start epoch in ISO 8601 UTC format (e.g., '2027-01-01T00:00:00')
- **duration_s**: Total propagation duration in seconds
- **dt_s**: Time step between output epochs (default: 10.0 s)
- **altitude_km**: Circular orbit altitude above WGS84 ellipsoid (default: 510.0 km)
- **inclination_deg**: Orbital inclination (default: 97.44°)
- **raan_deg**: Right ascension of ascending node at epoch (default: 90.0°)
- **bstar**: SGP4 drag coefficient (default: 0.0)

**Returns:**
- A pandas DataFrame with one row per epoch and columns:
    - `epoch`, `pos_eci_x`, `pos_eci_y`, `pos_eci_z`, `vel_eci_x`, `vel_eci_y`, `vel_eci_z`, `lat_deg`, `lon_deg`, `alt_km`, `speed_ms`

## Example Usage
```python
from src.geometry.nb01_orbit_propagator_2026_04_05 import propagate_orbit

df = propagate_orbit('2027-01-01T00:00:00', duration_s=600)
print(df.head())
```

## Notes
- All physical constants are imported from src/constants; do not hardcode values.
- Output positions are in ECI J2000 (metres); SGP4 outputs in TEME, but the difference is negligible for this application.
- The DataFrame index is integer-based for easy row lookup.
- Defensive assertions catch unit errors and SGP4 propagation failures.

## Dependencies
- sgp4 >= 2.21
- astropy >= 5.0
- numpy >= 1.24
- pandas >= 2.0

## References
- [S06_nb01_orbit_propagator_2026-04-05.md](../../specs/S06_nb01_orbit_propagator_2026-04-05.md)
- [SGP4 Python Documentation](https://pypi.org/project/sgp4/)
- [Astropy Documentation](https://docs.astropy.org/)
