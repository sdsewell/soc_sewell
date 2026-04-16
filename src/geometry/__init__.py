# geometry/__init__.py — Geometry and simulation chain subpackage
# Implements: NB00 (S05), NB01 (S06), NB02 (S07)
#
# Re-export from current dated implementation files per S01 Section 10.
# Add one line per module as each is implemented, e.g.:
#   from geometry.nb00_wind_map_2026_XX_XX import *  # noqa: F401, F403
from src.geometry.nb01_orbit_propagator_2026_04_16 import propagate_orbit  # noqa: F401
from src.geometry.nb02a_boresight_2026_04_06 import (  # noqa: F401
    compute_synthetic_quaternion, compute_los_eci,
)
from src.geometry.nb02b_tangent_point_2026_04_06 import compute_tangent_point  # noqa: F401
from src.geometry.nb02c_los_projection_2026_04_06 import (  # noqa: F401
    enu_unit_vectors_eci, earth_rotation_velocity_eci, compute_v_rel,
)
from src.geometry.nb02d_l1c_calibrator_2026_04_06 import (  # noqa: F401
    compose_v_rel, remove_spacecraft_velocity,
)
