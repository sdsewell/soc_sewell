"""
Tests for M07 L2 vector wind retrieval.
Spec: specs/S16_m07_wind_retrieval_2026-04-06.md
"""

import numpy as np
import pytest
from astropy.time import Time

from src.fpi.m07_wind_retrieval_2026_04_06 import (
    WindObservation,
    WindResult,
    WindResultFlags,
    compute_sensitivity_coefficients,
    pair_observations,
    retrieve_wind_vectors,
)
from src.geometry.nb02c_los_projection_2026_04_06 import enu_unit_vectors_eci


# ---------------------------------------------------------------------------
# Private test helpers
# ---------------------------------------------------------------------------

_EPOCH_BASE = Time("2027-01-01T00:00:00", scale="utc")
_DEP_RAD    = np.radians(15.73)
_COS_D      = np.cos(_DEP_RAD)
_SIN_D      = np.sin(_DEP_RAD)


def _build_synthetic_obs_pair(
    v_zonal_truth: float,
    v_merid_truth: float,
    lat_deg: float = 30.0,
    lon_deg: float = 0.0,
    sigma: float = 9.8,
    look_modes: tuple = ("along_track", "cross_track"),
    delta_t_s: float = 5640.0,
) -> tuple:
    """
    Build a synthetic (along_track, cross_track) observation pair at the
    given lat/lon with known wind truth.

    Uses an idealised orbit geometry where:
        along_track  LOS = cos(δ) * ê_north − sin(δ) * ê_up
        cross_track  LOS = cos(δ) * ê_east  − sin(δ) * ê_up

    This gives a well-conditioned 2×2 matrix at all latitudes:
        A = [[A_e_at ≈ 0,      A_n_at ≈ cos(δ)],
             [A_e_ct ≈ cos(δ), A_n_ct ≈ 0     ]]
    condition number ≈ 1.

    v_wind_LOS is computed directly from the truth wind projection —
    no FPI fringe simulation required.
    """
    epochs = [_EPOCH_BASE, Time(_EPOCH_BASE.unix + delta_t_s, format="unix", scale="utc")]
    obs_list = []

    for mode, epoch in zip(look_modes, epochs):
        e_east, e_north, e_up = enu_unit_vectors_eci(lat_deg, lon_deg, epoch)

        if mode == "along_track":
            los_eci = _COS_D * e_north - _SIN_D * e_up
        else:  # cross_track
            los_eci = _COS_D * e_east  - _SIN_D * e_up
        los_eci = los_eci / np.linalg.norm(los_eci)

        # Project truth wind onto LOS
        A_e = float(np.dot(e_east,  los_eci))
        A_n = float(np.dot(e_north, los_eci))
        v_wind_LOS = v_zonal_truth * A_e + v_merid_truth * A_n

        obs_list.append(WindObservation(
            epoch_utc       = float(epoch.unix),
            look_mode       = mode,
            tp_lat_deg      = float(lat_deg),
            tp_lon_deg      = float(lon_deg),
            tp_alt_km       = 250.0,
            v_rel_ms        = 0.0,
            sigma_v_rel_ms  = float(sigma),
            V_sc_LOS        = 0.0,
            v_earth_LOS     = 0.0,
            v_wind_LOS      = v_wind_LOS,
            los_eci         = los_eci,
            e_east_eci      = e_east,
            e_north_eci     = e_north,
            m06_quality_flags = 0,
        ))

    return obs_list[0], obs_list[1]


def _make_obs(
    look_mode: str,
    epoch: float,
    lat: float,
    lon: float = 0.0,
    v_wind_LOS: float = 0.0,
    sigma: float = 9.8,
) -> WindObservation:
    """
    Create a minimal WindObservation with specified pairing fields.
    Used by T6 and T7 pairing-window tests.
    """
    t = _EPOCH_BASE
    e_east, e_north, e_up = enu_unit_vectors_eci(lat, lon, t)

    if look_mode == "along_track":
        los = _COS_D * e_north - _SIN_D * e_up
    else:
        los = _COS_D * e_east  - _SIN_D * e_up
    los = los / np.linalg.norm(los)

    return WindObservation(
        epoch_utc       = float(epoch),
        look_mode       = look_mode,
        tp_lat_deg      = float(lat),
        tp_lon_deg      = float(lon),
        tp_alt_km       = 250.0,
        v_rel_ms        = 0.0,
        sigma_v_rel_ms  = float(sigma),
        V_sc_LOS        = 0.0,
        v_earth_LOS     = 0.0,
        v_wind_LOS      = v_wind_LOS,
        los_eci         = los,
        e_east_eci      = e_east,
        e_north_eci     = e_north,
        m06_quality_flags = 0,
    )


def _make_synthetic_wind_result() -> WindResult:
    """Build a valid WindResult from a synthetic pair. Used by T1."""
    obs_at, obs_ct = _build_synthetic_obs_pair(100.0, 50.0)
    results = retrieve_wind_vectors([obs_at, obs_ct])
    assert len(results) == 1, "Expected exactly 1 result from synthetic pair"
    return results[0]


# ---------------------------------------------------------------------------
# T1 — S04 compliance: two_sigma fields
# ---------------------------------------------------------------------------

def test_two_sigma_convention():
    """All two_sigma_ fields must equal exactly 2.0 × sigma_."""
    result = _make_synthetic_wind_result()
    assert abs(result.two_sigma_v_zonal_ms      - 2.0 * result.sigma_v_zonal_ms)      < 1e-15, \
        f"two_sigma_v_zonal = {result.two_sigma_v_zonal_ms} ≠ 2 × {result.sigma_v_zonal_ms}"
    assert abs(result.two_sigma_v_meridional_ms - 2.0 * result.sigma_v_meridional_ms) < 1e-15, \
        f"two_sigma_v_merid = {result.two_sigma_v_meridional_ms} ≠ 2 × {result.sigma_v_meridional_ms}"


# ---------------------------------------------------------------------------
# T2 — Uniform wind round-trip: known v_zonal and v_merid recovered
# ---------------------------------------------------------------------------

def test_uniform_wind_round_trip():
    """
    Inject known uniform wind field. M07 must recover v_zonal and v_meridional
    to within 1 m/s (geometric round-trip, no noise).
    """
    v_zonal_truth = 100.0   # m/s eastward
    v_merid_truth =  50.0   # m/s northward

    obs_at, obs_ct = _build_synthetic_obs_pair(
        v_zonal_truth, v_merid_truth, lat_deg=30.0,
    )
    results = retrieve_wind_vectors([obs_at, obs_ct])

    assert len(results) == 1, f"Expected 1 result, got {len(results)}"
    r = results[0]
    assert abs(r.v_zonal_ms      - v_zonal_truth) < 1.0, \
        f"v_zonal error = {r.v_zonal_ms - v_zonal_truth:.4f} m/s"
    assert abs(r.v_meridional_ms - v_merid_truth) < 1.0, \
        f"v_merid error = {r.v_meridional_ms - v_merid_truth:.4f} m/s"
    assert r.quality_flags == WindResultFlags.GOOD, \
        f"Expected GOOD flags, got 0x{r.quality_flags:x}"


# ---------------------------------------------------------------------------
# T3 — Zero wind: recovered components near zero
# ---------------------------------------------------------------------------

def test_zero_wind():
    """Zero wind input must give near-zero output (< 1 m/s)."""
    obs_at, obs_ct = _build_synthetic_obs_pair(0.0, 0.0, lat_deg=15.0)
    results = retrieve_wind_vectors([obs_at, obs_ct])
    assert len(results) == 1
    r = results[0]
    assert abs(r.v_zonal_ms)      < 1.0, f"|v_zonal| = {abs(r.v_zonal_ms):.4f} m/s"
    assert abs(r.v_meridional_ms) < 1.0, f"|v_merid| = {abs(r.v_meridional_ms):.4f} m/s"


# ---------------------------------------------------------------------------
# T4 — Uncertainty propagation: sigma scales with input sigma
# ---------------------------------------------------------------------------

def test_uncertainty_propagation():
    """
    Doubling sigma_v_rel on both observations must approximately double
    sigma_v_zonal and sigma_v_meridional (ratio in [1.5, 2.5]).
    """
    obs1_at, obs1_ct = _build_synthetic_obs_pair(100.0, 50.0, sigma=5.0)
    obs2_at, obs2_ct = _build_synthetic_obs_pair(100.0, 50.0, sigma=10.0)

    r1 = retrieve_wind_vectors([obs1_at, obs1_ct])[0]
    r2 = retrieve_wind_vectors([obs2_at, obs2_ct])[0]

    ratio_z = r2.sigma_v_zonal_ms   / r1.sigma_v_zonal_ms
    ratio_m = r2.sigma_v_meridional_ms / r1.sigma_v_meridional_ms
    assert 1.5 < ratio_z < 2.5, f"sigma_v_zonal ratio = {ratio_z:.3f}; expected in [1.5, 2.5]"
    assert 1.5 < ratio_m < 2.5, f"sigma_v_merid ratio = {ratio_m:.3f}; expected in [1.5, 2.5]"


# ---------------------------------------------------------------------------
# T5 — Condition number flag
# ---------------------------------------------------------------------------

def test_ill_conditioned_flagged():
    """
    Two along-track observations — pairing algorithm produces no pairs
    (requires AT + CT), so len(results) == 0 or the result is ILL_CONDITIONED.
    """
    obs1, obs2 = _build_synthetic_obs_pair(
        100.0, 50.0, look_modes=("along_track", "along_track")
    )
    results = retrieve_wind_vectors([obs1, obs2])
    # Two AT obs cannot be paired — expect 0 results
    if len(results) > 0:
        assert results[0].quality_flags & WindResultFlags.ILL_CONDITIONED, \
            "Parallel LOS pair must be flagged ILL_CONDITIONED"


# ---------------------------------------------------------------------------
# T6 — Pairing: respects max_delta_t_s
# ---------------------------------------------------------------------------

def test_pairing_time_window():
    """Observations separated by > max_delta_t_s must not be paired."""
    obs_at = _make_obs(look_mode="along_track", epoch=0.0,    lat=30.0)
    obs_ct = _make_obs(look_mode="cross_track", epoch=8000.0, lat=30.0)
    results = retrieve_wind_vectors([obs_at, obs_ct], max_delta_t_s=7000.0)
    assert len(results) == 0, \
        f"Observations 8000 s apart must not be paired (max=7000 s), got {len(results)}"


# ---------------------------------------------------------------------------
# T7 — Pairing: respects lat_bin_deg
# ---------------------------------------------------------------------------

def test_pairing_latitude_window():
    """Observations separated by > lat_bin_deg must not be paired."""
    obs_at = _make_obs(look_mode="along_track", epoch=0.0,    lat=30.0)
    obs_ct = _make_obs(look_mode="cross_track", epoch=5640.0, lat=33.0)
    results = retrieve_wind_vectors([obs_at, obs_ct], lat_bin_deg=2.0)
    assert len(results) == 0, \
        f"Observations 3° apart must not be paired (lat_bin=2°), got {len(results)}"


# ---------------------------------------------------------------------------
# T8 — Sign convention: positive v_zonal = eastward
# ---------------------------------------------------------------------------

def test_wind_sign_convention():
    """
    Positive v_zonal must correspond to eastward wind.
    Verify using sensitivity coefficients and the recovered wind.
    """
    obs_at, obs_ct = _build_synthetic_obs_pair(100.0, 0.0, lat_deg=0.0)

    A_e_at, A_n_at = compute_sensitivity_coefficients(obs_at)
    A_e_ct, A_n_ct = compute_sensitivity_coefficients(obs_ct)

    # Cross-track should have larger east sensitivity than along-track
    assert abs(A_e_ct) > abs(A_e_at), \
        (f"Cross-track east sensitivity |A_e_ct|={abs(A_e_ct):.4f} must exceed "
         f"along-track |A_e_at|={abs(A_e_at):.4f}")

    results = retrieve_wind_vectors([obs_at, obs_ct])
    assert len(results) == 1
    assert results[0].v_zonal_ms > 0, \
        f"Eastward 100 m/s wind must give positive v_zonal, got {results[0].v_zonal_ms:.2f}"
