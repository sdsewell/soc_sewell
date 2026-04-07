"""
M07 — L2 vector wind retrieval: decomposes LOS winds into horizontal components.

Spec:        docs/specs/S16_m07_wind_retrieval_2026-04-06.md
Spec date:   2026-04-06
Generated:   2026-04-07
Tool:        Claude Code
Last tested: 2026-04-07  (8/8 tests pass)
Depends on:  src.constants, numpy
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from src.constants import LAT_RANGE_DEG, SC_ORBITAL_PERIOD_S


# ---------------------------------------------------------------------------
# Quality flags
# ---------------------------------------------------------------------------


class WindResultFlags:
    """Bitmask quality flags for WindResult. Uses bits 4+ per S04."""
    GOOD             = 0x00
    ILL_CONDITIONED  = 0x10   # condition_number > 100
    LARGE_DELTA_T    = 0x20   # |delta_t_s| > 7000 s (> ~1.2 orbits)
    AT_OBS_DEGRADED  = 0x40   # along-track M06 flags non-GOOD
    CT_OBS_DEGRADED  = 0x80   # cross-track M06 flags non-GOOD
    OUT_OF_LAT_BAND  = 0x100  # |lat| > LAT_RANGE_DEG[1]


# ---------------------------------------------------------------------------
# Input data structure
# ---------------------------------------------------------------------------


@dataclass
class WindObservation:
    """
    A single FPI wind observation — one epoch, one look mode.
    Constructed from M06 output + NB02c output for the same epoch.
    """
    epoch_utc:       float      # Unix timestamp, seconds
    look_mode:       str        # 'along_track' or 'cross_track'
    tp_lat_deg:      float      # tangent point geodetic latitude, deg
    tp_lon_deg:      float      # tangent point geodetic longitude, deg
    tp_alt_km:       float      # tangent point geodetic altitude, km

    # From M06 AirglowFitResult
    v_rel_ms:        float      # LOS Doppler observable, m/s
    sigma_v_rel_ms:  float      # 1σ uncertainty on v_rel, m/s

    # From NB02c compute_v_rel() dict
    V_sc_LOS:        float      # spacecraft velocity LOS projection, m/s
    v_earth_LOS:     float      # Earth rotation LOS projection, m/s
    v_wind_LOS:      float      # atmospheric wind LOS projection, m/s
                                # = v_rel + V_sc_LOS + v_earth_LOS

    # From NB02c enu_unit_vectors_eci() and compute_los_eci()
    los_eci:         np.ndarray  # shape (3,), unit LOS vector in ECI
    e_east_eci:      np.ndarray  # shape (3,), unit East vector in ECI
    e_north_eci:     np.ndarray  # shape (3,), unit North vector in ECI

    # Quality
    m06_quality_flags: int       # from AirglowFitResult.quality_flags


# ---------------------------------------------------------------------------
# Output data structure
# ---------------------------------------------------------------------------


@dataclass
class WindResult:
    """
    L2 horizontal wind vector at one geographic location.
    Produced by M07 from one along-track + one cross-track pair.
    Per S04: every fitted quantity has sigma_ and two_sigma_ fields.
    """
    # Geolocation (midpoint of the two tangent points)
    lat_deg:    float
    lon_deg:    float
    alt_km:     float

    # Wind components — positive = eastward / northward (S02 Section 6.6)
    v_zonal_ms:             float
    sigma_v_zonal_ms:       float
    two_sigma_v_zonal_ms:   float   # exactly 2 × sigma_v_zonal_ms  (S04)

    v_meridional_ms:            float
    sigma_v_meridional_ms:      float
    two_sigma_v_meridional_ms:  float   # exactly 2 × sigma_v_meridional_ms

    # Timing
    epoch_at_utc:  float   # Unix timestamp of along-track observation
    epoch_ct_utc:  float   # Unix timestamp of cross-track observation
    delta_t_s:     float   # time separation (ct - at), seconds

    # Geometry diagnostics
    condition_number:  float
    A_e_at:  float   # east sensitivity, along-track
    A_n_at:  float   # north sensitivity, along-track
    A_e_ct:  float   # east sensitivity, cross-track
    A_n_ct:  float   # north sensitivity, cross-track

    # Input observations used
    v_wind_LOS_at:  float
    v_wind_LOS_ct:  float

    # Quality
    n_obs_at:    int
    n_obs_ct:    int
    quality_flags:  int


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def compute_sensitivity_coefficients(
    obs: WindObservation,
) -> tuple[float, float]:
    """
    Compute the east and north sensitivity coefficients for one observation.

    A_e = ê_east  · los_eci   (east  component of LOS projection)
    A_n = ê_north · los_eci   (north component of LOS projection)

    Parameters
    ----------
    obs : WindObservation
        Must have los_eci, e_east_eci, e_north_eci populated.

    Returns
    -------
    (A_e, A_n) : tuple[float, float]
        Dimensionless numbers in (-1, +1).
    """
    A_e = float(np.dot(obs.e_east_eci,  obs.los_eci))
    A_n = float(np.dot(obs.e_north_eci, obs.los_eci))
    return A_e, A_n


def pair_observations(
    observations: list,
    lat_bin_deg: float = 2.0,
    max_delta_t_s: float = 7000.0,
    lat_range_deg: tuple = LAT_RANGE_DEG,
) -> list:
    """
    Match along-track observations to cross-track observations at
    approximately the same geographic location.

    Algorithm
    ---------
    1. Separate observations into along-track and cross-track lists.
    2. For each along-track observation within lat_range_deg:
       a. Find all cross-track observations with |delta_lat| < lat_bin_deg/2
          AND |delta_t| < max_delta_t_s.
       b. Among candidates, select the one with the smallest |delta_lat|.
       c. If no candidate found, skip this along-track observation.
    3. Each along-track observation may appear in at most one pair.
       Each cross-track observation may appear in at most one pair.
       (Nearest-neighbour matching, no reuse.)

    Parameters
    ----------
    observations : list[WindObservation]
    lat_bin_deg  : float — maximum latitude separation, degrees. Default 2.0.
    max_delta_t_s: float — maximum time separation, seconds. Default 7000 s.
    lat_range_deg: tuple — (min_lat, max_lat) science latitude band.

    Returns
    -------
    list of (along_track_obs, cross_track_obs) tuples
    """
    at_obs = [o for o in observations if o.look_mode == "along_track"]
    ct_obs = [o for o in observations if o.look_mode == "cross_track"]

    half_bin = lat_bin_deg / 2.0
    used_ct: set[int] = set()
    pairs = []

    for obs_at in at_obs:
        # Only pair observations within the science latitude band
        if not (lat_range_deg[0] <= obs_at.tp_lat_deg <= lat_range_deg[1]):
            continue

        best_ct   = None
        best_dlat = float("inf")
        best_idx  = -1

        for j, obs_ct in enumerate(ct_obs):
            if j in used_ct:
                continue
            dlat = abs(obs_at.tp_lat_deg - obs_ct.tp_lat_deg)
            dt   = abs(obs_at.epoch_utc  - obs_ct.epoch_utc)
            if dlat < half_bin and dt < max_delta_t_s:
                if dlat < best_dlat:
                    best_dlat = dlat
                    best_ct   = obs_ct
                    best_idx  = j

        if best_ct is not None:
            pairs.append((obs_at, best_ct))
            used_ct.add(best_idx)

    return pairs


def retrieve_wind_vectors(
    observations: list,
    lat_bin_deg: float = 2.0,
    max_delta_t_s: float = 7000.0,
    max_condition_number: float = 100.0,
    lat_range_deg: tuple = LAT_RANGE_DEG,
) -> list:
    """
    Decompose paired LOS wind observations into horizontal wind vectors.

    For each matched (along-track, cross-track) pair:
    1. Compute sensitivity coefficients A_e and A_n for each observation.
    2. Build the 2×2 matrix A.
    3. Check condition number — flag ILL_CONDITIONED if > max_condition_number.
    4. Solve the 2×2 system for [v_zonal, v_merid].
    5. Propagate uncertainties through the matrix inverse.
    6. Assign geolocation as the midpoint of the two tangent points.
    7. Package into a WindResult with quality flags.

    Parameters
    ----------
    observations       : list[WindObservation]
    lat_bin_deg        : float — latitude matching window, degrees. Default 2.0.
    max_delta_t_s      : float — maximum time separation, seconds. Default 7000.
    max_condition_number: float — matrix condition number threshold. Default 100.
    lat_range_deg      : tuple — science latitude band. Default from S03.

    Returns
    -------
    list[WindResult]
        Sorted by epoch_at_utc. Empty if no pairs found.
        ILL_CONDITIONED pairs are included but flagged.
    """
    pairs = pair_observations(observations, lat_bin_deg, max_delta_t_s, lat_range_deg)

    results = []
    for obs_at, obs_ct in pairs:
        # --- Sensitivity coefficients ---
        A_e_at, A_n_at = compute_sensitivity_coefficients(obs_at)
        A_e_ct, A_n_ct = compute_sensitivity_coefficients(obs_ct)

        # --- 2×2 system ---
        A_mat   = np.array([[A_e_at, A_n_at],
                             [A_e_ct, A_n_ct]], dtype=float)
        b_vec   = np.array([float(obs_at.v_wind_LOS),
                             float(obs_ct.v_wind_LOS)], dtype=float)
        sigma_b = np.array([float(obs_at.sigma_v_rel_ms),
                             float(obs_ct.sigma_v_rel_ms)], dtype=float)

        # --- Condition number ---
        cond = float(np.linalg.cond(A_mat))
        flags = WindResultFlags.GOOD
        if cond > max_condition_number:
            flags |= WindResultFlags.ILL_CONDITIONED

        # --- Solve ---
        try:
            sol = np.linalg.solve(A_mat, b_vec)
            v_zonal = float(sol[0])
            v_merid = float(sol[1])
        except np.linalg.LinAlgError:
            flags   |= WindResultFlags.ILL_CONDITIONED
            v_zonal  = float("nan")
            v_merid  = float("nan")

        # --- Uncertainty propagation ---
        try:
            A_inv = np.linalg.inv(A_mat)
            cov   = A_inv @ np.diag(sigma_b ** 2) @ A_inv.T
            sigma_zonal = float(np.sqrt(max(0.0, cov[0, 0])))
            sigma_merid = float(np.sqrt(max(0.0, cov[1, 1])))
        except np.linalg.LinAlgError:
            sigma_zonal = float("nan")
            sigma_merid = float("nan")

        # --- S04 two_sigma convention: set immediately after sigma ---
        two_sigma_zonal = 2.0 * sigma_zonal
        two_sigma_merid = 2.0 * sigma_merid

        # --- Quality flags ---
        delta_t = float(obs_ct.epoch_utc - obs_at.epoch_utc)
        if abs(delta_t) > max_delta_t_s:
            flags |= WindResultFlags.LARGE_DELTA_T
        if obs_at.m06_quality_flags != WindResultFlags.GOOD:
            flags |= WindResultFlags.AT_OBS_DEGRADED
        if obs_ct.m06_quality_flags != WindResultFlags.GOOD:
            flags |= WindResultFlags.CT_OBS_DEGRADED

        # --- Geolocation: midpoint ---
        lat_mid = (float(obs_at.tp_lat_deg) + float(obs_ct.tp_lat_deg)) / 2.0
        # Longitude midpoint with wrap-around handling
        lon_at = float(obs_at.tp_lon_deg)
        lon_ct = float(obs_ct.tp_lon_deg)
        dlon   = lon_ct - lon_at
        if dlon >  180.0:
            dlon -= 360.0
        elif dlon < -180.0:
            dlon += 360.0
        lon_mid = ((lon_at + dlon / 2.0) + 180.0) % 360.0 - 180.0

        alt_mid = (float(obs_at.tp_alt_km) + float(obs_ct.tp_alt_km)) / 2.0

        if abs(lat_mid) > lat_range_deg[1]:
            flags |= WindResultFlags.OUT_OF_LAT_BAND

        results.append(WindResult(
            lat_deg   = lat_mid,
            lon_deg   = lon_mid,
            alt_km    = alt_mid,
            v_zonal_ms           = v_zonal,
            sigma_v_zonal_ms     = sigma_zonal,
            two_sigma_v_zonal_ms = two_sigma_zonal,
            v_meridional_ms              = v_merid,
            sigma_v_meridional_ms        = sigma_merid,
            two_sigma_v_meridional_ms    = two_sigma_merid,
            epoch_at_utc = float(obs_at.epoch_utc),
            epoch_ct_utc = float(obs_ct.epoch_utc),
            delta_t_s    = delta_t,
            condition_number = cond,
            A_e_at = A_e_at,
            A_n_at = A_n_at,
            A_e_ct = A_e_ct,
            A_n_ct = A_n_ct,
            v_wind_LOS_at = float(obs_at.v_wind_LOS),
            v_wind_LOS_ct = float(obs_ct.v_wind_LOS),
            n_obs_at      = 1,
            n_obs_ct      = 1,
            quality_flags = flags,
        ))

    results.sort(key=lambda r: r.epoch_at_utc)
    return results
