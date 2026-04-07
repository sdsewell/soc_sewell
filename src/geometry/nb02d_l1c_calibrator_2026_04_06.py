"""
NB02d — L1c calibrator: compose and remove spacecraft velocity from v_rel.

Spec:        S07_nb02_geometry_2026-04-05.md  Section 4.4
Spec date:   2026-04-05
Generated:   2026-04-07
Depends on:  nothing (pure arithmetic)
"""

from __future__ import annotations


def compose_v_rel(
    v_wind_LOS: float,
    V_sc_LOS: float,
    v_earth_LOS: float,
) -> float:
    """
    Forward composition: combine the three LOS velocity terms into v_rel.

    v_rel = v_wind_LOS - V_sc_LOS - v_earth_LOS

    Used in the synthetic forward pipeline to build the ground truth.

    Parameters
    ----------
    v_wind_LOS  : float, m/s — atmospheric wind projected onto LOS
    V_sc_LOS    : float, m/s — spacecraft velocity projected onto LOS
    v_earth_LOS : float, m/s — Earth rotation velocity projected onto LOS

    Returns
    -------
    float : v_rel, m/s
    """
    return v_wind_LOS - V_sc_LOS - v_earth_LOS


def remove_spacecraft_velocity(
    v_rel: float,
    V_sc_LOS: float,
    v_earth_LOS: float,
) -> float:
    """
    Inverse (L1c step): recover v_wind_LOS from the measured v_rel.

    v_wind_LOS = v_rel + V_sc_LOS + v_earth_LOS

    Applied to real or simulated FPI measurements after M06 has recovered
    v_rel from the fringe pattern.  This is the L1c calibration step.

    Round-trip accuracy requirement: < 1e-10 m/s (machine precision).
    If this test fails, there is a sign error — not a numerical issue.

    Parameters
    ----------
    v_rel       : float, m/s — Doppler observable recovered from fringe fit
    V_sc_LOS    : float, m/s — spacecraft velocity projected onto LOS
    v_earth_LOS : float, m/s — Earth rotation velocity projected onto LOS

    Returns
    -------
    float : v_wind_LOS, m/s
    """
    return v_rel + V_sc_LOS + v_earth_LOS
