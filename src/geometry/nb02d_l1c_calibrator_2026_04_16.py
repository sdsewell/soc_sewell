"""
NB02d — L1c calibrator: compose and remove spacecraft velocity terms.

Spec:      NB02_geometry_2026-04-16.md
Spec date: 2026-04-16
Generated: 2026-04-16
Tool:      Claude Code

compose_v_rel()              — forward model used by M04 (synthetic images).
remove_spacecraft_velocity() — L1c inverse, feeds v_wind_LOS to M07.
Round-trip accuracy: < 1e-10 m/s (machine precision).
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
    This is the forward model used in the synthetic image generator (M04)
    to embed a known wind signal. The inverse is remove_spacecraft_velocity().

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
    v_rel from the fringe pattern. The recovered v_wind_LOS is then passed
    to M07 for WLS wind vector retrieval.

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
