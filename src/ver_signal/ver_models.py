"""
Signal (photon flux) calculator for WindCube FPI.

Plots dayglow (Meneses model + Solomon & Abreu 1989 model) and nightglow
(Meneses model) volume emission rate profiles for OI 630.0 nm.

All emission arrays use the column convention:
    col 0 : Volume Emission Rate  [cm^-3 s^-1]
    col 1 : Altitude               [km]

Conversion to plot units (m^-3 s^-1 sr^-1):
    VER_plot = VER_cm * 1e6 / (4 * pi) / 10
    (the /10 matches the original Meneses scaling; applied identically to
    Solomon data so the two dayglow datasets share the same vertical scale)

CHANGES (2026-04-15 v2):
  - Added Solomon & Abreu (1989), JGR 94 A6, Fig 1 dayglow model profiles
    for SZA = 30 deg, 50 deg, 70 deg, digitised from the scanned figure.
    Values read from a log10 x-axis (1-1000 cm^-3 s^-1) and linear y-axis
    (150-400 km); estimated digitisation uncertainty approx +/-5-10%.
  - Meneses dayglow and nightglow unchanged from v1.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import astropy.constants as C
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator, CubicSpline

logging.basicConfig(format="%(levelname)s: %(asctime)s %(message)s")
logging.getLogger().setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Mission parameters
# ---------------------------------------------------------------------------

R_orbit = C.R_earth.value + 500_000.0
observation_height = 250_000.0
pitch_angle = np.arccos((C.R_earth.value + observation_height) / R_orbit)
d_view = R_orbit * np.sin(pitch_angle)
aperture = 0.06 * 0.03
field_of_view = np.radians(2.35)
pixels = 512
pixel_solid_angle = (field_of_view / pixels) ** 2


# ---------------------------------------------------------------------------
# Meneses dayglow emission model  [VER cm^-3 s^-1, altitude km]
# Rows ordered high->low altitude.
# ---------------------------------------------------------------------------

MenesesDayglow = np.array([
    [0.0,        1000.0],
    [0.0,         900.0],
    [0.0,         800.0],
    [0.0,         700.0],
    [0.0,         600.0],
    [0.0,         500.0],
    [2.84161,   399.17853],
    [3.13098,   395.07119],
    [3.52790,   390.14239],
    [3.88716,   386.30887],
    [4.31508,   381.92771],
    [4.86211,   377.27273],
    [5.64443,   371.79628],
    [6.40760,   366.86747],
    [7.21990,   362.48631],
    [8.07472,   358.37897],
    [9.58609,   352.08105],
    [10.88221,  347.97371],
    [12.44607,  343.04491],
    [14.44865,  337.84228],
    [17.15305,  332.09200],
    [19.91297,  327.16320],
    [24.53855,  319.76999],
    [28.70010,  314.56736],
    [32.82454,  308.81709],
    [38.96842,  301.97152],
    [43.90850,  297.59036],
    [52.51730,  290.74480],
    [59.61808,  284.99452],
    [69.72885,  277.87514],
    [80.94823,  271.02957],
    [92.58117,  264.18401],
    [106.67871, 257.06462],
    [121.10257, 249.94524],
    [143.76971, 241.73056],
    [160.79165, 234.61117],
    [182.53203, 224.47974],
    [196.66927, 215.99124],
    [208.76343, 204.21687],
    [204.14333, 192.99014],
    [186.66303, 184.50164],
    [170.67952, 176.83461],
    [141.64065, 166.70318],
    [122.00934, 160.95290],
    [105.09892, 155.47645],
    [91.89311,  150.54765],
])

MenesesDayglowEmission = PchipInterpolator(
    np.flip(MenesesDayglow[:, 1]) * 1000,                  # altitude [m] ascending
    np.flip(MenesesDayglow[:, 0]) * 100**3 / 4 / np.pi / 10,  # m^-3 s^-1 sr^-1
)


# ---------------------------------------------------------------------------
# Solomon & Abreu (1989) dayglow model  [VER cm^-3 s^-1, altitude km]
#
# Reference:
#   Solomon, S.C. and Abreu, V.J. (1989),
#   "The 630 nm Dayglow", J. Geophys. Res., 94(A6), 6817-6824.
#   doi:10.1029/JA094iA06p06817
#
# Values digitised by hand from Figure 1 (three modelled profiles for
# solar zenith angles 30, 50, 70 deg, computed with MSIS-86 + IRI).
# The x-axis of Fig 1 is log10 scale 1-1000 cm^-3 s^-1; y-axis linear
# 150-400 km.  Estimated reading uncertainty: +/-5-10% in VER.
# Rows ordered high->low altitude (same convention as Meneses array).
# ---------------------------------------------------------------------------

Solomon_SZA30 = np.array([
    #  VER    alt(km)
    [  2.5,   400.0],
    [  3.8,   390.0],
    [  5.8,   380.0],
    [  8.5,   370.0],
    [ 12.5,   360.0],
    [ 18.0,   350.0],
    [ 26.0,   340.0],
    [ 36.0,   330.0],
    [ 50.0,   320.0],
    [ 66.0,   310.0],
    [ 84.0,   300.0],
    [104.0,   290.0],
    [125.0,   280.0],
    [147.0,   270.0],
    [167.0,   260.0],
    [183.0,   250.0],
    [195.0,   240.0],
    [202.0,   230.0],
    [204.0,   222.0],   # peak
    [200.0,   215.0],
    [192.0,   210.0],
    [178.0,   205.0],
    [158.0,   200.0],
    [133.0,   195.0],
    [105.0,   190.0],
    [ 76.0,   185.0],
    [ 51.0,   180.0],
    [ 31.0,   175.0],
    [ 17.0,   170.0],
    [  8.5,   165.0],
    [  3.8,   160.0],
    [  1.5,   155.0],
    [  0.5,   150.0],
])

Solomon_SZA50 = np.array([
    [  1.8,   400.0],
    [  2.8,   390.0],
    [  4.3,   380.0],
    [  6.5,   370.0],
    [  9.5,   360.0],
    [ 14.0,   350.0],
    [ 20.0,   340.0],
    [ 28.0,   330.0],
    [ 39.0,   320.0],
    [ 52.0,   310.0],
    [ 67.0,   300.0],
    [ 84.0,   290.0],
    [103.0,   280.0],
    [121.0,   270.0],
    [140.0,   260.0],
    [156.0,   250.0],
    [168.0,   240.0],
    [175.0,   230.0],
    [177.0,   222.0],   # peak
    [173.0,   215.0],
    [164.0,   210.0],
    [150.0,   205.0],
    [130.0,   200.0],
    [107.0,   195.0],
    [ 82.0,   190.0],
    [ 58.0,   185.0],
    [ 38.0,   180.0],
    [ 23.0,   175.0],
    [ 12.5,   170.0],
    [  6.0,   165.0],
    [  2.5,   160.0],
    [  0.9,   155.0],
    [  0.3,   150.0],
])

Solomon_SZA70 = np.array([
    [  0.9,   400.0],
    [  1.5,   390.0],
    [  2.3,   380.0],
    [  3.5,   370.0],
    [  5.5,   360.0],
    [  8.0,   350.0],
    [ 12.0,   340.0],
    [ 17.0,   330.0],
    [ 24.0,   320.0],
    [ 33.0,   310.0],
    [ 44.0,   300.0],
    [ 56.0,   290.0],
    [ 70.0,   280.0],
    [ 84.0,   270.0],
    [ 97.0,   260.0],
    [110.0,   250.0],
    [120.0,   240.0],
    [127.0,   230.0],
    [130.0,   222.0],   # peak
    [127.0,   215.0],
    [120.0,   210.0],
    [109.0,   205.0],
    [ 94.0,   200.0],
    [ 76.0,   195.0],
    [ 57.0,   190.0],
    [ 40.0,   185.0],
    [ 26.0,   180.0],
    [ 15.5,   175.0],
    [  8.5,   170.0],
    [  4.0,   165.0],
    [  1.6,   160.0],
    [  0.55,  155.0],
    [  0.17,  150.0],
])


def _sol_interp(arr):
    """PchipInterpolator from a Solomon array (rows high->low altitude)."""
    return PchipInterpolator(
        np.flip(arr[:, 1]) * 1000,
        np.flip(arr[:, 0]) * 100**3 / 4 / np.pi / 10,
    )


SolomonEmission_SZA30 = _sol_interp(Solomon_SZA30)
SolomonEmission_SZA50 = _sol_interp(Solomon_SZA50)
SolomonEmission_SZA70 = _sol_interp(Solomon_SZA70)


# ---------------------------------------------------------------------------
# Meneses nightglow emission model  [VER cm^-3 s^-1, altitude km]
# ---------------------------------------------------------------------------

NightglowData = np.array([
    [0,       0],
    [0,       100],
    [0.03,    150],
    [0.07256894049346663,  191.97898423817867],
    [0.1451378809869368,   196.07705779334503],
    [0.362844702467342,    199.85989492119091],
    [0.6531204644412192,   204.58844133099825],
    [1.0159651669085612,   209.31698774080564],
    [1.7416545718432506,   213.41506129597198],
    [2.612481857764875,    216.56742556917692],
    [3.4833091436864994,   219.40455341506132],
    [4.354136429608127,    221.92644483362523],
    [5.224963715529752,    225.07880910683014],
    [6.168359941944848,    226.6549912434326],
    [7.039187227866472,    228.23117338003505],
    [7.837445573294628,    230.122591943958],
    [8.708272859216253,    231.69877408056044],
    [9.57910014513788,     233.2749562171629],
    [10.522496371552974,   234.85113835376535],
    [11.248185776487661,   238.63397548161123],
    [11.393323657474602,   242.4168126094571],
    [10.957910014513788,   247.14535901926448],
    [10.740203193033379,   251.55866900175135],
    [11.32075471698113,    255.34150612959724],
    [11.611030478955007,   259.7548161120841],
    [11.248185776487661,   264.79859894921196],
    [11.393323657474602,   268.58143607705784],
    [11.175616835994193,   272.3642732049037],
    [10.522496371552974,   276.4623467600701],
    [9.65166908563135,     279.9299474605955],
    [8.780841799709725,    282.13660245183894],
    [7.910014513788097,    284.3432574430824],
    [7.039187227866472,    287.18038528896676],
    [6.095791001451376,    290.96322241681264],
    [5.3701015965166885,   295.061295971979],
    [4.499274310595064,    298.84413309982494],
    [3.7735849056603765,   302.62697022767077],
    [3.265602322206094,    307.0402802101577],
    [2.83018867924528,     311.4535901926445],
    [2.322206095790998,    315.5516637478109],
    [1.8867924528301874,   319.9649737302978],
    [1.5239477503628436,   324.06304728546417],
    [1.3062409288824384,   328.16112084063053],
    [1.0159651669085612,   332.5744308231174],
    [0.7256894049346858,   336.9877408056043],
    [0.6531204644412192,   339.5096322241682],
    [0.5,   345],
    [0.3,   350],
    [0.2,   360],
    [0.1,   380],
    [0.05,  400],
    [0,     500],
    [0,     600],
    [0,     700],
    [0,     800],
    [0,     900],
    [0,    1000],
])

NightglowEmission = CubicSpline(
    NightglowData[:, 1] * 1000,
    NightglowData[:, 0] * 100**3 / 4 / np.pi,
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    h      = np.linspace(150_000, 500_000, num=300)
    h_km   = h / 1000
    h_sol    = np.linspace(150_000, 400_000, num=200)   # Solomon range only
    h_sol_km = h_sol / 1000

    fig, ax = plt.subplots(figsize=(8, 7))

    # ---- Meneses dayglow ----
    ax.plot(MenesesDayglowEmission(h), h_km,
            color="steelblue", lw=2.0,
            label="Dayglow — Meneses")
    ax.plot(
        MenesesDayglow[:, 0] * 100**3 / 4 / np.pi / 10,
        MenesesDayglow[:, 1],
        ".", color="steelblue", ms=5, label="_nolegend_",
    )

    # ---- Solomon & Abreu (1989) dayglow — SZA 30, 50, 70 ----
    ax.plot(SolomonEmission_SZA30(h_sol), h_sol_km,
            color="tomato", lw=1.8, ls="-",
            label="Dayglow — Solomon & Abreu (1989)  SZA=30°")
    ax.plot(SolomonEmission_SZA50(h_sol), h_sol_km,
            color="tomato", lw=1.8, ls="--",
            label="Dayglow — Solomon & Abreu (1989)  SZA=50°")
    ax.plot(SolomonEmission_SZA70(h_sol), h_sol_km,
            color="tomato", lw=1.8, ls=":",
            label="Dayglow — Solomon & Abreu (1989)  SZA=70°")

    # digitised scatter markers for all three Solomon profiles
    for arr in (Solomon_SZA30, Solomon_SZA50, Solomon_SZA70):
        ax.plot(
            arr[:, 0] * 100**3 / 4 / np.pi / 10,
            arr[:, 1],
            "x", color="tomato", ms=5, lw=1.0, label="_nolegend_",
        )

    # ---- Meneses nightglow ----
    ax.plot(NightglowEmission(h), h_km,
            color="seagreen", lw=2.0,
            label="Nightglow — Meneses")
    ax.plot(
        NightglowData[:, 0] * 100**3 / 4 / np.pi,
        NightglowData[:, 1],
        ".", color="seagreen", ms=5, label="_nolegend_",
    )

    ax.set_xlabel(r"Volume Emission Rate [m$^{-3}$ s$^{-1}$ sr$^{-1}$]")
    ax.set_ylabel("Height [km]")
    ax.set_title("OI 630.0 nm Emission Models")
    ax.set_ylim(140, 520)
    ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
