import logging
from numba import jit, cfunc
import numpy as np
from functools import lru_cache, partial
from scipy.integrate import fixed_quad
import scipy.ndimage as ndimage
import astropy.constants as C


logger = logging.getLogger(__name__)

pixel_size = 16e-6  # m from CCD97 data sheet
camera_lens = 0.2  # m
pixels = 512
FOV = 2 * np.atan(pixels / 2 * pixel_size / camera_lens)
logging.info(f"FOV is {np.degrees(FOV):.2f} square deg")
alpha = FOV / 2
sigmas = np.array([0.8, 0.1, -0.05]) * 2 / pixels
reflectivity = 0.8
separation = 20e-3  # m
integration_time = 10.0  # s
transmission = 0.85
FPA_temperature = 243.15  # K

calibration_wavelengths = np.array([640.2248e-9, 638.29914e-9])  # m, Ne line
calibration_wavelengths_relative_intensity = np.array([1, 0.8])
line_wavelength = 630.008e-9  # m, O2 line

read_noise = 4.5  # e-


# Transmission profile
# Given a wavelength $\lambda$, field angle $\theta$, etalon reflectivity $r$ and transmission $t=1-r$, and etalon separation $d$
@jit
def etalon_transmission(theta, wavelength, separation, reflectivity):
    transmission = 1 - reflectivity
    peak_transmission = transmission**2 / (1 - reflectivity) ** 2
    delta = 2 * np.pi * separation / wavelength * np.cos(theta)
    return (
        peak_transmission
        * 1
        / (
            1
            + (4 * reflectivity / (1 - reflectivity) ** 2) * np.sin(delta) ** 2
        )
    )


# PSF sigma function
@jit
def sigma(r):
    # r is normalized to 1
    return sigmas[0] + sigmas[1] * r + sigmas[2] * r**2


# PSF
@jit
def psf(s, r):
    ss = sigma(r) ** 2
    return 1 / np.sqrt(2 * np.pi * ss) * np.exp(-((s - r) ** 2) / ss)


# Angle through the Etalon for a given field point.
@jit
def theta(r, alpha = FOV / 2):
    return np.arctan(alpha * r)


## $\tilde A$
# Harding et al. (2014) Eq. 5 but with modified PSF function


# integrand function
def Atilde_integrand(s, r, wavelength, separation, reflectivity):
    return etalon_transmission(theta(s), wavelength, separation, reflectivity) * psf(s, r)
nb_Atilde_integrand = cfunc("float64(float64, float64, float64, float64, float64)")(Atilde_integrand)


# Atilde function
@np.vectorize(excluded=["wavelength", "factor", "separation", "reflectivity"])
@lru_cache(maxsize=1000000)
def Atilde(r, wavelength, separation, reflectivity, factor=3):
    if not np.isscalar(r):
        raise TypeError(
            f"Expected r to be a scalar, but got a {type(r).__name__}."
        )

    res = fixed_quad(
        nb_Atilde_integrand,
        r - factor * sigmas[0],
        r + factor * sigmas[0],
        n=12,
        args=(r, wavelength, separation, reflectivity)
    )

    return res[0]


## Airglow Emission function
def Y(wavelength, wavelength_center, delta_wavelength, Y_bg, Y_line):
    return Y_bg + Y_line / (delta_wavelength * np.sqrt(2 * np.pi)) * np.exp(
        -(((wavelength - wavelength_center) / delta_wavelength) ** 2) / 2
    )
nb_Y = cfunc("float64(float64, float64, float64, float64, float64)")(Y)


## Signal function

# integrand
def S_integrand(wlen, r, wavelength_center, delta_wavelength, Y_bg, Y_line, separation, wavelength):
    return Atilde(r, wlen, separation, reflectivity) * nb_Y(wlen, wavelength_center, delta_wavelength, Y_bg, Y_line)


# signal
@np.vectorize(excluded=[
        "v_dop",
        "temperature",
        "v_spacecraft",
        "Y_bg",
        "Y_line",
        "factor",
])
@lru_cache(maxsize=1000000)
def S(r, v_dop, temperature, v_spacecraft, Y_bg, Y_line, separation, wavelength, factor=5):
    if not np.isscalar(r):
        raise TypeError(
            f"Expected r to be a scalar, but got a {type(r).__name__}."
        )

    # calculate the thermal broadening
    # Oxygen mass is 15.999 u
    delta_wavelength = (
        line_wavelength
        / C.c.value
        * np.sqrt(C.k_B.value * temperature / 2 / 15.999 / C.u.value)
    )
    # calculate the central wavelength
    wavelength_center = line_wavelength * (
        1 + (v_dop + v_spacecraft) / C.c.value
    )

    res = fixed_quad(
        S_integrand,
        wavelength_center - factor * delta_wavelength,
        wavelength_center + factor * delta_wavelength,
        n=48,
        args=(r, wavelength_center, delta_wavelength, Y_bg, Y_line, separation, wavelength)
    )

    return res[0]


## Dark current
@jit
def darkcurrent(T):
    return 4.56e8 * T**3 * np.exp(-9080 / T)


## Radial mean calculation function
def radial_mean(data, cx, cy):
    # radial averaging
    px = data.shape
    xgrid, ygrid = np.ogrid[0 : px[0], 0 : px[1]]
    rcxcy = np.hypot(xgrid - cy, ygrid - cx)
    rbin = np.round(rcxcy * 4) / 4
    r = np.unique(rbin)

    mean = ndimage.mean(data, labels=rbin, index=r)
    stdev = ndimage.standard_deviation(data, labels=rbin, index=r)
    count = ndimage.sum(np.ones_like(data), labels=rbin, index=r)
    sem = stdev / np.sqrt(count)

    return r, mean, sem, stdev