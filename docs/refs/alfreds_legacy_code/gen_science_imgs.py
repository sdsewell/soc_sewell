# just the observation portion of instrument model notebook 
import logging
from functools import partial

import astropy.constants as C
import cv2
import matplotlib.colors as colors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import curve_fit
from tqdm import trange

import WindCube 

# output directory for where to images! we dont want to save them in github so they should be saved
# up a directory and into a directory specific for output 
outdir = '/home/dfarrell/windcube/simulation_output/' 

temps = [-30, -20, -10, 0]
# temps = [-30]
temps_k = [t + 273.15 for t in temps] 
means = [] 
stddevs = [] 

fig, ax = plt.subplots(1, 4, figsize=(12, 4))
fig2, ax2 = plt.subplots(4, 1, figsize=(6,8))

for i, fpa_temp in enumerate(temps_k): 
    print('on fpa temp (k) ', fpa_temp, ' and (c) ', temps[i])


    plt.set_loglevel("warning")
    rng = np.random.default_rng()

    deltavd = 2.5  # Etalon separation velocity error
    deltavs = 1.0  # spacecraft velocity error
    deltav0 = 1.0  # zero-point velocity error

    deltavw = 5.0  # goal wind speed error

    photon_flux = WindCube.transmission * np.load("photon_flux.npy")
    calibration_flux = 10.0  # guess for now
    thisexposure = 1 # alfred also recommended changing this from 10 to 5. so put exposure in the fig title


    def synth_sci_data(dx, dy, v_dop, temperature, v_spacecraft):
        # synthesize a data set
        data_rx = np.linspace(-1, 1, num=WindCube.pixels) + dx * 2 / WindCube.pixels
        data_ry = np.linspace(-1, 1, num=WindCube.pixels) + dy * 2 / WindCube.pixels
        xx, yy = np.meshgrid(data_rx, data_ry)  # shape (pixels, pixels)
        rr = np.sqrt(xx**2 + yy**2)
        unique_r, unique_r_index = np.unique(rr, return_inverse=True)
        a = WindCube.S(
            unique_r,
            v_dop,
            temperature,
            v_spacecraft,
            0,
            1,
            WindCube.separation,
            WindCube.reflectivity,
        )[unique_r_index]

        return a


    def observe_sci_data(a, exposure, fpa_temp):
        s1 = rng.poisson(
            lam=(a * photon_flux + WindCube.darkcurrent(fpa_temp))
            * thisexposure
        ) + rng.poisson(lam=WindCube.read_noise * np.ones_like(a))

        return s1


    def sci_fit_function_simple(r, v_dop, scale, offset, v_spacecraft, temperature):
        signal = WindCube.S(
            r,
            v_dop,
            temperature,
            v_spacecraft,
            0.0,
            1.0,
            WindCube.separation,
            WindCube.reflectivity,
        )
        return scale * signal + offset


    def fit_sci_data_simple(r, radial_mean, sem, v_spacecraft, temperature):
        # initial guess [v_dop (km/s), temperature (K), scale, offset]
        p0 = [
            np.float64(rng.normal() * 10),
            np.float64(np.mean(photon_flux) * thisexposure),
            np.float64(
                WindCube.darkcurrent(fpa_temp) * thisexposure
            ),
        ]
        logging.debug(f"Starting from {p0}")

        # absolute_sigma=True treats yerr as absolute 1-sigma errors -> covariance scaled appropriately
        popt, pcov, infodict, mesg, ier = curve_fit(
            partial(
                sci_fit_function_simple,
                v_spacecraft=v_spacecraft,
                temperature=temperature,
            ),
            r,
            radial_mean,
            p0=p0,
            sigma=sem,
            absolute_sigma=True,
            maxfev=20000,
            full_output=True,
            ftol=1e-12,
        )

        perr = np.sqrt(np.diag(pcov))
        logging.debug(f"Found {popt} ± {perr}, message: {mesg}")

        return popt, perr, ier


    def analyze_sci_data(s1, v_spacecraft, cx, cy, temperature):
        r, s1_mean, s1_sem, s1_stdev = WindCube.radial_mean(s1, cx, cy)
        rr = r / np.max(r) * np.sqrt(2)
        w = np.nonzero(rr <= 0.5)

        (v_dop, scale, offset), (v_dop_err, scale_err, offset_err), ier = (
            fit_sci_data_simple(
                rr[w], s1_mean[w], s1_sem[w], v_spacecraft, temperature
            )
        )

        return v_dop, ier


    def plot_sci_data_analysis(
        r, s1_mean, s1_sem, v_dop, scale, offset, v_spacecraft, temperature
    ):
        rr = r / np.max(r) * np.sqrt(2)
        fig, ax = plt.subplots()
        ax.plot(r, s1_mean, label="Data", color="C0")
        ax.fill_between(
            r,
            s1_mean - s1_sem,
            s1_mean + s1_sem,
            color="C0",
            alpha=0.3,
            label="±1 SEM",
        )
        ax.plot(
            r,
            sci_fit_function_simple(
                rr, v_dop, scale, offset, v_spacecraft, temperature
            ),
            color="C1",
            label="Fit",
        )
        ax.legend()
        
    # scott wants three images per round 
    #(1) single image with noise  
    #(2) radial avg plus fit of rings 
    #(3) histogram 
    # save each of these into a file so I dont have to keep running! 
    
    # general params - v_dop changed from 0 to 5 m/s 
    dx = 0.0
    dy = 0.0
    cx = 255.5
    cy = 255.5
    v_dop = 5.0
    temperature = 500.0
    v_spacecraft = 0.0
    
    # (1) single science image (colorbar 0 to highest)
    a = synth_sci_data(dx, dy, v_dop, temperature, v_spacecraft)
    s1 = observe_sci_data(a, thisexposure, fpa_temp)
    r, mean, sem, stdev = WindCube.radial_mean(s1, cx, cy)
    ax[i].imshow(s1, vmin=0)
    ax[i].set_title(f"$exp = {thisexposure}$ s, $T = {temps[i]}$ C")
    
    # (2) radial mean 
    ax2[i].plot(r, mean, label="Data", color="C0")
    ax2[i].fill_between(
        r,
        mean - stdev,
        mean + stdev,
        color="C0",
        alpha=0.3,
        label=r"$\pm1\sigma$",
    )
    ax2[i].set_xlabel("Radius (pixels)")
    ax2[i].set_ylabel("Radial Mean Value (DN)")
    ax2[i].set_title(f"$exp = {thisexposure}$ s, $T = {temps[i]}$ C")
    ax2[i].legend()


fig.tight_layout()
fig.show()
figname = 'simulated_science_imgs_'+str(thisexposure)+'s_exp.pdf'
fig.savefig(outdir+figname)
figname = 'simulated_science_imgs_'+str(thisexposure)+'s_exp.png'
fig.savefig(outdir+figname)

fig2.tight_layout()
fig2.show()
figname = 'radial_mean_vals_'+str(thisexposure)+'s_exp.pdf'
fig2.savefig(outdir+figname)
figname = 'radial_mean_vals_'+str(thisexposure)+'s_exp.png'
fig2.savefig(outdir+figname)
