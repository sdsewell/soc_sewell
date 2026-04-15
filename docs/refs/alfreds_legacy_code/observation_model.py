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

exps = [1, 5, 10] 

for j, thisexposure in enumerate(exps): 

    print('exposure = ', thisexposure) 

    temps = [-30, -20, -10, 0]
    # temps = [-30]
    temps_k = [t + 273.15 for t in temps] 
    means = [] 
    stddevs = [] 

    fig, ax = plt.subplots(1, 4, figsize=(14, 4))

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


        # general params - v_dop changed from 0 to 5 m/s 
        # exposures of 1 5 and 10 s for the four temps (put exps in outer loop) 
        dx = 0.0
        dy = 0.0
        cx = 255.5
        cy = 255.5
        v_dop = 5.0
        temperature = 500.0
        v_spacecraft = 0.0
        # thisexposure = 1  # s
        
        def synth_sci_data(dx, dy, vdop, temperature, v_spacecraft):
            # synthesize a data set
            data_rx = np.linspace(-1, 1, num=WindCube.pixels) + dx * 2 / WindCube.pixels
            data_ry = np.linspace(-1, 1, num=WindCube.pixels) + dy * 2 / WindCube.pixels
            xx, yy = np.meshgrid(data_rx, data_ry)  # shape (pixels, pixels)
            rr = np.sqrt(xx**2 + yy**2)
            unique_r, unique_r_index = np.unique(rr, return_inverse=True)
            a = WindCube.S(
                unique_r,
                vdop,
                temperature,
                v_spacecraft,
                0,
                1,
                WindCube.separation,
                WindCube.reflectivity,
            )[unique_r_index]

            return a


        def observe_sci_data(a, exposure, temp):
            s1 = rng.poisson(
                lam=(a * photon_flux + WindCube.darkcurrent(temp))
                * exposure
            ) + rng.poisson(lam=WindCube.read_noise * np.ones_like(a))

            return s1


        def sci_fit_function_simple(r, vdop, scale, offset, v_spacecraft, temperature):
            signal = WindCube.S(
                r,
                vdop,
                temperature,
                v_spacecraft,
                0.0,
                1.0,
                WindCube.separation,
                WindCube.reflectivity,
            )
            return scale * signal + offset


        def fit_sci_data_simple(r, radial_mean, sem, v_spacecraft, temperature, temp):
            # initial guess [v_dop (km/s), temperature (K), scale, offset]
            p0 = [
                np.float64(rng.normal() * 10),
                np.float64(np.mean(photon_flux) * thisexposure),
                np.float64(
                    WindCube.darkcurrent(temp) * thisexposure
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


        def analyze_sci_data(s1, v_spacecraft, cx, cy, temperature, temp):
            r, s1_mean, s1_sem, s1_stdev = WindCube.radial_mean(s1, cx, cy)
            rr = r / np.max(r) * np.sqrt(2)
            w = np.nonzero(rr <= 0.5)

            (vdop, scale, offset), (vdop_err, scale_err, offset_err), ier = (
                fit_sci_data_simple(
                    rr[w], s1_mean[w], s1_sem[w], v_spacecraft, temperature, temp
                )
            )

            return vdop, ier


        def plot_sci_data_analysis(
            r, s1_mean, s1_sem, vdop, scale, offset, v_spacecraft, temperature
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
                    rr, vdop, scale, offset, v_spacecraft, temperature
                ),
                color="C1",
                label="Fit",
            )
            ax.legend()
            
            

        a = synth_sci_data(dx, dy, v_dop, temperature, v_spacecraft)
        s1 = observe_sci_data(a, thisexposure, fpa_temp)
        r, mean, sem, stdev = WindCube.radial_mean(s1, cx, cy)
        
        
        # (3) histogram 
        samples = 1000

        def wrapper(a, i):
            try:
                thisvdop, ier = analyze_sci_data(
                    observe_sci_data(a, thisexposure, fpa_temp), v_spacecraft, cx, cy, temperature, fpa_temp
                )
                logging.debug(f"Finished sample {i}, found v_dop = {thisvdop}.")
            except Exception as e:
                logging.warning(f"Something went wrong with sample {i}.")
                thisvdop = float("nan")
            return thisvdop, ier


        logging.getLogger().setLevel(logging.INFO)
        vdops = np.array(list([wrapper(a, i) for i in range(samples)]))

        temp_c = temps[i] 
        histogram, edges = np.histogram(vdops[:, 0], bins=50, density=True)
        np.savetxt(outdir+'vdop_data_'+str(thisexposure)+'exp_'+str(temp_c)+'C.txt', vdops[:, 0])
        ax[i].stairs(histogram, edges, color="C0")
        ax[i].axvline(np.mean(vdops[:,0]), ls=":", color="C0")
        ax[i].set_xlabel(r"$\Delta v_\mathrm{dop}$ [m/s]")
        ax[i].set_xlim(-40, 40)
        ax[i].set_ylabel(r"Probability Density")
        ax[i].set_title(f"{np.mean(vdops[:,0]):.2f} ± {np.std(vdops[:,0]):.2f} m/s, {temp_c} C")
        print(f"Delta v_dop = {np.mean(vdops[:,0]):.2f} ± {np.std(vdops[:,0]):.2f} m/s")
        print(f"v_dop recovered = {np.mean([v_dop - v for v in vdops[:,0]]):.2f} ± {np.std(vdops[:,0]):.2f} m/s")
        means.append(np.mean(vdops[:,0]))
        stddevs.append(np.std(vdops[:,0]))

    fig.tight_layout()  
    figname = 'vdop_probdensity_'+str(thisexposure)+'s_exp.pdf'
    fig.savefig(outdir+figname)
    figname = 'vdop_probdensity_'+str(thisexposure)+'s_exp.png'
    fig.savefig(outdir+figname)

    
