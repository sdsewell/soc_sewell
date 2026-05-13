"""
Script:  H03_airglow_synthesis_2026_05_13.py
Purpose: Generate a synthetic OI 630 nm airglow fringe image using the
         WindCube FPI forward model, display it with its pixel histogram,
         and optionally save the image and 1D radial profile as .npy files.

         This script is the interactive companion to m03_airglow_synthesis.
         It prompts the user for all parameters that vary on orbit, provides
         physically motivated defaults, then calls synthesise_airglow_image()
         and displays the result.

═══════════════════════════════════════════════════════════════════════════════
 ORBIT-VARIABLE PARAMETERS (prompted via dialog boxes)
═══════════════════════════════════════════════════════════════════════════════

  Observation mode
    cross_track  — even orbits; LOS ⊥ orbit track; v_rel ≈ ±500 m/s (wind)
    along_track  — odd orbits;  LOS ∥ orbit track; v_rel ≈ −7000 m/s (orbital)

  Line-of-sight velocity (v_rel, m/s)
    Cross-track default: 0 m/s  (pure zero-wind reference)
    Along-track default: −7000 m/s  (typical along-track)

  CCD binning
    2×2 (default, flight mode): 256×256 pixels, α = 1.6084e-4 rad/px
    1×1 (ground test / calibration mode): 512×512 pixels, α = 8.042e-5 rad/px

  Etalon gap t (mm)
    Operational (Tolansky-recovered): 20.1069746 mm  ← default
    Changes with etalon temperature: ~1 nm / K (fused silica spacer)
    Provide phase-corrected t_eff from H05 for faithful fringe placement.

  Effective reflectivity R
    Default: 0.239  (R1 from H05 calibration at λ₁=640.2 nm, close to OI)
    Changes with coating temperature; ~0.001 / K typical

  OI line intensity Y_line (ADU)
    Typical quiet-time airglow: 500–2000 ADU
    Default: 6480 ADU  (H03 reference value)

  Exposure time t_exp (s)
    Science (airglow) default:    10 s
    Calibration (neon) typical:  120 s

  Focal-plane temperature T_fp (°C)
    Design operating point: −20 °C
    Dark current computed from Teledyne e2v CCD97 formula.

═══════════════════════════════════════════════════════════════════════════════
 OUTPUT FIGURE (2 panels)
═══════════════════════════════════════════════════════════════════════════════

  Top:    2D synthetic airglow fringe image (greyscale, log-stretch)
          Annotated with all synthesis parameters.

  Bottom: Pixel value histogram of the image.
          Shows noise distribution; for Poisson + read noise should be
          approximately Gaussian centred on B.

═══════════════════════════════════════════════════════════════════════════════
 OPTIONAL SAVES
═══════════════════════════════════════════════════════════════════════════════

  After displaying the figure, prompts whether to save:
    <stem>_airglow_image_2d.npy    — 2D noisy image (256×256 or 512×512)
    <stem>_airglow_profile_vs_r2.npy — (2, R_bins) array: r², profile(r)

  The profile file has the same format as the neon calibration profile
  files and can be loaded directly by H06_airglow_inversion_2026_05_13.py.

Run from repo root:
    python src/processing/H03_airglow_synthesis_2026_05_13.py
"""

import datetime
import pathlib
import sys
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import logging

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Make repo root importable
# ---------------------------------------------------------------------------
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.fpi.airy_forward_model_2026_05_05 import InstrumentParams  # noqa
from src.fpi.m03_airglow_synthesis_2026_05_13 import (              # noqa
    synthesise_airglow_image,
)
from windcube.constants import (                                     # noqa
    OI_WAVELENGTH_AIR_M,
    SPEED_OF_LIGHT_MS,
)

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s  %(name)s  %(message)s")
log = logging.getLogger("H03")

# ---------------------------------------------------------------------------
# Defaults — all can be overridden via dialog
# ---------------------------------------------------------------------------
_DEFAULTS = {
    # Observation
    'observation_mode': 'cross_track',  # 'cross_track' or 'along_track'
    'v_rel_ms':         0.0,            # m/s; 0 = zero-wind reference
    # CCD / binning
    'binning':          2,              # 1 or 2
    'image_size':       256,            # pixels (set from binning)
    # Etalon — use H05 phase-corrected t_eff as default
    't_eff_mm':         20.1069746,     # mm; phase-corrected from H05
    # Optics (from H05 calibration result)
    'R_refl':           0.239,          # effective reflectivity at OI wavelength
    'alpha_rad_px':     1.6084e-4,      # rad/px (2×2 binned)
    # Intensity envelope (from H05)
    'I0':               6479.9,         # ADU
    'I1':              -0.04684,
    'I2':              -0.02599,
    'sigma0':           0.5528,         # px
    'B':                2010.7,         # ADU bias pedestal
    # Source
    'Y_line':           6480.0,         # ADU; H03 reference value
    'Y_bg':             0.0,            # ADU per wavelength bin
    # Noise — physical CCD model (Teledyne e2v CCD97)
    'add_noise':        True,
    'exp_time_s':       10.0,           # s; science default
    'T_focal_plane_C': -20.0,           # °C; design operating point
    'gain_e_per_adu':   1.0,            # e-/ADU (TBC from FlatSat)
    'read_noise_e':     2.2,            # e- rms (OSH amp, 50 kHz CDS)
    'Qdd_at_20C':       400.0,          # e-/pix/s at 20 °C (CCD97 typical)
    # Synthesis resolution
    'R_bins':           500,
    'L_synth':          300,
    'n_fsr':            10.0,
    'r_max_px':         110.0,          # px; flight/FlatSat value
}


def _ask_observation_mode(root):
    """Prompt for observation mode and v_rel."""
    mode = simpledialog.askstring(
        "Observation mode",
        "Observation mode:\n"
        "  cross_track  — even orbits; wind ≈ ±500 m/s\n"
        "  along_track  — odd orbits;  v_rel ≈ −7000 m/s\n"
        "  none         — no mode validation\n\n"
        "Enter: cross_track / along_track / none",
        initialvalue=_DEFAULTS['observation_mode'],
        parent=root) or _DEFAULTS['observation_mode']
    mode = mode.strip().lower()
    if mode == 'none':
        mode = None

    if mode == 'along_track':
        v_default = -7000.0
        v_min, v_max = -8000.0, -6000.0
        hint = "Along-track: spacecraft + wind.\nTypical: −7000 m/s\nRange: −8000 to −6000"
    elif mode == 'cross_track':
        v_default = 0.0
        v_min, v_max = -1000.0, 1000.0
        hint = "Cross-track: thermospheric wind only.\nTypical: ±200 m/s storm, 0 = zero-wind\nRange: −1000 to +1000"
    else:
        v_default = _DEFAULTS['v_rel_ms']
        v_min, v_max = -15000.0, 15000.0
        hint = "No mode validation.\nPositive = recession (redshift).\nRange: −15000 to +15000"

    v_rel = simpledialog.askfloat(
        "v_rel (m/s)",
        f"{hint}\n\nLine-of-sight velocity (m/s):",
        initialvalue=v_default,
        minvalue=v_min, maxvalue=v_max,
        parent=root) or v_default

    return mode, v_rel


def _ask_instrument_params(root):
    """Prompt for all instrument parameters with defaults."""
    binning = simpledialog.askinteger(
        "CCD binning",
        "On-chip binning factor:\n  2 = 2×2 binned (256×256, flight default)\n  1 = unbinned (512×512)",
        initialvalue=_DEFAULTS['binning'], minvalue=1, maxvalue=4,
        parent=root) or _DEFAULTS['binning']

    # Scale alpha and image_size with binning
    alpha_2x2 = _DEFAULTS['alpha_rad_px']
    alpha_default = alpha_2x2 * (2.0 / binning)
    image_size = 256 * (2 // binning) if binning <= 2 else 256

    t_eff_mm = simpledialog.askfloat(
        "Etalon gap t_eff (mm)",
        "Phase-corrected etalon gap (mm).\n"
        "Use t_eff from H05 calibration for faithful fringe placement.\n"
        "Tolansky t ± ~0.1 nm changes fringe position by ~0.3 FSR.",
        initialvalue=_DEFAULTS['t_eff_mm'],
        minvalue=19.0, maxvalue=21.0,
        parent=root) or _DEFAULTS['t_eff_mm']

    R_refl = simpledialog.askfloat(
        "Reflectivity",
        "Effective reflectivity R\n"
        "Effective etalon reflectivity at OI 630 nm.\n"
        "Use R1 from H05 calibration (λ₁=640.2 nm, closest to OI).\n"
        "Default: 0.239  (R1 from H05 at λ=640.2 nm)",
        initialvalue=_DEFAULTS['R_refl'],
        minvalue=0.05, maxvalue=0.95,
        parent=root) or _DEFAULTS['R_refl']

    alpha = simpledialog.askfloat(
        "Plate scale α (rad/px)",
        f"Plate scale (rad/px) for binning={binning}×{binning}.\n"
        f"2×2 default: 1.6084e-4   1×1: 8.042e-5",
        initialvalue=alpha_default,
        minvalue=1e-5, maxvalue=1e-3,
        parent=root) or alpha_default

    r_max_px = simpledialog.askfloat(
        "r_max (pixels)",
        "Maximum usable fringe radius (pixels).\n"
        "Flight / FlatSat: 110   Synthetic only: 128",
        initialvalue=_DEFAULTS['r_max_px'],
        minvalue=50.0, maxvalue=200.0,
        parent=root) or _DEFAULTS['r_max_px']

    I0 = simpledialog.askfloat(
        "I₀ (ADU)",
        "Mean intensity envelope I₀ (ADU).\n"
        "H05 calibration result: 6479.9",
        initialvalue=_DEFAULTS['I0'], minvalue=10.0, maxvalue=50000.0,
        parent=root) or _DEFAULTS['I0']

    B = simpledialog.askfloat(
        "Bias B (ADU)",
        "CCD bias pedestal B (ADU).\n"
        "H05 calibration result: 2010.7",
        initialvalue=_DEFAULTS['B'], minvalue=0.0, maxvalue=5000.0,
        parent=root) or _DEFAULTS['B']

    sigma0 = simpledialog.askfloat(
        "PSF width σ₀ (pixels)",
        "Gaussian PSF base width σ₀ (pixels).\n"
        "H05 calibration result: 0.5528 px  (constant blur; σ₁=σ₂=0)",
        initialvalue=_DEFAULTS['sigma0'], minvalue=0.01, maxvalue=5.0,
        parent=root) or _DEFAULTS['sigma0']

    return dict(binning=binning, image_size=image_size, t_eff_mm=t_eff_mm,
                R_refl=R_refl, alpha=alpha, r_max_px=r_max_px,
                I0=I0, B=B, sigma0=sigma0)


def _ask_source_and_noise(root):
    """Prompt for source intensity and physical CCD noise parameters."""
    Y_line = simpledialog.askfloat(
        "Line intensity",
        "OI line intensity Y_line (ADU)\n"
        "Quiet-time range: 500–2000 ADU\n"
        "Default: 6480 ADU  (H03 reference value)",
        initialvalue=_DEFAULTS['Y_line'], minvalue=100.0, maxvalue=30000.0,
        parent=root) or _DEFAULTS['Y_line']

    Y_bg = simpledialog.askfloat(
        "Sky background Y_bg (ADU/bin)",
        "Spectrally flat sky background (ADU per wavelength bin).\n"
        "0 = pure emission line",
        initialvalue=_DEFAULTS['Y_bg'], minvalue=0.0, maxvalue=5000.0,
        parent=root) or _DEFAULTS['Y_bg']

    exp_time_s = simpledialog.askfloat(
        "Exposure time",
        "Exposure time t_exp (s)\n"
        "Science (airglow) default:     10 s\n"
        "Calibration (neon) typical:   120 s\n"
        "Longer exposures increase dark current noise.",
        initialvalue=_DEFAULTS['exp_time_s'],
        minvalue=0.1, maxvalue=300.0,
        parent=root) or _DEFAULTS['exp_time_s']

    T_focal_plane_C = simpledialog.askfloat(
        "Focal-plane temperature",
        "Focal-plane temperature T_fp (°C)\n"
        "Design operating point:  −20 °C\n"
        "Warm on-orbit risk:      −10 °C\n"
        "Cooling failure:         +20 °C\n"
        "Range: −40 to +20 °C\n"
        "\n"
        "Dark current is computed from the Teledyne e2v CCD97 formula\n"
        "(datasheet page 4): Qdd = 400 e-/pix/s at 20 °C.\n"
        "Gain = 1.0 e-/ADU   Read noise = 2.2 e- rms (OSH, 50 kHz CDS).",
        initialvalue=_DEFAULTS['T_focal_plane_C'],
        minvalue=-40.0, maxvalue=20.0,
        parent=root)
    if T_focal_plane_C is None:
        T_focal_plane_C = _DEFAULTS['T_focal_plane_C']

    return dict(Y_line=Y_line, Y_bg=Y_bg,
                exp_time_s=exp_time_s,
                T_focal_plane_C=T_focal_plane_C)


def make_figure(result: dict, params: InstrumentParams,
                obs_mode, v_rel_ms: float) -> plt.Figure:
    """
    Two-panel figure: top = 2D image, bottom = pixel histogram.
    """
    image   = result['image_2d']
    profile = result['profile_1d']
    r_grid  = result['r_grid']
    lam_c   = result['lambda_c_m']
    forder  = result['fringe_order_offset']
    snr_act = result['snr_actual']
    exp_time_s      = result['exp_time_s']
    T_fp            = result['T_focal_plane_C']
    dark_rate       = result['dark_rate_e_per_s']

    # Doppler velocity from lambda_c
    v_check = SPEED_OF_LIGHT_MS * (lam_c - OI_WAVELENGTH_AIR_M) / OI_WAVELENGTH_AIR_M

    fig = plt.figure(figsize=(12, 10))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1.5],
                            hspace=0.30, top=0.88, bottom=0.07,
                            left=0.08, right=0.97)
    ax_img = fig.add_subplot(gs[0])
    ax_his = fig.add_subplot(gs[1])

    # ---- Top: 2D image ----
    # Log stretch to show both bright peaks and dark troughs
    img_pos = np.clip(image, 1e-3, None)
    vmin    = float(np.percentile(img_pos, 2))
    vmax    = float(np.percentile(img_pos, 98))
    im = ax_img.imshow(image, origin='lower', cmap='gray',
                       vmin=vmin, vmax=vmax, aspect='equal')
    plt.colorbar(im, ax=ax_img, fraction=0.03, pad=0.01, label='ADU')
    ax_img.set_title(
        f"Synthetic OI 630 nm airglow fringe  "
        f"({'noisy' if result['snr_actual'] < np.inf else 'noiseless'})",
        fontsize=11, fontweight='bold')
    ax_img.set_xlabel("Column (pixels)", fontsize=9)
    ax_img.set_ylabel("Row (pixels)", fontsize=9)

    # Annotation box
    mode_str = obs_mode if obs_mode else "none"
    ann = (
        f"v_rel = {v_rel_ms:+.1f} m/s  ({v_check:+.1f} m/s check)\n"
        f"λ_c = {lam_c*1e9:.6f} nm   fringe_order_offset = {forder}\n"
        f"t = {params.t*1e3:.7f} mm   α = {params.alpha:.4e} rad/px\n"
        f"R = {params.R_refl:.4f}   σ₀ = {params.sigma0:.4f} px\n"
        f"I₀ = {params.I0:.0f}   B = {params.B:.0f} ADU\n"
        f"t_exp = {exp_time_s:.1f} s   T_fp = {T_fp:.1f} °C\n"
        f"dark = {dark_rate:.4f} e-/pix/s   SNR_actual = {snr_act:.2f}\n"
        f"mode = {mode_str}   image = {image.shape[0]}×{image.shape[1]} px\n"
        f"Profile ΔS = {float(profile.max()-profile.min()):.1f} ADU"
    )
    ax_img.text(0.02, 0.98, ann, transform=ax_img.transAxes,
                va='top', ha='left', fontsize=7.5,
                fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          edgecolor='grey', alpha=0.88))

    # ---- Bottom: histogram ----
    flat   = image.ravel()
    n_bins = min(200, int(np.sqrt(len(flat))))
    ax_his.hist(flat, bins=n_bins, color='steelblue', alpha=0.75,
                edgecolor='none', density=True)
    ax_his.axvline(float(params.B), color='firebrick', lw=1.2, ls='--',
                   label=f'B = {params.B:.0f} ADU')
    ax_his.axvline(float(np.median(flat)), color='darkorange', lw=1.0,
                   ls=':', label=f'median = {np.median(flat):.0f} ADU')
    ax_his.set_xlabel("Pixel value  (ADU)", fontsize=10)
    ax_his.set_ylabel("Density", fontsize=10)
    ax_his.set_title(
        "Pixel value histogram  (Poisson + read noise → approximately Gaussian)",
        fontsize=9)
    ax_his.legend(fontsize=8.5)
    ax_his.grid(True, alpha=0.2)

    fig.suptitle("WindCube FPI — Synthetic OI 630 nm Airglow Image  (H03)",
                 fontsize=12, fontweight='bold', y=0.96)
    return fig


def main():

    _tk = tk.Tk(); _tk.withdraw()

    # ---- Step 1: Observation mode and velocity ----
    obs_mode, v_rel_ms = _ask_observation_mode(_tk)

    # ---- Step 2: Instrument parameters ----
    instr = _ask_instrument_params(_tk)

    # ---- Step 3: Source, exposure time, focal-plane temperature ----
    src = _ask_source_and_noise(_tk)

    _tk.destroy()

    # ---- Build InstrumentParams ----
    params = InstrumentParams(
        t      = instr['t_eff_mm'] * 1e-3,
        R_refl  = instr['R_refl'],
        n      = 1.0,
        alpha  = instr['alpha'],
        r_max  = instr['r_max_px'],
        I0     = instr['I0'],
        I1     = _DEFAULTS['I1'],
        I2     = _DEFAULTS['I2'],
        sigma0 = instr['sigma0'],
        sigma1 = 0.0,
        sigma2 = 0.0,
        B      = instr['B'],
    )

    # ---- Synthesise ----
    print(f"\nSynthesising airglow image…")
    print(f"  mode={obs_mode}   v_rel={v_rel_ms:+.1f} m/s")
    print(f"  t={params.t*1e3:.7f} mm   R={params.R_refl:.4f}   "
          f"α={params.alpha:.4e}   σ₀={params.sigma0:.4f}")

    result = synthesise_airglow_image(
        params           = params,
        v_rel_ms         = v_rel_ms,
        Y_line           = src['Y_line'],
        Y_bg             = src['Y_bg'],
        image_size       = instr['image_size'],
        R_bins           = _DEFAULTS['R_bins'],
        L_synth          = _DEFAULTS['L_synth'],
        n_fsr            = _DEFAULTS['n_fsr'],
        observation_mode = obs_mode,
        add_noise        = _DEFAULTS['add_noise'],
        exp_time_s       = src['exp_time_s'],
        T_focal_plane_C  = src['T_focal_plane_C'],
        gain_e_per_adu   = _DEFAULTS['gain_e_per_adu'],
        read_noise_e     = _DEFAULTS['read_noise_e'],
        Qdd_at_20C       = _DEFAULTS['Qdd_at_20C'],
        rng              = np.random.default_rng(42),
    )

    print(f"  Y_line={src['Y_line']:.0f}   t_exp={src['exp_time_s']:.1f}s   "
          f"T_fp={src['T_focal_plane_C']:.1f}°C   "
          f"dark={result['dark_rate_e_per_s']:.4f} e-/pix/s   "
          f"SNR_actual={result['snr_actual']:.2f}")
    print(f"  λ_c = {result['lambda_c_m']*1e9:.6f} nm   "
          f"fringe_order_offset = {result['fringe_order_offset']}")
    print(f"  Profile: ΔS = {float(result['profile_1d'].max()-result['profile_1d'].min()):.1f} ADU  "
          f"peak = {float(result['profile_1d'].max()):.1f}  "
          f"trough = {float(result['profile_1d'].min()):.1f}")

    # ---- Figure ----
    fig = make_figure(result, params, obs_mode, v_rel_ms)
    plt.show()

    # ---- Optional save ----
    _tk3 = tk.Tk(); _tk3.withdraw()
    do_save = messagebox.askyesno(
        "Save outputs?",
        "Save the 2D image and 1D profile as .npy files?\n\n"
        "  • <stem>_airglow_image_2d.npy\n"
        "  • <stem>_airglow_profile_vs_r2.npy  ← load into H06",
        parent=_tk3)

    if do_save:
        save_dir = filedialog.askdirectory(
            title="Choose save directory", parent=_tk3)
        _tk3.destroy()

        if save_dir:
            save_dir = pathlib.Path(save_dir)
            ts    = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
            mode_tag = (obs_mode or 'nomode').replace('_', '')
            stem  = f"airglow_{mode_tag}_v{int(v_rel_ms):+d}ms_{ts}"

            # 2D image
            img_path = save_dir / f"{stem}_airglow_image_2d.npy"
            np.save(img_path, result['image_2d'])
            print(f"\n✓ 2D image saved:   {img_path}")

            # 1D profile vs r²  — (2, R_bins) format matching H06 expectation
            r2_grid = result['r_grid'] ** 2
            prof_r2 = np.stack([r2_grid, result['profile_1d']], axis=0)
            prof_path = save_dir / f"{stem}_airglow_profile_vs_r2.npy"
            np.save(prof_path, prof_r2)
            print(f"✓ Profile saved:    {prof_path}")
            print(f"  (Load into H06 with: np.load(path, allow_pickle=False))")

            # Metadata sidecar
            meta = {
                'v_rel_ms':             v_rel_ms,
                'observation_mode':     obs_mode,
                'lambda_c_m':           result['lambda_c_m'],
                'fringe_order_offset':  result['fringe_order_offset'],
                'exp_time_s':           src['exp_time_s'],
                'T_focal_plane_C':      src['T_focal_plane_C'],
                'dark_rate_e_per_s':    result['dark_rate_e_per_s'],
                'dark_e_per_pixel':     result['dark_e_per_pixel'],
                'snr_actual':           result['snr_actual'],
                't_eff_mm':             instr['t_eff_mm'],
                'R_refl':               instr['R_refl'],
                'alpha_rad_px':         instr['alpha'],
                'sigma0_px':            instr['sigma0'],
                'I0':                   instr['I0'],
                'B':                    instr['B'],
                'Y_line':               src['Y_line'],
                'Y_bg':                 src['Y_bg'],
                'image_size':           instr['image_size'],
                'r_max_px':             instr['r_max_px'],
                'date_utc':             datetime.datetime.utcnow().isoformat(),
                'script':               'H03_airglow_synthesis_2026_05_13.py',
            }
            meta_path = save_dir / f"{stem}_meta.npy"
            np.save(meta_path, meta)
            print(f"✓ Metadata saved:   {meta_path}")
        else:
            _tk3.destroy()
    else:
        _tk3.destroy()
        print("\nOutputs not saved.")


if __name__ == "__main__":
    main()
