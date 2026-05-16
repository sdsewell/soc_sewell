# HWM14 Windows Installation Notes

**Date:** 2026-05-16  
**Author:** Scott Sewell / HAO SOC  
**Environment:** Windows 11, Anaconda3, conda-forge  
**Status:** ✅ Verified working

---

## Overview

HWM14 (Horizontal Wind Model 2014, Drob et al. 2015) is a NASA NRL empirical model
of thermospheric horizontal winds. It is implemented in Fortran 90 and wrapped for
Python via `f2py`. The Python interface used here is
[rilma/pyHWM14](https://github.com/rilma/pyHWM14) (`pyhwm2014` package).

HWM14 is used in the WindCube FPI pipeline G01 synthetic data generator
(`WIND_MAP_REGISTRY`, `hwm14` builder) to provide realistic thermospheric wind
fields for Monte Carlo simulation runs.

---

## Installed Location

| Item | Path |
|------|------|
| Conda env | `C:\Users\sewell\.conda\envs\hwm14_env` |
| pyHWM14 repo | `C:\Users\sewell\Documents\GitHub\soc_sewell\pyHWM14` |
| Binary data files | `C:\Users\sewell\Documents\GitHub\soc_sewell\pyHWM14\data\` |
| Activation script | `C:\Users\sewell\.conda\envs\hwm14_env\etc\conda\activate.d\hwm14_env_vars.bat` |

---

## Conda Environment

```
conda create -n hwm14_env python=3.12 gfortran -c conda-forge
conda activate hwm14_env
conda install -n hwm14_env -c conda-forge cmake ninja meson
pip install numpy scikit-build-core setuptools-scm
```

**Python version:** 3.12 (required — pyhwm2014 pyproject.toml requires `>=3.12`)  
**Key packages:** gfortran, cmake 4.3.2, ninja 1.13.2, meson 1.11.1

---

## Build and Install

```
cd C:\Users\sewell\Documents\GitHub\soc_sewell
git clone https://github.com/rilma/pyHWM14.git
cd pyHWM14
pip install -e . --no-build-isolation
```

### Critical Windows gotchas

#### 1. `Library\bin` not on PATH
conda on Windows does **not** automatically add the compiler directory to the shell
PATH on `conda activate`. The compilers (`gfortran.exe`, `gcc.exe`) live in
`Library\bin` but only `Scripts\` is added. CMake therefore cannot find them.

**Fix:** The activation script (see below) prepends `Library\bin` to PATH on every
`conda activate hwm14_env`.

#### 2. CMake generator must be Ninja
The new `pyhwm2014` build system uses `scikit-build-core` + CMake instead of the
old `f2py` + `distutils` path. CMake on Windows defaults to `NMake Makefiles`
(requires Visual Studio) or `MinGW Makefiles` (requires `mingw32-make`). Neither is
present in the conda env. **Use Ninja** instead.

**Fix:** `set CMAKE_GENERATOR=Ninja` (set in activation script).

#### 3. Git symlink in data directory
`pyhwm2014/data` is a Git symlink. On Windows, Git stores symlinks as plain text
files containing the target path string. This causes `data.py` to resolve the wrong
`HWMPATH` at import time.

**Fix:** Patch `data.py` line 12:
```python
# Before (broken on Windows):
HWMPATH: str = str(Path(__file__).parent / "data")

# After (fixed):
HWMPATH: str = str(Path(__file__).parent.parent / "data")
```
This fix has been applied to the cloned repo.

---

## Conda Activation Script

File: `C:\Users\sewell\.conda\envs\hwm14_env\etc\conda\activate.d\hwm14_env_vars.bat`

```bat
@echo off
set "LIBRARY_BIN=%CONDA_PREFIX%\Library\bin"
set "PATH=%LIBRARY_BIN%;%PATH%"
set "FC=%LIBRARY_BIN%\gfortran.exe"
set "CC=%LIBRARY_BIN%\gcc.exe"
set "CMAKE_GENERATOR=Ninja"
set "HWMPATH=C:\Users\sewell\Documents\GitHub\soc_sewell\pyHWM14\data"
```

**Important:** This file must be written using Python or a text editor — **not**
`echo` redirection in cmd.exe. The cmd.exe `echo` command expands `%VAR%` tokens
immediately, corrupting the file with the current runtime values instead of the
literal variable references needed for a `.bat` script.

---

## Required Binary Data Files

HWM14 requires three binary data files at runtime. They are included in the
`pyHWM14` repo under `data/`:

| File | Size | Contents |
|------|------|----------|
| `hwm123114.bin` | 193,608 bytes | Quiet-time wind coefficients |
| `dwm07b104i.dat` | 4,848 bytes | Storm disturbance wind model |
| `gd2qd.dat` | 1,500 bytes | Geodetic → quasi-dipole coordinates |

HWM14 searches for these files in order:
1. Directory defined by `HWMPATH` environment variable
2. Current working directory
3. `../Meta/` relative to current working directory

---

## Verification Query

```python
from pyhwm2014 import HWM14

hwm = HWM14(
    alt=250.0,           # km — OI 630 nm thermospheric layer
    altlim=[250., 250.],
    altstp=1,
    year=2024,
    day=80,              # spring equinox
    ut=12.0,             # noon UT
    glat=0.0,            # geographic equator
    glon=0.0,
    ap=[-1, 10],         # ap=-1 = climatology mode; ap[1] = Ap index
    option=1,
    verbose=False
)

print(f"Zonal wind (U):      {hwm.Uwind[0]:+.1f} m/s  (+ = eastward)")
print(f"Meridional wind (V): {hwm.Vwind[0]:+.1f} m/s  (+ = northward)")
```

**Verified output (2026-05-16):**
```
Zonal wind (U):      -76.8 m/s
Meridional wind (V): -3.3 m/s
```

Expected range: ±200 m/s at 250 km quiet-time equatorial conditions.

---

## VS Code Integration

1. `Ctrl+Shift+P` → **Python: Select Interpreter**
2. Select: `C:\Users\sewell\.conda\envs\hwm14_env\python.exe`

Add to `.vscode/settings.json` in the `soc_sewell` workspace:
```json
{
  "python.defaultInterpreterPath": "C:/Users/sewell/.conda/envs/hwm14_env/python.exe"
}
```

---

## Reference

Drob, D. P., et al. (2015), An update to the Horizontal Wind Model (HWM): The quiet
time thermosphere, *Earth and Space Science*, 2,
doi:[10.1002/2014EA000089](https://doi.org/10.1002/2014EA000089)
