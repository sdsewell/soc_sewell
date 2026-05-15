# S_single_v2 — invert_single_frame.py: Wire H06 Full Pipeline
## WindCube SOC Pipeline — Specification v1.0
**Spec ID:** S_single_v2  
**Spec file:** `specs/S_single_v2_invert_single_frame_2026-05-14.md`  
**Date:** 2026-05-14  
**Status:** Authoritative  
**Script:** `scripts/invert_single_frame.py`

**Depends on:**
- `windcube/fpi_pipeline.py` (S_L01) — `process_science_frame()`
- `windcube/wind_retrieval.py` (H07) — `compute_los_geometry()`

---

## 1. Purpose and scope

This spec adds H06 fringe fitting as the **preferred** v_rel source in
`invert_single_frame.py`, while preserving `--v-rel` as a fallback.

**What changes:**
- New `--use-h06` flag (default: auto — active when `--v-rel` not supplied)
- Stage 2 (v_rel recovery) gains an H06 path using `process_science_frame()`
- Stage 3 (H07 geometry) must run **before** H06 to provide `v_los_prior_ms`
  so the execution order becomes: ingest → dark → geometry → H06 → correct
- A new H06 diagnostic panel is added to the printed output
- The `--v-rel` override and all existing output are fully preserved

**What does NOT change:**
- All CLI arguments (one new optional flag added)
- `_make_plots()` — unchanged
- All printed output blocks — new H06 block inserted between velocity
  decomposition header and the v_rel lines
- Dark subtraction logic — unchanged

---

## 2. New CLI argument

Add one new optional argument after `--sigma-v`:

```python
parser.add_argument(
    "--use-h06",
    action="store_true",
    default=False,
    help=(
        "Use H06 fringe fitting to recover v_rel from raw pixels. "
        "Requires cal frames in the same folder as the science frame. "
        "Activated automatically when --v-rel is not supplied."
    ),
)
```

**Activation logic** (in `_run()`, before Stage 2):

```python
use_h06 = args.use_h06 or (args.v_rel is None)
if use_h06 and args.v_rel is not None:
    print("NOTE: --v-rel supplied — using override, ignoring --use-h06")
    use_h06 = False
```

---

## 3. Reordered processing stages

The stage order changes from:
```
Ingest → Dark → v_rel (Stage 2) → Geometry (Stage 3) → Correct (Stage 4)
```
To:
```
Ingest → Dark → Geometry (Stage 3) → H06/v_rel (Stage 2) → Correct (Stage 4)
```

Stage 3 (geometry) must run first because H06 needs `v_los_prior_ms`.
Stage 2 is renamed and expanded but still falls back to `--v-rel`.

The existing Stage 3 code block is moved to run immediately after dark
subtraction, before Stage 2. This is a code reorder only — no logic changes
to any geometry code.

---

## 4. H06 cal frame discovery

Add a helper function to find cal frames in the same folder:

```python
def _find_cal_frames(science_path: Path) -> list[Path]:
    """
    Find *_cal_*.bin files in the same folder as the science frame.
    Returns sorted list of Path objects. Empty list if none found.
    """
    return sorted(science_path.parent.glob("*_cal*.bin"))
```

Add a helper to find and load dark frames for master dark construction:

```python
def _build_master_dark_from_folder(science_path: Path, lua_timestamp_ms: int):
    """
    Find all *_dark.bin files in the same folder, load them, and return
    a float32 master dark array (sum of all dark frames) plus the count.

    Returns (master_dark_array, n_dark_frames) or (None, 0) if no darks found.
    Uses ingest_real_image() to load pixels.
    """
```

---

## 5. Updated Stage 2 — v_rel recovery with H06

Replace the current Stage 2 block with:

```python
    # ── Stage 2 — v_rel recovery ─────────────────────────────────────────────
    # Priority order:
    #   1. H06 fringe fitting (if use_h06=True and cal frames available)
    #   2. --v-rel override (if supplied)
    #   3. Not available (print guidance)
    #
    # NOTE: geom must be computed before this stage (see Stage 3 below).
    #       v_los_prior_ms = geom.V_sc_LOS + geom.v_earth_LOS

    v_rel = None
    sigma_v = args.sigma_v
    airglow_result = None   # AirglowResult if H06 was run, else None

    if use_h06:
        cal_frames = _find_cal_frames(path)
        if not cal_frames:
            print(
                "WARNING: H06 requested but no *_cal*.bin files found in "
                f"{path.parent}\n"
                "         Falling back to --v-rel if supplied."
            )
        else:
            # Build per-folder master calibration from all available cal frames
            print(f"H06 mode: found {len(cal_frames)} cal frame(s) in folder")
            try:
                from windcube.fpi_pipeline import (
                    process_cal_frame,
                    average_calibrations,
                    process_science_frame,
                )
                from src.metadata.p01_image_metadata_2026_04_06 import ingest_real_image as _ingest

                # Dark-subtract and invert each cal frame
                cal_results = []
                master_dark_arr, n_dark = _build_master_dark_from_folder(
                    path, meta.lua_timestamp
                )
                for cal_path in cal_frames:
                    try:
                        _, cal_pixels = _ingest(cal_path,
                                                h_target_km_obs=args.h_target_km)
                        if master_dark_arr is not None and n_dark > 0:
                            cal_ds = (cal_pixels.astype(np.float32)
                                      - master_dark_arr / n_dark)
                        else:
                            cal_ds = cal_pixels.astype(np.float32)
                        cal_result = process_cal_frame(cal_ds, r_max_px=110.0)
                        cal_results.append(cal_result)
                        print(f"  Cal: {cal_path.name}  "
                              f"chi2={cal_result.chi2_red:.3f}  "
                              f"converged={cal_result.converged}")
                    except Exception as exc:
                        print(f"  WARNING: Cal frame failed: {cal_path.name}: {exc}")

                if not cal_results:
                    print("WARNING: All cal frames failed — falling back to --v-rel")
                else:
                    master_cal = average_calibrations(cal_results)
                    print(
                        f"Master cal: {len(cal_results)} frames averaged  "
                        f"t_m={master_cal.t_m*1e3:.6f} mm  "
                        f"epsilon_cal={master_cal.epsilon_cal:.6f}"
                    )

                    # v_los_prior from geometry (already computed in Stage 3)
                    v_los_prior = (geom.V_sc_LOS + geom.v_earth_LOS
                                   if geom is not None else 0.0)

                    # Dark-subtract science frame
                    if master_dark_arr is not None and n_dark > 0:
                        pixels_ds = np.clip(
                            image.astype(np.float32) - master_dark_arr / n_dark,
                            0, 16383,
                        ).astype(np.float32)
                    else:
                        pixels_ds = image.astype(np.float32)

                    # H06 inversion
                    airglow_result = process_science_frame(
                        pixels_ds      = pixels_ds,
                        master_cal     = master_cal,
                        v_los_prior_ms = v_los_prior,
                        r_max_px       = 110.0,
                    )
                    v_rel   = airglow_result.v_rel_ms
                    sigma_v = airglow_result.sigma_v_ms
                    print(
                        f"H06 result: v_rel={v_rel:+.2f} m/s  "
                        f"sigma_v={sigma_v:.2f} m/s  "
                        f"chi2={airglow_result.chi2_red:.3f}  "
                        f"converged={airglow_result.converged}"
                    )
            except Exception as exc:
                print(f"WARNING: H06 pipeline failed: {exc}")
                print("         Falling back to --v-rel if supplied.")

    # --v-rel override (fallback or explicit)
    if v_rel is None and args.v_rel is not None:
        v_rel  = args.v_rel
        sigma_v = args.sigma_v

    if v_rel is None:
        print("WARNING: v_rel not available.")
        print("  H06 requires cal frames in the same folder, OR")
        print("  supply --v-rel <value> to proceed past this stage.")
```

---

## 6. H06 diagnostic panel in printed output

After the existing `[ VELOCITY DECOMPOSITION (H07 Stage C) ]` header and
V_sc_LOS / v_earth_LOS lines, but **before** the v_rel line, insert a new
diagnostic panel when `airglow_result` is not None:

```python
        if airglow_result is not None:
            print()
            print("[ H06 FRINGE FIT ]")
            print(f"  chi2_red   : {airglow_result.chi2_red:.4f}")
            print(f"  converged  : {airglow_result.converged}")
            print(f"  scan_ambig : {airglow_result.scan_ambiguous}")
            print(f"  n_bins     : {airglow_result.n_bins}")
            print(f"  lc_m       : {airglow_result.lc_m*1e9:.7f} nm")
            print(f"  Y_line     : {airglow_result.Y_line:.2f} ADU")
            print(f"  B_sci      : {airglow_result.B_sci:.2f} ADU")
            budget = "✓" if airglow_result.budget_ok else "✗"
            print(f"  sigma_v    : {airglow_result.sigma_v_ms:.3f} m/s  "
                  f"{budget} STM budget (≤9.8 m/s)")
```

---

## 7. Module docstring update

Add to the module docstring:

```
H06 mode (default when --v-rel not supplied):
  v_rel is recovered from raw pixels via H06 fringe fitting.
  Requires *_cal*.bin files in the same folder.
  Builds a master calibration from all available cal frames,
  then runs process_science_frame() for the science frame.
  Falls back to --v-rel if no cal frames found or H06 fails.
```

---

## 8. File location

```
soc_sewell/
└── scripts/
    └── invert_single_frame.py   ← edit in place
```

---

## 9. Instructions for Claude Code

Read this entire spec and the current `scripts/invert_single_frame.py`
before writing any code.

**Step-by-step:**

1. Add `--use-h06` to `_parse_args()` (spec §2).

2. Add `_find_cal_frames()` and `_build_master_dark_from_folder()` (spec §4).
   For `_build_master_dark_from_folder`: scan for *_dark.bin files in
   the same folder, load each with ingest_real_image(), sum the pixel
   arrays as float32, return (master_dark_sum, n_dark).

3. Reorder stages in `_run()` (spec §3):
   Move the Stage 3 geometry block to run BEFORE Stage 2.
   This means geometry is attempted first, then v_rel recovery.
   The geometry result (geom) may be None if it fails — the H06 path
   handles this by using v_los_prior=0.0 as fallback.

4. Replace Stage 2 with the H06 path (spec §5).
   CRITICAL: preserve the `--v-rel` fallback exactly.
   CRITICAL: the `airglow_result` variable is initialised to None.
   Only set it if H06 runs successfully.

5. Add H06 diagnostic panel to printed output (spec §6).
   Insert it in the correct position within the existing output block.

6. Update module docstring (spec §7).

7. Run smoke test:
   ```
   python scripts/invert_single_frame.py --help
   ```
   Confirm `--use-h06` is present.

8. Run pytest tests/ -v — no regressions.

9. Commit:
   ```
   feat(single): wire H06 pipeline into invert_single_frame.py
   Implements: S_single_v2_invert_single_frame_2026-05-14.md
   --v-rel fallback fully preserved. H06 mode default when --v-rel absent.
   ```

**Do not:**
- Change `_make_plots()` or any plot logic
- Change any printed output except adding the H06 diagnostic panel
- Remove or rename any existing function or argument

---

*End of S_single_v2 specification v1.0 — 2026-05-14*
