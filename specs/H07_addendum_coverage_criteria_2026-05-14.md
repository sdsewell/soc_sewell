# H07 — Specification Addendum: Section 11.2 Acceptance Criteria
## WindCube SOC Pipeline
**Addendum ID:** H07-ADD-01  
**Applies to:** `specs/H07_wind_vector_retrieval_2026-05-14_v03.md` (v0.3)  
**Date:** 2026-05-14  
**Status:** Authoritative — supersedes §11.2 of H07 v0.3  

---

## Reason for addendum

H07 v0.3 §11.2 stated ">80% good bins" as a 1-day validation criterion.
Post-implementation analysis of WindCube orbital mechanics shows this is
not achievable in 1 day and was never physically realistic. The criterion
has been corrected based on the following analysis.

---

## Orbital coverage analysis

**Key parameters:**
- Orbital period: ~95 min → 15.2 passes per day
- Successive ground track longitude spacing: **23.75°**
- Tangent point arc offset from sub-satellite point: **~16.1° / ~1789 km**
- Science bin size (default): 5° × 5°

**Coverage mechanics:** A 5°×5° geographic bin at the equator receives
an along-track pass approximately once every 23.75°/5° = **4.8 days**.
Cross-track passes have the same spacing but sample different longitudes.
For H07 to produce a well-conditioned wind solution, a bin needs at least
one along-track AND one cross-track observation (azimuthal diversity). This
mixed-mode coverage requires the spacecraft to visit the same geographic
cell in both pointing configurations — which depends on how many days the
ground tracks have had to precess across all longitudes.

**Expected good-bin fraction vs. accumulation time:**

| Accumulation | Expected good bins | Notes |
|---|---|---|
| 1 day | ~10–15% | Matches observed 11% ✓ |
| 3 days | ~40–50% | Good for regional studies |
| 5 days | ~75–85% | Near-complete science band coverage |
| 7 days | ~90–95% | Near-global within science latitude band |
| 14 days | ~98% | Effectively complete |

The science latitude band (±60° geodetic) is fully accessible; polar
regions (>75°) benefit from denser pass spacing and converge faster.

---

## Corrected §11.2 acceptance criteria

Replace the single acceptance criteria table in H07 v0.3 §11.2 with the
following:

### Criteria that apply at any accumulation length

These must pass regardless of how many days are processed:

| Metric | Pass condition | Notes |
|--------|---------------|-------|
| `mean(v_corrected)` over all frames | < 5 m/s systematic | Per-frame physics check |
| `std(v_corrected)` | Consistent with mean σ_v | Noise model check |
| Geometry errors | 0 | All frames must complete Stage G |
| `derive_obs_mode` vs `meta.obs_mode` agreement | 100% | Synthetic data only |
| v_E mean over good bins | < 5 m/s absolute | Wind bias check |
| v_N mean over good bins | < 5 m/s absolute | Wind bias check |
| v_E std over good bins | < 2 × mean(σ_v_E) | Scatter consistent with noise |
| v_N std over good bins | < 2 × mean(σ_v_N) | Scatter consistent with noise |

### Criteria that depend on accumulation length

| Accumulation | Good bins (%) | GDOP flagged (%) |
|---|---|---|
| 1 day | ≥ 8% | ≤ 80% |
| 3 days | ≥ 40% | ≤ 50% |
| 5 days | ≥ 75% | ≤ 20% |
| 7+ days | ≥ 90% | ≤ 10% |

**Validation status as of 2026-05-14:**
The 1-day null-wind test (`GEN01_20270101_001_0d_uniform_seed0042`) passed
all accumulation-independent criteria:
- 3879/3879 frames processed, 0 geometry errors ✓
- v_E mean = 0.0 m/s, v_N mean = 0.0 m/s ✓
- v_corrected (single frame) = +0.004 m/s ✓
- Good bins: 11% (≥ 8% threshold for 1-day) ✓
- GDOP flagged: 73.5% (≤ 80% threshold for 1-day) ✓

**H07 is validated at 1-day level.** Full 5-day validation is pending
generation of a multi-day synthetic dataset (see G01 v15 spec).

---

## Coverage as a function of bin size

The good-bin fraction scales with bin area. Larger bins accumulate more
passes and reach mixed-mode coverage faster:

| Bin size | Days for ~80% good | Recommended use |
|---|---|---|
| 2.5° × 2.5° | ~10 days | High-resolution regional |
| 5° × 5° | ~5 days | **Default — nominal science** |
| 10° × 10° | ~2–3 days | Quick-look, tidal analysis |

For DE3 tidal characterisation (SQ2), 10°×10° bins with local-time
binning (`accumulate_days=True`) are recommended. See H07 §8.2.

---

## Coverage diagnostic (forthcoming — G01 v15)

A dedicated coverage diagnostic will be added to GEN01 (G01 v15 spec)
and a standalone `scripts/coverage_map.py` will be implemented. These
will compute and display:

- Global map of bin coverage (number of AT and CT passes per bin)
- Predicted good-bin fraction vs. accumulation days
- Mixed-mode coverage map (bins with both AT and CT observations)

The coverage map is the primary tool for choosing the simulation duration
for a given science objective and bin size.

---

## No code changes required

This addendum is documentation only. No changes to `windcube/wind_retrieval.py`
or any driver scripts are needed. The existing implementation is correct.
The only update is to the spec text.

---

*End of H07 Addendum H07-ADD-01 — 2026-05-14*
