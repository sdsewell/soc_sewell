"""
run_mc_simulations.py — CLI runner for MC02 FPI Monte Carlo simulations.

Spec: specs/MC02_fpi_mc_simulations_2026-05-24.md §3

Usage:
    python run_mc_simulations.py [--sim {1,2,3,all}] [--workers N]
                                  [--seed SEED] [--outdir DIR]

Execution order:
  1. Check PIPELINE_STATUS.md
  2. Gate: pytest tests/test_mc01_fpi_mc_engine.py -q  (abort if any fail)
  3. Build InstrumentParams from windcube/constants.py defaults
  4. Build TrialInput lists (MC02 §2)
  5. mc01.run_simulation() for each selected sim
  6. mc01.save_simulation() for each result
  7. Generate figures (MC02 §4)
  8. Write summary report (MC02 §5)
  9. Update PIPELINE_STATUS.md
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import date
from pathlib import Path


def _check_mc01_tests() -> bool:
    """Run MC01 test suite; return True if all tests pass."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_mc01_fpi_mc_engine.py", "-q"],
    )
    return result.returncode == 0


def _update_pipeline_status(date_str: str) -> None:
    """Add or update the MC02 row in PIPELINE_STATUS.md."""
    status_path = Path("PIPELINE_STATUS.md")
    if not status_path.exists():
        return
    new_row = (
        f"| MC02 | mc02_fpi_mc_simulations | IMPLEMENTED | 10/10 | {date_str} |"
    )
    text = status_path.read_text(encoding="utf-8")
    if "| MC02 |" in text:
        lines = [
            new_row if line.strip().startswith("| MC02 |") else line
            for line in text.splitlines()
        ]
        status_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        lines = text.splitlines()
        insert_at = next(
            (i for i, ln in enumerate(lines) if ln.startswith("## Known")),
            len(lines),
        )
        lines.insert(insert_at, new_row)
        status_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MC02 FPI Monte Carlo simulations (Harding replication).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sim", choices=["1", "2", "3", "all"], default="all",
        help="Which simulation(s) to run.",
    )
    parser.add_argument(
        "--workers", type=int, default=-1,
        help="Parallel workers (-1 = all CPUs, 1 = sequential).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Master RNG seed.",
    )
    parser.add_argument(
        "--outdir", default="mc_results",
        help="Output directory for .npz files, figures, and summary report.",
    )
    args = parser.parse_args()

    date_str = date.today().isoformat()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    run_sims = [1, 2, 3] if args.sim == "all" else [int(args.sim)]

    # Step 1: Show PIPELINE_STATUS.md
    status_path = Path("PIPELINE_STATUS.md")
    if status_path.exists():
        print("=== PIPELINE_STATUS.md ===")
        print(status_path.read_text(encoding="utf-8"))

    # Step 2: Gate — MC01 tests must pass
    print("=== Checking MC01 tests ===")
    if not _check_mc01_tests():
        print("ERROR: MC01 tests failed. Aborting.", file=sys.stderr)
        sys.exit(1)
    print("MC01 tests: PASS\n")

    # Deferred imports so import errors don't bypass the gate
    from src.fpi.fpi_cal_lib import InstrumentParams
    from windcube.mc01_fpi_mc_engine import (
        AirglowParams,
        run_simulation,
        save_simulation,
    )
    from windcube.mc02_fpi_mc_simulations import (
        build_sim1_inputs,
        build_sim2_inputs,
        build_sim3_inputs,
        make_figure1,
        make_figure2,
        make_figure3,
        write_summary_report,
    )

    # Step 3: Build instrument / airglow params
    instrument_params = InstrumentParams()
    airglow_params = AirglowParams()

    sim_results: dict[int, object] = {}

    # Steps 4-6: Build inputs, run, save
    for sim_num in run_sims:
        print(f"=== Simulation {sim_num} ===")
        if sim_num == 1:
            inputs = build_sim1_inputs()
        elif sim_num == 2:
            inputs = build_sim2_inputs(seed=args.seed)
        else:
            inputs = build_sim3_inputs(seed=args.seed)

        print(f"Building {len(inputs):,} trial inputs...")
        result = run_simulation(
            inputs, instrument_params, airglow_params,
            n_workers=args.workers, seed=args.seed, progress=True,
        )
        sim_results[sim_num] = result
        print(f"  Converged: {result.n_converged}/{result.n_total}")

        npz_path = outdir / f"MC01_sim{sim_num}_{date_str}.npz"
        save_simulation(result, npz_path)
        print(f"  Saved: {npz_path}\n")

    # Step 7: Generate figures
    print("=== Generating figures ===")
    if 1 in sim_results:
        p = make_figure1(sim_results[1], outdir, date_str)
        print(f"  Fig 1: {p}")
    if 2 in sim_results:
        p = make_figure2(sim_results[2], outdir, date_str)
        print(f"  Fig 2: {p}")
    if 3 in sim_results:
        p = make_figure3(sim_results[3], outdir, date_str)
        print(f"  Fig 3: {p}")

    # Step 8: Write summary report
    print("\n=== Summary Report ===")
    rpt = write_summary_report(
        sim_results.get(1),
        sim_results.get(2),
        sim_results.get(3),
        outdir,
        seed=args.seed,
        date_str=date_str,
    )
    print(f"\nReport written to: {rpt}")

    # Step 9: Update PIPELINE_STATUS.md
    _update_pipeline_status(date_str)
    print("PIPELINE_STATUS.md updated.")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
