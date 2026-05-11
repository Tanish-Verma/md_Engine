# ~/md_Engine/PhaseTransition/run_one_point.py

import os
import sys
import argparse
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from md_engine import MDSimulation, SimulationConfig

N_CELLS = 4

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rho",    type=float, required=True)
    parser.add_argument("--T_star", type=float, required=True)
    parser.add_argument("--outdir", type=str,   required=True)
    args = parser.parse_args()

    rho    = args.rho
    T_star = args.T_star
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # Sanitize filenames: avoid floating point issues like 0.30000000000000004
    rho_str    = f"{rho:.4f}"
    T_star_str = f"{T_star:.4f}"

    results_file = os.path.join(outdir, f"result_rho{rho_str}_T{T_star_str}.h5")

    # Skip if already done (allows resubmission of failed jobs safely)
    if os.path.exists(results_file):
        print(f"[SKIP] {results_file} already exists.")
        return

    print(f"[START] rho={rho_str}, T*={T_star_str}", flush=True)
    t0 = time.time()

    config = SimulationConfig(
        n_cells      = N_CELLS,
        rho_star     = rho,
        t_star_values= [T_star],   # single temperature per job
        prod_steps   = 6000,
        n_workers    = 1,          # one worker, one CPU
        equil_steps  = 2000,
        save_positions= False,
        save_velocities= False,
    )

    sim = MDSimulation(config)
    sim.run(
        output_dir   = outdir,
        plot_rdf     = False,      # no plots during data collection
        plot_msd     = False,
        run_diag     = False,
        to_save      = True,
        results_name = os.path.basename(results_file),
    )

    elapsed = time.time() - t0
    print(f"[DONE] rho={rho_str}, T*={T_star_str} — {elapsed:.1f}s", flush=True)

if __name__ == "__main__":
    main()