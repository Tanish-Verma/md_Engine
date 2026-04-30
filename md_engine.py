import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from multiprocessing import Pool
from scipy.constants import k as k_B          # Boltzmann constant in J/K
import os


def prompt(label, default, cast=float):
    val = input(f"  {label} [{default}]: ").strip()
    return cast(val) if val else cast(default)




class MDSimulation:
    def __init__(self,
                 sigma=3.405,
                 eps= 0.0140 * 1.60218e-19,
                 n_cells=4,
                 rho_star=0.8442,
                 dt=0.005,
                 r_cutoff=2.5,
                 r_skin=3.0,
                 equil_steps=2000,
                 prod_steps=2000,
                 rescale_interval=10,
                 rdf_interval=10,
                 rdf_bins=200,
                 t_star_values=None,
                 animate_equil_steps=200,
                 animate_prod_steps=200,
                 animate_frame_skip=2,
                 animate_t_star_values=None,
                 n_workers=None):

        self.sigma            = sigma
        self.eps              = eps
        self.eps_kb           = eps / k_B          # ε/k_B in Kelvin
        self.n_cells          = n_cells
        self.rho_star         = rho_star
        self.dt               = dt
        self.r_cutoff         = r_cutoff
        self.r_skin           = r_skin
        self.r_cutoff_sq      = r_cutoff ** 2
        self.r_skin_sq        = r_skin ** 2
        self.thresh_sq        = (0.5 * (r_skin - r_cutoff)) ** 2
        self.equil_steps      = equil_steps
        self.prod_steps       = prod_steps
        self.rescale_interval = rescale_interval
        self.rdf_interval     = rdf_interval
        self.rdf_bins         = rdf_bins
        plot_rdf              = bool(input("\nPlot RDF after simulation? [y/n] (default: y): ").strip().lower() or "y" == "y")
        self.plot_rdf         = plot_rdf

        # T* sweep (reduced temperatures)
        self.t_star_values = (t_star_values
                              if t_star_values is not None
                              else [0.01, 0.2, 0.4, 0.6, 0.8, 1.0])

        if n_workers is None:
            cpu_count = os.cpu_count() or 1
            n_tasks = len(self.t_star_values)
            self.n_workers = min(n_tasks, cpu_count)
        else:
            self.n_workers = n_workers

        self.animate_equil_steps    = animate_equil_steps
        self.animate_prod_steps     = animate_prod_steps
        self.animate_frame_skip     = animate_frame_skip
        self.animate_t_star_values  = (animate_t_star_values
                                       if animate_t_star_values is not None
                                       else self.t_star_values)

        # Diagnostics arrays — populated from parallel worker results
        # keyed by T_star after run() completes
        self.diagnostics = {}   # T_star -> dict with kin_E, pot_E, tot_E, P

    # ------------------------------------------------------------------ constructors

    @classmethod
    def with_defaults(cls):
        print("\n[Using default parameters]")
        return cls()

    @classmethod
    def from_user_input(cls):
        print("\n[Press Enter to keep default]")

        print("\n--- Physical constants ---")
        sigma = prompt("sigma (Å)", 3.405)
        eps   = prompt("epsilon (J)  [Ar default: 2.183e-21]", 2.183e-21)
        print(f"  → eps/kB = {eps / k_B:.4f} K")

        print("\n--- System setup ---")
        n_cells  = prompt("unit cells per side", 4, int)
        rho_star = prompt("reduced density ρ*", 0.8442)

        print("\n--- Integration ---")
        dt = prompt("time step dt", 0.005)

        print("\n--- Neighbour list ---")
        r_cutoff = prompt("cutoff radius r_c", 2.5)
        r_skin   = prompt("skin radius r_skin", 3.0)

        print("\n--- Run length ---")
        equil_steps      = prompt("equilibration steps", 1000, int)
        prod_steps       = prompt("production steps", 1000, int)
        rescale_interval = prompt("velocity rescale every N steps", 10, int)
        rdf_interval     = prompt("RDF sample every N steps", 10, int)
        rdf_bins         = prompt("RDF histogram bins", 200, int)
        plot_rdf         = prompt("plot RDF? (1=yes, 0=no)", 1, int)
        plot_rdf         = bool(plot_rdf)

        print("\n--- Temperature sweep (T* = reduced temperature) ---")
        raw = input("  T* values space-separated [0.01 0.2 0.4 0.6 0.8 1.0]: ").strip()
        t_star_values = [float(t) for t in raw.split()] if raw else [0.01, 0.2, 0.4, 0.6, 0.8, 1.0]

        print("\n--- Parallel workers ---")
        n_workers = prompt("worker processes (0 = auto)", 0, int)
        n_workers = None if n_workers == 0 else n_workers

        print("\n--- Animation options ---")
        animate_equil_steps = prompt("animation equil steps per T*", 200, int)
        animate_prod_steps  = prompt("animation production steps per T*", 200, int)
        animate_frame_skip  = prompt("save every N-th frame", 2, int)

        return cls(
            sigma=sigma, eps=eps, n_cells=n_cells, rho_star=rho_star,
            dt=dt, r_cutoff=r_cutoff, r_skin=r_skin,
            equil_steps=equil_steps, prod_steps=prod_steps,
            rescale_interval=rescale_interval, rdf_interval=rdf_interval,
            rdf_bins=rdf_bins, plot_rdf=plot_rdf, t_star_values=t_star_values,
            animate_equil_steps=animate_equil_steps,
            animate_prod_steps=animate_prod_steps,
            animate_frame_skip=animate_frame_skip,
            n_workers=n_workers,
        )


    def lj_properties(self, r_sq):
        rc_sq   = self.r_cutoff_sq
        inv_rc2 = 1.0 / rc_sq
        phi_rc  = 4 * (inv_rc2 ** 6 - inv_rc2 ** 3)
        f_rc_r  = 48 * inv_rc2 * (inv_rc2 ** 6 - 0.5 * inv_rc2 ** 3)
        inv_r2  = 1.0 / r_sq
        inv_r6  = inv_r2 ** 3
        inv_r12 = inv_r2 ** 6
        f_over_r = 48 * inv_r2 * (inv_r12 - 0.5 * inv_r6) - f_rc_r
        pot      = 4 * (inv_r12 - inv_r6) - phi_rc + 0.5 * f_rc_r * (r_sq - rc_sq)
        return pot, f_over_r

    def calculate_forces(self, pos, neighbors, L):
        forces = np.zeros_like(pos)
        if not len(neighbors):
            return 0.0, forces
        i, j   = neighbors[:, 0], neighbors[:, 1]
        dr     = self.minimum_image(pos[i], pos[j], L)
        r_sq   = np.sum(dr ** 2, axis=1)
        mask   = r_sq < self.r_cutoff_sq
        pot, f_over_r = self.lj_properties(r_sq[mask])
        f_vecs = f_over_r[:, np.newaxis] * dr[mask]
        np.add.at(forces, i[mask],  f_vecs)
        np.add.at(forces, j[mask], -f_vecs)
        return np.sum(pot), forces

    @staticmethod
    def minimum_image(p1, p2, L):
        dr = p1 - p2
        return dr - L * np.round(dr / L)

    def generate_fcc_lattice(self):
        n, rho = self.n_cells, self.rho_star
        L      = (4 * n ** 3 / rho) ** (1 / 3)
        a      = L / n
        basis  = np.array([[0, 0, 0], [0.5, 0.5, 0],
                           [0.5, 0, 0.5], [0, 0.5, 0.5]]) * a
        pos    = [np.array([ix, iy, iz]) * a + b
                  for ix in range(n) for iy in range(n) for iz in range(n)
                  for b in basis]
        return np.array(pos), L

    def update_neighbor_list(self, pos, L):
        dr      = self.minimum_image(pos[:, np.newaxis, :], pos[np.newaxis, :, :], L)
        dist_sq = np.sum(dr ** 2, axis=2)
        i, j    = np.where(np.triu(dist_sq < self.r_skin_sq, k=1))
        return np.stack((i, j), axis=1)

    def verlet_step(self, pos, vel, f, L, nb):
        pos_new = (pos + vel * self.dt + 0.5 * f * self.dt ** 2) % L
        v_mid   = vel + 0.5 * f * self.dt
        pe, f_new = self.calculate_forces(pos_new, nb, L)
        return pos_new, v_mid + 0.5 * f_new * self.dt, pe, f_new

    @staticmethod
    def rescale_velocities(v, T):
        T_inst = np.sum(v ** 2) / (3 * len(v))
        return v * np.sqrt(T / T_inst) if T_inst > 0 else v

    def calculate_rdf(self, pos, L, dr_bin, r_max):
        N      = len(pos)
        dr_vec = self.minimum_image(pos[:, np.newaxis, :], pos[np.newaxis, :, :], L)
        r      = np.sqrt(np.sum(dr_vec ** 2, axis=2))[np.triu_indices(N, k=1)]
        hist, edges = np.histogram(r, bins=self.rdf_bins, range=(0, r_max))
        c = (edges[:-1] + edges[1:]) / 2.0
        return c, (2.0 * hist * L ** 3) / (N ** 2 * 4 * np.pi * c ** 2 * dr_bin)

    def needs_rebuild(self, pos, pos_ref, L):
        return np.max(np.sum(
            self.minimum_image(pos, pos_ref, L) ** 2, axis=1)) > self.thresh_sq

    # ------------------------------------------------------------------ run (parallel)
    @staticmethod
    def _run_one_temperature(args):
        """
        Run equilibration + production for a single T* and return the averaged g(r)
        plus full per-step energy and momentum traces for diagnostics.
        """
        (T_star, pos0, L, sigma, eps_kb,
        dt, r_cutoff, r_skin,
        equil_steps, prod_steps,
        rescale_interval, rdf_interval, rdf_bins,
        seed) = args

        rng = np.random.default_rng(seed)

        r_cutoff_sq = r_cutoff ** 2
        r_skin_sq   = r_skin ** 2
        thresh_sq   = (0.5 * (r_skin - r_cutoff)) ** 2
        r_max       = L / 2.0
        dr_bin      = r_max / rdf_bins

        # ---- local physics helpers ----

        def minimum_image(p1, p2):
            dr = p1 - p2
            return dr - L * np.round(dr / L)

        def lj_properties(r_sq):
            inv_rc2 = 1.0 / r_cutoff_sq
            phi_rc  = 4 * (inv_rc2 ** 6 - inv_rc2 ** 3)
            f_rc_r  = 48 * inv_rc2 * (inv_rc2 ** 6 - 0.5 * inv_rc2 ** 3)
            inv_r2  = 1.0 / r_sq
            inv_r6  = inv_r2 ** 3
            inv_r12 = inv_r2 ** 6
            f_over_r = 48 * inv_r2 * (inv_r12 - 0.5 * inv_r6) - f_rc_r
            pot      = 4 * (inv_r12 - inv_r6) - phi_rc + 0.5 * f_rc_r * (r_sq - r_cutoff_sq)
            return pot, f_over_r

        def calculate_forces(pos, neighbors):
            forces = np.zeros_like(pos)
            if not len(neighbors):
                return 0.0, forces
            i, j  = neighbors[:, 0], neighbors[:, 1]
            dr    = minimum_image(pos[i], pos[j])
            r_sq  = np.sum(dr ** 2, axis=1)
            mask  = r_sq < r_cutoff_sq
            pot, f_over_r = lj_properties(r_sq[mask])
            f_vecs = f_over_r[:, np.newaxis] * dr[mask]
            np.add.at(forces, i[mask],  f_vecs)
            np.add.at(forces, j[mask], -f_vecs)
            return np.sum(pot), forces

        def update_neighbor_list(pos):
            dr      = minimum_image(pos[:, np.newaxis, :], pos[np.newaxis, :, :])
            dist_sq = np.sum(dr ** 2, axis=2)
            ii, jj  = np.where(np.triu(dist_sq < r_skin_sq, k=1))
            return np.stack((ii, jj), axis=1)

        def verlet_step(pos, vel, f, nb):
            pos_new = (pos + vel * dt + 0.5 * f * dt ** 2) % L
            v_mid   = vel + 0.5 * f * dt
            pe, f_new = calculate_forces(pos_new, nb)
            return pos_new, v_mid + 0.5 * f_new * dt, pe, f_new

        def rescale_velocities(v, T):
            T_inst = np.sum(v ** 2) / (3 * len(v))
            return v * np.sqrt(T / T_inst) if T_inst > 0 else v

        def needs_rebuild(pos, pos_ref):
            return np.max(np.sum(minimum_image(pos, pos_ref) ** 2, axis=1)) > thresh_sq

        def calculate_rdf(pos):
            N      = len(pos)
            dr_vec = minimum_image(pos[:, np.newaxis, :], pos[np.newaxis, :, :])
            r      = np.sqrt(np.sum(dr_vec ** 2, axis=2))[np.triu_indices(N, k=1)]
            hist, edges = np.histogram(r, bins=rdf_bins, range=(0, r_max))
            c = (edges[:-1] + edges[1:]) / 2.0
            return c, (2.0 * hist * L ** 3) / (len(pos) ** 2 * 4 * np.pi * c ** 2 * dr_bin)

        # ---- simulation ----
        pos     = pos0.copy()
        pos_ref = pos.copy()
        N       = len(pos)
        vel     = rng.standard_normal((N, 3))
        vel    -= vel.mean(axis=0)
        vel     = rescale_velocities(vel, T_star)
        nb_list = update_neighbor_list(pos)
        _, force = calculate_forces(pos, nb_list)

        total_steps = equil_steps + prod_steps

        # Diagnostic arrays — one entry per step over the full run
        kin_E_trace = np.zeros(total_steps)
        pot_E_trace = np.zeros(total_steps)
        tot_E_trace = np.zeros(total_steps)
        P_trace     = np.zeros((total_steps, 3))   # total momentum per step

        T_K = T_star * eps_kb          # for printing only
        print(f"  [T* = {T_star:.3f} | T = {T_K:.2f} K]  equilibrating ({equil_steps} steps)...",
            flush=True)

        for step in range(equil_steps):
            pos, vel, pe, force = verlet_step(pos, vel, force, nb_list)
            if needs_rebuild(pos, pos_ref):
                nb_list = update_neighbor_list(pos)
                pos_ref = pos.copy()
            if step % rescale_interval == 0:
                vel = rescale_velocities(vel, T_star)

            # --- record diagnostics ---
            ke = 0.5 * np.sum(vel ** 2)
            kin_E_trace[step] = ke
            pot_E_trace[step] = pe
            tot_E_trace[step] = ke + pe
            P_trace[step]     = vel.sum(axis=0)   # total momentum (reduced units)

        print(f"  [T* = {T_star:.3f}]  production ({prod_steps} steps)...", flush=True)
        g_acc = np.zeros(rdf_bins)
        count = 0
        r_centers = None

        for step in range(prod_steps):
            global_step = equil_steps + step
            pos, vel, pe, force = verlet_step(pos, vel, force, nb_list)
            if needs_rebuild(pos, pos_ref):
                nb_list = update_neighbor_list(pos)
                pos_ref = pos.copy()

            # --- record diagnostics ---
            ke = 0.5 * np.sum(vel ** 2)
            kin_E_trace[global_step] = ke
            pot_E_trace[global_step] = pe
            tot_E_trace[global_step] = ke + pe
            P_trace[global_step]     = vel.sum(axis=0)

            if step % rdf_interval == 0:
                r_centers, g_curr = calculate_rdf(pos)
                g_acc += g_curr
                count += 1

        g_avg  = g_acc / count
        peak_g = float(np.max(g_avg))
        print(f"  [T* = {T_star:.3f}]  done — {count} RDF samples, g(r) peak = {peak_g:.2f}",
            flush=True)

        diag = {
            "kin_E": kin_E_trace,
            "pot_E": pot_E_trace,
            "tot_E": tot_E_trace,
            "P":     P_trace,
        }

        return T_star, r_centers, g_avg, peak_g, diag


    def run(self, output_file="melting_rdf.pdf", run_diag=True, diag_output_dir="diagnostics"):
        pos0, L = self.generate_fcc_lattice()
        N       = len(pos0)
        print(f"\nSystem: N = {N} atoms, L = {L:.4f} σ")
        print(f"ε/k_B = {self.eps_kb:.4f} K   (ε = {self.eps:.4e} J,  k_B = {k_B:.6e} J/K)")
        print(f"T* values: {self.t_star_values}")
        print(f"Equivalent T (K): {[round(t * self.eps_kb, 2) for t in self.t_star_values]}")
        print(f"Launching {len(self.t_star_values)} workers (n_workers={self.n_workers})...\n")

        # Build argument tuples for each temperature (one per worker process)
        worker_args = [
            (T_star, pos0, L, self.sigma, self.eps_kb,
             self.dt, self.r_cutoff, self.r_skin,
             self.equil_steps, self.prod_steps,
             self.rescale_interval, self.rdf_interval, self.rdf_bins,
             idx)                                   # seed = index for reproducibility
            for idx, T_star in enumerate(self.t_star_values)
        ]

        with Pool(processes=self.n_workers) as pool:
            results = pool.map(MDSimulation._run_one_temperature, worker_args)

        # results is a list of (T_star, r_centers, g_avg, peak_g, diag)
        # sort by T* so the legend is ordered
        results.sort(key=lambda x: x[0])

        # Store diagnostics on self, keyed by T_star
        for T_star, r_centers, g_avg, peak_g, diag in results:
            self.diagnostics[T_star] = diag

        if self.plot_rdf:
            self.plot_radial_distribution_function(output_file, results)

        if run_diag:
            run_diagnostics(self, output_dir=diag_output_dir)

        return results

    def plot_radial_distribution_function(self, output_file, results):
        plt.figure(figsize=(10, 6))
        for T_star, r_centers, g_avg, peak_g, diag in results:
            T_K   = T_star * self.eps_kb
            phase = _classify_phase(peak_g)
            plt.plot(r_centers * self.sigma, g_avg,
                     label=f"T* = {T_star:.2f}  ({T_K:.1f} K)  {phase}  peak={peak_g:.2f}")
            print(f"  T* = {T_star:.3f}  T = {T_K:.2f} K  →  {phase}  (g(r) peak = {peak_g:.2f})")

        plt.title("Radial Distribution Function $g(r)$ during Melting")
        plt.xlabel("Distance (Å)")
        plt.ylabel("$g(r)$")
        plt.axhline(1.0, color="black", linestyle="--", alpha=0.5)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        plt.show()
        print(f"\nComplete. Plot saved as '{output_file}'")


# ------------------------------------------------------------------ diagnostics

def run_diagnostics(sim, output_dir="diagnostics"):
    """
    Plot and save energy + momentum diagnostics for every temperature that was run.
    Each temperature gets its own pair of PNG files saved under `output_dir/`.
    A console summary (mean energy, fluctuation, max momentum) is printed for each T*.
    """
    if not sim.diagnostics:
        print("[Diagnostics] No data available. Run the simulation first.")
        return

    os.makedirs(output_dir, exist_ok=True)
    N_atoms   = 4 * sim.n_cells ** 3
    total_steps = sim.equil_steps + sim.prod_steps
    steps_arr   = np.arange(total_steps)

    print(f"\n{'='*60}")
    print(f"  DIAGNOSTICS  —  {N_atoms} atoms, {len(sim.diagnostics)} temperatures")
    print(f"  Output directory: '{output_dir}/'")
    print(f"{'='*60}")

    for T_star in sorted(sim.diagnostics):
        diag  = sim.diagnostics[T_star]
        T_K   = T_star * sim.eps_kb
        phase = _classify_phase(float(np.max(diag["tot_E"])))   # rough; peak_g not stored here
        tag   = f"T_star_{T_star:.3f}".replace(".", "p")

        kin_E = diag["kin_E"]
        pot_E = diag["pot_E"]
        tot_E = diag["tot_E"]
        P     = diag["P"]

        # ---- Energy plot ----
        fig_e, ax_e = plt.subplots(figsize=(10, 5))
        ax_e.plot(steps_arr, kin_E, label="Kinetic Energy",   color="blue",  alpha=0.6)
        ax_e.plot(steps_arr, pot_E, label="Potential Energy", color="green", alpha=0.6)
        ax_e.plot(steps_arr, tot_E, label="Total Energy",     color="red",   linewidth=2)
        ax_e.axvspan(0, sim.equil_steps, alpha=0.1, color="gray", label="Equilibration")
        ax_e.set_xlabel("Step")
        ax_e.set_ylabel("Energy (reduced units)")
        ax_e.set_title(f"Energy Conservation  |  T* = {T_star:.3f}  (T = {T_K:.2f} K)  |  {N_atoms} atoms")
        ax_e.legend()
        ax_e.grid(True, linestyle="--", alpha=0.5)
        fig_e.tight_layout()
        energy_file = os.path.join(output_dir, f"energy_{tag}.png")
        fig_e.savefig(energy_file, dpi=150)
        plt.close(fig_e)

        # ---- Momentum plot ----
        fig_p, ax_p = plt.subplots(figsize=(10, 4))
        P_mag = np.linalg.norm(P, axis=1)
        ax_p.plot(steps_arr, P_mag, label="|P| total", color="black", linewidth=1.5)
        ax_p.axvspan(0, sim.equil_steps, alpha=0.1, color="gray", label="Equilibration")
        ax_p.set_xlabel("Step")
        ax_p.set_ylabel("Total Momentum magnitude (reduced units)")
        ax_p.set_title(f"Momentum Conservation  |  T* = {T_star:.3f}  (T = {T_K:.2f} K)  |  {N_atoms} atoms")
        ax_p.legend()
        ax_p.grid(True, linestyle="--", alpha=0.5)
        fig_p.tight_layout()
        momentum_file = os.path.join(output_dir, f"momentum_{tag}.png")
        fig_p.savefig(momentum_file, dpi=150)
        plt.close(fig_p)

        # ---- Console summary ----
        prod_E   = tot_E[sim.equil_steps:]
        E_mean   = float(np.mean(prod_E))
        E_fluc   = float(np.std(prod_E))
        rel_fluc = E_fluc / abs(E_mean) if E_mean != 0 else float("inf")
        stability = "stable" if rel_fluc < 1e-4 else "UNSTABLE"
        P_max     = np.max(np.abs(P), axis=0)

        print(f"\n  T* = {T_star:.3f}  (T = {T_K:.2f} K)")
        print(f"    Mean Total Energy   : {E_mean:.6f}")
        print(f"    RMS Fluctuation     : {E_fluc:.6f}")
        print(f"    Relative fluctuation: {rel_fluc:.2e}  → numerically {stability}")
        print(f"    Max |Px|={P_max[0]:.4f}  |Py|={P_max[1]:.4f}  |Pz|={P_max[2]:.4f}")
        print(f"    Energy plot  → {energy_file}")
        print(f"    Momentum plot→ {momentum_file}")

    print(f"\n{'='*60}")
    print("Done running diagnostics.\n")



def _classify_phase(peak_g):
    if peak_g > 3.0:
        return "SOLID"
    elif peak_g > 1.8:
        return "LIQUID"
    return "GAS"


def animate(sim, output_file="md_engine",format = "both"):

    gif_file = output_file + ".gif"
    mp4_file = output_file + ".mp4"

    pos0, L  = sim.generate_fcc_lattice()
    N        = len(pos0)
    r_max    = L / 2.0
    dr_bin   = r_max / sim.rdf_bins
    skip     = sim.animate_frame_skip
    total_eq = sim.animate_equil_steps
    total_pr = sim.animate_prod_steps

    frames        = []
    finished_rdfs = []
    all_speeds    = []

    t_star_list = sim.animate_t_star_values
    print(f"\nAnimation  —  {len(t_star_list)} temperatures, "
          f"{total_eq} equil + {total_pr} prod steps, frame skip={skip}")

    pos     = pos0.copy()
    pos_ref = pos.copy()
    T_star  = t_star_list[0]
    vel     = np.random.normal(0.0, np.sqrt(T_star), (N, 3))
    vel    -= vel.mean(axis=0)
    nb_list = sim.update_neighbor_list(pos, L)
    _, force = sim.calculate_forces(pos, nb_list, L)

    for T_star in t_star_list:
        T_K  = T_star * sim.eps_kb
        vel  = sim.rescale_velocities(vel, T_star)
        print(f"  T* = {T_star:.3f}  (T = {T_K:.1f} K)  equilibrating...", end="", flush=True)

        for step in range(total_eq):
            pos, vel, _, force = sim.verlet_step(pos, vel, force, L, nb_list)
            if sim.needs_rebuild(pos, pos_ref, L):
                nb_list, pos_ref = sim.update_neighbor_list(pos, L), pos.copy()
            if step % sim.rescale_interval == 0:
                vel = sim.rescale_velocities(vel, T_star)

        print(" recording...", end="", flush=True)
        g_acc = np.zeros(sim.rdf_bins)
        count = 0

        for step in range(total_pr):
            pos, vel, _, force = sim.verlet_step(pos, vel, force, L, nb_list)
            if sim.needs_rebuild(pos, pos_ref, L):
                nb_list, pos_ref = sim.update_neighbor_list(pos, L), pos.copy()

            if step % skip == 0:
                speed    = np.sqrt(np.sum(vel ** 2, axis=1))
                r_centers, g_curr = sim.calculate_rdf(pos, L, dr_bin, r_max)
                g_acc   += g_curr
                count   += 1
                g_running = g_acc / count
                all_speeds.append(speed)
                frames.append((pos[:, :2].copy(), speed, r_centers, g_running,
                               T_star, T_K, float(np.max(g_running))))

        finished_rdfs.append((T_star, T_K, r_centers, g_acc / count))
        print(f" {count} frames.")

    s_concat     = np.concatenate(all_speeds)
    s_min, s_max = float(np.percentile(s_concat, 2)), float(np.percentile(s_concat, 98))
    total_frames = len(frames)
    n_t          = len(t_star_list)
    print(f"\nTotal frames: {total_frames}.  Building figure...")

    fig = plt.figure(figsize=(11, 5.5), facecolor="#0d0d0d")
    fig.set_size_inches(11, 5.6)
    gs  = fig.add_gridspec(1, 2, wspace=0.3, left=0.05, right=0.97, top=0.80, bottom=0.13)
    ax_slab = fig.add_subplot(gs[0])
    ax_rdf  = fig.add_subplot(gs[1])

    for ax in (ax_slab, ax_rdf):
        ax.set_facecolor("#111111")
        ax.tick_params(colors="0.7")
        for sp in ax.spines.values():
            sp.set_color("0.3")

    ax_slab.set_xlim(0, L)
    ax_slab.set_ylim(0, L)
    ax_slab.set_aspect("equal")
    ax_slab.set_xlabel("x (σ)", color="0.7")
    ax_slab.set_ylabel("y (σ)", color="0.7")
    slab_title = ax_slab.set_title("", color="white", fontsize=11)

    init_xy, init_spd = frames[0][0], frames[0][1]
    sc   = ax_slab.scatter(init_xy[:, 0], init_xy[:, 1],
                           c=init_spd, cmap="plasma", vmin=s_min, vmax=s_max,
                           s=18, alpha=0.9, linewidths=0)
    cbar = fig.colorbar(sc, ax=ax_slab, pad=0.02)
    cbar.set_label("Speed |v| (σ/t*)", color="0.7", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="0.7")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="0.7")

    ax_rdf.set_xlim(0, r_max * sim.sigma)
    ax_rdf.set_ylim(0, 5.5)
    ax_rdf.set_xlabel("r  (Å)", color="0.7")
    ax_rdf.set_ylabel("g(r)", color="0.7")
    ax_rdf.axhline(1.0, color="0.4", ls="--", lw=0.8)
    rdf_title = ax_rdf.set_title("", color="white", fontsize=11)

    ghost_cmap      = plt.cm.cool
    live_line_rdf,  = ax_rdf.plot([], [], color="#ff6b35", lw=1.8, zorder=5)
    ghost_lines_rdf = []
    ghost_handles   = []
    drawn_ghosts    = set()

    main_title       = fig.suptitle("", color="white", fontsize=14, y=0.96)
    explanation_text = fig.text(0.5, 0.90, "", color="#a8dadc", fontsize=11, ha="center")

    phase_colors = {"SOLID": "#3498db", "LIQUID": "#e74c3c", "GAS": "#2ecc71"}

    def update(fi):
        atom_xy, speed, r_centers, g_running, T_star_f, T_K_f, peak_g = frames[fi]
        phase = _classify_phase(peak_g)

        sc.set_offsets(atom_xy)
        sc.set_array(speed)
        slab_title.set_text(f"Atoms (xy)  |  T* = {T_star_f:.3f}  ({T_K_f:.1f} K)")

        for idx, (T_g, T_K_g, r_g, g_g) in enumerate(finished_rdfs):
            if T_g < T_star_f and idx not in drawn_ghosts:
                col = ghost_cmap(idx / max(n_t - 1, 1))
                gl, = ax_rdf.plot(r_g * sim.sigma, g_g,
                                  color=col, lw=1.0, alpha=0.4,
                                  label=f"T*={T_g:.2f}", zorder=2)
                ghost_lines_rdf.append(gl)
                ghost_handles.append(gl)
                drawn_ghosts.add(idx)

        live_line_rdf.set_data(r_centers * sim.sigma, g_running)
        live_line_rdf.set_label(f"T*={T_star_f:.2f} (live)")
        rdf_title.set_text("g(r) running avg")
        ax_rdf.legend(handles=ghost_handles + [live_line_rdf],
                      loc="upper right", fontsize=7,
                      framealpha=0.2, labelcolor="white")

        explanation_text.set_text(f"g(r) peak={peak_g:.2f}")
        main_title.set_text(f"Melting & Diffusion  |  PHASE: {phase}")
        main_title.set_color(phase_colors.get(phase, "white"))

        return (sc, live_line_rdf, slab_title, rdf_title, main_title, explanation_text)

    ani = FuncAnimation(fig, update, frames=total_frames, interval=60, blit=False)

    if format in ("gif", "both"):
        print(f"\nSaving GIF  → '{gif_file}' ({total_frames} frames) ...")
        try:
            ani.save(gif_file, writer="pillow", fps=20, dpi=130)
            print(f"  ✓ GIF saved: '{gif_file}'")
        except Exception as e:
            print(f"  ✗ GIF failed: {e}")

# ---- MP4 ----
    if format in ("mp4", "both"):
        print(f"Saving MP4  → '{mp4_file}' ({total_frames} frames) ...")
        try:
            from matplotlib.animation import FFMpegWriter
            mp4_writer = FFMpegWriter(
                fps=20,
                codec="libx264",
                extra_args=[
                    "-pix_fmt", "yuv420p",
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",  # 👈 FIX
                    "-crf", "18",
                    "-preset", "fast"
                ]
            )
            ani.save(mp4_file, writer=mp4_writer, dpi=130)
            print(f"  ✓ MP4 saved: '{mp4_file}'")
        except Exception as e:
            print(f"  ✗ MP4 failed: {e}")
    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    print("=" * 55)
    print("   Lennard-Jones MD Simulation — RDF Analysis")
    print("=" * 55)
    print(f"   scipy k_B = {k_B:.6e} J/K")
    print("=" * 55)
    print("\nParameter mode:\n  [1] Run Full Simulation (RDF Analysis)\n  [2] Run Animation Sweep")

    while True:
        choice = input("\nSelect option (1 or 2): ").strip()
        if choice in ("1", "2"):
            break
        print("  Please enter 1 or 2.")

    # Ask whether to run diagnostics
    if choice == "1":
        while True:
            diag_choice = input("\nRun diagnostics after simulation? [y/n] (default: y): ").strip().lower()
            if diag_choice in ("", "y", "yes"):
                run_diag = True
                break
            elif diag_choice in ("n", "no"):
                run_diag = False
                break
            print("  Please enter y or n.")

        diag_dir = "diagnostics"
        if run_diag:
            raw_dir = input(f"  Diagnostics output folder [diagnostics]: ").strip()
            diag_dir = raw_dir if raw_dir else "diagnostics"

    sim = MDSimulation.with_defaults()
    if choice == "1":
        sim.run(run_diag=run_diag, diag_output_dir=diag_dir)
        if run_diag:
            run_diagnostics(sim, output_dir=diag_dir)
    else:
        while True:
            fmt = input("\nOutput format? [gif/mp4/both] (default: both): ").strip().lower()
            if fmt in ("", "both"):
                fmt = "both"
                break
            elif fmt in ("gif", "mp4"):
                break
            print("  Please enter gif, mp4, or both.")
        animate(sim,format = fmt)