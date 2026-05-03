import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from multiprocessing import Pool
from scipy.constants import k as k_B
from dataclasses import dataclass, field
from typing import Optional
import os


# ── DATA STRUCTURES ───────────────────────────────────────────────────────────

@dataclass
class SimulationConfig:
    """Configuration parameters for MD simulation. Defaulted to Krypton parameters."""
    # Material parameters
    sigma:          float         = 3.65
    eps:            float         = 0.0140

    # System setup
    n_cells:        int           = 4
    rho_star:       float         = 0.8442
    cell_param_a:   Optional[float] = field(default=None)

    # Integration
    dt:             float         = 0.005
    r_cutoff:       float         = 2.5
    r_skin:         float         = 3.0

    # Run length
    equil_steps:      int         = 2000
    prod_steps:       int         = 4000
    rescale_interval: int         = 10
    rdf_interval:     int         = 10
    rdf_bins:         int         = 200

    # Analysis
    plot_rdf:         bool        = True
    t_star_values:    list        = field(default_factory=lambda: [0.01, 0.2, 0.4, 0.6, 0.8, 1.0])
    n_workers:        Optional[int] = field(default=None)

    # Animation
    animate_equil_steps:   int          = 2000
    animate_prod_steps:    int          = 2000
    animate_frame_skip:    int          = 10
    animate_t_star_values: Optional[list] = field(default=None)


# CLI 

class CLI:
    """Command-line interface for MD simulation setup and control."""

    INERT_GASES = {
        "Ar": {"name": "Argon",   "sigma": 3.40, "epsilon": 0.0104},
        "Kr": {"name": "Krypton", "sigma": 3.65, "epsilon": 0.0140},
        "Xe": {"name": "Xenon",   "sigma": 3.98, "epsilon": 0.0200},
        "Ne": {"name": "Neon",    "sigma": 2.74, "epsilon": 0.0031},
    }

    @staticmethod
    def prompt(label, default, cast=float):
        """Prompt user for input with a default value."""
        val = input(f"  {label} [{default}]: ").strip()
        return cast(val) if val else cast(default)

    @classmethod
    def display_inert_gas_table(cls):
        """Display available inert gases and their LJ parameters."""
        print("\n" + "=" * 70)
        print("  INERT GAS DATABASE  (Lennard-Jones Parameters)")
        print("=" * 70)
        print(f"{'Key':<6} {'Gas':<12} {'σ (Å)':<12} {'ε (eV)':<12}")
        print("-" * 70)
        for key, props in cls.INERT_GASES.items():
            print(f"{key:<6} {props['name']:<12} {props['sigma']:<12.2f} {props['epsilon']:<12.4f}")
        print("=" * 70 + "\n")

    @classmethod
    def select_inert_gas(cls):
        """Allow user to select an inert gas or enter custom parameters."""
        cls.display_inert_gas_table()
        while True:
            choice = input("  Select gas [Ar/Kr/Xe/Ne] or [custom]: ").strip()
            if choice in cls.INERT_GASES:
                gas = cls.INERT_GASES[choice]
                print(f"\n  ✓ Selected {gas['name']}")
                print(f"    σ = {gas['sigma']} Å")
                print(f"    ε = {gas['epsilon']} eV\n")
                return gas["sigma"], gas["epsilon"]
            elif choice.lower() == "custom":
                print("\n  [Custom parameters]")
                sigma = cls.prompt("sigma (Å)", 3.40)
                eps   = cls.prompt("epsilon (eV)", 0.0104)
                return sigma, eps
            else:
                print("  Invalid choice. Please enter Ar, Kr, Xe, Ne, or custom.")

    @staticmethod
    def with_defaults():
        """Create MDSimulation with default configuration."""
        print("\n[Using default parameters]")
        return MDSimulation(SimulationConfig())

    @classmethod
    def from_user_input(cls):
        """Create MDSimulation by prompting user for all parameters."""
        print("\n[Press Enter to keep default]")

        print("\n--- Gas Selection ---")
        sigma, eps = cls.select_inert_gas()

        print("\n--- System setup ---")
        n_cells = cls.prompt("unit cells per side", 4, int)

        print("\n  Density specification:")
        print("    [1] Use ρ* (reduced number density)")
        print("    [2] Use cell parameter a (lattice constant)")
        density_mode = input("  Choose [1] or [2] (default: 1): ").strip()

        if density_mode == "2":
            cell_param_a = cls.prompt("cell parameter a (in σ)", 1.68)
            rho_star     = None
        else:
            cell_param_a = None
            rho_star     = cls.prompt("number density ρ*", 0.8442)

        print("\n--- Integration ---")
        dt = cls.prompt("time step dt", 0.005)

        print("\n--- Neighbour list ---")
        r_cutoff = cls.prompt("cutoff radius r_c", 2.5)
        r_skin   = cls.prompt("skin radius r_skin", 3.0)

        print("\n--- Run length ---")
        equil_steps      = cls.prompt("equilibration steps", 2000, int)
        prod_steps       = cls.prompt("production steps", 4000, int)
        rescale_interval = cls.prompt("velocity rescale every N steps", 10, int)
        rdf_interval     = cls.prompt("RDF sample every N steps", 10, int)
        rdf_bins         = cls.prompt("RDF histogram bins", 200, int)
        plot_rdf         = bool(cls.prompt("plot RDF? (1=yes, 0=no)", 1, int))

        print("\n--- Temperature sweep (T* = reduced temperature) ---")
        raw           = input("  T* values space-separated [0.01 0.2 0.4 0.6 0.8 1.0]: ").strip()
        t_star_values = [float(t) for t in raw.split()] if raw else [0.01, 0.2, 0.4, 0.6, 0.8, 1.0]

        print("\n--- Parallel workers ---")
        n_workers = cls.prompt("worker processes (0 = auto)", 0, int)
        n_workers = None if n_workers == 0 else n_workers

        print("\n--- Animation options ---")
        animate_equil_steps = cls.prompt("animation equil steps per T*", 2000, int)
        animate_prod_steps  = cls.prompt("animation production steps per T*", 2000, int)
        animate_frame_skip  = cls.prompt("save every N-th frame", 10, int)

        config = SimulationConfig(
            sigma            = sigma,
            eps              = eps,
            n_cells          = n_cells,
            rho_star         = rho_star,
            cell_param_a     = cell_param_a,
            dt               = dt,
            r_cutoff         = r_cutoff,
            r_skin           = r_skin,
            equil_steps      = equil_steps,
            prod_steps       = prod_steps,
            rescale_interval = rescale_interval,
            rdf_interval     = rdf_interval,
            rdf_bins         = rdf_bins,
            plot_rdf         = plot_rdf,
            t_star_values    = t_star_values,
            n_workers        = n_workers,
            animate_equil_steps = animate_equil_steps,
            animate_prod_steps  = animate_prod_steps,
            animate_frame_skip  = animate_frame_skip,
        )
        return MDSimulation(config)


# MD SIMULATION

class MDSimulation:

    def __init__(self, config: SimulationConfig):
        """Initialize MDSimulation with a configuration object."""
        # Material parameters
        self.sigma  = config.sigma
        self.eps    = config.eps
        self.eps_kb = config.eps * (1.603e-19 / k_B)

        # System setup
        self.n_cells = config.n_cells
        if config.cell_param_a is not None:
            L             = config.cell_param_a * config.n_cells
            self.rho_star = 4 * config.n_cells ** 3 / (L ** 3)
            self.cell_param_a = config.cell_param_a
            print(f"  Cell parameter a = {config.cell_param_a:.4f} σ")
            print(f"  Derived ρ* = {self.rho_star:.4f}")
        else:
            self.rho_star     = config.rho_star
            self.cell_param_a = None

        # Integration parameters
        self.dt          = config.dt
        self.r_cutoff    = config.r_cutoff
        self.r_skin      = config.r_skin
        self.r_cutoff_sq = config.r_cutoff ** 2
        self.r_skin_sq   = config.r_skin ** 2
        self.thresh_sq   = (0.5 * (config.r_skin - config.r_cutoff)) ** 2

        # Run parameters
        self.equil_steps      = config.equil_steps
        self.prod_steps       = config.prod_steps
        self.rescale_interval = config.rescale_interval
        self.rdf_interval     = config.rdf_interval
        self.rdf_bins         = config.rdf_bins
        self.plot_rdf         = config.plot_rdf

        # Temperature sweep
        self.t_star_values = config.t_star_values or [0.01, 0.2, 0.4, 0.6, 0.8, 1.0]

        # Worker processes
        if config.n_workers is None:
            cpu_count      = os.cpu_count() or 1
            self.n_workers = min(len(self.t_star_values), cpu_count)
        else:
            self.n_workers = config.n_workers

        # Animation parameters
        self.animate_equil_steps    = config.animate_equil_steps
        self.animate_prod_steps     = config.animate_prod_steps
        self.animate_frame_skip     = config.animate_frame_skip
        self.animate_t_star_values  = config.animate_t_star_values or self.t_star_values

        # Results storage
        self.simulation_results: dict = {}

    @classmethod
    def from_params(cls, **kwargs):
        """Create MDSimulation from keyword arguments by building a SimulationConfig internally."""
        return cls(SimulationConfig(**kwargs))

    # PHYSICS

    @staticmethod
    def _calculate_lennard_jones_properties(r_sq, r_cutoff_sq):
        """Shifted-force LJ potential and force/r for vectorised pair arrays."""
        inv_rc2     = 1.0 / r_cutoff_sq
        inv_rc6     = inv_rc2 ** 3
        inv_rc12    = inv_rc6 ** 2
        phi_cut     = 4.0 * (inv_rc12 - inv_rc6)
        force_by_rc = 48.0 * inv_rc2 * (inv_rc12 - 0.5 * inv_rc6)

        inv_r2      = 1.0 / r_sq
        inv_r6      = inv_r2 ** 3
        inv_r12     = inv_r6 ** 2
        phi         = 4.0 * (inv_r12 - inv_r6)
        force_by_r  = 48.0 * inv_r2 * (inv_r12 - 0.5 * inv_r6)

        force_by_r  = force_by_r - force_by_rc
        potE        = phi - phi_cut + 0.5 * force_by_rc * (r_sq - r_cutoff_sq)
        return potE, force_by_r

    @staticmethod
    def _calculate_forces(pos, neighbors, L, r_cutoff_sq):
        """Accumulate LJ forces over neighbour-list pairs."""
        forces = np.zeros_like(pos)
        if not len(neighbors):
            return 0.0, forces
        i, j   = neighbors[:, 0], neighbors[:, 1]
        dr     = MDSimulation._apply_minimum_image(pos[i], pos[j], L)
        r_sq   = np.sum(dr ** 2, axis=1)
        mask   = r_sq < r_cutoff_sq
        pot, f_over_r = MDSimulation._calculate_lennard_jones_properties(r_sq[mask], r_cutoff_sq)
        f_vecs = f_over_r[:, np.newaxis] * dr[mask]
        np.add.at(forces, i[mask],  f_vecs)
        np.add.at(forces, j[mask], -f_vecs)
        return np.sum(pot), forces

    @staticmethod
    def _apply_minimum_image(p1, p2, L):
        """Minimum image convention for periodic boundaries."""
        dr = p1 - p2
        return dr - L * np.round(dr / L)

    @staticmethod
    def _generate_fcc_lattice(n_cells, rho_star):
        """Place atoms on an FCC lattice consistent with the target density."""
        L     = (4 * n_cells ** 3 / rho_star) ** (1.0 / 3.0)
        a     = L / n_cells
        basis = np.array([[0, 0, 0], [0.5, 0.5, 0],
                          [0.5, 0, 0.5], [0, 0.5, 0.5]]) * a
        pos   = [np.array([ix, iy, iz]) * a + b
                 for ix in range(n_cells)
                 for iy in range(n_cells)
                 for iz in range(n_cells)
                 for b in basis]
        return np.array(pos), L

    @staticmethod
    def _update_neighbor_list(pos, L, r_skin_sq):
        """Build Verlet neighbour list for all pairs within r_skin."""
        dr      = MDSimulation._apply_minimum_image(
                      pos[:, np.newaxis, :], pos[np.newaxis, :, :], L)
        dist_sq = np.sum(dr ** 2, axis=2)
        i, j    = np.where(np.triu(dist_sq < r_skin_sq, k=1))
        return np.stack((i, j), axis=1)

    @staticmethod
    def _verlet_step(pos, vel, force, dt, box_length, neighbor_list, r_cutoff_sq):
        """Single velocity-Verlet integration step."""
        pos_new   = (pos + vel * dt + 0.5 * force * dt ** 2) % box_length
        vel_mid   = vel + 0.5 * force * dt
        pe, f_new = MDSimulation._calculate_forces(pos_new, neighbor_list, box_length, r_cutoff_sq)
        return pos_new, vel_mid + 0.5 * f_new * dt, pe, f_new

    @staticmethod
    def _rescale_velocities(v, T):
        """Rescale velocities to match target reduced temperature T*."""
        T_inst = np.sum(v ** 2) / (3 * len(v))
        return v * np.sqrt(T / T_inst) if T_inst > 0 else v

    @staticmethod
    def _calculate_rdf(pos, L, dr_bin, rdf_bins, r_max):
        """Compute radial distribution function g(r)."""
        N      = len(pos)
        dr_vec = MDSimulation._apply_minimum_image(
                     pos[:, np.newaxis, :], pos[np.newaxis, :, :], L)
        r      = np.sqrt(np.sum(dr_vec ** 2, axis=2))[np.triu_indices(N, k=1)]
        hist, edges = np.histogram(r, bins=rdf_bins, range=(0, r_max))
        c = (edges[:-1] + edges[1:]) / 2.0
        return c, (2.0 * hist * L ** 3) / (N ** 2 * 4 * np.pi * c ** 2 * dr_bin)

    @staticmethod
    def _needs_rebuild(pos, pos_ref, L, thresh_sq):
        """Return True if any atom has drifted beyond the rebuild threshold."""
        return np.max(np.sum(
            MDSimulation._apply_minimum_image(pos, pos_ref, L) ** 2, axis=1)) > thresh_sq

    @staticmethod
    def _classify_phase(g_avg, r_centers):
        """Heuristic phase classification from g(r) shape. Will be improved."""
        g_max       = np.max(g_avg)
        n_tail      = max(1, len(g_avg) // 10)
        deviation   = abs(float(np.mean(g_avg[-n_tail:])) - 1.0)
        if deviation < 0.03 and g_max < 3.5:
            return "LIQUID"
        return "SOLID"

    # DATACLASSES

    @dataclass
    class WorkerArgs:
        T_star:           float
        pos0:             np.ndarray
        L:                float
        eps_kb:           float
        dt:               float
        r_cutoff:         float
        r_skin:           float
        equil_steps:      int
        prod_steps:       int
        rescale_interval: int
        rdf_interval:     int
        rdf_bins:         int

    @dataclass
    class SimulationResult:
        T_star:      float
        T_kelvin:    float
        r_centers:   np.ndarray
        g_avg:       np.ndarray
        kin_E:       np.ndarray
        pot_E:       np.ndarray
        tot_E:       np.ndarray
        momentum:    np.ndarray
        equil_steps: int
        prod_steps:  int

    # PARALLEL WORKER

    @staticmethod
    def _run_one_temperature(args: 'MDSimulation.WorkerArgs') -> 'MDSimulation.SimulationResult':

        r_cutoff_sq = args.r_cutoff ** 2
        r_skin_sq   = args.r_skin ** 2
        thresh_sq   = (0.5 * (args.r_skin - args.r_cutoff)) ** 2
        r_max       = args.L / 2.0
        dr_bin      = r_max / args.rdf_bins

        rng      = np.random.default_rng(int(args.T_star * 1e6))
        pos      = args.pos0.copy()
        pos_ref  = pos.copy()
        N        = len(pos)
        vel      = rng.standard_normal((N, 3))
        vel     -= vel.mean(axis=0)
        vel      = MDSimulation._rescale_velocities(vel, args.T_star)
        nb_list  = MDSimulation._update_neighbor_list(pos, args.L, r_skin_sq)
        _, force = MDSimulation._calculate_forces(pos, nb_list, args.L, r_cutoff_sq)

        total_steps = args.equil_steps + args.prod_steps
        kin_E_trace = np.zeros(total_steps)
        pot_E_trace = np.zeros(total_steps)
        tot_E_trace = np.zeros(total_steps)
        mom_trace   = np.zeros((total_steps, 3))

        T_K = args.T_star * args.eps_kb
        print(f"  [T* = {args.T_star:.3f} | T = {T_K:.2f} K]  "
              f"equilibrating ({args.equil_steps} steps)...", flush=True)

        for step in range(args.equil_steps):
            pos, vel, pe, force = MDSimulation._verlet_step(
                pos, vel, force, args.dt, args.L, nb_list, r_cutoff_sq)
            if MDSimulation._needs_rebuild(pos, pos_ref, args.L, thresh_sq):
                nb_list = MDSimulation._update_neighbor_list(pos, args.L, r_skin_sq)
                pos_ref = pos.copy()
            if step % args.rescale_interval == 0:
                vel = MDSimulation._rescale_velocities(vel, args.T_star)
            ke                = 0.5 * np.sum(vel ** 2)
            kin_E_trace[step] = ke
            pot_E_trace[step] = pe
            tot_E_trace[step] = ke + pe
            mom_trace[step]   = vel.sum(axis=0)

        print(f"  [T* = {args.T_star:.3f}]  production ({args.prod_steps} steps)...", flush=True)

        g_acc     = np.zeros(args.rdf_bins)
        count     = 0
        r_centers = None

        for step in range(args.prod_steps):
            global_step = args.equil_steps + step
            pos, vel, pe, force = MDSimulation._verlet_step(
                pos, vel, force, args.dt, args.L, nb_list, r_cutoff_sq)
            if MDSimulation._needs_rebuild(pos, pos_ref, args.L, thresh_sq):
                nb_list = MDSimulation._update_neighbor_list(pos, args.L, r_skin_sq)
                pos_ref = pos.copy()
            ke                       = 0.5 * np.sum(vel ** 2)
            kin_E_trace[global_step] = ke
            pot_E_trace[global_step] = pe
            tot_E_trace[global_step] = ke + pe
            mom_trace[global_step]   = vel.sum(axis=0)
            if step % args.rdf_interval == 0:
                r_centers, g_curr = MDSimulation._calculate_rdf(
                    pos, args.L, dr_bin, args.rdf_bins, r_max)
                g_acc += g_curr
                count += 1

        return MDSimulation.SimulationResult(
            T_star      = args.T_star,
            T_kelvin    = T_K,
            r_centers   = r_centers,
            g_avg       = g_acc / count,
            kin_E       = kin_E_trace,
            pot_E       = pot_E_trace,
            tot_E       = tot_E_trace,
            momentum    = mom_trace,
            equil_steps = args.equil_steps,
            prod_steps  = args.prod_steps,
        )

    # ORCHESTRATION

    def run(self, output_file="melting_rdf.pdf", run_diag=True, diag_output_dir="diagnostics"):
        """Run the full parallel temperature sweep and collect results."""
        pos0, L = self._generate_fcc_lattice(self.n_cells, self.rho_star)
        N       = len(pos0)
        print(f"\nSystem: N = {N} atoms, L = {L:.4f} σ")
        print(f"ε/k_B = {self.eps_kb:.4f} K   (ε = {self.eps:.4e} eV,  k_B = {k_B:.6e} J/K)")
        print(f"T* values: {self.t_star_values}")
        print(f"Equivalent T (K): {[round(t * self.eps_kb, 3) for t in self.t_star_values]}")
        print(f"Launching {len(self.t_star_values)} tasks across {self.n_workers} workers...\n")

        worker_args = [
            MDSimulation.WorkerArgs(
                T_star           = T_star,
                pos0             = pos0,
                L                = L,
                eps_kb           = self.eps_kb,
                dt               = self.dt,
                r_cutoff         = self.r_cutoff,
                r_skin           = self.r_skin,
                equil_steps      = self.equil_steps,
                prod_steps       = self.prod_steps,
                rescale_interval = self.rescale_interval,
                rdf_interval     = self.rdf_interval,
                rdf_bins         = self.rdf_bins,
            )
            for T_star in self.t_star_values
        ]

        with Pool(processes=self.n_workers) as pool:
            results: list[MDSimulation.SimulationResult] = pool.map(
                MDSimulation._run_one_temperature, worker_args)

        results.sort(key=lambda r: r.T_star)
        for result in results:
            self.simulation_results[result.T_star] = result

        if self.plot_rdf:
            self.plot_radial_distribution_function(output_file, results)
        if run_diag:
            self.run_diagnostics(output_dir=diag_output_dir)

        return results

    # ANALYSIS & OUTPUT 

    def plot_radial_distribution_function(self, output_file, results):
        """Plot g(r) for all temperatures on a single figure."""
        plt.figure(figsize=(10, 6))
        for result in results:
            phase = self._classify_phase(result.g_avg, result.r_centers)
            plt.plot(
                result.r_centers * self.sigma,
                result.g_avg,
                label=f"T* = {result.T_star:.3f}  ({result.T_kelvin:.1f} K)  {phase}",
            )
            print(f"  T* = {result.T_star:.3f}  T = {result.T_kelvin:.2f} K  →  {phase}")
        plt.title("Radial Distribution Function $g(r)$ — Lennard-Jones System")
        plt.xlabel("Distance (Å)")
        plt.ylabel("$g(r)$")
        plt.axhline(1.0, color="black", linestyle="--", alpha=0.5)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        plt.show()
        print(f"\nComplete. Plot saved as '{output_file}'")

    # DIAGNOSTICS

    def run_diagnostics(self, output_dir="diagnostics"):
        """Save energy and momentum conservation plots for every temperature."""
        if not self.simulation_results:
            print("[Diagnostics] No data available. Run the simulation first.")
            return

        os.makedirs(output_dir, exist_ok=True)
        first_result = next(iter(self.simulation_results.values()))
        N_atoms      = 4 * self.n_cells ** 3
        total_steps  = first_result.equil_steps + first_result.prod_steps
        steps_arr    = np.arange(total_steps)

        print(f"\n{'='*60}")
        print(f"  DIAGNOSTICS  —  {N_atoms} atoms, {len(self.simulation_results)} temperatures")
        print(f"  Output directory: '{output_dir}/'")
        print(f"{'='*60}")

        for T_star in sorted(self.simulation_results):
            result = self.simulation_results[T_star]
            T_K    = result.T_kelvin
            tag    = f"T_star_{T_star:.3f}".replace(".", "p")

            fig, (ax_e, ax_p) = plt.subplots(2, 1, figsize=(10, 9))

            # Energy subplot
            ax_e.plot(steps_arr, result.kin_E, label="Kinetic Energy",   color="blue",  alpha=0.6)
            ax_e.plot(steps_arr, result.pot_E, label="Potential Energy", color="green", alpha=0.6)
            ax_e.plot(steps_arr, result.tot_E, label="Total Energy",     color="red",   linewidth=2)
            ax_e.axvspan(0, result.equil_steps, alpha=0.1, color="gray", label="Equilibration")
            ax_e.set_xlabel("Step")
            ax_e.set_ylabel("Energy (reduced units)")
            ax_e.set_title(f"Energy Conservation  |  T* = {T_star:.3f}  (T = {T_K:.2f} K)  |  {N_atoms} atoms")
            ax_e.legend()
            ax_e.grid(True, linestyle="--", alpha=0.5)

            # Momentum subplot
            P_mag = np.linalg.norm(result.momentum, axis=1)
            ax_p.plot(steps_arr, P_mag, label="|P| total", color="black", linewidth=1.5)
            ax_p.axvspan(0, result.equil_steps, alpha=0.1, color="gray", label="Equilibration")
            ax_p.set_xlabel("Step")
            ax_p.set_ylabel("Total Momentum magnitude (reduced units)")
            ax_p.set_title(f"Momentum Conservation  |  T* = {T_star:.3f}  (T = {T_K:.2f} K)  |  {N_atoms} atoms")
            ax_p.legend()
            ax_p.grid(True, linestyle="--", alpha=0.5)

            fig.tight_layout()
            combined_file = os.path.join(output_dir, f"diagnostics_{tag}.pdf")
            fig.savefig(combined_file, dpi=150)
            plt.close(fig)

            prod_E   = result.tot_E[result.equil_steps:]
            E_mean   = float(np.mean(prod_E))
            E_fluc   = float(np.std(prod_E))
            rel_fluc = E_fluc / abs(E_mean) if E_mean != 0 else float("inf")
            stability = "✓ stable" if rel_fluc < 1e-4 else "✗ UNSTABLE — consider reducing dt"
            P_max     = np.max(np.abs(result.momentum), axis=0)

            print(f"\n  T* = {T_star:.3f}  (T = {T_K:.2f} K)")
            print(f"    Mean Total Energy   : {E_mean:.6f}")
            print(f"    RMS Fluctuation     : {E_fluc:.6f}")
            print(f"    Relative fluctuation: {rel_fluc:.2e}  → {stability}")
            print(f"    Max |Px|={P_max[0]:.4f}  |Py|={P_max[1]:.4f}  |Pz|={P_max[2]:.4f}")
            print(f"    Combined plot  → {combined_file}")

        print(f"\n{'='*60}")
        print("Done running diagnostics.\n")

    # ANIMATION

    def animate(self, output_file="md_engine", format="both"):
        """Run a sequential heating sweep and save an animated visualisation."""
        gif_file = output_file + ".gif"
        mp4_file = output_file + ".mp4"

        pos0, L  = self._generate_fcc_lattice(self.n_cells, self.rho_star)
        N        = len(pos0)
        r_max    = L / 2.0
        dr_bin   = r_max / self.rdf_bins
        skip     = self.animate_frame_skip
        total_eq = self.animate_equil_steps
        total_pr = self.animate_prod_steps

        frames        = []
        finished_rdfs = []
        all_speeds    = []
        t_star_list   = self.animate_t_star_values

        print(f"\nAnimation  —  {len(t_star_list)} temperatures, "
              f"{total_eq} equil + {total_pr} prod steps, frame skip={skip}")

        rng     = np.random.default_rng(42)
        pos     = pos0.copy()
        pos_ref = pos.copy()
        vel     = rng.standard_normal((N, 3))
        vel    -= vel.mean(axis=0)
        vel     = self._rescale_velocities(vel, t_star_list[0])
        nb_list = self._update_neighbor_list(pos, L, self.r_skin_sq)
        _, force = self._calculate_forces(pos, nb_list, L, self.r_cutoff_sq)

        for T_star in t_star_list:
            T_K = T_star * self.eps_kb
            vel = self._rescale_velocities(vel, T_star)
            print(f"  T* = {T_star:.3f}  (T = {T_K:.1f} K)  equilibrating...", end="", flush=True)

            for step in range(total_eq):
                pos, vel, _, force = self._verlet_step(
                    pos, vel, force, self.dt, L, nb_list, self.r_cutoff_sq)
                if self._needs_rebuild(pos, pos_ref, L, self.thresh_sq):
                    nb_list = self._update_neighbor_list(pos, L, self.r_skin_sq)
                    pos_ref = pos.copy()
                if step % self.rescale_interval == 0:
                    vel = self._rescale_velocities(vel, T_star)

            print(" recording...", end="", flush=True)
            g_acc = np.zeros(self.rdf_bins)
            count = 0

            for step in range(total_pr):
                pos, vel, _, force = self._verlet_step(
                    pos, vel, force, self.dt, L, nb_list, self.r_cutoff_sq)
                if self._needs_rebuild(pos, pos_ref, L, self.thresh_sq):
                    nb_list = self._update_neighbor_list(pos, L, self.r_skin_sq)
                    pos_ref = pos.copy()

                if step % skip == 0:
                    speed               = np.sqrt(np.sum(vel ** 2, axis=1))
                    r_centers, g_curr   = self._calculate_rdf(pos, L, dr_bin, self.rdf_bins, r_max)
                    g_acc              += g_curr
                    count              += 1
                    g_running           = g_acc / count
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
        gs      = fig.add_gridspec(1, 2, wspace=0.3, left=0.05, right=0.97, top=0.80, bottom=0.13)
        ax_slab = fig.add_subplot(gs[0])
        ax_rdf  = fig.add_subplot(gs[1])

        for ax in (ax_slab, ax_rdf):
            ax.set_facecolor("#111111")
            ax.tick_params(colors="0.7")
            for sp in ax.spines.values():
                sp.set_color("0.3")

        ax_slab.set_xlim(0, L);  ax_slab.set_ylim(0, L)
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

        ax_rdf.set_xlim(0, r_max * self.sigma)
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
        main_title      = fig.suptitle("", color="white", fontsize=14, y=0.96)
        info_text       = fig.text(0.5, 0.90, "", color="#a8dadc", fontsize=11, ha="center")
        phase_colors    = {"SOLID": "#3498db", "LIQUID": "#e74c3c", "GAS": "#2ecc71"}

        def update(fi):
            atom_xy, speed, r_centers, g_running, T_star_f, T_K_f, peak_g = frames[fi]
            phase = self._classify_phase(g_running, r_centers)
            sc.set_offsets(atom_xy)
            sc.set_array(speed)
            slab_title.set_text(f"Atoms (xy)  |  T* = {T_star_f:.3f}  ({T_K_f:.1f} K)")
            for idx, (T_g, T_K_g, r_g, g_g) in enumerate(finished_rdfs):
                if T_g < T_star_f and idx not in drawn_ghosts:
                    col    = ghost_cmap(idx / max(n_t - 1, 1))
                    phase_g = self._classify_phase(g_g, r_g)
                    gl, = ax_rdf.plot(r_g * self.sigma, g_g,
                                      color=col, lw=1.0, alpha=0.4,
                                      label=f"T*={T_g:.2f} ({phase_g})", zorder=2)
                    ghost_lines_rdf.append(gl)
                    ghost_handles.append(gl)
                    drawn_ghosts.add(idx)
            live_line_rdf.set_data(r_centers * self.sigma, g_running)
            live_line_rdf.set_label(f"T*={T_star_f:.2f} (live)")
            rdf_title.set_text("g(r) running avg")
            ax_rdf.legend(handles=ghost_handles + [live_line_rdf],
                          loc="upper right", fontsize=7,
                          framealpha=0.2, labelcolor="white")
            info_text.set_text(f"g(r) peak = {peak_g:.2f}")
            main_title.set_text(f"Melting & Diffusion  |  PHASE: {phase}")
            main_title.set_color(phase_colors.get(phase, "white"))
            return (sc, live_line_rdf, slab_title, rdf_title, main_title, info_text)

        ani = FuncAnimation(fig, update, frames=total_frames, interval=60, blit=False)

        if format in ("gif", "both"):
            print(f"\nSaving GIF  → '{gif_file}' ({total_frames} frames) ...")
            try:
                ani.save(gif_file, writer="pillow", fps=20, dpi=130)
                print(f"  ✓ GIF saved: '{gif_file}'")
            except Exception as e:
                print(f"  ✗ GIF failed: {e}")

        if format in ("mp4", "both"):
            print(f"Saving MP4  → '{mp4_file}' ({total_frames} frames) ...")
            try:
                from matplotlib.animation import FFMpegWriter
                mp4_writer = FFMpegWriter(
                    fps       = 20,
                    codec     = "libx264",
                    extra_args= [
                        "-pix_fmt", "yuv420p",
                        "-vf",      "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                        "-crf",     "18",
                        "-preset",  "fast",
                    ]
                )
                ani.save(mp4_file, writer=mp4_writer, dpi=130)
                print(f"  ✓ MP4 saved: '{mp4_file}'")
            except FileNotFoundError:
                print("  ✗ MP4 failed: FFmpeg not found — install FFmpeg or use format='gif'")
            except Exception as e:
                print(f"  ✗ MP4 failed: {e}")

        try:
            plt.show()
        except Exception:
            pass


# MAIN WORKFLOW
def main():
    print("\n" + "=" * 70)
    print("   LENNARD-JONES MD SIMULATION — RDF ANALYSIS")
    print("=" * 70)

    while True:
        print("\n" + "-" * 70)
        print("  NEW SIMULATION")
        print("-" * 70)

        default = input("\nUse default parameters? [y/n] (default: y): ").strip().lower()
        sim = CLI.with_defaults() if default in ("", "y", "yes") else CLI.from_user_input()

        print("\n--- Post-simulation analysis ---")
        run_diag = input("Run diagnostics after simulation? [y/n] (default: y): ").strip().lower()
        run_diag = run_diag in ("", "y", "yes")

        diag_dir = "diagnostics"
        if run_diag:
            raw_dir  = input("  Diagnostics output folder [diagnostics]: ").strip()
            diag_dir = raw_dir if raw_dir else "diagnostics"

        print("\n" + "=" * 70)
        print("  RUNNING SIMULATION...")
        print("=" * 70)
        sim.run(run_diag=run_diag, diag_output_dir=diag_dir)

        print("\n" + "=" * 70)
        print("  SIMULATION COMPLETE")
        print("=" * 70)

        while True:
            print("\n  What would you like to do?")
            print("    [1] Run animation sweep")
            print("    [2] Run another simulation")
            print("    [3] Exit")
            post_choice = input("\n  Select option [1/2/3]: ").strip()

            if post_choice == "1":
                while True:
                    fmt = input("\n  Output format? [gif/mp4/both] (default: both): ").strip().lower()
                    if fmt in ("", "both"):
                        fmt = "both";  break
                    elif fmt in ("gif", "mp4"):
                        break
                    print("    Please enter gif, mp4, or both.")
                sim.animate(format=fmt)

            elif post_choice == "2":
                break

            elif post_choice == "3":
                print("\n" + "=" * 70)
                print("  Thank you for using MD Engine!")
                print("=" * 70 + "\n")
                return

            else:
                print("    Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()