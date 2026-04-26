import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def prompt(label, default, cast=float):
    val = input(f"  {label} [{default}]: ").strip()
    return cast(val) if val else cast(default)


class MDSimulation:
    def __init__(self, sigma=3.405, eps_kb=162.46, n_cells=4, rho_star=0.8442,
                 dt=0.005, r_cutoff=2.5, r_skin=3.0, equil_steps=2000,
                 prod_steps=2000, rescale_interval=10, rdf_interval=10,
                 rdf_bins=200, temperatures=None,
                 animate_equil_steps=200,
                 animate_prod_steps=200,
                 animate_frame_skip=2,
                 animate_temperatures=None):
        self.sigma = sigma
        self.eps_kb = eps_kb
        self.n_cells = n_cells
        self.rho_star = rho_star
        self.dt = dt
        self.r_cutoff = r_cutoff
        self.r_skin = r_skin
        self.r_cutoff_sq = r_cutoff ** 2
        self.r_skin_sq = r_skin ** 2
        self.thresh_sq = (0.5 * (r_skin - r_cutoff)) ** 2
        self.equil_steps = equil_steps
        self.prod_steps = prod_steps
        self.rescale_interval = rescale_interval
        self.rdf_interval = rdf_interval
        self.rdf_bins = rdf_bins
        self.temperatures = temperatures if temperatures is not None else [5.0, 32.49, 64.98, 97.48, 129.97, 162.46]

        self.animate_equil_steps = animate_equil_steps
        self.animate_prod_steps = animate_prod_steps
        self.animate_frame_skip = animate_frame_skip
        self.animate_temperatures = animate_temperatures if animate_temperatures is not None \
            else self.temperatures
        self.kin_E = np.zeros(self.equil_steps + self.prod_steps)
        self.pot_E = np.zeros(self.equil_steps + self.prod_steps)
        self.tot_E = np.zeros(self.equil_steps + self.prod_steps)
        self.P = np.zeros((self.equil_steps + self.prod_steps, 3))

    @classmethod
    def with_defaults(cls):
        print("\n[Using default parameters]")
        return cls()

    @classmethod
    def from_user_input(cls):
        print("\n[Press Enter to keep default]")
        print("\n--- Physical constants ---")
        sigma  = prompt("sigma (Å)", 3.405)
        eps_kb = prompt("eps/kB (K)", 119.8)
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
        print("\n--- Temperature sweep ---")
        raw = input("  temperatures in Kelvin (space-separated) [32.5 65.0 97.5 130.0 162.5]: ").strip()
        temperatures = [float(t) for t in raw.split()] if raw else [32.5, 65.0, 97.5, 130.0, 162.5]
        print("\n--- Animation options ---")
        animate_equil_steps = prompt("animation equil steps per T*", 200, int)
        animate_prod_steps  = prompt("animation production steps per T*", 200, int)
        animate_frame_skip  = prompt("save every N-th frame", 2, int)
        return cls(sigma=sigma, eps_kb=eps_kb, n_cells=n_cells, rho_star=rho_star,
                   dt=dt, r_cutoff=r_cutoff, r_skin=r_skin, equil_steps=equil_steps,
                   prod_steps=prod_steps, rescale_interval=rescale_interval,
                   rdf_interval=rdf_interval, rdf_bins=rdf_bins, temperatures=temperatures,
                   animate_equil_steps=animate_equil_steps,
                   animate_prod_steps=animate_prod_steps,
                   animate_frame_skip=animate_frame_skip)

    # ------------------------------------------------------------------ core
    def _lj_properties(self, r_sq):
        rc_sq = self.r_cutoff_sq
        inv_rc2 = 1.0 / rc_sq
        phi_rc = 4 * (inv_rc2 ** 6 - inv_rc2 ** 3)
        f_rc_r = 48 * inv_rc2 * (inv_rc2 ** 6 - 0.5 * inv_rc2 ** 3)
        inv_r2 = 1.0 / r_sq
        inv_r6, inv_r12 = inv_r2 ** 3, inv_r2 ** 6
        f_over_r = 48 * inv_r2 * (inv_r12 - 0.5 * inv_r6) - f_rc_r
        pot = 4 * (inv_r12 - inv_r6) - phi_rc + 0.5 * f_rc_r * (r_sq - rc_sq)
        return pot, f_over_r

    def _calculate_forces(self, pos, neighbors, L):
        forces = np.zeros_like(pos)
        if not len(neighbors):
            return 0.0, forces
        i, j = neighbors[:, 0], neighbors[:, 1]
        dr = self._minimum_image(pos[i], pos[j], L)
        r_sq = np.sum(dr ** 2, axis=1)
        mask = r_sq < self.r_cutoff_sq
        pot, f_over_r = self._lj_properties(r_sq[mask])
        f_vecs = f_over_r[:, np.newaxis] * dr[mask]
        np.add.at(forces, i[mask],  f_vecs)
        np.add.at(forces, j[mask], -f_vecs)
        return np.sum(pot), forces

    @staticmethod
    def _minimum_image(p1, p2, L):
        dr = p1 - p2
        return dr - L * np.round(dr / L)

    def _generate_fcc_lattice(self):
        n, rho = self.n_cells, self.rho_star
        L = (4 * n ** 3 / rho) ** (1 / 3)
        a = L / n
        basis = np.array([[0, 0, 0], [0.5, 0.5, 0],
                           [0.5, 0, 0.5], [0, 0.5, 0.5]]) * a
        pos = [np.array([ix, iy, iz]) * a + b
               for ix in range(n) for iy in range(n) for iz in range(n)
               for b in basis]
        return np.array(pos), L

    def _update_neighbor_list(self, pos, L):
        dr = self._minimum_image(pos[:, np.newaxis, :], pos[np.newaxis, :, :], L)
        dist_sq = np.sum(dr ** 2, axis=2)
        i, j = np.where(np.triu(dist_sq < self.r_skin_sq, k=1))
        return np.stack((i, j), axis=1)

    def _verlet_step(self, pos, vel, f, L, nb):
        pos_new = (pos + vel * self.dt + 0.5 * f * self.dt ** 2) % L
        v_mid = vel + 0.5 * f * self.dt
        pe, f_new = self._calculate_forces(pos_new, nb, L)
        return pos_new, v_mid + 0.5 * f_new * self.dt, pe, f_new

    @staticmethod
    def _initialize_velocities(N, T):
        v = np.random.normal(0.0, np.sqrt(T), (N, 3))
        return v - np.mean(v, axis=0)

    @staticmethod
    def _rescale_velocities(v, T):
        T_inst = np.sum(v ** 2) / (3 * len(v))
        return v * np.sqrt(T / T_inst) if T_inst > 0 else v

    def _calculate_rdf(self, pos, L, dr_bin, r_max):
        N = len(pos)
        dr_vec = self._minimum_image(pos[:, np.newaxis, :], pos[np.newaxis, :, :], L)
        r = np.sqrt(np.sum(dr_vec ** 2, axis=2))[np.triu_indices(N, k=1)]
        hist, edges = np.histogram(r, bins=self.rdf_bins, range=(0, r_max))
        c = (edges[:-1] + edges[1:]) / 2.0
        return c, (2.0 * hist * L ** 3) / (N ** 2 * 4 * np.pi * c ** 2 * dr_bin)

    def _needs_rebuild(self, pos, pos_ref, L):
        return np.max(np.sum(
            self._minimum_image(pos, pos_ref, L) ** 2, axis=1)) > self.thresh_sq

    # ------------------------------------------------------------------ run
    def run(self, output_file="melting_rdf.png", animate=False):
        if animate:
            self.animate()
            return

        pos, L = self._generate_fcc_lattice()
        N = len(pos)
        r_max = L / 2.0
        dr_bin = r_max / self.rdf_bins
        print(f"\nSystem: N = {N} atoms, L = {L:.4f} σ")
        plt.figure(figsize=(10, 6))

        T_star = self.temperatures[0] / self.eps_kb
        pos_ref = pos.copy()
        vel = self._initialize_velocities(N, T_star)
        nb_list = self._update_neighbor_list(pos, L)
        pe, force = self._calculate_forces(pos, nb_list, L)

        for T_target in self.temperatures:
            T_star = T_target / self.eps_kb
            print(f"\n--- T = {T_target} K (T* = {T_star:.3f}) ---")
            vel = self._rescale_velocities(vel, T_star)

            print(f"  Equilibrating ({self.equil_steps} steps)...")
            for step in range(self.equil_steps):
                pos, vel, pe, force = self._verlet_step(pos, vel, force, L, nb_list)
                self.pot_E[step] = pe
                self.P[step] = np.sum(vel, axis=0)
                self.kin_E[step] = 0.5*np.sum(vel**2)
                self.tot_E[step] = self.pot_E[step] + self.kin_E[step]
                if self._needs_rebuild(pos, pos_ref, L):
                    nb_list, pos_ref = self._update_neighbor_list(pos, L), pos.copy()
                if step % self.rescale_interval == 0:
                    vel = self._rescale_velocities(vel, T_star)

            print(f"  Production ({self.prod_steps} steps)...")
            g_acc = np.zeros(self.rdf_bins)
            count = 0

            for step in range(self.prod_steps):
                pos, vel, pe, force = self._verlet_step(pos, vel, force, L, nb_list)
                self.pot_E[step+self.equil_steps] = pe
                self.P[step+self.equil_steps] = np.sum(vel, axis=0)
                self.kin_E[step+self.equil_steps] = 0.5*np.sum(vel**2)
                self.tot_E[step+self.equil_steps] = self.pot_E[step+self.equil_steps] + self.kin_E[step+self.equil_steps]
                if self._needs_rebuild(pos, pos_ref, L):
                    nb_list, pos_ref = self._update_neighbor_list(pos, L), pos.copy()
                if step % self.rdf_interval == 0:
                    r_centers, g_curr = self._calculate_rdf(pos, L, dr_bin, r_max)
                    g_acc += g_curr
                    count += 1

            g_avg = g_acc / count
            peak_g = float(np.max(g_avg))
            phase = "Solid" if peak_g > 3.0 else "Transition" if peak_g > 1.8 else "Liquid"

            plt.plot(r_centers * self.sigma, g_avg,
                     label=f"T = {T_target} K ({phase}, peak={peak_g:.2f})")
            print(f"  Done — averaged {count} RDF snapshots. g(r) peak = {peak_g:.2f} → {phase}")

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
    
    def run_diagnostics(self, output_files = ["energy_plot.pdf", "momentum_plot.pdf"]):
        steps_arr = np.arange(self.equil_steps + self.prod_steps)
        plt.plot(steps_arr, self.kin_E,      label='Kinetic Energy',    color='blue',  alpha=0.6)
        plt.plot(steps_arr, self.pot_E,      label='Potential Energy',    color='green', alpha=0.6)
        plt.plot(steps_arr, self.tot_E, label='Total Energy', color='red',   linewidth=2)
        plt.axvspan(0, self.equil_steps, alpha=0.1, color='gray', label='Equilibration')
        plt.xlabel('Step'); plt.ylabel('Energy (reduced units)')
        plt.title('Energy Conservation  |  ' + str(4*self.n_cells**3) + ' atoms')
        plt.legend(); plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig(output_files[0], dpi=150)
        plt.show()
        P_mag = np.linalg.norm(self.P, axis=1)
        plt.plot(steps_arr, P_mag,      label='Total Momentum (P)',    color='black',  linewidth=2)
        plt.xlabel('Step'); plt.ylabel('Total Momentum (reduced units)')
        plt.title('Momentum Conservation  |  ' + str(4*self.n_cells**3) + ' atoms')
        plt.legend(); plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig(output_files[1], dpi=150)
        plt.show()
        E_mean = np.mean(self.tot_E[self.equil_steps:])
        E_fluc = np.std(self.tot_E[self.equil_steps:])
        rel_fluc = E_fluc/np.abs(E_mean)
        print(f"Mean Total Energy is {E_mean} and RMS Fluctuation is {E_fluc}\n")
        if(rel_fluc < 1e-4):
            print(f"With a relative fluctuation of {rel_fluc} < 1e-4, the system is numerically stable!\n")
        else:
            print(f"With a relative fluctuation of {rel_fluc} > 1e-4, the system is numerically unstable.\n")
        P_x = np.max(np.abs(self.P[:, 0]))
        P_y = np.max(np.abs(self.P[:, 1]))
        P_z = np.max(np.abs(self.P[:, 2]))
        print(f"The max values of components P_x, P_y and P_z are {P_x}, {P_y} and {P_z} respectively\n")
        print("Done running Diagnostics.\n")


    def _classify_phase(self, signals):
        peak_g = signals.get('peak_g', 0)
        if peak_g > 3.0:
            phase = 'SOLID'
        elif peak_g > 1.8:
            phase = 'LIQUID'
        else:
            phase = 'GAS'
        return phase, signals
    
    # ------------------------------------------------------------------ animate
    def animate(self, output_file="md_engine.gif"):
        import os
        base, _ = os.path.splitext(output_file)
        gif_file = base + ".gif"
        mp4_file = base + ".mp4"

        pos0, L = self._generate_fcc_lattice()
        N        = len(pos0)
        r_max    = L / 2.0
        dr_bin   = r_max / self.rdf_bins
        skip     = self.animate_frame_skip
        total_eq = self.animate_equil_steps
        total_pr = self.animate_prod_steps

        frames        = []
        finished_rdfs = []
        all_speeds    = []

        print(f"\nAnimation  —  {len(self.animate_temperatures)} temperatures, "
              f"{total_eq} equil + {total_pr} prod steps, frame skip={skip}")

        pos = pos0.copy()
        pos_ref = pos.copy()
        T_star = self.animate_temperatures[0] / self.eps_kb
        vel    = self._initialize_velocities(N, T_star)
        nb_list = self._update_neighbor_list(pos, L)
        pe, force = self._calculate_forces(pos, nb_list, L)

        for T_target in self.animate_temperatures:
            T_star = T_target / self.eps_kb
            vel = self._rescale_velocities(vel, T_star)
            print(f"  T = {T_target} K (T* = {T_star:.3f})  equilibrating...", end="", flush=True)

            for step in range(total_eq):
                pos, vel, pe, force = self._verlet_step(pos, vel, force, L, nb_list)
                if self._needs_rebuild(pos, pos_ref, L):
                    nb_list, pos_ref = self._update_neighbor_list(pos, L), pos.copy()
                if step % self.rescale_interval == 0:
                    vel = self._rescale_velocities(vel, T_star)

            print(f" recording...", end="", flush=True)
            g_acc = np.zeros(self.rdf_bins)
            count = 0

            for step in range(total_pr):
                pos, vel, pe, force = self._verlet_step(pos, vel, force, L, nb_list)
                if self._needs_rebuild(pos, pos_ref, L):
                    nb_list, pos_ref = self._update_neighbor_list(pos, L), pos.copy()

                if step % skip == 0:
                    speed = np.sqrt(np.sum(vel ** 2, axis=1))
                    all_speeds.append(speed)

                    atom_xy = pos[:, :2]

                    r_centers, g_curr = self._calculate_rdf(pos, L, dr_bin, r_max)
                    g_acc += g_curr
                    count += 1

                    g_running = g_acc / count
                    phase_signals = {'peak_g': float(np.max(g_running))}

                    frames.append((atom_xy, speed,
                                   r_centers, g_running,
                                   T_target, count,
                                   phase_signals))

            finished_rdfs.append((T_target, r_centers, g_acc / count))
            print(f" {count} frames.")

        s_concat = np.concatenate(all_speeds)
        s_min, s_max = float(np.percentile(s_concat, 2)), float(np.percentile(s_concat, 98))
        total_frames = len(frames)
        print(f"\nTotal frames: {total_frames}.  Building figure...")

        fig = plt.figure(figsize=(11, 5.5), facecolor="#0d0d0d")
        gs  = fig.add_gridspec(1, 2, wspace=0.3, left=0.05, right=0.97,
                               top=0.80, bottom=0.13)
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
        sc = ax_slab.scatter(
            init_xy[:, 0], init_xy[:, 1],
            c=init_spd, cmap="plasma", vmin=s_min, vmax=s_max,
            s=18, alpha=0.9, linewidths=0,
        )
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

        ghost_lines_rdf  = []
        ghost_cmap       = plt.cm.cool
        n_t              = len(self.animate_temperatures)
        live_line_rdf, = ax_rdf.plot([], [], color="#ff6b35", lw=1.8, zorder=5)

        main_title       = fig.suptitle("", color="white", fontsize=14, y=0.96)
        explanation_text = fig.text(0.5, 0.90, "", color="#a8dadc", fontsize=11, ha="center")

        drawn_ghosts      = set()
        ghost_handles_rdf = []

        def update(fi):
            (atom_xy, speed, r_centers, g_running,
             T_now, n_avg, ph_signals) = frames[fi]

            phase, signals = self._classify_phase(ph_signals)

            phase_colors = {'SOLID': "#3498db", 'LIQUID': "#e74c3c", 'GAS': "#2ecc71"}
            p_col = phase_colors.get(phase, "white")

            sc.set_offsets(atom_xy)
            sc.set_array(speed)
            slab_title.set_text(f"Atoms (xy)  |  T = {T_now} K")

            for idx, (T_g, r_g, g_g) in enumerate(finished_rdfs):
                if T_g < T_now and idx not in drawn_ghosts:
                    col = ghost_cmap(idx / max(n_t - 1, 1))
                    gl_rdf, = ax_rdf.plot(r_g * self.sigma, g_g,
                                          color=col, lw=1.0, alpha=0.4,
                                          label=f"T={T_g} K", zorder=2)
                    ghost_lines_rdf.append(gl_rdf)
                    ghost_handles_rdf.append(gl_rdf)
                    drawn_ghosts.add(idx)

            # FIX: update live line label and rebuild legend every frame
            # so the label always reflects the current temperature T_now,
            # not whatever temperature was active when the legend was last drawn.
            live_line_rdf.set_data(r_centers * self.sigma, g_running)
            live_line_rdf.set_label(f"T={T_now} K (live)")
            rdf_title.set_text("g(r) running avg")
            ax_rdf.legend(handles=ghost_handles_rdf + [live_line_rdf],
                          loc="upper right", fontsize=7,
                          framealpha=0.2, labelcolor="white")

            peak_str = f"g(r) peak={signals.get('peak_g', 0):.2f}"
            explanation_text.set_text(peak_str)

            main_title.set_text(f"Melting & Diffusion  |  PHASE: {phase}")
            main_title.set_color(p_col)

            return (sc, live_line_rdf, slab_title, rdf_title,
                    main_title, explanation_text)

        ani = FuncAnimation(fig, update, frames=total_frames, interval=60, blit=False)

        print(f"\nSaving GIF  → '{gif_file}' ({total_frames} frames) ...")
        try:
            ani.save(gif_file, writer="pillow", fps=20, dpi=130)
            print(f"  ✓ GIF saved: '{gif_file}'")
        except Exception as e:
            print(f"  ✗ GIF failed: {e}")

        print(f"Saving MP4  → '{mp4_file}' ({total_frames} frames) ...")
        try:
            from matplotlib.animation import FFMpegWriter
            mp4_writer = FFMpegWriter(
                fps=20,
                codec="libx264",
                extra_args=[
                    "-pix_fmt", "yuv420p",
                    "-crf", "18",
                    "-preset", "fast",
                ],
            )
            ani.save(mp4_file, writer=mp4_writer, dpi=130)
            print(f"  ✓ MP4 saved: '{mp4_file}'")
        except Exception as e:
            print(f"  ✗ MP4 failed: {e}")
            print("    Make sure FFmpeg is installed:  conda install ffmpeg  or  apt install ffmpeg")

        try:
            plt.show()
        except Exception:
            pass


# ------------------------------------------------------------------ entry point
if __name__ == "__main__":
    print("=" * 50)
    print("   Lennard-Jones MD Simulation — RDF Analysis")
    print("=" * 50)
    print("\nParameter mode:\n  [1] Run Full Simulation (RDF Analysis)\n  [2] Run Animation Sweep")
    while True:
        choice = input("\nSelect option (1 or 2): ").strip()
        if choice == "1":
            sim = MDSimulation.with_defaults()
            sim.run()
            sim.run_diagnostics()
            break
        elif choice == "2":
            sim = MDSimulation.with_defaults()
            sim.run(animate=True)
            sim.run_diagnostics()
            break
        else:
            print("  Please enter 1 or 2.")