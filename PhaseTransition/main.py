import re
import warnings
from pathlib import Path
import os,sys 
import h5py
import numpy as np
import scipy as sp
import scipy.integrate
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from md_engine import MDSimulation, SimulationConfig

N_CELLS    = 10
DATA_DIR   = Path(f"data_{N_CELLS}")
FILE_GLOB  = "result_rho*_T*.h5"


def get_pressure(r,g_r,rho_star,T_star,sim,LRC=False):
    r_squared = r**2
    _,forces = sim._calculate_lennard_jones_properties(r_squared,SimulationConfig.r_cutoff**2)
    forces[r > SimulationConfig.r_cutoff] = 0.0

    integral = sp.integrate.simpson(forces*r_squared**2*g_r, x=r)
    P_star = rho_star*T_star + (2.0/3)*np.pi*rho_star**2*integral

    P_star_LRC = (
        (32.0/9)*np.pi*rho_star**2*(SimulationConfig.r_cutoff**(-9)) 
      - (16.0/3)*np.pi*rho_star**2*(SimulationConfig.r_cutoff**(-3))
    )

    if LRC:
        return P_star + P_star_LRC
    else:
        return P_star


_FNAME_RE = re.compile(r"result_rho([0-9.]+)_T([0-9.]+)\.h5$")

def _parse_filename(path: Path):
    """Return (rho, T) floats parsed from filename, or None if no match."""
    m = _FNAME_RE.search(path.name)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None


def load_single_file(path: Path):
    """
    Load one HDF5 result file.
 
    Returns a dict with keys:
        rho, T, r, g_avg, msd_arr, time_step, D, P, phase
    or raises on malformed files.
    """
    parsed = _parse_filename(path)
    if parsed is None:
        raise ValueError(f"Cannot parse rho/T from filename: {path.name}")
    rho_star, T_star = parsed
 
    with h5py.File(path, "r") as f:
        # ── locate the temperature group ──────────────────────────────────
        # Group name is e.g. 'T_0.5000'; fall back to first group if needed
        grp_key = f"T_{T_star:.4f}"
        if grp_key not in f:
            # Try to find a matching key
            candidates = [k for k in f.keys() if k.startswith("T_")]
            if not candidates:
                raise KeyError(f"No temperature group found in {path.name}")
            grp_key = candidates[0]
            warnings.warn(f"{path.name}: expected group '{f'T_{T_star:.4f}'}', using '{grp_key}'")
 
        grp = f[grp_key]
 
        g_avg     = np.array(grp["g_avg"],     dtype=float)
        r_centers = np.array(grp["r_centers"], dtype=float)
        msd_arr   = np.array(grp["msd"],       dtype=float)
        time_step = np.array(grp["time_step"], dtype=float)
 
        # ── optional: running average (not used here but validated) ──────
        # g_run = np.array(grp["g_running_avg"])  # available if needed
 
    _, _, D = MDSimulation._calculate_diffusion_coefficient(msd_arr, time_step)
    P   = get_pressure(r_centers, g_avg, rho_star, T_star, MDSimulation, LRC=True)
    phase = MDSimulation._classify_phase(g_avg=g_avg,D=D, msd=msd_arr)
 
    return dict(
        rho=rho_star, T=T_star,
        r=r_centers, g_avg=g_avg,
        msd_arr=msd_arr, time_step=time_step,
        D=D, P=P, phase=phase,
    )

def load_all_files(data_dir=DATA_DIR, glob=FILE_GLOB, verbose=True):
    """
    Scan data_dir for all matching HDF5 files, load each one,
    and return a list of result dicts.
 
    Skips (with a warning) any file that raises an exception.
    """
    files = sorted(Path(data_dir).glob(glob))
    if not files:
        raise FileNotFoundError(
            f"No files matching '{glob}' found in '{data_dir}'. "
            "Check DATA_DIR and FILE_GLOB at the top of this script."
        )
 
    if verbose:
        print(f"Found {len(files)} files in '{data_dir}'. Loading …")
 
    results, errors = [], []
    for i, path in enumerate(files, 1):
        try:
            rec = load_single_file(path)
            results.append(rec)
            if verbose and i % 20 == 0:
                print(f"  {i}/{len(files)} loaded …")
        except Exception as exc:
            warnings.warn(f"Skipping {path.name}: {exc}")
            errors.append((path.name, str(exc)))
 
    if verbose:
        print(f"✓ Successfully loaded {len(results)}/{len(files)} files.")
        if errors:
            print(f"✗ {len(errors)} file(s) skipped:")
            for name, msg in errors:
                print(f"    {name}: {msg}")
 
    return results

PHASE_COLOURS = {
    "SOLID":         "#3A86FF",   # blue
    "LIQUID":        "#06D6A0",   # teal-green
    "GAS":           "#FFD166",   # amber
    "SUPERCRITICAL?":"#FF6B6B",   # coral
    "GAS+SOLID":     "#0022FF",   
    "SOLID+GAS":     "#0022FF",
    "LIQUID+SOLID":  "#8338EC",
    "SOLID+LIQUID":  "#8338EC",
}
DEFAULT_COLOUR = "#AAAAAA" 

def plot_pt_diagram(results, save_path="phase_diagram_PT.pdf"):
    T_vals  = np.array([r["T"] for r in results])
    P_vals  = np.array([r["P"] for r in results])
    phases  = [r["phase"] for r in results]
    colours = [PHASE_COLOURS.get(p, DEFAULT_COLOUR) for p in phases]
 
    # ── unique phases present in data ─────────────────────────────────────
    unique_phases = sorted(set(phases))
 
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_facecolor("#0D1117")
    fig.patch.set_facecolor("#0D1117")
 
    sc = ax.scatter(
        T_vals, P_vals,
        c=colours, s=60, alpha=0.85,
        linewidths=0.4, edgecolors="white",
        zorder=3,
    )
 
    # Grid
    ax.grid(color="#2A2F38", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
 
    # Labels
    ax.set_xlabel("Temperature  T*", fontsize=13, color="white", labelpad=8)
    ax.set_ylabel("Pressure  P*",    fontsize=13, color="white", labelpad=8)
    ax.set_title("Phase Diagram  –  P* vs T*", fontsize=15,
                 color="white", fontweight="bold", pad=14)
 
    ax.tick_params(colors="white", labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
 
    # Legend
    handles = [
        mpatches.Patch(facecolor=PHASE_COLOURS.get(p, DEFAULT_COLOUR),
                       edgecolor="white", linewidth=0.5, label=p)
        for p in unique_phases
    ]
    ax.legend(handles=handles, framealpha=0.25, labelcolor="white",
              facecolor="#1A1F27", edgecolor="#555", fontsize=10)
 
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved → {save_path}")
    plt.show()

def plot_vt_diagrams(results, save_path="phase_diagram_VT.pdf"):
    T_vals  = np.array([r["T"] for r in results])
    V_vals  = (1.0 / np.array([r["rho"] for r in results]))*4*N_CELLS**3
    phases  = [r["phase"] for r in results]
    colours = [PHASE_COLOURS.get(p, DEFAULT_COLOUR) for p in phases]
 
    # ── unique phases present in data ─────────────────────────────────────
    unique_phases = sorted(set(phases))
 
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_facecolor("#0D1117")
    fig.patch.set_facecolor("#0D1117")
 
    sc = ax.scatter(
        T_vals, V_vals,
        c=colours, s=60, alpha=0.85,
        linewidths=0.4, edgecolors="white",
        zorder=3,
    )
 
    # Grid
    ax.grid(color="#2A2F38", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
 
    # Labels
    ax.set_xlabel("Temperature  T*", fontsize=13, color="white", labelpad=8)
    ax.set_ylabel("Volume  V*",    fontsize=13, color="white", labelpad=8)
    ax.set_title("Phase Diagram  –  V* vs T*", fontsize=15,
                 color="white", fontweight="bold", pad=14)
 
    ax.tick_params(colors="white", labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
 
    # Legend
    handles = [
        mpatches.Patch(facecolor=PHASE_COLOURS.get(p, DEFAULT_COLOUR),
                       edgecolor="white", linewidth=0.5, label=p)
        for p in unique_phases
    ]
    ax.legend(handles=handles, framealpha=0.25, labelcolor="white",
              facecolor="#1A1F27", edgecolor="#555", fontsize=10)
 
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved → {save_path}")
    plt.show()


def plot_rhot_diagrams(results, save_path="phase_diagram_RHOT.pdf"):
    T_vals  = np.array([r["T"] for r in results])
    rho_vals  = np.array([r["rho"] for r in results])
    phases  = [r["phase"] for r in results]
    colours = [PHASE_COLOURS.get(p, DEFAULT_COLOUR) for p in phases]
 
    # ── unique phases present in data ─────────────────────────────────────
    unique_phases = sorted(set(phases))
 
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_facecolor("#0D1117")
    fig.patch.set_facecolor("#0D1117")
 
    sc = ax.scatter(
        T_vals, rho_vals,
        c=colours, s=60, alpha=0.85,
        linewidths=0.4, edgecolors="white",
        zorder=3,
    )
 
    # Grid
    ax.grid(color="#2A2F38", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
 
    # Labels
    ax.set_xlabel("Temperature  T*", fontsize=13, color="white", labelpad=8)
    ax.set_ylabel("Density  ρ*",    fontsize=13, color="white", labelpad=8)
    ax.set_title("Phase Diagram  –  ρ* vs T*", fontsize=15,
                 color="white", fontweight="bold", pad=14)
 
    ax.tick_params(colors="white", labelsize=10)
    ax.set_xticks(T_vals)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
 
    # Legend
    handles = [
        mpatches.Patch(facecolor=PHASE_COLOURS.get(p, DEFAULT_COLOUR),
                       edgecolor="white", linewidth=0.5, label=p)
        for p in unique_phases
    ]
    ax.legend(handles=handles, framealpha=0.25, labelcolor="white",
              facecolor="#1A1F27", edgecolor="#555", fontsize=10)
 
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved → {save_path}")
    plt.show()

if __name__ == "__main__":
    results = load_all_files(verbose=False)
    plot_pt_diagram(results)
    plot_vt_diagrams(results)
    plot_rhot_diagrams(results)