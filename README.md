# Molecular Dynamics Engine

A Python molecular dynamics simulator for noble-gas systems and phase-transition analysis.

## What it does

This project runs molecular dynamics simulations with Lennard-Jones interactions for noble-gas systems.

It can:

- build an FCC lattice at a chosen reduced density
- sweep over multiple reduced temperatures in parallel
- record trajectories, energies, momentum, radial distribution functions, and mean squared displacement
- estimate diffusion coefficients from the MSD curve
- classify results into solid, liquid, gas, or mixed phases
- save simulation output to HDF5 for later analysis
- load saved phase-transition data and plot phase diagrams
- generate diagnostics and animation data from stored runs

The implementation uses a few standard simulation methods and optimisations. It uses a velocity-Verlet integrator, periodic boundaries with the minimum-image convention, and a Verlet neighbour list with a skin distance so pair forces are not rebuilt from scratch every step. The Lennard-Jones potential is shifted with a force correction at the cutoff. During equilibration, velocities are rescaled to the target reduced temperature. The temperature sweep also runs across multiple worker processes when more than one temperature is requested.

## Installation

```bash
cd md_Engine
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run the main simulator from the project root:

```bash
python md_engine.py
```

The script prompts for simulation settings, then writes `simulation_config.yaml` and result files in HDF5 format.

Run a single phase-transition point:

```bash
python PhaseTransition/run_one_point.py --rho 0.3000 --T_star 1.0000 --outdir PhaseTransition/data_4
```

Plot phase-diagram data from the saved files:

```bash
python PhaseTransition/main.py
```

The main simulation can also produce RDF and MSD plots, run diagnostics, and build animations through the methods on `MDSimulation` in `md_engine.py`.

## Requirements/Dependencies

Python 3 with the packages listed in `requirements.txt`.

Key packages used by the project include `numpy==2.4.4`, `scipy==1.17.1`, `matplotlib==3.10.8`, `h5py==3.16.0`, `PyYAML==6.0.3`, and `hdf5plugin==6.0.0`.

## Configuration

`md_engine.py` reads and writes `simulation_config.yaml` in the project root. The phase-transition scripts also use the constants at the top of `PhaseTransition/main.py` and `PhaseTransition/run_one_point.py` to choose the data directory, file pattern, and system size.
