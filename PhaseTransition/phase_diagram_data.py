import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from md_engine import MDSimulation, SimulationConfig
import numpy as np



rho = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
T_star = np.linspace(0.5, 2.0, 21)


for r in rho:
    config = SimulationConfig(n_cells=10,t_star_values=T_star, rho_star = r,prod_steps= 6000, n_workers=5)
    sim = MDSimulation(config)
    sim.run(plot_rdf= True, plot_msd= True, rdf_plot_name=f"rdf_rho_{r}.pdf", msd_plot_name=f"msd_rho_{r}.pdf", to_save=True,results_name=f"results_rho_{r}.h5")
    
