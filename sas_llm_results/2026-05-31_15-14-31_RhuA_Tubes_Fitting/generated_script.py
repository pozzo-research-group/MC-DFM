#--- Import required packages ----------------------------------------------------
import sys
sys.path.append('../')
from Scattering_Simulator import pairwise_method
from Scattering_Simulator import fitting
from Scattering_Simulator import Read_PDB
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

#--- Load experimental SAXS data --------------------------------------------------
# Experimental data must be in a two‑column text file: q    I(q)
exp_file = 'exp_data.txt'                      # <-- replace with your file name
exp_q, exp_Iq = np.loadtxt(exp_file, unpack=True)

# Convert to the format used by the fitting utilities
exp_data = np.column_stack((exp_q, exp_Iq))

#--- Load simulated datasets (text files) -----------------------------------------
# Each file is assumed to have two columns: q   I(q)
sim_files = ['sim1.txt', 'sim2.txt', 'sim3.txt']   # <-- replace/add your files

sim_q_list  = []   # list to hold q arrays from each simulation
sim_Iq_list = []   # list to hold I(q) arrays from each simulation

for f in sim_files:
    q_sim, Iq_sim = np.loadtxt(f, unpack=True)
    sim_q_list.append(q_sim)
    sim_Iq_list.append(Iq_sim)

#--- Interpolate all simulated curves onto the experimental q grid ---------------
# (You could also use fitting.convert_data, but numpy.interp is sufficient here)
Iq_interp_list = [np.interp(exp_q, q, Iq) for q, Iq in zip(sim_q_list, sim_Iq_list)]

#--- Define objective function for optimisation ----------------------------------
def objective_function(parameters,
                       param_config,
                       exp_data=None,
                       mode="single",
                       plot=False):
    """
    Parameters: array containing [weight_1, weight_2, ..., weight_N, scale, background]
    param_config: dictionary containing bounds for each parameter
    exp_data: 2‑column array (q, I(q)) of experimental data
    """
    # Scale parameters to sensible ranges
    params = fitting.scale_parameters(parameters, param_config)

    # Extract weights and scale
    n_weights = len(param_config) - 2         # Two extra params: scale & background
    weights   = params[:n_weights]
    scale     = params[-2]
    background= params[-1]

    # Normalise weights so they sum to one (optional but keeps optimisation stable)
    weights = weights / np.sum(weights)

    # Construct the weighted sum of the interpolated simulated curves
    model_Iq = np.zeros_like(exp_q)
    for w, Iq in zip(weights, Iq_interp_list):
        model_Iq += w * Iq

    # Apply global scaling and background shift
    model_Iq = scale * model_Iq + background

    # Log‑log error (robust to wide intensity range)
    error = np.mean(np.abs(np.log10(np.abs(exp_data[:, 1])) -
                           np.log10(np.abs(model_Iq))))

    # fitting.run_optimization maximises fitness, so return negative error
    return -error, np.column_stack((exp_q, model_Iq))

#--- Define parameter bounds ------------------------------------------------------
# Example bounds: 3 weights, 2 extra params (scale, background)
param_config = {
    "w1": (0.0, 1.0),          # weight for simulation 1
    "w2": (0.0, 1.0),          # weight for simulation 2
    "w3": (0.0, 1.0),          # weight for simulation 3
    "scale": (1e-6, 1e-3),     # global scaling factor
    "background": (1e-10, 1e-3) # constant background offset
}

#--- Run optimisation -------------------------------------------------------------
# Adjust batch_size, mutation_rate, iterations as needed
best_solution, fitness_history = fitting.run_optimization(
    exp_data,
    param_config,
    batch_size=4,
    mutation_rate=0.1,
    iterations=5,
    objective_function=objective_function,
    mode="single"
)

#--- Evaluate best fit -------------------------------------------------------------
best_solution_scaled = fitting.scale_parameters(best_solution, param_config)
score, best_fit = objective_function(best_solution, param_config, exp_data)

#--- Plot experimental data and best fit -----------------------------------------
# The plotting function expects the model as a 2‑column array (q, I(q))
fitting.plot_simulated_and_experimental(exp_data, best_fit, best_solution_scaled, param_config)

#--- (Optional) – Plot individual component contributions -------------------------
# Normalise weights to plot individual components on same scale
weights = best_solution_scaled[:len(param_config)-2]
weights = weights / np.sum(weights)
for i, (w, Iq) in enumerate(zip(weights, Iq_interp_list)):
    plt.figure()
    plt.semilogy(exp_q, w * Iq, label=f'Component {i+1} (weight={w:.2f})')
    plt.xlabel('q (Å⁻¹)')
    plt.ylabel('I(q) (arbitrary units)')
    plt.legend()
    plt.title(f'Surface of Component {i+1}')
    plt.show()