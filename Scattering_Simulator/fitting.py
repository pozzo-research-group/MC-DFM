import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import math
sys.path.append('../')
from Scattering_Simulator import pairwise_method
from Scattering_Simulator import Read_PDB
from genetic_algorithm import GA_functions as GA
from genetic_algorithm import curve_fitting as cf
from scipy import integrate
import torch
import os
from datetime import datetime


def relative_coordinates(volume):
    '''This function is used to center any set of coordinates at the origin'''
    rel_x = volume[:,0] - np.mean(volume[:,0])
    rel_y = volume[:,1] - np.mean(volume[:,1])
    rel_z = volume[:,2] - np.mean(volume[:,2])
    relative_volume = np.hstack((rel_x.reshape(-1,1), rel_y.reshape(-1,1), rel_z.reshape(-1,1), volume[:,-1].reshape(-1,1)))
    return relative_volume

def rotate_coordinates(coords, angle, axis="z"):
    """
    Rotate a set of 3D coordinates about x, y, or z axis.

    Parameters
    ----------
    coords : numpy array (N,3)
        Array of [x,y,z] coordinates
    angle : float
        Rotation angle in degrees
    axis : str
        'x', 'y', or 'z'

    Returns
    -------
    rotated_coords : numpy array (N,3)
        Rotated coordinates
    """

    angle = np.deg2rad(angle)

    if axis == "x":
        R = np.array([
            [1, 0, 0],
            [0, np.cos(angle), np.sin(angle)],
            [0, -np.sin(angle), np.cos(angle)]
        ])

    elif axis == "y":
        R = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])

    elif axis == "z":
        R = np.array([
            [np.cos(angle), np.sin(angle), 0],
            [-np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])

    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")

    rotated_coords = coords @ R.T

    return rotated_coords


def extract_data(file_path):
    """
    Extract numeric data from a .dat file.
    Skips comment lines starting with # or non-numeric rows.
    Returns numpy array and pandas DataFrame.
    """

    data = []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            try:
                # Split by whitespace and convert to float
                row = [float(value) for value in line.split()]
                data.append(row)
            except ValueError:
                # Skip lines that can't be converted to float
                continue

    if not data:
        raise ValueError("No numeric data found in file.")

    data_array = np.array(data)

    # Create DataFrame with generic column names
    columns = [f"col_{i+1}" for i in range(data_array.shape[1])]
    df = pd.DataFrame(data_array, columns=columns)

    return data_array, df

def scale_parameters(parameters, param_config):
    """
    Scales parameters from [0,1] to user-defined bounds.

    parameters: 1D numpy array
    param_config:
        {"param_name": (lower, upper)}
        or
        {"param_name": (lower, upper, "int")}

    Returns:
        dict of scaled parameters
    """
    scaled = {}

    for i, (name, config) in enumerate(param_config.items()):

        lower, upper = config[:2]
        param_type = config[2] if len(config) > 2 else "float"

        value = parameters[i] * (upper - lower) + lower

        if param_type == "int":
            value = int(round(value))
            value = max(lower, min(upper, value))  # ensure bounds

        scaled[name] = value

    return scaled

def convert_data(data, model):
    '''This function bins the model data so that it has the same number of data points as the experimental data
    inputs:
    - data: the experimental data with q and intensity as the two colums in the 2D array
    - model: the simulated data with q and intensity as the two columns in the 2D array
    output:
    - new_model_data: the binned model data with q and I as the two columns in the 2D array'''
    model_x = model[:,0]
    model_y = model[:,1]
    index = np.linspace(0, len(model_x)-1, len(model_x)) 
    model_q_new = []
    model_I_new = []
    for i in range(len(data)):
        data_q = data[i,0]
        array = np.abs(model_x - data_q)
        array = np.hstack((array.reshape(-1,1), index.reshape(-1,1)))
        array = array[np.argsort(array[:, 0])]
        loc = int(array[0,1])
        model_q_new.append(model_x[loc])
        model_I_new.append(model_y[loc])
    q = np.array(model_q_new).reshape(-1,1)
    I = np.array(model_I_new).reshape(-1,1)
    new_model_data = np.hstack((q, I))
    return new_model_data

def create_lattice_coords(d):
    """
    Creates simple dimer lattice.
    """
    return np.array([[0, 0, 0], [d, 0, 0]])

def invariant(data):
    q = data[:, 0]
    I = data[:, 1]
    return np.abs(integrate.simpson(q**2 * I, q))

def simulate_scattering(
    coordinates,
    q,
    histogram_bins,
    n_samples,
    mode="single"
):
    simulator = pairwise_method.scattering_simulator(n_samples)
    simulator.sample_building_block(coordinates) 
    simulator.use_building_block_as_structure()
    if mode == "single":
        Iq = simulator.simulate_scattering_curve_fast(
            coordinates, histogram_bins, q
        ).cpu()
        return np.array(Iq)

    elif mode == "multiple":
        simulator = pairwise_method.scattering_simulator(n_samples)
        simulator.sample_building_block(coordinates)
        simulator.use_building_block_as_structure()
        I_q = simulator.simulate_multiple_scattering_curves(
            coordinates, histogram_bins, q
        ).cpu()

        return np.array(torch.mean(I_q[:, 1:], dim=1))

    else:
        raise ValueError("mode must be 'single' or 'multiple'")
    

def simulate_scattering_lattice(
    coordinates,
    lattice_coords,
    q,
    histogram_bins,
    n_samples,
    mode="single"
):
    simulator = pairwise_method.scattering_simulator(n_samples)

    if mode == "single":
        Iq = simulator.simulate_scattering_curve_fast_lattice(
            coordinates, lattice_coords, histogram_bins, q
        ).cpu()
        return np.array(Iq)

    elif mode == "multiple":
        simulator.sample_building_block(coordinates)
        simulator.sample_lattice_coordinates(lattice_coords)
        simulator.calculate_structure_coordinates()

        I_q = simulator.simulate_multiple_scattering_curves_lattice_coords(
            coordinates, lattice_coords, histogram_bins, q
        ).cpu()
        return np.array(torch.mean(I_q[:, 1:], dim=1))

    else:
        raise ValueError("mode must be 'single' or 'multiple'")
    

def obtain_scattering_object_lattice(
    coordinates,
    lattice_coords,
    n_samples,
):
    simulator = pairwise_method.scattering_simulator(n_samples)
    simulator.sample_building_block(coordinates)
    simulator.sample_lattice_coordinates(lattice_coords)
    simulator.calculate_structure_coordinates()
    struc = simulator.structure_coordinates_1[::50]
    return struc

def obtain_scattering_object(
    coordinates,
    n_samples,
):
    simulator = pairwise_method.scattering_simulator(n_samples)
    simulator.sample_building_block(coordinates)
    simulator.use_building_block_as_structure()
    struc = simulator.structure_coordinates_1[::50]
    return struc



def evaluate_population(
    population,
    param_config,
    exp_data,
    objective_function,
    mode="single"
):
    fitness = []
    for individual in population:
        score, _ = objective_function(
            individual,
            param_config,
            exp_data=exp_data,
            mode=mode,
            plot=False
        )
        fitness.append(score)

    return np.array(fitness).reshape(-1, 1)


def run_optimization(
    exp_data,
    param_config,
    batch_size,
    mutation_rate,
    iterations,
    objective_function, 
    mode="single"
):
    """
    param_config controls number of parameters automatically.
    """

    n_params = len(param_config)

    # Random population
    x = np.random.rand(batch_size, n_params)

    y = evaluate_population(x, param_config, exp_data, objective_function, mode)

    alg = GA.genetic_algorithm(batch_size, mutation_rate)

    for _ in range(iterations):
        x = alg.run(x, y)
        y = evaluate_population(x, param_config, exp_data, objective_function, mode)

    return alg.best_solution(), alg.max_fitness_lst

def save_parameters_txt(parameters, filename="best_solution_scaled.txt"):
    """
    Save parameter dictionary to a text file.

    Parameters
    ----------
    parameters : dict
        Dictionary of parameter names and values
    filename : str
        Output text file name
    """
    
    with open(filename, "w") as f:
        f.write("Best Solution Parameters\n")
        f.write("------------------------\n")
        
        for name, value in parameters.items():
            f.write(f"{name}: {value}\n")

def save_saxs_txt(data, path):
    """
    Save SAXS data (q, I) to a text file.

    Parameters
    ----------
    data : numpy array (N,2)
        SAXS data with columns [q, I]
    filename : str
        Output text file name
    """

    header = "q I"
    np.savetxt(path, data, header=header)
    
def plot_structure(coordinates, n_samples):
    struc = obtain_scattering_object(coordinates, n_samples)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(
        struc[:,0][0:10000],
        struc[:,1][0:10000],
        struc[:,2][0:10000],
        alpha=1,
        s=20
    )

    # ---- compute limits ----
    x_min, x_max = struc[:,0].min(), struc[:,0].max()
    y_min, y_max = struc[:,1].min(), struc[:,1].max()
    z_min, z_max = struc[:,2].min(), struc[:,2].max()

    # centers
    x_mid = (x_max + x_min) / 2
    y_mid = (y_max + y_min) / 2
    z_mid = (z_max + z_min) / 2

    # largest range
    max_range = max(
        x_max - x_min,
        y_max - y_min,
        z_max - z_min
    ) / 2

    # set equal limits
    ax.set_xlim(x_mid - max_range, x_mid + max_range)
    ax.set_ylim(y_mid - max_range, y_mid + max_range)
    ax.set_zlim(z_mid - max_range, z_mid + max_range)

    ax.set_xlabel('x-axis ($\\AA$)', fontsize=18)
    ax.set_ylabel('y-axis ($\\AA$)', fontsize=18)
    ax.set_zlabel('z-axis ($\\AA$)', fontsize=18)

    def rotate(angle):
        ax.view_init(azim=angle)
    filenames = os.listdir('../sas_llm_results')[-1]
    path = '../sas_llm_results/' + filenames
    dt = datetime.now().strftime("%H-%M-%S")
    plt.savefig(path + '/Scattering_Structure' + dt + '.png', dpi=600, bbox_inches="tight")
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # x-y projection
    axes[0].scatter(
        struc[:, 0][0:10000],
        struc[:, 1][0:10000],
        alpha=1,
        s=20
    )
    axes[0].set_xlabel('x-axis ($\\AA$)', fontsize=14)
    axes[0].set_ylabel('y-axis ($\\AA$)', fontsize=14)
    axes[0].set_title('x-y Projection')
    #axes[0].set_aspect('equal', adjustable='box')

    # x-z projection
    axes[1].scatter(
        struc[:, 0][0:10000],
        struc[:, 2][0:10000],
        alpha=1,
        s=20
    )
    axes[1].set_xlabel('x-axis ($\\AA$)', fontsize=14)
    axes[1].set_ylabel('z-axis ($\\AA$)', fontsize=14)
    axes[1].set_title('x-z Projection')
    #axes[1].set_aspect('equal', adjustable='box')

    # y-z projection
    axes[2].scatter(
        struc[:, 1][0:10000],
        struc[:, 2][0:10000],
        alpha=1,
        s=20
    )
    axes[2].set_xlabel('y-axis ($\\AA$)', fontsize=14)
    axes[2].set_ylabel('z-axis ($\\AA$)', fontsize=14)
    axes[2].set_title('y-z Projection')
    #axes[2].set_aspect('equal', adjustable='box')

    plt.tight_layout()
    filenames = os.listdir('../sas_llm_results')[-1]
    path = '../sas_llm_results/' + filenames
    dt = datetime.now().strftime("%H-%M-%S")
    plt.savefig(path + '/Scattering_Structure_2D' + dt + '.png', dpi=600, bbox_inches="tight")
    plt.show()

    return fig, ax

def plot_structure_lattice(coordinates, lattice_coordinates, n_samples):
    struc = obtain_scattering_object_lattice(coordinates, lattice_coordinates, n_samples)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(
        struc[:,0][0:10000],
        struc[:,1][0:10000],
        struc[:,2][0:10000],
        alpha=1,
        s=20
    )

    # ---- compute limits ----
    x_min, x_max = struc[:,0].min(), struc[:,0].max()
    y_min, y_max = struc[:,1].min(), struc[:,1].max()
    z_min, z_max = struc[:,2].min(), struc[:,2].max()

    # centers
    x_mid = (x_max + x_min) / 2
    y_mid = (y_max + y_min) / 2
    z_mid = (z_max + z_min) / 2

    # largest range
    max_range = max(
        x_max - x_min,
        y_max - y_min,
        z_max - z_min
    ) / 2

    # set equal limits
    ax.set_xlim(x_mid - max_range, x_mid + max_range)
    ax.set_ylim(y_mid - max_range, y_mid + max_range)
    ax.set_zlim(z_mid - max_range, z_mid + max_range)
    
    ax.set_xlabel('x-axis ($\\AA$)', fontsize=18)
    ax.set_ylabel('y-axis ($\\AA$)', fontsize=18)
    ax.set_zlabel('z-axis ($\\AA$)', fontsize=18)

    def rotate(angle):
        ax.view_init(azim=angle)

    
    filenames = os.listdir('../sas_llm_results')[-1]
    path = '../sas_llm_results/' + filenames
    dt = datetime.now().strftime("%H-%M-%S")
    plt.savefig(path + '/Scattering_Structure' + dt + '.png', dpi=600, bbox_inches="tight")
    plt.show()

    # ---- x-y projection ----

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    # x-y projection
    axes[0].scatter(
        struc[:, 0][0:10000],
        struc[:, 1][0:10000],
        alpha=1,
        s=20
    )
    axes[0].set_xlabel('x-axis ($\\AA$)', fontsize=14)
    axes[0].set_ylabel('y-axis ($\\AA$)', fontsize=14)
    axes[0].set_title('x-y Projection')
    #axes[0].set_aspect('equal', adjustable='box')

    # x-z projection
    axes[1].scatter(
        struc[:, 0][0:10000],
        struc[:, 2][0:10000],
        alpha=1,
        s=20
    )
    axes[1].set_xlabel('x-axis ($\\AA$)', fontsize=14)
    axes[1].set_ylabel('z-axis ($\\AA$)', fontsize=14)
    axes[1].set_title('x-z Projection')
    #axes[1].set_aspect('equal', adjustable='box')

    # y-z projection
    axes[2].scatter(
        struc[:, 1][0:10000],
        struc[:, 2][0:10000],
        alpha=1,
        s=20
    )
    axes[2].set_xlabel('y-axis ($\\AA$)', fontsize=14)
    axes[2].set_ylabel('z-axis ($\\AA$)', fontsize=14)
    axes[2].set_title('y-z Projection')
    #axes[2].set_aspect('equal', adjustable='box')

    plt.tight_layout()

    filenames = os.listdir('../sas_llm_results')[-1]
    path = '../sas_llm_results/' + filenames
    dt = datetime.now().strftime("%H-%M-%S")
    plt.savefig(path + '/Scattering_Structure_2D' + dt + '.png', dpi=600, bbox_inches="tight")
    plt.show()
    return fig, ax

def plot_intensity(q, I_q):
    plt.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots(figsize=(7,7))
    plt.plot(q, I_q, linewidth = 3, label = 'Simulation', color ='k')
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('I(q) (Arb. Unit.)')
    plt.xlabel('q ($\\AA^{-1}$)')
    filenames = os.listdir('../sas_llm_results')[-1]
    path = '../sas_llm_results/' + filenames
    plt.savefig(path  + '/Intensity.png', dpi=600, bbox_inches="tight")
    #plt.show()
    save_saxs_txt(np.hstack((q.reshape(-1,1), I_q.reshape(-1,1))), path + '/Intensity.txt')


def scale_intensity(Iq, inv, volume_or_N_subunits, proportion):
    return Iq/inv*volume_or_N_subunits*proportion


def plot_simulated_and_experimental(exp_data, best_fit, best_solution, param_config):
    # Plotting #
    plt.rcParams.update({'font.size': 22})
    fig, ax = plt.subplots(figsize=(7,7), ncols = 1, nrows=2,  gridspec_kw={'height_ratios': [3, 1]})
    q = best_fit[:,0]
    ax[0].errorbar(exp_data[:,0], exp_data[:,1], yerr=exp_data[:,2], label = 'Experimental Data', color = 'blue', capsize=3, fmt='o', zorder=0)
    ax[0].plot(best_fit[:,0], best_fit[:,1], color = 'k', linewidth=3, label = 'Model Best Fit')
    ax[0].set_yscale('log')
    ax[0].set_xscale('log')
    ax[0].set_ylabel('I(q) ($cm^{-1}$)')
    ax[0].set_xticks([])
    ax[0].legend(fontsize=16)
    plt.subplots_adjust(hspace=0)
    model_data = cf.convert_data(exp_data, np.hstack((q.reshape(-1,1), best_fit[:,1].reshape(-1,1))))
    model_residual = (np.log(model_data[:,1]/exp_data[:,1]))
    ax[1].plot(exp_data[:,0], model_residual, color ='k', linewidth = 3)
    ax[1].set_ylim([-2,2])
    ax[1].set_yticks([-1,1])
    ax[1].set_xscale('log')
    ax[1].set_ylabel('R')
    ax[1].hlines(0, np.min(exp_data[:,0]), np.max(exp_data[:,0]), color='red', linewidth=3, linestyle='--')
    ax[1].set_xlabel('q ($\\AA^{-1}$)')
    filenames = os.listdir('../sas_llm_results')[-1]
    path = '../sas_llm_results/' + filenames
    plt.savefig(path + '/Intensity.png', dpi=600, bbox_inches="tight")
    save_parameters_txt(best_solution, filename= path + "/Fit_Results.txt")
    save_parameters_txt(param_config, filename= path + "/Fit_Limits.txt")
    save_saxs_txt(best_fit,  path + '/Intensity_Best_Fit.txt')