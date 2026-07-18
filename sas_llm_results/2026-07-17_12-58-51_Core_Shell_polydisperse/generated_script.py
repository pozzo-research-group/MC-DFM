# To simulate a polydisperse mixture of core-shell spheres, we need to generate multiple spheres 
# with core radii drawn from a Gaussian distribution, calculate their volumes to scale 
# the points (Monte Carlo sampling) for each component, and then sum their intensities.

import sys
sys.path.append('../')
from Scattering_Simulator import pairwise_method
from Scattering_Simulator import fitting
from Scattering_Simulator import Read_PDB
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

# Set seed for reproducibility of the Gaussian sampling
np.random.seed(42)

def generate_core_shell_sphere(core_radius, shell_thickness, core_sld, shell_sld, n_points):
    """
    Generates coordinates for a single core-shell sphere.
    The number of points for core and shell is scaled by their volume ratio.
    """
    shell_radius = core_radius + shell_thickness
    vol_core = (4/3) * np.pi * (core_radius**3)
    vol_shell = (4/3) * np.pi * (shell_radius**3 - core_radius**3)
    total_vol = vol_core + vol_shell
    
    # Calculate number of points for each component based on volume ratio
    n_core = int(n_points * (vol_core / total_vol))
    n_shell = n_points - n_core
    
    # Sample core points (uniformly within sphere)
    # Using the transformation method for uniform sampling in a sphere
    r_core = core_radius * (np.random.rand(n_core)**(1/3))
    theta_core = np.arccos(2 * np.random.rand(n_core) - 1)
    phi_core = 2 * np.pi * np.random.rand(n_core)
    
    x_core = r_core * np.sin(theta_core) * np.cos(phi_core)
    y_core = r_core * np.sin(theta_core) * np.sin(phi_core)
    z_core = r_core * np.cos(theta_core)
    core_coords = np.column_stack((x_core, y_core, z_core, np.full(n_core, core_sld)))
    
    # Sample shell points (uniformly within spherical shell)
    r_shell = np.power(np.random.rand(n_shell) * (shell_radius**3 - core_radius**3) + core_radius**3, 1/3)
    theta_shell = np.arccos(2 * np.random.rand(n_shell) - 1)
    phi_shell = 2 * np.pi * np.random.rand(n_shell)
    
    x_shell = r_shell * np.sin(theta_shell) * np.cos(phi_shell)
    y_shell = r_shell * np.sin(theta_shell) * np.sin(phi_shell)
    z_shell = r_shell * np.cos(theta_shell)
    shell_coords = np.column_stack((x_shell, y_shell, z_shell, np.full(n_shell, shell_sld)))
    
    return np.vstack((core_coords, shell_coords)), vol_core + vol_shell

# Parameters
core_mean = 30.0
core_std = 15.0
shell_thickness = 15.0
core_sld = 5.0
shell_sld = 3.0
n_spheres = 50  # Number of spheres in the polydisperse mixture
n_points_per_sphere = 50000 # Points per sphere for simulation
n_pairwise = 10000000
histogram_bins = 10000
q = np.geomspace(0.01, 0.5, 3000)

Iq_total = np.zeros_like(q)

# We assume equal proportions of each sphere size in the mixture for the weighting
prop_per_sphere = 1.0 / n_spheres

print(f"Simulating {n_spheres} spheres...")

for i in range(n_spheres):
    # Sample core radius from Gaussian distribution
    r_core_sampled = np.random.normal(core_mean, core_std)
    # Ensure radius is positive
    r_core_sampled = max(r_core_sampled, 1.0)
    
    # Generate structure
    coords, vol = generate_core_shell_sphere(r_core_sampled, shell_thickness, core_sld, shell_sld, n_points_per_sphere)
    
    # Simulate scattering for this specific sphere
    Iq_i = fitting.simulate_scattering(coords, q, histogram_bins, n_pairwise, mode='single')
    
    # Calculate invariant
    Inv_i = fitting.invariant(np.column_stack((q, Iq_i)))
    
    # Scale intensity by volume and proportion
    # Since we are treating them as a mixture of objects, we scale by volume 
    # and the relative proportion (1/N)
    Iq_i_scaled = fitting.scale_intensity(Iq_i, Inv_i, vol, prop_per_sphere)
    
    Iq_total += Iq_i_scaled

# Plot the results
fitting.plot_intensity(q, Iq_total)

# Note: We cannot plot the structure_lattice easily for a polydisperse mixture 
# of different sized spheres using the lattice function, but we can visualize one.
print("Plotting one sample sphere from the distribution...")
sample_coords, _ = generate_core_shell_sphere(core_mean, shell_thickness, core_sld, shell_sld, 10000)
fitting.plot_structure(sample_coords, 10000)