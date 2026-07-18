# Since the user is asking for a specific geometric shape (cylinder) rather than a PDB file, 
# I will generate the coordinates for a cylinder. 
# Note: The MC-DFM approach relies on coordinate arrays (x, y, z, SLD).
# To model polydispersity, we will simulate multiple cylinders sampled from the normal distribution 
# and sum their scaled intensities.

import sys
sys.path.append('../')
from Scattering_Simulator import pairwise_method
from Scattering_Simulator import fitting
from Scattering_Simulator import Read_PDB
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

# Function to generate coordinates for a cylinder
def generate_cylinder_coordinates(length, radius, n_points=5000):
    """
    Generates coordinates for a cylinder.
    Returns: array of [x, y, z, SLD]
    """
    # Generate random points within the cylinder
    # z is along the length
    z = np.random.uniform(-length/2, length/2, n_points)
    # r is the radial distance from center
    r = radius * np.sqrt(np.random.uniform(0, 1, n_points))
    theta = np.random.uniform(0, 2*np.pi, n_points)
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    # Create the array with a dummy SLD of 1.0 (contrast)
    coords = np.zeros((n_points, 4))
    coords[:, 0] = x
    coords[:, 1] = y
    coords[:, 2] = z
    coords[:, 3] = 1.0 
    return coords

def volume_cylinder(length, radius):
    return np.pi * (radius**2) * length

# Parameters
L = 200.0       # Length in Angstrom
R_mean = 20.0   # Mean radius in Angstrom
R_std = 10.0    # Std deviation in Angstrom
n_samples = 50   # Number of cylinders to sample for polydispersity
n_points_per_cyl = 10000 # Points per cylinder for sampling

# Scattering setup
n_pairwise = 10000000
histogram_bins = 10000
q = np.geomspace(0.01, 0.5, 3000)

# To model polydispersity, we sample radii from a normal distribution
# We will simulate several cylinders and sum their intensities.
# To represent a continuous distribution via Monte Carlo, 
# we scale each by its proportion in the sample.

Iq_total = np.zeros_like(q)

# Loop to simulate multiple cylinders representing the distribution
for i in range(n_samples):
    # Sample radius from normal distribution
    # We use max(1, ...) to ensure radius is physically valid
    R_sampled = max(1.0, np.random.normal(R_mean, R_std))
    
    # Generate coordinates
    coords = generate_cylinder_coordinates(L, R_sampled, n_points=n_points_per_cyl)
    
    # Simulate scattering for this specific cylinder
    Iq_single = fitting.simulate_scattering(coords, q, histogram_bins, n_pairwise, mode='single')
    
    # Calculate invariant
    Inv_single = fitting.invariant(np.column_stack((q, Iq_single)))
    
    # Calculate volume for scaling (polydispersity of objects requires volume scaling)
    vol = volume_cylinder(L, R_sampled)
    
    # Each cylinder in our sample represents 1/n_samples of the total mixture
    prop = 1.0 / n_samples
    
    # Scale intensity based on its volume and proportion
    Iq_scaled = fitting.scale_intensity(Iq_single, Inv_single, vol, prop)
    
    # Add to total
    Iq_total += Iq_scaled

# Plotting the resulting polydisperse scattering curve
fitting.plot_intensity(q, Iq_total)

# Note: Plotting the "structure" for a polydisperse ensemble is not represented 
# by a single structure object, so we plot the mean cylinder structure for visualization.
mean_coords = generate_cylinder_coordinates(L, R_mean, 10000)
fitting.plot_structure(mean_coords, 100000)