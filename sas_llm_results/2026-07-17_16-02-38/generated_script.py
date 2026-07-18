# Import necessary packages
import sys
sys.path.append('../')
from Scattering_Simulator import pairwise_method
from Scattering_Simulator import fitting
from Scattering_Simulator import Read_PDB
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

# Function to generate a sphere with a specific radius
# To ensure the object has at least 50,000 coordinates as requested
def generate_sphere(radius, n_points=60000):
    phi = np.random.uniform(0, 2 * np.pi, n_points)
    costheta = np.random.uniform(-1, 1, n_points)
    theta = np.arccos(costheta)
    
    r = radius
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    
    # For a solid sphere simulation, we sample volume points
    # Scaling points to represent the volume density
    u = np.random.uniform(0, 1, n_points)
    r_vol = radius * (u**(1/3))
    x_vol = r_vol * np.sin(theta) * np.cos(phi)
    y_vol = r_vol * np.sin(theta) * np.sin(phi)
    z_vol = r_vol * np.cos(theta)
    
    # Creating the coordinate array: [x, y, z, SLD]
    # Assuming SLD = 1 for a generic scatterer
    coords = np.zeros((n_points, 4))
    coords[:, 0] = x_vol
    coords[:, 1] = y_vol
    coords[:, 2] = z_vol
    coords[:, 3] = 1.0 
    return coords

def volume_sphere(radius):
    return (4/3) * np.pi * (radius**3)

# Parameters for polydispersity
mean_radius = 50
std_dev = 25
n_samples = 10 # Number of different sizes to sample to represent the distribution

# Simulation parameters
n_pairwise = 10000000
histogram_bins = 10000
q = np.geomspace(0.01, 1, 3000)

# To simulate polydispersity, we sample multiple radii from the normal distribution,
# simulate their scattering, scale them by their volume and proportion, and sum them.
# We assume an equal proportion for each sampled size in this implementation.

total_Iq = np.zeros_like(q)
prop_per_size = 1.0 / n_samples

for i in range(n_samples):
    # Sample a radius from the normal distribution
    current_radius = np.random.normal(mean_radius, std_dev)
    
    # Ensure radius is positive
    if current_radius <= 0:
        current_radius = 1e-5
        
    # Generate coordinates for this specific sphere
    coords = generate_sphere(current_radius)
    
    # Calculate volume for scaling
    vol = volume_sphere(current_radius)
    
    # Simulate scattering for this single sphere
    Iq_single = fitting.simulate_scattering(coords, q, histogram_bins, n_pairwise, mode='single')
    
    # Calculate invariant
    Inv_single = fitting.invariant(np.column_stack((q, Iq_single)))
    
    # Scale the intensity by volume and its proportion in the mixture
    # Using fitting.scale_intensity(Iq, Invariant, Volume, Proportion)
    Iq_scaled = fitting.scale_intensity(Iq_single, Inv_single, vol, prop_per_size)
    
    # Add to the total scattering
    total_Iq += Iq_scaled

# Plot the resulting polydisperse scattering curve
fitting.plot_intensity(q, total_Iq)

# Note: Because this is a mixture of different sizes, 
# 'fitting.plot_structure' is typically used for a single object.