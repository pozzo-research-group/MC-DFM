# To simulate a polydisperse system of spheres, we need to generate multiple spheres 
# with radii sampled from a normal distribution. Since the MC-DFM method 
# simulates the scattering of specific coordinate sets, we will generate a 
# collection of spheres, each with a different radius, and then 
# combine their scattering curves using the appropriate scaling rules.

import sys
sys.path.append('../')
from Scattering_Simulator import pairwise_method
from Scattering_Simulator import fitting
from Scattering_Simulator import Read_PDB
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

# Function to generate a sphere of a given radius
def generate_sphere_coordinates(radius, n_points=100000):
    """
    Generates coordinates for a sphere with a given radius.
    All atoms are assumed to have an SLD of 1.0 for simplicity.
    """
    phi = np.random.uniform(0, 2*np.pi, n_points)
    costheta = np.random.uniform(-1, 1, n_points)
    theta = np.arccos(costheta)
    
    r = radius # For a solid sphere, we sample the volume
    # To make it a solid sphere rather than a shell, sample r via cube root
    r = radius * np.cbrt(np.random.uniform(0, 1, n_points))
    
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    
    # Create array with x, y, z and SLD (1.0)
    sld = np.ones(n_points)
    coords = np.column_stack((x, y, z, sld))
    return coords

def volume_sphere(radius):
    return (4/3) * np.pi * (radius**3)

# Parameters for polydispersity
mean_radius = 50.0
std_dev = 25.0
num_spheres_to_sample = 50  # Number of spheres to represent the distribution
n_pairwise = 10000000 
histogram_bins = 10000 
q = np.geomspace(0.01, 1, 3000)

# Initialize total scattering
Iq_total = np.zeros_like(q)

# We treat the sample as a mixture where each sphere in our sample represents 
# a portion of the total population. 
# The weight (proportion) of each sphere in the total scattering is 1/num_spheres_to_sample.
prop = 1.0 / num_spheres_to_sample

print(f"Simulating polydisperse spheres (Mean R={mean_radius}, Std={std_dev})...")

for i in range(num_spheres_to_sample):
    # Sample radius from normal distribution
    # Ensure radius is positive
    radius = max(1.0, np.random.normal(mean_radius, std_dev))
    
    # Generate coordinates for this specific sphere
    coords = generate_sphere_coordinates(radius)
    
    # Calculate volume for scaling
    vol = volume_sphere(radius)
    
    # Simulate scattering for this single sphere
    Iq_single = fitting.simulate_scattering(coords, q, histogram_bins, n_pairwise, mode='single')
    
    # Calculate invariant
    Inv = fitting.invariant(np.column_stack((q, Iq_single)))
    
    # Scale the intensity:
    # Because we are modeling a mixture of different sized spheres, 
    # we scale by the volume of the object and its proportion in the mixture.
    Iq_scaled = fitting.scale_intensity(Iq_single, Inv, vol, prop)
    # Add to total
    Iq_total += Iq_scaled

# Plot the resulting polydisperse scattering curve
fitting.plot_intensity(q, Iq_total)
plt.title("Polydisperse Sphere Scattering (Normal Distribution)")
plt.show()