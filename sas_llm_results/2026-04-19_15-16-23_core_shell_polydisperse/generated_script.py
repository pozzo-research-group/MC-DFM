# Imports required for the MC-DFM simulation
import sys
sys.path.append('../')
from Scattering_Simulator import pairwise_method
from Scattering_Simulator import fitting
from Scattering_Simulator import Read_PDB
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------
# Helper functions to generate random points inside a sphere or shell
# --------------------------------------------------------------------
def generate_uniform_points_in_sphere(radius, N, seed=None):
    """
    Generate N random points uniformly inside a sphere of given radius.
    """
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    r = radius * np.cbrt(u)
    phi = rng.uniform(0, 2*np.pi, N)
    costheta = rng.uniform(-1, 1, N)
    theta = np.arccos(costheta)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.column_stack((x, y, z))

def generate_uniform_points_in_shell(inner_r, outer_r, N, seed=None):
    """
    Generate N random points uniformly inside a spherical shell
    bounded by inner_r and outer_r.
    """
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    r = ((outer_r**3 - inner_r**3) * u + inner_r**3)**(1/3.0)
    phi = rng.uniform(0, 2*np.pi, N)
    costheta = rng.uniform(-1, 1, N)
    theta = np.arccos(costheta)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.column_stack((x, y, z))

# --------------------------------------------------------------------
# Parameters for the polydisperse core–shell sphere system
# --------------------------------------------------------------------
core_radius_mean   = 30.0        # Å
core_radius_std    = 15.0        # Å
shell_thickness    = 15.0        # Å (constant)
SLD_core           = 5.0
SLD_shell          = 3.0

# --------------------------------------------------------------------
# Generate a Gaussian distribution of core radii and discretize
# --------------------------------------------------------------------
np.random.seed(42)
num_samples = 1000
core_radii_all = np.random.normal(loc=core_radius_mean, scale=core_radius_std, size=num_samples)
core_radii_all = core_radii_all[core_radii_all > 0]   # keep only positive radii

num_bins = 40
hist, bin_edges = np.histogram(core_radii_all, bins=num_bins, density=False)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])   # representative core radius for each bin

# Compute volumes for each size class
vol_core = (4.0/3.0) * np.pi * bin_centers**3
# Minimal core volume used for scaling
V_min = vol_core.mean()

# Proportions of each size class
proportions = hist / hist.sum()

# --------------------------------------------------------------------
# Prepare core–shell objects for each size class
# --------------------------------------------------------------------
objects = []
base_core_points = 2000   # base number of core points before scaling

for i, r_core in enumerate(bin_centers):
    r_shell_outer = r_core + shell_thickness
    
    V_c = (4.0/3.0) * np.pi * r_core**3
    V_s = (4.0/3.0) * np.pi * (r_shell_outer**3 - r_core**3)
    
    # Scale core points proportionally to volume
    N_core = max(int(base_core_points * V_c / V_min), 1)
    # Use volume ratio to determine shell points
    N_shell = max(int(N_core * V_s / V_c), 1)
    
    # Generate coordinates
    core_coords = generate_uniform_points_in_sphere(r_core, N_core, seed=42+i)
    shell_coords = generate_uniform_points_in_shell(r_core, r_shell_outer, N_shell, seed=77+i)
    
    # Add SLD difference as the 4th column
    core_coords = np.hstack((core_coords, np.full((N_core, 1), SLD_core)))
    shell_coords = np.hstack((shell_coords, np.full((N_shell, 1), SLD_shell)))
    
    # Combine core and shell into a single coordinate array
    sphere_coords = np.vstack((core_coords, shell_coords))
    objects.append(sphere_coords)

# --------------------------------------------------------------------
# Generate the center-of-mass coordinates for all spheres in the box
# --------------------------------------------------------------------
size_of_object = 2 * np.max(bin_centers + shell_thickness)  # ensure spheres do not overlap
center_of_mass_coordinates = pairwise_method.generate_coordinates_for_polydisperse_system(
    size_of_object=size_of_object)

# --------------------------------------------------------------------
# Place the spheres on the generated coordinates with the specified proportions
# --------------------------------------------------------------------
all_coordinates = pairwise_method.place_objects_on_coordinates(
    center_of_mass_coordinates,
    objects,
    proportions)

# --------------------------------------------------------------------
# Simulate the scattering curve for the polydisperse core–shell system
# --------------------------------------------------------------------
n_pairwise = 10000000
histogram_bins = 10000
q = np.geomspace(0.01, 1.0, 3000)

Iq = fitting.simulate_scattering(
    all_coordinates,
    q,
    histogram_bins,
    n_pairwise,
    mode='single')

# --------------------------------------------------------------------
# Plot the results
# --------------------------------------------------------------------
fitting.plot_intensity(q, Iq)
fitting.plot_structure(all_coordinates, 1000000)