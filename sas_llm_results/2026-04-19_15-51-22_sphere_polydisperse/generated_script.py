# ------------------------------------------------------------
# Imports required for the MC-DFM simulation
# ------------------------------------------------------------
import sys
sys.path.append('../')
from Scattering_Simulator import pairwise_method
from Scattering_Simulator import fitting
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------
# Helper function to generate random points uniformly inside a sphere
# ------------------------------------------------------------
def generate_uniform_points_in_sphere(radius, N, seed=None):
    """
    Generates N random points uniformly distributed inside a sphere of given radius.
    """
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    r = radius * np.cbrt(u)                       # radius^3 proportional to u
    phi = rng.uniform(0, 2*np.pi, N)
    costheta = rng.uniform(-1, 1, N)
    theta = np.arccos(costheta)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.column_stack((x, y, z))

# ------------------------------------------------------------
# Parameters for the polydisperse sphere system
# ------------------------------------------------------------
radius_mean   = 50.0      # Å
radius_std    = 25.0      # Å
num_samples   = 10000     # Number of random radii drawn from the distribution
num_bins      = 40        # Discrete size classes to represent the distribution
points_per_sphere = 20000  # Base number of points per sphere type before volume scaling
total_spheres = 100        # Total number of spheres to be placed in the box

rng = np.random.default_rng(42)

# Sample radii from the normal distribution and keep only positive values
sample_radii = rng.normal(radius_mean, radius_std, size=num_samples)
sample_radii = sample_radii[sample_radii > 0]

# Bin the radii to form discrete sphere size classes
hist, bin_edges = np.histogram(sample_radii, bins=num_bins, density=False)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])   # Representative radius for each bin

# Compute volume for each size class
volumes = (4.0/3.0) * np.pi * bin_centers**3

# Determine the probability (proportion) of each size class in the sample
proportions = hist / hist.sum()

# Compute the average volume to use for scaling point counts
avg_volume = volumes.mean()

# Prepare the list of coordinate arrays (one per size class)
objects = []
for i, rad in enumerate(bin_centers):
    # Number of points for this sphere type, scaled by its volume relative to the average
    N = int(points_per_sphere * (volumes[i] / avg_volume))
    if N < 1:
        N = 1
    coords = generate_uniform_points_in_sphere(rad, N, seed=42+i)
    # Add the SLD difference as the 4th column (uniform SLD; adjust if needed)
    coords = np.hstack((coords, np.full((N, 1), 1.0)))
    objects.append(coords)

# ------------------------------------------------------------
# Generate the center‑of‑mass coordinates for all spheres in the box
# ------------------------------------------------------------
# Ensure spheres do not overlap by using twice the largest radius as spacing
size_of_object = 2 * np.max(bin_centers)
center_of_mass_coordinates = pairwise_method.generate_coordinates_for_polydisperse_system(
    size_of_object=size_of_object)

# ------------------------------------------------------------
# Place the spheres on the generated coordinates with the specified proportions
# ------------------------------------------------------------
all_coordinates = pairwise_method.place_objects_on_coordinates(
    center_of_mass_coordinates,
    objects,
    proportions)

# ------------------------------------------------------------
# Simulate the scattering curve for the polydisperse sphere system
# ------------------------------------------------------------
n_pairwise = 10000000
histogram_bins = 10000
q = np.geomspace(0.01, 1.0, 3000)

Iq = fitting.simulate_scattering(
    all_coordinates,
    q,
    histogram_bins,
    n_pairwise,
    mode='single')

# ------------------------------------------------------------
# Plot the results
# ------------------------------------------------------------
fitting.plot_intensity(q, Iq)
fitting.plot_structure(all_coordinates, 1000000)  # 3‑D point cloud of the polydisperse spheres