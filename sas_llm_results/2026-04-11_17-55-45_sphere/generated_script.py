import sys
sys.path.append('../')
from Scattering_Simulator import pairwise_method
from Scattering_Simulator import fitting
from Scattering_Simulator import Read_PDB
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

def generate_uniform_points_in_sphere(radius, N):
    """
    Generate N random points uniformly inside a sphere of given radius.
    """
    rng = np.random.default_rng(seed=42)
    u = rng.random(N)
    r = radius * np.cbrt(u)                       # radius^3 proportional to u
    phi = rng.uniform(0, 2*np.pi, N)
    costheta = rng.uniform(-1, 1, N)
    theta = np.arccos(costheta)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.column_stack((x, y, z))

# Parameters for the sphere
radius        = 50.0          # Å
SLD_value     = 1.0           # Contrast difference (arbitrary units)
num_points    = 20000         # Number of sampling points inside the sphere

# Generate coordinates with SLD
coords_3d      = generate_uniform_points_in_sphere(radius, num_points)
coordinates    = np.hstack((coords_3d, np.full((num_points, 1), SLD_value)))

# Scattering simulation settings
n_pairwise     = 10000000
histogram_bins = 10000
q                  = np.geomspace(0.01, 1.0, 3000)

# Perform the MC‐DFM scattering simulation
Iq = fitting.simulate_scattering(coordinates,
                                 q,
                                 histogram_bins,
                                 n_pairwise,
                                 mode='single')

# Plot results
fitting.plot_intensity(q, Iq)
fitting.plot_structure(coordinates, 1000000)