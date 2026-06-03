# Imports required for MC‑DFM usage
import sys
sys.path.append('../')
from Scattering_Simulator import pairwise_method
from Scattering_Simulator import fitting
from Scattering_Simulator import Read_PDB
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------------
# Generate a uniformly random point set inside a right circular cylinder
# ----------------------------------------------------------------------
def generate_uniform_points_in_cylinder(radius, length, N, seed=None):
    """
    Generates N random points uniformly distributed inside a right circular
    cylinder aligned along the z‑axis. The cylinder extends from
    z = -length/2 to +length/2.

    Parameters
    ----------
    radius : float
        Cylinder radius in Å.
    length : float
        Cylinder length in Å.
    N : int
        Number of points to generate.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    points : ndarray of shape (N, 3)
        Cartesian coordinates of the points.
    """
    rng = np.random.default_rng(seed)
    # Uniform radial coordinate (via sqrt for proper area weighting)
    u = rng.random(N)
    r = radius * np.sqrt(u)
    # Uniform azimuthal angle
    phi = rng.uniform(0, 2*np.pi, N)
    # Convert to x,y
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    # Uniform z position within the cylinder
    z = rng.uniform(-length/2.0, length/2.0, N)
    return np.column_stack((x, y, z))

# ----------------------------------------------------------------------
# Specify rod geometry
# ----------------------------------------------------------------------
rod_radius   = 20.0      # Å
rod_length   = 200.0     # Å
SLD_rod      = 1.0       # SLD difference (arbitrary unit)

# ----------------------------------------------------------------------
# Generate scattering point coordinates for the rod
# ----------------------------------------------------------------------
N_points = 10000   # total number of Monte‑Carlo points
np.random.seed(42)  # reproducibility
rod_coords = generate_uniform_points_in_cylinder(rod_radius, rod_length, N_points, seed=42)

# Add the SLD difference as the fourth column
rod_coords = np.hstack((rod_coords, np.full((N_points, 1), SLD_rod)))

# ----------------------------------------------------------------------
# MC‑DFM scattering simulation
# ----------------------------------------------------------------------
n_pairwise    = 10000000
histogram_bins = 10000
q = np.geomspace(0.01, 1.0, 3000)   # q values in Å⁻¹

Iq = fitting.simulate_scattering(
    rod_coords,
    q,
    histogram_bins,
    n_pairwise,
    mode='single'
)

# ----------------------------------------------------------------------
# Plot the simulated intensity and the 3‑D point cloud of the rod
# ----------------------------------------------------------------------
fitting.plot_intensity(q, Iq)  # Plot I(q)

fitting.plot_structure(rod_coords, 1000000)  # Visualize the rod point cloud