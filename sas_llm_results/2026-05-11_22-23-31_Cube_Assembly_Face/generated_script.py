# ------------------------------------------------------------
# Imports required for the MC-DFM simulation
# ------------------------------------------------------------
import sys
sys.path.append('../')
from Scattering_Simulator import pairwise_method
from Scattering_Simulator import fitting
from Scattering_Simulator import Read_PDB
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------
# Helper function to generate random points uniformly inside a cube
# ------------------------------------------------------------
def generate_uniform_points_in_cube(edge_length, N, seed=None):
    """
    Generate N random points uniformly inside a cube of given edge length.
    The cube is centered at the origin.
    """
    rng = np.random.default_rng(seed)
    half = edge_length / 2.0
    # Uniform coordinates in [-half, +half]
    xs = rng.uniform(-half, half, N)
    ys = rng.uniform(-half, half, N)
    zs = rng.uniform(-half, half, N)
    return np.column_stack((xs, ys, zs))

# ------------------------------------------------------------
# Parameters for the cube system
# ------------------------------------------------------------
edge_length_nm = 10.0     # nm
edge_length = edge_length_nm * 10.0   # Å
N_points_cube = 20000     # number of sampling points per cube
SLD_cube = 1.0            # arbitrary SLD difference

# ------------------------------------------------------------
# Generate coordinates for a single cube centered at the origin
# ------------------------------------------------------------
np.random.seed(42)  # reproducibility
cube_coords = generate_uniform_points_in_cube(edge_length, N_points_cube, seed=42)
# Add SLD difference as the 4th column
cube_coords = np.hstack((cube_coords, np.full((N_points_cube, 1), SLD_cube)))

# ------------------------------------------------------------
# Create lattice coordinates for the linear line of 3 touching cubes
# ------------------------------------------------------------
# Translation distance is equal to the cube edge length
d = edge_length
lattice_coords = np.array([
    [0.0,   0.0, 0.0,   0.0, 0.0, 0.0],  # first cube at origin
    [d,     0.0, 0.0,   0.0, 0.0, 0.0],  # second cube touches first
    [2*d,   0.0, 0.0,   0.0, 0.0, 0.0]   # third cube touches second
])

# ------------------------------------------------------------
# Simulation parameters
# ------------------------------------------------------------
n_pairwise = 10000000
histogram_bins = 10000
q = np.geomspace(0.001, 0.3, 3000)

# ------------------------------------------------------------
# Simulate the scattering curve for the linear line of cubes
# ------------------------------------------------------------
Iq_cubes = fitting.simulate_scattering_lattice(
    cube_coords,
    lattice_coords,
    q,
    histogram_bins,
    n_pairwise,
    mode='single'
)

# ------------------------------------------------------------
# Plot the results
# ------------------------------------------------------------
fitting.plot_intensity(q, Iq_cubes)
fitting.plot_structure_lattice(cube_coords, lattice_coords, 1000000)
# ------------------------------------------------------------
# End of script
# ------------------------------------------------------------