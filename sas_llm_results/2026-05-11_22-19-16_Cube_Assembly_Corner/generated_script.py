# ------------------------------------------------------------
# Imports required for the MC-DFM simulation
# ------------------------------------------------------------
import sys                       # #import sys
sys.path.append('../')           # #sys.path.append('../')   # required for relative imports
from Scattering_Simulator import pairwise_method
from Scattering_Simulator import fitting
from Scattering_Simulator import Read_PDB
from mpl_toolkits.mplot3d import axes3d   # #from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt           # #import matplotlib.pyplot as plt
import numpy as np                        # #import numpy as np

# ------------------------------------------------------------
# Helper function to generate random points inside a cube
# ------------------------------------------------------------
def generate_uniform_points_in_cube(side_length, N, seed=None):
    """
    Generates N random points uniformly distributed inside a cube
    of given side_length, centered at the origin.
    """
    rng = np.random.default_rng(seed)
    # Coordinates in range [-L/2, L/2] for each axis
    points = rng.uniform(-side_length/2, side_length/2, size=(N, 3))
    return points

# ------------------------------------------------------------
# Parameters for the simulation
# ------------------------------------------------------------
cube_edge = 100.0          # Å (10 nm)
points_per_cube = 5000     # number of sampling points inside each cube
SLD_cube = 1.0             # arbitrary (can be changed if needed)

# Create the coordinates for a single cube centered at the origin
cube_points = generate_uniform_points_in_cube(cube_edge, points_per_cube, seed=42)
# Add SLD as the 4th column
cube_coords = np.hstack((cube_points, np.full((points_per_cube, 1), SLD_cube)))

# ------------------------------------------------------------
# Define the lattice coordinates for three cubes lined up corner‑to‑corner
# The translation vector between cubes is (L, L, L) where L = cube_edge
lattice_coords = np.array([[0, 0, 0, 0, 0, 0],
                           [cube_edge, cube_edge, cube_edge, 0, 0, 0],
                           [2*cube_edge, 2*cube_edge, 2*cube_edge, 0, 0, 0]])

# ------------------------------------------------------------
# Scattering simulation
# ------------------------------------------------------------
n_pairwise = 10000000
histogram_bins = 10000
q = np.geomspace(0.001, 0.3, 3000)

# Simulate scattering using the lattice (three copies of the cube)
Iq_cubes = fitting.simulate_scattering_lattice(
    cube_coords,
    lattice_coords,
    q,
    histogram_bins,
    n_pairwise,
    mode='single'
)

# ------------------------------------------------------------
# Plot results
# ------------------------------------------------------------
fitting.plot_intensity(q, Iq_cubes)                            # Plot scattering curve
fitting.plot_structure_lattice(cube_coords, lattice_coords, 1000000)  # Visualize the three‑cube assembly