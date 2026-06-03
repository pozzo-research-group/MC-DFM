# Import required libraries and modules
import sys
sys.path.append('../')
from Scattering_Simulator import pairwise_method
from Scattering_Simulator import fitting
from Scattering_Simulator import Read_PDB
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------------------
# Load the two protein structures from PDB files
# -------------------------------------------------------------
# Replace the following file paths with the correct paths on your system
cage_pdb_path   = '../Data/PDB/9op9.pdb'   # larger protein cage
small_pdb_path  = '../Data/PDB/3vcd.pdb'  # smaller protein to place inside cage

# Load coordinates; the last column contains the SLD contrast
coordinates_cage   = Read_PDB.load_pdb(cage_pdb_path)
coordinates_small  = Read_PDB.load_pdb(small_pdb_path)

# -------------------------------------------------------------
# Set the desired SLD contrast values
# -------------------------------------------------------------
coordinates_cage[:, -1] = 1.0   # larger cage SLD contrast
coordinates_small[:, -1] = 5.0  # smaller protein SLD contrast

# -------------------------------------------------------------
# Translate the smaller protein so that its center aligns with the center of the cage
# -------------------------------------------------------------
# Compute centers of the two objects
center_cage  = np.mean(coordinates_cage[:, :3], axis=0)
center_small = np.mean(coordinates_small[:, :3], axis=0)

# Move smaller protein to origin
coordinates_small_centered = coordinates_small.copy()
coordinates_small_centered[:, :3] -= center_small

# Translate smaller protein to cage center
coordinates_small_centered[:, :3] += center_cage

# -------------------------------------------------------------
# Combine the two structures into a single coordinate array
# -------------------------------------------------------------
coordinates_combined = np.vstack((coordinates_cage, 
                                  coordinates_small_centered))
# The combined array keeps the SLD column for each atom.

# -------------------------------------------------------------
# Create a 3x3 2D lattice of the combined assembly
# -------------------------------------------------------------
# Determine bounding box of the cage (to set translation spacing)
x_min, x_max = np.min(coordinates_cage[:, 0]), np.max(coordinates_cage[:, 0])
y_min, y_max = np.min(coordinates_cage[:, 1]), np.max(coordinates_cage[:, 1])

# Calculate spacing such that cages do not overlap
# Add a small buffer (e.g., 5 Å) to ensure no overlap
spacing_x = (x_max - x_min) + 5.0
spacing_y = (y_max - y_min) + 5.0

# Generate lattice coordinate rows: [tx, ty, tz, rot_x, rot_y, rot_z]
lattice_list = []
for i in range(3):
    for j in range(3):
        tx = i * spacing_x
        ty = j * spacing_y
        tz = 0.0
        rot_x, rot_y, rot_z = 0.0, 0.0, 0.0   # no rotation for the cages
        lattice_list.append([tx, ty, tz, rot_x, rot_y, rot_z])

lattice_coordinates = np.array(lattice_list)  # shape (9,6)

# -------------------------------------------------------------
# Define q grid and simulation parameters
# -------------------------------------------------------------
q = np.geomspace(0.0001, 1.0, 3000)           # momentum transfer range [1/Å]
n_pairwise       = 10000000
histogram_bins   = 10000

# -------------------------------------------------------------
# Simulate the scattering intensity for the 3x3 lattice
# -------------------------------------------------------------
Iq_lattice = fitting.simulate_scattering_lattice(
    coordinates_combined,
    lattice_coordinates,
    q,
    histogram_bins,
    n_pairwise,
    mode='single'   # use single-molecule mode for each lattice point
)

# -------------------------------------------------------------
# Plot the simulated intensity and the structure of the lattice
# -------------------------------------------------------------
fitting.plot_intensity(q, Iq_lattice)                       # intensity vs q
fitting.plot_structure_lattice(coordinates_combined,       # plot the lattice structure
                                lattice_coordinates,
                                1000000)                 # number of points for visualization