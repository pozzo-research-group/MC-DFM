import sys
sys.path.append('../')
from Scattering_Simulator import pairwise_method
from Scattering_Simulator import fitting
from Scattering_Simulator import Read_PDB
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

# Load the protein coordinates from the PDB files
# The files should contain the coordinates and default SLD values in column 4
coordinates_large = Read_PDB.load_pdb('../Data/PDB/9op9.pdb')
coordinates_small = Read_PDB.load_pdb('../Data/PDB/3vcd.pdb')

# ---- Set the desired SLD contrasts ----
# Larger protein cage has SLD contrast = 1
coordinates_large[:, 3] = 1.0
# Smaller protein inside the cage has SLD contrast = 5
coordinates_small[:, 3] = 5.0

# ---- Translate the smaller protein so that its center aligns with the large cage's center ----
center_large = np.mean(coordinates_large[:, 0:3], axis=0)
center_small = np.mean(coordinates_small[:, 0:3], axis=0)
shift_vector = center_large - center_small
coordinates_small_shifted = coordinates_small.copy()
coordinates_small_shifted[:, 0:3] += shift_vector

# ---- Combine the two coordinate sets into a single object ----
coordinates_combined = np.vstack((coordinates_large, coordinates_small_shifted))

# ---- Parameters for the simulation ----
n_pairwise = 10000000           # Always set to 10 million for good statistics
histogram_bins = 10000
q = np.geomspace(0.0001, 1.0, 3000)  # q-range in Å⁻¹

# ---- Simulate the scattering intensity ----
Iq = fitting.simulate_scattering(
        coordinates_combined,
        q,
        histogram_bins,
        n_pairwise,
        mode='multiple'
     )

# ---- Plot the resulting intensity profile ----
fitting.plot_intensity(q, Iq)

# ---- Visualize the 3D structure of the combined assembly ----
fitting.plot_structure(coordinates_combined, 1000000)   # The second argument is the number of points to plot