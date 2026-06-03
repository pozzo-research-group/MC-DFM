import sys
sys.path.append('../')
from Scattering_Simulator import pairwise_method
from Scattering_Simulator import fitting
from Scattering_Simulator import Read_PDB
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

# Load the protein coordinates from a PDB file
#coordinates = Read_PDB.load_pdb('../Data/PDB/9op9.pdb' )  # first 3 columns: x,y,z; 4th column: SLD difference

coordinates = Read_PDB.load_pdb('../Data/PDB/3vcd.pdb')  # first 3 columns: x,y,z; 4th column: SLD difference



# Define simulation parameters
n_pairwise = 10000000          # number of Monte‑Carlo pairwise samples
histogram_bins = 10000         # number of bins for the distance histogram
q = np.geomspace(0.0001, 1, 3000)  # q‑vector for the scattering curve

# Simulate the scattering curve for a single protein
Iq = fitting.simulate_scattering(coordinates, q, histogram_bins, n_pairwise, mode='multiple')

# Plot the intensity and the structure (visualisation)
fitting.plot_intensity(q, Iq)
fitting.plot_structure(coordinates, 1000000)  # plot the protein structure