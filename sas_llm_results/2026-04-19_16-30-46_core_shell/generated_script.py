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
# Helper functions to generate random points inside a sphere or shell
# ------------------------------------------------------------
def generate_uniform_points_in_sphere(radius, N, seed=None):
    """
    Generate N random points uniformly inside a sphere of given radius.
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

# ------------------------------------------------------------
# Parameters for core–shell sphere
# ------------------------------------------------------------
core_radius   = 30.0   # Å
shell_thick   = 15.0   # Å
shell_radius  = core_radius + shell_thick  # 45 Å
SLD_core      = 5.0
SLD_shell     = 3.0

# ------------------------------------------------------------
# Determine number of points proportional to volumes
# ------------------------------------------------------------
V_core   = (4.0/3.0)*np.pi*core_radius**3
V_shell  = (4.0/3.0)*np.pi*(shell_radius**3 - core_radius**3)
V_total  = V_core + V_shell

N_total   = 20000   # total number of scattering points (adjustable)
N_core    = int(N_total * V_core / V_total)
N_shell   = N_total - N_core

# ------------------------------------------------------------
# Generate coordinates with appropriate SLD values
# ------------------------------------------------------------
np.random.seed(42)  # reproducibility

core_coords   = generate_uniform_points_in_sphere(core_radius, N_core, seed=42)
shell_coords  = generate_uniform_points_in_shell(core_radius, shell_radius, N_shell, seed=77)

# Add SLD difference as the 4th column
core_coords   = np.hstack((core_coords, np.full((N_core, 1), SLD_core)))
shell_coords  = np.hstack((shell_coords, np.full((N_shell, 1), SLD_shell)))

# Combine into single coordinate array
coordinates = np.vstack((core_coords, shell_coords))

# ------------------------------------------------------------
# SCATTERING SIMULATION
# ------------------------------------------------------------
n_pairwise    = 10000000
histogram_bins = 10000
q = np.geomspace(0.01, 1.0, 3000)

Iq = fitting.simulate_scattering(coordinates, q, histogram_bins, n_pairwise, mode='single')

# ------------------------------------------------------------
# PLOT RESULTS
# ------------------------------------------------------------
fitting.plot_intensity(q, Iq)
fitting.plot_structure(coordinates, 1000000)  # Show the 3-D point cloud of the core‑shell sphere