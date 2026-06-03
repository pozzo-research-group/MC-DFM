import sys
sys.path.append('../')
from Scattering_Simulator import pairwise_method
from Scattering_Simulator import fitting
from Scattering_Simulator import Read_PDB
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------
Diam          = 10.0          # Å (10 nm diameter)
radius        = Diam / 2.0     # sphere radius
points_per_sphere = 2000       # number of random points inside each sphere
SLD_val       = 1.0            # uniform SLD difference for all spheres

# ------------------------------------------------------------------
# Helper: generate random points uniformly inside a sphere
# ------------------------------------------------------------------
def random_points_in_sphere(center, radius, N, seed=None):
    rng = np.random.default_rng(seed)
    u   = rng.random(N)
    r   = radius * np.cbrt(u)
    phi = rng.uniform(0, 2*np.pi, N)
    costheta = rng.uniform(-1,1,N)
    theta = np.arccos(costheta)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    points = np.vstack((x,y,z)).T + center
    return points

# ------------------------------------------------------------------
# Assembly generators
# ------------------------------------------------------------------
def monomer():
    points = random_points_in_sphere(np.array([0,0,0]), radius, points_per_sphere)
    coords = np.hstack((points, np.full((points_per_sphere,1), SLD_val)))
    return coords

def doublet():
    centers = [np.array([0,0,0]), np.array([Diam,0,0])]
    all_coords = []
    for idx, c in enumerate(centers):
        pts = random_points_in_sphere(c, radius, points_per_sphere, seed=idx)
        all_coords.append(np.hstack((pts, np.full((points_per_sphere,1), SLD_val))))
    return np.vstack(all_coords)

def triplet():
    centers = [np.array([0,0,0]),
               np.array([Diam,0,0]),
               np.array([Diam/2, np.sqrt(3)/2*Diam, 0])]
    all_coords = []
    for idx, c in enumerate(centers):
        pts = random_points_in_sphere(c, radius, points_per_sphere, seed=idx)
        all_coords.append(np.hstack((pts, np.full((points_per_sphere,1), SLD_val))))
    return np.vstack(all_coords)

def tetrahedral():
    # Base triangle
    c1 = np.array([0,0,0])
    c2 = np.array([Diam,0,0])
    c3 = np.array([Diam/2, np.sqrt(3)/2*Diam, 0])
    # Centroid of triangle
    centroid = np.array([Diam/2, np.sqrt(3)/6*Diam, 0])
    z = np.sqrt(2/3) * Diam
    c4 = centroid + np.array([0,0,z])
    centers = [c1,c2,c3,c4]
    all_coords = []
    for idx, c in enumerate(centers):
        pts = random_points_in_sphere(c, radius, points_per_sphere, seed=idx)
        all_coords.append(np.hstack((pts, np.full((points_per_sphere,1), SLD_val))))
    return np.vstack(all_coords)

def dipyramidal():
    # Base triangle
    c1 = np.array([0,0,0])
    c2 = np.array([Diam,0,0])
    c3 = np.array([Diam/2, np.sqrt(3)/2*Diam, 0])
    # Centroid of triangle
    centroid = np.array([Diam/2, np.sqrt(3)/6*Diam, 0])
    z = np.sqrt(2/3) * Diam
    c4 = centroid + np.array([0,0,z])            # top center
    c5 = centroid + np.array([0,0,-z])           # bottom center
    centers = [c1,c2,c3,c4,c5]
    all_coords = []
    for idx, c in enumerate(centers):
        pts = random_points_in_sphere(c, radius, points_per_sphere, seed=idx)
        all_coords.append(np.hstack((pts, np.full((points_per_sphere,1), SLD_val))))
    return np.vstack(all_coords)

# ------------------------------------------------------------------
# Create the list of object coordinate arrays
# ------------------------------------------------------------------
objects = [monomer(), doublet(), triplet(), tetrahedral(), dipyramidal()]

# ------------------------------------------------------------------
# Proportions (equal for all five types)
# ------------------------------------------------------------------
proportions = [0.2] * len(objects)

# ------------------------------------------------------------------
# Generate center‑of‑mass coordinates for all assemblies
# ------------------------------------------------------------------
# Size of the smallest sphere that should separate objects; set larger than
# maximum assembly extent to avoid overlap
size_of_object = 60.0  # Å

center_of_mass_coordinates = pairwise_method.generate_coordinates_for_polydisperse_system(
    size_of_object=size_of_object)

# Limit the number of objects to a manageable number (e.g., 100 assemblies)
total_objects = 100
if center_of_mass_coordinates.shape[0] > total_objects:
    center_of_mass_coordinates = center_of_mass_coordinates[:total_objects,:]
else:
    # If fewer are generated, we regenerate until we have enough
    needed = total_objects - center_of_mass_coordinates.shape[0]
    extra = pairwise_method.generate_coordinates_for_polydisperse_system(
        size_of_object=size_of_object)
    center_of_mass_coordinates = np.vstack((center_of_mass_coordinates,
                                            extra[:needed,:]))

# ------------------------------------------------------------------
# Place each assembly at the generated coordinates
# ------------------------------------------------------------------
all_coordinates = pairwise_method.place_objects_on_coordinates(
    center_of_mass_coordinates,
    objects,
    proportions)

# ------------------------------------------------------------------
# Scattering simulation parameters
# ------------------------------------------------------------------
n_pairwise     = 10000000
histogram_bins  = 10000
q = np.geomspace(0.1, 1.0, 3000)

# ------------------------------------------------------------------
# Simulate scattering
# ------------------------------------------------------------------
Iq = fitting.simulate_scattering(
    all_coordinates,
    q,
    histogram_bins,
    n_pairwise,
    mode='single')

# ------------------------------------------------------------------
# Plot results
# ------------------------------------------------------------------
fitting.plot_intensity(q, Iq)
fitting.plot_structure(all_coordinates, 1000000)   # 1 million points for visualization