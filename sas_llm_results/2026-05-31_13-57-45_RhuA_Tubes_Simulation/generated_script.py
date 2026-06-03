# Import required packages
import sys
sys.path.append('../')
from Scattering_Simulator import pairwise_method
from Scattering_Simulator import fitting
from Scattering_Simulator import Read_PDB
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
import math


# ------------------------------------------------------------------
# 1. Load the atomic coordinates of the rectangular‑prism protein
# ------------------------------------------------------------------
# The PDB file should contain a single rectangular prism shaped protein
# of dimensions 7 × 7 × 5 nm (i.e. 70 × 70 × 50 Å).  The file must
# pack a fourth column with the SLD difference of each atom.
# Replace the path with the actual location of your PDB file.
coordinates = Read_PDB.load_pdb('../Data/PDB/RhuA/1gt7.pdb')
def relative_coordinates(volume):
    '''This function is used to center any set of coordinates at the origin'''
    rel_x = volume[:,0] - np.mean(volume[:,0])
    rel_y = volume[:,1] - np.mean(volume[:,1])
    rel_z = volume[:,2] - np.mean(volume[:,2])
    relative_volume = np.hstack((rel_x.reshape(-1,1), rel_y.reshape(-1,1), rel_z.reshape(-1,1), volume[:,-1].reshape(-1,1)))
    return relative_volume

def rotation_angle(x,z):
    '''This function converts cartesian coordinates to polar'''
    angle = -np.arctan2(z, x)*180/math.pi
    return np.round(angle)


def rotate_coordinates_y(x, z, angle):
    '''This function performs a rigid body rotation around the y-axis'''
    angle = angle*math.pi/180
    x_new = x*np.cos(angle) + z*np.sin(angle)
    z_new = -x*np.sin(angle) + z*np.cos(angle)
    return x_new, z_new

def rotate_coordinates_x(y, z, angle):
    '''This function performs a rigid body rotation around the x-axis'''
    angle = angle*math.pi/180
    y_new = y*np.cos(angle) + z*np.sin(angle)
    z_new = -y*np.sin(angle) + z*np.cos(angle)
    return y_new, z_new

def rotate_coordinates_z(x, y, angle):
    '''This function performs a rigid body rotation around the z-axis'''
    angle = angle*math.pi/180
    x_new = x*np.cos(angle) + y*np.sin(angle)
    y_new = -x*np.sin(angle) + y*np.cos(angle)
    return x_new, y_new

#coordinates = Read_PDB.load_pdb('../Data/PDB/RhuA/1ojr.pdb1') #this function loads the pdb file
coordinates = Read_PDB.load_pdb('../Data/PDB/RhuA/1gt7.pdb')
coordinates = relative_coordinates(coordinates)
volume = coordinates.copy()
#This rotates the protein around the specifed axis
coordinates[:,1], coordinates[:,2] = rotate_coordinates_x(coordinates[:,1], coordinates[:,2], 180)
coordinates = np.hstack((coordinates[:,0].reshape(-1,1), coordinates[:,2].reshape(-1,1), coordinates[:,1].reshape(-1,1), coordinates[:,-1].reshape(-1,1)))
coordinates[:,0], coordinates[:,1] = rotate_coordinates_z(coordinates[:,0], coordinates[:,1], 0)
volume_rotated = np.hstack((coordinates[:,0].reshape(-1,1), coordinates[:,2].reshape(-1,1), coordinates[:,1].reshape(-1,1), coordinates[:,-1].reshape(-1,1)))
coordinates[:,0], coordinates[:,2] = rotate_coordinates_y(coordinates[:,0], coordinates[:,2], 30 + 180)
volume_rotated = np.hstack((coordinates[:,0].reshape(-1,1), coordinates[:,2].reshape(-1,1), coordinates[:,1].reshape(-1,1), coordinates[:,-1].reshape(-1,1)))
coordinates = volume_rotated


# fig = plt.figure(figsize=(6,6))
# ax = plt.axes(projection='3d')
# ax.scatter(volume_rotated[:,0],volume_rotated[:,1],volume_rotated[:,2], color = 'blue', alpha = 0.1, s = 30)
# ax.set_xlabel('x [$\AA$]')
# ax.set_ylabel('y [$\AA$]')
# ax.set_zlabel('z [$\AA$]')
# ax.set_title('Top Conformation')
building_block = coordinates



# ----- Parameters ---------------------------------------------------------

# Protein sub‑unit dimensions (in Å) – 7 × 7 × 5 nm
SUBUNIT_W = 70.0   # width of the 7 nm face (x‑direction)
SUBUNIT_H = 50.0   # height of the 5 nm face (z‑direction)
SUBUNIT_T = 70.0   # third dimension (y‑direction), not used for geometry

# Tube construction parameters
N_ROTATIONS  = 40                     # number of full helical turns
AXIAL_GAP    = SUBUNIT_H              # axial separation between sub‑units (Å)
CIRCUMF = [13, 15, 17, 19, 21, 23, 25, 27, 29]   # sub‑units per turn

# Simulation setting
N_PAIRWISE    = 10000000   # Keep as required
HISTOGRAM_BIN = 10000
Q = np.geomspace(0.001, 0.2, 3000)   # Q‑vector in Å⁻¹


# -------------------------------------------------------------------------

def build_lattice_coords(N_per_circ, rotations=40,
                         subunit_width=SUBUNIT_W,
                         axial_gap=AXIAL_GAP):
    """
    Build lattice coordinates for a helical tube.
    Each row of the returned array is [dx, dy, dz, rot_x, rot_y, rot_z].
    The translation dx,dy,dz gives the centre of a sub‑unit in Å,
    and rot_y is the rotation applied about the y‑axis (degrees).
    """
    # Radius so that the 7 nm width fits around the circumference
    r = (subunit_width * N_per_circ) / (2.0 * np.pi)

    lattice = []

    for idx in range(N_per_circ * rotations):
        # Position in cylindrical coordinates
        ring_i       = idx % N_per_circ
        roll_i       = idx // N_per_circ
        theta = 2.0 * np.pi * ring_i / N_per_circ
        x = r * np.cos(theta)
        z = r * np.sin(theta)
        y = roll_i * axial_gap

        # Base rotation that keeps the 7x5 face tangent to the tube
        base_rot = -np.arctan2(z, x) * 180.0 / np.pi + 90.0
        # Alternate 180° for successive sub‑units to create up‑down pattern
        if (ring_i % 2) == 1:
            base_rot += 180.0        # flip around the y‑axis

        # Wrap angle to [0, 360)
        base_rot = base_rot % 360.0

        lattice.append([x, y, z, 0.0, base_rot, 0.0])

    return np.array(lattice)


def simulate_tube(N_per_circ, subunit_coords,
                  rotations=40, prop=1.0/len(CIRCUMF)):
    """
    Simulate scattering from a single tube with the given
    number of sub‑units per turn.
    Returns the scaled intensity for use in a mixture.
    """
    lattice_coords = build_lattice_coords(N_per_circ, rotations)
    # Simulate the tube (mode='single' for a single assembly)
    Iq = fitting.simulate_scattering_lattice(subunit_coords,
                                              lattice_coords,
                                              Q,
                                              HISTOGRAM_BIN,
                                              N_PAIRWISE,
                                              mode='multiple')
    # Number of scattering sub‑units in this tube
    N_subunits = N_per_circ * rotations
    # Invariant
    Inv = fitting.invariant(np.column_stack((Q, Iq)))
    # Scale intensity by invariant, sub‑unit count and mixture proportion
    Iq_scaled = fitting.scale_intensity(Iq, Inv, N_subunits, prop)
    return Iq_scaled


# Load the protein sub‑unit coordinates once
subunit_coords = coordinates

# -------------------------------------------------------------------------
# Simulate each tube type, plot, and save
iq_dict = {}
for N_circ in CIRCUMF:
    # Prop is 1/number_of_tubes for an equal mixture
    prop = 1.0 / len(CIRCUMF)
    Iq_scaled = simulate_tube(N_circ, subunit_coords,
                              rotations=N_ROTATIONS,
                              prop=prop)

    iq_dict[N_circ] = Iq_scaled

    # Plot the individual tube scattering curve
    fitting.plot_intensity(Q, Iq_scaled)

    # Save scattering intensity to a file for later use
    filename = f'scattering_intensity_{N_circ}.txt'
    np.savetxt(filename, np.column_stack((Q, Iq_scaled)),
               header='Q(Å⁻¹)  I(q)')

# -------------------------------------------------------------------------
# Construct the total mixture (equal proportions of all tube types)
Iq_mixture = sum(iq_dict.values())

# Plot the mixture
fitting.plot_intensity(Q, Iq_mixture)

# Save the mixture intensity
np.savetxt('scattering_intensity_mixture.txt',
           np.column_stack((Q, Iq_mixture)),
           header='Q(Å⁻¹)  I(q)')

# Optionally display plots
plt.legend()
plt.show()