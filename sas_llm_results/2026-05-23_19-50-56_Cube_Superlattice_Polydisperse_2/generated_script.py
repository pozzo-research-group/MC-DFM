# Setup import paths and modules -------------------------------------------------
import sys
sys.path.append('../')
from Scattering_Simulator import pairwise_method
from Scattering_Simulator import fitting
from Scattering_Simulator import Read_PDB
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------------------------------------
# 1. Generate coordinates for a single 10 nm cube --------------------------------
#    Edge length is 10 nm = 100 Å.  Points are sampled on a 2 Å grid to keep the
#    point count reasonable (~132 k points).  The fourth column is the SLD
#    difference; we use 1 for simplicity.
# -------------------------------------------------------------------------------
def generate_cube_coords(edge_length_angstrom, grid_step=2.0, sld_diff=1.0):
    """Return an (N,4) array of coordinates for a cube centered at the origin."""
    half = edge_length_angstrom/2.0
    coords_1d = np.arange(-half, half+grid_step/2.0, grid_step)
    X, Y, Z = np.meshgrid(coords_1d, coords_1d, coords_1d, indexing='ij')
    points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    sld_col = np.full((points.shape[0], 1), sld_diff)
    return np.hstack((points, sld_col))

cube_edge = 100.0                      # 10 nm = 100 Å
cube_coords = generate_cube_coords(cube_edge)  # (N,4) array

# -------------------------------------------------------------------------------
# 2. Helper function to create lattice coordinates -----------------------------
#    For a superlattice of size (nx,ny,nz).  Centers are at (i*spacing, j*spacing, k*spacing)
#    with spacing 20 nm = 200 Å.  No rotations are needed for cubes.
# -------------------------------------------------------------------------------
def create_lattice_coords(nx, ny, nz, spacing_angstrom=200.0):
    """Return an (M,6) array of lattice coordinates for a superlattice."""
    xs = np.arange(nx) * spacing_angstrom
    ys = np.arange(ny) * spacing_angstrom
    zs = np.arange(nz) * spacing_angstrom
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    transl = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    # zero rotations for all cubes
    rotations = np.zeros_like(transl)
    return np.hstack((transl, rotations))

# -------------------------------------------------------------------------------
# 3. Simulation parameters -----------------------------------------------------
# -------------------------------------------------------------------------------
n_pairwise     = 10000000   # keep as required
histogram_bins = 10000
q = np.geomspace(0.0001, 0.2, 3000)  # q‑vector in inverse Å⁻¹

# -------------------------------------------------------------------------------
# 4. Define lattice sizes to mix -----------------------------------------------
lattice_sizes = {
    '2x2x2': (2, 2, 2),
    '2x5x2': (2, 5, 2),
    '3x4x2': (3, 4, 2),
    '3x3x3': (3, 3, 3),
    '10x2x1': (10, 2, 1)
}
prop = 1.0 / len(lattice_sizes)   # equal mixture

# -------------------------------------------------------------------------------
# 5. Loop over lattices, simulate, scale, and save each curve ------------------
scaled_curves = {}
for name, dims in lattice_sizes.items():
    # Create lattice coordinates
    lattice_coords = create_lattice_coords(*dims)
    # Number of cubes in superlattice
    N_subunits = lattice_coords.shape[0]
    # Simulate scattering for this superlattice
    Iq = fitting.simulate_scattering_lattice(
        cube_coords, lattice_coords, q, histogram_bins, n_pairwise, mode='multiple'
    )
    # Compute invariant
    Inv = fitting.invariant(np.column_stack((q, Iq)))
    # Scale the intensity: by number of cubes * mixture proportion
    Iq_scaled = fitting.scale_intensity(Iq, Inv, N_subunits, prop)
    # Save each scaled curve to a file for later inspection
    filename = f'scaled_curve_{name}.txt'
    np.savetxt(filename, np.column_stack((q, Iq_scaled)), header='q Iq')
    # Store in dictionary for summation
    scaled_curves[name] = Iq_scaled
    print(f'Processed {name}, saved to {filename}')

# -------------------------------------------------------------------------------
# 6. Compute weighted sum (equal weights) ------------------------------------
Iq_total = sum(scaled_curves.values())

# -------------------------------------------------------------------------------
# 7. Plot total scattering curve -----------------------------------------------
fitting.plot_intensity(q, Iq_total + 1e3)

# -------------------------------------------------------------------------------
# 8. (Optional) Save the total curve
np.savetxt('scaled_curve_total.txt', np.column_stack((q, Iq_total)), header='q Iq')