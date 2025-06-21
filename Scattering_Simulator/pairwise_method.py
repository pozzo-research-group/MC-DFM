import torch
import numpy as np 
from scipy import integrate
import matplotlib.pyplot as plt

class scattering_simulator:
    def __init__(self, n_samples, device=None):
        self.n_samples = n_samples
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def to_tensor(self, array):
        """Converts a NumPy array or a PyTorch tensor to a tensor on the specified device."""
        if isinstance(array, torch.Tensor):
            # If already a tensor, move it to the correct device and dtype
            return array.to(device=self.device, dtype=torch.float32)
        elif isinstance(array, np.ndarray):
            # Convert NumPy array to tensor without copying if possible
            return torch.from_numpy(array).to(device=self.device, dtype=torch.float32)
        else:
            # Handle other types explicitly to prevent unexpected behavior
            raise TypeError("Input must be a NumPy array or a PyTorch tensor.")


    def sample_building_block(self, building_block_coordinates):
        '''Randomly samples "n_samples" number of points from  "building_block_coordinates".
        
        inputs: 
        - building_block_coordinates: array with x, y, z positions of the building block coordinates. The 
        array should have 3 columns and any number of rows.
        
        results:
        - self.building_block_coordinates_1: an array with randomly sampled coordinates from the building block
        - self.building_block_coordinates_2: an array with randomly sampled coordinates from the building block.'''
        building_block_coordinates = self.relative_coordinates(building_block_coordinates)
        rand_num1 = torch.randint(0, building_block_coordinates.shape[0], (self.n_samples,), device=self.device)
        rand_num2 = torch.randint(0, building_block_coordinates.shape[0], (self.n_samples,), device=self.device)
        self.building_block_coordinates_1 = building_block_coordinates[rand_num1, :]
        self.building_block_coordinates_2 = building_block_coordinates[rand_num2, :]


    def sample_lattice_coordinates(self, lattice_coordinates, save=False):
        '''Randomly samples ''n_samples'' number of points from the lattice.
        
        inputs:
        - lattice_coordinates: 3D coordinates of the lattice.
        
        results:
        - self.lattice_coordinates_1: an array with randomly sampled coordinates from the lattice
        - self.lattice_coordinates_2: an array with randomly sampled coordinates from the lattice.'''
        lattice_coordinates = self.to_tensor(lattice_coordinates)
        rand_num = torch.randint(0, lattice_coordinates.shape[0], (self.n_samples,), device=self.device)
        self.lattice_coordinates_1 = lattice_coordinates[rand_num, :]
        rand_num = torch.randint(0, lattice_coordinates.shape[0], (self.n_samples,), device=self.device)
        self.lattice_coordinates_2 = lattice_coordinates[rand_num, :]


    def calculate_structure_coordinates(self, save=False):
        '''Adds the building block coordinates to the lattice coordinates to obtain the structure coordinates.
        
        inputs (automatically loaded):
        - self.building_block_coordinates
        - self.lattice_coordinates 
        
        results:
        - self.structure_coordinates.'''
        if self.building_block_coordinates_1.shape[1] == 3:
            self.structure_coordinates_1 = self.building_block_coordinates_1 + self.lattice_coordinates_1
            self.structure_coordinates_2 = self.building_block_coordinates_2 + self.lattice_coordinates_2
        else:
            if self.lattice_coordinates_1.shape[1] == 3:
                self.structure_coordinates_1 = self.building_block_coordinates_1[:, :-1] + self.lattice_coordinates_1
                self.structure_coordinates_2 = self.building_block_coordinates_2[:, :-1] + self.lattice_coordinates_2
                self.structure_coordinates_1 = torch.cat((self.structure_coordinates_1, self.building_block_coordinates_1[:, -1:].reshape(-1, 1)), dim=1)
                self.structure_coordinates_2 = torch.cat((self.structure_coordinates_2, self.building_block_coordinates_2[:, -1:].reshape(-1, 1)), dim=1)
            else:
                self.structure_coordinates_1 = self.rotate_building_block(self.building_block_coordinates_1[:, :-1], self.lattice_coordinates_1)
                self.structure_coordinates_1 = torch.cat((self.structure_coordinates_1, self.building_block_coordinates_1[:, -1:].reshape(-1, 1)), dim=1)
                self.structure_coordinates_2 = self.rotate_building_block(self.building_block_coordinates_2[:, :-1], self.lattice_coordinates_2)
                self.structure_coordinates_2 = torch.cat((self.structure_coordinates_2, self.building_block_coordinates_2[:, -1:].reshape(-1, 1)), dim=1)

        if not save:
            self.lattice_coordinates_1 = None
            self.lattice_coordinates_2 = None


    def simulate_scattering_curve_fast_lattice(self, coordinates, lattice, bins, q, save=False):
        '''Runs the simulation to calculate the scattering curve in a loop with a reduced number of n_samples.
           It is slightly faster than the normal method. 
           This function is used when there is a building block and lattice coordinates. 

        - coordinates: coordinates of the building block structure
        - lattice: the coordinates of the lattice where each building block is placed 
        - bins: number of bins used to create the histogram 
        - q: the momentum transfer vector (q) 
        outputs:
        - self.I_q: the scattering intensity curve as a function of q.'''
        original_samples = self.n_samples
        self.n_samples = int(torch.round(torch.tensor(self.n_samples / 5, device=self.device)))
        all_p_r = None

        for _ in range(5):
            self.sample_building_block(coordinates)
            self.sample_lattice_coordinates(lattice)
            self.calculate_structure_coordinates()
            self.distance_function(save=False)
            self.create_histogram(bins)

            if all_p_r is None:
                all_p_r = ray.get(self.p_r.clone())
            else:
                all_p_r += ray.get(self.p_r)

        self.p_r = all_p_r
        self.convert_to_intensity(q)
        self.n_samples = original_samples  # Restore original number of samples
        return self.I_q
    
    def use_building_block_as_structure(self, save=False):
        '''Uses the building block as the structure.
        
        inputs (automatically loaded):
        - self.building_block_coordinates
        - self.lattice_coordinates 
        
        results:
        - self.structure_coordinates.'''
        self.structure_coordinates_1 = self.building_block_coordinates_1
        self.structure_coordinates_2 = self.building_block_coordinates_2

        if not save:
            self.lattice_coordinates_1 = None
            self.lattice_coordinates_2 = None

    def relative_coordinates(self, building_block_coordinates):
        '''Used to center the building block coordinates to have a center at coordinates (0,0,0).
        
        inputs:
        - building_block_coordinates: an array of the x-y-z coordinates of the building block. The array
        should have 3 columns and any number of rows.
        
        outputs:
        - building_block_coordinates_centered: an array of the x-y-z coordinates of the building block with a center
        at point (0,0,0).'''
        building_block_coordinates = self.to_tensor(building_block_coordinates)
        rel_coords = building_block_coordinates - building_block_coordinates.mean(dim=0, keepdim=True)
        return rel_coords


    def distance_function(self, save):
        '''Calculates the pairwise Euclidean distance between the rows of two different arrays.
        
        inputs (automatically loaded):
        - self.structure_coordinates_1: The randomly sampled x-y-z coordinates of the structure 
        - self.structure_coordinates_2: The randomly sampled x-y-z coordinates of the structure 
        
        results:
        - self.distances: a 1D array of the pairwise distances of the two randomly sampled arrays of the structure coordinates.'''
        p1 = self.structure_coordinates_1
        p2 = self.structure_coordinates_2
        self.distances = torch.sqrt(((p1 - p2)**2).sum(dim=1))


    def create_histogram(self, bins):
        '''Creates a histogram of the pairwise distances between two randomly selected points from the structure coordinates.
        
        inputs:
        - bins: number of bins for the histogram
        - self.distances: a 1D array of the pairwise distances of the two randomly sampled arrays of the structure coordinates
        
        results:
        - self.p_r: the pairwise distribution function  
        - self.r: the pairwise distances of the two randomly sampled coordinates of the structure.'''
        hist = torch.histc(self.distances, bins=bins, min=0.0, max=self.distances.max())
        self.p_r = hist
        self.r = torch.linspace(0, self.distances.max(), steps=bins, device=self.device)


    def convert_to_intensity(self, q):
        '''Converts the pairwise distribution function into the scattering intensity as a function of q. 
        
        inputs:
        - q: the momentum transfer vector (q) 
        - self.p_r: the pairwise distribution function  
        - self.r: the pairwise distances of the two randomly sampled coordinates of the structure
        
        results:
        - self.I_q: the scattering intensity curve as a function of q.'''
        I_q = []
        for q_val in q:
            integrand = 4 * torch.pi * self.p_r * torch.sin(q_val * self.r) / (q_val * self.r)
            integrand[torch.isnan(integrand)] = 0
            I_q.append(torch.trapz(integrand, self.r))
        self.I_q = torch.tensor(I_q, device=self.device)

    def convert_to_pairwise(self, r, I, q):
        '''Converts the scattering intensity back to the pairwise distribution function.
        
        inputs: 
        - r: the pairwise distances of the two randomly sampled coordinates of the structure
        - I: the scattering intensity 
        - q: the momentum transfer vector
        
        outputs:
        - p_r: the pairwise distribution which is a function of (r).'''
        r, I, q = map(self.to_tensor, (r, I, q))
        p_r = []
        for r_val in r:
            integrand = I * q * r_val * torch.sin(q * r_val)
            integrand[torch.isnan(integrand)] = 0
            p_r.append(torch.trapz(integrand, q))
        return torch.tensor(p_r, device=self.device)

    def simulate_scattering_curve(self, bins, q, save=False):
        '''Function to run the calculation of the coordinates of the structure to the scattering intensity curve.
        
        inputs:
        - bins: number of bins used to create the histogram 
        - q: the momentum transfer vector (q) 
        
        outputs:
        - self.I_q: the scattering intensity curve as a function of q.'''
        self.distance_function(save=save)
        self.create_histogram(bins)
        self.convert_to_intensity(q)
        return self.I_q


    def simulate_scattering_curve_fast(self, coordinates, bins, q, save=False):
        '''Runs the simulation to calculate the scattering curve in a loop with a reduced number of n_samples:
            It is slightly faster than the normal method of calculating the scattering curve. 
            Only works with building block setting.
        
        inputs:
        - coordinates: coordinates of the building block structure
        - bins: number of bins used to create the histogram 
        - q: the momentum transfer vector (q) 
        
        outputs:
        - self.I_q: the scattering intensity curve as a function of q.'''

        reduced_samples = int(torch.round(torch.tensor(self.n_samples / 5, device=self.device)))
        all_p_r = None

        for _ in range(5):
            self.n_samples = reduced_samples
            self.sample_building_block(coordinates)
            self.use_building_block_as_structure()
            self.distance_function(save=False)
            self.create_histogram(bins)
            if all_p_r is None:
                all_p_r = self.p_r.clone()
            else:
                all_p_r += self.p_r

        self.p_r = all_p_r
        self.convert_to_intensity(q)
        return self.I_q
    
    # def simulate_scattering_curve_fast_lattice(self, coordinates, lattice, bins, q, save=False):
    #     """
    #     Parallelized version using Ray for calculating scattering curve.
    #     """
    #     original_samples = self.n_samples
    #     self.n_samples = int(torch.round(torch.tensor(self.n_samples / 5, device=self.device)))

    #     all_p_r = None
    #     tasks = []  # List to store Ray tasks for parallel execution

    #     for _ in range(5):
    #         # Schedule remote tasks for each step
    #         task1 = self.sample_building_block(coordinates)
    #         task2 = self.sample_lattice_coordinates(lattice)
    #         task3 = self.calculate_structure_coordinates()
    #         task4 = self.distance_function(False)
    #         task5 = self.create_histogram(bins)

    #         # Collect results asynchronously using Ray's tasks
    #         tasks.append((task1, task2, task3, task4, task5))

    #     # Process results after tasks complete
    #     for task_group in tasks:
    #         results = task_group
    #         # For `all_p_r`, accumulate the p(r) results
    #         if all_p_r is None:
    #             all_p_r = results[4]  # Assuming `p_r` comes from `create_histogram_remote`
    #         else:
    #             all_p_r += results[4]

    #     # Update instance variables
    #     self.p_r = all_p_r
    #     self.convert_to_intensity(q)
    #     self.n_samples = original_samples  # Restore original number of samples

    #     return self.I_q
    

    def simulate_scattering_curve_fast_lattice(self, coordinates, lattice, bins, q, save=False):
        '''Runs the simulation to calculate the scattering curve in a loop with a reduced number of n_samples.
           It is slightly faster than the normal method. 
           This function is used when there is a building block and lattice coordinates. 

        - coordinates: coordinates of the building block structure
        - lattice: the coordinates of the lattice where each building block is placed 
        - bins: number of bins used to create the histogram 
        - q: the momentum transfer vector (q) 
        outputs:
        - self.I_q: the scattering intensity curve as a function of q.'''

        original_samples = self.n_samples
        self.n_samples = int(torch.round(torch.tensor(self.n_samples / 5, device=self.device)))
        all_p_r = None

        for _ in range(5):
            self.sample_building_block(coordinates)
            self.sample_lattice_coordinates(lattice)
            self.calculate_structure_coordinates()
            self.distance_function(save=False)
            self.create_histogram(bins)

            if all_p_r is None:
                all_p_r = self.p_r.clone()
            else:
                all_p_r += self.p_r

        self.p_r = all_p_r
        self.convert_to_intensity(q)
        self.n_samples = original_samples  # Restore original number of samples
        return self.I_q




    def simulate_multiple_scattering_curves(self, coordinates, bins, q, save=False):
        '''Function to obtain multiple scattering curves of the same structure using different values of n_samples.
        This results in a more accurate scattering curve. 
        This function only works for the building block setting. 
        
        inputs:
        - coordinates: the coordinates of the building block 
        - bins: number of bins used to create the histogram 
        - q: the momentum transfer vector (q) 
        outputs:
        - Intensities: the scattering intensity curves as a function of q.'''
        n_samples_array = torch.linspace(0.5, 1.5, 10, device=self.device) * self.n_samples
        Intensities = []

        for n_samples in n_samples_array:
            self.n_samples = int(torch.round(n_samples))
            self.sample_building_block(coordinates)
            self.use_building_block_as_structure()
            self.distance_function(save=save)
            self.create_histogram(bins)
            self.convert_to_intensity(q)
            self.I_q = self.I_q / self.I_q[0]
            Intensities.append(self.I_q.unsqueeze(1))

        return torch.cat(Intensities, dim=1)

    def simulate_multiple_scattering_curves_lattice_coords(self, coordinates, lattice_coords, bins, q, save=False):
        '''Function to obtain multiple scattering curves of the same structure using different values of n_samples.
        This results in a more accurate scattering curve. 
        This function only works for the building block and lattice coordinates setting. 
        
        inputs:
        - coordinates: the coordinates of the building block 
        - bins: number of bins used to create the histogram 
        - q: the momentum transfer vector (q) 
        outputs:
        - Intensities: the scattering intensity curves as a function of q.'''
        n_samples_array = torch.linspace(0.5, 1.5, 10, device=self.device) * self.n_samples
        Intensities = []

        for n_samples in n_samples_array:
            self.n_samples = int(torch.round(n_samples))
            self.simulate_scattering_curve_fast_lattice(coordinates, lattice_coords, bins, q)
            Intensities.append(self.I_q.unsqueeze(1))

        return torch.cat(Intensities, dim=1)

    def rotate_coordinates_y(self, x, z, angle):
        '''Function to rotate a building block around the y-axis.
        
        inputs:
        - x: the x-coordinates of the building block
        - z: the z-coordinates of the building block
        - angle: the angle of rotation around the axis
        
        outputs:
        - x_new: the new x-coordinates of the rotated building block
        - z_new: the new z-coordinates of the rotated building block.'''
        angle = angle * torch.pi / 180
        x_new = x * torch.cos(angle) + z * torch.sin(angle)
        z_new = -x * torch.sin(angle) + z * torch.cos(angle)
        return x_new, z_new


    def rotate_coordinates_y(self, x, z, angle):
            '''Function to rotate a building block around the y-axis.
            
            inputs:
            - x: the x-coordinates of the building block
            - z: the z-coordinates of the building block
            - angle: the angle of rotation around the axis
            
            outputs:
            - x_new: the new x-coordinates of the rotated building block
            - z_new: the new z-coordinates of the rotated building block.'''
            angle = angle * torch.pi / 180
            x_new = x * torch.cos(angle) + z * torch.sin(angle)
            z_new = -x * torch.sin(angle) + z * torch.cos(angle)
            return x_new, z_new

    def rotate_coordinates_x(self, y, z, angle):
        '''Function to rotate a building block around the x-axis.
        
        inputs:
        - y: the y-coordinates of the building block
        - z: the z-coordinates of the building block
        - angle: the angle of rotation around the axis
        
        outputs:
        - y_new: the new y-coordinates of the rotated building block
        - z_new: the new z-coordinates of the rotated building block.'''
        angle = angle * torch.pi / 180
        y_new = y * torch.cos(angle) + z * torch.sin(angle)
        z_new = -y * torch.sin(angle) + z * torch.cos(angle)
        return y_new, z_new

    def rotate_coordinates_z(self, x, y, angle):
        '''Function to rotate a building block around the z-axis.
        
        inputs:
        - x: the x-coordinates of the building block
        - y: the y-coordinates of the building block
        - angle: the angle of rotation around the axis
        
        outputs:
        - x_new: the new x-coordinates of the rotated building block
        - y_new: the new y-coordinates of the rotated building block.'''
        angle = angle * torch.pi / 180
        x_new = x * torch.cos(angle) + y * torch.sin(angle)
        y_new = -x * torch.sin(angle) + y * torch.cos(angle)
        return x_new, y_new

    def rotate_building_block(self, coordinates, center):
        '''This function rotates a building block about the x, y, z axis and places them on a specific coordinate.
        
        inputs:
        - coordinates: matrix of building block coordinates with x, y, z in each column
        - center: matrix of lattice coordinates which is where the building block will be centered at. This is the lattice_coordinates with an extra 3 columns at the end which 
        account for the amount of rotation around the x, y, z axis needed for each point of the lattice.
        
        outputs:
        - coordinates: the coordinates of the structure, which is the building block with rotation plus the lattice.'''
        coordinates = self.to_tensor(coordinates)
        center = self.to_tensor(center)

        coordinates[:, 1], coordinates[:, 2] = self.rotate_coordinates_x(coordinates[:, 1], coordinates[:, 2], center[:, 3])
        coordinates[:, 0], coordinates[:, 2] = self.rotate_coordinates_y(coordinates[:, 0], coordinates[:, 2], center[:, 4])
        coordinates[:, 0], coordinates[:, 1] = self.rotate_coordinates_z(coordinates[:, 0], coordinates[:, 1], center[:, 5])

        coordinates[:, 0] += center[:, 0]
        coordinates[:, 1] += center[:, 1]
        coordinates[:, 2] += center[:, 2]
        return coordinates



############################ Functions to scattering curves with periodic boundary conditions #############################

def create_superbox_with_orientations(data, box_length, n_replicas=3):
    """
    Replicates a periodic simulation box n_replicas^3 times to form a superbox,
    preserving orientation vectors.

    Parameters:
        data (np.ndarray): Array of shape (N, 6) with [x, y, z, ox, oy, oz].
        box_length (float): Length of the cubic simulation box.
        n_replicas (int): Number of box copies along each axis (default 3).

    Returns:
        np.ndarray: Array of shape (N * n_replicas^3, 6) with replicated positions and orientations.
    """
    assert data.shape[1] == 6, "Input array must be of shape (N, 6)"
    
    positions = data[:, :3]
    orientations = data[:, 3:]
    
    offset_range = np.arange(-(n_replicas // 2), n_replicas // 2 + 1)
    shifts = np.array(np.meshgrid(offset_range, offset_range, offset_range)).T.reshape(-1, 3)
    
    all_data = []
    for shift in shifts:
        shift_vector = shift * box_length
        replicated_positions = positions + shift_vector
        replicated = np.hstack([replicated_positions, orientations])
        all_data.append(replicated)
    
    return np.vstack(all_data)


def extract_subvolume_centered(data, subbox_size, verbose=True):
    """
    Extracts a subvolume centered at the geometric center of the superbox,
    and reports the number of particles in the full and extracted region.

    Parameters:
        data (np.ndarray): Array of shape (N, 6), with [x, y, z, ox, oy, oz]
        subbox_size (float or tuple): Size of the subvolume (lx, ly, lz)
        verbose (bool): Whether to print counts (default True)

    Returns:
        np.ndarray: Filtered array (M, 6) within the central subvolume
    """
    if isinstance(subbox_size, (float, int)):
        subbox_size = np.array([subbox_size] * 3)
    else:
        subbox_size = np.array(subbox_size)

    positions = data[:, :3]

    # Compute bounding box and center of the superbox
    min_coords = positions.min(axis=0)
    max_coords = positions.max(axis=0)
    superbox_center = (min_coords + max_coords) / 2

    # Compute subvolume bounds
    half_sub = subbox_size / 2
    lower_bound = superbox_center - half_sub
    upper_bound = superbox_center + half_sub

    # Create mask for particles inside the subvolume
    mask = np.all((positions >= lower_bound) & (positions <= upper_bound), axis=1)
    subvolume = data[mask]

    # Verbose reporting
    #if verbose:
    #    print(f"Total particles in superbox: {len(data)}")
    #    print(f"Particles in subvolume: {len(subvolume)}")
    proportion = len(subvolume)/len(data)
    return proportion, subvolume


def invariant(data):
    '''Calculates the invariant'''
    q = data[:,0]
    I = data[:,1]
    invariant = integrate.simpson(q**2*I, q)
    return invariant


def calculate_scattering_curve_with_PCB(lattice_coordinates, points, box_length_simulation, histogram_bins, q, sub_box_fraction, plot):
    superbox_data = create_superbox_with_orientations(lattice_coordinates, box_length=box_length_simulation)
    plt.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots(figsize=(10,7))
    for i in range(len(sub_box_fraction)):
        sub_box_length = box_length_simulation*sub_box_fraction[i]
        proportion, sub_box = extract_subvolume_centered(superbox_data, sub_box_length)
        n_samples = 10000000
        simulator = scattering_simulator(n_samples)
        simulator.sample_building_block(points)
        simulator.sample_lattice_coordinates(sub_box)
        simulator.calculate_structure_coordinates()
        I_q = simulator.simulate_scattering_curve_fast_lattice(points, sub_box, histogram_bins, q, save=False).numpy()
        inv_I_q = invariant(np.hstack((q.reshape(-1,1), I_q.reshape(-1,1))))
        I_q_scaled = I_q/inv_I_q*proportion
        ax.plot(q, I_q_scaled, linewidth = 3)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_ylabel('Intensity (arb. unit)')
        ax.set_xlabel('q ($\\AA^{-1}$)')
        #plt.legend(fontsize=14)
        data = np.hstack((q.reshape(-1,1), I_q.reshape(-1,1)))
        #np.save(path + scattering_curve_' + str(i) + '.npy', data)   
        print('Simulation of Subbox ', i, '/', len(sub_box_fraction)-1 ,'complete ...')
        if i == 0:
            I_q_avg = I_q_scaled
        else:
            I_q_avg = I_q_avg + I_q_scaled 
    ax.plot(q, I_q_avg, color='k', linewidth = 3)
    data = np.hstack((q.reshape(-1,1), I_q_avg.reshape(-1,1)))
    if plot == False:
        plt.close()
    #plt.savefig(path + scattering_curve_plot.png', dpi=600, bbox_inches="tight")
    return data 
