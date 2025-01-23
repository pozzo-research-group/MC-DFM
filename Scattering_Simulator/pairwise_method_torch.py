import torch
import ray


class scattering_simulator:
    def __init__(self, n_samples, device=None):
        self.n_samples = n_samples
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def to_tensor(self, array):
        '''Converts a NumPy array to a PyTorch tensor on the specified device.'''
        return torch.tensor(array, device=self.device, dtype=torch.float32)

    @ray.remote
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

    @ray.remote
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

    @ray.remote
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

    @ray.remote
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
            self.sample_building_block.remote(coordinates)
            self.sample_lattice_coordinates.remote(lattice)
            self.calculate_structure_coordinates.remote()
            self.distance_function.remote(save=False)
            self.create_histogram.remote(bins)

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

    @ray.remote
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

    @ray.remote
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
            self.sample_building_block.remote(coordinates)
            self.use_building_block_as_structure.remote()
            self.distance_function.remote(save=False)
            self.create_histogram.remote(bins)
            if all_p_r is None:
                all_p_r = self.p_r.clone()
            else:
                all_p_r += self.p_r

        self.p_r = all_p_r
        self.convert_to_intensity(q)
        return self.I_q
    
    @ray.remote
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
            self.sample_building_block.remote(coordinates)
            self.sample_lattice_coordinates.remote(lattice)
            self.calculate_structure_coordinates.remote()
            self.distance_function.remote(save=False)
            self.create_histogram.remote(bins)

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
