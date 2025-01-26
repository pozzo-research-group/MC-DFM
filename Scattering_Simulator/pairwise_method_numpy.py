import numpy as np
import math
import scipy
import h5py

class scattering_simulator:
    def __init__(self, n_samples):
        self.n_samples = n_samples
        #np.random.seed(1)
        return 

    def sample_building_block(self, building_block_coordinates):
        '''Randomly samples "n_samples" number of points from  "building_block_coordinates"
        inputs: 
        - building_block_coordinates: array with x,y,z poitions of the building block coordinates. The 
        array should have 3 columns and any number of rows
        - n: scalar to determine how many samples should be taken from the building block coordinates
        results:
        - self.building_block_coordinates_1: an array with randomly sampled coordinates from the building block
        - self.building_block_coordinates_2: an array with randomly sampled coordinates from the building block
        '''
        #np.random.seed(1)
        building_block_coordinates = self.relative_coordinates(building_block_coordinates)
        rand_num1 = np.random.randint(0,int(len(building_block_coordinates)), self.n_samples)
        rand_num2 = np.random.randint(0,int(len(building_block_coordinates)), self.n_samples)
        self.building_block_coordinates_1 = building_block_coordinates[rand_num1,:]
        self.building_block_coordinates_2 = building_block_coordinates[rand_num2,:]

    def sample_lattice_function(self, d_x, d_y, d_z, lattice_points_x, lattice_points_y, lattice_points_z, save=False):
        '''Randomly samples ''n_samples'' number of points from the lattice
        inputs:
         - d_x: spacing or distance between consecutive lattice points in the x direction
         - d_y: spacing or distance between consecutive lattice points in the y direction
         - d_z: spacing or distance between consecutive lattice points in the z direction
         - lattice_points_x: number of lattice points in the x direction
         - lattice_points_y: number of lattice points in the y direction
         - lattice_points_z: number of lattice points in the z direction
        results:
         - self.lattice_coordinates_1: an array with randomly sampled coordinates from the lattice
         - self.lattice_coordinates_2: an array with randomly sampled coordinates from the lattice
         '''
        #np.random.seed(1)
        rand_num_x = np.random.randint(0,lattice_points_x, self.n_samples)
        rand_num_y = np.random.randint(0,lattice_points_y, self.n_samples)
        rand_num_z = np.random.randint(0,lattice_points_z, self.n_samples)

        lattice_coordinates_x = d_x*rand_num_x
        lattice_coordinates_y = d_y*rand_num_y
        lattice_coordinates_z = d_z*rand_num_z
        self.lattice_coordinates_1 = np.hstack((lattice_coordinates_x.reshape(-1,1), lattice_coordinates_y.reshape(-1,1), lattice_coordinates_z.reshape(-1,1)))
        
        rand_num_x = np.random.randint(0,lattice_points_x, self.n_samples)
        rand_num_y = np.random.randint(0,lattice_points_y, self.n_samples)
        rand_num_z = np.random.randint(0,lattice_points_z, self.n_samples)

        lattice_coordinates_x = d_x*rand_num_x
        lattice_coordinates_y = d_y*rand_num_y
        lattice_coordinates_z = d_z*rand_num_z
        self.lattice_coordinates_2 = np.hstack((lattice_coordinates_x.reshape(-1,1), lattice_coordinates_y.reshape(-1,1), lattice_coordinates_z.reshape(-1,1)))
        #free up storage 
        if save == False:
            rand_num_x = 0
            rand_num_y = 0
            rand_num_z = 0

    def sample_lattice_coordinates(self, lattice_coordinates, save=False):
        '''Randomly samples ''n_samples'' number of points from the lattice
        inputs:
         - lattice_coordinates: 3d coordinates of the lattice 
         results:
         - self.lattice_coordinates_1: an array with randomly sampled coordinates from the lattice
         - self.lattice_coordinates_2: an array with randomly sampled coordinates from the lattice
         '''
        #np.random.seed(1)
        rand_num = np.random.randint(0,lattice_coordinates.shape[0], self.n_samples)
        self.lattice_coordinates_1 = lattice_coordinates[rand_num, :]        
        rand_num = np.random.randint(0,lattice_coordinates.shape[0], self.n_samples)
        self.lattice_coordinates_2 = lattice_coordinates[rand_num, :]


    def calculate_structure_coordinates(self, save=False):
        '''Adds the building block coordinates to the lattice coordinates to obtain the structure coordinates
        inputs (automatically loaded):
        - self.building_block_coordinates
        - self.lattice_coordinates 
        results:
        - self.structure coordinates
        '''
        if self.building_block_coordinates_1.shape[1] == 3: #uniform SLD
            self.structure_coordinates_1 = self.building_block_coordinates_1 + self.lattice_coordinates_1
            self.structure_coordinates_2 = self.building_block_coordinates_2 + self.lattice_coordinates_2
        else: # different SLD 
            if self.lattice_coordinates_1.shape[1] == 3: #no rotation 
                self.structure_coordinates_1 = self.building_block_coordinates_1[:,:-1] + self.lattice_coordinates_1
                self.structure_coordinates_2 = self.building_block_coordinates_2[:,:-1] + self.lattice_coordinates_2
                self.structure_coordinates_1 = np.hstack((self.structure_coordinates_1, self.building_block_coordinates_1[:,-1].reshape(-1,1)))
                self.structure_coordinates_2 = np.hstack((self.structure_coordinates_2, self.building_block_coordinates_2[:,-1].reshape(-1,1)))
            else: #rotation of building block included 
                self.structure_coordinates_1 = self.rotate_building_block(self.building_block_coordinates_1[:,:-1], self.lattice_coordinates_1)
                self.structure_coordinates_1 = np.hstack((self.structure_coordinates_1, self.building_block_coordinates_1[:,-1].reshape(-1,1)))
                
                self.structure_coordinates_2 = self.rotate_building_block(self.building_block_coordinates_2[:,:-1], self.lattice_coordinates_2)
                self.structure_coordinates_2 = np.hstack((self.structure_coordinates_2, self.building_block_coordinates_2[:,-1].reshape(-1,1)))


        #find center of structure 
        # center = np.mean(self.structure_coordinates_1, axis = 0)
        # p1 = np.array(center[0]*len(self.structure_coordinates_1)).reshape(-1,1)
        # p2 = np.array(center[1]*len(self.structure_coordinates_1)).reshape(-1,1)
        # p3 = np.array(center[2]*len(self.structure_coordinates_1)).reshape(-1,1)
        # self.structure_coordinates_1 = np.hstack((p1, p2, p3))


        #free up storage
        if save == False:
            #self.building_block_coordinates_1 = 0
            #self.building_block_coordinates_2 = 0
            self.lattice_coordinates_1 = 0
            self.lattice_coordinates_2 = 0


    def use_building_block_as_structure(self, save=False):
        '''Uses the building block as the structure 
        inputs (automatically loaded):
        - self.building_block_coordinates
        - self.lattice_coordinates 
        results:
        - self.structure coordinates
        '''
        self.structure_coordinates_1 = self.building_block_coordinates_1
        self.structure_coordinates_2 = self.building_block_coordinates_2
        #free up storage
        if save == False:
            #self.building_block_coordinates_1 = 0
            #self.building_block_coordinates_2 = 0
            self.lattice_coordinates_1 = 0
            self.lattice_coordinates_2 = 0


    def relative_coordinates(self, building_block_coordinates):
        '''Used to center the building block coordinates to have a center at coordinates (0,0,0)
        inputs:
        - building_block_coordinates_centered: an array of the x-y-z coordinates of the building block. The array
        should have 3 columns and any number of rows 
        outputs:
        - building_block_coordinates_centered: an array of the x-y-z coordinates of the building block with a center
        at point (0,0,0)
        '''
        rel_x = building_block_coordinates[:,0] - np.mean(building_block_coordinates[:,0])
        rel_y = building_block_coordinates[:,1] - np.mean(building_block_coordinates[:,1])
        rel_z = building_block_coordinates[:,2] - np.mean(building_block_coordinates[:,2])
        if building_block_coordinates.shape[1] == 3: #uniform SLD 
            building_block_coordinates_centered = np.hstack((rel_x.reshape(-1,1), rel_y.reshape(-1,1), rel_z.reshape(-1,1)))
        else: #difference in SLD
            building_block_coordinates_centered = np.hstack((rel_x.reshape(-1,1), rel_y.reshape(-1,1), rel_z.reshape(-1,1), building_block_coordinates[:,3].reshape(-1,1)))
        return building_block_coordinates_centered

    def distance_function(self, save):
        '''Calculates the pairwise euclidean distance between the rows of two different arrays
        inputs (automatically loaded):
        - self.structure_coordinates_1: The randomly sampled x-y-z coordinates of the structure 
        - self.structure_coordinates_2: The randomly sampled x-y-z coordinates of the structure 
        result:
        -self.distances: a 1D array of the pairwise distances of the two randomly sampled arrays of the structure coordinates
        '''
        p1 = self.structure_coordinates_1
        p2 = self.structure_coordinates_2
        self.distances = np.sqrt((p1[:,0] - p2[:,0])**2 + (p1[:,1] - p2[:,1])**2 + (p1[:,2] - p2[:,2])**2)
        #free up storage
        #if save == False:
            #self.structure_coordinates_1 = 0
            #self.structure_coordinates_2 = 0

    def create_histogram(self, bins):
        '''Creates a histogram of the pairwise distances between two randomly selected points from the structre coordinates
        inputs:
        - bins: number of bins for the histogram
        - self.distances: a 1D array of the pairwise distances of the two randomly sampled arrays of the structure coordinates
        results:
        - self.p_r: the pairwise distribution function  
        - self.r: the pairwise distances of the two randomly sampled coordinates of the structure
        '''
        if self.structure_coordinates_1.shape[1] == 3:
            x = np.histogram(self.distances, bins = bins)
        else:
            SLD = self.structure_coordinates_1[:,-1]*self.structure_coordinates_2[:,-1]
            x = np.histogram(self.distances, bins = bins, weights = SLD)
        self.p_r = x[0]
        self.r = x[1][1:]
        # prevent self sampling which eliminates a "0" distance
        if self.r[0] == 0:
            self.p_r[0] = 0

    def convert_to_intensity(self, q):
        '''Converts the pairwise distribution function into the scattering intensity as a function of q 
        inputs:
        - q: the momentum transfer vector (q) 
        - self.p_r: the pairwise distribution function  
        - self.r: the pairwise distances of the two randomly sampled coordinates of the structure
        results:
        - self.I_q: the scattering intensity curve as a function of q 
        '''
        I_q = []
        self.q = q
        for i in range(len(q)):
            I = (scipy.integrate.simpson(4*np.pi*self.p_r*np.sin(q[i]*self.r)/q[i]/self.r, self.r))
            I_q.append(I)
        self.I_q = np.array(I_q)


    def convert_to_pairwise(self, r, I, q):
        '''Converts the scattering intensity back to the pairwise distribution function
        inputs: 
        - r: the pairwise distances of the two randomly sampled coordinates of the structure
        - I: the scattering intensity 
        - q: the momentum transfer vector
        outputs:
        - p_r: the pairwise distribution which is a function of (r)'''
        p_r = []
        for i in range(len(r)):
            p = scipy.integrate.simpson(I*q*r[i]*np.sin(q*r[i]), q)
            p_r.append(p)
        p_r = np.array(p_r)
        return p_r

    def simulate_scattering_curve(self, bins, q, save=False):
        '''Function to run the calculation of the coordinates of the structure to the scattering intensity curve
        inputs:
        - bins: number of bins used to create the histogram 
        - q: the momentum transfer vector (q) 
        outputs:
        - self.I_q: the scattering intensity curve as a function of q '''
        self.distance_function(save=save)
        self.create_histogram(bins)
        self.convert_to_intensity(q)
        return self.I_q
    
    def simulate_scattering_curve_fast(self, coordinates, bins, q, save=False):
        '''Runs the simulation to calculate the scattering curve in a loop with a reduced number of n_samples:
            It is slightly faster than the normal method of calculating the scattering curve 
            Only works with buliding block setting
        - coordinates: coordinates of the building block structure
        - bins: number of bins used to create the histogram 
        - q: the momentum transfer vector (q) 
        outputs:
        - self.I_q: the scattering intensity curve as a function of q '''

        self.n_samples = int(np.round(self.n_samples/5))
        for i in range(5):
            self.sample_building_block(coordinates)
            self.use_building_block_as_structure()
            self.distance_function(save=False)
            self.create_histogram(bins)
            if i == 0:
                all_p_r = self.p_r
            else:
                all_p_r = all_p_r + self.p_r
        self.p_r = all_p_r
        self.convert_to_intensity(q)
        return self.I_q
    

    def simulate_scattering_curve_fast_lattice(self, coordinates, lattice, bins, q, save=False):
        '''Runs the simulation to calculate the scattering curve in a loop with a reduced number of n_samples.
           It is slightly faster than the normal method. 
           This function is used when there is a building block and lattice coordinates. 

        - coordinates: coordinates of the building block structure
        - lattice: the coordinates of the lattice where each building block is placed 
        - bins: number of bins used to create the histogram 
        - q: the momentum transfer vector (q) 
        outputs:
        - self.I_q: the scattering intensity curve as a function of q '''

        self.n_samples = int(np.round(self.n_samples/5))
        for i in range(5):
            self.sample_building_block(coordinates)
            self.sample_lattice_coordinates(lattice)
            self.calculate_structure_coordinates()
            self.distance_function(save=False)
            self.create_histogram(bins)
            if i == 0:
                all_p_r = self.p_r
            else:
                all_p_r = all_p_r + self.p_r
        self.p_r = all_p_r
        self.convert_to_intensity(q)
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
        - Intensities: the scattering intensity curves as a function of q'''
        n_samples_array  = np.linspace(0.5, 1.5, 10)*self.n_samples
        for i in range(len(n_samples_array)):
            self.n_samples = int(np.round(n_samples_array[i]))
            self.sample_building_block(coordinates)
            self.use_building_block_as_structure()
            self.distance_function(save=save)
            self.create_histogram(bins)
            self.convert_to_intensity(q)
            self.I_q = self.I_q/self.I_q[0]
            if i == 0:
                Intensities = self.I_q.reshape(-1,1)
            else:
                Intensities = np.hstack((Intensities, self.I_q.reshape(-1,1)))
        return Intensities


    def simulate_multiple_scattering_curves_lattice_coords(self, coordinates, lattice_coords, bins, q,save=False):
        '''Function to obtain multiple scattering curves of the same structure using different values of n_samples.
        This results in a more accurate scattering curve. 
        This function only works for the building block and lattice coordinates setting. 
        inputs:
        - coordinates: the coordinates of the building block 
        - bins: number of bins used to create the histogram 
        - q: the momentum transfer vector (q) 
        outputs:
        - Intensities: the scattering intensity curves as a function of q'''
        n_samples_array  = np.linspace(0.5, 1.5, 10)*self.n_samples
        for i in range(len(n_samples_array)):
            self.n_samples = int(np.round(n_samples_array[i]))
            self.I_q = self.simulate_scattering_curve_fast_lattice(coordinates, lattice_coords, bins, q)
            #self.I_q = self.I_q/self.I_q[0]
            if i == 0:
                Intensities = self.I_q.reshape(-1,1)
            else:
                Intensities = np.hstack((Intensities, self.I_q.reshape(-1,1)))
        return Intensities

    def save_h5py(self, dir):
        '''Saves results in h5py format into specified directory
        inputs:
        - dir: the directory where the file should be saved'''
        h5f = h5py.File(dir, 'w')
        h5f.create_dataset('q', data=self.q)
        h5f.create_dataset('I', data=self.I_q)
        h5f.create_dataset('r', data=self.r)
        h5f.create_dataset('p_r', data=self.p_r)
        #h5f.create_dataset('structure', data=self.structure_coordinates_1)
        h5f.create_dataset('N', data=self.n_samples)


    def rotate_coordinates_y(self, x, z, angle):
        '''Function to rotate a building block around the y-axis
        inputs:
        - x: the x-coordinates of the building block
        - z: the z-coordinates of the building block
        - angle: the angle of rotation around the axis
        outputs:
        - x_new: the new x-coordinates of the rotated building block
        - z_new: the new z-coordinates of the rotated building block 
        '''
        angle = angle*math.pi/180
        x_new = x*np.cos(angle) + z*np.sin(angle)
        z_new = -x*np.sin(angle) + z*np.cos(angle)
        return x_new, z_new

    def rotate_coordinates_x(self, y, z, angle):
        '''Function to rotate a building block around the x-axis
        inputs:
        - y: the y-coordinates of the building block
        - z: the z-coordinates of the building block
        - angle: the angle of rotation around the axis
        outputs:
        - y_new: the new y-coordinates of the rotated building block
        - z_new: the new z-coordinates of the rotated building block 
        '''
        angle = angle*math.pi/180
        y_new = y*np.cos(angle) + z*np.sin(angle)
        z_new = -y*np.sin(angle) + z*np.cos(angle)
        return y_new, z_new

    def rotate_coordinates_z(self, x, y, angle):
        '''Function to rotate a building block around the z-axis
        inputs:
        - x: the x-coordinates of the building block
        - y: the y-coordinates of the building block
        - angle: the angle of rotation around the axis
        outputs:
        - x_new: the new x-coordinates of the rotated building block
        - y_new: the new y-coordinates of the rotated building block 
        '''
        angle = angle*math.pi/180
        x_new = x*np.cos(angle) + y*np.sin(angle)
        y_new = -x*np.sin(angle) + y*np.cos(angle)
        return x_new, y_new

    def rotate_building_block(self, coordinates, center):
        ''''
        This function rotates a building block about the x,y,z axis and places them on a specific coordinate.
        inputs:
        - coordinates: matrix of building block coordinates with x,y,z in each column
        - center: matrix of lattice coordinates which is where the building block will be centered at. This is the lattice_coordinates with an extra 3 columns at the end which 
        account for the amount of rotation around the x,y,z axis needed for the each point of the lattice.
        outputs:
        - coordinates: the coordinates of the structure, which is the building block with rotation plus the lattice
        '''
        coordinates[:,1], coordinates[:,2] = self.rotate_coordinates_x(coordinates[:,1], coordinates[:,2], center[:,3])
        coordinates[:,0], coordinates[:,2] = self.rotate_coordinates_y(coordinates[:,0], coordinates[:,2], center[:,4])
        coordinates[:,0], coordinates[:,1] = self.rotate_coordinates_z(coordinates[:,0], coordinates[:,1], center[:,5])
        
        coordinates[:,0] = coordinates[:,0] + center[:,0]
        coordinates[:,1] = coordinates[:,1] + center[:,1]
        coordinates[:,2] = coordinates[:,2] + center[:,2]
        return coordinates
    





    ########## The following equations are used to run the Debye method #############

    def prepare_modified_debye_distance(self):
        '''removes any 0 distances from the distance array'''
        if self.building_block_coordinates_1.shape[1] == 4: #Non constant SLD
            self.SLD_1 = self.structure_coordinates_1[:,-1]
            self.SLD_2 = self.structure_coordinates_2[:,-1]
        self.distance_function(save=False)
        self.distances = np.hstack((self.distances.reshape(-1,1), self.SLD_1.reshape(-1,1), self.SLD_2.reshape(-1,1)))
        self.distances = np.delete(self.distances, np.where(self.distances == 0)[0], axis=0)

    def modified_debye(self, q):
        '''Modified Debye equation to calculate the scattering intensity from a specified number of pairwise distances
        inputs:
        - q: the momentum transfer vector (q) 
        - dist: The pairwise distances of randomly sampled points
        outputs:
        - Intensity: Intensity of scattering curve as a function of q'''
        I_list = []
        for i in range(len(q)):
            if self.building_block_coordinates_1.shape[1] == 4:
                I = np.sum(self.distances[:,-2]*self.distances[:,-1]*np.sin(q[i]*self.distances[:,0])/q[i]/self.distances[:,0])
            else:
                I = np.sum(np.sin(q[i]*self.distances)/q[i]/self.distances)
            I_list.append(I)
        I = np.array(I_list)
        return I 

    def meshgrid_to_array(self, xx):
        '''Equation to perform a meshgrid and then convert the resulting AxA array into a 1D array.
        '''
        x_lst = []
        for j in range(xx.shape[0]):
            for k in range(xx.shape[1]):
                x_lst.append(xx[j,k])
        coordinates = np.array(x_lst).reshape(-1,1)
        return coordinates 

    def debye(self, q, distance):
        '''Debye equation using matrix operations 
        '''
        q_times_distance = distance*q
        I_q = np.sum(np.sin(q_times_distance)/q_times_distance)
        return I_q

    def distance_func(self, p1, p2):
        '''Calculates the pairwise euclidean distance between the rows of two different arrays
        inputs (automatically loaded):
        - self.structure_coordinates_1: The randomly sampled x-y-z coordinates of the structure 
        - self.structure_coordinates_2: The randomly sampled x-y-z coordinates of the structure 
        result:
        -self.distances: a 1D array of the pairwise distances of the two randomly sampled arrays of the structure coordinates
        '''
        distances = np.sqrt((p1[:,0] - p2[:,0])**2 + (p1[:,1] - p2[:,1])**2 + (p1[:,2] - p2[:,2])**2)
        return distances 

    def calculate_debye(self, coordinates, q):
        '''Used to calculate the Debye equation using matrix operations. The inputs are the coordinates and the q value
        and the outputs are the intensity. 
        '''
        coord_index = np.linspace(0, len(coordinates)-1, len(coordinates))
        xx,yy = np.meshgrid(coord_index, coord_index)
        coord_index_1 = self.meshgrid_to_array(xx).astype(int).flatten()
        coord_index_2 = self.meshgrid_to_array(yy).astype(int).flatten()
        coordinates_1 = coordinates[coord_index_1,:]
        coordinates_2 = coordinates[coord_index_2,:]
        distance = self.distance_func(coordinates_1, coordinates_2)
        distance = np.delete(distance, np.where(distance == 0)[0])
        I_q = []
        for i in range(len(q)):
            I = self.debye(q[i], distance)
            I_q.append(I)
        I_q = np.array(I_q)
        return I_q




