import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt

class scattering_simulator:
    def __init__(self, n_samples):
        self.n_samples = n_samples
        np.random.seed(1)
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
        np.random.seed(1)
        building_block_coordinates = self.relative_coordinates(building_block_coordinates)
        rand_num1 = np.random.randint(0,len(building_block_coordinates), self.n_samples)
        rand_num2 = np.random.randint(0,len(building_block_coordinates), self.n_samples)
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
        np.random.seed(1)
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
            self.structure_coordinates_1 = self.building_block_coordinates_1[:,:-1] + self.lattice_coordinates_1
            self.structure_coordinates_2 = self.building_block_coordinates_2[:,:-1] + self.lattice_coordinates_2
            self.structure_coordinates_1 = np.hstack((self.structure_coordinates_1, self.building_block_coordinates_1[:,-1].reshape(-1,1)))
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
        #self.p_r = self.p_r/np.max(self.p_r)

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
        for i in range(len(q)):
            #I = (scipy.integrate.simps(self.p_r*np.sin(q[i]*self.r)/q[i]/self.r, self.r))*2 + self.n_samples
            I = (scipy.integrate.simps(4*np.pi*self.p_r*np.sin(q[i]*self.r)/q[i]/self.r, self.r))
            #I = np.sum(np.cos(q[i]*self.distances))**2 + np.sum(np.sin(q[i]*self.distances))**2
            I_q.append(I)
        self.I_q = np.array(I_q)

        # self.p_r = self.convert_to_pairwise(self.r, self.I_q, q)
        

        # I_q = []
        # for i in range(len(q)):
        #     I = (scipy.integrate.simps(4*np.pi*self.p_r*np.sin(q[i]*self.r)/q[i]/self.r, self.r))
        #     #I = np.sum(np.cos(q[i]*self.distances))**2 + np.sum(np.sin(q[i]*self.distances))**2
        #     I_q.append(I)
        # self.I_q = np.array(I_q)


    def convert_to_pairwise(self, r, I, q):
        p_r = []
        for i in range(len(r)):
            p = scipy.integrate.simps(I*q*r[i]*np.sin(q*r[i]), q)
            p_r.append(p)
        p_r = np.array(p_r)
        return p_r

    def convert_to_smeared_intensity(self, q, smearing):
        I_qsmeared = []
        q_mean = np.mean(q)
        R = 1/np.sqrt(2*np.pi*smearing**2)*np.exp(-(q - q_mean)**2/(2*smearing**2))
        Iqs = np.sum(self.I_q*R)
        self.I_q = Iqs


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


    def simulate_smeared_scattering_curve(self, bins, q, smearing, save=False):
        '''Function to run the calculation of the coordinates of the structure to the scattering intensity curve
        inputs:
        - bins: number of bins used to create the histogram 
        - q: the momentum transfer vector (q) 
        outputs:
        - self.I_q: the scattering intensity curve as a function of q '''
        self.distance_function(save=save)
        self.create_histogram(bins)
        self.convert_to_intensity(q)
        self.convert_to_smeared_intensity(q, smearing)
        return self.I_q


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




