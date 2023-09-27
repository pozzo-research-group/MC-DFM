import numpy as np

class genetic_algorithm:
    def __init__(self, n_offspring, mutation_rate):
        '''
        hyperparameters of the genetic algorithm
        -n_offspring: this integer determines how many samples will be suggested in the next iteration. Similar to batch size
        -mutation_rate: this float value from 0-1 balances exploration vs exploitation. 0 = exploitation, 1 = exploration
        '''
        self.n_parents = 50
        self.n_offspring = n_offspring
        self.mutation_rate = mutation_rate
        self.max_fitness_lst = []
        self.best_solution_lst = []
        return


    def best_solution(self):
        '''
        Returns the best solution found from the optimization
        '''
        best_solution_loc = np.argmax(self.max_fitness_lst)
        return self.best_solution_lst[best_solution_loc]

    def initialize(self, x, y):
        self.x = x
        self.y = y

    def fitness(self):
        '''
        Sorts the x array according to its correspoinding y value. A greater y value
        will give the row of the x array a higher chance to proceed to the next generation
        '''
        new_array = np.hstack((self.x, self.y.reshape(-1, 1)))
        self.sorted_array = new_array[np.argsort(new_array[:, -1])]
        lower_fitness, upper_fitness = np.array_split(self.sorted_array, 2)
        self.sorted_array = upper_fitness
        self.median_fitness = np.median(self.y)
        self.max_fitness = np.max(self.y)
        self.max_fitness_lst.append(self.max_fitness)
        self.best_solution_lst.append(self.sorted_array[-1,0:-1])


    def select_parents(self):
        '''
        Randomly selects parents, the ones with a higher fitness will have
        a higher chance of being selected. Uses a roulette wheel approach
        where the probability of being selected is proportional to the fitness.
        '''
        fitness_list = self.sorted_array[:, -1]
        fitness_sum = np.sum(fitness_list)
        probability = fitness_list/fitness_sum
        cumsum = np.cumsum(probability)
        for itr in range(self.n_parents):
            rand_num = np.random.rand()
            for i in range(cumsum.shape[0]):
                if cumsum[i] > rand_num:
                    UB = cumsum[i]
                    if i == 0:
                        LB = cumsum[i]
                        break
                    else:
                        LB = cumsum[i-1]
                        break
            if itr == 0:
                self.parents = self.sorted_array[i]
                LB = LB
                LB = UB
            else:
                self.parents = np.vstack((self.parents, self.sorted_array[i]))


    def crossover(self):
        '''
           Performs a crossover between the parents to create offspring that
           have characteristcs of both parents. The way this algorithm works is by
           converting the float numbers into strings and then exchanging them between
           two parents 
        '''

        for i in range(self.n_offspring):
            random_row1 = int(np.round(np.random.rand()*self.parents.shape[0]-1))
            random_row2 = int(np.round(np.random.rand()*self.parents.shape[0]-1))
            p1 = self.parents[random_row1, :]  # selects first parent
            p2 = self.parents[random_row2, :]  # selects second parent
            row_of_concs = []
            for n_stocks in range(self.parents.shape[1]-1):
                p1_conc = str(p1[n_stocks])
                p2_conc = str(p2[n_stocks])

                def normalize_sig_figs(p1_conc):
                    if len(p1_conc) < 5:
                        p1_conc = p1_conc + '0' + '0'
                    return p1_conc

                def cross_parents(p1_conc, p2_conc):
                    zero = p1_conc[0]
                    decimal = p1_conc[1]
                    p1_digit1 = p1_conc[2]
                    p1_digit2 = p1_conc[3]
                    p1_digit3 = p1_conc[4]
                    p2_digit1 = p2_conc[2]
                    p2_digit2 = p2_conc[3]
                    p2_digit3 = p2_conc[4]
                    random_number = np.random.rand()
                    if random_number < 0.5:
                        digit1 = p1_digit1
                    else:
                        digit1 = p2_digit1
                    random_number = np.random.rand()
                    if random_number < 0.5:
                        digit2 = p1_digit2
                    else:
                        digit2 = p2_digit2
                    random_number = np.random.rand()
                    if random_number < 0.5:
                        digit3 = p1_digit3
                    else:
                        digit3 = p2_digit3
                    offspring_conc = float(zero + decimal +
                                           digit1 + digit2 + digit3)
                    return offspring_conc
                p1_conc = normalize_sig_figs(p1_conc)
                p2_conc = normalize_sig_figs(p2_conc)
                offspring_conc = cross_parents(p1_conc, p2_conc)
                row_of_concs.append(offspring_conc)
            row_of_offspring = np.asarray(row_of_concs)
            if i == 0:
                offspring = row_of_offspring
            else:
                offspring = np.vstack((offspring, row_of_offspring))
        self.offspring = offspring

    def mutation(self):
        '''
            Performs a mutation on some of the values in the offspring array.
            It converts the value to a string and then changes one of the
            digits to a random number.
        '''

        def normalize_sig_figs(p1_red_conc):
            if len(p1_red_conc) < 5:
                p1_red_conc = p1_red_conc + '0' + '0'
            return p1_red_conc

        self.array = self.offspring.copy()
        for j in range(self.offspring.shape[0]):
            for i in range(self.offspring.shape[1]):
                if np.random.rand() < self.mutation_rate:
                    conc = str(self.offspring[j, i])
                    conc = normalize_sig_figs(conc)
                    column = int(np.round(np.random.uniform(2, 4)))
                    random_int = str(int(np.round(np.random.uniform(0, 9))))
                    if column == 2:
                        digit1 = random_int
                        digit2 = conc[3]
                        digit3 = conc[4]
                    elif column == 3:
                        digit1 = conc[2]
                        digit2 = random_int
                        digit3 = conc[4]
                    else:
                        digit1 = conc[2]
                        digit2 = conc[3]
                        digit3 = random_int
                    mutated_conc = conc[0] + conc[1] + digit1 + digit2 + digit3
                    mutated_conc = float(mutated_conc)
                    self.array[j, i] = mutated_conc

    def run(self, x, y):
        self.initialize(x, y)
        self.fitness()
        self.select_parents()
        self.crossover()
        self.mutation()
        return self.array