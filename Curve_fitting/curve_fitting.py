import numpy as np
import pandas as pd
import sys
import os
sys.path.append('../')
from genetic_algorithm import GA_functions as GA
import scipy


def convert_data(data, model):
    model_x = model[:,0]
    model_y = model[:,1]
    index = np.linspace(0, len(model_x)-1, len(model_x)) 
    model_q_new = []
    model_I_new = []
    for i in range(len(data)):
        data_q = data[i,0]
        array = np.abs(model_x - data_q)
        array = np.hstack((array.reshape(-1,1), index.reshape(-1,1)))
        array = array[np.argsort(array[:, 0])]
        loc = int(array[0,1])
        model_q_new.append(model_x[loc])
        model_I_new.append(model_y[loc])
    q = np.array(model_q_new).reshape(-1,1)
    I = np.array(model_I_new).reshape(-1,1)
    new_model_data = np.hstack((q, I))
    return new_model_data

def evaluate_obj_func(weights, models, exp_data): #This will be our objective function 
    models_copy = models.copy()
    for i in range(len(weights)-1):
        models_copy[:,i] = weights[i]*models[:,i]
    model_avg = np.mean(models_copy, axis=1)*weights[-1] #*1e-8
    #Normalize
    #model_avg = model_avg/model_avg[0]
    #exp_data = exp_data/exp_data[0]    
    # Get error
    error = np.mean(np.abs(np.log10(exp_data) - np.log10(model_avg+1e-20)))
    return -error, model_avg

def evaluate_obj_func_loop(x, models, exp_data):
    for i in range(x.shape[0]): # This is a loop that runs the test function for each row of an array
        y_row, _ = evaluate_obj_func(x[i,:], models, exp_data)
        if i == 0:
            y = y_row
        else:
            y = np.vstack((y, y_row))
    return y

def overall_max(scores): # This creates a list of the cumulative best scores from the scores in a list
    all_max = [scores[0]]
    for i in range(len(scores)):
        if scores[i] > all_max[i]:
            all_max.append(scores[i])
        else:
            all_max.append(all_max[i])
    return all_max

def invariant(data):
    q = data[:,0]
    I = data[:,1]
    invariant = scipy.integrate.simps(q**2*I, q)
    return invariant

def run_optimization(exp_data, models, batch_size, mutation_rate, iterations):
    x = np.random.rand(batch_size, models.shape[1]+1)
    y = evaluate_obj_func_loop(x, models, exp_data)
    alg = GA.genetic_algorithm(batch_size, mutation_rate)
    for i in range(iterations):
        x = alg.run(x,y)
        y = evaluate_obj_func_loop(x, models, exp_data)
    return alg.best_solution()
        