import numpy as np
import sys
import os

def open_file(name):
    with open(name) as pdbfile:
        atom = []
        x_pos = []
        y_pos = []
        z_pos = []
        for line in pdbfile:
            if line[:4] == 'ATOM' or line[:6] == "HETATM":
                #print(line)
                # Split the line
                splitted_line = [line[:6], line[6:11], line[12:16], line[17:20], line[21], line[22:26], line[30:38], line[38:46], line[46:54]]
                #print(splitted_line)
                atom.append(splitted_line[2])
                x_pos.append(splitted_line[-3])
                y_pos.append(splitted_line[-2])
                z_pos.append(splitted_line[-1])
    return atom, x_pos, y_pos, z_pos

def remove_spaces(lst):
    new_lst = []
    for i in range(len(lst)):
        new_string = lst[i].strip()
        new_lst.append(new_string)
    return new_lst

def convert_str_to_float(lst):
    array = np.array([float(i) for i in lst])
    return array 

def atom_to_sld(atom):
    sld = []
    for i in range(len(atom)):
        if atom[i][0] == 'H':
            sld.append(1.68e-5)
        elif atom[i][0] == 'C':
            sld.append(8.50e-6)
        elif atom[i][0] == 'N':
            sld.append(8.52e-6)
        elif atom[i][0] == 'O':
            sld.append(8.54e-6)
        elif atom[i][0] == 'P':
            sld.append(8.38e-6)
        elif atom[i][0] == 'S':
            sld.append(8.64e-6)
    sld = np.array(sld)
    return sld

def load_pdb(filename):
    '''Returns an array where the first 3 columns contain the x,y,z coordinates of the atoms, and the last column contains
       the SLD of the atoms'''
    atom, x_pos, y_pos, z_pos = open_file(filename)
    atom = remove_spaces(atom)
    x_pos = remove_spaces(x_pos)
    y_pos = remove_spaces(y_pos)
    z_pos = remove_spaces(z_pos)
    x_pos = convert_str_to_float(x_pos)
    y_pos = convert_str_to_float(y_pos)
    z_pos = convert_str_to_float(z_pos)
    sld = atom_to_sld(atom)
    coordinates = np.hstack((x_pos.reshape(-1,1), y_pos.reshape(-1,1), z_pos.reshape(-1,1), sld.reshape(-1,1)))
    return coordinates