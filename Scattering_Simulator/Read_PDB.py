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

# Using SLD values from periodic table for scattering 
# https://ncnr.nist.gov/instruments/magik/Periodic.html

def atom_to_sld_SANS(atom):
    '''These values are for neutrons'''
    sld = []
    for i in range(len(atom)):
        if atom[i][0] == 'H':
            sld.append(3.28e-6)
        elif atom[i][0] == 'C':
            sld.append(3.28e-6)
        elif atom[i][0] == 'N':
            sld.append(3.28e-6)
        elif atom[i][0] == 'O':
            sld.append(3.28e-6)
        elif atom[i][0] == 'P':
            sld.append(3.28e-6)
        elif atom[i][0] == 'S':
            sld.append(3.28e-6)
        else:
            if atom[i][1] == 'H':
                sld.append(3.28e-6)
            elif atom[i][1] == 'C':
                sld.append(3.28e-6)
            elif atom[i][1] == 'N':
                sld.append(3.28e-6)
            elif atom[i][1] == 'O':
                sld.append(3.28e-6)
            elif atom[i][1] == 'P':
                sld.append(3.28e-6)
            elif atom[i][1] == 'S':
                sld.append(3.28e-6)
    sld = np.array(sld)    
    sld_D2O = 6.35e-6
    sld = np.array(sld) - sld_D2O
    return sld


def atom_to_sld(atom):
    '''These values are for X-rays'''
    sld = []
    for i in range(len(atom)):
        if atom[i][0] == 'H':
            sld.append(1.19e-6)
        elif atom[i][0] == 'C':
            sld.append(17.86e-6)
        elif atom[i][0] == 'N':
            sld.append(6.88e-6)
        elif atom[i][0] == 'O':
            sld.append(9.73e-6)
        elif atom[i][0] == 'P':
            sld.append(15.26e-6)
        elif atom[i][0] == 'S':
            sld.append(17.90e-6)
        else:
            if atom[i][1] == 'H':
                sld.append(1.19e-6)
            elif atom[i][1] == 'C':
                sld.append(18.71e-6)
            elif atom[i][1] == 'N':
                sld.append(6.88e-6)
            elif atom[i][1] == 'O':
                sld.append(9.73e-6)
            elif atom[i][1] == 'P':
                sld.append(15.26e-6)
            elif atom[i][1] == 'S':
                sld.append(17.90e-6)
    sld = np.array(sld)    
    sld_water = 9.46e-6
    sld = np.array(sld) - sld_water
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


def load_pdb_SANS(filename):
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
    sld = atom_to_sld_SANS(atom)
    coordinates = np.hstack((x_pos.reshape(-1,1), y_pos.reshape(-1,1), z_pos.reshape(-1,1), sld.reshape(-1,1)))
    return coordinates


def export_PDB(coordinates, dir):
    '''Creates a PDB style file in a txt format'''
    length = len(coordinates)
    coordinates = np.round(coordinates, 2)
    col1 = np.array(['ATOM     ']*length).reshape(-1,1)
    col2 = np.array(np.arange(1, length+1, 1)).reshape(-1,1)
    col3 = np.array([' O']*length).reshape(-1,1)
    col4 = np.array(['  SER']*length).reshape(-1,1)
    col5 = np.array([ 'A  ']*length).reshape(-1,1)
    col6 = np.array([1]*length).reshape(-1,1)
    col7 = np.array(['   ']*length).reshape(-1,1)
    col8 = np.array(coordinates[:,0]).reshape(-1,1)
    col9 = np.array(coordinates[:,1]).reshape(-1,1)
    col10 = np.array(['   ']*length).reshape(-1,1)
    col11 = np.array(coordinates[:,2]).reshape(-1,1)
    col12 = np.array([' 1.00 ']*length).reshape(-1,1)
    col13 = np.array(['0.00        ']*length).reshape(-1,1)
    file = np.hstack((col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col3))
    np.savetxt(dir, file,  fmt="%s") #save merged data as npy 
    return file


def read_DAT_file(name):
    '''reads DAT files which are files from the Xenocs SAXS instrument
    input: the file path
    output: an array with q, I, and dI as the columns'''

    with open(name) as pdbfile:
        q = []
        I = []
        dI = []
        dq = []
        start = 10000
        for i,line in enumerate(pdbfile):
            if 'q(A-1)' in line:
                start = i
            if i > start: 
                splitted_line = [line[0:15], line[15:40], line[45:100], line[39:]]
                try:
                    q.append(float(splitted_line[0]))
                    I.append(float(splitted_line[1]))
                    dI.append(float(splitted_line[2]))
                    dq.append(float(splitted_line[3]))
                    
                    #float(splitted_line[2])
                    #q.append(splitted_line[0])
                    #I.append(splitted_line[1])
                    #dI.append(splitted_line[2]) 
                except:
                    a = 1
        q = np.array([float(i) for i in q]).reshape(-1,1)
        I = np.array([float(i) for i in I]).reshape(-1,1)
        dI = np.array([float(i) for i in dI]).reshape(-1,1)
        #dq = np.array([float(i) for i in dq]).reshape(-1,1)
        data = np.hstack((q, I, dI))
    return data
