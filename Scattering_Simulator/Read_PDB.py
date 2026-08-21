import numpy as np
import sys
import os

def open_file(name, first_model_only=True):
    '''Parses ATOM/HETATM records from a PDB file using fixed-column positions.

    Robustness features:
    - reads the element symbol from columns 77-78 (falling back to the atom name)
      so every atom is identified, including metals and other heteroatoms;
    - keeps only the primary alternate location (blank or "A") to avoid double
      counting atoms with multiple conformations;
    - reads only the first MODEL by default (so NMR ensembles are not duplicated);
      pass first_model_only=False to read every model, e.g. for a biological
      assembly file (.pdb1) where each model is a symmetry copy;
    - skips malformed lines instead of raising.

    Returns:
    - elements: list of element symbols (upper-case) for each atom
    - coords: (N, 3) float array of x, y, z coordinates.'''
    elements = []
    coords = []
    model_count = 0
    with open(name) as pdbfile:
        for line in pdbfile:
            if line.startswith('MODEL'):
                model_count += 1
                continue
            if first_model_only and model_count > 1:
                continue
            if line[:4] == 'ATOM' or line[:6] == 'HETATM':
                # Keep only the primary alternate location.
                altloc = line[16] if len(line) > 16 else ' '
                if altloc not in (' ', 'A'):
                    continue
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except (ValueError, IndexError):
                    continue
                # Prefer the element column (77-78); fall back to the atom name.
                element = line[76:78].strip().upper() if len(line) >= 78 else ''
                if not element:
                    name_field = line[12:16].strip()
                    name_field = ''.join(ch for ch in name_field if not ch.isdigit())
                    element = name_field[:1].upper()
                elements.append(element)
                coords.append((x, y, z))
    return elements, np.array(coords, dtype=float)

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

# Per-atom scattering length densities keyed by element symbol.
# https://ncnr.nist.gov/instruments/magik/Periodic.html
_SANS_SLD = {'H': 3.28e-6, 'C': 3.28e-6, 'N': 3.28e-6,
             'O': 3.28e-6, 'P': 3.28e-6, 'S': 3.28e-6}
_XRAY_SLD = {'H': 1.19e-6, 'C': 17.86e-6, 'N': 6.88e-6,
             'O': 9.73e-6, 'P': 15.26e-6, 'S': 17.90e-6}


def atom_to_sld_SANS(elements):
    '''Neutron SLD for each element, minus the D2O background.

    Every atom gets a value: elements outside the standard set (metals, halogens,
    etc.) use a carbon-like fallback so the returned array always matches the
    number of atoms.'''
    fallback = _SANS_SLD['C']
    sld = np.array([_SANS_SLD.get(e, fallback) for e in elements])
    return sld - 6.35e-6


def atom_to_sld(elements):
    '''X-ray SLD for each element, minus the water background.

    Every atom gets a value: elements outside the standard set (metals, halogens,
    etc.) use a carbon-like fallback so the returned array always matches the
    number of atoms.'''
    fallback = _XRAY_SLD['C']
    sld = np.array([_XRAY_SLD.get(e, fallback) for e in elements])
    return sld - 9.46e-6



def download_pdb(pdb_id, dest_dir='.', assembly=False):
    '''Downloads a structure from the RCSB Protein Data Bank and returns the local file path.

    inputs:
    - pdb_id: 4-character PDB ID, e.g. "6lyz" (case-insensitive).
    - dest_dir: directory to save the file into (created if it does not exist).
    - assembly: if True, download the first biological assembly (.pdb1) instead of
      the asymmetric unit (.pdb).

    output:
    - path to the downloaded .pdb file.'''
    import urllib.request

    pdb_id = pdb_id.strip().lower()
    ext = 'pdb1' if assembly else 'pdb'
    url = f'https://files.rcsb.org/download/{pdb_id}.{ext}'
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, f'{pdb_id}.pdb')
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as e:
        raise RuntimeError(f"Could not download PDB '{pdb_id}' from {url}: {e}")
    return dest


def load_pdb(filename, first_model_only=True):
    '''Returns an array where the first 3 columns contain the x,y,z coordinates of the atoms,
       and the last column contains the X-ray SLD of the atoms.

       Pass first_model_only=False to load every model (e.g. a biological assembly, .pdb1).'''
    elements, coords = open_file(filename, first_model_only=first_model_only)
    if len(elements) == 0:
        raise RuntimeError(f"No atoms could be parsed from '{filename}'.")
    sld = atom_to_sld(elements)
    return np.hstack((coords, sld.reshape(-1, 1)))


def load_pdb_SANS(filename, first_model_only=True):
    '''Returns an array where the first 3 columns contain the x,y,z coordinates of the atoms,
       and the last column contains the neutron SLD of the atoms.

       Pass first_model_only=False to load every model (e.g. a biological assembly, .pdb1).'''
    elements, coords = open_file(filename, first_model_only=first_model_only)
    if len(elements) == 0:
        raise RuntimeError(f"No atoms could be parsed from '{filename}'.")
    sld = atom_to_sld_SANS(elements)
    return np.hstack((coords, sld.reshape(-1, 1)))


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
