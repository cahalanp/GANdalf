#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script Name: calculate_bispectrum.py
Author: Paddy Cahalane
Date: 26/8/24

Description
-----------
This script calculates the bispectrum coefficients for molecules from a dataset of XYZ files.
The coefficients are saved as `.npy` files for further analysis. The script allows for the
selection of different molecules and parameters such as cutoff radii and the bispectrum twojmax.

Usage
-----
Run the script from the command line with the following optional arguments:
    
    -n : Specifies the molecule index (default is 5).
    -rc : Cutoff radius value for all species (default is 1.2).
    -j : Value for twojmax parameter, controls the size of bispectrum (default is 8).
    -s : The name of the directory to choose - one of ["train", "test", "val"].
    -x : Indices of which compiled xyz file to compute bispectrum with
    -u : Manually choose dataset dorectory and file name - overrides args.s

Dependencies
------------
- Python 3.x
- NumPy
- cobe (for dataset handling and bispectrum calculation)
- config.py module (in the parent directory)
- utils.py module (for various utilities)

Examples
--------
To calculate the bispectrum for the fifth molecule type with the default settings:
$ python calculate_bispectrum.py -n 5 -rc 1.2 -j 8 -s "train" -x 1
""" 

import os
import time
import argparse
import numpy as np
from typing import List

from modules import config
from gan import utils_e as utils
from gan import utils_plot

from cobe import Dataset, Datadict
from cobe.descriptors import Bispectrum

def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns
    -------
    argparse.Namespace
        A namespace containing the parsed arguments specifying molecule type, cutoff radius, and two_j_max (num of features per sample)
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, help='What molecule?', default=9)
    parser.add_argument('-rc', type=float, help='rc value for all species', default=1.8) #3.0
    parser.add_argument('-j', type=int, help='twojmax=8 -> ||B||=55', default=8)
    parser.add_argument('-s', type=str, help='set_type', default="train")
    parser.add_argument('-x', type=int, help='Which compiled xyz file to compute on', default=0)
    parser.add_argument('-u', type=int, help='Manually choose dataset - overrides args.s', default=1)
    return parser.parse_args()

def select_dataset_directory(load_path: str, set_type: str = "train", user_flag: bool = False) -> str:
    """
    Allows the user to select a dataset directory from available options.

    Parameters
    ----------
    load_path
        The path where the dataset directories are located.
    set_type
        The name of the directory to choose - one of ["train", "test", "val"].
    user_flag
        Whether to override {set_type} ad manually query user for the directory.

    Returns
    -------
    str
        The name of the selected dataset directory.
    """

    dirs = [dr for dr in os.listdir(load_path) if os.path.isdir(load_path+dr)]
    # if user_flag then manually select directory 
    if user_flag:
        print('\nPrinting all Directories in Path:', load_path, '\n')
        for i, dr in enumerate(dirs):
            print(' {}) {}'.format(i, dr))
        chosen_dir_ind = int(input('\nChoose Directory index:'))
    # otherwise chose directory with 
    else:
        chosen_dir_ind = dirs.index(set_type)
        
    chosen_dir = dirs[chosen_dir_ind]
    return chosen_dir

def select_xyz_file(load_path: str, which_xyz: int = 0, user_flag: bool = False) -> str:  
    """
    Allows the user to select an XYZ file from the chosen directory.

    Parameters
    ----------
    load_path 
        The path where the XYZ files are located.
    which_xyz 
        Which .xyz file to choose to calulate bispectrum on, listed in order of increasing indices.
    user_flag 
        Whether to override which_xyz and manually query user to choose file.

    Returns
    -------
    str
        The name of the selected XYZ file without extension.
    """
 
    # Sort files in load dir
    fls = os.listdir(load_path)
    if len(fls) > 1:
        fls.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        fls = [f for f in fls if f[0:4] != "enes"]

    if user_flag:
        print('\nPrinting all Files in Path:', load_path, '\n')
        for i, fl in enumerate(fls):
            if fl[0] != 'e':
                print(i,')',fl)
        # User Input Choose File 
        chosen_file = int(input('\nChoose file index:'))
        file_desc = fls[chosen_file][:-4]
    else:
        file_desc = fls[which_xyz][:-4]
        
    return file_desc

def load_dataset(load_path: str, file_desc: str) -> Dataset:
    """
    Loads the XYZ data into a cobe dataset object.

    Parameters
    ----------
    load_path 
        The path to the XYZ file.
    file_desc 
        The file descriptor (name without extension).

    Returns
    -------
    Dataset
        The COBE dataset loaded with XYZ data (and optionally energies).
    """

    # Load XYZ coords into Cobe Dataset Object
    data = Dataset(os.path.join(load_path, file_desc + ".xyz"))
  
    # Set Energies
    energy_file = 'enes_' + file_desc + ".xyz"
    data.set_energies(os.path.join(load_path, file_desc + ".xyz"), os.path.join(load_path, energy_file))
  
    return data    
    
def plot_cutoff_radius(
    data: Dataset, 
    new_types: List[int], 
    rc: List[float], 
    rcutfac: float
) -> None:
    """
    Plots a 2D projection of the molecule, highlighting the cutoff radius around an atom of each species.

    Parameters
    ----------
    data 
        The COBE dataset containing XYZ data.
    new_types 
        A list of species types (integers) present in the molecule.
    rc 
        A list of cutoff radii corresponding to each species type.
    rcutfac 
        A scaling factor applied to the cutoff radii.

    Returns
    -------
    None
        This function does not return a value. It exits the script after plotting.
    """

    rc_plot = {new_types[i]: rc_i*rcutfac for i, rc_i in enumerate(rc)} # {0:1.23,1:1.9}
    label = utils.convert_labels(data.trajectory[0].get_chemical_symbols())
    positions = data.trajectory[0].get_positions()
    utils_plot.plot_rc(positions, label, rc_plot)

def calculate_bispectrum(
    data: Dataset, 
    rc: List[float], 
    rcutfac: float, 
    rfac0: float, 
    twojmax: int
) -> Datadict:
    """
    Calculates the bispectrum for a given dataset.

    Parameters
    ----------
    data
        The COBE dataset containing XYZ data.
    rc 
        The cutoff radii for the bispectrum calculation.
    rcutfac
        Scaling factor for the cutoff radius (default is 1.5).
    rfac0
        The value of rfac0 used in bispectrum calculation (default is 0.99).
    twojmax
        The twojmax parameter for bispectrum calculation (default is 8).

    Returns
    -------
    Datadict
        The calculated bispectrum for the dataset contained in a cobe Datadict.
    """

    print('\nCalculating B for all molecules:')
    start_time = time.time()
    data_d = Bispectrum(data, 
                        rcutfac=rcutfac,
                        rfac0=rfac0,
                        twojmax=twojmax,
                        rc=rc,
                        pbc_flag=True,
                       )

    print('\nTime= {}'.format(time.time() - start_time))

    return data_d
    
def save_bispectrum_results(
    save_path: str, 
    file_desc: str, 
    data: Dataset, 
    data_d: Datadict, 
    info: dict
) -> None:
    """
    Saves the bispectrum components, atomic labels, XYZ coordinates, and energies as `.npy` files.

    Parameters
    ----------
    save_path
        The path where the results will be saved.
    file_desc
        The file descriptor (name without extension).
    data 
        The original cobe dataset containing the XYZ coordinates.
    data_d
        The bispectrum data datadict object.
    info
        Dictionary containing metadata about the dataset.

    Returns
    -------
    None
    """

    # Convert labels from [1,6] -> [0,1]
    labels = []
    for i in range(len(data_d.data['types'])):
        labels.append(utils.convert_labels(data_d.data['types'][i]))

    # Get XYZs as array
    XYZs = np.array([data.trajectory[i].get_positions() for i in range(data.n_molecules)])
    
    # Save B, L, info, and E as npy files
    np.save(os.path.join(save_path, 'info_'+file_desc), info)
    np.save(os.path.join(save_path, 'B_'+file_desc), data_d.data['descriptors'])
    np.save(os.path.join(save_path, 'X_'+file_desc), XYZs)
    np.save(os.path.join(save_path, 'E_'+file_desc), data_d.data['energies'])
    np.save(os.path.join(save_path, 'L_'+file_desc), labels)

    print('\nSuccessfully Saved to Path: \n' + save_path + '\n')
    print(' 1) {} \n 2) {} \n 3) {} \n 4) {} \n 5) {}'.format('info_'+file_desc, 'B_'+file_desc, 'X_'+file_desc, 'E_'+file_desc, 'L_'+file_desc))

def main() -> None:
    """
    Main function to execute the script. Parses arguments, selects dataset, calculates the bispectrum, and saves the results.
    """
    
    # Argparser and paths
    args = parse_arguments()
    mol_name = config.MOL_LIST[args.n]
    load_path = os.path.join(config.BASE_PATH, 'data/xyz/{}/'.format(mol_name))
    save_path = os.path.join(config.BASE_PATH, 'data/bispec/{}/'.format(mol_name))
    
    # Print chosen parameters
    print('\nSelected Molecule: {}'.format(mol_name))
    print('Selected RC: {}'.format(args.rc))
    print('Selected two_j_max: {}'.format(args.j))
    
    set_type = select_dataset_directory(load_path, args.s, args.u)
    
    # Setting final load and save paths
    load_path = os.path.join(load_path, set_type)
    save_path = os.path.join(save_path, set_type)
    print('\nLoading From: {}\nSaving To: {}'.format(load_path, save_path), '\n')
    
    # Make new save dir if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # choose which compile .xyz file on which to compute bispectrum components
    file_desc = select_xyz_file(load_path, args.x, args.u)
    
    # load mols as cobe dataset
    data = load_dataset(load_path, file_desc)
    
    # Define Parameters for LAMMPS calculation
    twojmax = args.j
    rcutfac = config.RCUTFAC
    rfac0 = config.RFAC0
    rc = [args.rc for _ in range(len(data.types))]
    new_types = utils.convert_labels(data.types, force_convert=True) 
    rc_dict = {new_types[i]: rc_i for i, rc_i in enumerate(rc)} # dict w/ species as keys and RC's as values
    
    # OPTIONAL - Plot Rc then Exit
    plot_flag = 1
    if plot_flag:
        plot_cutoff_radius(data, new_types, rc, rcutfac)
    
    # Perform LAMMPS calculation
    data_d = calculate_bispectrum(data, rc, rcutfac, rfac0, twojmax)

    # Create Info Dict about Dataset
    info = {'N_mols': data_d.data['n_molecules'],
            'N_at': np.array(data_d.data['types']).shape[1], 
            'N_feats': data_d.data['n_descriptors'],
            'N_species': len(data_d.data['all_types']),
            'set_name': config.MOL_LIST[args.n], 
            'atomic_labels': utils.convert_labels(data_d.data['types'][0]),
            'species': utils.convert_labels(data_d.data['all_types']),
            'rcutfac':rcutfac,
            'rc_dict':rc_dict,
           }

    print('\nInfo Dict:')
    for i in info:
        print('%s:' %i, info[i])
    
    save_bispectrum_results(save_path, file_desc, data, data_d, info)
    
if __name__ == "__main__":
    main()
