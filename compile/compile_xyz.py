#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script Name: compile_xyz.py
Author: Paddy Cahalane
Date: 26/8/24

Description
-----------
This script processes a specified .xyz file containing molecular data, splits it into 
training, testing, and validation datasets, and saves them into separate .xyz files.

The data is split into smaller .xyz files to avoid memory issues when computing the 
bispectrum components representation using LAMMPS in compute_b.py. 

Usage
-----
Run the script from the command line with the following optional arguments:
    -n : Specifies the molecule type (e.g., benzene=2).
    -d : Specifies which dataset to process (default is 1).
    -u : Manually select the dataset file (overrides the -d argument).
    -p : Number of molecules per output .xyz file (default is 50000).
    -s : Whether to shuffle molecules before splitting (default is True).

Dependencies
------------
- Python 3.x
- NumPy
- config.py module (in the parent directory)

Examples
--------
To process benzene(n=2), shuffle the data, and save 50,000 molecules per file:
$ python split_xyz.py -n 2 -sh True -p 50000
"""

import os
import shutil
import argparse
import numpy as np
from random import shuffle
from typing import List, Tuple

#sys.path.append('../modules')

from modules import config

def parse_arguments() -> argparse.Namespace:

    """
    Parse command-line arguments for splitting .xyz files into train, test, and validation datasets.

    Returns
    -------
    argparse.Namespace
        The parsed arguments including molecule type, dataset index, manual user override, number of molecules per .xyz file, and shuffle flag.
    """

    parser = argparse.ArgumentParser(description="Split specified .xyz file into train/test/val .xyz files")
    parser.add_argument('-n', type=int, help='Which molecule type e.g. benzene=2', default=6)
    parser.add_argument('-d', type=int, help='Which molecule dataset', default=1)
    parser.add_argument('-u', type=bool, help='Manually choose dataset - overrides args.d', default=0)
    parser.add_argument('-p', type=int, help='Number of mols per .xyz file', default=50000)
    parser.add_argument('-s', type=bool, help='Shuffle molecules?', default=1)
    return parser.parse_args()

def user_select_dataset(possible_datasets: List[str]) -> str:
    """
    Allow the user to manually select a dataset file from a list of possible datasets.

    Parameters
    ----------
    possible_datasets
        List of available .xyz files.

    Returns
    -------
    str
        The name of the selected dataset file.
    """

    print('\n----------------------------------')
    print('All Raw .XYZ Files in Directory:')
    for i, fl in enumerate(possible_datasets):
        print('{}) {}'.format(i, fl))
    chosen_file = int(input('\nChoose file index:'))
    dataset_name = possible_datasets[chosen_file]
    return dataset_name

def load_data(file_path: str) -> List[str]:
    """
    Load and read data from the specified .xyz file.

    Parameters
    ----------
    file_path : str
        The path to the .xyz file.

    Returns
    -------
    List[str]
        The content of the .xyz file split into lines.
    """

    with open(file_path) as f:
        data = f.read().split('\n')
    return data

def get_molecule_indices(data: List[str], N_at: int) -> Tuple[np.ndarray, int]:
    """
    Compute the indices for each molecule in the dataset and determine the total number of molecules.

    Parameters
    ----------
    data
        The content of the .xyz file split into lines.
    N_at
        The number of atoms per molecule.

    Returns
    -------
    np.ndarray
        Array of indices for each molecule.
    int
        Total number of molecules in the dataset.
    """

    L = len(data)
    N_mol = L // (N_at+2)
    all_inds = np.arange(0, L-(N_at+2), N_at+2)
    return all_inds, N_mol

def split_data_indices(all_inds: np.ndarray, N_mol: int, shuffle_flag: bool = True) -> dict:
    """
    Split the molecule indices into training, testing, and validation sets.

    Parameters
    ----------
    all_inds
        Array of indices for each molecule.
    N_mol
        Total number of molecules.
    shuffle_flag
        Whether to shuffle the indices before splitting (default is True).

    Returns
    -------
    data_split
        A dictionary with keys 'train', 'test', and 'val', mapping to lists of indices for each set.
    """

    if shuffle_flag:
        shuffle(all_inds)

    # Find train/test/val sets sizes
    N_train = int(N_mol * 0.85) 
    N_test = int(N_mol * 0.10) 
    N_val = int(N_mol * 0.05)

    data_split = {"train": all_inds[:N_train],
                  "test": all_inds[N_train:N_train+N_test],
                  "val": all_inds[N_train+N_test:]
                 }

    return data_split

def save_molecule_files(
    indices: List[str], 
    data: List[str], 
    N_at: int, 
    save_path: str, 
    chunk_size: int
) -> None:
    """
    Save molecule data into .xyz files for a specific dataset split (train, test, val).

    Parameters
    ----------
    indices
        List of indices for the current dataset split.
    data
        The content of the .xyz file split into lines.
    N_at
        The number of atoms per molecule.
    save_path
        The directory path to save the output .xyz files.
    chunk_size
        Number of molecules per output file.
        
    Returns
    -------
    None
    """

    print('\nSaving to Path:', save_path)

    # for each xyz file to be created
    for i_file in range(0, len(indices), chunk_size):

        # select relevant lines in big .xyz based on start/end indices
        file_inds = indices[i_file:i_file+chunk_size]

        name_ene = 'enes_{}_{}.xyz'.format(i_file, i_file+len(file_inds)) # energy file of just energies
        name_xyz = '{}_{}.xyz'.format(i_file, i_file+len(file_inds)) # .xyz file of specified size (almost same as big .xyz file)

        with open(os.path.join(save_path, name_ene),'w') as ene_file, open(os.path.join(save_path, name_xyz),'w') as xyz_file:
            # for the starting index of each mol
            for ind in file_inds:
                ene_file.write('{}\n'.format(data[ind+1]))
                xyz_file.write('{}\n{}\n'.format(data[ind],data[ind+1]))

                # list of xyz coords of each mol
                mol_data = data[2+ind:2+ind+N_at] 

                for line in mol_data:
                    xyz_file.write(line + "\n")

        print(' {}) {} \n {}) {}'.format(i_file, name_ene, i_file, name_xyz))

    print('\nFinished Saving Files')

def main() -> None:
    """
    Main function to handle the entire process of selecting a dataset, loading data, splitting it, and saving it to files.
    """
    
    # get paths
    args = parse_arguments()
    mol_name = config.MOL_LIST[args.n] 
    load_path = os.path.join(config.BASE_PATH, 'data/xyz/{}/'.format(mol_name))

    # select .xyz dataset
    poss_dataset = [file for file in os.listdir(load_path) if os.path.isfile(load_path+file)]
    poss_dataset = [f for f in poss_dataset if f[-4:] == ".xyz"]
    if args.u:
        dataset_name = user_select_dataset(poss_dataset)
    else:
        dataset_name = poss_dataset[args.d]

    print('\nSelected Molecule Type: {} \nChoose File: {} \nLoading From: {}\nSaving To: {}'.format(mol_name, dataset_name, load_path, load_path+'{set_type}'))

    data = load_data(os.path.join(load_path, dataset_name))
    N_at = int(data[0])
    all_inds, N_mol = get_molecule_indices(data, N_at)

    # Get mol indices to be in each set_type 
    indices_split = split_data_indices(all_inds, N_mol, shuffle_flag=args.s)

    # - Loop over Train/Test/Val
    #     - Create Save Directory
    #        - Loop over Number of Files per Train/Test/Val Set
    #             - Create Energy File
    #             - Create XYZ File

    for set_type in ['train','test','val']:
        # path to save to - delete if there is an existing one
        save_path = os.path.join(load_path, set_type)
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.makedirs(save_path)

        save_molecule_files(indices_split[set_type], data, N_at, save_path, args.p)

if __name__ == "__main__":
    main()



