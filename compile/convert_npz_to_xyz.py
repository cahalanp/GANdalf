#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script Name: convert_npz_to_xyz.py
Author: Paddy Cahalane
Date: 26/8/24

Description
-----------
This script converts molecular data stored in a `.npz` file into an `.xyz` format.
It handles molecular coordinates, forces, energies, and nuclear charges, and writes them
to an `.xyz` file that can be used for further analysis in computational chemistry.

Usage
-----
Run the script from the command line with the following optional arguments:
    -n : Specifies the molecule type index (from module config).

Dependencies
------------
- Mendeleev
- Python 3.x
- NumPy
- config.py module (in the parent directory)

Examples
--------
To convert a dataset with the sixth molecule type and save the output to an `.xyz` file:
$ python convert_npz_to_xyz.py -n 6
"""
   
import os
import sys
import shutil
import argparse
import numpy as np

from mendeleev import element

#sys.path.append('../modules')
#sys.path.append('../gan')

from modules import config

def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns
    -------
    argparse.Namespace
        A namespace containing the parsed arguement n specifying molecule type to convert (from config)
    """

    parser = argparse.ArgumentParser(description="Convert .npz data to .xyz data")
    parser.add_argument('-n', type=int, help='Molecule index', default=6)
    return parser.parse_args()
   
def load_data(load_path: str) -> str:
    """
    Allows the user to manually select a dataset from available `.npz` files.

    Parameters
    ----------
    load_path
        The path where the `.npz` files are located.

    Returns
    -------
    str
        The name of the selected `.npz` file.
    """

    fls = [file for file in os.listdir(load_path) if os.path.isfile(load_path+file)]
    dataset_name = [f for f in fls if f.endswith(".npz")][0] # choose the first dataset name with .npz file type
    return dataset_name

def convert_npz_to_xyz(dataset_name: str, load_path: str, save_path: str) -> None:
    """
    Converts a `.npz` file containing molecular data to an `.xyz` file format.

    Parameters
    ----------
    dataset_name
        The name of the `.npz` file to be converted.
    load_path
        The path where the `.npz` file is located.
    save_path
        The path where the `.xyz` file will be saved.
    """

    # change name from *.npz to *.xyz
    new_dataset_name = dataset_name[:-4] + ".xyz"

    f = np.load(os.path.join(load_path, dataset_name))

    L = f["nuclear_charges"]
    X = f["coords"]
    F = f["forces"]
    E = f["energies"]

    # convert labels from nuc charge to atomic symbols
    print("L",L)
    L_new = [element(int(l)).symbol for l in L]

    n_at = len(L_new)
    n_mol = len(E)
    with open(os.path.join(save_path, new_dataset_name), "w+") as f_out:
        for i_mol in range(n_mol):
            f_out.write('{}\n'.format(n_at))
            f_out.write('{}\n'.format(E[i_mol]))

            for i_at in range(n_at):
                atom = L_new[i_at]
                (x, y, z) = X[i_mol][i_at]
                (fx, fy, fz) = F[i_mol][i_at]
                f_out.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(atom, x, y, z, fx, fy, fz))

def main() -> None:
    """
    Main function to execute the script. Parses arguments, selects dataset, and converts it.
    """
    
    args = parse_arguments()
    mol_name = config.MOL_LIST[args.n] 
    load_path = os.path.join(config.BASE_PATH, 'data/xyz/{}/'.format(mol_name))
    save_path = load_path

    # select .xyz dataset
    dataset_name = load_data(load_path)
    new_dataset_name = dataset_name[:-4] + ".xyz"
    print('\nSelected Molecule Type: {} \nChoose File: {} \nLoading From: {}\nSaving To: {}'.format(mol_name, dataset_name, load_path, save_path))

    # convert
    convert_npz_to_xyz(dataset_name, load_path, save_path) 
    print('Saved {} To {}'.format(new_dataset_name, save_path))

if __name__ == "__main__":
    main()
   
   
