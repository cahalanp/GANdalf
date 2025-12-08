#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script Name: merge_b.py
Author: Paddy Cahalane
Date: 26/8/24

Description
-----------
This script merges all `.npy` files in a specified directory into larger `.npy` files to overcome 
memory limitations encountered when processing large datasets of molecules. It also calculates the most diverse
molecules from a random subset, using a SOAP-like kernel, to be used later for multiple inverions initialisations .

Usage
-----
Run the script from the command line with the following optional arguments:
    -n : Specifies the molecule index from the `mol_list` (default is 5).
    -p : Number of molecules to include in the final dataset. If -1, all molecules are included (default is -1).
    -s : The name of the directory to choose - one of ["train", "test", "val"].
    -u : Manually choose dataset dorectory and file name - overrides args.s
    -N : Number of random samples used for calculating the most diverse starting mols.
    -M : Number of most diverse starting mols to save.

Dependencies
------------
- Python 3.x
- NumPy
- Scikit-learn (for shuffling data)
- utils.py module (for utility functions)
- config.py module (for global constants)

Examples
--------
To merge `.npy` files for the fifth molecule type and include all molecules:
$ python merge_b.py -n 5 -p -1 -u 1 -N 5000 -M 20
"""

import os
import tqdm
import shutil
import argparse
import numpy as np
from typing import List, Tuple

from ase.io import write

from modules import config
from gan import utils_e as utils
 
def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments for molecule type, and number of final molecules.
    """

    parser = argparse.ArgumentParser(description="Merge molecular .npy files into large datasets.")
    parser.add_argument('-n', type=int, help='Molecule type index - as in list in config.py', default=6)
    parser.add_argument('-p', type=int, help="Number of molecules in final dataset (-1 for all)", default=-1)
    parser.add_argument('-s', type=str, help='Dataset type - one of ["train", "test", "val"] ', default="train")
    parser.add_argument('-u', type=int, help='Manually choose dataset - overrides args.s', default=0)
    parser.add_argument('-N', type=int, help='Number of samples for diverse starting mols', default=10000)
    parser.add_argument('-M', type=int, help='Number of most diverse starting mols', default=100)
    return parser.parse_args()

def select_dataset_directory(
    load_path: str, 
    set_type: str = "train", 
    user_flag: bool = False
) -> str:
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

    dirs = [dr for dr in os.listdir(load_path) if os.path.isdir(os.path.join(load_path, dr))]

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

def load_data(load_path: str) -> Tuple[dict, List[dict]]:
    """
    Loads the data from the specified directory.

    Parameters
    ----------
    load_path : str
        The path from which to load the `.npy` files.

    Returns
    -------
    dict
        Dictionary containing descriptors, atomic positions, energies, and species labels as lists of ndarrays, one array for each molecule type.
    List[dict]
        List of dictionaries containing all relevant information about the datasets.
    """

    data_ds = {"Bs": [], "Ls": [], "Xs": [], "Es": [],}
    infos = []
    fls = os.listdir(load_path)

    # Remove Directories and Ideal files
    fls = [fl for fl in fls if not os.path.isdir(os.path.join(load_path, fl))]

    # Sort Files to Correct Order
    fls.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    print('\nNumber of Files in Dir:' + str(len(fls)))
    print('Files in Dir:')

    # Load data from fls into lists in data_ds
    for i, fl in enumerate(fls):
        if fl[0] == 'B':
            B_i = np.load(os.path.join(load_path, fl))
            data_ds["Bs"].append(B_i)
            print('{}) {} - {}'.format(i, fl, B_i.shape))
        if fl[0] == 'L':
            data_ds["Ls"].append(np.load(os.path.join(load_path, fl)))
        if fl[0] == 'X':
            X_i = np.load(os.path.join(load_path, fl))
            data_ds["Xs"].append(X_i)
            print('{}) {} - {}'.format(i, fl, X_i.shape))
        if fl[0] == 'E':
            data_ds["Es"].append(np.load(os.path.join(load_path, fl)))
            print('{}) {}'.format(i, fl))
        if fl[0] == 'i':
            infos.append(np.load(os.path.join(load_path, fl), allow_pickle=True).item())

    return data_ds, infos

def concatenate_data(data_ds: dict, infos: List[dict], num_mols: int) -> Tuple[dict, dict]:
    """
    Concatenates data arrays.

    Parameters
    ----------
    dict
        Dictionary containing data in lists of arrays.
    infos : list of dict
        Information dictionaries.
    num_mols : int
        Number of molecules to include in the final dataset.

    Returns
    -------
    data_d
        Dictionary of concatenated data arrays.
    dict
        Updated information dictionary.
    """

    # Concatenate lists into big arrays
    B_all = np.concatenate((data_ds["Bs"]))
    L_all = np.concatenate((data_ds["Ls"]))
    X_all = np.concatenate((data_ds["Xs"]))
    E_all = np.concatenate((data_ds["Es"]))
    info_all = infos[0]
    
    num_mols = num_mols if num_mols != -1 else B_all.shape[0]
    info_all['N_mols'] = num_mols

    print('\nTotal Num Molecules:', num_mols)    
    print('Unique Species in L:', np.unique(L_all))

    data_d = {}
    data_d["B"] = B_all[:num_mols]
    data_d["L"] = L_all[:num_mols]
    data_d["X"] = X_all[:num_mols]
    data_d["E"] = E_all[:num_mols]

    return data_d, info_all
    
def save_data(
    save_path: str, 
    data_d: dict, 
    info_all: dict, 
    num_mols: int
) -> None:
    """
    Saves the data arrays and information dictionary to the specified path.

    Parameters
    ----------
    save_path
        The path where the data will be saved.
    data_d
        Dictionary containing all data to be saved. 
    info_all 
        Updated information dictionary.
    num_mols
        Number of molecules to include in the final dataset.
        
    Returns
    -------
    None
    """       

    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    # Print info
    print('\n')    
    for i in info_all:
        print('%s:' %i, info_all[i])

    # Save arrays
    np.save(os.path.join(save_path, str(num_mols) + '_info'), info_all)
    np.save(os.path.join(save_path, str(num_mols) + '_B'), data_d["B"])
    np.save(os.path.join(save_path, str(num_mols) + '_L'), data_d["L"])
    np.save(os.path.join(save_path, str(num_mols) + '_X'), data_d["X"])
    np.save(os.path.join(save_path, str(num_mols) + '_E'), data_d["E"])

    # Save info as .txt
    print('\nSaving Info as .txt')
    with open(os.path.join(save_path, str(num_mols)+'_info.txt'), 'w') as f:
        print(info_all, file=f)

    print('\nSuccessfully Saved to Path: \n' + save_path)

    print('\n 1) {}.npy \n 2) {}.npy \n 3) {}.npy \n 4) {}.npy \n 5) {}.npy \n'.format(str(num_mols) + '_info', str(num_mols) + '_B', str(num_mols) + '_L', str(num_mols) + '_X', str(num_mols) + '_E'))

def starting_mols(
    save_path: str,
    data_d: dict,
    N_samples: int,
    N_start: int,
    gamma: int = 4
) -> None:
    """
    Selects the most diverse molecules from a subset for each molecule type.

    Parameters
    ----------
    save_path
        The path where the data will be saved.
    data_d
        Dictionary containing all data.
    N_samples 
        Number of molecules to consider when selecting the diverse subset - big affect on computation times. 
    N_start
        Number of molecules to include in the diverse subset.
    gamma
        Scaling factor 
        
    Returns
    -------
    None
    """       

    # make save path dir
    os.makedirs(save_path, exist_ok=True)

    # load from data/bipsec
    rand_inds = np.arange(len(data_d["B"]))
    np.random.shuffle(rand_inds)
    rand_inds = rand_inds[:N_samples]
    Bs = data_d["B"][rand_inds]
    Ls = [utils.inv_convert_labels(l) for l in data_d["L"][rand_inds]]
    Xs = data_d["X"][rand_inds]
    Es = data_d["E"][rand_inds]
    Bs_flat = Bs.reshape((len(Bs), -1))

    # calc similarity kernel like in SOAP in bartok "on chemical environments"
    dots = np.zeros((N_samples, N_samples))
    print("\nCalculating Similarity Kernel for {} Random Samples...".format(N_samples))
    for i in tqdm.tqdm(range(N_samples)):
        for j in range(i, N_samples):
            dots[i,j] = (np.dot(Bs_flat[i], Bs_flat[j])/np.sqrt(np.dot(Bs_flat[i],Bs_flat[i])*np.dot(Bs_flat[j],Bs_flat[j]))) ** gamma
            dots[j,i] = dots[i,j]

    sum_dots = dots.sum(axis=1)
    min_inds = np.argsort(sum_dots)
    
    # select M most diverse molecules
    div_inds = [min_inds[0]] # start with most diverse molecule from kernel
    # loop through M molecules
    for m in range(1,N_start):
        remaining = set(min_inds) - set(div_inds) # set difference for molecules not included in diverse set M    
        min_sim_counter = 1e8
        min_ind = None
        
        # loop through not selected molecules i
        for i in remaining:    
            # calc similarities between selected mol i and all the diverse mols in M
            sims = np.array([dots[i,d] for d in div_inds])
            min_sim = np.min(sims)

            # update with best molecule so far
            if min_sim < min_sim_counter:
                min_sim_counter = min_sim
                min_ind = i

        # add molecule i with the lowest minimum similiarity to M
        div_inds.append(min_ind)
        
    # convert to ase mols and align
    mols = [utils.create_ase_mol(Xs[ind], Ls[ind]) for ind in div_inds]
    mols_a = utils.align_mols(mols, ref_idx=-1)

    # center mols
    for mol_a in mols_a: 
        mol_a.set_cell([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]])
        mol_a.set_pbc([False, False, False])
        mol_a.center()
        
    #view(mols_a)

    # delete all files currently in directory
    fls = os.listdir(save_path)
    for f in fls:    
        os.remove(os.path.join(save_path, f))

    # save diverse mols as diff starting xyz
    for i,ind in enumerate(div_inds):
        mols_a[i].info = {'Energy': Es[ind]}		
        write(os.path.join(save_path, '{}.xyz'.format(i)), mols_a[i])
        
    print("\nSaved {} Starting Mols to:\n {}\n".format(N_start, save_path))

def main() -> None:
    """
    Main function to execute the script. Parses arguments, load data, merges data, and saves the results.
    """

    # parse arguments
    args = parse_arguments()
    mol_name = config.MOL_LIST[args.n]
    print('\nSelected Molecule: {}\n'.format(mol_name))

    # Load path
    load_path = os.path.join(config.BASE_PATH, 'data/bispec', mol_name)
    chosen_dir = select_dataset_directory(load_path, args.s, args.u)
    load_path = os.path.join(load_path, chosen_dir)
    print('\nLoading From: {}\n'.format(load_path))

    # Load data
    data_ds, infos = load_data(load_path)
    data_d, info_all = concatenate_data(data_ds, infos, args.p)

    # Save data
    save_path = os.path.join(load_path, str(info_all['N_mols']))
    save_data(save_path, data_d, info_all, info_all['N_mols'])

    # calc the starting mols with which to start the inversion
    if args.N != 0:
        save_path = os.path.join(config.BASE_PATH, 'gan/invert/start/', mol_name)
        starting_mols(save_path, data_d, args.N, args.M)

if __name__ == "__main__":
    main()

