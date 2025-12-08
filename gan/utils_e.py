#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import rmsd
import tqdm
import copy
import torch
import numpy as np
from pathlib import Path

from random import shuffle
from typing import Dict, Tuple, List, Any, Optional, Union, Sequence

import cobe
from cobe import Dataset
from cobe.descriptors import compute_bispectrum

from ase import Atoms
from ase.io.extxyz import write_xyz

import matplotlib as matplotlib
    
from pyscf import gto, dft

from modules import config
from modules import G_e as G
from gan import dataset as data

def generate_like_train(
    num_mols: int,
    set_inds: List[int],
    nn_G: torch.nn.Module,
    info: dict,
    GS: dict
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    Generate molecular bispectrum (B), atomic labels (L), and energies (E) 
    using a trained generator to match the training dataset distribution.

    Parameters
    ----------
    num_mols
        Total number of molecules to generate per dataset.
    set_inds
        Indices of datasets in `info` to generate samples for - use [0] if only one dataset is present.
    nn_G
        Generator torch neural network.
    info
        Combined info dictionary containing: 'N_all_species', 'N_feats', 'atomic_labels', 'N_at', etc.
    GS
        GAN training settings dictionary, must include 'N_batch'.

    Returns
    -------
    b_gen : list of torch.Tensor
        List of tensors of generated bispectrum componenets, shape [N_set][N_mols, N_all_species, N_feats].
    l_gen : list of torch.Tensor
        List of tensors of generated atomic labels, shape [N_set][N_mols, N_at].
    e_gen : list of torch.Tensor
        List of tensors of generated energies, shape [N_set][N_mols].
    """
    
    b_gen, l_gen, e_gen = [], [], []
        
    # Number of molecules to generate for each batch, and for the final batch too, which may be smaller
    num_batch = num_mols // GS['N_batch']
    remainder_mols = num_mols % GS['N_batch']
    nn_G.eval()

    # For each set
    for set_ind in set_inds:
    
        # Empty tensors for B and L
        b_set = torch.zeros((num_mols, info['N_all_species'], info['N_feats']))
        l_set = torch.zeros((num_mols, info['N_at'][set_ind]), dtype=torch.long)
        e_set = torch.zeros((num_mols, 1))

        # For each batch
        for i_batch in range(num_batch+1):
        
            # Make seeds for nn_G
            z, l, e = G.make_seeds(info, GS, l=torch.tensor(info['atomic_labels'][set_ind]))
            #e = torch.rand(e.shape)
            
            # Generate batch
            b = nn_G(z, l, e)

            b = b.permute((1,0,2)).detach().cpu()
            l = l.detach().cpu()
            e = e.cpu()
            
            # Repeat L changing from [N_at] to [N_batch,N_at]
            l = l.repeat(b.shape[0], 1)
            
            # Convert i_batch into mol_inds
            if i_batch != num_batch:
                mols_inds = torch.arange(i_batch*GS['N_batch'], (i_batch+1)*GS['N_batch'], dtype=torch.long)

            # If its the last batch also shorten generated batch to remainder_mols
            else:
                b = b[:remainder_mols]
                l = l[:remainder_mols]
                e = e[:remainder_mols]
                mols_inds = torch.arange(i_batch*GS['N_batch'], i_batch*GS['N_batch'] + remainder_mols, dtype=torch.long)
            
            b_set[mols_inds] = b
            l_set[mols_inds] = l
            e_set[mols_inds] = e
            
        b_gen.append(b_set)
        l_gen.append(l_set)
        e_gen.append(e_set)
        
    e_gen = data.E_inv_scaling([e.squeeze(1) for e in e_gen], info, info['E_norms'])
    
    return b_gen, l_gen, e_gen
    
def get_train(
    num_mols: int,
    set_inds: List[int],
    dataset: data.BLE_DataSet,
    info: dict,
    GS: dict
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    = Function that gets specified number of samples from specified datasets
    
    Parameters
    ----------
    num_mols
        Total number of molecules to generate per dataset.
    set_inds
        Indices of datasets in `info` to generate samples for - use [0] if only one dataset is present.
    dataset
        Custom PyTorch dataset class created with data.BLE_DataSet()
    info
        Combined info dictionary
    GS
        GAN training settings dictionary
        
    Returns
    -------
    b_real : list of torch.Tensor
        List of tensors of real bispectrum componenets, shape [N_set][N_mols, N_all_species, N_feats].
    l_real : list of torch.Tensor
        List of tensors of real atomic labels, shape [N_set][N_mols, N_at].
    e_real : list of torch.Tensor
        List of tensors of real energies, shape [N_set][N_mols].
    """
    
    b, l, e = [], [], []
        
    # Integer num of batchs needed and frac of single batch to return num_mols molecules
    num_batch = num_mols // GS['N_batch']
    remainder_mols = num_mols % GS['N_batch']

    # List of random indices for each set - must obey rules of dataset __getitem__
    batch_inds = []
    for i, set_ind in enumerate(set_inds):
        if set_ind == 0:
            start_mol = 0
        else:
            start_mol = sum(info['N_mol'][:set_ind])
        batch_inds.append(np.random.randint(start_mol, start_mol+info['N_mol'][set_ind], num_batch+1))

    # For each set
    for i_set, set_ind in enumerate(set_inds):
    
        # Empty tensors for b and l for each set
        b_set = torch.zeros((num_mols, info['N_all_species'], info['N_feats']))#, dtype=torch.float)
        l_set = torch.zeros((num_mols, info['N_at'][set_ind]), dtype=torch.long)
        e_set = torch.zeros((num_mols,))

        batch_inds_set = batch_inds[i_set]
        
        for i_batch in range(len(batch_inds_set)):
            # Get a batch of random training samples - b.shape=[N_batch,N_at,N_feat] and l.shape=[N_batch,N_at] 
            _b, _l, _e = dataset.__getitem__(batch_inds_set[i_batch])
            _b = _b.permute((1,0,2))
            _l = _l.repeat(_b.shape[0], 1)
            # _e =  FIX SHAPE
            
            # convert i_batch into mol_inds
            if i_batch != len(batch_inds[i_set]) - 1:
                mols_inds = torch.arange(i_batch*GS['N_batch'], (i_batch+1)*GS['N_batch'], dtype=torch.long)

            # if its the last batch also shorten batch to remainder_mols
            else:
                _b = _b[:remainder_mols]
                _l = _l[:remainder_mols]
                _e = _e[:remainder_mols]
                mols_inds = torch.arange(i_batch*GS['N_batch'], i_batch*GS['N_batch'] + remainder_mols, dtype=torch.long)
                
            b_set[mols_inds] = _b
            l_set[mols_inds] = _l
            e_set[mols_inds] = _e
            
        b.append(b_set)
        l.append(l_set)
        e.append(e_set)
    
    return b, l, e
    
def compute_ia_dists(xyz_coords: Union[np.ndarray, list]) -> np.ndarray:
    """
    Compute the pairwise interatomic distances between all atoms.

    Parameters
    ----------
    xyz_coords 
        Cartesian coordinates of atoms of shape (N_at, 3).

    Returns
    -------
    np.ndarray
        Matrix of shape (N_at, N_at) where element [i, j] is the Euclidean distance between atom i and atom j.
    """
    
    N_at = len(xyz_coords)
    A = np.zeros((N_at, N_at))
    for i in range(N_at - 1):
        for j in range(i + 1, N_at):
            A[i, j] = np.linalg.norm(xyz_coords[i] - xyz_coords[j], axis=2)
            A[j, i] = A[i, j]
    return A

def triu_label_combinations(atomic_labels: List[str]) -> np.ndarray:
    """
    Generate all unique upper-triangular (i < j) combinations of atomic labels.

    Parameters
    ----------
    atomic_labels
        List of atomic symbols for atoms in a molecule (e.g. ['C', 'H', 'H', 'O']).

    Returns
    -------
    np.ndarray
        Array of unique pairwise label combinations (sorted alphabetically), of shape [N_pairs], where N_pairs = N_at * (N_at - 1) / 2.

    Example
    -------
    >>> triu_label_combinations(['C', 'H', 'O'])
    array(['CH', 'CO', 'HO'], dtype='<U2')
    """
    
    N_at = len(atomic_labels)
    pairs = []
    for i in range(N_at - 1):
        for j in range(i + 1, N_at):
            tmp = atomic_labels[i] + atomic_labels[j]
            tmp = ''.join(sorted(tmp))
            pairs.append(tmp)
    
    return np.array(pairs)        
    
def convert_labels(
    old_labels:Union[List[Union[str, int]], np.ndarray],
    force_convert: bool = False
) -> np.ndarray:
    """
    Convert atomic labels (symbols or atomic numbers) to standardized integer encodings.

    Conversion mapping:
        'H' or 1 → 0
        'C' or 6 → 1
        'N' or 7 → 2
        'O' or 8 → 3

    Parameters
    ----------
    old_labels
        Atomic labels, e.g. ['C', 'H', 'O'] or [6, 1, 8], maybe already encoded as integers [0, 1, 2, 3].
    force_convert
        If True, conversion is applied even if the input appears already encoded.

    Returns
    -------
    np.ndarray of int
        Encoded integer labels where each label ∈ {0, 1, 2, 3}.

    Raises
    ------
    ValueError
        If any label is unrecognized or cannot be mapped.
    """
    
    # Get unique labels
    unq_old_labels = set(np.unique(old_labels))
    all_possible_labels = set([0,1,2,3])
    
    # If input Labels are in desired int format - do nothing unless specified by force_convert
    if unq_old_labels.issubset(all_possible_labels) and not force_convert:
        return old_labels
        
    # Else convert to new_labels with L_dict
    else:
        new_labels = []
        L_dict = {'H':0,'C':1,'N':2,'O':3}
        
        # For each old label convert to int label
        for i, label in enumerate(old_labels):
            if label == 'H' or label == 1:
                new_labels.append(L_dict['H'])
            elif label == 'C' or label == 6:
                new_labels.append(L_dict['C'])
            elif label == 'N' or label == 7:
                new_labels.append(L_dict['N'])
            elif label == 'O' or label == 8:
                new_labels.append(L_dict['O'])
            # If old label not recognised then break
            else:
                print('Unrecongised Label:\n    index={}\n    label={}'.format(i, label))
                raise ValueError(f"Unrecognized atomic label") 
                
        return new_labels

def inv_convert_labels(old_labels: Union[List[Union[int, str]], np.ndarray]) -> np.ndarray:
    """
    Convert integer atomic labels back to atomic symbols.

    Conversion mapping:
        0 → 'H'
        1 → 'C'
        2 → 'N'
        3 → 'O'

    Parameters
    ----------
    old_labels : list or np.ndarray of int or str
        Atomic labels to convert, e.g. [0, 1, 3] or ['H', 'C', 'O'].

    Returns
    -------
    np.ndarray of str
        Converted atomic symbols where each label ∈ {'H', 'C', 'N', 'O'}.

    Raises
    ------
    ValueError
        If any label is unrecognized or cannot be mapped.
    """

    # Get unique labels
    unq_old_labels = set(np.unique(old_labels))
    all_possible_labels = set(['H','C','N','O'])
    
    # If input Labels are in desired str format - do nothing
    if unq_old_labels.issubset(all_possible_labels):
        return old_labels
        
    # Else convert to new_labels with L_dict
    else:
        new_labels = []
        L_dict = {0:'H',1:'C',2:'N',3:'O'}
        
        # For each old label convert to str label
        for i, label in enumerate(old_labels):
            if label == 0:
                new_labels.append(L_dict[0])
            elif label == 1:
                new_labels.append(L_dict[1])
            elif label == 2:
                new_labels.append(L_dict[2])
            elif label == 3:
                new_labels.append(L_dict[3])
            # If old label not recognised then break
            else:
                print('Unrecongised Label:\n    index={}\n    label={}'.format(i, label))
                raise ValueError(f"Unrecognized integer label") 
                
        return new_labels
             
def invert_b(
    start_molecule: List[cobe.Dataset],
    target_b: Union[List[np.ndarray], List[torch.Tensor]],
    target_l: Union[List[np.ndarray], List[torch.Tensor]],
    target_e: Union[List[np.ndarray], List[torch.Tensor]],
    info: dict,
    rcutfac: int = 0,
    rc: List[float] = [0.0],
    rfac0: float = 0.99,
    twojmax: int = 8,
    pbc_flag: bool = True,
    pbar_disable: bool = False,
    loss_threshold: float = 0.2,
    max_N: int = 10000,
    lr: float = 0.000008,
    alpha: float = 8e-4,
    eta: int = 1000,
    patience: int = 1000
) -> Tuple[List[cobe.Dataset], np.ndarray, str]:    
    """
    Calculate Cartesian coordinates that correspond to a target set of bispecrum components.  

    Parameters
    ----------
    start_molecule
        ASE Mol object created with cobe.Dataset - works best of is relaxed xyz coords of molecule.
    target_b
        Target bispectrum components of molecule to be inverted, shape [N_species, N_feats].
    target_l
        Integer labels for atoms in target_b, shape [N_at,].
    target_e
        Target energy value of molecule, shape [1,].
    rcutfac
        Scale factor applied to all cutoff radii e.g. info['rcutfac'].
    rc
        Cutoff radii, one for each type e.g. info['rc_dict'].
    rfac0
        Parameter in distance to angle conversion (0 < rcutfac < 1).
    twojmax
        Band limit for bispectrum components - controls number of features.
    loss_threshold
        Value of inversion loss function which is defined to be fully converged -> break.
    max_N
        Maximum number of optimization iterations to perform before stopping process. 
    lr
        Learning rate controlling step size of each update.
    alpha
        Learning rate controlling amplitude of random noise. 
    eta
        Exponential decay constant of random noise.
    patience
        Number of iterations to wait before breaking without an decrease in loss values.
        
    Returns
    -------
    trajectory : List[cobe.Dataset]
        ASE Atoms-like molecule objects, shape [N_iterations,].
    loss: np.ndarray
        Loss function values for each iteration, shape [N_iterations,]
    convergence_type
        Label for dir name in which to save .xyz file - either {'conv','not_conv'}
    """
        
    # Get atom types - in same order as ASE and LAMMPS e.g. [6,1,8] 
    l = start_molecule.trajectory[-1].types
    all_species = start_molecule.types
    all_species2 = convert_labels(start_molecule.types, force_convert=True)

    # Define Useful Parameters
    n_atoms = len(target_l)
    n_species = len(target_b)
    n_features = len(target_b[0])
    target_species = sorted(info['all_species'])

    # Rearrange target B - change from the sorted order used in GAN to order used in LAMMPS
    sum_bt = np.zeros((n_species, n_features))
    for i in range(n_species):
        sum_bt[all_species2.index(target_species[i])] += target_b[i]

    # Set initial energy
    start_molecule.set_energy(target_e)
    
    # Initialise data to be returned
    trajectory = [start_molecule.trajectory[-1]]
    loss = np.zeros(max_N)
    convergence_type = 'not_conv'

    # Make arrays to be appended to
    pbar = tqdm.tqdm(range(max_N), disable=pbar_disable)
    sum_b_arr = np.zeros((max_N, n_species, n_features))
    early_stop_counter = 0
    
    # For each iteration try to make sum_b closer to sum_bt
    for n in pbar: 

        prototype_mol = copy.deepcopy(trajectory[-1])

        compute_bispectrum(prototype_mol, 
                                 types=start_molecule.types,
                                 rcutfac=rcutfac,
                                 rfac0=rfac0,
                                 twojmax=twojmax,           
                                 rc=rc,
                                 pbc_flag=pbc_flag                   
                                 )  

        b = prototype_mol.descriptors
        bd = prototype_mol.descriptors_d
        
        # Sum up all Bs of same species 
        sum_b = np.zeros((n_species, n_features))
        for i in range(n_atoms):
            sum_b[all_species.index(l[i])] += b[i]

        # Calc loss
        for j in range(n_species):
            loss[n] += (1/n_species) * np.dot(sum_bt[j]-sum_b[j], sum_bt[j]-sum_b[j])

        # Calc derivative of loss
        grad_loss = np.zeros((n_atoms, 3))
        for i in range(n_atoms):
            for j in range(n_species):
                for k in range(3):
                    grad_loss[i][k] += -2 * np.dot(sum_bt[j]-sum_b[j], bd[i][j][k])

        # Update XYZ positions
        for i in range(n_atoms):
            loss_term = lr * grad_loss[i]
            noise_term = alpha * np.exp(-n/eta) * np.random.uniform(-1, 1, (3,))
            prototype_mol.positions[i] = np.array(prototype_mol.positions[i]) + loss_term + noise_term

        # Make progress bar print loss value   
        pbar.set_description("Loss: {}".format(np.round(loss[n],7)))
        
        # If optimization Converges then Break loop
        if loss[n] < loss_threshold:
            print("Loss Threshold Reached ...\n")
            convergence_type = 'conv'
            break
    
        # Append updated Molecule
        sum_b_arr[n] = sum_b
        #trajectory.append(prototype_mol)
        trajectory = [prototype_mol]

        # If Loss is not improved for patience - then break
        if early_stop_counter > patience:
            print("Early Stopping ...\n")
            break
        elif np.round(loss[n],8) >= np.round(loss[n-1],8):
            early_stop_counter += 1
        elif np.round(loss[n],8) < np.round(loss[n-1],8):
            early_stop_counter = 0

    # remove extra zeros from loss array
    loss = loss[:n+1]
    
    return trajectory, loss, convergence_type
    
def gen_xyz(
    nn_G: torch.nn.Module,
    nn_E: torch.nn.Module,
    info: Dict[str, Any],
    GS: Dict[str, Any],
    real_data: Optional[Dict[str, torch.Tensor]],
    get_trj: bool = True,
    save_trj: bool = False,
    save_end: bool = True,
    pbc_flag: bool = True,
    rfac0: float = 0.99,
    loss_threshold: float = 0.2,
    max_N: int = 2000,
    start_N: int = 500,
    start_M: int = 20,
    lr: float = 5e-8,
    alpha: float = 8e-4,
    eta: float = 1000,
    rand_target_ind: bool = False,
) -> Tuple[List[List[Any]], List[List[float]], str]:
    """
    Generate atomic coordinates by inverting bispectral componenets descriptors.

    This function orchestrates:
      - constructing/choosing target B, L, E
      - choosing starting structures
      - running multiple short optimizations per starter (invert_b)
      - selecting best starter and running a longer optimization
      - optionally running DFT and saving final trajectory

    Parameters
    ----------
    num_mols
        Number of molecules to generate (per dataset in set_inds).
    set_inds
        Indexes of the datasets in `info` to operate on.
    nn_G, nn_E
        Generator and energy predictor models (PyTorch).
    info
        Combined dataset info dictionary (must contain keys used inside).
    GS
        Global settings dict (must contain 'mols', 'model_path', 'N_batch', 'device', etc.)
    real_data
        Dictionary containing training data for 'Bs_sum', 'Ls', 'Es', and 'Xs' used for inversion; otherwise generate new data with nn_G.
    get_trj
        Whether to return all intermediate XYZ coordinates during inversion process, for all molecules.
    save_trj
        Whether to write all intermediate XYZ coordinates during inversion process, for all molecules.
    save_end
        Whether to write final, inverted XYZ coordinates, with associated energies.
    rfac0, weights, pbc_flag, loss_threshold, max_N, start_N, start_M, lr, alpha, eta
        Optimization / inversion hyperparameters (passed to invert_b).
    rand_target_ind
        If True, choose random target molecule indices instead of deterministic ones.

    Returns
    -------
    trjs : list of list
        Returned trajectory objects, shape of  [len(set_inds)][num_mols] (each entry is a trajectory/list of ASE Atoms)
    losses : list of list
        Loss history per inversion, shape of [len(set_inds)][num_mols]
    save_path : str
        Directory used for saving (if any); empty string if not saving.
    """

    # Variables for Path
    base_dir = os.path.join(config.BASE_PATH, 'gan/invert/')
    conv_types = ['conv','not_conv']
    
    # Parameters
    num_mols = GS['N_mols'] # 
    set_inds = [i for i in range(len(GS['mols']))]
    names = [config.MOL_LIST[int(i)] for i in GS['mols']] # Names of Training Datasets for Loading starting mol
    
    # Get Save Path
    # - if Gen then save in dir gan/invert/datasets_type/model_name/mol_name/conv_type
    if real_data is None: 
        dir_name = get_dir_name(names)
        _model_ind = GS['model_path'].index(str(dir_name))
        _model_name = GS['model_path'][_model_ind:]
        save_path = base_dir + str(_model_name)
        print("save_path",save_path)
        
    # - if Real then save in dir gan/invert/real/datasets_type/mol_name/conv_type
    else:
        data_type = [str(real_data["Bs_sum"][i_set].shape[1]) for i_set in range(len(names))]
        data_type = '_'.join(data_type)
        save_path = os.path.join(base_dir, data_type, '/real/')
    
    # If save_end True then get dir path    
    if save_end or save_trj:
        end_paths, trj_paths = [], []
        
        for i, name_set in enumerate(names):
            end_paths.append(os.path.join(save_path, name_set, 'end/'))  
            trj_paths.append(os.path.join(save_path, name_set, 'trj/'))    
            
            # if dir already exists, find ID of last saved item
            if not os.path.exists(os.path.join(end_paths[i], conv_types[0])):
                for conv_type in conv_types:
                    os.makedirs(os.path.join(end_paths[i], conv_type))
                    os.makedirs(os.path.join(trj_paths[i], conv_type))
                    
    # get target mol ind         
    target_mol_inds = [0 for _ in set_inds]
    if real_data is not None:   
        # if rand_flag
        if rand_target_ind:
            target_mol_inds = [torch.randint(0, info['N_mols'][i_set], (1,)).item() for i_set in range(len(set_inds))]
        # if save_end
        elif save_end:
            target_mol_inds = [sum([len(os.listdir(os.path.join(end_paths[i_set], conv_type))) for conv_type in conv_types]) for i_set in range(len(set_inds))]
    
    # Generate B if Real B not supplied
    if real_data is None:
        b, l, e = generate_like_train(num_mols, set_inds, nn_G, info, GS)
        #b = scale.inv_fs(b, l, info, GS)
        #b = [torch.tensor(info['B_stds'].inverse_transform(b[0].reshape((-1,info['N_all_species']*info['N_feats'])))).reshape((-1,info['N_all_species'],info['N_feats']))] 
        e_pred = [nn_E(b[i_set].permute((1,0,2))).detach().squeeze(1) for i_set in range(len(set_inds))]
        e_pred = data.E_inv_scaling(e_pred, info, info['E_norms'])
        b = data.B_inv_scaling(b, info, info['B_stds'])
        
    elif real_data is not None:
        b = [real_data["Bs_sum"][set_ind] for set_ind in set_inds]
        l = [real_data["Ls"][set_ind] for set_ind in set_inds]
        e = [real_data["Es"][set_ind].unsqueeze(1) for set_ind in set_inds]
        x = [real_data["Xs"][set_ind] for set_ind in set_inds]
        b = data.sum_over_species(b, l, info, GS)

    # Empty lists to be Returned
    trjs, losses = [], []

    # For each dataset specified 
    for i_set, set_ind in enumerate(set_inds):
    
        trjs_set, losses_set = [], []
        
        # change shape from [N_set][N_mols,...] to [N_mols,...]
        _b, _l, _e = b[i_set].numpy(), l[i_set].numpy(), e[i_set].numpy() 
        target_mol_ind = target_mol_inds[i_set]
        end_path = end_paths[i_set]
        trj_path = trj_paths[i_set]
        
        # Set twojmax, rc, rcutfac, and weights
        twojmax = config.TWOJMAX_DICT[info['N_feats']]
        rc = [info['rc_dict'][species] for species in info['species'][set_ind]] 
        rcutfac = info['rcutfac']
        weights = [1.0 for _ in range(info['N_species'][set_ind])]

        # Starting Mols Paths
        start_mol_base_path = os.path.join(base_dir, 'start/{}/'.format(names[set_ind]))
        start_fls = os.listdir(start_mol_base_path)
        start_fls.sort(key=lambda x: float(x[:-4]))    
        start_mols = [Dataset(os.path.join(start_mol_base_path, s)) for s in start_fls][:start_M]
        
        # Print Info
        print('\n\n-----------------------------------')
        print('Inverting {} Molecules of {}'.format(num_mols, names[set_ind]))
        print('-----------------------------------')
        print('Loading Initial XYZs from: {}'.format(start_mol_base_path))
        if save_end:
            print('Saving End XYZ to: {}'.format(end_path))
        if save_trj:
            print('Saving Trajectories to: {}'.format(trj_path))
        print('\nFirst Sample ID: {}'.format(target_mol_ind))
        print('rc', rc)
        print('rcutfac', rcutfac)
        print('weights',weights,'\n')
        
        # For each sample to be Inverted 
        for i, i_mol in enumerate(range(target_mol_ind, target_mol_ind+num_mols)):
            # select i-th sample to be inverted 
            target_b = _b[i_mol]
            target_l = _l[i_mol]
            target_e = _e[i_mol] # remove list - float
            print("target_e",target_e)
            
            trjs_start = []
            losses_start = []

            # For each Starting Mol
            print('\nSample ID: {}'.format(i_mol))
            print("\nChoosing from {} Start Mols".format(len(start_mols)))
            for j, start_mol in tqdm.tqdm(enumerate(start_mols), total=len(start_mols)):
                # Inversion
                trj, loss, conv_type = invert_b(copy.deepcopy(start_mol),
                                                          target_b,
                                                          target_l,
                                                          target_e,
                                                          info=info,
                                                          rcutfac=rcutfac,
                                                          rc=rc,
                                                          rfac0=rfac0,
                                                          twojmax=twojmax,
                                                          pbc_flag=pbc_flag,
                                                          pbar_disable=True,
                                                          loss_threshold=loss_threshold,
                                                          max_N=start_N,
                                                          lr=lr,
                                                          alpha=alpha,
                                                          eta=eta,
                                                          )
                trjs_start.append(trj)
                losses_start.append(loss)
        
            # Choose Start Mol with lowest Loss
            print("\nFinal losses for each starting Mol:\n",[np.round(l[-1],3) for l in losses_start])            
            chosen_start_ind = np.argsort([l[-1] for l in losses_start])[0]
            print("Choosen Start Mol: ind={}, loss={}".format(chosen_start_ind, losses_start[chosen_start_ind][-1]))    
            trj_start, loss_start = trjs_start[chosen_start_ind], losses_start[chosen_start_ind]

            # update chosen start mol with new positions
            start_mol = start_mols[chosen_start_ind]
            start_mol.trajectory[-1].set_positions(trj_start[-1].get_positions())

            # Inversion
            print("\nInverting Best Start Mol")
            trj, loss, conv_type = invert_b(copy.deepcopy(start_mol),
                                                      target_b,
                                                      target_l,
                                                      target_e,
                                                      info=info,
                                                      rcutfac=rcutfac,
                                                      rc=rc,
                                                      rfac0=rfac0,
                                                      twojmax=twojmax,
                                                      pbc_flag=pbc_flag,
                                                      loss_threshold=loss_threshold,
                                                      max_N=max_N-start_N,
                                                      lr=lr,
                                                      alpha=alpha,
                                                      eta=eta,
                                                      )

            # calc pyscf energy
            print(trj[-1].get_positions())
            print(target_l)
            pyscf_e = dft(trj[-1].get_positions(), target_l)
            for n in np.arange(0, len(trj)):
                trj[n].info = {'Energy': target_e}    

            # Save end xyz
            if save_end:
                if real_data is None:
                    trj[-1].info = {'Energy': target_e, 'Energy_pyscf': pyscf_e, 'Energy_nn': e_pred[i_set][i_mol].detach().item()}
                    print(trj[-1].info)
                else:
                    pyscf_og = dft(x[i_set][i_mol], l[i_set][i_mol]) # review this
                    trj[-1].info = {'Energy': target_e, 'Energy_pyscf': pyscf_e, 'Energy_pyscf_og': pyscf_og}
                #write_xyz(end_path+conv_type+'/end_{}_{}.xyz'.format(last_end_dict[conv_type]+i, np.round(loss[-1],2)), trj[-1])
                write_xyz(end_path+conv_type+'/end{}_{}.xyz'.format(i_mol, np.round(loss[-1],6)), trj[-1])
       
        # Appending to Empty Lists for each Set        
        trjs.append(trjs_set)
        losses.append(losses_set)
    
    return trjs, losses, save_path
    
def read_xyz(num_mols: int, path: str, rand_flag: bool = True) -> Tuple[List[np.ndarray], List[List[str]]]:
    """
    Read atomic coordinates and labels from a `.xyz` file, optionally selecting a random subset of molecules.

    Parameters
    ----------
    num_mols
        Number of molecules to return.
    path
        Path to the `.xyz` file to read.
    rand_flag
        Whether to randomise molecules from the file.

    Returns
    -------
    xyz : list of np.ndarray
        Cartesian coordinates for each molecule - each has shape `(N_atoms, 3)`.
    labels : list of list of str
        Atomic symbols for each molecule - each has shape `N_atoms`.
    """
    
    xyz, l = [], []
    
    with open(path) as f: 
        # List of each line in big .xyz
        data = f.read()
        data = data.split('\n')
        
        # Get loop parameters
        N_lines = len(data)
        N_at = int(data[0])
        N_mol = N_lines // (N_at+2)

        # Array of indices of first line of each mol (e.g. N_mol) 
        mol_start_indices = np.arange(0,N_lines-(N_at+2),N_at+2)
        
        # Whether to randomise
        if rand_flag:
            shuffle(mol_start_indices)
            
        selected_indices = mol_start_indices[:num_mols]
        
        for ind in selected_indices:
        
            # List of xyz coords of each mol
            mol_data = data[2 + ind : 2 + ind + N_at] 
            mol_data = [line.split() for line in mol_data]   
        
            # Write coords and forces into lines of small .xyz - either 4 or 7 columns
            xyz_mol, l_mol = [], []
            for n in range(N_at):                
                if len(mol_data[0]) == 4:
                    (atom, x, y, z) = mol_data[n]
                elif len(mol_data[0]) == 7:
                    (atom, x, y, z, _, _, _) = mol_data[n]
                else:
                    raise ValueError(f"Unexpected .xyz line format with {len(mol_data[0])} columns.")
                xyz_mol.append([float(i) for i in [x,y,z]])
                l_mol.append(atom) 
    
            xyz.append(xyz_mol)
            l.append(l_mol)
            
    return xyz, l
    
def read_xyz_dir(
    num_mols: int,
    dir_path: str,
    energy_flag: bool = False,
    dft_energy_flag: bool = False,
    og_energy_flag: bool = False,
    nn_energy_flag: bool = False,
    shuffle_flag: bool = True,
    sorted_flag: bool = True
) -> Union[
    Tuple[List[np.ndarray], List[List[str]]],
    Tuple[List[np.ndarray], List[List[str]], List[float]],
    Tuple[List[np.ndarray], List[List[str]], List[float], List[Optional[float]]],
    Tuple[List[np.ndarray], List[List[str]], List[float], List[Optional[float]], List[Optional[float]]],
]:
    """
    Read a specified number of molecular `.xyz` files from a directory, optionally extracting one or more energy values from the comment line.

    Parameters
    ----------
    num_mols
        Number of molecules to read from the directory.
    dir_path
        Path to the directory containing multiple seperate `.xyz` files.
    energy_flag
        Whether to extract standard "Energy=" values from the comment line.
    dft_energy_flag
        Whether to extract "Energy_pyscf=" values (DFT energies).
    og_energy_flag
        Whether to extract "Energy_pyscf_og=" values (original DFT energies).
    nn_energy_flag
        Whether to extract "Energy_nn=" values (neural network predicted energies).
    shuffle_flag
        Whether to shuffle the file list before reading.
    sorted_flag
        Whether to sort the file list numerically after shuffling - based on inversion loss function values in file name.

    Returns
    -------
    xyz : list of np.ndarray
        Atomic coordinates for each molecule, shape `(N_atoms, 3)`.
    labels : list of list of str
        Atom labels for each molecule, shape `(N_atoms,)`.
    e, e2, e3, e4 : list of float or None, optional
        Energy lists returned depending on which flags are set.

    Notes
    -----
    - Energy values are read from the comment line in each file.
    - If an expected energy tag is missing, `None` is stored for that entry.
    """
    
    xyz, l = [], []
    e, e2, e3, e4 = [], [], [], []
    
    # Ensure trailing slash
    dir_path = os.path.abspath(dir_path)
    fls = [f for f in os.listdir(dir_path) if f.endswith(".xyz")]

    if not fls:
        raise FileNotFoundError(f"No .xyz files found in directory: {dir_path}")
    
    # Randomise molecules
    if shuffle_flag:
        shuffle(fls)
        
    # Sort based on inversion loss function values
    if sorted_flag:
        try:
            fls.sort(key=lambda x: float(x[x.index("_")+1:-4]))
        except:
            fls.sort(key=lambda x: int(x[:-4]))
            
    # Limit number of molecules to available files
    num_mols = min(num_mols, len(fls))
    
    for i in range(num_mols):
    
        with open(dir_path+fls[i], 'r') as f: 
        
            # List of each line in big .xyz
            lines = f.read().split('\n')
            
            # Get loop parameters
            N_at = int(lines[0])
            comment_line = lines[1]
                
            # Get energy
            if energy_flag:      
                try:
                    start_index = comment_line.index("Energy=-") + len("Energy=")
                    end_index = start_index + comment_line[start_index:].index(" ")
                    e_mol = comment_line[start_index:end_index]
                    e.append(float(e_mol))
                except:
                    e.append(None)

            # Get dft energy
            if dft_energy_flag:
                try:
                    start_index = comment_line.index("Energy_pyscf=-") + len("Energy_pyscf=")
                    end_index = start_index + comment_line[start_index:].index(" ")
                    e_mol = comment_line[start_index:end_index]
                    e2.append(float(e_mol))
                except:
                    e2.append(None)
                    
            # Get dft energy 
            if og_energy_flag:
                try:
                    start_index = comment_line.index("Energy_pyscf_og=-") + len("Energy_pyscf_og=")
                    end_index = start_index + 16
                    e_mol = comment_line[start_index:end_index]
                    e3.append(float(e_mol))
                except:
                    e3.append(None)
                    
            # Get nn energy 
            if nn_energy_flag:
                try:
                    start_index = comment_line.index("Energy_nn=-") + len("Energy_nn=")
                    try:
                        end_index = start_index + comment_line[start_index:].index(" ")
                    except:
                        end_index = start_index + comment_line[start_index:].index("\n")
                    e_mol = comment_line[start_index:end_index]
                    e4.append(float(e_mol))
                except:
                    e4.append(None)
    
            xyz_mol, l_mol = [], []

            # List of xyz coords of each mol
            mol_data = lines[2:2+N_at] 
            mol_data = [q.split() for q in mol_data]   
            
            # Parse atomic coordinates and labels
            for n in range(N_at):
                if len(mol_data[0]) == 4:
                    (atom, x, y, z) = mol_data[n]
                elif len(mol_data[0]) == 7:
                    (atom, x, y, z, _, _, _) = mol_data[n]
                
                xyz_mol.append([float(i) for i in [x,y,z]])
                l_mol.append(atom) 

            xyz.append(xyz_mol)
            l.append(l_mol)

    # Return outputs based on which flags were set
    if nn_energy_flag:
        return xyz, l, e, e2, e4 
    if og_energy_flag:
        return xyz, l, e, e2, e3 
    if dft_energy_flag:
        return xyz, l, e, e2 
    if energy_flag:
        return xyz, l, e 
    else:
        return xyz, l 
             
def get_dir_name(names: List[str]) -> str:
    """
    Generate a concise directory name from a list of dataset names.

    Parameters
    ----------
    names : list of str
        List of dataset names, e.g. ['12_benzene', '15_toluene'].

    Returns
    -------
    str
        Combined directory name composed of extracted numeric parts - example: '12_15'
    """

    nums = ["".join([s for s in name if s.isdigit()]) for name in names]
    dir_name = '_'.join(nums)
    return dir_name
    
def pyscf_atom_parser(xyz_coords: Sequence[Sequence[float]], species_labels: Sequence[int]) -> str:
    """
    Convert atomic coordinates and integer species labels into a string compatible with PySCF's `gto.M` atoms.

    Parameters
    ----------
    xyz_coords : sequence of sequences of float
        Cartesian coordinates of atoms with shape (N_atoms, 3).
    species_labels : sequence of int
        Integer atomic species identifiers (e.g., 0 for H, 1 for C, etc.) converted to element symbols via `inv_convert_labels`.

    Returns
    -------
    str
        A single string formatted like an .XYZ file for PySCF, with atoms separated by semicolons.
    """

    scf_coords = []
    species_labels = inv_convert_labels(species_labels)
    
    # write coords and forces into lines of small .xyz
    for n in range(len(xyz_coords)):
        (x, y, z) = xyz_coords[n]
        atom = species_labels[n]
        
        scf_coords.append('{},{},{},{}'.format(atom,x,y,z))
        
    return ';'.join(scf_coords)
    
def dft(xyz: List[torch.Tensor], l: List[torch.Tensor], basis: str = "def2svp"): # other basis is "pbe"

    """
    Compute the DFT total energy (in kcal/mol) for a molecule using PySCF.

    Parameters
    ----------
    xyz : sequence of sequences of float
        Atomic coordinates of shape (N_atoms, 3) - each inner sequence gives [x, y, z] in Ångströms.
    l : sequence of str
        List of atomic symbols corresponding to the atoms in `xyz` : ['C', 'H', 'H', 'H', 'H']
    basis : str
        Basis set to use for DFT calculations.

    Returns
    -------
    float
        The total DFT energy in kcal/mol.

    Notes
    -----
    - The function uses the PBE exchange-correlation functional.
    - Conversion from Hartree to kcal/mol uses `config.HARTREE2KCAL`.
    """
    
    # Convert input geometry into PySCF-readable string
    scf_coords = pyscf_atom_parser(xyz, l)

    # Build molecule with PySCF
    mol = gto.M(atom=scf_coords, basis=basis, parse_arg=False)

    # Set up Kohn-Sham DFT with PBE functional
    mf = dft.KS(mol)
    mf.xc = "pbe"

    # Run SCF calculation
    e_hartree = mf.kernel()

    # Convert Hartree → kcal/mol
    e_kcal = e_hartree * config.HARTREE2KCAL

    return e_kcal

def align_mols(mols: List[Atoms], ref_idx: int = 0) -> List[Atoms]:
    """
    = Function that uses <rmsd> package to translate and rotate all ase mols in line with the idx specified mol such that rmsd is minimized
    """
    
    """
    Align a list of ASE molecule objects (conformers) to a reference conformer using RMSD minimization.

    Parameters
    ----------
    mols
        List of ASE `Atoms` objects representing molecular conformers to align.
    ref_idx
        Index of the reference conformer in `mols` to align all others to.

    Returns
    -------
    list[ase.Atoms]
        List of aligned ASE `Atoms` objects (deep copies of the originals).

    Notes
    -----
    - Hydrogen atoms are ignored in centroid calculation and alignment weighting.
    """
    
    mols = copy.deepcopy(mols)
    mol0 = mols[ref_idx]
    pos_0 = mol0.get_positions()

    # Calc centroid for conformer 0 ignoring Hydrogens                             
    pos_0_noH = np.array([p for i,p in enumerate(pos_0) if mol0.get_chemical_symbols()[i] != "H"])
    cen_0 = rmsd.centroid(pos_0_noH)
    pos_0 -= cen_0
    pos_0_noH -= cen_0
    mol0.set_positions(pos_0)

    # Weights for kabsch_weighted where w_i=0 if i=H else w_i=1
    w = np.array([1 if mol0.get_chemical_symbols()[i] != "H" else 0 for i in range(mol0.get_number_of_atoms())])

    # For all other conformers - trans and rotate to match 0th one
    for mol in mols:

        pos_i = mol.get_positions()

        # Calc centroid for conformer i ignoring Hydrogens                             
        pos_i_noH = np.array([p for j,p in enumerate(pos_i) if mol.get_chemical_symbols()[j] != "H"])
        cen_i = rmsd.centroid(pos_i_noH)
        pos_i -= cen_i
        pos_i_noH -= cen_i

        U,V,_rmsd = rmsd.kabsch_weighted(pos_i, pos_0, w)
        pos_i = np.dot(pos_i, U)
        mol.set_positions(pos_i)

    return mols 
    
def kcal2ev(es, n_atoms: int):
    return (es * config.KCAL2EV) / n_atoms
    
def create_ase_mol(xyz, l):
    # function to take in a list of xyz and a list of atomic numbers and return ase Atoms object
    return Atoms(l, positions=xyz)

def remove_base(base_path: str, abs_path: str) -> str:
    """
    Function that converts an absolute path into a relative path.

    Parameters
    ----------
    base_path : str
        Relative base path to project root 
    abs_path : str
        System-dependent absolute path to be converted to relative path.

    Returns
    -------
    str
        The converted new relative base path.
    """
    
    base = Path(base_path).resolve()
    p = Path(abs_path).resolve()
    return str(p.relative_to(base))

def make_relative_GS(GS: dict) -> dict:
    """
    Function to make all paths in GS relative to config.BASE_PATH.

    Parameters
    ----------
    GS : dict
        GAN training settings dictionary.

    Returns
    -------
    dict
        Updated GAN training settings dictionary with relative paths.
    """
    
    GS2 = copy.deepcopy(GS)
    GS2['model_path'] = remove_base(config.BASE_PATH, GS['model_path'])
    GS2['data_paths'] = [remove_base(config.BASE_PATH, p) for p in GS['data_paths']]
    return GS2
    
