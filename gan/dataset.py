#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import numpy as np
from typing import Tuple, List, Optional, Union, Dict

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import torch
from torch.utils.data import Dataset

from modules import config

def combined_info(infos: List[dict]) -> dict:
    """
    Forward pass of the generator.

    Parameters
    ----------
    infos
        List of dictionaries containing information about each individual molecule type dataset.

    Returns
    -------
    dict
        Single dictionary where each key contains include info about all sperate datasets.  
    """

    # Try/except is to allow a list of single dict as input
    try: 
        info = copy.deepcopy(infos[0])
    except:
        info = copy.deepcopy(infos)
        infos = [copy.deepcopy(infos)]
        
    info['N_set'] = len(infos)
    info['N_mols'] = [infos[i]['N_mols'] for i in range(info['N_set'])]
    info['N_at'] = [infos[i]['N_at'] for i in range(info['N_set'])]
    info['atomic_labels'] = [infos[i]['atomic_labels'] for i in range(info['N_set'])] # removed sorting?
    info['set_names'] = [infos[i]['set_name'] for i in range(info['N_set'])] # added sorting
    
    species_list = [infos[i]['species'] for i in range(info['N_set'])]
    info['all_species'] = list(np.unique([i for species_set in species_list for i in species_set]))
    info['species'] = [sorted(infos[i]['species']) for i in range(info['N_set'])]
    info['N_species'] = [len(infos[i]['species']) for i in range(info['N_set'])]
    info['N_all_species'] = len(info['all_species'])
    
    info['labels_to_names'] = {}
    
    # Get all unique {species:rc} dicts
    for i in range(info['N_set']):
        info['rc_dict'].update(infos[i]['rc_dict'])

    return info

def sum_over_species(b: List[torch.Tensor], l: List[torch.Tensor], info: dict) -> List[torch.Tensor]:
    """
    Sums descriptors tensors for each sample over all atoms of the same species - needed to match inversion method.
    
    Parameters
    ----------
    b
        List of descriptor tensors of shape [N_set][N_mol,N_at,N_b].
    l
        List of species label tensors of shape [N_set][N_mol,N_at].

    Returns
    -------
    List[torch.Tensor]
        List of summed descriptor tensors of shape [N_set][N_mol,N_species,N_b].
    
    Notes
    -----
        - Species outputted should be ordered in increasing atomic weights e.g. (0,1,2,3).
        - Assumes l is not randomised for each sample. 
    """
    
    # check if N_set is first dimension 
    # if b and l are tensors then there is no set dimension -> add outer list to conform
    flag = 0
    if type(l) == type(torch.tensor(0)) and type(b) == type(torch.tensor(0)):
        l = [l]
        b = [copy.deepcopy(b.detach())]
        flag = 1
    # if b and l are lists then procede
    else:
        b = copy.deepcopy(b)
        
    b_new = []
    N_set = len(b)
    B_ind_dict = {species: i for i, species in enumerate(info['all_species'])} # species label to B ind dict
    
    # For each dataset
    for i_set in range(N_set):
        
        N_mol = len(b[i_set])
        species_set = torch.unique(l[i_set])

        b_set = torch.zeros((N_mol, info['N_all_species'], info['N_feats']))
                    
        # For each sample of mol in input
        for i_species, species in enumerate(species_set):
            labels = l[i_set][0] # assume L is the same for all samples in set - take first mol
            if N_mol == 1:
                labels = l[i_set]
            species_inds = [i for i, label in enumerate(labels) if label == species]
            B_ind = B_ind_dict[species.item()]
            
            for species_ind in species_inds:
                b_set[:,B_ind] += b[i_set][:,species_ind].squeeze(0)

        b_new.append(b_set)
        
    return b_new

def species_seperator(b: list[torch.Tensor], l: list[torch.Tensor], info: dict) -> List[List[list[torch.Tensor]]]:
    """
    Separate bispectrum components descriptors per atomic species.

    Parameters
    ----------
    b
        List of tensors of shape [N_set][N_mol, N_at, N_b] for bispectrum components for molecules in each dataset.
    l
        List of tensors of shape [N_set][N_mol, N_at] for atom species labels for each molecule.
    info
        Combined information dictionary.

    Returns
    -------
    List[List[torch.Tensor]]
        Nested list of shape [N_set][N_species][N_mol,N_at_species,N_b] where:
        - Outer list corresponds to different molecule type datatsets.
        - Inner list contains tensors per atomic species.

    Notes
    -----
    - The species order is sorted in increasing label value (e.g. [0, 1, 2, 3]).
    - Designed primarily for data visualization and analysis.
    - Assumes each molecule within a dataset shares the same atomic composition.
    """

    N_set = len(b)
    b_set = []
    
    # For each dataset
    for i_set in range(N_set):
        
        b_species = []
        N_mol = len(b[i_set])
        unq_species_set = sorted(np.unique(l[i_set]))
        
        # For each species
        for species in unq_species_set:
            # Find number of atoms of each species in molecule
            N_at_species = len([label for label in l[i_set][0] if label == species])
            b_mol = torch.zeros((N_mol, N_at_species, info['N_feats']))
                                
            # For each sample of mol in input
            for i_mol in range(N_mol):    
                labels = l[i_set][i_mol]
                species_inds = [i for i, label in enumerate(labels) if label == species]

                b_mol[i_mol] = b[i_set][i_mol][species_inds]
            b_species.append(b_mol)                
        b_set.append(b_species)
    
    return b_set
    
def B_scaling(
    Bs: List[torch.Tensor], 
    info: dict, 
    stds: Optional[List[StandardScaler]] = None
) -> Union[
    Tuple[List[torch.Tensor], List[StandardScaler]],
    List[torch.Tensor]
]:
    """
    Perform feature scaling (mean=0, std=1) on descriptor tensors B for each dataset in a list.

    Parameters
    ---------- 
    Bs
        List of descriptor tensors, each of shape [N_mol,N_species,N_feat]
    info
        Combined dataset information dictionary.
    stds
        An optional list of sklearn StandardScalers used to perform feature scaling - created/fitted if not provided.

    Returns
    -------
    If stds is None:
        List[torch.Tensor]
            Standardised tensors of the same shapes as Bs.
        List[StandardScaler]
            The fitted scalers for each dataset.
    If stds is provided:
        List[torch.Tensor]
            Standardised tensors of the same shapes as Bs.
    """

    if stds is None:
        stds = [StandardScaler() for _ in range(info["N_set"])]
        stds_flag = True
    else:
        stds_flag = False
        
    Bs_std = []     
    for i in range(info["N_set"]):
        B = Bs[i]
        n_mol, n_species, n_feat = B.shape

        if stds_flag:
            B_std = stds[i].fit_transform(B.reshape(-1, n_species * n_feat))
        else:
            B_std = stds[i].transform(B.reshape(-1, n_species * n_feat))

        Bs_std.append(torch.tensor(B_std, dtype=B.dtype).reshape(n_mol, n_species, n_feat))
    
    return (Bs_std, stds) if stds_flag else Bs_std

    
def B_inv_scaling(
    Bs_std: List[torch.Tensor], 
    info: dict, 
    stds: List[StandardScaler]
) -> List[torch.Tensor]:
    """
    Perform inverse feature scaling on scaled descriptor tensors B for each dataset in a list.

    Parameters
    ----------
    Bs_std
        List of descriptor tensors, each of shape [N_mol,N_species,N_feat]
    info
        Combined dataset information dictionary.
    stds
        List of sklearn StandardScalers used to perform inverse feature scaling.

    Returns
    -------
    List[torch.Tensor]
        The original unscaled descriptor tensors of the same shapes as Bs_std.
    """

    Bs = []     
    for i in range(info["N_set"]):
        B_std = Bs_std[i]
        B = stds[i].inverse_transform(B_std.reshape((-1, info['N_all_species']*info['N_feats'])))
        
        Bs.append(torch.tensor(B.reshape((-1, info['N_all_species'], info['N_feats'])).type(dtype=B_std.type())))
    
    return Bs
    
def E_scaling(
    Es: List[torch.Tensor], 
    info: dict, 
    norms: Optional[List[MinMaxScaler]] = None
) -> Tuple[List[torch.Tensor], List[MinMaxScaler]]:
    """
    Perform feature scaling (max=1, min=0) on target energy tensors E for each dataset in a list.

    Parameters
    ----------
    Es
        List of energy tensors, each of shape [N_mol,]
    info
        Combined dataset information dictionary.
    norms
        An optional list of sklearn MinMaxScaler used to perform feature scaling - created/fitted if not provided.

    Returns
    -------
    List[torch.Tensor]
        Normalised tensors of the same shapes as Es.
    List[MinMaxScaler]
        The fitted scalers for each dataset.
    """
    
    if norms is None:
        norms = [MinMaxScaler() for _ in range(info["N_set"])]
        norms_flag = True
    else:
        norms_flag = False

    Es_norm = []     
    for i in range(info["N_set"]):
        E = Es[i]
        if norms_flag:  
            E_norm = torch.tensor(norms[i].fit_transform(Es[i].unsqueeze(1).cpu()))
        else:
            E_norm = torch.tensor(norms[i].transform(Es[i].unsqueeze(1).cpu()))
            
        Es_norm.append(E_norm.squeeze(1).type(dtype=E.type()))
    
    return Es_norm, norms
    
def E_inv_scaling(
    Es_norm: List[torch.Tensor], 
    info: dict, 
    norms: List[MinMaxScaler]
) -> List[torch.Tensor]:
    """
    Perform inverse feature scaling (max=1, min=0) on normalised energy tensors E for each dataset in a list.

    Parameters
    ----------
    Es_norm
        List of normalised energy tensors, each of shape [N_mol,]
    info
        Combined dataset information dictionary.
    norms
        A list of sklearn MinMaxScaler used to perform inverse feature scaling.

    Returns
    -------
    List[torch.Tensor]
        Original, unnormalised tensors of the same shapes as Es_norm.
    """

    Es = []     
    for i in range(info["N_set"]):
        E_norm = Es_norm[i]
        E = norms[i].inverse_transform(E_norm.unsqueeze(1).cpu())
        Es.append(torch.tensor(E.squeeze(0).type(dtype=Es_norm[0].type())))
    
    return Es
         
class BL_DataSet(Dataset):
    """
    PyTorch Dataset for molecular descriptor data used to train a basic GAN model.

    This dataset handles multiple molecular datasets (sets) and allows sampling of batches
    of molecular feature tensors (`Bs_std`) and atomic labels (`Ls`) for training or evaluation.

    Parameters
    ----------
    dataset_dict : dict
        Dictionary containing standardized dataset tensors:
            - 'Bs_std' : list of torch.Tensor
                Standardised descriptor tensor each of shape (N_mol, N_species, N_feats).
            - 'Ls' : list of torch.Tensor
                Species label tensors each of shape (N_mol, N_atoms).
                
    info : dict
        Combined dataset information dictionary.

    GS : dict
        GAN training settings dictionary.
    """
    
    def __init__(self, dataset_dict: Dict[str, List[torch.Tensor]], info: dict, GS: dict) -> None:

        self.info = info
        self.bs = dataset_dict['Bs_std']
        self.ls = dataset_dict['Ls']
        self.N_batch = GS['N_batch']
        self.N_mols = [bs.shape[0] for bs in self.bs]

    def __len__(self) -> int:
        """Return the total number of molecules across all sets."""
        return int(sum(self.N_mols))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve a single data sample consisting of (B, L) tensors.

        The global index `idx` is mapped to the appropriate dataset (set index)
        and local molecule index within that set. Random batches of Bs
        are sampled uniformly within the chosen dataset.

        Parameters
        ----------
        idx
            Global sample index in the range [0, sum(N_mols)) - across all datasets.

        Returns
        -------
        b: torch.Tensor 
            Descriptor tensor of shape (N_species, N_batch, N_feats).
        
        l: torch.Tensor 
            Species label tensor of shape (N_atoms,).
        """
        
        # Split idx E {0,sum(self.info['N_mol'])} into two indexs: set_ind E {0,N_set} and idx(_new) E {0,self.info['N_mol'][set_ind]}
        set_ind = 0        
        for i_set in range(self.info['N_set']):
            if idx >= self.N_mols[i_set]:
                set_ind += 1
                idx -= self.N_mols[i_set]
            else:
                break

        # Sample uniform batches here (diff Bs but same Ls) -> batchsize in dataloader must be 1
        idxs = torch.randint(0, self.N_mols[set_ind], (self.N_batch,))
        b = self.bs[set_ind][idxs].permute((1,0,2))
        l = self.ls[set_ind][idx]
        
        return b, l

    def my_collate(self, batch):
        b,l = zip(*batch) # b,l are one-element tuples
        return torch.squeeze(b[0], dim=0), torch.squeeze(l[0], dim=0)
            
class BLE_DataSet(Dataset):
    """
    PyTorch Dataset for molecular descriptor data used to train an energy-conditioned GAN model.

    This dataset handles multiple molecular datasets (sets) and allows sampling of batches
    of molecular feature tensors (`Bs_std`), atomic labels (`Ls`), and energies (`Es_norm`)
    for training or evaluation.

    Parameters
    ----------
    dataset_dict : dict
        Dictionary containing standardized dataset tensors:
            - 'Bs_std' : list of torch.Tensor
                Standardised descriptor tensor each of shape (N_mol, N_species, N_feats).
            - 'Ls' : list of torch.Tensor
                Species label tensors each of shape (N_mol, N_atoms).
            - 'Es_norm' : list of torch.Tensor
                Normalised energy tensors each of shape (N_mol,).
                
    info : dict
        Combined dataset information dictionary.

    GS : dict
        GAN training settings dictionary.
    """
    
    def __init__(self, dataset_dict: Dict[str, List[torch.Tensor]], info: dict, GS: dict) -> None:

        self.info = info
        self.bs = dataset_dict['Bs_std']
        self.ls = dataset_dict['Ls']
        self.es = dataset_dict['Es_norm']
        self.N_batch = GS['N_batch']
        self.N_mols = [len(_es) for _es in self.es]

    def __len__(self) -> int:
        """Return the total number of molecules across all sets."""
        return int(sum(self.N_mols))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieve a single data sample consisting of (B, L, E) tensors.

        The global index `idx` is mapped to the appropriate dataset (set index)
        and local molecule index within that set. Random batches of Bs and Es
        are sampled uniformly within the chosen dataset.

        Parameters
        ----------
        idx
            Global sample index in the range [0, sum(N_mols)) - across all datasets.

        Returns
        -------
        b: torch.Tensor 
            Descriptor tensor of shape (N_species, N_batch, N_feats).
        
        l: torch.Tensor 
            Species label tensor of shape (N_atoms,).
        
        e: torch.Tensor
            Energies tensor of shape (N_batch, 1).
        """
        
        # Split idx E {0,sum(self.info['N_mol'])} into two indexs: set_ind E {0,N_set} and idx(_new) E {0,self.info['N_mol'][set_ind]}
        set_ind = 0        
        for i_set in range(self.info['N_set']):
            if idx >= self.N_mols[i_set]:
                set_ind += 1
                idx -= self.N_mols[i_set]
            else:
                break

        # Sample uniform batches here (diff Bs but same Ls) -> batchsize in dataloader must be 1
        idxs = torch.randint(0, self.N_mols[set_ind], (self.N_batch,))
        b = self.bs[set_ind][idxs].permute((1,0,2))
        l = self.ls[set_ind][idx]
        e = self.es[set_ind][idxs].unsqueeze(1)

        return b, l, e

    def my_collate(self, batch):
        b,l,e = zip(*batch) # b,l are one-element tuples
        return torch.squeeze(b[0], dim=0), torch.squeeze(l[0], dim=0), torch.squeeze(e[0], dim=0)
               
