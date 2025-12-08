#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from typing import Optional, Tuple

import torch
import torch.nn as nn

from modules import config
    
def make_seeds(
    info: dict, 
    GS: dict, 
    l: Optional[torch.Tensor] = None, 
    N_batch: int = -1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate latent seeds and atomic labels for GAN training or inference.

    Parameters
    ----------
    info
        Dataset information dictionary containing keys such as:
        - 'N_set': number of available label sets.
        - 'atomic_labels': list of atomic label arrays.
        - 'kdes': list of KDE models for energy sampling.
    GS
        GAN settings dictionary containing keys such as:
        - 'device': target computation device (e.g., 'cuda' or 'cpu').
        - 'distro': latent distribution type ('uniform' or 'normal').
        - 'N_z': dimensionality of the latent space.
        - 'N_batch': default batch size for sampling.
    l
        Tensor of atomic species labels to use. If not provided, a random label set is chosen.
    N_batch
        Number of latent seeds to generate. Defaults to `GS['N_batch']` if not specified.

    Returns
    -------
    z
        Generated latent vectors of shape `(N_batch, GS['N_z'])`.
    l
        Tensor of atomic labels corresponding to the selected dataset set.
    """
    
    # Either use supplied labels or randomly choose labels from info dict
    if l is None:
        set_ind = torch.randint(0, info['N_set'], (1,)) # only 1 L as uniform batches
        l = torch.tensor(info['atomic_labels'][set_ind]).to(GS['device'])
    else:
        #l = torch.tensor(l)
        l = l.clone().to(GS['device'])#.detach().requires_grad_(True)
        set_ind = info['atomic_labels'].index(l.tolist())
        
    # by default create a batch of seeds - alternatively can specify N_batch to produce desired number of seeds
    if N_batch == -1:
        N_batch = GS['N_batch']
    N_at = len(l)    
        
    # Choose latent distribution 
    if GS['distro'] == 'uniform':
        z = torch.FloatTensor(N_batch, GS['N_z']).uniform_(-1, 1).to(GS['device'])
    elif GS['distro'] == 'normal':
        z = torch.randn(N_batch, GS['N_z']).to(GS['device'])
    else:
        raise Exception('distro')
        
    return z, l

def last_epoch(GS: dict) -> int:
    """
    Retrieve the most recent epoch number from saved model files.

    Parameters
    ----------
    GS
        GAN settings dictionary containing key 'model_path' where model checkpoints are stored.

    Returns
    -------
    int
        The largest epoch number corresponding to the most recently saved model checkpoint.

    Raises
    ------
    FileNotFoundError
        If `GS["model_path"]` does not exist.
    ValueError
        If no valid checkpoint files (starting with `'D'` followed by a number) are found.
    """
    
    # Get valid path where model checkpoints are saved 
    model_path = GS.get("model_path")
    if not model_path or not os.path.isdir(model_path):
        raise FileNotFoundError("Model directory not found: {}".format(model_path))

    # Get all model epoch saves as list
    fls = []
    for fl in os.listdir(model_path):
        if fl.startswith("D") and fl[1:].isdigit():
            fls.append(int(fl[1:]))

    if not fls:
        raise ValueError("No valid checkpoint files found in {} (expected filenames like 'D10', 'D50', etc.)".format(model_path))

    # return highest epoch
    fls = np.sort(fls)
    return int(fls[-1])


class Generator(nn.Module): 
    """
    GAN generator for producing molecular features conditioned on atomic labels.

    Parameters
    ----------
    info : dict
        Dataset information dictionary containing:
        - 'N_set': int, number of distincy molecule types (seperate datasets)
        - 'N_feats': int, number of features per atom
        - 'N_all_species': int, number of species across all molecule types
        - 'atomic_labels': list of atomic species label arrays
        
    GS : dict
        GAN settings dictionary containing:
        - 'l_fc': whether to use FC layer for labels
        - 'depth': number of hidden layers
        - 'device': torch device (cpu/cuda)
        - 'N_z': latent vector size
        - 'N_units_G': hidden units for generator
        - 'N_units_l': hidden units for label FC
    """
    
    def __init__(self, info: dict, GS: dict):
        super().__init__()
        self.N_set = info["N_set"]    
        self.N_feats = info['N_feats']
        self.N_species = info["N_all_species"]
        self.atomic_labels = info['atomic_labels']
        
        self.l_fc = GS['l_fc']
        self.depth = GS['depth']
        self.device = GS['device']
        self.N_z = GS["N_z"]
        self.N_units = GS['N_units_G']
        self.N_units_l = GS['N_units_l']
        
        self.act = nn.ReLU()
        
        # Input layer depending on whether to pass l through FC
        if self.l_fc:
            self.fc_l1 = nn.Linear(self.N_set, self.N_units_l, bias=True)
            self.fc_g1 = nn.Linear(self.N_z + self.N_units_l, self.N_units, bias=True)
        else:
            self.fc_g1 = nn.Linear(self.N_z + self.N_set, self.N_units, bias=True)
            
        # Define hidden/output layers for each depth level
        if self.depth == 0:
            self.fc_g2 = nn.Linear(self.N_units, self.N_species*self.N_feats, bias=True)
            
        elif self.depth == 1:
            self.fc_g2 = nn.Linear(self.N_units, 2*self.N_units, bias=True)
            self.fc_g3 = nn.Linear(2*self.N_units, self.N_species*self.N_feats, bias=True)
            
        elif self.depth == 2:
            self.fc_g2 = nn.Linear(self.N_units, 2*self.N_units, bias=True)
            self.fc_g3 = nn.Linear(2*self.N_units, 4*self.N_units, bias=True)
            self.fc_g4 = nn.Linear(4*self.N_units, self.N_species*self.N_feats, bias=True)

    def forward(self, z: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the generator.

        Parameters
        ----------
        z
            Latent vector tensor of shape (N_batch, N_z)
        l
            Atomic labels tensor of shape (N_atoms,)

        Returns
        -------
        torch.Tensor
            Generated features of shape (N_species, N_batch, N_feats)
            
        Raises
        ------
        ValueError
            If GS['depth'] is not one of {0, 1, 2}.
        ValueError
            If the inputted labels `l` do not match any atomic label set contained in info dictionary.
        """
    
        # Defining useful variables from inputted shapes
        N_batch = z.shape[0]
        N_at = len(l)
        
        print_flag = 0
        if print_flag:
            print('\n----------------------\nGenerator\n----------------------')
            print("l.shape",l.shape)
            print("l",l)
            print("z.shape",z.shape)
            print('N_batch',N_batch)
            print('N_at',N_at)

        # Convert L to idx in Emb Vocab - Molecules Types not Atom Species are Words in vocab 
        for i_set, atomic_label in enumerate(self.atomic_labels): 
            atomic_label = torch.tensor(atomic_label, dtype=torch.long).to(self.device)
            if len(atomic_label) == N_at:
                if torch.all(l == atomic_label):
                    l_idx = torch.tensor(i_set, dtype=torch.long)  
                    break
        else:
            raise ValueError("Label `l` does not match any atomic label set.")
        
        # One-hot encode L and change to correct shape
        l_emb = torch.eye(self.N_set)[l_idx].to(self.device).repeat(N_batch, 1) # change shape from [N_emb_l] to [N_batch,N_emb_l]
        if self.l_fc:
            l_emb = self.fc_l1(l_emb) # Optional pass L through FC Layer
        l = l_emb
            
        if print_flag:
            print("z.shape",z.shape)
            print("l[0]",l[0])
            print("l.shape",l.shape)
            print("cat((z,l), 1)[0]",torch.cat((z,l), 1)[0])
            print("cat((z,l), 1).shape",torch.cat((z,l), 1).shape)
            
        # Pass [z,l] through FC layers
        x = torch.cat((z, l_emb), dim=1)

        if self.depth == 0:
            x = self.act(self.fc_g1(x))
            x = self.fc_g2(x)
        elif self.depth == 1:
            x = self.act(self.fc_g1(x))
            x = self.act(self.fc_g2(x))
            x = self.fc_g3(x)
        elif self.depth == 2:
            x = self.act(self.fc_g1(x))
            x = self.act(self.fc_g2(x))
            x = self.act(self.fc_g3(x))
            x = self.fc_g4(x)
        else:
            raise ValueError("Invalid generator depth: {}. Must be one of [0, 1, 2].".format(self.depth))

        x = x.view(N_batch, self.N_species, self.N_feats).permute((1, 0, 2))

        return x
                
                
