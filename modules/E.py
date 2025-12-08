#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np

import torch
import torch.nn as nn

from modules import config

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
        If no valid checkpoint files (starting with `'E'` followed by a number) are found.
    """
    
    # Get valid path where model checkpoints are saved 
    model_path = GS.get("model_path")
    if not model_path or not os.path.isdir(model_path):
        raise FileNotFoundError("Model directory not found: {}".format(model_path))

    # Get all model epoch saves as list
    fls = []
    for fl in os.listdir(model_path):
        if fl.startswith("E") and fl[1:].isdigit():
            fls.append(int(fl[1:]))

    if not fls:
        raise ValueError("No valid checkpoint files found in {} (expected filenames like 'E10', 'E50', etc.)".format(model_path))

    # return highest epoch
    fls = np.sort(fls)
    return int(fls[-1])
    
class Energyator(nn.Module):
    """
    Energy prediction model for enforcing GAN energy alignment.

    Parameters
    ----------
    info : dict
        Dataset information dictionary containing:
        - 'N_feats': int, number of features per atom
        - 'N_all_species': int, number of species across all molecule types
        
    GS : dict
        GAN settings dictionary containing:
        - 'depth': number of hidden layers
        - 'device': torch device (cpu/cuda)
        - 'N_units_E': hidden units for eneryator
    """
    
    def __init__(self, info: dict, GS: dict):
        super().__init__()
        
        self.N_feats = info['N_feats']
        self.all_species = info["all_species"]
        self.N_species = info["N_all_species"]
        
        self.depth = GS['depth']
        self.device = GS['device']
        self.N_units = GS['N_units_E']
        
        self.species_dict = {s:i for i,s in enumerate(self.all_species)}
        
        self.drop = nn.Dropout(0.01)
        self.act = nn.LeakyReLU(0.1)
        self.out_act = nn.Sigmoid()

        # Define layers for each depth level
        if self.depth == 0:
            self.fc_e1 = nn.ModuleList([nn.Linear(self.N_feats, self.N_units, bias=True) for _ in range(self.N_species)])
            self.fc_e2 = nn.ModuleList([nn.Linear(self.N_units,            1, bias=True) for _ in range(self.N_species)])
                        
        elif self.depth == 1:
            self.fc_e1 = nn.ModuleList([nn.Linear(self.N_feats, self.N_units, bias=True) for _ in range(self.N_species)])
            self.fc_e2 = nn.ModuleList([nn.Linear(self.N_units, self.N_units, bias=True) for _ in range(self.N_species)])
            self.fc_e3 = nn.ModuleList([nn.Linear(self.N_units,            1, bias=True) for _ in range(self.N_species)])
                
        elif self.depth == 2:
            self.fc_e1 = nn.ModuleList([nn.Linear(self.N_feats, self.N_units, bias=True) for _ in range(self.N_species)])
            self.fc_e2 = nn.ModuleList([nn.Linear(self.N_units, self.N_units, bias=True) for _ in range(self.N_species)])
            self.fc_e3 = nn.ModuleList([nn.Linear(self.N_units, self.N_units, bias=True) for _ in range(self.N_species)])
            self.fc_e4 = nn.ModuleList([nn.Linear(self.N_units,            1, bias=True) for _ in range(self.N_species)])

    def forward(self, b: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the energy prediction model.

        Parameters
        ----------
        b
            Descriptors tensor of shape (N_species, N_batch, N_feats)

        Returns
        -------
        torch.Tensor
            Predicted energies tensor of shape (N_batch, 1)
        """
        
        # Defining useful variables from inputted shapes
        N_species = b.shape[0]
        N_batch = b.shape[1]

        print_flag = 0
        if print_flag:
            print('\n----------------------\nEnergyator\n----------------------')
            print("b.shape",b.shape)
            print("N_species",N_species)
            print("N_batch",N_batch)
            print("self.species_dict",self.species_dict)
            print('----------------------\n')
            
        # Empty tensor to collect atomic energies predictions
        A = torch.zeros((N_species,N_batch,1)).to(self.device)
        
        for i in range(N_species):
            k = i # assumes that all inputted samples are in the same species ordering
            
            if self.depth == 0:
                x = self.drop(self.act(self.fc_e1[k](b[i])))
                A[i] = self.fc_e2[k](x)
            elif self.depth == 1:
                x = self.drop(self.act(self.fc_e1[k](b[i])))
                x = self.drop(self.act(self.fc_e2[k](x)))
                A[i] = self.fc_e3[k](x)
            elif self.depth == 2:
                x = self.drop(self.act(self.fc_e1[k](b[i])))
                x = self.drop(self.act(self.fc_e2[k](x)))
                x = self.drop(self.act(self.fc_e3[k](x)))
                A[i] = self.fc_e4[k](x)
            else:
                raise ValueError("Invalid generator depth: {}. Must be one of [0, 1, 2].".format(self.depth))
        
        return self.out_act(A.sum(0))

        
