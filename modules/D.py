#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from modules import config
        
def gradient_penalty(
    nn_D: nn.Module, 
    real_b: torch.Tensor, 
    real_l: torch.Tensor, 
    fake_b: torch.Tensor
) -> torch.Tensor:
    """
    Compute the gradient penalty for WGAN-GP.

    Parameters
    ----------
    nn_D
        The discriminator network.
    real_b
        Real batch tensor of shape (N_atoms, N_batch, N_feats).
    real_l
        Conditioning labels for the batch.
    fake_b
        Generated batch tensor of shape (N_atoms, N_batch, N_feats).

    Returns
    -------
    torch.Tensor
        Scalar gradient penalty term.
    """
    
    device = "cuda:"+str(fake_b.get_device()) if torch.cuda.is_available() else "cpu"

    n_at, n_batch, n_feat = real_b.shape
    #torch.autograd.set_detect_anomaly(True)
    torch.autograd.set_detect_anomaly(False)
    
    # Interpolation between real and fake batches
    epsilon = torch.rand((1, n_batch, 1)).to(device)
    epsilon = epsilon.repeat(n_at, 1, n_feat)
    interpolated_b = real_b * epsilon + fake_b * (1 - epsilon)
    
    # Compute discriminator score for interpolated batch
    interpolated_score = nn_D(interpolated_b, real_l) # fake_l must be same as real_l ???
    
    # Compute gradients w.r.t. interpolated batch
    gradient = torch.autograd.grad(
        inputs = interpolated_b,
        outputs = interpolated_score,
        grad_outputs = torch.ones_like(interpolated_score).to(device),
        create_graph = True,
        retain_graph = True,
    )[0]

    gradient_norm = torch.sqrt(torch.sum(gradient ** 2, dim=(0, 2)) + 1e-12)
    #gradient_norm = torch.sqrt(torch.sum(gradient ** 2, dim=1) + 1e-12)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)

    return gradient_penalty
    
class Discriminator(nn.Module):
    """
    GAN discriminator for assessing molecular features conditioned on atomic labels.

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
        - 'N_units_D': hidden units for discriminator
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
        self.N_units = GS['N_units_D']
        self.N_units_l = GS['N_units_l']
        
        self.act = nn.LeakyReLU(0.2)

        # Whether to pass l through FC
        if self.l_fc:
            self.fc_l1 = nn.Linear(self.N_set, self.N_units_l, bias=True)
        else:
            self.N_units_l = self.N_set
            
        # Define layers for each depth level
        if self.depth == 0:
            self.fc_d1 = nn.Linear(self.N_species*self.N_feats + self.N_units_l, self.N_units, bias=True)
            self.fc_d2 = nn.Linear(self.N_units, 1, bias=True)
            
        elif self.depth == 1:
            self.fc_d1 = nn.Linear(self.N_species*self.N_feats + self.N_units_l, 2*self.N_units, bias=True)
            self.fc_d2 = nn.Linear(2*self.N_units, self.N_units, bias=True)
            self.fc_d3 = nn.Linear(self.N_units, 1, bias=True)
            
        elif self.depth == 2:
            self.fc_d1 = nn.Linear(self.N_species*self.N_feats + self.N_units_l, 4*self.N_units, bias=True)
            self.fc_d2 = nn.Linear(4*self.N_units, 2*self.N_units, bias=True)
            self.fc_d3 = nn.Linear(2*self.N_units, self.N_units, bias=True)
            self.fc_d4 = nn.Linear(self.N_units, 1, bias=True)
            
    def forward(self, b: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the discriminator.

        Parameters
        ----------
        b
            Descriptors tensor of shape (N_species, N_batch, N_feats)
        l
            Atomic labels tensor of shape (N_atoms,)

        Returns
        -------
        torch.Tensor
            Scores quantifying how real or fake the inputted samples are - tensor of shape (N_batch, 1)
            
        Raises
        ------
        ValueError
            If GS['depth'] is not one of {0, 1, 2}.
        ValueError
            If the inputted labels `l` do not match any atomic label set contained in info dictionary.
        """
        
        # Defining useful variables from inputted shapes
        N_species = b.shape[0]
        N_batch = b.shape[1]
        N_at = len(l)
        
        print_flag = 0
        if print_flag:
            print('\n----------------------\nDiscriminator\n----------------------')
            print("b.shape",b.shape)
            print("N_species",N_species)
            print("N_at",N_at)
            print("N_batch",N_batch)
            print("l.shape",l.shape)
        
        # Change B from [N_at, N_batch, N_feats] to [N_batch, N_at*N_feats]
        b = b.permute((1,0,2))
        b = torch.flatten(b, start_dim=1)

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
            print("b.shape",b.shape)
            print("l[0]",l[0])
            print("l.shape",l.shape)
            print("cat((b,l), 1)[0]",torch.cat((b,l), 1)[0])
            print("cat((b,l), 1).shape",torch.cat((b,l), 1).shape)
        
        # Pass [b,l] through FC layers
        x = torch.cat((b,l), 1)
                
        if self.depth == 0:
            x = self.act(self.fc_d1(x))
            x = self.fc_d2(x)
        elif self.depth == 1:
            x = self.act(self.fc_d1(x))
            x = self.act(self.fc_d2(x))
            x = self.fc_d3(x)
        elif self.depth == 2:
            x = self.act(self.fc_d1(x))
            x = self.act(self.fc_d2(x))
            x = self.act(self.fc_d3(x))
            x = self.fc_d4(x)
        else:
            raise ValueError("Invalid generator depth: {}. Must be one of [0, 1, 2].".format(self.depth))
        
        x = x.view(N_batch, 1)
        
        return x
    
        
