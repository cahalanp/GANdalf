#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import tqdm
import rmsd
import copy
import torch
import shutil
import numpy as np
import seaborn as sns
from pathlib import Path

from cobe import Dataset, Datadict
from cobe.descriptors import Bispectrum
from cobe.descriptors import compute_bispectrum
from ase import Atoms
from ase.io import Trajectory, write
from ase.visualize import view
import matplotlib as matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
from sklearn.decomposition import PCA
from random import shuffle

from pyscf import gto, dft, scf

from modules import config
from modules import G
from gan import dataset as data

torch.set_default_tensor_type(torch.DoubleTensor)

def generate_like_train(num_mols, set_inds, nn_G, info, GS, G_conv=False, G_comp=False):
    """
    = Function that generates specified number of samples from specified generator
    
    INPUTS:
        1) num_mols - [int] number of samples to generate
        2) set_inds - list of [int] of indexes of datasets in info to generate - set to [0] if only 1 mol in train set
        3) nn_G - generator
        4) info - combined info dictionary
        5) GS - gan training settings dictionary
        6) G_conv - [bool] for whether layers of nn_G are conv (=T) or fc (=F)
        7) G_comp - [bool] for whether input of G.make_seeds is z,l (=T) or z (=F)
    
    OUTPUTS:
        1) B - list of [tensor] of shape [N_set][N_mols,N_at,N_feat] 
        2) L - list of [tensor] of shape [N_set][N_mols,N_at] 
    """
    b_gen, l_gen = [], []
        
    num_batch = num_mols // GS['N_batch']
    remainder_mols = num_mols % GS['N_batch']
    nn_G.eval()

    # for each set
    for set_ind in set_inds:
    
        # Empty tensors for B and L
        b_set = torch.zeros((num_mols, info['N_all_species'], info['N_feats']))
        l_set = torch.zeros((num_mols, info['N_at'][set_ind]), dtype=torch.long)

        # for each batch
        for i_batch in range(num_batch+1):
        
            # Make seeds for nn_G
            if G_comp:
                z, _l = G.make_seeds(info, GS, l=torch.tensor(info['atomic_labels'][set_ind]))
            else:
                z = G.make_seeds(info, GS)
                _l = torch.tensor(info['atomic_labels'][set_ind])
                
            # Generate a batch of samples - b.shape=[N_batch,N_at,N_feat] and l.shape=[N_batch,N_at] 
            # if no comp info then change from [N_at,N_batch,N_b]
            if G_comp == False and G_conv == False:
                _b = nn_G(z).permute((1,0,2)).detach()
            # if nn_G is conv then remove colour channel
            elif G_conv == True and G_comp == False:
                _b = nn_G(z).squeeze(1).detach()
            # input comp info and change shape
            elif G_conv == False and G_comp == True:
                _b = nn_G(z, _l).permute((1,0,2)).detach()
            # input comp info and remove colour channel
            elif G_conv == True and G_comp == True:
                _b = nn_G(z, _l).squeeze(1).detach()
            # repeat L changing from [N_at] to [N_batch,N_at]
            _l = _l.repeat(_b.shape[0], 1).detach()
            
            """
            # remove null mols from output of nn_G 
            if info['N_at'][set_ind] < max(info['N_at']):
                print('Removing Null Mols from nn_G output')
                _b_new = []
            
                for i_at in range(_b.shape[1]):
                    if len(torch.unique(_b[:,i_at])) != 1:
                        _b_new.append(_b[:,i_at])
                _b = torch.stack(_b_new).permute((1,0,2))
            """
            
            # convert i_batch into mol_inds
            if i_batch != num_batch:
                mols_inds = torch.arange(i_batch*GS['N_batch'], (i_batch+1)*GS['N_batch'], dtype=torch.long)

            # if its the last batch also shorten generated batch to remainder_mols
            else:
                _b = _b[:remainder_mols]
                _l = _l[:remainder_mols]
                mols_inds = torch.arange(i_batch*GS['N_batch'], i_batch*GS['N_batch'] + remainder_mols, dtype=torch.long)
            
            b_set[mols_inds] = _b
            l_set[mols_inds] = _l
            
        b_gen.append(b_set)
        l_gen.append(l_set)
        
    return b_gen, l_gen

def get_train(num_mols, set_inds, dataset, info, GS):
    """
    = Function that gets specified number of samples from specified datasets
    
    INPUTS:
        1) num_mols - [int] number of training molecules to return
        2) set_inds - list of [int] of indexes of datasets in info to generate - set to [0] if only 1 mol in train set
        3) dataset - custom PyTorch dataset class created with data.BL_DataSet()
        4) info - combined info dictionary
        5) GS - gan training settings dictionary
        
    OUTPUTS:
        1) B - list of tensors of shape [N_set][N_mols,N_at,N_feat] 
        2) L - list of tensors of shape [N_set][N_mols,N_at] 
    """
    b, l = [], []
        
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

        batch_inds_set = batch_inds[i_set]
        
        for i_batch in range(len(batch_inds_set)):
            # Get a batch of random training samples - b.shape=[N_batch,N_at,N_feat] and l.shape=[N_batch,N_at] 
            _b, _l = dataset.__getitem__(batch_inds_set[i_batch])
            print('_b.shape',_b.shape)
            _b = _b.permute((1,0,2))
            print('_b.shape',_b.shape)
            _l = _l.repeat(_b.shape[0], 1)
            
            # convert i_batch into mol_inds
            if i_batch != len(batch_inds[i_set]) - 1:
                mols_inds = torch.arange(i_batch*GS['N_batch'], (i_batch+1)*GS['N_batch'], dtype=torch.long)

            # if its the last batch also shorten batch to remainder_mols
            else:
                print('_b.shape',_b.shape)
                _b = _b[:remainder_mols]
                print('_b.shape',_b.shape)
                _l = _l[:remainder_mols]
                mols_inds = torch.arange(i_batch*GS['N_batch'], i_batch*GS['N_batch'] + remainder_mols, dtype=torch.long)
                
            b_set[mols_inds] = _b
            l_set[mols_inds] = _l
            
        b.append(b_set)
        l.append(l_set)
    
    return b, l
    
    
def get_ideal_b(l, info, GS):
    """
    = function for returning ideal B for a given L composition 
    
    INPUTS:
        1) l - list of tensors of int labels for atoms in b - shape [N_set][N_mol,N_at]
        2) info - combined info dictionary
        3) GS - gan training settings dictionary
        p
    OUTPUTS:    
        1) b - list of list of tensors - shape [N_set][1,N_at,N_b]
    """
    
    # Get name of molecule for set of first molecule's l
    set_name = config.mol_comp[tuple(sorted(l[0].cpu().detach().numpy()))]
    
    # Get set index
    set_num = info['set_names'].index(set_name)
    
    ideal_b = torch.tensor(np.load(GS['data_paths'][set_num] + 'ideal_B.npy'))
    
    return ideal_b
    
def species_seperator(b, l, info):
    """
    = Function to return B for each species - used primarily in plotting
    
    INPUTS:
        1) b - list of tensors of shape [N_set][N_mol,N_at,N_b]
        2) l - list of tensors of shape [N_set][N_mol,N_at]
        3) info - combined info dictionary
        
    OUTPUTS:
        1) b - list of list of tensors of shape [N_set][N_species][N_mol,N_at_species,N_b]        
        
    NOTES:
        - species outputted should be ordered in increasing weight e.g. (0,1,2,3)
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
        
def species_seperator2(b, l, info):
    """
    = Function to return B for each species - used primarily in plotting
    
    INPUTS:
        1) b - list of tensors of shape [N_set][N_mol,N_species,N_b]
        2) l - list of tensors of shape [N_set][N_mol,N_species]
        3) info - combined info dictionary
        
    OUTPUTS:
        1) b - list of list of tensors of shape [N_set][N_species][N_mol,1,N_b]        
        
    NOTES:
        - species outputted should be ordered in increasing weight e.g. (0,1,2,3)
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
            N_at_species = 1
            b_mol = torch.zeros((N_mol, N_at_species, info['N_feats']))
                                
            # For each sample of mol in input
            for i_mol in range(N_mol):    
                species_ind = info['all_species'].index(species)

                b_mol[i_mol] = b[i_set][i_mol][species_ind]
            b_species.append(b_mol)                
        b_set.append(b_species)
    
    return b_set
    
def sum_over_species(b, l, info):
    """
    = Function that sums B over similiar species
    
    INPUTS:
        1) b - list of tensors of shape [N_set][N_mol,N_at,N_b]
        2) l - list of tensors of shape [N_set][N_mol,N_at]
        3) info - combined info dictionary
        
    OUTPUTS:
        1) b - list of list of tensors of shape [N_set][N_species][N_mol,N_at_species,N_b]        
        
    NOTES:
        - species outputted should be ordered in increasing weight e.g. (0,1,2,3)
    """
    N_set = len(b)
    b_new = []
    
    # For each dataset
    for i_set in range(N_set):
        
        N_mol = len(b[i_set])
        N_all_species = info['N_all_species']

        b_set = np.zeros((N_mol, N_all_species, info['N_feats']))
                        
        for i_species, species in enumerate(info['species'][i_set]):
            labels = l[i_set][0] # assume L is same for all mols
            species_inds = [i for i, label in enumerate(labels) if label == species]
            
            print('info[species][i_set]',info['species'][i_set])
            print('labels',labels)
            print('species_inds',species_inds)
            print('b_set[0,species].shape',b_set[0,species].shape)
            print('b[i_set][0,species_inds].squeeze(0).shape',b[i_set][0,species_inds].squeeze(0).shape)
            
            for species_ind in species_inds:
                print('b[i_set][0,species_ind].shape',b[i_set][0,species_ind].shape)
                print('b[i_set][0,species_ind].squeeze(0).shape',b[i_set][0,species_ind].squeeze(0).shape)
                b_set[:,species] += b[i_set][:][species_ind].squeeze(0)

        b_new.append(b_set)
        
    return b_new
    
def combine_mean_and_std(b_means, b_stds, n_samples):
    """
    = Function that calculates the weighted mean and standard deviation for a combined dataset
      from the means and standard devaition of the individual datasets to be combined. 

    INPUTS:
        1) b_means - list of tensors of shape [N_set][N_b]
        2) b_stds - list of tensors of shape [N_set][N_b]
        3) n_samples - tensor of ints of shape [N_set]
            = number of samples used in calculating mean and std (e.g. N_mol_species*N_at)
        
    OUTPUTS:
        1) b_mean - combined mean tensors of shape [N_b]
        2) b_std - combined std tensors of shape [N_b]
            
    """
    # Calculate Mean and Std of joined Datasets for each species
    # empty arrays to append to 
    
    # convert list of tensor to tensor to be able to use fancy indexing
    b_means = torch.stack(b_means)
    b_stds = torch.stack(b_stds)
    
    # set N_b as variable instead of explicitly as 55 - better practice
    n_features = b_means[0].shape[-1]
    
    # empty tensors to append to and to return
    b_mean = torch.zeros((n_features,))    
    b_std = torch.zeros((n_features,))
    
    # dtype of n_samples is converted to dtype of b_means so it can be multiplied in torch.dot(...)
    n_samples = torch.tensor(n_samples, dtype=b_means[0].dtype)
        
    # for each feature
    for i_b in range(n_features):
        # calc combined mean (weighted mean)
        b_mean[i_b] = torch.dot(n_samples, b_means[:,i_b]) / torch.sum(n_samples)
        
        # calc combined std
        q1 = b_stds[:,i_b]**2
        q2 = (b_means[:,i_b] - b_mean[i_b])**2
                    
        b_std[i_b] = torch.sqrt(torch.dot(n_samples, q1+q2) / torch.sum(n_samples))
    
    return b_mean, b_std
    
    
def image_b(info, GS, b, b_g=None, save=True, show=False):
    """
    = Function that plots a single real (and a single generated) B as an image of dimension [N_at,N_feat] 
    
    INPUTS:
        1) info - combined info dictionary
        2) GS - gan training settings dictionary
        3) b - single molecule tensors of shape [N_at,N_b]
        4) b_g - single molecule tensors of shape [N_at,N_b]
        5) save - [bool] whether to save plot to path '.../gandalf/gan/plots/'
        6) show - [bool] whether to show plots immediately
        
    OUTPUTS:
        - colourmap image vizualisation of B (and B_g)  
    
    NOTES:
        - called every 0.1 epochs and saved to path '.../gandalf/gan/savednet/.../plots' 
        - if b_g are also supplied (e.g. manually in plot.py) then images are saved to path '.../gandalf/gan/plots'  
    """
    
    plt.figure()
    # only 1 subplot if b_g not supplied else 2 subplots
    num_subplots = 1 if b_g is None else 2
    subplot_names = {0:'Real',1:'Fake'} # dict for conversion: i_plot -> name 
    
    # plot real and fake images side by side as subplots
    for i_plot in range(num_subplots):
        plt.subplot(121+i_plot)
        plt.axis('off')
        plt.title(subplot_names[i_plot])
        if i_plot == 0:
            plt.imshow(b)
        else:
            plt.imshow(b_g)
                        
    if save:
        now = datetime.now()
        time_date = now.strftime('(%H%M_%d%m)')
        dir_path = GS['model_path'] + '/plots/'
        
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        if b_g is None:
            plt.savefig('plots/image_b_'+str(time_date)+'.png')
        else:
            plt.savefig(dir_path+str(time_date)+'.png')
        plt.close()
        
    if show:
        plt.show()
    
        
def images_b(info, GS, b, b_g=None, max_rows=40, inv_fs=False, save=True, show=False):
    """
    = Function that plots a single real (and a single generated) B as an image of dimension [N_at,N_feat] 
    
    INPUTS:
        1) info - combined info dictionary
        2) GS - gan training settings dictionary
        3) b - arrays of shape [N_mol,N_at,N_b]
        4) b_g - arrays of shape [N_mol,N_at,N_b]
        5) save - [bool] whether to save plot to path '.../gandalf/gan/plots/'
        6) show - [bool] whether to show plots immediately
        
        
    OUTPUTS:
        - colourmap image vizualisation of B (and B_g)  
    
    NOTES:
        - called every 0.1 epochs and saved to path '.../gandalf/gan/savednet/.../plots' 
        - if b_g are also supplied (e.g. manually in plot.py) then images are saved to path '.../gandalf/gan/plots'  
    """
    n_mol, n_at, n_feats = b.shape
    
    b = b.permute((1,0,2)).reshape(-1, n_feats) 
    if b_g is not None:
        b_g = b_g.permute((1,0,2)).reshape(-1, n_feats) 
        
    # Get reasonable max num of rows in images
    n_mols = max_rows // n_at
    
    b = b[:n_mols*n_at]
    if b_g is not None:
        b_g = b_g[:n_mols*n_at]
        
    # Create new array with NaNs padding in between rows
    new_b_row = n_mols*n_at+n_mols-1
    new_b = np.empty((new_b_row, n_feats)) * np.nan
    if b_g is not None:
        new_b_g = np.empty((new_b_row, n_feats)) * np.nan

    i_mol = 0
    pad_rows = []
    
    # Get indexes of rows to be padded
    for i_row in range(new_b_row):
        if i_row % n_at == 0 and i_row != 0:
            i_mol += 1
        pad_rows.append(i_mol*n_at + (i_mol-1))

    i_mol = 0
    
    # Copy old B to new B for every row except padded ones
    for i_row in range(new_b_row):
        i_old_row = i_row-i_mol
        
        if i_row in pad_rows:
            i_mol += 1
            pass
        else:
            new_b[i_row] = b[i_old_row]
            if b_g is not None:
                new_b_g[i_row] = b_g[i_old_row]
    
    plt.figure()
    # only 1 subplot if b_g not supplied else 2 subplots
    num_subplots = 1 if b_g is None else 2
    subplot_names = {0:'Real',1:'Fake'} # dict for conversion: i_plot -> name 
    
    # plot real and fake images side by side as subplots
    for i_plot in range(num_subplots):
        plt.subplot(121+i_plot)
        plt.axis('off')
        plt.title(subplot_names[i_plot])

        current_cmap = matplotlib.cm.get_cmap()
        current_cmap = current_cmap.set_bad(color='white')

        if i_plot == 0:
            plt.imshow(new_b, cmap=current_cmap)
        else:
            plt.imshow(new_b_g, cmap=current_cmap)
                        
    if save:
        now = datetime.now()
        time_date = now.strftime('(%H%M_%d%m)')
        dir_path = GS['model_path'] + '/plots/'
        
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        if b_g is None:
            plt.savefig('plots/images_b_'+str(time_date)+'.png')
        else:
            plt.savefig(dir_path+str(time_date)+'.png')
        plt.close()
        
    if show:
        plt.show()
    
def images_b2(info, GS, b, l, b_g=None, l_g=None, max_rows=40, inv_fs=False, save=True, show=False):
    """
    = Function that plots a single real (and a single generated) B as an image of dimension [N_at,N_feat] 
    
    INPUTS:
        1) info - combined info dictionary
        2) GS - gan training settings dictionary
        3) b - arrays of shape [N_mol,N_at,N_b]
        4) b_g - arrays of shape [N_mol,N_at,N_b]
        5) save - [bool] whether to save plot to path '.../gandalf/gan/plots/'
        6) show - [bool] whether to show plots immediately
        
    OUTPUTS:
        - colourmap image vizualisation of B (and B_g)  
    
    NOTES:
        - called every 0.1 epochs and saved to path '.../gandalf/gan/savednet/.../plots' 
        - if b_g are also supplied (e.g. manually in plot.py) then images are saved to path '.../gandalf/gan/plots'  
    """
    
    # if l is tensor -> make lists
    if type(l) == type(torch.tensor(0)) and type(b) == type(torch.tensor(0)):
        # if l is only for single mol in batch -> repeat (assumes first dim of b is N_mol)
        if l.dim() == 1:
            l = l.repeat(b.shape[0], 1)
            if b_g is not None:
                l_g = l_g.repeat(b_g.shape[0], 1)
            
        l = [l]
        b = [copy.deepcopy(b.detach())]
        
        if b_g is not None:
            l_g = [l_g]
            b_g = [copy.deepcopy(b_g.detach())]
        
    for i_set, l_set in enumerate(l):
        set_name = config.mol_comp[tuple(sorted(l_set[0].cpu().detach().numpy()))]
        set_ind = info['set_names'].index(set_name)
        
        _b, _l = b[i_set], l[i_set]
        if b_g is not None:
            _b_g, _l_g = b_g[i_set], l_g[i_set]
        
        # Handy Shape Parameters
        n_mol, n_at, n_feats = _b.shape
            
        # Inverse Feature Scaling
        if inv_fs:
            _b = data.inv_fs(_b, _l, info, GS)
            if b_g is not None:
                _b_g = data.inv_fs(_b_g, _l_g, info, GS)
        
        # Change from [N_mol,N_at,N_b] [N_mol*N_at,N_b]
        _b = _b.reshape(-1, n_feats) 
        if b_g is not None:
            _b_g = _b_g.reshape(-1, n_feats) 
            
        # Get reasonable max num of rows in images
        n_mols = max_rows // n_at
        
        _b = _b[:n_mols*n_at].cpu()
        if b_g is not None:
            _b_g = _b_g[:n_mols*n_at].cpu()
            
        # Create new array with NaNs padding in between rows
        new_b_row = n_mols*n_at+n_mols-1
        new_b = np.empty((new_b_row, n_feats)) * np.nan
        if b_g is not None:
            new_b_g = np.empty((new_b_row, n_feats)) * np.nan

        i_mol = 0
        pad_rows = []
        
        # Get indexes of rows to be padded
        for i_row in range(new_b_row):
            if i_row % n_at == 0 and i_row != 0:
                i_mol += 1
            pad_rows.append(i_mol*n_at + (i_mol-1))

        i_mol = 0
        
        # Copy old B to new B for every row except padded ones
        for i_row in range(new_b_row):
            i_old_row = i_row-i_mol
            
            if i_row in pad_rows:
                i_mol += 1
                pass
            else:
                new_b[i_row] = _b[i_old_row]
                if b_g is not None:
                    new_b_g[i_row] = _b_g[i_old_row]
        
        plt.figure()
        # only 1 subplot if b_g not supplied else 2 subplots
        num_subplots = 1 if b_g is None else 2
        plt.suptitle('B for '+set_name)
        subplot_names = {0:'Real',1:'Fake'} # dict for conversion: i_plot -> name 
        
        # plot real and fake images side by side as subplots
        for i_plot in range(num_subplots):
            plt.subplot(121+i_plot)
            plt.title(subplot_names[i_plot])
            plt.axis('off')

            current_cmap = matplotlib.cm.get_cmap().copy()
            current_cmap = current_cmap.set_bad(color='white')


            if i_plot == 0:
                plt.imshow(new_b, cmap=current_cmap)
            else:
                plt.imshow(new_b_g, cmap=current_cmap)
            
        
        # If no b_g then save to .../gan/plots/ otherwise save to .../gan/savednet/                
        if save:
            now = datetime.now()
            time_date = now.strftime('(%H%M_%d%m)')
            dir_path = GS['model_path'] + '/plots/'
            
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            if b_g is None:
                plt.savefig('plots/images_b_'+str(time_date)+'_'+set_name+'.png')
            else:
                plt.savefig(dir_path+str(time_date)+'_'+set_name+'.png')
            plt.close()
            
        if show:
            plt.show()

def plot_e(info, GS, E_model, b, l, e, b_g=None, l_g=None, e_g=None, save=True, show=True):
    """
    = Function that plots the model predicted energy vs labelled energies for generated 
      data against the real energy data
    
    INPUTS:
        1) info - combined info dictionary
        2) GS - gan training settings dictionary
        3) b - list of list of tensors of shape [N_set][N_mol,N_species,N_b]
        
    OUTPUTS:
    
    NOTES:
        - must use .permute((1,0,2)) on inputted b if called during training inner looop
    """
            
    # Get Name of Plot
    print('l.shape',l.shape)
    set_name = config.mol_comp[tuple(sorted(l.cpu().detach().numpy()))]
    print('set_name',set_name)
    print('b.shape',b.shape)
    print('b_g.shape',b_g.shape)
    
    """
    print('b.shape',b.shape)
    print('b_g.shape',b_g.shape)
    plt.figure()
    plt.subplot(121)
    for i in range(128):
        for j in range(1):
            plt.plot(range(info['N_feats']), b[i,j], linestyle='dotted')
            plt.plot(range(info['N_feats']), b_g[i,j], linestyle='dashed')
    plt.subplot(122)
    for i in range(128):
        for j in range(1):
            plt.plot(range(info['N_feats']), b[i,1+j], linestyle='dotted')
            plt.plot(range(info['N_feats']), b_g[i,1+j], linestyle='dashed')
    plt.show()
    
    """
    
    # Inverse FS and reshape from [N_mol,N_species,N_b] to [N_mol,N_species*N_b]
    #b = data.inv_fs(b, l, info, GS)
    b = b.reshape((-1,info['N_all_species']*info['N_feats']))
    
    if b_g is not None:
        #b_g = data.inv_fs(b_g, l_g, info, GS)
        b_g = b_g.reshape((-1,info['N_all_species']*info['N_feats']))
        
    # Predict E with model
    ep = E_model.predict(b.detach().numpy())
    ep = torch.tensor(ep)
    #ep = data.inv_fs_e(ep, l, info, GS)
    
    if b_g is not None:
        ep_g = E_model.predict(b_g.detach().numpy())
        ep_g = torch.tensor(ep_g)
        #ep_g = data.inv_fs_e(ep_g, l_g, info, GS)

    """
    print('b.shape',b.shape)
    print('b_g.shape',b_g.shape)
    plt.figure()
    for i in range(128):
        plt.plot(range(info['N_all_species']*info['N_feats']), b[i], linestyle='dotted')
        plt.plot(range(info['N_all_species']*info['N_feats']), b_g[i], linestyle='dashed')
    plt.show()
    
    print('b.shape',b.shape)
    print('b_g.shape',b_g.shape)
    plt.figure()
    plt.plot(range(info['N_all_species']*info['N_feats']), b[0], linestyle='dotted')
    plt.plot(range(info['N_all_species']*info['N_feats']), b_g[0], linestyle='dashed')
    plt.show()
    """
    
    #e = data.inv_fs_e(e, l, info, GS)
    e = e.squeeze(1)
    if b_g is not None:
        #e_g = data.inv_fs_e(e_g, l_g, info, GS)
        e_g = e_g.squeeze(1)
    
    print('e.shape',e.shape)
    print('ep.shape',ep.shape)
    print('\ne',e)
    print('\nep',ep)
    if b_g is not None:
        print('e_g.shape',e_g.shape)
        print('ep_g.shape',ep_g.shape)
        print('\ne_g',e_g)
        print('\nep_g',ep_g)

    plt.figure()
    plt.scatter(e, ep, label='Real', c='red', s=150, alpha=0.5)
    if b_g is not None:
        plt.scatter(e_g, ep_g, label='Fake', c='blue', s=150, alpha=0.5)
    
    plt.plot(e, e, linestyle=':', c='red')

    plt.xlabel('$E_{DFT}$', fontsize=15)
    plt.ylabel('$E_{pred}$', fontsize=15)
    plt.title('Plot of $E_{DFT}$ vs $E_{pred}$ using SNAP', fontsize=20)
    plt.legend()

    # If no b_g then save to .../gan/plots/ otherwise save to .../gan/savednet/                
    if save:
        now = datetime.now()
        time_date = now.strftime('(%H%M_%d%m)')
        dir_path = GS['model_path'] + '/plots/'
        
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        if b_g is None:
            plt.savefig('plots/plot_e_'+str(time_date)+'_'+set_name+'.png', bbox_inches='tight')
        else:
            plt.savefig(dir_path+'plot_e_'+str(time_date)+'_'+set_name+'.png', bbox_inches='tight')
        plt.close()
        
    if show:
        plt.show()

def plot_e_nn(info, GS, E_model, b, l, e, b_g=None, l_g=None, e_g=None, save=True, show=True):
    """
    = Function that plots the model predicted energy vs labelled energies for generated 
      data against the real energy data
    
    INPUTS:
        1) info - combined info dictionary
        2) GS - gan training settings dictionary
        3) b - list of list of tensors of shape [N_set][N_mol,N_species,N_b]
        
    OUTPUTS:
    
    NOTES:
        - must use .permute((1,0,2)) on inputted b if called during training inner looop
    """
            
    # Get Name of Plot
    #print('l.shape',l.shape)
    set_name = config.mol_comp[tuple(sorted(l.cpu().detach().numpy()))]
    #print('set_name',set_name)
    #print('b.shape',b.shape)
    #print('b_g.shape',b_g.shape)
    
    """
    print('b.shape',b.shape)
    print('b_g.shape',b_g.shape)
    plt.figure()
    plt.subplot(121)
    for i in range(128):
        for j in range(1):
            plt.plot(range(info['N_feats']), b[i,j], linestyle='dotted')
            plt.plot(range(info['N_feats']), b_g[i,j], linestyle='dashed')
    plt.subplot(122)
    for i in range(128):
        for j in range(1):
            plt.plot(range(info['N_feats']), b[i,1+j], linestyle='dotted')
            plt.plot(range(info['N_feats']), b_g[i,1+j], linestyle='dashed')
    plt.show()
    
    """
    # Predict E with model
    ep = E_model(b).cpu().detach().numpy()
    #ep = data.inv_fs_e(ep, l, info, GS)
    
    if b_g is not None:
        ep_g = E_model(b_g).cpu().detach().numpy()
        #ep_g = data.inv_fs_e(ep_g, l_g, info, GS)

    """
    print('b.shape',b.shape)
    print('b_g.shape',b_g.shape)
    plt.figure()
    for i in range(128):
        plt.plot(range(info['N_all_species']*info['N_feats']), b[i], linestyle='dotted')
        plt.plot(range(info['N_all_species']*info['N_feats']), b_g[i], linestyle='dashed')
    plt.show()
    
    print('b.shape',b.shape)
    print('b_g.shape',b_g.shape)
    plt.figure()
    plt.plot(range(info['N_all_species']*info['N_feats']), b[0], linestyle='dotted')
    plt.plot(range(info['N_all_species']*info['N_feats']), b_g[0], linestyle='dashed')
    plt.show()
    """
    
    #e = data.inv_fs_e(e, l, info, GS)
    e = e.squeeze(1)
    ep = ep.squeeze(1)
    if b_g is not None:
        #e_g = data.inv_fs_e(e_g, l_g, info, GS)
        e_g = e_g.squeeze(1)
        ep_g = ep_g.squeeze(1)
    
    """
    print('e.shape',e.shape)
    print('ep.shape',ep.shape)
    print('\ne[:5]',e[:5])
    print('\nep[:5]',ep[:5])
    if b_g is not None:
        print('e_g.shape',e_g.shape)
        print('ep_g.shape',ep_g.shape)
        print('\ne_g[:5]',e_g[:5])
        print('\nep_g[:5]',ep_g[:5])
    """

    plt.figure()
    plt.scatter(e, ep, label='Real', c='red', s=150, alpha=0.5)
    if b_g is not None:
        plt.scatter(e_g, ep_g, label='Fake', c='blue', s=150, alpha=0.5)
    
    plt.plot(e, e, linestyle=':', c='red')

    plt.xlabel('$E_{DFT}$', fontsize=15)
    plt.ylabel('$E_{pred}$', fontsize=15)
    plt.title('Plot of $E_{DFT}$ vs $E_{pred}$ using SNAP', fontsize=20)
    plt.legend()

    # If no b_g then save to .../gan/plots/ otherwise save to .../gan/savednet/                
    if save:
        now = datetime.now()
        time_date = now.strftime('(%H%M_%d%m)')
        dir_path = GS['model_path'] + '/plots/'
        
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        if b_g is None:
            plt.savefig('plots/plot_e_'+str(time_date)+'_'+set_name+'.png', bbox_inches='tight')
        else:
            plt.savefig(dir_path+'plot_e_'+str(time_date)+'_'+set_name+'.png', bbox_inches='tight')
        plt.close()
        
    if show:
        plt.show()

def plot_b(info, GS, b, l, b_g=None, l_g=None, save=True, show=True, plot_info=False):
    """
    = Function that plots the mean and standardard deviation of inputted B for each dataset 
      and for each species in input
    
    INPUTS:
        1) info - combined info dictionary
        2) GS - gan training settings dictionary
        3) b - list of list of tensors of shape [N_set][N_mol,N_at,N_b]
        4) l - list of tensors of shape [N_set][N_mol,N_at]
        5) b_g - list of list of tensors of shape [N_set][N_mol,N_at,N_b] - default=None
            = if supplied will also plot generated B against real B
        6) l_g - list of tensors of shape [N_set][N_mol,N_at] - default=None
        7) save - [bool] whether to save plot to path '.../gandalf/gan/plots/'
        8) show - [bool] whether to show plots immediately
        9) plot_info - [bool] whether to plot raw dataset mean and std from info
        
        
    OUTPUTS:
        - plots of mean and standard deviation for real (and generated) B for each dataset and 
          for each species in inputted B
    
    NOTES:
        - called plot_b because it plots for all datasets and species, as opposed to 
          other plotter functions 1) plot_b_species and 2) plot_b_set_species
    """
    # List of names of each set molecule 
    names = [config.mol_comp[tuple(sorted(l[i_set][0].cpu().detach().numpy()))] for i_set in range(len(b))]
    
    b = species_seperator2(b, l, info)
    if b_g is not None:
        b_g = species_seperator2(b_g, l_g, info)

    # Lists of means and std - shape [N_set][N_species][N_b]
    B_r_mean, B_r_std, B_g_mean, B_g_std = [], [], [], []
    for i_set in range(len(b)):
        B_r_mean_set, B_r_std_set, B_g_mean_set, B_g_std_set = [], [], [], []
        for i_species in range(len(b[i_set])):
            B_r_mean_set.append(torch.mean(b[i_set][i_species], dim=(0,1)))
            B_r_std_set.append(torch.std(b[i_set][i_species], dim=(0,1)))
            
            if b_g is not None:
                B_g_mean_set.append(torch.mean(b_g[i_set][i_species], dim=(0,1)))
                B_g_std_set.append(torch.std(b_g[i_set][i_species], dim=(0,1)))

        B_r_mean.append(B_r_mean_set)
        B_r_std.append(B_r_std_set)
        
        if b_g is not None:
            B_g_mean.append(B_g_mean_set)
            B_g_std.append(B_g_std_set)
    
    for i_set in range(len(b)):
        plt.figure(figsize=(10,6))
        i_plot = 0
        unq_species = np.unique(l[i_set])
        num_species = len(unq_species)

        for i_species, species in enumerate(unq_species):
            species_name = config.species_dict[species]
            
            # Attempt to make number of subplots variable - not tested for more than 3
            if num_species < 4:
                plt_list = [1, num_species, 1]
            else:
                nrows = num_species//2
                ncols = num_species//2 + num_species%(num_species//2)
                plt_list = [nrows, ncols, 1]
            plt_list[2] += i_plot 
            plt.subplot(*plt_list)
            
            # plot real b
            plt.plot(range(info['N_feats']), B_r_mean[i_set][i_species], color='b', label='Real '+str(species_name)+' mean')
            plt.fill_between(range(info['N_feats']), B_r_mean[i_set][i_species] - B_r_std[i_set][i_species], 
                                                 B_r_mean[i_set][i_species] + B_r_std[i_set][i_species],
                                                 color='b',
                                                 alpha = 0.2, 
                                                 label='Real '+str(species_name)+' std')
            
            # plot generated b if supplied
            if b_g is not None:                        
                plt.plot(range(info['N_feats']), B_g_mean[i_set][i_species], color='orange', label='Gen '+str(species_name)+' mean')
                plt.fill_between(range(info['N_feats']), B_g_mean[i_set][i_species] - B_g_std[i_set][i_species], 
                                                     B_g_mean[i_set][i_species] + B_g_std[i_set][i_species],
                                                     color='orange',
                                                     alpha = 0.2, 
                                                     label='Gen '+str(species_name)+' std')
                                                     
            # plot raw dataset b from info if supplied
            if plot_info:
                plt.plot(range(info['N_feats']), info['B_mean_'+str(species)], color='b', label='info[\'B_mean_'+str(species)+'\']')
                plt.fill_between(range(info['N_feats']), info['B_mean_'+str(species)] - info['B_std_'+str(species)], 
                                                     info['B_mean_'+str(species)] + info['B_std_'+str(species)],
                                                     color='r',
                                                     alpha = 0.2, 
                                                     label='info[\'B_std_'+str(species)+'\']')

            # set title        
            if b_g is not None:                        
                plt.suptitle('Plot of B_gen vs B_real', fontsize=10)
                plt.title('for '+str(species_name)+' atoms in '+str(names[i_set]), fontsize=8)
            else:                                     
                plt.suptitle('Plot of B_real', fontsize=10)
                plt.title('for '+str(species_name)+' atoms in '+str(names[i_set]), fontsize=8)
                
            if i_plot == 0:    
                plt.ylabel('B magnitude', fontsize=8)
            plt.xlabel('B element', fontsize=8)
            plt.legend(fontsize=8)
            plt.grid(True)
            
            # iterate to new subplot
            i_plot += 1
            
        # If no b_g then save to .../gan/plots/ otherwise save to .../gan/savednet/                
        if save:
            now = datetime.now()
            time_date = now.strftime('(%H%M_%d%m)')
            dir_path = GS['model_path'] + '/plots/'
            
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            if b_g is None:
                plt.savefig('plots/plot_b_'+str(names[i_set][0])+'_'+str(time_date)+'.png', bbox_inches='tight')
            else:
                plt.savefig(dir_path+'plot_b_'+str(names[i_set][0])+'_'+str(time_date)+'.png', bbox_inches='tight')
            plt.close()
            
        if show:
            plt.show()
            
def plot_toulene(xyz, l, xyz_g, l_g):
    """
    = Function that plots rotation of methyl group H atoms in toulene 

    INPUTS:
        1) xyz - list of xyz cartesian coordinates from training dataset - shape [N_mol][N_at][3]
        2) l - list of int labels - shape [N_mol][N_at]
        3) xyz_g - list of xyz cartesian coordinates from generator - shape [N_mol][N_at][3]
        4) l_g - list of int labels - shape [N_mol][N_at]

    OUTPUTS:
        - plots of mean and standard deviation for real (and generated) B

    NOTES:
        - strict ordering of atoms to specify metyl group
    """
    xyz, xyz_g = np.array(xyz), np.array(xyz_g)
    l, l_g = np.array(l), np.array(l_g)
    
    for i_mol, xyz_i in enumerate(xyz):
        xyz_c = xyz[i_mol][:7]
        
        pca = PCA(n_components=2)
        xy_c = pca.fit_transform(xyz_c)
        #pca.fit(xyz_c)
        
        plane_vectors = pca.components_
        normal_vector = np.cross(plane_vectors[0], plane_vectors[1])
        
        print('plane_vectors[0]',plane_vectors[0])
        print('plane_vectors[1]',plane_vectors[1])
        print('plane_vectors[1][0]',plane_vectors[1][0])
        print('normal_vector',normal_vector)
        print('xyz_c[:][0]',xyz_c[:,0])
        print('xyz_c[:][1]',xyz_c[:,1])
        print('xyz_c[:][2]',xyz_c[:,2])
        print('xyz_c[:]',xyz_c[:])
        print('np.array(xyz_c).shape',np.array(xyz_c).shape)
        print('np.array(xyz).shape',np.array(xyz).shape)
        print('xyz[i_mol][:][0]',xyz[i_mol][:][0])
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(xyz[i_mol][:,0], xyz[i_mol][:,1], xyz[i_mol][:,2])
        
        x,y,z = xyz_c[0][0], xyz_c[0][1], xyz_c[0][2]
        ax.quiver(x,y,z, plane_vectors[0][0], plane_vectors[0][1], plane_vectors[0][2], color='r', label='1')
        ax.quiver(x,y,z, plane_vectors[1][0], plane_vectors[1][1], plane_vectors[1][2], color='g', label='2')
        ax.quiver(x,y,z, normal_vector[0], normal_vector[1], normal_vector[2], color='b', label='normal')
        plt.legend()
        plt.show()
    
def plot_b_single2(info, GS, dataset, nn_G, set_inds=[0], plot_r_std=True, num_real=1000, G_conv=False, G_comp=False, save=True, show=True, inv_fs=True, plot_ideal=True):
    """
    = Function that plots a single real B and a single generated B differentiated the species
    
    INPUTS:
        1) info - combined info dictionary
        2) GS - gan training settings dictionary
        3) dataset - custom PyTorch dataset class created with data.BL_DataSet()
        4) nn_G - generator model
        5) set_inds - list of [int] of indexes of datasets in info to generate - set to [0] if only 1 mol in train set
        6) plot_r_std - [bool] for whether to plot mean and std for num_real molecules
        6) num_real - [int] number of real molecules to get - used for plotting real mean and std
        7) G_conv - [bool] for whether layers of nn_G are conv (=T) or fc (=F)
        8) G_comp - [bool] for whether input of G.make_seeds is z,l (=T) or z (=F)
        9) save - [bool] whether to save plot to path '.../gandalf/gan/plots/'
        10) show - [bool] whether to show plots immediately
        11) inv_fs - [bool] whether to apply inverse scaling to plotted B
        12) plot_ideal - [bool] whether to also load and plot ideal molecule B
        
    OUTPUTS:
        - plots of mean and standard deviation for real (and generated) B
    """
    # Get a single real and gen molecule
    B_r, L_r = get_train(1000, set_inds, dataset, info, GS)
    B_g, L_g = generate_like_train(1, set_inds, nn_G, info, GS, G_conv=G_conv, G_comp=G_comp)

    # Apply Inverse Feature Scaling
    if inv_fs:
        B_r = data.inv_fs(B_r, L_r, info, GS)
        B_g = data.inv_fs(B_g, L_g, info, GS)

    # List of names of each set molecule 
    names = [config.mol_comp[tuple(sorted(L_r[i_set][0].cpu().detach().numpy()))] for i_set in range(len(set_inds))]
    
    # Get B and L for ideal molecules for all sets
    if plot_ideal:
        B_eq = [torch.tensor(np.load(GS['data_paths'][i_set] + 'ideal_B.npy')) for i_set in range(info['N_set'])]
        L_eq = [torch.tensor(np.load(GS['data_paths'][i_set] + 'ideal_L.npy')).unsqueeze(0) for i_set in range(info['N_set'])]
        if not inv_fs:
            B_eq = data.fs(B_eq, L_eq, info, GS)
            
    # Seperate B species -> [N_set][N_species][N_mol,N_at_species,N_feat]
    B_r = species_seperator2(B_r, L_r, info)
    B_g = species_seperator2(B_g, L_g, info)
    
    if plot_ideal:
        B_eq = species_seperator2(B_eq, L_eq, info)
        
    print('B_r[0][0]',B_r[0][0])
    if plot_r_std:
        # Lists of means and std - shape [N_set][N_species][N_b]
        B_r_mean, B_r_std = [], []
        for i_set in range(len(B_r)):
            B_r_mean_set, B_r_std_set = [], []
            for i_species in range(len(B_r[i_set])):
                B_r_mean_set.append(torch.mean(B_r[i_set][i_species], dim=(0,1)))
                B_r_std_set.append(torch.std(B_r[i_set][i_species], dim=(0,1)))
            B_r_mean.append(B_r_mean_set)
            B_r_std.append(B_r_std_set)
    
    # For each specified dataset
    for i_set, set_ind in enumerate(set_inds):
        plt.figure()#figsize=(10,6), dpi=200)
        i_plot = 0
        unq_species = np.unique(L_r[set_ind])
        num_species = len(unq_species)
        num_feats = range(info['N_feats'])
        markers = ['x','o','*',   ] # markers for species

        # For each species in set
        for i_species, species in enumerate(unq_species):
            species_name = config.species_dict[species]
            
            # Get subplot plot iteration
            # - attempt to make number of subplots variable - not tested for more than 3
            if num_species < 4:
                plt_list = [1, num_species, 1]
            else:
                nrows = num_species//2
                ncols = num_species//2 + num_species%(num_species//2)
                plt_list = [nrows, ncols, 1]
            plt_list[2] += i_plot 
            plt.subplot(*plt_list)
            
            # Plot B_real mean and std
            plt.errorbar(num_feats, B_r_mean[i_set][i_species], yerr=B_r_std[i_set][i_species], label='$B_{real}(\mu\;\sigma)$', fmt='b_', markersize=3, alpha=0.3)#, linestyle='dotted')#, marker=markers[i_species])
            plt.errorbar(num_feats, B_r_mean[i_set][i_species], yerr=B_r_std[i_set][i_species], fmt='b_', markersize=3, alpha=0.3)#, linestyle='dashdot')#, marker=markers[i_species])
    
            """
            # plot raw dataset b from info if supplied
            if plot_info:
                plt.plot(range(info['N_feats']), info['B_mean_'+str(species)], color='b', label='info[\'B_mean_'+str(species)+'\']')
                plt.fill_between(range(info['N_feats']), info['B_mean_'+str(species)] - info['B_std_'+str(species)], 
                                                     info['B_mean_'+str(species)] + info['B_std_'+str(species)],
                                                     color='r',
                                                     alpha = 0.2, 
                                                     label='info[\'B_std_'+str(species)+'\']')
            """
                                                     
            # Plot B_real for each atom in single mol
            for i_atom in range(len(B_r[i_set][i_species][0])):
                if i_atom == 0:
                    plt.scatter(num_feats, B_r[i_set][i_species][0,i_atom], label='$B_{real}$', color='blue', marker='o')#, linestyle='dotted')#, marker=markers[i_species])
                else:
                    plt.scatter(num_feats, B_r[i_set][i_species][0,i_atom], color='blue', marker='o')#, linestyle='dashdot')#, marker=markers[i_species])

            # Plot B_gen for each atom in single mol        
            for i_atom in range(len(B_g[i_set][i_species][0])):
                if i_atom == 0:
                    plt.scatter(num_feats, B_g[i_set][i_species][0,i_atom], label='$B_{gen}$', color='orange', marker='*')#, linestyle='dotted')
                else:
                    plt.scatter(num_feats, B_g[i_set][i_species][0,i_atom], color='orange', marker='*')#, linestyle='dotted')

            # Plot B_ideal for each atom in single mol
            if plot_ideal:        
                for i_atom in range(len(B_eq[i_set][i_species][0])):
                    if i_atom == 0:
                        plt.scatter(num_feats, B_eq[i_set][i_species][0,i_atom], label='$B_{eq}$', color='green', marker='x')#, linestyle='loosely dashed')
                    else:
                        plt.scatter(num_feats, B_eq[i_set][i_species][0,i_atom], color='green', marker='x')#, linestyle='lo')

            # Plot settings 
            plt.suptitle('Plot of B_gen vs B_real', fontsize=10)
            plt.title('for '+str(species_name)+' atoms in '+str(names[i_set]), fontsize=8)
            if i_plot == 0:    
                plt.ylabel('B magnitude', fontsize=8)
            plt.xlabel('B element', fontsize=8)
            plt.legend(fontsize=8)
            plt.grid(True)
            
            # iterate to new subplot
            i_plot += 1
            
        if save:
            now = datetime.now()
            time_date = now.strftime('(%H%M_%d%m)')
            plt.savefig('plots/b_'+str(names[set_ind][0])+'_'+str(time_date)+'.png')
            
        if show:
            plt.show()
                        
def plot_rc(xyz_coords, labels, rc_dict, save=True, show=True, h2_flag=False):
    """
    = Function that plots the cutoff radius for each species over a 2D projection of a molecule 

    INPUTS:
        1) xyz_coords - list of xyz cartesian coordinates - shape [N_at][3]
        2) labels - list of int label for each atom - shape [N_at]
        3) rc_dict - dictionary mapping int label to cutoff radius (*rcutfac) e.g. {1:1.35,0:1.35}
        4) save - [bool] whether to save plot to path '.../gandalf/gan/plots/'
        5) show - [bool] whether to show plots immediately
        6) h2_flag - [bool] if N_at=2 then mol is a line and need to manually set box boundaries
        
    OUTPUTS:
        - 2D plot of molecule with circles displaying cutoff radius 
        
    NOTES:
        - performs PCA on inputted 3D cart coords transforming into 2D mol
        - LAMMPS internally multiplies rc by 2 so multiply by 2 here to plot
          the correct rc.
    """
    # reduce dim from [N_at, 3] to [N_at, 2] - flatten in z-plane
    pca = PCA(n_components=2)
    xy_coords = pca.fit_transform(xyz_coords)
    
    N_species = len(np.unique(labels))
    N_at = len(labels)
        
    # Multiply rc by factor of 2
    rc_dict.update({key: rc_dict[key]*2.0 for key in rc_dict.keys()})
        
    # Defining needed iterables for species
    colors=['r','b','g','m','y','c'] # set colors for species
    species_list = sorted(list(rc_dict.keys()))
    
    # seperate xyz coords [N_at, 2] into [N_species][N_at_species,2] 
    species_coords = []
    for i_species, species in enumerate(species_list):
        species_inds = [i for i in range(N_at) if species == labels[i]]
        species_coords.append(xy_coords[species_inds])
    
    # Create Plot Axis
    ax = plt.gca()
    ax.cla()
    
    for i_species, species in enumerate(species_list):
        # scatter plot atom 2D coords as coloured dots
        ax.scatter(species_coords[i_species][:,0], 
                      species_coords[i_species][:,1], 
                      color=colors[species], 
                      label=config.species_dict[species]+': $R_{cut}$'+' ='+str(np.round(rc_dict[species],2))+'$\AA$')

        # plot Rcut as circle around 1 atom of each species
        circle=plt.Circle((species_coords[i_species][0,0], species_coords[i_species][0,1]), 
                                rc_dict[species], 
                                color=colors[species], 
                                fill=False)
        ax.add_artist(circle)
        
        # plot tiny dashed circle central atom of Rcut circle for clarity purposes 
        circle=plt.Circle((species_coords[i_species][0,0], species_coords[i_species][0,1]), 
                                0.1, 
                                linestyle='dashed',
                                color=colors[species], 
                                fill=False)
        ax.add_artist(circle)
    
    # plot settings
    plt.yticks([])
    plt.xticks([])
    ax.set_aspect('equal')
    plt.legend()#bbox_to_anchor=(1.1, 1.15))

    if h2_flag:
        species_l = 0
        plt.ylim(-1.2*rc_dict[species_l], 1.2*rc_dict[species_l])
        plt.xlim(-1.2*rc_dict[species_l], 1.2*rc_dict[species_l])        
        
    if save:
        now = datetime.now()
        time_date = now.strftime('(%H%M_%d%m)')
        start_time = time.time()
        plt.savefig(config.base_path + '/gan/plots/rc_'+str(time_date)+'.png', format='png')
                
    if show:
        plt.show()

def compute_ia_dists(xyz_coords):
    """
    = Function that computes the interatomic distances between each atom. 

    INPUTS:
        1) xyz_coords - list of xyz positions of each atom - shape [N_at][3]

    OUTPUTS:
        1) A - numpy 2D array of interatomic distances - shape [N_at][N_at]
    """
    N_at = len(xyz_coords)
    A = np.zeros((N_at, N_at))
    for i in range(N_at - 1):
        for j in range(i + 1, N_at):
            A[i, j] = np.linalg.norm(xyz_coords[i] - xyz_coords[j], ord=2)
            A[j, i] = A[i, j]
    return A

def triu_label_combinations(atomic_labels):
    N_at = len(atomic_labels)
    Z = []
    for i in range(N_at - 1):
        for j in range(i + 1, N_at):
            tmp = atomic_labels[i] + atomic_labels[j]
            tmp = ''.join(sorted(tmp))
            Z.append(tmp)
    return np.array(Z)        
        
def plot_bonds(xyz_r, l_r, xyz_g, l_g, save=True, show=True, sns_flag=False, num_bins=200, label_2='G', density=True):
    """
    = Function that plots the bond length distribution for each species combination in inputted molecules 

    INPUTS:
        1) xyz_r - list of xyz cartesian coordinates from training dataset - shape [N_mol][N_at][3]
        2) l_r - list of int labels - shape [N_mol][N_at]
        3) xyz_g - list of xyz cartesian coordinates from generator - shape [N_mol][N_at][3]
        4) l_g - list of int labels - shape [N_mol][N_at]
        5) save - [bool] whether to save plot to path '.../gandalf/gan/plots/'
        6) show - [bool] whether to show plots immediately
        7) sns_flag - [bool] whether to use seaborn (=T) or matplotlib (=F) for making histograms (sns better for H2 ?)
        8) num_bins - [int] number of bins in histogram plot 
        9) label_2 - [int] name for green plotted lines - default='G' and alternative is 'I' for Inverted
        
    OUTPUTS:
        - a figure containing histogram subplots 
        
    NOTES:
        - only tested for benzene and H2
    """    
    # Convert lists to arrays
    xyz_r, xyz_g = np.array(xyz_r), np.array(xyz_g)
    l_r, l_g = np.array(l_r), np.array(l_g)
    
    # Sort labels and then xyz for Real mols
    for i, l_r_mol in enumerate(l_r):
        sorted_inds = np.argsort(l_r_mol)
        xyz_r[i] = xyz_r[i][sorted_inds]
        l_r[i] = l_r_mol[sorted_inds]
        
        
    # Sort labels and then xyz for Gen mols
    for i, l_g_mol in enumerate(l_g):
        sorted_inds = np.argsort(l_g_mol)
        xyz_g[i] = xyz_g[i][sorted_inds]
        l_g[i] = l_g_mol[sorted_inds]
    
    # Get inter-atomic distances matrices    
    inds = np.triu_indices(n=len(l_r[0]), k=1)
    ia_dists_r = np.array([compute_ia_dists(mol)[inds] for mol in xyz_r])
    ia_dists_g = np.array([compute_ia_dists(mol)[inds] for mol in xyz_g])
    
    # Find unique species combinations (e.g. for benzene = C-C,H-H,C-H)
    str_labels = inv_convert_labels(l_r[0])
    bonds = triu_label_combinations(str_labels)
    unq_bonds = np.unique(bonds)
    
    # Colors and Plot Labels
    data_label = ['R', label_2]
    color = ('C0', 'C2')
    
    # Create Subplot for each Unique species combination
    num_rows = len(unq_bonds)
    fig, axs = plt.subplots(nrows=num_rows, ncols=1)
    
    # If only one species present -> only one bond subplot shown
    # put axs in list so it can be indexed as normal
    if num_rows == 1:
        axs = [axs]

    # For each subplot
    for i in range(num_rows):
        # Find Interatomic Dists corressonding to this bond 
        inds = np.where(bonds == unq_bonds[i])[0]
        data_i = [ia_dists_r[:, inds].flatten(), ia_dists_g[:, inds].flatten()]
        
        # Create Hist plot with labels only for first subplot
        if i == 0:
            if sns_flag:
                for j, data_ij in enumerate(data_i): 
                    sns.distplot(data_ij, bins=num_bins, color=color[j], label=data_label[j], ax=axs[i])
            else:
                axs[i].hist(data_i, density=density, bins=num_bins, histtype='step', color=color, label=data_label, lw=1)
        else:
            if sns_flag:
                for j, data_ij in enumerate(data_i): 
                    sns.distplot(data_ij, bins=num_bins, color=color[j], ax=axs[i])
            else:
                axs[i].hist(data_i, density=density, bins=num_bins, histtype='step', color=color, lw=1)

    # Plot Settings
        axs[i].set_ylabel('Count ({}-{})'.format(unq_bonds[i][0], unq_bonds[i][1]), fontsize=8)

    axs[-1].set_xlabel('Distance ($\AA$)', fontsize=8)
    axs[0].legend(fontsize=8)
    axs[0].set_title('Histograms of Bond Length Distributions', fontsize=10)

    fig.tight_layout()        
    
    if save:
        now = datetime.now()
        time_date = now.strftime('(%H%M_%d%m)')
        plt.savefig(config.base_path + '/gan/plots/bonds_'+str(time_date)+'.png')
    
    if show:
        plt.show()
    
def convert_labels(old_labels, force_convert=False):
    """
    = Function that converts from str and LAMMPS atomic weight labels to int labels 
      for NN species-wise indexing (and conditional embedding/encoding).

    INPUTS:
        1) old_labels - list of shape [N_at] where each label one of {'H','C','O','N'} or {1,6,7,8}
        2) force_convert - [bool] for forcing conversion for where set of old labels E {0,1,2,3} - default=False
        
    OUTPUTS:
        1) new_labels - list of shape [N_at] where each label E {0,1,2,3}
        
    NOTES:
        - does nothing if old_labels is like desired output labels unless force_convert
        - only tested for benzene, need to try for toulene
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
                break
                
        return new_labels

def inv_convert_labels(old_labels):
    """
    = Function that converts from int labels to str labels 

    INPUTS:
        1) old_labels - list of shape [N_at] where each label one of {0,1,2,3}
        
    OUTPUTS:
        1) new_labels - list of shape [N_at] where each label one of {'H','C','O','N'}
        
    NOTES:
        - does nothing if old_labels is like desired output labels
        - only tested for benzene, need to try for toulene
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
                break
                
        return new_labels
        
def invert_b(start_molecule, target_b, target_l, info,
                 rcutfac=0, rc=[], rfac0=0, twojmax=0, weights=[], 
                 loss_threshold=0.2, max_N=10000, 
                 lr=0.000008, alpha=8e-4, eta=1000):    
    """
    = Function that finds xyz coords for given bispec comps.

    INPUTS:
        1) start_molecule - ASE Mol object created with cobe.Dataset - works best of is relaxed xyz coords of molecule
        2) target_b - list of bispec comps of (generated) molecule to be inverted - shape [N_species, N_feats] 
        3) target_l - list of int labels for atoms in target_b - shape [N_at]
        
        4) rcutfac - [float] data factor applied to all cutoff radii e.g. info['rcutfac'] ***
        5) rc - list of [float] cutoff radii, one for each type e.g. info['rc_dict'] = {1:0.6,0:0.5} ***
        6) rfac0 - [float] parameter in distance to angle conversion (0 < rcutfac < 1) - default always 0.99
        7) twojmax - [int] band limit for bispectrum components - controls number of bispec comps,
                          with default always set to 8 -> 55 bispec comps
        8) weights - list of [floats] neighbor weights, one for each type - default of [1.1,0.37] - NEED TO 
                          EXPERIMENT WITH THESE VALUES AND TO APPEND TO INFO DICT
        9) loss_threshold - [float] value of inversion loss function which is defined to be fully converged -> break
        10) max_N - [int] maximum number of optimizations to perform before stopping process 
        11) lr - [float] learning rate controlling step size of each update
        12) alpha - [float] learning rate controlling size of random noise 
        13) eta - [float] exponential decay constant of random noise
        
    OUTPUTS:
        1) trajectory - list of xyz coords for each iteration - shape [N,N_at,3]
        2) loss - list of [float] loss values for each iteration - shape [N,]
        3) convergence_type - [str] label for dir name in which to save .xyz file - either {'conv','not_conv'}
        
    NOTES:
        - only tested for benzene and H2, need to try for toulene
        - needs testing to see if shuffled atom order works
        - doesn't seem to work for ethanol - maybe only works for {H,C}
    """
    # Get Unique Species in target_l
    #_, inds = np.unique(target_l, return_index=True)
    #all_species = list(target_l[np.sort(inds)])
    #print('target_l',target_l)    
    #print('all_species',all_species)    
    
    # get atom types - in same order as ASE and LAMMPS e.g. [6,1,8] 
    l = start_molecule.trajectory[-1].types
    #_, inds = np.unique(l, return_index=True)
    #all_species = convert_labels(l[np.sort(inds)], force_convert=True)

    all_species2 = start_molecule.types
    all_species = convert_labels(start_molecule.types, force_convert=True)
    
    # Define Useful Parameters
    n_atoms = len(target_l)
    n_species = len(target_b)
    n_features = len(target_b[0])
    target_species = sorted(info['all_species'])
    
    # Sort target_b to same order as trj[-1].types 
    inds = [target_species.index(species) for species in all_species]
    sum_bt = copy.deepcopy([target_b[ind] for ind in inds])
    target_species = copy.deepcopy([target_species[ind] for ind in inds])
    
    # Make trj list where first entry is ideal mol trj
    trajectory = []
    trajectory.append(start_molecule.trajectory[-1])
    
    # Empty arrays
    loss = np.zeros(max_N)
    #sum_bt = [np.zeros(n_features) for _ in range(n_species)]
    #sum_bt = copy.deepcopy(target_b)
    
    print_flag = 0
    if print_flag:
        print('l',l)    
        print('all_species',all_species)    
        print('target_l',target_l)
        print('target_species',target_species)
        print('n_atoms',n_atoms)    
        print('n_features',n_features)    
        print('n_species',n_species)    
        print('start_molecule.trajectory[-1].positions',start_molecule.trajectory[-1].positions)

    # Sum up all Bs of same species for Target Mol
    #for i in range(n_atoms):
    #    sum_bt[all_species.index(target_l[i])] += target_b[i]
    
    convergence_type = 'not_conv'
    pbar = tqdm.tqdm(range(max_N))
    
    sum_b_arr = np.zeros((max_N, n_species, n_features))
    
    # For each iteration try make sum_b closer to sum_bt
    for n in pbar: 
        #print('\nn',n)   
        #print('trajectory[-1].positions',trajectory[-1].positions)

        print('trajectory[n]',trajectory[n])    
        print('rcutfac',rcutfac)
        print('rfac0',rfac0)
        print('twojmax',twojmax)    
        print('rc',rc)
        print('weights',weights)

        compute_bispectrum(trajectory[n],     
                                 types=start_molecule.types, # only needed for new dask version
                                  rcutfac=rcutfac,
                                  rfac0=rfac0,
                                 twojmax=twojmax,           
                                 rc=rc,
                                 weights=weights                       
                                 )  

        if print_flag:
            print('n',n)  
            print('len(trajectory)',len(trajectory))  
            print('trajectory[n].types',trajectory[n].types)   
            
        #print('trajectory[-1].positions',trajectory[-1].positions)
        # Set trj lists to be empty (???)
        trajectory.append(copy.deepcopy(trajectory[n]))
        trajectory[-1].descriptors = [[] for i in range(n_atoms)]
        trajectory[-1].descriptors_d = [[[[] for k in range(3)] for j in range(n_species)] for i in range(n_atoms)]
        
        if print_flag:
            print('len(trajectory)',len(trajectory))
            print('n',n)
            
        # Empty arrays
        grad_loss = [np.zeros(3) for _ in range(n_atoms)]
        sum_b = [np.zeros(n_features) for _ in range(n_species)]
        
        # Sum up all Bs of same species 
        trajectory[n].types = convert_labels(trajectory[n].types, force_convert=True)
        for i in range(n_atoms):
            sum_b[all_species.index(trajectory[n].types[i])] += trajectory[n].descriptors[i]
        
        """        
        plt.figure()
        # plot 2 subplots for unique species
        for i_species, species in enumerate(all_species):
            plt.subplot(*(1,len(all_species),1+i_species))
            plt.scatter(range(n_features), sum_bt[i_species], label='B_target', color='orange', marker='*')
            plt.scatter(range(n_features), sum_b[i_species], label='B_eq', color='green', marker='x')
            plt.title('Species:'+str(species))
            plt.legend()
        
        plt.show()
        """
        
        sum_b_arr[n] = np.array(sum_b)

        if print_flag:
            print('trajectory[n].types',trajectory[n].types)  
                        
        # Calc loss
        for j in range(n_species):
            loss[n] += np.dot(sum_bt[j]-sum_b[j], sum_bt[j]-sum_b[j])
        
        # Calc derivative of loss
        for i in range(n_atoms):   
            for j in range(n_species):
                for k in range(3):        
                    grad_loss[i][k] += -2*np.dot(sum_bt[j]-sum_b[j], trajectory[n].descriptors_d[i][j][k])

            # Update XYZ positions
            loss_term = lr * np.array(grad_loss[i])
            #noise_term = alpha * np.exp(-n/eta) * np.random.uniform(0,1,(1,))
            noise_term = alpha * np.exp(-n/eta) * np.random.uniform(-1,1,(3,))
            trajectory[-1].positions[i] = np.array(trajectory[-1].positions[i]) + loss_term + noise_term

        #print('trajectory[-1].positions',trajectory[-1].positions)
        
        # Make progress bar print loss value   
        pbar.set_description("Loss: {}".format(np.round(loss[n],5)))
        
        # If optimization Converges then Break loop
        if loss[n] < loss_threshold:
            convergence_type = 'conv'
            break
    
    # remove extra zeros from loss array
    loss = loss[:n+1]
    
    from matplotlib.widgets import Slider, Button
    
    plot_slider = 1
    if plot_slider:
    
        def f(i, b_sum, i_species):
            return b_sum[i][i_species]
            
        t = np.arange(0,max_N)
        i_plot = 1
        line_list = []
        
        # Create the figure and the line that we will manipulate
        fig, ax = plt.subplots()
        
        # Make a horizontal slider to control the Iteration.
        axcolor = 'lightgoldenrodyellow'
        ax_iter = plt.axes([0.2, 0.01, 0.65, 0.035], facecolor=axcolor)
        iter_slider = Slider(
             ax=ax_iter,
             label='Iteration N',
             valmin=0,
             valmax=max_N-1,
             valinit=0, 
             valstep=1)
             
        for i_species, species in enumerate(all_species):
            plt.subplot(*[1,len(all_species),i_plot])
            line, = plt.plot(range(n_features), f(0, sum_b_arr, i_species),'bo',label='sum_b')
            plt.plot(range(n_features), sum_bt[i_species],'r*',label='sum_bt')
            ax.set_xlabel('B element')
            ax.set_ylabel('B magnitude')
            plt.title('Species:'+str(all_species[i_species]))
            
            i_plot += 1
            line_list.append(line)

        if len(all_species) >= 1:
            # The function to be called anytime a slider's value changes
            def update0(val):
                line = line_list[0]
                line.set_ydata(f(iter_slider.val, sum_b_arr, 0))
                fig.canvas.draw_idle()

            # register the update function with each slider
            iter_slider.on_changed(update0)
            
        if len(all_species) >= 2:
            def update1(val):
                line = line_list[1]
                line.set_ydata(f(iter_slider.val, sum_b_arr, 1))
                fig.canvas.draw_idle()
            iter_slider.on_changed(update1)        
        
        if len(all_species) >= 3:
            def update2(val):
                line = line_list[2]
                line.set_ydata(f(iter_slider.val, sum_b_arr, 2))
                fig.canvas.draw_idle()
            iter_slider.on_changed(update2)    
                
        if len(all_species) >= 4:
            def update3(val):
                line = line_list[3]
                line.set_ydata(f(iter_slider.val, sum_b_arr, 3))
                fig.canvas.draw_idle()
            iter_slider.on_changed(update3)
            
        """
        plt.subplot(122)
        line2, = plt.plot(range(n_features), f(0, sum_b_arr, 1),'bo',label='sum_b')
        plt.plot(range(n_features), sum_bt[1],'r*',label='sum_bt')
        ax.set_xlabel('B element')
        ax.set_ylabel('B magnitude')
        plt.title('Species:'+str(all_species[1]))
        plt.legend()
        """
        plt.show()
    
    """
    plt.figure()
    
    plt.subplot(121)
    plt.plot(range(n_features),slider,'bo',label='sum_bt')
    plt.plot(range(n_features),sum_b[0],'r*',label='sum_b')
    plt.legend()
    plt.title('Species:'+str(all_species[0]))
    
    plt.subplot(122)
    plt.plot(range(n_features),sum_bt[1],'bo',label='sum_bt')
    plt.plot(range(n_features),sum_b[1],'r*',label='sum_b')
    plt.legend()
    
    plt.title('Species:'+str(all_species[1]))
    
    plt.show()
    """
    
    return trajectory, loss, convergence_type
    
def gen_xyz(num_mols, set_inds, nn_G, info, GS,
                b_real=None, l_real=None, 
                get_trj=True, save_trj=False, save_end=True,
                rfac0=0.99, weights=[],
                loss_threshold=0.2, max_N=2000,
                lr=0.00000005, alpha=8e-4, eta=1000, 
                G_comp=False, G_conv=False):
    """
    = Function that finds xyz coords for given bispec comps.

    INPUTS:
        1) num_mols - [int] number of molecules to generate and to invert
        2) set_inds - list of [int] of indexes of datasets in info to generate e.g. [0,1] = {12,15}
        3) nn_G - generator NN to generate bispec comp samples with
        4) info - {dict} combined info dictionary
        5) GS - {dict} gan training settings dictionary
        6) b_real - list of [tensors] of B of real mols of shape [N_set][N_mol,N_at,N_feat] - used to verify inversion process I(B(X)) = X **
        7) l_real - list of [tensors] of labels of real mols of shape [N_set][N_mol,N_at] **
        8) get_trj - [bool] specifying whether to return all iterations of trj or just the final one
        9) save_trj - [bool] specifying whether to write every 20th trj to .xyz file in save dir ***
        10) save_end - [bool] specifying whether to write the final trj to .xyz file in save dir ***
        11) rfac0 - [float] parameter in distance to angle conversion (0 < rcutfac < 1) - default always 0.99 ****
        12) weights - list of [floats] neighbor weights, one for each type - ASSUMED TO BE [1.0 for _ in range(N_species))] ****
        13) loss_threshold - [float] value of inversion loss function which is defined to be fully converged -> break
        14) max_N - [int] maximum number of optimizations to perform before stopping process 
        15) lr - [float] learning rate controlling step size of each update
        16) alpha - [float] learning rate controlling size of random noise 
        17) eta - [float] exponential decay constant of random noise
        18) G_conv - [bool] for whether layers of nn_G are conv (=T) or fc (=F)
        19) G_comp - [bool] for whether input of G.make_seeds is z,l (=T) or z (=F)
        
    OUTPUTS:
        1) trjs - list of list of Molecule objects of shape [N_set][N_mol][N_trj] *****
        2) losses - list of list of loss for the inversion of each sample of shape [N_set][N_mol][N]
        3) save_path - [str] path to dir in which end and trj are saved - differs for Real and Gen 
        
    NOTES:
        - only tested for benzene and H2, need to try for toulene
        
        ** if b_real and l_real are not supplied, then will generate_like_train 
        *** save directory named after the nn_G that generated the original samples
        **** These (rfac0 and weights) should be improved by appending to info dict like rcutfac and rc
        ***** N_trj depends on get_trj - if False then N_trj=1 but if true then N_trj=N_mol//20
    """
    # Find twojmax that corresponds to info['N_feats']
    twojmax_dict = {55:8, 30:6} # keys:info['N_feats'] and values:twojmax
    twojmax = twojmax_dict[info['N_feats']]
     
    # Variables for Path
    base_dir = config.base_path + 'gan/invert/'
    conv_types = ['conv','not_conv']
    
    # Names of Training Datasets for Loading starting mol
    names = [config.mol_list[i] for i in GS['mols']]
    
    # Get Save Path
    # - if Gen then save in dir gan/invert/datasets_type/model_name/mol_name/conv_type
    if b_real is None: 
        dir_name = get_dir_name(names)
        _model_ind = GS['model_path'].index(str(dir_name))
        _model_name = GS['model_path'][_model_ind:]
        save_path = base_dir + str(_model_name)
        
    # - if Real then save in dir gan/invert/real/datasets_type/mol_name/conv_type
    else:
        data_type = [str(b_real[i_set].shape[1]) for i_set in range(len(b_real))]
        data_type = '_'.join(data_type)
        save_path = base_dir + data_type + '/real/'
    
    # If save_trj True then get dir path    
    if save_trj:
        for name_set in names:
            trj_path = save_path + name_set + '/trj/'    
            # if dir already exists, find ID of last saved item
            if os.path.exists(trj_path):
                last_trj_dict = dict()
                for conv_type in conv_types:
                    fls = os.listdir(trj_path + conv_type + '/')
                    if len(fls) == 0:
                        last_trj_dict[conv_type] = 0
                    else:
                        fls.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
                        last_trj_id = fls[-1][4:-4]
                        last_trj_dict[conv_type] = int(last_trj_id)+1
            # if dir doesn't exist, create new one
            else:
                last_trj_dict = dict()
                for conv_type in conv_types:
                    os.makedirs(trj_path + conv_type + '/')
                    last_trj_dict[conv_type] = 0
    
    # If save_end True then get dir path    
    if save_end:
        for name_set in names:
            end_path = save_path + name_set + '/end/'    
            # if dir already exists, find ID of last saved item
            if os.path.exists(end_path + conv_types[0]):
                last_end_dict = dict()
                for conv_type in conv_types:
                    fls = os.listdir(end_path + conv_type + '/')
                    if len(fls) == 0:
                        last_end_dict[conv_type] = 0
                    else:
                        fls.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
                        last_end_id = fls[-1][4:-4]
                        last_end_dict[conv_type] = int(last_end_id)+1
            # if dir doesn't exist, create new one
            else:
                last_end_dict = dict() 
                for conv_type in conv_types:
                    os.makedirs(end_path + conv_type + '/')
                    last_end_dict[conv_type] = 0
            
    # Generate B if Real B not supplied
    if b_real is None:
        b, l = generate_like_train(num_mols, set_inds, nn_G, info, GS, G_comp=G_comp, G_conv=G_conv)
        #b = data.inv_fs(b, l, info, GS)
        print(b[0].shape)
        b = [torch.tensor(info['std'].inverse_transform(b[0].reshape((-1,info['N_all_species']*info['N_feats'])))).reshape((-1,info['N_all_species'],info['N_feats']))]    
    elif b_real is not None:
        b = [b_real[set_ind] for set_ind in set_inds]
        l = [l_real[set_ind] for set_ind in set_inds]
        b = data.sum_over_species(b, l, info, GS)
        
    # Empty lists to be Returned
    trjs, losses = [], []

    # For each dataset specified 
    for i_set, set_ind in enumerate(set_inds):
    
        trjs_set, losses_set = [], []
        
        # set save_end and save_trj paths 
        trj_path = save_path+str(names[set_ind])+'/trj/'
        end_path = save_path+str(names[set_ind])+'/end/'
        
        # Set Starting XYZ Coords
        start_mol_path = base_dir + 'start/{}_ideal.xyz'.format(names[set_ind])
        #center_xyz_mol(1, base_dir+'start/', '{}_ideal.xyz'.format(names[set_ind]), new_file_name='{}_ideal2.xyz'.format(names[set_ind]))
        #sys.exit()
        
        start_mol = Dataset(start_mol_path)
    
        # change shape from [N_set][N_mols,...] to [N_mols,...]
        _b, _l = b[i_set].numpy(), l[i_set].numpy() 
        
        # resume from id (i_mol suffix) of last inverted mol
        start_mol_ind = 0
        if save_end and b_real is not None:
            start_mol_ind = max(last_end_dict.values())
            
        # Set rc, rcutfac, and weights
        rc = [info['rc_dict'][species] for species in info['species'][set_ind]] 
        rcutfac = info['rcutfac']
        weights = [1.0 for _ in range(info['N_species'][set_ind])]
            
        # Print Info
        print('\n\n-----------------------------------')
        print('Inverting {} Molecules of {}'.format(num_mols, names[set_ind]))
        print('-----------------------------------')
        print('Loading Initial XYZ from: {}'.format(start_mol_path))
        if save_end:
            print('Saving End XYZ to: {}'.format(end_path))
        if save_trj:
            print('Saving Trajectories to: {}'.format(trj_path))
        print('\nFirst Sample ID: {}'.format(start_mol_ind))
        print('rc', rc)
        print('rcutfac', rcutfac)
        print('weights',weights,'\n')
        
        # For each sample to be Inverted 
        for i, i_mol in enumerate(range(start_mol_ind, start_mol_ind+num_mols)):
            # select i-th sample to be inverted 
            target_b = _b[i_mol]
            target_l = _l[i_mol]
            
            print('start_mol_ind',start_mol_ind)
            print('target_b',target_b)

            """
            # sort from shuffled into decreasing order to match LAMMPS i.e. [1,1,1,1,1,1,0,0,0,0,0,0]
            sorted_inds = np.argsort(target_l)
            target_b = target_b[sorted_inds][::-1]
            target_l = target_l[sorted_inds][::-1]
            """
            print('target_l',target_l)
            print('info[all_species]',info['all_species'])
            
            # Inversion
            trj, loss, conv_type = invert_b(copy.deepcopy(start_mol),
                                                      target_b,
                                                      target_l,
                                                      info=info,
                                                      rcutfac=rcutfac,
                                                      rc=rc,
                                                      rfac0=rfac0,
                                                      twojmax=twojmax,
                                                      weights=weights,
                                                      loss_threshold=loss_threshold,
                                                      max_N=max_N,
                                                      lr=lr,
                                                      alpha=alpha,
                                                      eta=eta,
                                                      )
                                    
            print('conv_type:', conv_type)    
                            
            # Appending to Empty Lists for each Molecule
            # - if get_trj -> return many iterations of xyz coords - used for movies
            if get_trj:
                trjs_set.append(trj[0:-1:20])
            # - else just return end one
            else:
                trjs_set.append(trj[-1])
            losses_set.append(loss)
            
            # Save trajectory
            if save_trj:
                for n in np.arange(1, len(trj), 20):  
                    if n==1:       
                        write(trj_path+conv_type+'/trj_{}.xyz'.format(last_trj_dict[conv_type]+i+1), trj[n])
                    else:
                        write(trj_path+conv_type+'/trj_{}.xyz'.format(last_trj_dict[conv_type]+i+1), trj[n], append=True)
                        
            # Save end xyz
            if save_end:
                write(end_path+conv_type+'/end_{}.xyz'.format(last_end_dict[conv_type]+i), trj[-1])

        # Appending to Empty Lists for each Set        
        trjs.append(trjs_set)
        losses.append(losses_set)
    
    return trjs, losses, save_path
        
def read_xyz(num_mols, path):
    """
    = Function that randomly reads the xyz coords and labels for a specified 
      number of molecules from a specified .xyz file.

    INPUTS:
        1) num_mols - number of molecules to return
        2) path - path to the .xyz file from which to read the data 
        
    OUTPUTS:
        1) xyz - list of xyz coords - shape [N_mol][N_at][3]
        2) l - list of str labels - shape [N_mol][N_at]
        
    NOTES:
        - only tested for benzene, need to try for toulene
    """
    xyz, l = [], []
    
    with open(path) as f: 
        #list of each line in big .xyz
        data = f.read()
        data = data.split('\n')
        
        # get loop parameters
        N_lines = len(data)
        N_at = int(data[0])
        N_mol = N_lines // (N_at+2)

        # array of indices of first line of each mol (e.g. N_mol) 
        all_inds = np.arange(0,N_lines-(N_at+2),N_at+2)
        shuffle(all_inds)
        inds = all_inds[:num_mols]
        
        for ind in inds:
        
            xyz_mol, l_mol = [], []
    
            # list of xyz coords of each mol
            mol_data = data[2+ind:2+ind+N_at] 
            mol_data = [q.split() for q in mol_data]                   

            # write coords and forces into lines of small .xyz
            for n in range(N_at):                
                if len(mol_data[0]) == 4:
                    (atom, x, y, z) = mol_data[n]
                elif len(mol_data[0]) == 7:
                    (atom, x, y, z, _, _, _) = mol_data[n]
                xyz_mol.append([float(i) for i in [x,y,z]])
                l_mol.append(atom) 
    
            xyz.append(xyz_mol)
            l.append(l_mol)
            
    return xyz, l
    
def read_xyz_dir(num_mols, dir_path):
    """
    = Function that randomly reads the xyz coords and labels for a specified 
      number of molecules from a specified directory containing many small .xyz files.

    INPUTS:
        1) num_mols - number of molecules to return
        2) dir_path - path to the directory containing .xyz files from which to read the data 
        
    OUTPUTS:
        1) xyz - list of xyz coords - shape [N_mol][N_at][3]
        2) l - list of str labels - shape [N_mol][N_at]
        
    NOTES:
        - only tested for benzene, need to try for toulene
        - only works for .xyz files without forces
    """
    xyz, l = [], []
    
    fls = os.listdir(dir_path)
    shuffle(fls)
    
    if num_mols > len(fls):
        num_mols = len(fls)
    
    if dir_path[-1] != '/':
        dir_path += '/'
    
    for i in range(num_mols):
        with open(dir_path+fls[i]) as f: 
            #list of each line in big .xyz
            data = f.read()
            data = data.split('\n')
            
            # get loop parameters
            N_at = int(data[0])

            xyz_mol, l_mol = [], []

            # list of xyz coords of each mol
            mol_data = data[2:2+N_at] 
            mol_data = [q.split() for q in mol_data]   
            
            # write coords and forces into lines of small .xyz
            for n in range(N_at):
                if len(mol_data[0]) == 4:
                    (atom, x, y, z) = mol_data[n]
                elif len(mol_data[0]) == 7:
                    (atom, x, y, z, _, _, _) = mol_data[n]
                
                xyz_mol.append([float(i) for i in [x,y,z]])
                l_mol.append(atom) 

            xyz.append(xyz_mol)
            l.append(l_mol)
            
    return xyz, l 
    
def save_xyz(_xyz, _l, save_path, file_name):
    """
    = Function that saves the inputted xyz coords and labels as a 
      .xyz file to a specified path.

    INPUTS:
        1) _xyz - cartesian coordinates of all atoms in molecule 
               - list of list of [floats] of shape [N_mol][N_at][3] or [N_mol][N_at][6] 
        2) _l - atomic labels of atoms in xyz
              - list of list of [str] or [int] of shape [N_mol][N_at]
        3) save_path - [str] path to the directory in which to save the .xyz file.
        4) file_name - [str] name of file to be saved.
        5) single_xyz_flag - [int] f flag spec 
        
    OUTPUTS:
        1) a .xyz file containing inputs xyz and l located at save_path/file_name
        
    NOTES:
        - atomic labels always converted to str type labels (from int type)
        - also works for and saves forces correctly 
    """
    # Add '/' to end if not there
    if save_path[-1] != '/':
        save_path += '/'
        
    # Make Save_Dir if doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Add '.xyz' onto end of file_name
    if file_name[-4:] != '.xyz':
        file_name += '.xyz'

    # if only a single xyz mol is supplied - wither
    if type(_xyz[0][0]) == type(1.0) or type(_xyz[0][0]) == type(np.array([1.])[0]) or type(_xyz[0][0]) == type(torch.tensor([1.])[0]):
        _xyz, _l = [_xyz], [_l]

    # Defining Parameters and Converting L to str type    
    n_mol = len(_l)
    n_at = len(_l[0])
    _l = [inv_convert_labels(_l[i]) for i in range(n_mol)]
    
    with open(save_path+file_name, 'w') as f: 
        for i_mol in range(n_mol):
            f.write('{}\n'.format(n_at))
            f.write('Lattice="100.0 0.0 0.0 0.0 100.0 0.0 0.0 0.0 100.0" Properties=species:S:1:pos:R:3 pbc="F F F"\n')

            # write coords and forces into lines of small .xyz
            for i_at in range(n_at):
                atom = _l[i_mol][i_at]
                if len(_xyz[i_mol][i_at]) == 3:
                    (x,y,z) = _xyz[i_mol][i_at] 
                    f.write("{}\t{}\t{}\t{}\n".format(atom,x,y,z))
                elif len(_xyz[i_mol][i_at]) == 6:
                    (x,y,z,fx,fy,fz) = _xyz[i_mol][i_at] 
                    f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(atom,x,y,z,fx,fy,fz))
                else:
                    print('ERROR: Unknown shape of XYZ')
                    print('len(xyz[n_mol][n_at]:',len(_xyz[i_mol][i_at]))
        print('\n----------------------------------')
        print('Successfully saved file: {}'.format(file_name))
    print('To path: {}'.format(save_path))

def create_random_xyz(dir_path, old_file_name, new_file_name, fac=1.7, offset=0.0, update_old=False):
    """

    """
    old = open(dir_path+old_file_name, 'r')
    _data = old.read()
    _data = _data.split('\n')
    N_at = int(_data[0])
    old.close()

    with open(dir_path+new_file_name, 'w') as new: 
            
        new.write("{}\n{}\n".format(_data[0],_data[1]))
        # list of xyz coords of each mol
        mol_data = _data[2:2+N_at] 
        mol_data = [q.split() for q in mol_data] 
                         
        rand_coords = np.random.rand(N_at,3) * fac + offset
        
        for n in range(N_at):
            if len(mol_data[0]) == 4:
                (atom, x, y, z) = mol_data[n]
            elif len(mol_data[0]) == 7:
                (atom, x, y, z, _, _, _) = mol_data[n]
            (xr, yr, zr) = rand_coords[n]
            if update_old:
                (x,y,z) = [float(i)*fac+offset for i in [x,y,z]]
                new.write("{}\t{}\t{}\t{}\n".format(atom,x,y,z))
            else:
                new.write("{}\t{}\t{}\t{}\n".format(atom,xr,yr,zr))
                
def get_dir_name(names):
    """
    = Function that creates the appropriate path directory name based on
      the training datasets included.

    INPUTS:
        1) names - list of str names of training datasets as from loading module e.g. ['12_benzene','15_toulene']
        
    OUTPUTS:
        1) dir_name - str name of directory e.g. '12_15'
    """
    dir_name = '_'.join([name[:2] for name in names])
    
    return dir_name
    
    
def plot_invert_loss(loss, save=True, show=True):

    # Plot loss 
    plt.figure()
    plt.plot(np.arange(len(loss)), loss, c='r')
    plt.title('Plot of Loss for Bispectrum Inversion')
    plt.xlabel('Step')
    plt.ylabel('Loss')   
    plt.ydata('log')     
    
    if save:
        now = datetime.now()
        time_date = now.strftime('(%H%M_%d%m)')
        plt.savefig(config.base_path + '/gan/plots/inv_loss_'+str(time_date)+'.png')
    
    if show:
        plt.show()
        
        
def center_xyz_mol(num_mols, dir_path, file_name, add_number=50, new_file_name=None):
    
    xyz, l = read_xyz(num_mols, dir_path + file_name)
    xyz = np.add(xyz, add_number)
    
    print('\nSuccessfully translated molecule position')
    print('New 0th atom position:\n', xyz[0])
    
    if new_file_name is None:
        save_xyz(xyz, l, dir_path, file_name)
    else:
        save_xyz(xyz, l, dir_path, new_file_name)
        
def pyscf_atom_parser(_xyz_coords, _species_labels):
    """
    = function to parse xyz coords into proper pyscf format
    
    INPUTS:
        1) _xyz_coords - list of floats of cart coords - shape [N_at,3]
        2) _species_labels - list of ints for species type - must convert to str labels
        
    OUTPUTS:
        1) scf_coords - list of strs
    """

    scf_coords = []
    _species_labels = inv_convert_labels(_species_labels)
    
    # write coords and forces into lines of small .xyz
    for n in range(len(_xyz_coords)):
        (x, y, z) = _xyz_coords[n]
        atom = _species_labels[n]
        
        scf_coords.append('{},{},{},{}'.format(atom,x,y,z))
        
    return ';'.join(scf_coords)

def dft(_xyz, _l):

    scf_coords = pyscf_atom_parser(_xyz, _l)

    mol_py = gto.M(atom=scf_coords) #basis=...

    mf = mol_py.RKS()
    mf.xc = "pbe"

    mf.kernel()

    #E_dft.append(mf.e_tot*627.5)
    #E_dft.append(mf.e_tot)

    return mf.e_tot*627.5    

def create_ase_mol(xyz, l):
    # function to take in a list of xyz and a list of atomic numbers and return ase Atoms object
    l = inv_convert_labels(l)
    return Atoms(l, positions=xyz)

def align_mols(_mols, idx=0):
    """
    = Function that uses <rmsd> package to translate and rotate all ase mols in line with the idx specified mol such that rmsd is minimized
    """
    _mols = copy.deepcopy(_mols)
    _mol0 = _mols[idx]
    pos_0 = _mol0.get_positions()

    # Calc centroid for conformer 0 ignoring Hydrogens                             
    pos_0_noH = np.array([p for i,p in enumerate(pos_0) if _mol0.get_chemical_symbols()[i] != "H"])
    cen_0 = rmsd.centroid(pos_0_noH)
    pos_0 -= cen_0
    pos_0_noH -= cen_0
    _mol0.set_positions(pos_0)

    # Update conformer 0 positions after centroid 
    #for i in range(_mol2.GetNumAtoms()):
    #    x,y,z = pos_0[i]
    #    _mol0.SetAtomPosition(i,Point3D(x,y,z))

    # Weights for kabsch_weighted where w_i=0 if i=H else w_i=1
    w = np.array([1 if _mol0.get_chemical_symbols()[i] != "H" else 0 for i in range(_mol0.get_number_of_atoms())])

    # For all other conformers - trans and rotate to match 0th one
    for _mol in _mols:

        pos_i = _mol.get_positions()

        # Calc centroid for conformer i ignoring Hydrogens                             
        pos_i_noH = np.array([p for j,p in enumerate(pos_i) if _mol.get_chemical_symbols()[j] != "H"])
        cen_i = rmsd.centroid(pos_i_noH)
        pos_i -= cen_i
        pos_i_noH -= cen_i

        U,V,_rmsd = rmsd.kabsch_weighted(pos_i, pos_0, w)
        #print(i,V,_rmsd)
        pos_i = np.dot(pos_i, U)
        _mol.set_positions(pos_i)

        #for i in range(_mol2.GetNumAtoms()):
        #    x,y,z = pos_i[i]
        #    conf_i.SetAtomPosition(i,Point3D(x,y,z))

    return _mols 

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
    


