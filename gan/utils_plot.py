#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import torch
import numpy as np
import seaborn as sns
from typing import List, Tuple, Dict

from datetime import datetime
from itertools import combinations

import matplotlib as matplotlib
import matplotlib.pyplot as plt
    
from sklearn.decomposition import PCA

import modules.config as config
from gan import dataset as data
from gan import utils_e as utils

def construct_data_paths(mol_indices: str, set_type: str = "train") -> Tuple[List[str], List[str], List[str], List[int]]:
    """
    Build paths for datasets and print dataset choices.
    
    Parameters
    ----------
    mol_indices
        Must be a string of integers specifying which molecule types from config.MOL_LIST to consider. 
    set_type
        Which dataset type to choose from - one of ["train", "test", "val"]. 
    
    Returns
    -------
    names
        List of molecule type names in standard format (e.g. 12_benzene).
    pretty_names
        List of molecule type names in pretty format (e.g. Benzene), suitable for plotting.
    data_paths
        List of full dataset paths.
    num_mols
        List of number of molecules in each dataset.
    """
    
    names = [config.MOL_LIST[int(i)] for i in mol_indices]
    pretty_names = [config.MOL_LIST3[int(i)] for i in mol_indices]

    data_paths = []
    num_mols = []

    for name in names:
        load_dir_path = os.path.join(config.BASE_PATH, 'data/bispec', name)

        # Print all directories in load_path
        dirs = [dr for dr in os.listdir(os.path.join(load_dir_path,set_type)) if os.path.isdir(os.path.join(load_dir_path,set_type,dr))]

        # If theres only one dataset for chosen molecule then automatically pick that
        if len(dirs) == 1:
            chosen_dr_ind = 0
        # Otherwise choose manually by user input
        else:
            chosen_dr_ind = int(input('\nChoose Set Size:'))
        set_size = dirs[chosen_dr_ind] + '/'

        data_paths.append(os.path.join(load_dir_path, set_type, set_size))
        num_mols.append(int(set_size[:-1]))
        
    return names, pretty_names, data_paths, num_mols

def load_data(data_paths: List[str], num_mols: List[int]) -> Tuple[Dict[str, List[torch.Tensor]], List[dict]]:
    """
    Load descriptors, species labels, XYZ coordinates, and energies from dataset files.
    
    Parameters
    ----------
    data_paths
        List of full dataset paths.
    num_mols
        List of the number of molecules contained within each data_path - used as directory name.
    
    Returns
    -------
    data: dict
        Dictionary containing the following lists of torch.tensors, for each dataset.
            - Original descriptors Bs
            - Descriptors Bs_sum
            - Descriptors Bs_std
            - Species labels Ls
            - Cartesian coordinates Xs
            - Energies Es
            - Energies Es_norm
    info: List[dict]
        List of each moelcule type info dictionary.
    """
       
    # Load combined dataset info dictionary 
    infos = []
    for i in range(len(data_paths)):
        infos.append(np.load(data_paths[i] + str(num_mols[i]) + '_info.npy', allow_pickle=True).item())
    #info = data.combined_info(infos)
        
    # Load data as tensors in dictionary
    data_dict = {'Bs': [], 'Bs_sum': [], 'Bs_std': [], 'Ls': [], 'Xs': [], 'Es': [], 'Es': []}
    for i in range(len(data_paths)):
        b = torch.tensor(np.load(os.path.join(data_paths[i], str(num_mols[i])+'_B.npy')), dtype=torch.float32)
        l = torch.tensor(np.load(os.path.join(data_paths[i], str(num_mols[i])+'_L.npy')), dtype=torch.int32)
        x = torch.tensor(np.load(os.path.join(data_paths[i], str(num_mols[i])+'_X.npy')), dtype=torch.float32)
        e = torch.tensor(np.load(os.path.join(data_paths[i], str(num_mols[i])+'_E.npy')), dtype=torch.float32)

        data_dict['Bs'].append(b), data_dict['Ls'].append(l), data_dict['Xs'].append(x), data_dict['Es'].append(e)

        # Return summed and standardised B
        data_dict['Bs_sum'].append(data.sum_over_species(b, l, infos[i]))
        b_std, std = data.B_scaling(data_dict['Bs_sum'][i], data_dict['Ls'], infos[i])
        data_dict['Bs_std'].append(b_std)
        infos[i]["std"] = std

    return data_dict, infos
    
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
        time_date = now.strftime('(%H%M%S_%d%m)')
        dir_path = GS['model_path'] + '/plots/'
        
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        if b_g is None:
            #plt.savefig('plots/image_b_'+str(time_date)+'.png')
            plt.savefig('plots/image_b_'+str(time_date)+'.pdf', bbox_inches="tight")
        else:
            #plt.savefig(dir_path+str(time_date)+'.png')
            plt.savefig(dir_path+str(time_date)+'.pdf', bbox_inches="tight")
        plt.close()
        
    if show:
        plt.show()
    
def images_b(info, GS, _b, _b_g=None, max_rows=40, save=True, show=False):
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

    # Handy Shape Parameters
    n_mol, n_at, n_feats = _b.shape
        
    # Inverse Feature Scaling - NEED TO IMPLEMENT
    
    # Change from [N_mol,N_at,N_b] [N_mol*N_at,N_b]
    _b = _b.reshape(-1, n_feats) 
    if _b_g is not None:
        _b_g = _b_g.reshape(-1, n_feats) 
        
    # Get reasonable max num of rows in images
    n_mols = max_rows // n_at
    
    _b = _b[:n_mols*n_at].cpu()
    if _b_g is not None:
        _b_g = _b_g[:n_mols*n_at].cpu()
        
    # Create new array with NaNs padding in between rows
    new_b_row = n_mols*n_at+n_mols-1
    new_b = np.empty((new_b_row, n_feats)) * np.nan
    if _b_g is not None:
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
            if _b_g is not None:
                new_b_g[i_row] = _b_g[i_old_row]
    
    plt.figure()
    # only 1 subplot if b_g not supplied else 2 subplots
    num_subplots = 1 if _b_g is None else 2
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
        time_date = now.strftime('(%H%M%S_%d%m)')
        dir_path = GS['model_path'] + '/plots/'
        
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        if _b_g is None:
            plt.savefig('plots/images_b_'+str(time_date)+'.png', dpi=400)
            #plt.savefig('plots/images_b_'+str(time_date)+'_'+set_name+'.pdf', bbox_inches="tight")
        else:
            plt.savefig(dir_path+str(time_date)+'.png', dpi=400)
            #plt.savefig(dir_path+str(time_date)+'.pdf', bbox_inches="tight")
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
            
    # Predict E with model
    ep = E_model(b).cpu().detach().numpy()
    #ep = scale.inv_fs_e(ep, l, info, GS)
    
    if b_g is not None:
        ep_g = E_model(b_g).cpu().detach().numpy()
        #ep_g = scale.inv_fs_e(ep_g, l_g, info, GS)

    #e = scale.inv_fs_e(e, l, info, GS)
    e = e.squeeze(1)
    ep = ep.squeeze(1)
    if b_g is not None:
        #e_g = scale.inv_fs_e(e_g, l_g, info, GS)
        e_g = e_g.squeeze(1)
        ep_g = ep_g.squeeze(1)
    
    plt.figure()
    plt.scatter(e, ep, label='Real', c='red', s=150, alpha=0.5)
    if b_g is not None:
        plt.scatter(e_g, ep_g, label='Fake', c='blue', s=150, alpha=0.5)
    
    plt.plot(e, e, linestyle=':', c='red')

    plt.xlabel('$E_{Desired}$', fontsize=15)
    plt.ylabel('$E_{NN}$', fontsize=15)
    plt.title('Plot of $E_{Desired}$ vs $E_{NN}$', fontsize=20)
    plt.legend()

    # If no b_g then save to .../gan/plots/ otherwise save to .../gan/savednet/                
    if save:
        now = datetime.now()
        time_date = now.strftime('(%H%M_%d%m)')
        dir_path = GS['model_path'] + '/plots/'
        
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        if b_g is None:
            plt.savefig('plots/plot_e_'+str(time_date)+'.png', bbox_inches='tight')
        else:
            plt.savefig(dir_path+'plot_e_'+str(time_date)+'.png', bbox_inches='tight')
        plt.close()
        
    if show:
        plt.show()

def plot_b(info, GS, b, l, b_g=None, l_g=None, save=True, show=True, plot_info=False, seperate=False):
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
    names = info['set_names']
    
    if seperate:
        b = data.species_seperator(b, l, info)
        if b_g is not None:
            b_g = data.species_seperator(b_g, l_g, info)

    # Lists of means and std - shape [N_set][N_species,N_b]
    B_r_mean, B_r_std, B_g_mean, B_g_std = [], [], [], []
    for i_set in range(len(b)):
        B_r_mean.append(torch.mean(b[i_set], dim=(0)))
        B_r_std.append(torch.std(b[i_set], dim=(0)))
        
        if b_g is not None:
            B_g_mean.append(torch.mean(b_g[i_set], dim=(0)))
            B_g_std.append(torch.std(b_g[i_set], dim=(0)))
    
    for i_set in range(len(b)):
        plt.figure(figsize=(10,6))
        i_plot = 0
        unq_species = np.unique(l[i_set])
        num_species = len(unq_species)

        for i_species, species in enumerate(unq_species):
            species_name = config.SPECIES_TO_SYMBOL[species]
            
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
            plt.legend(fontsize=8, loc='upper left')
            plt.grid(True)
            
            # iterate to new subplot
            i_plot += 1
            
        # If no b_g then save to .../gan/plots/ otherwise save to .../gan/savednet/                
        if save:
            now = datetime.now()
            time_date = now.strftime('(%H%M%S_%d%m)')
            dir_path = GS['model_path'] + '/plots/'
            
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            if b_g is None:
                plt.savefig('plots/plot_b_'+str(time_date)+'.png', bbox_inches='tight')
            else:
                plt.savefig(dir_path+'plot_b_'+str(time_date)+'.png', bbox_inches='tight')
            plt.close()
            
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
                      label=config.SPECIES_TO_SYMBOL[species]+': $R_{cut}$'+' ='+str(np.round(rc_dict[species],2))+'$\AA$')

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
        plt.savefig(os.path.join(config.BASE_PATH, 'gan/plots/rc_'+str(time_date)+'.png'), format='png')
                
    if show:
        plt.show()

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
    ia_dists_r = np.array([utils.compute_ia_dists(mol)[inds] for mol in xyz_r])
    ia_dists_g = np.array([utils.compute_ia_dists(mol)[inds] for mol in xyz_g])
    
    # Find unique species combinations (e.g. for benzene = C-C,H-H,C-H)
    str_labels = utils.inv_convert_labels(l_r[0])
    bonds = utils.triu_label_combinations(str_labels)
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
        plt.savefig(config.BASE_PATH + '/gan/plots/bonds_'+str(time_date)+'.png')
    
    if show:
        plt.show()

def plot_bonds_all(xyz_r, l_r, xyz_g, l_g, save=True, show=True, sns_flag=False, num_bins=200, label_2='G', density=True):
    """
    = Function that plots the bond length distribution in inputted molecules 

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
    ia_dists_r = np.array([utils.compute_ia_dists(mol)[inds] for mol in xyz_r]).flatten()
    ia_dists_g = np.array([utils.compute_ia_dists(mol)[inds] for mol in xyz_g]).flatten()
    data = [ia_dists_r, ia_dists_g]

    print("ia_dists_r.shape",ia_dists_r.shape)
    print("ia_dists_g.shape",ia_dists_g.shape)

    # Colors and Plot Labels
    data_labels = ['R', label_2]
    colors = ['C0', 'C2']

    fig, ax = plt.subplots()    

    for i in range(len(data_labels)):
        if sns_flag:
            sns.distplot(data[i], bins=num_bins, color=colors[i], label=data_labels[i], ax=ax)
        else:
            ax.hist(data[i], density=density, bins=num_bins, histtype='step', color=colors[i], label=data_labels[i], lw=1)

    # Plot Settings
    ax.set_ylabel('Count', fontsize=8)
    ax.set_xlabel('Distance ($\AA$)', fontsize=8)
    ax.legend(fontsize=8)
    ax.set_title('Histograms of Bond Length Distributions', fontsize=10)

    fig.tight_layout()        
    
    if save:
        now = datetime.now()
        time_date = now.strftime('(%H%M_%d%m)')
        plt.savefig(config.BASE_PATH + '/gan/plots/bonds_all_'+str(time_date)+'.png')
    
    if show:
        plt.show()
    
def plot_angles(xyz_r, l_r, xyz_g, l_g, save=True, show=True, sns_flag=False, num_bins=200, label_2='G', density=True, num_r=5000):
    """
    = Function that plots the bond length distribution in inputted molecules 

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

    from ase import Atoms

    # Convert lists to arrays
    xyz_r, xyz_g = np.array(xyz_r), np.array(xyz_g)
    l_r, l_g = np.array(l_r), np.array(l_g)

    # Convert to ASE atoms
    l_r2, l_g2 = utils.inv_convert_labels(l_r), utils.inv_convert_labels(l_g) 

    atoms_r = [Atoms(l_r[i], positions=xyz_r[i]) for i in range(len(xyz_r))][:num_r]
    atoms_g = [Atoms(l_g[i], positions=xyz_g[i]) for i in range(len(xyz_g))]

    n_atoms = len(l_r[0])
    #angles_inds = set()
    #angles_r = {}

    #for atom_r in atoms_r:
    #    for i in range(n_atoms):
    #        for j in range(i+1, n_atoms):
    #            for k in range(j+2, n_atoms):
    #                a = f"{i}{j}{k}"
    #                if a not in angles_inds:
    #                    angles_r[a] = atom_r.get_angle(i, j, k)
    #                    angles_inds.add(a)     
    
    triplets = combinations(range(n_atoms), 3)
    
    angles_r, angles_g = [], []
    for l,m,n in triplets:
        for atom_r in atoms_r:
            angles_r.append(atom_r.get_angle(l,m,n))
        for atom_g in atoms_g:
            angles_g.append(atom_g.get_angle(l,m,n))

    data = [angles_r, angles_g]

    # Colors and Plot Labels
    data_labels = ['R', label_2]
    colors = ['C0', 'C2']

    fig, ax = plt.subplots()    
    for i in range(len(data_labels)):
        if sns_flag:
            sns.distplot(data[i], bins=num_bins, color=colors[i], label=data_labels[i], ax=ax)
        else:
            ax.hist(data[i], density=density, bins=num_bins, histtype='step', color=colors[i], label=data_labels[i], lw=1)

    # Plot Settings
    ax.set_ylabel('Count', fontsize=8)
    ax.set_xlabel('Distance ($\AA$)', fontsize=8)
    ax.legend(fontsize=8)
    ax.set_title('Histograms of Bond Angle Distributions', fontsize=10)

    fig.tight_layout()        
    
    if save:
        now = datetime.now()
        time_date = now.strftime('(%H%M_%d%m)')
        plt.savefig(config.BASE_PATH + '/gan/plots/bonds_angles_'+str(time_date)+'.png')
    
    if show:
        plt.show()
 
def plot_invert_loss(loss, save=True, show=True):

    # Plot loss 
    plt.figure()
    plt.plot(np.arange(len(loss)), loss, c='r')
    plt.title('Plot of Loss for Bispectrum Inversion')
    plt.xlabel('Step')
    plt.ylabel('Loss')   
    plt.yscale('log')     
    
    if save:
        now = datetime.now()
        time_date = now.strftime('(%H%M_%d%m)')
        plt.savefig(config.BASE_PATH + '/gan/plots/inv_loss_'+str(time_date)+'.png')
    
    if show:
        plt.show()
        
def plot_nne_acc(
    E_real: torch.Tensor, 
    E_pred: torch.Tensor, 
    nn_E: torch.nn.Module, 
    test_dl: torch.utils.data.DataLoader, 
    GS: dict
) -> None:
    """
    Function that plots the alignment of nn_E predicted energies vs real energies as a parity plot.
    """

    nn_E.eval()
    B_real2, L_real, E_real2 = next(iter(test_dl))

    with torch.no_grad():
        E_pred2 = nn_E(B_real2.to(GS['device']))

    plt.figure()
    plt.scatter(E_real.cpu().detach(), E_pred.cpu().detach(), label='train', c='red', s=150, alpha=0.5)
    plt.scatter(E_real2.cpu().detach(), E_pred2.cpu().detach(), label='test', c='green', s=150, alpha=0.5)
    plt.plot(E_real.cpu().detach(), E_real.cpu().detach(), linestyle=':', c='red')

    plt.xlabel('$E_{DFT}$', fontsize=15)
    plt.ylabel('$E_{pred}$', fontsize=15)
    plt.title('Plot of $E_{DFT}$ vs $E_{pred}$', fontsize=20)
    plt.legend()

    now = datetime.now()
    time_date = now.strftime('(%H%M_%d%m)')

    plt.savefig(os.path.join(GS['model_path'], 'plots/nn_E_acc_{}.png'.format(time_date)), bbox_inches='tight')
    plt.close()

def plot_nne_loss(nn_E: torch.nn.Module, Cost: torch.Tensor, GS: dict) -> None:
    """
    Function that plots the loss function values of energy prediction model for both training and test datasets.
    """

    nn_E.load_state_dict(torch.load(os.path.join(GS['model_path'], 'E')))
    np.save(os.path.join(GS['model_path'], 'Cost_E.npy'), Cost)

    plt.figure()
    plt.plot(range(len(Cost[:,0])), Cost[:,0], label='train')
    plt.plot(range(len(Cost[:,1])), Cost[:,1], label='test')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.yscale('log')
    plt.legend()
    plt.savefig(os.path.join(GS['model_path'], 'plots/nn_E_cost.png'), bbox_inches='tight')
    plt.close()

    
