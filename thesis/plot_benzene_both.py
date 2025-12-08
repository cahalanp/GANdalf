#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Version:
    - Generate samples from a given nn_G
        - can use generate_like_train()
        - or generate_for_label() (not implemented - workaround is to feed as B_real, L_real)
    - Set values for lr, alpha, eta (from invert_b2.py)
    - Invert samples
    - Make plot_bonds with inverted samples
"""

#-----------------------------------------------------------------------------------------------------------------------
# - Importing Modules
# - Arg-Parser 
#-----------------------------------------------------------------------------------------------------------------------     

import os
import sys
import copy
import time
import argparse
import numpy as np
from datetime import datetime
from random import shuffle
from itertools import combinations

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
plt.rcParams['text.usetex'] = True

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pyscf import gto, dft, scf

from ase import Atoms
from ase.io import Trajectory, write, read
from ase.visualize import view
import ase.gui as Gui

sys.path.append('../modules')
sys.path.append('../')

#import dataset as data 
import dataset_e2 as data 
#import utils as utils
import utils_e as utils
import config

import G_b3 as G # basic linear G w/ comp

from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser()
parser.add_argument('-mols', type=str, default="246")# '9_ethanol', '9_malonaldehyde', '12_benzene', '12_uracil', '15_toluene', '16_salicylic_acid', '18_naphthalene', '21_aspirin',
args = parser.parse_args()

torch.set_default_tensor_type(torch.DoubleTensor)

#-----------------------------------------------------------------------------------------------------------------------
# - Set Paths
# - Load GS Dict
# - XYZ Data Path
#-----------------------------------------------------------------------------------------------------------------------     
   
# Data Paths Info
names = [config.mol_list[int(i)] for i in args.mols]
names2 = [config.mol_list3[int(i)] for i in args.mols]

# Choose DataSets and Set Sizes and make Paths
data_paths = []
num_mols = []
for i_set in range(len(args.mols)):
    load_dir_path = os.path.join(config.base_path, 'data/bispec/{}/'.format(names[i_set]))

    # Print all directories in load_path
    dirs = [dr for dr in os.listdir(load_dir_path) if os.path.isdir(load_dir_path+dr)]
    print('\n-------------------------------------------------')
    print('Printing all Set Types for {}:'.format(names[i_set]))
    for i, dr in enumerate(dirs):
        print(' {}) {}'.format(i, dr))

    # If theres only one dataset for chosen molecule then automatically pick that
    if len(dirs) == 1:
        chosen_dr_ind = 0
    # Otherwise choose manually by user input
    else:
        chosen_dr_ind = int(input('\nChoose Set Type:'))
    set_type = dirs[chosen_dr_ind] + '/'

    # Print all directories in load_path
    dirs = [dr for dr in os.listdir(load_dir_path+set_type) if os.path.isdir(load_dir_path+set_type+dr)]
    #print('\n-------------------------------------------------')
    #print('\nPrinting all Set Sizes for {}:'.format(names[i_set]))
    #for i, dr in enumerate(dirs):
    #    print(' {}) {}'.format(i, dr))

    # If theres only one dataset for chosen molecule then automatically pick that
    if len(dirs) == 1:
        chosen_dr_ind = 0
    # Otherwise choose manually by user input
    else:
        chosen_dr_ind = int(input('\nChoose Set Size:'))
    set_size = dirs[chosen_dr_ind] + '/'

    data_paths.append(load_dir_path + set_type + set_size)
    num_mols.append(int(set_size[:-1]))
    
# Print Load/Save Paths
print('\nLoading Data From:')
for i in range(len(args.mols)):
    print('    {}) {}'.format(i, data_paths[i]))

#-----------------------------------------------------------------------------------------------------------------------
# - Load B Data
# - Combine Info Dicts
# - Load Model
#-----------------------------------------------------------------------------------------------------------------------  
   
# Load Info - seperate to load data because args.e
infos = []
for i in range(len(args.mols)):
    infos.append(np.load(data_paths[i] + str(num_mols[i]) + '_info.npy', allow_pickle=True).item())
info = data.combined_info(infos)
    
# Load B and L Data
Bs, Xs, Es, Ls = [], [], [], []
for i in range(len(args.mols)):
    #Bs.append(torch.tensor(np.load(data_paths[i] + str(num_mols[i]) + '_B.npy')))
    Xs.append(torch.tensor(np.load(data_paths[i] + str(num_mols[i]) + '_X.npy')))
    Es.append(torch.tensor(np.load(data_paths[i] + str(num_mols[i]) + '_E.npy')))
    Ls.append(torch.tensor(np.load(data_paths[i] + str(num_mols[i]) + '_L.npy')))

# Add B_min and B_max to Info for inv_fs (=2,4)
#bs2 = copy.deepcopy(Bs)    
#Bs2_sum = data.sum_over_species(bs2, Ls, info)

#std = StandardScaler()
#Bs2_sum = [torch.tensor(std.fit_transform(Bs2_sum[0].reshape((-1, info['N_all_species']*info['N_feats'])))).reshape((-1, info['N_all_species'], info['N_feats']))]
#info["std"] = std

# Print Combined Dataset Info
print('\n-Combined Dataset Information')
for i in info:
    print('%s:' % i, info[i])
    
GS = {"mols": args.mols}
     
#-----------------------------------------------------------------------------------------------------------------------
# - Generate XYZ Coords using gen_xyz()
#-----------------------------------------------------------------------------------------------------------------------    

s_G = "real"
s_G = "0043_1511"
conv_type = "not_conv"
N_max_r = 10000
N_max_g = 300
set_ind = 2
#s_G = [9 model names]

data_flag = 1
if data_flag:

    """
    bonds
    """    

    data = []
    xyz_r, l_r = Xs[set_ind], Ls[set_ind]
    
    # Get Inverted XYZ 
    temp = 0
    if not temp:
        dir_name = utils.get_dir_name(names)
        invert_path = "./invert/{}/{}/".format(dir_name, s_G)  
        print('invert_path+names[set_ind]',invert_path+names[set_ind])
        xyz_g, l_g = utils.read_xyz_dir(N_max_r, invert_path+names[set_ind]+'/end/'+conv_type, sorted_flag=1)
        l_g = [utils.convert_labels(l) for l in l_g]
        
    else:
        xyz_g = copy.deepcopy(xyz_r)        
        l_g = copy.deepcopy(l_r)
        shuffle(xyz_g)
        shuffle(l_g)

    # Convert lists to arrays
    xyz_r, xyz_g = np.array(xyz_r)[:N_max_r], np.array(xyz_g)[:N_max_g]
    l_r, l_g = np.array(l_r)[:N_max_r], np.array(l_g)[:N_max_g]
    
    # Sort labels and then xyz for Real mols
    sorted_inds = np.argsort(l_r[0])
    for j, l_r_mol in enumerate(l_r):
        xyz_r[j] = xyz_r[j][sorted_inds]
        l_r[j] = l_r_mol[sorted_inds]
        
    print("xyz_g",xyz_g)
    print("l_g",l_g)
        
    # Sort labels and then xyz for Gen mols
    sorted_inds = np.argsort(l_g[0])
    for j, l_g_mol in enumerate(l_g):
        xyz_g[j] = xyz_g[j][sorted_inds]
        l_g[j] = l_g_mol[sorted_inds]
        
    # Get inter-atomic distances matrices    
    inds = np.triu_indices(n=len(l_r[0]), k=1)
    ia_dists_r = np.array([utils.compute_ia_dists(mol)[inds] for mol in xyz_r])
    ia_dists_g = np.array([utils.compute_ia_dists(mol)[inds] for mol in xyz_g])
    
    #data_b = np.array([ia_dists_r, ia_dists_g])
    
    # Find unique species combinations (e.g. for benzene = C-C,H-H,C-H)
    str_labels = utils.inv_convert_labels(l_r[0])
    bonds = utils.triu_label_combinations(str_labels)
    unq_bonds = np.unique(bonds)
    print("unq_bonds", unq_bonds)
    unq_bonds = ['HH', 'CH', 'CC']
    print("\nstr_labels", str_labels)
    print("unq_bonds", unq_bonds)

    # Find Interatomic Dists corresponding to this bond 
    data_br, data_bg = [], []
    for i in range(len(unq_bonds)):
        inds = np.where(bonds == unq_bonds[i])[0]
        #data_i = [ia_dists_r[:, inds].flatten(), ia_dists_g[:, inds].flatten()]
        data_br.append(ia_dists_r[:, inds].flatten())
        data_bg.append(ia_dists_g[:, inds].flatten())
        
        np.save("./thesis/plot_benzene_multi/data/{}_{}_bonds_r.npy".format(names[set_ind], unq_bonds[i]), ia_dists_r[:, inds].flatten()) 
        np.save("./thesis/plot_benzene_multi/data/{}_{}_bonds_g.npy".format(names[set_ind], unq_bonds[i]), ia_dists_g[:, inds].flatten()) 
        
    #sys.exit()
    """
    angles
    """    
        
    from FOX import MultiMolecule
    
    labels_dict = {}
    for i,s in enumerate(str_labels):
        if s not in labels_dict.keys():
            labels_dict[s] = [i]
        else:
            labels_dict[s].append(i)
    print("labels_dict",labels_dict)
    mols_r = MultiMolecule(xyz_r, labels_dict)
    mols_g = MultiMolecule(xyz_g, labels_dict)
    
    H_perms = [["C","H","C"], ["C","H","H"], ["H","H","H"]]
    C_perms = [["C","C","C"], ["C","C","H"], ["H","C","H"]]
    all_perms = [H_perms, C_perms]
    
    adf_r = mols_r.init_adf(r_max=5)
    adf_g = mols_g.init_adf(r_max=5)
    #adf_r = mols_r.init_adf(weight=None)
    #adf_g = mols_r.init_adf(weight=None)
        
    data_ar, data_ag = np.zeros((3,180)), np.zeros((3,180))
    for i,perms in enumerate(all_perms):
        for perm in perms:
            data_ar[i] += adf_r[" ".join(perm)].to_numpy()
            data_ag[i] += adf_g[" ".join(perm)].to_numpy()
            
    np.save("./thesis/plot_benzene_multi/data/{}_angle_dist_r.npy".format(names[set_ind]), data_ar) 
    np.save("./thesis/plot_benzene_multi/data/{}_angle_dist_g.npy".format(names[set_ind]), data_ag) 
            
label_2 = 'Inverted'
label_2 = 'Generated'
density = True

plot_flag = 1
if plot_flag:

    # Colors and Plot Labels
    data_labels = ['Real', label_2]
    colors = ['C1', 'C0']
    names_ax = ['C-C', 'C-H', 'H-H', 'X-C-X', 'X-H-X']

    #fig = plt.figure()
    #gs = GridSpec(17, 6, figure=fig)
    #ax1 = fig.add_subplot(gs[:3, :])
    #ax2 = fig.add_subplot(gs[3:6, :])
    #ax3 = fig.add_subplot(gs[6:9, :])
    #ax4 = fig.add_subplot(gs[10:13, :])
    #ax5 = fig.add_subplot(gs[13:16, :])
    #axs = [ax1, ax2, ax3, ax4, ax5]
    
    fig = plt.figure()
    gs = GridSpec(6, 6, figure=fig)
    ax1 = fig.add_subplot(gs[:2, :3])
    ax2 = fig.add_subplot(gs[2:4, :3])
    ax3 = fig.add_subplot(gs[4:6, :3])
    ax4 = fig.add_subplot(gs[:3, 3:])
    ax5 = fig.add_subplot(gs[3:6, 3:])
    axs = [ax1, ax2, ax3, ax4, ax5]
    
    #fig, axs = plt.subplots(nrows=5, ncols=1, sharex=False, sharey=False) 
    
    bins = [50,80,80]
    #bins = [20,20,15,50,50]

    for i in [0, 1, 2]:
        axs[i].hist(data_br[i], density=density, bins=bins[i], histtype='step', color=colors[0], label=data_labels[0], lw=1)
        axs[i].hist(data_bg[i], density=density, bins=bins[i], histtype='step', color=colors[1], label=data_labels[1], lw=1)
        
        axs[i].set_title(r"\underline{%s}" % names_ax[i], fontsize=17)
        #axs[i].text(0.5, 0.85, r"\underline{%s}" % names_ax[i], ha='center', va='center', transform=axs[i].transAxes, fontsize=17)
        #axs[i].set_title(str(names_ax[i]), fontsize=10)
        
        axs[i].xaxis.set_major_locator(plt.MaxNLocator(5))
        axs[i].yaxis.set_major_locator(plt.MaxNLocator(3))
        #axs[i].set_xlabel('$r_{ij}$ [\AA]', fontsize=17)
        #from matplotlib.ticker import MultipleLocator
        #ml = MultipleLocator(0.1)
        #axs[i].xaxis.set_minor_locator(ml)
        axs[i].tick_params(axis='both', which='major', labelsize=15)

    sum_bar = 2 # must divide into 180 evenly
    n_bar = 180//sum_bar
    for i,j in enumerate([3, 4]):
    
        # normalise 
        data_ar[i] = data_ar[i] / sum(data_ar[i])
        data_ag[i] = data_ag[i] / sum(data_ag[i])
    
        dummy_x = np.arange(n_bar)
        _, _, poly_r = axs[j].hist(dummy_x, density=density, bins=dummy_x, histtype='step', color=colors[0], label=data_labels[0], lw=1)
        _, _, poly_g = axs[j].hist(dummy_x, density=density, bins=dummy_x, histtype='step', color=colors[1], label=data_labels[1], lw=1)

        xy_r, xy_g = poly_r[0].get_xy(), poly_g[0].get_xy()
        
        data_ar_i, data_ag_i = np.zeros(n_bar), np.zeros(n_bar)
        for l,k in enumerate(np.arange(0, 180, sum_bar)):
            data_ar_i[l] = sum(data_ar[i][k:k+sum_bar])/sum_bar
            data_ag_i[l] = sum(data_ag[i][k:k+sum_bar])/sum_bar
        
        for k in range(1,n_bar):
            xy_r[(k*2)-1][1] = data_ar_i[k]
            xy_r[(k*2)][1] = data_ar_i[k]
            xy_g[(k*2)-1][1] = data_ag_i[k]
            xy_g[(k*2)][1] = data_ag_i[k]
        
        poly_r[0].set_xy(xy_r), poly_g[0].set_xy(xy_g)
        
        axs[j].set_ylim(0,max(data_ag[i])*1.2)
        #axs[j].set_xlim(0,n_bar)
        #axs[j].set_title(str(names_ax[j]), fontsize=12)
        axs[j].set_title(r"\underline{%s}" % names_ax[j], fontsize=17)
        #axs[j].text(0.5, 0.85, r"\underline{%s}" % names_ax[i], ha='center', va='center', transform=axs[j].transAxes, fontsize=17)
        #axs[j].set_xlabel(r'$\mathrm{\theta}_{ijk}$ [deg]', fontsize=17)
        
        axs[j].set_xticks([x/sum_bar for x in [0, 45, 90, 135, 180]]) 
        #axs[j].set_xticks([0, 30, 60]) 
        #axs[j].xaxis.set_major_locator(plt.MaxNLocator(5))
        axs[j].set_xticklabels(['0', '45', '90', '135', '180'])
        axs[j].yaxis.set_major_locator(plt.MaxNLocator(3))
        #from matplotlib.ticker import MultipleLocator
        #ml = MultipleLocator(15)
        #axs[j].xaxis.set_minor_locator(ml)
        axs[j].tick_params(axis='both', which='major', labelsize=15)

    # Plot Settings
    #fig.supylabel('Count', fontsize=8)
    fig.text(-0.005, 0.48, 'Frequency [arb. unit]', ha='center', va='center', rotation='vertical', fontsize=17)
    axs[3].legend(fontsize=14, frameon=False,)
    axs[2].set_xlabel('$r_{ij}$ [\AA]', fontsize=17)
    axs[4].set_xlabel(r'$\mathrm{\theta}_{ijk}$ [deg]', fontsize=17)
    
    #axs[0].scatter(x_train, y_train, s=14, alpha=0.7, color="black", label="Data")
    #axs[0].plot(x_data, y_pred_under, color="red", alpha=0.8, label="Under Fit")
    #axs[0].legend(loc="upper left", frameon=False,)
    #axs[0].set_xticks([])
    #axs[0].set_yticks([])

    fig.set_figheight(7)
    fig.set_figwidth(10)
    #fig.tight_layout()  
    fig.tight_layout(h_pad=1.0)  
    #fig.tight_layout(pad=1.1, h_pad=4.0, w_pad=1.1)      
      
    now = datetime.now()
    time_date = now.strftime('(%H%M_%d%m)')
    plt.savefig('./thesis/plot_benzene_multi/plots/{}_{}.pdf'.format(names[set_ind], time_date), bbox_inches="tight")
    #plt.show()


