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

import seaborn as sns
import matplotlib.pyplot as plt
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
parser.add_argument('-mols', type=str, default="0123456789")# '9_ethanol', '9_malonaldehyde', '12_benzene', '12_uracil', '15_toluene', '16_salicylic_acid', '18_naphthalene', '21_aspirin',
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


s_G = ["real" for _ in range(10)]
s_G = ['0022_0110', '1720_0210','0022_0110','1653_0210','0013_0510','0014_0510','0933_0810','0933_0810','1246_1211','1130_0711']
N_max_r = 10000
N_max_g = 200
conv_types = ['conv', 'not_conv', 'not_conv', 'not_conv', 'not_conv', 'not_conv', 'not_conv', 'not_conv', 'not_conv', 'not_conv', 'not_conv' ]
conv_types = ['not_conv', 'not_conv', 'not_conv', 'not_conv', 'not_conv', 'not_conv', 'not_conv', 'not_conv', 'not_conv', 'not_conv', 'not_conv' ]
actual_mol = 7

bonds_flag = 0
if bonds_flag:
    for i in range(10):
        xyz_r, l_r = Xs[i], Ls[i]
        
        # Get Inverted XYZ 
        temp = 0
        if not temp:
            invert_path = "./invert/{}/{}/".format(len(Xs[i][0]), s_G[i])  
            print('invert_path+names[i]',invert_path+names[i])
            xyz_g, l_g = utils.read_xyz_dir(N_max_g, invert_path+names[i]+'/end/'+conv_types[i], sorted_flag=1)
            l_g = [utils.convert_labels(l) for l in l_g]
        else:
            xyz_g = copy.deepcopy(xyz_r)        
            l_g = copy.deepcopy(l_r)
            shuffle(xyz_g)
            shuffle(l_g)

        # Convert lists to arrays
        xyz_r, xyz_g = np.array(xyz_r)[:N_max_r], np.array(xyz_g)[:N_max_g]
        l_r, l_g = np.array(l_r)[:N_max_r], np.array(l_g)[:N_max_g]
        
        #print("xyz_r.shape",xyz_r.shape)
        #print("xyz_g.shape",xyz_g.shape)
        #print("l_r.shape",l_r.shape)
        #print("l_g.shape",l_g.shape)

        # Sort labels and then xyz for Real mols
        sorted_inds = np.argsort(l_r[0])
        for j, l_r_mol in enumerate(l_r):
            xyz_r[j] = xyz_r[j][sorted_inds]
            l_r[j] = l_r_mol[sorted_inds]
            
        # Sort labels and then xyz for Gen mols
        sorted_inds = np.argsort(l_g[0])
        for j, l_g_mol in enumerate(l_g):
            xyz_g[j] = xyz_g[j][sorted_inds]
            l_g[j] = l_g_mol[sorted_inds]

        # Get inter-atomic distances matrices    
        inds = np.triu_indices(n=len(l_r[0]), k=1)
        ia_dists_r = np.array([utils.compute_ia_dists(mol)[inds] for mol in xyz_r]).flatten()
        ia_dists_g = np.array([utils.compute_ia_dists(mol)[inds] for mol in xyz_g]).flatten()
        
        np.save("./thesis/plot_all_bonds/data/{}_r.npy".format(names[i]), ia_dists_r) 
        np.save("./thesis/plot_all_bonds/data/{}_g.npy".format(names[i]), ia_dists_g) 

        #print("ia_dists_r.shape",ia_dists_r.shape)
        #print("ia_dists_g.shape",ia_dists_g.shape)

d_bin = 0.04
label_2 = 'Generated'
#label_2 = 'Inverted'
density = True

plot_flag = 1
if plot_flag:

    data_r, data_g = [], []
    for i in range(len(names)):
        if i == actual_mol:
            data_r.append(np.load("./thesis/plot_all_bonds/data/{}_r.npy".format(names[i])))
            data_g.append(np.load("./thesis/plot_all_bonds/data/{}_g.npy".format(names[i])))

    # Colors and Plot Labels
    data_labels = ['Real', label_2]
    colors = ['C1', 'C0']

    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=False) 

    print(ax)

    #ax.text('Count', fontsize=8)  
    ax.text(0.5, 0.85, r"\underline{%s}" % names2[actual_mol], ha='center', va='center', transform=ax.transAxes, fontsize=15) 
    ax.tick_params(axis='both', labelsize=13)
    ax.set_xlim(0.8,9)
    
    from matplotlib.ticker import MultipleLocator
    ml = MultipleLocator(0.25)
    ax.xaxis.set_minor_locator(ml)
    
    bins0 = np.arange(min(data_r[0]), max(data_r[0]) + d_bin, d_bin)
    bins1 = np.arange(min(data_g[0]), max(data_g[0]) + d_bin, d_bin)
    ax.hist(data_r[0], density=density, bins=bins0, histtype='step', color=colors[0], label=data_labels[0], lw=1)
    ax.hist(data_g[0], density=density, bins=bins1, histtype='step', color=colors[1], label=data_labels[1], lw=1)

    # Plot Settings
    #fig.supylabel('Count', fontsize=8)
    fig.text(0.065, 0.5, 'Frequency [arb. unit]', ha='center', va='center', rotation='vertical', fontsize=17)
    ax.set_xlabel('\Large{$r_{ij}$} [\AA]', fontsize=17)
    ax.legend(fontsize=15, frameon=False,)
    
    #axs[0].scatter(x_train, y_train, s=14, alpha=0.7, color="black", label="Data")
    #axs[0].plot(x_data, y_pred_under, color="red", alpha=0.8, label="Under Fit")
    #axs[0].legend(loc="upper left", frameon=False,)
    #axs[0].set_xticks([])
    #axs[0].set_yticks([])

    #fig.tight_layout()        
    fig.set_figheight(3)   
    #fig.set_figheight(14)
    fig.set_figwidth(10)

    now = datetime.now()
    time_date = now.strftime('(%H%M_%d%m)')
    #plt.savefig('./thesis/plot_all_bonds/plots/all_bonds_'+str(time_date)+'.pdf', bbox_inches='tight')
    plt.savefig('./thesis/plot_all_bonds/plots/all_bonds2_'+str(time_date)+'.png', dpi=500, bbox_inches='tight')
    #plt.show()


