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
import matplotlib
from datetime import datetime
from random import shuffle
from itertools import combinations

import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
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

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
parser.add_argument('-mols', type=str, default="0")# '9_ethanol', '9_malonaldehyde', '12_benzene', '12_uracil', '15_toluene', '16_salicylic_acid', '18_naphthalene', '21_aspirin',
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
s_G = "0022_0110"
s_G_dict = {'0':'0022_0110', '1':'1720_0210', '2':'0022_0110', '3':'1653_0210', '4':'0013_0510', '5':'0014_0510', '6':'0933_0810', '7':'0933_0810', '8':'1246_1211', '9':'1130_0711'}
s_G = s_G_dict[args.mols]
conv_type = "conv"
conv_type = "not_conv"
N_max_r = 200
N_max_g = 200
#s_G = [9 model names]

data_flag = 1
if data_flag:

    data = []
    xyz_r, l_r, e_r  = Xs[0], Ls[0], Es[0]
    
    # Get Inverted XYZ 
    temp = 0
    if not temp:
        invert_path = "./invert/{}/{}/".format(len(Xs[0][0]), s_G)  
        print('invert_path+names[0]',invert_path+names[0])
        xyz_g, l_g_str, _, e_g = utils.read_xyz_dir(N_max_r, invert_path+names[0]+'/end/'+conv_type, sorted_flag=1, energy_flag=1, dft_energy_flag=1)
        l_g = [utils.convert_labels(l) for l in l_g_str]
        
    else:
        xyz_g = copy.deepcopy(xyz_r)        
        l_g = copy.deepcopy(l_r)
        shuffle(xyz_g)
        shuffle(l_g)

    # Convert lists to arrays
    xyz_r, xyz_g = np.array(xyz_r)[:N_max_r], np.array(xyz_g)[:N_max_g]
    l_r, l_g = np.array(l_r)[:N_max_r], np.array(l_g)[:N_max_g]
    e_r, e_g = np.array(e_r)[:N_max_r], np.array(e_g)[:N_max_g]
    
    # Sort labels and then xyz for Real mols
    #sorted_inds = np.argsort(l_r[0])
    #for j, l_r_mol in enumerate(l_r):
    #    xyz_r[j] = xyz_r[j][sorted_inds]
    #    l_r[j] = l_r_mol[sorted_inds]
        
    # Sort labels and then xyz for Gen mols
    #sorted_inds = np.argsort(l_g[0])
    #for j, l_g_mol in enumerate(l_g):
    #    xyz_g[j] = xyz_g[j][sorted_inds]
    #    l_g[j] = l_g_mol[sorted_inds]
        
    print("xyz_r.shape",np.array(xyz_r).shape)    
    print("xyz_g.shape",np.array(xyz_g).shape) 
    
    #print("xyz_r",xyz_r)    
    #print("xyz_g",xyz_g) 
    
    print("l_g_str[0]",l_g_str[0]) 
    mols_r = [Atoms(l_g_str[0], positions=x) for x in xyz_r]
    mols_g = [Atoms(l_g_str[0], positions=x) for x in xyz_g]
    
    #view(mols_r)
    #view(mols_g)

    mols = mols_r + mols_g
    mols_align = utils.align_mols(mols)
    
    xyz_r = np.array([m.get_positions() for m in mols_align[:N_max_r]])
    xyz_g = np.array([m.get_positions() for m in mols_align[N_max_r:]]) 
    
    xyz_r = xyz_r.reshape((-1, len(l_r[0])*3))
    xyz_g = xyz_g.reshape((-1, len(l_g[0])*3))
    
    print("xyz_r.shape",np.array(xyz_r).shape)    
    print("xyz_g.shape",np.array(xyz_g).shape)
    
    print("xyz_r.shape",np.array(xyz_r).shape) 
    print("e_r.shape",np.array(e_r).shape)
    
    #e_r2 = torch.tensor(e_r[:,None]).repeat(1,10).numpy()
    #e_g2 = torch.tensor(e_g[:,None]).repeat(1,10).numpy()
    #xyz_r = np.concatenate((xyz_r,e_r2), axis=1)
    #xyz_g = np.concatenate((xyz_g,e_g2), axis=1)
    
    print("xyz_r.shape",np.array(xyz_r).shape)    
    print("xyz_g.shape",np.array(xyz_g).shape)
    
    print("xyz_g[0][:5]",xyz_g[0][:5])
    
    # standardise all props
    xyz_r_std = StandardScaler().fit_transform(xyz_r)
    xyz_g_std = StandardScaler().fit_transform(xyz_g)
    
    #print("xyz_r_std",xyz_r_std)    
    #print("xyz_g_std",xyz_g_std)
    print("xyz_g_std[0][:5]",xyz_g_std[0][:5])
    
    pca = PCA(n_components=2)
    xyz_r_std_pca = pca.fit_transform(xyz_r_std)
    #pca = PCA(n_components=2)
    #xyz_g_std_pca = pca.fit_transform(xyz_g_std)
    xyz_g_std_pca = pca.transform(xyz_g_std)
        
    #print("xyz_r_std_pca",xyz_r_std_pca)    
    #print("xyz_g_std_pca",xyz_g_std_pca)
    
    print("xyz_r_std_pca.shape",xyz_r_std_pca.shape)    
    print("xyz_g_std_pca.shape",xyz_g_std_pca.shape)
    
    np.save("./thesis/plot_pca/data/{}_xyz_r.npy".format(names[0]), xyz_r_std_pca) 
    np.save("./thesis/plot_pca/data/{}_xyz_g.npy".format(names[0]), xyz_g_std_pca) 
    np.save("./thesis/plot_pca/data/{}_e_r.npy".format(names[0]), e_r) 
    np.save("./thesis/plot_pca/data/{}_e_g.npy".format(names[0]), e_g) 
                
label2 = 'Inverted'
label2 = 'Generated'
density = True

plot_flag = 1
if plot_flag:

    xyz_r = np.load("./thesis/plot_pca/data/{}_xyz_r.npy".format(names[0]))
    xyz_g = np.load("./thesis/plot_pca/data/{}_xyz_g.npy".format(names[0]))
    
    e_r = np.load("./thesis/plot_pca/data/{}_e_r.npy".format(names[0]))
    e_g = np.load("./thesis/plot_pca/data/{}_e_g.npy".format(names[0]))
    
    e_r = utils.kcal2ev(e_r, info['N_at'][0])
    e_g = utils.kcal2ev(e_g, info['N_at'][0])
        
    # Colors and Plot Labels
    data_labels = ['Real', label2]
    colors = ['C1', 'C0']
    
    _min = np.min(np.concatenate((e_r, e_g)))  
    _max = np.max(np.concatenate((e_r, e_g)))  
    
    # sort for legend colors
    inds_r = np.argsort(e_r)[::-1]
    inds_g = np.argsort(e_g)[::-1]
    e_r = e_r[inds_r]
    e_g = e_g[inds_g]
    xyz_r = xyz_r[inds_r]
    xyz_g = xyz_g[inds_g]
    
    fig, ax = plt.subplots()
    
    a = ax.scatter(xyz_r[:,0], xyz_r[:,1], c=e_r, marker='o', label=data_labels[0], cmap='Oranges', vmin=np.round(_min,2), vmax=np.round(_max,2))
    b = ax.scatter(xyz_g[:,0], xyz_g[:,1], c=e_g, marker='X', label=data_labels[1], cmap='Blues', vmin=np.round(_min,2), vmax=np.round(_max,2))
    #ax.scatter(xyz_r[:,0], xyz_r[:,1], color=colors[0], label=data_labels[0])
    #ax.scatter(xyz_g[:,0], xyz_g[:,1], color=colors[1], label=data_labels[1])
   
    #divider = make_axes_locatable(ax)
    #divider_kwargs = dict(position="right", size="5%", pad=0.2) 
    #fig.colorbar(a, cax=divider.append_axes(**divider_kwargs), label='DFT Energy [eV/atom]')
    #fig.colorbar(b, cax=divider.append_axes(**divider_kwargs), format=matplotlib.ticker.FuncFormatter(lambda x, pos: ''))
    
    cax1 = ax.inset_axes([1.12, 0.0, 0.06, 1.0])
    cax2 = ax.inset_axes([1.03, 0.0, 0.06, 1.0])
    cbar = fig.colorbar(a, cax=cax1)
    fig.colorbar(b, cax=cax2, format=matplotlib.ticker.FuncFormatter(lambda x, pos: ''))
    
    ax.set_xlabel('PCA1', fontsize=15)
    ax.set_ylabel('PCA2', fontsize=15)
    
    ax.tick_params(axis='both', which='major', labelsize=14)
    cbar.ax.tick_params(labelsize=13) 
    
    # Plot Settings
    #fig.supylabel('Count', fontsize=8)
    fig.text(1.0, 0.5, 'DFT Energy [eV/atom]', ha='center', va='center', rotation='vertical', fontsize=16)
    #axs[3].set_ylabel('Frequency [arb. unit]', fontsize=11)
    #axs[-1].set_xlabel('Interatomic Distance [\AA]', fontsize=13)
    #axs[4].legend(fontsize=15, frameon=False,)
    
    #axs[0].scatter(x_train, y_train, s=14, alpha=0.7, color="black", label="Data")
    #axs[0].plot(x_data, y_pred_under, color="red", alpha=0.8, label="Under Fit")
    ax.legend(loc="upper right", frameon=True, fontsize=14)
    #axs[0].set_xticks([])
    #axs[0].set_yticks([])

    fig.set_figheight(7)
    fig.set_figwidth(10)
    #fig.tight_layout(pad=3.7, h_pad=1.9, w_pad=1.5)  
        
    # tune plot
    #ax.set_aspect('equal')
    # ax.grid()
    plt.tight_layout()
    
    now = datetime.now()
    time_date = now.strftime('(%H%M_%d%m)')
    plt.savefig('./thesis/plot_pca/plots/{}_{}.pdf'.format(names[0],time_date), bbox_inches="tight")
    #plt.show()
    
    
    


