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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
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
import scale as scale
import naming

import G_b3 as G # basic linear G w/ comp

from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser()
parser.add_argument('-n', type=str, default='1') # '9_ethanol', '9_malonaldehyde', '12_benzene', '12_uracil', '15_toluene', '16_salicylic_acid', '18_naphthalene', '21_aspirin','20_para','24_azob'
args = parser.parse_args()

torch.set_default_tensor_type(torch.DoubleTensor)

#-----------------------------------------------------------------------------------------------------------------------
# - Set Paths
# - Load GS Dict
# - XYZ Data Path
#-----------------------------------------------------------------------------------------------------------------------     
   
# Data Paths Info
names = [naming.mol_list[int(i)] for i in args.n]

# Choose DataSets and Set Sizes and make Paths
data_paths = []
num_mols = []
for i_set in range(len(args.n)):
    load_dir_path = naming.base_path + 'data/bispec/{}/'.format(names[i_set])

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
    print('\n-------------------------------------------------')
    print('\nPrinting all Set Sizes for {}:'.format(names[i_set]))
    for i, dr in enumerate(dirs):
        print(' {}) {}'.format(i, dr))

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
for i in range(len(args.n)):
    print('    {}) {}'.format(i, data_paths[i]))

#-----------------------------------------------------------------------------------------------------------------------
# - Load B Data
# - Combine Info Dicts
# - Load Model
#-----------------------------------------------------------------------------------------------------------------------  
   
# Load Info - seperate to load data because args.e
infos = []
for i in range(len(args.n)):
    infos.append(np.load(data_paths[i] + str(num_mols[i]) + '_info.npy', allow_pickle=True).item())
info = data.combined_info(infos)
    
# Load B and L Data
Bs, Xs, Es, Ls = [], [], [], []
for i in range(len(args.n)):
    Bs.append(torch.tensor(np.load(data_paths[i] + str(num_mols[i]) + '_B.npy')))
    Xs.append(torch.tensor(np.load(data_paths[i] + str(num_mols[i]) + '_X.npy')))
    if info['energy']:
        Es.append(torch.tensor(np.load(data_paths[i] + str(num_mols[i]) + '_E.npy')))
    Ls.append(torch.tensor(np.load(data_paths[i] + str(num_mols[i]) + '_L.npy')))

# Add B_min and B_max to Info for inv_fs (=2,4)
#bs2 = copy.deepcopy(Bs)    
#Bs2_sum = scale.sum_over_species(bs2, Ls, info)

#std = StandardScaler()
#Bs2_sum = [torch.tensor(std.fit_transform(Bs2_sum[0].reshape((-1, info['N_all_species']*info['N_feats'])))).reshape((-1, info['N_all_species'], info['N_feats']))]
#info["std"] = std

# Print Combined Dataset Info
#print('\n-Combined Dataset Information')
#for i in info:
    #print('%s:' % i, info[i])
    
GS = {"mols": [int(i) for i in args.n]}
     
#-----------------------------------------------------------------------------------------------------------------------
# - Generate XYZ Coords using gen_xyz()
#-----------------------------------------------------------------------------------------------------------------------    

data_flag = 0
if data_flag:
    # Inversion Parameters
    t0 = time.time()
    num_mols = 1
    set_inds = [0]

    np.random.seed(421)
    torch.manual_seed(421)
    _, losses, _, losses_start, _ = utils.gen_xyz_og(num_mols, set_inds, None, info, GS,
                                                                        b_real=Bs, l_real=Ls, e_real=Es, # uncomment this line to invert gen B
                                                                        get_trj=False, save_trj=False, save_end=False, pbc_flag=True,
                                                                        loss_threshold=2e-10, max_N=100000, start_N=500, start_M=10,
                                                                        lr=3e-7, alpha=1e-3, eta=1e2, #ethanol lr=6e-7
                                                                        rand_target_ind=True,
                                                                        #lr=9.9999e-7, alpha=1e-3, eta=1e2, 
                                                                        )
                                                                        
    np.save('./thesis/plot_loss/data/losses.npy', losses)
    np.save('./thesis/plot_loss/data/losses_start.npy', losses_start)

    print("\nTime:",time.time() - t0)
    
plot_flag = 1
if plot_flag:
    
    losses = np.load('./thesis/plot_loss/data/losses.npy')
    losses_start = np.load('./thesis/plot_loss/data/losses_start.npy')
    
    print("losses.shape",losses.shape)
    print("losses_start.shape",losses_start.shape)

    
    # get best mol
    final_losses = np.array([l[-1] for l in losses_start[0]])
    print(final_losses)
    best_start_ind = np.argmin(final_losses)
    print(best_start_ind)

    plt.figure(figsize=(10,8))
    
    #plt.plot(np.arange(len(losses_start[0][0]), len(losses[0][0])), losses[0][0][len(losses_start[0][0]):], color=colors[1])
    plt.plot(range(len(losses[0][0])), losses[0][0], color='C1')
    for i,loss in enumerate(losses_start[0]):
        if i != best_start_ind:
            plt.plot(np.arange(len(loss)), loss, color='C0')

    
    plt.ylim(0.07,12000) 
    plt.xlim(0.6,110000)     

    plt.vlines(500, 0.01, 12000, linestyle=":", color="C0")
    plt.text(550, 7000, "Initial Threshold", color="C0", fontsize=14)
    plt.text(740, 5000, r"$N_{init}$=500", color="C0", fontsize=14)
    
    plt.hlines(losses[0][0][-1], 0.6, 110000, linestyle=":", color="C1")
    plt.text(1400, 0.185, "Convergence Threshold", color="C1", fontsize=14)
    plt.text(2500, 0.13, r"$N_{iter}$=100,000", color="C1", fontsize=14)

    plt.yscale('log')
    plt.xscale('log')
    
    plt.ylabel("Loss", fontsize=17)
    plt.xlabel("Iteration", fontsize=17)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    #plt.legend(frameon=False, loc='upper right', fontsize=12)
    
    time_date = datetime.now().strftime('%H%M_%d%m')
    plt.savefig('./thesis/plot_loss/plots/loss_'+str(time_date)+'.pdf', bbox_inches='tight')

    sys.exit()



