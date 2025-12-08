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
import tqdm
import argparse
import numpy as np
from random import shuffle
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
import config

import G_b3 as G # basic linear G w/ comp

from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser()
parser.add_argument('-mols', type=str, default="123456789")# '9_ethanol', '9_malonaldehyde', '12_benzene', '12_uracil', '15_toluene', '16_salicylic_acid', '18_naphthalene', '21_aspirin',
parser.add_argument('-s', type=str, default="train")# '9_ethanol', '9_malonaldehyde', '12_benzene', '12_uracil', '15_toluene', '16_salicylic_acid', '18_naphthalene', '21_aspirin',
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
    load_dir_path = config.base_path + '/data/bispec/{}/'.format(names[i_set])

    # Print all directories in load_path
    dirs = [dr for dr in os.listdir(load_dir_path) if os.path.isdir(load_dir_path+dr)]
    #print('\n-------------------------------------------------')
    #print('Printing all Set Types for {}:'.format(names[i_set]))
    #for i, dr in enumerate(dirs):
    #    print(' {}) {}'.format(i, dr))

    # If theres only one dataset for chosen molecule then automatically pick that
    set_type = args.s

    # Print all directories in load_path
    dirs = [dr for dr in os.listdir(os.path.join(load_dir_path,set_type)) if os.path.isdir(os.path.join(load_dir_path,set_type,dr))]
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

    data_paths.append(os.path.join(load_dir_path, set_type, set_size))
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
    infos.append(np.load(os.path.join(data_paths[i], str(num_mols[i]) + '_info.npy'), allow_pickle=True).item())
info = data.combined_info(infos)
    
# Load B and L Data
Bs, Xs, Es, Ls = [], [], [], []
for i in range(len(args.mols)):
    #Bs.append(torch.tensor(np.load(data_paths[i] + str(num_mols[i]) + '_B.npy')))
    Xs.append(torch.tensor(np.load(os.path.join(data_paths[i], str(num_mols[i]) + '_X.npy'))))
    Es.append(torch.tensor(np.load(os.path.join(data_paths[i], str(num_mols[i]) + '_E.npy'))))
    Ls.append(torch.tensor(np.load(os.path.join(data_paths[i], str(num_mols[i]) + '_L.npy'))))

# Add B_min and B_max to Info for inv_fs (=2,4)
#bs2 = copy.deepcopy(Bs)    
#Bs2_sum = scale.sum_over_species(bs2, Ls, info)

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

s_G = ['real' for _ in names] 
n_mol = 100
conv_types = ['not_conv', 'not_conv', 'not_conv', 'not_conv', 'not_conv', 'not_conv', 'not_conv', 'not_conv', 'not_conv', 'not_conv' ]

data_flag = 0
if data_flag:

    for i in range(len(names)):
    
        # Get Inverted XYZ 
        temp = 0
        if not temp:
            invert_path = "./invert/{}/{}/".format(len(Xs[i][0]), s_G[i])  
            path = invert_path+names[i]+'/end/'+conv_types[i]
            print("path",path)
            xyzs, ls, target_es, pyscf_es, og_Es = utils.read_xyz_dir(n_mol, path, sorted_flag=1, energy_flag=1, dft_energy_flag=1, og_energy_flag=1)
            
        else:
            rand_inds = np.arange(0,len(Xs[i]))
            shuffle(rand_inds)
            rand_inds = rand_inds[:n_mol]
            xyzs, ls, target_es = Xs[i][rand_inds], Ls[i][rand_inds], Es[i][rand_inds]
            
            pyscf_es = target_es + torch.randn((len(target_es),))
            
        xyzs = np.array(xyzs)
        ls = np.array(ls)
        target_es = np.array(target_es)
        pyscf_es = np.array(pyscf_es)
            
        print("\n\nxyzs.shape",xyzs.shape)
        print("ls.shape",ls.shape)
        print("target_es.shape",target_es.shape)
        print("pyscf_es.shape",pyscf_es.shape)
        print("names[i]",names[i])
        print("og_Es",og_Es)
        
        pyscf_flag = 1
        og_Es2 = copy.deepcopy(og_Es)
        if pyscf_flag:
            for j in tqdm.tqdm(range(n_mol)):
                if og_Es[j] is None:
                    print(names[i])
                    print(og_Es[j])
                    og_Es2[j] = utils.dft(xyzs[j], ls[j])
            np.save('./thesis/plot_all_parity/data/{}_target_es2.npy'.format(names[i]), og_Es2)

        np.save('./thesis/plot_all_parity/data/{}_target_es.npy'.format(names[i]), target_es)
        np.save('./thesis/plot_all_parity/data/{}_pyscf_es.npy'.format(names[i]), pyscf_es)
     
plot_flag = 1
if plot_flag:

    target_Es, pyscf_Es = [], []
    
    for i,name in enumerate(names):
        _p = np.load('./thesis/plot_all_parity/data/{}_pyscf_es.npy'.format(name))
        _t = np.load('./thesis/plot_all_parity/data/{}_target_es.npy'.format(name))
        _t = np.load('./thesis/plot_all_parity/data/{}_target_es2.npy'.format(name))
        
        _p = utils.kcal2ev(_p, info['N_at'][i])
        _t = utils.kcal2ev(_t, info['N_at'][i])
        
        print(_p[:4])        
        print(_t[:4])
        
        pyscf_Es.append(_p)
        target_Es.append(_t)
        
        #if i in [0,1,2]:
        #    target_Es.append(np.load('./thesis/plot_all_parity/data/{}_target_es2.npy'.format(name)))
        #else:
        #    target_Es.append(np.load('./thesis/plot_all_parity/data/{}_target_es.npy'.format(name)))
        
    # Create Figure
    fig, axs = plt.subplots(nrows=3, ncols=3)
    
    axs = axs.flatten()
    
    ticks = [[-806.05, -806.10, -806.15], [-525.52, -525.54, -525.56], [-938.84, -938.88, -938.92, ], [-491.53, -491.56, -491.59], [-842.00, -842.03, -842.06], 
             [-582.12, -582.14, -582.16], [-699.92, -699.95, -699.98], [-838.91, -838.94, -838.97], [-648.04, -648.06, -648.08]]
    #minors = [0.01, 0.004, 0.008, 0.006, 0.006, 0.004, 0.006, 0.006, 0.004]

    #for ax, name, target_es, pyscf_es in zip(axs, names2, target_Es, pyscf_Es):
    for i in range(len(names)):
        target_es, pyscf_es = target_Es[i], pyscf_Es[i]
        
        # calc metrics
        r = 5
        mae = np.round(np.mean(np.abs(np.subtract(pyscf_es, target_es))), r)
        mae = "%.2e" % mae
        rms = np.round(np.sqrt(np.mean((np.subtract(pyscf_es, target_es))**2)), r)
        _max = np.round(max(np.subtract(pyscf_es, target_es)), r)

        # Square plot    
        axs[i].set_aspect("equal")
        min_max = [min(np.concatenate((pyscf_es, target_es))), max(np.concatenate((pyscf_es, target_es)))]
        offset = (min_max[1] - min_max[0]) * 0.1
        axs[i].set_ylim(min_max[0]-offset, min_max[1]+offset)
        axs[i].set_xlim(min_max[0]-offset, min_max[1]+offset)

        # Draw the scatter plot and marginals.
        y_min_max = [min(pyscf_es), max(pyscf_es)]
        xy1, xy2 = [y_min_max[0], y_min_max[0]], [y_min_max[1], y_min_max[1]]
        axs[i].axline(xy1, xy2, linestyle=":", c='black', alpha=0.8)
        axs[i].scatter(target_es, pyscf_es, c='C0', alpha=0.5)
    
        # Text box
        props = dict(boxstyle='square,pad=-0.1', facecolor='white', alpha=0.0, edgecolor='none')
        axs[i].text(0.035, 0.975, r"\underline{%s}" % names2[i], transform=axs[i].transAxes, fontsize=16, verticalalignment='top', bbox=props)
        axs[i].text(0.035, 0.95, '\nMAE={} [eV/atom]'.format(mae), transform=axs[i].transAxes, fontsize=14, verticalalignment='top', bbox=props)
        
        # ticks
        from matplotlib.ticker import AutoMinorLocator, MultipleLocator
        #n = 3
        #axs[i].xaxis.set_major_locator(plt.MaxNLocator(n))
        #axs[i].yaxis.set_major_locator(plt.MaxNLocator(n))
        #axs[i].xaxis.set_minor_locator(plt.MaxNLocator(3*5))
        #axs[i].yaxis.set_minor_locator(plt.MaxNLocator(3*5))
        #axs[i].xaxis.set_major_locator(MultipleLocator(0.02))
        #axs[i].yaxis.set_major_locator(MultipleLocator(0.02))
        axs[i].set_xticks(ticks[i])        
        axs[i].set_yticks(ticks[i])
        #axs[i].xaxis.set_minor_locator(MultipleLocator(minors[i]))
        #axs[i].yaxis.set_minor_locator(MultipleLocator(minors[i]))
        axs[i].xaxis.set_minor_locator(AutoMinorLocator(5))
        axs[i].yaxis.set_minor_locator(AutoMinorLocator(5))
                
        axs[i].tick_params(axis='both', which='major', labelsize=16)
        axs[i].ticklabel_format(useOffset=False)
        
        # grid
        axs[i].grid(which='major', alpha=0.5, linewidth=1.1)
        axs[i].grid(which='minor', alpha=0.2)

    #axs[i].set_ylabel("Inverted DFT Energy [eV/atom] ", fontsize=15)
    #axs[i].set_xlabel('Original DFT Energy [eV/atom]', fontsize=15)
    axs[3].set_ylabel('Inverted DFT Energy [eV/atom]', fontsize=19)
    axs[7].set_xlabel('Original DFT Energy [eV/atom]', fontsize=19)
    #fig.text(0.075, 0.493, 'Inverted DFT Energy [eV/atom]', ha='center', va='center', rotation='vertical', fontsize=17)
    #fig.text(0.515, 0.06, 'Original DFT Energy [eV/atom]', ha='center', va='center', rotation='horizontal', fontsize=17)
        
    fig.set_figheight(12)
    fig.set_figwidth(12)
    fig.tight_layout(pad=1.0, h_pad=-8.5, w_pad=-1.5)      
    
    # Save Plot
    time_date = datetime.now().strftime('%H%M_%d%m')
    plt.savefig('./thesis/plot_all_parity/plots/all_parity_{}.pdf'.format(time_date), bbox_inches='tight')
    
    sys.exit()
    
    
    
    
    
    
    
