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
from matplotlib.gridspec import GridSpec
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
parser.add_argument('-mols', type=str, default="0123456789")# '9_ethanol', '9_malonaldehyde', '12_benzene', '12_uracil', '15_toluene', '16_salicylic_acid', '18_naphthalene', '21_aspirin',
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
s_G = ['0022_0110', '1720_0210','0022_0110','1653_0210','0013_0510','0014_0510','0933_0810','0933_0810','1246_1211','1130_0711']
#s_G = ['0022_0110', '1720_0210','0022_0110','1653_0210','0013_0510','0014_0510','0043_1511','0933_0810','1246_1211','1130_0711']
n_mol = 200
conv_types = ['not_conv', 'not_conv', 'not_conv', 'not_conv', 'not_conv', 'not_conv', 'not_conv', 'not_conv', 'not_conv', 'not_conv' ]
conv_types = ['not_conv', 'not_conv', 'not_conv', 'not_conv', 'not_conv', 'not_conv', 'not_conv', 'not_conv2', 'not_conv2', 'not_conv' ]

data_flag = 1
if data_flag:

    for i in range(len(names)):
    
        # Get Inverted XYZ 
        temp = 0
        if not temp:
            dir_name = utils.get_dir_name([names[i]])
            #print(dir_name)
            #if i == 6:
            #    dir_name = "12_15_18"
            invert_path = "./invert/{}/{}/".format(dir_name, s_G[i])  
            path = invert_path+names[i]+'/end/'+conv_types[i]
            print("path",path)
            #xyzs, ls, target_es, pyscf_es, og_Es = utils.read_xyz_dir(n_mol, path, sorted_flag=1, energy_flag=1, dft_energy_flag=1, og_energy_flag=1)
            xyzs, ls, target_es, pyscf_es, nn_Es = utils.read_xyz_dir(n_mol, path, sorted_flag=1, energy_flag=1, dft_energy_flag=1, nn_energy_flag=1)
            
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
        print("nn_Es",nn_Es)
        
        pyscf_flag = 0
        nn_Es2 = copy.deepcopy(nn_Es)
        if pyscf_flag:
            for j in tqdm.tqdm(range(n_mol)):
                if nn_Es[j] is None:
                    print(names[i])
                    print(nn_Es[j])
                    nn_Es2[j] = utils.dft(xyzs[j], ls[j])
            np.save('./thesis/plot_ten_parity/data/{}_target_es2.npy'.format(names[i]), nn_Es2)

        np.save('./thesis/plot_ten_parity/data/{}_target_es.npy'.format(names[i]), target_es)
        np.save('./thesis/plot_ten_parity/data/{}_pyscf_es.npy'.format(names[i]), pyscf_es)
     
plot_flag = 1
if plot_flag:

    target_Es, pyscf_Es = [], []
    
    for i,name in enumerate(names):
        _p = np.load('./thesis/plot_ten_parity/data/{}_pyscf_es.npy'.format(name))
        _t = np.load('./thesis/plot_ten_parity/data/{}_target_es.npy'.format(name))
        #_t = np.load('./thesis/plot_ten_parity/data/{}_target_es2.npy'.format(name))
        
        _p = utils.kcal2ev(_p, info['N_at'][i])
        _t = utils.kcal2ev(_t, info['N_at'][i])
        
        print(_p[:4])        
        print(_t[:4])
        
        pyscf_Es.append(_p)
        target_Es.append(_t)
        
        #if i in [0,1,2]:
        #    target_Es.append(np.load('./thesis/plot_ten_parity/data/{}_target_es2.npy'.format(name)))
        #else:
        #    target_Es.append(np.load('./thesis/plot_ten_parity/data/{}_target_es.npy'.format(name)))
        
    # Create Figure
    fig = plt.figure()

    gs = GridSpec(9, 13, figure=fig)
    ax1 = fig.add_subplot(gs[1:3, 1:5])
    ax2 = fig.add_subplot(gs[1:3, 5:9])
    ax3 = fig.add_subplot(gs[1:3, 9:])
    ax4 = fig.add_subplot(gs[3:5, 1:5])
    ax5 = fig.add_subplot(gs[3:5, 5:9])
    ax6 = fig.add_subplot(gs[3:5, 9:])
    ax7 = fig.add_subplot(gs[5:7, 2:6])
    ax8 = fig.add_subplot(gs[5:7, 8:12])
    ax9 = fig.add_subplot(gs[7:9, 2:6])
    ax10 = fig.add_subplot(gs[7:9, 8:12])
    axs = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10]
    
    ticks = [[-467.79, -467.74, -467.69],[-806.05, -806.10, -806.15], [-525.52, -525.54, -525.56], [-938.84, -938.88, -938.92, ], [-491.53, -491.56, -491.59], [-842.00, -842.03, -842.06], 
             [-582.10, -582.14, -582.18], [-699.88, -699.93, -699.98], [-838.87, -838.93, -838.99], [-648.03, -648.06, -648.09]]
    ticks = [[-467.79, -467.72, -467.65],[-805.95, -806.05, -806.15], [-525.48, -525.52, -525.56], [-938.80, -938.86, -938.92], [-491.54, -491.57, -491.60], [-841.99, -842.03, -842.06], 
             [-582.10, -582.14, -582.18], [-699.86, -699.92, -699.98], [-838.83, -838.91, -838.99], [-648.00, -648.05, -648.10]]
    #minors = [0.01, 0.004, 0.008, 0.006, 0.006, 0.004, 0.006, 0.006, 0.004]

    #for ax, name, target_es, pyscf_es in zip(axs, names2, target_Es, pyscf_Es):
    for i in range(len(names)):
        target_es, pyscf_es = target_Es[i], pyscf_Es[i]
        
        def get_r2(y, y_hat):
            y_bar = y.mean()
            ss_tot = ((y-y_bar)**2).sum()
            ss_res = ((y-y_hat)**2).sum()
            return 1 - (ss_res/ss_tot)
            
        r2 = get_r2(target_es, pyscf_es)
        print("name, r2", names[i], r2)
        
        # calc metrics
        r = 4
        mae = np.round(np.mean(np.abs(np.subtract(pyscf_es, target_es))), r)
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
        axs[i].text(0.05, 0.95, r"\underline{%s}" % names2[i], transform=axs[i].transAxes, fontsize=14, verticalalignment='top', bbox=props)
        axs[i].text(0.05, 0.925, '\nMAE={} [eV/atom]'.format(mae), transform=axs[i].transAxes, fontsize=13, verticalalignment='top', bbox=props)
        #axs[i].text(0.05, 0.90, r'\text{R}^2=%s' % str(np.round(r2,2)), transform=axs[i].transAxes, fontsize=13, verticalalignment='top', bbox=props)
        axs[i].text(0.05, 0.8, r"$R^2$={%s}" % str(np.round(r2,3)), transform=axs[i].transAxes, fontsize=13, verticalalignment='top', bbox=props)
        
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
                
        axs[i].tick_params(axis='both', which='major', labelsize=15)
        axs[i].ticklabel_format(useOffset=False)
        
        # grid
        axs[i].grid(which='major', alpha=0.5, linewidth=1.1)
        axs[i].grid(which='minor', alpha=0.2)

    #axs[i].set_ylabel("Inverted DFT Energy [eV/atom] ", fontsize=15)
    #axs[i].set_xlabel('Original DFT Energy [eV/atom]', fontsize=15)
    #axs[3].set_ylabel('Inverted DFT Energy [eV/atom]', fontsize=17)
    #axs[7].set_xlabel('Original DFT Energy [eV/atom]', fontsize=17)
    fig.text(0.005, 0.493, 'Generated DFT Energy [eV/atom]', ha='center', va='center', rotation='vertical', fontsize=19)
    fig.text(0.515, 0.00, 'Desired DFT Energy [eV/atom]', ha='center', va='center', rotation='horizontal', fontsize=19)
    #fig.text(0.001, 0.493, 'Inverted DFT Energy [eV/atom]', ha='center', va='center', rotation='vertical', fontsize=17)
    #fig.text(0.515, 0.01, 'Original DFT Energy [eV/atom]', ha='center', va='center', rotation='horizontal', fontsize=17)
        
    fig.set_figheight(17)
    fig.set_figwidth(13)
    fig.tight_layout(pad=1.0, h_pad=1.0, w_pad=1.0) 
    #fig.tight_layout()         
    
    # Save Plot
    time_date = datetime.now().strftime('%H%M_%d%m')
    plt.savefig('./thesis/plot_ten_parity/plots/all_ten_parity_{}.pdf'.format(time_date), bbox_inches='tight')
    
    print("finished")
    sys.exit()
    
    
    
    
    
    
    
