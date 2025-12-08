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
import config

import G_b3 as G # basic linear G w/ comp

from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser()
parser.add_argument('-mols', type=int, default=[0]) # '9_ethanol', '9_malonaldehyde', '12_benzene', '12_uracil', '15_toluene', '16_salicylic_acid', '18_naphthalene', '21_aspirin',
args = parser.parse_args()

torch.set_default_tensor_type(torch.DoubleTensor)

#-----------------------------------------------------------------------------------------------------------------------
# - Set Paths
# - Load GS Dict
# - XYZ Data Path
#-----------------------------------------------------------------------------------------------------------------------     
   
# Data Paths Info
names = [config.mol_list[i] for i in args.mols]

# Choose DataSets and Set Sizes and make Paths
data_paths = []
num_mols = []
for i_set in range(len(args.mols)):
    load_dir_path = config.base_path + '/data/bispec/{}/'.format(names[i_set])

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
    Bs.append(torch.tensor(np.load(data_paths[i] + str(num_mols[i]) + '_B.npy')))
    Xs.append(torch.tensor(np.load(data_paths[i] + str(num_mols[i]) + '_X.npy')))
    Es.append(torch.tensor(np.load(data_paths[i] + str(num_mols[i]) + '_E.npy')))
    Ls.append(torch.tensor(np.load(data_paths[i] + str(num_mols[i]) + '_L.npy')))

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

s_G = 'real'
#s_G = '0022_0110' # ethanol
#s_G = '2247_3110' # ethanol
#s_G = '1703_0111' # ethanol
#s_G = '1720_0210' # mal
#s_G = '0022_0110' # benzene
#s_G = '2351_2910' # benzene
#s_G = '2352_2910' # benzene
#s_G = '2200_3110' # benzene
#s_G = '1653_0210' # uracil
#s_G = '0013_0510' # toluene
#s_G = '0014_0510' # sal acid
#s_G = '0933_0810' # naph
#s_G = '0933_0810' # par
#s_G = '1247_1211' # asp new clip
#s_G = '1246_1211' # asp og gp
#s_G = '1130_0711' # azo og gp is best
#s_G = '1129_0711' # azo new clip
conv_type = "conv"
#conv_type = "not_conv"
#conv_type = "not_conv2"
#conv_type = "not_conv2"
n_mol = 200

data_flag = 1
if data_flag:

    # Get Inverted XYZ 
    invert_path = "./invert/{}/{}/".format(len(Xs[0][0]), s_G)  
    print('invert_path+names[0]',invert_path+names[0])
    path = invert_path+names[0]+'/end/'+conv_type
    #xyzs, ls, target_Es, pyscf_Es = utils.read_xyz_dir(n_mol, path, sorted_flag=1, energy_flag=1, dft_energy_flag=1)
    xyzs, ls, target_Es, pyscf_Es, og_Es = utils.read_xyz_dir(n_mol, path, sorted_flag=1, energy_flag=1, dft_energy_flag=1, og_energy_flag=1)
    #xyzs, ls, target_Es, pyscf_Es = utils.read_xyz_dir(n_mol, path, sorted_flag=1, energy_flag=1, dft_energy_flag=1, og_energy_flag=0)
    #xyzs, ls, target_Es, pyscf_Es = utils.read_xyz_dir(n_mol, path, sorted_flag=0, energy_flag=1, dft_energy_flag=1, og_energy_flag=0)
    #xyzs, ls, target_Es, pyscf_Es, nn_Es = utils.read_xyz_dir(n_mol, path, sorted_flag=1, energy_flag=1, dft_energy_flag=1, nn_energy_flag=1)
    #target_Es = og_Es
    #print(nn_Es)
    #target_Es = nn_Es
    #pyscf_Es = nn_Es
    #target_Es = Es[0]
    #pyscf_Es = Es[0]

    print(np.array(target_Es).shape)
    print(np.array(pyscf_Es).shape)
    print(np.array(og_Es).shape)
    
    print(np.array(target_Es)[:10])
    print(np.array(pyscf_Es)[:10])
    print(np.array(og_Es)[:10])
    
    pyscf_flag = 0
    target_Es2 = []
    if pyscf_flag:
        for i in range(n_mol):
            target_Es2.append(utils.dft(xyzs[i], ls[i]))
        
        np.save('./thesis/plot_parity/data/{}_target_Es2.npy'.format(names[0]), target_Es2)
            
    np.save('./thesis/plot_parity/data/{}_target_Es.npy'.format(names[0]), og_Es)
    #np.save('./thesis/plot_parity/data/{}_target_Es.npy'.format(names[0]), target_Es)
    np.save('./thesis/plot_parity/data/{}_pyscf_Es.npy'.format(names[0]), pyscf_Es)
        
plot_flag = 1
if plot_flag:

    target_Es = np.load('./thesis/plot_parity/data/{}_target_Es.npy'.format(names[0]), allow_pickle=True)# *1000
    #target_Es = np.load('./thesis/plot_parity/data/{}_target_Es2.npy'.format(names[0]), allow_pickle=True)
    pyscf_Es = np.load('./thesis/plot_parity/data/{}_pyscf_Es.npy'.format(names[0]), allow_pickle=True)# *1000
    
    print(np.array(target_Es)[:10])
    print(np.array(pyscf_Es)[:10])
    target_Es = utils.kcal2ev(target_Es, info['N_at'][0])
    pyscf_Es = utils.kcal2ev(pyscf_Es, info['N_at'][0])
    
    print(np.array(target_Es)[:10])
    print(np.array(pyscf_Es)[:10])
    print(np.array(target_Es)[:10]-np.array(pyscf_Es)[:10])
     
    # calc metrics
    r = 8
    mae = np.round(np.mean(np.abs(np.subtract(pyscf_Es, target_Es))), r)
    rms = np.round(np.sqrt(np.mean((np.subtract(pyscf_Es, target_Es))**2)), r)
    _max = np.round(max(np.subtract(pyscf_Es, target_Es)), r)
    
    # Create Figure
    fig = plt.figure(figsize=(8, 10))
    gs = fig.add_gridspec(3, 2, width_ratios=(4, 1), height_ratios=(1, 4, 1),
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.0, hspace=0.0)
                          
    # Create the Axes.
    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
    ax_delta = fig.add_subplot(gs[2, 0], sharex=ax)
    ax_delta_hist = fig.add_subplot(gs[2, 1], sharey=ax_delta)

    # Square plot    
    #ax.set_aspect("equal")
    min_max = [min(np.concatenate((pyscf_Es, target_Es))), max(np.concatenate((pyscf_Es, target_Es)))]
    offset = (min_max[1] - min_max[0]) * 0.1
    ax.set_ylim(min_max[0]-offset, min_max[1]+offset)
    ax.set_xlim(min_max[0]-offset, min_max[1]+offset)
    
    # Draw the scatter plot and marginals.
    y_min_max = [min(pyscf_Es), max(pyscf_Es)]
    xy1, xy2 = [y_min_max[0], y_min_max[0]], [y_min_max[1], y_min_max[1]]
    ax.axline(xy1, xy2, linestyle=":", c='black', alpha=0.8)
    ax.scatter(target_Es, pyscf_Es, c='C0', alpha=0.5)

    ax_histx.hist(target_Es, bins=40, histtype="step", color="C0")
    ax_histy.hist(pyscf_Es, orientation="horizontal", bins=40, histtype="step", color="C0")
    
    #delta_fac = 1e5
    #delta_fac_str = r'\times{10}^{-%s}' % str(int(np.log10(delta_fac))) 
    delta_fac = 1
    delta_fac_str = '' 
    delta = np.subtract(pyscf_Es, target_Es)*delta_fac
    ax_delta.scatter(target_Es, delta, c='C0', alpha=0.5)
    ax_delta.axline([y_min_max[0], 0], [y_min_max[1], 0], linestyle=":", c='C0')

    ax_delta_hist.hist(delta, bins=15, orientation="horizontal", histtype="step", color="C0")
    ax_delta_hist.axline([0, 0], [ax_delta_hist.get_xlim()[1], 0], linestyle=":", c='black', alpha=0.7)
    
    # Text box
    props = dict(boxstyle='square', facecolor='white', alpha=0.5, edgecolor='none')
    textstr = 'MAE = {} [eV/atom]\nRMS = {} [eV/atom]\nMAX = {} [eV/atom]'.format(mae, rms, _max)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=13, verticalalignment='top', bbox=props)
    
    # Remove the tick labels
    ax.tick_params(axis="x", labelbottom=False)
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)
    ax_histy.tick_params(axis="x", labelbottom=False)
    ax_delta_hist.tick_params(axis="y", labelleft=False)
    ax.spines[['top', 'bottom']].set_visible(False)
    
    # Tick label sizes
    s = 11
    ax.tick_params(axis='both', labelsize=s)
    ax_histx.tick_params(axis='both', labelsize=s)
    ax_histy.tick_params(axis='both', labelsize=s)
    ax_histy.tick_params(axis='both', labelsize=s)
    ax_delta.tick_params(axis='both', labelsize=s)
    ax_delta_hist.tick_params(axis='both', labelsize=s)
        
    # minor ticks
    from matplotlib.ticker import MultipleLocator
    #ax.xaxis.set_minor_locator(MultipleLocator(0.005))
    #ax.yaxis.set_minor_locator(MultipleLocator(0.005))
    #ax_histy.yaxis.set_minor_locator(MultipleLocator(0.005))
    #ax_histx.xaxis.set_minor_locator(MultipleLocator(0.005))

    # grids
    alpha1 = 0.5
    alpha2 = 0.2
    ax.grid(which='major', alpha=alpha1, linewidth=1.1)
    ax.grid(which='minor', alpha=alpha2)
    ax_delta.grid(which='major', axis='x', alpha=alpha1, linewidth=1.1)
    ax_delta.grid(which='minor', axis='x', alpha=alpha2)
    ax_histx.grid(which='major', axis='x', alpha=alpha1, linewidth=1.1)
    ax_histx.grid(which='minor', axis='x', alpha=alpha2)
    ax_histy.grid(which='major', axis='y', alpha=alpha1, linewidth=1.1)
    ax_histy.grid(which='minor', axis='y', alpha=alpha2)
        
    # Axis Labels
    #ax.set_ylabel("Inverted DFT Energy [eV/atom] ", fontsize=15)
    #ax_delta.set_ylabel("Residuals [eV/atom]", fontsize=13)
    #ax_histx.set_ylabel("Freq [arb. units]", fontsize=13)
    #ax_delta.set_yticks([r"5{\times}10^{-5}","0","5{\times}10^{5}"]) # ONLY FOR ETHANOL
    fig.text(0.002, 0.5, 'Inverted DFT Energy [eV/atom]', ha='center', va='center', rotation='vertical', fontsize=15)
    fig.text(0.002, 0.830, 'Freq [arb. units]', ha='center', va='center', rotation='vertical', fontsize=13)
    
    #ax_delta.yaxis.set_major_formatter(r"%d\times10^{-5}")
    #ax_delta.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:%d}"))
    #fig.text(0.01, 0.165, 'Residuals [$%s$ eV/atom]' % delta_fac_str, ha='center', va='center', rotation='vertical', fontsize=13)
    
    ax_delta.yaxis.set_major_locator(MultipleLocator(5e-5))
    ax_delta.set_yticks([-5e-5, 0.0, 5e-5], labels=[r'$-5{\times}{10}^{-5}$', r'$0$', r'$5{\times}{10}^{-5}$'])
    fig.text(0.002, 0.165, 'Residuals [eV/atom]', ha='center', va='center', rotation='vertical', fontsize=13)
    
    ax_delta.set_xlabel("Original DFT Energy [eV/atom]", fontsize=15)
    ax_delta_hist.set_xlabel("Freq [arb. units]", fontsize=13)
    
    # Save Plot
    time_date = datetime.now().strftime('%H%M_%d%m')
    plt.savefig('./thesis/plot_parity/plots/parity_{}_{}.pdf'.format(names[0],time_date), bbox_inches='tight')
    
    sys.exit()
    
    
    
