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

from cobe import Dataset, Datadict

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
parser.add_argument('-mols', type=int, default=[1])# '9_ethanol', '9_malonaldehyde', '12_benzene', '12_uracil', '15_toluene', '16_salicylic_acid', '18_naphthalene', '21_aspirin',
parser.add_argument('-m', type=int, default=0)# '9_ethanol', '9_malonaldehyde', '12_benzene', '12_uracil', '15_toluene', '16_salicylic_acid', '18_naphthalene', '21_aspirin',
args = parser.parse_args()

torch.set_default_tensor_type(torch.DoubleTensor)

#-----------------------------------------------------------------------------------------------------------------------
# - Set Paths
# - Load GS Dict
# - XYZ Data Path
#-----------------------------------------------------------------------------------------------------------------------     
   
# Data Paths Info
names = [naming.mol_list[i] for i in args.mols]

# Choose DataSets and Set Sizes and make Paths
data_paths = []
num_mols = []
for i_set in range(len(args.mols)):
    load_dir_path = naming.base_path + '/data/bispec/{}/'.format(names[i_set])

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
#print('\n-Combined Dataset Information')
#for i in info:
#    print('%s:' % i, info[i])
    
GS = {"mols": args.mols}
     
#-----------------------------------------------------------------------------------------------------------------------
# - Generate XYZ Coords using gen_xyz()
#-----------------------------------------------------------------------------------------------------------------------    

start_N=50000
start_M=5
target_mol_ind = 1008

trj_flag = 0
if trj_flag:

    set_inds = [0]
    pbc_flag=True
    loss_threshold=2e-12
    lr=5e-7
    alpha=1e-3
    eta=1e2
    patience = start_N

    # Find twojmax that corresponds to info['N_feats']
    twojmax_dict = {55:8, 30:6} # keys:info['N_feats'] and values:twojmax
    twojmax = twojmax_dict[info['N_feats']]

    # Generate B if Real B not supplied
    real_flag = 1
    if not real_flag:
        b, l, e = generate_like_train(1, set_inds, nn_G, info, GS, G_comp=G_comp, G_conv=G_conv)
        #b = scale.inv_fs(b, l, info, GS)
        b = [torch.tensor(info['std'].inverse_transform(b[0].reshape((-1,info['N_all_species']*info['N_feats'])))).reshape((-1,info['N_all_species'],info['N_feats']))]    
        #sys.exit()
    else:
        b = [Bs[set_ind] for set_ind in set_inds]
        l = [Ls[set_ind] for set_ind in set_inds]
        e = [Es[set_ind].unsqueeze(1) for set_ind in set_inds]
        b = scale.sum_over_species(b, l, info, GS)

    # For each dataset specified 
    for i_set, set_ind in enumerate(set_inds):
    
        # change shape from [N_set][N_mols,...] to [N_mols,...]
        _b, _l, _e = b[i_set].numpy(), l[i_set].numpy(), e[i_set].numpy() 
        
        # Set rc, rcutfac, and weights
        rc = [info['rc_dict'][species] for species in info['species'][set_ind]] 
        rcutfac = info['rcutfac']
        weights = [1.0 for _ in range(info['N_species'][set_ind])]
        rfac0=0.99

        # Starting Mols Paths
        start_mol_base_path = './invert/start/{}/'.format(names[set_ind])
        start_fls = os.listdir(start_mol_base_path)
        start_fls.sort(key=lambda x: float(x[:-4]))    
        start_mols = [Dataset(start_mol_base_path + s) for s in start_fls][:start_M]
        
        # For each sample to be Inverted 
        for i, i_mol in enumerate(range(target_mol_ind, target_mol_ind+1)):
            # select i-th sample to be inverted 
            target_b = _b[i_mol]
            target_l = _l[i_mol]
            target_e = _e[i_mol] # remove list - float          
            
            trjs_start = []
            losses_start = []

            # save all trjs_start for thesis plot 
            _path_i = './thesis/plot_energy_conv/data/{}/'.format(target_mol_ind)
            print("_path_i",_path_i)
            os.makedirs(_path_i, exist_ok=True)
            
            # For each Starting Mol
            print('\nSample ID: {}'.format(i_mol))
            print("\nChoosing {} from {} Start Mols".format(args.m, len(start_mols)))
            
            print("target_e",target_e)
            
            # Inversion
            trj, loss, conv_type = utils.invert_b2(copy.deepcopy(start_mols[args.m]),
                                                      target_b,
                                                      target_l,
                                                      target_e,
                                                      info=info,
                                                      rcutfac=rcutfac,
                                                      rc=rc,
                                                      rfac0=rfac0,
                                                      twojmax=twojmax,
                                                      weights=weights,
                                                      pbc_flag=pbc_flag,
                                                      pbar_disable=False,
                                                      loss_threshold=loss_threshold,
                                                      patience=patience,
                                                      #loss_threshold=1e-8,
                                                      max_N=start_N,
                                                      lr=lr,
                                                      alpha=alpha,
                                                      eta=eta,
                                                      )
                                                      

            print("target_e",target_e)
            path_i = _path_i + 'trj_{}.xyz'.format(args.m)
            for _j,_trj in enumerate(trj):
                if _j == 0:
                    _trj.info = {'Energy': target_e[0]} 
                    write(path_i, _trj)
                else:
                    write(path_i, _trj, append=True)  

            #print(loss)
            np.save(_path_i+'loss_{}.npy'.format(args.m), loss)
            if args.m != start_M-1:
                sys.exit()
                              
n_dft = 200
pyscf_flag = 0
if pyscf_flag:

    trj_path = "./thesis/plot_energy_conv/data/{}/".format(target_mol_ind)
    fls = os.listdir(trj_path)
    fls = [f for f in fls if f[0] != "l"]
    fls.sort(key=lambda x: float(x[x.index('_')+1:-4]))
    #fls.sort(key=lambda x: float(x[:-4]))    
    fls = fls[:start_M]
    
    #trjs = [Trajectory(trj_path + f) for f in fls] 
    #target_Es = [trj[0].info['Energy'] for trj in trjs]
    #print(target_Es)
    XYZs, Ls = [], []
    for f in fls:
        xyz, l = utils.read_xyz(-1, trj_path+f, rand_flag=0)
        XYZs.append(xyz), Ls.append(l)
        
    print(np.array(XYZs).shape)
    print(np.array(Ls).shape)
    
    # target E
    #"""
    target_Es = []
    for f in fls:
        with open(trj_path+f) as f: 
            #list of each line in big .xyz
            data = f.read()
            data = data.split('\n')
                
            start_index = data[1].index("Energy=-") + len("Energy=")
            end_index = start_index + data[1][start_index:].index(".") + 10
            e_mol = data[1][start_index:end_index]
            target_Es.append(float(e_mol))
    print(target_Es)
    if not all([target_Es[0] == e for e in target_Es]):
        print("Not all target_Es the same")
        sys.exit()
    #"""
        
    xyz_inds = [int(n) for n in np.linspace(0, len(XYZs[0])-1, n_dft)]
    print(xyz_inds)
    
    # loop over each trajectory
    for i, xyzs in enumerate(XYZs):
        _pyscf_Es = []
        # loop over every xyc in trj
        for j,xyz in enumerate(xyzs):
            if j in xyz_inds:
                _pyscf_Es.append(utils.dft(xyz, Ls[i][j]))
        np.save('./thesis/plot_energy_conv/data/pyscf_Es_{}.npy'.format(i), _pyscf_Es)
        
    np.save('./thesis/plot_energy_conv/data/target_E.npy', target_Es[0])
    np.save('./thesis/plot_energy_conv/data/xyz_inds.npy', xyz_inds)
        
plot_conv_flag = 1
if plot_conv_flag:

    pyscf_Es = []
    for i in range(start_M):
        pyscf_Es.append(np.load('./thesis/plot_energy_conv/data/pyscf_Es_{}.npy'.format(i)))
    pyscf_Es = np.array(pyscf_Es)   
    target_E = np.load('./thesis/plot_energy_conv/data/target_E.npy')
    xyz_inds = np.load('./thesis/plot_energy_conv/data/xyz_inds.npy')
    xyz_inds[0] += 1 # avoid log(0)
    
    pyscf_Es = utils.convert_kcal2ev(pyscf_Es, info['N_at'][0])
    target_E = utils.convert_kcal2ev(target_E, info['N_at'][0])
    
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9'] 
    
    # get best mol
    final_Es = np.array([Es[-1] for Es in pyscf_Es])
    delta_Es = np.abs(final_Es - target_E)
    print(target_E)
    print(final_Es)
    print(delta_Es)
    best_mol_ind = np.argmin(delta_Es)
    print(best_mol_ind)
    
    plt.figure(figsize=(10,8))
    
    plt.hlines(target_E, 0, xyz_inds[-1], linestyle=":", color='black', linewidth=4, alpha=1.0, label="Target Energy = {} [eV/atom]".format(np.round(target_E, 4)))
    
    for i in range(start_M):
        if i == best_mol_ind:
            plt.plot(xyz_inds, pyscf_Es[i], color=colors[i], linewidth=4, label= "Final Energy = {} [eV/atom]".format(np.round(final_Es[i], 4)))
        else:
            plt.plot(xyz_inds, pyscf_Es[i], color=colors[i], linewidth=2, alpha=0.7)
    
    #plt.xscale('log')
    #plt.xlim(0, xyz_inds[-1]*1.1)
    
    plt.ylabel("DFT Energy [eV/atom]", fontsize=19)
    plt.xlabel("Iteration", fontsize=19)
    plt.legend(frameon=False, loc='upper right', fontsize=16)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=16)
    
    time_date = datetime.now().strftime('%H%M_%d%m')
    plt.savefig('./thesis/plot_energy_conv/plots/energy_conv_'+str(time_date)+'.pdf', bbox_inches='tight')
    
    #plt.show()
    sys.exit()

    #"""
    

