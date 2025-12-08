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
import matplotlib
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

now = datetime.now()
time_date = now.strftime('(%H%M_%d%m)')

bs = np.load("./thesis/plot_bispec/data/bispec3.npy")

print(bs)
print(bs.shape)

plt.figure()
plt.axis('off')
plt.imshow(bs)
plt.savefig("./thesis/plot_bispec/plots/b.png", dpi=500)
plt.close

pad = 3
bs2 = np.empty((len(bs)*pad, len(bs[0]))) * np.nan

print("bs.shape",bs.shape)
print("bs2.shape",bs2.shape)
print(np.array([j for j in range(len(bs))])*pad)

for i,i2 in enumerate(np.array([j for j in range(len(bs))])*pad):
    print(i,i2)
    print(bs[i])
    print(bs2[i2])
    bs2[i2,:] = bs[i,:]
    
print(bs2[:,0])
print(bs2[0])
print(bs2[1])
print(bs2[2])
print(bs2[3])
    
current_cmap = matplotlib.cm.get_cmap()
current_cmap = current_cmap.set_bad(color='white')

plt.figure()
plt.axis('off')
plt.imshow(bs2, cmap=current_cmap)
plt.savefig("./thesis/plot_bispec/plots/b2.png", dpi=500)


for i,b in enumerate(bs):
    plt.figure()
    plt.axis('off')
    plt.imshow(bs[i].reshape(1,len(bs[0])))
    plt.savefig("./thesis/plot_bispec/plots/b_{}.png".format(i), dpi=500)

plt.figure()
plt.xticks([])
plt.yticks([])
plt.plot(range(len(bs[0])), bs[0], color="blue", linewidth=2)
plt.savefig("./thesis/plot_bispec/plots/b_C.png", dpi=500)


plt.figure()
#plt.axis('off')
plt.xticks([])
plt.yticks([])
plt.plot(range(len(bs[0])), bs[7], color="red", linewidth=2)
plt.savefig("./thesis/plot_bispec/plots/b_H.png", dpi=500)















