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

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

data_flag = 0
if data_flag:

    tmp = 0
    if tmp:
        losses_start = []
        for i in [1,2,3,5]:
            losses_start.append(np.linspace(i*1e3, i*8e-1, 500))
        loss_final = np.linspace(4e3, 1e-1, 50000)
    else:
        losses_start = np.load('./thesis/plot_loss/data/losses_start.npy', allow_pickle=True)
        loss_final = np.load('./thesis/plot_loss/data/losses.npy', allow_pickle=True)[0][0]
        
    np.save('./thesis/plot_loss_diagram/data/losses_start.npy', losses_start)
    np.save('./thesis/plot_loss_diagram/data/loss_final.npy', loss_final)
        
plot_flag = 1
if plot_flag:

    loss_final = np.load('./thesis/plot_loss_diagram/data/loss_final.npy', allow_pickle=True)
    losses_start = np.load('./thesis/plot_loss_diagram/data/losses_start.npy', allow_pickle=True)[0]

    print(loss_final.shape)
    print(losses_start.shape)

    fig,ax = plt.subplots()
    
    start_inds = [0,1,6,8]
    for i,loss_start in enumerate(losses_start):
        if i in start_inds:
            #ax.plot(range(len(loss_start)), loss_start, label=str(i), linewidth=4)
            ax.plot(range(len(loss_start)), loss_start, color='C0', linewidth=4)
                
    #plt.legend()
    ax.plot(range(len(loss_final)), loss_final, color='C1', linewidth=6)
    
    ax.set_ylabel('Loss', fontsize=60)    
    ax.set_xlabel('Iteration', fontsize=60)

    ticks_y = [0.1, 10, 1000]
    #ticks_y = [0.1, 1, 10, 100, 1000, 10000]
    ticks_x = [1, 100, 10000]
    #ticks_x = [1, 10, 100, 1000, 10000, 100000]
    #labels_x = [1, 100, 10000]

    ax.set_xscale('log')
    ax.set_yscale('log')
    
    print(np.logspace(0,2,10))
    print(np.logspace(2,4,10))
    print(np.logspace(0,4,20))
    print(np.concatenate((np.logspace(0,2,10), np.logspace(2,4,10))))
    
    print(np.linspace(1,100,11))
    print(np.linspace(100,10000,11))
    print(np.linspace(1,10000,20))
    print(np.concatenate((np.linspace(1,100,10), np.linspace(100,10000,10))))
        
    #ax.set_xticks(ticks[i])        
    #sax.set_yticks(ticks[i])
    #ax.tick_params(axis='both', width=2, length=8, which='major')
    #x.tick_params(axis='both', width=2, length=8, which='minor')
    
    ax.spines['top'].set_linewidth(0)
    ax.spines['right'].set_linewidth(0)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)
    
    #ax.set_yticklabels([])
    #ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_xticks([])
    #ax.set_xticks(ticks_x) 
    #ax.set_yticks(ticks_y)

    fig.set_figheight(5)
    fig.set_figwidth(7)
    
    # Save Plot
    time_date = datetime.now().strftime('%H%M_%d%m')
    plt.savefig('./thesis/plot_loss_diagram/plots/loss_{}.png'.format(time_date), dpi=800, bbox_inches='tight')
    
    
