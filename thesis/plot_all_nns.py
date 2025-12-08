#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np
from datetime import datetime

import torch

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import AutoMinorLocator
plt.rcParams['text.usetex'] = True

from modules import config
from modules import E

from gan import utils_plot
from gan import utils_e as utils

def parse_arguments() -> argparse.ArgumentParser:
    """
    Parses command-line arguments.

    Returns
    -------
    args
        Parsed arguments object. 
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-mols', type=str, help="indices of config.MOL_LIST to plot", default="0123456789") # '9_ethanol', '9_malonaldehyde', '12_benzene', '12_uracil', '15_toluene', '16_salicylic_acid', '18_naphthalene', '21_aspirin',
    parser.add_argument('-set_type', type=str, help="which dataset type to choose from - ['train', 'test', 'val']", default="val")
    parser.add_argument('-data_flag', type=bool, help="whether to calculate data used for plotting", default=1)
    parser.add_argument('-plot_flag', type=bool, help="whether to save a new plot", default=1)
    args = parser.parse_args()
    
    return args

def main():

    # Parse args and Load data
    args = parse_arguments()
    names, pretty_names, data_paths, num_mols = utils_plot.construct_data_paths(args.mols, args.set_type)
    data_dict, infos = utils_plot.load_data(data_paths, num_mols)

    # Number of molecules to include    
    n_mol = 200

    # Directories names where inverted molecules are kept
    s_G = ['0022_0110', '1720_0210','0022_0110','1653_0210','0013_0510','0014_0510','0933_0810','0933_0810','1246_1211','1130_0711']

    if args.data_flag:

        for i in range(len(names)):
            dir_name = utils.get_dir_name([names[i]])
            model_path = "./gan/savednet/{}/{}/".format(dir_name, s_G[i])
        
            _GS = np.load(os.path.join(model_path, 'GS.npy'), allow_pickle=True).item()
            _info = infos[i]

            _info["N_all_species"] = len(_info['species'])
            _info["all_species"] = sorted(_info['species'])
            _GS['device'] = "cpu"

            nn_E = E.Energyator(_info, _GS)
            nn_E.load_state_dict(torch.load(os.path.join(model_path, 'E')))
            
            rand_inds = torch.randint(0, len(data_dict['Es'][i]), (n_mol,))
            target_es = data_dict['Es'][i][rand_inds]
            
            b = data_dict['Bs'][i][rand_inds].permute((1,0,2))
            
            print(b.shape)

            nn_E.eval()
            nn_es = nn_E(b).detach()
            
            nn_es = norms[i].inverse_transform(nn_es).squeeze(1)
                
            np.save('./plots/plot_all_nns/data/{}_target_es.npy'.format(names[i]), target_es)
            np.save('./plots/plot_all_nns/data/{}_nn_es.npy'.format(names[i]), nn_es)
        
    if args.plot_flag:

        target_Es, nn_Es = [], []
        
        for i,name in enumerate(names):
            _p = np.load('./plots/plot_all_nns/data/{}_nn_es.npy'.format(name))
            _t = np.load('./plots/plot_all_nns/data/{}_target_es.npy'.format(name))
            
            _p = utils.kcal2ev(_p, infos['N_at'][i])
            _t = utils.kcal2ev(_t, infos['N_at'][i])
            
            nn_Es.append(_p)
            target_Es.append(_t)
            
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
        
        ticks = [[-467.76, -467.72, -467.68],[-806.05, -806.10, -806.15], [-525.52, -525.54, -525.56], [-938.82, -938.87, -938.92], [-491.53, -491.56, -491.59], [-841.98, -842.02, -842.06], 
                [-582.10, -582.13, -582.16], [-699.90, -699.94, -699.98], [-838.89, -838.93, -838.97], [-648.03, -648.06, -648.09]]

        for i in range(len(names)):
            target_es, nn_es = target_Es[i], nn_Es[i]
            
            def get_r2(y, y_hat):
                y_bar = y.mean()
                ss_tot = ((y-y_bar)**2).sum()
                ss_res = ((y-y_hat)**2).sum()
                return 1 - (ss_res/ss_tot)
            
            # calc metrics
            r = 5
            mae = np.round(np.mean(np.abs(np.subtract(nn_es, target_es))), r)
            rms = np.round(np.sqrt(np.mean((np.subtract(nn_es, target_es))**2)), r)
            _max = np.round(max(np.subtract(nn_es, target_es)), r)
            r2 = get_r2(target_es, nn_es)
            print("name, r2", names[i], r2)

            # Square plot    
            axs[i].set_aspect("equal")
            min_max = [min(np.concatenate((nn_es, target_es))), max(np.concatenate((nn_es, target_es)))]
            offset = (min_max[1] - min_max[0]) * 0.1
            axs[i].set_ylim(min_max[0]-offset, min_max[1]+offset)
            axs[i].set_xlim(min_max[0]-offset, min_max[1]+offset)

            # Draw the scatter plot and marginals.
            y_min_max = [min(nn_es), max(nn_es)]
            xy1, xy2 = [y_min_max[0], y_min_max[0]], [y_min_max[1], y_min_max[1]]
            axs[i].axline(xy1, xy2, linestyle=":", c='black', alpha=0.8)
            axs[i].scatter(target_es, nn_es, c='C1', alpha=0.5)
        
            # Text box
            props = dict(boxstyle='square,pad=-0.1', facecolor='white', alpha=0.0, edgecolor='none')
            axs[i].text(0.05, 0.96, r"\underline{%s}" % pretty_names[i], transform=axs[i].transAxes, fontsize=18, verticalalignment='top', bbox=props)
            axs[i].text(0.05, 0.925, '\nMAE={} [eV/atom]'.format(mae), transform=axs[i].transAxes, fontsize=15, verticalalignment='top', bbox=props)
            axs[i].text(0.05, 0.775, r"$R^2$={%s}" % str(np.round(r2,3)), transform=axs[i].transAxes, fontsize=15, verticalalignment='top', bbox=props)
            
            # ticks
            axs[i].set_xticks(ticks[i])        
            axs[i].set_yticks(ticks[i])
            axs[i].xaxis.set_minor_locator(AutoMinorLocator(5))
            axs[i].yaxis.set_minor_locator(AutoMinorLocator(5))
                    
            axs[i].tick_params(axis='both', which='major', labelsize=19)
            axs[i].ticklabel_format(useOffset=False)
            
            # grid
            axs[i].grid(which='major', alpha=0.5, linewidth=1.1)
            axs[i].grid(which='minor', alpha=0.2)
            
        fig.text(0.00, 0.493, 'BPNN Energy [eV/atom]', ha='center', va='center', rotation='vertical', fontsize=23)
        fig.text(0.515, -0.01, 'Original DFT Energy [eV/atom]', ha='center', va='center', rotation='horizontal', fontsize=23)
            
        fig.set_figheight(17)
        fig.set_figwidth(13)
        fig.tight_layout(pad=1.0, h_pad=1.0, w_pad=1.0) 
        
        # Save Plot
        time_date = datetime.now().strftime('%H%M_%d%m')
        plt.savefig('./plots/plot_all_nns/plots/all_ten_parity_{}.pdf'.format(time_date), bbox_inches='tight')
        
        print("finished")
        sys.exit()
        
if __name__ == "__main__":
    main()