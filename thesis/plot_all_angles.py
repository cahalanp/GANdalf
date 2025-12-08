#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#-----------------------------------------------------------------------------------------------------------------------
# - Importing Modules
# - Arg-Parser 
#-----------------------------------------------------------------------------------------------------------------------     

import copy
import argparse
import numpy as np
from datetime import datetime
from random import shuffle

from FOX import MultiMolecule

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

from modules import config
 
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
    parser.add_argument('-data_flag', type=bool, help="whether to calculate data used for plotting.", default=0)
    parser.add_argument('-plot_flag', type=bool, help="whether to save a new plot", default=1)
    parser.add_argument('-real_or_gen', type=bool, help="whether to plot real (T) or gen (F) inverted mols", default=0)
    args = parser.parse_args()
    
    return args

def main():

    # Parse args and Load data
    args = parse_arguments()
    names, pretty_names, data_paths, num_mols = utils_plot.construct_data_paths(args.mols)
    data_dict, _ = utils_plot.load_data(data_paths, num_mols)
    
    # Number of molecules to include    
    N_max_r = 10000
    N_max_g = 200

    # Whether to use mols saved in converged or unconverged directories
    conv_types = ['not_conv', 'not_conv', 'not_conv', 'not_conv', 'not_conv', 'not_conv', 'not_conv', 'not_conv', 'not_conv', 'not_conv', 'not_conv' ]

    # Directories names where inverted molecules are kept
    if args.real_or_gen:
        label_2 = 'Inverted'
        s_G = ["real" for _ in range(len(names))]
    else:
        label_2 = 'Generated'
        s_G = ['0022_0110', '1720_0210','0022_0110','1653_0210','0013_0510','0014_0510','0933_0810','0933_0810','1246_1211','1130_0711']

    if args.data_flag:

        data = []
        
        for i in range(len(names)):
            xyz_r, l_r = data_dict["Xs"][i], data_dict["Ls"][i]
            
            # Get Inverted XYZ 
            temp = 0
            if not temp:
                invert_path = "./invert/{}/{}/".format(len(data_dict["Xs"][i][0]), s_G[i])  
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
            
            # Convert to FOX atoms
            str_labels = utils.inv_convert_labels(l_r[0])

            labels_dict = {}
            for j,s in enumerate(str_labels):
                if s not in labels_dict.keys():
                    labels_dict[s] = [j]
                else:
                    labels_dict[s].append(j)

            mols_r = MultiMolecule(xyz_r, labels_dict)
            mols_g = MultiMolecule(xyz_g, labels_dict)
            
            # Calc Angle Distribution Function
            adf_r = mols_r.init_adf()
            adf_g = mols_g.init_adf()
            
            data.append([adf_r.sum(1).to_numpy(), adf_g.sum(1).to_numpy()])
            np.save("./thesis/plot_all_angles/data/{}.npy".format(names[i]), data[i]) 

    # Load previously saved plotting data
    else:
        data = []
        for i in range(len(names)):
            data.append(np.load("./thesis/plot_all_angles/data/{}.npy".format(names[i])))

    # Actually write/show plot
    if args.plot_flag:

        # Colors and Plot Labels
        #d_bin = 1
        density = True
        data_labels = ['Real', label_2]
        colors = ['C1', 'C0']

        fig, axs = plt.subplots(nrows=len(names), ncols=1, sharex=True, sharey=False) 

        for i in range(len(names)):
        
            # normalise 
            data[i][0] = data[i][0] / sum(data[i][0])
            data[i][1] = data[i][1] / sum(data[i][1])
            
            #axs[i].text('Count', fontsize=8)  
            axs[i].text(0.5, 0.85, r"\underline{%s}" % pretty_names[i], ha='center', va='center', transform=axs[i].transAxes, fontsize=15)
            axs[i].tick_params(axis='both', labelsize=13)
            axs[i].set_xticks([0, 30, 60, 90, 120, 150, 180]) 
            axs[i].set_xlim(-3,183)
            from matplotlib.ticker import MultipleLocator
            ml = MultipleLocator(7.5)
            axs[i].xaxis.set_minor_locator(ml)
            
            #bins0 = np.arange(min(data[i][0]), max(data[i][0]) + d_bin, d_bin)
            #bins1 = np.arange(min(data[i][1]), max(data[i][1]) + d_bin, d_bin)
            #axs[i].hist(data[i][0], density=density, bins=bins0, histtype='step', color=colors[0], label=data_labels[0], lw=1)
            #axs[i].hist(data[i][1], density=density, bins=bins1, histtype='step', color=colors[1], label=data_labels[1], lw=1)
            
            dummy_x = np.arange(180)
            _, _, poly_r = axs[i].hist(dummy_x, density=density, bins=range(0,180), histtype='step', color=colors[0], label=data_labels[0], lw=1)
            _, _, poly_g = axs[i].hist(dummy_x, density=density, bins=range(0,180), histtype='step', color=colors[1], label=data_labels[1], lw=1)

            xy_r, xy_g = poly_r[0].get_xy(), poly_g[0].get_xy()
            for k in range(1,180):
                xy_r[(k*2)-1][1] = data[i][0][k]
                xy_r[(k*2)][1] = data[i][0][k]
                xy_g[(k*2)-1][1] = data[i][1][k]
                xy_g[(k*2)][1] = data[i][1][k]
            poly_r[0].set_xy(xy_r), poly_g[0].set_xy(xy_g)
            
            axs[i].set_ylim(0,max(data[i][1])*1.2)

        # Plot Settings
        #fig.supylabel('Count', fontsize=8)
        fig.text(0.065, 0.5, 'Frequency [arb. unit]', ha='center', va='center', rotation='vertical', fontsize=17)
        axs[-1].set_xlabel(r'$\mathrm{\theta}_{ijk}$ [deg]', fontsize=17)
        axs[0].legend(fontsize=15, loc="upper right", frameon=False,)
        
        #axs[0].scatter(x_train, y_train, s=14, alpha=0.7, color="black", label="Data")
        #axs[0].plot(x_data, y_pred_under, color="red", alpha=0.8, label="Under Fit")
        #axs[0].legend(loc="upper left", frameon=False,)
        #axs[0].set_xticks([])
        #axs[0].set_yticks([])

        #fig.tight_layout()        
        fig.set_figheight(20)     
        #fig.set_figheight(14)
        fig.set_figwidth(10)

        now = datetime.now()
        time_date = now.strftime('(%H%M_%d%m)')
        #plt.savefig('./thesis/plot_all_angles/plots/all_angles_'+str(time_date)+'.pdf', bbox_inches='tight')
        plt.savefig('./thesis/plot_all_angles/plots/all_angles_'+str(time_date)+'.png', dpi=500, bbox_inches='tight')
        #plt.show()
        

if __name__ == "__main__":
    main()

