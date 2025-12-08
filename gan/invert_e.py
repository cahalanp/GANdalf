#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np

from ase.visualize import view

from modules import config
from modules import E as E

from gan import utils_plot
from gan import utils_e as utils
from gan import train_e as train

def parse_arguments() -> argparse.ArgumentParser:
    """
    Parses command-line arguments.

    Returns
    -------
    args
        Parsed arguments.  
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-mols', type=str, help="Which molecule types to invert - indices in config.mol_list", default='9') # '9_ethanol', '9_malonaldehyde', '12_benzene', '12_uracil', '15_toluene', '16_salicylic_acid', '18_naphthalene', '20_para', '21_aspirin', '24_azob'
    parser.add_argument('-N_inv', type=int, help="Number of conformations to invert (for each set_ind)", default=1)
    parser.add_argument('-real_flag', type=bool, help="Invert real (True) or generated (False) molecules", default=False)
    #parser.add_argument('-s_G', type=str, help="Name of dir containing model/data", default='0022_0110') # ethanol
    #parser.add_argument('-s_G', type=str, help="Name of dir containing model/data", default='1720_0210') # mal
    #parser.add_argument('-s_G', type=str, help="Name of dir containing model/data", default='0022_0110') # benz
    #parser.add_argument('-s_G', type=str, help="Name of dir containing model/data", default='1653_0210') # ura
    #parser.add_argument('-s_G', type=str, help="Name of dir containing model/data", default='0013_0510') # tol
    #parser.add_argument('-s_G', type=str, help="Name of dir containing model/data", default='0014_0510') # sal
    #parser.add_argument('-s_G', type=str, help="Name of dir containing model/data", default='0933_0810') # naph
    #parser.add_argument('-s_G', type=str, help="Name of dir containing model/data", default='0933_0810') # para
    #parser.add_argument('-s_G', type=str, help="Name of dir containing model/data", default='1130_0711') # asp
    parser.add_argument('-s_G', type=str, help="Name of dir containing model/data", default='1130_0711') # azo
    #parser.add_argument('-s_G', type=str, help="Name of dir containing model/data", default='0043_1511') # 246
    parser.add_argument('-device', type=str, default='cpu') # azo
    args = parser.parse_args()

    return args

def load_GS(args: argparse.ArgumentParser) -> dict:
    """
    Converts args to GS dict.

    Returns
    -------
    dict
        GAN Settings dictionary.  
    """

    # Data Paths Info
    names = [config.MOL_LIST[int(i)] for i in args.mols]

    # Model Path
    dir_name = utils.get_dir_name(names)
    model_path = os.path.join(config.BASE_PATH, 'gan/savednet/{}/{}/'.format(dir_name, args.s_G))

    # Load GS
    GS = np.load(os.path.join(model_path, 'GS.npy'), allow_pickle=True).item()  

    GS["model_path"] = model_path
    GS["device"] = args.device # device
    GS["N_inv"] = args.N_inv # number of molecules to invert
    GS["real_flag"] = args.real_flag # whether to invert real or gen data

    return GS
    
def invert_molecules(nn_G, nn_E, info, GS, train_data, **kwargs):
    """
    Simple wrapper for utils.gen_xyz(), to allow kwargs.
    
    Invert generated bispectrum components into 3D coordinates.
    """
    
    default_kwargs = dict(
        get_trj = False, 
        save_trj = False, 
        save_end = True, 
        pbc_flag = True,
        loss_threshold = 2e-10,
        max_N = 100000, 
        start_N = 500, 
        start_M = 50,
        lr = 3e-7, 
        alpha = 1e-3, 
        eta = 1e2,
        rand_target_ind = False
    )
    
    default_kwargs.update(kwargs)
    return utils.gen_xyz(nn_G, nn_E, info, GS, train_data, **default_kwargs)

def main() -> None:
    """
    Main function to run the script. Creates training settings, loads data, initialises models, trains energy model, and then trains GAN.   
    """

    # Load command-line args and GS
    args = parse_arguments()
    GS = load_GS(args)
    
    # Load scaled data
    train_data, test_data, info = train._load_data(GS)
    train_data, test_data, info = train._feature_scaling(train_data, test_data, info, GS)
    
    # Load models
    _, nn_G, nn_E = train.load_models(info, GS)

    # Load 
    trjs, losses, _ = invert_molecules(nn_G, nn_E, info, GS, train_data)

    # Plot Loss and view Optimization Trjectory for 1st mol
    plot_flag = 0
    if plot_flag:
        for i in GS["mols"]:
            utils_plot.plot_invert_loss(losses[i][0])
            view(trjs[i][0])
            pass


if __name__ == "__main__":
    main()



