#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from typing import Tuple
from datetime import datetime

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader

from modules import config
from gan import utils
from gan import utils_plot
from gan import dataset as data 

from modules import D
from modules import G

def _parse_arguments() -> dict:
    """
    Parses command-line arguments.

    Returns
    -------
    GS
        GAN settings dictionary containing all required parameters for initialising and training. 
    """
    
    parser = argparse.ArgumentParser()

    # Dataset Parameters
    parser.add_argument('-mols', type=str, help='which molecule dataset(s) to select - e.g. "146"', default="9") # '9_ethanol', '9_malonaldehyde', '12_benzene', '12_uracil', '15_toluene', '16_salicylic_acid', '18_naphthalene', '21_aspirin',
    parser.add_argument('-test_set_split', type=float, help='amount of dataset to reserve for testing', default=0.05)

    # Training Parameters
    parser.add_argument('-resume', type=str, help='resume training GAN - if blank str "" then train from stratch', default='0102_0512')  
    parser.add_argument('-N_crit', type=int, help='num of updates for nn_D compared to nn_G', default=5)
    parser.add_argument('-N_epoch', type=int, help='total number of epochs for GAN', default=5) #5000
    parser.add_argument('-N_check', type=int, help='save NN every number of epoch', default=50)
    parser.add_argument('-N_batch', type=int, help='number of molecules averaged for each NN update', default=128)
    parser.add_argument('-N_work', type=int, help='num of workers for loading data in PyTorch - NOT TESTED', default=1)
    parser.add_argument('-gpu', type=int, help='which gpu to train on', default=1)

    # GAN Model Parameters
    parser.add_argument('-depth', type=int, help='choose depth of nn_D and nn_G', default=2)
    parser.add_argument('-distro', type=str, help='latent space distribution E {normal, uniform}', default='normal')
    parser.add_argument('-N_z', type=int, help='dimension of latent space', default=100)
    parser.add_argument('-N_units_G', type=int, help='control for num of nodes in nn_G', default=60)
    parser.add_argument('-N_units_D', type=int, help='control for num of nodes in nn_D', default=65)
    parser.add_argument('-l_fc', type=int, help='bool for whether to pass composition embedding through fc layers', default=0)
    parser.add_argument('-N_units_l', type=int, help='control for num of nodes in comp emb fc layers', default=40)

    # Loss Func Parameters
    parser.add_argument('-lr_G', type=float, help='learning rate for nn_G', default=5e-5) # was 1e-5
    parser.add_argument('-lr_D', type=float, help='learning rate for nn_D', default=5e-5) # was 1e-5
    parser.add_argument('-gp', type=int, help='bool for whether to apply gradient penalty (=1) or clipping (=0)', default=0)
    parser.add_argument('-lambda_gp', type=float, help='gradient penalty loss function coefficient', default=10.0)

    # Return args as dictionary GAN settings
    GS = vars(parser.parse_args())
    GS['names'] = [config.MOL_LIST[int(i)] for i in GS['mols']]
    GS['device'] = torch.device('cuda:'+str(GS['gpu']) if torch.cuda.is_available() else 'cpu')
    GS['time_date'] = datetime.now().strftime('%H%M_%d%m')

    return GS

# Choose DataSets and Set Sizes and make Paths
def _select_directory(base_path: str, prompt: str = "Set Type") -> str:
    """
    Helper method to select a directory based on user input.
    
    Parameters
    ----------
    base_path
        The base path to look for directories.
    prompt
        The prompt to display to the user - for reuseability.

    Returns
    -------
    dir
        The selected directory name.
    """

    dirs = [dr for dr in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, dr))]
    print('\n-------------------------------------------------')
    print('{} Options:'.format(prompt))
    for i, dr in enumerate(dirs):
        print(f'{i}) {dr}')

    # If theres only one option automatically pick that - otherwise manually select
    chosen_dr_ind = 0 if len(dirs) == 1 else int(input(f'\nChoose {prompt}: '))
    return dirs[chosen_dr_ind]

def _get_dataset_paths(GS: dict) -> dict:
    """
    Determine the dataset paths and sizes based on user input.

    Parameters
    ----------
    GS
        GAN settings dictionary.

    Returns
    -------
    GS
        Updated GAN settings dictionary with training dataset paths info.
    """

    data_paths = []
    N_mols = []

    # For each molecule type - get dataset info
    for i_set in range(len(GS['mols'])):
        load_dir_path = os.path.join(config.BASE_PATH, 'data/bispec/{}/'.format(GS['names'][i_set]))

        set_type = _select_directory(load_dir_path, "Set Type")
        set_size = _select_directory(os.path.join(load_dir_path, set_type), "Set Size")

        data_paths.append(os.path.join(load_dir_path, set_type, set_size))
        N_mols.append(int(set_size))

    # Update GS
    GS['data_paths'] = data_paths
    GS['N_mols'] = N_mols
    
    return GS

def _create_directories(GS: dict) -> dict:
    """
    Create model training path - or resume from specified one.

    Parameters
    ----------
    GS
        GAN settings dictionary.

    Returns
    -------
    GS
        Updated GAN settings dictionary with model path.
    """

    # Model Paths
    dir_name = utils.get_dir_name(GS["names"])

    # resume training
    if GS['resume'] != "":
        model_path = os.path.join(config.BASE_PATH, 'gan/savednet/{}/{}/'.format(dir_name, GS['resume']))
    # else create new dir
    else:
        model_path = os.path.join(config.BASE_PATH, 'gan/savednet/{}/{}/'.format(dir_name, GS['time_date']))
        if os.path.exists(model_path):
             shutil.rmtree(model_path)
        os.makedirs(model_path)
        os.makedirs(os.path.join(model_path, 'plots'))
        
    GS['model_path'] = model_path
    
    return GS
    
def _update_gan_settings(GS: dict) -> dict:
    """
    Update GS dict with additional training settings.

    Parameters
    ----------
    GS
        GAN settings dictionary.

    Returns
    -------
    GS
        Updated GAN settings dictionary.
    """

    # Print Load/Save Paths
    print('\nLoading From:')
    for i in range(len(GS["mols"])):
        print('    {}) {}'.format(i, GS['data_paths'][i]))
    print('\nSaving To: {}'.format(GS['model_path']))

    GS['N_updates'] = int(sum(GS['N_mols']) * (1.0 - GS['test_set_split'])) // GS['N_batch'] # Work out Update Size

    # Load Previous Model and Overwrite Settings
    if GS['resume'] != "":
        GS_old = np.load(os.path.join(GS['model_path'], 'GS.npy'), allow_pickle=True).item()
        print('\n Resuming Model Training From:', GS['model_path'])
        for key in ['N_units_D', 'N_units_G', 'N_units_l', 'l_fc', 'N_emb_l', 'gp', 'distro', 'depth', 'N_z', 'N_E']:
            GS[key] = GS_old[key]

    # Print Final GS
    print('\n\n-GAN Settings')
    for i in GS:
        print('%s:' % i, GS[i])

    # Save relative-path GS as .npy and as .txt
    GS_rel = utils.make_relative_GS(GS)
    np.save(os.path.join(GS['model_path'], 'GS'), GS_rel)
    print('\nSaving GAN Settings as .txt')
    with open(os.path.join(GS['model_path'], 'GS.txt'), 'w') as f:
        print(GS_rel, file=f)
        
    return GS
    
def initialise_gan_settings() -> dict:
    """
    Parse command-line arguments into a final GAN settings dictionary.

    Returns
    -------
    GS
        GAN settings dictionary.
    """

    GS = _parse_arguments()
    GS = _get_dataset_paths(GS)
    GS = _create_directories(GS)
    GS = _update_gan_settings(GS)

    return GS
    
def _load_data(GS: dict) -> Tuple[dict, dict, dict]:
    """
    Load training data from paths specified in GS as dictionaries of torch.tensors.

    Parameters
    ----------
    GS
        GAN settings dictionary.

    Returns
    -------
    train_data
        Training dataset dictionary containing all relevant torch.tensors.
    test_data
        Testing dataset dictionary containing all relevant torch.tensors.
    info
        Dataset information dictionary.
    """

    # Load Info - seperate to load data because args.e
    infos = []
    for i in range(len(GS['mols'])):
        infos.append(np.load(os.path.join(GS['data_paths'][i], str(GS['N_mols'][i])+'_info.npy'), allow_pickle=True).item())
    info = data.combined_info(infos)

    # Load Train and Test Data
    train_data = {'Bs': [], 'Ls': [], 'Xs': [], 'Es': []}
    test_data = {'Bs': [], 'Ls': [], 'Xs': [], 'Es': []}
    for i in range(len(GS['mols'])):
        b = torch.tensor(np.load(os.path.join(GS['data_paths'][i], str(GS['N_mols'][i])+'_B.npy')), dtype=torch.float32)
        l = torch.tensor(np.load(os.path.join(GS['data_paths'][i], str(GS['N_mols'][i])+'_L.npy')), dtype=torch.int32)
        x = torch.tensor(np.load(os.path.join(GS['data_paths'][i], str(GS['N_mols'][i])+'_X.npy')), dtype=torch.float32)
        e = torch.tensor(np.load(os.path.join(GS['data_paths'][i], str(GS['N_mols'][i])+'_E.npy')), dtype=torch.float32)

        b, b2, l, l2, x, x2, e, e2 = train_test_split(b, l, x, e, test_size=GS['test_set_split'], shuffle=True)

        train_data['Bs'].append(b), train_data['Ls'].append(l), train_data['Xs'].append(x), train_data['Es'].append(e)
        test_data['Bs'].append(b2), test_data['Ls'].append(l2), test_data['Xs'].append(x2), test_data['Es'].append(e2)
        
    return train_data, test_data, info

def _feature_scaling(
    train_data: dict, 
    test_data: dict,
    info: dict,
    GS: dict
) -> Tuple[dict, dict, dict]:
    """
    Scale descriptors and energies using meand and standard deviations from info.

    Parameters
    ----------
    train_data
        Training dataset dictionary containing all relevant arrays.
    test_data
        Testing dataset dictionary containing all relevant arrays.
    info
        Dataset information dictionary.
    GS
        GAN settings dictionary.

    Returns
    -------
    train_data
        Scaled training dataset dictionary containing all relevant arrays.
    test_data
        Scaled testing dataset dictionary containing all relevant arrays.
    info
        Updated dataset information dictionary with energy stats.
    """

    # Append Feature Scaling Stuff to Info
    train_data['Bs_sum'] = data.sum_over_species(train_data['Bs'], train_data['Ls'], info)
    test_data['Bs_sum'] = data.sum_over_species(test_data['Bs'], test_data['Ls'], info)

    # FS B with standardScaler - fit/transform train and only transform test
    train_data['Bs_std'], stds = data.B_scaling(train_data['Bs_sum'], info)   
    test_data['Bs_std'] = data.B_scaling(test_data['Bs_sum'], info, stds)   

    # Add FS for B/E t0 info
    info['B_stds'] = stds

    # Save info 
    print('\nSaving Info as .txt')
    np.save(os.path.join(GS['model_path'], 'info'), info)
    with open(os.path.join(GS['model_path'], 'info.txt'), 'w') as f:
        print(info, file=f)
        
    return train_data, test_data, info

def _get_dataloaders(
    train_data: dict, 
    test_data: dict,
    info: dict,
    GS: dict
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create torch dataloaders for inner training loop using train/test datasets.

    Parameters
    ----------
    train_data
        Training dataset dictionary containing all relevant arrays.
    test_data
        Testing dataset dictionary containing all relevant arrays.
    info
        Dataset information dictionary.
    GS
        GAN settings dictionary.

    Returns
    -------
    train_dl
        Torch dataloader returning training batches of descriptors, labels, and energies at each call.
    test_dl
        Torch dataloader returning testing batches of descriptors, labels, and energies at each call.
    """
    train_dataset = data.BL_DataSet(train_data, info, GS)    
    train_dl = DataLoader(train_dataset, batch_size=GS['N_batch'], shuffle=True, collate_fn=train_dataset.my_collate)   
    
    test_dataset = data.BL_DataSet(test_data, info, GS)    
    test_dl = DataLoader(test_dataset, batch_size=GS['N_batch'], shuffle=True, collate_fn=test_dataset.my_collate)       

    return train_dl, test_dl

def initialise_dataloaders(GS: dict) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, dict]:
    """
    Load training/testing datasets from GAN settings, and convert them into torch dataloders. 

    Parameters
    ----------
    GS
        GAN settings dictionary.

    Returns
    -------
    train_dl 
        Torch dataloader returning training batches of descriptors, labels, and energies at each call.
    test_dl
        Torch dataloader returning testing batches of descriptors, labels, and energies at each call.
    info
        Dataset information dictionary.
    """
    
    train_data, test_data, info = _load_data(GS)
    train_data, test_data, info = _feature_scaling(train_data, test_data, info, GS)
    train_dl, test_dl = _get_dataloaders(train_data, test_data, info, GS)
    
    return train_dl, test_dl, info
    
# -----------------------------------------------------------------------------------------------------------------------
# - Initialise(/Load) Models
# -----------------------------------------------------------------------------------------------------------------------
        
def load_models(info: dict, GS: dict) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """
    Load training/testing datasets from GAN settings, and convert them into torch dataloders. 

    Parameters
    ----------
    info
        Dataset information dictionary.
    GS
        GAN settings dictionary.

    Returns
    -------
    nn_D
        GAN discriminator model.
    nn_G
        GAN generator model.
    """
    
    # Initialise Models
    nn_D, nn_G = D.Discriminator(info, GS), G.Generator(info, GS)
  
    # Load Models to GPU/CPU
    nn_D, nn_G = nn_D.to(GS['device']), nn_G.to(GS['device'])

    # Resume G and D
    if GS['resume'] != "":
        start_epoch = G.last_epoch(GS)
        D_path = os.path.join(GS['model_path'], 'D'+str(start_epoch))
        G_path = os.path.join(GS['model_path'], 'G'+str(start_epoch))
        nn_D.load_state_dict(torch.load(D_path, map_location=GS['device']))
        nn_G.load_state_dict(torch.load(G_path, map_location=GS['device']))

    print('\nnn_D total_params:', sum(p.numel() for p in nn_D.parameters()))
    print('nn_G total_params:', sum(p.numel() for p in nn_G.parameters()))
    
    return nn_D, nn_G
    
# -----------------------------------------------------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------------------------------------------------     

def train_gan(
    nn_D: torch.nn.Module, 
    nn_G: torch.nn.Module, 
    train_dl: torch.utils.data.DataLoader,
    test_dl: torch.utils.data.DataLoader, 
    info: dict, 
    GS: dict
) -> None:
    """
    Main GAN training loop, yielding a trained nn_G used to generate fake samples.  

    Parameters
    ----------
    nn_D
        GAN discriminator model.
    nn_G
        GAN generator model.
    train_dl
        Torch dataloader returning training batches.
    test_dl
        Torch dataloader returning testing batches.
    info
        Dataset information dictionary.
    GS
        GAN settings dictionary.

    Returns
    -------
    None
        
    Notes
    -----
    - The new weights for nn_D and nn_G models are saved every checkpoint at relevent paths in GS.
    - Loss curve, descriptor, and energy alignment plots are automatically saved in same dir as weights.
    - Works for either gradient penalty (slow) or weight clipping (faster).     
    """

    # Optimizers and E loss function
    D_op = torch.optim.RMSprop(nn_D.parameters(), lr=GS['lr_D'])
    G_op = torch.optim.RMSprop(nn_G.parameters(), lr=GS['lr_G'])
    
    Cost = []
    for e in range(1, GS['N_epoch']+1):
        cost_i = []

        nn_G.train(), nn_D.train()
        for i, (B_real, L_real) in tqdm(enumerate(train_dl),
                                                total=GS['N_updates']-1,
                                                desc="epoch {}".format(e)
                                                ):
            B_real, L_real = B_real.to(GS['device']), L_real.to(GS['device'])
            
            for _ in range(GS['N_crit']):
                D_op.zero_grad()

                # Train nn_D on Real Data
                D_real = nn_D(B_real, L_real)

                # Train nn_D on Fake Data
                Z, L_fake = G.make_seeds(info, GS)
                B_fake = nn_G(Z, L_fake)
                D_fake = nn_D(B_fake, L_fake)

                # Update nn_D    
                D_loss = -(torch.mean(D_real) - torch.mean(D_fake))

                # apply gradient penalty 
                if GS['gp']:
                    D_loss += GS['lambda_gp'] * D.gradient_penalty(nn_D, B_real, L_real, B_fake)
                    D_loss.backward(retain_graph=True)
                    D_op.step()

                # else apply weight clipping
                else:
                    D_loss.backward()
                    D_op.step()

                    for p in nn_D.parameters():
                        p.data.clamp_(-0.01, 0.01)

            # Train nn_G
            G_op.zero_grad()
            Z, L_fake = G.make_seeds(info, GS)
            B_fake = nn_G(Z, L_fake) 
            D_fake = nn_D(B_fake, L_fake)

            # Update nn_G
            G_loss = -torch.mean(D_fake)
            G_loss.backward()
            G_op.step()

            # Statistics
            cost_i.append([D_loss.item(), G_loss.item()])

            # Break Loop when Epoch is up - needed as len(BL_dl) is sum(num_mols)
            if i == GS['N_updates']-1:
                break

        # Print updates to command line
        Cost.append(np.mean(cost_i, axis=0))
        if len(Cost) == 1:
            print('Epoch: {}, D_L:{}, G_L: {}'.format(e, np.round(Cost[0][0],5), np.round(Cost[0][1],5)))
        else:
            print('Epoch: {}, D_L:{}, G_L: {}'.format(e, np.round(Cost[-1][0],5), np.round(Cost[-1][1],5)))

        # Every checkpoint - save models and plots
        if e % GS['N_check'] == 0 or e == 1:
            # Save models
            print('Saving Model')
            torch.save(nn_G.state_dict(), os.path.join(GS['model_path'], 'G'+str(e)))
            torch.save(nn_D.state_dict(), os.path.join(GS['model_path'], 'D'+str(e)))
            
            # Generate Plots
            utils_plot.plot_b(info, GS, [B_real.cpu().detach().permute((1,0,2))], [L_real.cpu().detach()], [B_fake.cpu().detach().permute((1,0,2))], [L_fake.cpu().detach()], show=False)   
            utils_plot.images_b(info, GS, B_real.detach().permute((1,0,2)), B_fake.detach().permute((1,0,2)))                      
            
    # Save model costs for loss plots
    np.save(os.path.join(GS['model_path'], 'Cost.npy'), Cost)

def main() -> None:
    """
    Main function to run the script. Creates training settings, loads data, initialises models, trains energy model, and then trains GAN.   
    """

    GS = initialise_gan_settings()
    train_dl, test_dl, info = initialise_dataloaders(GS)
    nn_D, nn_G = load_models(info, GS)
    train_gan(nn_D, nn_G, train_dl, test_dl, info, GS)

if __name__ == "__main__":
    main()


