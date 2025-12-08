#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script Name: config.py
Author: Paddy Cahalane
Date: 26/8/25

Description
-----------
This module provides global configuration settings for the GANdalf bispectrum
generation and inversion framework. It defines:

- Project base paths
- Chemical element sets used in datasets
- Species-to-index mappings
- Descriptor hyperparameters (e.g., cutoff factors, twojmax values)
- Dataset naming conventions

These settings are imported across the project and must be loaded before any
tensor creation (e.g., to ensure the correct PyTorch default dtype).

Usage
-----
Import this module whenever global constants, species mappings, or descriptor
parameters are required:

    import config
    print(config.ELEMENTS)
    print(config.SPECIES_TO_SYMBOL)

Dependencies
------------
- Python 3.x
- NumPy
- PyTorch
- mendeleev (periodic table data)

Examples
--------
Retrieve the atomic number of species index 1:

    from config import SPECIES_TO_Z
    print(SPECIES_TO_Z[1])

Or list all dataset names:

    from config import MOL_LIST2
    print(MOL_LIST2)
"""

import os
import torch
import numpy as np
from mendeleev import element

# -----------------------------------------------------------------------------
# Torch default settings
# -----------------------------------------------------------------------------
# Must be set before any tensor creation across all related scripts.
torch.set_default_dtype(torch.float32)

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
# Automatically compute project root directory - absolute paths are needed in other scripts.
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
# -----------------------------------------------------------------------------
# Elements included in the dataset
# -----------------------------------------------------------------------------
# User-defined list of elements present in all relevant molecules.
ELEMENTS = ["H", "C", "N", "O"]

# -----------------------------------------------------------------------------
# LAMMPS constants 
# -----------------------------------------------------------------------------
RCUTFAC = 1.0
RFAC0 = 0.99
HARTREE2KCAL = 627.509474
KCAL2EV = 0.0433641
TWOJMAX_DICT = {30:6, 55:8} # keys=info['N_feats'] and values=twojmax

"""
# Sort elements based on atomic number
atomic_numbers = [element(e).atomic_number for e in elements]
sort_inds = np.argsort(atomic_numbers)
elements = [elements[i] for i in sort_inds]
atomic_numbers = [atomic_numbers[i] for i in sort_inds]

# Dict with int composition labels as keys and species names as values
# species_dict = {0:'H',1:'C',2:'N',3:'O'}
species_dict = {i:s for i,s in enumerate(elements)}

# Dict with species names as keys and int composition labels as values
#species_dict2 = {'H':0,'C':1,'N':2,'O':3}
species_dict2 = {s:i for i,s in enumerate(elements)}

# Dict with int composition labels as keys and atomic number as values
species_dict3 = {i:z for i,z in enumerate(atomic_numbers)}

"""
# -----------------------------------------------------------------------------
# Element sorting and species dictionaries
# -----------------------------------------------------------------------------
atomic_numbers_unsorted = [element(sym).atomic_number for sym in ELEMENTS]
sort_indices = np.argsort(atomic_numbers_unsorted)

# Sorted lists
ELEMENTS = [ELEMENTS[i] for i in sort_indices]
ATOMIC_NUMBERS = [atomic_numbers_unsorted[i] for i in sort_indices]

# Species index → element symbol (e.g., {0: "H", 1: "C", ...}) - formerly species_dict
SPECIES_TO_SYMBOL = {i: sym for i, sym in enumerate(ELEMENTS)}

# Element symbol → species index (e.g., {"H": 0, "C": 1, ...}) - formerly species_dict2
SYMBOL_TO_SPECIES = {sym: i for i, sym in enumerate(ELEMENTS)}

# Species index → atomic number (e.g., {0: 1, 1: 6, ...}) - formerly species_dict3
SPECIES_TO_Z = {i: Z for i, Z in enumerate(ATOMIC_NUMBERS)}

# -----------------------------------------------------------------------------
# Dataset naming conventions
# -----------------------------------------------------------------------------
# Original dataset folder names with number of atoms
MOL_LIST = [
    "9_ethanol", "9_malonaldehyde", "12_benzene", "12_uracil",
    "15_toluene", "16_salicylic_acid", "18_naphthalene",
    "20_paracetamol", "21_aspirin", "24_azobenzene",
]

# Clean lowercase names
MOL_LIST2 = [
    "ethanol", "malonaldehyde", "benzene", "uracil", "toluene",
    "salicylic_acid", "naphthalene", "paracetamol", "aspirin", "azobenzene",
]

# Pretty display names
MOL_LIST3 = [
    "Ethanol", "Malonaldehyde", "Benzene", "Uracil", "Toluene",
    "Salicylic Acid", "Naphthalene", "Paracetamol", "Aspirin", "Azobenzene",
]


            
