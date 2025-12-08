#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
This module is dedicated to the computation of the bisectrum 
components (it makes use of LAMMPS)

@author: matteo
'''

from cobe import Datadict
from lammps import lammps
from lammps import LMP_STYLE_ATOM, LMP_TYPE_ARRAY
from ase.io import lammpsdata
from ase.visualize import view
from tqdm import tqdm
from datetime import datetime

import sys
import numpy as np
import dask
from lammps.numpy_wrapper import numpy_wrapper
from cobe.descriptors.utils.lammps import atoms2lammpscmds
import numpy as np
from ase.data import atomic_masses

from dask.distributed import Client
from dask import delayed, compute

sys.path.append('home/cahalanp/phd/gandalf/james/1singlemode/lmpdat')

def Bispectrum(dataset, rcutfac=0, rfac0=0, twojmax=0, rc=[], pbar=True, pbc_flag=False):

    if pbar:
        for i in tqdm(range(dataset.n_molecules)):
            compute_bispectrum(dataset.trajectory[i],
                                dataset.types,
                                rcutfac,
                                rfac0,
                                twojmax,
                                rc,
                                pbc_flag
                                )
    else:
        for i in range(dataset.n_molecules):
            compute_bispectrum(dataset.trajectory[i],
                                dataset.types,
                                rcutfac,
                                rfac0,
                                twojmax,
                                rc,
                                pbc_flag
                                )
                                
    return Datadict(dataset)

def compute_bispectrum(molecule, types=[], rcutfac=0, rfac0=0, twojmax=0, rc=[], pbc_flag=False):

    header = ["units metal", 
                "dimension 3",
                "boundary p p p",
                "atom_style atomic",
                "box tilt large",
                "atom_modify map array sort 0 10.0" 
                ] 
                
    molecule.cell = [[100.0, 0.00, 0.00],
                          [0.00, 100.0, 0.00],
                          [0.0, 0.00, 100.0]]

    typeMap = {}
    for i,n in enumerate(types):
        typeMap[n] = [i+1,atomic_masses[n]]

    molecule_cmd = atoms2lammpscmds(molecule,typeMap)

    compute_settings = str(rcutfac) + " "
    compute_settings += str(rfac0) + " "
    compute_settings += str(twojmax) + " " 
    compute_settings += " ".join([str(rc_i) for rc_i in rc]) + " "
    compute_settings += " ".join([str(1.0) for _ in range(len(set(types)))]) 
    compute_settings += " rmin0 0.0" 

    b_cmd = "compute b all sna/atom " + compute_settings
    bd_cmd = "compute bd all snad/atom " + compute_settings 

    n_b = int((twojmax/2+1)*(twojmax/2+1.5)*(twojmax/2+2)/3)  # Wood, Thopson J., Chem. Phys. 148, 241721 (2018) tab.1 

    body = ['pair_style lj/cut 20','pair_coeff * * 1 1'] + [b_cmd] + [bd_cmd] + ["run 0"]

    cmd_list = header+molecule_cmd+body
    
    #print(cmd_list)
    #print(compute_settings)
    #sys.exit()

    lmp = lammps(cmdargs = ['-l','none','-screen','none'])
    npw = numpy_wrapper(lmp)
    lmp.commands_list(cmd_list)

    b = npw.extract_compute('b',LMP_STYLE_ATOM,LMP_TYPE_ARRAY).copy()
    bd = npw.extract_compute("bd", LMP_STYLE_ATOM,LMP_TYPE_ARRAY).copy()

    lmp.close()

    bd_out = np.zeros((len(bd),len(types),3,n_b))

    for i in range(len(bd)):
        c=0

        for j in range(len(types)):
            for k in range(3):
                for p in range(n_b):                        
                    bd_out[i][j][k][p] = bd[i][c]
                    c+=1  

    molecule.descriptors = b
    molecule.descriptors_d = bd_out

def compute_bispectrum2(molecule, types=[], rcutfac=0, rfac0=0, twojmax=0, rc=[], weights=[], pbc_flag=False):

    rc = str(rc)

    rc = rc.replace('[', '')
    rc = rc.replace(']', '')
    rc = rc.replace(',', '')

    weights = str(weights)

    weights = weights.replace('[', '')
    weights = weights.replace(']', '')
    weights = weights.replace(',', '')

    n_b = int((twojmax/2+1)*(twojmax/2+1.5) *
              (twojmax/2+2)/3)   # Wood, Thopson tab.1

    molecule.cell = [[100.0, 0.00, 0.00],
                     [0.00, 100.0, 0.00],
                     [0.0, 0.00, 100.0]]

    #molecule.wrap(pbc=True, center=True, pretty_translation=True)   
    #molecule.wrap(pbc=False, center=False, pretty_translation=False) 
    molecule.wrap(pbc=pbc_flag, center=pbc_flag, pretty_translation=pbc_flag)
    #molecule.wrap(pbc=False, center=False, pretty_translation=False) 
    
    lammps_data_path = 'cache_1.lmpdat'
    lammpsdata.write_lammps_data(lammps_data_path,
                                 molecule, 
                                 specorder=None, force_skew=False,
                                 prismobj=None, velocities=False, units="real",
                                 atom_style='atomic')
    
    # pyLAMMPS
    args = "-screen none"
    words = args.split()
    lmp = lammps(cmdargs=words)
    #lmp = lammps()

    lmp.command("units          real")
    lmp.command("dimension      3")
    lmp.command("boundary       p p p")
    lmp.command("atom_style     atomic")
    lmp.command("read_data      {}".format(lammps_data_path))
    lmp.command("replicate      1 1 1")

    for i in range(molecule.n_types):
        lmp.command("mass "+str(i+1)+" 10.00")

    lmp.command("pair_style lj/cut 20.0")
    lmp.command("pair_coeff * * 20.0 20.0")

    lmp.command("compute b all sna/atom "+str(rcutfac)+" " +
                str(rfac0)+" "+str(twojmax)+" "+rc+" "+weights+" rmin0 0.0")
    lmp.command("compute bd all snad/atom "+str(rcutfac)+" " +
                str(rfac0)+" "+str(twojmax)+" "+rc+" "+weights+" rmin0 0.0")

    #lmp.command("dump mydump_1 all custom 1 b_test.dat c_b[*]")
    #lmp.command("dump mydump_2 all custom 2 bd_test.dat c_bd[*]")

    lmp.command("run 0")

    b = lmp.extract_compute("b", 1, 2)  # Extract LAMMPS bispectrum components
    bd = lmp.extract_compute("bd", 1, 2)

    for i in range(molecule.n_atoms):
        for j in range(n_b):
            molecule.descriptors[i].append(b[i][j])

    for i in range(molecule.n_atoms):
        c = 0
        for j in range(molecule.n_types):
            for k in range(3):
                for p in range(n_b):
                    molecule.descriptors_d[i][j][k].append(bd[i][c])
                    #molecule.descriptors_d[i][j][k][p].append(bd[i][c])
                    c += 1
                              
