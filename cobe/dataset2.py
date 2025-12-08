 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
This module defines the Set class

@author: matteo
'''
import numpy as np
import os

from cobe.molecule import Molecule
from ase import Atom, Atoms
from ase.io import read
from ase.io.extxyz import read_xyz

class Dataset2():
    '''
    Dataset class
    ---------    
    This is the main class of this library. It represent a set of Molecules.
              
    Parameters
    ----------        
    trajectory : list of Molecules
        List containing the molecules composing the Dataset object.
    energies : list of float
        List of n_molecus elements, the i-element correspond to the mass of
        the i-molecule
    labels : list of str 
        list of the diffirent labels contained in the molecules of the dataset.
    types : list of str 
        list of the diffirent types contained in the molecules of the dataset.
    n_molecules : int
        Number of molecules contained in the dataset.   
    n_types : int
    
    types : dictionary of str
        Dictionary containing the correspondence between the atomic species
        and the type.
    
    Methods  
    -------  
          
    '''
    
    def __init__(self, xs, l):
        
        self.trajectory = list(map(Molecule, [Atoms(l, positions=xs[i]) for i in range(len(xs))]))
            
        self.energies = None
        self.n_molecules = len(self.trajectory) 
        
        names = sorted(set(self.trajectory[0].symbols))
        self.types = []
        
        for name in names:
            self.types.append(Atom(name).number)       
        
        self.n_types = len(self.types)
        
    def set_energies(self, es):
        '''
        set_energies
        ------------        
        Sets the energy of the molecules contained in the file xyz_file to the 
        values contained in the file energy_file.
        
        Parameters
        ----------
        xyz_file : str
            path to the file containing the coordinates of the molecules.
        energy_file : str
            path to the file containing the energies of the molecules, 
            it assumes '\n' as a delimiter.

        Returns
        -------
        None.
        
        '''        
        for i in range(len(self.sets_info)):
            if self.sets_info[i]['filename']==xyz_file:
                energies = np.array(np.genfromtxt(energy_file, delimiter='\n'))  
                for j in range(self.sets_info[i]['n_molecules']):
                    self.trajectory[j+self.sets_info[i]['beginning']].energy = energies[j]    
        
        
    def save_descriptors(self, path='.', name='dataset'):        
        os.mkdir(path+'/'+name)
        
        for i in range(self.n_molecules):
            np.savetxt(path+'/'+name+'/d_'+str(i)+'.csv', np.array(self.trajectory[i].descriptors).transpose(), delimiter=",")
        
    def load_descriptors(self, path='./dataset'):
        for i in range(self.n_molecules):
            self.trajectory[i].descriptors = np.genfromtxt(path+'/d_'+str(i)+'.csv', delimiter=',').transpose()
            
    def remove_descriptors(self):
        for i in range(self.n_molecules):
            self.trajectory[i].descriptors = [[] for i in range(self.trajectory[i].n_atoms)]
