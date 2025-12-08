 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
This module defines the Molecule object

@author: matteo
'''
import numpy as np

from ase import Atoms

class Molecule(Atoms): 
    '''
    
    Molecule object
    ---------------
    The Molecule object represents an isolated molecule.
    
    This class is inherited from the Atoms object of ase.
    
    Parameters (in addition to the Atoms class):
    ----------    
    descriptors : list of descriptors
        List with n_atoms elements, the i-element of the list 
        is a list of descriptors associated to the i-atom.
    energy : float 
        Energy of the molecule.
    masses : list of float
        List with n_atoms elements, the i-element correspond to the mass 
        of the i-atom
    n_atoms : int
        Number of atoms composing the molecule.
    n_species : int
        Number of different species present in the molecule.
    n_labels : int
        Number of different labels present in the molecule.
    n_types : int
        Number of different labels present in the molecule.
        
    special : dict of str (to do)
        Dictionary containing the correspondence label-species        
    
    types : list of int
        List with n_atoms elements, the i-element correspond to the 
        type of the i-atom. 
        
    
    distances :
    angles :
        .
        .
        .  
        
    Methods
    -------  
        
    '''

    def __init__(self, symbols=None,
                 positions=None, numbers=None,
                 tags=None, momenta=None, masses=None,
                 magmoms=None, charges=None,
                 scaled_positions=None,
                 cell=None, pbc=None, celldisp=None,
                 constraint=None,
                 calculator=None,
                 info=None,
                 velocities=None,
                 ):
                   
        Atoms.__init__(self, symbols,
                 positions, numbers,
                 tags, momenta, masses,
                 magmoms, charges,
                 scaled_positions,
                 cell, pbc, celldisp,
                 constraint,
                 calculator,
                 info,
                 velocities)    
        
        self.descriptors = [[] for i in range(len(self))]
                
        self.err = None
        try:
            self.energy = self.info['Energy']
        except ValueError:
            self.energy = None
        except KeyError:
            self.energy = None
        self.masses = None
        self.n_atoms = len(self)
        self.n_labels = len(set(self.get_chemical_symbols()))         
        self.types = self.get_atomic_numbers()
        self.n_types = len(set(self.types))
        try:
            self.forces = self.arrays['momenta']
        except ValueError:
            self.forces = None
        except KeyError:
            self.forces = None
            
        self.descriptors_d = [[[[] for k in range(3)] for j in range(self.n_atoms)] for i in range(self.n_atoms)]
        
        
        
       
