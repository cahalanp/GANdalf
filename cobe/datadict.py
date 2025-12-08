                 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
This module defines the Datadict class

@author: matteo
'''
import numpy as np
from cobe import Dataset
from random import shuffle
     
class Datadict():
      
    def __init__(self, dataset=None):
        
        self.data = {'n_molecules':None,
                     'all_types':None,  
                     'descriptors':[],
                     'descriptors_d':[],
                     'energies':[],
                     'forces':[],
                     'types':[],
                     'n_descriptors':None,
                     'scaling':None
                     }  
        
        if(dataset!=None):
        
            self.data['n_molecules'] = dataset.n_molecules
            self.data['all_types'] = dataset.types
            
            for n in range(self.data['n_molecules']):
                self.data['descriptors'].append(dataset.trajectory[n].descriptors)
                self.data['descriptors_d'].append(dataset.trajectory[n].descriptors_d)            
                self.data['energies'].append(dataset.trajectory[n].energy)
                self.data['forces'].append(dataset.trajectory[n].forces)
                self.data['types'].append(dataset.trajectory[n].types)    
            
            self.data['n_descriptors'] = len(self.data['descriptors'][0][0])               
           
    def add(self, datadict):
        
        for n in range(self.data['n_molecules']):
            for i in range(len(self.data['descriptors'][n])):
                self.data['descriptors'][n][i].extend(datadict.data['descriptors'][n][i])
                for j in range(len(self.data['descriptors'][n])):
                    for k in range(3):
                        self.data['descriptors_d'][n][i][j][k].extend(datadict.data['descriptors_d'][n][i][j][k])
        
        self.data['n_descriptors'] = len(self.data['descriptors'][0][0]) 
        
    def split(self, fraction=20, mix=True, val=False, seed=None):
        train = Datadict()
        test = Datadict()
        
        if (seed!=None):
            np.random.seed(seed)
         
        new_order = np.arange(0, self.data['n_molecules'])
        shuffle(new_order)
        
        n_fraction = int(self.data['n_molecules']*fraction/100)        
        
        if(val==False):
            train.data['n_molecules'] = self.data['n_molecules']-n_fraction
            train.data['all_types'] = self.data['all_types']
            train.data['n_descriptors'] = self.data['n_descriptors']
            
            test.data['n_molecules'] = n_fraction
            test.data['all_types'] = self.data['all_types']
            test.data['n_descriptors'] = self.data['n_descriptors']            
            
            for i in range(train.data['n_molecules']):
                train.data['descriptors'].append(self.data['descriptors'][new_order[i]])
                train.data['descriptors_d'].append(self.data['descriptors_d'][new_order[i]])         
                train.data['energies'].append(self.data['energies'][new_order[i]]) 
                train.data['forces'].append(self.data['forces'][new_order[i]]) 
                train.data['types'].append(self.data['types'][new_order[i]])   
                
            for i in range(train.data['n_molecules'],test.data['n_molecules']+train.data['n_molecules']):
                test.data['descriptors'].append(self.data['descriptors'][new_order[i]])
                test.data['descriptors_d'].append(self.data['descriptors_d'][new_order[i]])         
                test.data['energies'].append(self.data['energies'][new_order[i]]) 
                test.data['forces'].append(self.data['forces'][new_order[i]]) 
                test.data['types'].append(self.data['types'][new_order[i]])
                
            return train, test
            
        else:
            val = Datadict()
            
            train.data['n_molecules'] = self.data['n_molecules']-2*n_fraction
            train.data['all_types'] = self.data['all_types']
            train.data['n_descriptors'] = self.data['n_descriptors']
            
            val.data['n_molecules'] = n_fraction
            val.data['all_types'] = self.data['all_types']
            val.data['n_descriptors'] = self.data['n_descriptors']
            
            test.data['n_molecules'] = n_fraction
            test.data['all_types'] = self.data['all_types']
            test.data['n_descriptors'] = self.data['n_descriptors']            
            
            for i in range(train.data['n_molecules']):
                train.data['descriptors'].append(self.data['descriptors'][new_order[i]])
                train.data['descriptors_d'].append(self.data['descriptors_d'][new_order[i]])         
                train.data['energies'].append(self.data['energies'][new_order[i]]) 
                train.data['forces'].append(self.data['forces'][new_order[i]]) 
                train.data['types'].append(self.data['types'][new_order[i]]) 
                
            for i in range(train.data['n_molecules'],val.data['n_molecules']+train.data['n_molecules']):
                val.data['descriptors'].append(self.data['descriptors'][new_order[i]])
                val.data['descriptors_d'].append(self.data['descriptors_d'][new_order[i]])         
                val.data['energies'].append(self.data['energies'][new_order[i]]) 
                val.data['forces'].append(self.data['forces'][new_order[i]]) 
                val.data['types'].append(self.data['types'][new_order[i]])
                
            for i in range(val.data['n_molecules']+train.data['n_molecules'],test.data['n_molecules']+val.data['n_molecules']+train.data['n_molecules']):
                test.data['descriptors'].append(self.data['descriptors'][new_order[i]])
                test.data['descriptors_d'].append(self.data['descriptors_d'][new_order[i]])         
                test.data['energies'].append(self.data['energies'][new_order[i]]) 
                test.data['forces'].append(self.data['forces'][new_order[i]]) 
                test.data['types'].append(self.data['types'][new_order[i]])
                
            return train, val, test