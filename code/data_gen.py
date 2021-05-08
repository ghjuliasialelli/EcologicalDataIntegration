#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 15:39:10 2021

@author: ghjuliasialelli

Data generator.

Ref: 
    - https://mahmoudyusof.github.io/facial-keypoint-detection/data-generator/
    - https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence
"""

import numpy as np
from tensorflow.keras.utils import Sequence

offsets = {'ACD': 5,
 'L8': 5,
 'S2': 15}

class DataGenerator(Sequence):

    def __init__(self, num_samples=1000000, shuffle=False, batch_size=100):
        """
        Initializes a data generator object
          :param file_name: image.npy
          :param shuffle: shuffle the data after each epoch
          :param batch_size: The size of each batch returned by __getitem__
        """
        self.num_samples = num_samples
        self.ACD = np.load('ACD_patch.npy')
        self.l8 = np.load('L8_patch.npy')
        self.s2 = np.load('S2_patch.npy')
        
        self.l8_offset = offsets['L8']
        self.l8_num_sq = self.l8.shape[1] // self.l8_offset
        
        self.s2_offset = offsets['S2']
        self.s2_num_sq = self.s2.shape[1] // self.s2_offset
        
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        """ Number of batches. """ 
        return self.num_samples // self.batch_size
    
    def __getitem__(self, idx):
        
        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_l8 = np.empty((self.batch_size, self.l8_offset, self.l8_offset, 4))
        batch_s2 = np.empty((self.batch_size, self.s2_offset, self.s2_offset, 4))
        batch_y = np.empty((self.batch_size, 5, 5, 1))
        
        for idx in indices :
            # l8 
            i = idx // self.l8_num_sq
            j = idx % self.l8_num_sq
            X_l8 = self.l8[:, i:i+self.l8_offset, j:j+self.l8_offset]
            X_l8 = X_l8.reshape((self.l8_offset, self.l8_offset, X_l8.shape[0]))
            batch_l8[i,] = X_l8
            
            # s2
            i = idx // self.s2_num_sq
            j = idx % self.s2_num_sq
            X_s2 = self.s2[:, i:i+self.s2_offset, j:j+self.s2_offset]
            X_s2 = X_s2.reshape((self.s2_offset, self.s2_offset, X_s2.shape[0]))
            batch_s2[i,] = X_s2
            
            # ACD
            Y = self.ACD[i:i+offsets['ACD'], j:j+offsets['ACD']]
            Y = Y.reshape((5, 5, 1))
            batch_y[i,] = Y
        
        x = {'l8_input' : batch_l8, 's2_input' : batch_s2}
        y = {'l8_preds' : batch_y, 's2_preds' : batch_y, 'concat_preds' : batch_y}
        
        
        X = []
        Y = []
        for l8,s2,y in zip(batch_l8, batch_s2, batch_y):
            X.append({'l8_input' : l8, 's2_input' : s2})
            Y.append({'l8_preds' : y, 's2_preds' : y, 'concat_preds' : y})
        
        
        #return (x, y)
        return (X,Y)
    
# train_gen = DataGenerator(...)

    
    
    