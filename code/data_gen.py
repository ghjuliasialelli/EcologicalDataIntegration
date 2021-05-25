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
import rasterio as rs
from rasterio.windows import Window

offsets = {'ACD': 5,
 'L8': 5,
 'S2': 15,
 'NICFI' : 30}

class DataGenerator(Sequence):

    def __init__(self, samples, batch_size, shuffle = True):
        """
        Initializes a data generator object
          :param batch_size: The size of each batch returned by __getitem__
        """
        self.indices = samples
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.offsets = [5, 5, 15, 30]
        self.shapes = [5000, 5000, 15000, 30000]
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle : np.random.shuffle(self.indices)
    
    def __len__(self):
        """ Number of batches. """ 
        return len(self.indices) // self.batch_size
    
    def __getitem__(self, idx):
        """
        Fetch the next batch. 
            :param idx: batch's index, as given by the __len__ method.
        """
        indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        assert (len(indices) == self.batch_size)
        
        batch_y = np.empty((self.batch_size, 5, 5, 1))
        batch_l8 = np.empty((self.batch_size, 5, 5, 4))
        batch_s2 = np.empty((self.batch_size, 15, 15, 4))
        batch_nicfi = np.empty((self.batch_size, 30, 30, 4))
        batches = [batch_y, batch_l8, batch_s2, batch_nicfi]
                
        for d, data_path in enumerate(['ACD_patch.tiff', 'L8_patch.tiff', 'S2_patch.tiff','NICFI_patch.tiff']) : 
            with rs.open(data_path) as data : 
                data_offset = self.offsets[d]                
                for b, idx in enumerate(indices) :
                    i = idx // 1000
                    j = idx % 1000
                    window = Window.from_slices((i, i + data_offset), (j, j + data_offset))
                    patch = data.read(window = window).reshape((data_offset, data_offset, data.count))    
                    batches[d][b] = patch
        
        x = {'l8_input' : batch_l8, 's2_input' : batch_s2, 'nicfi_input' : batch_nicfi}
        y = {'l8_preds' : batch_y, 's2_preds' : batch_y, 'nicfi_preds' : batch_y, 'concat_preds' : batch_y}
        
        return (x, y)

    
    
    