#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 22:17:08 2021

@author: ghjuliasialelli

M3Fusion (tf) implementation attempt

Ref:
    - https://github.com/kkgadiraju/M3Fusion/blob/main/train.py
    
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, GRU, TimeDistributed, Dense, Permute, Dot, Lambda, Conv2D, BatchNormalization, MaxPool2D, Concatenate, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model


"""
    RNN component
    For high-spatial resolution (LANDSAT-8) data integration.
"""

rnn_input = Input(batch_shape = (None, 16, 1), name = 'rnn_input')
#   input_shape : shape tuple, not including the batch axis

def build_rnn_model(rnn_input):
    gru_sq = GRU(units = 1024, return_sequences = True, name = 'GRU')(rnn_input)
    #   units : positive integer, dimensionality of the output space
    #   activation : activation function (default to tanh)
    #   recurrent_activation : activation function to use for the recurrent step (default to sigmoid)
    #   return_sequences : return the full sequence instead of the last output in the output sequence
    
    v_a = TimeDistributed(Dense(units = 1024, activation = 'tanh', name = 'v_a'))(gru_sq)
    #   The input should be at least 3D, and the dimension of index one will be considered to be the temporal dimension.
    #   units : positive integer, dimensionality of the output space
    
    lambda_a = TimeDistributed(Dense(units = 1, activation = 'softmax', name = 'lambda_a'))(v_a)
    #   units : positive integer, dimensionality of the output space
    
    lambda_a_reshape = Permute(dims = (2,1))(lambda_a)
    #   dims : tuple of integers ; (2,1) permutes the first and second dimension of the input
    
    rnn_dot = Dot(axes = (2, 1))([lambda_a_reshape, gru_sq])
    #   axes : two integers corresponding to the desired axis from the first input and the desired axis from the second input, respectively
    
    rnn_feat = Lambda(lambda y: tf.squeeze(y, axis = 1))(rnn_dot)
    #   axis : selects a subset of the entries of length one in the shape
    
    rnn_preds = Dense(units = 1, activation = 'linear', name = 'rnn_preds')(rnn_feat)
    #   units : positive integer, dimensionality of the output space
    
    return rnn_feat, rnn_preds

rnn_feat, rnn_preds = build_rnn_model(rnn_input)

"""
    CNN component
    For very high-spatial resolution (Planet) data integration.
"""

def convolutional_block(input_layer, FILTERS, KERNEL_SIZE, PADDING = 'valid'):
    conv = Conv2D(filters = FILTERS, kernel_size = KERNEL_SIZE, padding = PADDING, activation = 'relu')(input_layer)
    #   filters : integer, the number of output filters in the convolution
    #   kernel_size : tuple/list of 2 integers, specifying the height and width of the 2D convolution window
    #   activation : if not specified, no activation is applied
    bn = BatchNormalization(axis = -1)(conv)
    #   axis : the axis that should be normalized
    return bn

cnn_input = Input(batch_shape = (None, 25, 25, 5), name = 'cnn_input')

def build_cnn_model(cnn_input):
    conv1 = convolutional_block(cnn_input, 256, (7,7))
    # output shape : ()
    
    pool = MaxPool2D(pool_size = (2,2), strides = 2)(conv1)
    #   pool_size : tuple of 2 integers, window size over which to take the maximum
    #   strides : tuple of 2 integers, or None. Specifies how far the pooling window moves for each pooling step. If None, it will default to pool_size.
    
    conv2 = convolutional_block(pool, 512, (3,3))
    
    conv3 = convolutional_block(conv2, 512, (3,3), PADDING = 'same')
    
    conv_concat = Concatenate(axis = -1)([conv2, conv3])
    
    conv4 = convolutional_block(conv_concat, 512, (1,1))
    
    cnn_feat = GlobalAveragePooling2D()(conv4)
    
    cnn_preds = Dense(1, activation = 'linear', name = 'cnn_preds')(cnn_feat)
    
    return cnn_feat, cnn_preds

cnn_feat, cnn_preds = build_cnn_model(cnn_input)


"""
    MERGING THE TWO.
"""

LEARNING_RATE = 0.1
WEIGHTS = [0.3, 0.3, 1] 
BS = 2
EPOCHS = 1
#train_generator = ... 
#val_generator = ...

#   A common pattern is to pass a tf.data.Dataset, generator, or tf.keras.utils.Sequence 
#   to the x argument of fit, which will in fact yield not only features (x) but optionally 
#   targets (y) and sample weights. Keras requires that the output of such iterator-likes be 
#   unambiguous. The iterator should return a tuple of length 1, 2, or 3, where the optional 
#   second and third elements will be used for y and sample_weight respectively.

concatenated = Concatenate()([cnn_feat, rnn_feat])
concat_preds = Dense(1, activation = 'linear', name = 'concat_preds')(concatenated)
model = Model(inputs = [cnn_input, rnn_input], outputs=[cnn_preds, rnn_preds, concat_preds])
model.compile(loss = {'cnn_preds': 'mae', 'rnn_preds': 'mae', 'concat_preds': 'mae'}, optimizer = Adam(lr = LEARNING_RATE), loss_weights = WEIGHTS, metrics=['accuracy']) 


import rasterio as rs

fp = r'quads/1681-1058.tiff'
img = rs.open(fp)
bands, wid, hei = img.count, img.width, img.height 
img_arr = img.read().reshape(wid, hei, bands)[:25, :25, :]
imgs = np.array([img_arr, img_arr, img_arr, img_arr, img_arr, img_arr])
time_series = np.random.rand(6, 16,1)
res = np.random.rand(6, 1)

x_cnn = imgs
x_rnn = time_series
y_cnn = y_rnn = y_concat = res

x = {'cnn_input' : x_cnn, 'rnn_input' : x_rnn}
y = {'cnn_preds' : res, 'rnn_preds' : res, 'concat_preds' : res}

# generate a model by training 
history = model.fit(x,  
                    y,
                    batch_size = BS,
            		epochs = EPOCHS,
            		verbose = 1) 


