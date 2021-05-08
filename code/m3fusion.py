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
from tensorflow.keras.layers import Input, GRU, TimeDistributed, Dense, Permute, Dot, Lambda, Conv2D, BatchNormalization, MaxPool2D, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from rasterio.windows import Window
import rasterio as rs


"""
    RNN component
"""
rnn_input = Input(batch_shape = (None, 16, 1), name = 'rnn_input')
def build_rnn_model(rnn_input):
    gru_sq = GRU(units = 1024, return_sequences = True, name = 'GRU')(rnn_input)
    v_a = TimeDistributed(Dense(units = 1024, activation = 'tanh', name = 'v_a'))(gru_sq)
    lambda_a = TimeDistributed(Dense(units = 1, activation = 'softmax', name = 'lambda_a'))(v_a)
    lambda_a_reshape = Permute(dims = (2,1))(lambda_a)
    rnn_dot = Dot(axes = (2, 1))([lambda_a_reshape, gru_sq])
    rnn_feat = Lambda(lambda y: tf.squeeze(y, axis = 1))(rnn_dot)
    rnn_preds = Dense(units = 1, activation = 'linear', name = 'rnn_preds')(rnn_feat)
    return rnn_feat, rnn_preds
rnn_feat, rnn_preds = build_rnn_model(rnn_input)

"""
    CNN component
"""

def convolutional_block(input_layer, FILTERS, KERNEL_SIZE, PADDING = 'valid'):
    conv = Conv2D(filters = FILTERS, kernel_size = KERNEL_SIZE, padding = PADDING, activation = 'relu')(input_layer)
    bn = BatchNormalization(axis = -1)(conv)
    return bn

def build_30x30_cnn_model(cnn_input, model_name):
    conv1 = convolutional_block(cnn_input, 32, (1,1))
    pool = MaxPool2D(pool_size = (2,2), strides = 2)(conv1)
    conv2 = convolutional_block(pool, 64, (3,3))
    conv3 = convolutional_block(conv2, 64, (3,3), PADDING = 'same')
    conv_concat = Concatenate(axis = -1)([conv2, conv3])
    cnn_feat = convolutional_block(conv_concat, 32, (1,1))
    cnn_preds = Dense(1, activation = 'linear', name = model_name+'_preds')(cnn_feat)
    return cnn_feat, cnn_preds

def build_15x15_cnn_model(cnn_input, model_name):
    conv1 = convolutional_block(cnn_input, 32, (1,1))
    pool = MaxPool2D(pool_size = (2,2), strides = 2)(conv1)
    conv2 = convolutional_block(pool, 64, (3,3))
    conv3 = convolutional_block(conv2, 64, (3,3), PADDING = 'same')
    conv_concat = Concatenate(axis = -1)([conv2, conv3])
    cnn_feat = convolutional_block(conv_concat, 32, (1,1))
    cnn_preds = Dense(1, activation = 'linear', name = model_name+'_preds')(cnn_feat)
    return cnn_feat, cnn_preds


def build_5x5_cnn_model(cnn_input, model_name):
    conv1 = convolutional_block(cnn_input, 32, (1,1))
    conv2 = convolutional_block(conv1, 64, (1,1))
    conv3 = convolutional_block(conv2, 64, (1,1), PADDING = 'same')
    conv_concat = Concatenate(axis = -1)([conv2, conv3])
    cnn_feat = convolutional_block(conv_concat, 32, (1,1))
    cnn_preds = Dense(1, activation = 'linear', name = model_name+'_preds')(cnn_feat)
    return cnn_feat, cnn_preds

# TO DO : linear or sigmoid activation function? Since the input is in [0,1]
# TO DO : need to de-normalize the output no?

"""
    MERGING.
"""

LEARNING_RATE = 0.0001
WEIGHTS = [0.3, 0.3, 1] 
BS = 10
EPOCHS = 100

l8_input = Input(batch_shape = (BS, 5, 5, 4), name = 'l8_input')
l8_feat, l8_preds = build_5x5_cnn_model(l8_input, 'l8')

s2_input = Input(batch_shape = (BS, 15, 15, 4), name = 's2_input')
s2_feat, s2_preds = build_15x15_cnn_model(s2_input, 's2')

nicfi_input = Input(batch_shape = (BS, 30, 30, 4), name = 'nicfi_input')
nicfi_feat, nicfi_preds = build_30x30_cnn_model(s2_input, 'nicfi')

concatenated = Concatenate()([l8_feat, s2_feat, nicfi_feat])
concat_preds = Dense(1, activation = 'linear', name = 'concat_preds')(concatenated)

model_inputs = [l8_input, s2_input, nicfi_input]
model_outputs = [s2_preds, l8_preds, nicfi_preds, concat_preds]
model_losses = {pred:'mse' for pred in ['s2_preds', 'l8_preds', 'nicfi_preds', 'concat_preds']}

model = Model(inputs = model_inputs, outputs = model_outputs)
model.summary()
model.compile(loss = model_losses, optimizer = Adam(lr = LEARNING_RATE), loss_weights = WEIGHTS)

l8 = np.load('L8_patch.npy')
s2 = np.load('S2_patch.npy')
ACD = np.load('ACD_patch.npy')
ACD = ACD.reshape(1, ACD.shape[1], ACD.shape[1])
NUM_SAMPLES = 1000

def generate_data(num_samples, data, offsets = [5, 5, 15, 30]):
    indices = np.random.choice(1000000, num_samples, replace = False)
    patched_dataset = []
    
    # Iteration for ACD, L8, and S2
    for i, data in enumerate(data) : 
        data_offset = offsets[i]
        data_num_sq = data.shape[1] // data_offset
        patch_data = []
        
        for idx in indices :
            i = idx // data_num_sq
            j = idx % data_num_sq
            patch = data[:, i : i + data_offset, j : j + data_offset].reshape((data_offset, data_offset, data.shape[0]))
            patch_data.append(patch)
            
        patched_dataset.append(patch_data)
    
    # For NICFI 
    with rs.open('NICFI_patch.tiff') as nicfi :
        data_offset = offsets[-1]
        data_num_sq = nicfi.height // data_offset
        patch_data = []
        
        for idx in indices :
            i = idx // data_num_sq
            j = idx % data_num_sq
            window = Window.from_slices((i, i + data_offset), (j, j + data_offset))
            patch = nicfi.read(window = window).reshape((data_offset, data_offset, nicfi.count))            
            patch_data.append(patch)
            
        patched_dataset.append(patch_data)
    
            
    batch_y = np.array(patched_dataset[0])
    x = {k: v for k,v in zip(['l8_input', 's2_input', 'nicfi_input'], patched_dataset[1:])}
    y = {k: batch_y for k in ['l8_preds', 's2_preds', 'nicfi_preds']}
    
    return x,y 


data = [ACD, l8, s2] 
x, y = generate_data(NUM_SAMPLES, data)

history = model.fit(x, y, validation_split = 0.2, batch_size = BS, epochs = EPOCHS, verbose = 1) 















def manual_data_gen(indices):
    l8_offset = 5
    s2_offset = 15
    acd_offset = 5
    l8_num_sq = l8.shape[1] // l8_offset
    s2_num_sq = s2.shape[1] // s2_offset
    
    batch_l8 = []
    batch_s2 = []
    batch_y = []
    
    for idx in indices :
        i = idx // l8_num_sq
        j = idx % l8_num_sq
        X_l8 = l8[:, i:i+l8_offset, j:j+l8_offset]
        X_l8 = X_l8.reshape((l8_offset, l8_offset, X_l8.shape[0]))
        batch_l8.append(X_l8)
        
        # s2
        i = idx // s2_num_sq
        j = idx % s2_num_sq
        X_s2 = s2[:, i:i+s2_offset, j:j+s2_offset]
        X_s2 = X_s2.reshape((s2_offset, s2_offset, X_s2.shape[0]))
        batch_s2.append(X_s2)
        
        # ACD
        Y = ACD[i:i+acd_offset, j:j+acd_offset]
        Y = Y.reshape((5, 5, 1))
        batch_y.append(Y)
    
    batch_y = np.array(batch_y)
    x = {'l8_input' : np.array(batch_l8), 's2_input' : np.array(batch_s2)}
    y = {'l8_preds' : batch_y, 's2_preds' : batch_y, 'concat_preds' : batch_y}
    return x,y 

def baby_model(cnn_input, rnn_input, cnn_preds, rnn_preds, concat_preds):
    model = Model(inputs = [cnn_input, rnn_input], outputs=[cnn_preds, rnn_preds, concat_preds])
    model.compile(loss = {'cnn_preds': 'mae', 'rnn_preds': 'mae', 'concat_preds': 'mae'}, optimizer = Adam(lr = LEARNING_RATE), loss_weights = WEIGHTS, metrics=['accuracy']) 
    
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
    y = {'cnn_preds' : y_cnn, 'rnn_preds' : y_rnn, 'concat_preds' : y_concat}
    
    # generate a model by training 
    history = model.fit(x, y, batch_size = BS, epochs = EPOCHS, verbose = 1) 

    return model, history




