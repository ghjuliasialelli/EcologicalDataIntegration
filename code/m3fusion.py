#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 22:17:08 2021

@author: ghjuliasialelli

M3Fusion (tf) implementation attempt

Ref:
    - https://github.com/kkgadiraju/M3Fusion/blob/main/train.py
    - https://gdal.org/programs/gdal_merge.html
    
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, GRU, TimeDistributed, Dense, Permute, Dot, Lambda, Conv2D, BatchNormalization, MaxPool2D, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from data_gen import DataGenerator


"""
    RNN component : UNUSED
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
    pool = MaxPool2D(pool_size = (2,2), strides = 2)(pool)
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
# TO DO : need to de-normalize the output no? (will be easy, only ACD)

"""
    MERGING.
"""

LEARNING_RATE = 0.0001
WEIGHTS = [0.3, 0.3, 0.3, 1] 
BS = 10
EPOCHS = 100

l8_input = Input(batch_shape = (BS, 5, 5, 4), name = 'l8_input')
l8_feat, l8_preds = build_5x5_cnn_model(l8_input, 'l8')

s2_input = Input(batch_shape = (BS, 15, 15, 4), name = 's2_input')
s2_feat, s2_preds = build_15x15_cnn_model(s2_input, 's2')

nicfi_input = Input(batch_shape = (BS, 30, 30, 4), name = 'nicfi_input')
nicfi_feat, nicfi_preds = build_30x30_cnn_model(nicfi_input, 'nicfi')

concatenated = Concatenate(axis = -1)([l8_feat, s2_feat, nicfi_feat])
concat_preds = Dense(1, activation = 'linear', name = 'concat_preds')(concatenated)

model_inputs = [l8_input, s2_input, nicfi_input]
model_outputs = [s2_preds, l8_preds, nicfi_preds, concat_preds]
model_losses = {pred:'mse' for pred in ['s2_preds', 'l8_preds', 'nicfi_preds', 'concat_preds']}

model = Model(inputs = model_inputs, outputs = model_outputs)
model.summary()

model.compile(loss = model_losses, optimizer = Adam(lr = LEARNING_RATE), loss_weights = WEIGHTS)
train_data = DataGenerator(num_samples = 1000, batch_size = BS)
history = model.fit(train_data, batch_size = BS, epochs = 2, verbose = 1) 
# WARNING:tensorflow:Gradients do not exist for variables ['concat_preds/kernel:0', 'concat_preds/bias:0'] when minimizing the loss.

