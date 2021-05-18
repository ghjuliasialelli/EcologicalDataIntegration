#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 22:17:08 2021

@author: ghjuliasialelli

M3Fusion (tf) implementation attempt

Ref:
    - https://github.com/kkgadiraju/M3Fusion/blob/main/train.py
    - https://gdal.org/programs/gdal_merge.html
    
Main packages : 
    python==3.7.10
    tensorflow==2.4.1
    numpy==1.19.5
    rasterio==1.2.2
    
Commands for Leohnard cluster execution : 
    source $HOME/miniconda3/bin/activate
    conda create -n semproj python=3.7.10
    conda activate semproj
    pip3 install -r requirements.txt
    bsub -W 1:00 -R "rusage[ngpus_excl_p=1]" -R "rusage[mem=2048]" -R "rusage[scratch=5000]" python3 m3fusion.py -bs 128 -ns 1.0 -ep 100 -lr 0.0001 -s2 0.3 -l8 0.3 -ni 0.3
"""

from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization, MaxPool2D, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from data_gen import DataGenerator
import pickle
import argparse
from os.path import isdir
from os import mkdir

######################################################################################################
#  CNN component
######################################################################################################

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

######################################################################################################
# Command line parsing
# -bs 128 -ns 1.0 -ep 100 -lr 0.0001 -s2 0.3 -l8 0.3 -ni 0.3
######################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("-bs",  type = int,   help = "batch size",                      default = 128)
parser.add_argument("-ns",  type = float, help = "% of number of samples to use",   default = 1.0)
parser.add_argument("-ep",  type = int,   help = "epochs",                          default = 100)
parser.add_argument("-lr",  type = float, help = "learning reate",                  default = 0.0001)
parser.add_argument("-s2",  type = float, help = "s2 weight",                       default = 0.3)
parser.add_argument("-l8",  type = float, help = "l8 weight",                       default = 0.3)
parser.add_argument("-ni",  type = float, help = "nicfi weight",                    default = 0.3)
args = parser.parse_args()

LEARNING_RATE = args.lr
WEIGHTS = [args.s2, args.l8, args.ni, 1] 
BS = args.bs
EPOCHS = args.ep
NUM_SAMPLES = int(args.ns * 1000000)


######################################################################################################
# Model building
######################################################################################################

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
train_data = DataGenerator(num_samples = NUM_SAMPLES, batch_size = BS)
history = model.fit(train_data, batch_size = BS, epochs = EPOCHS, verbose = 1) 

# Construct experiment name
if not isdir('experiments'): mkdir('experiments')
fn = '-'.join(list(map(lambda x: ':'.join([x[0], str(x[1])]), list(args.dict.items()))))
f = open('experiments/{}.pkl'.format(fn), 'wb')
pickle.dump(history.history, f)
f.close()

