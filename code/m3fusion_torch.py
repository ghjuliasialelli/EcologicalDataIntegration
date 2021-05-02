#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 08:33:04 2021

@author: ghjuliasialelli

M3Fusion (torch) implementation attempt

Ref:
    - https://kushalj001.github.io/black-box-ml/lstm/pytorch/torchtext/nlp/sentiment-analysis/2020/01/10/Building-Sequential-Models-In-PyTorch.html
    
"""

from torch import nn 


def __init__(self, input_size, hidden_size, embedding_dim, dropout, num_layers, output_dim):
    
    super().__init__()
    
    self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=embedding_dim)
    
    self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers,
                        batch_first=True)
    
    self.dropout = nn.Dropout(p=dropout)
    
    self.linear = nn.Linear(in_features=hidden_size, out_features=output_dim)
    

def forward(self, x):
    
    # x = [batch_size, seq_len] = [64, seq_len] as seq_len depends on the batch. 
    
    embed = self.embedding(x)
    
    # embed = [batch_size, seq_len, embedding_dim] = [64, seq_len, 100]
    
    # These can be intuitively interpreted as: each example in the batch 
    # has a length of seq_len and each word in the sequence is represented
    # by a vector of size 100.
    
    output, (hidden, cell) = self.lstm(embed)
    
    # output = [batch_size, seq_len, hidden_size] = [64, seq_len, 128]
    # hidden = [num_layers*num_directions, batch_size, hidden_size] = [1, 64, 128]
    # cell = [num_layers*num_directions, batch_size, hidden_size] = [1, 64, 128]
    
    # output is the concatenation of the hidden state from every time step, 
    # whereas hidden is simply the final hidden state. 
    # We verify this using the assert statement. 
    
    output = output.permute(1,0,2)
    
    # hidden = [1, 64, 128]
    # output = [seq_len, 64, 128]
    
    assert torch.equal(output[-1,:,:], hidden.squeeze(0))
 
    preds = self.linear(output[-1,:,:])
    
    # preds = [64, 1]
    
    return preds
    