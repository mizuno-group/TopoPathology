# -*- coding: utf-8 -*-
"""
Created on 2023-07-18 (Tue) 20:23:00

PNA (Principle Neighborhood Aggregation)

@author: I.Azuma
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding, Linear, ModuleList, ReLU, Sequential, Dropout

from torch_geometric.nn import BatchNorm, PNAConv, global_add_pool
from torch_geometric.nn import GCNConv
from torch_geometric.utils import degree


class Flatten(torch.nn.Module):
    def forward(self, input):
        return input.view(input.size(0))

class PNA(torch.nn.Module):
    def __init__(self, deg, num_class):
        super(PNA, self).__init__()

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']
        
        self.conv1 = GCNConv(6, 75)

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(5):
            conv = PNAConv(in_channels=75, out_channels=75,
                           aggregators=aggregators, scalers=scalers, deg=deg,
                           towers=1, pre_layers=1, post_layers=1,
                           divide_input=False)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(75))

        self.mlp = Sequential(Linear(75, 50), 
                              ReLU(), 
                              Linear(50, 25), 
                              ReLU(), 
                              Dropout(p=0.2),
                              Linear(25, num_class))
        
        #self.flatten = Flatten()

    def forward(self, data):
        # gputil_usage()
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.conv1(x, edge_index)
           
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index)))

        x = global_add_pool(x, batch)
        x = self.mlp(x)
        #x = self.flatten(x)

        return x