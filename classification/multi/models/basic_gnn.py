# -*- coding: utf-8 -*-
"""
Created on 2023-07-12 (Wed) 22:04:36

GNN model

@author: I.Azuma
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_max
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GCNConv

class Net(torch.nn.Module):
    def __init__(self,hidden_channels,n_classes):
        super(Net, self).__init__()
        self.conv1 = GCNConv(6, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.linear1 = torch.nn.Linear(hidden_channels,n_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        
        # 2. Readout layer
        x, _ = scatter_max(x, data.batch, dim=0)
        #x = scatter_mean(x, data.batch, dim=0)

        # 3. Apply a final classifier
        x = self.linear1(x)
        return x

model = Net(hidden_channels=64,n_classes=7)

