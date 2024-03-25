# -*- coding: utf-8 -*-
"""
Created on 2023-07-18 (Tue) 11:57:52

GraphSAGE

@author: I.Azuma
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_max
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import SAGEConv


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_classes, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, out_dim)
        self.linear1 = torch.nn.Linear(out_dim,n_classes)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(data.x, edge_index)
        x = F.relu(x)
        #x = F.dropout(x, p=self.dropout)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        #x = F.dropout(x, p=self.dropout)
        
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        #x = F.dropout(x, p=self.dropout)

        # readout layer
        #x, _ = scatter_max(x, data.batch, dim=0)
        x = scatter_mean(x, data.batch, dim=0)

        # final classifier
        x = self.linear1(x)
        return x
