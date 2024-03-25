# -*- coding: utf-8 -*-
"""
Created on 2023-07-18 (Tue) 15:58:13

GIN (graph isomorphism network) Topk

reference:
https://www.kaggle.com/code/peterralbovsky1/pytorch-geometric-cifar-exploration/notebook

@author: I.Azuma
"""
import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GraphConv, GINConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

class GINTopK(torch.nn.Module):
    def __init__(self, num_feature, nhid, num_class):
        super(GINTopK, self).__init__()
        self.conv1 = GINConv(Seq(Lin(num_feature, nhid), ReLU(), Lin(nhid, nhid)))
        self.pool1 = TopKPooling(nhid, ratio=0.8)
        self.conv2 = GINConv(Seq(Lin(nhid, nhid), ReLU(), Lin(nhid, nhid)))
        self.pool2 = TopKPooling(nhid, ratio=0.8)
        self.conv3 = GINConv(Seq(Lin(nhid, nhid), ReLU(), Lin(nhid, nhid)))
        self.pool3 = TopKPooling(nhid, ratio=0.8)
        self.conv4 = GINConv(Seq(Lin(nhid, nhid), ReLU(), Lin(nhid, nhid)))
        self.pool4 = TopKPooling(nhid, ratio=0.8)

        self.lin1 = torch.nn.Linear(2*nhid, nhid)
        self.lin2 = torch.nn.Linear(nhid, nhid//2)
        self.lin3 = torch.nn.Linear(nhid//2, num_class)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = F.relu(self.conv1(x, edge_index, ))
        x, edge_index, edge_attr, batch, _, _ = self.pool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, edge_attr, batch, _, _ = self.pool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, edge_attr, batch, _, _ = self.pool3(x, edge_index, edge_attr, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        
        x = F.relu(self.conv4(x, edge_index))
        x, edge_index, edge_attr, batch, _, _ = self.pool4(x, edge_index, edge_attr, batch)
        x4 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3 + x4

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        y1 = self.lin3(x)

        return y1
