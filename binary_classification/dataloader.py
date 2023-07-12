# -*- coding: utf-8 -*-
"""
Created on 2023-07-12 (Wed) 21:58:18

dataloader for BRACS binary classification

@author: I.Azuma
"""
#%%
import torch
from torch_geometric.data import Data,DataLoader

import glob
from tqdm import tqdm
import numpy as np

from dgl.data.utils import load_graphs

#%%
def collect_data(bin_path='/workspace/Pathology_Graph/230712_cg_classification/results/centroids_based_graph/cell_graphs/train/*.bin'):
    l = glob.glob(bin_path)
    data_list = []
    for path in tqdm(l):
        g_list, label_dict = load_graphs(path)
        edge_info = g_list[0].edges()
        node_feature = g_list[0].ndata['feat']

        x = torch.tensor(node_feature, dtype=torch.float)
        edge_index = torch.tensor([np.array(edge_info[0]),np.array(edge_info[1])])
        label = label_dict['label']
        if int(label) == 6:
            label = 1
        else:
            label = 0
        d = Data(x=x,edge_index=edge_index.contiguous(),t=label)
        data_list.append(d)
    
    return data_list

