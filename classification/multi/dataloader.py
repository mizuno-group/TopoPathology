# -*- coding: utf-8 -*-
"""
Created on 2023-07-17 (Mon) 00:58:03

dataloader for BRACS multi classification

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

def collect_multi_data(path_list=['/workspace/mnt/data1/Azuma/Pathology/centroids_based_graph/0_N/cell_graphs/train','/workspace/mnt/data1/Azuma/Pathology/centroids_based_graph/3_ADH/cell_graphs/train','/workspace/mnt/data1/Azuma/Pathology/centroids_based_graph/6_IC/cell_graphs/train'],original_label=[0,3,6]):
    data_list = []
    original_label = sorted(original_label)
    for path in path_list:
        bin_path = path+'/*.bin'
        l = glob.glob(bin_path)
        for path in tqdm(l):
            g_list, label_dict = load_graphs(path)
            edge_info = g_list[0].edges()
            node_feature = g_list[0].ndata['feat']

            x = torch.tensor(node_feature, dtype=torch.float)
            edge_index = torch.tensor([np.array(edge_info[0]),np.array(edge_info[1])])
            label = original_label.index(label_dict['label'])
            d = Data(x=x,edge_index=edge_index.contiguous(),t=label)
            data_list.append(d)
    return data_list