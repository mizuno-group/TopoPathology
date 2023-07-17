# -*- coding: utf-8 -*-
"""
Created on 2023-07-17 (Mon) 17:30:57

Script for testing cell graph classification on BRACS

@author: I.Azuma
"""
#%%
import os
import torch
print(torch.__version__) # 2.0.1
print(torch.cuda.get_device_name()) # NVIDIA GeForce RTX 3090
from torch_geometric.data import Data,DataLoader
import torch.nn as nn

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

import sys
sys.path.append('/workspace/home/azuma/github/TopoPathology')
from classification.multi import dataloader
from classification.multi.models import basic_gnn

import warnings
warnings.filterwarnings('ignore')

# %% inference
test_list = [
    '/workspace/mnt/data1/Azuma/Pathology/centroids_based_graph/0_N/cell_graphs/test',
    '/workspace/mnt/data1/Azuma/Pathology/centroids_based_graph/1_PB/cell_graphs/test',
    '/workspace/mnt/data1/Azuma/Pathology/centroids_based_graph/2_UDH/cell_graphs/test',
    '/workspace/mnt/data1/Azuma/Pathology/centroids_based_graph/3_ADH/cell_graphs/test',
    '/workspace/mnt/data1/Azuma/Pathology/centroids_based_graph/4_FEA/cell_graphs/test',
    '/workspace/mnt/data1/Azuma/Pathology/centroids_based_graph/5_DCIS/cell_graphs/test',
    '/workspace/mnt/data1/Azuma/Pathology/centroids_based_graph/6_IC/cell_graphs/test',]
test_data = dataloader.collect_multi_data(path_list=test_list,original_label=[0,1,2,3,4,5,6])

# %%
model_path = '/workspace/home/azuma/Pathology_Graph/230712_cg_classification/230712_binary_classification_dev/230717_layer_dev/results/baseline/model_best_val_accuracy.pt' # pre-trained model
batch_size = 128
n_classes = 2
IS_CUDA = torch.cuda.is_available()
DEVICE = 'cuda:0' if IS_CUDA else 'cpu'

# dataloader
testloader = DataLoader(test_data, batch_size=batch_size)

model = basic_gnn.Net(hidden_channels=64,n_classes=n_classes).to(DEVICE)
model.load_state_dict(torch.load(model_path))

optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
criterion = nn.CrossEntropyLoss()

# %%
def test(loader):
    model.eval()
    torch.manual_seed(123)

    pred_total =  []
    label_total = []
    for data in trainloader:  # Iterate in batches over the training/test dataset.
        data = data.to(DEVICE)
        out = model(data)  
        pred = out.argmax(dim=1)  # Use the class with highest probability.

        pred_total.extend(list(pred.cpu().detach().numpy()))
        label_total.extend(list(data.t.cpu().detach().numpy()))

    acc_score = accuracy_score(label_total, pred_total)
    wf1_score = f1_score(label_total, pred_total, average='weighted')
    pre_score = precision_score(label_total, pred_total, average=None)
    rec_score = recall_score(label_total, pred_total, average=None)
        
    total_scores = (acc_score, wf1_score, pre_score, rec_score)

    return total_scores   # Derive ratio of correct predictions.

test_res = test(testloader)
print(f'Test Acc: {test_res[0]:.4f}','Test Prec:',test_res[3])