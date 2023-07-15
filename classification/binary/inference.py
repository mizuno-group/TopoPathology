# -*- coding: utf-8 -*-
"""
Created on 2023-07-13 (Thu) 12:13:48

Script for testing cell graph classification on BRACS

@author: I.Azuma
"""
#%%
import torch
print(torch.__version__) # 2.0.1
print(torch.cuda.get_device_name()) # NVIDIA GeForce RTX 3090
from torch_geometric.data import Data,DataLoader
import torch.nn as nn

from sklearn.metrics import accuracy_score, f1_score

import sys
sys.path.append('/workspace/github/TopoPathology')
from binary_classification import dataloader
from binary_classification.models import basic_gnn

#%%
test_data = dataloader.collect_data(bin_path='/workspace/Pathology_Graph/230712_cg_classification/results/centroids_based_graph/cell_graphs/test/*.bin')

#%%
model_path = '/workspace/Pathology_Graph/230712_cg_classification/230712_binary_classification_dev/results/230712_basic_pt/model_best_val_accuracy.pt' # pre-trained model

batch_size = 32
IS_CUDA = torch.cuda.is_available()
DEVICE = 'cuda:0' if IS_CUDA else 'cpu'

# dataloader
testloader = DataLoader(test_data, batch_size=batch_size)

model = basic_gnn.Net(hidden_channels=64).to(DEVICE)
model.load_state_dict(torch.load(model_path))

optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
criterion = nn.CrossEntropyLoss()

def test(loader):
    model.eval()

    correct = 0
    wf1 = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        data = data.to(DEVICE)
        out = model(data)  
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.t).sum())  # Check against ground-truth labels.
        wf1 += f1_score(data.t.cpu(), pred.cpu(), average='weighted')
    
    acc = correct / len(loader.dataset)
    wf1 = wf1 / len(loader)

    return acc, wf1   # Derive ratio of correct predictions.

# inference process
test_acc, test_wf1 = test(testloader)

print(f'Test Acc: {test_acc:.4f}')
