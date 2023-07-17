# -*- coding: utf-8 -*-
"""
Created on 2023-07-13 (Thu) 11:42:46

Script for training cell graph classification on BRACS

@author: I.Azuma
"""
#%%
import torch
print(torch.__version__) # 2.0.1
print(torch.cuda.get_device_name()) # NVIDIA GeForce RTX 3090
from torch_geometric.loader import DataLoader
import torch.nn as nn

from sklearn.metrics import accuracy_score, f1_score

import sys
sys.path.append('/workspace/home/azuma/github/TopoPathology')
from classification.binary import dataloader
from classification.binary.models import basic_gnn

#%% data loader
train_data = dataloader.collect_data(bin_path='/workspace/mnt/data1/Azuma/Pathology/centroids_based_graph/cell_graphs/train/*.bin')
val_data = dataloader.collect_data(bin_path='/workspace/mnt/data1/Azuma/Pathology/centroids_based_graph/cell_graphs/val/*.bin')
#test_data = dataloader.collect_data(bin_path='/workspace/Pathology_Graph/230712_cg_classification/results/centroids_based_graph/cell_graphs/test/*.bin')

#%%
batch_size = 32
epoch_num = 30

IS_CUDA = torch.cuda.is_available()
DEVICE = 'cuda:0' if IS_CUDA else 'cpu'

# dataloader
trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
valloader = DataLoader(val_data, batch_size=batch_size)
# testloader = DataLoader(test_data, batch_size=batch_size)

model = basic_gnn.Net(hidden_channels=64).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
criterion = nn.CrossEntropyLoss()

def train():
    model.train()

    for data in trainloader:  # Iterate in batches over the training dataset.
         data = data.to(DEVICE)
         out = model(data)  # Perform a single forward pass.
         loss = criterion(out, data.t)  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.
    
    return loss

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

# learning process
best_val_accuracy = 0.
model_path = '/workspace/Pathology_Graph/230712_cg_classification/230712_binary_classification_dev/results/230712_basic_pt'

train_res = []
valid_res = []
loss_res = []
for epoch in range(epoch_num):
    loss = train()
    loss_res.append(loss.cpu().detach().numpy())
    train_acc, train_wf1 = test(trainloader)
    train_res.append(train_acc)
    valid_acc, valid_wf1 = test(valloader)
    valid_res.append(valid_acc)

    # compute & store accuracy + model
    if valid_acc > best_val_accuracy:
        best_val_accuracy = valid_acc
        torch.save(model.state_dict(), os.path.join(model_path, 'model_best_val_accuracy.pt'))

    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Valid Acc: {valid_acc:.4f}')

#%% plot
import matplotlib.pyplot as plt

# accuracy
fig,ax = plt.subplots()
plt.plot(train_res,label='train')
plt.plot(valid_res,label='valid')
plt.xlabel("epochs")
plt.ylabel("Accuracy")

plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().yaxis.set_ticks_position('left')
plt.gca().xaxis.set_ticks_position('bottom')
ax.set_axisbelow(True)
ax.grid(color="#ababab",linewidth=0.5)
plt.legend(shadow=True,loc='best')
plt.show()

# loass
fig,ax = plt.subplots()
plt.plot(loss_res,label='train')
plt.xlabel("epochs")
plt.ylabel("Loss")

plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().yaxis.set_ticks_position('left')
plt.gca().xaxis.set_ticks_position('bottom')
ax.set_axisbelow(True)
ax.grid(color="#ababab",linewidth=0.5)
plt.legend(shadow=True,loc='best')
plt.show()

# %%
