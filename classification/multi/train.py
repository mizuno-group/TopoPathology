# -*- coding: utf-8 -*-
"""
Created on 2023-07-17 (Mon) 17:29:12

Script for training cell graph classification on BRACS

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

import logging
logging.basicConfig(level=logging.INFO,format="%(asctime)s %(name)s %(levelname)7s %(message)s",filename='/workspace/home/azuma/Pathology_Graph/230712_cg_classification/230715_multi_classification/230716_basic_GNN/results/log_texts/230717_prelim.txt', filemode='r')
logger = logging.getLogger(__name__)

#%%
train_list = [
    '/workspace/mnt/data1/Azuma/Pathology/centroids_based_graph/0_N/cell_graphs/train',
    '/workspace/mnt/data1/Azuma/Pathology/centroids_based_graph/1_PB/cell_graphs/train',
    '/workspace/mnt/data1/Azuma/Pathology/centroids_based_graph/2_UDH/cell_graphs/train',
    '/workspace/mnt/data1/Azuma/Pathology/centroids_based_graph/3_ADH/cell_graphs/train',
    '/workspace/mnt/data1/Azuma/Pathology/centroids_based_graph/4_FEA/cell_graphs/train',
    '/workspace/mnt/data1/Azuma/Pathology/centroids_based_graph/5_DCIS/cell_graphs/train',
    '/workspace/mnt/data1/Azuma/Pathology/centroids_based_graph/6_IC/cell_graphs/train']
train_data = dataloader.collect_multi_data(path_list=train_list,original_label=[0,1,2,3,4,5,6])

val_list = [
    '/workspace/mnt/data1/Azuma/Pathology/centroids_based_graph/0_N/cell_graphs/val',
    '/workspace/mnt/data1/Azuma/Pathology/centroids_based_graph/1_PB/cell_graphs/val',
    '/workspace/mnt/data1/Azuma/Pathology/centroids_based_graph/2_UDH/cell_graphs/val',
    '/workspace/mnt/data1/Azuma/Pathology/centroids_based_graph/3_ADH/cell_graphs/val',
    '/workspace/mnt/data1/Azuma/Pathology/centroids_based_graph/4_FEA/cell_graphs/val',
    '/workspace/mnt/data1/Azuma/Pathology/centroids_based_graph/5_DCIS/cell_graphs/val',
    '/workspace/mnt/data1/Azuma/Pathology/centroids_based_graph/6_IC/cell_graphs/val']
val_data = dataloader.collect_multi_data(path_list=val_list,original_label=[0,1,2,3,4,5,6])
# %%
# %%
batch_size = 64
epoch_num = 30
n_classes = 7
lr = 1e-2
hidden_channels = 128

IS_CUDA = torch.cuda.is_available()
DEVICE = 'cuda:0' if IS_CUDA else 'cpu'

# dataloader
trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valloader = DataLoader(val_data, batch_size=batch_size,shuffle=True)

model = basic_gnn.Net(hidden_channels=hidden_channels,n_classes=n_classes).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(),lr=lr)
criterion = nn.CrossEntropyLoss()

logger.info("Parameters: {}".format({"batch":batch_size,"lr":lr,"hidden_channels":hidden_channels}))
# %%
def train():
    model.train()
    torch.manual_seed(123)
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
    torch.manual_seed(123)

    pred_total =  []
    label_total = []
    for data in loader:  # Iterate in batches over the training/test dataset.
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

# learning process
best_val_accuracy = 0.
model_path = '/workspace/home/azuma/Pathology_Graph/230712_cg_classification/230715_multi_classification/230716_basic_GNN/results/230717_basic_pt'

train_res = []
valid_res = []
loss_res = []
for epoch in range(epoch_num):
    loss = train()
    loss_res.append(loss.cpu().detach().numpy())
    train_scores = test(trainloader)
    train_res.append(train_scores)
    valid_scores = test(valloader)
    valid_res.append(valid_scores)

    # compute & store accuracy + model
    train_acc = train_scores[0]
    valid_acc = valid_scores[0]
    if valid_acc > best_val_accuracy:
        best_val_accuracy = valid_acc
        torch.save(model.state_dict(), os.path.join(model_path, 'model_best_val_accuracy.pt'))

    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Valid Acc: {valid_acc:.4f}', 'Train Prec:',['{:.4f}'.format(n) for n in train_scores[2]],'Valid Prec:',['{:.4f}'.format(n) for n in valid_scores[2]])

# %% plot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
color_list = list(mcolors.TABLEAU_COLORS.keys())

# accuracy
fig,ax = plt.subplots()
plt.plot([t[0] for t in train_res],label='train')
plt.plot([t[0] for t in valid_res],label='valid')
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

# precision
fig,ax = plt.subplots()
prect_res = [t[3] for t in train_res]
prect_t = np.array(prect_res).T
precv_res = [t[3] for t in valid_res]
precv_t = np.array(precv_res).T

for i in range(len(prect_t)):
    plt.plot(prect_t[i],label=str(i)+": train",linestyle="solid",color=color_list[i])
    plt.plot(precv_t[i],label=str(i)+": valid",linestyle="dashed",color=color_list[i])

plt.xlabel("epochs")
plt.ylabel("Precision")
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().yaxis.set_ticks_position('left')
plt.gca().xaxis.set_ticks_position('bottom')
ax.set_axisbelow(True)
ax.grid(color="#ababab",linewidth=0.5)
plt.legend(loc='upper right',shadow=True,bbox_to_anchor=(1.21, 1.01))
plt.show()

# loas
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
logger.info("Model: {}".format(model))
logger.info("Train ACC: {}".format(train_res[-1][0]))
logger.info("Valid ACC: {}".format(valid_res[-1][0]))
#logger.info("Test ACC: {}".format(test_res[0]))