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
from _utils import utils
from _utils import plot_utils_dev as pud

import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=logging.INFO,format="%(asctime)s %(name)s %(levelname)7s %(message)s",filename='/workspace/home/azuma/Pathology_Graph/230712_cg_classification/230715_multi_classification/230716_basic_GNN/results/log_texts/230717_prelim.txt', filemode='w')
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

test_list = [
    '/workspace/mnt/data1/Azuma/Pathology/centroids_based_graph/0_N/cell_graphs/test',
    '/workspace/mnt/data1/Azuma/Pathology/centroids_based_graph/1_PB/cell_graphs/test',
    '/workspace/mnt/data1/Azuma/Pathology/centroids_based_graph/2_UDH/cell_graphs/test',
    '/workspace/mnt/data1/Azuma/Pathology/centroids_based_graph/3_ADH/cell_graphs/test',
    '/workspace/mnt/data1/Azuma/Pathology/centroids_based_graph/4_FEA/cell_graphs/test',
    '/workspace/mnt/data1/Azuma/Pathology/centroids_based_graph/5_DCIS/cell_graphs/test',
    '/workspace/mnt/data1/Azuma/Pathology/centroids_based_graph/6_IC/cell_graphs/test']
test_data = dataloader.collect_multi_data(path_list=test_list,original_label=[0,1,2,3,4,5,6])

# %%
batch_size = 64
epoch_num = 30
n_classes = 7
lr = 1e-2
hidden_channels = 128

IS_CUDA = torch.cuda.is_available()
DEVICE = 'cuda:0' if IS_CUDA else 'cpu'

# dataloader
utils.fix_seed(seed=123,fix_gpu=True)
trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valloader = DataLoader(val_data, batch_size=batch_size)
testloader = DataLoader(test_data, batch_size=batch_size)

model = basic_gnn.Net(hidden_channels=hidden_channels,n_classes=n_classes).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(),lr=lr)
criterion = nn.CrossEntropyLoss()

logger.info("Parameters: {}".format({"batch":batch_size,"lr":lr,"hidden_channels":hidden_channels}))
# %%
def train(trainloader,validloader):
    model.train()
    utils.fix_seed(seed=123,fix_gpu=False)
    for data in trainloader:  # Iterate in batches over the training dataset.
         data = data.to(DEVICE)
         out = model(data)  # Perform a single forward pass.
         loss = criterion(out, data.t)  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.
    
    utils.fix_seed(seed=123,fix_gpu=False)
    valid_batch_loss = []
    with torch.no_grad():
        for data in validloader:
            data = data.to(DEVICE)
            out = model(data)
            valid_loss = criterion(out, data.t)  # Compute the loss.
            valid_batch_loss.append(valid_loss.cpu().detach().numpy())

    return loss, np.mean(valid_batch_loss)

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

# %%
# learning process
best_val_accuracy = 0.
model_path = '/workspace/home/azuma/Pathology_Graph/230712_cg_classification/230715_multi_classification/230716_basic_GNN/results/230717_graphsage_pt/model_best_val_accuracy.pt'

train_res = []
valid_res = []
train_loss_res = []
valid_loss_res = []
for epoch in range(epoch_num):
    loss, valid_loss = train(trainloader,valloader)
    train_loss_res.append(loss.cpu().detach().numpy())
    valid_loss_res.append(valid_loss)
    train_scores = test(trainloader)
    train_res.append(train_scores)
    valid_scores = test(valloader)
    valid_res.append(valid_scores)

    # compute & store accuracy + model
    train_acc = train_scores[0]
    valid_acc = valid_scores[0]
    if valid_acc > best_val_accuracy:
        best_val_accuracy = valid_acc
        then_train_accuracy = train_acc
        torch.save(model.state_dict(), model_path)

    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Valid Acc: {valid_acc:.4f}', 'Train Prec:',['{:.4f}'.format(n) for n in train_scores[2]],'Valid Prec:',['{:.4f}'.format(n) for n in valid_scores[2]])

# %% inference
model = basic_gnn.Net(hidden_channels=64,n_classes=n_classes).to(DEVICE)
model.load_state_dict(torch.load(model_path))

optimizer = torch.optim.Adam(model.parameters(),lr=lr)
criterion = nn.CrossEntropyLoss()

test_res = test(testloader)
print(f'Test Acc: {test_res[0]:.4f}','Test Prec:',test_res[2])

logger.info("Train ACC: {}".format(then_train_accuracy))
logger.info("Valid ACC: {}".format(best_val_accuracy))
logger.info("Test ACC: {}".format(test_res[0]))

pud.plot_acc(train_res,valid_res)
pud.plot_loss(train_loss_res,valid_loss_res)
pud.plot_each_prec(train_res,valid_res)