# -*- coding: utf-8 -*-
"""
Created on 2023-06-05 (Mon) 23:08:12

train

@author: I.Azuma
"""
#%%
import os
from glob import glob
from PIL import Image
from tqdm import tqdm

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

import matplotlib.pyplot as plt

import sys
sys.path.append('/workspace/github/TopoPathology')
from baselines.dataloader import PathDataset, ImageTransform

#%%
train_path = '/workspace/datasource/BRACS/previous_versions/Version1_MedIA/Images/train/'
val_path = '/workspace/datasource/BRACS/previous_versions/Version1_MedIA/Images/val/'
# 1. get train image path
subdirs = os.listdir(train_path)
train_file_list = []
for subdir in (subdirs + ['']):  # look for all the subdirs AND the image path
    train_file_list += glob(os.path.join(train_path, subdir, '*.png'))

# 2. get val image path
subdirs = os.listdir(val_path)
valid_file_list = []
for subdir in (subdirs + ['']):  # look for all the subdirs AND the image path
    valid_file_list += glob(os.path.join(val_path, subdir, '*.png'))

#%%
degree_classes = ['N', 'PB','UDH','ADH','FEA','DCIS','IC']
# resize size
resize = 32
# Dataset
train_dataset = PathDataset(
    file_list=train_file_list, classes=degree_classes,
    transform=ImageTransform(resize),
    phase='train'
)
valid_dataset = PathDataset(
    file_list=valid_file_list, classes=degree_classes,
    transform=ImageTransform(resize),
    phase='valid'
)

#%%
# batch size
batch_size = 64
# DataLoader
train_dataloader = data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)

valid_dataloader = data.DataLoader(
    valid_dataset, batch_size=32, shuffle=False)

# dict
dataloaders_dict = {
    'train': train_dataloader, 
    'valid': valid_dataloader
}

#%%
from baselines.models import cnn

IS_CUDA = torch.cuda.is_available()
DEVICE = 'cuda:0' if IS_CUDA else 'cpu'
#DEVICE = 'cpu' # 00:31<06:06, 16.62s/it

model = cnn.CNN()
model.to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

# epoch number
num_epochs = 30

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch+1, num_epochs))
    print('-------------')
    
    for phase in ['train', 'valid']:
        
        if phase == 'train':
            # training phase
            model.train()
        else:
            # evaluation phase
            model.eval()
        
        # loss sum
        epoch_loss = 0.0
        # correct sumnext(model.parameters()).is_cudanext(model.parameters()).is_cudanext(model.parameters()).is_cudanext(model.parameters()).is_cudanext(model.parameters()).is_cudanext(model.parameters()).is_cudanext(model.parameters()).is_cudanext(model.parameters()).is_cudanext(model.parameters()).is_cuda
        epoch_corrects = 0
        
        for inputs, labels in tqdm(dataloaders_dict[phase]):
            inputs=inputs.to(DEVICE)
            labels=labels.to(DEVICE)
            # optimizer initialization
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # label prediction
                _, preds = torch.max(outputs, 1)
                
                # backward
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                epoch_loss += loss.item() * inputs.size(0)
                
                # update
                epoch_corrects += torch.sum(preds == labels.data)

        # display
        epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
        epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))