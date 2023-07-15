# -*- coding: utf-8 -*-
"""
Created on 2023-06-29 (Thu) 12:52:30

U-Net

@author: I.Azuma
"""
#%%
import torch
print(torch.__version__) # 2.0.1
print(torch.cuda.get_device_name()) # NVIDIA GeForce RTX 3090
from torchsummary import summary
import torch.nn as nn
from torch.optim import Adam

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.append('/workspace/github/TopoPathology/consep_segmentation')
import preprocessing as pp
from models import conventional_unet
# %%
train_data = pp.CoNSeP_Train(train_dir='/workspace/datasource/consep/CoNSeP/Train')
test_data = pp.CoNSeP_Test(test_dir='/workspace/datasource/consep/CoNSeP/Test')

batch_size = 32
train_batch = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
test_batch = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, pin_memory=True)

# %%
IS_CUDA = torch.cuda.is_available()
DEVICE = 'cuda:0' if IS_CUDA else 'cpu'

model = conventional_unet.unet_model().to(DEVICE)
print(summary(model, (3,256,256)))

#%% learning phase
LEARNING_RATE = 1e-4
num_epochs = 10

loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
scaler = torch.cuda.amp.GradScaler()

for epoch in range(num_epochs):
    loop = tqdm(enumerate(train_batch),total=len(train_batch))
    for batch_idx, (raw_data, raw_targets) in loop:
        data = torch.permute(raw_data,(0,3,1,2)) # torch.Size([32, 3, 256, 256])
        data = data.to(DEVICE)
        data = data.type(torch.float32) # float 32
        targets = raw_targets.to(DEVICE) # torch.Size([32, 256, 256])
        targets = targets.type(torch.long)
        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

torch.save(model.state_dict(),'/workspace/Cell_Segment/230628_concep/230628_unet_on_cencep/results/230629_unet_model.pth')
# %% load pre-trained model
pre_model = conventional_unet.unet_model().to(DEVICE)
pre_model.load_state_dict(torch.load('/workspace/Cell_Segment/230628_concep/230628_unet_on_cencep/results/230629_unet_model.pth'))

# %% metric
def check_accuracy(loader, model):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = torch.permute(x,(0,3,1,2)) # torch.Size([32, 3, 256, 256])
            x = x.type(torch.float32)
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            softmax = nn.Softmax(dim=1)
            preds = torch.argmax(softmax(model(x)),axis=1)
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()

check_accuracy(train_batch, pre_model)
check_accuracy(test_batch, pre_model)

#%% visualization
for x,y in test_batch:
    x = torch.permute(x,(0,3,1,2)) # torch.Size([32, 3, 256, 256])
    x = x.type(torch.float32)
    x = x.to(DEVICE)
    fig , ax =  plt.subplots(3, 3, figsize=(18, 18))
    softmax = nn.Softmax(dim=1)
    preds = torch.argmax(softmax(model(x)),axis=1).to('cpu')
    img1 = np.transpose(np.array(x[0,:,:,:].to('cpu')),(1,2,0))
    preds1 = np.array(preds[0,:,:])
    mask1 = np.array(y[0,:,:])
    img2 = np.transpose(np.array(x[1,:,:,:].to('cpu')),(1,2,0))
    preds2 = np.array(preds[1,:,:])
    mask2 = np.array(y[1,:,:])
    img3 = np.transpose(np.array(x[2,:,:,:].to('cpu')),(1,2,0))
    preds3 = np.array(preds[2,:,:])
    mask3 = np.array(y[2,:,:])
    ax[0,0].set_title('Image')
    ax[0,1].set_title('Prediction')
    ax[0,2].set_title('Mask')
    ax[1,0].set_title('Image')
    ax[1,1].set_title('Prediction')
    ax[1,2].set_title('Mask')
    ax[2,0].set_title('Image')
    ax[2,1].set_title('Prediction')
    ax[2,2].set_title('Mask')
    ax[0][0].axis("off")
    ax[1][0].axis("off")
    ax[2][0].axis("off")
    ax[0][1].axis("off")
    ax[1][1].axis("off")
    ax[2][1].axis("off")
    ax[0][2].axis("off")
    ax[1][2].axis("off")
    ax[2][2].axis("off")
    ax[0][0].imshow(img1)
    ax[0][1].imshow(preds1)
    ax[0][2].imshow(mask1)
    ax[1][0].imshow(img2)
    ax[1][1].imshow(preds2)
    ax[1][2].imshow(mask2)
    ax[2][0].imshow(img3)
    ax[2][1].imshow(preds3)
    ax[2][2].imshow(mask3)   
    break
# %%
