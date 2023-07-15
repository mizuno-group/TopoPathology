# -*- coding: utf-8 -*-
"""
Created on 2023-06-05 (Mon) 22:44:58

MLP model

@author: I.Azuma
"""
#%%
import os
from glob import glob
from PIL import Image

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

#%%
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),        # (batch, 64, r, r) # r=resize
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),                                                # (batch, 64, r/2, r/2) # r=resize
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),      # (batch, 128, r/2, r/2) # r=resize
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),                                                # (batch, 128, r/4, r/4) # r=resize
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
          nn.Linear(in_features=4 * 4 * 32, out_features=7),
          #nn.Linear(in_features=64, out_features=7)
        )


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

"""
net = CNN()
print(net)

CNN(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (1): ReLU(inplace=True)
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): ReLU(inplace=True)
    (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): ReLU(inplace=True)
    (8): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (9): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (10): ReLU(inplace=True)
  )
  (classifier): Linear(in_features=2048, out_features=7, bias=True)
)
"""