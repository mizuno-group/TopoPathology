# -*- coding: utf-8 -*-
"""
Created on 2023-06-05 (Mon) 23:02:39

dataloader

@author: I.Azuma
"""
import os
from glob import glob
from PIL import Image

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

import matplotlib.pyplot as plt
#%%
def make_filepath_list():
    """
    path to train and validation datasets
    
    Returns
    -------
    train_file_list: list
    valid_file_list: list
    """
    train_file_list = []
    valid_file_list = []

    for top_dir in os.listdir('./Images/'):
        file_dir = os.path.join('./Images/', top_dir)
        file_list = os.listdir(file_dir)

        num_data = len(file_list)
        num_split = int(num_data * 0.8)

        train_file_list += [os.path.join('./Images', top_dir, file).replace('\\', '/') for file in file_list[:num_split]]
        valid_file_list += [os.path.join('./Images', top_dir, file).replace('\\', '/') for file in file_list[num_split:]]
    
    return train_file_list, valid_file_list

class ImageTransform(nn.Module):
    """

    Attributes
    ----------
    resize: int
    mean: (R, G, B)
    std: (R, G, B)
    """
    def __init__(self, resize):
        self.data_trasnform = {
            'train': transforms.Compose([
                transforms.RandomHorizontalFlip(),
                # resize×resize
                transforms.Resize((resize, resize)),
                transforms.ToTensor()
            ]),
            'valid': transforms.Compose([
                # resize×resize
                transforms.Resize((resize, resize)),
                transforms.ToTensor()
            ])
        }
    
    def __call__(self, img, phase='train'):
        return self.data_trasnform[phase](img)

class PathDataset(data.Dataset):
    """
    Datasets class
    
    Attrbutes
    ---------
    file_list: list
    classes: list
        label names
    transform: object
        Preprocessing class instance
    phase: 'train' or 'valid'
    """
    def __init__(self, file_list, classes, transform=None, phase='train'):
        self.file_list = file_list
        self.transform = transform
        self.classes = classes
        self.phase = phase
    
    def __len__(self):
        """
        number of images
        """
        return len(self.file_list)
    
    def __getitem__(self, index):
        """
        Obtain Tensor format data and labels for preprocessed image data
        """
        # load
        img_path = self.file_list[index]
        img = Image.open(img_path)
        
        # preprocessing
        img_transformed = self.transform(img, self.phase)
        
        # extract label
        label = self.file_list[index].split('/')[-2].split('_')[1]
        label = self.classes.index(label)
        
        return img_transformed, label