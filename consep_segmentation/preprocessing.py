# -*- coding: utf-8 -*-
"""
Created on 2023-06-29 (Thu) 12:42:02

dataloader for CoNSeP dataset
https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/

Class values:
other
inflammatory
healthy epithelial
dysplastic/malignant epithelial
fibroblast
muscle
endothelial

code references
- https://www.kaggle.com/code/benyaminghahremani/nuclei-multi-class-semantic-segmentation-u-net/notebook

------------------------------------------------------------
import os
import zipfile

local_zip = '/workspace/datasource/concep/consep_dataset.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('./consep')

@author: I.Azuma
"""
#%%
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import scipy.io as scpio
from torch.utils.data import Dataset

import torch

import sys
sys.path.append('/workspace/github/histocartography')

from histocartography.preprocessing import (
    VahadaneStainNormalizer         # stain normalizer
)
norm_target_path = '/workspace/github/hact-net/data/target.png'

#%% helper functions
# random cropping patches from images with arbitrary size
def random_crop(img, target_shape=(256,256),seed=None):
    n_rows = img.shape[0]
    n_cols = img.shape[1]

    row_size, col_size = target_shape

    start_row = np.random.RandomState(seed).choice(range(n_rows-row_size))
    start_col = np.random.RandomState(seed).choice(range(n_cols-col_size))

    return img[start_row:start_row+row_size,start_col:start_col+col_size]

# making images into superpixels (patches)
def make_pathces(img, target_shape=(256,256),num_channels=3):
    n_rows = np.ceil(img.shape[0]/target_shape[0]).astype('int')
    n_cols = np.ceil(img.shape[1]/target_shape[1]).astype('int')

    if img.shape[0]%target_shape[0] != 0:
        img = np.concatenate((img[:,:],np.flipud(img[(img.shape[0]-(n_rows*target_shape[0])):,:])),axis=0)

    if img.shape[1]%target_shape[1] != 0:
        img = np.concatenate((img[:,:],np.fliplr(img[:,(img.shape[1]-(n_cols*target_shape[1])):])),axis=1)
    num_patches = n_cols * n_rows

    if len(img.shape) == 2:
        patches = np.empty(shape=(n_rows,n_cols,target_shape[0],target_shape[1]))
    else:
        patches = np.empty(shape=(n_rows,n_cols,target_shape[0],target_shape[1], num_channels))

    for i in range(0, img.shape[0], target_shape[0]):
        for j in range(0, img.shape[1], target_shape[1]):
            patches[i//target_shape[0],j//target_shape[1]] = np.copy(img[i:i+target_shape[0],j:j+target_shape[1]])
    return patches

#%% Train class
class CoNSeP_Train(Dataset):
    def __init__(self, train_dir='./consep/CoNSeP/Train',stain_norm=False):
        IMG_HEIGHT = 256
        IMG_WIDTH = 256
        num_patches = 16 # FIXME: Improve automatic recognition of PATCH number
        num_random_cropped_patches = 1
        num_augmentations = 3
        total_num_of_patches = (num_patches + num_random_cropped_patches)*num_augmentations
        n = round(np.sqrt(num_patches))

        train_images = sorted(os.listdir(train_dir + '/Images'))
        train_ground_truth = sorted(os.listdir(train_dir + '/Labels'))

        # creating the new dataset
        X_train = np.zeros((len(train_images)*total_num_of_patches,IMG_HEIGHT,IMG_WIDTH,3))
        y_train = np.zeros((len(train_images)*total_num_of_patches,IMG_HEIGHT,IMG_WIDTH))

        # train set
        index = 0
        if stain_norm:
            normalizer = VahadaneStainNormalizer(target_path=norm_target_path) 
        for inx, img_dir in tqdm(enumerate(train_images)):
            # importing images
            full_img_dir = os.path.join(train_dir+'/Images/', img_dir)
            img = plt.imread(full_img_dir)

            # importing masks
            img_type_map = scpio.loadmat(os.path.join(train_dir+'/Labels/',train_ground_truth[inx]))['type_map']
            img_type_map_patches = make_pathces(img_type_map,target_shape=(IMG_HEIGHT,IMG_WIDTH))
            
            # omitting the forth channels of the images
            if img.shape[2] == 4: # FIXME: what is the forth channel ?
                img = img[:,:,:3]
            else:
                img = img
            
            # stain normalizations
            if stain_norm:
                try:
                    normalizer = VahadaneStainNormalizer(target_path=norm_target_path)
                    img = normalizer.process(img)
                except:
                    pass
            
            # making the image into 16 equal patches
            img_patches = make_pathces(img,target_shape=(IMG_HEIGHT,IMG_WIDTH))
            #add the patches to the dataset
            for i in range(n):
                for j in range(n):
                    pos = np.random.choice([1,2,3])# for random rotation choosen from the list (90, 180, 270)
                    X_train[index] = img_patches[i][j]
                    X_train[index+1] = np.rot90(X_train[index], pos)# rotating image and add it to dataset
                    X_train[index+2] = np.fliplr(X_train[index])# flipping the image and add it to dataset

                    y_train[index] = img_type_map_patches[i][j]
                    y_train[index+1] = np.rot90(y_train[index], pos)
                    y_train[index+2] = np.fliplr(y_train[index])
                    index += 3 # slide three
            #random cropping the image 'num_random_cropped_patches' times
            for i in range(num_random_cropped_patches):
                seed = np.random.randint(10000)
                pos = np.random.choice([1,2,3])
                X_train[index] = random_crop(img, target_shape=(IMG_HEIGHT,IMG_WIDTH), seed=seed)
                X_train[index+1] = np.rot90(X_train[index], pos)
                X_train[index+2] = np.fliplr(X_train[index])

                y_train[index] = random_crop(img_type_map, target_shape=(IMG_HEIGHT,IMG_WIDTH), seed=seed)
                y_train[index+1] = np.rot90(y_train[index], pos)
                y_train[index+2] = np.fliplr(y_train[index])
                index += 3
        
        # label normalization
        # combinning the 3,4,5 classes with each ohter and 6,7 classes likewise
        y_train[np.where(y_train == 4)] = 3
        y_train[np.where((y_train == 5)|(y_train == 6)|(y_train == 7))] = 4

        self.X_train = X_train
        self.y_train = y_train

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self,index):
        img = self.X_train[index]
        mask =  self.y_train[index]
        return img, mask

# %% Test class
class CoNSeP_Test(Dataset):
    def __init__(self, test_dir='./consep/CoNSeP/Test',stain_norm=False):
        IMG_HEIGHT = 256
        IMG_WIDTH = 256
        num_patches = 16
        num_random_cropped_patches = 1
        num_augmentations = 3
        total_num_of_patches = (num_patches + num_random_cropped_patches)*num_augmentations
        n = round(np.sqrt(num_patches))

        test_images = sorted(os.listdir(test_dir + '/Images'))
        test_ground_truth = sorted(os.listdir(test_dir + '/Labels'))

        # creating the new dataset
        X_test = np.zeros((len(test_images)*num_patches,IMG_HEIGHT,IMG_WIDTH,3))
        y_test = np.zeros((len(test_images)*num_patches,IMG_HEIGHT,IMG_WIDTH))

        # test set 
        test_index = 0
        if stain_norm:
            normalizer = VahadaneStainNormalizer(target_path=norm_target_path)       
        for inx, img_dir in tqdm(enumerate(test_images)):
            # importing images
            full_img_dir = os.path.join(test_dir+'/Images/', img_dir)
            img = plt.imread(full_img_dir)
            
            # importing masks
            img_type_map = scpio.loadmat(os.path.join(test_dir+'/Labels/',test_ground_truth[inx]))['type_map']
            img_type_map_patches = make_pathces(img_type_map,target_shape=(IMG_HEIGHT,IMG_WIDTH))
            
            # omitting the forth channels of the images
            if img.shape[2] == 4:
                img = img[:,:,:3]
            else:
                img = img
            
            # stain normalization
            if stain_norm:
                try:
                    img = normalizer.process(img)
                except:
                    pass
            
            # making the image into 16 equal patches
            img_patches = make_pathces(img,target_shape=(IMG_HEIGHT,IMG_WIDTH))
            #add the patches to the dataset
            for i in range(n):
                for j in range(n):
                    X_test[test_index] = img_patches[i][j]

                    y_test[test_index] = img_type_map_patches[i][j]
                    test_index +=1
        
        # label normalization
        # combinning the 3,4,5 classes with each ohter and 6,7 classes likewise
        y_test[np.where(y_test == 4)] = 3
        y_test[np.where((y_test == 5)|(y_test == 6)|(y_test == 7))] = 4

        self.X_test = X_test
        self.y_test = y_test

    def __len__(self):
        return len(self.X_test)

    def __getitem__(self,index):
        img =self.X_test[index]
        mask =  self.y_test[index]
        return img, mask

# %% Inference class
class CoNSeP_Inference(Dataset):
    def __init__(self, inference_dir='./consep/CoNSeP/Test',stain_norm=False,
                IMG_HEIGHT = 512,IMG_WIDTH = 512):
        num_random_cropped_patches  = 1

        inference_images = sorted(os.listdir(inference_dir + '/Images'))
        inf_ground_truth = sorted(os.listdir(inference_dir + '/Labels'))

        # creating the new dataset
        X_inference = np.zeros((len(inference_images),IMG_HEIGHT,IMG_WIDTH,3))
        y_inference = np.zeros((len(inference_images),IMG_HEIGHT,IMG_WIDTH))

        # test set 
        inf_index = 0
        if stain_norm:
            normalizer = VahadaneStainNormalizer(target_path=norm_target_path)       
        for inx, img_dir in tqdm(enumerate(inference_images)):
            # importing images
            full_img_dir = os.path.join(inference_dir+'/Images/', img_dir)
            img = plt.imread(full_img_dir)

            # importing mask
            img_type_map = scpio.loadmat(os.path.join(inference_dir+'/Labels/',inf_ground_truth[inx]))['type_map']
            
            # omitting the forth channels of the images
            if img.shape[2] == 4:
                img = img[:,:,:3]
            else:
                img = img
            
            # stain normalization
            if stain_norm:
                try:
                    img = normalizer.process(img)
                except:
                    pass

            #add the patches to the dataset
            for i in range(num_random_cropped_patches):
                seed = np.random.randint(10000)
                X_inference[inx] = random_crop(img, target_shape=(IMG_HEIGHT,IMG_WIDTH), seed=seed)
                y_inference[inx] = random_crop(img_type_map, target_shape=(IMG_HEIGHT,IMG_WIDTH), seed=seed)

        self.X_inference = X_inference
        self.y_inference = y_inference

    def __len__(self):
        return len(self.X_inference)

    def __getitem__(self,index):
        img =self.X_inference[index]
        mask =  self.y_inference[index]
        return img, mask

#%% raw inference
class CoNSeP_Raw_Inf(Dataset):
    def __init__(self, inference_dir='./consep/CoNSeP/Test',stain_norm=False):
        IMG_HEIGHT = 1000
        IMG_WIDTH = 1000

        inference_images = sorted(os.listdir(inference_dir + '/Images'))
        inf_ground_truth = sorted(os.listdir(inference_dir + '/Labels'))

        # creating the new dataset
        X_inference = np.zeros((len(inference_images),IMG_HEIGHT,IMG_WIDTH,3))
        y_inference = np.zeros((len(inference_images),IMG_HEIGHT,IMG_WIDTH))

        # test set 
        inf_index = 0
        if stain_norm:
            normalizer = VahadaneStainNormalizer(target_path=norm_target_path)
        for inx, img_dir in tqdm(enumerate(inference_images)):
            # importing images
            full_img_dir = os.path.join(inference_dir+'/Images/', img_dir)
            img = plt.imread(full_img_dir)

            # importing mask
            img_type_map = scpio.loadmat(os.path.join(inference_dir+'/Labels/',inf_ground_truth[inx]))['type_map']
            
            # omitting the forth channels of the images
            if img.shape[2] == 4:
                img = img[:,:,:3]
            else:
                img = img
            
            # stain normalization
            if stain_norm:
                try:
                    img = normalizer.process(img)
                except:
                    pass
            
            X_inference[inx] = img
            y_inference[inx] = img_type_map

        self.X_inference = X_inference
        self.y_inference = y_inference

    def __len__(self):
        return len(self.X_inference)

    def __getitem__(self,index):
        img =self.X_inference[index]
        mask =  self.y_inference[index]
        return img, mask


#%% inference on BRCA-M2C
class BRCAM2C_Inference(Dataset):
    def __init__(self, inference_dir='/workspace/github/Dataset-BRCA-M2C/images',stain_norm=False):
        IMG_HEIGHT = 256
        IMG_WIDTH = 256
        num_random_cropped_patches  = 1
        num_patches = 4
        n = round(np.sqrt(num_patches))

        inference_images = sorted(os.listdir(inference_dir + '/'))

        # creating the new dataset
        X_inference = np.zeros((len(inference_images)*num_patches,IMG_HEIGHT,IMG_WIDTH,3))

        # inference set 
        index = 0
        if stain_norm:
            normalizer = VahadaneStainNormalizer(target_path=norm_target_path)
        for inx, img_dir in tqdm(enumerate(inference_images)):
            # importing images
            full_img_dir = os.path.join(inference_dir+'/', img_dir)
            img = plt.imread(full_img_dir)

             # stain normalization
            if stain_norm:
                try:
                    img = normalizer.process(img)
                except:
                    pass

            img_patches = make_pathces(img,target_shape=(IMG_HEIGHT,IMG_WIDTH))
            for i in range(n):
                for j in range(n):
                    X_inference[index] = img_patches[i][j]
                    index += 1
        self.X_inference = X_inference

    def __len__(self):
        return len(self.X_inference)

    def __getitem__(self,index):
        img =self.X_inference[index]
        return img
