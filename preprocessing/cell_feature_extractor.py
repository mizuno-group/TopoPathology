# -*- coding: utf-8 -*-
"""
Created on 2023-07-11 (Tue) 12:13:51

Cell Feature Extractor.

@author: I.Azuma
"""
#%%
import json
import numpy as np
import scipy.io as sio
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt

#%%
class CellFeatureExtractor():
    def __init__(self,
                mat_path = '/workspace/github/TopoPathology/HoVerNet_res/BRACS/BRACS_291_IC_20.mat',
                json_path = '/workspace/github/TopoPathology/HoVerNet_res/BRACS/BRACS_291_IC_20.json'):
        self.mat_path = mat_path
        self.json_path = json_path

    def load_data(self):
        # load mat file
        result_mat = sio.loadmat(self.mat_path)
        self.inst_map = result_mat['inst_map']
        raw_map = result_mat['raw_map']
        self.feature_map = raw_map[:,:,4::] # extract softmax feature

        # load json file
        with open(self.json_path) as json_file:
            data = json.load(json_file)
            self.nuc_info = data['nuc']

    def single_extractor(self,cell=416,remove_contour=True,do_plot=False):
        """ Extract morphological features

        Args:
            cell (int, optional): Cell identification number. Defaults to 416.
            remove_contour (bool, optional): Remove contour from node feature target area. Defaults to True.
            do_plot (bool, optional): Plot the class frequency. Defaults to False.

        Returns:
           node_feature (ndarray): Cell feature for the segmented area.
        """
        area = np.where(self.inst_map==cell)
        area_list = list(zip(area[0],area[1])) # [(r1,c1),(r2,c2),...]
        inst_info = self.nuc_info[str(cell)]
        inst_contour = inst_info['contour']
        contour_list = [(t[1],t[0]) for t in inst_contour] # [(r1,c1),(r2,c2),...]
        if remove_contour:
            without_countour = sorted(list(set(area_list) - set(contour_list))) # remove contour pixels
            area_feature = self.feature_map[[t[0] for t in without_countour],[t[1] for t in without_countour],:]
        else:
            area_feature = self.feature_map[area[0],area[1],:] # with contour

        if do_plot:
            for v in area_feature:
                plt.scatter([i for i in range(len(v))],v,s=5,color='black')
            plt.xlabel('PanNuke class')
            plt.ylabel('Probability')
            plt.show()

        # node featue definition
        node_feature = area_feature.mean(axis=0).astype(float) # sum of the feature is corrected to 1
        return node_feature
    
    def conduct(self):
        cell_number = self.inst_map.max()
        for cell in tqdm(range(1,cell_number+1)):
            node_feature = self.single_extractor(cell=cell,remove_contour=True,do_plot=False)
            inst_info = self.nuc_info[str(cell)]
            inst_info['node_feature'] = list(node_feature) # add node feature information
    
    def savejson(self,save_json_path='/workspace/github/TopoPathology/feature_extractor/results/BRACS_291_IC_20_res.json'):
        with open(save_json_path, "w") as handle:
            json.dump(self.nuc_info, handle)