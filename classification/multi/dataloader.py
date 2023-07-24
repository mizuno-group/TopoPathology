# -*- coding: utf-8 -*-
"""
Created on 2023-07-17 (Mon) 00:58:03

dataloader for BRACS multi classification

@author: I.Azuma
"""
#%%
import torch
from torch_geometric.data import Data,DataLoader

import glob
from tqdm import tqdm
import numpy as np

from dgl.data.utils import load_graphs

#%% 
class BRACSDataset(Dataset):
    """BRACS dataset."""

    def __init__(
            self,
            cg_path: str = None,
            tg_path: str = None,
            assign_mat_path: str = None,
            load_in_ram: bool = False,
    ):
        """
        BRACS dataset constructor.

        Args:
            cg_path (str, optional): Cell Graph path to a given split (eg, cell_graphs/test/). Defaults to None.
            tg_path (str, optional): Tissue Graph path. Defaults to None.
            assign_mat_path (str, optional): Assignment matrices path. Defaults to None.
            load_in_ram (bool, optional): Loading data in RAM. Defaults to False.
        """
        super(BRACSDataset, self).__init__()

        assert not (cg_path is None and tg_path is None), "You must provide path to at least 1 modality."

        self.cg_path = cg_path
        self.tg_path = tg_path
        self.assign_mat_path = assign_mat_path
        self.load_in_ram = load_in_ram

        if cg_path is not None:
            self._load_cg()


    def _load_cg(self):
        """
        Load cell graphs
        """
        self.cg_fnames = glob(os.path.join(self.cg_path, '*.bin'))
        self.cg_fnames.sort()
        self.num_cg = len(self.cg_fnames)
        if self.load_in_ram:
            cell_graphs = [load_graphs(os.path.join(self.cg_path, fname)) for fname in self.cg_fnames]
            self.cell_graphs = [entry[0][0] for entry in cell_graphs]
            self.cell_graph_labels = [entry[1]['label'].item() for entry in cell_graphs]


    def __getitem__(self, index):
        """
        Get an example.
        Args:
            index (int): index of the example.
        """

        # 1. HACT configuration
        if hasattr(self, 'num_tg') and hasattr(self, 'num_cg'):
            if self.load_in_ram:
                cg = self.cell_graphs[index]
                tg = self.tissue_graphs[index]
                assign_mat = self.assign_matrices[index]
                assert self.cell_graph_labels[index] == self.tissue_graph_labels[index], "The CG and TG are not the same. There was an issue while creating HACT."
                label = self.cell_graph_labels[index]
            else:
                cg, label = load_graphs(self.cg_fnames[index])
                cg = cg[0]
                label = label['label'].item()
                tg, _ = load_graphs(self.tg_fnames[index])
                tg = tg[0]
                assign_mat = h5_to_tensor(self.assign_fnames[index]).float().t()

            cg = set_graph_on_cuda(cg) if IS_CUDA else cg
            tg = set_graph_on_cuda(tg) if IS_CUDA else tg
            assign_mat = assign_mat.cuda() if IS_CUDA else assign_mat

            return cg, tg, assign_mat, label

        # 2. TG-GNN configuration 
        elif hasattr(self, 'num_tg'):
            if self.load_in_ram:
                tg = self.tissue_graphs[index]
                label = self.tissue_graph_labels[index]
            else:
                tg, label = load_graphs(self.tg_fnames[index])
                label = label['label'].item()
                tg = tg[0]
            tg = set_graph_on_cuda(tg) if IS_CUDA else tg
            return tg, label

        # 3. CG-GNN configuration 
        else:
            if self.load_in_ram:
                cg = self.cell_graphs[index]
                label = self.cell_graph_labels[index]
            else:
                cg, label = load_graphs(self.cg_fnames[index])
                label = label['label'].item()
                cg = cg[0]
            cg = set_graph_on_cuda(cg) if IS_CUDA else cg
            return cg, label

    def __len__(self):
        """Return the number of samples in the BRACS dataset."""
        if hasattr(self, 'num_cg'):
            return self.num_cg
        else:
            return self.num_tg

def make_data_loader(
        batch_size,
        shuffle=True,
        num_workers=0,
        **kwargs
    ):
    """
    Create a BRACS data loader.
    """

    dataset = BRACSDataset(**kwargs)
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate
        )

    return dataloader

# %% original for conducting simple graph neural networks

def collect_data(bin_path='/workspace/Pathology_Graph/230712_cg_classification/results/centroids_based_graph/cell_graphs/train/*.bin'):
    l = glob.glob(bin_path)
    data_list = []
    for path in tqdm(l):
        g_list, label_dict = load_graphs(path)
        edge_info = g_list[0].edges()
        node_feature = g_list[0].ndata['feat']

        x = torch.tensor(node_feature, dtype=torch.float)
        edge_index = torch.tensor([np.array(edge_info[0]),np.array(edge_info[1])])
        label = label_dict['label']
        if int(label) == 6:
            label = 1
        else:
            label = 0
        d = Data(x=x,edge_index=edge_index.contiguous(),t=label)
        data_list.append(d)
    
    return data_list

def collect_multi_data(path_list=['/workspace/mnt/data1/Azuma/Pathology/centroids_based_graph/0_N/cell_graphs/train','/workspace/mnt/data1/Azuma/Pathology/centroids_based_graph/3_ADH/cell_graphs/train','/workspace/mnt/data1/Azuma/Pathology/centroids_based_graph/6_IC/cell_graphs/train'],original_label=[0,3,6]):
    data_list = []
    original_label = sorted(original_label)
    for path in path_list:
        bin_path = path+'/*.bin'
        l = glob.glob(bin_path)
        for path in tqdm(l):
            g_list, label_dict = load_graphs(path)
            edge_info = g_list[0].edges()
            node_feature = g_list[0].ndata['feat']

            x = torch.tensor(node_feature, dtype=torch.float)
            edge_index = torch.tensor([np.array(edge_info[0]),np.array(edge_info[1])])
            label = original_label.index(label_dict['label'])
            d = Data(x=x,edge_index=edge_index.contiguous(),t=label)
            data_list.append(d)
    return data_list