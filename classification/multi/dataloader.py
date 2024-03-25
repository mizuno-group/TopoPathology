# -*- coding: utf-8 -*-
"""
Created on 2023-07-17 (Mon) 00:58:03

dataloader for BRACS multi classification

@author: I.Azuma
"""
#%%
import os
import h5py

import torch
from torch_geometric.data import Data,DataLoader
from torch.utils.data import Dataset
import torch.utils.data

from glob import glob
from tqdm import tqdm
import numpy as np

import dgl
from dgl.data.utils import load_graphs

from _utils import utils

IS_CUDA = torch.cuda.is_available()
DEVICE = 'cuda:0' if IS_CUDA else 'cpu'
COLLATE_FN = {
    'DGLGraph': lambda x: dgl.batch(x),
    'Tensor': lambda x: x,
    'int': lambda x: torch.LongTensor(x).to(DEVICE)
}

#%%
def h5_to_tensor(h5_path):
    h5_object = h5py.File(h5_path, 'r')
    out = torch.from_numpy(np.array(h5_object['assignment_matrix']))
    return out

class BRACSDataset(Dataset):
    """BRACS dataset."""

    def __init__(
            self,
            cg_path_list: list = None,
            tg_path_list: list= None,
            assign_mat_path: str = None,
            load_in_ram: bool = False,
    ):
        """
        BRACS dataset constructor.

        Args:
            cg_path (str, optional): Cell Graph path to a given split (eg, ['/workspace/mnt/data1/Azuma/Pathology/centroids_based_graph/0_N/cell_graphs/train','/workspace/mnt/data1/Azuma/Pathology/centroids_based_graph/3_ADH/cell_graphs/train','/workspace/mnt/data1/Azuma/Pathology/centroids_based_graph/6_IC/cell_graphs/train']). Defaults to None.
            tg_path (str, optional): Tissue Graph path. Defaults to None.
            assign_mat_path (str, optional): Assignment matrices path. Defaults to None.
            load_in_ram (bool, optional): Loading data in RAM. Defaults to False.
        """
        super(BRACSDataset, self).__init__()

        assert not (cg_path_list is None and tg_path_list is None), "You must provide path to at least 1 modality."

        self.cg_path_list = cg_path_list
        self.tg_path_list = tg_path_list
        self.assign_mat_path = assign_mat_path
        self.load_in_ram = load_in_ram

        if cg_path_list is not None:
            self._load_cg()

        if tg_path_list is not None:
            self._load_tg()

        if assign_mat_path is not None:
            self._load_assign_mat()

    def _load_cg(self):
        """
        Load cell graphs
        """
        cg_fnames = []
        for p in self.cg_path_list:
            cg_fnames.extend(list(glob(os.path.join(p, '*.bin'))))
        self.cg_fnames = sorted(cg_fnames)

        self.num_cg = len(self.cg_fnames)
        if self.load_in_ram:
            cell_graphs = [load_graphs(fname) for fname in self.cg_fnames]
            self.cell_graphs = [entry[0][0] for entry in cell_graphs]
            self.cell_graph_labels = [entry[1]['label'].item() for entry in cell_graphs]
    
    def _load_tg(self):
        """
        Load tissue graphs
        """
        tg_fnames = []
        for p in self.tg_path_list:
            tg_fnames.extend(list(glob(os.path.join(p, '*.bin'))))
        """
        # grade selection # FIXME: binary classification
        final_tg_fnames = [] 
        for t in tg_fnames:
            if '_ADH_' in t:
                final_tg_fnames.append(t)
            elif '_UDH_' in t:
                final_tg_fnames.append(t)
            else:
                pass
        self.tg_fnames = sorted(final_tg_fnames)
        """
        self.tg_fnames = sorted(tg_fnames)
        self.num_tg = len(self.tg_fnames)
        if self.load_in_ram:
            tissue_graphs = [load_graphs(fname) for fname in self.tg_fnames]
            self.tissue_graphs = [entry[0][0] for entry in tissue_graphs]
            self.tissue_graph_labels = [entry[1]['label'].item() for entry in tissue_graphs]

    def _load_assign_mat(self):
        """
        Load assignment matrices 
        """
        self.assign_fnames = glob(os.path.join(self.assign_mat_path, '*.h5'))
        self.assign_fnames.sort()
        self.num_assign_mat = len(self.assign_fnames)
        if self.load_in_ram:
            self.assign_matrices = [
                h5_to_tensor(os.path.join(self.assign_mat_path, fname)).float().t()
                    for fname in self.assign_fnames
            ]

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

            cg = utils.set_graph_on_cuda(cg) if IS_CUDA else cg
            tg = utils.set_graph_on_cuda(tg) if IS_CUDA else tg
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
            tg = utils.set_graph_on_cuda(tg) if IS_CUDA else tg
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
            cg = utils.set_graph_on_cuda(cg) if IS_CUDA else cg
            return cg, label

    def __len__(self):
        """Return the number of samples in the BRACS dataset."""
        if hasattr(self, 'num_cg'):
            return self.num_cg
        else:
            return self.num_tg

def collate(batch):
    """
    Collate a batch.
    Args:
        batch (torch.tensor): a batch of examples.
    Returns:
        data: (tuple)
        labels: (torch.LongTensor)
    """
    def collate_fn(batch, id, type):
        return COLLATE_FN[type]([example[id] for example in batch])

    # collate the data
    num_modalities = len(batch[0])  # should 2 if CG or TG processing or 4 if HACT
    batch = tuple([collate_fn(batch, mod_id, type(batch[0][mod_id]).__name__)
                  for mod_id in range(num_modalities)])

    return batch

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
def collect_data(bin_path='graph/bin/path/e.g./train/*.bin',target_label=[0,3,6]):
    l = glob(bin_path)
    data_list = []
    for path in tqdm(l):
        g_list, label_dict = load_graphs(path)
        label = label_dict['label']
        if label in target_label:
            edge_info = g_list[0].edges()
            node_feature = g_list[0].ndata['feat']

            x = torch.tensor(node_feature, dtype=torch.float)
            edge_index = torch.tensor([np.array(edge_info[0]),np.array(edge_info[1])])

            new_label = target_label.index(label)
            d = Data(x=x,edge_index=edge_index.contiguous(),t=new_label)
            data_list.append(d)
        else:
            pass
    
    return data_list

def collect_multi_data(path_list=['/workspace/mnt/data1/Azuma/Pathology/centroids_based_graph/0_N/cell_graphs/train','/workspace/mnt/data1/Azuma/Pathology/centroids_based_graph/3_ADH/cell_graphs/train','/workspace/mnt/data1/Azuma/Pathology/centroids_based_graph/6_IC/cell_graphs/train'],original_label=[0,3,6]):
    data_list = []
    original_label = sorted(original_label)
    for path in path_list:
        bin_path = path+'/*.bin'
        l = glob(bin_path)
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