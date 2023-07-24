# -*- coding: utf-8 -*-
"""
Created on 2023-07-24 (Mon) 15:38:19

@author: I.Azuma
"""
# %%
import random
import numpy as np

import torch

# %%
def fix_seed(seed:int=None,fix_gpu:bool=False):
    """ fix seed """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if fix_gpu:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True