import os, sys
import h5py
import torch
import torch_geometric
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm
from torch_geometric.nn import global_mean_pool, global_max_pool
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GENConv, GraphConv, JumpingKnowledge, GATv2Conv, GINConv
# from torch.nn import Sequential, ReLU, Linear, Embedding
# from torch_geometric.nn import JumpingKnowledge, Set2Set, BatchNorm, GraphNorm, LayerNorm
from torch_geometric.loader import DataLoader
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedKFold
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import time
import random
import shutil
from model import Net
from torch.nn import Linear, BatchNorm1d, ReLU, Dropout
# from Focal_loss import *
# from radam import RAdam

class Dataset(InMemoryDataset):
    def __init__(self, root, org_h5py_source, transform=None, pre_transform=None):
        super(Dataset, self).__init__(root, transform, pre_transform)
        self.org_h5py_source = org_h5py_source
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['TrainingData.pt']

    def _download(self):
        pass

    def process(self):
        data_list = []
        for file in tqdm(os.listdir(self.org_h5py_source)):
            with h5py.File(os.path.join(self.org_h5py_source, file), 'r') as f:
                X = f['X'][()]
                eI = f['eI'][()]
                eAttr = f['edge_weight'][()]
                y = f['y'][()]
            X = torch.tensor(X, dtype=torch.float32)
            eI = torch.tensor(eI, dtype=torch.long)
            y = torch.tensor(y, dtype=torch.long)
            eAttr = torch.tensor(eAttr, dtype=torch.float32)
            data = Data(x=X, edge_index=eI, edge_attr=eAttr, y=y)
            data_list.append(data)
        data, slice = self.collate(data_list)
        torch.save((data, slice), self.processed_paths[0])
