#@title Define an MNIST PointCloud as subclass of InMemoryDataset
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from torch_geometric.utils import erdos_renyi_graph, dense_to_sparse
from torch_geometric.data import InMemoryDataset, Data, download_url, extract_zip
import torch_geometric.transforms as T

import os
import numpy as np
import networkx as nx
import pandas as pd
from tqdm import tqdm
from sklearn import preprocessing
from scipy.io import loadmat

class CElegansGraph(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, dense=False):
        self.dense = dense
        super(CElegansGraph, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])   
        
    @property
    def raw_file_names(self):
        ''' List of the raw files '''
        return ['datasets/GHermChem_Edges.csv', 'datasets/GHermChem_Nodes.csv', 
                'datasets/GHermElec_Sym_Edges.csv', 'datasets/GHermElec_Sym_Nodes.csv'] 
               
    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Michi's dataset
        url = 'https://github.com/metaconsciousgroup/worm-graph/raw/download/Herm_Nodes_Edges.zip' # base url
        filename = os.path.join('Herm_Nodes_Edges.zip')
        folder = os.path.join(self.raw_dir, 'datasets')
        download_url(url=url, folder=folder, filename=filename, ) # download zip file 
        extract_zip(filename, folder=self.raw_dir) # extract zip file
        os.unlink(filename) # remove zip file

    def process(self):
        # dataset adapted from from Cook et al. (2019) SI5
        train_path = os.path.join(self.raw_dir, 'Dataset', 'train.csv')
        test_path = os.path.join(self.raw_dir, 'Dataset', 'test.csv')
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
        query = df.loc[:, df.columns != 'label']
        xyz = np.expand_dims(query.to_numpy(), 0).reshape(len(df), -1, 3)
        xyz = [t[t[:,-1] != -1, :] for t in xyz] # remove null values
        xyz = [t / np.array([27, 27, 255]) for t in xyz] # normalize
        n_points = [t.shape[0] for t in xyz]
        df = pd.DataFrame({'xy': xyz, 'label': df.label, 'n_points': n_points})

        # processing
        data_list = []
        ids_list = df.index.unique()
        for g_idx in tqdm(ids_list):
          # node features
          n_points = df.iloc[g_idx].n_points
          node_ids = np.arange(n_points)
          x = torch.tensor(df.iloc[g_idx].xy, dtype=torch.float)
          pos = x.clone() # node position matrix

          # edges info
          I_n = torch.eye(n_points)  # only self connections
          edge_index, _ = dense_to_sparse(I_n)
          if self.dense: # random connectivity
            edge_index = erdos_renyi_graph(n_points, 0.1)
          
          # graph label
          label = df.loc[g_idx].label
          y = torch.tensor([label], dtype=torch.long)
          
          # construct graph as a Data object
          graph = Data(x=x, edge_index=edge_index, edge_attr=None, y=y)
          data_list.append(graph)
            
        # Apply the functions specified in pre_filter and pre_transform
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # Store the processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])  