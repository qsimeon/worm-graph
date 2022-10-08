#@title Define an MNIST PointCloud as subclass of InMemoryDataset
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from torch_geometric.utils import coalesce
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
        # dataset adapted from from Cook et al. (2019) SI5
        url = 'https://github.com/metaconsciousgroup/worm-graph/raw/main/data/raw_data.zip' # base url
        filename = os.path.join('Herm_Nodes_Edges.zip')
        folder = os.path.join(self.raw_dir, 'datasets')
        download_url(url=url, folder=folder, filename=filename, ) # download zip file 
        extract_zip(filename, folder=self.raw_dir) # extract zip file
        os.unlink(filename) # remove zip file

    def process(self):
        # chemical synapse
        GHermChem_Edges = pd.read_csv('GHermChem_Edges.csv') # edges
        GHermChem_Nodes =  pd.read_csv('GHermChem_Nodes.csv') # nodes
        # gap junctions
        GHermElec_Sym_Edges = pd.read_csv('GHermElec_Sym_Edges.csv') # edges
        GHermElec_Sym_Nodes =  pd.read_csv('GHermElec_Sym_Nodes.csv') # nodes
        # neurons involved in gap junctions
        df = GHermElec_Sym_Nodes
        Ggap_nodes = df[df['Group'].str.contains("Neuron")].sort_values(by=['Name']).reset_index()
        # neurons involved in chemical synapses
        df = GHermChem_Nodes
        Gsyn_nodes = df[df['Group'].str.contains("Neuron")].sort_values(by=['Name']).reset_index()
        # gap junctions
        df = GHermElec_Sym_Edges
        inds = [i for i in GHermElec_Sym_Edges.index if 
                df.iloc[i]['EndNodes_1'] in set(Ggap_nodes.Name) and 
                df.iloc[i]['EndNodes_2'] in set(Ggap_nodes.Name)] # indices
        Ggap_edges = df.iloc[inds].reset_index(drop=True)
        # chemical synapses
        df = GHermChem_Edges
        inds = [i for i in GHermChem_Edges.index if 
                df.iloc[i]['EndNodes_1'] in set(Gsyn_nodes.Name) and 
                df.iloc[i]['EndNodes_2'] in set(Gsyn_nodes.Name)] # indices
        Gsyn_edges = df.iloc[inds].reset_index(drop=True)
        # map neuron names (IDs) to indices
        neuron_id  = dict(zip(Gsyn_nodes.Name.values, Gsyn_nodes.index.values))
        id_neuron = dict(zip(Gsyn_nodes.index.values, Gsyn_nodes.Name.values))
        # edge_index for gap junctions
        arr = Ggap_edges[['EndNodes_1', 'EndNodes_2']].values
        ggap_edge_index = torch.empty(*arr.shape, dtype=torch.long) 
        for i, row in enumerate(arr):
            ggap_edge_index[i,:] = torch.tensor([neuron_id[x] for x in row], dtype=torch.long)
        ggap_edge_index = ggap_edge_index.T # [2, num_edges]
        # edge_index for chemical synapses
        arr = Gsyn_edges[['EndNodes_1', 'EndNodes_2']].values
        gsyn_edge_index = torch.empty(*arr.shape, dtype=torch.long) 
        for i, row in enumerate(arr):
            gsyn_edge_index[i,:] = torch.tensor([neuron_id[x] for x in row], dtype=torch.long)
        gsyn_edge_index = gsyn_edge_index.T # [2, num_edges]
        num_edge_features = 2
        # edge_attr for gap junctions
        num_edges = len(Ggap_edges)
        ggap_edge_attr = torch.empty(num_edges, num_edge_features, 
                                    dtype=torch.float) # [num_edges, num_edge_features]
        for i, weight in enumerate(Ggap_edges.Weight.values):
            ggap_edge_attr[i,:] = torch.tensor([weight, 0],cdtype=torch.float) # electrical synapse encoded as [1,0]
        # edge_attr for chemical synapses
        num_edges = len(Gsyn_edges)
        gsyn_edge_attr = torch.empty(num_edges, num_edge_features, 
                             dtype=torch.float) # [num_edges, num_edge_features]
        for i, weight in enumerate(Gsyn_edges.Weight.values):
            gsyn_edge_attr[i,:] = torch.tensor([0, weight], dtype=torch.float) # chemical synapse encoded as [0,1]
        # data.x node feature matrix
        num_nodes = len(Gsyn_nodes)
        num_node_features = 1024
        # generate random data
        x = torch.rand(num_nodes, num_node_features, dtype=torch.float) # [num_nodes, num_node_features]
        # data.y target to train against
        le = preprocessing.LabelEncoder()
        le.fit(Gsyn_nodes.Group.values)
        y = torch.tensor(le.transform(Gsyn_nodes.Group.values), dtype=torch.float) # [num_nodes, 1]
        # graph for electrical connectivity
        electrical_graph = Data(x=x, edge_index=ggap_edge_index, edge_attr=ggap_edge_attr, y=y)
        # graph for chemical connectivity
        chemical_graph = Data(x=x, edge_index=gsyn_edge_index, edge_attr=gsyn_edge_attr, y=y)
        # merge electrical and chemical graphs into a single connectome graph
        edge_index = torch.hstack((electrical_graph.edge_index, chemical_graph.edge_index)) # features = [elec_wt, chem_wt]
        edge_attr = torch.vstack((electrical_graph.edge_attr, chemical_graph.edge_attr)) 
        edge_index, edge_attr = coalesce(edge_index, edge_attr, reduce="add")
        assert all(chemical_graph.y == electrical_graph.y), "Node labels not matched!"
        x = chemical_graph.x 
        y = chemical_graph.y
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        # add graph to datalist
        data_list.append(graph)
        # apply the functions specified in pre_filter and pre_transform
        data_list = [data for data in data_list if self.pre_filter(data)]
        data_list = [self.pre_transform(data) for data in data_list]
        # store the processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])  