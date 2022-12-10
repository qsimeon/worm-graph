import os
import torch
from utils import ROOT_DIR
import numpy as np
import pandas as pd
from sklearn import preprocessing
from torch_geometric.data import Data
from torch_geometric.utils import coalesce

def preprocess(raw_dir, raw_files):
    '''If the `graph_tensors.pt` file is not found, this function gets 
    called to create it from the raw open source connectome data.
    The connectom data useed here is from Cook et al., 2019  downloaded from 
    https://wormwiring.org/matlab%20scripts/Premaratne%20MATLAB-ready%20files%20.zip.
    The data in .mat files was processed into .csv files containing the chemical and electrical
    connectivity for the adult hermaphrodite C. elegans.'''
    # checking
    assert all([os.path.exists(os.path.join(raw_dir, rf)) for rf in raw_files])
    # list of names of all C. elegans neurons
    neuron_names = set(pd.read_csv(os.path.join(raw_dir, 'neuron_names.txt'), sep=' ', header=None, names=['neuron']).neuron)
    #TODO: Use a better connectome! This one from Cook et al., 2019 is very inaccurate
    # chemical synapses
    GHermChem_Edges = pd.read_csv(os.path.join(raw_dir, 'GHermChem_Edges.csv')) # edges
    GHermChem_Nodes =  pd.read_csv(os.path.join(raw_dir, 'GHermChem_Nodes.csv')) # nodes
    # gap junctions
    GHermElec_Sym_Edges = pd.read_csv(os.path.join(raw_dir, 'GHermElec_Sym_Edges.csv')) # edges
    GHermElec_Sym_Nodes =  pd.read_csv(os.path.join(raw_dir, 'GHermElec_Sym_Nodes.csv')) # nodes
    # neurons involved in gap junctions
    df = GHermElec_Sym_Nodes
    df['Name'] = [v.replace('0','') if not v.endswith('0') else v for v in df['Name']]
    Ggap_nodes = df[df['Name'].isin(neuron_names)].sort_values(by=['Name']).reset_index()
    # neurons involved in chemical synapses
    df = GHermChem_Nodes
    df['Name'] = [v.replace('0','') if not v.endswith('0') else v for v in df['Name']]
    Gsyn_nodes = df[df['Name'].isin(neuron_names)].sort_values(by=['Name']).reset_index()
    # gap junctions
    df = GHermElec_Sym_Edges
    df['EndNodes_1'] = [v.replace('0','') if not v.endswith('0') else v for v in df['EndNodes_1']]
    df['EndNodes_2'] = [v.replace('0','') if not v.endswith('0') else v for v in df['EndNodes_2']]
    inds = [i for i in GHermElec_Sym_Edges.index if 
            df.iloc[i]['EndNodes_1'] in set(Ggap_nodes.Name) and 
            df.iloc[i]['EndNodes_2'] in set(Ggap_nodes.Name)] # indices
    Ggap_edges = df.iloc[inds].reset_index(drop=True)
    # chemical synapses
    df = GHermChem_Edges
    df['EndNodes_1'] = [v.replace('0','') if not v.endswith('0') else v for v in df['EndNodes_1']]
    df['EndNodes_2'] = [v.replace('0','') if not v.endswith('0') else v for v in df['EndNodes_2']]
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
    # edge attributes
    num_edge_features = 2
    # edge_attr for gap junctions
    num_edges = len(Ggap_edges)
    ggap_edge_attr = torch.empty(num_edges, num_edge_features, 
                                dtype=torch.float) # [num_edges, num_edge_features]
    for i, weight in enumerate(Ggap_edges.Weight.values):
        ggap_edge_attr[i,:] = torch.tensor([weight, 0], dtype=torch.float) # electrical synapse encoded as [1,0]
    # edge_attr for chemical synapses
    num_edges = len(Gsyn_edges)
    gsyn_edge_attr = torch.empty(num_edges, num_edge_features, 
                            dtype=torch.float) # [num_edges, num_edge_features]
    for i, weight in enumerate(Gsyn_edges.Weight.values):
        gsyn_edge_attr[i,:] = torch.tensor([0, weight], dtype=torch.float) # chemical synapse encoded as [0,1]
    # data.x node feature matrix
    num_nodes = len(Gsyn_nodes)
    num_node_features = 1024
    # generate random data TODO: inject real data istead !
    x = torch.rand(num_nodes, num_node_features, dtype=torch.float) # [num_nodes, num_node_features]
    # data.y target to train against
    le = preprocessing.LabelEncoder()
    le.fit(Gsyn_nodes.Group.values)
    y = torch.tensor(le.transform(Gsyn_nodes.Group.values), dtype=torch.float) # [num_nodes, 1]
    # save the mapping of encodings to type of neuron
    codes = np.unique(y)
    types = np.unique(Gsyn_nodes.Group.values)
    node_type = dict(zip(codes, types))
    # graph for electrical connectivity
    electrical_graph = Data(x=x, edge_index=ggap_edge_index, edge_attr=ggap_edge_attr, y=y)
    # graph for chemical connectivity
    chemical_graph = Data(x=x, edge_index=gsyn_edge_index, edge_attr=gsyn_edge_attr, y=y)
    # merge electrical and chemical graphs into a single connectome graph
    edge_index = torch.hstack((electrical_graph.edge_index, chemical_graph.edge_index)) 
    edge_attr = torch.vstack((electrical_graph.edge_attr, chemical_graph.edge_attr)) 
    edge_index, edge_attr = coalesce(edge_index, edge_attr, reduce="add") # features = [elec_wt, chem_wt]
    assert all(chemical_graph.y == electrical_graph.y), "Node labels not matched!"
    x = chemical_graph.x 
    y = chemical_graph.y
    # basic attributes of PyG Data object
    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    # add some additional attributes to the graph
    neuron_names = list(id_neuron.values())
    df = pd.read_csv(os.path.join(raw_dir, 'LowResAtlasWithHighResHeadsAndTails.csv'), 
                    header=None, names=['neuron', 'x', 'y', 'z'])
    df = df[df.neuron.isin(neuron_names)]
    valids = set(df.neuron)
    keys = [k for k in id_neuron if id_neuron[k] in valids]
    values = list(df[df.neuron.isin(valids)][['x', 'z']].values)
    # initialize position dict then replace with atlas coordinates if available
    pos = dict(zip(np.arange(graph.num_nodes), np.zeros(shape=(graph.num_nodes, 2))))
    for k,v in zip(keys, values): pos[k] = v
    # assign each node its global node index
    n_id = torch.arange(graph.num_nodes)
    # save the tensors to use as raw data in the future.
    graph_tensors = {'edge_index': edge_index, 'edge_attr': edge_attr, 'pos': pos,
                 'x': x, 'y':  y, 'id_neuron': id_neuron, 'node_type': node_type, 'n_id': n_id}
    torch.save(graph_tensors, os.path.join(ROOT_DIR, 'preprocessing', 'graph_tensors.pt'))