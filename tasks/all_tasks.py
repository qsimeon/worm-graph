import os
import torch
import copy
import numpy as np
from utils import sliding_windows
from torch_geometric_temporal.signal import temporal_signal_split
from torch_geometric_temporal.signal.static_graph_temporal_signal import StaticGraphTemporalSignal
from data.connectome_dataset import CElegansDataset

class GraphTask(object):
    '''This class takes in a graph in the form of a PyG Data object
    and returns a new instance of a graph that has the required features
    and targets for the particular self-supervised task.'''
    def __init__(self, graph):
        super(GraphTask, self).__init__()
        torch.manual_seed(2022)
        self.graph = graph

class OneStepPrediction(GraphTask):
    '''
    Returns constructor for a StaticGraphTemporalSignal data iterator.
    StaticGraphTemporalSignalBatch is designed for temporal signals defined on a static graph.
    https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/signal.html#id1 
    '''
    def __init__(self, graph, dataset=None, max_time=None, seq_length=9, train_ratio=0.5):
        '''
        graph: instance of CElegansDataset.
        dataset: dataset to inject on graph; array-like w/ shape (neurons, time).
        '''
        super(OneStepPrediction, self).__init__(graph)

        # parse the dataset to impose on the graph
        if dataset is None:
            dataset = copy.deepcopy(self.graph.x) # graphs initialized with random data
        else:
            assert dataset.shape[0] < self.graph.num_nodes and dataset.shape[0] < dataset.shape[1],  "Reshape neural data as (neurons, time)!"
            dataset = torch.tensor(dataset, dtype=torch.float)
        num_neurons, len_time = dataset.shape

        # inject the data into the nodes of the graph
        new_graph = copy.deepcopy(graph) # don't modify original graph
        if max_time is None: # length of time series
            self.max_time = len_time
        else:
            self.max_time = min(max_time, len_time)
        new_graph.x = torch.rand(self.graph.num_nodes, self.max_time, dtype=torch.float)
        if dataset is not None:
            inds = np.random.choice(self.graph.num_nodes, size=num_neurons, replace=False) # neuron indices
            new_graph.x[inds, :] = dataset[:, :max_time]

        # use the new graph
        self.graph = new_graph

        # adapt node features and edge weights 
        self.seq_length = seq_length
        Xs, Ys = sliding_windows(new_graph.x.clone().detach().numpy(), self.seq_length)
        features = list(Xs.transpose((0,2,1)))
        targets = list(Ys.squeeze())
        edge_index = self.graph.edge_index.clone().detach().numpy()
        edge_weight = self.graph.edge_attr.clone().detach().sum(axis=1).numpy()

        # create a StaticGraphTemporalSignal data iterator
        self.temporal_dataset = StaticGraphTemporalSignal(edge_index, edge_weight, features, targets)
        
        # split into train and test sets
        self.train_ratio = train_ratio
        self.train_dataset, self.test_dataset = temporal_signal_split(self.temporal_dataset, train_ratio=self.train_ratio)

    def __call__(self):
        ''''Returns instance of StaticGraphTemporalSignal.'''
        return self.temporal_dataset
    
    def train_test_split(self):
        return self.train_dataset, self.test_dataset
    
    @property
    def node_count(self):
        return self.graph.num_nodes

    @property
    def node_features(self):
        return self.seq_length

    @property
    def train_size(self):
        return self.train_dataset.snapshot_count
        
    @property
    def test_size(self):
        return self.test_dataset.snapshot_count
    
    @property 
    def data_size(self):
        '''Returns the number of snapshots. 
        Each snapshot is s a Pytorch Geometric Data object.'''
        return self.temporal_dataset.snapshot_count
        

if __name__ == "__main__":
    # run some tasks
    graph = CElegansDataset()[0]
    task = OneStepPrediction(graph)
    task = OneStepPrediction(graph, seq_length=3)
    task = OneStepPrediction(graph, max_time=1000, train_ratio=0.3)
    print("Built one-step prediction task successfully!")