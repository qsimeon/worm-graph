import torch
from torch_geometric.data import Data
from data.map_dataset import MapDataset
from data.batch_sampler import BatchSampler
from data.load_connectome import CElegansDataset
from data.load_neural_activity import load_Uzel2022
from torch_geometric_temporal.signal import temporal_signal_split
from torch_geometric_temporal.signal.static_graph_temporal_signal import StaticGraphTemporalSignal

class GraphTask(object):
    '''
    This class takes in a graph in the form of a PyG Data object
    and returns an an iterator that has the required features and
    targets for the particular self-supervised task.
    Args:
        graph: CElegansDataset, a single worm graph.
        seq_len: int, the length of input, target sequences to use.
    '''
    def __init__(self, graph, seq_len=11):
        super(GraphTask, self).__init__()
        torch.manual_seed(2022)
        self.graph = graph
        self.seq_len = seq_len

class OneStepPrediction(GraphTask):
    '''
    Returns constructor for a StaticGraphTemporalSignal data iterator.
    StaticGraphTemporalSignalBatch is designed for temporal signals defined on a static graph.
    https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/signal.html#id1 
    '''
    def __init__(self, graph, dataset, neuron_ids):
        '''
        graph: CElegansDataset, a single worm graph.
        dataset: torch.tensor, dataset of single signal to inject on graph w/ (time, neurons).
        neuron_ids: dict, mapping of neuron_names to indices of the dataset.
        '''
        
        super(OneStepPrediction, self).__init__(graph)
        # parse the dataset to impose on the graph
        dataset = dataset.squeeze()
        # check input sizes
        assert dataset.ndim == 2 and dataset.shape[1] < dataset.shape[0] and \
            dataset.shape[1] <= self.graph.num_nodes, "Reshape neural data as (time, neurons)!"
        assert len(neuron_ids) <= dataset.shape[1]
        # find the graph nodes matching the neurons in the dataset 
        graph_inds = [k-1 for k,v in graph.id_neuron.items() if v in set(neuron_ids.values())] # neuron indices in connectome
        data_inds = [k_-1 for k_,v_ in neuron_ids.items() if v_ in set(graph.id_neuron.values())] # neuron indices in sparse dataset
        # inject the data into the nodes of the graph
        new_x = torch.zeros(graph.num_nodes, dataset.shape[0], dtype=torch.float64)
        new_x[graph_inds, :] = dataset[:, data_inds].T
        # don't modify original graph
        self.graph = Data(x=new_x, y=graph.y, edge_index=graph.edge_index, edge_attr=graph.edge_attr, 
                        node_type=graph.node_type, pos=graph.pos, id_neuron=graph.id_neuron)
        # generate many/all sequences of length seq_len
        __ = MapDataset(self.graph.x.T.unsqueeze(-1), tau=1, seq_len=self.seq_len, size=5000, increasing=False)
        loader = torch.utils.data.DataLoader(__, batch_sampler=BatchSampler(__.batch_indices))
        Xs, Ys, _ = next(iter(loader))
        # adapt node features and edge weights 
        features = [arr.detach().numpy().T for arr in Xs]
        targets = [arr.detach().numpy().T for arr in Ys.squeeze()]
        edge_index = self.graph.edge_index.clone().detach().numpy()
        edge_weight = self.graph.edge_attr.clone().detach().sum(axis=1).numpy()
        # create a StaticGraphTemporalSignal data iterator
        self.temporal_dataset = StaticGraphTemporalSignal(edge_index, edge_weight, features, targets)
        # split into train and test sets
        self.train_dataset, self.test_dataset = temporal_signal_split(self.temporal_dataset, train_ratio=0.5)

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
    # run one-step-prediction task
    graph = CElegansDataset()[0]
    Uzel2022 = load_Uzel2022(); single_worm_dataset = Uzel2022['worm1']
    dataset, neuron_ids = single_worm_dataset['data'], single_worm_dataset['neuron_ids']
    task = OneStepPrediction(graph, dataset, neuron_ids)
    print("Built one-step prediction task successfully!")