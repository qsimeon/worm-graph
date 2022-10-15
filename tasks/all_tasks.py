import torch
import copy
import numpy as np
from utils import sliding_windows

class GraphTask(object):
    '''This class takes in a graph in the form of a PyG Data object
    and returns a new instance of a graph that has the required features
    and targets for the particular self-supervised task.'''
    def __init__(self, graph):
        super(GraphTask, self).__init__()
        torch.manual_seed(2022)
        self.graph = graph



class OneStepPrediction(GraphTask):
    '''Returns StaticGraphTemporalSignal data object for one-time step prediction.'''
    def __init__(self, graph, dataset=None, max_time=1024, 
                    train_ratio=0.2, seq_length=9):
        super(OneStepPrediction, self).__init__(graph)
        self.dataset = dataset
        _, num_real = self.dataset.shape
        _, T = self.graph.x.numpy().shape
        print("How much real data do we have?", self.dataset.shape) # (time, neurons)
        print("Current data on connectome graph:", self.graph.x.numpy().shape) # (neurons, time)

        # replace data
        inds = np.random.choice(graph.num_nodes, size=num_real, replace=False)
        new_graph = copy.deepcopy(graph) # don't want to modify original graph
        new_graph.x = graph.x[:, :T]
        new_graph.x[inds, :] = torch.tensor(self.dataset.T[:, :T], dtype=torch.float32)


# class NStepsPrediction(GraphTask):

class NeuronClassication(GraphTask):
    '''Returns a graph object'''



