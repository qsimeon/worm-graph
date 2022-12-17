import torch
from utils import DEVICE as device


def add_train_val_mask(graph, train_ratio=0.7, train_mask=None):
    """
    Mutates a C. elegans connectome graph with injected data
    to include a training and validation mask.
    Retruns the graph with `train_mask` and `val_mask` masks added 
    as attributes.
    """
    # create the train and validation masks
    if train_mask is not None:
        assert train_mask.ndim==1 and train_mask.size(0)==graph.num_nodes, "Invalid train_mask provided."
    else:
        train_mask = torch.rand(graph.num_nodes) < train_ratio
    val_mask = ~train_mask
    # make the train and test masks attributes of the data graph
    graph.train_mask = train_mask
    graph.val_mask = val_mask
    # put graph on GPU
    graph = graph.to(device)
    # return the graph with train and validation masks
    return graph