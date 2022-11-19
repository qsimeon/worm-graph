import os
import torch
import numpy as np

# defines `worm_graph` as the root directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# get GPU if available
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# helper functions
def sliding_windows(dataset, seq_length):
    '''
    Function for creating 1-timestep ahead 
    prediction datasets.
    data: array-like w/ shape (neurons, time)
    seq_length: int s | s < time, previous time to regress on
    '''
    x = []
    y = []
    num_batches = dataset.shape[1]-seq_length-1
    for i in range(num_batches): # determines number of batches
        _x = dataset.T[i:(i+seq_length)] # time to regress on
        _y = dataset.T[i+seq_length] # next time-step(s)
        x.append(_x)
        y.append(_y)
    return np.array(x), np.array(y)