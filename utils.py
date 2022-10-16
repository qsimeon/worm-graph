import os
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def sliding_windows(data, seq_length, pred_length=1):
    '''
    Function for creating 1-timestep ahead 
    prediction datasets.
    data: numpy.ndarray of shape (time, neurons)
    seq_length: int: s | s < time, previous time to regress on
    pred_length: int : p | p < s < time, future time to predict
    '''
    x = []
    y = []
    num_batches = len(data)-seq_length-1
    for i in range(num_batches): # determines number of batches
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length:(i+seq_length+pred_length)] # next time-step(s)
        x.append(_x)
        y.append(_y)
    return np.array(x), np.array(y)