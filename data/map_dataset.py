import torch
import numpy as np
from utils import DEVICE as device

class MapDataset(torch.utils.data.Dataset):
  '''
  A custom neural activity time-series prediction dataset.
  Using MapDataset will ensure that every sequence sample
  you generate is unique.
  '''
  def __init__(self, D, neurons=None, tau=1, seq_len=None, size=1000, 
               increasing=False, feature_mask=None):
    '''
    Initialization.
    Args:
      D: torch.tensor, data w/ shape (max_time, num_neurons, num_features).
      neurons: int or array-like, index of neuron(s) to return data for.
      tau: int, 0 <= tau < max_time//2.
      size: int, number of (input, target) data pairs to generate.
      seq_len: int or list, length of input/target sequences to generate.
      increasing: bool, whether to sample shorter sequences first.
      feature_mask: torch.tensor, what features to use.
    Returns:
      (X, Y, metadata): tuple, btach of data samples
        X: torch.tensor, input tensor w/ shape (batch_size, seq_len, num_neurons, num_features)
        Y: torch.tensor, target tensor w/ same shape as X
        metadata: dict, dictionary with information about samples, keys: 'seq_len', 'start' index , 'end' index
    '''
    super(MapDataset, self).__init__()
    self.seq_len = seq_len
    self.max_time, num_neurons, num_features = D.shape
    self.increasing = increasing

    if feature_mask is not None:
      assert len(feature_mask) == num_features, "`feature_mask` must have shape (%s, 1)."%num_features
      assert feature_mask.sum() > 0, "`feature_mask` must select at leat 1 feature."
      self.feature_mask = feature_mask  
    else:
      self.feature_mask = torch.tensor([1] + (num_features-1)*[0]).to(torch.bool)

    # enforce a constraint on using the neurons or signals as features
    if self.feature_mask.sum() == 1: # single signal
      if neurons is not None: 
        self.neurons = np.array(neurons) # use the subset of neurons given
      else: # neurons is None
        self.neurons = np.arange(num_neurons) # use all the neurons
    else: # multiple signals
      if neurons is not None: 
        assert np.array(neurons).size == 1, "only select 1 neuron when > 1 signals as features."
        self.neurons = np.array(neurons) # use the single neuron given
      else: 
        self.neurons = np.array([0]) # use the first neuron

    self.num_neurons = self.neurons.size
    # number of features equals: number of neurons if one signal; number of signals if multiple
    self.num_features = self.feature_mask.sum() if self.num_neurons == 1 else self.num_neurons
    self.D = D
    assert 0 <= tau < max_time//2, "`tau` must be  0 <= tau < max_time//2"
    self.tau = tau
    self.size = size
    self.counter = 0 
    self.data_samples, self.batch_indices = self.__data_generator()

  def __len__(self):
    '''Denotes the total number of samples.'''
    return len(self.data_samples)

  def __getitem__(self, index):
    '''Generates one sample of data.'''
    return self.data_samples[index]

  def __data_generator(self):
    '''
    Private method for generating all possible data samples.
    '''
    data_samples = []
    batch_indices = []
    # define length of time
    T = self.max_time
    # iterate over all seqeunce lengths, L, if a fixed one was not given
    seq_lens = [self.seq_len] if self.seq_len is not None else (
        range(1, T-self.tau+1) if self.increasing else range(T-self.tau, 0, -1)
        )
    for L in seq_lens:
      # a batch contains all data of a certain length
      batch = []
      # iterate over all start indices
      for start in range(0, T-L-self.tau+1):
        # define an end index
        end = start + L
        # data samples: input, X_tau and target, Y_tau
        X_tau = self.D[start:end, self.neurons, self.feature_mask].to(device)
        Y_tau = self.D[start+self.tau:end+self.tau, self.neurons, self.feature_mask].to(device)
        # store metadata about the sample
        tau = torch.tensor(self.tau).to(device)
        meta = {'seq_len': L, 'start': start, 'end': end, 'tau': tau}
        # append to data samples
        data_samples.append((X_tau, Y_tau, meta))
        # append index to batch
        batch.append(self.counter)
        self.counter += 1
        # we only want a number of samples up to self.size
        if self.counter >= self.size:
          break
      batch_indices.append(batch)
      # we only want a number of samples up to self.size
      if self.counter >= self.size:
        break
    # size of dataset
    self.size = self.counter
    
    return data_samples, batch_indices