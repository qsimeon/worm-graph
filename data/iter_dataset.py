import torch
import numpy as np
from utils import DEVICE as device

class IterDataset(torch.utils.data.IterableDataset):
  '''
  A custom neural activity time-series prediction dataset.
  Using IterDataset samples sequences randomly so samples may
  not be unique. However, IterDataset is faster than MapDatset.
  An iterable-style dataset is an instance of a subclass of IterableDataset 
  that implements the __iter__() protocol, and represents an iterable over data samples. 
  This type of datasets is particularly suitable for cases where random reads are expensive 
  or even improbable, and where the batch size depends on the fetched data.
  For example, such a dataset, when called iter(dataset), could return a stream of data 
  reading from a database, a remote server, or even logs generated in real time.
  '''
  def __init__(self, D, neurons=None, tau=1, seq_len=None, size=1000, 
               feature_mask=None):
    '''
    Args:
      D: torch.tensor, data w/ shape (max_time, num_neurons, num_features).
      neurons: int or array-like, index of neuron(s) to return data for.
      tau: int, 0 <= tau < max_time//2.
      seq_len: None or int, if specified only generate sequences of this length.
      size: int, number of (input, target) data pairs to generate.
      feature_mask: torch.tensor, what features to use.
    Returns:
      (X, Y, meta): tuple, batch of data samples
        X: torch.tensor, input tensor w/ shape (batch_size, seq_len, num_neurons, num_features)
        Y: torch.tensor, target tensor w/ same shape as X
        meta: dict, metadata / information about samples, keys: 'seq_len', 'start' index , 'end' index
    '''
    super(IterDataset, self).__init__()
    self.seq_len = seq_len
    self.max_time, num_neurons, num_features = D.shape

    if feature_mask is not None:
      assert len(feature_mask) == num_features, '`feature_mask` must have shape (%s, 1).'%num_features
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
        assert np.array(neurons).size == 1, 'only select 1 neuron when > 1 signals as features.'
        self.neurons = np.array(neurons) # use the single neuron given
      else: 
        self.neurons = np.array([0]) # use the first neuron

    self.num_neurons = self.neurons.size
    # number of features equals: number of neurons if one signal; number of signals if multiple
    self.num_features = self.feature_mask.sum() if self.num_neurons == 1 else self.num_neurons
    self.D = D
    assert 0 <= tau < self.max_time//2, '`tau` must be  0 <= tau < max_time//2'
    self.tau = tau
    self.size = size

  def __len__(self):
    '''Denotes the total number of samples.'''
    return self.size

  def __iter__(self):
    '''Must implement if the dataset represents an interable of data.'''
    # return iterator over data samples
    return self.data_generator()

  def data_generator(self):
    '''
    Helper function for creating a data iterator.
    '''
    torch.manual_seed(0)
    for i in range(self.size):
      # define length of time
      T = self.max_time
      # sample a sequence length, L, if a fixed one was not given
      L = self.seq_len if self.seq_len is not None else np.random.randint(
          low=1, high=T-self.tau+1, dtype=int) 
      # sample a start index
      start = np.random.randint(low=0, high=T-L-self.tau+1, dtype=int)
      # define an end index
      end = start + L
      # data samples: input, X_tau and target, Y_tau
      X_tau = self.D[start:end, self.neurons, self.feature_mask].to(device)
      Y_tau = self.D[start+self.tau:end+self.tau, self.neurons, self.feature_mask].to(device)
      # also return some metadata
      tau = torch.tensor(self.tau).to(device)
      meta = {'seq_len': L, 'start': start, 'end': end, 'tau': tau}
      # yield a single data sample
      yield X_tau, Y_tau, meta