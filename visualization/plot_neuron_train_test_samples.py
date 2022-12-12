import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from data.map_dataset import MapDataset
from data.batch_sampler import BatchSampler

def plot_neuron_train_test_samples(single_worm_dataset, neuron_idx, 
                                   num_samples, seq_len, tau):
  '''
  Visualizes example train and test samples for a single neuron from the worm.
  Args:
    single_worm_dataset: dict, dataset for a single worm.
    neuron_idx: int, index in the neuron in the dataset.
    num_samples: int, the number of train (or test) examples to plot.
    seq_len: int, the length of the input (or target) time series.
    tau: int, the amount the target series is shifted by.
  '''
  calcium_data = single_worm_dataset['data']
  neuron_ids = single_worm_dataset['neuron_ids']
  max_time = single_worm_dataset['max_time']
  
  idx = neuron_idx
  nid = neuron_ids[idx]

  n_ex = num_samples
  yshifts = np.random.uniform(low=0.5, high=1.0, size=n_ex)
  
  seq_len = seq_len
  tau = tau
  eps = 0.05

  # datasets (only for visualizing)
  train_dataset = MapDataset(calcium_data[:max_time//2], tau=tau, seq_len=seq_len, size=n_ex, 
                            increasing=True, reverse=True)
  test_dataset = MapDataset(calcium_data[max_time//2:], tau=tau, seq_len=seq_len, size=n_ex, 
                            increasing=False, reverse=False)

  # dataloaders (only for visualizing)
  train_sampler = BatchSampler(train_dataset.batch_indices)
  train_loader = torch.utils.data.DataLoader(train_dataset, 
                                            batch_sampler=train_sampler)
  test_sampler = BatchSampler(test_dataset.batch_indices)
  test_loader = torch.utils.data.DataLoader(test_dataset, 
                                            batch_sampler=test_sampler)

  fig, axs = plt.subplots(1, 2, figsize=(20,5))

  plt.gca().set_prop_cycle(None)
  color = plt.cm.Paired(np.linspace(0, 1, 12))
  axs[0].set_prop_cycle(mpl.cycler(color=color))

  # training set
  trainX, trainY, metadata = next(iter(train_loader))
  batch_indices = np.random.choice(trainX.shape[0], n_ex, replace=False) # pick n example traces
  for _, batch_idx in enumerate(batch_indices):
    yshift = yshifts[_]
    # input sequence
    axs[0].plot(range(metadata['start'][batch_idx], metadata['end'][batch_idx]), 
                yshift + trainX[batch_idx, :, idx].cpu(), linewidth=2);
    # target sequence
    axs[0].plot(range(metadata['start'][batch_idx] + metadata['tau'][batch_idx], 
                      metadata['end'][batch_idx] + metadata['tau'][batch_idx]),  
                    eps + yshift + trainY[batch_idx, :, idx].cpu(), linewidth=1);

  axs[0].axvline(x=train_dataset.max_time, c='k', linestyle='--')
  axs[0].set_xlabel('time (s)')
  axs[0].set_yticks([])
  axs[0].set_ylabel('$\Delta F/F$ (random offsets)')
  axs[0].set_title('train set');

  plt.gca().set_prop_cycle(None)
  color = plt.cm.Paired(np.linspace(0, 1, 12))
  axs[1].set_prop_cycle(mpl.cycler(color=color))

  # test set
  testX, testY, metadata = next(iter(test_loader))
  batch_indices = np.random.choice(testX.shape[0], n_ex, replace=False) # pick n example traces
  for _, batch_idx in enumerate(batch_indices):
    yshift = yshifts[_]
    axs[1].plot(range(train_dataset.max_time + metadata['start'][batch_idx], 
                      train_dataset.max_time + metadata['end'][batch_idx]), 
                yshift + testX[batch_idx, :, idx].cpu(), linewidth=2);
    axs[1].plot(range(train_dataset.max_time + metadata['start'][batch_idx] + metadata['tau'][batch_idx], 
                      train_dataset.max_time + metadata['end'][batch_idx] + metadata['tau'][batch_idx]),  
                eps + yshift + testY[batch_idx, :, idx].cpu(), linewidth=1);

  axs[1].axvline(x=train_dataset.max_time, c='k', linestyle='--')
  axs[1].set_xlabel('time (s)')
  axs[1].set_yticks([])
  axs[1].set_ylabel('$\Delta F/F$ (random offsets)')
  axs[1].set_title('test set');

  miny = min(axs[0].get_ylim()[0], axs[1].get_ylim()[0])
  maxy = max(axs[0].get_ylim()[1], axs[1].get_ylim()[1])
  axs[0].set_ylim([miny, maxy])
  axs[1].set_ylim([miny, maxy])

  plt.suptitle("%s Example Samples of the Ca2+ Signal " 
              "from Neuron %s, $L$ = %s, τ = %s"%(n_ex, nid, seq_len, tau))
  plt.show()