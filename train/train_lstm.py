import torch
from utils import DEVICE as device
from data.map_dataset import MapDataset
from data.batch_sampler import BatchSampler
from models.rnn_models import NetworkLSTM
import numpy as np

#@title Pytorch style train and test pipelines.
#@markdown 

def train(loader, model, optimizer, no_grad=False):
  """
  Train a model given a dataset for a single epoch.
    Args:
        loader: training set dataloader
        model: instance of a NetworkLSTM
        optimizer: gradient descent optimizer with model params on it
        no_grad: bool, flag whether to turn of gradient computations
    Returns:
        losses: dict w/ keys train_loss and base_train_loss
  """
  model.train()
  criterion = model.loss_fn()
  base_loss, train_loss = 0, 0
  # Iterate in batches over the training dataset.
  for i, data in enumerate(loader): 
      X_train, Y_train, meta = data 
      tau = meta['tau'][0]
      # Baseline: loss if the model predicted the residual to be 0
      base = criterion(torch.zeros_like(Y_train), Y_train-X_train)/(tau+1)
      # Train
      if no_grad:
        with torch.no_grad():
          Y_tr = model(X_train) # Forward pass.
          loss = criterion(Y_tr, Y_train-X_train)/(tau+1) # Compute training loss.
      else:
        Y_tr = model(X_train) # Forward pass.
        loss = criterion(Y_tr, Y_train-X_train)/(tau+1) # Compute training loss.
        loss.backward()  # Derive gradients.
      optimizer.step()  # Update parameters based on gradients.
      optimizer.zero_grad()  # Clear gradients.
      # Store train and baseline loss.
      base_loss += base.detach().item()
      train_loss += loss.detach().item()
  # Average train and baseline losses
  losses = {'train_loss': train_loss/(i+1), 'base_train_loss': base_loss/(i+1)}
  # return mean train and baseline losses
  return losses


def test(loader, model):
  """
  Evaluate a model on a given dataset.
      loader: test/validation set dataloader
      model: instance of a NetworkLSTM
  Returns:
      losses: dict w/ keys test_loss and base_test_loss
  """
  model.eval() # this turns of grad
  criterion = model.loss_fn()
  base_loss, test_loss = 0, 0
  # Iterate in batches over the validation dataset.
  for i, data in enumerate(loader): 
    X_test, Y_test, meta = data
    tau = meta['tau'][0]
    # Baseline: loss if the model predicted the residual to be 0
    base = criterion(torch.zeros_like(Y_test), Y_test-X_test)/(tau+1)
    # Test
    with torch.no_grad():
      Y_te = model(X_test) # Forward pass.
      loss = criterion(Y_te, Y_test-X_test)/(tau+1) # Compute the validation loss.
    # Store test and baseline loss.
    base_loss += base.detach().item()
    test_loss += loss.detach().item()
  # Average test and baseline losses
  losses = {'test_loss': test_loss/(i+1), 'base_test_loss': base_loss/(i+1)}
  return losses


def optimize_model(dataset, model, num_epochs, seq_len=range(1,11,1)):
  """
  Creates train and test loaders given a task/dataset.
  Creates the optimizer given the model.
  Trains and validates the model for specified number of epochs.
  Returns a dict of epochs, and train, test and baseline losses.
  """
  # put model on device
  model = model.to(device)
  # create optimizer
  optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
  # create MapDatasets
  max_time = len(dataset)
  train_dataset = MapDataset(dataset[:max_time//2], tau=1, seq_len=seq_len, 
                           increasing=False, reverse=True, size=np.inf)
  test_dataset = MapDataset(dataset[max_time//2:], tau=1, seq_len=seq_len, 
                          increasing=False, reverse=True, size=np.inf)
  # create train and test loaders
  train_sampler = BatchSampler(train_dataset.batch_indices)
  train_loader = torch.utils.data.DataLoader(train_dataset, 
                                            batch_sampler=train_sampler)
  test_sampler = BatchSampler(test_dataset.batch_indices)
  test_loader = torch.utils.data.DataLoader(test_dataset, 
                                          batch_sampler=test_sampler)
  # create log dictionary to return
  log = {'base_train_losses': [], 'base_test_losses': [], 
         'train_losses': [], 'test_losses': [], 'epochs': []}
  # iterate over the training data multiple times
  for epoch in range(num_epochs+1):
    is_zeroth_epoch = epoch==0 # don't compute gradient for first pass
    # train the model
    train_log = train(train_loader, model, optimizer, no_grad=is_zeroth_epoch)
    test_log = test(test_loader, model)
    base_train_loss, train_loss = train_log['base_train_loss'], train_log['train_loss']
    base_test_loss, test_loss = test_log['base_test_loss'], test_log['test_loss']
    if epoch % (num_epochs//10) == 0:
      print(f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val. Loss: {test_loss:.4f}")
      log['epochs'].append(epoch)
      log['base_train_losses'].append(base_train_loss)
      log['base_test_losses'].append(base_test_loss)
      log['train_losses'].append(train_loss)
      log['test_losses'].append(test_loss)
  # return optimized model
  return model, log


def lstm_hidden_size_experiment(dataset, num_epochs, input_size, num_layers=1, 
                              hid_mult=np.array([3, 2]), seq_len=range(1,11,1)):
  """
  Helper function to experiment with different input sizes for the LSTM model.
  dataset: the dataset to train on.
  num_epochs: number of epochs to train for.
  input_size: number of input features (neurons).
  num_layers: number of hidden layers to use in the LSTM.
  hid_mult: np.array of integers to multiple input_size by.
  seq_len: array of sequnce lengths to train on.
  """
  hidden_experiment = dict()
  # we experiment with different hidden sizes
  for hidden_size in input_size*hid_mult:
    hidden_size = int(hidden_size)
    print()
    print("Hidden size: %d" % hidden_size)
    print("~~~~~~~~~~~~~~~")
    # initialize model, optimizer and loss function
    lstm_model = NetworkLSTM(input_size, hidden_size, num_layers).double() 
    # optimize the model
    lstm_model, log = optimize_model(dataset=dataset, model=lstm_model, 
                                  num_epochs=num_epochs, seq_len=seq_len)
    # log results of this experiment      
    hidden_experiment[hidden_size] = log
  return hidden_experiment