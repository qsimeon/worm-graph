import numpy as np
from models.rnn_models import NetworkLSTM
from train.train_main import optimize_model


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