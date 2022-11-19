import torch.nn as nn

class NetworkLSTM(nn.Module):
    '''
    A model of the C. elegans neural network using an LSTM.
    Given an input sequence of length $L$ and an offset $\tau$, 
    this model is trained to output the sequence of length $L$ 
    that occurs $tau$ steps later.
    '''
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(NetworkLSTM, self).__init__()
      
        assert hidden_size >= input_size, "Model requires hidden_size >= input_size."
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        if self.hidden_size == self.input_size:
          self.proj_size = 0
        else: 
          self.proj_size = input_size

        self.elbo_loss = 0
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, proj_size=self.proj_size,
                            bias=False, batch_first=True)
        
    def loss_fn(self):
        '''
        The loss function to be used for this model.
        '''
        return nn.MSELoss()

    def forward(self, x, tau, state=None):
        '''
        x: batch of data
        tau: singleton tensor of int
        '''
        # Propagate input through LSTM
        lstm_out = x
        # Repeat for tau>0 offset target
        for i in range(tau):
          lstm_out, state = self.lstm(lstm_out, state) 
        return lstm_out, state