import torch, torchvision
import torch.nn as nn
import numpy as np

class WeirdCTRNN(nn.Module):
    """My weird Continuous-time RNN.

    Parameters:
        time_steps: Number of time steps to predict
        hidden_size: Number of hidden neurons
        dt: discretization time step in ms. 
            If None, dt equals time constant tau

    Inputs:
        input: tensor of shape (seq_len, batch, input_size)
        hidden: tensor of shape (batch, hidden_size), initial hidden activity
            if None, hidden is initialized through self.init_hidden()
        
    Outputs:
        output: tensor of shape (seq_len, batch, hidden_size)
        hidden: tensor of shape (batch, hidden_size), final hidden activity
    """

    def __init__(self, block_size, num_neurons, loops, g, dt=None, tau=100):
        super().__init__()
        self.block_size = block_size
        self.num_neurons = num_neurons
        self.loops = loops
        self.g = g
        self.tau = tau
        self.dt = dt

        if self.dt is None:
            alpha = 1
        else:
            alpha = self.dt / self.tau
        self.alpha = alpha

        self.J = nn.Linear(num_neurons, num_neurons, bias=False)
        self.phi = nn.Tanh() # fire rate function
        self.loss = nn.MSELoss() # nn.HuberLoss()

    def genMatrixJ(self):
        J = np.random.randn(self.num_neurons, self.num_neurons)
        J = self.g / np.sqrt(self.num_neurons) * J
        np.fill_diagonal(J, 0)
        J = torch.from_numpy(J).float()
        return J

    def forward(self, x, y=None):
        """Forward pass of the model.

        x : inputs
            Tensor of shape (batch, block_size, num_neurons)
        y : targets
            Tensor of shape (batch, block_size, num_neurons)
        l : loss
        out : predictions
            Tensor of shape (batch, block_size, num_neurons)
        """
        
        for _ in range(self.loops):
            fire_rate = self.J(self.phi(x))
            x = x + self.dt * (-x + fire_rate) / self.tau

        if y is None:
            return x, None
        else:
            return x, self.loss(x, y)
    
    def generate(self, context, max_new_tokens):
        # This is the function we will use to generate data

        # context is a (time_steps, num_neurons) tensor in the current context
        for _ in range(max_new_tokens):
            out, _ = self(context) # Get the predictions
            out = out[-1,:].reshape(1, -1) # Focus only on the last time step => (1, num_neurons)
            context = torch.cat([context, out], dim=0) # Append the new tokens to the current context => (time_steps+1, num_neurons)
        return context