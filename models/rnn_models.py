import torch
import torch.nn as nn

#TODO: Add a batch normalization layer.
#TODO: Use LSTMCell instead of full LSTM.

#LSTM network model.
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
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, proj_size=self.proj_size,
                            bias=False, batch_first=True)
        
    def loss_fn(self):
        '''
        The loss function to be used with this model.
        '''
        return torch.nn.MSELoss()

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

# Variational autoencoder LSTM model.
class VAE_LSTM(nn.Module):
    '''
    The NetworkLSTM model with self-supervised VAE loss.
    The VAE loss regularizes the model by encouraging the 
    network to learn cell states that can reliably reproduce
    the inputs.
    '''
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(VAE_LSTM, self).__init__()
        # checking sizes
        assert hidden_size >= input_size, "Model requires hidden_size >= input_size."
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        if self.hidden_size == self.input_size:
          self.proj_size = 0
        else: 
          self.proj_size = input_size
        # LSTM
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, proj_size=self.proj_size,
                            bias=False, batch_first=True)
        # variational autoencoder (VAE)
        self.elbo_loss = 0
        latent_dim = hidden_size
        enc_out_dim = hidden_size
        # TODO: use the cell state c_n as the encoding!
        self.encoder = nn.Linear(input_size, enc_out_dim) # output hyperparameters
        self.decoder = nn.Linear(latent_dim, input_size) # logits
        # distribution parameters
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_var = nn.Linear(enc_out_dim, latent_dim)
        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    def loss_fn(self):
        '''The loss function to be used for this model.'''
        return torch.nn.MSELoss()

    def gaussian_likelihood(self, mean, logscale, sample):
        scale = torch.exp(logscale)
        dist = torch.distributions.Normal(mean, scale)
        log_pxz = dist.log_prob(sample)
        return log_pxz.sum(dim=(1, 2))

    def kl_divergence(self, z, mu, std):
        '''Monte carlo KL divergence computation.'''
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)
        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(dim=(1, 2))
        return kl

    def forward(self, x, tau, state=None):
        '''
        x: batch of data
        tau: singleton tensor of int
        '''
        # LSTM part
        # Propagate input through LSTM
        lstm_out = x
        # Repeat for tau>0 offset target
        for i in range(tau):
          lstm_out, state = self.lstm(lstm_out, state)
        # VAE part
        # encode x to get the mu and variance parameters
        # TODO: use the cell state c_n as the encoding!
        print(x.shape, state[-1].shape)
        x_encoded = self.encoder(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)
        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        # decoded 
        x_hat = self.decoder(z)
        # reconstruction loss
        recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)
        # kl
        kl = self.kl_divergence(z, mu, std)
        # elbo
        elbo = (kl - recon_loss)
        elbo = elbo.mean()
        self.elbo_loss = elbo
        return lstm_out, state