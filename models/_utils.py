from models._pkg import *


class LinearNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        """
        A simple linear regression model to use as a baseline.
        The output shape will be the same shape as that of the input.
        """
        super(LinearNN, self).__init__()
        self.input_size = input_size
        self.output_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Linear model
        self.input_hidden = torch.nn.Linear(self.input_size, self.hidden_size)
        self.hidden_hidden = (self.num_layers - 1) * (
            torch.nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.hidden_output = torch.nn.Linear(self.hidden_size, self.output_size)
        self.model = nn.Sequential(
            self.input_hidden, *self.hidden_hidden, self.hidden_output
        )

    def loss_fn(self):
        """The loss function to be used with this model."""
        return torch.nn.MSELoss()

    def forward(self, input, tau=1):
        """
        input: batch of data
        tau: time offset of target
        """
        # Repeat for tau>0 offset target
        for i in range(tau):
            output = self.model(input)
        return output


class NetworkLSTM(torch.nn.Module):
    """
    A model of the C. elegans neural network using an LSTM.
    Given an input sequence of length $L$ and an offset $\tau$,
    this model is trained to output the sequence of length $L$
    that occurs $tau$ steps later.
    TODO: Add a batch normalization layer.
    """

    def __init__(self, input_size, hidden_size, num_layers=1):
        """
        The output size will be the same as the input size.
        """
        super(NetworkLSTM, self).__init__()
        self.input_size = input_size
        self.output_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # LSTM
        self.lstm = torch.nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )
        # Readout
        self.linear = torch.nn.Linear(self.hidden_size, self.output_size)

    def loss_fn(self):
        """
        The loss function to be used with this model.
        """
        return torch.nn.MSELoss()

    def forward(self, input, tau=1):
        """
        Propagate input through the LSTM.
        input: batch of data
        tau: time offset of target
        """
        # lstm_out is all of the hidden states throughout the sequence
        lstm_out, self.hidden = self.lstm(input)
        readout = self.linear(lstm_out)
        lstm_out = torch.sigmoid(readout)
        for i in range(1, tau):
            lstm_out, self.hidden = self.lstm(lstm_out, self.hidden)
            readout = self.linear(lstm_out)
            lstm_out = torch.sigmoid(readout)
        return lstm_out


# Variational autoencoder LSTM model.
class VAELSTM(nn.Module):
    """
    Same as the NetworkLSTM model but with an added
    self-supervised VAE loss. The VAE loss regularizes
    the model by encouraging the network to learn cell states
    that can reliably reproduce the inputs.
    """

    def __init__(self, input_size, hidden_size, num_layers=1):
        super(VAELSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # LSTM
        self.lstm = torch.nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )
        # readout
        self.linear = torch.nn.Linear(self.hidden_size, self.input_size)
        # variational autoencoder (VAE)
        self.elbo_loss = 0
        # logits
        self.decoder = nn.Linear(hidden_size, input_size)
        # distribution parameters
        self.fc_mu = nn.Linear(hidden_size, hidden_size)
        self.fc_var = nn.Linear(hidden_size, hidden_size)
        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    def loss_fn(self):
        """
        The loss function to be used with this model.
        """

        def func(input, target, **kwargs):
            return self.elbo_loss + torch.nn.MSELoss(**kwargs)(input, target)

        return func

    def gaussian_likelihood(self, mean, logscale, sample):
        scale = torch.exp(logscale)
        dist = torch.distributions.Normal(mean, scale)
        log_pxz = dist.log_prob(sample)
        return log_pxz.sum(dim=(1, 2))

    def kl_divergence(self, z, mu, std):
        """
        Monte-Carlo KL divergence computation.
        """
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)
        # 3. kl divergence calculation
        kl = log_qzx - log_pz
        kl = kl.sum(dim=(1, 2))
        return kl

    def forward(self, input, tau=1, state=None):
        """
        input: batch of data
        tau: time offset of target
        """
        # LSTM part
        # Propagate input through LSTM
        lstm_out, self.hidden = self.lstm(input)
        readout = self.linear(lstm_out)
        lstm_out = torch.sigmoid(readout)
        # Repeat for tau>0 offset target
        for i in range(tau):
            lstm_out, self.hidden = self.lstm(lstm_out, self.hidden)
            readout = self.linear(lstm_out)
            lstm_out = torch.sigmoid(readout)
        # VAE part
        # encode x to get the mu and variance parameters
        # self.hidden[-1][-1] last cell state (batch_size x hidden_size)
        h_final, c_final = self.hidden
        x_encoded = c_final[-1].unsqueeze(1).repeat(1, input.size(1), 1)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)
        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()  # batch_size x hidden_size
        # decoded
        x_hat = self.decoder(z)  # batch_size x input size
        # reconstruction loss
        recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, input)
        # kl
        kl = self.kl_divergence(z, mu, std)
        # elbo
        elbo = kl - recon_loss
        elbo = elbo.mean()
        self.elbo_loss = elbo
        return lstm_out


class GraphNN(torch.nn.Module):
    """
    Applies a single graph convolutional layer separately to the
    two graphs induced by using only chemical synapses or
    gap junctions, respectively, as the edges.
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphNN, self).__init__()
        # a separte learnable conv. layer for each type of synapse
        self.elec_conv = GCNConv(
            in_channels=input_dim, out_channels=hidden_dim, improved=True
        )
        self.chem_conv = GCNConv(
            in_channels=input_dim, out_channels=hidden_dim, improved=True
        )
        # readout layer transforms node features to output
        self.linear = torch.nn.Linear(
            in_features=2 * hidden_dim, out_features=output_dim
        )

    def forward(self, x, edge_index, edge_attr):
        elec_weight = edge_attr[:, 0]
        chem_weight = edge_attr[:, 1]
        elec_hid = self.elec_conv(x, edge_index, elec_weight)
        chem_hid = self.chem_conv(x, edge_index, chem_weight)
        hidden = torch.cat([elec_hid, chem_hid], dim=-1)
        output = self.linear(hidden)
        return output
