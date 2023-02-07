from _pkg import *


class LinearNN(torch.nn.Module):
    def __init__(self, input_size):
        """
        A simple linear regression model to use as a baseline.
        The output will be the same shape as the input.
        """
        super(LinearNN, self).__init__()
        self.input_size = input_size
        self.output_size = input_size
        self.linear = torch.nn.Linear(self.input_size, self.output_size)

    def loss_fn(self):
        """The loss function to be used with this model."""
        return torch.nn.MSELoss()

    def forward(self, input, tau=1):
        """
        input: batch of data
        """
        # Repeat for tau>0 offset target
        for i in range(tau):
            output = self.linear(input)
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
        assert hidden_size >= input_size, "Model requires hidden_size >= input_size."
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )
        self.linear = torch.nn.Linear(self.hidden_size, self.input_size)

    def loss_fn(self):
        """
        The loss function to be used for this model.
        """
        return torch.nn.MSELoss()

    def forward(self, input, tau=1):
        """
        input: batch of data
        tau: singleton tensor of int
        """
        # Propagate input through LSTM
        # lstm_out is all of the hidden states throughout the sequence
        lstm_out, self.hidden = self.lstm(input)
        lstm_out = torch.nn.functional.relu(self.linear(lstm_out))
        for i in range(1, tau):
            lstm_out, self.hidden = self.lstm(lstm_out, self.hidden)
            lstm_out = torch.nn.functional.relu(self.linear(lstm_out))
        return lstm_out


# Variational autoencoder LSTM model.
class VAE_LSTM(nn.Module):
    """
    The NetworkLSTM model with self-supervised VAE loss.
    The VAE loss regularizes the model by encouraging the
    network to learn cell states that can reliably reproduce
    the inputs.
    """

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
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            proj_size=self.proj_size,
            bias=False,
            batch_first=True,
        )
        # variational autoencoder (VAE)
        self.elbo_loss = 0
        latent_dim = hidden_size
        enc_out_dim = hidden_size
        # TODO: use the cell state c_n as the encoding!
        self.encoder = nn.Linear(input_size, enc_out_dim)  # output hyperparameters
        self.decoder = nn.Linear(latent_dim, input_size)  # logits
        # distribution parameters
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_var = nn.Linear(enc_out_dim, latent_dim)
        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    def loss_fn(self):
        """The loss function to be used for this model."""
        return torch.nn.MSELoss()

    def gaussian_likelihood(self, mean, logscale, sample):
        scale = torch.exp(logscale)
        dist = torch.distributions.Normal(mean, scale)
        log_pxz = dist.log_prob(sample)
        return log_pxz.sum(dim=(1, 2))

    def kl_divergence(self, z, mu, std):
        """Monte carlo KL divergence computation."""
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)
        # kl
        kl = log_qzx - log_pz
        kl = kl.sum(dim=(1, 2))
        return kl

    def forward(self, x, tau, state=None):
        """
        x: batch of data
        tau: singleton tensor of int
        """
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
        elbo = kl - recon_loss
        elbo = elbo.mean()
        self.elbo_loss = elbo
        return lstm_out, state


class ConvGNN(torch.nn.Module):
    """TODO: docstrings
    Class to use for subclassing all convoultuional GNNs"""

    def __init__(self):
        super(ConvGNN, self).__init__()


class DeepGCNConv(ConvGNN):
    """TODO: want classes to be understandable from their names."""

    def __init__(self, hidden_channels, num_node_features, num_classes):
        super(DeepGCNConv, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, graph):
        x = graph.x
        edge_index = graph.edge_index
        batch = graph.batch
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x


class DeepGATConv(ConvGNN):
    def __init__(self, hidden_channels, num_node_features, num_classes):
        super(DeepGATConv, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = DeepGATConv(num_node_features, hidden_channels)
        self.conv2 = DeepGATConv(hidden_channels, hidden_channels)
        self.conv3 = DeepGATConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, graph):
        x = graph.x
        edge_index = graph.edge_index
        batch = graph.batch
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x


class GraphNN(torch.nn.Module):
    """Predict the residual at the next time step, given a history of length L"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphNN, self).__init__()
        # need a conv. layer for each type of synapse
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
