from models._pkg import *


class LinearNN(torch.nn.Module):
    """
    A simple linear regression model to use as a baseline.
    For all our models:
        1. The output will be the same shape as the input.
        2. A method called `loss_fn` that specifies the specific
            loss function to be used by the model. The default
            loss function we use is `torch.nn.MSELoss()`.
        3. A readout layer is implemented and will always be
            called `self.linear`.
        4. There is at least 1 hidden layer. The value of the
            argument `num_layers` specifies the number of hidden layers.
        5. When it is possible for a model to be multi-layered,
            the `num_layers` is used to create the desired number of
            layers. Otherwise `num_layers` is implied to be 1.
        6. TODO: A method called `sample` should be implemented to allow
            sampling spontaneous neural activity from the network.
            Need to read the literature on generative, score-based and energy-based
            models to figure out how to implement this.
    """

    def __init__(self, input_size, hidden_size, num_layers=1, loss=None):
        super(LinearNN, self).__init__()
        # Loss function
        if (loss is None) or (str(loss).lower() == "l1"):
            self.loss = torch.nn.L1Loss
        elif str(loss).lower() == "mse":
            self.loss = torch.nn.MSELoss
        elif str(loss).lower() == "huber":
            self.loss == torch.nn.HuberLoss
        else:
            self.loss = torch.nn.L1Loss
        # Setup
        self.input_size = input_size
        self.output_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Input and hidden layers
        self.input_hidden = (
            torch.nn.Linear(self.input_size, self.hidden_size),
            torch.nn.ReLU(),
        )
        self.hidden_hidden = (self.num_layers - 1) * (
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(),
        )
        # Readout
        self.linear = torch.nn.Linear(self.hidden_size, self.output_size)
        # Full model
        self.model = torch.nn.Sequential(
            *self.input_hidden,
            *self.hidden_hidden,
            self.linear,
        )

    def loss_fn(self):
        """
        The loss function to be used with this model.
        """
        return self.loss()

    def get_input_size(self):
        return self.input_size

    def get_hidden_size(self):
        return self.hidden_size

    def forward(self, input, tau=1):
        """
        input: batch of data
        tau: time offset of target
        """
        readout = self.model(input)
        output = readout
        # repeat for target with tau>0 offset
        for i in range(1, tau):
            readout = self.model(input)
            output = readout
        return output


class NeuralCFC(torch.nn.Module):
    """
    Neural Circuit Policy (NCP) Closed-form continuous time (CfC) model.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        loss: Callable = None,
    ):
        """
        The output size will be the same as the input size.
        """
        super(NeuralCFC, self).__init__()
        # Loss function
        if (loss is None) or (str(loss).lower() == "l1"):
            self.loss = torch.nn.L1Loss
        elif str(loss).lower() == "mse":
            self.loss = torch.nn.MSELoss
        elif str(loss).lower() == "huber":
            self.loss == torch.nn.HuberLoss
        else:
            self.loss = torch.nn.L1Loss
        # Setup
        self.input_size = input_size
        self.output_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # NCP wired CfC
        self.wiring = AutoNCP(
            self.hidden_size * (1 + self.num_layers),
            self.hidden_size,
        )
        self.rnn = CfC(self.input_size, self.wiring)
        # Readout
        self.linear = torch.nn.Linear(self.hidden_size, self.output_size)

    def loss_fn(self):
        """
        The loss function to be used with this model.
        """
        return self.loss()

    def get_input_size(self):
        return self.input_size

    def get_hidden_size(self):
        return self.hidden_size

    def forward(self, input, tau=1):
        """
        Propagate input through the continuous time NN.
        input: batch of data
        tau: time offset of target
        """
        rnn_out, self.hidden = self.rnn(input)
        readout = self.linear(rnn_out)  # projection
        rnn_out = readout
        # repeat for target with tau>0 offset
        for i in range(1, tau):
            rnn_out, self.hidden = self.rnn(rnn_out, self.hidden)
            readout = self.linear(rnn_out)
            rnn_out = readout
        return rnn_out


class NetworkLSTM(torch.nn.Module):
    """
    A model of the C. elegans neural network using an LSTM.
    Given an input sequence of length $L$ and an offset $\tau$,
    this model is trained to output the sequence of length $L$
    that occurs $tau$ steps later.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        loss: Callable = None,
    ):
        """
        The output size will be the same as the input size.
        """
        super(NetworkLSTM, self).__init__()
        # Loss function
        if (loss is None) or (str(loss).lower() == "l1"):
            self.loss = torch.nn.L1Loss
        elif str(loss).lower() == "mse":
            self.loss = torch.nn.MSELoss
        elif str(loss).lower() == "huber":
            self.loss == torch.nn.HuberLoss
        else:
            self.loss = torch.nn.L1Loss
        # Setup
        self.input_size = input_size
        self.output_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # LSTM
        self.lstm = torch.nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=True,
            batch_first=True,
        )
        # re-initialize LSTM recurrent hidden to hidden weights
        for ind in range(self.num_layers):
            weight_hh = getattr(self.lstm, "weight_hh_l{}".format(ind))
            # torch.nn.init.eye_(weight_hh)
            torch.nn.init.kaiming_uniform_(
                weight_hh, mode="fan_in", nonlinearity="relu"
            )
        # Readout
        self.linear = torch.nn.Linear(self.hidden_size, self.output_size)

    def loss_fn(self):
        """
        The loss function to be used with this model.
        """
        return self.loss()

    def get_input_size(self):
        return self.input_size

    def get_hidden_size(self):
        return self.hidden_size

    def forward(self, input: torch.Tensor, tau: int = 1):
        """
        Propagate input through the LSTM.
        input: batch of data
        tau: time offset of target
        """
        lstm_out, self.hidden = self.lstm(input)
        readout = self.linear(lstm_out)  # projection
        lstm_out = readout
        # repeat for target with tau>0 offset
        for i in range(1, tau):
            lstm_out, self.hidden = self.lstm(lstm_out, self.hidden)
            readout = self.linear(lstm_out)
            lstm_out = readout
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
