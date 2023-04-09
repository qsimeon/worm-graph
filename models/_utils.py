from models._pkg import *


class Head(torch.nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.key = torch.nn.Linear(n_embd, head_size, bias=False)
        self.query = torch.nn.Linear(n_embd, head_size, bias=False)
        self.value = torch.nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B,T,C)
        q = self.query(x)  # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = torch.nn.functional.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,C)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class MultiHeadAttention(torch.nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, n_head, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.heads = torch.nn.ModuleList(
            [Head(head_size, n_embd, block_size, dropout) for _ in range(n_head)]
        )
        self.proj = torch.nn.Linear(n_embd, n_embd)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(torch.nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_embd, 4 * n_embd),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * n_embd, n_embd),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(torch.nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, block_size, n_head, dropout):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = torch.nn.LayerNorm(n_embd)
        self.ln2 = torch.nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class PositionalEncoding(torch.nn.Module):
    """
    Sinuosoidal positional encoding from Attention is All You Need paper.
    """

    def __init__(
        self,
        n_embd: int,
        max_len: int = 5000,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_embd, 2) * (-math.log(10000.0) / n_embd))
        pe = torch.zeros(1, max_len, n_embd)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):  # x has shape (batch_size, block_size, n_embd)
        """
        Args:
            x: Tensor, shape (batch_size, seq_len, embedding_dim)
        """
        x = x + self.pe[:, : x.size(1), :]  # add positional encoding to input
        return self.dropout(x)


class NeuralTransformer(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,  # currently unused
        num_layers=1,
        loss=None,
    ):
        """
        Neural activity data is continuous valued so
        we can treat as if it is already emebedded.
        """
        super(NeuralTransformer, self).__init__()
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
        self.hidden_size = hidden_size  # currently unused
        self.num_layers = num_layers
        self.n_head = 2  # TODO: make a function of hidden_size
        self.block_size = 10000  # maximum attention block (i.e. context) size
        self.dropout = 0.1
        # Identity layer
        self.identity = torch.nn.Identity()
        # Transformer parts
        self.position_encoding = PositionalEncoding(
            self.input_size,
            max_len=self.block_size,
            dropout=self.dropout,
        )
        self.blocks = torch.nn.Sequential(
            *(
                Block(
                    n_embd=self.input_size,
                    block_size=self.block_size,
                    n_head=self.n_head,
                    dropout=self.dropout,
                )
                for _ in range(self.num_layers)
            )
        )
        self.ln_f = torch.nn.LayerNorm(self.input_size)  # final layer norm
        # Readout
        self.linear = torch.nn.Linear(
            self.input_size, self.output_size
        )  # linear readout

    def forward(self, input, tau=0):
        """
        (B, T, C) = input.shape = (batch_size, max_time, input_size)
        """
        if tau < 1:
            output = self.identity(input)
        else:
            x = self.position_encoding(input)  # (B,T,C)
            x = self.blocks(x)  # (B,T,C)
            x = self.ln_f(x)  # (B,T,C)
            readout = self.linear(x)  # (B,T,C)
            output = readout
        # repeat for target with tau>0 offset
        for i in range(1, tau):
            x = self.position_encoding(output)  # (B,T,C)
            x = self.blocks(x)  # (B,T,C)
            x = self.ln_f(x)  # (B,T,C)
            readout = self.linear(x)  # (B,T,C)
            output = readout
        return output

    def loss_fn(self):
        """
        The loss function to be used with this model.
        """
        return self.loss()
        # def func(input, target, **kwargs):
        #     return self.elbo_loss + self.loss(**kwargs)(input, target)

        # return func

    def get_input_size(self):
        return self.input_size

    def get_hidden_size(self):
        return self.hidden_size

    def get_num_layers(self):
        return self.num_layers


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
            the `num_layers` argument is used to create the desired number
            of layers. Otherwise `num_layers` defaults to 1.
        6. TODO: A method called `sample` or `generate` should be implemented to allow
            sampling spontaneous neural activity from the network.
            Need to read the literature on generative models, score-/energy-based
            models, and diffusion models to understand out how to implement this.
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
        # Identity layer
        self.identity = torch.nn.Identity()
        # Input and hidden layers
        self.input_hidden = (
            torch.nn.Linear(self.input_size, self.hidden_size),
            # torch.nn.ReLU(),
            torch.nn.ELU(),
            torch.nn.LayerNorm(self.hidden_size),
        )
        self.hidden_hidden = (self.num_layers - 1) * (
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            # torch.nn.ReLU(),
            torch.nn.ELU(),
            torch.nn.LayerNorm(self.hidden_size),
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

    def get_num_layers(self):
        return self.num_layers

    def forward(self, input, tau=0):
        """
        input: batch of data
        tau: time offset of target
        """
        if tau < 1:
            output = self.identity(input)
        else:
            readout = self.model(input)
            output = readout
        # repeat for target with tau>0 offset
        for i in range(1, tau):
            readout = self.model(output)
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
        # Identity layer
        self.identity = torch.nn.Identity()
        # NCP wired CfC
        self.wiring = AutoNCP(
            self.hidden_size * (1 + self.num_layers),
            self.hidden_size,
        )
        self.rnn = CfC(self.input_size, self.wiring)
        # Initialize hidden state
        self.hidden = None
        # Layer norm
        self.layer_norm = torch.nn.LayerNorm(self.hidden_size)
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

    def get_num_layers(self):
        return self.num_layers

    def forward(self, input, tau=0):
        """
        Propagate input through the continuous time NN.
        input: batch of data
        tau: time offset of target
        """
        if tau < 1:
            rnn_out = self.identity(input)
        else:
            rnn_out, self.hidden = self.rnn(input)
            rnn_out = self.layer_norm(rnn_out)  # layer normalization
            readout = self.linear(rnn_out)  # projection
            rnn_out = readout
        # repeat for target with tau>0 offset
        for i in range(1, tau):
            rnn_out, self.hidden = self.rnn(rnn_out, self.hidden)
            rnn_out = self.layer_norm(rnn_out)  # layer normalization
            readout = self.linear(rnn_out)  # projection
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
        # Identity layer
        self.identity = torch.nn.Identity()
        # LSTM
        self.lstm = torch.nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=True,
            batch_first=True,
        )
        # Initialize LSTM recurrent hidden to hidden weights
        for ind in range(self.num_layers):
            weight_hh = getattr(self.lstm, "weight_hh_l{}".format(ind))
            torch.nn.init.kaiming_uniform_(
                weight_hh, mode="fan_in", nonlinearity="relu"
            )
        # Initialize hidden state
        self.hidden = None
        # Layer norm
        self.layer_norm = torch.nn.LayerNorm(self.hidden_size)
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

    def get_num_layers(self):
        return self.num_layers

    def forward(self, input, tau=0):
        """
        Propagate input through the LSTM.
        input: batch of data
        tau: time offset of target
        """
        if tau < 1:
            lstm_out = self.identity(input)
        else:
            lstm_out, self.hidden = self.lstm(input)
            lstm_out = self.layer_norm(lstm_out)  # layer normalization
            readout = self.linear(lstm_out)  # projection
            lstm_out = readout
        # repeat for target with tau>0 offset
        for i in range(1, tau):
            lstm_out, self.hidden = self.lstm(lstm_out, self.hidden)
            lstm_out = self.layer_norm(lstm_out)  # layer normalization
            readout = self.linear(lstm_out)  # projection
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
