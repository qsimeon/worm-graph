from models._pkg import *


# # # Transformer Parts (Self-Attention, Feed-Forward, Positional Encoding) # # #
# # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
class Head(torch.nn.Module):
    """
    One head of self-attention.
    """

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
    """
    Multiple heads of self-attention in parallel.
    """

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
    """
    A simple linear layer followed by a non-linearity.
    """

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


class TransformerBlock(torch.nn.Module):
    """
    Transformer block: communication followed by computation.
    """

    def __init__(self, n_embd, block_size, n_head, dropout):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = torch.nn.LayerNorm(n_embd)
        self.ln2 = torch.nn.LayerNorm(n_embd)

    def forward(self, x):
        # notice the use of residual (skip) connections
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
        max_len: int = MAX_TOKEN_LEN,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_len = max_len
        self.dropout = torch.nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_embd, 2) * (-math.log(10000.0) / n_embd))
        pe = torch.zeros(1, max_len, n_embd)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape (batch_size, seq_len, embedding_dim)
        """
        x = x + self.pe[:, : x.size(1), :]  # add positional encoding to input
        return self.dropout(x)


# # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


# # # Backbones or Inner Parts of Other Models # # #
# # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
class CTRNN(torch.nn.Module):
    """Continuous-time RNN.

    Parameters:
        input_size: Number of input neurons
        hidden_size: Number of hidden neurons
        dt: discretization time step in ms.
            If None, dt equals time constant tau

    Inputs:
        input: tensor of shape (seq_len, batch, input_size)
        hidden: tensor of shape (batch, hidden_size), initial hidden activity
            if None, hidden is initialized through self.init_hidden()

    Outputs:
        output: tensor of shape (batch, seq_len, hidden_size)
        hidden: tensor of shape (batch, hidden_size), final hidden activity
    """

    def __init__(self, input_size, hidden_size, tau=100, dt=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tau = tau  # this gives an interpration of percentage
        if dt is None:
            alpha = 1
        else:
            alpha = dt / self.tau  # think of as a decay percentage
        self.alpha = alpha

        self.input2h = torch.nn.Linear(input_size, hidden_size)
        self.h2h = torch.nn.Linear(hidden_size, hidden_size)

    def init_hidden(self, input_shape):
        device = next(self.parameters()).device
        batch_size = input_shape[0]  # beacuse batch_first=True
        hidden = torch.zeros(batch_size, self.hidden_size).to(device)
        return hidden

    def recurrence(self, input, hidden):
        """
        Run network for one time step.

        Inputs:
            input: tensor of shape (batch, input_size)
            hidden: tensor of shape (batch, hidden_size)

        Outputs:
            h_new: tensor of shape (batch, hidden_size),
                network activity at the next time step
        """
        h_new = torch.relu(self.input2h(input) + self.h2h(hidden))
        h_new = hidden * (1 - self.alpha) + h_new * self.alpha
        return h_new

    def forward(self, input, hidden=None):
        """
        Propagate input through the network.
        NOTE: Because we use batch_first=True,
        input has shape (batch, seq_len, input_size).
        """

        # If hidden activity is not provided, initialize it
        if hidden is None:
            hidden = self.init_hidden(input.shape)

        # Loop through time
        output = []
        steps = range(input.size(1))
        for i in steps:
            hidden = self.recurrence(input[:, i, :], hidden)
            output.append(hidden)

        # Stack together output from all time steps
        output = torch.stack(output, dim=1)  # (batch, seq_len, hidden_size)
        return output, hidden


class FeedForwardBlock(torch.nn.Module):
    """
    Feedforward block.
    """

    def __init__(self, n_embd, dropout):
        # n_embd: embedding dimension
        super().__init__()
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln = torch.nn.LayerNorm(n_embd)

    def forward(self, x):
        # notice the use of residual (skip) connections
        x = x + self.ffwd(self.ln(x))
        return x


class InnerHiddenModel(torch.nn.Module):
    """
    Inner hidden (latent) models.
    """

    def __init__(self, hidden_hidden_model: torch.nn.Module, hidden_state=None):
        super().__init__()
        self.hidden_hidden = hidden_hidden_model
        self.hidden = hidden_state

    def forward(self, x):
        if self.hidden is None:
            x = self.hidden_hidden(x)
        else:
            x, self.hidden = self.hidden_hidden(x, self.hidden)
        return x

    def set_hidden(self, hidden_state):
        self.hidden = hidden_state
        return None


# # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


# # # Models (super class and sub classes) # # #
# # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
class Model(torch.nn.Module):
    """
    Super class for all models.

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
        7. Getter methods for the input size, hidden size, and number of layers called
            `get_input_size`, `get_hidden_size`, and `get_num_layers`, respectively.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        loss: Union[Callable, None] = None,
        fft_reg_param: float = 0.0,
        l1_reg_param: float = 0.0,
    ):
        """Defines attributes common to all models."""
        super(Model, self).__init__()
        assert (
            isinstance(fft_reg_param, float) and 0.0 <= fft_reg_param <= 1.0
        ), "The regularization parameter `fft_reg_param` must be a float between 0.0 and 1.0."
        assert (
            isinstance(l1_reg_param, float) and 0.0 <= l1_reg_param <= 1.0
        ), "The regularization parameter `l1_reg_param` must be a float between 0.0 and 1.0."
        # Loss function
        if (loss is None) or (str(loss).lower() == "l1"):
            self.loss = torch.nn.L1Loss
        elif str(loss).lower() == "mse":
            self.loss = torch.nn.MSELoss
        elif str(loss).lower() == "huber":
            self.loss = torch.nn.HuberLoss
        elif str(loss).lower() == "poisson":
            self.loss = torch.nn.PoissonNLLLoss
        else:
            self.loss = torch.nn.MSELoss

        # Name of original loss function
        self.loss_name = self.loss.__name__[:-4]
        # Setup
        self.input_size = input_size  # Number of neurons (302)
        self.output_size = input_size  # Number of neurons (302)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.fft_reg_param = fft_reg_param
        self.l1_reg_param = l1_reg_param
        # Initialize hidden state
        self._init_hidden()
        # Initialize the tau
        self.tau = 1  # next-timestep prediction
        # Identity layer
        self.identity = torch.nn.Identity()
        # Input to hidden transformation - placeholder
        self.input_hidden = torch.nn.Linear(self.input_size, self.hidden_size)
        # Hidden to hidden transformation - placeholder
        self.hidden_hidden = torch.nn.Linear(self.hidden_size, self.hidden_size)
        # # # DEBUGGING # # #
        # Instantiate internal hidden model - placeholder
        self.inner_hidden_model = InnerHiddenModel(self.hidden_hidden, self.hidden)
        # # # DEBUGGING # # #
        # Linear readout
        self.linear = torch.nn.Linear(self.hidden_size, self.output_size)
        # Initialize weights
        self._init_weights()

    # Initialization functions for setting hidden states and weights.
    def _init_hidden(self):
        self.hidden = None
        return self.hidden

    def _init_weights(self):
        # Initialize the readout bias
        torch.nn.init.zeros_(self.linear.bias)
        # Initialize the readout weights
        torch.nn.init.zeros_(self.linear.weight)
        # torch.nn.init.xavier_uniform_(self.linear.weight) # Xavier Initialization
        # torch.nn.init.kaiming_uniform_(self.linear.weight, nonlinearity='relu') # He Initialization
        return None

    def init_hidden(self, input_shape):
        raise NotImplementedError()

    # Getter functions for returning all attributes needed to reinstantiate a similar model
    def get_input_size(self):
        return self.input_size

    def get_hidden_size(self):
        return self.hidden_size

    def get_num_layers(self):
        return self.num_layers

    def get_loss_name(self):
        return self.loss_name

    def get_fft_reg_param(self):
        return self.fft_reg_param

    def get_l1_reg_param(self):
        return self.l1_reg_param

    @autocast()
    def forward(self, input: torch.Tensor, mask: torch.Tensor, tau: int = 1):
        """
        Forward method for simple linear regression model.

        Parameters
        ----------
        input : torch.Tensor
            Input data with shape (batch, seq_len, neurons)
        mask : torch.Tensor
            Mask on the neurons with shape (neurons,)
        tau : int, optional
            Time offset of target
        """

        # store the tau
        self.tau = tau
        # initialize hidden state
        self.hidden = self.init_hidden(input.shape)
        # # # DEBUGGING # # #
        # set hidden state of internal model
        self.inner_hidden_model.set_hidden(self.hidden)
        # # # DEBUGGING # # #
        # recast the mask to the input type and shape
        mask = mask.view(1, 1, -1).to(input.dtype)
        # initialize output tensor with input tensor
        output = self.identity(input * mask)
        # loop through tau
        for _ in range(tau):
            # multiply input by the mask
            input = output * mask
            # transform the input
            input_hidden_out = self.input_hidden(input)
            # concatenate into a single latent
            latent_out = input_hidden_out
            # transform the latent
            # # # DEBUGGING # # #
            hidden_out = self.inner_hidden_model(latent_out)
            # # # DEBUGGING # # #
            # hidden_out, self.hidden = self.hidden_hidden(latent_out, self.hidden)
            # perform a linear readout to get the output
            readout = self.linear(hidden_out)
            output = readout
        return output

    def loss_fn(self):
        """
        The loss function to be used by all the models.

        This custom loss function combines a primary loss function with
        two additional regularization terms:

        1. Fast Fourier Transform (FFT) matching: This regularization term encourages the frequency
        distribution of the model's sequence output to match that of the target sequence. We can think
        of this as performing a frequency distribution matching (FDM) operation on the model's output.
        It may help the model to learn the inherent frequencies in the target data and thus produce
        output sequences that are more similar to the target in the frequency domain.

        2. L1 regularization on all model weights: This regularization term encourages the model to use
        fewer parameters, effectively making the model more sparse. This can help to prevent
        overfitting, make the model more interpretable, and improve generalization by encouraging the
        model to use only the most important features. The L1 penalty is the sum of the absolute
        values of the weights.

        Both regularization terms are controlled by their respective regularization parameters:
        `fft_reg_param` and `l1_reg_param`.
        """

        def loss(prediction, target, **kwargs):
            """
            Calculate loss with FFT regularization and
            L1 regularization on all model weights.
            Arguments:
                prediction: (batch_size, seq_len, input_size)
                target: (batch_size, seq_len, input_size)
            """
            # apply exponential recency decay factor to original loss
            half_life = prediction.size(1) / 2  # half-life = seq_len / 2
            # # TODO: play around with time constant (a.k.a half-life, decay rate) parameter
            # half_life = 1e-10 # ~ infinitisemal time constant
            # half_life = 1e10 # ~ infinite time constant
            kernel = (
                torch.flip(
                    torch.exp(-torch.arange(prediction.size(1)) / half_life),
                    dims=[-1],
                )
                .view(1, -1, 1)
                .to(prediction.device)
            )
            # calculate next time step prediction loss
            original_loss = self.loss(**kwargs)(
                # kernel * prediction,
                # kernel * target,  # weigh more recent time steps more heavily
                # # NOTE: the next two options are extreme ends of spectrum from using an exponential recency decay
                # prediction[:, -self.tau :, :], # only consider latest time steps
                # target[:, -self.tau :, :], # equivalent to infinitisemal time constant
                prediction,  # consider all time steps
                target,  # equivalent to infinite time constant
            )
            # FFT regularization term
            fft_loss = 0.0
            if self.fft_reg_param > 0.0:
                # calculate FFT and take the real part
                input_fft = torch.fft.rfft(prediction, dim=-2).real
                target_fft = torch.fft.rfft(target, dim=-2).real
                # calculate average difference between real parts of FFTs
                fft_loss += torch.mean(torch.abs(input_fft - target_fft))
            # L1 regularization term
            l1_loss = 0.0
            if self.l1_reg_param > 0.0:
                # calculate L1 regularization term for all weights
                for param in self.parameters():
                    l1_loss += torch.mean(torch.abs(param))
            # combine original loss with regularization terms
            regularized_loss = (
                original_loss
                + self.fft_reg_param * fft_loss
                + self.l1_reg_param * l1_loss
            ) / (1.0 + self.fft_reg_param + self.l1_reg_param)
            return regularized_loss

        return loss

    def generate(
        self,
        inputs: torch.Tensor,
        future_timesteps: int = 1,
        context_len: int = MAX_TOKEN_LEN,
        mask: Union[torch.Tensor, None] = None,
    ):
        """
        Generate future timesteps of neural activity.
        """
        # check dimensions of input
        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(0)
        else:
            assert inputs.ndim == 3, "Inputs must have shape (B, T, C)."
        # create mask if none is provided
        mask = torch.ones(inputs.shape[-1], dtype=torch.bool) if mask is None else mask
        # use the full sequence as the context
        context_len = min(context_len, inputs.size(1))
        # TODO: remove this print statement
        print(
            "input length:",
            inputs.size(1),
            "\ncontext length:",
            context_len,
            "\nfuture timesteps:",
            future_timesteps,
            end="\n\n",
        )
        # initialize output tensor
        output = torch.zeros(
            (inputs.size(0), inputs.size(1) + future_timesteps, inputs.size(2)),
            device=inputs.device,
        )
        output[:, : inputs.size(1), :] = inputs  # (B, T, C)
        # generate future timesteps
        with torch.no_grad():
            for i in range(future_timesteps):
                # condition on the previous context_len timesteps
                input_cond = output[
                    :, inputs.size(1) - context_len + i : inputs.size(1) + i, :
                ]  # (B, T, C)
                # get the prediction of next timestep
                input_forward = self(input_cond, mask, tau=1)
                # focus only on the last time step
                next_timestep = input_forward[:, -1, :]  # (B, C)
                # append predicted next timestep to the running sequence
                output[:, inputs.size(1) + i, :] = next_timestep  # (B, T+1, C)
        return output  # (B, T+timesteps, C)

    def sample(self, timesteps):
        """
        Sample spontaneous neural activity from the model.
        TODO: Figure out how to use diffusion models to sample from the network.
        """
        pass


class LinearNN(Model):
    """
    TODO: Test model with/without using information from the neuron mask.
    A simple linear regression model to use as a baseline.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        loss: Union[Callable, None] = None,
        fft_reg_param: float = 0.0,
        l1_reg_param: float = 0.0,
    ):
        super(LinearNN, self).__init__(
            input_size,
            hidden_size,
            num_layers,
            loss,
            fft_reg_param,
            l1_reg_param,
        )
        # Special parameters for this model
        self.dropout = 0.0  # dropout rate

        # Embedding
        self.embedding = torch.nn.Linear(
            self.input_size,
            self.hidden_size,
        )  # combine input and mask

        # Feedforward blocks
        self.blocks = torch.nn.Sequential(
            *(
                FeedForwardBlock(
                    n_embd=self.hidden_size,
                    dropout=self.dropout,
                )
                for _ in range(self.num_layers)
            )
        )

        # Input to hidden transformation
        self.input_hidden = torch.nn.Sequential(
            self.embedding,
            torch.nn.ReLU(),
            # NOTE: YES use LayerNorm here!
            torch.nn.LayerNorm(self.hidden_size),
        )

        # Linear layer: Hidden to hidden transformation
        self.hidden_hidden = torch.nn.Sequential(
            self.blocks,
            torch.nn.ReLU(),
        )
        # # # DEBUGGING # # #
        # Instantiate internal hidden model
        self.inner_hidden_model = InnerHiddenModel(self.hidden_hidden, self.hidden)
        # # # DEBUGGING # # #

    def init_hidden(self, input_shape=None):
        return None


class NeuralTransformer(Model):
    """
    Transformer model for neural activity data.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        loss: Union[Callable, None] = None,
        fft_reg_param: float = 0.0,
        l1_reg_param: float = 0.0,
    ):
        """
        TODO: Cite Andrej Kaparthy's tutorial on "How to code GPT from scratch".
        Neural activity data is continuous valued and thus
        can naturally be treated as if it were already emebedded.
        However, to maintain notational similarity with the original
        Transformer architecture, we use a linear layer to perform
        expansion recoding - which acts as an embedding but is really
        just a linear projection.
        """
        super(NeuralTransformer, self).__init__(
            input_size,
            hidden_size,
            num_layers,
            loss,
            fft_reg_param,
            l1_reg_param,
        )

        # Special transformer parameters
        # TODO: Make a way to ensure `n_head` is a divisor of `hidden_size`
        self.n_head = (  # NOTE: this must be divisor of `hidden_size`
            4  # number of attention heads;
        )
        self.block_size = MAX_TOKEN_LEN  # maximum attention block (i.e. context) size
        self.dropout = 0.0  # dropout rate

        # Positional encoding
        self.position_encoding = PositionalEncoding(
            self.input_size,
            max_len=self.block_size,
            dropout=self.dropout,
        )

        # Embedding
        self.embedding = torch.nn.Linear(
            self.input_size,
            self.hidden_size,
        )  # combine input and mask

        # Transformer blocks
        self.blocks = torch.nn.Sequential(
            *(
                TransformerBlock(
                    n_embd=self.hidden_size,
                    block_size=self.block_size,
                    n_head=self.n_head,
                    dropout=self.dropout,
                )
                for _ in range(self.num_layers)
            )
        )

        # Input to hidden transformation
        self.input_hidden = torch.nn.Sequential(
            # NOTE: Position encoding before embedding improved performance.
            self.position_encoding,
            self.embedding,
            torch.nn.ReLU(),
            # NOTE: Do NOT use LayerNorm here!
        )

        # Hidden to hidden transformation
        self.hidden_hidden = torch.nn.Sequential(
            self.blocks,
            torch.nn.ReLU(),
            # NOTE: Do NOT use LayerNorm here!
        )
        # # # DEBUGGING # # #
        # Instantiate internal hidden model
        self.inner_hidden_model = InnerHiddenModel(self.hidden_hidden, self.hidden)
        # # # DEBUGGING # # #

    def init_hidden(self, input_shape=None):
        return None


class NetworkRNN(Model):
    """
    A model of the C. elegans nervous system using a continuous-time RNN backbone.
    TODO: Cite tutorial by Guangyu Robert Yang and associated primer paper.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        loss: Union[Callable, None] = None,
        fft_reg_param: float = 0.0,
        l1_reg_param: float = 0.0,
    ):
        """
        The output size will be the same as the input size.
        NOTE: The `num_layers` parameter is not being used in this model.
        TODO: Implement a way to use the `num_layers` parameter in this model.
        """
        super(NetworkRNN, self).__init__(
            input_size,
            hidden_size,
            num_layers,
            loss,
            fft_reg_param,
            l1_reg_param,
        )

        # Input to hidden transformation
        self.input_hidden = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_size),
            torch.nn.ReLU(),
            # NOTE: YES use LayerNorm here!
            torch.nn.LayerNorm(self.hidden_size),
        )

        # Hidden to hidden transformation: Continuous time RNN (CTRNN) layer
        self.hidden_hidden = CTRNN(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,  # combine input and mask
            dt=25,
        )
        # # # DEBUGGING # # #
        # Instantiate internal hidden model
        self.inner_hidden_model = InnerHiddenModel(self.hidden_hidden, self.hidden)
        # # # DEBUGGING # # #

    def init_hidden(self, input_shape):
        device = next(self.parameters()).device
        batch_size = input_shape[0]  # beacuse batch_first=True
        hidden = torch.zeros(batch_size, self.hidden_size).to(device)
        return hidden


class NeuralCFC(Model):
    """
    Neural Circuit Policy (NCP) Closed-form continuous time (CfC) model.
    TODO: Cite the paper by Ramin hasani, Daniela Rus et al.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,  # unused
        loss: Union[Callable, None] = None,
        fft_reg_param: float = 0.0,
        l1_reg_param: float = 0.0,
    ):
        """
        The output size will be the same as the input size.
        NOTE: The num_layers parameter is not being used in this model.
        TODO: Implement a way to use the `num_layers` parameter in this model.
        """

        super(NeuralCFC, self).__init__(
            input_size,
            hidden_size,
            num_layers,
            loss,
            fft_reg_param,
            l1_reg_param,
        )

        # Input to hidden transformation
        self.input_hidden = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_size),
            torch.nn.ReLU(),
            # NOTE: YES use LayerNorm here!
            torch.nn.LayerNorm(self.hidden_size),
        )

        # Hidden to hidden transformation: Closed-form continuous-time (CfC) layer
        self.hidden_hidden = CfC(
            input_size=self.hidden_size,  # combine input and mask
            units=self.hidden_size,
            activation="relu",
        )
        # # # DEBUGGING # # #
        # Instantiate internal hidden model
        self.inner_hidden_model = InnerHiddenModel(self.hidden_hidden, self.hidden)
        # # # DEBUGGING # # #

        # Initialize RNN weights
        self.init_weights()

    def init_hidden(self, input_shape):
        """
        Inititializes the hidden state of the RNN.
        """
        device = next(self.parameters()).device
        batch_size = input_shape[0]  # because batch_first=True
        hidden = torch.zeros(batch_size, self.hidden_size).to(device)
        return hidden

    def init_weights(self):
        """
        Initializes the weights of the RNN.
        """
        for name, param in self.hidden_hidden.named_parameters():
            if "weight" in name:  # weights
                torch.nn.init.xavier_uniform_(param.data, gain=1.5)
            elif "bias" in name:  # biases
                # param.data.fill_(0)
                torch.nn.init.zeros_(param.data)


class NetworkLSTM(Model):
    """
    A model of the _C. elegans_ neural network using an LSTM.
    Given an input sequence of length $L$ and an offset $\tau$,
    this model is trained to output the sequence of length $L$
    that occurs $tau$ time steps after the start of the input sequence.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        loss: Union[Callable, None] = None,
        fft_reg_param: float = 0.0,
        l1_reg_param: float = 0.0,
    ):
        """
        The output size will be the same as the input size.
        """
        super(NetworkLSTM, self).__init__(
            input_size,
            hidden_size,
            num_layers,
            loss,
            fft_reg_param,
            l1_reg_param,
        )

        # Input to hidden transformation
        self.input_hidden = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_size),
            torch.nn.ReLU(),
            # NOTE: YES use LayerNorm here!
            torch.nn.LayerNorm(self.hidden_size),
        )

        # Hidden to hidden transformation: Long-short term memory (LSTM) layer
        self.hidden_hidden = torch.nn.LSTM(
            input_size=self.hidden_size,  # combine input and mask
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=True,
            batch_first=True,
        )
        # # # DEBUGGING # # #
        # Instantiate internal hidden model
        self.inner_hidden_model = InnerHiddenModel(self.hidden_hidden, self.hidden)
        # # # DEBUGGING # # #

        # Initialize LSTM weights
        self.init_weights()

    def init_hidden(self, input_shape):
        """
        Inititializes the hidden and cell states of the LSTM.
        """
        device = next(self.parameters()).device
        batch_size = input_shape[0]  # because batch_first=True
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return (h0, c0)

    def init_weights(self):
        """
        Initializes the weights of the LSTM.
        """
        for name, param in self.hidden_hidden.named_parameters():
            if "weight_ih" in name:  # Input-hidden weights
                torch.nn.init.xavier_uniform_(param.data, gain=1.5)
            elif "weight_hh" in name:  # Hidden-hidden weights
                torch.nn.init.orthogonal_(param.data)
            elif "bias" in name:  # Bias weights
                # param.data.fill_(0)
                torch.nn.init.zeros_(param.data)


class NetworkGCN(Model):
    """
    A graph neural network model of the _C. elegans_ nervous system.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        loss: Union[Callable, None] = None,
        fft_reg_param: float = 0.0,
        l1_reg_param: float = 0.0,
        edge_index: torch.Tensor,
    ):
        super(NetworkGCN, self).__init__(
            input_dim,
            hidden_dim,
            num_layers,
            loss,
            fft_reg_param,
            l1_reg_param,
        )

        self.layers = torch.nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
        self.layers.append(GCNConv(hidden_dim, output_dim))

        self.edge_index = edge_index

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = torch.relu(layer(x, self.edge_index))
        return x
