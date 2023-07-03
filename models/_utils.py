from models._pkg import *


# Transformer Parts
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
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


class Block(torch.nn.Module):
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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# # # Inner Model Parts
# # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


# # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


# Models (super class and sub classes)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
class Model(torch.nn.Module):
    """Super class for all models.

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
            `get_input_size`, `get_hidden_size`, and `get_num_layers`, respectively."""

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
        # Initialize hidden state (for models that use them)
        self.hidden = None
        # Initialize the tau
        self.tau = 1  # next-timestep prediction
        # Identity layer
        self.identity = torch.nn.Identity()
        # Linear readout
        self.linear = torch.nn.Linear(self.hidden_size, self.output_size)
        torch.nn.init.zeros_(self.linear.bias)  # initialize bias to zeros
        # Initialize the readout weights
        torch.nn.init.zeros_(self.linear.weight)  # Zeros Initialization
        # torch.nn.init.xavier_uniform_(self.linear.weight) # Xavier Initialization
        # torch.nn.init.kaiming_uniform_(self.linear.weight, nonlinearity='relu') # He Initialization

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
            """
            # calculate next time step prediction loss
            # TODO: apply recency exponential decay factor to original loss
            original_loss = self.loss(**kwargs)(
                prediction[:, -self.tau :, :],
                target[:, -self.tau :, :],
            )  # only consider the new time steps
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
        context_len: int = 200,
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

    def sample(self, length):
        """
        Sample spontaneous neural activity from the network.
        TODO: Figure out how to use diffusion models to sample from the network.
        """
        pass

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


class LinearNN(Model):
    """
    TODO: Test model with masking.
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

        # Input to hidden transformation
        self.input_hidden = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(self.hidden_size),
        )

        # Mask to hidden transformation
        self.mask_hidden = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(self.hidden_size),
        )

        # Linear layer: Hidden to hidden transformation
        self.hidden_hidden = torch.nn.Sequential(
            torch.nn.Linear(
                2 * self.hidden_size, self.hidden_size
            ),  # concatenating input and mask
            torch.nn.ReLU(),
            torch.nn.LayerNorm(self.hidden_size),
            *(
                (self.num_layers - 1)
                * (
                    torch.nn.Linear(self.hidden_size, self.hidden_size),
                    torch.nn.ReLU(),
                    torch.nn.LayerNorm(self.hidden_size),
                )
            ),
        )

    @autocast()
    def forward(self, input: torch.Tensor, mask: torch.Tensor, tau: int = 1):
        """Forward method for simple linear regression model.

        Parameters
        ----------
        input : torch.Tensor
            Input data with shape (batch, seq_len, neurons)
        mask: torch.Tensor
            Mask on the neurons with shape (neurons,)
        tau : int, optional
            Time offset of target
        """
        # store the tau
        self.tau = tau
        # recast the mask to the input type and shape
        mask = torch.broadcast_to(mask.to(input.dtype), input.shape)
        # control flow for tau
        if tau < 1:
            output = self.identity(input)
        else:
            # # multiply the input by the mask
            # input = input * mask
            # transform the input
            input_hidden_out = self.input_hidden(input)
            # transform the mask
            mask_hidden_out = self.mask_hidden(mask)
            # concatenate into a single latent
            latent_out = torch.cat((input_hidden_out, mask_hidden_out), dim=-1)
            # transform the latent
            hidden_out = self.hidden_hidden(latent_out)
            # perform a linear readout to get the output
            readout = self.linear(hidden_out)
            output = readout
        # repeat for target with tau>0 offset
        for _ in range(1, tau):
            # # multiply input by the mask
            # input = output * mask
            # transform the input
            input_hidden_out = self.input_hidden(input)
            # transform the mask
            mask_hidden_out = self.mask_hidden(mask.to(output.dtype))
            # concatenate into a single latent
            latent_out = torch.cat((input_hidden_out, mask_hidden_out), dim=-1)
            # transform the  latent
            hidden_out = self.hidden_hidden(latent_out)
            # perform a linear readout to get the output
            readout = self.linear(hidden_out)
            output = readout
        return output


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
        )

        # Mask to hidden transformation
        self.mask_hidden = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_size),
            torch.nn.ReLU(),
        )

        # LSTM layer: Hidden to hidden transformation
        self.lstm = torch.nn.LSTM(
            input_size=2 * self.hidden_size,  # concatenating input and mask
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=True,
            batch_first=True,
        )

        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:  # Input-hidden weights
                torch.nn.init.xavier_uniform_(param.data, gain=1.0)
            elif "weight_hh" in name:  # Hidden-hidden weights
                torch.nn.init.orthogonal_(param.data)
            elif "bias" in name:  # Bias weights
                param.data.fill_(0)

    def init_hidden(self, input_shape):
        """
        Inititializes the hidden and cell states vectors of the LSTM.
        """
        device = next(self.parameters()).device
        batch_size = input_shape[0]  # because batch_first=True
        h0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(device)
        return (h0, c0)

    @autocast()
    def forward(self, input: torch.Tensor, mask: torch.Tensor, tau: int = 1):
        """Forward method for simple linear regression model.

        Parameters
        ----------
        input : torch.Tensor
            Input data with shape (batch, seq_len, neurons)
        mask: torch.Tensor
            Mask on the neurons with shape (neurons,)
        tau : int, optional
            Time offset of target
        """
        # initialize hidden state
        self.hidden = self.init_hidden(input.shape)
        # recast the mask to the input type and shape
        mask = torch.broadcast_to(mask.to(input.dtype), input.shape)
        # control flow for tau
        if tau < 1:  # return the input sequence
            output = self.identity(input)
        else:  # do one-step prediction
            # # multiply the input by the mask
            # input = input * mask
            # transform the input
            input_hidden_out = self.input_hidden(input)
            # transform the mask
            mask_hidden_out = self.mask_hidden(mask)
            # concatenate into a single latent
            latent_out = torch.cat((input_hidden_out, mask_hidden_out), dim=-1)
            # transform the latent
            hidden_out, self.hidden = self.lstm(latent_out, self.hidden)
            # perform a linear readout to get the output
            readout = self.linear(hidden_out)
            output = readout
        # do the remaining tau-1 steps of prediction
        for _ in range(1, tau):
            # # multiply input by the mask
            # input = output * mask
            # transform the input
            input_hidden_out = self.input_hidden(input)
            # transform the mask
            mask_hidden_out = self.mask_hidden(mask)
            # concatenate into a single latent
            latent_out = torch.cat((input_hidden_out, mask_hidden_out), dim=-1)
            # transform the latent
            hidden_out, self.hidden = self.lstm(latent_out, self.hidden)
            # perform a linear readout to get the output
            readout = self.linear(hidden_out)
            output = readout
        return output


class NeuralTransformer(Model):
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
        self.n_head = (
            2  # number of attention heads; NOTE: must be divisor of hidden_size
        )
        self.block_size = MAX_TOKEN_LEN  # maximum attention block (i.e. context) size
        self.dropout = 0.0  # dropout rate

        # Positional encoding
        self.position_encoding = PositionalEncoding(
            self.hidden_size,
            max_len=self.block_size,
            dropout=self.dropout,
        )

        # Embedding
        self.embedding = torch.nn.Linear(
            2 * self.hidden_size,
            self.hidden_size,
        )  # concating input and mask

        # Transformer blocks
        self.blocks = torch.nn.Sequential(
            *(
                Block(
                    n_embd=self.hidden_size,
                    block_size=self.block_size,
                    n_head=self.n_head,
                    dropout=self.dropout,
                )
                for _ in range(self.num_layers)
            )
        )

        # Layer normalization
        self.layer_norm = torch.nn.LayerNorm(self.hidden_size)

        # Input to hidden transformation
        self.input_hidden = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_size),
            torch.nn.ReLU(),
        )

        # Mask to hidden transformation
        self.mask_hidden = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_size),
            torch.nn.ReLU(),
        )

        # Hidden to hidden transformation
        self.hidden_hidden = torch.nn.Sequential(  # input has shape (B, T, 2*C')
            self.embedding,  # (B, T, C')
            self.position_encoding,  # (B, T, C')
            self.blocks,  # (B, T, C')
            self.layer_norm,  # (B, T, C')
        )

    @autocast()
    def forward(self, input: torch.Tensor, mask: torch.Tensor, tau: int = 1):
        """Forward method for a transformer model.
        Parameters
        ----------
        input : torch.Tensor
            Input data with shape (batch, seq_len, neurons)
        mask: torch.Tensor
            Mask on the neurons with shape (neurons,)
        tau : int, optional
            Time offset of target
        """
        # store the tau
        self.tau = tau
        # recast the mask to the input type and shape
        mask = torch.broadcast_to(mask.to(input.dtype), input.shape)
        # control flow for tau
        if tau < 1:
            output = self.identity(input)
        else:
            # # multiply the input by the mask
            # input = input * mask
            # transform the input
            input_hidden_out = self.input_hidden(input)
            # transform the mask
            mask_hidden_out = self.mask_hidden(mask)
            # concatenate into a single latent
            latent_out = torch.cat((input_hidden_out, mask_hidden_out), dim=-1)
            # transform the latent
            hidden_out = self.hidden_hidden(latent_out)
            # perform a linear readout to get the output
            readout = self.linear(hidden_out)
            output = readout
        # repeat for target with tau>0 offset
        for _ in range(1, tau):
            # # multiply input by the mask
            # input = output * mask
            # transform the input
            input_hidden_out = self.input_hidden(input)
            # transform the mask
            mask_hidden_out = self.mask_hidden(mask.to(output.dtype))
            # concatenate into a single latent
            latent_out = torch.cat((input_hidden_out, mask_hidden_out), dim=-1)
            # transform the  latent
            hidden_out = self.hidden_hidden(latent_out)
            # perform a linear readout to get the output
            readout = self.linear(hidden_out)
            output = readout
        return output


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
        The num_layers parameter is not being used in this model.
        TODO: Implement a way to use the num_layers parameter.
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
        )

        # Mask to hidden transformation
        self.mask_hidden = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_size),
            torch.nn.ReLU(),
        )

        # RNN layer: Hidden to hidden transformation
        self.rnn = CfC(
            input_size=2 * self.hidden_size,  # concatenating input and mask
            units=self.hidden_size,
            activation="relu",
        )

        # Initialize RNN weights
        for name, param in self.rnn.named_parameters():
            if "weight" in name:  # weights
                torch.nn.init.xavier_uniform_(param.data, gain=1.0)
            elif "bias" in name:  # biases
                param.data.fill_(0)

        # Initialize hidden state
        self.hidden = None

    def init_hidden(self, input_shape):
        """
        Inititializes the hidden state of the RNN.
        """
        device = next(self.parameters()).device
        batch_size = input_shape[0]  # beacuse batch_first=True
        hidden = torch.randn(batch_size, self.hidden_size).to(device)
        return hidden

    @autocast()
    def forward(self, input: torch.Tensor, mask: torch.Tensor, tau: int = 1):
        """Forward method for simple linear regression model.

        Parameters
        ----------
        input : torch.Tensor
            Input data with shape (batch, seq_len, neurons)
        mask: torch.Tensor
            Mask on the neurons with shape (neurons,)
        tau : int, optional
            Time offset of target
        """
        # initialize hidden state
        self.hidden = self.init_hidden(input.shape)
        # recast the mask to the input type and shape
        mask = torch.broadcast_to(mask.to(input.dtype), input.shape)
        # control flow for tau
        if tau < 1:  # return the input sequence
            output = self.identity(input)
        else:  # do one-step prediction
            # # multiply the input by the mask
            # input = input * mask
            # transform the input
            input_hidden_out = self.input_hidden(input)
            # transform the mask
            mask_hidden_out = self.mask_hidden(mask)
            # concatenate into a single latent
            latent_out = torch.cat((input_hidden_out, mask_hidden_out), dim=-1)
            # transform the latent
            hidden_out, self.hidden = self.rnn(latent_out, self.hidden)
            # perform a linear readout to get the output
            readout = self.linear(hidden_out)
            output = readout
        # do the remaining tau-1 steps of prediction
        for _ in range(1, tau):
            # # multiply input by the mask
            # input = output * mask
            # transform the input
            input_hidden_out = self.input_hidden(input)
            # transform the mask
            mask_hidden_out = self.mask_hidden(mask)
            # concatenate into a single latent
            latent_out = torch.cat((input_hidden_out, mask_hidden_out), dim=-1)
            # transform the latent
            hidden_out, self.hidden = self.rnn(latent_out, self.hidden)
            # perform a linear readout to get the output
            readout = self.linear(hidden_out)
            output = readout
        return output


# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# # Graph Neural Network models
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# class GraphNN(torch.nn.Module):
#     """
#     Applies a single graph convolutional layer separately to the
#     two graphs induced by using only chemical synapses or
#     gap junctions, respectively, as the edges.
#     """

#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(GraphNN, self).__init__()
#         # a separte learnable conv. layer for each type of synapse
#         self.elec_conv = GCNConv(
#             in_channels=input_dim, out_channels=hidden_dim, improved=True
#         )
#         self.chem_conv = GCNConv(
#             in_channels=input_dim, out_channels=hidden_dim, improved=True
#         )
#         # readout layer transforms node features to output
#         self.linear = torch.nn.Linear(
#             in_features=2 * hidden_dim, out_features=output_dim
#         )

#     def forward(self, x, edge_index, edge_attr):
#         elec_weight = edge_attr[:, 0]
#         chem_weight = edge_attr[:, 1]
#         elec_hid = self.elec_conv(x, edge_index, elec_weight)
#         chem_hid = self.chem_conv(x, edge_index, chem_weight)
#         hidden = torch.cat([elec_hid, chem_hid], dim=-1)
#         output = self.linear(hidden)
#         return output

# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
