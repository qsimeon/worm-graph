from models._pkg import *

# Init logger
logger = logging.getLogger(__name__)


### Custom loss function (MASE) # # #
# # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
class MASELoss(torch.nn.Module):
    """
    Mean Absolute Scaled Error (MASE) Loss Function.
    Supports 'none', 'mean', and 'sum' reductions.
    ---
    Example usage:
    mase_loss = MASELoss(reduction='mean')
    loss = mase_loss(y_pred, target)
    """

    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_pred, target):
        """
        Forward pass for MASE Loss.

        Parameters:
            y_pred (torch.Tensor): Predicted values with shape (batch_size, seq_len, num_features)
            target (torch.Tensor): Actual values with shape (batch_size, seq_len, num_features)

        Returns:
            torch.Tensor: The MASE loss.
        """
        # Ensure the target and predictions have the same shape
        assert (
            y_pred.shape == target.shape
        ), "y_pred and target must have the same shape"
        # Calculate the Mean Absolute Error of the one-step naive forecast
        if target.ndim == 2:  # if 1-timestep
            mean_naive_error = torch.tensor(1.0)
        else:  # if sequence
            naive_forecast_errors = torch.abs(target[:, 1:, :] - target[:, :-1, :])
            mean_naive_error = 1e-8 + torch.mean(
                naive_forecast_errors, dim=1, keepdim=True
            )  # Average over seq_len

        # Calculate the Mean Absolute Error of the predictions
        prediction_errors = torch.abs(y_pred - target).expand_as(target)
        mase_errors = prediction_errors / mean_naive_error

        # Apply reduction
        if self.reduction == "none":
            return mase_errors
        elif self.reduction == "mean":
            return torch.mean(mase_errors)
        elif self.reduction == "sum":
            return torch.sum(mase_errors)
        else:
            raise ValueError(f"Invalid reduction type: {self.reduction}")


# # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


# # # Helper functions # # #
# # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


def find_largest_divisor(hidden_size):
    # Set the maximum number of heads
    max_heads = 5
    # Iterate backwards from 5 down to 1 to find the largest divisor
    for n in range(max_heads, 0, -1):
        if hidden_size % n == 0:
            return n
    # If no divisor found between 1 and 5, default to 1
    return 1


def print_parameters(model, verbose=False):
    table = PrettyTable(["Module", "Parameters", "Trainable"])

    total_params = 0
    total_trainable = 0

    for name, parameter in model.named_parameters():
        num_params = torch.prod(torch.tensor(parameter.size())).item()
        total_params += num_params

        trainable = parameter.requires_grad
        if trainable:
            total_trainable += num_params

        table.add_row([name, num_params, trainable])

    if verbose:
        print(table)
        print("Total Parameters:", total_params)
        print("Total Trainable Parameters:", total_trainable)
    return total_params, total_trainable


# # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


# # # "Cores" or Inner Models for Different Model Architectures # # #
# # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


class PositionalEncoding(torch.nn.Module):
    """
    Sinuosoidal positional encoding from Attention is All You Need paper,
    with the minor modification that we use the first dimension as the batch
    dimension (i.e. batch_first=True).
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 1000,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = torch.nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)  # batch_first=True
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape (batch_size, seq_len, embedding_dim)
        """
        ### DEBUG ###
        try:
            x = x * math.sqrt(self.d_model)  # DEBUG: is this important?
            x = x + self.pe[:, : x.size(1), :]  # add positional encoding to input
        except Exception as e:
            logger.info(
                f"DEBUG (x, d_model, pe): {x.shape, self.d_model, self.pe.shape}"
            )
            logger.error(f"The error that occurred: {e}")
        ### DEBUG ###
        return self.dropout(x)


class CausalTransformer(torch.nn.Module):
    """
    We use a single Transformer Encoder layer as the hidden-hidden model.
    Sets `is_causal=True` in forward method of TransformerEncoderLayer.
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.transformer = torch.nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.is_causal = True

    def forward(self, src):
        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(
            src.size(1),
            device=src.device,
        )
        out = self.transformer(src, src_mask=causal_mask, is_causal=self.is_causal)
        return out


class FeedForward(torch.nn.Module):
    """
    A simple linear layer followed by a non-linearity and dropout.
    n_embd: embedding dimension or width of the single hidden layer.
    dropout: probability of dropping a neuron.
    """

    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.ffwd = torch.nn.Sequential(
            torch.nn.Linear(n_embd, n_embd),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        Uses residual ("skip") connection.
        """
        x = x + self.ffwd(x)
        return x


class CTRNN(torch.nn.Module):
    """Continuous-time RNN.

    Parameters:
        input_size: Number of input neurons
        hidden_size: Number of hidden neurons

    Inputs:
        input: tensor of shape (seq_len, batch, input_size)
        hidden: tensor of shape (batch, hidden_size), initial hidden activity
            if None, hidden is initialized through self.init_hidden()

    Outputs:
        output: tensor of shape (batch, seq_len, hidden_size)
        hidden: tensor of shape (batch, hidden_size), final hidden activity
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.register_parameter(
            name="alpha", param=torch.nn.Parameter(torch.ones(1, hidden_size))
        )
        self.input2h = torch.nn.Linear(input_size, hidden_size)
        self.h2h = torch.nn.Linear(hidden_size, hidden_size)

    def init_hidden(self, input_shape):
        device = next(self.parameters()).device
        batch_size = input_shape[0]  # because batch_first=True
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
        # the sigmoid contrains alpha such that 0 <= alpha <=1
        h_new = hidden * (1 - self.alpha.sigmoid()) + h_new * self.alpha.sigmoid()
        return h_new

    def forward(self, input, hidden=None):
        """
        Propagate input through the network.
        NOTE: Because we use batch_first=True, input has shape (batch, seq_len, input_size).
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


# # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


### A commmon interface that encapsulates the "Core" of Inner Model of different architectures ###
# # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


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

# # # Model super class: Common interface all model architectures # # # #
# Provides the input-output backbone and allows changeable inner "cores". #
# # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


class Model(torch.nn.Module):
    """
    Super class for all models.

    For all our models:
        1. The output `readout` will be the same shape as the input.
        2. A method called `loss_fn` that specifies the specific
            loss function to be used by the model. The default
            loss function we use is `torch.nn.MSELoss()`.
        3. A readout layer is implemented and will always be
            called `self.linear`.
        4. The core of all models is called `self.hidden_hidden` and it is
            comprised of a single hidden layer of an architecture of choice.
        7. Getter methods for the input size and hidden size called
            `get_input_size`, and `get_hidden_size`, respectively.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: Union[int, None],
        loss: Union[Callable, None] = None,
        fft_reg_param: float = 0.0,
        l1_reg_param: float = 0.0,
    ):
        """
        Defines attributes common to all models.
        """
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
        elif str(loss).lower() == "mase":
            self.loss = MASELoss
        else:
            self.loss = torch.nn.MSELoss

        # Name of original loss function
        self.loss_name = self.loss.__name__[:-4]
        # Setup
        self.input_size = input_size  # Number of neurons (302)
        self.output_size = input_size  # Number of neurons (302)
        self.hidden_size = hidden_size if hidden_size is not None else input_size
        self.fft_reg_param = fft_reg_param
        self.l1_reg_param = l1_reg_param
        # Initialize hidden state
        self._init_hidden()
        # Identity layer
        self.identity = torch.nn.Identity()
        # Input to hidden transformation - placeholder
        self.input_hidden = (
            torch.nn.Linear(self.input_size, self.hidden_size)
            if hidden_size is not None
            else torch.nn.Identity()
        )
        # Hidden to hidden transformation - placeholder
        self.hidden_hidden = (
            torch.nn.Linear(self.hidden_size, self.hidden_size)
            if hidden_size is not None
            else torch.nn.Identity()
        )
        # Instantiate internal hidden model - placeholder
        self.inner_hidden_model = InnerHiddenModel(self.hidden_hidden, self.hidden)
        # Linear readout
        self.linear = torch.nn.Linear(self.hidden_size, self.output_size)
        ### DEBUG ###
        # Optional layer normalization
        self.layer_norm = torch.nn.LayerNorm(self.hidden_size, elementwise_affine=True)
        ### DEBUG ###
        # Initialize weights
        self._init_weights()

    # Initialization functions for setting hidden states and weights.
    def _init_hidden(self):
        self.hidden = None
        return None

    def _init_weights(self):
        # Initialize the readout bias
        torch.nn.init.zeros_(self.linear.bias)
        # Initialize the readout weights
        # torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.xavier_uniform_(self.linear.weight)  # Xavier Initialization
        # torch.nn.init.kaiming_uniform_(self.linear.weight, nonlinearity='relu') # He Initialization
        return None

    def init_hidden(self, input_shape=None):
        raise NotImplementedError()

    # Getter functions for returning all attributes needed to reinstantiate a similar model
    def get_input_size(self):
        return self.input_size

    def get_hidden_size(self):
        return self.hidden_size

    def get_loss_name(self):
        return self.loss_name

    def get_fft_reg_param(self):
        return self.fft_reg_param

    def get_l1_reg_param(self):
        return self.l1_reg_param

    @torch.autocast(device_type=DEVICE.type, dtype=torch.half)
    def forward(self, input: torch.Tensor, mask: torch.Tensor):
        """
        Forward method for simple linear regression model.

        Parameters
        ----------
        input : torch.Tensor
            Input data with shape (batch, seq_len, neurons)
        mask : torch.Tensor
            Mask on the neurons with shape (batch, neurons)
        """
        # initialize hidden state
        self.hidden = self.init_hidden(input.shape)
        # set hidden state of internal model
        self.inner_hidden_model.set_hidden(self.hidden)
        # recast the mask to the input type and shape
        mask = mask.unsqueeze(1).expand_as(input)
        # multiply input by the mask
        input = self.identity(input * mask)
        # transform the input
        input_hidden_out = self.input_hidden(input)
        # concatenate into a single latent
        latent_out = input_hidden_out
        # transform the latent
        hidden_out = self.inner_hidden_model(latent_out)
        # perform a linear readout to get the output
        readout = self.linear(hidden_out)
        return readout

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
            Calculate loss with added FFT and L1 regularization
            on the trainable model parameters.
            Arguments:
                prediction: (batch_size, seq_len, input_size)
                target: (batch_size, seq_len, input_size)
            """
            # calculate next time step prediction loss
            original_loss = self.loss(reduction="none", **kwargs)(
                prediction,
                target,
            )
            # FFT regularization term
            fft_loss = 0.0
            if self.fft_reg_param > 0.0:
                # calculate FFT and take the real part
                input_fft = torch.fft.rfft(prediction, dim=-2).real
                target_fft = torch.fft.rfft(target, dim=-2).real
                # calculate average difference between real parts of FFTs
                fft_loss += torch.abs(input_fft - target_fft).mean()
            # L1 regularization term
            l1_loss = 0.0
            if self.l1_reg_param > 0.0:
                # calculate L1 regularization term for all weights
                for param in self.parameters():
                    l1_loss += torch.abs(param).mean()
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
        input: torch.Tensor,
        mask: torch.Tensor,
        nb_ts_to_generate: int,
        context_window: int,
        autoregressive: bool = True,
    ):
        """
        Generate future neural activity from the model.

        Parameters
        ----------
        input : torch.Tensor
            Input data with shape (batch_size, seq_len, neurons)
        mask : torch.Tensor
            Mask on the neurons with shape (batch_size, neurons)
        nb_ts_to_generate : int
            Number of time steps to generate
        context_window : int
            Number of time steps to use as context

        Returns
        -------
        generated_tensor : torch.Tensor
            Generated data with shape (nb_ts_to_generate, neurons)
        """

        # Set model to evaluation mode
        self.eval()

        # If generating values autoregressively
        if autoregressive:
            input = input[
                :, :context_window, :
            ]  # shape (batch_size, context_window, neurons)
            ### DEBUG ###
            # Create a normalizer for the input
            normalizer = torch.nn.LayerNorm(context_window, elementwise_affine=False)
            ### DEBUG ###

        # Otherwise defaults to ground-truth feeding
        generated_values = []
        with torch.no_grad():
            # Loop through time
            for t in range(nb_ts_to_generate):
                # Get the last context_window values of the input tensor
                x = input[
                    :, t : context_window + t, :
                ]  # shape (batch_size, context_window, neurons)
                ### DEBUG ###
                if autoregressive and t > 0:
                    # Normalize the input along the temporal dimension
                    x = normalizer(
                        x.view(
                            -1,
                            self.input_size,
                            context_window,
                        )
                    ).view(-1, context_window, self.input_size)
                ### DEBUG ###

                # Get predictions
                predictions = self(
                    x, mask
                )  # shape (batch_size, context_window, neurons)

                # Get last predicted value
                last_time_step = predictions[:, -1, :].unsqueeze(
                    0
                )  # shape (batch_size, 1, neurons)

                # Append the prediction to the generated_values list and input tensor
                generated_values.append(last_time_step)
                input = torch.cat(
                    (input, last_time_step), dim=1
                )  # add the prediction to the input tensor

        # Stack the generated values to a tensor
        generated_tensor = torch.cat(
            generated_values, dim=1
        )  # shape (batch_size, nb_ts_to_generate, neurons)

        return generated_tensor

    def sample(self, nb_ts_to_sample: int):
        """
        Sample spontaneous neural activity from the model.
        TODO: Figure out how to use diffusion models to do this.
        """
        pass


# # # Models subclasses: Individually differentiated model architectures # # # #
# Use the same model backbone provided by Model but with a distinct core or inner hidden model. #
# # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


class NaivePredictor(Model):
    """
    A parameter-less model that simply copies the input as its output.
    Serves as our baseline model. Memory-less and feature-less.
    NOTE: This model will throw an error if you try to train it.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: Union[int, None] = None,
        loss: Union[Callable, None] = None,
        fft_reg_param=0.0,
        l1_reg_param=0.0,
    ):
        super(NaivePredictor, self).__init__(
            input_size,
            None,  # hidden_size
            loss,
            fft_reg_param,
            l1_reg_param,
        )

        # Input to hidden transformation
        self.input_hidden = torch.nn.Identity()

        # Hidden to hidden transformation
        self.hidden_hidden = torch.nn.Identity()
        # Instantiate internal hidden model
        self.inner_hidden_model = InnerHiddenModel(self.hidden_hidden, self.hidden)

        ### DEBUG ###
        # Override the linear readout
        self.linear = torch.nn.Identity()

        ### DEBUG ###
        # Create a dud parameter to avoid errors
        self._ = torch.nn.Parameter(torch.tensor(0.0))

    def init_hidden(self, input_shape=None):
        return None


class LinearRegression(Model):
    """
    A simple linear regression model.
    Memory-less but can learn linear feature regression.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: Union[int, None] = None,
        loss: Union[Callable, None] = None,
        fft_reg_param=0.0,
        l1_reg_param=0.0,
    ):
        super(LinearRegression, self).__init__(
            input_size,
            None,  # hidden_size
            loss,
            fft_reg_param,
            l1_reg_param,
        )

        # Input to hidden transformation
        self.input_hidden = torch.nn.Identity()

        # Hidden to hidden transformation
        self.hidden_hidden = torch.nn.Identity()
        # Instantiate internal hidden model
        self.inner_hidden_model = InnerHiddenModel(self.hidden_hidden, self.hidden)

    def init_hidden(self, input_shape=None):
        return None


class FeatureFFNN(Model):
    """
    A simple nonlinear regression model.
    FFNN stands for FeedForward Neural Network.
    Unlike the LSTM and Transformer models, this
    model has no temporal memory and can only learn
    a fixed nonlinear feature regression function that
    it applies at every time step independently.
    Memory-less but can learn nonlinear feature regression.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: Union[int, None] = None,
        loss: Union[Callable, None] = None,
        fft_reg_param: float = 0.0,
        l1_reg_param: float = 0.0,
    ):
        super(FeatureFFNN, self).__init__(
            input_size,
            hidden_size,
            loss,
            fft_reg_param,
            l1_reg_param,
        )
        # Special parameters for this model
        self.dropout = 0.1  # dropout rate

        # Embedding
        self.embedding = torch.nn.Linear(
            self.input_size,
            self.hidden_size,
        )  # combine input and mask

        # Input to hidden transformation
        self.input_hidden = torch.nn.Sequential(
            self.embedding,
            torch.nn.ReLU(),
            # NOTE: Do NOT use LayerNorm here!
        )

        # Hidden to hidden transformation: FeedForward layer
        self.hidden_hidden = FeedForward(
            n_embd=self.hidden_size,
            dropout=self.dropout,
        )
        # Instantiate internal hidden model
        self.inner_hidden_model = InnerHiddenModel(self.hidden_hidden, self.hidden)

    def init_hidden(self, input_shape=None):
        return None


class NeuralTransformer(Model):
    """
    Transformer model for neural activity data.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: Union[int, None] = None,
        loss: Union[Callable, None] = None,
        fft_reg_param: float = 0.0,
        l1_reg_param: float = 0.0,
    ):
        """
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
            loss,
            fft_reg_param,
            l1_reg_param,
        )

        # Special transformer parameters
        self.n_head = find_largest_divisor(
            hidden_size
        )  # number of attention heads (NOTE: must be divisor of `hidden_size`)
        logger.info(f"Num. attn. heads {self.n_head}")
        self.dropout = 0.1  # dropout ratedropout=self.dropout,

        # # Positional encoding
        # self.positional_encoding = PositionalEncoding(
        #     self.input_size, # if positional_encoding before embedding
        #     dropout=self.dropout,
        # )

        # Embedding
        self.embedding = torch.nn.Linear(
            self.input_size,
            self.hidden_size,
        )  # combine input and mask

        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            self.hidden_size,  # if positional_encoding after embedding
            dropout=self.dropout,
        )

        # Input to hidden transformation
        self.input_hidden = torch.nn.Sequential(
            # # NOTE: Positional encoding before embedding improved performance.
            # self.positional_encoding,
            self.embedding,
            self.positional_encoding,  # DEBUG: Is positional_encoding after better?
            torch.nn.ReLU(),
            # NOTE: Do NOT use LayerNorm here! (it's already in the TransformerEncoderLayer)
        )

        # Hidden to hidden transformation: TransformerEncoderLayer
        self.hidden_hidden = CausalTransformer(
            d_model=self.hidden_size,
            nhead=self.n_head,
            dim_feedforward=self.hidden_size,
            dropout=self.dropout,
        )

        # Instantiate internal hidden model
        self.inner_hidden_model = InnerHiddenModel(self.hidden_hidden, self.hidden)

    def init_hidden(self, input_shape=None):
        return None


class NetworkCTRNN(Model):
    """
    A model of the C. elegans nervous system using a continuous-time RNN backbone.
    TODO: Cite tutorial by Guangyu Robert Yang and the paper: Artificial Neural Networks for Neuroscientists: A Primer.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: Union[int, None] = None,
        loss: Union[Callable, None] = None,
        fft_reg_param: float = 0.0,
        l1_reg_param: float = 0.0,
    ):
        """
        The output size will be the same as the input size.
        """
        super(NetworkCTRNN, self).__init__(
            input_size,
            hidden_size,
            loss,
            fft_reg_param,
            l1_reg_param,
        )

        # Input to hidden transformation
        self.input_hidden = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_size),
            torch.nn.ReLU(),
            # NOTE: YES use LayerNorm here!
            self.layer_norm,
        )

        # Hidden to hidden transformation: Continuous time RNN (CTRNN) layer
        self.hidden_hidden = CTRNN(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,  # combine input and mask
        )
        # Instantiate internal hidden model
        self.inner_hidden_model = InnerHiddenModel(self.hidden_hidden, self.hidden)

    def init_hidden(self, input_shape):
        device = next(self.parameters()).device
        batch_size = input_shape[0]  # because batch_first=True
        hidden = torch.zeros(batch_size, self.hidden_size).to(device)
        return hidden


class LiquidCfC(Model):
    """
    Neural Circuit Policy (NCP) Closed-form continuous time (CfC) model.
    TODO: Cite Nature Machine Intelligence 2022 paper by Ramin Hasani, Daniela Rus et al.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: Union[int, None] = None,
        loss: Union[Callable, None] = None,
        fft_reg_param: float = 0.0,
        l1_reg_param: float = 0.0,
    ):
        """
        The output size will be the same as the input size.
        """

        super(LiquidCfC, self).__init__(
            input_size,
            hidden_size,
            loss,
            fft_reg_param,
            l1_reg_param,
        )

        # Input to hidden transformation
        self.input_hidden = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_size),
            torch.nn.ReLU(),
            # NOTE: YES use LayerNorm here!
            self.layer_norm,
        )

        # Hidden to hidden transformation: Closed-form continuous-time (CfC) layer
        self.hidden_hidden = CfC(
            input_size=self.hidden_size,  # combine input and mask
            units=self.hidden_size,
            activation="relu",
        )

        # Instantiate internal hidden model
        self.inner_hidden_model = InnerHiddenModel(self.hidden_hidden, self.hidden)

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
                torch.nn.init.zeros_(param.data)


class NetworkLSTM(Model):
    """
    A model of the _C. elegans_ neural network using an LSTM.
    Given an input sequence of length $L$ and an offset this
    model is trained to output the sequence of length $L$ that
    occurs 1 time steps after the start of the input sequence.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: Union[int, None] = None,
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
            loss,
            fft_reg_param,
            l1_reg_param,
        )

        # Input to hidden transformation
        self.input_hidden = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_size),
            torch.nn.ReLU(),
            # NOTE: YES use LayerNorm here!
            self.layer_norm,
        )

        # Hidden to hidden transformation: Long-short term memory (LSTM) layer
        self.hidden_hidden = torch.nn.LSTM(
            input_size=self.hidden_size,  # combine input and mask
            hidden_size=self.hidden_size,
            bias=True,
            batch_first=True,
        )

        # Instantiate internal hidden model
        self.inner_hidden_model = InnerHiddenModel(self.hidden_hidden, self.hidden)

        # Initialize LSTM weights
        self.init_weights()

    def init_hidden(self, input_shape):
        """
        Inititializes the hidden and cell states of the LSTM.
        """
        device = next(self.parameters()).device
        batch_size = input_shape[0]  # because batch_first=True
        h0 = torch.zeros(1, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(1, batch_size, self.hidden_size).to(device)
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


# # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
