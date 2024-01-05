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
            mean_naive_error = 1e-16 + torch.mean(
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


def load_model_from_checkpoint(checkpoint_path):
    """
    Load a model from a checkpoint file.

    Args:
        checkpoint_path (str): The path to the checkpoint file.

    Returns:
        model: The loaded model.
    """
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(ROOT_DIR, checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model_name = checkpoint["model_name"]
    input_size = checkpoint["input_size"]
    hidden_size = checkpoint["hidden_size"]
    loss_name = checkpoint["loss_name"]
    l1_reg_param = checkpoint["l1_reg_param"]
    model_state_dict = checkpoint["model_state_dict"]
    model = eval(model_name)(
        input_size,
        hidden_size,
        loss=loss_name,
        l1_reg_param=l1_reg_param,
    ).to(DEVICE)
    model.load_state_dict(model_state_dict)
    return model


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


class MultiChannelReadout(torch.nn.Module):
    """
    A module for readout logits for each input channel.
    """

    def __init__(self, num_channels, hidden_size, output_size):
        super().__init__()
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.readouts = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.hidden_size, self.output_size)
                for _ in range(self.num_channels)
            ]
        )

    def forward(self, x):
        """
        Args:
            x: Tensor, shape (batch_size, seq_len, hidden_size)
        Output:
            output: Tensor, shape (batch_size, seq_len, output_size)
        """
        output = []
        for _ in range(self.num_channels):
            output.append(self.readouts[_](x[:, :, _]))
        return torch.hstack(output)


class MultiChannelEmbedding(torch.nn.Module):
    """
    A module for obtaining an embedding for each channel/dimension of a
    high-dimensional input tensor w/ shape (batch_size, seq_len, input_size).
    Why? Because if using high-dimensional tokens, we will need an embedding
    for each feature dimension.
    """

    def __init__(self, num_channels, num_embeddings, embedding_dim):
        super().__init__()
        self.num_channels = num_channels
        self.num_embeddings = (
            num_embeddings + 1
        )  # 0-th index token used for unmasked values
        self.embedding_dim = embedding_dim
        self.embeddings = torch.nn.ModuleList(
            [
                torch.nn.Embedding(self.num_embeddings, self.embedding_dim)
                for _ in range(self.num_channels)
            ]
        )

    @torch.autocast(device_type=DEVICE.type, dtype=torch.long)
    def forward(self, x):
        """
        Args:
            x: Tensor, shape (batch_size, seq_len, num_channels)
        Output:
            output: Tensor, shape (batch_size, seq_len, embedding_dim)
        """
        batch_size, seq_len, num_channels = x.shape

        # Reshape x to a 2D tensor for embedding lookup
        x_reshaped = x.view(-1, num_channels)  # (batch_size * seq_len, num_channels)

        # Perform embedding lookup for each channel and concatenate
        embedded = torch.cat(
            [self.embeddings[i](x_reshaped[:, i]) for i in range(num_channels)], dim=-1
        )

        # Reshape back to the original batch_size and seq_len, sum across channels
        output = embedded.view(
            batch_size, seq_len, num_channels, self.embedding_dim
        ).sum(dim=2)
        logger.info(f"FINISHED MultiChannelEmbedding.forward\n\n")  # DEBUG
        return output

    # @torch.autocast(device_type=DEVICE.type, dtype=torch.long)
    # def forward(self, x):
    #     """
    #     Args:
    #         x: Tensor, shape (batch_size, seq_len, num_channels)
    #     Output:
    #         output: Tensor, shape (batch_size, seq_len, embedding_dim)
    #     """
    #     # Embed each feature and then sum them together to get the final embedding
    #     output = 0
    #     for _ in range(self.num_channels):
    #         output += self.embeddings[_](
    #             x[:, :, _]
    #         )  # (batch_size, seq_len, hidden_size)
    #     logger.info(f"FINISHED MultiChannelEmbedding.forward\n\n")  # DEBUG
    #     return output


class PositionalEncoding(torch.nn.Module):
    """
    Sinuosoidal positional encoding from Attention is All You Need paper,
    with the minor modification that we use the first dimension as the batch
    dimension (i.e. batch_first=True).
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = BLOCK_SIZE,
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
        x = x * math.sqrt(
            self.d_model
        )  # normalization used in the original transformer paper
        x = x + self.pe[:, : x.size(1), :]  # add positional encoding to input
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
        l1_reg_param: float = 0.0,
        v2: bool = True,
    ):
        """
        Defines attributes common to all models.
        """
        super(Model, self).__init__()
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
        # Embedding layer
        self.embedding = torch.nn.Linear(
            self.input_size,
            self.hidden_size,
        )
        # Linear readout
        self.linear = torch.nn.Linear(self.hidden_size, self.output_size)
        # Optional layer normalization
        self.layer_norm = torch.nn.LayerNorm(self.hidden_size, elementwise_affine=True)
        # Initialize weights
        self._init_weights()

        ## >>> DEBUG: modifications for new token mode >>> ###
        self.v2 = v2
        if self.v2:
            # New attributes and parameters for this version
            self.n_token = (
                NUM_TOKENS  # number of tokens to approximate continuous values
            )
            self.output_size = self.n_token  # modify output size to be number of tokens
            self.random_projection = torch.nn.Parameter(
                torch.randn(self.n_token, self.input_size), requires_grad=False
            )  # fixed random projection matrix (not learned, not updated)

            # self.token_neural_map = torch.nn.Parameter(
            #     torch.zeros(self.n_token, self.input_size), requires_grad=False
            # )  # mapping of tokens to neural vectors (not learned but is updated)
            ### >>> DEBUG: multi-dimensional tokenization and embedding >>> ###
            self.token_neural_map = torch.nn.Parameter(
                torch.zeros(self.n_token, 1), requires_grad=False
            )  # mapping of tokens to neural values (not learned but is updated)
            ### <<< DEBUG: multi-dimensional tokenization and embedding <<< ###

            # Modify embedding layer to be a lookup table
            # self.embedding = torch.nn.Embedding(
            #     num_embeddings=self.n_token, embedding_dim=self.hidden_size
            # )  # embedding lookup table (learned)
            ### >>> DEBUG: multi-dimensional tokenization and embedding >>> ###
            self.embedding = MultiChannelEmbedding(
                num_channels=self.input_size,
                num_embeddings=self.n_token,
                embedding_dim=self.hidden_size,
            )  # embedding lookup table (learned)
            ### <<< DEBUG: multi-dimensional tokenization and embedding <<< ###

            # Adjust linear readout to output token logits
            self.linear = torch.nn.Linear(self.hidden_size, self.n_token)

            # Re-initialize weights
            self._init_weights()

            # Alias methods to new versions
            self.forward = self.forward_v2
            self.loss_fn = self.loss_fn_v2
            self.generate = self.generate_v2
        ### <<< DEBUG: modifications for new token mode <<< ###

    # Initialization functions for setting hidden states and weights.
    def _init_hidden(self):
        self.hidden = None
        return None

    def _init_weights(self):
        # Initialize the readout bias
        torch.nn.init.zeros_(self.linear.bias)
        # Initialize the readout weights
        # torch.nn.init.zeros_(self.linear.weight) # Zero Initialization
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

    def get_l1_reg_param(self):
        return self.l1_reg_param

    def calculate_distances(self, neural_sequence, token_matrix, feature_mask=None):
        """
        Helper method to calculate Euclidean distances between neural sequence vectors and token matrix vectors.

        Args:
            neural_sequence (torch.Tensor): Shape (batch_size, seq_len, input_size).
            token_matrix (torch.Tensor): Shape (num_tokens, input_size).
            feature_mask (torch.Tensor, optional): Shape (batch_size, input_size). If None, all features are considered.

        Returns:
            torch.Tensor: Distances for each batch. Shape (batch_size, seq_len, num_tokens).

        The function computes distances considering only the selected features by the mask for each batch.
        """
        batch_size, seq_len, input_size = neural_sequence.shape
        num_tokens = token_matrix.shape[0]

        if feature_mask is None:
            feature_mask = torch.ones(
                (batch_size, input_size),
                dtype=torch.bool,
                device=neural_sequence.device,
            )

        # Initialize distances tensor
        distances = torch.empty(
            (batch_size, seq_len, num_tokens), device=neural_sequence.device
        )

        # Compute distances by looping over each batch
        # NOTE: We NEED to do this because each batch item may use a different feature mask
        # start_time = time.time()
        for b in range(batch_size):
            V = feature_mask[b]  # (input_size,)
            assert V.sum().item() > 0, "Feature mask cannot be all False."
            S = neural_sequence[b].unsqueeze(1)  # (seq_len, 1, input_size)
            M = token_matrix.unsqueeze(0)  # (1, num_tokens, input_size)
            D = S - M  # (seq_len, num_tokens, input_size)
            dist = torch.linalg.vector_norm(
                D[V.expand_as(D)].view(seq_len, num_tokens, -1),
                dim=-1,
            )  # (seq_len, num_tokens)
            distances[b] = dist

        # Return the distances matrix
        return distances

    def tokenize_neural_data(
        self,
        neural_sequence: torch.Tensor,
        feature_mask: Union[None, torch.Tensor] = None,
        token_matrix: Union[None, torch.Tensor] = None,
    ):
        """
        Convert the high-dimensional sequence of neural states to a 1-D sequence of tokens.
        Args:
            neural_sequence: tensor of shape (batch_size, seq_len, input_size)
            feature_mask: tensor of shape (batch_size, input_size)
            token_matrix: tensor of shape (num_tokens, input_size)
        Output:
            token_sequence: tensor of shape (batch_size, seq_len)
        """
        # Ensure inputs are the correct shapes
        assert (
            neural_sequence.ndim == 3 and neural_sequence.shape[-1] == self.input_size
        ), "`neural_sequence` must have shape (batch_size, seq_len, input_size)"
        batch_size, seq_len, input_size = neural_sequence.shape
        if feature_mask is None:
            feature_mask = torch.ones(
                (batch_size, input_size),
                dtype=torch.bool,
                device=neural_sequence.device,
            )
        assert (
            feature_mask.ndim == 2 and feature_mask.shape[-1] == self.input_size
        ), "`feature_mask` must have shape (batch_size, input_size)"
        assert feature_mask.sum().item() > 0, "`feature_mask` cannot be all False."
        if token_matrix is None:
            token_matrix = self.random_projection.to(neural_sequence.device)
        assert (
            token_matrix.ndim == 2 and token_matrix.shape[-1] == self.input_size
        ), "`token_matrix` must have shape (num_tokens, input_size)"
        num_tokens = token_matrix.shape[0]

        # Use the `calculate_distance` method
        distances = self.calculate_distances(
            neural_sequence,
            token_matrix,
            feature_mask,
        )  # (batch_size, seq_len, num_tokens)

        # Find the minimum indices along the num_tokens dimension
        token_sequence = distances.argmin(dim=-1)  # (batch_size, seq_len)

        # Ensure the data type is long tensor
        token_sequence = token_sequence.to(torch.long)

        # Update the mapping of tokens to neural vectors
        # NOTE: The `token_neural_map` is used purely for generation
        for token in torch.unique(token_sequence).tolist():
            self.token_neural_map[token] = 0.8 * self.token_neural_map[
                token
            ] + 0.2 * neural_sequence[token_sequence == token].mean(dim=0)
        logger.info(
            f"self.token_neural_map[token]:\t{self.token_neural_map[token].shape, self.token_neural_map[token].dtype}\n{self.token_neural_map[token]}\n\n"
        )  # DEBUG

        # Return the tokenized sequence with shape (batch_size, seq_len)
        return token_sequence

    def tokenize_neural_data_multi_channel(
        self,
        neural_sequence: torch.Tensor,
        feature_mask: Union[None, torch.Tensor] = None,
        num_tokens: int = NUM_TOKENS,
    ):
        """
        An alternative method for tokenizing the neural data that
        tokenizes each feature dimension independently.
        Args:
            neural_sequence: tensor of shape (batch_size, seq_len, input_size)
            feature_mask: tensor of shape (batch_size, input_size)
            num_tokens: int > input_size
        Output:
            token_sequence: tensor of shape (batch_size, seq_len, input_size)
        """
        # Check input sizes
        batch_size, seq_len, input_size = neural_sequence.shape
        assert (
            neural_sequence.ndim == 3 and neural_sequence.shape[-1] == self.input_size
        ), "`neural_sequence` must have shape (batch_size, seq_len, input_size)"
        batch_size, seq_len, input_size = neural_sequence.shape
        if feature_mask is None:
            feature_mask = torch.ones(
                (batch_size, input_size),
                dtype=torch.bool,
                device=neural_sequence.device,
            )
        assert (
            feature_mask.ndim == 2 and feature_mask.shape[-1] == self.input_size
        ), "`feature_mask` must have shape (batch_size, input_size)"
        assert feature_mask.sum().item() > 0, "`feature_mask` cannot be all False."
        assert num_tokens > input_size, "`num_tokens` must be greater than `input_size`"

        # Create bin edges
        bin_edges = torch.tensor(
            norm.ppf(torch.linspace(0, 1, num_tokens + 1)),
            device=neural_sequence.device,
        )

        # Expand bin_edges for broadcasting
        bin_edges_expanded = bin_edges.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # Initialize the output tensor
        token_tensor = torch.zeros_like(neural_sequence, dtype=torch.long)

        # Apply binning using broadcasting
        bool_arr = (
            neural_sequence.unsqueeze(-1) > bin_edges_expanded[:, :, :, :-1]
        ) * (neural_sequence.unsqueeze(-1) <= bin_edges_expanded[:, :, :, 1:])
        token_tensor = bool_arr.to(torch.int64).argmax(dim=-1)

        # Apply the feature mask
        token_tensor *= feature_mask.unsqueeze(1).expand_as(token_tensor)

        # Ensure the data type is long tensor
        token_tensor = token_tensor.to(torch.long)

        # Update the mapping of tokens to neural values
        # NOTE: The `token_neural_map` is used purely for generation
        for token in torch.unique(token_tensor).tolist():
            self.token_neural_map[token] = (
                0.8 * self.token_neural_map[token]
                + 0.2 * neural_sequence[token_tensor == token].mean()
            )
        logger.info(
            f"self.token_neural_map[token]:\t{self.token_neural_map[token].shape, self.token_neural_map[token].dtype}\n{self.token_neural_map[token]}\n\n"
        )  # DEBUG

        # Return the tokenized data tensor with shape (batch_size, seq_len, input_size)
        return token_tensor

    @torch.autocast(device_type=DEVICE.type, dtype=torch.half)
    def forward(self, input: torch.Tensor, mask: torch.Tensor):
        """
        Common forward method for all models.

        Parameters
        ----------
        input : torch.Tensor
            Input data with shape (batch, seq_len, neurons)
        mask : torch.Tensor
            Mask on the neurons with shape (batch, neurons)
        """
        # Route to the appropriate forward method
        if self.v2:
            return self.forward_v2(input, mask)

        # Initialize hidden state
        self.hidden = self.init_hidden(input.shape)

        # Set hidden state of internal model
        self.inner_hidden_model.set_hidden(self.hidden)

        # Recast the mask to the input shape
        mask = mask.unsqueeze(1).expand_as(input)

        # Multiply input by the mask
        input_activity = self.identity(input * mask)

        # Transform the input into a latent
        latent_out = self.input_hidden(input_activity)

        # Transform the latent
        hidden_out = self.inner_hidden_model(latent_out)

        # Perform a linear readout to get the output
        output = self.linear(hidden_out)

        # Return output
        return output

    ### >>> DEBUG: modified forward method for new token mode >>> ###
    @torch.autocast(device_type=DEVICE.type, dtype=torch.half)
    def forward_v2(self, input: torch.Tensor, mask: torch.Tensor):
        """
        Special forward method for the newer version (v2) of the models
        based on first tokenizing the high-dimensional neural data before
        doing sequence modeling to mimic the approach used in Transformers.
        """
        # Initialize hidden state
        self.hidden = self.init_hidden(input.shape)

        # Set hidden state of internal model
        self.inner_hidden_model.set_hidden(self.hidden)

        # # Convert the multi-dimensional input sequence into a 1-D sequence of tokens
        # input_tokens = self.tokenize_neural_data(
        #     neural_sequence=input, feature_mask=mask
        # ) # (batch_size, seq_len)

        # Convert the multi-dimensional input seqeunce into a multi-dimensional sequence of tokens
        ### >>> DEBUG: multi-dimensional tokenization and embedding >>> ###
        input_tokens = self.tokenize_neural_data_multi_channel(
            neural_sequence=input, feature_mask=mask
        )  # (batch_size, seq_len, input_size)
        ### <<< DEBUG: multi-dimensional tokenization and embedding <<< ###

        # Embed the tokens and then transform to a latent
        latent_out = self.input_hidden(input_tokens)
        logger.info(f"latent_out: \t {latent_out.shape, latent_out.dtype}\n\n")  # DEBUG

        # Transform the latent
        hidden_out = self.inner_hidden_model(latent_out)
        logger.info(f"hidden_out: \t {hidden_out.shape, hidden_out.dtype}\n\n")  # DEBUG

        # Perform a linear readout to get the output
        output_logits = self.linear(hidden_out)
        logger.info(
            f"output_logits: \t {output_logits.shape, output_logits.dtype}\n\n"
        )  # DEBUG

        # Return output
        return output_logits

    ### <<< DEBUG: modified forward method for new token mode <<< ###

    def loss_fn(self):
        """
        The loss function to be used by all the models.

        This custom loss function combines a primary loss function with an additional L1 regularization
        on all model weights. This regularization term encourages the model to use fewer non-zero parameters,
        effectively making the model more sparse. This can help to prevent overfitting, make the model more
        interpretable, and improve generalization by encouraging the model to use only the most important
        features. The L1 penalty is the sum of the absolute values of the weights.
        """
        # Route to the appropriate loss_fn method
        if self.v2:
            return self.loss_fn_v2()

        def loss(output, target, mask=None, **kwargs):
            """
            Calculate loss with added FFT and L1 regularization
            on the trainable model parameters.
            Arguments:
                output: (batch_size, seq_len, input_size)
                target: (batch_size, seq_len, input_size)
                mask: (batch_size, input_size)
            """
            # Default mask to all True if not provided
            if mask is None:
                mask = torch.ones(
                    target.shape[0], target.shape[-1], dtype=torch.bool
                ).to(target.device)

            # Expand feature mask along temporal dimension
            expanded_mask = mask.unsqueeze(1).expand_as(
                output
            )  # temporally invariant & feature equivariant

            # Mask the invalid positions in `output` and `target`
            masked_output = output * expanded_mask.float()
            masked_target = target * expanded_mask.float()

            # Compute the loss without reduction
            masked_loss = self.loss(reduction="none", **kwargs)(
                masked_output, masked_target
            )

            # Normalize the loss by the total number of data points
            norm_factor = masked_loss[expanded_mask].size(dim=0)

            # Calculate next time step prediction loss before adding regularization
            original_loss = masked_loss[expanded_mask].sum() / norm_factor

            # L1 regularization term
            l1_loss = 0.0
            if self.l1_reg_param > 0.0:
                # calculate L1 regularization term for all weights
                for param in self.parameters():
                    l1_loss += torch.abs(param).mean()

            # Add the L1 penality to the original loss
            regularized_loss = original_loss + self.l1_reg_param * l1_loss

            # Return the regularized loss
            return regularized_loss

        return loss

    ### >>> DEBUG: different loss function needed for new token mode >>> ###
    def loss_fn_v2(self):
        """
        Special loss function for the newer version (v2) of the models based
        on how loss is calculated in Transformers.
        """

        def loss(output, target, mask=None, **kwargs):
            """
            Args:
                output: tensor w/ shape ``[batch_size, seq_len, n_token]``
                target: tensor w/ shape ``[batch_size, seq_len, input_size]``
                mask: tensor w/ shape ``[batch_size, input_size]``
            """
            # Default mask to all True if not provided
            if mask is None:
                mask = torch.ones(
                    target.shape[0], target.shape[-1], dtype=torch.bool
                ).to(target.device)

            # Flatten output logits along batch x time dimensions
            output = output.view(-1, self.n_token)

            # Convert target from neural vector sequence to token sequence then flatten
            target = self.tokenize_neural_data(
                neural_sequence=target, feature_mask=mask
            ).view(-1)

            # Calculate cross entropy loss
            ce_loss = torch.nn.CrossEntropyLoss(reduction="mean", **kwargs)(
                output, target
            )

            # Return loss
            return ce_loss

        return loss

    ### <<< DEBUG: different loss function needed for new token mode <<< ###

    @torch.no_grad()
    def generate(
        self,
        input: torch.Tensor,
        mask: torch.Tensor,
        num_new_timesteps: int,
        autoregressive: bool = True,
        context_window: int = BLOCK_SIZE,
    ):
        """
        Generate future neural activity from the model.

        Parameters
        ----------
        input : torch.Tensor
            Input data with shape (batch_size, seq_len, neurons)
        mask : torch.Tensor
            Mask on the neurons with shape (batch_size, neurons)
        num_new_timesteps : int
            Number of time steps to generate
        autoregressive : bool
            Whether to generate values autoregressively or not
        context_window : int
            Number of time steps to use as context

        Returns
        -------
        generated_tensor : torch.Tensor
            Generated data with shape (num_new_timesteps, neurons)
        """
        # Route to the appropriate generate method
        if self.v2:
            return self.generate_v2()

        # Set model to evaluation mode
        self.eval()

        # If generating values autoregressively
        if autoregressive:
            input = input[
                :, :context_window, :
            ]  # shape (batch_size, context_window, neurons)
        # Otherwise defaults to ground-truth feeding
        else:
            pass

        # Initialize the list of generated values
        generated_values = []

        # Loop through time
        for t in range(num_new_timesteps):
            # Get the last context_window values of the input tensor
            input_cond = input[
                :, t : context_window + t, :
            ]  # shape (batch_size, context_window, neurons)

            # Get predictions
            predictions = self(
                input_cond, mask
            )  # shape (batch_size, context_window, neurons)

            # Get last predicted value
            input_next = predictions[:, [-1], :]  # shape (batch_size, 1, neurons)

            # Append the prediction to the generated_values list and input tensor
            generated_values.append(input_next)
            input = torch.cat(
                (input, input_next), dim=1
            )  # add the prediction to the input tensor

        # Stack the generated values to a tensor
        generated_tensor = torch.cat(
            generated_values, dim=1
        )  # shape (batch_size, num_new_timesteps, neurons)

        # Only return the newly generated values
        return generated_tensor

    ### >>> DEBUG: different generate method needed for new token mode >>> ###
    @torch.no_grad()
    def generate_v2(
        self,
        input: torch.Tensor,
        mask: torch.Tensor,
        num_new_timesteps: int,
        autoregressive: bool = True,
        context_window: int = BLOCK_SIZE,
        temperature=1.0,
        top_k: Union[int, None] = 1,
    ):
        """
        Special generate method for the newer version (v2) of the models based on how generation is done in Transformers.
        Take a conditioning sequence of neural data input with shape (batch_size, seq_len, input_size) and
        completes the sequence num_new_timesteps times, feeding the predictions back into the model each time.
        In the v2 version, models take neural data as input, internally tokenize them and outputs the next token as output.
        Therefore, we must convert the output token back to neural data. We do this by finding sampling from the distribution of
        neural vectors that correspond to the token. We then append the sampled neural vector to the running sequence and continue.
        """
        # Set model to evaluation mode
        self.eval()

        # If generating values autoregressively
        if autoregressive:
            input = input[
                :, :context_window, :
            ]  # shape (batch_size, context_window, neurons)
        # Otherwise defaults to ground-truth feeding
        else:
            pass

        # Initialize the list of generated values
        generated_values = []

        # Loop through time
        for t in range(num_new_timesteps):
            # If the sequence context is growing too long we must crop it
            input_cond = input[:, t : context_window + t, :]

            # Forward the model to get the output
            output = self(input_cond, mask)

            # Pluck the logits at the final step and scale by desired temperature
            logits = output[:, -1, :] / temperature

            # Optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            # Apply softmax to convert logits to (normalized) probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)

            # Sample from the distribution to get the next token
            token_next = torch.multinomial(probs, num_samples=1)

            # Convert the token back to a neural vector
            sample_mu = self.token_neural_map[token_next.item()].expand(
                input.size(0), 1, self.input_size
            )  # the sample mean

            # Sample a neural vector from the multivariate normal distribution
            input_next = torch.normal(mean=sample_mu, std=0.01).to(
                input.device
            )  # use variance << 1.0

            # Append the prediction to the generated_values list
            generated_values.append(input_next)

            # Append sampled data to the running sequence and continue
            input = torch.cat((input, input_next), dim=1)

        # Stack the generated values to a tensor
        generated_tensor = torch.cat(
            generated_values, dim=1
        )  # shape (batch_size, num_new_timesteps, neurons)

        # Only return the newly generated values
        return generated_tensor

    ### <<< DEBUG: different generate method needed for new token mode <<< ###

    def sample(self, num_new_timesteps: int):
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
        hidden_size: Union[int, None] = None,  # unused in this model
        loss: Union[Callable, None] = None,
        l1_reg_param=0.0,
    ):
        super(NaivePredictor, self).__init__(
            input_size,
            None,  # hidden_size
            loss,
            l1_reg_param,
        )

        # Input to hidden transformation
        self.input_hidden = torch.nn.Identity()

        # Hidden to hidden transformation
        self.hidden_hidden = torch.nn.Identity()
        # Instantiate internal hidden model
        self.inner_hidden_model = InnerHiddenModel(self.hidden_hidden, self.hidden)

        # Override the linear readout
        self.linear = torch.nn.Identity()

        # Create a dud parameter to avoid errors with the optimizer
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
        hidden_size: Union[int, None] = None,  # unused in this model
        loss: Union[Callable, None] = None,
        l1_reg_param=0.0,
    ):
        super(LinearRegression, self).__init__(
            input_size,
            None,  # hidden_size
            loss,
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
        l1_reg_param: float = 0.0,
    ):
        super(FeatureFFNN, self).__init__(
            input_size,
            hidden_size,
            loss,
            l1_reg_param,
        )
        # Special parameters for this model
        self.dropout = 0.1  # dropout rate

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
        l1_reg_param: float = 0.0,
    ):
        """
        Neural activity data is continuous valued and thus
        can naturally be treated as if it were already emebedded.
        However, to maintain notational similarity with the original
        Transformer architecture, we use a linear layer to perform
        expansion recoding. This replaces the embedding layer in the
        traditional Transformer but all it is really just a linear projection.
        """
        # NOTE: Transformer only works with even `d_model`
        if hidden_size % 2 != 0:
            logger.info(f"Changing hidden_size from {hidden_size} to {hidden_size+1}.")
            hidden_size = hidden_size + 1
        else:
            logger.info(f"Using hidden_size: {hidden_size}.")
            hidden_size = hidden_size

        # Initialize super class
        super(NeuralTransformer, self).__init__(
            input_size,
            hidden_size,
            loss,
            l1_reg_param,
        )

        # Special transformer parameters
        self.n_head = find_largest_divisor(
            hidden_size
        )  # number of attention heads (NOTE: must be divisor of `hidden_size`)
        logger.info(f"Number of attention heads: {self.n_head}.")
        self.dropout = 0.1  # dropout rate,

        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            self.hidden_size,  # if positional_encoding after embedding
            dropout=self.dropout,
        )

        # Input to hidden transformation
        self.input_hidden = torch.nn.Sequential(
            self.embedding,
            self.positional_encoding,
            # torch.nn.ReLU(),  # Should we exclude the ReLU here for the transformer model?
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
        l1_reg_param: float = 0.0,
    ):
        """
        The output size will be the same as the input size.
        """
        super(NetworkCTRNN, self).__init__(
            input_size,
            hidden_size,
            loss,
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
        l1_reg_param: float = 0.0,
    ):
        """
        The output size will be the same as the input size.
        """

        super(LiquidCfC, self).__init__(
            input_size,
            hidden_size,
            loss,
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
        l1_reg_param: float = 0.0,
    ):
        """
        The output size will be the same as the input size.
        """
        super(NetworkLSTM, self).__init__(
            input_size,
            hidden_size,
            loss,
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
