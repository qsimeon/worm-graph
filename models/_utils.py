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
    TODO: Improve and validate this implementation of the MASE loss.
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
        assert y_pred.shape == target.shape, "y_pred and target must have the same shape"
        # Calculate the Mean Absolute Error of the one-step naive forecast
        if target.ndim == 2:  # if 1-timestep
            mean_naive_error = torch.tensor(1.0)
        else:  # if sequence
            naive_forecast_errors = torch.abs(target[:, 1:, :] - target[:, :-1, :])
            mean_naive_error = torch.mean(
                naive_forecast_errors, dim=1, keepdim=True
            )  # average over seq_len

        # Calculate the Mean Absolute Error of the predictions
        prediction_errors = torch.abs(y_pred - target).expand_as(target)
        mase_errors = prediction_errors / mean_naive_error.clamp(
            min=1e-6
        )  # avoid division by 0 by ensuring error are at least 1e-8

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


def load_model_checkpoint(checkpoint_path):
    """
    Load a model from a checkpoint file. The checkpoint should
    contain all the neccesary information to reinstantiate the model
    using its constructor.

    Args:
        checkpoint_path (str): The path to the checkpoint file.

    Returns:
        model: The loaded model.
    """
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(ROOT_DIR, checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    # State dictionaries
    model_state_dict = checkpoint["model_state_dict"]
    # Names
    model_name = checkpoint["model_name"]
    # Model parameters
    input_size = checkpoint["input_size"]
    hidden_size = checkpoint["hidden_size"]
    loss_name = checkpoint["loss_name"]
    l1_reg_param = checkpoint["l1_reg_param"]
    # New attributes for version 2
    version_2 = checkpoint.get("version_2", VERSION_2)
    num_tokens = checkpoint.get("num_tokens", NUM_TOKENS)
    # Reinstantiate the model
    model = eval(model_name)(
        input_size,
        hidden_size,
        loss=loss_name,
        l1_reg_param=l1_reg_param,
        version_2=version_2,
        num_tokens=num_tokens,
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
    A module for obtaining a readout function for each channel in a multi-dimensional input.
    """

    def __init__(self, num_channels, in_features, out_features):
        super().__init__()
        self.num_channels = num_channels
        self.in_features = in_features
        self.out_features = out_features
        self.readouts = torch.nn.ModuleList(
            [torch.nn.Linear(self.in_features, self.out_features) for _ in range(self.num_channels)]
        )

    def forward(self, x):
        """
        Args:
            x: Tensor, shape (batch_size, seq_len, in_features)
        Output:
            output: Tensor, shape (batch_size, seq_len, num_channels, out_features)
        """
        batch_size, seq_len, in_features = x.shape
        assert (
            in_features == self.in_features
        ), "Input dimension does not match expected `in_features`."
        output = torch.empty(batch_size, seq_len, self.num_channels, self.out_features).to(x.device)
        for _ in range(self.num_channels):
            output[:, :, _, :] = self.readouts[_](x)
        return output  # (batch_size, seq_len, num_channels, out_features)


class MultiChannelEmbedding(torch.nn.Module):
    """
    A module for obtaining an embedding table for each channel of a multi-dimensional token.
    """

    def __init__(self, num_channels, num_embeddings, embedding_dim):
        super().__init__()
        self.num_channels = num_channels
        self.num_embeddings = num_embeddings + 1  # 0-th index token used for unmasked values
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
        # Get the input shape
        batch_size, seq_len, num_channels = x.shape
        # Reshape x to a 2D tensor for embedding lookup
        x_reshaped = x.view(-1, num_channels)  # (batch_size * seq_len, num_channels)
        # Perform embedding lookup for each channel and concatenate
        embedded = torch.cat(
            [self.embeddings[i](x_reshaped[:, i]) for i in range(num_channels)], dim=-1
        )
        # Reshape back to the original batch_size and seq_len, sum across channels
        output = embedded.view(batch_size, seq_len, num_channels, self.embedding_dim).sum(
            dim=2
        )  # (batch_size, seq_len, embedding_dim)
        # Return the embedding output
        return output


class ToFloat32(torch.nn.Module):
    """
    A custom layer for type conversion.
    """

    def forward(self, x):
        return x.to(torch.float32)


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
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)  # batch_first=True
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape (batch_size, seq_len, embedding_dim)
        """
        x = x * math.sqrt(self.d_model)  # normalization used in the original transformer paper
        x = x + self.pe[:, : x.size(1), :]  # add positional encoding to input
        return self.dropout(x)


class CausalTransformer(torch.nn.Module):
    """
    We use a single Transformer Encoder layer as the hidden-hidden model.
    Sets `is_causal=True` in forward method of TransformerEncoderLayer.
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.num_layers = 1  # single layer
        self.is_causal = True  # causal attention
        self.encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(self.encoder_layer, self.num_layers)

    def forward(self, src):
        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(
            src.size(1),
            device=src.device,
        )
        out = self.transformer_encoder(src, mask=causal_mask, is_causal=self.is_causal)
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


class SelfAttention(torch.nn.Module):
    """A single self-attention layer.

    Parameters:
        embed_dim: embedding dimension
        num_heads: number of attention heads
        dropout: probability of dropping a neuron

    Inputs:
        input: tensor of shape (batch, seq_len, embed_dim)

    """

    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout,
            batch_first=True,
        )

    def forward(self, src):
        """
        NOTE: Because we use batch_first=True, src must have shape (batch, seq_len, embed_dim).
        """
        # Create a causal attention mask
        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(
            src.size(1),
            device=src.device,
        )
        # Apply self-attention
        attn_output, _ = self.attn(
            key=src,
            query=src,
            value=src,
            attn_mask=causal_mask,
            is_causal=True,
            need_weights=False,
            average_attn_weights=True,
        )
        # Return attention output w/ shape (batch, seq_len, embed_dim)
        return attn_output


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
        self.register_parameter(name="alpha", param=torch.nn.Parameter(torch.ones(1, hidden_size)))
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
        Propagate input through the network. NOTE: Because we use
        batch_first=True, input has shape (batch, seq_len, input_size).
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

# # # Model super class: Common interface for all model architectures # # # #
# Provides the input-output backbone and allows changeable inner modules a.k.a "cores". #
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
        `get_input_size`, `get_hidden_size`, `get_loss_name`, and `get_l1_reg_param`.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: Union[int, None],
        loss: Union[Callable, None] = None,
        l1_reg_param: float = 0.0,
        # New attributes for version 2
        version_2: bool = VERSION_2,
        num_tokens: int = NUM_TOKENS,
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
        # NOTE: The output_size is same as the input_size because the model is a self-supervised autoencoder.
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
        # Instantiate internal hidden model (i.e. the "core") - placeholder
        self.inner_hidden_model = InnerHiddenModel(
            hidden_hidden_model=self.hidden_hidden,
            hidden_state=self.hidden,
        )
        # Embedding layer - placeholder
        self.latent_embedding = torch.nn.Linear(
            self.input_size,
            self.hidden_size,
        )
        # Linear readout
        self.linear = torch.nn.Linear(self.hidden_size, self.output_size)
        # Optional layer normalization
        self.layer_norm = torch.nn.LayerNorm(self.hidden_size, elementwise_affine=True)
        # Model version_2 tokenizes neural data either as a 1-D sequence
        self.version_2 = version_2
        self.num_tokens = num_tokens
        # New attributes and parameters for version 2 with tokenization
        if self.version_2:
            # Number of tokens to approximate continuous values
            self.num_tokens = num_tokens
            # Modify output size to be number of tokens
            self.output_size = self.num_tokens
            # Initialize the neural embedding map from tokens to neural vectors.
            # NOTE: This is equivalent to the codebook in VQ-VAEs.
            neural_embedding = torch.zeros(  # not learned but updated using EMA
                self.num_tokens, self.input_size
            )  # maps tokens to vectors
            self.register_buffer("neural_embedding", neural_embedding)
            # Create bin edges for tokenizing continuous-valued z-scored data
            # NOTE: num_tokens bin_edges means there are num_tokens-1 bins for masked values;
            # the 0-indexed bin will be used for unmasked values.
            bin_edges = torch.tensor(norm.ppf(torch.linspace(0, 1, self.num_tokens)))
            self.register_buffer("bin_edges", bin_edges)
            # Modify embedding layer to be a lookup table
            self.latent_embedding = torch.nn.Embedding(
                num_embeddings=self.num_tokens, embedding_dim=self.hidden_size
            )  # embedding lookup table (learned)
            # Adjust linear readout to output token logits
            self.linear = torch.nn.Linear(self.hidden_size, self.num_tokens)
            # Initialize weights
            self._init_weights()
            # Alias methods to new versions
            self.forward = self.forward_v2
            self.loss_fn = self.loss_fn_v2
            self.generate = self.generate_v2

    # Initialization functions for setting hidden states and weights.
    def _init_hidden(self):
        self.hidden = None
        return None

    def _init_weights(self):
        # Initialize the readout bias
        torch.nn.init.zeros_(self.linear.bias)
        # Initialize the readout weights
        torch.nn.init.xavier_uniform_(self.linear.weight)
        # Initialize the embedding weights
        torch.nn.init.normal_(self.latent_embedding.weight)
        return None

    def init_hidden(self, input_shape=None):
        """
        Enforce the all models have a `init_hidden` method which initilaizes the hidden state of the "core".
        """
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

    @torch.autocast(device_type=DEVICE.type, dtype=torch.half)
    def sequence_normalization(self, neural_sequence):
        """
        TODO: Get this working.
        Normalize the neural seqeunce causally.
        """
        device = neural_sequence.device
        batch_size, _, _ = neural_sequence.shape
        neural_sequence_normalized = torch.empty_like(neural_sequence, device=device)
        transform = CausalNormalizer()
        # Normalize using CausalNormalizer
        for b in range(batch_size):
            X = neural_sequence[b].cpu().numpy()
            X = transform.fit_transform(X)
            neural_sequence_normalized[b] = torch.from_numpy(X).to(neural_sequence.dtype).to(device)
        # Return the normalized sequence
        return neural_sequence_normalized

    @torch.autocast(device_type=DEVICE.type, dtype=torch.half)
    def calculate_distances(self, neural_sequence, token_matrix, feature_mask=None):
        """
        Efficiently calculates Euclidean distances between neural sequence vectors and token matrix vectors.

        Args:
            neural_sequence (torch.Tensor): Shape (batch_size, seq_len, input_size).
            token_matrix (torch.Tensor): Shape (num_tokens, input_size).
            feature_mask (torch.Tensor, optional): Shape (batch_size, input_size). If None, all features are considered.

        Returns:
            torch.Tensor: Distances for each batch. Shape (batch_size, seq_len, num_tokens).
        """
        # Get input shapes
        batch_size, _, input_size = neural_sequence.shape
        assert (
            input_size == token_matrix.shape[-1]
        ), "Expected `token_matrix` to have same input size as `neural_sequence`."
        # Set feature_mask to all True if it is None
        if feature_mask is None:
            feature_mask = torch.ones(
                (batch_size, input_size),
                dtype=torch.bool,
                device=neural_sequence.device,
            )
        # Applying the feature mask to the neural sequence
        masked_neural_sequence = neural_sequence * feature_mask.unsqueeze(
            1
        )  # (batch_size, seq_len, input_size) * (batch_size, 1, input_size) -> (batch_size, seq_len, input_size)
        # Applying the feature mask to the token matrix
        masked_token_matrix = token_matrix.unsqueeze(0) * feature_mask.unsqueeze(
            1
        )  # (1, num_tokens, input_size) * (batch_size, 1, input_size) -> (batch_size, num_tokens, input_size)

        ### >>> FAST, NEW Implementation >>> ###
        # Fast and memory efficient distance calculation
        TERM_1 = masked_neural_sequence.pow(2).sum(dim=-1, keepdim=True)  # (batch_size, seq_len, 1)
        TERM_2 = -2 * torch.bmm(
            masked_neural_sequence,
            masked_token_matrix.permute(0, 2, 1),
        )  # (batch_size, seq_len, num_tokens)
        TERM_3 = (
            masked_token_matrix.pow(2).sum(dim=-1, keepdim=True).permute(0, 2, 1)
        )  # (batch_size, 1, num_tokens)
        distances = TERM_1 + TERM_2 + TERM_3  # (batch_size, seq_len, num_tokens)
        ### <<< FAST, NEW Implementation <<< ###

        # ### >>> SLOW, ORIGINAL Implementation >>> ###
        # # Expand dimensions for broadcasting
        # masked_neural_sequence_expanded = masked_neural_sequence.unsqueeze(
        #     2
        # )  # (batch_size, seq_len, 1, input_size)
        # masked_token_matrix_expanded = masked_token_matrix.unsqueeze(
        #     1
        # )  # (batch_size, 1, num_tokens, input_size)
        # distances = torch.mean(
        #     (masked_neural_sequence_expanded - masked_token_matrix_expanded)
        #     ** 2,  # (batch_size, seq_len, 1, input_size) - (batch_size, 1, num_tokens, input_size) -> (batch_size, seq_len, num_tokens, input_size)
        #     dim=-1,
        # )  # (batch_size, seq_len, num_tokens)
        # ### <<< SLOW, ORIGINAL Implementation <<< ###

        # Return distance matrix
        return distances

    @torch.autocast(device_type=DEVICE.type, dtype=torch.half)
    def tokenize_neural_data(
        self,
        neural_sequence: torch.Tensor,
        feature_mask: Union[None, torch.Tensor] = None,
        token_matrix: Union[None, torch.Tensor] = None,
        decay: float = 0.5,
    ):
        """
        Convert the high-dimensional sequence of neural states to a 1-D sequence of tokens.
        The approach used is similar to that of VQ-VAEs where the neural data is treated as the
        encoder output, the decoder input is the nearest-neighbor codebook vector, and the tokens
        are the indices of those vectors in the codebok. The decoder is treated as rest of the
        model after the tokenization step. The dimensionality of the embedding space is the same
        as the dimensionality of the neural data (i.e. `input_size` or `num_channels`).

        Args:
            neural_sequence: tensor of shape (batch_size, seq_len, input_size)
            feature_mask: tensor of shape (batch_size, input_size)
            token_matrix: tensor of shape (num_tokens, input_size)
            decay: float EMA decay factor for updating the neural embedding
        Output:
            token_sequence: tensor of shape (batch_size, seq_len)
        """
        # Ensure the neural_sequence has the correct shapes
        assert (
            neural_sequence.ndim == 3
        ), "`neural_sequence` must have shape (batch_size, seq_len, input_size)"
        batch_size, _, input_size = neural_sequence.shape
        # Set feature_mask to all True if it is None
        if feature_mask is None:
            feature_mask = torch.ones(
                (batch_size, input_size),
                dtype=torch.bool,
                device=neural_sequence.device,
            )
        assert (
            feature_mask.ndim == 2 and feature_mask.shape[-1] == input_size
        ), "`feature_mask` must have shape (batch_size, input_size)"
        assert feature_mask.sum().item() > 0, "`feature_mask` cannot be all False."
        if token_matrix is None:
            token_matrix = self.neural_embedding
        assert (
            token_matrix.ndim == 2 and token_matrix.shape[-1] == input_size
        ), "`token_matrix` must have shape (num_tokens, input_size)"
        # Move token_matrix to same device as neural_sequence
        token_matrix = token_matrix.to(neural_sequence.device)
        # Get shapes from the token_matrix
        # NOTE: We could use the attributes self.num_tokens and self.input_size of the model;
        # but getting the sizes this way allos us to use this method as a standalone function.
        num_tokens, _ = token_matrix.shape

        # PART 1: Tokenize the neural data
        # Calculate distances between neural data and embedding vectors
        distances = self.calculate_distances(
            neural_sequence,
            token_matrix,
            feature_mask,
        )  # (batch_size, seq_len, num_tokens)
        # Find the minimum indices along the tokens dimension
        token_sequence = distances.argmin(dim=-1)  # (batch_size, seq_len)
        # Skip updating the neural_embedding if we are in eval mode
        if not torch.is_grad_enabled():  # not updating neural_embedding
            return token_sequence
        else:
            pass  # otherwise model is training, so update neural_embedding

        # PART 2: Update `self.neural_embedding`

        # # ### >>> FAST but INCORRECT, ORIGINAL Implementation >>> ###
        # # NOTE: Correctly only updates positions in `self.neural_embedding` that correspond to tokens observed in the current batch;
        # # but fails to only update positions in `self.neural_embedding` that correspond to masked inputs/features in the current batch.
        # # Thus, this version will only work fully as expected if all inputs are masked (i.e. `feature_mask`` is all True).
        # # Flatten neural sequence and token sequence
        # token_sequence_flat = token_sequence.view(-1)  # (batch_size * seq_len, )
        # neural_flat = neural_sequence.view(
        #     -1, self.input_size
        # )  # (batch_size * seq_len, input_size)
        # # Initialize sums and counts tensors
        # token_sums = torch.zeros_like(
        #     self.neural_embedding,
        #     device=neural_flat.device,
        #     dtype=neural_flat.dtype,
        # )
        # token_counts = torch.zeros(
        #     self.num_tokens,
        #     device=neural_flat.device,
        #     dtype=neural_flat.dtype,
        # )
        # # Accumulate sums for each token
        # token_sums.index_add_(
        #     dim=0, index=token_sequence_flat, source=neural_flat
        # ) 
        # # Count occurrences of each token
        # token_counts.index_add_(
        #     dim=0,
        #     index=token_sequence_flat,
        #     source=torch.ones_like(
        #         token_sequence_flat,
        #         dtype=token_counts.dtype,
        #     ),
        # ) 
        # # Compute means and apply EMA update
        # new_token_means = token_sums / token_counts.unsqueeze(1).clamp(min=1)  # avoid division by 0
        # new_token_means = new_token_means.to(self.neural_embedding.device)  # move to same device
        # observed_tokens = token_sequence_flat.unique()
        # decay = 0.5  # EMA decay factor
        # OLD = self.neural_embedding[observed_tokens]
        # NEW = new_token_means[observed_tokens]
        # self.neural_embedding[observed_tokens] = decay * OLD + (1 - decay) * NEW
        # # ### <<< FAST but INCORRECT, ORIGINAL Implementation <<< ###

        ### >>> SLOW but CORRECT, NEW Implementation >>> ###
        # NOTE: Updates positions in `self.neural_embedding` that correspond to observed tokens and masked inputs.
        # Get positions of masked/observed input features
        masked_input_positions = feature_mask.nonzero(
            as_tuple=False
        )  # (<= batch_size * input_size, 2)
        # Get unique values and their counts in batch dimension (first column)
        _, counts = torch.unique(masked_input_positions[:, 0], return_counts=True)
        # Split the tensor into groups based on the batch dimension
        batch_groups = torch.split(masked_input_positions, counts.tolist())
        # For each batch index update the neural embedding only using observed tokens and maksed features
        decay = 0.5  # decay factor for EMA
        for group in batch_groups:  # bigO(batch_size)
            batch_idx = group[:, 0].unique().item()
            observed_inputs = group[:, 1]
            batch_tokens = token_sequence[batch_idx]  # (seq_len, ) 
            batch_inputs = neural_sequence[batch_idx]  # (seq_len, input_size) 
            for token in batch_tokens.unique():
                OLD = self.neural_embedding[token, observed_inputs]
                NEW = batch_inputs[batch_tokens == token].mean(dim=0)[observed_inputs]
                self.neural_embedding[token, observed_inputs] = decay * OLD + (1 - decay) * NEW
        ### <<< SLOW but CORRECT, NEW Implementation <<< ###

        # Return the tokenized sequence
        return token_sequence
        
    @torch.autocast(device_type=DEVICE.type, dtype=torch.long)
    def bin_tensor(self, nt):
        """
        Converts a neural tensor of continuous values from a standard normal
        distribution into a tensor of discrete values by indexing them into
        bins defined by self.bin_edges.

        Args:
            nt: neural tensor (batch_size, seq_len, input_size)
        Output:
            it: index tensor (batch_size, seq_len, input_size)
        """
        b1 = nt.unsqueeze(-1) > self.bin_edges[:-1]
        b2 = nt.unsqueeze(-1) <= self.bin_edges[1:]
        bool_arr = (b1 * b2).to(torch.long)
        it = bool_arr.argmax(dim=-1) + 1
        return it

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
        if self.version_2:
            return self.forward_v2(input, mask)
        # Initialize hidden state
        self.hidden = self.init_hidden(input.shape)
        # Set hidden state of internal model
        self.inner_hidden_model.set_hidden(self.hidden)
        # Multiply input by the mask (expanded to match input shape)
        input_activity = self.identity(
            input * mask.unsqueeze(1).expand_as(input)
        )  # (batch_size, seq_len, input_size)
        # ### DEBUG ###
        # # Normalize the input sequence 
        # input_activity = self.sequence_normalization(input_activity)
        # ### DEBUG ###
        # Transform the input into a latent
        latent_out = self.input_hidden(input_activity)  # (batch_size, seq_len, hidden_size)
        # Transform the latent
        hidden_out = self.inner_hidden_model(latent_out)  # (batch_size, seq_len, hidden_size)
        # Perform a linear readout to get the output
        output = self.linear(hidden_out)  # (batch_size, seq_len, input_size)
        # Return output neural data
        return output

    @torch.autocast(device_type=DEVICE.type, dtype=torch.half)
    def forward_v2(self, input: torch.Tensor, mask: torch.Tensor):
        """
        Special forward method for the newer version (version_2) of the models
        based on first tokenizing the high-dimensional neural data before
        doing sequence modeling to mimic the approach used in Transformers.
        """
        # Initialize hidden state
        self.hidden = self.init_hidden(input.shape)
        # Set hidden state of internal model
        self.inner_hidden_model.set_hidden(self.hidden)
        # Multiply input by the mask (expanded to match input shape)
        input_activity = self.identity(
            input * mask.unsqueeze(1).expand_as(input)
        )  # (batch_size, seq_len, input_size)
        # ### DEBUG ###
        # # Normalize the input sequence 
        # input_activity = self.sequence_normalization(input_activity)
        # ### DEBUG ###
        # Convert the high-D neural sequence into a 1-D token sequence
        input_tokens = self.tokenize_neural_data(
            neural_sequence=input_activity,
            feature_mask=mask,
        )  # (batch_size, seq_len)
        # Embed the tokens and then transform to a latent
        latent_out = self.input_hidden(input_tokens)  # (batch_size, seq_len, hidden_size)
        # Transform the latent
        hidden_out = self.inner_hidden_model(latent_out)  # (batch_size, seq_len, hidden_size)
        # Perform a linear readout to get the output
        output_logits = self.linear(hidden_out)  # (batch_size, seq_len, num_tokens)
        # Return output token logits
        return output_logits

    @torch.autocast(device_type=DEVICE.type, dtype=torch.half)
    def loss_fn(self):
        """
        The loss function to be used by all the models (default versions).
        This custom loss function combines a primary loss function with an additional
        L1 regularization on all model weights. This regularization term encourages the
        model to use fewer non-zero parameters, effectively making the model more sparse.
        This can help to prevent overfitting, make the model more interpretable, and improve
        generalization by encouraging the model to use only the most important features. The
        L1 penalty is the sum of the absolute values of the weights.
        """
        # Route to the appropriate loss_fn method
        if self.version_2:
            return self.loss_fn_v2()

        def loss(output, target, mask=None, **kwargs):
            """
            Calculate loss with added L1 regularization
            on the trainable model parameters.

            Arguments:
                output: (batch_size, seq_len, input_size)
                target: (batch_size, seq_len, input_size)
                mask: (batch_size, input_size)
            """
            # Default mask to all True if not provided
            if mask is None:
                mask = torch.ones(target.shape[0], target.shape[-1], dtype=torch.bool).to(
                    target.device
                )
            # Expand feature mask along temporal dimension
            expanded_mask = mask.unsqueeze(1).expand_as(
                output
            )  # temporally invariant & feature equivariant
            # Mask the invalid positions in `output` and `target`
            masked_output = output * expanded_mask.float()
            masked_target = target * expanded_mask.float()
            # Compute the reconstruction loss without reduction
            masked_recon_loss = self.loss(reduction="none", **kwargs)(masked_output, masked_target)
            # Normalize the loss by the total number of data points
            norm_factor = masked_recon_loss[expanded_mask].size(dim=0)
            # Calculate next time step prediction loss w/out regularization
            recon_loss = masked_recon_loss[expanded_mask].sum() / norm_factor
            # L1 regularization term
            l1_loss = 0.0
            if self.l1_reg_param > 0.0:
                # Calculate L1 regularization term for all weights
                for param in self.parameters():
                    l1_loss += torch.abs(param).mean()
            # Add the L1 penality to the original loss
            reg_loss = self.l1_reg_param * l1_loss
            total_loss = recon_loss + reg_loss
            # Return loss
            return total_loss

        return loss

    ### >>> DEBUG: Different loss function needed for new token mode >>> ###
    @torch.autocast(device_type=DEVICE.type, dtype=torch.half)
    def loss_fn_v2(self):
        """
        Special loss function for the newer version (version_2) of the models based
        on how loss is calculated in Transformers which operate on tokenized data.
        """

        def loss(output, target, mask=None, **kwargs):
            """
            Args:
                output: tensor w/ shape ``[batch_size, seq_len, num_tokens]``
                target: tensor w/ shape ``[batch_size, seq_len, input_size]``
                mask: tensor w/ shape ``[batch_size, input_size]``
            """
            # Default mask to all True if not provided
            if mask is None:
                mask = torch.ones(target.shape[0], target.shape[-1], dtype=torch.bool).to(
                    target.device
                )
            # Flatten output logits along batch x time dimensions
            output = output.view(
                -1, self.num_tokens
            )  # (batch_size, seq_len, num_tokens) -> (batch_size * seq_len, num_tokens)
            # Convert target from neural vector sequence to token sequence.
            # NOTE: torch.no_grad() prevents `self.neural_embedding` from being updated based on targets.
            with torch.no_grad():
                target = self.tokenize_neural_data(
                    neural_sequence=target,
                    feature_mask=mask,
                )
                target = target.view(-1)  # (batch_size, seq_len) -> (batch_size * seq_len)
            # Calculate cross entropy loss from predicted token logits and target tokens.
            ce_loss = torch.nn.CrossEntropyLoss(reduction="mean", **kwargs)(output, target)
            # Calculate the total loss
            total_loss = ce_loss
            # Return loss
            return total_loss

        return loss

    ### <<< DEBUG: Different loss function needed for new token mode <<< ###

    @torch.no_grad()
    def generate(
        self,
        input: torch.Tensor,
        mask: torch.Tensor,
        num_new_timesteps: int,
        context_window: int = BLOCK_SIZE,
    ):
        """
        Generate future neural activity from the model. Take a conditioning sequence of
        neural data input with shape (batch_size, seq_len, input_size) and completes the
        sequence num_new_timesteps times. Generations are made autoregressively where the
        predictions are fed back into the model after each generation step.

        Parameters
        ----------
        input : torch.Tensor
            Input data with shape (batch_size, seq_len, neurons)
        mask : torch.Tensor
            Mask on the neurons with shape (batch_size, neurons)
        num_new_timesteps : int
            Number of time steps to generate
        context_window : int
            Number of time steps to use as context

        Returns
        -------
        generated_tensor : torch.Tensor
            Generated data with shape (num_new_timesteps, neurons)

        TODO: Need to normalize adaptively to avoid generations that blow up.
        """
        # Route to the appropriate generate method
        if self.version_2:
            return self.generate_v2(input, mask, num_new_timesteps, context_window)
        # Set model to evaluation mode
        self.eval()
        # Detach and copy the input
        input_copy = input.detach().clone()
        # Loop through time
        for _ in range(num_new_timesteps):
            # If the sequence context is growing too long we must crop it
            input_cond = input_copy[
                :, -context_window:, :
            ].detach()  # (batch_size, context_window, neurons)
            # Forward the model to get the predictions
            predictions = self(input_cond, mask)  # (batch_size, context_window, neurons)
            # Get the last predicted value
            input_next = predictions[:, [-1], :]  # (batch_size, 1, neurons)
            # TODO: Make adaptive normalization a function of the model itself;
            # The current clamping is just a quick fix to prevent exploding values.
            input_next = torch.clamp(input_next, min=input_cond.min(), max=input_cond.max())
            # Append the prediction to the the running sequence and continue
            input_copy = torch.cat(
                (input_copy, input_next), dim=1
            )  # generating values autoregressively
        # Get only the newly generated time steps
        generated_values = input_copy[
            :, -num_new_timesteps:, :
        ].detach()  # (batch_size, num_new_timesteps, input_size)
        # Return the generations
        return generated_values

    ### >>> DEBUG: Different generate method needed for new token mode >>> ###
    @torch.no_grad()
    def generate_v2(
        self,
        input: torch.Tensor,
        mask: torch.Tensor,
        num_new_timesteps: int,
        context_window: int = BLOCK_SIZE,
        temperature=1.0,
        top_k: Union[int, None] = None,
    ):
        """
        Special generate method for the newer version (version_2) of the models based on how
        generation is done in Transformers. In the newer version (version_2), models take neural
        data as input and outputs token logits. Therefore, we must convert the token logits back to
        neural data to be fed back into the model. We sample from the distribution over the predicted
        next token, retrieve the mean neural data value(s) corresponding to that token, append
        the neural data value(s) to the running neural data sequence, then repeat this process.
        """
        # Set model to evaluation mode
        self.eval()
        # Detach and copy the input
        input_copy = input.detach().clone()
        # Get input shapes
        batch_size, _, input_size = input.shape
        # Loop through time
        for _ in range(num_new_timesteps):
            # If the sequence context is growing too long we must crop it
            input_cond = input_copy[
                :, -context_window:, :
            ]  # (batch_size, context_window, input_size)
            # Forward the model to get the output
            output = self(input_cond, mask)  # (batch_size, context_window, num_tokens)
            # Pluck the logits at the final step and scale by desired temperature
            logits = output[:, -1, :] / temperature  # (batch_size, num_tokens)
            # Optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # Apply softmax to convert logits to (normalized) probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)  # (batch_size, num_tokens)
            # Sample from the distribution to get the next token
            token_next = torch.multinomial(probs, num_samples=1).view(
                batch_size, 1
            )  # (batch_size, 1)
            # Convert tokens to neural data using neural_embedding
            input_next = self.neural_embedding[token_next].view(  # (batch_size, 1)
                batch_size, 1, input_size
            )  # (batch_size, 1, input_size)
            # Append sampled data to the running sequence and continue
            input_copy = torch.cat(
                (input_copy, input_next), dim=1
            )  # generating values autoregressively
        # Get only the newly generated time steps
        generated_values = input_copy[
            :, -num_new_timesteps:, :
        ]  # (batch_size, num_new_timesteps, input_size)
        # Return the generations
        return generated_values

    ### <<< DEBUG: Different generate method needed for new token mode <<< ###

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
    NOTE:
    (1) This model will throw an error if you try to train it because
    it has no trainable parameters and thus has no gradient function.
    (2) This model does is not defined to work with version_2.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: Union[int, None] = None,  # unused in this model
        loss: Union[Callable, None] = None,
        l1_reg_param=0.0,
        **kwargs,
    ):
        # NaivePredictor does not work with version_2
        if kwargs.get("version_2", True):
            logger.info(
                f"NaivePredictor does not work with version_2 "
                "because it does not output token logits. "
                "Switching to version_1."
            )
        kwargs["version_2"] = False
        # Initialize super class
        super(NaivePredictor, self).__init__(
            input_size,
            None,  # hidden_size
            loss,
            l1_reg_param,
            **kwargs,
        )
        # Input to hidden transformation
        self.input_hidden = torch.nn.Identity()
        # Hidden to hidden transformation
        self.hidden_hidden = torch.nn.Identity()
        # Instantiate internal hidden model (i.e. the "core")
        self.inner_hidden_model = InnerHiddenModel(
            hidden_hidden_model=self.hidden_hidden,
            hidden_state=self.hidden,
        )
        # Override the linear readout
        self.linear = torch.nn.Identity()
        # Create a dud parameter to avoid errors with the optimizer
        self._ = torch.nn.Parameter(torch.tensor(0.0))

    def init_hidden(self, input_shape=None):
        return None


class LinearRegression(Model):
    """
    A simple linear regression model.
    This model can only learn a fixed linear feature regression
    function that it applies at every time step independently.
    Memory-less but can learn linear feature regression.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: Union[int, None] = None,  # unused in this model
        loss: Union[Callable, None] = None,
        l1_reg_param=0.0,
        **kwargs,
    ):
        # Initialize super class
        super(LinearRegression, self).__init__(
            input_size,
            None,  # hidden_size
            loss,
            l1_reg_param,
            **kwargs,
        )
        # Input to hidden transformation
        self.input_hidden = torch.nn.Sequential(
            self.latent_embedding,
            # NOTE: No ReLU because that would be nonlinear model.
        )
        # Hidden to hidden transformation
        self.hidden_hidden = torch.nn.Identity()
        # Instantiate internal hidden model (i.e. the "core")
        self.inner_hidden_model = InnerHiddenModel(
            hidden_hidden_model=self.hidden_hidden,
            hidden_state=self.hidden,
        )

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
        **kwargs,
    ):
        # Initialize super class
        super(FeatureFFNN, self).__init__(
            input_size,
            hidden_size,
            loss,
            l1_reg_param,
            **kwargs,
        )
        # Special parameters for this model
        self.dropout = 0.1  # dropout rate
        # Input to hidden transformation
        self.input_hidden = torch.nn.Sequential(
            self.latent_embedding,
            # NOTE: Do NOT use LayerNorm here!
        )
        # Hidden to hidden transformation: FeedForward layer
        self.hidden_hidden = FeedForward(
            n_embd=self.hidden_size,
            dropout=self.dropout,
        )
        # Instantiate internal hidden model (i.e. the "core")
        self.inner_hidden_model = InnerHiddenModel(
            hidden_hidden_model=self.hidden_hidden,
            hidden_state=self.hidden,
        )

    def init_hidden(self, input_shape=None):
        return None


class PureAttention(Model):
    """
    TODO
    A model that used just the multi-head attention mechanism of the Transformer encoder
    as its internal "core. This is in contrast to NeuralTransformer which uses a complete
    TransformerEncoderLayer as its "core" or inner hidden model.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: Union[int, None] = None,
        loss: Union[Callable, None] = None,
        l1_reg_param: float = 0.0,
        **kwargs,
    ):
        # NOTE: Attention only works with even `embed_dim`
        if hidden_size % 2 != 0:
            logger.info(f"Changing hidden_size from {hidden_size} to {hidden_size+1}.")
            hidden_size = hidden_size + 1
        else:
            logger.info(f"Using hidden_size: {hidden_size}.")

        # Initialize super class
        super(PureAttention, self).__init__(
            input_size,
            hidden_size,
            loss,
            l1_reg_param,
            **kwargs,
        )
        # Special attention parameters
        self.num_heads = find_largest_divisor(
            hidden_size
        )  # number of attention heads (NOTE: must be divisor of `hidden_size`)
        logger.info(f"Number of attention heads: {self.num_heads}.")
        self.dropout = 0.1  # dropout rate
        # Positional encoding (NOTE: must be after embedding)
        self.positional_encoding = PositionalEncoding(
            d_model=self.hidden_size,
            dropout=self.dropout,
        )
        # Input to hidden transformation
        self.input_hidden = torch.nn.Sequential(
            self.latent_embedding,
            self.positional_encoding,
            # NOTE: Do NOT use LayerNorm here!
        )
        # Hidden to hidden transformation: Multihead Attention layer
        self.hidden_hidden = SelfAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_heads,
            dropout=self.dropout,
        )
        # Instantiate internal hidden model (i.e. the "core")
        self.inner_hidden_model = InnerHiddenModel(
            hidden_hidden_model=self.hidden_hidden,
            hidden_state=self.hidden,
        )

    def init_hidden(self, input_shape=None):
        return None


class NeuralTransformer(Model):
    """
    Transformer model for neural activity data.
    Neural activity data is continuous valued and thus
    can naturally be treated as if it were already embedded.
    However, to maintain notational similarity with the original
    Transformer architecture, we use a linear layer to perform
    expansion recoding. This replaces the embedding layer in the
    traditional Transformer but it is really just a linear projection.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: Union[int, None] = None,
        loss: Union[Callable, None] = None,
        l1_reg_param: float = 0.0,
        **kwargs,
    ):
        # NOTE: Transformer only works with even `d_model`
        if hidden_size % 2 != 0:
            logger.info(f"Changing hidden_size from {hidden_size} to {hidden_size+1}.")
            hidden_size = hidden_size + 1
        else:
            logger.info(f"Using hidden_size: {hidden_size}.")

        # Initialize super class
        super(NeuralTransformer, self).__init__(
            input_size,
            hidden_size,
            loss,
            l1_reg_param,
            **kwargs,
        )
        # Special attention parameters
        self.num_heads = find_largest_divisor(
            hidden_size
        )  # number of attention heads (NOTE: must be divisor of `hidden_size`)
        logger.info(f"Number of attention heads: {self.num_heads}.")
        self.dropout = 0.1  # dropout rate
        # Positional encoding (NOTE: must be after embedding)
        self.positional_encoding = PositionalEncoding(
            d_model=self.hidden_size,
            dropout=self.dropout,
        )
        # Input to hidden transformation
        self.input_hidden = torch.nn.Sequential(
            self.latent_embedding,
            self.positional_encoding,
            # NOTE: No LayerNorm because it is already part of TransformerEncoderLayer.  
        )
        # Hidden to hidden transformation: TransformerEncoderLayer
        self.hidden_hidden = CausalTransformer(
            d_model=self.hidden_size,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_size,
            dropout=self.dropout,
        )
        # Instantiate internal hidden model (i.e. the "core")
        self.inner_hidden_model = InnerHiddenModel(
            hidden_hidden_model=self.hidden_hidden,
            hidden_state=self.hidden,
        )

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
        **kwargs,
    ):
        # Initialize super class
        super(NetworkCTRNN, self).__init__(
            input_size,
            hidden_size,
            loss,
            l1_reg_param,
            **kwargs,
        )
        # Input to hidden transformation
        self.input_hidden = torch.nn.Sequential(
            self.latent_embedding,
            # NOTE: YES use LayerNorm here!
            self.layer_norm,
        )
        # Hidden to hidden transformation: Continuous time RNN (CTRNN) layer
        self.hidden_hidden = CTRNN(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
        )
        # Instantiate internal hidden model (i.e. the "core")
        self.inner_hidden_model = InnerHiddenModel(
            hidden_hidden_model=self.hidden_hidden,
            hidden_state=self.hidden,
        )

    def init_hidden(self, input_shape):
        device = next(self.parameters()).device
        batch_size = input_shape[0]  # because batch_first=True
        hidden = torch.zeros(batch_size, self.hidden_size).to(device)
        return hidden


class LiquidCfC(Model):
    """
    Neural Circuit Policy (NCP) Closed-form continuous time (CfC) model.
    Hasani, R., Lechner, M., Amini, A. et al. Closed-form continuous-time neural networks.
    Nat Mach Intell 4, 992–1003 (2022). https://doi.org/10.1038/s42256-022-00556-7.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: Union[int, None] = None,
        loss: Union[Callable, None] = None,
        l1_reg_param: float = 0.0,
        **kwargs,
    ):
        # Initialize super class
        super(LiquidCfC, self).__init__(
            input_size,
            hidden_size,
            loss,
            l1_reg_param,
            **kwargs,
        )
        # Input to hidden transformation
        self.input_hidden = torch.nn.Sequential(
            self.latent_embedding,
            # NOTE: YES use LayerNorm here!
            self.layer_norm,
        )
        # Hidden to hidden transformation: Closed-form continuous-time (CfC) layer
        self.hidden_hidden = CfC(
            input_size=self.hidden_size,
            units=self.hidden_size,
            activation="relu",
        )
        # Instantiate internal hidden model (i.e. the "core")
        self.inner_hidden_model = InnerHiddenModel(
            hidden_hidden_model=self.hidden_hidden,
            hidden_state=self.hidden,
        )
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
        **kwargs,
    ):
        # Initialize super class
        super(NetworkLSTM, self).__init__(
            input_size,
            hidden_size,
            loss,
            l1_reg_param,
            **kwargs,
        )
        # Input to hidden transformation
        self.input_hidden = torch.nn.Sequential(
            self.latent_embedding,
            # NOTE: YES use LayerNorm here!
            self.layer_norm,
        )
        # Hidden to hidden transformation: Long-short term memory (LSTM) layer
        self.hidden_hidden = torch.nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            bias=True,
            batch_first=True,
        )
        # Instantiate internal hidden model (i.e. the "core")
        self.inner_hidden_model = InnerHiddenModel(
            hidden_hidden_model=self.hidden_hidden,
            hidden_state=self.hidden,
        )
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
