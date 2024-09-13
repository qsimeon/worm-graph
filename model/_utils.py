from model._pkg import *

# Init logger
logger = logging.getLogger(__name__)


### Custom loss function (MASE) # # #
# # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
class MASELoss(torch.nn.Module):
    """Mean Absolute Scaled Error (MASE) loss function.

    This loss function is used to measure the accuracy of forecasts
    by comparing the mean absolute error of the forecast to the mean
    absolute error of a naive forecast.

    Attributes:
        seasonality: An integer representing the seasonality of the data.
    
    TODO: Improve and validate this implementation of the MASE loss.
    """

    def __init__(self, reduction="mean"):
        """Initializes the MASELoss with the given seasonality.

        Args:
            seasonality: The seasonality of the data. Default is 1.
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, y_pred, target):
        """Calculates the MASE loss between the predicted and true values.

        Args:
            y_pred: The predicted values as a tensor.
            y_true: The true values as a tensor.

        Returns:
            A tensor representing the MASE loss.
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


def load_model_checkpoint(checkpoint_path):
    """Load a model from a checkpoint file.

    The checkpoint should contain all the necessary information to
    reinstantiate the model using its constructor.

    Args:
        checkpoint_path: The path to the checkpoint file.

    Returns:
        The loaded model.
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
    l1_norm_reg_param = checkpoint["l1_norm_reg_param"]
    connectome_reg_param = checkpoint["connectome_reg_param"]
    # New attributes for version 2
    version_2 = checkpoint.get("version_2", VERSION_2)
    num_tokens = checkpoint.get("num_tokens", NUM_TOKENS)
    # Reinstantiate the model
    model = eval(model_name)(
        input_size,
        hidden_size,
        loss=loss_name,
        l1_norm_reg_param=l1_norm_reg_param,
        connectome_reg_param=connectome_reg_param,
        version_2=version_2,
        num_tokens=num_tokens,
    ).to(DEVICE)
    model.load_state_dict(model_state_dict)
    return model


def print_parameters(model, verbose=False):
    """Prints the parameters of a model in a table format.

    This function prints a table of the model's parameters, including the
    module name, the number of parameters, and whether they are trainable.
    If verbose is True, it also prints the total number of parameters and
    the total number of trainable parameters.

    Args:
        model: The model whose parameters are to be printed.
        verbose: If True, prints the total number of parameters and the
            total number of trainable parameters.

    Returns:
        A tuple containing the total number of parameters and the total
        number of trainable parameters.
    """
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

# # # Some helpful module implementations to be used in some model architectures # # #
# # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


class RMSNorm(torch.nn.Module):
    """Root-Mean-Square (RMS) normalization.

    This class implements the RMS normalization technique, which normalizes
    the input tensor based on the root mean square of its elements.

    Attributes:
        eps: A small value to avoid division by zero.
        weight: A learnable parameter for scaling the normalized input.
    """
    def __init__(self, d_model: int, eps: float = 1e-5):
        """Initializes the RMSNorm with the given model dimension and epsilon.

        Args:
            d_model: The dimension of the model.
            eps: A small value to avoid division by zero. Default is 1e-5.
        """
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        """Applies RMS normalization to the input tensor.

        Args:
            x: The input tensor.

        Returns:
            The normalized tensor.
        """
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output


class PositionalEncoding(torch.nn.Module):
    """Sinuosoidal positional encoding from 'Attention is All You Need' paper.

    This class implements the sinusoidal positional encoding technique, which
    adds positional information to the input tensor. The first dimension is
    used as the batch dimension (i.e., batch_first=True).

    Attributes:
        d_model: The dimension of the model.
        max_len: The maximum length of the input sequences.
        dropout: The dropout rate.
        pe: The positional encoding tensor.
    """
    def __init__(
        self,
        d_model: int,
        max_len: int = BLOCK_SIZE,
        dropout: float = 0.1,
    ):
        """Initializes the PositionalEncoding with the given model dimension, maximum length, and dropout rate.

        Args:
            d_model: The dimension of the model.
            max_len: The maximum length of the input sequences. Default is BLOCK_SIZE.
            dropout: The dropout rate. Default is 0.1.
        """
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
        """Adds positional encoding to the input tensor.

        Args:
            x: Tensor, shape (batch_size, seq_len, embedding_dim).

        Returns:
            The input tensor with added positional encoding.
        """
        x = x * math.sqrt(self.d_model)  # normalization used in Transformers
        x = x + self.pe[:, : x.size(1), :]  # add positional encoding to input
        return self.dropout(x)


# # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


# # # "Cores" or Inner Models for Different Model Architectures # # #
# # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


class CausalTransformer(torch.nn.Module):
    """A single Transformer Encoder layer with causal attention.

    This class uses a single Transformer Encoder layer as the hidden-hidden model.
    It sets `is_causal=True` in the forward method of TransformerEncoderLayer.

    Attributes:
        num_layers: The number of layers in the Transformer Encoder.
        is_causal: A boolean indicating whether causal attention is used.
        encoder_layer: The Transformer Encoder layer.
        transformer_encoder: The Transformer Encoder.
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        """Initializes the CausalTransformer with the given parameters.

        Args:
            d_model: The dimension of the model.
            nhead: The number of attention heads.
            dim_feedforward: The dimension of the feedforward network.
            dropout: The dropout rate. Default is 0.1.
        """
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
        """Applies the Transformer Encoder with causal attention to the input tensor.

        Args:
            src: The input tensor with shape (batch_size, seq_len, d_model).

        Returns:
            The output tensor after applying the Transformer Encoder.
        """
        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(
            src.size(1),
            device=src.device,
        )
        out = self.transformer_encoder(src, mask=causal_mask, is_causal=self.is_causal)
        return out


class FeedForward(torch.nn.Module):
    """Simple linear layer followed by a non-linearity and dropout.

    Attributes:
        ffwd: A sequential container of the linear layer, ReLU activation, and dropout.
    """
    def __init__(self, n_embd, dropout=0.1):
        """Initializes the FeedForward layer with the given embedding dimension and dropout rate.

        Args:
            n_embd: The embedding dimension or width of the single hidden layer.
            dropout: The dropout rate. Default is 0.1.
        """
        super().__init__()
        self.ffwd = torch.nn.Sequential(
            torch.nn.Linear(n_embd, n_embd),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x):
        """Applies the feedforward network to the input tensor.

        Args:
            x: The input tensor.

        Returns:
            The output tensor after applying the feedforward network.
        """
        return self.ffwd(x)


class SelfAttention(torch.nn.Module):
    """A single self-attention layer.

    Attributes:
        attn: A multi-head attention module.

    Args:
        embed_dim: The embedding dimension.
        num_heads: The number of attention heads.
        dropout: The dropout rate.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """Initializes the SelfAttention layer with the given parameters.

        Args:
            embed_dim: The embedding dimension.
            num_heads: The number of attention heads.
            dropout: The dropout rate. Default is 0.1.
        """
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout,
            batch_first=True,
        )

    def forward(self, src):
        """Applies self-attention to the input tensor.

        Args:
            src: The input tensor with shape (batch, seq_len, embed_dim).

        Returns:
            The output tensor after applying self-attention.
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


class SSM(torch.nn.Module):
    """State Space Model.

    Parameters:
        input_size: Number of input neurons.
        hidden_size: Number of hidden neurons.
        decode: Whether to use recurrence (True) or convolution (False) to propagate input.

    Inputs:
        input: Tensor of shape (batch_size, seq_len, input_size).
        hidden: Tensor of shape (batch_size, hidden_size), initial hidden activity.
            If None, hidden is initialized through self.init_hidden().

    Outputs:
        output: Tensor of shape (batch_size, seq_len, hidden_size).
        hidden: Tensor of shape (batch_size, hidden_size), final hidden activity.
    """
    def __init__(self, input_size, hidden_size, decode=False):
        """Initializes the SSM with the given input size, hidden size, and decode mode.

        Args:
            input_size: Number of input neurons.
            hidden_size: Number of hidden neurons.
            decode: Whether to use recurrence (True) or convolution (False) to propagate input.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.decode = decode
        # SSM parameters
        self.A = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.A.weight = torch.nn.Parameter(self.make_HiPPO(self.hidden_size), requires_grad=False)
        self.B = torch.nn.Linear(self.input_size, self.hidden_size, bias=False)
        self.register_parameter(name="step", param=torch.nn.Parameter(torch.ones(hidden_size)))

    def discretize(self):
        """Discretizes the continuous-time SSM parameters.

        Returns:
            A tuple containing the discretized A and B matrices.
        """
        I = torch.eye(self.hidden_size, device=self.B.weight.device, dtype=self.B.weight.dtype)
        step_half_A = (self.step / 2.0) * self.A.weight
        BL = torch.linalg.inv(I - step_half_A)
        Ab = BL @ (I + step_half_A)
        Bb = (BL * self.step.unsqueeze(0)) @ self.B.weight
        # Instead of assigning the values directly, return them and use them in forward pass
        return Ab, Bb

    def make_HiPPO(self, N):
        """Creates the HiPPO matrix for the given size.

        Args:
            N: The size of the HiPPO matrix.

        Returns:
            The HiPPO matrix.
        """
        P = torch.sqrt(1 + 2 * torch.arange(N))
        A = P.unsqueeze(-1) @ P.unsqueeze(0)
        A = torch.tril(A) - torch.diag(torch.arange(N))
        return -A

    def init_hidden(self, shape, device):
        """Initializes the hidden state.

        Args:
            shape: The shape of the input tensor.
            device: The device to initialize the hidden state on.

        Returns:
            The initialized hidden state tensor.
        """
        batch_size = shape[0]  # because batch_first=True
        hidden = torch.zeros(batch_size, self.hidden_size).to(device)
        return hidden

    def recurrence(self, input, hidden, A, B):
        """Runs the network for one time step.

        Args:
            input: Tensor of shape (batch_size, input_size).
            hidden: Tensor of shape (batch_size, hidden_size).
            A: The discretized A matrix.
            B: The discretized B matrix.

        Returns:
            The network activity at the next time step.
        """
        # Perform the recurrence update
        h_new = A @ hidden.unsqueeze(-1) + B @ input.unsqueeze(-1)
        return h_new.squeeze(-1)

    def K_conv(self, Ab, Bb, L):
        """Computes the kernels for the convolutional mode of the SSM.

        Args:
            Ab: The discretized A matrix.
            Bb: The discretized B matrix.
            L: The sequence length.

        Returns:
            The kernels for the convolutional mode of the SSM.
        
        NOTE: Warning: this implementation is naive and unstable. In practice it will fail
        to work for more than very small lengths. However, we are going to replace it with
        S4 in Part 2, so for now we just keep it around as a placeholder.
        """
        assert Ab.shape == (self.hidden_size, self.hidden_size) and Bb.shape == (
            self.hidden_size,
            self.input_size,
        ), "SSM parameters are not of the right shape."
        Kernels = torch.stack([(torch.matrix_power(Ab, l) @ Bb) for l in range(L)]).permute(
            (1, 0, -1)
        )
        return Kernels.real  # (hidden_size, L, input_size)

    # ### DEBUG ###
    # # NOTE: This is from Sasha Rush's blog post to simplify computing the matrix power.
    # # I don't fully understand it yet. "To address the problem of computing powers of 𝐴,
    # # we introduce another technique. Instead of computing the SSM convolution filter 𝐾
    # # directly, we introduce a generating function on its coefficients and compute evaluations of it."

    # def K_gen_simple(self, Ab, Bb, L):
    #     '''Not to be used! Naive implementation of the generating function.'''
    #     K = self.K_conv(Ab, Bb, L) # (hidden_size, L, input_size)
    #     def gen(z):
    #         # Sum across the length dimenison
    #         return torch.sum(K * (z ** torch.arange(L)), dim=1) # (hidden_size, input_size)
    #     return gen

    # # The generating function essentially converts the SSM convolution filter from the time domain to frequency domain.
    # # This transformation is also called z-transform (up to a minus sign) in control engineering literature.
    # # Importantly, it preserves the same information, and the desired SSM convolution filter can be recovered.
    # # Once the z-transform of a discrete sequence known, we can obtain the filter’s discrete fourier transform from evaluations of its z-transform at the roots of unity.
    # # Then, we can apply inverse fourier transformation, stably by applying an FFT, to recover the filter.

    # def conv_from_gen(self, gen, L):
    #     # Evaluate at roots of unity
    #     # Generating function is (-)z-transform, so we evaluate at (-)root
    #     Omega_L = torch.exp(-2j * torch.pi * torch.arange(L) / L)
    #     atRoots = torch.stack([gen(omega) for omega in Omega_L]) # (L, hidden_size, input_size)
    #     # Inverse FFT
    #     # TODO: Is there one particular dimension we should apply the inverse FFT to?
    #     out = torch.fft.ifft(atRoots, n=L, dim=0).permute((1, 0, -1)) # (hidden_size, L, input_size)
    #     return out.real

    # # More importantly, in the generating function we can replace the matrix power with an inverse!
    # # And for all 𝑧 at the L-roots of unity we have 𝑧^𝐿=1 so that term is removed.
    # # We then pull this constant term into a new 𝐶. Critically, this function does not call K_conv.

    # def K_gen_inverse(self, Ab, Bb, L):
    #     '''Clever implementation of the generating function.

    #     We will want to replace any calls like this `K_conv(*ssm, L=L)`
    #     with this `conv_from_gen(K_gen_inverse(*ssm, L=L), L)` instead.
    #     '''
    #     I = torch.eye(Ab.shape[0]).to(Ab.device)
    #     Ab_L = torch.matrix_power(Ab, L) # (hidden_size, hidden_size)
    #     Ct = (I - Ab_L) # (hidden_size, hidden_size)
    #     # NOTE: Bb has shape (hidden_size, input_size)
    #     return lambda z: (Ct.to(z.dtype).conj() @ torch.linalg.inv(I - Ab * z.to(Ab.device)) @ Bb.to(z.dtype)) # (hidden_size, input_size)

    # def K_conv_fast(self, Ab, Bb, L):
    #     '''TODO: Replace `K_conv` with `K_conv_fast` in the forward pass.'''
    #     Kernels =  self.conv_from_gen(self.K_gen_inverse(Ab, Bb, L), L)
    #     return Kernels.real # (hidden_size, L, input_size)
    # ### DEBUG ###

    def causal_convolution(self, u, Kernels):
        """Computes the result of applying the filters `Kernels` to inputs `u` using the convolution theorem with Fast Fourier Transform (FFT).

        Args:
            u: Input tensor with shape (seq_len, input_size).
            Kernels: 3D tensor with `hidden_size` independent kernels, each with shape (seq_len, input_size).

        Returns:
            The output tensor with shape (seq_len, hidden_size).
        """
        seq_len = u.size(0)
        assert (
            Kernels.size(0) == self.hidden_size and Kernels.size(-1) == self.input_size
        ), f"`Kernels` should be a tensor w/ shape (hidden_size, seq_len, input_size) but instead has shape {Kernels.shape}."
        assert (
            Kernels.size(1) == seq_len
        ), "The kernels and input do not have the same sequence length."
        outputs = []
        for K in Kernels:  # O(hidden_size) operations
            ud = torch.fft.rfft(u.float(), dim=0)  # (seq_len, input_size)
            Kd = torch.fft.rfft(K.float(), dim=0)  # (seq_len, input_size)
            # NOTE: This next step is very memory intensive. Scales with `seq_len` x `hidden_size`.
            # We quickly go over the GPU memory limit when training large models (i.e large `hidden_size`).
            out = torch.fft.irfft(ud * Kd, n=seq_len, dim=0).sum(dim=-1)  # (seq_len,)
            outputs.append(out)  # (hidden_size, seq_len)
        outputs = torch.stack(outputs).t()  # (seq_len, hidden_size)
        return outputs

    def batch_causal_convolution(self, input, Kernels):
        """Applies causal convolution to a batch of inputs.

        Args:
            input: Input tensor with shape (batch_size, seq_len, input_size).
            Kernels: 3D tensor with `hidden_size` independent kernels, each with shape (seq_len, input_size).

        Returns:
            The output tensor with shape (batch_size, seq_len, hidden_size).
        """
        return torch.vmap(self.causal_convolution, in_dims=(0, None), out_dims=0)(input, Kernels)

    def forward(self, input, hidden=None):
        """Propagates input through the network.

        Args:
            input: Tensor with shape (batch_size, seq_len, input_size).
            hidden: Tensor with shape (batch_size, hidden_size), initial hidden activity. If None, hidden is initialized through self.init_hidden().

        Returns:
            A tuple containing the output tensor with shape (batch_size, seq_len, hidden_size) and the final hidden activity tensor with shape (batch_size, hidden_size).
        """
        # If hidden activity is not provided, initialize it
        if hidden is None:
            hidden = self.init_hidden(input.shape, input.device)
        # Get the discretized SSM parameters
        Ab, Bb = self.discretize()
        # Run the SSM
        if not self.decode:  # CNN mode
            # Most of the work is to build this filter (aka kernel)
            Kernels = self.K_conv(Ab, Bb, input.size(1))
            # # ### DEBUG ###
            # # # TODO: Replace with the faster generating function approach
            # Kernels = self.K_conv_fast(Ab, Bb, input.size(1))
            # # ### DEBUG  ###
            # The actual call to the network is just this (huge) convolution
            output = self.batch_causal_convolution(input, Kernels)
        else:  # RNN mode
            seq_len = input.size(1)
            output = []
            # Loop through time
            for i in range(seq_len):
                # `hidden` is just the most recent state
                hidden = self.recurrence(input[:, i, :], hidden, Ab, Bb)
                output.append(hidden)
            # Stack together output from all time steps
            output = torch.stack(output, dim=1)  # (batch_size, seq_len, hidden_size)
        return output, hidden


class CTRNN(torch.nn.Module):
    """Continuous-time RNN.

    Parameters:
        input_size: Number of input neurons.
        hidden_size: Number of hidden neurons.

    Inputs:
        input: Tensor of shape (batch_size, seq_len, input_size).
        hidden: Tensor of shape (batch_size, hidden_size), initial hidden activity.
            If None, hidden is initialized through self.init_hidden().

    Outputs:
        output: Tensor of shape (batch_size, seq_len, hidden_size).
        hidden: Tensor of shape (batch_size, hidden_size), final hidden activity.
    """

    def __init__(self, input_size, hidden_size):
        """Initializes the CTRNN with the given input size and hidden size.

        Args:
            input_size: Number of input neurons.
            hidden_size: Number of hidden neurons.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.register_parameter(name="alpha", param=torch.nn.Parameter(torch.ones(hidden_size)))
        self.input2h = torch.nn.Linear(input_size, hidden_size)
        self.h2h = torch.nn.Linear(hidden_size, hidden_size)

    def init_hidden(self, shape, device):
        """Initializes the hidden state.

        Args:
            shape: The shape of the input tensor.
            device: The device to initialize the hidden state on.

        Returns:
            The initialized hidden state tensor.
        """
        batch_size = shape[0]  # because batch_first=True
        hidden = torch.zeros(batch_size, self.hidden_size).to(device)
        return hidden

    def recurrence(self, input, hidden):
        """Runs the network for one time step.

        Args:
            input: Tensor of shape (batch_size, input_size).
            hidden: Tensor of shape (batch_size, hidden_size).

        Returns:
            The network activity at the next time step.
        """
        # Perform the recurrence update
        # h_new = torch.relu(self.input2h(input) + self.h2h(hidden)) # ReLU nonlinearity
        h_new = torch.tanh(self.input2h(input) + self.h2h(hidden))  # Tanh nonlinearity
        # The sigmoid contrains alpha such that 0 <= alpha <=1
        h_new = hidden * (1 - self.alpha.sigmoid()) + h_new * self.alpha.sigmoid()
        return h_new

    def forward(self, input, hidden=None):
        """Propagates input through the network.

        Args:
            input: Tensor with shape (batch_size, seq_len, input_size).
            hidden: Tensor with shape (batch_size, hidden_size), initial hidden activity. If None, hidden is initialized through self.init_hidden().

        Returns:
            A tuple containing the output tensor with shape (batch_size, seq_len, hidden_size) and the final hidden activity tensor with shape (batch_size, hidden_size).
        """
        # If hidden activity is not provided, initialize it
        if hidden is None:
            hidden = self.init_hidden(input.shape, input.device)
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
    """Inner hidden (latent) model.

    This class encapsulates the "core" of the inner model of different architectures.

    Attributes:
        hidden_hidden: The hidden-hidden model (core) of the architecture.
        hidden: The hidden state of the model.
    """
    def __init__(self, hidden_hidden_model: torch.nn.Module, hidden_state=None):
        """Initializes the InnerHiddenModel with the given hidden-hidden model and hidden state.

        Args:
            hidden_hidden_model: The hidden-hidden model (core) of the architecture.
            hidden_state: The initial hidden state of the model. Default is None.
        """
        super().__init__()
        self.hidden_hidden = hidden_hidden_model
        self.hidden = hidden_state

    def forward(self, x):
        """Propagates input through the inner hidden model.

        Args:
            x: The input tensor.

        Returns:
            The output tensor after applying the inner hidden model.
        """
        if self.hidden is None:
            x = self.hidden_hidden(x)
        else:
            # Call forward method on input with hidden state
            x, self.hidden = self.hidden_hidden(x, self.hidden)
        return x

    def set_hidden(self, hidden_state):
        """Sets the hidden state of the model.

        Args:
            hidden_state: The hidden state to set.
        """
        self.hidden = hidden_state
        return None


# # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# # # Model super class: Common interface for all model architectures # # # #
# Provides the input-output backbone and allows changeable inner modules a.k.a "cores". #
# # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


class Model(torch.nn.Module):
    """
    Super class for all model.

    For all our model:
        1. The output `readout` will be the same shape as the input.
        2. A method called `loss_fn` that specifies the specific
            loss function to be used by the model. The default
            loss function we use is `torch.nn.MSELoss()`.
        3. A readout layer is implemented and will always be
            called `self.linear_readout`.
        4. The core of all model is called `self.hidden_hidden` and it is
            comprised of a single hidden layer of an architecture of choice.
        5. Getter methods for the input size and hidden size called
        `get_input_size`, `get_hidden_size`, `get_loss_name`, and `get_l1_norm_reg_param`.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: Union[int, None],
        loss: Union[Callable, None] = None,
        l1_norm_reg_param: float = 0.0,
        connectome_reg_param: float = 0.0,
        # New attributes for version 2
        version_2: bool = VERSION_2,
        num_tokens: int = NUM_TOKENS,
        # Additional keyword arguments
        # TODO: Need to add these to checkpoints.
        normalization: Union[str, None] = None,  # 'rms_norm', 'layer_norm'
        positional_encoding: Union[bool, None] = False,
    ):
        """
        Initializes the Model with the given parameters.

        Args:
            input_size: The number of input neurons.
            hidden_size: The number of hidden neurons.
            loss: The loss function to be used by the model. Default is None.
            l1_norm_reg_param: The L1 norm regularization parameter. Default is 0.0.
            connectome_reg_param: The connectome regularization parameter. Default is 0.0.
            version_2: Whether to use version 2 of the model. Default is VERSION_2.
            num_tokens: The number of tokens for version 2. Default is NUM_TOKENS.
            normalization: The type of normalization to use. Default is None.
            positional_encoding: Whether to use positional encoding. Default is False.
        """
        super(Model, self).__init__()
        assert (
            isinstance(l1_norm_reg_param, float) and 0.0 <= l1_norm_reg_param <= 1.0
        ), "The regularization parameter `l1_norm_reg_param` must be a float between 0 and 1."
        assert (
            isinstance(connectome_reg_param, float) and 0.0 <= connectome_reg_param <= 1.0
        ), "The regularization parameter `connectome_reg_param` must be a float between 0 and 1."
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
        self.input_size = input_size  # Number of neurons (NUM_NEURONS)
        self.output_size = input_size  # Number of neurons (NUM_NEURONS)
        # NOTE: The output_size is same as the input_size because the model is a self-supervised autoencoder.
        self.hidden_size = hidden_size if hidden_size is not None else input_size
        self.l1_norm_reg_param = l1_norm_reg_param
        self.connectome_reg_param = connectome_reg_param
        # Load the connectome graph
        self.load_connectome()
        # Initialize hidden state
        self._init_hidden()
        # Identity layer - used if needed
        self.identity = torch.nn.Identity()
        # Embedding layer - placeholder
        self.latent_embedding = torch.nn.Linear(
            self.input_size,
            self.hidden_size,
        )
        # Optional positional encoding layer
        if positional_encoding:
            self.positional_encoding = PositionalEncoding(d_model=self.hidden_size)
        else:
            self.positional_encoding = self.identity
        # Optional normalization layer
        if normalization == "rms_norm":
            self.normalization = RMSNorm(self.hidden_size)
        elif normalization == "layer_norm":
            self.normalization = torch.nn.LayerNorm(self.hidden_size, elementwise_affine=True)
        else:
            self.normalization = self.identity
        # Input to hidden transformation block - placeholder
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
        # Linear readout
        self.linear_readout = torch.nn.Linear(self.hidden_size, self.output_size)
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
            # Define a vector of EMA decay values used to update the neural embedding
            ema_decay = torch.ones(self.input_size) * 0.5
            self.register_buffer("ema_decay", ema_decay)
            # Modify embedding layer to be a lookup table
            self.latent_embedding = torch.nn.Embedding(
                num_embeddings=self.num_tokens, embedding_dim=self.hidden_size
            )  # embedding lookup table (learned)
            # Adjust linear readout to output token logits
            self.linear_readout = torch.nn.Linear(self.hidden_size, self.num_tokens)
            # Alias methods to new versions
            self.forward = self.forward_v2
            self.loss_fn = self.loss_fn_v2
            self.generate = self.generate_v2
        # Initialize weights
        self._init_weights()

    # Initialization functions for setting hidden states and weights.
    def _init_hidden(self):
        """Initializes the hidden state."""
        self.hidden = None
        return None

    def _init_weights(self):
        """Initializes the weights of the model."""
        # Initialize the readout bias
        torch.nn.init.zeros_(self.linear_readout.bias)
        # Initialize the readout weights
        torch.nn.init.xavier_uniform_(self.linear_readout.weight)
        # Initialize the embedding weights
        torch.nn.init.normal_(self.latent_embedding.weight)
        # Initialize the best linear approximation of model weights
        self.register_parameter("ols_weights", torch.nn.Parameter(torch.eye(self.input_size)))
        return None

    @abstractmethod
    def init_hidden(self, shape=None, device=None):
        """
        Initializes the hidden state of the "core".

        Args:
            shape: The shape of the input tensor.
            device: The device to initialize the hidden state on.

        Returns:
            The initialized hidden state tensor.
            
        NOTE: Enforce that all model have an `init_hidden` method which initializes the hidden state of the "core".
        This function must be overridden in subclasses with the specified argument 'shape'.
        """
        raise NotImplementedError()

    # Getter functions for returning all attributes needed to reinstantiate a similar model
    def get_input_size(self):
         """Returns the input size of the model."""
        return self.input_size

    def get_hidden_size(self):
        """Returns the hidden size of the model."""
        return self.hidden_size

    def get_loss_name(self):
        """Returns the name of the loss function."""
        return self.loss_name

    def get_l1_norm_reg_param(self):
        """Returns the L1 norm regularization parameter."""
        return self.l1_norm_reg_param

    def get_connectome_reg_param(self):
        """Returns the connectome regularization parameter."""
        return self.connectome_reg_param

    @torch.autocast(
        device_type=DEVICE.type, dtype=torch.half if "cuda" in DEVICE.type else torch.bfloat16
    )
    def calculate_distances(self, neural_sequence, token_matrix, feature_mask=None):
        """Efficiently calculates Euclidean distances between neural sequence vectors and token matrix vectors.

        Args:
            neural_sequence: Tensor of shape (batch_size, seq_len, input_size).
            token_matrix: Tensor of shape (num_tokens, input_size).
            feature_mask: Tensor of shape (batch_size, input_size). If None, all features are considered.

        Returns:
            distances: Distance matrix for each batch. Shape (batch_size, seq_len, num_tokens).
        """
        # Get input shapes
        batch_size, _, input_size = neural_sequence.shape
        assert input_size == token_matrix.size(
            -1
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
        # Return distance matrix
        return distances

    @torch.autocast(
        device_type=DEVICE.type, dtype=torch.half if "cuda" in DEVICE.type else torch.bfloat16
    )
    def tokenize_neural_data(
        self,
        neural_sequence: torch.Tensor,
        feature_mask: Union[None, torch.Tensor] = None,
        token_matrix: Union[None, torch.Tensor] = None,
        decay: float = 0.5,  # TODO: Make a vector and hyperparameter
    ):
        """
        Convert the high-dimensional sequence of neural states to a 1-D sequence of tokens.
        The approach used is similar to that of VQ-VAEs where the neural data is treated as the
        encoder output, the decoder input is the nearest-neighbor codebook vector, and the tokens
        are the indices of those vectors in the codebok. The decoder is treated as the rest of the
        model after the tokenization step. The dimensionality of the embedding space is the same
        as the dimensionality of the neural data (i.e. `input_size` or `num_channels`).

        Args:
            neural_sequence: Tensor with shape (batch_size, seq_len, input_size).
            feature_mask: Tensor with shape (batch_size, input_size).
            token_matrix: (optional) Tensor with shape (num_tokens, input_size).
            decay: (optional) float EMA decay factor for updating the neural embedding.

        Returns:
            token_sequence: Tensor of shape (batch_size, seq_len).
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
            feature_mask.ndim == 2 and feature_mask.size(-1) == input_size
        ), "`feature_mask` must have shape `(batch_size, input_size)`"
        assert feature_mask.sum().item() > 0, "`feature_mask` cannot be all False."
        if token_matrix is None:
            token_matrix = self.neural_embedding
        assert (
            token_matrix.ndim == 2 and token_matrix.size(-1) == input_size
        ), "`token_matrix` must have shape `(num_tokens, input_size)`"
        # Move token_matrix to same device as neural_sequence
        token_matrix = token_matrix.to(neural_sequence.device)
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
        ### >>> SLOW but CORRECT, NEW Implementation >>> ###
        # NOTE: Updates positions in `self.neural_embedding` that correspond to observed tokens and masked inputs.
        # Get positions of masked input features (i.e. the observed/measured neurons)
        masked_input_positions = feature_mask.nonzero(
            as_tuple=False
        )  # (<= batch_size * input_size, 2)
        # Get unique values and their counts in batch dimension (first column)
        _, counts = torch.unique(masked_input_positions[:, 0], return_counts=True)
        # Split the tensor into groups based on the batch dimension
        batch_groups = torch.split(masked_input_positions, counts.tolist())
        # For each batch index update the neural embedding only using observed tokens and masked features
        for group in batch_groups:  # time ~ bigO(batch_size)
            batch_idx = group[:, 0].unique().item()  # (1, )
            observed_inputs = group[:, 1]  # (<= batch_size * input_size, )
            batch_tokens = token_sequence[batch_idx]  # (seq_len, )
            batch_inputs = neural_sequence[batch_idx]  # (seq_len, input_size)
            for token in batch_tokens.unique():
                decay = self.ema_decay[observed_inputs]  # (input_size, )
                OLD = self.neural_embedding[token, observed_inputs]  # (input_size, )
                NEW = batch_inputs[batch_tokens == token].mean(dim=0)[
                    observed_inputs
                ]  # (input_size, )
                self.neural_embedding[token, observed_inputs] = (
                    decay * OLD + (1 - decay) * NEW
                )  # (input_size, )
        ### <<< SLOW but CORRECT, NEW Implementation <<< ###
        # Return the tokenized sequence
        return token_sequence

    def bin_tensor(self, nt):
        """Converts a neural tensor of continuous values from a standard normal
        distribution into a tensor of discrete values by indexing them into
        bins defined by self.bin_edges.

        Args:
            nt: Neural tensor (batch_size, seq_len, input_size).

        Returns:
            it: Index tensor (batch_size, seq_len, input_size).
        """
        b1 = nt.unsqueeze(-1) > self.bin_edges[:-1]
        b2 = nt.unsqueeze(-1) <= self.bin_edges[1:]
        bool_arr = (b1 * b2).to(torch.long)
        it = bool_arr.argmax(dim=-1) + 1
        return it

    ### DEBUG ###
    def load_connectome(self):
        """
        Loads the connectome from a pre-saved graph and makes it an attribute of the model.
        Distinguishes between the electrical and chemical weight matrices in the connectome.
        """
        graph_tensors = torch.load(
            os.path.join(ROOT_DIR, "data", "processed", "connectome", "graph_tensors.pt")
        )
        connectome = Data(**graph_tensors)
        assert (
            connectome.num_nodes == self.input_size
        ), "Input size must match number of nodes in connectome."
        elec_weights = to_dense_adj(
            edge_index=connectome.edge_index, edge_attr=connectome.edge_attr[:, 0]
        ).squeeze(0)
        self.register_buffer("elec_weights", elec_weights)
        chem_weights = to_dense_adj(
            edge_index=connectome.edge_index, edge_attr=connectome.edge_attr[:, 1]
        ).squeeze(0)
        self.register_buffer("chem_weights", chem_weights)
        return None

    def compute_ols_weights(self, model_in, model_out):
        """
        A helper function that computes the best linear approximation of
        the model weights using ordinary least squares (OLS) regression.
        
        Args:
            model_in: Input tensor.
            model_out: Output tensor.
            
        NOTE: We tried using `torch.linalg.lstsq(A, B)` as recommended
        (instead of `torch.linalg.pinv(A) @ B` which we are uccrently using)
        but the solution that returned contained NaN values which broke backprop.
        """
        # Don't do this time-intensive computation if it's not going to be used
        if self.connectome_reg_param == 0.0 or self.version_2:
            ols_tensor = torch.eye(self.input_size)
        # Don't compute/update the OLS if in inference mode
        elif not torch.is_grad_enabled():
            ols_tensor = self.ols_weights.data
        else:
            A, B = model_in.float(), model_out.float()
            ols_tensor = torch.linalg.pinv(A) @ B
            if torch.isnan(ols_tensor).any():
                ols_tensor = torch.nan_to_num(ols_tensor, nan=0.0)  # replace nans with 0
            if ols_tensor.ndim == 3:
                ols_tensor = torch.nanmean(ols_tensor, dim=0)  # average over batch
        # Save the current best linear approximation of the model's weights
        self.register_parameter("ols_weights", torch.nn.Parameter(ols_tensor))
        return None

    ### DEBUG ###

    @torch.autocast(
        device_type=DEVICE.type, dtype=torch.half if "cuda" in DEVICE.type else torch.bfloat16
    )
    def forward(self, input: torch.Tensor, mask: torch.Tensor):
        """Common forward method for all models.

        Args:
            input: Input data with shape (batch, seq_len, neurons).
            mask: Mask on the neurons with shape (batch, neurons).

        Returns:
            The output tensor after applying the model.
        """
        # Route to the appropriate forward method
        if self.version_2:
            return self.forward_v2(input, mask)
        # Initialize hidden state
        self.hidden = self.init_hidden(input.shape, input.device)
        # Set hidden state of internal model
        self.inner_hidden_model.set_hidden(self.hidden)
        # Multiply input by the mask (expanded to match input shape)
        input_activity = self.identity(
            input * mask.unsqueeze(1).expand_as(input)
        )  # (batch_size, seq_len, input_size)
        # Transform the input into a latent
        latent_out = self.input_hidden(input_activity)  # (batch_size, seq_len, hidden_size)
        # Transform the latent
        hidden_out = self.inner_hidden_model(latent_out)  # (batch_size, seq_len, hidden_size)
        # Perform a linear readout to get the output
        output = self.linear_readout(hidden_out)  # (batch_size, seq_len, input_size)
        ### DEBUG ###
        # Compute best linear approxmation of model weights using OLS estimate
        self.compute_ols_weights(model_in=input_activity, model_out=output)
        ### DEBUG ###
        # Return the output neural data
        return output

    @torch.autocast(
        device_type=DEVICE.type, dtype=torch.half if "cuda" in DEVICE.type else torch.bfloat16
    )
    def forward_v2(self, input: torch.Tensor, mask: torch.Tensor):
        """Special forward method for the newer version (version_2) of the model
        based on first tokenizing the high-dimensional neural data before
        doing sequence modeling to mimic the approach used in Transformers.

        Args:
            input: Input data with shape (batch, seq_len, neurons).
            mask: Mask on the neurons with shape (batch, neurons).

        Returns:
            The output tensor after applying the model.
        """
        # Initialize hidden state
        self.hidden = self.init_hidden(input.shape, input.device)
        # Set hidden state of internal model
        self.inner_hidden_model.set_hidden(self.hidden)
        # Multiply input by the mask (expanded to match input shape)
        input_activity = self.identity(
            input * mask.unsqueeze(1).expand_as(input)
        )  # (batch_size, seq_len, input_size)
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
        output_logits = self.linear_readout(hidden_out)  # (batch_size, seq_len, num_tokens)
        # Return output token logits
        return output_logits

    @torch.autocast(
        device_type=DEVICE.type, dtype=torch.half if "cuda" in DEVICE.type else torch.bfloat16
    )
    def loss_fn(self):
        """
        The loss function to be used by all the model (default versions).
        This custom loss function combines a primary loss function with an additional
        L1 regularization on all model weights. This regularization term encourages the
        model to use fewer non-zero parameters, effectively making the model more sparse.
        This can help to prevent overfitting, make the model more interpretable, and improve
        generalization by encouraging the model to use only the most important features. The
        L1 penalty is the sum of the absolute values of the weights.
        """
        # Route to the appropriate `loss_fn` method
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
                mask = torch.ones(target.size(0), target.size(-1), dtype=torch.bool).to(
                    target.device
                )
            # No need to expand mask; use broadcasting to apply the mask
            masked_output = output * mask.unsqueeze(
                1
            )  # mask.unsqueeze(1) has shape [batch_size, 1, input_size]
            masked_target = target * mask.unsqueeze(1)
            # Compute the reconstruction loss without reduction
            masked_recon_loss = self.loss(reduction="none", **kwargs)(
                masked_output, masked_target
            ).float()
            # Consider only the masked dimensions for summing the loss
            valid_data_mask = (
                mask.unsqueeze(1).expand_as(output).bool()
            )  # for use in indexing without storing expanded tensor
            # Normalize the loss by the total number of valid data points
            norm_factor = valid_data_mask.sum()
            mrlv = masked_recon_loss[valid_data_mask]
            numerator = mrlv.sum()
            recon_loss = numerator / torch.clamp(norm_factor, min=1)
            # L1 regularization term
            l1_loss = 0.0
            l1_reg_loss = 0.0
            if self.l1_norm_reg_param > 0.0:
                # Calculate L1 regularization term for all weights
                for param in self.parameters():
                    l1_loss += torch.abs(param).mean()
                l1_reg_loss = self.l1_norm_reg_param * l1_loss
            # Connectome regularization term
            connectome_loss = 0.0
            connectome_reg_loss = 0.0
            if self.connectome_reg_param > 0.0:
                # Calculate the connectome regularization term
                # NOTE: Squared OLS weights because the connectome weights are non-negative
                param = torch.square(self.ols_weights) - (self.chem_weights + self.elec_weights)
                connectome_loss = torch.norm(param, p="fro")
                connectome_reg_loss = self.connectome_reg_param * connectome_loss
            # Add the L1 and connectome penalties to the original loss
            total_loss = recon_loss + l1_reg_loss + connectome_reg_loss
            # Return loss
            return total_loss

        # Return the inner custom loss function
        return loss

    ### >>> DEBUG: A different loss function more akin to that used with LM heads is needed for the verion_2
    # mode of our model which first discretize the neural data into tokens before forward method. >>> ###
    @torch.autocast(
        device_type=DEVICE.type, dtype=torch.half if "cuda" in DEVICE.type else torch.bfloat16
    )
    def loss_fn_v2(self):
        """
        Special loss function for the newer version (version_2) of the model based
        on how loss is calculated in Transformers which operate on tokenized data.
        NOTE: Version 2 of the model cannot apply the connectome matching regularization
            term that is available to Version 1 model because it Version 2 model
            input and output on tokens instead of neural states.
        """

        def loss(output, target, mask=None, **kwargs):
            """
            Args:
                output: tensor w/ shape `[batch_size, seq_len, num_tokens]`
                target: tensor w/ shape `[batch_size, seq_len, input_size]`
                mask: tensor w/ shape `[batch_size, input_size]``
            """
            # Default mask to all True if not provided
            if mask is None:
                mask = torch.ones(target.size(0), target.size(-1), dtype=torch.bool).to(
                    target.device
                )
            # Flatten output logits along batch x time dimensions
            output = output.view(
                -1, self.num_tokens
            )  # (batch_size, seq_len, num_tokens) -> (batch_size * seq_len, num_tokens)
            # Convert target from neural vector sequence to token sequence.
            target = self.tokenize_neural_data(
                neural_sequence=target,
                feature_mask=mask,
            )
            target = target.view(-1)  # (batch_size, seq_len) -> (batch_size * seq_len)
            # Calculate cross entropy loss from predicted token logits and target tokens.
            ce_loss = torch.nn.CrossEntropyLoss(reduction="mean", **kwargs)(output, target)
            # L1 regularization term
            l1_loss = 0.0
            l1_reg_loss = 0.0
            if self.l1_norm_reg_param > 0.0:
                # Calculate L1 regularization term for all weights
                for param in self.parameters():
                    l1_loss += torch.abs(param).mean()
                l1_reg_loss = self.l1_norm_reg_param * l1_loss
            # Add the L1 penalty to the original loss
            total_loss = ce_loss + l1_reg_loss
            # NOTE: Version 2 model do not have the connectome regularization term.
            # Return loss
            return total_loss

        # Return the inner custom loss function
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
        torch.Tensor : Generated data with shape (batch_size, num_new_timesteps, neurons)
        """
        # Route to the appropriate generate method
        if self.version_2:
            return self.generate_v2(input, mask, num_new_timesteps, context_window)
        # Set model to evaluation mode
        self.eval()
        # Detach and copy input and convert to dtype of model
        dtype = next(self.parameters()).dtype
        input_copy = input.detach().clone().to(dtype=dtype)
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
            # Append the prediction to the running sequence and continue
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
        Special generate method for the newer version (version_2) of the model based on how
        generation is done in Transformers. In the newer version (version_2), model takes continuous
        valued neural data as input but outputs token logits. Therefore, we must convert the token logits
        back to continuous neural data to be fed back into the model. To do this, we sample from the
        distribution over the predicted next token, retrieve the mean neural state vector corresponding to that
        token, append that neural data state vector to the running neural data sequence, then repeat this process.
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

class LatentModel(torch.nn.Module):
    """
    Super class for all model.

    For all our model:
        1. The output `readout` will be the same shape as the input.
        2. A method called `loss_fn` that specifies the specific
            loss function to be used by the model. The default
            loss function we use is `torch.nn.MSELoss()`.
        3. A readout layer is implemented and will always be
            called `self.linear_readout`.
        4. The core of all model is called `self.hidden_hidden` and it is
            comprised of a single hidden layer of an architecture of choice.
        7. Getter methods for the input size and hidden size called
        `get_input_size`, `get_hidden_size`, `get_loss_name`, and `get_l1_norm_reg_param`.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: Union[int, None],
        loss: Union[Callable, None] = None,
        l1_norm_reg_param: float = 0.0,
        connectome_reg_param: float = 0.0,
        # New attributes for version 2
        version_2: bool = VERSION_2,
        num_tokens: int = NUM_TOKENS,
        # Additional keyword arguments
        # TODO: Need to add these to checkpoints.
        normalization: Union[str, None] = None,  # 'rms_norm', 'layer_norm'
        positional_encoding: Union[bool, None] = False,
    ):
        """
        Defines attributes common to all model.
        """
        super(LatentModel, self).__init__()
        assert (
            isinstance(l1_norm_reg_param, float) and 0.0 <= l1_norm_reg_param <= 1.0
        ), "The regularization parameter `l1_norm_reg_param` must be a float between 0 and 1."
        assert (
            isinstance(connectome_reg_param, float) and 0.0 <= connectome_reg_param <= 1.0
        ), "The regularization parameter `connectome_reg_param` must be a float between 0 and 1."
        # Loss function
        if (loss is None) or (str(loss).lower() == "l1"):
            self.loss = torch.nn.L1Loss
            self.latent_loss = torch.nn.L1Loss
        elif str(loss).lower() == "mse":
            self.loss = torch.nn.MSELoss
            self.latent_loss = torch.nn.MSELoss
        elif str(loss).lower() == "huber":
            self.loss = torch.nn.HuberLoss
            self.latent_loss = torch.nn.HuberLoss
        elif str(loss).lower() == "mase":
            self.loss = MASELoss
            self.latent_loss = MASELoss
        else:
            self.loss = torch.nn.MSELoss
            self.latent_loss = torch.nn.MSELoss
        # Name of original loss function
        self.loss_name = self.loss.__name__[:-4]
        # Setup
        self.input_size = input_size  # Number of neurons (NUM_NEURONS)
        self.output_size = input_size  # Number of neurons (NUM_NEURONS)
        # NOTE: The output_size is same as the input_size because the model is a self-supervised autoencoder.
        self.hidden_size = hidden_size if hidden_size is not None else input_size
        self.l1_norm_reg_param = l1_norm_reg_param
        self.connectome_reg_param = connectome_reg_param
        # Load the connectome graph
        self.load_connectome()
        # Initialize hidden state
        self._init_hidden()
        # Identity layer - used if needed
        self.identity = torch.nn.Identity()
        # Embedding layer - placeholder
        self.latent_embedding = torch.nn.Linear(
            self.input_size,
            self.hidden_size,
        )
        # Optional positional encoding layer
        if positional_encoding:
            self.positional_encoding = PositionalEncoding(d_model=self.hidden_size)
        else:
            self.positional_encoding = self.identity
        # Optional normalization layer
        if normalization == "rms_norm":
            self.normalization = RMSNorm(self.hidden_size)
        elif normalization == "layer_norm":
            self.normalization = torch.nn.LayerNorm(self.hidden_size, elementwise_affine=True)
        else:
            self.normalization = self.identity
        # Input to hidden transformation block - placeholder
        # self.input_hidden = (
        #     torch.nn.Linear(self.input_size, self.hidden_size)
        #     if hidden_size is not None
        #     else torch.nn.Identity()
        # )
        # Hidden to hidden transformation - placeholder
        self.input_hidden = (
            torch.nn.Linear(self.input_size, self.hidden_size)
            if hidden_size is not None
            else torch.nn.Identity()
        )
        self.hidden_output = (
            torch.nn.Linear(self.hidden_size, self.output_size)
            if hidden_size is not None
            else torch.nn.Identity()
        )
        # Instantiate internal hidden model (i.e. the "core") - placeholder
        self.inner_input_hidden_model = InnerHiddenModel(
            hidden_hidden_model=self.input_hidden,
            hidden_state=self.hidden,
        )
        self.inner_hidden_output_model = InnerHiddenModel(
            hidden_hidden_model=self.hidden_output,
            hidden_state=self.latent_hidden,
        )
        # Linear readout
        # self.linear_readout = torch.nn.Linear(self.hidden_size, self.output_size)
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
            # Define a vector of EMA decay values used to update the neural embedding
            ema_decay = torch.ones(self.input_size) * 0.5
            self.register_buffer("ema_decay", ema_decay)
            # Modify embedding layer to be a lookup table
            self.latent_embedding = torch.nn.Embedding(
                num_embeddings=self.num_tokens, embedding_dim=self.hidden_size
            )  # embedding lookup table (learned)
            # Adjust linear readout to output token logits
            self.linear_readout = torch.nn.Linear(self.hidden_size, self.num_tokens)
            # Alias methods to new versions
            self.forward = self.forward_v2
            self.loss_fn = self.loss_fn_v2
            self.generate = self.generate_v2
        # Initialize weights
        self._init_weights()

    # Initialization functions for setting hidden states and weights.
    def _init_hidden(self):
        self.hidden = None
        self.latent_hidden = None
        return self.hidden, self.latent_hidden

    def _init_weights(self):
        # Initialize the readout bias
        # torch.nn.init.zeros_(self.linear_readout.bias)
        # Initialize the readout weights
        # torch.nn.init.xavier_uniform_(self.linear_readout.weight)
        # Initialize the embedding weights
        # torch.nn.init.normal_(self.latent_embedding.weight)
        # Initialize the best linear approximation of model weights
        self.register_parameter("ols_weights", torch.nn.Parameter(torch.eye(self.input_size)))
        return None

    @abstractmethod
    def init_hidden(self, shape=None, device=None):
        """
        Enforce that all model have an `init_hidden` method which initializes the hidden state of the "core".
        This function must be overridden in subclasses with the specified argument 'shape'.
        """
        raise NotImplementedError()

    # Getter functions for returning all attributes needed to reinstantiate a similar model
    def get_input_size(self):
        return self.input_size

    def get_hidden_size(self):
        return self.hidden_size

    def get_loss_name(self):
        return self.loss_name

    def get_l1_norm_reg_param(self):
        return self.l1_norm_reg_param

    def get_connectome_reg_param(self):
        return self.connectome_reg_param

    @torch.autocast(
        device_type=DEVICE.type, dtype=torch.half if "cuda" in DEVICE.type else torch.bfloat16
    )
    def calculate_distances(self, neural_sequence, token_matrix, feature_mask=None):
        """
        Efficiently calculates Euclidean distances between neural sequence vectors and token matrix vectors.

        Args:
            neural_sequence (torch.Tensor): Shape (batch_size, seq_len, input_size).
            token_matrix (torch.Tensor): Shape (num_tokens, input_size).
            feature_mask (torch.Tensor, optional): Shape (batch_size, input_size). If None, all features are considered.

        Returns:
            distances (torch.Tensor): Distance matrix for each batch. Shape (batch_size, seq_len, num_tokens).
        """
        # Get input shapes
        batch_size, _, input_size = neural_sequence.shape
        assert input_size == token_matrix.size(
            -1
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
        # Return distance matrix
        return distances

    @torch.autocast(
        device_type=DEVICE.type, dtype=torch.half if "cuda" in DEVICE.type else torch.bfloat16
    )
    def tokenize_neural_data(
        self,
        neural_sequence: torch.Tensor,
        feature_mask: Union[None, torch.Tensor] = None,
        token_matrix: Union[None, torch.Tensor] = None,
        decay: float = 0.5,  # TODO: Make a vector and hyperparameter
    ):
        """
        Convert the high-dimensional sequence of neural states to a 1-D sequence of tokens.
        The approach used is similar to that of VQ-VAEs where the neural data is treated as the
        encoder output, the decoder input is the nearest-neighbor codebook vector, and the tokens
        are the indices of those vectors in the codebok. The decoder is treated as the rest of the
        model after the tokenization step. The dimensionality of the embedding space is the same
        as the dimensionality of the neural data (i.e. `input_size` or `num_channels`).

        Args:
            neural_sequence: tensor with shape (batch_size, seq_len, input_size)
            feature_mask: tensor with shape (batch_size, input_size)
            token_matrix: (optional) tensor with shape (num_tokens, input_size)
            decay: (optional) float EMA decay factor for updating the neural embedding
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
            feature_mask.ndim == 2 and feature_mask.size(-1) == input_size
        ), "`feature_mask` must have shape `(batch_size, input_size)`"
        assert feature_mask.sum().item() > 0, "`feature_mask` cannot be all False."
        if token_matrix is None:
            token_matrix = self.neural_embedding
        assert (
            token_matrix.ndim == 2 and token_matrix.size(-1) == input_size
        ), "`token_matrix` must have shape `(num_tokens, input_size)`"
        # Move token_matrix to same device as neural_sequence
        token_matrix = token_matrix.to(neural_sequence.device)
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
        ### >>> SLOW but CORRECT, NEW Implementation >>> ###
        # NOTE: Updates positions in `self.neural_embedding` that correspond to observed tokens and masked inputs.
        # Get positions of masked input features (i.e. the observed/measured neurons)
        masked_input_positions = feature_mask.nonzero(
            as_tuple=False
        )  # (<= batch_size * input_size, 2)
        # Get unique values and their counts in batch dimension (first column)
        _, counts = torch.unique(masked_input_positions[:, 0], return_counts=True)
        # Split the tensor into groups based on the batch dimension
        batch_groups = torch.split(masked_input_positions, counts.tolist())
        # For each batch index update the neural embedding only using observed tokens and masked features
        for group in batch_groups:  # time ~ bigO(batch_size)
            batch_idx = group[:, 0].unique().item()  # (1, )
            observed_inputs = group[:, 1]  # (<= batch_size * input_size, )
            batch_tokens = token_sequence[batch_idx]  # (seq_len, )
            batch_inputs = neural_sequence[batch_idx]  # (seq_len, input_size)
            for token in batch_tokens.unique():
                decay = self.ema_decay[observed_inputs]  # (input_size, )
                OLD = self.neural_embedding[token, observed_inputs]  # (input_size, )
                NEW = batch_inputs[batch_tokens == token].mean(dim=0)[
                    observed_inputs
                ]  # (input_size, )
                self.neural_embedding[token, observed_inputs] = (
                    decay * OLD + (1 - decay) * NEW
                )  # (input_size, )
        ### <<< SLOW but CORRECT, NEW Implementation <<< ###
        # Return the tokenized sequence
        return token_sequence

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

    ### DEBUG ###
    def load_connectome(self):
        """
        Loads the connectome from a pre-saved graph and makes it an attribute of the model.
        Distinguishes between the electrical and chemical weight matrices in the connectome.
        """
        graph_tensors = torch.load(
            os.path.join(ROOT_DIR, "data", "processed", "connectome", "graph_tensors.pt")
        )
        connectome = Data(**graph_tensors)
        assert (
            connectome.num_nodes == self.input_size
        ), "Input size must match number of nodes in connectome."
        elec_weights = to_dense_adj(
            edge_index=connectome.edge_index, edge_attr=connectome.edge_attr[:, 0]
        ).squeeze(0)
        self.register_buffer("elec_weights", elec_weights)
        chem_weights = to_dense_adj(
            edge_index=connectome.edge_index, edge_attr=connectome.edge_attr[:, 1]
        ).squeeze(0)
        self.register_buffer("chem_weights", chem_weights)
        return None

    def compute_ols_weights(self, model_in, model_out):
        """
        A helper function that computes the best linear approximation of
        the model weights using ordinary least squares (OLS) regression.
        NOTE: We tried using `torch.linalg.lstsq(A, B)` as recommended
        (instead of `torch.linalg.pinv(A) @ B` which we are uccrently using)
        but the solution that returned contained NaN values which broke backprop.
        """
        # Don't do this time-intensive computation if it's not going to be used
        if self.connectome_reg_param == 0.0 or self.version_2:
            ols_tensor = torch.eye(self.input_size)
        # Don't compute/update the OLS if in inference mode
        elif not torch.is_grad_enabled():
            ols_tensor = self.ols_weights.data
        else:
            A, B = model_in.float(), model_out.float()
            ols_tensor = torch.linalg.pinv(A) @ B
            if torch.isnan(ols_tensor).any():
                ols_tensor = torch.nan_to_num(ols_tensor, nan=0.0)  # replace nans with 0
            if ols_tensor.ndim == 3:
                ols_tensor = torch.nanmean(ols_tensor, dim=0)  # average over batch
        # Save the current best linear approximation of the model's weights
        self.register_parameter("ols_weights", torch.nn.Parameter(ols_tensor))
        return None

    ### DEBUG ###

    @torch.autocast(
        device_type=DEVICE.type, dtype=torch.half if "cuda" in DEVICE.type else torch.bfloat16
    )
    def forward(self, input: torch.Tensor, mask: torch.Tensor):
        """
        Common forward method for all model.

        Parameters
        ----------
        input : torch.Tensor
            Input data with shape (batch, seq_len, neurons)
        mask : torch.Tensor
            Mask on the neurons with shape (batch, neurons)
        """
        # Route to the appropriate forward method
        # if self.version_2:
        #     return self.forward_v2(input, mask)
        # Initialize hidden state
        self.hidden, self.latent_hidden = self.init_hidden(input.shape, input.device)
        # Set hidden state of internal model
        self.inner_input_hidden_model.set_hidden(self.hidden)
        self.inner_hidden_output_model.set_hidden(self.latent_hidden)
        # Multiply input by the mask (expanded to match input shape)
        input_activity = self.identity(
            input * mask.unsqueeze(1).expand_as(input)
        )  # (batch_size, seq_len, input_size)
        # Transform the input into a latent
        # latent_out = self.input_hidden(input_activity)  # (batch_size, seq_len, hidden_size)
        # Transform the latent
        hidden_out = self.inner_input_hidden_model(input_activity)  # (batch_size, seq_len, hidden_size)
        # Perform a linear readout to get the output
        output = self.inner_hidden_output_model(hidden_out)
        # output = self.linear_readout(hidden_out)  # (batch_size, seq_len, input_size)
        ### DEBUG ###
        # Compute best linear approxmation of model weights using OLS estimate
        self.compute_ols_weights(model_in=input_activity, model_out=output)
        ### DEBUG ###
        # Return the output neural data
        return hidden_out, output

    @torch.autocast(
        device_type=DEVICE.type, dtype=torch.half if "cuda" in DEVICE.type else torch.bfloat16
    )
    def forward_v2(self, input: torch.Tensor, mask: torch.Tensor):
        """
        Special forward method for the newer version (version_2) of the model
        based on first tokenizing the high-dimensional neural data before
        doing sequence modeling to mimic the approach used in Transformers.
        """
        # Initialize hidden state
        self.hidden = self.init_hidden(input.shape, input.device)
        # Set hidden state of internal model
        self.inner_hidden_model.set_hidden(self.hidden)
        # Multiply input by the mask (expanded to match input shape)
        input_activity = self.identity(
            input * mask.unsqueeze(1).expand_as(input)
        )  # (batch_size, seq_len, input_size)
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
        output_logits = self.linear_readout(hidden_out)  # (batch_size, seq_len, num_tokens)
        # Return output token logits
        return output_logits

    @torch.autocast(
        device_type=DEVICE.type, dtype=torch.half if "cuda" in DEVICE.type else torch.bfloat16
    )
    def loss_fn(self):
        """
        The loss function to be used by all the model (default versions).
        This custom loss function combines a primary loss function with an additional
        L1 regularization on all model weights. This regularization term encourages the
        model to use fewer non-zero parameters, effectively making the model more sparse.
        This can help to prevent overfitting, make the model more interpretable, and improve
        generalization by encouraging the model to use only the most important features. The
        L1 penalty is the sum of the absolute values of the weights.
        """
        # Route to the appropriate `loss_fn` method
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
                mask = torch.ones(target.size(0), target.size(-1), dtype=torch.bool).to(
                    target.device
                )
            # No need to expand mask; use broadcasting to apply the mask
            masked_output = output * mask.unsqueeze(
                1
            )  # mask.unsqueeze(1) has shape [batch_size, 1, input_size]
            masked_target = target * mask.unsqueeze(1)
            # Compute the reconstruction loss without reduction
            masked_recon_loss = self.loss(reduction="none", **kwargs)(
                masked_output, masked_target
            ).float()
            # Consider only the masked dimensions for summing the loss
            valid_data_mask = (
                mask.unsqueeze(1).expand_as(output).bool()
            )  # for use in indexing without storing expanded tensor
            # Normalize the loss by the total number of valid data points
            norm_factor = valid_data_mask.sum()
            mrlv = masked_recon_loss[valid_data_mask]
            numerator = mrlv.sum()
            recon_loss = numerator / torch.clamp(norm_factor, min=1)
            # L1 regularization term
            l1_loss = 0.0
            l1_reg_loss = 0.0
            if self.l1_norm_reg_param > 0.0:
                # Calculate L1 regularization term for all weights
                for param in self.parameters():
                    l1_loss += torch.abs(param).mean()
                l1_reg_loss = self.l1_norm_reg_param * l1_loss
            # Connectome regularization term
            connectome_loss = 0.0
            connectome_reg_loss = 0.0
            if self.connectome_reg_param > 0.0:
                # Calculate the connectome regularization term
                # NOTE: Squared OLS weights because the connectome weights are non-negative
                param = torch.square(self.ols_weights) - (self.chem_weights + self.elec_weights)
                connectome_loss = torch.norm(param, p="fro")
                connectome_reg_loss = self.connectome_reg_param * connectome_loss
            # Add the L1 and connectome penalties to the original loss
            total_loss = recon_loss + l1_reg_loss + connectome_reg_loss
            # Return loss
            return total_loss

        # Return the inner custom loss function
        return loss
    
    @torch.autocast(
        device_type=DEVICE.type, dtype=torch.half if "cuda" in DEVICE.type else torch.bfloat16
    )
    def latent_loss_fn(self):
        """
        The loss function to be used by all the model (default versions).
        This custom loss function combines a primary loss function with an additional
        L1 regularization on all model weights. This regularization term encourages the
        model to use fewer non-zero parameters, effectively making the model more sparse.
        This can help to prevent overfitting, make the model more interpretable, and improve
        generalization by encouraging the model to use only the most important features. The
        L1 penalty is the sum of the absolute values of the weights.
        """
        # Route to the appropriate `loss_fn` method
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
                mask = torch.ones(target.size(0), target.size(-1), dtype=torch.bool).to(
                    target.device
                )
            # No need to expand mask; use broadcasting to apply the mask
            masked_output = output * mask.unsqueeze(
                1
            )  # mask.unsqueeze(1) has shape [batch_size, 1, input_size]
            masked_target = target * mask.unsqueeze(1)
            # Compute the reconstruction loss without reduction
            masked_recon_loss = self.loss(reduction="none", **kwargs)(
                masked_output, masked_target
            ).float()
            # Consider only the masked dimensions for summing the loss
            valid_data_mask = (
                mask.unsqueeze(1).expand_as(output).bool()
            )  # for use in indexing without storing expanded tensor
            # Normalize the loss by the total number of valid data points
            norm_factor = valid_data_mask.sum()
            mrlv = masked_recon_loss[valid_data_mask]
            numerator = mrlv.sum()
            recon_loss = numerator / torch.clamp(norm_factor, min=1)
            # L1 regularization term
            l1_loss = 0.0
            l1_reg_loss = 0.0
            if self.l1_norm_reg_param > 0.0:
                # Calculate L1 regularization term for all weights
                for param in self.parameters():
                    l1_loss += torch.abs(param).mean()
                l1_reg_loss = self.l1_norm_reg_param * l1_loss
            # Connectome regularization term
            connectome_loss = 0.0
            connectome_reg_loss = 0.0
            if self.connectome_reg_param > 0.0:
                # Calculate the connectome regularization term
                # NOTE: Squared OLS weights because the connectome weights are non-negative
                param = torch.square(self.ols_weights) - (self.chem_weights + self.elec_weights)
                connectome_loss = torch.norm(param, p="fro")
                connectome_reg_loss = self.connectome_reg_param * connectome_loss
            # Add the L1 and connectome penalties to the original loss
            total_loss = recon_loss + l1_reg_loss + connectome_reg_loss
            # Return loss
            return total_loss

        # Return the inner custom loss function
        return loss
            
    ### >>> DEBUG: A different loss function more akin to that used with LM heads is needed for the verion_2
    # mode of our model which first discretize the neural data into tokens before forward method. >>> ###
    @torch.autocast(
        device_type=DEVICE.type, dtype=torch.half if "cuda" in DEVICE.type else torch.bfloat16
    )
    def loss_fn_v2(self):
        """
        Special loss function for the newer version (version_2) of the model based
        on how loss is calculated in Transformers which operate on tokenized data.
        NOTE: Version 2 of the model cannot apply the connectome matching regularization
            term that is available to Version 1 model because it Version 2 model
            input and output on tokens instead of neural states.
        """

        def loss(output, target, mask=None, **kwargs):
            """
            Args:
                output: tensor w/ shape `[batch_size, seq_len, num_tokens]`
                target: tensor w/ shape `[batch_size, seq_len, input_size]`
                mask: tensor w/ shape `[batch_size, input_size]``
            """
            # Default mask to all True if not provided
            if mask is None:
                mask = torch.ones(target.size(0), target.size(-1), dtype=torch.bool).to(
                    target.device
                )
            # Flatten output logits along batch x time dimensions
            output = output.view(
                -1, self.num_tokens
            )  # (batch_size, seq_len, num_tokens) -> (batch_size * seq_len, num_tokens)
            # Convert target from neural vector sequence to token sequence.
            target = self.tokenize_neural_data(
                neural_sequence=target,
                feature_mask=mask,
            )
            target = target.view(-1)  # (batch_size, seq_len) -> (batch_size * seq_len)
            # Calculate cross entropy loss from predicted token logits and target tokens.
            ce_loss = torch.nn.CrossEntropyLoss(reduction="mean", **kwargs)(output, target)
            # L1 regularization term
            l1_loss = 0.0
            l1_reg_loss = 0.0
            if self.l1_norm_reg_param > 0.0:
                # Calculate L1 regularization term for all weights
                for param in self.parameters():
                    l1_loss += torch.abs(param).mean()
                l1_reg_loss = self.l1_norm_reg_param * l1_loss
            # Add the L1 penalty to the original loss
            total_loss = ce_loss + l1_reg_loss
            # NOTE: Version 2 model do not have the connectome regularization term.
            # Return loss
            return total_loss

        # Return the inner custom loss function
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
        torch.Tensor : Generated data with shape (batch_size, num_new_timesteps, neurons)
        """
        # Route to the appropriate generate method
        if self.version_2:
            return self.generate_v2(input, mask, num_new_timesteps, context_window)
        # Set model to evaluation mode
        self.eval()
        # Detach and copy input and convert to dtype of model
        dtype = next(self.parameters()).dtype
        input_copy = input.detach().clone().to(dtype=dtype)
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
            # Append the prediction to the running sequence and continue
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
        Special generate method for the newer version (version_2) of the model based on how
        generation is done in Transformers. In the newer version (version_2), model takes continuous
        valued neural data as input but outputs token logits. Therefore, we must convert the token logits
        back to continuous neural data to be fed back into the model. To do this, we sample from the
        distribution over the predicted next token, retrieve the mean neural state vector corresponding to that
        token, append that neural data state vector to the running neural data sequence, then repeat this process.
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

    Attributes:
        input_hidden: Identity transformation for input to hidden.
        hidden_hidden: Identity transformation for hidden to hidden.
        inner_hidden_model: The internal hidden model (core).
        linear_readout: Identity transformation for the linear readout.
    
    NOTE:
    (1) This model will throw an error if you try to train it because
    it has no trainable parameters and thus has no gradient function.
    (2) This model does is not designed to work with version_2.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: Union[int, None] = None,  # unused in this model
        loss: Union[Callable, None] = None,
        l1_norm_reg_param=0.0,
        **kwargs,
    ):
        """
        Initializes the NaivePredictor with the given parameters.

        Args:
            input_size: The number of input neurons.
            hidden_size: The number of hidden neurons (unused in this model).
            loss: The loss function to be used by the model. Default is None.
            l1_norm_reg_param: The L1 norm regularization parameter. Default is 0.0.
            **kwargs: Additional keyword arguments.
        """
        # NaivePredictor does not work with version_2
        if kwargs.get("version_2", True):
            logger.info(
                f"NaivePredictor does not work with version_2 "
                "because it does not output token logits. "
                "Switching to version_1."
            )
        kwargs["version_2"] = False
        # Specify positional encoding and normalization
        kwargs["positional_encoding"] = False
        kwargs["normalization"] = None
        # Initialize super class
        super(NaivePredictor, self).__init__(
            input_size,
            None,  # hidden_size
            loss,
            l1_norm_reg_param,
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
        self.linear_readout = torch.nn.Identity()
        # Create a dud parameter to avoid errors with the optimizer
        self._ = torch.nn.Parameter(torch.tensor(0.0))

    def init_hidden(self, shape=None, device=None):
        """Initializes the hidden state.

        Args:
            shape: The shape of the input tensor.
            device: The device to initialize the hidden state on.

        Returns:
            None
        """
        return None


class LinearRegression(Model):
    """
    A simple linear regression model.
    This model can only learn a fixed linear feature regression
    function that it applies at every time step independently. It
    is thus memoryless, but can learn linear feature regression.

    Attributes:
        input_hidden: Transformation from input to hidden.
        hidden_hidden: Identity transformation for hidden to hidden.
        inner_hidden_model: The internal hidden model (core).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: Union[int, None] = None,  # unused in this model
        loss: Union[Callable, None] = None,
        l1_norm_reg_param=0.0,
        **kwargs,
    ):
        """
        Initializes the LinearRegression model with the given parameters.

        Args:
            input_size: The number of input neurons.
            hidden_size: The number of hidden neurons (unused in this model).
            loss: The loss function to be used by the model. Default is None.
            l1_norm_reg_param: The L1 norm regularization parameter. Default is 0.0.
            **kwargs: Additional keyword arguments.
        """
        # Specify positional encoding and normalization
        kwargs["positional_encoding"] = False
        kwargs["normalization"] = None
        # Initialize super class
        super(LinearRegression, self).__init__(
            input_size,
            None,  # hidden_size
            loss,
            l1_norm_reg_param,
            **kwargs,
        )
        # Input to hidden transformation
        self.input_hidden = torch.nn.Sequential(
            self.latent_embedding,
            self.positional_encoding,
            self.normalization,
        )
        # Hidden to hidden transformation
        self.hidden_hidden = torch.nn.Identity()
        # Instantiate internal hidden model (i.e. the "core")
        self.inner_hidden_model = InnerHiddenModel(
            hidden_hidden_model=self.hidden_hidden,
            hidden_state=self.hidden,
        )

    def init_hidden(self, shape=None, device=None):
        """Initializes the hidden state.

        Args:
            shape: The shape of the input tensor.
            device: The device to initialize the hidden state on.

        Returns:
            None
        """
        return None


class FeatureFFNN(Model):
    """
    A simple nonlinear regression model.
    FFNN stands for FeedForward Neural Network.
    Unlike the LSTM and Transformer model, this
    model has no temporal memory and can only learn
    a fixed nonlinear feature regression function that
    it applies at every time step independently. It is thus
    memoryless, but can learn nonlinear feature regression.

    Attributes:
        dropout: The dropout rate.
        input_hidden: Transformation from input to hidden.
        hidden_hidden: FeedForward layer for hidden to hidden transformation.
        inner_hidden_model: The internal hidden model (core).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: Union[int, None] = None,
        loss: Union[Callable, None] = None,
        l1_norm_reg_param: float = 0.0,
        connectome_reg_param: float = 0.0,
        **kwargs,
    ):
        """
        Initializes the FeatureFFNN model with the given parameters.

        Args:
            input_size: The number of input neurons.
            hidden_size: The number of hidden neurons.
            loss: The loss function to be used by the model. Default is None.
            l1_norm_reg_param: The L1 norm regularization parameter. Default is 0.0.
            connectome_reg_param: The connectome regularization parameter. Default is 0.0.
            **kwargs: Additional keyword arguments.
        """
        # Specify positional encoding and normalization
        kwargs["positional_encoding"] = False
        kwargs["normalization"] = "layer_norm"
        # Initialize super class
        super(FeatureFFNN, self).__init__(
            input_size,
            hidden_size,
            loss,
            l1_norm_reg_param,
            connectome_reg_param,
            **kwargs,
        )
        # Special parameters for this model
        self.dropout = 0.1  # dropout rate
        # Input to hidden transformation
        self.input_hidden = torch.nn.Sequential(
            self.latent_embedding,
            self.positional_encoding,
            self.normalization,
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

    def init_hidden(self, shape=None, device=None):
        """Initializes the hidden state.

        Args:
            shape: The shape of the input tensor.
            device: The device to initialize the hidden state on.

        Returns:
            None
        """
        return None


class PureAttention(Model):
    """
    A model that uses just the multi-head attention mechanism of the Transformer encoder
    as its internal "core". This is in contrast to NeuralTransformer which uses a complete
    TransformerEncoderLayer as its "core" or inner hidden model.

    Attributes:
        num_heads: The number of attention heads.
        dropout: The dropout rate.
        input_hidden: Transformation from input to hidden.
        hidden_hidden: Multihead Attention layer for hidden to hidden transformation.
        inner_hidden_model: The internal hidden model (core).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: Union[int, None] = None,
        loss: Union[Callable, None] = None,
        l1_norm_reg_param: float = 0.0,
        connectome_reg_param: float = 0.0,
        **kwargs,
    ):
        """
        Initializes the PureAttention model with the given parameters.

        Args:
            input_size: The number of input neurons.
            hidden_size: The number of hidden neurons.
            loss: The loss function to be used by the model. Default is None.
            l1_norm_reg_param: The L1 norm regularization parameter. Default is 0.0.
            connectome_reg_param: The connectome regularization parameter. Default is 0.0.
            **kwargs: Additional keyword arguments.
        """
        # NOTE: Attention only works with even `embed_dim`
        if hidden_size % 2 != 0:
            logger.info(f"Changing hidden_size from {hidden_size} to {hidden_size+1}.")
            hidden_size = hidden_size + 1
        else:
            logger.info(f"Using hidden_size: {hidden_size}.")
        # Specify positional encoding and normalization
        kwargs["positional_encoding"] = True
        kwargs["normalization"] = None
        # Initialize super class
        super(PureAttention, self).__init__(
            input_size,
            hidden_size,
            loss,
            l1_norm_reg_param,
            connectome_reg_param,
            **kwargs,
        )
        # Special parameters for this model
        # NOTE: Number of attention heads must be divisor of `hidden_size`
        self.num_heads = max([i for i in range(1, 9) if hidden_size % i == 0])
        logger.info(f"Number of attention heads: {self.num_heads}.")
        self.dropout = 0.1  # dropout rate
        # Input to hidden transformation
        self.input_hidden = torch.nn.Sequential(
            self.latent_embedding,
            self.positional_encoding,
            self.normalization,
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

    def init_hidden(self, shape=None, device=None):
        """Initializes the hidden state.

        Args:
            shape: The shape of the input tensor.
            device: The device to initialize the hidden state on.

        Returns:
            None
        """
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

    Attributes:
        num_heads: The number of attention heads.
        dropout: The dropout rate.
        input_hidden: Transformation from input to hidden.
        hidden_hidden: TransformerEncoderLayer for hidden to hidden transformation.
        inner_hidden_model: The internal hidden model (core).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: Union[int, None] = None,
        loss: Union[Callable, None] = None,
        l1_norm_reg_param: float = 0.0,
        connectome_reg_param: float = 0.0,
        **kwargs,
    ):
        """
        Initializes the NeuralTransformer model with the given parameters.

        Args:
            input_size: The number of input neurons.
            hidden_size: The number of hidden neurons.
            loss: The loss function to be used by the model. Default is None.
            l1_norm_reg_param: The L1 norm regularization parameter. Default is 0.0.
            connectome_reg_param: The connectome regularization parameter. Default is 0.0.
            **kwargs: Additional keyword arguments.
        """
        # NOTE: Transformer only works with even `d_model`
        if hidden_size % 2 != 0:
            logger.info(f"Changing hidden_size from {hidden_size} to {hidden_size+1}.")
            hidden_size = hidden_size + 1
        else:
            logger.info(f"Using hidden_size: {hidden_size}.")
        # Specify positional encoding and normalization
        kwargs["positional_encoding"] = True
        kwargs["normalization"] = None
        # Initialize super class
        super(NeuralTransformer, self).__init__(
            input_size,
            hidden_size,
            loss,
            l1_norm_reg_param,
            connectome_reg_param,
            **kwargs,
        )
        # Special parameters for this model
        # NOTE: Number of attention heads must be divisor of `hidden_size`
        self.num_heads = max([i for i in range(1, 9) if hidden_size % i == 0])
        logger.info(f"Number of attention heads: {self.num_heads}.")
        self.dropout = 0.1  # dropout rate
        # Input to hidden transformation
        self.input_hidden = torch.nn.Sequential(
            self.latent_embedding,
            self.positional_encoding,
            self.normalization,
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

    def init_hidden(self, shape=None, device=None):
        """Initializes the hidden state.

        Args:
            shape: The shape of the input tensor.
            device: The device to initialize the hidden state on.

        Returns:
            None
        """
        return None


class HippoSSM(Model):
    """
    A model of the _C. elegans_ nervous system using a state-space model (SSM) backbone
    that utilizes the HiPPo matrix for long-term sequence modeling.

    Attributes:
        decode: Whether to use convolution (True) or recurrence (False) mode of SSM.
        input_hidden: Transformation from input to hidden.
        hidden_hidden: State Space Model (SSM) layer for hidden to hidden transformation.
        inner_hidden_model: The internal hidden model (core).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: Union[int, None] = None,
        loss: Union[Callable, None] = None,
        l1_norm_reg_param: float = 0.0,
        connectome_reg_param: float = 0.0,
        **kwargs,
    ):
        """
        Initializes the HippoSSM model with the given parameters.

        Args:
            input_size: The number of input neurons.
            hidden_size: The number of hidden neurons.
            loss: The loss function to be used by the model. Default is None.
            l1_norm_reg_param: The L1 norm regularization parameter. Default is 0.0.
            connectome_reg_param: The connectome regularization parameter. Default is 0.0.
            **kwargs: Additional keyword arguments.
        """
        # Specify positional encoding and normalization
        kwargs["positional_encoding"] = False
        kwargs["normalization"] = "layer_norm"
        # Initialize super class
        super(HippoSSM, self).__init__(
            input_size,
            hidden_size,
            loss,
            l1_norm_reg_param,
            connectome_reg_param,
            **kwargs,
        )
        # Special parameters for this model
        self.decode = False  # convolution (True) or recurrence (False) mode of SSM
        # Input to hidden transformation
        self.input_hidden = torch.nn.Sequential(
            self.latent_embedding,
            self.positional_encoding,
            self.normalization,
        )
        # Hidden to hidden transformation: State Space Model (SSM) layer
        self.hidden_hidden = SSM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            decode=self.decode,
        )
        # Instantiate internal hidden model (i.e. the "core")
        self.inner_hidden_model = InnerHiddenModel(
            hidden_hidden_model=self.hidden_hidden,
            hidden_state=self.hidden,
        )

    def init_hidden(self, shape, device):
        """Initializes the hidden state.

        Args:
            shape: The shape of the input tensor.
            device: The device to initialize the hidden state on.

        Returns:
            The initialized hidden state tensor.
        """
        batch_size = shape[0]  # because batch_first=True
        hidden = torch.zeros(batch_size, self.hidden_size).to(device)
        return hidden


class NetworkCTRNN(Model):
    """
    A model of the C. elegans nervous system using a continuous-time RNN backbone.
    TODO: Cite tutorial by Guangyu Robert Yang and the paper: Artificial Neural Networks for Neuroscientists: A Primer.

    Attributes:
        input_hidden: Transformation from input to hidden.
        hidden_hidden: Continuous time RNN (CTRNN) layer for hidden to hidden transformation.
        inner_hidden_model: The internal hidden model (core).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: Union[int, None] = None,
        loss: Union[Callable, None] = None,
        l1_norm_reg_param: float = 0.0,
        connectome_reg_param: float = 0.0,
        **kwargs,
    ):
        """
        Initializes the NetworkCTRNN model with the given parameters.

        Args:
            input_size: The number of input neurons.
            hidden_size: The number of hidden neurons.
            loss: The loss function to be used by the model. Default is None.
            l1_norm_reg_param: The L1 norm regularization parameter. Default is 0.0.
            connectome_reg_param: The connectome regularization parameter. Default is 0.0.
            **kwargs: Additional keyword arguments.
        """
        # Specify positional encoding and normalization
        kwargs["positional_encoding"] = False
        kwargs["normalization"] = "layer_norm"
        # Initialize super class
        super(NetworkCTRNN, self).__init__(
            input_size,
            hidden_size,
            loss,
            l1_norm_reg_param,
            connectome_reg_param,
            **kwargs,
        )
        # Input to hidden transformation
        self.input_hidden = torch.nn.Sequential(
            self.latent_embedding,
            self.positional_encoding,
            self.normalization,
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

    def init_hidden(self, shape, device):
        """Initializes the hidden state.

        Args:
            shape: The shape of the input tensor.
            device: The device to initialize the hidden state on.

        Returns:
            The initialized hidden state tensor.
        """
        batch_size = shape[0]  # because batch_first=True
        hidden = torch.zeros(batch_size, self.hidden_size).to(device)
        return hidden


class LiquidCfC(Model):
    """
    Neural Circuit Policy (NCP) Closed-form continuous time (CfC) model.
    Hasani, R., Lechner, M., Amini, A. et al. Closed-form continuous-time neural networks.
    Nat Mach Intell 4, 992–1003 (2022). https://doi.org/10.1038/s42256-022-00556-7.

    Attributes:
        input_hidden: Transformation from input to hidden.
        hidden_hidden: Closed-form continuous-time (CfC) layer for hidden to hidden transformation.
        inner_hidden_model: The internal hidden model (core).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: Union[int, None] = None,
        loss: Union[Callable, None] = None,
        l1_norm_reg_param: float = 0.0,
        connectome_reg_param: float = 0.0,
        **kwargs,
    ):
        """
        Initializes the LiquidCfC model with the given parameters.

        Args:
            input_size: The number of input neurons.
            hidden_size: The number of hidden neurons.
            loss: The loss function to be used by the model. Default is None.
            l1_norm_reg_param: The L1 norm regularization parameter. Default is 0.0.
            connectome_reg_param: The connectome regularization parameter. Default is 0.0.
            **kwargs: Additional keyword arguments.
        """
        # Specify positional encoding and normalization
        kwargs["positional_encoding"] = False
        kwargs["normalization"] = "layer_norm"
        # Initialize super class
        super(LiquidCfC, self).__init__(
            input_size,
            hidden_size,
            loss,
            l1_norm_reg_param,
            connectome_reg_param,
            **kwargs,
        )
        # Input to hidden transformation
        self.input_hidden = torch.nn.Sequential(
            self.latent_embedding,
            self.positional_encoding,
            self.normalization,
        )
        # Hidden to hidden transformation: Closed-form continuous-time (CfC) layer
        self.hidden_hidden = CfC(
            input_size=self.hidden_size,
            units=self.hidden_size,
            # activation="relu", # DEBUG: default is "lecun_tanh"
        )
        # Instantiate internal hidden model (i.e. the "core")
        self.inner_hidden_model = InnerHiddenModel(
            hidden_hidden_model=self.hidden_hidden,
            hidden_state=self.hidden,
        )
        # Initialize RNN weights
        self.init_weights()

    def init_hidden(self, shape, device):
        """
        Initializes the hidden state of the RNN.

        Args:
            shape: The shape of the input tensor.
            device: The device to initialize the hidden state on.

        Returns:
            The initialized hidden state tensor.
        """
        batch_size = shape[0]  # because batch_first=True
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

    Attributes:
        input_hidden: Transformation from input to hidden.
        hidden_hidden: Long-short term memory (LSTM) layer for hidden to hidden transformation.
        inner_hidden_model: The internal hidden model (core).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: Union[int, None] = None,
        loss: Union[Callable, None] = None,
        l1_norm_reg_param: float = 0.0,
        connectome_reg_param: float = 0.0,
        **kwargs,
    ):
        """
        Initializes the NetworkLSTM model with the given parameters.

        Args:
            input_size: The number of input neurons.
            hidden_size: The number of hidden neurons.
            loss: The loss function to be used by the model. Default is None.
            l1_norm_reg_param: The L1 norm regularization parameter. Default is 0.0.
            connectome_reg_param: The connectome regularization parameter. Default is 0.0.
            **kwargs: Additional keyword arguments.
        """
        # Specify positional encoding and normalization
        kwargs["positional_encoding"] = False
        kwargs["normalization"] = "layer_norm"
        # Initialize super class
        super(NetworkLSTM, self).__init__(
            input_size,
            hidden_size,
            loss,
            l1_norm_reg_param,
            connectome_reg_param,
            **kwargs,
        )
        # Input to hidden transformation
        self.input_hidden = torch.nn.Sequential(
            self.latent_embedding,
            self.positional_encoding,
            self.normalization,
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

    def init_hidden(self, shape, device):
        """
        Initializes the hidden and cell states of the LSTM.

        Args:
            shape: The shape of the input tensor.
            device: The device to initialize the hidden state on.

        Returns:
            A tuple containing the initialized hidden state tensor and cell state tensor.
        """
        batch_size = shape[0]  # because batch_first=True
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


class NetworkLatentLSTM(torch.nn.Module):
    """
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: Union[int, None] = None,
        loss: Union[Callable, None] = None,
        l1_norm_reg_param: float = 0.0,
        connectome_reg_param: float = 0.0,
        **kwargs,
    ):
        # Specify positional encoding and normalization
        kwargs["positional_encoding"] = False
        kwargs["normalization"] = "layer_norm"
        # Initialize super class
        super(NetworkLatentLSTM, self).__init__()
        
        self.loss = torch.nn.MSELoss
        self.latent_loss = torch.nn.MSELoss
        
        self.l1_norm_reg_param = 0
        self.connectome_reg_param = 0
        
        self.identity = torch.nn.Identity()
        
        self.input_hidden = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=True,
            batch_first=True,
        )
        
        self.hidden_output = torch.nn.LSTM(
            input_size=hidden_size,
            hidden_size=input_size,
            bias=True,
            batch_first=True,
        )
        # Initialize LSTM weights
        self.init_weights()
    
    @torch.autocast(
        device_type=DEVICE.type, dtype=torch.half if "cuda" in DEVICE.type else torch.bfloat16
    )
    def forward(self, input: torch.Tensor, mask: torch.Tensor):
        """
        Common forward method for all model.

        Parameters
        ----------
        input : torch.Tensor
            Input data with shape (batch, seq_len, neurons)
        mask : torch.Tensor
            Mask on the neurons with shape (batch, neurons)
        """
        # Route to the appropriate forward method
        # if self.version_2:
        #     return self.forward_v2(input, mask)
        # Multiply input by the mask (expanded to match input shape)
        # input_activity = self.identity(
        #     input * mask.unsqueeze(1).expand_as(input)
        # )  # (batch_size, seq_len, input_size)
        # Transform the input into a latent
        # latent_out = self.input_hidden(input_activity)  # (batch_size, seq_len, hidden_size)
        # Transform the latent
        hidden_out, (latent_h, latent_c) = self.input_hidden(input)  # (batch_size, seq_len, hidden_size)
        # Perform a linear readout to get the output
        output, (out_h, out_c) = self.hidden_output(hidden_out)
        # output = self.linear_readout(hidden_out)  # (batch_size, seq_len, input_size)
        ### DEBUG ###
        # Compute best linear approxmation of model weights using OLS estimate
        # self.compute_ols_weights(model_in=input_activity, model_out=output)
        ### DEBUG ###
        # Return the output neural data
        return hidden_out[:, :-1, :], output

    def init_hidden(self, shape, device):
        """
        Inititializes the hidden and cell states of the LSTM.
        """
        batch_size = shape[0]  # because batch_first=True
        h0 = torch.zeros(1, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(1, batch_size, self.hidden_size).to(device)
        return (h0, c0)
    
    def init_weights(self):
        """
        Initializes the weights of the LSTM.
        """
        for name, param in self.input_hidden.named_parameters():
            if "weight_ih" in name:  # Input-hidden weights
                torch.nn.init.xavier_uniform_(param.data, gain=1.5)
            elif "weight_hh" in name:  # Hidden-hidden weights
                torch.nn.init.orthogonal_(param.data)
            elif "bias" in name:  # Bias weights
                # param.data.fill_(0)
                torch.nn.init.zeros_(param.data)
        
        for name, param in self.hidden_output.named_parameters():
            if "weight_ih" in name:  # Input-hidden weights
                torch.nn.init.xavier_uniform_(param.data, gain=1.5)
            elif "weight_hh" in name:  # Hidden-hidden weights
                torch.nn.init.orthogonal_(param.data)
            elif "bias" in name:  # Bias weights
                # param.data.fill_(0)
                torch.nn.init.zeros_(param.data)
                
    @torch.autocast(
        device_type=DEVICE.type, dtype=torch.half if "cuda" in DEVICE.type else torch.bfloat16
    )
    def loss_fn(self):
        """
        The loss function to be used by all the model (default versions).
        This custom loss function combines a primary loss function with an additional
        L1 regularization on all model weights. This regularization term encourages the
        model to use fewer non-zero parameters, effectively making the model more sparse.
        This can help to prevent overfitting, make the model more interpretable, and improve
        generalization by encouraging the model to use only the most important features. The
        L1 penalty is the sum of the absolute values of the weights.
        """

        def loss(output, target, mask=None, **kwargs):
            """
            Calculate loss with added L1 regularization
            on the trainable model parameters.

            Arguments:
                output: (batch_size, seq_len, input_size)
                target: (batch_size, seq_len, input_size)
                mask: (batch_size, input_size)
            """
            mask=None
            # Default mask to all True if not provided
            if mask is None:
                mask = torch.ones(target.size(0), target.size(-1), dtype=torch.bool).to(
                    target.device
                )
            # No need to expand mask; use broadcasting to apply the mask
            masked_output = output * mask.unsqueeze(
                1
            )  # mask.unsqueeze(1) has shape [batch_size, 1, input_size]
            masked_target = target * mask.unsqueeze(1)
            # Compute the reconstruction loss without reduction
            masked_recon_loss = self.loss(reduction="none", **kwargs)(
                masked_output, masked_target
            ).float()
            # Consider only the masked dimensions for summing the loss
            valid_data_mask = (
                mask.unsqueeze(1).expand_as(output).bool()
            )  # for use in indexing without storing expanded tensor
            # Normalize the loss by the total number of valid data points
            norm_factor = valid_data_mask.sum()
            mrlv = masked_recon_loss[valid_data_mask]
            numerator = mrlv.sum()
            recon_loss = numerator / torch.clamp(norm_factor, min=1)
            # L1 regularization term
            l1_loss = 0.0
            l1_reg_loss = 0.0
            if self.l1_norm_reg_param > 0.0:
                # Calculate L1 regularization term for all weights
                for param in self.parameters():
                    l1_loss += torch.abs(param).mean()
                l1_reg_loss = self.l1_norm_reg_param * l1_loss
            # Connectome regularization term
            connectome_loss = 0.0
            connectome_reg_loss = 0.0
            if self.connectome_reg_param > 0.0:
                # Calculate the connectome regularization term
                # NOTE: Squared OLS weights because the connectome weights are non-negative
                param = torch.square(self.ols_weights) - (self.chem_weights + self.elec_weights)
                connectome_loss = torch.norm(param, p="fro")
                connectome_reg_loss = self.connectome_reg_param * connectome_loss
            # Add the L1 and connectome penalties to the original loss
            total_loss = recon_loss + l1_reg_loss + connectome_reg_loss
            # Return loss
            return total_loss

        # Return the inner custom loss function
        return loss
    
    @torch.autocast(
        device_type=DEVICE.type, dtype=torch.half if "cuda" in DEVICE.type else torch.bfloat16
    )
    def latent_loss_fn(self):
        """
        The loss function to be used by all the model (default versions).
        This custom loss function combines a primary loss function with an additional
        L1 regularization on all model weights. This regularization term encourages the
        model to use fewer non-zero parameters, effectively making the model more sparse.
        This can help to prevent overfitting, make the model more interpretable, and improve
        generalization by encouraging the model to use only the most important features. The
        L1 penalty is the sum of the absolute values of the weights.
        """
        def loss(output, target, mask=None, **kwargs):
            """
            Calculate loss with added L1 regularization
            on the trainable model parameters.

            Arguments:
                output: (batch_size, seq_len, input_size)
                target: (batch_size, seq_len, input_size)
                mask: (batch_size, input_size)
            """
            mask=None
            # Default mask to all True if not provided
            if mask is None:
                mask = torch.ones(target.size(0), target.size(-1), dtype=torch.bool).to(
                    target.device
                )
            # No need to expand mask; use broadcasting to apply the mask
            masked_output = output * mask.unsqueeze(
                1
            )  # mask.unsqueeze(1) has shape [batch_size, 1, input_size]
            masked_target = target * mask.unsqueeze(1)
            # Compute the reconstruction loss without reduction
            masked_recon_loss = self.latent_loss(reduction="none", **kwargs)(
                masked_output, masked_target
            ).float()
            # Consider only the masked dimensions for summing the loss
            valid_data_mask = (
                mask.unsqueeze(1).expand_as(output).bool()
            )  # for use in indexing without storing expanded tensor
            # Normalize the loss by the total number of valid data points
            norm_factor = valid_data_mask.sum()
            mrlv = masked_recon_loss[valid_data_mask]
            numerator = mrlv.sum()
            recon_loss = numerator / torch.clamp(norm_factor, min=1)
            # L1 regularization term
            l1_loss = 0.0
            l1_reg_loss = 0.0
            if self.l1_norm_reg_param > 0.0:
                # Calculate L1 regularization term for all weights
                for param in self.parameters():
                    l1_loss += torch.abs(param).mean()
                l1_reg_loss = self.l1_norm_reg_param * l1_loss
            # Connectome regularization term
            connectome_loss = 0.0
            connectome_reg_loss = 0.0
            if self.connectome_reg_param > 0.0:
                # Calculate the connectome regularization term
                # NOTE: Squared OLS weights because the connectome weights are non-negative
                param = torch.square(self.ols_weights) - (self.chem_weights + self.elec_weights)
                connectome_loss = torch.norm(param, p="fro")
                connectome_reg_loss = self.connectome_reg_param * connectome_loss
            # Add the L1 and connectome penalties to the original loss
            total_loss = recon_loss + l1_reg_loss + connectome_reg_loss
            # Return loss
            return total_loss

        # Return the inner custom loss function
        return loss

# # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
