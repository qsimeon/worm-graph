from models._pkg import *

# Init logger
logger = logging.getLogger(__name__)


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

# # # "Cores" or Inner Models for Different Model Architectures # # #
# # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

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
        self.register_parameter(name="alpha", param=torch.nn.Parameter(torch.ones(1, hidden_size)))
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

#region Graph Convolutional Network (GCN): Core / Inner Model for NetworkGCN (work-in-progress)
# # TODO: Work on this model more.
# class GCNModel(torch.nn.Module):
#     """
#     Graph Convolutional Network (GCN) model for _C. elegans_ connectome graph.
#     THIS IS A WORK-IN-PROGRESS
#     """

#     def __init__(
#         self,
#         input_size,
#         hidden_size,
#     ):
#         super().__init__()
#         # Load the connectome graph
#         graph_tensors = torch.load(
#             os.path.join(
#                 ROOT_DIR, "data", "processed", "connectome", "graph_tensors.pt"
#             )
#         )
#         graph = Data(**graph_tensors)
#         assert (
#             graph.num_nodes == input_size
#         ), "Input size must match number of nodes in connectome."

#         # Set attributes
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.edge_index = graph.edge_index
#         self.edge_attr = graph.edge_attr

#         # Define the GCN layers
#         self.elec_conv = GCNConv(
#             in_channels=-1,
#             out_channels=self.hidden_size,
#             improved=True,
#         )  # electrical synapse convolutions,
#         self.chem_conv = GCNConv(
#             in_channels=-1,
#             out_channels=self.hidden_size,
#             improved=True,
#         )  # chemical synapse convolutions
#         self.hid_proj = torch.nn.Linear(
#             in_features=2 * self.hidden_size, out_features=self.hidden_size
#         )  # projection to latent space (i.e hidden state)

#         # Check if first forward call
#         self.first_forward = True
#         self.device = torch.device("cpu")
#         self.random_projection = None

#     def forward(self, x):
#         """
#         GCNConv layers:
#             - input: node features (|V|, F_in), edge indices (2,|E|), edge weights (|E|) (optional)
#             - output: node features (|V|, F_out)

#         x: input tensor w/ shape (batch_size, seq_len, input_size)

#         input_size = 302, which is the number of nodes |V| in the connectome graph of _C. elegans_.
#         """
#         # Check that the input shape is as expected
#         batch_size, seq_len, input_size = x.shape
#         assert input_size == self.input_size, "Incorrectly shaped input tensor."

#         # Move GCNConv layers to same device as data
#         if self.first_forward:
#             self.device = x.device
#             self.elec_conv = self.elec_conv.to(self.device)
#             self.chem_conv = self.chem_conv.to(self.device)
#             self.random_projection = torch.randn(
#                 self.input_size,
#                 seq_len,
#                 requires_grad=False,
#                 dtype=torch.float,
#                 device=self.device,
#             )  # (hidden_size, seq_len)
#             self.first_forward = False

#         # Reshape the input (batch_size, |V| = input_size = 302, seq_len)
#         x = torch.transpose(x, 1, 2)

#         # Create a list of Data objects.
#         data_list = [
#             Data(
#                 x=x[i].to(self.device),
#                 edge_index=self.edge_index.to(self.device),
#                 edge_attr=self.edge_attr.to(self.device),
#             )
#             for i in range(x.size(0))
#         ]

#         # Convert this list into a Batch object.
#         batch = Batch.from_data_list(data_list)

#         # Chemical synapses convolution
#         elec_weight = batch.edge_attr[:, 0]
#         elec_hidden = self.elec_conv(
#             x=batch.x,
#             edge_index=batch.edge_index,
#             edge_weight=elec_weight,
#         )

#         # Gap junctions convolution
#         chem_weight = batch.edge_attr[:, 1]
#         chem_hidden = self.chem_conv(
#             x=batch.x,
#             edge_index=batch.edge_index,
#             edge_weight=chem_weight,
#         )

#         # Concatenate into a single latent
#         hidden = torch.cat([elec_hidden, chem_hidden], dim=-1)

#         # Transform back to the input space
#         x = self.hid_proj(hidden).T  # (batch_size, input_size, hidden_size)
#         x = x.reshape(self.hidden_size, batch_size, self.input_size)
#         x = x @ self.random_projection  # (hidden_size, batch_size, seq_len)
#         x = x.reshape(batch_size, seq_len, self.hidden_size)

#         return x  # (batch_size, seq_len, hidden_size)
#endregion

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
# Provides the input-output backbone and allows changeable mode "cores" # 
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
        4. The core of all models is called `self.hidden_hidden` and it is
            comprised of a single hidden layer of an architecture of choice.
        7. Getter methods for the input size and hidden size called
            `get_input_size`, and `get_hidden_size`, respectively.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
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
        else:
            self.loss = torch.nn.MSELoss

        # Name of original loss function
        self.loss_name = self.loss.__name__[:-4]
        # Setup
        self.input_size = input_size  # Number of neurons (302)
        self.output_size = input_size  # Number of neurons (302)
        self.hidden_size = hidden_size
        self.fft_reg_param = fft_reg_param
        self.l1_reg_param = l1_reg_param
        # Initialize hidden state
        self._init_hidden()
        # Initialize the tau (1 := next-timestep prediction)
        self.tau = 1  
        # Identity layer
        self.identity = torch.nn.Identity()
        # Input to hidden transformation - placeholder
        self.input_hidden = torch.nn.Linear(self.input_size, self.hidden_size)
        # Hidden to hidden transformation - placeholder
        self.hidden_hidden = torch.nn.Linear(self.hidden_size, self.hidden_size)
        # Instantiate internal hidden model - placeholder
        self.inner_hidden_model = InnerHiddenModel(self.hidden_hidden, self.hidden)
        # Linear readout
        self.linear = torch.nn.Linear(self.hidden_size, self.output_size)
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

    @autocast()
    def forward(self, input: torch.Tensor, mask: torch.Tensor, tau: int = 1):
        """
        Forward method for simple linear regression model.

        Parameters
        ----------
        input : torch.Tensor
            Input data with shape (batch, seq_len, neurons)
        mask : torch.Tensor
            Mask on the neurons with shape (batch, neurons)
        tau : int, optional
            Time offset of target
        """

        # store the tau
        self.tau = tau
        # initialize hidden state
        self.hidden = self.init_hidden(input.shape)
        # set hidden state of internal model
        self.inner_hidden_model.set_hidden(self.hidden)
        # recast the mask to the input type and shape
        mask = mask.unsqueeze(1).expand_as(input)
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
            hidden_out = self.inner_hidden_model(latent_out)
            # perform a linear readout to get the output
            readout = self.linear(hidden_out)
            output = readout
        return output.float()  # casting to float fixed autocast problem

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
            Input data with shape (batch, seq_len, neurons)
        mask : torch.Tensor
            Mask on the neurons with shape (batch, neurons)
        nb_ts_to_generate : int
            Number of time steps to generate
        context_window : int
            Number of time steps to use as context

        Returns
        -------
        generated_tensor : torch.Tensor
            Generated data with shape (nb_ts_to_generate, neurons)
        """

        self.eval()  # set model to evaluation mode
        

        if autoregressive:
            # Generate values autoregressively
            input = input[:, :context_window, :]  # shape (1, context_window, 302)

        generated_values = []
        with torch.no_grad():

            for t in range(nb_ts_to_generate):
                # Get the last context_window values of the input tensor
                x = input[
                    :, t : context_window + t, :
                ]  # shape (1, context_window, 302)

                # Get predictions
                predictions = self(
                    x, mask, tau=self.tau
                )  # shape (1, context_window, 302)

                # Get last predicted value
                last_time_step = predictions[:, -1, :].unsqueeze(0)  # shape (1, 1, 302)

                # Append the prediction to the generated_values list and input tensor
                generated_values.append(last_time_step)
                input = torch.cat(
                    (input, last_time_step), dim=1
                )  # add the prediction to the input tensor

        # Stack the generated values to a tensor
        generated_tensor = torch.cat(
            generated_values, dim=1
        )  # shape (nb_ts_to_generate, 302)

        return generated_tensor

    def sample(self, nb_ts_to_sample: int):
        """
        Sample spontaneous neural activity from the model.
        TODO: Figure out how to use diffusion models to sample from the network.
        """
        pass


# # # Models subclasses: Indidividually differentiated model architectures # # # #
# Use the same model backbone provided by Model but with a distinct core or inner hidden model # 
# # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

class LinearNN(Model):
    """
    TODO: Test model with/without using information from the neuron mask.
    A simple linear regression model to use as a baseline.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        loss: Union[Callable, None] = None,
        fft_reg_param: float = 0.0,
        l1_reg_param: float = 0.0,
    ):
        super(LinearNN, self).__init__(
            input_size,
            hidden_size,
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


        # Input to hidden transformation
        self.input_hidden = torch.nn.Sequential(
            self.embedding,
            torch.nn.ReLU(),
            # NOTE: YES use LayerNorm here!
            torch.nn.LayerNorm(self.hidden_size),
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
        hidden_size: int,
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
        self.n_head = 1 # number of attention heads; NOTE: this must be divisor of `hidden_size`
        self.dropout = 0.0  # dropout ratedropout=self.dropout,

        # Embedding
        self.embedding = torch.nn.Linear(
            self.input_size,
            self.hidden_size,
        )  # combine input and mask

        # Input to hidden transformation
        self.input_hidden = torch.nn.Sequential(
            # NOTE: Position encoding before embedding improved performance.
            self.positional_encoding,
            self.embedding,
            torch.nn.ReLU(),
            # NOTE: Do NOT use LayerNorm here!
        )

        # Hidden to hidden transformation: Transformer Encoder layer
        self.hidden_hidden = torch.nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=self.n_head, 
                                                              dim_feedforward=self.hidden_size, dropout=self.dropout,
                                                              activation="relu", batch_first=True, 
                                                              norm_first=True)

        # Instantiate internal hidden model
        self.inner_hidden_model = InnerHiddenModel(self.hidden_hidden, self.hidden)

    def init_hidden(self, input_shape=None):
        return None


class NetworkRNN(Model):
    """
    A model of the C. elegans nervous system using a continuous-time RNN backbone.
    TODO: Cite tutorial by Guangyu Robert Yang and the paper: Artificial Neural Networks for Neuroscientists: A Primer.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        loss: Union[Callable, None] = None,
        fft_reg_param: float = 0.0,
        l1_reg_param: float = 0.0,
    ):
        """
        The output size will be the same as the input size.
        """
        super(NetworkRNN, self).__init__(
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
            torch.nn.LayerNorm(self.hidden_size),
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
        batch_size = input_shape[0]  # beacuse batch_first=True
        hidden = torch.zeros(batch_size, self.hidden_size).to(device)
        return hidden


class NeuralCFC(Model):
    """
    Neural Circuit Policy (NCP) Closed-form continuous time (CfC) model.
    TODO: Cite Nature Machine Intelligence 2022 paper by Ramin Hasani, Daniela Rus et al.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        loss: Union[Callable, None] = None,
        fft_reg_param: float = 0.0,
        l1_reg_param: float = 0.0,
    ):
        """
        The output size will be the same as the input size.
        """

        super(NeuralCFC, self).__init__(
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
            torch.nn.LayerNorm(self.hidden_size),
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
    Given an input sequence of length $L$ and an offset $\tau$,
    this model is trained to output the sequence of length $L$
    that occurs $tau$ time steps after the start of the input sequence.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
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
            torch.nn.LayerNorm(self.hidden_size),
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


#region NetworkGCN: attempt Graph Neural Neural Network Architecture (work-in-progress)
# # TODO: Work on this model more.
# class NetworkGCN(Model):
#     """
#     A graph neural network model of the _C. elegans_ nervous system.
#     """

#     def __init__(
#         self,
#         input_size: int,
#         hidden_size: int,
#         loss: Union[Callable, None] = None,
#         fft_reg_param: float = 0.0,
#         l1_reg_param: float = 0.0,
#     ):
#         super(NetworkGCN, self).__init__(
#             input_size,
#             hidden_size,
#             loss,
#             fft_reg_param,
#             l1_reg_param,
#         )

#         # Input to hidden transformation: Graph Convolutional Network (GCN) layer
#         self.input_hidden = GCNModel(self.input_size, self.hidden_size)

#         # Hidden to hidden transformation: Identity layer
#         self.hidden_hidden = torch.nn.Sequential(
#             self.identity,
#             torch.nn.ReLU(),
#             # NOTE: Do NOT use LayerNorm here!
#         )

#         # Instantiate internal hidden model
#         self.inner_hidden_model = InnerHiddenModel(self.hidden_hidden, self.hidden)

#     def init_hidden(self, input_shape=None):
#         """Initialize the hidden state of the inner model."""
#         self.hidden = None
#         return None
#endregion

# # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#