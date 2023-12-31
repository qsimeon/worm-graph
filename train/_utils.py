from train._pkg import *

# Init logger
logger = logging.getLogger(__name__)


class EarlyStopping:
    # https://github.com/jeffheaton/t81_558_deep_learning/blob/pytorch/t81_558_class_03_4_early_stop.ipynb
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_model = None
        self.best_loss = None
        self.counter = 0

    def __call__(self, model, val_loss):
        if val_loss is None or math.isnan(val_loss):
            logger.info("Validation loss is not a valid number (NaN).")
            return True

        if self.best_model is None:
            self.best_model = copy.deepcopy(model)
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model)
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model.load_state_dict(model.state_dict())
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1

        if self.counter >= self.patience:
            return True
        return False


def save_model(model, path, other_info=dict()):
    """
    Saves a PyTorch model to disk.

    Args:
        model (nn.Module): The PyTorch model to save.
        path (str): The path to save the model to.
        other_info (dict, optional): Any additional information to save with the model. Defaults to an empty dictionary.
    """
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    base_info = {
        # State dictionaries
        "model_state_dict": model.state_dict(),
        # Names
        "model_name": model.__class__.__name__,
        # Number of parameters
        "num_trainable_params": num_trainable_params,
        # Model parameters
        "input_size": model.get_input_size(),
        "hidden_size": model.get_hidden_size(),
        "loss_name": model.get_loss_name(),
        "l1_reg_param": model.get_l1_reg_param(),
    }
    save_info = {**base_info, **other_info}
    torch.save(save_info, path)


def compute_loss_vectorized(loss_fn, X, Y, mask):
    """
    Computes the loss of X and Y, taking into account the masks.

    Parameters
    ----------
    loss_fn : torch.nn.Module
        A loss function instance. Needs to be initialized with reduction='none'.
    X : torch.Tensor
        A batch of input sequences. Shape: (batch_size, seq_len, input_size).
    Y : torch.Tensor
        A batch of target sequences. Shape: (batch_size, seq_len, input_size).
    mask : torch.Tensor
        A batch of masks. Shape: (batch_size, input_size).

    Returns
    -------
    loss : torch.Tensor
        The mean loss of X and Y.
    """
    # Expand masks along temporal dimension to match the shape of X and Y
    expanded_mask = mask.unsqueeze(1).expand_as(
        X
    )  # the mask is a feature mask; temporally invariant, feature equivariant

    # Mask the invalid positions in X and Y
    masked_X = X * expanded_mask.float()
    masked_Y = Y * expanded_mask.float()

    # Compute the loss without reduction (.e. reduction='none' in `loss_fn`)
    masked_loss = loss_fn(masked_X, masked_Y)

    # Normalize the loss by the total number of data points
    norm_factor = masked_loss[expanded_mask].shape[
        0
    ]  # batch_size * seq_len * masked_input_size
    loss = masked_loss[expanded_mask].sum() / norm_factor

    return loss
