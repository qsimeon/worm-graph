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


def save_model(model, path):
    torch.save(
        {
            # State dictionaries
            "model_state_dict": model.state_dict(),
            # Names
            "model_name": model.__class__.__name__,
            # Model parameters
            "input_size": model.get_input_size(),
            "hidden_size": model.get_hidden_size(),
            "loss_name": model.get_loss_name(),
            "fft_reg_param": model.get_fft_reg_param(),
            "l1_reg_param": model.get_l1_reg_param(),
        },
        path,
    )


def compute_loss_vectorized(loss_fn, X, Y, masks):
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
    masks : torch.Tensor
        A batch of masks. Shape: (batch_size, seq_len).

    Returns
    -------
    loss : torch.Tensor
        The mean loss of X and Y.
    """

    # Expand masks to match the shape of X and Y
    expanded_masks = masks.unsqueeze(1).expand_as(X)

    # Mask the invalid positions in X and Y
    masked_X = X * expanded_masks.float()
    masked_Y = Y * expanded_masks.float()

    # Compute the loss considering only the valid positions
    masked_loss = loss_fn(masked_X, masked_Y)

    # Normalize the loss by the number of valid positions
    norm_factor = masked_loss[expanded_masks].shape[0]
    loss = masked_loss[expanded_masks].sum() / norm_factor

    return loss
